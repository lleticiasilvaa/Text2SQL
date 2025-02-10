#!/usr/bin/env python3
import argparse
import torch
import os
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed
)
from trl import SFTTrainer
from trl.trainer.sft_config import SFTConfig
import bitsandbytes as bnb
from huggingface_hub import login, HfFileSystem
import pandas as pd
import sqlparse
import re
from typing import List, Dict, Any

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune a language model for SQL generation or schema linking')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, required=True,
                       help='Name of the base model to fine-tune')
    parser.add_argument('--output_name', type=str, required=True,
                       help='Name for the fine-tuned model output')
    parser.add_argument('--prev_checkpoint', type=str, default=None,
                       help='Previous checkpoint to continue training from')
    
    # Authentication
    parser.add_argument('--read_token', type=str, required=True,
                       help='HuggingFace read token')
    parser.add_argument('--write_token', type=str, required=True,
                       help='HuggingFace write token')
    
    # Dataset configuration
    parser.add_argument('--dataset_id', type=str, required=True,
                       help='HuggingFace dataset ID')
    parser.add_argument('--task_type', choices=['sql', 'schema'], required=True,
                       help='Task type: sql generation or schema linking')
    parser.add_argument('--question_field', type=str, default='question_en',
                       help='Field name in dataset containing the question')
    parser.add_argument('--schema_field', type=str, default='schema_SQLDatabase',
                       help='Field name in dataset containing the database schema')
    parser.add_argument('--output_field', type=str, default='query',
                       help='Field name in dataset containing the SQL query (for SQL task) or schema linking JSON (for schema task)')
    
    # Training configuration
    parser.add_argument('--seed', type=int, default=14,
                       help='Random seed for reproducibility')
    parser.add_argument('--train_batch_size', type=int, default=1,
                       help='Training batch size per device')
    parser.add_argument('--eval_batch_size', type=int, default=1,
                       help='Evaluation batch size per device')
    parser.add_argument('--grad_accum_steps', type=int, default=8,
                       help='Number of gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=1,
                       help='Number of training epochs')
    parser.add_argument('--save_steps', type=int, default=250,
                       help='Save checkpoint every N steps')
    
    return parser.parse_args()

def setup_model_and_tokenizer(args: argparse.Namespace):
    """Initialize model and tokenizer with appropriate configuration."""
    # Determine compute type based on hardware capabilities
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        token=args.read_token,
        map_device="auto",
        add_eos_token=True,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=compute_dtype,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
        token=args.read_token,
    )
    
    return model, tokenizer

def prepare_data(dataset: Dataset, args: argparse.Namespace, tokenizer: AutoTokenizer) -> tuple:
    """Prepare and split the dataset for training."""
    df = dataset.to_pandas()
    
    if args.task_type == 'sql':
        df = df.apply(lambda x: apply_sql_template(x, tokenizer, args), axis=1)
    else:
        df = df.apply(lambda x: apply_schema_template(x, tokenizer, args), axis=1)
    
    # Create dataset splits
    _df = pd.DataFrame({'text': df.sample(frac=1, random_state=args.seed)['text']})
    _df = Dataset.from_pandas(_df)
    train_dataset, valid_dataset = _df.train_test_split(
        test_size=0.01, shuffle=True, seed=args.seed
    ).values()
    
    return train_dataset, valid_dataset

def replace_alias(query: str) -> str:
    """Replace SQL aliases with their original table names."""
    alias_pattern = re.compile(r'(\bFROM\b|\bJOIN\b)\s+(\w+)\s+AS\s+(\w+)', re.IGNORECASE)
    aliases = {match.group(3): match.group(2) for match in alias_pattern.finditer(query)}
    
    for alias, table in aliases.items():
        query = re.sub(r'\b' + alias + r'\b', table, query)
    
    query = re.sub(r'\bAS\s+\w+', '', query, flags=re.IGNORECASE)
    return query

def to_sql(query: str) -> str:
    """Format SQL query with consistent styling."""
    return sqlparse.format(replace_alias(query), reindent=True, keyword_case='UPPER')

def apply_sql_template(row: Dict[str, Any], tokenizer: AutoTokenizer, args: argparse.Namespace) -> Dict[str, Any]:
    """Apply template for SQL generation task."""
    question = row[args.question_field]
    schema = row[args.schema_field]
    sql = to_sql(row[args.output_field])
    
    system_message = "Given a user question and the schema of a database, your task is to generate an SQL query that accurately answers the question based on the provided schema."
    
    chat = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"\n# Schema:\n```sql\n{schema}\n```\n\n# Question: {question}\n"},
        {'role': 'assistant', 'content': f"\n```sql\n{sql}\n```\n"}
    ]
    
    row['text'] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    return row

def apply_schema_template(row: Dict[str, Any], tokenizer: AutoTokenizer, args: argparse.Namespace) -> Dict[str, Any]:
    """Apply template for schema linking task."""
    question = row[args.question_field]
    schema = row[args.schema_field]
    schema_linking = row[args.output_field]
    
    system_message = "Given a user question and the schema of a database, your task is to generate an JSON with the the names of tables and columns of the schema that the question is referring to."
    
    chat = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"# Schema:\n```sql\n{schema}\n```\n\n# Question: {question}"},
        {'role': 'assistant', 'content': f"```json\n{schema_linking}\n```"}
    ]
    
    row['text'] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    return row

def setup_trainer(model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                 train_dataset: Dataset, valid_dataset: Dataset,
                 args: argparse.Namespace) -> SFTTrainer:
    """Configure and return the SFT trainer."""
    # Find all linear modules for LoRA
    modules = find_all_linear_names(model)
    
    # Configure LoRA
    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.1,
        r=64,
        target_modules=modules,
    )
    
    # Configure training arguments
    training_args = SFTConfig(
        output_dir="out",
        dataset_text_field="text",
        max_seq_length=2048,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_strategy="steps",
        logging_steps=args.save_steps,
        logging_dir="out/logs",
        eval_strategy="steps",
        eval_steps=args.save_steps,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        gradient_checkpointing=True,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        learning_rate=args.learning_rate,
        weight_decay=0.001,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        seed=args.seed,
        report_to=["tensorboard"],
        push_to_hub=True,
        hub_strategy="all_checkpoints",
        hub_model_id=args.output_name,
        label_names=["labels"]
    )
    
    return SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=peft_config,
        args=training_args
    )

def find_all_linear_names(model: AutoModelForCausalLM) -> List[str]:
    """Find all linear module names in the model."""
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return list(lora_module_names)

def main():
    args = parse_args()
    set_seed(args.seed)
    login(token=args.write_token)
    
    # Load dataset
    dataset = load_dataset(args.dataset_id, split="train")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args)
    
    # Prepare datasets
    train_dataset, valid_dataset = prepare_data(dataset, args, tokenizer)
    
    # Setup and start training
    trainer = setup_trainer(model, tokenizer, train_dataset, valid_dataset, args)
    
    if args.prev_checkpoint:
        trainer.train(f"out/{args.prev_checkpoint}")
    else:
        trainer.train()
    
    trainer.push_to_hub()

if __name__ == "__main__":
    main()