# Toward Small Language Models for Text-to-SQL and for Schema-Linking

This repository contains code for fine-tuning large language models for SQL-related tasks using the PEFT (Parameter-Efficient Fine-Tuning) method. The code supports two main tasks:
1. SQL Query Generation
2. Schema Linking

## Features

- Support for both SQL generation and schema linking tasks
- Integration with HuggingFace's Transformers and datasets
- 4-bit quantization using bitsandbytes
- LoRA (Low-Rank Adaptation) fine-tuning
- Automatic checkpoint saving and model pushing to HuggingFace Hub
- Comprehensive command-line interface for easy configuration

## Requirements

```bash
pip install accelerate peft bitsandbytes transformers trl datasets huggingface_hub pandas sqlparse torch
```

## Usage

### Basic Command

```bash
python finetuning.py \
    --model_name <base_model_name> \
    --output_name <output_model_name> \
    --read_token <huggingface_read_token> \
    --write_token <huggingface_write_token> \
    --task_type <sql|schema>
```

### Full Configuration Options

```bash
python finetuning.py \
    --model_name <base_model_name> \
    --output_name <output_model_name> \
    --read_token <huggingface_read_token> \
    --write_token <huggingface_write_token> \
    --task_type <sql|schema> \
    --dataset_id <huggingface_dataset_id> \
    --prev_checkpoint <checkpoint_name> \
    --seed <random_seed> \
    --train_batch_size <batch_size> \
    --eval_batch_size <batch_size> \
    --grad_accum_steps <steps> \
    --learning_rate <lr> \
    --num_epochs <epochs> \
    --save_steps <steps>
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_name` | Base model to fine-tune (required) | - |
| `--output_name` | Name for the fine-tuned model (required) | - |
| `--read_token` | HuggingFace read token (required) | - |
| `--write_token` | HuggingFace write token (required) | - |
| `--task_type` | Task type: 'sql' or 'schema' (required) | - |
| `--dataset_id` | HuggingFace dataset ID | 'NESPED-GEN/spider_selector_schemaReduzido' |
| `--prev_checkpoint` | Previous checkpoint to continue training | None |
| `--seed` | Random seed for reproducibility | 14 |
| `--train_batch_size` | Training batch size per device | 1 |
| `--eval_batch_size` | Evaluation batch size per device | 1 |
| `--grad_accum_steps` | Gradient accumulation steps | 8 |
| `--learning_rate` | Learning rate | 1e-4 |
| `--num_epochs` | Number of training epochs | 1 |
| `--save_steps` | Save checkpoint every N steps | 250 |

## Training Process

1. The script first loads and prepares the specified dataset
2. Initializes the model with 4-bit quantization
3. Applies LoRA fine-tuning configuration
4. Trains the model using the SFTTrainer from TRL
5. Automatically saves checkpoints and pushes to HuggingFace Hub

## Model Outputs

- For SQL generation: The model generates SQL queries based on natural language questions and database schemas
- For schema linking: The model identifies relevant tables and columns from the schema based on the question

## Checkpoints and Logging

- Checkpoints are saved every `save_steps` steps
- Training logs are saved in the `out/logs` directory
- TensorBoard logs are available for monitoring training progress
- Models are automatically pushed to HuggingFace Hub after training

## Directory Structure

```
.
├── train.py                # Main training script
├── out/                    # Output directory
│   ├── checkpoint-{step}/  # Saved model checkpoints
│   └── logs/               # Training logs
├── deploy/                 # Deployment code
├── notebooks/              # Jupyter notebooks used for pre-processing, training, and evaluation
├── results/                # Some of our results
└── README.md               # This file
```

## Advanced Features

### Checkpoint Resumption

To resume training from a previous checkpoint:

```bash
python finetuning.py \
    --model_name <base_model_name> \
    --output_name <output_model_name> \
    --prev_checkpoint checkpoint-1000 \
    ... [other parameters]
```

### Model Templates

#### SQL Generation Template
The model uses a specific template for SQL generation tasks:
```
System: Given a user question and the schema of a database...
User: [Schema details + Question]
Assistant: [Generated SQL query]
```

#### Schema Linking Template
For schema linking tasks:
```
System: Given a user question and the schema of a database...
User: [Schema details + Question]
Assistant: [JSON with relevant tables/columns]
```

## Monitoring Training

### TensorBoard

To monitor training progress using TensorBoard:

```bash
tensorboard --logdir out/logs
```

This will show:
- Training loss
- Evaluation loss
- Learning rate schedule
- Other training metrics

### HuggingFace Hub Integration

The script automatically:
- Pushes checkpoints to the Hub
- Updates model cards
- Maintains version history

## Troubleshooting

### Common Issues

1. Out of Memory (OOM) Errors
   - Reduce batch size
   - Increase gradient accumulation steps
   - Enable gradient checkpointing

2. Training Instability
   - Adjust learning rate
   - Modify warmup ratio
   - Check gradient clipping value

3. Hub Authentication Issues
   - Verify token permissions
   - Ensure repository exists