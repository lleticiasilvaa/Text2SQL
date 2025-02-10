from typing import Dict, Any, List, Optional, TypedDict
from dataclasses import dataclass
from transformers import pipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

SCHEMA_LINKING = "schema_linking"
TEXT_TO_SQL = "text_to_sql"

class ModelConfig(TypedDict):
    model: str
    lora_schema_linking: str
    lora_text_to_sql: str
    db_path: str
    mode: str
    quantize: bool
    
class ChatMessage(TypedDict):
    role: str
    content: str

    
    
      
def get_model_and_tokenizer(model_id, quantize=False):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    if quantize:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            use_cache=True,
            quantization_config=bnb_config,
            # attn_implementation = "flash_attention_2"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            use_cache=True,
            # attn_implementation = "flash_attention_2"
        )
    return model, tokenizer

class LLMManager:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.active_adapter = SCHEMA_LINKING
        params = {
            "task":"text-generation",
            "max_new_tokens":500,
            "do_sample":False,
            "temperature": None,
            "return_full_text":False,
            "stop_sequence": "<|im_end|>"
        }
        
        if config['mode'] == 'adapters':
            self.model, self.tokenizer = get_model_and_tokenizer(config['model'], quantize=config['quantize'])
            self.model.load_adapter(config['lora_schema_linking'], SCHEMA_LINKING)
            self.model.load_adapter(config['lora_text_to_sql'], TEXT_TO_SQL)
            
            params["eos_token_id"] = self.tokenizer.eos_token_id
            params["pad_token_id"] = self.tokenizer.eos_token_id
            
            self.llm = pipeline(model=self.model, tokenizer=self.tokenizer, **params)
        
        else:
          
            self.model_1, self.tokenizer1 = get_model_and_tokenizer(config['lora_schema_linking'], quantize=False)
            self.model_2, self.tokenizer2 = get_model_and_tokenizer(config['lora_text_to_sql'], quantize=True)
            
            params["eos_token_id"] = self.tokenizer1.eos_token_id
            params["pad_token_id"] = self.tokenizer1.eos_token_id
            
            self.schema_linking_llm = pipeline(model=self.model_1, tokenizer=self.tokenizer1, **params)
            self.text_to_sql_llm = pipeline(model=self.model_2,  tokenizer=self.tokenizer2, **params)
        
            self.llm = self.schema_linking_llm

        

    def set_adapter(self, adapter: Any) -> None:
        self.active_adapter = adapter
        if self.config['mode'] == 'adapters':
            self.llm.model.set_adapter(adapter)
        else:        
            if adapter == SCHEMA_LINKING:
                self.llm = self.schema_linking_llm
            else:
                self.llm = self.text_to_sql_llm  
    

    def generate_completion(
        self,
        messages: List[ChatMessage],
        max_tokens: int,
        temperature: float = 0
    ) -> str:
        text = self.llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        
        if self.active_adapter == SCHEMA_LINKING:
            text += '```json\n'
            
        response = self.llm(
            text
        )
        
        response = response[0]['generated_text']
        
        if self.active_adapter == SCHEMA_LINKING:
            response = "```json\n" + response
        
        return response