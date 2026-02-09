"""
QLoRA Training Pipeline with Unsloth/Axolotl integration
Optimized for 24GB VRAM GPUs
"""

import os
import torch
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from datasets import load_dataset
import yaml


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA training"""
    # Model
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    tokenizer_name: Optional[str] = None
    
    # QLoRA parameters
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # LoRA parameters
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training
    output_dir: str = "./models/checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # Optimization
    optim: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    max_seq_length: int = 8192
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True
    
    # Data
    dataset_path: str = "data/processed/train.jsonl"
    dataset_text_field: str = "text"
    

class QLoRATrainer:
    """Trainer for QLoRA fine-tuning"""
    
    def __init__(self, config: QLoRAConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def setup_model(self):
        """Initialize model with 4-bit quantization"""
        print("Loading model with 4-bit quantization...")
        
        # BitsAndBytes config for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
        )
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # LoRA config
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        print(f"Trainable parameters: {self.model.print_trainable_parameters()}")
        
    def setup_tokenizer(self):
        """Initialize tokenizer"""
        tokenizer_name = self.config.tokenizer_name or self.config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            padding_side="right",
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
    def load_dataset(self):
        """Load and preprocess dataset"""
        print(f"Loading dataset from {self.config.dataset_path}...")
        
        # Load JSONL dataset
        dataset = load_dataset("json", data_files=self.config.dataset_path, split="train")
        
        # Split into train/validation
        dataset = dataset.train_test_split(test_size=0.05, seed=42)
        
        # Tokenization function
        def tokenize_function(examples):
            outputs = self.tokenizer(
                examples[self.config.dataset_text_field],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            outputs["labels"] = outputs["input_ids"].clone()
            return outputs
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing dataset",
        )
        
        return tokenized_dataset
    
    def setup_training_args(self):
        """Configure training arguments"""
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=0.01,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            lr_scheduler_type=self.config.lr_scheduler_type,
            
            # Optimization
            optim=self.config.optim,
            gradient_checkpointing=self.config.gradient_checkpointing,
            
            # Mixed precision
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            
            # Logging
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            
            # Save best model
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            save_total_limit=3,
            
            # Reporting
            report_to=["tensorboard", "wandb"],
            logging_dir=f"{self.config.output_dir}/logs",
            
            # Other
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
        )
        
        return training_args
    
    def train(self):
        """Run training"""
        # Setup
        self.setup_tokenizer()
        self.setup_model()
        
        # Load data
        dataset = self.load_dataset()
        
        # Training arguments
        training_args = self.setup_training_args()
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=self.tokenizer,
        )
        
        # Train
        print("Starting training...")
        self.trainer.train()
        
        # Save final model
        print(f"Saving model to {self.config.output_dir}/final")
        self.trainer.save_model(f"{self.config.output_dir}/final")
        self.tokenizer.save_pretrained(f"{self.config.output_dir}/final")
        
    def save_merged_model(self, output_path: str):
        """Merge LoRA weights and save full model"""
        print(f"Merging LoRA weights and saving to {output_path}...")
        
        # Merge LoRA weights back into base model
        merged_model = self.model.merge_and_unload()
        
        # Save merged model
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        print("Model saved successfully!")


class UnslothTrainer(QLoRATrainer):
    """Enhanced trainer using Unsloth for 2x speedup"""
    
    def __init__(self, config: QLoRAConfig):
        super().__init__(config)
        try:
            from unsloth import FastLanguageModel
            self.use_unsloth = True
            self.FastLanguageModel = FastLanguageModel
        except ImportError:
            print("Warning: Unsloth not installed, falling back to standard QLoRA")
            self.use_unsloth = False
    
    def setup_model(self):
        """Initialize model with Unsloth acceleration"""
        if not self.use_unsloth:
            return super().setup_model()
        
        print("Loading model with Unsloth acceleration...")
        
        # Load model with Unsloth
        self.model, self.tokenizer = self.FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=self.config.load_in_4bit,
        )
        
        # Apply LoRA with Unsloth
        self.model = self.FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            use_gradient_checkpointing=self.config.gradient_checkpointing,
        )
        
        print("Model loaded with Unsloth acceleration")


def load_config_from_yaml(config_path: str) -> QLoRAConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract relevant fields
    training_config = config_dict.get('training', {})
    lora_config = config_dict.get('lora', {})
    
    return QLoRAConfig(
        model_name=config_dict.get('model', {}).get('base_model', 'meta-llama/Meta-Llama-3-8B'),
        output_dir=training_config.get('output_dir', './models/checkpoints'),
        num_train_epochs=training_config.get('num_train_epochs', 3),
        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 1),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 16),
        learning_rate=training_config.get('learning_rate', 2e-4),
        lora_r=lora_config.get('r', 64),
        lora_alpha=lora_config.get('lora_alpha', 16),
        lora_dropout=lora_config.get('lora_dropout', 0.05),
        lora_target_modules=lora_config.get('target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
    )


if __name__ == "__main__":
    # Example usage
    config = QLoRAConfig()
    
    # Use Unsloth trainer for 2x speedup
    trainer = UnslothTrainer(config)
    trainer.train()
    
    # Save merged model
    trainer.save_merged_model("./models/final_merged")
