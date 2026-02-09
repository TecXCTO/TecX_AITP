"""
Main Training Script
Orchestrates the complete training pipeline
"""

import argparse
import yaml
import os
from pathlib import Path
import torch
import wandb
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from training.trainer import QLoRATrainer, UnslothTrainer, load_config_from_yaml


def setup_wandb(project_name: str, config: dict):
    """Initialize Weights & Biases logging"""
    wandb.init(
        project=project_name,
        config=config,
        name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


def main():
    parser = argparse.ArgumentParser(description="Train Custom LLM")
    parser.add_argument("--config", type=str, default="config/training_config.yaml", help="Path to config file")
    parser.add_argument("--model-config", type=str, default="config/model_config.yaml", help="Path to model config")
    parser.add_argument("--use-unsloth", action="store_true", help="Use Unsloth for faster training")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("--dataset", type=str, help="Override dataset path")
    
    args = parser.parse_args()
    
    # Load configurations
    print("Loading configuration...")
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    with open(args.model_config, 'r') as f:
        model_config_dict = yaml.safe_load(f)
    
    # Load QLoRA config
    qlora_config = load_config_from_yaml(args.config)
    
    # Override with command line arguments
    if args.output_dir:
        qlora_config.output_dir = args.output_dir
    if args.dataset:
        qlora_config.dataset_path = args.dataset
    
    # Create output directory
    Path(qlora_config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup W&B if enabled
    if args.wandb:
        wandb_config = config_dict.get('wandb', {})
        setup_wandb(
            wandb_config.get('project', 'custom-llm-training'),
            {**config_dict, **model_config_dict}
        )
    
    # Initialize trainer
    print("\n=== Initializing Trainer ===")
    if args.use_unsloth:
        print("Using Unsloth for 2x faster training")
        trainer = UnslothTrainer(qlora_config)
    else:
        print("Using standard QLoRA training")
        trainer = QLoRATrainer(qlora_config)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: No GPU detected. Training will be very slow!")
    
    # Start training
    print("\n=== Starting Training ===")
    try:
        trainer.train()
        
        # Save merged model
        print("\n=== Saving Merged Model ===")
        merged_path = f"{qlora_config.output_dir}/merged"
        trainer.save_merged_model(merged_path)
        
        print("\n=== Training Complete! ===")
        print(f"Checkpoints saved to: {qlora_config.output_dir}")
        print(f"Merged model saved to: {merged_path}")
        
        if args.wandb:
            wandb.finish()
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if args.wandb:
            wandb.finish()
    except Exception as e:
        print(f"\nError during training: {e}")
        if args.wandb:
            wandb.finish()
        raise


if __name__ == "__main__":
    main()
