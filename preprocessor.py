"""
Data Preprocessing Pipeline
Handles JSONL ingestion, cleaning, and tokenization for 128k context window
"""

import json
import jsonlines
import pandas as pd
import re
from typing import List, Dict, Optional, Iterator
from pathlib import Path
from dataclasses import dataclass
from transformers import AutoTokenizer
from datasets import Dataset
import multiprocessing as mp
from tqdm import tqdm


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing"""
    input_path: str
    output_dir: str
    tokenizer_name: str = "meta-llama/Meta-Llama-3-8B"
    max_length: int = 131072  # 128k context
    min_length: int = 10
    chunk_size: int = 1000
    num_workers: int = 8
    remove_duplicates: bool = True
    clean_html: bool = True
    clean_urls: bool = True
    normalize_whitespace: bool = True


class DataCleaner:
    """Utilities for cleaning raw text data"""
    
    @staticmethod
    def clean_html(text: str) -> str:
        """Remove HTML tags and entities"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Decode common HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        return text
    
    @staticmethod
    def clean_urls(text: str) -> str:
        """Remove or replace URLs"""
        # Replace URLs with [URL] token
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
        return text
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace characters"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    @staticmethod
    def remove_special_chars(text: str) -> str:
        """Remove problematic special characters"""
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        return text
    
    @staticmethod
    def clean_text(text: str, config: PreprocessingConfig) -> str:
        """Apply all cleaning operations"""
        if config.clean_html:
            text = DataCleaner.clean_html(text)
        
        if config.clean_urls:
            text = DataCleaner.clean_urls(text)
        
        text = DataCleaner.remove_special_chars(text)
        
        if config.normalize_whitespace:
            text = DataCleaner.normalize_whitespace(text)
        
        return text


class JSONLProcessor:
    """Process JSONL files for LLM training"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.cleaner = DataCleaner()
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def read_jsonl(self, file_path: str) -> Iterator[Dict]:
        """Read JSONL file line by line"""
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                yield obj
    
    def process_single_example(self, example: Dict) -> Optional[Dict]:
        """Process a single training example"""
        # Extract text field (adjust field name as needed)
        text = example.get('text') or example.get('content') or example.get('prompt', '')
        
        if not text or len(text) < self.config.min_length:
            return None
        
        # Clean text
        text = self.cleaner.clean_text(text, self.config)
        
        # Tokenize to check length
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Skip if too long
        if len(tokens) > self.config.max_length:
            # Optionally truncate or skip
            tokens = tokens[:self.config.max_length]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        # Skip if too short after cleaning
        if len(tokens) < self.config.min_length:
            return None
        
        return {
            'text': text,
            'token_count': len(tokens),
            'metadata': {k: v for k, v in example.items() if k not in ['text', 'content', 'prompt']}
        }
    
    def process_batch(self, examples: List[Dict]) -> List[Dict]:
        """Process a batch of examples"""
        processed = []
        for example in examples:
            result = self.process_single_example(example)
            if result:
                processed.append(result)
        return processed
    
    def process_file(self, input_path: str, output_path: str):
        """Process entire JSONL file"""
        print(f"Processing {input_path}...")
        
        # Read all examples
        examples = list(self.read_jsonl(input_path))
        print(f"Loaded {len(examples)} examples")
        
        # Process in parallel
        with mp.Pool(processes=self.config.num_workers) as pool:
            # Split into chunks
            chunk_size = len(examples) // self.config.num_workers + 1
            chunks = [examples[i:i + chunk_size] for i in range(0, len(examples), chunk_size)]
            
            # Process chunks
            results = list(tqdm(
                pool.imap(self.process_batch, chunks),
                total=len(chunks),
                desc="Processing chunks"
            ))
        
        # Flatten results
        processed_examples = [item for sublist in results for item in sublist]
        print(f"Processed {len(processed_examples)} examples (kept {len(processed_examples)/len(examples)*100:.1f}%)")
        
        # Remove duplicates if enabled
        if self.config.remove_duplicates:
            processed_examples = self.remove_duplicates(processed_examples)
            print(f"After deduplication: {len(processed_examples)} examples")
        
        # Save to JSONL
        self.save_jsonl(processed_examples, output_path)
        
        # Also save as Hugging Face Dataset
        dataset = Dataset.from_list(processed_examples)
        dataset.save_to_disk(output_path.replace('.jsonl', '_dataset'))
        
        # Print statistics
        self.print_statistics(processed_examples)
    
    def remove_duplicates(self, examples: List[Dict]) -> List[Dict]:
        """Remove duplicate examples based on text"""
        seen = set()
        unique = []
        
        for example in examples:
            text_hash = hash(example['text'])
            if text_hash not in seen:
                seen.add(text_hash)
                unique.append(example)
        
        return unique
    
    def save_jsonl(self, examples: List[Dict], output_path: str):
        """Save processed examples to JSONL"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with jsonlines.open(output_path, mode='w') as writer:
            writer.write_all(examples)
        
        print(f"Saved {len(examples)} examples to {output_path}")
    
    def print_statistics(self, examples: List[Dict]):
        """Print dataset statistics"""
        token_counts = [ex['token_count'] for ex in examples]
        
        print("\n=== Dataset Statistics ===")
        print(f"Total examples: {len(examples)}")
        print(f"Average tokens: {sum(token_counts) / len(token_counts):.0f}")
        print(f"Min tokens: {min(token_counts)}")
        print(f"Max tokens: {max(token_counts)}")
        print(f"Median tokens: {sorted(token_counts)[len(token_counts)//2]}")
        
        # Token distribution
        bins = [0, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
        for i in range(len(bins) - 1):
            count = sum(1 for tc in token_counts if bins[i] <= tc < bins[i+1])
            print(f"  {bins[i]}-{bins[i+1]} tokens: {count} ({count/len(token_counts)*100:.1f}%)")
    
    def create_instruction_dataset(self, examples: List[Dict], instruction_template: str):
        """Format examples with instruction template"""
        formatted = []
        
        for example in examples:
            # Apply instruction template
            text = instruction_template.format(**example)
            formatted.append({'text': text})
        
        return formatted


def create_chat_format(instruction: str, response: str, system_prompt: Optional[str] = None) -> str:
    """Format data in chat template"""
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": instruction})
    messages.append({"role": "assistant", "content": response})
    
    # Format as Llama-3 chat template
    formatted = ""
    for msg in messages:
        formatted += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
    
    return formatted


def main():
    """Main preprocessing pipeline"""
    config = PreprocessingConfig(
        input_path="data/raw/dataset.jsonl",
        output_dir="data/processed",
        tokenizer_name="meta-llama/Meta-Llama-3-8B",
        max_length=131072,  # 128k context
        min_length=10,
        num_workers=8,
    )
    
    processor = JSONLProcessor(config)
    
    # Process the dataset
    output_path = f"{config.output_dir}/train.jsonl"
    processor.process_file(config.input_path, output_path)
    
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
