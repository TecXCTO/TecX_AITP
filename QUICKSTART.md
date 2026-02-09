# Custom LLM Project - Quick Reference

## 🚀 Quick Start Commands

```bash
# 1. Setup Environment
bash scripts/setup_environment.sh

# 2. Test Installation
python tests/test_environment.py

# 3. Prepare Data
python src/data/preprocessor.py \
  --input data/raw/dataset.jsonl \
  --output data/processed/

# 4. Train Model
python scripts/run_training.py \
  --config config/training_config.yaml \
  --use-unsloth \
  --wandb

# 5. Deploy API
python src/deployment/api_server.py \
  --model-path models/checkpoints/final/merged \
  --host 0.0.0.0 \
  --port 8000

# 6. Export to GGUF
python src/deployment/gguf_export.py \
  --model-path models/checkpoints/final/merged \
  --output-dir models/gguf \
  --quantization q4_k_m
```

## 📁 Project Structure

```
custom_llm_project/
├── config/                      # Configuration files
│   ├── model_config.yaml       # Model architecture settings
│   ├── training_config.yaml    # Training hyperparameters
│   └── rag_config.yaml         # RAG configuration
│
├── src/                        # Source code
│   ├── model/                  # Model architecture
│   │   └── llm_architecture.py # Custom LLM with MoE + GQA
│   ├── training/              # Training pipeline
│   │   └── trainer.py         # QLoRA/Unsloth trainer
│   ├── data/                  # Data processing
│   │   └── preprocessor.py    # JSONL preprocessor
│   ├── rag/                   # RAG system
│   │   └── rag_pipeline.py    # FAISS/Pinecone RAG
│   └── deployment/            # Deployment
│       ├── api_server.py      # FastAPI server
│       └── gguf_export.py     # GGUF converter
│
├── scripts/                    # Utility scripts
│   ├── setup_environment.sh   # Environment setup
│   └── run_training.py        # Training orchestrator
│
├── data/                       # Data directories
│   ├── raw/                   # Raw JSONL files
│   ├── processed/             # Processed datasets
│   └── knowledge_base/        # RAG documents
│
├── models/                     # Model storage
│   ├── checkpoints/           # Training checkpoints
│   ├── fine_tuned/            # Final models
│   └── gguf/                  # GGUF exports
│
├── examples/                   # Usage examples
│   └── api_client_examples.py
│
├── tests/                      # Test scripts
│   └── test_environment.py
│
├── README.md                   # Project overview
├── USAGE_GUIDE.md             # Comprehensive guide
├── ARCHITECTURE.md            # Technical details
└── requirements.txt           # Dependencies
```

## ⚙️ Key Components

### 1. Model Architecture
- **Base**: Llama-3 derivative (8B params)
- **MoE**: 8 experts, top-2 routing
- **GQA**: 32 Q heads, 8 KV heads
- **RoPE**: Extended for 128k context
- **Location**: `src/model/llm_architecture.py`

### 2. Training Pipeline
- **QLoRA**: 4-bit quantization
- **Unsloth**: 2x speedup
- **LoRA**: r=64, alpha=16
- **Location**: `src/training/trainer.py`

### 3. Data Processing
- **JSONL**: Input format
- **Cleaning**: HTML, URLs, whitespace
- **Tokenization**: 128k context support
- **Location**: `src/data/preprocessor.py`

### 4. RAG System
- **Vector Stores**: FAISS, Pinecone
- **Embeddings**: Sentence Transformers
- **Chunking**: 512 tokens, 50 overlap
- **Location**: `src/rag/rag_pipeline.py`

### 5. Deployment
- **FastAPI**: OpenAI-compatible API
- **GGUF**: LM Studio/Ollama export
- **Streaming**: Server-sent events
- **Location**: `src/deployment/`

## 🎯 Common Tasks

### Training on Custom Data

```bash
# 1. Format your data as JSONL
echo '{"text": "Your training data here..."}' > data/raw/train.jsonl

# 2. Preprocess
python src/data/preprocessor.py --input data/raw/train.jsonl

# 3. Train
python scripts/run_training.py --use-unsloth --wandb
```

### Using RAG

```python
from src.rag.rag_pipeline import RAGPipeline, RAGConfig

config = RAGConfig(top_k=5)
rag = RAGPipeline("models/fine_tuned/final", config)

# Index documents
rag.index_documents(["Doc 1...", "Doc 2..."])

# Query
result = rag.query("Your question?")
print(result['answer'])
```

### API Deployment

```bash
# Start server
python src/deployment/api_server.py \
  --model-path models/fine_tuned/final

# Test with curl
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7
  }'
```

### GGUF Export

```bash
# Export for LM Studio/Ollama
python src/deployment/gguf_export.py \
  --model-path models/fine_tuned/final \
  --quantization q4_k_m \
  --create-modelfile

# Use with Ollama
ollama create my-model -f models/gguf/Modelfile
ollama run my-model
```

## 📊 Configuration Tips

### For 24GB VRAM
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
max_seq_length: 8192
```

### For 16GB VRAM
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
max_seq_length: 4096
```

### For 12GB VRAM
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 64
max_seq_length: 2048
```

## 🐛 Troubleshooting

### Out of Memory
- Reduce `max_seq_length`
- Increase `gradient_accumulation_steps`
- Enable CPU offloading in config

### Slow Training
- Use `--use-unsloth` flag
- Check GPU utilization: `nvidia-smi -l 1`
- Ensure Flash Attention is installed

### Import Errors
```bash
pip install -r requirements.txt --force-reinstall
pip install flash-attn --no-build-isolation
```

### API Connection Failed
```bash
# Check if server is running
curl http://localhost:8000/health

# Start server if needed
python src/deployment/api_server.py --model-path <path>
```

## 📚 Documentation

- **README.md**: Project overview
- **USAGE_GUIDE.md**: Detailed usage instructions
- **ARCHITECTURE.md**: Technical architecture details
- **config/**: YAML configuration files

## 🔗 Useful Links

### Testing
```bash
# Test environment
python tests/test_environment.py

# Test API
python examples/api_client_examples.py
```

### Monitoring
```bash
# TensorBoard
tensorboard --logdir logs/

# W&B
# Visit: https://wandb.ai
```

### Model Hub
```bash
# Save to Hugging Face Hub
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="models/fine_tuned/final",
    repo_id="username/model-name"
)
```

## ⚡ Performance Tips

1. **Use Unsloth**: 2x training speedup
2. **Enable Flash Attention**: Faster attention computation
3. **Gradient Checkpointing**: Lower memory usage
4. **Mixed Precision**: Use bfloat16
5. **Batch Inference**: Process multiple requests together

## 🎓 Learning Resources

- Llama 3 Model Card
- QLoRA Paper: arXiv:2305.14314
- Flash Attention: arXiv:2307.08691
- Mixture of Experts: arXiv:2101.03961

## 📞 Support

For issues or questions:
1. Check USAGE_GUIDE.md
2. Review ARCHITECTURE.md
3. Run test_environment.py
4. Check configuration files

---

**Happy Building! 🚀**
