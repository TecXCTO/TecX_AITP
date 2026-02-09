# Custom LLM Development Environment - Usage Guide

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Data Preparation](#data-preparation)
3. [Training](#training)
4. [RAG Integration](#rag-integration)
5. [Deployment](#deployment)
6. [GGUF Export](#gguf-export)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Make setup script executable
chmod +x scripts/setup_environment.sh

# Run setup
bash scripts/setup_environment.sh

# Activate virtual environment (if using)
source venv/bin/activate
```

### 2. Configuration

Edit configuration files in `config/`:
- `model_config.yaml` - Model architecture settings
- `training_config.yaml` - Training hyperparameters
- `rag_config.yaml` - RAG settings

Fill in your API keys in `.env`:
```env
WANDB_API_KEY=your_key_here
HUGGING_FACE_TOKEN=your_token_here
PINECONE_API_KEY=your_pinecone_key
```

---

## 📊 Data Preparation

### Format Your Data

Your data should be in JSONL format (one JSON object per line):

```jsonl
{"text": "This is a training example..."}
{"text": "Another training example..."}
```

Or with instruction-response pairs:

```jsonl
{"instruction": "What is AI?", "response": "AI is artificial intelligence..."}
{"instruction": "Explain machine learning", "response": "Machine learning is..."}
```

### Preprocessing Pipeline

```bash
# Basic preprocessing
python src/data/preprocessor.py \
  --input data/raw/dataset.jsonl \
  --output data/processed/

# With custom settings
python src/data/preprocessor.py \
  --input data/raw/dataset.jsonl \
  --output data/processed/ \
  --max-length 131072 \
  --min-length 10 \
  --num-workers 8
```

### Data Statistics

The preprocessor will output:
- Total examples
- Token distribution
- Average/min/max token counts
- Cleaned dataset saved to `data/processed/train.jsonl`

---

## 🎓 Training

### Standard QLoRA Training

```bash
python scripts/run_training.py \
  --config config/training_config.yaml \
  --model-config config/model_config.yaml \
  --wandb
```

### Unsloth (2x Faster Training)

```bash
python scripts/run_training.py \
  --config config/training_config.yaml \
  --use-unsloth \
  --wandb
```

### Custom Configuration

```bash
python scripts/run_training.py \
  --config config/training_config.yaml \
  --output-dir ./my_models \
  --dataset ./my_data/train.jsonl
```

### Training Parameters

Key parameters in `config/training_config.yaml`:

```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16  # Effective batch size = 16
  learning_rate: 2.0e-4
  max_seq_length: 8192
```

### Memory Optimization

For 24GB VRAM:
- Batch size: 1
- Gradient accumulation: 16
- Max sequence length: 8192
- 4-bit quantization enabled
- Gradient checkpointing enabled

For 16GB VRAM:
- Reduce max_seq_length to 4096
- Increase gradient_accumulation_steps to 32

### Monitoring Training

**Tensorboard:**
```bash
tensorboard --logdir logs/
```

**Weights & Biases:**
- Enable with `--wandb` flag
- View at https://wandb.ai

---

## 🔍 RAG Integration

### 1. Prepare Knowledge Base

Place your documents in `data/knowledge_base/`:
```
data/knowledge_base/
├── document1.txt
├── document2.pdf
└── document3.md
```

### 2. Initialize RAG System

```python
from src.rag.rag_pipeline import RAGPipeline, RAGConfig

# Configure RAG
config = RAGConfig(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    vector_store_type="faiss",
    top_k=5,
    score_threshold=0.7
)

# Initialize pipeline
rag = RAGPipeline(
    llm_model_name="models/fine_tuned/final",
    config=config
)

# Index documents
documents = [
    "Your document text here...",
    "Another document...",
]
rag.index_documents(documents)

# Save index
rag.vector_store.save("vector_stores/my_index")
```

### 3. Query with RAG

```python
# Load existing index
rag.vector_store.load("vector_stores/my_index")

# Query
result = rag.query("What is machine learning?", return_sources=True)
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

### Using Pinecone

```python
from src.rag.rag_pipeline import PineconeVectorStore

vector_store = PineconeVectorStore(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-west1-gcp",
    index_name="llm-knowledge-base",
    dimension=768
)
```

---

## 🚀 Deployment

### FastAPI Server (OpenAI-Compatible)

```bash
python src/deployment/api_server.py \
  --model-path models/fine_tuned/final \
  --host 0.0.0.0 \
  --port 8000
```

### Test the API

```bash
# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "custom-llm",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'

# Health check
curl http://localhost:8000/health
```

### Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # API key not required
)

response = client.chat.completions.create(
    model="custom-llm",
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)
```

### Docker Deployment

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD ["python", "src/deployment/api_server.py", "--model-path", "models/fine_tuned/final"]
```

Build and run:
```bash
docker build -t custom-llm-api .
docker run -p 8000:8000 --gpus all custom-llm-api
```

---

## 📦 GGUF Export

Export your model for local execution with LM Studio or Ollama.

### Single Quantization

```bash
python src/deployment/gguf_export.py \
  --model-path models/fine_tuned/final \
  --output-dir models/gguf \
  --quantization q4_k_m \
  --create-modelfile
```

### All Variants

```bash
python src/deployment/gguf_export.py \
  --model-path models/fine_tuned/final \
  --output-dir models/gguf \
  --all-variants
```

This creates:
- `model-q4_k_m.gguf` - 4-bit medium (recommended, ~4GB)
- `model-q5_k_m.gguf` - 5-bit high quality (~5GB)
- `model-q8_0.gguf` - 8-bit very high quality (~8GB)

### Quantization Guide

| Format | Size | Quality | Speed | Use Case |
|--------|------|---------|-------|----------|
| q4_k_m | ~4GB | Medium | Fast | General use, recommended |
| q5_k_m | ~5GB | High | Medium | Better quality |
| q8_0 | ~8GB | Very High | Slower | Maximum quality |
| f16 | ~16GB | Original | Slowest | Reference |

### Using with LM Studio

1. Open LM Studio
2. Go to "Import Model"
3. Select your GGUF file
4. Model ready to use!

### Using with Ollama

```bash
# Create model from Modelfile
ollama create custom-llm -f models/gguf/Modelfile

# Run model
ollama run custom-llm

# Or use in API
ollama serve
curl http://localhost:11434/api/generate -d '{
  "model": "custom-llm",
  "prompt": "Why is the sky blue?"
}'
```

---

## 🔧 Troubleshooting

### Out of Memory

**Problem:** CUDA out of memory during training

**Solutions:**
1. Reduce `max_seq_length` in config
2. Increase `gradient_accumulation_steps`
3. Reduce `per_device_train_batch_size` to 1
4. Enable more aggressive gradient checkpointing

### Slow Training

**Problem:** Training is very slow

**Solutions:**
1. Use Unsloth: `--use-unsloth`
2. Enable Flash Attention 2 (auto-enabled if available)
3. Check GPU utilization: `nvidia-smi -l 1`
4. Reduce sequence length for faster iterations

### Model Quality Issues

**Problem:** Model not performing well

**Solutions:**
1. Increase training epochs
2. Check data quality and diversity
3. Adjust learning rate
4. Use higher LoRA rank (r=64 or r=128)
5. Ensure sufficient training data (10k+ examples)

### Import Errors

**Problem:** Module not found errors

**Solutions:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Install Flash Attention separately
pip install flash-attn --no-build-isolation

# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

### RAG Not Finding Relevant Documents

**Problem:** RAG returning irrelevant results

**Solutions:**
1. Adjust `score_threshold` (lower = more results)
2. Increase `top_k` value
3. Use better embedding model (e.g., `BAAI/bge-large-en-v1.5`)
4. Reduce `chunk_size` for more granular search
5. Add reranking with cross-encoder

---

## 📚 Additional Resources

### Architecture Details
- **MoE**: Every other layer uses Mixture of Experts
- **GQA**: 8 KV heads for 32 query heads (4:1 ratio)
- **RoPE**: Extended for 128k context window
- **Quantization**: QLoRA 4-bit with NF4

### Performance Benchmarks
- **Training Speed**: 2x faster with Unsloth
- **VRAM Usage**: ~18GB on 24GB GPU (batch=1, seq=8192)
- **Inference Speed**: ~30 tokens/sec on RTX 4090
- **Context Length**: Up to 128k tokens

### Best Practices
1. Start with small dataset to test pipeline
2. Monitor GPU utilization during training
3. Save checkpoints frequently
4. Use W&B for experiment tracking
5. Test different quantization levels
6. Validate model outputs regularly

---

## 🎯 Example Workflows

### Complete Training Pipeline

```bash
# 1. Setup environment
bash scripts/setup_environment.sh

# 2. Prepare data
python src/data/preprocessor.py \
  --input data/raw/dataset.jsonl \
  --output data/processed/

# 3. Train model
python scripts/run_training.py \
  --config config/training_config.yaml \
  --use-unsloth \
  --wandb

# 4. Export to GGUF
python src/deployment/gguf_export.py \
  --model-path models/checkpoints/final/merged \
  --output-dir models/gguf \
  --quantization q4_k_m

# 5. Deploy API
python src/deployment/api_server.py \
  --model-path models/checkpoints/final/merged
```

### RAG-Enabled Deployment

```python
# Initialize RAG
from src.rag.rag_pipeline import RAGPipeline, RAGConfig

config = RAGConfig(top_k=5, score_threshold=0.7)
rag = RAGPipeline("models/fine_tuned/final", config)

# Index knowledge base
with open("data/knowledge_base/docs.txt") as f:
    docs = f.read().split("\n\n")
rag.index_documents(docs)

# Query with context
result = rag.query("Explain the new product features")
print(result['answer'])
```

---

For more help, check the documentation or open an issue on GitHub!
