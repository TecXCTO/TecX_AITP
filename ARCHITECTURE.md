# Custom LLM Architecture - Technical Overview

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Raw Data (JSONL)                                                │
│       │                                                          │
│       ├──► Data Preprocessor                                     │
│       │    • HTML Cleaning                                       │
│       │    • URL Filtering                                       │
│       │    • Whitespace Normalization                            │
│       │    • Tokenization (128k context)                         │
│       │    • Deduplication                                       │
│       │                                                          │
│       └──► Processed Dataset                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      MODEL ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Embeddings (128256 vocab)                                │
│       │                                                          │
│       ├──► 32 Transformer Blocks                                 │
│       │    ┌───────────────────────────────────┐                │
│       │    │  Block (Alternating MoE/FFN)      │                │
│       │    │  ┌─────────────────────────────┐  │                │
│       │    │  │ RMSNorm                     │  │                │
│       │    │  │ Grouped-Query Attention     │  │                │
│       │    │  │ • 32 Q heads, 8 KV heads    │  │                │
│       │    │  │ • RoPE (500k base)          │  │                │
│       │    │  │ • Flash Attention 2         │  │                │
│       │    │  └─────────────────────────────┘  │                │
│       │    │  ┌─────────────────────────────┐  │                │
│       │    │  │ RMSNorm                     │  │                │
│       │    │  │ MoE / FFN Layer             │  │                │
│       │    │  │ • 8 experts (MoE)           │  │                │
│       │    │  │ • Top-2 routing             │  │                │
│       │    │  │ • SiLU activation           │  │                │
│       │    │  └─────────────────────────────┘  │                │
│       │    └───────────────────────────────────┘                │
│       │                                                          │
│       ├──► RMSNorm                                               │
│       │                                                          │
│       └──► LM Head (128256 vocab)                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Base Model (Llama-3-8B)                                         │
│       │                                                          │
│       ├──► 4-bit Quantization (QLoRA)                            │
│       │    • NF4 quantization                                    │
│       │    • Double quantization                                 │
│       │    • bfloat16 compute                                    │
│       │                                                          │
│       ├──► LoRA Adaptation                                       │
│       │    • r=64, alpha=16                                      │
│       │    • Target: Q,K,V,O,Gate,Up,Down                        │
│       │    • Dropout: 0.05                                       │
│       │                                                          │
│       ├──► Optimization                                          │
│       │    • Unsloth (2x speedup)                                │
│       │    • Gradient Checkpointing                              │
│       │    • Paged AdamW 8-bit                                   │
│       │    • Cosine LR Schedule                                  │
│       │                                                          │
│       └──► Fine-tuned Model                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    RAG SYSTEM (Optional)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Knowledge Base Documents                                        │
│       │                                                          │
│       ├──► Document Chunking                                     │
│       │    • 512 token chunks                                    │
│       │    • 50 token overlap                                    │
│       │                                                          │
│       ├──► Embedding Generation                                  │
│       │    • Sentence Transformers                               │
│       │    • 768-dim vectors                                     │
│       │                                                          │
│       ├──► Vector Storage                                        │
│       │    ┌────────────┬──────────────┐                        │
│       │    │   FAISS    │   Pinecone   │                        │
│       │    │  (Local)   │   (Cloud)    │                        │
│       │    └────────────┴──────────────┘                        │
│       │                                                          │
│       └──► Retrieval Pipeline                                    │
│            • Query encoding                                      │
│            • Top-k similarity search                             │
│            • Context injection                                   │
│            • LLM generation                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     DEPLOYMENT OPTIONS                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Option 1: FastAPI Server                                        │
│  ┌──────────────────────────────────────────────┐               │
│  │  OpenAI-Compatible REST API                  │               │
│  │  • /v1/chat/completions                      │               │
│  │  • /v1/completions                           │               │
│  │  • Streaming support                         │               │
│  │  • CORS enabled                              │               │
│  └──────────────────────────────────────────────┘               │
│                                                                  │
│  Option 2: GGUF Export                                           │
│  ┌──────────────────────────────────────────────┐               │
│  │  Local Execution                             │               │
│  │  • LM Studio                                 │               │
│  │  • Ollama                                    │               │
│  │  • llama.cpp                                 │               │
│  │  • Multiple quantization levels              │               │
│  └──────────────────────────────────────────────┘               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Key Specifications

### Model Architecture
- **Base**: Llama-3 derivative (8B parameters)
- **Hidden Size**: 4096
- **Layers**: 32
- **Attention Heads**: 32 (Query), 8 (Key/Value - GQA)
- **Head Dimension**: 128
- **Intermediate Size**: 14336
- **Vocabulary**: 128,256 tokens
- **Max Context**: 131,072 tokens (128k)

### Mixture of Experts (MoE)
- **Experts**: 8 per layer
- **Active**: 2 experts per token
- **Routing**: Learned gating network
- **Layers**: Every other layer uses MoE
- **Efficiency**: Reduced active parameters during inference

### Attention Mechanism
- **Type**: Grouped-Query Attention (GQA)
- **Ratio**: 4:1 (Query:Key/Value heads)
- **Benefits**: 
  - Faster inference (fewer KV cache entries)
  - Better quality than MQA
  - Lower memory usage
  
### Positional Encoding
- **Type**: Rotary Positional Embeddings (RoPE)
- **Base**: 500,000 (extended for long context)
- **Scaling**: 8x for 128k context
- **Original Max**: 8,192 positions

### Training Configuration
- **Quantization**: QLoRA 4-bit (NF4)
- **LoRA Rank**: 64
- **LoRA Alpha**: 16
- **Target Modules**: Q, K, V, O projections + MLP
- **Optimizer**: Paged AdamW 8-bit
- **Learning Rate**: 2e-4
- **Scheduler**: Cosine with warmup
- **Batch Size**: 1 (effective: 16 with grad accumulation)
- **Precision**: bfloat16

### Memory Requirements
- **Training**: ~18GB VRAM (with QLoRA on 24GB GPU)
- **Inference**: 
  - FP16: ~16GB
  - 4-bit: ~4GB
  - 8-bit: ~8GB

### Performance
- **Training Speed**: 2x faster with Unsloth
- **Inference Speed**: ~30 tokens/sec (RTX 4090)
- **Context Processing**: Flash Attention 2 optimized

## 🔧 Technology Stack

### Core ML Framework
- PyTorch 2.1+
- Transformers 4.36+
- PEFT 0.7+
- BitsAndBytes 0.41+

### Training Acceleration
- Unsloth / Axolotl
- Flash Attention 2
- Gradient Checkpointing
- Mixed Precision Training

### Data Processing
- Datasets (Hugging Face)
- Tokenizers
- Pandas, NumPy
- JSONL support

### RAG Stack
- FAISS / Pinecone
- Sentence Transformers
- LangChain (optional)

### Deployment
- FastAPI
- Uvicorn
- GGUF conversion tools
- Docker support

### Monitoring
- Weights & Biases
- TensorBoard
- Custom logging

## 💡 Design Decisions

### Why Mixture of Experts?
- **Efficiency**: Only activate 2/8 experts per token
- **Capacity**: More total parameters without increasing computation
- **Specialization**: Different experts learn different patterns

### Why Grouped-Query Attention?
- **Speed**: 3-4x faster than Multi-Head Attention
- **Quality**: Better than Multi-Query Attention
- **Memory**: Reduced KV cache size for long contexts

### Why QLoRA?
- **Memory**: Train large models on consumer GPUs
- **Quality**: Minimal performance loss vs full fine-tuning
- **Flexibility**: Easy to merge or swap adapters

### Why 128k Context?
- **Long Documents**: Process entire books, papers
- **RAG**: Fit more retrieved context
- **Conversation**: Longer chat histories

## 📈 Scalability

### Horizontal Scaling
- Multi-GPU training with DeepSpeed
- Distributed inference with model parallelism
- API server clustering

### Vertical Scaling
- Larger models (70B with same architecture)
- More experts per layer
- Deeper networks

### Optimization Paths
- Speculative decoding
- Quantization-aware training
- Knowledge distillation
- Pruning and sparsity

## 🔒 Production Considerations

### Security
- API authentication
- Rate limiting
- Input validation
- Safe content filtering

### Reliability
- Health checks
- Graceful degradation
- Error handling
- Logging and monitoring

### Performance
- Model caching
- Batch processing
- Connection pooling
- Response streaming

### Compliance
- Data privacy
- Model governance
- Audit logging
- Version control

## 🎯 Use Cases

### Supported Tasks
- Text generation
- Question answering
- Code generation
- Summarization
- Translation
- Chat assistance
- RAG-based Q&A
- Custom domain adaptation

### Industry Applications
- Customer support
- Content creation
- Research assistance
- Code development
- Documentation generation
- Knowledge management

## 📚 References

### Papers
- Llama 3: [Meta AI Blog]
- Mixtral MoE: [Mistral AI Paper]
- GQA: "GQA: Training Generalized Multi-Query Transformer Models"
- RoPE: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- QLoRA: "QLoRA: Efficient Finetuning of Quantized LLMs"
- Flash Attention: "FlashAttention-2: Faster Attention with Better Parallelism"

### Repositories
- Transformers: huggingface/transformers
- PEFT: huggingface/peft
- Unsloth: unslothai/unsloth
- Axolotl: OpenAccess-AI-Collective/axolotl
- llama.cpp: ggerganov/llama.cpp

---

**Version**: 1.0.0  
**Last Updated**: February 2026  
**License**: MIT
