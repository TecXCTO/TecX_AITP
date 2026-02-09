#!/bin/bash
# Environment Setup Script for Custom LLM Development

set -e

echo "=== Custom LLM Environment Setup ==="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA available:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: CUDA not detected. GPU training may not be available."
fi

echo ""
echo "=== Installing Dependencies ==="

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
echo "Installing core dependencies..."
pip install -r requirements.txt

# Install Flash Attention (if CUDA available)
if command -v nvidia-smi &> /dev/null; then
    echo "Installing Flash Attention 2..."
    pip install flash-attn --no-build-isolation
fi

# Install Unsloth for faster training (optional)
echo "Installing Unsloth..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || echo "Unsloth installation failed, skipping..."

# Install Axolotl (optional alternative)
# echo "Installing Axolotl..."
# pip install axolotl

# Create necessary directories
echo ""
echo "=== Creating Project Directories ==="
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/knowledge_base
mkdir -p models/checkpoints
mkdir -p models/fine_tuned
mkdir -p models/gguf
mkdir -p logs
mkdir -p vector_stores
mkdir -p tests

echo "Directories created:"
echo "  - data/raw (place your raw JSONL files here)"
echo "  - data/processed (processed datasets)"
echo "  - data/knowledge_base (RAG documents)"
echo "  - models/checkpoints (training checkpoints)"
echo "  - models/fine_tuned (final models)"
echo "  - models/gguf (GGUF exports)"
echo "  - logs (training logs)"
echo "  - vector_stores (FAISS/Pinecone indices)"

# Download base model (optional)
echo ""
read -p "Download base Llama-3-8B model? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading Llama-3-8B..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'meta-llama/Meta-Llama-3-8B',
    local_dir='models/base/llama-3-8b',
    local_dir_use_symlinks=False
)
    "
fi

# Setup git hooks (optional)
echo ""
echo "=== Setting Up Git Hooks ==="
pip install pre-commit
pre-commit install || echo "Pre-commit hooks setup skipped"

# Create .env file
echo ""
echo "=== Creating .env file ==="
cat > .env << EOF
# API Keys (fill in your own)
WANDB_API_KEY=
HUGGING_FACE_TOKEN=
PINECONE_API_KEY=
PINECONE_ENVIRONMENT=

# Model Configuration
BASE_MODEL=meta-llama/Meta-Llama-3-8B
MAX_SEQ_LENGTH=8192

# Training Configuration
BATCH_SIZE=1
GRADIENT_ACCUMULATION=16
LEARNING_RATE=2e-4
NUM_EPOCHS=3

# Paths
DATA_PATH=data/processed/train.jsonl
OUTPUT_DIR=models/checkpoints
EOF

echo ".env file created. Please fill in your API keys."

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Fill in API keys in .env file"
echo "2. Place your training data in data/raw/dataset.jsonl"
echo "3. Run preprocessing: python src/data/preprocessor.py --input data/raw/dataset.jsonl --output data/processed/"
echo "4. Start training: python scripts/run_training.py --config config/training_config.yaml"
echo ""
echo "For faster training with Unsloth:"
echo "  python scripts/run_training.py --config config/training_config.yaml --use-unsloth"
echo ""
echo "Happy training! 🚀"
