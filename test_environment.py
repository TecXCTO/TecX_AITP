"""
Test Script - Validate Environment and Components
"""

import sys
import torch
from pathlib import Path

def test_imports():
    """Test if all required packages are installed"""
    print("=== Testing Imports ===\n")
    
    packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("accelerate", "Accelerate"),
        ("peft", "PEFT"),
        ("bitsandbytes", "BitsAndBytes"),
        ("datasets", "Datasets"),
        ("fastapi", "FastAPI"),
        ("sentence_transformers", "Sentence Transformers"),
    ]
    
    failed = []
    for package, name in packages:
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            failed.append(name)
    
    # Optional packages
    print("\nOptional packages:")
    optional = [
        ("unsloth", "Unsloth"),
        ("flash_attn", "Flash Attention"),
        ("wandb", "Weights & Biases"),
    ]
    
    for package, name in optional:
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"- {name} - Not installed (optional)")
    
    if failed:
        print(f"\n❌ Missing required packages: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages installed!")
        return True


def test_cuda():
    """Test CUDA availability"""
    print("\n=== Testing CUDA ===\n")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("⚠️  CUDA not available - Training will be CPU-only (very slow)")
        return False


def test_flash_attention():
    """Test Flash Attention availability"""
    print("\n=== Testing Flash Attention ===\n")
    
    try:
        import flash_attn
        print(f"✓ Flash Attention 2 available")
        return True
    except ImportError:
        print("- Flash Attention 2 not available (optional but recommended)")
        print("  Install with: pip install flash-attn --no-build-isolation")
        return False


def test_model_architecture():
    """Test custom model architecture"""
    print("\n=== Testing Model Architecture ===\n")
    
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        from src.model.llm_architecture import CustomLLM, ModelConfig
        
        # Create small test model
        config = ModelConfig(
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            vocab_size=1000,
        )
        
        model = CustomLLM(config)
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ Model architecture working")
        print(f"  Test model parameters: {num_params/1e6:.2f}M")
        
        # Test forward pass
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            output = model(input_ids)
        
        print(f"✓ Forward pass working")
        print(f"  Output shape: {output['logits'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model architecture test failed: {e}")
        return False


def test_data_preprocessing():
    """Test data preprocessing"""
    print("\n=== Testing Data Preprocessing ===\n")
    
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        from src.data.preprocessor import DataCleaner, PreprocessingConfig
        
        cleaner = DataCleaner()
        
        # Test cleaning functions
        test_text = "<html>Test &nbsp; text   with  HTML</html>"
        cleaned = cleaner.clean_text(test_text, PreprocessingConfig(
            input_path="",
            output_dir="",
            clean_html=True,
            normalize_whitespace=True
        ))
        
        print(f"✓ Data cleaning working")
        print(f"  Input: {test_text}")
        print(f"  Output: {cleaned}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data preprocessing test failed: {e}")
        return False


def test_rag_components():
    """Test RAG components"""
    print("\n=== Testing RAG Components ===\n")
    
    try:
        import faiss
        print("✓ FAISS available")
    except ImportError:
        print("⚠️  FAISS not available")
        print("  Install with: pip install faiss-gpu (or faiss-cpu)")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Test embedding model
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(["Test sentence"])
        
        print(f"✓ Sentence Transformers working")
        print(f"  Embedding dimension: {embeddings.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"✗ RAG components test failed: {e}")
        return False


def test_directory_structure():
    """Test if directory structure is set up"""
    print("\n=== Testing Directory Structure ===\n")
    
    required_dirs = [
        "data/raw",
        "data/processed",
        "models/checkpoints",
        "config",
        "src/model",
        "src/training",
        "src/data",
        "src/rag",
        "src/deployment",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - MISSING")
            all_exist = False
    
    if all_exist:
        print("\n✅ Directory structure correct!")
    else:
        print("\n⚠️  Some directories missing. Run setup script.")
    
    return all_exist


def run_all_tests():
    """Run all validation tests"""
    print("=" * 60)
    print("CUSTOM LLM ENVIRONMENT VALIDATION")
    print("=" * 60)
    
    results = {
        "Imports": test_imports(),
        "CUDA": test_cuda(),
        "Flash Attention": test_flash_attention(),
        "Model Architecture": test_model_architecture(),
        "Data Preprocessing": test_data_preprocessing(),
        "RAG Components": test_rag_components(),
        "Directory Structure": test_directory_structure(),
    }
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Environment is ready.")
        print("\nNext steps:")
        print("1. Place your data in data/raw/dataset.jsonl")
        print("2. Run preprocessing: python src/data/preprocessor.py")
        print("3. Start training: python scripts/run_training.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Install CUDA: https://developer.nvidia.com/cuda-downloads")
        print("- Run setup script: bash scripts/setup_environment.sh")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
