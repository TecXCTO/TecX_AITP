"""
GGUF Export Script
Convert trained models to GGUF format for local execution via LM Studio or Ollama
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class GGUFExporter:
    """Export models to GGUF format"""
    
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def check_llama_cpp(self) -> bool:
        """Check if llama.cpp is available"""
        try:
            result = subprocess.run(
                ["python", "-c", "import llama_cpp"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def install_llama_cpp(self):
        """Install llama-cpp-python"""
        print("Installing llama-cpp-python...")
        subprocess.run([
            "pip", "install", "llama-cpp-python", "--upgrade",
            "--force-reinstall", "--no-cache-dir"
        ])
    
    def export_to_fp16(self) -> str:
        """Export model to FP16 format (intermediate step)"""
        print("Loading model for FP16 export...")
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Save in FP16
        fp16_path = f"{self.output_dir}/fp16"
        Path(fp16_path).mkdir(parents=True, exist_ok=True)
        
        print(f"Saving FP16 model to {fp16_path}...")
        model.save_pretrained(fp16_path, max_shard_size="5GB")
        tokenizer.save_pretrained(fp16_path)
        
        return fp16_path
    
    def convert_to_gguf(self, fp16_path: str, quantization: str = "q4_k_m"):
        """
        Convert FP16 model to GGUF format
        
        Quantization options:
        - q4_0: 4-bit, lower quality
        - q4_k_m: 4-bit, medium quality (recommended)
        - q5_k_m: 5-bit, high quality
        - q8_0: 8-bit, very high quality
        - f16: 16-bit, original quality
        """
        print(f"Converting to GGUF format with {quantization} quantization...")
        
        # Clone llama.cpp if not exists
        llamacpp_dir = "/tmp/llama.cpp"
        if not Path(llamacpp_dir).exists():
            print("Cloning llama.cpp repository...")
            subprocess.run([
                "git", "clone", "https://github.com/ggerganov/llama.cpp.git",
                llamacpp_dir
            ])
        
        # Convert to GGUF using llama.cpp scripts
        convert_script = f"{llamacpp_dir}/convert.py"
        
        if not Path(convert_script).exists():
            convert_script = f"{llamacpp_dir}/convert-hf-to-gguf.py"
        
        # Output GGUF file
        output_file = f"{self.output_dir}/model-{quantization}.gguf"
        
        # Run conversion
        print("Running conversion script...")
        subprocess.run([
            "python", convert_script,
            fp16_path,
            "--outfile", output_file,
            "--outtype", quantization
        ])
        
        print(f"GGUF model saved to: {output_file}")
        return output_file
    
    def export(self, quantization: str = "q4_k_m", keep_fp16: bool = False):
        """Complete export pipeline"""
        print("=== Starting GGUF Export ===")
        
        # Step 1: Export to FP16
        fp16_path = self.export_to_fp16()
        
        # Step 2: Convert to GGUF
        gguf_path = self.convert_to_gguf(fp16_path, quantization)
        
        # Clean up FP16 if not needed
        if not keep_fp16:
            print("Cleaning up FP16 files...")
            shutil.rmtree(fp16_path)
        
        print("\n=== Export Complete ===")
        print(f"GGUF model: {gguf_path}")
        print(f"Size: {Path(gguf_path).stat().st_size / (1024**3):.2f} GB")
        print("\nYou can now use this model with:")
        print("- LM Studio: Import the GGUF file")
        print("- Ollama: Create a Modelfile and import")
        print("- llama.cpp: Use directly with llama-cli")
        
        return gguf_path


class OllamaModelfile:
    """Generate Ollama Modelfile for custom model"""
    
    @staticmethod
    def create_modelfile(
        gguf_path: str,
        output_path: str,
        model_name: str = "custom-llm",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
    ):
        """Create Ollama Modelfile"""
        modelfile_content = f"""# Modelfile for {model_name}
FROM {gguf_path}

# Model parameters
PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER top_k {top_k}
PARAMETER num_ctx 8192

"""
        
        if system_prompt:
            modelfile_content += f'''# System prompt
SYSTEM """{system_prompt}"""

'''
        
        modelfile_content += """# Chat template (Llama-3 style)
TEMPLATE \"\"\"
{{- if .System }}
<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>
{{- end }}
{{- range .Messages }}
<|start_header_id|>{{ .Role }}<|end_header_id|>

{{ .Content }}<|eot_id|>
{{- end }}
<|start_header_id|>assistant<|end_header_id|>

\"\"\"

# Stop sequences
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
"""
        
        # Write Modelfile
        with open(output_path, 'w') as f:
            f.write(modelfile_content)
        
        print(f"Modelfile created at: {output_path}")
        print("\nTo use with Ollama:")
        print(f"1. ollama create {model_name} -f {output_path}")
        print(f"2. ollama run {model_name}")


def create_quantization_variants(model_path: str, output_dir: str):
    """Create multiple quantization variants"""
    exporter = GGUFExporter(model_path, output_dir)
    
    # Export FP16 once
    fp16_path = exporter.export_to_fp16()
    
    # Create different quantization levels
    quantizations = {
        "q4_k_m": "4-bit medium quality (recommended)",
        "q5_k_m": "5-bit high quality",
        "q8_0": "8-bit very high quality",
    }
    
    for quant, description in quantizations.items():
        print(f"\n=== Creating {quant} variant ({description}) ===")
        exporter.convert_to_gguf(fp16_path, quant)
    
    # Clean up FP16
    print("Cleaning up FP16 files...")
    shutil.rmtree(fp16_path)
    
    print("\n=== All variants created ===")


def main():
    parser = argparse.ArgumentParser(description="Export model to GGUF format")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--output-dir", default="./models/gguf", help="Output directory")
    parser.add_argument(
        "--quantization",
        default="q4_k_m",
        choices=["q4_0", "q4_k_m", "q5_k_m", "q8_0", "f16"],
        help="Quantization level"
    )
    parser.add_argument("--keep-fp16", action="store_true", help="Keep FP16 intermediate files")
    parser.add_argument("--all-variants", action="store_true", help="Create all quantization variants")
    parser.add_argument("--create-modelfile", action="store_true", help="Create Ollama Modelfile")
    parser.add_argument("--model-name", default="custom-llm", help="Model name for Ollama")
    
    args = parser.parse_args()
    
    if args.all_variants:
        create_quantization_variants(args.model_path, args.output_dir)
    else:
        exporter = GGUFExporter(args.model_path, args.output_dir)
        gguf_path = exporter.export(args.quantization, args.keep_fp16)
        
        if args.create_modelfile:
            modelfile_path = f"{args.output_dir}/Modelfile"
            OllamaModelfile.create_modelfile(
                gguf_path,
                modelfile_path,
                model_name=args.model_name
            )


if __name__ == "__main__":
    main()
