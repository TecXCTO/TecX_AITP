"""
API Client Examples
Test your deployed model with various use cases
"""

import requests
import json
from typing import List, Dict


class CustomLLMClient:
    """Client for interacting with deployed Custom LLM API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False
    ):
        """Send chat completion request"""
        url = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "model": "custom-llm",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        if stream:
            return self._stream_response(url, payload)
        else:
            response = requests.post(url, json=payload)
            return response.json()
    
    def text_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ):
        """Send text completion request"""
        url = f"{self.base_url}/v1/completions"
        
        payload = {
            "model": "custom-llm",
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(url, json=payload)
        return response.json()
    
    def _stream_response(self, url: str, payload: dict):
        """Handle streaming response"""
        with requests.post(url, json=payload, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data != '[DONE]':
                            yield json.loads(data)
    
    def health_check(self):
        """Check API health"""
        url = f"{self.base_url}/health"
        response = requests.get(url)
        return response.json()
    
    def list_models(self):
        """List available models"""
        url = f"{self.base_url}/v1/models"
        response = requests.get(url)
        return response.json()


def example_simple_chat():
    """Example: Simple chat interaction"""
    print("=== Simple Chat Example ===\n")
    
    client = CustomLLMClient()
    
    messages = [
        {"role": "user", "content": "What is machine learning?"}
    ]
    
    response = client.chat_completion(messages, temperature=0.7, max_tokens=200)
    
    print(f"User: {messages[0]['content']}")
    print(f"Assistant: {response['choices'][0]['message']['content']}")
    print(f"\nTokens used: {response['usage']['total_tokens']}")


def example_multi_turn_chat():
    """Example: Multi-turn conversation"""
    print("\n=== Multi-turn Chat Example ===\n")
    
    client = CustomLLMClient()
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant specialized in technology."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language known for its simplicity and versatility."},
        {"role": "user", "content": "What are its main uses?"}
    ]
    
    response = client.chat_completion(messages, temperature=0.7, max_tokens=300)
    
    print("Conversation:")
    for msg in messages:
        print(f"{msg['role'].title()}: {msg['content']}")
    
    print(f"\nAssistant: {response['choices'][0]['message']['content']}")


def example_streaming_chat():
    """Example: Streaming response"""
    print("\n=== Streaming Chat Example ===\n")
    
    client = CustomLLMClient()
    
    messages = [
        {"role": "user", "content": "Write a short poem about AI"}
    ]
    
    print("User: Write a short poem about AI")
    print("Assistant: ", end="", flush=True)
    
    for chunk in client.chat_completion(messages, stream=True):
        if 'choices' in chunk and chunk['choices']:
            content = chunk['choices'][0].get('delta', {}).get('content', '')
            print(content, end="", flush=True)
    
    print("\n")


def example_code_generation():
    """Example: Code generation"""
    print("\n=== Code Generation Example ===\n")
    
    client = CustomLLMClient()
    
    messages = [
        {"role": "system", "content": "You are an expert Python programmer."},
        {"role": "user", "content": "Write a function to calculate the Fibonacci sequence"}
    ]
    
    response = client.chat_completion(messages, temperature=0.3, max_tokens=500)
    
    print("Request: Write a function to calculate the Fibonacci sequence")
    print("\nGenerated Code:")
    print(response['choices'][0]['message']['content'])


def example_text_completion():
    """Example: Text completion"""
    print("\n=== Text Completion Example ===\n")
    
    client = CustomLLMClient()
    
    prompt = "The three laws of robotics are:"
    
    response = client.text_completion(prompt, temperature=0.5, max_tokens=200)
    
    print(f"Prompt: {prompt}")
    print(f"Completion: {response['choices'][0]['text']}")


def example_creative_writing():
    """Example: Creative writing"""
    print("\n=== Creative Writing Example ===\n")
    
    client = CustomLLMClient()
    
    messages = [
        {"role": "system", "content": "You are a creative writer."},
        {"role": "user", "content": "Write the opening paragraph of a sci-fi story about AI"}
    ]
    
    response = client.chat_completion(messages, temperature=0.9, max_tokens=300)
    
    print("Story Opening:")
    print(response['choices'][0]['message']['content'])


def example_with_rag():
    """Example: Using with RAG context"""
    print("\n=== RAG-Enhanced Example ===\n")
    
    client = CustomLLMClient()
    
    # Simulate retrieved context
    context = """
    Context: Our company's new product, AI Assistant Pro, features:
    - Advanced natural language understanding
    - Multi-modal capabilities (text, image, audio)
    - Customizable for different industries
    - 99.9% uptime SLA
    - Deployed on secure cloud infrastructure
    """
    
    messages = [
        {"role": "system", "content": "Answer based on the provided context."},
        {"role": "user", "content": f"{context}\n\nQuestion: What are the key features of AI Assistant Pro?"}
    ]
    
    response = client.chat_completion(messages, temperature=0.3, max_tokens=300)
    
    print("Question: What are the key features of AI Assistant Pro?")
    print(f"\nAnswer: {response['choices'][0]['message']['content']}")


def benchmark_performance():
    """Benchmark API performance"""
    print("\n=== Performance Benchmark ===\n")
    
    import time
    
    client = CustomLLMClient()
    
    # Test different prompt lengths
    test_cases = [
        ("Short prompt", "Hello!"),
        ("Medium prompt", "Explain the concept of neural networks in detail."),
        ("Long prompt", "Provide a comprehensive overview of machine learning, including supervised learning, unsupervised learning, reinforcement learning, and deep learning. Include examples and applications." * 3)
    ]
    
    results = []
    
    for name, prompt in test_cases:
        start_time = time.time()
        
        response = client.text_completion(prompt, max_tokens=100)
        
        end_time = time.time()
        duration = end_time - start_time
        
        tokens = response['usage']['total_tokens']
        tokens_per_sec = tokens / duration
        
        results.append({
            'name': name,
            'duration': duration,
            'tokens': tokens,
            'tokens_per_sec': tokens_per_sec
        })
    
    print("Benchmark Results:")
    print("-" * 60)
    for result in results:
        print(f"{result['name']}:")
        print(f"  Duration: {result['duration']:.2f}s")
        print(f"  Tokens: {result['tokens']}")
        print(f"  Speed: {result['tokens_per_sec']:.1f} tokens/sec")
        print()


def test_health_and_models():
    """Test health check and model listing"""
    print("\n=== Health Check & Models ===\n")
    
    client = CustomLLMClient()
    
    # Health check
    health = client.health_check()
    print(f"API Status: {health['status']}")
    print(f"Model Loaded: {health['model_loaded']}")
    
    # List models
    models = client.list_models()
    print(f"\nAvailable Models:")
    for model in models['data']:
        print(f"  - {model['id']}")


def run_all_examples():
    """Run all example use cases"""
    print("=" * 60)
    print("CUSTOM LLM API CLIENT EXAMPLES")
    print("=" * 60)
    
    try:
        # Check if API is running
        client = CustomLLMClient()
        client.health_check()
        
        # Run examples
        test_health_and_models()
        example_simple_chat()
        example_multi_turn_chat()
        example_code_generation()
        example_text_completion()
        example_creative_writing()
        example_with_rag()
        example_streaming_chat()
        benchmark_performance()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server")
        print("Make sure the API server is running:")
        print("  python src/deployment/api_server.py --model-path models/fine_tuned/final")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    run_all_examples()
