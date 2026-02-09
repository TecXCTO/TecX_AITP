"""
FastAPI Server - OpenAI-compatible API
Deploy your custom LLM with an API compatible with OpenAI's format
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, AsyncGenerator
import torch
import uvicorn
import time
import uuid
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import asyncio


# Request/Response Models (OpenAI-compatible)
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "custom-llm"
    messages: List[Message]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stream: bool = False
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop: Optional[List[str]] = None


class CompletionRequest(BaseModel):
    model: str = "custom-llm"
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stream: bool = False
    stop: Optional[List[str]] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "custom"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# LLM Model Manager
class LLMModelManager:
    """Manages model loading and inference"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load model and tokenizer"""
        print(f"Loading model from {self.model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def format_chat_prompt(self, messages: List[Message]) -> str:
        """Format messages into a single prompt (Llama-3 style)"""
        formatted = ""
        
        for message in messages:
            role = message.role
            content = message.content
            
            if role == "system":
                formatted += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "user":
                formatted += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "assistant":
                formatted += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
        
        # Add assistant header for response
        formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return formatted
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate completion"""
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        
        prompt_tokens = input_ids.shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_tokens = outputs[0][prompt_tokens:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Handle stop sequences
        if stop:
            for stop_seq in stop:
                if stop_seq in response:
                    response = response.split(stop_seq)[0]
        
        completion_tokens = len(generated_tokens)
        
        return {
            'text': response,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
        }
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AsyncGenerator[str, None]:
        """Generate completion with streaming"""
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        
        # Setup streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        
        # Generation kwargs
        generation_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_new_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'do_sample': temperature > 0,
            'streamer': streamer,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        
        # Start generation in background thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream tokens
        for text in streamer:
            yield text
            await asyncio.sleep(0.01)  # Small delay for smooth streaming


# Initialize FastAPI app
app = FastAPI(
    title="Custom LLM API",
    description="OpenAI-compatible API for custom LLM",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model manager
model_manager: Optional[LLMModelManager] = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model_manager
    # Set your model path here
    model_path = "models/fine_tuned/final"  # Update this path
    model_manager = LLMModelManager(model_path)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Custom LLM API Server",
        "version": "1.0.0",
        "endpoints": ["/v1/chat/completions", "/v1/completions", "/v1/models"]
    }


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models (OpenAI-compatible)"""
    return ModelsResponse(
        object="list",
        data=[
            ModelInfo(
                id="custom-llm",
                created=int(time.time()),
                owned_by="custom"
            )
        ]
    )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create chat completion (OpenAI-compatible)"""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Format prompt
    prompt = model_manager.format_chat_prompt(request.messages)
    
    # Handle streaming
    if request.stream:
        async def generate_stream():
            chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
            
            async for token in model_manager.generate_stream(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            ):
                chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {str(chunk)}\n\n"
            
            # Final chunk
            final_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {str(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    
    # Non-streaming response
    result = model_manager.generate(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
    )
    
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=result['text']),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=result['prompt_tokens'],
            completion_tokens=result['completion_tokens'],
            total_tokens=result['total_tokens']
        )
    )


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Create text completion (OpenAI-compatible)"""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    result = model_manager.generate(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
    )
    
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "text": result['text'],
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": result['prompt_tokens'],
            "completion_tokens": result['completion_tokens'],
            "total_tokens": result['total_tokens']
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_manager is not None,
        "timestamp": datetime.now().isoformat()
    }


def main(host: str = "0.0.0.0", port: int = 8000, model_path: str = None):
    """Run the API server"""
    if model_path:
        global model_manager
        model_manager = LLMModelManager(model_path)
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Custom LLM API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-path", required=True, help="Path to model")
    
    args = parser.parse_args()
    main(host=args.host, port=args.port, model_path=args.model_path)
