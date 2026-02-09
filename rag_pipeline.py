"""
RAG (Retrieval-Augmented Generation) Implementation
Supports FAISS and Pinecone for vector storage
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import pickle

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store_type: str = "faiss"  # faiss, pinecone, chromadb
    top_k: int = 5
    score_threshold: float = 0.7
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_context_length: int = 4096
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingModel:
    """Wrapper for embedding models"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """Encode texts to embeddings"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings


class FAISSVectorStore:
    """FAISS-based vector store for fast similarity search"""
    
    def __init__(self, dimension: int, index_type: str = "IVF"):
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("Please install faiss: pip install faiss-gpu or faiss-cpu")
        
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.documents = []
        self.metadata = []
        
        self._create_index()
    
    def _create_index(self):
        """Create FAISS index"""
        if self.index_type == "Flat":
            # Exact search (slower but accurate)
            self.index = self.faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVF":
            # Approximate search with inverted file index
            quantizer = self.faiss.IndexFlatL2(self.dimension)
            self.index = self.faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World graphs
            self.index = self.faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def add_documents(self, embeddings: np.ndarray, documents: List[str], metadata: List[Dict] = None):
        """Add documents to the index"""
        embeddings = embeddings.astype('float32')
        
        # Train index if needed
        if isinstance(self.index, self.faiss.IndexIVFFlat) and not self.index.is_trained:
            self.index.train(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        self.documents.extend(documents)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[List[str], List[float], List[Dict]]:
        """Search for similar documents"""
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Convert distances to similarity scores (for L2 distance)
        scores = 1 / (1 + distances[0])
        
        # Get documents and metadata
        results = []
        result_scores = []
        result_metadata = []
        
        for idx, score in zip(indices[0], scores):
            if idx < len(self.documents):
                results.append(self.documents[idx])
                result_scores.append(float(score))
                result_metadata.append(self.metadata[idx])
        
        return results, result_scores, result_metadata
    
    def save(self, path: str):
        """Save index and documents"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        self.faiss.write_index(self.index, f"{path}/index.faiss")
        
        # Save documents and metadata
        with open(f"{path}/documents.pkl", 'wb') as f:
            pickle.dump({'documents': self.documents, 'metadata': self.metadata}, f)
    
    def load(self, path: str):
        """Load index and documents"""
        # Load FAISS index
        self.index = self.faiss.read_index(f"{path}/index.faiss")
        
        # Load documents and metadata
        with open(f"{path}/documents.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']


class PineconeVectorStore:
    """Pinecone-based vector store"""
    
    def __init__(self, api_key: str, environment: str, index_name: str, dimension: int):
        try:
            import pinecone
            self.pinecone = pinecone
        except ImportError:
            raise ImportError("Please install pinecone: pip install pinecone-client")
        
        # Initialize Pinecone
        self.pinecone.init(api_key=api_key, environment=environment)
        
        # Create or connect to index
        if index_name not in self.pinecone.list_indexes():
            self.pinecone.create_index(
                index_name,
                dimension=dimension,
                metric="cosine"
            )
        
        self.index = self.pinecone.Index(index_name)
    
    def add_documents(self, embeddings: np.ndarray, documents: List[str], metadata: List[Dict] = None):
        """Add documents to Pinecone"""
        vectors = []
        
        for i, (embedding, doc) in enumerate(zip(embeddings, documents)):
            vector_id = f"doc_{i}"
            vector_metadata = metadata[i] if metadata else {}
            vector_metadata['text'] = doc
            
            vectors.append({
                'id': vector_id,
                'values': embedding.tolist(),
                'metadata': vector_metadata
            })
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[List[str], List[float], List[Dict]]:
        """Search in Pinecone"""
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        documents = []
        scores = []
        metadata = []
        
        for match in results['matches']:
            documents.append(match['metadata']['text'])
            scores.append(match['score'])
            meta = {k: v for k, v in match['metadata'].items() if k != 'text'}
            metadata.append(meta)
        
        return documents, scores, metadata


class DocumentChunker:
    """Split documents into chunks for embedding"""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def chunk_documents(documents: List[str], chunk_size: int = 512, chunk_overlap: int = 50) -> Tuple[List[str], List[Dict]]:
        """Chunk multiple documents with metadata"""
        all_chunks = []
        all_metadata = []
        
        for doc_idx, doc in enumerate(documents):
            chunks = DocumentChunker.chunk_text(doc, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
            
            # Add metadata for each chunk
            for chunk_idx in range(len(chunks)):
                all_metadata.append({
                    'doc_idx': doc_idx,
                    'chunk_idx': chunk_idx,
                    'total_chunks': len(chunks)
                })
        
        return all_chunks, all_metadata


class RAGPipeline:
    """Complete RAG pipeline with retrieval and generation"""
    
    def __init__(
        self,
        llm_model_name: str,
        config: RAGConfig,
        vector_store: Optional[FAISSVectorStore] = None
    ):
        self.config = config
        
        # Initialize embedding model
        self.embedding_model = EmbeddingModel(config.embedding_model, config.device)
        
        # Initialize vector store
        if vector_store is None:
            if config.vector_store_type == "faiss":
                self.vector_store = FAISSVectorStore(
                    dimension=self.embedding_model.dimension,
                    index_type="IVF"
                )
            else:
                raise ValueError(f"Vector store type {config.vector_store_type} not supported in this example")
        else:
            self.vector_store = vector_store
        
        # Initialize LLM
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        
        self.chunker = DocumentChunker()
    
    def index_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """Index documents for retrieval"""
        print(f"Chunking {len(documents)} documents...")
        chunks, chunk_metadata = self.chunker.chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        print(f"Embedding {len(chunks)} chunks...")
        embeddings = self.embedding_model.encode(chunks)
        
        print("Adding to vector store...")
        self.vector_store.add_documents(embeddings, chunks, chunk_metadata)
        
        print(f"Indexed {len(chunks)} chunks from {len(documents)} documents")
    
    def retrieve(self, query: str) -> Tuple[List[str], List[float]]:
        """Retrieve relevant documents for a query"""
        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search
        documents, scores, metadata = self.vector_store.search(
            query_embedding,
            top_k=self.config.top_k
        )
        
        # Filter by threshold
        filtered_docs = []
        filtered_scores = []
        
        for doc, score in zip(documents, scores):
            if score >= self.config.score_threshold:
                filtered_docs.append(doc)
                filtered_scores.append(score)
        
        return filtered_docs, filtered_scores
    
    def generate_with_context(self, query: str, context: List[str], max_length: int = 512) -> str:
        """Generate response using retrieved context"""
        # Format prompt with context
        context_text = "\n\n".join(context)
        prompt = f"""Context information:
{context_text}

Question: {query}

Answer based on the context above:"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_context_length)
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part
        if "Answer based on the context above:" in response:
            response = response.split("Answer based on the context above:")[-1].strip()
        
        return response
    
    def query(self, query: str, return_sources: bool = True) -> Dict[str, any]:
        """Complete RAG query with retrieval and generation"""
        # Retrieve relevant documents
        context_docs, scores = self.retrieve(query)
        
        if not context_docs:
            return {
                'answer': "I don't have enough context to answer this question.",
                'sources': [],
                'scores': []
            }
        
        # Generate answer
        answer = self.generate_with_context(query, context_docs)
        
        result = {'answer': answer}
        
        if return_sources:
            result['sources'] = context_docs
            result['scores'] = scores
        
        return result


if __name__ == "__main__":
    # Example usage
    config = RAGConfig(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vector_store_type="faiss",
        top_k=5,
    )
    
    # Initialize RAG pipeline
    rag = RAGPipeline(
        llm_model_name="meta-llama/Meta-Llama-3-8B",
        config=config
    )
    
    # Index some documents
    documents = [
        "The capital of France is Paris. It is known for the Eiffel Tower.",
        "Python is a high-level programming language. It is widely used for machine learning.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
    ]
    
    rag.index_documents(documents)
    
    # Query
    result = rag.query("What is the capital of France?")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
