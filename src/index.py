"""
Vector indexing module for Process Copilot Mini
What to learn here: Embedding generation, FAISS vector search, persistence,
and how to build scalable similarity search systems.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging

import faiss
from sentence_transformers import SentenceTransformer

from .config import Config
from .ingest import DocumentChunk, DocumentIngester
from .utils import setup_logging, format_citations

logger = setup_logging()

class VectorIndex:
    """
    Manages document embeddings and FAISS vector search.
    
    Key concepts:
    - Embeddings: Dense vector representations of text that capture semantic meaning
    - FAISS: Efficient similarity search library for high-dimensional vectors
    - Persistence: Save/load index to disk to avoid recomputing embeddings
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.chunks: List[DocumentChunk] = []
        
        # Files for persistence
        self.index_file = Config.INDEX_DIR / "faiss_index.bin"
        self.chunks_file = Config.INDEX_DIR / "chunks_metadata.pkl"
        
        Config.ensure_directories()
    
    def _load_model(self):
        """
        Load sentence transformer model lazily.
        
        Why lazy loading:
        - Model loading is expensive, only do it when needed
        - Allows importing this module without loading the model
        - Better for testing and development
        """
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
    
    def build_index(self, chunks: List[DocumentChunk] = None) -> bool:
        """
        Build FAISS index from document chunks.
        
        Process:
        1. Load document chunks (from file or parameter)
        2. Generate embeddings for all chunk texts
        3. Create FAISS index and add embeddings
        4. Save index and metadata to disk
        
        Args:
            chunks: List of document chunks (if None, loads from disk)
            
        Returns:
            bool: Success status
        """
        try:
            # Load chunks if not provided
            if chunks is None:
                ingester = DocumentIngester()
                chunks = ingester.load_chunks()
            
            if not chunks:
                logger.warning("No chunks available to build index")
                return False
            
            self.chunks = chunks
            logger.info(f"Building index for {len(chunks)} chunks")
            
            # Load embedding model
            self._load_model()
            
            # Extract texts for embedding
            texts = [chunk.text for chunk in chunks]
            
            # Generate embeddings in batches for memory efficiency
            logger.info("Generating embeddings...")
            embeddings = self.model.encode(
                texts, 
                show_progress_bar=True,
                convert_to_numpy=True,
                batch_size=32  # Adjust based on available memory
            )
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            logger.info(f"Creating FAISS index with dimension {dimension}")
            
            # Use IndexFlatL2 for exact search (good for smaller datasets)
            # For larger datasets, consider IndexIVFFlat or IndexHNSWFlat
            self.index = faiss.IndexFlatL2(dimension)
            
            # Add embeddings to index
            self.index.add(embeddings.astype(np.float32))
            
            logger.info(f"Index built with {self.index.ntotal} vectors")
            
            # Save to disk
            self._save_index()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            return False
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_file))
            
            # Save chunks metadata
            with open(self.chunks_file, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            logger.info(f"Index saved to {self.index_file}")
            logger.info(f"Metadata saved to {self.chunks_file}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def load_index(self) -> bool:
        """
        Load FAISS index and metadata from disk.
        
        Returns:
            bool: Success status
        """
        try:
            if not self.index_file.exists() or not self.chunks_file.exists():
                logger.warning("Index files not found. Run build_index() first.")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_file))
            
            # Load chunks metadata
            with open(self.chunks_file, 'rb') as f:
                self.chunks = pickle.load(f)
            
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            logger.info(f"Loaded metadata for {len(self.chunks)} chunks")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.
        
        Process:
        1. Generate embedding for query text
        2. Search FAISS index for nearest neighbors
        3. Return chunks with similarity scores
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of search results with text, title, page, score
        """
        k = k or Config.RETRIEVAL_K
        
        try:
            # Ensure model and index are loaded
            if self.index is None:
                if not self.load_index():
                    logger.error("No index available for search")
                    return []
            
            if self.model is None:
                self._load_model()
            
            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            # Search index
            scores, indices = self.index.search(
                query_embedding.astype(np.float32), 
                min(k, self.index.ntotal)
            )
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks):  # Valid index
                    chunk = self.chunks[idx]
                    results.append({
                        'text': chunk.text,
                        'title': chunk.title,
                        'page': chunk.page,
                        'score': float(score),  # L2 distance (lower is better)
                        'chunk_id': chunk.chunk_id
                    })
            
            # Convert L2 distances to similarity scores (higher is better)
            # Use negative distance as similarity score
            for result in results:
                result['score'] = -result['score']
            
            # Sort by similarity (highest first)  
            results.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"Found {len(results)} results for query: '{query[:50]}...'")
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

def main():
    """
    Standalone script to build vector index.
    Run: python -m src.index
    """
    print("Building vector index...")
    
    # Create index
    index = VectorIndex()
    
    # Build from ingested documents
    success = index.build_index()
    
    if success:
        print(f"✅ Index built successfully!")
        print(f"📁 Index saved to: {index.index_file}")
        print(f"📄 Metadata saved to: {index.chunks_file}")
        
        # Test search
        print("\n🔍 Testing search...")
        results = index.search("temperature control", k=3)
        
        if results:
            print(f"Found {len(results)} test results:")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']} (page {result['page']}) - score: {result['score']:.3f}")
        else:
            print("No test results found")
    else:
        print("❌ Failed to build index")
        print("Make sure you have PDF files in data/pdfs/ and run 'python -m src.ingest' first")

if __name__ == "__main__":
    main()
