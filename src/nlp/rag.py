from typing import List, Dict, Any, Optional
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.utils import embedding_functions
from src.utils.logging import log

class RAGEngine:
    """
    Advanced Retrieval-Augmented Generation (RAG) Engine with 
    2-stage retrieval (Hybrid Search + Re-ranking).
    """
    
    def __init__(
        self, 
        collection_name: str = "nexus_knowledge",
        embedding_model: str = "all-MiniLM-L6-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        persist_directory: str = "data/vector_db"
    ):
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        
        # Setup ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            embedding_function=self.embedding_func
        )
        
        # Setup Re-ranker
        log.info(f"Loading Re-ranker: {cross_encoder_model}")
        self.re_ranker = CrossEncoder(cross_encoder_model)
        log.info("RAG Engine initialized.")

    def ingest(self, documents: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
        """
        Adds documents to the knowledge base.
        """
        if not ids:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        log.info(f"Ingesting {len(documents)} documents into vector database...")
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        log.info("Ingestion complete.")

    def query(self, user_query: str, top_k_initial: int = 15, top_k_rerank: int = 4) -> Dict[str, Any]:
        """
        Performs 2-stage retrieval:
        1. Dense Retrieval (Semantic Search)
        2. Cross-Encoder Re-ranking
        """
        log.info(f"Processing query: {user_query}")
        
        # Stage 1: Dense Retrieval
        results = self.collection.query(
            query_texts=[user_query],
            n_results=top_k_initial
        )
        
        candidates = results['documents'][0]
        if not candidates:
            log.warning("No relevant documents found in stage 1.")
            return {"query": user_query, "results": [], "context": ""}
            
        # Stage 2: Re-ranking
        log.info(f"Re-ranking {len(candidates)} candidates...")
        pairs = [[user_query, doc] for doc in candidates]
        scores = self.re_ranker.predict(pairs)
        
        # Sort by score
        ranked_results = sorted(
            zip(candidates, scores), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k_rerank]
        
        top_docs = [doc for doc, score in ranked_results]
        context = "\n\n".join(top_docs)
        
        log.info(f"Retrieval finished. Top score: {ranked_results[0][1]:.4f}")
        
        return {
            "query": user_query,
            "ranked_results": [{"doc": doc, "score": float(score)} for doc, score in ranked_results],
            "context": context
        }
