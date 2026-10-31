import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from app.config import settings
from app.utils.logging_config import setup_logging

logger = setup_logging(settings.LOG_LEVEL)


class BaseVectorStore(ABC):
    @abstractmethod
    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        threshold: float = 0.35,
        conversation_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_document_chunks(self, document_id: str) -> None:
        pass


class LocalMemoryVectorStore(BaseVectorStore):
    """
    In-memory vector store computing cosine similarity.
    Provides fast, isolated, thread-safe vector retrieval for the local MVP.
    """
    def __init__(self):
        self._entries: List[Dict[str, Any]] = []

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        for chunk, emb in zip(chunks, embeddings):
            entry = {
                "chunk_id": chunk["id"],
                "document_id": chunk["document_id"],
                "document_name": chunk["document_name"],
                "page_number": chunk.get("page_number"),
                "chunk_index": chunk["chunk_index"],
                "chunk_text": chunk["chunk_text"],
                "conversation_id": chunk.get("conversation_id"),
                "embedding": emb,
            }
            self._entries.append(entry)
        logger.info(f"Added {len(chunks)} chunks to vector store. Total entries: {len(self._entries)}")

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        threshold: float = 0.35,
        conversation_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        scored = []
        for entry in self._entries:
            # Filter by conversation if entry belongs to a specific conversation
            if conversation_id and entry.get("conversation_id"):
                if entry["conversation_id"] != conversation_id:
                    continue

            score = self._cosine_similarity(query_embedding, entry["embedding"])
            if score >= threshold:
                item = dict(entry)
                item["score"] = round(score, 4)
                scored.append(item)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def delete_document_chunks(self, document_id: str) -> None:
        initial_len = len(self._entries)
        self._entries = [e for e in self._entries if e["document_id"] != document_id]
        removed = initial_len - len(self._entries)
        logger.info(f"Removed {removed} vector entries for document {document_id}")


# Global vector store instance
vector_store = LocalMemoryVectorStore()
