import math
import re
from abc import ABC, abstractmethod
from typing import List, Optional
import httpx
from app.config import settings
from app.utils.logging_config import setup_logging

logger = setup_logging(settings.LOG_LEVEL)


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates dense vector embeddings for a list of text strings."""
        pass

    def embed_query(self, query: str) -> List[float]:
        """Generates embedding for a single query string."""
        return self.embed_texts([query])[0]


class LocalHashEmbeddingProvider(BaseEmbeddingProvider):
    """
    Offline semantic feature hashing embedding provider.
    Computes normalized n-gram token and subword frequency vectors (dim=128)
    with L2 normalization. Excellent for fast, zero-dependency, deterministic local retrieval.
    """
    def __init__(self, dimension: int = 128):
        self.dim = dimension

    def _hash_token(self, token: str) -> int:
        h = 2166136261
        for ch in token:
            h = (h ^ ord(ch)) * 16777619
        return abs(h) % self.dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            vec = [0.0] * self.dim
            tokens = re.findall(r"\w+", text.lower())
            
            # Word tokens
            for t in tokens:
                idx = self._hash_token(t)
                vec[idx] += 1.0

            # Character 3-grams for subword matching
            for i in range(len(text) - 2):
                ngram = text[i:i+3].lower()
                idx = self._hash_token(ngram)
                vec[idx] += 0.5

            # L2 normalization
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            else:
                vec = [0.0] * self.dim

            embeddings.append(vec)
        return embeddings


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    Adapter for OpenAI and OpenAI-compatible embedding endpoints.
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.EMBEDDING_API_KEY or settings.LLM_API_KEY
        self.model = model or settings.EMBEDDING_MODEL
        self.base_url = (settings.LLM_BASE_URL or "https://api.openai.com/v1").rstrip("/")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            logger.warning("Embedding API key not configured, falling back to LocalHashEmbeddingProvider")
            return LocalHashEmbeddingProvider().embed_texts(texts)

        url = f"{self.base_url}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": self.model,
            "input": texts,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(url, headers=headers, json=data)
                res.raise_for_status()
                res_json = res.json()
                return [item["embedding"] for item in res_json["data"]]
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}, using local fallback")
            return LocalHashEmbeddingProvider().embed_texts(texts)


def get_embedding_provider() -> BaseEmbeddingProvider:
    provider = settings.EMBEDDING_PROVIDER.lower().strip()
    if provider == "openai" and (settings.EMBEDDING_API_KEY or settings.LLM_API_KEY):
        return OpenAIEmbeddingProvider()
    return LocalHashEmbeddingProvider()
