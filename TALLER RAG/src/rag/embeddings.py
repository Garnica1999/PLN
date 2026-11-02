"""Codificación de consultas y pasajes con el mismo modelo."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
import requests

from .config import settings

QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query: "
)


class Embedder(ABC):
    dims: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


class OllamaEmbedder(Embedder):
    """Cliente del servicio de embeddings remoto (Ollama en el servidor)."""

    def __init__(
        self,
        url: str = settings.embeddings_url,
        model: str = settings.embeddings_model,
        dims: int = settings.embeddings_dims,
        batch: int = settings.embeddings_batch,
        timeout: int = settings.request_timeout,
        retries: int = 3,
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.dims = dims
        self.batch = max(1, batch)
        self.name = model
        self.timeout = timeout
        self.retries = retries

    def _post(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": texts,
            "options": {"num_ctx": 512},
        }
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = requests.post(
                    f"{self.url}/api/embed", json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                vectors = data.get("embeddings")
                if not vectors or len(vectors) != len(texts):
                    raise ValueError("respuesta de embeddings incompleta")
                return vectors
            except (requests.RequestException, ValueError) as error:
                last_error = error
                if attempt < self.retries - 1:
                    time.sleep(2**attempt)
        raise RuntimeError(f"fallo el servicio de embeddings: {last_error}")

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch):
            vectors.extend(self._post(texts[start : start + self.batch]))
        array = np.asarray(vectors, dtype=np.float32)
        if self.dims and array.shape[1] > self.dims:
            array = array[:, : self.dims]
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.maximum(norms, 1e-12)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        prefix = QUERY_INSTRUCTION if "qwen3-embedding" in self.model.lower() else ""
        return self._encode([prefix + text for text in texts])

    def health(self) -> bool:
        try:
            response = requests.get(f"{self.url}/api/tags", timeout=5)
            return response.ok
        except requests.RequestException:
            return False


class SentenceTransformerEmbedder(Embedder):
    """Embedder local con SentenceTransformer. E5 usa prefijos; bge-m3 no."""

    def __init__(self, model: str | None = None, prefixes: bool = True) -> None:
        import os

        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(int(os.getenv("TORCH_THREADS", "8")))
        model = model or os.getenv("EMBEDDINGS_MODEL", "intfloat/multilingual-e5-base")
        self.name = model
        self.prefixes = prefixes
        self.model = SentenceTransformer(model, device=os.getenv("EMBEDDING_DEVICE", "cpu"))
        if os.getenv("EMBEDDING_HALF", "false").lower() == "true":
            self.model.half()
        self.dims = int(self.model.get_sentence_embedding_dimension())
        self.batch = int(os.getenv("EMBEDDINGS_BATCH", "64"))

    def _texts(self, kind: str, texts: list[str]) -> list[str]:
        prefix = {"passage": "passage: ", "query": "query: "}[kind] if self.prefixes else ""
        return [prefix + text for text in texts]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            self._texts("passage", texts),
            normalize_embeddings=True,
            batch_size=self.batch,
        )

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            self._texts("query", texts),
            normalize_embeddings=True,
            batch_size=self.batch,
        )

    def health(self) -> bool:
        return True


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    if settings.embedder == "bge":
        return SentenceTransformerEmbedder(model=settings.embeddings_model, prefixes=False)
    if settings.embedder == "e5":
        return SentenceTransformerEmbedder()
    return OllamaEmbedder()
