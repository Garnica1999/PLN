"""Configuración central, leída de variables de entorno."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def _float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    milvus_uri: str
    collection: str
    embeddings_url: str
    embeddings_model: str
    embeddings_dims: int
    embeddings_batch: int
    embedder: str
    llm_url: str
    llm_model: str
    max_context_tokens: int
    model_context_tokens: int
    max_output_tokens: int
    gemma4_input_tokens: int
    gemma4_output_tokens: int
    reasoning_tokens: int
    top_k: int
    min_score: float
    parquet_path: str
    request_timeout: int
    prompt_version: str
    search_ef: int
    sparse_collection: str
    hybrid_dense: float
    hybrid_sparse: float
    retriever: str
    rerank: bool
    rerank_model: str
    rerank_device: str
    rerank_batch: int
    rerank_top_n: int
    rerank_score_ratio: float
    domain_classifier_path: str
    domain_threshold: float

    @property
    def encoder_tag(self) -> str:
        """Etiqueta que deben llevar las colecciones de este encoder."""
        if self.embedder == "bge":
            return "bge_m3"
        if self.embedder == "e5":
            return "e5base"
        if "qwen3-embedding" in self.embeddings_model.lower():
            return "qwen3emb"
        return "ollama"


def load_settings() -> Settings:
    return Settings(
        milvus_uri=os.getenv("MILVUS_URI", "http://milvus:19530"),
        collection=os.getenv("MILVUS_COLLECTION", "rag_pilot_qwen3emb8b_d1024_fixed_v1"),
        embeddings_url=os.getenv("EMBEDDINGS_URL", "http://192.168.1.100:11435"),
        embeddings_model=os.getenv("EMBEDDINGS_MODEL", "qwen3-embedding:8b"),
        embeddings_dims=_int("EMBEDDINGS_DIMS", 1024),
        embeddings_batch=_int("EMBEDDINGS_BATCH", 32),
        embedder=os.getenv("EMBEDDER", "ollama"),
        llm_url=os.getenv("OLLAMA_URL", "http://ollama:11434"),
        llm_model=os.getenv("LLM_MODEL", "gemma2:2b"),
        max_context_tokens=_int("MAX_CONTEXT_TOKENS", 16384),
        model_context_tokens=_int("MODEL_CONTEXT_TOKENS", 8192),
        max_output_tokens=_int("MAX_OUTPUT_TOKENS", 8192),
        gemma4_input_tokens=_int("GEMMA4_INPUT_TOKENS", 16384),
        gemma4_output_tokens=_int("GEMMA4_OUTPUT_TOKENS", 100000),
        reasoning_tokens=_int("REASONING_TOKENS", 1024),
        top_k=_int("TOP_K", 5),
        min_score=_float("MIN_SCORE", 0.0),
        parquet_path=os.getenv("PARQUET_PATH", "/work/data/documentos_rag.parquet"),
        request_timeout=_int("REQUEST_TIMEOUT", 240),
        prompt_version=os.getenv("PROMPT_VERSION", "v2"),
        search_ef=_int("SEARCH_EF", 200),
        sparse_collection=os.getenv("SPARSE_COLLECTION", "rag_full_e5base_d768_v2_bm25_v1"),
        hybrid_dense=_float("HYBRID_DENSE", 0.3),
        hybrid_sparse=_float("HYBRID_SPARSE", 0.7),
        retriever=os.getenv("RETRIEVER", "sparse"),
        rerank=os.getenv("RERANK", "1") == "1",
        rerank_model=os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        rerank_device=os.getenv("RERANK_DEVICE", "cuda"),
        rerank_batch=_int("RERANK_BATCH", 16),
        rerank_top_n=_int("RERANK_TOP_N", 0),
        rerank_score_ratio=_float("RERANK_SCORE_RATIO", 0.0),
        domain_classifier_path=os.getenv("DOMAIN_CLASSIFIER_PATH", "modelos/domain_classifier.json"),
        domain_threshold=_float("DOMAIN_THRESHOLD", 0.5),
    )


settings = load_settings()
