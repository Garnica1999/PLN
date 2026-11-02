"""El prefijo de qwen3 no debe aplicarse a otros modelos remotos."""
from __future__ import annotations

import numpy as np

from src.rag.embeddings import OllamaEmbedder


def test_prefijo_solo_para_qwen(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_encode(self, texts: list[str]) -> np.ndarray:
        captured["texts"] = texts
        return np.zeros((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(OllamaEmbedder, "_encode", fake_encode)
    OllamaEmbedder(model="qwen3-embedding:8b").embed_queries(["hola"])
    assert captured["texts"][0].startswith("Instruct:")
    OllamaEmbedder(model="bge-m3").embed_queries(["hola"])
    assert captured["texts"][0] == "hola"
