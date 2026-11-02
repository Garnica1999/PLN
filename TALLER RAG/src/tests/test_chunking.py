"""Pruebas de segmentación: offsets, cobertura y límites."""
from __future__ import annotations

import pytest

from src.rag.chunking import (
    chunk_fixed,
    chunk_sentences,
    estimate_tokens,
    split_sentences,
)


def test_fixed_offsets_y_cobertura() -> None:
    text = "abcdefghij" * 300
    chunks = list(chunk_fixed(text, size=1000, overlap=150))
    assert all(chunk.text == text[chunk.start : chunk.end] for chunk in chunks)
    assert chunks[-1].end == len(text)
    assert max(len(chunk.text) for chunk in chunks) <= 1000
    assert chunks[1].start == chunks[0].end - 150


def test_fixed_parametros_invalidos() -> None:
    with pytest.raises(ValueError):
        list(chunk_fixed("x", size=100, overlap=100))
    with pytest.raises(ValueError):
        list(chunk_fixed("x", size=0, overlap=0))


def test_sentences_offsets_y_limite() -> None:
    text = "Una oracion corta. Otra oracion distinta. " * 40
    chunks = list(chunk_sentences(text, max_tokens=60, overlap_sentences=1))
    assert all(chunk.text == text[chunk.start : chunk.end] for chunk in chunks)
    assert chunks[-1].end >= len(text.strip()) - 1
    for chunk in chunks:
        assert estimate_tokens(chunk.text) <= 60 or len(split_sentences(chunk.text)) == 1


def test_sentences_texto_sin_oraciones() -> None:
    chunks = list(chunk_sentences("sin puntuacion final", max_tokens=10))
    assert len(chunks) == 1
    assert chunks[0].text == "sin puntuacion final"
