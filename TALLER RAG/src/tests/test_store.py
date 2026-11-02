"""Pruebas del esquema de Milvus sin conexión real."""
from __future__ import annotations

import pytest

from src.rag.store import collection_dim, count, ensure_collection


class FakeClient:
    def has_collection(self, name: str) -> bool:
        return True

    def describe_collection(self, name: str) -> dict:
        return {"fields": [{"name": "vector", "params": {"dim": 768}}]}


def test_collection_dim() -> None:
    assert collection_dim(FakeClient(), "x") == 768


def test_ensure_collection_acepta_la_misma_dimension() -> None:
    ensure_collection(FakeClient(), "x", 768)


def test_ensure_collection_rechaza_otra_dimension() -> None:
    with pytest.raises(ValueError):
        ensure_collection(FakeClient(), "x", 1024)


class CountClient:
    def __init__(self) -> None:
        self.flushed = False

    def has_collection(self, name: str) -> bool:
        return True

    def query(self, **kwargs):
        return [{"count(*)": 7}]

    def flush(self, name: str) -> None:
        self.flushed = True


def test_count_no_hace_flush_por_defecto() -> None:
    client = CountClient()
    assert count(client, "x") == 7
    assert client.flushed is False


def test_count_exacto_hace_flush() -> None:
    client = CountClient()
    assert count(client, "x", exact=True) == 7
    assert client.flushed is True
