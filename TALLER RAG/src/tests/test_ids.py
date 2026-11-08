"""El id del chunk debe ser estable y depender de todos los parámetros."""
from __future__ import annotations

from src.rag.chunking import chunk_identity


def test_id_estable() -> None:
    first = chunk_identity("v1", "doc", "fixed", 1000, 384, 150, 1, 0, "hash")
    second = chunk_identity("v1", "doc", "fixed", 1000, 384, 150, 1, 0, "hash")
    assert first == second


def test_id_cambia_con_el_solape() -> None:
    base = chunk_identity("v1", "doc", "fixed", 1000, 384, 150, 1, 0, "hash")
    other = chunk_identity("v1", "doc", "fixed", 1000, 384, 200, 1, 0, "hash")
    assert base != other
