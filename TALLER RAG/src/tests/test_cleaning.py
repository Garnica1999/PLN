"""Pruebas de limpieza, filtro de ruido y deduplicación por SimHash."""
from __future__ import annotations

from src.rag.cleaning import clean_chunk, skips_document
from src.rag.corpus import truncate_bytes
from src.rag.ingest_variant import SimHashIndex
from src.rag.quality import is_junk
from src.rag.textometry import conservative_clean, simhash64


def test_v2_quita_html_urls_y_espacios() -> None:
    cleaned = conservative_clean("Hola <b>mundo</b> https://ejemplo.com  fin")
    assert "<b>" not in cleaned
    assert "https://" not in cleaned
    assert "  " not in cleaned


def test_v1_no_toca_el_texto() -> None:
    assert clean_chunk("v1", "a  b") == "a  b"


def test_is_junk() -> None:
    assert is_junk("Error 505") is True
    assert is_junk("Una noticia normal") is False


def test_skips_document_por_variante() -> None:
    assert skips_document("v0", "Error 505") is False
    assert skips_document("v1", "Error 505") is True


def test_simhash_index_rechaza_cercano() -> None:
    index = SimHashIndex()
    base = simhash64("una frase suficientemente larga para tener varios shingles de prueba")
    assert index.add_if_new(base) is True
    assert index.add_if_new(base ^ 1) is False
    assert index.add_if_new(simhash64("texto completamente distinto sobre economia y elecciones")) is True


def test_truncate_bytes_respeta_el_limite() -> None:
    value = "á" * 600
    out = truncate_bytes(value, 1024)
    assert len(out.encode("utf-8")) <= 1024
    assert out.startswith("á")
    assert truncate_bytes("abc", 10) == "abc"
