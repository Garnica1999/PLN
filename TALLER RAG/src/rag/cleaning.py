"""Variantes de limpieza del texto para el experimento pareado."""
from __future__ import annotations

from src.rag.quality import is_junk
from src.rag.textometry import aggressive_clean, conservative_clean

VARIANTS = ("v0", "v1", "v2", "v3", "v4")

DESCRIPTIONS = {
    "v0": "texto crudo, sin filtro de calidad ni normalizacion",
    "v1": "descarta paginas de error del crawler, texto intacto",
    "v2": "v1 + NFKC, sin HTML ni URLs, espacios colapsados",
    "v3": "v2 + deduplicacion de near-duplicates a nivel de chunk",
    "v4": "v3 + normalizacion agresiva (minusculas, sin acentos ni puntuacion), control negativo",
}


def skips_document(variant: str, text: str) -> bool:
    if variant == "v0":
        return False
    return is_junk(text)


def clean_chunk(variant: str, text: str) -> str:
    if variant in ("v0", "v1"):
        return text
    if variant in ("v2", "v3"):
        return conservative_clean(text)
    if variant == "v4":
        return aggressive_clean(text)
    raise ValueError(f"variante desconocida: {variant}")


def dedupes_chunks(variant: str) -> bool:
    return variant in ("v3", "v4")
