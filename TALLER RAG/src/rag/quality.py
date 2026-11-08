"""Filtros de calidad del corpus.

El crawler dejó páginas de error y avisos de cuota repetidos en miles de documentos.
Esos textos dominan la recuperación porque son casi idénticos entre sí.
"""
from __future__ import annotations

BOILERPLATE_MARKERS = (
    "haz excedido el máximo de peticiones",
    "error 505",
    "access denied",
    "403 forbidden",
    "404 not found",
    "enable javascript",
    "just a moment",
)


def is_junk(text: str) -> bool:
    """Marca documentos que son páginas de error o avisos del crawler."""
    lower = (text or "").lower()
    return any(marker in lower for marker in BOILERPLATE_MARKERS)
