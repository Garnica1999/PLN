"""Utilidades compartidas sobre el corpus."""
from __future__ import annotations

LANGUAGE_MAP = {
    "spa": "es",
    "eng": "en",
    "fra": "fr",
    "deu": "de",
    "por": "pt",
    "ita": "it",
    "nld": "nl",
    "rus": "ru",
    "jpn": "ja",
    "zho": "zh",
    "hun": "hu",
}


def normalize_language(value: object) -> str | None:
    """Reduce el campo language a un código ISO de dos letras cuando sea posible."""
    if not isinstance(value, str):
        return None
    code = value.strip().lower()
    if not code:
        return None
    if code in LANGUAGE_MAP:
        return LANGUAGE_MAP[code]
    base = code.replace("_", "-").split("-")[0]
    if len(base) == 2 and base.isalpha():
        return base
    return None


def normalize_date(value: object) -> str:
    """Deja la fecha en formato AAAA-MM-DD para que los filtros de Milvus sean comparables."""
    if not isinstance(value, str):
        return ""
    text = value.strip()
    return text[:10] if len(text) >= 10 and text[4] == "-" and text[7] == "-" else ""


def truncate_bytes(value: str, max_bytes: int) -> str:
    """Milvus limita VARCHAR en bytes UTF-8, no en caracteres."""
    encoded = (value or "").encode("utf-8")
    if len(encoded) <= max_bytes:
        return value or ""
    return encoded[:max_bytes].decode("utf-8", errors="ignore")
