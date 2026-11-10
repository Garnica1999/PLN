"""Extracción y normalización de entidades para el grafo de conocimiento."""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

try:  # spaCy es opcional: la API puede correr sin él y usar el fallback regex.
    import spacy
except ImportError:  # pragma: no cover
    spacy = None

TYPE_MAP = {
    "PER": "PER",
    "PERSON": "PER",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "FAC": "LOC",
    "EVENT": "EVENT",
    "NORP": "ORG",
    "MISC": "MISC",
}
ARTICLES = {"el", "la", "los", "las", "un", "una", "unos", "unas", "the", "a", "an"}
STOPWORDS = {
    "senor", "senora", "presidente", "ministro", "director", "gobierno",
    "pais", "ciudad", "departamento", "municipio", "ano", "anos", "hoy",
}
MIN_LENGTH = 3
MAX_TOKENS = 6


def normalize(name: str) -> str:
    """Reduce un nombre a una clave estable: sin tildes, minúsculas y sin artículos."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = [token for token in text.split() if token not in ARTICLES]
    return " ".join(tokens).strip()


def entity_type(label: str) -> str | None:
    """Traduce la etiqueta de spaCy a un tipo canónico del grafo."""
    return TYPE_MAP.get(label)


@lru_cache(maxsize=2)
def get_nlp(language: str):
    """Carga perezosa del modelo de spaCy por familia de idioma."""
    if spacy is None:
        return None
    model = "es_core_news_lg" if language.startswith("es") else "en_core_web_lg"
    try:
        return spacy.load(model, disable=["parser", "lemmatizer"])
    except OSError:
        return None


def language_family(language: str | None) -> str | None:
    """Devuelve es o en para los idiomas cubiertos; None para el resto."""
    value = (language or "").lower()
    if value.startswith("es"):
        return "es"
    if value.startswith("en"):
        return "en"
    return None


PROPER = re.compile(
    r"\b[A-ZÁÉÍÓÚÑ][\wáéíóúñ.\-]*"
    r"(?:\s+(?:de|del|la|las|los|y|of|the|&)\s+[A-ZÁÉÍÓÚÑ][\wáéíóúñ.\-]*"
    r"|\s+[A-ZÁÉÍÓÚÑ][\wáéíóúñ.\-]*){0,4}"
)


def _accept(key: str) -> bool:
    return (
        MIN_LENGTH <= len(key)
        and len(key.split()) <= MAX_TOKENS
        and not key.isdigit()
        and key not in STOPWORDS
    )


def extract_simple(text: str) -> dict[str, dict]:
    """Fallback sin spaCy: secuencias de mayúsculas como nombres propios."""
    found: dict[str, dict] = {}
    for match in PROPER.finditer(text):
        name = match.group(0).strip()
        key = normalize(name)
        if not _accept(key):
            continue
        current = found.get(key)
        if current is None:
            found[key] = {"key": key, "name": name, "type": "MISC", "count": 1}
        else:
            current["count"] += 1
    return found


def extract(text: str, language: str | None = "es") -> dict[str, dict]:
    """Extrae entidades de un texto y las agrupa por clave normalizada."""
    family = language_family(language)
    if not family or not text:
        return {}
    nlp = get_nlp(family)
    if nlp is None:
        return extract_simple(text)
    doc = nlp(text[:2000])
    found: dict[str, dict] = {}
    for entity in doc.ents:
        kind = entity_type(entity.label_)
        if not kind:
            continue
        key = normalize(entity.text)
        if not _accept(key):
            continue
        current = found.get(key)
        if current is None:
            found[key] = {"key": key, "name": entity.text.strip(), "type": kind, "count": 1}
        else:
            current["count"] += 1
    return found


def extract_query(text: str) -> list[dict]:
    """Extrae entidades de una consulta probando español e inglés."""
    merged: dict[str, dict] = {}
    for family in ("es", "en"):
        for key, value in extract(text, family).items():
            if key not in merged:
                merged[key] = value
    return list(merged.values())


def mentioned(name: str, context: str) -> bool:
    """Comprueba si una entidad normalizada aparece en un contexto."""
    return normalize(name) in normalize(context)
