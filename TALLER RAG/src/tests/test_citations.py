"""Pruebas de citas, abstención y normalización de idioma."""
from __future__ import annotations

from src.rag.corpus import normalize_date, normalize_language
from src.rag.pipeline import ABSTENTION_TEXT, extract_citations


def test_extrae_citas_unicas_y_ordenadas() -> None:
    assert extract_citations("Afirma A [2] y B [1] y C [2].") == [1, 2]


def test_sin_citas() -> None:
    assert extract_citations("Una respuesta sin evidencia numerada.") == []


def test_texto_de_abstencion() -> None:
    assert ABSTENTION_TEXT.lower().startswith("no hay evidencia suficiente")


def test_normaliza_idioma() -> None:
    assert normalize_language("es-co") == "es"
    assert normalize_language("en-US") == "en"
    assert normalize_language("spa") == "es"
    assert normalize_language("EN") == "en"
    assert normalize_language(None) is None
    assert normalize_language("") is None


def test_normaliza_fecha() -> None:
    assert normalize_date("2026-08-16T21:20:04-03:00") == "2026-08-16"
    assert normalize_date("sin fecha") == ""
    assert normalize_date(None) == ""
