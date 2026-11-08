"""test_pdf_ficha.py
Fecha: 2026-10-31
Descripción: Pruebas de la ficha PDF de la sentencia (P4). Verifican que el
    generador produzca un PDF válido de una sola página, que el nombre del
    archivo sea seguro y que no invente secciones cuando faltan datos.
Estado del arte aplicado: validación del binario con pypdf en memoria y
    fixtures con la forma real de /v1/link.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO

import pytest
from pypdf import PdfReader

from pdf_ficha import construir_ficha, nombre_ficha

LINK = {
    "id": "2195",
    "canon_detectado": "T-293-1997",
    "area": {
        "top3": [
            {"area": "derechos_fundamentales", "prob": 0.4012},
            {"area": "tutela", "prob": 0.3011},
            {"area": "debido_proceso", "prob": 0.1204},
        ],
        "via": "logreg",
        "exploratoria": True,
    },
    "articulos_extraidos": [
        {"numero": "86", "bloque": "acciones", "texto_150chars": "Toda persona..."},
        {
            "numero": "13",
            "bloque": "derechos",
            "texto_150chars": "Todas las personas...",
        },
    ],
    "articulos_recuperados": [
        {
            "numero": "86",
            "score": 149.1889,
            "bloque": "acciones",
            "texto_150chars": "...",
        },
        {
            "numero": "13",
            "score": 101.5,
            "bloque": "derechos",
            "texto_150chars": "...",
        },
    ],
    "sentencias_citadas": [
        {
            "canon": "T-406-1992",
            "id": "123",
            "resuelta": True,
            "ano": 1992,
            "pagerank": 0.0012,
        },
        {"canon": "T-999-1999", "id": "999", "resuelta": False},
    ],
    "resumen": "Providencia valida: 2 articulos CP extraidos y 2 recuperados.",
}


def test_ficha_is_single_page_pdf() -> None:
    """La ficha es un PDF de una página con el canon y el id."""
    datos = construir_ficha(LINK, generado=datetime(2026, 10, 31, 12, 0))
    assert datos.startswith(b"%PDF")
    assert len(datos) > 2000
    lector = PdfReader(BytesIO(datos))
    assert len(lector.pages) == 1
    texto = lector.pages[0].extract_text()
    assert "T-293-1997" in texto
    assert "2195" in texto
    assert "ADR-19" in texto
    assert "2026-10-31" in texto


def test_ficha_omite_secciones_sin_datos() -> None:
    """Sin area ni citas resueltas, el PDF sigue siendo válido y no las inventa."""
    minimo = {
        "id": "1",
        "canon_detectado": "C-001-1992",
        "resumen": "Sin articulos extraidos.",
    }
    datos = construir_ficha(minimo, generado=datetime(2026, 10, 31, 12, 0))
    assert datos.startswith(b"%PDF")
    texto = PdfReader(BytesIO(datos)).pages[0].extract_text()
    assert "Area tematica" not in texto
    assert "Sentencias citadas resueltas" not in texto
    assert "Sin articulos extraidos." in texto


def test_nombre_ficha_sanea_el_canon() -> None:
    """El nombre del archivo usa canon e id y elimina caracteres inseguros."""
    assert nombre_ficha(LINK) == "ficha_T-293-1997_2195.pdf"
    assert "/" not in nombre_ficha({"canon_detectado": "T-293/97", "id": "2195"})


def test_ficha_sin_datos_falla() -> None:
    """Un link vacío no produce un PDF vacío."""
    with pytest.raises(ValueError):
        construir_ficha({})
