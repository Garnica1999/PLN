"""test_extraccion.py

Fecha: 2026-09-04
Descripcion: Tests de la Fase 3 (extraccion y segmentacion de los tomos CEV,
    scripts/01_extraer_libros.py):
      (1) parser del TOC con un TOC sintetico (PDF generado en memoria con
          PyMuPDF): entradas con numero de columna derecha, titulos
          multilinea, entradas de varias filas y niveles por sangria x0;
      (2) matcher de encabezados sobre paginas sinteticas (bloque contiguo,
          bloque por filas separadas por blancos y no-match de parrafo);
      (3) segmentos reales del tomo HALLAZGOS ("La Colombia herida" y
          "Acercarnos a la experiencia de las victimas": existen, en orden y
          con texto no vacio) -- se omite con skip si el corpus no existe.

El modulo de produccion se importa con importlib porque su nombre empieza
con digitos (convencion de scripts de experimento).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import fitz
import pytest

RUTA_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_SCRIPT = RUTA_PROYECTO / "scripts" / "01_extraer_libros.py"
RUTA_SEGMENTOS_HALLAZGOS = RUTA_PROYECTO / "data" / "corpus" / "segmentos" / "HALLAZGOS.jsonl"

_spec = importlib.util.spec_from_file_location("extraccion_libros", RUTA_SCRIPT)
ext = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(ext)


# ---------------------------------------------------------------------------
# TOC sintetico
# ---------------------------------------------------------------------------


def _crear_pdf_toc_sintetico() -> Path:
    """Genera un PDF temporal con un TOC sintetico que replica la morfologia
    real observada en los tomos CEV: 'Contenido' como encabezado, entradas con
    titulo a la izquierda (x0 escalonado 70/112.5/155 pt) y numero de pagina
    impreso en la columna derecha (x0 ~480), titulos multilinea con el numero
    en la ultima linea y entradas de varias filas separadas por blancos.

    Returns:
        Ruta del PDF temporal.
    """
    ruta = RUTA_PROYECTO / "data" / "corpus" / "_toc_sintetico_test.pdf"
    doc = fitz.open()
    pagina = doc.new_page()  # A4: 595 x 842 pt
    pagina.insert_text((70, 40), "Contenido", fontsize=14)
    filas: list[tuple[float, float, str]] = [
        # (x0, y, texto)
        (70, 80, "hallazgos 29"),
        (70, 100, "La Colombia herida 31"),
        (112.5, 130, "Acercarnos a la experiencia"),
        (112.5, 150, "de las victimas 35"),
        (155, 170, "Un impacto masivo 35"),
        (70, 200, "Ante el derrumbe"),
        (70, 220, "Introduccion a El libro 44"),
    ]
    for x0, y, texto in filas:
        # el numero de la columna derecha se inserta por separado para
        # replicar el patron real (titulo ... numero a x0 >= 0.78*ancho)
        partes = texto.split()
        numero = None
        if partes[-1].isdigit():
            numero = partes.pop()
        if partes:
            pagina.insert_text((x0, y), " ".join(partes), fontsize=12)
        if numero:
            pagina.insert_text((480, y), numero, fontsize=12)
    doc.save(ruta)
    doc.close()
    return ruta


def test_parser_toc_sintetico() -> None:
    """El parser reconstruye titulos, paginas impresas y niveles (por x0)."""
    ruta_pdf = _crear_pdf_toc_sintetico()
    try:
        entradas = ext.parsear_entradas_toc(ruta_pdf, [1])
        ext.asignar_niveles(entradas)
        assert len(entradas) == 5
        titulos = [e["titulo"] for e in entradas]
        assert titulos == [
            "hallazgos",
            "La Colombia herida",
            "Acercarnos a la experiencia de las victimas",
            "Un impacto masivo",
            "Ante el derrumbe Introduccion a El libro",
        ]
        assert [e["pagina_impresa"] for e in entradas] == [29, 31, 35, 35, 44]
        assert [e["nivel"] for e in entradas] == [1, 1, 2, 3, 1]
        # la entrada de varias filas conserva sus renglones visuales
        assert entradas[-1]["lineas"] == ["Ante el derrumbe", "Introduccion a El libro"]
    finally:
        ruta_pdf.unlink(missing_ok=True)


def test_parser_toc_ignora_encabezado_y_numero_suelto() -> None:
    """'Contenido' no genera entrada y un numero de columna suelto sin titulo
    previo (pie de pagina del TOC) tampoco."""
    ruta_pdf = RUTA_PROYECTO / "data" / "corpus" / "_toc_sintetico_test2.pdf"
    doc = fitz.open()
    pagina = doc.new_page()
    pagina.insert_text((70, 40), "Contenido", fontsize=14)
    pagina.insert_text((70, 80), "hallazgos", fontsize=12)
    pagina.insert_text((480, 80), "29", fontsize=12)
    pagina.insert_text((480, 300), "29", fontsize=12)  # numero suelto
    doc.save(ruta_pdf)
    doc.close()
    try:
        entradas = ext.parsear_entradas_toc(ruta_pdf, [1])
        assert len(entradas) == 1
        assert entradas[0]["titulo"] == "hallazgos"
        assert entradas[0]["pagina_impresa"] == 29
    finally:
        ruta_pdf.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Matcher de encabezados (regresion del fix multi-paso)
# ---------------------------------------------------------------------------


def _tokens(texto: str) -> list[str]:
    """Tokens normalizados de un texto (delegado de norm_es del modulo)."""
    return ext.norm_es(texto).split()


def test_matcher_titulo_contiguo() -> None:
    """Titulo de una o varias lineas pegadas tras un blanco."""
    pagina = "texto anterior\n\nAcercarnos a la experiencia\nde las victimas\n\nEl parrafo sigue"
    tokens = _tokens("Acercarnos a la experiencia de las victimas")
    filas = [tokens]
    assert ext._pagina_contiene_titulo(pagina, tokens, filas, set())


def test_matcher_titulo_por_filas_con_blanco() -> None:
    """Titulo TOC de dos filas separadas por blanco en el cuerpo
    ('La espera' / 'Introduccion a El libro del porvenir')."""
    pagina = "La espera\n\nIntroduccion a El libro del porvenir\n\nas comisiones reciben"
    tokens = _tokens("La espera Introduccion a El libro del porvenir")
    filas = [_tokens("La espera"), _tokens("Introduccion a El libro del porvenir")]
    assert ext._pagina_contiene_titulo(pagina, tokens, filas, set())


def test_matcher_titulo_largo_en_lineas_cortas() -> None:
    """Titulo muy largo compuesto en ~10 lineas cortas centradas (caso de las
    recomendaciones de HALLAZGOS que motivo el limite dinamico de lineas)."""
    lineas_titulo = [
        "Para superar", "la impunidad de", "graves violaciones", "de los derechos",
        "humanos e infracciones", "al DIH, judicializar", "los entramados de",
        "criminalidad organizada", "y corrupcion y mejorar el", "acceso a la justicia local",
    ]
    pagina = "\n".join(lineas_titulo) + "\n\nQue habria que hacer que esto no se repita"
    titulo = " ".join(lineas_titulo)
    tokens = _tokens(titulo)
    assert len(tokens) > 20
    assert ext._pagina_contiene_titulo(pagina, tokens, [tokens], set())


def test_matcher_no_confunde_parrafo() -> None:
    """Un parrafo que menciona el titulo sin blanco previo no es encabezado."""
    pagina = "La guerra en Colombia se configuro desde el campo politico\n" \
             "y alli se condujo la accion de la fuerza publica.\n" \
             "La Colombia herida fue el punto de partida del analisis."
    tokens = _tokens("La Colombia herida")
    assert not ext._pagina_contiene_titulo(pagina, tokens, [tokens], set())


# ---------------------------------------------------------------------------
# Segmentos reales de HALLAZGOS (skip si el corpus no existe)
# ---------------------------------------------------------------------------


def _segmentos_hallazgos() -> list[dict]:
    """Carga los segmentos reales del tomo HALLAZGOS.

    Returns:
        Segmentos del tomo (solo campos de interes).
    """
    with open(RUTA_SEGMENTOS_HALLAZGOS, encoding="utf-8") as fh:
        return [json.loads(linea) for linea in fh]


@pytest.mark.skipif(
    not RUTA_SEGMENTOS_HALLAZGOS.exists(),
    reason="Corpus no extraido: ejecutar scripts/01_extraer_libros.py --libros HALLAZGOS",
)
def test_segmentos_reales_hallazgos() -> None:
    """Los segmentos 'La Colombia herida' y 'Acercarnos a la experiencia de
    las victimas' existen en el corpus real de HALLAZGOS, en orden de pagina
    y con texto no vacio."""
    segmentos = _segmentos_hallazgos()
    assert len(segmentos) > 0

    def buscar(titulo_norm: str) -> list[dict]:
        return [s for s in segmentos if ext.norm_es(s["titulo"]) == titulo_norm]

    colombia = buscar("la colombia herida")
    acercarnos = buscar("acercarnos a la experiencia de las victimas")
    assert colombia, "falta el segmento 'La Colombia herida'"
    assert acercarnos, "falta el segmento 'Acercarnos a la experiencia de las victimas'"
    for s in [*colombia, *acercarnos]:
        assert s["texto"].strip(), f"segmento vacio: {s['id_segmento']}"
        assert s["n_palabras"] > 0
    assert colombia[0]["pagina_inicio"] < acercarnos[0]["pagina_inicio"]


@pytest.mark.skipif(
    not RUTA_SEGMENTOS_HALLAZGOS.exists(),
    reason="Corpus no extraido: ejecutar scripts/01_extraer_libros.py --libros HALLAZGOS",
)
def test_segmentos_reales_sin_huecos_estructurales() -> None:
    """Los segmentos reales tienen id unico, rango de paginas valido y no se
    solapan entre si."""
    segmentos = _segmentos_hallazgos()
    ids = [s["id_segmento"] for s in segmentos]
    assert len(ids) == len(set(ids)), "hay ids de segmento duplicados"
    for s in segmentos:
        assert s["pagina_inicio"] <= s["pagina_fin"]
    rangos = sorted((s["pagina_inicio"], s["pagina_fin"]) for s in segmentos)
    # sin solapamiento entre rangos consecutivos (pueden tocarse sin invadirse)
    for i in range(1, len(rangos)):
        assert rangos[i][0] > rangos[i - 1][1], "rangos de segmentos solapados"
