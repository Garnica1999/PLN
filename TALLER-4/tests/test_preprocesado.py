"""test_preprocesado.py

Fecha: 2026-09-04
Descripcion: Tests de la Fase 4 (scripts/02_preprocesar.py):
    (1) normalizacion estructural de entrevistas con regex (ADR-002 y
        arquitectura (c)): entrada sintetica con encabezado de sistema,
        anotaciones [..], anonimizados, etiquetas de hablante y guiones;
        salida esperada EXACTA + contadores por categoria;
    (2) filtro linguistico de tokens (ADR-003): anios 1900-2025 conservados,
        numeros fuera de rango eliminados, categorias canonicas como un solo
        token, stopwords de spaCy y muletillas eliminadas, lematizacion en
        minusculas;
    (3) unidades: es_anio y regla de inclusion de muletillas (umbrales).

No depende del corpus completo (todo el material es sintetico); los tests que
usan spaCy se omiten con skip si el modelo no esta instalado. El modulo de
produccion se importa con importlib porque su nombre empieza con digitos
(convencion de scripts de experimento).
"""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

RUTA_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_SCRIPT = RUTA_PROYECTO / "scripts" / "02_preprocesar.py"

_spec = importlib.util.spec_from_file_location("preprocesar_fase4", RUTA_SCRIPT)
prep = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(prep)


@pytest.fixture(scope="session")
def nlp():
    """Pipeline es_core_news_md sin ner/parser (igual que la produccion)."""
    try:
        return prep.construir_nlp()
    except OSError as exc:  # pragma: no cover - depende del entorno
        pytest.skip(f"modelo es_core_news_md no disponible: {exc}")


# ---------------------------------------------------------------------------
# (1) Normalizacion estructural
# ---------------------------------------------------------------------------

ENTRADA_SINTETICA = (
    "001-VI-00019_(48201)_\n"
    "TEST: Bajo la apariencia [INTERRUP] se vivía \n"
    "un régimen [INC: inaudible] de terror: [risas]\n"
    "ENT1: Estamos en ORGANIZACIÓN PÚBLICA 1 y LUGAR2, FAMILIAR3\n"
    "X: y CONOCIDO1, INSTITUCION EDUCATIVA1 y PARTIDO POLITICO1 y FISCALIA2\n"
    "--------------------------------------------------\n"
    "ENT: adiós -----"
)

SALIDA_ESPERADA = (
    "Bajo la apariencia se vivía un régimen de terror: "
    "Estamos en organizacion_publica y lugar, familiar y conocido, "
    "institucion_educativa y partido_politico y anonimo adiós"
)


def test_normalizacion_entrevista_sintetica_exacta() -> None:
    contadores = prep._nuevos_contadores_normalizacion()
    resultado = prep.normalizar_texto_entrevista(ENTRADA_SINTETICA, contadores)
    assert resultado == SALIDA_ESPERADA


def test_normalizacion_contadores_por_categoria() -> None:
    contadores = prep._nuevos_contadores_normalizacion()
    prep.normalizar_texto_entrevista(ENTRADA_SINTETICA, contadores)
    assert contadores["n_codigos_lider"] == 1
    assert contadores["n_brackets"] == 3
    assert contadores["n_hablantes_linea"] == 4  # TEST, ENT1, X, ENT
    assert contadores["n_guiones"] == 2  # linea separadora y 'adiós -----'
    assert contadores["residuales_caps_digit"] == 0
    assert contadores["anonimizados"] == Counter(
        {
            "organizacion_publica": 1,
            "lugar": 1,
            "familiar": 1,
            "conocido": 1,
            "institucion_educativa": 1,
            "partido_politico": 1,
            "anonimo": 1,
        }
    )


def test_normalizacion_codigo_embebido_con_timecode() -> None:
    texto = "la Policía 001-VI-00011_(48078): 00:45:25 siguió el caso"
    contadores = prep._nuevos_contadores_normalizacion()
    resultado = prep.normalizar_texto_entrevista(texto, contadores)
    assert resultado == "la Policía siguió el caso"
    assert contadores["n_codigos_embebidos"] == 1


def test_normalizacion_linea_audio() -> None:
    texto = "AUDIO 434-VI-00012 (47589)\nENT: buenos días"
    contadores = prep._nuevos_contadores_normalizacion()
    resultado = prep.normalizar_texto_entrevista(texto, contadores)
    assert resultado == "buenos días"
    assert contadores["n_lineas_audio"] == 1


def test_normalizacion_org_con_numero_y_palabra_en_minuscula() -> None:
    # El numero debe consumirse aunque luego siga una palabra en minuscula.
    texto = "la ORGANIZACIÓN PÚBLICA 1 y otras ORGANIZACIÓN PÚBLICA 2 más"
    contadores = prep._nuevos_contadores_normalizacion()
    resultado = prep.normalizar_texto_entrevista(texto, contadores)
    assert resultado == "la organizacion_publica y otras organizacion_publica más"
    assert contadores["anonimizados"]["organizacion_publica"] == 2


def test_normalizacion_org_sin_numero_seguida_de_minuscula_no_se_toca() -> None:
    # Sin numero y seguida de minuscula no es una marca de anonimizacion.
    texto = "la ORGANIZACIÓN PÚBLICA es la que funcionaba en LUGAR1"
    contadores = prep._nuevos_contadores_normalizacion()
    resultado = prep.normalizar_texto_entrevista(texto, contadores)
    assert resultado == "la ORGANIZACIÓN PÚBLICA es la que funcionaba en lugar"
    assert contadores["anonimizados"]["organizacion_publica"] == 0


def test_normalizacion_variantes_ortograficas_y_pegadas() -> None:
    texto = "ORGANIZACION PUBLICA1 y ORGANIZACIÓN PUBLICA2 y ORGANIZACION SOCIAL10"
    contadores = prep._nuevos_contadores_normalizacion()
    resultado = prep.normalizar_texto_entrevista(texto, contadores)
    assert resultado == (
        "organizacion_publica y organizacion_publica y organizacion_social"
    )


def test_normalizacion_etiquetas_aisladas_y_lineas_solo_etiqueta() -> None:
    texto = "bueno, TEST, gracias\nTEST\nENT1\nENT: hola qué tal"
    contadores = prep._nuevos_contadores_normalizacion()
    resultado = prep.normalizar_texto_entrevista(texto, contadores)
    assert resultado == "bueno, gracias hola qué tal"
    assert contadores["n_hablantes_linea"] == 1  # ENT
    assert contadores["n_lineas_solo_etiqueta"] == 2  # TEST y ENT1 en su linea
    assert contadores["n_etiquetas_aisladas"] == 1  # TEST a mitad de linea


def test_normalizacion_conserva_mayusculas_y_anios() -> None:
    # Control (c)6: normalizacion estructural sin lowercase.
    texto = "Hola Colombia desde 1985. ORGANIZACIÓN PÚBLICA 2"
    contadores = prep._nuevos_contadores_normalizacion()
    resultado = prep.normalizar_texto_entrevista(texto, contadores)
    assert resultado == "Hola Colombia desde 1985. organizacion_publica"


# ---------------------------------------------------------------------------
# (2) Filtro linguistico de tokens (ADR-003)
# ---------------------------------------------------------------------------


def test_filtro_anios_categorias_y_muletillas(nlp) -> None:
    texto = (
        "En 1991 llegaron 1899 y 2026 cosas a organizacion_publica, "
        "eh pues mirá Colombia abc12"
    )
    crudos, lemas, conteos = prep.filtrar_doc(nlp(texto), {"eh", "mirá"})
    assert crudos == [
        "en", "1991", "llegaron", "1899", "y", "2026", "cosas", "a",
        "organizacion_publica", ",", "eh", "pues", "mirá", "colombia", "abc12",
    ]
    assert lemas == ["1991", "llegar", "cosa", "organizacion_publica", "colombia"]
    assert conteos["raw"] == 15
    assert conteos["lemas"] == 5
    assert conteos["num_no_anio"] == 2  # 1899 y 2026
    assert conteos["no_alfa"] == 2  # coma y abc12


def test_filtro_anios_en_los_limites(nlp) -> None:
    texto = "En 1900 y 2025 pasaron cosas, en 1899 y 2026 no"
    _, lemas, conteos = prep.filtrar_doc(nlp(texto), set())
    assert lemas == ["1900", "2025", "pasar", "cosa"]
    assert conteos["num_no_anio"] == 2  # 1899 y 2026


def test_filtro_numeros_sueltos_fuera_de_rango(nlp) -> None:
    texto = "El 123 y el 1899 y el 1991 y el 9999"
    _, lemas, conteos = prep.filtrar_doc(nlp(texto), set())
    assert lemas == ["1991"]  # unico anio dentro de [1900, 2025]
    assert conteos["num_no_anio"] == 3  # 123, 1899, 9999


def test_filtro_stopwords_spacy_y_muletilla_desde_archivo(nlp) -> None:
    texto = "pues bueno entonces digamos eh mmm que esto es una prueba"
    _, lemas, conteos = prep.filtrar_doc(nlp(texto), {"eh", "mmm", "digamos"})
    # pues/bueno/entonces/que/esto/es/una son stopwords de spaCy; eh/mmm/
    # digamos son muletillas del archivo extra.
    assert lemas == ["prueba"]
    assert conteos["stopword"] == 7
    assert conteos["muletilla"] == 3


# ---------------------------------------------------------------------------
# (3) Unidades
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "esperado"),
    [
        ("1900", True),
        ("1991", True),
        ("2025", True),
        ("1899", False),
        ("2026", False),
        ("9999", False),
        ("123", False),
        ("abc", False),
        ("", False),
    ],
)
def test_es_anio(token: str, esperado: bool) -> None:
    assert prep.es_anio(token) is esperado


def test_decision_muletilla_umbrales() -> None:
    # Evidencia: docs>=5 y total>=20.
    decision, motivo = prep._decision_muletilla("eh", 176, 10, en_spacy=False)
    assert decision == "incluir"
    assert "evidencia" in motivo
    # Nucleo de interjecciones con presencia minima.
    decision, _ = prep._decision_muletilla("mmm", 12, 6, en_spacy=False)
    assert decision == "incluir"
    # Fuera de umbrales.
    decision, _ = prep._decision_muletilla("mirá", 6, 2, en_spacy=False)
    assert decision == "excluir"
    # Ya cubierta por spaCy.
    decision, _ = prep._decision_muletilla("pues", 100, 30, en_spacy=True)
    assert decision == "ya_en_spacy"


def test_categorias_canonicas_definidas() -> None:
    assert prep.CATEGORIAS_CANONICAS == {
        "organizacion_publica",
        "organizacion_social",
        "lugar",
        "familiar",
        "conocido",
        "institucion_educativa",
        "partido_politico",
        "anonimo",
    }
