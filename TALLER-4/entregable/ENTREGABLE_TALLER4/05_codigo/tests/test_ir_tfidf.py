"""test_ir_tfidf.py

Fecha: 2026-09-04
Descripcion: Tests de la Fase 6 (experimentos/experimento_1_tfidf_base.py),
modelo de IR TF-IDF del TALLER-4 (entrevistas = queries, segmentos de libro
= coleccion, ADR-005). Cubren:
  (1) sanity del pipeline sobre datos sinteticos: fit del TfidfVectorizer
      sobre la coleccion y transform de queries (vocabulario compartido);
  (2) ranking top-k correcto con similitud coseno verificable a mano
      (query identica a un documento -> score 1.0; doc con 1 de 2 terminos
      -> coseno = sqrt(2)/2 cuando df = n_docs, idf = 1);
  (3) parada en consumo: elimina muletillas ("pues", "o sea") y ruido
      tipografico ("cion"), y conserva anos 1900-2025;
  (4) correccion de lemas en consumo: atreveir -> atrever.

No depende del corpus completo (todo el material es sintetico). El modulo de
produccion se importa con importlib porque su nombre no es un identificador
valido (convencion de scripts de experimento, patron de test_preprocesado).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

RUTA_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_SCRIPT = RUTA_PROYECTO / "experimentos" / "experimento_1_tfidf_base.py"

_spec = importlib.util.spec_from_file_location("experimento_1_tfidf", RUTA_SCRIPT)
exp = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(exp)


# ---------------------------------------------------------------------------
# Fixtures sinteticos (mismo pipeline que produccion, sin datos reales)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vectorizador_config() -> dict:
    """Config base del vectorizador (sin min_df estricto para juguetes)."""
    config = {
        "analyzer": "word",
        "token_pattern": r"(?u)\b\w+\b",
        "lowercase": True,
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "norm": "l2",
        "use_idf": True,
        "smooth_idf": True,
        "min_df": 1,
        "max_df": 1.0,
        "dtype": np.float32,
    }
    return config


@pytest.fixture(scope="module")
def parada_con_muletillas() -> set[str]:
    """Parada de la variante base (spaCy + muletillas del config real)."""
    return exp.construir_parada(incluir_muletillas=True)


@pytest.fixture(scope="module")
def correcciones() -> dict[str, str]:
    """Correcciones de lemas del config real (atreveir -> atrever, etc.)."""
    return exp.cargar_correcciones()


# ---------------------------------------------------------------------------
# (1) Sanity del pipeline: fit/transform con vocabulario compartido
# ---------------------------------------------------------------------------


def test_fit_transform_vocabulario_compartido(vectorizador_config):
    """El vectorizador fit en la coleccion transforma queries compartidas."""
    coleccion = ["gato perro", "pajaro arbol"]
    query = ["gato perro"]
    vectorizador = exp.construir_vectorizador(vectorizador_config)
    x_c = vectorizador.fit_transform(coleccion)
    x_q = vectorizador.transform(query)
    assert x_c.shape == (2, len(vectorizador.vocabulary_))
    assert x_q.shape == (1, len(vectorizador.vocabulary_))
    # Con ngram_range=(1,2): unigramas + bigrama 'gato perro' (3 features).
    assert "gato" in vectorizador.vocabulary_
    assert "perro" in vectorizador.vocabulary_
    assert "gato perro" in vectorizador.vocabulary_
    assert x_q.nnz == 3


def test_coseno_y_topk_verificables_a_mano(vectorizador_config):
    """Top-k exacto con similitud coseno calculable a mano (idf = 1).

    Coleccion ["gato perro", "gato", "perro"]: 'gato' y 'perro' tienen
    df = 2 = N-1, luego su idf suave coincide y se cancela en el coseno.
    La query "gato perro" da coseno 1.0 con "gato perro" (vector identico) y
    coseno idf^2/(sqrt(2)*idf*idf) = sqrt(2)/2 con "gato".
    """
    config_unigramas = {**vectorizador_config, "ngram_range": (1, 1)}
    coleccion = ["gato perro", "gato", "perro"]
    query = ["gato perro"]
    vectorizador = exp.construir_vectorizador(config_unigramas)
    x_c = vectorizador.fit_transform(coleccion)
    x_q = vectorizador.transform(query)
    sim = exp.similitud_coseno_batches(x_q, x_c, batch_size=1)
    ranking = exp.construir_ranking(sim, ["q1"], ["doc0", "doc1", "doc2"], top_k=2)
    assert list(ranking["rango"]) == [1, 2]
    assert list(ranking["id_segmento"]) == ["doc0", "doc1"]
    assert np.allclose(ranking["score"].to_numpy(float), [1.0, np.sqrt(2) / 2])


def test_query_sin_match_no_genera_filas(vectorizador_config):
    """Una query sin terminos compartidos no produce filas de ranking."""
    coleccion = ["gato perro", "pajaro arbol"]
    query = ["pez rio"]
    vectorizador = exp.construir_vectorizador(vectorizador_config)
    x_c = vectorizador.fit_transform(coleccion)
    x_q = vectorizador.transform(query)
    sim = exp.similitud_coseno_batches(x_q, x_c, batch_size=2)
    ranking = exp.construir_ranking(sim, ["q1"], ["doc0", "doc1"], top_k=5)
    assert ranking.empty


def test_documento_vacio_manejado(vectorizador_config):
    """Un documento vacio en las queries produce fila de ceros, no error."""
    coleccion = ["gato perro"]
    query = [""]
    vectorizador = exp.construir_vectorizador(vectorizador_config)
    x_c = vectorizador.fit_transform(coleccion)
    x_q = vectorizador.transform(query)
    assert x_q.shape[0] == 1 and x_q.nnz == 0
    sim = exp.similitud_coseno_batches(x_q, x_c, batch_size=1)
    ranking = exp.construir_ranking(sim, ["q_vacia"], ["doc0"], top_k=5)
    assert ranking.empty


# ---------------------------------------------------------------------------
# (2) Parada en consumo: muletillas, ruido tipografico y anos
# ---------------------------------------------------------------------------


def test_parada_elimina_muletillas_pues_y_o_sea(parada_con_muletillas):
    """'pues' y 'o sea' (osea) caen en la parada en consumo de la variante base."""
    limpio = exp.limpiar_linea(
        "hablar pues osea digamos pueblo", parada_con_muletillas, {}
    )
    assert limpio == "hablar pueblo"


def test_correccion_atreveir_a_atrever(parada_con_muletillas, correcciones):
    """El lema corrupto 'atreveir' se corrige a 'atrever' en consumo."""
    limpio = exp.limpiar_linea("atreveír atrever", parada_con_muletillas, correcciones)
    assert limpio == "atrever atrever"
    # Sin correcciones (variante C) el token anomalo sobrevive.
    sin_correccion = exp.limpiar_linea("atreveír atrever", parada_con_muletillas, {})
    assert sin_correccion == "atreveír atrever"


def test_anios_1900_2025_conservados(parada_con_muletillas):
    """Los anos del conflicto no caen en la parada (token_pattern los conserva)."""
    limpio = exp.limpiar_linea("1991 1985 2016 conflicto", parada_con_muletillas, {})
    assert limpio == "1991 1985 2016 conflicto"


def test_ruido_tipografico_eliminado():
    """Fragmentos de palabras del PDF ('cion', 'ibid.') no entran al vocabulario."""
    parada = exp.construir_parada(incluir_muletillas=True)
    limpio = exp.limpiar_linea("violencia ción ibíd project", parada, {})
    assert limpio == "violencia"


def test_variante_b_parada_solo_spacy():
    """La variante B excluye las muletillas pero conserva ruido y spaCy.

    'pues' lo cubre spaCy; se verifica con muletillas del habla que spaCy no
    tiene ('osea', 'vea').
    """
    parada_b = exp.construir_parada(incluir_muletillas=False)
    assert "osea" not in parada_b  # muletilla del habla, ausente en spaCy
    assert "vea" not in parada_b  # muletilla del habla, ausente en spaCy
    assert "ción" in parada_b  # ruido tipografico (siempre excluido)
    assert "ser" in parada_b  # lema funcional cubierto por spaCy
    limpio = exp.limpiar_linea("hablar osea pueblo", parada_b, {})
    assert limpio == "hablar osea pueblo"


# ---------------------------------------------------------------------------
# (3) Muestra de evaluacion: determinismo y estructura
# ---------------------------------------------------------------------------


def test_seleccion_muestra_determinista():
    """La seleccion de la muestra es reproducible con la misma semilla."""
    ranking = exp.construir_ranking(
        np.array([[1.0, 0.5, 0.2, 0.0], [0.9, 0.8, 0.1, 0.0]], dtype=np.float32),
        ["q_a", "q_b"],
        ["seg_0", "seg_1", "seg_2", "seg_3"],
        top_k=5,
    )
    ranking["tomo"] = ["HALLAZGOS", "HALLAZGOS", "NNA", "NNA", "HALLAZGOS", "NNA"]
    primera = exp.seleccionar_muestra_entrevistas(ranking, semilla=7, objetivo=1)
    segunda = exp.seleccionar_muestra_entrevistas(ranking, semilla=7, objetivo=1)
    assert primera == segunda
    assert primera == ["q_a"]  # unica candidata con >= 3 matches


def test_fragmento_texto_colapsa_layout():
    """El fragmento elimina el encabezado repetido y colapsa el layout."""
    texto = "Introducción\n               Haber estado en esos lugares\nme hizo decir."
    fragmento = exp.fragmento_texto(texto, titulo="Introducción")
    assert fragmento.startswith("Haber estado en esos lugares")
    assert "  " not in fragmento and "\n" not in fragmento
    assert len(fragmento) <= exp.MAX_CHARS_FRAGMENTO
