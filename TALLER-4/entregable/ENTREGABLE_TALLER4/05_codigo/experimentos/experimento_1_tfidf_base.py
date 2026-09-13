"""experimento_1_tfidf_base.py

Fecha: 2026-09-04
Descripcion: Modelo de Recuperacion de Informacion TF-IDF del TALLER-4
(Fase 6, punto 3 del taller). Direccion de matching (ADR-005): cada
entrevista (2.486) es una query y la coleccion son los segmentos de libro
(917). Fit del TfidfVectorizer sobre los lemas de los segmentos de libro y
transform sobre los lemas de las entrevistas; similitud coseno por batches;
ranking top-5 por query.

Estado del arte aplicado: TF-IDF con saturacion de tf (Salton y Buckley,
1988), norma l2 (Manning et al., 2008), ngram_range=(1,2) por colocaciones
de dominio del corpus (ADR-005). Config congelada en el EDA (Fase 5) en
config/tfidf_config.json (min_df=2, max_df=0.85, sublinear_tf, norm=l2).

Criterios heredados del EDA (scripts/03_eda.py), replicados aqui y no
importados (el modulo 03_eda.py tiene nombre con digito inicial, solo es
importable via importlib, y arrastra matplotlib/wordcloud):
  - Parada EN CONSUMO sobre lemas: spacy.lang.es.STOP_WORDS +
    config/stopwords_habla.txt (51 muletillas) + ruido tipografico de libros
    (42 tokens, lista congelada en config/tfidf_config.json, verificada
    identica a RUIDO_LIBROS de 03_eda.py). La Fase 4 filtro por superficie,
    por eso los lemas residuales funcionales (ser, estar, haber...) se
    eliminan aqui sobre el lema (equivalente a parada_en_consumo()).
  - Correcciones de lemas anomalos del lematizador en consumo:
    config/lemas_correcciones.json (62 mapas, p. ej. atreveir -> atrever).
  - Anos 1900-2025: ya filtrados en la Fase 4; aqui no se aplica filtro
    adicional y el token_pattern los conserva.

Variantes comparativas (arquitectura (g), experimentos 2 y 3):
  A: ngram_range=(1,1) (sin bigramas).
  B: sin parada de muletillas (solo spaCy STOP_WORDS; se conservan las
     correcciones y el ruido tipografico).
  C: sin correcciones de lemas (parada completa intacta).
El experimento BASE (default) es el que genera la muestra de evaluacion.

Uso:
  /home/carlos/venv/bin/python experimentos/experimento_1_tfidf_base.py
  /home/carlos/venv/bin/python experimentos/experimento_1_tfidf_base.py \
      --variante A --batch-size 512
El ranking del base se escribe solo con --force si el CSV ya existe; las
variantes escriben siempre resultados/ranking_top5_<variante>.csv.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.es import STOP_WORDS

# ===========================================================================
# SECCION 0: configuracion y constantes
# ===========================================================================

RUTA_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_PREPROCESADO = RUTA_PROYECTO / "data" / "corpus" / "preprocesado"
RUTA_CORPUS = RUTA_PROYECTO / "data" / "corpus"
RUTA_CONFIG = RUTA_PROYECTO / "config"
RUTA_RESULTADOS = RUTA_PROYECTO / "resultados"
RUTA_METRICAS = RUTA_RESULTADOS / "metricas.json"

# Queries por tomo del top-1 para la muestra de evaluacion y orden canonico
# de tomos del Informe (plan/PLAN.md, Fase 3).
TOMOS_ORDEN = (
    "HALLAZGOS",
    "CONVOCATORIA",
    "NARRATIVA",
    "SUFRIR",
    "TESTIMONIAL",
    "CUERPO",
    "NNA",
    "VIOLACIONES",
    "ETNICO",
    "FUERA",
)
N_POR_TOMO_MUESTRA = 3  # entrevistas por tomo en la muestra de evaluacion
TAMANO_MUESTRA = 30  # objetivo de entrevistas en evaluacion_muestra.csv
SEMILLA_MUESTRA = 42  # semilla fija y documentada del muestreo (determinista)
SEMILLA_EJEMPLOS = 42  # semilla de los ejemplos top-1 impresos en consola
MAX_CHARS_FRAGMENTO = 500  # caracteres del fragmento para el etiquetado

RANKING_COLUMNAS = [
    "id_doc",
    "rango",
    "id_segmento",
    "score",
    "tomo",
    "titulo_segmento",
]
VARIANTES_VALIDAS = ("base", "A", "B", "C")

# ---------------------------------------------------------------------------
# Carga de configuracion congelada (replica el criterio de scripts/03_eda.py)
# ---------------------------------------------------------------------------


def cargar_correcciones() -> dict[str, str]:
    """Carga el mapa de correcciones de lemas de config/lemas_correcciones.json.

    Returns:
        Mapa lema anomalo -> lema correcto (claves con la forma del corpus).

    Raises:
        FileNotFoundError: si la config congelada no existe.
    """
    ruta = RUTA_CONFIG / "lemas_correcciones.json"
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    correcciones = datos.get("correcciones", {})
    if not isinstance(correcciones, dict):
        raise TypeError(f"{ruta.name}: 'correcciones' debe ser un objeto")
    return {str(k): str(v) for k, v in correcciones.items()}


def cargar_muletillas_config() -> set[str]:
    """Carga los tokens de config/stopwords_habla.txt (ignora '#' y vacias).

    Returns:
        Conjunto de muletillas congeladas en el EDA (51 tokens).
    """
    ruta = RUTA_CONFIG / "stopwords_habla.txt"
    return {
        linea.strip()
        for linea in ruta.read_text(encoding="utf-8").splitlines()
        if linea.strip() and not linea.startswith("#")
    }


def cargar_ruido_libros() -> set[str]:
    """Carga el ruido tipografico de libros excluido en el EDA.

    Los 42 tokens (fragmentos de palabras partidas por el guion del PDF y
    ruido bibliografico como 'ibid.') se leen de config/tfidf_config.json
    (lista congelada, verificada identica a RUIDO_LIBROS de 03_eda.py). La
    config recomienda pasarlos a parada en la Fase 6 si superan min_df=2 en
    la coleccion; aqui se aplican como parada (equivalente a stop_words).

    Returns:
        Conjunto de tokens de ruido a excluir del vocabulario.
    """
    ruta = RUTA_CONFIG / "tfidf_config.json"
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    ruido = datos.get("ruido_tipografico_libros_excluido_en_eda", [])
    if not isinstance(ruido, list):
        raise TypeError(f"{ruta.name}: 'ruido_tipografico_libros_excluido_en_eda' debe ser lista")
    return {str(token) for token in ruido}


def cargar_config_vectorizador() -> dict[str, Any]:
    """Carga los parametros del vectorizador de config/tfidf_config.json.

    Returns:
        Dict con los kwargs de TfidfVectorizer congelados en el EDA (Fase 5).
    """
    ruta = RUTA_CONFIG / "tfidf_config.json"
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    config = dict(datos.get("vectorizador", {}))
    config["ngram_range"] = tuple(int(v) for v in config["ngram_range"])
    return config


def construir_parada(incluir_muletillas: bool = True) -> set[str]:
    """Construye la parada en consumo sobre lemas para una variante.

    Parada = spacy.lang.es.STOP_WORDS + (muletillas si incluir_muletillas)
    + ruido tipografico de libros (siempre: no es muletilla y no debe entrar
    al vocabulario de la coleccion, criterio del EDA).

    Args:
        incluir_muletillas: False para la variante B (parada solo spaCy).

    Returns:
        Conjunto de tokens que se eliminan del texto lematizado.
    """
    parada = set(STOP_WORDS) | cargar_ruido_libros()
    if incluir_muletillas:
        parada |= cargar_muletillas_config()
    return parada


# ---------------------------------------------------------------------------
# Preprocesado en consumo (parada + correcciones sobre los lemas)
# ---------------------------------------------------------------------------


def limpiar_linea(linea: str, parada: set[str], correcciones: dict[str, str]) -> str:
    """Limpia una linea de lemas aplicando correcciones y parada en consumo.

    Replica la semantica de corregir_y_filtrar() de scripts/03_eda.py:
    por token, primero la correccion (lema -> lema) y, sobre el resultado,
    la parada. Los archivos *_lemas.txt no se reescriben (insumo congelado).

    Args:
        linea: una linea de *_lemas.txt (lemas separados por espacio).
        parada: conjunto de tokens a eliminar.
        correcciones: mapa lema anomalo -> lema correcto.

    Returns:
        Texto limpio con los lemas de contenido separados por espacio.
    """
    salida: list[str] = []
    for token in linea.split():
        if correcciones:
            token = correcciones.get(token, token)
        if token not in parada:
            salida.append(token)
    return " ".join(salida)


def preprocesar_corpus(
    lineas: list[str], parada: set[str], correcciones: dict[str, str]
) -> tuple[list[str], list[int]]:
    """Aplica limpiar_linea a todas las lineas de un corpus.

    Args:
        lineas: documentos lematizados (1 doc por linea).
        parada: conjunto de tokens a eliminar.
        correcciones: mapa de correcciones de lemas.

    Returns:
        Tupla (textos_limpios, n_tokens_por_doc_limpio). Los documentos que
        quedan vacios se conservan como string vacio (el vectorizador produce
        una fila de ceros y el ranking los omite).
    """
    textos: list[str] = []
    conteos: list[int] = []
    for linea in lineas:
        limpio = limpiar_linea(linea, parada, correcciones)
        textos.append(limpio)
        conteos.append(len(limpio.split()) if limpio else 0)
    return textos, conteos


# ---------------------------------------------------------------------------
# Carga de lemas y metadatos (alineados por linea, validacion de orden)
# ---------------------------------------------------------------------------


def cargar_lemas_y_metadatos(corpus: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Carga *_lemas.txt y *_docs.json validando su alineacion por linea.

    Args:
        corpus: "libros" (coleccion) o "entrevistas" (queries).

    Returns:
        Tupla (lineas_de_lemas, metadatos de *_docs.json en el mismo orden).

    Raises:
        RuntimeError: si el numero de lineas no coincide con los metadatos.
    """
    ruta_lemas = RUTA_PREPROCESADO / f"{corpus}_lemas.txt"
    ruta_docs = RUTA_PREPROCESADO / f"{corpus}_docs.json"
    lineas = [linea.rstrip("\n") for linea in ruta_lemas.open(encoding="utf-8")]
    metadatos = json.loads(ruta_docs.read_text(encoding="utf-8"))["docs"]
    if len(lineas) != len(metadatos):
        raise RuntimeError(
            f"{corpus}: {len(lineas)} lineas de lemas vs {len(metadatos)} "
            "registros de metadatos; alineacion rota"
        )
    return lineas, metadatos


# ---------------------------------------------------------------------------
# Modelo: vectorizador, similitud coseno por batches y ranking
# ---------------------------------------------------------------------------


def construir_vectorizador(config: dict[str, Any]) -> TfidfVectorizer:
    """Construye el TfidfVectorizer con la config congelada (dtype float32).

    La parada y las correcciones se aplican ANTES en el texto (limpiar_linea),
    por lo que el vectorizador no recibe stop_words: el texto de entrada ya
    solo contiene lemas de contenido. Es equivalente a pasarlos por
    stop_words (sklearn los excluye de la formacion de n-gramas en fit y en
    transform) y mantiene el vectorizador agnostico a la variante.

    Args:
        config: kwargs de config/tfidf_config.json["vectorizador"].

    Returns:
        TfidfVectorizer listo para fit sobre la coleccion.
    """
    parametros = dict(config)
    parametros.setdefault("dtype", np.float32)
    return TfidfVectorizer(**parametros)


def similitud_coseno_batches(
    x_queries: sparse.csr_matrix,
    x_coleccion: sparse.csr_matrix,
    batch_size: int = 512,
) -> np.ndarray:
    """Similitud coseno query-coleccion por batches (producto sparse).

    Los vectores TF-IDF estan normalizados con norm=l2, por lo que el
    producto matricial equivale al coseno. Las queries se procesan en batches
    para acotar la memoria; la salida (2.486 x 917 en float32) es manejable.

    Args:
        x_queries: matriz TF-IDF de las queries (n_queries x n_features).
        x_coleccion: matriz TF-IDF de la coleccion (n_docs x n_features).
        batch_size: numero de queries por batch.

    Returns:
        Matriz densa float32 (n_queries x n_docs) de similitudes coseno.
    """
    coleccion_t = x_coleccion.T.astype(np.float32)
    bloques: list[np.ndarray] = []
    for inicio in range(0, x_queries.shape[0], batch_size):
        bloque = x_queries[inicio : inicio + batch_size].astype(np.float32)
        sim = (bloque @ coleccion_t).toarray()
        bloques.append(sim)
    return np.vstack(bloques)


def construir_ranking(
    similitudes: np.ndarray,
    ids_query: list[str],
    ids_coleccion: list[str],
    top_k: int = 5,
) -> pd.DataFrame:
    """Construye el ranking top-k por query desde la matriz de similitudes.

    Se conservan unicamente los pares con score > 0 (sin vocabulario
    compartido no hay match); una query sin ningun match no genera filas
    (queda reportada como no cubierta). El desempate es estable por posicion
    de la coleccion (argpartition no usado: orden estable con argsort).

    Args:
        similitudes: matriz (n_queries x n_docs) de similitudes coseno.
        ids_query: id_doc de cada query, en el orden de las filas.
        ids_coleccion: id_segmento de cada documento, en el orden de columnas.
        top_k: profundidad del ranking (5 para el entregable).

    Returns:
        DataFrame con columnas [id_doc, rango, id_segmento, score].
    """
    filas: list[dict[str, Any]] = []
    for i_query, id_doc in enumerate(ids_query):
        scores = similitudes[i_query]
        # Solo posiciones con score > 0, ordenadas de mayor a menor (estable).
        orden = np.argsort(-scores, kind="stable")
        for rango, posicion in enumerate(orden, start=1):
            if rango > top_k:
                break
            score = float(scores[posicion])
            if score <= 0.0:
                break
            filas.append(
                {
                    "id_doc": id_doc,
                    "rango": rango,
                    "id_segmento": ids_coleccion[int(posicion)],
                    "score": round(score, 6),
                }
            )
    ranking = pd.DataFrame(filas, columns=["id_doc", "rango", "id_segmento", "score"])
    ranking["id_segmento"] = ranking["id_segmento"].astype(str)
    return ranking


def enriquecer_ranking(
    ranking: pd.DataFrame,
    metadatos_segmentos: list[dict[str, Any]],
    ids_segmentos: list[str],
) -> pd.DataFrame:
    """Agrega tomo y titulo del segmento al ranking.

    Args:
        ranking: DataFrame de construir_ranking (columnas basicas).
        metadatos_segmentos: registros de libros_docs.json en orden.
        ids_segmentos: id_segmento en el mismo orden.

    Returns:
        Ranking con las columnas finales
        [id_doc, rango, id_segmento, score, tomo, titulo_segmento].
    """
    ids_validos = set(ids_segmentos)
    mapa = {
        registro["id"]: (registro["tomo"], registro["titulo"])
        for registro in metadatos_segmentos
        if registro["id"] in ids_validos
    }
    ranking = ranking.copy()
    tomos: list[str] = []
    titulos: list[str] = []
    for id_segmento in ranking["id_segmento"]:
        tomo, titulo = mapa.get(id_segmento, ("", ""))
        tomos.append(tomo)
        titulos.append(titulo)
    ranking["tomo"] = tomos
    ranking["titulo_segmento"] = titulos
    return ranking[RANKING_COLUMNAS]


# ---------------------------------------------------------------------------
# Evaluacion cualitativa: muestra determinista para etiquetado humano
# ---------------------------------------------------------------------------


def _normalizar_encabezado(texto: str) -> str:
    """Normaliza un titulo/encabezado para comparacion sin acentos."""
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = texto.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9 ]+", " ", texto).strip()


def fragmento_texto(texto: str, titulo: str | None = None) -> str:
    """Primer fragmento legible (~MAX_CHARS_FRAGMENTO chars) de un texto.

    Elimina la primera linea si repite el titulo del segmento (el extractor
    de la Fase 3 dejo el encabezado al inicio del texto) y colapsa el layout
    del PDF (saltos de linea y sangrias) a un unico espacio.

    Args:
        texto: texto crudo del segmento o de la entrevista normalizada.
        titulo: titulo del segmento para descartar el encabezado repetido.

    Returns:
        Fragmento de apertura del texto, truncado si es mas largo.
    """
    lineas = [linea.strip() for linea in texto.splitlines() if linea.strip()]
    if titulo and lineas:
        objetivo = _normalizar_encabezado(titulo)
        primera = _normalizar_encabezado(lineas[0])
        if objetivo and primera == objetivo:
            lineas = lineas[1:]
    fragmento = " ".join(lineas)
    if len(fragmento) > MAX_CHARS_FRAGMENTO:
        fragmento = fragmento[:MAX_CHARS_FRAGMENTO]
    return fragmento


def seleccionar_muestra_entrevistas(
    ranking_top5: pd.DataFrame,
    semilla: int = SEMILLA_MUESTRA,
    objetivo: int = TAMANO_MUESTRA,
) -> list[str]:
    """Selecciona entrevistas (id_doc) para la muestra de etiquetado.

    Muestreo determinista estratificado por el TOMO del top-1 de cada
    entrevista: hasta N_POR_TOMO_MUESTRA entrevistas por tomo en el orden
    canonico de TOMOS_ORDEN y, si no se alcanza el objetivo, se completa con
    el resto de candidatas. Solo entran entrevistas con >= 3 matches (top-3
    completo para el etiquetador). Semilla fija y documentada.

    Args:
        ranking_top5: ranking con columnas [id_doc, rango, id_segmento,...].
        semilla: semilla del generador aleatorio (determinismo).
        objetivo: numero de entrevistas objetivo de la muestra.

    Returns:
        Lista de id_doc seleccionados (orden de la muestra).
    """
    rng = np.random.default_rng(semilla)
    n_matches = ranking_top5.groupby("id_doc")["rango"].max()
    candidatas = set(n_matches.index[n_matches >= 3].tolist())
    top1 = ranking_top5[ranking_top5["rango"] == 1]
    tomo_por_query = dict(zip(top1["id_doc"], top1["tomo"], strict=True))
    por_tomo: dict[str, list[str]] = {tomo: [] for tomo in TOMOS_ORDEN}
    # Iteracion ordenada: el orden de un set de strings depende del hash seed
    # del proceso (PYTHONHASHSEED) y el rng.shuffle() de abajo NO es inmune al
    # orden inicial; sin sorted() la "muestra determinista" varia entre
    # procesos (misma semilla 42, salida distinta). Determinismo real.
    for id_doc in sorted(candidatas):
        por_tomo[tomo_por_query.get(id_doc, "")].append(id_doc)
    seleccionadas: list[str] = []
    for tomo in TOMOS_ORDEN:
        mezcladas = por_tomo[tomo]
        rng.shuffle(mezcladas)
        seleccionadas.extend(mezcladas[:N_POR_TOMO_MUESTRA])
    if len(seleccionadas) < objetivo:
        resto = sorted(candidatas - set(seleccionadas))
        rng.shuffle(resto)
        seleccionadas.extend(resto[: objetivo - len(seleccionadas)])
    return seleccionadas[:objetivo]


# ---------------------------------------------------------------------------
# Reporte y registro de metricas
# ---------------------------------------------------------------------------


def resumen_vector(valores: np.ndarray) -> dict[str, float]:
    """Resumen estadistico (percentiles, media) de un vector de scores."""
    if valores.size == 0:
        return {}
    percentiles = np.percentile(valores, [25, 50, 75])
    return {
        "min": round(float(valores.min()), 6),
        "p25": round(float(percentiles[0]), 6),
        "mediana": round(float(percentiles[1]), 6),
        "media": round(float(valores.mean()), 6),
        "p75": round(float(percentiles[2]), 6),
        "max": round(float(valores.max()), 6),
    }


def escribir_reporte(
    ruta: Path,
    estadisticas: dict[str, Any],
    ranking_top5: pd.DataFrame,
    texto_queries_vacias: list[str],
    tiempo_total_s: float,
) -> None:
    """Escribe resultados/ir_tfidf_report.json (solo experimento base).

    Args:
        ruta: ruta del archivo JSON de salida.
        estadisticas: config aplicada, parada, correcciones y vocabulario.
        ranking_top5: ranking base con [id_doc, rango, id_segmento, score,...].
        texto_queries_vacias: lineas (id_doc) sin tokens tras la parada.
        tiempo_total_s: duracion total del experimento.
    """
    top1 = ranking_top5[ranking_top5["rango"] == 1].copy()
    queries_con_match = set(ranking_top5["id_doc"])
    n_queries = int(estadisticas["n_queries"])
    tomos = top1["tomo"].value_counts()
    distribucion = [
        {
            "tomo": str(tomo),
            "n_queries": int(n),
            "pct": round(100.0 * int(n) / len(top1), 2) if len(top1) else 0.0,
        }
        for tomo, n in tomos.items()
    ]
    reporte: dict[str, Any] = {
        "_comentario": (
            "Informe del experimento 1 (base) de IR TF-IDF del TALLER-4 "
            "(Fase 6). Evaluacion P@1/P@5 pendiente de etiquetado humano de "
            "resultados/evaluacion_muestra.csv ({relacionado, parcial, no}); "
            "no hay gold estandar (estado_del_arte)."
        ),
        "experimento": "fase6_tfidf_base",
        "script": "experimentos/experimento_1_tfidf_base.py",
        "variante": "base",
        "direccion_matching": "entrevistas = queries; segmentos de libro = coleccion (ADR-005)",
        "fecha": "2026-09-04",
        "config_vectorizador": {k: v for k, v in estadisticas["config"].items()},
        "parada_en_consumo": estadisticas["parada"],
        "n_correcciones_lemas": estadisticas["n_correcciones_lemas"],
        "corpus": {
            "coleccion_n_docs": estadisticas["n_docs_coleccion"],
            "queries_n_docs": n_queries,
            "queries_vacias_tras_parada_n": len(texto_queries_vacias),
            "queries_vacias_tras_parada_ids": texto_queries_vacias,
            "nota_vacias": (
                "2 documentos de entrevista vacios legitimos (indices 717 y "
                "1674 de entrevistas_lemas.txt); no generan filas de ranking."
            ),
            "lemas_coleccion_antes_parada": estadisticas["lemas_coleccion"],
            "lemas_coleccion_despues_parada": estadisticas[
                "lemas_coleccion_limpios"
            ],
            "lemas_queries_antes_parada": estadisticas["lemas_queries"],
            "lemas_queries_despues_parada": estadisticas["lemas_queries_limpios"],
        },
        "vocabulario": {
            "n_features": estadisticas["n_features"],
            "n_unigramas": estadisticas["n_unigramas"],
            "n_bigramas": estadisticas["n_bigramas"],
            "densidad_matriz_tf_coleccion": estadisticas["densidad"],
        },
        "ranking": {
            "top_k": 5,
            "filas_ranking_top5": len(ranking_top5),
            "queries_con_al_menos_un_match": len(queries_con_match),
            "cobertura_pct": (
                round(100.0 * len(queries_con_match) / n_queries, 2)
                if n_queries
                else 0.0
            ),
        },
        "top1": {
            "n_queries_con_top1": len(top1),
            "distribucion_tomos": distribucion,
            "score": resumen_vector(top1["score"].to_numpy(dtype=float)),
        },
        "evaluacion_cualitativa": {
            "archivo": "resultados/evaluacion_muestra.csv",
            "n_filas": int(estadisticas.get("n_filas_muestra", 0)),
            "semilla": SEMILLA_MUESTRA,
            "criterio": (
                "muestra determinista estratificada por tomo del top-1: hasta "
                f"{N_POR_TOMO_MUESTRA} entrevistas por tomo (orden canonico de "
                f"plan/PLAN.md) hasta {TAMANO_MUESTRA} entrevistas con top-3 "
                "completo; fragmentos de ~500 chars; columna 'etiqueta' vacia "
                "para etiquetado humano posterior."
            ),
            "guia_etiquetado": (
                "etiquetar cada fila segun la relacion de la entrevista "
                "(fragmento_entrevista) con el segmento r1 "
                "(fragmento_segmento_r1): 'relacionado' si trata el mismo "
                "tema/violencia/territorio, 'parcial' si hay solape parcial, "
                "'no' si no hay relacion evidente."
            ),
        },
        "tiempo_total_s": round(tiempo_total_s, 1),
        "estado": "completado",
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(reporte, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def actualizar_metricas(entrada: dict[str, Any]) -> None:
    """Anade la entrada fase6_tfidf_base a resultados/metricas.json.

    Idempotente: reemplaza la entrada previa de la misma fase (patron de
    scripts/03_eda.py, actualizar_metricas).

    Args:
        entrada: dict de metricas de la fase 6.
    """
    metricas = json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))
    metricas["metricas"] = [
        m for m in metricas.get("metricas", []) if m.get("experimento") != "fase6_tfidf"
    ]
    metricas.setdefault("metricas", []).append(entrada)
    RUTA_METRICAS.write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Persistencia de rankings
# ---------------------------------------------------------------------------


def escribir_csv(marco: pd.DataFrame, ruta: Path, force: bool = False) -> None:
    """Escribe un DataFrame como CSV (no sobrescribe sin force).

    Args:
        marco: DataFrame a persistir.
        ruta: ruta del archivo CSV.
        force: si False y el archivo existe, aborta con RuntimeError.

    Raises:
        RuntimeError: si el archivo existe y force es False.
    """
    if ruta.exists() and not force:
        raise RuntimeError(
            f"{ruta} ya existe; use --force para sobrescribirlo "
            "(o corra una variante con nombre propio: --variante A|B|C)"
        )
    ruta.parent.mkdir(parents=True, exist_ok=True)
    marco.to_csv(ruta, index=False)
    print(f"  [csv] {ruta.relative_to(RUTA_PROYECTO)} ({len(marco)} filas)")


# ---------------------------------------------------------------------------
# Resumen de consola
# ---------------------------------------------------------------------------


def imprimir_resumen(
    variante: str,
    ranking_top5: pd.DataFrame,
    n_queries: int,
    n_features: int,
    tiempo_s: float,
) -> None:
    """Imprime el resumen legible final (top tomos y ejemplos top-1).

    Args:
        variante: identificador del experimento ejecutado.
        ranking_top5: ranking del experimento.
        n_queries: numero total de queries del corpus.
        n_features: tamano del vocabulario (unigramas + bigramas).
        tiempo_s: duracion total del experimento.
    """
    top1 = ranking_top5[ranking_top5["rango"] == 1].copy()
    queries_con_match = top1["id_doc"].nunique()
    print("\n==================== RESUMEN FASE 6 ====================")
    print(
        f"[{variante}] queries con match: {queries_con_match}/{n_queries} "
        f"({100.0 * queries_con_match / n_queries:.2f} %) | "
        f"filas ranking top-5: {len(ranking_top5)}"
    )
    print(f"[{variante}] features: {n_features} | tiempo: {tiempo_s:.1f} s")
    tomos = top1["tomo"].value_counts(normalize=True).head(3)
    top_tomos = " | ".join(f"{tomo} {100.0 * pct:.1f}%" for tomo, pct in tomos.items())
    print(f"[{variante}] top-1 por tomo: {top_tomos}")
    scores = top1["score"].to_numpy(dtype=float)
    if scores.size:
        print(
            f"[{variante}] score top-1: media {scores.mean():.4f} | "
            f"mediana {np.median(scores):.4f} | max {scores.max():.4f}"
        )
    if variante == "base":
        rng = np.random.default_rng(SEMILLA_EJEMPLOS)
        candidatas = sorted(top1["id_doc"].unique().tolist())
        rng.shuffle(candidatas)
        print(f"[{variante}] ejemplos top-1 (semilla {SEMILLA_EJEMPLOS}):")
        for id_doc in candidatas[:3]:
            fila = top1[top1["id_doc"] == id_doc].iloc[0]
            print(
                f"    {id_doc} -> {fila['id_segmento']} "
                f"({fila['tomo']}) score={fila['score']:.4f} | {fila['titulo_segmento']}"
            )
    print("=========================================================")


# ---------------------------------------------------------------------------
# Seccion de integracion: orquestacion de la variante solicitada
# ---------------------------------------------------------------------------


def _resolver_variante(
    variante: str,
) -> tuple[dict[str, Any], set[str], dict[str, str]]:
    """Resuelve la config efectiva (vectorizador, parada, correcciones).

    Args:
        variante: "base", "A", "B" o "C" (ver docstring del modulo).

    Returns:
        Tupla (config_vectorizador, parada, correcciones) para la variante.

    Raises:
        ValueError: si la variante no es valida.
    """
    if variante not in VARIANTES_VALIDAS:
        raise ValueError(f"variante invalida: {variante} (validas: {VARIANTES_VALIDAS})")
    config = cargar_config_vectorizador()
    if variante == "A":
        config["ngram_range"] = (1, 1)
    parada = construir_parada(incluir_muletillas=(variante != "B"))
    correcciones = cargar_correcciones() if variante != "C" else {}
    return config, parada, correcciones


def correr_variante(
    variante: str,
    lineas_libros: list[str],
    docs_libros: list[dict[str, Any]],
    lineas_entrevistas: list[str],
    docs_entrevistas: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ejecuta el pipeline completo de una variante (sin persistencia).

    Args:
        variante: identificador del experimento ("base", "A", "B", "C").
        lineas_libros: lemas de la coleccion (1 segmento por linea).
        docs_libros: metadatos de los segmentos (libros_docs.json).
        lineas_entrevistas: lemas de las queries (1 entrevista por linea).
        docs_entrevistas: metadatos de las entrevistas (entrevistas_docs.json).

    Returns:
        Contexto con ranking_top5, config aplicada, parada usada, n de
        correcciones, vocabulario, matrices, conteos y estadisticas.
    """
    config, parada, correcciones = _resolver_variante(variante)
    print(
        f"\n[experimento {variante}] ngram={config['ngram_range']} | "
        f"parada={len(parada)} tokens | correcciones={len(correcciones)}"
    )

    # ===== SECCION 2: preprocesado en consumo (parada + correcciones) =====
    libros_limpios, n_lemas_libros = preprocesar_corpus(lineas_libros, parada, correcciones)
    entrevistas_limpios, n_lemas_entrevistas = preprocesar_corpus(
        lineas_entrevistas, parada, correcciones
    )
    vacias_tras_parada = [
        docs_entrevistas[i]["id"]
        for i, n in enumerate(n_lemas_entrevistas)
        if n == 0
    ]
    print(
        f"  lemas coleccion: {sum(n_lemas_libros)} -> "
        f"docs limpios {sum(1 for n in n_lemas_libros if n > 0)}/{len(lineas_libros)}"
    )
    print(
        f"  lemas queries: {sum(n_lemas_entrevistas)} -> "
        f"queries no vacias {sum(1 for n in n_lemas_entrevistas if n > 0)}/"
        f"{len(lineas_entrevistas)} | vacias tras parada: {len(vacias_tras_parada)}"
    )

    # ===== SECCION 3: vectorizador TF-IDF (fit coleccion, transform queries) =====
    vectorizador = construir_vectorizador(config)
    x_coleccion = vectorizador.fit_transform(libros_limpios)
    x_queries = vectorizador.transform(entrevistas_limpios)
    n_features = len(vectorizador.vocabulary_)
    n_bigramas = sum(1 for termino in vectorizador.vocabulary_ if " " in termino)
    densidad = x_coleccion.nnz / max(x_coleccion.shape[0] * n_features, 1)
    print(
        f"  fit coleccion: {x_coleccion.shape} | transform queries: {x_queries.shape}"
    )
    print(
        f"  vocabulario: {n_features} features "
        f"({n_features - n_bigramas} unigramas + {n_bigramas} bigramas) | "
        f"densidad tf coleccion: {densidad:.6f}"
    )

    # ===== SECCION 4: similitud coseno por batches y ranking top-5 =====
    similitudes = similitud_coseno_batches(x_queries, x_coleccion)
    ids_segmentos = [registro["id"] for registro in docs_libros]
    ids_queries = [registro["id"] for registro in docs_entrevistas]
    ranking = construir_ranking(similitudes, ids_queries, ids_segmentos, top_k=5)
    ranking_top5 = enriquecer_ranking(ranking, docs_libros, ids_segmentos)
    print(f"  ranking top-5: {len(ranking_top5)} filas | head:")
    print(ranking_top5.head(8).to_string(index=False))

    contexto: dict[str, Any] = {
        "ranking_top5": ranking_top5,
        "similitudes": similitudes,
        "config": config,
        "parada_tokens": len(parada),
        "n_correcciones_lemas": len(correcciones),
        "n_features": n_features,
        "n_bigramas": n_bigramas,
        "n_unigramas": n_features - n_bigramas,
        "densidad": densidad,
        "n_docs_coleccion": len(lineas_libros),
        "n_queries": len(lineas_entrevistas),
        "lemas_coleccion": sum(len(linea.split()) for linea in lineas_libros),
        "lemas_coleccion_limpios": sum(n_lemas_libros),
        "lemas_queries": sum(len(linea.split()) for linea in lineas_entrevistas),
        "lemas_queries_limpios": sum(n_lemas_entrevistas),
        "queries_vacias": vacias_tras_parada,
        "vectorizador": vectorizador,
    }
    return contexto


def escribir_archivos_base(
    contexto: dict[str, Any],
    docs_libros: list[dict[str, Any]],
    docs_entrevistas: list[dict[str, Any]],
    force: bool,
    tiempo_total_s: float,
) -> None:
    """Persiste los entregables del experimento base.

    Escribe ranking_top5.csv, ranking_top10.csv, evaluacion_muestra.csv,
    ir_tfidf_report.json y actualiza resultados/metricas.json.

    Args:
        contexto: contexto de correr_variante (variante base).
        docs_libros: metadatos de segmentos (orden de la coleccion).
        docs_entrevistas: metadatos de entrevistas (orden de las queries).
        force: sobrescribir archivos existentes.
        tiempo_total_s: duracion total del experimento.
    """
    ranking_top5 = contexto["ranking_top5"]
    # ===== SECCION 5: guardado del ranking (top-5 y top-10) =====
    escribir_csv(ranking_top5, RUTA_RESULTADOS / "ranking_top5.csv", force=force)
    ranking_top10 = enriquecer_ranking(
        construir_ranking(contexto["similitudes"], [d["id"] for d in docs_entrevistas], [d["id"] for d in docs_libros], top_k=10),
        docs_libros,
        [d["id"] for d in docs_libros],
    )
    escribir_csv(ranking_top10, RUTA_RESULTADOS / "ranking_top10.csv", force=force)

    # ===== SECCION 6: muestra de evaluacion cualitativa (solo base) =====
    seleccionadas = seleccionar_muestra_entrevistas(ranking_top5)
    print(
        f"  muestra de evaluacion: {len(seleccionadas)} entrevistas "
        f"(semilla {SEMILLA_MUESTRA})"
    )
    segmentos = [
        json.loads(linea)
        for linea in (RUTA_CORPUS / "segmentos_libros.jsonl").open(encoding="utf-8")
    ]
    texto_segmento = {seg["id_segmento"]: seg for seg in segmentos}
    filas_muestra = _construir_filas_muestra(
        seleccionadas, ranking_top5, docs_entrevistas, texto_segmento
    )
    marco_muestra = pd.DataFrame(
        filas_muestra,
        columns=[
            "id_doc",
            "r1_id_segmento",
            "r1_tomo",
            "r1_titulo",
            "r1_score",
            "r2_id_segmento",
            "r2_tomo",
            "r2_titulo",
            "r2_score",
            "r3_id_segmento",
            "r3_tomo",
            "r3_titulo",
            "r3_score",
            "fragmento_segmento_r1",
            "fragmento_entrevista",
            "etiqueta",
        ],
    )
    marco_muestra["etiqueta"] = ""  # campo vacio para etiquetado humano
    escribir_csv(marco_muestra, RUTA_RESULTADOS / "evaluacion_muestra.csv", force=force)
    contexto["n_filas_muestra"] = len(marco_muestra)

    # ===== SECCION 7: reporte JSON y metricas acumulativas =====
    texto_vacias = contexto["queries_vacias"]
    contexto["parada"] = {
        "spacy_stop_words": len(STOP_WORDS),
        "muletillas_habla": len(cargar_muletillas_config()),
        "ruido_libros": len(cargar_ruido_libros()),
        "total_efectiva": contexto["parada_tokens"],
    }
    escribir_reporte(
        RUTA_RESULTADOS / "ir_tfidf_report.json",
        contexto,
        ranking_top5,
        texto_vacias,
        tiempo_total_s,
    )
    print("  [json] resultados/ir_tfidf_report.json")
    entrada_metricas = {
        "experimento": "fase6_tfidf",
        "fase": "6",
        "script": "experimentos/experimento_1_tfidf_base.py",
        "fecha": "2026-09-04",
        "descripcion": (
            "Modelo IR TF-IDF (punto 3 del taller): entrevistas = queries "
            "(2.486), segmentos de libro = coleccion (917); fit sobre lemas "
            "de libros con parada en consumo y correcciones; coseno por "
            "batches; ranking top-5 por query"
        ),
        "direccion_matching": "entrevistas = queries; segmentos = coleccion",
        "config_vectorizador": contexto["config"],
        "vocabulario_features": contexto["n_features"],
        "filas_ranking_top5": len(ranking_top5),
        "queries_con_match": ranking_top5["id_doc"].nunique(),
        "top1_score_media": round(
            float(ranking_top5[ranking_top5["rango"] == 1]["score"].mean()), 6
        ),
        "muestra_evaluacion_filas": len(marco_muestra),
        "muestra_evaluacion_semilla": SEMILLA_MUESTRA,
        "tiempo_total_s": round(tiempo_total_s, 1),
        "estado": "completado",
    }
    actualizar_metricas(entrada_metricas)
    print("  [json] resultados/metricas.json: entrada fase6_tfidf actualizada")


def _construir_filas_muestra(
    seleccionadas: list[str],
    ranking_top5: pd.DataFrame,
    docs_entrevistas: list[dict[str, Any]],
    texto_segmento: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Construye las filas (formato ancho) de evaluacion_muestra.csv.

    Para cada entrevista: metadatos de sus 3 primeros segmentos, fragmento
    del segmento r1 y fragmento de la entrevista (primeros ~500 chars). El
    texto de la entrevista se lee en streaming de
    data/corpus/entrevistas_normalizadas.jsonl (mismo orden que los lemas).

    Args:
        seleccionadas: id_doc de las entrevistas de la muestra.
        ranking_top5: ranking base con los top-5 por query.
        docs_entrevistas: metadatos de entrevistas en orden de linea.
        texto_segmento: mapa id_segmento -> registro con 'texto' y 'titulo'.

    Returns:
        Filas del CSV de evaluacion (campo 'etiqueta' vacio).
    """
    posicion_por_id = {registro["id"]: i for i, registro in enumerate(docs_entrevistas)}
    indices_muestra = sorted({posicion_por_id[id_doc] for id_doc in seleccionadas})
    texto_por_id: dict[str, str] = {}
    with (RUTA_CORPUS / "entrevistas_normalizadas.jsonl").open(encoding="utf-8") as fh:
        for i, linea in enumerate(fh):
            if i in indices_muestra:
                registro = json.loads(linea)
                texto_por_id[registro["id_doc"]] = registro["text_normalizado"]
    filas: list[dict[str, Any]] = []
    for id_doc in seleccionadas:
        filas_top = ranking_top5[ranking_top5["id_doc"] == id_doc].sort_values("rango")
        fila: dict[str, Any] = {"id_doc": id_doc}
        for rango in (1, 2, 3):
            sub = filas_top[filas_top["rango"] == rango]
            prefijo = f"r{rango}"
            if len(sub):
                reg = sub.iloc[0]
                fila[f"{prefijo}_id_segmento"] = reg["id_segmento"]
                fila[f"{prefijo}_tomo"] = reg["tomo"]
                fila[f"{prefijo}_titulo"] = reg["titulo_segmento"]
                fila[f"{prefijo}_score"] = reg["score"]
            else:
                fila[f"{prefijo}_id_segmento"] = ""
                fila[f"{prefijo}_tomo"] = ""
                fila[f"{prefijo}_titulo"] = ""
                fila[f"{prefijo}_score"] = np.nan
        segmento_r1 = texto_segmento.get(fila["r1_id_segmento"])
        fila["fragmento_segmento_r1"] = (
            fragmento_texto(segmento_r1["texto"], segmento_r1["titulo"])
            if segmento_r1
            else ""
        )
        fila["fragmento_entrevista"] = fragmento_texto(texto_por_id.get(id_doc, ""))
        fila["etiqueta"] = ""
        filas.append(fila)
    return filas


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parsear_argumentos(argv: list[str] | None = None) -> argparse.Namespace:
    """Construye el parser de argumentos de la linea de comandos."""
    parser = argparse.ArgumentParser(
        description="Modelo IR TF-IDF TALLER-4 (Fase 6): entrevistas = queries, "
        "segmentos de libro = coleccion. ADR-005."
    )
    parser.add_argument(
        "--variante",
        choices=VARIANTES_VALIDAS,
        default="base",
        help="base: config congelada (default); A: ngram (1,1); B: sin "
        "muletillas; C: sin correcciones de lemas",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="queries por batch en el producto de similitud (default: 512)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="sobrescribe los archivos fijos del base (ranking_top5.csv, "
        "ranking_top10.csv, evaluacion_muestra.csv, ir_tfidf_report.json) si "
        "ya existen",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada: ejecuta el experimento de la variante solicitada."""
    inicio = time.time()
    args = parsear_argumentos(argv)
    try:
        # ===== SECCION 1: carga de lemas y metadatos (alineados) =====
        lineas_libros, docs_libros = cargar_lemas_y_metadatos("libros")
        lineas_entrevistas, docs_entrevistas = cargar_lemas_y_metadatos(
            "entrevistas"
        )
        print(
            f"Coleccion (segmentos de libro): {len(lineas_libros)} docs | "
            f"queries (entrevistas): {len(lineas_entrevistas)} docs"
        )
        contexto = correr_variante(
            args.variante,
            lineas_libros,
            docs_libros,
            lineas_entrevistas,
            docs_entrevistas,
        )
        ranking_top5 = contexto["ranking_top5"]
        if args.variante == "base":
            escribir_archivos_base(
                contexto, docs_libros, docs_entrevistas, args.force, time.time() - inicio
            )
        else:
            # Variantes: ranking con nombre propio, sin tocar los del base.
            ruta_variante = RUTA_RESULTADOS / f"ranking_top5_{args.variante}.csv"
            ranking_top5.to_csv(ruta_variante, index=False)
            print(
                f"  [csv] resultados/ranking_top5_{args.variante}.csv "
                f"({len(ranking_top5)} filas)"
            )
        imprimir_resumen(
            args.variante,
            ranking_top5,
            contexto["n_queries"],
            contexto["n_features"],
            time.time() - inicio,
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - reporteria de errores del main
        import traceback

        traceback.print_exc()
        print(f"Error en el experimento {args.variante}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
