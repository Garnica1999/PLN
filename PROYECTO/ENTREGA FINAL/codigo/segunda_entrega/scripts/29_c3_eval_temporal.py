"""29_c3_eval_temporal.py
Fecha: 2026-10-04
Descripcion: Evaluacion C3 de la deriva temporal de la ruta ganadora v13
    (hibrido_b2_rerank_chunks_top50_b1: hibrido 0.5*BM25 + 0.5*B2 sobre
    top-50 candidatos, rerank del cross-encoder B1 por chunks con
    max-pooling y prior de hubs alpha=1.5, salida k=10) sobre el corpus
    2022-2026 (entidades NER silver generadas por scripts/28_c3_preparar_corpus.py)
    frente al benchmark historico M7 (90 casos, 1992-2021) congelado en
    resultados/vinculo_cp_benchmark_v13.json.
    Metricas: nDCG@10, F1@5, F1@10, MRR, P@5 y R@5 con IC 95% bootstrap
    (1000 remuestreos, seed 42), desglose por anio y tipo, y comparacion
    descriptiva (NO pareada) de periodos historicos vs 2022-2026.
    Salidas: resultados/c3_eval_temporal.json, figuras/c3_deriva_temporal.png
    y experimentos/29_c3_eval_temporal.log. Con --smoke/--limite todo se
    escribe en /tmp/opencode/c3_eval_smoke/ y nunca se sobrescriben los
    artefactos reales.
Estado del arte aplicado: protocolo M7 (ADR-5) reutilizado por importlib
    desde scripts/12_evaluar_vinculo_cp_v2.py: indice de pasajes permanentes
    de la CP (385 articulos), chunking BM25 800/200 chars, consulta = texto
    completo, candidatos top-50 del hibrido pre-registrado 0.5/0.5, pares
    (chunk, pasaje) por mejor pasaje BM25 + top-3 chunks con max-pooling,
    prior de hubs de co-citacion (alpha=1.5), IC bootstrap no parametrico
    (Efron-Tibshirani) y semilla 42. La referencia de relevancia son las
    citas NER silver (script 28); "gold" queda reservado para anotacion
    humana (regla de terminologia del proyecto).
"""

import argparse
import contextlib
import importlib.util
import json
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

CONFIG = {
    "version": "c3-eval-temporal-v1",
    "semilla": 42,
    "por_anio": 20,
    "smoke_limite": 10,
    "estado_ok": "ok",
    "estrategia": "hibrido_b2_rerank_chunks_top50_b1",
    "contextos": {"sin_hubs": 0.0, "con_hubs": 1.5},
    "contexto_primario": "con_hubs",
    "k_candidatos": 50,
    "k_medidas": 5,
    "k_ndcg": 10,
    "bootstrap": 1000,
    "periodo_c3": "2022-2026",
    "ruta_script_12": "scripts/12_evaluar_vinculo_cp_v2.py",
    "ruta_corpus_c3": "data/corpus_2022_2026/corpus_2022_2026.parquet",
    "ruta_entidades_c3": "data/corpus_2022_2026/entidades_2022_2026.parquet",
    "ruta_benchmark_v13": "resultados/vinculo_cp_benchmark_v13.json",
    "ruta_salida_json": "resultados/c3_eval_temporal.json",
    "ruta_salida_figura": "figuras/c3_deriva_temporal.png",
    "ruta_log": "experimentos/29_c3_eval_temporal.log",
    "dir_smoke": "/tmp/opencode/c3_eval_smoke",
}

ROOT = Path(__file__).resolve().parent.parent

# Medidas exigidas por la spec C3 (las cinco del protocolo M7 mas R@5).
MEDIDAS_C3 = (
    "ndcg_at_10",
    "f1_at_5",
    "f1_at_10",
    "mrr",
    "p_at_5",
    "r_at_5",
)


class _DuplicadorSalida:
    """Duplica stdout y stderr hacia el archivo de log de la corrida.

    Los print() del pipeline (incluidos los de las funciones reutilizadas
    del script 12) y los tracebacks de error quedan asi registrados en
    experimentos/. El flujo original se captura al construir el objeto,
    de modo que no hay recursion aunque se instale en ambos canales.
    """

    def __init__(self, archivo) -> None:
        """Inicializa el duplicador.

        Args:
            archivo: Flujo de texto abierto en modo escritura.
        """
        self._archivo = archivo
        self._original = sys.stdout

    def write(self, datos: str) -> None:
        """Escribe en consola y en el archivo de log.

        Args:
            datos: Texto a escribir.
        """
        self._original.write(datos)
        self._archivo.write(datos)

    def flush(self) -> None:
        """Fuerza el vaciado de ambos flujos."""
        self._original.flush()
        self._archivo.flush()


# =====================================================================
# SECCION 1: CONFIGURACION, INSUMOS Y CARGA (SOLO LECTURA)
# =====================================================================
def importar_evaluador_v12(ruta: Path) -> object:
    """Importa scripts/12_evaluar_vinculo_cp_v2.py por importlib.

    El nombre del archivo empieza por digito y no admite import estandar.
    Se reutilizan sin duplicar logica: resolver_rutas, configurar_semillas,
    preparar_pasajes_constitucion, cuotas_proporcionales, muestrear_anio,
    chunks_query, construir_indice_bm25, scores_bm25_caso,
    ranking_desde_scores, ranking_con_prior, metricas_ranking,
    bootstrap_ic_media, asignar_periodo, ndcg_at_k, cargar_o_construir_prior_hubs,
    resolver_modelo_b1/b2, aplicar_modo_b2, calcular_denso_b2 y
    ranking_rerank_chunks.

    Args:
        ruta: Ruta absoluta del script 12.

    Returns:
        Modulo del evaluador v12 con CONFIG resuelto contra su ROOT.

    Raises:
        FileNotFoundError: Si la ruta del script 12 no existe.
    """
    if not ruta.is_file():
        raise FileNotFoundError(f"no existe el evaluador v12: {ruta}")
    spec = importlib.util.spec_from_file_location("evaluador_v12", ruta)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar el evaluador v12: {ruta}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    modulo.resolver_rutas()
    return modulo


def exigir_archivo(ruta: Path, descripcion: str) -> None:
    """Verifica que un insumo exista (fail-fast antes de usar GPU).

    Args:
        ruta: Ruta del archivo requerido.
        descripcion: Nombre legible del insumo para el mensaje de error.

    Raises:
        FileNotFoundError: Si la ruta no es un archivo.
    """
    if not ruta.is_file():
        raise FileNotFoundError(
            f"falta {descripcion}: {ruta} (generalo con "
            "scripts/28_c3_preparar_corpus.py antes de --evaluar)"
        )


def validar_columnas(df: pd.DataFrame, requeridas: set, descripcion: str) -> None:
    """Valida que un DataFrame traiga las columnas requeridas.

    Args:
        df: DataFrame cargado.
        requeridas: Conjunto de nombres de columna exigidos.
        descripcion: Nombre legible del insumo para el mensaje de error.

    Raises:
        ValueError: Si falta alguna columna.
    """
    faltantes = sorted(requeridas - set(df.columns))
    if faltantes:
        raise ValueError(f"{descripcion}: columnas faltantes {faltantes}")


def cargar_corpus_ligero(ruta: Path) -> pd.DataFrame:
    """Carga el marco ligero del corpus 2022-2026 (sin textos).

    Args:
        ruta: Ruta del parquet del corpus del script 28.

    Returns:
        DataFrame con id, tipo, anio y estado normalizados.

    Raises:
        ValueError: Si faltan columnas requeridas.
    """
    df = pd.read_parquet(ruta, columns=["id", "tipo", "anio", "estado"])
    validar_columnas(df, {"id", "tipo", "anio", "estado"}, "corpus C3")
    df["id"] = df["id"].astype(str)
    df["tipo"] = df["tipo"].astype(str)
    df["anio"] = df["anio"].astype(int)
    df["estado"] = df["estado"].astype(str)
    return df


def _decodificar_lista(crudo: object) -> list:
    """Decodifica una celda de lista JSON del parquet de entidades.

    Args:
        crudo: Valor crudo (string JSON, lista o nulo).

    Returns:
        Lista de elementos; vacia si el valor es nulo o vacio.

    Raises:
        ValueError: Si el string no es JSON valido.
        TypeError: Si el JSON decodificado no es una lista.
    """
    if crudo is None:
        return []
    if isinstance(crudo, float) and pd.isna(crudo):
        return []
    if isinstance(crudo, (list, tuple)):
        return list(crudo)
    texto = str(crudo).strip()
    if not texto:
        return []
    try:
        datos = json.loads(texto)
    except json.JSONDecodeError as exc:
        raise ValueError(f"arts_cp con JSON invalido: {texto[:80]}") from exc
    if not isinstance(datos, list):
        raise TypeError(f"arts_cp no es una lista JSON: {texto[:80]}")
    return datos


def cargar_referencia_silver(
    ruta: Path, canon_valido: set
) -> tuple[dict, dict]:
    """Carga las citas CP NER silver por sentencia (referencia de relevancia).

    Filtra cada lista contra los canones validos del parquet completo de la
    CP (misma regla de arts_validos del script 12) y deduplica preservando
    el orden de aparicion.

    Args:
        ruta: Ruta del parquet de entidades del script 28.
        canon_valido: Conjunto de canones validos (497 del parquet completo).

    Returns:
        Tupla (referencia, stats): dict {id: [canones validos]} y conteos
        de docs, docs sin cita CP y citas validas totales.

    Raises:
        ValueError: Si faltan columnas requeridas.
    """
    df = pd.read_parquet(ruta, columns=["id", "arts_cp"])
    validar_columnas(df, {"id", "arts_cp"}, "entidades C3")
    referencia: dict[str, list[str]] = {}
    n_sin_cita = 0
    n_citas = 0
    for sid, crudo in zip(df["id"].astype(str), df["arts_cp"], strict=True):
        arts = _decodificar_lista(crudo)
        validos = [a for a in dict.fromkeys(str(x) for x in arts) if a in canon_valido]
        referencia[sid] = validos
        if not validos:
            n_sin_cita += 1
        n_citas += len(validos)
    stats = {
        "n_docs": len(referencia),
        "n_sin_cita_cp": int(n_sin_cita),
        "n_citas_validas": int(n_citas),
    }
    return referencia, stats


def cargar_benchmark_v13(ruta: Path) -> dict:
    """Carga el JSON congelado del benchmark v13 (solo lectura).

    Args:
        ruta: Ruta de resultados/vinculo_cp_benchmark_v13.json.

    Returns:
        Dict del benchmark.

    Raises:
        ValueError: Si falta alguna seccion requerida.
    """
    with ruta.open("r", encoding="utf-8") as manejador:
        datos = json.load(manejador)
    for clave in ("muestra", "comparacion_b2", "top10_por_caso"):
        if clave not in datos:
            raise ValueError(f"benchmark v13 sin la seccion requerida: {clave}")
    return datos


# =====================================================================
# SECCION 2: MUESTRA ESTRATIFICADA POR ANIO
# =====================================================================
def seleccionar_muestra(
    mod: object,
    marco: pd.DataFrame,
    referencia: dict,
    por_anio: int,
    semilla: int,
) -> tuple[pd.DataFrame, dict]:
    """Selecciona la muestra C3 estratificada por anio (seed fija).

    Regla: docs con estado ok y >= 1 cita CP valida; hasta por_anio docs por
    anio con cupos por tipo proporcionales (mayor resto) y relleno con
    cualquier tipo si un tipo no alcanza su cuota. Reutiliza los helpers
    cuotas_proporcionales y muestrear_anio del script 12 (mismo criterio
    determinista: orden por id, tipos alfabeticos, rng con semilla).

    Args:
        mod: Modulo del evaluador v12.
        marco: DataFrame del corpus (id, tipo, anio, estado).
        referencia: Dict {id: citas CP validas}.
        por_anio: Maximo de casos por anio.
        semilla: Semilla del generador aleatorio.

    Returns:
        Tupla (muestra, stats) con la muestra ordenada por (anio, id) y los
        conteos de poblacion elegible.

    Raises:
        RuntimeError: Si no hay ningun doc elegible.
    """
    elegibles = marco[
        (marco["estado"] == CONFIG["estado_ok"]) & marco["id"].isin(referencia)
    ].copy()
    elegibles = elegibles[
        elegibles["id"].map(lambda sid: len(referencia.get(sid, [])) > 0)
    ].copy()
    if elegibles.empty:
        raise RuntimeError(
            "no hay docs ok con >=1 cita CP valida: revisa las salidas de "
            "scripts/28_c3_preparar_corpus.py"
        )
    # Columna ano (sin i) duplicada para reutilizar los helpers del script 12.
    elegibles["ano"] = elegibles["anio"]
    rng = random.Random(semilla)
    filas = []
    for anio in sorted(elegibles["anio"].unique()):
        grupo = elegibles[elegibles["anio"] == anio].sort_values("id")
        cupos = mod.cuotas_proporcionales(
            grupo["tipo"].value_counts().to_dict(), min(por_anio, len(grupo))
        )
        filas.append(mod.muestrear_anio(grupo, cupos, rng))
    muestra = (
        pd.concat(filas, ignore_index=True)
        .sort_values(["anio", "id"])
        .reset_index(drop=True)
    )
    stats = {
        "regla": (
            "docs estado ok con >=1 cita CP valida (NER silver del script "
            "28 filtrado contra los 497 canones del parquet completo); "
            f"hasta {por_anio} docs/anio con cupos por tipo proporcionales "
            "(mayor resto) y relleno de deficit con cualquier tipo; "
            f"muestreo determinista seed {semilla} (helpers del script 12)"
        ),
        "n_elegibles": len(elegibles),
        "poblacion_por_anio": {
            int(clave): int(valor)
            for clave, valor in elegibles["anio"]
            .value_counts()
            .sort_index()
            .items()
        },
        "poblacion_por_tipo": {
            str(clave): int(valor)
            for clave, valor in elegibles["tipo"].value_counts().items()
        },
    }
    return muestra, stats


def resumen_conteos(muestra: pd.DataFrame) -> dict:
    """Resume los conteos de una muestra por anio y tipo.

    Args:
        muestra: DataFrame con columnas anio y tipo.

    Returns:
        Dict con n, por_anio y por_tipo.
    """
    return {
        "n": len(muestra),
        "por_anio": {
            int(clave): int(valor)
            for clave, valor in muestra["anio"].value_counts().sort_index().items()
        },
        "por_tipo": {
            str(clave): int(valor)
            for clave, valor in muestra["tipo"].value_counts().sort_index().items()
        },
    }


def limitar_muestra(muestra: pd.DataFrame, limite: int | None) -> pd.DataFrame:
    """Recorta la muestra a los primeros N casos (modo smoke/limite).

    Args:
        muestra: Muestra ordenada por (anio, id).
        limite: Numero maximo de casos; None conserva la muestra completa.

    Returns:
        Muestra recortada y reindexada.
    """
    if limite is None or limite >= len(muestra):
        return muestra
    return muestra.head(limite).reset_index(drop=True)


# =====================================================================
# SECCION 3: PASAJES, CASOS Y BM25 (RUTA GANADORA, 1a ETAPA)
# =====================================================================
def preparar_pasajes(mod: object) -> tuple:
    """Prepara los pasajes permanentes de la CP (protocolo M7).

    Args:
        mod: Modulo del evaluador v12.

    Returns:
        Tupla (pasajes, canones, grupos_indices, canon_valido) del script 12.
    """
    return mod.preparar_pasajes_constitucion(False)


def construir_casos(
    mod: object,
    muestra: pd.DataFrame,
    textos_por_id: dict,
    referencia: dict,
) -> list[dict]:
    """Construye los casos C3 con texto, ventana y chunks de query.

    Args:
        mod: Modulo del evaluador v12.
        muestra: Muestra de casos (id, tipo, anio).
        textos_por_id: Dict {id: texto E1-limpiado}.
        referencia: Dict {id: citas CP validas}.

    Returns:
        Lista de casos con id, tipo, ano, texto, arts, ventana, chunks_bm25
        y rankings vacios.

    Raises:
        RuntimeError: Si algun caso de la muestra no tiene texto.
    """
    casos = []
    sin_texto = []
    for _, fila in muestra.iterrows():
        sid = str(fila["id"])
        texto = textos_por_id.get(sid)
        if texto is None or not str(texto).strip():
            sin_texto.append(sid)
            continue
        casos.append(
            {
                "id": sid,
                "tipo": str(fila["tipo"]),
                "ano": int(fila["anio"]),
                "texto": str(texto),
                "arts": list(referencia[sid]),
            }
        )
    if sin_texto:
        raise RuntimeError(f"casos sin texto en el corpus: {sin_texto[:5]}")
    for caso in casos:
        caso["ventana"] = caso["texto"][: mod.CONFIG["ventana_query_chars"]]
        caso["chunks_bm25"] = mod.chunks_query(caso["ventana"])
        caso["rankings"] = {}
    return casos


def calcular_bm25(
    mod: object,
    casos: list[dict],
    pasajes: pd.DataFrame,
    canones: list,
    grupos_indices: list,
) -> float:
    """Calcula los scores BM25 y la matriz (chunk, pasaje) por caso.

    Args:
        mod: Modulo del evaluador v12.
        casos: Casos con chunks_bm25.
        pasajes: Pasajes del indice activo.
        canones: Canones alineados con los vectores de scores.
        grupos_indices: Arrays de posiciones de pasajes por articulo.

    Returns:
        Segundos de la etapa BM25.
    """
    inicio = time.time()
    indice = mod.construir_indice_bm25(pasajes)
    for caso in casos:
        scores, matriz = mod.scores_bm25_caso(
            indice, grupos_indices, canones, caso["chunks_bm25"]
        )
        caso["scores_bm25"] = scores
        caso["matriz_bm25"] = matriz
        caso["rankings"]["bm25"] = mod.ranking_desde_scores(scores, canones)
    return time.time() - inicio


# =====================================================================
# SECCION 4: MODELOS DE LA RUTA GANADORA (GPU CON GUARD)
# =====================================================================
def resolver_modelos_ganadora(mod: object) -> tuple[str, str]:
    """Resuelve y registra los modelos B2/B1 de la ruta ganadora v13.

    Usa los resolvers fail-fast del script 12 (ruta local modelos/b2_retriever
    y modelos/b1_reranker) y registra la estrategia
    hibrido_b2_rerank_chunks_top50_b1 en FUENTE_SCORES/ESTRATEGIAS_RERANK
    llamando a aplicar_modo_b2, de modo que ranking_con_prior trunque a 10
    con la misma semantica del run v13.

    Args:
        mod: Modulo del evaluador v12.

    Returns:
        Tupla (modelo_b2, modelo_b1) con rutas absolutas.

    Raises:
        FileNotFoundError: Si falta alguno de los modelos locales.
    """
    modelo_b2 = mod.resolver_modelo_b2(None)
    modelo_b1 = mod.resolver_modelo_b1(None)
    mod.aplicar_modo_b2(None, modelo_b2, modelo_b1)
    return modelo_b2, modelo_b1


def calcular_denso_b2(
    mod: object,
    casos: list[dict],
    pasajes: pd.DataFrame,
    canones: list,
    grupos_indices: list,
    dispositivo: str,
    modelo_b2: str,
) -> tuple[dict, str]:
    """Calcula el denso B2 y el hibrido 0.5*bm25 + 0.5*B2 por caso.

    Delega en calcular_denso_b2 del script 12 (cache de pasajes por slug en
    data/cache_emb_cp/<slug>.joblib) y devuelve la clave del vector hibrido
    materializado en cada caso.

    Args:
        mod: Modulo del evaluador v12.
        casos: Casos con scores_bm25 y ventana.
        pasajes: Pasajes del indice activo.
        canones: Canones alineados con los vectores de scores.
        grupos_indices: Arrays de posiciones de pasajes por articulo.
        dispositivo: "cuda"/"cpu" del guard de lib/reranker_bge.py.
        modelo_b2: Ruta local del retriever denso B2.

    Returns:
        Tupla (stats, clave_hibrido).
    """
    mod.CONFIG["b2"]["modelo"] = modelo_b2
    inicio = time.time()
    stats = mod.calcular_denso_b2(
        casos, pasajes, canones, grupos_indices, dispositivo
    )
    stats["segundos_total"] = round(time.time() - inicio, 1)
    clave_hibrido = f"scores_hibrido_{stats['slug']}"
    return stats, clave_hibrido


def rerankear_ganadora(
    mod: object,
    casos: list[dict],
    pasajes: pd.DataFrame,
    canones: list,
    grupos_indices: list,
    reranker: object,
    estrategia: str,
    k_candidatos: int,
    clave_hibrido: str,
) -> dict:
    """Rerankea el top-k del hibrido B2 con el cross-encoder B1 por chunks.

    Mismos pares (chunk, pasaje) y max-pooling que la ruta de producto:
    mejor pasaje BM25 del candidato + top-3 chunks de la query, agregacion
    por maximo, salida truncada a 10 (ranking_rerank_chunks del script 12).

    Args:
        mod: Modulo del evaluador v12.
        casos: Casos con matriz_bm25, chunks_bm25 y scores del hibrido B2.
        pasajes: Pasajes del indice activo.
        canones: Canones alineados con los vectores de scores.
        grupos_indices: Arrays de posiciones de pasajes por articulo.
        reranker: RerankerBGE cargado (modelo B1).
        estrategia: Nombre de la estrategia ganadora.
        k_candidatos: Candidatos de la 1a etapa (50 en v13).
        clave_hibrido: Clave del vector de scores hibrido B2 en cada caso.

    Returns:
        Dict de stats del rerank (segundos, pares por caso, max tokens).
    """
    clave_scores = f"scores_{estrategia}"
    max_tokens = []
    inicio = time.time()
    for caso in casos:
        ranking, scores, max_tok = mod.ranking_rerank_chunks(
            reranker,
            caso[clave_hibrido],
            caso["matriz_bm25"],
            caso["chunks_bm25"],
            canones,
            pasajes,
            grupos_indices,
            k_candidatos,
        )
        caso[clave_scores] = scores
        caso["rankings"][estrategia] = ranking
        max_tokens.append(max_tok)
    return {
        "segundos": round(time.time() - inicio, 1),
        "pares_por_caso": int(k_candidatos * mod.CONFIG["k_chunks_rerank"]),
        "max_tokens_par": int(max(max_tokens)) if max_tokens else 0,
    }


# =====================================================================
# SECCION 5: EVALUACION (METRICAS + IC BOOTSTRAP)
# =====================================================================
def medidas_caso(mod: object, relevantes: list, ranking: list) -> dict:
    """Calcula las seis medidas C3 de un caso (P/R/F1@5, F1@10, MRR, nDCG@10).

    Args:
        mod: Modulo del evaluador v12.
        relevantes: Citas CP de referencia del caso.
        ranking: Ranking de producto de la estrategia (<= 10).

    Returns:
        Dict redondeado a 6 decimales con las medidas de MEDIDAS_C3.
    """
    m5 = mod.metricas_ranking(relevantes, ranking, CONFIG["k_medidas"])
    m10 = mod.metricas_ranking(relevantes, ranking, CONFIG["k_ndcg"])
    valores = {
        "ndcg_at_10": m5["ndcg_at_10"],
        "f1_at_5": m5["f1_at_k"],
        "f1_at_10": m10["f1_at_k"],
        "mrr": m5["mrr"],
        "p_at_5": m5["p_at_k"],
        "r_at_5": m5["r_at_k"],
    }
    return {medida: round(float(valores[medida]), 6) for medida in MEDIDAS_C3}


def evaluar_contexto(
    mod: object,
    casos: list[dict],
    estrategia: str,
    prior: dict,
    canones: list,
    alpha: float,
) -> list[dict]:
    """Evalua la estrategia en un contexto del prior de hubs (alpha fijo).

    Reconstruye el ranking de producto con ranking_con_prior (la misma
    funcion que usa el run v13 para aplicar el prior y truncar a 10) y
    calcula las medidas por caso.

    Args:
        mod: Modulo del evaluador v12.
        casos: Casos con scores de la estrategia.
        estrategia: Nombre de la estrategia ganadora.
        prior: Dict {canon: valor} del prior de hubs.
        canones: Canones alineados con los vectores de scores.
        alpha: Peso del prior (0.0 sin hubs; 1.5 ruta de producto).

    Returns:
        Lista de filas por caso con id, ano, tipo, n_relevantes y medidas.
    """
    clave_scores = f"scores_{estrategia}"
    filas = []
    for caso in casos:
        ranking, _ = mod.ranking_con_prior(
            caso[clave_scores], canones, estrategia, prior, alpha
        )
        fila = {
            "id": caso["id"],
            "ano": int(caso["ano"]),
            "tipo": caso["tipo"],
            "n_relevantes": len(caso["arts"]),
        }
        fila.update(medidas_caso(mod, caso["arts"], ranking))
        filas.append(fila)
    return filas


def agregar_medidas(
    mod: object, filas: list[dict], n_bootstrap: int, semilla: int
) -> dict:
    """Agrega una lista de filas por caso con IC 95% bootstrap por medida.

    Args:
        mod: Modulo del evaluador v12 (bootstrap_ic_media).
        filas: Filas por caso con las medidas de MEDIDAS_C3.
        n_bootstrap: Numero de remuestreos bootstrap.
        semilla: Semilla del generador.

    Returns:
        Dict {"n": n, "medidas": {medida: {media, p2_5, p97_5, n}}}.
    """
    return {
        "n": len(filas),
        "medidas": {
            medida: mod.bootstrap_ic_media(
                np.asarray([fila[medida] for fila in filas], dtype=np.float64),
                n_bootstrap,
                semilla,
            )
            for medida in MEDIDAS_C3
        },
    }


def desglosar_por(
    mod: object, filas: list[dict], campo: str, n_bootstrap: int, semilla: int
) -> dict:
    """Agrupa filas por un campo (ano o tipo) y agrega cada grupo con IC.

    Args:
        mod: Modulo del evaluador v12.
        filas: Filas por caso.
        campo: Campo de agrupacion ("ano" o "tipo").
        n_bootstrap: Numero de remuestreos bootstrap.
        semilla: Semilla del generador.

    Returns:
        Dict {valor_del_campo: agregado con IC}.
    """
    grupos: dict = {}
    for fila in filas:
        grupos.setdefault(fila[campo], []).append(fila)
    return {
        clave: agregar_medidas(mod, grupo, n_bootstrap, semilla)
        for clave, grupo in sorted(grupos.items())
    }


def evaluar_ganadora(
    mod: object, casos: list[dict], estrategia: str, prior: dict, canones: list
) -> dict:
    """Evalua la ruta ganadora en los dos contextos del prior de hubs.

    Args:
        mod: Modulo del evaluador v12.
        casos: Casos con scores de la estrategia.
        estrategia: Nombre de la estrategia ganadora.
        prior: Dict {canon: valor} del prior de hubs.
        canones: Canones alineados con los vectores de scores.

    Returns:
        Dict {contexto: {alpha, global, por_anio, por_tipo, por_caso}}.
    """
    resultados = {}
    for nombre, alpha in CONFIG["contextos"].items():
        filas = evaluar_contexto(
            mod, casos, estrategia, prior, canones, alpha
        )
        resultados[nombre] = {
            "alpha": float(alpha),
            "global": agregar_medidas(
                mod, filas, CONFIG["bootstrap"], CONFIG["semilla"]
            ),
            "por_anio": desglosar_por(
                mod, filas, "ano", CONFIG["bootstrap"], CONFIG["semilla"]
            ),
            "por_tipo": desglosar_por(
                mod, filas, "tipo", CONFIG["bootstrap"], CONFIG["semilla"]
            ),
            "por_caso": filas,
        }
    return resultados


# =====================================================================
# SECCION 6: DERIVA TEMPORAL (HISTORICO vs 2022-2026)
# =====================================================================
def cargar_serie_historica(
    mod: object, bench: dict, estrategia: str, contexto: str
) -> tuple[list, list, str]:
    """Extrae la serie de nDCG@10 por caso del benchmark v13 congelado.

    Fuente primaria: comparacion_b2.metricas_por_caso[contexto][estrategia].
    ndcg_at_10 (alpha del contexto; con_hubs = 1.5, la ruta de producto).
    Fallback documentado si esa seccion no existiera: recalcular nDCG@10
    desde top10_por_caso (rankings de producto) y la referencia de la
    muestra.

    Args:
        mod: Modulo del evaluador v12.
        bench: Benchmark v13 cargado.
        estrategia: Nombre de la estrategia ganadora.
        contexto: Contexto del prior ("con_hubs" en la ruta de producto).

    Returns:
        Tupla (valores, anios, origen) alineada con bench["muestra"].

    Raises:
        ValueError: Si la serie no tiene el mismo largo que la muestra.
    """
    muestra = bench["muestra"]
    comparacion = (bench.get("comparacion_b2") or {}).get(
        "metricas_por_caso", {}
    )
    datos = (comparacion.get(contexto) or {}).get(estrategia)
    if datos and "ndcg_at_10" in datos:
        valores = [float(valor) for valor in datos["ndcg_at_10"]]
        origen = "comparacion_b2.metricas_por_caso"
    else:
        top10 = bench.get("top10_por_caso") or {}
        valores = [
            float(
                mod.ndcg_at_k(
                    list(caso["arts_cp"]),
                    [
                        str(entrada["id"])
                        for entrada in top10.get(str(caso["id"]), {}).get(
                            estrategia, []
                        )
                    ],
                    mod.CONFIG["k_ndcg"],
                )
            )
            for caso in muestra
        ]
        origen = "top10_por_caso (fallback)"
    if len(valores) != len(muestra):
        raise ValueError(
            f"serie historica con {len(valores)} valores != "
            f"{len(muestra)} casos de la muestra v13"
        )
    anios = [int(caso["ano"]) for caso in muestra]
    return valores, anios, origen


def agregar_por_periodo(
    mod: object, valores: list, anios: list, n_bootstrap: int, semilla: int
) -> dict:
    """Agrega la serie historica por periodo cronologico con IC 95%.

    Args:
        mod: Modulo del evaluador v12.
        valores: nDCG@10 por caso historico.
        anios: Anio por caso (mismo orden).
        n_bootstrap: Numero de remuestreos bootstrap.
        semilla: Semilla del generador.

    Returns:
        Dict {periodo: {n, media, p2_5, p97_5}}.
    """
    grupos: dict = {}
    for valor, anio in zip(valores, anios, strict=True):
        periodo = mod.asignar_periodo(anio)
        grupos.setdefault(periodo, []).append(valor)
    return {
        periodo: mod.bootstrap_ic_media(
            np.asarray(lista, dtype=np.float64), n_bootstrap, semilla
        )
        for periodo, lista in sorted(grupos.items())
        if periodo != "desconocido"
    }


def comparar_deriva(
    historico: dict, c3_global: dict, historico_global: dict
) -> dict:
    """Calcula deltas descriptivos C3 2022-2026 menos cada periodo historico.

    La comparacion NO es pareada (muestras, corpus y fuente de etiquetas
    distintas: M2 regex del benchmark vs NER silver del script 28); los
    deltas son descriptivos y se reportan junto a los ICs de ambos lados.

    Args:
        historico: Agregado por periodo historico.
        c3_global: Agregado global de la evaluacion C3 (contexto primario).
        historico_global: Agregado bootstrap de los 90 casos historicos.

    Returns:
        Dict {periodo_o_global: {n, historico, c3, delta}}.
    """
    c3 = float(c3_global["medidas"]["ndcg_at_10"]["media"])
    deltas = {
        periodo: {
            "n": int(datos["n"]),
            "historico": float(datos["media"]),
            "c3": c3,
            "delta": c3 - float(datos["media"]),
        }
        for periodo, datos in historico.items()
    }
    deltas["1992-2021"] = {
        "n": int(historico_global["n"]),
        "historico": float(historico_global["media"]),
        "c3": c3,
        "delta": c3 - float(historico_global["media"]),
    }
    return deltas


def graficar_deriva(
    historico: dict,
    c3_por_anio: dict,
    c3_global: dict,
    historico_global: dict,
    ruta: Path,
    contexto: str,
) -> None:
    """Genera la figura de deriva temporal (periodos historicos vs C3).

    Panel izquierdo: nDCG@10 medio por periodo historico y global 2022-2026
    con barras de IC 95% bootstrap. Panel derecho: nDCG@10 medio por anio del
    corpus C3 con IC y la linea de referencia del promedio historico.

    Args:
        historico: Agregado por periodo historico.
        c3_por_anio: Agregado por anio de la evaluacion C3.
        c3_global: Agregado global de la evaluacion C3.
        historico_global: Agregado bootstrap de los 90 casos historicos.
        ruta: Ruta de salida de la figura.
        contexto: Nombre del contexto del prior para el titulo.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    periodos = list(historico)
    etiquetas = periodos + [CONFIG["periodo_c3"]]
    medias = [float(historico[p]["media"]) for p in periodos] + [
        float(c3_global["medidas"]["ndcg_at_10"]["media"])
    ]
    bajos = [float(historico[p]["media"] - historico[p]["p2_5"]) for p in periodos]
    bajos.append(
        float(
            c3_global["medidas"]["ndcg_at_10"]["media"]
            - c3_global["medidas"]["ndcg_at_10"]["p2_5"]
        )
    )
    altos = [float(historico[p]["p97_5"] - historico[p]["media"]) for p in periodos]
    altos.append(
        float(
            c3_global["medidas"]["ndcg_at_10"]["p97_5"]
            - c3_global["medidas"]["ndcg_at_10"]["media"]
        )
    )
    tamanos = [int(historico[p]["n"]) for p in periodos] + [int(c3_global["n"])]
    colores = ["#4c72b0"] * len(periodos) + ["#dd8452"]

    figura, ejes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ejes[0].bar(
        range(len(etiquetas)),
        medias,
        yerr=[bajos, altos],
        color=colores,
        capsize=4,
        width=0.6,
    )
    for posicion, (media, tamano, alto) in enumerate(
        zip(medias, tamanos, altos, strict=True)
    ):
        ejes[0].text(
            posicion,
            media + alto + 0.04,
            f"{media:.3f}\nn={tamano}",
            ha="center",
            fontsize=9,
        )
    ejes[0].set_xticks(range(len(etiquetas)), etiquetas)
    ejes[0].set_ylabel("nDCG@10 medio (IC 95% bootstrap)")
    ejes[0].set_title("Periodos historicos vs corpus 2022-2026")
    ejes[0].set_ylim(0.0, 1.0)
    ejes[0].grid(axis="y", alpha=0.3)

    anios = sorted(int(anio) for anio in c3_por_anio)
    medias_anio = [
        float(c3_por_anio[anio]["medidas"]["ndcg_at_10"]["media"])
        for anio in anios
    ]
    bajos_anio = [
        float(
            c3_por_anio[anio]["medidas"]["ndcg_at_10"]["media"]
            - c3_por_anio[anio]["medidas"]["ndcg_at_10"]["p2_5"]
        )
        for anio in anios
    ]
    altos_anio = [
        float(
            c3_por_anio[anio]["medidas"]["ndcg_at_10"]["p97_5"]
            - c3_por_anio[anio]["medidas"]["ndcg_at_10"]["media"]
        )
        for anio in anios
    ]
    ejes[1].errorbar(
        anios,
        medias_anio,
        yerr=[bajos_anio, altos_anio],
        marker="o",
        color="#dd8452",
        capsize=4,
    )
    ejes[1].axhline(
        float(historico_global["media"]),
        color="#4c72b0",
        linestyle="--",
        label=f"historico 1992-2021 ({historico_global['media']:.3f})",
    )
    for anio, media, datos in zip(anios, medias_anio, (
        c3_por_anio[anio] for anio in anios
    ), strict=True):
        ejes[1].annotate(
            f"n={datos['n']}",
            (anio, media),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )
    ejes[1].set_xticks(anios)
    ejes[1].set_xlabel("Anio del corpus C3")
    ejes[1].set_ylabel("nDCG@10 medio (IC 95% bootstrap)")
    ejes[1].set_title("Corpus 2022-2026 por anio")
    ejes[1].set_ylim(0.0, 1.0)
    ejes[1].grid(axis="y", alpha=0.3)
    ejes[1].legend(loc="lower right", fontsize=9)

    figura.suptitle(
        "Deriva temporal del vinculo sentencia-articulo CP (nDCG@10)\n"
        f"Estrategia v13 {CONFIG['estrategia']} | contexto {contexto} "
        f"(alpha={CONFIG['contextos'][contexto]})"
    )
    figura.tight_layout()
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(figura)


# =====================================================================
# SECCION 7: SALIDAS, CLI Y ORQUESTACION
# =====================================================================
def resolver_rutas_salida(limite: int | None) -> dict:
    """Resuelve las rutas de salida segun el modo (real o smoke/limite).

    Args:
        limite: Valor efectivo de --limite (None en el run real).

    Returns:
        Dict con rutas json/figura/log y el flag sobrescribir.
    """
    if limite is None:
        return {
            "json": ROOT / CONFIG["ruta_salida_json"],
            "figura": ROOT / CONFIG["ruta_salida_figura"],
            "log": ROOT / CONFIG["ruta_log"],
            "sobrescribir": False,
        }
    base = Path(CONFIG["dir_smoke"])
    return {
        "json": base / "resultados" / "c3_eval_temporal_smoke.json",
        "figura": base / "figuras" / "c3_deriva_temporal_smoke.png",
        "log": base / "experimentos" / "29_c3_eval_temporal.log",
        "sobrescribir": True,
    }


def comprobar_salidas(rutas: dict) -> None:
    """Impide sobrescribir artefactos reales de corridas previas.

    Args:
        rutas: Dict de rutas resuelto por resolver_rutas_salida.

    Raises:
        FileExistsError: Si el JSON o la figura reales ya existen.
    """
    if rutas["sobrescribir"]:
        return
    for clave, descripcion in (("json", "JSON"), ("figura", "figura")):
        if rutas[clave].exists():
            raise FileExistsError(
                f"ya existe el {descripcion} {rutas[clave]}: no se "
                "sobrescribe; renombra o elimina el artefacto antes de "
                "re-ejecutar"
            )


def guardar_json_nuevo(ruta: Path, contenido: dict, sobrescribir: bool) -> None:
    """Guarda el JSON de resultados (sin sobrescribir en modo real).

    Args:
        ruta: Ruta de salida.
        contenido: Dict serializable.
        sobrescribir: True en modo smoke (directorio scratch).

    Raises:
        FileExistsError: Si la ruta existe y no se permite sobrescribir.
    """
    if ruta.exists() and not sobrescribir:
        raise FileExistsError(f"no se sobrescribe {ruta}")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8") as manejador:
        json.dump(contenido, manejador, ensure_ascii=False, indent=2)


def parsear_args() -> argparse.Namespace:
    """Define y parsea la interfaz de linea de comandos.

    Returns:
        Namespace con evaluar, por_anio, limite y smoke.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluacion C3 de deriva temporal de la ruta ganadora v13 "
            "(hibrido B2 top-50 + rerank B1 + hubs alpha=1.5) sobre el "
            "corpus 2022-2026 frente al benchmark historico M7."
        )
    )
    parser.add_argument(
        "--evaluar",
        action="store_true",
        help="Ejecuta la evaluacion completa (sin este flag el script solo informa).",
    )
    parser.add_argument(
        "--por-anio",
        type=int,
        default=CONFIG["por_anio"],
        help=f"Maximo de casos por anio en la muestra (default {CONFIG['por_anio']}).",
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help=(
            "Recorta la muestra a N casos y redirige las salidas a "
            f"{CONFIG['dir_smoke']}/ (prueba de fontaneria, nunca toca artefactos reales)."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Atajo de --limite {CONFIG['smoke_limite']} con salidas en {CONFIG['dir_smoke']}/.",
    )
    args = parser.parse_args()
    if args.por_anio < 1:
        parser.error("--por-anio debe ser >= 1")
    if args.limite is not None and args.limite < 1:
        parser.error("--limite debe ser >= 1")
    return args


def _ejecutar(args: argparse.Namespace, rutas: dict, limite: int | None) -> None:
    """Orquesta la evaluacion C3 completa (secciones 1 a 7).

    Args:
        args: Namespace del CLI.
        rutas: Dict de rutas de salida.
        limite: Limite efectivo de casos (None en el run real).
    """
    inicio_total = time.time()
    print("=" * 78)
    print("29_c3_eval_temporal.py | evaluacion C3 de deriva temporal")
    print(
        f"fecha: {datetime.now(timezone.utc).isoformat()} | "
        f"modo: {'smoke/limite' if limite is not None else 'real'} | "
        f"semilla: {CONFIG['semilla']}"
    )
    print(f"salidas: {rutas['json']} | {rutas['figura']} | {rutas['log']}")
    print("=" * 78)

    mod = importar_evaluador_v12(ROOT / CONFIG["ruta_script_12"])
    mod.configurar_semillas(CONFIG["semilla"])
    modelo_b2, modelo_b1 = resolver_modelos_ganadora(mod)
    print(f"Ruta ganadora: {CONFIG['estrategia']}")
    print(f"Retriever B2: {modelo_b2}")
    print(f"Reranker B1: {modelo_b1}")
    comprobar_salidas(rutas)

    ruta_corpus = ROOT / CONFIG["ruta_corpus_c3"]
    ruta_entidades = ROOT / CONFIG["ruta_entidades_c3"]
    ruta_benchmark = ROOT / CONFIG["ruta_benchmark_v13"]
    exigir_archivo(ruta_corpus, "corpus 2022-2026 (script 28)")
    exigir_archivo(ruta_entidades, "entidades NER 2022-2026 (script 28)")
    exigir_archivo(ruta_benchmark, "benchmark v13 congelado")

    print("[1/6] Insumos y pasajes permanentes de la CP (solo lectura)")
    corpus = cargar_corpus_ligero(ruta_corpus)
    pasajes, canones, grupos_indices, canon_valido = preparar_pasajes(mod)
    referencia, stats_ref = cargar_referencia_silver(ruta_entidades, canon_valido)
    bench = cargar_benchmark_v13(ruta_benchmark)
    prior = mod.cargar_o_construir_prior_hubs()
    n_ok = int((corpus["estado"] == CONFIG["estado_ok"]).sum())
    print(
        f"Corpus: {len(corpus)} docs ({n_ok} ok) | entidades: "
        f"{stats_ref['n_docs']} docs ({stats_ref['n_sin_cita_cp']} sin cita CP)"
    )
    print(
        f"CP: {len(canones)} canones indexados / {len(pasajes)} pasajes | "
        f"gold valido: {len(canon_valido)} canones | prior: {len(prior)}"
    )
    print(
        f"Benchmark v13: {len(bench['muestra'])} casos historicos | "
        f"serie en comparacion_b2 (con_hubs=1.5)"
    )

    print(f"[2/6] Muestra estratificada por anio (por_anio={args.por_anio})")
    muestra, stats_muestra = seleccionar_muestra(
        mod, corpus, referencia, args.por_anio, CONFIG["semilla"]
    )
    muestra = limitar_muestra(muestra, limite)
    stats_muestra["seleccion"] = resumen_conteos(muestra)
    if limite is not None:
        print(
            f"[smoke] --limite/--smoke: muestra recortada a {len(muestra)} "
            "casos (primeros por orden anio, id; NO es el run final)"
        )
    print(
        f"Elegibles: {stats_muestra['n_elegibles']} | seleccionados: "
        f"{stats_muestra['seleccion']['n']} | por anio: "
        f"{stats_muestra['seleccion']['por_anio']} | por tipo: "
        f"{stats_muestra['seleccion']['por_tipo']}"
    )

    print("[3/6] Casos, textos y BM25 por pasajes")
    textos = mod.leer_filas_por_id(
        str(ruta_corpus), ["id", "texto"], muestra["id"].tolist()
    )
    textos_por_id = dict(
        zip(textos["id"].astype(str), textos["texto"], strict=True)
    )
    casos = construir_casos(mod, muestra, textos_por_id, referencia)
    seg_bm25 = calcular_bm25(mod, casos, pasajes, canones, grupos_indices)
    print(
        f"Casos: {len(casos)} | BM25: {seg_bm25:.1f} s | media chunks "
        f"{np.mean([len(caso['chunks_bm25']) for caso in casos]):.1f}"
    )

    print("[4/6] Modelos de la ruta ganadora (GPU con guard de VRAM)")
    from lib.reranker_bge import RerankerBGE, elegir_dispositivo

    dispositivo = elegir_dispositivo()
    print(
        f"Dispositivo: {dispositivo} | torch.cuda disponible: "
        f"{torch.cuda.is_available()}"
    )
    stats_b2, clave_hibrido = calcular_denso_b2(
        mod, casos, pasajes, canones, grupos_indices, dispositivo, modelo_b2
    )
    reranker = RerankerBGE(modelo_id=modelo_b1, dispositivo=dispositivo)
    try:
        stats_rerank = rerankear_ganadora(
            mod,
            casos,
            pasajes,
            canones,
            grupos_indices,
            reranker,
            CONFIG["estrategia"],
            CONFIG["k_candidatos"],
            clave_hibrido,
        )
    finally:
        del reranker
        if dispositivo == "cuda":
            torch.cuda.empty_cache()
    print(
        f"Rerank B1: {stats_rerank['pares_por_caso']} pares/caso en "
        f"{stats_rerank['segundos']} s | max tokens par: "
        f"{stats_rerank['max_tokens_par']}"
    )

    print("[5/6] Evaluacion (metricas + IC bootstrap) y deriva temporal")
    evaluacion = evaluar_ganadora(
        mod, casos, CONFIG["estrategia"], prior, canones
    )
    primario = evaluacion[CONFIG["contexto_primario"]]
    print(
        f"C3 {CONFIG['contexto_primario']} nDCG@10="
        f"{primario['global']['medidas']['ndcg_at_10']['media']:.3f} "
        f"[{primario['global']['medidas']['ndcg_at_10']['p2_5']:.3f}, "
        f"{primario['global']['medidas']['ndcg_at_10']['p97_5']:.3f}]"
    )
    valores_hist, anios_hist, origen_hist = cargar_serie_historica(
        mod, bench, CONFIG["estrategia"], CONFIG["contexto_primario"]
    )
    historico = agregar_por_periodo(
        mod, valores_hist, anios_hist, CONFIG["bootstrap"], CONFIG["semilla"]
    )
    historico_global = mod.bootstrap_ic_media(
        np.asarray(valores_hist, dtype=np.float64),
        CONFIG["bootstrap"],
        CONFIG["semilla"],
    )
    deltas = comparar_deriva(historico, primario["global"], historico_global)
    print(f"Serie historica: {origen_hist} ({historico_global['n']} casos)")
    for periodo, datos in historico.items():
        print(
            f"  {periodo}: nDCG@10={datos['media']:.3f} "
            f"[{datos['p2_5']:.3f}, {datos['p97_5']:.3f}] (n={datos['n']})"
        )
    for periodo, datos in deltas.items():
        print(
            f"  delta C3 - {periodo}: {datos['delta']:+.3f} "
            "(descriptivo, NO pareado)"
        )

    print("[6/6] Figura y guardado de resultados")
    graficar_deriva(
        historico,
        primario["por_anio"],
        primario["global"],
        historico_global,
        rutas["figura"],
        CONFIG["contexto_primario"],
    )
    salida = {
        "fecha": datetime.now(timezone.utc).isoformat(),
        "version": CONFIG["version"],
        "script": "scripts/29_c3_eval_temporal.py",
        "descripcion": (
            "Deriva temporal de la ruta ganadora v13 "
            f"({CONFIG['estrategia']}) sobre el corpus 2022-2026 (citas "
            "NER silver del script 28) frente al benchmark historico M7 "
            "de 90 casos (1992-2021) congelado en v13."
        ),
        "terminologia": (
            "La referencia de relevancia son citas NER silver; 'gold' "
            "queda reservado para anotacion humana (regla del proyecto)."
        ),
        "modo": {"smoke_o_limite": limite is not None, "limite": limite},
        "config": {
            "semilla": CONFIG["semilla"],
            "por_anio": args.por_anio,
            "estrategia": CONFIG["estrategia"],
            "contextos": dict(CONFIG["contextos"]),
            "contexto_primario": CONFIG["contexto_primario"],
            "k_candidatos": CONFIG["k_candidatos"],
            "k_medidas": CONFIG["k_medidas"],
            "k_ndcg": CONFIG["k_ndcg"],
            "bootstrap": CONFIG["bootstrap"],
            "incluir_transitorios": False,
            "chunk_bm25_chars": mod.CONFIG["chunk_bm25_chars"],
            "solape_bm25_chars": mod.CONFIG["solape_bm25_chars"],
            "tamano_chunk": mod.CONFIG["tamano_chunk"],
            "solape_chunk": mod.CONFIG["solape_chunk"],
            "pasaje_max_palabras": mod.CONFIG["pasaje_max_palabras"],
            "k_chunks_rerank": mod.CONFIG["k_chunks_rerank"],
            "k_rerank_salida": mod.CONFIG["k_rerank_salida"],
            "pesos_hibrido": dict(mod.CONFIG["pesos_hibrido"]),
            "ventana_query_chars": mod.CONFIG["ventana_query_chars"],
            "nota": (
                "Protocolo M7 reutilizado por importlib del script 12; "
                "el prior de hubs se aplica con ranking_con_prior (misma "
                "funcion del run v13) y las salidas de la estrategia se "
                "truncan a k=10."
            ),
        },
        "insumos": {
            "corpus_c3": str(ruta_corpus),
            "entidades_c3": str(ruta_entidades),
            "benchmark_v13": str(ruta_benchmark),
            "docs_corpus": len(corpus),
            "docs_ok": n_ok,
            "docs_elegibles_cita_cp": int(stats_muestra["n_elegibles"]),
            "docs_entidades": int(stats_ref["n_docs"]),
            "docs_entidades_sin_cita_cp": int(stats_ref["n_sin_cita_cp"]),
            "canones_indexados": len(canones),
            "canones_validos_gold": len(canon_valido),
            "pasajes": len(pasajes),
            "prior_hubs_canones": len(prior),
            "benchmark_casos": len(bench["muestra"]),
        },
        "muestra": {
            **stats_muestra,
            "casos": [
                {
                    "id": caso["id"],
                    "tipo": caso["tipo"],
                    "ano": caso["ano"],
                    "n_arts": len(caso["arts"]),
                    "arts_cp": caso["arts"],
                }
                for caso in casos
            ],
        },
        "evaluacion": evaluacion,
        "deriva_temporal": {
            "fuente_historica": {
                "archivo": str(ruta_benchmark),
                "estrategia": CONFIG["estrategia"],
                "contexto": CONFIG["contexto_primario"],
                "alpha": CONFIG["contextos"][CONFIG["contexto_primario"]],
                "origen_serie": origen_hist,
                "n_casos": int(historico_global["n"]),
                "nota": (
                    "La serie historica sale de comparacion_b2."
                    "metricas_por_caso del v13 (contexto con_hubs, "
                    "alpha=1.5, la ruta de producto). El global/por_periodo "
                    "del v13 corresponde al run con alpha=0 (sin "
                    "--hubs-alpha), por lo que no se usa como fuente."
                ),
            },
            "historico_por_periodo": historico,
            "historico_global": historico_global,
            "c3_2022_2026": primario["global"],
            "delta_descriptivo": deltas,
            "nota": (
                "Delta C3 2022-2026 menos cada periodo historico. NO es "
                "pareado: las muestras, el corpus y la fuente de etiquetas "
                "difieren (M2 regex del benchmark vs NER silver del script "
                "28); se reporta como evidencia descriptiva con los ICs de "
                "ambos lados."
            ),
        },
        "rerank_stats": stats_rerank,
        "b2_stats": stats_b2,
        "tiempos_segundos": {
            "bm25": round(seg_bm25, 1),
            "b2_pasajes": stats_b2.get("segundos_pasajes"),
            "b2_query": stats_b2.get("segundos_query"),
            "b2_total": stats_b2.get("segundos_total"),
            "rerank_b1": stats_rerank["segundos"],
            "total": round(time.time() - inicio_total, 1),
        },
    }
    guardar_json_nuevo(rutas["json"], salida, rutas["sobrescribir"])
    print(f"JSON: {rutas['json']}")
    print(f"Figura: {rutas['figura']}")

    globales = primario["global"]["medidas"]
    print("-" * 78)
    print(f"Resumen C3 ({CONFIG['contexto_primario']}, n={primario['global']['n']}):")
    print(
        f"  nDCG@10={globales['ndcg_at_10']['media']:.3f} | "
        f"F1@5={globales['f1_at_5']['media']:.3f} | "
        f"F1@10={globales['f1_at_10']['media']:.3f} | "
        f"MRR={globales['mrr']['media']:.3f} | "
        f"P@5={globales['p_at_5']['media']:.3f} | "
        f"R@5={globales['r_at_5']['media']:.3f}"
    )
    print(
        "  historico 1992-2021: nDCG@10="
        f"{historico_global['media']:.3f} "
        f"[{historico_global['p2_5']:.3f}, {historico_global['p97_5']:.3f}]"
    )
    print(
        "  delta C3 - historico: "
        f"{deltas['1992-2021']['delta']:+.3f} (descriptivo, no pareado)"
    )
    print(f"  tiempo total: {salida['tiempos_segundos']['total']} s")
    print("=" * 78)


def main() -> None:
    """Punto de entrada: CLI, log duplicado y manejo de errores."""
    args = parsear_args()
    if not args.evaluar:
        print(
            "Nada que ejecutar: usa --evaluar para lanzar la evaluacion C3 "
            "(--help para opciones)."
        )
        return
    limite = (
        args.limite
        if args.limite is not None
        else (CONFIG["smoke_limite"] if args.smoke else None)
    )
    rutas = resolver_rutas_salida(limite)
    rutas["log"].parent.mkdir(parents=True, exist_ok=True)
    with (
        rutas["log"].open("w", encoding="utf-8") as archivo,
        contextlib.redirect_stdout(_DuplicadorSalida(archivo)),
        contextlib.redirect_stderr(_DuplicadorSalida(archivo)),
    ):
        try:
            _ejecutar(args, rutas, limite)
        except Exception as exc:
            # El traceback va al log via stderr redirigido; la marca evita
            # que el manejador externo lo duplique en consola.
            traceback.print_exc()
            exc._reportado_c3 = True
            raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - reporta el fallo y sale con codigo 1
        if not getattr(exc, "_reportado_c3", False):
            traceback.print_exc()
        sys.exit(1)
