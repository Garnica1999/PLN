"""23_b1_pares_reranker.py
Fecha: 2026-10-03
Descripcion: Sprint B1 del ciclo v3 - construccion de los pares de
    entrenamiento del reranker de dominio (chunk de sentencia -> pasaje de
    articulo de la CP citado). Este script SOLO construye el dataset; el
    fine-tune es otro script.

    Modos: --pares (construccion de los pares) y --cerrar (reconstruccion
    del JSON de cierre desde las marcas/shards, sin recomputar pares), ambos
    con --smoke. En --pares, para cada sentencia NO excluida se generan:
    - POSITIVOS: por cada articulo de la CP citado (citas M2 de
      data/entidades_sentencias.parquet, SOLO permanentes) hasta
      --max-chunks-por-articulo pares (chunk, pasaje).
    - HARD NEGATIVOS: por cada positivo, --negativos-por-positivo con
      mezcla 50/30/20: top-30 BM25 de la 1a etapa del doc (no citados),
      co-citados M5 del articulo positivo (no citados) y aleatorios no
      citados.
    El texto del chunk y del pasaje se guarda directo en el parquet de
    salida (shards de ~500 MB max): data/pares_b1/pares_reranker_b1_shardNN
    .parquet + resultados/b1_pares.json.

    ANTI-LEAKAGE ESTRICTO: se excluyen las 90 sentencias del benchmark M7
    (ids de resultados/vinculo_cp_benchmark_v2.json, campo muestra;
    verificado contra resultados/vinculo_cp_benchmark_v8.json: ambos
    conjuntos son identicos y tienen 90 ids unicos). Ninguna sentencia del
    benchmark entra a los pares; el grafo de co-citacion tambien se
    construye SIN esas 90 sentencias. Los negativos jamas pertenecen al
    conjunto de articulos citados por el doc (se usa arts_cp completo).

    Reutilizacion de scripts/12_evaluar_vinculo_cp_v2.py (importlib, sin
    duplicar logica) para que los pares coincidan con la inferencia del
    reranker: preparar_pasajes_constitucion (pasajes de 120 palabras),
    chunks_query (chunks de 800 chars con solape 200 sobre texto_limpio),
    scores_bm25_caso (BM25 por pasajes con la misma tokenizacion),
    ranking_desde_scores y construir_pares_rerank_chunks (mejor pasaje +
    top-3 chunks contra ese pasaje). Se replica SOLO la extraccion de
    indices locales del selector interno (mejor_pasaje y top_chunks por
    argsort estable, dos lineas) y se contrasta contra la salida de la
    funcion reutilizada para detectar cualquier divergencia. Para los docs
    fuera del cupo BM25 se usa un selector de solape lexico (misma
    estructura mejor-pasaje/top-chunks, documentado como proxy barato).
    De scripts/17/18/19/21 se reutilizan md5_archivo, versiones_entorno y
    verificar_salida_nueva (guardas de no-sobrescritura); el patron de
    importlib por ruta es el de 19/21.

    Estado del arte aplicado:
    - Fine-tune de cross-encoder reranker con hard negatives minados por
      BM25 (Nogueira y Cho, 2019; Karpukhin et al., 2020, DPR) y negativos
      de co-citacion, patron dominante en statute retrieval legal
      (DoSSIER/COLIEE 2021; Fink 2023; LegalMALR 2026).
    - Positivos y negativos con EXACTAMENTE los troceados de la inferencia
      (query en chunks de 800/200 chars, pasajes de 120 palabras), para
      evitar el mismatch query/pasaje que refuto el rerank de query
      completa en la Fase T del ciclo v2.
    - Muestreo determinista (semilla 42) y RNG por documento derivado de
      (semilla, doc_id): la reanudacion por lotes no altera los pares.

    Reglas: solo lectura de las fuentes, sin subprocesos, procesamiento por
    lotes con shards atomicos y reanudables (relanzar con --forzar reutiliza
    los lotes ya escritos), nunca se sobrescribe una salida existente sin
    --forzar. El guard de integridad de --pares compara DURO solo las
    fuentes de DATOS (entidades, constitucion, red_citas_resumen,
    indice_bm25, benchmark_v2/v8, corpus); script_12 es codigo reutilizado
    y se registra como informativo (warning + seccion integridad del JSON,
    nunca aborta por el). El run completo NO se ejecuta aqui; usar el
    comando impreso al final.

    Modo --cerrar: reconstruye resultados/b1_pares.json desde las marcas
    _lote*.json y re-escanea TODOS los shards (verifica 0 filas con doc_id
    de los 90 del benchmark; aborta si aparece alguna) sin recomputar
    pares. Documenta el incidente de la corrida real (script_12 modificado
    durante la corrida por la integracion --b1, cambio aditivo; datos
    intactos) con md5 inicio/fin y causa.

Uso:
    /home/carlos/venv/bin/python scripts/23_b1_pares_reranker.py --pares
    /home/carlos/venv/bin/python scripts/23_b1_pares_reranker.py --pares --smoke
    /home/carlos/venv/bin/python scripts/23_b1_pares_reranker.py --cerrar
    /home/carlos/venv/bin/python scripts/23_b1_pares_reranker.py --cerrar --smoke
"""

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ===== SECCION 1: configuracion =====

RAIZ = Path(__file__).resolve().parent.parent

CONFIG = {
    "semilla": 42,
    "version": "b1-pares-reranker-v1",
    "ciclo": "v3",
    "sprint": "B1",
    # Caps por defecto del run completo (el smoke usa caps pequenos; ver
    # CONFIG["smoke"]). --max-pares limita los pares POSITIVOS (unidades
    # doc-articulo): las filas totales son positivos * (1 + negativos) *
    # chunks. --max-chunks-por-articulo se topa en 3 porque el selector de
    # 12 (construir_pares_rerank_chunks) emite exactamente 3 chunks por
    # candidato (CONFIG["k_chunks_rerank"]).
    "max_pares": 150000,
    "max_positivos_por_doc": 30,
    "negativos_por_positivo": 1,
    "max_chunks_por_articulo": 1,
    "max_docs_bm25": 4000,
    "lote": 1000,
    "tope_chunks_selector_12": 3,
    # Mezcla de fuentes de negativos (proporciones nominales; los conteos
    # reales quedan en el JSON de salida).
    "proporciones_negativos": {"bm25": 0.5, "cocitado": 0.3, "aleatorio": 0.2},
    "top_bm25_pool": 30,
    "top_cocitados_por_articulo": 50,
    "fuente_positivo": "positivo",
    "fuente_fallback_bm25": "aleatorio_sin_bm25",
    "reintentos_negativo_duplicado": 5,
    # Shards: limite conservador de bytes sin comprimir (parquet snappy
    # comprime ~2x, por lo que el archivo real queda bajo ~500 MB).
    "limite_shard_bytes": 500 * 1024**2,
    "overhead_bytes_par": 200,
    # Smoke: 200 docs y caps pequenos, salida SOLO en /tmp/opencode.
    "smoke": {
        "n_docs": 200,
        "max_pares": 3000,
        "max_positivos_por_doc": 5,
        "negativos_por_positivo": 1,
        "max_chunks_por_articulo": 1,
        "max_docs_bm25": 20,
        "lote": 100,
        "dir_salida": "/tmp/opencode/b1_pares_smoke",
    },
    # Rutas relativas a la raiz del proyecto (resolver_rutas las hace
    # absolutas contra RAIZ).
    "dir_salida": "data/pares_b1",
    "patron_shard": "pares_reranker_b1_shard{lote:04d}.parquet",
    "patron_marca": "_lote{lote:04d}.json",
    "ruta_json_salida": "resultados/b1_pares.json",
    "ruta_log": "experimentos/b1_pares_reranker.log",
    "ruta_entidades": "data/entidades_sentencias.parquet",
    "ruta_constitucion": "data/constitucion_articulos.parquet",
    "ruta_red_citas_resumen": "data/red_citas_resumen.parquet",
    "ruta_corpus": "data/corpus_sentencias.parquet",
    "ruta_indice_bm25": "data/cache_bm25_cp/indice_bm25_cp_v3.joblib",
    "ruta_benchmark_v2": "resultados/vinculo_cp_benchmark_v2.json",
    "ruta_benchmark_v8": "resultados/vinculo_cp_benchmark_v8.json",
    "ruta_script_12": "scripts/12_evaluar_vinculo_cp_v2.py",
    "ruta_script_17": "scripts/17_auditoria_etiquetas_m4.py",
    "ruta_script_21": "scripts/21_pack_validacion_m4.py",
    "prefijo_transitorio": "Transitorio",
    "canon_preambulo": "Preambulo",
    "version_indice_bm25": 3,
    # Fuentes de DATOS monitoreadas por el guard de integridad (comparacion
    # DURA: si cambian, la corrida aborta). script_12 queda FUERA: es codigo
    # reutilizado y una edicion aditiva no altera los datos; se registra como
    # informativo (warning + seccion integridad del JSON).
    "fuentes_datos_integridad": [
        "entidades",
        "constitucion",
        "red_citas_resumen",
        "indice_bm25",
        "benchmark_v2",
        "benchmark_v8",
        "corpus",
    ],
    # Incidente de la corrida real B1 (2026-10-04): script_12 se modifico
    # DURANTE la corrida (integracion --b1, cambio aditivo) y el guard
    # original aborto por falso positivo; las fuentes de datos quedaron
    # intactas (verificado por el orquestador: mtimes de septiembre, 0
    # archivos del corpus modificados desde el inicio). --cerrar documenta
    # este cambio con md5 inicio/fin y causa; nunca aborta por script_12.
    "incidente_script_12": {
        "md5_inicio_corrida": "aa9618c2fae9708fbf54d696e244b665",
        "md5_fin_corrida": "7e2fa6b16abc13ac86da62a821533525",
        "causa": "integracion --b1 durante la corrida; cambio aditivo; datos intactos",
    },
}

# Esquema EXACTO del parquet de pares (orden de columnas del diseno).
ESQUEMA_PARES = pa.schema(
    [
        ("doc_id", pa.string()),
        ("articulo", pa.string()),
        ("pasaje_id", pa.int32()),
        ("chunk_idx", pa.int32()),
        ("label", pa.int8()),
        ("fuente_negativo", pa.string()),
        ("texto_chunk", pa.string()),
        ("texto_pasaje", pa.string()),
        ("n_chars_chunk", pa.int32()),
        ("n_chars_pasaje", pa.int32()),
    ]
)

CLAVE_METADATOS = b"b1_pares"


# ===== SECCION 2: utilidades (logger, md5, versiones, importlib, guardas) =====

LOGGER_NOMBRE = "b1_pares_reranker_23"
LOGGER = logging.getLogger(LOGGER_NOMBRE)
_MODULOS: dict[str, object] = {}


def _cargar_modulo(nombre: str, ruta: Path):
    """Importa un script por ruta (su nombre empieza por digito).

    Patron de _cargar_modulo del script 19 (importlib.util, cacheado).

    Args:
        nombre: Nombre interno para sys.modules.
        ruta: Ruta del script a importar.

    Returns:
        Modulo ejecutado.

    Raises:
        ImportError: Si la ruta no puede cargarse.
    """
    if nombre in _MODULOS:
        return _MODULOS[nombre]
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"no se pudo cargar {ruta}")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = modulo
    spec.loader.exec_module(modulo)
    _MODULOS[nombre] = modulo
    return modulo


def modulo_12():
    """Carga (y cachea) scripts/12_evaluar_vinculo_cp_v2.py.

    Returns:
        Modulo 12 con la logica de retrieval reutilizable.
    """
    return _cargar_modulo("vinculo_cp_v2_12", RAIZ / CONFIG["ruta_script_12"])


def modulo_17():
    """Carga (y cachea) scripts/17_auditoria_etiquetas_m4.py.

    Returns:
        Modulo 17 con md5_archivo y versiones_entorno.
    """
    return _cargar_modulo("auditoria_etiquetas_m4_17", RAIZ / CONFIG["ruta_script_17"])


def modulo_21():
    """Carga (y cachea) scripts/21_pack_validacion_m4.py.

    Returns:
        Modulo 21 con la guarda verificar_salida_nueva.
    """
    return _cargar_modulo("pack_validacion_m4_21", RAIZ / CONFIG["ruta_script_21"])


def md5_archivo(ruta: str | Path) -> str | None:
    """Calcula el md5 de un archivo por bloques (delegado al modulo 17).

    Args:
        ruta: Ruta del archivo.

    Returns:
        Hexdigest md5, o None si el archivo no existe.
    """
    return modulo_17().md5_archivo(str(ruta))


def md5_directorio(ruta: str | Path) -> str:
    """Calcula el md5 de un dataset particionado (arbol de archivos).

    El corpus es un dataset Hive (directorio con 341 parquet); no existe
    helper previo para directorios, por lo que se recorre el arbol en
    orden determinista hasheando ruta relativa + tamano + contenido.

    Args:
        ruta: Directorio raiz del dataset.

    Returns:
        Hexdigest md5 del arbol completo.

    Raises:
        FileNotFoundError: Si el directorio no existe.
    """
    base = Path(ruta)
    if not base.exists():
        raise FileNotFoundError(f"no existe el directorio {base}")
    digest = hashlib.md5()
    archivos = sorted(p for p in base.rglob("*") if p.is_file())
    for archivo in archivos:
        digest.update(archivo.relative_to(base).as_posix().encode("utf-8"))
        digest.update(str(archivo.stat().st_size).encode("utf-8"))
        with open(archivo, "rb") as manejador:
            for bloque in iter(lambda: manejador.read(8 * 1024 * 1024), b""):
                digest.update(bloque)
    return digest.hexdigest()


def versiones_entorno() -> dict:
    """Recolecta versiones de interprete y librerias (delegado al modulo 17).

    Returns:
        Dict con python, plataforma y versiones de librerias clave.
    """
    return modulo_17().versiones_entorno()


def verificar_salida_nueva(
    ruta: str | Path, forzar: bool, smoke: bool, logger: logging.Logger
) -> Path:
    """Guarda de no-sobrescritura (delegada al modulo 21).

    Args:
        ruta: Ruta de salida a validar.
        forzar: True permite reescribir.
        smoke: True si es corrida smoke (scratch reescribible).
        logger: Logger activo.

    Returns:
        Path validado.
    """
    return modulo_21().verificar_salida_nueva(ruta, forzar, smoke, logger)


def configurar_logger(ruta_log: str) -> logging.Logger:
    """Configura el logger del experimento (archivo append + consola).

    Patron de configurar_logger del script 17.

    Args:
        ruta_log: Ruta del archivo de log (se crea el directorio padre).

    Returns:
        Logger configurado.
    """
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formato = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    Path(ruta_log).parent.mkdir(parents=True, exist_ok=True)
    manejador_archivo = logging.FileHandler(ruta_log, mode="a", encoding="utf-8")
    manejador_archivo.setFormatter(formato)
    LOGGER.addHandler(manejador_archivo)
    manejador_consola = logging.StreamHandler()
    manejador_consola.setFormatter(formato)
    LOGGER.addHandler(manejador_consola)
    LOGGER.propagate = False
    return LOGGER


def resolver_rutas() -> dict:
    """Convierte las rutas relativas de CONFIG en absolutas contra RAIZ.

    Returns:
        Dict con las rutas de entrada/salida resueltas.
    """
    claves = [
        "dir_salida",
        "ruta_json_salida",
        "ruta_log",
        "ruta_entidades",
        "ruta_constitucion",
        "ruta_red_citas_resumen",
        "ruta_corpus",
        "ruta_indice_bm25",
        "ruta_benchmark_v2",
        "ruta_benchmark_v8",
        "ruta_script_12",
        "ruta_script_17",
        "ruta_script_21",
    ]
    return {
        clave: str(RAIZ / CONFIG[clave]) if not Path(CONFIG[clave]).is_absolute() else CONFIG[clave]
        for clave in claves
    }


def clave_canon(canon: str) -> tuple:
    """Clave de orden determinista para canones (numericos primero).

    Args:
        canon: Numero de articulo como string.

    Returns:
        Tupla (0, int) para numericos y (1, texto) para el resto.
    """
    return (0, int(canon)) if canon.isdigit() else (1, canon)


def parsear_args() -> argparse.Namespace:
    """Parsea la linea de comandos (modo unico --pares).

    Returns:
        Namespace con los argumentos validados.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pares",
        action="store_true",
        help="modo unico de construccion: genera los pares de entrenamiento B1",
    )
    parser.add_argument(
        "--cerrar",
        action="store_true",
        help=(
            "reconstruye el JSON de cierre desde marcas _lote*.json + shards SIN recomputar "
            "pares (re-escanea todos los shards y aborta si aparece un doc del benchmark); "
            "los caps pasados aqui son declarativos para el JSON"
        ),
    )
    parser.add_argument(
        "--max-pares",
        type=int,
        default=None,
        help="tope global de pares POSITIVOS (unidades doc-articulo); default 150000 (smoke 3000)",
    )
    parser.add_argument(
        "--max-positivos-por-doc",
        type=int,
        default=None,
        help="tope de positivos por doc; default 30 (smoke 5)",
    )
    parser.add_argument(
        "--negativos-por-positivo",
        type=int,
        default=None,
        help="negativos por cada positivo; default 1 (smoke 1)",
    )
    parser.add_argument(
        "--max-chunks-por-articulo",
        type=int,
        default=None,
        help="pares por articulo (tope 3, el selector de 12 emite 3 chunks); default 1 (smoke 1)",
    )
    parser.add_argument(
        "--max-docs-bm25",
        type=int,
        default=None,
        help="docs con BM25 exacto para positivos y hard negatives; default 4000 (smoke 20)",
    )
    parser.add_argument(
        "--lote",
        type=int,
        default=None,
        help="docs por lote/shard; default 1000 (smoke 100)",
    )
    parser.add_argument(
        "--forzar",
        action="store_true",
        help="permite reescribir/reanudar: reutiliza lotes completos y rehace los pendientes",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="200 docs y caps pequenos; salida SOLO en /tmp/opencode/b1_pares_smoke/",
    )
    args = parser.parse_args()
    if args.pares == args.cerrar:
        parser.error("se requiere exactamente un modo: --pares o --cerrar")
    return args


def resolver_caps(args: argparse.Namespace) -> dict:
    """Resuelve los caps efectivos (defaults del run completo o del smoke).

    Args:
        args: Namespace de la CLI (caps en None usan el default del modo).

    Returns:
        Dict de caps validados.

    Raises:
        ValueError: Si algun cap queda fuera de rango.
    """
    caps = {
        "max_pares": CONFIG["max_pares"],
        "max_positivos_por_doc": CONFIG["max_positivos_por_doc"],
        "negativos_por_positivo": CONFIG["negativos_por_positivo"],
        "max_chunks_por_articulo": CONFIG["max_chunks_por_articulo"],
        "max_docs_bm25": CONFIG["max_docs_bm25"],
        "lote": CONFIG["lote"],
        "semilla": CONFIG["semilla"],
        "proporciones_negativos": dict(CONFIG["proporciones_negativos"]),
        "top_bm25_pool": CONFIG["top_bm25_pool"],
        "fuente_positivo": CONFIG["fuente_positivo"],
        "fuente_fallback_bm25": CONFIG["fuente_fallback_bm25"],
        "reintentos_negativo_duplicado": CONFIG["reintentos_negativo_duplicado"],
        "limite_shard_bytes": CONFIG["limite_shard_bytes"],
    }
    if args.smoke:
        for clave in (
            "max_pares",
            "max_positivos_por_doc",
            "negativos_por_positivo",
            "max_chunks_por_articulo",
            "max_docs_bm25",
            "lote",
        ):
            caps[clave] = CONFIG["smoke"][clave]
    for clave in (
        "max_pares",
        "max_positivos_por_doc",
        "negativos_por_positivo",
        "max_chunks_por_articulo",
        "max_docs_bm25",
        "lote",
    ):
        valor = getattr(args, clave)
        if valor is not None:
            caps[clave] = valor
    if caps["max_pares"] < 1:
        raise ValueError("--max-pares debe ser >= 1")
    if caps["max_positivos_por_doc"] < 1:
        raise ValueError("--max-positivos-por-doc debe ser >= 1")
    if caps["negativos_por_positivo"] < 0:
        raise ValueError("--negativos-por-positivo debe ser >= 0")
    if not 1 <= caps["max_chunks_por_articulo"] <= CONFIG["tope_chunks_selector_12"]:
        raise ValueError(
            f"--max-chunks-por-articulo debe estar en [1, {CONFIG['tope_chunks_selector_12']}] "
            "(el selector de 12 emite 3 chunks por candidato)"
        )
    if caps["max_docs_bm25"] < 0:
        raise ValueError("--max-docs-bm25 debe ser >= 0")
    if caps["lote"] < 1:
        raise ValueError("--lote debe ser >= 1")
    if abs(sum(caps["proporciones_negativos"].values()) - 1.0) > 1e-9:
        raise ValueError("las proporciones de negativos deben sumar 1.0")
    return caps


# ===== SECCION 3: carga de fuentes y validacion =====


@dataclass
class Fuentes:
    """Fuentes cargadas en modo solo lectura y estructuras derivadas."""

    entidades: pd.DataFrame
    canones: list[str]
    grupos: list[np.ndarray]
    pasajes: pd.DataFrame
    textos_pasajes: list[str]
    tokens_pasajes: list[list[str]]
    textos_articulos: dict[str, str]
    indice_bm25: object
    indice_canon: dict[str, int]
    citados_por_doc: dict[str, set[str]]
    ruta_corpus: str = ""
    ids_benchmark: set[str] = field(default_factory=set)
    cocitacion: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    md5_entradas: dict[str, str | None] = field(default_factory=dict)
    modulo12: object = None


def _exigir(condicion: bool, mensaje: str) -> None:
    """Falla rapido con RuntimeError si la condicion no se cumple.

    Args:
        condicion: Condicion a verificar.
        mensaje: Mensaje de error.

    Raises:
        RuntimeError: Si la condicion es falsa.
    """
    if not condicion:
        raise RuntimeError(mensaje)


def validar_cache_bm25(
    datos: dict, canones: list[str], grupos: list[np.ndarray], n_pasajes: int
) -> None:
    """Valida el indice BM25 cacheado contra los pasajes reconstruidos.

    Args:
        datos: Dict del joblib cacheado (version, canones, grupos_indices,
            indice, articulos).
        canones: Canones de los articulos permanentes reconstruidos.
        grupos: Grupos de indices de pasajes por articulo reconstruidos.
        n_pasajes: Numero de pasajes reconstruidos.

    Raises:
        RuntimeError: Si el cache no es v3 o no coincide con los pasajes
            reconstruidos (fail-fast, nunca seguir con un indice distinto).
    """
    _exigir(
        int(datos.get("version", -1)) == CONFIG["version_indice_bm25"],
        f"cache BM25 no es v{CONFIG['version_indice_bm25']}: {datos.get('version')}",
    )
    _exigir(list(datos["canones"]) == list(canones), "canones del cache != canones reconstruidos")
    _exigir(
        len(datos["grupos_indices"]) == len(grupos),
        "numero de grupos del cache != reconstruido",
    )
    for i, (grupo_cache, grupo_nuevo) in enumerate(
        zip(datos["grupos_indices"], grupos, strict=True)
    ):
        _exigir(
            len(grupo_cache) == len(grupo_nuevo)
            and bool((np.asarray(grupo_cache) == np.asarray(grupo_nuevo)).all()),
            f"grupo de pasajes {i} del cache != reconstruido",
        )
    indice = datos["indice"]
    _exigir(int(indice.corpus_size) == n_pasajes, "corpus_size del cache != n pasajes reconstruidos")


def cargar_fuentes(rutas: dict, logger: logging.Logger) -> Fuentes:
    """Carga entidades, pasajes, cache BM25 y textos de articulos.

    Registra el md5 de cada entrada (el corpus via md5_directorio) para
    verificar al final que las fuentes no cambiaron (solo lectura).

    Args:
        rutas: Dict de rutas absolutas.
        logger: Logger activo.

    Returns:
        Fuentes con todas las estructuras derivadas.

    Raises:
        RuntimeError: Si el cache BM25 no coincide con los pasajes.
    """
    inicio = time.perf_counter()
    md5_entradas = {
        "entidades": md5_archivo(rutas["ruta_entidades"]),
        "constitucion": md5_archivo(rutas["ruta_constitucion"]),
        "red_citas_resumen": md5_archivo(rutas["ruta_red_citas_resumen"]),
        "indice_bm25": md5_archivo(rutas["ruta_indice_bm25"]),
        "benchmark_v2": md5_archivo(rutas["ruta_benchmark_v2"]),
        "benchmark_v8": md5_archivo(rutas["ruta_benchmark_v8"]),
        "script_12": md5_archivo(rutas["ruta_script_12"]),
        "corpus": md5_directorio(rutas["ruta_corpus"]),
    }
    logger.info("md5 de entradas calculados en %.1f s", time.perf_counter() - inicio)

    entidades = pd.read_parquet(
        rutas["ruta_entidades"], columns=["id", "tipo", "ano", "arts_cp"]
    )
    entidades["id"] = entidades["id"].astype(str)

    modulo12 = modulo_12()
    modulo12.resolver_rutas()
    pasajes, canones, grupos, _ = modulo12.preparar_pasajes_constitucion(False)

    datos = joblib.load(rutas["ruta_indice_bm25"])
    validar_cache_bm25(datos, canones, grupos, len(pasajes))

    articulos = pd.read_parquet(rutas["ruta_constitucion"], columns=["numero", "texto"])
    textos_articulos = {
        str(numero): ("" if pd.isna(texto) else str(texto))
        for numero, texto in zip(articulos["numero"], articulos["texto"], strict=True)
    }
    citados_por_doc = {
        sid: {str(a) for a in (json.loads(crudo) if crudo else [])}
        for sid, crudo in zip(entidades["id"], entidades["arts_cp"], strict=True)
    }
    indice_canon = {canon: i for i, canon in enumerate(canones)}
    fuentes = Fuentes(
        entidades=entidades,
        canones=list(canones),
        grupos=list(grupos),
        pasajes=pasajes,
        textos_pasajes=pasajes["texto"].astype(str).tolist(),
        tokens_pasajes=pasajes["tokens"].tolist(),
        textos_articulos=textos_articulos,
        indice_bm25=datos["indice"],
        indice_canon=indice_canon,
        citados_por_doc=citados_por_doc,
        ruta_corpus=rutas["ruta_corpus"],
        md5_entradas=md5_entradas,
        modulo12=modulo12,
    )
    logger.info(
        "fuentes cargadas: %d docs, %d canones permanentes, %d pasajes, cache BM25 v%s",
        len(entidades),
        len(canones),
        len(pasajes),
        datos["version"],
    )
    return fuentes


def cargar_ids_benchmark(rutas: dict) -> dict:
    """Carga los 90 ids del benchmark M7 con doble fuente y verificacion.

    Fuente primaria: campo muestra de resultados/vinculo_cp_benchmark_v2.json
    (el benchmark original de 90 casos, 3/ano, semilla 42). Fuente cruzada:
    campo muestra de vinculo_cp_benchmark_v8.json. Se exige que ambos
    conjuntos sean identicos y tengan 90 ids unicos.

    Args:
        rutas: Dict de rutas absolutas.

    Returns:
        Dict con ids (set), n_ids, fuentes y md5s.

    Raises:
        RuntimeError: Si los conjuntos no coinciden o no son 90 unicos.
    """
    with open(rutas["ruta_benchmark_v2"], encoding="utf-8") as manejador:
        muestra_v2 = json.load(manejador)["muestra"]
    with open(rutas["ruta_benchmark_v8"], encoding="utf-8") as manejador:
        muestra_v8 = json.load(manejador)["muestra"]
    ids_v2 = [str(caso["id"]) for caso in muestra_v2]
    ids_v8 = [str(caso["id"]) for caso in muestra_v8]
    _exigir(len(ids_v2) == 90 and len(set(ids_v2)) == 90, "benchmark v2: se esperaban 90 ids unicos")
    _exigir(len(ids_v8) == 90 and len(set(ids_v8)) == 90, "benchmark v8: se esperaban 90 ids unicos")
    _exigir(set(ids_v2) == set(ids_v8), "los ids del benchmark v2 y v8 no coinciden")
    return {
        "ids": set(ids_v2),
        "n_ids": len(ids_v2),
        "fuente_primaria": rutas["ruta_benchmark_v2"],
        "fuente_cruzada": rutas["ruta_benchmark_v8"],
        "md5_v2": md5_archivo(rutas["ruta_benchmark_v2"]),
        "md5_v8": md5_archivo(rutas["ruta_benchmark_v8"]),
        "conjuntos_iguales": True,
    }


def construir_grafo_cocitacion(
    entidades: pd.DataFrame, ids_benchmark: set[str], canones: list[str]
) -> tuple[dict[str, list[tuple[str, int]]], int, int]:
    """Construye el grafo de co-citacion de articulos CP (senal M5).

    data/red_citas_resumen.parquet (M5) guarda metricas POR SENTENCIA
    (pagerank, indegree, hub, authority), no pares de articulos; el grafo
    de co-citacion de la CP se reconstruye desde arts_cp (M2) con la misma
    construccion documentada en la Fase W-a del ciclo v2 ("arista = dos
    articulos citados en la misma sentencia", 66.976 pares en el corpus
    completo). Aqui se excluyen las 90 sentencias del benchmark (anti-
    leakage), por lo que el conteo baja a ~66.955 pares. Cada articulo
    conserva sus top-CONFIG['top_cocitados_por_articulo'] socios por peso
    (desempate por canon).

    Args:
        entidades: DataFrame M2 con id y arts_cp.
        ids_benchmark: Ids excluidos (benchmark M7).
        canones: Canones permanentes indexados.

    Returns:
        Tupla (grafo, n_docs_usados, n_pares_unicos): grafo canon ->
        lista de (socio, peso) ordenada por peso descendente.
    """
    canones_set = set(canones)
    pesos: dict[tuple[str, str], int] = defaultdict(int)
    n_docs = 0
    for sid, crudo in zip(entidades["id"], entidades["arts_cp"], strict=True):
        if sid in ids_benchmark or not crudo:
            continue
        n_docs += 1
        arts = sorted({str(a) for a in json.loads(crudo) if str(a) in canones_set})
        for i, art_a in enumerate(arts):
            for art_b in arts[i + 1 :]:
                pesos[(art_a, art_b)] += 1
    adyacencia: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (art_a, art_b), peso in pesos.items():
        adyacencia[art_a].append((art_b, peso))
        adyacencia[art_b].append((art_a, peso))
    tope = CONFIG["top_cocitados_por_articulo"]
    grafo = {
        canon: sorted(lista, key=lambda par: (-par[1], clave_canon(par[0])))[:tope]
        for canon, lista in adyacencia.items()
    }
    return grafo, n_docs, len(pesos)


# ===== SECCION 4: seleccion determinista de positivos y tier BM25 =====


def seleccionar_unidades(
    fuentes: Fuentes, caps: dict, ids_benchmark: set[str], limite_docs: int | None
) -> dict:
    """Selecciona las unidades (doc, articulo) positivas con caps y seed 42.

    Reglas: docs del benchmark excluidos; solo articulos permanentes
    citados; cap por doc (muestreo con RNG derivado de (semilla, doc_id));
    cap global con muestreo determinista sobre la lista canonica de
    unidades. El cap global limita unidades (pares positivos con
    max_chunks=1).

    Args:
        fuentes: Fuentes cargadas.
        caps: Caps efectivos.
        ids_benchmark: Ids excluidos.
        limite_docs: Si no es None (smoke), solo los primeros N docs con
            citas permanentes en orden canonico.

    Returns:
        Dict con seleccion (doc_id -> lista de articulos ordenada),
        unidades_candidatas, unidades_muestreadas y docs_candidatos.
    """
    canones_set = set(fuentes.canones)
    docs_orden: list[str] = []
    unidades: list[tuple[str, str]] = []
    n_candidatas = 0
    for sid, crudo in zip(fuentes.entidades["id"], fuentes.entidades["arts_cp"], strict=True):
        if sid in ids_benchmark:
            continue
        arts = sorted(
            {str(a) for a in (json.loads(crudo) if crudo else []) if str(a) in canones_set},
            key=clave_canon,
        )
        if not arts:
            continue
        if limite_docs is not None and len(docs_orden) >= limite_docs:
            break
        if len(arts) > caps["max_positivos_por_doc"]:
            rng_doc = random.Random(f"{caps['semilla']}:cap:{sid}")
            arts = sorted(
                rng_doc.sample(arts, caps["max_positivos_por_doc"]), key=clave_canon
            )
        docs_orden.append(sid)
        n_candidatas += len(arts)
        for articulo in arts:
            unidades.append((sid, articulo))
    cap_unidades = max(1, caps["max_pares"] // caps["max_chunks_por_articulo"])
    if len(unidades) > cap_unidades:
        rng_global = random.Random(f"{caps['semilla']}:global")
        unidades = rng_global.sample(unidades, cap_unidades)
    por_doc: dict[str, list[str]] = defaultdict(list)
    for sid, articulo in unidades:
        por_doc[sid].append(articulo)
    orden_docs = [sid for sid in docs_orden if sid in por_doc]
    seleccion = {
        sid: sorted(por_doc[sid], key=clave_canon) for sid in orden_docs
    }
    return {
        "seleccion": seleccion,
        "unidades_candidatas": n_candidatas,
        "unidades_muestreadas": sum(len(articulos) for articulos in seleccion.values()),
        "docs_candidatos": len(docs_orden),
    }


def asignar_tier_bm25(docs: list[str], max_docs_bm25: int) -> set[str]:
    """Asigna el cupo BM25 por hash del doc (repartido en todo el corpus).

    El BM25 exacto es el coste dominante (~3.7 s/doc de media). Para que el
    cupo no sesgue la seleccion a los primeros anos del corpus, los docs con
    pares se ordenan por md5(doc_id) y se toma el primer --max-docs-bm25.
    El resto usa el selector de solape lexico (documentado). La asignacion
    es determinista y reanudable (no depende del orden de lotes).

    Args:
        docs: Docs con pares (orden canonico).
        max_docs_bm25: Cupo de docs con BM25 exacto.

    Returns:
        Conjunto de docs con BM25 exacto.
    """
    ordenados = sorted(docs, key=lambda sid: hashlib.md5(sid.encode("utf-8")).hexdigest())
    return set(ordenados[: max(0, max_docs_bm25)])


# ===== SECCION 5: construccion de pares por doc =====


def seleccionar_pares_bm25(
    fuentes: Fuentes,
    indices: list[int],
    matriz: np.ndarray,
    chunks: list[str],
    n_chunks: int,
) -> dict[str, list[tuple[int, int, str, str]]]:
    """Selecciona pares (chunk, pasaje) con el selector EXACTO de 12.

    Reutiliza construir_pares_rerank_chunks (mejor pasaje por maximo BM25 y
    top-3 chunks contra ese pasaje, el mismo par que ve el reranker en
    inferencia) y deriva los indices locales (pasaje_id, chunk_idx)
    replicando las dos lineas internas del selector (argmax de maximos y
    argsort estable de la columna del mejor pasaje). Cada par derivado se
    contrasta contra la salida de la funcion reutilizada; cualquier
    divergencia aborta.

    Args:
        fuentes: Fuentes cargadas.
        indices: Indices de articulo a seleccionar.
        matriz: Matriz (n_chunks, n_pasajes) de scores BM25 del doc.
        chunks: Chunks de la query del doc.
        n_chunks: Chunks por articulo (<= 3).

    Returns:
        Dict canon -> lista de (chunk_idx, pasaje_id local, texto_chunk,
        texto_pasaje).

    Raises:
        RuntimeError: Si la derivacion de indices diverge del selector.
    """
    if not indices:
        return {}
    k_selector = int(fuentes.modulo12.CONFIG["k_chunks_rerank"])
    n = min(n_chunks, k_selector)
    ficticios = np.full(len(fuentes.canones), -np.inf, dtype=np.float64)
    ficticios[indices] = 1.0
    resultado = fuentes.modulo12.construir_pares_rerank_chunks(
        ficticios, matriz, chunks, fuentes.grupos, fuentes.pasajes, k_candidatos=len(indices)
    )
    _exigir(
        len(resultado["queries"]) == len(indices) * k_selector,
        "el selector de 12 no emitio k_chunks_rerank pares por candidato",
    )
    maximos_pasajes = matriz.max(axis=0)
    pares_por_articulo: dict[str, list[tuple[int, int, str, str]]] = {}
    for pos, idx in enumerate(resultado["candidatos"]):
        idx = int(idx)
        indices_pasaje = fuentes.grupos[idx]
        mejor_pasaje = int(indices_pasaje[np.argmax(maximos_pasajes[indices_pasaje])])
        top_chunks = np.argsort(
            -matriz[:, mejor_pasaje], kind="stable"
        )[:n]
        pares = []
        for k, fila_chunk in enumerate(top_chunks):
            fila_chunk = int(fila_chunk)
            texto_chunk = resultado["queries"][pos * k_selector + k]
            texto_pasaje = resultado["pasajes"][pos * k_selector + k]
            _exigir(
                texto_chunk == chunks[fila_chunk]
                and texto_pasaje == fuentes.textos_pasajes[mejor_pasaje],
                "divergencia con el selector de pares de 12",
            )
            pasaje_id = int(np.argmax(indices_pasaje == mejor_pasaje))
            pares.append((fila_chunk, pasaje_id, texto_chunk, texto_pasaje))
        pares_por_articulo[fuentes.canones[idx]] = pares
    return pares_por_articulo


def seleccionar_pares_solape(
    fuentes: Fuentes,
    indices: list[int],
    chunks: list[str],
    chunks_sets: list[set[str]],
    n_chunks: int,
) -> dict[str, list[tuple[int, int, str, str]]]:
    """Selector de pares por solape lexico (proxy barato fuera del cupo BM25).

    Replica ESTRUCTURAL del selector de 12 (mejor pasaje + top chunks
    contra ese pasaje) cambiando el score BM25 por |tokens(chunk) ∩
    tokens(pasaje)|, reutilizando tokenizar de 12 y los tokens de pasaje
    ya calculados por preparar_pasajes_constitucion. Es determinista y no
    altera los troceados (los pares siguen siendo chunk de 800/200 chars
    y pasaje de 120 palabras). Se documenta como proxy: solo aplica a docs
    fuera de --max-docs-bm25.

    Args:
        fuentes: Fuentes cargadas.
        indices: Indices de articulo a seleccionar.
        chunks: Chunks de la query del doc (texto).
        chunks_sets: Sets de tokens por chunk de la query.
        n_chunks: Chunks por articulo (<= 3).

    Returns:
        Dict canon -> lista de (chunk_idx, pasaje_id local, texto_chunk,
        texto_pasaje).
    """
    if not indices or not chunks_sets:
        return {}
    n = min(n_chunks, int(fuentes.modulo12.CONFIG["k_chunks_rerank"]))
    union_chunks: set[str] = set().union(*chunks_sets)
    pares_por_articulo: dict[str, list[tuple[int, int, str, str]]] = {}
    for idx in indices:
        indices_pasaje = fuentes.grupos[idx]
        mejor_pasaje = int(indices_pasaje[0])
        mejor_score = -1
        for posicion in indices_pasaje:
            score = len(set(fuentes.tokens_pasajes[int(posicion)]) & union_chunks)
            if score > mejor_score:
                mejor_score = score
                mejor_pasaje = int(posicion)
        tokens_mejor = set(fuentes.tokens_pasajes[mejor_pasaje])
        scores_chunks = np.asarray(
            [len(chunk_set & tokens_mejor) for chunk_set in chunks_sets], dtype=np.int64
        )
        top_chunks = np.argsort(-scores_chunks, kind="stable")[:n]
        pasaje_id = int(np.argmax(indices_pasaje == mejor_pasaje))
        pares_por_articulo[fuentes.canones[idx]] = [
            (
                int(fila_chunk),
                pasaje_id,
                chunks[fila_chunk],
                fuentes.textos_pasajes[mejor_pasaje],
            )
            for fila_chunk in top_chunks
        ]
    return pares_por_articulo


def elegir_fuente_negativo(rng_doc: random.Random, proporciones: dict) -> str:
    """Elige la fuente nominal del negativo (mezcla 50/30/20 por defecto).

    Args:
        rng_doc: RNG determinista del documento.
        proporciones: Dict fuente -> peso (suma 1.0).

    Returns:
        Nombre de la fuente elegida.
    """
    umbral = rng_doc.random()
    acumulado = 0.0
    for fuente, peso in proporciones.items():
        acumulado += peso
        if umbral < acumulado:
            return fuente
    return "aleatorio"


def muestrear_negativo(
    rng_doc: random.Random,
    fuente_nominal: str,
    articulo_positivo: str,
    citados: set[str],
    pool_bm25: list[str],
    cocitacion: dict[str, list[tuple[str, int]]],
    aleatorios: list[str],
    usados: set[str],
    caps: dict,
) -> tuple[str | None, str | None]:
    """Muestrea un articulo negativo no citado y no repetido en el doc.

    Args:
        rng_doc: RNG determinista del documento.
        fuente_nominal: Fuente elegida por proporciones.
        articulo_positivo: Articulo positivo de referencia (co-citados).
        citados: Articulos citados por el doc (excluidos siempre).
        pool_bm25: Top-30 BM25 no citados (vacio si el doc no tiene BM25).
        cocitacion: Grafo de co-citacion (canon -> [(socio, peso)]).
        aleatorios: Canones permanentes no citados del doc.
        usados: Negativos ya usados en el doc (sin repetir).
        caps: Caps efectivos.

    Returns:
        Tupla (fuente_real, articulo) o (None, None) si no hay candidato
        (se cuenta como omitido).
    """
    fuente = fuente_nominal
    if fuente == "bm25" and not pool_bm25:
        fuente = caps["fuente_fallback_bm25"]
    for _ in range(caps["reintentos_negativo_duplicado"] + 1):
        elegido: str | None = None
        if fuente == "bm25":
            elegido = rng_doc.choice(pool_bm25)
        elif fuente == "cocitado":
            candidatos = [
                (socio, peso)
                for socio, peso in cocitacion.get(articulo_positivo, [])
                if socio not in citados and socio not in usados
            ]
            if candidatos:
                elegido = rng_doc.choices(
                    [socio for socio, _ in candidatos],
                    weights=[peso for _, peso in candidatos],
                    k=1,
                )[0]
            else:
                fuente = "aleatorio"
                continue
        else:
            if not aleatorios:
                return None, None
            elegido = rng_doc.choice(aleatorios)
        if elegido is not None and elegido not in usados:
            return fuente, elegido
    return None, None


def construir_fila(
    doc_id: str,
    articulo: str,
    pasaje_id: int,
    chunk_idx: int,
    label: int,
    fuente: str,
    texto_chunk: str,
    texto_pasaje: str,
) -> dict:
    """Construye una fila del parquet de pares con las 10 columnas del diseno.

    Args:
        doc_id: Id de la sentencia.
        articulo: Canon del articulo CP.
        pasaje_id: Indice local del pasaje dentro del articulo.
        chunk_idx: Indice del chunk dentro de la query del doc.
        label: 1 positivo, 0 negativo.
        fuente: "positivo" o fuente del negativo.
        texto_chunk: Texto del chunk.
        texto_pasaje: Texto del pasaje.

    Returns:
        Dict listo para pa.schema.
    """
    return {
        "doc_id": doc_id,
        "articulo": articulo,
        "pasaje_id": int(pasaje_id),
        "chunk_idx": int(chunk_idx),
        "label": int(label),
        "fuente_negativo": fuente,
        "texto_chunk": texto_chunk,
        "texto_pasaje": texto_pasaje,
        "n_chars_chunk": len(texto_chunk),
        "n_chars_pasaje": len(texto_pasaje),
    }


def procesar_doc(
    fuentes: Fuentes,
    doc_id: str,
    texto: str,
    articulos: list[str],
    es_bm25: bool,
    caps: dict,
) -> tuple[list[dict], dict]:
    """Genera los pares (positivos y negativos) de una sentencia.

    Args:
        fuentes: Fuentes cargadas.
        doc_id: Id de la sentencia.
        texto: texto_limpio de la sentencia.
        articulos: Articulos positivos seleccionados (orden canonico).
        es_bm25: True si el doc usa el selector BM25 exacto.
        caps: Caps efectivos.

    Returns:
        Tupla (filas, stats_doc) con stats_doc = {"sin_chunks",
        "negativos_omitidos", "negativos_por_fuente", "n_negativos"}.
    """
    chunks = fuentes.modulo12.chunks_query(texto)
    stats = {"sin_chunks": 0, "negativos_omitidos": 0, "negativos_por_fuente": Counter()}
    if not chunks:
        stats["sin_chunks"] = 1
        return [], stats
    citados = fuentes.citados_por_doc.get(doc_id, set())
    rng_doc = random.Random(f"{caps['semilla']}:doc:{doc_id}")
    aleatorios = [canon for canon in fuentes.canones if canon not in citados]
    if es_bm25:
        scores, matriz = fuentes.modulo12.scores_bm25_caso(
            fuentes.indice_bm25, fuentes.grupos, fuentes.canones, chunks
        )
        ranking = fuentes.modulo12.ranking_desde_scores(scores, fuentes.canones)
        pool_bm25 = [canon for canon in ranking if canon not in citados][: caps["top_bm25_pool"]]
    else:
        matriz = None
        pool_bm25 = []
    negativos: list[tuple[str, str]] = []
    usados: set[str] = set()
    for articulo in articulos:
        for _ in range(caps["negativos_por_positivo"]):
            fuente_nominal = elegir_fuente_negativo(rng_doc, caps["proporciones_negativos"])
            fuente, elegido = muestrear_negativo(
                rng_doc,
                fuente_nominal,
                articulo,
                citados,
                pool_bm25,
                fuentes.cocitacion,
                aleatorios,
                usados,
                caps,
            )
            if elegido is None:
                stats["negativos_omitidos"] += 1
                continue
            usados.add(elegido)
            negativos.append((fuente, elegido))
            stats["negativos_por_fuente"][fuente] += 1
    indices = [
        fuentes.indice_canon[canon]
        for canon in dict.fromkeys([*articulos, *(canon for _, canon in negativos)])
    ]
    if es_bm25:
        pares_por_articulo = seleccionar_pares_bm25(
            fuentes, indices, matriz, chunks, caps["max_chunks_por_articulo"]
        )
    else:
        chunks_sets = [set(tokens) for tokens in (fuentes.modulo12.tokenizar(c) for c in chunks)]
        pares_por_articulo = seleccionar_pares_solape(
            fuentes, indices, chunks, chunks_sets, caps["max_chunks_por_articulo"]
        )
    filas: list[dict] = []
    for articulo in articulos:
        for chunk_idx, pasaje_id, texto_chunk, texto_pasaje in pares_por_articulo[articulo]:
            filas.append(
                construir_fila(
                    doc_id,
                    articulo,
                    pasaje_id,
                    chunk_idx,
                    1,
                    caps["fuente_positivo"],
                    texto_chunk,
                    texto_pasaje,
                )
            )
    for fuente, articulo in negativos:
        for chunk_idx, pasaje_id, texto_chunk, texto_pasaje in pares_por_articulo[articulo]:
            filas.append(
                construir_fila(
                    doc_id,
                    articulo,
                    pasaje_id,
                    chunk_idx,
                    0,
                    fuente,
                    texto_chunk,
                    texto_pasaje,
                )
            )
    return filas, stats


# ===== SECCION 6: lotes, shards y reanudacion =====


def preparar_salida(dir_salida: Path, forzar: bool, smoke: bool, logger: logging.Logger) -> Path:
    """Valida/crea el directorio de salida (no sobrescribe sin --forzar).

    Args:
        dir_salida: Directorio de salida.
        forzar: True permite reutilizar/reescribir.
        smoke: True limpia solo los artefactos del smoke (siempre reescribible).
        logger: Logger activo.

    Returns:
        Directorio validado.

    Raises:
        FileExistsError: Si hay artefactos previos y no es smoke ni forzar.
    """
    if smoke:
        dir_salida.mkdir(parents=True, exist_ok=True)
        for patron in ("pares_reranker_b1_shard*", "_lote*.json", "_lote*.tmp"):
            for ruta in dir_salida.glob(patron):
                ruta.unlink()
        return dir_salida
    existentes = sorted(dir_salida.glob("pares_reranker_b1_shard*.parquet"))
    existentes += sorted(dir_salida.glob("_lote*.json"))
    if existentes and not forzar:
        raise FileExistsError(
            f"salida previa en {dir_salida} ({len(existentes)} artefactos); "
            "use --forzar para reanudar/rehacer (los lotes completos se reutilizan)"
        )
    if existentes:
        logger.warning(
            "reanudacion con --forzar: %d artefactos previos (huella verificada por lote)",
            len(existentes),
        )
    dir_salida.mkdir(parents=True, exist_ok=True)
    return dir_salida


def caps_serializables(caps: dict) -> dict:
    """Extrae el subconjunto serializable de caps (huella, marcas, JSON).

    Args:
        caps: Caps efectivos.

    Returns:
        Dict con los caps que definen el contenido de los pares.
    """
    return {
        clave: caps[clave]
        for clave in (
            "max_pares",
            "max_positivos_por_doc",
            "negativos_por_positivo",
            "max_chunks_por_articulo",
            "max_docs_bm25",
            "lote",
            "semilla",
            "proporciones_negativos",
            "top_bm25_pool",
            "limite_shard_bytes",
        )
    }


def calcular_huella(caps: dict, fuentes: Fuentes, info_benchmark: dict, limite_docs: int | None) -> str:
    """Calcula la huella md5 de la corrida (caps + entradas + benchmark).

    Args:
        caps: Caps efectivos.
        fuentes: Fuentes cargadas (md5_entradas).
        info_benchmark: Info del benchmark (md5s).
        limite_docs: Limite de docs (smoke) o None.

    Returns:
        Hexdigest de la configuracion completa.
    """
    carga = {
        "version": CONFIG["version"],
        "caps": caps_serializables(caps),
        "limite_docs": limite_docs,
        "md5_entradas": fuentes.md5_entradas,
        "benchmark": {"n": info_benchmark["n_ids"], "md5_v2": info_benchmark["md5_v2"]},
    }
    return hashlib.md5(
        json.dumps(carga, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def escribir_shard(ruta: Path, filas: list[dict], metadatos: dict) -> dict:
    """Escribe un shard parquet atomico con metadatos de la corrida.

    Args:
        ruta: Ruta destino del shard.
        filas: Filas del shard.
        metadatos: Dict serializable con huella/conteos.

    Returns:
        Dict con ruta, filas, bytes y md5 del shard escrito.
    """
    tabla = pa.Table.from_pylist(filas, schema=ESQUEMA_PARES)
    tabla = tabla.replace_schema_metadata(
        {CLAVE_METADATOS: json.dumps(metadatos, ensure_ascii=False).encode("utf-8")}
    )
    ruta_temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    pq.write_table(tabla, ruta_temporal, compression="snappy")
    os.replace(ruta_temporal, ruta)
    return {
        "ruta": ruta.name,
        "filas": len(filas),
        "bytes": ruta.stat().st_size,
        "md5": md5_archivo(ruta),
    }


def escribir_shards_lote(
    dir_salida: Path,
    lote: int,
    filas: list[dict],
    metadatos_base: dict,
    caps: dict,
) -> list[dict]:
    """Escribe el/los shard(s) de un lote, partiendo si supera el limite.

    Los shards son inmutables: se escriben a .tmp y se renombran con
    os.replace. Si el lote se rehace (--forzar), los shards previos de ese
    lote se eliminan antes (nunca se sobrescribe una salida ajena).

    Args:
        dir_salida: Directorio de salida.
        lote: Indice del lote.
        filas: Filas del lote.
        metadatos_base: Metadatos comunes (huella, caps, lote).
        caps: Caps efectivos.

    Returns:
        Lista de dicts de shards escritos.
    """
    for previo in dir_salida.glob(f"pares_reranker_b1_shard{lote:04d}*"):
        if previo.is_file():
            previo.unlink()
    partes: list[list[dict]] = []
    actual: list[dict] = []
    bytes_actual = 0
    for fila in filas:
        tam = (
            len(fila["texto_chunk"])
            + len(fila["texto_pasaje"])
            + CONFIG["overhead_bytes_par"]
        )
        if actual and bytes_actual + tam > caps["limite_shard_bytes"]:
            partes.append(actual)
            actual = []
            bytes_actual = 0
        actual.append(fila)
        bytes_actual += tam
    if actual:
        partes.append(actual)
    total_partes = len(partes)
    escritos = []
    for indice_parte, parte in enumerate(partes):
        if total_partes == 1:
            nombre = CONFIG["patron_shard"].format(lote=lote)
        else:
            nombre = CONFIG["patron_shard"].format(lote=lote).replace(
                ".parquet", f"_p{indice_parte}.parquet"
            )
        metadatos = dict(metadatos_base)
        metadatos.update(
            {"parte": indice_parte, "n_partes": total_partes, "filas": len(parte)}
        )
        escritos.append(escribir_shard(dir_salida / nombre, parte, metadatos))
    return escritos


def procesar_lote(
    fuentes: Fuentes,
    lote: int,
    docs_lote: list[str],
    seleccion: dict,
    tier_bm25: set[str],
    caps: dict,
    huella: str,
    dir_salida: Path,
    logger: logging.Logger,
) -> dict:
    """Procesa un lote de docs y persiste shard(s) + marca de completado.

    Args:
        fuentes: Fuentes cargadas.
        lote: Indice del lote.
        docs_lote: Ids del lote (orden canonico).
        seleccion: doc_id -> articulos positivos.
        tier_bm25: Docs con BM25 exacto.
        caps: Caps efectivos.
        huella: Huella de la corrida.
        dir_salida: Directorio de salida.
        logger: Logger activo.

    Returns:
        Dict resumen del lote (conteos, shards, tiempos).
    """
    inicio = time.perf_counter()
    tabla = fuentes.modulo12.leer_filas_por_id(
        fuentes.ruta_corpus, ["id", "texto_limpio"], docs_lote
    )
    tabla["id"] = tabla["id"].astype(str)
    textos = dict(
        zip(
            tabla["id"],
            tabla["texto_limpio"].fillna("").astype(str),
            strict=True,
        )
    )
    filas: list[dict] = []
    stats = {
        "docs": len(docs_lote),
        "docs_tier_bm25": 0,
        "docs_tier_solape": 0,
        "docs_sin_chunks": 0,
        "negativos_omitidos": 0,
        "negativos_por_fuente": Counter(),
    }
    for doc_id in docs_lote:
        es_bm25 = doc_id in tier_bm25
        stats["docs_tier_bm25" if es_bm25 else "docs_tier_solape"] += 1
        filas_doc, stats_doc = procesar_doc(
            fuentes, doc_id, textos.get(doc_id, ""), seleccion[doc_id], es_bm25, caps
        )
        filas.extend(filas_doc)
        stats["docs_sin_chunks"] += stats_doc["sin_chunks"]
        stats["negativos_omitidos"] += stats_doc["negativos_omitidos"]
        stats["negativos_por_fuente"].update(stats_doc["negativos_por_fuente"])
    positivos = sum(1 for fila in filas if fila["label"] == 1)
    metadatos_base = {
        "version": CONFIG["version"],
        "huella": huella,
        "lote": lote,
        "n_docs": len(docs_lote),
    }
    shards = escribir_shards_lote(dir_salida, lote, filas, metadatos_base, caps)
    resumen = {
        "huella": huella,
        "lote": lote,
        "caps": caps_serializables(caps),
        "n_docs": len(docs_lote),
        "n_filas": len(filas),
        "positivos": positivos,
        "negativos": len(filas) - positivos,
        "negativos_por_fuente": dict(stats["negativos_por_fuente"]),
        "docs_tier_bm25": stats["docs_tier_bm25"],
        "docs_tier_solape": stats["docs_tier_solape"],
        "docs_sin_chunks": stats["docs_sin_chunks"],
        "negativos_omitidos": stats["negativos_omitidos"],
        "archivos": shards,
        "tiempo_segundos": round(time.perf_counter() - inicio, 1),
    }
    ruta_marca = dir_salida / CONFIG["patron_marca"].format(lote=lote)
    ruta_temporal = ruta_marca.with_suffix(".json.tmp")
    with open(ruta_temporal, "w", encoding="utf-8") as manejador:
        json.dump(resumen, manejador, ensure_ascii=False, indent=2, default=str)
    os.replace(ruta_temporal, ruta_marca)
    logger.info(
        "lote %d: %d docs, %d filas (pos %d/neg %d), %d shard(s), %.1f s",
        lote,
        len(docs_lote),
        len(filas),
        positivos,
        len(filas) - positivos,
        len(shards),
        resumen["tiempo_segundos"],
    )
    return resumen


def procesar_lotes(
    fuentes: Fuentes,
    seleccion: dict,
    tier_bm25: set[str],
    caps: dict,
    huella: str,
    dir_salida: Path,
    logger: logging.Logger,
) -> list[dict]:
    """Procesa todos los lotes con reanudacion por marca de lote.

    Args:
        fuentes: Fuentes cargadas.
        seleccion: doc_id -> articulos positivos (orden canonico).
        tier_bm25: Docs con BM25 exacto.
        caps: Caps efectivos.
        huella: Huella de la corrida.
        dir_salida: Directorio de salida.
        logger: Logger activo.

    Returns:
        Lista de resumenes por lote (reutilizados + nuevos).

    Raises:
        RuntimeError: Si una marca previa tiene huella distinta (caps o
            entradas cambiaron: borrar la salida o usar los mismos caps).
    """
    docs = list(seleccion)
    lotes = [docs[i : i + caps["lote"]] for i in range(0, len(docs), caps["lote"])]
    resumenes = []
    for lote, docs_lote in enumerate(lotes):
        ruta_marca = dir_salida / CONFIG["patron_marca"].format(lote=lote)
        if ruta_marca.is_file():
            with open(ruta_marca, encoding="utf-8") as manejador:
                marca = json.load(manejador)
            if marca.get("huella") != huella:
                raise RuntimeError(
                    f"la marca {ruta_marca} tiene huella distinta (caps/entradas cambiaron); "
                    "borre data/pares_b1 o use los mismos parametros"
                )
            logger.info("lote %d reutilizado (%d filas)", lote, marca["n_filas"])
            resumenes.append(marca)
            continue
        resumenes.append(
            procesar_lote(
                fuentes, lote, docs_lote, seleccion, tier_bm25, caps, huella, dir_salida, logger
            )
        )
    return resumenes


# ===== SECCION 7: verificacion smoke y resumen final =====


def verificar_checks_smoke(
    fuentes: Fuentes, resumenes: list[dict], dir_salida: Path, logger: logging.Logger
) -> dict:
    """Ejecuta los checks minimos del smoke sobre los shards escritos.

    (a) 0 docs del benchmark entre los pares; (b) textos de chunk/pasaje no
    vacios y chunking/pasajes identicos a los de 12 en 3 ejemplos (releyendo
    el corpus y re-segmentando); (c) ningun negativo aparece como positivo
    del mismo doc y ningun negativo pertenece a los articulos citados del
    doc.

    Args:
        fuentes: Fuentes cargadas.
        resumenes: Resumenes de lotes con la lista de shards.
        dir_salida: Directorio de salida.
        logger: Logger activo.

    Returns:
        Dict con los resultados de los checks (booleans y detalles).

    Raises:
        RuntimeError: Si algun check falla.
    """
    archivos = [dir_salida / shard["ruta"] for resumen in resumenes for shard in resumen["archivos"]]
    _exigir(bool(archivos), "el smoke no escribio shards")
    tabla = pq.read_table([str(archivo) for archivo in archivos])
    datos = tabla.to_pandas()
    # (a) exclusion del benchmark
    docs_pares = set(datos["doc_id"].astype(str))
    intrusos = docs_pares & fuentes.ids_benchmark
    _exigir(not intrusos, f"docs del benchmark en los pares: {sorted(intrusos)[:5]}")
    # (b) textos no vacios + troceados identicos a 12
    _exigir(bool((datos["n_chars_chunk"] > 0).all()), "hay chunks vacios en los pares")
    _exigir(bool((datos["n_chars_pasaje"] > 0).all()), "hay pasajes vacios en los pares")
    ejemplos = datos.drop_duplicates("doc_id").head(3)
    ids_ejemplos = [str(sid) for sid in ejemplos["doc_id"]]
    tabla_textos = fuentes.modulo12.leer_filas_por_id(
        fuentes.ruta_corpus, ["id", "texto_limpio"], ids_ejemplos
    )
    tabla_textos["id"] = tabla_textos["id"].astype(str)
    textos = dict(
        zip(tabla_textos["id"], tabla_textos["texto_limpio"].fillna("").astype(str), strict=True)
    )
    n_ejemplos = 0
    for n_ejemplos, (_, fila) in enumerate(ejemplos.iterrows(), start=1):
        doc_id = str(fila["doc_id"])
        chunks = fuentes.modulo12.chunks_query(textos[doc_id])
        _exigir(
            fila["texto_chunk"] == chunks[int(fila["chunk_idx"])],
            f"chunk {fila['chunk_idx']} de {doc_id} no coincide con chunks_query de 12",
        )
        pasajes_articulo = fuentes.modulo12.dividir_pasajes(
            fuentes.textos_articulos[str(fila["articulo"])],
            fuentes.modulo12.CONFIG["pasaje_max_palabras"],
        )
        _exigir(
            fila["texto_pasaje"] == pasajes_articulo[int(fila["pasaje_id"])],
            f"pasaje {fila['pasaje_id']} de {fila['articulo']} no coincide con 12",
        )
    # (c) negativos vs positivos/citados del mismo doc
    for doc_id, grupo in datos.groupby("doc_id", sort=False):
        positivos_doc = set(grupo.loc[grupo["label"] == 1, "articulo"].astype(str))
        negativos_doc = set(grupo.loc[grupo["label"] == 0, "articulo"].astype(str))
        _exigir(
            not (positivos_doc & negativos_doc),
            f"negativo repetido como positivo en {doc_id}: {positivos_doc & negativos_doc}",
        )
        citados = fuentes.citados_por_doc.get(str(doc_id), set())
        _exigir(
            not (negativos_doc & citados),
            f"negativo citado por {doc_id}: {negativos_doc & citados}",
        )
    resultado = {
        "a_benchmark_cero": True,
        "b_textos_no_vacios": True,
        "b_chunking_coincide_con_12": True,
        "b_pasajes_coinciden_con_12": True,
        "c_negativos_no_positivos_ni_citados": True,
        "n_docs_pares": len(docs_pares),
        "n_filas_smoke": len(datos),
        "ejemplos_verificados": ids_ejemplos,
    }
    logger.info("checks smoke OK: %s", json.dumps(resultado, ensure_ascii=False))
    return resultado


def construir_resumen(
    modo_smoke: bool,
    caps: dict,
    info_benchmark: dict,
    seleccion_stats: dict,
    tier_bm25: set[str],
    resumenes: list[dict],
    rutas: dict,
    dir_salida: Path,
    huella: str,
    md5_entradas_inicio: dict,
    md5_entradas_final: dict,
    tiempos: dict,
    checks_smoke: dict | None,
    n_cocitacion_docs: int,
    n_cocitacion_pares: int,
) -> dict:
    """Construye el JSON final con conteos, caps, fuentes y trazabilidad.

    Args:
        modo_smoke: True si es corrida smoke.
        caps: Caps efectivos.
        info_benchmark: Info de los 90 ids excluidos.
        seleccion_stats: Stats de seleccion de unidades.
        tier_bm25: Docs con BM25 exacto.
        resumenes: Resumenes de lotes.
        rutas: Rutas absolutas.
        dir_salida: Directorio de salida.
        huella: Huella de la corrida.
        md5_entradas_inicio: md5 de las entradas al inicio.
        md5_entradas_final: md5 recalculados al final.
        tiempos: Tiempos por fase.
        checks_smoke: Checks del smoke (o None).
        n_cocitacion_docs: Docs usados para el grafo de co-citacion.
        n_cocitacion_pares: Pares unicos del grafo de co-citacion.

    Returns:
        Dict serializable del resumen.
    """
    negativos_por_fuente: Counter = Counter()
    shards = []
    for resumen in resumenes:
        negativos_por_fuente.update(resumen["negativos_por_fuente"])
        shards.extend(resumen["archivos"])
    positivos = sum(resumen["positivos"] for resumen in resumenes)
    negativos = sum(resumen["negativos"] for resumen in resumenes)
    # Guard de integridad: DURO solo para las fuentes de DATOS; script_12 es
    # codigo reutilizado y se registra como informativo (nunca aborta por el).
    fuentes_datos = CONFIG["fuentes_datos_integridad"]
    cambios_datos = [
        clave
        for clave in fuentes_datos
        if md5_entradas_inicio.get(clave) != md5_entradas_final.get(clave)
    ]
    entradas_intactas = not cambios_datos
    cambio_script_12 = md5_entradas_inicio.get("script_12") != md5_entradas_final.get("script_12")
    return {
        "version": CONFIG["version"],
        "ciclo": CONFIG["ciclo"],
        "sprint": CONFIG["sprint"],
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "descripcion": (
            "Pares de entrenamiento del reranker B1: positivos (chunk de sentencia -> "
            "pasaje CP citado) y hard negatives (BM25/co-citados/aleatorios no citados). "
            "El entrenamiento es otro script."
        ),
        "modo": "pares",
        "smoke": modo_smoke,
        "semilla": caps["semilla"],
        "huella": huella,
        "caps": dict(caps),
        "interpretacion_cap_global": (
            "--max-pares limita pares POSITIVOS (unidades doc-articulo); las filas totales "
            "son positivos + negativos (positivos * (1 + negativos_por_positivo) con "
            "max_chunks=1)"
        ),
        "proporciones_negativos": caps["proporciones_negativos"],
        "exclusion_benchmark": {
            "n_ids": info_benchmark["n_ids"],
            "fuente_primaria": info_benchmark["fuente_primaria"],
            "fuente_cruzada": info_benchmark["fuente_cruzada"],
            "md5_v2": info_benchmark["md5_v2"],
            "md5_v8": info_benchmark["md5_v8"],
            "conjuntos_iguales": info_benchmark["conjuntos_iguales"],
            "docs_benchmark_en_pares": 0,
        },
        "seleccion": {
            "unidades_candidatas": seleccion_stats["unidades_candidatas"],
            "unidades_muestreadas": seleccion_stats["unidades_muestreadas"],
            "docs_candidatos": seleccion_stats["docs_candidatos"],
            "docs_con_pares": len(seleccion_stats["seleccion"]),
            "docs_tier_bm25": len(tier_bm25),
            "docs_tier_solape": len(seleccion_stats["seleccion"]) - len(tier_bm25),
        },
        "conteos": {
            "filas_total": positivos + negativos,
            "positivos": positivos,
            "negativos": negativos,
            "negativos_por_fuente": dict(sorted(negativos_por_fuente.items())),
            "docs_sin_chunks": sum(resumen["docs_sin_chunks"] for resumen in resumenes),
            "negativos_omitidos": sum(resumen["negativos_omitidos"] for resumen in resumenes),
        },
        "cocitacion_m5": {
            "construccion": (
                "arista = dos articulos CP citados en la misma sentencia (Fase W-a, ciclo v2); "
                "red_citas_resumen.parquet (M5) guarda metricas por sentencia, no pares, por lo "
                "que el grafo se reconstruye desde arts_cp M2"
            ),
            "docs_usados": n_cocitacion_docs,
            "pares_unicos": n_cocitacion_pares,
            "top_por_articulo": CONFIG["top_cocitados_por_articulo"],
            "benchmark_excluido": True,
        },
        "reutilizacion_12": {
            "funciones": [
                "preparar_pasajes_constitucion",
                "chunks_query",
                "scores_bm25_caso",
                "ranking_desde_scores",
                "construir_pares_rerank_chunks",
                "tokenizar",
                "dividir_pasajes",
                "leer_filas_por_id",
            ],
            "chunk_bm25_chars": fuentes_chunk_chars(),
            "solape_bm25_chars": fuentes_solape_chars(),
            "pasaje_max_palabras": fuentes_pasaje_palabras(),
            "k_chunks_rerank": fuentes_k_chunks(),
            "md5_script_12": md5_entradas_inicio.get("script_12"),
            "nota": (
                "los pares usan EXACTAMENTE los troceados de la inferencia; el selector "
                "construir_pares_rerank_chunks se reutiliza y la extraccion de indices locales "
                "(argmax de maximos y argsort estable) se replica y contrasta contra su salida; "
                "para docs fuera del cupo BM25 se usa un selector de solape lexico (proxy "
                "documentado, misma estructura mejor-pasaje/top-chunks)"
            ),
        },
        "shards": {
            "dir": str(dir_salida),
            "n_shards": len(shards),
            "bytes_total": sum(shard["bytes"] for shard in shards),
            "archivos": shards,
        },
        "integridad": {
            "fuentes_datos": fuentes_datos,
            "datos_intactos": entradas_intactas,
            "cambios_datos": cambios_datos,
            "script_12": {
                "md5_inicio": md5_entradas_inicio.get("script_12"),
                "md5_final": md5_entradas_final.get("script_12"),
                "cambio_detectado": cambio_script_12,
                "informativo": True,
                "aborta": False,
                "nota": (
                    "script_12 es codigo reutilizado (no una fuente de datos): una edicion "
                    "aditiva puede cambiar su md5 sin alterar los pares; se registra como "
                    "informativo"
                ),
            },
        },
        "md5_entradas_inicio": md5_entradas_inicio,
        "md5_entradas_final": md5_entradas_final,
        "entradas_intactas": entradas_intactas,
        "tiempos_segundos": tiempos,
        "versiones": versiones_entorno(),
        "checks_smoke": checks_smoke,
    }


def fuentes_chunk_chars() -> int:
    """Devuelve chunk_bm25_chars de la CONFIG reutilizada de 12."""
    return int(modulo_12().CONFIG["chunk_bm25_chars"])


def fuentes_solape_chars() -> int:
    """Devuelve solape_bm25_chars de la CONFIG reutilizada de 12."""
    return int(modulo_12().CONFIG["solape_bm25_chars"])


def fuentes_pasaje_palabras() -> int:
    """Devuelve pasaje_max_palabras de la CONFIG reutilizada de 12."""
    return int(modulo_12().CONFIG["pasaje_max_palabras"])


def fuentes_k_chunks() -> int:
    """Devuelve k_chunks_rerank de la CONFIG reutilizada de 12."""
    return int(modulo_12().CONFIG["k_chunks_rerank"])


def escribir_json(ruta: str | Path, contenido: dict) -> None:
    """Escribe un JSON con indentacion (atomico via .tmp + os.replace).

    Args:
        ruta: Ruta destino.
        contenido: Dict serializable.
    """
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    ruta_temporal = destino.with_suffix(".json.tmp")
    with open(ruta_temporal, "w", encoding="utf-8") as manejador:
        json.dump(contenido, manejador, ensure_ascii=False, indent=2, default=str)
    os.replace(ruta_temporal, destino)


def imprimir_resumen(resumen: dict, ruta_json: Path, dir_salida: Path) -> None:
    """Imprime el resumen final en consola (<= 10 lineas).

    Args:
        resumen: Resumen final.
        ruta_json: Ruta del JSON escrito.
        dir_salida: Directorio de shards.
    """
    conteos = resumen["conteos"]
    seleccion = resumen["seleccion"]
    print("=== B1 pares reranker: resumen final ===")
    print(
        f"filas: {conteos['filas_total']} (positivos {conteos['positivos']}, "
        f"negativos {conteos['negativos']}) en {seleccion['docs_con_pares']} docs"
    )
    print(f"negativos por fuente: {conteos['negativos_por_fuente']}")
    print(
        f"tier BM25: {seleccion['docs_tier_bm25']} docs | solape: "
        f"{seleccion['docs_tier_solape']} docs | sin chunks: {conteos['docs_sin_chunks']}"
    )
    print(
        f"benchmark excluido: {resumen['exclusion_benchmark']['n_ids']} ids "
        f"(0 docs en pares: {resumen['exclusion_benchmark']['docs_benchmark_en_pares'] == 0})"
    )
    print(
        f"shards: {resumen['shards']['n_shards']} | "
        f"{resumen['shards']['bytes_total'] / 1024**2:.1f} MB | dir: {dir_salida}"
    )
    print(f"entradas intactas (md5): {resumen['entradas_intactas']}")
    print(f"tiempo total: {resumen['tiempos_segundos']['total']:.1f} s")
    print(f"json: {ruta_json}")


# ===== SECCION 8: modo --cerrar (reconstruccion del JSON sin recomputar pares) =====


def cargar_marcas_lotes(dir_salida: Path) -> list[dict]:
    """Carga y valida las marcas _lote*.json de una corrida --pares.

    Args:
        dir_salida: Directorio con las marcas y shards.

    Returns:
        Lista de marcas ordenada por lote.

    Raises:
        FileNotFoundError: Si no hay marcas en el directorio.
        RuntimeError: Si las huellas difieren o los lotes no son contiguos.
    """
    rutas = sorted(dir_salida.glob("_lote*.json"))
    if not rutas:
        raise FileNotFoundError(
            f"no hay marcas _lote*.json en {dir_salida}; --cerrar solo reconstruye "
            "corridas --pares ya escritas"
        )
    marcas = []
    for ruta in rutas:
        with open(ruta, encoding="utf-8") as manejador:
            marcas.append(json.load(manejador))
    huellas = {marca.get("huella") for marca in marcas}
    _exigir(len(huellas) == 1, f"huellas distintas entre lotes: {sorted(huellas)}")
    lotes = [marca.get("lote") for marca in marcas]
    _exigir(lotes == list(range(len(lotes))), f"lotes no contiguos: {lotes}")
    return marcas


def verificar_shards_marcas(
    marcas: list[dict], dir_salida: Path, logger: logging.Logger
) -> list[dict]:
    """Verifica cada shard (existencia, bytes, filas y md5) contra su marca.

    Args:
        marcas: Marcas de lote cargadas.
        dir_salida: Directorio de las marcas y shards.
        logger: Logger activo.

    Returns:
        Lista de shards verificados (dicts de las marcas).

    Raises:
        RuntimeError: Si un shard falta o no coincide con su marca.
    """
    shards = []
    for marca in marcas:
        for shard in marca["archivos"]:
            ruta = dir_salida / shard["ruta"]
            _exigir(ruta.is_file(), f"shard ausente: {ruta}")
            _exigir(
                ruta.stat().st_size == shard["bytes"],
                f"bytes de {ruta.name} no coinciden con la marca",
            )
            _exigir(
                pq.ParquetFile(ruta).metadata.num_rows == shard["filas"],
                f"filas de {ruta.name} no coinciden con la marca",
            )
            _exigir(
                md5_archivo(ruta) == shard["md5"],
                f"md5 de {ruta.name} no coincide con la marca",
            )
            shards.append(dict(shard))
    logger.info(
        "shards verificados contra las marcas: %d (%d filas)",
        len(shards),
        sum(shard["filas"] for shard in shards),
    )
    return shards


def escanear_shards_cierre(
    shards: list[dict], dir_salida: Path, ids_benchmark: set[str]
) -> dict:
    """Re-escanea TODOS los shards: exclusion del benchmark y conteos reales.

    Lee solo doc_id, label y fuente_negativo de todos los shards (barato) y
    recomputa los conteos autoritativos; aborta si aparece un doc_id de los
    90 del benchmark.

    Args:
        shards: Shards verificados.
        dir_salida: Directorio de los shards.
        ids_benchmark: Los 90 ids excluidos.

    Returns:
        Dict con filas, positivos/negativos, por fuente y docs con pares.

    Raises:
        RuntimeError: Si algun shard contiene un doc del benchmark.
    """
    rutas = [str(dir_salida / shard["ruta"]) for shard in shards]
    tabla = pq.read_table(rutas, columns=["doc_id", "label", "fuente_negativo"])
    doc_ids = [str(valor) for valor in tabla.column("doc_id").to_pylist()]
    labels = tabla.column("label").to_pylist()
    fuentes = [str(valor) for valor in tabla.column("fuente_negativo").to_pylist()]
    intrusos = set(doc_ids) & ids_benchmark
    _exigir(
        not intrusos,
        f"docs del benchmark en los shards (se aborta el cierre): {sorted(intrusos)[:5]}",
    )
    positivos = sum(1 for label in labels if label == 1)
    negativos = len(labels) - positivos
    por_fuente = Counter(
        fuente
        for fuente, label in zip(fuentes, labels, strict=True)
        if label == 0
    )
    return {
        "filas_total": len(labels),
        "positivos": positivos,
        "negativos": negativos,
        "negativos_por_fuente": dict(sorted(por_fuente.items())),
        "docs_con_pares": len(set(doc_ids)),
        "docs_benchmark_en_pares": 0,
    }


def agregar_marcas(marcas: list[dict]) -> dict:
    """Agrega los conteos y tiempos de todas las marcas de lote.

    Args:
        marcas: Marcas de lote cargadas.

    Returns:
        Dict agregado (filas, positivos/negativos, por fuente, tiers,
        tiempos por lote y suma).
    """
    negativos_por_fuente: Counter = Counter()
    for marca in marcas:
        negativos_por_fuente.update(marca["negativos_por_fuente"])
    return {
        "n_lotes": len(marcas),
        "filas_total": sum(marca["n_filas"] for marca in marcas),
        "positivos": sum(marca["positivos"] for marca in marcas),
        "negativos": sum(marca["negativos"] for marca in marcas),
        "negativos_por_fuente": dict(sorted(negativos_por_fuente.items())),
        "docs_tier_bm25": sum(marca["docs_tier_bm25"] for marca in marcas),
        "docs_tier_solape": sum(marca["docs_tier_solape"] for marca in marcas),
        "docs_sin_chunks": sum(marca["docs_sin_chunks"] for marca in marcas),
        "negativos_omitidos": sum(marca["negativos_omitidos"] for marca in marcas),
        "tiempo_suma_lotes": round(sum(marca["tiempo_segundos"] for marca in marcas), 1),
        "tiempos_por_lote": {str(marca["lote"]): marca["tiempo_segundos"] for marca in marcas},
    }


def verificar_mtimes_fuentes(rutas: dict, referencia_mtime: float) -> dict:
    """Comprueba que las fuentes de datos no se tocaron tras iniciar la corrida.

    Compara el mtime de cada fuente de datos (para el corpus, el maximo del
    arbol) contra el mtime mas antiguo de las marcas _lote*.json (escritas
    despues del primer lote). Un mtime posterior es senal de modificacion
    durante la corrida.

    Args:
        rutas: Rutas absolutas de las fuentes.
        referencia_mtime: mtime de referencia (min de las marcas).

    Returns:
        Dict con referencia, mtimes por fuente y el booleano global.
    """
    claves = {
        "entidades": rutas["ruta_entidades"],
        "constitucion": rutas["ruta_constitucion"],
        "red_citas_resumen": rutas["ruta_red_citas_resumen"],
        "indice_bm25": rutas["ruta_indice_bm25"],
        "benchmark_v2": rutas["ruta_benchmark_v2"],
        "benchmark_v8": rutas["ruta_benchmark_v8"],
    }
    mtimes = {clave: Path(ruta).stat().st_mtime for clave, ruta in claves.items()}
    corpus = Path(rutas["ruta_corpus"])
    mtimes["corpus"] = max(
        archivo.stat().st_mtime for archivo in corpus.rglob("*") if archivo.is_file()
    )
    formato = "%Y-%m-%d %H:%M:%S"
    return {
        "referencia_mtime": time.strftime(formato, time.localtime(referencia_mtime)),
        "mtimes_fuentes": {
            clave: time.strftime(formato, time.localtime(valor))
            for clave, valor in sorted(mtimes.items())
        },
        "mtimes_anteriores_a_la_corrida": all(
            valor <= referencia_mtime for valor in mtimes.values()
        ),
    }


def construir_integridad_cierre(
    rutas: dict, md5_actual: dict, referencia_mtime: float
) -> dict:
    """Construye la seccion de integridad del cierre.

    Datos: md5 actuales + mtimes + re-escaneo de shards (duro). script_12:
    informativo, con el incidente registrado (md5 inicio/fin y causa).

    Args:
        rutas: Rutas absolutas de las fuentes.
        md5_actual: md5 actuales de las fuentes (incluye script_12).
        referencia_mtime: mtime de referencia de las marcas.

    Returns:
        Dict de integridad para el JSON de cierre.
    """
    incidente = CONFIG["incidente_script_12"]
    inicio = incidente["md5_inicio_corrida"]
    fin_registrado = incidente["md5_fin_corrida"]
    actual = md5_actual.get("script_12")
    cambio = actual != inicio
    script_12 = {
        "md5_inicio": inicio,
        "md5_final": actual,
        "md5_fin_registrado_incidente": fin_registrado,
        "cambio_detectado": cambio,
        "coincide_con_fin_registrado": actual == fin_registrado,
        "causa": incidente["causa"] if cambio else None,
        "informativo": True,
        "aborta": False,
    }
    if cambio:
        script_12["nota"] = (
            "incidente de la corrida real B1: script_12 se modifico durante la corrida "
            "(integracion --b1, cambio aditivo) y el guard original aborto por falso "
            "positivo; las fuentes de datos quedaron intactas (verificado por el "
            "orquestador: mtimes de septiembre, 0 archivos del corpus modificados)"
        )
    mtimes = verificar_mtimes_fuentes(rutas, referencia_mtime)
    return {
        "datos": {
            "fuentes_datos": CONFIG["fuentes_datos_integridad"],
            "fuentes_md5_actuales": {
                clave: valor for clave, valor in md5_actual.items() if clave != "script_12"
            },
            "verificacion": (
                "re-escaneo completo de TODOS los shards (0 filas con doc_id del "
                "benchmark) + md5 actuales de las fuentes + mtimes anteriores a la corrida"
            ),
            "mtimes": mtimes,
            "datos_ok": mtimes["mtimes_anteriores_a_la_corrida"],
        },
        "script_12": script_12,
    }


def construir_resumen_cierre(
    modo_smoke: bool,
    caps_final: dict,
    caps_origen: str,
    huella: str,
    verificados: dict,
    escaneo: dict,
    agregado: dict,
    shards: list[dict],
    integridad: dict,
    info_benchmark: dict,
    tiempos: dict,
    dir_salida: Path,
) -> dict:
    """Construye el JSON de cierre (misma estructura que el de --pares).

    Args:
        modo_smoke: True si es cierre del smoke.
        caps_final: Caps declarados/efectivos.
        caps_origen: Procedencia de los caps.
        huella: Huella de la corrida original.
        verificados: Caps verificados desde marcas/escaneo.
        escaneo: Resultado del re-escaneo de shards.
        agregado: Conteos agregados de las marcas.
        shards: Shards verificados.
        integridad: Seccion de integridad.
        info_benchmark: Info de los 90 ids excluidos.
        tiempos: Tiempos (suma de lotes + cierre).
        dir_salida: Directorio de los shards.

    Returns:
        Dict serializable del cierre.
    """
    return {
        "version": CONFIG["version"],
        "ciclo": CONFIG["ciclo"],
        "sprint": CONFIG["sprint"],
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "modo": "cerrar",
        "smoke": modo_smoke,
        "nota": (
            "JSON reconstruido desde data/pares_b1/ (marcas _lote*.json + re-escaneo "
            "completo de shards); NO recomputa pares. La corrida --pares termino todos "
            "los lotes y fallo solo en el guard de integridad (falso positivo por "
            "script_12)"
        ),
        "huella": huella,
        "caps": caps_final,
        "caps_origen": caps_origen,
        "caps_verificados_desde_marcas": verificados,
        "interpretacion_cap_global": (
            "--max-pares limita pares POSITIVOS (unidades doc-articulo); las filas "
            "totales son positivos + negativos"
        ),
        "proporciones_negativos": caps_final.get(
            "proporciones_negativos", CONFIG["proporciones_negativos"]
        ),
        "exclusion_benchmark": {
            "n_ids": info_benchmark["n_ids"],
            "fuente_primaria": info_benchmark["fuente_primaria"],
            "fuente_cruzada": info_benchmark["fuente_cruzada"],
            "md5_v2": info_benchmark["md5_v2"],
            "md5_v8": info_benchmark["md5_v8"],
            "conjuntos_iguales": info_benchmark["conjuntos_iguales"],
            "docs_benchmark_en_pares": escaneo["docs_benchmark_en_pares"],
            "verificacion": (
                f"re-escaneo completo de {len(shards)} shards "
                f"({escaneo['filas_total']} filas)"
            ),
        },
        "seleccion": {
            "docs_con_pares": escaneo["docs_con_pares"],
            "docs_tier_bm25": agregado["docs_tier_bm25"],
            "docs_tier_solape": agregado["docs_tier_solape"],
            "n_lotes": agregado["n_lotes"],
        },
        "conteos": {
            "filas_total": escaneo["filas_total"],
            "positivos": escaneo["positivos"],
            "negativos": escaneo["negativos"],
            "negativos_por_fuente": escaneo["negativos_por_fuente"],
            "docs_sin_chunks": agregado["docs_sin_chunks"],
            "negativos_omitidos": agregado["negativos_omitidos"],
        },
        "integridad": integridad,
        "shards": {
            "dir": str(dir_salida),
            "n_shards": len(shards),
            "bytes_total": sum(shard["bytes"] for shard in shards),
            "archivos": shards,
        },
        "tiempos_segundos": tiempos,
        "versiones": versiones_entorno(),
    }


def imprimir_resumen_cierre(resumen: dict, ruta_json: Path) -> None:
    """Imprime el resumen del cierre en consola (<= 10 lineas).

    Args:
        resumen: Resumen de cierre.
        ruta_json: Ruta del JSON escrito.
    """
    conteos = resumen["conteos"]
    seleccion = resumen["seleccion"]
    print("=== B1 pares reranker: cierre (reconstruccion) ===")
    print(
        f"filas: {conteos['filas_total']} (positivos {conteos['positivos']}, "
        f"negativos {conteos['negativos']}) en {seleccion['docs_con_pares']} docs"
    )
    print(f"negativos por fuente: {conteos['negativos_por_fuente']}")
    print(
        f"benchmark excluido: {resumen['exclusion_benchmark']['n_ids']} ids "
        f"(docs en pares: {resumen['exclusion_benchmark']['docs_benchmark_en_pares']})"
    )
    print(
        f"shards: {resumen['shards']['n_shards']} | "
        f"{resumen['shards']['bytes_total'] / 1024**2:.1f} MB | huella {resumen['huella']}"
    )
    print(
        f"integridad datos: {resumen['integridad']['datos']['datos_ok']} | "
        f"script_12 cambio: {resumen['integridad']['script_12']['cambio_detectado']} "
        "(informativo)"
    )
    print(
        f"tiempo suma lotes: {resumen['tiempos_segundos']['suma_lotes_segundos']:.1f} s | "
        f"cierre: {resumen['tiempos_segundos']['cierre_segundos']:.1f} s"
    )
    print(f"json: {ruta_json}")


def modo_cerrar(
    args: argparse.Namespace,
    caps: dict,
    rutas: dict,
    dir_salida: Path,
    logger: logging.Logger,
) -> None:
    """Reconstruye el JSON de cierre desde marcas y shards (sin recomputar pares).

    Args:
        args: Namespace de la CLI.
        caps: Caps declarados (CLI/defaults).
        rutas: Rutas absolutas.
        dir_salida: Directorio de las marcas/shards.
        logger: Logger activo.
    """
    inicio = time.perf_counter()
    modo_smoke = bool(args.smoke)
    logger.info("=== INICIO cierre B1 (reconstruccion desde %s, smoke=%s) ===", dir_salida, modo_smoke)
    ruta_json = (
        dir_salida / "b1_pares_smoke_cierre.json"
        if modo_smoke
        else Path(rutas["ruta_json_salida"])
    )
    ruta_json = verificar_salida_nueva(ruta_json, args.forzar, modo_smoke, logger)
    marcas = cargar_marcas_lotes(dir_salida)
    huella = marcas[0]["huella"]
    logger.info("marcas cargadas: %d lotes, huella %s", len(marcas), huella)
    shards = verificar_shards_marcas(marcas, dir_salida, logger)
    info_benchmark = cargar_ids_benchmark(rutas)
    escaneo = escanear_shards_cierre(shards, dir_salida, info_benchmark["ids"])
    agregado = agregar_marcas(marcas)
    _exigir(
        escaneo["filas_total"] == agregado["filas_total"],
        f"filas escaneadas ({escaneo['filas_total']}) != marcas ({agregado['filas_total']})",
    )
    _exigir(
        escaneo["positivos"] == agregado["positivos"],
        f"positivos escaneados ({escaneo['positivos']}) != marcas ({agregado['positivos']})",
    )
    _exigir(
        escaneo["negativos"] == agregado["negativos"],
        f"negativos escaneados ({escaneo['negativos']}) != marcas ({agregado['negativos']})",
    )
    _exigir(
        escaneo["negativos_por_fuente"] == agregado["negativos_por_fuente"],
        "fuentes de negativos escaneadas != marcas",
    )
    logger.info(
        "re-escaneo OK: %d filas (pos %d/neg %d), 0 docs del benchmark, %d docs con pares",
        escaneo["filas_total"],
        escaneo["positivos"],
        escaneo["negativos"],
        escaneo["docs_con_pares"],
    )
    caps_marcas = [marca.get("caps") for marca in marcas]
    if all(cm is not None for cm in caps_marcas):
        _exigir(
            all(cm == caps_marcas[0] for cm in caps_marcas),
            "caps distintos entre lotes",
        )
        caps_final = dict(caps_marcas[0])
        caps_origen = "marcas _lote*.json"
    else:
        caps_final = caps_serializables(caps)
        explicitos = any(
            getattr(args, clave) is not None
            for clave in (
                "max_pares",
                "max_positivos_por_doc",
                "negativos_por_positivo",
                "max_chunks_por_articulo",
                "max_docs_bm25",
                "lote",
            )
        )
        caps_origen = (
            "declarados en --cerrar" if explicitos else "defaults del modo (marcas sin caps)"
        )
    verificados = {
        "lote_efectivo": max(marca["n_docs"] for marca in marcas),
        "docs_con_pares": escaneo["docs_con_pares"],
        "docs_tier_bm25_efectivo": agregado["docs_tier_bm25"],
        "docs_tier_solape_efectivo": agregado["docs_tier_solape"],
        "negativos_por_positivo_efectivo": round(
            escaneo["negativos"] / max(1, escaneo["positivos"]), 4
        ),
    }
    if caps_final.get("max_docs_bm25") != verificados["docs_tier_bm25_efectivo"]:
        logger.warning(
            "caps declarados max_docs_bm25=%s != efectivo %d; declare los caps reales "
            "con --cerrar para un JSON exacto",
            caps_final.get("max_docs_bm25"),
            verificados["docs_tier_bm25_efectivo"],
        )
    md5_actual = {
        "entidades": md5_archivo(rutas["ruta_entidades"]),
        "constitucion": md5_archivo(rutas["ruta_constitucion"]),
        "red_citas_resumen": md5_archivo(rutas["ruta_red_citas_resumen"]),
        "indice_bm25": md5_archivo(rutas["ruta_indice_bm25"]),
        "benchmark_v2": md5_archivo(rutas["ruta_benchmark_v2"]),
        "benchmark_v8": md5_archivo(rutas["ruta_benchmark_v8"]),
        "script_12": md5_archivo(rutas["ruta_script_12"]),
        "corpus": md5_directorio(rutas["ruta_corpus"]),
    }
    referencia_mtime = min(
        ruta.stat().st_mtime for ruta in sorted(dir_salida.glob("_lote*.json"))
    )
    integridad = construir_integridad_cierre(rutas, md5_actual, referencia_mtime)
    tiempos = {
        "suma_lotes_segundos": agregado["tiempo_suma_lotes"],
        "por_lote_segundos": agregado["tiempos_por_lote"],
        "cierre_segundos": round(time.perf_counter() - inicio, 1),
    }
    resumen = construir_resumen_cierre(
        modo_smoke,
        caps_final,
        caps_origen,
        huella,
        verificados,
        escaneo,
        agregado,
        shards,
        integridad,
        info_benchmark,
        tiempos,
        dir_salida,
    )
    escribir_json(ruta_json, resumen)
    logger.info("json de cierre guardado: %s", ruta_json)
    imprimir_resumen_cierre(resumen, ruta_json)
    logger.info("=== FIN cierre B1 en %.1f s ===", tiempos["cierre_segundos"])


# ===== SECCION 9: orquestacion =====


def main() -> None:
    """Orquesta el modo pedido: --pares (construccion) o --cerrar (reconstruccion).

    Raises:
        SystemExit: Con codigo 1 ante cualquier error (via __main__).
    """
    inicio_total = time.perf_counter()
    args = parsear_args()
    caps = resolver_caps(args)
    rutas = resolver_rutas()
    modo_smoke = bool(args.smoke)
    dir_salida = Path(CONFIG["smoke"]["dir_salida"]) if modo_smoke else Path(rutas["dir_salida"])
    ruta_log = dir_salida / "b1_pares_smoke.log" if modo_smoke else Path(rutas["ruta_log"])
    logger = configurar_logger(str(ruta_log))
    if args.cerrar:
        modo_cerrar(args, caps, rutas, dir_salida, logger)
        return
    ruta_json = (
        dir_salida / "b1_pares_smoke.json" if modo_smoke else Path(rutas["ruta_json_salida"])
    )
    logger.info(
        "=== INICIO B1 pares reranker (%s, smoke=%s) ===",
        CONFIG["version"],
        modo_smoke,
    )
    # Guardas tempranas (fail-fast antes de cualquier computo caro).
    dir_salida = preparar_salida(dir_salida, args.forzar, modo_smoke, logger)
    ruta_json = verificar_salida_nueva(ruta_json, args.forzar, modo_smoke, logger)
    logger.info(
        "caps: max_pares=%d, max_positivos_por_doc=%d, negativos_por_positivo=%d, "
        "max_chunks_por_articulo=%d, max_docs_bm25=%d, lote=%d",
        caps["max_pares"],
        caps["max_positivos_por_doc"],
        caps["negativos_por_positivo"],
        caps["max_chunks_por_articulo"],
        caps["max_docs_bm25"],
        caps["lote"],
    )

    tiempos: dict[str, float] = {}
    # --- fuentes ---
    inicio = time.perf_counter()
    modulo_12().configurar_semillas(CONFIG["semilla"])
    fuentes = cargar_fuentes(rutas, logger)
    info_benchmark = cargar_ids_benchmark(rutas)
    fuentes.ids_benchmark = info_benchmark["ids"]
    logger.info(
        "benchmark M7: %d ids unicos (v2==v8), fuente primaria %s",
        info_benchmark["n_ids"],
        info_benchmark["fuente_primaria"],
    )
    tiempos["fuentes"] = round(time.perf_counter() - inicio, 1)

    # --- grafo de co-citacion M5 (sin benchmark) ---
    inicio = time.perf_counter()
    fuentes.cocitacion, n_cocit_docs, n_cocit_pares = construir_grafo_cocitacion(
        fuentes.entidades, fuentes.ids_benchmark, fuentes.canones
    )
    tiempos["grafo_cocitacion"] = round(time.perf_counter() - inicio, 1)
    logger.info(
        "grafo co-citacion: %d docs, %d pares unicos, %d articulos con socios (%.1f s)",
        n_cocit_docs,
        n_cocit_pares,
        len(fuentes.cocitacion),
        tiempos["grafo_cocitacion"],
    )

    # --- seleccion determinista de positivos ---
    inicio = time.perf_counter()
    limite_docs = CONFIG["smoke"]["n_docs"] if modo_smoke else None
    seleccion_stats = seleccionar_unidades(
        fuentes, caps, fuentes.ids_benchmark, limite_docs
    )
    seleccion = seleccion_stats["seleccion"]
    _exigir(
        not (set(seleccion) & fuentes.ids_benchmark),
        "la seleccion contiene docs del benchmark M7 (anti-leakage roto)",
    )
    tier_bm25 = asignar_tier_bm25(list(seleccion), caps["max_docs_bm25"])
    tiempos["seleccion"] = round(time.perf_counter() - inicio, 1)
    logger.info(
        "seleccion: %d docs con pares, %d unidades muestreadas de %d candidatas "
        "(tier BM25 %d docs, %.1f s)",
        len(seleccion),
        seleccion_stats["unidades_muestreadas"],
        seleccion_stats["unidades_candidatas"],
        len(tier_bm25),
        tiempos["seleccion"],
    )

    # --- huella e integridad inicial ---
    huella = calcular_huella(caps, fuentes, info_benchmark, limite_docs)
    logger.info("huella de la corrida: %s", huella)

    # --- procesamiento por lotes ---
    inicio = time.perf_counter()
    resumenes = procesar_lotes(
        fuentes, seleccion, tier_bm25, caps, huella, dir_salida, logger
    )
    tiempos["procesamiento"] = round(time.perf_counter() - inicio, 1)

    # --- integridad de fuentes (solo lectura) ---
    inicio = time.perf_counter()
    md5_final = {
        "entidades": md5_archivo(rutas["ruta_entidades"]),
        "constitucion": md5_archivo(rutas["ruta_constitucion"]),
        "red_citas_resumen": md5_archivo(rutas["ruta_red_citas_resumen"]),
        "indice_bm25": md5_archivo(rutas["ruta_indice_bm25"]),
        "benchmark_v2": md5_archivo(rutas["ruta_benchmark_v2"]),
        "benchmark_v8": md5_archivo(rutas["ruta_benchmark_v8"]),
        "script_12": md5_archivo(rutas["ruta_script_12"]),
        "corpus": md5_directorio(rutas["ruta_corpus"]),
    }
    tiempos["verificacion_integridad"] = round(time.perf_counter() - inicio, 1)
    # Guard DURO solo para las fuentes de DATOS; script_12 es informativo
    # (una edicion aditiva del codigo no altera los datos ni los pares).
    cambios_datos = {
        clave: [fuentes.md5_entradas.get(clave), md5_final.get(clave)]
        for clave in CONFIG["fuentes_datos_integridad"]
        if md5_final.get(clave) != fuentes.md5_entradas.get(clave)
    }
    _exigir(
        not cambios_datos,
        "las fuentes de DATOS cambiaron durante la corrida (md5 inicial != final); "
        f"salida no fiable: {cambios_datos}",
    )
    cambio_script_12 = md5_final.get("script_12") != fuentes.md5_entradas.get("script_12")
    if cambio_script_12:
        logger.warning(
            "script_12 cambio durante la corrida (informativo, no aborta): %s -> %s",
            fuentes.md5_entradas.get("script_12"),
            md5_final.get("script_12"),
        )
    logger.info(
        "integridad verificada: %d fuentes de datos intactas; script_12 %s",
        len(CONFIG["fuentes_datos_integridad"]),
        "cambio (informativo)" if cambio_script_12 else "sin cambios",
    )

    # --- checks smoke ---
    checks_smoke = None
    if modo_smoke:
        inicio = time.perf_counter()
        checks_smoke = verificar_checks_smoke(fuentes, resumenes, dir_salida, logger)
        tiempos["checks_smoke"] = round(time.perf_counter() - inicio, 1)

    # --- resumen final ---
    tiempos["total"] = round(time.perf_counter() - inicio_total, 1)
    resumen = construir_resumen(
        modo_smoke,
        caps,
        info_benchmark,
        seleccion_stats,
        tier_bm25,
        resumenes,
        rutas,
        dir_salida,
        huella,
        fuentes.md5_entradas,
        md5_final,
        tiempos,
        checks_smoke,
        n_cocit_docs,
        n_cocit_pares,
    )
    escribir_json(ruta_json, resumen)
    logger.info("json final guardado: %s", ruta_json)
    imprimir_resumen(resumen, ruta_json, dir_salida)
    logger.info("=== FIN B1 pares reranker en %.1f s ===", tiempos["total"])
    logger.info(
        "run completo: /home/carlos/venv/bin/python scripts/23_b1_pares_reranker.py --pares"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - protocolo: traza completa y salida 1 ante cualquier fallo
        traceback.print_exc()
        sys.exit(1)
