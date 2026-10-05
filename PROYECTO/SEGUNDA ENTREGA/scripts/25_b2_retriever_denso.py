"""25_b2_retriever_denso.py
Fecha: 2026-10-04
Descripcion: Sprint B2 del ciclo v3 - fine-tune contrastivo de un retriever
denso de dominio (bi-encoder) para la 1a etapa de M7, reutilizando los pares
de B1 (data/pares_b1/*.parquet, 23 shards, 299999 filas). Modo unico
--entrenar (con --smoke). Este script NO construye pares: solo entrena.

    Datos: lee los shards en orden canonico y por lotes (un shard en memoria
    a la vez; nunca todos los textos de golpe). Verifica el esquema exacto
    (doc_id, articulo, pasaje_id, chunk_idx, label, fuente_negativo,
    texto_chunk, texto_pasaje, n_chars_chunk, n_chars_pasaje), la huella de
    metadatos comun (b1_pares) y el ANTI-LEAKAGE: las 90 sentencias del
    benchmark M7 (ids de vinculo_cp_benchmark_v2.json y v8, ambos identicos)
    no pueden aparecer en ningun shard; si aparecen, aborta. Deja la huella
    de esa exclusion en el JSON de salida.

    Construccion de ejemplos: de las filas label=1 se forman pares
    (anchor=texto_chunk, positive=texto_pasaje). Los hard negatives
    (label=0) se enlazan al positivo correspondiente por la clave
    (doc_id, huella(texto_chunk)): si una clave tiene positivos y negativos,
    cada combinacion (positivo, negativo) es una tripleta
    (anchor, positive, negative) con hard negative explicito; las claves de
    positivo sin negativo enlazable entrenan como pares (MNRL con negativos
    in-batch). El enlace por chunk es exacto (mismo chunk de la sentencia),
    pero parcial por construccion de B1 (el chunk elegido para el articulo
    negativo no siempre coincide con el del positivo); la tasa de enlace
    queda documentada en el JSON. Cap --max-tripletas (default 200000;
    0 = todas) sobre las tripletas de train; las de validacion nunca se
    recortan.

    Split train/val POR doc_id (semilla 42): ningun par/tripleta de un mismo
    doc cae en ambos lados. --val-frac default 0.05.

    Modelo: SentenceTransformer del mejor embedding legal de A2 (ALIA-MrBERT
    legal; slug "alia" del catalogo de scripts/12_evaluar_vinculo_cp_v2.py)
    con fp16 en GPU (AMP autocast + GradScaler), loss
    MultipleNegativesRankingLoss (soporta tripletas con hard negatives en
    sentence-transformers moderno; se verifico en 6.0.1), AdamW con weight
    decay 0.01, warmup lineal 10 % del total de pasos, grad clipping 1.0,
    semilla 42. Gradient checkpointing en el encoder ACTIVADO por defecto
    (sin el, batch 64 no cabe en la GPU de 16 GB: OOM medido desde batch 48 y
    pico ~13.9 GB con batch 32; con el, batch 64 pico ~6.25 GB a ~55
    ejemplos/s, sondeo 2026-10-04); se puede desactivar con
    --sin-gradient-checkpointing. DataLoader batch --batch (default 64) con
    num_workers=0 (sin subprocesos) y progreso a log (perdida media, lr y
    ejemplos/s cada --log-cada pasos; show_progress_bar equivalente en el
    log).

    Validacion: al final de cada epoca se evalua en val la accuracy de
    retrieval (coseno anchor-positive > anchor-negative por tripleta) y un
    recall@k proxy (k=1/5/10 y MRR sobre los enlaces anchor-positive de val,
    candidatos = pasajes unicos de val). El mejor modelo por accuracy
    (desempate: recall@10) se guarda en modelos/b2_retriever/; si val no
    tuviera tripletas, se selecciona por recall@1. Checkpoints por epoca en
    modelos/b2_retriever/checkpoints/epoca_NN/ (modelo ST + estado.pt con
    optimizador, scheduler, scaler y RNG): reanudable sin --forzar.

    GPU/VRAM: exige CUDA/ROCm y VRAM libre >= --vram-minima-gb (default 7;
    el pico medido con gradient checkpointing y batch 64 es ~6.25 GB); si no,
    aborta (no se entrena en CPU). Se registra la VRAM pico.

    Salidas (run real): resultados/b2_entrenamiento_v1.json (modelo base,
    hiperparametros, tamanos, curva de loss, val por epoca, tiempos, VRAM
    pico, md5 de shards y huella de exclusion de los 90),
    experimentos/25_b2_entrenar.log, figuras/fig_b2_entrenamiento.png,
    modelos/b2_retriever/ (mejor modelo + checkpoints/). El smoke escribe
    SOLO en /tmp/opencode/b2_smoke/ y aplica los caps del diseno (2000 filas
    leidas, 20 pasos de entrenamiento; en el smoke no se guardan modelo ni
    checkpoints porque tmpfs es limitado). Cualquier flag explicito tiene
    prioridad sobre los caps del smoke.

    Reglas: sin subprocesos; nunca se sobrescribe una salida previa sin
    --forzar (con checkpoint y sin --forzar se reanuda; --forzar reinicia
    desde cero borrando SOLO el directorio propio del modelo); reanudacion
    determinista por epoca (mismo orden con la semilla 42); memoria
    consciente (un shard a la vez); tiempos impresos. El run completo NO se
    ejecuta aqui.

Estado del arte aplicado: recuperacion densa con bi-encoders contrastivos
(Karpukhin et al., 2020, DPR; Gao et al., 2021, SimCSE), hard negatives
minados por BM25/co-citacion (Nogueira y Cho, 2019; patron dominante en
statute retrieval legal: DoSSIER/COLIEE; LegalMALR 2026), fine-tune de
embeddings legales en espanol (ALIA-MrBERT legal, mejor recall de
candidatos de A2: 0.723 @100) con MultipleNegativesRankingLoss (InfoNCE con
negativos in-batch + hard negatives), AMP fp16 y warmup lineal 10 %.

Uso:
    /home/carlos/venv/bin/python scripts/25_b2_retriever_denso.py --entrenar
    /home/carlos/venv/bin/python scripts/25_b2_retriever_denso.py --entrenar --smoke

ciclo v3 - continuacion de B2 (2026-10-05): --variante SLUG redirige TODAS las
    salidas reales (modelo, JSON, log, figura y clave de metricas) con el
    sufijo dado (p. ej. ep2 -> modelos/b2_retriever_ep2,
    resultados/b2_entrenamiento_ep2.json, clave b2_retriever_ep2) para no
    tocar los artefactos del run original; y --inicializar-desde RUTA carga
    los pesos desde esa ruta en vez del modelo base (p. ej.
    --inicializar-desde modelos/b2_retriever para una 2a epoca con
    --lr 1e-5). Con --variante, la reanudacion/--forzar operan solo sobre el
    directorio de la variante.
"""

import argparse
import hashlib
import json
import logging
import math
import os
import random
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

# La GPU de 16 GB es compartida: expandable_segments evita OOM por
# fragmentacion del pool de PyTorch (mismo ajuste que el script 24).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # backend Agg fijado antes de pyplot
import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import (
    MultipleNegativesRankingLoss,
)
from torch.utils.data import DataLoader, IterableDataset
from transformers import get_linear_schedule_with_warmup

# ===== SECCION 1: configuracion =====

RAIZ = Path(__file__).resolve().parent.parent

CONFIG = {
    "semilla": 42,
    "version": "b2-retriever-denso-v1",
    "ciclo": "v3",
    "sprint": "B2",
    # Modelo base: ALIA-MrBERT legal, slug "alia" del catalogo de embeddings
    # de A2 en scripts/12_evaluar_vinculo_cp_v2.py (hf_id exacto verificado
    # alli; mejor recall de candidatos de A2: 0.723 @100). ModernBERT legal
    # ES de 150M, mean pooling, prompts vacios: sin prefijos.
    "modelo": "SINAI/ALIA-MrBERT-es-legal-administrative-embeddings",
    "max_seq_length": 512,
    # Esquema exacto de los pares de B1 (orden del diseno de 23).
    "columnas": [
        "doc_id",
        "articulo",
        "pasaje_id",
        "chunk_idx",
        "label",
        "fuente_negativo",
        "texto_chunk",
        "texto_pasaje",
        "n_chars_chunk",
        "n_chars_pasaje",
    ],
    "columnas_lectura": [
        "doc_id",
        "label",
        "texto_chunk",
        "texto_pasaje",
        "fuente_negativo",
    ],
    "clave_metadatos": b"b1_pares",
    "patron_shard": "pares_reranker_b1_shard*.parquet",
    # Rutas del run real (relativas a RAIZ) y del smoke (absolutas en /tmp).
    "dir_pares": "data/pares_b1",
    "ruta_benchmark_v2": "resultados/vinculo_cp_benchmark_v2.json",
    "ruta_benchmark_v8": "resultados/vinculo_cp_benchmark_v8.json",
    "dir_modelo": "modelos/b2_retriever",
    "ruta_json": "resultados/b2_entrenamiento_v1.json",
    "ruta_log": "experimentos/25_b2_entrenar.log",
    "ruta_figura": "figuras/fig_b2_entrenamiento.png",
    "ruta_metricas": "resultados/metricas.json",
    "clave_metricas": "b2_retriever",
    "dir_smoke": "/tmp/opencode/b2_smoke",
    # Optimizacion (fijos del diseno, no expuestos en CLI).
    "warmup_ratio": 0.10,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "amp": True,
    "log_cada": 50,
    "batch_eval": 256,
    # Gradient checkpointing en el encoder (default ON): sin el, batch 64 no
    # cabe en la GPU de 16 GB (pico medido ~13.9 GB con batch 32 y OOM con
    # batch 48+); con el, batch 64 pico ~6.25 GB y ~55 ejemplos/s (sondeo
    # 2026-10-04). Se puede desactivar con --sin-gradient-checkpointing.
    "gradient_checkpointing": True,
    # Defaults del run real y caps del smoke (el CLI explicito manda).
    "defaults": {
        "lr": 2e-5,
        "epochs": 1,
        "batch": 64,
        "val_frac": 0.05,
        "max_tripletas": 200000,
        # 7 GB: el pico medido con GC y batch 64 es ~6.25 GB; se exige margen.
        "vram_minima_gb": 7.0,
    },
    "smoke": {
        "max_filas": 2000,
        "max_pasos": 20,
        "lr": 2e-5,
        "epochs": 1,
        "batch": 64,
        "val_frac": 0.05,
        "max_tripletas": 0,
        "vram_minima_gb": 7.0,
        "log_cada": 5,
    },
}

LOGGER_NOMBRE = "b2_retriever_denso_25"
LOGGER = logging.getLogger(LOGGER_NOMBRE)


# ===== SECCION 2: utilidades (logger, md5, versiones, guardas) =====


def md5_archivo(ruta: str | Path) -> str | None:
    """Calcula el md5 de un archivo por bloques de 1 MiB.

    Args:
        ruta: Ruta del archivo.

    Returns:
        Hexdigest md5, o None si el archivo no existe.
    """
    ruta = Path(ruta)
    if not ruta.is_file():
        return None
    md5 = hashlib.md5()
    with open(ruta, "rb") as manejador:
        for bloque in iter(lambda: manejador.read(1024 * 1024), b""):
            md5.update(bloque)
    return md5.hexdigest()


def huella_texto(texto: str) -> str:
    """Huella blake2b (12 bytes) de un texto.

    Se usa como clave de enlace/deduplicacion en memoria (mas barata que
    conservar los textos completos de todos los shards).

    Args:
        texto: Texto a hashear.

    Returns:
        Hexdigest blake2b de 24 caracteres.
    """
    return hashlib.blake2b(texto.encode("utf-8"), digest_size=12).hexdigest()


def configurar_logger(ruta_log: str | Path) -> logging.Logger:
    """Configura el logger del experimento (archivo append + consola).

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


def configurar_semillas(semilla: int) -> None:
    """Fija las semillas de random/numpy/torch (CPU y GPU).

    Args:
        semilla: Semilla global del experimento.
    """
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def resolver_salidas(smoke: bool, variante: str | None = None) -> dict:
    """Resuelve las rutas de salida del modo real, del smoke o de una variante.

    Args:
        smoke: True para salidas exclusivamente en /tmp/opencode.
        variante: Sufijo de variante (p. ej. "ep2") que redirige TODAS las
            salidas reales para no tocar los artefactos del run original.

    Returns:
        Dict con dir_modelo, dir_checkpoints, ruta_json, ruta_log y
        ruta_figura.
    """
    if smoke:
        base = Path(CONFIG["dir_smoke"])
        return {
            "dir_salida": base,
            "dir_modelo": base / "modelo",
            "dir_checkpoints": base / "checkpoints",
            "ruta_json": base / "b2_entrenamiento_smoke.json",
            "ruta_log": base / "25_b2_entrenar_smoke.log",
            "ruta_figura": base / "fig_b2_entrenamiento.png",
        }
    if variante:
        dir_modelo = RAIZ / f"modelos/b2_retriever_{variante}"
        return {
            "dir_salida": dir_modelo,
            "dir_modelo": dir_modelo,
            "dir_checkpoints": dir_modelo / "checkpoints",
            "ruta_json": RAIZ / f"resultados/b2_entrenamiento_{variante}.json",
            "ruta_log": RAIZ / f"experimentos/25_b2_entrenar_{variante}.log",
            "ruta_figura": RAIZ / f"figuras/fig_b2_entrenamiento_{variante}.png",
        }
    dir_modelo = RAIZ / CONFIG["dir_modelo"]
    return {
        "dir_salida": dir_modelo,
        "dir_modelo": dir_modelo,
        "dir_checkpoints": dir_modelo / "checkpoints",
        "ruta_json": RAIZ / CONFIG["ruta_json"],
        "ruta_log": RAIZ / CONFIG["ruta_log"],
        "ruta_figura": RAIZ / CONFIG["ruta_figura"],
    }


def limpiar_salida_smoke(salidas: dict) -> None:
    """Limpia el directorio propio del smoke ANTES de abrir el log.

    El FileHandler del logger abre el log dentro de este directorio; si el
    rmtree ocurriera despues, el log quedaria en un inodo borrado y el
    archivo desapareceria del disco. Por eso la limpieza se hace primero.

    Args:
        salidas: Dict de rutas resueltas del smoke.
    """
    base = salidas["dir_salida"]
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)


def buscar_ultimo_checkpoint(dir_checkpoints: Path) -> Path | None:
    """Busca el checkpoint de epoca mas reciente y completo.

    Args:
        dir_checkpoints: Directorio de checkpoints (epoca_NN/).

    Returns:
        Ruta del checkpoint mas reciente con estado.pt y modelo/, o None.
    """
    if not dir_checkpoints.is_dir():
        return None
    candidatos = []
    for ruta in dir_checkpoints.iterdir():
        if not ruta.is_dir() or not ruta.name.startswith("epoca_"):
            continue
        if (ruta / "estado.pt").is_file() and (ruta / "modelo").is_dir():
            candidatos.append(ruta)
    if not candidatos:
        return None
    return max(candidatos, key=lambda ruta: int(ruta.name.split("_")[-1]))


def preparar_salidas(salidas: dict, forzar: bool, smoke: bool) -> Path | None:
    """Valida/crea las salidas y decide reanudacion (no sobrescribe).

    Args:
        salidas: Dict de rutas resueltas.
        forzar: True permite reiniciar desde cero (borra SOLO el directorio
            propio del modelo).
        smoke: True limpia unicamente el directorio del smoke.

    Returns:
        Checkpoint desde el que reanudar, o None para entrenar desde cero.

    Raises:
        FileExistsError: Si hay artefactos previos y no se puede reanudar
            ni se paso --forzar.
    """
    if smoke:
        salidas["dir_salida"].mkdir(parents=True, exist_ok=True)
        LOGGER.info("smoke: directorio propio listo en %s", salidas["dir_salida"])
        return None
    dir_modelo = salidas["dir_modelo"]
    tiene_artefactos = dir_modelo.exists() and any(dir_modelo.iterdir())
    checkpoint = buscar_ultimo_checkpoint(salidas["dir_checkpoints"])
    if forzar and tiene_artefactos:
        shutil.rmtree(dir_modelo)
        LOGGER.info("--forzar: se elimino el directorio propio %s", dir_modelo)
        tiene_artefactos = False
        checkpoint = None
    if tiene_artefactos and checkpoint is None:
        raise FileExistsError(
            f"ya existen artefactos en {dir_modelo} sin checkpoint reanudable; "
            "use --forzar para reiniciar (borra solo ese directorio)"
        )
    ruta_json = salidas["ruta_json"]
    if ruta_json.exists() and not forzar and checkpoint is None:
        raise FileExistsError(
            f"ya existe {ruta_json}; use --forzar para reiniciar el experimento"
        )
    if checkpoint is not None:
        LOGGER.info("reanudacion detectada: %s", checkpoint)
    return checkpoint


def versiones_entorno() -> dict:
    """Versiones del entorno de entrenamiento y datos de GPU.

    Returns:
        Dict con python/plataforma/librerias y datos de GPU.
    """
    import platform

    import pandas as pd
    import sentence_transformers
    import transformers

    return {
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pq.__version__ if hasattr(pq, "__version__") else None,
        "hip": getattr(torch.version, "hip", None),
        "cuda_disponible": bool(torch.cuda.is_available()),
        "gpu": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def escribir_json(ruta: str | Path, contenido: dict) -> None:
    """Escribe un JSON de forma atomica (tmp + reemplazo).

    Args:
        ruta: Ruta destino.
        contenido: Contenido serializable.
    """
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    with open(temporal, "w", encoding="utf-8") as manejador:
        json.dump(contenido, manejador, ensure_ascii=False, indent=1)
    os.replace(temporal, ruta)


# ===== SECCION 3: datos (shards, esquema, benchmark, split) =====


def descubrir_shards(dir_pares: Path, logger: logging.Logger) -> list[Path]:
    """Descubre los shards de pares de B1 en orden canonico.

    Args:
        dir_pares: Directorio de shards.
        logger: Logger activo.

    Returns:
        Lista de rutas ordenada.

    Raises:
        RuntimeError: Si no hay shards.
    """
    shards = sorted(dir_pares.glob(CONFIG["patron_shard"]))
    _exigir(bool(shards), f"no se encontraron shards en {dir_pares}")
    logger.info("shards descubiertos: %d en %s", len(shards), dir_pares)
    return shards


def validar_esquema_shards(shards: list[Path], logger: logging.Logger) -> dict:
    """Valida columnas, huella de metadatos y conteo de filas de los shards.

    Args:
        shards: Rutas de shards.
        logger: Logger activo.

    Returns:
        Dict con huella, version y n_shards.

    Raises:
        RuntimeError: Si el esquema, la huella o el conteo no coinciden.
    """
    huellas: set[str] = set()
    versiones: set[str] = set()
    for ruta in shards:
        archivo = pq.ParquetFile(ruta)
        nombres = archivo.schema_arrow.names
        _exigir(
            nombres == CONFIG["columnas"],
            f"esquema inesperado en {ruta.name}: {nombres}",
        )
        metadatos = archivo.metadata.metadata or {}
        crudo = metadatos.get(CONFIG["clave_metadatos"])
        _exigir(crudo is not None, f"falta la huella b1_pares en {ruta.name}")
        info = json.loads(crudo.decode("utf-8"))
        _exigir(
            int(info.get("filas", -1)) == archivo.metadata.num_rows,
            f"filas del metadato != filas reales en {ruta.name}",
        )
        huellas.add(str(info.get("huella")))
        versiones.add(str(info.get("version")))
    _exigir(len(huellas) == 1, f"huellas distintas entre shards: {sorted(huellas)}")
    _exigir(
        len(versiones) == 1,
        f"versiones distintas entre shards: {sorted(versiones)}",
    )
    huella = next(iter(huellas))
    logger.info(
        "esquema validado: %d shards, huella %s, version %s",
        len(shards),
        huella,
        next(iter(versiones)),
    )
    return {"huella": huella, "version": next(iter(versiones)), "n_shards": len(shards)}


def cargar_ids_benchmark(cfg: dict, logger: logging.Logger) -> dict:
    """Carga los 90 ids del benchmark M7 con doble fuente y verificacion.

    Fuente primaria: campo muestra de vinculo_cp_benchmark_v2.json (benchmark
    original de 90 casos). Fuente cruzada: vinculo_cp_benchmark_v8.json. Se
    exige que ambos conjuntos sean identicos y tengan 90 ids unicos.

    Args:
        cfg: Configuracion con rutas.
        logger: Logger activo.

    Returns:
        Dict con ids (set), n_ids, huella de exclusion y md5s.

    Raises:
        RuntimeError: Si los conjuntos no coinciden o no son 90 unicos.
    """
    ruta_v2 = RAIZ / cfg["ruta_benchmark_v2"]
    ruta_v8 = RAIZ / cfg["ruta_benchmark_v8"]
    with open(ruta_v2, encoding="utf-8") as manejador:
        muestra_v2 = json.load(manejador)["muestra"]
    with open(ruta_v8, encoding="utf-8") as manejador:
        muestra_v8 = json.load(manejador)["muestra"]
    ids_v2 = [str(caso["id"]) for caso in muestra_v2]
    ids_v8 = [str(caso["id"]) for caso in muestra_v8]
    _exigir(len(ids_v2) == 90 and len(set(ids_v2)) == 90, "benchmark v2: se esperaban 90 ids unicos")
    _exigir(len(ids_v8) == 90 and len(set(ids_v8)) == 90, "benchmark v8: se esperaban 90 ids unicos")
    _exigir(set(ids_v2) == set(ids_v8), "los ids del benchmark v2 y v8 no coinciden")
    ids = set(ids_v2)
    huella = hashlib.md5("|".join(sorted(ids)).encode("utf-8")).hexdigest()
    logger.info("benchmark M7 cargado: 90 ids unicos, huella de exclusion %s", huella)
    return {
        "ids": ids,
        "n_ids": 90,
        "huella_exclusion": huella,
        "fuente_primaria": str(ruta_v2),
        "fuente_cruzada": str(ruta_v8),
        "md5_v2": md5_archivo(ruta_v2),
        "md5_v8": md5_archivo(ruta_v8),
    }


def calcular_filas_por_shard(shards: list[Path], max_filas: int) -> list[int]:
    """Calcula cuantas filas se leen de cada shard (cap del smoke).

    Usa la metadata de parquet (num_rows), sin leer datos.

    Args:
        shards: Rutas de shards.
        max_filas: Maximo de filas totales (0 = sin cap).

    Returns:
        Lista de conteos por shard (en orden canonico).
    """
    filas: list[int] = []
    restante = max_filas if max_filas and max_filas > 0 else None
    for ruta in shards:
        n = pq.ParquetFile(ruta).metadata.num_rows
        if restante is not None:
            n = min(n, restante)
            restante -= n
        filas.append(int(n))
    return filas


def leer_indices(
    shards: list[Path],
    filas_por_shard: list[int],
    ids_benchmark: set[str],
    logger: logging.Logger,
) -> dict:
    """Lee los doc_id por shard y verifica la exclusion del benchmark.

    Args:
        shards: Rutas de shards.
        filas_por_shard: Filas a leer por shard.
        ids_benchmark: Ids excluidos (benchmark M7).
        logger: Logger activo.

    Returns:
        Dict con docs (set), filas leidas y contadores de benchmark.

    Raises:
        RuntimeError: Si aparece un doc del benchmark en los shards.
    """
    docs: set[str] = set()
    filas = 0
    filas_benchmark = 0
    docs_benchmark: set[str] = set()
    docs_repetidos = 0
    for ruta, n in zip(shards, filas_por_shard):
        if n == 0:
            continue
        columna = pq.read_table(ruta, columns=["doc_id"]).column("doc_id")
        valores = columna.to_pylist()[:n]
        docs_shard = set(valores)
        docs_repetidos += len(docs_shard & docs)
        docs.update(docs_shard)
        filas += n
        for valor in valores:
            if valor in ids_benchmark:
                filas_benchmark += 1
                docs_benchmark.add(valor)
    _exigir(
        filas_benchmark == 0,
        f"ANTI-LEAKAGE ROTO: {filas_benchmark} filas de {len(docs_benchmark)} "
        "docs del benchmark M7 aparecen en los shards",
    )
    _exigir(
        docs_repetidos == 0,
        f"hay {docs_repetidos} docs repartidos entre shards: el enlace por "
        "(doc_id, chunk) y el split por doc dejarian de ser exactos",
    )
    logger.info(
        "anti-leakage verificado: %d filas leidas, %d docs, 0 filas de los 90 "
        "docs del benchmark",
        filas,
        len(docs),
    )
    return {
        "docs": docs,
        "filas": filas,
        "filas_benchmark": filas_benchmark,
        "docs_benchmark": len(docs_benchmark),
    }


def dividir_docs_por_doc(
    docs: set[str], val_frac: float, semilla: int, logger: logging.Logger
) -> tuple[set[str], set[str]]:
    """Divide los doc_id en train/val con semilla fija (sin fuga por doc).

    Replica el metodo del script 24: docs ordenados, barajado con
    random.Random(f"{semilla}:split:doc"), n_val = round(val_frac * n).

    Args:
        docs: Doc ids unicos.
        val_frac: Fraccion de docs para validacion.
        semilla: Semilla del split.
        logger: Logger activo.

    Returns:
        Tupla (docs_train, docs_val) disjuntos.

    Raises:
        RuntimeError: Si los conjuntos se solapan o quedan vacios.
    """
    ordenados = sorted(docs)
    barajados = list(ordenados)
    random.Random(f"{semilla}:split:doc").shuffle(barajados)
    n_val = max(1, round(val_frac * len(ordenados)))
    docs_val = set(barajados[:n_val])
    docs_train = set(barajados[n_val:])
    _exigir(not (docs_train & docs_val), "fuga de doc_id entre train y val")
    _exigir(bool(docs_train) and bool(docs_val), "split vacio (revisar --val-frac)")
    logger.info(
        "split por doc_id: %d docs train / %d docs val (val_frac=%.3f, semilla=%d)",
        len(docs_train),
        len(docs_val),
        val_frac,
        semilla,
    )
    return docs_train, docs_val


def construir_ejemplos(
    shards: list[Path],
    filas_por_shard: list[int],
    docs_train: set[str],
    docs_val: set[str],
    cfg: dict,
    logger: logging.Logger,
) -> tuple[dict, dict]:
    """Construye tripletas (con hard negatives) y pares por shard.

    Por shard (un shard en memoria a la vez): de las filas label=1 se forman
    pares; las claves (doc_id, huella(texto_chunk)) con positivos y negativos
    generan tripletas (anchor, positive, negative) por cada combinacion
    positivo x negativo; las claves sin negativo enlazable quedan como pares.
    Los ejemplos se separan train/val por doc_id.

    Args:
        shards: Rutas de shards.
        filas_por_shard: Filas a leer por shard.
        docs_train: Docs de entrenamiento.
        docs_val: Docs de validacion.
        cfg: Configuracion efectiva.
        logger: Logger activo.

    Returns:
        Tupla (refs, stats): refs[i] = dict con arrays int32
        (tripletas_train (n,2) = [fila_pos, fila_neg], pares_train (m,) =
        [fila_pos], y las versiones _val); stats = conteos del dataset.
    """
    cap = int(cfg["max_tripletas"])
    rng_cap = random.Random(f"{cfg['semilla']}:cap:tripletas")
    refs = {
        i: {
            "tripletas_train": [],
            "pares_train": [],
            "tripletas_val": [],
            "pares_val": [],
        }
        for i in range(len(shards))
    }
    todas_train: list[tuple[int, int, int]] = []
    stats = {
        "filas_leidas": 0,
        "positivos": 0,
        "negativos": 0,
        "claves_positivas": 0,
        "claves_negativas": 0,
        "claves_negativas_enlazadas": 0,
        "filas_negativas": 0,
        "filas_negativas_enlazadas": 0,
        "negativos_por_fuente_total": Counter(),
        "fuentes_negativo_tripletas": Counter(),
    }
    for i, ruta in enumerate(shards):
        n_filas = filas_por_shard[i]
        if n_filas == 0:
            continue
        tabla = pq.read_table(ruta, columns=CONFIG["columnas_lectura"]).to_pandas()
        if n_filas < len(tabla):
            tabla = tabla.iloc[:n_filas]
        doc_arr = tabla["doc_id"].to_numpy()
        label_arr = tabla["label"].to_numpy()
        chunk_arr = tabla["texto_chunk"].to_numpy()
        pasaje_arr = tabla["texto_pasaje"].to_numpy()
        fuente_arr = tabla["fuente_negativo"].to_numpy()
        positivos: dict[tuple[str, str], list[tuple[int, str]]] = {}
        negativos: dict[tuple[str, str], list[tuple[int, str, str]]] = {}
        for fila in range(len(tabla)):
            clave = (doc_arr[fila], huella_texto(chunk_arr[fila]))
            if label_arr[fila] == 1:
                stats["positivos"] += 1
                positivos.setdefault(clave, []).append(
                    (fila, huella_texto(pasaje_arr[fila]))
                )
            else:
                stats["negativos"] += 1
                fuente = str(fuente_arr[fila])
                negativos.setdefault(clave, []).append(
                    (fila, huella_texto(pasaje_arr[fila]), fuente)
                )
                stats["negativos_por_fuente_total"][fuente] += 1
        stats["filas_leidas"] += len(tabla)
        stats["claves_positivas"] += len(positivos)
        stats["claves_negativas"] += len(negativos)
        for clave, lista_neg in negativos.items():
            stats["filas_negativas"] += len(lista_neg)
            if clave in positivos:
                stats["claves_negativas_enlazadas"] += 1
                stats["filas_negativas_enlazadas"] += len(lista_neg)
        for clave, lista_pos in positivos.items():
            lista_neg = negativos.get(clave)
            doc = clave[0]
            en_val = doc in docs_val
            if lista_neg:
                vistos: set[tuple[str, str]] = set()
                for fila_pos, huella_pos in lista_pos:
                    for fila_neg, huella_neg, fuente in lista_neg:
                        if (huella_pos, huella_neg) in vistos:
                            continue
                        vistos.add((huella_pos, huella_neg))
                        if en_val:
                            refs[i]["tripletas_val"].append((fila_pos, fila_neg))
                        else:
                            todas_train.append((i, fila_pos, fila_neg))
                            stats["fuentes_negativo_tripletas"][fuente] += 1
            else:
                vistos_p: set[str] = set()
                for fila_pos, huella_pos in lista_pos:
                    if huella_pos in vistos_p:
                        continue
                    vistos_p.add(huella_pos)
                    if en_val:
                        refs[i]["pares_val"].append((fila_pos,))
                    else:
                        refs[i]["pares_train"].append((fila_pos,))
        del tabla, doc_arr, label_arr, chunk_arr, pasaje_arr, fuente_arr
        logger.info(
            "shard %d/%d (%s): %d filas | tripletas train acumuladas %d",
            i + 1,
            len(shards),
            ruta.name,
            n_filas,
            len(todas_train),
        )
    stats["tripletas_train_antes_cap"] = len(todas_train)
    stats["cap_tripletas_aplicado"] = bool(cap > 0 and len(todas_train) > cap)
    if stats["cap_tripletas_aplicado"]:
        todas_train = rng_cap.sample(todas_train, cap)
    for i, fila_pos, fila_neg in todas_train:
        refs[i]["tripletas_train"].append((fila_pos, fila_neg))
    salida = {}
    for i in range(len(shards)):
        salida[i] = {
            "tripletas_train": np.array(
                refs[i]["tripletas_train"], dtype=np.int32
            ).reshape(-1, 2),
            "pares_train": np.array(
                refs[i]["pares_train"], dtype=np.int32
            ).reshape(-1),
            "tripletas_val": np.array(
                refs[i]["tripletas_val"], dtype=np.int32
            ).reshape(-1, 2),
            "pares_val": np.array(refs[i]["pares_val"], dtype=np.int32).reshape(-1),
        }
    stats["tripletas_train"] = int(sum(len(r["tripletas_train"]) for r in salida.values()))
    stats["pares_train"] = int(sum(len(r["pares_train"]) for r in salida.values()))
    stats["tripletas_val"] = int(sum(len(r["tripletas_val"]) for r in salida.values()))
    stats["pares_val"] = int(sum(len(r["pares_val"]) for r in salida.values()))
    stats["docs_train"] = len(docs_train)
    stats["docs_val"] = len(docs_val)
    stats["tasa_claves_enlazadas"] = (
        stats["claves_negativas_enlazadas"] / stats["claves_negativas"]
        if stats["claves_negativas"]
        else 0.0
    )
    stats["tasa_filas_enlazadas"] = (
        stats["filas_negativas_enlazadas"] / stats["filas_negativas"]
        if stats["filas_negativas"]
        else 0.0
    )
    stats["negativos_por_fuente_total"] = dict(stats["negativos_por_fuente_total"])
    stats["fuentes_negativo_tripletas"] = dict(stats["fuentes_negativo_tripletas"])
    logger.info(
        "ejemplos: %d tripletas train (%d antes del cap) + %d pares train | "
        "val: %d tripletas + %d pares",
        stats["tripletas_train"],
        stats["tripletas_train_antes_cap"],
        stats["pares_train"],
        stats["tripletas_val"],
        stats["pares_val"],
    )
    logger.info(
        "enlace de hard negatives: %d/%d claves negativas enlazables (%.1f %%), "
        "%d/%d filas negativas enlazadas (%.1f %%)",
        stats["claves_negativas_enlazadas"],
        stats["claves_negativas"],
        100.0 * stats["tasa_claves_enlazadas"],
        stats["filas_negativas_enlazadas"],
        stats["filas_negativas"],
        100.0 * stats["tasa_filas_enlazadas"],
    )
    return salida, stats


def calcular_md5_shards(shards: list[Path], usados: list[bool], logger: logging.Logger) -> dict:
    """Calcula el md5 de los shards usados y un md5 combinado.

    Args:
        shards: Rutas de shards.
        usados: Mascara de shards efectivamente leidos.
        logger: Logger activo.

    Returns:
        Dict con md5 por shard (nombre -> md5) y md5 combinado.
    """
    por_shard = {
        ruta.name: md5_archivo(ruta)
        for ruta, usar in zip(shards, usados)
        if usar
    }
    combinado = hashlib.md5(
        "\n".join(f"{k}:{v}" for k, v in sorted(por_shard.items())).encode("utf-8")
    ).hexdigest()
    logger.info("md5 de %d shards calculado (combinado %s)", len(por_shard), combinado)
    return {"por_shard": por_shard, "combinado": combinado, "n_shards": len(por_shard)}


# ===== SECCION 4: GPU, modelo y optimizacion =====


def exigir_gpu(vram_minima_gb: float, logger: logging.Logger) -> dict:
    """Exige CUDA/ROCm disponible y VRAM libre suficiente.

    Args:
        vram_minima_gb: VRAM libre minima en GiB.
        logger: Logger activo.

    Returns:
        Dict con nombre de GPU, VRAM libre y total en GB.

    Raises:
        RuntimeError: Si no hay CUDA/ROCm o la VRAM libre es insuficiente.
    """
    _exigir(
        torch.cuda.is_available(),
        "no hay CUDA/ROCm disponible: este script entrena SOLO en GPU",
    )
    libre, total = torch.cuda.mem_get_info()
    libre_gb = libre / 1e9
    _exigir(
        libre_gb >= vram_minima_gb,
        f"VRAM libre insuficiente: {libre_gb:.2f} GB < {vram_minima_gb:.1f} GB "
        f"(total {total / 1e9:.2f} GB); libere GPU o baje --vram-minima-gb",
    )
    info = {
        "gpu": torch.cuda.get_device_name(0),
        "vram_libre_gb": round(libre_gb, 2),
        "vram_total_gb": round(total / 1e9, 2),
    }
    logger.info(
        "GPU %s: %.2f GB libres de %.2f GB (minimo exigido %.1f GB)",
        info["gpu"],
        info["vram_libre_gb"],
        info["vram_total_gb"],
        vram_minima_gb,
    )
    return info


def cargar_modelo(cfg: dict, logger: logging.Logger) -> SentenceTransformer:
    """Carga el SentenceTransformer base en GPU.

    Args:
        cfg: Configuracion efectiva.
        logger: Logger activo.

    Returns:
        Modelo cargado en cuda con max_seq_length fijado.
    """
    t0 = time.time()
    modelo = SentenceTransformer(cfg["modelo"], device="cuda")
    modelo.max_seq_length = int(cfg["max_seq_length"])
    logger.info(
        "modelo %s cargado en %.1f s (max_seq_length=%d, dim=%d)",
        cfg["modelo"],
        time.time() - t0,
        modelo.max_seq_length,
        modelo.get_sentence_embedding_dimension(),
    )
    return modelo


def activar_gradient_checkpointing(
    modelo: SentenceTransformer, cfg: dict, logger: logging.Logger
) -> None:
    """Activa gradient checkpointing en el encoder si la configuracion lo pide.

    Sin gradient checkpointing, batch 64 no cabe en la GPU de 16 GB (OOM
    medido desde batch 48; batch 32 pico ~13.9 GB); con el activado, batch 64
    pico ~6.25 GB a ~55 ejemplos/s (sondeo 2026-10-04). El costo es re-calcular
    el forward en el backward (~20-30 % mas lento por paso).

    Args:
        modelo: SentenceTransformer cargado.
        cfg: Configuracion efectiva (gradient_checkpointing).
        logger: Logger activo.

    Raises:
        RuntimeError: Si no se pudo activar en el encoder.
    """
    if not cfg["gradient_checkpointing"]:
        logger.info(
            "gradient checkpointing desactivado (--sin-gradient-checkpointing)"
        )
        return
    modelo[0].auto_model.gradient_checkpointing_enable()
    _exigir(
        bool(getattr(modelo[0].auto_model, "is_gradient_checkpointing", False)),
        "no se pudo activar gradient checkpointing en el encoder",
    )
    logger.info(
        "gradient checkpointing activado en el encoder (batch %d cabe en "
        "~6.3 GB)",
        cfg["batch"],
    )


def construir_optimizador(modelo: SentenceTransformer, cfg: dict) -> torch.optim.Optimizer:
    """Construye AdamW con weight decay desactivado en bias/LayerNorm.

    Args:
        modelo: Modelo a optimizar.
        cfg: Configuracion efectiva (lr, weight_decay).

    Returns:
        Optimizador AdamW.
    """
    # ModernBERT (ALIA-MrBERT) no usa biases y nombra sus normas
    # norm/attn_norm/mlp_norm/final_norm (sin submodulo "LayerNorm"): se
    # excluyen tambien esos pesos del weight decay.
    sin_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
    ]
    grupos = [
        {
            "params": [
                parametro
                for nombre, parametro in modelo.named_parameters()
                if not any(patron in nombre for patron in sin_decay)
            ],
            "weight_decay": cfg["weight_decay"],
        },
        {
            "params": [
                parametro
                for nombre, parametro in modelo.named_parameters()
                if any(patron in nombre for patron in sin_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(grupos, lr=cfg["lr"])


def preprocesar_lote(modelo: SentenceTransformer, textos: list[str]) -> dict:
    """Tokeniza un lote y mueve los tensores a la GPU.

    Args:
        modelo: SentenceTransformer (con tokenizer/preprocess).
        textos: Lista de textos.

    Returns:
        Dict de features con los tensores en el device del modelo.
    """
    device = next(modelo.parameters()).device
    features = modelo.preprocess(textos)
    return {
        clave: (valor.to(device) if torch.is_tensor(valor) else valor)
        for clave, valor in features.items()
    }


# ===== SECCION 5: iterador de ejemplos (streaming por shard) =====


class IteradorEjemplos(IterableDataset):
    """IterableDataset que emite lotes homogeneos leyendo un shard a la vez.

    Por epoca: baraja el orden de shards y, dentro de cada shard, baraja las
    tripletas y los pares y emite lotes intercalados (un lote de tripletas,
    uno de pares, ...). Cada lote es una lista de dicts
    {"anchor", "positive"} o {"anchor", "positive", "negative"}; nunca se
    mezclan ambos tipos en un lote (el loss necesita columnas uniformes).
    Determinista por (semilla, epoca).
    """

    def __init__(
        self,
        refs: dict,
        shards: list[Path],
        batch: int,
        semilla: int,
    ) -> None:
        """Inicializa el iterador.

        Args:
            refs: Dict por shard con arrays de tripletas/pares train.
            shards: Rutas de shards.
            batch: Tamano de lote.
            semilla: Semilla global.
        """
        self.refs = refs
        self.shards = shards
        self.batch = batch
        self.semilla = semilla
        self.epoca = 0

    def set_epoca(self, epoca: int) -> None:
        """Fija la epoca (controla el barajado deterministico).

        Args:
            epoca: Numero de epoca (0-based).
        """
        self.epoca = epoca

    def _lotes(
        self,
        arreglo: np.ndarray,
        chunks: np.ndarray,
        pasajes: np.ndarray,
        es_tripleta: bool,
        rng: random.Random,
    ):
        """Genera lotes de ejemplos de un arreglo de referencias.

        Args:
            arreglo: Array (n, 2) de tripletas o (n,) de pares.
            chunks: Textos de chunk del shard (indexados por fila).
            pasajes: Textos de pasaje del shard (indexados por fila).
            es_tripleta: True si el arreglo contiene tripletas.
            rng: Generador aleatorio de la epoca.

        Yields:
            Listas de dicts con los textos del lote.
        """
        n = len(arreglo)
        if n == 0:
            return
        orden = list(range(n))
        rng.shuffle(orden)
        for inicio in range(0, n, self.batch):
            indices = orden[inicio : inicio + self.batch]
            lote = []
            for j in indices:
                if es_tripleta:
                    fila_pos = int(arreglo[j, 0])
                    fila_neg = int(arreglo[j, 1])
                    lote.append(
                        {
                            "anchor": str(chunks[fila_pos]),
                            "positive": str(pasajes[fila_pos]),
                            "negative": str(pasajes[fila_neg]),
                        }
                    )
                else:
                    fila = int(arreglo[j])
                    lote.append(
                        {
                            "anchor": str(chunks[fila]),
                            "positive": str(pasajes[fila]),
                        }
                    )
            yield lote

    def __iter__(self):
        """Itera los lotes de una epoca (un shard en memoria a la vez).

        Yields:
            Listas de dicts (lotes homogeneos de tripletas o pares).
        """
        rng = random.Random(f"{self.semilla}:epoca:{self.epoca}")
        orden = sorted(self.refs.keys())
        rng.shuffle(orden)
        for i in orden:
            r = self.refs[i]
            if len(r["tripletas_train"]) == 0 and len(r["pares_train"]) == 0:
                continue
            tabla = pq.read_table(
                self.shards[i], columns=["texto_chunk", "texto_pasaje"]
            ).to_pandas()
            chunks = tabla["texto_chunk"].to_numpy()
            pasajes = tabla["texto_pasaje"].to_numpy()
            generador_t = self._lotes(
                r["tripletas_train"], chunks, pasajes, True, rng
            )
            generador_p = self._lotes(r["pares_train"], chunks, pasajes, False, rng)
            activo_t = True
            activo_p = True
            while activo_t or activo_p:
                if activo_t:
                    try:
                        yield next(generador_t)
                    except StopIteration:
                        activo_t = False
                if activo_p:
                    try:
                        yield next(generador_p)
                    except StopIteration:
                        activo_p = False
            del tabla, chunks, pasajes

    def __len__(self) -> int:
        """Numero total de lotes de una epoca.

        Returns:
            Suma de lotes de tripletas y pares de todos los shards.
        """
        total = 0
        for r in self.refs.values():
            total += math.ceil(len(r["tripletas_train"]) / self.batch)
            total += math.ceil(len(r["pares_train"]) / self.batch)
        return total


# ===== SECCION 6: evaluacion en validacion =====


def evaluar_validacion(
    modelo: SentenceTransformer,
    refs: dict,
    shards: list[Path],
    cfg: dict,
    logger: logging.Logger,
) -> dict | None:
    """Evalua accuracy de tripletas y recall@k proxy sobre validacion.

    Codifica (con embeddings normalizados) los textos unicos de val y calcula:
    (1) accuracy de retrieval: coseno(anchor, positive) > coseno(anchor,
    negative) por tripleta; (2) recall@k (k=1/5/10) y MRR sobre los enlaces
    anchor-positive (candidatos = pasajes unicos de val), un proxy barato del
    recall de retrieval de la 1a etapa.

    Args:
        modelo: Modelo a evaluar.
        refs: Dict de referencias por shard (usa las claves _val).
        shards: Rutas de shards.
        cfg: Configuracion efectiva.
        logger: Logger activo.

    Returns:
        Dict de metricas, o None si val no tiene ejemplos.
    """
    t0 = time.time()
    tripletas: list[tuple[str, str, str]] = []
    enlaces: list[tuple[str, str]] = []
    for i, r in refs.items():
        if len(r["tripletas_val"]) == 0 and len(r["pares_val"]) == 0:
            continue
        tabla = pq.read_table(
            shards[i], columns=["texto_chunk", "texto_pasaje"]
        ).to_pandas()
        chunks = tabla["texto_chunk"].to_numpy()
        pasajes = tabla["texto_pasaje"].to_numpy()
        for fila_pos, fila_neg in r["tripletas_val"]:
            ancla = str(chunks[fila_pos])
            positivo = str(pasajes[fila_pos])
            tripletas.append((ancla, positivo, str(pasajes[fila_neg])))
            enlaces.append((ancla, positivo))
        for fila_pos in r["pares_val"]:
            fila_pos = int(fila_pos)
            enlaces.append((str(chunks[fila_pos]), str(pasajes[fila_pos])))
        del tabla, chunks, pasajes
    if not tripletas and not enlaces:
        logger.warning("validacion sin ejemplos: no se puede evaluar")
        return None
    anclas_unicas: dict[str, int] = {}
    pasajes_unicos: dict[str, int] = {}
    for ancla, positivo, _ in tripletas:
        anclas_unicas.setdefault(ancla, len(anclas_unicas))
        pasajes_unicos.setdefault(positivo, len(pasajes_unicos))
    for ancla, positivo in enlaces:
        anclas_unicas.setdefault(ancla, len(anclas_unicas))
        pasajes_unicos.setdefault(positivo, len(pasajes_unicos))
    for _, _, negativo in tripletas:
        pasajes_unicos.setdefault(negativo, len(pasajes_unicos))
    lista_anclas = list(anclas_unicas)
    lista_pasajes = list(pasajes_unicos)
    modelo.eval()
    with torch.inference_mode():
        emb_anclas = modelo.encode(
            lista_anclas,
            batch_size=cfg["batch_eval"],
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        emb_pasajes = modelo.encode(
            lista_pasajes,
            batch_size=cfg["batch_eval"],
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
    ea = emb_anclas.float().cpu().numpy()
    ep = emb_pasajes.float().cpu().numpy()
    resultado = {
        "n_tripletas": len(tripletas),
        "n_pares": len(enlaces) - len(tripletas),
        "n_anclas": len(lista_anclas),
        "n_pasajes": len(lista_pasajes),
    }
    if tripletas:
        ia = np.array([anclas_unicas[t[0]] for t in tripletas])
        ip = np.array([pasajes_unicos[t[1]] for t in tripletas])
        ineg = np.array([pasajes_unicos[t[2]] for t in tripletas])
        sim_pos = np.einsum("ij,ij->i", ea[ia], ep[ip])
        sim_neg = np.einsum("ij,ij->i", ea[ia], ep[ineg])
        resultado["accuracy_tripletas"] = float((sim_pos > sim_neg).mean())
    else:
        resultado["accuracy_tripletas"] = None
    ia_en = np.array([anclas_unicas[a] for a, _ in enlaces])
    ip_en = np.array([pasajes_unicos[p] for _, p in enlaces])
    similitudes = ea @ ep.T
    orden = np.argsort(-similitudes, axis=1)
    rangos = np.empty_like(orden)
    rangos[np.arange(len(lista_anclas))[:, None], orden] = np.arange(
        1, len(lista_pasajes) + 1
    )
    rango_enlace = rangos[ia_en, ip_en]
    for k in (1, 5, 10):
        resultado[f"recall@{k}"] = float((rango_enlace <= k).mean())
    resultado["mrr"] = float((1.0 / rango_enlace).mean())
    resultado["accuracy_seleccion"] = (
        resultado["accuracy_tripletas"]
        if resultado["accuracy_tripletas"] is not None
        else resultado["recall@1"]
    )
    resultado["segundos"] = round(time.time() - t0, 1)
    logger.info(
        "val: %d tripletas + %d pares | accuracy %.4f | recall@1 %.4f | "
        "recall@5 %.4f | recall@10 %.4f | MRR %.4f | %.1f s",
        resultado["n_tripletas"],
        resultado["n_pares"],
        resultado["accuracy_tripletas"]
        if resultado["accuracy_tripletas"] is not None
        else float("nan"),
        resultado["recall@1"],
        resultado["recall@5"],
        resultado["recall@10"],
        resultado["mrr"],
        resultado["segundos"],
    )
    return resultado


# ===== SECCION 7: checkpoints (guardado y reanudacion) =====


def capturar_rng() -> dict:
    """Captura el estado de los generadores aleatorios.

    Returns:
        Dict con estados de python/numpy/torch/cuda.
    """
    estado = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        estado["cuda"] = torch.cuda.get_rng_state_all()
    return estado


def restaurar_rng(estado: dict) -> None:
    """Restaura el estado de los generadores aleatorios.

    Args:
        estado: Dict guardado por capturar_rng.
    """
    random.setstate(estado["python"])
    np.random.set_state(estado["numpy"])
    torch.set_rng_state(estado["torch"])
    if torch.cuda.is_available() and "cuda" in estado:
        torch.cuda.set_rng_state_all(estado["cuda"])


def guardar_checkpoint(
    modelo: SentenceTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    estado: dict,
    dir_checkpoints: Path,
    epoca: int,
    logger: logging.Logger,
) -> Path:
    """Guarda checkpoint reanudable de una epoca (modelo ST + estado.pt).

    Args:
        modelo: Modelo entrenado.
        optimizer: Optimizador.
        scheduler: Scheduler de lr.
        scaler: GradScaler de AMP.
        estado: Estado acumulado (pasos, mejor, historial, tiempos, RNG).
        dir_checkpoints: Directorio de checkpoints.
        epoca: Epoca completada (1-based).
        logger: Logger activo.

    Returns:
        Ruta del checkpoint guardado.
    """
    destino = dir_checkpoints / f"epoca_{epoca:02d}"
    destino.mkdir(parents=True, exist_ok=True)
    modelo.save(str(destino / "modelo"))
    contenido = {
        "epoca": epoca,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": capturar_rng(),
        **estado,
    }
    torch.save(contenido, destino / "estado.pt")
    logger.info("checkpoint guardado en %s", destino)
    return destino


def cargar_estado_checkpoint(checkpoint: Path, logger: logging.Logger) -> dict:
    """Carga el estado.pt de un checkpoint (sin el modelo).

    Args:
        checkpoint: Directorio del checkpoint (epoca_NN).
        logger: Logger activo.

    Returns:
        Dict de estado guardado.
    """
    estado = torch.load(checkpoint / "estado.pt", weights_only=False)
    logger.info(
        "estado cargado: epoca %s, paso %s, mejor %s",
        estado.get("epoca"),
        estado.get("paso_global"),
        estado.get("mejor_metrica"),
    )
    return estado


# ===== SECCION 8: bucle de entrenamiento =====


def _perdida_lote(
    criterio: MultipleNegativesRankingLoss,
    modelo: SentenceTransformer,
    lote: list[dict],
    amp: bool,
) -> torch.Tensor:
    """Calcula la loss MNRL de un lote homogeneo (pares o tripletas).

    Args:
        criterio: Loss MultipleNegativesRankingLoss.
        modelo: Modelo (para tokenizar con preprocess).
        lote: Lista de dicts con anchor/positive y opcional negative.
        amp: True si se usa autocast fp16.

    Returns:
        Loss escalar del lote.
    """
    anclas = [ejemplo["anchor"] for ejemplo in lote]
    positivos = [ejemplo["positive"] for ejemplo in lote]
    features = [preprocesar_lote(modelo, anclas), preprocesar_lote(modelo, positivos)]
    if "negative" in lote[0]:
        negativos = [ejemplo["negative"] for ejemplo in lote]
        features.append(preprocesar_lote(modelo, negativos))
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
        return criterio(features, labels=torch.tensor([]))


def entrenar(
    modelo: SentenceTransformer,
    refs: dict,
    shards: list[Path],
    docs_val: set[str],
    cfg: dict,
    salidas: dict,
    estado_previo: dict | None,
    guardar: bool,
    logger: logging.Logger,
) -> dict:
    """Entrena por epocas con AMP, evalua en val y guarda el mejor modelo.

    Args:
        modelo: Modelo base (o cargado desde checkpoint al reanudar).
        refs: Referencias por shard (tripletas/pares train y val).
        shards: Rutas de shards.
        docs_val: Docs de validacion (informativo).
        cfg: Configuracion efectiva.
        salidas: Rutas resueltas.
        estado_previo: Estado de checkpoint para reanudar, o None.
        guardar: True en run real (modelo/checkpoints); False en smoke.
        logger: Logger activo.

    Returns:
        Dict con historial (curva de loss, val por epoca, mejor, tiempos,
        VRAM pico, pasos).

    Raises:
        RuntimeError: Si el checkpoint no es compatible con la configuracion.
    """
    device = next(modelo.parameters()).device
    criterio = MultipleNegativesRankingLoss(modelo)
    optimizer = construir_optimizador(modelo, cfg)
    iterador = IteradorEjemplos(refs, shards, cfg["batch"], cfg["semilla"])
    pasos_por_epoca = len(iterador)
    pasos_totales = pasos_por_epoca * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(cfg["warmup_ratio"] * pasos_totales)),
        num_training_steps=max(1, pasos_totales),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])
    historial = {
        "curva_loss": [],
        "val_por_epoca": [],
        "mejor_metrica": None,
        "mejor_epoca": None,
        "paso_global": 0,
        "epoca_inicio": 0,
        "tiempo_entrenamiento_s": 0.0,
        "tiempo_evaluacion_s": 0.0,
        "por_epoca_s": [],
        "vram_pico_gb": 0.0,
        "reanudado": False,
    }
    if estado_previo is not None:
        _exigir(
            estado_previo.get("epochs") == cfg["epochs"]
            and estado_previo.get("batch") == cfg["batch"],
            "checkpoint incompatible: reanude con el mismo --epochs y --batch",
        )
        optimizer.load_state_dict(estado_previo["optimizer"])
        scheduler.load_state_dict(estado_previo["scheduler"])
        scaler.load_state_dict(estado_previo["scaler"])
        restaurar_rng(estado_previo["rng"])
        historial.update(
            {
                "curva_loss": estado_previo.get("curva_loss", []),
                "val_por_epoca": estado_previo.get("val_por_epoca", []),
                "mejor_metrica": estado_previo.get("mejor_metrica"),
                "mejor_epoca": estado_previo.get("mejor_epoca"),
                "paso_global": int(estado_previo.get("paso_global", 0)),
                "epoca_inicio": int(estado_previo.get("epoca", 0)),
                "tiempo_entrenamiento_s": float(
                    estado_previo.get("tiempo_entrenamiento_s", 0.0)
                ),
                "tiempo_evaluacion_s": float(
                    estado_previo.get("tiempo_evaluacion_s", 0.0)
                ),
                "por_epoca_s": estado_previo.get("por_epoca_s", []),
                "vram_pico_gb": float(estado_previo.get("vram_pico_gb", 0.0)),
                "reanudado": True,
            }
        )
    logger.info(
        "entrenamiento: %d pasos/epoca x %d epocas = %d pasos (warmup %d) | "
        "batch %d | lr %.1e | AMP %s",
        pasos_por_epoca,
        cfg["epochs"],
        pasos_totales,
        max(1, int(cfg["warmup_ratio"] * pasos_totales)),
        cfg["batch"],
        cfg["lr"],
        cfg["amp"],
    )
    torch.cuda.reset_peak_memory_stats()
    max_pasos = int(cfg.get("max_pasos", 0) or 0)
    t_inicio = time.time()
    ejemplos_entrenados = 0
    pasos_ejecutados = 0
    for epoca in range(historial["epoca_inicio"], cfg["epochs"]):
        iterador.set_epoca(epoca)
        cargador = DataLoader(iterador, batch_size=None, num_workers=0)
        modelo.train()
        t_epoca = time.time()
        ventana: list[float] = []
        suma_ventana = 0.0
        suma_epoca = 0.0
        n_epoca = 0
        t_ultimo_log = time.time()
        ejemplos_ultimo_log = 0
        pasos_epoca = 0
        for lote in cargador:
            if max_pasos and pasos_ejecutados >= max_pasos:
                break
            perdida = _perdida_lote(criterio, modelo, lote, cfg["amp"])
            scaler.scale(perdida).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), cfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            valor = float(perdida.detach())
            ventana.append(valor)
            suma_ventana += valor
            suma_epoca += valor
            n_epoca += 1
            ejemplos_entrenados += len(lote)
            pasos_ejecutados += 1
            pasos_epoca += 1
            historial["paso_global"] += 1
            if pasos_ejecutados % cfg["log_cada"] == 0:
                ahora = time.time()
                lr_actual = scheduler.get_last_lr()[0]
                velocidad = (ejemplos_entrenados - ejemplos_ultimo_log) / max(
                    ahora - t_ultimo_log, 1e-9
                )
                logger.info(
                    "epoca %d | paso %d/%d | perdida %.4f | lr %.3e | "
                    "%.1f ejemplos/s",
                    epoca + 1,
                    pasos_epoca,
                    pasos_por_epoca,
                    suma_ventana / len(ventana),
                    lr_actual,
                    velocidad,
                )
                historial["curva_loss"].append(
                    {
                        "epoca": epoca + 1,
                        "paso": historial["paso_global"],
                        "perdida": suma_ventana / len(ventana),
                        "lr": lr_actual,
                    }
                )
                ventana = []
                suma_ventana = 0.0
                t_ultimo_log = ahora
                ejemplos_ultimo_log = ejemplos_entrenados
        if ventana:
            historial["curva_loss"].append(
                {
                    "epoca": epoca + 1,
                    "paso": historial["paso_global"],
                    "perdida": suma_ventana / len(ventana),
                    "lr": scheduler.get_last_lr()[0],
                }
            )
        segundos_epoca = time.time() - t_epoca
        historial["por_epoca_s"].append(round(segundos_epoca, 1))
        historial["tiempo_entrenamiento_s"] += segundos_epoca
        vram_actual = torch.cuda.max_memory_allocated() / 1e9
        historial["vram_pico_gb"] = max(historial["vram_pico_gb"], vram_actual)
        logger.info(
            "epoca %d completada: %d pasos, %.1f s (%.1f ejemplos/s), "
            "perdida media %.4f, VRAM pico %.2f GB",
            epoca + 1,
            pasos_epoca,
            segundos_epoca,
            ejemplos_entrenados / max(segundos_epoca, 1e-9),
            suma_epoca / max(n_epoca, 1),
            vram_actual,
        )
        metricas_val = evaluar_validacion(modelo, refs, shards, cfg, logger)
        historial["tiempo_evaluacion_s"] += (
            metricas_val["segundos"] if metricas_val else 0.0
        )
        if metricas_val is not None:
            metricas_val = {"epoca": epoca + 1, **metricas_val}
            mejora = (
                historial["mejor_metrica"] is None
                or metricas_val["accuracy_seleccion"] > historial["mejor_metrica"]
            )
            metricas_val["mejor"] = bool(mejora)
            historial["val_por_epoca"].append(metricas_val)
            if mejora:
                historial["mejor_metrica"] = metricas_val["accuracy_seleccion"]
                historial["mejor_epoca"] = epoca + 1
                if guardar:
                    modelo.save(str(salidas["dir_modelo"]))
                    logger.info(
                        "nuevo mejor modelo (epoca %d, %s=%.4f) guardado en %s",
                        epoca + 1,
                        "accuracy" if metricas_val["accuracy_tripletas"] is not None else "recall@1",
                        metricas_val["accuracy_seleccion"],
                        salidas["dir_modelo"],
                    )
        if guardar:
            estado_ckpt = {
                "epochs": cfg["epochs"],
                "batch": cfg["batch"],
                "lr": cfg["lr"],
                "curva_loss": historial["curva_loss"],
                "val_por_epoca": historial["val_por_epoca"],
                "mejor_metrica": historial["mejor_metrica"],
                "mejor_epoca": historial["mejor_epoca"],
                "paso_global": historial["paso_global"],
                "pasos_por_epoca": pasos_por_epoca,
                "tiempo_entrenamiento_s": historial["tiempo_entrenamiento_s"],
                "tiempo_evaluacion_s": historial["tiempo_evaluacion_s"],
                "por_epoca_s": historial["por_epoca_s"],
                "vram_pico_gb": historial["vram_pico_gb"],
            }
            guardar_checkpoint(
                modelo,
                optimizer,
                scheduler,
                scaler,
                estado_ckpt,
                salidas["dir_checkpoints"],
                epoca + 1,
                logger,
            )
        if max_pasos and pasos_ejecutados >= max_pasos:
            logger.info("tope de pasos del smoke alcanzado (%d)", max_pasos)
            break
    historial["tiempo_total_s"] = round(time.time() - t_inicio, 1)
    historial["pasos_ejecutados"] = pasos_ejecutados
    historial["ejemplos_entrenados"] = ejemplos_entrenados
    historial["vram_pico_gb"] = round(
        max(historial["vram_pico_gb"], torch.cuda.max_memory_allocated() / 1e9), 2
    )
    historial["pasos_por_epoca"] = pasos_por_epoca
    historial["pasos_totales"] = pasos_totales
    historial["docs_val"] = len(docs_val)
    historial["device"] = str(device)
    if guardar and historial["mejor_epoca"] is None:
        modelo.save(str(salidas["dir_modelo"]))
        logger.warning(
            "val sin metricas: se guarda el modelo final en %s",
            salidas["dir_modelo"],
        )
    return historial


# ===== SECCION 9: salidas (figura, JSON, metricas, resumen) =====


def graficar_entrenamiento(historial: dict, ruta_figura: Path, cfg: dict) -> None:
    """Genera la figura de entrenamiento (loss y metricas de val).

    Args:
        historial: Historial de entrenamiento.
        ruta_figura: Ruta de salida de la figura.
        cfg: Configuracion efectiva.
    """
    ruta_figura.parent.mkdir(parents=True, exist_ok=True)
    figura, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    curva = historial["curva_loss"]
    if curva:
        ejes[0].plot(
            [p["paso"] for p in curva],
            [p["perdida"] for p in curva],
            marker="o",
            markersize=3,
            linewidth=1.2,
            color="#1f77b4",
        )
        ejes[0].set_xlabel("paso")
        ejes[0].set_ylabel("perdida MNRL (media de ventana)")
        ejes[0].set_title("Curva de loss")
        ejes[0].grid(alpha=0.3)
    else:
        ejes[0].text(0.5, 0.5, "sin curva de loss", ha="center", va="center")
    val = historial["val_por_epoca"]
    if val:
        epocas = [m["epoca"] for m in val]
        acc = [
            m["accuracy_tripletas"] if m["accuracy_tripletas"] is not None else np.nan
            for m in val
        ]
        ejes[1].plot(epocas, acc, marker="o", label="accuracy tripletas")
        ejes[1].plot(
            epocas,
            [m["recall@10"] for m in val],
            marker="s",
            label="recall@10 (proxy)",
        )
        ejes[1].plot(
            epocas,
            [m["recall@1"] for m in val],
            marker="^",
            label="recall@1 (proxy)",
        )
        ejes[1].set_xlabel("epoca")
        ejes[1].set_ylabel("metrica")
        ejes[1].set_title("Validacion")
        ejes[1].grid(alpha=0.3)
        ejes[1].legend(fontsize=8)
        if historial["mejor_epoca"] is not None:
            ejes[1].axvline(
                historial["mejor_epoca"],
                color="gray",
                linestyle="--",
                linewidth=1,
                label="mejor",
            )
    else:
        ejes[1].text(0.5, 0.5, "sin metricas de val", ha="center", va="center")
    figura.suptitle(
        f"B2 retriever denso | {cfg['modelo'].split('/')[-1]} | "
        f"{cfg['epochs']} epoca(s), batch {cfg['batch']}, lr {cfg['lr']:.1e}"
    )
    figura.tight_layout()
    figura.savefig(ruta_figura, dpi=150)
    plt.close(figura)


def construir_resumen(
    cfg: dict,
    esquema: dict,
    benchmark: dict,
    indices: dict,
    stats: dict,
    md5s: dict,
    info_gpu: dict,
    historial: dict,
    smoke: bool,
    salidas: dict,
) -> dict:
    """Construye el resumen serializable del experimento.

    Args:
        cfg: Configuracion efectiva.
        esquema: Info de esquema/huella de los shards.
        benchmark: Info del benchmark y su huella de exclusion.
        indices: Info de la lectura de indices (anti-leakage).
        stats: Conteos de ejemplos.
        md5s: Md5 de shards.
        info_gpu: Info de GPU/VRAM inicial.
        historial: Historial de entrenamiento.
        smoke: True si es corrida de humo.
        salidas: Rutas resueltas.

    Returns:
        Dict del resumen completo.
    """
    mejor = None
    if historial["val_por_epoca"]:
        mejor = max(
            historial["val_por_epoca"], key=lambda m: m["accuracy_seleccion"]
        )
    return {
        "version": cfg["version"],
        "ciclo": cfg["ciclo"],
        "sprint": cfg["sprint"],
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "smoke": smoke,
        "modelo_base": cfg["modelo"],
        "hiperparametros": {
            "lr": cfg["lr"],
            "epochs": cfg["epochs"],
            "batch": cfg["batch"],
            "val_frac": cfg["val_frac"],
            "max_tripletas": cfg["max_tripletas"],
            "warmup_ratio": cfg["warmup_ratio"],
            "weight_decay": cfg["weight_decay"],
            "max_grad_norm": cfg["max_grad_norm"],
            "amp": cfg["amp"],
            "gradient_checkpointing": cfg["gradient_checkpointing"],
            "semilla": cfg["semilla"],
            "log_cada": cfg["log_cada"],
            "max_seq_length": cfg["max_seq_length"],
            "max_filas": cfg.get("max_filas", 0),
            "max_pasos": cfg.get("max_pasos", 0),
        },
        "datos": {
            "dir_pares": cfg["dir_pares"],
            "esquema_shards": esquema,
            "filas_leidas": indices["filas"],
            "positivos": stats["positivos"],
            "negativos": stats["negativos"],
            "docs": len(indices["docs"]),
            "docs_train": stats["docs_train"],
            "docs_val": stats["docs_val"],
            "tripletas_train_antes_cap": stats["tripletas_train_antes_cap"],
            "tripletas_train": stats["tripletas_train"],
            "tripletas_val": stats["tripletas_val"],
            "pares_train": stats["pares_train"],
            "pares_val": stats["pares_val"],
            "cap_tripletas_aplicado": stats["cap_tripletas_aplicado"],
            "enlace_hard_negatives": {
                "claves_negativas": stats["claves_negativas"],
                "claves_negativas_enlazadas": stats["claves_negativas_enlazadas"],
                "tasa_claves_enlazadas": round(stats["tasa_claves_enlazadas"], 4),
                "filas_negativas": stats["filas_negativas"],
                "filas_negativas_enlazadas": stats["filas_negativas_enlazadas"],
                "tasa_filas_enlazadas": round(stats["tasa_filas_enlazadas"], 4),
                "nota": (
                    "enlace exacto por (doc_id, huella(texto_chunk)); el resto "
                    "de positivos entrena como pares (MNRL con negativos "
                    "in-batch)"
                ),
            },
            "negativos_por_fuente_total": stats["negativos_por_fuente_total"],
            "fuentes_negativo_en_tripletas": stats["fuentes_negativo_tripletas"],
            "md5_shards": md5s,
        },
        "exclusion_benchmark": {
            "n_ids": benchmark["n_ids"],
            "fuente_primaria": benchmark["fuente_primaria"],
            "fuente_cruzada": benchmark["fuente_cruzada"],
            "md5_v2": benchmark["md5_v2"],
            "md5_v8": benchmark["md5_v8"],
            "huella_exclusion": benchmark["huella_exclusion"],
            "filas_benchmark_en_shards": indices["filas_benchmark"],
            "docs_benchmark_en_shards": indices["docs_benchmark"],
            "verificado": indices["filas_benchmark"] == 0,
        },
        "entrenamiento": {
            "pasos_por_epoca": historial["pasos_por_epoca"],
            "pasos_totales": historial["pasos_totales"],
            "pasos_ejecutados": historial["pasos_ejecutados"],
            "ejemplos_entrenados": historial["ejemplos_entrenados"],
            "reanudado": historial["reanudado"],
            "curva_loss": historial["curva_loss"],
            "val_por_epoca": historial["val_por_epoca"],
            "mejor_epoca": historial["mejor_epoca"],
            "mejor_metrica": historial["mejor_metrica"],
            "tiempos": {
                "total_s": historial["tiempo_total_s"],
                "entrenamiento_s": round(historial["tiempo_entrenamiento_s"], 1),
                "evaluacion_s": round(historial["tiempo_evaluacion_s"], 1),
                "por_epoca_s": historial["por_epoca_s"],
                "ejemplos_por_s": round(
                    historial["ejemplos_entrenados"]
                    / max(historial["tiempo_entrenamiento_s"], 1e-9),
                    2,
                ),
            },
            "vram_pico_gb": historial["vram_pico_gb"],
            "vram_inicial": info_gpu,
        },
        "mejor": mejor,
        "entorno": versiones_entorno(),
        "artefactos": {
            "modelo": None if smoke else str(salidas["dir_modelo"]),
            "checkpoints": None if smoke else str(salidas["dir_checkpoints"]),
            "json": str(salidas["ruta_json"]),
            "log": str(salidas["ruta_log"]),
            "figura": str(salidas["ruta_figura"]),
        },
        "notas": [
            (
                "Anti-leakage: los pares de B1 ya excluyen las 90 sentencias del "
                "benchmark; se reverifico por re-escaneo de shards."
            ),
            (
                "Enlace de hard negatives parcial por construccion de B1; las "
                "claves sin enlace entrenan como pares."
            ),
            (
                "Recall@k de val es un proxy con candidatos = pasajes unicos de "
                "val (no el indice completo de la CP)."
            ),
            (
                "Smoke: no se guardan modelo ni checkpoints (tmpfs limitado); "
                "el run real los guarda en modelos/b2_retriever/."
                if smoke
                else "Gradient checkpointing en el encoder (ver hiperparametros)."
            ),
        ],
    }


def actualizar_metricas(
    resumen: dict, logger: logging.Logger, clave_metricas: str | None = None
) -> None:
    """Agrega una entrada compacta a resultados/metricas.json.

    Preserva todo el contenido previo; solo agrega/actualiza la clave del
    experimento.

    Args:
        resumen: Resumen del experimento.
        logger: Logger activo.
        clave_metricas: Clave destino (default: la del run original; las
            variantes pasan la suya para no sobrescribirla).
    """
    clave = clave_metricas or CONFIG["clave_metricas"]
    ruta = RAIZ / CONFIG["ruta_metricas"]
    contenido = {}
    if ruta.is_file():
        with open(ruta, encoding="utf-8") as manejador:
            contenido = json.load(manejador)
    entrenamiento = resumen["entrenamiento"]
    datos = resumen["datos"]
    entrada = {
        "experimento": "b2_retriever_denso",
        "nombre": (
            "Sprint B2: retriever denso de dominio (ALIA-MrBERT legal "
            "fine-tuneado con pares/tripletas de B1)"
        ),
        "fecha": resumen["fecha"][:10],
        "modelo_base": resumen["modelo_base"],
        "script": "scripts/25_b2_retriever_denso.py",
        "datos": {
            "tripletas_train": datos["tripletas_train"],
            "pares_train": datos["pares_train"],
            "tripletas_val": datos["tripletas_val"],
            "pares_val": datos["pares_val"],
            "tasa_filas_negativas_enlazadas": datos["enlace_hard_negatives"][
                "tasa_filas_enlazadas"
            ],
            "exclusion_benchmark": resumen["exclusion_benchmark"][
                "verificado"
            ],
        },
        "entrenamiento": {
            "epochs": resumen["hiperparametros"]["epochs"],
            "batch": resumen["hiperparametros"]["batch"],
            "lr": resumen["hiperparametros"]["lr"],
            "pasos_ejecutados": entrenamiento["pasos_ejecutados"],
            "vram_pico_gb": entrenamiento["vram_pico_gb"],
            "tiempos": entrenamiento["tiempos"],
        },
        "val": {
            "mejor_epoca": entrenamiento["mejor_epoca"],
            "mejor_metrica": entrenamiento["mejor_metrica"],
            "por_epoca": entrenamiento["val_por_epoca"],
        },
        "artefactos": resumen["artefactos"],
        "nota": (
            "Entrada escrita por el propio script; el gate de adopcion de B2 "
            "se decide con la evaluacion end-to-end (recall 1a etapa) fuera "
            "de este entrenamiento."
        ),
    }
    contenido[clave] = entrada
    escribir_json(ruta, contenido)
    logger.info("metricas.json actualizado (clave %s)", clave)


def imprimir_resumen(resumen: dict) -> None:
    """Imprime el resumen final en consola (10 lineas maximo).

    Args:
        resumen: Resumen del experimento.
    """
    datos = resumen["datos"]
    ent = resumen["entrenamiento"]
    mejor = resumen["mejor"]
    lineas = [
        "B2 retriever denso | resumen",
        f"  modelo base: {resumen['modelo_base']}",
        (
            f"  datos: {datos['tripletas_train']} tripletas + "
            f"{datos['pares_train']} pares (train) | val "
            f"{datos['tripletas_val']} + {datos['pares_val']} | enlace filas "
            f"negativas "
            f"{datos['enlace_hard_negatives']['tasa_filas_enlazadas']:.1%}"
        ),
        (
            "  anti-leakage: 90 ids excluidos, filas del benchmark en shards = "
            f"{resumen['exclusion_benchmark']['filas_benchmark_en_shards']}"
        ),
        (
            f"  entrenamiento: {ent['pasos_ejecutados']}/{ent['pasos_totales']} "
            f"pasos, {ent['tiempos']['entrenamiento_s']:.0f} s, "
            f"{ent['tiempos']['ejemplos_por_s']:.1f} ejemplos/s, "
            f"VRAM pico {ent['vram_pico_gb']:.2f} GB"
        ),
    ]
    if mejor is not None:
        acc = mejor["accuracy_tripletas"]
        lineas.append(
            f"  val mejor (epoca {mejor['epoca']}): "
            f"accuracy tripletas {'n/d' if acc is None else f'{acc:.4f}'} | "
            f"recall@1 {mejor['recall@1']:.4f} | recall@5 {mejor['recall@5']:.4f} | "
            f"recall@10 {mejor['recall@10']:.4f} | MRR {mejor['mrr']:.4f}"
        )
    lineas.append(
        f"  modelo: {resumen['artefactos']['modelo'] or 'no guardado (smoke)'}"
    )
    lineas.append(
        f"  salidas: {resumen['artefactos']['json']} | "
        f"{resumen['artefactos']['figura']} | {resumen['artefactos']['log']}"
    )
    for linea in lineas:
        LOGGER.info(linea)


# ===== SECCION 10: orquestacion =====


def parsear_args() -> argparse.Namespace:
    """Define y parsea la CLI.

    Returns:
        Argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Sprint B2: fine-tune contrastivo del retriever denso de dominio "
            "(ALIA-MrBERT legal) con los pares de B1"
        )
    )
    parser.add_argument(
        "--entrenar",
        action="store_true",
        help="modo unico disponible: entrena el retriever denso",
    )
    parser.add_argument("--epochs", type=int, default=None, help="epocas (default 1)")
    parser.add_argument("--batch", type=int, default=None, help="batch (default 64)")
    parser.add_argument("--lr", type=float, default=None, help="learning rate (default 2e-5)")
    parser.add_argument(
        "--max-tripletas",
        type=int,
        default=None,
        help="cap de tripletas de train (default 200000; 0 = todas)",
    )
    parser.add_argument(
        "--val-frac", type=float, default=None, help="fraccion de docs en val (default 0.05)"
    )
    parser.add_argument(
        "--variante",
        default=None,
        help=(
            "sufijo de variante para las salidas (p. ej. ep2 -> modelos/"
            "b2_retriever_ep2, resultados/b2_entrenamiento_ep2.json, clave "
            "b2_retriever_ep2); no toca los artefactos del run original"
        ),
    )
    parser.add_argument(
        "--inicializar-desde",
        default=None,
        help=(
            "ruta local (o hub) para inicializar los pesos en vez del modelo "
            "base (p. ej. modelos/b2_retriever para continuar con otra epoca)"
        ),
    )
    parser.add_argument(
        "--vram-minima-gb",
        type=float,
        default=None,
        help="VRAM libre minima exigida (default 7)",
    )
    parser.add_argument(
        "--sin-gradient-checkpointing",
        action="store_true",
        help=(
            "desactiva gradient checkpointing en el encoder (default: activado; "
            "sin el, batch 64 no cabe en la GPU de 16 GB)"
        ),
    )
    parser.add_argument(
        "--forzar", action="store_true", help="reinicia y borra SOLO el directorio propio"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="corrida de humo (2000 filas, 20 pasos, salidas solo en /tmp/opencode/b2_smoke)",
    )
    return parser.parse_args()


def resolver_config(args: argparse.Namespace, smoke: bool) -> dict:
    """Resuelve la configuracion efectiva (defaults, caps del smoke, CLI).

    Args:
        args: Argumentos de la CLI.
        smoke: True si es corrida de humo.

    Returns:
        Dict de configuracion efectiva.
    """
    cfg = dict(CONFIG["defaults"])
    cfg["max_filas"] = 0
    cfg["max_pasos"] = 0
    cfg["log_cada"] = CONFIG["log_cada"]
    if smoke:
        cfg.update(CONFIG["smoke"])
    for clave in ("epochs", "batch", "lr", "max_tripletas", "val_frac", "vram_minima_gb"):
        valor = getattr(args, clave)
        if valor is not None:
            cfg[clave] = valor
    if args.variante is not None:
        _exigir(
            args.variante[0].isalpha()
            and all(caracter.isalnum() or caracter == "_" for caracter in args.variante),
            "--variante solo admite [a-z0-9_] y debe iniciar con letra",
        )
    cfg["modelo"] = args.inicializar_desde or CONFIG["modelo"]
    cfg["max_seq_length"] = CONFIG["max_seq_length"]
    cfg["semilla"] = CONFIG["semilla"]
    cfg["version"] = CONFIG["version"]
    cfg["ciclo"] = CONFIG["ciclo"]
    cfg["sprint"] = CONFIG["sprint"]
    cfg["warmup_ratio"] = CONFIG["warmup_ratio"]
    cfg["weight_decay"] = CONFIG["weight_decay"]
    cfg["max_grad_norm"] = CONFIG["max_grad_norm"]
    cfg["amp"] = CONFIG["amp"]
    cfg["gradient_checkpointing"] = (
        CONFIG["gradient_checkpointing"] and not args.sin_gradient_checkpointing
    )
    cfg["batch_eval"] = CONFIG["batch_eval"]
    cfg["dir_pares"] = CONFIG["dir_pares"]
    cfg["ruta_benchmark_v2"] = CONFIG["ruta_benchmark_v2"]
    cfg["ruta_benchmark_v8"] = CONFIG["ruta_benchmark_v8"]
    _exigir(cfg["epochs"] >= 1, "--epochs debe ser >= 1")
    _exigir(cfg["batch"] >= 1, "--batch debe ser >= 1")
    _exigir(0.0 < cfg["val_frac"] < 1.0, "--val-frac debe estar en (0, 1)")
    _exigir(cfg["max_tripletas"] >= 0, "--max-tripletas debe ser >= 0")
    return cfg


def main() -> None:
    """Orquesta el experimento completo.

    Raises:
        SystemExit: Si el modo no es --entrenar o si falla la ejecucion.
    """
    args = parsear_args()
    if not args.entrenar:
        print("modo unico disponible: --entrenar (use --help para detalles)")
        sys.exit(2)
    smoke = bool(args.smoke)
    salidas = resolver_salidas(smoke, args.variante)
    if smoke:
        limpiar_salida_smoke(salidas)
    logger = configurar_logger(salidas["ruta_log"])
    t0 = time.time()
    logger.info("=" * 72)
    logger.info(
        "B2 retriever denso | inicio %s | smoke=%s",
        time.strftime("%Y-%m-%d %H:%M:%S"),
        smoke,
    )
    cfg = resolver_config(args, smoke)
    logger.info(
        "config: modelo=%s | lr=%s epochs=%s batch=%s val_frac=%s "
        "max_tripletas=%s vram_min=%s | gc=%s",
        cfg["modelo"],
        cfg["lr"],
        cfg["epochs"],
        cfg["batch"],
        cfg["val_frac"],
        cfg["max_tripletas"],
        cfg["vram_minima_gb"],
        cfg["gradient_checkpointing"],
    )
    configurar_semillas(cfg["semilla"])
    info_gpu = exigir_gpu(cfg["vram_minima_gb"], logger)
    checkpoint = preparar_salidas(salidas, forzar=args.forzar, smoke=smoke)
    shards = descubrir_shards(RAIZ / cfg["dir_pares"], logger)
    esquema = validar_esquema_shards(shards, logger)
    benchmark = cargar_ids_benchmark(cfg, logger)
    filas_por_shard = calcular_filas_por_shard(shards, cfg["max_filas"])
    indices = leer_indices(shards, filas_por_shard, benchmark["ids"], logger)
    docs_train, docs_val = dividir_docs_por_doc(
        indices["docs"], cfg["val_frac"], cfg["semilla"], logger
    )
    md5s = calcular_md5_shards(
        shards, [n > 0 for n in filas_por_shard], logger
    )
    refs, stats = construir_ejemplos(
        shards, filas_por_shard, docs_train, docs_val, cfg, logger
    )
    estado_previo = None
    if checkpoint is not None:
        modelo = SentenceTransformer(str(checkpoint / "modelo"), device="cuda")
        modelo.max_seq_length = int(cfg["max_seq_length"])
        estado_previo = cargar_estado_checkpoint(checkpoint, logger)
        logger.info("modelo recargado desde el checkpoint %s", checkpoint)
    else:
        modelo = cargar_modelo(cfg, logger)
    activar_gradient_checkpointing(modelo, cfg, logger)
    historial = entrenar(
        modelo,
        refs,
        shards,
        docs_val,
        cfg,
        salidas,
        estado_previo,
        guardar=not smoke,
        logger=logger,
    )
    resumen = construir_resumen(
        cfg, esquema, benchmark, indices, stats, md5s, info_gpu, historial, smoke, salidas
    )
    resumen["tiempos_totales_s"] = round(time.time() - t0, 1)
    escribir_json(salidas["ruta_json"], resumen)
    graficar_entrenamiento(historial, salidas["ruta_figura"], cfg)
    if not smoke:
        clave_metricas = f"b2_retriever_{args.variante}" if args.variante else None
        actualizar_metricas(resumen, logger, clave_metricas)
    logger.info("JSON escrito en %s", salidas["ruta_json"])
    logger.info("figura escrita en %s", salidas["ruta_figura"])
    logger.info(
        "tiempo total del script: %.1f s (entrenamiento %.1f s, evaluacion %.1f s)",
        resumen["tiempos_totales_s"],
        historial["tiempo_entrenamiento_s"],
        historial["tiempo_evaluacion_s"],
    )
    imprimir_resumen(resumen)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - se reporta el traceback completo
        traceback.print_exc()
        sys.exit(1)
