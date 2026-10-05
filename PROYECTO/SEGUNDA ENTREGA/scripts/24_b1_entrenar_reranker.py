"""24_b1_entrenar_reranker.py
Fecha: 2026-10-03
Descripcion: Sprint B1 del ciclo v3 - fine-tune del reranker de dominio
(cross-encoder) con los pares construidos por scripts/23_b1_pares_reranker.py
(chunk de sentencia -> pasaje de articulo de la CP citado; positivos +
hard negatives BM25/co-citados/aleatorios). Modo unico --entrenar (con
--smoke). Este script NO construye pares: solo entrena.

    Datos: lee los shards data/pares_b1/pares_reranker_b1_shard*.parquet en
    orden canonico (o los del smoke en /tmp/opencode/b1_pares_smoke), valida
    el esquema exacto (doc_id, articulo, pasaje_id, chunk_idx, label,
    fuente_negativo, texto_chunk, texto_pasaje, n_chars_chunk,
    n_chars_pasaje) y la huella de metadatos comun a todos los shards, y
    divide train/val POR doc_id (semilla 42): ningun par de un mismo doc cae
    en ambos lados. --max-filas limita las filas de train (0 = todas; el val
    se conserva completo para poder evaluar). Los textos se procesan por
    lotes de shards (un shard en memoria a la vez), nunca todos de golpe.

    Modelo: BAAI/bge-reranker-v2-m3 (AutoModelForSequenceClassification,
    num_labels=1) con pares (texto_chunk, texto_pasaje). Truncado del PAR a
    512 tokens con politica "longest_first" (recorta primero la secuencia
    mas larga y reparte el recorte entre chunk y pasaje), identica a la
    inferencia del producto (scripts/lib/reranker_bge.py documenta el
    truncamiento del par a 512 con longest_first). La fraccion de pares que
    supera 512 tokens se mide en una muestra y queda en el JSON.

    Entrenamiento: BCEWithLogitsLoss, AdamW, fp16 AMP (autocast + GradScaler),
    warmup lineal 10 %, grad clipping 1.0, semilla 42. Evaluacion en val cada
    --eval-cada pasos (AUC + accuracy + loss) y al final de cada epoca; se
    guarda el mejor modelo (save_pretrained) y checkpoints resumibles
    rotativos. El mejor se elige por AUC (desempate por loss) y, si el val
    tuviera una sola clase, por -loss.

    GPU/VRAM: exige CUDA/ROCm disponible y VRAM libre >= --vram-minima-gb
    (default 8); si no, aborta (no se entrena en CPU). Log de VRAM pico.

    Salidas (run real): resultados/b1_entrenamiento_v1.json,
    experimentos/24_b1_entrenar.log, figuras/fig_b1_entrenamiento.png,
    modelos/b1_reranker/ (mejor modelo + checkpoints/). El smoke escribe SOLO
    en /tmp/opencode/b1_train_smoke/ y ajusta los caps (200 filas de train,
    2 epocas = 24 pasos, eval cada 10 pasos); cualquier flag explicito tiene
    prioridad sobre los caps del smoke. En smoke NO se guardan checkpoints
    intermedios: un checkpoint completo pesa ~6,3 GB y /tmp es tmpfs de
    16 GB (solo se guarda uno al final); en el run real el checkpoint es
    rotativo y se reescribe en cada evaluacion.

    GPU/VRAM: la placa de 16 GB es justa para el modelo fp32 (2,3 GB) +
    gradientes (2,3 GB) + AdamW (4,6 GB). El smoke mostro OOM por
    fragmentacion en el paso del optimizador; se fija
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (respetando una
    configuracion externa) y el OOM se reporta con la reanudacion sugerida
    (--batch mitad x --acumulacion doble, mismo batch efectivo).

    Reglas: sin subprocesos; nunca se sobrescribe una salida previa sin
    --forzar (con checkpoint y sin --forzar se reanuda; --forzar reinicia
    desde cero borrando SOLO el directorio propio del modelo); reanudacion
    determinista por epoca/shard (mismo orden y misma permutacion con la
    semilla 42); memoria consciente (un shard a la vez); tiempos impresos.
    El run completo NO se ejecuta aqui.

Estado del arte aplicado: fine-tune de cross-encoder con hard negatives
minados por BM25 (Nogueira y Cho, 2019; Karpukhin et al., 2020, DPR) y
negativos de co-citacion (patron dominante en statute retrieval legal:
DoSSIER/COLIEE; LegalMALR 2026), base bge-reranker-v2-m3 (568M, Apache-2.0;
mejor reranker de A3 en el ciclo v3), fp16 AMP con GradScaler, AdamW con
weight decay 0.01 (receta estandar de fine-tune de rerankers BGE), warmup
lineal 10 % y grad clipping 1.0.

Uso:
    /home/carlos/venv/bin/python scripts/24_b1_entrenar_reranker.py --entrenar
    /home/carlos/venv/bin/python scripts/24_b1_entrenar_reranker.py --entrenar --smoke
"""

import argparse
import hashlib
import importlib.util
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
from dataclasses import dataclass, field
from pathlib import Path

# La GPU (16 GB) es justa para el modelo fp32 (2,3 GB) + gradientes (2,3 GB)
# + estados de AdamW (4,6 GB): el smoke con batch 16 mostro OOM por
# fragmentacion (6,2 GB reservados sin reutilizar) en el paso del optimizador.
# expandable_segments permite reutilizar ese pool; setdefault respeta una
# configuracion externa del usuario.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # backend Agg fijado antes de pyplot
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import transformers
from sklearn.metrics import roc_auc_score
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ===== SECCION 1: configuracion =====

RAIZ = Path(__file__).resolve().parent.parent

CONFIG = {
    "semilla": 42,
    "version": "b1-entrenar-reranker-v1",
    "ciclo": "v3",
    "sprint": "B1",
    # Modelo y tokenizacion.
    "modelo": "BAAI/bge-reranker-v2-m3",
    "num_labels": 1,
    "max_length": 512,
    "politica_truncado": "longest_first",
    # Optimizacion (valores fijos del diseno, no expuestos en CLI).
    "grad_clip": 1.0,
    "warmup_frac": 0.10,
    "weight_decay": 0.01,
    "hilos": 32,
    # Esquema exacto de los pares (orden del diseno de 23).
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
    "clave_metadatos": b"b1_pares",
    "patron_shard": "pares_reranker_b1_shard*.parquet",
    # Rutas del run real (relativas a RAIZ) y del smoke (absolutas en /tmp).
    "dir_pares": "data/pares_b1",
    "dir_pares_smoke": "/tmp/opencode/b1_pares_smoke",
    "dir_modelo": "modelos/b1_reranker",
    "nombre_checkpoint": "ultimo.pt",
    "ruta_json": "resultados/b1_entrenamiento_v1.json",
    "ruta_log": "experimentos/24_b1_entrenar.log",
    "ruta_figura": "figuras/fig_b1_entrenamiento.png",
    "dir_smoke": "/tmp/opencode/b1_train_smoke",
    # Muestras de validacion (no cargan todos los textos).
    "muestra_validacion": 5000,
    "muestra_truncado": 2000,
    "ruta_script_17": "scripts/17_auditoria_etiquetas_m4.py",
    # Defaults del run real y del smoke (el CLI explicito tiene prioridad).
    "defaults": {
        "lr": 2e-5,
        "epochs": 1,
        "batch": 16,
        "acumulacion": 1,
        "eval_cada": 2000,
        "val_frac": 0.05,
        "vram_minima_gb": 8.0,
        "max_filas": 0,
    },
    "smoke": {
        "lr": 2e-5,
        "epochs": 2,
        "batch": 16,
        "acumulacion": 1,
        "eval_cada": 10,
        "val_frac": 0.05,
        "vram_minima_gb": 8.0,
        "max_filas": 200,
    },
}

LOGGER_NOMBRE = "b1_entrenar_reranker_24"
LOGGER = logging.getLogger(LOGGER_NOMBRE)
_MODULOS: dict[str, object] = {}


# ===== SECCION 2: utilidades (logger, md5, versiones, importlib, guardas) =====


def _cargar_modulo(nombre: str, ruta: Path):
    """Importa un script por ruta (su nombre empieza por digito).

    Patron de _cargar_modulo de los scripts 19/23 (importlib.util, cacheado).

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


def modulo_17():
    """Carga (y cachea) scripts/17_auditoria_etiquetas_m4.py.

    Returns:
        Modulo 17 con md5_archivo y versiones_entorno.
    """
    return _cargar_modulo("auditoria_etiquetas_m4_17", RAIZ / CONFIG["ruta_script_17"])


def md5_archivo(ruta: str | Path) -> str | None:
    """Calcula el md5 de un archivo por bloques (delegado al modulo 17).

    Args:
        ruta: Ruta del archivo.

    Returns:
        Hexdigest md5, o None si el archivo no existe.
    """
    return modulo_17().md5_archivo(str(ruta))


def versiones_entorno() -> dict:
    """Versiones del entorno de entrenamiento (reutiliza y extiende el 17).

    Returns:
        Dict con python/plataforma/librerias y datos de GPU (torch,
        transformers, HIP, nombre de GPU).
    """
    versiones = dict(modulo_17().versiones_entorno())
    versiones.update(
        {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "hip": getattr(torch.version, "hip", None),
            "cuda_disponible": bool(torch.cuda.is_available()),
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
    )
    return versiones


def configurar_logger(ruta_log: str | Path) -> logging.Logger:
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


def resolver_salidas(smoke: bool) -> dict:
    """Resuelve las rutas de salida del modo real o del smoke.

    Args:
        smoke: True para salidas exclusivamente en /tmp/opencode.

    Returns:
        Dict con dir_salida, dir_modelo, dir_checkpoints, ruta_checkpoint,
        ruta_json, ruta_log y ruta_figura.
    """
    if smoke:
        base = Path(CONFIG["dir_smoke"])
        return {
            "dir_salida": base,
            "dir_modelo": base / "modelo",
            "dir_checkpoints": base / "checkpoints",
            "ruta_checkpoint": base / "checkpoints" / CONFIG["nombre_checkpoint"],
            "ruta_json": base / "b1_entrenamiento_smoke.json",
            "ruta_log": base / "24_b1_entrenar_smoke.log",
            "ruta_figura": base / "fig_b1_entrenamiento.png",
        }
    dir_modelo = RAIZ / CONFIG["dir_modelo"]
    return {
        "dir_salida": dir_modelo,
        "dir_modelo": dir_modelo,
        "dir_checkpoints": dir_modelo / "checkpoints",
        "ruta_checkpoint": dir_modelo / "checkpoints" / CONFIG["nombre_checkpoint"],
        "ruta_json": RAIZ / CONFIG["ruta_json"],
        "ruta_log": RAIZ / CONFIG["ruta_log"],
        "ruta_figura": RAIZ / CONFIG["ruta_figura"],
    }


def limpiar_smoke(salidas: dict) -> None:
    """Limpia el directorio scratch del smoke (siempre reescribible).

    Args:
        salidas: Dict de rutas de salida.
    """
    shutil.rmtree(salidas["dir_salida"], ignore_errors=True)
    salidas["dir_salida"].mkdir(parents=True, exist_ok=True)


def verificar_guardas(
    salidas: dict, forzar: bool, smoke: bool, logger: logging.Logger
) -> bool:
    """Valida salidas previas y decide si la corrida reanuda.

    Reglas: en smoke todo es scratch; en el run real, si existe un checkpoint
    y no se pasa --forzar se reanuda; si el directorio del modelo o el
    JSON/figura existen sin checkpoint ni --forzar se aborta; con --forzar se
    borra SOLO el directorio propio del modelo y se reinicia desde cero.

    Args:
        salidas: Dict de rutas de salida.
        forzar: True permite reiniciar desde cero.
        smoke: True si es corrida smoke.
        logger: Logger activo.

    Returns:
        True si la corrida reanuda desde checkpoint.

    Raises:
        FileExistsError: Si hay salidas previas y no se puede reanudar ni
            forzar.
    """
    if smoke:
        salidas["dir_modelo"].mkdir(parents=True, exist_ok=True)
        salidas["dir_checkpoints"].mkdir(parents=True, exist_ok=True)
        return False
    contenido_previo = (
        sorted(salidas["dir_modelo"].iterdir()) if salidas["dir_modelo"].is_dir() else []
    )
    reanudar = salidas["ruta_checkpoint"].is_file() and not forzar
    if reanudar:
        salidas["dir_checkpoints"].mkdir(parents=True, exist_ok=True)
        logger.info("checkpoint previo detectado: se reanuda desde %s", salidas["ruta_checkpoint"])
        return True
    previos = [
        ruta
        for ruta in (salidas["ruta_json"], salidas["ruta_figura"])
        if ruta.exists()
    ]
    if contenido_previo:
        previos.append(salidas["dir_modelo"])
    if previos and not forzar:
        raise FileExistsError(
            f"salidas previas ({', '.join(str(r) for r in previos)}); "
            "use --forzar para reiniciar desde cero (o conserve el checkpoint para reanudar)"
        )
    if forzar and contenido_previo:
        logger.warning(
            "--forzar: se borra el directorio propio %s y se reinicia desde cero",
            salidas["dir_modelo"],
        )
        shutil.rmtree(salidas["dir_modelo"])
    salidas["dir_modelo"].mkdir(parents=True, exist_ok=True)
    salidas["dir_checkpoints"].mkdir(parents=True, exist_ok=True)
    return False


def resolver_config(args: argparse.Namespace, smoke: bool) -> dict:
    """Resuelve la configuracion efectiva (defaults del modo + CLI explicito).

    Args:
        args: Namespace de la CLI (valores None usan el default del modo).
        smoke: True para aplicar los caps del smoke.

    Returns:
        Dict de configuracion resuelta (json-serializable).

    Raises:
        ValueError: Si algun hiperparametro queda fuera de rango.
    """
    cfg = dict(CONFIG["defaults"])
    if smoke:
        cfg.update(CONFIG["smoke"])
    for clave in (
        "lr",
        "epochs",
        "batch",
        "acumulacion",
        "eval_cada",
        "val_frac",
        "vram_minima_gb",
        "max_filas",
    ):
        valor = getattr(args, clave)
        if valor is not None:
            cfg[clave] = valor
    if cfg["epochs"] < 1:
        raise ValueError("--epochs debe ser >= 1")
    if cfg["batch"] < 1:
        raise ValueError("--batch debe ser >= 1")
    if cfg["acumulacion"] < 1:
        raise ValueError("--acumulacion debe ser >= 1")
    if cfg["eval_cada"] < 1:
        raise ValueError("--eval-cada debe ser >= 1")
    if not 0.0 < cfg["val_frac"] < 1.0:
        raise ValueError("--val-frac debe estar en (0, 1)")
    if cfg["lr"] <= 0:
        raise ValueError("--lr debe ser > 0")
    if cfg["vram_minima_gb"] <= 0:
        raise ValueError("--vram-minima-gb debe ser > 0")
    if cfg["max_filas"] < 0:
        raise ValueError("--max-filas debe ser >= 0 (0 = todas)")
    cfg["semilla"] = CONFIG["semilla"]
    cfg["modelo"] = CONFIG["modelo"]
    cfg["num_labels"] = CONFIG["num_labels"]
    cfg["max_length"] = CONFIG["max_length"]
    cfg["politica_truncado"] = CONFIG["politica_truncado"]
    cfg["grad_clip"] = CONFIG["grad_clip"]
    cfg["warmup_frac"] = CONFIG["warmup_frac"]
    cfg["weight_decay"] = CONFIG["weight_decay"]
    cfg["filas_por_paso"] = cfg["batch"] * cfg["acumulacion"]
    return cfg


def parsear_args() -> argparse.Namespace:
    """Parsea la linea de comandos (modo unico --entrenar).

    Returns:
        Namespace con los argumentos validados.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--entrenar",
        action="store_true",
        help="modo unico: fine-tune del reranker B1 con los pares de 23",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="epocas de entrenamiento; default 1 (smoke 2)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="micro-lote en GPU; default 16",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="learning rate de AdamW; default 2e-5",
    )
    parser.add_argument(
        "--max-filas",
        type=int,
        default=None,
        help="tope de filas de train, 0 = todas (el val queda completo); default 0 (smoke 200)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=None,
        help="fraccion de docs para validacion; default 0.05",
    )
    parser.add_argument(
        "--eval-cada",
        type=int,
        default=None,
        help="evaluacion en val cada N pasos; default 2000 (smoke 10)",
    )
    parser.add_argument(
        "--acumulacion",
        type=int,
        default=None,
        help="pasos de acumulacion de gradiente; default 1",
    )
    parser.add_argument(
        "--vram-minima-gb",
        type=float,
        default=None,
        help="VRAM libre minima exigida; default 8",
    )
    parser.add_argument(
        "--forzar",
        action="store_true",
        help="reinicia desde cero (borra SOLO el directorio propio del modelo); sin esto, con checkpoint se reanuda",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="200 filas de train y salida SOLO en /tmp/opencode/b1_train_smoke/",
    )
    args = parser.parse_args()
    if not args.entrenar:
        parser.error("el modo --entrenar es obligatorio (unico modo disponible)")
    return args


# ===== SECCION 3: descubrimiento de shards, esquema y split por doc =====


@dataclass
class PlanShard:
    """Plan de un shard para train/val (filas usadas y pasos derivados)."""

    ruta: Path
    nombre: str
    n_train_usar: int
    n_val: int
    n_pasos: int


def descubrir_shards(directorio: Path) -> list[Path]:
    """Descubre los shards de pares en orden canonico.

    Args:
        directorio: Directorio con los parquet de pares.

    Returns:
        Lista ordenada de rutas.

    Raises:
        FileNotFoundError: Si el directorio no existe o no hay shards.
    """
    if not directorio.is_dir():
        raise FileNotFoundError(f"no existe el directorio de pares {directorio}")
    shards = sorted(directorio.glob(CONFIG["patron_shard"]))
    if not shards:
        raise FileNotFoundError(f"no hay shards {CONFIG['patron_shard']} en {directorio}")
    return shards


def validar_esquema_shards(shards: list[Path], logger: logging.Logger) -> dict:
    """Valida columnas, huella y conteo de filas de todos los shards.

    Exige el esquema exacto del diseno, que todos los shards compartan la
    misma huella de metadatos (b1_pares) y que el campo filas coincida con el
    numero real de filas (guardia contra escrituras parciales).

    Args:
        shards: Rutas de los shards.
        logger: Logger activo.

    Returns:
        Dict con huella, version y n_shards.

    Raises:
        RuntimeError: Si el esquema, la huella o el conteo no coinciden.
    """
    huellas: set[str] = set()
    versiones: set[str] = set()
    for ruta in shards:
        esquema = pq.read_schema(ruta)
        columnas = list(esquema.names)
        _exigir(
            columnas == CONFIG["columnas"],
            f"esquema inesperado en {ruta.name}: {columnas} != {CONFIG['columnas']}",
        )
        metadatos = esquema.metadata or {}
        crudo = metadatos.get(CONFIG["clave_metadatos"])
        _exigir(crudo is not None, f"{ruta.name} sin metadatos {CONFIG['clave_metadatos']!r}")
        info = json.loads(crudo)
        n_filas = pq.ParquetFile(ruta).metadata.num_rows
        _exigir(
            int(info.get("filas", -1)) == n_filas,
            f"{ruta.name}: filas de metadatos ({info.get('filas')}) != reales ({n_filas})",
        )
        huellas.add(str(info.get("huella")))
        versiones.add(str(info.get("version")))
    _exigir(len(huellas) == 1, f"huellas distintas entre shards: {sorted(huellas)}")
    _exigir(len(versiones) == 1, f"versiones distintas entre shards: {sorted(versiones)}")
    huella = next(iter(huellas))
    logger.info(
        "esquema validado: %d shards, huella %s, version %s",
        len(shards),
        huella,
        next(iter(versiones)),
    )
    return {"huella": huella, "version": next(iter(versiones)), "n_shards": len(shards)}


def leer_indices(shards: list[Path], logger: logging.Logger) -> list[dict]:
    """Lee doc_id/label/fuente/n_chars de cada shard (un shard a la vez).

    Args:
        shards: Rutas de los shards.
        logger: Logger activo.

    Returns:
        Lista de dicts por shard con doc_ids, labels, fuentes y n_filas.

    Raises:
        RuntimeError: Si hay labels fuera de {0,1}, doc_id nulo o textos
            vacios segun n_chars.
    """
    indices = []
    for ruta in shards:
        tabla = pq.read_table(
            ruta,
            columns=["doc_id", "label", "fuente_negativo", "n_chars_chunk", "n_chars_pasaje"],
        )
        doc_ids = np.asarray(tabla["doc_id"].to_pylist(), dtype=object)
        labels = np.asarray(tabla["label"].to_numpy(), dtype=np.int64)
        fuentes = np.asarray(tabla["fuente_negativo"].to_pylist(), dtype=object)
        n_chunk = tabla["n_chars_chunk"].to_numpy()
        n_pasaje = tabla["n_chars_pasaje"].to_numpy()
        _exigir(
            set(np.unique(labels)).issubset({0, 1}),
            f"{ruta.name}: labels fuera de {{0,1}}: {sorted(set(np.unique(labels)))}",
        )
        _exigir(
            all(isinstance(doc, str) and doc for doc in doc_ids),
            f"{ruta.name}: doc_id nulo o vacio",
        )
        _exigir(int(n_chunk.min()) > 0, f"{ruta.name}: n_chars_chunk <= 0")
        _exigir(int(n_pasaje.min()) > 0, f"{ruta.name}: n_chars_pasaje <= 0")
        indices.append(
            {
                "ruta": ruta,
                "nombre": ruta.name,
                "doc_ids": doc_ids,
                "labels": labels,
                "fuentes": fuentes,
                "n_filas": len(labels),
            }
        )
    logger.info(
        "indice leido: %d filas en %d shards",
        sum(item["n_filas"] for item in indices),
        len(indices),
    )
    return indices


def dividir_docs_por_doc(
    indices: list[dict], val_frac: float, semilla: int, logger: logging.Logger
) -> tuple[set[str], set[str]]:
    """Divide los doc_id en train/val con semilla fija (sin fuga por doc).

    Args:
        indices: Indice por shard (doc_ids de todas las filas).
        val_frac: Fraccion de docs para validacion.
        semilla: Semilla del split.
        logger: Logger activo.

    Returns:
        Tupla (docs_train, docs_val) disjuntos.

    Raises:
        RuntimeError: Si los conjuntos se solapan o quedan vacios.
    """
    docs = sorted({str(doc) for item in indices for doc in item["doc_ids"]})
    barajados = list(docs)
    random.Random(f"{semilla}:split:doc").shuffle(barajados)
    n_val = max(1, round(val_frac * len(docs)))
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


def construir_plan(
    indices: list[dict],
    docs_train: set[str],
    docs_val: set[str],
    cfg: dict,
    logger: logging.Logger,
) -> tuple[list[PlanShard], list[PlanShard], dict]:
    """Construye el plan train/val y los conteos por clase y fuente.

    El cap --max-filas se aplica a las filas de train en orden canonico
    (shard, orden de archivo); el val se conserva completo. De cada shard se
    descarta la cola incompleta (menor que batch*acumulacion) para que el
    numero de pasos y la reanudacion sean exactos.

    Args:
        indices: Indice por shard.
        docs_train: Docs de entrenamiento.
        docs_val: Docs de validacion.
        cfg: Configuracion resuelta.
        logger: Logger activo.

    Returns:
        Tupla (plan_train, plan_val, conteos) con conteos por split/fuente.

    Raises:
        RuntimeError: Si no quedan pasos de entrenamiento o el val es vacio.
    """
    train_arr = np.array(sorted(docs_train))
    val_arr = np.array(sorted(docs_val))
    conteos: dict = {
        "filas_disponibles": 0,
        "positivos_disponibles": 0,
        "negativos_disponibles": 0,
        "fuentes_disponibles": Counter(),
        "train_filas": 0,
        "train_filas_efectivas": 0,
        "train_positivos": 0,
        "train_negativos": 0,
        "train_fuentes": Counter(),
        "val_filas": 0,
        "val_positivos": 0,
        "val_negativos": 0,
        "val_fuentes": Counter(),
    }
    restante = cfg["max_filas"] if cfg["max_filas"] > 0 else None
    plan_train: list[PlanShard] = []
    plan_val: list[PlanShard] = []
    for item in indices:
        labels = item["labels"]
        fuentes = item["fuentes"]
        es_train = np.isin(item["doc_ids"], train_arr)
        es_val = np.isin(item["doc_ids"], val_arr)
        # Conteos globales disponibles.
        conteos["filas_disponibles"] += item["n_filas"]
        conteos["positivos_disponibles"] += int((labels == 1).sum())
        conteos["negativos_disponibles"] += int((labels == 0).sum())
        conteos["fuentes_disponibles"].update(fuentes[labels == 0].tolist())
        # Train (con cap en orden canonico).
        idx_train = np.flatnonzero(es_train)
        n_usar = len(idx_train) if restante is None else min(len(idx_train), restante)
        if restante is not None:
            restante -= n_usar
        idx_train = idx_train[:n_usar]
        n_pasos = n_usar // cfg["filas_por_paso"]
        conteos["train_filas"] += n_usar
        conteos["train_filas_efectivas"] += n_pasos * cfg["filas_por_paso"]
        conteos["train_positivos"] += int((labels[idx_train] == 1).sum())
        conteos["train_negativos"] += int((labels[idx_train] == 0).sum())
        conteos["train_fuentes"].update(fuentes[idx_train][labels[idx_train] == 0].tolist())
        if n_pasos > 0:
            plan_train.append(
                PlanShard(item["ruta"], item["nombre"], n_usar, 0, n_pasos)
            )
        # Val (completo, sin cap).
        idx_val = np.flatnonzero(es_val)
        n_val = len(idx_val)
        conteos["val_filas"] += n_val
        conteos["val_positivos"] += int((labels[idx_val] == 1).sum())
        conteos["val_negativos"] += int((labels[idx_val] == 0).sum())
        conteos["val_fuentes"].update(fuentes[idx_val][labels[idx_val] == 0].tolist())
        if n_val > 0:
            plan_val.append(PlanShard(item["ruta"], item["nombre"], 0, n_val, 0))
    pasos_totales = sum(plan.n_pasos for plan in plan_train)
    _exigir(
        pasos_totales > 0,
        "no hay pasos de entrenamiento (revisar --max-filas / tamano de datos)",
    )
    _exigir(conteos["val_filas"] > 0, "val vacio (revisar --val-frac)")
    logger.info(
        "plan: train %d filas seleccionadas (%d efectivas, %d pasos/epoca) en %d shards | "
        "val %d filas en %d shards",
        conteos["train_filas"],
        conteos["train_filas_efectivas"],
        pasos_totales,
        len(plan_train),
        conteos["val_filas"],
        len(plan_val),
    )
    logger.info(
        "clases train: %d pos / %d neg | val: %d pos / %d neg",
        conteos["train_positivos"],
        conteos["train_negativos"],
        conteos["val_positivos"],
        conteos["val_negativos"],
    )
    logger.info(
        "fuentes negativas train: %s | val: %s",
        dict(sorted(conteos["train_fuentes"].items())),
        dict(sorted(conteos["val_fuentes"].items())),
    )
    return plan_train, plan_val, conteos


# ===== SECCION 4: validacion de datos y politica de truncado =====


def leer_muestra(shards: list[Path], n_max: int) -> pd.DataFrame:
    """Lee hasta n_max filas de los primeros shards (sin cargar todo).

    Args:
        shards: Rutas de los shards.
        n_max: Maximo de filas a leer.

    Returns:
        DataFrame con las columnas del par.
    """
    partes: list[pd.DataFrame] = []
    restante = n_max
    for ruta in shards:
        if restante <= 0:
            break
        tabla = pq.read_table(
            ruta,
            columns=[
                "doc_id",
                "label",
                "fuente_negativo",
                "texto_chunk",
                "texto_pasaje",
                "n_chars_chunk",
                "n_chars_pasaje",
            ],
        )
        df = tabla.to_pandas()
        if len(df) > restante:
            df = df.iloc[:restante]
        partes.append(df)
        restante -= len(df)
    return pd.concat(partes, ignore_index=True)


def medir_truncado(df: pd.DataFrame, tokenizer, cfg: dict, logger: logging.Logger) -> dict:
    """Mide la longitud real de los pares (sin truncar) en una muestra.

    Args:
        df: Muestra con texto_chunk y texto_pasaje.
        tokenizer: Tokenizer del reranker.
        cfg: Configuracion resuelta.
        logger: Logger activo.

    Returns:
        Dict con percentiles de longitud de par y fraccion > max_length.
    """
    sub = df.head(CONFIG["muestra_truncado"])
    codificado = tokenizer(
        sub["texto_chunk"].tolist(),
        sub["texto_pasaje"].tolist(),
        truncation=False,
        padding=False,
        return_length=True,
    )
    longitudes = np.asarray(codificado["length"], dtype=np.int64)
    resultado = {
        "politica": cfg["politica_truncado"],
        "max_length": cfg["max_length"],
        "muestra": len(longitudes),
        "p50": float(np.percentile(longitudes, 50)),
        "p90": float(np.percentile(longitudes, 90)),
        "p99": float(np.percentile(longitudes, 99)),
        "max": int(longitudes.max()),
        "fraccion_mayor_512": float((longitudes > cfg["max_length"]).mean()),
    }
    logger.info(
        "truncado (par, sin truncar): p50=%.0f p90=%.0f p99=%.0f max=%d | >512: %.2f%%",
        resultado["p50"],
        resultado["p90"],
        resultado["p99"],
        resultado["max"],
        100.0 * resultado["fraccion_mayor_512"],
    )
    return resultado


def validar_datos(
    shards: list[Path], tokenizer, cfg: dict, logger: logging.Logger
) -> dict:
    """Valida textos/labels en una muestra y mide el truncado real.

    Args:
        shards: Rutas de los shards.
        tokenizer: Tokenizer del reranker.
        cfg: Configuracion resuelta.
        logger: Logger activo.

    Returns:
        Dict con resultado de la validacion y la medicion de truncado.

    Raises:
        RuntimeError: Si hay nulos, textos vacios o longitudes incoherentes.
    """
    muestra = leer_muestra(shards, CONFIG["muestra_validacion"])
    _exigir(
        set(np.unique(muestra["label"].to_numpy())).issubset({0, 1}),
        "labels fuera de {0,1} en la muestra",
    )
    for columna in ("doc_id", "texto_chunk", "texto_pasaje", "fuente_negativo"):
        _exigir(not bool(muestra[columna].isna().any()), f"nulos en {columna} (muestra)")
    _exigir(
        bool((muestra["texto_chunk"].str.len() > 0).all())
        and bool((muestra["texto_pasaje"].str.len() > 0).all()),
        "textos vacios en la muestra",
    )
    _exigir(
        bool((muestra["texto_chunk"].str.len() == muestra["n_chars_chunk"]).all())
        and bool((muestra["texto_pasaje"].str.len() == muestra["n_chars_pasaje"]).all()),
        "n_chars no coincide con la longitud real del texto",
    )
    truncado = medir_truncado(muestra, tokenizer, cfg, logger)
    fuentes = Counter(muestra.loc[muestra["label"] == 0, "fuente_negativo"].tolist())
    resultado = {
        "filas_muestra": len(muestra),
        "positivos_muestra": int((muestra["label"] == 1).sum()),
        "negativos_muestra": int((muestra["label"] == 0).sum()),
        "fuentes_muestra": dict(sorted(fuentes.items())),
        "truncado": truncado,
    }
    logger.info(
        "muestra validada: %d filas (%d pos / %d neg); fuentes: %s",
        resultado["filas_muestra"],
        resultado["positivos_muestra"],
        resultado["negativos_muestra"],
        resultado["fuentes_muestra"],
    )
    return resultado


# ===== SECCION 5: GPU, modelo y tokenizacion =====


def exigir_gpu(
    vram_minima_gb: float, logger: logging.Logger
) -> tuple[str, float, float]:
    """Exige CUDA/ROCm con VRAM libre suficiente (no se entrena en CPU).

    Args:
        vram_minima_gb: VRAM libre minima en GB.
        logger: Logger activo.

    Returns:
        Tupla (dispositivo, VRAM libre GB, VRAM total GB).

    Raises:
        RuntimeError: Si CUDA no esta disponible o la VRAM libre es menor al
            umbral.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA/ROCm no disponible: el fine-tune exige GPU (no se admite CPU). "
            "Verifique el driver ROCm y torch.cuda.is_available()."
        )
    libre, total = torch.cuda.mem_get_info()
    libre_gb = libre / 1024**3
    if libre_gb < vram_minima_gb:
        raise RuntimeError(
            f"VRAM libre {libre_gb:.2f} GB < --vram-minima-gb {vram_minima_gb:.2f} GB; "
            "libere la GPU (otros procesos) o ajuste el umbral. No se entrena en CPU."
        )
    logger.info(
        "GPU: %s | VRAM libre %.2f GB / total %.2f GB (minimo exigido %.2f GB)",
        torch.cuda.get_device_name(0),
        libre_gb,
        total / 1024**3,
        vram_minima_gb,
    )
    return "cuda", libre_gb, total / 1024**3


def cargar_modelo(
    modelo_id: str, num_labels: int, dispositivo: str, logger: logging.Logger
) -> tuple[torch.nn.Module, int]:
    """Carga el modelo base del reranker (fp32 en GPU).

    El modelo se carga en float32 (pesos maestros) y el paso fp16 lo aplica
    autocast + GradScaler.

    Args:
        modelo_id: Id HF del cross-encoder.
        num_labels: Numero de logits de relevancia (1).
        dispositivo: Dispositivo de entrenamiento.
        logger: Logger activo.

    Returns:
        Tupla (modelo, n_parametros).
    """
    modelo = AutoModelForSequenceClassification.from_pretrained(
        modelo_id, num_labels=num_labels
    )
    _exigir(
        int(modelo.config.num_labels) == num_labels,
        f"num_labels del modelo ({modelo.config.num_labels}) != {num_labels}",
    )
    n_parametros = sum(param.numel() for param in modelo.parameters())
    modelo.to(dispositivo)
    logger.info(
        "modelo cargado: %s | %d parametros (%.1f M) | device=%s",
        modelo_id,
        n_parametros,
        n_parametros / 1e6,
        dispositivo,
    )
    return modelo, n_parametros


def tokenizar_pares(
    tokenizer, chunks: list[str], pasajes: list[str], dispositivo: str, max_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokeniza pares (chunk, pasaje) con truncado del par a max_length.

    Args:
        tokenizer: Tokenizer del reranker.
        chunks: Textos del chunk de la sentencia.
        pasajes: Textos del pasaje del articulo.
        dispositivo: Dispositivo destino de los tensores.
        max_length: Longitud maxima del par (tokens).

    Returns:
        Tupla (input_ids, attention_mask) en el dispositivo.
    """
    codificado = tokenizer(
        chunks,
        pasajes,
        padding=True,
        truncation=CONFIG["politica_truncado"],
        max_length=max_length,
        return_tensors="pt",
    )
    return (
        codificado["input_ids"].to(dispositivo, non_blocking=True),
        codificado["attention_mask"].to(dispositivo, non_blocking=True),
    )


def trocear(secuencia, tamano: int):
    """Genera sublistas contiguas de tamano fijo.

    Args:
        secuencia: Secuencia indexable.
        tamano: Tamano de cada trozo.

    Yields:
        Listas de hasta tamano elementos.
    """
    for inicio in range(0, len(secuencia), tamano):
        yield secuencia[inicio : inicio + tamano]


# ===== SECCION 6: estado, evaluacion y checkpoints =====


@dataclass
class EstadoEntrenamiento:
    """Estado mutable del entrenamiento (serializable en el checkpoint)."""

    paso_global: int = 0
    epoca: int = 0
    paso_en_epoca: int = 0
    mejor_puntaje: float = float("-inf")
    mejor_auc: float = float("nan")
    mejor_loss: float = float("inf")
    mejor_paso: int = -1
    ultimo_paso_evaluado: int = -1
    historial: dict = field(default_factory=lambda: {"pasos": [], "loss": [], "lr": []})
    evals: list = field(default_factory=list)
    reanudado: bool = False


@dataclass
class Contexto:
    """Dependencias del bucle de entrenamiento (evita firmas largas)."""

    modelo: torch.nn.Module
    tokenizer: object
    optimizador: torch.optim.Optimizer
    planificador: object
    escalador: torch.amp.GradScaler
    plan_train: list
    plan_val: list
    docs_train: set
    docs_val: set
    cfg: dict
    dispositivo: str
    salidas: dict
    smoke: bool
    loss_fn: torch.nn.Module
    logger: logging.Logger


def cargar_filas_train(plan_shard: PlanShard, docs_train: set[str]) -> pd.DataFrame:
    """Lee un shard y devuelve sus filas de train (cap incluidas).

    Args:
        plan_shard: Plan del shard (n_train_usar en orden de archivo).
        docs_train: Docs de entrenamiento.

    Returns:
        DataFrame con doc_id, label, texto_chunk y texto_pasaje.
    """
    tabla = pq.read_table(
        plan_shard.ruta,
        columns=["doc_id", "label", "texto_chunk", "texto_pasaje"],
    )
    df = tabla.to_pandas()
    df["doc_id"] = df["doc_id"].astype(str)
    sub = df[df["doc_id"].isin(docs_train)]
    return sub.iloc[: plan_shard.n_train_usar].reset_index(drop=True)


def permutar_filas(n: int, semilla: int, epoca: int, nombre_shard: str) -> list[int]:
    """Permutacion determinista de las filas de un shard en una epoca.

    Args:
        n: Numero de filas.
        semilla: Semilla global.
        epoca: Indice de la epoca.
        nombre_shard: Nombre del shard (parte de la semilla derivada).

    Returns:
        Lista con el orden de filas.
    """
    rng = random.Random(f"{semilla}:epoca:{epoca}:shard:{nombre_shard}")
    orden = list(range(n))
    rng.shuffle(orden)
    return orden


def ejecutar_evaluacion(ctx: Contexto, paso: int) -> dict:
    """Evalua el modelo en val (loss, AUC y accuracy) por lotes de shards.

    Args:
        ctx: Contexto del entrenamiento.
        paso: Paso global de la evaluacion.

    Returns:
        Dict con paso, loss_val, auc, acc, n_filas, positivos, negativos y
        segundos.
    """
    inicio = time.perf_counter()
    ctx.modelo.eval()
    probs: list[float] = []
    etiquetas: list[float] = []
    perdida_total, n_total = 0.0, 0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for plan_shard in ctx.plan_val:
            tabla = pq.read_table(
                plan_shard.ruta,
                columns=["doc_id", "label", "texto_chunk", "texto_pasaje"],
            ).to_pandas()
            tabla["doc_id"] = tabla["doc_id"].astype(str)
            sub = tabla[tabla["doc_id"].isin(ctx.docs_val)]
            textos_chunk = sub["texto_chunk"].tolist()
            textos_pasaje = sub["texto_pasaje"].tolist()
            etiquetas_shard = sub["label"].to_numpy(dtype=np.float32)
            for inicio_lote in range(0, len(sub), ctx.cfg["batch"]):
                fin = min(inicio_lote + ctx.cfg["batch"], len(sub))
                indices = list(range(inicio_lote, fin))
                ids, mascara = tokenizar_pares(
                    ctx.tokenizer,
                    [textos_chunk[i] for i in indices],
                    [textos_pasaje[i] for i in indices],
                    ctx.dispositivo,
                    ctx.cfg["max_length"],
                )
                etiquetas_t = torch.from_numpy(etiquetas_shard[indices]).to(
                    ctx.dispositivo, non_blocking=True
                )
                logits = ctx.modelo(input_ids=ids, attention_mask=mascara).logits.view(-1)
                logits = logits.float()
                perdida = ctx.loss_fn(logits, etiquetas_t)
                perdida_total += float(perdida.detach()) * (fin - inicio_lote)
                n_total += fin - inicio_lote
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                etiquetas.extend(etiquetas_t.cpu().numpy().tolist())
    ctx.modelo.train()
    probs_np = np.asarray(probs, dtype=np.float64)
    etiq_np = np.asarray(etiquetas, dtype=np.float64)
    auc = (
        float(roc_auc_score(etiq_np, probs_np))
        if len(np.unique(etiq_np)) == 2
        else float("nan")
    )
    acc = float(((probs_np >= 0.5) == etiq_np).mean()) if n_total else float("nan")
    metricas = {
        "paso": paso,
        "loss_val": perdida_total / n_total if n_total else float("nan"),
        "auc": auc,
        "acc": acc,
        "n_filas": n_total,
        "positivos": int((etiq_np == 1).sum()),
        "negativos": int((etiq_np == 0).sum()),
        "segundos": round(time.perf_counter() - inicio, 2),
    }
    ctx.logger.info(
        "eval paso %d: loss %.4f | AUC %s | acc %s | %d filas (%.1f s)",
        paso,
        metricas["loss_val"],
        f"{auc:.4f}" if math.isfinite(auc) else "n/d",
        f"{acc:.4f}" if math.isfinite(acc) else "n/d",
        n_total,
        metricas["segundos"],
    )
    return metricas


def guardar_mejor_modelo(ctx: Contexto) -> None:
    """Guarda el mejor modelo y el tokenizer (save_pretrained).

    Args:
        ctx: Contexto del entrenamiento.
    """
    inicio = time.perf_counter()
    ctx.modelo.save_pretrained(ctx.salidas["dir_modelo"])
    ctx.tokenizer.save_pretrained(ctx.salidas["dir_modelo"])
    ctx.logger.info(
        "mejor modelo guardado en %s (%.1f s)",
        ctx.salidas["dir_modelo"],
        time.perf_counter() - inicio,
    )


def actualizar_mejor(ctx: Contexto, estado: EstadoEntrenamiento, metricas: dict) -> bool:
    """Actualiza el mejor modelo si la metrica de val mejora.

    Criterio: AUC (desempate por loss); si el val tuviera una sola clase
    (AUC indefinida), se usa -loss.

    Args:
        ctx: Contexto del entrenamiento.
        estado: Estado mutable del entrenamiento.
        metricas: Metricas de la evaluacion.

    Returns:
        True si se actualizo el mejor modelo.
    """
    puntaje = (
        metricas["auc"]
        if math.isfinite(metricas["auc"])
        else -metricas["loss_val"]
    )
    if puntaje <= estado.mejor_puntaje:
        return False
    estado.mejor_puntaje = puntaje
    estado.mejor_auc = metricas["auc"]
    estado.mejor_loss = metricas["loss_val"]
    estado.mejor_paso = metricas["paso"]
    guardar_mejor_modelo(ctx)
    return True


def bytes_estado(objeto) -> int:
    """Estima los bytes de tensores de una estructura anidada.

    Args:
        objeto: Estructura con tensores, dicts, listas o escalares.

    Returns:
        Bytes estimados (los tensores compartidos pueden contarse dos veces).
    """
    if isinstance(objeto, torch.Tensor):
        return objeto.numel() * objeto.element_size()
    if isinstance(objeto, dict):
        return sum(bytes_estado(valor) for valor in objeto.values())
    if isinstance(objeto, (list, tuple)):
        return sum(bytes_estado(valor) for valor in objeto)
    return 0


def _rng_byte_tensor_cpu(valor: object) -> torch.Tensor | None:
    """Normaliza un estado RNG a ByteTensor contiguo de CPU.

    torch.load con map_location=cuda devuelve los estados RNG como tensores
    uint8 en GPU, y torch.set_rng_state / torch.cuda.set_rng_state_all exigen
    ByteTensors de CPU. Tambien tolera ndarray, bytes y secuencias de enteros
    por si el checkpoint trajera una representacion no canonica.

    Args:
        valor: Estado RNG guardado en cualquier representacion razonable.

    Returns:
        ByteTensor contiguo en CPU, o None si el valor no es convertible.
    """
    if valor is None:
        return None
    try:
        if torch.is_tensor(valor):
            return valor.detach().to(device="cpu", dtype=torch.uint8).contiguous().clone()
        if isinstance(valor, np.ndarray):
            return torch.as_tensor(np.ascontiguousarray(valor, dtype=np.uint8)).clone()
        if isinstance(valor, (bytes, bytearray, memoryview)):
            return torch.frombuffer(bytearray(valor), dtype=torch.uint8).clone()
        if isinstance(valor, (list, tuple)):
            return torch.as_tensor(np.asarray(valor, dtype=np.uint8)).clone()
    except (TypeError, ValueError, RuntimeError):
        return None
    return None


def capturar_rng_canonico() -> dict:
    """Captura los estados RNG en su representacion canonica serializable.

    Returns:
        Dict con python (tupla de random), numpy (tupla de np.random), torch
        (ByteTensor CPU) y cuda (lista de ByteTensors CPU; vacia sin GPU).
    """
    estados_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "python": tuple(random.getstate()),
        "numpy": np.random.get_state(),
        "torch": _rng_byte_tensor_cpu(torch.get_rng_state()),
        "cuda": [_rng_byte_tensor_cpu(estado) for estado in estados_cuda],
    }


def _restaurar_rng_python(valor: object, logger: logging.Logger) -> None:
    """Restaura el estado de random de Python si tiene forma valida.

    Args:
        valor: Estado guardado (tupla version/interno/gauss).
        logger: Logger activo.
    """
    if isinstance(valor, (tuple, list)) and len(valor) == 3:
        version, interno, gauss = valor
        try:
            random.setstate(
                (
                    int(version),
                    tuple(int(elemento) for elemento in interno),
                    None if gauss is None else float(gauss),
                )
            )
            return
        except (TypeError, ValueError, OverflowError):
            pass
    logger.warning("estado RNG de python no restaurable (%s): se omite", type(valor).__name__)


def _restaurar_rng_numpy(valor: object, logger: logging.Logger) -> None:
    """Restaura el estado de numpy (Mersenne Twister) si tiene forma valida.

    Args:
        valor: Estado guardado (nombre, claves, posicion, has_gauss, cached).
        logger: Logger activo.
    """
    if isinstance(valor, (tuple, list)) and len(valor) == 5:
        nombre, claves, posicion, has_gauss, cached = valor
        try:
            if torch.is_tensor(claves):
                claves = claves.detach().cpu().numpy()
            nombre = nombre.decode("ascii") if isinstance(nombre, bytes) else str(nombre)
            np.random.set_state(
                (
                    nombre,
                    np.ascontiguousarray(np.asarray(claves), dtype=np.uint32),
                    int(posicion),
                    int(has_gauss),
                    float(cached),
                )
            )
            return
        except (TypeError, ValueError, IndexError):
            pass
    logger.warning("estado RNG de numpy no restaurable (%s): se omite", type(valor).__name__)


def restaurar_rng(rng_guardado: dict | None, logger: logging.Logger) -> None:
    """Restaura los estados RNG del checkpoint de forma defensiva.

    Normaliza cada estado a su tipo canonico antes de aplicarlo (torch.load
    con map_location=cuda devuelve los RNG como tensores uint8 en GPU). Un
    estado irrecuperable se avisa y se omite sin abortar la reanudacion: el
    orden de shards y la permutacion de filas son deterministas por semilla
    y no dependen del RNG global.

    Args:
        rng_guardado: Dict "rng" del checkpoint (None si no lo trae).
        logger: Logger activo.
    """
    if not isinstance(rng_guardado, dict) or not rng_guardado:
        logger.warning("checkpoint sin estado RNG: se continua sin restaurarlo")
        return
    _restaurar_rng_python(rng_guardado.get("python"), logger)
    _restaurar_rng_numpy(rng_guardado.get("numpy"), logger)
    estado_torch = _rng_byte_tensor_cpu(rng_guardado.get("torch"))
    if estado_torch is not None and estado_torch.numel() > 0:
        try:
            torch.set_rng_state(estado_torch)
        except (RuntimeError, TypeError):
            logger.warning("no se pudo aplicar el estado RNG de torch: se omite")
            estado_torch = None
    else:
        logger.warning("estado RNG de torch no restaurable: se omite")
        estado_torch = None
    estados_cuda = rng_guardado.get("cuda") or []
    if torch.is_tensor(estados_cuda):
        estados_cuda = [estados_cuda]
    dispositivos_cuda = 0
    if torch.cuda.is_available() and isinstance(estados_cuda, (list, tuple)) and estados_cuda:
        convertidos = [_rng_byte_tensor_cpu(estado) for estado in estados_cuda]
        if all(estado is not None and estado.numel() > 0 for estado in convertidos):
            try:
                torch.cuda.set_rng_state_all(convertidos)
                dispositivos_cuda = len(convertidos)
            except (RuntimeError, TypeError):
                logger.warning("no se pudo aplicar el estado RNG de cuda: se omite")
        else:
            logger.warning("estado RNG de cuda parcialmente no restaurable: se omite")
    logger.info(
        "estados RNG restaurados (torch %s, cuda %d dispositivos)",
        "si" if estado_torch is not None else "no",
        dispositivos_cuda,
    )


def construir_estado_checkpoint(ctx: Contexto, estado: EstadoEntrenamiento) -> dict:
    """Construye el dict serializable del checkpoint (modelo, optimizador, RNG).

    Args:
        ctx: Contexto del entrenamiento.
        estado: Estado mutable del entrenamiento.

    Returns:
        Dict completo para torch.save.
    """
    cfg = ctx.cfg
    return {
        "version": CONFIG["version"],
        "config_resuelta": {
            clave: cfg[clave]
            for clave in (
                "modelo",
                "max_length",
                "lr",
                "epochs",
                "batch",
                "acumulacion",
                "filas_por_paso",
                "eval_cada",
                "val_frac",
                "max_filas",
                "semilla",
            )
        },
        "paso_global": estado.paso_global,
        "epoca": estado.epoca,
        "paso_en_epoca": estado.paso_en_epoca,
        "mejor_puntaje": estado.mejor_puntaje,
        "mejor_auc": estado.mejor_auc,
        "mejor_loss": estado.mejor_loss,
        "mejor_paso": estado.mejor_paso,
        "ultimo_paso_evaluado": estado.ultimo_paso_evaluado,
        "historial": estado.historial,
        "evals": estado.evals,
        "modelo": ctx.modelo.state_dict(),
        "optimizador": ctx.optimizador.state_dict(),
        "escalador": ctx.escalador.state_dict(),
        "planificador": ctx.planificador.state_dict(),
        "rng": capturar_rng_canonico(),
    }


def guardar_checkpoint(
    ruta: Path, contenido: dict, logger: logging.Logger
) -> dict:
    """Guarda un checkpoint resumible (rotativo, atomico via .tmp + replace).

    Antes de escribir verifica el espacio libre en disco (evita llenar /tmp,
    que es tmpfs de 16 GB, o el disco de /home).

    Args:
        ruta: Ruta destino del checkpoint.
        contenido: Dict del checkpoint.
        logger: Logger activo.

    Returns:
        Dict con bytes_estimados, segundos y ruta.

    Raises:
        RuntimeError: Si el espacio libre es insuficiente.
    """
    inicio = time.perf_counter()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    estimado = bytes_estado(contenido)
    libre = shutil.disk_usage(ruta.parent).free
    _exigir(
        libre > estimado * 1.15,
        f"espacio libre insuficiente para el checkpoint en {ruta.parent} "
        f"({libre / 1024**3:.1f} GB libres vs ~{estimado / 1024**3:.1f} GB estimados)",
    )
    temporal = ruta.with_suffix(".pt.tmp")
    torch.save(contenido, temporal)
    os.replace(temporal, ruta)
    segundos = round(time.perf_counter() - inicio, 2)
    logger.info(
        "checkpoint guardado: %s (~%.1f GB, %.1f s)",
        ruta,
        estimado / 1024**3,
        segundos,
    )
    return {"bytes_estimados": estimado, "segundos": segundos, "ruta": str(ruta)}


def cargar_checkpoint(
    ruta: Path, cfg: dict, dispositivo: str, logger: logging.Logger
) -> dict:
    """Carga un checkpoint y valida que la configuracion sea compatible.

    La reanudacion exige misma configuracion (modelo, lr, epochs, max_filas,
    val_frac, semilla, eval_cada y batch efectivo); se permite cambiar
    batch/acumulacion manteniendo el producto (recuperacion ante OOM).

    Args:
        ruta: Ruta del checkpoint.
        cfg: Configuracion resuelta actual.
        dispositivo: Dispositivo destino.
        logger: Logger activo.

    Returns:
        Dict del checkpoint.

    Raises:
        RuntimeError: Si la configuracion guardada no es compatible.
    """
    estado = torch.load(
        ruta, map_location=dispositivo, weights_only=False, mmap=True
    )
    guardada = estado["config_resuelta"]
    for clave in (
        "modelo",
        "max_length",
        "lr",
        "epochs",
        "eval_cada",
        "val_frac",
        "max_filas",
        "semilla",
    ):
        _exigir(
            guardada[clave] == cfg[clave],
            f"checkpoint incompatible: {clave} guardado {guardada[clave]!r} != actual {cfg[clave]!r} "
            "(use --forzar para reiniciar)",
        )
    _exigir(
        guardada["filas_por_paso"] == cfg["filas_por_paso"],
        f"checkpoint incompatible: batch efectivo guardado {guardada['filas_por_paso']} != "
        f"actual {cfg['filas_por_paso']} (se permite cambiar batch/acumulacion solo con el mismo producto)",
    )
    logger.info(
        "checkpoint cargado: paso %d, epoca %d, mejor AUC %s",
        estado["paso_global"],
        estado["epoca"],
        estado["mejor_auc"],
    )
    return estado


# ===== SECCION 7: bucle de entrenamiento =====


def paso_micro_lote(
    ctx: Contexto,
    textos_chunk: list[str],
    textos_pasaje: list[str],
    etiquetas: np.ndarray,
    indices: list[int],
) -> float:
    """Ejecuta forward/backward de un micro-lote con AMP y acumulacion.

    Args:
        ctx: Contexto del entrenamiento.
        textos_chunk: Textos de chunk del shard.
        textos_pasaje: Textos de pasaje del shard.
        etiquetas: Etiquetas del shard (float32).
        indices: Indices del micro-lote.

    Returns:
        Loss del micro-lote (float).
    """
    ids, mascara = tokenizar_pares(
        ctx.tokenizer,
        [textos_chunk[i] for i in indices],
        [textos_pasaje[i] for i in indices],
        ctx.dispositivo,
        ctx.cfg["max_length"],
    )
    etiquetas_t = torch.from_numpy(etiquetas[indices]).to(
        ctx.dispositivo, non_blocking=True
    )
    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits = ctx.modelo(input_ids=ids, attention_mask=mascara).logits.view(-1)
        perdida = ctx.loss_fn(logits.float(), etiquetas_t)
    ctx.escalador.scale(perdida / ctx.cfg["acumulacion"]).backward()
    return float(perdida.detach())


def optimizar_paso(ctx: Contexto) -> None:
    """Aplica clipping, paso del optimizador, scheduler y reset de gradientes.

    Args:
        ctx: Contexto del entrenamiento.
    """
    ctx.escalador.unscale_(ctx.optimizador)
    torch.nn.utils.clip_grad_norm_(ctx.modelo.parameters(), ctx.cfg["grad_clip"])
    ctx.escalador.step(ctx.optimizador)
    ctx.escalador.update()
    ctx.planificador.step()
    ctx.optimizador.zero_grad(set_to_none=True)


def registrar_evaluacion(
    ctx: Contexto, estado: EstadoEntrenamiento, guardar_ckpt: bool
) -> None:
    """Evalua en val, actualiza el mejor modelo y (opcional) checkpoint.

    Args:
        ctx: Contexto del entrenamiento.
        estado: Estado mutable del entrenamiento.
        guardar_ckpt: True para persistir el checkpoint resumible.
    """
    metricas = ejecutar_evaluacion(ctx, estado.paso_global)
    estado.evals.append(metricas)
    estado.ultimo_paso_evaluado = estado.paso_global
    if actualizar_mejor(ctx, estado, metricas):
        ctx.logger.info(
            "nuevo mejor (AUC %s, loss %.4f) en el paso %d",
            f"{metricas['auc']:.4f}" if math.isfinite(metricas["auc"]) else "n/d",
            metricas["loss_val"],
            estado.paso_global,
        )
    if guardar_ckpt:
        guardar_checkpoint(
            ctx.salidas["ruta_checkpoint"],
            construir_estado_checkpoint(ctx, estado),
            ctx.logger,
        )


def entrenar(ctx: Contexto, estado: EstadoEntrenamiento) -> dict:
    """Bucle principal de fine-tune (epocas, AMP, evaluacion y checkpoints).

    El orden de shards por epoca y la permutacion de filas por shard son
    deterministas con la semilla 42, por lo que la reanudacion a mitad de
    epoca puede saltar exactamente los pasos ya completados.

    Args:
        ctx: Contexto del entrenamiento.
        estado: Estado mutable del entrenamiento (puede venir de un checkpoint).

    Returns:
        Dict con tiempos, pasos, filas procesadas y contadores.

    Raises:
        RuntimeError: Ante OOM de VRAM (mensaje con la reanudacion sugerida).
    """
    cfg = ctx.cfg
    log_cada = max(10, cfg["eval_cada"] // 10)
    pasos_por_epoca = sum(p.n_pasos for p in ctx.plan_train)
    epoca_inicial = estado.epoca
    paso_inicial = estado.paso_global
    inicio_entrenamiento = time.perf_counter()
    filas_procesadas = 0
    segundos_computo = 0.0
    filas_ventana, perdidas_ventana = 0, []
    inicio_ventana = time.perf_counter()
    ctx.modelo.train()
    if estado.paso_global >= pasos_por_epoca * cfg["epochs"]:
        ctx.logger.info("el checkpoint ya cubre todas las epocas: no hay pasos pendientes")
        return {
            "tiempos": {"entrenamiento": 0.0},
            "pasos_finales": estado.paso_global,
            "pasos_ejecutados": 0,
            "filas_procesadas": 0,
            "segundos_computo": 0.0,
        }
    for epoca in range(epoca_inicial, cfg["epochs"]):
        orden_shards = list(ctx.plan_train)
        random.Random(f"{cfg['semilla']}:epoca:{epoca}:shards").shuffle(orden_shards)
        saltar_pasos = estado.paso_en_epoca if epoca == epoca_inicial else 0
        pasos_esperados_epoca = pasos_por_epoca - saltar_pasos
        estado.paso_en_epoca = 0
        inicio_epoca = time.perf_counter()
        for plan_shard in orden_shards:
            if saltar_pasos >= plan_shard.n_pasos:
                saltar_pasos -= plan_shard.n_pasos
                continue
            filas = cargar_filas_train(plan_shard, ctx.docs_train)
            _exigir(
                len(filas) == plan_shard.n_train_usar,
                f"{plan_shard.nombre}: filas de train ({len(filas)}) != plan ({plan_shard.n_train_usar})",
            )
            orden = permutar_filas(len(filas), cfg["semilla"], epoca, plan_shard.nombre)
            inicio_slice = saltar_pasos * cfg["filas_por_paso"]
            saltar_pasos = 0
            orden = orden[inicio_slice:]
            textos_chunk = filas["texto_chunk"].tolist()
            textos_pasaje = filas["texto_pasaje"].tolist()
            etiquetas = filas["label"].to_numpy(dtype=np.float32)
            # Solo grupos completos: la cola menor que filas_por_paso se
            # descarta (coherente con n_pasos = n_usar // filas_por_paso).
            n_grupos = len(orden) // cfg["filas_por_paso"]
            for grupo in trocear(orden[: n_grupos * cfg["filas_por_paso"]], cfg["filas_por_paso"]):
                perdidas_grupo: list[float] = []
                ctx.optimizador.zero_grad(set_to_none=True)
                inicio_grupo = time.perf_counter()
                try:
                    for micro in trocear(grupo, cfg["batch"]):
                        perdidas_grupo.append(
                            paso_micro_lote(
                                ctx, textos_chunk, textos_pasaje, etiquetas, micro
                            )
                        )
                    optimizar_paso(ctx)
                except torch.cuda.OutOfMemoryError as error:
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"OOM de VRAM con batch {cfg['batch']} x acumulacion {cfg['acumulacion']}. "
                        f"Reanude con --batch {max(1, cfg['batch'] // 2)} "
                        f"--acumulacion {cfg['acumulacion'] * 2} (mismo batch efectivo) "
                        "para conservar los pasos ya completados."
                    ) from error
                segundos_computo += time.perf_counter() - inicio_grupo
                estado.paso_global += 1
                estado.paso_en_epoca += 1
                perdida_paso = sum(perdidas_grupo) / len(perdidas_grupo)
                estado.historial["pasos"].append(estado.paso_global)
                estado.historial["loss"].append(perdida_paso)
                estado.historial["lr"].append(ctx.planificador.get_last_lr()[0])
                filas_procesadas += len(grupo)
                filas_ventana += len(grupo)
                perdidas_ventana.append(perdida_paso)
                if estado.paso_global % log_cada == 0:
                    segundos = time.perf_counter() - inicio_ventana
                    ctx.logger.info(
                        "epoca %d | paso %d/%d | loss %.4f | lr %.3e | %.1f filas/s | %.1f s",
                        epoca,
                        estado.paso_global,
                        pasos_por_epoca * cfg["epochs"],
                        sum(perdidas_ventana) / len(perdidas_ventana),
                        estado.historial["lr"][-1],
                        filas_ventana / segundos if segundos > 0 else float("nan"),
                        segundos,
                    )
                    filas_ventana, perdidas_ventana = 0, []
                    inicio_ventana = time.perf_counter()
                if estado.paso_global % cfg["eval_cada"] == 0:
                    # En smoke el checkpoint solo se guarda al final: /tmp es
                    # tmpfs de 16 GB y un checkpoint completo pesa ~6,3 GB.
                    registrar_evaluacion(ctx, estado, guardar_ckpt=not ctx.smoke)
        # Fin de epoca: verificacion de pasos, evaluacion y checkpoint unico.
        _exigir(
            estado.paso_en_epoca == pasos_esperados_epoca,
            f"epoca {epoca}: {estado.paso_en_epoca} pasos ejecutados != "
            f"{pasos_esperados_epoca} esperados (bug de plan/iteracion)",
        )
        if estado.paso_global != estado.ultimo_paso_evaluado:
            registrar_evaluacion(ctx, estado, guardar_ckpt=False)
        estado.epoca = epoca + 1
        estado.paso_en_epoca = 0
        ctx.logger.info(
            "epoca %d completada en %.1f s (paso global %d)",
            epoca,
            time.perf_counter() - inicio_epoca,
            estado.paso_global,
        )
        if not ctx.smoke or epoca == cfg["epochs"] - 1:
            guardar_checkpoint(
                ctx.salidas["ruta_checkpoint"],
                construir_estado_checkpoint(ctx, estado),
                ctx.logger,
            )
    tiempos = {"entrenamiento": round(time.perf_counter() - inicio_entrenamiento, 1)}
    return {
        "tiempos": tiempos,
        "pasos_finales": estado.paso_global,
        "pasos_ejecutados": estado.paso_global - paso_inicial,
        "filas_procesadas": filas_procesadas,
        "segundos_computo": round(segundos_computo, 1),
    }


# ===== SECCION 8: guardado de resultados (JSON, figura, resumen) =====


def md5_modelo_final(dir_modelo: Path) -> dict:
    """Calcula el md5 del modelo final (archivos de nivel superior).

    Excluye subdirectorios (checkpoints, que pesan GB y no forman parte del
    artefacto desplegable).

    Args:
        dir_modelo: Directorio del mejor modelo.

    Returns:
        Dict con md5 combinado y detalle por archivo.
    """
    archivos = sorted(ruta for ruta in dir_modelo.iterdir() if ruta.is_file())
    digest = hashlib.md5()
    detalle = []
    for archivo in archivos:
        md5 = md5_archivo(archivo)
        detalle.append(
            {"archivo": archivo.name, "bytes": archivo.stat().st_size, "md5": md5}
        )
        digest.update(archivo.name.encode("utf-8"))
        digest.update(str(archivo.stat().st_size).encode("utf-8"))
        digest.update((md5 or "").encode("utf-8"))
    return {"md5": digest.hexdigest(), "archivos": detalle}


def graficar_entrenamiento(
    estado: EstadoEntrenamiento, ruta_figura: Path, logger: logging.Logger
) -> None:
    """Grafica la curva de loss (train) y AUC/accuracy (val) por paso.

    Args:
        estado: Estado del entrenamiento con historial y evals.
        ruta_figura: Ruta destino de la figura.
        logger: Logger activo.
    """
    ruta_figura.parent.mkdir(parents=True, exist_ok=True)
    figura, (eje_loss, eje_metricas) = plt.subplots(1, 2, figsize=(12, 4.5))
    pasos = estado.historial["pasos"]
    perdidas = estado.historial["loss"]
    eje_loss.plot(pasos, perdidas, lw=0.6, alpha=0.35, label="loss por paso")
    if len(perdidas) >= 10:
        ventana = max(5, len(perdidas) // 50)
        suave = pd.Series(perdidas).rolling(ventana, min_periods=1).mean()
        eje_loss.plot(pasos, suave, lw=1.6, label=f"media movil ({ventana})")
    eje_loss.set_xlabel("paso (optimizador)")
    eje_loss.set_ylabel("BCE (train)")
    eje_loss.set_title("Loss de entrenamiento")
    eje_loss.legend()
    eje_loss.grid(alpha=0.3)
    if estado.evals:
        pasos_eval = [metrica["paso"] for metrica in estado.evals]
        eje_metricas.plot(
            pasos_eval,
            [metrica["auc"] for metrica in estado.evals],
            marker="o",
            label="AUC val",
        )
        eje_metricas.plot(
            pasos_eval,
            [metrica["acc"] for metrica in estado.evals],
            marker="s",
            label="accuracy val",
        )
        eje_metricas.set_xlabel("paso (optimizador)")
        eje_metricas.set_ylabel("metrica")
        eje_metricas.set_title("Validacion (AUC / accuracy)")
        eje_metricas.legend()
        eje_metricas.grid(alpha=0.3)
    figura.suptitle("B1 reranker de dominio: entrenamiento")
    figura.tight_layout()
    figura.savefig(ruta_figura, dpi=150, bbox_inches="tight")
    plt.close(figura)
    logger.info("figura guardada: %s", ruta_figura)


def _finito_o_none(valor):
    """Devuelve el float si es finito y None en caso contrario (JSON valido).

    Args:
        valor: Valor numerico o None.

    Returns:
        Float finito o None (evita NaN en el JSON).
    """
    if isinstance(valor, (int, float)) and math.isfinite(float(valor)):
        return float(valor)
    return None


def construir_resumen(
    smoke: bool,
    cfg: dict,
    n_parametros: int,
    info_shards: dict,
    md5_shards: list[dict],
    datos: dict,
    estado: EstadoEntrenamiento,
    tiempos: dict,
    rendimiento: dict,
    vram: dict,
    md5_modelo: dict,
    salidas: dict,
    reanudado: bool,
) -> dict:
    """Construye el JSON final del entrenamiento.

    Args:
        smoke: True si es corrida smoke.
        cfg: Configuracion resuelta.
        n_parametros: Parametros del modelo base.
        info_shards: Huella/version/n_shards.
        md5_shards: Detalle md5 por shard.
        datos: Seccion de datos (split, conteos, truncado).
        estado: Estado final del entrenamiento.
        tiempos: Tiempos por fase.
        rendimiento: Pasos/s y filas/s.
        vram: VRAM inicial y pico.
        md5_modelo: Md5 del modelo final.
        salidas: Rutas de salida.
        reanudado: True si la corrida reanudo de checkpoint.

    Returns:
        Dict serializable del resumen.
    """
    mejor = {
        "paso": estado.mejor_paso,
        "auc": _finito_o_none(estado.mejor_auc),
        "loss_val": _finito_o_none(estado.mejor_loss),
    }
    evals = [
        {
            "paso": metrica["paso"],
            "loss_val": _finito_o_none(metrica["loss_val"]),
            "auc": _finito_o_none(metrica["auc"]),
            "acc": _finito_o_none(metrica["acc"]),
            "n_filas": metrica["n_filas"],
            "positivos": metrica["positivos"],
            "negativos": metrica["negativos"],
            "segundos": metrica["segundos"],
        }
        for metrica in estado.evals
    ]
    return {
        "version": CONFIG["version"],
        "ciclo": CONFIG["ciclo"],
        "sprint": CONFIG["sprint"],
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "descripcion": (
            "Fine-tune del reranker de dominio B1 (bge-reranker-v2-m3) con los pares "
            "de 23 (chunk de sentencia -> pasaje CP citado; hard negatives BM25/"
            "co-citados/aleatorios). Split train/val por doc_id; test externo = "
            "benchmark M7 (excluido en los pares)."
        ),
        "modo": "entrenar",
        "smoke": smoke,
        "semilla": cfg["semilla"],
        "reanudado": reanudado,
        "modelo_base": {
            "id": cfg["modelo"],
            "num_labels": cfg["num_labels"],
            "parametros": n_parametros,
        },
        "hiperparametros": {
            "lr": cfg["lr"],
            "epochs": cfg["epochs"],
            "batch": cfg["batch"],
            "acumulacion": cfg["acumulacion"],
            "batch_efectivo": cfg["filas_por_paso"],
            "eval_cada": cfg["eval_cada"],
            "max_filas": cfg["max_filas"],
            "val_frac": cfg["val_frac"],
            "grad_clip": cfg["grad_clip"],
            "warmup_frac": cfg["warmup_frac"],
            "weight_decay": cfg["weight_decay"],
            "max_length": cfg["max_length"],
            "politica_truncado": cfg["politica_truncado"],
            "optimizador": "AdamW",
            "loss": "BCEWithLogitsLoss",
            "amp": "fp16 (autocast + GradScaler)",
        },
        "datos": {
            **datos,
            "huella": info_shards["huella"],
            "version_pares": info_shards["version"],
            "n_shards": info_shards["n_shards"],
            "md5_shards": md5_shards,
        },
        "entrenamiento": {
            "paso_global_final": estado.paso_global,
            "epocas_completadas": estado.epoca,
            "historial": estado.historial,
            "evals": evals,
            "mejor": mejor,
        },
        "tiempos_segundos": tiempos,
        "rendimiento": rendimiento,
        "vram": vram,
        "modelo_final": {
            "dir": str(salidas["dir_modelo"]),
            "md5": md5_modelo["md5"],
            "archivos": md5_modelo["archivos"],
        },
        "salidas": {
            "json": str(salidas["ruta_json"]),
            "log": str(salidas["ruta_log"]),
            "figura": str(salidas["ruta_figura"]),
            "checkpoint": str(salidas["ruta_checkpoint"]),
        },
        "versiones": versiones_entorno(),
    }


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


def imprimir_resumen(resumen: dict) -> None:
    """Imprime el resumen final en consola (<= 10 lineas).

    Args:
        resumen: Resumen final.
    """
    datos = resumen["datos"]
    train = datos["train"]
    val = datos["val"]
    mejor = resumen["entrenamiento"]["mejor"]
    vram = resumen["vram"]
    print("=== B1 entrenamiento reranker: resumen final ===")
    print(
        f"modelo: {resumen['modelo_base']['id']} "
        f"({resumen['modelo_base']['parametros'] / 1e6:.1f} M parametros)"
    )
    print(
        f"datos: {datos['n_shards']} shards | train {train['filas_efectivas']} filas "
        f"({train['positivos']} pos / {train['negativos']} neg) | "
        f"val {val['filas']} filas ({val['positivos']} pos / {val['negativos']} neg)"
    )
    print(
        f"entrenamiento: {resumen['entrenamiento']['paso_global_final']} pasos, "
        f"lr {resumen['hiperparametros']['lr']:.1e}, batch "
        f"{resumen['hiperparametros']['batch']}x{resumen['hiperparametros']['acumulacion']}, "
        f"truncado {resumen['hiperparametros']['politica_truncado']} a "
        f"{resumen['hiperparametros']['max_length']}"
    )
    auc_txt = f"{mejor['auc']:.4f}" if mejor["auc"] is not None else "n/d"
    loss_txt = f"{mejor['loss_val']:.4f}" if mejor["loss_val"] is not None else "n/d"
    print(
        f"mejor val: AUC {auc_txt} / loss {loss_txt} "
        f"(paso {mejor['paso']})"
    )
    print(
        f"tiempo: {resumen['tiempos_segundos']['total']:.1f} s | "
        f"{resumen['rendimiento']['pasos_por_segundo']:.2f} pasos/s | "
        f"{resumen['rendimiento']['filas_por_segundo']:.1f} filas/s"
    )
    print(
        f"VRAM pico: {vram['pico_allocated_gb']:.2f} GB alloc / "
        f"{vram['pico_reserved_gb']:.2f} GB reserved"
    )
    print(f"modelo final: {resumen['modelo_final']['dir']} | md5 {resumen['modelo_final']['md5']}")
    print(f"json: {resumen['salidas']['json']}")
    print(f"figura: {resumen['salidas']['figura']} | log: {resumen['salidas']['log']}")


# ===== SECCION 9: orquestacion =====


def elegir_dir_pares(smoke: bool, logger: logging.Logger) -> Path:
    """Elige el directorio de shards (real o smoke de pares).

    En smoke se prefieren los shards reales si ya existen; si no, los pares
    del smoke en /tmp/opencode/b1_pares_smoke.

    Args:
        smoke: True si es corrida smoke.
        logger: Logger activo.

    Returns:
        Directorio de shards elegido.
    """
    real = RAIZ / CONFIG["dir_pares"]
    if not smoke:
        return real
    if list(real.glob(CONFIG["patron_shard"])):
        logger.info("smoke: usando shards reales de %s", real)
        return real
    logger.info("smoke: sin shards reales; usando pares del smoke %s", CONFIG["dir_pares_smoke"])
    return Path(CONFIG["dir_pares_smoke"])


def main() -> None:
    """Orquesta el modo --entrenar: datos, modelo, bucle y guardado.

    Raises:
        SystemExit: Con codigo 1 ante cualquier error (via __main__).
    """
    inicio_total = time.perf_counter()
    args = parsear_args()
    modo_smoke = bool(args.smoke)
    cfg = resolver_config(args, modo_smoke)
    salidas = resolver_salidas(modo_smoke)
    if modo_smoke:
        limpiar_smoke(salidas)
    logger = configurar_logger(salidas["ruta_log"])
    logger.info(
        "=== INICIO B1 entrenamiento reranker (%s, smoke=%s) ===",
        CONFIG["version"],
        modo_smoke,
    )
    logger.info(
        "hiperparametros: lr=%g, epochs=%d, batch=%d, acumulacion=%d, eval_cada=%d, "
        "val_frac=%g, max_filas=%d, vram_minima_gb=%g",
        cfg["lr"],
        cfg["epochs"],
        cfg["batch"],
        cfg["acumulacion"],
        cfg["eval_cada"],
        cfg["val_frac"],
        cfg["max_filas"],
        cfg["vram_minima_gb"],
    )
    reanudar = verificar_guardas(salidas, args.forzar, modo_smoke, logger)
    torch.set_num_threads(min(CONFIG["hilos"], os.cpu_count() or CONFIG["hilos"]))
    configurar_semillas(cfg["semilla"])
    dispositivo, libre_inicio, total_vram = exigir_gpu(cfg["vram_minima_gb"], logger)

    # --- datos ---
    inicio = time.perf_counter()
    dir_pares = elegir_dir_pares(modo_smoke, logger)
    shards = descubrir_shards(dir_pares)
    info_shards = validar_esquema_shards(shards, logger)
    indices = leer_indices(shards, logger)
    docs_train, docs_val = dividir_docs_por_doc(
        indices, cfg["val_frac"], cfg["semilla"], logger
    )
    plan_train, plan_val, conteos = construir_plan(
        indices, docs_train, docs_val, cfg, logger
    )
    md5_shards = [
        {
            "archivo": shard.name,
            "bytes": shard.stat().st_size,
            "md5": md5_archivo(shard),
        }
        for shard in shards
    ]
    tiempos = {"datos": round(time.perf_counter() - inicio, 1)}

    # --- tokenizer y validacion de muestra ---
    inicio = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(cfg["modelo"])
    validacion = validar_datos(shards, tokenizer, cfg, logger)
    tiempos["tokenizer_validacion"] = round(time.perf_counter() - inicio, 1)

    # --- modelo y optimizacion ---
    inicio = time.perf_counter()
    modelo, n_parametros = cargar_modelo(
        cfg["modelo"], cfg["num_labels"], dispositivo, logger
    )
    pasos_por_epoca = sum(plan.n_pasos for plan in plan_train)
    pasos_totales = pasos_por_epoca * cfg["epochs"]
    optimizador = torch.optim.AdamW(
        modelo.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    planificador = get_linear_schedule_with_warmup(
        optimizador,
        num_warmup_steps=max(1, int(cfg["warmup_frac"] * pasos_totales)),
        num_training_steps=max(1, pasos_totales),
    )
    escalador = torch.amp.GradScaler("cuda")
    ctx = Contexto(
        modelo=modelo,
        tokenizer=tokenizer,
        optimizador=optimizador,
        planificador=planificador,
        escalador=escalador,
        plan_train=plan_train,
        plan_val=plan_val,
        docs_train=docs_train,
        docs_val=docs_val,
        cfg=cfg,
        dispositivo=dispositivo,
        salidas=salidas,
        smoke=modo_smoke,
        loss_fn=nn.BCEWithLogitsLoss(),
        logger=logger,
    )
    estado = EstadoEntrenamiento()
    if reanudar:
        cargado = cargar_checkpoint(salidas["ruta_checkpoint"], cfg, dispositivo, logger)
        modelo.load_state_dict(cargado["modelo"])
        optimizador.load_state_dict(cargado["optimizador"])
        escalador.load_state_dict(cargado["escalador"])
        planificador.load_state_dict(cargado["planificador"])
        estado.paso_global = int(cargado["paso_global"])
        estado.epoca = int(cargado["epoca"])
        estado.paso_en_epoca = int(cargado["paso_en_epoca"])
        estado.mejor_puntaje = float(cargado["mejor_puntaje"])
        estado.mejor_auc = float(cargado["mejor_auc"])
        estado.mejor_loss = float(cargado["mejor_loss"])
        estado.mejor_paso = int(cargado["mejor_paso"])
        estado.ultimo_paso_evaluado = int(cargado["ultimo_paso_evaluado"])
        estado.historial = cargado["historial"]
        estado.evals = cargado["evals"]
        estado.reanudado = True
        restaurar_rng(cargado.get("rng"), logger)
        logger.info("estado restaurado: reanudacion en el paso %d", estado.paso_global)
    tiempos["modelo"] = round(time.perf_counter() - inicio, 1)
    torch.cuda.reset_peak_memory_stats()
    logger.info(
        "pasos: %d por epoca x %d epocas = %d pasos totales (warmup %d)",
        pasos_por_epoca,
        cfg["epochs"],
        pasos_totales,
        max(1, int(cfg["warmup_frac"] * pasos_totales)),
    )

    # --- entrenamiento ---
    resultado = entrenar(ctx, estado)

    # --- verificacion del checkpoint en smoke (carga real, sin GPU) ---
    if modo_smoke:
        cargado = torch.load(
            salidas["ruta_checkpoint"], map_location="cpu", weights_only=False, mmap=True
        )
        _exigir(
            "modelo" in cargado and "optimizador" in cargado,
            "checkpoint del smoke incompleto (faltan estado del modelo u optimizador)",
        )
        logger.info(
            "checkpoint del smoke verificado (carga OK): paso %s", cargado["paso_global"]
        )

    # --- guardado final ---
    inicio = time.perf_counter()
    graficar_entrenamiento(estado, salidas["ruta_figura"], logger)
    md5_modelo = md5_modelo_final(salidas["dir_modelo"])
    tiempos["entrenamiento"] = resultado["tiempos"]["entrenamiento"]
    tiempos["total"] = round(time.perf_counter() - inicio_total, 1)
    segundos_entrenamiento = resultado["tiempos"]["entrenamiento"]
    segundos_computo = resultado["segundos_computo"]
    rendimiento = {
        "pasos_por_segundo": (
            round(resultado["pasos_ejecutados"] / segundos_computo, 3)
            if segundos_computo > 0
            else 0.0
        ),
        "filas_por_segundo": (
            round(resultado["filas_procesadas"] / segundos_computo, 1)
            if segundos_computo > 0
            else 0.0
        ),
        "segundos_computo": segundos_computo,
        "segundos_entrenamiento_total": segundos_entrenamiento,
        "segundos_eval_y_checkpoint": round(
            segundos_entrenamiento - segundos_computo, 1
        ),
    }
    vram = {
        "libre_inicio_gb": round(libre_inicio, 2),
        "total_gb": round(total_vram, 2),
        "pico_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "pico_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
    }
    datos = {
        "dir_shards": str(dir_pares),
        "filas_disponibles": conteos["filas_disponibles"],
        "positivos_disponibles": conteos["positivos_disponibles"],
        "negativos_disponibles": conteos["negativos_disponibles"],
        "fuentes_disponibles": dict(sorted(conteos["fuentes_disponibles"].items())),
        "docs_disponibles": len(docs_train) + len(docs_val),
        "split": {
            "criterio": "por doc_id (sin fuga entre train y val)",
            "semilla": cfg["semilla"],
            "val_frac": cfg["val_frac"],
            "docs_train": len(docs_train),
            "docs_val": len(docs_val),
        },
        "train": {
            "shards": len(plan_train),
            "filas_seleccionadas": conteos["train_filas"],
            "filas_efectivas": conteos["train_filas_efectivas"],
            "positivos": conteos["train_positivos"],
            "negativos": conteos["train_negativos"],
            "fuentes": dict(sorted(conteos["train_fuentes"].items())),
            "pasos_por_epoca": pasos_por_epoca,
        },
        "val": {
            "shards": len(plan_val),
            "filas": conteos["val_filas"],
            "positivos": conteos["val_positivos"],
            "negativos": conteos["val_negativos"],
            "fuentes": dict(sorted(conteos["val_fuentes"].items())),
        },
        "validacion_muestra": validacion,
    }
    resumen = construir_resumen(
        smoke=modo_smoke,
        cfg=cfg,
        n_parametros=n_parametros,
        info_shards=info_shards,
        md5_shards=md5_shards,
        datos=datos,
        estado=estado,
        tiempos=tiempos,
        rendimiento=rendimiento,
        vram=vram,
        md5_modelo=md5_modelo,
        salidas=salidas,
        reanudado=estado.reanudado,
    )
    escribir_json(salidas["ruta_json"], resumen)
    logger.info("json final guardado: %s", salidas["ruta_json"])
    imprimir_resumen(resumen)
    logger.info("=== FIN B1 entrenamiento reranker en %.1f s ===", tiempos["total"])
    logger.info(
        "run real: /home/carlos/venv/bin/python scripts/24_b1_entrenar_reranker.py --entrenar"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - protocolo: traza completa y salida 1 ante cualquier fallo
        traceback.print_exc()
        sys.exit(1)
