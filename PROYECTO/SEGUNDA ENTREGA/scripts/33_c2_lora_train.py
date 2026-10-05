"""33_c2_lora_train.py
Fecha: 2026-10-05
Descripcion: Frente C2 del ciclo v3 - juicio binario punto a punto
    "el fragmento cita el articulo N de la CP?" entrenado con LoRA sobre
    ALIA-7B-Instruct local (modelos/alia_7b_instruct). Un script completo
    con los modos --preparar-datos (tambien aceptado como --dataset) y
    --entrenar, mas --todo y --smoke.

    Modo --preparar-datos (CPU, sin cargar el modelo 7B):
      1. Fuente: 23 shards data/pares_b1/pares_reranker_b1_shard*.parquet
         (299.999 pares B1: 150.000 positivos silver + 149.999 negativos
         minados por BM25/co-cita/aleatorios).
      2. Exclusion dura: los 90 ids del benchmark M7
         (resultados/vinculo_cp_benchmark_v2.json y v8, conjuntos identicos)
         y los packs gold (resultados/gold_mini y resultados/gold_150 si
         existe); se aborta si un id aparece en los shards o en los splits.
      3. Split por documento identico al de B1 (script 24): docs ordenados,
         random.Random("42:split:doc"), val_frac 0.05 -> 21.844/1.150 docs.
      4. Muestreo determinista train: RNG por doc
         random.Random("42:c2lora:doc:{doc_id}"), hasta 3 positivos y 2
         negativos por doc, cap global 20.000 (10.000/10.000) con
         random.Random("42:c2lora:global"). Val: 3.000 (1.500/1.500) y
         val_eval: 1.000 (500/500) SUBCONJUNTO de val.
      5. Textos: chunk <= 600 chars y pasaje <= 800 chars (spec 2.3),
         render con el chat template del tokenizer y truncado de seguridad a
         max_seq (default 768) preservando SIEMPRE los tokens de respuesta.
      6. Dedupe por (doc_id, articulo, chunk_idx, pasaje_id) y por md5 de
         (texto_chunk, texto_pasaje, label); chequeo cruzado train/val.
      7. Salidas: data/c2_lora/sft_v1/{train,val,val_eval}.parquet con
         formato instruction/input/output (+ tokens pre-renderizados),
         resultados/c2_lora/dataset_v1.json y figuras/c2_lora_dataset.png.

    Modo --entrenar (GPU exclusiva; aborta si hay menos de 15,5 GiB libres):
      bf16-LoRA (sin QLoRA): r=16, alpha=32, dropout=0,05, all-linear
      (q,k,v,o,gate,up,down), lr 1e-4 cosine con warmup 3 % y min_lr 10 %,
      batch 1 x grad-accum 16, AdamW (0,9/0,999; eps 1e-8; wd 0), grad clip
      1,0, max_seq 768, autocast bf16 sin GradScaler, gradient checkpointing
      (use_reentrant=False) + enable_input_require_grads + use_cache=False,
      logits_to_keep para calcular la loss completion-only (solo tokens de
      respuesta; verificacion del primer token de respuesta en runtime).
      Checkpoints cada 250 pasos en modelos/c2_lora/checkpoints/ultimo/
      (adapter + optimizer + scheduler + step + RNG) y mejor adapter en
      modelos/c2_lora/adapter_v1/. Reanudacion con --reanudar, presupuesto
      con --max-horas, eval periodica en val_eval (loss, acc, F1, AUC por
      P("Si"), matriz de confusion) y early stopping (patience 4). Al cierre:
      resultados/c2_lora/entrenamiento_v1.json y figuras/c2_lora_curvas.png.

    Modo --smoke: mini-dataset en /tmp/opencode/c2_lora_smoke (mismo
    pipeline con caps pequenos), 20 pasos y medicion de throughput y VRAM
    pico; gate obligatorio del plan antes del run largo. No toca artefactos
    reales.

    Detalle critico verificado en runtime: el chat template de ALIA renderiza
    "<|im_start|>assistant" SIN espacio, por lo que "Si"/"No" en contexto
    usan ids distintos a los standalone: "Si"=112502 y "No"=7704 (los
    standalone del plan son 34351/2125, tokens con marcador de inicio de
    palabra). La evaluacion por logits usa los ids EN CONTEXTO derivados del
    propio render (el JSON del dataset los registra); el script aborta si la
    verificacion falla.

    BUG DE SCORING CORREGIDO (2026-10-05, falso stop del run v1): la
    evaluacion interna usaba batch 4 con PADDING DERECHO; los logits se
    recortan con logits_to_keep a las ULTIMAS K posiciones del tensor
    padeado, de modo que solo la fila mas larga de cada lote tenia su
    posicion de respuesta dentro de la ventana y las filas cortas se
    puntuaban en posiciones equivocadas (AUC interno 0,44 con ~97 % "No",
    mientras la ruta validada del script 34 sobre el mismo adaptador daba
    AUC 0,80 / acc 0,72). La ruta de ENTRENAMIENTO no estaba afectada
    (batch 1 sin padding). Fix: padding IZQUIERDO en evaluar (como el script
    34) + guarda que prohibe padding derecho con filas de distinto largo;
    la loss completion-only queda alineada en todas las filas. Los evals
    previos del checkpoint se etiquetan "bug_padding" al reanudar y se
    reinician mejor_auc/sin_mejora (metricas incompatibles). El modo
    --evaluar permite re-evaluar cualquier adaptador guardado con la ruta
    corregida (control base bf16 con --sin-adaptador).

    Estado del arte aplicado:
    - LoRA (Hu et al. 2021, arXiv:2106.09685) con r=16/alpha=32/dropout 0,05
      y all-linear, receta de instruct tuning de rango bajo (Biderman et al.
      2024, arXiv:2405.09673) para tarea estrecha con menor olvido.
    - bf16-LoRA en lugar de QLoRA (Dettmers et al. 2023, arXiv:2305.14314)
      porque el presupuesto de 16 GB alcanza con las recetas de memoria
      (checkpointing, batch 1 x acumulacion, logits_to_keep) y evita la
      dependencia fragil bitsandbytes en ROCm/py3.14 (ADR-20).
    - Completion-only loss (masking del prompt) como en SFT de instruction
      tuning; el score del juez es P("Si") por logits en un forward, sin
      generar (patron de rerank por logits, LegalMALR arXiv:2601.17692).
    - Evaluacion con AUC/accuracy/F1 y matriz de confusion
      (metricas-evaluation) y early stopping sobre validacion (skill
      deep-learning-training).

    Reglas: solo lectura de las fuentes B1; nunca se ajusta nada con val/test;
    el benchmark M7 y el gold B3 son holdout y jamas entran a train/val;
    salidas con md5; nada se sobrescribe sin --forzar (la reanudacion
    reescribe solo sus propios artefactos).

Uso:
    /home/carlos/venv/bin/python scripts/33_c2_lora_train.py --preparar-datos
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid \\
        /home/carlos/venv/bin/python scripts/33_c2_lora_train.py --entrenar \\
        > experimentos/33_c2_lora_train.log 2>&1 &
    /home/carlos/venv/bin/python scripts/33_c2_lora_train.py --entrenar --reanudar
    /home/carlos/venv/bin/python scripts/33_c2_lora_train.py --entrenar --max-ejemplos 4000 --max-horas 3
    /home/carlos/venv/bin/python scripts/33_c2_lora_train.py --smoke
    /home/carlos/venv/bin/python scripts/33_c2_lora_train.py --evaluar --max-items 400
    /home/carlos/venv/bin/python scripts/33_c2_lora_train.py --evaluar --sin-adaptador
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import logging
import math
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

# Configuracion de allocator ANTES de importar torch (receta de memoria del
# plan: reduce la fragmentacion del pool en ROCm).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===== SECCION 1: configuracion =====

RAIZ = Path(__file__).resolve().parent.parent

CONFIG = {
    "version_dataset": "c2-lora-dataset-v1",
    "version_entrenamiento": "c2-lora-train-v1",
    "semilla": 42,
    # --- rutas de datos (relativas a la raiz del proyecto) ---
    "dir_pares": "data/pares_b1",
    "patron_shard": "pares_reranker_b1_shard*.parquet",
    "ruta_benchmark_v2": "resultados/vinculo_cp_benchmark_v2.json",
    "ruta_benchmark_v8": "resultados/vinculo_cp_benchmark_v8.json",
    "ruta_gold_mini": "resultados/gold_mini/muestra_gold.jsonl",
    "ruta_gold_150": "resultados/gold_150/muestra_gold.jsonl",
    # --- dataset SFT ---
    "dir_sft": "data/c2_lora/sft_v1",
    "ruta_dataset_json": "resultados/c2_lora/dataset_v1.json",
    "ruta_figura_dataset": "figuras/c2_lora_dataset.png",
    "val_frac": 0.05,
    "docs_esperados": {"total": 22994, "train": 21844, "val": 1150},
    "cap_positivos_por_doc": 3,
    "cap_negativos_por_doc": 2,
    "train_pos": 10000,
    "train_neg": 10000,
    "val_pos": 1500,
    "val_neg": 1500,
    "val_eval_pos": 500,
    "val_eval_neg": 500,
    "max_chars_chunk": 600,
    "max_chars_pasaje": 800,
    "system": (
        "Eres un asistente jurídico experto en derecho constitucional colombiano. "
        "Respondes únicamente 'Sí' o 'No'."
    ),
    "plantilla_user": (
        "FRAGMENTO DE SENTENCIA:\n{chunk}\n\n"
        "ARTÍCULO {art} DE LA CONSTITUCIÓN:\n{pasaje}\n\n"
        "¿El fragmento cita el artículo {art}? Responde solo \"Sí\" o \"No\"."
    ),
    "ids_standalone_esperados": {"Sí": 34351, "No": 2125},
    # --- modelo / PEFT ---
    "ruta_modelo": "modelos/alia_7b_instruct",
    "max_seq": 768,
    "attn_implementation": "sdpa",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    # --- entrenamiento ---
    "lr": 1e-4,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.0,
    "batch": 1,
    "grad_accum": 16,
    "epocas": 1,
    "epocas_max": 2,
    "warmup_frac": 0.03,
    "min_lr_frac": 0.10,
    "grad_clip": 1.0,
    "eval_cada": 250,
    "checkpoint_cada": 250,
    "early_stop_patience": 4,
    "auc_diagnostico": 0.70,
    "paso_diagnostico": 1000,
    "vram_min_gib": 15.5,
    "eval_batch": 4,
    "max_ejemplos": None,
    "max_horas": None,
    "max_pasos": None,
    # --- rutas de entrenamiento ---
    "dir_modelos": "modelos/c2_lora",
    "dir_checkpoint": "modelos/c2_lora/checkpoints/ultimo",
    "dir_adapter": "modelos/c2_lora/adapter_v1",
    "ruta_train_json": "resultados/c2_lora/entrenamiento_v1.json",
    "ruta_figura_curvas": "figuras/c2_lora_curvas.png",
    "ruta_log": "experimentos/33_c2_lora_train.log",
    # --- smoke: caps y salidas SOLO en /tmp/opencode ---
    "smoke": {
        "dir_salida": "/tmp/opencode/c2_lora_smoke",
        "max_ejemplos": 96,
        "max_pasos": 20,
        "eval_cada": 10,
        "checkpoint_cada": 10,
        "epocas": 5,
        "caps": {
            "train_pos": 48,
            "train_neg": 48,
            "val_pos": 24,
            "val_neg": 24,
            "val_eval_pos": 16,
            "val_eval_neg": 16,
        },
    },
}

# Esquema EXACTO de los parquet SFT (formato instruction/input/output +
# trazabilidad + tokens pre-renderizados). Semantica:
#   - n_tokens_prompt / n_tokens_total: longitudes del render ORIGINAL (antes
#     del truncado de seguridad); truncado=True si se recorto el prompt para
#     respetar max_seq (la respuesta NUNCA se recorta).
#   - n_chars_chunk / n_chars_pasaje: longitud de los textos YA recortados a
#     los caps de caracteres del spec (600/800).
#   - prompt_ids / resp_ids: tokens exactos con los que se entrena (chat
#     template del tokenizer; resp_ids incluye <|im_end|>).
ESQUEMA_SFT = pa.schema(
    [
        ("doc_id", pa.string()),
        ("articulo", pa.string()),
        ("pasaje_id", pa.int32()),
        ("chunk_idx", pa.int32()),
        ("label", pa.int8()),
        ("fuente_negativo", pa.string()),
        ("instruction", pa.string()),
        ("input", pa.string()),
        ("output", pa.string()),
        ("texto_chunk", pa.string()),
        ("texto_pasaje", pa.string()),
        ("n_chars_chunk", pa.int32()),
        ("n_chars_pasaje", pa.int32()),
        ("prompt_ids", pa.list_(pa.int32())),
        ("resp_ids", pa.list_(pa.int32())),
        ("n_tokens_prompt", pa.int32()),
        ("n_tokens_total", pa.int32()),
        ("truncado", pa.bool_()),
        ("md5_par", pa.string()),
    ]
)

CLAVE_METADATOS = b"c2_lora_sft"

# Claves de CONFIG que son rutas relativas a la raiz del proyecto.
CLAVES_RUTA = (
    "dir_pares",
    "ruta_benchmark_v2",
    "ruta_benchmark_v8",
    "ruta_gold_mini",
    "ruta_gold_150",
    "dir_sft",
    "ruta_dataset_json",
    "ruta_figura_dataset",
    "ruta_modelo",
    "dir_modelos",
    "dir_checkpoint",
    "dir_adapter",
    "ruta_train_json",
    "ruta_figura_curvas",
    "ruta_log",
)

# Columnas del parquet B1 de pares (script 23) que consume el dataset.
COLUMNAS_PARES = [
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
]


# ===== SECCION 2: utilidades (logger, md5, versiones, guardas, semillas) =====

LOGGER_NOMBRE = "c2_lora_train_33"
LOGGER = logging.getLogger(LOGGER_NOMBRE)
_MODULOS: dict[str, object] = {}


def _cargar_modulo(nombre: str, ruta: Path):
    """Importa un script por ruta (su nombre empieza por digito).

    Patron de importlib cacheado del script 23 (sin duplicar logica).

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
    return _cargar_modulo("auditoria_etiquetas_m4_17", RAIZ / "scripts/17_auditoria_etiquetas_m4.py")


def modulo_21():
    """Carga (y cachea) scripts/21_pack_validacion_m4.py.

    Returns:
        Modulo 21 con la guarda verificar_salida_nueva.
    """
    return _cargar_modulo("pack_validacion_m4_21", RAIZ / "scripts/21_pack_validacion_m4.py")


def md5_archivo(ruta: str | Path) -> str | None:
    """Calcula el md5 de un archivo por bloques (delegado al modulo 17).

    Args:
        ruta: Ruta del archivo.

    Returns:
        Hexdigest md5, o None si el archivo no existe.
    """
    return modulo_17().md5_archivo(str(ruta))


def md5_bytes(texto: str) -> str:
    """Calcula el md5 de un string en utf-8.

    Args:
        texto: Contenido a hashear.

    Returns:
        Hexdigest md5.
    """
    return hashlib.md5(texto.encode("utf-8")).hexdigest()


def versiones_entorno() -> dict:
    """Reune versiones del entorno (script 17) mas las del stack ML/DL.

    Returns:
        Dict con python, plataforma y versiones de librerias clave.
    """
    versiones = dict(modulo_17().versiones_entorno())
    import accelerate
    import peft
    import transformers

    versiones.update(
        {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
            "cuda_disponible": torch.cuda.is_available(),
        }
    )
    if torch.cuda.is_available():
        versiones["gpu"] = torch.cuda.get_device_name(0)
        versiones["rocm"] = getattr(torch.version, "hip", None)
    return versiones


def verificar_salida_nueva(ruta: str | Path, forzar: bool, smoke: bool, logger: logging.Logger) -> Path:
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


def fijar_semillas(semilla: int, sembrar_cuda: bool) -> None:
    """Fija las semillas de random, numpy y torch.

    Args:
        semilla: Semilla entera.
        sembrar_cuda: True para sembrar tambien CUDA (solo en modo GPU).
    """
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if sembrar_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def resolver_rutas(cfg: dict) -> dict:
    """Convierte las rutas de CONFIG en absolutas contra RAIZ.

    Args:
        cfg: Configuracion efectiva (con overrides ya aplicados).

    Returns:
        Dict clave -> ruta absoluta (str).
    """
    return {
        clave: cfg[clave] if Path(cfg[clave]).is_absolute() else str(RAIZ / cfg[clave])
        for clave in CLAVES_RUTA
    }


# ===== SECCION 3: dataset SFT (modo --preparar-datos) =====


def leer_exclusiones(rutas: dict, logger: logging.Logger) -> dict:
    """Carga los ids excluidos: benchmark M7 (90) y packs gold.

    Fuente primaria: campo muestra de vinculo_cp_benchmark_v2.json; fuente
    cruzada: vinculo_cp_benchmark_v8.json (deben ser identicos, 90 unicos).
    Packs gold: gold_mini (50 casos, subconjunto del benchmark) y gold_150
    si existe (40 benchmark + 110 C3).

    Args:
        rutas: Rutas absolutas.
        logger: Logger activo.

    Returns:
        Dict con benchmark (set), gold (set) y detalle (md5s/conteos).

    Raises:
        RuntimeError: Si los conjuntos del benchmark no coinciden.
    """
    with open(rutas["ruta_benchmark_v2"], encoding="utf-8") as manejador:
        muestra_v2 = json.load(manejador)["muestra"]
    with open(rutas["ruta_benchmark_v8"], encoding="utf-8") as manejador:
        muestra_v8 = json.load(manejador)["muestra"]
    ids_v2 = [str(caso["id"]) for caso in muestra_v2]
    ids_v8 = [str(caso["id"]) for caso in muestra_v8]
    _exigir(len(ids_v2) == 90 and len(set(ids_v2)) == 90, "benchmark v2: se esperaban 90 ids unicos")
    _exigir(set(ids_v2) == set(ids_v8), "los ids del benchmark v2 y v8 no coinciden")

    ids_gold: set[str] = set()
    detalle_gold: dict[str, dict] = {}
    for nombre, clave in (("gold_mini", "ruta_gold_mini"), ("gold_150", "ruta_gold_150")):
        ruta = Path(rutas[clave])
        if not ruta.exists():
            detalle_gold[nombre] = {"existe": False}
            continue
        with open(ruta, encoding="utf-8") as manejador:
            filas = [json.loads(linea) for linea in manejador if linea.strip()]
        ids = {str(fila["id"]) for fila in filas}
        ids_gold |= ids
        detalle_gold[nombre] = {
            "existe": True,
            "n_ids": len(ids),
            "md5": md5_archivo(ruta),
            "subconjunto_benchmark": ids <= set(ids_v2),
        }
    logger.info(
        "exclusiones: benchmark %d ids; gold %d ids (%s)",
        len(set(ids_v2)),
        len(ids_gold),
        ", ".join(f"{n}={d.get('n_ids', 0)}" for n, d in detalle_gold.items()),
    )
    return {
        "benchmark": set(ids_v2),
        "gold": ids_gold,
        "detalle": {
            "benchmark": {
                "n_ids": len(set(ids_v2)),
                "md5_v2": md5_archivo(rutas["ruta_benchmark_v2"]),
                "md5_v8": md5_archivo(rutas["ruta_benchmark_v8"]),
                "conjuntos_iguales": True,
            },
            "gold": detalle_gold,
        },
    }


def descubrir_shards(rutas: dict, logger: logging.Logger) -> list[Path]:
    """Descubre los shards B1 en orden canonico.

    Args:
        rutas: Rutas absolutas.
        logger: Logger activo.

    Returns:
        Lista ordenada de rutas.

    Raises:
        FileNotFoundError: Si no hay shards.
    """
    shards = sorted(Path(rutas["dir_pares"]).glob(CONFIG["patron_shard"]))
    if not shards:
        raise FileNotFoundError(f"no hay shards {CONFIG['patron_shard']} en {rutas['dir_pares']}")
    logger.info("shards descubiertos: %d", len(shards))
    return shards


def validar_shards(shards: list[Path], logger: logging.Logger) -> dict:
    """Valida esquema y metadatos de los shards y calcula sus md5.

    Exige el esquema exacto del script 23 y que el campo filas de los
    metadatos coincida con el numero real de filas (guardia contra
    escrituras parciales). Los md5 (88 MB en total) quedan en el JSON.

    Args:
        shards: Rutas de los shards.
        logger: Logger activo.

    Returns:
        Dict con n_shards, n_filas, bytes y md5 por shard.

    Raises:
        RuntimeError: Si el esquema o el conteo no coinciden.
    """
    detalle = []
    n_filas_total = 0
    for ruta in shards:
        columnas = list(pq.read_schema(ruta).names)
        _exigir(columnas == COLUMNAS_PARES, f"esquema inesperado en {ruta.name}: {columnas}")
        metadatos = pq.read_schema(ruta).metadata or {}
        crudo = metadatos.get(b"b1_pares")
        _exigir(crudo is not None, f"{ruta.name} sin metadatos b1_pares")
        info = json.loads(crudo)
        n_filas = pq.ParquetFile(ruta).metadata.num_rows
        _exigir(
            int(info.get("filas", -1)) == n_filas,
            f"{ruta.name}: filas de metadatos ({info.get('filas')}) != reales ({n_filas})",
        )
        n_filas_total += n_filas
        detalle.append(
            {
                "nombre": ruta.name,
                "filas": n_filas,
                "bytes": ruta.stat().st_size,
                "md5": md5_archivo(ruta),
            }
        )
    logger.info("shards validados: %d shards, %d filas", len(shards), n_filas_total)
    return {"n_shards": len(shards), "n_filas": n_filas_total, "shards": detalle}


def leer_indices(shards: list[Path], logger: logging.Logger) -> list[dict]:
    """Lee las columnas de indice de cada shard (sin textos).

    Args:
        shards: Rutas de los shards.
        logger: Logger activo.

    Returns:
        Lista por shard con arrays de doc_id/articulo/pasaje_id/chunk_idx/
        label/fuente_negativo y n_filas.

    Raises:
        RuntimeError: Si hay labels fuera de {0,1} o doc_id invalido.
    """
    indices = []
    for ruta in shards:
        tabla = pq.read_table(
            ruta,
            columns=["doc_id", "articulo", "pasaje_id", "chunk_idx", "label", "fuente_negativo"],
        )
        labels = tabla["label"].to_numpy()
        doc_ids = np.asarray(tabla["doc_id"].to_pylist(), dtype=object)
        _exigir(set(np.unique(labels)).issubset({0, 1}), f"{ruta.name}: labels fuera de {{0,1}}")
        _exigir(
            all(isinstance(doc, str) and doc for doc in doc_ids),
            f"{ruta.name}: doc_id nulo o vacio",
        )
        indices.append(
            {
                "ruta": ruta,
                "doc_id": doc_ids,
                "articulo": np.asarray(tabla["articulo"].to_pylist(), dtype=object),
                "pasaje_id": tabla["pasaje_id"].to_numpy(),
                "chunk_idx": tabla["chunk_idx"].to_numpy(),
                "label": labels.astype(np.int64),
                "fuente": np.asarray(tabla["fuente_negativo"].to_pylist(), dtype=object),
                "n_filas": len(labels),
            }
        )
    logger.info("indices leidos: %d filas en %d shards", sum(i["n_filas"] for i in indices), len(indices))
    return indices


def dividir_docs_b1(indices: list[dict], cfg: dict, logger: logging.Logger) -> tuple[set[str], set[str], dict]:
    """Replica el split por documento de B1 (script 24, semilla 42).

    docs ordenados + random.Random("42:split:doc").shuffle + val_frac 0.05
    (n_val = max(1, round(val_frac * n_docs))). Aborta si los conteos no
    coinciden con los de B1 o si hay fuga entre train y val.

    Args:
        indices: Indice por shard.
        cfg: Configuracion efectiva.
        logger: Logger activo.

    Returns:
        Tupla (docs_train, docs_val, info) con info del split.

    Raises:
        RuntimeError: Si los conteos no coinciden con B1 o hay fuga.
    """
    docs = sorted({str(doc) for item in indices for doc in item["doc_id"]})
    barajados = list(docs)
    random.Random(f"{cfg['semilla']}:split:doc").shuffle(barajados)
    n_val = max(1, round(cfg["val_frac"] * len(docs)))
    docs_val = set(barajados[:n_val])
    docs_train = set(barajados[n_val:])
    _exigir(not (docs_train & docs_val), "fuga de doc_id entre train y val")
    esperado = cfg["docs_esperados"]
    _exigir(
        len(docs) == esperado["total"]
        and len(docs_train) == esperado["train"]
        and len(docs_val) == esperado["val"],
        (
            f"split distinto al de B1: docs={len(docs)} (esperado {esperado['total']}), "
            f"train={len(docs_train)} (esperado {esperado['train']}), "
            f"val={len(docs_val)} (esperado {esperado['val']})"
        ),
    )
    info = {
        "metodo": "docs ordenados + random.Random('42:split:doc') + val_frac 0.05 (identico a B1, script 24)",
        "n_docs": len(docs),
        "n_docs_train": len(docs_train),
        "n_docs_val": len(docs_val),
        "val_frac": cfg["val_frac"],
        "semilla": cfg["semilla"],
        "disjuntos": True,
    }
    logger.info(
        "split B1 replicado: %d docs train / %d docs val (total %d)",
        len(docs_train),
        len(docs_val),
        len(docs),
    )
    return docs_train, docs_val, info


def mapear_filas_por_doc(indices: list[dict]) -> tuple[list[dict], dict]:
    """Construye el mapa doc -> posiciones de fila por shard.

    Args:
        indices: Indice por shard.

    Returns:
        Tupla (por_shard, shard_de_doc): por_shard[s] = dict doc -> np.array
        de posiciones; shard_de_doc = doc -> indice de shard.

    Raises:
        RuntimeError: Si un doc aparece en mas de un shard.
    """
    por_shard: list[dict] = []
    shard_de_doc: dict[str, int] = {}
    for s, item in enumerate(indices):
        agrupado: dict[str, list[int]] = defaultdict(list)
        for pos, doc in enumerate(item["doc_id"]):
            agrupado[doc].append(pos)
        _exigir(
            all(doc not in shard_de_doc for doc in agrupado),
            "un doc aparece en mas de un shard (sharding inesperado)",
        )
        por_shard.append({doc: np.asarray(pos, dtype=np.int64) for doc, pos in agrupado.items()})
        shard_de_doc.update({doc: s for doc in agrupado})
    return por_shard, shard_de_doc


def _muestrear_doc(
    indices: list[dict],
    por_shard: list[dict],
    shard_de_doc: dict,
    doc: str,
    rng: random.Random,
    cap_pos: int,
    cap_neg: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Muestrea hasta cap_pos positivos y cap_neg negativos de un doc.

    Args:
        indices: Indice por shard.
        por_shard: Mapa doc -> posiciones por shard.
        shard_de_doc: Mapa doc -> shard.
        doc: Id del documento.
        rng: RNG determinista del documento.
        cap_pos: Cap de positivos.
        cap_neg: Cap de negativos.

    Returns:
        Tupla (seleccion_pos, seleccion_neg) como listas de (shard, pos).
    """
    s = shard_de_doc[doc]
    posiciones = por_shard[s][doc]
    labels = indices[s]["label"][posiciones]
    pool_pos = posiciones[labels == 1].tolist()
    pool_neg = posiciones[labels == 0].tolist()
    sel_pos = sorted(rng.sample(pool_pos, min(cap_pos, len(pool_pos)))) if pool_pos else []
    sel_neg = sorted(rng.sample(pool_neg, min(cap_neg, len(pool_neg)))) if pool_neg else []
    return [(s, p) for p in sel_pos], [(s, p) for p in sel_neg]


def muestrear_train(
    indices: list[dict],
    por_shard: list[dict],
    shard_de_doc: dict,
    docs_train: set[str],
    cfg: dict,
    logger: logging.Logger,
) -> tuple[list[tuple[int, int]], dict]:
    """Muestreo determinista del subset de train (cap global 50/50).

    Reglas del spec: RNG por doc random.Random("42:c2lora:doc:{doc_id}"),
    hasta 3 positivos y 2 negativos por doc, y cap global 10.000/10.000 con
    shuffle random.Random("42:c2lora:global"). Se toman candidatos con un
    margen del 5 % por clase para que el dedupe posterior no reduzca el
    objetivo (el recorte exacto lo hace recortar_balance).

    Args:
        indices: Indice por shard.
        por_shard: Mapa doc -> posiciones por shard.
        shard_de_doc: Mapa doc -> shard.
        docs_train: Docs de train.
        cfg: Configuracion efectiva.
        logger: Logger activo.

    Returns:
        Tupla (seleccion, stats) con la seleccion de candidatos (shard, pos)
        en orden determinista y estadisticas del muestreo.
    """
    lista_pos: list[tuple[int, int]] = []
    lista_neg: list[tuple[int, int]] = []
    docs_con_pares = 0
    for doc in sorted(docs_train):
        rng_doc = random.Random(f"{cfg['semilla']}:c2lora:doc:{doc}")
        sel_pos, sel_neg = _muestrear_doc(
            indices,
            por_shard,
            shard_de_doc,
            doc,
            rng_doc,
            cfg["cap_positivos_por_doc"],
            cfg["cap_negativos_por_doc"],
        )
        if sel_pos:
            docs_con_pares += 1
        lista_pos.extend(sel_pos)
        lista_neg.extend(sel_neg)
    rng_global = random.Random(f"{cfg['semilla']}:c2lora:global")
    rng_global.shuffle(lista_pos)
    rng_global.shuffle(lista_neg)
    margen = max(64, math.ceil(0.05 * max(cfg["train_pos"], cfg["train_neg"])))
    n_pos = min(cfg["train_pos"] + margen, len(lista_pos))
    n_neg = min(cfg["train_neg"] + margen, len(lista_neg))
    seleccion = lista_pos[:n_pos] + lista_neg[:n_neg]
    rng_global.shuffle(seleccion)
    stats = {
        "docs_train": len(docs_train),
        "docs_con_pares_muestreados": docs_con_pares,
        "muestreados_positivos": len(lista_pos),
        "muestreados_negativos": len(lista_neg),
        "candidatos_positivos": n_pos,
        "candidatos_negativos": n_neg,
        "objetivo_positivos": cfg["train_pos"],
        "objetivo_negativos": cfg["train_neg"],
    }
    logger.info(
        "muestreo train: %d candidatos pos / %d candidatos neg (de %d/%d muestreados)",
        n_pos,
        n_neg,
        len(lista_pos),
        len(lista_neg),
    )
    return seleccion, stats


def muestrear_val(
    indices: list[dict],
    por_shard: list[dict],
    shard_de_doc: dict,
    docs_val: set[str],
    cfg: dict,
    logger: logging.Logger,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict]:
    """Muestrea val (1.500/1.500) y val_eval (500/500) solo con docs val.

    val_eval es un SUBCONJUNTO de val (evaluacion periodica consistente con
    la final). Semillas: "42:c2lora:val" y "42:c2lora:val_eval".

    Args:
        indices: Indice por shard.
        por_shard: Mapa doc -> posiciones por shard.
        shard_de_doc: Mapa doc -> shard.
        docs_val: Docs de validacion.
        cfg: Configuracion efectiva.
        logger: Logger activo.

    Returns:
        Tupla (seleccion_val, seleccion_val_eval, stats).
    """
    pool_pos: list[tuple[int, int]] = []
    pool_neg: list[tuple[int, int]] = []
    for doc in sorted(docs_val):
        s = shard_de_doc[doc]
        posiciones = por_shard[s][doc]
        labels = indices[s]["label"][posiciones]
        pool_pos.extend((s, int(p)) for p in posiciones[labels == 1])
        pool_neg.extend((s, int(p)) for p in posiciones[labels == 0])
    rng_val = random.Random(f"{cfg['semilla']}:c2lora:val")
    rng_val.shuffle(pool_pos)
    rng_val.shuffle(pool_neg)
    margen = max(32, math.ceil(0.05 * max(cfg["val_pos"], cfg["val_neg"])))
    n_pos = min(cfg["val_pos"] + margen, len(pool_pos))
    n_neg = min(cfg["val_neg"] + margen, len(pool_neg))
    seleccion = pool_pos[:n_pos] + pool_neg[:n_neg]
    rng_val.shuffle(seleccion)
    # val_eval: subconjunto estratificado de val.
    val_pos = [clave for clave in seleccion if indices[clave[0]]["label"][clave[1]] == 1]
    val_neg = [clave for clave in seleccion if indices[clave[0]]["label"][clave[1]] == 0]
    rng_eval = random.Random(f"{cfg['semilla']}:c2lora:val_eval")
    rng_eval.shuffle(val_pos)
    rng_eval.shuffle(val_neg)
    n_eval = min(
        cfg["val_eval_pos"] + max(16, math.ceil(0.05 * cfg["val_eval_pos"])),
        cfg["val_eval_neg"] + max(16, math.ceil(0.05 * cfg["val_eval_neg"])),
        len(val_pos),
        len(val_neg),
    )
    seleccion_eval = val_pos[:n_eval] + val_neg[:n_eval]
    rng_eval.shuffle(seleccion_eval)
    stats = {
        "docs_val": len(docs_val),
        "disponibles_positivos": len(pool_pos),
        "disponibles_negativos": len(pool_neg),
        "candidatos_val": len(seleccion),
        "candidatos_val_eval": len(seleccion_eval),
        "objetivo_val": [cfg["val_pos"], cfg["val_neg"]],
        "objetivo_val_eval": [cfg["val_eval_pos"], cfg["val_eval_neg"]],
        "val_eval_subconjunto_val": set(seleccion_eval) <= set(seleccion),
    }
    logger.info(
        "muestreo val: %d pares (val) y %d pares (val_eval) de %d/%d disponibles",
        len(seleccion),
        len(seleccion_eval),
        len(pool_pos),
        len(pool_neg),
    )
    return seleccion, seleccion_eval, stats


def leer_textos(
    shards: list[Path], seleccion: list[tuple[int, int]], logger: logging.Logger
) -> dict[tuple[int, int], tuple[str, str]]:
    """Lee los textos de los pares seleccionados (un shard a la vez).

    Verifica que n_chars_chunk/n_chars_pasaje del parquet coincidan con el
    largo real de los textos (integridad del shard).

    Args:
        shards: Rutas de los shards.
        seleccion: Lista de (shard, pos) seleccionados.
        logger: Logger activo.

    Returns:
        Dict (shard, pos) -> (texto_chunk, texto_pasaje).

    Raises:
        RuntimeError: Si el largo de un texto no coincide con n_chars.
    """
    por_shard: dict[int, list[int]] = defaultdict(list)
    for s, p in seleccion:
        por_shard[s].append(p)
    textos: dict[tuple[int, int], tuple[str, str]] = {}
    for s, posiciones in por_shard.items():
        posiciones = sorted(posiciones)
        tabla = pq.read_table(
            shards[s],
            columns=["texto_chunk", "texto_pasaje", "n_chars_chunk", "n_chars_pasaje"],
        )
        sub = tabla.take(pa.array(posiciones, type=pa.int64()))
        for p, tc, tp, nc, npas in zip(
            posiciones,
            sub["texto_chunk"].to_pylist(),
            sub["texto_pasaje"].to_pylist(),
            sub["n_chars_chunk"].to_pylist(),
            sub["n_chars_pasaje"].to_pylist(),
            strict=True,
        ):
            _exigir(len(tc) == nc and len(tp) == npas, f"n_chars inconsistente en shard {s} pos {p}")
            textos[(s, p)] = (tc, tp)
    logger.info("textos leidos: %d pares de %d shards", len(textos), len(por_shard))
    return textos


def construir_fila_sft(
    indices: list[dict],
    clave: tuple[int, int],
    textos: dict[tuple[int, int], tuple[str, str]],
    cfg: dict,
) -> dict:
    """Construye una fila SFT (instruction/input/output + trazabilidad).

    Args:
        indices: Indice por shard.
        clave: (shard, pos) del par B1.
        textos: Dict de textos por clave.
        cfg: Configuracion efectiva.

    Returns:
        Dict de la fila (incluye la clave interna "_clave").
    """
    s, pos = clave
    item = indices[s]
    texto_chunk, texto_pasaje = textos[clave]
    chunk = texto_chunk[: cfg["max_chars_chunk"]]
    pasaje = texto_pasaje[: cfg["max_chars_pasaje"]]
    articulo = str(item["articulo"][pos])
    label = int(item["label"][pos])
    salida = "Sí" if label == 1 else "No"
    entrada = cfg["plantilla_user"].format(chunk=chunk, art=articulo, pasaje=pasaje)
    return {
        "doc_id": str(item["doc_id"][pos]),
        "articulo": articulo,
        "pasaje_id": int(item["pasaje_id"][pos]),
        "chunk_idx": int(item["chunk_idx"][pos]),
        "label": label,
        "fuente_negativo": str(item["fuente"][pos]),
        "instruction": cfg["system"],
        "input": entrada,
        "output": salida,
        "texto_chunk": chunk,
        "texto_pasaje": pasaje,
        "n_chars_chunk": len(chunk),
        "n_chars_pasaje": len(pasaje),
        "md5_par": md5_bytes(f"{chunk}\x1f{pasaje}\x1f{label}"),
        "_clave": f"{s}:{pos}",
    }


def _aplicar_plantilla(tokenizer, mensajes: list[dict], add_generation_prompt: bool) -> list[int]:
    """Renderiza el chat template del tokenizer y devuelve ids de tokens.

    Compatibilidad transformers 5.x: apply_chat_template con tokenize=True
    puede devolver BatchEncoding (dict) o lista segun return_dict.

    Args:
        tokenizer: Tokenizer de ALIA.
        mensajes: Lista de mensajes {role, content}.
        add_generation_prompt: True para cerrar en "<|im_start|>assistant".

    Returns:
        Lista de ids de tokens.
    """
    try:
        salida = tokenizer.apply_chat_template(
            mensajes,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=False,
        )
    except TypeError:
        salida = tokenizer.apply_chat_template(
            mensajes, tokenize=True, add_generation_prompt=add_generation_prompt
        )
    if hasattr(salida, "keys"):
        salida = salida["input_ids"]
    return [int(t) for t in salida]


def construir_secuencia(
    tokenizer, instruction: str, input_: str, output: str, max_seq: int
) -> dict:
    """Renderiza el par SFT y verifica la consistencia del chat template.

    El prompt (system+user+generation prompt) debe ser prefijo exacto del
    chat completo; la respuesta son los tokens restantes (contenido +
    <|im_end|>). Si el total supera max_seq se recorta SOLO el prompt (la
    respuesta jamas se toca): red de seguridad del spec.

    Args:
        tokenizer: Tokenizer de ALIA.
        instruction: System prompt.
        input_: Contenido del usuario.
        output: "Si" o "No".
        max_seq: Longitud maxima de la secuencia.

    Returns:
        Dict con prompt_ids, resp_ids, n_tokens_prompt, n_tokens_total y
        truncado.

    Raises:
        RuntimeError: Si el prompt no es prefijo del chat o la respuesta
            queda vacia.
    """
    base = [{"role": "system", "content": instruction}, {"role": "user", "content": input_}]
    prompt_ids = _aplicar_plantilla(tokenizer, base, True)
    full_ids = _aplicar_plantilla(
        tokenizer, base + [{"role": "assistant", "content": output}], False
    )
    _exigir(
        full_ids[: len(prompt_ids)] == prompt_ids,
        "el prompt no es prefijo del chat completo (plantilla del tokenizer inconsistente)",
    )
    resp_ids = full_ids[len(prompt_ids) :]
    _exigir(bool(resp_ids), "la respuesta quedo sin tokens")
    n_prompt, n_total = len(prompt_ids), len(full_ids)
    truncado = n_total > max_seq
    if truncado:
        prompt_ids = prompt_ids[: max_seq - len(resp_ids)]
    return {
        "prompt_ids": prompt_ids,
        "resp_ids": resp_ids,
        "n_tokens_prompt": n_prompt,
        "n_tokens_total": n_total,
        "truncado": truncado,
    }


def ids_respuesta_contexto(tokenizer, cfg: dict, logger: logging.Logger) -> dict:
    """Verifica los ids de "Si"/"No" standalone y EN CONTEXTO.

    El chat template de ALIA renderiza "<|im_start|>assistant" sin espacio;
    el primer token de la respuesta es por tanto la variante sin marcador de
    inicio de palabra (distinta de los ids standalone del plan). Este
    chequeo es critico para la loss y para el score P("Si") por logits.

    Args:
        tokenizer: Tokenizer de ALIA.
        cfg: Configuracion efectiva.
        logger: Logger activo.

    Returns:
        Dict {"standalone": {...}, "contexto": {...}, "esperados": {...}}.
    """
    probe_user = cfg["plantilla_user"].format(chunk="Texto de prueba.", art="86", pasaje="Pasaje de prueba.")
    resultado: dict = {"standalone": {}, "contexto": {}, "esperados": dict(cfg["ids_standalone_esperados"])}
    for salida in ("Sí", "No"):
        secuencia = construir_secuencia(tokenizer, cfg["system"], probe_user, salida, cfg["max_seq"])
        resultado["contexto"][salida] = int(secuencia["resp_ids"][0])
        resultado["standalone"][salida] = int(tokenizer(salida, add_special_tokens=False)["input_ids"][0])
    if resultado["standalone"] != resultado["esperados"]:
        logger.warning(
            "ids standalone de 'Si'/'No' difieren de los esperados por el plan: %s != %s",
            resultado["standalone"],
            resultado["esperados"],
        )
    logger.info(
        "ids de respuesta verificados: contexto Sí=%d No=%d (standalone Sí=%d No=%d)",
        resultado["contexto"]["Sí"],
        resultado["contexto"]["No"],
        resultado["standalone"]["Sí"],
        resultado["standalone"]["No"],
    )
    return resultado


def tokenizar_filas(
    filas: list[dict], tokenizer, cfg: dict, ids_ctx: dict, logger: logging.Logger
) -> None:
    """Pre-renderiza y tokeniza todas las filas de un split (in-place).

    Args:
        filas: Filas SFT (se actualizan con prompt_ids/resp_ids/...).
        tokenizer: Tokenizer de ALIA.
        cfg: Configuracion efectiva.
        ids_ctx: Salida de ids_respuesta_contexto.
        logger: Logger activo.

    Raises:
        RuntimeError: Si el primer token de respuesta no coincide con el id
            en contexto esperado para la etiqueta.
    """
    for fila in filas:
        secuencia = construir_secuencia(
            tokenizer, fila["instruction"], fila["input"], fila["output"], cfg["max_seq"]
        )
        esperado = ids_ctx["contexto"][fila["output"]]
        _exigir(
            secuencia["resp_ids"][0] == esperado,
            f"primer token de respuesta {secuencia['resp_ids'][0]} != esperado {esperado} "
            f"para output={fila['output']!r} (doc {fila['doc_id']})",
        )
        fila.update(secuencia)
    logger.info("filas tokenizadas: %d", len(filas))


def deduplicar(filas: list[dict], logger: logging.Logger, nombre: str) -> tuple[list[dict], dict]:
    """Deduplica por clave dura y por md5 de contenido.

    Args:
        filas: Filas SFT.
        logger: Logger activo.
        nombre: Nombre del split (para el log).

    Returns:
        Tupla (filas_dedup, stats) con duplicados descartados por tipo.
    """
    vistos_clave: set[tuple] = set()
    vistos_md5: set[str] = set()
    salida: list[dict] = []
    dup_clave = 0
    dup_md5 = 0
    for fila in filas:
        clave = (fila["doc_id"], fila["articulo"], fila["chunk_idx"], fila["pasaje_id"])
        if clave in vistos_clave:
            dup_clave += 1
            continue
        vistos_clave.add(clave)
        if fila["md5_par"] in vistos_md5:
            dup_md5 += 1
            continue
        vistos_md5.add(fila["md5_par"])
        salida.append(fila)
    stats = {"entrada": len(filas), "salida": len(salida), "dup_clave": dup_clave, "dup_md5": dup_md5}
    logger.info(
        "dedupe %s: %d -> %d (dup_clave=%d, dup_md5=%d)",
        nombre,
        len(filas),
        len(salida),
        dup_clave,
        dup_md5,
    )
    return salida, stats


def recortar_balance(
    filas: list[dict], objetivo_pos: int, objetivo_neg: int, logger: logging.Logger, nombre: str
) -> tuple[list[dict], dict]:
    """Recorta el split a los objetivos exactos por clase (50/50).

    Conserva el orden determinista del muestreo (los primeros objetivo_pos
    positivos y objetivo_neg negativos). Si una clase no alcanza el
    objetivo, se advierte (los candidatos traian margen para el dedupe).

    Args:
        filas: Filas SFT del split (ya deduplicadas).
        objetivo_pos: Numero objetivo de positivos.
        objetivo_neg: Numero objetivo de negativos.
        logger: Logger activo.
        nombre: Nombre del split.

    Returns:
        Tupla (filas_recortadas, info).
    """
    disponibles = {0: 0, 1: 0}
    for fila in filas:
        disponibles[fila["label"]] += 1
    if disponibles[1] < objetivo_pos or disponibles[0] < objetivo_neg:
        logger.warning(
            "%s: disponibles pos=%d neg=%d < objetivo pos=%d neg=%d (candidatos insuficientes)",
            nombre,
            disponibles[1],
            disponibles[0],
            objetivo_pos,
            objetivo_neg,
        )
    contadores = {0: 0, 1: 0}
    objetivo = {0: objetivo_neg, 1: objetivo_pos}
    salida = []
    for fila in filas:
        etiqueta = fila["label"]
        if contadores[etiqueta] < objetivo[etiqueta]:
            contadores[etiqueta] += 1
            salida.append(fila)
    info = {
        "disponibles_positivos": disponibles[1],
        "disponibles_negativos": disponibles[0],
        "objetivo_positivos": objetivo_pos,
        "objetivo_negativos": objetivo_neg,
        "salida_positivos": contadores[1],
        "salida_negativos": contadores[0],
    }
    logger.info(
        "recorte %s: %d positivos / %d negativos (objetivo %d/%d)",
        nombre,
        contadores[1],
        contadores[0],
        objetivo_pos,
        objetivo_neg,
    )
    return salida, info


def percentiles(valores: list[float]) -> dict:
    """Calcula p50/p95/max/media de una lista de valores.

    Args:
        valores: Lista de numeros.

    Returns:
        Dict con p50, p95, max y media (None si la lista esta vacia).
    """
    if not valores:
        return {"p50": None, "p95": None, "max": None, "media": None}
    arreglo = np.asarray(valores, dtype=np.float64)
    return {
        "p50": float(np.percentile(arreglo, 50)),
        "p95": float(np.percentile(arreglo, 95)),
        "max": float(arreglo.max()),
        "media": float(arreglo.mean()),
    }


def estadisticas_split(filas: list[dict], nombre: str) -> dict:
    """Calcula conteos, balance, mix de fuentes y longitudes de un split.

    Args:
        filas: Filas SFT del split.
        nombre: Nombre del split.

    Returns:
        Dict serializable con las estadisticas.
    """
    labels = np.asarray([fila["label"] for fila in filas], dtype=np.int64)
    n = len(filas)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    truncados = int(sum(bool(fila["truncado"]) for fila in filas))
    return {
        "nombre": nombre,
        "n": n,
        "n_positivos": n_pos,
        "n_negativos": n_neg,
        "balance_positivo": float(n_pos / n) if n else 0.0,
        "docs": len({fila["doc_id"] for fila in filas}),
        "fuentes_negativos": dict(Counter(fila["fuente_negativo"] for fila in filas if fila["label"] == 0)),
        "truncados": truncados,
        "pct_truncados": float(100.0 * truncados / n) if n else 0.0,
        "tokens_total": percentiles([fila["n_tokens_total"] for fila in filas]),
        "tokens_prompt": percentiles([fila["n_tokens_prompt"] for fila in filas]),
        "chars_textos": percentiles([fila["n_chars_chunk"] + fila["n_chars_pasaje"] for fila in filas]),
    }


def fila_para_parquet(fila: dict) -> dict:
    """Extrae del dict interno solo las columnas del esquema SFT.

    Args:
        fila: Fila con posibles claves internas ("_clave").

    Returns:
        Dict con las claves de ESQUEMA_SFT en orden.
    """
    return {nombre: fila[nombre] for nombre in ESQUEMA_SFT.names}


def escribir_parquet(ruta: str | Path, filas: list[dict], metadatos: dict) -> dict:
    """Escribe un parquet SFT atomico con metadatos de la corrida.

    Args:
        ruta: Ruta destino.
        filas: Filas SFT.
        metadatos: Dict serializable (version/huella/conteos).

    Returns:
        Dict con ruta, filas, bytes y md5.
    """
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabla = pa.Table.from_pylist([fila_para_parquet(fila) for fila in filas], schema=ESQUEMA_SFT)
    tabla = tabla.replace_schema_metadata(
        {CLAVE_METADATOS: json.dumps(metadatos, ensure_ascii=False).encode("utf-8")}
    )
    temporal = destino.parent / (destino.name + ".tmp")
    pq.write_table(tabla, temporal, compression="snappy")
    os.replace(temporal, destino)
    return {
        "ruta": str(destino),
        "filas": len(filas),
        "bytes": destino.stat().st_size,
        "md5": md5_archivo(destino),
    }


def dibujar_dataset(stats: dict, filas_por_split: dict, ruta: str | Path) -> None:
    """Genera la figura del dataset (longitudes, balance, fuentes, truncado).

    Args:
        stats: Dict de estadisticas por split.
        filas_por_split: Dict nombre -> filas (para histogramas).
        ruta: Ruta de la figura.
    """
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura, ejes = plt.subplots(2, 2, figsize=(12, 8))
    colores = {"train": "#1f77b4", "val": "#ff7f0e", "val_eval": "#2ca02c"}
    for nombre, filas in filas_por_split.items():
        longitudes = [fila["n_tokens_total"] for fila in filas]
        ejes[0, 0].hist(longitudes, bins=60, alpha=0.55, label=nombre, color=colores.get(nombre))
    ejes[0, 0].axvline(CONFIG["max_seq"], color="red", linestyle="--", label=f"max_seq={CONFIG['max_seq']}")
    ejes[0, 0].set_title("Longitud total de tokens (prompt + respuesta)")
    ejes[0, 0].set_xlabel("tokens")
    ejes[0, 0].set_ylabel("pares")
    ejes[0, 0].legend()
    nombres = list(filas_por_split)
    x = np.arange(len(nombres))
    pos = [stats[n]["n_positivos"] for n in nombres]
    neg = [stats[n]["n_negativos"] for n in nombres]
    ejes[0, 1].bar(x - 0.2, pos, width=0.4, label="Sí (1)", color="#2ca02c")
    ejes[0, 1].bar(x + 0.2, neg, width=0.4, label="No (0)", color="#d62728")
    ejes[0, 1].set_xticks(x, nombres)
    ejes[0, 1].set_title("Balance de clases por split")
    ejes[0, 1].legend()
    fuentes = stats["train"]["fuentes_negativos"]
    if fuentes:
        etiquetas = list(fuentes)
        ejes[1, 0].bar(etiquetas, [fuentes[k] for k in etiquetas], color="#9467bd")
        ejes[1, 0].tick_params(axis="x", rotation=30)
    ejes[1, 0].set_title("Fuentes de negativos (train)")
    ejes[1, 0].set_ylabel("pares")
    truncados = [stats[n]["pct_truncados"] for n in nombres]
    ejes[1, 1].bar(nombres, truncados, color="#8c564b")
    ejes[1, 1].axhline(2.0, color="red", linestyle="--", label="criterio 2 %")
    ejes[1, 1].set_title("% de pares truncados a max_seq")
    ejes[1, 1].legend()
    figura.tight_layout()
    figura.savefig(destino, dpi=130)
    plt.close(figura)


def imprimir_ejemplos(filas: list[dict], logger: logging.Logger, n: int = 3) -> None:
    """Imprime ejemplos formateados del split de train (verificacion visual).

    Args:
        filas: Filas SFT de train.
        logger: Logger activo.
        n: Numero de ejemplos a imprimir.
    """
    elegidas = []
    positivos = [fila for fila in filas if fila["label"] == 1]
    negativos = [fila for fila in filas if fila["label"] == 0]
    if positivos:
        elegidas.append(positivos[0])
    elegidas.extend(negativos[: n - len(elegidas)])
    for fila in filas:
        if len(elegidas) >= n:
            break
        if fila not in elegidas:
            elegidas.append(fila)
    for indice, fila in enumerate(elegidas[:n], start=1):
        logger.info(
            "EJEMPLO %d | doc=%s art=%s label=%d output=%r tokens=%d truncado=%s",
            indice,
            fila["doc_id"],
            fila["articulo"],
            fila["label"],
            fila["output"],
            fila["n_tokens_total"],
            fila["truncado"],
        )
        logger.info("SYSTEM: %s", fila["instruction"])
        logger.info("USER: %s", fila["input"][:700].replace("\n", " | "))
        logger.info("ASSISTANT: %s", fila["output"])


def preparar_datos(
    cfg: dict, rutas: dict, tokenizer, logger: logging.Logger, forzar: bool, reusar: bool
) -> dict:
    """Construye el dataset SFT del C2 (modo --preparar-datos).

    Args:
        cfg: Configuracion efectiva.
        rutas: Rutas absolutas.
        tokenizer: Tokenizer de ALIA.
        logger: Logger activo.
        forzar: True permite reescribir salidas existentes.
        reusar: True reutiliza un dataset existente (smoke).

    Returns:
        Dict de cierre con conteos, integridad, longitudes y md5.

    Raises:
        RuntimeError: Si falla cualquier chequeo anti-leakage o de split.
    """
    inicio = time.perf_counter()
    rutas_salida = {
        "train": Path(rutas["dir_sft"]) / "train.parquet",
        "val": Path(rutas["dir_sft"]) / "val.parquet",
        "val_eval": Path(rutas["dir_sft"]) / "val_eval.parquet",
    }
    if reusar and all(r.exists() for r in rutas_salida.values()):
        logger.info("dataset SFT existente reutilizado (smoke): %s", rutas["dir_sft"])
        return {"reutilizado": True, "rutas": {k: str(v) for k, v in rutas_salida.items()}}
    for ruta in rutas_salida.values():
        verificar_salida_nueva(ruta, forzar, cfg["smoke"], logger)
    verificar_salida_nueva(rutas["ruta_dataset_json"], forzar, cfg["smoke"], logger)
    verificar_salida_nueva(rutas["ruta_figura_dataset"], forzar, cfg["smoke"], logger)

    exclusiones = leer_exclusiones(rutas, logger)
    shards = descubrir_shards(rutas, logger)
    info_shards = validar_shards(shards, logger)
    indices = leer_indices(shards, logger)

    docs_shards = {str(doc) for item in indices for doc in item["doc_id"]}
    n_bench_shards = len(docs_shards & exclusiones["benchmark"])
    n_gold_shards = len(docs_shards & exclusiones["gold"])
    _exigir(n_bench_shards == 0, f"ABORT: {n_bench_shards} docs del benchmark M7 aparecen en los shards B1")
    _exigir(n_gold_shards == 0, f"ABORT: {n_gold_shards} docs de packs gold aparecen en los shards B1")
    logger.info(
        "[anti-leakage] benchmark: %d ids, en shards: 0; gold: %d ids, en shards: 0",
        len(exclusiones["benchmark"]),
        len(exclusiones["gold"]),
    )

    docs_train, docs_val, info_split = dividir_docs_b1(indices, cfg, logger)
    por_shard, shard_de_doc = mapear_filas_por_doc(indices)
    sel_train, stats_muestreo_train = muestrear_train(
        indices, por_shard, shard_de_doc, docs_train, cfg, logger
    )
    sel_val, sel_val_eval, stats_muestreo_val = muestrear_val(
        indices, por_shard, shard_de_doc, docs_val, cfg, logger
    )
    _exigir(
        stats_muestreo_val["val_eval_subconjunto_val"],
        "val_eval no es subconjunto de val",
    )
    textos = leer_textos(shards, sel_train + sel_val + sel_val_eval, logger)
    filas_train = [construir_fila_sft(indices, clave, textos, cfg) for clave in sel_train]
    filas_val = [construir_fila_sft(indices, clave, textos, cfg) for clave in sel_val]
    filas_val_eval = [construir_fila_sft(indices, clave, textos, cfg) for clave in sel_val_eval]

    filas_train, dedup_train = deduplicar(filas_train, logger, "train")
    filas_val, dedup_val = deduplicar(filas_val, logger, "val")
    filas_val_eval, dedup_eval = deduplicar(filas_val_eval, logger, "val_eval")
    md5_train = {fila["md5_par"] for fila in filas_train}
    n_fuga_val = sum(1 for fila in filas_val if fila["md5_par"] in md5_train)
    n_fuga_eval = sum(1 for fila in filas_val_eval if fila["md5_par"] in md5_train)
    filas_val = [fila for fila in filas_val if fila["md5_par"] not in md5_train]
    filas_val_eval = [fila for fila in filas_val_eval if fila["md5_par"] not in md5_train]
    logger.info(
        "chequeo cruzado train/val: %d filas de val y %d de val_eval con md5 presente en train (descartadas)",
        n_fuga_val,
        n_fuga_eval,
    )
    filas_train, balance_train = recortar_balance(
        filas_train, cfg["train_pos"], cfg["train_neg"], logger, "train"
    )
    filas_val, balance_val = recortar_balance(
        filas_val, cfg["val_pos"], cfg["val_neg"], logger, "val"
    )
    claves_val = {fila["_clave"] for fila in filas_val}
    filas_val_eval = [fila for fila in filas_val_eval if fila["_clave"] in claves_val]
    filas_val_eval, balance_eval = recortar_balance(
        filas_val_eval, cfg["val_eval_pos"], cfg["val_eval_neg"], logger, "val_eval"
    )

    ids_ctx = ids_respuesta_contexto(tokenizer, cfg, logger)
    tokenizar_filas(filas_train, tokenizer, cfg, ids_ctx, logger)
    tokenizar_filas(filas_val, tokenizer, cfg, ids_ctx, logger)
    tokenizar_filas(filas_val_eval, tokenizer, cfg, ids_ctx, logger)

    n_bench_train = len({fila["doc_id"] for fila in filas_train} & exclusiones["benchmark"])
    n_bench_val = len({fila["doc_id"] for fila in filas_val} & exclusiones["benchmark"])
    n_gold_train = len({fila["doc_id"] for fila in filas_train} & exclusiones["gold"])
    n_gold_val = len({fila["doc_id"] for fila in filas_val} & exclusiones["gold"])
    logger.info(
        "[anti-leakage] en train/val -> benchmark: %d/%d; gold: %d/%d (deben ser 0)",
        n_bench_train,
        n_bench_val,
        n_gold_train,
        n_gold_val,
    )
    _exigir(
        n_bench_train == 0 and n_bench_val == 0 and n_gold_train == 0 and n_gold_val == 0,
        "ABORT: docs del benchmark/gold dentro de train o val",
    )

    huella = md5_bytes(
        json.dumps(
            {
                "version": cfg["version_dataset"],
                "semilla": cfg["semilla"],
                "val_frac": cfg["val_frac"],
                "caps": {
                    clave: cfg[clave]
                    for clave in (
                        "cap_positivos_por_doc",
                        "cap_negativos_por_doc",
                        "train_pos",
                        "train_neg",
                        "val_pos",
                        "val_neg",
                        "val_eval_pos",
                        "val_eval_neg",
                        "max_chars_chunk",
                        "max_chars_pasaje",
                        "max_seq",
                    )
                },
                "shards_md5": [s["md5"] for s in info_shards["shards"]],
                "benchmark_md5": exclusiones["detalle"]["benchmark"]["md5_v2"],
            },
            sort_keys=True,
        )
    )
    metadatos = {"version": cfg["version_dataset"], "huella": huella, "smoke": cfg["smoke"]}
    salidas = {
        "train": escribir_parquet(rutas_salida["train"], filas_train, metadatos),
        "val": escribir_parquet(rutas_salida["val"], filas_val, metadatos),
        "val_eval": escribir_parquet(rutas_salida["val_eval"], filas_val_eval, metadatos),
    }
    stats = {
        "train": estadisticas_split(filas_train, "train"),
        "val": estadisticas_split(filas_val, "val"),
        "val_eval": estadisticas_split(filas_val_eval, "val_eval"),
    }
    for nombre, stat in stats.items():
        logger.info(
            "split %s: n=%d (pos=%d, neg=%d), docs=%d, truncados=%.2f %%, tokens p50/p95/max=%.0f/%.0f/%.0f",
            nombre,
            stat["n"],
            stat["n_positivos"],
            stat["n_negativos"],
            stat["docs"],
            stat["pct_truncados"],
            stat["tokens_total"]["p50"],
            stat["tokens_total"]["p95"],
            stat["tokens_total"]["max"],
        )
        if stat["n"] and abs(stat["balance_positivo"] - 0.5) > 0.02:
            logger.warning("balance fuera de 50 +/- 2 %% en %s: %.4f", nombre, stat["balance_positivo"])
        if stat["pct_truncados"] >= 2.0:
            logger.warning("truncados >= 2 %% en %s: %.2f %%", nombre, stat["pct_truncados"])
    imprimir_ejemplos(filas_train, logger, n=3)
    dibujar_dataset(stats, {"train": filas_train, "val": filas_val, "val_eval": filas_val_eval}, rutas["ruta_figura_dataset"])

    cierre = {
        "version": cfg["version_dataset"],
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "modo": "smoke" if cfg["smoke"] else "completo",
        "huella": huella,
        "entorno": versiones_entorno(),
        "fuentes": {
            "shards": info_shards,
            "benchmark": exclusiones["detalle"]["benchmark"],
            "gold": exclusiones["detalle"]["gold"],
        },
        "integridad": {
            "benchmark_en_shards": n_bench_shards,
            "gold_en_shards": n_gold_shards,
            "benchmark_en_train": n_bench_train,
            "benchmark_en_val": n_bench_val,
            "gold_en_train": n_gold_train,
            "gold_en_val": n_gold_val,
            "split_identico_b1": True,
            "disjuncion_train_val": True,
            "val_eval_subconjunto_val": True,
            "fuga_md5_train_a_val": n_fuga_val,
            "fuga_md5_train_a_val_eval": n_fuga_eval,
        },
        "split": info_split,
        "muestreo": {"train": stats_muestreo_train, "val": stats_muestreo_val},
        "dedupe": {"train": dedup_train, "val": dedup_val, "val_eval": dedup_eval},
        "balance": {"train": balance_train, "val": balance_val, "val_eval": balance_eval},
        "conteos": stats,
        "longitudes": {
            "max_seq": cfg["max_seq"],
            "caps_chars": {"chunk": cfg["max_chars_chunk"], "pasaje": cfg["max_chars_pasaje"]},
            "tokens_total_train": stats["train"]["tokens_total"],
            "tokens_total_val": stats["val"]["tokens_total"],
            "criterio_truncado_pct": 2.0,
        },
        "ids_respuesta": ids_ctx,
        "salidas": salidas,
        "config": {
            clave: cfg[clave]
            for clave in (
                "semilla",
                "val_frac",
                "cap_positivos_por_doc",
                "cap_negativos_por_doc",
                "train_pos",
                "train_neg",
                "val_pos",
                "val_neg",
                "val_eval_pos",
                "val_eval_neg",
                "max_chars_chunk",
                "max_chars_pasaje",
                "max_seq",
                "system",
                "plantilla_user",
            )
        },
    }
    destino_json = Path(rutas["ruta_dataset_json"])
    destino_json.parent.mkdir(parents=True, exist_ok=True)
    with open(destino_json, "w", encoding="utf-8") as manejador:
        json.dump(cierre, manejador, ensure_ascii=False, indent=2)
    logger.info(
        "dataset listo en %.1f s | train=%d val=%d val_eval=%d | json=%s | figura=%s",
        time.perf_counter() - inicio,
        len(filas_train),
        len(filas_val),
        len(filas_val_eval),
        rutas["ruta_dataset_json"],
        rutas["ruta_figura_dataset"],
    )
    return cierre


# ===== SECCION 4: modelo y collator completion-only =====


def verificar_modelo_base(cfg: dict, rutas: dict, logger: logging.Logger) -> dict:
    """Verifica los safetensors del modelo base contra el index.

    Args:
        cfg: Configuracion efectiva.
        rutas: Rutas absolutas.
        logger: Logger activo.

    Returns:
        Dict con n_shards, bytes y total declarado.

    Raises:
        RuntimeError: Si falta un shard o el tamano no cuadra con el index.
    """
    dir_modelo = Path(rutas["ruta_modelo"])
    index = json.loads((dir_modelo / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    faltantes = [s for s in shards if not (dir_modelo / s).exists()]
    _exigir(not faltantes, f"faltan shards del modelo base: {faltantes}")
    bytes_reales = sum((dir_modelo / s).stat().st_size for s in shards)
    declarado = int(index["metadata"]["total_size"])
    _exigir(
        abs(bytes_reales - declarado) <= 0.01 * declarado,
        f"safetensors ({bytes_reales} bytes) != index ({declarado} bytes)",
    )
    logger.info(
        "modelo base verificado: %d safetensors, %.2f GiB (index OK)",
        len(shards),
        bytes_reales / 1024**3,
    )
    return {"n_shards": len(shards), "bytes": bytes_reales, "total_declarado": declarado}


def cargar_tokenizer(rutas: dict):
    """Carga el tokenizer local de ALIA.

    Args:
        rutas: Rutas absolutas.

    Returns:
        Tokenizer con pad_token = eos (necesario para el padding del collator).
    """
    tokenizer = AutoTokenizer.from_pretrained(rutas["ruta_modelo"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def cargar_modelo_lora(cfg: dict, rutas: dict, device: str, dir_reanudar: str | None, logger: logging.Logger):
    """Carga el modelo base en bf16 y aplica (o reanuda) el adaptador LoRA.

    Args:
        cfg: Configuracion efectiva.
        rutas: Rutas absolutas.
        device: "cuda".
        dir_reanudar: Directorio del checkpoint a reanudar (o None).
        logger: Logger activo.

    Returns:
        PeftModel listo para entrenar.

    Raises:
        RuntimeError: Si el adaptador a reanudar no coincide en config.
    """
    base = AutoModelForCausalLM.from_pretrained(
        rutas["ruta_modelo"],
        dtype=torch.bfloat16,
        attn_implementation=cfg["attn_implementation"],
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    if dir_reanudar is not None:
        config_guardada = json.loads(
            (Path(dir_reanudar) / "adapter_config.json").read_text(encoding="utf-8")
        )
        _exigir(
            int(config_guardada["r"]) == cfg["lora_r"]
            and int(config_guardada["lora_alpha"]) == cfg["lora_alpha"],
            (
                f"adaptador del checkpoint (r={config_guardada['r']}, alpha={config_guardada['lora_alpha']}) "
                f"!= config pedida (r={cfg['lora_r']}, alpha={cfg['lora_alpha']})"
            ),
        )
        modulos_guardados = config_guardada.get("target_modules")
        if isinstance(modulos_guardados, list):
            _exigir(
                sorted(modulos_guardados) == sorted(cfg["target_modules"]),
                f"target_modules del checkpoint ({modulos_guardados}) != config pedida ({cfg['target_modules']})",
            )
        modelo = PeftModel.from_pretrained(base, dir_reanudar, is_trainable=True)
        logger.info("adaptador reanudado desde %s", dir_reanudar)
    else:
        config_lora = LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(cfg["target_modules"]),
        )
        modelo = get_peft_model(base, config_lora)
    modelo.to(device)
    if hasattr(modelo, "print_trainable_parameters"):
        modelo.print_trainable_parameters()
    return modelo


def _base_del_modelo(modelo):
    """Obtiene el modelo transformers subyacente de un PeftModel.

    Args:
        modelo: PeftModel o modelo transformers.

    Returns:
        Modelo transformers subyacente.
    """
    return modelo.get_base_model() if hasattr(modelo, "get_base_model") else modelo


def activar_checkpointing(modelo, activo: bool) -> None:
    """Activa/desactiva gradient checkpointing (uso alrededor de la eval).

    Args:
        modelo: PeftModel.
        activo: True para activar (use_reentrant=False), False para desactivar.
    """
    base = _base_del_modelo(modelo)
    if not hasattr(base, "gradient_checkpointing_enable"):
        return
    if activo:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        base.gradient_checkpointing_disable()


def cargar_sft(ruta: str | Path) -> dict:
    """Carga un parquet SFT como dict de listas por columna.

    Args:
        ruta: Ruta del parquet.

    Returns:
        Dict columna -> lista de valores.
    """
    tabla = pq.read_table(ruta)
    return {nombre: tabla[nombre].to_pylist() for nombre in tabla.column_names}


def preparar_ejemplos(datos: dict, max_seq: int) -> list[dict]:
    """Convierte el parquet SFT en ejemplos tokenizados listos para el loop.

    Si max_seq es menor que el render guardado, se recorta SOLO el prompt
    (la respuesta se preserva); permite el ladder 768 -> 512 sin re-render.

    Args:
        datos: Dict de listas por columna (cargar_sft).
        max_seq: Longitud maxima efectiva.

    Returns:
        Lista de dicts con prompt_ids, resp_ids y label.
    """
    ejemplos = []
    for i in range(len(datos["label"])):
        prompt = [int(t) for t in datos["prompt_ids"][i]]
        resp = [int(t) for t in datos["resp_ids"][i]]
        if len(prompt) + len(resp) > max_seq:
            prompt = prompt[: max_seq - len(resp)]
        ejemplos.append({"prompt_ids": prompt, "resp_ids": resp, "label": int(datos["label"][i])})
    return ejemplos


def collar_lote(ejemplos: list[dict], pad_id: int, lado: str = "derecha") -> dict:
    """Collator completion-only: targets -100 en el prompt.

    Devuelve input_ids/attention_mask con padding y una matriz de targets
    alineada a las ULTIMAS K-1 posiciones (K = max_resp + 1), de modo que
    cada fila aporta exactamente sus tokens de respuesta a la loss.

    CRITICO (bug de scoring corregido 2026-10-05): con batch > 1 el padding
    debe ser IZQUIERDO. Los logits se recortan con logits_to_keep a las
    ULTIMAS K posiciones del tensor PADEADO; con padding derecho esas
    posiciones pertenecen a las filas largas y la loss/score de las filas
    cortas se leia en posiciones equivocadas (falso stop del run v1:
    AUC 0,44 interno vs 0,80 con la ruta validada del script 34, que usa
    padding izquierdo). Con padding izquierdo la respuesta de TODAS las
    filas queda al final del tensor y las formulas de offset (K - r - 1)
    son exactas para cada fila. Con batch=1 (sin padding) ambos lados son
    equivalentes, por lo que la ruta de entrenamiento no cambia.

    Args:
        ejemplos: Lista de dicts con prompt_ids/resp_ids/label.
        pad_id: Id de padding (eos).
        lado: "derecha" (default, batch 1) o "izquierda" (batch > 1).

    Returns:
        Dict de tensores: input_ids, attention_mask, targets, resp_lens.

    Raises:
        RuntimeError: Si hay padding con lado="derecha" (desalinea la
            loss/score por fila) o si el lado es invalido.
    """
    _exigir(lado in ("derecha", "izquierda"), f"lado de padding invalido: {lado!r}")
    n = len(ejemplos)
    largos = [len(e["prompt_ids"]) + len(e["resp_ids"]) for e in ejemplos]
    largo_max = max(largos)
    hay_padding = any(largo < largo_max for largo in largos)
    _exigir(
        lado == "izquierda" or not hay_padding,
        "padding derecho con filas de distinto largo desalinea la loss/score por fila: "
        "use lado='izquierda' cuando haya padding",
    )
    k = max(len(e["resp_ids"]) for e in ejemplos) + 1
    input_ids = torch.full((n, largo_max), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n, largo_max), dtype=torch.long)
    targets = torch.full((n, k - 1), -100, dtype=torch.long)
    resp_lens = torch.zeros(n, dtype=torch.long)
    for i, ejemplo in enumerate(ejemplos):
        ids = ejemplo["prompt_ids"] + ejemplo["resp_ids"]
        p = len(ejemplo["prompt_ids"])
        r = len(ejemplo["resp_ids"])
        if lado == "izquierda":
            inicio = largo_max - len(ids)
            input_ids[i, inicio:] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, inicio:] = 1
        else:
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
        resp_lens[i] = r
        desplazamiento = k - r - 1
        for j in range(r):
            targets[i, desplazamiento + j] = ids[p + j]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "targets": targets,
        "resp_lens": resp_lens,
    }


def forward_completion(modelo, lote: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward con loss completion-only usando logits_to_keep.

    El modelo devuelve logits solo para las ultimas K posiciones
    (logits_to_keep=K, verificado en transformers 5.16); la loss se calcula
    contra la matriz de targets (posiciones de respuesta, -100 en el resto).

    Args:
        modelo: PeftModel en modo train o eval.
        lote: Salida de collar_lote (tensores en CPU).
        device: "cuda".

    Returns:
        Tupla (loss, logits) con logits de shape (B, K, vocab).
    """
    input_ids = lote["input_ids"].to(device)
    attention_mask = lote["attention_mask"].to(device)
    targets = lote["targets"].to(device)
    k = targets.shape[1] + 1
    salida = modelo(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=k)
    logits = salida.logits
    planos = logits[:, : k - 1, :].float().reshape(-1, logits.shape[-1])
    perdida = F.cross_entropy(planos, targets.reshape(-1), ignore_index=-100)
    return perdida, logits


# ===== SECCION 5: entrenamiento, evaluacion y checkpoints =====


def lr_en_paso(paso: int, total: int, base: float, warmup_frac: float, min_frac: float) -> float:
    """Learning rate del paso con warmup lineal + cosine hasta min_lr.

    Args:
        paso: Paso de optimizador (1-based).
        total: Total de pasos planificados.
        base: LR base.
        warmup_frac: Fraccion de warmup (0,03).
        min_frac: Fraccion minima del LR base (0,10).

    Returns:
        LR del paso.
    """
    calentamiento = max(1, round(warmup_frac * total))
    minimo = base * min_frac
    if paso <= calentamiento:
        return base * paso / calentamiento
    avance = min(1.0, (paso - calentamiento) / max(1, total - calentamiento))
    return minimo + (base - minimo) * 0.5 * (1.0 + math.cos(math.pi * avance))


def guardar_checkpoint(modelo, optimizer, estado: dict, rutas: dict, logger: logging.Logger) -> None:
    """Guarda el checkpoint reanudable (adapter + optimizer + estado + RNG).

    Args:
        modelo: PeftModel.
        optimizer: AdamW.
        estado: Dict de estado (paso, epoca, mejor, historial).
        rutas: Rutas absolutas.
        logger: Logger activo.
    """
    destino = Path(rutas["dir_checkpoint"])
    destino.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(destino)
    torch.save({"optimizer": optimizer.state_dict(), "estado": estado}, destino / "entrenamiento_estado.pt")
    torch.save(
        {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        destino / "rng.pt",
    )
    logger.info("checkpoint guardado en %s (paso=%d)", destino, estado["paso"])


def cargar_checkpoint(modelo, optimizer, rutas: dict, device: str, logger: logging.Logger) -> dict:
    """Reanuda el estado desde modelos/c2_lora/checkpoints/ultimo.

    Args:
        modelo: PeftModel ya cargado desde el mismo directorio.
        optimizer: AdamW a poblar.
        rutas: Rutas absolutas.
        device: "cuda".
        logger: Logger activo.

    Returns:
        Dict de estado (paso, epoca, mejor_auc, historial, config, tokens).

    Raises:
        RuntimeError: Si no existe el checkpoint.
    """
    destino = Path(rutas["dir_checkpoint"])
    _exigir(
        (destino / "entrenamiento_estado.pt").exists(),
        f"no hay checkpoint reanudable en {destino} (ejecute --entrenar sin --reanudar)",
    )
    datos = torch.load(destino / "entrenamiento_estado.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(datos["optimizer"])
    for estado_param in optimizer.state.values():
        for clave, valor in estado_param.items():
            if torch.is_tensor(valor):
                estado_param[clave] = valor.to(device)
    rng = torch.load(destino / "rng.pt", map_location="cpu", weights_only=False)
    torch.set_rng_state(rng["torch"])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    logger.info("checkpoint reanudado: paso=%d, epoca=%d", datos["estado"]["paso"], datos["estado"]["epoca"])
    return datos["estado"]


@torch.inference_mode()
def evaluar(
    modelo,
    ejemplos: list[dict],
    cfg: dict,
    tokenizer,
    device: str,
    logger: logging.Logger,
) -> dict:
    """Evalua el modelo en un split: loss, acc, F1, AUC y matriz de confusion.

    El score del juez es P("Si") normalizada sobre el par {Si, No} en la
    primera posicion de respuesta (ids en contexto verificados); el umbral
    de decision es 0.5 (la optimizacion de umbral es del script 34, con val).

    Args:
        modelo: PeftModel.
        ejemplos: Ejemplos tokenizados.
        cfg: Configuracion efectiva.
        tokenizer: Tokenizer (pad).
        device: "cuda".
        logger: Logger activo.

    Returns:
        Dict con loss, acc, f1_si, f1_macro, auc, matriz y n.
    """
    del logger
    activar_checkpointing(modelo, False)
    try:
        modelo.eval()
        id_si = cfg["ids_respuesta"]["contexto"]["Sí"]
        id_no = cfg["ids_respuesta"]["contexto"]["No"]
        suma_perdida = 0.0
        n_tokens_validos = 0
        probs: list[float] = []
        etiquetas: list[int] = []
        lote_tam = cfg["eval_batch"]
        for inicio in range(0, len(ejemplos), lote_tam):
            # Padding IZQUIERDO (fix del falso stop): con batch > 1 las filas
            # cortas quedan alineadas a la derecha y la posicion de respuesta
            # (offset K - r - 1) es exacta para todas.
            lote = collar_lote(
                ejemplos[inicio : inicio + lote_tam], tokenizer.pad_token_id, lado="izquierda"
            )
            perdida, logits = forward_completion(modelo, lote, device)
            objetivos = lote["targets"].to(device)
            n_validos = int((objetivos != -100).sum())
            suma_perdida += float(perdida) * n_validos
            n_tokens_validos += n_validos
            k = logits.shape[1]
            desplazamientos = (k - lote["resp_lens"] - 1).to(device)
            filas = torch.arange(logits.shape[0], device=device)
            # Posicion real en el tensor padeado de la ventana recortada por
            # logits_to_keep: offset j -> posicion largo_max - k + j.
            largo_max = lote["input_ids"].shape[1]
            posiciones = largo_max - k + desplazamientos
            _exigir(
                bool(lote["attention_mask"][filas.cpu(), posiciones.cpu()].all()),
                "posicion de scoring sobre padding: el padding izquierdo no se aplico correctamente",
            )
            logit_si = logits[filas, desplazamientos, id_si]
            logit_no = logits[filas, desplazamientos, id_no]
            par = torch.softmax(torch.stack([logit_si, logit_no], dim=1).float(), dim=1)[:, 0]
            probs.extend(par.tolist())
            etiquetas.extend(e["label"] for e in ejemplos[inicio : inicio + lote_tam])
        etiquetas_np = np.asarray(etiquetas, dtype=np.int64)
        probs_np = np.asarray(probs, dtype=np.float64)
        predicciones = (probs_np >= 0.5).astype(np.int64)
        metricas = {
            "loss": float(suma_perdida / n_tokens_validos) if n_tokens_validos else None,
            "acc": float(accuracy_score(etiquetas_np, predicciones)),
            "f1_si": float(f1_score(etiquetas_np, predicciones, pos_label=1, zero_division=0)),
            "f1_macro": float(f1_score(etiquetas_np, predicciones, average="macro", zero_division=0)),
            "auc": (
                float(roc_auc_score(etiquetas_np, probs_np))
                if len(np.unique(etiquetas_np)) > 1
                else None
            ),
            "matriz": confusion_matrix(etiquetas_np, predicciones, labels=[0, 1]).tolist(),
            "n": len(ejemplos),
        }
        return metricas
    finally:
        activar_checkpointing(modelo, True)
        modelo.train()


def dibujar_curvas(historial: list[dict], ruta: str | Path) -> None:
    """Grafica las curvas de entrenamiento (loss, acc, AUC vs paso).

    Args:
        historial: Lista de dicts por evaluacion.
        ruta: Ruta de la figura.
    """
    if not historial:
        return
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    pasos = [h["paso"] for h in historial]
    figura, ejes = plt.subplots(1, 3, figsize=(15, 4))
    # Si el historial mezcla scoring antiguo (bug_padding) y corregido, se
    # marca el primer paso con metrica corregida (el salto no es del modelo).
    primeros_corregidos = [h["paso"] for h in historial if h.get("scoring") == "corregido"]
    if primeros_corregidos and any(h.get("scoring") == "bug_padding" for h in historial):
        for eje in ejes:
            eje.axvline(primeros_corregidos[0], color="gray", linestyle=":", linewidth=1)
    ejes[0].plot(pasos, [h["loss_train"] for h in historial], marker="o")
    ejes[0].set_title("Loss de train (media por ventana)")
    ejes[0].set_xlabel("paso")
    ejes[1].plot(pasos, [h["acc"] for h in historial], marker="o", color="#2ca02c")
    ejes[1].axhline(0.78, color="red", linestyle="--", label="objetivo 0,78")
    ejes[1].set_title("Accuracy val_eval")
    ejes[1].set_xlabel("paso")
    ejes[1].legend()
    aucs = [h["auc"] for h in historial]
    ejes[2].plot(pasos, [a if a is not None else np.nan for a in aucs], marker="o", color="#9467bd")
    ejes[2].axhline(0.85, color="red", linestyle="--", label="objetivo 0,85")
    ejes[2].set_title("AUC val_eval")
    ejes[2].set_xlabel("paso")
    ejes[2].legend()
    figura.tight_layout()
    figura.savefig(destino, dpi=130)
    plt.close(figura)


def resolver_max_seq_evaluacion(cfg: dict, rutas: dict, args: argparse.Namespace, logger: logging.Logger) -> int:
    """Alinea max_seq al del entrenamiento registrado (patron del script 34).

    Prioridad: --max-seq explicito > config.max_seq del JSON de entrenamiento
    > config.max_seq del checkpoint > default del CONFIG. Evita evaluar con
    un presupuesto de prompt distinto al usado al entrenar.

    Args:
        cfg: Configuracion efectiva.
        rutas: Rutas absolutas.
        args: Namespace de la CLI.
        logger: Logger activo.

    Returns:
        max_seq efectivo para la evaluacion.
    """
    if args.max_seq is not None:
        return cfg["max_seq"]
    ruta_json = Path(rutas["ruta_train_json"])
    if ruta_json.exists():
        try:
            max_seq = int(json.loads(ruta_json.read_text(encoding="utf-8"))["config"]["max_seq"])
        except (KeyError, ValueError, json.JSONDecodeError):
            max_seq = None
        if max_seq is not None:
            if max_seq != cfg["max_seq"]:
                logger.info(
                    "max_seq alineado al del entrenamiento: %d -> %d (use --max-seq para forzarlo)",
                    cfg["max_seq"],
                    max_seq,
                )
            return max_seq
    ruta_estado = Path(rutas["dir_checkpoint"]) / "entrenamiento_estado.pt"
    if ruta_estado.exists():
        estado = torch.load(ruta_estado, map_location="cpu", weights_only=False)["estado"]
        max_seq = int(estado["config"]["max_seq"])
        if max_seq != cfg["max_seq"]:
            logger.info("max_seq alineado al del checkpoint: %d -> %d", cfg["max_seq"], max_seq)
        return max_seq
    return cfg["max_seq"]


def cargar_modelo_eval(cfg: dict, rutas: dict, device: str, ruta_adapter: str | None, logger: logging.Logger):
    """Carga el modelo base bf16 y (opcionalmente) el adaptador para evaluar.

    No activa gradient checkpointing (evaluacion pura en inference_mode) ni
    envuelve el modelo si ruta_adapter es None (control base bf16).

    Args:
        cfg: Configuracion efectiva.
        rutas: Rutas absolutas.
        device: "cuda".
        ruta_adapter: Ruta del adaptador o None para el control base.
        logger: Logger activo.

    Returns:
        Modelo en modo eval listo para puntuar.
    """
    base = AutoModelForCausalLM.from_pretrained(
        rutas["ruta_modelo"],
        dtype=torch.bfloat16,
        attn_implementation=cfg["attn_implementation"],
    )
    base.config.use_cache = False
    if ruta_adapter is not None:
        modelo = PeftModel.from_pretrained(base, ruta_adapter, is_trainable=False)
        logger.info("adaptador cargado para evaluacion desde %s (sin merge)", ruta_adapter)
    else:
        modelo = base
        logger.info("control base bf16 sin adaptador (disable_adapter equivalente)")
    modelo.to(device)
    modelo.eval()
    return modelo


def evaluar_adaptador(
    cfg: dict,
    rutas: dict,
    tokenizer,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> dict:
    """Evalua un adaptador guardado con la ruta de scoring CORREGIDA (--evaluar).

    Carga val.parquet (o los primeros --max-items pares), alinea max_seq al
    del entrenamiento y calcula loss/acc/F1/AUC/matriz con el mismo scoring
    del bucle de entrenamiento (padding izquierdo). Con --sin-adaptador
    evalua el modelo base bf16 como control. Escribe un JSON con metricas,
    md5 del adaptador y la version de scoring; NO toca el JSON de
    entrenamiento ni el checkpoint.

    Args:
        cfg: Configuracion efectiva.
        rutas: Rutas absolutas.
        tokenizer: Tokenizer de ALIA.
        args: Namespace de la CLI.
        logger: Logger activo.

    Returns:
        Dict de cierre con las metricas.

    Raises:
        RuntimeError: Si falta val.parquet o la VRAM es insuficiente.
    """
    device = "cuda"
    _verificar_gpu_exclusiva(cfg, logger)
    cfg["max_seq"] = resolver_max_seq_evaluacion(cfg, rutas, args, logger)
    ruta_val = Path(rutas["dir_sft"]) / "val.parquet"
    _exigir(ruta_val.exists(), f"falta {ruta_val} (ejecute --preparar-datos primero)")
    datos_val = cargar_sft(ruta_val)
    ejemplos = preparar_ejemplos(datos_val, cfg["max_seq"])
    if args.max_items is not None:
        ejemplos = ejemplos[: args.max_items]
        datos_val = {
            clave: datos_val[clave][: args.max_items]
            for clave in ("doc_id", "label")
        }
    ids_ctx = ids_respuesta_contexto(tokenizer, cfg, logger)
    cfg["ids_respuesta"] = ids_ctx
    ruta_adapter = None if args.sin_adaptador else (args.adapter or rutas["dir_adapter"])
    if ruta_adapter is not None:
        _exigir(Path(ruta_adapter).exists(), f"no existe el adaptador {ruta_adapter}")
    modelo = cargar_modelo_eval(cfg, rutas, device, ruta_adapter, logger)
    inicio = time.perf_counter()
    metricas = evaluar(modelo, ejemplos, cfg, tokenizer, device, logger)
    duracion = time.perf_counter() - inicio
    logger.info(
        "evaluacion (%s, n=%d, max_seq=%d): loss=%s acc=%.4f f1_si=%.4f auc=%s matriz=%s [%.1f s]",
        "adaptador" if ruta_adapter else "base",
        len(ejemplos),
        cfg["max_seq"],
        f"{metricas['loss']:.4f}" if metricas["loss"] is not None else "n/a",
        metricas["acc"],
        metricas["f1_si"],
        f"{metricas['auc']:.4f}" if metricas["auc"] is not None else "n/a",
        metricas["matriz"],
        duracion,
    )
    cierre = {
        "version": "c2-lora-eval-interna-v1",
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "scripts/33_c2_lora_train.py",
        "modo": "evaluar",
        "scoring": "corregido (padding izquierdo; ids en contexto verificados)",
        "dispositivo": torch.cuda.get_device_name(0),
        "max_seq": cfg["max_seq"],
        "n_items": len(ejemplos),
        "n_docs": len({doc for doc in datos_val["doc_id"]}),
        "val": {
            "ruta": str(ruta_val),
            "md5": md5_archivo(ruta_val),
            "orden": "primeros --max-items pares de val.parquet (mismo criterio que scripts/34)",
        },
        "adaptador": {
            "ruta": ruta_adapter,
            "md5_modelo": (
                md5_archivo(Path(ruta_adapter) / "adapter_model.safetensors") if ruta_adapter else None
            ),
            "md5_config": md5_archivo(Path(ruta_adapter) / "adapter_config.json") if ruta_adapter else None,
        },
        "ids_respuesta": ids_ctx,
        "metricas": metricas,
        "tiempos": {"segundos": duracion},
        "entorno": versiones_entorno(),
    }
    ruta_salida = args.salida or str(RAIZ / "resultados/c2_lora/eval_interna_v1.json")
    destino = verificar_salida_nueva(ruta_salida, args.forzar, cfg["smoke"], logger)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as manejador:
        json.dump(cierre, manejador, ensure_ascii=False, indent=2)
    logger.info("evaluacion guardada: %s", destino)
    return cierre


def _verificar_gpu_exclusiva(cfg: dict, logger: logging.Logger) -> tuple[float, float]:
    """Exige GPU disponible con la VRAM libre minima del plan.

    Args:
        cfg: Configuracion efectiva.
        logger: Logger activo.

    Returns:
        Tupla (libre_gib, total_gib).

    Raises:
        RuntimeError: Si no hay GPU o la VRAM libre es insuficiente.
    """
    _exigir(torch.cuda.is_available(), "CUDA/ROCm no disponible: se requiere GPU")
    libre, total = torch.cuda.mem_get_info()
    libre_gib = libre / 1024**3
    logger.info(
        "GPU %s: %.2f GiB libres de %.2f GiB",
        torch.cuda.get_device_name(0),
        libre_gib,
        total / 1024**3,
    )
    _exigir(
        libre_gib >= cfg["vram_min_gib"],
        (
            f"VRAM libre insuficiente: {libre_gib:.2f} GiB < {cfg['vram_min_gib']} GiB. "
            "El plan exige GPU EXCLUSIVA (cierrese Chrome/navegador y todo job GPU)."
        ),
    )
    return libre_gib, total / 1024**3


def migrar_metricas_scoring(estado: dict, logger: logging.Logger) -> dict:
    """Migra el estado de un checkpoint previo al fix de scoring.

    Los checkpoints anteriores al fix (padding derecho en evaluar) guardaron
    evals invalidos: se etiquetan "bug_padding" y se reinician
    mejor_auc/sin_mejora para no comparar metricas incompatibles (el primer
    eval corregido define el mejor). Idempotente.

    Args:
        estado: Estado cargado del checkpoint (se modifica in-place).
        logger: Logger activo.

    Returns:
        El mismo estado migrado.
    """
    if estado.get("scoring_version") == "corregido":
        return estado
    entradas = estado.get("historial", [])
    for entrada in entradas:
        entrada.setdefault("scoring", "bug_padding")
    logger.warning(
        "checkpoint previo al fix de scoring: %d evals marcados 'bug_padding'; "
        "se reinician mejor_auc (era %s) y sin_mejora (era %d) para la metrica corregida",
        len(entradas),
        estado.get("mejor_auc"),
        estado.get("sin_mejora", 0),
    )
    estado["mejor_auc"] = None
    estado["mejor_paso"] = 0
    estado["sin_mejora"] = 0
    estado["scoring_version"] = "corregido"
    return estado


def entrenar(
    cfg: dict,
    rutas: dict,
    tokenizer,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> dict:
    """Entrena el adaptador LoRA del juez C2 (modo --entrenar / --smoke).

    Args:
        cfg: Configuracion efectiva.
        rutas: Rutas absolutas.
        tokenizer: Tokenizer de ALIA.
        args: Namespace de la CLI.
        logger: Logger activo.

    Returns:
        Dict de cierre del entrenamiento.

    Raises:
        RuntimeError: Si no hay GPU, VRAM insuficiente o dataset faltante.
    """
    device = "cuda"
    _, total_gib = _verificar_gpu_exclusiva(cfg, logger)
    info_modelo = verificar_modelo_base(cfg, rutas, logger)

    rutas_sft = {
        "train": Path(rutas["dir_sft"]) / "train.parquet",
        "val": Path(rutas["dir_sft"]) / "val.parquet",
        "val_eval": Path(rutas["dir_sft"]) / "val_eval.parquet",
    }
    for ruta in rutas_sft.values():
        _exigir(ruta.exists(), f"falta {ruta} (ejecute --preparar-datos primero)")
    datos_train = cargar_sft(rutas_sft["train"])
    datos_val = cargar_sft(rutas_sft["val"])
    datos_val_eval = cargar_sft(rutas_sft["val_eval"])
    ejemplos_train = preparar_ejemplos(datos_train, cfg["max_seq"])
    ejemplos_val = preparar_ejemplos(datos_val, cfg["max_seq"])
    ejemplos_val_eval = preparar_ejemplos(datos_val_eval, cfg["max_seq"])
    if cfg["max_ejemplos"]:
        ejemplos_train = ejemplos_train[: cfg["max_ejemplos"]]
    logger.info(
        "datos: train=%d val=%d val_eval=%d (max_seq=%d, max_ejemplos=%s)",
        len(ejemplos_train),
        len(ejemplos_val),
        len(ejemplos_val_eval),
        cfg["max_seq"],
        cfg["max_ejemplos"],
    )

    pasos_por_epoca = math.ceil(len(ejemplos_train) / (cfg["batch"] * cfg["grad_accum"]))
    total_steps = pasos_por_epoca * cfg["epocas"]
    if cfg["max_pasos"]:
        total_steps = min(total_steps, cfg["max_pasos"])
    _exigir(total_steps >= 1, "no hay pasos de entrenamiento (revisar --max-ejemplos)")

    dir_reanudar = rutas["dir_checkpoint"] if args.reanudar else None
    modelo = cargar_modelo_lora(cfg, rutas, device, dir_reanudar, logger)
    parametros = [p for p in modelo.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parametros,
        lr=cfg["lr"],
        betas=tuple(cfg["betas"]),
        eps=cfg["eps"],
        weight_decay=cfg["weight_decay"],
    )
    estado = {
        "paso": 0,
        "epoca": 0,
        "mejor_auc": None,
        "mejor_paso": 0,
        "sin_mejora": 0,
        "total_steps": total_steps,
        "pasos_por_epoca": pasos_por_epoca,
        "tokens": 0,
        "historial": [],
        "scoring_version": "corregido",
        "config": {
            "lora_r": cfg["lora_r"],
            "lora_alpha": cfg["lora_alpha"],
            "lr": cfg["lr"],
            "max_seq": cfg["max_seq"],
            "batch": cfg["batch"],
            "grad_accum": cfg["grad_accum"],
            "target_modules": list(cfg["target_modules"]),
        },
    }
    if args.reanudar:
        estado = cargar_checkpoint(modelo, optimizer, rutas, device, logger)
        for clave in ("lora_r", "lora_alpha", "max_seq", "batch", "grad_accum", "target_modules"):
            guardado = estado["config"].get(clave)
            if guardado is not None:
                _exigir(
                    guardado == cfg[clave],
                    f"config del checkpoint difiere en {clave}: {guardado} != {cfg[clave]}",
                )
        if estado["config"]["lr"] != cfg["lr"]:
            logger.warning(
                "LR del checkpoint (%g) prevalece sobre --lr (%g) para mantener el schedule",
                estado["config"]["lr"],
                cfg["lr"],
            )
            cfg["lr"] = estado["config"]["lr"]
        # Migracion de metricas: los checkpoints previos al fix de scoring
        # (padding derecho) guardaron evals invalidos; se etiquetan como
        # "bug_padding" y se reinician mejor_auc/sin_mejora para no comparar
        # metricas incompatibles (el primer eval corregido define el mejor).
        estado = migrar_metricas_scoring(estado, logger)
        total_steps = estado["total_steps"]
        pasos_por_epoca = estado["pasos_por_epoca"]
    historial = estado["historial"]

    inicio = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    modelo.train()
    parada = "completado"
    while estado["epoca"] < cfg["epocas"]:
        epoca = estado["epoca"]
        orden = list(range(len(ejemplos_train)))
        random.Random(f"{cfg['semilla']}:c2lora:epoca:{epoca}").shuffle(orden)
        paso_en_epoca = estado["paso"] - epoca * pasos_por_epoca
        inicio_micro = paso_en_epoca * cfg["grad_accum"]
        micro_contador = inicio_micro
        perdidas_ventana: list[float] = []
        for inicio_idx in range(inicio_micro, len(orden), cfg["batch"]):
            indices_lote = orden[inicio_idx : inicio_idx + cfg["batch"]]
            lote = collar_lote([ejemplos_train[i] for i in indices_lote], tokenizer.pad_token_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                perdida, _ = forward_completion(modelo, lote, device)
            (perdida / cfg["grad_accum"]).backward()
            estado["tokens"] += sum(
                len(ejemplos_train[i]["prompt_ids"]) + len(ejemplos_train[i]["resp_ids"])
                for i in indices_lote
            )
            perdidas_ventana.append(float(perdida.detach()) * cfg["grad_accum"])
            micro_contador += 1
            fin_ciclo = (micro_contador % cfg["grad_accum"] == 0) or (
                inicio_idx + cfg["batch"] >= len(orden)
            )
            if not fin_ciclo:
                continue
            torch.nn.utils.clip_grad_norm_(parametros, cfg["grad_clip"])
            estado["paso"] += 1
            lr = lr_en_paso(
                estado["paso"], total_steps, cfg["lr"], cfg["warmup_frac"], cfg["min_lr_frac"]
            )
            for grupo in optimizer.param_groups:
                grupo["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss_media = sum(perdidas_ventana) / len(perdidas_ventana)
            perdidas_ventana = []
            _exigir(
                math.isfinite(loss_media),
                f"loss no finita en el paso {estado['paso']} (NaN/Inf): parar y diagnosticar "
                "(ladder del plan: reducir lr a 5e-5)",
            )
            if estado["paso"] % cfg["checkpoint_cada"] == 0:
                guardar_checkpoint(modelo, optimizer, estado, rutas, logger)
            if estado["paso"] % cfg["eval_cada"] == 0:
                metricas = evaluar(modelo, ejemplos_val_eval, cfg, tokenizer, device, logger)
                registro = {
                    "paso": estado["paso"],
                    "epoca": epoca,
                    "lr": float(lr),
                    "loss_train": float(loss_media),
                    "scoring": "corregido",
                    **metricas,
                }
                historial.append(registro)
                logger.info(
                    "eval paso %d: loss=%.4f acc=%.4f f1_si=%.4f auc=%s matriz=%s",
                    estado["paso"],
                    metricas["loss"] if metricas["loss"] is not None else -1.0,
                    metricas["acc"],
                    metricas["f1_si"],
                    f"{metricas['auc']:.4f}" if metricas["auc"] is not None else "n/a",
                    metricas["matriz"],
                )
                auc = metricas["auc"]
                if auc is not None and (estado["mejor_auc"] is None or auc > estado["mejor_auc"] + 1e-6):
                    estado["mejor_auc"] = float(auc)
                    estado["mejor_paso"] = estado["paso"]
                    estado["sin_mejora"] = 0
                    modelo.save_pretrained(rutas["dir_adapter"])
                    with open(Path(rutas["dir_adapter"]) / "mejor.json", "w", encoding="utf-8") as manejador:
                        json.dump(
                            {"paso": estado["paso"], "auc": float(auc), "acc": metricas["acc"]},
                            manejador,
                            ensure_ascii=False,
                            indent=2,
                        )
                    logger.info("nuevo mejor AUC=%.4f (paso %d): adapter guardado", auc, estado["paso"])
                else:
                    estado["sin_mejora"] += 1
                if estado["sin_mejora"] >= cfg["early_stop_patience"]:
                    parada = "early_stopping"
                    logger.info("early stopping: %d evals sin mejora", estado["sin_mejora"])
                    break
                if estado["paso"] >= cfg["paso_diagnostico"] and (
                    metricas["auc"] is None or metricas["auc"] < cfg["auc_diagnostico"]
                ):
                    parada = "diagnostico_auc_baja"
                    logger.warning(
                        "diagnostico: AUC val < %.2f tras %d pasos; parada para diagnostico",
                        cfg["auc_diagnostico"],
                        estado["paso"],
                    )
                    break
            if cfg["max_pasos"] and estado["paso"] >= cfg["max_pasos"]:
                parada = "max_pasos"
                logger.info("max_pasos alcanzado (%d)", cfg["max_pasos"])
                break
            if cfg["max_horas"] and (time.perf_counter() - inicio) / 3600.0 > cfg["max_horas"]:
                parada = "max_horas"
                logger.info(
                    "presupuesto de %.2f h agotado en el paso %d; reanudable con --reanudar",
                    cfg["max_horas"],
                    estado["paso"],
                )
                break
        if parada in ("early_stopping", "diagnostico_auc_baja", "max_pasos", "max_horas"):
            break
        estado["epoca"] += 1
        guardar_checkpoint(modelo, optimizer, estado, rutas, logger)

    duracion = time.perf_counter() - inicio
    guardar_checkpoint(modelo, optimizer, estado, rutas, logger)
    if estado["mejor_auc"] is None:
        modelo.save_pretrained(rutas["dir_adapter"])
        logger.warning("sin mejor AUC registrado: adapter final guardado como mejor")
    metricas_finales = evaluar(
        modelo,
        ejemplos_val if not cfg["smoke"] else ejemplos_val_eval,
        cfg,
        tokenizer,
        device,
        logger,
    )
    logger.info(
        "eval final (%s): acc=%.4f f1_si=%.4f auc=%s",
        "val" if not cfg["smoke"] else "val_eval",
        metricas_finales["acc"],
        metricas_finales["f1_si"],
        f"{metricas_finales['auc']:.4f}" if metricas_finales["auc"] is not None else "n/a",
    )
    dibujar_curvas(historial, rutas["ruta_figura_curvas"])
    vram_pico = torch.cuda.max_memory_allocated() / 1024**3
    tokens_por_segundo = estado["tokens"] / duracion if duracion > 0 else 0.0
    cierre = {
        "version": cfg["version_entrenamiento"],
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "modo": "smoke" if cfg["smoke"] else "completo",
        "config": {
            clave: cfg[clave]
            for clave in (
                "semilla",
                "max_seq",
                "lora_r",
                "lora_alpha",
                "lora_dropout",
                "target_modules",
                "lr",
                "betas",
                "eps",
                "weight_decay",
                "batch",
                "grad_accum",
                "epocas",
                "warmup_frac",
                "min_lr_frac",
                "grad_clip",
                "eval_cada",
                "checkpoint_cada",
                "early_stop_patience",
                "max_ejemplos",
                "max_horas",
                "max_pasos",
            )
        },
        "peft": {
            "r": cfg["lora_r"],
            "alpha": cfg["lora_alpha"],
            "dropout": cfg["lora_dropout"],
            "target_modules": list(cfg["target_modules"]),
        },
        "dataset": {
            "dir_sft": rutas["dir_sft"],
            "n_train": len(ejemplos_train),
            "n_val": len(ejemplos_val),
            "n_val_eval": len(ejemplos_val_eval),
            "md5": {
                nombre: md5_archivo(ruta) for nombre, ruta in rutas_sft.items()
            },
        },
        "modelo_base": info_modelo,
        "entorno": versiones_entorno(),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "vram_total_gib": total_gib,
            "vram_pico_gib": vram_pico,
        },
        "entrenamiento": {
            "pasos": estado["paso"],
            "epocas_completadas": estado["epoca"],
            "parada": parada,
            "mejor_paso": estado["mejor_paso"],
            "mejor_auc": estado["mejor_auc"],
            "sin_mejora": estado["sin_mejora"],
            "total_steps_plan": total_steps,
            "scoring": "corregido (padding izquierdo en evaluar; evals con etiqueta bug_padding en curvas son del scoring antiguo)",
        },
        "curvas": historial,
        "eval_final": metricas_finales,
        "tiempos": {
            "segundos": duracion,
            "horas": duracion / 3600.0,
            "tokens_procesados": estado["tokens"],
            "tokens_por_segundo": tokens_por_segundo,
        },
        "salidas": {
            "checkpoint": rutas["dir_checkpoint"],
            "adapter": rutas["dir_adapter"],
            "json": rutas["ruta_train_json"],
            "figura": rutas["ruta_figura_curvas"],
        },
    }
    destino_json = Path(rutas["ruta_train_json"])
    destino_json.parent.mkdir(parents=True, exist_ok=True)
    with open(destino_json, "w", encoding="utf-8") as manejador:
        json.dump(cierre, manejador, ensure_ascii=False, indent=2)
    logger.info(
        "cierre: pasos=%d parada=%s mejor_auc=%s vram_pico=%.2f GiB tok/s=%.0f | json=%s",
        estado["paso"],
        parada,
        f"{estado['mejor_auc']:.4f}" if estado["mejor_auc"] is not None else "n/a",
        vram_pico,
        tokens_por_segundo,
        rutas["ruta_train_json"],
    )
    return cierre


# ===== SECCION 6: CLI y main =====


def parsear_args() -> argparse.Namespace:
    """Parsea la linea de comandos.

    Returns:
        Namespace con los argumentos validados.

    Raises:
        SystemExit: Si no se elige exactamente un modo.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--preparar-datos",
        "--dataset",
        dest="preparar_datos",
        action="store_true",
        help="construye el dataset SFT (CPU, sin cargar el modelo 7B)",
    )
    parser.add_argument("--entrenar", action="store_true", help="entrena el LoRA (GPU exclusiva)")
    parser.add_argument("--todo", action="store_true", help="preparar-datos + entrenar")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="mini-dataset en /tmp/opencode + 20 pasos (gate de VRAM/throughput)",
    )
    parser.add_argument(
        "--evaluar",
        action="store_true",
        help="evalua un adaptador guardado en val con la ruta de scoring corregida",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="ruta del adaptador para --evaluar (default modelos/c2_lora/adapter_v1)",
    )
    parser.add_argument(
        "--sin-adaptador",
        action="store_true",
        help="control de --evaluar: evalua el modelo base bf16 sin adaptador",
    )
    parser.add_argument("--max-items", type=int, default=None, help="tope de items de val en --evaluar")
    parser.add_argument(
        "--salida",
        type=str,
        default=None,
        help="JSON de salida de --evaluar (default resultados/c2_lora/eval_interna_v1.json)",
    )
    parser.add_argument("--forzar", action="store_true", help="permite reescribir salidas existentes")
    parser.add_argument("--reanudar", action="store_true", help="reanuda desde el ultimo checkpoint")
    parser.add_argument("--max-ejemplos", type=int, default=None, help="tope de ejemplos de train")
    parser.add_argument("--max-horas", type=float, default=None, help="presupuesto de tiempo del run")
    parser.add_argument("--max-pasos", type=int, default=None, help="tope de pasos de optimizador")
    parser.add_argument("--max-seq", type=int, default=None, help="longitud maxima (ladder 768->512)")
    parser.add_argument("--lr", type=float, default=None, help="learning rate base")
    parser.add_argument("--r", type=int, default=None, help="rango LoRA (alpha = 2*r)")
    parser.add_argument("--epocas", type=int, default=None, help="numero de epocas (default 1, max 2)")
    parser.add_argument(
        "--target-modules",
        type=str,
        default=None,
        help="modulos LoRA separados por coma (ladder OOM: q_proj,k_proj,v_proj,o_proj)",
    )
    parser.add_argument("--eval-cada", type=int, default=None, help="eval periodica cada N pasos")
    parser.add_argument("--checkpoint-cada", type=int, default=None, help="checkpoint cada N pasos")
    args = parser.parse_args()
    modos = [args.preparar_datos, args.entrenar, args.todo, args.smoke, args.evaluar]
    if sum(bool(m) for m in modos) != 1:
        parser.error(
            "se requiere exactamente un modo: --preparar-datos, --entrenar, --todo, --smoke o --evaluar"
        )
    if args.smoke and args.reanudar:
        parser.error("--smoke no admite --reanudar")
    if args.evaluar and args.reanudar:
        parser.error("--evaluar no admite --reanudar")
    return args


def resolver_config(args: argparse.Namespace) -> dict:
    """Resuelve la configuracion efectiva (overrides CLI + rutas smoke).

    Args:
        args: Namespace de la CLI.

    Returns:
        Configuracion efectiva (dict).

    Raises:
        SystemExit: Si un valor queda fuera de rango.
    """
    cfg = copy.deepcopy(CONFIG)
    cfg["smoke"] = bool(args.smoke)
    if args.max_seq is not None:
        if args.max_seq < 64:
            raise SystemExit("--max-seq debe ser >= 64")
        cfg["max_seq"] = args.max_seq
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.r is not None:
        if args.r < 1:
            raise SystemExit("--r debe ser >= 1")
        cfg["lora_r"] = args.r
        cfg["lora_alpha"] = 2 * args.r
    if args.target_modules is not None:
        modulos = [m.strip() for m in args.target_modules.split(",") if m.strip()]
        if not modulos:
            raise SystemExit("--target-modules no puede estar vacio")
        cfg["target_modules"] = modulos
    if args.epocas is not None:
        if not 1 <= args.epocas <= cfg["epocas_max"]:
            raise SystemExit("--epocas debe estar en [1, 2] (spec: 1 epoca, max 2)")
        cfg["epocas"] = args.epocas
    if args.eval_cada is not None:
        cfg["eval_cada"] = args.eval_cada
    if args.checkpoint_cada is not None:
        cfg["checkpoint_cada"] = args.checkpoint_cada
    if args.max_ejemplos is not None:
        cfg["max_ejemplos"] = args.max_ejemplos
    if args.max_horas is not None:
        cfg["max_horas"] = args.max_horas
    if args.max_pasos is not None:
        cfg["max_pasos"] = args.max_pasos
    if args.max_items is not None and args.max_items < 1:
        raise SystemExit("--max-items debe ser >= 1")
    if args.evaluar and args.sin_adaptador and args.adapter is not None:
        raise SystemExit("--evaluar admite --adapter o --sin-adaptador, no ambos")
    if args.smoke:
        base = CONFIG["smoke"]["dir_salida"]
        cfg["epocas"] = CONFIG["smoke"]["epocas"]
        cfg["max_ejemplos"] = args.max_ejemplos if args.max_ejemplos is not None else CONFIG["smoke"]["max_ejemplos"]
        cfg["max_pasos"] = args.max_pasos if args.max_pasos is not None else CONFIG["smoke"]["max_pasos"]
        cfg["eval_cada"] = args.eval_cada if args.eval_cada is not None else CONFIG["smoke"]["eval_cada"]
        cfg["checkpoint_cada"] = (
            args.checkpoint_cada if args.checkpoint_cada is not None else CONFIG["smoke"]["checkpoint_cada"]
        )
        for clave, valor in CONFIG["smoke"]["caps"].items():
            cfg[clave] = valor
        cfg["dir_sft"] = f"{base}/sft_v1"
        cfg["ruta_dataset_json"] = f"{base}/dataset_smoke.json"
        cfg["ruta_figura_dataset"] = f"{base}/figura_dataset_smoke.png"
        cfg["dir_modelos"] = f"{base}/modelos"
        cfg["dir_checkpoint"] = f"{base}/checkpoints/ultimo"
        cfg["dir_adapter"] = f"{base}/adapter"
        cfg["ruta_train_json"] = f"{base}/entrenamiento_smoke.json"
        cfg["ruta_figura_curvas"] = f"{base}/curvas_smoke.png"
        cfg["ruta_log"] = f"{base}/smoke.log"
    cfg["rutas"] = resolver_rutas(cfg)
    return cfg


def main() -> int:
    """Punto de entrada: despacha los modos preparar-datos/entrenar/smoke/evaluar.

    Returns:
        Codigo de salida (0 si todo va bien).

    Raises:
        RuntimeError: Errores de integridad o de ejecucion (via _exigir).
    """
    args = parsear_args()
    cfg = resolver_config(args)
    rutas = cfg["rutas"]
    logger = configurar_logger(rutas["ruta_log"])
    logger.info("=" * 80)
    logger.info("33_c2_lora_train.py | modo=%s | smoke=%s", vars(args), cfg["smoke"])
    fijar_semillas(cfg["semilla"], sembrar_cuda=not args.preparar_datos)
    tokenizer = cargar_tokenizer(rutas)

    if args.preparar_datos or args.todo or args.smoke:
        preparar_datos(cfg, rutas, tokenizer, logger, forzar=args.forzar, reusar=args.smoke)
    if args.entrenar or args.todo or args.smoke:
        ids_ctx = ids_respuesta_contexto(tokenizer, cfg, logger)
        cfg["ids_respuesta"] = ids_ctx
        entrenar(cfg, rutas, tokenizer, args, logger)
    if args.evaluar:
        evaluar_adaptador(cfg, rutas, tokenizer, args, logger)
    logger.info("fin del script")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 - protocolo: traza completa y salida 1 ante cualquier fallo
        traceback.print_exc()
        sys.exit(1)
