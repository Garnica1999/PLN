"""34_c2_lora_eval.py
Fecha: 2026-10-05
Descripcion: Frente C2 del ciclo v3 - evaluacion del juez binario punto a
    punto entrenado con LoRA en scripts/33_c2_lora_train.py ("el fragmento
    cita el articulo N de la CP?" -> "Si"/"No", score P("Si") por logits en
    un forward, sin generar). Un script completo con los tres niveles del
    plan (PLAN_c2_lora.md seccion 4) y modos --nivel 1|2|3|todo:

    Nivel 1 - val interna (E1): accuracy/F1/AUC sobre el val del dataset
      (data/c2_lora/sft_v1/val.parquet, 3.000 pares, docs de B1) y umbral de
      decision optimizado SOLO con val_eval (subset de val). Controles:
      adaptador vs base bf16 (disable_adapter) vs el eval_final registrado
      por el entrenamiento (resultados/c2_lora/entrenamiento_v1.json).
      Criterio S2: acc >= 0,78, AUC >= 0,85 y delta acc vs base >= +0,15
      con IC95 > 0 (bootstrap pareado por documento).

    Nivel 2 - gold humano B3 (E2): 820 items / 50 casos de
      resultados/gold_mini (respuestas_carlos.jsonl, grados 2/1/0). Fragmento
      = encabezado/extracto del caso (muestra_gold.jsonl); articulo = texto de
      data/constitucion_articulos.parquet. Metricas: AUC(grado >= 1),
      AUC(grado 2), Spearman con el grado, exactitud binaria, nDCG@10
      graduado por caso (ganancia 2^g - 1, maquinaria del script 16) y
      comparacion pareada por caso vs base bf16. Referencias: preanotaciones
      respuestas_llm*.jsonl y acuerdo carlos|llm. Criterio S3: AUC(>=1) >=
      0,75, Spearman >= 0,30 y delta AUC vs base >= +0,10 (IC95 > 0).

    Nivel 3 - rerank end-to-end (E3): el score del juez reordena candidatos
      del benchmark M7 (90 casos, referencia silver arts_cp NER). Fuente de
      candidatos por defecto = top-10 congelado de la ruta v13 en
      resultados/vinculo_cp_benchmark_v13.json (los JSON del benchmark NO
      guardan el top-50: limitacion documentada; se reordena el top-10 con
      pares (chunk, pasaje) reconstruidos por la maquinaria BM25 del script
      12, idénticos a la inferencia B1). Con --recomputar-top50 se recalcula
      la primera etapa completa (hibrido 0.5*BM25 + 0.5*B2, rerank B1 por
      chunks, max-pooling; GPU) y se evaluan las rutas del plan:
      c2_lora_top50, c2_lora_cascada_b1_top20 y c2_lora_fusion_b1
      (alpha 0,5 primario; 0,25/0,75 como sensibilidad). Comparaciones
      pareadas (bootstrap 1000/42) vs v13 (per-case congelado en
      comparacion_b2.metricas_por_caso), vs el control zero-shot bf16
      (disable_adapter, mismos prompts) y vs el C2 zero-shot documentado
      (resultados/c2_llm/eval_rerank_llm.json); consistencia A/B por mitades
      y gate de latencia p95 <= 30 s. Criterio S4: LoRA >= zero-shot bf16
      + 0,05 nDCG@10 (IC95 > 0); adopcion solo si cascada/fusion > v13
      (0,6907) con delta >= +0,01, IC95 > 0 y A/B consistente.

    Modo --autotest (CPU, sin modelo 7B): valida la carga de los 820 items
    B3 + grados, los joins con articulos/casos, la construccion de pares
    (fragmento, articulo) con el formato exacto del 33, las metricas con
    datos sinteticos y los insumos del benchmark (top-10 y referencias
    congeladas). Modo --smoke: dry-run completo de los tres niveles con un
    LLaMA diminuto aleatorio + peft en /tmp/opencode (mismo flujo, caps
    pequenos); sirve de gate antes de correr con el modelo real.

    Cache de scores por md5(prompt ids) en
    resultados/c2_lora/cache_scores_v1.jsonl (reutiliza E2/E3 entre corridas
    y checkpoints). El adaptador se carga con PeftModel.from_pretrained sin
    merge (no duplica los 15,5 GB); --checkpoint permite evaluar un
    checkpoint intermedio de modelos/c2_lora/checkpoints/.

    Anti-leakage: jamas se ajusta nada con test; el umbral se optimiza con
    val_eval; el benchmark y el gold B3 son holdout y solo se leen; se
    verifica que ningun doc de val pertenezca a las 90 del benchmark y que
    los 50 casos del gold sean subconjunto del benchmark.

    Estado del arte aplicado:
    - Evaluacion por logits sin generacion (patron de rerank LegalMALR
      arXiv:2601.17692; COLIEE arXiv:2607.11400).
    - Bootstrap no parametrico de la media y PAREADO por caso/documento
      (Efron 1979; protocolo del proyecto ADR-16, 1000 remuestreos seed 42);
      en B3 el remuestreo es por caso (cluster bootstrap: los items de un
      caso no son independientes).
    - Optimizacion de umbral con val (Youden 1950 / F1-max; skill
      threshold-optimization), nunca con test.
    - Relevancia graduada y nDCG con ganancia exponencial (Jarvelin y
      Kekalainen 2002) reutilizando la maquinaria del script 16.
    - Control zero-shot bf16 con disable_adapter (comparacion bf16-vs-bf16
      sin el confound de cuantizacion del GGUF Q4 congelado; PLAN seccion 5).

    Reglas: solo lectura de datasets/benchmark/gold; nada se sobrescribe sin
    --forzar (el JSON de salida se fusiona por nivel); figuras y JSON con el
    nombre del experimento; sin emojis.

Uso:
    # Validacion CPU (no toca la GPU ocupada por el entrenamiento):
    /home/carlos/venv/bin/python scripts/34_c2_lora_eval.py --autotest
    /home/carlos/venv/bin/python scripts/34_c2_lora_eval.py --smoke --dispositivo cpu

    # Niveles reales (tras el entrenamiento, GPU libre):
    /home/carlos/venv/bin/python scripts/34_c2_lora_eval.py --nivel 1
    /home/carlos/venv/bin/python scripts/34_c2_lora_eval.py --nivel 2
    /home/carlos/venv/bin/python scripts/34_c2_lora_eval.py --nivel 3
    /home/carlos/venv/bin/python scripts/34_c2_lora_eval.py --nivel todo
    /home/carlos/venv/bin/python scripts/34_c2_lora_eval.py --nivel 3 --recomputar-top50
    /home/carlos/venv/bin/python scripts/34_c2_lora_eval.py --nivel 1 \\
        --checkpoint modelos/c2_lora/checkpoints/ultimo
"""

import argparse
import copy
import importlib.util
import json
import logging
import os
import pickle
import random
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

# Configuracion de allocator ANTES de importar torch (receta de memoria del
# plan, identica al script 33).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from transformers import AutoModelForCausalLM

# ===== SECCION 1: configuracion =====

RAIZ = Path(__file__).resolve().parent.parent

CONFIG = {
    "version": "c2-lora-eval-v1",
    "semilla": 42,
    # --- dataset SFT (script 33) ---
    "dir_sft": "data/c2_lora/sft_v1",
    "ruta_dataset_json": "resultados/c2_lora/dataset_v1.json",
    "ruta_train_json": "resultados/c2_lora/entrenamiento_v1.json",
    # --- modelo ---
    "ruta_modelo": "modelos/alia_7b_instruct",
    "dir_adapter": "modelos/c2_lora/adapter_v1",
    "dir_checkpoints": "modelos/c2_lora/checkpoints",
    "attn_implementation": "sdpa",
    "vram_min_gib_eval": 15.0,
    # --- gold humano B3 (script 16) ---
    "dir_gold": "resultados/gold_mini",
    "ruta_gold_respuestas": "resultados/gold_mini/respuestas_carlos.jsonl",
    "ruta_gold_muestra": "resultados/gold_mini/muestra_gold.jsonl",
    "ruta_gold_candidatos": "resultados/gold_mini/candidatos.jsonl",
    "ruta_gold_evaluacion": "resultados/gold_mini/evaluacion_gold_mini.json",
    "ruta_gold_acuerdo": "resultados/gold_mini/acuerdo_gold_mini.json",
    "rutas_gold_llm": [
        "resultados/gold_mini/respuestas_llm.jsonl",
        "resultados/gold_mini/respuestas_llm_gemma4.jsonl",
    ],
    "ruta_articulos": "data/constitucion_articulos.parquet",
    # --- benchmark M7 ---
    "ruta_benchmark_v2": "resultados/vinculo_cp_benchmark_v2.json",
    "rutas_benchmark": {
        "v13": "resultados/vinculo_cp_benchmark_v13.json",
        "v14_ep2": "resultados/vinculo_cp_benchmark_v14_ep2.json",
    },
    "ruta_corpus": "data/corpus_sentencias.parquet",
    "ruta_eval_rerank_llm": "resultados/c2_llm/eval_rerank_llm.json",
    "ruta_metricas": "resultados/metricas.json",
    "estrategia_v13": "hibrido_b2_rerank_chunks_top50_b1",
    "contexto_v13": "con_hubs",
    "alpha_hubs": 1.5,
    # --- plantilla EXACTA del script 33 (se verifica contra mod33 en autotest) ---
    "system": (
        "Eres un asistente jurídico experto en derecho constitucional colombiano. "
        "Respondes únicamente 'Sí' o 'No'."
    ),
    "plantilla_user": (
        "FRAGMENTO DE SENTENCIA:\n{chunk}\n\n"
        "ARTÍCULO {art} DE LA CONSTITUCIÓN:\n{pasaje}\n\n"
        "¿El fragmento cita el artículo {art}? Responde solo \"Sí\" o \"No\"."
    ),
    # --- evaluacion ---
    "max_seq": 768,
    "batch": 4,
    "max_chars_fragmento_gold": 1500,
    "max_chars_pasaje": 800,
    "max_chars_chunk_benchmark": 600,
    "umbral": 0.5,
    "criterio_umbral": "youden",
    "bootstrap": 1000,
    "alphas_fusion": [0.25, 0.5, 0.75],
    "alpha_fusion_primario": 0.5,
    "k_candidatos_top50": 50,
    "k_candidatos_cascada": 20,
    "gate_p95_seg": 30.0,
    "k_salida": 10,
    "max_intentos_bootstrap": 20,
    # --- criterios de exito / parada (PLAN seccion 4.3) ---
    "criterios_s2": {"acc": 0.78, "auc": 0.85, "delta_acc_vs_base": 0.15},
    "criterios_s3": {"auc_ge1": 0.75, "spearman": 0.30, "delta_auc_vs_base": 0.10},
    "criterios_s4": {
        "delta_ndcg_vs_base": 0.05,
        "delta_ndcg_adopcion_vs_v13": 0.01,
        "delta_gold_ndcg_vs_v13_min": -0.005,
        "gate_p95_seg": 30.0,
    },
    "referencias_c2": {
        "v13_ndcg_at_10_sin_hubs": 0.6926,
        "v13_ndcg_at_10_con_hubs": 0.6907,
        "c2_rerank_llm_ndcg_at_10": 0.473,
    },
    # --- salidas ---
    "ruta_salida": "resultados/c2_lora/eval_c2_lora_v1.json",
    "ruta_cache": "resultados/c2_lora/cache_scores_v1.jsonl",
    "ruta_candidatos_top50": "resultados/c2_lora/candidatos_top50_v1.pkl",
    "ruta_log": "experimentos/34_c2_lora_eval.log",
    "ruta_figura_val": "figuras/c2_lora_eval_val.png",
    "ruta_figura_gold": "figuras/fig_c2_lora_gold.png",
    "ruta_figura_benchmark": "figuras/fig_c2_lora_eval_benchmark.png",
    # --- smoke: caps y salidas SOLO en /tmp/opencode ---
    "smoke": {
        "dir_salida": "/tmp/opencode/c2_lora_eval_smoke",
        "max_items_val": 8,
        "max_casos_gold": 2,
        "max_casos_benchmark": 2,
        "ocultos": 64,
        "capas": 2,
        "cabezas": 4,
        "intermedio": 128,
        "batch": 2,
        "max_seq": 256,
        "lora_r": 4,
        "lora_alpha": 8,
    },
}

# Claves de CONFIG que son rutas relativas a la raiz del proyecto.
CLAVES_RUTA = (
    "dir_sft",
    "ruta_dataset_json",
    "ruta_train_json",
    "ruta_modelo",
    "dir_adapter",
    "dir_checkpoints",
    "dir_gold",
    "ruta_gold_respuestas",
    "ruta_gold_muestra",
    "ruta_gold_candidatos",
    "ruta_gold_evaluacion",
    "ruta_gold_acuerdo",
    "ruta_articulos",
    "ruta_benchmark_v2",
    "ruta_corpus",
    "ruta_eval_rerank_llm",
    "ruta_metricas",
    "ruta_salida",
    "ruta_cache",
    "ruta_candidatos_top50",
    "ruta_log",
    "ruta_figura_val",
    "ruta_figura_gold",
    "ruta_figura_benchmark",
)

# Nota de limitacion del nivel 3 (se serializa en el JSON de salida).
LIMITACION_TOP50 = (
    "Los JSON del benchmark (v13/v14_ep2) solo guardan el top-10 por estrategia "
    "(top10_por_caso) y las metricas por caso congeladas; el top-50 de la primera "
    "etapa NO es reconstruible desde los artefactos guardados. La ruta por defecto "
    "reordena el top-10 con pares (chunk, pasaje) reconstruidos por la maquinaria "
    "BM25 del script 12 (identicos a la inferencia B1): mide capacidad de reordenar, "
    "no de recuperar articulos fuera del top-10. El top-50 completo (c2_lora_top50, "
    "cascada B1 top-20 y fusion sobre 50 candidatos) requiere --recomputar-top50 "
    "(GPU: B2 + reranker B1)."
)


# ===== SECCION 2: utilidades (logger, importlib, md5, guardas, semillas) =====

LOGGER_NOMBRE = "c2_lora_eval_34"
LOGGER = logging.getLogger(LOGGER_NOMBRE)
_MODULOS: dict[str, object] = {}


def _cargar_modulo(nombre: str, ruta: Path):
    """Importa un script por ruta (su nombre empieza por digito).

    Patron de importlib cacheado del script 33 (sin duplicar logica).

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
        Modulo 12 (pasajes de la CP, BM25, hibrido, bootstrap, hubs).
    """
    return _cargar_modulo("evaluar_vinculo_cp_v2_12", RAIZ / "scripts/12_evaluar_vinculo_cp_v2.py")


def modulo_16():
    """Carga (y cachea) scripts/16_anotador_gold.py.

    Returns:
        Modulo 16 (gold graduado, nDCG graduado, catalogo de articulos).
    """
    return _cargar_modulo("anotador_gold_16", RAIZ / "scripts/16_anotador_gold.py")


def modulo_17():
    """Carga (y cachea) scripts/17_auditoria_etiquetas_m4.py.

    Returns:
        Modulo 17 (md5 y versiones de entorno).
    """
    return _cargar_modulo("auditoria_etiquetas_m4_17", RAIZ / "scripts/17_auditoria_etiquetas_m4.py")


def modulo_21():
    """Carga (y cachea) scripts/21_pack_validacion_m4.py.

    Returns:
        Modulo 21 (guardas de no-sobrescritura).
    """
    return _cargar_modulo("pack_validacion_m4_21", RAIZ / "scripts/21_pack_validacion_m4.py")


def modulo_30():
    """Carga (y cachea) scripts/30_c2_llm_query.py.

    Returns:
        Modulo 30 (casos de ruta v13, BM25/denso/rerank reutilizables).
    """
    return _cargar_modulo("c2_llm_query_30", RAIZ / "scripts/30_c2_llm_query.py")


def modulo_33():
    """Carga (y cachea) scripts/33_c2_lora_train.py.

    Returns:
        Modulo 33 (cargar_sft, preparar_ejemplos, chat template, ids).
    """
    return _cargar_modulo("c2_lora_train_33", RAIZ / "scripts/33_c2_lora_train.py")


def md5_archivo(ruta: str | Path) -> str | None:
    """Calcula el md5 de un archivo por bloques (delegado al modulo 17).

    Args:
        ruta: Ruta del archivo.

    Returns:
        Hexdigest md5, o None si el archivo no existe.
    """
    return modulo_17().md5_archivo(str(ruta))


def versiones_entorno() -> dict:
    """Reune versiones del entorno mas las del stack ML/DL (script 33).

    Returns:
        Dict con python, plataforma y versiones de librerias clave.
    """
    return modulo_33().versiones_entorno()


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


def fijar_semillas(semilla: int) -> None:
    """Fija las semillas de random, numpy y torch (CPU y CUDA).

    Args:
        semilla: Semilla entera.
    """
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def resolver_rutas(cfg: dict) -> dict:
    """Convierte las rutas de CONFIG en absolutas contra RAIZ.

    Args:
        cfg: Configuracion efectiva (con overrides ya aplicados).

    Returns:
        Dict clave -> ruta absoluta (str); rutas_benchmark se resuelve
        como dict de variantes.
    """
    rutas = {
        clave: cfg[clave] if Path(cfg[clave]).is_absolute() else str(RAIZ / cfg[clave])
        for clave in CLAVES_RUTA
    }
    rutas["rutas_benchmark"] = {
        variante: (ruta if Path(ruta).is_absolute() else str(RAIZ / ruta))
        for variante, ruta in cfg["rutas_benchmark"].items()
    }
    rutas["rutas_gold_llm"] = [
        ruta if Path(ruta).is_absolute() else str(RAIZ / ruta) for ruta in cfg["rutas_gold_llm"]
    ]
    return rutas


def leer_jsonl(ruta: str | Path) -> list[dict]:
    """Carga un JSON Lines linea por linea.

    Args:
        ruta: Ruta del archivo.

    Returns:
        Lista de registros.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si alguna linea no es JSON valido.
    """
    archivo = Path(ruta)
    if not archivo.exists():
        raise FileNotFoundError(f"no existe {archivo}")
    registros = []
    with open(archivo, encoding="utf-8") as manejador:
        for numero, linea in enumerate(manejador, 1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                registros.append(json.loads(linea))
            except json.JSONDecodeError as exc:
                raise ValueError(f"jsonl invalido en {archivo}:{numero}") from exc
    return registros


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
        "p50": round(float(np.percentile(arreglo, 50)), 2),
        "p95": round(float(np.percentile(arreglo, 95)), 2),
        "max": round(float(arreglo.max()), 2),
        "media": round(float(arreglo.mean()), 2),
    }


# ===== SECCION 3: modelo, prompts y cache de scores =====


def cargar_tokenizer(rutas: dict):
    """Carga el tokenizer local de ALIA (delegado al script 33).

    Args:
        rutas: Rutas absolutas.

    Returns:
        Tokenizer con pad_token = eos.
    """
    return modulo_33().cargar_tokenizer(rutas)


def verificar_ids_respuesta(tokenizer, rutas: dict, logger: logging.Logger) -> dict:
    """Deriva y verifica los ids EN CONTEXTO de "Si"/"No" (script 33).

    Contrasta los ids derivados del render con los registrados por el
    dataset v1 (si existe el JSON) para detectar cualquier deriva del
    tokenizer o de la plantilla.

    Args:
        tokenizer: Tokenizer de ALIA.
        rutas: Rutas absolutas.
        logger: Logger activo.

    Returns:
        Dict {"standalone": {...}, "contexto": {...}, "esperados": {...}}.
    """
    ids_ctx = modulo_33().ids_respuesta_contexto(tokenizer, modulo_33().CONFIG, logger)
    ruta_json = Path(rutas["ruta_dataset_json"])
    if ruta_json.exists():
        with open(ruta_json, encoding="utf-8") as manejador:
            registrados = json.load(manejador).get("ids_respuesta", {}).get("contexto")
        if registrados is not None and {str(k): int(v) for k, v in registrados.items()} != ids_ctx["contexto"]:
            logger.warning(
                "ids de respuesta en contexto derivados %s != registrados por el dataset %s",
                ids_ctx["contexto"],
                registrados,
            )
    return ids_ctx


def crear_modelo_smoke(tokenizer, cfg: dict, device: str, logger: logging.Logger):
    """Construye un LLaMA diminuto aleatorio + LoRA para el dry-run smoke.

    No descarga nada: configura un LlamaForCausalLM con la vocab del
    tokenizer real de ALIA y pesos aleatorios, le aplica un adaptador LoRA
    (r=4) y perturba lora_B para que el adaptador tenga efecto medible.

    Args:
        tokenizer: Tokenizer de ALIA (define vocab_size).
        cfg: Configuracion efectiva (los caps del modelo estan en
            CONFIG["smoke"], no en cfg["smoke"], que es el flag booleano).
        device: "cpu" o "cuda".
        logger: Logger activo.

    Returns:
        PeftModel diminuto en modo eval.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    parametros = CONFIG["smoke"]
    config_llama = LlamaConfig(
        vocab_size=int(tokenizer.vocab_size),
        hidden_size=parametros["ocultos"],
        intermediate_size=parametros["intermedio"],
        num_hidden_layers=parametros["capas"],
        num_attention_heads=parametros["cabezas"],
        num_key_value_heads=parametros["cabezas"],
        max_position_embeddings=parametros["max_seq"],
        tie_word_embeddings=True,
    )
    base = LlamaForCausalLM(config_llama)
    base.config.use_cache = False
    config_lora = LoraConfig(
        r=parametros["lora_r"],
        lora_alpha=parametros["lora_alpha"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    modelo = get_peft_model(base, config_lora)
    with torch.no_grad():
        for nombre, parametro in modelo.named_parameters():
            if "lora_B" in nombre:
                parametro.normal_(0.0, 0.02)
    modelo.to(device)
    modelo.eval()
    total = sum(p.numel() for p in modelo.parameters())
    logger.info(
        "[smoke] LLaMA diminuto: vocab=%d ocultos=%d capas=%d | %.1f M params | device=%s",
        config_llama.vocab_size,
        config_llama.hidden_size,
        config_llama.num_hidden_layers,
        total / 1e6,
        device,
    )
    return modelo


def cargar_modelo_evaluacion(cfg: dict, rutas: dict, args: argparse.Namespace, device: str, logger: logging.Logger):
    """Carga el modelo base en bf16 y el adaptador LoRA (sin merge).

    En modo smoke devuelve el modelo diminuto aleatorio. En modo real exige
    que exista el directorio del adaptador (--adaptador o --checkpoint) y
    aborta si la VRAM libre es insuficiente (el entrenamiento puede estar
    ocupando la GPU).

    Args:
        cfg: Configuracion efectiva.
        rutas: Rutas absolutas.
        args: Namespace de la CLI.
        device: "cuda" o "cpu".
        logger: Logger activo.

    Returns:
        PeftModel listo para evaluar.

    Raises:
        RuntimeError: Si falta el adaptador o la VRAM es insuficiente.
    """
    tokenizer = cargar_tokenizer(rutas)
    if cfg["smoke"]:
        return crear_modelo_smoke(tokenizer, cfg, device, logger)
    _exigir(Path(rutas["ruta_modelo"]).is_dir(), f"no existe el modelo base {rutas['ruta_modelo']}")
    dir_adaptador = Path(rutas["dir_adapter"])
    _exigir(
        (dir_adaptador / "adapter_config.json").exists(),
        (
            f"no existe el adaptador en {dir_adaptador}. Si el entrenamiento sigue en curso, "
            "espere a que guarde checkpoints o use --checkpoint modelos/c2_lora/checkpoints/ultimo."
        ),
    )
    if device == "cuda":
        libre, total = torch.cuda.mem_get_info()
        libre_gib = libre / 1024**3
        logger.info("GPU %s: %.2f GiB libres de %.2f GiB", torch.cuda.get_device_name(0), libre_gib, total / 1024**3)
        _exigir(
            libre_gib >= cfg["vram_min_gib_eval"],
            (
                f"VRAM libre insuficiente para evaluar: {libre_gib:.2f} GiB < {cfg['vram_min_gib_eval']} GiB. "
                "Probablemente el entrenamiento sigue ocupando la GPU; espere a que termine."
            ),
        )
    logger.info("cargando modelo base bf16 desde %s", rutas["ruta_modelo"])
    base = AutoModelForCausalLM.from_pretrained(
        rutas["ruta_modelo"],
        dtype=torch.bfloat16,
        attn_implementation=cfg["attn_implementation"],
    )
    base.config.use_cache = False
    modelo = PeftModel.from_pretrained(base, str(dir_adaptador), is_trainable=False)
    modelo.to(device)
    modelo.eval()
    logger.info("adaptador cargado desde %s (sin merge)", dir_adaptador)
    return modelo


def md5_adaptador(rutas: dict) -> dict:
    """Calcula las huellas del adaptador evaluado (config y pesos).

    Args:
        rutas: Rutas absolutas.

    Returns:
        Dict con ruta, md5 de adapter_config.json y del safetensors/binario.
    """
    dir_adaptador = Path(rutas["dir_adapter"])
    huellas = {"ruta": str(dir_adaptador)}
    for nombre in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
        ruta = dir_adaptador / nombre
        if ruta.exists():
            huellas[f"md5_{nombre.replace('.', '_')}"] = md5_archivo(ruta)
    return huellas


def max_seq_del_entrenamiento(rutas: dict, logger: logging.Logger) -> int | None:
    """Lee el max_seq con el que se entreno el adaptador (si es deducible).

    Fuente primaria: resultados/c2_lora/entrenamiento_v1.json (config).
    Fuente secundaria: el estado reanudable del checkpoint ultimo
    (entrenamiento_estado.pt), util cuando se evalua un checkpoint
    intermedio antes del cierre del run. Cualquier fallo de lectura se
    degrada a None (la alineacion es una comodidad, no un requisito).

    Args:
        rutas: Rutas absolutas.
        logger: Logger activo.

    Returns:
        max_seq del entrenamiento o None si no puede deducirse.
    """
    ruta_json = Path(rutas["ruta_train_json"])
    if ruta_json.exists():
        try:
            with open(ruta_json, encoding="utf-8") as manejador:
                valor = (json.load(manejador).get("config") or {}).get("max_seq")
            if valor:
                return int(valor)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("no se pudo leer %s: %s", ruta_json, exc)
    ruta_estado = Path(rutas["dir_checkpoints"]) / "ultimo" / "entrenamiento_estado.pt"
    if ruta_estado.exists():
        try:
            estado = torch.load(ruta_estado, map_location="cpu", weights_only=False)
            valor = ((estado.get("estado") or {}).get("config") or {}).get("max_seq")
            if valor:
                return int(valor)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("no se pudo leer %s (checkpoint en escritura?): %s", ruta_estado, exc)
    return None


def render_prompt_ids(tokenizer, system: str, usuario: str, max_seq: int) -> list[int]:
    """Renderiza el prompt (system + user + generation prompt) y lo trunca.

    Reserva 2 tokens para la respuesta ("Si"/"No" + <|im_end|>), igual que
    el script 33; la truncacion corta SOLO la cola del prompt.

    Args:
        tokenizer: Tokenizer de ALIA.
        system: System prompt.
        usuario: Contenido del usuario.
        max_seq: Longitud maxima total.

    Returns:
        Lista de ids del prompt (<= max_seq - 2).
    """
    base = [{"role": "system", "content": system}, {"role": "user", "content": usuario}]
    ids = modulo_33()._aplicar_plantilla(tokenizer, base, True)
    return ids[: max_seq - 2]


def construir_prompt_usuario(
    cfg: dict,
    fragmento: str,
    articulo: str,
    pasaje: str,
    max_chars_fragmento: int | None = None,
    max_chars_pasaje: int | None = None,
) -> str:
    """Construye el mensaje de usuario con la plantilla exacta del script 33.

    Args:
        cfg: Configuracion efectiva.
        fragmento: Fragmento/encabezado de la sentencia.
        articulo: Numero del articulo (formato del canon del proyecto).
        pasaje: Texto del articulo de la CP.
        max_chars_fragmento: Cap de caracteres del fragmento (default: cfg).
        max_chars_pasaje: Cap de caracteres del pasaje (default: cfg).

    Returns:
        Mensaje de usuario renderizado.
    """
    cap_fragmento = cfg["max_chars_fragmento_gold"] if max_chars_fragmento is None else max_chars_fragmento
    cap_pasaje = cfg["max_chars_pasaje"] if max_chars_pasaje is None else max_chars_pasaje
    return cfg["plantilla_user"].format(
        chunk=fragmento[:cap_fragmento],
        art=articulo,
        pasaje=pasaje[:cap_pasaje],
    )


def render_con_presupuesto(
    tokenizer,
    cfg: dict,
    fragmento: str,
    articulo: str,
    pasaje: str,
    max_chars_fragmento: int | None = None,
    max_chars_pasaje: int | None = None,
) -> dict:
    """Renderiza el prompt respetando el presupuesto de tokens.

    Si el prompt excede max_seq - 2, reduce primero el FRAGMENTO y, si aun
    no cabe, tambien el PASAJE (re-render): el texto del articulo y la
    pregunta final NUNCA se descartan por truncado de cola. Devuelve
    trazabilidad del ajuste.

    Args:
        tokenizer: Tokenizer de ALIA.
        cfg: Configuracion efectiva.
        fragmento: Fragmento/encabezado de la sentencia.
        articulo: Numero del articulo.
        pasaje: Texto del articulo de la CP.
        max_chars_fragmento: Cap inicial del fragmento.
        max_chars_pasaje: Cap del pasaje.

    Returns:
        Dict con prompt_ids, n_chars_fragmento, n_chars_pasaje, ajustado
        y n_tokens_prompt.
    """
    cap_fragmento = cfg["max_chars_fragmento_gold"] if max_chars_fragmento is None else max_chars_fragmento
    cap_pasaje = cfg["max_chars_pasaje"] if max_chars_pasaje is None else max_chars_pasaje
    fragmento_actual = fragmento[:cap_fragmento]
    pasaje_actual = pasaje[:cap_pasaje]
    presupuesto = cfg["max_seq"] - 2

    def _render(fragmento_render: str, pasaje_render: str) -> list[int]:
        """Renderiza system+user+generation prompt sin truncar."""
        return modulo_33()._aplicar_plantilla(
            tokenizer,
            [
                {"role": "system", "content": cfg["system"]},
                {
                    "role": "user",
                    "content": construir_prompt_usuario(
                        cfg, fragmento_render, articulo, pasaje_render, cap_fragmento, cap_pasaje
                    ),
                },
            ],
            True,
        )

    render = _render(fragmento_actual, pasaje_actual)
    ajustado = len(render) > presupuesto
    while len(render) > presupuesto and len(fragmento_actual) > 40:
        fragmento_actual = fragmento_actual[: max(40, int(0.8 * len(fragmento_actual)))]
        render = _render(fragmento_actual, pasaje_actual)
    while len(render) > presupuesto and len(pasaje_actual) > 40:
        pasaje_actual = pasaje_actual[: max(40, int(0.8 * len(pasaje_actual)))]
        render = _render(fragmento_actual, pasaje_actual)
    prompt_ids = render[:presupuesto]
    return {
        "prompt_ids": prompt_ids,
        "n_chars_fragmento": len(fragmento_actual),
        "n_chars_pasaje": len(pasaje_actual),
        "ajustado": ajustado,
        "n_tokens_prompt": len(prompt_ids),
    }


class CacheScores:
    """Cache de scores P("Si") por md5 de los prompt ids.

    Evita recomputar los mismos prompts entre corridas y checkpoints. El
    JSONL se carga al inicio y se agrega (append) por lote, de modo que un
    corte no pierde los scores ya calculados.
    """

    def __init__(self, ruta: str | Path, activa: bool, modelo_tag: str):
        """Inicializa el cache.

        Args:
            ruta: Ruta del JSONL de cache.
            activa: False desactiva la lectura/escritura (smoke).
            modelo_tag: Etiqueta del modelo/adaptador (parte de la clave).
        """
        self.ruta = Path(ruta)
        self.activa = activa
        self.modelo_tag = modelo_tag
        self.mapa: dict[str, float] = {}
        self.pendientes: list[dict] = []
        if activa and self.ruta.exists():
            for registro in leer_jsonl(self.ruta):
                clave = registro.get("clave")
                score = registro.get("score")
                if isinstance(clave, str) and isinstance(score, (int, float)):
                    self.mapa[clave] = float(score)

    def clave(self, prompt_ids: list[int], max_seq: int, modo: str, etiqueta: str) -> str:
        """Calcula la clave md5 de un prompt.

        Args:
            prompt_ids: Ids del prompt renderizado.
            max_seq: Longitud maxima usada (afecta la truncacion).
            modo: "val", "gold" o "benchmark".
            etiqueta: "lora" o "base".

        Returns:
            Hexdigest md5 de la clave.
        """
        cuerpo = f"{self.modelo_tag}|{modo}|{etiqueta}|{max_seq}|" + ",".join(str(t) for t in prompt_ids)
        return modulo_33().md5_bytes(cuerpo)

    def obtener(self, clave: str) -> float | None:
        """Devuelve el score cacheado o None.

        Args:
            clave: Clave md5.

        Returns:
            Score float o None.
        """
        return self.mapa.get(clave)

    def registrar(self, clave: str, score: float, modo: str, etiqueta: str) -> None:
        """Registra un score nuevo (memoria + cola de append).

        Args:
            clave: Clave md5.
            score: P("Si") calculada.
            modo: "val", "gold" o "benchmark".
            etiqueta: "lora" o "base".
        """
        self.mapa[clave] = float(score)
        self.pendientes.append(
            {
                "clave": clave,
                "score": float(score),
                "modo": modo,
                "etiqueta": etiqueta,
                "modelo": self.modelo_tag,
            }
        )

    def guardar(self) -> int:
        """Agrega los scores pendientes al JSONL (append + flush).

        Returns:
            Numero de registros escritos en esta llamada.
        """
        if not self.activa or not self.pendientes:
            return 0
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ruta, "a", encoding="utf-8") as manejador:
            manejador.writelines(
                json.dumps(registro, ensure_ascii=False) + "\n" for registro in self.pendientes
            )
            manejador.flush()
            os.fsync(manejador.fileno())
        escritos = len(self.pendientes)
        self.pendientes = []
        return escritos


def _collar_scoring(prompts_ids: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Collator de scoring con padding IZQUIERDO.

    El padding izquierdo garantiza que la ultima columna sea siempre el
    ultimo token real de cada prompt (con logits_to_keep=1 el logit del
    siguiente token se lee sin offsets por fila).

    Args:
        prompts_ids: Lista de ids de prompt.
        pad_id: Id de padding (eos).

    Returns:
        Tupla (input_ids, attention_mask) con shape (B, L).
    """
    n = len(prompts_ids)
    largo = max(len(ids) for ids in prompts_ids)
    input_ids = torch.full((n, largo), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n, largo), dtype=torch.long)
    for i, ids in enumerate(prompts_ids):
        input_ids[i, largo - len(ids) :] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, largo - len(ids) :] = 1
    return input_ids, attention_mask


def _logits_ultima_posicion(modelo, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Devuelve los logits del ultimo token real de cada fila.

    Usa logits_to_keep=1 (receta de memoria del plan: vocab 256k) con
    fallback al forward completo indexado por la ultima posicion valida.

    Args:
        modelo: Modelo causal (PeftModel).
        input_ids: Tensor (B, L) con padding izquierdo.
        attention_mask: Tensor (B, L).

    Returns:
        Tensor (B, vocab) con los logits del ultimo token real.
    """
    try:
        salida = modelo(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=1)
        return salida.logits[:, 0, :]
    except TypeError:
        salida = modelo(input_ids=input_ids, attention_mask=attention_mask)
        return salida.logits[:, -1, :]


@torch.inference_mode()
def puntuar_prompts_con_stats(
    modelo,
    tokenizer,
    prompts_ids: list[list[int]],
    etiqueta: str,
    modo: str,
    cfg: dict,
    device: str,
    ids_ctx: dict,
    cache: CacheScores,
    logger: logging.Logger,
) -> tuple[list[float], dict]:
    """Calcula P("Si") por logits y devuelve estadisticas del scoring.

    Version con estadisticas de puntuar_prompts (que delega aqui): expone
    el tiempo por prompt SIN cache, necesario para estimar la latencia de
    rutas cuyos prompts ya estaban cacheados (p. ej. la cascada top-20,
    subconjunto del top-50).

    Args:
        modelo: PeftModel en modo eval.
        tokenizer: Tokenizer (pad_token_id).
        prompts_ids: Lista de ids de prompt.
        etiqueta: "lora" o "base" (para el cache).
        modo: "val", "gold" o "benchmark" (para el cache).
        cfg: Configuracion efectiva.
        device: "cuda" o "cpu".
        ids_ctx: Salida de verificar_ids_respuesta.
        cache: Cache de scores.
        logger: Logger activo.

    Returns:
        Tupla (probs, stats) con stats = {n, n_cache, segundos,
        segundos_por_prompt_sin_cache}.
    """
    id_si = int(ids_ctx["contexto"]["Sí"])
    id_no = int(ids_ctx["contexto"]["No"])
    claves = [cache.clave(ids, cfg["max_seq"], modo, etiqueta) for ids in prompts_ids]
    probs: list[float | None] = [None] * len(prompts_ids)
    for i, clave in enumerate(claves):
        valor = cache.obtener(clave)
        if valor is not None:
            probs[i] = valor
    pendientes = [i for i, valor in enumerate(probs) if valor is None]
    inicio = time.perf_counter()
    for inicio_lote in range(0, len(pendientes), cfg["batch"]):
        indices = pendientes[inicio_lote : inicio_lote + cfg["batch"]]
        input_ids, attention_mask = _collar_scoring([prompts_ids[i] for i in indices], tokenizer.pad_token_id)
        logits = _logits_ultima_posicion(modelo, input_ids.to(device), attention_mask.to(device))
        par = logits[:, [id_si, id_no]].float()
        probabilidades = torch.softmax(par, dim=1)[:, 0]
        for i, probabilidad in zip(indices, probabilidades.tolist(), strict=True):
            probs[i] = float(probabilidad)
            cache.registrar(claves[i], float(probabilidad), modo, etiqueta)
        if len(cache.pendientes) >= 256:
            cache.guardar()
    cache.guardar()
    segundos = time.perf_counter() - inicio
    n_cache = len(prompts_ids) - len(pendientes)
    stats = {
        "n": len(prompts_ids),
        "n_cache": n_cache,
        "segundos": segundos,
        "segundos_por_prompt_sin_cache": segundos / len(pendientes) if pendientes else None,
    }
    logger.info(
        "scoring %s/%s: %d prompts (%d de cache) en %.1f s (%.1f prompts/s)",
        modo,
        etiqueta,
        len(prompts_ids),
        n_cache,
        segundos,
        len(pendientes) / segundos if segundos > 0 else 0.0,
    )
    return [float(valor) for valor in probs], stats


@torch.inference_mode()
def puntuar_prompts(
    modelo,
    tokenizer,
    prompts_ids: list[list[int]],
    etiqueta: str,
    modo: str,
    cfg: dict,
    device: str,
    ids_ctx: dict,
    cache: CacheScores,
    logger: logging.Logger,
) -> list[float]:
    """Calcula P("Si") por logits para una lista de prompts (sin generar).

    Score = softmax sobre el par {id Si, id No} en la posicion del primer
    token de respuesta (ids EN CONTEXTO verificados). Usa cache por md5 y
    procesa por lotes con padding izquierdo.

    Args:
        modelo: PeftModel en modo eval.
        tokenizer: Tokenizer (pad_token_id).
        prompts_ids: Lista de ids de prompt.
        etiqueta: "lora" o "base" (para el cache).
        modo: "val", "gold" o "benchmark" (para el cache).
        cfg: Configuracion efectiva.
        device: "cuda" o "cpu".
        ids_ctx: Salida de verificar_ids_respuesta.
        cache: Cache de scores.
        logger: Logger activo.

    Returns:
        Lista de P("Si") en el orden de entrada.
    """
    probs, _ = puntuar_prompts_con_stats(
        modelo, tokenizer, prompts_ids, etiqueta, modo, cfg, device, ids_ctx, cache, logger
    )
    return probs


# ===== SECCION 4: metricas comunes (clasificacion, bootstrap, umbral) =====


def auc_seguro(etiquetas: np.ndarray, scores: np.ndarray) -> float | None:
    """Calcula AUC devolviendo None si falta alguna clase.

    Args:
        etiquetas: Etiquetas binarias.
        scores: Scores continuos.

    Returns:
        AUC float o None.
    """
    if len(np.unique(etiquetas)) < 2:
        return None
    return float(roc_auc_score(etiquetas, scores))


def spearman_seguro(scores: np.ndarray, grados: np.ndarray) -> float | None:
    """Calcula Spearman devolviendo None si algun vector es constante.

    Args:
        scores: Scores continuos.
        grados: Grados ordinales.

    Returns:
        Coeficiente de Spearman o None.
    """
    if len(np.unique(scores)) < 2 or len(np.unique(grados)) < 2:
        return None
    from scipy.stats import spearmanr

    return float(spearmanr(scores, grados).statistic)


def metricas_clasificacion(etiquetas: np.ndarray, probs: np.ndarray, umbral: float) -> dict:
    """Calcula acc, F1 (Si y macro), AUC y matriz de confusion.

    Args:
        etiquetas: Etiquetas binarias (0/1).
        probs: P("Si") por item.
        umbral: Umbral de decision sobre P("Si").

    Returns:
        Dict con acc, f1_si, f1_macro, auc, matriz y n.
    """
    predicciones = (probs >= umbral).astype(np.int64)
    return {
        "acc": float(accuracy_score(etiquetas, predicciones)),
        "f1_si": float(f1_score(etiquetas, predicciones, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(etiquetas, predicciones, average="macro", zero_division=0)),
        "auc": auc_seguro(etiquetas, probs),
        "matriz": confusion_matrix(etiquetas, predicciones, labels=[0, 1]).tolist(),
        "umbral": float(umbral),
        "n": len(etiquetas),
    }


def optimizar_umbral(etiquetas: np.ndarray, probs: np.ndarray, criterio: str) -> dict:
    """Optimiza el umbral de decision con validacion (nunca con test).

    Recorre los umbrales candidatos (puntos medios entre probs ordenadas)
    y maximiza Youden J (sensibilidad + especificidad - 1) o F1 de la
    clase positiva.

    Args:
        etiquetas: Etiquetas binarias.
        probs: P("Si") por item.
        criterio: "youden" o "f1".

    Returns:
        Dict con umbral, valor del criterio, youden, f1 y n.

    Raises:
        ValueError: Si el criterio no es valido o falta una clase.
    """
    if criterio not in ("youden", "f1"):
        raise ValueError(f"criterio de umbral desconocido: {criterio}")
    if len(np.unique(etiquetas)) < 2:
        return {"umbral": 0.5, "criterio": criterio, "valor": None, "youden": None, "f1": None, "n": len(etiquetas)}
    orden = np.argsort(probs)
    probs_ordenadas = probs[orden]
    candidatos = np.unique(np.concatenate([[0.0], (probs_ordenadas[:-1] + probs_ordenadas[1:]) / 2.0, [1.0]]))
    mejor = {"umbral": 0.5, "valor": -np.inf, "youden": None, "f1": None}
    for umbral in candidatos:
        predicciones = (probs >= umbral).astype(np.int64)
        tp = int(((predicciones == 1) & (etiquetas == 1)).sum())
        tn = int(((predicciones == 0) & (etiquetas == 0)).sum())
        fp = int(((predicciones == 1) & (etiquetas == 0)).sum())
        fn = int(((predicciones == 0) & (etiquetas == 1)).sum())
        sensibilidad = tp / (tp + fn) if (tp + fn) else 0.0
        especificidad = tn / (tn + fp) if (tn + fp) else 0.0
        youden = sensibilidad + especificidad - 1.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        valor = youden if criterio == "youden" else f1
        if valor > mejor["valor"] + 1e-12:
            mejor = {"umbral": float(umbral), "valor": float(valor), "youden": float(youden), "f1": float(f1)}
    return {"criterio": criterio, "n": len(etiquetas), **mejor}


def bootstrap_pareado_grupos(
    etiquetas: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    grupos: list[str],
    funcion,
    n_remuestreos: int,
    semilla: int,
    max_intentos: int,
) -> dict:
    """Bootstrap pareado por clusters (caso o documento) de a - b.

    Remuestrea GRUPOS completos con reemplazo (los items de un caso/doc no
    son independientes) y calcula la metrica en a y b sobre el mismo
    remuestreo; repite hasta lograr n_remuestreos validos (la metrica puede
    ser indefinida en un remuestreo sin ambas clases y se descarta).

    Args:
        etiquetas: Etiquetas binarias por item.
        probs_a: Scores de la ruta nueva.
        probs_b: Scores de la referencia.
        grupos: Id de caso/doc por item.
        funcion: Callable (etiquetas, scores) -> float | None.
        n_remuestreos: Remuestreos objetivo.
        semilla: Semilla del generador.
        max_intentos: Tope de intentos (evita bucles infinitos).

    Returns:
        Dict con n_validos, media_a, media_b, media_delta, p2_5, p97_5,
        prob_mejora y gana (extremo inferior del IC > 0).
    """
    grupos_unicos = sorted(set(grupos))
    indices_por_grupo: dict[str, list[int]] = defaultdict(list)
    for i, grupo in enumerate(grupos):
        indices_por_grupo[grupo].append(i)
    rng = np.random.default_rng(semilla)
    deltas: list[float] = []
    valores_a: list[float] = []
    valores_b: list[float] = []
    intentos = 0
    max_total = max_intentos * n_remuestreos
    while len(deltas) < n_remuestreos and intentos < max_total:
        intentos += 1
        seleccion = rng.integers(0, len(grupos_unicos), size=len(grupos_unicos))
        indices = np.concatenate([indices_por_grupo[grupos_unicos[i]] for i in seleccion])
        valor_a = funcion(etiquetas[indices], probs_a[indices])
        valor_b = funcion(etiquetas[indices], probs_b[indices])
        if valor_a is None or valor_b is None:
            continue
        valores_a.append(float(valor_a))
        valores_b.append(float(valor_b))
        deltas.append(float(valor_a) - float(valor_b))
    if not deltas:
        return {
            "n_validos": 0,
            "media_a": None,
            "media_b": None,
            "media_delta": None,
            "p2_5": None,
            "p97_5": None,
            "prob_mejora": None,
            "gana": False,
        }
    arreglo = np.asarray(deltas, dtype=np.float64)
    p2_5 = float(np.percentile(arreglo, 2.5))
    p97_5 = float(np.percentile(arreglo, 97.5))
    return {
        "n_validos": len(deltas),
        "intentos": intentos,
        "media_a": round(float(np.mean(valores_a)), 4),
        "media_b": round(float(np.mean(valores_b)), 4),
        "media_delta": round(float(arreglo.mean()), 4),
        "p2_5": round(p2_5, 4),
        "p97_5": round(p97_5, 4),
        "prob_mejora": round(float((arreglo > 0).mean()), 4),
        "gana": bool(p2_5 > 0.0),
    }


def bootstrap_pareado_casos(
    valores_a: np.ndarray, valores_b: np.ndarray, n_remuestreos: int, semilla: int
) -> dict:
    """Bootstrap pareado por caso (delegado al modulo 12, 1000/42).

    Args:
        valores_a: Metrica por caso de la ruta nueva.
        valores_b: Metrica por caso de la referencia.
        n_remuestreos: Remuestreos (1000).
        semilla: Semilla (42).

    Returns:
        Dict del bootstrap pareado del modulo 12.
    """
    return modulo_12().bootstrap_ic_pareado(valores_a, valores_b, n_remuestreos, semilla)


def bootstrap_ic_media(valores: np.ndarray, n_remuestreos: int, semilla: int) -> dict:
    """IC 95% bootstrap de la media (delegado al modulo 12).

    Args:
        valores: Valores por caso.
        n_remuestreos: Remuestreos (1000).
        semilla: Semilla (42).

    Returns:
        Dict con n, media, p2_5, p97_5.
    """
    return modulo_12().bootstrap_ic_media(valores, n_remuestreos, semilla)


# ===== SECCION 5: nivel 1 - val interna (E1) =====


def cargar_ejemplos_sft(rutas: dict, cfg: dict, nombre: str) -> tuple[dict, list[dict]]:
    """Carga un split SFT y prepara los ejemplos de scoring.

    Args:
        rutas: Rutas absolutas.
        cfg: Configuracion efectiva.
        nombre: "val" o "val_eval".

    Returns:
        Tupla (datos, ejemplos) con datos crudos (doc_id, label) y
        ejemplos {"prompt_ids", "label", "doc_id"}.

    Raises:
        FileNotFoundError: Si falta el parquet del split.
    """
    ruta = Path(rutas["dir_sft"]) / f"{nombre}.parquet"
    _exigir(ruta.exists(), f"falta {ruta} (ejecute 33_c2_lora_train.py --preparar-datos)")
    datos = modulo_33().cargar_sft(ruta)
    ejemplos = modulo_33().preparar_ejemplos(datos, cfg["max_seq"])
    for ejemplo, doc_id in zip(ejemplos, datos["doc_id"], strict=True):
        ejemplo["doc_id"] = str(doc_id)
    return datos, ejemplos


def evaluar_nivel1(
    modelo,
    tokenizer,
    cfg: dict,
    rutas: dict,
    device: str,
    ids_ctx: dict,
    cache: CacheScores,
    exclusiones: dict,
    logger: logging.Logger,
) -> dict:
    """Evalua el juez en el val interno (E1): adaptador vs base bf16.

    Args:
        modelo: PeftModel.
        tokenizer: Tokenizer.
        cfg: Configuracion efectiva.
        rutas: Rutas absolutas.
        device: "cuda" o "cpu".
        ids_ctx: Ids de respuesta en contexto.
        cache: Cache de scores.
        exclusiones: Dict con benchmark/gold ids (anti-leakage).
        logger: Logger activo.

    Returns:
        Dict de cierre del nivel 1.
    """
    datos_val, ejemplos_val = cargar_ejemplos_sft(rutas, cfg, "val")
    datos_eval, ejemplos_eval = cargar_ejemplos_sft(rutas, cfg, "val_eval")
    if cfg["max_items_val"]:
        ejemplos_val = ejemplos_val[: cfg["max_items_val"]]
        datos_val = {"doc_id": datos_val["doc_id"][: cfg["max_items_val"]], "label": datos_val["label"][: cfg["max_items_val"]]}
        ejemplos_eval = ejemplos_eval[: cfg["max_items_val"]]
        datos_eval = {
            "doc_id": datos_eval["doc_id"][: cfg["max_items_val"]],
            "label": datos_eval["label"][: cfg["max_items_val"]],
        }
    docs_val = [str(doc) for doc in datos_val["doc_id"]]
    fuga = len(set(docs_val) & exclusiones["benchmark"])
    _exigir(fuga == 0, f"ABORT anti-leakage: {fuga} docs de val pertenecen al benchmark M7")
    logger.info(
        "nivel 1: val=%d pares (%d docs) | val_eval=%d pares",
        len(ejemplos_val),
        len(set(docs_val)),
        len(ejemplos_eval),
    )

    prompts_val = [ejemplo["prompt_ids"] for ejemplo in ejemplos_val]
    prompts_eval = [ejemplo["prompt_ids"] for ejemplo in ejemplos_eval]
    probs_lora = puntuar_prompts(
        modelo, tokenizer, prompts_val, "lora", "val", cfg, device, ids_ctx, cache, logger
    )
    probs_lora_eval = puntuar_prompts(
        modelo, tokenizer, prompts_eval, "lora", "val_eval", cfg, device, ids_ctx, cache, logger
    )
    with modelo.disable_adapter():
        probs_base = puntuar_prompts(
            modelo, tokenizer, prompts_val, "base", "val", cfg, device, ids_ctx, cache, logger
        )
        probs_base_eval = puntuar_prompts(
            modelo, tokenizer, prompts_eval, "base", "val_eval", cfg, device, ids_ctx, cache, logger
        )

    y_val = np.asarray(datos_val["label"], dtype=np.int64)
    y_eval = np.asarray(datos_eval["label"], dtype=np.int64)
    arreglo_lora = np.asarray(probs_lora, dtype=np.float64)
    arreglo_lora_eval = np.asarray(probs_lora_eval, dtype=np.float64)
    arreglo_base = np.asarray(probs_base, dtype=np.float64)
    arreglo_base_eval = np.asarray(probs_base_eval, dtype=np.float64)

    umbral = optimizar_umbral(y_eval, arreglo_lora_eval, cfg["criterio_umbral"])
    metricas_lora = metricas_clasificacion(y_val, arreglo_lora, cfg["umbral"])
    metricas_lora_umbral = metricas_clasificacion(y_val, arreglo_lora, umbral["umbral"])
    metricas_base = metricas_clasificacion(y_val, arreglo_base, cfg["umbral"])
    metricas_lora_eval = metricas_clasificacion(y_eval, arreglo_lora_eval, cfg["umbral"])
    metricas_base_eval = metricas_clasificacion(y_eval, arreglo_base_eval, cfg["umbral"])

    def funcion_acc(etiquetas: np.ndarray, scores: np.ndarray) -> float:
        """Accuracy al umbral 0,5 para el bootstrap pareado."""
        return float(accuracy_score(etiquetas, (scores >= cfg["umbral"]).astype(int)))

    def funcion_auc(etiquetas: np.ndarray, scores: np.ndarray) -> float | None:
        """AUC para el bootstrap pareado (None si falta una clase)."""
        return auc_seguro(etiquetas, scores)

    def funcion_f1(etiquetas: np.ndarray, scores: np.ndarray) -> float:
        """F1 de la clase positiva al umbral 0,5 para el bootstrap pareado."""
        return float(f1_score(etiquetas, (scores >= cfg["umbral"]).astype(int), pos_label=1, zero_division=0))

    delta_acc = bootstrap_pareado_grupos(
        y_val, arreglo_lora, arreglo_base, docs_val, funcion_acc, cfg["bootstrap"], cfg["semilla"], cfg["max_intentos_bootstrap"]
    )
    delta_auc = bootstrap_pareado_grupos(
        y_val, arreglo_lora, arreglo_base, docs_val, funcion_auc, cfg["bootstrap"], cfg["semilla"], cfg["max_intentos_bootstrap"]
    )
    delta_f1 = bootstrap_pareado_grupos(
        y_val, arreglo_lora, arreglo_base, docs_val, funcion_f1, cfg["bootstrap"], cfg["semilla"], cfg["max_intentos_bootstrap"]
    )

    referencia_entrenamiento = None
    ruta_train = Path(rutas["ruta_train_json"])
    if ruta_train.exists():
        with open(ruta_train, encoding="utf-8") as manejador:
            entrenamiento = json.load(manejador)
        referencia_entrenamiento = {
            "archivo": str(ruta_train),
            "md5": md5_archivo(ruta_train),
            "eval_final": entrenamiento.get("eval_final"),
            "mejor_paso": entrenamiento.get("entrenamiento", {}).get("mejor_paso"),
            "mejor_auc": entrenamiento.get("entrenamiento", {}).get("mejor_auc"),
        }
        logger.info(
            "referencia del entrenamiento: mejor_auc=%s (paso %s) | eval_final acc=%s",
            referencia_entrenamiento["mejor_auc"],
            referencia_entrenamiento["mejor_paso"],
            (referencia_entrenamiento["eval_final"] or {}).get("acc"),
        )
    else:
        logger.warning("sin %s: no hay referencia del eval_final del entrenamiento", ruta_train)

    criterios = cfg["criterios_s2"]
    cumple = {
        "acc": metricas_lora["acc"] >= criterios["acc"],
        "auc": (metricas_lora["auc"] is not None and metricas_lora["auc"] >= criterios["auc"]),
        "delta_acc_vs_base": delta_acc["gana"] and delta_acc["media_delta"] >= criterios["delta_acc_vs_base"],
    }
    cierre = {
        "split": "val.parquet (docs de B1)",
        "md5_val": md5_archivo(Path(rutas["dir_sft"]) / "val.parquet"),
        "md5_val_eval": md5_archivo(Path(rutas["dir_sft"]) / "val_eval.parquet"),
        "n_val": len(ejemplos_val),
        "n_val_eval": len(ejemplos_eval),
        "n_docs_val": len(set(docs_val)),
        "umbral_optimizado": {**umbral, "origen": "val_eval (subset de val; nunca test)"},
        "lora": {
            "umbral_0_5": metricas_lora,
            "umbral_optimo": metricas_lora_umbral,
            "val_eval_umbral_0_5": metricas_lora_eval,
        },
        "base_bf16": {"umbral_0_5": metricas_base, "val_eval_umbral_0_5": metricas_base_eval},
        "delta_lora_vs_base": {"acc": delta_acc, "auc": delta_auc, "f1_si": delta_f1},
        "referencia_entrenamiento": referencia_entrenamiento,
        "anti_leakage": {"docs_val_en_benchmark": fuga},
        "criterios_s2": {
            "umbrales": criterios,
            "cumple": cumple,
            "resultado": "GO" if all(cumple.values()) else "NO_CUMPLE",
        },
    }
    logger.info(
        "nivel 1: lora acc=%.4f auc=%s f1_si=%.4f | base acc=%.4f auc=%s | delta_acc=%+.4f [%.4f, %.4f]",
        metricas_lora["acc"],
        f"{metricas_lora['auc']:.4f}" if metricas_lora["auc"] is not None else "n/a",
        metricas_lora["f1_si"],
        metricas_base["acc"],
        f"{metricas_base['auc']:.4f}" if metricas_base["auc"] is not None else "n/a",
        delta_acc["media_delta"],
        delta_acc["p2_5"],
        delta_acc["p97_5"],
    )
    return cierre


def dibujar_nivel1(cierre: dict, ruta: str | Path) -> None:
    """Figura del nivel 1: distribucion de scores y metricas vs base.

    Args:
        cierre: Cierre del nivel 1.
        ruta: Ruta de la figura.
    """
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    metricas_lora = cierre["lora"]["umbral_0_5"]
    metricas_base = cierre["base_bf16"]["umbral_0_5"]
    nombres = ["acc", "f1_si", "auc"]
    x = np.arange(len(nombres))
    valores_lora = [metricas_lora[n] if metricas_lora[n] is not None else 0.0 for n in nombres]
    valores_base = [metricas_base[n] if metricas_base[n] is not None else 0.0 for n in nombres]
    ejes[0].bar(x - 0.2, valores_lora, width=0.4, label="LoRA", color="#1f77b4")
    ejes[0].bar(x + 0.2, valores_base, width=0.4, label="base bf16", color="#7f7f7f")
    ejes[0].set_xticks(x, nombres)
    ejes[0].axhline(0.78, color="red", linestyle="--", linewidth=1, label="acc objetivo 0,78")
    ejes[0].axhline(0.85, color="orange", linestyle="--", linewidth=1, label="auc objetivo 0,85")
    ejes[0].set_ylim(0, 1.05)
    ejes[0].set_title("Nivel 1 - val (3.000 pares)")
    ejes[0].legend(fontsize=8)
    delta = cierre["delta_lora_vs_base"]["acc"]
    media = delta["media_delta"] if delta["media_delta"] is not None else 0.0
    inferior = media - delta["p2_5"] if delta["p2_5"] is not None else 0.0
    superior = delta["p97_5"] - media if delta["p97_5"] is not None else 0.0
    ejes[1].bar([0], [media], yerr=[[inferior], [superior]], capsize=6, color="#2ca02c")
    ejes[1].axhline(0, color="red", linestyle="--")
    ejes[1].axhline(0.15, color="orange", linestyle="--", linewidth=1, label="objetivo +0,15")
    ejes[1].set_xticks([0], ["acc"])
    ejes[1].set_title(
        f"Delta acc LoRA - base: {media:+.3f}\nIC95 [{delta['p2_5']:+.3f}, {delta['p97_5']:+.3f}] (bootstrap por doc)"
    )
    ejes[1].legend(fontsize=8)
    figura.tight_layout()
    figura.savefig(destino, dpi=130)
    plt.close(figura)


# ===== SECCION 6: nivel 2 - gold humano B3 (E2) =====


def cargar_gold(cfg: dict, rutas: dict, logger: logging.Logger) -> dict:
    """Carga el gold humano B3: respuestas, extractos, candidatos y catalogo.

    Verifica que cada respuesta corresponda al pool de candidatos del caso
    (id, indice_candidato, articulo), que todos los articulos tengan texto
    en el catalogo de la CP (fallback al snippet del pack) y que los 50
    casos sean subconjunto del benchmark M7.

    Args:
        cfg: Configuracion efectiva.
        rutas: Rutas absolutas.
        logger: Logger activo.

    Returns:
        Dict con items, gold_por_id, extractos, catalogo y stats.

    Raises:
        RuntimeError: Si falla cualquier verificacion de integridad.
    """
    respuestas = leer_jsonl(rutas["ruta_gold_respuestas"])
    _exigir(len(respuestas) > 0, "respuestas del gold vacias")
    muestra = {str(registro["id"]): registro for registro in leer_jsonl(rutas["ruta_gold_muestra"])}
    candidatos = {str(registro["id"]): registro for registro in leer_jsonl(rutas["ruta_gold_candidatos"])}
    catalogo = modulo_16().cargar_catalogo_articulos({"articulos": rutas["ruta_articulos"]})
    _exigir(len(catalogo) == 497, f"catalogo de articulos inesperado: {len(catalogo)} filas")

    items = []
    sin_catalogo = 0
    for registro in respuestas:
        id_caso = str(registro["id"])
        _exigir(id_caso in candidatos, f"caso {id_caso} del gold sin candidatos en el pack")
        _exigir(id_caso in muestra, f"caso {id_caso} del gold sin extracto en muestra_gold")
        indice = int(registro["indice_candidato"])
        lista = candidatos[id_caso]["candidatos"]
        _exigir(1 <= indice <= len(lista), f"indice_candidato fuera de rango en el caso {id_caso}")
        candidato = lista[indice - 1]
        _exigir(
            str(candidato["articulo"]) == str(registro["articulo"]),
            f"(id, indice_candidato, articulo) inconsistente en {id_caso}: {candidato['articulo']} != {registro['articulo']}",
        )
        articulo = modulo_16().normalizar_articulo(registro["articulo"])
        texto = catalogo.get(articulo, "")
        if not texto:
            texto = str(candidato.get("snippet", ""))
            sin_catalogo += 1
        extracto = str(muestra[id_caso].get("extracto", ""))
        _exigir(bool(extracto.strip()), f"caso {id_caso} sin extracto en muestra_gold")
        items.append(
            {
                "id": id_caso,
                "articulo": articulo,
                "grado": int(registro["nota_grado"]),
                "fragmento": extracto,
                "pasaje": texto,
            }
        )
    grados = {item["grado"] for item in items}
    _exigir(grados <= {0, 1, 2}, f"grados fuera de {{0,1,2}}: {sorted(grados)}")
    n_casos = len({item["id"] for item in items})
    logger.info(
        "gold B3: %d items / %d casos | grados=%s | sin texto en catalogo=%d (fallback snippet)",
        len(items),
        n_casos,
        {g: sum(1 for item in items if item["grado"] == g) for g in (0, 1, 2)},
        sin_catalogo,
    )
    return {"items": items, "muestra": muestra, "candidatos": candidatos, "catalogo": catalogo}


def gold_por_caso(items: list[dict]) -> dict[str, dict[str, int]]:
    """Agrupa los grados por caso (id -> articulo -> grado).

    Args:
        items: Items del gold.

    Returns:
        Dict id_caso -> (articulo -> grado).
    """
    por_id: dict[str, dict[str, int]] = defaultdict(dict)
    for item in items:
        por_id[item["id"]][item["articulo"]] = item["grado"]
    return dict(por_id)


def cargar_referencias_llm(cfg: dict, rutas: dict, items: list[dict], logger: logging.Logger) -> dict:
    """Evalua las preanotaciones LLM del gold como referencia (no gate).

    Args:
        cfg: Configuracion efectiva.
        rutas: Rutas absolutas.
        items: Items del gold humano (para las etiquetas).
        logger: Logger activo.

    Returns:
        Dict por archivo con n, AUC(>=1), Spearman y kappa vs el humano
        (calculada por el script 16 si esta disponible).
    """
    grados_humanos = {(item["id"], item["articulo"]): item["grado"] for item in items}
    referencia = {}
    for ruta in rutas["rutas_gold_llm"]:
        if not Path(ruta).exists():
            continue
        registros = leer_jsonl(ruta)
        pares = []
        for registro in registros:
            clave = (str(registro["id"]), modulo_16().normalizar_articulo(registro["articulo"]))
            if clave in grados_humanos and registro.get("nota_grado") in (0, 1, 2):
                pares.append((int(registro["nota_grado"]), grados_humanos[clave]))
        if not pares:
            continue
        notas = np.asarray([p[0] for p in pares], dtype=np.float64)
        humanos = np.asarray([p[1] for p in pares], dtype=np.float64)
        nombre = Path(ruta).stem.replace("respuestas_", "")
        referencia[nombre] = {
            "archivo": ruta,
            "md5": md5_archivo(ruta),
            "n": len(pares),
            "auc_nota_llm_para_humano_ge1": auc_seguro((humanos >= 1).astype(int), notas),
            "spearman_nota_llm_vs_humano": spearman_seguro(notas, humanos),
        }
        logger.info("referencia LLM %s: n=%d auc=%.4f spearman=%.4f", nombre, len(pares), referencia[nombre]["auc_nota_llm_para_humano_ge1"], referencia[nombre]["spearman_nota_llm_vs_humano"])
    ruta_acuerdo = Path(rutas["ruta_gold_acuerdo"])
    if ruta_acuerdo.exists():
        with open(ruta_acuerdo, encoding="utf-8") as manejador:
            acuerdo = json.load(manejador)
        referencia["acuerdo"] = {
            "archivo": str(ruta_acuerdo),
            "kappas_pares": acuerdo.get("kappas_pares"),
            "nota": "kappa carlos|llm de referencia (preanotacion automatica, no humano-humano)",
        }
    return referencia


def evaluar_nivel2(
    modelo,
    tokenizer,
    cfg: dict,
    rutas: dict,
    device: str,
    ids_ctx: dict,
    cache: CacheScores,
    exclusiones: dict,
    logger: logging.Logger,
) -> dict:
    """Evalua el juez contra el gold humano B3 (E2).

    Args:
        modelo: PeftModel.
        tokenizer: Tokenizer.
        cfg: Configuracion efectiva.
        rutas: Rutas absolutas.
        device: "cuda" o "cpu".
        ids_ctx: Ids de respuesta en contexto.
        cache: Cache de scores.
        exclusiones: Dict con benchmark/gold ids (anti-leakage).
        logger: Logger activo.

    Returns:
        Dict de cierre del nivel 2.
    """
    gold = cargar_gold(cfg, rutas, logger)
    items = gold["items"]
    ids_casos = {item["id"] for item in items}
    _exigir(
        ids_casos <= exclusiones["benchmark"],
        f"ABORT anti-leakage: {len(ids_casos - exclusiones['benchmark'])} casos del gold fuera del benchmark",
    )
    if cfg["max_casos_gold"]:
        casos = sorted(ids_casos, key=modulo_16().clave_id)[: cfg["max_casos_gold"]]
        items = [item for item in items if item["id"] in casos]
    logger.info("nivel 2: %d items / %d casos del gold", len(items), len({item["id"] for item in items}))

    prompts = []
    ajustes = []
    for item in items:
        render = render_con_presupuesto(
            tokenizer, cfg, item["fragmento"], item["articulo"], item["pasaje"]
        )
        prompts.append(render["prompt_ids"])
        ajustes.append(render["ajustado"])
    probs_lora = puntuar_prompts(modelo, tokenizer, prompts, "lora", "gold", cfg, device, ids_ctx, cache, logger)
    with modelo.disable_adapter():
        probs_base = puntuar_prompts(modelo, tokenizer, prompts, "base", "gold", cfg, device, ids_ctx, cache, logger)

    grados = np.asarray([item["grado"] for item in items], dtype=np.int64)
    casos = [item["id"] for item in items]
    arreglo_lora = np.asarray(probs_lora, dtype=np.float64)
    arreglo_base = np.asarray(probs_base, dtype=np.float64)
    etiquetas_ge1 = (grados >= 1).astype(np.int64)
    etiquetas_2 = (grados == 2).astype(np.int64)

    def metricas_gold(probs: np.ndarray) -> dict:
        """Metricas del gold para un vector de scores.

        Args:
            probs: P("Si") por item.

        Returns:
            Dict con AUC(>=1), AUC(grado 2), Spearman y exactitud binaria.
        """
        return {
            "auc_ge1": auc_seguro(etiquetas_ge1, probs),
            "auc_grado2": auc_seguro(etiquetas_2, probs),
            "spearman": spearman_seguro(probs, grados),
            "acc_binaria_0_5": float(accuracy_score(etiquetas_ge1, (probs >= cfg["umbral"]).astype(int))),
            "f1_binaria_0_5": float(
                f1_score(etiquetas_ge1, (probs >= cfg["umbral"]).astype(int), pos_label=1, zero_division=0)
            ),
            "n": len(probs),
        }

    metricas_lora = metricas_gold(arreglo_lora)
    metricas_base = metricas_gold(arreglo_base)

    # nDCG@10 graduado por caso con la maquinaria del script 16.
    ranking_por_id = {}
    for id_caso in sorted({item["id"] for item in items}, key=modulo_16().clave_id):
        sub = [(probs_lora[i], items[i]["articulo"]) for i in range(len(items)) if items[i]["id"] == id_caso]
        sub.sort(key=lambda par: (-par[0], par[1]))
        ranking_por_id[id_caso] = [articulo for _, articulo in sub]
    gold_id = gold_por_caso(items)
    por_caso_ndcg, n_fuera_pool = modulo_16().evaluar_ruta(gold_id, ranking_por_id, "completo")
    resumen_ndcg = modulo_16().resumir_metricas_ruta(por_caso_ndcg, logger)

    delta_auc = bootstrap_pareado_grupos(
        etiquetas_ge1,
        arreglo_lora,
        arreglo_base,
        casos,
        lambda etiquetas, scores: auc_seguro(etiquetas, scores),
        cfg["bootstrap"],
        cfg["semilla"],
        cfg["max_intentos_bootstrap"],
    )
    delta_spearman = bootstrap_pareado_grupos(
        grados,
        arreglo_lora,
        arreglo_base,
        casos,
        lambda etiquetas, scores: spearman_seguro(scores, etiquetas),
        cfg["bootstrap"],
        cfg["semilla"],
        cfg["max_intentos_bootstrap"],
    )

    referencias = cargar_referencias_llm(cfg, rutas, gold["items"], logger)
    v13_gold = None
    ruta_eval_gold = Path(rutas["ruta_gold_evaluacion"])
    if ruta_eval_gold.exists():
        with open(ruta_eval_gold, encoding="utf-8") as manejador:
            evaluacion = json.load(manejador)
        try:
            v13_gold = evaluacion["anotadores"]["carlos"]["rutas"]["b2_v13"]["variantes"]["completo"]["metricas"]["ndcg_at_10"]["media"]
        except (KeyError, TypeError):
            v13_gold = None

    criterios = cfg["criterios_s3"]
    cumple = {
        "auc_ge1": metricas_lora["auc_ge1"] is not None and metricas_lora["auc_ge1"] >= criterios["auc_ge1"],
        "spearman": metricas_lora["spearman"] is not None and metricas_lora["spearman"] >= criterios["spearman"],
        "delta_auc_vs_base": delta_auc["gana"] and delta_auc["media_delta"] >= criterios["delta_auc_vs_base"],
    }
    cierre = {
        "n_items": len(items),
        "n_casos": len({item["id"] for item in items}),
        "distribucion_grados": {str(g): int((grados == g).sum()) for g in (0, 1, 2)},
        "lora": {
            **metricas_lora,
            "ndcg_at_10_graduado": resumen_ndcg["ndcg_at_10"],
            "p_at_10_binario": resumen_ndcg["p_at_10"],
            "r_at_10_binario": resumen_ndcg["r_at_10"],
            "n_items_fuera_pool": n_fuera_pool,
        },
        "base_bf16": metricas_base,
        "delta_lora_vs_base": {"auc_ge1": delta_auc, "spearman": delta_spearman},
        "referencias": referencias,
        "v13_gold_ndcg_at_10": v13_gold,
        "ajuste_prompt": {
            "n_ajustados": int(sum(ajustes)),
            "max_chars_fragmento": cfg["max_chars_fragmento_gold"],
            "max_chars_pasaje": cfg["max_chars_pasaje"],
        },
        "criterios_s3": {
            "umbrales": criterios,
            "cumple": cumple,
            "resultado": "GO" if all(cumple.values()) else "NO_CUMPLE",
        },
    }
    logger.info(
        "nivel 2: lora auc_ge1=%s auc_2=%s spearman=%s | base auc_ge1=%s | delta_auc=%+.4f [%.4f, %.4f]",
        metricas_lora["auc_ge1"],
        metricas_lora["auc_grado2"],
        metricas_lora["spearman"],
        metricas_base["auc_ge1"],
        delta_auc["media_delta"],
        delta_auc["p2_5"],
        delta_auc["p97_5"],
    )
    dibujar_nivel2(cierre, probs_lora, [item["grado"] for item in items], rutas["ruta_figura_gold"])
    return cierre


def dibujar_nivel2(cierre: dict, scores_lora: list[float], grados: list[int], ruta: str | Path) -> None:
    """Figura del nivel 2: scores por grado y curva ROC (grado >= 1).

    Args:
        cierre: Cierre del nivel 2.
        scores_lora: P("Si") por item.
        grados: Grado humano por item.
        ruta: Ruta de la figura.
    """
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    arreglo = np.asarray(scores_lora, dtype=np.float64)
    grados_np = np.asarray(grados, dtype=np.int64)
    grupos = [arreglo[grados_np == g] for g in (0, 1, 2)]
    etiquetas = [f"grado {g} (n={len(valores)})" for g, valores in zip((0, 1, 2), grupos, strict=True)]
    ejes[0].boxplot(grupos, tick_labels=etiquetas)
    ejes[0].set_ylabel('P("Si")')
    ejes[0].set_title("Nivel 2 - gold humano B3 (820 items)")
    etiquetas_ge1 = (grados_np >= 1).astype(int)
    if len(np.unique(etiquetas_ge1)) > 1:
        fpr, tpr, _ = roc_curve(etiquetas_ge1, arreglo)
        ejes[1].plot(fpr, tpr, color="#d62728", label=f"AUC(>=1)={cierre['lora']['auc_ge1']:.3f}")
        ejes[1].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ejes[1].set_xlabel("FPR")
        ejes[1].set_ylabel("TPR")
        ejes[1].set_title("ROC - grado humano >= 1")
        ejes[1].legend()
    figura.tight_layout()
    figura.savefig(destino, dpi=130)
    plt.close(figura)


# ===== SECCION 7: nivel 3 - rerank end-to-end (E3) =====


def cargar_benchmark(rutas: dict, variante: str, logger: logging.Logger) -> dict:
    """Carga el benchmark M7 (muestra, top-10 y referencias congeladas).

    Args:
        rutas: Rutas absolutas.
        variante: "v13" o "v14_ep2".
        logger: Logger activo.

    Returns:
        Dict con archivo, datos y estrategia.

    Raises:
        FileNotFoundError: Si no existe el JSON.
    """
    ruta = rutas["rutas_benchmark"][variante]
    _exigir(Path(ruta).exists(), f"no existe el benchmark {ruta}")
    with open(ruta, encoding="utf-8") as manejador:
        datos = json.load(manejador)
    _exigir(len(datos.get("muestra", [])) == 90, f"muestra del benchmark inesperada: {len(datos.get('muestra', []))}")
    _exigir(datos.get("top10_por_caso"), "benchmark sin top10_por_caso")
    logger.info("benchmark %s cargado: version=%s", ruta, datos.get("version"))
    return {"archivo": ruta, "md5": md5_archivo(ruta), "variante": variante, "datos": datos}


def referencias_v13_congeladas(bench: dict, estrategia: str, contexto: str) -> dict[str, dict]:
    """Extrae las metricas por caso congeladas de la ruta v13.

    Args:
        bench: Benchmark cargado (cargar_benchmark).
        estrategia: Estrategia de producto.
        contexto: "con_hubs" o "sin_hubs".

    Returns:
        Dict id_caso -> {"ndcg_at_10", "f1_at_10"}.

    Raises:
        RuntimeError: Si el benchmark no trae la referencia.
    """
    referencia = bench["datos"].get("comparacion_b2", {}).get("metricas_por_caso", {}).get(contexto, {}).get(estrategia)
    _exigir(referencia is not None, f"benchmark sin metricas_por_caso[{contexto}][{estrategia}]")
    filas = {}
    for indice, caso in enumerate(bench["datos"]["muestra"]):
        filas[str(caso["id"])] = {
            "ndcg_at_10": float(referencia["ndcg_at_10"][indice]),
            "f1_at_10": float(referencia["f1_at_10"][indice]),
        }
    return filas


def referencias_c2_zero_shot(rutas: dict, logger: logging.Logger) -> dict[str, dict]:
    """Carga las metricas por caso del C2 zero-shot listwise congelado.

    Args:
        rutas: Rutas absolutas.
        logger: Logger activo.

    Returns:
        Dict id_caso -> {"ndcg_at_10", "f1_at_10"}; vacio si no existe.
    """
    ruta = Path(rutas["ruta_eval_rerank_llm"])
    if not ruta.exists():
        logger.warning("sin %s: no hay referencia C2 zero-shot por caso", ruta)
        return {}
    with open(ruta, encoding="utf-8") as manejador:
        evaluacion = json.load(manejador)
    filas = {}
    for fila in evaluacion.get("evaluacion", {}).get("por_caso", []):
        metricas = fila.get("metricas_llm", {})
        if "ndcg_at_10" in metricas:
            filas[str(fila["id"])] = {
                "ndcg_at_10": float(metricas["ndcg_at_10"]),
                "f1_at_10": float(metricas["f1_at_10"]),
            }
    logger.info("referencia C2 zero-shot: %d casos", len(filas))
    return filas


def medir_ruta(
    mod12: object,
    mod30: object,
    muestra: list[dict],
    ranking_por_id: dict[str, list[str]],
) -> list[dict]:
    """Calcula las seis medidas C2 por caso para un ranking por caso.

    Args:
        mod12: Modulo 12.
        mod30: Modulo 30.
        muestra: Casos del benchmark (id, arts_cp).
        ranking_por_id: Dict id_caso -> ranking (<= 10).

    Returns:
        Lista de filas por caso (id, ano, tipo, n_relevantes, medidas).
    """
    filas = []
    for caso in muestra:
        id_caso = str(caso["id"])
        fila = {
            "id": id_caso,
            "ano": int(caso["ano"]),
            "tipo": caso["tipo"],
            "n_relevantes": len(caso["arts_cp"]),
        }
        fila.update(mod30.medidas_caso(mod12, caso["arts_cp"], ranking_por_id.get(id_caso, [])))
        filas.append(fila)
    return filas


def ranking_con_prior_local(mod12: object, scores: np.ndarray, canones: list, prior: dict, alpha: float, k: int) -> list[str]:
    """Aplica el prior de hubs y trunca el ranking a k (ruta de producto).

    Args:
        mod12: Modulo 12.
        scores: Vector de scores (con -inf fuera de candidatos).
        canones: Canones alineados con scores.
        prior: Prior de hubs.
        alpha: Peso del prior (1,5 en la ruta v13).
        k: Profundidad de salida (10).

    Returns:
        Ranking de canones (<= k).
    """
    priorizado = mod12.aplicar_prior_scores(scores, canones, prior, alpha)
    return mod12.ranking_desde_scores(priorizado, canones)[:k]


def fusionar_scores(scores_b1: np.ndarray, scores_lora: np.ndarray, alpha: float) -> np.ndarray:
    """Fusiona min-max los scores de B1 y del juez sobre los candidatos.

    Args:
        scores_b1: Scores del reranker B1 (o del top-10 congelado).
        scores_lora: Scores del juez LoRA.
        alpha: Peso de B1 (1-alpha para el juez).

    Returns:
        Vector fusionado (con -inf fuera de candidatos comunes).
    """
    indice = np.flatnonzero(np.isfinite(scores_b1) & np.isfinite(scores_lora))
    fusionado = np.full(len(scores_b1), -np.inf, dtype=np.float64)
    if indice.size == 0:
        return fusionado
    b1 = scores_b1[indice]
    lora = scores_lora[indice]
    b1_norm = (b1 - b1.min()) / (b1.max() - b1.min()) if b1.max() > b1.min() else np.zeros_like(b1)
    lora_norm = (lora - lora.min()) / (lora.max() - lora.min()) if lora.max() > lora.min() else np.zeros_like(lora)
    fusionado[indice] = alpha * b1_norm + (1.0 - alpha) * lora_norm
    return fusionado


def resumir_ruta_nivel3(
    mod12: object,
    mod30: object,
    muestra: list[dict],
    filas: list[dict],
    latencias: list[float],
    n_bootstrap: int,
    semilla: int,
) -> dict:
    """Resume una ruta del nivel 3: medias con IC, por caso y latencias.

    Args:
        mod12: Modulo 12.
        mod30: Modulo 30.
        muestra: Casos del benchmark.
        filas: Filas por caso (medir_ruta).
        latencias: Latencia por caso del scoring del juez (segundos).
        n_bootstrap: Remuestreos bootstrap.
        semilla: Semilla del bootstrap.

    Returns:
        Dict con n, medidas (media + IC), metricas_por_caso y latencias.
    """
    agregado = mod30.agregar_medidas(mod12, filas, n_bootstrap, semilla)
    resumen_latencia = percentiles(latencias)
    return {
        "n": len(filas),
        "medidas": agregado["medidas"],
        "metricas_por_caso": filas,
        "latencias_seg": {**resumen_latencia, "gate_p95_seg": CONFIG["gate_p95_seg"]},
        "gate_p95_cumple": resumen_latencia["p95"] is not None
        and resumen_latencia["p95"] <= CONFIG["gate_p95_seg"],
    }


def comparar_rutas(
    filas_a: list[dict],
    filas_b: list[dict],
    n_bootstrap: int,
    semilla: int,
) -> dict:
    """Compara dos rutas pareadas por caso (nDCG@10 y F1@10).

    Args:
        filas_a: Filas por caso de la ruta nueva.
        filas_b: Filas por caso de la referencia.
        n_bootstrap: Remuestreos bootstrap.
        semilla: Semilla del bootstrap.

    Returns:
        Dict medida -> IC pareado (media_delta, p2_5, p97_5, prob_mejora,
        gana) y n_casos_comunes.

    Raises:
        ValueError: Si las rutas no estan alineadas por id.
    """
    ids_a = [fila["id"] for fila in filas_a]
    ids_b = [fila["id"] for fila in filas_b]
    if ids_a != ids_b:
        raise ValueError("rutas desalineadas por id en la comparacion pareada")
    medidas = {}
    for medida in ("ndcg_at_10", "f1_at_10"):
        medidas[medida] = bootstrap_pareado_casos(
            np.asarray([fila[medida] for fila in filas_a], dtype=np.float64),
            np.asarray([fila[medida] for fila in filas_b], dtype=np.float64),
            n_bootstrap,
            semilla,
        )
    return {"n_casos_comunes": len(ids_a), "medidas": medidas}


def consistencia_ab(filas_a: list[dict], filas_ref: list[dict], etiqueta_a: str) -> dict:
    """Consistencia A/B por mitades de casos (orden de la muestra).

    Args:
        filas_a: Filas por caso de la ruta nueva (mismo orden que la ref).
        filas_ref: Filas por caso de la referencia.
        etiqueta_a: Nombre de la ruta nueva (para el reporte).

    Returns:
        Dict con delta nDCG@10 por mitad y veredicto de consistencia.
    """
    n = len(filas_a)
    mitad = n // 2
    resultado = {"n_por_mitad": [mitad, n - mitad]}
    for nombre, rango in (("mitad_a", range(mitad)), ("mitad_b", range(mitad, n))):
        deltas = [
            filas_a[i]["ndcg_at_10"] - filas_ref[i]["ndcg_at_10"] for i in rango
        ]
        resultado[nombre] = {"media_delta_ndcg_at_10": round(float(np.mean(deltas)), 4) if deltas else None}
    delta_a = resultado["mitad_a"]["media_delta_ndcg_at_10"]
    delta_b = resultado["mitad_b"]["media_delta_ndcg_at_10"]
    resultado["consistente"] = bool(
        delta_a is not None and delta_b is not None and delta_a > 0 and delta_b > 0
    )
    resultado["etiqueta"] = etiqueta_a
    return resultado


def pares_para_caso(
    mod12: object,
    scores_candidatos: np.ndarray,
    caso: dict,
    pasajes,
    grupos_indices: list,
    k_candidatos: int,
) -> dict:
    """Construye los pares (chunk, pasaje) de los candidatos de un caso.

    Delega en la maquinaria del script 12 (mejor pasaje BM25 del candidato +
    top-3 chunks de la query; identico a la inferencia B1).

    Args:
        mod12: Modulo 12.
        scores_candidatos: Scores que ordenan a los candidatos.
        caso: Caso con matriz_bm25 y chunks_bm25.
        pasajes: Pasajes de la CP.
        grupos_indices: Grupos de pasajes por articulo.
        k_candidatos: Numero de candidatos.

    Returns:
        Dict con candidatos, queries, pasajes, posiciones.
    """
    return mod12.construir_pares_rerank_chunks(
        scores_candidatos,
        caso["matriz_bm25"],
        caso["chunks_bm25"],
        grupos_indices,
        pasajes,
        k_candidatos,
    )


def prompts_desde_pares(
    tokenizer,
    cfg: dict,
    pares: dict,
    canones: list,
) -> tuple[list[list[int]], list[str], list[bool]]:
    """Renderiza los prompts de un lote de pares (chunk, pasaje).

    Args:
        tokenizer: Tokenizer.
        cfg: Configuracion efectiva.
        pares: Salida de construir_pares_rerank_chunks.
        canones: Canones alineados con pares["candidatos"].

    Returns:
        Tupla (prompts_ids, articulos_por_par, ajustes).
    """
    prompts = []
    articulos = []
    ajustes = []
    for posicion, query, pasaje in zip(pares["posiciones"], pares["queries"], pares["pasajes"], strict=True):
        articulo = canones[pares["candidatos"][posicion]]
        render = render_con_presupuesto(
            tokenizer,
            cfg,
            query,
            articulo,
            pasaje,
            max_chars_fragmento=cfg["max_chars_chunk_benchmark"],
        )
        prompts.append(render["prompt_ids"])
        articulos.append(articulo)
        ajustes.append(render["ajustado"])
    return prompts, articulos, ajustes


def evaluar_nivel3_fallback(
    modelo,
    tokenizer,
    cfg: dict,
    rutas: dict,
    device: str,
    ids_ctx: dict,
    cache: CacheScores,
    bench: dict,
    logger: logging.Logger,
) -> dict:
    """Nivel 3 con el top-10 congelado del benchmark (sin recomputar B2/B1).

    Reordena el top-10 de la ruta v13 con el juez LoRA (pares reconstruidos
    por BM25, idénticos a B1), con control zero-shot bf16 y fusion con los
    scores B1 guardados. Limitacion documentada: no hay top-50.

    Args:
        modelo: PeftModel.
        tokenizer: Tokenizer.
        cfg: Configuracion efectiva.
        rutas: Rutas absolutas.
        device: "cuda" o "cpu".
        ids_ctx: Ids de respuesta en contexto.
        cache: Cache de scores.
        bench: Benchmark cargado.
        logger: Logger activo.

    Returns:
        Dict de cierre del nivel 3 (fuente de candidatos = json_top10).
    """
    mod12 = modulo_12()
    mod30 = modulo_30()
    datos = bench["datos"]
    estrategia = cfg["estrategia_v13"]
    muestra = datos["muestra"]
    if cfg["max_casos_benchmark"]:
        muestra = muestra[: cfg["max_casos_benchmark"]]
    ids = [str(caso["id"]) for caso in muestra]
    logger.info("nivel 3 (json_top10): %d casos | estrategia=%s", len(muestra), estrategia)

    textos = mod30.leer_textos_por_id(mod12, Path(rutas["ruta_corpus"]), ids)
    pasajes, canones, grupos_indices, canon_valido = mod12.preparar_pasajes_constitucion(False)
    posicion_canon = {canon: i for i, canon in enumerate(canones)}
    prior = mod12.cargar_o_construir_prior_hubs()
    indice_bm25 = mod12.construir_indice_bm25(pasajes)
    logger.info(
        "CP: %d canones indexados / %d pasajes | %d canones validos | prior %d",
        len(canones),
        len(pasajes),
        len(canon_valido),
        len(prior),
    )

    rankings = {nombre: {} for nombre in ("c2_lora_top10", "base_bf16_top10")}
    for alpha in cfg["alphas_fusion"]:
        rankings[f"c2_lora_fusion_b1_top10_a{alpha:g}"] = {}
    latencias = {"c2_lora_top10": [], "base_bf16_top10": []}
    faltantes_canon = 0
    ajustados = 0

    for caso in muestra:
        id_caso = str(caso["id"])
        top = datos["top10_por_caso"][id_caso][estrategia]
        ventana = textos[id_caso]["texto_limpio"][: mod12.CONFIG["ventana_query_chars"]]
        chunks = mod12.chunks_query(ventana)
        _, matriz = mod12.scores_bm25_caso(indice_bm25, grupos_indices, canones, chunks)
        scores_orden = np.full(len(canones), -np.inf, dtype=np.float64)
        scores_b1 = np.full(len(canones), -np.inf, dtype=np.float64)
        for posicion, item in enumerate(top):
            canon = modulo_16().normalizar_articulo(item["id"])
            indice = posicion_canon.get(canon)
            if indice is None:
                faltantes_canon += 1
                continue
            scores_orden[indice] = -float(posicion)
            scores_b1[indice] = float(item["score"])
        caso_bm25 = {"matriz_bm25": matriz, "chunks_bm25": chunks}
        pares = pares_para_caso(mod12, scores_orden, caso_bm25, pasajes, grupos_indices, len(top))
        prompts, _, ajustes_caso = prompts_desde_pares(tokenizer, cfg, pares, canones)
        ajustados += int(sum(ajustes_caso))
        inicio = time.perf_counter()
        probs_lora = puntuar_prompts(modelo, tokenizer, prompts, "lora", "benchmark", cfg, device, ids_ctx, cache, logger)
        latencias["c2_lora_top10"].append(time.perf_counter() - inicio)
        with modelo.disable_adapter():
            probs_base = puntuar_prompts(modelo, tokenizer, prompts, "base", "benchmark", cfg, device, ids_ctx, cache, logger)
        scores_lora = mod12.scores_maxpool_por_candidato(
            np.asarray(probs_lora, dtype=np.float64), pares["posiciones"], pares["candidatos"], len(canones)
        )
        scores_base = mod12.scores_maxpool_por_candidato(
            np.asarray(probs_base, dtype=np.float64), pares["posiciones"], pares["candidatos"], len(canones)
        )
        rankings["c2_lora_top10"][id_caso] = ranking_con_prior_local(
            mod12, scores_lora, canones, prior, cfg["alpha_hubs"], cfg["k_salida"]
        )
        rankings["base_bf16_top10"][id_caso] = ranking_con_prior_local(
            mod12, scores_base, canones, prior, cfg["alpha_hubs"], cfg["k_salida"]
        )
        for alpha in cfg["alphas_fusion"]:
            fusionado = fusionar_scores(scores_b1, scores_lora, alpha)
            rankings[f"c2_lora_fusion_b1_top10_a{alpha:g}"][id_caso] = ranking_con_prior_local(
                mod12, fusionado, canones, prior, cfg["alpha_hubs"], cfg["k_salida"]
            )
    if faltantes_canon:
        logger.warning("nivel 3: %d candidatos del top-10 sin canon en el indice", faltantes_canon)
    logger.info("nivel 3: prompts ajustados por presupuesto: %d", ajustados)

    filas_por_ruta = {
        nombre: medir_ruta(mod12, mod30, muestra, ranking_por_id)
        for nombre, ranking_por_id in rankings.items()
    }
    latencias_por_ruta = {
        nombre: latencias.get(nombre, latencias["c2_lora_top10"]) for nombre in rankings
    }
    resumen = {
        nombre: resumir_ruta_nivel3(
            mod12,
            mod30,
            muestra,
            filas,
            latencias_por_ruta[nombre],
            cfg["bootstrap"],
            cfg["semilla"],
        )
        for nombre, filas in filas_por_ruta.items()
    }
    return armar_cierre_nivel3(
        cfg, rutas, bench, "json_top10", resumen, filas_por_ruta, faltantes_canon, logger
    )


def precomputar_candidatos_top50(cfg: dict, rutas: dict, bench: dict, logger: logging.Logger) -> dict:
    """Recomputa la 1a etapa (hibrido top-50 + rerank B1) SIN el 7B cargado.

    Fase GPU (B2 + B1) que se ejecuta ANTES de cargar el juez de 7B: el
    presupuesto de 16 GB no admite 7B (14,5 GiB) + B2 + B1 a la vez. Deja
    en RAM todo lo necesario para el scoring posterior (casos con
    scores_bm25, scores hibridos y scores B1, pares top-50 ya construidos,
    pasajes, canones, grupos y prior) y libera B2 y B1 antes de devolver.

    Mapeo de interfaz: construir_casos_ruta del script 30 espera la clave
    "arts" en cada caso de la muestra; la muestra del benchmark usa
    "arts_cp", de modo que se adapta aqui sin modificar el script 30.

    Args:
        cfg: Configuracion efectiva.
        rutas: Rutas absolutas.
        bench: Benchmark cargado.
        logger: Logger activo.

    Returns:
        Dict con casos, clave_hibrido, pasajes, canones, grupos_indices,
        prior, pares50 (id -> pares), muestra, stats_b2 y dispositivo.
    """
    mod12 = modulo_12()
    mod30 = modulo_30()
    datos = bench["datos"]
    muestra = datos["muestra"]
    if cfg["max_casos_benchmark"]:
        muestra = muestra[: cfg["max_casos_benchmark"]]
    ids = [str(caso["id"]) for caso in muestra]
    k_top50 = cfg["k_candidatos_top50"]
    logger.info(
        "nivel 3 (recomputado): precomputando candidatos top-%d de %d casos con B2 + B1 (7B aun sin cargar)",
        k_top50,
        len(muestra),
    )
    textos = mod30.leer_textos_por_id(mod12, Path(rutas["ruta_corpus"]), ids)
    pasajes, canones, grupos_indices, canon_valido = mod12.preparar_pasajes_constitucion(False)
    prior = mod12.cargar_o_construir_prior_hubs()
    logger.info(
        "CP: %d canones indexados / %d pasajes | %d canones validos | prior %d",
        len(canones),
        len(pasajes),
        len(canon_valido),
        len(prior),
    )
    muestra_ruta = [
        {"id": caso["id"], "tipo": caso["tipo"], "ano": caso["ano"], "arts": list(caso["arts_cp"])}
        for caso in muestra
    ]
    casos = mod30.construir_casos_ruta(mod12, muestra_ruta, textos, {}, usar_consulta_llm=False)
    mod30.calcular_bm25_ruta(mod12, casos, pasajes, canones, grupos_indices)

    from lib.reranker_bge import RerankerBGE, elegir_dispositivo

    dispositivo = elegir_dispositivo()
    modelo_b2 = mod12.resolver_modelo_b2(None)
    modelo_b1 = mod12.resolver_modelo_b1(None)
    mod12.aplicar_modo_b2(None, modelo_b2, modelo_b1)
    stats_b2, clave_hibrido = mod30.calcular_denso_ruta(mod12, casos, pasajes, canones, grupos_indices, dispositivo)
    logger.info("B2 recomputado: %s", stats_b2.get("slug", "n/a"))

    reranker = RerankerBGE(modelo_id=modelo_b1, dispositivo=dispositivo)
    try:
        for caso in casos:
            _, scores_b1, _ = mod12.ranking_rerank_chunks(
                reranker,
                caso[clave_hibrido],
                caso["matriz_bm25"],
                caso["chunks_bm25"],
                canones,
                pasajes,
                grupos_indices,
                k_top50,
            )
            caso["scores_b1"] = scores_b1
    finally:
        del reranker
        if dispositivo == "cuda":
            torch.cuda.empty_cache()

    pares50 = {
        str(caso["id"]): pares_para_caso(
            mod12, caso[clave_hibrido], caso, pasajes, grupos_indices, k_top50
        )
        for caso in casos
    }
    # Casos "slim": se descartan texto crudo/limpio y ventana (grandes y ya
    # consumidos por los pares) para que el pickle del precomputo sea liviano.
    claves_slim = ("id", "tipo", "ano", "arts", "matriz_bm25", "chunks_bm25", "scores_bm25", "scores_b1")
    casos_slim = []
    for caso in casos:
        slim = {clave: caso[clave] for clave in claves_slim if clave in caso}
        slim[clave_hibrido] = caso[clave_hibrido]
        casos_slim.append(slim)
    logger.info(
        "precomputo listo: %d casos, %d pares/caso | B2 y B1 liberados antes de cargar el 7B",
        len(casos_slim),
        k_top50 * mod12.CONFIG["k_chunks_rerank"],
    )
    return {
        "meta": {
            "variante_benchmark": bench["variante"],
            "n_casos": len(casos_slim),
            "ids": [str(caso["id"]) for caso in casos_slim],
            "k_candidatos_top50": k_top50,
            "clave_hibrido": clave_hibrido,
            "stats_b2": stats_b2,
            "dispositivo_b2_b1": dispositivo,
            "estrategia": cfg["estrategia_v13"],
        },
        "casos": casos_slim,
        "clave_hibrido": clave_hibrido,
        "pasajes": pasajes,
        "canones": canones,
        "grupos_indices": grupos_indices,
        "prior": prior,
        "pares50": pares50,
        "muestra": muestra,
        "stats_b2": stats_b2,
        "dispositivo_b2_b1": dispositivo,
    }


def guardar_candidatos_pickle(ruta: str | Path, precomputado: dict, logger: logging.Logger) -> dict:
    """Escribe el precomputo de candidatos a disco (pickle atomico).

    Args:
        ruta: Ruta destino del pickle.
        precomputado: Salida de precomputar_candidatos_top50.
        logger: Logger activo.

    Returns:
        Dict con ruta, bytes y md5 del archivo escrito.
    """
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.parent / (destino.name + ".tmp")
    with open(temporal, "wb") as manejador:
        pickle.dump(precomputado, manejador, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporal, destino)
    info = {"ruta": str(destino), "bytes": destino.stat().st_size, "md5": md5_archivo(destino)}
    logger.info(
        "candidatos top-50 guardados en %s (%.1f MB, %d casos)",
        destino,
        info["bytes"] / 1024**2,
        precomputado["meta"]["n_casos"],
    )
    return info


def cargar_candidatos_pickle(ruta: str | Path, logger: logging.Logger) -> dict:
    """Carga el precomputo de candidatos desde disco.

    Args:
        ruta: Ruta del pickle.
        logger: Logger activo.

    Returns:
        Dict del precomputo.

    Raises:
        FileNotFoundError: Si no existe el archivo.
    """
    archivo = Path(ruta)
    if not archivo.exists():
        raise FileNotFoundError(f"no existe el precomputo de candidatos {archivo}")
    with open(archivo, "rb") as manejador:
        precomputado = pickle.load(manejador)
    logger.info(
        "candidatos top-50 cargados de %s (%d casos, variante %s)",
        archivo,
        precomputado["meta"]["n_casos"],
        precomputado["meta"]["variante_benchmark"],
    )
    return precomputado


def obtener_candidatos_top50(cfg: dict, rutas: dict, args: argparse.Namespace, logger: logging.Logger) -> dict:
    """Obtiene el precomputo top-50: cache en disco o subproceso GPU.

    El precomputo (B2 + B1) corre en un SUBPROCESO: al terminar, el proceso
    libera toda la VRAM (ROCm no devuelve del todo la memoria del proceso
    padre con empty_cache) y deja el resultado en un pickle, de modo que el
    juez de 7B se carga despues con la GPU practicamente limpia y una
    eventual reanudacion no repite B2/B1. Si el pickle existe y coincide en
    variante/numero de casos se reutiliza; --recomputar-candidatos fuerza
    la regeneracion.

    Args:
        cfg: Configuracion efectiva.
        rutas: Rutas absolutas.
        args: Namespace de la CLI.
        logger: Logger activo.

    Returns:
        Dict del precomputo (ver precomputar_candidatos_top50).

    Raises:
        RuntimeError: Si el subproceso falla o no deja el pickle.
    """
    ruta = Path(rutas["ruta_candidatos_top50"])
    n_casos_esperados = cfg["max_casos_benchmark"] or 90
    if ruta.exists() and not args.recomputar_candidatos:
        try:
            precomputado = cargar_candidatos_pickle(ruta, logger)
            meta = precomputado.get("meta", {})
            if (
                meta.get("variante_benchmark") == cfg["benchmark_variante"]
                and meta.get("n_casos") == n_casos_esperados
                and meta.get("k_candidatos_top50") == cfg["k_candidatos_top50"]
            ):
                return precomputado
            logger.warning(
                "el pickle de candidatos no coincide (variante=%s, n_casos=%s, k=%s); se regenera",
                meta.get("variante_benchmark"),
                meta.get("n_casos"),
                meta.get("k_candidatos_top50"),
            )
        except (pickle.UnpicklingError, EOFError, KeyError, AttributeError, OSError) as exc:
            logger.warning("pickle de candidatos ilegible (%s); se regenera", exc)
    comando = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--precomputar-top50",
        str(ruta),
        "--benchmark",
        cfg["benchmark_variante"],
    ]
    if cfg["max_casos_benchmark"]:
        comando += ["--max-casos", str(cfg["max_casos_benchmark"])]
    logger.info("precomputando candidatos top-50 en subproceso: %s", " ".join(comando))
    resultado = subprocess.run(comando, cwd=str(RAIZ), check=False)
    _exigir(resultado.returncode == 0, f"el subproceso de precomputo fallo (exit {resultado.returncode})")
    _exigir(ruta.exists(), f"el subproceso de precomputo no genero {ruta}")
    return cargar_candidatos_pickle(ruta, logger)


def evaluar_nivel3_recomputado(
    modelo,
    tokenizer,
    cfg: dict,
    rutas: dict,
    device: str,
    ids_ctx: dict,
    cache: CacheScores,
    bench: dict,
    precomputado: dict | None,
    logger: logging.Logger,
) -> dict:
    """Nivel 3 con candidatos top-50 precomputados (fase del juez 7B).

    Consume la salida de precomputar_candidatos_top50 (B2/B1 ya liberados)
    y puntua con el juez LoRA y el control base bf16 las rutas del plan:
    c2_lora_top50, c2_lora_cascada_b1_top20 (B1 top-50 -> top-20 -> juez)
    y c2_lora_fusion_b1 (rejilla alpha), mas v13 recalculado para la
    consistencia contra el per-case congelado.

    Args:
        modelo: PeftModel.
        tokenizer: Tokenizer.
        cfg: Configuracion efectiva.
        rutas: Rutas absolutas.
        device: "cuda" o "cpu".
        ids_ctx: Ids de respuesta en contexto.
        cache: Cache de scores.
        bench: Benchmark cargado.
        precomputado: Salida de precomputar_candidatos_top50 (obligatoria).
        logger: Logger activo.

    Returns:
        Dict de cierre del nivel 3 (fuente de candidatos = recomputado_top50).

    Raises:
        RuntimeError: Si falta el precomputo de candidatos.
    """
    _exigir(
        precomputado is not None,
        "el nivel 3 recomputado requiere precomputar_candidatos_top50 antes de cargar el 7B",
    )
    mod12 = modulo_12()
    mod30 = modulo_30()
    datos = bench["datos"]
    muestra = precomputado["muestra"]
    casos = precomputado["casos"]
    pasajes = precomputado["pasajes"]
    canones = precomputado["canones"]
    grupos_indices = precomputado["grupos_indices"]
    prior = precomputado["prior"]
    pares50_por_caso = precomputado["pares50"]
    estrategia = cfg["estrategia_v13"]
    logger.info(
        "nivel 3 (recomputado): scoring de %d casos con el juez (top-%d y cascada B1 top-%d)",
        len(casos),
        cfg["k_candidatos_top50"],
        cfg["k_candidatos_cascada"],
    )

    nombres_fusion = [f"c2_lora_fusion_b1_a{alpha:g}" for alpha in cfg["alphas_fusion"]]
    rankings = {
        nombre: {}
        for nombre in ("c2_lora_top50", "c2_lora_cascada_b1_top20", "base_bf16_top50", "v13_recalculado", *nombres_fusion)
    }
    latencias = {"c2_lora_top50": [], "c2_lora_cascada_b1_top20": [], "base_bf16_top50": []}
    # Tiempo por prompt sin cache por caso (para estimar la latencia de rutas
    # cuyos prompts salieron del cache; la cascada top-20 es subconjunto del
    # top-50 y normalmente no vuelve a puntuar).
    sp_lora_hist: list[float] = []
    sp_base_hist: list[float] = []
    ajustados = 0
    for caso in casos:
        id_caso = str(caso["id"])
        pares50 = pares50_por_caso[id_caso]
        prompts, _, ajustes_caso = prompts_desde_pares(tokenizer, cfg, pares50, canones)
        ajustados += int(sum(ajustes_caso))
        probs_lora, stats_lora = puntuar_prompts_con_stats(
            modelo, tokenizer, prompts, "lora", "benchmark", cfg, device, ids_ctx, cache, logger
        )
        with modelo.disable_adapter():
            probs_base, stats_base = puntuar_prompts_con_stats(
                modelo, tokenizer, prompts, "base", "benchmark", cfg, device, ids_ctx, cache, logger
            )
        sp_lora = stats_lora["segundos_por_prompt_sin_cache"] or (
            sum(sp_lora_hist) / len(sp_lora_hist) if sp_lora_hist else 0.0
        )
        sp_base = stats_base["segundos_por_prompt_sin_cache"] or (
            sum(sp_base_hist) / len(sp_base_hist) if sp_base_hist else 0.0
        )
        sp_lora_hist.append(sp_lora)
        sp_base_hist.append(sp_base)
        latencias["c2_lora_top50"].append(stats_lora["n"] * sp_lora)
        latencias["base_bf16_top50"].append(stats_base["n"] * sp_base)
        scores_lora = mod12.scores_maxpool_por_candidato(
            np.asarray(probs_lora, dtype=np.float64), pares50["posiciones"], pares50["candidatos"], len(canones)
        )
        scores_base = mod12.scores_maxpool_por_candidato(
            np.asarray(probs_base, dtype=np.float64), pares50["posiciones"], pares50["candidatos"], len(canones)
        )
        scores_b1 = caso["scores_b1"]
        rankings["c2_lora_top50"][id_caso] = ranking_con_prior_local(
            mod12, scores_lora, canones, prior, cfg["alpha_hubs"], cfg["k_salida"]
        )
        rankings["base_bf16_top50"][id_caso] = ranking_con_prior_local(
            mod12, scores_base, canones, prior, cfg["alpha_hubs"], cfg["k_salida"]
        )
        ranking_v13, _ = mod12.ranking_con_prior(
            scores_b1, canones, estrategia, prior, cfg["alpha_hubs"]
        )
        rankings["v13_recalculado"][id_caso] = ranking_v13[: cfg["k_salida"]]
        pares20 = pares_para_caso(
            mod12, scores_b1, caso, pasajes, grupos_indices, cfg["k_candidatos_cascada"]
        )
        prompts20, _, ajustes20 = prompts_desde_pares(tokenizer, cfg, pares20, canones)
        ajustados += int(sum(ajustes20))
        probs_lora20, stats_lora20 = puntuar_prompts_con_stats(
            modelo, tokenizer, prompts20, "lora", "benchmark", cfg, device, ids_ctx, cache, logger
        )
        sp20 = stats_lora20["segundos_por_prompt_sin_cache"] or sp_lora
        latencias["c2_lora_cascada_b1_top20"].append(stats_lora20["n"] * sp20)
        scores_lora20 = mod12.scores_maxpool_por_candidato(
            np.asarray(probs_lora20, dtype=np.float64), pares20["posiciones"], pares20["candidatos"], len(canones)
        )
        rankings["c2_lora_cascada_b1_top20"][id_caso] = ranking_con_prior_local(
            mod12, scores_lora20, canones, prior, cfg["alpha_hubs"], cfg["k_salida"]
        )
        for alpha, nombre in zip(cfg["alphas_fusion"], nombres_fusion, strict=True):
            fusionado = fusionar_scores(scores_b1, scores_lora, alpha)
            rankings[nombre][id_caso] = ranking_con_prior_local(
                mod12, fusionado, canones, prior, cfg["alpha_hubs"], cfg["k_salida"]
            )
    logger.info("nivel 3 recomputado: prompts ajustados por presupuesto: %d", ajustados)

    filas_por_ruta = {
        nombre: medir_ruta(mod12, mod30, muestra, ranking_por_id)
        for nombre, ranking_por_id in rankings.items()
    }
    latencias_por_ruta = {
        nombre: latencias.get(nombre, latencias["c2_lora_top50"]) for nombre in rankings
    }
    resumen = {
        nombre: resumir_ruta_nivel3(
            mod12,
            mod30,
            muestra,
            filas,
            latencias_por_ruta[nombre],
            cfg["bootstrap"],
            cfg["semilla"],
        )
        for nombre, filas in filas_por_ruta.items()
    }
    consistencia = mod30.consistencia_congelada(
        datos, filas_por_ruta["v13_recalculado"], estrategia, cfg["contexto_v13"]
    )
    cierre = armar_cierre_nivel3(
        cfg, rutas, bench, "recomputado_top50", resumen, filas_por_ruta, 0, logger
    )
    cierre["consistencia_v13_recalculado"] = consistencia
    cierre["latencias_nota"] = (
        "latencia por caso estimada como n_prompts_de_la_ruta x tiempo por prompt medido sin cache "
        "(la cascada top-20 reutiliza prompts del top-50 via cache de scores; la fusion comparte la "
        "latencia del top-50; el scoring de la cascada fue "
        f"{'desde cache' if not any(l > 0 for l in latencias['c2_lora_cascada_b1_top20']) else 'medido'})"
    )
    logger.info(
        "consistencia v13 recalculado vs congelado: %s",
        consistencia.get("consistente"),
    )
    return cierre


def armar_cierre_nivel3(
    cfg: dict,
    rutas: dict,
    bench: dict,
    fuente: str,
    resumen: dict,
    filas_por_ruta: dict,
    faltantes_canon: int,
    logger: logging.Logger,
) -> dict:
    """Arma el cierre del nivel 3: comparaciones, A/B, referencias y S4.

    Args:
        cfg: Configuracion efectiva.
        rutas: Rutas absolutas.
        bench: Benchmark cargado.
        fuente: "json_top10" o "recomputado_top50".
        resumen: Resumen por ruta (resumir_ruta_nivel3).
        filas_por_ruta: Filas por caso por ruta.
        faltantes_canon: Candidatos sin canon (trazabilidad).
        logger: Logger activo.

    Returns:
        Dict de cierre del nivel 3.
    """
    datos = bench["datos"]
    estrategia = cfg["estrategia_v13"]
    refs_v13 = referencias_v13_congeladas(bench, estrategia, cfg["contexto_v13"])
    refs_c2 = referencias_c2_zero_shot(rutas, logger)

    if fuente == "json_top10":
        ruta_lora = "c2_lora_top10"
        ruta_base = "base_bf16_top10"
        ruta_fusion_primaria = f"c2_lora_fusion_b1_top10_a{cfg['alpha_fusion_primario']:g}"
    else:
        ruta_lora = "c2_lora_top50"
        ruta_base = "base_bf16_top50"
        ruta_fusion_primaria = f"c2_lora_fusion_b1_a{cfg['alpha_fusion_primario']:g}"
    rutas_adopcion = [
        nombre
        for nombre in resumen
        if nombre.startswith("c2_lora_cascada") or (nombre.startswith("c2_lora_fusion") and fuente == "recomputado_top50")
    ]
    if fuente == "json_top10":
        rutas_adopcion = [ruta_fusion_primaria]

    filas_v13 = [
        {"id": fila["id"], **refs_v13[fila["id"]]}
        for fila in filas_por_ruta[ruta_lora]
        if fila["id"] in refs_v13
    ]
    filas_lora = [fila for fila in filas_por_ruta[ruta_lora] if fila["id"] in refs_v13]
    comparaciones = {
        f"{ruta_lora}_vs_v13": comparar_rutas(filas_lora, filas_v13, cfg["bootstrap"], cfg["semilla"]),
        f"{ruta_lora}_vs_base_bf16": comparar_rutas(
            filas_por_ruta[ruta_lora], filas_por_ruta[ruta_base], cfg["bootstrap"], cfg["semilla"]
        ),
    }
    for nombre in rutas_adopcion:
        comparaciones[f"{nombre}_vs_v13"] = comparar_rutas(
            filas_por_ruta[nombre], filas_v13, cfg["bootstrap"], cfg["semilla"]
        )
    if refs_c2:
        filas_c2 = [
            {"id": fila["id"], **refs_c2[fila["id"]]}
            for fila in filas_por_ruta[ruta_lora]
            if fila["id"] in refs_c2
        ]
        filas_lora_c2 = [fila for fila in filas_por_ruta[ruta_lora] if fila["id"] in refs_c2]
        comparaciones[f"{ruta_lora}_vs_c2_zero_shot"] = comparar_rutas(
            filas_lora_c2, filas_c2, cfg["bootstrap"], cfg["semilla"]
        )
    ab_por_ruta = {
        nombre: consistencia_ab(
            [fila for fila in filas_por_ruta[nombre] if fila["id"] in refs_v13],
            filas_v13,
            nombre,
        )
        for nombre in rutas_adopcion
    }

    criterios = cfg["criterios_s4"]
    delta_base = comparaciones[f"{ruta_lora}_vs_base_bf16"]["medidas"]["ndcg_at_10"]
    cumple_adaptacion = delta_base["gana"] and delta_base["media_delta"] >= criterios["delta_ndcg_vs_base"]
    adopcion = {}
    for nombre in rutas_adopcion:
        delta_v13 = comparaciones[f"{nombre}_vs_v13"]["medidas"]["ndcg_at_10"]
        p95 = resumen[nombre]["latencias_seg"]["p95"]
        gate_latencia = p95 is not None and p95 <= criterios["gate_p95_seg"]
        cumple = {
            "delta_ndcg_vs_v13": delta_v13["gana"] and delta_v13["media_delta"] >= criterios["delta_ndcg_adopcion_vs_v13"],
            "ab_consistente": ab_por_ruta[nombre]["consistente"],
            "gate_p95": gate_latencia,
        }
        adopcion[nombre] = {
            "delta_vs_v13": delta_v13,
            "p95_seg": p95,
            "cumple": cumple,
            "resultado": "CANDIDATA_ADOPCION" if all(v for v in cumple.values() if v is not None) else "NO_ADOPCION",
        }

    referencias = dict(cfg["referencias_c2"])
    global_v13 = datos.get("comparacion_b2", {}).get("metricas_globales", {})
    if global_v13:
        referencias["v13_metricas_globales"] = global_v13
    ruta_metricas = Path(rutas["ruta_metricas"])
    if ruta_metricas.exists():
        with open(ruta_metricas, encoding="utf-8") as manejador:
            metricas_documentadas = json.load(manejador)
        for clave in ("c2_rerank_llm", "c2_query_llm"):
            entrada = metricas_documentadas.get(clave)
            if not entrada:
                continue
            pareada = entrada.get("evaluacion_pareada", {})
            referencias[f"documentado_{clave}"] = {
                "ndcg_at_10": (pareada.get("llm") or {}).get("ndcg_at_10"),
                "f1_at_10": (pareada.get("llm") or {}).get("f1_at_10"),
                "v13_ndcg_at_10": (pareada.get("v13") or {}).get("ndcg_at_10"),
                "fuente": str(ruta_metricas),
            }
    cierre = {
        "fuente_candidatos": fuente,
        "limitacion": LIMITACION_TOP50 if fuente == "json_top10" else None,
        "benchmark": {
            "archivo": bench["archivo"],
            "md5": bench["md5"],
            "variante": bench["variante"],
            "version": datos.get("version"),
            "n_casos": len(filas_por_ruta[ruta_lora]),
            "estrategia": estrategia,
            "contexto": cfg["contexto_v13"],
            "alpha_hubs": cfg["alpha_hubs"],
        },
        "rutas": resumen,
        "comparaciones": comparaciones,
        "v13_por_caso": filas_v13,
        "consistencia_ab": ab_por_ruta,
        "rutas_adopcion": rutas_adopcion,
        "adopcion": adopcion,
        "referencias_congeladas": referencias,
        "candidatos_sin_canon": faltantes_canon,
        "criterios_s4": {
            "umbrales": criterios,
            "adaptacion_funciona": cumple_adaptacion,
            "delta_ndcg_vs_base_bf16": delta_base,
            "resultado_adaptacion": "GO" if cumple_adaptacion else "NO_CUMPLE",
        },
    }
    n_casos = len(filas_v13)
    ndcg_v13 = float(np.mean([fila["ndcg_at_10"] for fila in filas_v13])) if filas_v13 else float("nan")
    logger.info(
        "nivel 3: %s nDCG@10=%.4f | v13 congelado=%.4f | delta=%+.4f [%+.4f, %+.4f] | vs base delta=%+.4f",
        ruta_lora,
        resumen[ruta_lora]["medidas"]["ndcg_at_10"]["media"],
        ndcg_v13,
        comparaciones[f"{ruta_lora}_vs_v13"]["medidas"]["ndcg_at_10"]["media_delta"],
        comparaciones[f"{ruta_lora}_vs_v13"]["medidas"]["ndcg_at_10"]["p2_5"],
        comparaciones[f"{ruta_lora}_vs_v13"]["medidas"]["ndcg_at_10"]["p97_5"],
        delta_base["media_delta"],
    )
    logger.info("nivel 3: comparaciones pareadas con %d casos comunes", n_casos)
    return cierre


def dibujar_nivel3(cierre: dict, ruta: str | Path) -> None:
    """Figura del nivel 3: nDCG@10 por ruta y deltas por caso vs v13.

    Args:
        cierre: Cierre del nivel 3.
        ruta: Ruta de la figura.
    """
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura, ejes = plt.subplots(1, 2, figsize=(13, 4.8))
    nombres = list(cierre["rutas"])
    medias = [cierre["rutas"][n]["medidas"]["ndcg_at_10"]["media"] for n in nombres]
    inferiores = [cierre["rutas"][n]["medidas"]["ndcg_at_10"]["media"] - cierre["rutas"][n]["medidas"]["ndcg_at_10"]["p2_5"] for n in nombres]
    superiores = [cierre["rutas"][n]["medidas"]["ndcg_at_10"]["p97_5"] - cierre["rutas"][n]["medidas"]["ndcg_at_10"]["media"] for n in nombres]
    x = np.arange(len(nombres))
    ejes[0].bar(x, medias, yerr=[inferiores, superiores], capsize=3, color="#1f77b4")
    ejes[0].axhline(cierre["referencias_congeladas"]["v13_ndcg_at_10_con_hubs"], color="red", linestyle="--", label="v13 con hubs (0,6907)")
    ejes[0].axhline(cierre["referencias_congeladas"]["c2_rerank_llm_ndcg_at_10"], color="gray", linestyle=":", label="C2 zero-shot listwise (0,473)")
    ejes[0].set_xticks(x, [n.replace("_", "\n") for n in nombres], fontsize=7)
    ejes[0].set_ylabel("nDCG@10")
    ejes[0].set_title("Nivel 3 - rutas (IC95 bootstrap)")
    ejes[0].legend(fontsize=8)
    ruta_lora = "c2_lora_top10" if cierre["fuente_candidatos"] == "json_top10" else "c2_lora_top50"
    refs_v13 = {fila["id"]: fila["ndcg_at_10"] for fila in cierre.get("v13_por_caso", [])}
    if refs_v13:
        deltas = [
            fila["ndcg_at_10"] - refs_v13[fila["id"]]
            for fila in cierre["rutas"][ruta_lora]["metricas_por_caso"]
            if fila["id"] in refs_v13
        ]
        ejes[1].hist(deltas, bins=20, color="#9467bd", edgecolor="black")
        ejes[1].axvline(0, color="red", linestyle="--")
        ejes[1].set_xlabel(f"nDCG@10 ({ruta_lora}) - v13 congelado")
        ejes[1].set_title(
            f"Delta por caso (media {np.mean(deltas):+.4f})\n"
            f"comparacion pareada del cierre: media {cierre['comparaciones'][f'{ruta_lora}_vs_v13']['medidas']['ndcg_at_10']['media_delta']:+.4f}"
        )
    figura.tight_layout()
    figura.savefig(destino, dpi=130)
    plt.close(figura)


# ===== SECCION 8: autotest, criterios globales, salida y CLI =====


def autotest(cfg: dict, rutas: dict, tokenizer, logger: logging.Logger) -> None:
    """Validacion CPU del pipeline de evaluacion (sin modelo 7B ni GPU).

    Verifica: metricas con datos sinteticos (AUC, Spearman, umbral,
    bootstrap pareado), carga de los 820 items B3 + grados + joins, el
    formato de prompt identico al script 33 (ids en contexto), la
    integridad del val (anti-leakage) y los insumos del benchmark (top-10
    y referencias congeladas). No escribe artefactos reales.

    Args:
        cfg: Configuracion efectiva.
        rutas: Rutas absolutas.
        tokenizer: Tokenizer de ALIA.
        logger: Logger activo.

    Raises:
        RuntimeError: Si falla cualquier verificacion.
    """
    logger.info("autotest: metricas con datos sinteticos")
    rng = np.random.default_rng(cfg["semilla"])
    n = 400
    etiquetas = np.concatenate([np.ones(n // 2, dtype=int), np.zeros(n // 2, dtype=int)])
    perfectas = np.concatenate([rng.uniform(0.8, 1.0, n // 2), rng.uniform(0.0, 0.2, n // 2)])
    aleatorias = rng.uniform(0.0, 1.0, n)
    _exigir(auc_seguro(etiquetas, perfectas) == 1.0, "AUC perfecta != 1.0")
    _exigir(metricas_clasificacion(etiquetas, perfectas, 0.5)["acc"] == 1.0, "acc perfecta != 1.0")
    _exigir(
        abs(spearman_seguro(np.arange(10.0), np.arange(10.0)) - 1.0) < 1e-9,
        "Spearman monotono != 1.0",
    )
    umbral = optimizar_umbral(etiquetas, perfectas, "youden")
    _exigir(0.2 <= umbral["umbral"] <= 0.8, f"umbral optimo fuera de rango: {umbral}")
    grupos = [f"caso_{i % 20}" for i in range(n)]
    delta = bootstrap_pareado_grupos(
        etiquetas,
        perfectas,
        aleatorias,
        grupos,
        lambda y, s: auc_seguro(y, s),
        cfg["bootstrap"],
        cfg["semilla"],
        cfg["max_intentos_bootstrap"],
    )
    _exigir(delta["media_delta"] > 0 and delta["prob_mejora"] > 0.95, f"bootstrap sintetico inesperado: {delta}")
    logger.info("autotest PASS: metricas sinteticas (AUC, acc, Spearman, umbral, bootstrap)")

    logger.info("autotest: gold B3 (820 items) + joins")
    gold = cargar_gold(cfg, rutas, logger)
    items = gold["items"]
    _exigir(len(items) == 820, f"items del gold: {len(items)} != 820")
    _exigir(len({item["id"] for item in items}) == 50, "casos del gold != 50")
    _exigir(all(item["pasaje"].strip() for item in items), "items del gold sin texto de articulo")
    _exigir(all(item["fragmento"].strip() for item in items), "items del gold sin extracto")
    benchmark_ids = {str(caso["id"]) for caso in json.loads(Path(rutas["ruta_benchmark_v2"]).read_text(encoding="utf-8"))["muestra"]}
    _exigir({item["id"] for item in items} <= benchmark_ids, "casos del gold fuera del benchmark")
    logger.info("autotest PASS: gold B3 (%d items, %d casos, grados=%s)", len(items), 50, {g: sum(1 for i in items if i["grado"] == g) for g in (0, 1, 2)})

    logger.info("autotest: formato de prompt identico al 33 (ids en contexto)")
    _exigir(cfg["system"] == modulo_33().CONFIG["system"], "system del 34 != system del 33")
    _exigir(cfg["plantilla_user"] == modulo_33().CONFIG["plantilla_user"], "plantilla_user del 34 != del 33")
    ids_ctx = verificar_ids_respuesta(tokenizer, rutas, logger)
    for salida in ("Sí", "No"):
        usuario = construir_prompt_usuario(cfg, items[0]["fragmento"], items[0]["articulo"], items[0]["pasaje"])
        secuencia = modulo_33().construir_secuencia(tokenizer, cfg["system"], usuario, salida, cfg["max_seq"])
        _exigir(
            secuencia["resp_ids"][0] == ids_ctx["contexto"][salida],
            f"primer token de respuesta de {salida!r} != id en contexto",
        )
    render = render_con_presupuesto(tokenizer, cfg, items[0]["fragmento"], items[0]["articulo"], items[0]["pasaje"])
    _exigir(len(render["prompt_ids"]) <= cfg["max_seq"] - 2, "prompt del gold excede max_seq - 2")
    render_largo = render_con_presupuesto(
        tokenizer, {**cfg, "max_seq": 256}, "fragmento de prueba " * 400, items[0]["articulo"], items[0]["pasaje"]
    )
    _exigir(render_largo["ajustado"], "un fragmento largo no activo el ajuste por presupuesto")
    texto_render = tokenizer.decode(render_largo["prompt_ids"])
    _exigir("¿El fragmento cita el artículo" in texto_render, "el ajuste por presupuesto perdio la pregunta final")
    _exigir(items[0]["pasaje"][:50] in texto_render, "el ajuste por presupuesto perdio el texto del articulo")
    _exigir(len(render_largo["prompt_ids"]) <= 254, "prompt ajustado excede el presupuesto")
    logger.info("autotest PASS: prompts con la plantilla e ids del 33 (n_tokens=%d)", render["n_tokens_prompt"])

    logger.info("autotest: val interno (3000) y anti-leakage")
    datos_val, ejemplos_val = cargar_ejemplos_sft(rutas, cfg, "val")
    _exigir(len(ejemplos_val) == 3000, f"val: {len(ejemplos_val)} != 3000")
    labels = np.asarray(datos_val["label"], dtype=int)
    _exigir(int((labels == 1).sum()) == 1500 and int((labels == 0).sum()) == 1500, "val desbalanceado")
    docs_val = {str(doc) for doc in datos_val["doc_id"]}
    _exigir(not (docs_val & benchmark_ids), "docs de val dentro del benchmark (leakage)")
    _exigir(all(len(ejemplo["prompt_ids"]) <= cfg["max_seq"] - 2 for ejemplo in ejemplos_val), "prompt de val excede max_seq - 2")
    logger.info("autotest PASS: val 3000 (1500/1500), %d docs, sin fuga al benchmark", len(docs_val))

    logger.info("autotest: insumos del benchmark (top-10 y referencias congeladas)")
    bench = cargar_benchmark(rutas, cfg["benchmark_variante"], logger)
    refs = referencias_v13_congeladas(bench, cfg["estrategia_v13"], cfg["contexto_v13"])
    _exigir(len(refs) == 90, f"referencias v13 congeladas: {len(refs)} != 90")
    top10 = bench["datos"]["top10_por_caso"]
    _exigir(
        all(len(estrategias.get(cfg["estrategia_v13"], [])) == 10 for estrategias in top10.values()),
        "top-10 incompleto en el benchmark",
    )
    logger.info("autotest PASS: benchmark %s (90 casos, top-10 completo, referencias congeladas)", bench["variante"])

    logger.info("autotest: construccion de pares (chunk, pasaje) del nivel 3 (BM25 CPU)")
    mod12 = modulo_12()
    mod30 = modulo_30()
    muestra = bench["datos"]["muestra"][:1]
    textos = mod30.leer_textos_por_id(mod12, Path(rutas["ruta_corpus"]), [str(muestra[0]["id"])])
    pasajes, canones, grupos_indices, _canon_valido = mod12.preparar_pasajes_constitucion(False)
    posicion_canon = {canon: i for i, canon in enumerate(canones)}
    indice_bm25 = mod12.construir_indice_bm25(pasajes)
    id_caso = str(muestra[0]["id"])
    top = top10[id_caso][cfg["estrategia_v13"]]
    ventana = textos[id_caso]["texto_limpio"][: mod12.CONFIG["ventana_query_chars"]]
    chunks = mod12.chunks_query(ventana)
    _, matriz = mod12.scores_bm25_caso(indice_bm25, grupos_indices, canones, chunks)
    scores_orden = np.full(len(canones), -np.inf, dtype=np.float64)
    for posicion, item in enumerate(top):
        scores_orden[posicion_canon[modulo_16().normalizar_articulo(item["id"])]] = -float(posicion)
    caso_bm25 = {"matriz_bm25": matriz, "chunks_bm25": chunks}
    pares = pares_para_caso(mod12, scores_orden, caso_bm25, pasajes, grupos_indices, len(top))
    _exigir(len(pares["queries"]) == 30, f"pares del nivel 3: {len(pares['queries'])} != 30")
    _exigir(all(p.strip() for p in pares["pasajes"]), "pares con pasaje vacio")
    candidatos = {canones[indice] for indice in pares["candidatos"]}
    esperados = {modulo_16().normalizar_articulo(item["id"]) for item in top}
    _exigir(candidatos == esperados, f"candidatos != top-10: {candidatos ^ esperados}")
    prompts, articulos, _ = prompts_desde_pares(tokenizer, cfg, pares, canones)
    _exigir(len(prompts) == 30 and len(set(articulos)) == 10, "prompts del nivel 3 inconsistentes")
    logger.info("autotest PASS: nivel 3 con %d pares (10 candidatos x 3 chunks), pasajes y prompts OK", len(prompts))
    logger.info("autotest: TODAS las verificaciones CPU pasaron")


def evaluar_criterios(salida: dict) -> dict:
    """Recalcula el bloque de criterios globales desde las secciones.

    Args:
        salida: JSON de salida (con las secciones disponibles).

    Returns:
        Dict con S2/S3/S4 y el resumen GO/PARTIAL/STOP.
    """
    criterios = {}
    mapa = {"nivel1": "criterios_s2", "nivel2": "criterios_s3", "nivel3": "criterios_s4"}
    for clave, clave_criterio in mapa.items():
        if clave in salida and salida[clave] is not None:
            criterios[clave] = salida[clave].get(clave_criterio)
    resultados = [c["resultado"] for c in criterios.values() if c and c.get("resultado")]
    if not resultados:
        global_ = None
    elif all(r == "GO" for r in resultados):
        global_ = "GO"
    elif all(r == "NO_CUMPLE" for r in resultados):
        global_ = "STOP"
    else:
        global_ = "PARTIAL"
    return {"por_nivel": criterios, "resultado_global": global_}


def guardar_salida(ruta: str | Path, seccion: str, cierre: dict, forzar: bool, logger: logging.Logger) -> dict:
    """Fusiona la seccion del nivel en el JSON de salida.

    Si el JSON existe, conserva las secciones previas y reemplaza solo la
    del nivel corrido (requiere --forzar si esa seccion ya existe). Recalcula
    el bloque de criterios globales.

    Args:
        ruta: Ruta del JSON de salida.
        seccion: "nivel1", "nivel2" o "nivel3".
        cierre: Cierre del nivel.
        forzar: True permite reemplazar una seccion existente.
        logger: Logger activo.

    Returns:
        Dict completo guardado.

    Raises:
        FileExistsError: Si la seccion ya existe y no hay --forzar.
    """
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    existente: dict = {}
    if destino.exists():
        try:
            with open(destino, encoding="utf-8") as manejador:
                existente = json.load(manejador)
        except json.JSONDecodeError:
            logger.warning("JSON de salida existente ilegible; se reescribe")
            existente = {}
        if seccion in existente and not forzar:
            raise FileExistsError(
                f"{destino} ya tiene la seccion {seccion}; use --forzar para reemplazarla"
            )
    salida = dict(existente)
    salida.update(
        {
            "version": CONFIG["version"],
            "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
            "script": "scripts/34_c2_lora_eval.py",
        }
    )
    salida[seccion] = cierre
    salida["criterios"] = evaluar_criterios(salida)
    with open(destino, "w", encoding="utf-8") as manejador:
        json.dump(salida, manejador, ensure_ascii=False, indent=2)
    logger.info("salida guardada: %s (seccion %s)", destino, seccion)
    return salida


def parsear_args() -> argparse.Namespace:
    """Parsea la linea de comandos.

    Returns:
        Namespace con los argumentos validados.

    Raises:
        SystemExit: Si la combinacion de modos es invalida.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--nivel",
        type=str,
        default=None,
        help="niveles a evaluar: 1, 2, 3, lista '1,3' o 'todo'",
    )
    parser.add_argument("--adaptador", type=str, default=None, help="ruta del adaptador LoRA (default modelos/c2_lora/adapter_v1)")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint intermedio a evaluar (reemplaza --adaptador)")
    parser.add_argument("--modelo-base", type=str, default=None, help="ruta del modelo base (default modelos/alia_7b_instruct)")
    parser.add_argument("--max-seq", type=int, default=None, help="longitud maxima (default 768)")
    parser.add_argument("--batch", type=int, default=None, help="batch de scoring (default 4)")
    parser.add_argument("--salida", type=str, default=None, help="JSON de salida (default resultados/c2_lora/eval_c2_lora_v1.json)")
    parser.add_argument("--benchmark", choices=("v13", "v14_ep2"), default="v13", help="benchmark de referencia del nivel 3")
    parser.add_argument("--recomputar-top50", action="store_true", help="recalcula la primera etapa (top-50 + B1; GPU)")
    parser.add_argument(
        "--recomputar-candidatos",
        action="store_true",
        help="regenera el pickle de candidatos top-50 aunque exista",
    )
    parser.add_argument(
        "--precomputar-top50",
        type=str,
        default=None,
        metavar="RUTA_PKL",
        help="modo interno: precomputa candidatos top-50 (B2 + B1) y los guarda en RUTA_PKL (subproceso del flujo --recomputar-top50)",
    )
    parser.add_argument("--dispositivo", choices=("auto", "cuda", "cpu"), default="auto", help="dispositivo de evaluacion")
    parser.add_argument("--max-items", type=int, default=None, help="tope de items de val (nivel 1)")
    parser.add_argument("--max-casos", type=int, default=None, help="tope de casos de gold/benchmark")
    parser.add_argument("--forzar", action="store_true", help="permite reemplazar la seccion del nivel en el JSON")
    parser.add_argument("--smoke", action="store_true", help="dry-run con modelo diminuto en /tmp/opencode")
    parser.add_argument("--autotest", action="store_true", help="validacion CPU de datos/metricas/pares (sin modelo)")
    args = parser.parse_args()
    if args.smoke and args.recomputar_top50:
        parser.error("--smoke no admite --recomputar-top50 (requiere B2 y B1 reales)")
    if args.smoke and args.precomputar_top50:
        parser.error("--smoke no admite --precomputar-top50")
    if not args.autotest and not args.precomputar_top50 and args.nivel is None:
        parser.error("se requiere --nivel 1|2|3|todo (o --autotest/--smoke)")
    if args.nivel is not None:
        niveles = [n.strip() for n in args.nivel.split(",") if n.strip()]
        if "todo" in niveles:
            niveles = ["1", "2", "3"]
        if not niveles or any(n not in ("1", "2", "3") for n in niveles):
            parser.error("--nivel debe ser 1, 2, 3, una lista como '1,3' o 'todo'")
        args.niveles = sorted(set(niveles), key=int)
    else:
        args.niveles = []
    return args


def resolver_config(args: argparse.Namespace) -> dict:
    """Resuelve la configuracion efectiva (overrides CLI + rutas smoke).

    Args:
        args: Namespace de la CLI.

    Returns:
        Configuracion efectiva con "rutas" resueltas.

    Raises:
        SystemExit: Si un valor queda fuera de rango.
    """
    cfg = copy.deepcopy(CONFIG)
    cfg["smoke"] = bool(args.smoke)
    cfg["benchmark_variante"] = args.benchmark
    cfg["max_items_val"] = None
    cfg["max_casos_gold"] = None
    cfg["max_casos_benchmark"] = None
    if args.modelo_base is not None:
        cfg["ruta_modelo"] = args.modelo_base
    if args.adaptador is not None:
        cfg["dir_adapter"] = args.adaptador
    if args.checkpoint is not None:
        cfg["dir_adapter"] = args.checkpoint
    if args.max_seq is not None:
        if args.max_seq < 64:
            raise SystemExit("--max-seq debe ser >= 64")
        cfg["max_seq"] = args.max_seq
    if args.batch is not None:
        if args.batch < 1:
            raise SystemExit("--batch debe ser >= 1")
        cfg["batch"] = args.batch
    if args.salida is not None:
        cfg["ruta_salida"] = args.salida
    if args.max_items is not None:
        cfg["max_items_val"] = args.max_items
    if args.max_casos is not None:
        cfg["max_casos_gold"] = args.max_casos
        cfg["max_casos_benchmark"] = args.max_casos
    if args.smoke:
        base = CONFIG["smoke"]["dir_salida"]
        cfg["max_seq"] = CONFIG["smoke"]["max_seq"]
        cfg["batch"] = CONFIG["smoke"]["batch"]
        cfg["max_items_val"] = args.max_items if args.max_items is not None else CONFIG["smoke"]["max_items_val"]
        cfg["max_casos_gold"] = args.max_casos if args.max_casos is not None else CONFIG["smoke"]["max_casos_gold"]
        cfg["max_casos_benchmark"] = (
            args.max_casos if args.max_casos is not None else CONFIG["smoke"]["max_casos_benchmark"]
        )
        cfg["ruta_salida"] = args.salida if args.salida is not None else f"{base}/eval_c2_lora_smoke.json"
        cfg["ruta_cache"] = f"{base}/cache_scores_smoke.jsonl"
        cfg["ruta_log"] = f"{base}/34_c2_lora_eval_smoke.log"
        cfg["ruta_figura_val"] = f"{base}/fig_c2_lora_val_smoke.png"
        cfg["ruta_figura_gold"] = f"{base}/fig_c2_lora_gold_smoke.png"
        cfg["ruta_figura_benchmark"] = f"{base}/fig_c2_lora_benchmark_smoke.png"
    cfg["rutas"] = resolver_rutas(cfg)
    return cfg


def resolver_dispositivo(args: argparse.Namespace, cfg: dict) -> str:
    """Resuelve el dispositivo de evaluacion.

    Args:
        args: Namespace de la CLI.
        cfg: Configuracion efectiva.

    Returns:
        "cuda" o "cpu".
    """
    if args.dispositivo == "cpu":
        return "cpu"
    if args.dispositivo == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--dispositivo cuda pero torch.cuda no esta disponible")
        return "cuda"
    if cfg["smoke"]:
        return "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] / 1024**3 >= 6.0 else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> int:
    """Punto de entrada: despacha autotest/smoke/niveles 1-3.

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
    logger.info("34_c2_lora_eval.py | args=%s", vars(args))
    fijar_semillas(cfg["semilla"])

    if args.precomputar_top50:
        # Modo interno (subproceso de --recomputar-top50): B2 + B1 y pickle.
        # No carga tokenizer ni el 7B; al salir, la VRAM queda totalmente libre.
        bench = cargar_benchmark(rutas, cfg["benchmark_variante"], logger)
        precomputado = precomputar_candidatos_top50(cfg, rutas, bench, logger)
        guardar_candidatos_pickle(args.precomputar_top50, precomputado, logger)
        logger.info("fin del precomputo de candidatos")
        return 0

    tokenizer = cargar_tokenizer(rutas)

    if args.autotest:
        autotest(cfg, rutas, tokenizer, logger)
        logger.info("fin del autotest")
        return 0

    device = resolver_dispositivo(args, cfg)
    ids_ctx = verificar_ids_respuesta(tokenizer, rutas, logger)
    exclusiones = {"benchmark": set(), "gold": set()}
    bench = cargar_benchmark(rutas, cfg["benchmark_variante"], logger)
    exclusiones["benchmark"] = {str(caso["id"]) for caso in bench["datos"]["muestra"]}

    if not cfg["smoke"]:
        max_seq_ent = max_seq_del_entrenamiento(rutas, logger)
        if max_seq_ent is not None:
            # El cap de render del dataset (768) es el que vio el modelo sin
            # truncacion; solo se alinea al max_seq del run si este trunco
            # prompts de verdad (max de tokens del train > max_seq del run).
            truncado_entrenamiento = False
            ruta_dataset = Path(rutas["ruta_dataset_json"])
            if ruta_dataset.exists():
                try:
                    with open(ruta_dataset, encoding="utf-8") as manejador:
                        conteos = json.load(manejador).get("conteos", {})
                    max_tokens_train = (conteos.get("train", {}).get("tokens_total") or {}).get("max")
                    truncado_entrenamiento = bool(max_tokens_train and max_tokens_train > max_seq_ent)
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("no se pudo leer %s: %s", ruta_dataset, exc)
            if truncado_entrenamiento and args.max_seq is None and max_seq_ent != cfg["max_seq"]:
                logger.info(
                    "max_seq alineado al del entrenamiento: %d -> %d (el run trunco prompts; use --max-seq para forzarlo)",
                    cfg["max_seq"],
                    max_seq_ent,
                )
                cfg["max_seq"] = max_seq_ent
            elif not truncado_entrenamiento:
                logger.info(
                    "max_seq de evaluacion=%d (cap de render del dataset); entrenamiento con max_seq=%d sin truncacion efectiva",
                    cfg["max_seq"],
                    max_seq_ent,
                )
            if args.max_seq is not None and max_seq_ent != cfg["max_seq"]:
                logger.warning(
                    "max_seq de evaluacion (%d) != max_seq del entrenamiento (%d): la truncacion no sera identica",
                    cfg["max_seq"],
                    max_seq_ent,
                )

    if cfg["smoke"]:
        cfg["rutas"]["dir_adapter"] = str(Path(CONFIG["smoke"]["dir_salida"]) / "adapter_smoke")
    # Fase GPU de candidatos (B2 + B1) ANTES de cargar el 7B: el presupuesto
    # de 16 GB no admite ambos a la vez. El precomputo corre en subproceso
    # (VRAM 100% liberada al salir) y queda cacheado en disco.
    precomputado = None
    if "3" in args.niveles and args.recomputar_top50:
        precomputado = obtener_candidatos_top50(cfg, rutas, args, logger)
    modelo = cargar_modelo_evaluacion(cfg, rutas, args, device, logger)
    modelo_tag = "smoke" if cfg["smoke"] else (
        Path(rutas["dir_adapter"]).name + "|" + (md5_archivo(Path(rutas["dir_adapter"]) / "adapter_config.json") or "sin_config")
    )
    cache = CacheScores(rutas["ruta_cache"], activa=not cfg["smoke"], modelo_tag=modelo_tag)

    salida = {}
    if "1" in args.niveles:
        cierre = evaluar_nivel1(
            modelo, tokenizer, cfg, rutas, device, ids_ctx, cache, exclusiones, logger
        )
        dibujar_nivel1(cierre, rutas["ruta_figura_val"])
        salida = guardar_salida(rutas["ruta_salida"], "nivel1", cierre, args.forzar, logger)
    if "2" in args.niveles:
        cierre = evaluar_nivel2(
            modelo, tokenizer, cfg, rutas, device, ids_ctx, cache, exclusiones, logger
        )
        salida = guardar_salida(rutas["ruta_salida"], "nivel2", cierre, args.forzar, logger)
    if "3" in args.niveles:
        if args.recomputar_top50:
            cierre = evaluar_nivel3_recomputado(
                modelo, tokenizer, cfg, rutas, device, ids_ctx, cache, bench, precomputado, logger
            )
        else:
            cierre = evaluar_nivel3_fallback(
                modelo, tokenizer, cfg, rutas, device, ids_ctx, cache, bench, logger
            )
        dibujar_nivel3(cierre, rutas["ruta_figura_benchmark"])
        salida = guardar_salida(rutas["ruta_salida"], "nivel3", cierre, args.forzar, logger)

    if not salida:
        logger.warning("ningun nivel ejecutado")
    else:
        logger.info("criterios globales: %s", salida.get("criterios", {}).get("resultado_global"))
    cache.guardar()
    logger.info("fin del script")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 - protocolo: traza completa y salida 1 ante cualquier fallo
        traceback.print_exc()
        sys.exit(1)
