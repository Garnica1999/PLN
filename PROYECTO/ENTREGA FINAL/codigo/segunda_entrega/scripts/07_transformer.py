"""Transformer M4: fine-tuning BETO para area tematica.

v1 (Fase L3, 2026-09-09): truncamiento cabeza+cola 256+256 tokens (ADR-3
variante a). Resultado congelado (resultados/m4_transformer.json, INTOCABLE
segun ADR-11): macro-F1 test 0.2080/0.2042 — perdio contra LogReg 0.2870
porque la ventana 512 descartaba ~95 % del contenido de docs de ~11k
palabras (ADR-10).

v2 (ciclo 2 - Fase V, 2026-09-10): modo chunk-mean (Chalkidis et al. 2022,
"Processing Long Legal Documents with Pre-trained Transformers: Modding
LegalBERT and Longformer", arXiv:2211.00974). El documento se trocea en
chunks de 256 tokens con sliding window (stride 128, solapamiento 50 %),
BETO codifica cada chunk, el embedding del documento es la MEDIA de los
pooler outputs de sus chunks (mismo pooling que v1) y la cabeza clasifica
sobre ese agregado; backprop a traves de la media (todos los chunks
contribuyen al gradiente). En documentos legales largos este metodo vence
al truncamiento (evidencia del paper y diagnostico ADR-10).

Decisiones de implementacion (documentadas):
    - ESTRATEGIA_LONGITUD = "chunk_mean" es el default; el modo "truncado"
      (v1) queda accesible para comparabilidad, pero ABORTA si intenta
      escribir los resultados v1 (proteccion ADR-11).
    - Hiperparametros iguales al run v1 real: 3 epocas, batch efectivo 32
      docs, lr 2e-5 (el JSON v1 registra lr 2e-5; el plan v2 cita 5e-5 por
      error — prevalece el valor real del run v1 para comparabilidad),
      seed 42, mejor checkpoint por macro-F1 val (logica v1).
    - Micro-lote de forward = todos los chunks de UN doc (<=64 chunks,
      pico VRAM medido ~7.4 GB); se acumulan gradientes de 32 docs por paso
      de optimizador (batch efectivo 32 docs, igual que v1).
    - MAX_CHUNKS=64 (~8k tokens utiles por doc con stride 128): los docs
      con mas chunks se submuestrean uniformemente (np.linspace,
      determinista). Trade-off documentado: docs muy largos (p99 ~46k
      palabras) ven una fraccion de su contenido, pero ~64 chunks ya cubre
      ~8k tokens (~5x mas contenido que v1) y mantiene el run < 24 h.
    - AMP fp16 (GradScaler): v1 corria fp32; sin AMP la estimacion supera
      24 h/escenario. Throughput medido en esta GPU (RX 9070 XT): ~182
      chunks/s en train (fwd+bwd+step), ~455 chunks/s en eval.
    - Estimacion de tiempo MEDIDA en smoke (200 docs, 53.3 chunks/doc
      promedio, 64 max): ~0.15 s/doc train, ~0.05 s/doc eval ->
      ~45 min/epoca de train (16.557 docs) + ~3 min val (3.549) ->
      ~2.5-3 h por escenario, ~5-6 h ambos (lejos del tope de 24 h;
      no fue necesario reducir max_chunks a 32).
    - Correccion menor vs v1: evaluar_beto v1 deja el modelo en eval() sin
      restaurar train() (epocas 2-3 de v1 entrenaron sin dropout). El modo
      chunk_mean restaura train() al inicio de cada epoca; v1 no se toca.
    - Guard de VRAM como v1 (espera reintento cada 5 min), con umbral
      10 GB libres para chunk_mean (pico medido 7.4 GB + margen).

v2 iter2 (ciclo v2 - Fase V, 2026-09-11): DEADLOCK del DataLoader resuelto
con cache de chunks. Los runs completos de chunk_mean se colgaban SIEMPRE
en el DataLoader multiproceso (main en dataloader._try_get_data esperando
lote, 8 workers idle esperando indices; Python 3.14 + fork; evidencia en
experimentos/m4_transformer_chunks_stdout_faulthandler_evidencia.log).
Ahora los chunks se pre-tokenizan UNA vez a disco con
scripts/14_precache_chunks_beto.py (proceso sin torch/CUDA) y el
entrenamiento usa num_workers=0 + pin_memory=False leyendo la cache
(data/cache_chunks_beto/{texto,texto_mask}/shard_*.pkl): sin procesos
worker no hay deadlock posible y el arranque es mas rapido. La
tokenizacion/troceo/enmascarado viven en scripts/lib/chunks_beto.py
(fuente unica compartida con 14). La cache es obligatoria para
chunk_mean (fail-fast si falta o no cubre los ids).

Doble escenario (ADR-2): completo vs citas enmascaradas sobre el texto
crudo (regex M2 portado; enmascara "articulo N ... constitucion",
"ley N de AAAA", "T-293 de 1997" por [CITA]).

Uso:
    /home/carlos/venv/bin/python scripts/07_transformer.py            # chunk_mean completo
    /home/carlos/venv/bin/python scripts/07_transformer.py --limite 200 --epocas 1   # smoke a /tmp/opencode/
    /home/carlos/venv/bin/python scripts/07_transformer.py --estrategia truncado     # v1 (protegido)
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from lib.chunks_beto import (
    CHUNK_STRIDE,
    CHUNK_TOKENS,
    MAX_CHUNKS,
    MODELO_BETO,
    enmascarar_texto_crudo,
    hash_config_chunking,
    trocear_doc,
)
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ===== SECCION 1: configuracion =====
# Estrategia de longitud: "chunk_mean" (v2 Fase V, Chalkidis 2022) o
# "truncado" (v1 cabeza+cola 256+256). Default v2: chunk_mean.
ESTRATEGIA_LONGITUD = "chunk_mean"

CONFIG = {
    "semilla": 42,
    "modelo": MODELO_BETO,
    "max_tokens": 512,  # modo truncado v1: 256 cabeza + 256 cola (ADR-3)
    "batch_train": 8,  # modo truncado v1
    "batch_eval": 16,  # docs por lote de evaluacion (ambos modos)
    "grad_accum": 4,  # modo truncado v1: batch efectivo 32
    "epochs": 3,
    "lr": 2e-5,  # valor real del run v1 (JSON v1), no 5e-5 del plan
    "vram_min_gb": 6.0,  # truncado v1; chunk_mean lo sube a 10.0
    "vram_espera_seg": 300,  # reintento cada 5 min
    "vram_max_esperas": 288,  # hasta 24 h
    "ruta_splits": "data/splits_m4.parquet",
    "ruta_corpus": "data/corpus_sentencias.parquet",
    "ruta_etiquetas": "data/etiquetas_areas.parquet",
    "dir_modelos": "modelos/m4_transformer",
    "ruta_metricas": "resultados/m4_transformer.json",
    "ruta_log": "experimentos/m4_transformer.log",
    # --- v2 Fase V: chunk-mean (Chalkidis 2022) ---
    "chunk_tokens": CHUNK_TOKENS,  # tokens por chunk (fuente: lib/chunks_beto)
    "chunk_stride": CHUNK_STRIDE,  # solapamiento 50 % entre ventanas
    "max_chunks": MAX_CHUNKS,  # tope por doc (~8k tokens utiles); muestreo uniforme si excede
    "dir_cache_chunks": "data/cache_chunks_beto",  # cache de 14_precache_chunks_beto.py
    "batch_efectivo_docs": 32,  # docs por paso de optimizador (igual que v1)
    "micro_chunks_eval": 128,  # chunks por forward en evaluacion (sin backward)
    # v2 iter2 (Fase V): 0 workers evita el deadlock del DataLoader
    # multiproceso a escala completa (Python 3.14 + fork, incidente
    # 2026-09-11; la tokenizacion vive en 14_precache_chunks_beto.py).
    "num_workers": 0,
    "amp": True,  # fp16 con GradScaler (v1 era fp32; ver docstring)
    "log_cada_lotes": 25,  # frecuencia de log de train
}

LOGGER = logging.getLogger("m4_transformer")

# Enmascarado sobre TEXTO CRUDO (mas fiable que a nivel de lemas).
# v2 iter2 (Fase V): RE_CITA_CRUDO y enmascarar_texto_crudo se movieron a
# scripts/lib/chunks_beto.py (fuente unica compartida con 14: la cache
# debe enmascarar EXACTAMENTE igual que este script).


def configurar_logging() -> logging.Logger:
    """Configura el logger del experimento."""
    Path(CONFIG["ruta_log"]).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(CONFIG["ruta_log"], mode="w"),
            logging.StreamHandler(),
        ],
    )
    return LOGGER


def esperar_vram(vram_min_gb: float | None = None) -> torch.device:
    """Espera ciclicamente hasta que la GPU tenga VRAM libre suficiente.

    Args:
        vram_min_gb: Umbral de VRAM libre en GB; None usa CONFIG["vram_min_gb"].

    Returns:
        Dispositivo cuda cuando hay VRAM libre suficiente.
    """
    umbral = CONFIG["vram_min_gb"] if vram_min_gb is None else vram_min_gb
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dispositivo.type != "cuda":
        LOGGER.warning("sin CUDA: continuaria en CPU (inviable para fine-tuning)")
        return dispositivo
    esperas = 0
    while True:
        libre, _ = torch.cuda.mem_get_info()
        gb = libre / 1e9
        if gb >= umbral:
            LOGGER.info("VRAM libre %.1f GB: suficiente", gb)
            return dispositivo
        esperas += 1
        if esperas > CONFIG["vram_max_esperas"]:
            raise RuntimeError(f"VRAM insuficiente tras {esperas} esperas ({gb:.1f} GB)")
        LOGGER.info("VRAM libre %.1f GB < %.1f: espero %d s (intento %d)", gb, umbral, CONFIG["vram_espera_seg"], esperas)
        time.sleep(CONFIG["vram_espera_seg"])


def cargar_datos(construir_mask: bool = True) -> pd.DataFrame:
    """Carga corpus + etiquetas + splits oficiales.

    Args:
        construir_mask: Si True agrega texto_mask con el enmascarado sobre
            texto crudo (necesario para truncado v1). chunk_mean v2 lo
            desactiva porque los chunks enmascarados ya vienen de la cache
            de 14_precache_chunks_beto.py (ahorra ~1.5 GB de regex).

    Returns:
        DataFrame con id, texto, area y particion.
    """
    corpus = pq.read_table(CONFIG["ruta_corpus"], columns=["id", "texto"]).to_pandas()
    etiquetas = pq.read_table(CONFIG["ruta_etiquetas"], columns=["id", "area_primaria"]).to_pandas()
    splits = pq.read_table(CONFIG["ruta_splits"]).to_pandas()
    df = corpus.merge(etiquetas, on="id").merge(splits, on="id")
    df = df[df["area_primaria"] != "sin_area_definida"].reset_index(drop=True)
    # Clases raras agrupadas igual que en 06 (consistencia de etiquetas).
    conteo = df["area_primaria"].value_counts()
    raras = conteo[conteo < 3].index.tolist()
    if raras:
        df.loc[df["area_primaria"].isin(raras), "area_primaria"] = "otras_areas"
    if construir_mask:
        df["texto_mask"] = df["texto"].map(enmascarar_texto_crudo)
    LOGGER.info("datos: %d filas, %d clases", len(df), df["area_primaria"].nunique())
    return df


class SentenciasDataset(Dataset):
    """Dataset de sentencias tokenizadas cabeza+cola para BETO."""

    def __init__(self, textos: pd.Series, etiquetas: pd.Series, tokenizer, clases: dict):
        self.textos = textos.tolist()
        self.etiquetas = etiquetas.map(clases).tolist()
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.textos)

    def __getitem__(self, idx: int) -> dict:
        texto = self.textos[idx]
        # Truncamiento cabeza+cola (ADR-3): tokenizar completo con
        # truncamiento largo y reconstruir 256+256.
        enc = self.tokenizer(
            texto,
            max_length=CONFIG["max_tokens"],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.etiquetas[idx], dtype=torch.long),
        }


def entrenar_beto(df: pd.DataFrame, columna: str, dispositivo: torch.device, etiqueta: str) -> dict:
    """Fine-tunea BETO en un escenario y evalua en test.

    Args:
        df: DataFrame con texto, area y particion.
        columna: texto (completo) o texto_mask (enmascarado).
        dispositivo: cuda/cpu.
        etiqueta: Nombre del escenario para logs/guardado.

    Returns:
        Dict con macro_f1, accuracy y ruta del modelo.
    """
    torch.manual_seed(CONFIG["semilla"])
    np.random.seed(CONFIG["semilla"])
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["modelo"])
    clases = {c: i for i, c in enumerate(sorted(df["area_primaria"].unique()))}

    tr = df[df["particion"] == "train"]
    va = df[df["particion"] == "val"]
    te = df[df["particion"] == "test"]
    dl_tr = DataLoader(SentenciasDataset(tr[columna], tr["area_primaria"], tokenizer, clases), batch_size=CONFIG["batch_train"], shuffle=True)
    dl_va = DataLoader(SentenciasDataset(va[columna], va["area_primaria"], tokenizer, clases), batch_size=CONFIG["batch_eval"])
    dl_te = DataLoader(SentenciasDataset(te[columna], te["area_primaria"], tokenizer, clases), batch_size=CONFIG["batch_eval"])

    modelo = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["modelo"], num_labels=len(clases)
    ).to(dispositivo)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=CONFIG["lr"])
    total_pasos = len(dl_tr) * CONFIG["epochs"] // CONFIG["grad_accum"]
    scheduler = get_linear_schedule_with_warmup(optimizador, int(0.06 * total_pasos), total_pasos)

    mejor_f1 = -1.0
    ruta_modelo = f"{CONFIG['dir_modelos']}/beto_{etiqueta}"
    modelo.train()
    paso = 0
    for epoca in range(CONFIG["epochs"]):
        perdida_epoca = 0.0
        for i, lote in enumerate(dl_tr):
            lote = {k: v.to(dispositivo) for k, v in lote.items()}
            salida = modelo(**lote)
            perdida = salida.loss / CONFIG["grad_accum"]
            perdida.backward()
            perdida_epoca += salida.loss.item()
            if (i + 1) % CONFIG["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                optimizador.step()
                scheduler.step()
                optimizador.zero_grad()
                paso += 1
        # Validacion al final de cada epoca.
        f1_va = evaluar_beto(modelo, dl_va, dispositivo)
        LOGGER.info("[%s] epoca %d: perdida=%.4f macro-F1 val=%.4f", etiqueta, epoca + 1, perdida_epoca / len(dl_tr), f1_va)
        if f1_va > mejor_f1:
            mejor_f1 = f1_va
            Path(CONFIG["dir_modelos"]).mkdir(parents=True, exist_ok=True)
            modelo.save_pretrained(ruta_modelo)
            tokenizer.save_pretrained(ruta_modelo)

    # Evaluar el mejor checkpoint en test.
    modelo = AutoModelForSequenceClassification.from_pretrained(ruta_modelo).to(dispositivo)
    y_pred, y_true = predecir(modelo, dl_te, dispositivo)
    inv = {v: k for k, v in clases.items()}
    y_pred_lbl = [inv[int(p)] for p in y_pred]
    y_true_lbl = [inv[int(t)] for t in y_true]
    return {
        "macro_f1": float(f1_score(y_true_lbl, y_pred_lbl, average="macro")),
        "accuracy": float(accuracy_score(y_true_lbl, y_pred_lbl)),
        "modelo": ruta_modelo,
        "mejor_f1_val": float(mejor_f1),
    }


def evaluar_beto(modelo, dl: DataLoader, dispositivo: torch.device) -> float:
    """Calcula macro-F1 de un modelo sobre un DataLoader.

    Args:
        modelo: Modelo BETO.
        dl: DataLoader de evaluacion.
        dispositivo: cuda/cpu.

    Returns:
        macro-F1.
    """
    modelo.eval()
    y_pred, y_true = predecir(modelo, dl, dispositivo)
    return f1_score(y_true, y_pred, average="macro")


def predecir(modelo, dl: DataLoader, dispositivo: torch.device) -> tuple[list, list]:
    """Predice etiquetas (codes) sobre un DataLoader.

    Args:
        modelo: Modelo BETO.
        dl: DataLoader.
        dispositivo: cuda/cpu.

    Returns:
        Tupla (y_pred codes, y_true codes).
    """
    modelo.eval()
    preds, trues = [], []
    with torch.no_grad():
        for lote in dl:
            mask = lote["attention_mask"].to(dispositivo)
            ids = lote["input_ids"].to(dispositivo)
            etiquetas = lote["labels"]
            logits = modelo(input_ids=ids, attention_mask=mask).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            trues.extend(etiquetas.tolist())
    return preds, trues


# ===== SECCION 2 (v2 Fase V): chunk-mean (Chalkidis 2022, arXiv:2211.00974) =====
# trocear_doc, enmascarar_texto_crudo y los parametros de chunking viven en
# scripts/lib/chunks_beto.py (fuente unica compartida con 14_precache).


def colapsar_lote_chunks(lote: list[dict]) -> dict:
    """Colapsa un lote de docs (con N chunks cada uno) en tensores planos.

    Args:
        lote: Lista de dicts {chunks: list[list[int]], labels: int}.

    Returns:
        Dict con input_ids/attention_mask (num_chunks, chunk) rellenados
        con pad, labels (n_docs,) y _n_chunks: numero de chunks de cada
        doc (para la media por documento en forward_doc_chunk_mean).
    """
    chunk = CONFIG["chunk_tokens"]
    pad = 1  # BETO pad_token_id
    n_filas = sum(len(d["chunks"]) for d in lote)
    input_ids = torch.full((n_filas, chunk), pad, dtype=torch.long)
    attention_mask = torch.zeros((n_filas, chunk), dtype=torch.long)
    fila = 0
    n_chunks = []
    for d in lote:
        for ch in d["chunks"]:
            input_ids[fila, : len(ch)] = torch.tensor(ch, dtype=torch.long)
            attention_mask[fila, : len(ch)] = 1
            fila += 1
        n_chunks.append(len(d["chunks"]))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor([d["labels"] for d in lote], dtype=torch.long),
        "_n_chunks": n_chunks,
    }


def submuestra_estratificada(df: pd.DataFrame, limite: int) -> pd.DataFrame:
    """Submuestra estratificada por area_primaria (smoke test).

    Args:
        df: Particion completa (train/val/test).
        limite: Tamano objetivo de la submuestra.

    Returns:
        DataFrame con <=limite filas que conserva la proporcion de clases.
    """
    frac = min(1.0, limite / len(df))
    sub = df.groupby("area_primaria", group_keys=False).sample(frac=frac, random_state=CONFIG["semilla"])
    return sub.reset_index(drop=True)


def cargar_cache_chunks(columna: str, dir_cache: str) -> dict:
    """Carga la cache de chunks pre-tokenizados de una columna.

    v2 iter2 (Fase V): la cache la genera scripts/14_precache_chunks_beto.py
    (proceso sin torch) una sola vez; asi el entrenamiento no tokeniza ni
    usa workers del DataLoader (deadlock Python 3.14 + fork; evidencia en
    experimentos/m4_transformer_chunks_stdout_faulthandler_evidencia.log).

    Args:
        columna: "texto" o "texto_mask".
        dir_cache: Directorio raiz de la cache (subdir por columna).

    Returns:
        Dict {id_doc: [np.ndarray uint16 por chunk]}.

    Raises:
        RuntimeError: si falta la cache, su config no coincide con CONFIG
            o el contenido no cubre lo declarado en meta.json.
    """
    ruta_meta = Path(dir_cache) / "meta.json"
    if not ruta_meta.exists():
        raise RuntimeError(f"Falta {dir_cache}: ejecuta scripts/14_precache_chunks_beto.py")
    meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
    esperado = hash_config_chunking(
        CONFIG["modelo"], CONFIG["chunk_tokens"], CONFIG["chunk_stride"], CONFIG["max_chunks"]
    )
    if meta.get("config_hash") != esperado:
        raise RuntimeError(
            f"cache {dir_cache} generada con otra configuracion de chunking "
            f"(config_hash={meta.get('config_hash')} != {esperado}); regenera con scripts/14_precache_chunks_beto.py"
        )
    info = (meta.get("columnas") or {}).get(columna)
    if info is None:
        raise RuntimeError(f"cache {dir_cache} sin columna {columna}; ejecuta scripts/14_precache_chunks_beto.py")
    cache: dict[int, list[np.ndarray]] = {}
    for nombre in info["shards"]:
        ruta = Path(dir_cache) / columna / nombre
        if not ruta.exists():
            raise RuntimeError(f"shard faltante {ruta}; reejecuta scripts/14_precache_chunks_beto.py (reanuda)")
        with open(ruta, "rb") as manejador:
            for id_doc, chunks in pickle.load(manejador):
                cache[int(id_doc)] = chunks
    if len(cache) != int(info["n_docs"]):
        raise RuntimeError(
            f"cache {dir_cache}/{columna} incompleta: {len(cache)} docs != meta {info['n_docs']}; "
            "reejecuta scripts/14_precache_chunks_beto.py (reanuda)"
        )
    LOGGER.info("[cache] %s: %d docs, %d shards", columna, len(cache), len(info["shards"]))
    return cache


class ChunksDataset(Dataset):
    """Dataset de docs chunk-mean alimentado por cache pre-tokenizada.

    v2 iter2 (Fase V): en entrenamiento se usa SIEMPRE la cache (sin
    tokenizer, sin workers). El parametro tokenizer/textos queda solo para
    depuracion puntual (fallback fuera de entrenamiento); la tokenizacion
    en workers del DataLoader causaba el deadlock a escala completa.
    """

    def __init__(self, ids: pd.Series, etiquetas: pd.Series, clases: dict, cache: dict | None = None, tokenizer=None, textos: pd.Series | None = None):
        """Inicializa el dataset.

        Args:
            ids: Ids de los documentos (clave de la cache).
            etiquetas: Etiquetas de texto (area_primaria).
            clases: Mapeo area -> indice.
            cache: Dict {id: chunks uint16} de cargar_cache_chunks.
            tokenizer: Tokenizer (solo fallback de depuracion).
            textos: Textos de la columna (solo fallback de depuracion).

        Raises:
            ValueError: si no se entrega cache ni tokenizer + textos.
        """
        self.ids = [int(i) for i in ids]
        self.etiquetas = etiquetas.map(clases).tolist()
        self.cache = cache
        self.tokenizer = tokenizer
        self.textos = None if textos is None else list(textos)
        if cache is None and (tokenizer is None or self.textos is None):
            raise ValueError("ChunksDataset requiere cache o (tokenizer + textos)")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        """Devuelve los chunks del doc (uint16 -> int) y su etiqueta.

        Args:
            idx: Indice en el dataset.

        Returns:
            Dict {chunks: list[list[int]], labels: int}.

        Raises:
            KeyError: si el id no esta en la cache.
        """
        doc_id = self.ids[idx]
        if self.cache is not None:
            if doc_id not in self.cache:
                raise KeyError(f"doc {doc_id} sin chunks en la cache; reejecuta scripts/14_precache_chunks_beto.py")
            chunks = [chunk.tolist() for chunk in self.cache[doc_id]]
        else:
            enc = self.tokenizer(self.textos[idx], truncation=False)
            chunks = trocear_doc(
                enc["input_ids"],
                CONFIG["chunk_tokens"],
                CONFIG["chunk_stride"],
                CONFIG["max_chunks"],
            )
        return {"chunks": chunks, "labels": self.etiquetas[idx]}


def collate_chunks(lote: list[dict]) -> dict:
    """Funcion collate_fn del DataLoader de chunks (delega en colapsar_lote_chunks)."""
    return colapsar_lote_chunks(lote)


def forward_doc_chunk_mean(modelo, lote: dict, dispositivo: torch.device, micro: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward chunk-mean de un lote de docs (Chalkidis 2022).

    Codifica los chunks de cada doc (micro-lotes de `micro` chunks por
    forward para acotar VRAM), promedia los pooler outputs de sus chunks
    (mismo pooling que v1) y aplica la cabeza sobre el agregado.

    Args:
        modelo: BertForSequenceClassification.
        lote: Dict de colapsar_lote_chunks (input_ids, attention_mask, labels).
        dispositivo: cuda.
        micro: Chunks por forward; None = todos los chunks de un doc por
            forward (train: <=64 chunks, pico medido ~8.4 GB).

    Returns:
        Tupla (logits (n_docs, n_clases), labels (n_docs,)).
    """
    ids = lote["input_ids"].to(dispositivo, non_blocking=True)
    mask = lote["attention_mask"].to(dispositivo, non_blocking=True)
    n_filas = ids.shape[0]
    micro = n_filas if micro is None else min(micro, n_filas)
    reps = []
    for ini in range(0, n_filas, micro):
        salida = modelo.bert(input_ids=ids[ini : ini + micro], attention_mask=mask[ini : ini + micro])
        reps.append(salida.pooler_output)
    reps = torch.cat(reps, dim=0)  # (num_chunks_lote, 768)
    # Media por doc: cada doc aporta sus filas contiguas.
    logits_docs = []
    fila = 0
    for d in lote["_n_chunks"]:
        logits_docs.append(modelo.classifier(modelo.dropout(reps[fila : fila + d].mean(dim=0))))
        fila += d
    logits = torch.stack(logits_docs, dim=0)
    return logits, lote["labels"].to(dispositivo)


def entrenar_beto_chunk_mean(df: pd.DataFrame, columna: str, dispositivo: torch.device, etiqueta: str, limite: int | None = None, epocas: int | None = None, dir_cache: str | None = None) -> dict:
    """Fine-tunea BETO chunk-mean en un escenario y evalua en test (v2 Fase V).

    v2 iter2: los chunks se leen de la cache pre-tokenizada
    (cargar_cache_chunks) y los DataLoader usan num_workers=0 (sin procesos
    worker: el multiproceso se colgaba a escala completa; ver docstring).

    Args:
        df: DataFrame con id, area y particion (texto no se usa con cache).
        columna: texto (completo) o texto_mask (enmascarado).
        dispositivo: cuda.
        etiqueta: Nombre del escenario para logs/guardado.
        limite: Submuestra de docs por particion (smoke); None = completo.
        epocas: Sobreescribe CONFIG["epochs"] (smoke); None = config.
        dir_cache: Directorio de cache; None usa CONFIG["dir_cache_chunks"].

    Returns:
        Dict con macro_f1, accuracy, mejor_f1_val, historial por epoca,
        estadisticas de chunks y ruta del modelo.

    Raises:
        RuntimeError: si la cache falta, no cubre los ids o un split queda
            vacio tras el filtrado smoke.
    """
    torch.manual_seed(CONFIG["semilla"])
    np.random.seed(CONFIG["semilla"])
    epocas_cfg = CONFIG["epochs"] if epocas is None else epocas
    dir_cache = CONFIG["dir_cache_chunks"] if dir_cache is None else dir_cache
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["modelo"])
    cache = cargar_cache_chunks(columna, dir_cache)

    # Cobertura de la cache: full exige todos los ids; smoke filtra a los
    # docs disponibles (la cache smoke es un subconjunto pequeno).
    ids_df = {int(i) for i in df["id"]}
    faltantes = ids_df - set(cache)
    if faltantes:
        if limite is None:
            raise RuntimeError(
                f"cache {dir_cache}/{columna} incompleta: faltan {len(faltantes)} ids "
                f"(p. ej. {sorted(faltantes)[:5]}); ejecuta scripts/14_precache_chunks_beto.py"
            )
        LOGGER.info(
            "[smoke] cache %s cubre %d/%d docs; se filtra a los presentes",
            columna, len(ids_df) - len(faltantes), len(ids_df),
        )
        df = df[df["id"].astype(int).isin(cache)].reset_index(drop=True)

    clases = {c: i for i, c in enumerate(sorted(df["area_primaria"].unique()))}
    tr = df[df["particion"] == "train"]
    va = df[df["particion"] == "val"]
    te = df[df["particion"] == "test"]
    for nombre, particion in [("train", tr), ("val", va), ("test", te)]:
        if len(particion) == 0:
            raise RuntimeError(
                f"split {nombre} vacio tras intersectar con la cache {dir_cache}/{columna}; "
                "genera una cache que cubra los ids del split (scripts/14_precache_chunks_beto.py)"
            )
    if limite is not None:
        # Submuestra estratificada por area para conservar el desbalanceo.
        tr = submuestra_estratificada(tr, limite)
        va = submuestra_estratificada(va, limite)
        te = submuestra_estratificada(te, limite)
        LOGGER.info("[smoke] submuestra: train=%d val=%d test=%d", len(tr), len(va), len(te))
        for nombre, particion in [("train", tr), ("val", va), ("test", te)]:
            if len(particion) == 0:
                raise RuntimeError(f"split {nombre} vacio en la submuestra smoke (cache {dir_cache}/{columna})")

    # v2 iter2 (Fase V): num_workers=0 + pin_memory=False. El DataLoader
    # multiproceso (8 workers + fork) se colgaba SIEMPRE a escala de
    # 16.557 docs (Python 3.14 + fork, incidente 2026-09-11; evidencia en
    # experimentos/m4_transformer_chunks_stdout_faulthandler_evidencia.log).
    # La tokenizacion ya ocurrio UNA vez en scripts/14_precache_chunks_beto.py.
    dl_tr = DataLoader(
        ChunksDataset(tr["id"], tr["area_primaria"], clases, cache=cache),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_chunks,
        pin_memory=False,
        persistent_workers=False,
    )
    dl_va = DataLoader(
        ChunksDataset(va["id"], va["area_primaria"], clases, cache=cache),
        batch_size=CONFIG["batch_eval"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_chunks,
        pin_memory=False,
    )
    dl_te = DataLoader(
        ChunksDataset(te["id"], te["area_primaria"], clases, cache=cache),
        batch_size=CONFIG["batch_eval"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_chunks,
        pin_memory=False,
    )

    modelo = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["modelo"], num_labels=len(clases)
    ).to(dispositivo)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=CONFIG["lr"])
    acum = CONFIG["batch_efectivo_docs"]
    total_pasos = max(1, len(dl_tr) * epocas_cfg // acum)
    scheduler = get_linear_schedule_with_warmup(optimizador, int(0.06 * total_pasos), total_pasos)
    scaler = torch.amp.GradScaler(dispositivo.type, enabled=CONFIG["amp"])

    mejor_f1 = -1.0
    ruta_modelo = f"{CONFIG['dir_modelos']}/beto_{etiqueta}"
    historial = []
    chunks_stats = {"total_docs": 0, "total_chunks": 0, "max_obs": 0}
    t0 = time.time()
    for epoca in range(epocas_cfg):
        modelo.train()  # corrige el bug v1 (eval() sin restaurar train())
        perdida_epoca = 0.0
        n_docs = 0
        optimizador.zero_grad()
        for i, lote in enumerate(dl_tr):
            with torch.autocast(dispositivo.type, dtype=torch.float16, enabled=CONFIG["amp"]):
                logits, y = forward_doc_chunk_mean(modelo, lote, dispositivo)
                perdida = torch.nn.functional.cross_entropy(logits, y)
            scaler.scale(perdida / acum).backward()
            perdida_epoca += perdida.item()
            n_docs += 1
            chunks_stats["total_docs"] += 1
            chunks_stats["total_chunks"] += lote["input_ids"].shape[0]
            chunks_stats["max_obs"] = max(chunks_stats["max_obs"], int(lote["input_ids"].shape[0]))
            if (i + 1) % acum == 0:
                scaler.unscale_(optimizador)
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                scaler.step(optimizador)
                scaler.update()
                scheduler.step()
                optimizador.zero_grad()
                if (i + 1) % (acum * CONFIG["log_cada_lotes"]) == 0:
                    LOGGER.info(
                        "[%s] epoca %d: %d/%d docs, perdida=%.4f, %.1f docs/s",
                        etiqueta, epoca + 1, i + 1, len(dl_tr),
                        perdida_epoca / n_docs, n_docs / max(time.time() - t0, 1e-9),
                    )
        # Paso final si la epoca no cierra exacto en acum.
        if n_docs % acum != 0:
            scaler.unscale_(optimizador)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            scaler.step(optimizador)
            scaler.update()  # fix 2026-09-11: faltaba y crasheaba el 1er unscale_ del epoch siguiente
            scheduler.step()
            optimizador.zero_grad()
        # Validacion al final de cada epoca (logica v1).
        f1_va, perdida_va = evaluar_beto_chunk_mean(modelo, dl_va, dispositivo, con_perdida=True)
        LOGGER.info(
            "[%s] epoca %d: perdida_train=%.4f perdida_val=%.4f macro-F1 val=%.4f",
            etiqueta, epoca + 1, perdida_epoca / max(n_docs, 1), perdida_va, f1_va,
        )
        historial.append({
            "epoca": epoca + 1,
            "perdida_train": perdida_epoca / max(n_docs, 1),
            "perdida_val": perdida_va,
            "macro_f1_val": float(f1_va),
        })
        if f1_va > mejor_f1:
            mejor_f1 = f1_va
            Path(CONFIG["dir_modelos"]).mkdir(parents=True, exist_ok=True)
            modelo.save_pretrained(ruta_modelo)
            tokenizer.save_pretrained(ruta_modelo)

    # Evaluar el mejor checkpoint en test (logica v1).
    modelo = AutoModelForSequenceClassification.from_pretrained(ruta_modelo).to(dispositivo)
    y_pred, y_true = predecir_chunk_mean(modelo, dl_te, dispositivo)
    inv = {v: k for k, v in clases.items()}
    y_pred_lbl = [inv[int(p)] for p in y_pred]
    y_true_lbl = [inv[int(t)] for t in y_true]
    return {
        "macro_f1": float(f1_score(y_true_lbl, y_pred_lbl, average="macro")),
        "accuracy": float(accuracy_score(y_true_lbl, y_pred_lbl)),
        "modelo": ruta_modelo,
        "mejor_f1_val": float(mejor_f1),
        "historial_epocas": historial,
        "chunks_promedio_doc": chunks_stats["total_chunks"] / max(chunks_stats["total_docs"], 1),
        "chunks_max_obs": chunks_stats["max_obs"],
    }


def evaluar_beto_chunk_mean(modelo, dl: DataLoader, dispositivo: torch.device, con_perdida: bool = False) -> tuple[float, float] | float:
    """Evalua macro-F1 (y perdida opcional) chunk-mean sobre un DataLoader.

    Args:
        modelo: Modelo BETO.
        dl: DataLoader de chunks (batch de docs).
        dispositivo: cuda.
        con_perdida: Si True, devuelve tambien la perdida media por doc.

    Returns:
        macro-F1 o tupla (macro-F1, perdida) si con_perdida.
    """
    modelo.eval()
    preds, trues, perdidas = [], [], []
    with torch.no_grad():
        for lote in dl:
            with torch.autocast(dispositivo.type, dtype=torch.float16, enabled=CONFIG["amp"]):
                logits, y = forward_doc_chunk_mean(modelo, lote, dispositivo, micro=CONFIG["micro_chunks_eval"])
            if con_perdida:
                perdidas.append(torch.nn.functional.cross_entropy(logits, y).item())
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            trues.extend(y.cpu().tolist())
    f1 = f1_score(trues, preds, average="macro")
    if con_perdida:
        return f1, float(np.mean(perdidas)) if perdidas else float("nan")
    return f1


def predecir_chunk_mean(modelo, dl: DataLoader, dispositivo: torch.device) -> tuple[list, list]:
    """Predice etiquetas (codes) chunk-mean sobre un DataLoader.

    Args:
        modelo: Modelo BETO.
        dl: DataLoader de chunks.
        dispositivo: cuda.

    Returns:
        Tupla (y_pred codes, y_true codes).
    """
    modelo.eval()
    preds, trues = [], []
    with torch.no_grad():
        for lote in dl:
            with torch.autocast(dispositivo.type, dtype=torch.float16, enabled=CONFIG["amp"]):
                logits, y = forward_doc_chunk_mean(modelo, lote, dispositivo, micro=CONFIG["micro_chunks_eval"])
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            trues.extend(y.cpu().tolist())
    return preds, trues


def parsear_args() -> argparse.Namespace:
    """Parsea los argumentos de linea de comandos.

    Returns:
        Namespace con estrategia, limite, epocas, salida y cache_dir.
    """
    parser = argparse.ArgumentParser(description="Fine-tuning BETO M4 (v2 Fase V: chunk-mean)")
    parser.add_argument("--estrategia", choices=["chunk_mean", "truncado"], default=ESTRATEGIA_LONGITUD,
                        help="Estrategia de longitud (default: chunk_mean)")
    parser.add_argument("--limite", type=int, default=None,
                        help="Submuestra de docs por particion (smoke test)")
    parser.add_argument("--epocas", type=int, default=None,
                        help="Sobreescribe epocas (smoke test)")
    parser.add_argument("--salida", type=str, default=None,
                        help="Ruta del JSON de metricas (default: segun estrategia)")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Directorio de la cache de chunks (default: data/cache_chunks_beto; smoke: /tmp/opencode/cache_chunks_smoke)")
    return parser.parse_args()


def aplicar_rutas(args: argparse.Namespace) -> None:
    """Decide rutas de salida (metricas, modelos, log) segun estrategia.

    OBLIGATORIO llamar ANTES de configurar_logging(): el FileHandler abre
    CONFIG["ruta_log"] y en v1 ese path apuntaba al log del run congelado
    (incidente 2026-09-10: el smoke trunco experimentos/m4_transformer.log;
    restaurado desde m4_transformer_stdout.log).

    Args:
        args: Namespace de parsear_args().
    """
    if args.cache_dir is not None:
        CONFIG["dir_cache_chunks"] = args.cache_dir
    if args.estrategia == "chunk_mean":
        CONFIG["dir_modelos"] = "modelos/m4_transformer_chunks"
        CONFIG["ruta_metricas"] = "resultados/m4_transformer_chunks.json"
        CONFIG["ruta_log"] = "experimentos/m4_transformer_chunks.log"
        CONFIG["vram_min_gb"] = 10.0  # pico medido ~7.4 GB + margen
    else:
        # Proteccion ADR-11: el modo truncado v1 no puede sobrescribir
        # los resultados v1 congelados.
        if args.salida is None:
            raise RuntimeError(
                "modo truncado (v1) sin --salida: aborta para proteger "
                "resultados/m4_transformer.json (ADR-11). Usa --salida /tmp/opencode/xxx.json"
            )
        if args.limite is not None or args.epocas is not None:
            raise RuntimeError("--limite/--epocas solo aplican a chunk_mean; truncado es v1 exacto")
        CONFIG["ruta_metricas"] = args.salida
        CONFIG["ruta_log"] = "experimentos/m4_transformer_truncado_v2.log"
        CONFIG["dir_modelos"] = "modelos/m4_transformer_truncado_v2"
    if args.limite is not None:
        # Smoke: modelos, log y (sin --salida) JSON a /tmp/opencode/, NUNCA a resultados/.
        CONFIG["dir_modelos"] = "/tmp/opencode/smoke_m4_transformer_chunks"
        CONFIG["ruta_log"] = "/tmp/opencode/smoke_m4_transformer_chunks.log"
        CONFIG["ruta_metricas"] = "/tmp/opencode/smoke_m4_transformer_chunks.json"
    if args.salida is not None:
        # El --salida explicito siempre gana (seguro: solo /tmp en smoke).
        CONFIG["ruta_metricas"] = args.salida


def main(args: argparse.Namespace) -> None:
    """Ejecuta el fine-tuning BETO en ambos escenarios (chunk_mean o truncado).

    Args:
        args: Namespace de parsear_args(); aplicar_rutas ya ajusto CONFIG.
    """
    t0 = time.time()
    LOGGER.info("inicio transformer M4 (BETO) estrategia=%s", args.estrategia)
    if args.limite is not None:
        LOGGER.info("[smoke] limite=%d docs/particion, salidas a /tmp/opencode/", args.limite)
    dispositivo = esperar_vram()
    if dispositivo.type != "cuda":
        raise RuntimeError("fine-tuning BETO requiere GPU; aborta (documentado)")
    df = cargar_datos(construir_mask=args.estrategia == "truncado")
    resultados = {
        "fecha": pd.Timestamp.now().isoformat(),
        "config": CONFIG,
        "modelo": CONFIG["modelo"],
        "estrategia_longitud": f"chunk_mean_{CONFIG['chunk_tokens']}_stride{CONFIG['chunk_stride']}_max{CONFIG['max_chunks']}"
        if args.estrategia == "chunk_mean" else "truncado_256_256_v1",
    }
    for escenario, columna in [("completo", "texto"), ("enmascarado", "texto_mask")]:
        LOGGER.info("fine-tuning escenario %s", escenario)
        if args.estrategia == "chunk_mean":
            resultados[escenario] = entrenar_beto_chunk_mean(
                df, columna, dispositivo, escenario, limite=args.limite, epocas=args.epocas,
                dir_cache=CONFIG["dir_cache_chunks"],
            )
        else:
            resultados[escenario] = entrenar_beto(df, columna, dispositivo, escenario)
        LOGGER.info("[%s] macro-F1 test=%.4f acc=%.4f", escenario, resultados[escenario]["macro_f1"], resultados[escenario]["accuracy"])
    resultados["gap_leakage"] = resultados["completo"]["macro_f1"] - resultados["enmascarado"]["macro_f1"]
    resultados["tiempo_total_seg"] = time.time() - t0
    Path(CONFIG["ruta_metricas"]).parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG["ruta_metricas"], "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2, default=str)
    LOGGER.info("fin en %.0f s; gap leakage=%.4f", resultados["tiempo_total_seg"], resultados["gap_leakage"])


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)  # bug forkserver Python 3.14 (skill gpu-optimization)
    torch.set_num_threads(32)  # 32 hilos logicos (skill ml-environment)
    ARGS = parsear_args()
    aplicar_rutas(ARGS)  # ANTES de configurar_logging: abre el log correcto
    configurar_logging()
    main(ARGS)
