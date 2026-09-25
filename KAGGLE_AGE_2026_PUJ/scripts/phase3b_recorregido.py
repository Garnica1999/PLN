"""FASE 3b — Corrección del bug de forma de palabra, recálculo y re-evaluación.

Contexto (auditoría de Fase 3)
    Se detectaron dos defectos en las features estructuradas de segunda generación:
    1. ``TOKEN_RE`` solo aceptaba minúsculas en la alternativa de palabra, de modo que las palabras
       capitalizadas o en mayúsculas se fragmentaban y **cuatro features de forma quedaban en cero**
       (`shape_allcaps_ratio`, `shape_capitalized_ratio`, `shape_alphanumeric_ratio`,
       `first_word_is_upper`).
    2. ``shape_alphanumeric_ratio`` seguía siendo cero por definición: el tokenizador separa dígitos
       de letras, así que ningún token contenía ambos.
    Además, en el pipeline de v11 el bloque NB-SVM se anexaba al final del lado test pero no del lado
    train, desplazando 6 columnas y colapsando el meta-modelo MLP.

Acciones
    - Corregir el tokenizador y recalcular las features alfanuméricas con una expresión propia.
    - Recalcular ``age_structured_features.parquet`` y ``_test.parquet``.
    - Re-evaluar LogReg vs MLP como meta-modelo con las features corregidas (LOFO 5 folds).

Salida: ``experiments/phase3b_recorregido.json``.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import advanced_features as af  # noqa: E402
from src import config as cfg  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "phase3b_recorregido.json"

#: Tokens que contienen letras **y** dígitos (p. ej. "abc123"), que el tokenizador separa.
ALNUM_RE = re.compile(r"\b(?=\w*\d)(?=\w*[A-Za-z])\w+\b")


def add_alphanumeric_ratio(frame: pd.DataFrame, texts: pd.Series) -> pd.DataFrame:
    """Recalcula ``shape_alphanumeric_ratio`` sobre el texto crudo."""
    counts = texts.fillna("").astype(str).apply(lambda value: len(ALNUM_RE.findall(value)))
    words = texts.fillna("").astype(str).str.findall(af.TOKEN_RE.pattern).apply(len).replace(0, 1)
    frame = frame.copy()
    frame["shape_alphanumeric_ratio"] = (counts / words).to_numpy()
    return frame


def build_local(texts: pd.Series, verbose: bool = True) -> pd.DataFrame:
    """Construye las seis familias locales con las correcciones aplicadas."""
    frames = []
    for name, builder in (("A_distribucion", af.distribution_features),
                          ("B_char", af.char_distribution_features),
                          ("D_riqueza", af.lexical_richness_features),
                          ("HT_forma", af.word_shape_features),
                          ("IMU_posicion", af.position_features),
                          ("NRS_estructura", af.structure_entropy_features)):
        started = time.time()
        frame = builder(texts)
        frames.append(frame)
        if verbose:
            print(f"    [{name}] {frame.shape[1]} ({round(time.time() - started, 1)}s)", flush=True)
    combined = pd.concat(frames, axis=1)
    return add_alphanumeric_ratio(combined, texts)


def main() -> int:
    """Recalcula las features estructuradas y re-evalúa los meta-modelos."""
    started = time.time()
    from src import data
    from scripts.phase2_ponderacion import build_features, evaluate_global

    print("[1/3] recalculando features estructuradas de train...", flush=True)
    train = data.load_train()
    test = data.load_test()
    structured_train = build_local(train[cfg.TEXT_COL])
    structured_test = build_local(test[cfg.TEXT_COL])
    structured_train.to_parquet(cfg.FEATURES / "age_structured_features.parquet", index=False)
    structured_test.to_parquet(cfg.FEATURES / "age_structured_features_test.parquet", index=False)
    zero_columns = [column for column in structured_train.columns
                    if structured_train[column].std() == 0]
    print(f"    train {structured_train.shape} | test {structured_test.shape} | "
          f"columnas con varianza nula: {zero_columns}", flush=True)

    print("[2/3] reconstruyendo la matriz del meta-modelo...", flush=True)
    features, y, dense, tweet_type, length_band = build_features()
    print(f"    features: {features.shape}", flush=True)

    print("[3/3] re-evaluando meta-modelos (LOFO 5 folds)...", flush=True)
    from src import cv
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    folds = cv.make_folds(y, groups=clusters["cluster"].to_numpy()[: len(y)],
                          n_splits=5, n_repeats=1, scheme="group")
    results = {}
    for label, kind in (("LogReg", "logreg"), ("MLP", "mlp")):
        mean, std, per_fold = evaluate_global(features, y, folds, kind)
        results[label] = {"macro_f1": round(mean, 4), "std": round(std, 4), "per_fold": per_fold}
        print(f"    {label:7s}: {mean:.4f} ± {std:.4f} folds={per_fold}", flush=True)

    payload = {
        "features_estructuradas": {"n": int(structured_train.shape[1]),
                                   "columnas_varianza_nula": zero_columns},
        "meta_logreg": results["LogReg"], "meta_mlp": results["MLP"],
        "delta_mlp": round(results["MLP"]["macro_f1"] - results["LogReg"]["macro_f1"], 4),
        "referencia_previa": {"logreg": 0.5906, "mlp": 0.5966},
        "segundos": round(time.time() - started, 1),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ndelta MLP vs LogReg: {payload['delta_mlp']:+.4f} "
          f"(referencia previa: +0,0060)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
