"""E-018 — Meta-modelo enriquecido con features densas (condicionamiento por longitud).

Pregunta
    El análisis de errores mostró que la longitud domina el error (52,5% en el quintil más corto
    vs 37,2% en el más largo). ¿Aprovecha el meta-modelo esa estructura si recibe también las
    features densas (longitud, retuit, menciones, URLs, textometría) además de los scores de los
    modelos base?

Hipótesis
    H16: darle al meta-modelo la longitud y el estilo le permite **condicionar** la combinación de
    modelos por tipo de tweet (p. ej. confiar más en el canal de menciones en tweets cortos),
    mejorando macro-F1 sin reentrenar los modelos base.

Protocolo
    Leave-one-fold-out idéntico al de la Fase 20b, con las mismas 9 fuentes de scores.
    Se comparan tres variantes: scores (referencia), scores + densas, y densas + interacción
    longitud×ordinal.

Salida
    ``experiments/e018_meta_enriquecido.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, data, textometry  # noqa: E402
from src.config import set_seed  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "e018_meta_enriquecido.json"
BASE_SOURCES = [
    ("oof_E013_margenes.npy", False), ("oof_E016_margenes.npy", False),
    ("oof_E-009.npy", True), ("oof_E-011.npy", True), ("oof_E-010.npy", True),
    ("oof_E014_margenes.npy", False), ("oof_E015_ordinal.npy", False),
    ("oof_E015_softmax.npy", True), ("oof_E017_probabilidades.npy", True),
]
DENSE_FEATURES = [name for name in textometry.TEXTOMETRY_FEATURES + textometry.STYLE_FEATURES
                  if name not in {"n_emoji", "emoji_diversity", "has_emoji"}]


def load_scores() -> np.ndarray:
    """Carga las 9 matrices OOF de scores en el orden canónico."""
    matrices = []
    for filename, probabilistic in BASE_SOURCES:
        matrix = np.load(cfg.FEATURES / filename)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        matrices.append(np.log(np.clip(matrix, 1e-12, None)) if probabilistic else matrix)
    return np.concatenate(matrices, axis=1)


def evaluate(features: np.ndarray, y: np.ndarray, folds: list[dict]) -> dict:
    """Evalúa un conjunto de features de meta-modelo con leave-one-fold-out."""
    scores = []
    for fold in folds:
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        meta = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                    random_state=cfg.SEED))])
        meta.fit(features[train_idx], y[train_idx])
        scores.append(float(f1_score(y[valid_idx], meta.predict(features[valid_idx]),
                                     average="macro", labels=cfg.CLASSES)))
    return {"macro_f1_mean": float(np.mean(scores)), "macro_f1_std": float(np.std(scores)),
            "per_fold": [round(value, 4) for value in scores]}


def main() -> int:
    """Compara el meta-modelo con y sin features densas."""
    set_seed()
    started = time.time()
    train = data.load_train()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]
    y = train[cfg.TARGET].astype(str).to_numpy()
    folds = cv.make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")

    scores = load_scores()
    dense = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")[DENSE_FEATURES].to_numpy(float)
    ordinal = np.load(cfg.FEATURES / "oof_E015_ordinal.npy").reshape(-1, 1)
    interactions = np.column_stack([
        dense[:, DENSE_FEATURES.index("n_chars")] * ordinal[:, 0],
        dense[:, DENSE_FEATURES.index("is_rt")] * ordinal[:, 0],
        dense[:, DENSE_FEATURES.index("n_mention")] * ordinal[:, 0],
    ])

    variants = {
        "scores (referencia)": scores,
        "scores + densas": np.concatenate([scores, dense], axis=1),
        "scores + densas + interacciones": np.concatenate([scores, dense, interactions], axis=1),
    }
    results = {}
    for label, features in variants.items():
        results[label] = evaluate(features, y, folds)
        print(f"[{label:34s}] macro-F1 LOFO {results[label]['macro_f1_mean']:.4f} "
              f"± {results[label]['macro_f1_std']:.4f} ({features.shape[1]} features)", flush=True)

    best = max(results, key=lambda key: results[key]["macro_f1_mean"])
    payload = {"resultados": results, "mejor": best,
               "referencia_fase20b": 0.5748, "segundos": round(time.time() - started, 1)}
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMejor variante: {best} = {results[best]['macro_f1_mean']:.4f} "
          f"(referencia Fase 20b: 0,5748)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
