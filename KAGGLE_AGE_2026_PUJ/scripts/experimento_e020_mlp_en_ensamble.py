"""E-020 — ¿Aporta el MLP denso (E-019b) al ensamble como miembro diverso?

Pregunta
    El MLP sobre densas+SVD rinde mal en solitario (0,4071) pero ve el problema de otra forma.
    ¿Su inclusión en el meta-modelo mejora la combinación?

Protocolo
    Leave-one-fold-out idéntico a E-018, comparando la referencia (9 fuentes + 35 densas) contra
    la variante que añade las probabilidades del MLP.

Salida
    ``experiments/e020_mlp_en_ensamble.json``.
"""

from __future__ import annotations

import json
import sys
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

FILES = [("oof_E013_margenes.npy", False), ("oof_E016_margenes.npy", False),
         ("oof_E-009.npy", True), ("oof_E-011.npy", True), ("oof_E-010.npy", True),
         ("oof_E014_margenes.npy", False), ("oof_E015_ordinal.npy", False),
         ("oof_E015_softmax.npy", True), ("oof_E017_probabilidades.npy", True)]
DENSE = [name for name in textometry.TEXTOMETRY_FEATURES + textometry.STYLE_FEATURES
         if name not in {"n_emoji", "emoji_diversity", "has_emoji"}]


def load_sources(files: list[tuple[str, bool]]) -> np.ndarray:
    """Carga las matrices OOF indicadas y aplica log a las probabilísticas."""
    matrices = []
    for filename, probabilistic in files:
        matrix = np.load(cfg.FEATURES / filename)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        matrices.append(np.log(np.clip(matrix, 1e-12, None)) if probabilistic else matrix)
    return np.concatenate(matrices, axis=1)


def evaluate(features: np.ndarray, y: np.ndarray, folds: list[dict]) -> tuple[float, float]:
    """macro-F1 medio y desviación con leave-one-fold-out."""
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
    return float(np.mean(scores)), float(np.std(scores))


def main() -> int:
    """Compara el ensamble con y sin el MLP denso."""
    set_seed()
    train = data.load_train()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]
    y = train[cfg.TARGET].astype(str).to_numpy()
    folds = cv.make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")

    dense_train = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")[DENSE].to_numpy(float)
    mlp = np.log(np.clip(np.load(cfg.FEATURES / "oof_E019b_densas_svd.npy"), 1e-12, None))
    scores = load_sources(FILES)

    reference = np.concatenate([scores, dense_train], axis=1)
    with_mlp = np.concatenate([scores, mlp, dense_train], axis=1)

    mean_reference, std_reference = evaluate(reference, y, folds)
    print(f"E-018 (referencia, {reference.shape[1]} features): "
          f"{mean_reference:.4f} ± {std_reference:.4f}", flush=True)
    mean_mlp, std_mlp = evaluate(with_mlp, y, folds)
    print(f"E-020 (+ MLP densas+SVD, {with_mlp.shape[1]} features): "
          f"{mean_mlp:.4f} ± {std_mlp:.4f}", flush=True)
    print(f"aporte del MLP al ensamble: {mean_mlp - mean_reference:+.4f}")

    payload = {"e018_referencia": {"macro_f1": mean_reference, "std": std_reference},
               "e020_con_mlp": {"macro_f1": mean_mlp, "std": std_mlp},
               "aporte": mean_mlp - mean_reference,
               "decision": ("incluir el MLP" if mean_mlp > mean_reference + 0.001
                            else "no incluir el MLP (aporte nulo o negativo)")}
    (cfg.EXPERIMENTS / "e020_mlp_en_ensamble.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"decisión: {payload['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
