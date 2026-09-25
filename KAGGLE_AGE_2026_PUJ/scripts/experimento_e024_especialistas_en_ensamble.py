"""E-024 — ¿Aportan los especialistas por longitud al ensamble?

Evalúa con leave-one-fold-out la incorporación de los dos SVM especializados (E-023a cortos,
E-023b largos) a la mejor combinación conocida (10 fuentes + ordinal + centroides + densas).

Salida: ``experiments/e024_especialistas_en_ensamble.json``.
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

RESULTS_PATH = cfg.EXPERIMENTS / "e024_especialistas_en_ensamble.json"
BASE_SOURCES = [("oof_E013_margenes.npy", False), ("oof_E016_margenes.npy", False),
                ("oof_E-009.npy", True), ("oof_E-011.npy", True), ("oof_E-010.npy", True),
                ("oof_E014_margenes.npy", False), ("oof_E015_ordinal.npy", False),
                ("oof_E015_softmax.npy", True), ("oof_E017_probabilidades.npy", True),
                ("oof_E019b_densas_svd.npy", True)]
ORDINAL_SOURCE = [("oof_E021a_ordinal.npy", True)]
CENTROID_SOURCE = [("oof_E021b_rocchio.npy", True)]
SPECIALIST_SOURCES = [("oof_E023a_corto.npy", False), ("oof_E023b_largo.npy", False)]
DENSE_FEATURES = [name for name in textometry.TEXTOMETRY_FEATURES + textometry.STYLE_FEATURES
                  if name not in {"n_emoji", "emoji_diversity", "has_emoji"}]


def load(sources: list[tuple[str, bool]]) -> np.ndarray:
    """Carga matrices OOF con log en las probabilísticas."""
    matrices = []
    for filename, probabilistic in sources:
        matrix = np.load(cfg.FEATURES / filename)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        matrices.append(np.log(np.clip(matrix, 1e-12, None)) if probabilistic else matrix)
    return np.concatenate(matrices, axis=1)


def evaluate(features: np.ndarray, y: np.ndarray, folds: list[dict]) -> tuple[float, float]:
    """macro-F1 con meta-modelo LogReg y leave-one-fold-out."""
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
    """Compara las combinaciones con y sin especialistas por longitud."""
    set_seed()
    started = time.time()
    train = data.load_train()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]
    y = train[cfg.TARGET].astype(str).to_numpy()
    folds = cv.make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")

    dense = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")[DENSE_FEATURES].to_numpy(float)
    base = load(BASE_SOURCES)
    ordinal = load(ORDINAL_SOURCE)
    centroids = load(CENTROID_SOURCE)
    specialists = load(SPECIALIST_SOURCES)

    variants = {
        "referencia (10 fuentes + densas)": np.concatenate([base, dense], axis=1),
        "mejor conocida (+ordinal+centroides)": np.concatenate([base, ordinal, centroids, dense],
                                                              axis=1),
        "+ especialistas de longitud": np.concatenate([base, ordinal, centroids, specialists, dense],
                                                     axis=1),
    }
    rows = []
    for label, features in variants.items():
        mean, std = evaluate(features, y, folds)
        rows.append({"combinacion": label, "n_features": features.shape[1],
                     "macro_f1": round(mean, 4), "std": round(std, 4)})
        print(f"[{label:38s}] {mean:.4f} ± {std:.4f} ({features.shape[1]} features)", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(cfg.TABLES / "e024_combinaciones.csv", index=False)
    best = table.loc[table["macro_f1"].idxmax()]
    payload = {
        "combinaciones": rows,
        "mejor": best["combinacion"],
        "mejor_macro_f1": float(best["macro_f1"]),
        "aporte_especialistas": round(float(table.iloc[2]["macro_f1"] - table.iloc[1]["macro_f1"]), 4),
        "segundos": round(time.time() - started, 1),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAporte de los especialistas: {payload['aporte_especialistas']:+.4f}")
    print(f"Mejor combinación: {payload['mejor']} = {payload['mejor_macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
