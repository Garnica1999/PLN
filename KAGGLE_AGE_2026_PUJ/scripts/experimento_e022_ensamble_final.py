"""E-022 — Ensamble con miembros ordinales/centroides y meta-modelo no lineal.

Pregunta
    1. ¿Añaden el modelo ordinal acumulativo (E-021a) y el de centroides (E-021b) al ensamble?
    2. ¿Un meta-modelo no lineal (LightGBM) aprovecha mejor las 90+ features del stack que la
       regresión logística?

Protocolo
    Leave-one-fold-out idéntico a E-018/E-020 sobre las mismas particiones agrupadas. Todas las
    matrices son out-of-fold, por lo que ningún modelo ve la etiqueta del fold que predice.

Salida
    ``experiments/e022_ensamble_final.json`` y ``reports/tables/e022_combinaciones.csv``.
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

RESULTS_PATH = cfg.EXPERIMENTS / "e022_ensamble_final.json"

#: Fuentes de scores (archivo, ¿aplicar log?).
BASE_SOURCES = [("oof_E013_margenes.npy", False), ("oof_E016_margenes.npy", False),
                ("oof_E-009.npy", True), ("oof_E-011.npy", True), ("oof_E-010.npy", True),
                ("oof_E014_margenes.npy", False), ("oof_E015_ordinal.npy", False),
                ("oof_E015_softmax.npy", True), ("oof_E017_probabilidades.npy", True),
                ("oof_E019b_densas_svd.npy", True)]
NEW_SOURCES = {"ordinal_acumulativo": ("oof_E021a_ordinal.npy", True),
               "centroides": ("oof_E021b_rocchio.npy", True)}

DENSE_FEATURES = [name for name in textometry.TEXTOMETRY_FEATURES + textometry.STYLE_FEATURES
                  if name not in {"n_emoji", "emoji_diversity", "has_emoji"}]


def load_sources(sources: list[tuple[str, bool]]) -> np.ndarray:
    """Carga matrices OOF y aplica log a las probabilísticas."""
    matrices = []
    for filename, probabilistic in sources:
        matrix = np.load(cfg.FEATURES / filename)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        matrices.append(np.log(np.clip(matrix, 1e-12, None)) if probabilistic else matrix)
    return np.concatenate(matrices, axis=1)


def evaluate_logreg(features: np.ndarray, y: np.ndarray, folds: list[dict]) -> tuple[float, float]:
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


def evaluate_lightgbm(features: np.ndarray, y: np.ndarray, folds: list[dict],
                      n_estimators: int = 250) -> tuple[float, float]:
    """macro-F1 con meta-modelo LightGBM y leave-one-fold-out."""
    import lightgbm as lgb

    scores = []
    for fold in folds:
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        model = lgb.LGBMClassifier(objective="multiclass", num_class=len(cfg.CLASSES),
                                   n_estimators=n_estimators, learning_rate=0.1,
                                   num_leaves=31, min_child_samples=50, subsample=0.9,
                                   colsample_bytree=0.8, reg_lambda=1.0, n_jobs=32,
                                   random_state=cfg.SEED, verbose=-1)
        model.fit(features[train_idx], y[train_idx])
        scores.append(float(f1_score(y[valid_idx], model.predict(features[valid_idx]),
                                     average="macro", labels=cfg.CLASSES)))
        print(f"    [lgbm fold {fold['fold']}] macro-F1 {scores[-1]:.4f}", flush=True)
    return float(np.mean(scores)), float(np.std(scores))


def main() -> int:
    """Evalúa las combinaciones finales del ensamble."""
    set_seed()
    started = time.time()
    train = data.load_train()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]
    y = train[cfg.TARGET].astype(str).to_numpy()
    folds = cv.make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")

    dense = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")[DENSE_FEATURES].to_numpy(float)
    base = load_sources(BASE_SOURCES)
    extra_ordinal = load_sources([NEW_SOURCES["ordinal_acumulativo"]])
    extra_centroids = load_sources([NEW_SOURCES["centroides"]])

    reference = np.concatenate([base, dense], axis=1)
    with_ordinal = np.concatenate([base, extra_ordinal, dense], axis=1)
    with_both = np.concatenate([base, extra_ordinal, extra_centroids, dense], axis=1)

    rows: list[dict] = []

    mean_ref, std_ref = evaluate_logreg(reference, y, folds)
    rows.append({"combinacion": "referencia: 10 fuentes + densas (LogReg meta)",
                 "n_features": reference.shape[1], "meta": "LogReg",
                 "macro_f1": round(mean_ref, 4), "std": round(std_ref, 4)})
    print(f"[referencia] {mean_ref:.4f} ± {std_ref:.4f}", flush=True)

    mean_ord, std_ord = evaluate_logreg(with_ordinal, y, folds)
    rows.append({"combinacion": "+ modelo ordinal acumulativo", "n_features": with_ordinal.shape[1],
                 "meta": "LogReg", "macro_f1": round(mean_ord, 4), "std": round(std_ord, 4)})
    print(f"[+ordinal] {mean_ord:.4f} ± {std_ord:.4f}", flush=True)

    mean_both, std_both = evaluate_logreg(with_both, y, folds)
    rows.append({"combinacion": "+ ordinal + centroides", "n_features": with_both.shape[1],
                 "meta": "LogReg", "macro_f1": round(mean_both, 4), "std": round(std_both, 4)})
    print(f"[+ordinal+centroides] {mean_both:.4f} ± {std_both:.4f}", flush=True)

    best_features = with_both if mean_both >= max(mean_ref, mean_ord) else (
        with_ordinal if mean_ord >= mean_ref else reference)
    best_label = ("+ ordinal + centroides" if best_features is with_both else
                  "+ ordinal" if best_features is with_ordinal else "referencia")
    mean_lgbm, std_lgbm = evaluate_lightgbm(best_features, y, folds)
    rows.append({"combinacion": f"{best_label} con meta LightGBM",
                 "n_features": best_features.shape[1], "meta": "LightGBM",
                 "macro_f1": round(mean_lgbm, 4), "std": round(std_lgbm, 4)})
    print(f"[{best_label} + LightGBM] {mean_lgbm:.4f} ± {std_lgbm:.4f}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(cfg.TABLES / "e022_combinaciones.csv", index=False)
    best_row = table.loc[table["macro_f1"].idxmax()]
    payload = {
        "combinaciones": rows,
        "mejor": best_row["combinacion"],
        "mejor_macro_f1": float(best_row["macro_f1"]),
        "referencia_e020": 0.5851,
        "aporte_ordinal": round(mean_ord - mean_ref, 4),
        "aporte_centroides": round(mean_both - mean_ord, 4),
        "aporte_meta_lightgbm": round(mean_lgbm - float(best_row["macro_f1"]) if
                                      best_row["meta"] == "LightGBM" else 0.0, 4),
        "segundos": round(time.time() - started, 1),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== E-022 — ENSAMBLE FINAL ===")
    print(table.to_string(index=False))
    print(f"\nMejor: {payload['mejor']} = {payload['mejor_macro_f1']:.4f} "
          f"(referencia E-020: 0,5851)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
