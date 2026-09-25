"""Fase 20 (v2) — Re-evaluación del ensamble con los modelos derivados del análisis de errores.

Pregunta
    ¿Añadir el canal de hashtags (E-014) y/o la señal ordinal (E-015) al stacking mejora la
    combinación de 4 modelos (0,5656)?

Protocolo
    - Todas las matrices son out-of-fold de las **mismas** particiones agrupadas.
    - Evaluación **leave-one-fold-out** del meta-modelo (nunca ve el fold que predice).
    - Meta-modelo: LogisticRegression sobre features estandarizadas.
    - Se comparan varias combinaciones y se elige la mejor por LOFO.

Salida
    ``experiments/fase20b_ensamble_extendido.json`` y ``reports/tables/fase20b_combinaciones.csv``.
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
from src import cv, data  # noqa: E402
from src.config import set_seed  # noqa: E402

#: Fuentes de features: archivo -> transformación aplicada.
SOURCES = {
    "svm": ("oof_E013_margenes.npy", "margenes"),
    "logreg": ("oof_E-009.npy", "log"),
    "sgd": ("oof_E-011.npy", "log"),
    "cnb": ("oof_E-010.npy", "log"),
    "hashtags": ("oof_E014_margenes.npy", "margenes"),
    "ordinal": ("oof_E015_ordinal.npy", "crudo"),
    "ordinal_soft": ("oof_E015_softmax.npy", "log"),
    "svm700k": ("oof_E016_margenes.npy", "margenes"),
    "word": ("oof_E017_probabilidades.npy", "log"),
}

COMBINATIONS = {
    "4 base (Fase 20)": ["svm", "logreg", "sgd", "cnb"],
    "4 base + hashtags": ["svm", "logreg", "sgd", "cnb", "hashtags"],
    "4 base + hashtags + ordinal": ["svm", "logreg", "sgd", "cnb", "hashtags", "ordinal",
                                    "ordinal_soft"],
    "5 base (+700k)": ["svm", "svm700k", "logreg", "sgd", "cnb"],
    "5 base + hashtags + ordinal": ["svm", "svm700k", "logreg", "sgd", "cnb", "hashtags",
                                    "ordinal", "ordinal_soft"],
    "5 base + hashtags + ordinal + word": ["svm", "svm700k", "logreg", "sgd", "cnb", "hashtags",
                                           "ordinal", "ordinal_soft", "word"],
    "svm700k + hashtags + ordinal + word": ["svm700k", "hashtags", "ordinal", "ordinal_soft",
                                            "word"],
}


def load_source(name: str) -> np.ndarray:
    """Carga y transforma la matriz OOF de una fuente."""
    filename, transform = SOURCES[name]
    matrix = np.load(cfg.FEATURES / filename)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if transform == "log":
        return np.log(np.clip(matrix, 1e-12, None))
    return matrix


def main() -> int:
    """Evalúa las combinaciones de ensamble con leave-one-fold-out."""
    set_seed()
    started = time.time()
    train = data.load_train()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]
    y = train[cfg.TARGET].astype(str).to_numpy()
    folds = cv.make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")

    cache = {name: load_source(name) for name in SOURCES}
    rows: list[dict] = []
    detail: dict = {}

    for label, names in COMBINATIONS.items():
        features = np.concatenate([cache[name] for name in names], axis=1)
        fold_scores = []
        for fold in folds:
            train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
            valid_idx = fold["val_idx"]
            meta = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                           random_state=cfg.SEED)),
            ])
            meta.fit(features[train_idx], y[train_idx])
            predictions = meta.predict(features[valid_idx])
            fold_scores.append(float(f1_score(y[valid_idx], predictions, average="macro",
                                              labels=cfg.CLASSES)))
        rows.append({
            "combinacion": label, "n_features": features.shape[1],
            "macro_f1_lofo": round(float(np.mean(fold_scores)), 4),
            "std": round(float(np.std(fold_scores)), 4),
            "per_fold": [round(value, 4) for value in fold_scores],
        })
        detail[label] = {"fuentes": names, "per_fold": fold_scores}
        print(f"[{label:32s}] macro-F1 LOFO {np.mean(fold_scores):.4f} "
              f"(features {features.shape[1]})", flush=True)

    table = pd.DataFrame(rows).sort_values("macro_f1_lofo", ascending=False)
    table.drop(columns=["per_fold"]).to_csv(cfg.TABLES / "fase20b_combinaciones.csv", index=False)
    best = table.iloc[0]
    summary = {
        "combinaciones": table.drop(columns=["per_fold"]).to_dict(orient="records"),
        "mejor_combinacion": best["combinacion"],
        "mejor_macro_f1_lofo": float(best["macro_f1_lofo"]),
        "referencia_fase20": 0.5656,
        "mejor_individual": 0.5628,
        "detalle": detail,
        "segundos": round(time.time() - started, 1),
    }
    (cfg.EXPERIMENTS / "fase20b_ensamble_extendido.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== ENSAMBLE EXTENDIDO (LOFO) ===")
    print(table.drop(columns=["per_fold"]).to_string(index=False))
    print(f"\nMejor: {best['combinacion']} = {best['macro_f1_lofo']:.4f} "
          f"(referencia Fase 20: 0,5656 · mejor individual: 0,5628)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
