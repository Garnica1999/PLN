"""Fase 20 — Ensamble: ¿aportan los modelos diversos sobre el mejor individual?

Pregunta
    ¿La combinación de modelos clásicos (SVM lineal, regresión logística, SGD y ComplementNB)
    supera al mejor modelo individual bajo validación honesta?

Hipótesis
    H15: existe diversidad de errores entre un modelo de márgenes (LinearSVC) y modelos
    probabilísticos (LogReg, SGD, ComplementNB), de modo que el voto suave o el stacking
    reducen el error respecto al mejor individual.

Protocolo
    - Todas las matrices out-of-fold provienen de las **mismas particiones** agrupadas.
    - El **stacking** se evalúa con leave-one-fold-out: el meta-modelo se entrena con el OOF de
      4 folds y se evalúa en el fold restante (nunca ve el fold evaluado).
    - Se reporta la tasa de desacuerdo entre modelos como medida de diversidad.

Salida
    ``experiments/fase20_ensamble.json`` y ``reports/tables/fase20_ensamble.csv``.
"""

from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, data  # noqa: E402
from src.config import set_seed  # noqa: E402

SOURCES = {
    "E-013_svm": {"archivo": "oof_E013_margenes.npy", "tipo": "margenes"},
    "E-009_logreg": {"archivo": "oof_E-009.npy", "tipo": "probabilidades"},
    "E-011_sgd": {"archivo": "oof_E-011.npy", "tipo": "probabilidades"},
    "E-010_cnb": {"archivo": "oof_E-010.npy", "tipo": "probabilidades"},
}


def softmax(matrix: np.ndarray) -> np.ndarray:
    """Convierte márgenes en pseudo-probabilidades estables."""
    shifted = matrix - matrix.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def macro_f1(predictions: np.ndarray, y: np.ndarray) -> float:
    """macro-F1 con las clases canónicas."""
    return float(f1_score(y, np.array(cfg.CLASSES)[predictions], average="macro",
                          labels=cfg.CLASSES))


def main() -> int:
    """Evalúa diversidad, voto suave y stacking sobre las matrices out-of-fold."""
    set_seed()
    started = time.time()
    train = data.load_train()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]
    y = train[cfg.TARGET].astype(str).to_numpy()
    folds = cv.make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")

    scores: dict[str, np.ndarray] = {}
    for name, spec in SOURCES.items():
        path = cfg.FEATURES / spec["archivo"]
        if not path.exists():
            print(f"[skip] {name}: falta {path.name}")
            continue
        matrix = np.load(path)
        scores[name] = matrix if spec["tipo"] == "margenes" else np.log(np.clip(matrix, 1e-12, None))

    names = list(scores)
    print("Modelos disponibles:", names)

    # --- 1. Desempeño individual y diversidad ---
    individual: dict[str, float] = {}
    predictions: dict[str, np.ndarray] = {}
    for name in names:
        predictions[name] = np.argmax(scores[name], axis=1)
        individual[name] = macro_f1(predictions[name], y)

    diversity_rows = []
    for left, right in combinations(names, 2):
        disagreement = float((predictions[left] != predictions[right]).mean())
        diversity_rows.append({"modelo_a": left, "modelo_b": right,
                               "desacuerdo": round(disagreement, 4)})
    diversity = pd.DataFrame(diversity_rows)

    # --- 2. Voto suave (promedio de log-probabilidades) ---
    stacked_scores = np.stack([scores[name] for name in names], axis=0)
    soft_vote_predictions = np.argmax(stacked_scores.mean(axis=0), axis=1)
    soft_vote_score = macro_f1(soft_vote_predictions, y)

    # --- 3. Stacking con leave-one-fold-out ---
    features = np.concatenate([scores[name] for name in names], axis=1)
    fold_scores: list[dict] = []
    for fold in folds:
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        meta = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                                  random_state=cfg.SEED)
        meta.fit(features[train_idx], y[train_idx])
        fold_predictions = meta.predict(features[valid_idx])
        fold_scores.append({
            "fold": fold["fold"],
            "stacking": f1_score(y[valid_idx], fold_predictions, average="macro",
                                 labels=cfg.CLASSES),
            "mejor_individual": f1_score(y[valid_idx],
                                         np.array(cfg.CLASSES)[predictions["E-013_svm"][valid_idx]],
                                         average="macro", labels=cfg.CLASSES),
        })
    stacking_frame = pd.DataFrame(fold_scores)
    stacking_score = float(stacking_frame["stacking"].mean())
    best_individual_foldwise = float(stacking_frame["mejor_individual"].mean())

    summary = {
        "individual": {name: round(value, 4) for name, value in individual.items()},
        "mejor_individual": max(individual, key=individual.get),
        "mejor_individual_macro_f1": round(max(individual.values()), 4),
        "voto_suave_macro_f1": round(soft_vote_score, 4),
        "stacking_macro_f1_lofo": round(stacking_score, 4),
        "mejor_individual_en_mismos_folds": round(best_individual_foldwise, 4),
        "folds_stacking_gana": f"{int((stacking_frame['stacking'] > stacking_frame['mejor_individual']).sum())}/5",
        "diversidad": diversity.to_dict(orient="records"),
        "segundos": round(time.time() - started, 1),
    }
    (cfg.EXPERIMENTS / "fase20_ensamble.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    diversity.to_csv(cfg.TABLES / "fase20_diversidad.csv", index=False)

    table = pd.DataFrame([
        {"modelo": "SVM C=0.2 (individual)", "macro_f1": round(individual.get("E-013_svm", float("nan")), 4)},
        {"modelo": "LogReg (individual)", "macro_f1": round(individual.get("E-009_logreg", float("nan")), 4)},
        {"modelo": "SGD (individual)", "macro_f1": round(individual.get("E-011_sgd", float("nan")), 4)},
        {"modelo": "ComplementNB (individual)", "macro_f1": round(individual.get("E-010_cnb", float("nan")), 4)},
        {"modelo": "Voto suave (4 modelos)", "macro_f1": round(soft_vote_score, 4)},
        {"modelo": "Stacking LOFO (LogReg meta)", "macro_f1": round(stacking_score, 4)},
    ])
    table.to_csv(cfg.TABLES / "fase20_ensamble.csv", index=False)
    print("\n=== FASE 20 — ENSAMBLE ===")
    print(table.to_string(index=False))
    print("\nDiversidad (tasa de desacuerdo):")
    print(diversity.to_string(index=False))
    print(f"\nFolds donde el stacking supera al mejor individual: {summary['folds_stacking_gana']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
