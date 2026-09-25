"""Fase 19 — Optimización de macro-F1 mediante ajuste de decisión por clase.

Pregunta
    Con clases balanceadas, ¿gana algo ajustar la decisión (pesos por clase sobre las
    probabilidades) en lugar de usar el argmax crudo?

Hipótesis
    H12 (gate A): el ajuste de prior/umbral por clase mejora macro-F1 cuando los errores son
    asimétricos entre clases (la literatura lo respalda: Li et al. 2010; Parambath et al. 2014;
    Narasimhan et al., ver D-008).

Protocolo (honesto, sin optimismo)
    - **Leave-one-fold-out (LOFO)**: para cada fold k se ajustan los pesos con las predicciones
      out-of-fold de los otros 4 folds y se evalúan en el fold k. La media LOFO es la
      estimación honesta de la ganancia.
    - Se reporta además la cota optimista (pesos ajustados y evaluados sobre todo el OOF).
    - El argmax crudo es la referencia.

Salida
    ``experiments/fase19_pesos_por_clase.json`` y ``reports/tables/fase19_macro_f1.csv``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.config import set_seed  # noqa: E402

GRID = np.round(np.arange(0.70, 1.41, 0.02), 3)
PASSES = 3
OOF_FILES = {"E-009": "LogisticRegression (OvR)", "E-011": "SGDClassifier (log_loss)"}


def macro_f1_weighted(probabilities: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """macro-F1 tras ponderar las probabilidades por clase y tomar argmax."""
    predictions = cfg.CLASSES and np.array(cfg.CLASSES)[np.argmax(probabilities * weights, axis=1)]
    return float(f1_score(y, predictions, average="macro"))


def tune_weights(probabilities: np.ndarray, y: np.ndarray, passes: int = PASSES,
                 grid: np.ndarray = GRID) -> tuple[np.ndarray, float]:
    """Ajusta pesos por clase por ascenso coordenado para maximizar macro-F1.

    Returns
    -------
    tuple
        ``(pesos, macro_f1_ajustado)``.
    """
    weights = np.ones(len(cfg.CLASSES))
    best_score = macro_f1_weighted(probabilities, y, weights)
    for _ in range(passes):
        improved = False
        for class_index in range(len(cfg.CLASSES)):
            current = weights[class_index]
            local_best, local_score = current, best_score
            for candidate in grid:
                weights[class_index] = candidate
                score = macro_f1_weighted(probabilities, y, weights)
                if score > local_score + 1e-6:
                    local_best, local_score = candidate, score
            weights[class_index] = local_best
            if local_score > best_score + 1e-6:
                best_score, improved = local_score, True
        if not improved:
            break
    return weights, best_score


def main() -> int:
    """Ejecuta el ajuste LOFO y guarda pesos y resultados."""
    set_seed()
    started = time.time()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    from src import cv as cv_module
    from src import data

    train = data.load_train()
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]
    y = train[cfg.TARGET].astype(str).to_numpy()
    folds = cv_module.make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")

    rows: list[dict] = []
    payload: dict = {"protocolo": "leave-one-fold-out sobre predicciones out-of-fold",
                     "grid": [float(GRID.min()), float(GRID.max())], "modelos": {}}

    for key, nombre in OOF_FILES.items():
        path = cfg.FEATURES / f"oof_{key}.npy"
        if not path.exists():
            print(f"[skip] {key}: no hay probabilidades OOF")
            continue
        probabilities = np.load(path)
        base_scores, tuned_scores, tuned_weights = [], [], []
        for fold in folds:
            valid_idx = fold["val_idx"]
            other_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
            base_scores.append(macro_f1_weighted(probabilities[valid_idx], y[valid_idx],
                                                 np.ones(len(cfg.CLASSES))))
            weights_fold, _ = tune_weights(probabilities[other_idx], y[other_idx])
            tuned_scores.append(macro_f1_weighted(probabilities[valid_idx], y[valid_idx],
                                                  weights_fold))
            tuned_weights.append(weights_fold.tolist())

        global_weights, global_score = tune_weights(probabilities, y)
        rows.append({
            "modelo": key, "descripcion": nombre,
            "macro_f1_argmax": round(float(np.mean(base_scores)), 4),
            "macro_f1_lofo": round(float(np.mean(tuned_scores)), 4),
            "ganancia_lofo": round(float(np.mean(tuned_scores) - np.mean(base_scores)), 4),
            "folds_con_mejora": f"{int(np.sum(np.array(tuned_scores) > np.array(base_scores)))}/5",
            "macro_f1_cota_optimista": round(global_score, 4),
            "pesos_globales": [round(value, 3) for value in global_weights.tolist()],
            "pesos_por_fold": tuned_weights,
        })
        payload["modelos"][key] = {
            "nombre": nombre, "pesos_globales": global_weights.tolist(),
            "macro_f1_argmax": float(np.mean(base_scores)),
            "macro_f1_lofo": float(np.mean(tuned_scores)),
            "macro_f1_cota_optimista": global_score,
        }
        print(f"[{key}] argmax {np.mean(base_scores):.4f} -> LOFO {np.mean(tuned_scores):.4f} "
              f"({np.mean(tuned_scores) - np.mean(base_scores):+.4f}) | "
              f"pesos {np.round(global_weights, 3).tolist()}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(cfg.TABLES / "fase19_macro_f1.csv", index=False)
    payload["segundos"] = round(time.time() - started, 1)
    (cfg.EXPERIMENTS / "fase19_pesos_por_clase.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== FASE 19 — AJUSTE DE DECISIÓN PARA MACRO-F1 ===")
    print(table.to_string(index=False))
    print("\nComparación con los modelos sin ajuste: LinearSVC C=0.3 = 0,5624 · "
          "LogReg = 0,5603 · E-008 = 0,5557")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
