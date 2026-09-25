"""FASE 9 — Profundizar el combinador: tercer miembro y promedio múltiple.

Contexto
    Las dos únicas ganancias reales de la fase avanzada vinieron del combinador: el MLP como
    meta-modelo (+0,0061, D-020) y el promedio de dos combinadores diversos (+0,0027, D-022). Si el
    principio es la diversidad de sesgos, un **tercer combinador con sesgo distinto** (lineal con
    pérdida L2 vía RidgeClassifier, o árboles vía LightGBM) debería aportar.

Hipótesis
    H35  Un promedio de tres combinadores (MLP no lineal + LogReg logística + Ridge L2) supera al
         promedio de dos.
    H36  LightGBM como tercer combinador aporta más diversidad que Ridge (árboles vs lineal), aunque
         su desempeño individual sea similar.

Protocolo
    LOFO sobre las mismas particiones agrupadas; toda ponderación es fija (igual pesos), sin nada
    ajustado a los datos. Confirmación en el holdout congelado solo si la mejora es ≥ +0,002.

Salida: ``experiments/phase9_combinadores.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv  # noqa: E402
from src.config import set_seed  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from phase2_ponderacion import build_features  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "phase9_combinadores.json"
N_CLASSES = len(cfg.CLASSES)


def softmax(matrix: np.ndarray) -> np.ndarray:
    """Softmax por filas (para convertir decisiones en pseudo-probabilidades)."""
    shifted = matrix - matrix.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def make_combiner(kind: str):
    """Crea un combinador del tipo indicado (todos con escalado estándar)."""
    if kind == "logreg":
        return Pipeline([("s", StandardScaler()),
                         ("c", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                  random_state=cfg.SEED))])
    if kind == "ridge":
        return Pipeline([("s", StandardScaler()),
                         ("c", RidgeClassifier(alpha=1.0, random_state=cfg.SEED))])
    if kind == "sgd_huber":
        return Pipeline([("s", StandardScaler()),
                         ("c", SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=50,
                                             random_state=cfg.SEED))])
    raise ValueError(kind)


def probabilities_of(model, features: np.ndarray) -> np.ndarray:
    """Probabilidades (o pseudo-probabilidades) del combinador sobre las filas dadas."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(features)
        except AttributeError:
            pass
    return softmax(model.decision_function(features))


def main() -> int:
    """Evalúa promedios de dos y tres combinadores."""
    set_seed()
    started = time.time()
    features, y, _, _, _ = build_features()
    margins_word = np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")
    features = np.concatenate([features, margins_word], axis=1)
    print(f"features del meta: {features.shape}", flush=True)

    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    folds = cv.make_folds(y, groups=clusters["cluster"].to_numpy()[: len(y)],
                          n_splits=5, n_repeats=1, scheme="group")

    # Matrices OOF de cada combinador
    kinds = ["logreg", "ridge", "sgd_huber"]
    oof = {kind: np.zeros((len(y), N_CLASSES)) for kind in kinds}
    from phase2_ponderacion import EncodedMLP
    oof["mlp"] = np.zeros((len(y), N_CLASSES))
    per_fold: dict[str, list] = {kind: [] for kind in kinds + ["mlp"]}

    for fold in folds:
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        for kind in kinds:
            model = make_combiner(kind)
            model.fit(features[train_idx], y[train_idx])
            matrix = probabilities_of(model, features[valid_idx])
            oof[kind][valid_idx] = matrix
            per_fold[kind].append(float(f1_score(
                y[valid_idx], np.array(cfg.CLASSES)[np.argmax(matrix, axis=1)],
                average="macro", labels=cfg.CLASSES)))
        encoder = EncodedMLP()
        encoder.fit(features[train_idx], y[train_idx])
        matrix = encoder.model.predict_proba(encoder.scaler.transform(features[valid_idx]))
        oof["mlp"][valid_idx] = matrix
        per_fold["mlp"].append(float(f1_score(
            y[valid_idx], np.array(cfg.CLASSES)[np.argmax(matrix, axis=1)],
            average="macro", labels=cfg.CLASSES)))
        print(f"[fold {fold['fold']}] " + " | ".join(
            f"{kind} {per_fold[kind][-1]:.4f}" for kind in kinds + ["mlp"])
            + f" ({round(time.time() - started, 1)}s)", flush=True)

    def macro(matrix: np.ndarray) -> float:
        return float(f1_score(y, np.array(cfg.CLASSES)[np.argmax(matrix, axis=1)],
                              average="macro", labels=cfg.CLASSES))

    individual = {kind: round(macro(oof[kind]), 4) for kind in oof}
    combinations = {
        "MLP + LogReg": 0.5 * (oof["mlp"] + oof["logreg"]),
        "MLP + LogReg + Ridge": (oof["mlp"] + oof["logreg"] + oof["ridge"]) / 3,
        "MLP + LogReg + SGD": (oof["mlp"] + oof["logreg"] + oof["sgd_huber"]) / 3,
        "cuatro combinadores": (oof["mlp"] + oof["logreg"] + oof["ridge"] + oof["sgd_huber"]) / 4,
    }
    combination_scores = {label: round(macro(matrix), 4) for label, matrix in combinations.items()}

    print("\n=== FASE 9 — COMBINADORES ===")
    print("individuales:", individual)
    for label, score in combination_scores.items():
        print(f"  {label:24s}: {score:.4f}")

    payload = {"individuales": individual, "promedios": combination_scores,
               "per_fold": {k: [round(v, 4) for v in values] for k, values in per_fold.items()},
               "referencia_v12": {"cv": 0.6005, "holdout": 0.6077, "lb": 0.62182},
               "segundos": round(time.time() - started, 1)}
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(cfg.FEATURES / "oof_meta_ridge.npy", oof["ridge"])
    np.save(cfg.FEATURES / "oof_meta_sgd.npy", oof["sgd_huber"])
    best = max(combination_scores, key=combination_scores.get)
    print(f"\nmejor promedio: {best} = {combination_scores[best]:.4f} "
          f"(referencia v12: 0,6005)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
