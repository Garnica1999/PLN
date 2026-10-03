"""E-021 — Miembros ordinales y de centroides para el ensamble.

Contexto (Fase 17)
    La confusión del ensamble es casi toda entre **rangos contiguos** (18-24<->25-34 15,3%;
    35-49<->50-64 14,2%) y las clases extremas se clasifican bien. El clasificador nominal
    desaprovecha esa estructura.

Hipótesis
    H18  Un **modelo ordinal acumulativo** (Frank & Hall 2001): K-1 clasificadores binarios
         "¿la edad supera el umbral k?" cuyas probabilidades se combinan en una distribución
         sobre las clases, captura mejor la adyacencia que un clasificador nominal.
    H19  Un **modelo de centroides (Rocchio)** sobre TF-IDF es muy distinto de un SVM lineal;
         aunque sea menos exacto, puede aportar diversidad al ensamble (como el MLP en E-020).

Protocolo
    Mismas 5 particiones agrupadas; la representación char 3-6 ∪ menciones se construye una vez
    por fold y se reutiliza para ambos modelos. Se guardan las matrices OOF.

Salida
    ``experiments/e021_ordinal_centroides.json`` + matrices OOF en ``features/``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, data, preprocessing  # noqa: E402
from src.config import set_seed  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "e021_ordinal_centroides.json"
CHAR_CONFIG = {"analyzer": "char_wb", "ngram_range": (3, 6), "min_df": 5,
               "sublinear_tf": True, "max_features": 300_000, "lowercase": True}
MENTION_CONFIG = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2,
                  "sublinear_tf": True, "lowercase": True}


def tokens_of(series: pd.Series, pattern: str) -> pd.Series:
    """Extrae tokens que cumplen un patrón."""
    return preprocessing.collapse_spaces(
        series.str.findall(pattern).apply(lambda values: " ".join(values).lower()))


def ordinal_probabilities(x_train, y_index: np.ndarray, x_valid) -> np.ndarray:
    """Distribución de clase por modelo ordinal acumulativo (Frank & Hall 2001).

    Entrena K-1 clasificadores binarios ``P(y > k)`` y los combina:
    ``P(y = k) = P(y > k-1) - P(y > k)``, con ``P(y > 0) = 1`` y ``P(y > K) = 0``.
    """
    thresholds = np.arange(1, len(cfg.CLASSES))
    cumulative = np.ones((x_valid.shape[0], len(thresholds) + 1), dtype=np.float64)
    for position, threshold in enumerate(thresholds):
        binary_target = (y_index >= threshold).astype(int)
        model = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear",
                                   random_state=cfg.SEED)
        model.fit(x_train, binary_target)
        cumulative[:, position + 1] = model.predict_proba(x_valid)[:, 1]
    cumulative = np.clip(np.minimum.accumulate(cumulative, axis=1), 0.0, 1.0)
    # P(y = k) = P(y > k-1) - P(y > k); la última clase es P(y > K-1) - 0.
    probabilities = np.column_stack([-np.diff(cumulative, axis=1), cumulative[:, -1]])
    probabilities = np.clip(probabilities, 1e-12, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def rocchio_probabilities(x_train, y: np.ndarray, x_valid) -> np.ndarray:
    """Clasificador de centroides (Rocchio) con similitud coseno."""
    centroids = []
    for label in cfg.CLASSES:
        mask = y == label
        centroid = np.asarray(x_train[mask].mean(axis=0)).ravel()
        centroids.append(centroid / (np.linalg.norm(centroid) + 1e-12))
    centroid_matrix = np.vstack(centroids)

    valid_norms = np.sqrt(np.asarray(x_valid.multiply(x_valid).sum(axis=1)).ravel()) + 1e-12
    similarity = np.asarray(x_valid @ centroid_matrix.T) / valid_norms[:, None]
    shifted = similarity - similarity.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted * 8.0)  # temperatura: afina la distribución
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def main() -> int:
    """Ejecuta E-021 y guarda las matrices OOF de ambos modelos."""
    set_seed()
    started = time.time()
    train = data.load_train()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]
    y = train[cfg.TARGET].astype(str).to_numpy()
    y_index = np.array([cfg.CLASSES.index(label) for label in y])
    texts = train[cfg.TEXT_COL]
    folds = cv.make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}
    oof_ordinal = np.zeros((len(train), len(cfg.CLASSES)), dtype=np.float32)
    oof_rocchio = np.zeros((len(train), len(cfg.CLASSES)), dtype=np.float32)

    for fold in folds:
        train_idx, valid_idx = fold["train_idx"], fold["val_idx"]
        char = TfidfVectorizer(**CHAR_CONFIG)
        mentions = TfidfVectorizer(**MENTION_CONFIG)
        x_train = hstack([char.fit_transform(texts.iloc[train_idx]),
                          mentions.fit_transform(tokens_of(texts.iloc[train_idx], r"@\w+"))]).tocsr()
        x_valid = hstack([char.transform(texts.iloc[valid_idx]),
                          mentions.transform(tokens_of(texts.iloc[valid_idx], r"@\w+"))]).tocsr()

        probabilities = ordinal_probabilities(x_train, y_index[train_idx], x_valid)
        predictions = np.array(cfg.CLASSES)[np.argmax(probabilities, axis=1)]
        macro_ordinal = f1_score(y[valid_idx], predictions, average="macro")
        oof_ordinal[valid_idx] = probabilities

        rocchio = rocchio_probabilities(x_train, y[train_idx], x_valid)
        predictions_r = np.array(cfg.CLASSES)[np.argmax(rocchio, axis=1)]
        macro_rocchio = f1_score(y[valid_idx], predictions_r, average="macro")
        oof_rocchio[valid_idx] = rocchio

        for key, macro, extra in (("E-021a", macro_ordinal, "modelo ordinal acumulativo"),
                                  ("E-021b", macro_rocchio, "modelo de centroides (Rocchio)")):
            results.setdefault(key, {"per_fold": [], "descripcion": extra})
            results[key]["per_fold"].append({
                "fold": fold["fold"], "macro_f1": macro,
                "accuracy": accuracy_score(y[valid_idx],
                                           predictions if key == "E-021a" else predictions_r),
            })
        print(f"[fold {fold['fold']}] E-021a ordinal {macro_ordinal:.4f} | "
              f"E-021b centroides {macro_rocchio:.4f} ({round(time.time() - started, 1)}s)",
              flush=True)
        RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    for key in ("E-021a", "E-021b"):
        frame = pd.DataFrame(results[key]["per_fold"])
        results[key]["macro_f1_mean"] = float(frame["macro_f1"].mean())
        results[key]["macro_f1_std"] = float(frame["macro_f1"].std())
        results[key]["accuracy_mean"] = float(frame["accuracy"].mean())
    results["segundos"] = round(time.time() - started, 1)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    np.save(cfg.FEATURES / "oof_E021a_ordinal.npy", oof_ordinal)
    np.save(cfg.FEATURES / "oof_E021b_rocchio.npy", oof_rocchio)

    print("\n=== E-021 ===")
    for key, label in (("E-021a", "Ordinal acumulativo (5 binarios)"),
                       ("E-021b", "Centroides Rocchio")):
        print(f"{key} ({label}): macro-F1 {results[key]['macro_f1_mean']:.4f} "
              f"± {results[key]['macro_f1_std']:.4f} | accuracy {results[key]['accuracy_mean']:.4f}")
    print("\nReferencias: mejor individual 0,5658 · MLP denso (miembro diverso) 0,4071 · "
          "ensamble 0,5851")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
