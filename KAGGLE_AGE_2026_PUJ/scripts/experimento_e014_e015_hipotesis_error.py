"""E-014 y E-015 — Hipótesis derivadas del análisis de errores (Fase 17).

Contexto
    El error se concentra en las clases medias (18-24 F1=0,476; 25-34 F1=0,449) y la confusión
    es casi toda entre clases **adyacentes** (18-24<->25-34 15,3%; 35-49<->50-64 14,2%). Además, los
    tweets cortos concentran el error (52,5% en el quintil más corto vs 37,2% en el más largo).

E-014 — Canal de hashtags
    Hipótesis: los hashtags son marcadores temáticos muy distintivos por clase (EDA) y su
    identidad aporta señal adicional a la de menciones y caracteres.

E-015 — Regresión ordinal auxiliar
    Hipótesis (Nguyen et al. 2011; D-008): aprovechar el **orden** de los rangos de edad con una
    regresión Ridge sobre el punto medio del rango permite explotar la estructura adyacente que
    el clasificador nominal desaprovecha.

Protocolo
    Mismas 5 particiones agrupadas; la representación char+menciones se construye una vez por
    fold y se reutiliza. Se guardan las matrices OOF de ambos modelos para re-evaluar el ensamble.

Salida
    ``experiments/hipotesis_error_resultados.json`` y matrices OOF en ``features/``.
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
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, data, preprocessing  # noqa: E402
from src.config import set_seed  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "hipotesis_error_resultados.json"
CHAR_CONFIG = {"analyzer": "char_wb", "ngram_range": (3, 6), "min_df": 5,
               "sublinear_tf": True, "max_features": 300_000, "lowercase": True}
MENTION_CONFIG = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2,
                  "sublinear_tf": True, "lowercase": True}
HASHTAG_CONFIG = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2,
                  "sublinear_tf": True, "lowercase": True}

#: Punto medio de cada rango de edad (65-xx se representa con 70 por ser el extremo abierto).
CLASS_MIDPOINTS = {"13-17": 15.0, "18-24": 21.0, "25-34": 29.5, "35-49": 42.0,
                   "50-64": 57.0, "65-xx": 70.0}
#: Cortes entre puntos medios consecutivos para convertir la predicción continua en clase.
MIDPOINT_THRESHOLDS = [(15.0 + 21.0) / 2, (21.0 + 29.5) / 2, (29.5 + 42.0) / 2,
                       (42.0 + 57.0) / 2, (57.0 + 70.0) / 2]


def tokens_of(series: pd.Series, pattern: str) -> pd.Series:
    """Extrae tokens que cumplen un patrón y los une en una cadena."""
    return preprocessing.collapse_spaces(
        series.str.findall(pattern).apply(lambda values: " ".join(values).lower()))


def ordinal_to_class(values: np.ndarray) -> np.ndarray:
    """Convierte una predicción continua en clase usando los cortes entre puntos medios."""
    indices = np.searchsorted(MIDPOINT_THRESHOLDS, values)
    return np.array(cfg.CLASSES)[indices]


def main() -> int:
    """Ejecuta E-014 y E-015 y guarda sus matrices OOF."""
    set_seed()
    started = time.time()
    train = data.load_train()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]
    y = train[cfg.TARGET].astype(str).to_numpy()
    texts = train[cfg.TEXT_COL]
    ordinal_target = np.array([CLASS_MIDPOINTS[label] for label in y])
    folds = cv.make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}
    margins_e014 = np.zeros((len(train), len(cfg.CLASSES)), dtype=np.float32)
    ordinal_e015 = np.zeros(len(train), dtype=np.float32)
    probabilities_e015 = np.zeros((len(train), len(cfg.CLASSES)), dtype=np.float32)

    for fold in folds:
        train_idx, valid_idx = fold["train_idx"], fold["val_idx"]
        char = TfidfVectorizer(**CHAR_CONFIG)
        mentions = TfidfVectorizer(**MENTION_CONFIG)
        x_train = hstack([char.fit_transform(texts.iloc[train_idx]),
                          mentions.fit_transform(tokens_of(texts.iloc[train_idx], r"@\w+"))]).tocsr()
        x_valid = hstack([char.transform(texts.iloc[valid_idx]),
                          mentions.transform(tokens_of(texts.iloc[valid_idx], r"@\w+"))]).tocsr()

        # --- E-014: canal de hashtags ---
        hashtags = TfidfVectorizer(**HASHTAG_CONFIG)
        x_train_h = hstack([x_train,
                            hashtags.fit_transform(tokens_of(texts.iloc[train_idx], r"#\w+"))]).tocsr()
        x_valid_h = hstack([x_valid,
                            hashtags.transform(tokens_of(texts.iloc[valid_idx], r"#\w+"))]).tocsr()
        svm = LinearSVC(C=0.2, max_iter=3000)
        svm.fit(x_train_h, y[train_idx])
        predictions_h = svm.predict(x_valid_h)
        macro_h = f1_score(y[valid_idx], predictions_h, average="macro")
        order = [list(svm.classes_).index(label) for label in cfg.CLASSES]
        margins_e014[valid_idx] = svm.decision_function(x_valid_h)[:, order]
        results.setdefault("E-014", {"per_fold": []})
        results["E-014"]["per_fold"].append({
            "fold": fold["fold"], "macro_f1": macro_h,
            "accuracy": accuracy_score(y[valid_idx], predictions_h),
            "n_features": int(x_train_h.shape[1]),
        })
        print(f"[fold {fold['fold']}] E-014 (char+menciones+hashtags) macro-F1 {macro_h:.4f}",
              flush=True)

        # --- E-015: regresión ordinal auxiliar ---
        ridge = Ridge(alpha=1.0, solver="sparse_cg", max_iter=2000, random_state=cfg.SEED)
        ridge.fit(x_train, ordinal_target[train_idx])
        continuous = ridge.predict(x_valid)
        predictions_o = ordinal_to_class(continuous)
        macro_o = f1_score(y[valid_idx], predictions_o, average="macro")
        ordinal_e015[valid_idx] = continuous
        # Distribución "suave" por clase: softmax negativo de la distancia al punto medio.
        distances = np.abs(continuous[:, None] - np.array([CLASS_MIDPOINTS[c] for c in cfg.CLASSES]))
        soft = np.exp(-distances / 5.0)
        probabilities_e015[valid_idx] = soft / soft.sum(axis=1, keepdims=True)
        results.setdefault("E-015", {"per_fold": []})
        results["E-015"]["per_fold"].append({
            "fold": fold["fold"], "macro_f1": macro_o,
            "accuracy": accuracy_score(y[valid_idx], predictions_o),
        })
        print(f"[fold {fold['fold']}] E-015 (ordinal Ridge)          macro-F1 {macro_o:.4f}",
              flush=True)

        RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    for key in ("E-014", "E-015"):
        frame = pd.DataFrame(results[key]["per_fold"])
        results[key]["macro_f1_mean"] = float(frame["macro_f1"].mean())
        results[key]["macro_f1_std"] = float(frame["macro_f1"].std())
        results[key]["accuracy_mean"] = float(frame["accuracy"].mean())
    results["segundos"] = round(time.time() - started, 1)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    np.save(cfg.FEATURES / "oof_E014_margenes.npy", margins_e014)
    np.save(cfg.FEATURES / "oof_E015_ordinal.npy", ordinal_e015)
    np.save(cfg.FEATURES / "oof_E015_softmax.npy", probabilities_e015)
    np.save(cfg.FEATURES / "oof_labels.npy", y)

    print("\n=== HIPÓTESIS DERIVADAS DEL ANÁLISIS DE ERRORES ===")
    for key, label in (("E-014", "canal de hashtags"), ("E-015", "regresión ordinal")):
        print(f"{key} ({label}): macro-F1 {results[key]['macro_f1_mean']:.4f} "
              f"± {results[key]['macro_f1_std']:.4f} | accuracy {results[key]['accuracy_mean']:.4f}")
    print("\nReferencias: mejor individual 0,5628 (SVM C=0.2) · ensamble 0,5656 · E-008 0,5557")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
