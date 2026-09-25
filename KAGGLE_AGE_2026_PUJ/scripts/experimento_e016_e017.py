"""E-016 y E-017 — Dos apuestas de representación/diversidad antes de cerrar el ensamble.

E-016 — Vocabulario de caracteres más grande
    El vectorizador char_wb 3-6 llegó **exactamente** al tope de 300.000 features, señal de que
    el límite está truncando vocabulario útil. Se prueba con 700.000.

E-017 — Modelo de palabra como miembro diverso del ensamble
    Los word n-grams son peores en solitario (0,4971) pero ven el texto de otra forma: su tasa de
    desacuerdo con el modelo de caracteres puede alimentar el stacking (la diversidad es lo que
    justificó la ganancia de ComplementNB en la Fase 20).

Protocolo
    Mismas particiones agrupadas; se guardan las matrices OOF para re-evaluar el ensamble.

Salida
    ``experiments/e016_e017_resultados.json`` + matrices OOF en ``features/``.
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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, data, preprocessing  # noqa: E402
from src.config import set_seed  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "e016_e017_resultados.json"
CHAR_CONFIG = {"analyzer": "char_wb", "ngram_range": (3, 6), "min_df": 3,
               "sublinear_tf": True, "max_features": 700_000, "lowercase": True}
MENTION_CONFIG = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2,
                  "sublinear_tf": True, "lowercase": True}
WORD_CONFIG = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 3,
               "sublinear_tf": True, "max_features": 300_000, "lowercase": True}


def tokens_of(series: pd.Series, pattern: str) -> pd.Series:
    """Extrae tokens que cumplen un patrón."""
    return preprocessing.collapse_spaces(
        series.str.findall(pattern).apply(lambda values: " ".join(values).lower()))


def main() -> int:
    """Ejecuta E-016 (vocabulario grande) y E-017 (modelo de palabra OOF)."""
    set_seed()
    started = time.time()
    train = data.load_train()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]
    y = train[cfg.TARGET].astype(str).to_numpy()
    texts = train[cfg.TEXT_COL]
    folds = cv.make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}
    margins_e016 = np.zeros((len(train), len(cfg.CLASSES)), dtype=np.float32)
    probabilities_e017 = np.zeros((len(train), len(cfg.CLASSES)), dtype=np.float32)

    for fold in folds:
        train_idx, valid_idx = fold["train_idx"], fold["val_idx"]

        # --- E-016: char 3-6 con 700k features + canal de menciones ---
        char = TfidfVectorizer(**CHAR_CONFIG)
        mentions = TfidfVectorizer(**MENTION_CONFIG)
        x_train = hstack([char.fit_transform(texts.iloc[train_idx]),
                          mentions.fit_transform(tokens_of(texts.iloc[train_idx], r"@\w+"))]).tocsr()
        x_valid = hstack([char.transform(texts.iloc[valid_idx]),
                          mentions.transform(tokens_of(texts.iloc[valid_idx], r"@\w+"))]).tocsr()
        svm = LinearSVC(C=0.2, max_iter=3000)
        svm.fit(x_train, y[train_idx])
        predictions = svm.predict(x_valid)
        macro = f1_score(y[valid_idx], predictions, average="macro")
        order = [list(svm.classes_).index(label) for label in cfg.CLASSES]
        margins_e016[valid_idx] = svm.decision_function(x_valid)[:, order]
        results.setdefault("E-016", {"per_fold": []})
        results["E-016"]["per_fold"].append({
            "fold": fold["fold"], "macro_f1": macro,
            "accuracy": accuracy_score(y[valid_idx], predictions),
            "n_features": int(x_train.shape[1]),
        })
        print(f"[fold {fold['fold']}] E-016 (char 3-6, 700k) macro-F1 {macro:.4f} "
              f"| features {x_train.shape[1]:,}", flush=True)

        # --- E-017: modelo de palabra (OOF para el ensamble) ---
        word = TfidfVectorizer(**WORD_CONFIG)
        x_train_w = word.fit_transform(texts.iloc[train_idx])
        x_valid_w = word.transform(texts.iloc[valid_idx])
        logreg = OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=1000,
                                                        solver="liblinear",
                                                        random_state=cfg.SEED))
        logreg.fit(x_train_w, y[train_idx])
        predictions_w = logreg.predict(x_valid_w)
        macro_w = f1_score(y[valid_idx], predictions_w, average="macro")
        order_w = [list(logreg.classes_).index(label) for label in cfg.CLASSES]
        probabilities_e017[valid_idx] = logreg.predict_proba(x_valid_w)[:, order_w]
        results.setdefault("E-017", {"per_fold": []})
        results["E-017"]["per_fold"].append({
            "fold": fold["fold"], "macro_f1": macro_w,
            "accuracy": accuracy_score(y[valid_idx], predictions_w),
        })
        print(f"[fold {fold['fold']}] E-017 (word 1-2 LogReg)  macro-F1 {macro_w:.4f}", flush=True)

        RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    for key in ("E-016", "E-017"):
        frame = pd.DataFrame(results[key]["per_fold"])
        results[key]["macro_f1_mean"] = float(frame["macro_f1"].mean())
        results[key]["macro_f1_std"] = float(frame["macro_f1"].std())
        results[key]["accuracy_mean"] = float(frame["accuracy"].mean())
    results["segundos"] = round(time.time() - started, 1)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    np.save(cfg.FEATURES / "oof_E016_margenes.npy", margins_e016)
    np.save(cfg.FEATURES / "oof_E017_probabilidades.npy", probabilities_e017)

    print("\n=== E-016 / E-017 ===")
    for key, label in (("E-016", "char 3-6 con 700k features + menciones"),
                       ("E-017", "word 1-2 + LogReg (miembro diverso)")):
        print(f"{key} ({label}): macro-F1 {results[key]['macro_f1_mean']:.4f} "
              f"± {results[key]['macro_f1_std']:.4f}")
    print("\nReferencias: E-013 (char 3-6 300k + menciones, C=0.2) 0,5628 · ensamble 0,5682")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
