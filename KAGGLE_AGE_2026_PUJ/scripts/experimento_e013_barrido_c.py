"""E-013 — Barrido de regularización de LinearSVC y generación de márgenes OOF.

Pregunta
    ¿Cuánta regularización conviene con 314k features dispersas y 169k documentos?

Hipótesis
    H14: con vocabulario de alta dimensión y clases solapadas, una regularización más fuerte
    (C menor) reduce varianza y mejora macro-F1, con un óptimo intermedio.

Protocolo
    La representación ganadora se construye **una vez por fold** y se reutiliza para todos los
    valores de C (ahorro de cómputo). Se guardan los márgenes ``decision_function``
    out-of-fold del mejor C para el ensamble de la Fase 20.

Salida
    ``experiments/barrido_c_resultados.json`` y ``features/oof_E013_margenes.npy``.
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, data, preprocessing  # noqa: E402
from src.config import set_seed  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "barrido_c_resultados.json"
CHAR_CONFIG = {"analyzer": "char_wb", "ngram_range": (3, 6), "min_df": 5,
               "sublinear_tf": True, "max_features": 300_000, "lowercase": True}
MENTION_CONFIG = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2,
                  "sublinear_tf": True, "lowercase": True}
C_VALUES = [0.1, 0.2, 0.3, 0.5, 1.0]


def mention_tokens(series: pd.Series) -> pd.Series:
    """Tokens de mención (canal dedicado)."""
    return preprocessing.collapse_spaces(
        series.str.findall(r"@\w+").apply(lambda values: " ".join(values).lower()))


def main() -> int:
    """Ejecuta el barrido de C y guarda los márgenes del mejor valor."""
    set_seed()
    started = time.time()
    train = data.load_train()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]
    y = train[cfg.TARGET].astype(str).to_numpy()
    texts = train[cfg.TEXT_COL]
    folds = cv.make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")

    margins = {value: np.zeros((len(train), len(cfg.CLASSES)), dtype=np.float32)
               for value in C_VALUES}
    per_fold: dict[float, list[dict]] = {value: [] for value in C_VALUES}

    for fold in folds:
        train_idx, valid_idx = fold["train_idx"], fold["val_idx"]
        char = TfidfVectorizer(**CHAR_CONFIG)
        mentions = TfidfVectorizer(**MENTION_CONFIG)
        x_train = hstack([char.fit_transform(texts.iloc[train_idx]),
                          mentions.fit_transform(mention_tokens(texts.iloc[train_idx]))]).tocsr()
        x_valid = hstack([char.transform(texts.iloc[valid_idx]),
                          mentions.transform(mention_tokens(texts.iloc[valid_idx]))]).tocsr()

        for value in C_VALUES:
            fit_started = time.time()
            model = LinearSVC(C=value, max_iter=3000)
            model.fit(x_train, y[train_idx])
            predictions = model.predict(x_valid)
            macro = f1_score(y[valid_idx], predictions, average="macro")
            per_fold[value].append({
                "fold": fold["fold"], "macro_f1": macro,
                "weighted_f1": f1_score(y[valid_idx], predictions, average="weighted"),
                "accuracy": accuracy_score(y[valid_idx], predictions),
                "seconds": round(time.time() - fit_started, 1),
            })
            decision = model.decision_function(x_valid)
            column_order = [list(model.classes_).index(label) for label in cfg.CLASSES]
            margins[value][valid_idx] = decision[:, column_order]
            print(f"[fold {fold['fold']}] C={value:<4} macro-F1 {macro:.4f} "
                  f"({round(time.time() - fit_started, 1)}s)", flush=True)

    summary: dict = {"valores_C": C_VALUES, "resultados": {}}
    for value in C_VALUES:
        frame = pd.DataFrame(per_fold[value])
        summary["resultados"][str(value)] = {
            "macro_f1_mean": float(frame["macro_f1"].mean()),
            "macro_f1_std": float(frame["macro_f1"].std()),
            "weighted_f1_mean": float(frame["weighted_f1"].mean()),
            "accuracy_mean": float(frame["accuracy"].mean()),
            "seconds_total": float(frame["seconds"].sum()),
            "per_fold": per_fold[value],
        }
    best_c = max(C_VALUES, key=lambda value: summary["resultados"][str(value)]["macro_f1_mean"])
    summary["mejor_C"] = best_c
    summary["segundos_totales"] = round(time.time() - started, 1)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    np.save(cfg.FEATURES / "oof_E013_margenes.npy", margins[best_c])
    np.save(cfg.FEATURES / "oof_E013_labels.npy", y)
    np.save(cfg.FEATURES / "oof_E013_bestC.npy", np.array([best_c]))

    table = pd.DataFrame([
        {"C": value, "macro_f1": round(summary["resultados"][str(value)]["macro_f1_mean"], 4),
         "std": round(summary["resultados"][str(value)]["macro_f1_std"], 4),
         "accuracy": round(summary["resultados"][str(value)]["accuracy_mean"], 4)}
        for value in C_VALUES
    ])
    table.to_csv(cfg.TABLES / "barrido_c.csv", index=False)
    print("\n=== BARRIDO DE C ===")
    print(table.to_string(index=False))
    print(f"mejor C = {best_c} | márgenes OOF guardados en features/oof_E013_margenes.npy")
    print(f"tiempo total: {summary['segundos_totales']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
