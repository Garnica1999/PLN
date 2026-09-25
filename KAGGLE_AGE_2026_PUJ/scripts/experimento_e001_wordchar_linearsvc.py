"""E-001 — Baseline fuerte: TF-IDF (word 1-2 + char_wb 3-5) + LinearSVC.

Pregunta
    ¿Qué desempeño alcanza una representación dispersa clásica (n-gramas de palabra y de
    carácter con TF-IDF sublinear) con un clasificador lineal, bajo el protocolo de
    validación agrupado por clústeres de near-duplicados?

Hipótesis
    H6 (Fase 3): los char n-grams aportan señal ortográfica complementaria a las palabras.
    El estado del arte (D-008) sugiere que esta combinación es la baseline fuerte del
    problema, incluso frente a embeddings (PAN 2019: 0,8972 vs 0,8444).

Protocolo
    - Validación: StratifiedGroupKFold (5 folds) agrupado por clúster de near-duplicados.
    - Vectorizadores ajustados **solo con el train de cada fold** (nunca con validación).
    - Métricas: macro-F1 (primaria), weighted-F1, micro-F1 y accuracy; media ± desviación.

Salidas
    ``experiments/E-001_resultados.json`` y una fila nueva en ``experiments.csv``.
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
from src import cv, data  # noqa: E402
from src.config import set_seed  # noqa: E402

EXPERIMENT_ID = "E-001"
HYPOTHESIS = ("TF-IDF de n-gramas de palabra (1-2) unido a n-gramas de carácter (char_wb 3-5) "
              "con LinearSVC captura señal léxica y ortográfica suficiente para superar "
              "ampliamente el azar bajo validación agrupada por clúster.")
CONFIG = {
    "word_tfidf": {"ngram_range": (1, 2), "min_df": 3, "sublinear_tf": True,
                   "max_features": 300_000, "lowercase": True},
    "char_tfidf": {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 5,
                   "sublinear_tf": True, "max_features": 300_000, "lowercase": True},
    "model": {"name": "LinearSVC", "C": 1.0, "max_iter": 3000},
    "cv": {"scheme": "group", "n_splits": 5, "n_repeats": 1, "seed": cfg.SEED},
}


def build_features(train_texts: pd.Series, valid_texts: pd.Series) -> tuple:
    """Ajusta los vectorizadores con train y transforma train y validación."""
    word = TfidfVectorizer(**CONFIG["word_tfidf"])
    char = TfidfVectorizer(**CONFIG["char_tfidf"])
    x_train = hstack([word.fit_transform(train_texts), char.fit_transform(train_texts)]).tocsr()
    x_valid = hstack([word.transform(valid_texts), char.transform(valid_texts)]).tocsr()
    return x_train, x_valid, x_train.shape[1]


def main() -> int:
    """Ejecuta el experimento completo y registra los resultados."""
    set_seed()
    started = time.time()
    train = data.load_train()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]
    y = train[cfg.TARGET].astype(str).to_numpy()
    texts = train[cfg.TEXT_COL]

    folds = cv.make_folds(y, groups=cluster_ids, n_splits=CONFIG["cv"]["n_splits"],
                          n_repeats=CONFIG["cv"]["n_repeats"], scheme=CONFIG["cv"]["scheme"])

    rows: list[dict] = []
    n_features = 0
    for fold in folds:
        fold_started = time.time()
        train_idx, valid_idx = fold["train_idx"], fold["val_idx"]
        x_train, x_valid, n_features = build_features(texts.iloc[train_idx], texts.iloc[valid_idx])
        model = LinearSVC(**{k: v for k, v in CONFIG["model"].items() if k != "name"})
        model.fit(x_train, y[train_idx])
        predictions = model.predict(x_valid)
        row = {
            "fold": fold["fold"],
            "macro_f1": f1_score(y[valid_idx], predictions, average="macro"),
            "weighted_f1": f1_score(y[valid_idx], predictions, average="weighted"),
            "micro_f1": f1_score(y[valid_idx], predictions, average="micro"),
            "accuracy": accuracy_score(y[valid_idx], predictions),
            "n_features": n_features,
            "seconds": round(time.time() - fold_started, 1),
        }
        rows.append(row)
        print(f"[fold {row['fold']}] macro_f1={row['macro_f1']:.4f} acc={row['accuracy']:.4f} "
              f"features={n_features:,} ({row['seconds']}s)", flush=True)

    results = pd.DataFrame(rows)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "macro_f1_mean": float(results["macro_f1"].mean()),
        "macro_f1_std": float(results["macro_f1"].std()),
        "weighted_f1_mean": float(results["weighted_f1"].mean()),
        "micro_f1_mean": float(results["micro_f1"].mean()),
        "accuracy_mean": float(results["accuracy"].mean()),
        "n_features": int(n_features),
        "seconds_total": round(time.time() - started, 1),
        "config": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
                   for k, v in CONFIG.items()},
        "per_fold": rows,
    }
    (cfg.EXPERIMENTS / f"{EXPERIMENT_ID}_resultados.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    register_experiment(summary, train_texts=texts)

    print("\n=== E-001 RESUMEN ===")
    print(f"macro-F1  : {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}")
    print(f"weighted  : {summary['weighted_f1_mean']:.4f}")
    print(f"accuracy  : {summary['accuracy_mean']:.4f}")
    print(f"features  : {summary['n_features']:,} | tiempo: {summary['seconds_total']}s")
    return 0


def register_experiment(summary: dict, train_texts: pd.Series) -> None:
    """Añade la fila del experimento a ``experiments.csv`` (registro obligatorio)."""
    registry_path = cfg.ROOT / "experiments.csv"
    row = {
        "experiment_id": summary["experiment_id"],
        "fecha": time.strftime("%Y-%m-%d"),
        "bloque": "2",
        "etapa": "B - baseline",
        "pregunta": "¿Qué desempeño da la unión TF-IDF word(1-2)+char_wb(3-5) con LinearSVC?",
        "hipotesis": HYPOTHESIS,
        "features": "texto crudo",
        "preprocessing": "ninguno (solo strip del padding)",
        "representacion": "TF-IDF sublinear word 1-2 + char_wb 3-5 (max_features 300k cada uno)",
        "modelo": "LinearSVC",
        "hiperparametros": json.dumps(CONFIG["model"], ensure_ascii=False),
        "validacion": "StratifiedGroupKFold 5 folds agrupado por clúster de near-duplicados",
        "macro_f1_mean": round(summary["macro_f1_mean"], 5),
        "macro_f1_std": round(summary["macro_f1_std"], 5),
        "weighted_f1_mean": round(summary["weighted_f1_mean"], 5),
        "micro_f1_mean": round(summary["micro_f1_mean"], 5),
        "accuracy_mean": round(summary["accuracy_mean"], 5),
        "resultado": f"macro-F1 {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}",
        "interpretacion": "pendiente de análisis comparativo con las variantes E-002..E-007",
        "decision": "conservar como baseline de referencia de la Etapa B",
        "duracion_seg": summary["seconds_total"],
        "artefactos": f"experiments/{summary['experiment_id']}_resultados.json",
    }
    header = ("experiment_id,fecha,bloque,etapa,pregunta,hipotesis,features,preprocessing,"
              "representacion,modelo,hiperparametros,validacion,macro_f1_mean,macro_f1_std,"
              "weighted_f1_mean,micro_f1_mean,accuracy_mean,resultado,interpretacion,decision,"
              "duracion_seg,artefactos")
    if not registry_path.exists():
        registry_path.write_text(header + "\n", encoding="utf-8")
    existing = pd.read_csv(registry_path)
    if summary["experiment_id"] in set(existing["experiment_id"].astype(str)):
        print(f"[registro] {summary['experiment_id']} ya estaba en experiments.csv (no se duplica)")
        return
    # Escritura vía pandas para que el escapado de comas y comillas sea correcto.
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(registry_path, index=False, columns=header.split(","))
    print(f"[registro] {summary['experiment_id']} añadido a experiments.csv")


if __name__ == "__main__":
    raise SystemExit(main())
