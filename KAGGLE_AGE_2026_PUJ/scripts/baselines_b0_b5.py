"""Fase 12 — Baselines B0, B1 y B5 (los B2, B3 y B4 ya están medidos en la matriz).

B0  DummyClassifier(strategy="prior")                -> suelo estadístico.
B1  TF-IDF word 1-2 + LogisticRegression             -> ¿un modelo probabilístico mejora al SVM?
B2  TF-IDF word 1-2 + LinearSVC                      -> ya medido (E-002b = 0,4971).
B3  TF-IDF char_wb 3-5 + LinearSVC                   -> ya medido (E-003b = 0,5408).
B4  TF-IDF word 1-2 ∪ char_wb 3-5 + LinearSVC        -> ya medido (E-001 = 0,5344).
B5  Features densas (textometría + estilo) + LightGBM -> ¿la señal densa compite con la dispersa?

Protocolo idéntico a la matriz: StratifiedGroupKFold 5 folds agrupado por clúster.

Salida: ``experiments/baselines_resultados.json`` y filas en ``experiments.csv``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, data, textometry  # noqa: E402
from src.config import set_seed  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "baselines_resultados.json"
DENSE_FEATURES = [name for name in textometry.TEXTOMETRY_FEATURES + textometry.STYLE_FEATURES
                  if name not in {"n_emoji", "emoji_diversity", "has_emoji"}]


def evaluate(name: str, fit_predict, y: np.ndarray, folds: list[dict]) -> dict:
    """Evalúa un baseline sobre los folds agrupados."""
    rows: list[dict] = []
    per_class: list[dict] = []
    for fold in folds:
        started = time.time()
        train_idx, valid_idx = fold["train_idx"], fold["val_idx"]
        predictions = fit_predict(train_idx, valid_idx)
        rows.append({
            "fold": fold["fold"],
            "macro_f1": f1_score(y[valid_idx], predictions, average="macro"),
            "weighted_f1": f1_score(y[valid_idx], predictions, average="weighted"),
            "micro_f1": f1_score(y[valid_idx], predictions, average="micro"),
            "accuracy": accuracy_score(y[valid_idx], predictions),
            "seconds": round(time.time() - started, 1),
        })
        per_class.append(dict(zip(cfg.CLASSES,
                                  f1_score(y[valid_idx], predictions, average=None,
                                           labels=cfg.CLASSES), strict=True)))
    frame = pd.DataFrame(rows)
    return {
        "id": name,
        "macro_f1_mean": float(frame["macro_f1"].mean()),
        "macro_f1_std": float(frame["macro_f1"].std()),
        "weighted_f1_mean": float(frame["weighted_f1"].mean()),
        "micro_f1_mean": float(frame["micro_f1"].mean()),
        "accuracy_mean": float(frame["accuracy"].mean()),
        "seconds_total": round(float(frame["seconds"].sum()), 1),
        "per_class_f1": {label: float(np.mean([row[label] for row in per_class]))
                         for label in cfg.CLASSES},
        "per_fold": rows,
    }


def main() -> int:
    """Ejecuta B0, B1 y B5 y registra los resultados."""
    set_seed()
    train = data.load_train()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]
    y = train[cfg.TARGET].astype(str).to_numpy()
    texts = train[cfg.TEXT_COL]
    dense = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")[DENSE_FEATURES].to_numpy(float)
    folds = cv.make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}

    if "B0" not in results:
        print("[run ] B0 — DummyClassifier(prior)", flush=True)
        def b0_fit_predict(train_idx, valid_idx):
            model = DummyClassifier(strategy="prior", random_state=cfg.SEED)
            model.fit(dense[train_idx], y[train_idx])
            return model.predict(dense[valid_idx])
        results["B0"] = evaluate("B0", b0_fit_predict, y, folds)
        print(f"        macro-F1 {results['B0']['macro_f1_mean']:.4f}", flush=True)

    if "B1" not in results:
        print("[run ] B1 — TF-IDF word 1-2 + LogisticRegression (One-vs-Rest)", flush=True)
        from sklearn.multiclass import OneVsRestClassifier
        def b1_fit_predict(train_idx, valid_idx):
            # En scikit-learn 1.8 el solver 'liblinear' no soporta multiclase directo:
            # se usa el esquema One-vs-Rest explícito (equivalente y más rápido en disperso).
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                                          max_features=300_000, lowercase=True)),
                ("clf", OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=1000,
                                                               solver="liblinear",
                                                               random_state=cfg.SEED))),
            ])
            pipeline.fit(texts.iloc[train_idx], y[train_idx])
            return pipeline.predict(texts.iloc[valid_idx])
        results["B1"] = evaluate("B1", b1_fit_predict, y, folds)
        print(f"        macro-F1 {results['B1']['macro_f1_mean']:.4f}", flush=True)

    if "B5" not in results:
        print("[run ] B5 — features densas + LightGBM", flush=True)
        import lightgbm as lgb
        def b5_fit_predict(train_idx, valid_idx):
            model = lgb.LGBMClassifier(
                objective="multiclass", num_class=len(cfg.CLASSES), n_estimators=400,
                learning_rate=0.08, num_leaves=63, min_child_samples=40, subsample=0.9,
                colsample_bytree=0.9, reg_lambda=1.0, n_jobs=32, random_state=cfg.SEED,
                verbose=-1)
            model.fit(dense[train_idx], y[train_idx])
            return model.predict(dense[valid_idx])
        results["B5"] = evaluate("B5", b5_fit_predict, y, folds)
        print(f"        macro-F1 {results['B5']['macro_f1_mean']:.4f}", flush=True)

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    registry_path = cfg.ROOT / "experiments.csv"
    existing = pd.read_csv(registry_path)
    questions = {
        "B0": ("¿Cuál es el suelo estadístico (clase mayoritaria/prior)?",
               "Un clasificador trivial ronda 1/6 de macro-F1 con clases balanceadas.", "dense+prior"),
        "B1": ("¿Un modelo probabilístico lineal supera al SVM sobre word n-grams?",
               "LogisticRegression ofrece probabilidades calibradas; podría igualar o superar a "
               "LinearSVC en word n-grams.", "TF-IDF word 1-2 + LogisticRegression"),
        "B5": ("¿La señal densa (textometría+estilo) compite con la dispersa?",
               "La textometría tiene ε² alto, pero es un espacio de 35 dimensiones: debería "
               "quedar por debajo de char n-grams.", "35 features densas + LightGBM"),
    }
    for key, summary in results.items():
        if key in set(existing["experiment_id"].astype(str)):
            continue
        pregunta, hipotesis, representacion = questions[key]
        row = {
            "experiment_id": key, "fecha": time.strftime("%Y-%m-%d"), "bloque": "2",
            "etapa": "B - baseline", "pregunta": pregunta, "hipotesis": hipotesis,
            "features": representacion, "preprocessing": "crudo" if key != "B5" else "densas",
            "representacion": representacion, "modelo": key,
            "hiperparametros": "{}", "validacion": "StratifiedGroupKFold 5 folds agrupado",
            "macro_f1_mean": round(summary["macro_f1_mean"], 5),
            "macro_f1_std": round(summary["macro_f1_std"], 5),
            "weighted_f1_mean": round(summary["weighted_f1_mean"], 5),
            "micro_f1_mean": round(summary["micro_f1_mean"], 5),
            "accuracy_mean": round(summary["accuracy_mean"], 5),
            "resultado": f"macro-F1 {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}",
            "interpretacion": "pendiente de la tabla comparativa de baselines",
            "decision": "pendiente", "duracion_seg": summary["seconds_total"],
            "artefactos": "experiments/baselines_resultados.json",
        }
        existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        existing.to_csv(registry_path, index=False)
        print(f"[registro] {key} añadido a experiments.csv", flush=True)

    print("\n=== BASELINES ===")
    for key, summary in sorted(results.items()):
        print(f"{key}: macro-F1 {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f} "
              f"| accuracy {summary['accuracy_mean']:.4f} | {summary['seconds_total']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
