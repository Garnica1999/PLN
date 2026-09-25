"""E-009..E-011 — Modelos clásicos sobre la representación ganadora (Fase 13).

Pregunta
    Sobre la representación adoptada (char_wb 3-6 ∪ canal de menciones), ¿qué modelos clásicos
    compiten con el LinearSVC y cuáles aportan **probabilidades** (necesarias para el ajuste de
    decisión de macro-F1 de la Fase 19)?

Hipótesis
    H13a  Un modelo probabilístico (LogisticRegression) iguala o supera al SVM y habilita el
          ajuste de umbral/prior por clase.
    H13b  ComplementNB, aunque rápido, rinde por debajo por su supuesto de independencia.
    H13c  SGD con pérdida logística se acerca a LogReg con mucho menor costo.

Protocolo
    - Mismas 5 particiones agrupadas por clúster; la representación se construye **una vez por
      fold** y se reutiliza para los tres modelos (ahorro de cómputo).
    - Se guardan las **probabilidades out-of-fold** del mejor modelo probabilístico en
      ``features/oof_<modelo>.npy`` para la Fase 19.

Salida
    ``experiments/modelos_clasicos_resultados.json``, filas en ``experiments.csv`` y matrices OOF.
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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, data, preprocessing  # noqa: E402
from src.config import set_seed  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "modelos_clasicos_resultados.json"
CHAR_CONFIG = {"analyzer": "char_wb", "ngram_range": (3, 6), "min_df": 5,
               "sublinear_tf": True, "max_features": 300_000, "lowercase": True}
MENTION_CONFIG = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2,
                  "sublinear_tf": True, "lowercase": True}

MODELS = {
    "E-009": {
        "nombre": "LogisticRegression (OvR, liblinear, C=4)",
        "factory": lambda: OneVsRestClassifier(
            LogisticRegression(C=4.0, max_iter=1000, solver="liblinear", random_state=cfg.SEED)),
        "probabilistico": True,
        "hipotesis": "H13a: el modelo probabilístico iguala al SVM y habilita el ajuste de decisión.",
    },
    "E-010": {
        "nombre": "ComplementNB (alpha=0.3)",
        "factory": lambda: ComplementNB(alpha=0.3),
        "probabilistico": True,
        "hipotesis": "H13b: Naive Bayes rinde por debajo por su supuesto de independencia.",
    },
    "E-011": {
        "nombre": "SGDClassifier (log_loss, alpha=1e-6)",
        "factory": lambda: SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=30,
                                         tol=1e-4, random_state=cfg.SEED, n_jobs=32),
        "probabilistico": True,
        "hipotesis": "H13c: SGD logístico se acerca a LogReg con mucho menor costo.",
    },
    "E-012": {
        "nombre": "LinearSVC (C=0.3)",
        "factory": lambda: LinearSVC(C=0.3, max_iter=3000),
        "probabilistico": False,
        "hipotesis": "Regularización más fuerte: menos varianza con 314k features dispersas.",
    },
}


def mention_tokens(series: pd.Series) -> pd.Series:
    """Extrae los tokens de mención (canal dedicado de la representación ganadora)."""
    return preprocessing.collapse_spaces(
        series.str.findall(r"@\w+").apply(lambda values: " ".join(values).lower()))


def build_representation(train_texts: pd.Series, valid_texts: pd.Series):
    """Construye la representación ganadora ajustando los vectorizadores solo con train."""
    char = TfidfVectorizer(**CHAR_CONFIG)
    mentions = TfidfVectorizer(**MENTION_CONFIG)
    x_train = hstack([char.fit_transform(train_texts),
                      mentions.fit_transform(mention_tokens(train_texts))]).tocsr()
    x_valid = hstack([char.transform(valid_texts),
                      mentions.transform(mention_tokens(valid_texts))]).tocsr()
    return x_train, x_valid


def main() -> int:
    """Ejecuta los modelos clásicos sobre la representación ganadora."""
    set_seed()
    started = time.time()
    train = data.load_train()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]
    y = train[cfg.TARGET].astype(str).to_numpy()
    labels_sorted = np.array(cfg.CLASSES)
    texts = train[cfg.TEXT_COL]
    folds = cv.make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}
    oof_probabilities: dict[str, np.ndarray] = {
        key: np.zeros((len(train), len(cfg.CLASSES)), dtype=np.float32)
        for key, spec in MODELS.items() if spec["probabilistico"]
    }

    for fold in folds:
        fold_started = time.time()
        train_idx, valid_idx = fold["train_idx"], fold["val_idx"]
        x_train, x_valid = build_representation(texts.iloc[train_idx], texts.iloc[valid_idx])
        print(f"[fold {fold['fold']}] representación {x_train.shape} "
              f"({round(time.time() - fold_started, 1)}s)", flush=True)

        for key, spec in MODELS.items():
            if key in results and results[key].get("completo"):
                continue
            model = spec["factory"]()
            fit_started = time.time()
            model.fit(x_train, y[train_idx])
            predictions = model.predict(x_valid)
            macro = f1_score(y[valid_idx], predictions, average="macro")
            results.setdefault(key, {
                "id": key, "nombre": spec["nombre"], "hipotesis": spec["hipotesis"],
                "per_fold": [], "per_class": [],
            })
            results[key]["per_fold"].append({
                "fold": fold["fold"], "macro_f1": macro,
                "weighted_f1": f1_score(y[valid_idx], predictions, average="weighted"),
                "accuracy": accuracy_score(y[valid_idx], predictions),
                "seconds": round(time.time() - fit_started, 1),
            })
            results[key]["per_class"].append(dict(zip(
                cfg.CLASSES, f1_score(y[valid_idx], predictions, average=None,
                                      labels=cfg.CLASSES), strict=True)))
            if spec["probabilistico"]:
                probabilities = model.predict_proba(x_valid)
                # Reordena las columnas al orden canónico de clases del proyecto.
                column_order = [list(model.classes_).index(label) for label in cfg.CLASSES]
                oof_probabilities[key][valid_idx] = probabilities[:, column_order]
            print(f"          {key} {spec['nombre'][:38]:38s} macro-F1 {macro:.4f}", flush=True)

        RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                encoding="utf-8")

    # Consolidación de métricas
    for key, payload in results.items():
        frame = pd.DataFrame(payload["per_fold"])
        payload["macro_f1_mean"] = float(frame["macro_f1"].mean())
        payload["macro_f1_std"] = float(frame["macro_f1"].std())
        payload["weighted_f1_mean"] = float(frame["weighted_f1"].mean())
        payload["accuracy_mean"] = float(frame["accuracy"].mean())
        payload["seconds_total"] = float(frame["seconds"].sum())
        payload["per_class_f1"] = {label: float(np.mean([row[label] for row in payload["per_class"]]))
                                   for label in cfg.CLASSES}
        payload["completo"] = True
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Persistencia de las probabilidades OOF para la Fase 19
    for key, matrix in oof_probabilities.items():
        if matrix.sum() > 0:
            path = cfg.FEATURES / f"oof_{key}.npy"
            np.save(path, matrix)
            np.save(cfg.FEATURES / f"oof_{key}_labels.npy", y)
            print(f"[oof] {path} ({matrix.shape})", flush=True)

    register(results)
    table = pd.DataFrame([
        {"id": key, "modelo": value["nombre"], "macro_f1": round(value["macro_f1_mean"], 4),
         "std": round(value["macro_f1_std"], 4), "accuracy": round(value["accuracy_mean"], 4),
         "segundos": round(value["seconds_total"], 1)}
        for key, value in results.items()
    ]).sort_values("macro_f1", ascending=False)
    table.to_csv(cfg.TABLES / "modelos_clasicos_comparativa.csv", index=False)
    print("\n=== MODELOS CLÁSICOS ===")
    print(table.to_string(index=False))
    print(f"tiempo total: {round(time.time() - started, 1)}s")
    return 0


def register(results: dict) -> None:
    """Añade cada modelo evaluado a ``experiments.csv``."""
    registry_path = cfg.ROOT / "experiments.csv"
    existing = pd.read_csv(registry_path)
    for key, payload in results.items():
        if key in set(existing["experiment_id"].astype(str)):
            continue
        row = {
            "experiment_id": key, "fecha": time.strftime("%Y-%m-%d"), "bloque": "3",
            "etapa": "D - modelos clasicos",
            "pregunta": f"¿Cómo rinde {payload['nombre']} sobre la representación ganadora?",
            "hipotesis": payload["hipotesis"],
            "features": "char_wb 3-6 + canal de menciones", "preprocessing": "crudo",
            "representacion": "TF-IDF char_wb 3-6 ∪ menciones", "modelo": payload["nombre"],
            "hiperparametros": "{}",
            "validacion": "StratifiedGroupKFold 5 folds agrupado por clúster",
            "macro_f1_mean": round(payload["macro_f1_mean"], 5),
            "macro_f1_std": round(payload["macro_f1_std"], 5),
            "weighted_f1_mean": round(payload["weighted_f1_mean"], 5),
            "micro_f1_mean": "", "accuracy_mean": round(payload["accuracy_mean"], 5),
            "resultado": f"macro-F1 {payload['macro_f1_mean']:.4f} ± {payload['macro_f1_std']:.4f}",
            "interpretacion": "pendiente del análisis comparativo de modelos",
            "decision": "pendiente", "duracion_seg": round(payload["seconds_total"], 1),
            "artefactos": "experiments/modelos_clasicos_resultados.json",
        }
        existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    existing.to_csv(registry_path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
