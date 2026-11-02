"""Entrena y evalua el clasificador de dominio dentro de la API.

Usa el mismo embedder bge-m3 y el mismo `search_hits` que la compuerta, para
que las features de entrenamiento y de inferencia sean identicas.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.rag.config import settings
from src.rag.domain import question_type, retrieval_features
from src.rag.embeddings import get_embedder
from src.rag.pipeline import search_hits

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

DATA = Path("eval/domain/domain_questions.jsonl")
MODEL_DIR = Path("modelos")
TABLE_DIR = Path("reports/tablas")
COLLECTION = settings.collection
EMBEDDING_MODEL = settings.embeddings_model


def build_features(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    print("calculando embeddings con el embedder de la API...", flush=True)
    embeddings = get_embedder().embed_queries([row["question"] for row in rows])
    print("extrayendo features de retrieval...", flush=True)
    retrieval = []
    types = []
    for index, row in enumerate(rows, start=1):
        hits, _ = search_hits(row["question"], k=5, collection=COLLECTION)
        retrieval.append(retrieval_features(row["question"], hits))
        types.append(question_type(row["question"]))
        if index % 20 == 0:
            print(f"  {index}/{len(rows)}", flush=True)
    features = np.hstack([embeddings, np.asarray(retrieval), np.asarray(types)])
    names = [f"emb_{i}" for i in range(embeddings.shape[1])]
    names += ["top_score", "mean_top5", "score_gap", "title_overlap", "entity_in_title"]
    names += ["type_capital", "type_poblacion", "type_autor", "type_fecha", "type_lugar", "type_cantidad", "type_otro"]
    return features, names


def train_logistic(x_train, y_train, x_dev, y_dev):
    best = None
    best_f1 = -1.0
    for c_value in (0.001, 0.01, 0.1, 1.0, 10.0):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=c_value, class_weight="balanced", max_iter=5000, random_state=42)),
            ]
        )
        model.fit(x_train, y_train)
        score = f1_score(y_dev, model.predict(x_dev), pos_label=1)
        print(f"  logistica C={c_value}: dev F1={score:.3f}", flush=True)
        if score > best_f1:
            best_f1 = score
            best = model
    return best, best_f1


def train_lightgbm(x_train, y_train, x_dev, y_dev):
    if LGBMClassifier is None:
        print("  lightgbm no disponible; se omite", flush=True)
        return None, -1.0
    best = None
    best_f1 = -1.0
    for leaves in (3, 7, 15):
        model = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=leaves,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        model.fit(x_train, y_train)
        score = f1_score(y_dev, model.predict(x_dev), pos_label=1)
        print(f"  lightgbm leaves={leaves}: dev F1={score:.3f}", flush=True)
        if score > best_f1:
            best_f1 = score
            best = model
    return best, best_f1


def score_model(model, x, y) -> dict:
    predictions = model.predict(x)
    probabilities = model.predict_proba(x)[:, 1]
    return {
        "precision": precision_score(y, predictions, pos_label=1, zero_division=0),
        "recall": recall_score(y, predictions, pos_label=1, zero_division=0),
        "f1": f1_score(y, predictions, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y, predictions),
        "false_rejection": float(np.mean(predictions[y == 1] == 0)) if np.any(y == 1) else 0.0,
        "false_acceptance": float(np.mean(predictions[y == 0] == 1)) if np.any(y == 0) else 0.0,
        "confusion": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        "predictions": predictions,
        "probabilities": probabilities,
    }


def main() -> None:
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"{len(rows)} preguntas", flush=True)
    features, names = build_features(rows)
    labels = np.asarray([1 if row["label"] == "dominio" else 0 for row in rows])
    splits = np.asarray([row["split"] for row in rows])

    x_train, y_train = features[splits == "train"], labels[splits == "train"]
    x_dev, y_dev = features[splits == "dev"], labels[splits == "dev"]
    x_test, y_test = features[splits == "test"], labels[splits == "test"]
    print(f"train {len(y_train)} | dev {len(y_dev)} | test {len(y_test)}", flush=True)

    print("regresion logistica", flush=True)
    logistic, logistic_f1 = train_logistic(x_train, y_train, x_dev, y_dev)
    print("lightgbm", flush=True)
    lightgbm, lightgbm_f1 = train_lightgbm(x_train, y_train, x_dev, y_dev)

    if lightgbm is not None and lightgbm_f1 > logistic_f1:
        name, model = "lightgbm", lightgbm
    else:
        name, model = "logistic", logistic
    print(f"elegido {name} con dev F1={max(logistic_f1, lightgbm_f1):.3f}", flush=True)

    metrics = []
    for split, x_split, y_split in (("dev", x_dev, y_dev), ("test", x_test, y_test)):
        result = score_model(model, x_split, y_split)
        metrics.append(
            {
                "model": name,
                "split": split,
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "accuracy": result["accuracy"],
                "false_rejection": result["false_rejection"],
                "false_acceptance": result["false_acceptance"],
                "confusion": json.dumps(result["confusion"]),
            }
        )
        print(
            f"{split}: P={result['precision']:.3f} R={result['recall']:.3f} "
            f"F1={result['f1']:.3f} falsos rechazos={result['false_rejection']:.3f} "
            f"falsos aceptados={result['false_acceptance']:.3f}",
            flush=True,
        )

    dev_probabilities = model.predict_proba(x_dev)[:, 1]
    neg_max = float(dev_probabilities[y_dev == 0].max())
    pos_min = float(dev_probabilities[y_dev == 1].min())
    suggested = round((neg_max + pos_min) / 2, 2)
    print(
        f"dev margen: max fuera={neg_max:.3f} min dominio={pos_min:.3f} umbral sugerido={suggested:.2f}",
        flush=True,
    )
    test_probabilities = model.predict_proba(x_test)[:, 1]
    for threshold in (0.40, 0.45, 0.50):
        dev_pred = dev_probabilities >= threshold
        test_pred = test_probabilities >= threshold
        print(
            f"umbral {threshold:.2f}: dev F1={f1_score(y_dev, dev_pred, pos_label=1):.3f} | "
            f"test F1={f1_score(y_test, test_pred, pos_label=1):.3f} "
            f"falsos rechazos={float(np.mean(test_pred[y_test == 1] == 0)):.3f} "
            f"falsos aceptados={float(np.mean(test_pred[y_test == 0] == 1)):.3f}",
            flush=True,
        )

    test_result = score_model(model, x_test, y_test)
    test_indices = list(np.where(splits == "test")[0])
    prediction_rows = []
    for index in test_indices:
        row = rows[index]
        position = test_indices.index(index)
        prediction_rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "label": row["label"],
                "subtype": row["subtype"],
                "group": row["group"],
                "prediction": "dominio" if test_result["predictions"][position] else "fuera",
                "prob_domain": round(float(test_result["probabilities"][position]), 4),
            }
        )

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(TABLE_DIR / "domain_classifier_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(TABLE_DIR / "domain_classifier_predictions.csv", index=False)
    joblib.dump(
        {"model": model, "feature_names": names, "embedding_model": EMBEDDING_MODEL},
        MODEL_DIR / "domain_classifier.joblib",
    )
    if name == "logistic":
        scaler = model.named_steps["scaler"]
        clf = model.named_steps["clf"]
        payload = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]),
            "feature_names": names,
            "embedding_model": EMBEDDING_MODEL,
            "threshold": 0.5,
        }
        (MODEL_DIR / "domain_classifier.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print("exportado modelos/domain_classifier.json", flush=True)

    print("test por caso:", flush=True)
    for row in prediction_rows:
        print(f"  {row['prediction']:7} {row['prob_domain']:.3f} {row['label']:7} {row['subtype']:14} {row['question'][:70]}", flush=True)


if __name__ == "__main__":
    main()
