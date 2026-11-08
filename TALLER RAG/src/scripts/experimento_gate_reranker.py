"""Experimento: compuerta de dominio con puntuaciones del cross-encoder.

Compara la compuerta actual, basada en embeddings y features de retrieval, con
una que anade la puntuacion del reranker sobre el top-5. La senal del
cross-encoder mide si la evidencia recuperada responde a la pregunta, que es
justo lo que la compuerta necesita decidir.

El script es completo: carga, features, entrenamiento, seleccion por validacion
cruzada, evaluacion en el test interno y en el conjunto reservado, y exporta el
modelo y las metricas. No modifica experimentos anteriores.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.rag.domain import question_type, retrieval_features
from src.rag.pipeline import get_reranker, search_hits

DATA = Path("eval/domain/domain_questions.jsonl")
RESERVED = Path("eval/silver/silver_reservado.jsonl")
MODEL_OUT = Path("modelos/domain_classifier_v2.json")
TABLE_DIR = Path("reports/tablas")
COLLECTION = "rag_full_bge_m3_d1024_v1"
THRESHOLD = 0.4
C_GRID = (0.1, 1.0, 10.0)


def build_features(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Features de la compuerta: cross-encoder, retrieval y tipo de pregunta."""
    reranker = get_reranker()
    features = []
    for index, row in enumerate(rows, start=1):
        hits, _ = search_hits(row["question"], k=5, collection=COLLECTION)
        if hits:
            scores = reranker.predict(
                [(row["question"], hit.get("text", "")) for hit in hits],
                batch_size=16,
                show_progress_bar=False,
            )
            ordered = sorted((float(score) for score in scores), reverse=True)
        else:
            ordered = [0.0]
        ce_top = ordered[0]
        ce_mean = float(np.mean(ordered))
        ce_gap = ce_top - (ordered[1] if len(ordered) > 1 else ce_top)
        ce_min = float(min(ordered))
        features.append(
            [ce_top, ce_mean, ce_gap, ce_min]
            + retrieval_features(row["question"], hits)
            + question_type(row["question"])
        )
        if index % 20 == 0:
            print(f"  features {index}/{len(rows)}", flush=True)
    names = [
        "ce_top",
        "ce_mean",
        "ce_gap",
        "ce_min",
        "top_score",
        "mean_top5",
        "score_gap",
        "title_overlap",
        "entity_in_title",
        "type_capital",
        "type_poblacion",
        "type_autor",
        "type_fecha",
        "type_lugar",
        "type_cantidad",
        "type_otro",
    ]
    return np.asarray(features, dtype=np.float32), names


def make_model(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def rates(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = probabilities >= threshold
    return {
        "f1": f1_score(labels, predictions, pos_label=1),
        "false_rejection": float(np.mean(predictions[labels == 1] == 0)),
        "false_acceptance": float(np.mean(predictions[labels == 0] == 1)),
        "accuracy": float(np.mean(predictions == labels)),
    }


def main() -> None:
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = np.asarray([1 if row["label"] == "dominio" else 0 for row in rows])
    splits = np.asarray([row["split"] for row in rows])
    print(f"{len(rows)} preguntas de entrenamiento", flush=True)
    features, names = build_features(rows)

    train = splits == "train"
    dev = splits == "dev"
    test = splits == "test"

    # Seleccion de C por validacion cruzada sobre train+dev, sin tocar test.
    fit_mask = train | dev
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []
    best_c = None
    best_f1 = -1.0
    for c_value in C_GRID:
        scores = []
        for fit_index, val_index in cv.split(features[fit_mask], labels[fit_mask]):
            model = make_model(c_value)
            model.fit(features[fit_mask][fit_index], labels[fit_mask][fit_index])
            probabilities = model.predict_proba(features[fit_mask][val_index])[:, 1]
            scores.append(rates(labels[fit_mask][val_index], probabilities, THRESHOLD)["f1"])
        mean_f1 = float(np.mean(scores))
        cv_rows.append({"C": c_value, "threshold": THRESHOLD, "cv_f1": mean_f1})
        print(f"  C={c_value}: CV F1={mean_f1:.3f}", flush=True)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_c = c_value

    print(f"C elegido {best_c} con CV F1={best_f1:.3f}", flush=True)

    # Test interno: se entrena solo con train y se evalua en test.
    holdout = make_model(best_c)
    holdout.fit(features[train], labels[train])
    holdout_probabilities = holdout.predict_proba(features[test])[:, 1]
    test_rates = rates(labels[test], holdout_probabilities, THRESHOLD)
    print(
        f"test: F1={test_rates['f1']:.3f} FR={test_rates['false_rejection']:.3f} "
        f"FA={test_rates['false_acceptance']:.3f}",
        flush=True,
    )

    # Modelo final con train+dev, que es el que se exporta.
    final = make_model(best_c)
    final.fit(features[fit_mask], labels[fit_mask])
    scaler = final.named_steps["scaler"]
    clf = final.named_steps["clf"]
    payload = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "feature_names": names,
        "threshold": THRESHOLD,
        "model": "logistic_ce",
        "C": best_c,
    }
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"exportado {MODEL_OUT}", flush=True)

    # Evaluacion externa: conjunto reservado, una sola vez.
    reserved = [json.loads(line) for line in RESERVED.read_text(encoding="utf-8").splitlines() if line.strip()]
    reserved_features, _ = build_features(reserved)
    reserved_labels = np.asarray([1 if row["type"] == "directa" else 0 for row in reserved])
    reserved_probabilities = final.predict_proba(reserved_features)[:, 1]
    reserved_rates = rates(reserved_labels, reserved_probabilities, THRESHOLD)
    print(
        f"reservado: F1={reserved_rates['f1']:.3f} "
        f"FR={reserved_rates['false_rejection']:.3f} "
        f"FA={reserved_rates['false_acceptance']:.3f}",
        flush=True,
    )

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cv_rows).to_csv(TABLE_DIR / "gate_reranker_cv.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "logistic_ce",
                "split": "test",
                "threshold": THRESHOLD,
                "C": best_c,
                **test_rates,
            },
            {
                "model": "logistic_ce",
                "split": "reservado",
                "threshold": THRESHOLD,
                "C": best_c,
                **reserved_rates,
            },
        ]
    ).to_csv(TABLE_DIR / "gate_reranker_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "id": row["id"],
                "type": row["type"],
                "question": row["question"],
                "probability": round(float(probability), 4),
                "prediction": "dominio" if probability >= THRESHOLD else "fuera",
            }
            for row, probability in zip(reserved, reserved_probabilities)
        ]
    ).to_csv(TABLE_DIR / "gate_reranker_reservado.csv", index=False)
    print("artefactos escritos en reports/tablas/", flush=True)


if __name__ == "__main__":
    main()
