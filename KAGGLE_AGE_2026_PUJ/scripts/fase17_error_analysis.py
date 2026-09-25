"""Fase 17 — Análisis de errores del ensamble (predicciones out-of-fold).

Objetivo
    Identificar **qué fenómenos** causan los errores y qué característica falta, en lugar de
    buscar mejoras a ciegas. Responde: ¿dónde se equivoca el modelo y por qué?

Protocolo
    - Predicciones out-of-fold del ensamble con el mismo esquema leave-one-fold-out de la Fase 20
      (el meta-modelo nunca ve el fold que predice).
    - Cortes de error por longitud, retuit, menciones, URLs, texto vacío y pertenencia a un
      clúster de near-duplicados.
    - Ejemplos reales de los pares de clases más confundidos.

Salida
    ``experiments/fase17_error_analysis.json``, ``reports/tables/fase17_*.csv`` y figuras.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, data  # noqa: E402
from src.config import set_seed  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})
OOF_FILES = ["oof_E013_margenes.npy", "oof_E-009.npy", "oof_E-011.npy", "oof_E-010.npy"]


def build_oof_predictions(y: np.ndarray, folds: list[dict]) -> np.ndarray:
    """Predicciones out-of-fold del ensamble (meta-modelo con leave-one-fold-out)."""
    matrices = []
    for name in OOF_FILES:
        matrix = np.load(cfg.FEATURES / name)
        matrices.append(matrix if name == OOF_FILES[0] else np.log(np.clip(matrix, 1e-12, None)))
    features = np.concatenate(matrices, axis=1)
    predictions = np.empty(len(y), dtype=object)
    for fold in folds:
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        meta = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=cfg.SEED)
        meta.fit(features[train_idx], y[train_idx])
        predictions[valid_idx] = meta.predict(features[valid_idx])
    return predictions.astype(str)


def fig_confusion(matrix: pd.DataFrame, normalized: pd.DataFrame) -> str:
    """Mapa de calor de la matriz de confusión normalizada por clase real."""
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    image = axis.imshow(normalized.to_numpy(), cmap="Blues", vmin=0, vmax=normalized.to_numpy().max())
    axis.set_xticks(range(len(cfg.CLASSES)), cfg.CLASSES, rotation=45, ha="right")
    axis.set_yticks(range(len(cfg.CLASSES)), cfg.CLASSES)
    axis.set_xlabel("Predicho")
    axis.set_ylabel("Real")
    axis.set_title("Matriz de confusión del ensamble (out-of-fold, normalizada por fila)")
    for row in range(len(cfg.CLASSES)):
        for column in range(len(cfg.CLASSES)):
            value = normalized.to_numpy()[row, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                      color="white" if value > 0.5 * normalized.to_numpy().max() else "black")
    figure.colorbar(image, ax=axis, shrink=0.8)
    path = cfg.FIGURES / "fase17_matriz_confusion.png"
    figure.savefig(path)
    plt.close(figure)
    return str(path)


def fig_error_by_bucket(table: pd.DataFrame, bucket_column: str, title: str, name: str) -> str:
    """Barras de tasa de error por bucket."""
    figure, axis = plt.subplots(figsize=(7.0, 3.2))
    axis.bar(table[bucket_column].astype(str), table["error_rate"], color="#c05621")
    for index, value in enumerate(table["error_rate"]):
        axis.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    axis.set_ylabel("Tasa de error")
    axis.set_title(title)
    axis.tick_params(axis="x", labelsize=8)
    path = cfg.FIGURES / name
    figure.savefig(path)
    plt.close(figure)
    return str(path)


def main() -> int:
    """Ejecuta el análisis de errores completo."""
    set_seed()
    started = time.time()
    train = data.load_train()
    dense = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]
    y = train[cfg.TARGET].astype(str).to_numpy()
    folds = cv.make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")

    predictions = build_oof_predictions(y, folds)
    accuracy = float((predictions == y).mean())
    macro = float(f1_score(y, predictions, average="macro", labels=cfg.CLASSES))
    print(f"OOF del ensamble: accuracy {accuracy:.4f} | macro-F1 {macro:.4f}", flush=True)

    # --- 1. Matriz de confusión ---
    matrix = pd.DataFrame(confusion_matrix(y, predictions, labels=cfg.CLASSES),
                          index=cfg.CLASSES, columns=cfg.CLASSES)
    normalized = matrix.div(matrix.sum(axis=1), axis=0)
    matrix.to_csv(cfg.TABLES / "fase17_matriz_confusion.csv")
    normalized.round(4).to_csv(cfg.TABLES / "fase17_matriz_confusion_normalizada.csv")
    print(fig_confusion(matrix, normalized), flush=True)

    # --- 2. Pares confundidos ---
    off_diagonal = []
    for true_label in cfg.CLASSES:
        for predicted_label in cfg.CLASSES:
            if true_label == predicted_label:
                continue
            count = int(matrix.loc[true_label, predicted_label])
            off_diagonal.append({
                "real": true_label, "predicho": predicted_label, "casos": count,
                "tasa_sobre_clase": round(count / int(matrix.loc[true_label].sum()), 4),
            })
    confused = pd.DataFrame(off_diagonal).sort_values("casos", ascending=False)
    confused.to_csv(cfg.TABLES / "fase17_pares_confundidos.csv", index=False)

    # --- 3. Reporte por clase ---
    report = pd.DataFrame(classification_report(y, predictions, labels=cfg.CLASSES,
                                                output_dict=True, zero_division=0)).T
    report.round(4).to_csv(cfg.TABLES / "fase17_reporte_por_clase.csv")

    # --- 4. Error por cortes ---
    frame = pd.DataFrame({"y": y, "pred": predictions, "correcto": (predictions == y).astype(int),
                          "n_chars": dense["n_chars"].to_numpy(),
                          "is_rt": dense["is_rt"].to_numpy(),
                          "n_mention": dense["n_mention"].to_numpy(),
                          "n_url": dense["n_url"].to_numpy(),
                          "cluster": cluster_ids})
    cluster_sizes = pd.Series(cluster_ids).map(pd.Series(cluster_ids).value_counts())
    frame["en_cluster"] = (cluster_sizes > 1).to_numpy()

    buckets: dict[str, pd.DataFrame] = {}
    frame["bucket_longitud"] = pd.qcut(frame["n_chars"], 5,
                                       labels=["Q1 (más cortos)", "Q2", "Q3", "Q4", "Q5 (más largos)"])
    for column, label in (("bucket_longitud", "longitud"), ("is_rt", "retuit"),
                          ("n_url", "urls"), ("en_cluster", "near-duplicado")):
        table = (frame.groupby(column, observed=True)
                 .agg(n=("correcto", "size"), error_rate=("correcto", lambda s: round(1 - s.mean(), 4)))
                 .reset_index().rename(columns={column: "bucket"}))
        table["corte"] = label
        buckets[label] = table
    bucket_table = pd.concat(buckets.values(), ignore_index=True)
    bucket_table.to_csv(cfg.TABLES / "fase17_error_por_cortes.csv", index=False)
    print(fig_error_by_bucket(buckets["longitud"], "bucket",
                             "Tasa de error por quintil de longitud del tweet",
                             "fase17_error_longitud.png"), flush=True)

    # --- 5. Ejemplos de error en los pares más confundidos ---
    examples: list[dict] = []
    for _, row in confused.head(6).iterrows():
        subset = frame[(frame["y"] == row["real"]) & (frame["pred"] == row["predicho"])]
        for _, sample in subset.head(3).iterrows():
            index = sample.name
            examples.append({
                "real": row["real"], "predicho": row["predicho"],
                "texto": train[cfg.TEXT_COL].iloc[index][:160],
                "n_chars": int(sample["n_chars"]),
            })
    examples_table = pd.DataFrame(examples)
    examples_table.to_csv(cfg.TABLES / "fase17_ejemplos_error.csv", index=False)

    # --- 6. Foco en el par adyacente 35-49 / 50-64 ---
    adjacent = frame[frame["y"].isin(["35-49", "50-64"])]
    adjacent_confusion = pd.crosstab(adjacent["y"], adjacent["pred"], normalize="index").round(4)

    summary = {
        "oof_accuracy": round(accuracy, 4),
        "oof_macro_f1": round(macro, 4),
        "top_pares_confundidos": confused.head(6).to_dict(orient="records"),
        "error_por_cortes": bucket_table.to_dict(orient="records"),
        "confusion_adyacente_35_49_50_64": adjacent_confusion.to_dict(),
        "f1_por_clase": report["f1-score"].round(4).to_dict(),
        "segundos": round(time.time() - started, 1),
    }
    (cfg.EXPERIMENTS / "fase17_error_analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== PARES MÁS CONFUNDIDOS ===")
    print(confused.head(8).to_string(index=False))
    print("\n=== F1 POR CLASE ===")
    print(report["f1-score"].round(4).to_string())
    print("\n=== ERROR POR CORTES ===")
    print(bucket_table.to_string(index=False))
    print("\n=== CONFUSIÓN ADYACENTE 35-49 / 50-64 ===")
    print(adjacent_confusion.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
