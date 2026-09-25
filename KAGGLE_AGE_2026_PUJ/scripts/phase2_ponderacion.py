"""FASE 2 / L6 — Ponderación condicional (meta-modelo 2.0) guiada por L1 y L5.

Observación (Fase 1)
    - **25,94%** de los tweets: el ensamble falla y **al menos un modelo base acierta**.
    - Los modelos que más rescatan son los **débiles pero diversos**: softmax ordinal (36%),
      centroides (31%), MLP densas+SVD (30,5%), LogReg palabra (27,9%), ComplementNB (25,2%).
    - El error depende fuertemente del **tipo de tweet** (`solo_url` 61,1%; `mencion_alta` 19,2%;
      `hashtag_alto` 20,4%) y de la **longitud** (54,5% en ≤10 caracteres → 35,1% en 130-141).
    Estas tres piezas apuntan a la misma hipótesis: el meta-modelo aplica **pesos globales** y no
    aprovecha que la fiabilidad de cada miembro depende del tipo de tweet.

Hipótesis
    H27  Un meta-modelo **segmentado** (uno por tipo de tweet o por banda de longitud) recupera parte
         de los casos donde un miembro acierta y el ensamble falla.
    H28  El efecto lo captura mejor la **longitud** (efecto monótono y continuo) que la tipología.
    H29  Un **MLP superficial** como meta-modelo puede aprender la ponderación condicional de forma
         suave sin segmentar (evita el problema de celdas pequeñas).

Protocolo
    Leave-one-fold-out sobre las mismas particiones agrupadas. Los segmentos con menos de 800 filas
    en el train del fold se fusionan con el segmento mayoritario.

Salida: ``experiments/phase2_ponderacion.json`` y ``reports/tables/phase2_ponderacion.csv``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, data, textometry  # noqa: E402
from src.config import set_seed  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "phase2_ponderacion.json"
MEMBER_FILES = [("oof_E013_margenes.npy", "raw"), ("oof_E016_margenes.npy", "raw"),
                ("oof_E-009.npy", "log"), ("oof_E-011.npy", "log"), ("oof_E-010.npy", "log"),
                ("oof_E014_margenes.npy", "raw"), ("oof_E015_ordinal.npy", "raw"),
                ("oof_E015_softmax.npy", "log"), ("oof_E017_probabilidades.npy", "log"),
                ("oof_E019b_densas_svd.npy", "log"), ("oof_E021a_ordinal.npy", "log"),
                ("oof_E021b_rocchio.npy", "log"), ("oof_E028a_pares.npy", "raw"),
                ("oof_ADV05_nbsvm.npy", "raw")]
DENSE_FEATURES = [name for name in textometry.TEXTOMETRY_FEATURES + textometry.STYLE_FEATURES
                  if name not in {"n_emoji", "emoji_diversity", "has_emoji"}]
MIN_SEGMENT = 800


def build_features() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Reúne la matriz del meta-modelo v10 y los metadatos de segmentación."""
    from scripts.phase1_diagnostico import classify_tweet_type

    train = data.load_train()
    dense = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")
    y = train[cfg.TARGET].astype(str).to_numpy()
    parts = []
    for filename, transform in MEMBER_FILES:
        matrix = np.load(cfg.FEATURES / filename)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        parts.append(np.log(np.clip(matrix, 1e-12, None)) if transform == "log" else matrix)
    structured = pd.read_parquet(cfg.FEATURES / "age_structured_features.parquet").to_numpy(float)
    supervised = np.load(cfg.FEATURES / "age_supervised_features.npy")
    features = np.concatenate([np.concatenate(parts, axis=1),
                               dense[DENSE_FEATURES].to_numpy(float), structured, supervised],
                              axis=1)
    tweet_type = classify_tweet_type(dense)
    length_band = pd.qcut(dense["n_chars"], 4, labels=["L1", "L2", "L3", "L4"]).astype(str).to_numpy()
    return features, y, dense, tweet_type, length_band


def merge_small_segments(segments: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fusiona segmentos con pocas filas en el segmento mayoritario."""
    counts = pd.Series(segments).value_counts()
    majority = counts.index[0]
    small = set(counts[counts < MIN_SEGMENT].index)
    return np.where(np.isin(segments, list(small)), majority, segments)


def evaluate_global(features: np.ndarray, y: np.ndarray, folds: list[dict],
                    model_kind: str = "logreg") -> tuple[float, float, list]:
    """Meta-modelo global (referencia)."""
    scores = []
    for fold in folds:
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        meta = make_model(model_kind)
        meta.fit(features[train_idx], y[train_idx])
        scores.append(float(f1_score(y[valid_idx], meta.predict(features[valid_idx]),
                                     average="macro", labels=cfg.CLASSES)))
    return float(np.mean(scores)), float(np.std(scores)), [round(value, 4) for value in scores]


class EncodedMLP:
    """MLP con etiquetas codificadas a enteros.

    scikit-learn 1.8 falla al calcular ``np.isnan`` sobre predicciones de tipo cadena en la ruta de
    *early stopping*; se codifican las etiquetas y se decodifican al predecir.
    """

    def __init__(self) -> None:
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, batch_size=512,
                                   learning_rate_init=1e-3, max_iter=40, early_stopping=True,
                                   n_iter_no_change=5, random_state=cfg.SEED)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "EncodedMLP":
        self.encoder.fit(labels)
        return self.model.fit(self.scaler.fit_transform(features), self.encoder.transform(labels)) and self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.encoder.inverse_transform(self.model.predict(self.scaler.transform(features)))


def make_model(kind: str):
    """Crea el meta-modelo indicado."""
    if kind == "logreg":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                    random_state=cfg.SEED))])
    if kind == "mlp":
        return EncodedMLP()
    raise ValueError(kind)


def evaluate_segmented(features: np.ndarray, y: np.ndarray, folds: list[dict],
                       segments: np.ndarray, kind: str = "logreg") -> tuple[float, float, list, list]:
    """Meta-modelo segmentado: un modelo por segmento, entrenado solo con sus filas."""
    encoder = LabelEncoder().fit(y)
    scores = []
    segment_rows: list[dict] = []
    for fold in folds:
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        segment_train = merge_small_segments(segments[train_idx], y[train_idx])
        predictions = np.empty(len(valid_idx), dtype=object)
        for segment in np.unique(segment_train):
            train_mask = segment_train == segment
            valid_mask = segments[valid_idx] == segment
            if valid_mask.sum() == 0:
                continue
            meta = make_model(kind)
            meta.fit(features[train_idx][train_mask], y[train_idx][train_mask])
            predictions[valid_mask] = meta.predict(features[valid_idx][valid_mask])
            segment_rows.append({
                "segmento": segment, "n_train": int(train_mask.sum()),
                "n_valid": int(valid_mask.sum()),
                "accuracy": round(float((predictions[valid_mask] == y[valid_idx][valid_mask]).mean()), 4),
            })
        # Filas cuyo segmento no existía en train: se predicen con el modelo global del fold.
        missing = pd.isna(predictions)
        if missing.any():
            meta = make_model(kind)
            meta.fit(features[train_idx], y[train_idx])
            predictions[missing] = meta.predict(features[valid_idx][missing])
        scores.append(float(f1_score(y[valid_idx], predictions.astype(str), average="macro",
                                     labels=cfg.CLASSES)))
    return float(np.mean(scores)), float(np.std(scores)), [round(v, 4) for v in scores], segment_rows


def main() -> int:
    """Evalúa la ponderación condicional por tipo de tweet, longitud y meta no lineal."""
    set_seed()
    started = time.time()
    features, y, dense, tweet_type, length_band = build_features()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    folds = cv.make_folds(y, groups=clusters["cluster"].to_numpy()[: len(y)],
                          n_splits=5, n_repeats=1, scheme="group")
    print(f"features del meta: {features.shape} | tipos: {len(set(tweet_type))} | "
          f"bandas de longitud: 4", flush=True)

    rows = []
    base_mean, base_std, base_folds = evaluate_global(features, y, folds, "logreg")
    rows.append({"variante": "global LogReg (referencia v10)", "macro_f1": round(base_mean, 4),
                 "std": round(base_std, 4), "per_fold": base_folds})
    print(f"[global LogReg] {base_mean:.4f} ± {base_std:.4f}", flush=True)

    mlp_mean, mlp_std, mlp_folds = evaluate_global(features, y, folds, "mlp")
    rows.append({"variante": "global MLP (ponderación suave)", "macro_f1": round(mlp_mean, 4),
                 "std": round(mlp_std, 4), "per_fold": mlp_folds})
    print(f"[global MLP] {mlp_mean:.4f} ± {mlp_std:.4f} ({mlp_mean - base_mean:+.4f})", flush=True)

    type_mean, type_std, type_folds, type_rows = evaluate_segmented(features, y, folds, tweet_type)
    rows.append({"variante": "segmentado por tipo de tweet", "macro_f1": round(type_mean, 4),
                 "std": round(type_std, 4), "per_fold": type_folds})
    print(f"[segmentado por tipo] {type_mean:.4f} ± {type_std:.4f} "
          f"({type_mean - base_mean:+.4f})", flush=True)

    length_mean, length_std, length_folds, length_rows = evaluate_segmented(features, y, folds,
                                                                            length_band)
    rows.append({"variante": "segmentado por banda de longitud", "macro_f1": round(length_mean, 4),
                 "std": round(length_std, 4), "per_fold": length_folds})
    print(f"[segmentado por longitud] {length_mean:.4f} ± {length_std:.4f} "
          f"({length_mean - base_mean:+.4f})", flush=True)

    table = pd.DataFrame(rows)
    table.drop(columns=["per_fold"]).to_csv(cfg.TABLES / "phase2_ponderacion.csv", index=False)
    best = table.loc[table["macro_f1"].idxmax()]
    payload = {
        "variantes": rows,
        "mejor": best["variante"], "mejor_macro_f1": float(best["macro_f1"]),
        "referencia_global": round(base_mean, 4),
        "detalle_por_tipo": type_rows,
        "detalle_por_longitud": length_rows,
        "segundos": round(time.time() - started, 1),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
    print("\n=== L6 — PONDERACIÓN CONDICIONAL ===")
    print(table.drop(columns=["per_fold"]).to_string(index=False))
    print(f"\nmejor: {payload['mejor']} = {payload['mejor_macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
