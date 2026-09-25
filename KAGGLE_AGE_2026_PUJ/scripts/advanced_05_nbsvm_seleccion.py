"""ADV-05 — NB-SVM (prioridad del brief §22/§P) y selección de features del meta-modelo (§47).

Parte A — **NB-SVM** (Wang & Manning 2012), uno-vs-rest: se transforma la representación TF-IDF
multiplicándola elemento a elemento por el log-count ratio de cada clase (calculado **solo con el
train del fold**) y se entrena un SVM lineal por clase. Es una representación clásica de la
literatura de clasificación de texto corto y no estaba evaluada en este proyecto.

Parte B — **Selección de features** del meta-modelo: con 323 features candidatas, se evalúa si un
subconjunto seleccionado (por información mutua estimada fuera del fold de evaluación) rinde igual
o mejor que el conjunto completo.

Protocolo
    - Parte A: primero **un solo fold** para medir el potencial a bajo costo; si el macro-F1
      standalone es competitivo (> 0,54), se corre completo.
    - Parte B: selección estimada en 4 folds y evaluada en el quinto (leave-one-fold-out).

Salida: ``experiments/advanced_05_nbsvm_seleccion.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, data, preprocessing, textometry  # noqa: E402
from src.config import set_seed  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "advanced_05_nbsvm_seleccion.json"
CHAR_CONFIG = {"analyzer": "char_wb", "ngram_range": (3, 6), "min_df": 5,
               "sublinear_tf": True, "max_features": 300_000, "lowercase": True}
MENTION_CONFIG = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2,
                  "sublinear_tf": True, "lowercase": True}
SOURCES = [("oof_E013_margenes.npy", False), ("oof_E016_margenes.npy", False),
           ("oof_E-009.npy", True), ("oof_E-011.npy", True), ("oof_E-010.npy", True),
           ("oof_E014_margenes.npy", False), ("oof_E015_ordinal.npy", False),
           ("oof_E015_softmax.npy", True), ("oof_E017_probabilidades.npy", True),
           ("oof_E019b_densas_svd.npy", True), ("oof_E021a_ordinal.npy", True),
           ("oof_E021b_rocchio.npy", True), ("oof_E028a_pares.npy", False)]
DENSE_FEATURES = [name for name in textometry.TEXTOMETRY_FEATURES + textometry.STYLE_FEATURES
                  if name not in {"n_emoji", "emoji_diversity", "has_emoji"}]


def tokens_of(series: pd.Series, pattern: str) -> pd.Series:
    """Extrae tokens que cumplen un patrón."""
    return preprocessing.collapse_spaces(
        series.str.findall(pattern).apply(lambda values: " ".join(values).lower()))


def nb_log_count_ratio(x_train: sp.csr_matrix, binary_target: np.ndarray) -> np.ndarray:
    """Log-count ratio de la clase positiva frente al resto (Wang & Manning 2012)."""
    positive = np.asarray(x_train[binary_target == 1].sum(axis=0)).ravel() + 1.0
    negative = np.asarray(x_train[binary_target == 0].sum(axis=0)).ravel() + 1.0
    return np.log((positive / positive.sum()) / (negative / negative.sum()))


def nb_svm_fold(texts: pd.Series, y: np.ndarray, train_idx: np.ndarray,
                valid_idx: np.ndarray, return_margins: bool = False):
    """Entrena NB-SVM uno-vs-rest en un fold y devuelve las predicciones (y márgenes)."""
    char = TfidfVectorizer(**CHAR_CONFIG)
    mentions = TfidfVectorizer(**MENTION_CONFIG)
    x_train = sp.hstack([char.fit_transform(texts.iloc[train_idx]),
                         mentions.fit_transform(tokens_of(texts.iloc[train_idx], r"@\w+"))]).tocsr()
    x_valid = sp.hstack([char.transform(texts.iloc[valid_idx]),
                         mentions.transform(tokens_of(texts.iloc[valid_idx], r"@\w+"))]).tocsr()
    y_train = y[train_idx]

    margins = np.zeros((len(valid_idx), len(cfg.CLASSES)), dtype=np.float32)
    for position, label in enumerate(cfg.CLASSES):
        binary_target = (y_train == label).astype(int)
        ratio = nb_log_count_ratio(x_train, binary_target)
        model = LinearSVC(C=0.2, max_iter=3000)
        model.fit(x_train.multiply(ratio).tocsr(), binary_target)
        margins[:, position] = model.decision_function(x_valid.multiply(ratio).tocsr())
    predictions = np.array(cfg.CLASSES)[np.argmax(margins, axis=1)]
    if return_margins:
        return predictions, margins
    return predictions


def evaluate_meta(features: np.ndarray, y: np.ndarray, folds: list[dict]) -> tuple[float, float]:
    """macro-F1 del meta-modelo con leave-one-fold-out."""
    scores = []
    for fold in folds:
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        meta = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                    random_state=cfg.SEED))])
        meta.fit(features[train_idx], y[train_idx])
        scores.append(float(f1_score(y[valid_idx], meta.predict(features[valid_idx]),
                                     average="macro", labels=cfg.CLASSES)))
    return float(np.mean(scores)), float(np.std(scores))


def main() -> int:
    """Ejecuta el test de NB-SVM y la selección de features del meta-modelo."""
    set_seed()
    started = time.time()
    payload: dict = {}
    train = data.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    folds = cv.make_folds(y, groups=clusters["cluster"].to_numpy()[: len(train)],
                          n_splits=5, n_repeats=1, scheme="group")

    # ---------------------------------------------------------------- Parte A: NB-SVM
    print("[A] NB-SVM — prueba de un fold...", flush=True)
    fold = folds[0]
    predictions = nb_svm_fold(train[cfg.TEXT_COL], y, fold["train_idx"], fold["val_idx"])
    macro_nb = float(f1_score(y[fold["val_idx"]], predictions, average="macro", labels=cfg.CLASSES))
    reference = float(f1_score(y[fold["val_idx"]],
                               np.array(cfg.CLASSES)[np.argmax(
                                   np.load(cfg.FEATURES / "oof_E016_margenes.npy")[fold["val_idx"]],
                                   axis=1)], average="macro", labels=cfg.CLASSES))
    print(f"    NB-SVM fold 0: {macro_nb:.4f} | referencia SVM TF-IDF mismo fold: {reference:.4f}",
          flush=True)
    payload["nbsvm_fold0"] = {"macro_f1": round(macro_nb, 4), "referencia_svm": round(reference, 4)}

    if macro_nb > 0.54:
        print("[A] competitivo: ejecutando los 5 folds y guardando márgenes OOF...", flush=True)
        margins_oof = np.zeros((len(train), len(cfg.CLASSES)), dtype=np.float32)
        per_fold = []
        for fold in folds:
            predictions, margins = nb_svm_fold(train[cfg.TEXT_COL], y, fold["train_idx"],
                                               fold["val_idx"], return_margins=True)
            margins_oof[fold["val_idx"]] = margins
            per_fold.append(float(f1_score(y[fold["val_idx"]], predictions, average="macro",
                                           labels=cfg.CLASSES)))
        np.save(cfg.FEATURES / "oof_ADV05_nbsvm.npy", margins_oof)
        payload["nbsvm_cv"] = {"macro_f1": round(float(np.mean(per_fold)), 4),
                               "per_fold": [round(value, 4) for value in per_fold]}
        print(f"    NB-SVM CV: {np.mean(per_fold):.4f} ± {np.std(per_fold):.4f}", flush=True)
    else:
        payload["nbsvm_cv"] = None
        print("    no competitivo en el fold de prueba: no se ejecuta completo", flush=True)

    # ---------------------------------------------------------------- Parte B: selección
    print("\n[B] selección de features del meta-modelo (LOFO)...", flush=True)
    dense = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")[DENSE_FEATURES].to_numpy(float)
    structured = pd.read_parquet(cfg.FEATURES / "age_structured_features.parquet").to_numpy(float)
    supervised = np.load(cfg.FEATURES / "age_supervised_features.npy")
    matrices = []
    for filename, probabilistic in SOURCES:
        matrix = np.load(cfg.FEATURES / filename)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        matrices.append(np.log(np.clip(matrix, 1e-12, None)) if probabilistic else matrix)
    features = np.concatenate([np.concatenate(matrices, axis=1), dense, structured, supervised],
                              axis=1)
    print(f"    features candidatas: {features.shape[1]}", flush=True)

    baseline_mean, baseline_std = evaluate_meta(features, y, folds)
    print(f"    conjunto completo: {baseline_mean:.4f} ± {baseline_std:.4f}", flush=True)

    selection_rows = [{"variante": f"completo ({features.shape[1]})",
                       "macro_f1": round(baseline_mean, 4), "std": round(baseline_std, 4)}]
    for k in (50, 100, 150, 200):
        scores = []
        for fold in folds:
            select_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
            valid_idx = fold["val_idx"]
            f_scores, _ = f_classif(features[select_idx], y[select_idx])
            f_scores = np.nan_to_num(f_scores, nan=0.0)
            keep = np.argsort(-f_scores)[:k]
            selected = features[:, keep]
            meta = Pipeline([("scaler", StandardScaler()),
                             ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                        random_state=cfg.SEED))])
            meta.fit(selected[select_idx], y[select_idx])
            scores.append(float(f1_score(y[valid_idx], meta.predict(selected[valid_idx]),
                                         average="macro", labels=cfg.CLASSES)))
        mean, std = float(np.mean(scores)), float(np.std(scores))
        selection_rows.append({"variante": f"top-{k} por ANOVA F",
                               "macro_f1": round(mean, 4), "std": round(std, 4)})
        print(f"    top-{k:3d}: {mean:.4f} ± {std:.4f} ({mean - baseline_mean:+.4f})", flush=True)

    payload["seleccion"] = selection_rows
    payload["segundos"] = round(time.time() - started, 1)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== ADV-05 — NB-SVM Y SELECCIÓN ===")
    print(pd.DataFrame(selection_rows).to_string(index=False))
    if payload["nbsvm_cv"]:
        print(f"\nNB-SVM CV: {payload['nbsvm_cv']['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
