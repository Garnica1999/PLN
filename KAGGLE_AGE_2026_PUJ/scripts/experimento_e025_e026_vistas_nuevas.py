"""E-025 y E-026 — Dos vistas nuevas: ponderación BM25 y kNN sobre SVD.

E-025 — BM25 en lugar de TF-IDF
    Hipótesis H21: BM25 (k1=1.2, b=0.75) satura la frecuencia de término y normaliza por longitud
    del documento, lo que suele superar a TF-IDF en documentos cortos. Es una ponderación
    estadística, no un modelo de vectorización (permitida por las reglas, D-001).

E-026 — kNN sobre la SVD del espacio de caracteres
    Hipótesis H22: un clasificador de vecinos más cercanos (k=25, similitud coseno) es una familia
    de modelo totalmente distinta de los lineales y de los árboles; aunque rinda menos solo, puede
    aportar diversidad al ensamble (como el MLP: +0,0066).

Protocolo
    Mismas 5 particiones agrupadas; la representación se construye una vez por fold y se reutiliza.
    Se guardan las matrices OOF de ambos modelos.

Salida
    ``experiments/e025_e026_vistas_nuevas.json`` + matrices OOF.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, data, preprocessing  # noqa: E402
from src.config import set_seed  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "e025_e026_vistas_nuevas.json"
CHAR_CONFIG = {"analyzer": "char_wb", "ngram_range": (3, 6), "min_df": 5,
               "max_features": 300_000, "lowercase": True}
MENTION_CONFIG = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2, "lowercase": True}
K1, B = 1.2, 0.75
N_NEIGHBORS = 25
SVD_COMPONENTS = 64


def tokens_of(series: pd.Series, pattern: str) -> pd.Series:
    """Extrae tokens que cumplen un patrón."""
    return preprocessing.collapse_spaces(
        series.str.findall(pattern).apply(lambda values: " ".join(values).lower()))


def bm25_weights(counts: sp.csr_matrix) -> sp.csr_matrix:
    """Convierte una matriz de conteos en pesos BM25 (k1, b) fila a fila.

    ``w_ij = idf_j * tf_ij * (k1 + 1) / (tf_ij + k1 * (1 - b + b * dl_i / avgdl))``

    Se opera directamente sobre las coordenadas de los elementos no nulos, que es la forma
    eficiente de aplicar una transformación elemento a elemento que depende de la fila.
    """
    n_docs, _ = counts.shape
    document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log((n_docs - document_frequency + 0.5) / (document_frequency + 0.5) + 1.0)
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    average_length = lengths.mean() if lengths.mean() > 0 else 1.0
    normalization = K1 * (1 - B + B * lengths / average_length)

    coordinates = counts.tocoo()
    term_frequency = coordinates.data.astype(np.float64)
    rows, columns = coordinates.row, coordinates.col
    numerator = term_frequency * (K1 + 1) * idf[columns]
    denominator = term_frequency + normalization[rows]
    values = numerator / denominator
    return sp.csr_matrix((values, (rows, columns)), shape=counts.shape)


def knn_cosine_probabilities(z_train: np.ndarray, y_train: np.ndarray,
                             z_valid: np.ndarray, k: int = N_NEIGHBORS) -> np.ndarray:
    """Probabilidades por voto ponderado de los k vecinos más cercanos (coseno)."""
    train_norms = np.linalg.norm(z_train, axis=1) + 1e-12
    valid_norms = np.linalg.norm(z_valid, axis=1) + 1e-12
    normalized_train = z_train / train_norms[:, None]
    normalized_valid = z_valid / valid_norms[:, None]

    class_index = {label: index for index, label in enumerate(cfg.CLASSES)}
    targets = np.array([class_index[label] for label in y_train])
    one_hot = np.zeros((len(y_train), len(cfg.CLASSES)), dtype=np.float32)
    one_hot[np.arange(len(y_train)), targets] = 1.0

    probabilities = np.zeros((z_valid.shape[0], len(cfg.CLASSES)), dtype=np.float32)
    block = 2000
    for start in range(0, z_valid.shape[0], block):
        similarity = normalized_valid[start:start + block] @ normalized_train.T
        top = np.argpartition(-similarity, k, axis=1)[:, :k]
        weights = np.take_along_axis(similarity, top, axis=1)
        votes = one_hot[top] * np.clip(weights, 0, None)[:, :, None]
        totals = votes.sum(axis=1) + 1e-12
        probabilities[start:start + block] = totals / totals.sum(axis=1, keepdims=True)
    return probabilities


def main() -> int:
    """Ejecuta E-025 (BM25) y E-026 (kNN sobre SVD) guardando sus OOF."""
    set_seed()
    started = time.time()
    train = data.load_train()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]
    y = train[cfg.TARGET].astype(str).to_numpy()
    texts = train[cfg.TEXT_COL]
    folds = cv.make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}
    oof_bm25 = np.zeros((len(train), len(cfg.CLASSES)), dtype=np.float32)
    oof_knn = np.zeros((len(train), len(cfg.CLASSES)), dtype=np.float32)

    for fold in folds:
        train_idx, valid_idx = fold["train_idx"], fold["val_idx"]

        # --- E-025: BM25 sobre char 3-6 + menciones ---
        char_counts = CountVectorizer(**CHAR_CONFIG)
        mention_counts = CountVectorizer(**MENTION_CONFIG)
        counts_train = sp.hstack([
            char_counts.fit_transform(texts.iloc[train_idx]),
            mention_counts.fit_transform(tokens_of(texts.iloc[train_idx], r"@\w+"))]).tocsr()
        counts_valid = sp.hstack([
            char_counts.transform(texts.iloc[valid_idx]),
            mention_counts.transform(tokens_of(texts.iloc[valid_idx], r"@\w+"))]).tocsr()
        weights_train = bm25_weights(counts_train)
        weights_valid = bm25_weights(counts_valid)

        model = LinearSVC(C=0.2, max_iter=3000)
        model.fit(weights_train, y[train_idx])
        order = [list(model.classes_).index(label) for label in cfg.CLASSES]
        decision = model.decision_function(weights_valid)[:, order]
        oof_bm25[valid_idx] = decision
        predictions = np.array(cfg.CLASSES)[np.argmax(decision, axis=1)]
        macro_bm25 = f1_score(y[valid_idx], predictions, average="macro")

        # --- E-026: kNN sobre SVD del espacio de caracteres ---
        char_tfidf = CountVectorizer(**CHAR_CONFIG)
        x_train = char_tfidf.fit_transform(texts.iloc[train_idx])
        x_valid = char_tfidf.transform(texts.iloc[valid_idx])
        svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=cfg.SEED)
        z_train = svd.fit_transform(x_train)
        z_valid = svd.transform(x_valid)
        probabilities = knn_cosine_probabilities(z_train, y[train_idx], z_valid)
        oof_knn[valid_idx] = probabilities
        predictions_knn = np.array(cfg.CLASSES)[np.argmax(probabilities, axis=1)]
        macro_knn = f1_score(y[valid_idx], predictions_knn, average="macro")

        for key, macro, extra, accuracy in (
                ("E-025", macro_bm25, "BM25 (k1=1.2, b=0.75) char 3-6 + menciones + SVM C=0.2",
                 accuracy_score(y[valid_idx], predictions)),
                ("E-026", macro_knn, f"kNN (k={N_NEIGHBORS}) coseno sobre SVD({SVD_COMPONENTS})",
                 accuracy_score(y[valid_idx], predictions_knn))):
            results.setdefault(key, {"per_fold": [], "descripcion": extra})
            results[key]["per_fold"].append({"fold": fold["fold"], "macro_f1": macro,
                                             "accuracy": accuracy})
        print(f"[fold {fold['fold']}] E-025 BM25 {macro_bm25:.4f} | E-026 kNN {macro_knn:.4f} "
              f"({round(time.time() - started, 1)}s)", flush=True)
        RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    for key in ("E-025", "E-026"):
        frame = pd.DataFrame(results[key]["per_fold"])
        results[key]["macro_f1_mean"] = float(frame["macro_f1"].mean())
        results[key]["macro_f1_std"] = float(frame["macro_f1"].std())
        results[key]["accuracy_mean"] = float(frame["accuracy"].mean())
    results["segundos"] = round(time.time() - started, 1)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    np.save(cfg.FEATURES / "oof_E025_bm25.npy", oof_bm25)
    np.save(cfg.FEATURES / "oof_E026_knn.npy", oof_knn)

    print("\n=== E-025 / E-026 — VISTAS NUEVAS ===")
    for key in ("E-025", "E-026"):
        print(f"{key} ({results[key]['descripcion']}): macro-F1 "
              f"{results[key]['macro_f1_mean']:.4f} ± {results[key]['macro_f1_std']:.4f}")
    print("\nReferencias: SVM TF-IDF 0,5658 · MLP denso 0,4071 · ensamble 0,5860")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
