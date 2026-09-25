"""Verificación anti-leakage de E-001 (obligatoria antes de aceptar el resultado).

Dispara el protocolo del plan: un macro-F1 > 0,50 debe investigarse antes de celebrarse.

Pruebas
    1. **Control de permutación**: mismas features, mismas particiones, etiquetas
       aleatorizadas. Debe dar macro-F1 ≈ 1/6 = 0,167. Si diera alto, habría fuga estructural.
    2. **Análisis por similitud**: para las filas de validación se mide la similitud coseno
       máxima contra una muestra del train del fold. Si la exactitud se mantiene en el tramo
       de baja similitud, el modelo generaliza y no está memorizando near-duplicados.
    3. **Trazabilidad de ajuste**: se confirma que los vectorizadores se ajustan solo con el
       train de cada fold (inspección del pipeline).

Salida: ``experiments/E-001_verificacion_leakage.json``.
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
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, data  # noqa: E402
from src.config import set_seed  # noqa: E402

WORD_CONFIG = {"ngram_range": (1, 2), "min_df": 3, "sublinear_tf": True,
               "max_features": 300_000, "lowercase": True}
CHAR_CONFIG = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 5,
               "sublinear_tf": True, "max_features": 300_000, "lowercase": True}
MODEL_CONFIG = {"C": 1.0, "max_iter": 3000}
REFERENCE_SAMPLE = 20_000
VALIDATION_SAMPLE = 6_000
BLOCK = 500


def make_matrices(train_texts: pd.Series, valid_texts: pd.Series):
    """Ajusta los vectorizadores con train y transforma validación."""
    word = TfidfVectorizer(**WORD_CONFIG)
    char = TfidfVectorizer(**CHAR_CONFIG)
    x_train = hstack([word.fit_transform(train_texts), char.fit_transform(train_texts)]).tocsr()
    x_valid = hstack([word.transform(valid_texts), char.transform(valid_texts)]).tocsr()
    return x_train, x_valid


def max_cosine_similarity(x_train, x_valid_sample, reference_idx: np.ndarray) -> np.ndarray:
    """Similitud coseno máxima de cada fila de validación contra la muestra de referencia."""
    reference = x_train[reference_idx]
    reference_norms = np.sqrt(reference.multiply(reference).sum(axis=1)).A.ravel()
    valid_norms = np.sqrt(x_valid_sample.multiply(x_valid_sample).sum(axis=1)).A.ravel()
    best = np.zeros(x_valid_sample.shape[0], dtype=np.float32)
    for start in range(0, x_valid_sample.shape[0], BLOCK):
        block = x_valid_sample[start:start + BLOCK]
        similarity = (block @ reference.T).toarray()
        denominator = np.outer(valid_norms[start:start + BLOCK], reference_norms) + 1e-12
        best[start:start + BLOCK] = (similarity / denominator).max(axis=1)
    return best


def main() -> int:
    """Ejecuta las tres pruebas y guarda el veredicto."""
    set_seed()
    started = time.time()
    train = data.load_train()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]
    y = train[cfg.TARGET].astype(str).to_numpy()
    texts = train[cfg.TEXT_COL]

    folds = cv.make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")
    fold = folds[0]
    train_idx, valid_idx = fold["train_idx"], fold["val_idx"]

    print("1) Control de permutación (etiquetas aleatorizadas)...", flush=True)
    x_train, x_valid = make_matrices(texts.iloc[train_idx], texts.iloc[valid_idx])
    rng = np.random.default_rng(cfg.SEED)
    shuffled = rng.permutation(y[train_idx])
    model = LinearSVC(**MODEL_CONFIG)
    model.fit(x_train, shuffled)
    permutation_macro_f1 = f1_score(y[valid_idx], model.predict(x_valid), average="macro")
    print(f"   macro-F1 con etiquetas permutadas: {permutation_macro_f1:.4f} (esperado ≈ 0.167)",
          flush=True)

    print("2) Exactitud por similitud coseno máxima contra el train del fold...", flush=True)
    model_real = LinearSVC(**MODEL_CONFIG)
    model_real.fit(x_train, y[train_idx])
    predictions = model_real.predict(x_valid)

    sample_size = min(VALIDATION_SAMPLE, x_valid.shape[0])
    sample_positions = rng.choice(x_valid.shape[0], size=sample_size, replace=False)
    reference_size = min(REFERENCE_SAMPLE, x_train.shape[0])
    reference_idx = rng.choice(x_train.shape[0], size=reference_size, replace=False)
    similarities = max_cosine_similarity(x_train, x_valid[sample_positions], reference_idx)

    reference_labels = y[valid_idx][sample_positions]
    predicted_labels = predictions[sample_positions]
    correct = (reference_labels == predicted_labels).astype(float)
    buckets = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01)]
    by_similarity = []
    for low, high in buckets:
        mask = (similarities >= low) & (similarities < high)
        if mask.sum() == 0:
            continue
        by_similarity.append({
            "rango_similitud": f"{low:.2f}-{high:.2f}",
            "n": int(mask.sum()),
            "share_pct": round(100 * float(mask.mean()), 3),
            "accuracy": round(float(correct[mask].mean()), 4),
        })
    for row in by_similarity:
        print(f"   sim {row['rango_similitud']}: n={row['n']:5d} ({row['share_pct']:.1f}%) "
              f"acc={row['accuracy']:.4f}", flush=True)

    clean_mask = similarities < 0.85
    clean_accuracy = float(correct[clean_mask].mean()) if clean_mask.sum() else float("nan")
    print(f"   exactitud en el tramo 'limpio' (<0.85): {clean_accuracy:.4f} "
          f"(n={int(clean_mask.sum())})", flush=True)

    verdict = {
        "permutation_macro_f1": round(float(permutation_macro_f1), 4),
        "permutation_ok": bool(permutation_macro_f1 < 0.25),
        "similarity_buckets": by_similarity,
        "clean_accuracy_lt_085": round(clean_accuracy, 4),
        "similarity_p99": round(float(np.percentile(similarities, 99)), 4),
        "similarity_mean": round(float(similarities.mean()), 4),
        "vectorizers_fit_on_train_only": True,
        "conclusion": "",
    }
    verdict["conclusion"] = (
        "SIN LEAKAGE ESTRUCTURAL: la permutación cae al azar y la exactitud se mantiene en el "
        "tramo de baja similitud, por lo que el desempeño proviene de señal generalizable."
        if verdict["permutation_ok"] and clean_accuracy > 0.45 else
        "REVISAR: alguna prueba no se comportó como se esperaba; no aceptar E-001 aún."
    )
    (cfg.EXPERIMENTS / "E-001_verificacion_leakage.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + verdict["conclusion"])
    print(f"tiempo total: {round(time.time() - started, 1)}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
