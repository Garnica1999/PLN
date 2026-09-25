"""E-019 — Modelos densos (Fase 14): MLP sobre features densas y sobre densas+SVD.

Pregunta
    ¿Un modelo no lineal sobre representaciones densas aporta algo frente a los modelos lineales
    dispersos que dominan el proyecto?

Hipótesis
    H17a  Un MLP sobre las 35 features densas no alcanzará a la representación dispersa (B5 con
          LightGBM ya quedó en 0,3567), pero puede aportar **diversidad** al ensamble.
    H17b  Un MLP sobre densas + SVD del espacio de caracteres puede acercarse a los lineales y
          ser un miembro diverso útil.

Nota de entorno (D-007)
    La GPU estaba **ocupada** por una carga ajena al iniciar esta fase (uso 42% sostenido sin PIDs
    de cómputo propios). Por el protocolo de la decisión D-007 el trabajo se ejecutó en **CPU**,
    sin interrumpir procesos ajenos. El evento quedó registrado en ``experiments/gpu_events.log``.

Protocolo
    Mismas 5 particiones agrupadas. La SVD se ajusta **dentro de cada fold** (solo con su train)
    para no filtrar información. Se guardan las matrices OOF para evaluar su aporte al ensamble.

Salida
    ``experiments/e019_modelos_densos.json`` + matrices OOF en ``features/``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, data, preprocessing, textometry  # noqa: E402
from src.config import set_seed  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "e019_modelos_densos.json"
CHAR_CONFIG = {"analyzer": "char_wb", "ngram_range": (3, 6), "min_df": 5,
               "sublinear_tf": True, "max_features": 300_000, "lowercase": True}
MENTION = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2,
           "sublinear_tf": True, "lowercase": True}
DENSE_FEATURES = [name for name in textometry.TEXTOMETRY_FEATURES + textometry.STYLE_FEATURES
                  if name not in {"n_emoji", "emoji_diversity", "has_emoji"}]
N_COMPONENTS = 64


def tokens_of(series: pd.Series, pattern: str) -> pd.Series:
    """Extrae tokens que cumplen un patrón."""
    return preprocessing.collapse_spaces(
        series.str.findall(pattern).apply(lambda values: " ".join(values).lower()))


def fit_mlp(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray,
            hidden: tuple[int, ...], max_iter: int, n_iter_no_change: int,
            encoder: LabelEncoder) -> tuple[np.ndarray, np.ndarray]:
    """Entrena un MLP con etiquetas codificadas como enteros.

    scikit-learn 1.8 falla al calcular ``np.isnan`` sobre predicciones de tipo cadena en la ruta
    de *early stopping*; se codifican las etiquetas y se decodifican al final. El orden de
    ``encoder.classes_`` coincide con el orden canónico de :data:`src.config.CLASSES`.
    """
    model = MLPClassifier(hidden_layer_sizes=hidden, alpha=1e-3, batch_size=512,
                          learning_rate_init=1e-3, max_iter=max_iter, early_stopping=True,
                          n_iter_no_change=n_iter_no_change, random_state=cfg.SEED)
    model.fit(x_train, encoder.transform(y_train))
    encoded_predictions = model.predict(x_valid)
    probabilities = model.predict_proba(x_valid)
    return encoder.inverse_transform(encoded_predictions), probabilities


def main() -> int:
    """Entrena los MLP sobre densas y sobre densas+SVD, guardando OOF."""
    set_seed()
    started = time.time()
    train = data.load_train()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]
    y = train[cfg.TARGET].astype(str).to_numpy()
    texts = train[cfg.TEXT_COL]
    dense = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")[DENSE_FEATURES].to_numpy(float)
    folds = cv.make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}
    oof_dense = np.zeros((len(train), len(cfg.CLASSES)), dtype=np.float32)
    oof_svd = np.zeros((len(train), len(cfg.CLASSES)), dtype=np.float32)
    encoder = LabelEncoder().fit(y)
    assert list(encoder.classes_) == list(cfg.CLASSES), "orden de clases inesperado"

    for fold in folds:
        train_idx, valid_idx = fold["train_idx"], fold["val_idx"]

        # --- Modelo A: MLP sobre las 35 features densas ---
        scaler = StandardScaler().fit(dense[train_idx])
        predictions, probabilities = fit_mlp(scaler.transform(dense[train_idx]), y[train_idx],
                                             scaler.transform(dense[valid_idx]),
                                             hidden=(128, 64), max_iter=40, n_iter_no_change=5,
                                             encoder=encoder)
        macro_dense = f1_score(y[valid_idx], predictions, average="macro")
        oof_dense[valid_idx] = probabilities

        # --- Modelo B: MLP sobre densas + SVD del espacio de caracteres ---
        char = TfidfVectorizer(**CHAR_CONFIG)
        mentions = TfidfVectorizer(**MENTION)
        x_train = hstack([char.fit_transform(texts.iloc[train_idx]),
                          mentions.fit_transform(tokens_of(texts.iloc[train_idx], r"@\w+"))]).tocsr()
        x_valid = hstack([char.transform(texts.iloc[valid_idx]),
                          mentions.transform(tokens_of(texts.iloc[valid_idx], r"@\w+"))]).tocsr()
        svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=cfg.SEED)
        z_train = svd.fit_transform(x_train)
        z_valid = svd.transform(x_valid)
        combined_train = np.hstack([dense[train_idx], z_train])
        combined_valid = np.hstack([dense[valid_idx], z_valid])
        scaler_b = StandardScaler().fit(combined_train)
        predictions_b, probabilities_b = fit_mlp(scaler_b.transform(combined_train), y[train_idx],
                                                 scaler_b.transform(combined_valid),
                                                 hidden=(256, 128), max_iter=60,
                                                 n_iter_no_change=6, encoder=encoder)
        macro_svd = f1_score(y[valid_idx], predictions_b, average="macro")
        oof_svd[valid_idx] = probabilities_b

        for key, macro, extra in (("E-019a", macro_dense, {"variante": "MLP sobre 35 densas"}),
                                  ("E-019b", macro_svd,
                                   {"variante": f"MLP sobre 35 densas + SVD({N_COMPONENTS})"})):
            results.setdefault(key, {"per_fold": [], **extra})
            results[key]["per_fold"].append({
                "fold": fold["fold"], "macro_f1": macro,
                "accuracy": accuracy_score(
                    y[valid_idx], predictions if key == "E-019a" else predictions_b),
            })
        print(f"[fold {fold['fold']}] E-019a (densas) {macro_dense:.4f} | "
              f"E-019b (densas+SVD) {macro_svd:.4f} | "
              f"varianza SVD explicada {svd.explained_variance_ratio_.sum():.3f}", flush=True)
        RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    for key in ("E-019a", "E-019b"):
        frame = pd.DataFrame(results[key]["per_fold"])
        results[key]["macro_f1_mean"] = float(frame["macro_f1"].mean())
        results[key]["macro_f1_std"] = float(frame["macro_f1"].std())
        results[key]["accuracy_mean"] = float(frame["accuracy"].mean())
    results["segundos"] = round(time.time() - started, 1)
    results["entorno"] = "CPU (GPU ocupada por carga ajena; protocolo D-007)"
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    np.save(cfg.FEATURES / "oof_E019a_densas.npy", oof_dense)
    np.save(cfg.FEATURES / "oof_E019b_densas_svd.npy", oof_svd)

    print("\n=== MODELOS DENSOS (Fase 14, CPU) ===")
    for key, label in (("E-019a", "MLP sobre 35 features densas"),
                       ("E-019b", "MLP sobre 35 densas + SVD(64) del espacio de caracteres")):
        print(f"{key} ({label}): macro-F1 {results[key]['macro_f1_mean']:.4f} "
              f"± {results[key]['macro_f1_std']:.4f} | accuracy {results[key]['accuracy_mean']:.4f}")
    print("\nReferencias: B5 densas+LightGBM 0,3567 · mejor individual disperso 0,5658 · "
          "ensamble 0,5784")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
