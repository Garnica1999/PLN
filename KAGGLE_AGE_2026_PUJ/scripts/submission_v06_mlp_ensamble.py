"""Submission v06 — ensamble con el MLP denso como miembro diverso (E-020).

Composición
    9 modelos base (idénticos a la v05) + **MLP sobre 35 features densas + SVD(64) del espacio de
    caracteres** (entrenado en CPU: la GPU estaba ocupada, protocolo D-007) + meta-modelo
    enriquecido con las 35 features densas (90 features en total).

Evidencia
    LOFO: **0,5851 ± 0,0042** (E-020) vs 0,5784 de la v05 (+0,0066). El MLP rinde 0,4071 en
    solitario, pero sus errores son lo bastante distintos para que el meta-modelo los aproveche:
    es el mayor aporte individual desde el canal de menciones.

Salida
    ``submissions/age_submission_v06.csv`` + ``experiments/submission_v06_resumen.json``.
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
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import data, preprocessing, submission, textometry  # noqa: E402
from src.config import set_seed  # noqa: E402

VERSION = 6
CHAR_300K = {"analyzer": "char_wb", "ngram_range": (3, 6), "min_df": 5,
             "sublinear_tf": True, "max_features": 300_000, "lowercase": True}
CHAR_700K = {"analyzer": "char_wb", "ngram_range": (3, 6), "min_df": 3,
             "sublinear_tf": True, "max_features": 700_000, "lowercase": True}
MENTION = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2,
           "sublinear_tf": True, "lowercase": True}
HASHTAG = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2,
           "sublinear_tf": True, "lowercase": True}
WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 3,
        "sublinear_tf": True, "max_features": 300_000, "lowercase": True}
CLASS_MIDPOINTS = {"13-17": 15.0, "18-24": 21.0, "25-34": 29.5, "35-49": 42.0,
                   "50-64": 57.0, "65-xx": 70.0}
OOF_FILES = [("oof_E013_margenes.npy", False), ("oof_E016_margenes.npy", False),
             ("oof_E-009.npy", True), ("oof_E-011.npy", True), ("oof_E-010.npy", True),
             ("oof_E014_margenes.npy", False), ("oof_E015_ordinal.npy", False),
             ("oof_E015_softmax.npy", True), ("oof_E017_probabilidades.npy", True),
             ("oof_E019b_densas_svd.npy", True)]
DENSE_FEATURES = [name for name in textometry.TEXTOMETRY_FEATURES + textometry.STYLE_FEATURES
                  if name not in {"n_emoji", "emoji_diversity", "has_emoji"}]
N_COMPONENTS = 64


def tokens_of(series: pd.Series, pattern: str) -> pd.Series:
    """Extrae tokens que cumplen un patrón."""
    return preprocessing.collapse_spaces(
        series.str.findall(pattern).apply(lambda values: " ".join(values).lower()))


def to_scores(raw: np.ndarray, classes, probabilistic: bool) -> np.ndarray:
    """Ordena columnas al orden canónico y aplica log a las probabilidades."""
    order = [list(classes).index(label) for label in cfg.CLASSES]
    scores = raw[:, order]
    return np.log(np.clip(scores, 1e-12, None)) if probabilistic else scores


def softmax_ordinal(continuous: np.ndarray) -> np.ndarray:
    """Softmax por distancia al punto medio (idéntico a E-015)."""
    midpoints = np.array([CLASS_MIDPOINTS[label] for label in cfg.CLASSES])
    distances = np.abs(continuous[:, None] - midpoints)
    soft = np.exp(-distances / 5.0)
    return soft / soft.sum(axis=1, keepdims=True)


def main() -> int:
    """Ajusta el ensamble con MLP sobre todo el train y genera la submission validada."""
    set_seed()
    started = time.time()
    train = data.load_train()
    test = data.load_test()
    sample = data.load_sample_submission()
    y = train[cfg.TARGET].astype(str).to_numpy()
    ordinal_target = np.array([CLASS_MIDPOINTS[label] for label in y])
    encoder = LabelEncoder().fit(y)

    char = TfidfVectorizer(**CHAR_300K)
    mentions = TfidfVectorizer(**MENTION)
    x300_train = hstack([char.fit_transform(train[cfg.TEXT_COL]),
                         mentions.fit_transform(tokens_of(train[cfg.TEXT_COL], r"@\w+"))]).tocsr()
    x300_test = hstack([char.transform(test[cfg.TEXT_COL]),
                        mentions.transform(tokens_of(test[cfg.TEXT_COL], r"@\w+"))]).tocsr()
    char700 = TfidfVectorizer(**CHAR_700K)
    mention700 = TfidfVectorizer(**MENTION)
    x700_train = hstack([char700.fit_transform(train[cfg.TEXT_COL]),
                         mention700.fit_transform(tokens_of(train[cfg.TEXT_COL], r"@\w+"))]).tocsr()
    x700_test = hstack([char700.transform(test[cfg.TEXT_COL]),
                        mention700.transform(tokens_of(test[cfg.TEXT_COL], r"@\w+"))]).tocsr()
    hashtag = TfidfVectorizer(**HASHTAG)
    xh_train = hstack([x300_train, hashtag.fit_transform(tokens_of(train[cfg.TEXT_COL], r"#\w+"))]).tocsr()
    xh_test = hstack([x300_test, hashtag.transform(tokens_of(test[cfg.TEXT_COL], r"#\w+"))]).tocsr()
    word = TfidfVectorizer(**WORD)
    xw_train, xw_test = word.fit_transform(train[cfg.TEXT_COL]), word.transform(test[cfg.TEXT_COL])
    dense_train = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")[DENSE_FEATURES].to_numpy(float)
    dense_test = pd.read_parquet(cfg.FEATURES / "dense_test.parquet")[DENSE_FEATURES].to_numpy(float)
    print(f"representaciones listas ({round(time.time() - started, 1)}s)", flush=True)

    test_scores: list[np.ndarray] = []
    model = LinearSVC(C=0.2, max_iter=3000).fit(x300_train, y)
    test_scores.append(to_scores(model.decision_function(x300_test), model.classes_, False))
    model = LinearSVC(C=0.2, max_iter=3000).fit(x700_train, y)
    test_scores.append(to_scores(model.decision_function(x700_test), model.classes_, False))
    print(f"[base] svm + svm700k ({round(time.time() - started, 1)}s)", flush=True)
    model = OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=1000, solver="liblinear",
                                                   random_state=cfg.SEED)).fit(x300_train, y)
    test_scores.append(to_scores(model.predict_proba(x300_test), model.classes_, True))
    print(f"[base] logreg ({round(time.time() - started, 1)}s)", flush=True)
    model = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=30, tol=1e-4,
                          random_state=cfg.SEED).fit(x300_train, y)
    test_scores.append(to_scores(model.predict_proba(x300_test), model.classes_, True))
    model = ComplementNB(alpha=0.3).fit(x300_train, y)
    test_scores.append(to_scores(model.predict_proba(x300_test), model.classes_, True))
    model = LinearSVC(C=0.2, max_iter=3000).fit(xh_train, y)
    test_scores.append(to_scores(model.decision_function(xh_test), model.classes_, False))
    print(f"[base] sgd + cnb + hashtags ({round(time.time() - started, 1)}s)", flush=True)
    ridge = Ridge(alpha=1.0, solver="sparse_cg", max_iter=2000, random_state=cfg.SEED)
    ridge.fit(x300_train, ordinal_target)
    continuous_test = ridge.predict(x300_test)
    test_scores.append(continuous_test.reshape(-1, 1))
    test_scores.append(np.log(np.clip(softmax_ordinal(continuous_test), 1e-12, None)))
    model = OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=1000, solver="liblinear",
                                                   random_state=cfg.SEED)).fit(xw_train, y)
    test_scores.append(to_scores(model.predict_proba(xw_test), model.classes_, True))
    print(f"[base] ordinal + word ({round(time.time() - started, 1)}s)", flush=True)

    # --- MLP sobre densas + SVD (miembro diverso) ---
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=cfg.SEED)
    z_train = svd.fit_transform(x300_train)
    z_test = svd.transform(x300_test)
    combined_train = np.hstack([dense_train, z_train])
    combined_test = np.hstack([dense_test, z_test])
    scaler = StandardScaler().fit(combined_train)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, batch_size=512,
                        learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                        n_iter_no_change=6, random_state=cfg.SEED)
    mlp.fit(scaler.transform(combined_train), encoder.transform(y))
    test_scores.append(np.log(np.clip(mlp.predict_proba(scaler.transform(combined_test)), 1e-12, None)))
    print(f"[base] mlp densas+svd ({round(time.time() - started, 1)}s)", flush=True)

    x_meta_test = np.concatenate(test_scores + [dense_test], axis=1)

    oof_matrices = []
    for filename, probabilistic in OOF_FILES:
        matrix = np.load(cfg.FEATURES / filename)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        oof_matrices.append(np.log(np.clip(matrix, 1e-12, None)) if probabilistic else matrix)
    x_meta_train = np.concatenate(oof_matrices + [dense_train], axis=1)
    assert x_meta_train.shape[1] == x_meta_test.shape[1], "desalineación del meta-modelo"

    meta = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                random_state=cfg.SEED))])
    meta.fit(x_meta_train, y)
    predictions = meta.predict(x_meta_test)

    candidate = pd.DataFrame({cfg.ID_COL: test[cfg.ID_COL].to_numpy(), cfg.TARGET: predictions})
    checks = submission.validate_submission(candidate, test[cfg.ID_COL], sample)
    path = submission.write_submission(candidate, VERSION, test[cfg.ID_COL], sample)

    summary = {
        "version": VERSION,
        "modelo": "Ensamble de 10 fuentes (incluye MLP densas+SVD) + meta enriquecido",
        "cv_macro_f1_lofo": 0.5851,
        "cv_macro_f1_std": 0.0042,
        "referencia_v05": {"cv": 0.5784, "lb": 0.59518},
        "entorno_mlp": "CPU (GPU ocupada por carga ajena; protocolo D-007)",
        "checks": checks,
        "distribucion_predicha": candidate[cfg.TARGET].value_counts()
                                 .reindex(cfg.CLASSES).fillna(0).astype(int).to_dict(),
        "segundos": round(time.time() - started, 1),
        "archivo": path,
        "recordatorio": "Subida manual. Con brecha ≈ +0,016 el LB esperado ronda 0,60.",
    }
    (cfg.EXPERIMENTS / "submission_v06_resumen.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("checks", "distribucion_predicha", "segundos")},
                     indent=2, ensure_ascii=False))
    print(f"\narchivo listo: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
