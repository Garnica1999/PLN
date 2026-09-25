"""Submission v05 — ensamble de 9 fuentes con meta-modelo enriquecido (E-018).

Composición
    9 modelos base (idénticos a la v04) + meta-modelo que recibe, además de los 49 scores,
    las **35 features densas** (longitud, retuit, menciones, URLs, textometría y estilo).
    Esto permite condicionar la combinación según el tipo de tweet, atacando el sesgo de error
    por longitud detectado en la Fase 17.

Evidencia
    LOFO: **0,5784 ± 0,0044** (E-018) vs 0,5748 de la v04 (+0,0036) y 0,5628 del mejor individual.

Salida
    ``submissions/age_submission_v05.csv`` + ``experiments/submission_v05_resumen.json``.
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
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import data, preprocessing, submission, textometry  # noqa: E402
from src.config import set_seed  # noqa: E402

VERSION = 5
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
             ("oof_E015_softmax.npy", True), ("oof_E017_probabilidades.npy", True)]
DENSE_FEATURES = [name for name in textometry.TEXTOMETRY_FEATURES + textometry.STYLE_FEATURES
                  if name not in {"n_emoji", "emoji_diversity", "has_emoji"}]


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
    """Ajusta el ensamble enriquecido sobre todo el train y genera la submission validada."""
    set_seed()
    started = time.time()
    train = data.load_train()
    test = data.load_test()
    sample = data.load_sample_submission()
    y = train[cfg.TARGET].astype(str).to_numpy()
    ordinal_target = np.array([CLASS_MIDPOINTS[label] for label in y])

    def representation(frame: pd.DataFrame, char_config: dict):
        char = TfidfVectorizer(**char_config)
        mentions = TfidfVectorizer(**MENTION)
        train_matrix = hstack([char.fit_transform(frame[cfg.TEXT_COL]),
                               mentions.fit_transform(tokens_of(frame[cfg.TEXT_COL], r"@\w+"))]).tocsr()
        return train_matrix, char, mentions

    x300_train, char300, mention300 = representation(train, CHAR_300K)
    x300_test = hstack([char300.transform(test[cfg.TEXT_COL]),
                        mention300.transform(tokens_of(test[cfg.TEXT_COL], r"@\w+"))]).tocsr()
    x700_train, char700, mention700 = representation(train, CHAR_700K)
    x700_test = hstack([char700.transform(test[cfg.TEXT_COL]),
                        mention700.transform(tokens_of(test[cfg.TEXT_COL], r"@\w+"))]).tocsr()
    hashtag = TfidfVectorizer(**HASHTAG)
    xh_train = hstack([x300_train, hashtag.fit_transform(tokens_of(train[cfg.TEXT_COL], r"#\w+"))]).tocsr()
    xh_test = hstack([x300_test, hashtag.transform(tokens_of(test[cfg.TEXT_COL], r"#\w+"))]).tocsr()
    word = TfidfVectorizer(**WORD)
    xw_train, xw_test = word.fit_transform(train[cfg.TEXT_COL]), word.transform(test[cfg.TEXT_COL])
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

    dense_test = pd.read_parquet(cfg.FEATURES / "dense_test.parquet")[DENSE_FEATURES].to_numpy(float)
    x_meta_test = np.concatenate(test_scores + [dense_test], axis=1)

    oof_matrices = []
    for filename, probabilistic in OOF_FILES:
        matrix = np.load(cfg.FEATURES / filename)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        oof_matrices.append(np.log(np.clip(matrix, 1e-12, None)) if probabilistic else matrix)
    dense_train = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")[DENSE_FEATURES].to_numpy(float)
    x_meta_train = np.concatenate(oof_matrices + [dense_train], axis=1)
    assert x_meta_train.shape[1] == x_meta_test.shape[1], "desalineación de features del meta-modelo"

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
        "modelo": "Ensamble de 9 fuentes + meta-modelo enriquecido con 35 features densas",
        "cv_macro_f1_lofo": 0.5784,
        "cv_macro_f1_std": 0.0044,
        "referencia_v04": {"cv": 0.5748, "lb": 0.59178},
        "checks": checks,
        "distribucion_predicha": candidate[cfg.TARGET].value_counts()
                                 .reindex(cfg.CLASSES).fillna(0).astype(int).to_dict(),
        "segundos": round(time.time() - started, 1),
        "archivo": path,
        "recordatorio": "Subida manual. Con brecha ≈ +0,017 el LB esperado ronda 0,595.",
    }
    (cfg.EXPERIMENTS / "submission_v05_resumen.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("checks", "distribucion_predicha", "segundos")},
                     indent=2, ensure_ascii=False))
    print(f"\narchivo listo: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
