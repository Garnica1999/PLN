"""Submission v04 — ensamble extendido de 9 fuentes (Fase 20b).

Composición (todas sobre la representación char_wb 3-6 ∪ canal de menciones, salvo donde se indica)
    1. svm           LinearSVC(C=0.2) con vocabulario char 300k
    2. svm700k       LinearSVC(C=0.2) con vocabulario char 700k (min_df=3)
    3. logreg        LogisticRegression(OvR, C=4)
    4. sgd           SGDClassifier(log_loss)
    5. cnb           ComplementNB(alpha=0.3)
    6. hashtags      LinearSVC(C=0.2) + canal de hashtags
    7. ordinal       Ridge sobre el punto medio del rango de edad (señal ordinal)
    8. ordinal_soft  softmax por distancia al punto medio (derivado de 7)
    9. word          LogisticRegression(OvR) sobre word 1-2 (miembro diverso)
    Meta-modelo: StandardScaler + LogisticRegression multinomial sobre las 49 columnas OOF.

Evidencia
    LOFO: **0,5748 ± 0,0048** (Fase 20b) vs 0,5656 de la v03 y 0,5628 del mejor individual.

Protocolo
    Modelos base ajustados con el train completo; meta-modelo entrenado con el OOF de todo el
    train (sin ver el test). Validación estricta del formato; sin envío automático.

Salida
    ``submissions/age_submission_v04.csv`` + ``experiments/submission_v04_resumen.json``.
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
from src import data, preprocessing, submission  # noqa: E402
from src.config import set_seed  # noqa: E402

VERSION = 4
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
ORDINAL_FILES = ["oof_E013_margenes.npy", "oof_E016_margenes.npy", "oof_E-009.npy",
                 "oof_E-011.npy", "oof_E-010.npy", "oof_E014_margenes.npy",
                 "oof_E015_ordinal.npy", "oof_E015_softmax.npy",
                 "oof_E017_probabilidades.npy"]


def tokens_of(series: pd.Series, pattern: str) -> pd.Series:
    """Extrae tokens que cumplen un patrón."""
    return preprocessing.collapse_spaces(
        series.str.findall(pattern).apply(lambda values: " ".join(values).lower()))


def to_scores(raw: np.ndarray, classes, probabilistic: bool) -> np.ndarray:
    """Ordena las columnas al orden canónico y aplica log a las probabilidades."""
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
    """Ajusta el ensamble extendido sobre todo el train y genera la submission validada."""
    set_seed()
    started = time.time()
    train = data.load_train()
    test = data.load_test()
    sample = data.load_sample_submission()
    y = train[cfg.TARGET].astype(str).to_numpy()
    ordinal_target = np.array([CLASS_MIDPOINTS[label] for label in y])

    # --- Representaciones ---
    def representation(frame: pd.DataFrame, char_config: dict):
        char = TfidfVectorizer(**char_config)
        mentions = TfidfVectorizer(**MENTION)
        return hstack([char.fit_transform(frame[cfg.TEXT_COL]),
                       mentions.fit_transform(tokens_of(frame[cfg.TEXT_COL], r"@\w+"))]).tocsr(), \
               char, mentions

    x300_train, char300, mention300 = representation(train, CHAR_300K)
    x300_test = hstack([char300.transform(test[cfg.TEXT_COL]),
                        mention300.transform(tokens_of(test[cfg.TEXT_COL], r"@\w+"))]).tocsr()

    x700_train, char700, mention700 = representation(train, CHAR_700K)
    x700_test = hstack([char700.transform(test[cfg.TEXT_COL]),
                        mention700.transform(tokens_of(test[cfg.TEXT_COL], r"@\w+"))]).tocsr()

    hashtag = TfidfVectorizer(**HASHTAG)
    xh_train = hstack([x300_train,
                       hashtag.fit_transform(tokens_of(train[cfg.TEXT_COL], r"#\w+"))]).tocsr()
    xh_test = hstack([x300_test,
                      hashtag.transform(tokens_of(test[cfg.TEXT_COL], r"#\w+"))]).tocsr()

    word = TfidfVectorizer(**WORD)
    xw_train = word.fit_transform(train[cfg.TEXT_COL])
    xw_test = word.transform(test[cfg.TEXT_COL])
    print(f"representaciones: {x300_train.shape[1]:,} | {x700_train.shape[1]:,} | "
          f"{xh_train.shape[1]:,} | {xw_train.shape[1]:,} ({round(time.time() - started, 1)}s)",
          flush=True)

    # --- Modelos base sobre test ---
    test_scores: list[np.ndarray] = []

    model = LinearSVC(C=0.2, max_iter=3000).fit(x300_train, y)
    test_scores.append(to_scores(model.decision_function(x300_test), model.classes_, False))
    print(f"[base] svm ({round(time.time() - started, 1)}s)", flush=True)

    model = LinearSVC(C=0.2, max_iter=3000).fit(x700_train, y)
    test_scores.append(to_scores(model.decision_function(x700_test), model.classes_, False))
    print(f"[base] svm700k ({round(time.time() - started, 1)}s)", flush=True)

    model = OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=1000, solver="liblinear",
                                                   random_state=cfg.SEED)).fit(x300_train, y)
    test_scores.append(to_scores(model.predict_proba(x300_test), model.classes_, True))
    print(f"[base] logreg ({round(time.time() - started, 1)}s)", flush=True)

    model = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=30, tol=1e-4,
                          random_state=cfg.SEED).fit(x300_train, y)
    test_scores.append(to_scores(model.predict_proba(x300_test), model.classes_, True))

    model = ComplementNB(alpha=0.3).fit(x300_train, y)
    test_scores.append(to_scores(model.predict_proba(x300_test), model.classes_, True))
    print(f"[base] sgd + cnb ({round(time.time() - started, 1)}s)", flush=True)

    model = LinearSVC(C=0.2, max_iter=3000).fit(xh_train, y)
    test_scores.append(to_scores(model.decision_function(xh_test), model.classes_, False))
    print(f"[base] hashtags ({round(time.time() - started, 1)}s)", flush=True)

    ridge = Ridge(alpha=1.0, solver="sparse_cg", max_iter=2000, random_state=cfg.SEED)
    ridge.fit(x300_train, ordinal_target)
    continuous_test = ridge.predict(x300_test)
    test_scores.append(continuous_test.reshape(-1, 1))
    test_scores.append(np.log(np.clip(softmax_ordinal(continuous_test), 1e-12, None)))
    print(f"[base] ordinal ({round(time.time() - started, 1)}s)", flush=True)

    model = OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=1000, solver="liblinear",
                                                   random_state=cfg.SEED)).fit(xw_train, y)
    test_scores.append(to_scores(model.predict_proba(xw_test), model.classes_, True))
    print(f"[base] word ({round(time.time() - started, 1)}s)", flush=True)

    x_meta_test = np.concatenate(test_scores, axis=1)

    # --- Meta-modelo entrenado con el OOF de todo el train ---
    oof_matrices = []
    for index, filename in enumerate(ORDINAL_FILES):
        matrix = np.load(cfg.FEATURES / filename)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        probabilistic = index in (2, 3, 4, 7, 8)  # logreg, sgd, cnb, ordinal_soft, word
        oof_matrices.append(np.log(np.clip(matrix, 1e-12, None)) if probabilistic else matrix)
    x_meta_train = np.concatenate(oof_matrices, axis=1)
    assert x_meta_train.shape[1] == x_meta_test.shape[1], "desalineación de features del meta-modelo"

    meta = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs", random_state=cfg.SEED)),
    ])
    meta.fit(x_meta_train, y)
    predictions = meta.predict(x_meta_test)

    candidate = pd.DataFrame({cfg.ID_COL: test[cfg.ID_COL].to_numpy(), cfg.TARGET: predictions})
    checks = submission.validate_submission(candidate, test[cfg.ID_COL], sample)
    path = submission.write_submission(candidate, VERSION, test[cfg.ID_COL], sample)

    summary = {
        "version": VERSION,
        "modelo": "Ensamble extendido de 9 fuentes -> meta LogReg (StandardScaler)",
        "cv_macro_f1_lofo": 0.5748,
        "cv_macro_f1_std": 0.0048,
        "referencia_v03": {"cv": 0.5656, "lb": 0.58081},
        "checks": checks,
        "distribucion_predicha": candidate[cfg.TARGET].value_counts()
                                 .reindex(cfg.CLASSES).fillna(0).astype(int).to_dict(),
        "segundos": round(time.time() - started, 1),
        "archivo": path,
        "recordatorio": ("Subida manual. Con la brecha CV<->LB estable (≈ +0,015), el LB esperado "
                         "ronda 0,59."),
    }
    (cfg.EXPERIMENTS / "submission_v04_resumen.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("checks", "distribucion_predicha", "segundos")},
                     indent=2, ensure_ascii=False))
    print(f"\narchivo listo: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
