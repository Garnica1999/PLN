"""ADV-06 — Submission v10: candidato con NB-SVM como miembro adicional.

Composición
    v09 (`AGE_ADVANCED_V01`, 267 features del meta) + **márgenes NB-SVM** (6 features), es decir
    273 features del meta. El NB-SVM (Wang & Manning 2012, uno-vs-rest con log-count ratio) rinde
    0,5641 en CV, comparable al mejor modelo individual, y aporta **+0,0007** al ensamble
    (4/5 folds) como miembro diverso.

Decisión de envío
    Con 9 de ~10 envíos puntuados ya usados, este archivo se genera y queda **listo pero sin
    subir**, a la espera de que Carlos decida si gasta el envío restante en una mejora de +0,0007
    (dentro del ruido) o lo reserva para una mejora mayor o para el cierre de la competencia.

Salida: ``submissions/age_submission_v10.csv`` + ``experiments/submission_v10_resumen.json``.
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import advanced_features as af  # noqa: E402
from src import config as cfg  # noqa: E402
from src import data, pipeline, preprocessing, submission, textometry  # noqa: E402
from src.config import set_seed  # noqa: E402

VERSION = 10
CHAR_CONFIG = {"analyzer": "char_wb", "ngram_range": (3, 6), "min_df": 5,
               "sublinear_tf": True, "max_features": 300_000, "lowercase": True}
MENTION_CONFIG = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2,
                  "sublinear_tf": True, "lowercase": True}
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


def main() -> int:
    """Ajusta el candidato v10 (v09 + NB-SVM) y genera la submission validada."""
    set_seed()
    started = time.time()
    train = data.load_train()
    test = data.load_test()
    sample = data.load_sample_submission()
    y = train[cfg.TARGET].astype(str).to_numpy()

    dense_train = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")[DENSE_FEATURES].to_numpy(float)
    dense_test = pd.read_parquet(cfg.FEATURES / "dense_test.parquet")[DENSE_FEATURES].to_numpy(float)
    structured_train = pd.read_parquet(cfg.FEATURES / "age_structured_features.parquet").to_numpy(float)
    structured_test = pd.read_parquet(cfg.FEATURES / "age_structured_features_test.parquet").to_numpy(float)
    supervised_train = np.load(cfg.FEATURES / "age_supervised_features.npy")

    # ---------------------------------------------------------------- NB-SVM sobre test
    print("[1/3] NB-SVM sobre todo el train...", flush=True)
    char = TfidfVectorizer(**CHAR_CONFIG)
    mentions = TfidfVectorizer(**MENTION_CONFIG)
    x_train = sp.hstack([char.fit_transform(train[cfg.TEXT_COL]),
                         mentions.fit_transform(tokens_of(train[cfg.TEXT_COL], r"@\w+"))]).tocsr()
    x_test = sp.hstack([char.transform(test[cfg.TEXT_COL]),
                        mentions.transform(tokens_of(test[cfg.TEXT_COL], r"@\w+"))]).tocsr()
    nb_margins = np.zeros((len(test), len(cfg.CLASSES)), dtype=np.float32)
    for position, label in enumerate(cfg.CLASSES):
        binary_target = (y == label).astype(int)
        ratio = nb_log_count_ratio(x_train, binary_target)
        model = LinearSVC(C=0.2, max_iter=3000)
        model.fit(x_train.multiply(ratio).tocsr(), binary_target)
        nb_margins[:, position] = model.decision_function(x_test.multiply(ratio).tocsr())
        print(f"    NB-SVM clase {label} ({round(time.time() - started, 1)}s)", flush=True)

    # ---------------------------------------------------------------- features supervisadas de test
    print("[2/3] features supervisadas de test...", flush=True)
    supervised_test = pd.concat([
        af.supervised_entity_features(train[cfg.TEXT_COL], y, test[cfg.TEXT_COL], "mention"),
        af.supervised_entity_features(train[cfg.TEXT_COL], y, test[cfg.TEXT_COL], "hashtag"),
        af.supervised_entity_features(train[cfg.TEXT_COL], y, test[cfg.TEXT_COL], "rt_source"),
        af.lexical_rarity_features(train[cfg.TEXT_COL], test[cfg.TEXT_COL]),
        af.class_lexicon_scores(train[cfg.TEXT_COL], y, test[cfg.TEXT_COL]),
    ], axis=1).to_numpy(float)

    # ---------------------------------------------------------------- modelos base y meta
    print("[3/3] modelos base + meta-modelo...", flush=True)
    representations = pipeline.build_representations(train, test)
    test_scores, _ = pipeline.fit_base_models(representations, y, dense_train, dense_test,
                                              verbose=True)
    x_meta_test = np.concatenate(test_scores + [dense_test, structured_test, supervised_test,
                                                nb_margins], axis=1)

    oof = []
    for filename, probabilistic in pipeline.OOF_FILES:
        matrix = np.load(cfg.FEATURES / filename)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        oof.append(np.log(np.clip(matrix, 1e-12, None)) if probabilistic else matrix)
    nb_oof = np.load(cfg.FEATURES / "oof_ADV05_nbsvm.npy")
    x_meta_train = np.concatenate([np.concatenate(oof, axis=1), dense_train, structured_train,
                                   supervised_train, nb_oof], axis=1)
    assert x_meta_train.shape[1] == x_meta_test.shape[1], (
        f"desalineación: {x_meta_train.shape} vs {x_meta_test.shape}")

    meta = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                random_state=cfg.SEED))])
    meta.fit(x_meta_train, y)
    predictions = meta.predict(x_meta_test)

    candidate = pd.DataFrame({cfg.ID_COL: test[cfg.ID_COL].to_numpy(), cfg.TARGET: predictions})
    checks = submission.validate_submission(candidate, test[cfg.ID_COL], sample)
    path = cfg.SUBMISSIONS / f"age_submission_v{VERSION}.csv"
    candidate.to_csv(path, index=False)

    summary = {
        "version": VERSION,
        "modelo": "v09 (AGE_ADVANCED_V01) + NB-SVM como miembro (273 features del meta)",
        "cv_macro_f1_lofo": 0.5904,
        "cv_std": 0.0047,
        "referencia_v09": {"cv": 0.5897, "holdout": 0.5966, "lb": 0.60807},
        "aporte_nbsvm": 0.0007,
        "checks": checks,
        "distribucion_predicha": candidate[cfg.TARGET].value_counts().reindex(cfg.CLASSES)
                                 .fillna(0).astype(int).to_dict(),
        "segundos": round(time.time() - started, 1),
        "archivo": str(path),
        "decision_de_envio": ("NO subir por ahora: +0,0007 está dentro del ruido y quedan ~1 "
                              "envíos puntuados. Reservar para una mejora mayor o para el cierre."),
    }
    (cfg.EXPERIMENTS / f"submission_v{VERSION:02d}_resumen.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: summary[k] for k in ("checks", "segundos", "decision_de_envio")},
                     indent=2, ensure_ascii=False))
    print(f"\narchivo listo (sin subir): {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
