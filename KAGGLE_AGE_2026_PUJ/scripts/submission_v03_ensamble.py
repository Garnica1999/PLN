"""Submission v03 — ensamble por stacking (Fase 20).

Composición
    4 modelos base sobre la representación ganadora (char_wb 3-6 ∪ canal de menciones):
    LinearSVC(C=0.2), LogisticRegression(OvR, C=4), SGD(log_loss) y ComplementNB.
    Meta-modelo: LogisticRegression multinomial entrenado sobre las predicciones out-of-fold
    (features = 4 modelos × 6 clases = 24 columnas).

Evidencia
    Stacking con leave-one-fold-out: **0,5656** vs 0,5628 del mejor individual, ganando en 5/5
    folds (`experiments/fase20_ensamble.json`).

Protocolo
    - Los 4 modelos base se ajustan con el train completo; el meta-modelo se entrena con el OOF
      de todo el train (cobertura completa, sin ver el test).
    - Validación estricta del formato; **sin envío automático** (subida manual).

Salida
    ``submissions/age_submission_v03.csv`` + ``experiments/submission_v03_resumen.json``.
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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import data, preprocessing, submission  # noqa: E402
from src.config import set_seed  # noqa: E402

VERSION = 3
CHAR_CONFIG = {"analyzer": "char_wb", "ngram_range": (3, 6), "min_df": 5,
               "sublinear_tf": True, "max_features": 300_000, "lowercase": True}
MENTION_CONFIG = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2,
                  "sublinear_tf": True, "lowercase": True}
MODELS = {
    "svm": lambda: LinearSVC(C=0.2, max_iter=3000),
    "logreg": lambda: OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=1000,
                                                            solver="liblinear",
                                                            random_state=cfg.SEED)),
    "sgd": lambda: SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=30, tol=1e-4,
                                 random_state=cfg.SEED),
    "cnb": lambda: ComplementNB(alpha=0.3),
}
PROBABILISTIC = {"logreg", "sgd", "cnb"}


def mention_tokens(series: pd.Series) -> pd.Series:
    """Tokens de mención (canal dedicado)."""
    return preprocessing.collapse_spaces(
        series.str.findall(r"@\w+").apply(lambda values: " ".join(values).lower()))


def canonical_scores(model, matrix, probabilistic: bool) -> np.ndarray:
    """Devuelve la matriz de scores (márgenes o log-probabilidades) en el orden canónico."""
    raw = model.predict_proba(matrix) if probabilistic else model.decision_function(matrix)
    column_order = [list(model.classes_).index(label) for label in cfg.CLASSES]
    scores = raw[:, column_order]
    if probabilistic:
        return np.log(np.clip(scores, 1e-12, None))
    return scores


def main() -> int:
    """Ajusta el ensamble sobre todo el train y genera la submission validada."""
    set_seed()
    started = time.time()
    train = data.load_train()
    test = data.load_test()
    sample = data.load_sample_submission()
    y = train[cfg.TARGET].astype(str).to_numpy()

    char = TfidfVectorizer(**CHAR_CONFIG)
    mentions = TfidfVectorizer(**MENTION_CONFIG)
    x_train = hstack([char.fit_transform(train[cfg.TEXT_COL]),
                      mentions.fit_transform(mention_tokens(train[cfg.TEXT_COL]))]).tocsr()
    x_test = hstack([char.transform(test[cfg.TEXT_COL]),
                     mentions.transform(mention_tokens(test[cfg.TEXT_COL]))]).tocsr()
    print(f"train {x_train.shape} | test {x_test.shape}", flush=True)

    test_features = []
    for name, factory in MODELS.items():
        model = factory()
        model.fit(x_train, y)
        test_features.append(canonical_scores(model, x_test, name in PROBABILISTIC))
        print(f"[base] {name} ajustado ({round(time.time() - started, 1)}s)", flush=True)
    x_meta_test = np.concatenate(test_features, axis=1)

    # Meta-modelo entrenado con el OOF de todo el train.
    oof_files = ["oof_E013_margenes.npy", "oof_E-009.npy", "oof_E-011.npy", "oof_E-010.npy"]
    oof_matrices = []
    for name in oof_files:
        matrix = np.load(cfg.FEATURES / name)
        oof_matrices.append(np.log(np.clip(matrix, 1e-12, None)) if name != oof_files[0] else matrix)
    x_meta_train = np.concatenate(oof_matrices, axis=1)
    meta = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=cfg.SEED)
    meta.fit(x_meta_train, y)
    predictions = meta.predict(x_meta_test)

    candidate = pd.DataFrame({cfg.ID_COL: test[cfg.ID_COL].to_numpy(), cfg.TARGET: predictions})
    checks = submission.validate_submission(candidate, test[cfg.ID_COL], sample)
    path = submission.write_submission(candidate, VERSION, test[cfg.ID_COL], sample)

    summary = {
        "version": VERSION,
        "modelo": "Stacking: LinearSVC(C=0.2) + LogReg(OvR) + SGD + ComplementNB -> LogReg meta",
        "cv_macro_f1_lofo": 0.5656,
        "cv_mejor_individual": 0.5628,
        "checks": checks,
        "distribucion_predicha": candidate[cfg.TARGET].value_counts()
                                 .reindex(cfg.CLASSES).fillna(0).astype(int).to_dict(),
        "segundos": round(time.time() - started, 1),
        "archivo": path,
        "recordatorio": ("Subida manual. Referencia: v02 = 0,57018 en el LB con CV 0,5557. "
                         "Este ensamble tiene CV 0,5656, por lo que se espera un LB superior."),
    }
    (cfg.EXPERIMENTS / "submission_v03_resumen.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("checks", "distribucion_predicha", "segundos")},
                     indent=2, ensure_ascii=False))
    print(f"\narchivo listo: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
