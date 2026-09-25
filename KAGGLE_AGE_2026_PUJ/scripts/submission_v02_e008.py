"""Submission v02 — representación ganadora E-008 (char_wb 3-6 + canal de menciones).

Justificación (evidencia de la Etapa B)
    - char_wb 3-6 supera a word n-grams por +0,0442 de macro-F1 (5/5 folds).
    - Un canal TF-IDF dedicado a los tokens de mención aporta +0,0144 sobre char solo (5/5).
    - El texto se deja **crudo**: normalizar (quitar menciones) cuesta hasta −0,1251.

Protocolo
    - Vectorizadores ajustados solo con el train completo; el test solo se transforma.
    - Validación estricta del formato contra ``age_submission.csv``.
    - No se envía a Kaggle: subida manual de Carlos.

Salida
    ``submissions/age_submission_v02.csv`` + ``experiments/submission_v02_resumen.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import data, preprocessing, submission  # noqa: E402
from src.config import set_seed  # noqa: E402

VERSION = 2
CHAR_CONFIG = {"analyzer": "char_wb", "ngram_range": (3, 6), "min_df": 5,
               "sublinear_tf": True, "max_features": 300_000, "lowercase": True}
MENTION_CONFIG = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2,
                  "sublinear_tf": True, "lowercase": True}
MODEL_CONFIG = {"C": 1.0, "max_iter": 3000}


def mention_tokens(series: pd.Series) -> pd.Series:
    """Extrae solo los tokens de mención (canal dedicado de E-008)."""
    return preprocessing.collapse_spaces(
        series.str.findall(r"@\w+").apply(lambda values: " ".join(values).lower()))


def main() -> int:
    """Ajusta E-008 sobre todo el train y genera la submission validada."""
    set_seed()
    started = time.time()
    train = data.load_train()
    test = data.load_test()
    sample = data.load_sample_submission()

    char = TfidfVectorizer(**CHAR_CONFIG)
    mentions = TfidfVectorizer(**MENTION_CONFIG)

    x_train = hstack([char.fit_transform(train[cfg.TEXT_COL]),
                      mentions.fit_transform(mention_tokens(train[cfg.TEXT_COL]))]).tocsr()
    x_test = hstack([char.transform(test[cfg.TEXT_COL]),
                     mentions.transform(mention_tokens(test[cfg.TEXT_COL]))]).tocsr()
    print(f"train {x_train.shape} | test {x_test.shape}", flush=True)

    model = LinearSVC(**MODEL_CONFIG)
    model.fit(x_train, train[cfg.TARGET].astype(str).to_numpy())
    predictions = model.predict(x_test)

    candidate = pd.DataFrame({cfg.ID_COL: test[cfg.ID_COL].to_numpy(), cfg.TARGET: predictions})
    checks = submission.validate_submission(candidate, test[cfg.ID_COL], sample)
    path = submission.write_submission(candidate, VERSION, test[cfg.ID_COL], sample)

    summary = {
        "version": VERSION,
        "modelo": "E-008 (TF-IDF char_wb 3-6 + canal de menciones) + LinearSVC(C=1)",
        "cv_macro_f1": 0.5557,
        "cv_macro_f1_std": 0.0052,
        "cv_weighted_f1": 0.5548,
        "checks": checks,
        "distribucion_predicha": candidate[cfg.TARGET].value_counts()
                                 .reindex(cfg.CLASSES).fillna(0).astype(int).to_dict(),
        "segundos": round(time.time() - started, 1),
        "archivo": path,
        "recordatorio": ("Subida manual. Comparar el score contra 0,5557 (CV agrupado) y contra "
                         "el 0,55235 de v01; la brecha esperada CV↔LB es ≈ +0,018 por los "
                         "near-duplicados del test."),
    }
    (cfg.EXPERIMENTS / "submission_v02_resumen.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: summary[k] for k in ("checks", "distribucion_predicha", "segundos")},
                     indent=2, ensure_ascii=False))
    print(f"\narchivo listo: {path}")
    print("NO se envió a Kaggle: súbelo manualmente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
