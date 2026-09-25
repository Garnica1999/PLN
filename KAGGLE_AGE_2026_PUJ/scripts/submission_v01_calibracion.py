"""Submission v01 (calibración) — E-001 ajustado sobre todo el train.

Objetivo
    Producir el primer archivo de envío con la baseline verificada de E-001 para:
    (1) identificar empíricamente si la métrica oficial es macro o weighted F1 (D-002);
    (2) medir la brecha entre el CV agrupado y el leaderboard.

Protocolo
    - Vectorizadores ajustados **solo con train completo** (el test jamás participa del ajuste).
    - Predicción sobre test y validación estricta del formato contra ``age_submission.csv``.
    - No se envía nada a Kaggle: el archivo queda listo para subida manual.

Salida
    ``submissions/age_submission_v01.csv`` + ``experiments/submission_v01_resumen.json``.
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
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import data, submission  # noqa: E402
from src.config import set_seed  # noqa: E402

WORD_CONFIG = {"ngram_range": (1, 2), "min_df": 3, "sublinear_tf": True,
               "max_features": 300_000, "lowercase": True}
CHAR_CONFIG = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 5,
               "sublinear_tf": True, "max_features": 300_000, "lowercase": True}
MODEL_CONFIG = {"C": 1.0, "max_iter": 3000}
VERSION = 1


def main() -> int:
    """Ajusta el modelo final de la baseline y genera la submission validada."""
    set_seed()
    started = time.time()
    train = data.load_train()
    test = data.load_test()
    sample = data.load_sample_submission()

    word = TfidfVectorizer(**WORD_CONFIG)
    char = TfidfVectorizer(**CHAR_CONFIG)
    x_train = hstack([word.fit_transform(train[cfg.TEXT_COL]),
                      char.fit_transform(train[cfg.TEXT_COL])]).tocsr()
    x_test = hstack([word.transform(test[cfg.TEXT_COL]),
                     char.transform(test[cfg.TEXT_COL])]).tocsr()
    print(f"train {x_train.shape} | test {x_test.shape}", flush=True)

    model = LinearSVC(**MODEL_CONFIG)
    model.fit(x_train, train[cfg.TARGET].astype(str).to_numpy())
    predictions = model.predict(x_test)

    candidate = pd.DataFrame({cfg.ID_COL: test[cfg.ID_COL].to_numpy(), cfg.TARGET: predictions})
    checks = submission.validate_submission(candidate, test[cfg.ID_COL], sample)
    path = submission.write_submission(candidate, VERSION, test[cfg.ID_COL], sample)

    distribution = candidate[cfg.TARGET].value_counts().reindex(cfg.CLASSES).fillna(0).astype(int)
    summary = {
        "version": VERSION,
        "modelo": "E-001 (TF-IDF word 1-2 + char_wb 3-5 + LinearSVC)",
        "checks": checks,
        "distribucion_predicha": distribution.to_dict(),
        "distribucion_train": train[cfg.TARGET].value_counts().reindex(cfg.CLASSES).to_dict(),
        "segundos": round(time.time() - started, 1),
        "archivo": path,
        "recordatorio": ("Subida manual. Al recibir el score público, comparar contra "
                         "macro-F1=0,5344 y weighted-F1=0,5335 del CV agrupado para identificar "
                         "la métrica oficial (D-002) y la brecha CV↔LB."),
    }
    (cfg.EXPERIMENTS / "submission_v01_resumen.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: summary[k] for k in ("checks", "distribucion_predicha", "segundos")},
                     indent=2, ensure_ascii=False))
    print(f"\narchivo listo: {path}")
    print("NO se envió a Kaggle: súbelo manualmente y comparte el score para calibrar la métrica.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
