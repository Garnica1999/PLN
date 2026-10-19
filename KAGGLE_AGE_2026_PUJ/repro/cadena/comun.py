"""Utilidades comunes de la cadena AGE (porte fiel del proyecto KAGGLE_AGE_2026_PUJ).

Regla de oro: cada funcion es un PORTE LITERAL del original citado; no se mejora ni se
reinterpreta nada. ASCII puro. Sin efectos en import.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["13-17", "18-24", "25-34", "35-49", "50-64", "65-xx"]
SEED = 42

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

ADJACENT_PAIRS = [("13-17", "18-24"), ("18-24", "25-34"), ("25-34", "35-49"),
                  ("35-49", "50-64"), ("50-64", "65-xx")]

SPACES_RE = re.compile(r"\s+")


def collapse_spaces(series: pd.Series) -> pd.Series:
    """Colapsa espacios repetidos y recorta extremos (src/preprocessing.py:98-100)."""
    return series.fillna("").astype(str).str.replace(SPACES_RE, " ", regex=True).str.strip()


def tokens_of(series: pd.Series, pattern: str) -> pd.Series:
    """Extrae tokens por patron y los une en minusculas (src/pipeline.py:75-78)."""
    return collapse_spaces(series.str.findall(pattern).apply(lambda values: " ".join(values).lower()))


def cargar_train(raiz: Path) -> pd.DataFrame:
    """Lee age_train.csv como data.load_train(): sep ';', utf-8-sig, texto sin padding."""
    marco = pd.read_csv(raiz / "data" / "raw" / "age_train.csv", sep=";", encoding="utf-8-sig")
    marco["text"] = marco["text"].fillna("").astype(str).str.strip()
    return marco


def cargar_test(raiz: Path) -> pd.DataFrame:
    """Lee age_test.csv con la misma convencion."""
    marco = pd.read_csv(raiz / "data" / "raw" / "age_test.csv", sep=";", encoding="utf-8-sig")
    marco["text"] = marco["text"].fillna("").astype(str).str.strip()
    return marco


def cargar_folds(raiz: Path) -> np.ndarray:
    """fold_idx canonico (protocolo congelado): 169.336 valores 0-4.

    En los spikes se lee del carril de artefactos; el notebook embebe este vector.
    """
    return np.load(raiz / "artifacts" / "p33_scores" / "todos" / "fold_idx.npy")


def to_scores(raw: np.ndarray, classes, probabilistic: bool) -> np.ndarray:
    """Reordena columnas al orden canonico y aplica log a las probabilidades (src/pipeline.py:89-94)."""
    orden = [list(classes).index(etiqueta) for etiqueta in CLASSES]
    scores = raw[:, orden]
    return np.log(np.clip(scores, 1e-12, None)) if probabilistic else scores


def softmax_ordinal(continuous: np.ndarray) -> np.ndarray:
    """Convierte la prediccion ordinal continua en distribucion suave (src/pipeline.py:81-86)."""
    midpoints = np.array([CLASS_MIDPOINTS[etiqueta] for etiqueta in CLASSES])
    distances = np.abs(continuous[:, None] - midpoints)
    soft = np.exp(-distances / 5.0)
    return soft / soft.sum(axis=1, keepdims=True)
