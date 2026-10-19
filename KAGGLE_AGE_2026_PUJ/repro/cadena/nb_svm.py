r"""NB-SVM (Wang & Manning 2012) uno-vs-rest: log-count ratio + SVM lineal por clase.

Porte literal de los originales del proyecto KAGGLE_AGE_2026_PUJ:
- scripts/advanced_05_nbsvm_seleccion.py:66-94 (ADV05: char_wb 3-6 300k + menciones).
- scripts/phase7_nbsvm_canal_meta.py:57-78 (ADV07: solo n-gramas de palabra).
- Lado test (mismo calculo reajustado con todo el train): scripts/phase3_holdout_v11.py:115-127
  y scripts/phase25_submission_fresca.py:77-89.

Reglas del paquete: ASCII puro; sin efectos en import; solo numpy/pandas/scipy/sklearn y
`cadena.comun`. Interfaz en plan/kaggle_notebook/port_interface.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from cadena.comun import CLASSES, tokens_of


def nb_log_count_ratio(x_train: sp.csr_matrix, binary_target: np.ndarray) -> np.ndarray:
    r"""Log-count ratio de la clase positiva frente al resto (Wang & Manning 2012).

    Porte literal de scripts/advanced_05_nbsvm_seleccion.py:66-70 y
    scripts/phase7_nbsvm_canal_meta.py:57-61:
    log((pos / pos.sum()) / (neg / neg.sum())) con +1.0 de suavizado en los conteos.
    """
    positive = np.asarray(x_train[binary_target == 1].sum(axis=0)).ravel() + 1.0
    negative = np.asarray(x_train[binary_target == 0].sum(axis=0)).ravel() + 1.0
    return np.log((positive / positive.sum()) / (negative / negative.sum()))


def nb_svm_margins(textos: pd.Series, y: np.ndarray, train_idx: np.ndarray,
                   valid_idx: np.ndarray, config: dict) -> np.ndarray:
    r"""Margenes NB-SVM uno-vs-rest del fold retenido (float32, columnas en CLASSES).

    Porte de scripts/advanced_05_nbsvm_seleccion.py:73-94 (ADV05, char300 + menciones) y
    scripts/phase7_nbsvm_canal_meta.py:64-78 (ADV07, solo palabras): vectorizador(es)
    TF-IDF ajustados SOLO con textos.iloc[train_idx]; por clase (en orden CLASSES)
    log-count ratio sobre el train del fold y LinearSVC(C=0.2, max_iter=3000) entrenado
    sobre x_train.multiply(ratio).tocsr(); decision_function sobre el fold retenido
    multiplicado por el mismo ratio. Las columnas ya quedan en el orden de CLASSES
    porque el bucle entrena una clase por posicion.

    config: parametros de TfidfVectorizer del canal principal. Clave opcional
    "mentions" = {"config": {...}, "pattern": r"@\w+"} para anadir un segundo canal
    TF-IDF sobre tokens_of(textos, pattern). ADV05 la usa con CHAR_300K + MENTION y
    patron r"@\w+"; ADV07 pasa WORD sin canal de menciones (el texto tal cual).

    train_idx/valid_idx: indices enteros (p. ej. np.flatnonzero) de las filas de
    entrenamiento y validacion, como fold["train_idx"]/fold["val_idx"] del protocolo
    congelado. Salida (len(valid_idx), 6) float32.
    """
    config = dict(config)
    mentions = config.pop("mentions", None)
    y = np.asarray(y)
    vectorizer = TfidfVectorizer(**config)
    if mentions is None:
        x_train = vectorizer.fit_transform(textos.iloc[train_idx])
        x_valid = vectorizer.transform(textos.iloc[valid_idx])
    else:
        mention_vectorizer = TfidfVectorizer(**mentions["config"])
        pattern = mentions["pattern"]
        x_train = sp.hstack([
            vectorizer.fit_transform(textos.iloc[train_idx]),
            mention_vectorizer.fit_transform(tokens_of(textos.iloc[train_idx], pattern)),
        ]).tocsr()
        x_valid = sp.hstack([
            vectorizer.transform(textos.iloc[valid_idx]),
            mention_vectorizer.transform(tokens_of(textos.iloc[valid_idx], pattern)),
        ]).tocsr()
    y_train = y[train_idx]

    margins = np.zeros((len(valid_idx), len(CLASSES)), dtype=np.float32)
    for position, label in enumerate(CLASSES):
        binary_target = (y_train == label).astype(int)
        ratio = nb_log_count_ratio(x_train, binary_target)
        model = LinearSVC(C=0.2, max_iter=3000)
        model.fit(x_train.multiply(ratio).tocsr(), binary_target)
        margins[:, position] = model.decision_function(x_valid.multiply(ratio).tocsr())
    return margins
