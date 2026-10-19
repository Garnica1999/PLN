"""Miembros OOF del bloque A de la cadena AGE (porte fiel del proyecto original).

Cada funcion reproduce EXACTAMENTE el script citado: mismos modelos, hiperparametros,
representaciones y orden de columnas. Protocolo comun: entrenar con ``fold_idx != fold``,
predecir ``fold_idx == fold`` y reensamblar por indice.

Fuentes:
- E-013: scripts/experimento_e013_barrido_c.py:80-92,112
- E-016: scripts/experimento_e016_e017.py:76-87,126
- E-009 / E-010 / E-011: scripts/experimento_e009_modelos_clasicos.py:55-74,147-151,171-175
- E-014 / E-015: scripts/experimento_e014_e015_hipotesis_error.py:101-112,122-132,151-153

Salidas: margenes y probabilidades (n, 6) float32 reordenados a ``comun.CLASSES``;
E-015 ordinal (n,) float32. ASCII puro. Sin efectos en import.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from cadena import comun

# Helpers compartidos (representacion y orden de columnas).


def _orden_canonico(classes: np.ndarray) -> list[int]:
    """Indices para reordenar columnas al orden de ``comun.CLASSES``.

    Patron literal de los originales:
    ``[list(model.classes_).index(label) for label in CLASSES]``
    (scripts/experimento_e013_barrido_c.py:91-92).
    """
    return [list(classes).index(etiqueta) for etiqueta in comun.CLASSES]


def _representacion_char_menciones(train_texts: pd.Series, valid_texts: pd.Series,
                                   char_config: dict) -> tuple[csr_matrix, csr_matrix]:
    """Representacion char + menciones ajustada solo con el train del fold.

    Fuente: scripts/experimento_e009_modelos_clasicos.py:90-98 (``build_representation``);
    bloque identico al de E-013/E-016/E-014/E-015
    (scripts/experimento_e013_barrido_c.py:71-76).
    """
    char = TfidfVectorizer(**char_config)
    menciones = TfidfVectorizer(**comun.MENTION)
    x_train = hstack([char.fit_transform(train_texts),
                      menciones.fit_transform(comun.tokens_of(train_texts, r"@\w+"))]).tocsr()
    x_valid = hstack([char.transform(valid_texts),
                      menciones.transform(comun.tokens_of(valid_texts, r"@\w+"))]).tocsr()
    return x_train, x_valid


# Miembros OOF (uno por artefacto del bloque A).


def oof_e013(textos: pd.Series, y: np.ndarray, fold_idx: np.ndarray) -> np.ndarray:
    """Margenes OOF de E-013.

    Fuente: scripts/experimento_e013_barrido_c.py:80-92,112.
    LinearSVC(C=0.2, max_iter=3000) sobre char_wb 3-6 (min_df=5, max_features=300k) +
    menciones word 1-1 (min_df=2); vectorizadores ajustados in-fold. Devuelve (n, 6)
    float32 con ``decision_function`` reordenado a ``comun.CLASSES``.
    """
    salida = np.zeros((len(textos), len(comun.CLASSES)), dtype=np.float32)
    for fold in np.unique(fold_idx):
        train_idx = fold_idx != fold
        valid_idx = fold_idx == fold
        x_train, x_valid = _representacion_char_menciones(textos.iloc[train_idx],
                                                          textos.iloc[valid_idx],
                                                          comun.CHAR_300K)
        modelo = LinearSVC(C=0.2, max_iter=3000)
        modelo.fit(x_train, y[train_idx])
        salida[valid_idx] = modelo.decision_function(x_valid)[:, _orden_canonico(modelo.classes_)]
    return salida


def oof_e016(textos: pd.Series, y: np.ndarray, fold_idx: np.ndarray) -> np.ndarray:
    """Margenes OOF de E-016.

    Fuente: scripts/experimento_e016_e017.py:76-87,126.
    LinearSVC(C=0.2, max_iter=3000) sobre char_wb 3-6 (min_df=3, max_features=700k) +
    menciones word 1-1 (min_df=2); vectorizadores ajustados in-fold. Devuelve (n, 6)
    float32 con ``decision_function`` reordenado a ``comun.CLASSES``.
    """
    salida = np.zeros((len(textos), len(comun.CLASSES)), dtype=np.float32)
    for fold in np.unique(fold_idx):
        train_idx = fold_idx != fold
        valid_idx = fold_idx == fold
        x_train, x_valid = _representacion_char_menciones(textos.iloc[train_idx],
                                                          textos.iloc[valid_idx],
                                                          comun.CHAR_700K)
        modelo = LinearSVC(C=0.2, max_iter=3000)
        modelo.fit(x_train, y[train_idx])
        salida[valid_idx] = modelo.decision_function(x_valid)[:, _orden_canonico(modelo.classes_)]
    return salida


def oof_e009(textos: pd.Series, y: np.ndarray, fold_idx: np.ndarray) -> np.ndarray:
    """Probabilidades OOF de E-009.

    Fuente: scripts/experimento_e009_modelos_clasicos.py:55-61,147-151,171-175.
    OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=1000, solver="liblinear",
    random_state=42)) sobre char_wb 3-6 (300k) + menciones. Devuelve (n, 6) float32 con
    ``predict_proba`` reordenada a ``comun.CLASSES``.
    """
    salida = np.zeros((len(textos), len(comun.CLASSES)), dtype=np.float32)
    for fold in np.unique(fold_idx):
        train_idx = fold_idx != fold
        valid_idx = fold_idx == fold
        x_train, x_valid = _representacion_char_menciones(textos.iloc[train_idx],
                                                          textos.iloc[valid_idx],
                                                          comun.CHAR_300K)
        modelo = OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=1000,
                                                        solver="liblinear",
                                                        random_state=comun.SEED))
        modelo.fit(x_train, y[train_idx])
        salida[valid_idx] = modelo.predict_proba(x_valid)[:, _orden_canonico(modelo.classes_)]
    return salida


def oof_e011(textos: pd.Series, y: np.ndarray, fold_idx: np.ndarray) -> np.ndarray:
    """Probabilidades OOF de E-011.

    Fuente: scripts/experimento_e009_modelos_clasicos.py:68-74,147-151,171-175.
    SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=30, tol=1e-4, random_state=42,
    n_jobs=32) sobre char_wb 3-6 (300k) + menciones. Devuelve (n, 6) float32 con
    ``predict_proba`` reordenada a ``comun.CLASSES``.
    """
    salida = np.zeros((len(textos), len(comun.CLASSES)), dtype=np.float32)
    for fold in np.unique(fold_idx):
        train_idx = fold_idx != fold
        valid_idx = fold_idx == fold
        x_train, x_valid = _representacion_char_menciones(textos.iloc[train_idx],
                                                          textos.iloc[valid_idx],
                                                          comun.CHAR_300K)
        modelo = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=30, tol=1e-4,
                               random_state=comun.SEED, n_jobs=32)
        modelo.fit(x_train, y[train_idx])
        salida[valid_idx] = modelo.predict_proba(x_valid)[:, _orden_canonico(modelo.classes_)]
    return salida


def oof_e010(textos: pd.Series, y: np.ndarray, fold_idx: np.ndarray) -> np.ndarray:
    """Probabilidades OOF de E-010.

    Fuente: scripts/experimento_e009_modelos_clasicos.py:62-67,147-151,171-175.
    ComplementNB(alpha=0.3) sobre char_wb 3-6 (300k) + menciones. Devuelve (n, 6) float32
    con ``predict_proba`` reordenada a ``comun.CLASSES``.
    """
    salida = np.zeros((len(textos), len(comun.CLASSES)), dtype=np.float32)
    for fold in np.unique(fold_idx):
        train_idx = fold_idx != fold
        valid_idx = fold_idx == fold
        x_train, x_valid = _representacion_char_menciones(textos.iloc[train_idx],
                                                          textos.iloc[valid_idx],
                                                          comun.CHAR_300K)
        modelo = ComplementNB(alpha=0.3)
        modelo.fit(x_train, y[train_idx])
        salida[valid_idx] = modelo.predict_proba(x_valid)[:, _orden_canonico(modelo.classes_)]
    return salida


def oof_e014(textos: pd.Series, y: np.ndarray, fold_idx: np.ndarray) -> np.ndarray:
    """Margenes OOF de E-014.

    Fuente: scripts/experimento_e014_e015_hipotesis_error.py:101-112,151.
    LinearSVC(C=0.2, max_iter=3000) sobre char_wb 3-6 (300k) + menciones + hashtags
    word 1-1 (min_df=2); vectorizadores ajustados in-fold. Devuelve (n, 6) float32 con
    ``decision_function`` reordenado a ``comun.CLASSES``.
    """
    salida = np.zeros((len(textos), len(comun.CLASSES)), dtype=np.float32)
    for fold in np.unique(fold_idx):
        train_idx = fold_idx != fold
        valid_idx = fold_idx == fold
        x_train, x_valid = _representacion_char_menciones(textos.iloc[train_idx],
                                                          textos.iloc[valid_idx],
                                                          comun.CHAR_300K)
        hashtags = TfidfVectorizer(**comun.HASHTAG)
        x_train_h = hstack([x_train, hashtags.fit_transform(
            comun.tokens_of(textos.iloc[train_idx], r"#\w+"))]).tocsr()
        x_valid_h = hstack([x_valid, hashtags.transform(
            comun.tokens_of(textos.iloc[valid_idx], r"#\w+"))]).tocsr()
        modelo = LinearSVC(C=0.2, max_iter=3000)
        modelo.fit(x_train_h, y[train_idx])
        salida[valid_idx] = modelo.decision_function(x_valid_h)[:, _orden_canonico(modelo.classes_)]
    return salida


def oof_e015_ordinal(textos: pd.Series, y: np.ndarray, fold_idx: np.ndarray) -> np.ndarray:
    """Prediccion ordinal continua OOF de E-015.

    Fuente: scripts/experimento_e014_e015_hipotesis_error.py:122-128,152.
    Ridge(alpha=1.0, solver="sparse_cg", max_iter=2000, random_state=42) sobre
    char_wb 3-6 (300k) + menciones, con objetivo ``comun.CLASS_MIDPOINTS`` de cada
    etiqueta (como el ``ordinal_target`` original). Devuelve (n,) float32.
    """
    objetivo = np.array([comun.CLASS_MIDPOINTS[etiqueta] for etiqueta in y])
    salida = np.zeros(len(textos), dtype=np.float32)
    for fold in np.unique(fold_idx):
        train_idx = fold_idx != fold
        valid_idx = fold_idx == fold
        x_train, x_valid = _representacion_char_menciones(textos.iloc[train_idx],
                                                          textos.iloc[valid_idx],
                                                          comun.CHAR_300K)
        modelo = Ridge(alpha=1.0, solver="sparse_cg", max_iter=2000, random_state=comun.SEED)
        modelo.fit(x_train, objetivo[train_idx])
        salida[valid_idx] = modelo.predict(x_valid)
    return salida


def oof_e015_softmax(textos: pd.Series, y: np.ndarray, fold_idx: np.ndarray) -> np.ndarray:
    """Distribucion suave OOF de E-015 (softmax ordinal).

    Fuente: scripts/experimento_e014_e015_hipotesis_error.py:129-132,153.
    Aplica ``comun.softmax_ordinal`` a la salida de ``oof_e015_ordinal``. Devuelve (n, 6)
    float32. El original calculaba el softmax sobre la prediccion continua float64 previa
    al casteo; aqui se aplica sobre la salida float32 (diferencia numerica esperada del
    orden de 1e-6).
    """
    return comun.softmax_ordinal(oof_e015_ordinal(textos, y, fold_idx)).astype(np.float32)
