r"""Miembros OOF del paquete B de la cadena AGE: E-017, E-019b, E-021a, E-021b, E-028a y ADV05.

Firma comun `(textos, y, fold_idx)`: por fold del protocolo congelado (5 folds) se entrena
con `fold_idx != fold` y se predice `fold_idx == fold`, reensamblando por indice. Salidas
float32 con columnas en CLASSES, salvo E-028a (columnas = ADJACENT_PAIRS).

Fuentes (porte literal, sin mejorar ni reinterpretar):
- E-017: scripts/experimento_e016_e017.py:97-108,127.
- E-019b: scripts/experimento_e019_modelos_densos.py:65-80,98-99,113-131,157.
- E-021a: scripts/experimento_e021_ordinal_centroides.py:57-75,110-122,150.
- E-021b: scripts/experimento_e021_ordinal_centroides.py:78-91,110-127,151.
- E-028a: scripts/experimento_e028_ordinal_fino.py:92-106,147.
- ADV05: scripts/advanced_05_nbsvm_seleccion.py:73-94,138-146.

Reglas del paquete: ASCII puro; sin efectos en import; solo numpy/pandas/scipy/sklearn y
`cadena` (comun/nb_svm). Interfaz en plan/kaggle_notebook/port_interface.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

from cadena.comun import (ADJACENT_PAIRS, CHAR_300K, CLASSES, MENTION, SEED, WORD,
                          tokens_of)
from cadena.nb_svm import nb_svm_margins


def _folds_estandar(fold_idx: np.ndarray):
    """Genera (fold, train_idx, valid_idx) del protocolo congelado (5 folds).

    Entrena con fold_idx != fold y predice fold_idx == fold; devuelve indices enteros,
    como fold["train_idx"]/fold["val_idx"] de src/cv.py:332-374 (make_folds agrupado).
    """
    fold_idx = np.asarray(fold_idx)
    for fold in range(5):
        yield fold, np.flatnonzero(fold_idx != fold), np.flatnonzero(fold_idx == fold)


def _tfidf_char_menciones(textos: pd.Series, train_idx: np.ndarray, valid_idx: np.ndarray):
    r"""TF-IDF in-fold char_wb 3-6 (CHAR_300K) + canal de menciones (MENTION, r"@\w+").

    Porte de scripts/experimento_e021_ordinal_centroides.py:112-117; identico en
    scripts/experimento_e019_modelos_densos.py:114-119 y
    scripts/experimento_e028_ordinal_fino.py:84-89.
    """
    char = TfidfVectorizer(**CHAR_300K)
    menciones = TfidfVectorizer(**MENTION)
    x_train = hstack([char.fit_transform(textos.iloc[train_idx]),
                      menciones.fit_transform(tokens_of(textos.iloc[train_idx],
                                                        r"@\w+"))]).tocsr()
    x_valid = hstack([char.transform(textos.iloc[valid_idx]),
                      menciones.transform(tokens_of(textos.iloc[valid_idx],
                                                    r"@\w+"))]).tocsr()
    return x_train, x_valid


def _ordinal_probabilidades(x_train, y_index: np.ndarray, x_valid) -> np.ndarray:
    r"""Distribucion de clase por modelo ordinal acumulativo (Frank & Hall 2001).

    Porte de scripts/experimento_e021_ordinal_centroides.py:57-75: K-1 clasificadores
    binarios P(y > k) con LogisticRegression(C=1.0, max_iter=1000, solver="liblinear",
    random_state=SEED); acumuladas monotonizadas con minimum.accumulate y clip [0, 1];
    P(y = k) por diferencias (-diff + ultima columna), clip 1e-12 y normalizacion.
    """
    thresholds = np.arange(1, len(CLASSES))
    cumulative = np.ones((x_valid.shape[0], len(thresholds) + 1), dtype=np.float64)
    for position, threshold in enumerate(thresholds):
        binary_target = (y_index >= threshold).astype(int)
        model = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear",
                                   random_state=SEED)
        model.fit(x_train, binary_target)
        cumulative[:, position + 1] = model.predict_proba(x_valid)[:, 1]
    cumulative = np.clip(np.minimum.accumulate(cumulative, axis=1), 0.0, 1.0)
    # P(y = k) = P(y > k-1) - P(y > k); la ultima clase es P(y > K-1) - 0.
    probabilities = np.column_stack([-np.diff(cumulative, axis=1), cumulative[:, -1]])
    probabilities = np.clip(probabilities, 1e-12, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _rocchio_probabilidades(x_train, y: np.ndarray, x_valid) -> np.ndarray:
    r"""Clasificador de centroides (Rocchio) con similitud coseno y softmax T=8.

    Porte de scripts/experimento_e021_ordinal_centroides.py:78-91: centroide medio por
    clase normalizado L2 (epsilon 1e-12); norma de cada fila valida (sqrt de la suma de
    cuadrados, epsilon 1e-12); similitud coseno; softmax desplazada por el maximo por
    fila con temperatura 8.0.
    """
    centroids = []
    for label in CLASSES:
        mask = y == label
        centroid = np.asarray(x_train[mask].mean(axis=0)).ravel()
        centroids.append(centroid / (np.linalg.norm(centroid) + 1e-12))
    centroid_matrix = np.vstack(centroids)

    valid_norms = np.sqrt(np.asarray(x_valid.multiply(x_valid).sum(axis=1)).ravel()) + 1e-12
    similarity = np.asarray(x_valid @ centroid_matrix.T) / valid_norms[:, None]
    shifted = similarity - similarity.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted * 8.0)  # temperatura: afina la distribucion
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def oof_e017(textos: pd.Series, y: np.ndarray, fold_idx: np.ndarray) -> np.ndarray:
    r"""Probabilidades OOF de E-017: LogReg OvR C=4.0 sobre word 1-2 (min_df=3, 300k).

    Porte de scripts/experimento_e016_e017.py:97-108 y 127: TfidfVectorizer(**WORD)
    ajustado in-fold; OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=1000,
    solver="liblinear", random_state=SEED)); predict_proba del fold retenido reordenada
    a CLASSES con logreg.classes_. Salida (n, 6) float32.
    """
    y = np.asarray(y)
    probabilidades = np.zeros((len(textos), len(CLASSES)), dtype=np.float32)
    for _fold, train_idx, valid_idx in _folds_estandar(fold_idx):
        word = TfidfVectorizer(**WORD)
        x_train = word.fit_transform(textos.iloc[train_idx])
        x_valid = word.transform(textos.iloc[valid_idx])
        logreg = OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=1000,
                                                        solver="liblinear",
                                                        random_state=SEED))
        logreg.fit(x_train, y[train_idx])
        orden = [list(logreg.classes_).index(etiqueta) for etiqueta in CLASSES]
        probabilidades[valid_idx] = logreg.predict_proba(x_valid)[:, orden]
    return probabilidades


def oof_e019b(textos: pd.Series, y: np.ndarray, fold_idx: np.ndarray,
              dense: pd.DataFrame) -> np.ndarray:
    r"""Probabilidades OOF de E-019b: MLP(256,128) sobre [35 densas + SVD(64)].

    Porte de scripts/experimento_e019_modelos_densos.py:113-131 y 157 (variante B) mas
    fit_mlp (:65-80) y el LabelEncoder con assert de orden CLASSES (:98-99): TF-IDF
    in-fold char_wb 3-6 (CHAR_300K) + menciones (MENTION, r"@\w+"); TruncatedSVD(
    n_components=64, random_state=SEED) ajustada SOLO con el train del fold; entrada
    [densas | z] escalada con StandardScaler (ajustado in-fold); MLPClassifier(
    hidden_layer_sizes=(256, 128), alpha=1e-3, batch_size=512, learning_rate_init=1e-3,
    max_iter=60, early_stopping=True, n_iter_no_change=6, random_state=SEED) con etiquetas
    codificadas como enteros (scikit-learn 1.8 falla en early stopping con etiquetas
    cadena). Las 35 densas = columnas de `dense` sin {n_emoji, emoji_diversity, has_emoji}.
    Salida (n, 6) float32.
    """
    y = np.asarray(y)
    columnas = [col for col in dense.columns
                if col not in {"n_emoji", "emoji_diversity", "has_emoji"}]
    assert len(columnas) == 35, f"se esperaban 35 densas y hay {len(columnas)}"
    densas = dense[columnas].to_numpy(float)
    encoder = LabelEncoder().fit(y)
    assert list(encoder.classes_) == list(CLASSES), "orden de clases inesperado"
    probabilidades = np.zeros((len(textos), len(CLASSES)), dtype=np.float32)
    for _fold, train_idx, valid_idx in _folds_estandar(fold_idx):
        x_train, x_valid = _tfidf_char_menciones(textos, train_idx, valid_idx)
        svd = TruncatedSVD(n_components=64, random_state=SEED)
        z_train = svd.fit_transform(x_train)
        z_valid = svd.transform(x_valid)
        combined_train = np.hstack([densas[train_idx], z_train])
        combined_valid = np.hstack([densas[valid_idx], z_valid])
        scaler = StandardScaler().fit(combined_train)
        model = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, batch_size=512,
                              learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                              n_iter_no_change=6, random_state=SEED)
        model.fit(scaler.transform(combined_train), encoder.transform(y[train_idx]))
        probabilidades[valid_idx] = model.predict_proba(scaler.transform(combined_valid))
    return probabilidades


def oof_e021a(textos: pd.Series, y: np.ndarray, fold_idx: np.ndarray) -> np.ndarray:
    r"""Probabilidades OOF de E-021a: ordinal acumulativo Frank & Hall (5 binarios).

    Porte de scripts/experimento_e021_ordinal_centroides.py:110-122 y 150: por fold,
    representacion in-fold char_wb 3-6 (CHAR_300K) + menciones; y_index = posicion de la
    clase en CLASSES; modelo ordinal acumulativo :57-75. Salida (n, 6) float32.
    """
    y = np.asarray(y)
    y_index = np.array([CLASSES.index(etiqueta) for etiqueta in y])
    probabilidades = np.zeros((len(textos), len(CLASSES)), dtype=np.float32)
    for _fold, train_idx, valid_idx in _folds_estandar(fold_idx):
        x_train, x_valid = _tfidf_char_menciones(textos, train_idx, valid_idx)
        probabilidades[valid_idx] = _ordinal_probabilidades(x_train, y_index[train_idx],
                                                            x_valid)
    return probabilidades


def oof_e021b(textos: pd.Series, y: np.ndarray, fold_idx: np.ndarray) -> np.ndarray:
    r"""Probabilidades OOF de E-021b: centroides Rocchio + softmax temperatura 8.0.

    Porte de scripts/experimento_e021_ordinal_centroides.py:110-127 y 151: por fold,
    representacion in-fold char_wb 3-6 (CHAR_300K) + menciones; centroides Rocchio :78-91
    sobre el train del fold. Salida (n, 6) float32.
    """
    y = np.asarray(y)
    probabilidades = np.zeros((len(textos), len(CLASSES)), dtype=np.float32)
    for _fold, train_idx, valid_idx in _folds_estandar(fold_idx):
        x_train, x_valid = _tfidf_char_menciones(textos, train_idx, valid_idx)
        probabilidades[valid_idx] = _rocchio_probabilidades(x_train, y[train_idx], x_valid)
    return probabilidades


def oof_e028a(textos: pd.Series, y: np.ndarray, fold_idx: np.ndarray) -> np.ndarray:
    r"""Margenes OOF de E-028a: 5 SVM entrenados SOLO con cada par adyacente.

    Porte de scripts/experimento_e028_ordinal_fino.py:92-106 y 147: representacion
    in-fold char_wb 3-6 (CHAR_300K) + menciones; por par (lower, upper) de ADJACENT_PAIRS
    mascara con las dos clases, objetivo (y_train == upper) y LinearSVC(C=0.2,
    max_iter=3000); decision_function del fold retenido (margen positivo -> clase mayor).
    Columnas = ADJACENT_PAIRS. Salida (n, 5) float32.
    """
    y = np.asarray(y)
    margenes = np.zeros((len(textos), len(ADJACENT_PAIRS)), dtype=np.float32)
    for _fold, train_idx, valid_idx in _folds_estandar(fold_idx):
        x_train, x_valid = _tfidf_char_menciones(textos, train_idx, valid_idx)
        y_train = y[train_idx]
        pair_margins = np.zeros((len(valid_idx), len(ADJACENT_PAIRS)), dtype=np.float32)
        for position, (lower, upper) in enumerate(ADJACENT_PAIRS):
            mask = (y_train == lower) | (y_train == upper)
            binary_target = (y_train[mask] == upper).astype(int)
            model = LinearSVC(C=0.2, max_iter=3000)
            model.fit(x_train[mask], binary_target)
            margin = model.decision_function(x_valid)
            pair_margins[:, position] = margin
        margenes[valid_idx] = pair_margins
    return margenes


def oof_adv05(textos: pd.Series, y: np.ndarray, fold_idx: np.ndarray) -> np.ndarray:
    r"""Margenes OOF de ADV05: NB-SVM uno-vs-rest sobre char300 + menciones.

    Porte de scripts/advanced_05_nbsvm_seleccion.py:73-94 (nb_svm_fold) y 138-146 (bucle
    OOF y guardado): por fold, TfidfVectorizer(**CHAR_300K) + TfidfVectorizer(**MENTION)
    sobre tokens_of(textos, r"@\w+"), log-count ratio por clase y LinearSVC(C=0.2,
    max_iter=3000) uno-vs-rest via nb_svm_margins. Salida (n, 6) float32.
    """
    y = np.asarray(y)
    config = dict(CHAR_300K)
    config["mentions"] = {"config": dict(MENTION), "pattern": r"@\w+"}
    margenes = np.zeros((len(textos), len(CLASSES)), dtype=np.float32)
    for _fold, train_idx, valid_idx in _folds_estandar(fold_idx):
        margenes[valid_idx] = nb_svm_margins(textos, y, train_idx, valid_idx, config)
    return margenes
