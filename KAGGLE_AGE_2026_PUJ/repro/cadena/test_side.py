"""Lado TEST de la cadena del campeon: re-ajuste de la cadena sobre el train completo.

Reproduce el camino de test del pipeline v12/V30 (matriz 279) sin dependencias del proyecto
original (prohibido importar `src`/`scripts`; solo `cadena.comun` y `cadena.nb_svm`):

    - representaciones dispersas: porte de ``src/pipeline.py:157-198``
      (``build_representations``), ajustadas solo con train y transformando test.
    - 13 bloques de scores de test: porte de ``src/pipeline.py:199-288``
      (``fit_base_models``) en el orden de ``OOF_FILES`` (``src/pipeline.py:59-65``), con la
      transformacion de ``MEMBER_FILES`` (``scripts/phase2_ponderacion.py:49-55``): ``log``
      si la fuente es probabilistica, crudo si no.
    - NB-SVM de palabras (ADV07): porte de ``scripts/phase25_submission_fresca.py:92-103``.
    - ensamblado (m, 279): porte de ``scripts/phase25_submission_fresca.py:108-111``.

Reglas del porte: ASCII puro, sin efectos en import, sin aleatoriedad anadida (semilla 42).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

from cadena.comun import (
    ADJACENT_PAIRS,
    CHAR_300K,
    CHAR_700K,
    CLASS_MIDPOINTS,
    CLASSES,
    HASHTAG,
    MENTION,
    SEED,
    WORD,
    softmax_ordinal,
    to_scores,
    tokens_of,
)
from cadena.nb_svm import nb_log_count_ratio

#: src/config.py:59 -- TEXT_COL = "text".
TEXT_COL = "text"

#: Componentes de la SVD del miembro E019b (src/pipeline.py:52).
SVD_COMPONENTS = 64


def adjacent_pair_margins(x_train, y_train: np.ndarray, x_valid) -> np.ndarray:
    """Margenes de 5 SVM entrenados **solo** con cada par de rangos contiguos (E-028a).

    Porte de ``src/pipeline.py:101-115``. Concentra la capacidad del modelo en las fronteras
    donde ocurre el error (la confusion del ensamble es casi toda entre rangos adyacentes).
    """
    margins = np.zeros((x_valid.shape[0], len(ADJACENT_PAIRS)), dtype=np.float32)
    for position, (lower, upper) in enumerate(ADJACENT_PAIRS):
        mask = (y_train == lower) | (y_train == upper)
        binary_target = (y_train[mask] == upper).astype(int)
        model = LinearSVC(C=0.2, max_iter=3000)
        model.fit(x_train[mask], binary_target)
        margins[:, position] = model.decision_function(x_valid)
    return margins


def ordinal_cumulative_probabilities(x_train, y_index: np.ndarray, x_valid) -> np.ndarray:
    """Modelo ordinal acumulativo (Frank & Hall 2001) sobre la representacion dispersa.

    Porte de ``src/pipeline.py:118-136``. Entrena K-1 clasificadores binarios ``P(y > k)`` y
    los combina en una distribucion: ``P(y = k) = P(y > k-1) - P(y > k)`` (E-021a: 0,5527).
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
    probabilities = np.column_stack([-np.diff(cumulative, axis=1), cumulative[:, -1]])
    probabilities = np.clip(probabilities, 1e-12, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def rocchio_probabilities(x_train, y: np.ndarray, x_valid) -> np.ndarray:
    """Clasificador de centroides (Rocchio) con similitud coseno (E-021b: 0,4017).

    Porte de ``src/pipeline.py:139-154``. Miembro deliberadamente debil pero **diverso**.
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
    exponentials = np.exp(shifted * 8.0)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def representaciones_test(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Ajusta los vectorizadores con train y transforma train y test.

    Porte de ``src/pipeline.py:157-198`` (``build_representations``). Configs congeladas
    ``CHAR_300K``/``CHAR_700K``/``MENTION``/``HASHTAG``/``WORD`` de ``cadena.comun``
    (= ``src/pipeline.py:42-51``). Los tokens de menciones y hashtags se extraen con
    ``tokens_of`` (``src/pipeline.py:75-78``) y patrones ``r"@\\w+"`` / ``r"#\\w+"``.

    Returns
    -------
    dict
        Matrices dispersas de cada canal mas el detalle de dimensiones (mismas claves que el
        original: x300/x700/xh/xw de train y test, y "dimensiones").
    """
    char300 = TfidfVectorizer(**CHAR_300K)
    mention300 = TfidfVectorizer(**MENTION)
    x300_train = hstack([char300.fit_transform(train[TEXT_COL]),
                         mention300.fit_transform(tokens_of(train[TEXT_COL], r"@\w+"))]).tocsr()
    x300_test = hstack([char300.transform(test[TEXT_COL]),
                        mention300.transform(tokens_of(test[TEXT_COL], r"@\w+"))]).tocsr()

    char700 = TfidfVectorizer(**CHAR_700K)
    mention700 = TfidfVectorizer(**MENTION)
    x700_train = hstack([char700.fit_transform(train[TEXT_COL]),
                         mention700.fit_transform(tokens_of(train[TEXT_COL], r"@\w+"))]).tocsr()
    x700_test = hstack([char700.transform(test[TEXT_COL]),
                        mention700.transform(tokens_of(test[TEXT_COL], r"@\w+"))]).tocsr()

    hashtag = TfidfVectorizer(**HASHTAG)
    xh_train = hstack([x300_train,
                       hashtag.fit_transform(tokens_of(train[TEXT_COL], r"#\w+"))]).tocsr()
    xh_test = hstack([x300_test,
                      hashtag.transform(tokens_of(test[TEXT_COL], r"#\w+"))]).tocsr()

    word = TfidfVectorizer(**WORD)
    xw_train = word.fit_transform(train[TEXT_COL])
    xw_test = word.transform(test[TEXT_COL])

    return {"x300_train": x300_train, "x300_test": x300_test,
            "x700_train": x700_train, "x700_test": x700_test,
            "xh_train": xh_train, "xh_test": xh_test,
            "xw_train": xw_train, "xw_test": xw_test,
            "dimensiones": {"char300+menciones": int(x300_train.shape[1]),
                            "char700+menciones": int(x700_train.shape[1]),
                            "char300+menciones+hashtags": int(xh_train.shape[1]),
                            "word12": int(xw_train.shape[1])}}


def base_models_test(reps: dict, y: np.ndarray, dense_train: np.ndarray, dense_test: np.ndarray,
                     textos_train: pd.Series, textos_test: pd.Series) -> list[np.ndarray]:
    """Ajusta los 13 modelos base sobre todo el train y devuelve sus scores de test.

    Porte de ``src/pipeline.py:199-288`` (``fit_base_models``). Los 13 bloques van en el orden
    de ``OOF_FILES`` (``src/pipeline.py:59-65``) con la transformacion ``log`` de
    ``MEMBER_FILES`` (``scripts/phase2_ponderacion.py:49-55``) cuando la fuente es
    probabilistica; el resto va crudo:

        1.  E013  LinearSVC(C=0.2, max_iter=3000) sobre x300  -> margenes (raw, 6)
        2.  E016  LinearSVC(C=0.2, max_iter=3000) sobre x700  -> margenes (raw, 6)
        3.  E-009 OneVsRest(LogReg C=4.0) sobre x300          -> proba log (6)
        4.  E-011 SGDClassifier(log_loss) sobre x300          -> proba log (6)
        5.  E-010 ComplementNB(alpha=0.3) sobre x300          -> proba log (6)
        6.  E014  LinearSVC(C=0.2, max_iter=3000) sobre xh    -> margenes (raw, 6)
        7.  E015_ordinal  Ridge(alpha=1.0, sparse_cg) sobre x300 -> continuo (raw, 1)
        8.  E015_softmax  log(softmax_ordinal(continuo))      -> log (6)
        9.  E017  OneVsRest(LogReg C=4.0) sobre xw            -> proba log (6)
        10. E019b MLP(256,128) sobre [35 densas + SVD64 de x300] -> proba log (6)
        11. E021a ordinal acumulativo sobre x300              -> log (6)
        12. E021b centroides Rocchio sobre x300               -> log (6)
        13. E028a margenes de 5 SVM de pares adyacentes       -> raw (5)

    Total 72 columnas (11 bloques de 6 + 1 de 1 + 1 de 5). ``textos_train``/``textos_test``
    forman parte de la interfaz congelada del porte; el cuerpo literal del original no los
    consume (toda la senal de texto entra por ``reps``). El ``detail`` con la varianza
    explicada por la SVD del original no forma parte de la interfaz (se retorna solo la lista).

    Parameters
    ----------
    reps : dict
        Salida de ``representaciones_test``.
    y : np.ndarray
        Etiquetas de train (strings de ``CLASSES``).
    dense_train, dense_test : np.ndarray
        35 features densas (``DENSE_FEATURES``, sin las 3 de emoji).
    textos_train, textos_test : pd.Series
        Textos crudos (no consumidos por el cuerpo portado).

    Returns
    -------
    list[np.ndarray]
        13 bloques de scores de test en el orden de ``OOF_FILES``.
    """
    scores: list[np.ndarray] = []

    model = LinearSVC(C=0.2, max_iter=3000).fit(reps["x300_train"], y)
    scores.append(to_scores(model.decision_function(reps["x300_test"]), model.classes_, False))
    model = LinearSVC(C=0.2, max_iter=3000).fit(reps["x700_train"], y)
    scores.append(to_scores(model.decision_function(reps["x700_test"]), model.classes_, False))

    model = OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=1000, solver="liblinear",
                                                   random_state=SEED))
    model.fit(reps["x300_train"], y)
    scores.append(to_scores(model.predict_proba(reps["x300_test"]), model.classes_, True))

    model = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=30, tol=1e-4,
                          random_state=SEED).fit(reps["x300_train"], y)
    scores.append(to_scores(model.predict_proba(reps["x300_test"]), model.classes_, True))
    model = ComplementNB(alpha=0.3).fit(reps["x300_train"], y)
    scores.append(to_scores(model.predict_proba(reps["x300_test"]), model.classes_, True))

    model = LinearSVC(C=0.2, max_iter=3000).fit(reps["xh_train"], y)
    scores.append(to_scores(model.decision_function(reps["xh_test"]), model.classes_, False))

    ordinal_target = np.array([CLASS_MIDPOINTS[label] for label in y])
    ridge = Ridge(alpha=1.0, solver="sparse_cg", max_iter=2000, random_state=SEED)
    ridge.fit(reps["x300_train"], ordinal_target)
    continuous_test = ridge.predict(reps["x300_test"])
    scores.append(continuous_test.reshape(-1, 1))
    scores.append(np.log(np.clip(softmax_ordinal(continuous_test), 1e-12, None)))

    model = OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=1000, solver="liblinear",
                                                   random_state=SEED))
    model.fit(reps["xw_train"], y)
    scores.append(to_scores(model.predict_proba(reps["xw_test"]), model.classes_, True))

    # MLP sobre densas + SVD del espacio de caracteres (miembro diverso, E019b).
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED)
    z_train = svd.fit_transform(reps["x300_train"])
    z_test = svd.transform(reps["x300_test"])
    scaler = StandardScaler().fit(np.hstack([dense_train, z_train]))
    encoder = LabelEncoder().fit(y)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, batch_size=512,
                        learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                        n_iter_no_change=6, random_state=SEED)
    mlp.fit(scaler.transform(np.hstack([dense_train, z_train])), encoder.transform(y))
    probabilities = mlp.predict_proba(scaler.transform(np.hstack([dense_test, z_test])))
    scores.append(np.log(np.clip(probabilities, 1e-12, None)))

    # Miembro ordinal acumulativo (Frank & Hall): 5 clasificadores binarios sobre x300.
    y_index = np.array([CLASSES.index(label) for label in y])
    ordinal_probabilities = ordinal_cumulative_probabilities(reps["x300_train"], y_index,
                                                             reps["x300_test"])
    scores.append(np.log(np.clip(ordinal_probabilities, 1e-12, None)))

    # Miembro de centroides (Rocchio): debil pero diverso.
    centroids = rocchio_probabilities(reps["x300_train"], y, reps["x300_test"])
    scores.append(np.log(np.clip(centroids, 1e-12, None)))

    # Supervision ordinal fina: margenes entre pares de rangos contiguos.
    pair_margins = adjacent_pair_margins(reps["x300_train"], y, reps["x300_test"])
    scores.append(pair_margins)

    return scores


def nb_word_test(textos_train: pd.Series, y: np.ndarray,
                 textos_test: pd.Series) -> np.ndarray:
    """NB-SVM de palabras (ADV07) re-ajustado sobre el train completo.

    Porte de ``scripts/phase25_submission_fresca.py:92-103`` (identico en
    ``scripts/phase8_holdout_v12.py:153-164``). Mismo vectorizador (``WORD`` de comun:
    analyzer word, ngramas 1-2, min_df=3, sublinear_tf, 300k, lowercase) ajustado sobre
    **todo el train**; log-count ratio por clase sobre todo el train (``nb_log_count_ratio``,
    ``scripts/phase7_nbsvm_canal_meta.py:57-61``); ``LinearSVC(C=0.2, max_iter=3000)``
    uno-vs-rest por clase y ``decision_function`` sobre el test.

    Returns
    -------
    np.ndarray
        (m, 6) float32, margenes crudos en el orden de ``CLASSES`` (entran sin transformar
        como ultimas 6 columnas de la matriz 279).
    """
    word = TfidfVectorizer(**WORD)
    x_train_w = word.fit_transform(textos_train)
    x_test_w = word.transform(textos_test)
    margenes = np.zeros((len(textos_test), len(CLASSES)), dtype=np.float32)
    for posicion, etiqueta in enumerate(CLASSES):
        binario = (y == etiqueta).astype(int)
        ratio = nb_log_count_ratio(x_train_w, binario)
        modelo = LinearSVC(C=0.2, max_iter=3000)
        modelo.fit(x_train_w.multiply(ratio).tocsr(), binario)
        margenes[:, posicion] = modelo.decision_function(x_test_w.multiply(ratio).tocsr())
    return margenes


def matriz_test_fresca(bloques_72: list[np.ndarray], nb_test: np.ndarray,
                       densas_test: np.ndarray, estructuradas_test: np.ndarray,
                       supervisadas_test: np.ndarray, nb_word: np.ndarray) -> np.ndarray:
    """Ensambla el lado test (m, 279) en el orden EXACTO del campeon.

    Porte de ``scripts/phase25_submission_fresca.py:108-111``. Orden de columnas:

        [72 scores del pipeline (13 bloques)] + [nb_test 6] + [densas 35] +
        [estructuradas 87] + [supervisadas 73] + [nb_word 6]  = 279

    Parameters
    ----------
    bloques_72 : list[np.ndarray]
        13 bloques de ``base_models_test`` (6x11 + ordinal 1 + pares 5).
    nb_test : np.ndarray
        (m, 6) NB-SVM de char300+menciones (ADV05 lado test; ver ``cadena.nb_svm``).
    densas_test, estructuradas_test, supervisadas_test : np.ndarray
        (m, 35), (m, 87) y (m, 73) respectivamente.
    nb_word : np.ndarray
        (m, 6) salida de ``nb_word_test`` (ADV07 lado test).

    Returns
    -------
    np.ndarray
        (m, 279); falla con ``AssertionError`` si el ancho no es 279.
    """
    matriz = np.concatenate(bloques_72 + [nb_test, densas_test, estructuradas_test,
                                          supervisadas_test, nb_word], axis=1)
    assert matriz.shape[1] == 279, f"lado test inesperado: {matriz.shape}"
    return matriz
