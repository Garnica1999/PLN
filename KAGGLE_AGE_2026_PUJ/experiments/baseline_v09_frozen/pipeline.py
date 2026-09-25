"""Pipeline final del proyecto (modelo ganador reproducible de extremo a extremo).

Implementa el ensamble de 10 fuentes con meta-modelo enriquecido (submission v06, la mejor del
proyecto: CV LOFO 0,5851 · LB 0,60273) como una función reutilizable que:

1. Carga train/test/plantilla desde ``data/raw`` (o del caché intermedio).
2. Construye las representaciones dispersas ajustando los vectorizadores **solo con train**.
3. Ajusta los 10 modelos base sobre todo el train.
4. Ajusta el meta-modelo con las matrices out-of-fold (nunca con el test).
5. Predice el test, valida el formato contra la plantilla y escribe la submission.

Garantías de la metodología (ver DECISIONS.md)
    - El test **nunca** participa del ajuste de vectorizadores, modelos ni meta-modelo.
    - Determinismo: semilla fija en todos los componentes aleatorios.
    - Validación estricta de la submission antes de escribirla.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

from src import config as cfg
from src import data, preprocessing, submission, textometry
from src.config import set_seed

#: Hiperparámetros congelados del modelo final (elegidos por evidencia, ver DECISIONS.md).
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
SVD_COMPONENTS = 64

#: Punto medio de cada rango de edad (65-xx se representa con 70 por ser extremo abierto).
CLASS_MIDPOINTS = {"13-17": 15.0, "18-24": 21.0, "25-34": 29.5, "35-49": 42.0,
                   "50-64": 57.0, "65-xx": 70.0}

#: Matrices out-of-fold de los modelos base (mismo orden que en el ensamble).
OOF_FILES = [("oof_E013_margenes.npy", False), ("oof_E016_margenes.npy", False),
             ("oof_E-009.npy", True), ("oof_E-011.npy", True), ("oof_E-010.npy", True),
             ("oof_E014_margenes.npy", False), ("oof_E015_ordinal.npy", False),
             ("oof_E015_softmax.npy", True), ("oof_E017_probabilidades.npy", True),
             ("oof_E019b_densas_svd.npy", True),
             ("oof_E021a_ordinal.npy", True), ("oof_E021b_rocchio.npy", True),
             ("oof_E028a_pares.npy", False)]

DENSE_FEATURES = [name for name in textometry.TEXTOMETRY_FEATURES + textometry.STYLE_FEATURES
                  if name not in {"n_emoji", "emoji_diversity", "has_emoji"}]

#: Desempeño de referencia del modelo final (evaluación leave-one-fold-out, Fase 20b/E-020).
REFERENCE_CV_MACRO_F1 = 0.5866
REFERENCE_CV_STD = 0.0038


def tokens_of(series: pd.Series, pattern: str) -> pd.Series:
    """Extrae los tokens que cumplen un patrón y los une en una cadena normalizada."""
    return preprocessing.collapse_spaces(
        series.str.findall(pattern).apply(lambda values: " ".join(values).lower()))


def softmax_ordinal(continuous: np.ndarray) -> np.ndarray:
    """Convierte la predicción ordinal continua en una distribución suave por clase."""
    midpoints = np.array([CLASS_MIDPOINTS[label] for label in cfg.CLASSES])
    distances = np.abs(continuous[:, None] - midpoints)
    soft = np.exp(-distances / 5.0)
    return soft / soft.sum(axis=1, keepdims=True)


def to_scores(raw: np.ndarray, classes, probabilistic: bool) -> np.ndarray:
    """Reordena las columnas al orden canónico y aplica log a las probabilidades."""
    order = [list(classes).index(label) for label in cfg.CLASSES]
    scores = raw[:, order]
    return np.log(np.clip(scores, 1e-12, None)) if probabilistic else scores


#: Fronteras entre rangos contiguos (supervisión ordinal fina, E-028a).
ADJACENT_PAIRS = [("13-17", "18-24"), ("18-24", "25-34"), ("25-34", "35-49"),
                  ("35-49", "50-64"), ("50-64", "65-xx")]


def adjacent_pair_margins(x_train, y_train: np.ndarray, x_valid) -> np.ndarray:
    """Márgenes de 5 SVM entrenados **solo** con cada par de rangos contiguos (E-028a).

    Concentra la capacidad del modelo en las fronteras donde ocurre el error (la confusión del
    ensamble es casi toda entre rangos adyacentes). Aportó +0,0006 al ensamble y la precisión
    media en las fronteras es 0,7428.
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
    """Modelo ordinal acumulativo (Frank & Hall 2001) sobre la representación dispersa.

    Entrena K-1 clasificadores binarios ``P(y > k)`` y los combina en una distribución:
    ``P(y = k) = P(y > k-1) - P(y > k)``. Captura la estructura adyacente de los rangos de edad,
    que el análisis de errores identificó como el fenómeno dominante (E-021a: 0,5527).
    """
    thresholds = np.arange(1, len(cfg.CLASSES))
    cumulative = np.ones((x_valid.shape[0], len(thresholds) + 1), dtype=np.float64)
    for position, threshold in enumerate(thresholds):
        binary_target = (y_index >= threshold).astype(int)
        model = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear",
                                   random_state=cfg.SEED)
        model.fit(x_train, binary_target)
        cumulative[:, position + 1] = model.predict_proba(x_valid)[:, 1]
    cumulative = np.clip(np.minimum.accumulate(cumulative, axis=1), 0.0, 1.0)
    probabilities = np.column_stack([-np.diff(cumulative, axis=1), cumulative[:, -1]])
    probabilities = np.clip(probabilities, 1e-12, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def rocchio_probabilities(x_train, y: np.ndarray, x_valid) -> np.ndarray:
    """Clasificador de centroides (Rocchio) con similitud coseno (E-021b: 0,4017).

    Es un miembro deliberadamente débil pero **diverso**: aportó +0,0009 al ensamble.
    """
    centroids = []
    for label in cfg.CLASSES:
        mask = y == label
        centroid = np.asarray(x_train[mask].mean(axis=0)).ravel()
        centroids.append(centroid / (np.linalg.norm(centroid) + 1e-12))
    centroid_matrix = np.vstack(centroids)
    valid_norms = np.sqrt(np.asarray(x_valid.multiply(x_valid).sum(axis=1)).ravel()) + 1e-12
    similarity = np.asarray(x_valid @ centroid_matrix.T) / valid_norms[:, None]
    shifted = similarity - similarity.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted * 8.0)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def build_representations(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Ajusta los vectorizadores con train y transforma train y test.

    Returns
    -------
    dict
        Matrices dispersas de cada canal más el detalle de dimensiones.
    """
    char300 = TfidfVectorizer(**CHAR_300K)
    mention300 = TfidfVectorizer(**MENTION)
    x300_train = hstack([char300.fit_transform(train[cfg.TEXT_COL]),
                         mention300.fit_transform(tokens_of(train[cfg.TEXT_COL], r"@\w+"))]).tocsr()
    x300_test = hstack([char300.transform(test[cfg.TEXT_COL]),
                        mention300.transform(tokens_of(test[cfg.TEXT_COL], r"@\w+"))]).tocsr()

    char700 = TfidfVectorizer(**CHAR_700K)
    mention700 = TfidfVectorizer(**MENTION)
    x700_train = hstack([char700.fit_transform(train[cfg.TEXT_COL]),
                         mention700.fit_transform(tokens_of(train[cfg.TEXT_COL], r"@\w+"))]).tocsr()
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

    return {"x300_train": x300_train, "x300_test": x300_test,
            "x700_train": x700_train, "x700_test": x700_test,
            "xh_train": xh_train, "xh_test": xh_test,
            "xw_train": xw_train, "xw_test": xw_test,
            "dimensiones": {"char300+menciones": int(x300_train.shape[1]),
                            "char700+menciones": int(x700_train.shape[1]),
                            "char300+menciones+hashtags": int(xh_train.shape[1]),
                            "word12": int(xw_train.shape[1])}}


def fit_base_models(representations: dict, y: np.ndarray, dense_train: np.ndarray,
                    dense_test: np.ndarray, verbose: bool = True) -> tuple[list[np.ndarray], dict]:
    """Ajusta los 10 modelos base sobre todo el train y devuelve sus scores de test.

    Returns
    -------
    tuple
        ``(lista_de_scores, detalle)`` donde ``detalle`` incluye la varianza explicada por la SVD.
    """
    started = time.time()
    scores: list[np.ndarray] = []
    detail: dict = {}

    def report(name: str) -> None:
        if verbose:
            print(f"[base] {name} ({round(time.time() - started, 1)}s)")

    model = LinearSVC(C=0.2, max_iter=3000).fit(representations["x300_train"], y)
    scores.append(to_scores(model.decision_function(representations["x300_test"]),
                            model.classes_, False))
    model = LinearSVC(C=0.2, max_iter=3000).fit(representations["x700_train"], y)
    scores.append(to_scores(model.decision_function(representations["x700_test"]),
                            model.classes_, False))
    report("svm 300k + svm 700k")

    model = OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=1000, solver="liblinear",
                                                   random_state=cfg.SEED))
    model.fit(representations["x300_train"], y)
    scores.append(to_scores(model.predict_proba(representations["x300_test"]), model.classes_, True))
    report("logreg OvR")

    model = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=30, tol=1e-4,
                          random_state=cfg.SEED).fit(representations["x300_train"], y)
    scores.append(to_scores(model.predict_proba(representations["x300_test"]), model.classes_, True))
    model = ComplementNB(alpha=0.3).fit(representations["x300_train"], y)
    scores.append(to_scores(model.predict_proba(representations["x300_test"]), model.classes_, True))

    model = LinearSVC(C=0.2, max_iter=3000).fit(representations["xh_train"], y)
    scores.append(to_scores(model.decision_function(representations["xh_test"]),
                            model.classes_, False))
    report("sgd + cnb + hashtags")

    ordinal_target = np.array([CLASS_MIDPOINTS[label] for label in y])
    ridge = Ridge(alpha=1.0, solver="sparse_cg", max_iter=2000, random_state=cfg.SEED)
    ridge.fit(representations["x300_train"], ordinal_target)
    continuous_test = ridge.predict(representations["x300_test"])
    scores.append(continuous_test.reshape(-1, 1))
    scores.append(np.log(np.clip(softmax_ordinal(continuous_test), 1e-12, None)))

    model = OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=1000, solver="liblinear",
                                                   random_state=cfg.SEED))
    model.fit(representations["xw_train"], y)
    scores.append(to_scores(model.predict_proba(representations["xw_test"]), model.classes_, True))
    report("ordinal + word")

    # MLP sobre densas + SVD del espacio de caracteres (miembro diverso, E-020).
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=cfg.SEED)
    z_train = svd.fit_transform(representations["x300_train"])
    z_test = svd.transform(representations["x300_test"])
    detail["svd_varianza_explicada"] = float(svd.explained_variance_ratio_.sum())
    scaler = StandardScaler().fit(np.hstack([dense_train, z_train]))
    encoder = LabelEncoder().fit(y)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, batch_size=512,
                        learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                        n_iter_no_change=6, random_state=cfg.SEED)
    mlp.fit(scaler.transform(np.hstack([dense_train, z_train])), encoder.transform(y))
    probabilities = mlp.predict_proba(scaler.transform(np.hstack([dense_test, z_test])))
    scores.append(np.log(np.clip(probabilities, 1e-12, None)))
    report("mlp densas+svd")

    # Miembro ordinal acumulativo (Frank & Hall): 5 clasificadores binarios sobre la representación.
    y_index = np.array([cfg.CLASSES.index(label) for label in y])
    ordinal_probabilities = ordinal_cumulative_probabilities(representations["x300_train"], y_index,
                                                            representations["x300_test"])
    scores.append(np.log(np.clip(ordinal_probabilities, 1e-12, None)))
    report("ordinal acumulativo")

    # Miembro de centroides (Rocchio): débil pero diverso.
    centroids = rocchio_probabilities(representations["x300_train"], y,
                                      representations["x300_test"])
    scores.append(np.log(np.clip(centroids, 1e-12, None)))
    report("centroides rocchio")

    # Supervisión ordinal fina: márgenes entre pares de rangos contiguos.
    pair_margins = adjacent_pair_margins(representations["x300_train"], y,
                                         representations["x300_test"])
    scores.append(pair_margins)
    report("pares adyacentes")

    return scores, detail


def fit_meta_model(test_scores: list[np.ndarray], dense_test: np.ndarray,
                   dense_train: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ajusta el meta-modelo con las matrices OOF y predice el test.

    Returns
    -------
    tuple
        ``(predicciones, matriz_de_features_del_meta)``.
    """
    x_meta_test = np.concatenate(test_scores + [dense_test], axis=1)
    oof_matrices = []
    for filename, probabilistic in OOF_FILES:
        matrix = np.load(cfg.FEATURES / filename)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        oof_matrices.append(np.log(np.clip(matrix, 1e-12, None)) if probabilistic else matrix)
    x_meta_train = np.concatenate(oof_matrices + [dense_train], axis=1)
    if x_meta_train.shape[1] != x_meta_test.shape[1]:
        raise ValueError(f"desalineación del meta-modelo: {x_meta_train.shape} vs {x_meta_test.shape}")

    meta = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                random_state=cfg.SEED))])
    meta.fit(x_meta_train, y)
    return meta.predict(x_meta_test), x_meta_test


def run(output_path: Path | str | None = None, verbose: bool = True) -> dict:
    """Ejecuta el pipeline completo y escribe la submission validada.

    Parameters
    ----------
    output_path : str or Path, optional
        Ruta de salida. Por defecto ``submissions/age_submission_final.csv``.

    Returns
    -------
    dict
        Resumen con dimensiones, checks de validación, distribución predicha y tiempos.
    """
    set_seed()
    started = time.time()
    output_path = Path(output_path) if output_path else cfg.SUBMISSIONS / "age_submission_final.csv"

    train = data.load_train()
    test = data.load_test()
    sample = data.load_sample_submission()
    y = train[cfg.TARGET].astype(str).to_numpy()
    dense_train = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")[DENSE_FEATURES].to_numpy(float)
    dense_test = pd.read_parquet(cfg.FEATURES / "dense_test.parquet")[DENSE_FEATURES].to_numpy(float)

    representations = build_representations(train, test)
    if verbose:
        print(f"representaciones: {representations['dimensiones']} "
              f"({round(time.time() - started, 1)}s)")

    test_scores, detail = fit_base_models(representations, y, dense_train, dense_test, verbose)
    predictions, x_meta = fit_meta_model(test_scores, dense_test, dense_train, y)

    candidate = pd.DataFrame({cfg.ID_COL: test[cfg.ID_COL].to_numpy(), cfg.TARGET: predictions})
    checks = submission.validate_submission(candidate, test[cfg.ID_COL], sample)
    candidate.to_csv(output_path, index=False)
    path = str(output_path)

    summary = {
        "modelo": "Ensamble de 10 fuentes + meta-modelo enriquecido (submission v06)",
        "cv_macro_f1_lofo": REFERENCE_CV_MACRO_F1,
        "cv_macro_f1_std": REFERENCE_CV_STD,
        "lb_publico_v06": 0.60273,
        "dimensiones": representations["dimensiones"],
        "meta_features": int(x_meta.shape[1]),
        "svd_varianza_explicada": round(detail.get("svd_varianza_explicada", float("nan")), 4),
        "checks": checks,
        "distribucion_predicha": candidate[cfg.TARGET].value_counts()
                                 .reindex(cfg.CLASSES).fillna(0).astype(int).to_dict(),
        "archivo": path,
        "segundos": round(time.time() - started, 1),
    }
    if verbose:
        print(json.dumps({k: summary[k] for k in ("checks", "distribucion_predicha", "segundos")},
                         indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
