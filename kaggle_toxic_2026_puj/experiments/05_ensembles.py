"""05_ensembles.py - Fase 13b: ensembles con evaluacion anidada y umbrales robustos.

Pregunta que responde
---------------------
1. Los cuatro modelos base (word (1,2), union word+char, LinearSVC y char_wb
   solo) son suficientemente diversos como para que un ensemble supere al mejor
   single ``MOD-03b_BEST`` (F1 OOF 0.7927 con umbral 0.5)?
2. Cual de las cuatro estrategias de ensemble (voto suave con pesos iguales,
   pesos ajustados por rejilla, stacking por etiqueta y mejor par/trio) generaliza
   mejor en una evaluacion ANIDADA (leave-one-fold-out)?
3. Que estrategia de umbral por etiqueta es mas robusta: 0.5 fijo, el optimo
   sobre el OOF completo (optimista) o la mediana de los optimos por pliegue?
4. Merece la pena generar una submission v04 con el ganador?

Metodo
------
- Split identico al de las Fases 4-5 y 13a:
  ``StratifiedGroupKFold(5, shuffle=True, random_state=42)`` con grupos =
  componentes de near-duplicado y estratificacion por la combinacion
  HS x TR x AG (D-004). Una sola particion para todos los modelos.
- Modelos base (vectorizador ajustado dentro de cada pliegue, sin fuga):
  * M1: TF-IDF word (1,2) min_df=2 sublinear + LR C=1.0 balanced (= B1/PRE-00).
  * M2: union word (1,2) + char_wb (3,5) + LR C=2.0 balanced (= MOD-03b_BEST).
  * M3: union + LinearSVC C=1.0 balanced, ``decision_function`` convertida a
    probabilidad con la sigmoide logistica (conversion monotona: umbral 0.5
    equivale exactamente a ``decision_function > 0``, que es la regla con la que
    la Fase 4-5 publico MOD-01_BEST).
  * M4: TF-IDF char_wb (3,5) + LR C=1.0 balanced (= REP-01).
- Replicacion: la F1 OOF a 0.5 y la dimension de cada base deben coincidir con
  el valor publicado en ``reports/variantes_representaciones.json`` (tolerancia
  1e-6 en F1); si M2 coincide con ``features/oof_mod03b.npz`` se reutiliza ese
  archivo.
- Evaluacion anidada: para cada pliegue de evaluacion ``k`` los pesos o el
  meta-modelo se ajustan SOLO con las probabilidades OOF de los otros cuatro
  pliegues y se aplican al pliegue ``k``. El valor anidado se agrega sobre las
  1731 filas (comparable con el 0.7927 publicado) y se reporta tambien la media
  y la desviacion por pliegue.
- LIMITACION DOCUMENTADA: las probabilidades base son OOF compartidas; la
  prediccion OOF de un pliegue proviene de modelos que vieron los otros pliegues,
  incluido el de evaluacion. Por eso la estimacion anidada del ensemble sigue
  siendo algo optimista frente al refit real con todo el train; el single se
  evalua con el MISMO protocolo para que la comparacion sea homogenea.
- Umbrales: rejilla 0.20-0.80 paso 0.01, estimados siempre sin el pliegue de
  evaluacion cuando la estrategia es anidada. El test nunca interviene.

Regla de decision de la submission
----------------------------------
Solo se reajusta y se genera ``submissions/toxic_submission_v04_<nombre>.csv`` si
el mejor candidato supera a ``MOD-03b_BEST`` (0.7926588861781015) por >= +0.005
absoluto en la evaluacion anidada. La submission se valida contra la plantilla
oficial (columnas, 1154 filas, ids identicos y en orden, valores en {0,1}, sin
nulos ni duplicados) antes de escribirse.

Artefactos
----------
- ``features/oof_bases.npz``: ids, y_true, etiquetas, nombres y proba OOF de
  M1..M4.
- ``reports/ensembles_diversidad.csv``: correlacion de errores, tasa de
  desacuerdo a 0.5 y Q-statistic por pares de bases.
- ``reports/ensembles_resultados.csv`` y ``reports/ensembles_resultados.json``:
  metricas anidadas, umbrales por estrategia y decision final.
- ``models/fase13b_<ganador>.joblib`` y la submission v04 solo si aplica.

Idempotencia
------------
Pliegues, probabilidades OOF, pesos, umbrales y decisiones son deterministas.
``experiments.csv``, ``submissions/leaderboard_log.csv`` y ``reports/contexto.md``
se actualizan eliminando primero las filas o la seccion de esta fase y
reescribiendolas, por lo que dos ejecuciones seguidas dejan el mismo contenido
salvo los campos de tiempo (ruido de maquina).

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/05_ensembles.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

CONFIG: Final[dict] = {
    "grupos_path": PROJECT_ROOT / "data" / "processed" / "grupos_near_duplicados.csv",
    "variantes_json": PROJECT_ROOT / "reports" / "variantes_representaciones.json",
    "oof_mod03b_npz": PROJECT_ROOT / "features" / "oof_mod03b.npz",
    "oof_bases_npz": PROJECT_ROOT / "features" / "oof_bases.npz",
    "diversidad_csv": PROJECT_ROOT / "reports" / "ensembles_diversidad.csv",
    "resultados_csv": PROJECT_ROOT / "reports" / "ensembles_resultados.csv",
    "resultados_json": PROJECT_ROOT / "reports" / "ensembles_resultados.json",
    "experiments_csv": PROJECT_ROOT / "experiments.csv",
    "leaderboard_csv": PROJECT_ROOT / "submissions" / "leaderboard_log.csv",
    "contexto_md": PROJECT_ROOT / "reports" / "contexto.md",
    "submissions_dir": PROJECT_ROOT / "submissions",
    "models_dir": PROJECT_ROOT / "models",
    "random_state": 42,
    "n_splits": 5,
    "umbral_base": 0.5,
    "rejilla_umbral": (0.20, 0.80, 0.01),
    "paso_pesos": 0.1,
    "tolerancia_replicacion": 1e-6,
    # Referencia del single publicado en reports/variantes_representaciones.json.
    "f1_mod03b_referencia": 0.7926588861781015,
    "id_mod03b": "MOD-03b_BEST",
    "umbral_mejora_submission": 0.005,
    "pliegues_minimos_mejora": 3,
    "n_filas_test": 1154,
    "columnas_submission": ("id", "HS", "TR", "AG"),
    "competencia": "KAGGLE_TOXIC_2026_PUJ",
    "fecha": "2026-09-22",
    "momento_contexto": "2026-09-22",
    "script": "experiments/05_ensembles.py",
    "id_submission": "S04",
}

# Parametros TF-IDF comunes (identicos a los de las Fases 4-5 y 13a).
TFIDF_COMUNES: Final[dict] = {
    "min_df": 2,
    "max_df": 1.0,
    "sublinear_tf": True,
    "lowercase": True,
    "strip_accents": None,
}

# Espacios TF-IDF usados por los modelos base. La clave es la representacion y el
# valor, la lista de bloques de ``TfidfVectorizer`` (mas de uno = FeatureUnion).
REPRESENTACIONES: Final[dict[str, tuple[dict, ...]]] = {
    "WORD_12": ({**TFIDF_COMUNES, "analyzer": "word", "ngram_range": (1, 2)},),
    "CHAR_35": ({**TFIDF_COMUNES, "analyzer": "char_wb", "ngram_range": (3, 5)},),
    "UNION": (
        {**TFIDF_COMUNES, "analyzer": "word", "ngram_range": (1, 2)},
        {**TFIDF_COMUNES, "analyzer": "char_wb", "ngram_range": (3, 5)},
    ),
}

# Experimento de la Fase 4-5 con el que se verifica cada representacion (F1 y
# dimension publicadas en ``reports/variantes_representaciones.json``).
REPRESENTACION_REFERENCIA: Final[dict[str, str]] = {
    "WORD_12": "PRE-00",
    "CHAR_35": "REP-01",
    "UNION": "MOD-03b_BEST",
}

COLUMNAS_DIVERSIDAD: Final[tuple[str, ...]] = (
    "modelo_a",
    "modelo_b",
    "correlacion_errores",
    "tasa_desacuerdo",
    "q_statistic",
    "n_celdas",
    "ambos_correctos",
    "ambos_incorrectos",
    "solo_a_correcto",
    "solo_b_correcto",
)

COLUMNAS_RESULTADOS: Final[tuple[str, ...]] = (
    "experiment_id",
    "ensemble",
    "descripcion",
    "f1_anidada_0.5",
    "f1_HS",
    "f1_TR",
    "f1_AG",
    "std_pliegues_0.5",
    "delta_vs_mod03b",
    "pliegues_mejorados",
    "delta_vs_single_anidado",
    "estrategia_umbrales",
    "f1_anidada_umbrales",
    "umbrales_finales",
    "decision",
)

COLUMNAS_EXPERIMENTS: Final[tuple[str, ...]] = (
    "experiment_id",
    "competition",
    "question",
    "hypothesis",
    "preprocessing",
    "feature_groups",
    "representation",
    "model",
    "hyperparameters",
    "validation",
    "f1",
    "macro_f1",
    "weighted_f1",
    "precision",
    "recall",
    "training_time",
    "dimension",
    "result",
    "interpretation",
    "decision",
    "date",
    "script",
    "notes",
)

COLUMNAS_LEADERBOARD: Final[tuple[str, ...]] = (
    "submission_id",
    "fecha",
    "archivo",
    "local_f1",
    "kaggle_f1",
    "diferencia",
    "features",
    "modelo",
    "hiperparametros",
    "interpretacion",
)

# Capacidad total del ensemble: word (1,2) 4024 + char_wb (3,5) 23407 + union
# 27431 columnas TF-IDF (la union es exactamente la concatenacion de los otros
# dos bloques, pero cada modelo tiene su propio vocabulario con pesos idf).
DIMENSION_ENSEMBLE: Final[int] = 4024 + 23407 + 27431


def log(message: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        message: texto a imprimir.
    """
    print(f"[05] {message}", flush=True)


# ===== SECCION 1: modelos base y utilidades de probabilidad =====


@dataclass(frozen=True)
class EspecBase:
    """Especificacion de un modelo base del ensemble.

    Attributes:
        identificador: id corto del modelo base (M1..M4).
        nombre: descripcion legible de representacion y clasificador.
        representacion: clave de :data:`REPRESENTACIONES`.
        tipo: ``logistic`` o ``svc``.
        parametros: hiperparametros del clasificador de scikit-learn.
        experimento_referencia: id del experimento de la Fase 4-5 con el que se
            verifica la replicacion de la F1 OOF.
        conversion: ``proba`` (``predict_proba``) o ``sigmoide`` (sigmoide de
            ``decision_function``).
    """

    identificador: str
    nombre: str
    representacion: str
    tipo: str
    parametros: dict
    experimento_referencia: str
    conversion: str


BASES: Final[tuple[EspecBase, ...]] = (
    EspecBase(
        identificador="M1",
        nombre="TF-IDF word (1,2) + LR C=1.0 balanced (B1)",
        representacion="WORD_12",
        tipo="logistic",
        parametros={
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "liblinear",
            "random_state": CONFIG["random_state"],
        },
        experimento_referencia="PRE-00",
        conversion="proba",
    ),
    EspecBase(
        identificador="M2",
        nombre="union word (1,2) + char_wb (3,5) + LR C=2.0 balanced (MOD-03b)",
        representacion="UNION",
        tipo="logistic",
        parametros={
            "C": 2.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "liblinear",
            "random_state": CONFIG["random_state"],
        },
        experimento_referencia="MOD-03b_BEST",
        conversion="proba",
    ),
    EspecBase(
        identificador="M3",
        nombre="union word (1,2) + char_wb (3,5) + LinearSVC C=1.0 balanced (MOD-01)",
        representacion="UNION",
        tipo="svc",
        parametros={
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 5000,
            "random_state": CONFIG["random_state"],
        },
        experimento_referencia="MOD-01_BEST",
        conversion="sigmoide",
    ),
    EspecBase(
        identificador="M4",
        nombre="TF-IDF char_wb (3,5) + LR C=1.0 balanced (REP-01)",
        representacion="CHAR_35",
        tipo="logistic",
        parametros={
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "liblinear",
            "random_state": CONFIG["random_state"],
        },
        experimento_referencia="REP-01",
        conversion="proba",
    ),
)


def construir_vectorizador(partes: Sequence[dict]):
    """Crea el vectorizador (o la union) de una representacion TF-IDF.

    Args:
        partes: parametros de ``TfidfVectorizer``; mas de uno construye una
            ``FeatureUnion`` que concatena matrices sparse.

    Returns:
        ``TfidfVectorizer`` si hay un solo bloque o ``FeatureUnion`` si hay varios.
    """
    if len(partes) == 1:
        return TfidfVectorizer(**partes[0])
    return FeatureUnion(
        [
            (f"bloque_{indice}", TfidfVectorizer(**parametros))
            for indice, parametros in enumerate(partes)
        ]
    )


def construir_modelo(espec: EspecBase):
    """Instancia un clasificador nuevo para un modelo base.

    Args:
        espec: especificacion del modelo base.

    Returns:
        Clasificador de scikit-learn sin ajustar.

    Raises:
        ValueError: si ``tipo`` no es uno de los soportados.
    """
    if espec.tipo == "logistic":
        return LogisticRegression(**espec.parametros)
    if espec.tipo == "svc":
        return LinearSVC(**espec.parametros)
    raise ValueError(f"Tipo de modelo base no soportado: {espec.tipo}")


def sigmoide(decisiones: np.ndarray) -> np.ndarray:
    """Convierte un ``decision_function`` en probabilidad con la sigmoide.

    Es una transformacion estrictamente monotona, por lo que el umbral 0.5
    reproduce exactamente la regla ``decision_function > 0`` con la que la Fase
    4-5 publico MOD-01_BEST. Se acota el argumento para evitar desbordamiento.

    Args:
        decisiones: vector de margenes del clasificador.

    Returns:
        Vector de probabilidades en ``(0, 1)``.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(decisiones, dtype=float), -500.0, 500.0)))


def probabilidades(modelo, matriz, espec: EspecBase) -> np.ndarray:
    """Devuelve la probabilidad de la clase positiva segun la conversion del base.

    Args:
        modelo: clasificador ajustado.
        matriz: matriz sparse de documentos.
        espec: especificacion del modelo base.

    Returns:
        Vector de probabilidades de la clase positiva.
    """
    if espec.conversion == "proba":
        return modelo.predict_proba(matriz)[:, 1]
    return sigmoide(modelo.decision_function(matriz))


def predecir_con_umbral(
    probabilidad: np.ndarray, umbrales: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Umbraliza una matriz de probabilidades columna a columna.

    Args:
        probabilidad: matriz ``(n, 3)`` con probabilidades de la clase positiva.
        umbrales: umbral por etiqueta, en el orden de :data:`ETIQUETAS`.

    Returns:
        Matriz ``(n, 3)`` con valores en ``{0, 1}``.
    """
    return (probabilidad >= np.asarray(umbrales, dtype=float)[None, :]).astype(int)


def f1_medio_columnas(y_true: np.ndarray, prediccion: np.ndarray) -> float:
    """Calcula la metrica primaria del proyecto (F1 medio sobre las 3 columnas).

    Args:
        y_true: matriz binaria verdadera ``(n, 3)``.
        prediccion: matriz binaria predicha ``(n, 3)``.

    Returns:
        Valor de ``f1_media_columnas``.
    """
    return float(metricas_multietiqueta(y_true, prediccion)["f1_media_columnas"])


def f1_binaria_rapida(verdad: np.ndarray, prediccion: np.ndarray) -> float:
    """F1 binaria de la clase positiva con conteos de confusion directos.

    Es matematicamente equivalente a ``f1_score(verdad, prediccion,
    zero_division=0)`` y se usa en las busquedas de pesos y de umbrales, donde la
    version de scikit-learn resulta demasiado costosa por el numero de
    evaluaciones. La equivalencia se verifica contra
    :func:`metricas_multietiqueta` al inicio de la fase.

    Args:
        verdad: vector binario verdadero.
        prediccion: vector binario predicho.

    Returns:
        F1 de la clase positiva (0.0 si no hay positivos verdaderos ni predichos).
    """
    verdad = np.asarray(verdad).astype(int)
    prediccion = np.asarray(prediccion).astype(int)
    verdaderos_positivos = float(np.count_nonzero((verdad == 1) & (prediccion == 1)))
    falsos_positivos = float(np.count_nonzero((verdad == 0) & (prediccion == 1)))
    falsos_negativos = float(np.count_nonzero((verdad == 1) & (prediccion == 0)))
    denominador = 2.0 * verdaderos_positivos + falsos_positivos + falsos_negativos
    if denominador == 0.0:
        return 0.0
    return 2.0 * verdaderos_positivos / denominador


def f1_columnas_rapido(y_true: np.ndarray, prediccion: np.ndarray) -> float:
    """Media de la F1 binaria positiva por columna (version rapida).

    Args:
        y_true: matriz binaria verdadera ``(n, 3)``.
        prediccion: matriz binaria predicha ``(n, 3)``.

    Returns:
        Media de las F1 binarias de la clase positiva, equivalente a
        ``f1_media_columnas``.
    """
    return float(
        np.mean(
            [
                f1_binaria_rapida(y_true[:, indice], prediccion[:, indice])
                for indice in range(y_true.shape[1])
            ]
        )
    )


def verificar_f1_rapida(
    y_true: np.ndarray, probabilidad: np.ndarray, umbrales: Sequence[float]
) -> dict:
    """Comprueba que la F1 rapida coincide con la metrica canonica del proyecto.

    Args:
        y_true: matriz binaria verdadera ``(n, 3)``.
        probabilidad: matriz ``(n, 3)`` de probabilidades.
        umbrales: umbral por etiqueta para binarizar.

    Returns:
        Diccionario con ambos valores y la diferencia.

    Raises:
        AssertionError: si las dos implementaciones no coinciden.
    """
    prediccion = predecir_con_umbral(probabilidad, umbrales)
    rapida = f1_columnas_rapido(y_true, prediccion)
    canonica = f1_medio_columnas(y_true, prediccion)
    diferencia = rapida - canonica
    assert abs(diferencia) <= 1e-12, (
        "La F1 rapida no coincide con metricas_multietiqueta: "
        f"{rapida!r} vs {canonica!r}"
    )
    return {"f1_rapida": rapida, "f1_canonica": canonica, "diferencia": diferencia}


def evaluar_prediccion(
    y_true: np.ndarray,
    prediccion: np.ndarray,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Resume las metricas de una prediccion completa y por pliegue.

    Args:
        y_true: matriz binaria verdadera ``(n, 3)``.
        prediccion: matriz binaria predicha ``(n, 3)``.
        pliegues: particion de validacion para el desglose por pliegue.

    Returns:
        Diccionario con ``f1_media_columnas``, ``f1_por_etiqueta``, ``macro6``,
        ``por_pliegue`` y ``std_pliegues`` (desviacion entre pliegues con ddof=0,
        la convencion de las Fases 4-5).
    """
    metricas = metricas_multietiqueta(y_true, prediccion)
    por_pliegue = [
        f1_medio_columnas(y_true[validacion], prediccion[validacion])
        for _, validacion in pliegues
    ]
    return {
        "f1_media_columnas": float(metricas["f1_media_columnas"]),
        "f1_por_etiqueta": dict(metricas["f1_por_etiqueta"]),
        "macro6": float(metricas["f1_macro6"]),
        "por_pliegue": por_pliegue,
        "std_pliegues": float(np.std(por_pliegue)),
    }


# ===== SECCION 2: carga de datos, particion y referencias =====


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Carga train, test y la asignacion de grupos de near-duplicado.

    Returns:
        Tupla ``(train, test, grupos)``; ``grupos`` es una serie indexada por
        ``id`` con el numero de grupo de near-duplicado.

    Raises:
        FileNotFoundError: si falta la particion generada en la Fase 0.
    """
    train = data_module.read_train()
    test = data_module.read_test()
    grupos_path: Path = CONFIG["grupos_path"]
    if not grupos_path.exists():
        raise FileNotFoundError(
            f"Falta {grupos_path}. Ejecuta experiments/00_auditoria_datos.py primero."
        )
    grupos = pd.read_csv(grupos_path).set_index("id")["grupo"]
    return train, test, grupos


def construir_pliegues(
    train: pd.DataFrame, grupos: pd.Series
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construye la particion 5-fold agrupada y estratificada (D-004).

    Replica literalmente la particion de las Fases 4-5 y 13a para que todas las
    metricas de esta fase sean comparables con el ranking publicado.

    Args:
        train: DataFrame de entrenamiento con ``id``, ``text`` y etiquetas.
        grupos: serie de grupos de near-duplicado indexada por ``id``.

    Returns:
        Lista de 5 tuplas ``(indices_entrenamiento, indices_validacion)`` con
        indices posicionales.

    Raises:
        ValueError: si algun ``id`` de train no tiene grupo asignado.
    """
    combinacion = (
        train["HS"].astype(str) + train["TR"].astype(str) + train["AG"].astype(str)
    )
    asignacion = train["id"].astype(int).map(grupos)
    if asignacion.isna().any():
        faltantes = int(asignacion.isna().sum())
        raise ValueError(
            f"{faltantes} ids de train sin grupo de near-duplicado; "
            "regenera data/processed/grupos_near_duplicados.csv"
        )
    divisor = StratifiedGroupKFold(
        n_splits=CONFIG["n_splits"],
        shuffle=True,
        random_state=CONFIG["random_state"],
    )
    pliegues = [
        (entrenamiento, validacion)
        for entrenamiento, validacion in divisor.split(
            train, combinacion, groups=asignacion.to_numpy()
        )
    ]
    for numero, (entrenamiento, validacion) in enumerate(pliegues, start=1):
        log(
            f"pliegue {numero}: {len(entrenamiento)} train / {len(validacion)} validacion"
        )
    return pliegues


def leer_referencias() -> dict[str, dict]:
    """Lee la F1 OOF y la dimension publicadas por la Fase 4-5.

    Returns:
        Diccionario ``experiment_id -> {"f1_media_columnas", "dimension"}``.

    Raises:
        FileNotFoundError: si falta ``reports/variantes_representaciones.json``.
    """
    ruta: Path = CONFIG["variantes_json"]
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta}; ejecuta el script 03 primero.")
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    return {
        configuracion["experiment_id"]: {
            "f1_media_columnas": float(configuracion["metricas_oof"]["f1_media_columnas"]),
            "dimension": int(configuracion["dimension"]),
        }
        for configuracion in datos["configuraciones"]
    }


# ===== SECCION 3: probabilidades out-of-fold de los modelos base =====


def calcular_oof_bases(
    train: pd.DataFrame, pliegues: Sequence[tuple[np.ndarray, np.ndarray]]
) -> dict[str, np.ndarray]:
    """Calcula las probabilidades OOF de los cuatro modelos base.

    El vectorizador se ajusta SIEMPRE con el pliegue de entrenamiento y se
    transforma el de validacion. Los dos modelos que comparten la union
    word+char reutilizan la misma matriz por pliegue: el vectorizador es
    determinista, de modo que compartirlo no altera ningun resultado y ahorra
    tiempo.

    Args:
        train: DataFrame de entrenamiento con ``text`` y etiquetas.
        pliegues: particion agrupada compartida.

    Returns:
        Diccionario ``identificador -> matriz (n, 3)`` de probabilidades OOF.
    """
    probabilidad = {
        espec.identificador: np.zeros((len(train), len(ETIQUETAS)), dtype=float)
        for espec in BASES
    }
    for entrenamiento, validacion in pliegues:
        matrices: dict[str, tuple[Any, Any]] = {}
        for clave, partes in REPRESENTACIONES.items():
            vectorizador = construir_vectorizador(partes)
            matrices[clave] = (
                vectorizador.fit_transform(train["text"].iloc[entrenamiento]),
                vectorizador.transform(train["text"].iloc[validacion]),
            )
        for espec in BASES:
            matriz_entrenamiento, matriz_validacion = matrices[espec.representacion]
            for indice, etiqueta in enumerate(ETIQUETAS):
                modelo = construir_modelo(espec)
                modelo.fit(
                    matriz_entrenamiento, train[etiqueta].iloc[entrenamiento]
                )
                probabilidad[espec.identificador][validacion, indice] = probabilidades(
                    modelo, matriz_validacion, espec
                )
    return probabilidad


def sincronizar_oof_mod03b(
    train: pd.DataFrame, probabilidad_m2: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Reutiliza ``features/oof_mod03b.npz`` si coincide con M2 recalculado.

    Args:
        train: DataFrame de entrenamiento (aporta los ids).
        probabilidad_m2: matriz ``(n, 3)`` recalculada para M2.

    Returns:
        Tupla ``(probabilidad, detalle)``; ``detalle`` documenta si se reutilizo
        el archivo y la diferencia maxima observada.
    """
    ruta: Path = CONFIG["oof_mod03b_npz"]
    if not ruta.exists():
        return probabilidad_m2, {"reutilizado": False, "motivo": "no existe el npz previo"}
    with np.load(ruta, allow_pickle=False) as datos:
        ids_guardados = [str(valor) for valor in datos["ids"]]
        proba_guardada = datos["proba"]
    ids_actuales = [str(valor) for valor in train["id"]]
    if ids_guardados != ids_actuales or proba_guardada.shape != probabilidad_m2.shape:
        return probabilidad_m2, {
            "reutilizado": False,
            "motivo": "ids o forma distintos entre el npz previo y el recalculo",
        }
    diferencia = float(np.max(np.abs(proba_guardada - probabilidad_m2)))
    coincide = diferencia <= 1e-12
    detalle = {
        "reutilizado": coincide,
        "diferencia_max_abs": diferencia,
        "motivo": (
            "el recalculo coincide con el npz de la Fase 13a"
            if coincide
            else "el recalculo difiere del npz previo; se usa el recalculado"
        ),
    }
    return (proba_guardada if coincide else probabilidad_m2), detalle


def guardar_oof_bases(train: pd.DataFrame, probabilidad: dict[str, np.ndarray]) -> Path:
    """Guarda las probabilidades OOF de las cuatro bases en un ``.npz``.

    Args:
        train: DataFrame de entrenamiento (aporta ``id`` y etiquetas).
        probabilidad: diccionario de matrices ``(n, 3)`` por modelo base.

    Returns:
        Ruta del archivo escrito.

    Raises:
        ValueError: si el archivo releido no tiene las formas esperadas.
    """
    destino: Path = CONFIG["oof_bases_npz"]
    destino.parent.mkdir(parents=True, exist_ok=True)
    contenido: dict[str, np.ndarray] = {
        "ids": np.asarray([str(valor) for valor in train["id"]], dtype=str),
        "y_true": train[list(ETIQUETAS)].to_numpy(),
        "etiquetas": np.asarray(ETIQUETAS),
        "modelos": np.asarray([espec.identificador for espec in BASES]),
    }
    for espec in BASES:
        contenido[f"proba_{espec.identificador}"] = probabilidad[espec.identificador]
    np.savez(destino, **contenido)
    with np.load(destino, allow_pickle=False) as releido:
        formas = {clave: releido[clave].shape for clave in releido.files}
    esperadas = {
        "ids": (len(train),),
        "y_true": (len(train), len(ETIQUETAS)),
        "etiquetas": (len(ETIQUETAS),),
        "modelos": (len(BASES),),
        **{f"proba_{espec.identificador}": (len(train), len(ETIQUETAS)) for espec in BASES},
    }
    if formas != esperadas:
        raise ValueError(f"El .npz guardado tiene formas inesperadas: {formas}")
    return destino


def verificar_bases(
    y_true: np.ndarray, probabilidad: dict[str, np.ndarray], referencias: dict[str, dict]
) -> dict[str, dict]:
    """Comprueba que cada base replica la F1 OOF publicada por la Fase 4-5.

    Args:
        y_true: matriz ``(n, 3)`` con las etiquetas verdaderas.
        probabilidad: probabilidades OOF por modelo base.
        referencias: valores publicados por :func:`leer_referencias`.

    Returns:
        Diccionario con la F1 reproducida, la referencia y la diferencia.

    Raises:
        AssertionError: si alguna base no reproduce su referencia dentro de la
            tolerancia declarada.
    """
    detalle: dict[str, dict] = {}
    for espec in BASES:
        prediccion = predecir_con_umbral(
            probabilidad[espec.identificador], [CONFIG["umbral_base"]] * len(ETIQUETAS)
        )
        f1_reproducida = f1_medio_columnas(y_true, prediccion)
        referencia = float(referencias[espec.experimento_referencia]["f1_media_columnas"])
        diferencia = f1_reproducida - referencia
        log(
            f"{espec.identificador} ({espec.experimento_referencia}): F1 OOF "
            f"{f1_reproducida:.6f} vs {referencia:.6f} (diferencia {diferencia:+.3e})"
        )
        assert abs(diferencia) <= CONFIG["tolerancia_replicacion"], (
            f"{espec.identificador} no replica {espec.experimento_referencia}: "
            f"{f1_reproducida!r} vs {referencia!r}"
        )
        detalle[espec.identificador] = {
            "nombre": espec.nombre,
            "representacion": espec.representacion,
            "conversion": espec.conversion,
            "experimento_referencia": espec.experimento_referencia,
            "f1_oof_0.5": f1_reproducida,
            "f1_referencia": referencia,
            "diferencia": diferencia,
            "tolerancia": CONFIG["tolerancia_replicacion"],
        }
    return detalle


def verificar_dimensiones(train: pd.DataFrame, referencias: dict[str, dict]) -> dict[str, dict]:
    """Comprueba que las dimensiones TF-IDF coinciden con las publicadas.

    Args:
        train: DataFrame de entrenamiento con ``text``.
        referencias: valores publicados por :func:`leer_referencias`.

    Returns:
        Diccionario ``representacion -> {"dimension", "referencia", "diferencia"}``.

    Raises:
        AssertionError: si alguna dimension publicada no coincide.
    """
    detalle: dict[str, dict] = {}
    for clave, partes in REPRESENTACIONES.items():
        vectorizador = construir_vectorizador(partes)
        dimension = int(vectorizador.fit_transform(train["text"]).shape[1])
        experimento = REPRESENTACION_REFERENCIA[clave]
        referencia = int(referencias[experimento]["dimension"])
        log(
            f"dimension {clave}: {dimension} vs {referencia} ({experimento})"
        )
        assert dimension == referencia, (
            f"La dimension de {clave} no coincide con {experimento}: "
            f"{dimension} vs {referencia}"
        )
        detalle[clave] = {
            "dimension": dimension,
            "experimento_referencia": experimento,
            "referencia": referencia,
            "diferencia": dimension - referencia,
        }
    return detalle


# ===== SECCION 4: analisis de diversidad =====


def analizar_diversidad(
    probabilidad: dict[str, np.ndarray], y_true: np.ndarray
) -> tuple[pd.DataFrame, list[dict]]:
    """Calcula diversidad por pares de los modelos base (a umbral 0.5).

    Se trabaja sobre las ``n * 3`` celdas (fila x etiqueta) para no perder la
    informacion por etiqueta. La correlacion de errores es la de Pearson entre
    los vectores de error (predicho distinto de verdadero); la tasa de desacuerdo
    es la fraccion de celdas con prediccion distinta; el Q-statistic es
    ``(N11*N00 - N01*N10) / (N11*N00 + N01*N10)`` con ``N11`` celdas donde ambos
    aciertan, ``N00`` donde ambos fallan, ``N01`` donde solo acierta el primero
    y ``N10`` donde solo acierta el segundo. Si el denominador es cero o la
    correlacion no esta definida (vector de error constante), se reporta 0.0.

    Args:
        probabilidad: probabilidades OOF por modelo base.
        y_true: matriz ``(n, 3)`` con las etiquetas verdaderas.

    Returns:
        Tupla ``(tabla, resumen)`` donde ``tabla`` es el DataFrame de pares y
        ``resumen`` la lista de metricas individuales por modelo.
    """
    identificadores = [espec.identificador for espec in BASES]
    predicciones = {
        espec.identificador: predecir_con_umbral(
            probabilidad[espec.identificador], [CONFIG["umbral_base"]] * len(ETIQUETAS)
        )
        for espec in BASES
    }
    errores = {
        identificador: (predicciones[identificador] != y_true).ravel()
        for identificador in identificadores
    }
    filas: list[dict] = []
    for modelo_a, modelo_b in itertools.combinations(identificadores, 2):
        error_a = errores[modelo_a]
        error_b = errores[modelo_b]
        desviacion_a = float(np.std(error_a.astype(float)))
        desviacion_b = float(np.std(error_b.astype(float)))
        if desviacion_a > 0 and desviacion_b > 0:
            correlacion = float(np.corrcoef(error_a.astype(float), error_b.astype(float))[0, 1])
        else:
            correlacion = 0.0
        prediccion_a = predicciones[modelo_a].ravel()
        prediccion_b = predicciones[modelo_b].ravel()
        verdad = y_true.ravel()
        acierto_a = prediccion_a == verdad
        acierto_b = prediccion_b == verdad
        n11 = int(np.sum(acierto_a & acierto_b))
        n00 = int(np.sum(~acierto_a & ~acierto_b))
        n01 = int(np.sum(acierto_a & ~acierto_b))
        n10 = int(np.sum(~acierto_a & acierto_b))
        denominador = n11 * n00 + n01 * n10
        q_statistic = (
            float((n11 * n00 - n01 * n10) / denominador) if denominador else 0.0
        )
        filas.append(
            {
                "modelo_a": modelo_a,
                "modelo_b": modelo_b,
                "correlacion_errores": round(correlacion, 6),
                "tasa_desacuerdo": round(float(np.mean(prediccion_a != prediccion_b)), 6),
                "q_statistic": round(q_statistic, 6),
                "n_celdas": int(verdad.size),
                "ambos_correctos": n11,
                "ambos_incorrectos": n00,
                "solo_a_correcto": n01,
                "solo_b_correcto": n10,
            }
        )
    resumen = [
        {
            "identificador": espec.identificador,
            "nombre": espec.nombre,
            "f1_media_columnas_0.5": f1_medio_columnas(y_true, predicciones[espec.identificador]),
            "tasa_error_celdas": round(float(np.mean(errores[espec.identificador])), 6),
        }
        for espec in BASES
    ]
    return pd.DataFrame(filas, columns=list(COLUMNAS_DIVERSIDAD)), resumen


# ===== SECCION 5: evaluacion anidada de ensembles =====


def combinar_pesos(proba: np.ndarray, pesos: Sequence[float]) -> np.ndarray:
    """Promedio ponderado de las probabilidades base.

    Args:
        proba: tensor ``(n, n_modelos, 3)`` de probabilidades.
        pesos: peso por modelo.

    Returns:
        Matriz ``(n, 3)`` con el promedio ponderado por etiqueta.
    """
    return np.tensordot(np.asarray(pesos, dtype=float), proba, axes=(0, 1))


def combinaciones_pesos(n_modelos: int, paso: float) -> list[tuple[float, ...]]:
    """Enumera la rejilla simplex de pesos con paso fijo.

    Args:
        n_modelos: numero de modelos base.
        paso: incremento de cada peso (deben sumar 1).

    Returns:
        Lista de tuplas de pesos en orden lexicografico determinista.
    """
    unidades = round(1.0 / paso)
    combinaciones: list[tuple[int, ...]] = []

    def _recursivo(prefijo: tuple[int, ...], restante: int, restantes: int) -> None:
        if restantes == 1:
            combinaciones.append(prefijo + (restante,))
            return
        for valor in range(restante + 1):
            _recursivo(prefijo + (valor,), restante - valor, restantes - 1)

    _recursivo((), unidades, n_modelos)
    return [tuple(unidad * paso for unidad in combinacion) for combinacion in combinaciones]


def candidatos_par_trio(n_modelos: int) -> list[tuple[int, ...]]:
    """Enumera los pares y los tríos de modelos base.

    Args:
        n_modelos: numero de modelos base.

    Returns:
        Lista determinista de combinaciones (primero los pares, luego los tríos).
    """
    indices = range(n_modelos)
    return list(itertools.combinations(indices, 2)) + list(itertools.combinations(indices, 3))


def pesos_de_combinacion(combinacion: Sequence[int], n_modelos: int) -> tuple[float, ...]:
    """Construye pesos iguales sobre los modelos de una combinacion.

    Args:
        combinacion: indices de los modelos seleccionados.
        n_modelos: numero total de modelos base.

    Returns:
        Tupla de pesos que suma 1.
    """
    peso = 1.0 / len(combinacion)
    return tuple(peso if indice in combinacion else 0.0 for indice in range(n_modelos))


def ajustar_pesos_iguales(proba: np.ndarray, y_true: np.ndarray) -> dict:
    """Pesos iguales sobre todos los modelos (ENS-01, sin ajuste real).

    Args:
        proba: tensor ``(n, n_modelos, 3)`` de probabilidades de ajuste.
        y_true: etiquetas verdaderas del conjunto de ajuste.

    Returns:
        Diccionario con los pesos.
    """
    n_modelos = proba.shape[1]
    return {"pesos": tuple([1.0 / n_modelos] * n_modelos)}


def ajustar_pesos_rejilla(proba: np.ndarray, y_true: np.ndarray) -> dict:
    """Busca los pesos del simplex que maximizan la F1 media a 0.5.

    La busqueda se hace SOLO con las probabilidades del conjunto de ajuste. El
    objetivo es la metrica primaria con umbral 0.5; los umbrales se optimizan
    despues, en una etapa separada, para no mezclar ambas decisiones.

    Args:
        proba: tensor ``(n, n_modelos, 3)`` de probabilidades de ajuste.
        y_true: etiquetas verdaderas del conjunto de ajuste.

    Returns:
        Diccionario con los pesos elegidos y la F1 alcanzada en el ajuste.
    """
    mejor_pesos = tuple([1.0 / proba.shape[1]] * proba.shape[1])
    mejor_valor = -1.0
    for pesos in combinaciones_pesos(proba.shape[1], CONFIG["paso_pesos"]):
        prediccion = predecir_con_umbral(
            combinar_pesos(proba, pesos), [CONFIG["umbral_base"]] * len(ETIQUETAS)
        )
        valor = f1_columnas_rapido(y_true, prediccion)
        if valor > mejor_valor:
            mejor_pesos = pesos
            mejor_valor = valor
    return {"pesos": mejor_pesos, "f1_ajuste": mejor_valor}


def ajustar_mejor_combinacion(proba: np.ndarray, y_true: np.ndarray) -> dict:
    """Selecciona el mejor par o trio de modelos con el conjunto de ajuste.

    Args:
        proba: tensor ``(n, n_modelos, 3)`` de probabilidades de ajuste.
        y_true: etiquetas verdaderas del conjunto de ajuste.

    Returns:
        Diccionario con la combinacion, los pesos iguales derivados y la F1
        alcanzada en el ajuste.
    """
    mejor_combinacion = tuple(range(min(2, proba.shape[1])))
    mejor_valor = -1.0
    for combinacion in candidatos_par_trio(proba.shape[1]):
        pesos = pesos_de_combinacion(combinacion, proba.shape[1])
        prediccion = predecir_con_umbral(
            combinar_pesos(proba, pesos), [CONFIG["umbral_base"]] * len(ETIQUETAS)
        )
        valor = f1_columnas_rapido(y_true, prediccion)
        if valor > mejor_valor:
            mejor_combinacion = combinacion
            mejor_valor = valor
    return {
        "combinacion": mejor_combinacion,
        "pesos": pesos_de_combinacion(mejor_combinacion, proba.shape[1]),
        "f1_ajuste": mejor_valor,
    }


def ajustar_stacking(proba: np.ndarray, y_true: np.ndarray) -> dict:
    """Ajusta un meta-modelo de regresion logistica por etiqueta.

    Args:
        proba: tensor ``(n, n_modelos, 3)`` de probabilidades de ajuste.
        y_true: etiquetas verdaderas del conjunto de ajuste.

    Returns:
        Diccionario con la lista de meta-modelos (uno por etiqueta).
    """
    meta_modelos = []
    for indice in range(len(ETIQUETAS)):
        meta = LogisticRegression(
            C=1.0, max_iter=1000, random_state=CONFIG["random_state"]
        )
        meta.fit(proba[:, :, indice], y_true[:, indice])
        meta_modelos.append(meta)
    return {"meta_modelos": meta_modelos}


def aplicar_stacking(params: dict, proba: np.ndarray) -> np.ndarray:
    """Aplica los meta-modelos de stacking a un bloque de probabilidades.

    Args:
        params: diccionario devuelto por :func:`ajustar_stacking`.
        proba: tensor ``(n, n_modelos, 3)`` de probabilidades a transformar.

    Returns:
        Matriz ``(n, 3)`` con la probabilidad del meta-modelo por etiqueta.
    """
    return np.column_stack(
        [
            meta.predict_proba(proba[:, :, indice])[:, 1]
            for indice, meta in enumerate(params["meta_modelos"])
        ]
    )


def resumir_pesos(params: dict) -> dict:
    """Resume pesos para el JSON (redondeados a 4 decimales).

    Args:
        params: diccionario del ajuste.

    Returns:
        Diccionario serializable.
    """
    return {"pesos": [round(float(peso), 4) for peso in params["pesos"]]}


def resumir_combinacion(params: dict) -> dict:
    """Resume la combinacion elegida y sus pesos.

    Args:
        params: diccionario del ajuste.

    Returns:
        Diccionario serializable.
    """
    return {
        "combinacion": [
            espec.identificador
            for posicion, espec in enumerate(BASES)
            if params["pesos"][posicion] > 0
        ],
        "pesos": [round(float(peso), 4) for peso in params["pesos"]],
    }


def resumir_stacking(params: dict) -> dict:
    """Resume los coeficientes del stacking por etiqueta.

    Args:
        params: diccionario del ajuste.

    Returns:
        Diccionario serializable con coeficientes e interceptos por etiqueta.
    """
    return {
        "coeficientes": {
            etiqueta: [round(float(valor), 6) for valor in params["meta_modelos"][indice].coef_[0]]
            for indice, etiqueta in enumerate(ETIQUETAS)
        },
        "interceptos": {
            etiqueta: round(float(params["meta_modelos"][indice].intercept_[0]), 6)
            for indice, etiqueta in enumerate(ETIQUETAS)
        },
    }


@dataclass(frozen=True)
class DefinicionEnsemble:
    """Definicion de una estrategia de ensemble.

    Attributes:
        identificador: id del experimento (ENS-01..ENS-04).
        nombre: descripcion corta.
        descripcion: descripcion metodologica para los reportes.
        sufijo_submission: sufijo del archivo de submission si resulta ganador.
        ajustar: funcion ``(proba, y_true) -> params`` que ajusta el ensemble con
            los pliegues de ajuste (o con todo el OOF en el refit final).
        aplicar: funcion ``(params, proba) -> proba_ensemble``.
        resumir: funcion ``params -> dict`` serializable para el JSON.
    """

    identificador: str
    nombre: str
    descripcion: str
    sufijo_submission: str
    ajustar: Callable[[np.ndarray, np.ndarray], dict]
    aplicar: Callable[[dict, np.ndarray], np.ndarray]
    resumir: Callable[[dict], dict]


@dataclass(frozen=True)
class ResultadoEnsemble:
    """Resultado de la evaluacion anidada de un ensemble.

    Attributes:
        definicion: definicion del ensemble.
        proba_anidada: matriz ``(n, 3)`` con la probabilidad anidada.
        metricas: resumen de :func:`evaluar_prediccion` a umbral 0.5.
        detalle_pliegues: parametros ajustados y F1 de cada pliegue.
        segundos: tiempo de la evaluacion anidada.
    """

    definicion: DefinicionEnsemble
    proba_anidada: np.ndarray
    metricas: dict
    detalle_pliegues: list[dict]
    segundos: float


def definiciones_ensemble() -> tuple[DefinicionEnsemble, ...]:
    """Construye las cuatro definiciones de ensemble de la fase.

    Returns:
        Tupla con ENS-01 (voto suave igual), ENS-02 (pesos por rejilla),
        ENS-03 (stacking por etiqueta) y ENS-04 (mejor par/trio).
    """
    return (
        DefinicionEnsemble(
            identificador="ENS-01",
            nombre="voto suave con pesos iguales (M1+M2+M3+M4)",
            descripcion=(
                "promedio simple de las probabilidades OOF de los cuatro modelos base; "
                "sin parametros ajustados"
            ),
            sufijo_submission="ens01_voto_suave_igual",
            ajustar=ajustar_pesos_iguales,
            aplicar=lambda params, proba: combinar_pesos(proba, params["pesos"]),
            resumir=resumir_pesos,
        ),
        DefinicionEnsemble(
            identificador="ENS-02",
            nombre="voto suave con pesos en rejilla simplex (paso 0.1)",
            descripcion=(
                "pesos del simplex {0, 0.1, ..., 1.0} elegidos maximizando la F1 media "
                "a 0.5 SOLO con los pliegues de ajuste"
            ),
            sufijo_submission="ens02_pesos_rejilla",
            ajustar=ajustar_pesos_rejilla,
            aplicar=lambda params, proba: combinar_pesos(proba, params["pesos"]),
            resumir=resumir_pesos,
        ),
        DefinicionEnsemble(
            identificador="ENS-03",
            nombre="stacking por etiqueta con LR sobre las 4 probabilidades OOF",
            descripcion=(
                "meta-modelo LogisticRegression(C=1.0) por etiqueta con las 4 "
                "probabilidades OOF como features; ajustado solo con los pliegues de ajuste"
            ),
            sufijo_submission="ens03_stacking_lr",
            ajustar=ajustar_stacking,
            aplicar=aplicar_stacking,
            resumir=resumir_stacking,
        ),
        DefinicionEnsemble(
            identificador="ENS-04",
            nombre="mejor par/trio de bases seleccionado de forma anidada",
            descripcion=(
                "voto suave con pesos iguales sobre el mejor par o trio (de los 10 "
                "candidatos) elegido con los pliegues de ajuste"
            ),
            sufijo_submission="ens04_mejor_par_trio",
            ajustar=ajustar_mejor_combinacion,
            aplicar=lambda params, proba: combinar_pesos(proba, params["pesos"]),
            resumir=resumir_combinacion,
        ),
    )


def evaluar_anidado(
    definicion: DefinicionEnsemble,
    proba_base: np.ndarray,
    y_true: np.ndarray,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
    f1_referencia_pliegues: Sequence[float],
) -> ResultadoEnsemble:
    """Evalua un ensemble dejando fuera el pliegue de evaluacion.

    Para cada pliegue ``k`` los parametros se ajustan con las probabilidades OOF
    de los otros cuatro pliegues y se aplican al pliegue ``k``. Los parametros
    ajustados y la F1 de cada pliegue quedan registrados para auditar la
    estabilidad de la estrategia.

    Args:
        definicion: definicion del ensemble.
        proba_base: tensor ``(n, n_modelos, 3)`` de probabilidades OOF.
        y_true: matriz ``(n, 3)`` con las etiquetas verdaderas.
        pliegues: particion agrupada compartida.
        f1_referencia_pliegues: F1 por pliegue del single de referencia, para
            contar en cuantos pliegues mejora el ensemble.

    Returns:
        Resultado anidado del ensemble.
    """
    inicio = time.perf_counter()
    n_filas = proba_base.shape[0]
    proba_anidada = np.zeros((n_filas, len(ETIQUETAS)), dtype=float)
    detalle_pliegues: list[dict] = []
    for numero, (_, validacion) in enumerate(pliegues, start=1):
        mascara_ajuste = np.ones(n_filas, dtype=bool)
        mascara_ajuste[validacion] = False
        parametros = definicion.ajustar(
            proba_base[mascara_ajuste], y_true[mascara_ajuste]
        )
        proba_anidada[validacion] = definicion.aplicar(parametros, proba_base[validacion])
        detalle_pliegues.append(
            {
                "pliegue": numero,
                "n_ajuste": int(mascara_ajuste.sum()),
                "n_validacion": len(validacion),
                "parametros": definicion.resumir(parametros),
                "f1_ajuste": (
                    None
                    if parametros.get("f1_ajuste") is None
                    else round(float(parametros["f1_ajuste"]), 6)
                ),
                "f1_validacion_0.5": round(
                    f1_medio_columnas(
                        y_true[validacion],
                        predecir_con_umbral(
                            proba_anidada[validacion], [CONFIG["umbral_base"]] * len(ETIQUETAS)
                        ),
                    ),
                    6,
                ),
            }
        )
    metricas = evaluar_prediccion(
        y_true,
        predecir_con_umbral(proba_anidada, [CONFIG["umbral_base"]] * len(ETIQUETAS)),
        pliegues,
    )
    metricas["pliegues_mejorados"] = int(
        sum(
            valor > referencia
            for valor, referencia in zip(
                metricas["por_pliegue"], f1_referencia_pliegues, strict=True
            )
        )
    )
    log(
        f"{definicion.identificador}: F1 anidada a 0.5 = {metricas['f1_media_columnas']:.6f} "
        f"(pliegues mejorados {metricas['pliegues_mejorados']}/5, "
        f"desviacion {metricas['std_pliegues']:.4f})"
    )
    return ResultadoEnsemble(
        definicion=definicion,
        proba_anidada=proba_anidada,
        metricas=metricas,
        detalle_pliegues=detalle_pliegues,
        segundos=round(time.perf_counter() - inicio, 2),
    )


def evaluar_ensembles(
    proba_base: np.ndarray,
    y_true: np.ndarray,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
    f1_referencia_pliegues: Sequence[float],
) -> list[ResultadoEnsemble]:
    """Evalua de forma anidada las cuatro estrategias de ensemble.

    Args:
        proba_base: tensor ``(n, n_modelos, 3)`` de probabilidades OOF.
        y_true: matriz ``(n, 3)`` con las etiquetas verdaderas.
        pliegues: particion agrupada compartida.
        f1_referencia_pliegues: F1 por pliegue del single de referencia.

    Returns:
        Resultados anidados en el orden de :func:`definiciones_ensemble`.
    """
    log("evaluacion anidada de ensembles (leave-one-fold-out)")
    return [
        evaluar_anidado(definicion, proba_base, y_true, pliegues, f1_referencia_pliegues)
        for definicion in definiciones_ensemble()
    ]


def elegir_ganador(resultados: Sequence[ResultadoEnsemble]) -> ResultadoEnsemble:
    """Elige el ensemble con mayor F1 anidada a umbral 0.5.

    El umbral 0.5 se usa como criterio de seleccion para no introducir sesgo de
    seleccion por umbrales; el ajuste de umbrales se evalua despues solo sobre el
    ganador.

    Args:
        resultados: resultados anidados en orden de ejecucion.

    Returns:
        El mejor resultado (empate: el primero).
    """
    ganador = resultados[0]
    for resultado in resultados[1:]:
        if (
            resultado.metricas["f1_media_columnas"]
            > ganador.metricas["f1_media_columnas"]
        ):
            ganador = resultado
    return ganador


# ===== SECCION 6: estrategias de umbral y decision =====


def rejilla_umbrales() -> np.ndarray:
    """Construye la rejilla de umbrales 0.20-0.80 con paso 0.01.

    Returns:
        Vector de umbrales redondeados a dos decimales.
    """
    inicio, fin, paso = CONFIG["rejilla_umbral"]
    return np.round(np.arange(inicio, fin + paso / 2, paso), 2)


def optimizar_umbrales(
    y_true: np.ndarray, probabilidad: np.ndarray, rejilla: np.ndarray
) -> list[float]:
    """Busca el umbral que maximiza la F1 binaria de cada etiqueta.

    Args:
        y_true: matriz binaria verdadera ``(n, 3)``.
        probabilidad: matriz ``(n, 3)`` de probabilidades de la clase positiva.
        rejilla: umbrales candidatos.

    Returns:
        Lista de umbrales optimos en el orden de :data:`ETIQUETAS` (empate: el
        umbral menor, primer maximo de la rejilla).
    """
    umbrales: list[float] = []
    for indice in range(len(ETIQUETAS)):
        verdad = y_true[:, indice]
        columna = probabilidad[:, indice]
        mejor_umbral = float(rejilla[0])
        mejor_f1 = -1.0
        for umbral in rejilla:
            valor = f1_binaria_rapida(verdad, (columna >= umbral).astype(int))
            if valor > mejor_f1:
                mejor_umbral = float(umbral)
                mejor_f1 = valor
        umbrales.append(mejor_umbral)
    return umbrales


def umbrales_a_diccionario(umbrales: Sequence[float]) -> dict[str, float]:
    """Convierte una lista de umbrales en un diccionario por etiqueta.

    Args:
        umbrales: umbrales en el orden de :data:`ETIQUETAS`.

    Returns:
        Diccionario ``etiqueta -> umbral``.
    """
    return {etiqueta: float(umbral) for etiqueta, umbral in zip(ETIQUETAS, umbrales)}


def evaluar_optimo_anidado(
    proba: np.ndarray,
    y_true: np.ndarray,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
    rejilla: np.ndarray,
) -> dict:
    """Evalua umbrales optimos con protocolo leave-one-fold-out.

    Para cada pliegue de evaluacion el umbral se optimiza con las filas de los
    otros cuatro pliegues; el umbral final del reajuste es el optimo sobre el OOF
    completo (misma regla, estimada con todos los datos).

    Args:
        proba: matriz ``(n, 3)`` de probabilidades.
        y_true: matriz ``(n, 3)`` con las etiquetas verdaderas.
        pliegues: particion agrupada compartida.
        rejilla: umbrales candidatos.

    Returns:
        Diccionario con la prediccion anidada, los umbrales por pliegue y los
        umbrales finales.
    """
    n_filas = proba.shape[0]
    prediccion = np.zeros((n_filas, len(ETIQUETAS)), dtype=int)
    umbrales_por_pliegue: list[dict] = []
    for numero, (_, validacion) in enumerate(pliegues, start=1):
        mascara_ajuste = np.ones(n_filas, dtype=bool)
        mascara_ajuste[validacion] = False
        umbrales = optimizar_umbrales(y_true[mascara_ajuste], proba[mascara_ajuste], rejilla)
        prediccion[validacion] = predecir_con_umbral(proba[validacion], umbrales)
        umbrales_por_pliegue.append(
            {"pliegue": numero, "umbrales": umbrales_a_diccionario(umbrales)}
        )
    return {
        "prediccion": prediccion,
        "umbrales_por_pliegue": umbrales_por_pliegue,
        "umbrales_finales": umbrales_a_diccionario(optimizar_umbrales(y_true, proba, rejilla)),
    }


def evaluar_mediana_por_pliegue(
    proba: np.ndarray,
    y_true: np.ndarray,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
    rejilla: np.ndarray,
) -> dict:
    """Evalua la mediana de los umbrales optimos por pliegue.

    Para cada pliegue de evaluacion se calcula el umbral optimo de cada uno de los
    otros cuatro pliegues por separado y se toma la mediana por etiqueta; los
    umbrales finales del reajuste son la mediana de los cinco optimos por pliegue.

    Args:
        proba: matriz ``(n, 3)`` de probabilidades.
        y_true: matriz ``(n, 3)`` con las etiquetas verdaderas.
        pliegues: particion agrupada compartida.
        rejilla: umbrales candidatos.

    Returns:
        Diccionario con la prediccion anidada, los umbrales por pliegue y los
        umbrales finales (medianas por etiqueta).
    """
    optimos_por_pliegue = [
        optimizar_umbrales(y_true[validacion], proba[validacion], rejilla)
        for _, validacion in pliegues
    ]
    prediccion = np.zeros((proba.shape[0], len(ETIQUETAS)), dtype=int)
    umbrales_por_pliegue: list[dict] = []
    for numero, (_, validacion) in enumerate(pliegues, start=1):
        ajuste = [
            optimos
            for indice, optimos in enumerate(optimos_por_pliegue)
            if indice != numero - 1
        ]
        mediana = np.median(np.asarray(ajuste, dtype=float), axis=0)
        prediccion[validacion] = predecir_con_umbral(proba[validacion], mediana)
        umbrales_por_pliegue.append(
            {"pliegue": numero, "umbrales": umbrales_a_diccionario(mediana)}
        )
    mediana_global = np.median(np.asarray(optimos_por_pliegue, dtype=float), axis=0)
    return {
        "prediccion": prediccion,
        "umbrales_por_pliegue": umbrales_por_pliegue,
        "umbrales_finales": umbrales_a_diccionario(mediana_global),
        "umbrales_optimos_por_pliegue": [
            umbrales_a_diccionario(optimos) for optimos in optimos_por_pliegue
        ],
    }


def comparar_estrategias_umbrales(
    proba: np.ndarray,
    y_true: np.ndarray,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
    rejilla: np.ndarray,
) -> dict:
    """Compara las tres estrategias de umbral de forma homogenea.

    Estrategias: (a) 0.5 fijo; (b) optimo sobre el OOF completo, que es una cota
    optimista porque umbral y metrica comparten los mismos datos, acompanado de su
    version anidada (leave-one-fold-out); (c) mediana de los umbrales optimos por
    pliegue. La estrategia elegida es la de mejor F1 anidada entre (a), (b)
    anidada y (c); los empates se resuelven a favor de 0.5, que es la mas simple.

    Args:
        proba: matriz ``(n, 3)`` de probabilidades.
        y_true: matriz ``(n, 3)`` con las etiquetas verdaderas.
        pliegues: particion agrupada compartida.
        rejilla: umbrales candidatos.

    Returns:
        Diccionario con las tres estrategias, la elegida, los umbrales finales y
        la F1 anidada alcanzada.
    """
    umbrales_fijos = [CONFIG["umbral_base"]] * len(ETIQUETAS)
    fijo = evaluar_prediccion(
        y_true, predecir_con_umbral(proba, umbrales_fijos), pliegues
    )
    fijo["umbrales"] = umbrales_a_diccionario(umbrales_fijos)

    umbrales_oof = optimizar_umbrales(y_true, proba, rejilla)
    optimista = evaluar_prediccion(
        y_true, predecir_con_umbral(proba, umbrales_oof), pliegues
    )
    optimista["umbrales"] = umbrales_a_diccionario(umbrales_oof)

    anidado = evaluar_optimo_anidado(proba, y_true, pliegues, rejilla)
    metricas_anidado = evaluar_prediccion(y_true, anidado["prediccion"], pliegues)

    mediana = evaluar_mediana_por_pliegue(proba, y_true, pliegues, rejilla)
    metricas_mediana = evaluar_prediccion(y_true, mediana["prediccion"], pliegues)

    candidatas = [
        ("fijo_0.5", fijo, fijo["umbrales"]),
        ("optimo_anidado", metricas_anidado, anidado["umbrales_finales"]),
        ("mediana_por_pliegue", metricas_mediana, mediana["umbrales_finales"]),
    ]
    elegida = candidatas[0]
    for candidata in candidatas[1:]:
        if candidata[1]["f1_media_columnas"] > elegida[1]["f1_media_columnas"]:
            elegida = candidata
    estrategias = {
        "fijo_0.5": {
            "anidada": True,
            "descripcion": "umbral 0.5 por etiqueta, sin ajuste",
            **fijo,
        },
        "optimo_oof_completo": {
            "anidada": False,
            "descripcion": (
                "optimo sobre el OOF completo; cota OPTIMISTA (umbral y metrica "
                "comparten los mismos datos)"
            ),
            **optimista,
        },
        "optimo_anidado": {
            "anidada": True,
            "descripcion": "optimo ajustado en los 4 pliegues no evaluados",
            "umbrales_por_pliegue": anidado["umbrales_por_pliegue"],
            **metricas_anidado,
            "umbrales": anidado["umbrales_finales"],
        },
        "mediana_por_pliegue": {
            "anidada": True,
            "descripcion": "mediana de los umbrales optimos por pliegue",
            "umbrales_por_pliegue": mediana["umbrales_por_pliegue"],
            "umbrales_optimos_por_pliegue": mediana["umbrales_optimos_por_pliegue"],
            **metricas_mediana,
            "umbrales": mediana["umbrales_finales"],
        },
    }
    return {
        "estrategias": estrategias,
        "estrategia_elegida": elegida[0],
        "umbrales_finales": elegida[2],
        "f1_mejor_anidada": float(elegida[1]["f1_media_columnas"]),
        "f1_mejor_anidada_por_pliegue": elegida[1]["por_pliegue"],
        "f1_mejor_anidada_std_pliegues": elegida[1]["std_pliegues"],
        "f1_mejor_anidada_por_etiqueta": elegida[1]["f1_por_etiqueta"],
    }


# ===== SECCION 7: refit con todo el train y submission =====


def ajustar_bases_final(
    train: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, list]]:
    """Ajusta vectorizadores y modelos base con TODO el train.

    Args:
        train: DataFrame de entrenamiento con ``text`` y etiquetas.

    Returns:
        Tupla ``(vectorizadores, modelos)``; ``vectorizadores`` es un diccionario
        por representacion y ``modelos`` otro por modelo base con la lista de
        clasificadores binarios en el orden de :data:`ETIQUETAS`.
    """
    vectorizadores: dict[str, Any] = {}
    modelos: dict[str, list] = {}
    matrices: dict[str, Any] = {}
    for clave, partes in REPRESENTACIONES.items():
        vectorizador = construir_vectorizador(partes)
        matrices[clave] = vectorizador.fit_transform(train["text"])
        vectorizadores[clave] = vectorizador
    for espec in BASES:
        modelos[espec.identificador] = []
        for indice, etiqueta in enumerate(ETIQUETAS):
            modelo = construir_modelo(espec)
            modelo.fit(matrices[espec.representacion], train[etiqueta])
            modelos[espec.identificador].append(modelo)
    return vectorizadores, modelos


def predecir_test_bases(
    test: pd.DataFrame,
    vectorizadores: dict[str, Any],
    modelos: dict[str, list],
) -> np.ndarray:
    """Calcula las probabilidades de los cuatro modelos base en el test.

    Args:
        test: DataFrame de test con ``text``.
        vectorizadores: vectorizadores ajustados con todo el train.
        modelos: clasificadores por modelo base.

    Returns:
        Tensor ``(n_test, n_modelos, 3)`` con las probabilidades.
    """
    proba = np.zeros((len(test), len(BASES), len(ETIQUETAS)), dtype=float)
    for posicion, espec in enumerate(BASES):
        matriz = vectorizadores[espec.representacion].transform(test["text"])
        for indice in range(len(ETIQUETAS)):
            proba[:, posicion, indice] = probabilidades(
                modelos[espec.identificador][indice], matriz, espec
            )
    return proba


def validar_submission(
    submission: pd.DataFrame, plantilla: pd.DataFrame, ruta: Path
) -> None:
    """Valida una submission contra la plantilla oficial antes de escribirla.

    Args:
        submission: DataFrame candidato con ``id, HS, TR, AG``.
        plantilla: plantilla oficial ``data/raw/submission.csv``.
        ruta: ruta destino (solo para mensajes de error).

    Raises:
        ValueError: si el formato, los ids, los valores o los nulos no cumplen
            el contrato de la competencia.
    """
    columnas = tuple(submission.columns)
    if columnas != CONFIG["columnas_submission"]:
        raise ValueError(f"{ruta.name}: columnas {columnas} distintas de id,HS,TR,AG")
    if len(submission) != CONFIG["n_filas_test"]:
        raise ValueError(
            f"{ruta.name}: {len(submission)} filas frente a "
            f"{CONFIG['n_filas_test']} esperadas"
        )
    if list(submission["id"]) != list(plantilla["id"]):
        raise ValueError(
            f"{ruta.name}: los ids no coinciden con la plantilla oficial en "
            "contenido o en orden"
        )
    if submission.isna().any().any():
        raise ValueError(f"{ruta.name}: hay valores nulos")
    if not submission["id"].is_unique:
        raise ValueError(f"{ruta.name}: hay ids duplicados")
    etiquetas = submission[list(ETIQUETAS)].to_numpy()
    if etiquetas.min() < 0 or etiquetas.max() > 1:
        raise ValueError(f"{ruta.name}: hay valores fuera de {{0, 1}}")
    if not np.isin(etiquetas, (0, 1)).all():
        raise ValueError(f"{ruta.name}: las etiquetas deben ser enteras 0/1")


def escribir_submission(
    test: pd.DataFrame,
    plantilla: pd.DataFrame,
    prediccion: np.ndarray,
    nombre: str,
) -> Path:
    """Construye, valida y escribe una submission.

    Args:
        test: DataFrame de test (mismo orden que la plantilla oficial).
        plantilla: plantilla oficial ``data/raw/submission.csv``.
        prediccion: matriz ``(n, 3)`` binaria en el orden de :data:`ETIQUETAS`.
        nombre: nombre del archivo dentro de ``submissions/``.

    Returns:
        Ruta del archivo de submission escrito.

    Raises:
        ValueError: si la submission no cumple el formato oficial.
    """
    salida = plantilla.copy()
    for indice, etiqueta in enumerate(ETIQUETAS):
        salida[etiqueta] = prediccion[:, indice]
    destino = CONFIG["submissions_dir"] / nombre
    validar_submission(salida, plantilla, destino)
    CONFIG["submissions_dir"].mkdir(parents=True, exist_ok=True)
    salida.to_csv(destino, index=False, encoding="utf-8")
    releida = pd.read_csv(destino, dtype={"id": str})
    validar_submission(releida, plantilla, destino)
    log(f"submission escrita y validada: {destino.name} ({len(releida)} filas)")
    return destino


def resumir_prediccion(train: pd.DataFrame, prediccion: np.ndarray) -> dict:
    """Resume los positivos predichos en el test frente a la prevalencia de train.

    Args:
        train: DataFrame de entrenamiento con etiquetas.
        prediccion: matriz ``(n, 3)`` binaria del test.

    Returns:
        Diccionario con el conteo y la tasa de positivos por etiqueta en test y
        la tasa de positivos de train como referencia.
    """
    return {
        "positivos_test": {
            etiqueta: int(prediccion[:, indice].sum())
            for indice, etiqueta in enumerate(ETIQUETAS)
        },
        "tasa_positivos_test": {
            etiqueta: round(float(prediccion[:, indice].mean()), 4)
            for indice, etiqueta in enumerate(ETIQUETAS)
        },
        "tasa_positivos_train": {
            etiqueta: round(float(train[etiqueta].mean()), 4) for etiqueta in ETIQUETAS
        },
    }


# ===== SECCION 8: registro en experiments.csv, leaderboard y contexto =====


def registrar_en_experiments(filas: Sequence[dict]) -> int:
    """Anade (o refresca) las filas de la fase en ``experiments.csv``.

    La operacion es idempotente: elimina las filas cuyo ``experiment_id``
    pertenece a esta fase y despues las reescribe con el esquema existente.

    Args:
        filas: filas completas con las columnas de ``COLUMNAS_EXPERIMENTS``.

    Returns:
        Numero de filas escritas.

    Raises:
        ValueError: si el esquema del archivo no coincide con el esperado.
    """
    destino: Path = CONFIG["experiments_csv"]
    existente = pd.read_csv(destino, dtype=str, keep_default_na=False)
    if list(existente.columns) != list(COLUMNAS_EXPERIMENTS):
        raise ValueError("El esquema de experiments.csv cambio inesperadamente")
    identificadores = {fila["experiment_id"] for fila in filas}
    conservadas = existente[~existente["experiment_id"].isin(identificadores)]
    nuevas = pd.DataFrame(
        [{clave: fila[clave] for clave in COLUMNAS_EXPERIMENTS} for fila in filas],
        columns=list(COLUMNAS_EXPERIMENTS),
    )
    combinado = pd.concat([conservadas, nuevas], ignore_index=True)
    combinado.to_csv(destino, index=False, encoding="utf-8")
    return len(nuevas)


def registrar_en_leaderboard(filas: Sequence[dict]) -> int:
    """Anade (o refresca) las filas de submission en ``leaderboard_log.csv``.

    El ``kaggle_f1`` se deja vacio: lo completa Carlos cuando envie la submission.
    La operacion es idempotente por ``submission_id``.

    Args:
        filas: filas con las columnas de ``COLUMNAS_LEADERBOARD``.

    Returns:
        Numero de filas escritas.

    Raises:
        ValueError: si el esquema del archivo no coincide con el esperado.
    """
    destino: Path = CONFIG["leaderboard_csv"]
    existente = pd.read_csv(destino, dtype=str, keep_default_na=False)
    if list(existente.columns) != list(COLUMNAS_LEADERBOARD):
        raise ValueError("El esquema de leaderboard_log.csv cambio inesperadamente")
    identificadores = {fila["submission_id"] for fila in filas}
    conservadas = existente[~existente["submission_id"].isin(identificadores)]
    nuevas = pd.DataFrame(
        [{clave: fila[clave] for clave in COLUMNAS_LEADERBOARD} for fila in filas],
        columns=list(COLUMNAS_LEADERBOARD),
    )
    combinado = pd.concat([conservadas, nuevas], ignore_index=True)
    combinado.to_csv(destino, index=False, encoding="utf-8")
    return len(nuevas)


def construir_fila_ensemble(
    resultado: ResultadoEnsemble,
    comparacion: dict,
    f1_mod03b: float,
    f1_single_anidado: float,
    decisiones: dict[str, str],
) -> dict:
    """Construye la fila de ``experiments.csv`` de un ensemble.

    Args:
        resultado: resultado anidado del ensemble.
        comparacion: comparacion de estrategias de umbral del ensemble.
        f1_mod03b: F1 del single de referencia publicado.
        f1_single_anidado: mejor F1 anidada del single bajo el mismo protocolo.
        decisiones: decision por ``experiment_id``.

    Returns:
        Diccionario con las columnas de ``COLUMNAS_EXPERIMENTS``.
    """
    metricas = resultado.metricas
    return {
        "experiment_id": resultado.definicion.identificador,
        "competition": CONFIG["competencia"],
        "question": (
            "Un ensemble de las 4 representaciones supera al mejor single "
            "(MOD-03b_BEST) en la evaluacion anidada?"
        ),
        "hypothesis": (
            "La diversidad entre word, char_wb, union y LinearSVC aporta senal "
            "complementaria que el single union+LR no captura"
        ),
        "preprocessing": "texto crudo, lowercase, sin stopwords",
        "feature_groups": "WORD_NGRAM+CHAR_NGRAM (union de 4 modelos base)",
        "representation": (
            "ensamble de TF-IDF word (1,2), union word (1,2)+char_wb (3,5) y "
            "char_wb (3,5), min_df=2, sublinear_tf"
        ),
        "model": resultado.definicion.nombre,
        "hyperparameters": json.dumps(
            {
                "estrategia": resultado.definicion.descripcion,
                "estrategia_umbrales": comparacion["estrategia_elegida"],
                "umbrales_finales": comparacion["umbrales_finales"],
                "paso_pesos": CONFIG["paso_pesos"],
                "rejilla_umbral": list(CONFIG["rejilla_umbral"]),
            },
            ensure_ascii=False,
        ),
        "validation": (
            "Evaluacion anidada leave-one-fold-out: pesos/meta-modelo ajustados en 4 "
            "pliegues y evaluados en el restante, sobre las probabilidades OOF de "
            "StratifiedGroupKFold(5, shuffle=True, random_state=42) agrupado por "
            "near-duplicado y estratificado por HSxTRxAG; limitacion: las "
            "probabilidades base son OOF compartidas y el single se evalua con el "
            "mismo protocolo para que la comparacion sea homogenea"
        ),
        "f1": round(metricas["f1_media_columnas"], 4),
        "macro_f1": round(metricas["macro6"], 4),
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": resultado.segundos,
        "dimension": DIMENSION_ENSEMBLE,
        "result": json.dumps(
            {
                "f1_anidada_0.5": round(metricas["f1_media_columnas"], 6),
                "f1_por_etiqueta_0.5": {
                    etiqueta: round(valor, 6)
                    for etiqueta, valor in metricas["f1_por_etiqueta"].items()
                },
                "f1_por_pliegue_0.5": [round(valor, 6) for valor in metricas["por_pliegue"]],
                "std_pliegues_0.5": round(metricas["std_pliegues"], 6),
                "pliegues_mejorados_vs_mod03b": metricas["pliegues_mejorados"],
                "delta_vs_mod03b_0.5": round(metricas["f1_media_columnas"] - f1_mod03b, 6),
                "delta_vs_single_anidado": round(metricas["f1_media_columnas"] - f1_single_anidado, 6),
                "estrategia_umbrales_elegida": comparacion["estrategia_elegida"],
                "umbrales_finales": comparacion["umbrales_finales"],
                "f1_anidada_mejores_umbrales": round(
                    comparacion["f1_mejor_anidada"], 6
                ),
                "f1_anidada_mejores_umbrales_por_pliegue": [
                    round(valor, 6) for valor in comparacion["f1_mejor_anidada_por_pliegue"]
                ],
                "delta_vs_mod03b_mejores_umbrales": round(
                    comparacion["f1_mejor_anidada"] - f1_mod03b, 6
                ),
                "f1_optimo_oof_completo": round(
                    comparacion["estrategias"]["optimo_oof_completo"][
                        "f1_media_columnas"
                    ],
                    6,
                ),
                "detalle_pliegues": resultado.detalle_pliegues,
            },
            ensure_ascii=False,
        ),
        "interpretation": (
            f"F1 anidada a 0.5 = {metricas['f1_media_columnas']:.4f} "
            f"(delta vs MOD-03b {metricas['f1_media_columnas'] - f1_mod03b:+.4f}, "
            f"{metricas['pliegues_mejorados']}/5 pliegues mejorados); con la estrategia "
            f"de umbrales '{comparacion['estrategia_elegida']}' = "
            f"{comparacion['f1_mejor_anidada']:.4f}; MOD-03b con el mismo protocolo = "
            f"{f1_single_anidado:.4f}"
        ),
        "decision": decisiones[resultado.definicion.identificador],
        "date": CONFIG["fecha"],
        "script": CONFIG["script"],
        "notes": (
            "Umbrales por etiqueta estimados sin el pliegue de evaluacion; el test "
            "solo se usa si el ganador supera la regla de +0.005 anidado. "
            f"segundos={resultado.segundos}"
        ),
    }


def construir_fila_submission(
    nombre_ganador: str,
    descripcion: str,
    umbrales: dict[str, float],
    f1_nested: float,
    macro6_nested: float,
    ruta: Path,
    resumen: dict,
    segundos: float,
) -> dict:
    """Construye la fila ``S04`` del registro de experimentos.

    Args:
        nombre_ganador: identificador del ensemble ganador.
        descripcion: descripcion del ensemble y sus umbrales.
        umbrales: umbrales finales por etiqueta.
        f1_nested: F1 anidada del ganador con esos umbrales.
        macro6_nested: macro-6 anidada del ganador con esos umbrales.
        ruta: ruta de la submission generada.
        resumen: resumen de positivos en test y train.
        segundos: segundos del refit y la prediccion.

    Returns:
        Diccionario con las columnas de ``COLUMNAS_EXPERIMENTS``.
    """
    return {
        "experiment_id": CONFIG["id_submission"],
        "competition": CONFIG["competencia"],
        "question": (
            "Que submission genera el mejor ensemble reajustado con todo el train y "
            "los umbrales robustos elegidos en la evaluacion anidada?"
        ),
        "hypothesis": (
            "El ganador de la Fase 13b supera a MOD-03b_BEST por >= +0.005 en la "
            "evaluacion anidada, por lo que el reajuste con todo el train mejora la "
            "submission v03"
        ),
        "preprocessing": "texto crudo, lowercase, sin stopwords",
        "feature_groups": "WORD_NGRAM+CHAR_NGRAM (union de 4 modelos base)",
        "representation": (
            "ensamble de TF-IDF word (1,2), union word (1,2)+char_wb (3,5) y "
            "char_wb (3,5), min_df=2, sublinear_tf"
        ),
        "model": descripcion,
        "hyperparameters": json.dumps(
            {
                "ganador": nombre_ganador,
                "umbrales_por_etiqueta": umbrales,
                "umbral_base": CONFIG["umbral_base"],
                "postproceso": "ninguno",
            },
            ensure_ascii=False,
        ),
        "validation": (
            "F1 anidada leave-one-fold-out sobre el split agrupado de las Fases 4-5 y "
            "13a; submission validada contra la plantilla oficial (columnas, 1154 "
            "filas, orden de ids, valores en {0,1}, sin nulos ni duplicados)"
        ),
        "f1": round(f1_nested, 4),
        "macro_f1": round(macro6_nested, 4),
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": round(segundos, 2),
        "dimension": DIMENSION_ENSEMBLE,
        "result": json.dumps(
            {
                "submission": str(ruta.relative_to(PROJECT_ROOT)),
                "f1_anidada_umbrales": round(f1_nested, 6),
                "positivos_test": resumen["positivos_test"],
                "tasa_positivos_test": resumen["tasa_positivos_test"],
                "tasa_positivos_train": resumen["tasa_positivos_train"],
            },
            ensure_ascii=False,
        ),
        "interpretation": (
            "Reajuste del ganador con todo el train y prediccion del test; pendiente de "
            "envio por Carlos a Kaggle"
        ),
        "decision": "submission generada; pendiente de envio por Carlos a Kaggle",
        "date": CONFIG["fecha"],
        "script": CONFIG["script"],
        "notes": (
            "Los pesos/meta-modelo del ensemble se ajustan con el OOF completo y se "
            "aplican a las probabilidades del refit; la F1 anidada es la estimacion "
            "honesta de generalizacion"
        ),
    }


def resumir_ensembles_una_linea(
    resultados: Sequence[ResultadoEnsemble],
    ganador: ResultadoEnsemble,
    comparacion_ganador: dict,
) -> str:
    """Resume los cuatro ensembles en una sola linea de contexto.

    Args:
        resultados: resultados anidados de los cuatro ensembles.
        ganador: ensemble ganador.
        comparacion_ganador: comparacion de umbrales del ganador.

    Returns:
        Texto con el identificador, la F1 anidada a 0.5 y, para el ganador, la F1
        con la estrategia de umbrales elegida.
    """
    fragmentos: list[str] = []
    for resultado in resultados:
        identificador = resultado.definicion.identificador
        fragmento = f"{identificador} {resultado.metricas['f1_media_columnas']:.4f}"
        if identificador == ganador.definicion.identificador:
            fragmento += f" (umbrales {comparacion_ganador['f1_mejor_anidada']:.4f})"
        fragmentos.append(fragmento)
    return "; ".join(fragmentos)


def construir_seccion_contexto(
    resultados: Sequence[ResultadoEnsemble],
    ganador: ResultadoEnsemble,
    comparacion_ganador: dict,
    comparacion_single: dict,
    diversidad: pd.DataFrame,
    decisiones: dict[str, str],
    rutas: dict[str, Path],
    f1_mod03b: float,
) -> str:
    """Redacta la seccion de handoff (maximo 20 lineas) de ``reports/contexto.md``.

    Args:
        resultados: resultados anidados de los cuatro ensembles.
        ganador: ensemble ganador por F1 anidada a 0.5.
        comparacion_ganador: estrategias de umbral del ganador.
        comparacion_single: estrategias de umbral del single de referencia.
        diversidad: tabla de diversidad por pares.
        decisiones: decision por ``experiment_id``.
        rutas: rutas de los artefactos de submission (vacio si no aplica).
        f1_mod03b: F1 del single publicado.

    Returns:
        Texto de la seccion, con encabezado y sin linea final en blanco.

    Raises:
        RuntimeError: si la seccion generada excede las 20 lineas permitidas.
    """
    linea_ensembles = resumir_ensembles_una_linea(resultados, ganador, comparacion_ganador)
    estrategias = comparacion_ganador["estrategias"]
    umbrales_finales = ", ".join(
        f"{etiqueta} {valor:.2f}"
        for etiqueta, valor in comparacion_ganador["umbrales_finales"].items()
    )
    delta_publicado = comparacion_ganador["f1_mejor_anidada"] - f1_mod03b
    delta_single = (
        comparacion_ganador["f1_mejor_anidada"] - comparacion_single["f1_mejor_anidada"]
    )
    if rutas:
        linea_submission = (
            "Submission generada y validada contra la plantilla oficial: "
            + "; ".join(f"`{ruta.relative_to(PROJECT_ROOT)}`" for ruta in rutas.values())
            + "."
        )
    else:
        linea_submission = (
            "NO se genera submission v04: el ganador no supera a MOD-03b_BEST por "
            f"+{CONFIG['umbral_mejora_submission']:.3f} en la evaluacion anidada "
            f"(delta {delta_publicado:+.4f}); se mantienen v02/v03 como control y apuesta."
        )
    correlacion_min = float(diversidad["correlacion_errores"].min())
    correlacion_max = float(diversidad["correlacion_errores"].max())
    desacuerdo_min = float(diversidad["tasa_desacuerdo"].min())
    desacuerdo_max = float(diversidad["tasa_desacuerdo"].max())
    lineas = [
        f"## Fase 13b - ml-trainer ({CONFIG['momento_contexto']})",
        "",
        (
            "Ensembles con evaluacion anidada sobre 4 modelos base OOF con el split agrupado "
            "de las Fases 4-5/13a (StratifiedGroupKFold(5, shuffle=True, random_state=42), "
            "near-duplicados, estratificado HSxTRxAG); sin fuga (vectorizador por pliegue)."
        ),
        (
            "Bases en `features/oof_bases.npz` (F1 OOF a 0.5 replicada con tolerancia 1e-6 "
            "frente a la Fase 4-5, dimensiones 4024/23407/27431 verificadas): M1 word+LR "
            "0.7587 (B1), M2 union+LR 0.7927 (MOD-03b), M3 union+LinearSVC 0.7849 (sigmoid "
            "= decision>0) y M4 char_wb+LR 0.7567 (REP-01)."
        ),
        (
            f"Diversidad por pares (`reports/ensembles_diversidad.csv`): correlacion de "
            f"errores {correlacion_min:.2f}..{correlacion_max:.2f}, desacuerdo a 0.5 "
            f"{desacuerdo_min:.3f}..{desacuerdo_max:.3f} y Q-statistic en la tabla."
        ),
        (
            "Ensembles anidados (pesos/meta-modelo ajustados en 4 pliegues y evaluados en el "
            f"restante): {linea_ensembles}."
        ),
        (
            f"Ganador: {ganador.definicion.identificador} ({ganador.definicion.nombre}); "
            f"F1 anidada 0.5 = {ganador.metricas['f1_media_columnas']:.4f} "
            f"({ganador.metricas['pliegues_mejorados']}/5 pliegues mejorados vs MOD-03b); "
            f"mejor variante con umbrales = {comparacion_ganador['f1_mejor_anidada']:.4f} "
            f"(delta vs MOD-03b {delta_publicado:+.4f}, delta vs el single con el mismo "
            f"protocolo {delta_single:+.4f})."
        ),
        (
            "Umbrales del ganador: 0.5 fijo = "
            f"{estrategias['fijo_0.5']['f1_media_columnas']:.4f}; optimo OOF completo = "
            f"{estrategias['optimo_oof_completo']['f1_media_columnas']:.4f} (optimista); "
            f"optimo anidado = {estrategias['optimo_anidado']['f1_media_columnas']:.4f}; "
            f"mediana por pliegue = {estrategias['mediana_por_pliegue']['f1_media_columnas']:.4f} "
            f"-> estrategia elegida {comparacion_ganador['estrategia_elegida']} "
            f"(umbrales finales {umbrales_finales})."
        ),
        linea_submission,
        (
            "Limitacion metodologica: las probabilidades base son OOF compartidas (la "
            "prediccion de un pliegue viene de modelos que vieron los otros, incluido el de "
            "evaluacion), por lo que la estimacion anidada sigue siendo algo optimista; el "
            "single se evaluo con el mismo protocolo y el test nunca se uso para decidir."
        ),
        (
            "Artefactos: `features/oof_bases.npz`, `reports/ensembles_diversidad.csv`, "
            "`reports/ensembles_resultados.csv` y `.json`; registro en `experiments.csv` "
            f"({', '.join(decisiones)})."
        ),
    ]
    while lineas and lineas[-1] == "":
        lineas.pop()
    if len(lineas) > 20:
        raise RuntimeError(f"La seccion de contexto excede 20 lineas ({len(lineas)})")
    return "\n".join(lineas)


def actualizar_contexto(seccion: str) -> Path:
    """Inserta la seccion de handoff al final de ``reports/contexto.md``.

    Idempotente: si la seccion de esta fase ya existe, se reemplaza en lugar de
    duplicarse.

    Args:
        seccion: texto de la seccion (sin linea final en blanco).

    Returns:
        Ruta del archivo actualizado.
    """
    destino: Path = CONFIG["contexto_md"]
    encabezado = f"## Fase 13b - ml-trainer ({CONFIG['momento_contexto']})"
    texto = destino.read_text(encoding="utf-8") if destino.exists() else ""
    if encabezado in texto:
        texto = texto[: texto.index(encabezado)]
    destino.write_text(
        texto.rstrip("\n") + "\n\n" + seccion.rstrip("\n") + "\n", encoding="utf-8"
    )
    return destino


# ===== SECCION 9: flujo principal =====


def construir_tabla_resultados(
    resultados: Sequence[ResultadoEnsemble],
    comparaciones: dict[str, dict],
    decisiones: dict[str, str],
    f1_mod03b: float,
    f1_single_anidado: float,
) -> pd.DataFrame:
    """Construye la tabla resumen de los ensambles anidados.

    Args:
        resultados: resultados anidados de los cuatro ensembles.
        comparaciones: comparacion de umbrales por ``experiment_id``.
        decisiones: decision por ``experiment_id``.
        f1_mod03b: F1 del single publicado.
        f1_single_anidado: mejor F1 anidada del single bajo el mismo protocolo.

    Returns:
        DataFrame con las columnas de ``COLUMNAS_RESULTADOS``.
    """
    filas: list[dict] = []
    for resultado in resultados:
        identificador = resultado.definicion.identificador
        comparacion = comparaciones[identificador]
        metricas = resultado.metricas
        filas.append(
            {
                "experiment_id": identificador,
                "ensemble": resultado.definicion.nombre,
                "descripcion": resultado.definicion.descripcion,
                "f1_anidada_0.5": round(metricas["f1_media_columnas"], 6),
                "f1_HS": round(metricas["f1_por_etiqueta"]["HS"], 6),
                "f1_TR": round(metricas["f1_por_etiqueta"]["TR"], 6),
                "f1_AG": round(metricas["f1_por_etiqueta"]["AG"], 6),
                "std_pliegues_0.5": round(metricas["std_pliegues"], 6),
                "delta_vs_mod03b": round(metricas["f1_media_columnas"] - f1_mod03b, 6),
                "pliegues_mejorados": metricas["pliegues_mejorados"],
                "delta_vs_single_anidado": round(
                    metricas["f1_media_columnas"] - f1_single_anidado, 6
                ),
                "estrategia_umbrales": comparacion["estrategia_elegida"],
                "f1_anidada_umbrales": round(comparacion["f1_mejor_anidada"], 6),
                "umbrales_finales": ", ".join(
                    f"{etiqueta}={valor:.2f}"
                    for etiqueta, valor in comparacion["umbrales_finales"].items()
                ),
                "decision": decisiones[identificador],
            }
        )
    return pd.DataFrame(filas, columns=list(COLUMNAS_RESULTADOS))


def main() -> None:
    """Ejecuta la Fase 13b completa: bases OOF, ensembles anidados y decision."""
    inicio = time.perf_counter()
    train, test, grupos = cargar_datos()
    plantilla = data_module.read_submission()
    if list(plantilla["id"]) != list(test["id"]):
        raise ValueError(
            "El orden de ids del test no coincide con la plantilla oficial; "
            "abortado antes de predecir"
        )
    log(
        f"train {train.shape} | test {test.shape} | plantilla {plantilla.shape} | "
        f"grupos {grupos.nunique()}"
    )
    referencias = leer_referencias()
    f1_mod03b = float(referencias[CONFIG["id_mod03b"]]["f1_media_columnas"])
    if abs(f1_mod03b - CONFIG["f1_mod03b_referencia"]) > CONFIG["tolerancia_replicacion"]:
        raise ValueError(
            "La referencia de MOD-03b cambio en variantes_representaciones.json: "
            f"{f1_mod03b!r} frente a {CONFIG['f1_mod03b_referencia']!r}"
        )

    pliegues = construir_pliegues(train, grupos)
    y_true = train[list(ETIQUETAS)].to_numpy()

    # 1) Probabilidades OOF de las cuatro bases y verificacion de replicacion.
    inicio_oof = time.perf_counter()
    probabilidad_bases = calcular_oof_bases(train, pliegues)
    segundos_oof = time.perf_counter() - inicio_oof
    probabilidad_bases["M2"], sincronizacion_m2 = sincronizar_oof_mod03b(
        train, probabilidad_bases["M2"]
    )
    log(
        f"M2 frente a features/oof_mod03b.npz: reutilizado={sincronizacion_m2['reutilizado']} "
        f"({sincronizacion_m2['motivo']})"
    )
    ruta_npz = guardar_oof_bases(train, probabilidad_bases)
    verificacion = verificar_bases(y_true, probabilidad_bases, referencias)
    dimensiones = verificar_dimensiones(train, referencias)
    verificacion_f1 = verificar_f1_rapida(
        y_true, probabilidad_bases["M2"], [CONFIG["umbral_base"]] * len(ETIQUETAS)
    )
    log(
        "equivalencia de la F1 rapida con metricas_multietiqueta: "
        f"diferencia {verificacion_f1['diferencia']:+.3e}"
    )
    log(f"probabilidades OOF guardadas en {ruta_npz.name} ({segundos_oof:.2f} s)")

    # 2) Diversidad por pares.
    tabla_diversidad, resumen_bases = analizar_diversidad(probabilidad_bases, y_true)
    CONFIG["diversidad_csv"].parent.mkdir(parents=True, exist_ok=True)
    tabla_diversidad.to_csv(CONFIG["diversidad_csv"], index=False, encoding="utf-8")
    log(f"diversidad por pares: {CONFIG['diversidad_csv'].name}")

    # 3) Evaluacion anidada de los cuatro ensembles.
    proba_apilada = np.stack(
        [probabilidad_bases[espec.identificador] for espec in BASES], axis=1
    )
    referencia_pliegues = evaluar_prediccion(
        y_true,
        predecir_con_umbral(probabilidad_bases["M2"], [CONFIG["umbral_base"]] * len(ETIQUETAS)),
        pliegues,
    )
    resultados = evaluar_ensembles(
        proba_apilada, y_true, pliegues, referencia_pliegues["por_pliegue"]
    )
    ganador = elegir_ganador(resultados)
    log(
        f"ganador por F1 anidada a 0.5: {ganador.definicion.identificador} "
        f"({ganador.metricas['f1_media_columnas']:.6f})"
    )

    # 4) Estrategias de umbral del ganador y del single bajo el mismo protocolo.
    rejilla = rejilla_umbrales()
    comparacion_ganador = comparar_estrategias_umbrales(
        ganador.proba_anidada, y_true, pliegues, rejilla
    )
    comparacion_single = comparar_estrategias_umbrales(
        probabilidad_bases["M2"], y_true, pliegues, rejilla
    )
    comparaciones = {
        resultado.definicion.identificador: comparar_estrategias_umbrales(
            resultado.proba_anidada, y_true, pliegues, rejilla
        )
        for resultado in resultados
    }
    comparaciones[ganador.definicion.identificador] = comparacion_ganador
    f1_single_anidado = comparacion_single["f1_mejor_anidada"]
    log(
        "single MOD-03b con el mismo protocolo: F1 anidada 0.5 = "
        f"{comparacion_single['estrategias']['fijo_0.5']['f1_media_columnas']:.6f}, mejor "
        f"con umbrales = {f1_single_anidado:.6f} "
        f"({comparacion_single['estrategia_elegida']})"
    )
    log(
        f"ganador con umbrales: {comparacion_ganador['f1_mejor_anidada']:.6f} "
        f"({comparacion_ganador['estrategia_elegida']}); delta vs MOD-03b "
        f"{comparacion_ganador['f1_mejor_anidada'] - f1_mod03b:+.4f}; delta vs single "
        f"anidado {comparacion_ganador['f1_mejor_anidada'] - f1_single_anidado:+.4f}"
    )

    # 5) Decision y submission condicional.
    delta_publicado = comparacion_ganador["f1_mejor_anidada"] - f1_mod03b
    genera_submission = bool(delta_publicado >= CONFIG["umbral_mejora_submission"])
    decisiones: dict[str, str] = {}
    for resultado in resultados:
        identificador = resultado.definicion.identificador
        if identificador != ganador.definicion.identificador:
            decisiones[identificador] = (
                f"descartada: F1 anidada {resultado.metricas['f1_media_columnas']:.4f} por "
                f"debajo del mejor ensemble ({ganador.metricas['f1_media_columnas']:.4f})"
            )
        elif genera_submission:
            decisiones[identificador] = (
                f"adoptada: supera a MOD-03b por {delta_publicado:+.4f} (>= "
                f"+{CONFIG['umbral_mejora_submission']:.3f}) en la evaluacion anidada; genera "
                "la submission v04"
            )
        else:
            decisiones[identificador] = (
                f"descartada: mejor ensemble anidado ({comparacion_ganador['f1_mejor_anidada']:.4f}) "
                f"pero no supera a MOD-03b por +{CONFIG['umbral_mejora_submission']:.3f} "
                f"(delta {delta_publicado:+.4f}); no se genera submission"
            )

    rutas: dict[str, Path] = {}
    segundos_refit = 0.0
    resumen_submission: dict | None = None
    macro6_ganador = None
    if genera_submission:
        inicio_refit = time.perf_counter()
        vectorizadores, modelos = ajustar_bases_final(train)
        proba_test = predecir_test_bases(test, vectorizadores, modelos)
        parametros_finales = ganador.definicion.ajustar(proba_apilada, y_true)
        proba_test_ensemble = ganador.definicion.aplicar(parametros_finales, proba_test)
        umbrales_finales = comparacion_ganador["umbrales_finales"]
        prediccion = predecir_con_umbral(
            proba_test_ensemble,
            [umbrales_finales[etiqueta] for etiqueta in ETIQUETAS],
        )
        segundos_refit = time.perf_counter() - inicio_refit
        nombre = f"toxic_submission_v04_{ganador.definicion.sufijo_submission}.csv"
        rutas["v04"] = escribir_submission(test, plantilla, prediccion, nombre)
        resumen_submission = resumir_prediccion(train, prediccion)
        macro6_ganador = float(
            metricas_multietiqueta(
                y_true, predecir_con_umbral(ganador.proba_anidada, [umbrales_finales[etiqueta] for etiqueta in ETIQUETAS])
            )["f1_macro6"]
        )
        ruta_modelo = CONFIG["models_dir"] / (
            f"fase13b_{ganador.definicion.identificador.lower()}.joblib"
        )
        CONFIG["models_dir"].mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizadores": vectorizadores,
                "modelos": modelos,
                "definicion": ganador.definicion.identificador,
                "descripcion": ganador.definicion.nombre,
                "parametros_ensemble": ganador.definicion.resumir(parametros_finales),
                "estrategia_umbrales": comparacion_ganador["estrategia_elegida"],
                "umbrales": umbrales_finales,
                "etiquetas": list(ETIQUETAS),
                "f1_anidada_umbrales": comparacion_ganador["f1_mejor_anidada"],
                "entrenado_con": "todo el train (1731 filas)",
                "fecha": CONFIG["fecha"],
                "script": CONFIG["script"],
            },
            ruta_modelo,
        )
        log(f"modelo final guardado: {ruta_modelo.name}")
        log(f"test v04: {json.dumps(resumen_submission, ensure_ascii=False)}")
    else:
        log(
            "NO se genera submission v04: el ganador no supera a MOD-03b por "
            f"+{CONFIG['umbral_mejora_submission']:.3f} en la evaluacion anidada "
            f"(delta {delta_publicado:+.4f})"
        )

    # 6) Registro en experiments.csv (ENS-01..ENS-04 y, si aplica, S04).
    filas_experiments = [
        construir_fila_ensemble(
            resultado,
            comparaciones[resultado.definicion.identificador],
            f1_mod03b,
            f1_single_anidado,
            decisiones,
        )
        for resultado in resultados
    ]
    if genera_submission and resumen_submission is not None and macro6_ganador is not None:
        filas_experiments.append(
            construir_fila_submission(
                ganador.definicion.identificador,
                (
                    f"{ganador.definicion.nombre} con umbrales "
                    f"{comparacion_ganador['estrategia_elegida']}"
                ),
                comparacion_ganador["umbrales_finales"],
                comparacion_ganador["f1_mejor_anidada"],
                macro6_ganador,
                rutas["v04"],
                resumen_submission,
                segundos_refit,
            )
        )
    escritas = registrar_en_experiments(filas_experiments)
    log(f"filas escritas en experiments.csv: {escritas}")

    # 7) Registro en el leaderboard log (sin kaggle_f1 todavia).
    if genera_submission and resumen_submission is not None:
        identificador_leaderboard = rutas["v04"].name.replace(
            "toxic_submission_", ""
        ).replace(".csv", "")
        escritas_lb = registrar_en_leaderboard(
            [
                {
                    "submission_id": identificador_leaderboard,
                    "fecha": CONFIG["fecha"],
                    "archivo": str(rutas["v04"].relative_to(PROJECT_ROOT)),
                    "local_f1": round(comparacion_ganador["f1_mejor_anidada"], 5),
                    "kaggle_f1": "",
                    "diferencia": "",
                    "features": (
                        "ensamble de TF-IDF word (1,2), union word (1,2)+char_wb (3,5) y "
                        "char_wb (3,5), min_df=2, sublinear_tf"
                    ),
                    "modelo": (
                        f"{ganador.definicion.nombre} con umbrales por etiqueta "
                        f"(estrategia {comparacion_ganador['estrategia_elegida']})"
                    ),
                    "hiperparametros": json.dumps(
                        {
                            "ganador": ganador.definicion.identificador,
                            "umbrales": comparacion_ganador["umbrales_finales"],
                            "f1_anidada": round(comparacion_ganador["f1_mejor_anidada"], 6),
                        },
                        ensure_ascii=False,
                    ),
                    "interpretacion": (
                        "Mejor ensemble de la Fase 13b con evaluacion anidada; F1 anidada "
                        f"{comparacion_ganador['f1_mejor_anidada']:.4f} frente a 0.7927 del "
                        "single MOD-03b. Recordatorio de calibracion: el gap local-LB de la "
                        "familia union word+char observado en v02/v03 fue de -0.013/-0.016."
                    ),
                }
            ]
        )
        log(f"filas escritas en leaderboard_log.csv: {escritas_lb}")
    else:
        log("leaderboard_log.csv sin cambios (no hay submission nueva)")

    # 8) Tablas de resultados, contexto y JSON de detalle.
    tabla_resultados = construir_tabla_resultados(
        resultados, comparaciones, decisiones, f1_mod03b, f1_single_anidado
    )
    tabla_resultados.to_csv(CONFIG["resultados_csv"], index=False, encoding="utf-8")
    log(f"resultados escritos en {CONFIG['resultados_csv'].name}")

    seccion = construir_seccion_contexto(
        resultados,
        ganador,
        comparacion_ganador,
        comparacion_single,
        tabla_diversidad,
        decisiones,
        rutas,
        f1_mod03b,
    )
    ruta_contexto = actualizar_contexto(seccion)
    log(f"contexto actualizado: {ruta_contexto.relative_to(PROJECT_ROOT)}")

    detalle = {
        "metadata": {
            "fecha": CONFIG["fecha"],
            "script": CONFIG["script"],
            "competencia": CONFIG["competencia"],
            "fase": "Fase 13b (ensembles con evaluacion anidada y umbrales robustos)",
            "metrica_primaria": "f1_media_columnas",
            "validacion": (
                f"StratifiedGroupKFold({CONFIG['n_splits']}, shuffle=True, "
                f"random_state={CONFIG['random_state']}) agrupado por near-duplicado y "
                "estratificado por HSxTRxAG; evaluacion anidada leave-one-fold-out"
            ),
            "limitacion_anidado": (
                "Las probabilidades base son OOF compartidas: la prediccion de un pliegue "
                "proviene de modelos entrenados con los otros pliegues, incluido el de "
                "evaluacion. La estimacion anidada del ensemble es por ello algo optimista "
                "frente a un refit real; el single se evalua con el mismo protocolo."
            ),
            "sin_fuga_test": "el test solo se usa para predecir si el ganador supera la regla",
            "tolerancia_replicacion": CONFIG["tolerancia_replicacion"],
            "segundos_oof": round(segundos_oof, 2),
            "segundos_refit": round(segundos_refit, 2),
            "regla_submission": (
                f"generar submission solo si el ganador supera a MOD-03b por >= +"
                f"{CONFIG['umbral_mejora_submission']:.3f} en la evaluacion anidada"
            ),
        },
        "verificacion_bases": verificacion,
        "verificacion_dimensiones": dimensiones,
        "verificacion_f1_rapida": verificacion_f1,
        "sincronizacion_oof_mod03b": sincronizacion_m2,
        "resumen_bases": resumen_bases,
        "diversidad": tabla_diversidad.to_dict(orient="records"),
        "ensembles": [
            {
                "experiment_id": resultado.definicion.identificador,
                "nombre": resultado.definicion.nombre,
                "descripcion": resultado.definicion.descripcion,
                "f1_anidada_0.5": round(resultado.metricas["f1_media_columnas"], 6),
                "f1_por_etiqueta_0.5": {
                    etiqueta: round(valor, 6)
                    for etiqueta, valor in resultado.metricas["f1_por_etiqueta"].items()
                },
                "f1_por_pliegue_0.5": [
                    round(valor, 6) for valor in resultado.metricas["por_pliegue"]
                ],
                "std_pliegues_0.5": round(resultado.metricas["std_pliegues"], 6),
                "delta_vs_mod03b_0.5": round(resultado.metricas["f1_media_columnas"] - f1_mod03b, 6),
                "delta_vs_single_anidado": round(
                    resultado.metricas["f1_media_columnas"] - f1_single_anidado, 6
                ),
                "pliegues_mejorados": resultado.metricas["pliegues_mejorados"],
                "segundos": resultado.segundos,
                "detalle_pliegues": resultado.detalle_pliegues,
                "umbrales": comparaciones[resultado.definicion.identificador],
            }
            for resultado in resultados
        ],
        "ganador": ganador.definicion.identificador,
        "umbrales_single_mod03b": comparacion_single,
        "decision": {
            "delta_ganador_vs_mod03b": round(delta_publicado, 6),
            "delta_ganador_vs_single_anidado": round(
                comparacion_ganador["f1_mejor_anidada"] - f1_single_anidado, 6
            ),
            "genera_submission": genera_submission,
            "umbral_mejora_submission": CONFIG["umbral_mejora_submission"],
            "motivo": (
                "el ganador supera la regla pre-registrada"
                if genera_submission
                else (
                    "el ganador no supera a MOD-03b por >= +0.005 en la evaluacion "
                    "anidada; se mantienen v02/v03"
                )
            ),
            "decisiones_por_ensemble": decisiones,
        },
        "submission": (
            {
                "ruta": str(rutas["v04"].relative_to(PROJECT_ROOT)),
                "umbrales": comparacion_ganador["umbrales_finales"],
                "estrategia_umbrales": comparacion_ganador["estrategia_elegida"],
                "f1_anidada": round(comparacion_ganador["f1_mejor_anidada"], 6),
                "prediccion_test": resumen_submission,
            }
            if rutas
            else None
        ),
        "artefactos": {
            "oof_bases_npz": str(ruta_npz.relative_to(PROJECT_ROOT)),
            "diversidad_csv": str(CONFIG["diversidad_csv"].relative_to(PROJECT_ROOT)),
            "resultados_csv": str(CONFIG["resultados_csv"].relative_to(PROJECT_ROOT)),
        },
    }
    CONFIG["resultados_json"].parent.mkdir(parents=True, exist_ok=True)
    CONFIG["resultados_json"].write_text(
        json.dumps(detalle, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    log(f"detalle: {CONFIG['resultados_json'].relative_to(PROJECT_ROOT)}")
    log(f"duracion total: {time.perf_counter() - inicio:.2f} s")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - el punto de entrada CLI reporta todo
        import traceback

        traceback.print_exc()
        sys.exit(1)
