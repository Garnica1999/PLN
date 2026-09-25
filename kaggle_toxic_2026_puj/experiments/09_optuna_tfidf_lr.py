"""09_optuna_tfidf_lr.py - Fase 10/27: optimizacion acotada de hiperparametros.

Pregunta que responde
---------------------
El pipeline ganador de las Fases 4-5/13a (``MOD-03b_BEST``: union TF-IDF word
(1,2) + char_wb (3,5), min_df=2, max_df=1.0, sublinear_tf y
``LogisticRegression(class_weight="balanced", C=2.0, liblinear)`` por etiqueta)
obtuvo ``f1_media_columnas`` out-of-fold 0.7927. ¿Existe alguna combinacion de
hiperparametros, dentro de un espacio acotado alrededor de esa configuracion,
que mejore esa metrica con el mismo protocolo de validacion?

Metodo
------
- Busqueda con Optuna (TPE, sampler con ``seed=42``) sobre tres familias de
  hiperparametros: el bloque de palabras, el bloque de caracteres ``char_wb`` y
  la regresion logistica (ver ``ESPACIO_WORD``, ``ESPACIO_CHAR`` y
  ``ESPACIO_MODELO`` en la Seccion 1).
- Presupuesto: ``n_trials`` = 50 y ``timeout`` = 20 minutos, con poda
  ``MedianPruner`` (arranque de 10 trials, sin poda en el primer pliegue y con
  al menos 3 trials comparables) para no gastar tiempo en trials malos.
- Validacion IDENTICA a la del proyecto: ``StratifiedGroupKFold(5, shuffle=True,
  random_state=42)`` con grupos de near-duplicado (D-004) y estratificacion por
  la combinacion HS x TR x AG; el vectorizador se ajusta dentro de cada pliegue.
- Metrica objetivo: ``f1_media_columnas`` out-of-fold con umbral 0.5 (D-009).
- Regla de decision pre-registrada: se adopta el mejor trial si mejora >= +0.005
  absoluto frente a MOD-03b y mejora en >= 3 de los 5 pliegues. Advertencia
  D-012: en la familia union word+char el gap local-LB medido es de -0.013 a
  -0.016, asi que se exige >= +0.010 local para dar una mejora por realista.

Artefactos
----------
- ``reports/optuna_estudio.json``: resumen del estudio, mejor trial, comparacion
  con MOD-03b, umbrales y resultado de la submission cuando aplica.
- ``reports/optuna_trials.csv``: una fila por trial (parametros, F1 out-of-fold
  del trial completado o media parcial si quedo podado, estado y duracion).
- ``reports/figuras/fig_25_optuna_historia.png``: historia de la optimizacion e
  importancia de los hiperparametros.
- ``submissions/toxic_submission_v05_optuna.csv``: solo si el mejor trial cumple
  la regla de decision; reajuste con todo el train y umbrales por etiqueta
  ajustados SOLO sobre las probabilidades OOF del mejor trial.
- ``models/fase10_optuna_lr_union.joblib``: modelo final reajustado (si aplica).
- Filas ``OPT-01`` y ``S05`` en ``experiments.csv`` y fila ``v05_optuna`` en
  ``submissions/leaderboard_log.csv`` (con ``kaggle_f1`` vacio).

Idempotencia
------------
``n_trials``, ``timeout``, el ``study_name``, la semilla del sampler y la del
modelo son fijos, y cada trial es determinista (misma particion, mismos datos y
mismo solver). El presupuesto de 50 trials tarda ~2-3 minutos, muy por debajo
del timeout de 20 minutos, por lo que el estudio se reproduce igual en la misma
maquina. Los registros (``experiments.csv``, ``submissions/leaderboard_log.csv``
y ``reports/contexto.md``) se reescriben por ``experiment_id`` /
``submission_id``, de modo que dos ejecuciones seguidas no duplican filas. La
unica columna no determinista es la duracion de cada trial (ruido de maquina).

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/09_optuna_tfidf_lr.py
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import joblib
import matplotlib
import numpy as np
import optuna
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from optuna.exceptions import TrialPruned
from optuna.importance import get_param_importances
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.evaluation import ETIQUETAS, MetricasMultiEtiqueta, metricas_multietiqueta

# ---------------------------------------------------------------------------
# Seccion 1 - Configuracion
# ---------------------------------------------------------------------------

CONFIG: Final[dict[str, Any]] = {
    "grupos_path": PROJECT_ROOT / "data" / "processed" / "grupos_near_duplicados.csv",
    "oof_referencia_npz": PROJECT_ROOT / "features" / "oof_mod03b.npz",
    "estudio_json": PROJECT_ROOT / "reports" / "optuna_estudio.json",
    "trials_csv": PROJECT_ROOT / "reports" / "optuna_trials.csv",
    "figura": PROJECT_ROOT / "reports" / "figuras" / "fig_25_optuna_historia.png",
    "experiments_csv": PROJECT_ROOT / "experiments.csv",
    "leaderboard_csv": PROJECT_ROOT / "submissions" / "leaderboard_log.csv",
    "contexto_md": PROJECT_ROOT / "reports" / "contexto.md",
    "submissions_dir": PROJECT_ROOT / "submissions",
    "modelos_dir": PROJECT_ROOT / "models",
    "submission_v05": "toxic_submission_v05_optuna.csv",
    "modelo_joblib": "fase10_optuna_lr_union.joblib",
    "random_state": 42,
    "n_splits": 5,
    "study_name": "opt01_tfidf_lr_union",
    "n_trials": 50,
    "timeout_segundos": 1200,
    "sampler_n_startup_trials": 10,
    "pruner_n_startup_trials": 10,
    "pruner_n_warmup_steps": 1,
    "pruner_n_min_trials": 3,
    "max_iter": 1000,
    "solver": "liblinear",
    "umbral_base": 0.5,
    "rejilla_umbral": (0.20, 0.80, 0.01),
    "umbral_mejora": 0.005,
    "pliegues_minimos_mejora": 3,
    "umbral_mejora_realista": 0.010,
    # Valor exacto publicado en reports/variantes_representaciones.json para
    # MOD-03b_BEST (la tabla CSV lo redondea a 0.7927).
    "f1_referencia": 0.7926588861781015,
    "f1_referencia_redondeada": 0.7927,
    "tolerancia": 1e-6,
    "id_experimento": "OPT-01",
    "id_submission": "S05",
    "id_leaderboard": "v05_optuna",
    "n_filas_test": 1154,
    "columnas_submission": ("id", "HS", "TR", "AG"),
    "competencia": "KAGGLE_TOXIC_2026_PUJ",
    "fecha": "2026-09-22",
    "encabezado_contexto": "## Fase 10/27 - ml-trainer (2026-09-22)",
    "script": "experiments/09_optuna_tfidf_lr.py",
}

# Espacio de busqueda acotado al entorno del ganador. Los rangos de n-gramas se
# codifican como texto ("1-2") para que el muestreo sea un categorico simple y
# el CSV de trials quede legible; se traducen a tuplas en params_desde_flat.
ESPACIO_WORD: Final[dict[str, tuple]] = {
    "ngram_range": ("1-1", "1-2", "1-3"),
    "min_df": (1, 2, 3, 5),
    "max_df": (0.9, 1.0),
    "sublinear_tf": (True, False),
}
ESPACIO_CHAR: Final[dict[str, tuple]] = {
    "ngram_range": ("3-5", "3-6", "2-5", "4-6"),
    "min_df": (2, 3, 5),
    "sublinear_tf": (True, False),
}
ESPACIO_MODELO: Final[dict[str, tuple | tuple[float, float]]] = {
    "C": (0.3, 10.0),
    "class_weight": ("balanced", "none"),
}

# Configuracion de MOD-03b_BEST copiada literalmente de
# experiments/04_refit_submissions_v02_v03.py (REP-05 + LR balanced C=2.0). Se
# usa para el respaldo de la referencia y como documentacion del punto de
# partida; la busqueda puede reencontrarla porque esta dentro del espacio.
PARAMS_REFERENCIA: Final[dict[str, dict[str, Any]]] = {
    "word": {
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 1.0,
        "sublinear_tf": True,
    },
    "char": {
        "ngram_range": (3, 5),
        "min_df": 2,
        "max_df": 1.0,
        "sublinear_tf": True,
    },
    "modelo": {"C": 2.0, "class_weight": "balanced"},
}

COLUMNAS_TRIALS: Final[tuple[str, ...]] = (
    "trial",
    "estado",
    "duracion_seg",
    "f1_media_columnas",
    "f1_parcial_media_pliegues",
    "f1_HS",
    "f1_TR",
    "f1_AG",
    "f1_macro6",
    "f1_pliegues",
    "dimension_media_pliegues",
    "word_ngram_range",
    "word_min_df",
    "word_max_df",
    "word_sublinear_tf",
    "char_ngram_range",
    "char_min_df",
    "char_sublinear_tf",
    "lr_C",
    "lr_class_weight",
    "params",
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

ESTADO_COMPLETADO: Final[str] = "completado"
ESTADO_PODADO: Final[str] = "podado"

CallbackPliegue = Callable[[int, float], None]


def log(message: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        message: texto a imprimir.
    """
    print(f"[09] {message}", flush=True)


def configurar_estilo() -> None:
    """Fija el estilo de las figuras (mismo criterio que las fases previas)."""
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["font.size"] = 10


# ---------------------------------------------------------------------------
# Seccion 2 - Datos, particion y referencia
# ---------------------------------------------------------------------------


def cargar_datos() -> tuple[pd.DataFrame, pd.Series]:
    """Carga el train oficial y la asignacion de grupos de near-duplicado.

    Returns:
        Tupla ``(train, grupos)`` donde ``grupos`` es una serie indexada por
        ``id`` con el numero de grupo de near-duplicado.

    Raises:
        FileNotFoundError: si falta la particion generada en la Fase 0.
    """
    train = data_module.read_train()
    grupos_path: Path = CONFIG["grupos_path"]
    if not grupos_path.exists():
        raise FileNotFoundError(
            f"Falta {grupos_path}. Ejecuta experiments/00_auditoria_datos.py primero."
        )
    grupos = pd.read_csv(grupos_path).set_index("id")["grupo"]
    return train, grupos


def construir_pliegues(
    train: pd.DataFrame, grupos: pd.Series
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construye la particion 5-fold agrupada y estratificada (D-004).

    Replica exactamente la particion de las Fases 4-5, 13a, 13b y 6-8 para que
    todas las metricas sean comparables.

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
            f"pliegue {numero}: {len(entrenamiento)} train / {len(validacion)} validacion "
            f"| grupos validacion {asignacion.iloc[validacion].nunique()}"
        )
    return pliegues


def predecir_con_umbrales(
    probabilidad: np.ndarray, umbrales: Sequence[float]
) -> np.ndarray:
    """Umbraliza una matriz de probabilidades columna a columna.

    Args:
        probabilidad: matriz ``(n, 3)`` con probabilidades de la clase positiva.
        umbrales: umbral por etiqueta, en el orden de :data:`ETIQUETAS`.

    Returns:
        Matriz ``(n, 3)`` con valores en ``{0, 1}``.
    """
    return (probabilidad >= np.asarray(umbrales, dtype=float)[None, :]).astype(int)


def metricas_con_umbral(
    y_true: np.ndarray, probabilidad: np.ndarray, umbral: float
) -> MetricasMultiEtiqueta:
    """Calcula las metricas multi-etiqueta con un umbral comun.

    Args:
        y_true: matriz ``(n, 3)`` con las etiquetas verdaderas.
        probabilidad: matriz ``(n, 3)`` con probabilidades de la clase positiva.
        umbral: umbral aplicado a las tres etiquetas.

    Returns:
        Diccionario de metricas de :func:`src.evaluation.metricas_multietiqueta`.
    """
    return metricas_multietiqueta(
        y_true, predecir_con_umbrales(probabilidad, [umbral] * len(ETIQUETAS))
    )


def metricas_por_pliegue(
    y_true: np.ndarray,
    probabilidad: np.ndarray,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
    umbral: float,
) -> list[dict]:
    """Calcula las metricas de cada pliegue con un umbral comun.

    Args:
        y_true: matriz ``(n, 3)`` con las etiquetas verdaderas.
        probabilidad: matriz ``(n, 3)`` con probabilidades de la clase positiva.
        pliegues: particion usada para generar las probabilidades.
        umbral: umbral aplicado a las tres etiquetas.

    Returns:
        Lista de diccionarios con ``pliegue``, ``n_validacion`` y las metricas.
    """
    resultado: list[dict] = []
    for numero, (_, validacion) in enumerate(pliegues, start=1):
        metrica = dict(metricas_con_umbral(y_true[validacion], probabilidad[validacion], umbral))
        metrica["pliegue"] = numero
        metrica["n_validacion"] = len(validacion)
        resultado.append(metrica)
    return resultado


def probabilidades_referencia(
    train: pd.DataFrame,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
    y_true: np.ndarray,
) -> np.ndarray:
    """Recalcula las probabilidades OOF de MOD-03b_BEST (respaldo).

    Solo se usa si ``features/oof_mod03b.npz`` no existe o no esta alineado con
    el train: vuelve a entrenar la configuracion de referencia con la particion
    del proyecto.

    Args:
        train: DataFrame de entrenamiento con ``text``.
        pliegues: particion agrupada compartida.
        y_true: matriz ``(n, 3)`` con las etiquetas verdaderas.

    Returns:
        Matriz ``(n, 3)`` con las probabilidades out-of-fold de la referencia.
    """
    resultado = evaluar_configuracion(PARAMS_REFERENCIA, train, pliegues, y_true)
    return resultado.probabilidad


def cargar_referencia(
    train: pd.DataFrame,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
    y_true: np.ndarray,
) -> tuple[np.ndarray, list[float], dict]:
    """Carga la referencia MOD-03b_BEST y verifica que reproduce 0.7927.

    Args:
        train: DataFrame de entrenamiento con ``id`` y ``text``.
        pliegues: particion agrupada compartida.
        y_true: matriz ``(n, 3)`` con las etiquetas verdaderas.

    Returns:
        Tupla ``(probabilidad, f1_por_pliegue, detalle)``; ``detalle`` documenta
        la fuente y las metricas de la referencia.

    Raises:
        AssertionError: si la F1 de la referencia no coincide con el valor
            publicado dentro de la tolerancia declarada.
    """
    ruta: Path = CONFIG["oof_referencia_npz"]
    fuente = str(ruta.relative_to(PROJECT_ROOT))
    probabilidad: np.ndarray | None = None
    if ruta.exists():
        with np.load(ruta, allow_pickle=False) as datos:
            ids = datos["ids"]
            y_guardado = datos["y_true"]
            probabilidad = np.asarray(datos["proba"], dtype=float)
        alineado = list(ids) == [str(valor) for valor in train["id"]] and np.array_equal(
            y_guardado, y_true
        )
        if not alineado:
            log("aviso: el .npz de referencia no esta alineado con train; se recalcula")
            probabilidad = None
    if probabilidad is None:
        fuente = "recalculo interno con PARAMS_REFERENCIA"
        probabilidad = probabilidades_referencia(train, pliegues, y_true)

    metricas = metricas_con_umbral(y_true, probabilidad, CONFIG["umbral_base"])
    f1_reproducida = float(metricas["f1_media_columnas"])
    referencia = float(CONFIG["f1_referencia"])
    diferencia = f1_reproducida - referencia
    log(
        f"referencia MOD-03b ({fuente}): f1_media_columnas {f1_reproducida:.6f} "
        f"(publicada {referencia:.6f}; diferencia {diferencia:+.3e})"
    )
    if abs(diferencia) > CONFIG["tolerancia"]:
        raise AssertionError(
            "La referencia MOD-03b no reproduce el valor publicado: "
            f"{f1_reproducida!r} vs {referencia!r}"
        )
    f1_pliegues = [
        round(float(metrica["f1_media_columnas"]), 6)
        for metrica in metricas_por_pliegue(
            y_true, probabilidad, pliegues, CONFIG["umbral_base"]
        )
    ]
    detalle = {
        "fuente": fuente,
        "f1_media_columnas": f1_reproducida,
        "f1_media_columnas_publicada": referencia,
        "diferencia_vs_publicada": diferencia,
        "f1_macro6": float(metricas["f1_macro6"]),
        "f1_por_etiqueta": {
            etiqueta: round(valor, 6)
            for etiqueta, valor in metricas["f1_por_etiqueta"].items()
        },
        "f1_por_pliegue": f1_pliegues,
        "umbral": CONFIG["umbral_base"],
    }
    return probabilidad, f1_pliegues, detalle


# ---------------------------------------------------------------------------
# Seccion 3 - Espacio de busqueda y evaluacion de una configuracion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResultadoEvaluacion:
    """Resultado de evaluar una configuracion con la particion agrupada.

    Attributes:
        probabilidad: matriz ``(n, 3)`` con probabilidades out-of-fold.
        metricas_oof: metricas sobre las predicciones out-of-fold completas.
        metricas_pliegues: metricas por pliegue con umbral 0.5.
        dimensiones_pliegue: numero de features del vectorizador de cada pliegue.
        dimension_train: tamano del vocabulario ajustado con todo el train, o
            ``None`` si no se solicito.
        segundos: tiempo de evaluacion.
    """

    probabilidad: np.ndarray
    metricas_oof: MetricasMultiEtiqueta
    metricas_pliegues: list[dict]
    dimensiones_pliegue: list[int]
    dimension_train: int | None
    segundos: float


def _parse_rango(texto: str) -> tuple[int, int]:
    """Traduce un rango de n-gramas codificado como texto a una tupla.

    Args:
        texto: rango en formato ``"min-max"``.

    Returns:
        Tupla ``(min, max)`` de enteros.

    Raises:
        ValueError: si el texto no tiene el formato esperado.
    """
    partes = texto.split("-")
    if len(partes) != 2 or not all(parte.isdigit() for parte in partes):
        raise ValueError(f"Rango de n-gramas invalido: {texto!r}")
    return int(partes[0]), int(partes[1])


def params_desde_flat(plano: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Convierte los parametros planos de Optuna a la estructura de evaluacion.

    Args:
        plano: diccionario ``nombre_parametro -> valor`` del trial.

    Returns:
        Diccionario con tres bloques: ``word``, ``char`` y ``modelo``.
    """
    return {
        "word": {
            "ngram_range": _parse_rango(str(plano["word_ngram_range"])),
            "min_df": int(plano["word_min_df"]),
            "max_df": float(plano["word_max_df"]),
            "sublinear_tf": bool(plano["word_sublinear_tf"]),
        },
        "char": {
            "ngram_range": _parse_rango(str(plano["char_ngram_range"])),
            "min_df": int(plano["char_min_df"]),
            "sublinear_tf": bool(plano["char_sublinear_tf"]),
        },
        "modelo": {
            "C": float(plano["lr_C"]),
            "class_weight": str(plano["lr_class_weight"]),
        },
    }


def muestrear_parametros(trial: optuna.Trial) -> dict[str, dict[str, Any]]:
    """Muestrea una configuracion del espacio acotado.

    Args:
        trial: trial de Optuna en curso.

    Returns:
        Diccionario con los bloques ``word``, ``char`` y ``modelo``.
    """
    for nombre, opciones in ESPACIO_WORD.items():
        trial.suggest_categorical(f"word_{nombre}", opciones)
    for nombre, opciones in ESPACIO_CHAR.items():
        trial.suggest_categorical(f"char_{nombre}", opciones)
    trial.suggest_float(
        "lr_C", ESPACIO_MODELO["C"][0], ESPACIO_MODELO["C"][1], log=True
    )
    trial.suggest_categorical("lr_class_weight", ESPACIO_MODELO["class_weight"])
    return params_desde_flat(trial.params)


def construir_vectorizador(params: dict[str, dict[str, Any]]) -> FeatureUnion:
    """Crea la union de vectorizadores del trial.

    ``lowercase=True`` y ``strip_accents=None`` son fijos: el corpus ya esta
    normalizado a ASCII (D-010) y no se "inventan" diacriticos.

    Args:
        params: bloques ``word`` y ``char`` con los hiperparametros TF-IDF.

    Returns:
        ``FeatureUnion`` con el bloque de palabras y el de caracteres.
    """
    analizadores = {"word": "word", "char": "char_wb"}
    bloques = []
    for indice, clave in enumerate(("word", "char")):
        parametros = {
            "analyzer": analizadores[clave],
            "lowercase": True,
            "strip_accents": None,
            **params[clave],
        }
        bloques.append((f"bloque_{indice}", TfidfVectorizer(**parametros)))
    return FeatureUnion(bloques)


def construir_modelo(params: dict[str, dict[str, Any]]) -> LogisticRegression:
    """Instancia un clasificador binario con los hiperparametros del trial.

    Args:
        params: bloque ``modelo`` con ``C`` y ``class_weight``; el valor ``none``
            se traduce a ``None`` (sin reponderacion de clases).

    Returns:
        ``LogisticRegression`` sin ajustar.
    """
    class_weight = params["modelo"]["class_weight"]
    return LogisticRegression(
        C=float(params["modelo"]["C"]),
        class_weight=None if class_weight == "none" else class_weight,
        max_iter=CONFIG["max_iter"],
        solver=CONFIG["solver"],
        random_state=CONFIG["random_state"],
    )


def evaluar_configuracion(
    params: dict[str, dict[str, Any]],
    train: pd.DataFrame,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
    y_true: np.ndarray,
    callback_pliegue: CallbackPliegue | None = None,
    incluir_dimension_train: bool = False,
) -> ResultadoEvaluacion:
    """Evalua una configuracion con la particion agrupada compartida.

    El vectorizador se ajusta SIEMPRE con el pliegue de entrenamiento y se
    transforma el de validacion; nunca se ve el pliegue de validacion ni el test.

    Args:
        params: bloques ``word``, ``char`` y ``modelo`` de la configuracion.
        train: DataFrame de entrenamiento con ``text``.
        pliegues: particion agrupada compartida.
        y_true: matriz ``(n, 3)`` con las etiquetas verdaderas.
        callback_pliegue: funcion opcional ``(numero, f1)`` que se invoca tras
            cada pliegue; se usa para reportar y podar en Optuna.
        incluir_dimension_train: si es True, ajusta ademas el vectorizador con
            todo el train para reportar la dimension final (solo diagnostico).

    Returns:
        :class:`ResultadoEvaluacion` con las probabilidades y las metricas.
    """
    inicio = time.perf_counter()
    probabilidad = np.zeros((len(train), len(ETIQUETAS)), dtype=float)
    dimensiones: list[int] = []
    for numero, (entrenamiento, validacion) in enumerate(pliegues, start=1):
        vectorizador = construir_vectorizador(params)
        matriz_entrenamiento = vectorizador.fit_transform(
            train["text"].iloc[entrenamiento]
        )
        matriz_validacion = vectorizador.transform(train["text"].iloc[validacion])
        dimensiones.append(int(matriz_entrenamiento.shape[1]))
        for indice in range(len(ETIQUETAS)):
            modelo = construir_modelo(params)
            modelo.fit(matriz_entrenamiento, y_true[entrenamiento, indice])
            probabilidad[validacion, indice] = modelo.predict_proba(matriz_validacion)[
                :, 1
            ]
        if callback_pliegue is not None:
            f1_pliegue = float(
                metricas_con_umbral(
                    y_true[validacion], probabilidad[validacion], CONFIG["umbral_base"]
                )["f1_media_columnas"]
            )
            callback_pliegue(numero, f1_pliegue)

    metricas_oof = metricas_con_umbral(y_true, probabilidad, CONFIG["umbral_base"])
    resultado = ResultadoEvaluacion(
        probabilidad=probabilidad,
        metricas_oof=metricas_oof,
        metricas_pliegues=metricas_por_pliegue(
            y_true, probabilidad, pliegues, CONFIG["umbral_base"]
        ),
        dimensiones_pliegue=dimensiones,
        dimension_train=(
            int(construir_vectorizador(params).fit_transform(train["text"]).shape[1])
            if incluir_dimension_train
            else None
        ),
        segundos=round(time.perf_counter() - inicio, 2),
    )
    return resultado


# ---------------------------------------------------------------------------
# Seccion 4 - Estudio de Optuna
# ---------------------------------------------------------------------------


def construir_objetivo(
    train: pd.DataFrame,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
    y_true: np.ndarray,
) -> Callable[[optuna.Trial], float]:
    """Fabrica la funcion objetivo del estudio.

    Args:
        train: DataFrame de entrenamiento con ``text``.
        pliegues: particion agrupada compartida.
        y_true: matriz ``(n, 3)`` con las etiquetas verdaderas.

    Returns:
        Funcion ``objetivo(trial) -> f1_media_columnas`` que reporta la F1 de
        cada pliegue al pruner y devuelve la metrica primaria out-of-fold.
    """

    def objetivo(trial: optuna.Trial) -> float:
        """Evalua una configuracion muestreada y devuelve su F1 OOF."""
        params = muestrear_parametros(trial)
        reportados: list[float] = []

        def reportar(numero: int, f1_pliegue: float) -> None:
            """Reporta la F1 de un pliegue y poda el trial si va mal.

            Args:
                numero: numero de pliegue (paso del pruner).
                f1_pliegue: F1 del pliegue recien evaluado.

            Raises:
                TrialPruned: si el pruner decide detener el trial.
            """
            reportados.append(round(f1_pliegue, 6))
            trial.set_user_attr("f1_pliegues", list(reportados))
            trial.report(f1_pliegue, step=numero)
            if trial.should_prune():
                raise TrialPruned()

        resultado = evaluar_configuracion(
            params, train, pliegues, y_true, callback_pliegue=reportar
        )
        metricas = resultado.metricas_oof
        trial.set_user_attr("f1_media_columnas", float(metricas["f1_media_columnas"]))
        trial.set_user_attr("f1_macro6", float(metricas["f1_macro6"]))
        trial.set_user_attr(
            "f1_por_etiqueta",
            {
                etiqueta: round(float(valor), 6)
                for etiqueta, valor in metricas["f1_por_etiqueta"].items()
            },
        )
        trial.set_user_attr(
            "f1_pliegues",
            [
                round(float(metrica["f1_media_columnas"]), 6)
                for metrica in resultado.metricas_pliegues
            ],
        )
        trial.set_user_attr(
            "dimension_media_pliegues",
            round(float(np.mean(resultado.dimensiones_pliegue)), 1),
        )
        trial.set_user_attr("segundos", resultado.segundos)
        return float(metricas["f1_media_columnas"])

    return objetivo


def crear_estudio() -> optuna.Study:
    """Crea el estudio de Optuna con sampler y pruner fijos.

    Returns:
        Estudio en memoria con ``study_name``, TPE con ``seed=42`` y
        ``MedianPruner`` con arranque y warmup documentados.
    """
    return optuna.create_study(
        study_name=CONFIG["study_name"],
        direction="maximize",
        sampler=TPESampler(
            seed=CONFIG["random_state"],
            n_startup_trials=CONFIG["sampler_n_startup_trials"],
        ),
        pruner=MedianPruner(
            n_startup_trials=CONFIG["pruner_n_startup_trials"],
            n_warmup_steps=CONFIG["pruner_n_warmup_steps"],
            n_min_trials=CONFIG["pruner_n_min_trials"],
        ),
    )


def media_parcial_pliegues(trial: optuna.trial.FrozenTrial) -> float | str:
    """Calcula la media de los pliegues reportados antes de podar un trial.

    Optuna deja ``trial.value`` con el ultimo valor reportado cuando el trial se
    poda, por lo que ese valor NO es la metrica out-of-fold de 5 pliegues. Se
    conserva aparte y con un nombre explicito para no confundirlo con
    ``f1_media_columnas``.

    Args:
        trial: trial del estudio.

    Returns:
        Media de los pliegues reportados, o cadena vacia si no hay ninguno.
    """
    pliegues = trial.user_attrs.get("f1_pliegues", [])
    if not pliegues:
        return ""
    return round(float(np.mean(pliegues)), 6)


def construir_tabla_trials(study: optuna.Study) -> pd.DataFrame:
    """Construye la tabla de trials del estudio.

    ``f1_media_columnas`` solo se rellena para los trials completados (los 5
    pliegues evaluados); en los trials podados se deja vacia y la informacion
    parcial va en ``f1_parcial_media_pliegues``.

    Args:
        study: estudio terminado.

    Returns:
        DataFrame con una fila por trial (incluidos los podados) y las columnas
        de :data:`COLUMNAS_TRIALS`.
    """
    filas: list[dict] = []
    for trial in study.trials:
        completado = trial.state == optuna.trial.TrialState.COMPLETE
        if completado:
            estado = ESTADO_COMPLETADO
        elif trial.state == optuna.trial.TrialState.PRUNED:
            estado = ESTADO_PODADO
        else:
            estado = trial.state.name.lower()
        por_etiqueta = trial.user_attrs.get("f1_por_etiqueta", {})
        filas.append(
            {
                "trial": trial.number,
                "estado": estado,
                "duracion_seg": (
                    round(trial.duration.total_seconds(), 2)
                    if trial.duration is not None
                    else ""
                ),
                "f1_media_columnas": (
                    round(float(trial.value), 6)
                    if completado and trial.value is not None
                    else ""
                ),
                "f1_parcial_media_pliegues": (
                    "" if completado else media_parcial_pliegues(trial)
                ),
                "f1_HS": por_etiqueta.get("HS", ""),
                "f1_TR": por_etiqueta.get("TR", ""),
                "f1_AG": por_etiqueta.get("AG", ""),
                "f1_macro6": (
                    trial.user_attrs.get("f1_macro6", "") if completado else ""
                ),
                "f1_pliegues": json.dumps(trial.user_attrs.get("f1_pliegues", [])),
                "dimension_media_pliegues": trial.user_attrs.get(
                    "dimension_media_pliegues", ""
                ),
                "word_ngram_range": trial.params.get("word_ngram_range", ""),
                "word_min_df": trial.params.get("word_min_df", ""),
                "word_max_df": trial.params.get("word_max_df", ""),
                "word_sublinear_tf": trial.params.get("word_sublinear_tf", ""),
                "char_ngram_range": trial.params.get("char_ngram_range", ""),
                "char_min_df": trial.params.get("char_min_df", ""),
                "char_sublinear_tf": trial.params.get("char_sublinear_tf", ""),
                "lr_C": (
                    round(float(trial.params["lr_C"]), 6)
                    if "lr_C" in trial.params
                    else ""
                ),
                "lr_class_weight": trial.params.get("lr_class_weight", ""),
                "params": json.dumps(trial.params, ensure_ascii=False, default=str),
            }
        )
    return pd.DataFrame(filas, columns=list(COLUMNAS_TRIALS))


def resumir_estudio(study: optuna.Study, segundos: float) -> dict:
    """Resume el estado del estudio y su presupuesto.

    Args:
        study: estudio terminado.
        segundos: duracion real de ``study.optimize``.

    Returns:
        Diccionario con los conteos por estado, la duracion y la configuracion
        del sampler y del pruner.
    """
    estados = [trial.state.name for trial in study.trials]
    return {
        "study_name": study.study_name,
        "direccion": "maximize",
        "n_trials_solicitados": CONFIG["n_trials"],
        "timeout_segundos": CONFIG["timeout_segundos"],
        "n_trials_ejecutados": len(study.trials),
        "n_completados": estados.count("COMPLETE"),
        "n_podados": estados.count("PRUNED"),
        "n_fallidos": estados.count("FAIL"),
        "duracion_seg": round(segundos, 2),
        "sampler": (
            f"TPESampler(seed={CONFIG['random_state']}, "
            f"n_startup_trials={CONFIG['sampler_n_startup_trials']})"
        ),
        "pruner": (
            f"MedianPruner(n_startup_trials={CONFIG['pruner_n_startup_trials']}, "
            f"n_warmup_steps={CONFIG['pruner_n_warmup_steps']}, "
            f"n_min_trials={CONFIG['pruner_n_min_trials']})"
        ),
        "metrica_objetivo": "f1_media_columnas out-of-fold con umbral 0.5",
        "espacio_busqueda": {
            "word": {clave: list(valor) for clave, valor in ESPACIO_WORD.items()},
            "char": {clave: list(valor) for clave, valor in ESPACIO_CHAR.items()},
            "modelo": {
                "C": {
                    "tipo": "log-uniforme",
                    "minimo": ESPACIO_MODELO["C"][0],
                    "maximo": ESPACIO_MODELO["C"][1],
                },
                "class_weight": list(ESPACIO_MODELO["class_weight"]),
            },
        },
    }


def comparar_con_referencia(
    f1_mejor: float, pliegues_mejor: Sequence[float], pliegues_referencia: Sequence[float]
) -> dict:
    """Compara el mejor trial con la referencia MOD-03b y aplica la regla.

    Args:
        f1_mejor: ``f1_media_columnas`` out-of-fold del mejor trial.
        pliegues_mejor: F1 por pliegue del mejor trial.
        pliegues_referencia: F1 por pliegue de MOD-03b (umbral 0.5).

    Returns:
        Diccionario con el delta, los pliegues mejorados, la decision de la
        regla pre-registrada y la advertencia de la familia (D-012).
    """
    delta = float(f1_mejor) - float(CONFIG["f1_referencia"])
    pliegues_mejorados = int(
        sum(
            mejor > referencia
            for mejor, referencia in zip(
                pliegues_mejor, pliegues_referencia, strict=True
            )
        )
    )
    cumple_regla = bool(
        delta >= CONFIG["umbral_mejora"]
        and pliegues_mejorados >= CONFIG["pliegues_minimos_mejora"]
    )
    realista = bool(delta >= CONFIG["umbral_mejora_realista"])
    return {
        "f1_mejor_trial": float(f1_mejor),
        "f1_referencia_mod03b": float(CONFIG["f1_referencia"]),
        "delta": delta,
        "pliegues_mejorados": pliegues_mejorados,
        "n_pliegues": len(pliegues_referencia),
        "f1_por_pliegue_mejor": [round(float(valor), 6) for valor in pliegues_mejor],
        "f1_por_pliegue_referencia": [
            round(float(valor), 6) for valor in pliegues_referencia
        ],
        "regla": (
            f"adoptar si delta >= +{CONFIG['umbral_mejora']:.3f} absoluto y "
            f"mejora en >= {CONFIG['pliegues_minimos_mejora']}/5 pliegues"
        ),
        "cumple_regla": cumple_regla,
        "adoptado": cumple_regla,
        "mejora_realista_d012": realista,
        "advertencia_d012": (
            "En la familia union word+char el gap local-LB medido es -0.013 a -0.016 "
            "(v02 -0.0128, v03 -0.0158, v04 -0.0234); D-012 exige >= +0.010 local para "
            "considerar una mejora realista en el leaderboard. Un delta entre +0.005 y "
            "+0.010 cumple la regla pre-registrada pero debe confirmarse con una sonda "
            "de LB antes de darla por real."
        ),
    }


# ---------------------------------------------------------------------------
# Seccion 5 - Umbrales por etiqueta sobre el OOF del mejor trial
# ---------------------------------------------------------------------------


def rejilla_umbrales() -> np.ndarray:
    """Construye la rejilla de umbrales 0.20-0.80 con paso 0.01.

    Returns:
        Vector de umbrales redondeados a dos decimales.
    """
    inicio, fin, paso = CONFIG["rejilla_umbral"]
    return np.round(np.arange(inicio, fin + paso / 2, paso), 2)


def optimizar_umbral(
    y_true: np.ndarray, probabilidad: np.ndarray, rejilla: np.ndarray
) -> tuple[float, float]:
    """Busca el umbral que maximiza la F1 binaria de una etiqueta.

    La busqueda se hace exclusivamente sobre probabilidades out-of-fold. Si dos
    umbrales empatan se conserva el menor (primer maximo de la rejilla), lo que
    deja la decision determinista.

    Args:
        y_true: vector binario verdadero de una etiqueta.
        probabilidad: vector de probabilidades out-of-fold de esa etiqueta.
        rejilla: umbrales candidatos.

    Returns:
        Tupla ``(umbral_optimo, f1_optima)``.
    """
    mejor_umbral = float(rejilla[0])
    mejor_f1 = -1.0
    for umbral in rejilla:
        prediccion = (probabilidad >= umbral).astype(int)
        valor = float(f1_score(y_true, prediccion, zero_division=0))
        if valor > mejor_f1:
            mejor_umbral = float(umbral)
            mejor_f1 = valor
    return mejor_umbral, mejor_f1


def evaluar_umbrales(
    y_true: np.ndarray,
    probabilidad: np.ndarray,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Ajusta y evalua los umbrales por etiqueta sobre el OOF del mejor trial.

    Args:
        y_true: matriz ``(n, 3)`` con las etiquetas verdaderas.
        probabilidad: matriz ``(n, 3)`` con las probabilidades out-of-fold del
            mejor trial.
        pliegues: particion usada para contar la mejora por pliegue.

    Returns:
        Diccionario con los umbrales, las metricas con 0.5 y con los umbrales
        optimos, el recuento de pliegues mejorados y la advertencia de sesgo.
    """
    rejilla = rejilla_umbrales()
    umbrales_optimos: dict[str, float] = {}
    por_etiqueta: dict[str, dict] = {}
    for indice, etiqueta in enumerate(ETIQUETAS):
        verdad = y_true[:, indice]
        columna = probabilidad[:, indice]
        umbral, f1_optima = optimizar_umbral(verdad, columna, rejilla)
        f1_base = float(
            f1_score(
                verdad,
                (columna >= CONFIG["umbral_base"]).astype(int),
                zero_division=0,
            )
        )
        umbrales_optimos[etiqueta] = umbral
        por_etiqueta[etiqueta] = {
            "umbral_optimo": umbral,
            "f1_umbral_0.5": f1_base,
            "f1_umbral_optimo": f1_optima,
            "ganancia": f1_optima - f1_base,
        }

    metricas_base = metricas_con_umbral(y_true, probabilidad, CONFIG["umbral_base"])
    umbrales = [umbrales_optimos[etiqueta] for etiqueta in ETIQUETAS]
    metricas_optimas = metricas_multietiqueta(
        y_true, predecir_con_umbrales(probabilidad, umbrales)
    )
    ganancia = float(metricas_optimas["f1_media_columnas"]) - float(
        metricas_base["f1_media_columnas"]
    )

    por_pliegue: list[dict] = []
    for numero, (_, validacion) in enumerate(pliegues, start=1):
        f1_base_pliegue = float(
            metricas_con_umbral(
                y_true[validacion], probabilidad[validacion], CONFIG["umbral_base"]
            )["f1_media_columnas"]
        )
        f1_optimo_pliegue = float(
            metricas_multietiqueta(
                y_true[validacion],
                predecir_con_umbrales(probabilidad[validacion], umbrales),
            )["f1_media_columnas"]
        )
        por_pliegue.append(
            {
                "pliegue": numero,
                "n_validacion": len(validacion),
                "f1_umbral_0.5": f1_base_pliegue,
                "f1_umbrales_optimos": f1_optimo_pliegue,
                "delta": f1_optimo_pliegue - f1_base_pliegue,
            }
        )
    pliegues_mejorados = int(sum(fila["delta"] > 0 for fila in por_pliegue))
    return {
        "rejilla": {
            "inicio": CONFIG["rejilla_umbral"][0],
            "fin": CONFIG["rejilla_umbral"][1],
            "paso": CONFIG["rejilla_umbral"][2],
        },
        "por_etiqueta": por_etiqueta,
        "umbrales_optimos": umbrales_optimos,
        "f1_media_columnas_umbral_0.5": float(metricas_base["f1_media_columnas"]),
        "f1_media_columnas_umbrales_optimos": float(
            metricas_optimas["f1_media_columnas"]
        ),
        "macro6_umbral_0.5": float(metricas_base["f1_macro6"]),
        "macro6_umbrales_optimos": float(metricas_optimas["f1_macro6"]),
        "f1_por_etiqueta_umbrales_optimos": {
            etiqueta: round(float(valor), 6)
            for etiqueta, valor in metricas_optimas["f1_por_etiqueta"].items()
        },
        "ganancia": ganancia,
        "por_pliegue": por_pliegue,
        "pliegues_mejorados": pliegues_mejorados,
        "advertencia_sesgo_optimista": (
            "Los umbrales y la metrica comparten el mismo OOF, asi que la ganancia es "
            "una cota optimista (la verificacion anidada de v03 la redujo de +0.0063 a "
            "+0.0014). Se registran los umbrales y el valor OOF, no una promesa."
        ),
    }


# ---------------------------------------------------------------------------
# Seccion 6 - Refit, submission y validacion
# ---------------------------------------------------------------------------


def ajustar_modelo_final(
    train: pd.DataFrame, params: dict[str, dict[str, Any]]
) -> tuple[FeatureUnion, list[LogisticRegression], int]:
    """Ajusta el vectorizador y los tres modelos con TODO el train.

    Args:
        train: DataFrame de entrenamiento con etiquetas.
        params: configuracion del mejor trial.

    Returns:
        Tupla ``(vectorizador, modelos, dimension)`` ajustados.
    """
    vectorizador = construir_vectorizador(params)
    matriz = vectorizador.fit_transform(train["text"])
    modelos = []
    for etiqueta in ETIQUETAS:
        modelo = construir_modelo(params)
        modelo.fit(matriz, train[etiqueta])
        modelos.append(modelo)
    return vectorizador, modelos, int(matriz.shape[1])


def predecir_test(
    test: pd.DataFrame,
    vectorizador: FeatureUnion,
    modelos: Sequence[LogisticRegression],
) -> np.ndarray:
    """Calcula las probabilidades de la clase positiva en el test.

    Args:
        test: DataFrame de test con ``text``.
        vectorizador: vectorizador ajustado con todo el train.
        modelos: modelos binarios ajustados con todo el train.

    Returns:
        Matriz ``(n, 3)`` con la probabilidad de la clase positiva por etiqueta.
    """
    matriz = vectorizador.transform(test["text"])
    return np.column_stack([modelo.predict_proba(matriz)[:, 1] for modelo in modelos])


def validar_submission(submission: pd.DataFrame, plantilla: pd.DataFrame, ruta: Path) -> None:
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

    La validacion ocurre en memoria: si algo falla se lanza la excepcion y NO se
    escribe ningun archivo invalido.

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


def guardar_modelo(
    vectorizador: FeatureUnion,
    modelos: Sequence[LogisticRegression],
    params: dict[str, dict[str, Any]],
    umbrales: dict[str, float],
    dimension: int,
    n_train: int,
    f1_oof: float,
) -> Path:
    """Guarda el modelo final con vectorizador, modelos, parametros y umbrales.

    Args:
        vectorizador: vectorizador ajustado con todo el train.
        modelos: modelos binarios ajustados con todo el train.
        params: configuracion del mejor trial.
        umbrales: umbral por etiqueta ajustado sobre el OOF del mejor trial.
        dimension: numero de features del vectorizador final.
        n_train: numero de filas con las que se reajusto el modelo.
        f1_oof: ``f1_media_columnas`` out-of-fold del mejor trial.

    Returns:
        Ruta del archivo ``.joblib`` escrito.
    """
    destino: Path = CONFIG["modelos_dir"] / CONFIG["modelo_joblib"]
    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizador": vectorizador,
            "modelos": list(modelos),
            "params": params,
            "umbrales": umbrales,
            "etiquetas": list(ETIQUETAS),
            "dimension": dimension,
            "f1_oof": f1_oof,
            "entrenado_con": f"todo el train ({n_train} filas)",
            "fecha": CONFIG["fecha"],
            "script": CONFIG["script"],
        },
        destino,
    )
    return destino


# ---------------------------------------------------------------------------
# Seccion 7 - Figura, registro y contexto
# ---------------------------------------------------------------------------


def guardar_figura_historia(
    study: optuna.Study, ruta: Path, f1_referencia: float
) -> Path:
    """Genera la figura de historia de optimizacion e importancias.

    Args:
        study: estudio terminado.
        ruta: ruta destino del PNG.
        f1_referencia: F1 de MOD-03b (linea de referencia).

    Returns:
        Ruta del archivo escrito.
    """
    configurar_estilo()
    completados = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    podados = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED
    ]
    figura, (eje_a, eje_b) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    eje_a.scatter(
        [trial.number for trial in completados],
        [trial.value for trial in completados],
        s=26,
        color="#1f77b4",
        label="trial completado",
    )
    parciales = [
        (trial.number, float(valor))
        for trial, valor in ((trial, media_parcial_pliegues(trial)) for trial in podados)
        if valor != ""
    ]
    if parciales:
        eje_a.scatter(
            [numero for numero, _ in parciales],
            [valor for _, valor in parciales],
            s=34,
            marker="x",
            color="#7f7f7f",
            label="trial podado (media parcial de pliegues)",
        )
    if completados:
        numeros = [trial.number for trial in completados]
        valores = [trial.value for trial in completados]
        eje_a.plot(
            numeros,
            np.maximum.accumulate(np.asarray(valores, dtype=float)),
            color="#ff7f0e",
            linewidth=1.8,
            label="mejor acumulado",
        )
    eje_a.axhline(
        f1_referencia,
        color="#d62728",
        linestyle="--",
        linewidth=1.3,
        label=f"MOD-03b ({f1_referencia:.4f})",
    )
    todos = (
        [trial.value for trial in completados]
        + [valor for _, valor in parciales]
        + [f1_referencia]
    )
    margen = 0.004
    eje_a.set_ylim(min(todos) - margen, max(todos) + margen)
    eje_a.set_xlabel("trial (orden de ejecucion)")
    eje_a.set_ylabel("f1_media_columnas OOF (umbral 0.5)")
    eje_a.set_title("Historia de la optimizacion (Optuna TPE, seed=42)")
    eje_a.legend(fontsize=8, loc="lower right")

    try:
        importancias = get_param_importances(study)
    except (RuntimeError, ValueError) as error:
        log(f"importancias no disponibles: {error}")
        importancias = {}
    if importancias:
        nombres = list(importancias.keys())[::-1]
        eje_b.barh(
            nombres, [importancias[nombre] for nombre in nombres], color="#2ca02c"
        )
        eje_b.set_xlabel("importancia relativa")
        eje_b.set_title("Importancia de los hiperparametros")
    else:
        eje_b.text(
            0.5,
            0.5,
            "Importancias no disponibles",
            ha="center",
            va="center",
            transform=eje_b.transAxes,
        )
        eje_b.set_axis_off()

    figura.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(figura)
    log(f"guardada figura {ruta.relative_to(PROJECT_ROOT)}")
    return ruta


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
    """Anade (o refresca) la fila de la submission en ``leaderboard_log.csv``.

    El ``kaggle_f1`` se deja vacio: lo completa Carlos cuando envie la
    submission. La operacion es idempotente por ``submission_id``.

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


def construir_fila_opt01(
    mejor: optuna.trial.FrozenTrial,
    params: dict[str, dict[str, Any]],
    comparacion: dict,
    resumen: dict,
    umbrales: dict,
    dimension: int,
) -> dict:
    """Construye la fila ``OPT-01`` del registro de experimentos.

    Args:
        mejor: mejor trial completado del estudio.
        params: configuracion del mejor trial.
        comparacion: comparacion con MOD-03b y decision de la regla.
        resumen: resumen del estudio (presupuesto, sampler, pruner).
        umbrales: resultado de :func:`evaluar_umbrales`.
        dimension: dimension del vectorizador ajustado con todo el train.

    Returns:
        Diccionario con las columnas de ``COLUMNAS_EXPERIMENTS``.
    """
    adoptado = bool(comparacion["adoptado"])
    decision = (
        "adoptada: el mejor trial cumple la regla pre-registrada; se reajusta con "
        "todo el train y se genera la submission v05"
        if adoptado
        else (
            "descartada: el mejor trial no cumple la regla pre-registrada "
            f"(delta {comparacion['delta']:+.4f} y {comparacion['pliegues_mejorados']}/5 "
            f"pliegues); se mantiene MOD-03b_BEST y NO se genera submission"
        )
    )
    return {
        "experiment_id": CONFIG["id_experimento"],
        "competition": CONFIG["competencia"],
        "question": (
            "¿Alguna combinacion de hiperparametros TF-IDF + regresion logistica, dentro "
            "de un espacio acotado, mejora la f1_media_columnas out-of-fold de "
            "MOD-03b_BEST (0.7927) con el mismo split agrupado?"
        ),
        "hypothesis": (
            "Una busqueda TPE acotada alrededor del ganador (n-gramas, min_df/max_df, "
            "sublinear_tf, C y class_weight) puede mejorar la metrica primaria en al "
            "menos +0.005 sin cambiar la representacion ni el protocolo"
        ),
        "preprocessing": "texto crudo, lowercase, sin stopwords (igual que la Fase 4-5)",
        "feature_groups": "WORD_NGRAM+CHAR_NGRAM",
        "representation": (
            "union TF-IDF word y char_wb con hiperparametros optimizados por Optuna"
        ),
        "model": "LogisticRegression liblinear x3 con hiperparametros optimizados",
        "hyperparameters": json.dumps(
            {
                "params": params,
                "estudio": {
                    "study_name": resumen["study_name"],
                    "n_trials_ejecutados": resumen["n_trials_ejecutados"],
                    "n_completados": resumen["n_completados"],
                    "n_podados": resumen["n_podados"],
                    "sampler": resumen["sampler"],
                    "pruner": resumen["pruner"],
                    "timeout_segundos": resumen["timeout_segundos"],
                },
                "trial": mejor.number,
                "umbrales_por_etiqueta": umbrales["umbrales_optimos"],
            },
            ensure_ascii=False,
        ),
        "validation": (
            "StratifiedGroupKFold(5, shuffle=True, random_state=42) con grupos de "
            "near-duplicado y estratificacion por HSxTRxAG (D-004); vectorizador ajustado "
            "dentro de cada pliegue; metrica objetivo = f1_media_columnas OOF con umbral "
            "0.5; comparacion por pliegue contra MOD-03b_BEST"
        ),
        "f1": round(float(comparacion["f1_mejor_trial"]), 4),
        "macro_f1": round(float(mejor.user_attrs.get("f1_macro6", 0.0)), 4),
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": resumen["duracion_seg"],
        "dimension": dimension,
        "result": json.dumps(
            {
                "trial": mejor.number,
                "f1_media_columnas": comparacion["f1_mejor_trial"],
                "f1_por_etiqueta": mejor.user_attrs.get("f1_por_etiqueta", {}),
                "f1_por_pliegue": comparacion["f1_por_pliegue_mejor"],
                "delta_vs_mod03b": comparacion["delta"],
                "pliegues_mejorados": comparacion["pliegues_mejorados"],
                "f1_media_columnas_umbrales_optimos": umbrales[
                    "f1_media_columnas_umbrales_optimos"
                ],
                "ganancia_umbrales": umbrales["ganancia"],
                "cumple_regla": comparacion["cumple_regla"],
                "mejora_realista_d012": comparacion["mejora_realista_d012"],
            },
            ensure_ascii=False,
        ),
        "interpretation": (
            f"Mejor trial #{mejor.number}: {comparacion['f1_mejor_trial']:.4f} OOF frente a "
            f"{comparacion['f1_referencia_mod03b']:.4f} de MOD-03b "
            f"(delta {comparacion['delta']:+.4f}, {comparacion['pliegues_mejorados']}/5 "
            f"pliegues mejorados)"
        ),
        "decision": decision,
        "date": CONFIG["fecha"],
        "script": CONFIG["script"],
        "notes": (
            "Busqueda Optuna TPE seed=42 con presupuesto 50 trials / 20 min y poda "
            "MedianPruner; el test nunca se uso para elegir el trial ni los umbrales. "
            f"Advertencia D-012: {comparacion['advertencia_d012']}"
        ),
    }


def construir_fila_s05(
    params: dict[str, dict[str, Any]],
    comparacion: dict,
    umbrales: dict,
    ruta: Path,
    resumen_prediccion: dict,
    dimension: int,
) -> dict:
    """Construye la fila ``S05`` del registro de experimentos.

    Args:
        params: configuracion del mejor trial.
        comparacion: comparacion con MOD-03b y decision de la regla.
        umbrales: resultado de :func:`evaluar_umbrales`.
        ruta: ruta de la submission generada.
        resumen_prediccion: conteo de positivos por etiqueta en test y en train.
        dimension: numero de features del vectorizador final.

    Returns:
        Diccionario con las columnas de ``COLUMNAS_EXPERIMENTS``.
    """
    return {
        "experiment_id": CONFIG["id_submission"],
        "competition": CONFIG["competencia"],
        "question": (
            "¿Que submission genera el mejor trial de la optimizacion reajustado con todo "
            "el train y umbrales por etiqueta ajustados sobre su OOF?"
        ),
        "hypothesis": (
            "El mejor trial de Optuna, reajustado con todo el train y con umbrales por "
            "etiqueta, mejora el LB frente a v03 (0.78306)"
        ),
        "preprocessing": "texto crudo, lowercase, sin stopwords",
        "feature_groups": "WORD_NGRAM+CHAR_NGRAM",
        "representation": (
            "union TF-IDF word y char_wb con hiperparametros optimizados por Optuna"
        ),
        "model": "LogisticRegression liblinear x3 con umbral optimo por etiqueta",
        "hyperparameters": json.dumps(
            {
                "params": params,
                "umbral_base": CONFIG["umbral_base"],
                "umbrales_por_etiqueta": umbrales["umbrales_optimos"],
                "postproceso": "ninguno",
            },
            ensure_ascii=False,
        ),
        "validation": (
            "Mejor trial del estudio Optuna (mismo split agrupado que MOD-03b); umbrales "
            "ajustados SOLO sobre las probabilidades out-of-fold del mejor trial; "
            "submission validada contra la plantilla oficial (columnas, 1154 filas, orden "
            "de ids, valores en {0,1}, sin nulos ni duplicados)"
        ),
        "f1": round(float(umbrales["f1_media_columnas_umbrales_optimos"]), 4),
        "macro_f1": round(float(umbrales["macro6_umbrales_optimos"]), 4),
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": "",
        "dimension": dimension,
        "result": json.dumps(
            {
                "submission": str(ruta.relative_to(PROJECT_ROOT)),
                "f1_oof_umbral_0.5": umbrales["f1_media_columnas_umbral_0.5"],
                "f1_oof_umbrales_optimos": umbrales[
                    "f1_media_columnas_umbrales_optimos"
                ],
                "ganancia_umbrales": umbrales["ganancia"],
                "delta_vs_mod03b_umbral_0.5": comparacion["delta"],
                "pliegues_mejorados": comparacion["pliegues_mejorados"],
                "positivos_test": resumen_prediccion["positivos_test"],
                "tasa_positivos_test": resumen_prediccion["tasa_positivos_test"],
                "tasa_positivos_train": resumen_prediccion["tasa_positivos_train"],
            },
            ensure_ascii=False,
        ),
        "interpretation": (
            "Refit del mejor trial de la Fase 10 con todo el train (1731 filas) y "
            "prediccion del test; modelo final en "
            f"models/{CONFIG['modelo_joblib']}"
        ),
        "decision": "submission generada; pendiente de envio por Carlos a Kaggle",
        "date": CONFIG["fecha"],
        "script": CONFIG["script"],
        "notes": (
            "La ganancia de umbrales es una cota optimista (mismo OOF); el delta del trial "
            "frente a MOD-03b es la evidencia principal. Advertencia D-012: en esta familia "
            "se exige >= +0.010 local para considerar la mejora realista en el LB"
        ),
    }


def construir_seccion_contexto(
    resumen: dict,
    comparacion: dict,
    umbrales: dict,
    rutas: dict[str, Path],
    filas: Sequence[str],
    top_trials: Sequence[optuna.trial.FrozenTrial],
) -> str:
    """Redacta la seccion de handoff (maximo 20 lineas) para ``contexto.md``.

    Args:
        resumen: resumen del estudio (presupuesto, sampler, pruner).
        comparacion: comparacion del mejor trial con MOD-03b.
        umbrales: resultado de :func:`evaluar_umbrales`.
        rutas: rutas de los artefactos generados.
        filas: identificadores registrados en ``experiments.csv``.
        top_trials: mejores trials completados, ordenados de mayor a menor F1.

    Returns:
        Texto de la seccion, con encabezado y sin linea final en blanco.

    Raises:
        RuntimeError: si la seccion generada excede las 20 lineas permitidas.
    """
    etiquetas = ", ".join(
        f"{etiqueta} {umbrales['por_etiqueta'][etiqueta]['umbral_optimo']:.2f}"
        for etiqueta in ETIQUETAS
    )
    decision = (
        f"Regla del proyecto (>= +{CONFIG['umbral_mejora']:.3f} y >= "
        f"{CONFIG['pliegues_minimos_mejora']}/5 pliegues): ADOPTADA "
        f"(delta {comparacion['delta']:+.4f}, {comparacion['pliegues_mejorados']}/5 "
        "pliegues mejorados)."
        if comparacion["adoptado"]
        else (
            f"Regla del proyecto (>= +{CONFIG['umbral_mejora']:.3f} y >= "
            f"{CONFIG['pliegues_minimos_mejora']}/5 pliegues): DESCARTADA "
            f"(delta {comparacion['delta']:+.4f}, {comparacion['pliegues_mejorados']}/5 "
            "pliegues mejorados); NO se genera submission."
        )
    )
    linea_realista = (
        "Advertencia D-012: el delta supera +0.010, por lo que la mejora entra en el "
        "rango que en esta familia (gap local-LB -0.013 a -0.016) se considera realista."
        if comparacion["mejora_realista_d012"]
        else (
            "Advertencia D-012: el delta NO alcanza +0.010; en la familia union word+char "
            "(gap local-LB -0.013 a -0.016) una mejora asi debe confirmarse con una sonda "
            "de leaderboard antes de darla por real."
        )
    )
    if "submission" in rutas:
        linea_submission = (
            f"Submission `{rutas['submission'].relative_to(PROJECT_ROOT)}` generada con "
            f"umbrales por etiqueta ({etiquetas}) ajustados SOLO sobre el OOF del mejor "
            "trial y validada contra la plantilla oficial (1154 filas, ids en orden, "
            "valores en {0,1}, sin nulos ni duplicados)."
        )
    else:
        linea_submission = (
            "NO se genera submission v05: el mejor trial no cumple la regla "
            "(se mantienen v02 como control y v03 como mejor envio)."
        )
    top_texto = "; ".join(
        f"#{trial.number} {float(trial.value):.4f}" for trial in top_trials
    )
    lineas = [
        CONFIG["encabezado_contexto"],
        "",
        (
            "Optimizacion acotada de hiperparametros del pipeline ganador con Optuna "
            "(TPE, sampler seed=42), presupuesto 50 trials / 20 min y poda MedianPruner "
            "(arranque 10, warmup 1, minimo 3 comparables)."
        ),
        (
            "Espacio acotado: word n-gramas {(1,1),(1,2),(1,3)} x min_df {1,2,3,5} x "
            "max_df {0.9,1.0} x sublinear_tf {True,False}; char_wb {(3,5),(3,6),(2,5),(4,6)} "
            "x min_df {2,3,5} x sublinear_tf; LR con C log-uniforme [0.3,10] y "
            "class_weight {balanced, none}."
        ),
        (
            "Cada trial = 5-fold del split de D-004 (StratifiedGroupKFold(5, shuffle=True, "
            "random_state=42), near-duplicados, estratificado HSxTRxAG) con vectorizador "
            "ajustado dentro de cada pliegue; metrica objetivo f1_media_columnas OOF (0.5)."
        ),
        (
            f"Estudio: {resumen['n_trials_ejecutados']} trials en "
            f"{resumen['duracion_seg']:.0f} s ({resumen['n_completados']} completados, "
            f"{resumen['n_podados']} podados); top-5: {top_texto}."
        ),
        (
            f"Mejor trial #{comparacion['trial']} con f1_media_columnas OOF "
            f"{comparacion['f1_mejor_trial']:.4f} frente a {comparacion['f1_referencia_mod03b']:.4f} "
            f"de MOD-03b (delta {comparacion['delta']:+.4f}); pliegues por pliegue: mejor "
            f"{comparacion['f1_por_pliegue_mejor']} frente a referencia "
            f"{comparacion['f1_por_pliegue_referencia']}."
        ),
        (
            "Mejores parametros (JSON): "
            + json.dumps(resumen["params_mejor_trial"], ensure_ascii=False)
        ),
        decision,
        linea_realista,
        (
            f"Umbrales por etiqueta del mejor trial (rejilla 0.20-0.80 paso 0.01 sobre su "
            f"OOF): {etiquetas}; f1_media_columnas {umbrales['f1_media_columnas_umbral_0.5']:.4f} "
            f"-> {umbrales['f1_media_columnas_umbrales_optimos']:.4f} "
            f"({umbrales['ganancia']:+.4f}, cota optimista)."
        ),
        linea_submission,
        (
            "Artefactos: `reports/optuna_estudio.json`, `reports/optuna_trials.csv`, "
            "`reports/figuras/fig_25_optuna_historia.png`"
            + (
                ", `models/fase10_optuna_lr_union.joblib`"
                if "modelo" in rutas
                else ""
            )
            + "."
        ),
        (
            "Registro: filas " + ", ".join(filas) + " en `experiments.csv`"
            + (
                " y fila v05_optuna en `submissions/leaderboard_log.csv` (kaggle_f1 vacio, "
                "pendiente de que Carlos envie la submission)."
                if "submission" in rutas
                else " (sin fila de leaderboard: no hay submission nueva)."
            )
        ),
        (
            "Sin fuga: el test solo se uso para predecir cuando se genero la submission; la "
            "seleccion del trial, la de los umbrales y la comparacion con MOD-03b usan "
            "exclusivamente probabilidades out-of-fold de train."
        ),
        (
            "Idempotente: study_name, semilla, n_trials y timeout fijos; los registros se "
            "reescriben por id y no se duplican al re-ejecutar."
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
    encabezado = CONFIG["encabezado_contexto"]
    texto = destino.read_text(encoding="utf-8") if destino.exists() else ""
    if encabezado in texto:
        texto = texto[: texto.index(encabezado)]
    destino.write_text(
        texto.rstrip("\n") + "\n\n" + seccion.rstrip("\n") + "\n", encoding="utf-8"
    )
    return destino


# ---------------------------------------------------------------------------
# Seccion 8 - Flujo principal
# ---------------------------------------------------------------------------


def main() -> None:
    """Ejecuta la Fase 10/27 completa: busqueda, decision, submission y registro."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    inicio = time.perf_counter()

    train, grupos = cargar_datos()
    test = data_module.read_test()
    plantilla = data_module.read_submission()
    if list(plantilla["id"]) != list(test["id"]):
        raise ValueError(
            "El orden de ids del test no coincide con la plantilla oficial; "
            "abortado antes de generar submissions"
        )
    log(
        f"train {train.shape} | test {test.shape} | plantilla {plantilla.shape} | "
        f"grupos {grupos.nunique()}"
    )

    pliegues = construir_pliegues(train, grupos)
    y_true = train[list(ETIQUETAS)].to_numpy()
    _, pliegues_referencia, detalle_referencia = cargar_referencia(
        train, pliegues, y_true
    )

    # 1) Busqueda de hiperparametros.
    log(
        f"Optuna: {CONFIG['n_trials']} trials, timeout {CONFIG['timeout_segundos']} s, "
        f"estudio {CONFIG['study_name']}"
    )
    estudio = crear_estudio()
    inicio_estudio = time.perf_counter()
    estudio.optimize(
        construir_objetivo(train, pliegues, y_true),
        n_trials=CONFIG["n_trials"],
        timeout=CONFIG["timeout_segundos"],
        show_progress_bar=False,
    )
    segundos_estudio = time.perf_counter() - inicio_estudio
    resumen = resumir_estudio(estudio, segundos_estudio)
    log(
        f"estudio terminado: {resumen['n_completados']} completados, "
        f"{resumen['n_podados']} podados en {segundos_estudio:.1f} s"
    )
    if resumen["n_completados"] == 0:
        raise RuntimeError("El estudio no tiene ningun trial completado; nada que decidir")

    completados = [
        trial
        for trial in estudio.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    top_trials = sorted(completados, key=lambda trial: float(trial.value), reverse=True)[
        :5
    ]
    mejor = estudio.best_trial
    params_mejor = params_desde_flat(dict(mejor.params))
    log(
        f"mejor trial #{mejor.number}: f1_media_columnas {float(mejor.value):.6f} con "
        f"{json.dumps(mejor.params, ensure_ascii=False, default=str)}"
    )

    # 2) Reevaluacion del mejor trial para obtener probabilidades OOF y dimension.
    reevaluacion = evaluar_configuracion(
        params_mejor, train, pliegues, y_true, incluir_dimension_train=True
    )
    diferencia_reevaluacion = float(reevaluacion.metricas_oof["f1_media_columnas"]) - float(
        mejor.value
    )
    log(
        f"reevaluacion del mejor trial: "
        f"{float(reevaluacion.metricas_oof['f1_media_columnas']):.6f} "
        f"(diferencia {diferencia_reevaluacion:+.3e} frente al valor del estudio)"
    )
    if abs(diferencia_reevaluacion) > CONFIG["tolerancia"]:
        raise AssertionError(
            "La reevaluacion del mejor trial no coincide con el valor del estudio: "
            f"{diferencia_reevaluacion!r}"
        )
    pliegues_mejor = [
        round(float(metrica["f1_media_columnas"]), 6)
        for metrica in reevaluacion.metricas_pliegues
    ]
    comparacion = comparar_con_referencia(
        float(reevaluacion.metricas_oof["f1_media_columnas"]),
        pliegues_mejor,
        pliegues_referencia,
    )
    comparacion["trial"] = mejor.number

    # 3) Umbrales por etiqueta sobre el OOF del mejor trial.
    umbrales = evaluar_umbrales(y_true, reevaluacion.probabilidad, pliegues)
    log(
        f"umbrales del mejor trial: {umbrales['umbrales_optimos']} | F1 OOF 0.5 "
        f"{umbrales['f1_media_columnas_umbral_0.5']:.4f} -> "
        f"{umbrales['f1_media_columnas_umbrales_optimos']:.4f} "
        f"({umbrales['ganancia']:+.4f})"
    )

    # 4) Submission solo si el mejor trial cumple la regla pre-registrada.
    rutas: dict[str, Path] = {}
    resumen_prediccion: dict | None = None
    tabla_trials = construir_tabla_trials(estudio)
    if comparacion["adoptado"]:
        vectorizador, modelos, dimension = ajustar_modelo_final(train, params_mejor)
        probabilidad_test = predecir_test(test, vectorizador, modelos)
        prediccion_test = predecir_con_umbrales(
            probabilidad_test,
            [umbrales["umbrales_optimos"][etiqueta] for etiqueta in ETIQUETAS],
        )
        rutas["submission"] = escribir_submission(
            test, plantilla, prediccion_test, CONFIG["submission_v05"]
        )
        resumen_prediccion = resumir_prediccion(train, prediccion_test)
        rutas["modelo"] = guardar_modelo(
            vectorizador,
            modelos,
            params_mejor,
            umbrales["umbrales_optimos"],
            int(dimension),
            len(train),
            float(comparacion["f1_mejor_trial"]),
        )
        log(f"modelo final guardado: {rutas['modelo'].name} (dimension {dimension})")
        log(f"test: {json.dumps(resumen_prediccion, ensure_ascii=False)}")
    else:
        dimension = int(np.mean(reevaluacion.dimensiones_pliegue))
        ruta_descartada = CONFIG["submissions_dir"] / CONFIG["submission_v05"]
        if ruta_descartada.exists():
            ruta_descartada.unlink()
            log(
                "el mejor trial no cumple la regla: se elimino la submission obsoleta "
                f"{ruta_descartada.name}"
            )
        log(
            "el mejor trial no cumple la regla pre-registrada: NO se genera submission "
            f"v05 (delta {comparacion['delta']:+.4f}, "
            f"{comparacion['pliegues_mejorados']}/5 pliegues)"
        )

    # 5) Registro en experiments.csv y leaderboard.
    dimension_train = (
        reevaluacion.dimension_train if reevaluacion.dimension_train is not None else dimension
    )
    filas_experiments = [
        construir_fila_opt01(
            mejor, params_mejor, comparacion, resumen, umbrales, int(dimension_train)
        )
    ]
    filas_identificadores = [CONFIG["id_experimento"]]
    if comparacion["adoptado"] and resumen_prediccion is not None:
        filas_experiments.append(
            construir_fila_s05(
                params_mejor,
                comparacion,
                umbrales,
                rutas["submission"],
                resumen_prediccion,
                int(dimension_train),
            )
        )
        filas_identificadores.append(CONFIG["id_submission"])
    escritas = registrar_en_experiments(filas_experiments)
    log(f"filas escritas en experiments.csv: {escritas}")
    if comparacion["adoptado"] and resumen_prediccion is not None:
        escritas_lb = registrar_en_leaderboard(
            [
                {
                    "submission_id": CONFIG["id_leaderboard"],
                    "fecha": CONFIG["fecha"],
                    "archivo": str(rutas["submission"].relative_to(PROJECT_ROOT)),
                    "local_f1": round(
                        float(umbrales["f1_media_columnas_umbrales_optimos"]), 5
                    ),
                    "kaggle_f1": "",
                    "diferencia": "",
                    "features": (
                        "union TF-IDF word y char_wb con hiperparametros optimizados por "
                        "Optuna (Fase 10)"
                    ),
                    "modelo": (
                        "LogisticRegression liblinear x3 (HS, TR, AG) con umbral optimo "
                        "por etiqueta, sin post-proceso"
                    ),
                    "hiperparametros": json.dumps(
                        {
                            "params": params_mejor,
                            "umbrales": umbrales["umbrales_optimos"],
                        },
                        ensure_ascii=False,
                    ),
                    "interpretacion": (
                        f"Mejor trial de la optimizacion Optuna (#{mejor.number}): F1 OOF "
                        f"{comparacion['f1_mejor_trial']:.4f} con umbral 0.5 "
                        f"({comparacion['delta']:+.4f} frente a MOD-03b, "
                        f"{comparacion['pliegues_mejorados']}/5 pliegues) y "
                        f"{umbrales['f1_media_columnas_umbrales_optimos']:.4f} con umbrales "
                        "por etiqueta. Pendiente de envio por Carlos; recordar que en esta "
                        "familia el gap local-LB medido es -0.013 a -0.016."
                    ),
                }
            ]
        )
        log(f"filas escritas en leaderboard_log.csv: {escritas_lb}")

    # 6) Artefactos: tabla de trials, figura, JSON de detalle y contexto.
    CONFIG["trials_csv"].parent.mkdir(parents=True, exist_ok=True)
    tabla_trials.to_csv(CONFIG["trials_csv"], index=False, encoding="utf-8")
    log(f"tabla de trials: {CONFIG['trials_csv'].relative_to(PROJECT_ROOT)}")
    ruta_figura = guardar_figura_historia(
        estudio, CONFIG["figura"], float(CONFIG["f1_referencia"])
    )

    resumen["params_mejor_trial"] = params_mejor
    resumen["dimension_train"] = int(dimension_train)
    resumen["dimension_media_pliegues"] = round(
        float(np.mean(reevaluacion.dimensiones_pliegue)), 1
    )
    detalle = {
        "metadata": {
            "fecha": CONFIG["fecha"],
            "script": CONFIG["script"],
            "competencia": CONFIG["competencia"],
            "fase": "Fase 10/27 (optimizacion acotada de hiperparametros con Optuna)",
            "metrica_primaria": "f1_media_columnas",
            "validacion": (
                f"StratifiedGroupKFold({CONFIG['n_splits']}, shuffle=True, "
                f"random_state={CONFIG['random_state']}) agrupado por near-duplicado y "
                "estratificado por HSxTRxAG"
            ),
            "sin_fuga": (
                "Vectorizador ajustado dentro de cada pliegue; el test solo se uso para "
                "predecir cuando el mejor trial cumplio la regla; ni el trial ni los "
                "umbrales usan el test"
            ),
            "idempotencia": (
                "study_name, semilla, n_trials y timeout fijos; cada trial es "
                "determinista; unica columna no determinista: la duracion"
            ),
            "segundos_estudio": round(segundos_estudio, 2),
            "segundos_totales": round(time.perf_counter() - inicio, 2),
        },
        "espacio_busqueda": resumen["espacio_busqueda"],
        "estudio": resumen,
        "referencia_mod03b": detalle_referencia,
        "mejor_trial": {
            "numero": mejor.number,
            "params": params_mejor,
            "f1_media_columnas": float(comparacion["f1_mejor_trial"]),
            "f1_por_etiqueta": mejor.user_attrs.get("f1_por_etiqueta", {}),
            "f1_macro6": float(mejor.user_attrs.get("f1_macro6", 0.0)),
            "f1_por_pliegue": pliegues_mejor,
            "dimension_train": int(dimension_train),
            "dimension_media_pliegues": resumen["dimension_media_pliegues"],
            "delta_vs_mod03b": comparacion["delta"],
            "pliegues_mejorados": comparacion["pliegues_mejorados"],
            "cumple_regla": comparacion["cumple_regla"],
            "mejora_realista_d012": comparacion["mejora_realista_d012"],
        },
        "comparacion": comparacion,
        "umbrales": umbrales,
        "submission": (
            {
                "ruta": str(rutas["submission"].relative_to(PROJECT_ROOT)),
                "umbrales": umbrales["umbrales_optimos"],
                "modelo": f"models/{CONFIG['modelo_joblib']}",
                "prediccion_test": resumen_prediccion,
            }
            if comparacion["adoptado"] and resumen_prediccion is not None
            else {
                "generada": False,
                "motivo": (
                    "El mejor trial no cumple la regla pre-registrada "
                    f"(delta {comparacion['delta']:+.4f}, "
                    f"{comparacion['pliegues_mejorados']}/5 pliegues)"
                ),
            }
        ),
        "top_trials": [
            {
                "numero": trial.number,
                "f1_media_columnas": float(trial.value),
                "params": trial.params,
            }
            for trial in top_trials
        ],
        "artefactos": {
            "trials_csv": str(CONFIG["trials_csv"].relative_to(PROJECT_ROOT)),
            "figura": str(ruta_figura.relative_to(PROJECT_ROOT)),
        },
        "advertencia_d012": comparacion["advertencia_d012"],
    }
    CONFIG["estudio_json"].parent.mkdir(parents=True, exist_ok=True)
    CONFIG["estudio_json"].write_text(
        json.dumps(detalle, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    log(f"detalle: {CONFIG['estudio_json'].relative_to(PROJECT_ROOT)}")

    # 7) Handoff en reports/contexto.md.
    seccion = construir_seccion_contexto(
        resumen,
        comparacion,
        umbrales,
        rutas,
        filas_identificadores,
        top_trials,
    )
    ruta_contexto = actualizar_contexto(seccion)
    log(f"contexto actualizado: {ruta_contexto.relative_to(PROJECT_ROOT)}")
    log(f"duracion total: {time.perf_counter() - inicio:.2f} s")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - el punto de entrada CLI reporta todo
        import traceback

        traceback.print_exc()
        sys.exit(1)
