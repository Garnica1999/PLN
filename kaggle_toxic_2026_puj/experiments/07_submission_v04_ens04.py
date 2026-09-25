"""07_submission_v04_ens04.py - Fase 13c: submission SONDA v04 con ENS-04.

Pregunta que responde
---------------------
1. Reconstruye el codigo, con el mismo split de las Fases 4-5, 13a y 13b, el
   ensemble ENS-04 (par M2+M3) y su F1 anidada con umbrales medianos por pliegue
   publicada en ``reports/ensembles_resultados.json`` (0.7985)?
2. Que submission de test produce el refit de ese par con todo el train y los
   umbrales medianos HS 0.44 / TR 0.47 / AG 0.44?

Contexto y decision
-------------------
Carlos decidio gastar UN envio de Kaggle en esta sonda (Fase 13c) para medir si
los ensembles generalizan al leaderboard: la Fase 13b no autorizo submission
porque ningun ensemble superaba a ``MOD-03b_BEST`` por >= +0.005 en la
evaluacion anidada, pero el par M2+M3 con umbrales medianos si mejora el mejor
valor local (0.7985 frente a 0.7927 del single a 0.5 y 0.7989 del single con
umbrales optimos, este ultimo una cota optimista). Este script NO envia nada a
Kaggle: solo genera el CSV y registra la fila de seguimiento.

Metodo
------
- Split identico al de las Fases 4-5, 13a y 13b:
  ``StratifiedGroupKFold(5, shuffle=True, random_state=42)`` con grupos =
  componentes de near-duplicado y estratificacion por la combinacion
  HS x TR x AG (D-004).
- Modelos base copiados literalmente de ``experiments/05_ensembles.py``:
  * M2: union TF-IDF word (1,2) + char_wb (3,5) (min_df=2, max_df=1.0,
    sublinear_tf) + ``LogisticRegression(C=2.0, class_weight="balanced",
    max_iter=1000, solver="liblinear")`` por etiqueta; probabilidad con
    ``predict_proba``.
  * M3: la misma representacion (mismo vectorizador por pliegue) +
    ``LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)`` por etiqueta;
    probabilidad con la sigmoide del ``decision_function`` (conversion monotona
    que en 0.5 reproduce exactamente la regla ``decision_function > 0``).
- ENS-04 es el promedio simple de M2 y M3: la Fase 13b verifica que ENS-04 elige
  el par (M2, M3) con pesos (0.5, 0.5) en los CINCO pliegues, por lo que la
  probabilidad anidada del ensemble coincide con el promedio directo de ambas
  bases. Esa equivalencia se comprueba contra el registro de la Fase 13b.
- Umbrales medianos por pliegue (estrategia ``mediana_por_pliegue`` de 05): para
  cada pliegue se optimiza el umbral de cada etiqueta SOLO con sus filas; la
  prediccion de un pliegue usa la mediana de los optimos de los otros cuatro, y
  los umbrales finales del refit son la mediana de los cinco optimos. Rejilla
  0.20-0.80 paso 0.01. El test nunca participa en la eleccion de umbrales.
- Reajuste final: vectorizador y los 3+3 clasificadores ajustados con TODO el
  train (1731 filas), probabilidad de test promediada y umbralizada con los
  umbrales medianos.

Artefactos
----------
- ``submissions/toxic_submission_v04_ens04_mediana.csv``: submission validada
  contra la plantilla oficial (columnas, 1154 filas, ids identicos y en orden,
  valores en {0,1}, sin nulos ni duplicados).
- ``features/oof_ens04_v04.npz``: evidencia del OOF recalculado de M2, M3 y del
  ensemble, con los umbrales y la F1 anidada.
- ``models/fase13c_ens04_m2m3.joblib``: vectorizador, los 6 clasificadores del
  refit, umbrales y metadatos.
- ``reports/fase13c_submission_v04_ens04.json``: detalle completo (referencia,
  verificacion, refit, submission y registro).
- Fila ``S04`` en ``experiments.csv``, fila de seguimiento en
  ``submissions/leaderboard_log.csv`` y seccion de handoff en
  ``reports/contexto.md``.

Idempotencia
------------
Pliegues, probabilidades OOF, umbrales, predicciones y submission son
deterministas. ``experiments.csv`` y ``submissions/leaderboard_log.csv`` se
actualizan eliminando primero la fila de esta fase, y la seccion de
``reports/contexto.md`` se reescribe, por lo que dos ejecuciones seguidas dejan
el mismo contenido salvo los campos de tiempo (ruido de maquina).

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/07_submission_v04_ens04.py
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence
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
    "resultados_json": PROJECT_ROOT / "reports" / "ensembles_resultados.json",
    "oof_bases_npz": PROJECT_ROOT / "features" / "oof_bases.npz",
    "oof_ens04_npz": PROJECT_ROOT / "features" / "oof_ens04_v04.npz",
    "modelo_joblib": PROJECT_ROOT / "models" / "fase13c_ens04_m2m3.joblib",
    "detalle_json": PROJECT_ROOT / "reports" / "fase13c_submission_v04_ens04.json",
    "experiments_csv": PROJECT_ROOT / "experiments.csv",
    "leaderboard_csv": PROJECT_ROOT / "submissions" / "leaderboard_log.csv",
    "contexto_md": PROJECT_ROOT / "reports" / "contexto.md",
    "submissions_dir": PROJECT_ROOT / "submissions",
    "submission": "toxic_submission_v04_ens04_mediana.csv",
    "random_state": 42,
    "n_splits": 5,
    "umbral_base": 0.5,
    "rejilla_umbral": (0.20, 0.80, 0.01),
    # Tolerancias: 1e-3 para la F1 anidada (valor publicado redondeado a 4
    # decimales) y 1e-6 para los umbrales y la comparacion con el .npz previo.
    "tolerancia_anidada": 1e-3,
    "tolerancia_umbral": 1e-6,
    "tolerancia_npz": 1e-6,
    "equivalencia_f1_rapida": 1e-12,
    "id_ensemble": "ENS-04",
    "f1_anidada_referencia": 0.7985,
    "id_submission": "S04",
    "id_leaderboard": "v04_ens04_mediana",
    "n_filas_test": 1154,
    "columnas_submission": ("id", "HS", "TR", "AG"),
    "competencia": "KAGGLE_TOXIC_2026_PUJ",
    "fecha": "2026-09-22",
    "momento_contexto": "2026-09-22",
    "script": "experiments/07_submission_v04_ens04.py",
    "lineas_maximas_contexto": 15,
    "envios_autorizados": 1,
}

# Parametros TF-IDF de la union word (1,2) + char_wb (3,5), copiados de 05.
TFIDF_UNION: Final[tuple[dict, ...]] = (
    {
        "min_df": 2,
        "max_df": 1.0,
        "sublinear_tf": True,
        "lowercase": True,
        "strip_accents": None,
        "analyzer": "word",
        "ngram_range": (1, 2),
    },
    {
        "min_df": 2,
        "max_df": 1.0,
        "sublinear_tf": True,
        "lowercase": True,
        "strip_accents": None,
        "analyzer": "char_wb",
        "ngram_range": (3, 5),
    },
)

NOMBRE_REPRESENTACION: Final[str] = (
    "union TF-IDF word (1,2) + char_wb (3,5), min_df=2, sublinear_tf"
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

ENCABEZADO_CONTEXTO: Final[str] = "## Fase 13c - sonda v04 (2026-09-22)"


def log(message: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        message: texto a imprimir.
    """
    print(f"[07] {message}", flush=True)


# ===== SECCION 1: modelos base M2/M3 y utilidades (copiadas de 05) =====


@dataclass(frozen=True)
class EspecBase:
    """Especificacion de un modelo base del ensemble.

    Attributes:
        identificador: identificador corto (M2 o M3).
        nombre: descripcion legible de representacion y clasificador.
        tipo: ``logistic`` o ``svc``.
        parametros: hiperparametros del clasificador de scikit-learn.
        conversion: ``proba`` (``predict_proba``) o ``sigmoide`` (sigmoide de
            ``decision_function``).
    """

    identificador: str
    nombre: str
    tipo: str
    parametros: dict
    conversion: str


BASES: Final[tuple[EspecBase, ...]] = (
    EspecBase(
        identificador="M2",
        nombre="union word (1,2) + char_wb (3,5) + LR C=2.0 balanced (MOD-03b)",
        tipo="logistic",
        parametros={
            "C": 2.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "liblinear",
            "random_state": CONFIG["random_state"],
        },
        conversion="proba",
    ),
    EspecBase(
        identificador="M3",
        nombre="union word (1,2) + char_wb (3,5) + LinearSVC C=1.0 balanced (MOD-01)",
        tipo="svc",
        parametros={
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 5000,
            "random_state": CONFIG["random_state"],
        },
        conversion="sigmoide",
    ),
)


def construir_vectorizador(partes: Sequence[dict]) -> FeatureUnion:
    """Crea la union de vectorizadores TF-IDF de una representacion.

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


def f1_binaria_rapida(verdad: np.ndarray, prediccion: np.ndarray) -> float:
    """F1 binaria de la clase positiva con conteos de confusion directos.

    Es matematicamente equivalente a ``f1_score(verdad, prediccion,
    zero_division=0)`` y se usa en la busqueda de umbrales, donde la version de
    scikit-learn resulta demasiado costosa por el numero de evaluaciones. La
    equivalencia se verifica con :data:`CONFIG["equivalencia_f1_rapida"]`.

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


def umbrales_a_diccionario(umbrales: Sequence[float]) -> dict[str, float]:
    """Convierte una lista de umbrales en un diccionario por etiqueta.

    Args:
        umbrales: umbrales en el orden de :data:`ETIQUETAS`.

    Returns:
        Diccionario ``etiqueta -> umbral``.
    """
    return {etiqueta: float(umbral) for etiqueta, umbral in zip(ETIQUETAS, umbrales)}


# ===== SECCION 2: datos, pliegues y probabilidades OOF de M2/M3 =====


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

    Replica literalmente la particion de las Fases 4-5, 13a y 13b para que todas
    las metricas sean comparables con el ranking publicado.

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


def calcular_oof_bases(
    train: pd.DataFrame, pliegues: Sequence[tuple[np.ndarray, np.ndarray]]
) -> dict[str, np.ndarray]:
    """Calcula las probabilidades OOF de M2 (LR) y M3 (LinearSVC).

    El vectorizador de la union se ajusta SIEMPRE con el pliegue de entrenamiento
    y se transforma el de validacion; los dos modelos comparten la matriz por
    pliegue (el vectorizador es determinista, de modo que compartirlo no altera
    ningun resultado y ahorra tiempo).

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
        vectorizador = construir_vectorizador(TFIDF_UNION)
        matriz_entrenamiento = vectorizador.fit_transform(train["text"].iloc[entrenamiento])
        matriz_validacion = vectorizador.transform(train["text"].iloc[validacion])
        for espec in BASES:
            for indice, etiqueta in enumerate(ETIQUETAS):
                modelo = construir_modelo(espec)
                modelo.fit(matriz_entrenamiento, train[etiqueta].iloc[entrenamiento])
                probabilidad[espec.identificador][validacion, indice] = probabilidades(
                    modelo, matriz_validacion, espec
                )
    return probabilidad


def promedio_bases(probabilidad: dict[str, np.ndarray]) -> np.ndarray:
    """Promedio simple de las probabilidades de las bases (definicion de ENS-04).

    Args:
        probabilidad: diccionario ``identificador -> matriz (n, 3)``.

    Returns:
        Matriz ``(n, 3)`` con el promedio aritmetico de las bases.
    """
    return np.mean(np.stack(list(probabilidad.values()), axis=0), axis=0)


def comparar_con_npz(train: pd.DataFrame, probabilidad: dict[str, np.ndarray]) -> dict:
    """Compara el OOF recalculado con ``features/oof_bases.npz`` de la Fase 13b.

    Args:
        train: DataFrame de entrenamiento (aporta los ids).
        probabilidad: probabilidades OOF recalculadas por modelo base.

    Returns:
        Diccionario con la diferencia maxima por base y el veredicto de
        sincronizacion; si el archivo no existe o los ids no coinciden se
        reporta ``disponible: False`` sin fallar.

    Raises:
        AssertionError: si el archivo existe, los ids coinciden y alguna base
            difiere mas alla de :data:`CONFIG["tolerancia_npz"]`.
    """
    ruta: Path = CONFIG["oof_bases_npz"]
    if not ruta.exists():
        log("no hay features/oof_bases.npz previo; se omite la comparacion")
        return {"disponible": False, "motivo": "no existe el npz de la Fase 13b"}
    with np.load(ruta, allow_pickle=False) as datos:
        ids_guardados = [str(valor) for valor in datos["ids"]]
        previas = {
            espec.identificador: datos[f"proba_{espec.identificador}"] for espec in BASES
        }
    ids_actuales = [str(valor) for valor in train["id"]]
    if ids_guardados != ids_actuales:
        log("los ids del npz previo no coinciden; se omite la comparacion")
        return {"disponible": False, "motivo": "ids distintos entre el npz y el recalculo"}
    diferencias = {
        identificador: float(np.max(np.abs(previas[identificador] - probabilidad[identificador])))
        for identificador in previas
    }
    for identificador, diferencia in diferencias.items():
        log(f"diferencia maxima vs oof_bases.npz en {identificador}: {diferencia:.3e}")
        assert diferencia <= CONFIG["tolerancia_npz"], (
            f"El OOF recalculado de {identificador} difiere del npz de la Fase 13b: "
            f"{diferencia:.3e} > {CONFIG['tolerancia_npz']:.0e}"
        )
    return {
        "disponible": True,
        "ruta": str(ruta.relative_to(PROJECT_ROOT)),
        "diferencia_maxima": diferencias,
        "tolerancia": CONFIG["tolerancia_npz"],
    }


def leer_referencia_ens04() -> dict:
    """Lee la referencia anidada de ENS-04 en ``reports/ensembles_resultados.json``.

    Returns:
        Diccionario con la F1 anidada a 0.5, la F1 anidada con umbrales medianos,
        los umbrales finales, la estrategia elegida y la combinacion seleccionada
        en cada pliegue.

    Raises:
        FileNotFoundError: si falta el registro de la Fase 13b.
        ValueError: si el registro no contiene el ensemble esperado.
    """
    ruta: Path = CONFIG["resultados_json"]
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta}; ejecuta experiments/05_ensembles.py primero.")
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    coincidencias = [
        ensemble for ensemble in datos["ensembles"] if ensemble["experiment_id"] == CONFIG["id_ensemble"]
    ]
    if not coincidencias:
        raise ValueError(f"{ruta.name} no contiene {CONFIG['id_ensemble']}")
    registro = coincidencias[0]
    return {
        "f1_anidada_0.5": float(registro["f1_anidada_0.5"]),
        "estrategia_umbrales": str(registro["umbrales"]["estrategia_elegida"]),
        "f1_anidada_umbrales": float(registro["umbrales"]["f1_mejor_anidada"]),
        "umbrales_finales": {
            etiqueta: float(valor)
            for etiqueta, valor in registro["umbrales"]["umbrales_finales"].items()
        },
        "combinacion_por_pliegue": [
            {
                "pliegue": fila["pliegue"],
                "combinacion": list(fila["parametros"]["combinacion"]),
                "pesos": [float(peso) for peso in fila["parametros"]["pesos"]],
            }
            for fila in registro["detalle_pliegues"]
        ],
    }


# ===== SECCION 3: umbrales medianos por pliegue y verificacion de ENS-04 =====


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


def evaluar_mediana_por_pliegue(
    proba: np.ndarray,
    y_true: np.ndarray,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
    rejilla: np.ndarray,
) -> dict:
    """Evalua la mediana de los umbrales optimos por pliegue (logica de 05).

    Para cada pliegue de evaluacion se calcula el umbral optimo de cada uno de los
    otros cuatro pliegues por separado y se toma la mediana por etiqueta; los
    umbrales finales del reajuste son la mediana de los cinco optimos por pliegue.

    Args:
        proba: matriz ``(n, 3)`` de probabilidades del ensemble.
        y_true: matriz ``(n, 3)`` con las etiquetas verdaderas.
        pliegues: particion agrupada compartida.
        rejilla: umbrales candidatos.

    Returns:
        Diccionario con la prediccion anidada, los umbrales por pliegue, los
        umbrales finales (medianas por etiqueta) y los optimos de cada pliegue.
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


def verificar_ens04(
    y_true: np.ndarray,
    probabilidad: dict[str, np.ndarray],
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
    referencia: dict,
) -> dict:
    """Verifica que la reconstruccion reproduce el ENS-04 anidado de la Fase 13b.

    Comprueba, con tolerancias declaradas: (a) que ENS-04 selecciono el par
    (M2, M3) con pesos (0.5, 0.5) en los cinco pliegues, (b) que la F1 anidada a
    0.5 coincide con la publicada, (c) que la F1 anidada con umbrales medianos
    coincide con la publicada (tolerancia 1e-3) y redondea a 0.7985, y (d) que los
    umbrales finales son HS 0.44 / TR 0.47 / AG 0.44. Tambien comprueba que la
    implementacion rapida de la F1 coincide con la canonica del proyecto.

    Args:
        y_true: matriz ``(n, 3)`` con las etiquetas verdaderas.
        probabilidad: probabilidades OOF de M2 y M3.
        pliegues: particion agrupada compartida.
        referencia: valores publicados por :func:`leer_referencia_ens04`.

    Returns:
        Diccionario con la probabilidad del ensemble, la evaluacion de umbrales
        medianos, las metricas y las diferencias frente a la referencia.

    Raises:
        AssertionError: si alguna verificacion no se cumple.
    """
    for fila in referencia["combinacion_por_pliegue"]:
        assert fila["combinacion"] == ["M2", "M3"], (
            "ENS-04 no eligio el par M2+M3 en el pliegue "
            f"{fila['pliegue']}: {fila['combinacion']}"
        )
        assert fila["pesos"] == [0.0, 0.5, 0.5, 0.0], (
            f"pesos inesperados en el pliegue {fila['pliegue']}: {fila['pesos']}"
        )
    assert referencia["estrategia_umbrales"] == "mediana_por_pliegue", (
        f"la estrategia publicada de ENS-04 es {referencia['estrategia_umbrales']!r}, "
        "no 'mediana_por_pliegue'"
    )

    proba_ens = promedio_bases(probabilidad)
    rejilla = rejilla_umbrales()
    mediana = evaluar_mediana_por_pliegue(proba_ens, y_true, pliegues, rejilla)
    metricas_mediana = metricas_multietiqueta(y_true, mediana["prediccion"])
    f1_mediana = float(metricas_mediana["f1_media_columnas"])
    f1_rapida = f1_columnas_rapido(y_true, mediana["prediccion"])
    diferencia_rapida = abs(f1_rapida - f1_mediana)
    assert diferencia_rapida <= CONFIG["equivalencia_f1_rapida"], (
        "La F1 rapida no coincide con metricas_multietiqueta: "
        f"{f1_rapida!r} vs {f1_mediana!r}"
    )
    assert len(mediana["prediccion"]) == len(y_true), "la prediccion anidada no cubre el train"

    metricas_05 = metricas_multietiqueta(
        y_true, predecir_con_umbral(proba_ens, [CONFIG["umbral_base"]] * len(ETIQUETAS))
    )
    diferencia_05 = float(metricas_05["f1_media_columnas"]) - referencia["f1_anidada_0.5"]
    assert abs(diferencia_05) <= CONFIG["tolerancia_anidada"], (
        "La F1 anidada a 0.5 no reproduce la Fase 13b: "
        f"{metricas_05['f1_media_columnas']!r} vs {referencia['f1_anidada_0.5']!r}"
    )

    diferencia = f1_mediana - referencia["f1_anidada_umbrales"]
    assert abs(diferencia) <= CONFIG["tolerancia_anidada"], (
        "La F1 anidada con umbrales medianos no reproduce la Fase 13b: "
        f"{f1_mediana!r} vs {referencia['f1_anidada_umbrales']!r} "
        f"(tolerancia {CONFIG['tolerancia_anidada']:.0e})"
    )
    assert round(f1_mediana, 4) == CONFIG["f1_anidada_referencia"], (
        "La F1 anidada no redondea a 0.7985: " f"{round(f1_mediana, 4)!r}"
    )
    for etiqueta in ETIQUETAS:
        diferencia_umbral = abs(
            mediana["umbrales_finales"][etiqueta] - referencia["umbrales_finales"][etiqueta]
        )
        assert diferencia_umbral <= CONFIG["tolerancia_umbral"], (
            f"El umbral mediano de {etiqueta} no coincide: "
            f"{mediana['umbrales_finales'][etiqueta]!r} vs "
            f"{referencia['umbrales_finales'][etiqueta]!r}"
        )

    log(
        f"ENS-04 verificado: F1 anidada con umbrales medianos {f1_mediana:.6f} "
        f"(referencia {referencia['f1_anidada_umbrales']:.6f}, diferencia {diferencia:+.3e}); "
        f"umbrales finales {mediana['umbrales_finales']}"
    )
    return {
        "proba_ens": proba_ens,
        "mediana": mediana,
        "metricas_mediana": metricas_mediana,
        "metricas_umbral_0.5": metricas_05,
        "f1_anidada_umbrales": f1_mediana,
        "diferencia_frente_a_referencia": diferencia,
        "diferencia_f1_rapida_canonica": diferencia_rapida,
        "referencia": referencia,
    }


def guardar_oof_ens04(train: pd.DataFrame, verificacion: dict) -> Path:
    """Guarda el OOF recalculado de M2, M3 y del ensemble como evidencia.

    Args:
        train: DataFrame de entrenamiento (aporta ``id`` y etiquetas).
        verificacion: resultado de :func:`verificar_ens04`.

    Returns:
        Ruta del archivo ``.npz`` escrito.

    Raises:
        ValueError: si el archivo releido no tiene las formas esperadas.
    """
    destino: Path = CONFIG["oof_ens04_npz"]
    destino.parent.mkdir(parents=True, exist_ok=True)
    contenido: dict[str, Any] = {
        "ids": np.asarray([str(valor) for valor in train["id"]], dtype=str),
        "y_true": train[list(ETIQUETAS)].to_numpy(),
        "etiquetas": np.asarray(ETIQUETAS),
        "proba_ens": verificacion["proba_ens"],
        "pred_mediana": verificacion["mediana"]["prediccion"],
        "umbrales_finales": np.asarray(
            [verificacion["mediana"]["umbrales_finales"][etiqueta] for etiqueta in ETIQUETAS],
            dtype=float,
        ),
        "f1_anidada_umbrales": np.asarray([verificacion["f1_anidada_umbrales"]], dtype=float),
    }
    np.savez(destino, **contenido)
    with np.load(destino, allow_pickle=False) as releido:
        formas = {clave: releido[clave].shape for clave in releido.files}
    esperadas = {
        "ids": (len(train),),
        "y_true": (len(train), len(ETIQUETAS)),
        "etiquetas": (len(ETIQUETAS),),
        "proba_ens": (len(train), len(ETIQUETAS)),
        "pred_mediana": (len(train), len(ETIQUETAS)),
        "umbrales_finales": (len(ETIQUETAS),),
        "f1_anidada_umbrales": (1,),
    }
    if formas != esperadas:
        raise ValueError(f"El .npz guardado tiene formas inesperadas: {formas}")
    return destino


# ===== SECCION 4: refit con todo el train, submission y modelo final =====


def ajustar_bases_final(
    train: pd.DataFrame,
) -> tuple[FeatureUnion, dict[str, list], int]:
    """Ajusta el vectorizador de la union y los 3+3 clasificadores con TODO el train.

    Args:
        train: DataFrame de entrenamiento con ``text`` y etiquetas.

    Returns:
        Tupla ``(vectorizador, modelos, dimension)``; ``modelos`` es un diccionario
        ``identificador -> lista de clasificadores`` en el orden de
        :data:`ETIQUETAS` y ``dimension`` el numero de columnas TF-IDF.
    """
    vectorizador = construir_vectorizador(TFIDF_UNION)
    matriz = vectorizador.fit_transform(train["text"])
    modelos: dict[str, list] = {}
    for espec in BASES:
        modelos[espec.identificador] = []
        for etiqueta in ETIQUETAS:
            modelo = construir_modelo(espec)
            modelo.fit(matriz, train[etiqueta])
            modelos[espec.identificador].append(modelo)
    return vectorizador, modelos, int(matriz.shape[1])


def predecir_test_bases(
    test: pd.DataFrame, vectorizador: FeatureUnion, modelos: dict[str, list]
) -> dict[str, np.ndarray]:
    """Calcula las probabilidades de test de M2 y M3.

    Args:
        test: DataFrame de test con ``text``.
        vectorizador: vectorizador de la union ajustado con todo el train.
        modelos: clasificadores por modelo base.

    Returns:
        Diccionario ``identificador -> matriz (n_test, 3)`` de probabilidades.
    """
    matriz = vectorizador.transform(test["text"])
    return {
        espec.identificador: np.column_stack(
            [
                probabilidades(modelos[espec.identificador][indice], matriz, espec)
                for indice in range(len(ETIQUETAS))
            ]
        )
        for espec in BASES
    }


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
    plantilla: pd.DataFrame, prediccion: np.ndarray, nombre: str
) -> Path:
    """Construye, valida y escribe la submission de la sonda.

    La validacion ocurre en memoria: si algo falla se lanza la excepcion y NO se
    escribe ningun archivo invalido.

    Args:
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
    modelos: dict[str, list],
    umbrales: dict[str, float],
    verificacion: dict,
    dimension: int,
) -> Path:
    """Guarda el modelo final del ensemble con vectorizador, modelos y umbrales.

    Args:
        vectorizador: vectorizador de la union ajustado con todo el train.
        modelos: clasificadores de M2 y M3 ajustados con todo el train.
        umbrales: umbrales medianos por etiqueta.
        verificacion: resultado de :func:`verificar_ens04`.
        dimension: numero de features del vectorizador final.

    Returns:
        Ruta del archivo ``.joblib`` escrito.
    """
    destino: Path = CONFIG["modelo_joblib"]
    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizador": vectorizador,
            "modelos": modelos,
            "umbrales": umbrales,
            "umbral_base": CONFIG["umbral_base"],
            "etiquetas": list(ETIQUETAS),
            "dimension": dimension,
            "ensemble": CONFIG["id_ensemble"],
            "bases": [espec.nombre for espec in BASES],
            "representacion": NOMBRE_REPRESENTACION,
            "f1_anidada_umbrales": verificacion["f1_anidada_umbrales"],
            "f1_anidada_0.5": float(verificacion["metricas_umbral_0.5"]["f1_media_columnas"]),
            "entrenado_con": "todo el train (1731 filas)",
            "fecha": CONFIG["fecha"],
            "script": CONFIG["script"],
        },
        destino,
    )
    return destino


# ===== SECCION 5: registro en experiments.csv, leaderboard y contexto =====


def registrar_en_experiments(filas: Sequence[dict]) -> int:
    """Anade (o refresca) la fila de la fase en ``experiments.csv``.

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
    """Anade (o refresca) la fila de la sonda en ``leaderboard_log.csv``.

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


def construir_fila_s04(
    verificacion: dict,
    umbrales: dict[str, float],
    ruta: Path,
    resumen: dict,
    segundos: float,
    dimension: int,
    sha256: str,
) -> dict:
    """Construye la fila ``S04`` del registro de experimentos.

    Args:
        verificacion: resultado de :func:`verificar_ens04`.
        umbrales: umbrales medianos por etiqueta aplicados en la submission.
        ruta: ruta de la submission generada.
        resumen: resumen de positivos en test y prevalencias de train.
        segundos: segundos del refit y la prediccion.
        dimension: numero de features del vectorizador de la union.
        sha256: huella sha256 del archivo de submission.

    Returns:
        Diccionario con las columnas de ``COLUMNAS_EXPERIMENTS``.
    """
    metricas = verificacion["metricas_mediana"]
    f1_anidada = float(verificacion["f1_anidada_umbrales"])
    f1_05 = float(verificacion["metricas_umbral_0.5"]["f1_media_columnas"])
    return {
        "experiment_id": CONFIG["id_submission"],
        "competition": CONFIG["competencia"],
        "question": (
            "¿Generaliza al leaderboard el ensemble ENS-04 (par M2+M3) reajustado con "
            "todo el train y umbrales medianos por pliegue?"
        ),
        "hypothesis": (
            "El promedio de M2 (LR) y M3 (LinearSVC) sobre la union word+char aporta "
            "diversidad que reduce el sesgo de cada clasificador, y los umbrales medianos "
            "por pliegue son mas robustos que los optimos sobre el OOF completo; la sonda "
            "mide ademas si el gap local-LB de la familia union+char depende del tipo de "
            "modelo o del umbral"
        ),
        "preprocessing": "texto crudo, lowercase, sin stopwords",
        "feature_groups": "WORD_NGRAM+CHAR_NGRAM",
        "representation": NOMBRE_REPRESENTACION,
        "model": (
            "ENS-04: promedio simple de M2 (LogisticRegression balanced C=2.0 x3) y M3 "
            "(LinearSVC balanced C=1.0 x3 con sigmoide de decision_function), reajustados "
            "con todo el train; umbrales medianos por etiqueta"
        ),
        "hyperparameters": json.dumps(
            {
                "ensemble": CONFIG["id_ensemble"],
                "estrategia": "promedio simple de M2 y M3 (par elegido en los 5 pliegues)",
                "estrategia_umbrales": "mediana_por_pliegue",
                "umbrales_por_etiqueta": umbrales,
                "umbral_base": CONFIG["umbral_base"],
                "rejilla_umbral": list(CONFIG["rejilla_umbral"]),
                "modelos": {
                    espec.identificador: espec.parametros for espec in BASES
                },
                "postproceso": "ninguno",
            },
            ensure_ascii=False,
        ),
        "validation": (
            "F1 anidada leave-one-fold-out 0.7985 con umbrales medianos por pliegue "
            "(umbral de cada pliegue calculado sin el pliegue evaluado) sobre el split "
            "StratifiedGroupKFold(5, shuffle=True, random_state=42) agrupado por "
            "near-duplicado y estratificado por HSxTRxAG; reproduccion verificada contra "
            "reports/ensembles_resultados.json (tolerancia 1e-3); submission validada "
            "contra la plantilla oficial (columnas, 1154 filas, orden de ids, valores en "
            "{0,1}, sin nulos ni duplicados)"
        ),
        "f1": round(f1_anidada, 4),
        "macro_f1": round(float(metricas["f1_macro6"]), 4),
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": round(segundos, 2),
        "dimension": dimension,
        "result": json.dumps(
            {
                "submission": str(ruta.relative_to(PROJECT_ROOT)),
                "sha256": sha256,
                "f1_anidada_umbrales": round(f1_anidada, 6),
                "f1_anidada_umbral_0.5": round(f1_05, 6),
                "referencia_fase_13b": round(
                    float(verificacion["referencia"]["f1_anidada_umbrales"]), 6
                ),
                "diferencia_vs_referencia": round(
                    float(verificacion["diferencia_frente_a_referencia"]), 9
                ),
                "f1_por_etiqueta_anidada": {
                    etiqueta: round(valor, 6)
                    for etiqueta, valor in metricas["f1_por_etiqueta"].items()
                },
                "umbrales_por_etiqueta": umbrales,
                "umbrales_mediana_por_pliegue": verificacion["mediana"][
                    "umbrales_por_pliegue"
                ],
                "umbrales_optimos_por_pliegue": verificacion["mediana"][
                    "umbrales_optimos_por_pliegue"
                ],
                "positivos_test": resumen["positivos_test"],
                "tasa_positivos_test": resumen["tasa_positivos_test"],
                "tasa_positivos_train": resumen["tasa_positivos_train"],
            },
            ensure_ascii=False,
        ),
        "interpretation": (
            f"F1 anidada a 0.5 = {f1_05:.4f} -> con la mediana de umbrales por pliegue = "
            f"{f1_anidada:.4f} (referencia 0.7985 de la Fase 13b; single MOD-03b a 0.5 = "
            "0.7927). La estimacion anidada sigue siendo algo optimista porque las "
            "probabilidades base son OOF compartidas; el test solo se uso para predecir."
        ),
        "decision": (
            "submission generada y registrada para la sonda v04 autorizada por Carlos "
            f"({CONFIG['envios_autorizados']} envio); pendiente de que Carlos la envie a "
            "Kaggle"
        ),
        "date": CONFIG["fecha"],
        "script": CONFIG["script"],
        "notes": (
            "Sonda de experimentacion decidida por Carlos con 1 envio de Kaggle. En v02 y "
            "v03 la familia union+char rindio -0.0128 y -0.0158 por debajo de su local, "
            "por lo que el LB esperado de esta sonda esta en torno a 0.783 +/- 0.003, con "
            "v03 (0.78306) como referencia a batir. El script no envia nada a Kaggle."
        ),
    }


def construir_fila_leaderboard(
    ruta: Path, verificacion: dict, umbrales: dict[str, float], resumen: dict
) -> dict:
    """Construye la fila de seguimiento de la sonda en ``leaderboard_log.csv``.

    Args:
        ruta: ruta de la submission generada.
        verificacion: resultado de :func:`verificar_ens04`.
        umbrales: umbrales medianos por etiqueta.
        resumen: resumen de positivos en test y prevalencias de train.

    Returns:
        Diccionario con las columnas de ``COLUMNAS_LEADERBOARD``.
    """
    f1_anidada = float(verificacion["f1_anidada_umbrales"])
    umbrales_texto = ", ".join(
        f"{etiqueta} {umbrales[etiqueta]:.2f}" for etiqueta in ETIQUETAS
    )
    positivos_texto = ", ".join(
        f"{etiqueta} {resumen['positivos_test'][etiqueta]}" for etiqueta in ETIQUETAS
    )
    return {
        "submission_id": CONFIG["id_leaderboard"],
        "fecha": CONFIG["fecha"],
        "archivo": str(ruta.relative_to(PROJECT_ROOT)),
        "local_f1": round(f1_anidada, 5),
        "kaggle_f1": "",
        "diferencia": "",
        "features": NOMBRE_REPRESENTACION,
        "modelo": (
            "ENS-04: promedio simple de M2 (LR balanced C=2.0 x3) y M3 (LinearSVC balanced "
            "C=1.0 x3) reajustados con todo el train, umbrales medianos por pliegue"
        ),
        "hiperparametros": json.dumps(
            {
                "ensemble": CONFIG["id_ensemble"],
                "M2": next(espec.parametros for espec in BASES if espec.identificador == "M2"),
                "M3": next(espec.parametros for espec in BASES if espec.identificador == "M3"),
                "umbrales": umbrales,
                "estrategia_umbrales": "mediana_por_pliegue",
            },
            ensure_ascii=False,
        ),
        "interpretacion": (
            "SONDA DE EXPERIMENTACION decidida por Carlos (1 envio) para medir si los "
            "ensembles generalizan al LB. Estimacion anidada con umbrales medianos 0.7985 "
            f"frente a 0.7927 del single a 0.5; umbrales {umbrales_texto}. Positivos "
            f"predichos: {positivos_texto}. En v02/v03 la familia union+char quedo -0.0128 "
            "y -0.0158 bajo su local, de modo que un LB en 0.783-0.786 seria coherente y "
            "solo un valor por encima de 0.78306 batiria a v03. kaggle_f1 pendiente de que "
            "Carlos envie la submission y lo registre."
        ),
    }


def construir_seccion_contexto(
    verificacion: dict,
    umbrales: dict[str, float],
    ruta: Path,
    resumen: dict,
    sha256: str,
    comparacion_npz: dict,
    ruta_oof: Path,
    ruta_modelo: Path,
) -> str:
    """Redacta la seccion de handoff (maximo 15 lineas) de ``reports/contexto.md``.

    Args:
        verificacion: resultado de :func:`verificar_ens04`.
        umbrales: umbrales medianos por etiqueta.
        ruta: ruta de la submission generada.
        resumen: resumen de positivos en test y prevalencias de train.
        sha256: huella sha256 de la submission.
        comparacion_npz: resultado de :func:`comparar_con_npz`.
        ruta_oof: ruta del ``.npz`` de evidencia del OOF.
        ruta_modelo: ruta del modelo final.

    Returns:
        Texto de la seccion, con encabezado y sin linea final en blanco.

    Raises:
        RuntimeError: si la seccion generada excede las lineas permitidas.
    """
    f1_anidada = float(verificacion["f1_anidada_umbrales"])
    f1_05 = float(verificacion["metricas_umbral_0.5"]["f1_media_columnas"])
    diferencia = float(verificacion["diferencia_frente_a_referencia"])
    umbrales_texto = " / ".join(
        f"{etiqueta} {umbrales[etiqueta]:.2f}" for etiqueta in ETIQUETAS
    )
    positivos_texto = " / ".join(
        f"{etiqueta} {resumen['positivos_test'][etiqueta]} "
        f"({100 * resumen['tasa_positivos_test'][etiqueta]:.2f} %)"
        for etiqueta in ETIQUETAS
    )
    train_texto = " / ".join(
        f"{etiqueta} {100 * resumen['tasa_positivos_train'][etiqueta]:.2f} %"
        for etiqueta in ETIQUETAS
    )
    lineas = [
        ENCABEZADO_CONTEXTO,
        "",
        (
            "Sonda de submission v04 decidida por Carlos (1 envio de Kaggle) con el "
            "ensemble `ENS-04` (par M2+M3: LR balanced C=2.0 y LinearSVC balanced C=1.0 "
            "sobre la union TF-IDF word (1,2)+char_wb (3,5)) y umbrales medianos por "
            "pliegue, para medir si los ensembles generalizan al leaderboard; el script "
            "`experiments/07_submission_v04_ens04.py` no envia nada a Kaggle."
        ),
        (
            "ASSERT reproducido: F1 anidada con umbrales medianos = "
            f"{f1_anidada:.6f} frente a "
            f"{float(verificacion['referencia']['f1_anidada_umbrales']):.6f} de "
            f"`reports/ensembles_resultados.json` (diferencia {diferencia:+.1e}, tolerancia "
            f"{CONFIG['tolerancia_anidada']:.0e}); redondea a 0.7985 y la F1 anidada a 0.5 "
            f"({f1_05:.6f}) coincide con la publicada."
        ),
        (
            "Umbrales medianos por pliegue usados: "
            f"{umbrales_texto} (mediana de los 5 optimos por pliegue de la rejilla "
            "0.20-0.80 paso 0.01, identicos al registro de la Fase 13b)."
        ),
        (
            "ENS-04 eligio el par (M2, M3) con pesos (0.5, 0.5) en los 5 pliegues, por lo "
            "que su probabilidad anidada es el promedio simple de ambas bases; el OOF "
            "recalculado coincide con `features/oof_bases.npz` ("
            + (
                ", ".join(
                    f"{clave} {valor:.1e}"
                    for clave, valor in comparacion_npz["diferencia_maxima"].items()
                )
                if comparacion_npz["disponible"]
                else "comparacion no disponible"
            )
            + f") y queda como evidencia en `{ruta_oof.relative_to(PROJECT_ROOT)}`."
        ),
        (
            f"Submission `{ruta.relative_to(PROJECT_ROOT)}` validada contra la plantilla "
            f"oficial (1154 filas, ids identicos y en orden, valores en {{0,1}}, sin nulos "
            f"ni duplicados); sha256 {sha256[:16]}...; modelo final en "
            f"`{ruta_modelo.relative_to(PROJECT_ROOT)}`."
        ),
        (
            f"Positivos predichos en test: {positivos_texto}; prevalencias de train: "
            f"{train_texto}."
        ),
        (
            f"Registro: `experiments.csv` ({CONFIG['id_submission']}, f1 0.7985 = estimacion "
            f"anidada) y `submissions/leaderboard_log.csv` ({CONFIG['id_leaderboard']} con "
            "`kaggle_f1` vacio, pendiente de que Carlos envie la submission)."
        ),
        (
            "ADVERTENCIA de honestidad: la F1 anidada (0.7985) es una cota algo optimista "
            "porque las probabilidades base son OOF compartidas; en v02 y v03 la familia "
            "union+char rindio -0.0128 y -0.0158 bajo su local, asi que el LB esperado ronda "
            "0.783 +/- 0.003 y la referencia a batir es v03 (0.78306)."
        ),
        (
            "Sin fuga: el test solo se uso para predecir; ni la seleccion del par, ni los "
            "umbrales ni ninguna decision usaron el test. Idempotente: dos ejecuciones "
            "seguidas dejan el mismo contenido salvo los tiempos."
        ),
    ]
    while lineas and lineas[-1] == "":
        lineas.pop()
    if len(lineas) > CONFIG["lineas_maximas_contexto"]:
        raise RuntimeError(
            f"La seccion de contexto excede {CONFIG['lineas_maximas_contexto']} lineas "
            f"({len(lineas)})"
        )
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
    texto = destino.read_text(encoding="utf-8") if destino.exists() else ""
    if ENCABEZADO_CONTEXTO in texto:
        texto = texto[: texto.index(ENCABEZADO_CONTEXTO)]
    destino.write_text(
        texto.rstrip("\n") + "\n\n" + seccion.rstrip("\n") + "\n", encoding="utf-8"
    )
    return destino


# ===== SECCION 6: flujo principal =====


def main() -> None:
    """Ejecuta la Fase 13c: verificacion de ENS-04, refit, submission y registro."""
    inicio = time.perf_counter()
    train, test, grupos = cargar_datos()
    plantilla = data_module.read_submission()
    if list(plantilla["id"]) != list(test["id"]):
        raise ValueError(
            "El orden de ids del test no coincide con la plantilla oficial; "
            "abortado antes de generar la submission"
        )
    log(
        f"train {train.shape} | test {test.shape} | plantilla {plantilla.shape} | "
        f"grupos {grupos.nunique()}"
    )

    pliegues = construir_pliegues(train, grupos)
    y_true = train[list(ETIQUETAS)].to_numpy()

    # 1) OOF de M2/M3 y verificacion del valor anidado de ENS-04.
    inicio_oof = time.perf_counter()
    probabilidad_oof = calcular_oof_bases(train, pliegues)
    segundos_oof = time.perf_counter() - inicio_oof
    for espec in BASES:
        f1_base = float(
            metricas_multietiqueta(
                y_true,
                predecir_con_umbral(
                    probabilidad_oof[espec.identificador], [CONFIG["umbral_base"]] * len(ETIQUETAS)
                ),
            )["f1_media_columnas"]
        )
        log(f"{espec.identificador} ({espec.nombre}): F1 OOF a 0.5 = {f1_base:.6f}")
    comparacion_npz = comparar_con_npz(train, probabilidad_oof)
    referencia = leer_referencia_ens04()
    verificacion = verificar_ens04(y_true, probabilidad_oof, pliegues, referencia)
    ruta_oof = guardar_oof_ens04(train, verificacion)
    log(f"OOF y verificacion en {segundos_oof:.2f} s; evidencia en {ruta_oof.name}")

    umbrales = {
        etiqueta: float(verificacion["mediana"]["umbrales_finales"][etiqueta])
        for etiqueta in ETIQUETAS
    }
    log(f"umbrales medianos por pliegue: {umbrales}")

    # 2) Refit con todo el train, prediccion del test y submission.
    inicio_refit = time.perf_counter()
    vectorizador, modelos, dimension = ajustar_bases_final(train)
    probabilidad_test = predecir_test_bases(test, vectorizador, modelos)
    proba_test_ens = promedio_bases(probabilidad_test)
    prediccion = predecir_con_umbral(
        proba_test_ens, [umbrales[etiqueta] for etiqueta in ETIQUETAS]
    )
    segundos_refit = time.perf_counter() - inicio_refit

    ruta_submission = escribir_submission(plantilla, prediccion, CONFIG["submission"])
    sha256 = data_module.sha256_of(ruta_submission)
    resumen = resumir_prediccion(train, prediccion)
    log(f"sha256 de la submission: {sha256}")
    log(f"positivos en test: {json.dumps(resumen, ensure_ascii=False)}")
    log(f"dimension de la union: {dimension}")

    ruta_modelo = guardar_modelo(vectorizador, modelos, umbrales, verificacion, dimension)
    log(f"modelo final guardado: {ruta_modelo.name}")

    # 3) Registro en experiments.csv (S04) y en el leaderboard log.
    filas_experiments = [
        construir_fila_s04(
            verificacion, umbrales, ruta_submission, resumen, segundos_refit, dimension, sha256
        )
    ]
    escritas = registrar_en_experiments(filas_experiments)
    log(f"filas escritas en experiments.csv: {escritas} ({CONFIG['id_submission']})")
    filas_leaderboard = [construir_fila_leaderboard(ruta_submission, verificacion, umbrales, resumen)]
    escritas_lb = registrar_en_leaderboard(filas_leaderboard)
    log(f"filas escritas en leaderboard_log.csv: {escritas_lb} ({CONFIG['id_leaderboard']})")

    # 4) Handoff en reports/contexto.md y detalle en JSON.
    seccion = construir_seccion_contexto(
        verificacion,
        umbrales,
        ruta_submission,
        resumen,
        sha256,
        comparacion_npz,
        ruta_oof,
        ruta_modelo,
    )
    ruta_contexto = actualizar_contexto(seccion)
    log(f"contexto actualizado: {ruta_contexto.relative_to(PROJECT_ROOT)}")

    detalle = {
        "metadata": {
            "fecha": CONFIG["fecha"],
            "script": CONFIG["script"],
            "competencia": CONFIG["competencia"],
            "fase": "Fase 13c (sonda v04 con ENS-04 y umbrales medianos)",
            "ensemble": CONFIG["id_ensemble"],
            "configuracion": (
                "promedio simple de M2 (LR balanced C=2.0 x3) y M3 (LinearSVC balanced "
                "C=1.0 x3) sobre la union TF-IDF word (1,2)+char_wb (3,5)"
            ),
            "validacion": (
                f"StratifiedGroupKFold({CONFIG['n_splits']}, shuffle=True, "
                f"random_state={CONFIG['random_state']}) agrupado por near-duplicado y "
                "estratificado por HSxTRxAG; evaluacion anidada leave-one-fold-out"
            ),
            "sin_fuga_test": "el test solo se uso para predecir",
            "envios_autorizados": CONFIG["envios_autorizados"],
            "segundos_oof": round(segundos_oof, 2),
            "segundos_refit": round(segundos_refit, 2),
        },
        "referencia_fase_13b": referencia,
        "verificacion": {
            "f1_anidada_umbrales": float(verificacion["f1_anidada_umbrales"]),
            "f1_anidada_umbral_0.5": float(
                verificacion["metricas_umbral_0.5"]["f1_media_columnas"]
            ),
            "macro6_anidada_umbrales": float(verificacion["metricas_mediana"]["f1_macro6"]),
            "f1_por_etiqueta_anidada": verificacion["metricas_mediana"]["f1_por_etiqueta"],
            "diferencia_vs_referencia": float(
                verificacion["diferencia_frente_a_referencia"]
            ),
            "diferencia_f1_rapida_canonica": float(
                verificacion["diferencia_f1_rapida_canonica"]
            ),
            "tolerancia": CONFIG["tolerancia_anidada"],
            "umbrales_finales": umbrales,
            "umbrales_por_pliegue": verificacion["mediana"]["umbrales_por_pliegue"],
            "umbrales_optimos_por_pliegue": verificacion["mediana"][
                "umbrales_optimos_por_pliegue"
            ],
            "comparacion_oof_bases_npz": comparacion_npz,
        },
        "refit": {
            "dimension_union": dimension,
            "segundos": round(segundos_refit, 2),
            "umbrales": umbrales,
        },
        "submission": {
            "ruta": str(ruta_submission.relative_to(PROJECT_ROOT)),
            "sha256": sha256,
            "prediccion_test": resumen,
        },
        "artefactos": {
            "oof_npz": str(ruta_oof.relative_to(PROJECT_ROOT)),
            "modelo_joblib": str(ruta_modelo.relative_to(PROJECT_ROOT)),
            "contexto_md": str(ruta_contexto.relative_to(PROJECT_ROOT)),
        },
    }
    CONFIG["detalle_json"].parent.mkdir(parents=True, exist_ok=True)
    CONFIG["detalle_json"].write_text(
        json.dumps(detalle, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    log(f"detalle: {CONFIG['detalle_json'].relative_to(PROJECT_ROOT)}")
    log(f"duracion total: {time.perf_counter() - inicio:.2f} s")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - el punto de entrada CLI reporta todo
        import traceback

        traceback.print_exc()
        sys.exit(1)
