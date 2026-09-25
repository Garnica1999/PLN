"""10_sondas_robustez.py - Fase 13d: dos sondas de robustez pre-registradas (D-013).

Pregunta que responde
---------------------
El gap local-LB crece con la capacidad del modelo (D-013): B1 (4.024 features,
C=1) -0.001; MOD-03b (27.431, C=2) -0.0128/-0.0158; ENS-04 -0.0234; OPT-01
(32.060, C=3.26) -0.0279. ¿Reducir la capacidad de la representacion de MOD-03b
mejora la transferencia al test sin perder nivel absoluto?

Metodo
------
- Control ``REF-MOD03b``: se recalcula MOD-03b_BEST (union word(1,2) min_df=2 +
  char_wb(3,5) min_df=2, LR C=2.0 balanced x3) para (a) verificar el arnes
  contra la F1 publicada (0.7926588861781015) y contra las probabilidades de
  ``features/oof_mod03b.npz``, y (b) medir su gap train-OOF con el MISMO
  procedimiento que las sondas.
- Sonda ``R1-C1`` (``v06``): misma representacion que MOD-03b con C=1.0.
- Sonda ``R2-MINDF5`` (``v07``): min_df=5 en los dos bloques y C=1.0.
- Validacion: ``StratifiedGroupKFold(5, shuffle=True, random_state=42)`` con
  grupos = componentes de near-duplicado (D-004) y estratificacion por la
  combinacion HS x TR x AG. Particion unica compartida con las fases previas.
- Por variante: F1 OOF con 0.5, umbrales optimos por etiqueta sobre las
  probabilidades OOF (rejilla 0.20-0.80 paso 0.01), F1 OOF con umbrales, F1 por
  etiqueta, std entre pliegues, dimension y gap train-OOF (F1 in-sample de los
  pliegues de entrenamiento menos F1 OOF con 0.5; diagnostico de sobreajuste).
- Submissions: refit con todo el train y los umbrales OOF de cada variante; se
  validan contra la plantilla oficial antes de escribirse. El test solo se usa
  para predecir.

Artefactos
----------
- ``submissions/toxic_submission_v06_robustez_c1.csv``
- ``submissions/toxic_submission_v07_robustez_mindf5.csv``
- ``models/fase13d_sonda_v06_c1.joblib`` y ``models/fase13d_sonda_v07_mindf5.joblib``
- ``reports/fase13d_sondas_robustez.json`` (detalle completo)
- filas ``R1-C1``, ``R2-MINDF5``, ``S06`` y ``S07`` en ``experiments.csv``
- filas ``v06_robustez_c1`` y ``v07_robustez_mindf5`` en
  ``submissions/leaderboard_log.csv`` (``kaggle_f1`` vacio) y seccion de handoff
  en ``reports/contexto.md``.

Idempotencia
------------
Los pliegues, las probabilidades, las metricas, los umbrales y las submissions
son deterministas. Los registros se reescriben por ``experiment_id`` /
``submission_id`` (nunca se duplican) y la seccion de ``reports/contexto.md`` se
reemplaza si ya existe. El unico campo no determinista es el tiempo de
ejecucion (ruido de maquina).

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/10_sondas_robustez.py
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.evaluation import ETIQUETAS, MetricasMultiEtiqueta, metricas_multietiqueta

CONFIG: Final[dict] = {
    "grupos_path": PROJECT_ROOT / "data" / "processed" / "grupos_near_duplicados.csv",
    "oof_control_npz": PROJECT_ROOT / "features" / "oof_mod03b.npz",
    "metricas_json": PROJECT_ROOT / "reports" / "fase13d_sondas_robustez.json",
    "experiments_csv": PROJECT_ROOT / "experiments.csv",
    "leaderboard_csv": PROJECT_ROOT / "submissions" / "leaderboard_log.csv",
    "contexto_md": PROJECT_ROOT / "reports" / "contexto.md",
    "submissions_dir": PROJECT_ROOT / "submissions",
    "modelos_dir": PROJECT_ROOT / "models",
    "random_state": 42,
    "n_splits": 5,
    "umbral_base": 0.5,
    "rejilla_umbral": (0.20, 0.80, 0.01),
    # Factor de la estimacion del test a partir de la prevalencia de train:
    # la sonda E00 (D-009) mostro que el test tiene ~7 % menos positivos.
    "factor_test_estimado": 0.93,
    # Referencia exacta publicada en reports/variantes_representaciones.json
    # para MOD-03b_BEST; el control debe reproducirla dentro de la tolerancia.
    "f1_oof_control": 0.7926588861781015,
    "tolerancia_control": 1e-6,
    "tolerancia_probabilidad_control": 1e-9,
    "umbral_lb_a_batir": 0.78306,
    "competencia": "KAGGLE_TOXIC_2026_PUJ",
    "fecha": "2026-09-23",
    "momento_contexto": "2026-09-23",
    "script": "experiments/10_sondas_robustez.py",
    "n_filas_test": 1154,
    "columnas_submission": ("id", "HS", "TR", "AG"),
    "encabezado_contexto": "## Fase 13d - sondas de robustez (2026-09-23)",
    "max_lineas_contexto": 15,
}

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

TEXTO_CRUDO: Final[str] = "texto crudo, lowercase, sin stopwords (igual que la Fase 4-5)"
GRUPO_FEATURES: Final[str] = "WORD_NGRAM+CHAR_NGRAM"
VALIDACION: Final[str] = (
    "StratifiedGroupKFold(5, shuffle=True, random_state=42) con grupos de "
    "near-duplicado (D-004) y estratificacion por HSxTRxAG; vectorizador ajustado "
    "dentro de cada pliegue; umbrales y comparaciones solo con probabilidades OOF"
)


def log(message: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        message: texto a imprimir.
    """
    print(f"[10] {message}", flush=True)


# ===== SECCION 1: variantes de la sonda =====


def _tfidf(min_df: int, analyzer: str, ngram_range: tuple[int, int]) -> dict:
    """Construye los parametros TF-IDF de un bloque de la union word+char.

    Se replican los parametros de la Fase 4-5 (``max_df=1.0``, ``sublinear_tf``,
    ``lowercase=True`` y ``strip_accents=None``) y solo varia ``min_df``, que es
    la palanca de capacidad de la sonda R2.

    Args:
        min_df: frecuencia minima de documento para conservar un termino.
        analyzer: ``word`` o ``char_wb``.
        ngram_range: rango de n-gramas del bloque.

    Returns:
        Diccionario de parametros para ``TfidfVectorizer``.
    """
    return {
        "min_df": min_df,
        "max_df": 1.0,
        "sublinear_tf": True,
        "lowercase": True,
        "strip_accents": None,
        "analyzer": analyzer,
        "ngram_range": ngram_range,
    }


@dataclass(frozen=True)
class VarianteSonda:
    """Especificacion de una variante evaluada en la fase.

    Attributes:
        identificador: id del experimento (``REF-MOD03b``, ``R1-C1``, ``R2-MINDF5``).
        clave: etiqueta corta de la sonda (``control``, ``v06``, ``v07``).
        descripcion: descripcion legible de la configuracion.
        pregunta: pregunta experimental registrada en ``experiments.csv``.
        hipotesis: hipotesis pre-registrada de la variante.
        partes_vectorizador: parametros de cada bloque TF-IDF de la union.
        c_modelo: valor de ``C`` de la regresion logistica.
        nombre_representacion: etiqueta de la representacion para los registros.
        archivo_submission: nombre del CSV de submission (vacio en el control).
        id_leaderboard: id de la fila de ``leaderboard_log.csv`` (vacio en el control).
    """

    identificador: str
    clave: str
    descripcion: str
    pregunta: str
    hipotesis: str
    partes_vectorizador: tuple[dict, ...]
    c_modelo: float
    nombre_representacion: str
    archivo_submission: str = ""
    id_leaderboard: str = ""

    @property
    def nombre_modelo(self) -> str:
        """Etiqueta del clasificador con su valor de ``C``."""
        return (
            f"LogisticRegression(class_weight=balanced, C={self.c_modelo}, "
            "liblinear) x3"
        )


CONTROL: Final[VarianteSonda] = VarianteSonda(
    identificador="REF-MOD03b",
    clave="control",
    descripcion=(
        "control MOD-03b_BEST (union word(1,2)+char_wb(3,5) min_df=2, LR C=2.0 "
        "balanced x3) para verificar el arnes y medir el gap train-OOF de referencia"
    ),
    pregunta="¿Reproduce el arnes de esta fase la F1 y las probabilidades OOF de MOD-03b_BEST?",
    hipotesis=(
        "El control reproduce 0.7926588861781015 y las probabilidades de "
        "features/oof_mod03b.npz, por lo que las sondas son comparables con las fases previas"
    ),
    partes_vectorizador=(
        _tfidf(2, "word", (1, 2)),
        _tfidf(2, "char_wb", (3, 5)),
    ),
    c_modelo=2.0,
    nombre_representacion=(
        "union TF-IDF word (1,2) + char_wb (3,5), min_df=2, sublinear_tf"
    ),
)

SONDAS: Final[tuple[VarianteSonda, ...]] = (
    VarianteSonda(
        identificador="R1-C1",
        clave="v06",
        descripcion=(
            "MOD-03b con C reducido de 2.0 a 1.0 (misma representacion min_df=2)"
        ),
        pregunta=(
            "¿Reducir C de 2.0 a 1.0 en la representacion de MOD-03b reduce el gap "
            "train-OOF y conserva la F1 out-of-fold?"
        ),
        hipotesis=(
            "Con C=1.0 el modelo memoriza menos el train (gap train-OOF menor) y la F1 "
            "OOF se mantiene dentro de 0.005 de MOD-03b; la transferencia al test mejora"
        ),
        partes_vectorizador=(
            _tfidf(2, "word", (1, 2)),
            _tfidf(2, "char_wb", (3, 5)),
        ),
        c_modelo=1.0,
        nombre_representacion=(
            "union TF-IDF word (1,2) + char_wb (3,5), min_df=2, sublinear_tf"
        ),
        archivo_submission="toxic_submission_v06_robustez_c1.csv",
        id_leaderboard="v06_robustez_c1",
    ),
    VarianteSonda(
        identificador="R2-MINDF5",
        clave="v07",
        descripcion=(
            "MOD-03b con min_df=5 en ambos bloques y C=1.0 (vocabulario reducido)"
        ),
        pregunta=(
            "¿Elevar min_df de 2 a 5 en los dos bloques (con C=1.0) elimina los n-gramas "
            "de baja frecuencia y reduce el gap train-OOF sin perder F1 out-of-fold?"
        ),
        hipotesis=(
            "min_df=5 descarta n-gramas que solo aparecen en train y reduce el gap "
            "train-OOF mas que C=1.0, con una perdida de F1 OOF menor que el ruido "
            "entre pliegues"
        ),
        partes_vectorizador=(
            _tfidf(5, "word", (1, 2)),
            _tfidf(5, "char_wb", (3, 5)),
        ),
        c_modelo=1.0,
        nombre_representacion=(
            "union TF-IDF word (1,2) + char_wb (3,5), min_df=5, sublinear_tf"
        ),
        archivo_submission="toxic_submission_v07_robustez_mindf5.csv",
        id_leaderboard="v07_robustez_mindf5",
    ),
)


# ===== SECCION 2: carga de datos y particion de validacion =====


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

    Replica exactamente la particion compartida por las fases previas
    (``shuffle=True``, ``random_state=42``, estratificacion por la combinacion
    HS x TR x AG) para que las metricas sean comparables.

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


# ===== SECCION 3: evaluacion out-of-fold y diagnostico de sobreajuste =====


@dataclass(frozen=True)
class EvaluacionOOF:
    """Resultado de la evaluacion out-of-fold de una variante.

    Attributes:
        probabilidad: matriz ``(n, 3)`` con la probabilidad OOF de la clase positiva.
        prediccion_train: matriz ``(n, 3)`` con la prediccion in-sample de cada fila
            (umbral 0.5) hecha por el modelo del pliegue que la uso para entrenar.
        f1_train_pliegues: F1 primaria in-sample de cada pliegue de entrenamiento.
        f1_train_media: media de ``f1_train_pliegues`` (diagnostico de sobreajuste).
        dimension: numero de features del vectorizador ajustado con todo el train.
        segundos: tiempo de la evaluacion out-of-fold.
    """

    probabilidad: np.ndarray
    prediccion_train: np.ndarray
    f1_train_pliegues: tuple[float, ...]
    f1_train_media: float
    dimension: int
    segundos: float


def construir_vectorizador(variante: VarianteSonda) -> FeatureUnion:
    """Crea la union de vectorizadores TF-IDF de una variante.

    Args:
        variante: especificacion de la variante.

    Returns:
        ``FeatureUnion`` con los bloques declarados en la variante.
    """
    return FeatureUnion(
        [
            (f"bloque_{indice}", TfidfVectorizer(**parametros))
            for indice, parametros in enumerate(variante.partes_vectorizador)
        ]
    )


def construir_modelo(variante: VarianteSonda) -> LogisticRegression:
    """Instancia un clasificador binario nuevo con los hiperparametros de la variante.

    Args:
        variante: especificacion de la variante.

    Returns:
        ``LogisticRegression`` sin ajustar.
    """
    return LogisticRegression(
        C=variante.c_modelo,
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
        random_state=CONFIG["random_state"],
    )


def evaluar_oof(
    variante: VarianteSonda,
    textos: pd.Series,
    y_true: np.ndarray,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
) -> EvaluacionOOF:
    """Calcula probabilidades OOF, prediccion in-sample y dimension de una variante.

    El vectorizador se ajusta SIEMPRE con el pliegue de entrenamiento y se
    transforma el de validacion; la prediccion in-sample se calcula sobre la
    matriz de entrenamiento del mismo pliegue y solo alimenta el diagnostico de
    sobreajuste (nunca la metrica de seleccion).

    Args:
        variante: especificacion de la variante.
        textos: serie de textos de train.
        y_true: matriz ``(n, 3)`` con las etiquetas verdaderas.
        pliegues: particion agrupada compartida.

    Returns:
        Evaluacion out-of-fold de la variante.
    """
    inicio = time.perf_counter()
    probabilidad = np.zeros(y_true.shape, dtype=float)
    prediccion_train = np.zeros(y_true.shape, dtype=int)
    f1_train_pliegues: list[float] = []
    for entrenamiento, validacion in pliegues:
        vectorizador = construir_vectorizador(variante)
        matriz_entrenamiento = vectorizador.fit_transform(textos.iloc[entrenamiento])
        matriz_validacion = vectorizador.transform(textos.iloc[validacion])
        for indice, _ in enumerate(ETIQUETAS):
            modelo = construir_modelo(variante)
            modelo.fit(matriz_entrenamiento, y_true[entrenamiento, indice])
            probabilidad[validacion, indice] = modelo.predict_proba(matriz_validacion)[:, 1]
            prediccion_train[entrenamiento, indice] = (
                modelo.predict_proba(matriz_entrenamiento)[:, 1]
                >= CONFIG["umbral_base"]
            ).astype(int)
        f1_train_pliegues.append(
            float(
                metricas_multietiqueta(
                    y_true[entrenamiento], prediccion_train[entrenamiento]
                )["f1_media_columnas"]
            )
        )
    dimension = int(construir_vectorizador(variante).fit_transform(textos).shape[1])
    evaluacion = EvaluacionOOF(
        probabilidad=probabilidad,
        prediccion_train=prediccion_train,
        f1_train_pliegues=tuple(f1_train_pliegues),
        f1_train_media=float(np.mean(f1_train_pliegues)),
        dimension=dimension,
        segundos=round(time.perf_counter() - inicio, 2),
    )
    log(
        f"{variante.identificador}: OOF en {evaluacion.segundos:.1f} s | dimension "
        f"{dimension} | F1 in-sample media {evaluacion.f1_train_media:.4f}"
    )
    return evaluacion


def verificar_control(
    evaluacion: EvaluacionOOF, train: pd.DataFrame
) -> dict:
    """Verifica que el control reproduce la F1 y las probabilidades OOF publicadas.

    Args:
        evaluacion: evaluacion OOF del control MOD-03b.
        train: DataFrame de entrenamiento (aporta ``id``).

    Returns:
        Diccionario con la comparacion de F1 y de probabilidades.

    Raises:
        AssertionError: si la F1 del control no coincide con la referencia o si
            las probabilidades difieren de ``features/oof_mod03b.npz``.
    """
    prediccion = (
        evaluacion.probabilidad >= CONFIG["umbral_base"]
    ).astype(int)
    f1_reproducida = float(
        metricas_multietiqueta(train[list(ETIQUETAS)].to_numpy(), prediccion)[
            "f1_media_columnas"
        ]
    )
    referencia = float(CONFIG["f1_oof_control"])
    diferencia = f1_reproducida - referencia
    log(
        f"control: F1 OOF {f1_reproducida:.9f} frente a {referencia:.9f} "
        f"(diferencia {diferencia:+.3e})"
    )
    assert abs(diferencia) <= CONFIG["tolerancia_control"], (
        "El control no reproduce la F1 de MOD-03b_BEST: "
        f"{f1_reproducida!r} vs {referencia!r}"
    )
    detalle = {
        "f1_oof_reproducida": f1_reproducida,
        "f1_oof_referencia": referencia,
        "diferencia_f1": diferencia,
        "tolerancia_f1": CONFIG["tolerancia_control"],
    }
    ruta_npz: Path = CONFIG["oof_control_npz"]
    if not ruta_npz.exists():
        log(f"advertencia: no existe {ruta_npz.name}; se omite la comparacion de probabilidades")
        detalle["comparacion_probabilidades"] = "no disponible"
        return detalle
    with np.load(ruta_npz, allow_pickle=False) as guardado:
        probabilidad_guardada = guardado["proba"]
        ids_guardados = guardado["ids"]
    ids_actuales = np.asarray([str(valor) for valor in train["id"]], dtype=str)
    if probabilidad_guardada.shape != evaluacion.probabilidad.shape:
        raise AssertionError(
            "Formas de probabilidad distintas entre el control y features/oof_mod03b.npz: "
            f"{evaluacion.probabilidad.shape} vs {probabilidad_guardada.shape}"
        )
    if not np.array_equal(ids_guardados, ids_actuales):
        raise AssertionError("Los ids de features/oof_mod03b.npz no coinciden con el train")
    diferencia_maxima = float(
        np.abs(probabilidad_guardada - evaluacion.probabilidad).max()
    )
    log(f"control: diferencia maxima de probabilidad vs oof_mod03b.npz {diferencia_maxima:.3e}")
    assert diferencia_maxima <= CONFIG["tolerancia_probabilidad_control"], (
        "Las probabilidades del control difieren de features/oof_mod03b.npz: "
        f"{diferencia_maxima!r}"
    )
    detalle["comparacion_probabilidades"] = {
        "archivo": str(ruta_npz.relative_to(PROJECT_ROOT)),
        "diferencia_maxima": diferencia_maxima,
        "tolerancia": CONFIG["tolerancia_probabilidad_control"],
        "coincide": True,
    }
    return detalle


# ===== SECCION 4: umbrales por etiqueta sobre OOF =====


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
    """Busca el umbral que maximiza la F1 binaria de una etiqueta sobre OOF.

    Si dos umbrales empatan se conserva el menor (primer maximo de la rejilla),
    lo que deja la decision determinista.

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


def ajustar_umbrales(y_true: np.ndarray, probabilidad: np.ndarray) -> dict[str, float]:
    """Ajusta el umbral optimo por etiqueta usando solo probabilidades OOF.

    Args:
        y_true: matriz ``(n, 3)`` con las etiquetas verdaderas.
        probabilidad: matriz ``(n, 3)`` con las probabilidades out-of-fold.

    Returns:
        Diccionario ``etiqueta -> umbral`` en el orden de :data:`ETIQUETAS`.
    """
    rejilla = rejilla_umbrales()
    return {
        etiqueta: optimizar_umbral(
            y_true[:, indice], probabilidad[:, indice], rejilla
        )[0]
        for indice, etiqueta in enumerate(ETIQUETAS)
    }


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


def umbrales_como_lista(umbrales: dict[str, float]) -> list[float]:
    """Convierte el diccionario de umbrales en la lista ordenada por etiqueta.

    Args:
        umbrales: diccionario ``etiqueta -> umbral``.

    Returns:
        Lista de umbrales en el orden de :data:`ETIQUETAS`.
    """
    return [umbrales[etiqueta] for etiqueta in ETIQUETAS]


# ===== SECCION 5: informe por variante =====


@dataclass
class InformeSonda:
    """Informe completo de una variante: metricas, umbrales y diagnostico.

    Attributes:
        variante: especificacion evaluada.
        evaluacion: resultado out-of-fold (probabilidades, in-sample, dimension).
        metricas_0_5: metricas multi-etiqueta con umbral 0.5.
        f1_pliegues: F1 primaria por pliegue con umbral 0.5.
        std_pliegues: desviacion tipica (poblacional) de ``f1_pliegues``.
        umbrales: umbral optimo por etiqueta ajustado sobre OOF.
        metricas_umbrales: metricas multi-etiqueta con los umbrales optimos.
        f1_pliegues_umbrales: F1 primaria por pliegue con los umbrales optimos.
        ganancia_umbrales: ``f1_media_columnas`` con umbrales menos con 0.5.
        gap_train_oof: ``f1_train_media`` menos ``f1_media_columnas`` con 0.5.
        segundos_umbrales: tiempo del ajuste y la evaluacion de umbrales.
    """

    variante: VarianteSonda
    evaluacion: EvaluacionOOF
    metricas_0_5: MetricasMultiEtiqueta
    f1_pliegues: tuple[float, ...]
    std_pliegues: float
    umbrales: dict[str, float]
    metricas_umbrales: MetricasMultiEtiqueta
    f1_pliegues_umbrales: tuple[float, ...]
    ganancia_umbrales: float
    gap_train_oof: float
    segundos_umbrales: float


def construir_informe(
    variante: VarianteSonda,
    evaluacion: EvaluacionOOF,
    y_true: np.ndarray,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
) -> InformeSonda:
    """Resume metricas, umbrales y gap train-OOF de una variante evaluada.

    Args:
        variante: especificacion de la variante.
        evaluacion: evaluacion out-of-fold de la variante.
        y_true: matriz ``(n, 3)`` con las etiquetas verdaderas.
        pliegues: particion agrupada compartida.

    Returns:
        Informe completo de la variante.
    """
    inicio = time.perf_counter()
    umbrales = ajustar_umbrales(y_true, evaluacion.probabilidad)
    prediccion_0_5 = predecir_con_umbrales(
        evaluacion.probabilidad, [CONFIG["umbral_base"]] * len(ETIQUETAS)
    )
    prediccion_umbrales = predecir_con_umbrales(
        evaluacion.probabilidad, umbrales_como_lista(umbrales)
    )
    metricas_0_5 = metricas_multietiqueta(y_true, prediccion_0_5)
    metricas_umbrales = metricas_multietiqueta(y_true, prediccion_umbrales)
    f1_pliegues = tuple(
        float(
            metricas_multietiqueta(y_true[validacion], prediccion_0_5[validacion])[
                "f1_media_columnas"
            ]
        )
        for _, validacion in pliegues
    )
    f1_pliegues_umbrales = tuple(
        float(
            metricas_multietiqueta(
                y_true[validacion], prediccion_umbrales[validacion]
            )["f1_media_columnas"]
        )
        for _, validacion in pliegues
    )
    informe = InformeSonda(
        variante=variante,
        evaluacion=evaluacion,
        metricas_0_5=metricas_0_5,
        f1_pliegues=f1_pliegues,
        std_pliegues=float(np.std(f1_pliegues)),
        umbrales=umbrales,
        metricas_umbrales=metricas_umbrales,
        f1_pliegues_umbrales=f1_pliegues_umbrales,
        ganancia_umbrales=float(
            metricas_umbrales["f1_media_columnas"] - metricas_0_5["f1_media_columnas"]
        ),
        gap_train_oof=float(
            evaluacion.f1_train_media - metricas_0_5["f1_media_columnas"]
        ),
        segundos_umbrales=round(time.perf_counter() - inicio, 2),
    )
    log(
        f"{variante.identificador}: F1 OOF 0.5 {metricas_0_5['f1_media_columnas']:.4f} "
        f"-> umbrales {metricas_umbrales['f1_media_columnas']:.4f} "
        f"({informe.ganancia_umbrales:+.4f}) | umbrales "
        f"{' '.join(f'{e}={umbrales[e]:.2f}' for e in ETIQUETAS)} | "
        f"gap train-OOF {informe.gap_train_oof:+.4f}"
    )
    return informe


def clasificar_gap(
    gap_control: float, gap_sonda: float
) -> str:
    """Clasifica el gap train-OOF de una sonda frente al control.

    Args:
        gap_control: gap train-OOF del control MOD-03b.
        gap_sonda: gap train-OOF de la sonda.

    Returns:
        Texto con la comparacion y el delta.
    """
    delta = gap_sonda - gap_control
    if delta < -0.005:
        return f"baja {abs(delta):.4f} frente al control"
    if delta > 0.005:
        return f"SUBE {delta:.4f} frente al control"
    return f"no baja de forma apreciable ({delta:+.4f} frente al control)"


def informe_a_dict(informe: InformeSonda) -> dict:
    """Convierte un informe en un diccionario serializable a JSON.

    Args:
        informe: informe de la variante.

    Returns:
        Diccionario con la configuracion, las metricas y el diagnostico.
    """
    return {
        "experiment_id": informe.variante.identificador,
        "clave": informe.variante.clave,
        "descripcion": informe.variante.descripcion,
        "representacion": informe.variante.nombre_representacion,
        "modelo": informe.variante.nombre_modelo,
        "parametros_vectorizador": list(informe.variante.partes_vectorizador),
        "parametros_modelo": {
            "C": informe.variante.c_modelo,
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "liblinear",
            "random_state": CONFIG["random_state"],
        },
        "dimension": informe.evaluacion.dimension,
        "segundos_oof": informe.evaluacion.segundos,
        "segundos_umbrales": informe.segundos_umbrales,
        "f1_media_columnas_umbral_0.5": round(
            float(informe.metricas_0_5["f1_media_columnas"]), 6
        ),
        "f1_media_columnas_umbrales_optimos": round(
            float(informe.metricas_umbrales["f1_media_columnas"]), 6
        ),
        "ganancia_umbrales": round(informe.ganancia_umbrales, 6),
        "macro6_umbral_0.5": round(float(informe.metricas_0_5["f1_macro6"]), 6),
        "macro6_umbrales_optimos": round(
            float(informe.metricas_umbrales["f1_macro6"]), 6
        ),
        "f1_por_etiqueta_umbral_0.5": {
            etiqueta: round(float(valor), 6)
            for etiqueta, valor in informe.metricas_0_5["f1_por_etiqueta"].items()
        },
        "f1_por_etiqueta_umbrales_optimos": {
            etiqueta: round(float(valor), 6)
            for etiqueta, valor in informe.metricas_umbrales[
                "f1_por_etiqueta"
            ].items()
        },
        "f1_por_pliegue_umbral_0.5": [round(valor, 6) for valor in informe.f1_pliegues],
        "f1_por_pliegue_umbrales_optimos": [
            round(valor, 6) for valor in informe.f1_pliegues_umbrales
        ],
        "std_pliegues": round(informe.std_pliegues, 6),
        "umbrales_optimos": informe.umbrales,
        "f1_train_media_columnas": round(informe.evaluacion.f1_train_media, 6),
        "f1_train_por_pliegue": [
            round(valor, 6) for valor in informe.evaluacion.f1_train_pliegues
        ],
        "gap_train_oof": round(informe.gap_train_oof, 6),
    }


# ===== SECCION 6: refit, submissions y validacion =====


def ajustar_modelo_final(
    variante: VarianteSonda, train: pd.DataFrame
) -> tuple[FeatureUnion, list[LogisticRegression], int]:
    """Ajusta el vectorizador y los tres modelos de una variante con todo el train.

    Args:
        variante: especificacion de la variante.
        train: DataFrame de entrenamiento con ``text`` y etiquetas.

    Returns:
        Tupla ``(vectorizador, modelos, dimension)`` ajustados.
    """
    vectorizador = construir_vectorizador(variante)
    matriz = vectorizador.fit_transform(train["text"])
    modelos = []
    for etiqueta in ETIQUETAS:
        modelo = construir_modelo(variante)
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
    return np.column_stack(
        [modelo.predict_proba(matriz)[:, 1] for modelo in modelos]
    )


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
    plantilla: pd.DataFrame,
    prediccion: np.ndarray,
    nombre: str,
) -> Path:
    """Construye, valida y escribe una submission.

    La validacion ocurre en memoria: si algo falla se lanza la excepcion y NO se
    escribe ningun archivo invalido. Despues de escribir se relee y se vuelve a
    validar contra la plantilla.

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

    Incluye la estimacion del test como ``factor_test_estimado`` (0.93, sonda E00)
    por la prevalencia de train, y la diferencia entre lo predicho y esa
    estimacion, que es la comparacion de calibracion de la sonda.

    Args:
        train: DataFrame de entrenamiento con etiquetas.
        prediccion: matriz ``(n, 3)`` binaria del test.

    Returns:
        Diccionario con conteos, tasas predichas, prevalencia de train, test
        estimado y diferencia por etiqueta.
    """
    factor = float(CONFIG["factor_test_estimado"])
    prevalencia = {
        etiqueta: round(float(train[etiqueta].mean()), 4) for etiqueta in ETIQUETAS
    }
    predichas = {
        etiqueta: round(float(prediccion[:, indice].mean()), 4)
        for indice, etiqueta in enumerate(ETIQUETAS)
    }
    estimadas = {
        etiqueta: round(factor * prevalencia[etiqueta], 4) for etiqueta in ETIQUETAS
    }
    return {
        "positivos_test": {
            etiqueta: int(prediccion[:, indice].sum())
            for indice, etiqueta in enumerate(ETIQUETAS)
        },
        "tasa_positivos_test": predichas,
        "tasa_positivos_train": prevalencia,
        "tasa_test_estimada": estimadas,
        "diferencia_vs_estimado": {
            etiqueta: round(predichas[etiqueta] - estimadas[etiqueta], 4)
            for etiqueta in ETIQUETAS
        },
        "factor_test_estimado": factor,
    }


def guardar_modelo(
    informe: InformeSonda,
    vectorizador: FeatureUnion,
    modelos: Sequence[LogisticRegression],
    umbrales: dict[str, float],
    dimension: int,
) -> Path:
    """Guarda el modelo final de una sonda con vectorizador, modelos y umbrales.

    Args:
        informe: informe de la variante (aporta la F1 OOF y la clave).
        vectorizador: vectorizador ajustado con todo el train.
        modelos: modelos binarios ajustados con todo el train.
        umbrales: umbral por etiqueta usado en la submission.
        dimension: numero de features del vectorizador final.

    Returns:
        Ruta del archivo ``.joblib`` escrito.
    """
    destino: Path = CONFIG["modelos_dir"] / f"fase13d_sonda_{informe.variante.clave}.joblib"
    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizador": vectorizador,
            "modelos": list(modelos),
            "umbrales": umbrales,
            "umbral_base": CONFIG["umbral_base"],
            "etiquetas": list(ETIQUETAS),
            "dimension": dimension,
            "representacion": informe.variante.nombre_representacion,
            "modelo": informe.variante.nombre_modelo,
            "f1_oof_umbral_0.5": float(informe.metricas_0_5["f1_media_columnas"]),
            "f1_oof_umbrales_optimos": float(
                informe.metricas_umbrales["f1_media_columnas"]
            ),
            "entrenado_con": "todo el train (1731 filas)",
            "fecha": CONFIG["fecha"],
            "script": CONFIG["script"],
        },
        destino,
    )
    log(f"modelo guardado: {destino.name} (dimension {dimension})")
    return destino


# ===== SECCION 7: registro en experiments.csv, leaderboard y contexto =====


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
    """Anade (o refresca) las filas de submissions en ``leaderboard_log.csv``.

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


def construir_fila_configuracion(informe: InformeSonda) -> dict:
    """Construye la fila ``R1-C1``/``R2-MINDF5`` del registro de experimentos.

    Args:
        informe: informe de la sonda.

    Returns:
        Diccionario con las columnas de ``COLUMNAS_EXPERIMENTS``.
    """
    variante = informe.variante
    return {
        "experiment_id": variante.identificador,
        "competition": CONFIG["competencia"],
        "question": variante.pregunta,
        "hypothesis": variante.hipotesis,
        "preprocessing": TEXTO_CRUDO,
        "feature_groups": GRUPO_FEATURES,
        "representation": variante.nombre_representacion,
        "model": variante.nombre_modelo,
        "hyperparameters": json.dumps(
            {
                "vectorizador": list(variante.partes_vectorizador),
                "modelo": {
                    "C": variante.c_modelo,
                    "class_weight": "balanced",
                    "max_iter": 1000,
                    "solver": "liblinear",
                    "random_state": CONFIG["random_state"],
                },
                "umbrales_optimos_oof": informe.umbrales,
                "rejilla_umbral": {
                    "inicio": CONFIG["rejilla_umbral"][0],
                    "fin": CONFIG["rejilla_umbral"][1],
                    "paso": CONFIG["rejilla_umbral"][2],
                },
            },
            ensure_ascii=False,
        ),
        "validation": VALIDACION,
        "f1": round(float(informe.metricas_0_5["f1_media_columnas"]), 4),
        "macro_f1": round(float(informe.metricas_0_5["f1_macro6"]), 4),
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": informe.evaluacion.segundos,
        "dimension": informe.evaluacion.dimension,
        "result": json.dumps(
            {
                "f1_media_columnas_umbral_0.5": round(
                    float(informe.metricas_0_5["f1_media_columnas"]), 6
                ),
                "f1_media_columnas_umbrales_optimos": round(
                    float(informe.metricas_umbrales["f1_media_columnas"]), 6
                ),
                "ganancia_umbrales": round(informe.ganancia_umbrales, 6),
                "f1_por_etiqueta_umbral_0.5": informe.metricas_0_5[
                    "f1_por_etiqueta"
                ],
                "f1_por_etiqueta_umbrales_optimos": informe.metricas_umbrales[
                    "f1_por_etiqueta"
                ],
                "f1_por_pliegue_umbral_0.5": list(informe.f1_pliegues),
                "std_pliegues": round(informe.std_pliegues, 6),
                "umbrales_optimos": informe.umbrales,
                "f1_train_media_columnas": round(informe.evaluacion.f1_train_media, 6),
                "gap_train_oof": round(informe.gap_train_oof, 6),
            },
            ensure_ascii=False,
        ),
        "interpretation": (
            f"F1 OOF con 0.5 = {informe.metricas_0_5['f1_media_columnas']:.4f}; con "
            f"umbrales OOF = {informe.metricas_umbrales['f1_media_columnas']:.4f} "
            f"({informe.ganancia_umbrales:+.4f}); std entre pliegues "
            f"{informe.std_pliegues:.4f}; dimension {informe.evaluacion.dimension}; "
            f"gap train-OOF {informe.gap_train_oof:+.4f} "
            f"(train {informe.evaluacion.f1_train_media:.4f})"
        ),
        "decision": (
            "sonda pre-registrada D-013 (maximo 2 envios): se adopta como modelo final "
            f"solo si el LB supera {CONFIG['umbral_lb_a_batir']:.5f} (v03); en caso "
            "contrario se mantiene MOD-03b con umbrales"
        ),
        "date": CONFIG["fecha"],
        "script": CONFIG["script"],
        "notes": (
            "Control de capacidad de D-013; el gap train-OOF es la F1 in-sample de los "
            "pliegues de entrenamiento menos la F1 OOF con 0.5 y se reporta como "
            "diagnostico de sobreajuste. Los umbrales se ajustan solo sobre OOF."
        ),
    }


def construir_fila_submission(
    informe: InformeSonda,
    resumen: dict,
    ruta: Path,
    segundos_refit: float,
) -> dict:
    """Construye la fila ``S06``/``S07`` del registro de experimentos.

    Args:
        informe: informe de la sonda.
        resumen: resumen de positivos predichos en el test.
        ruta: ruta de la submission generada.
        segundos_refit: tiempo del refit y la prediccion del test.

    Returns:
        Diccionario con las columnas de ``COLUMNAS_EXPERIMENTS``.
    """
    variante = informe.variante
    identificador = "S06" if variante.clave == "v06" else "S07"
    return {
        "experiment_id": identificador,
        "competition": CONFIG["competencia"],
        "question": (
            f"¿Que submission genera la sonda {variante.identificador} reajustada con "
            "todo el train y con los umbrales OOF por etiqueta?"
        ),
        "hypothesis": (
            f"La sonda {variante.identificador} transfiere mejor al test que v03 "
            f"({CONFIG['umbral_lb_a_batir']:.5f}) porque su menor capacidad reduce el "
            "gap train-OOF"
        ),
        "preprocessing": TEXTO_CRUDO,
        "feature_groups": GRUPO_FEATURES,
        "representation": variante.nombre_representacion,
        "model": (
            f"LogisticRegression(class_weight=balanced, C={variante.c_modelo}, "
            "liblinear) x3 con umbral optimo por etiqueta"
        ),
        "hyperparameters": json.dumps(
            {
                "modelo": {
                    "C": variante.c_modelo,
                    "class_weight": "balanced",
                    "max_iter": 1000,
                    "solver": "liblinear",
                    "random_state": CONFIG["random_state"],
                },
                "umbral_base": CONFIG["umbral_base"],
                "umbrales_por_etiqueta": informe.umbrales,
                "postproceso": "ninguno",
            },
            ensure_ascii=False,
        ),
        "validation": (
            "F1 OOF de la sonda con el split agrupado de D-004; umbrales ajustados SOLO "
            "sobre probabilidades out-of-fold; submission validada contra la plantilla "
            "oficial (columnas, 1154 filas, orden de ids, valores en {0,1}, sin nulos "
            "ni duplicados)"
        ),
        "f1": round(float(informe.metricas_umbrales["f1_media_columnas"]), 4),
        "macro_f1": round(float(informe.metricas_umbrales["f1_macro6"]), 4),
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": round(segundos_refit, 2),
        "dimension": informe.evaluacion.dimension,
        "result": json.dumps(
            {
                "submission": str(ruta.relative_to(PROJECT_ROOT)),
                "f1_oof_umbral_0.5": round(
                    float(informe.metricas_0_5["f1_media_columnas"]), 6
                ),
                "f1_oof_umbrales_optimos": round(
                    float(informe.metricas_umbrales["f1_media_columnas"]), 6
                ),
                "umbrales_optimos": informe.umbrales,
                "gap_train_oof": round(informe.gap_train_oof, 6),
                "positivos_test": resumen["positivos_test"],
                "tasa_positivos_test": resumen["tasa_positivos_test"],
                "tasa_positivos_train": resumen["tasa_positivos_train"],
                "tasa_test_estimada": resumen["tasa_test_estimada"],
                "diferencia_vs_estimado": resumen["diferencia_vs_estimado"],
            },
            ensure_ascii=False,
        ),
        "interpretation": (
            f"Refit de {variante.identificador} con todo el train (1731 filas); positivos "
            f"predichos HS {resumen['tasa_positivos_test']['HS']:.4f} / TR "
            f"{resumen['tasa_positivos_test']['TR']:.4f} / AG "
            f"{resumen['tasa_positivos_test']['AG']:.4f} frente al test estimado (0.93x "
            f"train) HS {resumen['tasa_test_estimada']['HS']:.4f} / TR "
            f"{resumen['tasa_test_estimada']['TR']:.4f} / AG "
            f"{resumen['tasa_test_estimada']['AG']:.4f}"
        ),
        "decision": (
            "submission generada; pendiente de envio por Carlos (sonda pre-registrada "
            f"D-013); se adopta si supera {CONFIG['umbral_lb_a_batir']:.5f}"
        ),
        "date": CONFIG["fecha"],
        "script": CONFIG["script"],
        "notes": (
            "La F1 OOF con umbrales es una cota optimista (umbral y metrica comparten el "
            "mismo OOF); en esta familia el gap local-LB medido es -0.013 a -0.016 "
            "(D-012/D-013), por lo que el resultado se decide con el leaderboard"
        ),
    }


def construir_fila_leaderboard(
    informe: InformeSonda,
    resumen: dict,
    ruta: Path,
) -> dict:
    """Construye la fila de ``leaderboard_log.csv`` de una sonda.

    Args:
        informe: informe de la sonda.
        resumen: resumen de positivos predichos en el test.
        ruta: ruta de la submission generada.

    Returns:
        Diccionario con las columnas de ``COLUMNAS_LEADERBOARD``.
    """
    variante = informe.variante
    return {
        "submission_id": variante.id_leaderboard,
        "fecha": CONFIG["fecha"],
        "archivo": str(ruta.relative_to(PROJECT_ROOT)),
        "local_f1": round(float(informe.metricas_umbrales["f1_media_columnas"]), 5),
        "kaggle_f1": "",
        "diferencia": "",
        "features": variante.nombre_representacion,
        "modelo": (
            f"LogisticRegression balanced C={variante.c_modelo} x3 (HS, TR, AG) con "
            "umbral optimo por etiqueta, sin post-proceso"
        ),
        "hiperparametros": json.dumps(
            {
                "C": variante.c_modelo,
                "class_weight": "balanced",
                "solver": "liblinear",
                "umbrales": informe.umbrales,
            },
            ensure_ascii=False,
        ),
        "interpretacion": (
            f"Sonda de robustez pre-registrada D-013 (maximo 2 envios): F1 OOF "
            f"{informe.metricas_0_5['f1_media_columnas']:.4f} con 0.5 y "
            f"{informe.metricas_umbrales['f1_media_columnas']:.4f} con umbrales OOF "
            f"({informe.ganancia_umbrales:+.4f}); gap train-OOF {informe.gap_train_oof:+.4f}; "
            f"dimension {informe.evaluacion.dimension}. Positivos predichos HS "
            f"{resumen['tasa_positivos_test']['HS']:.4f} / TR "
            f"{resumen['tasa_positivos_test']['TR']:.4f} / AG "
            f"{resumen['tasa_positivos_test']['AG']:.4f} frente al test estimado (0.93x "
            f"train) HS {resumen['tasa_test_estimada']['HS']:.4f} / TR "
            f"{resumen['tasa_test_estimada']['TR']:.4f} / AG "
            f"{resumen['tasa_test_estimada']['AG']:.4f}. Se adopta como modelo final solo "
            f"si supera el LB de v03 ({CONFIG['umbral_lb_a_batir']:.5f}); kaggle_f1 "
            "pendiente de que Carlos envie las dos sondas."
        ),
    }


def formato_pct(valor: float) -> str:
    """Formatea una tasa como porcentaje con un decimal.

    Args:
        valor: tasa en el rango ``[0, 1]``.

    Returns:
        Texto del estilo ``40.6 %``.
    """
    return f"{100 * valor:.1f} %"


def construir_seccion_contexto(
    control: InformeSonda,
    verificacion: dict,
    informes: Sequence[InformeSonda],
    resumenes: dict[str, dict],
    rutas: dict[str, Path],
    gap_control: float,
) -> str:
    """Redacta la seccion de handoff (maximo 15 lineas) para ``reports/contexto.md``.

    Args:
        control: informe del control MOD-03b.
        verificacion: resultado de :func:`verificar_control`.
        informes: informes de las dos sondas, en orden.
        resumenes: resumen de positivos predichos por clave de sonda.
        rutas: rutas de las submissions por clave de sonda.
        gap_control: gap train-OOF del control.

    Returns:
        Texto de la seccion, con encabezado y sin linea final en blanco.

    Raises:
        RuntimeError: si la seccion generada excede el maximo de lineas permitido.
    """
    veredictos = "; ".join(
        f"{informe.variante.identificador} {clasificar_gap(gap_control, informe.gap_train_oof)}"
        for informe in informes
    )
    comparacion_probabilidades = verificacion["comparacion_probabilidades"]
    if isinstance(comparacion_probabilidades, dict):
        texto_probabilidades = (
            "probabilidades identicas a `features/oof_mod03b.npz` (diferencia maxima "
            f"{comparacion_probabilidades['diferencia_maxima']:.1e})"
        )
    else:
        texto_probabilidades = "comparacion de probabilidades no disponible"
    linea_control = (
        f"Control REF-MOD03b (C=2.0, min_df=2): F1 OOF con 0.5 = "
        f"{verificacion['f1_oof_reproducida']:.6f} frente a "
        f"{verificacion['f1_oof_referencia']:.6f} de la Fase 4-5 (diferencia "
        f"{verificacion['diferencia_f1']:+.3e}); {texto_probabilidades}; dimension "
        f"{control.evaluacion.dimension}; gap train-OOF {control.gap_train_oof:+.4f}."
    )
    lineas_sondas = []
    for informe in informes:
        umbrales = " / ".join(
            f"{etiqueta} {informe.umbrales[etiqueta]:.2f}" for etiqueta in ETIQUETAS
        )
        lineas_sondas.append(
            f"{informe.variante.identificador} (`{informe.variante.clave}`): "
            f"{informe.variante.descripcion}; dimension {informe.evaluacion.dimension}; "
            f"F1 OOF 0.5 = {informe.metricas_0_5['f1_media_columnas']:.4f}; umbrales OOF "
            f"{umbrales}; F1 OOF con umbrales = "
            f"{informe.metricas_umbrales['f1_media_columnas']:.4f} "
            f"({informe.ganancia_umbrales:+.4f}); std entre pliegues "
            f"{informe.std_pliegues:.4f}; gap train-OOF {informe.gap_train_oof:+.4f}."
        )
    linea_gap = (
        f"Gap train-OOF (F1 in-sample de los pliegues de entrenamiento menos F1 OOF con "
        f"0.5): control {gap_control:+.4f}, "
        + ", ".join(
            f"{informe.variante.identificador} {informe.gap_train_oof:+.4f}"
            for informe in informes
        )
        + f" -> {veredictos}."
    )
    lineas_positivos = []
    for informe in informes:
        resumen = resumenes[informe.variante.clave]
        etiquetas = " / ".join(
            f"{etiqueta} {resumen['positivos_test'][etiqueta]} "
            f"({formato_pct(resumen['tasa_positivos_test'][etiqueta])})"
            for etiqueta in ETIQUETAS
        )
        lineas_positivos.append(
            f"Positivos predichos en test con los umbrales OOF de "
            f"{informe.variante.clave}: {etiquetas}."
        )
    resumen_referencia = resumenes[informes[0].variante.clave]
    linea_prevalencia = (
        "Referencia de calibracion: prevalencia de train HS "
        f"{formato_pct(resumen_referencia['tasa_positivos_train']['HS'])} / TR "
        f"{formato_pct(resumen_referencia['tasa_positivos_train']['TR'])} / AG "
        f"{formato_pct(resumen_referencia['tasa_positivos_train']['AG'])}; test estimado "
        f"(0.93x train) HS {formato_pct(resumen_referencia['tasa_test_estimada']['HS'])} / "
        f"TR {formato_pct(resumen_referencia['tasa_test_estimada']['TR'])} / AG "
        f"{formato_pct(resumen_referencia['tasa_test_estimada']['AG'])}."
    )
    linea_submissions = (
        "Submissions validadas contra la plantilla oficial (1154 filas, ids identicos y "
        "en orden, valores en {0,1}, sin nulos ni duplicados): "
        + " y ".join(
            f"`{rutas[informe.variante.clave].relative_to(PROJECT_ROOT)}`"
            for informe in informes
        )
        + "."
    )
    linea_registro = (
        "Registro: `experiments.csv` (R1-C1, R2-MINDF5, S06, S07) y "
        "`submissions/leaderboard_log.csv` (v06_robustez_c1, v07_robustez_mindf5 con "
        "`kaggle_f1` vacio, pendientes de que Carlos envie las dos sondas)."
    )
    linea_decision = (
        f"Decision pre-registrada D-013: se adopta como modelo final solo la sonda que "
        f"supere el LB de v03 ({CONFIG['umbral_lb_a_batir']:.5f}); prohibido sondear mas "
        "alla de estos dos envios sin una hipotesis nueva."
    )
    linea_honestidad = (
        "Honestidad: la F1 OOF con umbrales es una cota optimista (umbral y metrica "
        "comparten el mismo OOF) y en esta familia el gap local-LB medido es -0.013 a "
        "-0.016 (D-012), asi que el LB esperado de las sondas ronda 0.78 +/- 0.01."
    )
    linea_sin_fuga = (
        "Sin fuga: el test solo se uso para predecir; umbrales, seleccion y comparaciones "
        "usan exclusivamente probabilidades out-of-fold de train. Idempotente: dos "
        "ejecuciones dejan el mismo contenido salvo los tiempos."
    )
    lineas = [
        CONFIG["encabezado_contexto"],
        "",
        (
            "Sondas de robustez pre-registradas de D-013 (maximo 2 envios; script "
            "`experiments/10_sondas_robustez.py`): reducir la capacidad de la "
            "representacion de MOD-03b para probar si mejora la transferencia al test."
        ),
        linea_control,
        *lineas_sondas,
        linea_gap,
        *lineas_positivos,
        linea_prevalencia,
        linea_submissions,
        linea_registro,
        linea_decision,
        linea_honestidad,
        linea_sin_fuga,
    ]
    while lineas and lineas[-1] == "":
        lineas.pop()
    if len(lineas) > CONFIG["max_lineas_contexto"]:
        raise RuntimeError(
            f"La seccion de contexto excede {CONFIG['max_lineas_contexto']} lineas "
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
    encabezado = CONFIG["encabezado_contexto"]
    texto = destino.read_text(encoding="utf-8") if destino.exists() else ""
    if encabezado in texto:
        texto = texto[: texto.index(encabezado)]
    destino.write_text(
        texto.rstrip("\n") + "\n\n" + seccion.rstrip("\n") + "\n", encoding="utf-8"
    )
    return destino


# ===== SECCION 8: flujo principal =====


def main() -> None:
    """Ejecuta la Fase 13d completa: control, sondas, submissions y registro."""
    inicio = time.perf_counter()
    train, test, grupos = cargar_datos()
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
    textos = train["text"]

    # 1) Control: verificacion del arnes y gap train-OOF de referencia.
    evaluacion_control = evaluar_oof(CONTROL, textos, y_true, pliegues)
    verificacion = verificar_control(evaluacion_control, train)
    informe_control = construir_informe(CONTROL, evaluacion_control, y_true, pliegues)

    # 2) Sondas: evaluacion OOF, umbrales, refit, submission y modelo.
    informes: list[InformeSonda] = []
    resumenes: dict[str, dict] = {}
    rutas: dict[str, Path] = {}
    segundos_refit: dict[str, float] = {}
    for sonda in SONDAS:
        evaluacion = evaluar_oof(sonda, textos, y_true, pliegues)
        informe = construir_informe(sonda, evaluacion, y_true, pliegues)
        informes.append(informe)
        inicio_refit = time.perf_counter()
        vectorizador, modelos, dimension = ajustar_modelo_final(sonda, train)
        probabilidad_test = predecir_test(test, vectorizador, modelos)
        prediccion_test = predecir_con_umbrales(
            probabilidad_test, umbrales_como_lista(informe.umbrales)
        )
        segundos_refit[sonda.clave] = round(time.perf_counter() - inicio_refit, 2)
        if dimension != informe.evaluacion.dimension:
            raise ValueError(
                f"{sonda.identificador}: la dimension del refit ({dimension}) no coincide "
                f"con la de la evaluacion ({informe.evaluacion.dimension})"
            )
        resumenes[sonda.clave] = resumir_prediccion(train, prediccion_test)
        rutas[sonda.clave] = escribir_submission(
            plantilla, prediccion_test, sonda.archivo_submission
        )
        guardar_modelo(
            informe, vectorizador, modelos, informe.umbrales, dimension
        )
        log(f"{sonda.clave}: {json.dumps(resumenes[sonda.clave], ensure_ascii=False)}")

    # 3) Registro en experiments.csv.
    filas_experiments = [
        construir_fila_configuracion(informe) for informe in informes
    ] + [
        construir_fila_submission(
            informe, resumenes[informe.variante.clave], rutas[informe.variante.clave],
            segundos_refit[informe.variante.clave],
        )
        for informe in informes
    ]
    escritas = registrar_en_experiments(filas_experiments)
    log(f"filas escritas en experiments.csv: {escritas}")

    # 4) Registro en el leaderboard log (sin kaggle_f1 todavia).
    filas_leaderboard = [
        construir_fila_leaderboard(
            informe, resumenes[informe.variante.clave], rutas[informe.variante.clave]
        )
        for informe in informes
    ]
    escritas_lb = registrar_en_leaderboard(filas_leaderboard)
    log(f"filas escritas en leaderboard_log.csv: {escritas_lb}")

    # 5) Handoff en reports/contexto.md.
    seccion = construir_seccion_contexto(
        informe_control,
        verificacion,
        informes,
        resumenes,
        rutas,
        informe_control.gap_train_oof,
    )
    ruta_contexto = actualizar_contexto(seccion)
    log(f"contexto actualizado: {ruta_contexto.relative_to(PROJECT_ROOT)}")

    # 6) Detalle completo en JSON.
    detalle = {
        "metadata": {
            "fecha": CONFIG["fecha"],
            "script": CONFIG["script"],
            "competencia": CONFIG["competencia"],
            "fase": "Fase 13d (dos sondas de robustez pre-registradas, D-013)",
            "validacion": VALIDACION,
            "regla_pre_registrada": (
                f"se adopta la sonda que supere el LB de v03 "
                f"({CONFIG['umbral_lb_a_batir']:.5f}); maximo 2 envios"
            ),
            "control": CONTROL.identificador,
            "verificacion_control": verificacion,
            "gap_train_oof": (
                "F1 in-sample media de los pliegues de entrenamiento (umbral 0.5) menos "
                "F1 OOF con 0.5"
            ),
            "idempotencia": (
                "Determinista salvo los tiempos; los registros se reescriben por id"
            ),
        },
        "control": informe_a_dict(informe_control),
        "sondas": [informe_a_dict(informe) for informe in informes],
        "comparativa_gap": {
            "control": round(informe_control.gap_train_oof, 6),
            **{
                informe.variante.identificador: {
                    "gap_train_oof": round(informe.gap_train_oof, 6),
                    "delta_vs_control": round(
                        informe.gap_train_oof - informe_control.gap_train_oof, 6
                    ),
                    "veredicto": clasificar_gap(
                        informe_control.gap_train_oof, informe.gap_train_oof
                    ),
                }
                for informe in informes
            },
        },
        "submissions": {
            informe.variante.clave: {
                "experiment_id": (
                    "S06" if informe.variante.clave == "v06" else "S07"
                ),
                "ruta": str(rutas[informe.variante.clave].relative_to(PROJECT_ROOT)),
                "umbrales": informe.umbrales,
                "prediccion_test": resumenes[informe.variante.clave],
                "modelo_joblib": str(
                    (
                        CONFIG["modelos_dir"]
                        / f"fase13d_sonda_{informe.variante.clave}.joblib"
                    ).relative_to(PROJECT_ROOT)
                ),
            }
            for informe in informes
        },
    }
    CONFIG["metricas_json"].parent.mkdir(parents=True, exist_ok=True)
    CONFIG["metricas_json"].write_text(
        json.dumps(detalle, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    log(f"detalle: {CONFIG['metricas_json'].relative_to(PROJECT_ROOT)}")
    log(f"duracion total: {time.perf_counter() - inicio:.2f} s")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - el punto de entrada CLI reporta todo
        import traceback

        traceback.print_exc()
        sys.exit(1)
