"""Metricas de evaluacion y post-proceso para la clasificacion multi-etiqueta.

Contexto
--------
Competencia KAGGLE_TOXIC_2026_PUJ: tres etiquetas binarias por tweet (HS, TR,
AG). La metrica oficial es la media column-wise de la F1 binaria de la clase
positiva (``f1_media_columnas``, D-009); ``f1_macro6``, ``f1_micro`` y
``exact_match`` se reportan solo como diagnostico.

Trazabilidad
------------
``metricas_multietiqueta`` y ``aplicar_postproceso_jerarquico`` se
implementaron primero dentro de ``experiments/02_baseline_tfidf_lr.py``. Ese
script se conserva intacto como evidencia historica del baseline B1 (no se
modifica ni se borra), por lo que este modulo **duplica** aquella logica para
que los experimentos posteriores (03 en adelante) la reutilicen sin acoplarse a
un script de experimento. Si aqui cambia una definicion, la desviacion debe
registrarse en ``reports/contexto.md``: el script 02 mantiene la version
historica congelada.

Uso
---
    from src.evaluation import metricas_multietiqueta, aplicar_postproceso_jerarquico

    metricas = metricas_multietiqueta(y_true, y_pred)
    y_consistente = aplicar_postproceso_jerarquico(y_pred)
"""

from __future__ import annotations

from typing import Final, TypedDict

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import f1_score

ETIQUETAS: Final[tuple[str, ...]] = ("HS", "TR", "AG")

ArrayBinario = NDArray[np.integer]


class MetricasMultiEtiqueta(TypedDict):
    """Resultado de :func:`metricas_multietiqueta`.

    Attributes:
        f1_media_columnas: media de la F1 binaria (clase positiva) de las 3
            etiquetas; es la metrica primaria de seleccion local (D-009).
        f1_macro6: media de las F1 de las 6 clases (positiva y negativa por
            etiqueta); diagnostico.
        f1_micro: F1 global calculada sobre todas las celdas de la matriz.
        f1_samples: F1 promedio por fila (samples-average).
        exact_match: fraccion de filas con las tres etiquetas correctas.
        f1_por_etiqueta: F1 binaria de la clase positiva por etiqueta.
    """

    f1_media_columnas: float
    f1_macro6: float
    f1_micro: float
    f1_samples: float
    exact_match: float
    f1_por_etiqueta: dict[str, float]


def metricas_multietiqueta(
    y_true: ArrayBinario, y_pred: ArrayBinario
) -> MetricasMultiEtiqueta:
    """Calcula las metricas multi-etiqueta acordadas en el proyecto.

    Devuelve la metrica primaria (``f1_media_columnas``, D-009) junto con las
    secundarias de diagnostico y el desglose por etiqueta. Todas las F1 binarias
    usan ``zero_division=0`` para que un pliegue sin positivos predichos no
    produzca ``nan``.

    Args:
        y_true: matriz binaria verdadera de forma ``(n_filas, 3)`` en el orden
            de :data:`ETIQUETAS` (HS, TR, AG).
        y_pred: matriz binaria predicha, con la misma forma y orden que
            ``y_true``.

    Returns:
        Diccionario con las metricas globales y ``f1_por_etiqueta``.

    Raises:
        ValueError: si las formas no coinciden o el numero de columnas no es 3.
    """
    verdad = np.asarray(y_true)
    prediccion = np.asarray(y_pred)
    if verdad.shape != prediccion.shape:
        raise ValueError(
            f"y_true {verdad.shape} e y_pred {prediccion.shape} deben coincidir"
        )
    if verdad.ndim != 2 or verdad.shape[1] != len(ETIQUETAS):
        raise ValueError(
            f"Se esperaba una matriz (n, {len(ETIQUETAS)}); recibida {verdad.shape}"
        )

    f1_positivas = [
        float(f1_score(verdad[:, indice], prediccion[:, indice], zero_division=0))
        for indice in range(len(ETIQUETAS))
    ]
    f1_negativas = [
        float(
            f1_score(
                1 - verdad[:, indice], 1 - prediccion[:, indice], zero_division=0
            )
        )
        for indice in range(len(ETIQUETAS))
    ]
    return {
        "f1_media_columnas": float(np.mean(f1_positivas)),
        "f1_macro6": float(np.mean(f1_positivas + f1_negativas)),
        "f1_micro": float(
            f1_score(verdad.ravel(), prediccion.ravel(), zero_division=0)
        ),
        "f1_samples": float(
            f1_score(verdad, prediccion, average="samples", zero_division=0)
        ),
        "exact_match": float((verdad == prediccion).all(axis=1).mean()),
        "f1_por_etiqueta": {
            etiqueta: valor
            for etiqueta, valor in zip(ETIQUETAS, f1_positivas, strict=True)
        },
    }


def metrica_macro(y_true: ArrayBinario, y_pred: ArrayBinario) -> float:
    """Calcula la metrica primaria con aritmetica vectorial.

    Es identica a ``metricas_multietiqueta(y_true, y_pred)["f1_media_columnas"]``,
    pero Pensada para evaluarse miles de veces dentro de un bootstrap. Se usa en
    la fase v5 para estimar el intervalo de la diferencia entre dos candidatos.

    Nota de trazabilidad: los experimentos 02 y 14 de la fase v2 conservan su
    propia implementacion historica de la metrica; aqui solo se **anaden**
    funciones nuevas, sin cambiar ninguna definicion existente.

    Args:
        y_true: matriz binaria verdadera de forma ``(n_filas, 3)``.
        y_pred: matriz binaria predicha, con la misma forma.

    Returns:
        Media column-wise de la F1 binaria de la clase positiva.
    """
    columnas = []
    for indice in range(len(ETIQUETAS)):
        verdaderos = y_true[:, indice] == 1
        predichos = y_pred[:, indice] == 1
        verdaderos_positivos = float(np.sum(verdaderos & predichos))
        falsos_positivos = float(np.sum(~verdaderos & predichos))
        falsos_negativos = float(np.sum(verdaderos & ~predichos))
        denominador = 2 * verdaderos_positivos + falsos_positivos + falsos_negativos
        columnas.append(2 * verdaderos_positivos / denominador if denominador > 0 else 0.0)
    return float(np.mean(columnas))


def bootstrap_diferencia_pareada(
    y_true: ArrayBinario,
    pred_a: ArrayBinario,
    pred_b: ArrayBinario,
    repeticiones: int = 10_000,
    semilla: int = 20260925,
) -> dict[str, object]:
    """Estima el intervalo de bootstrap de la diferencia entre dos candidatos.

    El estimando es exactamente la metrica primaria: en cada remuestreo se
    recalcula ``f1_media_columnas`` de ambos candidatos sobre las mismas filas
    (bootstrap **pareado**), de modo que la correlacion entre ellos se conserva y
    el intervalo es mas estrecho que el de dos intervalos independientes.

    Args:
        y_true: matriz binaria verdadera de forma ``(n_filas, 3)``.
        pred_a: predicciones del candidato A.
        pred_b: predicciones de la referencia B.
        repeticiones: numero de remuestreos.
        semilla: semilla del generador, para reproducibilidad.

    Returns:
        Diccionario con la diferencia observada, su IC95, el balance de celdas y
        el numero de repeticiones.

    Raises:
        RuntimeError: si la version vectorial no reproduce la metrica oficial.
    """
    oficial_a = metricas_multietiqueta(y_true, pred_a)["f1_media_columnas"]
    oficial_b = metricas_multietiqueta(y_true, pred_b)["f1_media_columnas"]
    if abs(metrica_macro(y_true, pred_a) - oficial_a) > 1e-12:
        raise RuntimeError("metrica_macro no reproduce f1_media_columnas")
    if abs(metrica_macro(y_true, pred_b) - oficial_b) > 1e-12:
        raise RuntimeError("metrica_macro no reproduce f1_media_columnas")
    generador = np.random.default_rng(semilla)
    n = y_true.shape[0]
    diferencias = np.empty(repeticiones, dtype=float)
    for repeticion in range(repeticiones):
        indices = generador.integers(0, n, n)
        diferencias[repeticion] = metrica_macro(
            y_true[indices], pred_a[indices]
        ) - metrica_macro(y_true[indices], pred_b[indices])
    cuantiles = np.quantile(diferencias, [0.025, 0.5, 0.975])
    celdas_a = _f1_por_celda(y_true, pred_a)
    celdas_b = _f1_por_celda(y_true, pred_b)
    diferencia = celdas_a - celdas_b
    return {
        "metrica_a": oficial_a,
        "metrica_b": oficial_b,
        "diferencia_observada": round(oficial_a - oficial_b, 6),
        "ic95": [round(float(cuantiles[0]), 6), round(float(cuantiles[2]), 6)],
        "mediana_bootstrap": round(float(cuantiles[1]), 6),
        "celdas_mejoradas": int(np.sum(diferencia > 1e-9)),
        "celdas_igualadas": int(np.sum(np.abs(diferencia) <= 1e-9)),
        "celdas_empeoradas": int(np.sum(diferencia < -1e-9)),
        "repeticiones": repeticiones,
    }


def _f1_por_celda(y_true: ArrayBinario, y_pred: ArrayBinario) -> NDArray[np.float64]:
    """Calcula la F1 de cada celda (fila, etiqueta).

    Sirve para contar cuantas celdas mejora o rompe un candidato frente a otro; no
    es la metrica primaria y no se reporta como tal.

    Args:
        y_true: matriz binaria verdadera.
        y_pred: matriz binaria predicha.

    Returns:
        Vector con una F1 por celda.
    """
    columnas = []
    for indice in range(len(ETIQUETAS)):
        verdaderos = y_true[:, indice] == 1
        predichos = y_pred[:, indice] == 1
        verdaderos_positivos = (verdaderos & predichos).astype(float)
        falsos_positivos = (~verdaderos & predichos).astype(float)
        falsos_negativos = (verdaderos & ~predichos).astype(float)
        denominador = 2 * verdaderos_positivos + falsos_positivos + falsos_negativos
        columnas.append(
            np.where(denominador > 0, 2 * verdaderos_positivos / np.maximum(denominador, 1e-12), 0.0)
        )
    return np.mean(np.column_stack(columnas), axis=1)


def aplicar_postproceso_jerarquico(y_pred: ArrayBinario) -> ArrayBinario:
    """Fuerza TR=0 y AG=0 cuando HS=0 (estructura observada en train, D-001).

    El esquema de anotacion del corpus cumple TR=1 => HS=1 en 519/519 casos y
    AG=1 con HS=0 en solo 26 casos (1.5 %), por lo que anular TR y AG cuando el
    modelo predice HS=0 elimina falsos positivos hijos sin destruir casi ningun
    positivo verdadero (H8).

    Args:
        y_pred: matriz binaria predicha de forma ``(n_filas, 3)`` en el orden
            HS, TR, AG.

    Returns:
        Copia de la matriz con el post-proceso aplicado; la entrada no se
        modifica.

    Raises:
        ValueError: si la matriz no tiene 3 columnas.
    """
    ajustada = np.asarray(y_pred).copy()
    if ajustada.ndim != 2 or ajustada.shape[1] != len(ETIQUETAS):
        raise ValueError(
            f"Se esperaba una matriz (n, {len(ETIQUETAS)}); recibida {ajustada.shape}"
        )
    sin_discurso_de_odio = ajustada[:, 0] == 0
    ajustada[sin_discurso_de_odio, 1] = 0
    ajustada[sin_discurso_de_odio, 2] = 0
    return ajustada
