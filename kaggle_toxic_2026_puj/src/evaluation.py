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
