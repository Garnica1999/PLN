"""V10-03: arnes de evaluacion de candidatas con particiones repetidas y comparacion pareada.

Por que nace este modulo
-----------------------
V10-02 midio que la metrica corregida de V03 recorre mas de 0.017 entre particiones
equivalentes, con una desviacion tipica de 0.0053. Con una banda de ese tamano, un delta de
+0.002 medido en un solo calculo no distingue una mejora real de una particion afortunada, y
ese es exactamente el mecanismo por el que D-067 concluyo que la mejora local de las lineas
anteriores era el promediado y no la tecnica.

El addendum de protocolo de `tasks/plan_v10.md` obliga por eso a que **toda** candidata se
mida con tres cifras, y las tres tienen que salir del mismo instrumento:

1. delta medio sobre varias particiones de grupo con semillas distintas;
2. intervalo al 95 por ciento pareado del delta, con re-muestreo de filas;
3. recuento de particiones que mejoran, de 0 a ``K``.

Este modulo es ese instrumento. Las tres lineas de la via A lo importan, de modo que sus
resultados son comparables entre si y con la referencia.

Como esta escrito
-----------------
Una candidata es **una funcion** que recibe la matriz de probabilidades fuera de muestra de
la particion y devuelve una matriz binaria de predicciones. Asi el arnes no sabe nada de que
hace la candidata: no reimplementa la calibracion, ni la regla de decision, ni los
umbrales, y no puede introducir sesgo propio en la comparacion.

Lo que el arnes **no** hace, y es deliberado: no optimiza nada, no elige la mejor particion
y no promedia de forma que el resultado dependa de mas o menos informacion. Las tres cifras
se reportan tal cual salen.

Uso
---
Este fichero se carga con ``importlib`` porque vive en ``experiments/`` y no es un modulo
importable desde ``src/``. **El registro en ``sys.modules`` tiene que ocurrir antes** de
ejecutar el modulo, porque el decorador ``dataclass`` de la biblioteca estandar consulta
``sys.modules`` para resolver los tipos de los campos y lanza excepcion si no lo encuentra.
El patron correcto, que es el que usan las tres lineas de la via A, es:

    import sys
    from importlib import util

    especificacion = util.spec_from_file_location(
        "arnes", "experiments/v10_03_arnes_evaluacion.py"
    )
    arnes = util.module_from_spec(especificacion)
    sys.modules["arnes"] = arnes
    especificacion.loader.exec_module(arnes)

    veredicto = arnes.evaluar(nombre, textos, y, grupos, candidata)
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    ETIQUETAS,
    metrica_macro,
    metricas_multietiqueta,
)
from src.splits import RANDOM_STATE, combinaciones

UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
REFERENCIA_CANONICA = 0.7772114525165721
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
SEMILLAS_EVALUACION = (42, 7, 123, 2024, 5150)
CANDIDATAS_MEJORAN_MINIMO = 4
DELTA_MINIMO = 0.002
REPETICIONES_BOOTSTRAP = 2000
SEMILLA_BOOTSTRAP = 20260926

# Una candidata recibe las probabilidades de la particion y devuelve predicciones binarias.
Candidata = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass
class ParticionEvaluada:
    """Resultado de evaluar una candidata en una particion.

    Attributes:
        semilla: semilla con la que se construyo la particion.
        metrica_referencia: metrica de V03 con los umbrales congelados.
        metrica_candidata: metrica de la candidata.
        delta: diferencia candidata menos referencia.
        f1_referencia: F1 por etiqueta de la referencia.
        f1_candidata: F1 por etiqueta de la candidata.
        celdas_distintas: celdas en las que la candidata discrepa de la referencia.
    """

    semilla: int
    metrica_referencia: float
    metrica_candidata: float
    delta: float
    f1_referencia: dict[str, float]
    f1_candidata: dict[str, float]
    celdas_distintas: int

    @property
    def mejora(self) -> bool:
        """Indica si la candidata mejora en esta particion.

        Returns:
            ``True`` si el delta es estrictamente positivo.
        """
        return self.delta > 0.0


@dataclass
class Veredicto:
    """Veredicto completo de una candidata, con las tres cifras del addendum.

    Attributes:
        nombre: nombre de la candidata.
        particiones: detalle de cada particion evaluada.
        delta_medio: delta medio sobre las particiones.
        delta_mediano: delta mediano sobre las particiones.
        particiones_que_mejoran: recuento de particiones con delta positivo.
        total_particiones: numero de particiones evaluadas.
        ic95_inferior: extremo inferior del intervalo pareado al 95 por ciento.
        ic95_superior: extremo superior del intervalo pareado al 95 por ciento.
        delta_observado: diferencia de la particion de referencia, la canonica.
        fraccion_a_favor: fraccion de repeticiones de re-muestreo a favor de la candidata.
        celdas_distintas_promedio: celdas discrepantes en promedio.
        adoptable: si cumple los tres criterios del addendum.
        criterios: detalle booleano de cada criterio.
    """

    nombre: str
    particiones: list[ParticionEvaluada] = field(default_factory=list)
    delta_medio: float = 0.0
    delta_mediano: float = 0.0
    particiones_que_mejoran: int = 0
    total_particiones: int = 0
    ic95_inferior: float = 0.0
    ic95_superior: float = 0.0
    delta_observado: float = 0.0
    fraccion_a_favor: float = 0.0
    celdas_distintas_promedio: float = 0.0
    adoptable: bool = False
    criterios: dict[str, bool] = field(default_factory=dict)

    def resumen(self) -> str:
        """Construye una linea de resumen legible.

        Returns:
            Cadena con las cifras principales del veredicto.
        """
        marca = "ADOPTABLE" if self.adoptable else "NO ADOPTABLE"
        return (
            f"{self.nombre}: delta medio {self.delta_medio:+.6f} | "
            f"{self.particiones_que_mejoran}/{self.total_particiones} particiones mejoran | "
            f"IC95 pareado [{self.ic95_inferior:+.6f}, {self.ic95_superior:+.6f}] | "
            f"{self.celdas_distintas_promedio:.0f} celdas de media | {marca}"
        )

    def a_dict(self) -> dict[str, object]:
        """Convierte el veredicto en un diccionario serializable.

        Returns:
            Diccionario con todas las cifras del veredicto.
        """
        return {
            "nombre": self.nombre,
            "delta_medio": round(self.delta_medio, 6),
            "delta_mediano": round(self.delta_mediano, 6),
            "particiones_que_mejoran": self.particiones_que_mejoran,
            "total_particiones": self.total_particiones,
            "ic95_inferior": round(self.ic95_inferior, 6),
            "ic95_superior": round(self.ic95_superior, 6),
            "delta_observado_particion_canonica": round(self.delta_observada, 6),
            "fraccion_repeticiones_a_favor": round(self.fraccion_a_favor, 4),
            "celdas_distintas_promedio": round(self.celdas_distintas_promedio, 1),
            "adoptable": self.adoptable,
            "criterios": self.criterios,
            "detalle_particiones": [
                {
                    "semilla": p.semilla,
                    "metrica_referencia": round(p.metrica_referencia, 6),
                    "metrica_candidata": round(p.metrica_candidata, 6),
                    "delta": round(p.delta, 6),
                    "mejora": p.mejora,
                    "celdas_distintas": p.celdas_distintas,
                    "f1_referencia": p.f1_referencia,
                    "f1_candidata": p.f1_candidata,
                }
                for p in self.particiones
            ],
        }


def particion_por_semilla(
    textos: pd.Series, y: np.ndarray, grupos: np.ndarray, semilla: int, n_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construye una particion de grupo con la semilla indicada.

    Args:
        textos: serie de textos.
        y: matriz binaria de etiquetas.
        grupos: grupos fusionados.
        semilla: semilla del particionador.
        n_splits: numero de pliegues.

    Returns:
        Lista de pares de indices de entrenamiento y validacion.
    """
    marco = pd.DataFrame({e: y[:, i] for i, e in enumerate(ETIQUETAS)})
    return list(
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=semilla).split(
            textos, combinaciones(marco), groups=grupos
        )
    )


def oof_de_v03(
    textos: pd.Series, y: np.ndarray, pliegues: Sequence[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """Calcula las probabilidades fuera de muestra de V03 con los pliegues dados.

    El espacio de caracteristicas se ajusta **dentro** de cada pliegue de entrenamiento. Si
    se ajustase antes, sobre las filas de validacion, seria fuga, y V03 seria artificialmente
    buena.

    Args:
        textos: serie de textos.
        y: matriz binaria de etiquetas.
        pliegues: pares de indices de entrenamiento y validacion.

    Returns:
        Matriz de probabilidades con una columna por etiqueta.
    """
    probabilidades = np.zeros((len(y), 3), dtype=np.float64)
    for entrenamiento, validacion in pliegues:
        vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
        vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
        ajuste = sparse.hstack(
            [
                vectorizador_word.fit_transform(textos.iloc[entrenamiento]),
                vectorizador_char.fit_transform(textos.iloc[entrenamiento]),
            ],
            format="csr",
        )
        consulta = sparse.hstack(
            [
                vectorizador_word.transform(textos.iloc[validacion]),
                vectorizador_char.transform(textos.iloc[validacion]),
            ],
            format="csr",
        )
        for etiqueta in range(3):
            estimador = LogisticRegression(
                C=2.0,
                class_weight="balanced",
                max_iter=2000,
                solver="liblinear",
                random_state=RANDOM_STATE,
            )
            estimador.fit(ajuste, y[entrenamiento, etiqueta])
            probabilidades[validacion, etiqueta] = estimador.predict_proba(consulta)[:, 1]
    return probabilidades


def predicciones_de_referencia(probabilidades: np.ndarray) -> np.ndarray:
    """Aplica los umbrales congelados de V03.

    Args:
        probabilidades: matriz de probabilidades.

    Returns:
        Matriz binaria de predicciones.
    """
    return (probabilidades >= UMBRALES_V03[None, :]).astype(int)


def intervalo_pareado(
    y: np.ndarray,
    candidata: np.ndarray,
    referencia: np.ndarray,
    repeticiones: int = REPETICIONES_BOOTSTRAP,
    semilla: int = SEMILLA_BOOTSTRAP,
) -> tuple[float, float, float]:
    """Calcula el intervalo al 95 por ciento de la diferencia por comparacion pareada.

    La comparacion es pareada porque en cada repeticion se re-muestrean filas y se calcula la
    diferencia **sobre esas mismas filas** para los dos sistemas. Asi desaparece la varianza
    debida a que unas filas sean mas faciles que otras, que es la que domina la banda medida
    en V10-02.

    Args:
        y: matriz binaria de etiquetas reales.
        candidata: predicciones de la candidata.
        referencia: predicciones de la referencia.
        repeticiones: numero de repeticiones de re-muestreo.
        semilla: semilla del generador, para que el resultado sea reproducible.

    Returns:
        Tupla con la diferencia observada, el extremo inferior y el superior del intervalo.
    """
    generador = np.random.Generator(np.random.PCG64(semilla))
    diferencias = np.empty(repeticiones, dtype=np.float64)
    n_filas = len(y)
    for repeticion in range(repeticiones):
        indices = generador.integers(0, n_filas, n_filas)
        diferencias[repeticion] = metrica_macro(
            y[indices], candidata[indices]
        ) - metrica_macro(y[indices], referencia[indices])
    percentiles = np.percentile(diferencias, [2.5, 97.5])
    return (
        float(metrica_macro(y, candidata) - metrica_macro(y, referencia)),
        float(percentiles[0]),
        float(percentiles[1]),
    )


def evaluar(
    nombre: str,
    textos: pd.Series,
    y: np.ndarray,
    grupos: np.ndarray,
    candidata: Candidata,
    semillas: Sequence[int] = SEMILLAS_EVALUACION,
    n_splits: int = 5,
) -> Veredicto:
    """Evalua una candidata con las tres cifras del addendum de protocolo.

    Args:
        nombre: nombre de la candidata, para el informe.
        textos: serie de textos del desarrollo.
        y: matriz binaria de etiquetas.
        grupos: grupos fusionados del desarrollo.
        candidata: funcion que recibe probabilidades, etiquetas y grupos, y devuelve
            predicciones binarias.
        semillas: semillas de particion a evaluar.
        n_splits: numero de pliegues por particion.

    Returns:
        Veredicto con el detalle por particion y las tres cifras agregadas.
    """
    veredicto = Veredicto(nombre=nombre, total_particiones=len(semillas))
    for semilla in semillas:
        pliegues = particion_por_semilla(textos, y, grupos, semilla, n_splits)
        probabilidades = oof_de_v03(textos, y, pliegues)
        predicciones_referencia = predicciones_de_referencia(probabilidades)
        predicciones_candidata = candidata(probabilidades, y, grupos)
        metrica_referencia = metrica_macro(y, predicciones_referencia)
        metrica_candidata = metrica_macro(y, predicciones_candidata)
        veredicto.particiones.append(
            ParticionEvaluada(
                semilla=semilla,
                metrica_referencia=metrica_referencia,
                metrica_candidata=metrica_candidata,
                delta=metrica_candidata - metrica_referencia,
                f1_referencia={
                    k: round(v, 6)
                    for k, v in metricas_multietiqueta(y, predicciones_referencia)[
                        "f1_por_etiqueta"
                    ].items()
                },
                f1_candidata={
                    k: round(v, 6)
                    for k, v in metricas_multietiqueta(y, predicciones_candidata)[
                        "f1_por_etiqueta"
                    ].items()
                },
                celdas_distintas=int((predicciones_candidata != predicciones_referencia).sum()),
            )
        )

    deltas = np.array([p.delta for p in veredicto.particiones], dtype=np.float64)
    veredicto.delta_medio = float(deltas.mean())
    veredicto.delta_mediano = float(np.median(deltas))
    veredicto.particiones_que_mejoran = int((deltas > 0).sum())
    veredicto.celdas_distintas_promedio = float(
        np.mean([p.celdas_distintas for p in veredicto.particiones])
    )

    pliegues_canonicos = particion_por_semilla(textos, y, grupos, RANDOM_STATE, n_splits)
    probabilidades_canonicas = oof_de_v03(textos, y, pliegues_canonicos)
    referencia_canonica = predicciones_de_referencia(probabilidades_canonicas)
    candidata_canonica = candidata(probabilidades_canonicas, y, grupos)
    observada, inferior, superior = intervalo_pareado(y, candidata_canonica, referencia_canonica)
    veredicto.delta_observado = observada
    veredicto.ic95_inferior = inferior
    veredicto.ic95_superior = superior
    veredicto.fraccion_a_favor = float(
        veredicto.particiones_que_mejoran / max(1, veredicto.total_particiones)
    )

    veredicto.criterios = {
        "delta_medio_al_menos_0.002": bool(veredicto.delta_medio >= DELTA_MINIMO),
        "al_menos_4_de_5_particiones_mejoran": bool(
            veredicto.particiones_que_mejoran >= CANDIDATAS_MEJORAN_MINIMO
        ),
        "intervalo_pareado_no_incluye_cero": bool(
            veredicto.ic95_inferior > 0.0 or veredicto.ic95_superior < 0.0
        ),
    }
    veredicto.adoptable = all(veredicto.criterios.values())
    return veredicto


def tabla_filas(veredicto: Veredicto) -> list[list[str]]:
    """Construye la tabla de detalle por particion lista para un informe.

    Args:
        veredicto: veredicto de la candidata.

    Returns:
        Lista de filas, la primera de las cuales es la cabecera.
    """
    filas: list[list[str]] = [
        ["semilla", "referencia", "candidata", "delta", "mejora", "celdas"]
    ]
    for particion in veredicto.particiones:
        filas.append(
            [
                str(particion.semilla),
                f"{particion.metrica_referencia:.6f}",
                f"{particion.metrica_candidata:.6f}",
                f"{particion.delta:+.6f}",
                "si" if particion.mejora else "no",
                str(particion.celdas_distintas),
            ]
        )
    return filas
