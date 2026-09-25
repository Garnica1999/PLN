"""Protocolo de validacion v2: grupos fusionados, desarrollo/holdout y pliegues.

Resumen
-------
La validacion historica agrupaba solo por near-duplicado, lo que resultaba
optimista (D-015): al anadir el pseudo-autor (primera mencion ``@usuario``) como
criterio de agrupamiento, la F1 out-of-fold del modelo final cae de 0.7927 a
0.7838 y el gap frente al leaderboard se reduce de -0.0128 a -0.0039.

Este modulo centraliza:

- el calculo de pseudo-autores y de los grupos fusionados (union-find),
- la particion congelada desarrollo/holdout (``StratifiedGroupKFold(7)``),
- los pliegues corregidos del desarrollo (``StratifiedGroupKFold(5)``).

Regla: ninguna transformacion puede ajustarse con validacion ni con test; este
modulo solo define particiones.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRUPOS_NEAR_DUP_PATH = PROJECT_ROOT / "data" / "processed" / "grupos_near_duplicados.csv"

MENTION_PATTERN = re.compile(r"@(\w+)")
ETIQUETAS = ("HS", "TR", "AG")
N_SPLITS_HOLDOUT = 7
N_SPLITS_DEV = 5
RANDOM_STATE = 42


def cargar_grupos_near_duplicado() -> pd.Series:
    """Carga la asignacion de grupo de near-duplicado por id.

    Returns:
        Serie indexada por ``id`` con el grupo de near-duplicado.

    Raises:
        FileNotFoundError: si la auditoria de Fase 0 no se ha ejecutado.
    """
    if not GRUPOS_NEAR_DUP_PATH.exists():
        raise FileNotFoundError(
            f"Falta {GRUPOS_NEAR_DUP_PATH}. Ejecuta experiments/00_auditoria_datos.py."
        )
    return pd.read_csv(GRUPOS_NEAR_DUP_PATH).set_index("id")["grupo"]


def pseudo_autores(train: pd.DataFrame) -> list[str]:
    """Extrae el pseudo-autor (primera mencion) de cada tweet.

    Args:
        train: DataFrame con la columna ``text``.

    Returns:
        Lista de pseudo-autores alineada con ``train`` (cadena vacia si no hay).
    """
    autores = []
    for text in train["text"]:
        match = MENTION_PATTERN.search(text)
        autores.append(match.group(1).lower() if match else "")
    return autores


def grupos_fusionados(train: pd.DataFrame) -> np.ndarray:
    """Fusiona near-duplicados y pseudo-autores en una unica particion.

    Args:
        train: DataFrame con ``id`` y ``text``.

    Returns:
        Arreglo de grupos alineado con las filas de ``train``.
    """
    near_dup = cargar_grupos_near_duplicado()
    ids = train["id"].astype(int).to_numpy()
    autores = pseudo_autores(train)
    padre = list(range(len(ids)))

    def encontrar(nodo: int) -> int:
        """Devuelve la raiz del nodo con compresion de caminos."""
        while padre[nodo] != nodo:
            padre[nodo] = padre[padre[nodo]]
            nodo = padre[nodo]
        return nodo

    def unir(a: int, b: int) -> None:
        """Une dos nodos en la misma componente."""
        raiz_a, raiz_b = encontrar(a), encontrar(b)
        if raiz_a != raiz_b:
            padre[max(raiz_a, raiz_b)] = min(raiz_a, raiz_b)

    representante_near: dict[int, int] = {}
    for posicion, identificador in enumerate(ids):
        grupo = int(near_dup.get(identificador, identificador))
        if grupo in representante_near:
            unir(posicion, representante_near[grupo])
        else:
            representante_near[grupo] = posicion
    representante_autor: dict[str, int] = {}
    for posicion, autor in enumerate(autores):
        if not autor:
            continue
        if autor in representante_autor:
            unir(posicion, representante_autor[autor])
        else:
            representante_autor[autor] = posicion
    return np.array([encontrar(nodo) for nodo in range(len(ids))])


def combinaciones(train: pd.DataFrame) -> pd.Series:
    """Construye la etiqueta de combinacion HS x TR x AG por fila.

    Args:
        train: DataFrame con las columnas de etiquetas.

    Returns:
        Serie de cadenas de tres digitos (por ejemplo ``"101"``).
    """
    return (
        train["HS"].astype(str) + train["TR"].astype(str) + train["AG"].astype(str)
    )


def holdout_v2(train: pd.DataFrame, grupos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divide el train en desarrollo y holdout congelado, agrupado y estratificado.

    Args:
        train: DataFrame de entrenamiento completo.
        grupos: grupos fusionados alineados con ``train``.

    Returns:
        Tupla ``(indices_dev, indices_holdout)`` con indices posicionales.
    """
    divisor = StratifiedGroupKFold(
        n_splits=N_SPLITS_HOLDOUT, shuffle=True, random_state=RANDOM_STATE
    )
    pliegues = list(divisor.split(train, combinaciones(train), groups=grupos))
    # El pliegue 0 queda como holdout; el desarrollo es la union de las
    # validaciones de los pliegues restantes (los entrenamientos incluirian el
    # holdout y romperian la separacion).
    indices_holdout = np.sort(pliegues[0][1])
    indices_dev = np.sort(np.concatenate([pliegue[1] for pliegue in pliegues[1:]]))
    _verificar_sin_fuga(grupos, indices_dev, indices_holdout)
    return indices_dev, indices_holdout


def pliegues_corregidos(
    train_dev: pd.DataFrame,
    grupos_dev: np.ndarray,
    n_splits: int = N_SPLITS_DEV,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construye los pliegues corregidos del conjunto de desarrollo.

    Args:
        train_dev: DataFrame del conjunto de desarrollo.
        grupos_dev: grupos fusionados alineados con ``train_dev``.
        n_splits: numero de pliegues.

    Returns:
        Lista de tuplas ``(indices_entrenamiento, indices_validacion)``.
    """
    divisor = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    pliegues = []
    for indices_entrenamiento, indices_validacion in divisor.split(
        train_dev, combinaciones(train_dev), groups=grupos_dev
    ):
        _verificar_sin_fuga(grupos_dev, indices_entrenamiento, indices_validacion)
        pliegues.append((indices_entrenamiento, indices_validacion))
    return pliegues


def _verificar_sin_fuga(
    grupos: np.ndarray, indices_a: np.ndarray, indices_b: np.ndarray
) -> None:
    """Comprueba que dos particiones no comparten grupos.

    Args:
        grupos: grupos alineados con las filas.
        indices_a: indices de la primera particion.
        indices_b: indices de la segunda particion.

    Raises:
        RuntimeError: si hay grupos compartidos.
    """
    grupos_a = set(grupos[indices_a].tolist())
    grupos_b = set(grupos[indices_b].tolist())
    if grupos_a & grupos_b:
        raise RuntimeError("Hay grupos compartidos entre las particiones")
