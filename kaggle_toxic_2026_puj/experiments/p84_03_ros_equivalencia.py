"""p84_03_ros_equivalencia.py
Fecha: 2026-10-02
Descripcion: EXP-03 (P84-03): micro-verificacion de equivalencia del sobremuestreo aleatorio
(ROS) de positivos frente al control V03 en un clasificador lineal con penalizacion L2
(liblinear). El diseno original (medir si ROS supera a class_weight=balanced con Gate 1) es
matematicamente degenerado: para un modelo lineal convexo con L2 la duplicacion de filas es
una reasignacion de pesos por instancia. Este script mide esa equivalencia por pliegue y deja
un artefacto verificable; ROS queda cerrado como no promovible.

Brazos por pliegue (5 pliegues canonicos, vectorizador TF-IDF ajustado SOLO con outer_train y
ANTES de cualquier duplicacion):
- A : control V03 exacto: TF-IDF word(1,2)+char_wb(3,5) min_df=2 sublinear + LR C=2.0
      class_weight='balanced' liblinear rs=42. Debe reproducir features/oof_v03_final.npz con
      tolerancia 1e-12.
- B1: ROS r=1.0, n_pos' = round(r*n_neg), duplicacion fisica de positivos, pesos uniformes
      con masa total n: w = n/(n_pos' + n_neg), C=2. Equivalencia esperada con A.
- B2: ROS r=0.5 con la misma normalizacion y C=2 (reweighting mas suave que balanced).
- B3: ROS r=1.0 SIN normalizar (pesos 1), C=2 nominal; se reporta el C efectivo declarado
      C_efectivo = C * n_neg * (1 + r) / n por pliegue y etiqueta.

Duplicacion determinista: las copias se eligen con numpy.random.default_rng(20260926) sobre una
permutacion de los indices positivos LOCALES de outer_train (generador fresco por pliegue y
etiqueta para que el resultado sea invariante al subconjunto de pliegues ejecutado); se reportan
los recuentos enteros por pliegue y etiqueta. El objetivo n_pos' = round(r*n_neg) usa el round
half-to-even (bancario) de Python: con r=0.5 (B2) el n_pos' realizado depende de la paridad de
n_neg (si n_neg es impar, el empate .5 se redondea al entero par). La validacion nunca se
duplica ni se usa para elegir copias.

Criterio de equivalencia B1-A (pre-declarado): max|proba_B1 - proba_A| <= 1e-3 y
|delta_F1| <= 0.001 (lectura primaria: umbrales congelados 0.44/0.51/0.46). Si no se cumple:
STOP de la promocion y reporte de REFUTACION PARCIAL con la causa observada (redondeo entero de
las copias o trayectoria numerica del solver). Ningun brazo va a Gate 2 ni a EXP-06; la decision
pre-declarada de cierre se registra con los numeros.

Reglas duras de esta corrida:
- Gate 0: hashes congelados verificados; dev 1483 / holdout 0 / test 0; identidad V03 contra
  features/oof_v03_final.npz con tolerancia 1e-12 antes de leer cualquier brazo ROS.
- Nunca se abre data/raw/toxic_test.csv; nunca se evalua ni entrena con el holdout.
- Bootstrap pareado por filas, 10.000 remuestreos, semilla 20260926 (explicita).
- En --humo no se tocan experiments.csv, el npz OOF ni los informes finales; solo parciales
  con sufijo _humo.

Especificacion: docs/p84_spec.md seccion 4 (EXP-03, redisenada) + seccion 1 (convenciones).
Preregistro: docs/p84_preregistro.md (Gate 0, particiones y bootstrap).

Uso:
    /home/carlos/venv/bin/python experiments/p84_03_ros_equivalencia.py
    /home/carlos/venv/bin/python experiments/p84_03_ros_equivalencia.py --humo
    /home/carlos/venv/bin/python experiments/p84_03_ros_equivalencia.py --pliegues 1,2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train, sha256_of
from src.evaluation import (
    ETIQUETAS,
    bootstrap_diferencia_pareada,
    metrica_macro,
    metricas_multietiqueta,
)
from src.splits import grupos_fusionados, holdout_v2, pliegues_corregidos

# ===== SECCION 0: constantes congeladas, configuracion y cabeceras =====

HASH_TRAIN: Final[str] = "3d40e7345b588860cf4744914be88ddc094e4f03591516c8176ab40b7f76d42f"
HASH_OOF_V03: Final[str] = "559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab"
HASH_GRUPOS: Final[str] = "afabf9ae1c295ea8f203120bf28578cb4bdb4e8f16b89178d820baeaac62e622"

V03_CCV: Final[float] = 0.7772114525165721
V03_F1_AT_05: Final[float] = 0.771778476256122
UMBRALES_CONGELADOS: Final[np.ndarray] = np.array([0.44, 0.51, 0.46], dtype=np.float64)

TOL_IDENTIDAD: Final[float] = 1e-12
SEMILLA_BOOTSTRAP: Final[int] = 20260926
REPETICIONES_BOOTSTRAP: Final[int] = 10_000
SEMILLA_DUPLICACION: Final[int] = 20260926
SEMILLA_LR: Final[int] = 42
SEMILLA_GLOBAL: Final[int] = 20260926

TOL_EQUIV_PROBA: Final[float] = 1e-3
TOL_EQUIV_F1: Final[float] = 1e-3

FILAS_TRAIN: Final[int] = 1731
FILAS_DEV: Final[int] = 1483
FILAS_HOLDOUT: Final[int] = 248
TAMANOS_PLIEGUES: Final[list[int]] = [297, 297, 296, 296, 297]
N_PLIEGUES: Final[int] = 5
MAX_ITER_LIBLINEAR: Final[int] = 1000

BRAZOS: Final[dict[str, dict[str, Any]]] = {
    "A": {"tipo": "control_v03", "r": None, "normalizado": None},
    "B1": {"tipo": "ros", "r": 1.0, "normalizado": True},
    "B2": {"tipo": "ros", "r": 0.5, "normalizado": True},
    "B3": {"tipo": "ros", "r": 1.0, "normalizado": False},
}
BRAZOS_ROS: Final[tuple[str, ...]] = ("B1", "B2", "B3")

PARAMS_LR_CONTROL: Final[dict[str, Any]] = {
    "C": 2.0,
    "class_weight": "balanced",
    "max_iter": MAX_ITER_LIBLINEAR,
    "solver": "liblinear",
    "random_state": SEMILLA_LR,
}
PARAMS_LR_ROS: Final[dict[str, Any]] = {
    "C": 2.0,
    "class_weight": None,
    "max_iter": MAX_ITER_LIBLINEAR,
    "solver": "liblinear",
    "random_state": SEMILLA_LR,
}

CONFIG: Final[dict[str, Any]] = {
    "train_path": PROJECT_ROOT / "data" / "raw" / "toxic_train.csv",
    "grupos_path": PROJECT_ROOT / "data" / "processed" / "grupos_near_duplicados.csv",
    "oof_v03_path": PROJECT_ROOT / "features" / "oof_v03_final.npz",
    "experiments_csv": PROJECT_ROOT / "experiments.csv",
    "parciales_dir": PROJECT_ROOT / "reports" / "p84_03_parciales",
    "informe_json": PROJECT_ROOT / "reports" / "p84_03_ros.json",
    "informe_md": PROJECT_ROOT / "reports" / "p84_03_ros.md",
    "oof_npz": PROJECT_ROOT / "features" / "p84_03_oof_ros.npz",
    "fecha": "2026-10-02",
    "script": "experiments/p84_03_ros_equivalencia.py",
    "competencia": "KAGGLE_TOXIC_2026_PUJ",
    "pliegues_por_defecto": (1, 2, 3, 4, 5),
}

# Nota: data/raw/toxic_test.csv no se referencia en ninguna ruta de este script.
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

DECISION_PREDECLARADA: Final[str] = (
    "ROS cerrado por equivalencia con reweighting y class_weight; no promovible"
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso con el prefijo de la fase.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P84-03] {mensaje}", flush=True)


def verificar_ascii(texto: str, contexto: str) -> None:
    """Comprueba que un texto es ASCII puro.

    Args:
        texto: contenido a validar.
        contexto: descripcion usada en el mensaje de error.

    Raises:
        SystemExit: si hay algun caracter con codigo mayor que 127.
    """
    malos = sorted({caracter for caracter in texto if ord(caracter) > 127})
    if malos:
        raise SystemExit(f"STOP: {contexto} contiene caracteres no ASCII: {malos}")


def _json_default(objeto: Any) -> Any:
    """Convierte tipos de numpy y Path a tipos serializables en JSON.

    Args:
        objeto: valor a convertir.

    Returns:
        Valor serializable.

    Raises:
        TypeError: si el tipo no esta soportado.
    """
    if isinstance(objeto, np.integer):
        return int(objeto)
    if isinstance(objeto, np.floating):
        return float(objeto)
    if isinstance(objeto, np.bool_):
        return bool(objeto)
    if isinstance(objeto, np.ndarray):
        return objeto.tolist()
    if isinstance(objeto, Path):
        return str(objeto)
    raise TypeError(f"tipo no serializable en JSON: {type(objeto)}")


def escribir_json(ruta: Path, carga: dict[str, Any]) -> None:
    """Escribe un JSON ASCII con indentacion.

    Args:
        ruta: destino.
        carga: contenido serializable.
    """
    texto = json.dumps(carga, ensure_ascii=True, indent=2, default=_json_default)
    verificar_ascii(texto, f"JSON {ruta.name}")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto + "\n", encoding="ascii")


def escribir_texto(ruta: Path, texto: str) -> None:
    """Escribe un informe de texto ASCII.

    Args:
        ruta: destino.
        texto: contenido ASCII.
    """
    verificar_ascii(texto, f"informe {ruta.name}")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="ascii")


def verificar_sha256(ruta: Path, esperado: str, nombre: str) -> dict[str, Any]:
    """Verifica el sha256 de un artefacto congelado.

    Args:
        ruta: archivo a comprobar.
        esperado: sha256 congelado en el preregistro.
        nombre: etiqueta legible.

    Returns:
        Evidencia de la verificacion.

    Raises:
        SystemExit: si falta el archivo o el hash no coincide.
    """
    if not ruta.exists():
        raise SystemExit(f"STOP: falta el artefacto congelado {ruta}")
    real = sha256_of(ruta)
    if real != esperado:
        raise SystemExit(f"STOP: sha256 de {nombre} no coincide: {real} != {esperado}")
    return {
        "nombre": nombre,
        "ruta": str(ruta.relative_to(PROJECT_ROOT)),
        "sha256": real,
        "esperado": esperado,
        "coincide": True,
    }


def versiones_entorno() -> dict[str, str]:
    """Recoge las versiones relevantes del entorno.

    Returns:
        Diccionario con python, numpy, pandas, sklearn, scipy y device de computo.
    """
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
        "device": "cpu (liblinear; sin torch en esta corrida)",
    }


# ===== SECCION 1: Gate 0 (datos, particiones, contadores e identidad V03) =====


@dataclass
class DatosGate0:
    """Datos de desarrollo y particiones verificadas de Gate 0.

    Attributes:
        des: DataFrame de desarrollo (1483 filas).
        textos: textos alineados con des.
        y: etiquetas (1483, 3).
        ids: ids alineados con el OOF canonico.
        p_v03: probabilidades OOF de V03.
        pred_v03: predicciones de V03 con umbrales congelados.
        umbrales_v03: umbrales congelados [0.44, 0.51, 0.46].
        grupos_dev: grupos fusionados de desarrollo.
        pliegues: 5 pliegues (indices de entrenamiento, validacion).
        folde: numero de pliegue (1..5) por fila.
        evidencia: informe de Gate 0.
    """

    des: pd.DataFrame
    textos: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    p_v03: np.ndarray
    pred_v03: np.ndarray
    umbrales_v03: np.ndarray
    grupos_dev: np.ndarray
    pliegues: list[tuple[np.ndarray, np.ndarray]]
    folde: np.ndarray
    evidencia: dict[str, Any]


def cargar_gate0() -> DatosGate0:
    """Ejecuta Gate 0: hashes, particiones, contadores e identidad del OOF congelado.

    Returns:
        Datos y evidencia de Gate 0.

    Raises:
        SystemExit: si cualquier verificacion congelada falla.
    """
    hashes = [
        verificar_sha256(CONFIG["train_path"], HASH_TRAIN, "toxic_train.csv"),
        verificar_sha256(CONFIG["oof_v03_path"], HASH_OOF_V03, "oof_v03_final.npz"),
        verificar_sha256(CONFIG["grupos_path"], HASH_GRUPOS, "grupos_near_duplicados.csv"),
    ]

    train = read_train()
    if len(train) != FILAS_TRAIN:
        raise SystemExit(
            f"STOP: toxic_train.csv tiene {len(train)} filas, se esperaban {FILAS_TRAIN}"
        )
    grupos = grupos_fusionados(train)
    indice_dev, indice_holdout = holdout_v2(train, grupos)
    if len(indice_dev) != FILAS_DEV or len(indice_holdout) != FILAS_HOLDOUT:
        raise SystemExit(
            f"STOP: particion inesperada dev={len(indice_dev)} holdout={len(indice_holdout)}"
        )
    des = train.iloc[indice_dev].reset_index(drop=True)
    grupos_dev = grupos[indice_dev]

    oof = np.load(CONFIG["oof_v03_path"], allow_pickle=True)
    y = oof["y_true"].astype(np.int64)
    p_v03 = oof["proba"].astype(np.float64)
    ids_v03 = oof["ids"].astype(np.int64)
    umbrales_v03 = oof["umbrales"].astype(np.float64)
    if not np.array_equal(des["id"].to_numpy(dtype=np.int64), ids_v03):
        raise SystemExit("STOP: los ids de desarrollo no coinciden con el OOF canonico")
    y_dev = des[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    if not np.array_equal(y_dev, y):
        raise SystemExit("STOP: las etiquetas de desarrollo no coinciden con el OOF canonico")
    if not np.array_equal(umbrales_v03, UMBRALES_CONGELADOS):
        raise SystemExit(f"STOP: umbrales congelados inesperados: {umbrales_v03.tolist()}")
    pred_v03 = (p_v03 >= umbrales_v03[None, :]).astype(np.int64)
    if not np.array_equal(pred_v03, oof["pred"].astype(np.int64)):
        raise SystemExit("STOP: pred del npz no reproduce proba >= umbrales congelados")

    pliegues = pliegues_corregidos(des, grupos_dev)
    tamanos = [len(np.asarray(b)) for _, b in pliegues]
    if tamanos != TAMANOS_PLIEGUES:
        raise SystemExit(f"STOP: tamanos de pliegue inesperados: {tamanos}")
    folde = np.zeros(len(des), dtype=np.int64)
    for numero, (_, validacion) in enumerate(pliegues, start=1):
        folde[np.asarray(validacion)] = numero
    if not np.array_equal(folde, oof["pliegues"].astype(np.int64)):
        raise SystemExit("STOP: el numero de pliegue no coincide con el npz canonico")

    contadores = {
        "dev_filas_leidas": len(des),
        "holdout_filas_usadas": 0,
        "test_filas_usadas": 0,
        "toxic_test_abierto": False,
        "nota": "el holdout (248) se aparta por indice y no se usa; toxic_test.csv nunca se abre",
    }
    evidencia = {
        "hashes": hashes,
        "particiones": {
            "dev": len(des),
            "holdout": len(indice_holdout),
            "pliegues_tamanos": tamanos,
            "pliegues_sin_grupos_compartidos": True,
            "filas_cubiertas_una_vez": bool(
                (np.bincount(folde, minlength=6)[1:] == np.array(tamanos)).all()
            ),
        },
        "contadores": contadores,
        "oof_v03": {
            "fuente": str(oof["fuente"][0]) if "fuente" in oof else "",
            "corrected_cv_npz": float(oof["corrected_cv"][0]),
            "umbrales": umbrales_v03.tolist(),
            "f1_pliegues": [float(valor) for valor in np.asarray(oof["f1_pliegues"])],
            "f1_por_etiqueta": [float(valor) for valor in np.asarray(oof["f1_por_etiqueta"])],
        },
    }
    return DatosGate0(
        des=des,
        textos=des["text"].to_numpy(dtype=object),
        y=y,
        ids=ids_v03,
        p_v03=p_v03,
        pred_v03=pred_v03,
        umbrales_v03=umbrales_v03,
        grupos_dev=grupos_dev,
        pliegues=pliegues,
        folde=folde,
        evidencia=evidencia,
    )


def seleccion_pliegues(
    datos: DatosGate0, pliegues: Sequence[int]
) -> list[tuple[int, tuple[np.ndarray, np.ndarray]]]:
    """Empareja cada numero de pliegue solicitado con sus indices canonicos.

    Args:
        datos: datos de Gate 0.
        pliegues: numeros de pliegue (1..5) en orden.

    Returns:
        Lista de tuplas ``(numero, (indices_train, indices_validacion))``.

    Raises:
        SystemExit: si un numero de pliegue esta fuera de rango.
    """
    seleccion: list[tuple[int, tuple[np.ndarray, np.ndarray]]] = []
    for numero in pliegues:
        if numero < 1 or numero > len(datos.pliegues):
            raise SystemExit(f"STOP: pliegue fuera de rango: {numero}")
        seleccion.append((int(numero), datos.pliegues[numero - 1]))
    return seleccion


class RepresentacionTfidf:
    """TF-IDF union REP-05 ajustado siempre con el subconjunto indicado.

    Replica exacta del vectorizador de V03: word (1,2) min_df=2 sublinear +
    char_wb (3,5) min_df=2 sublinear, lowercase y sin strip_accents, en ese
    orden de concatenacion. Se ajusta SOLO con outer_train y ANTES de cualquier
    duplicacion de filas.
    """

    def __init__(self) -> None:
        """Construye los dos vectorizadores sin ajustar."""
        self.palabra = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True
        )
        self.caracter = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True
        )

    def ajustar_transformar(self, textos: np.ndarray) -> sparse.csr_matrix:
        """Ajusta el vocabulario y devuelve la matriz de los textos de ajuste.

        Se usa ``fit_transform`` para la matriz de entrenamiento, como V2/P71-10,
        y no ``fit`` + ``transform``: los dos caminos de sklearn no son
        bit-identicos (la diferencia se amplifica en liblinear hasta ~5e-12) y el
        control de identidad de V03 exige una tolerancia de 1e-12.

        Args:
            textos: textos de ajuste (nunca validacion ni test).

        Returns:
            Matriz CSR de entrenamiento.
        """
        matriz_palabra = self.palabra.fit_transform(textos)
        matriz_caracter = self.caracter.fit_transform(textos)
        return sparse.hstack([matriz_palabra, matriz_caracter], format="csr")

    def transformar(self, textos: np.ndarray) -> sparse.csr_matrix:
        """Transforma textos a la matriz dispersa concatenada.

        Args:
            textos: textos a transformar.

        Returns:
            Matriz CSR ``(n, dim_word + dim_char)``.
        """
        return sparse.hstack(
            [self.palabra.transform(textos), self.caracter.transform(textos)], format="csr"
        )

    def dimensiones(self) -> dict[str, int]:
        """Devuelve las dimensiones del vocabulario ajustado.

        Returns:
            Diccionario con dimensiones de word, char_wb y la union.
        """
        dim_word = len(self.palabra.vocabulary_)
        dim_char = len(self.caracter.vocabulary_)
        return {"word": dim_word, "char": dim_char, "union": dim_word + dim_char}


def matrices_pliegue(
    textos: np.ndarray, indice_train: np.ndarray, indice_val: np.ndarray
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict[str, int]]:
    """Ajusta la representacion con outer_train y transforma ambos subconjuntos.

    Args:
        textos: textos de desarrollo.
        indice_train: indices de outer_train.
        indice_val: indices de outer_val (nunca se usa para ajustar).

    Returns:
        Tupla ``(matriz_train, matriz_val, dimensiones)``.
    """
    representacion = RepresentacionTfidf()
    x_train = representacion.ajustar_transformar(textos[indice_train])
    x_val = representacion.transformar(textos[indice_val])
    return x_train, x_val, representacion.dimensiones()


# ===== SECCION 2: fase de control A (identidad V03) =====


def fase_control(
    datos: DatosGate0, seleccion: list[tuple[int, tuple[np.ndarray, np.ndarray]]]
) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    """Calcula el OOF del brazo A (control V03 exacto) por pliegue.

    Args:
        datos: datos de Gate 0.
        seleccion: pliegues a ejecutar emparejados con sus indices.

    Returns:
        Tupla ``(proba_A, detalle_por_pliegue)`` con proba de forma (1483, 3).
    """
    proba = np.zeros((len(datos.y), len(ETIQUETAS)), dtype=np.float64)
    detalle: dict[int, dict[str, Any]] = {}
    for numero, (idx_train, idx_val) in seleccion:
        inicio = time.perf_counter()
        a = np.asarray(idx_train)
        b = np.asarray(idx_val)
        x_train, x_val, dimensiones = matrices_pliegue(datos.textos, a, b)
        iteraciones: dict[str, int] = {}
        for j, etiqueta in enumerate(ETIQUETAS):
            modelo = LogisticRegression(**PARAMS_LR_CONTROL)
            modelo.fit(x_train, datos.y[a, j])
            proba[b, j] = modelo.predict_proba(x_val)[:, 1]
            iteraciones[etiqueta] = int(np.max(np.asarray(modelo.n_iter_)))
        detalle[numero] = {
            "pliegue": numero,
            "n_train": len(a),
            "n_val": len(b),
            "dimensiones": dimensiones,
            "n_iter": iteraciones,
            "segundos": round(time.perf_counter() - inicio, 3),
        }
        log(
            f"control A pliegue {numero}: train {len(a)} val {len(b)} "
            f"dim {dimensiones['union']} en {detalle[numero]['segundos']:.1f}s"
        )
    return proba, detalle


def control_identidad_v03(
    datos: DatosGate0,
    proba_A: np.ndarray,
    seleccion: list[tuple[int, tuple[np.ndarray, np.ndarray]]],
) -> dict[str, Any]:
    """Verifica que el brazo A reproduce el OOF canonico de V03.

    Para corridas parciales (o humo) la comprobacion se restringe a las filas de
    validacion de los pliegues ejecutados y se marca ``completa=False``; en la
    corrida completa exige ademas que el corrected_cv global coincida con
    0.7772114525165721 y que V03@0.5 sea 0.771778476256122 (tolerancia 1e-12).

    Args:
        datos: datos de Gate 0.
        proba_A: OOF reconstruido por el brazo A.
        seleccion: pliegues ejecutados.

    Returns:
        Evidencia de la identidad con tolerancia 1e-12.

    Raises:
        SystemExit: si alguna referencia no se reproduce.
    """
    cubiertas = np.concatenate([np.asarray(b) for _, (_, b) in seleccion])
    max_delta = float(np.max(np.abs(proba_A[cubiertas] - datos.p_v03[cubiertas])))
    pred_rec = (proba_A[cubiertas] >= datos.umbrales_v03[None, :]).astype(np.int64)
    pred_npz = datos.pred_v03[cubiertas]
    celdas_distintas = int(np.sum(pred_rec != pred_npz))
    f1_pliegues_rec: list[float] = []
    for _, (_, b) in seleccion:
        indice = np.asarray(b)
        f1_pliegues_rec.append(
            float(
                metrica_macro(
                    datos.y[indice], (proba_A[indice] >= datos.umbrales_v03[None, :]).astype(np.int64)
                )
            )
        )
    oof = np.load(CONFIG["oof_v03_path"], allow_pickle=True)
    f1_pliegues_npz = [float(valor) for valor in np.asarray(oof["f1_pliegues"])]
    f1_at_05 = float(metrica_macro(datos.y, (datos.p_v03 >= 0.5).astype(np.int64)))
    completa = len(seleccion) == N_PLIEGUES
    ccv_reconstruido: float | None = None
    delta_ccv: float | None = None
    if completa:
        pred_frozen = (proba_A >= datos.umbrales_v03[None, :]).astype(np.int64)
        ccv_reconstruido = float(metrica_macro(datos.y, pred_frozen))
        delta_ccv = abs(ccv_reconstruido - V03_CCV)

    fallos: list[str] = []
    if celdas_distintas:
        fallos.append(
            f"las predicciones congeladas difieren del npz en {celdas_distintas} celdas"
        )
    if max_delta > TOL_IDENTIDAD:
        fallos.append(f"max|delta proba| = {max_delta:g} > {TOL_IDENTIDAD:g}")
    for numero, valor_rec in zip([num for num, _ in seleccion], f1_pliegues_rec):
        if abs(valor_rec - f1_pliegues_npz[numero - 1]) > TOL_IDENTIDAD:
            fallos.append(
                f"F1 del pliegue {numero} no reproducible: {valor_rec!r} vs "
                f"{f1_pliegues_npz[numero - 1]!r}"
            )
    if abs(f1_at_05 - V03_F1_AT_05) > TOL_IDENTIDAD:
        fallos.append(f"V03@0.5 no reproducible: {f1_at_05!r} vs {V03_F1_AT_05!r}")
    if completa and delta_ccv is not None and delta_ccv > TOL_IDENTIDAD:
        fallos.append(
            f"corrected_cv no reproducible: {ccv_reconstruido!r} vs {V03_CCV!r}"
        )
    if fallos:
        raise SystemExit("STOP: control de identidad V03 fallido: " + "; ".join(fallos))

    return {
        "completa": completa,
        "pliegues_evaluados": [numero for numero, _ in seleccion],
        "max_delta_proba_vs_npz": max_delta,
        "tolerancia": TOL_IDENTIDAD,
        "celdas_de_prediccion_distintas": celdas_distintas,
        "f1_pliegues_reconstruidos": [round(valor, 12) for valor in f1_pliegues_rec],
        "f1_pliegues_npz": f1_pliegues_npz,
        "f1_at_05_recalculado": round(f1_at_05, 12),
        "f1_at_05_esperado": V03_F1_AT_05,
        "corrected_cv_reconstruido": None if ccv_reconstruido is None else round(ccv_reconstruido, 12),
        "corrected_cv_esperado": V03_CCV,
        "delta_corrected_cv": delta_ccv,
        "identidad_ok": True,
    }


# ===== SECCION 3: duplicacion determinista y brazos ROS =====


def construir_recuentos(n_pos: int, n_prima: int, semilla: int) -> np.ndarray:
    """Asigna cuantas copias recibe cada positivo con una permutacion sembrada.

    La copia k-esima elige el indice ``permutacion[k % n_pos]``; por tanto cada
    positivo recibe ``q`` o ``q + 1`` copias, con ``q = n_prima // n_pos``. Si
    ``n_prima < n_pos`` la asignacion es un subconjunto determinista y algunos
    positivos reciben cero copias (se reporta). El generador se crea fresco por
    (pliegue, etiqueta), de modo que la asignacion no depende del orden de
    ejecucion ni del subconjunto de pliegues solicitado.

    Nota de redondeo: ``n_prima`` se calcula aguas arriba como ``round(r * n_neg)``
    con el ``round`` half-to-even (bancario) de Python; con r=0.5 (brazo B2) el
    valor realizado depende de la paridad de ``n_neg``.

    Args:
        n_pos: numero de positivos originales en outer_train.
        n_prima: numero total de copias positivas objetivo (n_pos').
        semilla: semilla del generador (20260926 en esta fase).

    Returns:
        Arreglo entero de longitud n_pos con las copias por positivo.

    Raises:
        SystemExit: si no hay positivos o el objetivo de copias es menor que 1.
    """
    if n_pos <= 0:
        raise SystemExit("STOP: no hay positivos en outer_train para duplicar")
    if n_prima < 1:
        raise SystemExit("STOP: n_pos' < 1 no permite entrenar el brazo ROS")
    generador = np.random.default_rng(semilla)
    permutacion = generador.permutation(n_pos)
    ciclos, resto = divmod(n_prima, n_pos)
    recuentos = np.full(n_pos, ciclos, dtype=np.int64)
    recuentos[permutacion[:resto]] += 1
    return recuentos


def registro_etiqueta(
    pliegue: int,
    etiqueta: str,
    n_pos: int,
    n_neg: int,
    recuentos: np.ndarray,
    peso_uniforme: float | None,
    c_nominal: float,
) -> dict[str, Any]:
    """Construye el registro auditable de un brazo ROS para (pliegue, etiqueta).

    Args:
        pliegue: numero de pliegue.
        etiqueta: nombre de la etiqueta.
        n_pos: positivos originales en outer_train.
        n_neg: negativos en outer_train.
        recuentos: copias por positivo devueltas por :func:`construir_recuentos`.
        peso_uniforme: peso comun de las filas (None si el brazo no normaliza).
        c_nominal: C nominal del brazo (2.0).

    Returns:
        Diccionario con recuentos, pesos, masa relativa y C efectivo.
    """
    n = n_pos + n_neg
    n_prima = int(recuentos.sum())
    peso = 1.0 if peso_uniforme is None else float(peso_uniforme)
    pesos_efectivos = recuentos.astype(np.float64) * peso
    s_pos_balanced = n / (2.0 * n_pos)
    desviacion = float(np.max(np.abs(pesos_efectivos - s_pos_balanced) / s_pos_balanced))
    ciclos, resto = divmod(n_prima, n_pos)
    masa_relativa = float((n_prima + n_neg) * peso / n)
    return {
        "pliegue": pliegue,
        "etiqueta": etiqueta,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n": n,
        "n_prima": n_prima,
        "ciclos_q": ciclos,
        "resto": resto,
        "recuento_min": int(recuentos.min()),
        "recuento_max": int(recuentos.max()),
        "recuento_medio": round(float(recuentos.mean()), 10),
        "recuentos_distintos": int(np.unique(recuentos).size),
        "n_pos_sin_copia": int(np.sum(recuentos == 0)),
        "n_pos_con_copia_extra": int(np.sum(recuentos == ciclos + 1)),
        "peso_uniforme": None if peso_uniforme is None else round(peso, 10),
        "masa_total_relativa_n": round(masa_relativa, 10),
        "c_efectivo": round(c_nominal * masa_relativa, 6),
        "razon_peso_medio_pos_neg": round(n_prima / n_pos, 10),
        # n_pos'/n_neg: factor realizado de la razon media de pesos pos/neg frente a
        # class_weight='balanced' (no es la razon de pesos en si, que es
        # razon_peso_medio_pos_neg = n_pos'/n_pos; en B3 ademas pesa la masa total).
        "factor_razon_pesos_vs_balanced": round(n_prima / n_neg, 10),
        "desviacion_peso_max_rel_vs_balanced": round(desviacion, 6),
    }


def fase_ros(
    datos: DatosGate0, seleccion: list[tuple[int, tuple[np.ndarray, np.ndarray]]]
) -> tuple[dict[str, np.ndarray], dict[int, dict[str, Any]]]:
    """Calcula el OOF de los brazos B1, B2 y B3 por pliegue.

    La representacion se ajusta SOLO con outer_train y ANTES de duplicar; la
    duplicacion es fisica (filas repetidas) y ocurre unicamente en el ajuste del
    clasificador. La validacion nunca se duplica.

    Args:
        datos: datos de Gate 0.
        seleccion: pliegues a ejecutar emparejados con sus indices.

    Returns:
        Tupla ``(probas_por_brazo, detalle_por_pliegue)``.
    """
    probas: dict[str, np.ndarray] = {
        brazo: np.zeros((len(datos.y), len(ETIQUETAS)), dtype=np.float64)
        for brazo in BRAZOS_ROS
    }
    detalle: dict[int, dict[str, Any]] = {}
    for numero, (idx_train, idx_val) in seleccion:
        inicio = time.perf_counter()
        a = np.asarray(idx_train)
        b = np.asarray(idx_val)
        x_train, x_val, dimensiones = matrices_pliegue(datos.textos, a, b)
        registro_fold: dict[str, Any] = {
            "pliegue": numero,
            "n_train": len(a),
            "n_val": len(b),
            "dimensiones": dimensiones,
            "etiquetas": {},
            "n_iter": {},
            "f1_frozen": {},
            "f1_05": {},
        }
        for j, etiqueta in enumerate(ETIQUETAS):
            y_columna = datos.y[a, j]
            pos_local = np.flatnonzero(y_columna == 1)
            neg_local = np.flatnonzero(y_columna == 0)
            n_pos = len(pos_local)
            n_neg = len(neg_local)
            n = n_pos + n_neg
            registros_etiqueta: dict[str, Any] = {}
            for brazo in BRAZOS_ROS:
                especificacion = BRAZOS[brazo]
                r = float(especificacion["r"])
                normalizado = bool(especificacion["normalizado"])
                # round() de Python es half-to-even: con r=0.5 (B2) n_pos' depende de la
                # paridad de n_neg (n_neg impar => el empate .5 cae al entero par).
                n_prima = round(r * n_neg)
                recuentos = construir_recuentos(n_pos, n_prima, SEMILLA_DUPLICACION)
                filas = np.concatenate([neg_local, np.repeat(pos_local, recuentos)])
                y_fit = y_columna[filas]
                x_fit = x_train[filas]
                peso = n / (n_prima + n_neg) if normalizado else None
                modelo = LogisticRegression(**PARAMS_LR_ROS)
                if peso is None:
                    modelo.fit(x_fit, y_fit)
                else:
                    modelo.fit(
                        x_fit, y_fit, sample_weight=np.full(len(filas), peso, dtype=np.float64)
                    )
                probas[brazo][b, j] = modelo.predict_proba(x_val)[:, 1]
                registro_fold["n_iter"][f"{brazo}_{etiqueta}"] = int(
                    np.max(np.asarray(modelo.n_iter_))
                )
                registros_etiqueta[brazo] = registro_etiqueta(
                    numero,
                    etiqueta,
                    n_pos,
                    n_neg,
                    recuentos,
                    peso,
                    float(PARAMS_LR_ROS["C"]),
                )
            registro_fold["etiquetas"][etiqueta] = registros_etiqueta
        for brazo in BRAZOS_ROS:
            prediccion_frozen = (probas[brazo][b] >= datos.umbrales_v03[None, :]).astype(
                np.int64
            )
            registro_fold["f1_frozen"][brazo] = round(
                float(metrica_macro(datos.y[b], prediccion_frozen)), 8
            )
            registro_fold["f1_05"][brazo] = round(
                float(metrica_macro(datos.y[b], (probas[brazo][b] >= 0.5).astype(np.int64))),
                8,
            )
        registro_fold["segundos"] = round(time.perf_counter() - inicio, 3)
        detalle[numero] = registro_fold
        log(
            "brazos ROS pliegue "
            f"{numero}: dim {dimensiones['union']} en {registro_fold['segundos']:.1f}s | "
            f"F1@congelado B1 {registro_fold['f1_frozen']['B1']:.4f} "
            f"B2 {registro_fold['f1_frozen']['B2']:.4f} "
            f"B3 {registro_fold['f1_frozen']['B3']:.4f}"
        )
    return probas, detalle


# ===== SECCION 4: metricas, comparaciones y equivalencia =====


def metricas_de_lectura(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """Calcula las metricas publicadas de una lectura de umbral.

    Args:
        y: etiquetas verdaderas (n, 3).
        pred: predicciones binarias (n, 3).

    Returns:
        Diccionario con metricas primarias, de diagnostico y por etiqueta.
    """
    metricas = dict(metricas_multietiqueta(y, pred))
    metricas["precision_micro"] = float(
        precision_score(y, pred, average="micro", zero_division=0)
    )
    metricas["recall_micro"] = float(recall_score(y, pred, average="micro", zero_division=0))
    metricas["precision_por_etiqueta"] = {
        etiqueta: float(precision_score(y[:, j], pred[:, j], zero_division=0))
        for j, etiqueta in enumerate(ETIQUETAS)
    }
    metricas["recall_por_etiqueta"] = {
        etiqueta: float(recall_score(y[:, j], pred[:, j], zero_division=0))
        for j, etiqueta in enumerate(ETIQUETAS)
    }
    metricas["tasa_positivos"] = [float(valor) for valor in pred.mean(axis=0)]
    return metricas


def comparar_predicciones(
    datos: DatosGate0,
    pred: np.ndarray,
    pred_ref: np.ndarray,
    seleccion: list[tuple[int, tuple[np.ndarray, np.ndarray]]],
) -> dict[str, Any]:
    """Compara pareadamente una lectura contra una referencia con el protocolo comun.

    Args:
        datos: datos de Gate 0.
        pred: predicciones del candidato.
        pred_ref: predicciones de la referencia.
        seleccion: pliegues ejecutados (se usan sus validaciones para el delta por pliegue).

    Returns:
        Diccionario con delta global, deltas por pliegue, pliegues mejorados,
        rescate/dano, celdas distintas frente a la referencia (con el booleano
        ``predicciones_identicas`` derivado de ellas), tasas de positivos y F1
        por pliegue.
    """
    cubiertas = np.concatenate([np.asarray(b) for _, (_, b) in seleccion])
    f1_global = float(metrica_macro(datos.y[cubiertas], pred[cubiertas]))
    f1_ref = float(metrica_macro(datos.y[cubiertas], pred_ref[cubiertas]))
    deltas: list[float] = []
    f1_pliegues: list[float] = []
    f1_pliegues_ref: list[float] = []
    for _, (_, b) in seleccion:
        indice = np.asarray(b)
        valor = float(metrica_macro(datos.y[indice], pred[indice]))
        valor_ref = float(metrica_macro(datos.y[indice], pred_ref[indice]))
        f1_pliegues.append(round(valor, 8))
        f1_pliegues_ref.append(round(valor_ref, 8))
        deltas.append(round(valor - valor_ref, 8))
    correctas_ref = pred_ref[cubiertas] == datos.y[cubiertas]
    correctas_cand = pred[cubiertas] == datos.y[cubiertas]
    rescate = int(np.sum(~correctas_ref & correctas_cand))
    dano = int(np.sum(correctas_ref & ~correctas_cand))
    celdas_distintas = int(np.sum(pred[cubiertas] != pred_ref[cubiertas]))
    celdas_totales = int(datos.y[cubiertas].size)
    return {
        "f1_global": round(f1_global, 10),
        "f1_referencia_global": round(f1_ref, 10),
        "delta_global": round(f1_global - f1_ref, 10),
        "delta_medio_pliegues": round(float(np.mean(deltas)), 10),
        "delta_por_pliegue": deltas,
        "pliegues_mejorados": int(np.sum(np.array(deltas) > 0)),
        "f1_por_pliegue": f1_pliegues,
        "f1_por_pliegue_referencia": f1_pliegues_ref,
        "rescate_celdas": rescate,
        "dano_celdas": dano,
        "neto_celdas": rescate - dano,
        "celdas_distintas": celdas_distintas,
        "celdas_totales": celdas_totales,
        "predicciones_identicas": bool(celdas_distintas == 0),
        "tasa_positivos": [float(valor) for valor in pred[cubiertas].mean(axis=0)],
        "tasa_positivos_referencia": [
            float(valor) for valor in pred_ref[cubiertas].mean(axis=0)
        ],
    }


def bootstrap_comparacion(
    datos: DatosGate0, pred: np.ndarray, pred_ref: np.ndarray
) -> dict[str, Any]:
    """Calcula el IC95 bootstrap pareado de una comparacion (protocolo comun).

    Args:
        datos: datos de Gate 0.
        pred: predicciones del candidato.
        pred_ref: predicciones de la referencia.

    Returns:
        Diccionario de :func:`src.evaluation.bootstrap_diferencia_pareada` con
        repeticiones y semilla explicitas.
    """
    return bootstrap_diferencia_pareada(
        datos.y,
        pred,
        pred_ref,
        repeticiones=REPETICIONES_BOOTSTRAP,
        semilla=SEMILLA_BOOTSTRAP,
    )


def registro_comparacion(
    datos: DatosGate0,
    pred_cand: np.ndarray,
    pred_ref: np.ndarray,
    seleccion: list[tuple[int, tuple[np.ndarray, np.ndarray]]],
    con_bootstrap: bool,
) -> dict[str, Any]:
    """Empaqueta una comparacion pareada y anade el bootstrap si corresponde.

    Args:
        datos: datos de Gate 0.
        pred_cand: predicciones del candidato.
        pred_ref: predicciones de la referencia (nunca el candidato).
        seleccion: pliegues ejecutados.
        con_bootstrap: si True anade el IC95 bootstrap (solo corrida completa,
            porque el bootstrap pareado se calcula sobre las 1483 filas).

    Returns:
        Diccionario con ``comparacion`` (:func:`comparar_predicciones`) y, si
        ``con_bootstrap``, ``bootstrap`` (:func:`bootstrap_comparacion`).
    """
    registro: dict[str, Any] = {
        "comparacion": comparar_predicciones(datos, pred_cand, pred_ref, seleccion)
    }
    if con_bootstrap:
        registro["bootstrap"] = bootstrap_comparacion(datos, pred_cand, pred_ref)
    return registro


def diagnostico_causas(
    detalle_ros: dict[int, dict[str, Any]], completa: bool
) -> dict[str, Any]:
    """Resume la evidencia de redondeo entero y de trayectoria del solver.

    Se usa solo si la equivalencia B1-A no se cumple, para atribuir la causa con
    la evidencia disponible en el harness: dispersion de los recuentos enteros
    (pesos efectivos q/q+1 frente al peso balanced) y convergencia de liblinear.

    Args:
        detalle_ros: detalle por pliegue de la fase ROS.
        completa: si la corrida cubrio los 5 pliegues.

    Returns:
        Diccionario con la evidencia y la causa principal observada.
    """
    desviaciones: list[float] = []
    recuentos_no_uniformes = 0
    con_cero = 0
    for registro in detalle_ros.values():
        for registros_etiqueta in registro["etiquetas"].values():
            recuento_b1 = registros_etiqueta["B1"]
            desviaciones.append(recuento_b1["desviacion_peso_max_rel_vs_balanced"])
            if recuento_b1["recuentos_distintos"] > 1:
                recuentos_no_uniformes += 1
            if recuento_b1["n_pos_sin_copia"] > 0:
                con_cero += 1
    max_iter_observadas = [
        valor
        for registro in detalle_ros.values()
        for clave, valor in registro["n_iter"].items()
        if clave.startswith("B1_")
    ]
    no_convergencias = sum(1 for valor in max_iter_observadas if valor >= MAX_ITER_LIBLINEAR)
    evidencia_redondeo = bool(recuentos_no_uniformes or con_cero)
    evidencia_solver = bool(no_convergencias)
    if evidencia_redondeo and evidencia_solver:
        causa = "redondeo_entero + trayectoria_solver (no convergencia de liblinear)"
    elif evidencia_redondeo:
        causa = "redondeo_entero (recuentos q/q+1 no uniformes)"
    elif evidencia_solver:
        causa = "trayectoria_solver (no convergencia de liblinear)"
    else:
        causa = "no determinada con la evidencia del harness (otra fuente numerica)"
    return {
        "completa": completa,
        "max_desviacion_peso_rel_vs_balanced": round(float(max(desviaciones, default=0.0)), 6),
        "etiquetas_con_recuentos_no_uniformes": recuentos_no_uniformes,
        "etiquetas_con_positivos_sin_copia": con_cero,
        "max_iter_liblinear_b1": int(max(max_iter_observadas, default=0)),
        "max_iter_permitido": MAX_ITER_LIBLINEAR,
        "ajustes_b1_sin_converger": no_convergencias,
        "causa_principal": causa,
    }


def evaluar_equivalencia(
    datos: DatosGate0,
    probas: dict[str, np.ndarray],
    seleccion: list[tuple[int, tuple[np.ndarray, np.ndarray]]],
    detalle_ros: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Evalua el criterio pre-declarado de equivalencia B1-A.

    Criterio: ``max|proba_B1 - proba_A| <= 1e-3`` y ``|delta_F1| <= 0.001`` en
    la lectura primaria (umbrales congelados). En corridas parciales el veredicto
    es provisional; con los 5 pliegues es definitivo y, si se refuta, activa el
    STOP de promocion con el diagnostico de causas.

    Args:
        datos: datos de Gate 0.
        probas: probabilidades OOF por brazo.
        seleccion: pliegues ejecutados.
        detalle_ros: detalle por pliegue de la fase ROS.

    Returns:
        Diccionario con diferencias maximas, deltas de F1, veredicto y causa.
    """
    cubiertas = np.concatenate([np.asarray(b) for _, (_, b) in seleccion])
    diferencia = np.abs(probas["B1"] - probas["A"])
    max_global = float(np.max(diferencia[cubiertas]))
    por_pliegue = {
        str(numero): float(np.max(diferencia[np.asarray(b)]))
        for numero, (_, b) in seleccion
    }
    por_etiqueta = {
        etiqueta: float(np.max(diferencia[cubiertas][:, j]))
        for j, etiqueta in enumerate(ETIQUETAS)
    }
    y_cubiertas = datos.y[cubiertas]
    f1_a_frozen = float(
        metrica_macro(
            y_cubiertas,
            (probas["A"][cubiertas] >= datos.umbrales_v03[None, :]).astype(np.int64),
        )
    )
    f1_b1_frozen = float(
        metrica_macro(
            y_cubiertas,
            (probas["B1"][cubiertas] >= datos.umbrales_v03[None, :]).astype(np.int64),
        )
    )
    f1_a_05 = float(metrica_macro(y_cubiertas, (probas["A"][cubiertas] >= 0.5).astype(np.int64)))
    f1_b1_05 = float(
        metrica_macro(y_cubiertas, (probas["B1"][cubiertas] >= 0.5).astype(np.int64))
    )
    delta_frozen = f1_b1_frozen - f1_a_frozen
    delta_05 = f1_b1_05 - f1_a_05
    pasa_proba = bool(max_global <= TOL_EQUIV_PROBA)
    pasa_f1 = bool(abs(delta_frozen) <= TOL_EQUIV_F1)
    completa = len(seleccion) == N_PLIEGUES
    pasa = bool(pasa_proba and pasa_f1) if completa else None
    if not completa:
        veredicto = "PROVISIONAL (corrida parcial o humo)"
    elif pasa:
        veredicto = "EQUIVALENCIA CONFIRMADA"
    else:
        veredicto = "REFUTACION PARCIAL"
    resultado: dict[str, Any] = {
        "completa": completa,
        "lectura_primaria": "umbrales congelados 0.44/0.51/0.46",
        "max_delta_proba_global": round(max_global, 12),
        "max_delta_proba_por_pliegue": por_pliegue,
        "max_delta_proba_por_etiqueta": por_etiqueta,
        "delta_f1_frozen": round(delta_frozen, 10),
        "delta_f1_05": round(delta_05, 10),
        "tolerancia_proba": TOL_EQUIV_PROBA,
        "tolerancia_f1": TOL_EQUIV_F1,
        "pasa_proba": pasa_proba,
        "pasa_f1": pasa_f1,
        "pasa": pasa,
        "veredicto": veredicto,
    }
    if not completa or not pasa:
        resultado["diagnostico_causas"] = diagnostico_causas(detalle_ros, completa)
    return resultado


# ===== SECCION 5: salidas (parciales, npz, informes y CSV) =====


def escribir_parcial(
    numero: int,
    detalle_control: dict[str, Any],
    detalle_ros: dict[str, Any],
    humo: bool,
) -> Path:
    """Escribe el parcial JSON de un pliegue con detalle de control y ROS.

    Args:
        numero: numero de pliegue.
        detalle_control: detalle del brazo A en el pliegue.
        detalle_ros: detalle de los brazos ROS en el pliegue.
        humo: si True el parcial lleva el sufijo _humo.

    Returns:
        Ruta escrita.
    """
    directorio: Path = CONFIG["parciales_dir"]
    directorio.mkdir(parents=True, exist_ok=True)
    sufijo = "_humo" if humo else ""
    ruta = directorio / f"p84_03_p{numero}{sufijo}.json"
    carga = {
        "experiment_id": "P84-03",
        "modo_humo": humo,
        "pliegue": numero,
        "control": detalle_control,
        "ros": detalle_ros,
    }
    escribir_json(ruta, carga)
    return ruta


def guardar_oof_npz(datos: DatosGate0, probas: dict[str, np.ndarray]) -> Path:
    """Guarda el npz OOF con las probabilidades por brazo.

    Args:
        datos: datos de Gate 0.
        probas: probabilidades OOF por brazo (1483, 3).

    Returns:
        Ruta escrita.
    """
    ruta: Path = CONFIG["oof_npz"]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ruta,
        proba_A=probas["A"],
        proba_B1=probas["B1"],
        proba_B2=probas["B2"],
        proba_B3=probas["B3"],
        pliegues=datos.folde.astype(np.int64),
        y_true=datos.y.astype(np.int64),
        ids=datos.ids.astype(np.int64),
        umbrales=datos.umbrales_v03.astype(np.float64),
        etiquetas=np.array(ETIQUETAS),
    )
    return ruta


def upsert_experiments_csv(fila: dict[str, str]) -> dict[str, Any]:
    """Inserta o reemplaza la fila P84-03 en experiments.csv (23 columnas).

    Valida la cabecera, el ancho de todas las filas, la ausencia de ids
    duplicados y que la fila defina exactamente las 23 columnas antes de
    escribir. Escribe utf-8-sig y no duplica ids.

    Args:
        fila: valores por columna de la fila P84-03.

    Returns:
        Evidencia del upsert.

    Raises:
        SystemExit: si la estructura del CSV no es la esperada.
    """
    ruta: Path = CONFIG["experiments_csv"]
    if not ruta.exists():
        raise SystemExit(f"STOP: no existe {ruta}")
    with ruta.open(encoding="utf-8-sig", newline="") as manejador:
        filas = list(csv.reader(manejador))
    if not filas:
        raise SystemExit("STOP: experiments.csv esta vacio")
    cabecera = [columna.strip() for columna in filas[0]]
    if tuple(cabecera) != COLUMNAS_EXPERIMENTS:
        raise SystemExit(f"STOP: cabecera inesperada en experiments.csv: {cabecera}")
    cuerpo = filas[1:]
    ids = [registro[0] for registro in cuerpo if registro]
    if len(ids) != len(set(ids)):
        raise SystemExit("STOP: experiments.csv tiene ids duplicados")
    for numero, registro in enumerate(cuerpo, start=2):
        if len(registro) != len(COLUMNAS_EXPERIMENTS):
            raise SystemExit(
                f"STOP: la fila {numero} de experiments.csv tiene {len(registro)} columnas"
            )
    faltantes = [columna for columna in COLUMNAS_EXPERIMENTS if columna not in fila]
    if faltantes:
        raise SystemExit(f"STOP: faltan columnas en la fila P84-03: {faltantes}")
    sobrantes = sorted(set(fila) - set(COLUMNAS_EXPERIMENTS))
    if sobrantes:
        raise SystemExit(f"STOP: columnas desconocidas en la fila P84-03: {sobrantes}")
    nueva = [fila.get(columna, "") for columna in COLUMNAS_EXPERIMENTS]
    identificador = nueva[0]
    if identificador in ids:
        cuerpo[ids.index(identificador)] = nueva
        accion = "reemplazada"
    else:
        cuerpo.append(nueva)
        accion = "anadida"
    temporal = ruta.with_name(ruta.name + ".tmp")
    with temporal.open("w", encoding="utf-8-sig", newline="") as manejador:
        escritor = csv.writer(manejador, lineterminator="\n")
        escritor.writerow(COLUMNAS_EXPERIMENTS)
        escritor.writerows(cuerpo)
    os.replace(temporal, ruta)
    return {"accion": accion, "filas": len(cuerpo), "columnas": len(COLUMNAS_EXPERIMENTS)}


def construir_fila_experiments(
    carga: dict[str, Any],
    datos: DatosGate0,
    probas: dict[str, np.ndarray],
    comparaciones: dict[str, Any],
    equivalencia: dict[str, Any],
    segundos_totales: float,
) -> dict[str, str]:
    """Construye la fila P84-03 para experiments.csv.

    Args:
        carga: informe completo de la corrida.
        datos: datos de Gate 0.
        probas: probabilidades OOF por brazo.
        comparaciones: comparaciones calculadas.
        equivalencia: resultado de la equivalencia B1-A.
        segundos_totales: tiempo total de la corrida.

    Returns:
        Diccionario columna -> valor.
    """
    metricas_b1 = metricas_de_lectura(
        datos.y, (probas["B1"] >= datos.umbrales_v03[None, :]).astype(np.int64)
    )
    comparacion_b1 = comparaciones["B1"]["vs_A_frozen"]["comparacion"]
    resumen_equivalencia = {
        "max_delta_proba_B1_A": equivalencia["max_delta_proba_global"],
        "delta_f1_B1_A_frozen": equivalencia["delta_f1_frozen"],
        "pasa": equivalencia["pasa"],
        "veredicto": equivalencia["veredicto"],
    }
    resumen_brazos = {
        brazo: {
            "f1_frozen": comparaciones[brazo]["vs_A_frozen"]["comparacion"]["f1_global"],
            "delta_global_frozen_vs_A": comparaciones[brazo]["vs_A_frozen"]["comparacion"][
                "delta_global"
            ],
            "delta_medio_pliegues_vs_A": comparaciones[brazo]["vs_A_frozen"]["comparacion"][
                "delta_medio_pliegues"
            ],
            "ic95_vs_A": comparaciones[brazo]["vs_A_frozen"]["bootstrap"]["ic95"],
            "delta_global_frozen_vs_V03": comparaciones[brazo]["vs_V03_frozen"][
                "comparacion"
            ]["delta_global"],
            "rescate_celdas": comparaciones[brazo]["vs_V03_frozen"]["comparacion"][
                "rescate_celdas"
            ],
            "dano_celdas": comparaciones[brazo]["vs_V03_frozen"]["comparacion"]["dano_celdas"],
            "celdas_distintas_vs_V03": comparaciones[brazo]["vs_V03_frozen"]["comparacion"][
                "celdas_distintas"
            ],
            "predicciones_identicas_vs_V03": comparaciones[brazo]["vs_V03_frozen"][
                "comparacion"
            ]["predicciones_identicas"],
            "tasa_positivos": comparaciones[brazo]["vs_V03_frozen"]["comparacion"][
                "tasa_positivos"
            ],
        }
        for brazo in BRAZOS_ROS
    }
    resultado = {
        "micro_verificacion": "equivalencia ROS vs class_weight/reweighting",
        "equivalencia_B1_A": resumen_equivalencia,
        "brazos": resumen_brazos,
        "c_efectivo_B3_por_pliegue": carga["resumen_c_efectivo_B3"],
        "gate2": False,
        "exp06": False,
        "promocion": "ninguna",
    }
    if equivalencia["pasa"] is False:
        resultado["etiqueta_cierre"] = "CLOSED BY EQUIVALENCE"
    return {
        "experiment_id": "P84-03",
        "competition": str(CONFIG["competencia"]),
        "question": (
            "Aporta el sobremuestreo aleatorio (ROS) de positivos algo distinto de "
            "class_weight=balanced / reweighting en un modelo lineal con L2 (liblinear), o es "
            "solo una reasignacion de pesos por instancia? B1 (r=1.0 normalizado) debe "
            "reproducir al control V03 (tolerancia 1e-3 en probabilidades y 0.001 en F1)."
        ),
        "hypothesis": (
            "Pre-registrada: para un modelo lineal convexo con L2 la duplicacion de filas es "
            "exactamente una reasignacion de pesos; B1 debe ser equivalente a A salvo redondeo "
            "entero y trayectoria del solver; B2 y B3 caen en familias cerradas (class_weight "
            "V4-C04 y capacidad C); ROS no es promovible."
        ),
        "preprocessing": (
            "texto crudo; TF-IDF word(1,2)+char_wb(3,5) min_df=2 sublinear ajustado SOLO con "
            "outer_train y ANTES de duplicar; duplicacion fisica de positivos solo en el ajuste "
            "del clasificador; la validacion nunca se duplica ni se usa para elegir copias"
        ),
        "feature_groups": "WORD_NGRAM+CHAR_NGRAM",
        "representation": (
            "TF-IDF union word(1,2)+char_wb(3,5), min_df=2, sublinear_tf, lowercase, "
            "sin strip_accents"
        ),
        "model": (
            "LogisticRegression C=2.0 liblinear rs=42 (A: class_weight=balanced; "
            "B1/B2/B3: ROS determinista con y sin normalizacion de masa)"
        ),
        "hyperparameters": json.dumps(
            {
                "lr_A": PARAMS_LR_CONTROL,
                "lr_ros": PARAMS_LR_ROS,
                "brazos": {brazo: BRAZOS[brazo] for brazo in BRAZOS_ROS},
                "duplicacion": {
                    "semilla": SEMILLA_DUPLICACION,
                    "regla": "n_pos' = round(r*n_neg); copias sobre permutacion sembrada",
                    "peso_normalizado": "n/(n_pos'+n_neg)",
                },
            },
            ensure_ascii=True,
            default=_json_default,
        ),
        "validation": (
            "pliegues_corregidos de 5 pliegues sobre desarrollo (1483, grupos fusionados); "
            "lectura primaria con umbrales congelados 0.44/0.51/0.46 y diagnostico a 0.5; "
            "bootstrap pareado 10000 semilla 20260926; sin optimizacion de umbral; "
            "sin holdout ni test"
        ),
        "f1": f"{comparacion_b1['f1_global']:.6f}",
        "macro_f1": f"{metricas_b1['f1_macro6']:.6f}",
        "weighted_f1": "",
        "precision": f"{metricas_b1['precision_micro']:.6f}",
        "recall": f"{metricas_b1['recall_micro']:.6f}",
        "training_time": f"{segundos_totales:.2f}",
        "dimension": (
            "TF-IDF union word+char_wb por pliegue (dims en result); comun a A y a los brazos ROS"
        ),
        "result": json.dumps(resultado, ensure_ascii=True, default=_json_default),
        "interpretation": (
            f"Equivalencia B1-A: max|delta proba| {equivalencia['max_delta_proba_global']:.3e} "
            f"(tolerancia 1e-3), delta F1 congelado {equivalencia['delta_f1_frozen']:+.6f} "
            f"(tolerancia 0.001); veredicto {equivalencia['veredicto']}. "
            f"B2 delta global vs A {resumen_brazos['B2']['delta_global_frozen_vs_A']:+.6f}; "
            f"B3 delta global vs A {resumen_brazos['B3']['delta_global_frozen_vs_A']:+.6f}. "
            "La unidad de medida es el reweighting, no el remuestreo."
        ),
        "decision": carga["decision"]["texto"],
        "date": str(CONFIG["fecha"]),
        "script": str(CONFIG["script"]),
        "notes": (
            "EXP-03 del preregistro P84 (micro-verificacion de equivalencia; rediseno de la "
            "spec seccion 4). Contadores dev 1483 / holdout 0 / test 0; toxic_test.csv nunca se "
            "abre. Sin Gate 2 ni EXP-06. RUNLOG.md y DECISIONS.md quedan a cargo del trainer."
        ),
    }


def construir_informe_md(carga: dict[str, Any]) -> str:
    """Construye el informe Markdown ASCII de la corrida.

    Args:
        carga: informe JSON completo.

    Returns:
        Texto Markdown.
    """
    lineas: list[str] = []
    lineas.append("# P84-03: ROS frente a reweighting (EXP-03, micro-verificacion de equivalencia)")
    lineas.append("")
    lineas.append(f"- Fecha: {carga['fecha']}")
    lineas.append(f"- Script: {carga['script']}")
    modo = "HUMO (no interpretable)" if carga["modo_humo"] else (
        "corrida completa" if carga["corrida_completa"] else "corrida parcial (no interpretable)"
    )
    lineas.append(f"- Modo: {modo}")
    lineas.append(f"- Referencia: {carga['preregistro']['referencia']}")
    lineas.append(f"- Decision pre-declarada: {carga['preregistro']['decision_predeclarada']}")
    lineas.append("")
    lineas.append("## 1. Objetivo")
    lineas.append("")
    lineas.append(
        "El plan original pedia medir si ROS aporta mas que class_weight=balanced con Gate 1."
    )
    lineas.append(
        "Para un modelo lineal convexo con L2 (liblinear) la duplicacion de filas es una"
    )
    lineas.append(
        "reasignacion de pesos por instancia; el diseno se convierte en una micro-verificacion"
    )
    lineas.append(
        "de equivalencia. Ningun brazo es promovible: ni Gate 2 ni EXP-06."
    )
    lineas.append("")
    lineas.append("## 2. Fundamento algebraico")
    lineas.append("")
    lineas.append("Notacion: n = n_pos + n_neg (outer_train por pliegue y etiqueta);")
    lineas.append("n_pos' = round(r * n_neg) positivos tras ROS; r = n_pos'/n_neg.")
    lineas.append("")
    lineas.append("liblinear resuelve: 0.5 * ||w||^2 + C * sum_i s_i * loss_i.")
    lineas.append("")
    lineas.append("- class_weight='balanced' (control A): s_pos/s_neg = n_neg/n_pos.")
    lineas.append("- ROS normalizado (B1 r=1.0; B2 r=0.5), peso uniforme w = n/(n_pos'+n_neg):")
    lineas.append("  masa total = n (media 1); peso efectivo medio por positivo original =")
    lineas.append("  n_pos' * w / n_pos; razon frente al negativo = n_pos'/n_pos = r * n_neg/n_pos.")
    lineas.append("  Con r=1 la razon es exactamente balanced => B1 equivalente a A (salvo")
    lineas.append("  redondeo entero de las copias y trayectoria del solver). Con r=0.5 es un")
    lineas.append("  reweighting mas suave (familia class_weight V4-C04, cerrada).")
    lineas.append("- ROS sin normalizar (B3 r=1.0, pesos 1): masa total = n_neg*(1+r);")
    lineas.append("  C_efectivo = C * n_neg*(1+r)/n (con C=2 y r=1 ronda 2.25 en un pliegue")
    lineas.append("  tipico): mezcla de class_weight y capacidad (familias cerradas).")
    lineas.append("")
    lineas.append("## 3. Gate 0")
    lineas.append("")
    evidencia = carga["gate0"]["evidencia"]
    lineas.append("| Artefacto | sha256 (16) | Estado |")
    lineas.append("|---|---|---|")
    for hash_ in evidencia["hashes"]:
        lineas.append(f"| {hash_['nombre']} | {hash_['sha256'][:16]} | OK |")
    particiones = evidencia["particiones"]
    contadores = evidencia["contadores"]
    lineas.append("")
    lineas.append(
        f"- Particiones: dev {particiones['dev']}, holdout {particiones['holdout']}, "
        f"pliegues {particiones['pliegues_tamanos']}, sin grupos compartidos."
    )
    lineas.append(
        f"- Contadores: dev leidas {contadores['dev_filas_leidas']}, holdout usadas "
        f"{contadores['holdout_filas_usadas']}, test usadas {contadores['test_filas_usadas']}."
    )
    identidad = carga["gate0"]["identidad_v03"]
    lineas.append(
        f"- Identidad V03: completa {identidad['completa']}, max|delta proba| "
        f"{identidad['max_delta_proba_vs_npz']:.3e} (tolerancia {identidad['tolerancia']:g}), "
        f"celdas de prediccion distintas {identidad['celdas_de_prediccion_distintas']}."
    )
    if identidad["corrected_cv_reconstruido"] is not None:
        lineas.append(
            f"- corrected_cv reconstruido {identidad['corrected_cv_reconstruido']!r} vs "
            f"{identidad['corrected_cv_esperado']!r}."
        )
    lineas.append(
        f"- V03@0.5 recalculado {identidad['f1_at_05_recalculado']!r} vs "
        f"{identidad['f1_at_05_esperado']!r}."
    )
    lineas.append("")
    lineas.append("## 4. Brazos y duplicacion determinista")
    lineas.append("")
    lineas.append("| Brazo | Definicion |")
    lineas.append("|---|---|")
    lineas.append("| A | control V03: LR C=2.0 balanced liblinear rs=42 |")
    lineas.append("| B1 | ROS r=1.0, n_pos'=round(r*n_neg), pesos w=n/(n_pos'+n_neg), C=2 |")
    lineas.append("| B2 | ROS r=0.5, misma normalizacion, C=2 |")
    lineas.append("| B3 | ROS r=1.0 sin normalizar (pesos 1), C=2 nominal |")
    lineas.append("")
    lineas.append(
        "Duplicacion: copias asignadas con numpy.random.default_rng(20260926) sobre una"
    )
    lineas.append(
        "permutacion de los indices positivos de outer_train (fresco por pliegue y etiqueta);"
    )
    lineas.append("la validacion nunca se duplica ni se usa para elegir copias.")
    lineas.append("")
    lineas.append("| Pliegue | Etiqueta | Brazo | n_pos | n_neg | n_pos' | min/max copia | sin copia | peso | C_efectivo |")
    lineas.append("|---|---|---|---|---|---|---|---|---|---|")
    for clave in sorted(carga["pliegues"], key=int):
        registro = carga["pliegues"][clave]
        for etiqueta in ETIQUETAS:
            registros = registro["etiquetas"][etiqueta]
            for brazo in BRAZOS_ROS:
                rec = registros[brazo]
                peso_texto = "1" if rec["peso_uniforme"] is None else f"{rec['peso_uniforme']:.6f}"
                lineas.append(
                    f"| {clave} | {etiqueta} | {brazo} | {rec['n_pos']} | {rec['n_neg']} | "
                    f"{rec['n_prima']} | {rec['recuento_min']}/{rec['recuento_max']} | "
                    f"{rec['n_pos_sin_copia']} | {peso_texto} | {rec['c_efectivo']:.4f} |"
                )
    lineas.append("")
    lineas.append("## 5. Equivalencia B1-A")
    lineas.append("")
    equivalencia = carga["equivalencia_B1_A"]
    lineas.append(
        f"- max|proba_B1 - proba_A| global: {equivalencia['max_delta_proba_global']:.3e} "
        f"(tolerancia {equivalencia['tolerancia_proba']:g}): "
        f"{'PASA' if equivalencia['pasa_proba'] else 'NO PASA'}."
    )
    lineas.append(
        f"- |delta F1| congelado: {abs(equivalencia['delta_f1_frozen']):.6f} "
        f"(tolerancia {equivalencia['tolerancia_f1']:g}): "
        f"{'PASA' if equivalencia['pasa_f1'] else 'NO PASA'}."
    )
    lineas.append(f"- Veredicto: {equivalencia['veredicto']}.")
    if "diagnostico_causas" in equivalencia:
        causas = equivalencia["diagnostico_causas"]
        lineas.append(
            f"- Causa observada: {causas['causa_principal']}; max desviacion de peso relativa "
            f"{causas['max_desviacion_peso_rel_vs_balanced']:.6f}; ajustes B1 sin converger "
            f"{causas['ajustes_b1_sin_converger']}."
        )
    if carga["corrida_completa"]:
        lineas.append("")
        lineas.append("| Brazo | F1 congelado | delta vs A | IC95 vs A | delta vs V03 | pliegues | rescate/dano | tasa predicha |")
        lineas.append("|---|---|---|---|---|---|---|---|")
        for brazo in BRAZOS_ROS:
            comparacion = carga["comparaciones"][brazo]["vs_A_frozen"]
            comparacion_v03 = carga["comparaciones"][brazo]["vs_V03_frozen"]
            ic = comparacion["bootstrap"]["ic95"]
            lineas.append(
                f"| {brazo} | {comparacion['comparacion']['f1_global']:.6f} | "
                f"{comparacion['comparacion']['delta_global']:+.6f} | "
                f"[{ic[0]:+.6f}, {ic[1]:+.6f}] | "
                f"{comparacion_v03['comparacion']['delta_global']:+.6f} | "
                f"{comparacion['comparacion']['pliegues_mejorados']}/5 | "
                f"{comparacion_v03['comparacion']['rescate_celdas']}/"
                f"{comparacion_v03['comparacion']['dano_celdas']} | "
                f"{comparacion_v03['comparacion']['tasa_positivos']} |"
            )
        lineas.append("")
        for brazo in BRAZOS_ROS:
            comp_v03 = carga["comparaciones"][brazo]["vs_V03_frozen"]["comparacion"]
            if comp_v03["predicciones_identicas"]:
                texto_celdas = (
                    "predicciones binarias identicas (0 celdas distintas de "
                    f"{comp_v03['celdas_totales']})"
                )
            else:
                texto_celdas = (
                    f"predicciones binarias distintas en {comp_v03['celdas_distintas']} de "
                    f"{comp_v03['celdas_totales']} celdas"
                )
            lineas.append(
                f"- {brazo} frente a V03 (umbrales congelados): {texto_celdas}; "
                f"rescate {comp_v03['rescate_celdas']} / dano {comp_v03['dano_celdas']} "
                f"(neto {comp_v03['neto_celdas']:+d})."
            )
        lineas.append("")
        lineas.append("### 5.1 Precision y recall por etiqueta")
        lineas.append("")
        lineas.append("| Brazo | Lectura | Etiqueta | precision | recall |")
        lineas.append("|---|---|---|---|---|")
        for brazo in ("A", *BRAZOS_ROS):
            for lectura in ("congelada", "umbral_05"):
                for etiqueta in ETIQUETAS:
                    metrica = carga["metricas_por_brazo"][brazo][lectura]
                    lineas.append(
                        f"| {brazo} | {lectura} | {etiqueta} | "
                        f"{metrica['precision_por_etiqueta'][etiqueta]:.6f} | "
                        f"{metrica['recall_por_etiqueta'][etiqueta]:.6f} |"
                    )
    lineas.append("")
    lineas.append("## 6. Decision")
    lineas.append("")
    lineas.append(f"- {carga['decision']['texto']}")
    for motivo in carga["decision"]["motivos"]:
        lineas.append(f"- {motivo}")
    lineas.append("")
    lineas.append("## 7. Entorno y tiempos")
    lineas.append("")
    lineas.append(
        f"- Versiones: {carga['versiones']}."
    )
    lineas.append(f"- Tiempo total: {carga['tiempo_total_segundos']:.1f}s.")
    if carga["incidencias"]:
        lineas.append(f"- Incidencias: {carga['incidencias']}.")
    lineas.append("")
    return "\n".join(lineas)


def construir_decision(
    equivalencia: dict[str, Any], corrida_completa: bool
) -> dict[str, Any]:
    """Redacta la decision pre-declarada de cierre con los numeros.

    Args:
        equivalencia: resultado de la equivalencia B1-A.
        corrida_completa: si la corrida cubrio los 5 pliegues.

    Returns:
        Diccionario con el texto de decision, la etiqueta de cierre (si aplica)
        y los motivos.
    """
    if not corrida_completa:
        return {
            "texto": "PROVISIONAL: corrida parcial o humo; no se registra decision final",
            "predeclarada": DECISION_PREDECLARADA,
            "promocion": "ninguna",
            "etiqueta_cierre": None,
            "motivos": ["corrida parcial: sin OOF completo ni veredicto definitivo"],
        }
    numero = (
        f"max|delta proba B1-A| = {equivalencia['max_delta_proba_global']:.3e}, "
        f"delta F1 congelado = {equivalencia['delta_f1_frozen']:+.6f}, "
        f"delta F1 a 0.5 = {equivalencia['delta_f1_05']:+.6f}"
    )
    etiqueta_cierre: str | None = None
    if equivalencia["pasa"]:
        texto = f"{DECISION_PREDECLARADA}. Equivalencia B1-A confirmada ({numero})"
        motivos = [
            "B1 reproduce a A dentro de tolerancia (1e-3 en probabilidades y 0.001 en F1)",
            "B2 y B3 quedan medidos y etiquetados como class_weight/C, no como ROS",
            "ningun brazo va a Gate 2 ni a EXP-06",
        ]
    else:
        causa = equivalencia.get("diagnostico_causas", {}).get(
            "causa_principal", "no determinada"
        )
        texto = (
            "STOP: REFUTACION PARCIAL de la equivalencia B1-A "
            f"({numero}; causa observada: {causa}). {DECISION_PREDECLARADA}; "
            "CLOSED BY EQUIVALENCE"
        )
        motivos = [
            "el criterio pre-declarado de equivalencia B1-A no se cumplio",
            f"causa observada: {causa}",
            "no se promueve ningun brazo: sin Gate 2 ni EXP-06",
        ]
        etiqueta_cierre = "CLOSED BY EQUIVALENCE"
    return {
        "texto": texto,
        "predeclarada": DECISION_PREDECLARADA,
        "promocion": "ninguna",
        "etiqueta_cierre": etiqueta_cierre,
        "motivos": motivos,
    }


# ===== SECCION 6: CLI y orquestacion =====


def parsear_lista_enteros(texto: str, nombre: str, validos: Sequence[int]) -> tuple[int, ...]:
    """Parsea una lista de enteros separados por comas y valida el dominio.

    Args:
        texto: cadena de entrada.
        nombre: nombre del argumento para mensajes.
        validos: valores admitidos.

    Returns:
        Tupla de enteros sin duplicados.

    Raises:
        SystemExit: si el formato es invalido, hay duplicados o valores fuera del dominio.
    """
    try:
        valores = tuple(int(parte) for parte in texto.split(",") if parte.strip())
    except ValueError as error:
        raise SystemExit(f"STOP: --{nombre} invalido: {texto!r}") from error
    if not valores:
        raise SystemExit(f"STOP: --{nombre} no puede estar vacio")
    if len(set(valores)) != len(valores):
        raise SystemExit(f"STOP: --{nombre} tiene valores duplicados: {valores}")
    invalidos = [valor for valor in valores if valor not in validos]
    if invalidos:
        raise SystemExit(f"STOP: --{nombre} fuera de {list(validos)}: {invalidos}")
    return valores


def parsear_argumentos(argv: Sequence[str] | None) -> argparse.Namespace:
    """Define y parsea los argumentos de la linea de comandos.

    Args:
        argv: argumentos (None usa sys.argv[1:]).

    Returns:
        Namespace con humo y pliegues.
    """
    analizador = argparse.ArgumentParser(
        description=(
            "P84-03: micro-verificacion de equivalencia del sobremuestreo aleatorio (ROS) "
            "frente a class_weight/reweighting (control V03)."
        )
    )
    analizador.add_argument(
        "--humo",
        action="store_true",
        help="modo humo: pliegue 1; no toca experiments.csv, npz OOF ni informes finales",
    )
    analizador.add_argument(
        "--pliegues",
        default="1,2,3,4,5",
        help="pliegues separados por comas (defecto 1,2,3,4,5)",
    )
    return analizador.parse_args(argv)


def resumen_c_efectivo_b3(detalle_ros: dict[int, dict[str, Any]]) -> dict[str, list[float]]:
    """Extrae el C efectivo de B3 por pliegue y etiqueta para el informe.

    Args:
        detalle_ros: detalle por pliegue de la fase ROS.

    Returns:
        Diccionario pliegue -> lista de C efectivo por etiqueta.
    """
    resumen: dict[str, list[float]] = {}
    for numero in sorted(detalle_ros):
        resumen[str(numero)] = [
            detalle_ros[numero]["etiquetas"][etiqueta]["B3"]["c_efectivo"]
            for etiqueta in ETIQUETAS
        ]
    return resumen


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta P84-03 completo.

    Args:
        argv: argumentos de linea de comandos.

    Returns:
        Codigo de salida 0 si todo termino bien.
    """
    inicio = time.perf_counter()
    random.seed(SEMILLA_GLOBAL)
    np.random.seed(SEMILLA_GLOBAL)
    argumentos = parsear_argumentos(argv)
    humo = bool(argumentos.humo)
    pliegues = parsear_lista_enteros(
        argumentos.pliegues, "pliegues", CONFIG["pliegues_por_defecto"]
    )
    if humo:
        pliegues = (1,)
        log("MODO HUMO: pliegue 1; sin experiments.csv, npz OOF ni informes finales")
    corrida_completa = bool(set(pliegues) == set(CONFIG["pliegues_por_defecto"]))
    log(f"pliegues {pliegues} | humo {humo} | corrida completa {corrida_completa}")

    # ---------- Gate 0 ----------
    log("Gate 0: hashes congelados, particiones, contadores e identidad V03")
    datos = cargar_gate0()
    seleccion = seleccion_pliegues(datos, pliegues)
    log(
        f"   dev {len(datos.des)} | holdout usado 0 | test usado 0 | "
        f"pliegues {datos.evidencia['particiones']['pliegues_tamanos']}"
    )
    proba_A, detalle_control = fase_control(datos, seleccion)
    identidad = control_identidad_v03(datos, proba_A, seleccion)
    log(
        f"   identidad V03: completa {identidad['completa']} | max|delta proba| "
        f"{identidad['max_delta_proba_vs_npz']:.3e} (tolerancia {identidad['tolerancia']:g}) | "
        f"celdas distintas {identidad['celdas_de_prediccion_distintas']}"
    )
    if identidad["corrected_cv_reconstruido"] is not None:
        log(
            f"   corrected_cv {identidad['corrected_cv_reconstruido']!r} | "
            f"V03@0.5 {identidad['f1_at_05_recalculado']!r} | identidad OK"
        )

    # ---------- brazos ROS ----------
    probas, detalle_ros = fase_ros(datos, seleccion)
    probas["A"] = proba_A

    # ---------- parciales por pliegue ----------
    rutas_parciales: list[str] = []
    for numero, _ in seleccion:
        rutas_parciales.append(
            str(escribir_parcial(numero, detalle_control[numero], detalle_ros[numero], humo))
        )
    log(f"parciales escritos: {len(rutas_parciales)}")

    # ---------- comparaciones, metricas por etiqueta, equivalencia y decision ----------
    comparaciones: dict[str, Any] = {}
    metricas_por_brazo: dict[str, Any] = {}
    equivalencia: dict[str, Any] = {}
    incidencias: list[str] = []
    pred_A_frozen = (proba_A >= datos.umbrales_v03[None, :]).astype(np.int64)
    pred_A_05 = (proba_A >= 0.5).astype(np.int64)
    comparaciones["A_vs_V03_frozen"] = registro_comparacion(
        datos, pred_A_frozen, datos.pred_v03, seleccion, corrida_completa
    )
    for brazo in BRAZOS_ROS:
        pred_frozen = (probas[brazo] >= datos.umbrales_v03[None, :]).astype(np.int64)
        pred_05 = (probas[brazo] >= 0.5).astype(np.int64)
        comparaciones[brazo] = {
            "vs_A_frozen": registro_comparacion(
                datos, pred_frozen, pred_A_frozen, seleccion, corrida_completa
            ),
            "vs_V03_frozen": registro_comparacion(
                datos, pred_frozen, datos.pred_v03, seleccion, corrida_completa
            ),
            "vs_A_05": registro_comparacion(
                datos, pred_05, pred_A_05, seleccion, corrida_completa
            ),
        }
    if corrida_completa:
        metricas_por_brazo = {
            brazo: {
                "congelada": metricas_de_lectura(
                    datos.y, (probas[brazo] >= datos.umbrales_v03[None, :]).astype(np.int64)
                ),
                "umbral_05": metricas_de_lectura(
                    datos.y, (probas[brazo] >= 0.5).astype(np.int64)
                ),
            }
            for brazo in ("A", *BRAZOS_ROS)
        }
    equivalencia = evaluar_equivalencia(datos, probas, seleccion, detalle_ros)
    log(
        f"equivalencia B1-A{' provisional' if not corrida_completa else ''}: "
        f"max|delta proba| {equivalencia['max_delta_proba_global']:.3e} | "
        f"delta F1 congelado {equivalencia['delta_f1_frozen']:+.6f}"
        + (f" | veredicto {equivalencia['veredicto']}" if corrida_completa else "")
    )
    for brazo in BRAZOS_ROS:
        comp_v03 = comparaciones[brazo]["vs_V03_frozen"]["comparacion"]
        comp_a = comparaciones[brazo]["vs_A_frozen"]["comparacion"]
        texto_celdas = (
            "predicciones binarias identicas"
            if comp_v03["predicciones_identicas"]
            else (
                f"predicciones binarias distintas en {comp_v03['celdas_distintas']} "
                f"de {comp_v03['celdas_totales']} celdas"
            )
        )
        log(
            f"   {brazo} vs V03_frozen: {texto_celdas} (rescate {comp_v03['rescate_celdas']} / "
            f"dano {comp_v03['dano_celdas']}, neto {comp_v03['neto_celdas']:+d}) | "
            f"vs A_frozen: rescate {comp_a['rescate_celdas']} / dano {comp_a['dano_celdas']}"
        )

    decision = construir_decision(equivalencia, corrida_completa)
    tiempo_total = round(time.perf_counter() - inicio, 2)

    # ---------- informe ----------
    carga: dict[str, Any] = {
        "experiment_id": "P84-03",
        "script": str(CONFIG["script"]),
        "fecha": str(CONFIG["fecha"]),
        "modo_humo": humo,
        "corrida_completa": corrida_completa,
        "seleccion": {"pliegues": list(pliegues)},
        "preregistro": {
            "referencia": (
                "docs/p84_spec.md seccion 4 (EXP-03, micro-verificacion de equivalencia) + "
                "seccion 1 (convenciones comunes)"
            ),
            "decision_predeclarada": DECISION_PREDECLARADA,
            "criterio_equivalencia": {
                "max_delta_proba": TOL_EQUIV_PROBA,
                "max_abs_delta_f1": TOL_EQUIV_F1,
                "lectura_primaria": "umbrales congelados 0.44/0.51/0.46",
                "contingencia_refutacion": (
                    "REFUTACION PARCIAL con causa (redondeo entero o trayectoria del solver); "
                    "nunca promocion"
                ),
            },
            "gate2": False,
            "exp06": False,
        },
        "config": {
            "brazos": BRAZOS,
            "lr_control": PARAMS_LR_CONTROL,
            "lr_ros": PARAMS_LR_ROS,
            "duplicacion": {
                "semilla": SEMILLA_DUPLICACION,
                "regla": "n_pos' = round(r*n_neg); copias sobre permutacion sembrada",
                "validacion_nunca_duplicada": True,
            },
            "bootstrap": {"repeticiones": REPETICIONES_BOOTSTRAP, "semilla": SEMILLA_BOOTSTRAP},
            "umbrales_congelados": datos.umbrales_v03.tolist(),
        },
        "gate0": {"evidencia": datos.evidencia, "identidad_v03": identidad},
        "versiones": versiones_entorno(),
        "pliegues": {
            str(numero): {
                **{clave: valor for clave, valor in detalle_ros[numero].items() if clave != "etiquetas"},
                "etiquetas": detalle_ros[numero]["etiquetas"],
                "control": detalle_control[numero],
            }
            for numero, _ in seleccion
        },
        "comparaciones": comparaciones,
        "metricas_por_brazo": metricas_por_brazo,
        "equivalencia_B1_A": equivalencia,
        "resumen_c_efectivo_B3": resumen_c_efectivo_b3(detalle_ros),
        "decision": decision,
        "parciales": rutas_parciales,
        "incidencias": incidencias,
        "tiempo_total_segundos": tiempo_total,
    }

    if humo:
        log("modo humo: no se escriben experiments.csv, npz OOF ni informes finales")
    elif corrida_completa:
        ruta_npz = guardar_oof_npz(datos, probas)
        carga["oof_npz"] = str(ruta_npz)
        fila = construir_fila_experiments(
            carga, datos, probas, comparaciones, equivalencia, tiempo_total
        )
        carga["experiments_csv"] = upsert_experiments_csv(fila)
        escribir_json(CONFIG["informe_json"], carga)
        escribir_texto(CONFIG["informe_md"], construir_informe_md(carga))
        log(f"informes: {CONFIG['informe_json'].name} y {CONFIG['informe_md'].name}")
    else:
        ruta_json_parcial = CONFIG["parciales_dir"] / "p84_03_informe_parcial.json"
        ruta_md_parcial = CONFIG["parciales_dir"] / "p84_03_informe_parcial.md"
        escribir_json(ruta_json_parcial, carga)
        escribir_texto(ruta_md_parcial, construir_informe_md(carga))
        log(
            "AVISO: corrida parcial: sin experiments.csv ni informe final; informe etiquetado en "
            f"{ruta_json_parcial.name} y {ruta_md_parcial.name}"
        )

    # ---------- resumen final ----------
    log("=" * 78)
    log(f"P84-03 {'HUMO' if humo else 'OK'} | device cpu | {tiempo_total:.1f}s")
    log(
        f"   identidad V03: max|delta proba| {identidad['max_delta_proba_vs_npz']:.3e} "
        f"(tolerancia {identidad['tolerancia']:g})"
    )
    log(
        f"   equivalencia B1-A: max|delta proba| {equivalencia['max_delta_proba_global']:.3e} | "
        f"delta F1 {equivalencia['delta_f1_frozen']:+.6f} | {equivalencia['veredicto']}"
    )
    if corrida_completa:
        for brazo in BRAZOS_ROS:
            comparacion = comparaciones[brazo]["vs_A_frozen"]["comparacion"]
            ic = comparaciones[brazo]["vs_A_frozen"]["bootstrap"]["ic95"]
            log(
                f"   {brazo}: F1 congelado {comparacion['f1_global']:.6f} | delta vs A "
                f"{comparacion['delta_global']:+.6f} | IC95 [{ic[0]:+.6f}, {ic[1]:+.6f}]"
            )
    log(f"   decision: {decision['texto']}")
    log("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        codigo_salida = main()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - red de seguridad del punto de entrada
        traceback.print_exc()
        sys.exit(1)
    sys.exit(codigo_salida)
