"""p84_05_em_shrinkage_simulado.py
Fecha: 2026-10-02
Descripcion: EXP-05 (P84-05): correccion de prevalencia EM con shrinkage, simulacion OOF.

Mide si la correccion de odds por cambio de prevalencia (EM) mejora cuando se encoge hacia
cero (shrinkage geometrico) y el umbral se decide de forma robusta (minimax sobre un
intervalo de razones de prevalencia rho), en simulacion sobre el OOF congelado de V03.
Expectativa pre-registrada: muy baja. Si el gate falla, se cierra toda la familia
EM/threshold. Este script es numpy puro: no reentrena modelos y no toca test ni holdout.

Protocolo (docs/p84_spec.md seccion 6, EXP-05):
- Gate 0: sha256 del npz congelado; recalculo de corrected_cv (0.7772114525165721) y de
  V03@0.5 (0.771778476256122); contadores dev 1483 / holdout 0 / test 0. toxic_test.csv
  nunca se abre (no se referencia ninguna ruta de test en este script).
- Simulacion de shift por etiqueta y rho en {0.85, 0.90, 0.94, 1.00, 1.05}: pesos de
  importancia w_i = rho si y_i = 1 y w_i = 1 si y_i = 0. El estimando es la F1 ponderada
  2*sum(w*y*pred) / (sum(w*y) + sum(w*pred)) por etiqueta y su media. Es una simulacion
  analitica determinista que asume p(x|y) invariable (declarado).
- Estimador EM con shrinkage: rho_0 = 1; iterar hasta 1e-10
  p'_i = sigmoid(logit(p_i) + log rho_t), pi_t = sum(w_i p'_i)/sum(w_i),
  rho_{t+1} = (pi_t/(1-pi_t)) / (pi_train/(1-pi_train)); shrinkage
  rho_usado = exp(s * log rho_hat) con s = n_eff/(n_eff + m) y n_eff el tamano efectivo
  de Kish de los pesos; correccion p''_i = sigmoid(logit(p_i) + log rho_usado);
  prediccion p''_i >= t.
- Rejilla declarada: m en {0, 25, 50, 100}; t en 0.20 a 0.80 paso 0.01. Criterio minimax:
  maximizar el minimo de la F1 ponderada sobre las 5 rho; empate: menor m y despues t mas
  cercano al umbral congelado de V03 de esa etiqueta (0.44/0.51/0.46).
- Guardarrail de tasa: a rho = 1.00 la tasa predicha por etiqueta debe quedar a +-0.005 de
  la de V03; la seleccion LOFO filtra los candidatos que lo cumplen y publica el optimo
  sin restriccion como diagnostico.
- Seleccion LOFO: para cada pliegue k se selecciona (m, t) por etiqueta con las filas de
  los otros 4 pliegues y se aplica al pliegue k (el EM de aplicacion se estima con las
  filas del propio pliegue, convencion declarada). Se compara contra V03 (umbrales
  congelados, sin correccion) bajo los mismos pesos y la misma rho.
- Controles: (a) descomposicion contra una regla solo-umbral (mismo t, rho = 1, sin EM);
  (b) sanidad sin shift (delta a rho = 1.00 >= 0); (c) guardarrail de tasa.
- Gate unico: delta medio (promedio de las 5 rho) >= +0.002; >= 4/5 pliegues LOFO
  positivos; IC95 bootstrap pareado por filas (10.000, semilla 20260926, con pesos) que
  excluya cero; guardarrail de tasa cumplido; componente EM positiva.
- Salidas: reports/p84_05_em.json y .md; parciales por pliegue en reports/p84_05_parciales/;
  fila P84-05 en experiments.csv (upsert idempotente, 23 columnas, utf-8-sig).
- En --humo no se tocan experiments.csv, los informes finales ni el CSV; solo parciales con
  sufijo _humo.

Especificacion: docs/p84_spec.md seccion 6 (EXP-05) + seccion 1 (convenciones comunes).
Preregistro: docs/p84_preregistro.md (Gate 0, particiones, bootstrap 10000/20260926).

Uso:
    /home/carlos/venv/bin/python experiments/p84_05_em_shrinkage_simulado.py
    /home/carlos/venv/bin/python experiments/p84_05_em_shrinkage_simulado.py --humo
    /home/carlos/venv/bin/python experiments/p84_05_em_shrinkage_simulado.py --pliegues 1,2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import sha256_of
from src.evaluation import ETIQUETAS, metrica_macro, metricas_multietiqueta

# ===== SECCION 0: constantes congeladas, configuracion y utilidades =====

HASH_OOF_V03: Final[str] = "559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab"

V03_CCV: Final[float] = 0.7772114525165721
V03_F1_AT_05: Final[float] = 0.771778476256122
UMBRALES_CONGELADOS: Final[np.ndarray] = np.array([0.44, 0.51, 0.46], dtype=np.float64)

FILAS_DEV: Final[int] = 1483
FILAS_HOLDOUT_USADAS: Final[int] = 0
FILAS_TEST_USADAS: Final[int] = 0
N_PLIEGUES: Final[int] = 5
TAMANOS_PLIEGUES: Final[list[int]] = [297, 297, 296, 296, 297]

PREVALENCIAS_DECLARADAS: Final[dict[str, float]] = {
    "HS": 0.436952,
    "TR": 0.300067,
    "AG": 0.382333,
}

RHOS: Final[tuple[float, ...]] = (0.85, 0.90, 0.94, 1.00, 1.05)
RHO_SIN_SHIFT: Final[float] = 1.00
M_GRID: Final[tuple[int, ...]] = (0, 25, 50, 100)
T_GRID: Final[np.ndarray] = np.round(np.arange(0.20, 0.8001, 0.01), 2)

RHO_INICIAL: Final[float] = 1.0
TOL_EM: Final[float] = 1e-10
MAX_ITER_EM: Final[int] = 1000
TOL_TASA: Final[float] = 0.005
EPS: Final[float] = 1e-12

GATE_DELTA: Final[float] = 0.002
GATE_PLIEGUES: Final[int] = 4
REPETICIONES_BOOTSTRAP: Final[int] = 10_000
SEMILLA_BOOTSTRAP: Final[int] = 20260926
SEMILLA_GLOBAL: Final[int] = 20260926
LOTE_BOOTSTRAP: Final[int] = 2000

CONFIG: Final[dict[str, Any]] = {
    "oof_v03_path": PROJECT_ROOT / "features" / "oof_v03_final.npz",
    "experiments_csv": PROJECT_ROOT / "experiments.csv",
    "parciales_dir": PROJECT_ROOT / "reports" / "p84_05_parciales",
    "informe_json": PROJECT_ROOT / "reports" / "p84_05_em.json",
    "informe_md": PROJECT_ROOT / "reports" / "p84_05_em.md",
    "fecha": "2026-10-02",
    "script": "experiments/p84_05_em_shrinkage_simulado.py",
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

INDICE_RHO1: Final[int] = RHOS.index(RHO_SIN_SHIFT)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso con el prefijo de la fase.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P84-05] {mensaje}", flush=True)


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
        Diccionario con python, numpy, sklearn y el device declarado.
    """
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "device": "cpu (numpy puro, sin entrenamiento; GPU no usada)",
        "plataforma": platform.platform(),
    }


def parsear_lista_enteros(texto: str, nombre: str, permitidos: Sequence[int]) -> tuple[int, ...]:
    """Parsea una lista de enteros separados por comas.

    Args:
        texto: cadena con valores separados por comas.
        nombre: etiqueta usada en los mensajes de error.
        permitidos: valores admisibles.

    Returns:
        Tupla de enteros sin repetidos y ordenada.

    Raises:
        SystemExit: si el formato o los valores no son validos.
    """
    try:
        valores = tuple(int(parte.strip()) for parte in texto.split(",") if parte.strip())
    except ValueError as exc:
        raise SystemExit(f"STOP: {nombre} invalido: {texto!r}") from exc
    if not valores:
        raise SystemExit(f"STOP: {nombre} vacio")
    if len(set(valores)) != len(valores):
        raise SystemExit(f"STOP: {nombre} con valores repetidos: {valores}")
    invalidos = [valor for valor in valores if valor not in permitidos]
    if invalidos:
        raise SystemExit(f"STOP: {nombre} fuera de rango {list(permitidos)}: {invalidos}")
    return tuple(sorted(valores))


# ===== SECCION 1: Gate 0 (hash, identidad V03 y contadores) =====


@dataclass
class DatosGate0:
    """Datos congelados verificados en Gate 0.

    Attributes:
        y: etiquetas verdaderas (1483, 3) en el orden HS, TR, AG.
        p_v03: probabilidades OOF de V03 (1483, 3).
        pred_v03: predicciones de V03 con umbrales congelados (1483, 3).
        umbrales: umbrales congelados [0.44, 0.51, 0.46].
        folde: numero de pliegue por fila (1..5).
        ids: ids del OOF canonico.
        prevalencias: prevalencias exactas por etiqueta recomputadas desde y.
        evidencia: informe serializable de Gate 0.
    """

    y: np.ndarray
    p_v03: np.ndarray
    pred_v03: np.ndarray
    umbrales: np.ndarray
    folde: np.ndarray
    ids: np.ndarray
    prevalencias: np.ndarray
    evidencia: dict[str, Any]


def cargar_gate0() -> DatosGate0:
    """Ejecuta Gate 0 sobre el npz congelado de V03.

    Verifica el sha256, las formas, los contadores de acceso (dev 1483 / holdout 0 /
    test 0), los umbrales congelados, la particion en 5 pliegues y la identidad:
    corrected_cv = 0.7772114525165721 y V03@0.5 = 0.771778476256122, ambos
    recalculados desde el npz con tolerancia 1e-12.

    Returns:
        Datos y evidencia de Gate 0.

    Raises:
        SystemExit: si cualquier verificacion congelada falla.
    """
    hashes = [verificar_sha256(CONFIG["oof_v03_path"], HASH_OOF_V03, "oof_v03_final.npz")]
    oof = np.load(CONFIG["oof_v03_path"], allow_pickle=True)
    y = oof["y_true"].astype(np.int64)
    p_v03 = oof["proba"].astype(np.float64)
    pred = oof["pred"].astype(np.int64)
    umbrales = oof["umbrales"].astype(np.float64)
    folde = oof["pliegues"].astype(np.int64)
    ids = oof["ids"].astype(np.int64)

    if y.shape != (FILAS_DEV, len(ETIQUETAS)) or p_v03.shape != y.shape:
        raise SystemExit(f"STOP: formas inesperadas y {y.shape} / proba {p_v03.shape}")
    if folde.shape != (FILAS_DEV,) or ids.shape != (FILAS_DEV,):
        raise SystemExit("STOP: pliegues o ids con forma inesperada")
    if not np.isfinite(p_v03).all():
        raise SystemExit("STOP: proba contiene valores no finitos")
    if len(np.unique(ids)) != FILAS_DEV:
        raise SystemExit("STOP: ids repetidos en el OOF canonico")
    if not np.allclose(umbrales, UMBRALES_CONGELADOS, atol=0.0, rtol=0.0):
        raise SystemExit(f"STOP: umbrales congelados inesperados: {umbrales.tolist()}")
    if not np.array_equal((p_v03 >= umbrales[None, :]).astype(np.int64), pred):
        raise SystemExit("STOP: pred del npz no reproduce proba >= umbrales congelados")

    tamanos = [int(np.sum(folde == k)) for k in range(1, N_PLIEGUES + 1)]
    if tamanos != TAMANOS_PLIEGUES:
        raise SystemExit(f"STOP: tamanos de pliegue inesperados: {tamanos}")

    ccv_recalculado = metricas_multietiqueta(y, pred)["f1_media_columnas"]
    if abs(ccv_recalculado - V03_CCV) > 1e-12:
        raise SystemExit(
            f"STOP: corrected_cv recalculado {ccv_recalculado!r} != {V03_CCV!r} (tolerancia 1e-12)"
        )
    if abs(float(oof["corrected_cv"][0]) - V03_CCV) > 1e-12:
        raise SystemExit("STOP: corrected_cv guardado no coincide con la referencia")
    f1_05 = metrica_macro(y, (p_v03 >= 0.5).astype(np.int64))
    if abs(f1_05 - V03_F1_AT_05) > 1e-12:
        raise SystemExit(f"STOP: V03@0.5 recalculado {f1_05!r} != {V03_F1_AT_05!r}")

    f1_etiqueta_dict = metricas_multietiqueta(y, pred)["f1_por_etiqueta"]
    f1_etiqueta = [f1_etiqueta_dict[etiqueta] for etiqueta in ETIQUETAS]
    if not np.allclose(f1_etiqueta, oof["f1_por_etiqueta"], atol=1e-12, rtol=0.0):
        raise SystemExit("STOP: f1_por_etiqueta no se reproduce desde el npz")
    f1_pliegues = [
        float(metrica_macro(y[folde == k], pred[folde == k])) for k in range(1, N_PLIEGUES + 1)
    ]
    if not np.allclose(f1_pliegues, oof["f1_pliegues"], atol=1e-12, rtol=0.0):
        raise SystemExit("STOP: f1_pliegues no se reproduce desde el npz")

    prevalencias = y.mean(axis=0)
    desviaciones_prevalencia = {
        etiqueta: abs(float(prevalencias[c]) - PREVALENCIAS_DECLARADAS[etiqueta])
        for c, etiqueta in enumerate(ETIQUETAS)
    }
    if any(valor > 1e-6 for valor in desviaciones_prevalencia.values()):
        raise SystemExit(
            f"STOP: prevalencias recomputadas fuera de tolerancia: {desviaciones_prevalencia}"
        )

    contadores = {
        "dev_filas_leidas": FILAS_DEV,
        "holdout_filas_usadas": FILAS_HOLDOUT_USADAS,
        "test_filas_usadas": FILAS_TEST_USADAS,
        "toxic_test_abierto": False,
        "nota": (
            "el npz OOF es la unica fuente; no se lee toxic_train.csv ni el holdout (248) "
            "y toxic_test.csv nunca se abre"
        ),
    }
    evidencia = {
        "hashes": hashes,
        "contadores": contadores,
        "identidad_v03": {
            "corrected_cv_recalculado": ccv_recalculado,
            "corrected_cv_npz": float(oof["corrected_cv"][0]),
            "v03_f1_at_05_recalculado": f1_05,
            "v03_f1_at_05_referencia": V03_F1_AT_05,
            "tolerancia": 1e-12,
            "f1_por_etiqueta_recalculado": {
                etiqueta: f1_etiqueta[c] for c, etiqueta in enumerate(ETIQUETAS)
            },
            "f1_por_pliegue_recalculado": f1_pliegues,
            "umbrales": umbrales.tolist(),
        },
        "particiones": {
            "dev": FILAS_DEV,
            "holdout_usado": FILAS_HOLDOUT_USADAS,
            "pliegues_tamanos": tamanos,
            "filas_cubiertas_una_vez": bool(
                all(np.sum(folde == k) == tamano for k, tamano in enumerate(TAMANOS_PLIEGUES, 1))
            ),
        },
        "prevalencias": {
            "recomputadas": {
                etiqueta: float(prevalencias[c]) for c, etiqueta in enumerate(ETIQUETAS)
            },
            "declaradas": PREVALENCIAS_DECLARADAS,
            "desviacion_maxima": max(desviaciones_prevalencia.values()),
        },
    }
    return DatosGate0(
        y=y,
        p_v03=p_v03,
        pred_v03=pred,
        umbrales=umbrales,
        folde=folde,
        ids=ids,
        prevalencias=prevalencias,
        evidencia=evidencia,
    )


# ===== SECCION 2: estimando ponderado y estimador EM con shrinkage =====


@dataclass
class ResultadoEM:
    """Resultado de la iteracion de punto fijo del estimador EM.

    Attributes:
        rho: razon de odds estimada (ultimo valor iterado).
        iteraciones: numero de iteraciones ejecutadas.
        convergido: True si el residual final quedo por debajo de la tolerancia.
        residual: |rho_{t+1} - rho_t| final.
    """

    rho: float
    iteraciones: int
    convergido: bool
    residual: float


def sigmoide(z: np.ndarray) -> np.ndarray:
    """Sigmoide numericamente estable.

    Args:
        z: valores de entrada (log-odds).

    Returns:
        Sigmoide elemento a elemento, recortada a rango seguro.
    """
    recortado = np.clip(np.asarray(z, dtype=np.float64), -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-recortado))


def logit(p: np.ndarray) -> np.ndarray:
    """Logit con recorte para evitar infinitos.

    Args:
        p: probabilidades en (0, 1).

    Returns:
        log(p / (1 - p)) elemento a elemento.
    """
    recortada = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(recortada / (1.0 - recortada))


def pesos_importancia(y_columna: np.ndarray, rho: float) -> np.ndarray:
    """Construye los pesos de importancia del shift simulado.

    Args:
        y_columna: etiqueta binaria de una columna.
        rho: razon de prevalencia simulada.

    Returns:
        Vector de pesos w_i = rho si y_i = 1 y 1 si y_i = 0.
    """
    return np.where(np.asarray(y_columna) == 1, rho, 1.0).astype(np.float64)


def tamano_efectivo_kish(w: np.ndarray) -> float:
    """Tamano efectivo de Kish de los pesos.

    Args:
        w: pesos positivos.

    Returns:
        (suma(w) ** 2) / suma(w ** 2).
    """
    return float(np.sum(w) ** 2 / np.sum(w**2))


def f1_ponderada(y_columna: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    """F1 ponderada 2*sum(w*y*pred) / (sum(w*y) + sum(w*pred)).

    Args:
        y_columna: etiqueta binaria verdadera.
        pred: prediccion binaria (0/1).
        w: pesos por fila.

    Returns:
        F1 ponderada; 0.0 si el denominador es cero.
    """
    wy = w * y_columna
    numerador = 2.0 * float(np.sum(wy * pred))
    denominador = float(np.sum(wy) + np.sum(w * pred))
    return numerador / denominador if denominador > 0.0 else 0.0


def precision_ponderada(y_columna: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    """Precision ponderada sum(w*y*pred) / sum(w*pred).

    Args:
        y_columna: etiqueta binaria verdadera.
        pred: prediccion binaria (0/1).
        w: pesos por fila.

    Returns:
        Precision ponderada; 0.0 si no hay positivos predichos con peso.
    """
    denominador = float(np.sum(w * pred))
    if denominador <= 0.0:
        return 0.0
    return float(np.sum(w * y_columna * pred)) / denominador


def recall_ponderada(y_columna: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    """Recall ponderado sum(w*y*pred) / sum(w*y).

    Args:
        y_columna: etiqueta binaria verdadera.
        pred: prediccion binaria (0/1).
        w: pesos por fila.

    Returns:
        Recall ponderado; 0.0 si no hay positivos reales con peso.
    """
    denominador = float(np.sum(w * y_columna))
    if denominador <= 0.0:
        return 0.0
    return float(np.sum(w * y_columna * pred)) / denominador


def f1_por_umbral(p: np.ndarray, y_columna: np.ndarray, w: np.ndarray, rejilla: np.ndarray) -> np.ndarray:
    """Calcula la F1 ponderada para todos los umbrales de la rejilla.

    Args:
        p: probabilidades (o scores corregidos) de la etiqueta.
        y_columna: etiqueta binaria verdadera.
        w: pesos por fila.
        rejilla: umbrales ordenados.

    Returns:
        Vector de F1 ponderada con la misma longitud que la rejilla.
    """
    preds = (p[:, None] >= rejilla[None, :]).astype(np.float64)
    wy = w * y_columna
    numerador = 2.0 * (preds.T @ wy)
    denominador = float(np.sum(wy)) + (preds.T @ w)
    return np.where(denominador > 0.0, numerador / np.maximum(denominador, EPS), 0.0)


def estimar_rho_em(
    p: np.ndarray,
    w: np.ndarray,
    pi_train: float,
    tol: float = TOL_EM,
    max_iter: int = MAX_ITER_EM,
) -> ResultadoEM:
    """Estima la razon de odds por EM (punto fijo) con pesos de importancia.

    Itera rho_0 = 1:
        p'_i = sigmoid(logit(p_i) + log rho_t)
        pi_t = sum(w_i p'_i) / sum(w_i)
        rho_{t+1} = (pi_t/(1-pi_t)) / (pi_train/(1-pi_train))
    hasta que |rho_{t+1} - rho_t| <= tol.

    Args:
        p: probabilidades de la etiqueta.
        w: pesos de importancia de las filas.
        pi_train: prevalencia de referencia (exacta, recomputada en Gate 0).
        tol: tolerancia de convergencia.
        max_iter: maximo de iteraciones.

    Returns:
        ResultadoEM con la razon estimada y la traza de convergencia.
    """
    rho = RHO_INICIAL
    residual = float("inf")
    log_odds_train = math.log(pi_train / (1.0 - pi_train))
    suma_w = float(np.sum(w))
    for iteracion in range(1, max_iter + 1):
        p_ajustada = sigmoide(logit(p) + math.log(rho))
        pi_t = float(np.sum(w * p_ajustada) / suma_w)
        pi_t = min(max(pi_t, EPS), 1.0 - EPS)
        rho_siguiente = math.log(pi_t / (1.0 - pi_t)) - log_odds_train
        rho_siguiente = math.exp(rho_siguiente)
        residual = abs(rho_siguiente - rho)
        if residual <= tol:
            return ResultadoEM(
                rho=rho_siguiente, iteraciones=iteracion, convergido=True, residual=residual
            )
        rho = rho_siguiente
    return ResultadoEM(rho=rho, iteraciones=max_iter, convergido=False, residual=residual)


def rho_con_shrinkage(rho_hat: float, n_eff: float, m: int) -> tuple[float, float]:
    """Aplica el shrinkage geometrico hacia rho = 1.

    Args:
        rho_hat: razon EM estimada.
        n_eff: tamano efectivo de Kish de los pesos.
        m: intensidad del shrinkage en {0, 25, 50, 100}.

    Returns:
        Tupla (rho_usado, s) con s = n_eff / (n_eff + m).
    """
    s = n_eff / (n_eff + m) if (n_eff + m) > 0.0 else 1.0
    rho_usado = math.exp(s * math.log(rho_hat)) if rho_hat > 0.0 else 1.0
    return rho_usado, s


def corregir_probabilidades(p: np.ndarray, rho_usado: float) -> np.ndarray:
    """Aplica la correccion de odds p'' = sigmoid(logit(p) + log rho_usado).

    Args:
        p: probabilidades originales.
        rho_usado: razon de odds aplicada.

    Returns:
        Probabilidades corregidas.
    """
    return sigmoide(logit(p) + math.log(rho_usado))


def umbral_equivalente(t: float, rho_usado: float) -> float:
    """Umbral equivalente en la escala original: p''>=t  <=>  p>=umbral.

    Args:
        t: umbral sobre la probabilidad corregida.
        rho_usado: razon de odds aplicada.

    Returns:
        Umbral en la escala de p: t / (t + (1 - t) * rho_usado).
    """
    return t / (t + (1.0 - t) * rho_usado)


def resumen_em(resultado: ResultadoEM) -> dict[str, Any]:
    """Convierte un ResultadoEM en un diccionario serializable.

    Args:
        resultado: resultado del EM.

    Returns:
        Diccionario con rho, iteraciones, convergencia y residual.
    """
    return {
        "rho": round(resultado.rho, 12),
        "iteraciones": resultado.iteraciones,
        "convergido": bool(resultado.convergido),
        "residual": float(resultado.residual),
    }


def evaluar_superficie(
    p: np.ndarray,
    y_columna: np.ndarray,
    pi_train: float,
    pred_v03: np.ndarray,
) -> dict[str, Any]:
    """Evalua la superficie (m, t, rho) de la F1 ponderada sobre un subconjunto.

    El EM solo depende de (etiqueta, rho, filas): se estima una vez por rho. Para cada
    m se encoge rho_hat y para cada umbral se calcula la F1 ponderada con los pesos de
    cada rho. Tambien devuelve las tasas predichas por (m, t, rho) para el guardarrail.

    Args:
        p: probabilidades de la etiqueta en las filas del subconjunto.
        y_columna: etiqueta binaria en las filas del subconjunto.
        pi_train: prevalencia de referencia.
        pred_v03: predicciones binarias de V03 en las filas del subconjunto.

    Returns:
        Diccionario con superficie (n_m, n_t, n_rho), tasas, EM por rho, n_eff por rho
        y la tasa de V03 del subconjunto.
    """
    pesos = {rho: pesos_importancia(y_columna, rho) for rho in RHOS}
    resultados_em = {rho: estimar_rho_em(p, pesos[rho], pi_train) for rho in RHOS}
    n_eff = {rho: tamano_efectivo_kish(pesos[rho]) for rho in RHOS}
    n_m = len(M_GRID)
    n_t = len(T_GRID)
    n_rho = len(RHOS)
    superficie = np.zeros((n_m, n_t, n_rho), dtype=np.float64)
    tasas = np.zeros((n_m, n_t, n_rho), dtype=np.float64)
    for mi, m in enumerate(M_GRID):
        for ri, rho in enumerate(RHOS):
            rho_usado, _ = rho_con_shrinkage(resultados_em[rho].rho, n_eff[rho], m)
            p_corregida = corregir_probabilidades(p, rho_usado)
            superficie[mi, :, ri] = f1_por_umbral(p_corregida, y_columna, pesos[rho], T_GRID)
            tasas[mi, :, ri] = (p_corregida[:, None] >= T_GRID[None, :]).mean(axis=0)
    return {
        "superficie": superficie,
        "tasas": tasas,
        "em": resultados_em,
        "n_eff": n_eff,
        "tasa_v03": float(np.mean(np.asarray(pred_v03, dtype=np.float64))),
    }


def _mejor_candidato(
    minimax: np.ndarray, factible: np.ndarray, umbral_v03: float
) -> dict[str, Any]:
    """Selecciona el mejor (m, t) por minimax con los desempates declarados.

    Empates: menor m y despues t mas cercano al umbral congelado de V03.

    Args:
        minimax: matriz (n_m, n_t) con el minimo sobre rho.
        factible: mascara booleana (n_m, n_t) de candidatos admisibles.
        umbral_v03: umbral congelado de la etiqueta.

    Returns:
        Diccionario con m, t, valor minimax y numero de candidatos admisibles.

    Raises:
        RuntimeError: si no hay ningun candidato admisible.
    """
    candidatos = np.argwhere(factible)
    if candidatos.size == 0:
        raise RuntimeError("sin candidatos admisibles en la rejilla")
    mejor_clave: tuple[float, int, float, float] | None = None
    mejor_mi = 0
    mejor_ti = 0
    for mi, ti in candidatos:
        clave = (
            -float(minimax[mi, ti]),
            int(M_GRID[mi]),
            abs(float(T_GRID[ti]) - umbral_v03),
            float(T_GRID[ti]),
        )
        if mejor_clave is None or clave < mejor_clave:
            mejor_clave = clave
            mejor_mi = int(mi)
            mejor_ti = int(ti)
    return {
        "indice_m": mejor_mi,
        "indice_t": mejor_ti,
        "m": int(M_GRID[mejor_mi]),
        "t": float(T_GRID[mejor_ti]),
        "minimax": float(minimax[mejor_mi, mejor_ti]),
        "candidatos_admisibles": int(candidatos.shape[0]),
    }


def seleccionar_en_superficie(
    superficie: np.ndarray,
    tasas: np.ndarray,
    tasa_v03: float,
    umbral_v03: float,
    guardarrail: bool,
) -> dict[str, Any]:
    """Selecciona (m, t) por minimax, opcionalmente con guardarrail de tasa.

    El criterio es maximizar el minimo de la F1 ponderada sobre las 5 rho. Con
    guardarrail se exige, a rho = 1.00, |tasa predicha - tasa de V03| <= TOL_TASA. Si
    ningun candidato lo cumple, se devuelve el optimo sin restriccion marcado como no
    factible (el gate lo penaliza).

    Args:
        superficie: matriz (n_m, n_t, n_rho) de F1 ponderada.
        tasas: matriz (n_m, n_t, n_rho) de tasas predichas.
        tasa_v03: tasa predicha de V03 en las filas del subconjunto.
        umbral_v03: umbral congelado de V03 de la etiqueta.
        guardarrail: si True se aplica la restriccion de tasa a rho = 1.00.

    Returns:
        Diccionario con la seleccion, el valor minimax y la desviacion de tasa.
    """
    minimax = superficie.min(axis=2)
    if guardarrail:
        factible = np.abs(tasas[:, :, INDICE_RHO1] - tasa_v03) <= TOL_TASA
        hay_factibles = bool(factible.any())
    else:
        factible = np.ones_like(minimax, dtype=bool)
        hay_factibles = True
    if hay_factibles:
        seleccion = _mejor_candidato(minimax, factible, umbral_v03)
    else:
        seleccion = _mejor_candidato(minimax, np.ones_like(minimax, dtype=bool), umbral_v03)
        seleccion["candidatos_admisibles"] = 0
    mi, ti = int(seleccion["indice_m"]), int(seleccion["indice_t"])
    tasa_seleccionada = float(tasas[mi, ti, INDICE_RHO1])
    seleccion.update(
        {
            "guardarrail": bool(guardarrail),
            "factible": hay_factibles if guardarrail else True,
            "tasa_rho1": tasa_seleccionada,
            "tasa_v03_rho1": tasa_v03,
            "desviacion_tasa_rho1": tasa_seleccionada - tasa_v03,
        }
    )
    return seleccion


def aplicar_regla(
    p: np.ndarray,
    y_columna: np.ndarray,
    umbral_v03: float,
    m: int,
    t: float,
    pi_train: float,
    pred_v03: np.ndarray,
) -> dict[str, Any]:
    """Aplica la regla EM seleccionada a un pliegue y mide sus lecturas.

    Para cada rho se estima el EM en las filas evaluadas, se encoge con m, se corrige
    p'' = sigmoid(logit(p) + log rho_usado) y se predice p'' >= t. Se comparan la regla
    completa, la de solo umbral (p >= t, sin EM) y V03 (umbrales congelados) bajo los
    mismos pesos.

    Args:
        p: probabilidades de la etiqueta en el pliegue evaluado.
        y_columna: etiqueta binaria del pliegue evaluado.
        umbral_v03: umbral congelado de la etiqueta.
        m: shrinkage seleccionado.
        t: umbral seleccionado sobre la escala corregida.
        pi_train: prevalencia de referencia.
        pred_v03: predicciones binarias de V03 en el pliegue.

    Returns:
        Diccionario con lecturas por rho, EM, n_eff y las matrices de prediccion
        (n, n_rho) de la regla completa y de la regla solo-umbral.
    """
    n = p.shape[0]
    n_rho = len(RHOS)
    pred_completa = np.zeros((n, n_rho), dtype=bool)
    pred_solo_umbral = np.zeros((n, n_rho), dtype=bool)
    por_rho: dict[float, dict[str, Any]] = {}
    em_por_rho: dict[float, ResultadoEM] = {}
    n_eff_por_rho: dict[float, float] = {}
    for ri, rho in enumerate(RHOS):
        w = pesos_importancia(y_columna, rho)
        resultado_em = estimar_rho_em(p, w, pi_train)
        n_eff = tamano_efectivo_kish(w)
        rho_usado, s = rho_con_shrinkage(resultado_em.rho, n_eff, m)
        p_corregida = corregir_probabilidades(p, rho_usado)
        pred_c = p_corregida >= t
        pred_s = p >= t
        pred_completa[:, ri] = pred_c
        pred_solo_umbral[:, ri] = pred_s
        f1_completa = f1_ponderada(y_columna, pred_c.astype(np.float64), w)
        f1_solo_umbral = f1_ponderada(y_columna, pred_s.astype(np.float64), w)
        f1_v03 = f1_ponderada(y_columna, pred_v03.astype(np.float64), w)
        por_rho[rho] = {
            "rho_usado": round(rho_usado, 12),
            "s": round(s, 12),
            "f1": round(f1_completa, 12),
            "f1_v03": round(f1_v03, 12),
            "f1_solo_umbral": round(f1_solo_umbral, 12),
            "delta": round(f1_completa - f1_v03, 12),
            "delta_em": round(f1_completa - f1_solo_umbral, 12),
            "delta_solo_umbral_vs_v03": round(f1_solo_umbral - f1_v03, 12),
            "tasa": round(float(pred_c.mean()), 12),
            "tasa_v03": round(float(np.mean(pred_v03)), 12),
            "tasa_solo_umbral": round(float(pred_s.mean()), 12),
        }
        em_por_rho[rho] = resultado_em
        n_eff_por_rho[rho] = n_eff
    deltas = [por_rho[rho]["delta"] for rho in RHOS]
    deltas_em = [por_rho[rho]["delta_em"] for rho in RHOS]
    return {
        "por_rho": por_rho,
        "em": {rho: resumen_em(resultado) for rho, resultado in em_por_rho.items()},
        "n_eff": n_eff_por_rho,
        "delta_medio": float(np.mean(deltas)),
        "delta_medio_em": float(np.mean(deltas_em)),
        "tasa_v03_rho1": round(float(np.mean(pred_v03)), 12),
        "pred_completa": pred_completa,
        "pred_solo_umbral": pred_solo_umbral,
    }


# ===== SECCION 3: simulacion LOFO por pliegue =====


@dataclass
class ResultadoLOFO:
    """Predicciones LOFO y registros serializables de la simulacion.

    Attributes:
        pred_completa: predicciones (n, 3, n_rho) de la regla EM seleccionada.
        pred_solo_umbral: predicciones (n, 3, n_rho) de la regla solo-umbral.
        pred_v03: predicciones (n, 3) de V03 con umbrales congelados.
        registros: registro JSON por pliegue (seleccion, superficies y lecturas).
    """

    pred_completa: np.ndarray
    pred_solo_umbral: np.ndarray
    pred_v03: np.ndarray
    registros: dict[int, dict[str, Any]]


def _resumen_seleccion(
    seleccion: dict[str, Any], evaluacion: dict[str, Any]
) -> dict[str, Any]:
    """Resume una seleccion para el informe, anadiendo el umbral equivalente.

    Args:
        seleccion: salida de seleccionar_en_superficie.
        evaluacion: salida de evaluar_superficie con EM y n_eff del subconjunto.

    Returns:
        Diccionario serializable de la seleccion con t_equivalente.
    """
    resumen = dict(seleccion)
    resumen.pop("indice_m", None)
    resumen.pop("indice_t", None)
    rho_usado_rho1, _ = rho_con_shrinkage(
        evaluacion["em"][RHO_SIN_SHIFT].rho,
        evaluacion["n_eff"][RHO_SIN_SHIFT],
        int(seleccion["m"]),
    )
    resumen["t_equivalente"] = round(
        umbral_equivalente(float(seleccion["t"]), rho_usado_rho1), 12
    )
    mi, ti = int(seleccion["indice_m"]), int(seleccion["indice_t"])
    resumen["tasas_por_rho"] = {
        str(rho): round(float(evaluacion["tasas"][mi, ti, ri]), 12)
        for ri, rho in enumerate(RHOS)
    }
    return resumen


def _seleccionar_etiqueta(
    p_sel: np.ndarray,
    y_sel: np.ndarray,
    pi_train: float,
    umbral_v03: float,
    pred_v03_sel: np.ndarray,
) -> dict[str, Any]:
    """Selecciona (m, t) con y sin guardarrail para una etiqueta.

    Args:
        p_sel: probabilidades en las filas de seleccion.
        y_sel: etiqueta binaria en las filas de seleccion.
        pi_train: prevalencia de referencia.
        umbral_v03: umbral congelado de la etiqueta.
        pred_v03_sel: predicciones de V03 en las filas de seleccion.

    Returns:
        Diccionario con superficies, EM, n_eff y ambas selecciones.
    """
    evaluacion = evaluar_superficie(p_sel, y_sel, pi_train, pred_v03_sel)
    con_guardarrail = seleccionar_en_superficie(
        evaluacion["superficie"],
        evaluacion["tasas"],
        evaluacion["tasa_v03"],
        umbral_v03,
        guardarrail=True,
    )
    sin_restriccion = seleccionar_en_superficie(
        evaluacion["superficie"],
        evaluacion["tasas"],
        evaluacion["tasa_v03"],
        umbral_v03,
        guardarrail=False,
    )
    superficies = {
        "f1_ponderada_m_t_rho": np.round(evaluacion["superficie"], 10).tolist(),
        "minimax_sobre_rho": np.round(evaluacion["superficie"].min(axis=2), 10).tolist(),
        "factible_tasa_rho1": (
            np.abs(evaluacion["tasas"][:, :, INDICE_RHO1] - evaluacion["tasa_v03"]) <= TOL_TASA
        ).tolist(),
    }
    return {
        "superficie": superficies,
        "em_seleccion": {str(rho): resumen_em(res) for rho, res in evaluacion["em"].items()},
        "n_eff_seleccion": {str(rho): evaluacion["n_eff"][rho] for rho in RHOS},
        "tasa_v03_seleccion": evaluacion["tasa_v03"],
        "seleccion": _resumen_seleccion(con_guardarrail, evaluacion),
        "optimo_sin_restriccion": _resumen_seleccion(sin_restriccion, evaluacion),
    }


def ejecutar_lofo(datos: DatosGate0, pliegues: Sequence[int]) -> ResultadoLOFO:
    """Ejecuta la simulacion LOFO completa para los pliegues pedidos.

    Para cada pliegue k, la seleccion (m, t) usa las filas de los otros 4 pliegues; la
    aplicacion (EM incluido) usa las filas del pliegue k. La convencion de estimar el EM
    de aplicacion con las filas del propio pliegue queda declarada en el informe.

    Args:
        datos: datos verificados de Gate 0.
        pliegues: pliegues a evaluar (1..5).

    Returns:
        ResultadoLOFO con predicciones y registros por pliegue.
    """
    n = datos.y.shape[0]
    n_rho = len(RHOS)
    pred_completa = np.zeros((n, len(ETIQUETAS), n_rho), dtype=bool)
    pred_solo_umbral = np.zeros_like(pred_completa)
    pred_v03 = datos.pred_v03.astype(bool)
    registros: dict[int, dict[str, Any]] = {}
    for k in pliegues:
        filas_sel = np.flatnonzero(datos.folde != k)
        filas_eval = np.flatnonzero(datos.folde == k)
        registro: dict[str, Any] = {
            "pliegue": int(k),
            "filas_seleccion": int(filas_sel.size),
            "filas_evaluacion": int(filas_eval.size),
            "etiquetas": {},
        }
        for c, etiqueta in enumerate(ETIQUETAS):
            pi_train = float(datos.prevalencias[c])
            umbral_v03 = float(datos.umbrales[c])
            seleccion = _seleccionar_etiqueta(
                datos.p_v03[filas_sel, c],
                datos.y[filas_sel, c],
                pi_train,
                umbral_v03,
                pred_v03[filas_sel, c],
            )
            aplicacion = aplicar_regla(
                datos.p_v03[filas_eval, c],
                datos.y[filas_eval, c],
                umbral_v03,
                int(seleccion["seleccion"]["m"]),
                float(seleccion["seleccion"]["t"]),
                pi_train,
                pred_v03[filas_eval, c],
            )
            pred_completa[filas_eval, c, :] = aplicacion["pred_completa"]
            pred_solo_umbral[filas_eval, c, :] = aplicacion["pred_solo_umbral"]
            registro["etiquetas"][etiqueta] = {
                "pi_train": pi_train,
                "umbral_v03": umbral_v03,
                **seleccion,
                "aplicacion": {
                    "por_rho": {str(rho): aplicacion["por_rho"][rho] for rho in RHOS},
                    "em": {str(rho): aplicacion["em"][rho] for rho in RHOS},
                    "n_eff": {str(rho): aplicacion["n_eff"][rho] for rho in RHOS},
                    "delta_medio": aplicacion["delta_medio"],
                    "delta_medio_em": aplicacion["delta_medio_em"],
                },
            }
        etiquetas_k = registro["etiquetas"]
        registro["delta_pliegue"] = float(
            np.mean(
                [
                    etiquetas_k[etiqueta]["aplicacion"]["delta_medio"]
                    for etiqueta in ETIQUETAS
                ]
            )
        )
        registro["delta_pliegue_por_rho"] = {
            str(rho): float(
                np.mean(
                    [
                        etiquetas_k[etiqueta]["aplicacion"]["por_rho"][str(rho)]["delta"]
                        for etiqueta in ETIQUETAS
                    ]
                )
            )
            for rho in RHOS
        }
        registro["delta_pliegue_por_etiqueta"] = {
            etiqueta: etiquetas_k[etiqueta]["aplicacion"]["delta_medio"]
            for etiqueta in ETIQUETAS
        }
        registros[int(k)] = registro
        log(
            f"  pliegue {k}: seleccion "
            + " | ".join(
                f"{etiqueta} m={etiquetas_k[etiqueta]['seleccion']['m']} "
                f"t={etiquetas_k[etiqueta]['seleccion']['t']:.2f}"
                for etiqueta in ETIQUETAS
            )
            + f" | delta {registro['delta_pliegue']:+.6f}"
        )
    return ResultadoLOFO(
        pred_completa=pred_completa,
        pred_solo_umbral=pred_solo_umbral,
        pred_v03=pred_v03,
        registros=registros,
    )


# ===== SECCION 4: metricas globales, controles, bootstrap y gate =====


def _f1_global_por_rho(
    y_columna: np.ndarray,
    predicciones_por_rho: Sequence[np.ndarray],
    rhos: Sequence[float],
) -> dict[float, float]:
    """Calcula la F1 ponderada global de una etiqueta para cada rho.

    Args:
        y_columna: etiqueta binaria de las filas evaluadas.
        predicciones_por_rho: secuencia de predicciones binarias (una por rho).
        rhos: razones de prevalencia simuladas.

    Returns:
        Diccionario rho -> F1 ponderada.
    """
    return {
        rho: f1_ponderada(
            y_columna, predicciones_por_rho[ri], pesos_importancia(y_columna, rho)
        )
        for ri, rho in enumerate(rhos)
    }


def _precision_recall_global_por_rho(
    y_columna: np.ndarray,
    predicciones_por_rho: Sequence[np.ndarray],
    rhos: Sequence[float],
) -> tuple[dict[float, float], dict[float, float]]:
    """Calcula precision y recall ponderadas de una etiqueta para cada rho.

    Args:
        y_columna: etiqueta binaria de las filas evaluadas.
        predicciones_por_rho: secuencia de predicciones binarias (una por rho).
        rhos: razones de prevalencia simuladas.

    Returns:
        Tupla (precision por rho, recall por rho).
    """
    precision: dict[float, float] = {}
    recall: dict[float, float] = {}
    for ri, rho in enumerate(rhos):
        w = pesos_importancia(y_columna, rho)
        precision[rho] = precision_ponderada(y_columna, predicciones_por_rho[ri], w)
        recall[rho] = recall_ponderada(y_columna, predicciones_por_rho[ri], w)
    return precision, recall


def _delta_por_pliegue(
    datos: DatosGate0,
    resultado: ResultadoLOFO,
    pliegues: Sequence[int],
    rhos: Sequence[float],
) -> dict[str, Any]:
    """Calcula el delta por pliegue (promedio de las 5 rho) y su media.

    Args:
        datos: datos de Gate 0.
        resultado: predicciones LOFO.
        pliegues: pliegues evaluados.
        rhos: razones de prevalencia simuladas.

    Returns:
        Diccionario con deltas por pliegue, media y pliegues mejorados.
    """
    deltas: dict[str, float] = {}
    for k in pliegues:
        filas = datos.folde == k
        valores = []
        for c in range(len(ETIQUETAS)):
            for ri, rho in enumerate(rhos):
                y_col = datos.y[filas, c]
                w = pesos_importancia(y_col, rho)
                f1_cand = f1_ponderada(y_col, resultado.pred_completa[filas, c, ri], w)
                f1_v03 = f1_ponderada(y_col, resultado.pred_v03[filas, c], w)
                valores.append(f1_cand - f1_v03)
        deltas[str(k)] = float(np.mean(valores))
    return {
        "delta_por_pliegue": deltas,
        "delta_medio_pliegues": float(np.mean(list(deltas.values()))),
        "pliegues_mejorados": int(sum(1 for valor in deltas.values() if valor > 0.0)),
    }


def calcular_metricas(
    datos: DatosGate0,
    resultado: ResultadoLOFO,
    mask: np.ndarray,
    pliegues: Sequence[int],
) -> dict[str, Any]:
    """Calcula deltas globales, descomposicion, guardarrail y diagnosticos.

    Args:
        datos: datos de Gate 0.
        resultado: predicciones LOFO.
        mask: mascara booleana de filas evaluadas.
        pliegues: pliegues evaluados.

    Returns:
        Diccionario con f1 por regla, deltas, controles, tasas y resumenes.
    """
    y_eval = datos.y[mask]
    pred_c = resultado.pred_completa[mask]
    pred_s = resultado.pred_solo_umbral[mask]
    pred_v = resultado.pred_v03[mask]
    f1_cand: dict[str, dict[float, float]] = {}
    f1_v03: dict[str, dict[float, float]] = {}
    f1_solo: dict[str, dict[float, float]] = {}
    precision_cand: dict[str, dict[float, float]] = {}
    precision_v03: dict[str, dict[float, float]] = {}
    precision_solo: dict[str, dict[float, float]] = {}
    recall_cand: dict[str, dict[float, float]] = {}
    recall_v03: dict[str, dict[float, float]] = {}
    recall_solo: dict[str, dict[float, float]] = {}
    for c, etiqueta in enumerate(ETIQUETAS):
        predicciones_cand = [pred_c[:, c, ri] for ri in range(len(RHOS))]
        predicciones_solo = [pred_s[:, c, ri] for ri in range(len(RHOS))]
        f1_cand[etiqueta] = _f1_global_por_rho(y_eval[:, c], predicciones_cand, RHOS)
        f1_v03[etiqueta] = _f1_global_por_rho(
            y_eval[:, c], [pred_v[:, c]] * len(RHOS), RHOS
        )
        f1_solo[etiqueta] = _f1_global_por_rho(y_eval[:, c], predicciones_solo, RHOS)
        precision_cand[etiqueta], recall_cand[etiqueta] = _precision_recall_global_por_rho(
            y_eval[:, c], predicciones_cand, RHOS
        )
        precision_v03[etiqueta], recall_v03[etiqueta] = _precision_recall_global_por_rho(
            y_eval[:, c], [pred_v[:, c]] * len(RHOS), RHOS
        )
        precision_solo[etiqueta], recall_solo[etiqueta] = _precision_recall_global_por_rho(
            y_eval[:, c], predicciones_solo, RHOS
        )
    delta_cand_v03 = {
        rho: float(
            np.mean([f1_cand[etiqueta][rho] - f1_v03[etiqueta][rho] for etiqueta in ETIQUETAS])
        )
        for rho in RHOS
    }
    delta_em = {
        rho: float(
            np.mean([f1_cand[etiqueta][rho] - f1_solo[etiqueta][rho] for etiqueta in ETIQUETAS])
        )
        for rho in RHOS
    }
    delta_umbral = {
        rho: float(
            np.mean([f1_solo[etiqueta][rho] - f1_v03[etiqueta][rho] for etiqueta in ETIQUETAS])
        )
        for rho in RHOS
    }
    delta_rho1 = float(delta_cand_v03[RHO_SIN_SHIFT])
    sanity_rho1 = {str(k): None for k in pliegues}
    for k in pliegues:
        filas = datos.folde == k
        valores = []
        for c in range(len(ETIQUETAS)):
            y_col = datos.y[filas, c]
            w = pesos_importancia(y_col, RHO_SIN_SHIFT)
            f1_c = f1_ponderada(y_col, resultado.pred_completa[filas, c, INDICE_RHO1], w)
            f1_v = f1_ponderada(y_col, resultado.pred_v03[filas, c], w)
            valores.append(f1_c - f1_v)
        sanity_rho1[str(k)] = float(np.mean(valores))
    desviaciones = {}
    tasas_cand = {}
    tasas_v03 = {}
    for c, etiqueta in enumerate(ETIQUETAS):
        tasa_c = float(pred_c[:, c, INDICE_RHO1].mean())
        tasa_v = float(pred_v[:, c].mean())
        tasas_cand[etiqueta] = tasa_c
        tasas_v03[etiqueta] = tasa_v
        desviaciones[etiqueta] = {
            "tasa_candidato_rho1": round(tasa_c, 12),
            "tasa_v03": round(tasa_v, 12),
            "desviacion": round(tasa_c - tasa_v, 12),
            "cumple": bool(abs(tasa_c - tasa_v) <= TOL_TASA),
        }
    rescate = int(np.sum((pred_c[:, :, INDICE_RHO1] == 1) & (pred_v == 0)))
    dano = int(np.sum((pred_c[:, :, INDICE_RHO1] == 0) & (pred_v == 1)))
    delta_fold = _delta_por_pliegue(datos, resultado, pliegues, RHOS)
    return {
        "filas_evaluadas": int(mask.sum()),
        "f1_por_regla": {
            nombre: {
                etiqueta: {str(rho): valor for rho, valor in tabla.items()}
                for etiqueta, tabla in tabla_etiquetas.items()
            }
            for nombre, tabla_etiquetas in (
                ("candidato", f1_cand),
                ("v03", f1_v03),
                ("solo_umbral", f1_solo),
            )
        },
        "precision_por_regla": {
            nombre: {
                etiqueta: {str(rho): valor for rho, valor in tabla.items()}
                for etiqueta, tabla in tabla_etiquetas.items()
            }
            for nombre, tabla_etiquetas in (
                ("candidato", precision_cand),
                ("v03", precision_v03),
                ("solo_umbral", precision_solo),
            )
        },
        "recall_por_regla": {
            nombre: {
                etiqueta: {str(rho): valor for rho, valor in tabla.items()}
                for etiqueta, tabla in tabla_etiquetas.items()
            }
            for nombre, tabla_etiquetas in (
                ("candidato", recall_cand),
                ("v03", recall_v03),
                ("solo_umbral", recall_solo),
            )
        },
        "delta_global_por_rho": {str(rho): delta_cand_v03[rho] for rho in RHOS},
        "delta_medio": float(np.mean(list(delta_cand_v03.values()))),
        "delta_por_etiqueta_rho1": {
            etiqueta: float(f1_cand[etiqueta][RHO_SIN_SHIFT] - f1_v03[etiqueta][RHO_SIN_SHIFT])
            for etiqueta in ETIQUETAS
        },
        **delta_fold,
        "sanity_sin_shift_rho1": {
            "delta": delta_rho1,
            "delta_por_pliegue": sanity_rho1,
            "cumple": bool(delta_rho1 >= 0.0),
        },
        "descomposicion": {
            "componente_em_por_rho": {str(rho): delta_em[rho] for rho in RHOS},
            "componente_em_medio": float(np.mean(list(delta_em.values()))),
            "componente_em_medio_pliegues": float(
                np.mean(
                    [
                        resultado.registros[k]["etiquetas"][etiqueta]["aplicacion"]["delta_medio_em"]
                        for k in pliegues
                        for etiqueta in ETIQUETAS
                    ]
                )
            ),
            "componente_umbral_por_rho": {str(rho): delta_umbral[rho] for rho in RHOS},
            "componente_umbral_medio": float(np.mean(list(delta_umbral.values()))),
        },
        "tasas": {
            "candidato": {
                etiqueta: {
                    str(rho): float(pred_c[:, c, ri].mean()) for ri, rho in enumerate(RHOS)
                }
                for c, etiqueta in enumerate(ETIQUETAS)
            },
            "v03": tasas_v03,
            "solo_umbral": {
                etiqueta: {
                    str(rho): float(pred_s[:, c, ri].mean()) for ri, rho in enumerate(RHOS)
                }
                for c, etiqueta in enumerate(ETIQUETAS)
            },
        },
        "guardarrail_tasa": {
            "tolerancia": TOL_TASA,
            "por_etiqueta": desviaciones,
            "max_desviacion": float(max(abs(v["desviacion"]) for v in desviaciones.values())),
            "cumple": bool(all(v["cumple"] for v in desviaciones.values())),
        },
        "rescate_dano_rho1": {"rescate": rescate, "dano": dano, "neto": rescate - dano},
        "pi_simulada": {
            etiqueta: {
                str(rho): float(
                    rho * datos.prevalencias[c]
                    / (rho * datos.prevalencias[c] + (1.0 - datos.prevalencias[c]))
                )
                for rho in RHOS
            }
            for c, etiqueta in enumerate(ETIQUETAS)
        },
        "n_eff_dev": {
            etiqueta: {
                str(rho): tamano_efectivo_kish(
                    pesos_importancia(datos.y[:, c], rho)
                )
                for rho in RHOS
            }
            for c, etiqueta in enumerate(ETIQUETAS)
        },
    }


def bootstrap_delta(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    y: np.ndarray,
    delta_esperado: float | None = None,
    repeticiones: int = REPETICIONES_BOOTSTRAP,
    semilla: int = SEMILLA_BOOTSTRAP,
) -> dict[str, Any]:
    """IC95 bootstrap pareado por filas del delta de F1 ponderada (con pesos).

    El estadistico es la media sobre etiquetas y rho de la diferencia de F1 ponderada
    entre A y B en cada remuestreo de filas. Los pesos w_i = rho si y_i = 1 y 1 si
    y_i = 0 viajan con la fila. Las sumas por remuestreo se calculan por lotes con
    algebra matricial para no materializar la matriz completa de indices.

    Args:
        pred_a: predicciones de A (n, 3, n_rho) binarias.
        pred_b: predicciones de B (n, 3, n_rho) binarias.
        y: etiquetas (n, 3).
        delta_esperado: si se indica, se verifica que el estadistico observado coincida.
        repeticiones: numero de remuestreos.
        semilla: semilla del generador.

    Returns:
        Diccionario con delta observado, IC95, mediana, proporcion positiva y tiempos.

    Raises:
        RuntimeError: si el estadistico observado no reproduce delta_esperado.
    """
    inicio = time.perf_counter()
    n = y.shape[0]
    n_rho = pred_a.shape[2]
    a = pred_a.astype(np.float64)
    b = pred_b.astype(np.float64)
    columnas: list[np.ndarray] = []
    for c in range(len(ETIQUETAS)):
        yc = y[:, c].astype(np.float64)
        columnas.append(yc)
        for ri in range(n_rho):
            columnas.append(yc * a[:, c, ri])
        for ri in range(n_rho):
            columnas.append(a[:, c, ri])
        for ri in range(n_rho):
            columnas.append(yc * b[:, c, ri])
        for ri in range(n_rho):
            columnas.append(b[:, c, ri])
    vectores = np.column_stack(columnas)

    def deltas_desde_sumas(sumas: np.ndarray) -> np.ndarray:
        s = sumas.reshape(sumas.shape[0], len(ETIQUETAS), 1 + 4 * n_rho)
        n_pos = s[:, :, 0]
        deltas_rho = []
        for ri, rho in enumerate(RHOS[:n_rho]):
            tp_a = s[:, :, 1 + ri]
            npred_a = s[:, :, 1 + n_rho + ri]
            fp_a = npred_a - tp_a
            tp_b = s[:, :, 1 + 2 * n_rho + ri]
            npred_b = s[:, :, 1 + 3 * n_rho + ri]
            fp_b = npred_b - tp_b
            den_a = rho * n_pos + rho * tp_a + fp_a
            den_b = rho * n_pos + rho * tp_b + fp_b
            f1_a = np.where(den_a > 0.0, 2.0 * rho * tp_a / np.maximum(den_a, EPS), 0.0)
            f1_b = np.where(den_b > 0.0, 2.0 * rho * tp_b / np.maximum(den_b, EPS), 0.0)
            deltas_rho.append((f1_a - f1_b).mean(axis=1))
        return np.mean(np.column_stack(deltas_rho), axis=1)

    observado = float(deltas_desde_sumas(vectores.sum(axis=0, keepdims=True))[0])
    if delta_esperado is not None and abs(observado - delta_esperado) > 1e-9:
        raise RuntimeError(
            f"STOP: el bootstrap observado {observado!r} no reproduce {delta_esperado!r}"
        )
    generador = np.random.default_rng(semilla)
    valores = np.empty(repeticiones, dtype=np.float64)
    for inicio_lote in range(0, repeticiones, LOTE_BOOTSTRAP):
        fin = min(inicio_lote + LOTE_BOOTSTRAP, repeticiones)
        tamano = fin - inicio_lote
        indices = generador.integers(0, n, size=(tamano, n))
        plano = indices.ravel() + np.repeat(np.arange(tamano) * n, n)
        seleccion = np.bincount(plano, minlength=tamano * n).reshape(tamano, n)
        sumas = seleccion.astype(np.float64) @ vectores
        valores[inicio_lote:fin] = deltas_desde_sumas(sumas)
    cuantiles = np.quantile(valores, [0.025, 0.5, 0.975])
    limite_inferior = float(cuantiles[0])
    limite_superior = float(cuantiles[2])
    return {
        "delta_observado": round(observado, 12),
        "ic95": [round(limite_inferior, 12), round(limite_superior, 12)],
        "mediana_bootstrap": round(float(cuantiles[1]), 12),
        "proporcion_positiva": float(np.mean(valores > 0.0)),
        "excluye_cero": bool(limite_inferior > 0.0 or limite_superior < 0.0),
        "ic_inferior_positivo": bool(limite_inferior > 0.0),
        "repeticiones": repeticiones,
        "semilla": semilla,
        "tiempo_segundos": round(time.perf_counter() - inicio, 2),
    }


def evaluar_gate(
    metricas: dict[str, Any],
    bootstrap_principal: dict[str, Any],
    bootstrap_componente: dict[str, Any],
    selecciones_factibles: bool,
    em_convergido: bool,
    gate0_ok: bool,
) -> dict[str, Any]:
    """Evalua el gate unico de EXP-05 y emite la decision.

    Criterios: delta medio >= +0.002; >= 4/5 pliegues LOFO positivos; IC95 pareado
    (con pesos) con limite inferior > 0; guardarrail de tasa cumplido; componente EM
    positiva; sanidad sin shift >= 0; selecciones factibles y EM convergido.

    Args:
        metricas: metricas globales y controles.
        bootstrap_principal: IC del delta candidato vs V03.
        bootstrap_componente: IC del delta candidato vs solo-umbral (componente EM).
        selecciones_factibles: si todas las selecciones cumplieron el guardarrail.
        em_convergido: si todos los EM convergieron a la tolerancia declarada.
        gate0_ok: si Gate 0 paso (siempre True al llegar aqui).

    Returns:
        Diccionario con criterios, motivos, si pasa y texto de decision.
    """
    criterios = {
        "gate0_ok": bool(gate0_ok),
        "delta_medio_ge_0_002": bool(metricas["delta_medio"] >= GATE_DELTA),
        "pliegues_mejorados_ge_4": bool(metricas["pliegues_mejorados"] >= GATE_PLIEGUES),
        "ic95_excluye_cero": bool(bootstrap_principal["excluye_cero"]),
        "ic95_inferior_positivo": bool(bootstrap_principal["ic_inferior_positivo"]),
        "guardarrail_tasa": bool(metricas["guardarrail_tasa"]["cumple"]),
        "componente_em_positiva": bool(
            metricas["descomposicion"]["componente_em_medio"] > 0.0
        ),
        "sanidad_sin_shift_ge_0": bool(metricas["sanity_sin_shift_rho1"]["cumple"]),
        "selecciones_factibles": bool(selecciones_factibles),
        "em_convergido": bool(em_convergido),
    }
    motivos = []
    if not criterios["delta_medio_ge_0_002"]:
        motivos.append(
            f"delta medio {metricas['delta_medio']:+.6f} < +0.002"
        )
    if not criterios["pliegues_mejorados_ge_4"]:
        motivos.append(f"pliegues mejorados {metricas['pliegues_mejorados']}/5 < 4/5")
    if not criterios["ic95_excluye_cero"]:
        motivos.append(f"IC95 {bootstrap_principal['ic95']} incluye cero")
    elif not criterios["ic95_inferior_positivo"]:
        motivos.append(f"IC95 {bootstrap_principal['ic95']} excluye cero pero no es positivo")
    if not criterios["guardarrail_tasa"]:
        motivos.append(
            f"guardarrail de tasa: max desviacion "
            f"{metricas['guardarrail_tasa']['max_desviacion']:.6f} > {TOL_TASA:g}"
        )
    if not criterios["componente_em_positiva"]:
        motivos.append(
            "componente EM no positiva "
            f"({metricas['descomposicion']['componente_em_medio']:+.6f}): la ganancia es solo de umbral"
        )
    if not criterios["sanidad_sin_shift_ge_0"]:
        motivos.append(
            f"sanidad sin shift: delta rho=1.00 {metricas['sanity_sin_shift_rho1']['delta']:+.6f} < 0"
        )
    if not criterios["selecciones_factibles"]:
        motivos.append("alguna seleccion no tuvo candidato factible con el guardarrail de tasa")
    if not criterios["em_convergido"]:
        motivos.append("algun EM no convergio a la tolerancia 1e-10")
    pasa = all(criterios.values())
    if pasa:
        decision = (
            "ELEGIBLE PARA GATE 2 (segunda particion, receta congelada); sin submission automatica"
        )
        etiqueta_cierre = ""
    elif not criterios["componente_em_positiva"]:
        decision = (
            "CERRAR LA FAMILIA EM/threshold: la ganancia viene solo del cambio de umbral "
            "(familia de umbrales cerrada)"
        )
        etiqueta_cierre = "CLOSED: EM/threshold (ganancia solo de umbral)"
    else:
        decision = f"CERRAR LA FAMILIA EM/threshold: gate no cumplido ({'; '.join(motivos)})"
        etiqueta_cierre = "CLOSED: EM/threshold (gate no cumplido)"
    return {
        "evaluable": True,
        "criterios": criterios,
        "delta_medio": metricas["delta_medio"],
        "pliegues_mejorados": metricas["pliegues_mejorados"],
        "ic95": bootstrap_principal["ic95"],
        "componente_em_medio": metricas["descomposicion"]["componente_em_medio"],
        "ic95_componente_em": bootstrap_componente["ic95"],
        "guardarrail_max_desviacion": metricas["guardarrail_tasa"]["max_desviacion"],
        "pasa": bool(pasa),
        "motivos": motivos,
        "decision": decision,
        "etiqueta_cierre": etiqueta_cierre,
    }


# ===== SECCION 5: salidas (parciales, informes y CSV) =====


def escribir_parcial(numero: int, registro: dict[str, Any], humo: bool) -> Path:
    """Escribe el parcial JSON de un pliegue.

    Args:
        numero: numero de pliegue.
        registro: registro de la simulacion de ese pliegue.
        humo: si True el parcial lleva el sufijo _humo.

    Returns:
        Ruta escrita.
    """
    directorio: Path = CONFIG["parciales_dir"]
    directorio.mkdir(parents=True, exist_ok=True)
    sufijo = "_humo" if humo else ""
    ruta = directorio / f"p84_05_p{numero}{sufijo}.json"
    carga = {
        "experiment_id": "P84-05",
        "modo_humo": humo,
        "pliegue": numero,
        "registro": registro,
    }
    escribir_json(ruta, carga)
    return ruta


def upsert_experiments_csv(fila: dict[str, str]) -> dict[str, Any]:
    """Inserta o reemplaza la fila P84-05 en experiments.csv (23 columnas).

    Valida la cabecera, el ancho de todas las filas, la ausencia de ids duplicados y
    que la fila defina exactamente las 23 columnas antes de escribir. Escribe
    utf-8-sig y no duplica ids.

    Args:
        fila: valores por columna de la fila P84-05.

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
        raise SystemExit(f"STOP: faltan columnas en la fila P84-05: {faltantes}")
    sobrantes = sorted(set(fila) - set(COLUMNAS_EXPERIMENTS))
    if sobrantes:
        raise SystemExit(f"STOP: columnas desconocidas en la fila P84-05: {sobrantes}")
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


def construir_fila_experiments(informe: dict[str, Any], segundos_totales: float) -> dict[str, str]:
    """Construye la fila P84-05 para experiments.csv.

    Args:
        informe: informe completo de la corrida.
        segundos_totales: tiempo total de la corrida.

    Returns:
        Diccionario columna -> valor.
    """
    metricas = informe["metricas"]
    gate = informe["gate"]
    bootstrap = informe["bootstrap"]
    f1_candidato_rho1 = float(
        np.mean(
            [
                metricas["f1_por_regla"]["candidato"][etiqueta][str(RHO_SIN_SHIFT)]
                for etiqueta in ETIQUETAS
            ]
        )
    )
    descomposicion = metricas["descomposicion"]
    resultado = {
        "delta_medio": metricas["delta_medio"],
        "delta_medio_pliegues": metricas["delta_medio_pliegues"],
        "delta_por_pliegue": metricas["delta_por_pliegue"],
        "delta_global_por_rho": metricas["delta_global_por_rho"],
        "pliegues_mejorados": metricas["pliegues_mejorados"],
        "ic95_bootstrap": bootstrap["ic95"],
        "semilla_bootstrap": bootstrap["semilla"],
        "repeticiones_bootstrap": bootstrap["repeticiones"],
        "componente_em_medio": descomposicion["componente_em_medio"],
        "componente_em_por_rho": descomposicion["componente_em_por_rho"],
        "ic95_componente_em": informe["bootstrap_componente_em"]["ic95"],
        "sanity_sin_shift_rho1": metricas["sanity_sin_shift_rho1"],
        "guardarrail_tasa": metricas["guardarrail_tasa"],
        "tasas_candidato": metricas["tasas"]["candidato"],
        "tasas_v03": metricas["tasas"]["v03"],
        "rescate_dano_rho1": metricas["rescate_dano_rho1"],
        "f1_candidato_rho1": f1_candidato_rho1,
        "f1_v03_referencia": V03_CCV,
        "seleccion_lofo": {
            str(numero): {
                etiqueta: {
                    "m": informe["simulacion"]["folds"][str(numero)]["etiquetas"][etiqueta][
                        "seleccion"
                    ]["m"],
                    "t": informe["simulacion"]["folds"][str(numero)]["etiquetas"][etiqueta][
                        "seleccion"
                    ]["t"],
                    "minimax": informe["simulacion"]["folds"][str(numero)]["etiquetas"][etiqueta][
                        "seleccion"
                    ]["minimax"],
                }
                for etiqueta in ETIQUETAS
            }
            for numero in sorted(informe["simulacion"]["folds"], key=int)
        },
        "gate": gate,
        "informe_json": str(CONFIG["informe_json"].relative_to(PROJECT_ROOT)),
        "informe_md": str(CONFIG["informe_md"].relative_to(PROJECT_ROOT)),
    }
    return {
        "experiment_id": "P84-05",
        "competition": str(CONFIG["competencia"]),
        "question": (
            "Mejora la correccion EM de odds por cambio de prevalencia, encogida con shrinkage "
            "y con umbral seleccionado por minimax LOFO, sobre el OOF congelado de V03 "
            "(1483 filas de desarrollo), frente a V03 con umbrales congelados 0.44/0.51/0.46?"
        ),
        "hypothesis": (
            "Expectativa muy baja (sliver unico de la familia calibracion/umbral): el guardarrail "
            "de tasa a rho=1.00 y el criterio minimax pueden dejar la ganancia en cero; si la "
            "mejora proviene solo del cambio de umbral, la familia EM se cierra."
        ),
        "preprocessing": (
            "scores OOF congelados; ponderacion de importancia w_i = rho si y_i = 1 y 1 si "
            "y_i = 0; sin reentrenar modelos; sin holdout y sin test (toxic_test.csv nunca se abre)"
        ),
        "feature_groups": "OOF V03: proba (1483x3) + y_true + pliegues",
        "representation": (
            "probabilidades OOF congeladas de V03 con simulacion analitica del cambio de "
            "prevalencia (supuesto p(x|y) invariable)"
        ),
        "model": (
            "correccion EM de odds con shrinkage geometrico (m en {0,25,50,100}) y umbral "
            "t en 0.20-0.80 paso 0.01"
        ),
        "hyperparameters": json.dumps(
            {
                "rho": list(RHOS),
                "m": list(M_GRID),
                "t": {"min": 0.20, "max": 0.80, "paso": 0.01, "n": int(T_GRID.size)},
                "rho_inicial": RHO_INICIAL,
                "tolerancia_em": TOL_EM,
                "max_iter_em": MAX_ITER_EM,
                "shrinkage": "rho_usado = exp(s*log rho_hat), s = n_eff/(n_eff+m), n_eff Kish",
                "criterio": "minimax sobre las 5 rho; empate menor m y despues t mas cercano al congelado",
                "guardarrail_tasa": TOL_TASA,
                "bootstrap": {"repeticiones": REPETICIONES_BOOTSTRAP, "semilla": SEMILLA_BOOTSTRAP},
            },
            ensure_ascii=True,
        ),
        "validation": (
            "LOFO sobre los 5 pliegues canonicos: (m,t) se selecciona con los otros 4 pliegues "
            "y se aplica al pliegue k; delta por pliegue promedio de las 5 rho; bootstrap "
            "pareado por filas 10000/20260926 con pesos"
        ),
        "f1": f"{f1_candidato_rho1:.6f}",
        "macro_f1": "",
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": f"{segundos_totales:.2f} s (numpy puro, sin entrenamiento)",
        "dimension": "1483 filas x 3 etiquetas; rejilla m=4 x t=61 x rho=5",
        "result": json.dumps(resultado, ensure_ascii=True),
        "interpretation": (
            f"delta medio {metricas['delta_medio']:+.6f} (umbral +0.002); pliegues mejorados "
            f"{metricas['pliegues_mejorados']}/5; IC95 {bootstrap['ic95']}; guardarrail de tasa "
            f"{'OK' if metricas['guardarrail_tasa']['cumple'] else 'NO'}; componente EM "
            f"{descomposicion['componente_em_medio']:+.6f}"
        ),
        "decision": gate["decision"],
        "date": str(CONFIG["fecha"]),
        "script": str(CONFIG["script"]),
        "notes": (
            "EXP-05 del preregistro P84 (docs/p84_spec.md seccion 6). Contadores dev 1483 / "
            "holdout 0 / test 0; la fila se inserta o reemplaza por experiment_id (idempotente); "
            "RUNLOG.md y DECISIONS.md quedan a cargo del trainer."
        ),
    }


def _tabla_md(cabecera: Sequence[str], filas: Sequence[Sequence[Any]]) -> str:
    """Construye una tabla Markdown ASCII.

    Args:
        cabecera: nombres de columna.
        filas: filas de la tabla.

    Returns:
        Texto de la tabla terminado en salto de linea.
    """
    lineas = [
        "| " + " | ".join(str(valor) for valor in cabecera) + " |",
        "|" + "|".join("---" for _ in cabecera) + "|",
    ]
    for fila in filas:
        lineas.append("| " + " | ".join(str(valor) for valor in fila) + " |")
    return "\n".join(lineas) + "\n"


def construir_informe_md(informe: dict[str, Any]) -> str:
    """Construye el informe Markdown ASCII de la corrida.

    Args:
        informe: informe completo de la corrida.

    Returns:
        Texto Markdown.
    """
    datos = informe["gate0"]
    metricas = informe["metricas"]
    gate = informe["gate"]
    lineas: list[str] = []
    lineas.append("# P84-05: correccion de prevalencia EM con shrinkage (simulacion OOF)")
    lineas.append("")
    lineas.append(f"- Fecha: {informe['fecha']}; script: `{informe['script']}`.")
    lineas.append(
        f"- Modo: {'humo' if informe['modo_humo'] else 'completo'}; corrida completa: "
        f"{informe['corrida_completa']}; pliegues: {informe['pliegues']}."
    )
    lineas.append(
        "- Preregistro: docs/p84_spec.md seccion 6 + docs/p84_preregistro.md; sin test, sin "
        "holdout, sin submissions."
    )
    lineas.append("")
    lineas.append("## 1. Supuesto y estimando")
    lineas.append("")
    lineas.append(
        "Pesos de importancia w_i = rho si y_i = 1 y w_i = 1 si y_i = 0 sobre las filas del "
        "OOF; estimando: F1 ponderada 2*sum(w*y*pred)/(sum(w*y)+sum(w*pred)) por etiqueta y "
        "su media. Simulacion analitica determinista que asume p(x|y) invariable."
    )
    lineas.append("")
    lineas.append("## 2. Gate 0")
    lineas.append("")
    identidad = datos["identidad_v03"]
    lineas.append(
        f"- sha256 del npz verificado; corrected_cv recalculado {identidad['corrected_cv_recalculado']!r} "
        f"(referencia 0.7772114525165721); V03@0.5 recalculado "
        f"{identidad['v03_f1_at_05_recalculado']!r} (referencia 0.771778476256122)."
    )
    lineas.append(
        f"- Contadores: dev {datos['contadores']['dev_filas_leidas']} / holdout "
        f"{datos['contadores']['holdout_filas_usadas']} / test "
        f"{datos['contadores']['test_filas_usadas']}; toxic_test.csv nunca se abre."
    )
    lineas.append(
        f"- Pliegues: tamanos {datos['particiones']['pliegues_tamanos']}; umbrales "
        f"{identidad['umbrales']}."
    )
    lineas.append("")
    lineas.append("## 3. Seleccion LOFO (minimax con guardarrail)")
    lineas.append("")
    filas = []
    for numero in sorted(informe["simulacion"]["folds"], key=int):
        registro = informe["simulacion"]["folds"][numero]
        for etiqueta in ETIQUETAS:
            seleccion = registro["etiquetas"][etiqueta]["seleccion"]
            filas.append(
                [
                    numero,
                    etiqueta,
                    seleccion["m"],
                    f"{seleccion['t']:.2f}",
                    f"{seleccion['t_equivalente']:.4f}",
                    f"{seleccion['minimax']:.6f}",
                    f"{seleccion['tasa_rho1']:.6f}",
                    f"{seleccion['tasa_v03_rho1']:.6f}",
                    f"{seleccion['desviacion_tasa_rho1']:+.6f}",
                    seleccion["factible"],
                ]
            )
    lineas.append(
        _tabla_md(
            [
                "pliegue",
                "etiqueta",
                "m",
                "t",
                "t equiv",
                "minimax",
                "tasa rho=1",
                "tasa V03",
                "desv",
                "factible",
            ],
            filas,
        )
    )
    lineas.append("Optimo sin restriccion (diagnostico):")
    lineas.append("")
    filas = []
    for numero in sorted(informe["simulacion"]["folds"], key=int):
        registro = informe["simulacion"]["folds"][numero]
        for etiqueta in ETIQUETAS:
            libre = registro["etiquetas"][etiqueta]["optimo_sin_restriccion"]
            filas.append(
                [
                    numero,
                    etiqueta,
                    libre["m"],
                    f"{libre['t']:.2f}",
                    f"{libre['minimax']:.6f}",
                    f"{libre['tasa_rho1']:.6f}",
                    f"{libre['desviacion_tasa_rho1']:+.6f}",
                ]
            )
    lineas.append(
        _tabla_md(["pliegue", "etiqueta", "m", "t", "minimax", "tasa rho=1", "desv vs V03"], filas)
    )
    lineas.append("## 4. Resultados LOFO")
    lineas.append("")
    filas = [
        [numero, f"{metricas['delta_por_pliegue'][numero]:+.6f}"]
        for numero in sorted(metricas["delta_por_pliegue"], key=int)
    ]
    lineas.append(
        _tabla_md(["pliegue", "delta (promedio de las 5 rho)"], filas)
        + (
            f"\nDelta medio: {metricas['delta_medio']:+.6f} (umbral +0.002); delta medio por "
            f"pliegue: {metricas['delta_medio_pliegues']:+.6f}; pliegues mejorados: "
            f"{metricas['pliegues_mejorados']}/5.\n"
        )
    )
    filas = [
        [
            rho,
            f"{metricas['delta_global_por_rho'][str(rho)]:+.6f}",
            f"{metricas['descomposicion']['componente_em_por_rho'][str(rho)]:+.6f}",
            f"{metricas['descomposicion']['componente_umbral_por_rho'][str(rho)]:+.6f}",
        ]
        for rho in RHOS
    ]
    lineas.append(
        _tabla_md(
            ["rho", "delta candidato vs V03", "delta EM (vs solo-umbral)", "delta solo-umbral vs V03"],
            filas,
        )
    )
    lineas.append("## 5. Controles")
    lineas.append("")
    lineas.append(
        f"- Descomposicion: componente EM medio {metricas['descomposicion']['componente_em_medio']:+.6f} "
        f"(media por pliegue {metricas['descomposicion']['componente_em_medio_pliegues']:+.6f}); "
        f"componente umbral medio {metricas['descomposicion']['componente_umbral_medio']:+.6f}."
    )
    lineas.append(
        f"- Sanidad sin shift rho=1.00: delta {metricas['sanity_sin_shift_rho1']['delta']:+.6f} "
        f"({', '.join(f'p{k} {v:+.6f}' for k, v in metricas['sanity_sin_shift_rho1']['delta_por_pliegue'].items())})."
    )
    filas = [
        [
            etiqueta,
            f"{metricas['guardarrail_tasa']['por_etiqueta'][etiqueta]['tasa_candidato_rho1']:.6f}",
            f"{metricas['guardarrail_tasa']['por_etiqueta'][etiqueta]['tasa_v03']:.6f}",
            f"{metricas['guardarrail_tasa']['por_etiqueta'][etiqueta]['desviacion']:+.6f}",
            metricas["guardarrail_tasa"]["por_etiqueta"][etiqueta]["cumple"],
        ]
        for etiqueta in ETIQUETAS
    ]
    lineas.append(
        _tabla_md(["etiqueta", "tasa cand rho=1", "tasa V03", "desviacion", "cumple"], filas)
    )
    filas = []
    for etiqueta in ETIQUETAS:
        for rho in RHOS:
            filas.append(
                [
                    etiqueta,
                    rho,
                    f"{metricas['tasas']['candidato'][etiqueta][str(rho)]:.6f}",
                    f"{metricas['tasas']['solo_umbral'][etiqueta][str(rho)]:.6f}",
                ]
            )
    lineas.append(_tabla_md(["etiqueta", "rho", "tasa candidato", "tasa solo-umbral"], filas))
    lineas.append(
        f"- Rescate/dano a rho=1.00: {metricas['rescate_dano_rho1']['rescate']} / "
        f"{metricas['rescate_dano_rho1']['dano']} (neto {metricas['rescate_dano_rho1']['neto']:+d})."
    )
    lineas.append("")
    lineas.append("## 6. Bootstrap pareado (filas, con pesos)")
    lineas.append("")
    bootstrap = informe["bootstrap"]
    bootstrap_em = informe["bootstrap_componente_em"]
    if bootstrap is None or bootstrap_em is None:
        lineas.append(
            "- No ejecutado: el bootstrap y el gate exigen la corrida completa de los 5 pliegues."
        )
    else:
        lineas.append(
            f"- Delta candidato vs V03: {bootstrap['delta_observado']:+.6f}; IC95 "
            f"[{bootstrap['ic95'][0]:+.6f}, {bootstrap['ic95'][1]:+.6f}]; "
            f"{bootstrap['repeticiones']} remuestreos, semilla {bootstrap['semilla']}; "
            f"proporcion positiva {bootstrap['proporcion_positiva']:.4f}."
        )
        lineas.append(
            f"- Componente EM: {bootstrap_em['delta_observado']:+.6f}; IC95 "
            f"[{bootstrap_em['ic95'][0]:+.6f}, {bootstrap_em['ic95'][1]:+.6f}]."
        )
    lineas.append("")
    lineas.append("## 7. Gate y decision")
    lineas.append("")
    if gate["evaluable"]:
        for nombre, valor in gate["criterios"].items():
            lineas.append(f"- {nombre}: {valor}")
        lineas.append(f"- Decision: {gate['decision']}")
    else:
        lineas.append(f"- Gate no evaluable: {gate['motivo']}")
        lineas.append(f"- Decision provisional: {gate['decision']}")
    lineas.append("")
    lineas.append("## 8. n_eff de Kish (filas de seleccion por pliegue)")
    lineas.append("")
    filas = []
    for numero in sorted(informe["simulacion"]["folds"], key=int):
        registro = informe["simulacion"]["folds"][numero]
        for etiqueta in ETIQUETAS:
            neff = registro["etiquetas"][etiqueta]["n_eff_seleccion"]
            filas.append([numero, etiqueta] + [f"{neff[str(rho)]:.2f}" for rho in RHOS])
    lineas.append(_tabla_md(["pliegue", "etiqueta"] + [f"n_eff rho={rho}" for rho in RHOS], filas))
    lineas.append("n_eff de Kish sobre las 1483 filas de desarrollo (referencia):")
    lineas.append("")
    filas = [
        [etiqueta] + [f"{metricas['n_eff_dev'][etiqueta][str(rho)]:.2f}" for rho in RHOS]
        for etiqueta in ETIQUETAS
    ]
    lineas.append(_tabla_md(["etiqueta"] + [f"n_eff rho={rho}" for rho in RHOS], filas))
    lineas.append("## 9. Salidas y trazabilidad")
    lineas.append("")
    lineas.append(f"- Parciales: {', '.join(informe['parciales']) if informe['parciales'] else 'ninguno'}.")
    lineas.append(
        f"- Tiempo total: {informe['tiempo_total_segundos']:.1f} s; device CPU (numpy puro); "
        f"version numpy {informe['versiones']['numpy']}, sklearn {informe['versiones']['sklearn']}."
    )
    lineas.append(
        "- La superficie completa (m, t, rho) por pliegue y etiqueta esta en el JSON."
    )
    lineas.append("")
    return "\n".join(lineas)


# ===== SECCION 6: CLI y orquestacion =====


def parsear_argumentos(argv: Sequence[str] | None) -> argparse.Namespace:
    """Define y parsea los argumentos de la linea de comandos.

    Args:
        argv: argumentos (None usa sys.argv[1:]).

    Returns:
        Namespace con humo y pliegues.
    """
    analizador = argparse.ArgumentParser(
        description=(
            "P84-05: correccion EM de prevalencia con shrinkage y umbral, simulacion LOFO "
            "sobre el OOF congelado de V03 (numpy puro, sin test y sin holdout)."
        )
    )
    analizador.add_argument(
        "--humo",
        action="store_true",
        help="modo humo: pliegue 1; no toca experiments.csv ni informes finales",
    )
    analizador.add_argument(
        "--pliegues",
        default="1,2,3,4,5",
        help="pliegues separados por comas (defecto 1,2,3,4,5)",
    )
    return analizador.parse_args(argv)


def _todas_las_selecciones_factibles(registros: dict[int, dict[str, Any]]) -> bool:
    """Comprueba que todas las selecciones LOFO tuvieron candidato factible.

    Args:
        registros: registros por pliegue.

    Returns:
        True si todas las etiquetas de todos los pliegues son factibles.
    """
    return all(
        registro["etiquetas"][etiqueta]["seleccion"]["factible"]
        for registro in registros.values()
        for etiqueta in ETIQUETAS
    )


def _em_convergido(registros: dict[int, dict[str, Any]]) -> bool:
    """Comprueba que todos los EM (seleccion y aplicacion) convergieron.

    Args:
        registros: registros por pliegue.

    Returns:
        True si todos los EM convergieron a la tolerancia declarada.
    """
    for registro in registros.values():
        for etiqueta in ETIQUETAS:
            detalle = registro["etiquetas"][etiqueta]
            for resumen in list(detalle["em_seleccion"].values()) + list(
                detalle["aplicacion"]["em"].values()
            ):
                if not resumen["convergido"]:
                    return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta P84-05 completo.

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
        log("MODO HUMO: pliegue 1; sin experiments.csv, sin informes finales y sin CSV")
    corrida_completa = set(pliegues) == set(CONFIG["pliegues_por_defecto"])
    log(f"pliegues {pliegues} | humo {humo} | corrida completa {corrida_completa}")

    # ---------- Gate 0 ----------
    log("Gate 0: hash del npz, identidad V03 y contadores")
    datos = cargar_gate0()
    log(
        f"   corrected_cv {datos.evidencia['identidad_v03']['corrected_cv_recalculado']!r} | "
        f"V03@0.5 {datos.evidencia['identidad_v03']['v03_f1_at_05_recalculado']!r} | "
        f"dev {FILAS_DEV} | holdout 0 | test 0"
    )

    # ---------- simulacion LOFO ----------
    log("Simulacion LOFO: seleccion con 4 pliegues, aplicacion al pliegue restante")
    resultado = ejecutar_lofo(datos, pliegues)
    rutas_parciales = [str(escribir_parcial(k, resultado.registros[k], humo)) for k in pliegues]
    log(f"parciales escritos: {len(rutas_parciales)}")

    # ---------- metricas de las filas evaluadas ----------
    mask = np.isin(datos.folde, np.asarray(pliegues))
    metricas = calcular_metricas(datos, resultado, mask, pliegues)
    metricas["provisional"] = not corrida_completa
    log(
        f"delta medio {metricas['delta_medio']:+.6f} | pliegues mejorados "
        f"{metricas['pliegues_mejorados']}/5 | componente EM "
        f"{metricas['descomposicion']['componente_em_medio']:+.6f}"
    )

    # ---------- bootstrap y gate ----------
    bootstrap: dict[str, Any] | None = None
    bootstrap_componente: dict[str, Any] | None = None
    autotest_humo: dict[str, Any] | None = None
    if corrida_completa:
        log("Bootstrap pareado por filas (10000, semilla 20260926, con pesos)")
        pred_v03_rho = np.repeat(resultado.pred_v03[:, :, None], len(RHOS), axis=2)
        bootstrap = bootstrap_delta(
            resultado.pred_completa,
            pred_v03_rho,
            datos.y,
            delta_esperado=metricas["delta_medio"],
        )
        log(
            f"   delta {bootstrap['delta_observado']:+.6f} | IC95 "
            f"[{bootstrap['ic95'][0]:+.6f}, {bootstrap['ic95'][1]:+.6f}]"
        )
        bootstrap_componente = bootstrap_delta(
            resultado.pred_completa,
            resultado.pred_solo_umbral,
            datos.y,
            delta_esperado=metricas["descomposicion"]["componente_em_medio"],
        )
        gate = evaluar_gate(
            metricas,
            bootstrap,
            bootstrap_componente,
            _todas_las_selecciones_factibles(resultado.registros),
            _em_convergido(resultado.registros),
            gate0_ok=True,
        )
        log(f"   gate: {'PASA' if gate['pasa'] else 'NO PASA'} | {gate['decision']}")
    else:
        gate = {
            "evaluable": False,
            "motivo": "corrida parcial o humo: el gate exige los 5 pliegues LOFO",
            "decision": "PROVISIONAL: sin decision de cierre ni de promocion",
        }
        if humo:
            log("Autotest de humo del bootstrap sobre las filas del pliegue 1")
            pred_a = resultado.pred_completa[mask]
            pred_b = np.repeat(resultado.pred_v03[mask][:, :, None], len(RHOS), axis=2)
            autotest_humo = bootstrap_delta(
                pred_a,
                pred_b,
                datos.y[mask],
                delta_esperado=metricas["delta_medio"],
            )
            log(
                f"   autotest: delta {autotest_humo['delta_observado']:+.6f} | IC95 "
                f"[{autotest_humo['ic95'][0]:+.6f}, {autotest_humo['ic95'][1]:+.6f}] "
                "(no es una cifra de gate)"
            )

    tiempo_total = round(time.perf_counter() - inicio, 2)
    informe: dict[str, Any] = {
        "experiment_id": "P84-05",
        "script": str(CONFIG["script"]),
        "fecha": str(CONFIG["fecha"]),
        "modo_humo": humo,
        "corrida_completa": corrida_completa,
        "pliegues": list(pliegues),
        "preregistro": {
            "referencia": (
                "docs/p84_spec.md seccion 6 (EXP-05) + seccion 1 (convenciones comunes); "
                "docs/p84_preregistro.md"
            ),
            "supuesto": "p(x|y) invariable; simulacion analitica determinista por ponderacion de importancia",
            "convencion_em_aplicacion": (
                "el EM de aplicacion se estima con las filas del pliegue evaluado; la seleccion "
                "(m, t) usa solo los otros 4 pliegues (LOFO)"
            ),
            "delta_primario": "delta medio global (promedio de las 5 rho); delta medio por pliegue como diagnostico",
            "gate": {
                "delta_minimo": GATE_DELTA,
                "pliegues_minimos": GATE_PLIEGUES,
                "bootstrap": {"repeticiones": REPETICIONES_BOOTSTRAP, "semilla": SEMILLA_BOOTSTRAP},
                "guardarrail_tasa": TOL_TASA,
                "componente_em_positiva": True,
            },
        },
        "gate0": datos.evidencia,
        "config": {
            "rhos": list(RHOS),
            "m_grid": list(M_GRID),
            "rejilla_t": {"min": 0.20, "max": 0.80, "paso": 0.01, "n": int(T_GRID.size)},
            "rho_inicial_em": RHO_INICIAL,
            "tolerancia_em": TOL_EM,
            "max_iter_em": MAX_ITER_EM,
            "criterio_minimax": "maximizar el minimo de la F1 ponderada sobre las 5 rho",
            "desempate": "menor m y despues t mas cercano al umbral congelado de V03",
            "guardarrail_tasa": TOL_TASA,
            "semilla_global": SEMILLA_GLOBAL,
            "umbrales_congelados": datos.umbrales.tolist(),
        },
        "versiones": versiones_entorno(),
        "simulacion": {
            "folds": {str(k): resultado.registros[k] for k in pliegues},
            "convencion_em_aplicacion": (
                "seleccion (m, t) con los otros 4 pliegues; EM de aplicacion estimado con las "
                "filas del pliegue evaluado"
            ),
        },
        "metricas": metricas,
        "bootstrap": bootstrap,
        "bootstrap_componente_em": bootstrap_componente,
        "autotest_bootstrap_humo": autotest_humo,
        "gate": gate,
        "parciales": rutas_parciales,
        "tiempo_total_segundos": tiempo_total,
    }

    if humo:
        ruta_json = CONFIG["parciales_dir"] / "p84_05_informe_humo.json"
        ruta_md = CONFIG["parciales_dir"] / "p84_05_informe_humo.md"
        escribir_json(ruta_json, informe)
        escribir_texto(ruta_md, construir_informe_md(informe))
        log(f"modo humo: informe en {ruta_json} y {ruta_md} (sin CSV ni informes finales)")
    elif corrida_completa:
        fila = construir_fila_experiments(informe, tiempo_total)
        informe["experiments_csv"] = upsert_experiments_csv(fila)
        escribir_json(CONFIG["informe_json"], informe)
        escribir_texto(CONFIG["informe_md"], construir_informe_md(informe))
        log(f"informes: {CONFIG['informe_json'].name} y {CONFIG['informe_md'].name}")
    else:
        ruta_json = CONFIG["parciales_dir"] / "p84_05_informe_parcial.json"
        ruta_md = CONFIG["parciales_dir"] / "p84_05_informe_parcial.md"
        escribir_json(ruta_json, informe)
        escribir_texto(ruta_md, construir_informe_md(informe))
        log(
            "AVISO: corrida parcial: sin experiments.csv ni informe final; informe etiquetado en "
            f"{ruta_json.name} y {ruta_md.name}"
        )

    # ---------- resumen final ----------
    log("=" * 78)
    log(f"P84-05 {'HUMO' if humo else ('OK' if corrida_completa else 'PARCIAL')} | numpy puro | {tiempo_total:.1f}s")
    log(
        f"   gate0: npz verificado | corrected_cv "
        f"{datos.evidencia['identidad_v03']['corrected_cv_recalculado']!r} | V03@0.5 "
        f"{datos.evidencia['identidad_v03']['v03_f1_at_05_recalculado']!r}"
    )
    for k in pliegues:
        registro = resultado.registros[k]
        log(
            f"   pliegue {k}: delta {registro['delta_pliegue']:+.6f} | "
            + " ".join(
                f"{etiqueta}(m={registro['etiquetas'][etiqueta]['seleccion']['m']},"
                f"t={registro['etiquetas'][etiqueta]['seleccion']['t']:.2f})"
                for etiqueta in ETIQUETAS
            )
        )
    if corrida_completa:
        log(
            f"   delta medio {metricas['delta_medio']:+.6f} (>= +0.002) | pliegues mejorados "
            f"{metricas['pliegues_mejorados']}/5 | IC95 [{bootstrap['ic95'][0]:+.6f}, "
            f"{bootstrap['ic95'][1]:+.6f}]"
        )
        log(
            f"   controles: sanity rho=1 {metricas['sanity_sin_shift_rho1']['delta']:+.6f} | "
            f"componente EM {metricas['descomposicion']['componente_em_medio']:+.6f} | "
            f"guardarrail max {metricas['guardarrail_tasa']['max_desviacion']:.6f}"
        )
        log(f"   decision: {gate['decision']}")
    else:
        log("   corrida parcial/humo: gate no evaluable; sin decision de cierre ni promocion")
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
