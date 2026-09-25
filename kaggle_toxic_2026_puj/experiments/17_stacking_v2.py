"""P6 - Late fusion / stacking de segunda generacion (E2-18).

Pregunta que responde
---------------------
P1 dejo un meta-modelo de dependencia entre etiquetas adoptado como candidato
v2-A (``corrected_cv`` 0.7853 con verificacion estrictamente anidada) y P2 dejo
dos canales de targeting sin mejora standalone (CH-TGT+DIST 0.6484, TXT+CH
0.7356) pero con mas recall de AG en la rebanada de referencia colectiva
(0.487 vs 0.452 de V03). La hipotesis de P2 fue que esa senal solo puede
aparecer en una fusion tardia regularizada. ¿Aportan los canales de targeting
cuando se anaden al meta-modelo de dependencia, junto con todos los scores OOF,
sus margenes, sus entropias y rasgos estructurales?

Metodo
------
- Trabajo exclusivo sobre el DESARROLLO (1483 filas); el holdout congelado
  (``data/processed/holdout_v2_ids.csv``) no se carga para modelar.
- Pliegues corregidos v2 (D-016): ``StratifiedGroupKFold(5)`` con grupos
  fusionados (near-duplicado UNION pseudo-autor) y seed 42.
- Insumos OOF congelados: 4 bases (``v2_oof_base_corrected.npz``) y 2 canales de
  targeting (``v2_oof_targeting.npz``). Se verifica que ambos usan los mismos
  pliegues y las mismas etiquetas antes de usarlos.
- Matriz de meta-features (por conjunto): probabilidades de cada score, margen
  ``|p - 0.5|`` y entropia binaria; mas rasgos estructurales fijos (longitud en
  caracteres y tokens, mencion, hashtag, URL, comillas, segunda persona,
  colectivo estructural, risas, mayusculas sostenidas y elongacion).
- Variantes de meta-modelo evaluadas fold a fold (escalador y modelos ajustados
  solo con el entrenamiento del pliegue, prediccion en la validacion):
  META-BASES (LR con C en {0.1, 0.3, 1.0}), META+CH (bases + canales),
  META+CH+ESTR (bases + canales + estructurales ampliados), las mismas con
  RidgeClassifier (alpha en {0.3, 1.0, 3.0}) y LinearSVC (C en {0.3, 1.0}).
- Verificacion ANADIDA de honestidad: ademas de la evaluacion con el OOF
  congelado (que es la que pide el paquete, con el anidamiento definido como
  "el meta se ajusta con 4 pliegues y predice el quinto"), se reproduce la
  verificacion ESTRICTAMENTE ANIDADA de P1 (los 4 bases se reentrenan dentro de
  cada pliegue exterior) para (a) confirmar el 0.7853 de referencia y (b) medir
  con las mismas predicciones el IC bootstrap del delta. La variante con canales
  tambien se evalua con bases anidadas + canales congelados como control de
  robustez.
- Advertencia de honestidad: en la evaluacion con OOF congelado los scores de
  las filas de entrenamiento del meta provienen de modelos base que vieron el
  pliegue evaluado en su propio entrenamiento (stacking estandar); P1 midio que
  ese sesgo es <= 0.0003 en este corpus (0.7850 congelado vs 0.7853 anidado),
  por lo que la comparacion ENTRE variantes construidas igual es valida.
- Adopcion (pre-registrada en D-016 y el enunciado de P6): superar
  0.7853 + 0.005 con >= 3/5 pliegues mejorados, o complementariedad
  (``ensemble_gain`` >= 0.0015 frente a M2 a 0.5 con IC bootstrap > 0).
- Diagnostico: F1 por etiqueta y por pliegue, recall de AG en la rebanada de
  referencia colectiva (regla estructural del script 06, reutilizada via el
  script 16) y tabla de FP/FN corregidos e introducidos frente a V03.

Salidas
-------
``reports/stacking_v2.csv``, ``reports/stacking_v2.json``,
``reports/figuras/fig_29_stacking_v2.png``, filas ``E2-18-*`` en
``experiments.csv``, seccion ``## Fase v2 - P6 (2026-09-25)`` en
``reports/contexto.md``, actualizacion de ``tasks/todo.md`` (P6.1-P6.3 y Gate
P6) y, si el mejor meta cumple la regla, ``models/v2_meta_candidato.joblib`` y
``features/v2_meta_oof.npz``. No se genera submission ni se toca el holdout.

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/17_stacking_v2.py
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse
from scipy.special import expit, xlogy
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.evaluation import metricas_multietiqueta
from src.splits import (
    ETIQUETAS,
    RANDOM_STATE,
    combinaciones,
    grupos_fusionados,
    pliegues_corregidos,
)

IDS_PATH = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
OOF_BASES_PATH = PROJECT_ROOT / "features" / "v2_oof_base_corrected.npz"
OOF_TARGETING_PATH = PROJECT_ROOT / "features" / "v2_oof_targeting.npz"
CADENAS_JSON = PROJECT_ROOT / "reports" / "cadenas_etiquetas_v2.json"
REPORTE_CSV = PROJECT_ROOT / "reports" / "stacking_v2.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "stacking_v2.json"
FIGURA_PATH = PROJECT_ROOT / "reports" / "figuras" / "fig_29_stacking_v2.png"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"
CONTEXTO_MD = PROJECT_ROOT / "reports" / "contexto.md"
TODO_MD = PROJECT_ROOT / "tasks" / "todo.md"
MODELO_CANDIDATO_PATH = PROJECT_ROOT / "models" / "v2_meta_candidato.joblib"
OOF_META_PATH = PROJECT_ROOT / "features" / "v2_meta_oof.npz"

CONFIG: Final[dict[str, Any]] = {
    "fecha": "2026-09-25",
    "n_pliegues_externos": 5,
    "n_pliegues_internos": 4,
    "n_bootstrap": 2000,
    "meta_C": (0.1, 0.3, 1.0),
    "ridge_alpha": (0.3, 1.0, 3.0),
    "svc_C": (0.3, 1.0),
    "mejora_minima": 0.005,
    "ensemble_gain_minimo": 0.0015,
    "pliegues_minimos_mejorados": 3,
    "tolerancia_replicacion_v03": 5e-4,
    "tolerancia_replicacion_anidada": 2e-3,
    "rebanada_n_esperado": 363,
    "rebanada_ag_positivos_esperado": 115,
}

# Parametros identicos a los de V03 (experiments/14, modelo M2_union_lr_c2).
LR_PARAMS: Final[dict[str, Any]] = {
    "C": 2.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "liblinear",
    "random_state": RANDOM_STATE,
}

PATRON_URL = re.compile(r"https?://|www\.|t\.co")
PATRON_MENCION = re.compile(r"@\w+")
PATRON_HASHTAG = re.compile(r"#\w+")
PATRON_COMILLAS = re.compile(r"[\"'\u00ab\u00bb\u201c\u201d]")
# Marcadores plurales/colectivos fijos (lista estructural escrita a mano, no
# ajustada con datos: no puede introducir fuga).
PATRON_COLECTIVO = re.compile(
    r"\b(los|las|esos|esas|todos|todas|ellos|ellas|ustedes|nosotros|nosotras"
    r"|nuestros|nuestras|muchos|muchas|unos|unas|varios|varias|ambos|ambas"
    r"|gente|personas|mujeres|hombres|ninos|ninas|jovenes|ancianos|ancianas"
    r"|indigenas|extranjeros|extranjeras|inmigrantes|venezolanos|venezolanas"
    r"|colombianos|colombianas|funcionarios|politicos|periodistas)\b"
)
PATRON_SEGUNDA_PERSONA = re.compile(
    r"\b(tu|te|ti|tus|vos|usted|ustedes|contigo|tuyo|tuya|tuyos|tuyas"
    r"|sos|tenes|eres|estas|dejame|pideme)\b"
)
PATRON_RISAS = re.compile(r"(?i)\b(?:ja){2,}\w*|\b(?:je){2,}\w*|\b(?:jo){2,}\w*|\blol\b|\bxd+\b")
PATRON_MAYUSCULAS_SOSTENIDAS = re.compile(r"[A-Z]+")
PATRON_ELONGACION = re.compile(r"(.)\1{2,}")

# Los 8 rasgos estructurales de P1, en su mismo orden (reproducibilidad exacta
# de la matriz del meta de dependencia).
NOMBRES_ESTRUCTURALES_BASE: Final[tuple[str, ...]] = (
    "n_chars",
    "n_tokens",
    "tiene_mencion",
    "tiene_hashtag",
    "tiene_url",
    "tiene_colectivo",
    "tiene_segunda_persona",
    "tiene_comillas",
)
# Ampliacion pedida en P6: risas, mayusculas sostenidas y elongacion.
NOMBRES_ESTRUCTURALES_EXTRA: Final[tuple[str, ...]] = (
    "tiene_risas",
    "mayusculas_sostenidas",
    "tiene_elongacion",
)
NOMBRES_ESTRUCTURALES: Final[tuple[str, ...]] = (
    NOMBRES_ESTRUCTURALES_BASE + NOMBRES_ESTRUCTURALES_EXTRA
)

MODELOS_BASE: Final[tuple[str, ...]] = (
    "M1_word_lr",
    "M2_union_lr_c2",
    "M3_union_svc",
    "M4_char_lr",
)

# Conjuntos de features del meta: nombre -> (usa_canales, estructural_ampliado).
CONJUNTOS: Final[dict[str, tuple[bool, bool]]] = {
    "bases": (False, False),
    "bases_canales": (True, False),
    "bases_canales_ampliado": (True, True),
}


@dataclass(frozen=True)
class EspecMeta:
    """Especificacion de un meta-modelo concreto (variante + hiperparametro).

    Attributes:
        variante: nombre de la variante (agrupa varios hiperparametros).
        conjunto: clave de :data:`CONJUNTOS` con el conjunto de features.
        modelo: ``lr`` (LogisticRegression), ``ridge`` (RidgeClassifier) o
            ``svc`` (LinearSVC con sigmoide en la decision).
        parametro: valor del hiperparametro (``C`` o ``alpha``).
    """

    variante: str
    conjunto: str
    modelo: str
    parametro: float

    @property
    def key(self) -> str:
        """Identificador unico de la especificacion."""
        return f"{self.variante}|{self.conjunto}|{self.modelo}|{self.parametro:g}"


@dataclass(frozen=True)
class MatricesExteriores:
    """Matrices dispersas de un pliegue exterior con su CV interna.

    Attributes:
        internas: lista de ``(X_entrenamiento, X_validacion, idx_ent, idx_val)``
            de la CV interna; los indices son posicionales dentro del
            entrenamiento exterior.
        x_entrenamiento: matriz del entrenamiento exterior completo.
        x_validacion: matriz de la validacion exterior.
        idx_entrenamiento: indices posicionales del entrenamiento dentro del
            desarrollo.
        idx_validacion: indices posicionales de la validacion dentro del
            desarrollo.
    """

    internas: list[tuple[sparse.csr_matrix, sparse.csr_matrix, NDArray[np.int_], NDArray[np.int_]]]
    x_entrenamiento: sparse.csr_matrix
    x_validacion: sparse.csr_matrix
    idx_entrenamiento: NDArray[np.int_]
    idx_validacion: NDArray[np.int_]


def log(mensaje: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        mensaje: texto a imprimir.
    """
    print(f"[P6] {mensaje}", flush=True)


# ===== SECCION 1: carga, referencias y verificaciones =====


def cargar_desarrollo() -> tuple[pd.DataFrame, NDArray[np.int_], list[tuple[NDArray[np.int_], NDArray[np.int_]]]]:
    """Carga el desarrollo y sus pliegues corregidos v2.

    Returns:
        Tupla ``(desarrollo, grupos, pliegues)``. El holdout congelado queda
        excluido por construccion.
    """
    train = data_module.read_train()
    particion = pd.read_csv(IDS_PATH)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    grupos_completos = grupos_fusionados(train)
    mapa_grupo = dict(zip(train["id"].astype(int), grupos_completos.tolist()))
    grupos = np.array([mapa_grupo[identificador] for identificador in desarrollo["id"].astype(int)])
    pliegues = pliegues_corregidos(desarrollo, grupos)
    return desarrollo, grupos, pliegues


def cargar_referencias() -> dict[str, Any]:
    """Lee las referencias pre-registradas del reporte de P1.

    Args:
        Ninguno.

    Returns:
        Diccionario con el meta anidado de P1 (0.7853), V03 con umbrales
        (0.7772), M2 a 0.5 (0.7718) y sus F1 por pliegue y por etiqueta.
    """
    datos = json.loads(CADENAS_JSON.read_text(encoding="utf-8"))
    anidado = datos["meta_anidado"]
    return {
        "meta_anidado_cv": float(anidado["corrected_cv"]),
        "meta_anidado_pliegues": [float(valor) for valor in anidado["f1_pliegues"]],
        "meta_anidado_por_etiqueta": {
            etiqueta: float(valor) for etiqueta, valor in anidado["f1_por_etiqueta"].items()
        },
        "meta_anidado_mejor_C": float(anidado["mejor_C"]),
        "v03_umbrales_cv": float(datos["config"]["referencia_v03_umbrales"]),
        "m2_umbral_05_cv": float(datos["config"]["referencia_m2_umbral_05"]),
        "umbrales_v03": {
            etiqueta: float(valor) for etiqueta, valor in datos["config"]["umbrales_v03"].items()
        },
        "v03_pliegues": [float(valor) for valor in anidado["f1_pliegues_base_v03_umbrales"]],
        "m2_pliegues": [float(valor) for valor in anidado["f1_pliegues_base_m2_05"]],
    }


def cargar_oof(
    desarrollo: pd.DataFrame,
    grupos: NDArray[np.int_],
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
) -> tuple[NDArray[np.float64], tuple[str, ...], NDArray[np.float64], tuple[str, ...], dict[str, Any]]:
    """Carga y verifica los OOF congelados de bases y canales de targeting.

    Args:
        desarrollo: DataFrame del desarrollo.
        grupos: grupos fusionados alineados con ``desarrollo``.
        y_verdadera: etiquetas verdaderas ``(n, 3)`` en el orden de ETIQUETAS.
        pliegue_de_fila: pliegue corregido (1..5) de cada fila del desarrollo.

    Returns:
        Tupla ``(proba_bases, modelos_bases, proba_canales, canales, info)``.
        ``info`` resume las verificaciones realizadas.

    Raises:
        RuntimeError: si los insumos OOF no coinciden con el desarrollo.
    """
    mapa_fila = {identificador: posicion for posicion, identificador in enumerate(desarrollo["id"].astype(int))}
    datos_bases = np.load(OOF_BASES_PATH)
    ids_bases = datos_bases["ids"].astype(int)
    orden_bases = np.array([mapa_fila[identificador] for identificador in ids_bases])
    if not np.array_equal(datos_bases["y_true"], y_verdadera[orden_bases]):
        raise RuntimeError("Las etiquetas del OOF de bases no coinciden con el desarrollo")
    proba_bases = np.zeros((len(desarrollo), len(MODELOS_BASE), len(ETIQUETAS)))
    proba_bases[orden_bases] = datos_bases["proba"]
    etiquetas_bases = tuple(str(valor) for valor in datos_bases["etiquetas"])
    if etiquetas_bases != tuple(ETIQUETAS):
        raise RuntimeError(f"Orden de etiquetas inesperado en el OOF de bases: {etiquetas_bases}")
    modelos_bases = tuple(str(valor) for valor in datos_bases["modelos"])
    if modelos_bases != MODELOS_BASE:
        raise RuntimeError(f"Modelos base inesperados: {modelos_bases}")
    if not np.array_equal(datos_bases["grupos"].astype(int), grupos[orden_bases]):
        raise RuntimeError("Los grupos del OOF de bases no coinciden con los grupos fusionados")

    datos_canales = np.load(OOF_TARGETING_PATH)
    ids_canales = datos_canales["ids"].astype(int)
    orden_canales = np.array([mapa_fila[identificador] for identificador in ids_canales])
    if not np.array_equal(datos_canales["y_true"], y_verdadera[orden_canales]):
        raise RuntimeError("Las etiquetas del OOF de targeting no coinciden con el desarrollo")
    if not np.array_equal(datos_canales["pliegue"].astype(int), pliegue_de_fila[orden_canales]):
        raise RuntimeError("Los pliegues del OOF de targeting no coinciden con los pliegues corregidos")
    proba_canales = np.zeros((len(desarrollo), len(datos_canales["variantes"]), len(ETIQUETAS)))
    proba_canales[orden_canales] = datos_canales["proba"]
    canales = tuple(str(valor) for valor in datos_canales["variantes"])
    info = {
        "n_bases": len(modelos_bases),
        "modelos_bases": list(modelos_bases),
        "n_canales": len(canales),
        "canales": list(canales),
        "C_canales": [float(valor) for valor in datos_canales["C"]],
        "mismos_pliegues_canales": True,
        "y_true_coincide": True,
    }
    return proba_bases, modelos_bases, proba_canales, canales, info


def verificar_datos(
    desarrollo: pd.DataFrame, grupos: NDArray[np.int_], pliegue_de_fila: NDArray[np.int_]
) -> dict[str, Any]:
    """Comprueba integridad basica y resume el desarrollo.

    Args:
        desarrollo: DataFrame del conjunto de desarrollo.
        grupos: grupos fusionados alineados con ``desarrollo``.
        pliegue_de_fila: pliegue corregido de cada fila.

    Returns:
        Diccionario con el resumen de verificacion.
    """
    resumen = {
        "n_filas": len(desarrollo),
        "n_grupos": len(np.unique(grupos)),
        "nulos": int(desarrollo[list(ETIQUETAS) + ["text"]].isna().sum().sum()),
        "prevalencia": {etiqueta: float(desarrollo[etiqueta].mean()) for etiqueta in ETIQUETAS},
        "combinaciones": {clave: int(valor) for clave, valor in combinaciones(desarrollo).value_counts().items()},
        "tamano_pliegues": {int(numero): int(np.sum(pliegue_de_fila == numero)) for numero in sorted(set(pliegue_de_fila.tolist()))},
    }
    log(
        f"desarrollo n={resumen['n_filas']} grupos={resumen['n_grupos']} "
        f"nulos={resumen['nulos']} pliegues={resumen['tamano_pliegues']}"
    )
    return resumen


def verificar_replicacion_v03(
    proba_m2: NDArray[np.float64], y_verdadera: NDArray[np.int_], referencias: dict[str, Any]
) -> float:
    """Comprueba que el OOF de M2 con los umbrales de V03 reproduce 0.7772.

    Args:
        proba_m2: probabilidades OOF de M2 ``(n, 3)``.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        referencias: referencias pre-registradas.

    Returns:
        Diferencia absoluta entre la F1 obtenida y la referencia 0.7772.

    Raises:
        RuntimeError: si la diferencia supera la tolerancia configurada.
    """
    prediccion = (proba_m2 >= vector_umbrales(referencias["umbrales_v03"])).astype(int)
    f1 = metricas_multietiqueta(y_verdadera, prediccion)["f1_media_columnas"]
    diferencia = abs(f1 - referencias["v03_umbrales_cv"])
    if diferencia > CONFIG["tolerancia_replicacion_v03"]:
        raise RuntimeError(f"V03 no reproduce: {f1:.6f} vs {referencias['v03_umbrales_cv']}")
    log(
        f"V03 reproducido con el OOF congelado: {f1:.4f} vs {referencias['v03_umbrales_cv']:.4f} "
        f"(dif {diferencia:.2e})"
    )
    return diferencia


# ===== SECCION 2: bloques de features del meta =====


def rasgos_estructurales(textos: Sequence[str], ampliado: bool) -> tuple[NDArray[np.float64], list[str]]:
    """Calcula los rasgos estructurales fijos (sin ajuste con datos).

    Args:
        textos: secuencia de textos.
        ampliado: si se anaden risas, mayusculas sostenidas y elongacion.

    Returns:
        Tupla ``(matriz, nombres)`` con 8 u 11 columnas.
    """
    filas = []
    for texto in textos:
        mayusculas = [len(tramo) for tramo in PATRON_MAYUSCULAS_SOSTENIDAS.findall(texto)]
        fila = [
            len(texto),
            len(texto.split()),
            float(bool(PATRON_MENCION.search(texto))),
            float(bool(PATRON_HASHTAG.search(texto))),
            float(bool(PATRON_URL.search(texto))),
            float(bool(PATRON_COLECTIVO.search(texto))),
            float(bool(PATRON_SEGUNDA_PERSONA.search(texto))),
            float(bool(PATRON_COMILLAS.search(texto))),
        ]
        if ampliado:
            fila.extend(
                [
                    float(bool(PATRON_RISAS.search(texto))),
                    float(max(mayusculas) if mayusculas else 0),
                    float(bool(PATRON_ELONGACION.search(texto))),
                ]
            )
        filas.append(fila)
    nombres = list(NOMBRES_ESTRUCTURALES) if ampliado else list(NOMBRES_ESTRUCTURALES_BASE)
    return np.asarray(filas, dtype=float), nombres


def entropia_binaria(probabilidades: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calcula la entropia binaria por columna con recorte numerico.

    Args:
        probabilidades: matriz de probabilidades en ``[0, 1]``.

    Returns:
        Matriz de entropias con la misma forma.
    """
    recortadas = np.clip(probabilidades, 1e-6, 1 - 1e-6)
    return -(xlogy(recortadas, recortadas) + xlogy(1 - recortadas, 1 - recortadas))


def bloque_scores(proba: NDArray[np.float64], nombres_modelos: Sequence[str]) -> tuple[NDArray[np.float64], list[str]]:
    """Construye el bloque de scores (probabilidades, margenes y entropias).

    Args:
        proba: probabilidades ``(n, n_modelos, 3)``.
        nombres_modelos: nombre de cada modelo o canal.

    Returns:
        Tupla ``(matriz, nombres)`` con ``3 * n_modelos * 3`` columnas en el
        orden probabilidades, margenes y entropias.
    """
    n_filas = proba.shape[0]
    planas = proba.reshape(n_filas, -1)
    margenes = np.abs(planas - 0.5)
    entropias = entropia_binaria(planas)
    nombres = (
        [f"proba_{modelo}_{etiqueta}" for modelo in nombres_modelos for etiqueta in ETIQUETAS]
        + [f"margen_{modelo}_{etiqueta}" for modelo in nombres_modelos for etiqueta in ETIQUETAS]
        + [f"entropia_{modelo}_{etiqueta}" for modelo in nombres_modelos for etiqueta in ETIQUETAS]
    )
    return np.hstack([planas, margenes, entropias]), nombres


def construir_bloque_meta(
    proba_bases: NDArray[np.float64],
    proba_canales: NDArray[np.float64] | None,
    nombres_canales: Sequence[str],
    textos: Sequence[str],
    conjunto: str,
) -> tuple[NDArray[np.float64], list[str]]:
    """Construye la matriz de meta-features de un conjunto dado.

    Args:
        proba_bases: probabilidades de los 4 bases ``(n, 4, 3)``.
        proba_canales: probabilidades de los canales ``(n, k, 3)`` o ``None``.
        nombres_canales: nombre de cada canal (orden de ``proba_canales``).
        textos: textos alineados con las filas.
        conjunto: clave de :data:`CONJUNTOS`.

    Returns:
        Tupla ``(matriz, nombres)``.

    Raises:
        ValueError: si el conjunto no existe o faltan los canales.
    """
    if conjunto not in CONJUNTOS:
        raise ValueError(f"Conjunto desconocido: {conjunto}")
    usa_canales, ampliado = CONJUNTOS[conjunto]
    bloques: list[NDArray[np.float64]] = []
    nombres: list[str] = []
    matriz_bases, nombres_bases = bloque_scores(proba_bases, MODELOS_BASE)
    bloques.append(matriz_bases)
    nombres.extend(nombres_bases)
    if usa_canales:
        if proba_canales is None:
            raise ValueError("El conjunto exige canales y no se recibieron")
        matriz_canales, nombres_canales_bloque = bloque_scores(proba_canales, nombres_canales)
        bloques.append(matriz_canales)
        nombres.extend(nombres_canales_bloque)
    estructurales, nombres_estructurales = rasgos_estructurales(textos, ampliado)
    bloques.append(estructurales)
    nombres.extend(nombres_estructurales)
    return np.hstack(bloques), nombres


# ===== SECCION 3: evaluacion del meta fold a fold =====


def ajustar_modelo_meta(espec: EspecMeta, x_entrenamiento: NDArray[np.float64], y_entrenamiento: NDArray[np.int_], x_prediccion: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ajusta el meta-modelo de una especificacion y devuelve probabilidades.

    Args:
        espec: especificacion del meta-modelo.
        x_entrenamiento: features escaladas del entrenamiento.
        y_entrenamiento: etiqueta binaria del entrenamiento.
        x_prediccion: features escaladas a predecir.

    Returns:
        Probabilidades de la clase positiva (sigmoide de la decision en Ridge y
        LinearSVC, coherente con el tratamiento de M3 en el OOF de bases).

    Raises:
        ValueError: si el modelo de la especificacion no existe.
    """
    if espec.modelo == "lr":
        modelo = LogisticRegression(
            C=espec.parametro,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        )
        modelo.fit(x_entrenamiento, y_entrenamiento)
        return modelo.predict_proba(x_prediccion)[:, 1]
    if espec.modelo == "ridge":
        modelo_ridge = RidgeClassifier(alpha=espec.parametro, class_weight="balanced")
        modelo_ridge.fit(x_entrenamiento, y_entrenamiento)
        return expit(modelo_ridge.decision_function(x_prediccion))
    if espec.modelo == "svc":
        modelo_svc = LinearSVC(
            C=espec.parametro,
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
        )
        modelo_svc.fit(x_entrenamiento, y_entrenamiento)
        return expit(modelo_svc.decision_function(x_prediccion))
    raise ValueError(f"Modelo de meta desconocido: {espec.modelo}")


def evaluar_meta_por_pliegue(
    features_por_pliegue: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]],
    pliegues: list[tuple[NDArray[np.int_], NDArray[np.int_]]],
    y_verdadera: NDArray[np.int_],
    especificaciones: Sequence[EspecMeta],
) -> dict[str, NDArray[np.float64]]:
    """Evalua especificaciones del meta fold a fold.

    El escalador y los modelos se ajustan SOLO con el entrenamiento de cada
    pliegue; la validacion de ese pliegue no participa en ningun ajuste. La
    convencion es que ``features_por_pliegue[numero]`` corresponde a
    ``pliegues[numero - 1]`` (numeracion 1..k).

    Args:
        features_por_pliegue: features sin escalar ``(entrenamiento, validacion)``
            por numero de pliegue.
        pliegues: pliegues corregidos ``(entrenamiento, validacion)``.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        especificaciones: especificaciones a evaluar.

    Returns:
        Diccionario ``key -> probabilidades OOF`` de forma ``(n, 3)``.
    """
    probas = {espec.key: np.zeros((len(y_verdadera), len(ETIQUETAS))) for espec in especificaciones}
    for numero, (x_entrenamiento, x_validacion) in features_por_pliegue.items():
        idx_entrenamiento, idx_validacion = pliegues[numero - 1]
        escalador = StandardScaler().fit(x_entrenamiento)
        x_entrenamiento_escalado = escalador.transform(x_entrenamiento)
        x_validacion_escalada = escalador.transform(x_validacion)
        for espec in especificaciones:
            for posicion, _ in enumerate(ETIQUETAS):
                probas[espec.key][idx_validacion, posicion] = ajustar_modelo_meta(
                    espec,
                    x_entrenamiento_escalado,
                    y_verdadera[idx_entrenamiento, posicion],
                    x_validacion_escalada,
                )
    return probas


def expandir_especificaciones(
    variantes: Sequence[tuple[str, str, str, Sequence[float]]],
) -> tuple[EspecMeta, ...]:
    """Expande la tabla de variantes en especificaciones concretas.

    Args:
        variantes: lista de ``(nombre, conjunto, modelo, hiperparametros)``.

    Returns:
        Tupla de especificaciones.
    """
    return tuple(
        EspecMeta(variante=nombre, conjunto=conjunto, modelo=modelo, parametro=float(parametro))
        for nombre, conjunto, modelo, parametros in variantes
        for parametro in parametros
    )


# ===== SECCION 4: verificacion estrictamente anidada (patron de P1) =====


def cargar_modulo_base() -> Any:
    """Importa ``experiments/14_oof_corregido_v2.py`` para reutilizar su receta.

    Returns:
        Modulo del experimento 14, con ``construir_matrices`` (representacion de
        V03) y ``MODELOS`` (los cuatro modelos base del OOF corregido).
    """
    ruta = PROJECT_ROOT / "experiments" / "14_oof_corregido_v2.py"
    especificacion = importlib.util.spec_from_file_location("experimento_14_base", ruta)
    if especificacion is None or especificacion.loader is None:
        raise ImportError(f"No se pudo cargar {ruta}")
    modulo = importlib.util.module_from_spec(especificacion)
    # Registrar en sys.modules es necesario para que dataclasses/typing resuelvan
    # el modulo de origen durante exec_module (Python >= 3.12).
    sys.modules[especificacion.name] = modulo
    especificacion.loader.exec_module(modulo)
    if tuple(modulo.MODELOS) != MODELOS_BASE:
        raise RuntimeError("Los modelos base de experiments/14 no coinciden con el OOF congelado")
    return modulo


def construir_matrices_exteriores(
    construir_matrices: Callable,
    textos: pd.Series,
    idx_entrenamiento: NDArray[np.int_],
    idx_validacion: NDArray[np.int_],
    combinacion: pd.Series,
    grupos: NDArray[np.int_],
    representacion: str = "union",
) -> MatricesExteriores:
    """Ajusta los vectorizadores del pliegue exterior y de su CV interna.

    Args:
        construir_matrices: funcion importada de ``experiments/14``.
        textos: serie de textos del desarrollo.
        idx_entrenamiento: indices del entrenamiento exterior.
        idx_validacion: indices de la validacion exterior.
        combinacion: etiqueta de combinacion HS-TR-AG por fila.
        grupos: grupos fusionados alineados con el desarrollo.
        representacion: ``word``, ``char`` o ``union``.

    Returns:
        Estructura con las matrices internas y externas.
    """
    divisor = StratifiedGroupKFold(
        n_splits=CONFIG["n_pliegues_internos"], shuffle=True, random_state=RANDOM_STATE
    )
    textos_entrenamiento = textos.iloc[idx_entrenamiento].reset_index(drop=True)
    combinacion_entrenamiento = combinacion.iloc[idx_entrenamiento].reset_index(drop=True)
    grupos_entrenamiento = grupos[idx_entrenamiento]

    internas: list[tuple[sparse.csr_matrix, sparse.csr_matrix, NDArray[np.int_], NDArray[np.int_]]] = []
    for idx_interno_entrenamiento, idx_interno_validacion in divisor.split(
        textos_entrenamiento, combinacion_entrenamiento, groups=grupos_entrenamiento
    ):
        matriz_entrenamiento, matriz_validacion = construir_matrices(
            textos_entrenamiento.iloc[idx_interno_entrenamiento],
            textos_entrenamiento.iloc[idx_interno_validacion],
            representacion,
        )
        internas.append(
            (matriz_entrenamiento, matriz_validacion, idx_interno_entrenamiento, idx_interno_validacion)
        )

    matriz_exterior_entrenamiento, matriz_exterior_validacion = construir_matrices(
        textos.iloc[idx_entrenamiento], textos.iloc[idx_validacion], representacion
    )
    return MatricesExteriores(
        internas=internas,
        x_entrenamiento=matriz_exterior_entrenamiento,
        x_validacion=matriz_exterior_validacion,
        idx_entrenamiento=idx_entrenamiento,
        idx_validacion=idx_validacion,
    )


def ajustar_y_predecir(
    especificacion: dict,
    x_entrenamiento: sparse.csr_matrix,
    y_entrenamiento: NDArray[np.int_],
    x_prediccion: sparse.csr_matrix,
) -> NDArray[np.float64]:
    """Ajusta un modelo base y devuelve probabilidades (receta de ``experiments/14``).

    Args:
        especificacion: diccionario con ``tipo`` (``lr`` o ``svc``) y ``params``.
        x_entrenamiento: matriz del pliegue de entrenamiento.
        y_entrenamiento: etiqueta binaria del pliegue de entrenamiento.
        x_prediccion: matriz a predecir.

    Returns:
        Probabilidades (LR) o sigmoide del ``decision_function`` (SVC).
    """
    if especificacion["tipo"] == "lr":
        modelo = LogisticRegression(**especificacion["params"]).fit(x_entrenamiento, y_entrenamiento)
        return modelo.predict_proba(x_prediccion)[:, 1]
    modelo = LinearSVC(**especificacion["params"]).fit(x_entrenamiento, y_entrenamiento)
    return expit(modelo.decision_function(x_prediccion))


def probas_bases_anidadas(
    construccion_matrices: Callable,
    modulo_base: Any,
    textos: pd.Series,
    y_verdadera: NDArray[np.int_],
    idx_entrenamiento: NDArray[np.int_],
    idx_validacion: NDArray[np.int_],
    combinacion: pd.Series,
    grupos: NDArray[np.int_],
    matrices_union: MatricesExteriores,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reentrena los 4 bases dentro del pliegue exterior (anidamiento estricto).

    Reproduce el patron de ``experiments/15_cadenas_etiquetas_v2.py``: la CV
    interna de 4 pliegues produce las features del meta para el entrenamiento
    exterior y el modelo exterior (entrenado sin la validacion) produce las de
    la validacion.

    Args:
        construccion_matrices: ``construir_matrices`` de ``experiments/14``.
        modulo_base: modulo de ``experiments/14`` con la especificacion ``MODELOS``.
        textos: serie de textos del desarrollo.
        y_verdadera: matriz ``(n, 3)`` de etiquetas del desarrollo.
        idx_entrenamiento: indices del entrenamiento exterior.
        idx_validacion: indices de la validacion exterior.
        combinacion: etiqueta de combinacion HS-TR-AG por fila.
        grupos: grupos fusionados alineados con el desarrollo.
        matrices_union: matrices de la representacion de V03 del pliegue.

    Returns:
        Tupla ``(proba_entrenamiento, proba_validacion)`` con forma
        ``(n, 4, 3)`` cada una.
    """
    matrices_por_representacion = {"union": matrices_union}
    for representacion in ("word", "char"):
        matrices_por_representacion[representacion] = construir_matrices_exteriores(
            construccion_matrices,
            textos,
            idx_entrenamiento,
            idx_validacion,
            combinacion,
            grupos,
            representacion,
        )

    y_entrenamiento = y_verdadera[idx_entrenamiento]
    proba_entrenamiento = np.zeros((len(idx_entrenamiento), len(MODELOS_BASE), len(ETIQUETAS)))
    proba_validacion = np.zeros((len(idx_validacion), len(MODELOS_BASE), len(ETIQUETAS)))
    for posicion_modelo, (_, especificacion) in enumerate(modulo_base.MODELOS.items()):
        matrices = matrices_por_representacion[especificacion["representacion"]]
        for posicion_etiqueta, _ in enumerate(ETIQUETAS):
            for matriz_entrenamiento, matriz_validacion, idx_int_ent, idx_int_val in matrices.internas:
                proba_entrenamiento[idx_int_val, posicion_modelo, posicion_etiqueta] = ajustar_y_predecir(
                    especificacion,
                    matriz_entrenamiento,
                    y_entrenamiento[idx_int_ent, posicion_etiqueta],
                    matriz_validacion,
                )
            proba_validacion[:, posicion_modelo, posicion_etiqueta] = ajustar_y_predecir(
                especificacion,
                matrices.x_entrenamiento,
                y_entrenamiento[:, posicion_etiqueta],
                matrices.x_validacion,
            )
    return proba_entrenamiento, proba_validacion


# ===== SECCION 5: metricas, bootstrap, rebanadas y transiciones =====


def vector_umbrales(umbrales: dict[str, float]) -> NDArray[np.float64]:
    """Convierte el diccionario de umbrales en vector en el orden de etiquetas.

    Args:
        umbrales: umbral por etiqueta.

    Returns:
        Arreglo ``(3,)`` en el orden de :data:`ETIQUETAS`.
    """
    return np.array([umbrales[etiqueta] for etiqueta in ETIQUETAS])


def f1_media_rapida(y_verdadera: NDArray[np.int_], prediccion: NDArray[np.int_]) -> float:
    """Calcula la F1 media por columna de forma vectorizada.

    Es numericamente identica a ``metricas_multietiqueta(...)['f1_media_columnas']``
    (misma definicion binaria con ``zero_division=0``) pero sin la sobrecarga de
    sklearn, necesaria para los 2000 remuestreos del bootstrap.

    Args:
        y_verdadera: matriz binaria verdadera ``(n, k)``.
        prediccion: matriz binaria predicha ``(n, k)``.

    Returns:
        Media de la F1 binaria de la clase positiva por columna.
    """
    valores = []
    for posicion in range(y_verdadera.shape[1]):
        verdad = y_verdadera[:, posicion].astype(bool)
        aciertos = prediccion[:, posicion].astype(bool)
        verdaderos_positivos = int(np.count_nonzero(verdad & aciertos))
        falsos_positivos = int(np.count_nonzero(~verdad & aciertos))
        falsos_negativos = int(np.count_nonzero(verdad & ~aciertos))
        denominador = 2 * verdaderos_positivos + falsos_positivos + falsos_negativos
        valores.append(2 * verdaderos_positivos / denominador if denominador else 0.0)
    return float(np.mean(valores))


def verificar_metrica_rapida(proba: NDArray[np.float64], y_verdadera: NDArray[np.int_], umbrales: dict[str, float]) -> None:
    """Comprueba que la F1 rapida coincide con la metrica oficial del proyecto.

    Args:
        proba: probabilidades de referencia.
        y_verdadera: etiquetas verdaderas.
        umbrales: umbral por etiqueta.

    Raises:
        RuntimeError: si las dos implementaciones difieren.
    """
    prediccion = (proba >= vector_umbrales(umbrales)).astype(int)
    oficial = metricas_multietiqueta(y_verdadera, prediccion)["f1_media_columnas"]
    rapida = f1_media_rapida(y_verdadera, prediccion)
    if abs(oficial - rapida) > 1e-12:
        raise RuntimeError(f"La F1 rapida {rapida} no coincide con la oficial {oficial}")


def f1_por_pliegue(
    proba: NDArray[np.float64], y_verdadera: NDArray[np.int_], pliegue_de_fila: NDArray[np.int_], umbrales: dict[str, float]
) -> list[float]:
    """Calcula la F1 media por columna en cada pliegue.

    Args:
        proba: probabilidades ``(n, 3)``.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        pliegue_de_fila: numero de pliegue (1..k) de cada fila.
        umbrales: umbral por etiqueta.

    Returns:
        Lista de F1 por pliegue en orden creciente de pliegue.
    """
    valores = []
    for numero in sorted(np.unique(pliegue_de_fila).tolist()):
        mascara = pliegue_de_fila == numero
        valores.append(
            f1_media_rapida(
                y_verdadera[mascara],
                (proba[mascara] >= vector_umbrales(umbrales)).astype(int),
            )
        )
    return valores


def bootstrap_delta(
    proba_referencia: NDArray[np.float64],
    proba_variante: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    umbrales_referencia: dict[str, float],
    umbrales_variante: dict[str, float],
) -> dict[str, Any]:
    """Calcula el IC bootstrap pareado del delta de F1 a nivel de tweet.

    Args:
        proba_referencia: probabilidades de la referencia.
        proba_variante: probabilidades de la variante.
        y_verdadera: etiquetas verdaderas.
        umbrales_referencia: umbrales de la referencia.
        umbrales_variante: umbrales de la variante.

    Returns:
        Diccionario con media, cuantiles 2.5/97.5, fraccion positiva y numero de
        remuestreos.
    """
    generador = np.random.default_rng(RANDOM_STATE)
    n_filas = len(y_verdadera)
    prediccion_ref = (proba_referencia >= vector_umbrales(umbrales_referencia)).astype(int)
    prediccion_var = (proba_variante >= vector_umbrales(umbrales_variante)).astype(int)
    deltas = np.empty(CONFIG["n_bootstrap"], dtype=float)
    for iteracion in range(CONFIG["n_bootstrap"]):
        muestra = generador.integers(0, n_filas, size=n_filas)
        verdad = y_verdadera[muestra]
        deltas[iteracion] = f1_media_rapida(verdad, prediccion_var[muestra]) - f1_media_rapida(
            verdad, prediccion_ref[muestra]
        )
    return {
        "media": float(np.mean(deltas)),
        "ic_inferior": float(np.quantile(deltas, 0.025)),
        "ic_superior": float(np.quantile(deltas, 0.975)),
        "fraccion_positiva": float(np.mean(deltas > 0)),
        "remuestreos": int(CONFIG["n_bootstrap"]),
    }


def resumir_probabilidades(
    nombre: str,
    proba: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
    referencias: dict[str, Any],
) -> dict[str, Any]:
    """Resume una variante frente a las dos referencias del protocolo v2.

    Args:
        nombre: identificador de la variante.
        proba: probabilidades OOF de la variante.
        y_verdadera: etiquetas verdaderas.
        pliegue_de_fila: numero de pliegue de cada fila.
        referencias: referencias pre-registradas (0.7853, 0.7772 y sus pliegues).

    Returns:
        Diccionario con F1 global, por etiqueta, por pliegue y deltas.
    """
    sin_umbral = {etiqueta: 0.5 for etiqueta in ETIQUETAS}
    metrica = metricas_multietiqueta(y_verdadera, (proba >= 0.5).astype(int))
    f1_pliegues = f1_por_pliegue(proba, y_verdadera, pliegue_de_fila, sin_umbral)
    referencia = np.array(referencias["meta_anidado_pliegues"])
    v03 = np.array(referencias["v03_pliegues"])
    return {
        "variante": nombre,
        "corrected_cv": metrica["f1_media_columnas"],
        "f1_por_etiqueta": metrica["f1_por_etiqueta"],
        "f1_pliegues": f1_pliegues,
        "std_pliegues": float(np.std(f1_pliegues)),
        "delta_vs_v2a": metrica["f1_media_columnas"] - referencias["meta_anidado_cv"],
        "delta_vs_v03": metrica["f1_media_columnas"] - referencias["v03_umbrales_cv"],
        "pliegues_mejorados_vs_v2a": int(np.sum(np.array(f1_pliegues) > referencia)),
        "pliegues_mejorados_vs_v03": int(np.sum(np.array(f1_pliegues) > v03)),
        "f1_pliegues_base_v03": referencias["v03_pliegues"],
        "f1_pliegues_base_m2": referencias["m2_pliegues"],
    }


def evaluar_complementariedad(
    nombre: str,
    proba_variante: NDArray[np.float64],
    proba_m2: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    referencia_m2: float,
) -> dict[str, Any]:
    """Mide la complementariedad por promedio de probabilidades con M2 (V03 a 0.5).

    Args:
        nombre: identificador de la variante.
        proba_variante: probabilidades de la variante.
        proba_m2: probabilidades OOF de M2.
        y_verdadera: etiquetas verdaderas.
        referencia_m2: F1 de M2 a 0.5 en el desarrollo (criterio de ganancia).

    Returns:
        Diccionario con la F1 del ensemble, la ganancia y el bootstrap del delta
        (solo si la ganancia es positiva).
    """
    sin_umbral = {etiqueta: 0.5 for etiqueta in ETIQUETAS}
    proba_ensemble = (proba_variante + proba_m2) / 2.0
    f1_ensemble = f1_media_rapida(y_verdadera, (proba_ensemble >= 0.5).astype(int))
    ganancia = f1_ensemble - referencia_m2
    resultado = {
        "variante": nombre,
        "corrected_cv_ensemble": f1_ensemble,
        "ensemble_gain": ganancia,
        "cumple_umbral_gain": bool(ganancia >= CONFIG["ensemble_gain_minimo"]),
    }
    if ganancia > 0:
        resultado["bootstrap"] = bootstrap_delta(proba_m2, proba_ensemble, y_verdadera, sin_umbral, sin_umbral)
        resultado["aporta_ic_positivo"] = bool(resultado["bootstrap"]["ic_inferior"] > 0)
    else:
        resultado["aporta_ic_positivo"] = False
    return resultado


def f1_por_etiqueta_con_umbrales(
    proba: NDArray[np.float64], y_verdadera: NDArray[np.int_], umbrales: dict[str, float]
) -> dict[str, float]:
    """Calcula la F1 por etiqueta con umbrales dados.

    Args:
        proba: probabilidades ``(n, 3)``.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        umbrales: umbral por etiqueta.

    Returns:
        F1 de la clase positiva por etiqueta.
    """
    prediccion = (proba >= vector_umbrales(umbrales)).astype(int)
    return dict(metricas_multietiqueta(y_verdadera, prediccion)["f1_por_etiqueta"])


def metricas_rebanada(
    proba: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    umbrales: dict[str, float],
    mascara: NDArray[np.bool_],
    posicion_etiqueta: int = 2,
) -> dict[str, float]:
    """Calcula recall, precision y F1 de una etiqueta dentro de una rebanada.

    Args:
        proba: probabilidades ``(n, 3)``.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        umbrales: umbral por etiqueta.
        mascara: mascara booleana de la rebanada.
        posicion_etiqueta: columna a evaluar (por defecto AG).

    Returns:
        Diccionario con ``n``, positivos y las tres metricas.
    """
    prediccion = (proba[mascara] >= vector_umbrales(umbrales)).astype(int)
    verdad = y_verdadera[mascara]
    columna_verdad = verdad[:, posicion_etiqueta].astype(bool)
    columna_pred = prediccion[:, posicion_etiqueta].astype(bool)
    verdaderos_positivos = int(np.count_nonzero(columna_verdad & columna_pred))
    falsos_positivos = int(np.count_nonzero(~columna_verdad & columna_pred))
    falsos_negativos = int(np.count_nonzero(columna_verdad & ~columna_pred))
    precision = verdaderos_positivos / (verdaderos_positivos + falsos_positivos) if verdaderos_positivos + falsos_positivos else 0.0
    recall = verdaderos_positivos / (verdaderos_positivos + falsos_negativos) if verdaderos_positivos + falsos_negativos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": int(np.count_nonzero(mascara)),
        "positivos": int(np.count_nonzero(columna_verdad)),
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def transiciones_frente_a_v03(
    proba_v03: NDArray[np.float64], proba_meta: NDArray[np.float64], y_verdadera: NDArray[np.int_], umbrales_v03: dict[str, float]
) -> dict[str, Any]:
    """Cuenta los aciertos y errores que el meta corrige e introduce frente a V03.

    Args:
        proba_v03: probabilidades de V03 (M2).
        proba_meta: probabilidades del mejor meta.
        y_verdadera: etiquetas verdaderas.
        umbrales_v03: umbrales de V03.

    Returns:
        Diccionario por etiqueta con ``fp_corregidos`` (V03 decia 1 con verdad 0
        y el meta dice 0), ``fn_corregidos`` (V03 decia 0 con verdad 1 y el meta
        dice 1), ``fp_introducidos``, ``fn_introducidos`` y los positivos de cada
        modelo.
    """
    prediccion_v03 = (proba_v03 >= vector_umbrales(umbrales_v03)).astype(int)
    prediccion_meta = (proba_meta >= 0.5).astype(int)
    detalle: dict[str, Any] = {}
    for posicion, etiqueta in enumerate(ETIQUETAS):
        verdad = y_verdadera[:, posicion].astype(bool)
        v03 = prediccion_v03[:, posicion].astype(bool)
        meta = prediccion_meta[:, posicion].astype(bool)
        detalle[etiqueta] = {
            "fp_corregidos": int(np.count_nonzero(~verdad & v03 & ~meta)),
            "fn_corregidos": int(np.count_nonzero(verdad & ~v03 & meta)),
            "fp_introducidos": int(np.count_nonzero(~verdad & ~v03 & meta)),
            "fn_introducidos": int(np.count_nonzero(verdad & v03 & ~meta)),
            "positivos_v03": int(np.count_nonzero(v03)),
            "positivos_meta": int(np.count_nonzero(meta)),
            "positivos_reales": int(np.count_nonzero(verdad)),
        }
    return detalle


def seleccionar_mejor(
    resumenes: dict[str, dict[str, Any]],
    claves_por_variante: dict[str, list[str]],
    n_features_por_conjunto: dict[str, int],
) -> dict[str, Any]:
    """Elige la mejor especificacion de cada variante por corrected_cv.

    Args:
        resumenes: resumen por clave de especificacion.
        claves_por_variante: claves agrupadas por variante.
        n_features_por_conjunto: numero de columnas de cada conjunto de features.

    Returns:
        Diccionario ``variante -> resumen de la mejor clave`` con el parametro
        elegido y el detalle por parametro.
    """
    mejores: dict[str, Any] = {}
    for variante, claves in claves_por_variante.items():
        ordenadas = sorted(claves, key=lambda clave: resumenes[clave]["corrected_cv"], reverse=True)
        fila = dict(resumenes[ordenadas[0]])
        fila["variante"] = variante
        fila["mejor_key"] = ordenadas[0]
        fila["mejor_parametro"] = float(ordenadas[0].split("|")[-1])
        fila["conjunto"] = ordenadas[0].split("|")[1]
        fila["modelo"] = ordenadas[0].split("|")[2]
        fila["n_features"] = n_features_por_conjunto[fila["conjunto"]]
        fila["por_parametro"] = {clave.split("|")[-1]: round(resumenes[clave]["corrected_cv"], 4) for clave in claves}
        mejores[variante] = fila
    return mejores


# ===== SECCION 6: figura y artefactos =====


def generar_figura(
    tabla: pd.DataFrame,
    referencias: dict[str, Any],
    f1_etiquetas: dict[str, dict[str, float]],
    rebanada: dict[str, Any],
) -> None:
    """Genera la figura comparativa de variantes, etiquetas y rebanada colectiva.

    Args:
        tabla: tabla resumen de variantes.
        referencias: referencias pre-registradas.
        f1_etiquetas: F1 por etiqueta de V03, meta de bases y mejor variante.
        rebanada: recall de AG en la rebanada colectiva por modelo.
    """
    figura, (eje_a, eje_b) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.25, 1]})
    colores = ["#4c72b0" if "META" in nombre else "#8c8c8c" for nombre in tabla["variante"]]
    mejor_indice = int(np.argmax(tabla["corrected_cv"].to_numpy()))
    colores[mejor_indice] = "#dd8452"
    eje_a.barh(tabla["variante"], tabla["corrected_cv"], color=colores)
    eje_a.axvline(referencias["m2_umbral_05_cv"], color="#d62728", linestyle="--", linewidth=1.1, label=f"M2 a 0.5 ({referencias['m2_umbral_05_cv']:.4f})")
    eje_a.axvline(referencias["v03_umbrales_cv"], color="#2ca02c", linestyle="-.", linewidth=1.1, label=f"V03 umbrales ({referencias['v03_umbrales_cv']:.4f})")
    eje_a.axvline(referencias["meta_anidado_cv"], color="#9467bd", linestyle=":", linewidth=1.1, label=f"v2-A meta P1 ({referencias['meta_anidado_cv']:.4f})")
    eje_a.axvline(
        referencias["meta_anidado_cv"] + CONFIG["mejora_minima"],
        color="#7f7f7f",
        linestyle="-",
        linewidth=1.0,
        label=f"adopcion ({referencias['meta_anidado_cv'] + CONFIG['mejora_minima']:.4f})",
    )
    for posicion, valor in enumerate(tabla["corrected_cv"]):
        eje_a.text(valor + 0.0004, posicion, f"{valor:.4f}", va="center", fontsize=8)
    eje_a.set_xlabel("corrected_cv (F1 media por columna, umbral 0.5)")
    eje_a.set_title("P6 - Variantes del meta (desarrollo n=1483, protocolo v2)")
    eje_a.legend(loc="lower right", fontsize=7)
    limite_inferior = min(tabla["corrected_cv"].min(), referencias["m2_umbral_05_cv"]) - 0.004
    eje_a.set_xlim(limite_inferior, tabla["corrected_cv"].max() + 0.005)

    nombres = list(f1_etiquetas)
    posiciones = np.arange(len(ETIQUETAS))
    ancho = 0.8 / max(len(nombres), 1)
    for indice, nombre in enumerate(nombres):
        valores = [f1_etiquetas[nombre][etiqueta] for etiqueta in ETIQUETAS]
        eje_b.bar(posiciones + indice * ancho, valores, ancho, label=nombre)
    eje_b.set_xticks(posiciones + ancho * (len(nombres) - 1) / 2)
    eje_b.set_xticklabels(list(ETIQUETAS))
    eje_b.set_ylim(0.5, 1.0)
    eje_b.set_ylabel("F1 de la clase positiva")
    eje_b.set_title("F1 por etiqueta y recall de AG en la rebanada colectiva")
    for indice, nombre in enumerate(nombres):
        valor = rebanada[nombre]["recall"]
        eje_b.text(
            posiciones[-1] + indice * ancho,
            0.52,
            f"AG rec. col.\n{valor:.3f}",
            ha="center",
            fontsize=7,
            color="#333333",
        )
    eje_b.legend(loc="upper left", fontsize=8)
    figura.tight_layout()
    FIGURA_PATH.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(FIGURA_PATH, dpi=150)
    plt.close(figura)


def escribir_reporte_csv(tabla_filas: list[dict[str, Any]]) -> pd.DataFrame:
    """Escribe el CSV comparativo de variantes.

    Args:
        tabla_filas: filas del resumen.

    Returns:
        DataFrame escrito.
    """
    tabla = pd.DataFrame(tabla_filas)
    REPORTE_CSV.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(REPORTE_CSV, index=False)
    return tabla


def registrar_experimentos(mejores: dict[str, dict[str, Any]], duracion: float, referencias: dict[str, Any]) -> None:
    """Anade las filas ``E2-18-*`` a ``experiments.csv``.

    Args:
        mejores: resumen de la mejor especificacion de cada variante.
        duracion: segundos totales del experimento.
        referencias: referencias pre-registradas.
    """
    filas = []
    for variante, resumen in mejores.items():
        filas.append(
            {
                "experiment_id": f"E2-18-{variante}",
                "competition": "KAGGLE_TOXIC_2026_PUJ",
                "question": "Aportan los canales de targeting (y los rasgos estructurales) al meta-modelo de dependencia de P1?",
                "hypothesis": "La fusion tardia regularizada de scores OOF de bases + canales de targeting + incertidumbre + estructura supera el meta de P1 (0.7853)",
                "preprocessing": "sin texto: scores OOF de 4 bases y 2 canales con margenes y entropias + rasgos estructurales",
                "feature_groups": "META_SCORES+META_UNCERTAINTY+TARGETING_CHANNELS+STRUCTURAL",
                "representation": f"features densas ({resumen['n_features']} columnas) estandarizadas en el pliegue",
                "model": f"{resumen['modelo']} por etiqueta ({variante}, {resumen['mejor_parametro']})",
                "hyperparameters": json.dumps(
                    {
                        "variante": variante,
                        "conjunto": resumen["conjunto"],
                        "modelo": resumen["modelo"],
                        "mejor_parametro": resumen["mejor_parametro"],
                        "por_parametro": resumen["por_parametro"],
                    },
                    ensure_ascii=False,
                ),
                "validation": "5 pliegues corregidos v2 (grupos fusionados), meta ajustado fold a fold; variantes -ANID con los 4 bases reentrenados dentro del pliegue",
                "f1": round(resumen["corrected_cv"], 4),
                "macro_f1": "",
                "weighted_f1": "",
                "precision": "",
                "recall": "",
                "training_time": round(duracion, 2),
                "dimension": int(resumen["n_features"]),
                "result": json.dumps(
                    {
                        "corrected_cv": resumen["corrected_cv"],
                        "f1_por_etiqueta": resumen["f1_por_etiqueta"],
                        "pliegues": resumen["f1_pliegues"],
                        "std_pliegues": resumen["std_pliegues"],
                        "delta_vs_v2a_0.7853": resumen["delta_vs_v2a"],
                        "delta_vs_v03_0.7772": resumen["delta_vs_v03"],
                        "pliegues_mejorados_vs_v2a": resumen["pliegues_mejorados_vs_v2a"],
                        "ensemble_gain_vs_m2": resumen["ensemble_gain"],
                        "bootstrap_vs_v2a": resumen["bootstrap_vs_v2a"],
                    },
                    ensure_ascii=False,
                ),
                "interpretation": (
                    "Meta de segunda generacion sobre el desarrollo; comparado con v2-A (0.7853) y "
                    "V03 (0.7772). El anidamiento (variantes -ANID) reentrena los 4 bases en cada pliegue."
                ),
                "decision": "ADOPTADO (Gate P6)" if resumen.get("adoptado", False) else "No adoptado / diagnostico",
                "date": CONFIG["fecha"],
                "script": "experiments/17_stacking_v2.py",
                "notes": (
                    "historical_cv no aplica (protocolo v2); holdout congelado no utilizado; "
                    f"referencia v2-A {referencias['meta_anidado_cv']:.4f}"
                ),
            }
        )
    existente = pd.read_csv(EXPERIMENTS_CSV)
    identificadores = {fila["experiment_id"] for fila in filas}
    existente = existente[~existente["experiment_id"].isin(identificadores)]
    existente.to_csv(EXPERIMENTS_CSV, index=False)
    pd.DataFrame(filas).to_csv(EXPERIMENTS_CSV, mode="a", header=False, index=False)
    log(f"filas registradas: {len(filas)}")


def escribir_contexto(lineas: list[str]) -> None:
    """Reemplaza o anade la seccion de P6 en ``reports/contexto.md``.

    Args:
        lineas: lineas de la seccion (sin encabezado).
    """
    encabezado = "## Fase v2 - P6 (2026-09-25)"
    bloque = "\n".join([encabezado, *lineas])
    texto = CONTEXTO_MD.read_text(encoding="utf-8")
    if encabezado in texto:
        inicio = texto.index(encabezado)
        fin = texto.find("\n## ", inicio + len(encabezado))
        fin = len(texto) if fin == -1 else fin + 1
        texto = texto[:inicio] + bloque + "\n" + texto[fin:]
    else:
        texto = texto.rstrip("\n") + "\n\n" + bloque + "\n"
    CONTEXTO_MD.write_text(texto, encoding="utf-8")
    log("seccion de P6 escrita en reports/contexto.md")


def actualizar_todo(decision: dict[str, Any], mejor: dict[str, Any], canales_aportan: bool) -> None:
    """Actualiza P6.1-P6.3 y el Gate P6 en ``tasks/todo.md``.

    Args:
        decision: decision del gate.
        mejor: resumen del mejor meta.
        canales_aportan: si los canales mejoran al meta de bases.
    """
    aporte_canales = (
        "los canales SI aportan al meta" if canales_aportan else "los canales NO aportan al meta (linea cerrada)"
    )
    reemplazos = [
        (
            "- [pendiente] P6.1 Canales: texto, sintaxis, lexico/emocion, targeting, dependencia de etiquetas",
            (
                "- [hecho] P6.1 Canales: solo habia 2 canales OOF disponibles (4 bases de texto y 2 de targeting "
                f"CH-TGT+DIST y TXT+CH); el meta usa {mejor['n_features']} features (scores, margenes, entropias y "
                f"estructurales base+extra); sintaxis/lexico/emocion no existian aun como canales OOF: {aporte_canales}"
            ),
        ),
        (
            "- [pendiente] P6.2 Meta-modelo con OOF anidado + bootstrap por tweet (IC 95 %)",
            (
                "- [hecho] P6.2 Meta-modelo con OOF anidado + bootstrap por tweet (2000 remuestreos): mejor "
                f"{mejor['variante']} corrected_cv {mejor['corrected_cv']:.4f} (delta vs v2-A "
                f"{mejor['delta_vs_v2a']:+.4f}, pliegues {mejor['pliegues_mejorados_vs_v2a']}/5, IC bootstrap del delta "
                f"[{mejor['bootstrap_vs_v2a']['ic_inferior']:+.4f},{mejor['bootstrap_vs_v2a']['ic_superior']:+.4f}])"
            ),
        ),
        (
            "- [pendiente] P6.3 Gate P6",
            (
                f"- [hecho] P6.3 Gate P6: {'ADOPTADO' if decision['adoptado'] else 'CERRADO sin mejora'} "
                f"({decision['justificacion']})"
            ),
        ),
    ]
    texto = TODO_MD.read_text(encoding="utf-8")
    for viejo, nuevo in reemplazos:
        if viejo in texto:
            texto = texto.replace(viejo, nuevo)
    fila_gate = (
        f"| P6 | 2026-09-25 | {'SUPERADO' if decision['adoptado'] else 'CERRADO (sin mejora)'}. E2-18 stacking v2: mejor meta "
        f"{mejor['variante']} con corrected_cv {mejor['corrected_cv']:.4f} "
        f"({mejor['delta_vs_v03']:+.4f} vs V03 0.7772, {mejor['delta_vs_v2a']:+.4f} vs v2-A 0.7853, "
        f"{mejor['pliegues_mejorados_vs_v2a']}/5 pliegues); IC bootstrap del delta vs v2-A "
        f"[{mejor['bootstrap_vs_v2a']['ic_inferior']:+.4f},{mejor['bootstrap_vs_v2a']['ic_superior']:+.4f}]; "
        f"complementariedad ensemble_gain {mejor['ensemble_gain']:+.4f} con IC "
        f"[{mejor['bootstrap_ensemble']['ic_inferior']:+.4f},{mejor['bootstrap_ensemble']['ic_superior']:+.4f}]. "
        f"Canales de targeting en el meta: {'SI' if canales_aportan else 'NO'}. "
        "Evidencia: `experiments/17_stacking_v2.py`, `reports/stacking_v2.{csv,json}`, "
        "`reports/figuras/fig_29_stacking_v2.png` |"
    )
    if "| P6 |" in texto:
        lineas = [linea for linea in texto.splitlines() if not linea.startswith("| P6 |")]
        texto = "\n".join(lineas).rstrip("\n") + "\n"
    texto = texto.rstrip("\n") + "\n" + fila_gate + "\n"
    TODO_MD.write_text(texto, encoding="utf-8")
    log("tasks/todo.md actualizado (P6.1-P6.3 y Gate P6)")


def guardar_artefactos_candidato(
    mejor: dict[str, Any],
    espec: EspecMeta,
    proba: NDArray[np.float64],
    desarrollo: pd.DataFrame,
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
    proba_bases: NDArray[np.float64],
    proba_canales: NDArray[np.float64],
    nombres_canales: Sequence[str],
    nombres: list[str],
) -> dict[str, str]:
    """Guarda el meta candidato y sus probabilidades OOF para P7.

    Los modelos se reajustan sobre TODO el desarrollo con los scores OOF
    congelados (escalador incluido); para la submission final P7 debe regenerar
    los scores de bases y canales del train completo con las mismas recetas.

    Args:
        mejor: resumen del mejor meta.
        espec: especificacion elegida.
        proba: probabilidades OOF del mejor meta.
        desarrollo: DataFrame del desarrollo.
        y_verdadera: etiquetas verdaderas.
        pliegue_de_fila: pliegue de cada fila.
        proba_bases: probabilidades OOF de los 4 bases.
        proba_canales: probabilidades OOF de los canales.
        nombres_canales: nombre de cada canal.
        nombres: nombres de las columnas del meta.

    Returns:
        Diccionario con las rutas guardadas.
    """
    features, _ = construir_bloque_meta(
        proba_bases, proba_canales, nombres_canales, desarrollo["text"], espec.conjunto
    )
    escalador = StandardScaler().fit(features)
    features_escaladas = escalador.transform(features)
    modelos: dict[str, Any] = {}
    for posicion, etiqueta in enumerate(ETIQUETAS):
        if espec.modelo == "lr":
            modelo: Any = LogisticRegression(
                C=espec.parametro,
                class_weight="balanced",
                max_iter=2000,
                solver="liblinear",
                random_state=RANDOM_STATE,
            )
        elif espec.modelo == "ridge":
            modelo = RidgeClassifier(alpha=espec.parametro, class_weight="balanced")
        else:
            modelo = LinearSVC(C=espec.parametro, class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)
        modelo.fit(features_escaladas, y_verdadera[:, posicion])
        modelos[etiqueta] = modelo
    MODELO_CANDIDATO_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "fecha": CONFIG["fecha"],
            "variante": mejor["variante"],
            "conjunto": espec.conjunto,
            "modelo": espec.modelo,
            "mejor_parametro": espec.parametro,
            "nombres_features": nombres,
            "orden_columnas": list(range(len(nombres))),
            "escalador": escalador,
            "modelos": modelos,
            "evaluacion_anidada": bool(espec.variante.endswith("-ANID")),
            "corrected_cv": mejor["corrected_cv"],
            "f1_por_etiqueta": mejor["f1_por_etiqueta"],
            "advertencia": (
                "Ajustado sobre los scores OOF congelados del desarrollo; P7 debe regenerar los scores de "
                "bases y canales del train completo con las mismas recetas para predecir test."
            ),
        },
        MODELO_CANDIDATO_PATH,
    )
    np.savez_compressed(
        OOF_META_PATH,
        ids=desarrollo["id"].astype(int).to_numpy(),
        y_true=y_verdadera,
        pliegue=pliegue_de_fila,
        proba=proba,
        etiquetas=np.array(list(ETIQUETAS)),
        variante=np.array([mejor["variante"]]),
        conjunto=np.array([espec.conjunto]),
        modelo=np.array([espec.modelo]),
        parametro=np.array([espec.parametro]),
        nombres=np.array(nombres),
    )
    log(f"artefactos de P7 guardados: {MODELO_CANDIDATO_PATH.name}, {OOF_META_PATH.name}")
    return {"modelo": str(MODELO_CANDIDATO_PATH.relative_to(PROJECT_ROOT)), "oof": str(OOF_META_PATH.relative_to(PROJECT_ROOT))}


def cargar_rebanada_colectiva(desarrollo: pd.DataFrame) -> tuple[NDArray[np.bool_], str]:
    """Reutiliza el script 16 para obtener la rebanada de referencia colectiva.

    Args:
        desarrollo: DataFrame del desarrollo.

    Returns:
        Tupla ``(mascara, modo_anotacion)``.
    """
    ruta = PROJECT_ROOT / "experiments" / "16_targeting_v2.py"
    especificacion = importlib.util.spec_from_file_location("experimento_16_rebanada", ruta)
    if especificacion is None or especificacion.loader is None:
        raise ImportError(f"No se pudo cargar {ruta}")
    modulo = importlib.util.module_from_spec(especificacion)
    # El script 16 usa dataclasses: sin registrarlo en sys.modules, exec_module
    # no puede resolver su propio modulo durante la creacion de las clases.
    sys.modules[especificacion.name] = modulo
    especificacion.loader.exec_module(modulo)
    constantes, _ = modulo.cargar_constantes_error_analysis()
    documentos, modo = modulo.cargar_o_crear_anotaciones(desarrollo)
    mascara = modulo.rebanada_colectiva_referencia(documentos, constantes)
    return mascara, modo


# ===== SECCION 7: main =====


def main() -> None:
    """Ejecuta P6 (stacking v2), imprime el resumen y guarda los artefactos."""
    inicio = time.perf_counter()
    desarrollo, grupos, pliegues = cargar_desarrollo()
    y_verdadera = desarrollo[list(ETIQUETAS)].to_numpy()
    pliegue_de_fila = np.zeros(len(desarrollo), dtype=int)
    for numero, (_, idx_validacion) in enumerate(pliegues, start=1):
        pliegue_de_fila[idx_validacion] = numero
    verificacion = verificar_datos(desarrollo, grupos, pliegue_de_fila)
    referencias = cargar_referencias()
    log(
        f"referencias: v2-A meta anidado {referencias['meta_anidado_cv']:.4f}, "
        f"V03 umbrales {referencias['v03_umbrales_cv']:.4f}, M2 a 0.5 {referencias['m2_umbral_05_cv']:.4f}"
    )
    proba_bases, nombres_bases, proba_canales, canales, info_oof = cargar_oof(
        desarrollo, grupos, y_verdadera, pliegue_de_fila
    )
    log(f"OOF cargados: bases {list(nombres_bases)}; canales {list(canales)} (C {info_oof['C_canales']})")

    # Referencia V03 (M2 con umbrales por etiqueta) reconstruida del OOF congelado.
    posicion_m2 = nombres_bases.index("M2_union_lr_c2")
    proba_m2 = proba_bases[:, posicion_m2, :]
    verificacion["diferencia_replicacion_v03"] = verificar_replicacion_v03(proba_m2, y_verdadera, referencias)
    sin_umbral = {etiqueta: 0.5 for etiqueta in ETIQUETAS}
    verificar_metrica_rapida(proba_m2, y_verdadera, sin_umbral)
    log("F1 rapida verificada contra la metrica oficial del proyecto")

    # Conjuntos de features congeladas (OOF).
    bloques: dict[str, list[str]] = {}
    matrices: dict[str, NDArray[np.float64]] = {}
    for conjunto in CONJUNTOS:
        matriz, nombres = construir_bloque_meta(proba_bases, proba_canales, canales, desarrollo["text"], conjunto)
        matrices[conjunto] = matriz
        bloques[conjunto] = nombres
        log(f"conjunto {conjunto}: {matriz.shape[1]} features")

    variantes: tuple[tuple[str, str, str, Sequence[float]], ...] = (
        ("META-BASES", "bases", "lr", CONFIG["meta_C"]),
        ("META+CH", "bases_canales", "lr", CONFIG["meta_C"]),
        ("META+CH+ESTR", "bases_canales_ampliado", "lr", CONFIG["meta_C"]),
        ("META-BASES-RIDGE", "bases", "ridge", CONFIG["ridge_alpha"]),
        ("META+CH+ESTR-RIDGE", "bases_canales_ampliado", "ridge", CONFIG["ridge_alpha"]),
        ("META+CH+ESTR-SVC", "bases_canales_ampliado", "svc", CONFIG["svc_C"]),
    )
    especificaciones = expandir_especificaciones(variantes)
    claves_por_variante: dict[str, list[str]] = {}
    for espec in especificaciones:
        claves_por_variante.setdefault(espec.variante, []).append(espec.key)

    # Evaluacion con el OOF congelado (meta ajustado fold a fold).
    features_por_conjunto: dict[str, dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]]] = {}
    for conjunto, matriz in matrices.items():
        features_por_conjunto[conjunto] = {
            numero: (matriz[idx_entrenamiento], matriz[idx_validacion])
            for numero, (idx_entrenamiento, idx_validacion) in enumerate(pliegues, start=1)
        }
    probas_frozen: dict[str, NDArray[np.float64]] = {}
    for conjunto, por_pliegue in features_por_conjunto.items():
        especificaciones_conjunto = [espec for espec in especificaciones if espec.conjunto == conjunto]
        if not especificaciones_conjunto:
            continue
        probas_frozen.update(evaluar_meta_por_pliegue(por_pliegue, pliegues, y_verdadera, especificaciones_conjunto))
        log(f"meta congelado evaluado para {conjunto} ({len(especificaciones_conjunto)} especificaciones)")

    # Verificacion estrictamente anidada: los 4 bases se reentrenan por pliegue.
    modulo_base = cargar_modulo_base()
    construir_matrices = modulo_base.construir_matrices
    combinatoria = combinaciones(desarrollo)
    anidado_por_pliegue: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for numero, (idx_entrenamiento, idx_validacion) in enumerate(pliegues, start=1):
        matrices_union = construir_matrices_exteriores(
            construir_matrices, desarrollo["text"], idx_entrenamiento, idx_validacion, combinatoria, grupos
        )
        proba_entrenamiento, proba_validacion = probas_bases_anidadas(
            construir_matrices,
            modulo_base,
            desarrollo["text"],
            y_verdadera,
            idx_entrenamiento,
            idx_validacion,
            combinatoria,
            grupos,
            matrices_union,
        )
        anidado_por_pliegue[numero] = (proba_entrenamiento, proba_validacion)
        log(f"pliegue {numero}/5 anidado listo en {time.perf_counter() - inicio:.1f} s")

    variantes_anidadas: tuple[tuple[str, str, str, Sequence[float]], ...] = (
        ("META-BASES-ANID", "bases", "lr", CONFIG["meta_C"]),
        ("META+CH+ESTR-ANID", "bases_canales_ampliado", "lr", CONFIG["meta_C"]),
    )
    especificaciones_anidadas = expandir_especificaciones(variantes_anidadas)
    for espec in especificaciones_anidadas:
        claves_por_variante.setdefault(espec.variante, []).append(espec.key)
    features_anidadas_por_conjunto: dict[str, dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]]] = {}
    for conjunto in {espec.conjunto for espec in especificaciones_anidadas}:
        features_anidadas_por_conjunto[conjunto] = {}
        for numero, (idx_entrenamiento, idx_validacion) in enumerate(pliegues, start=1):
            proba_entrenamiento, proba_validacion = anidado_por_pliegue[numero]
            matriz_entrenamiento, _ = construir_bloque_meta(
                proba_entrenamiento,
                proba_canales[idx_entrenamiento],
                canales,
                desarrollo["text"].iloc[idx_entrenamiento],
                conjunto,
            )
            matriz_validacion, _ = construir_bloque_meta(
                proba_validacion,
                proba_canales[idx_validacion],
                canales,
                desarrollo["text"].iloc[idx_validacion],
                conjunto,
            )
            features_anidadas_por_conjunto[conjunto][numero] = (matriz_entrenamiento, matriz_validacion)
    probas_anidadas: dict[str, NDArray[np.float64]] = {}
    for conjunto, por_pliegue in features_anidadas_por_conjunto.items():
        especificaciones_conjunto = [espec for espec in especificaciones_anidadas if espec.conjunto == conjunto]
        probas_anidadas.update(evaluar_meta_por_pliegue(por_pliegue, pliegues, y_verdadera, especificaciones_conjunto))
    log("meta anidado estricto evaluado (reproduccion de P1)")

    # Resumen por especificacion y por variante.
    todas_las_probas = {**probas_frozen, **probas_anidadas}
    resumenes_por_key = {
        clave: resumir_probabilidades(clave, proba, y_verdadera, pliegue_de_fila, referencias)
        for clave, proba in todas_las_probas.items()
    }
    mejores = seleccionar_mejor(
        resumenes_por_key, claves_por_variante, {nombre: len(bloques[nombre]) for nombre in bloques}
    )
    for variante, fila in mejores.items():
        log(
            f"{variante}: corrected_cv {fila['corrected_cv']:.4f} (mejor {fila['mejor_parametro']:g}; "
            f"delta vs v2-A {fila['delta_vs_v2a']:+.4f}; pliegues {fila['pliegues_mejorados_vs_v2a']}/5)"
        )

    # Referencia anidada reproducida y comparacion con META-BASES (aportan los canales?).
    referencia_anidada = mejores["META-BASES-ANID"]
    mejor_variante = max(mejores, key=lambda nombre: mejores[nombre]["corrected_cv"])
    mejor = mejores[mejor_variante]
    esperado = referencias["meta_anidado_cv"]
    diferencia_anidada = abs(referencia_anidada["corrected_cv"] - esperado)
    log(
        f"reproduccion del meta anidado de P1: {referencia_anidada['corrected_cv']:.4f} vs {esperado:.4f} "
        f"(dif {diferencia_anidada:.2e}, mejor C de P1 {referencias['meta_anidado_mejor_C']:g})"
    )
    if diferencia_anidada > CONFIG["tolerancia_replicacion_anidada"]:
        log("AVISO: la reproduccion del meta anidado de P1 se desvia de la referencia")

    espec_mejor = next(espec for espec in (*especificaciones, *especificaciones_anidadas) if espec.key == mejor["mejor_key"])
    proba_mejor = todas_las_probas[mejor["mejor_key"]]
    referencia_mejor_proba = todas_las_probas[referencia_anidada["mejor_key"]]
    referencia_bases_proba = todas_las_probas[mejores["META-BASES"]["mejor_key"]]

    # Bootstrap pareado frente al meta anidado de P1 y complementariedad frente a M2.
    for variante in mejores:
        proba_variante = todas_las_probas[mejores[variante]["mejor_key"]]
        mejores[variante]["bootstrap_vs_v2a"] = bootstrap_delta(
            referencia_mejor_proba, proba_variante, y_verdadera, sin_umbral, sin_umbral
        )
        mejores[variante]["ensemble"] = evaluar_complementariedad(
            variante, proba_variante, proba_m2, y_verdadera, referencias["m2_umbral_05_cv"]
        )
        mejores[variante]["ensemble_gain"] = mejores[variante]["ensemble"]["ensemble_gain"]
    mejor = mejores[mejor_variante]
    mejor["bootstrap_ensemble"] = mejor["ensemble"].get(
        "bootstrap",
        {"media": float("nan"), "ic_inferior": float("nan"), "ic_superior": float("nan"), "fraccion_positiva": 0.0},
    )
    comparaciones = {
        "bootstrap_vs_v2a": mejor["bootstrap_vs_v2a"],
        "bootstrap_vs_meta_bases_congelado": bootstrap_delta(
            referencia_bases_proba, proba_mejor, y_verdadera, sin_umbral, sin_umbral
        ),
        "nota_referencia": (
            "El IC frente a v2-A usa las predicciones por tweet del meta anidado reproducido en este script "
            f"({referencia_anidada['corrected_cv']:.4f}); la referencia pre-registrada es {esperado:.4f} "
            f"(diferencia {diferencia_anidada:.1e})."
        ),
    }
    familia_canales = max(
        (nombre for nombre in mejores if nombre in ("META+CH", "META+CH+ESTR")),
        key=lambda nombre: mejores[nombre]["corrected_cv"],
    )
    canales_aportan = bool(
        mejores[familia_canales]["corrected_cv"] > mejores["META-BASES"]["corrected_cv"]
        or mejores[familia_canales]["corrected_cv"] > referencia_anidada["corrected_cv"]
    )
    comparacion_canales = {
        "variante_canales": familia_canales,
        "corrected_cv_canales": mejores[familia_canales]["corrected_cv"],
        "corrected_cv_meta_bases_congelado": mejores["META-BASES"]["corrected_cv"],
        "corrected_cv_meta_bases_anidado": referencia_anidada["corrected_cv"],
        "delta_canales_vs_bases_congelado": mejores[familia_canales]["corrected_cv"] - mejores["META-BASES"]["corrected_cv"],
        "delta_canales_vs_bases_anidado": mejores[familia_canales]["corrected_cv"] - referencia_anidada["corrected_cv"],
        "bootstrap_vs_bases_congelado": bootstrap_delta(
            referencia_bases_proba, todas_las_probas[mejores[familia_canales]["mejor_key"]], y_verdadera, sin_umbral, sin_umbral
        ),
        "bootstrap_vs_bases_anidado": bootstrap_delta(
            referencia_mejor_proba, todas_las_probas[mejores[familia_canales]["mejor_key"]], y_verdadera, sin_umbral, sin_umbral
        ),
        "aporta": canales_aportan,
    }
    log(
        f"canales: {comparacion_canales['variante_canales']} {comparacion_canales['corrected_cv_canales']:.4f} vs "
        f"bases congelado {comparacion_canales['corrected_cv_meta_bases_congelado']:.4f} y anidado "
        f"{comparacion_canales['corrected_cv_meta_bases_anidado']:.4f}"
    )

    # Diagnostico por rebanada y transiciones frente a V03.
    mascara_rebanada, modo_anotacion = cargar_rebanada_colectiva(desarrollo)
    n_rebanada = int(np.count_nonzero(mascara_rebanada))
    log(f"rebanada colectiva: n={n_rebanada} (anotacion {modo_anotacion})")
    etiquetas_diagnostico = {
        "V03": referencias["umbrales_v03"],
        "META-BASES": sin_umbral,
        mejor_variante: sin_umbral,
        "META-BASES-ANID": sin_umbral,
    }
    probas_diagnostico = {
        "V03": proba_m2,
        "META-BASES": referencia_bases_proba,
        mejor_variante: proba_mejor,
        "META-BASES-ANID": referencia_mejor_proba,
    }
    rebanada = {
        nombre: {
            "dentro": metricas_rebanada(proba, y_verdadera, etiquetas_diagnostico[nombre], mascara_rebanada),
            "fuera": metricas_rebanada(proba, y_verdadera, etiquetas_diagnostico[nombre], ~mascara_rebanada),
        }
        for nombre, proba in probas_diagnostico.items()
    }
    transiciones = transiciones_frente_a_v03(proba_m2, proba_mejor, y_verdadera, referencias["umbrales_v03"])

    # Decision (regla pre-registrada).
    objetivo_nivel = referencias["meta_anidado_cv"] + CONFIG["mejora_minima"]
    cumple_nivel = bool(mejor["corrected_cv"] >= objetivo_nivel)
    cumple_pliegues = bool(mejor["pliegues_mejorados_vs_v2a"] >= CONFIG["pliegues_minimos_mejorados"])
    cumple_complementariedad = bool(
        mejor["ensemble"]["ensemble_gain"] >= CONFIG["ensemble_gain_minimo"]
        and mejor["ensemble"].get("aporta_ic_positivo", False)
    )
    adoptado = bool((cumple_nivel and cumple_pliegues) or cumple_complementariedad)
    significativo = bool(mejor["bootstrap_vs_v2a"]["ic_inferior"] > 0)
    if adoptado and cumple_nivel and cumple_pliegues:
        via_adopcion = "nivel"
        justificacion = (
            f"Adoptado por nivel: corrected_cv {mejor['corrected_cv']:.4f} >= {objetivo_nivel:.4f} con "
            f"{mejor['pliegues_mejorados_vs_v2a']}/5 pliegues mejorados."
        )
    elif adoptado:
        via_adopcion = "complementariedad"
        justificacion = (
            f"Adoptado por complementariedad (ensemble_gain {mejor['ensemble']['ensemble_gain']:+.4f}, IC "
            f"[{mejor['bootstrap_ensemble']['ic_inferior']:+.4f},{mejor['bootstrap_ensemble']['ic_superior']:+.4f}]); "
            f"el nivel {mejor['corrected_cv']:.4f} no llega a {objetivo_nivel:.4f} y el IC del delta vs v2-A "
            f"[{mejor['bootstrap_vs_v2a']['ic_inferior']:+.4f},{mejor['bootstrap_vs_v2a']['ic_superior']:+.4f}] "
            f"{'excluye 0' if significativo else 'incluye 0 (mejora de nivel dentro del ruido)'}."
        )
    else:
        via_adopcion = None
        justificacion = "No adoptado: no alcanza el nivel exigido ni la complementariedad."
    decision = {
        "mejor_variante": mejor_variante,
        "mejor_key": mejor["mejor_key"],
        "mejor_parametro": espec_mejor.parametro,
        "modelo": espec_mejor.modelo,
        "conjunto": espec_mejor.conjunto,
        "es_variante_simple_de_p1": bool(espec_mejor.conjunto == "bases"),
        "corrected_cv": mejor["corrected_cv"],
        "criterio_nivel": {"objetivo": objetivo_nivel, "cumple": cumple_nivel},
        "criterio_pliegues": {
            "objetivo": CONFIG["pliegues_minimos_mejorados"],
            "pliegues_mejorados": mejor["pliegues_mejorados_vs_v2a"],
            "cumple": cumple_pliegues,
        },
        "criterio_complementariedad": {
            "objetivo": CONFIG["ensemble_gain_minimo"],
            "ensemble_gain": mejor["ensemble"]["ensemble_gain"],
            "ic_inferior": mejor["bootstrap_ensemble"]["ic_inferior"],
            "cumple": cumple_complementariedad,
        },
        "adoptado": adoptado,
        "via_adopcion": via_adopcion,
        "delta_vs_v2a_significativo": significativo,
        "justificacion": justificacion,
        "canales_aportan_al_meta": canales_aportan,
        "reproduccion_p1": {
            "corrected_cv_reproducido": referencia_anidada["corrected_cv"],
            "corrected_cv_referencia": esperado,
            "diferencia": diferencia_anidada,
            "mejor_parametro_reproducido": referencia_anidada["mejor_parametro"],
            "mejor_parametro_p1": referencias["meta_anidado_mejor_C"],
        },
        "artefactos_p7": None,
    }

    # Artefactos de P7 solo si la regla se cumple.
    for variante in mejores:
        mejores[variante]["adoptado"] = bool(adoptado and variante == mejor_variante)
    if adoptado:
        decision["artefactos_p7"] = guardar_artefactos_candidato(
            mejor,
            espec_mejor,
            proba_mejor,
            desarrollo,
            y_verdadera,
            pliegue_de_fila,
            proba_bases,
            proba_canales,
            canales,
            bloques[espec_mejor.conjunto],
        )

    # Tabla y reporte.
    filas_tabla = []
    for variante in claves_por_variante:
        fila = mejores[variante]
        filas_tabla.append(
            {
                "variante": variante,
                "conjunto": fila["mejor_key"].split("|")[1],
                "modelo": fila["mejor_key"].split("|")[2],
                "mejor_parametro": fila["mejor_parametro"],
                "por_parametro": fila["por_parametro"],
                "n_features": len(bloques[fila["mejor_key"].split("|")[1]]),
                "corrected_cv": round(fila["corrected_cv"], 4),
                "f1_HS": round(fila["f1_por_etiqueta"]["HS"], 4),
                "f1_TR": round(fila["f1_por_etiqueta"]["TR"], 4),
                "f1_AG": round(fila["f1_por_etiqueta"]["AG"], 4),
                "std_pliegues": round(fila["std_pliegues"], 4),
                "delta_vs_v2a": round(fila["delta_vs_v2a"], 4),
                "delta_vs_v03": round(fila["delta_vs_v03"], 4),
                "pliegues_mejorados_vs_v2a": fila["pliegues_mejorados_vs_v2a"],
                "pliegues_mejorados_vs_v03": fila["pliegues_mejorados_vs_v03"],
                "f1_pliegues": "|".join(f"{valor:.4f}" for valor in fila["f1_pliegues"]),
                "ensemble_gain": round(fila["ensemble_gain"], 4),
                "bootstrap_vs_v2a": fila["bootstrap_vs_v2a"],
                "decision": "ADOPTADO (Gate P6)" if (adoptado and variante == mejor_variante) else "No adoptado / diagnostico",
            }
        )
    tabla = escribir_reporte_csv(filas_tabla)

    reporte = {
        "fecha": CONFIG["fecha"],
        "protocolo": "v2 (D-016): grupos fusionados, desarrollo 1483, holdout congelado no usado",
        "pregunta": "Aportan los canales de targeting al meta-modelo de dependencia (v2-A) en la fusion tardia?",
        "verificacion_datos": verificacion,
        "referencias": referencias,
        "insumos": info_oof,
        "config": {clave: list(valor) if isinstance(valor, tuple) else valor for clave, valor in CONFIG.items()},
        "conjuntos": {nombre: {"n_features": len(bloques[nombre]), "nombres": bloques[nombre]} for nombre in bloques},
        "variantes": {
            variante: {
                **{clave: valor for clave, valor in fila.items() if clave != "ensemble"},
                "ensemble": fila["ensemble"],
            }
            for variante, fila in mejores.items()
        },
        "detalle_por_especificacion": resumenes_por_key,
        "comparaciones": comparaciones,
        "comparacion_canales": comparacion_canales,
        "reproduccion_p1_anidada": {
            "corrected_cv": referencia_anidada["corrected_cv"],
            "por_parametro": referencia_anidada["por_parametro"],
            "referencia": esperado,
            "diferencia": diferencia_anidada,
        },
        "rebanada_colectiva": {
            "n": n_rebanada,
            "n_esperado": CONFIG["rebanada_n_esperado"],
            "modo_anotacion": modo_anotacion,
            "metricas": rebanada,
        },
        "transiciones_vs_v03": transiciones,
        "decision": decision,
        "duracion_segundos": round(time.perf_counter() - inicio, 2),
    }
    REPORTE_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORTE_JSON.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    generar_figura(
        tabla,
        referencias,
        {
            "V03": f1_por_etiqueta_con_umbrales(proba_m2, y_verdadera, referencias["umbrales_v03"]),
            "META-BASES-ANID": referencia_anidada["f1_por_etiqueta"],
            mejor_variante: mejor["f1_por_etiqueta"],
        },
        {nombre: datos["dentro"] for nombre, datos in rebanada.items()},
    )
    registrar_experimentos(mejores, reporte["duracion_segundos"], referencias)
    actualizar_todo(decision, mejor, canales_aportan)

    lineas_contexto = [
        (
            f"P6 (E2-18) stacking de segunda generacion sobre el desarrollo (n={verificacion['n_filas']}), "
            "pliegues corregidos v2, holdout congelado intacto; insumos: OOF de 4 bases + 2 canales de targeting "
            "(CH-TGT+DIST, TXT+CH) y 11 rasgos estructurales (longitud, tokens, mencion, hashtag, URL, comillas, "
            "segunda persona, colectivo, risas, mayusculas sostenidas, elongacion)."
        ),
        (
            f"Insumos verificados: etiquetas y pliegues de los dos OOF coinciden con el desarrollo (pliegues del "
            f"targeting identicos a los corregidos); V03 reproducido en {referencias['v03_umbrales_cv']:.4f} con el "
            f"OOF congelado; meta anidado de P1 reproducido en {referencia_anidada['corrected_cv']:.4f} "
            f"(referencia {esperado:.4f}, dif {diferencia_anidada:.1e}, mejor C {referencia_anidada['mejor_parametro']:g} "
            f"vs C {referencias['meta_anidado_mejor_C']:g} de P1)."
        ),
        (
            "Variantes (corrected_cv, mejor hiperparametro, delta vs v2-A 0.7853, pliegues mejorados vs v2-A): "
            + "; ".join(
                f"{fila['variante']} {fila['corrected_cv']:.4f} ({fila['modelo']} {fila['mejor_parametro']:g}, "
                f"{fila['delta_vs_v2a']:+.4f}, {fila['pliegues_mejorados_vs_v2a']}/5)"
                for _, fila in tabla.iterrows()
            )
            + "."
        ),
        (
            f"Canales de targeting en el meta: {'APORTAN' if canales_aportan else 'NO APORTAN'}. "
            f"{comparacion_canales['variante_canales']} {comparacion_canales['corrected_cv_canales']:.4f} vs "
            f"META-BASES congelado {comparacion_canales['corrected_cv_meta_bases_congelado']:.4f} y anidado "
            f"{comparacion_canales['corrected_cv_meta_bases_anidado']:.4f}; IC bootstrap del delta vs bases congelado "
            f"[{comparacion_canales['bootstrap_vs_bases_congelado']['ic_inferior']:+.4f},"
            f"{comparacion_canales['bootstrap_vs_bases_congelado']['ic_superior']:+.4f}]."
        ),
        (
            f"IC bootstrap a nivel de tweet (2000 remuestreos) del mejor meta ({mejor_variante}, "
            f"corrected_cv {mejor['corrected_cv']:.4f}, por etiqueta HS {mejor['f1_por_etiqueta']['HS']:.3f} / "
            f"TR {mejor['f1_por_etiqueta']['TR']:.3f} / AG {mejor['f1_por_etiqueta']['AG']:.3f}) frente al anidado de "
            f"P1: [{mejor['bootstrap_vs_v2a']['ic_inferior']:+.4f},{mejor['bootstrap_vs_v2a']['ic_superior']:+.4f}] "
            f"(fraccion positiva {mejor['bootstrap_vs_v2a']['fraccion_positiva']:.3f}); frente a META-BASES congelado: "
            f"[{comparaciones['bootstrap_vs_meta_bases_congelado']['ic_inferior']:+.4f},"
            f"{comparaciones['bootstrap_vs_meta_bases_congelado']['ic_superior']:+.4f}]."
        ),
        (
            f"Rebanada de referencia colectiva (n={n_rebanada}): recall de AG V03 "
            f"{rebanada['V03']['dentro']['recall']:.3f} -> META-BASES-ANID "
            f"{rebanada['META-BASES-ANID']['dentro']['recall']:.3f} -> mejor meta {rebanada[mejor_variante]['dentro']['recall']:.3f} "
            f"(fuera de la rebanada {rebanada['V03']['fuera']['recall']:.3f} -> {rebanada[mejor_variante]['fuera']['recall']:.3f})."
        ),
        (
            "FP/FN del mejor meta frente a V03 (corregidos/introducidos): "
            + "; ".join(
                f"{etiqueta} {transiciones[etiqueta]['fp_corregidos'] + transiciones[etiqueta]['fn_corregidos']} "
                f"corregidos vs {transiciones[etiqueta]['fp_introducidos'] + transiciones[etiqueta]['fn_introducidos']} "
                f"introducidos"
                for etiqueta in ETIQUETAS
            )
            + "."
        ),
        (
            f"Gate P6: {'ADOPTADO' if adoptado else 'CERRADO SIN MEJORA'} - {justificacion} "
            f"Complementariedad con M2: ensemble_gain {mejor['ensemble']['ensemble_gain']:+.4f} (objetivo "
            f">= {CONFIG['ensemble_gain_minimo']}), IC "
            f"[{mejor['bootstrap_ensemble']['ic_inferior']:+.4f},{mejor['bootstrap_ensemble']['ic_superior']:+.4f}]."
        ),
        (
            "Sin fuga: holdout congelado no utilizado; escaladores, meta-modelos y seleccion de hiperparametros solo "
            "con el entrenamiento de cada pliegue o con el OOF del desarrollo; las variantes -ANID reentrenan los 4 "
            "bases dentro del pliegue exterior. Advertencia: en las variantes con OOF congelado los scores de "
            "entrenamiento del meta provienen de bases que vieron el pliegue evaluado (stacking estandar); P1 midio "
            "ese sesgo en <= 0.0003."
        ),
        (
            f"Duracion: {reporte['duracion_segundos']:.1f} s. Script: experiments/17_stacking_v2.py. Artefactos: "
            "reports/stacking_v2.csv y .json, reports/figuras/fig_29_stacking_v2.png, filas E2-18-* en experiments.csv"
            + (f", {decision['artefactos_p7']}" if decision["artefactos_p7"] else " (sin artefacto de P7: no adoptado)")
            + "."
        ),
    ]
    escribir_contexto(lineas_contexto)

    log("resumen final")
    for _, fila in tabla.iterrows():
        log(
            f"  {fila['variante']}: {fila['corrected_cv']:.4f} "
            f"(delta vs v2-A {fila['delta_vs_v2a']:+.4f}, pliegues {fila['pliegues_mejorados_vs_v2a']}/5, "
            f"ensemble_gain {fila['ensemble_gain']:+.4f})"
        )
    log(f"decision: {'ADOPTADO' if adoptado else 'NO ADOPTADO'} - {justificacion}")
    log(f"duracion total: {reporte['duracion_segundos']:.1f} s")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - reporte explicito y salida con codigo 1
        import traceback

        traceback.print_exc()
        print(f"[P6] ERROR: {error}", file=sys.stderr)
        sys.exit(1)
