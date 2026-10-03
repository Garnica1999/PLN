"""p84_04_sgns_corregido_anclado.py
Fecha: 2026-10-02
Descripcion: EXP-04 (P84-04): SGNS corregido (suma de las K contribuciones negativas) frente a
su control pareado de media 1/K y a una matriz aleatoria congelada, con ancla de implementacion
obligatoria ANTES de entrenar el embedding final. El corpus es EXTERNAL-ONLY
(wikipedia_es/emb_normal/tokens_10M.txt); el downstream es fijo: mean pooling de tokens en
vocabulario (OOV omitidos) + StandardScaler en outer_train + LR C=2.0 balanced liblinear rs=42,
con lectura primaria de umbrales congelados de V03 (0.44/0.51/0.46) y lectura secundaria anidada
(rejilla 0.20-0.80 elegida solo en particiones internas).
Estado del arte aplicado: skip-gram con negative sampling (Mikolov et al.; palabra2vec C++),
correccion del objetivo documentada en P79/P80/P81 del proyecto, convenciones P84 (Gate 0,
bootstrap pareado 10.000/semilla 20260926, ensemble_gain P6/P26).

Especificacion: docs/p84_spec.md seccion 5 (EXP-04) + seccion 1 (convenciones comunes).
Preregistro: docs/p84_preregistro.md (Gate 0, hashes, comparaciones).

Reglas duras de esta corrida:
- Gate 0: hashes congelados (corpus, binario fastText, oof_v03_final.npz) verificados; dev 1483 /
  holdout 0 / test 0; identidad V03 reconstruida por pliegue con tolerance max|delta proba| 1e-12.
- Ancla antes de entrenar el embedding final: (a) identidad algebraica N_suma = K*N_media y
  loss_suma = K*loss_media + (K-1)*P (P = media del log-sigma positivo, tol. relativa 1e-9, K=10);
  (b) gradiente por diferencias finitas del objetivo SUMA (error relativo <= 1e-5); (c) corpus
  plantado de dos familias disjuntas y entrenamiento del binario oficial `skipgram` frente a
  nuestra implementacion equivalente; se exige pureza_nuestra >= pureza_oficial - 0.05. El
  binario oficial se reintenta hasta 5 veces con el MISMO comando y semilla exigiendo un .vec
  finito (aborta de forma intermitente con EncounteredNaNError); si todos los intentos fallan,
  el veredicto es ANCLA NO LOGRADA sin abortar la corrida.
- Brazos S (suma), M (media 1/K) y R (aleatoria congelada). S y M comparten vocabulario, semilla,
  pares, permutacion y negativos; los hashes del flujo de la epoca 1 se verifican.
- loss_suma y loss_media NO se comparan entre si (distinto funcional y distinta magnitud de
  gradiente con el mismo lr); la comparacion S vs M confunde correccion con magnitud del
  gradiente y asi se publica.
- Nunca se abre data/raw/toxic_test.csv ni se evalua el holdout.
- Umbrales anidados por (pliegue, etiqueta) elegidos SOLO sobre las filas outer_train cubiertas
  por al menos un split interno (agrupando los inner_val de los 3 splits; las filas nunca
  cubiertas ya no entran como proba 0 sesgando el umbral); el vectorizador/scaler/LR se ajustan
  fuera de outer_val.
- Se publican las tres lecturas de umbral (congelada 0.44/0.51/0.46, 0.5 y anidada) de S, M y R
  con macro-6, micro, samples, exact_match y precision/recall por etiqueta.
- Firma de reanudacion v2: incluye el sha256 del propio script, de modo que cualquier fix
  invalida los parciales previos (los parciales v1 dejan de reanudarse).

Uso:
    /home/carlos/venv/bin/python experiments/p84_04_sgns_corregido_anclado.py
    /home/carlos/venv/bin/python experiments/p84_04_sgns_corregido_anclado.py --humo
    /home/carlos/venv/bin/python experiments/p84_04_sgns_corregido_anclado.py --reanudar
    /home/carlos/venv/bin/python experiments/p84_04_sgns_corregido_anclado.py --semillas 20261016
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import torch.nn.functional as F
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p73_02_textcnn_vs_random import partir_interna as partir_interna_p73

from src.data import read_train, sha256_of
from src.evaluation import (
    ETIQUETAS,
    bootstrap_diferencia_pareada,
    metrica_macro,
    metricas_multietiqueta,
)
from src.splits import grupos_fusionados, holdout_v2, pliegues_corregidos

# ===== SECCION 0: constantes congeladas, configuracion y utilidades =====

HASH_CORPUS: Final[str] = "00b839da6fbc377ee5375acc9a236e831e46ecf92cc544188b81aef1a826e871"
HASH_FASTTEXT: Final[str] = "59c51cc53802bcad9e7832145f6525a281d22d42e7c476bfbac545c0c392137f"
HASH_OOF_V03: Final[str] = "559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab"
HASH_TRAIN: Final[str] = "3d40e7345b588860cf4744914be88ddc094e4f03591516c8176ab40b7f76d42f"
HASH_GRUPOS: Final[str] = "afabf9ae1c295ea8f203120bf28578cb4bdb4e8f16b89178d820baeaac62e622"

V03_CCV: Final[float] = 0.7772114525165721
V03_F1_AT_05: Final[float] = 0.771778476256122
UMBRALES_CONGELADOS: Final[np.ndarray] = np.array([0.44, 0.51, 0.46], dtype=np.float64)
REJILLA_UMBRALES: Final[np.ndarray] = np.round(np.arange(20, 81, dtype=np.float64) / 100.0, 2)
TOL_IDENTIDAD: Final[float] = 1e-12
REPETICIONES_BOOTSTRAP: Final[int] = 10_000
SEMILLA_BOOTSTRAP: Final[int] = 20260926

FILAS_DEV: Final[int] = 1483
FILAS_HOLDOUT: Final[int] = 248
FILAS_TRAIN: Final[int] = 1731
TAMANOS_PLIEGUES: Final[list[int]] = [297, 297, 296, 296, 297]

BRAZOS: Final[tuple[str, ...]] = ("S", "M", "R")
AGREGACIONES: Final[dict[str, str]] = {"S": "suma", "M": "media", "R": "no_entrena"}
SEMILLAS_DEFECTO: Final[tuple[int, ...]] = (20261016, 20261017, 20261018)

# Codigos de flujo del generador aleatorio (independientes del brazo, para que S y M compartan
# exactamente la misma secuencia de pares, permutacion y negativos).
FLUJO_PARES: Final[int] = 1
FLUJO_NEGATIVOS: Final[int] = 2
FLUJO_INIT: Final[int] = 3
FLUJO_ALEATORIO: Final[int] = 4

SEMILLA_INTERNA_BASE: Final[int] = 20261101
N_INTERNOS: Final[int] = 3

# Version de la firma de reanudacion. La firma incluye el sha256 del script: cualquier cambio
# en este fichero invalida los parciales previos. v2: umbral anidado elegido solo sobre las
# filas inner_val cubiertas y publicacion de las metricas de lectura.
VERSION_SCRIPT: Final[int] = 2

SEMILLA_ANCLA: Final[int] = 20260926
K_ANCLA: Final[int] = 10
# El binario oficial fastText aborta de forma intermitente con EncounteredNaNError o escribe un
# .vec no finito incluso con el comando y la semilla fijos (medido en revision: 2/16 invocaciones
# identicas). Se reintenta con el MISMO comando y semilla; no se cambia el criterio de pureza.
INTENTOS_ANCLA_OFICIAL: Final[int] = 5
TOL_ANCLA_ALGEBRA: Final[float] = 1e-9
TOL_ANCLA_GRADIENTE: Final[float] = 1e-5
TOL_ANCLA_PUREZA: Final[float] = 0.05
PUREZA_K: Final[int] = 5

GATE_DELTA: Final[float] = 0.005
GATE_PLIEGUES: Final[int] = 4
GATE_ENSEMBLE_GAIN: Final[float] = 0.0015
GATE_BRECHA_MAX: Final[float] = 0.05

HUMO_MAX_TOKENS: Final[int] = 200_000
HITO_DECISION_SEGUNDOS: Final[float] = 2400.0

RE_TOKEN: Final[re.Pattern[str]] = re.compile(r"[#@]?\w+(?:['\u2019]\w+)*", re.UNICODE)


@dataclass
class ConfigSGNS:
    """Hiperparametros fijos del entrenador SGNS.

    Attributes:
        dim: dimension de los embeddings.
        ventana: ventana maxima (dinamica uniforme 1..ventana).
        min_count: frecuencia minima del vocabulario.
        negativos: numero K de negativos por par.
        submuestreo: umbral t del subsampling clasico.
        epocas: numero de epocas.
        lote: tamano de lote.
        lr0_por_ejemplo: learning rate inicial por ejemplo (se multiplica por el lote).
        lr_minimo: learning rate minimo del decaimiento lineal.
    """

    dim: int
    ventana: int
    min_count: int
    negativos: int
    submuestreo: float
    epocas: int
    lote: int
    lr0_por_ejemplo: float
    lr_minimo: float


CONFIG_SGNS: Final[ConfigSGNS] = ConfigSGNS(
    dim=200,
    ventana=5,
    min_count=3,
    negativos=10,
    submuestreo=1e-4,
    epocas=5,
    lote=256,
    lr0_por_ejemplo=0.025,
    lr_minimo=1e-3,
)
CONFIG_ANCLA: Final[ConfigSGNS] = ConfigSGNS(
    dim=32,
    ventana=5,
    min_count=1,
    negativos=10,
    submuestreo=1e-4,
    epocas=5,
    lote=256,
    lr0_por_ejemplo=0.025,
    lr_minimo=1e-3,
)

CONFIG: Final[dict[str, Any]] = {
    "corpus": PROJECT_ROOT / "data" / "external" / "corpora" / "wikipedia_es"
    / "emb_normal" / "tokens_10M.txt",
    "bin_fasttext": PROJECT_ROOT / "tools" / "fastText" / "build" / "fasttext",
    "train_path": PROJECT_ROOT / "data" / "raw" / "toxic_train.csv",
    "grupos_path": PROJECT_ROOT / "data" / "processed" / "grupos_near_duplicados.csv",
    "oof_v03_path": PROJECT_ROOT / "features" / "oof_v03_final.npz",
    "experiments_csv": PROJECT_ROOT / "experiments.csv",
    "parciales_dir": PROJECT_ROOT / "reports" / "p84_04_parciales",
    "embeddings_dir": PROJECT_ROOT / "modelos" / "p84",
    "oof_npz": PROJECT_ROOT / "features" / "p84_04_oof_emb.npz",
    "ancla_json": PROJECT_ROOT / "reports" / "p84_04_ancla.json",
    "informe_json": PROJECT_ROOT / "reports" / "p84_04_sgns.json",
    "informe_md": PROJECT_ROOT / "reports" / "p84_04_sgns.md",
    "tmp_dir": PROJECT_ROOT / "experimentos" / "p84_04_tmp",
    "fecha": "2026-10-02",
    "script": "experiments/p84_04_sgns_corregido_anclado.py",
    "competencia": "KAGGLE_TOXIC_2026_PUJ",
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

INDICE_ETIQUETA: Final[dict[str, int]] = {etiqueta: i for i, etiqueta in enumerate(ETIQUETAS)}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso con el prefijo de la fase.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P84-04] {mensaje}", flush=True)


def sembrar_global(semilla: int) -> torch.device:
    """Fija las semillas de python/numpy/torch y devuelve el device de computo.

    Args:
        semilla: semilla global de la corrida.

    Returns:
        Device CUDA si esta disponible; CPU en caso contrario.
    """
    import random as _random

    os.environ["PYTHONHASHSEED"] = str(semilla)
    _random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.set_num_threads(32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(semilla)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (RuntimeError, AttributeError) as exc:
            log(f"aviso: no se pudo activar el modo determinista: {exc}")
    return device


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
        Diccionario con python, numpy, pandas, sklearn, scipy y device.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device = f"cuda ({torch.cuda.get_device_name(0)})"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "device": device,
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
        verificar_sha256(CONFIG["corpus"], HASH_CORPUS, "tokens_10M.txt"),
        verificar_sha256(CONFIG["bin_fasttext"], HASH_FASTTEXT, "fasttext (binario oficial)"),
        verificar_sha256(CONFIG["oof_v03_path"], HASH_OOF_V03, "oof_v03_final.npz"),
        verificar_sha256(CONFIG["train_path"], HASH_TRAIN, "toxic_train.csv"),
        verificar_sha256(CONFIG["grupos_path"], HASH_GRUPOS, "grupos_near_duplicados.csv"),
    ]

    train = read_train()
    if len(train) != FILAS_TRAIN:
        raise SystemExit(f"STOP: toxic_train.csv tiene {len(train)} filas, se esperaban {FILAS_TRAIN}")
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
    if not np.allclose(umbrales_v03, UMBRALES_CONGELADOS, atol=0.0, rtol=0.0):
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


class RepresentacionTfidf:
    """TF-IDF union de V03: word (1,2) + char_wb (3,5), min_df=2, sublinear_tf.

    Replica exacta del vectorizador de V03/P71-10, en el mismo orden de concatenacion.
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

        Se usa fit_transform (no fit + transform) porque los dos caminos no son
        bit-identicos y el control de identidad exige 1e-12.

        Args:
            textos: textos de ajuste (nunca validacion ni test).

        Returns:
            Matriz CSR de entrenamiento.
        """
        matriz_palabra = self.palabra.fit_transform(textos)
        matriz_caracter = self.caracter.fit_transform(textos)
        return sparse.hstack([matriz_palabra, matriz_caracter], format="csr")

    def transformar(self, textos: np.ndarray) -> sparse.csr_matrix:
        """Transforma textos con el vocabulario ya ajustado.

        Args:
            textos: textos a transformar.

        Returns:
            Matriz CSR concatenada.
        """
        return sparse.hstack(
            [self.palabra.transform(textos), self.caracter.transform(textos)], format="csr"
        )


def reconstruir_v03(
    textos: np.ndarray, y: np.ndarray, pliegues: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruye el OOF de V03 con el pipeline canonico por pliegue.

    Args:
        textos: textos de desarrollo.
        y: etiquetas (1483, 3).
        pliegues: pliegues canonicos.

    Returns:
        Tupla (proba_validacion, proba_train) reconstruidas por pliegue.
    """
    proba = np.zeros((len(textos), len(ETIQUETAS)), dtype=np.float64)
    proba_train = np.zeros_like(proba)
    for indice_train, indice_val in pliegues:
        a = np.asarray(indice_train)
        b = np.asarray(indice_val)
        representacion = RepresentacionTfidf()
        x_a = representacion.ajustar_transformar(textos[a])
        x_b = representacion.transformar(textos[b])
        for j in range(len(ETIQUETAS)):
            modelo = LogisticRegression(
                C=2.0,
                class_weight="balanced",
                max_iter=1000,
                solver="liblinear",
                random_state=42,
            )
            modelo.fit(x_a, y[a, j])
            proba_train[a, j] = modelo.predict_proba(x_a)[:, 1]
            proba[b, j] = modelo.predict_proba(x_b)[:, 1]
    return proba, proba_train


def control_identidad_v03(datos: DatosGate0) -> dict[str, Any]:
    """Verifica que el harness reproduce el OOF canonico de V03 con tolerancia 1e-12.

    Args:
        datos: datos de Gate 0.

    Returns:
        Evidencia de identidad y brecha train-validacion de V03.

    Raises:
        SystemExit: si la tolerancia de 1e-12 no se cumple o el F1 no coincide.
    """
    proba_rec, proba_train = reconstruir_v03(datos.textos, datos.y, datos.pliegues)
    max_delta = float(np.max(np.abs(proba_rec - datos.p_v03)))
    pred_rec = (proba_rec >= datos.umbrales_v03[None, :]).astype(np.int64)
    f1_rec = float(metrica_macro(datos.y, pred_rec))
    f1_at_05 = float(metrica_macro(datos.y, (datos.p_v03 >= 0.5).astype(np.int64)))
    f1_pliegues_rec: list[float] = []
    brecha_por_pliegue: list[float] = []
    for indice_train, indice_val in datos.pliegues:
        a = np.asarray(indice_train)
        b = np.asarray(indice_val)
        f1_pliegues_rec.append(float(metrica_macro(datos.y[b], pred_rec[b])))
        f1_train_05 = float(metrica_macro(datos.y[a], (proba_train[a] >= 0.5).astype(np.int64)))
        f1_val_05 = float(metrica_macro(datos.y[b], (proba_rec[b] >= 0.5).astype(np.int64)))
        brecha_por_pliegue.append(f1_train_05 - f1_val_05)

    fallos: list[str] = []
    if not np.array_equal(pred_rec, datos.pred_v03):
        fallos.append(
            f"las predicciones con umbrales congelados difieren del npz en "
            f"{int(np.sum(pred_rec != datos.pred_v03))} celdas"
        )
    if abs(f1_rec - V03_CCV) > TOL_IDENTIDAD:
        fallos.append(f"|F1 - {V03_CCV}| = {abs(f1_rec - V03_CCV):g} > {TOL_IDENTIDAD:g}")
    if abs(f1_at_05 - V03_F1_AT_05) > TOL_IDENTIDAD:
        fallos.append(f"|F1@0.5 - {V03_F1_AT_05}| = {abs(f1_at_05 - V03_F1_AT_05):g}")
    if max_delta > TOL_IDENTIDAD:
        fallos.append(
            f"max|delta proba| = {max_delta:g} > {TOL_IDENTIDAD:g} "
            "(instrumento: revisar version de sklearn/liblinear o la ruta fit_transform)"
        )
    if fallos:
        raise SystemExit("STOP: control de identidad V03 fallido: " + "; ".join(fallos))
    return {
        "max_delta_proba_vs_npz": max_delta,
        "tolerancia_proba": TOL_IDENTIDAD,
        "proba_reproducida": True,
        "f1_congelado_reconstruido": f1_rec,
        "f1_congelado_esperado": V03_CCV,
        "delta_f1_congelado": abs(f1_rec - V03_CCV),
        "f1_at_05_recalculado": f1_at_05,
        "f1_at_05_esperado": V03_F1_AT_05,
        "delta_f1_at_05": abs(f1_at_05 - V03_F1_AT_05),
        "pred_reconstruida_igual_al_npz": True,
        "f1_por_pliegue_reconstruido": [round(valor, 8) for valor in f1_pliegues_rec],
        "brecha_train_validacion_v03_por_pliegue": [round(v, 6) for v in brecha_por_pliegue],
        "brecha_train_validacion_v03_media": float(np.mean(brecha_por_pliegue)),
    }


# ===== SECCION 2: nucleo SGNS (objetivo, modelo, datos y entrenamiento) =====


def objetivo_sgns(
    w_entrada: torch.Tensor,
    w_salida: torch.Tensor,
    centro: torch.Tensor,
    ctx: torch.Tensor,
    neg: torch.Tensor,
    agregacion: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Calcula el objetivo SGNS con la agregacion pedida del termino negativo.

    Objetivos (B = lote, K = negativos por par):
        loss_suma  = -mean_b pos_b - mean_b sum_k neg_bk   (SGNS publicado)
        loss_media = -mean_b pos_b - mean_b mean_k neg_bk  (objetivo antiguo)
    con pos_b = log sigmoide del producto interno positivo y neg_bk el log sigmoide
    negativo de cada par (centro, negativo k). Ambos promedian el termino positivo
    sobre el lote; solo cambia la agregacion sobre K.

    Args:
        w_entrada: matriz de entrada (V, D).
        w_salida: matriz de salida (V, D).
        centro: indices centrales (B,).
        ctx: indices de contexto positivos (B,).
        neg: indices negativos (B, K).
        agregacion: "suma" o "media".

    Returns:
        Tupla (perdida, extras) con extras pos, neg_suma y neg_media (escalares).

    Raises:
        ValueError: si la agregacion no es "suma" ni "media".
    """
    v = w_entrada[centro]
    u_pos = w_salida[ctx]
    u_neg = w_salida[neg]
    pos = F.logsigmoid((v * u_pos).sum(dim=1))
    neg_k = F.logsigmoid(-(v.unsqueeze(1) * u_neg).sum(dim=2))
    neg_suma = neg_k.sum(dim=1).mean()
    neg_media = neg_k.mean()
    if agregacion == "suma":
        perdida = -(pos.mean() + neg_suma)
    elif agregacion == "media":
        perdida = -(pos.mean() + neg_media)
    else:
        raise ValueError(f"agregacion desconocida: {agregacion!r}")
    extras = {"pos": pos.mean(), "neg_suma": neg_suma, "neg_media": neg_media}
    return perdida, extras


class SGNS(torch.nn.Module):
    """Skip-gram con negative sampling y agregacion configurable del termino negativo."""

    def __init__(self, w_entrada: np.ndarray, w_salida: np.ndarray) -> None:
        """Inicializa las dos matrices de embedding con valores dados.

        Args:
            w_entrada: matriz de entrada (V, D).
            w_salida: matriz de salida (V, D).
        """
        super().__init__()
        self.w_entrada = torch.nn.Parameter(torch.as_tensor(w_entrada, dtype=torch.float32))
        self.w_salida = torch.nn.Parameter(torch.as_tensor(w_salida, dtype=torch.float32))

    def forward(
        self,
        centro: torch.Tensor,
        ctx: torch.Tensor,
        neg: torch.Tensor,
        agregacion: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calcula la perdida del lote.

        Args:
            centro: indices centrales.
            ctx: indices de contexto positivos.
            neg: indices negativos (B, K).
            agregacion: "suma" o "media".

        Returns:
            Tupla (perdida, extras) de :func:`objetivo_sgns`.
        """
        return objetivo_sgns(self.w_entrada, self.w_salida, centro, ctx, neg, agregacion)


def leer_tokens(ruta: Path, max_tokens: int | None = None) -> list[list[str]]:
    """Lee el corpus y devuelve la lista de documentos tokenizados.

    Args:
        ruta: fichero con un documento normalizado por linea.
        max_tokens: tope opcional de tokens (modo humo); None lee todo.

    Returns:
        Lista de listas de tokens (solo documentos con >= 3 tokens).
    """
    docs: list[list[str]] = []
    total = 0
    with ruta.open(encoding="utf-8", errors="replace") as fh:
        for linea in fh:
            toks = RE_TOKEN.findall(linea)
            if len(toks) >= 3:
                docs.append(toks)
                total += len(toks)
            if max_tokens is not None and total >= max_tokens:
                break
    return docs


def construir_vocabulario(
    docs: list[list[str]], min_count: int, submuestreo: float
) -> tuple[dict[str, int], np.ndarray, Counter]:
    """Construye el vocabulario con min_count y la tabla de subsampling.

    Args:
        docs: documentos tokenizados.
        min_count: frecuencia minima.
        submuestreo: umbral t del muestreo clasico.

    Returns:
        Indice de palabras, probabilidad de conservar y contador completo.
    """
    cnt = Counter(t for d in docs for t in d)
    vocab = {w: i for i, (w, c) in enumerate(cnt.most_common()) if c >= min_count}
    total = sum(cnt.values())
    frec = np.zeros(len(vocab), dtype=np.float64)
    for w, i in vocab.items():
        frec[i] = cnt[w] / total
    p = (np.sqrt(frec / submuestreo) + 1.0) * (submuestreo / frec)
    p = np.clip(p, 0.0, 1.0)
    return vocab, p, cnt


def construir_ids_por_doc(docs: list[list[str]], vocab: dict[str, int]) -> list[np.ndarray]:
    """Convierte cada documento a indices del vocabulario (OOV descartados).

    Args:
        docs: documentos tokenizados.
        vocab: indice de palabras.

    Returns:
        Lista de arreglos int32 alineados con docs.
    """
    return [
        np.fromiter((vocab[t] for t in doc if t in vocab), dtype=np.int32, count=-1)
        for doc in docs
    ]


def tabla_negativos(vocab: dict[str, int], cnt: Counter) -> tuple[np.ndarray, np.ndarray]:
    """Construye la tabla de muestreo de negativos (ruido unigram elevado a 0.75).

    Args:
        vocab: indice de palabras.
        cnt: contador de frecuencias.

    Returns:
        Tupla (pesos normalizados, cdf acumulada con ultimo valor 1.0).
    """
    pesos = np.zeros(len(vocab), dtype=np.float64)
    for w, i in vocab.items():
        pesos[i] = float(cnt[w]) ** 0.75
    pesos = pesos / pesos.sum()
    cdf = np.cumsum(pesos)
    cdf[-1] = 1.0
    return pesos, cdf


def pares_de_epoca(
    ids_por_doc: list[np.ndarray],
    p: np.ndarray,
    rng: np.random.Generator,
    ventana: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera los pares centro-contexto de una epoca con ventana dinamica.

    Version vectorizada de la generacion de P71-04: por documento se aplica el
    subsampling, se sortea una ventana uniforme 1..ventana por posicion y se
    emiten los pares (k, k-d) y (k, k+d) con d <= ventana_k.

    Args:
        ids_por_doc: indices de cada documento.
        p: probabilidad de conservar cada palabra.
        rng: generador aleatorio del flujo de datos.
        ventana: ventana maxima.

    Returns:
        Tupla (centros, contextos) int32.
    """
    centros_l: list[np.ndarray] = []
    contextos_l: list[np.ndarray] = []
    for ids in ids_por_doc:
        if ids.size < 2:
            continue
        conservar = rng.random(ids.size) < p[ids]
        ids = ids[conservar]
        if ids.size < 2:
            continue
        ventanas = rng.integers(1, ventana + 1, size=ids.size)
        posiciones = np.arange(ids.size)
        for d in range(1, ventana + 1):
            izquierda = (ventanas >= d) & (posiciones >= d)
            derecha = (ventanas >= d) & (posiciones + d < ids.size)
            if izquierda.any():
                centros_l.append(ids[izquierda])
                contextos_l.append(ids[posiciones[izquierda] - d])
            if derecha.any():
                centros_l.append(ids[derecha])
                contextos_l.append(ids[posiciones[derecha] + d])
    if not centros_l:
        vacio = np.empty(0, dtype=np.int32)
        return vacio, vacio.copy()
    return np.concatenate(centros_l), np.concatenate(contextos_l)


@dataclass
class PlanPasos:
    """Plan de lotes y evidencia del flujo de datos.

    Attributes:
        lotes_por_epoca: numero de lotes por epoca.
        total_lotes: suma de lotes de todas las epocas.
        hash_pares_epoca1: sha256 de los pares de la epoca 1.
        hash_negativos_lote1: sha256 de los negativos del primer lote de la epoca 1.
    """

    lotes_por_epoca: list[int]
    total_lotes: int
    hash_pares_epoca1: str
    hash_negativos_lote1: str


def plan_de_pasos(
    ids_por_doc: list[np.ndarray],
    p: np.ndarray,
    cdf: np.ndarray,
    semilla: int,
    cfg: ConfigSGNS,
) -> PlanPasos:
    """Regenera los pares por epoca para fijar el decaimiento lineal del lr.

    El flujo depende solo de la semilla (nunca del brazo), de modo que S y M ven
    exactamente los mismos pares, permutacion y negativos. Se publican los hashes
    de la epoca 1 como evidencia verificable del pareado.

    Args:
        ids_por_doc: indices de cada documento.
        p: probabilidad de conservar cada palabra.
        cdf: cdf de la tabla de negativos.
        semilla: semilla del flujo de datos.
        cfg: configuracion del entrenador.

    Returns:
        Plan de lotes con los hashes del flujo.
    """
    lotes: list[int] = []
    hash_pares = ""
    n_pares_ep1 = 0
    for ep in range(1, cfg.epocas + 1):
        rng = np.random.default_rng([semilla, FLUJO_PARES, ep])
        centros, contextos = pares_de_epoca(ids_por_doc, p, rng, cfg.ventana)
        n_lotes = max(1, (len(centros) + cfg.lote - 1) // cfg.lote)
        lotes.append(n_lotes)
        if ep == 1:
            n_pares_ep1 = len(centros)
            resumen = hashlib.sha256()
            resumen.update(centros.tobytes())
            resumen.update(contextos.tobytes())
            hash_pares = resumen.hexdigest()
    rng_neg = np.random.default_rng([semilla, FLUJO_NEGATIVOS, 1])
    n_primer_lote = min(cfg.lote, max(n_pares_ep1, 1))
    u = rng_neg.random((n_primer_lote, cfg.negativos))
    negativos = np.searchsorted(cdf, u, side="right").astype(np.int64)
    hash_negativos = hashlib.sha256(negativos.tobytes()).hexdigest()
    return PlanPasos(
        lotes_por_epoca=lotes,
        total_lotes=int(sum(lotes)),
        hash_pares_epoca1=hash_pares,
        hash_negativos_lote1=hash_negativos,
    )


def inicializar_matrices(
    semilla: int, n_vocab: int, dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Inicializa las matrices de entrada y salida de forma pareada.

    La inicializacion depende solo de la semilla (no del brazo): S y M parten de
    los mismos pesos. Entrada uniforme +-0.5/dim; salida uniforme [-0.05, 0.05].

    Args:
        semilla: semilla del flujo de inicializacion.
        n_vocab: tamano del vocabulario.
        dim: dimension.

    Returns:
        Tupla (w_entrada, w_salida) float32.
    """
    rng = np.random.default_rng([semilla, FLUJO_INIT])
    limite = 0.5 / dim
    w_entrada = rng.uniform(-limite, limite, size=(n_vocab, dim)).astype(np.float32)
    w_salida = rng.uniform(-0.05, 0.05, size=(n_vocab, dim)).astype(np.float32)
    return w_entrada, w_salida


def generar_aleatoria(semilla: int, n_vocab: int, dim: int) -> np.ndarray:
    """Genera la matriz aleatoria congelada del control R.

    Misma forma y escala que la inicializacion de entrada de S y M
    (uniforme +-0.5/dim), sembrada por semilla de brazo. No entrena.

    Args:
        semilla: semilla del control aleatorio.
        n_vocab: tamano del vocabulario.
        dim: dimension.

    Returns:
        Matriz (V, D) float32.
    """
    rng = np.random.default_rng([semilla, FLUJO_ALEATORIO])
    limite = 0.5 / dim
    return rng.uniform(-limite, limite, size=(n_vocab, dim)).astype(np.float32)


def norma_gradiente_filas(
    modelo: SGNS, centro: torch.Tensor, ctx: torch.Tensor, neg: torch.Tensor
) -> torch.Tensor:
    """Norma L2 del gradiente de las filas tocadas por el lote.

    Se mide solo sobre las filas efectivamente actualizadas (gather), no sobre la
    matriz densa completa, para no penalizar el tiempo por lote. Devuelve un
    tensor para poder acumular sin sincronizar la GPU en cada lote.

    Args:
        modelo: modelo SGNS tras backward.
        centro: indices centrales del lote.
        ctx: indices de contexto.
        neg: indices negativos.

    Returns:
        Tensor escalar con la norma del gradiente de las filas tocadas.

    Raises:
        RuntimeError: si el gradiente aun no existe.
    """
    grad_entrada = modelo.w_entrada.grad
    grad_salida = modelo.w_salida.grad
    if grad_entrada is None or grad_salida is None:
        raise RuntimeError("no hay gradiente: llama a backward antes de medir la norma")
    with torch.no_grad():
        filas_entrada = torch.unique(centro)
        filas_salida = torch.unique(torch.cat((ctx.reshape(-1), neg.reshape(-1))))
        g_entrada = grad_entrada[filas_entrada]
        g_salida = grad_salida[filas_salida]
        norma = torch.sqrt(g_entrada.pow(2).sum() + g_salida.pow(2).sum())
    return norma


@dataclass
class ResultadoEntrenamiento:
    """Resultado de entrenar un brazo SGNS.

    Attributes:
        nombre: nombre del brazo.
        agregacion: "suma" o "media".
        w_entrada: matriz de entrada entrenada (V, D) float32.
        loss_por_epoca: perdida media por epoca (no comparable entre brazos).
        grad_media_por_epoca: norma media del gradiente por lote y epoca.
        grad_max_por_epoca: norma maxima del gradiente por lote y epoca.
        lr_inicial: learning rate inicial (0.025 x lote).
        lr_final: learning rate del ultimo paso.
        segundos: tiempo de entrenamiento.
        pasos: numero de pasos dados.
        hash_pares_epoca1: hash de los pares de la epoca 1.
        hash_negativos_lote1: hash de los negativos del primer lote.
    """

    nombre: str
    agregacion: str
    w_entrada: np.ndarray
    loss_por_epoca: list[float]
    grad_media_por_epoca: list[float]
    grad_max_por_epoca: list[float]
    lr_inicial: float
    lr_final: float
    segundos: float
    pasos: int
    hash_pares_epoca1: str
    hash_negativos_lote1: str


def entrenar_modelos(
    brazos: Sequence[tuple[str, str]],
    ids_por_doc: list[np.ndarray],
    p: np.ndarray,
    cdf: np.ndarray,
    semilla: int,
    cfg: ConfigSGNS,
    device: torch.device,
) -> dict[str, ResultadoEntrenamiento]:
    """Entrena uno o dos brazos SGNS con el MISMO flujo de datos por lote.

    Cuando se piden S y M a la vez, cada lote produce las mismas tuplas
    (centro, contexto, negativos) para los dos modelos y solo cambia la
    agregacion del termino negativo. El flujo aleatorio depende unicamente de la
    semilla, de modo que un entrenamiento por separado tambien coincide.

    Args:
        brazos: pares (nombre, agregacion) a entrenar; agregacion "suma" o "media".
        ids_por_doc: indices de cada documento.
        p: probabilidad de conservar cada palabra.
        cdf: cdf de la tabla de negativos.
        semilla: semilla del flujo de datos.
        cfg: configuracion del entrenador.
        device: device de computo.

    Returns:
        Diccionario nombre -> resultado.

    Raises:
        ValueError: si una agregacion no es "suma" ni "media".
    """
    for nombre, agregacion in brazos:
        if agregacion not in ("suma", "media"):
            raise ValueError(f"agregacion no entrenable: {agregacion!r} ({nombre})")

    plan = plan_de_pasos(ids_por_doc, p, cdf, semilla, cfg)
    modelos: dict[str, SGNS] = {}
    optimizadores: dict[str, torch.optim.SGD] = {}
    agregaciones = {nombre: agregacion for nombre, agregacion in brazos}
    nombres = [nombre for nombre, _ in brazos]
    for nombre, _ in brazos:
        w_entrada, w_salida = inicializar_matrices(semilla, len(p), cfg.dim)
        modelo = SGNS(w_entrada, w_salida).to(device)
        modelos[nombre] = modelo
        optimizadores[nombre] = torch.optim.SGD(modelo.parameters(), lr=cfg.lr0_por_ejemplo * cfg.lote)

    perdidas: dict[str, list[float]] = {nombre: [] for nombre in nombres}
    grad_media: dict[str, list[float]] = {nombre: [] for nombre in nombres}
    grad_max: dict[str, list[float]] = {nombre: [] for nombre in nombres}
    lr_final = cfg.lr0_por_ejemplo * cfg.lote
    paso = 0
    t0 = time.perf_counter()
    for ep, n_lotes in enumerate(plan.lotes_por_epoca, start=1):
        te = time.perf_counter()
        rng = np.random.default_rng([semilla, FLUJO_PARES, ep])
        centros, contextos = pares_de_epoca(ids_por_doc, p, rng, cfg.ventana)
        orden = rng.permutation(len(centros))
        centros = centros[orden]
        contextos = contextos[orden]
        del orden
        rng_neg = np.random.default_rng([semilla, FLUJO_NEGATIVOS, ep])
        acumulado: dict[str, dict[str, torch.Tensor]] = {
            nombre: {
                "loss": torch.zeros((), device=device),
                "grad_suma": torch.zeros((), device=device),
                "grad_max": torch.zeros((), device=device),
            }
            for nombre in nombres
        }
        for inicio in range(0, len(centros), cfg.lote):
            fin = min(inicio + cfg.lote, len(centros))
            c = torch.from_numpy(centros[inicio:fin].astype(np.int64)).to(device)
            x = torch.from_numpy(contextos[inicio:fin].astype(np.int64)).to(device)
            u = rng_neg.random((fin - inicio, cfg.negativos))
            neg_np = np.searchsorted(cdf, u, side="right").astype(np.int64)
            neg = torch.from_numpy(neg_np).to(device)
            lr = max(
                cfg.lr_minimo,
                cfg.lr0_por_ejemplo * cfg.lote * (1.0 - paso / max(plan.total_lotes, 1)),
            )
            lr_final = lr
            for nombre in nombres:
                modelo = modelos[nombre]
                optimizador = optimizadores[nombre]
                for grupo in optimizador.param_groups:
                    grupo["lr"] = lr
                optimizador.zero_grad(set_to_none=False)
                perdida, _ = modelo(c, x, neg, agregaciones[nombre])
                perdida.backward()
                norma = norma_gradiente_filas(modelo, c, x, neg)
                optimizador.step()
                with torch.no_grad():
                    acumulado[nombre]["loss"] += perdida.detach()
                    acumulado[nombre]["grad_suma"] += norma
                    acumulado[nombre]["grad_max"] = torch.maximum(
                        acumulado[nombre]["grad_max"], norma
                    )
            paso += 1
        for nombre in nombres:
            perdidas[nombre].append(
                round(float(acumulado[nombre]["loss"].item()) / max(n_lotes, 1), 6)
            )
            grad_media[nombre].append(
                round(float(acumulado[nombre]["grad_suma"].item()) / max(n_lotes, 1), 8)
            )
            grad_max[nombre].append(round(float(acumulado[nombre]["grad_max"].item()), 8))
        segundos_epoca = time.perf_counter() - te
        resumen = " | ".join(
            f"{nombre} loss {perdidas[nombre][-1]:.6f} grad_med {grad_media[nombre][-1]:.4g}"
            for nombre in nombres
        )
        log(
            f"   epoca {ep}/{cfg.epocas} ({semilla}) | {resumen} | lr {lr:.4f} | "
            f"{segundos_epoca:.0f}s"
        )
        if segundos_epoca > HITO_DECISION_SEGUNDOS:
            log(
                f"   HITO DE DECISION: la epoca tardo {segundos_epoca:.0f}s "
                f"(> {HITO_DECISION_SEGUNDOS:.0f}s); revisar presupuesto"
            )
    segundos = time.perf_counter() - t0
    resultados: dict[str, ResultadoEntrenamiento] = {}
    for nombre in nombres:
        w_entrada = modelos[nombre].w_entrada.detach().cpu().numpy().astype(np.float32)
        resultados[nombre] = ResultadoEntrenamiento(
            nombre=nombre,
            agregacion=agregaciones[nombre],
            w_entrada=w_entrada,
            loss_por_epoca=perdidas[nombre],
            grad_media_por_epoca=grad_media[nombre],
            grad_max_por_epoca=grad_max[nombre],
            lr_inicial=cfg.lr0_por_ejemplo * cfg.lote,
            lr_final=lr_final,
            segundos=round(segundos, 1),
            pasos=paso,
            hash_pares_epoca1=plan.hash_pares_epoca1,
            hash_negativos_lote1=plan.hash_negativos_lote1,
        )
    return resultados


# ===== SECCION 3: ancla de implementacion (algebra, gradiente y corpus plantado) =====


def chequeo_algebraico() -> dict[str, Any]:
    """Comprueba las identidades algebraicas entre el objetivo SUMA y el de media.

    Con los mismos tensores: N_suma = K * N_media y
    loss_suma = K * loss_media + (K - 1) * P, con P = media del log-sigma positivo.
    Tolerancia relativa 1e-9 y K = 10 (fijados en la spec).

    Returns:
        Evidencia con los dos errores relativos y el veredicto.
    """
    rng = np.random.default_rng([SEMILLA_ANCLA, 11])
    n_vocab, dim, lote = 37, 16, 12
    k = K_ANCLA
    w_entrada = torch.tensor(rng.normal(0.0, 0.15, (n_vocab, dim)), dtype=torch.float64)
    w_salida = torch.tensor(rng.normal(0.0, 0.15, (n_vocab, dim)), dtype=torch.float64)
    centro = torch.tensor(rng.integers(0, n_vocab, lote), dtype=torch.long)
    ctx = torch.tensor(rng.integers(0, n_vocab, lote), dtype=torch.long)
    neg = torch.tensor(rng.integers(0, n_vocab, (lote, k)), dtype=torch.long)

    _, extras = objetivo_sgns(w_entrada, w_salida, centro, ctx, neg, "suma")
    p_pos = float(extras["pos"])
    n_suma = float(extras["neg_suma"])
    n_media = float(extras["neg_media"])
    loss_suma = -(p_pos + n_suma)
    loss_media = -(p_pos + n_media)
    lado_derecho = k * loss_media + (k - 1) * p_pos
    error_n = abs(n_suma - k * n_media) / max(abs(n_suma), abs(k * n_media), 1e-300)
    error_loss = abs(loss_suma - lado_derecho) / max(abs(loss_suma), abs(lado_derecho), 1e-300)
    return {
        "K": k,
        "N_suma": n_suma,
        "K_por_N_media": k * n_media,
        "error_relativo_N": error_n,
        "P_media_log_sigma_positivo": p_pos,
        "loss_suma": loss_suma,
        "K_loss_media_mas_(K-1)_P": lado_derecho,
        "error_relativo_loss": error_loss,
        "tolerancia_relativa": TOL_ANCLA_ALGEBRA,
        "pasa": bool(error_n <= TOL_ANCLA_ALGEBRA and error_loss <= TOL_ANCLA_ALGEBRA),
    }


def chequeo_gradiente_finito() -> dict[str, Any]:
    """Verifica por diferencias finitas centradas el gradiente del objetivo SUMA.

    Se perturban entradas concretas de las matrices de entrada y salida sobre un
    lote sintetico diminuto en float64 (eps = 1e-6) y se compara el cociente
    incremental con el gradiente analitico. Tolerancia: error relativo <= 1e-5.

    Returns:
        Evidencia con el error relativo maximo y el veredicto.
    """
    rng = np.random.default_rng([SEMILLA_ANCLA, 12])
    n_vocab, dim, lote, k = 23, 8, 6, 4
    w_entrada = torch.tensor(
        rng.normal(0.0, 0.2, (n_vocab, dim)), dtype=torch.float64, requires_grad=True
    )
    w_salida = torch.tensor(
        rng.normal(0.0, 0.2, (n_vocab, dim)), dtype=torch.float64, requires_grad=True
    )
    centro = torch.tensor(rng.integers(0, n_vocab, lote), dtype=torch.long)
    ctx = torch.tensor(rng.integers(0, n_vocab, lote), dtype=torch.long)
    neg = torch.tensor(rng.integers(0, n_vocab, (lote, k)), dtype=torch.long)

    perdida, _ = objetivo_sgns(w_entrada, w_salida, centro, ctx, neg, "suma")
    perdida.backward()
    eps = 1e-6
    errores: list[float] = []
    for indice in range(8):
        i = int(rng.integers(0, n_vocab))
        j = int(rng.integers(0, dim))
        with torch.no_grad():
            original = float(w_entrada[i, j])
            w_entrada[i, j] = original + eps
            mas = float(objetivo_sgns(w_entrada, w_salida, centro, ctx, neg, "suma")[0])
            w_entrada[i, j] = original - eps
            menos = float(objetivo_sgns(w_entrada, w_salida, centro, ctx, neg, "suma")[0])
            w_entrada[i, j] = original
        finito = (mas - menos) / (2 * eps)
        analitico = float(w_entrada.grad[i, j])
        errores.append(abs(finito - analitico) / max(abs(finito), abs(analitico), 1e-12))
    for indice in range(8):
        i = int(rng.integers(0, n_vocab))
        j = int(rng.integers(0, dim))
        with torch.no_grad():
            original = float(w_salida[i, j])
            w_salida[i, j] = original + eps
            mas = float(objetivo_sgns(w_entrada, w_salida, centro, ctx, neg, "suma")[0])
            w_salida[i, j] = original - eps
            menos = float(objetivo_sgns(w_entrada, w_salida, centro, ctx, neg, "suma")[0])
            w_salida[i, j] = original
        finito = (mas - menos) / (2 * eps)
        analitico = float(w_salida.grad[i, j])
        errores.append(abs(finito - analitico) / max(abs(finito), abs(analitico), 1e-12))
    error_max = float(np.max(errores))
    return {
        "eps": eps,
        "n_entradas_probadas": len(errores),
        "error_relativo_max": error_max,
        "tolerancia": TOL_ANCLA_GRADIENTE,
        "pasa": bool(error_max <= TOL_ANCLA_GRADIENTE),
    }


def crear_corpus_plantado(ruta: Path) -> dict[str, Any]:
    """Crea el corpus plantado de dos familias disjuntas de palabras.

    Cada oracion mezcla solo palabras de una familia, de modo que la estructura
    de vecinos correcta es perfectamente conocida y la pureza de vecinos mide si
    la implementacion aprende esa estructura.

    Args:
        ruta: destino del corpus (una oracion por linea).

    Returns:
        Evidencia con ruta, familias, numero de oraciones y tokens.
    """
    import random as _random

    rng = _random.Random(SEMILLA_ANCLA)
    familia_a = [f"fA{i:02d}" for i in range(8)]
    familia_b = [f"fB{i:02d}" for i in range(8)]
    lineas: list[str] = []
    for _ in range(20_000):
        familia = familia_a if rng.random() < 0.5 else familia_b
        n = rng.randint(4, 8)
        lineas.append(" ".join(rng.choice(familia) for _ in range(n)))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(lineas) + "\n", encoding="ascii")
    familias = {palabra: "A" for palabra in familia_a}
    familias.update({palabra: "B" for palabra in familia_b})
    return {
        "ruta": str(ruta),
        "sha256_corpus": sha256_of(ruta),
        "oraciones": len(lineas),
        "tokens": sum(len(linea.split()) for linea in lineas),
        "vocabulario_objetivo": len(familias),
        "k_pureza": PUREZA_K,
        "familias": familias,
    }


def ejecutar_ancla_oficial(ruta_corpus: Path, base_salida: Path) -> dict[str, Any]:
    """Entrena el binario oficial fastText `skipgram` con la configuracion del ancla.

    Comando fijado por la spec: skipgram -minn 0 -maxn 0 -dim 32 -neg 10 -epoch 5
    -thread 1 -seed 20260926 -minCount 1. Se anade -verbose 0 (cosmetico, no
    altera el modelo) para no inundar la salida.

    El binario oficial aborta de forma intermitente con EncounteredNaNError o
    escribe un .vec no finito incluso con el comando y la semilla fijos (2/16
    invocaciones identicas medidas en revision). Por eso se reintenta hasta
    INTENTOS_ANCLA_OFICIAL veces con el MISMO comando y semilla, exigiendo que el
    .vec sea finito; no se modifica el criterio de pureza ni la configuracion.

    Args:
        ruta_corpus: corpus plantado.
        base_salida: ruta base del modelo (fastText anade .bin y .vec).

    Returns:
        Evidencia con ok, el comando, los intentos, el hash del .vec y la ruta.
        Si todos los intentos fallan devuelve ok=False sin lanzar; el llamador
        lo traduce a ANCLA NO LOGRADA.
    """
    comando = [
        str(CONFIG["bin_fasttext"]),
        "skipgram",
        "-input",
        str(ruta_corpus),
        "-output",
        str(base_salida),
        "-minn",
        "0",
        "-maxn",
        "0",
        "-dim",
        str(CONFIG_ANCLA.dim),
        "-neg",
        str(CONFIG_ANCLA.negativos),
        "-epoch",
        str(CONFIG_ANCLA.epocas),
        "-thread",
        "1",
        "-seed",
        str(SEMILLA_ANCLA),
        "-minCount",
        str(CONFIG_ANCLA.min_count),
        "-verbose",
        "0",
    ]
    ruta_vec = Path(str(base_salida) + ".vec")
    intentos: list[dict[str, Any]] = []
    for intento in range(1, INTENTOS_ANCLA_OFICIAL + 1):
        for sufijo in (".bin", ".vec"):
            Path(str(base_salida) + sufijo).unlink(missing_ok=True)
        resultado = subprocess.run(
            comando,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        if resultado.returncode != 0:
            cola = [linea for linea in (resultado.stderr or "").splitlines() if linea.strip()]
            intentos.append(
                {
                    "intento": intento,
                    "codigo": int(resultado.returncode),
                    "motivo": (cola[-1][:200] if cola else "fallo sin stderr"),
                }
            )
            continue
        if not ruta_vec.exists():
            intentos.append({"intento": intento, "codigo": 0, "motivo": f"no escribio {ruta_vec.name}"})
            continue
        try:
            _palabras, w_vec = leer_vec_fasttext(ruta_vec)
        except SystemExit as error:
            intentos.append({"intento": intento, "codigo": 0, "motivo": f".vec ilegible: {error}"})
            continue
        n_no_finitos = int((~np.isfinite(w_vec)).sum())
        if n_no_finitos:
            intentos.append(
                {
                    "intento": intento,
                    "codigo": 0,
                    "motivo": f".vec con {n_no_finitos} de {w_vec.size} celdas no finitas",
                }
            )
            continue
        intentos.append({"intento": intento, "codigo": 0, "motivo": "ok (vec finito)"})
        return {
            "ok": True,
            "intentos": intentos,
            "comando": comando,
            "ruta_vec": str(ruta_vec),
            "sha256_vec": sha256_of(ruta_vec),
        }
    return {
        "ok": False,
        "intentos": intentos,
        "comando": comando,
        "motivo": (
            f"los {INTENTOS_ANCLA_OFICIAL} intentos del binario oficial fallaron "
            "(EncounteredNaNError o .vec no finito)"
        ),
    }


def leer_vec_fasttext(ruta: Path) -> tuple[list[str], np.ndarray]:
    """Lee un fichero .vec de fastText.

    Args:
        ruta: fichero .vec.

    Returns:
        Tupla (palabras, matriz float64).
    """
    with ruta.open(encoding="utf-8", errors="replace") as fh:
        cabecera = fh.readline().split()
        n_palabras, dim = int(cabecera[0]), int(cabecera[1])
        palabras: list[str] = []
        filas: list[list[float]] = []
        for linea in fh:
            partes = linea.rstrip().split()
            if len(partes) != dim + 1:
                continue
            palabras.append(partes[0])
            filas.append([float(valor) for valor in partes[1:]])
    if len(palabras) != n_palabras:
        raise SystemExit(
            f"STOP: .vec con {len(palabras)} palabras, cabecera declara {n_palabras}"
        )
    return palabras, np.asarray(filas, dtype=np.float64)


def pureza_vecinos(
    w: np.ndarray, palabras: Sequence[str], familias: dict[str, str], k: int
) -> dict[str, Any]:
    """Calcula la pureza de vecinos por familia (coseno, top-k).

    Args:
        w: matriz de embeddings (V, D) alineada con palabras.
        palabras: nombres de las palabras.
        familias: palabra -> familia.
        k: numero de vecinos.

    Returns:
        Diccionario con pureza media, por palabra y tamano evaluado.
    """
    indices = [i for i, palabra in enumerate(palabras) if palabra in familias]
    if len(indices) != len(familias):
        raise SystemExit(
            f"STOP: el ancla esperaba {len(familias)} palabras y encontro {len(indices)}"
        )
    w = w[np.asarray(indices, dtype=np.int64)]
    nombres = [palabras[i] for i in indices]
    normas = np.linalg.norm(w, axis=1, keepdims=True)
    x = w / np.maximum(normas, 1e-12)
    similitud = x @ x.T
    np.fill_diagonal(similitud, -np.inf)
    vecinos = np.argsort(-similitud, axis=1)[:, :k]
    purities = [
        float(np.mean([familias[nombres[j]] == familias[nombres[i]] for j in fila]))
        for i, fila in enumerate(vecinos)
    ]
    return {
        "pureza_media": float(np.mean(purities)),
        "pureza_por_palabra": {
            nombres[i]: round(valor, 6) for i, valor in enumerate(purities)
        },
        "n_palabras": len(nombres),
        "k": k,
    }


def ejecutar_ancla() -> dict[str, Any]:
    """Ejecuta el ancla completa (algebra, gradiente y corpus plantado).

    Returns:
        Evidencia del ancla y veredicto ANCLA LOGRADA / ANCLA NO LOGRADA.
    """
    algebraica = chequeo_algebraico()
    log(
        f"   ancla algebraica: err_N {algebraica['error_relativo_N']:.3e} | "
        f"err_loss {algebraica['error_relativo_loss']:.3e} | pasa {algebraica['pasa']}"
    )
    gradiente = chequeo_gradiente_finito()
    log(
        f"   ancla gradiente: err_max {gradiente['error_relativo_max']:.3e} | "
        f"pasa {gradiente['pasa']}"
    )

    plantado = crear_corpus_plantado(CONFIG["tmp_dir"] / "ancla_plantado.txt")
    ruta_corpus = Path(plantado["ruta"])
    oficial = ejecutar_ancla_oficial(ruta_corpus, CONFIG["tmp_dir"] / "ancla_oficial")
    if not oficial["ok"]:
        log(
            f"   ancla ejecutable: binario oficial no logrado tras "
            f"{INTENTOS_ANCLA_OFICIAL} intentos; veredicto ANCLA NO LOGRADA"
        )
        return {
            "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": str(CONFIG["script"]),
            "algebraica": algebraica,
            "gradiente": gradiente,
            "ejecutable": {
                "corpus_plantado": plantado,
                "comando_oficial": oficial["comando"],
                "intentos_oficial": oficial["intentos"],
                "motivo": oficial["motivo"],
                "pasa": False,
                "nota": (
                    "el ancla verifica SEMANTICA de implementacion, no direccion de rendimiento; "
                    "no exige que la SUMA gane (P81 A6)"
                ),
            },
            "veredicto": "ANCLA NO LOGRADA",
            "ok": False,
            "advertencia": (
                "ANCLA NO LOGRADA: el binario oficial fastText no produjo un .vec finito en "
                f"{INTENTOS_ANCLA_OFICIAL} intentos; los resultados downstream NO se atribuyen a "
                "la semantica SGNS publicada (persiste el limite de P81)"
            ),
        }
    palabras_oficial, w_oficial = leer_vec_fasttext(Path(oficial["ruta_vec"]))
    pureza_oficial = pureza_vecinos(w_oficial, palabras_oficial, plantado["familias"], PUREZA_K)

    dev = sembrar_global(SEMILLA_ANCLA)
    docs = leer_tokens(ruta_corpus, None)
    vocab, p, cnt = construir_vocabulario(docs, CONFIG_ANCLA.min_count, CONFIG_ANCLA.submuestreo)
    ids_por_doc = construir_ids_por_doc(docs, vocab)
    _, cdf = tabla_negativos(vocab, cnt)
    resultado = entrenar_modelos(
        [("ancla_suma", "suma")], ids_por_doc, p, cdf, SEMILLA_ANCLA, CONFIG_ANCLA, dev
    )["ancla_suma"]
    palabras_nuestra = list(vocab.keys())
    pureza_nuestra = pureza_vecinos(
        resultado.w_entrada, palabras_nuestra, plantado["familias"], PUREZA_K
    )
    log(
        f"   ancla ejecutable: pureza oficial {pureza_oficial['pureza_media']:.4f} | "
        f"nuestra {pureza_nuestra['pureza_media']:.4f}"
    )
    ejecutable_ok = bool(
        pureza_nuestra["pureza_media"] >= pureza_oficial["pureza_media"] - TOL_ANCLA_PUREZA
    )
    ok = bool(algebraica["pasa"] and gradiente["pasa"] and ejecutable_ok)
    return {
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": str(CONFIG["script"]),
        "algebraica": algebraica,
        "gradiente": gradiente,
        "ejecutable": {
            "corpus_plantado": plantado,
            "comando_oficial": oficial["comando"],
            "intentos_oficial": oficial["intentos"],
            "config_nuestra": {
                "dim": CONFIG_ANCLA.dim,
                "epocas": CONFIG_ANCLA.epocas,
                "negativos": CONFIG_ANCLA.negativos,
                "min_count": CONFIG_ANCLA.min_count,
                "ventana": CONFIG_ANCLA.ventana,
                "lote": CONFIG_ANCLA.lote,
                "lr_inicial": CONFIG_ANCLA.lr0_por_ejemplo * CONFIG_ANCLA.lote,
            },
            "pureza_oficial": pureza_oficial,
            "pureza_nuestra": pureza_nuestra,
            "tolerancia_pureza": TOL_ANCLA_PUREZA,
            "pasa": ejecutable_ok,
            "nota": (
                "el ancla verifica SEMANTICA de implementacion, no direccion de rendimiento; "
                "no exige que la SUMA gane (P81 A6)"
            ),
        },
        "veredicto": "ANCLA LOGRADA" if ok else "ANCLA NO LOGRADA",
        "ok": ok,
        "advertencia": (
            ""
            if ok
            else (
                "ANCLA NO LOGRADA: los resultados downstream NO se atribuyen a la semantica SGNS "
                "publicada (persiste el limite de P81)"
            )
        ),
    }


# ===== SECCION 4: downstream fijo (pooling, umbrales anidados y fusion) =====


def cobertura_vocabulario(textos: np.ndarray, vocab: dict[str, int]) -> dict[str, Any]:
    """Calcula la cobertura del vocabulario sobre los textos de desarrollo.

    Args:
        textos: textos de desarrollo.
        vocab: indice de palabras.

    Returns:
        Diccionario con tokens, cobertura y filas sin ningun token.
    """
    n_tokens = 0
    n_cubiertos = 0
    filas_vacias = 0
    for texto in textos:
        tokens = RE_TOKEN.findall(str(texto))
        n_tokens += len(tokens)
        cubiertos = sum(1 for token in tokens if token in vocab)
        n_cubiertos += cubiertos
        if cubiertos == 0:
            filas_vacias += 1
    return {
        "tokens_dev": n_tokens,
        "tokens_cubiertos": n_cubiertos,
        "fraccion_cubierta": round(n_cubiertos / max(n_tokens, 1), 6),
        "filas_sin_ningun_token": filas_vacias,
    }


def pooling_medio(
    textos: np.ndarray, vocab: dict[str, int], w: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mean pooling de los tokens en vocabulario (los OOV se omiten).

    Args:
        textos: textos.
        vocab: indice de palabras.
        w: embeddings de entrada (V, D).

    Returns:
        Tupla (matriz densa float32, cobertura).
    """
    salida = np.zeros((len(textos), w.shape[1]), dtype=np.float32)
    n_tokens = np.zeros(len(textos), dtype=np.int64)
    n_cubiertos = np.zeros(len(textos), dtype=np.int64)
    for i, texto in enumerate(textos):
        tokens = RE_TOKEN.findall(str(texto))
        n_tokens[i] = len(tokens)
        ii = [vocab[token] for token in tokens if token in vocab]
        n_cubiertos[i] = len(ii)
        if ii:
            salida[i] = w[np.asarray(ii, dtype=np.int64)].mean(axis=0)
    cobertura = {
        "tokens_por_fila_media": round(float(n_tokens.mean()), 2),
        "tokens_cubiertos_media": round(float(n_cubiertos.mean()), 2),
        "fraccion_cubierta": round(float(n_cubiertos.sum() / max(n_tokens.sum(), 1)), 6),
        "filas_sin_ningun_token": int((n_cubiertos == 0).sum()),
    }
    return salida, cobertura


def f1_binaria(verdad: np.ndarray, pred: np.ndarray) -> float:
    """F1 binaria de la clase positiva con zero_division=0.

    Args:
        verdad: etiquetas binarias.
        pred: predicciones binarias.

    Returns:
        F1 de la clase positiva.
    """
    verdaderos = verdad == 1
    predichos = pred == 1
    tp = float(np.sum(verdaderos & predichos))
    fp = float(np.sum(~verdaderos & predichos))
    fn = float(np.sum(verdaderos & ~predichos))
    denominador = 2 * tp + fp + fn
    return 2 * tp / denominador if denominador > 0 else 0.0


def elegir_umbral(proba: np.ndarray, verdad: np.ndarray, rejilla: np.ndarray) -> float:
    """Elige el umbral de F1 maxima en la rejilla (empate: primer maximo).

    Args:
        proba: probabilidades de la etiqueta.
        verdad: etiquetas binarias.
        rejilla: umbrales candidatos.

    Returns:
        Umbral elegido.
    """
    valores = [f1_binaria(verdad, (proba >= umbral).astype(np.int64)) for umbral in rejilla]
    return float(rejilla[int(np.argmax(valores))])


def ajustar_lr(
    x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray
) -> np.ndarray:
    """Ajusta el LR del protocolo V03 y devuelve la probabilidad de la clase positiva.

    Args:
        x_train: caracteristicas de entrenamiento ya escaladas.
        y_train: etiqueta binaria.
        x_pred: caracteristicas a predecir ya escaladas.

    Returns:
        Probabilidades (n,) de la clase positiva.
    """
    modelo = LogisticRegression(
        C=2.0,
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
        random_state=42,
    )
    modelo.fit(x_train, y_train)
    return modelo.predict_proba(x_pred)[:, 1]


def evaluar_downstream(
    d: np.ndarray,
    y: np.ndarray,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    grupos_dev: np.ndarray,
    semilla: int,
    incidencias: list[str],
) -> dict[str, Any]:
    """Ejecuta el downstream fijo por pliegue sobre un bloque denso.

    Protocolo: StandardScaler ajustado en outer_train y LR C=2.0 balanced
    liblinear rs=42 por etiqueta. Lectura anidada: umbral por (pliegue, etiqueta)
    elegido SOLO sobre las filas del outer_train cubiertas por inner_val en al
    menos uno de los 3 splits internos (una prediccion por fila, la del ultimo
    split que la cubre; las filas nunca cubiertas no entran como proba 0).
    lectura congelada: umbrales de V03 [0.44, 0.51, 0.46]. Tambien se guarda la
    F1 train@0.5 del pliegue para la brecha D-057.

    Args:
        d: bloque denso (n, D) del brazo.
        y: etiquetas (n, 3).
        pliegues: pliegues canonicos.
        grupos_dev: grupos fusionados de desarrollo.
        semilla: semilla del brazo (para particiones internas).
        incidencias: lista donde anotar avisos.

    Returns:
        Diccionario con proba OOF, umbrales anidados, cobertura interna de las
        filas usadas para elegir el umbral, F1 por pliegue y brecha.
    """
    proba = np.zeros((len(y), len(ETIQUETAS)), dtype=np.float64)
    umbrales_anidados = np.zeros((len(pliegues), len(ETIQUETAS)), dtype=np.float64)
    f1_train_05: list[float] = []
    f1_val_05: list[float] = []
    cobertura_interna: list[dict[str, Any]] = []
    for numero, (indice_train, indice_val) in enumerate(pliegues, start=1):
        a = np.asarray(indice_train)
        b = np.asarray(indice_val)
        proba_interna = np.zeros((len(a), len(ETIQUETAS)), dtype=np.float64)
        cubierta = np.zeros(len(a), dtype=bool)
        for j in range(N_INTERNOS):
            semilla_interna = SEMILLA_INTERNA_BASE + 100 * numero + 7 * j
            mascara, intentos = partir_interna_p73(grupos_dev[a], y[a], semilla_interna)
            if intentos >= 20:
                incidencias.append(
                    f"partir_interna sin convergencia de prevalencia: semilla {semilla_interna}"
                )
            esc = StandardScaler().fit(d[a[mascara]])
            x_in = esc.transform(d[a[mascara]])
            x_va = esc.transform(d[a[~mascara]])
            for etiqueta in range(len(ETIQUETAS)):
                proba_interna[~mascara, etiqueta] = ajustar_lr(
                    x_in, y[a[mascara], etiqueta], x_va
                )
            cubierta |= ~mascara
        cobertura_interna.append(
            {
                "pliegue": numero,
                "filas_outer_train": len(a),
                "filas_cubiertas": int(cubierta.sum()),
                "fraccion_cubierta": round(float(cubierta.mean()), 6),
            }
        )
        y_cubierta = y[a][cubierta]
        for etiqueta in range(len(ETIQUETAS)):
            umbrales_anidados[numero - 1, etiqueta] = elegir_umbral(
                proba_interna[cubierta, etiqueta], y_cubierta[:, etiqueta], REJILLA_UMBRALES
            )
        esc = StandardScaler().fit(d[a])
        x_a = esc.transform(d[a])
        x_b = esc.transform(d[b])
        pred_train = np.zeros((len(a), len(ETIQUETAS)), dtype=np.int64)
        for etiqueta in range(len(ETIQUETAS)):
            modelo = LogisticRegression(
                C=2.0,
                class_weight="balanced",
                max_iter=1000,
                solver="liblinear",
                random_state=42,
            )
            modelo.fit(x_a, y[a, etiqueta])
            proba[b, etiqueta] = modelo.predict_proba(x_b)[:, 1]
            pred_train[:, etiqueta] = (modelo.predict_proba(x_a)[:, 1] >= 0.5).astype(np.int64)
        pred_val = (proba[b] >= 0.5).astype(np.int64)
        f1_train_05.append(float(metrica_macro(y[a], pred_train)))
        f1_val_05.append(float(metrica_macro(y[b], pred_val)))
    brecha = [train - val for train, val in zip(f1_train_05, f1_val_05, strict=True)]
    return {
        "proba": proba,
        "umbrales_anidados": umbrales_anidados,
        "cobertura_interna": cobertura_interna,
        "f1_train_05_por_pliegue": [round(valor, 10) for valor in f1_train_05],
        "f1_val_05_por_pliegue": [round(valor, 10) for valor in f1_val_05],
        "brecha_por_pliegue": [round(valor, 10) for valor in brecha],
        "brecha_media": round(float(np.mean(brecha)), 10),
    }


def prediccion_anidada(
    proba: np.ndarray, umbrales: np.ndarray, folde: np.ndarray
) -> np.ndarray:
    """Aplica los umbrales anidados de cada pliegue a las probabilidades OOF.

    Args:
        proba: probabilidades OOF (n, 3).
        umbrales: umbrales por pliegue y etiqueta (5, 3).
        folde: numero de pliegue (1..5) por fila.

    Returns:
        Predicciones binarias (n, 3).
    """
    pred = np.zeros_like(proba, dtype=np.int64)
    for numero in range(1, umbrales.shape[0] + 1):
        filas = folde == numero
        pred[filas] = (proba[filas] >= umbrales[numero - 1][None, :]).astype(np.int64)
    return pred


def cv_fusion_tfidf(
    textos: np.ndarray,
    d: np.ndarray,
    y: np.ndarray,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    escala: float,
) -> np.ndarray:
    """CV con el protocolo de V03 mas el bloque denso a escala de norma de fila.

    La fusion sigue la convencion P71-10: el bloque denso se escala para que su
    norma de fila media sea `escala` veces la del bloque TF-IDF. La escala queda
    declarada (1.0) y es exploratoria.

    Args:
        textos: textos de desarrollo.
        d: bloque denso (n, D).
        y: etiquetas (n, 3).
        pliegues: pliegues canonicos.
        escala: escala declarada de la fusion.

    Returns:
        Probabilidades OOF (n, 3).
    """
    proba = np.zeros((len(y), len(ETIQUETAS)), dtype=np.float64)
    for indice_train, indice_val in pliegues:
        a = np.asarray(indice_train)
        b = np.asarray(indice_val)
        p2 = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        c2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
        xi = sparse.hstack([p2.fit_transform(textos[a]), c2.fit_transform(textos[a])], format="csr")
        xv = sparse.hstack([p2.transform(textos[b]), c2.transform(textos[b])], format="csr")
        esc = StandardScaler().fit(d[a])
        da, db = esc.transform(d[a]), esc.transform(d[b])
        norma_d = float(np.sqrt((da ** 2).sum(axis=1).mean()))
        norma_x = float(np.sqrt(xi.multiply(xi).sum(axis=1).mean()))
        factor = escala * norma_x / max(norma_d, 1e-12)
        xi = sparse.hstack([xi, sparse.csr_matrix(da * factor)], format="csr")
        xv = sparse.hstack([xv, sparse.csr_matrix(db * factor)], format="csr")
        for etiqueta in range(len(ETIQUETAS)):
            modelo = LogisticRegression(
                C=2.0,
                class_weight="balanced",
                max_iter=1000,
                solver="liblinear",
                random_state=42,
            )
            modelo.fit(xi, y[a, etiqueta])
            proba[b, etiqueta] = modelo.predict_proba(xv)[:, 1]
    return proba


# ===== SECCION 5: lecturas, comparaciones pareadas y gate =====


def metricas_de_lectura(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """Calcula las metricas publicadas de una lectura de umbral.

    Publica la primaria (macro-6), las de diagnostico (micro, samples,
    exact_match) y precision/recall micro y por etiqueta, ademas de la tasa de
    positivos. Se usa para las lecturas congelada, 0.5 y anidada de cada brazo.

    Args:
        y: etiquetas verdaderas (n, 3).
        pred: predicciones binarias (n, 3).

    Returns:
        Diccionario con metricas primarias, de diagnostico y por etiqueta.
    """
    metricas = dict(metricas_multietiqueta(y, pred))
    metricas["precision_micro"] = float(precision_score(y, pred, average="micro", zero_division=0))
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
    datos: DatosGate0, pred: np.ndarray, pred_ref: np.ndarray
) -> dict[str, Any]:
    """Compara pareadamente una lectura contra una referencia.

    Args:
        datos: datos de Gate 0.
        pred: predicciones del candidato.
        pred_ref: predicciones de la referencia.

    Returns:
        Diccionario con delta global, deltas por pliegue, pliegues mejorados,
        rescate/dano, tasas de positivos y F1 por pliegue.
    """
    f1_global = float(metrica_macro(datos.y, pred))
    f1_ref = float(metrica_macro(datos.y, pred_ref))
    deltas: list[float] = []
    f1_pliegues: list[float] = []
    f1_pliegues_ref: list[float] = []
    for _, indice_val in datos.pliegues:
        b = np.asarray(indice_val)
        valor = float(metrica_macro(datos.y[b], pred[b]))
        valor_ref = float(metrica_macro(datos.y[b], pred_ref[b]))
        f1_pliegues.append(round(valor, 8))
        f1_pliegues_ref.append(round(valor_ref, 8))
        deltas.append(round(valor - valor_ref, 8))
    ok_ref = pred_ref == datos.y
    ok_cand = pred == datos.y
    rescate = int(np.sum(~ok_ref & ok_cand))
    dano = int(np.sum(ok_ref & ~ok_cand))
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
        "tasa_positivos": [float(valor) for valor in pred.mean(axis=0)],
        "tasa_positivos_referencia": [float(valor) for valor in pred_ref.mean(axis=0)],
    }


def comparar_por_semilla(
    semillas: Sequence[int],
    pred_por_semilla: dict[int, np.ndarray],
    ref_por_semilla: dict[int, np.ndarray],
    datos: DatosGate0,
) -> dict[str, Any]:
    """Compara un brazo contra una referencia pareando por semilla y pliegue.

    Args:
        semillas: semillas de la corrida.
        pred_por_semilla: predicciones del candidato por semilla.
        ref_por_semilla: predicciones de la referencia por semilla.
        datos: datos de Gate 0.

    Returns:
        Diccionario con deltas por semilla, deltas por (semilla, pliegue) y
        numero de pares ganados.
    """
    por_semilla: dict[str, Any] = {}
    ganados = 0
    for semilla in semillas:
        deltas: list[float] = []
        for _, indice_val in datos.pliegues:
            b = np.asarray(indice_val)
            delta = float(metrica_macro(datos.y[b], pred_por_semilla[semilla][b])) - float(
                metrica_macro(datos.y[b], ref_por_semilla[semilla][b])
            )
            deltas.append(round(delta, 8))
            if delta > 0:
                ganados += 1
        delta_global = float(metrica_macro(datos.y, pred_por_semilla[semilla])) - float(
            metrica_macro(datos.y, ref_por_semilla[semilla])
        )
        por_semilla[str(semilla)] = {
            "delta_global": round(delta_global, 10),
            "delta_medio_pliegues": round(float(np.mean(deltas)), 10),
            "delta_por_pliegue": deltas,
            "pliegues_mejorados": int(np.sum(np.array(deltas) > 0)),
        }
    return {
        "por_semilla": por_semilla,
        "pares_semilla_pliegue_ganados": ganados,
        "pares_semilla_pliegue_totales": len(semillas) * len(datos.pliegues),
    }


def bootstrap_ensemble_gain(
    y: np.ndarray,
    proba_candidato: np.ndarray,
    proba_v03: np.ndarray,
    repeticiones: int,
    semilla: int,
) -> dict[str, Any]:
    """Calcula ensemble_gain con IC bootstrap pareado por filas.

    Convencion P6/P26: F1 de la media de probabilidades a 0.5 menos la F1 de V03
    a 0.5 recalculada desde el npz; se verifica que 0.771778476256122 se
    reproduce antes de publicar.

    Args:
        y: etiquetas (n, 3).
        proba_candidato: probabilidades del candidato.
        proba_v03: probabilidades de V03.
        repeticiones: numero de remuestreos.
        semilla: semilla del generador.

    Returns:
        Diccionario con la referencia recalculada, el gain y su IC95.

    Raises:
        SystemExit: si la F1 de V03 a 0.5 no reproduce la constante congelada.
    """
    pred_media = ((proba_candidato + proba_v03) / 2.0 >= 0.5).astype(np.int64)
    pred_v03 = (proba_v03 >= 0.5).astype(np.int64)
    f1_media = float(metrica_macro(y, pred_media))
    f1_v03 = float(metrica_macro(y, pred_v03))
    if abs(f1_v03 - V03_F1_AT_05) > TOL_IDENTIDAD:
        raise SystemExit(f"STOP: V03@0.5 no reproduce en ensemble_gain: {f1_v03!r}")
    base = bootstrap_diferencia_pareada(
        y, pred_media, pred_v03, repeticiones=repeticiones, semilla=semilla
    )
    return {
        "f1_media_candidato_v03_05": round(f1_media, 10),
        "f1_v03_05_recalculado": round(f1_v03, 12),
        "ensemble_gain": round(f1_media - f1_v03, 10),
        "ic95": base["ic95"],
        "mediana_bootstrap": base["mediana_bootstrap"],
        "repeticiones": repeticiones,
        "semilla": semilla,
    }


def matriz_errores(y: np.ndarray, pred: np.ndarray, pred_ref: np.ndarray) -> dict[str, Any]:
    """Construye la matriz de solapamiento de errores frente a una referencia.

    Args:
        y: etiquetas verdaderas (n, 3).
        pred: predicciones del candidato.
        pred_ref: predicciones de la referencia.

    Returns:
        Recuentos por etiqueta y globales mas el Jaccard de conjuntos de error.
    """
    por_etiqueta: dict[str, Any] = {}
    for j, etiqueta in enumerate(ETIQUETAS):
        ok_cand = pred[:, j] == y[:, j]
        ok_ref = pred_ref[:, j] == y[:, j]
        error_cand = ~ok_cand
        error_ref = ~ok_ref
        interseccion = int(np.sum(error_cand & error_ref))
        union = int(np.sum(error_cand | error_ref))
        por_etiqueta[etiqueta] = {
            "ambos_correctos": int(np.sum(ok_cand & ok_ref)),
            "solo_candidato_error": int(np.sum(error_cand & ok_ref)),
            "solo_referencia_error": int(np.sum(ok_cand & error_ref)),
            "ambos_error": interseccion,
            "jaccard_error": round(interseccion / union, 6) if union > 0 else None,
        }
    ok_cand_global = (pred == y).ravel()
    ok_ref_global = (pred_ref == y).ravel()
    error_cand_global = ~ok_cand_global
    error_ref_global = ~ok_ref_global
    interseccion = int(np.sum(error_cand_global & error_ref_global))
    union = int(np.sum(error_cand_global | error_ref_global))
    globales = {
        "celdas_totales": int(y.size),
        "ambos_correctos": int(np.sum(ok_cand_global & ok_ref_global)),
        "solo_candidato_error": int(np.sum(error_cand_global & ok_ref_global)),
        "solo_referencia_error": int(np.sum(ok_cand_global & error_ref_global)),
        "ambos_error": interseccion,
        "jaccard_error": round(interseccion / union, 6) if union > 0 else None,
    }
    return {"por_etiqueta": por_etiqueta, "global": globales}


def evaluar_gate(
    comparaciones: dict[str, Any],
    ensemble_gain_s: dict[str, Any],
    brecha: dict[str, Any],
    ancla_ok: bool,
) -> dict[str, Any]:
    """Evalua el gate de EXP-04.

    Gate: S > M y S > R (delta > 0, >= 4/5 pliegues, IC95 > 0) y ademas
    (S vs V03 con +0.005 / 4/5 / IC95 > 0 o ensemble_gain >= 0.0015 con IC > 0).
    D-057: no promover si la brecha train-validacion de S supera a la de V03 en
    mas de +0.05 absoluto.

    Args:
        comparaciones: comparaciones primarias (congeladas) por par.
        ensemble_gain_s: ensemble_gain de S con V03.
        brecha: brechas train-validacion y capacidad.
        ancla_ok: veredicto del ancla de implementacion.

    Returns:
        Diccionario con subgates, decision y motivos.
    """

    def _subgate(nombre: str, delta_minimo: float) -> dict[str, Any]:
        """Construye un subgate a partir de la comparacion pareada.

        Args:
            nombre: clave de la comparacion.
            delta_minimo: delta minimo exigido.

        Returns:
            Diccionario del subgate.
        """
        comparacion = comparaciones[nombre]
        ic95 = comparacion["bootstrap"]["ic95"]
        delta = float(comparacion["delta_global"])
        pliegues = int(comparacion["pliegues_mejorados"])
        return {
            "comparacion": nombre,
            "delta_global": delta,
            "delta_minimo": delta_minimo,
            "pliegues_mejorados": pliegues,
            "pliegues_minimos": GATE_PLIEGUES,
            "ic95": ic95,
            "pasa": bool(delta >= delta_minimo and pliegues >= GATE_PLIEGUES and ic95[0] > 0),
        }

    control_m = _subgate("S_vs_M", 0.0)
    control_r = _subgate("S_vs_R", 0.0)
    standalone = _subgate("S_vs_V03", GATE_DELTA)
    gain = float(ensemble_gain_s["ensemble_gain"])
    ic_gain = ensemble_gain_s["ic95"]
    complementariedad = {
        "ensemble_gain": gain,
        "umbral_gain": GATE_ENSEMBLE_GAIN,
        "ic95": ic_gain,
        "pasa": bool(gain >= GATE_ENSEMBLE_GAIN and ic_gain[0] > 0),
    }
    brecha_ok = bool(brecha["brecha_ok"])
    elegible = bool(
        control_m["pasa"]
        and control_r["pasa"]
        and (standalone["pasa"] or complementariedad["pasa"])
        and brecha_ok
        and ancla_ok
    )
    motivos: list[str] = []
    if not ancla_ok:
        motivos.append(
            "ANCLA NO LOGRADA: los resultados downstream no se atribuyen a la semantica SGNS "
            "publicada; no promovible"
        )
    if not control_m["pasa"]:
        motivos.append(
            f"S vs M NO: delta {control_m['delta_global']:+.6f}, pliegues "
            f"{control_m['pliegues_mejorados']}/5, IC95 "
            f"[{control_m['ic95'][0]:+.6f}, {control_m['ic95'][1]:+.6f}]"
        )
    if not control_r["pasa"]:
        motivos.append(
            f"S vs R NO: delta {control_r['delta_global']:+.6f}, pliegues "
            f"{control_r['pliegues_mejorados']}/5, IC95 "
            f"[{control_r['ic95'][0]:+.6f}, {control_r['ic95'][1]:+.6f}]"
        )
    if not standalone["pasa"]:
        motivos.append(
            f"S vs V03 standalone NO: delta {standalone['delta_global']:+.6f} "
            f"(min {GATE_DELTA}), IC95 "
            f"[{standalone['ic95'][0]:+.6f}, {standalone['ic95'][1]:+.6f}]"
        )
    if not complementariedad["pasa"]:
        motivos.append(
            f"complementariedad NO: ensemble_gain {gain:+.6f} (min {GATE_ENSEMBLE_GAIN}), "
            f"IC95 [{ic_gain[0]:+.6f}, {ic_gain[1]:+.6f}]"
        )
    if not brecha_ok:
        motivos.append(
            f"D-057 NO: brecha S {brecha['brecha_S']:+.6f} vs V03 "
            f"{brecha['brecha_V03']:+.6f} (max +{GATE_BRECHA_MAX})"
        )
    if not motivos:
        motivos.append("todos los subgates pasan")
    if not ancla_ok:
        decision = (
            "NO PROMOVIBLE: ancla de implementacion no lograda; los resultados downstream no se "
            "atribuyen a la semantica SGNS publicada"
        )
    elif elegible:
        decision = "ELEGIBLE PARA EXP-06 (pendiente de Gate 2; sin submission)"
    else:
        decision = "CERRAR la linea SGNS corregido con el resultado verificado (sin rejilla nueva)"
    return {
        "control_S_vs_M": control_m,
        "control_S_vs_R": control_r,
        "standalone_S_vs_V03": standalone,
        "complementariedad": complementariedad,
        "brecha_ok": brecha_ok,
        "ancla_ok": ancla_ok,
        "elegible_exp06": elegible,
        "decision": decision,
        "motivos": motivos,
    }


# ===== SECCION 6: parciales, embeddings, OOF, informes y CSV =====


@dataclass
class Unidad:
    """Unidad completa (semilla, brazo): probabilidades, bloque denso y metadatos.

    Attributes:
        semilla: semilla.
        brazo: brazo ("S", "M" o "R").
        proba: probabilidades OOF (1483, 3).
        d_pool: bloque denso pooled (1483, D) float32.
        umbrales_anidados: umbrales anidados por pliegue y etiqueta (5, 3).
        meta: metadatos de la unidad (entrenamiento, downstream, cobertura).
    """

    semilla: int
    brazo: str
    proba: np.ndarray
    d_pool: np.ndarray
    umbrales_anidados: np.ndarray
    meta: dict[str, Any]


def rutas_parcial(semilla: int, brazo: str, humo: bool) -> tuple[Path, Path]:
    """Devuelve las rutas (json, npz) del parcial de una unidad.

    Args:
        semilla: semilla.
        brazo: brazo.
        humo: si True usa el sufijo _humo.

    Returns:
        Tupla (ruta_json, ruta_npz).
    """
    sufijo = "_humo" if humo else ""
    base = CONFIG["parciales_dir"] / f"p84_04_parcial_{semilla}_{brazo}{sufijo}"
    return base.with_suffix(".json"), base.with_suffix(".npz")


def escribir_parcial(unidad: Unidad, humo: bool) -> list[str]:
    """Escribe el parcial (npz + json) de una unidad completa.

    El npz guarda proba, bloque denso y umbrales; el json guarda los metadatos.
    El json se escribe despues del npz, de modo que su presencia implica
    unidad completa y reanudable.

    Args:
        unidad: unidad completa.
        humo: si True usa el sufijo _humo.

    Returns:
        Rutas escritas.
    """
    ruta_json, ruta_npz = rutas_parcial(unidad.semilla, unidad.brazo, humo)
    ruta_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ruta_npz,
        proba=unidad.proba,
        d_pool=unidad.d_pool,
        umbrales_anidados=unidad.umbrales_anidados,
    )
    unidad.meta["npz"] = str(ruta_npz.relative_to(PROJECT_ROOT))
    escribir_json(ruta_json, unidad.meta)
    return [str(ruta_json.relative_to(PROJECT_ROOT)), str(ruta_npz.relative_to(PROJECT_ROOT))]


def cargar_parcial(
    semilla: int, brazo: str, humo: bool, firma: dict[str, Any] | None = None
) -> Unidad | None:
    """Carga un parcial completo si existe, es valido y coincide con la firma.

    Args:
        semilla: semilla.
        brazo: brazo.
        humo: si True usa el sufijo _humo.
        firma: configuracion+firma del corpus de la corrida actual (incluye la
            version del script y su sha256); si se da, el parcial debe coincidir
            para poder reanudar.

    Returns:
        Unidad cargada o None si no existe, esta incompleta o no coincide.
    """
    ruta_json, ruta_npz = rutas_parcial(semilla, brazo, humo)
    if not (ruta_json.exists() and ruta_npz.exists()):
        return None
    try:
        meta = json.loads(ruta_json.read_text(encoding="ascii"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if meta.get("estado") != "completo":
        return None
    if int(meta.get("semilla", -1)) != semilla or meta.get("brazo") != brazo:
        return None
    if firma is not None and meta.get("firma") != firma:
        return None
    datos = np.load(ruta_npz, allow_pickle=True)
    proba = datos["proba"].astype(np.float64)
    d_pool = datos["d_pool"].astype(np.float32)
    umbrales = datos["umbrales_anidados"].astype(np.float64)
    if proba.shape != (FILAS_DEV, len(ETIQUETAS)):
        return None
    if d_pool.shape[0] != FILAS_DEV or umbrales.shape != (5, len(ETIQUETAS)):
        return None
    return Unidad(
        semilla=semilla,
        brazo=brazo,
        proba=proba,
        d_pool=d_pool,
        umbrales_anidados=umbrales,
        meta=meta,
    )


def guardar_embeddings(
    semilla: int, brazo: str, w: np.ndarray, palabras: list[str]
) -> dict[str, str]:
    """Guarda el embedding de una unidad en modelos/p84/.

    Args:
        semilla: semilla.
        brazo: brazo.
        w: matriz de entrada (V, D).
        palabras: vocabulario alineado con w.

    Returns:
        Diccionario con ruta relativa y sha256 del fichero.
    """
    ruta = CONFIG["embeddings_dir"] / f"emb_{semilla}_{brazo}.npz"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ruta,
        W=w.astype(np.float32),
        palabras=np.asarray(palabras, dtype=object),
        dim=int(w.shape[1]),
        semilla=int(semilla),
        brazo=brazo,
    )
    return {
        "ruta": str(ruta.relative_to(PROJECT_ROOT)),
        "sha256": sha256_of(ruta),
    }


def guardar_oof_npz(
    unidades: dict[tuple[int, str], Unidad],
    semillas: Sequence[int],
    proba_promedio: dict[str, np.ndarray],
    umbrales_anidados_promedio: dict[str, np.ndarray],
    datos: DatosGate0,
) -> Path:
    """Guarda el npz OOF con las probabilidades por brazo y semilla.

    Args:
        unidades: unidades completas indexadas por (semilla, brazo).
        semillas: semillas de la corrida.
        proba_promedio: probabilidad media por brazo.
        umbrales_anidados_promedio: umbrales anidados medios por brazo.
        datos: datos de Gate 0.

    Returns:
        Ruta del npz escrito.
    """
    ruta: Path = CONFIG["oof_npz"]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    carga: dict[str, Any] = {
        "ids": datos.ids,
        "y_true": datos.y,
        "pliegues": datos.folde,
        "semillas": np.asarray(semillas, dtype=np.int64),
        "umbrales_congelados": UMBRALES_CONGELADOS,
        "proba_promedio_S": proba_promedio["S"],
        "proba_promedio_M": proba_promedio["M"],
        "proba_promedio_R": proba_promedio["R"],
        "umbrales_anidados_promedio_S": umbrales_anidados_promedio["S"],
        "umbrales_anidados_promedio_M": umbrales_anidados_promedio["M"],
        "umbrales_anidados_promedio_R": umbrales_anidados_promedio["R"],
    }
    for brazo in BRAZOS:
        carga[f"proba_{brazo}"] = np.stack([unidades[(s, brazo)].proba for s in semillas])
        carga[f"umbrales_anidados_{brazo}"] = np.stack(
            [unidades[(s, brazo)].umbrales_anidados for s in semillas]
        )
    np.savez_compressed(ruta, **carga)
    return ruta


def upsert_experiments_csv(fila: dict[str, str]) -> dict[str, Any]:
    """Inserta o reemplaza la fila P84-04 en experiments.csv (23 columnas).

    Valida la cabecera, el ancho de todas las filas, la ausencia de ids
    duplicados y que la fila defina exactamente las 23 columnas antes de
    escribir. Escribe utf-8-sig y no duplica ids.

    Args:
        fila: valores por columna de la fila P84-04.

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
        raise SystemExit(f"STOP: faltan columnas en la fila P84-04: {faltantes}")
    sobrantes = sorted(set(fila) - set(COLUMNAS_EXPERIMENTS))
    if sobrantes:
        raise SystemExit(f"STOP: columnas desconocidas en la fila P84-04: {sobrantes}")
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
    carga: dict[str, Any], segundos_totales: float
) -> dict[str, str]:
    """Construye la fila P84-04 para experiments.csv.

    Args:
        carga: informe completo de la corrida.
        segundos_totales: tiempo total de la corrida.

    Returns:
        Diccionario columna -> valor.
    """
    comparaciones = carga["downstream"]["comparaciones"]
    s_vs_v03 = comparaciones["S_vs_V03"]
    ganancia = carga["downstream"]["ensemble_gain"]["S"]
    gate = carga["gate"]
    f1_por_semilla = carga["downstream"]["f1_congelada_por_semilla_brazo"]
    lecturas = carga["downstream"]["lecturas"]
    metricas_primarias = lecturas["S"]["congelada"]
    resultado = {
        "f1_congelada_S_promedio": carga["downstream"]["f1_congelada_promedio"]["S"],
        "f1_congelada_promedio_por_brazo": carga["downstream"]["f1_congelada_promedio"],
        "f1_congelada_por_semilla_brazo": f1_por_semilla,
        "delta_global_S_vs_M": comparaciones["S_vs_M"]["delta_global"],
        "delta_global_S_vs_R": comparaciones["S_vs_R"]["delta_global"],
        "delta_global_S_vs_V03": s_vs_v03["delta_global"],
        "pliegues_mejorados_S_vs_M": comparaciones["S_vs_M"]["pliegues_mejorados"],
        "pliegues_mejorados_S_vs_R": comparaciones["S_vs_R"]["pliegues_mejorados"],
        "pliegues_mejorados_S_vs_V03": s_vs_v03["pliegues_mejorados"],
        "ic95_S_vs_M": comparaciones["S_vs_M"]["bootstrap"]["ic95"],
        "ic95_S_vs_R": comparaciones["S_vs_R"]["bootstrap"]["ic95"],
        "ic95_S_vs_V03": s_vs_v03["bootstrap"]["ic95"],
        "ensemble_gain_S": ganancia,
        "ancla": carga["ancla"]["veredicto"],
        "brecha": carga["d057"],
        "lecturas_f1_macro6": {
            brazo: {
                clave: round(metricas["f1_macro6"], 10) for clave, metricas in lectura.items()
            }
            for brazo, lectura in lecturas.items()
        },
        "lectura_primaria_congelada": {
            "f1_macro6": metricas_primarias["f1_macro6"],
            "f1_micro": metricas_primarias["f1_micro"],
            "f1_samples": metricas_primarias["f1_samples"],
            "exact_match": metricas_primarias["exact_match"],
            "f1_por_etiqueta": metricas_primarias["f1_por_etiqueta"],
            "precision_micro": metricas_primarias["precision_micro"],
            "recall_micro": metricas_primarias["recall_micro"],
            "precision_por_etiqueta": metricas_primarias["precision_por_etiqueta"],
            "recall_por_etiqueta": metricas_primarias["recall_por_etiqueta"],
            "tasa_positivos": metricas_primarias["tasa_positivos"],
        },
        "gate": gate,
    }
    return {
        "experiment_id": "P84-04",
        "competition": str(CONFIG["competencia"]),
        "question": (
            "Cambia el downstream (mean pooling + LR) cuando se corrige la agregacion del "
            "termino negativo de SGNS (suma de las K contribuciones) frente a su control "
            "pareado de media 1/K y a una matriz aleatoria congelada, con ancla externa, y "
            "aporta complementariedad a V03 (corrected_cv 0.7772114525165721)?"
        ),
        "hypothesis": (
            "El entregable primario es cerrar el defecto de objetivo con evidencia verificable; "
            "S vs M confunde la correccion con la magnitud del gradiente y el unico camino "
            "plausible de promocion es la complementariedad (ensemble_gain >= 0.0015, IC95 > 0)."
        ),
        "preprocessing": (
            "corpus externo wikipedia_es 10M EXTERNAL-ONLY (sin texto de la competencia); "
            "tokenizador [#@]?\\w+(?:['\\u2019]\\w+)*, min_count 3, ventana 5 dinamica, "
            "subsampling 1e-4; downstream: mean pooling con OOV omitidos y StandardScaler "
            "ajustado en outer_train"
        ),
        "feature_groups": "embeddings SGNS entrenados desde cero (dim 200) + mean pooling",
        "representation": (
            "skip-gram con negative sampling corregido (S: suma de K=10 contribuciones; "
            "M: media 1/K; R: matriz aleatoria congelada)"
        ),
        "model": "SGNS propio en PyTorch (SGD, lr 0.025*lote con decaimiento lineal a 1e-3)",
        "hyperparameters": json.dumps(
            {
                "dim": CONFIG_SGNS.dim,
                "ventana": CONFIG_SGNS.ventana,
                "min_count": CONFIG_SGNS.min_count,
                "negativos": CONFIG_SGNS.negativos,
                "submuestreo": CONFIG_SGNS.submuestreo,
                "epocas": CONFIG_SGNS.epocas,
                "lote": CONFIG_SGNS.lote,
                "lr_inicial": CONFIG_SGNS.lr0_por_ejemplo * CONFIG_SGNS.lote,
                "lr_minimo": CONFIG_SGNS.lr_minimo,
                "init_entrada": "uniforme +-0.5/dim",
                "init_salida": "uniforme [-0.05, 0.05]",
                "semillas": list(SEMILLAS_DEFECTO),
            },
            ensure_ascii=True,
        ),
        "validation": (
            "pliegues_corregidos de 5 pliegues sobre desarrollo (1483, grupos fusionados); LR "
            "C=2.0 balanced liblinear rs=42 por pliegue; lectura primaria con umbrales "
            "congelados 0.44/0.51/0.46 y lectura secundaria anidada (rejilla 0.20-0.80 en "
            "particiones internas); bootstrap pareado 10000 semilla 20260926"
        ),
        "f1": f"{carga['downstream']['f1_congelada_promedio']['S']:.6f}",
        "macro_f1": f"{metricas_primarias['f1_macro6']:.6f}",
        "weighted_f1": "",
        "precision": f"{metricas_primarias['precision_micro']:.6f}",
        "recall": f"{metricas_primarias['recall_micro']:.6f}",
        "training_time": f"{segundos_totales:.2f}",
        "dimension": (
            f"dim 200; vocabulario {carga['corpus']['vocabulario']} palabras; corpus externo "
            "wikipedia_es tokens_10M; pooling denso 200"
        ),
        "result": json.dumps(resultado, ensure_ascii=True, default=_json_default),
        "interpretation": (
            f"S congelada {carga['downstream']['f1_congelada_promedio']['S']:.6f} vs V03 "
            f"{s_vs_v03['f1_referencia_global']:.6f}; delta S vs V03 "
            f"{s_vs_v03['delta_global']:+.6f} ({s_vs_v03['pliegues_mejorados']}/5, IC95 "
            f"[{s_vs_v03['bootstrap']['ic95'][0]:+.6f}, "
            f"{s_vs_v03['bootstrap']['ic95'][1]:+.6f}]); delta S vs M "
            f"{comparaciones['S_vs_M']['delta_global']:+.6f}; delta S vs R "
            f"{comparaciones['S_vs_R']['delta_global']:+.6f}; ensemble_gain "
            f"{ganancia['ensemble_gain']:+.6f} (IC95 [{ganancia['ic95'][0]:+.6f}, "
            f"{ganancia['ic95'][1]:+.6f}]); ancla {carga['ancla']['veredicto']}"
        ),
        "decision": str(gate["decision"]),
        "date": str(CONFIG["fecha"]),
        "script": str(CONFIG["script"]),
        "notes": (
            "EXP-04 del preregistro P84. loss_suma y loss_media NO se comparan entre si; S vs M "
            "confunde correccion con magnitud del gradiente. Contadores dev 1483 / holdout 0 / "
            "test 0; bootstrap 10000 semilla 20260926. RUNLOG.md y DECISIONS.md los actualiza la "
            "corrida del trainer, no el script."
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
    lineas.append("# P84-04: SGNS corregido y anclado (EXP-04)")
    lineas.append("")
    lineas.append(f"- Fecha: {carga['fecha']}")
    lineas.append(f"- Script: {carga['script']}")
    lineas.append("- Especificacion: docs/p84_spec.md seccion 5; preregistro docs/p84_preregistro.md")
    lineas.append(f"- Referencia V03: corrected_cv {V03_CCV}; V03@0.5 {V03_F1_AT_05}")
    lineas.append("- V03 remains the best verified leaderboard result.")
    lineas.append("")
    ancla = carga.get("ancla", {})
    lineas.append("## Ancla de implementacion")
    lineas.append("")
    lineas.append(f"- Veredicto: {ancla.get('veredicto', 'NO EJECUTADA')}")
    if ancla:
        algebraica = ancla.get("algebraica", {})
        gradiente = ancla.get("gradiente", {})
        ejecutable = ancla.get("ejecutable", {})
        pureza_oficial = ejecutable.get("pureza_oficial", {}).get("pureza_media")
        pureza_nuestra = ejecutable.get("pureza_nuestra", {}).get("pureza_media")
        lineas.append(
            f"- Algebraica: error relativo N {algebraica.get('error_relativo_N')}, loss "
            f"{algebraica.get('error_relativo_loss')} (tol {TOL_ANCLA_ALGEBRA})"
        )
        lineas.append(
            f"- Gradiente (diferencias finitas): error max {gradiente.get('error_relativo_max')} "
            f"(tol {TOL_ANCLA_GRADIENTE})"
        )
        lineas.append(
            f"- Ejecutable: pureza oficial {pureza_oficial} vs nuestra {pureza_nuestra} "
            f"(tol {TOL_ANCLA_PUREZA})"
        )
    lineas.append("")
    corpus = carga.get("corpus", {})
    lineas.append("## Corpus y vocabulario")
    lineas.append("")
    lineas.append(
        f"- Documentos {corpus.get('documentos')} | tokens {corpus.get('tokens')} | "
        f"vocabulario {corpus.get('vocabulario')} | cobertura dev "
        f"{corpus.get('cobertura_dev', {}).get('fraccion_cubierta')}"
    )
    lineas.append("")
    if carga.get("corrida_completa"):
        downstream = carga["downstream"]
        lineas.append("## Downstream (lectura primaria: umbrales congelados 0.44/0.51/0.46)")
        lineas.append("")
        lineas.append("| brazo | F1 congelada | por semilla |")
        lineas.append("|---|---|---|")
        for brazo in BRAZOS:
            por_semilla = downstream["f1_congelada_por_semilla_brazo"][brazo]
            lineas.append(
                f"| {brazo} | {downstream['f1_congelada_promedio'][brazo]:.6f} | "
                f"{', '.join(f'{s}: {v:.6f}' for s, v in por_semilla.items())} |"
            )
        lineas.append("")
        lineas.append("## Comparaciones pareadas (S menos referencia)")
        lineas.append("")
        lineas.append("| comparacion | delta | pliegues | IC95 | rescate/dano |")
        lineas.append("|---|---|---|---|---|")
        for nombre in ("S_vs_M", "S_vs_R", "S_vs_V03"):
            comparacion = downstream["comparaciones"][nombre]
            ic = comparacion["bootstrap"]["ic95"]
            lineas.append(
                f"| {nombre} | {comparacion['delta_global']:+.6f} | "
                f"{comparacion['pliegues_mejorados']}/5 | "
                f"[{ic[0]:+.6f}, {ic[1]:+.6f}] | "
                f"{comparacion['rescate_celdas']}/{comparacion['dano_celdas']} |"
            )
        lineas.append("")
        lineas.append("## ensemble_gain (media de probabilidades con V03 a 0.5)")
        lineas.append("")
        for brazo in BRAZOS:
            ganancia = downstream["ensemble_gain"][brazo]
            lineas.append(
                f"- {brazo}: gain {ganancia['ensemble_gain']:+.6f} | IC95 "
                f"[{ganancia['ic95'][0]:+.6f}, {ganancia['ic95'][1]:+.6f}]"
            )
        lineas.append("")
        lineas.append(
            "Nota de lectura: loss_suma y loss_media NO son comparables entre si (distinto "
            "funcional y distinta magnitud del gradiente negativo con el mismo lr); la "
            "comparacion S vs M confunde la correccion del objetivo con la magnitud del "
            "gradiente. Solo se comparan metricas downstream."
        )
        lineas.append("")
        d057 = carga.get("d057", {})
        lineas.append("## D-057 (brecha train-validacion)")
        lineas.append("")
        lineas.append(
            f"- brecha S {d057.get('brecha_S')} | V03 {d057.get('brecha_V03')} | "
            f"R {d057.get('brecha_R')} | M {d057.get('brecha_M')} | "
            f"max +{GATE_BRECHA_MAX} | ok {d057.get('brecha_ok')}"
        )
        lineas.append("")
        secundario = carga.get("secundario", {})
        lineas.append("## Downstream secundario (exploratorio)")
        lineas.append("")
        if secundario.get("ejecutado"):
            fusion = secundario["fusion_tfidf_escala_1_0"]
            lineas.append(
                f"- Fusion TF-IDF escala 1.0: F1 congelada {fusion['f1_congelada']:.6f} | "
                f"delta vs V03 {fusion['delta_vs_v03']:+.6f} | delta vs S "
                f"{fusion['delta_vs_S']:+.6f}"
            )
            lineas.append(
                "- Media de probabilidades (S+V03): ver ensemble_gain de S (diagnostico)."
            )
        else:
            lineas.append(f"- No ejecutado: {secundario.get('motivo')}")
        lineas.append("")
    gate = carga.get("gate", {})
    lineas.append("## Gate")
    lineas.append("")
    lineas.append(f"- Decision: {gate.get('decision')}")
    for motivo in gate.get("motivos", []):
        lineas.append(f"- {motivo}")
    lineas.append("")
    lineas.append(
        f"- Tiempo total: {carga.get('tiempo_total_segundos')} s | device: "
        f"{carga.get('versiones', {}).get('device')}"
    )
    lineas.append("")
    return "\n".join(lineas)


# ===== SECCION 7: CLI y orquestacion =====


def parsear_semillas(texto: str) -> tuple[int, ...]:
    """Parsea la lista de semillas separadas por comas.

    Args:
        texto: cadena de entrada.

    Returns:
        Tupla de semillas sin duplicados.

    Raises:
        SystemExit: si el formato es invalido o hay duplicados.
    """
    try:
        valores = tuple(int(parte) for parte in texto.split(",") if parte.strip())
    except ValueError as error:
        raise SystemExit(f"STOP: --semillas invalido: {texto!r}") from error
    if not valores:
        raise SystemExit("STOP: --semillas no puede estar vacio")
    if len(set(valores)) != len(valores):
        raise SystemExit(f"STOP: --semillas tiene valores duplicados: {valores}")
    return valores


def parsear_argumentos(argv: Sequence[str] | None) -> argparse.Namespace:
    """Define y parsea los argumentos de la linea de comandos.

    Args:
        argv: argumentos (None usa sys.argv[1:]).

    Returns:
        Namespace con humo, reanudar y semillas.
    """
    analizador = argparse.ArgumentParser(
        description="P84-04: EXP-04 (SGNS corregido y anclado) frente a V03."
    )
    analizador.add_argument(
        "--humo",
        action="store_true",
        help="modo humo: corpus diminuto, 1 semilla, 2 epocas; sin finales ni CSV",
    )
    analizador.add_argument(
        "--reanudar",
        action="store_true",
        help=(
            "salta las unidades (semilla, brazo) cuyo parcial ya esta completo y coincide con "
            f"la firma v{VERSION_SCRIPT} (incluye el sha256 del script)"
        ),
    )
    analizador.add_argument(
        "--semillas",
        default=",".join(str(s) for s in SEMILLAS_DEFECTO),
        help="semillas separadas por comas (defecto 20261016,20261017,20261018)",
    )
    return analizador.parse_args(argv)


def verificar_flujo_pareado(unidades: dict[tuple[int, str], Unidad], semillas: Sequence[int]) -> list[dict[str, Any]]:
    """Verifica que S y M compartieron pares, permutacion y negativos por semilla.

    Args:
        unidades: unidades completas.
        semillas: semillas de la corrida.

    Returns:
        Evidencia por semilla.

    Raises:
        SystemExit: si los hashes del flujo de S y M no coinciden.
    """
    evidencia: list[dict[str, Any]] = []
    for semilla in semillas:
        entrenamiento_s = unidades[(semilla, "S")].meta.get("entrenamiento") or {}
        entrenamiento_m = unidades[(semilla, "M")].meta.get("entrenamiento") or {}
        pares_s = entrenamiento_s.get("hash_pares_epoca1")
        pares_m = entrenamiento_m.get("hash_pares_epoca1")
        neg_s = entrenamiento_s.get("hash_negativos_lote1")
        neg_m = entrenamiento_m.get("hash_negativos_lote1")
        iguales = bool(pares_s and pares_s == pares_m and neg_s == neg_m)
        if not iguales:
            raise SystemExit(
                f"STOP: el flujo de datos de S y M no coincide en la semilla {semilla}: "
                f"pares {pares_s} vs {pares_m}, negativos {neg_s} vs {neg_m}"
            )
        evidencia.append(
            {
                "semilla": int(semilla),
                "hash_pares_epoca1": pares_s,
                "hash_negativos_lote1": neg_s,
                "iguales": True,
            }
        )
    return evidencia


def procesar_unidad(
    semilla: int,
    brazo: str,
    w: np.ndarray,
    meta_entrenamiento: dict[str, Any] | None,
    datos: DatosGate0,
    vocab: dict[str, int],
    palabras: list[str],
    humo: bool,
    incidencias: list[str],
    firma: dict[str, Any],
) -> Unidad:
    """Completa una unidad (semilla, brazo): pooling, downstream, parciales y embedding.

    Args:
        semilla: semilla.
        brazo: brazo.
        w: embedding de entrada (V, D).
        meta_entrenamiento: metadatos de entrenamiento (None para R).
        datos: datos de Gate 0.
        vocab: indice de palabras.
        palabras: vocabulario alineado con w.
        humo: si True no guarda embeddings definitivos.
        incidencias: lista donde anotar avisos.
        firma: configuracion+firma del corpus para validar la reanudacion.

    Returns:
        Unidad completa.
    """
    d_pool, cobertura = pooling_medio(datos.textos, vocab, w)
    down = evaluar_downstream(d_pool, datos.y, datos.pliegues, datos.grupos_dev, semilla, incidencias)
    pred_congelada = (down["proba"] >= UMBRALES_CONGELADOS[None, :]).astype(np.int64)
    f1_congelada = float(metrica_macro(datos.y, pred_congelada))
    pred_nidada = prediccion_anidada(down["proba"], down["umbrales_anidados"], datos.folde)
    f1_anidada = float(metrica_macro(datos.y, pred_nidada))
    meta: dict[str, Any] = {
        "experiment_id": "P84-04",
        "fecha": str(CONFIG["fecha"]),
        "semilla": int(semilla),
        "brazo": brazo,
        "agregacion": AGREGACIONES[brazo],
        "estado": "completo",
        "modo_humo": humo,
        "firma": firma,
        "entrenamiento": meta_entrenamiento,
        "downstream": {
            "f1_congelada": round(f1_congelada, 10),
            "f1_anidada": round(f1_anidada, 10),
            "umbrales_anidados": [[round(v, 2) for v in fila] for fila in down["umbrales_anidados"]],
            "cobertura_interna": down["cobertura_interna"],
            "f1_train_05_por_pliegue": down["f1_train_05_por_pliegue"],
            "f1_val_05_por_pliegue": down["f1_val_05_por_pliegue"],
            "brecha_por_pliegue": down["brecha_por_pliegue"],
            "brecha_media": down["brecha_media"],
        },
        "cobertura": cobertura,
        "versiones": versiones_entorno(),
    }
    if not humo:
        meta["embedding"] = guardar_embeddings(semilla, brazo, w, palabras)
    unidad = Unidad(
        semilla=semilla,
        brazo=brazo,
        proba=down["proba"],
        d_pool=d_pool,
        umbrales_anidados=down["umbrales_anidados"],
        meta=meta,
    )
    rutas = escribir_parcial(unidad, humo)
    log(
        f"   unidad {semilla}/{brazo} lista | F1 congelada {f1_congelada:.6f} | "
        f"F1 anidada {f1_anidada:.6f} | {', '.join(rutas)}"
    )
    return unidad


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta P84-04 completo.

    Args:
        argv: argumentos de linea de comandos.

    Returns:
        Codigo de salida 0 si todo termino bien.
    """
    inicio = time.perf_counter()
    argumentos = parsear_argumentos(argv)
    humo = bool(argumentos.humo)
    reanudar = bool(argumentos.reanudar)
    semillas = parsear_semillas(argumentos.semillas)
    if humo:
        semillas = semillas[:1]
        log("MODO HUMO: corpus diminuto, 1 semilla, 2 epocas; sin finales ni CSV")
    log(f"semillas {semillas} | reanudar {reanudar} | humo {humo}")

    device = sembrar_global(SEMILLA_BOOTSTRAP)
    cfg = replace(CONFIG_SGNS, epocas=2) if humo else CONFIG_SGNS
    firma = {
        "hash_corpus": HASH_CORPUS,
        "dim": cfg.dim,
        "ventana": cfg.ventana,
        "min_count": cfg.min_count,
        "negativos": cfg.negativos,
        "submuestreo": cfg.submuestreo,
        "epocas": cfg.epocas,
        "lote": cfg.lote,
        "lr0_por_ejemplo": cfg.lr0_por_ejemplo,
        "humo": humo,
        "version_script": VERSION_SCRIPT,
        "sha256_script": sha256_of(PROJECT_ROOT / CONFIG["script"]),
    }
    CONFIG["tmp_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["parciales_dir"].mkdir(parents=True, exist_ok=True)

    # ---------- Gate 0 ----------
    log("Gate 0: hashes congelados, particiones, contadores e identidad V03")
    datos = cargar_gate0()
    identidad = control_identidad_v03(datos)
    log(
        f"   dev {len(datos.des)} | holdout {datos.evidencia['particiones']['holdout']} usado 0 | "
        f"test usado 0 | pliegues {datos.evidencia['particiones']['pliegues_tamanos']}"
    )
    log(
        f"   identidad V03: max|delta proba| {identidad['max_delta_proba_vs_npz']:g} | "
        f"corrected_cv {identidad['f1_congelado_reconstruido']:.12f} | V03@0.5 "
        f"{identidad['f1_at_05_recalculado']:.12f} | tolerancia {TOL_IDENTIDAD:g} OK"
    )

    # ---------- ancla antes de entrenar el embedding final ----------
    log("Ancla de implementacion (algebra + gradiente + corpus plantado)")
    ancla = ejecutar_ancla()
    if humo:
        log("   modo humo: el ancla no se escribe en reports/p84_04_ancla.json")
    else:
        escribir_json(CONFIG["ancla_json"], ancla)
        log(f"   ancla escrita en {CONFIG['ancla_json'].name} antes de entrenar")
    if not ancla["ok"]:
        log(f"   {ancla['veredicto']}: {ancla['advertencia']}")

    # ---------- corpus y vocabulario ----------
    log("Corpus externo y vocabulario")
    docs = leer_tokens(CONFIG["corpus"], HUMO_MAX_TOKENS if humo else None)
    n_tokens = sum(len(d) for d in docs)
    vocab, p, cnt = construir_vocabulario(docs, cfg.min_count, cfg.submuestreo)
    ids_por_doc = construir_ids_por_doc(docs, vocab)
    log(
        f"   {len(docs):,} documentos | {n_tokens:,} tokens | vocabulario "
        f"{len(vocab):,} | min_count {cfg.min_count}"
    )
    cobertura_dev = cobertura_vocabulario(datos.textos, vocab)
    log(
        f"   cobertura dev: {100 * cobertura_dev['fraccion_cubierta']:.2f} por ciento | "
        f"filas sin token {cobertura_dev['filas_sin_ningun_token']}"
    )
    _, cdf = tabla_negativos(vocab, cnt)

    # ---------- bucle de unidades ----------
    unidades: dict[tuple[int, str], Unidad] = {}
    parciales: list[str] = []
    incidencias: list[str] = []
    palabras = list(vocab.keys())
    for semilla in semillas:
        log(f"semilla {semilla}: inicio")
        faltantes = [
            brazo
            for brazo in ("S", "M")
            if not (reanudar and cargar_parcial(semilla, brazo, humo, firma) is not None)
        ]
        resultados: dict[str, ResultadoEntrenamiento] = {}
        if faltantes:
            log(f"   entrenando {faltantes} con flujo pareado (semilla {semilla})")
            resultados = entrenar_modelos(
                [(brazo, AGREGACIONES[brazo]) for brazo in faltantes],
                ids_por_doc,
                p,
                cdf,
                semilla,
                cfg,
                device,
            )
        for brazo in BRAZOS:
            if reanudar:
                cargada = cargar_parcial(semilla, brazo, humo, firma)
                if cargada is not None:
                    unidades[(semilla, brazo)] = cargada
                    log(f"   reanudar: {semilla}/{brazo} cargado de parcial ({cargada.meta['npz']})")
                    continue
            if brazo == "R":
                w = generar_aleatoria(semilla, len(vocab), cfg.dim)
                meta_entrenamiento: dict[str, Any] | None = None
                log(f"   R: matriz aleatoria congelada {w.shape} (no entrena)")
            else:
                resultado = resultados[brazo]
                w = resultado.w_entrada
                meta_entrenamiento = {
                    "agregacion": resultado.agregacion,
                    "loss_por_epoca": resultado.loss_por_epoca,
                    "grad_norma_media_por_epoca": resultado.grad_media_por_epoca,
                    "grad_norma_max_por_epoca": resultado.grad_max_por_epoca,
                    "lr_inicial": resultado.lr_inicial,
                    "lr_final": resultado.lr_final,
                    "pasos": resultado.pasos,
                    "segundos": resultado.segundos,
                    "hash_pares_epoca1": resultado.hash_pares_epoca1,
                    "hash_negativos_lote1": resultado.hash_negativos_lote1,
                    "nota": (
                        "loss y normas de gradiente son diagnosticos del brazo; loss_suma y "
                        "loss_media no son comparables entre si"
                    ),
                }
                log(
                    f"   {brazo}: loss {resultado.loss_por_epoca} | grad media "
                    f"{resultado.grad_media_por_epoca}"
                )
            unidad = procesar_unidad(
                semilla,
                brazo,
                w,
                meta_entrenamiento,
                datos,
                vocab,
                palabras,
                humo,
                incidencias,
                firma,
            )
            unidades[(semilla, brazo)] = unidad
            parciales.extend(
                [unidad.meta["npz"], str(rutas_parcial(semilla, brazo, humo)[0].relative_to(PROJECT_ROOT))]
            )

    flujo = verificar_flujo_pareado(unidades, semillas)
    log(f"   flujo S/M verificado por semilla: {flujo}")

    # ---------- lecturas y comparaciones ----------
    proba_promedio = {
        brazo: np.mean(
            np.stack([unidades[(semilla, brazo)].proba for semilla in semillas]), axis=0
        )
        for brazo in BRAZOS
    }
    umbrales_anidados_promedio = {
        brazo: np.median(
            np.stack([unidades[(semilla, brazo)].umbrales_anidados for semilla in semillas]),
            axis=0,
        )
        for brazo in BRAZOS
    }
    pred_congelada = {
        brazo: (proba_promedio[brazo] >= UMBRALES_CONGELADOS[None, :]).astype(np.int64)
        for brazo in BRAZOS
    }
    pred_anidada = {
        brazo: prediccion_anidada(
            proba_promedio[brazo], umbrales_anidados_promedio[brazo], datos.folde
        )
        for brazo in BRAZOS
    }
    pred_umbral_05 = {
        brazo: (proba_promedio[brazo] >= 0.5).astype(np.int64) for brazo in BRAZOS
    }
    lecturas = {
        brazo: {
            "congelada": metricas_de_lectura(datos.y, pred_congelada[brazo]),
            "umbral_0_5": metricas_de_lectura(datos.y, pred_umbral_05[brazo]),
            "anidada": metricas_de_lectura(datos.y, pred_anidada[brazo]),
        }
        for brazo in BRAZOS
    }
    f1_congelada_promedio = {
        brazo: float(metrica_macro(datos.y, pred_congelada[brazo])) for brazo in BRAZOS
    }
    f1_anidada_promedio = {
        brazo: float(metrica_macro(datos.y, pred_anidada[brazo])) for brazo in BRAZOS
    }
    f1_congelada_por_semilla_brazo = {
        brazo: {
            str(semilla): float(
                metrica_macro(
                    datos.y,
                    (
                        unidades[(semilla, brazo)].proba >= UMBRALES_CONGELADOS[None, :]
                    ).astype(np.int64),
                )
            )
            for semilla in semillas
        }
        for brazo in BRAZOS
    }
    comparaciones: dict[str, Any] = {}
    for nombre, candidato, referencia in (
        ("S_vs_M", pred_congelada["S"], pred_congelada["M"]),
        ("S_vs_R", pred_congelada["S"], pred_congelada["R"]),
        ("S_vs_V03", pred_congelada["S"], datos.pred_v03),
        ("M_vs_V03", pred_congelada["M"], datos.pred_v03),
        ("R_vs_V03", pred_congelada["R"], datos.pred_v03),
    ):
        comparacion = comparar_predicciones(datos, candidato, referencia)
        comparacion["bootstrap"] = bootstrap_diferencia_pareada(
            datos.y,
            candidato,
            referencia,
            repeticiones=REPETICIONES_BOOTSTRAP,
            semilla=SEMILLA_BOOTSTRAP,
        )
        comparaciones[nombre] = comparacion
    comparaciones["S_vs_M"]["por_semilla"] = comparar_por_semilla(
        semillas,
        {s: (unidades[(s, "S")].proba >= UMBRALES_CONGELADOS[None, :]).astype(np.int64) for s in semillas},
        {s: (unidades[(s, "M")].proba >= UMBRALES_CONGELADOS[None, :]).astype(np.int64) for s in semillas},
        datos,
    )
    comparaciones["S_vs_R"]["por_semilla"] = comparar_por_semilla(
        semillas,
        {s: (unidades[(s, "S")].proba >= UMBRALES_CONGELADOS[None, :]).astype(np.int64) for s in semillas},
        {s: (unidades[(s, "R")].proba >= UMBRALES_CONGELADOS[None, :]).astype(np.int64) for s in semillas},
        datos,
    )
    ensemble_gain = {
        brazo: bootstrap_ensemble_gain(
            datos.y,
            proba_promedio[brazo],
            datos.p_v03,
            repeticiones=REPETICIONES_BOOTSTRAP,
            semilla=SEMILLA_BOOTSTRAP,
        )
        for brazo in BRAZOS
    }
    errores = {
        "S_vs_V03": matriz_errores(datos.y, pred_congelada["S"], datos.pred_v03),
        "S_vs_M": matriz_errores(datos.y, pred_congelada["S"], pred_congelada["M"]),
        "S_vs_R": matriz_errores(datos.y, pred_congelada["S"], pred_congelada["R"]),
    }
    brecha_S = float(
        np.mean([unidades[(s, "S")].meta["downstream"]["brecha_media"] for s in semillas])
    )
    brecha_M = float(
        np.mean([unidades[(s, "M")].meta["downstream"]["brecha_media"] for s in semillas])
    )
    brecha_R = float(
        np.mean([unidades[(s, "R")].meta["downstream"]["brecha_media"] for s in semillas])
    )
    brecha_V03 = float(identidad["brecha_train_validacion_v03_media"])
    brecha = {
        "brecha_S": round(brecha_S, 10),
        "brecha_M": round(brecha_M, 10),
        "brecha_R": round(brecha_R, 10),
        "brecha_V03": round(brecha_V03, 10),
        "max_delta_permitido": GATE_BRECHA_MAX,
        "brecha_ok": bool(brecha_S - brecha_V03 <= GATE_BRECHA_MAX),
        "capacidad": {
            "candidato": (
                "LR de 3 x (200 + 1) parametros por pliegue sobre bloque denso de 200; el "
                "embedding externo (V x 200 x 2) se entrena fuera del harness sobre corpus "
                "externo no etiquetado (D-030)"
            ),
            "control_v03": "LR de 3 x (dim TF-IDF + 1) por pliegue sobre TF-IDF word+char",
            "umbral_efectivo": (
                "0.005: el clasificador downstream no aumenta su capacidad de forma material "
                "frente a V03 (menos parametros que el control); se aplica la regla de brecha "
                "+0.05 de D-057"
            ),
        },
    }
    gate = evaluar_gate(comparaciones, ensemble_gain["S"], brecha, bool(ancla["ok"]))
    log(f"GATE: {gate['decision']}")
    for motivo in gate["motivos"]:
        log(f"   motivo: {motivo}")

    # ---------- downstream secundario (solo si S muestra senal sobre M) ----------
    corrida_completa = (not humo) and set(semillas) == set(SEMILLAS_DEFECTO)
    senal_s_sobre_m = bool(comparaciones["S_vs_M"]["delta_global"] > 0)
    secundario: dict[str, Any] = {"ejecutado": False, "etiqueta": "EXPLORATORIO"}
    if senal_s_sobre_m:
        d_s_promedio = np.mean(
            np.stack([unidades[(s, "S")].d_pool for s in semillas]), axis=0
        ).astype(np.float32)
        proba_fusion = cv_fusion_tfidf(datos.textos, d_s_promedio, datos.y, datos.pliegues, 1.0)
        pred_fusion = (proba_fusion >= UMBRALES_CONGELADOS[None, :]).astype(np.int64)
        f1_fusion = float(metrica_macro(datos.y, pred_fusion))
        secundario = {
            "ejecutado": True,
            "etiqueta": "EXPLORATORIO",
            "motivo": "S muestra senal sobre M (delta global > 0); fusion de familia cerrada",
            "fusion_tfidf_escala_1_0": {
                "f1_congelada": round(f1_fusion, 10),
                "delta_vs_v03": round(f1_fusion - V03_CCV, 10),
                "delta_vs_S": round(f1_fusion - f1_congelada_promedio["S"], 10),
                "bootstrap_vs_v03": bootstrap_diferencia_pareada(
                    datos.y,
                    pred_fusion,
                    datos.pred_v03,
                    repeticiones=REPETICIONES_BOOTSTRAP,
                    semilla=SEMILLA_BOOTSTRAP,
                ),
                "bootstrap_vs_S": bootstrap_diferencia_pareada(
                    datos.y,
                    pred_fusion,
                    pred_congelada["S"],
                    repeticiones=REPETICIONES_BOOTSTRAP,
                    semilla=SEMILLA_BOOTSTRAP,
                ),
                "escala": 1.0,
                "nota": "media de probabilidades S+V03: ver ensemble_gain de S",
            },
        }
    else:
        secundario["motivo"] = "S no muestra senal sobre M (delta global <= 0)"

    # ---------- salidas finales ----------
    tiempo_total = round(time.perf_counter() - inicio, 2)
    carga: dict[str, Any] = {
        "experiment_id": "P84-04",
        "script": str(CONFIG["script"]),
        "fecha": str(CONFIG["fecha"]),
        "modo_humo": humo,
        "corrida_completa": corrida_completa,
        "seleccion": {"semillas": list(semillas)},
        "preregistro": {
            "referencia": "docs/p84_spec.md seccion 5 (EXP-04) + docs/p84_preregistro.md",
            "criterio_cierre": (
                "gate: S > M y S > R (delta > 0, >= 4/5 pliegues, IC > 0) y ademas S vs V03 "
                "con +0.005/4/5/IC > 0 o ensemble_gain >= 0.0015 con IC > 0; si no, cerrar la "
                "linea con el resultado verificado"
            ),
            "gate": {
                "delta_minimo_S_vs_V03": GATE_DELTA,
                "pliegues_minimos": GATE_PLIEGUES,
                "ensemble_gain_minimo": GATE_ENSEMBLE_GAIN,
                "ic95_excluye_cero": True,
                "lectura_primaria": "umbrales congelados 0.44/0.51/0.46 (transferencia)",
                "lectura_secundaria": "umbral anidado en particiones internas",
                "ancla_obligatoria": True,
            },
        },
        "config": {
            "corpus": str(CONFIG["corpus"]),
            "hash_corpus": HASH_CORPUS,
            "bin_fasttext": str(CONFIG["bin_fasttext"]),
            "hash_fasttext": HASH_FASTTEXT,
            "sgns": {
                "dim": cfg.dim,
                "ventana": cfg.ventana,
                "min_count": cfg.min_count,
                "negativos": cfg.negativos,
                "submuestreo": cfg.submuestreo,
                "epocas": cfg.epocas,
                "lote": cfg.lote,
                "lr_inicial": cfg.lr0_por_ejemplo * cfg.lote,
                "lr_minimo": cfg.lr_minimo,
            },
            "brazos": {brazo: AGREGACIONES[brazo] for brazo in BRAZOS},
            "semillas": list(semillas),
            "rejilla_umbrales": [float(REJILLA_UMBRALES[0]), float(REJILLA_UMBRALES[-1]), 0.01],
            "bootstrap": {"repeticiones": REPETICIONES_BOOTSTRAP, "semilla": SEMILLA_BOOTSTRAP},
            "particiones_internas": (
                "partir_interna 80/20 por grupos; semillas 20261101 + 100*pliegue + 7*j, j=0..2"
            ),
        },
        "gate0": {"evidencia": datos.evidencia, "identidad_v03": identidad},
        "ancla": ancla,
        "corpus": {
            "ruta": str(CONFIG["corpus"].relative_to(PROJECT_ROOT)),
            "hash": HASH_CORPUS,
            "documentos": len(docs),
            "tokens": n_tokens,
            "vocabulario": len(vocab),
            "cobertura_dev": cobertura_dev,
            "external_only": True,
        },
        "flujo_pareado_S_M": flujo,
        "brazos": {},
        "downstream": {
            "lectura_primaria": "congelada 0.44/0.51/0.46",
            "lecturas": lecturas,
            "f1_congelada_promedio": {b: round(v, 10) for b, v in f1_congelada_promedio.items()},
            "f1_anidada_promedio": {b: round(v, 10) for b, v in f1_anidada_promedio.items()},
            "f1_congelada_por_semilla_brazo": f1_congelada_por_semilla_brazo,
            "umbrales_anidados_promedio": {
                b: [[round(v, 2) for v in fila] for fila in umbrales_anidados_promedio[b]]
                for b in BRAZOS
            },
            "comparaciones": comparaciones,
            "ensemble_gain": ensemble_gain,
            "matriz_errores": errores,
        },
        "d057": brecha,
        "secundario": secundario,
        "gate": gate,
        "parciales": parciales,
        "incidencias": incidencias,
        "tiempo_total_segundos": tiempo_total,
        "versiones": versiones_entorno(),
        "nota_lectura_s_vs_m": (
            "S vs M confunde la correccion del objetivo con la magnitud del gradiente: con el "
            "mismo lr, el termino negativo de S escala por K. No se comparan losses entre "
            "brazos; solo metricas downstream."
        ),
    }
    for brazo in BRAZOS:
        carga["brazos"][brazo] = {
            "agregacion": AGREGACIONES[brazo],
            "por_semilla": {
                str(semilla): {
                    "entrenamiento": unidades[(semilla, brazo)].meta.get("entrenamiento"),
                    "downstream": unidades[(semilla, brazo)].meta.get("downstream"),
                    "embedding": unidades[(semilla, brazo)].meta.get("embedding"),
                }
                for semilla in semillas
            },
        }

    if humo:
        ruta_json = CONFIG["parciales_dir"] / "p84_04_humo_informe.json"
        ruta_md = CONFIG["parciales_dir"] / "p84_04_humo_informe.md"
        escribir_json(ruta_json, carga)
        escribir_texto(ruta_md, construir_informe_md(carga))
        log("modo humo: sin CSV, npz final ni informes finales; ver reports/p84_04_parciales/")
    elif corrida_completa:
        ruta_npz = guardar_oof_npz(unidades, semillas, proba_promedio, umbrales_anidados_promedio, datos)
        carga["oof_npz"] = str(ruta_npz.relative_to(PROJECT_ROOT))
        fila = construir_fila_experiments(carga, tiempo_total)
        carga["experiments_csv"] = upsert_experiments_csv(fila)
        escribir_json(CONFIG["informe_json"], carga)
        escribir_texto(CONFIG["informe_md"], construir_informe_md(carga))
        log(f"informes: {CONFIG['informe_json'].name} y {CONFIG['informe_md'].name}")
    else:
        ruta_json = CONFIG["parciales_dir"] / "p84_04_informe_parcial.json"
        ruta_md = CONFIG["parciales_dir"] / "p84_04_informe_parcial.md"
        escribir_json(ruta_json, carga)
        escribir_texto(ruta_md, construir_informe_md(carga))
        log(
            "AVISO: corrida parcial: sin npz OOF ni experiments.csv; informe etiquetado en "
            f"{ruta_json.name} y {ruta_md.name}"
        )

    log("=" * 78)
    log(f"P84-04 {'HUMO' if humo else 'OK'} | {tiempo_total:.1f}s | device {carga['versiones']['device']}")
    log(
        f"   F1 congelada promedio: S {f1_congelada_promedio['S']:.6f} | "
        f"M {f1_congelada_promedio['M']:.6f} | R {f1_congelada_promedio['R']:.6f} | "
        f"V03 {V03_CCV:.6f}"
    )
    log(
        f"   S vs M delta {comparaciones['S_vs_M']['delta_global']:+.6f} "
        f"({comparaciones['S_vs_M']['pliegues_mejorados']}/5) | S vs R delta "
        f"{comparaciones['S_vs_R']['delta_global']:+.6f} "
        f"({comparaciones['S_vs_R']['pliegues_mejorados']}/5) | S vs V03 delta "
        f"{comparaciones['S_vs_V03']['delta_global']:+.6f} "
        f"({comparaciones['S_vs_V03']['pliegues_mejorados']}/5)"
    )
    log(
        f"   ensemble_gain S {ensemble_gain['S']['ensemble_gain']:+.6f} | IC95 "
        f"[{ensemble_gain['S']['ic95'][0]:+.6f}, {ensemble_gain['S']['ic95'][1]:+.6f}] | "
        f"ancla {ancla['veredicto']}"
    )
    log(f"   decision: {gate['decision']}")
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
