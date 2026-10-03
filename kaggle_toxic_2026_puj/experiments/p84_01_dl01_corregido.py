"""p84_01_dl01_corregido.py
Fecha: 2026-10-02
Descripcion: EXP-01 (P84-01): re-medicion corregida de DL-01 (MLP sobre TF-IDF union ->
TruncatedSVD(300) -> 256-128) sobre las 1483 filas de desarrollo, sin contaminacion de
pool ni de early stopping. La epoca E* se elige en particiones internas agrupadas y el
refit se hace sobre el outer_train completo durante exactamente E* epocas; outer_val se
predice UNA vez y las tres lecturas de umbral reutilizan la misma matriz.
Estado del arte aplicado: protocolo anidado de P75 (E* = mediana de 3 splits internos),
arquitectura y bucle de experiments/11_dl_desde_cero.py, identidad V03 verificable de
P71-10, freno D-057 y preregistro docs/p84_preregistro.md (Enmienda 1).

Especificacion: docs/p84_spec.md seccion 2 + seccion 1 (convenciones comunes).
Preregistro: docs/p84_preregistro.md (Gate 0 y Enmienda 1).

Reglas duras de esta corrida:
- Gate 0: hashes congelados verificados; dev 1483 / holdout 0 / test 0; identidad V03
  reconstruida por pliegue con tolerancia max|delta proba| <= 1e-12 (STOP si falla).
- Nunca se abre data/raw/toxic_test.csv. Nunca se evalua ni entrena con el holdout.
- outer_val se predice una vez por (semilla, pliegue); el contador aborta si se repite.
- Determinismo: fijar_semillas(semilla + j) por split interno y fijar_semillas(semilla)
  en el refit; pre-vuelo de doble pasada (semilla 42, pliegue 1) con tolerancia 1e-6 y
  fallback a CPU; guardia de evaluacion doble con torch.equal.
- AMP False: sin autocast ni GradScaler (evidencia medida en ROCm 7.2, ver 11_dl).
- Freno D-057: umbral efectivo +0.010 (aumento de capacidad) y brecha candidato-control
  <= +0.05; la brecha se mide en este mismo harness para V03 y para el MLP.

Uso:
    /home/carlos/venv/bin/python experiments/p84_01_dl01_corregido.py
    /home/carlos/venv/bin/python experiments/p84_01_dl01_corregido.py --humo
    /home/carlos/venv/bin/python experiments/p84_01_dl01_corregido.py --cpu
    /home/carlos/venv/bin/python experiments/p84_01_dl01_corregido.py --semillas 42,43,44 --pliegues 1,2,3,4,5
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import sklearn
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch import nn

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

# ===== SECCION 0: constantes congeladas, configuracion y cabeceras =====

HASH_TRAIN: Final[str] = "3d40e7345b588860cf4744914be88ddc094e4f03591516c8176ab40b7f76d42f"
HASH_OOF_V03: Final[str] = "559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab"
HASH_GRUPOS: Final[str] = "afabf9ae1c295ea8f203120bf28578cb4bdb4e8f16b89178d820baeaac62e622"

V03_CCV: Final[float] = 0.7772114525165721
V03_F1_AT_05: Final[float] = 0.771778476256122
UMBRALES_CONGELADOS: Final[np.ndarray] = np.array([0.44, 0.51, 0.46], dtype=np.float64)

TOL_IDENTIDAD: Final[float] = 1e-12
TOL_DETERMINISMO: Final[float] = 1e-6
SEMILLA_BOOTSTRAP: Final[int] = 20260926
REPETICIONES_BOOTSTRAP: Final[int] = 10_000

SEMILLA_INTERNA_BASE: Final[int] = 20261101
N_INTERNOS: Final[int] = 3
FRACCION_INTERNA: Final[float] = 0.8
MAX_INTENTOS_INTERNA: Final[int] = 20
TOL_PREVALENCIA: Final[float] = 0.07

GATE_DELTA: Final[float] = 0.005
GATE_DELTA_EFECTIVO: Final[float] = 0.010
GATE_PLIEGUES: Final[int] = 4
D057_BRECHA_MAXIMA: Final[float] = 0.05
N_PARAMETROS_MLP: Final[int] = 110_339

REJILLA_UMBRALES: Final[np.ndarray] = np.round(np.arange(20, 81, dtype=np.float64) / 100.0, 2)

FILAS_DEV: Final[int] = 1483
FILAS_HOLDOUT: Final[int] = 248
FILAS_TRAIN: Final[int] = 1731

# Nota: data/raw/toxic_test.csv no se referencia en ninguna ruta de este script.
CONFIG: Final[dict[str, Any]] = {
    "train_path": PROJECT_ROOT / "data" / "raw" / "toxic_train.csv",
    "grupos_path": PROJECT_ROOT / "data" / "processed" / "grupos_near_duplicados.csv",
    "oof_v03_path": PROJECT_ROOT / "features" / "oof_v03_final.npz",
    "experiments_csv": PROJECT_ROOT / "experiments.csv",
    "gpu_baseline_json": PROJECT_ROOT / "config" / "gpu_baseline.json",
    "gpu_preflight_log": PROJECT_ROOT / "experiments" / "gpu_preflight.log",
    "parciales_dir": PROJECT_ROOT / "reports" / "p84_01_parciales",
    "informe_json": PROJECT_ROOT / "reports" / "p84_01_dl01.json",
    "informe_md": PROJECT_ROOT / "reports" / "p84_01_dl01.md",
    "oof_npz": PROJECT_ROOT / "features" / "p84_01_oof_mlp.npz",
    "modelos_dir": PROJECT_ROOT / "modelos" / "p84",
    "fecha": "2026-10-02",
    "script": "experiments/p84_01_dl01_corregido.py",
    "competencia": "KAGGLE_TOXIC_2026_PUJ",
    "semillas_por_defecto": (42, 43, 44),
    "pliegues_por_defecto": (1, 2, 3, 4, 5),
    "semilla_primaria": 42,
    "umbral_primario": 0.5,
    "svd_componentes": 300,
    "svd_semilla": 42,
    "mlp_ocultas": (256, 128),
    "dropout": 0.3,
    "batch_size": 32,
    "max_epocas": 30,
    "paciencia": 5,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "clip_grad_norm": 1.0,
    "usar_amp": False,
    "vectorizador_rep_05": (
        "TF-IDF word (1,2) min_df=2 sublinear + char_wb (3,5) min_df=2 sublinear, "
        "lowercase, sin strip_accents"
    ),
    "procedencia_arquitectura": "experiments/11_dl_desde_cero.py (MLPMultiTarea)",
    "procedencia_particion": "experiments/p73_02_textcnn_vs_random.py (partir_interna)",
    "procedencia_identidad": "experiments/p71_10_downstream_corregido.py (reconstruccion V03)",
    "amp_evidencia": (
        "BiLSTM 3.42 s sin AMP vs 7.48 s con AMP; CharCNN 0.41 s vs 1.73 s "
        "(ROCm 7.2, RX 9070 XT); sin autocast ni GradScaler."
    ),
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


def log(mensaje: str) -> None:
    """Imprime una linea de progreso con el prefijo de la fase.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P84-01] {mensaje}", flush=True)


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


def leer_ruta(datos: Any, ruta: str) -> Any:
    """Lee una ruta anidada del informe y falla con un mensaje claro si falta.

    Args:
        datos: estructura raiz (informe JSON de la corrida).
        ruta: claves separadas por punto, p. ej. "gate0.evidencia.contadores".

    Returns:
        Valor almacenado en la ruta.

    Raises:
        SystemExit: si algun tramo de la ruta no existe o no es un objeto.
    """
    actual: Any = datos
    for tramo in ruta.split("."):
        if not isinstance(actual, dict) or tramo not in actual:
            raise SystemExit(f"STOP: falta la ruta '{ruta}' en el informe (tramo '{tramo}')")
        actual = actual[tramo]
    return actual


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


def fijar_semillas(semilla: int) -> None:
    """Fija las semillas y el modo de determinismo del proyecto.

    Copia el patron de ``experiments/11_dl_desde_cero.py``: se fija
    ``torch.use_deterministic_algorithms(False)`` de forma explicita porque la
    bandera en True provoco un coredump de GPU en este stack (ROCm 7.2).

    Args:
        semilla: semilla para random, numpy y torch.
    """
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def nombre_dispositivo(dispositivo: torch.device) -> str:
    """Devuelve una etiqueta legible del dispositivo.

    Args:
        dispositivo: dispositivo de torch.

    Returns:
        Cadena descriptiva.
    """
    if dispositivo.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return "cpu"


def resolver_dispositivo(usar_cpu: bool) -> torch.device:
    """Resuelve el dispositivo de computo.

    Args:
        usar_cpu: si es True se fuerza CPU sin consultar CUDA.

    Returns:
        Dispositivo de torch.
    """
    if usar_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def versiones_entorno() -> dict[str, str]:
    """Recoge las versiones relevantes del entorno.

    Returns:
        Diccionario con python, numpy, sklearn, torch y pandas.
    """
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "pandas": pd.__version__,
    }


# ===== SECCION 1: pre-vuelo de GPU (D-008) =====


def muestra_gpu() -> dict[str, Any]:
    """Toma una muestra del estado de la GPU con rocm-smi.

    Returns:
        Diccionario con uso_pct, vram_bytes y pids_kfd.

    Raises:
        RuntimeError: si rocm-smi falla o no devuelve los campos esperados.
    """
    proceso = subprocess.run(
        ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showpids"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    texto = proceso.stdout + proceso.stderr
    usos = [int(valor) for valor in re.findall(r"GPU use \(%\): (\d+)", texto)]
    vrams = [int(valor) for valor in re.findall(r"VRAM Total Used Memory \(B\): (\d+)", texto)]
    if not usos or not vrams:
        raise RuntimeError("rocm-smi no devolvio uso o VRAM; revisa la instalacion")
    seccion = texto.split("KFD Processes", maxsplit=1)[-1]
    if "No KFD PIDs currently running" in seccion:
        pids: list[int] = []
    else:
        pids = [
            int(linea.split()[0])
            for linea in seccion.splitlines()
            if linea.strip() and linea.strip()[0].isdigit()
        ]
    return {"uso_pct": max(usos), "vram_bytes": max(vrams), "pids_kfd": pids}


def registrar_preflight(evidencia: dict[str, Any], destino: Path) -> Path:
    """Anade la evidencia del pre-vuelo al log compartido de GPU.

    Args:
        evidencia: resultado de :func:`verificar_gpu_libre`.
        destino: ruta del log.

    Returns:
        Ruta del log actualizado.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    marca = time.strftime("%Y-%m-%d %H:%M:%S")
    lineas = [
        f"=== {marca} | {evidencia['script']} ===",
        f"comando: {evidencia['comando']} (5 muestras, 2 s entre muestras)",
        f"baseline VRAM: {evidencia['baseline_vram_bytes']} B",
        f"pid propio excluido del criterio de PID ajeno: {evidencia['pid_propio']}",
    ]
    for indice, muestra in enumerate(evidencia["muestras"], start=1):
        lineas.append(
            f"muestra {indice}: uso {muestra['uso_pct']} % | "
            f"VRAM {muestra['vram_bytes']} B | "
            f"KFD PIDs {muestra['pids_kfd'] or 'ninguno'}"
        )
    lineas.append(f"motivos de bloqueo: {evidencia['motivos_bloqueo'] or 'ninguno'}")
    lineas.append(f"decision: {evidencia['decision'].upper()}")
    lineas.append("")
    with destino.open("a", encoding="ascii") as manejador:
        manejador.write("\n".join(lineas) + "\n")
    return destino


def verificar_gpu_libre(baseline_path: Path, log_path: Path) -> dict[str, Any]:
    """Ejecuta el pre-vuelo de GPU de D-008 (5 muestras, 2 s entre muestras).

    Debe ejecutarse ANTES de inicializar el contexto CUDA de torch: en cuanto se
    llama a ``torch.cuda.is_available()`` el proceso aparece en KFD y se
    auto-bloquearia. El PID propio se excluye del criterio de PID ajeno.

    Args:
        baseline_path: JSON con el baseline de VRAM y los umbrales.
        log_path: log compartido de pre-vuelos.

    Returns:
        Evidencia con la decision libre/ocupada.

    Raises:
        SystemExit: si la GPU esta ocupada o rocm-smi no esta disponible.
    """
    config_gpu = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline = int(config_gpu["vram_usada_en_reposo_bytes"])
    umbrales = config_gpu["umbrales"]
    margen = int(umbrales["vram_sobre_baseline_bytes_bloquea"])
    uso_limite = int(umbrales["gpu_use_pct_sostenido_bloquea"])
    minimo = int(umbrales["min_muestras_sobre_umbral"])
    pid_propio = os.getpid()

    muestras: list[dict[str, Any]] = []
    try:
        for indice in range(5):
            if indice:
                time.sleep(2)
            muestra = muestra_gpu()
            muestras.append(muestra)
            log(
                f"pre-vuelo GPU {indice + 1}/5: uso {muestra['uso_pct']} % | "
                f"VRAM {muestra['vram_bytes'] / 1e9:.2f} GB | "
                f"KFD PIDs {muestra['pids_kfd'] or 'ninguno'}"
            )
    except RuntimeError as error:
        raise SystemExit(f"STOP: pre-vuelo GPU no disponible ({error}); use --cpu") from error

    pids_ajenos = sorted(
        {pid for muestra in muestras for pid in muestra["pids_kfd"] if pid != pid_propio}
    )
    vram_elevada = [m for m in muestras if m["vram_bytes"] > baseline + margen]
    uso_alto = [
        m for m in muestras if m["uso_pct"] >= uso_limite and m["vram_bytes"] > baseline
    ]
    motivos: list[str] = []
    if pids_ajenos:
        motivos.append(f"PID ajeno en KFD: {pids_ajenos}")
    if len(vram_elevada) >= minimo:
        motivos.append(
            f"VRAM sobre baseline+{margen / 1e9:.0f} GB en {len(vram_elevada)}/5 muestras"
        )
    if len(uso_alto) >= minimo:
        motivos.append(f"uso >= {uso_limite} % en {len(uso_alto)}/5 muestras")
    ocupada = bool(motivos)

    evidencia = {
        "fecha": CONFIG["fecha"],
        "script": CONFIG["script"],
        "comando": config_gpu["comando_chequeo"],
        "pid_propio": pid_propio,
        "baseline_vram_bytes": baseline,
        "margen_bloqueo_bytes": margen,
        "uso_limite_pct": uso_limite,
        "muestras": muestras,
        "pids_ajenos": pids_ajenos,
        "motivos_bloqueo": motivos,
        "decision": "ocupada" if ocupada else "libre",
    }
    registrar_preflight(evidencia, log_path)
    if ocupada:
        log("GPU OCUPADA: no se entrena. Evidencia en experiments/gpu_preflight.log")
        raise SystemExit("GPU OCUPADA: " + "; ".join(motivos))
    log("GPU libre: se procede con el entrenamiento")
    return evidencia


# ===== SECCION 2: Gate 0 (datos, particiones, contadores e identidad V03) =====


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
    """Ejecuta Gate 0: hashes, particiones, contadores e identidad del OOF.

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
    if tamanos != [297, 297, 296, 296, 297]:
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
            "filas_cubiertas_una_vez": bool((np.bincount(folde, minlength=6)[1:] == np.array(tamanos)).all()),
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


class RepresentacionTfidf:
    """TF-IDF union REP-05 ajustado siempre con el subconjunto indicado.

    Replica exacta del vectorizador de V03/P71-10: word (1,2) min_df=2 sublinear
    + char_wb (3,5) min_df=2 sublinear, lowercase y sin strip_accents, en ese
    orden de concatenacion.
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


def ajustar_proyeccion(x_train: sparse.csr_matrix) -> tuple[TruncatedSVD, StandardScaler]:
    """Ajusta TruncatedSVD(300, rs=42) y StandardScaler solo con el train.

    Args:
        x_train: matriz TF-IDF de entrenamiento.

    Returns:
        Tupla (svd, scaler) ajustada.

    Raises:
        SystemExit: si la matriz no tiene filas y columnas suficientes.
    """
    componentes = int(CONFIG["svd_componentes"])
    if min(x_train.shape) <= componentes:
        raise SystemExit(
            f"STOP: matriz {x_train.shape} insuficiente para TruncatedSVD({componentes})"
        )
    svd = TruncatedSVD(n_components=componentes, random_state=int(CONFIG["svd_semilla"]))
    svd.fit(x_train)
    scaler = StandardScaler().fit(svd.transform(x_train))
    return svd, scaler


def transformar_proyeccion(
    svd: TruncatedSVD, scaler: StandardScaler, x: sparse.csr_matrix
) -> np.ndarray:
    """Proyecta y escala una matriz TF-IDF con objetos ajustados en train.

    Args:
        svd: SVD ajustado.
        scaler: StandardScaler ajustado.
        x: matriz TF-IDF a transformar.

    Returns:
        Matriz densa float32 ``(n, 300)``.
    """
    return scaler.transform(svd.transform(x)).astype(np.float32)


def reconstruir_v03(
    textos: np.ndarray, y: np.ndarray, pliegues: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruye el OOF de V03 con el pipeline por pliegue.

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


def control_identidad_v03(datos: DatosGate0) -> tuple[dict[str, Any], list[float]]:
    """Verifica que el harness reproduce el OOF canonico de V03.

    Args:
        datos: datos de Gate 0.

    Returns:
        Tupla (evidencia del control, brecha train-validacion de V03 por pliegue).

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

    f1_pliegues_npz = [float(valor) for valor in np.asarray(datos.evidencia["oof_v03"].get("f1_pliegues", []))]
    evidencia = {
        "max_delta_proba_vs_npz": max_delta,
        "tolerancia_proba": TOL_IDENTIDAD,
        "proba_reproducida": bool(max_delta <= TOL_IDENTIDAD),
        "f1_congelado_reconstruido": f1_rec,
        "f1_congelado_esperado": V03_CCV,
        "delta_f1_congelado": abs(f1_rec - V03_CCV),
        "f1_at_05_recalculado": f1_at_05,
        "f1_at_05_esperado": V03_F1_AT_05,
        "delta_f1_at_05": abs(f1_at_05 - V03_F1_AT_05),
        "pred_reconstruida_igual_al_npz": bool(np.array_equal(pred_rec, datos.pred_v03)),
        "f1_por_pliegue_reconstruido": [round(v, 8) for v in f1_pliegues_rec],
        "f1_por_pliegue_npz": f1_pliegues_npz,
        "brecha_train_validacion_v03_por_pliegue": [round(v, 6) for v in brecha_por_pliegue],
        "brecha_train_validacion_v03_media": float(np.mean(brecha_por_pliegue)),
    }
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
    return evidencia, brecha_por_pliegue


def verificar_arquitectura(dim_entrada: int) -> dict[str, Any]:
    """Comprueba que la arquitectura declarada tiene 110339 parametros.

    Args:
        dim_entrada: dimension densa de entrada (300).

    Returns:
        Evidencia con el recuento de parametros.

    Raises:
        SystemExit: si el recuento no coincide con el preregistro.
    """
    modelo = MLPMultiTarea(dim_entrada, CONFIG["mlp_ocultas"], float(CONFIG["dropout"]))
    n_parametros = int(sum(parametro.numel() for parametro in modelo.parameters()))
    if n_parametros != N_PARAMETROS_MLP:
        raise SystemExit(
            f"STOP: la arquitectura tiene {n_parametros} parametros, se esperaban {N_PARAMETROS_MLP}"
        )
    return {
        "dim_entrada": dim_entrada,
        "n_parametros": n_parametros,
        "esperados": N_PARAMETROS_MLP,
        "coincide": True,
    }


# ===== SECCION 3: particion interna, equivalencia y umbrales anidados =====


def _partir_interna_local(
    grupos: np.ndarray, y: np.ndarray, semilla: int
) -> tuple[np.ndarray, int]:
    """Reimplementacion local de partir_interna para verificar equivalencia.

    Args:
        grupos: etiqueta de cluster por fila.
        y: etiquetas.
        semilla: semilla base.

    Returns:
        Mascara booleana de inner_train y el numero de intentos usados.
    """
    unicos = np.unique(grupos)
    prevalencia = y.mean(axis=0)
    mascara = np.zeros(len(grupos), dtype=bool)
    intentos = 0
    for intento in range(MAX_INTENTOS_INTERNA):
        intentos = intento + 1
        rng = np.random.default_rng(semilla + intento)
        permutacion = rng.permutation(len(unicos))
        n_sel = round(FRACCION_INTERNA * len(unicos))
        elegidos = set(unicos[permutacion[:n_sel]].tolist())
        mascara = np.array([grupo in elegidos for grupo in grupos])
        if not mascara.any() or mascara.all():
            continue
        prevalencia_val = y[~mascara].mean(axis=0)
        if float(np.abs(prevalencia - prevalencia_val).max()) <= TOL_PREVALENCIA:
            break
    return mascara, intentos


def partir_interna_equivalente(
    grupos: np.ndarray, y: np.ndarray, semilla: int
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Ejecuta partir_interna de P73 y verifica equivalencia con la copia local.

    Args:
        grupos: etiqueta de cluster por fila.
        y: etiquetas.
        semilla: semilla base.

    Returns:
        Tupla (mascara inner_train, intentos, evidencia de equivalencia).

    Raises:
        SystemExit: si la copia local no reproduce la mascara canonica.
    """
    mascara_p73, intentos_p73 = partir_interna_p73(grupos, y, semilla)
    mascara_local, intentos_local = _partir_interna_local(grupos, y, semilla)
    equivalentes = bool(
        np.array_equal(mascara_p73, mascara_local) and intentos_p73 == intentos_local
    )
    if not equivalentes:
        raise SystemExit(
            f"STOP: partir_interna no es equivalente para la semilla {semilla} "
            f"(intentos p73={intentos_p73}, local={intentos_local})"
        )
    return mascara_p73, intentos_p73, {
        "semilla": int(semilla),
        "intentos_p73": int(intentos_p73),
        "intentos_local": int(intentos_local),
        "equivalentes": True,
    }


def evaluar_prevalencia_interna(y: np.ndarray, mascara: np.ndarray) -> dict[str, Any]:
    """Comprueba la tolerancia de prevalencia marginal del holdout interno.

    Args:
        y: etiquetas del pliegue externo.
        mascara: mascara booleana de inner_train.

    Returns:
        Evidencia con el maximo delta de prevalencia por etiqueta.
    """
    prevalencia_global = y.mean(axis=0)
    prevalencia_val = y[~mascara].mean(axis=0)
    max_delta = float(np.abs(prevalencia_global - prevalencia_val).max())
    return {
        "max_delta_prevalencia": max_delta,
        "tolerancia": TOL_PREVALENCIA,
        "cumple": bool(max_delta <= TOL_PREVALENCIA),
    }


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


def elegir_umbrales_anidados(proba: np.ndarray, y: np.ndarray) -> list[float]:
    """Elige un umbral por etiqueta con las probabilidades internas agregadas.

    La rejilla declarada es 0.20 a 0.80 paso 0.01 y el criterio es la F1 maxima
    por etiqueta; en caso de empate se toma el primer maximo (menor umbral).

    Args:
        proba: probabilidades internas agregadas (n, 3).
        y: etiquetas internas (n, 3).

    Returns:
        Lista de 3 umbrales.
    """
    umbrales: list[float] = []
    for j in range(proba.shape[1]):
        valores = [
            f1_binaria(y[:, j], (proba[:, j] >= umbral).astype(np.int64))
            for umbral in REJILLA_UMBRALES
        ]
        indice = int(np.argmax(valores))
        umbrales.append(float(REJILLA_UMBRALES[indice]))
    return umbrales


# ===== SECCION 4: arquitectura MLP y entrenamiento =====


class MLPMultiTarea(nn.Module):
    """Perceptron multicapa sobre la proyeccion densa del TF-IDF.

    Reimplementacion de ``MLPMultiTarea`` de experiments/11_dl_desde_cero.py:
    Linear + ReLU + Dropout entre capas y salida de 3 logits.
    """

    def __init__(self, dim_entrada: int, ocultas: Sequence[int], dropout: float) -> None:
        """Inicializa las capas del MLP.

        Args:
            dim_entrada: dimension de entrada (componentes SVD).
            ocultas: anchos de las capas ocultas.
            dropout: probabilidad de dropout entre capas.
        """
        super().__init__()
        capas: list[nn.Module] = []
        anterior = dim_entrada
        for ancho in ocultas:
            capas += [nn.Linear(anterior, ancho), nn.ReLU(), nn.Dropout(dropout)]
            anterior = ancho
        capas.append(nn.Linear(anterior, len(ETIQUETAS)))
        self.red = nn.Sequential(*capas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calcula los 3 logits de un lote.

        Args:
            x: tensor denso ``(lote, dim_entrada)``.

        Returns:
            Tensor ``(lote, 3)`` de logits.
        """
        return self.red(x)


def peso_positivos(etiquetas: np.ndarray, dispositivo: torch.device) -> torch.Tensor:
    """Calcula pos_weight por etiqueta con el pliegue de entrenamiento.

    Args:
        etiquetas: matriz ``(n, 3)`` de etiquetas del pliegue de entrenamiento.
        dispositivo: dispositivo donde vive el tensor.

    Returns:
        Tensor ``(3,)`` con ``n_negativos / max(n_positivos, 1.0)`` en float32.
    """
    positivos = etiquetas.sum(axis=0).astype(np.float64)
    negativos = etiquetas.shape[0] - positivos
    return torch.tensor(
        negativos / np.maximum(positivos, 1.0), dtype=torch.float32, device=dispositivo
    )


def tensor_entrada(z: np.ndarray, dispositivo: torch.device) -> torch.Tensor:
    """Convierte una matriz densa en tensor float32 del dispositivo.

    Args:
        z: matriz float32.
        dispositivo: dispositivo destino.

    Returns:
        Tensor de torch.
    """
    return torch.as_tensor(np.ascontiguousarray(z), dtype=torch.float32, device=dispositivo)


def tensor_etiquetas(y: np.ndarray, dispositivo: torch.device) -> torch.Tensor:
    """Convierte etiquetas en tensor float32 del dispositivo.

    Args:
        y: etiquetas enteras ``(n, 3)``.
        dispositivo: dispositivo destino.

    Returns:
        Tensor float32.
    """
    return torch.as_tensor(np.ascontiguousarray(y), dtype=torch.float32, device=dispositivo)


def _paso_epoca(
    modelo: nn.Module,
    optimizador: torch.optim.Optimizer,
    criterio: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    clip: float,
) -> list[float]:
    """Ejecuta una epoca completa de entrenamiento por lotes.

    No se usa autocast ni GradScaler: AMP queda en False con la evidencia medida
    del proyecto (ver CONFIG["amp_evidencia"]).

    Args:
        modelo: red en modo train.
        optimizador: optimizador.
        criterio: perdida BCEWithLogitsLoss.
        x: entradas de entrenamiento.
        y: etiquetas float32.
        batch_size: tamano de lote.
        clip: norma maxima de gradiente.

    Returns:
        Perdidas medias por lote.
    """
    modelo.train()
    permutacion = torch.randperm(x.shape[0], device=x.device)
    perdidas: list[float] = []
    for inicio in range(0, x.shape[0], batch_size):
        indices = permutacion[inicio : inicio + batch_size]
        optimizador.zero_grad(set_to_none=True)
        logits = modelo(x[indices])
        perdida = criterio(logits, y[indices])
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), clip)
        optimizador.step()
        perdidas.append(float(perdida.detach()))
    return perdidas


@dataclass
class ResultadoEntrenamiento:
    """Curvas y resultado de un entrenamiento con early stopping.

    Attributes:
        mejor_epoca: epoca del mejor peso restaurado (E_j).
        mejor_f1_val: F1@0.5 de inner_val con los mejores pesos.
        f1_train_best: F1@0.5 de inner_train con los mejores pesos.
        probabilidad_val: probabilidades de inner_val con los mejores pesos.
        epocas_ejecutadas: epocas realmente ejecutadas (incluye paciencia).
        perdidas_train: perdida media de train por epoca.
        perdidas_val: perdida de validacion por epoca.
        f1_val_por_epoca: F1@0.5 de validacion por epoca.
        segundos: tiempo de entrenamiento.
    """

    mejor_epoca: int = 0
    mejor_f1_val: float = 0.0
    f1_train_best: float = 0.0
    probabilidad_val: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    epocas_ejecutadas: int = 0
    perdidas_train: list[float] = field(default_factory=list)
    perdidas_val: list[float] = field(default_factory=list)
    f1_val_por_epoca: list[float] = field(default_factory=list)
    segundos: float = 0.0


def entrenar_con_early_stopping(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    semilla: int,
    dispositivo: torch.device,
    max_epocas: int,
    paciencia: int,
) -> tuple[nn.Module, ResultadoEntrenamiento]:
    """Entrena un MLP con early stopping estricto sobre F1@0.5 de validacion.

    Fija las semillas con ``fijar_semillas(semilla)`` antes de construir el
    modelo, de modo que dos llamadas identicas reproducen la misma trayectoria.
    La parada es ``paciencia`` epocas sin mejora estricta y se restauran los
    pesos de la mejor epoca; E_j es esa mejor epoca, no las ejecutadas.

    Args:
        z_train: matriz densa de inner_train.
        y_train: etiquetas de inner_train.
        z_val: matriz densa de inner_val.
        y_val: etiquetas de inner_val.
        semilla: semilla del split (semilla + j).
        dispositivo: dispositivo de computo.
        max_epocas: maximo de epocas.
        paciencia: epocas sin mejora antes de parar.

    Returns:
        Tupla (modelo con los mejores pesos, resultado del entrenamiento).
    """
    inicio = time.perf_counter()
    fijar_semillas(semilla)
    modelo = MLPMultiTarea(z_train.shape[1], CONFIG["mlp_ocultas"], float(CONFIG["dropout"]))
    modelo.to(dispositivo)
    optimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=float(CONFIG["learning_rate"]),
        weight_decay=float(CONFIG["weight_decay"]),
    )
    criterio = nn.BCEWithLogitsLoss(pos_weight=peso_positivos(y_train, dispositivo))
    x_train = tensor_entrada(z_train, dispositivo)
    t_train = tensor_etiquetas(y_train, dispositivo)
    x_val = tensor_entrada(z_val, dispositivo)
    t_val = tensor_etiquetas(y_val, dispositivo)
    y_val_int = y_val.astype(np.int64)

    resultado = ResultadoEntrenamiento()
    mejor_estado = copy.deepcopy(modelo.state_dict())
    mejor_f1 = -1.0
    epocas_sin_mejora = 0
    batch_size = int(CONFIG["batch_size"])
    clip = float(CONFIG["clip_grad_norm"])

    for epoca in range(1, max_epocas + 1):
        perdidas = _paso_epoca(modelo, optimizador, criterio, x_train, t_train, batch_size, clip)
        modelo.eval()
        with torch.no_grad():
            logits_val = modelo(x_val)
            perdida_val = float(criterio(logits_val, t_val))
            proba_val = torch.sigmoid(logits_val).cpu().numpy().astype(np.float64)
        f1_val = float(metrica_macro(y_val_int, (proba_val >= 0.5).astype(np.int64)))
        resultado.perdidas_train.append(float(np.mean(perdidas)))
        resultado.perdidas_val.append(perdida_val)
        resultado.f1_val_por_epoca.append(f1_val)
        if f1_val > mejor_f1:
            mejor_f1 = f1_val
            resultado.mejor_epoca = epoca
            mejor_estado = copy.deepcopy(modelo.state_dict())
            epocas_sin_mejora = 0
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= paciencia:
                break

    modelo.load_state_dict(mejor_estado)
    modelo.eval()
    with torch.no_grad():
        proba_best = torch.sigmoid(modelo(x_val)).cpu().numpy().astype(np.float64)
        proba_train = torch.sigmoid(modelo(x_train)).cpu().numpy().astype(np.float64)
    resultado.probabilidad_val = proba_best
    resultado.mejor_f1_val = float(
        metrica_macro(y_val_int, (proba_best >= 0.5).astype(np.int64))
    )
    resultado.f1_train_best = float(
        metrica_macro(y_train.astype(np.int64), (proba_train >= 0.5).astype(np.int64))
    )
    resultado.epocas_ejecutadas = len(resultado.f1_val_por_epoca)
    resultado.segundos = round(time.perf_counter() - inicio, 2)
    return modelo, resultado


def entrenar_epocas_fijas(
    z_train: np.ndarray,
    y_train: np.ndarray,
    epocas: int,
    semilla: int,
    dispositivo: torch.device,
) -> tuple[nn.Module, list[float], float]:
    """Entrena un MLP nuevo exactamente ``epocas`` epocas, sin validacion.

    Fija las semillas con ``fijar_semillas(semilla)`` antes de construir el
    modelo. No se usa outer_val en ningun momento de este ajuste.

    Args:
        z_train: matriz densa de outer_train.
        y_train: etiquetas de outer_train.
        epocas: numero exacto de epocas (E*).
        semilla: semilla del refit.
        dispositivo: dispositivo de computo.

    Returns:
        Tupla (modelo entrenado, perdidas medias por epoca, segundos).
    """
    inicio = time.perf_counter()
    fijar_semillas(semilla)
    modelo = MLPMultiTarea(z_train.shape[1], CONFIG["mlp_ocultas"], float(CONFIG["dropout"]))
    modelo.to(dispositivo)
    optimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=float(CONFIG["learning_rate"]),
        weight_decay=float(CONFIG["weight_decay"]),
    )
    criterio = nn.BCEWithLogitsLoss(pos_weight=peso_positivos(y_train, dispositivo))
    x_train = tensor_entrada(z_train, dispositivo)
    t_train = tensor_etiquetas(y_train, dispositivo)
    perdidas: list[float] = []
    for _ in range(1, epocas + 1):
        paso = _paso_epoca(
            modelo,
            optimizador,
            criterio,
            x_train,
            t_train,
            int(CONFIG["batch_size"]),
            float(CONFIG["clip_grad_norm"]),
        )
        perdidas.append(float(np.mean(paso)))
    return modelo, perdidas, round(time.perf_counter() - inicio, 2)


class ContadorPrediccion:
    """Contador de pasadas oficiales y guardias sobre outer_val.

    Attributes:
        oficiales: pasadas que producen la matriz de lectura.
        guardias: pasadas de la verificacion torch.equal.
        lecturas_umbral: lecturas de umbral que reutilizan la matriz oficial.
    """

    def __init__(self) -> None:
        """Inicializa los contadores en cero."""
        self.oficiales = 0
        self.guardias = 0
        self.lecturas_umbral = 0

    def registrar_oficial(self) -> None:
        """Registra la pasada oficial; solo se admite una por combinacion.

        Raises:
            RuntimeError: si ya se registro una pasada oficial.
        """
        if self.oficiales >= 1:
            raise RuntimeError("STOP: segunda pasada oficial de outer_val en la misma combinacion")
        self.oficiales += 1

    def registrar_guardia(self) -> None:
        """Registra la pasada de la guardia de determinismo.

        Raises:
            RuntimeError: si ya se registro una guardia.
        """
        if self.guardias >= 1:
            raise RuntimeError("STOP: segunda guardia de determinismo en la misma combinacion")
        self.guardias += 1

    def registrar_lectura_umbral(self) -> None:
        """Registra una lectura de umbral que reutiliza la matriz oficial."""
        self.lecturas_umbral += 1

    def a_diccionario(self) -> dict[str, Any]:
        """Exporta los contadores con una nota interpretativa.

        Returns:
            Diccionario serializable.
        """
        return {
            "pasadas_oficiales_outer_val": self.oficiales,
            "pasadas_guardia_outer_val": self.guardias,
            "pasadas_forward_outer_val_totales": self.oficiales + self.guardias,
            "matrices_de_lectura": self.oficiales,
            "lecturas_de_umbral": self.lecturas_umbral,
            "nota": (
                "una pasada oficial genera la unica matriz; la guardia torch.equal es una "
                "verificacion adicional que no produce lecturas; las tres lecturas de umbral "
                "reutilizan la matriz oficial"
            ),
        }


def predecir_outer_val(
    modelo: nn.Module, x_val: torch.Tensor, contador: ContadorPrediccion
) -> np.ndarray:
    """Predice outer_val una vez con guardia de determinismo.

    Args:
        modelo: red en modo eval.
        x_val: tensor de outer_val.
        contador: contador de pasadas.

    Returns:
        Probabilidades float64 ``(n, 3)``.

    Raises:
        RuntimeError: si la evaluacion doble no es identica (torch.equal).
    """
    modelo.eval()
    with torch.no_grad():
        contador.registrar_oficial()
        p1 = torch.sigmoid(modelo(x_val))
        contador.registrar_guardia()
        p2 = torch.sigmoid(modelo(x_val))
    if not torch.equal(p1, p2):
        raise RuntimeError("STOP: outer_val no es determinista en modo eval (torch.equal falla)")
    return p1.cpu().numpy().astype(np.float64)


# ===== SECCION 5: una combinacion (pliegue, semilla) de EXP-01 =====


@dataclass
class ResultadoCombinacion:
    """Resultado completo de un (pliegue, semilla).

    Attributes:
        pliegue: numero de pliegue externo (1..5).
        semilla: semilla del modelo.
        e_j: epocas de los 3 splits internos.
        e_estrella: mediana de e_j.
        dispersion_e_j: desviacion tipica poblacional de e_j.
        info_splits: detalle por split interno.
        umbrales_anidados: umbrales por etiqueta elegidos en las internas.
        contador: contadores de prediccion de outer_val.
        f1_por_lectura: F1 del pliegue en las tres lecturas.
        f1_train_05: F1 de outer_train a 0.5 con los pesos finales.
        f1_val_05: F1 de outer_val a 0.5 (lectura primaria).
        brecha: f1_train_05 - f1_val_05.
        segundos: tiempo de la combinacion.
        proba_val: probabilidades de outer_val.
        pred_primaria: prediccion a 0.5.
        pred_anidada: prediccion con umbrales anidados.
        pred_congelada: prediccion con umbrales congelados de V03.
        incidencias: avisos registrados.
        ruta_checkpoint: ruta del checkpoint final (cadena vacia si no aplica).
    """

    pliegue: int
    semilla: int
    e_j: list[int]
    e_estrella: int
    dispersion_e_j: float
    info_splits: list[dict[str, Any]]
    umbrales_anidados: list[float]
    contador: dict[str, Any]
    f1_por_lectura: dict[str, float]
    f1_train_05: float
    f1_val_05: float
    brecha: float
    segundos: float
    proba_val: np.ndarray
    pred_primaria: np.ndarray
    pred_anidada: np.ndarray
    pred_congelada: np.ndarray
    incidencias: list[str]
    ruta_checkpoint: str


def _guardar_checkpoint(
    modelo: nn.Module,
    resultado: ResultadoCombinacion,
    dim_entrada: int,
) -> str:
    """Guarda el state_dict final por (semilla, pliegue) en modelos/p84/.

    Args:
        modelo: modelo del refit.
        resultado: resultado de la combinacion.
        dim_entrada: dimension densa de entrada.

    Returns:
        Ruta del checkpoint.
    """
    directorio: Path = CONFIG["modelos_dir"]
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / f"p84_01_mlp_s{resultado.semilla}_p{resultado.pliegue}.pt"
    carga = {
        "experiment_id": "P84-01",
        "semilla": resultado.semilla,
        "pliegue": resultado.pliegue,
        "E_estrella": resultado.e_estrella,
        "E_j": list(resultado.e_j),
        "dim_entrada": int(dim_entrada),
        "n_parametros": N_PARAMETROS_MLP,
        "umbrales_anidados": list(resultado.umbrales_anidados),
        "arquitectura": "MLP 300-256-128-3 ReLU dropout 0.3",
        "procedencia": str(CONFIG["procedencia_arquitectura"]),
        "state_dict": {
            clave: valor.detach().cpu() for clave, valor in modelo.state_dict().items()
        },
    }
    torch.save(carga, ruta)
    return str(ruta)


def entrenar_combinacion(
    pliegue: int,
    semilla: int,
    datos: DatosGate0,
    dispositivo: torch.device,
    guardar_checkpoint: bool,
) -> ResultadoCombinacion:
    """Ejecuta el protocolo EXP-01 para un (pliegue, semilla).

    Pasos: 3 splits internos con partir_interna (E_j por early stopping), E* =
    mediana, reajuste con outer_train durante exactamente E* epocas y una unica
    prediccion oficial de outer_val; las tres lecturas reutilizan esa matriz.

    Args:
        pliegue: numero de pliegue externo (1..5).
        semilla: semilla del modelo.
        datos: datos de Gate 0.
        dispositivo: dispositivo de computo.
        guardar_checkpoint: si True guarda el state_dict final.

    Returns:
        Resultado de la combinacion.
    """
    inicio = time.perf_counter()
    indice_train, indice_val = datos.pliegues[pliegue - 1]
    a = np.asarray(indice_train)
    b = np.asarray(indice_val)
    textos = datos.textos
    y = datos.y

    info_splits: list[dict[str, Any]] = []
    e_j: list[int] = []
    probas_internas: list[np.ndarray] = []
    etiquetas_internas: list[np.ndarray] = []
    incidencias: list[str] = []

    for j in range(N_INTERNOS):
        semilla_split = SEMILLA_INTERNA_BASE + 100 * pliegue + 7 * j
        mascara, intentos, equivalencia = partir_interna_equivalente(
            datos.grupos_dev[a], y[a], semilla_split
        )
        idx_train = a[mascara]
        idx_val = a[~mascara]
        prevalencia = evaluar_prevalencia_interna(y[a], mascara)
        if not prevalencia["cumple"]:
            mensaje = (
                f"AVISO pliegue {pliegue} semilla {semilla} split {j}: tolerancia de "
                f"prevalencia 0.07 no cumplida tras {intentos} intentos "
                f"(max delta {prevalencia['max_delta_prevalencia']:.4f})"
            )
            incidencias.append(mensaje)
            log(mensaje)

        representacion = RepresentacionTfidf()
        x_train = representacion.ajustar_transformar(textos[idx_train])
        x_val = representacion.transformar(textos[idx_val])
        svd, scaler = ajustar_proyeccion(x_train)
        z_train = transformar_proyeccion(svd, scaler, x_train)
        z_val = transformar_proyeccion(svd, scaler, x_val)

        _, resultado = entrenar_con_early_stopping(
            z_train,
            y[idx_train],
            z_val,
            y[idx_val],
            semilla=semilla + j,
            dispositivo=dispositivo,
            max_epocas=int(CONFIG["max_epocas"]),
            paciencia=int(CONFIG["paciencia"]),
        )
        e_j.append(resultado.mejor_epoca)
        probas_internas.append(resultado.probabilidad_val)
        etiquetas_internas.append(y[idx_val].astype(np.int64))
        info_splits.append(
            {
                "j": j,
                "semilla_split": semilla_split,
                "intentos": int(intentos),
                "equivalencia_particion": equivalencia,
                "prevalencia": prevalencia,
                "n_inner_train": len(idx_train),
                "n_inner_val": len(idx_val),
                "mejor_epoca": int(resultado.mejor_epoca),
                "mejor_f1_val": round(resultado.mejor_f1_val, 6),
                "f1_train_best": round(resultado.f1_train_best, 6),
                "epocas_ejecutadas": int(resultado.epocas_ejecutadas),
                "segundos": resultado.segundos,
            }
        )
        log(
            f"      pliegue {pliegue} semilla {semilla} split {j}: mejor epoca "
            f"{resultado.mejor_epoca} | F1 val {resultado.mejor_f1_val:.4f} | "
            f"epocas {resultado.epocas_ejecutadas} | {resultado.segundos:.1f}s"
        )

    e_estrella = int(np.median(e_j))
    dispersion = float(np.std(e_j, ddof=0))
    umbrales_anidados = elegir_umbrales_anidados(
        np.vstack(probas_internas), np.vstack(etiquetas_internas)
    )

    representacion = RepresentacionTfidf()
    x_train = representacion.ajustar_transformar(textos[a])
    x_val = representacion.transformar(textos[b])
    svd, scaler = ajustar_proyeccion(x_train)
    z_train = transformar_proyeccion(svd, scaler, x_train)
    z_val = transformar_proyeccion(svd, scaler, x_val)

    modelo, _, _segundos_refit = entrenar_epocas_fijas(
        z_train,
        y[a],
        epocas=e_estrella,
        semilla=semilla,
        dispositivo=dispositivo,
    )
    contador = ContadorPrediccion()
    proba_val = predecir_outer_val(modelo, tensor_entrada(z_val, dispositivo), contador)
    with torch.no_grad():
        proba_train = (
            torch.sigmoid(modelo(tensor_entrada(z_train, dispositivo)))
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    pred_primaria = (proba_val >= float(CONFIG["umbral_primario"])).astype(np.int64)
    pred_anidada = (proba_val >= np.asarray(umbrales_anidados)[None, :]).astype(np.int64)
    pred_congelada = (proba_val >= UMBRALES_CONGELADOS[None, :]).astype(np.int64)
    for _ in range(3):
        contador.registrar_lectura_umbral()
    f1_primaria = float(metrica_macro(y[b], pred_primaria))
    f1_anidada = float(metrica_macro(y[b], pred_anidada))
    f1_congelada = float(metrica_macro(y[b], pred_congelada))
    f1_train_05 = float(metrica_macro(y[a], (proba_train >= 0.5).astype(np.int64)))

    resultado = ResultadoCombinacion(
        pliegue=pliegue,
        semilla=semilla,
        e_j=list(e_j),
        e_estrella=e_estrella,
        dispersion_e_j=dispersion,
        info_splits=info_splits,
        umbrales_anidados=[float(v) for v in umbrales_anidados],
        contador=contador.a_diccionario(),
        f1_por_lectura={
            "primaria_umbral_0_5": round(f1_primaria, 8),
            "secundaria_anidada": round(f1_anidada, 8),
            "control_congelado_v03": round(f1_congelada, 8),
        },
        f1_train_05=round(f1_train_05, 8),
        f1_val_05=round(f1_primaria, 8),
        brecha=round(f1_train_05 - f1_primaria, 8),
        segundos=round(time.perf_counter() - inicio, 2),
        proba_val=proba_val,
        pred_primaria=pred_primaria,
        pred_anidada=pred_anidada,
        pred_congelada=pred_congelada,
        incidencias=list(incidencias),
        ruta_checkpoint="",
    )
    resultado.ruta_checkpoint = (
        _guardar_checkpoint(modelo, resultado, int(z_train.shape[1])) if guardar_checkpoint else ""
    )
    log(
        f"   pliegue {pliegue} semilla {semilla}: E* {e_estrella} (E_j {e_j}, sd "
        f"{dispersion:.2f}) | umbrales anidados {resultado.umbrales_anidados} | "
        f"F1@0.5 {f1_primaria:.6f} | anidada {f1_anidada:.6f} | congelada "
        f"{f1_congelada:.6f} | brecha {resultado.brecha:+.4f} | {resultado.segundos:.1f}s"
    )
    return resultado


# ===== SECCION 6: pre-vuelo de determinismo =====


def _doble_pasada_interna(
    datos: DatosGate0, dispositivo: torch.device
) -> dict[str, Any]:
    """Entrena dos veces el split interno (pliegue 1, j=0) y compara.

    Args:
        datos: datos de Gate 0.
        dispositivo: dispositivo donde se prueba.

    Returns:
        Evidencia con el maximo delta absoluto entre las dos probabilidades.
    """
    indice_train, _ = datos.pliegues[0]
    a = np.asarray(indice_train)
    semilla_split = SEMILLA_INTERNA_BASE + 100 * 1 + 0
    mascara, intentos, _ = partir_interna_equivalente(datos.grupos_dev[a], datos.y[a], semilla_split)
    idx_train = a[mascara]
    idx_val = a[~mascara]
    representacion = RepresentacionTfidf()
    x_train = representacion.ajustar_transformar(datos.textos[idx_train])
    x_val = representacion.transformar(datos.textos[idx_val])
    svd, scaler = ajustar_proyeccion(x_train)
    z_train = transformar_proyeccion(svd, scaler, x_train)
    z_val = transformar_proyeccion(svd, scaler, x_val)

    probabilidades: list[np.ndarray] = []
    epocas: list[int] = []
    for _ in range(2):
        _, resultado = entrenar_con_early_stopping(
            z_train,
            datos.y[idx_train],
            z_val,
            datos.y[idx_val],
            semilla=42,
            dispositivo=dispositivo,
            max_epocas=int(CONFIG["max_epocas"]),
            paciencia=int(CONFIG["paciencia"]),
        )
        probabilidades.append(resultado.probabilidad_val)
        epocas.append(int(resultado.mejor_epoca))
    max_delta = float(np.max(np.abs(probabilidades[0] - probabilidades[1])))
    return {
        "pliegue": 1,
        "semilla": 42,
        "semilla_split": semilla_split,
        "intentos": int(intentos),
        "n_inner_train": len(idx_train),
        "n_inner_val": len(idx_val),
        "mejores_epocas": epocas,
        "max_delta_proba": max_delta,
        "tolerancia": TOL_DETERMINISMO,
        "aprobado": bool(max_delta <= TOL_DETERMINISMO),
    }


def preflight_determinismo(
    datos: DatosGate0, dispositivo: torch.device
) -> tuple[torch.device, dict[str, Any]]:
    """Ejecuta el pre-vuelo de doble pasada (semilla 42, pliegue 1).

    Si la doble pasada falla en GPU se reintenta en CPU; si tambien falla en
    CPU se aborta. Si el fallback funciona, toda la corrida continua en CPU.

    Args:
        datos: datos de Gate 0.
        dispositivo: dispositivo inicial.

    Returns:
        Tupla (dispositivo final, evidencia del pre-vuelo).

    Raises:
        SystemExit: si la doble pasada no es determinista ni en CPU.
    """
    evidencia = _doble_pasada_interna(datos, dispositivo)
    evidencia["dispositivo_inicial"] = nombre_dispositivo(dispositivo)
    evidencia["fallback_cpu"] = False
    if evidencia["aprobado"]:
        log(
            f"   pre-vuelo determinismo: max|delta p| {evidencia['max_delta_proba']:.3e} "
            f"<= {TOL_DETERMINISMO:g} en {evidencia['dispositivo_inicial']}"
        )
        evidencia["dispositivo_final"] = evidencia["dispositivo_inicial"]
        return dispositivo, evidencia

    log(
        f"   AVISO: doble pasada no determinista en {evidencia['dispositivo_inicial']} "
        f"(max|delta p| {evidencia['max_delta_proba']:.3e}); se reintenta en CPU"
    )
    if dispositivo.type == "cpu":
        raise SystemExit("STOP: la doble pasada no es determinista ni en CPU")
    cpu = torch.device("cpu")
    evidencia_cpu = _doble_pasada_interna(datos, cpu)
    evidencia["cpu"] = evidencia_cpu
    if not evidencia_cpu["aprobado"]:
        raise SystemExit("STOP: la doble pasada no es determinista ni en GPU ni en CPU")
    evidencia["fallback_cpu"] = True
    evidencia["dispositivo_final"] = "cpu"
    log("   fallback a CPU aceptado: la corrida completa continua en CPU")
    return cpu, evidencia


# ===== SECCION 7: lecturas, comparaciones y resumen =====


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


def comparar_contra_v03(
    datos: DatosGate0, pred: np.ndarray, pred_v03: np.ndarray | None = None
) -> dict[str, Any]:
    """Compara pareadamente una lectura contra V03 con umbrales congelados.

    Args:
        datos: datos de Gate 0.
        pred: predicciones del candidato.
        pred_v03: predicciones de V03; si es None se usan las congeladas.

    Returns:
        Diccionario con delta global, deltas por pliegue, pliegues mejorados,
        rescate/dano, tasas de positivos y F1 por pliegue.
    """
    referencia = datos.pred_v03 if pred_v03 is None else pred_v03
    f1_global = float(metrica_macro(datos.y, pred))
    f1_v03 = float(metrica_macro(datos.y, referencia))
    deltas: list[float] = []
    f1_pliegues: list[float] = []
    f1_pliegues_v03: list[float] = []
    for _, indice_val in datos.pliegues:
        b = np.asarray(indice_val)
        valor = float(metrica_macro(datos.y[b], pred[b]))
        valor_v03 = float(metrica_macro(datos.y[b], referencia[b]))
        f1_pliegues.append(round(valor, 8))
        f1_pliegues_v03.append(round(valor_v03, 8))
        deltas.append(round(valor - valor_v03, 8))
    ok_v03 = referencia == datos.y
    ok_cand = pred == datos.y
    rescate = int(np.sum(~ok_v03 & ok_cand))
    dano = int(np.sum(ok_v03 & ~ok_cand))
    return {
        "f1_global": round(f1_global, 10),
        "f1_v03_global": round(f1_v03, 10),
        "delta_global": round(f1_global - f1_v03, 10),
        "delta_medio_pliegues": round(float(np.mean(deltas)), 10),
        "delta_por_pliegue": deltas,
        "pliegues_mejorados": int(np.sum(np.array(deltas) > 0)),
        "f1_por_pliegue": f1_pliegues,
        "f1_por_pliegue_v03": f1_pliegues_v03,
        "rescate_celdas": rescate,
        "dano_celdas": dano,
        "neto_celdas": rescate - dano,
        "tasa_positivos_v03": [float(valor) for valor in referencia.mean(axis=0)],
    }


def bootstrap_ensemble_gain(
    y: np.ndarray,
    proba_candidato: np.ndarray,
    proba_v03: np.ndarray,
    repeticiones: int,
    semilla: int,
) -> dict[str, Any]:
    """Calcula ensemble_gain con IC bootstrap pareado por filas.

    Convencion P6/P26: F1 de la media de probabilidades a 0.5 menos la F1 de
    V03 a 0.5 recalculada desde el npz; el IC remuestrea filas y recalcula la
    diferencia de las dos F1 en cada remuestreo.

    Args:
        y: etiquetas (n, 3).
        proba_candidato: probabilidades del candidato.
        proba_v03: probabilidades de V03.
        repeticiones: numero de remuestreos.
        semilla: semilla del generador.

    Returns:
        Diccionario con la referencia recalculada, el gain y su IC95.
    """
    pred_candidato = (proba_candidato >= 0.5).astype(np.int64)
    pred_media = ((proba_candidato + proba_v03) / 2.0 >= 0.5).astype(np.int64)
    pred_v03 = (proba_v03 >= 0.5).astype(np.int64)
    f1_media = float(metrica_macro(y, pred_media))
    f1_v03 = float(metrica_macro(y, pred_v03))
    generador = np.random.default_rng(semilla)
    n = y.shape[0]
    diferencias = np.empty(repeticiones, dtype=np.float64)
    for repeticion in range(repeticiones):
        indices = generador.integers(0, n, n)
        diferencias[repeticion] = metrica_macro(
            y[indices], pred_media[indices]
        ) - metrica_macro(y[indices], pred_v03[indices])
    return {
        "referencia_v03_at_0_5": round(f1_v03, 12),
        "f1_media_candidato_v03_at_0_5": round(f1_media, 12),
        "ensemble_gain": round(f1_media - f1_v03, 12),
        "ic95": [round(float(np.quantile(diferencias, 0.025)), 6),
                 round(float(np.quantile(diferencias, 0.975)), 6)],
        "repeticiones": repeticiones,
        "semilla": semilla,
        "f1_candidato_solo_at_0_5": round(float(metrica_macro(y, pred_candidato)), 12),
    }


def resumir_semilla(
    semilla: int,
    datos: DatosGate0,
    proba: np.ndarray,
    pred_primaria: np.ndarray,
    pred_anidada: np.ndarray,
    pred_congelada: np.ndarray,
    resultados: list[ResultadoCombinacion],
) -> dict[str, Any]:
    """Construye el resumen de una semilla con las tres lecturas completas.

    Args:
        semilla: semilla resumida.
        datos: datos de Gate 0.
        proba: probabilidades OOF de la semilla.
        pred_primaria: predicciones OOF a 0.5.
        pred_anidada: predicciones OOF con umbrales anidados.
        pred_congelada: predicciones OOF con umbrales congelados de V03.
        resultados: resultados por pliegue de la semilla.

    Returns:
        Diccionario resumen serializable.
    """
    lecturas: dict[str, Any] = {}
    for nombre, pred in (
        ("primaria_umbral_0_5", pred_primaria),
        ("secundaria_anidada", pred_anidada),
        ("control_congelado_v03", pred_congelada),
    ):
        comparacion = comparar_contra_v03(datos, pred)
        bootstrap = bootstrap_diferencia_pareada(
            y_true=datos.y,
            pred_a=pred,
            pred_b=datos.pred_v03,
            repeticiones=REPETICIONES_BOOTSTRAP,
            semilla=SEMILLA_BOOTSTRAP,
        )
        lecturas[nombre] = {
            "metricas": metricas_de_lectura(datos.y, pred),
            "comparacion": comparacion,
            "bootstrap": {
                "diferencia_observada": bootstrap["diferencia_observada"],
                "ic95": bootstrap["ic95"],
                "mediana_bootstrap": bootstrap["mediana_bootstrap"],
                "repeticiones": bootstrap["repeticiones"],
                "semilla": SEMILLA_BOOTSTRAP,
            },
        }

    brechas = [resultado.brecha for resultado in resultados]
    f1_train = [resultado.f1_train_05 for resultado in resultados]
    f1_val = [resultado.f1_val_05 for resultado in resultados]
    e_estrella = [resultado.e_estrella for resultado in resultados]
    dispersiones = [resultado.dispersion_e_j for resultado in resultados]
    return {
        "semilla": semilla,
        "lecturas": lecturas,
        "delta_primario": lecturas["primaria_umbral_0_5"]["comparacion"]["delta_global"],
        "ensemble_gain": bootstrap_ensemble_gain(
            datos.y,
            proba,
            datos.p_v03,
            repeticiones=REPETICIONES_BOOTSTRAP,
            semilla=SEMILLA_BOOTSTRAP,
        ),
        "brecha": {
            "media": round(float(np.mean(brechas)), 6),
            "por_pliegue": [round(resultado.brecha, 6) for resultado in resultados],
            "f1_train_05_media": round(float(np.mean(f1_train)), 6),
            "f1_val_05_media": round(float(np.mean(f1_val)), 6),
        },
        "E_estrella_por_pliegue": e_estrella,
        "dispersion_E_j_por_pliegue": [round(v, 4) for v in dispersiones],
        "tiempo_segundos": round(float(sum(resultado.segundos for resultado in resultados)), 2),
        "umbrales_anidados_por_pliegue": [list(resultado.umbrales_anidados) for resultado in resultados],
        "pliegues_con_aviso_prevalencia": [
            resultado.pliegue for resultado in resultados if resultado.incidencias
        ],
    }


def evaluar_estabilidad(deltas_por_semilla: dict[int, float], semilla_primaria: int) -> dict[str, Any]:
    """Evalua la condicion de estabilidad entre semillas pre-declarada.

    Args:
        deltas_por_semilla: delta global de la lectura primaria por semilla.
        semilla_primaria: semilla de referencia (42).

    Returns:
        Diccionario con mismo signo, desviaciones y veredicto.
    """
    valores = np.array([deltas_por_semilla[s] for s in sorted(deltas_por_semilla)], dtype=np.float64)
    delta_primario = float(deltas_por_semilla[semilla_primaria])
    signo_primario = np.sign(delta_primario)
    mismo_signo = bool(
        delta_primario != 0.0 and np.all(np.sign(valores) == signo_primario)
    )
    sd_poblacional = float(np.std(valores, ddof=0))
    sd_muestral = float(np.std(valores, ddof=1)) if valores.size > 1 else 0.0
    umbral = abs(delta_primario) / 2.0
    suficiente = bool(valores.size >= 3)
    estable = bool(suficiente and mismo_signo and sd_poblacional < umbral)
    return {
        "deltas_por_semilla": {str(s): round(float(deltas_por_semilla[s]), 10)
                               for s in sorted(deltas_por_semilla)},
        "semilla_primaria": semilla_primaria,
        "delta_primario": round(delta_primario, 10),
        "mismo_signo": mismo_signo,
        "sd_poblacional_ddof0": round(sd_poblacional, 10),
        "sd_muestral_ddof1": round(sd_muestral, 10),
        "umbral_sd": round(umbral, 10),
        "semillas_suficientes": suficiente,
        "estable": estable,
        "regla": "mismo signo en las 3 semillas y sd poblacional < delta_primario / 2",
    }


def evaluar_d057(brechas_candidato: list[float], brechas_control: list[float]) -> dict[str, Any]:
    """Aplica el freno D-057 (brecha, capacidad y umbral efectivo).

    Args:
        brechas_candidato: brechas train-validacion del MLP.
        brechas_control: brechas train-validacion de V03 en el mismo harness.

    Returns:
        Diccionario con las medias, la diferencia y el veredicto.
    """
    media_candidato = float(np.mean(brechas_candidato))
    media_control = float(np.mean(brechas_control))
    diferencia = media_candidato - media_control
    return {
        "brecha_candidato_media": round(media_candidato, 6),
        "brecha_control_media": round(media_control, 6),
        "diferencia": round(diferencia, 6),
        "limite": D057_BRECHA_MAXIMA,
        "ok": bool(diferencia <= D057_BRECHA_MAXIMA),
        "capacidad": {
            "n_parametros": N_PARAMETROS_MLP,
            "dimension_densa": int(CONFIG["svd_componentes"]),
            "aumenta_capacidad": True,
        },
        "umbral_efectivo": GATE_DELTA_EFECTIVO,
        "n_brechas_candidato": len(brechas_candidato),
        "n_brechas_control": len(brechas_control),
    }


def evaluar_gate(
    resumenes: dict[int, dict[str, Any]],
    estabilidad: dict[str, Any],
    d057: dict[str, Any],
    semillas: tuple[int, ...],
    pliegues: tuple[int, ...],
) -> dict[str, Any]:
    """Aplica el Gate 1 con estabilidad y freno D-057.

    Args:
        resumenes: resumen por semilla.
        estabilidad: resultado de :func:`evaluar_estabilidad`.
        d057: resultado de :func:`evaluar_d057`.
        semillas: semillas ejecutadas.
        pliegues: pliegues ejecutados.

    Returns:
        Diccionario con motivos, condiciones y decision.
    """
    primaria = int(CONFIG["semilla_primaria"])
    if primaria not in resumenes:
        motivo = f"falta la semilla primaria {primaria}"
        return {
            "evaluable": False,
            "motivo": motivo,
            "motivos": [motivo],
            "promueve": False,
            "decision": "NO EVALUABLE",
        }
    if tuple(sorted(pliegues)) != tuple(range(1, 6)):
        motivo = "se ejecuto un subconjunto de pliegues; el gate preregistrado exige 5"
        return {
            "evaluable": False,
            "motivo": motivo,
            "motivos": [motivo],
            "promueve": False,
            "decision": "NO EVALUABLE",
        }
    lectura = resumenes[primaria]["lecturas"]["primaria_umbral_0_5"]
    delta = float(lectura["comparacion"]["delta_global"])
    mejorados = int(lectura["comparacion"]["pliegues_mejorados"])
    ic_bajo = float(lectura["bootstrap"]["ic95"][0])
    gate_base = bool(delta >= GATE_DELTA and mejorados >= GATE_PLIEGUES and ic_bajo > 0)
    gate_efectivo = bool(delta >= GATE_DELTA_EFECTIVO)
    estable = bool(estabilidad["estable"])
    brecha_ok = bool(d057["ok"])
    promueve = bool(gate_base and gate_efectivo and estable and brecha_ok)
    motivos: list[str] = []
    if not gate_base:
        motivos.append(
            f"Gate 1 base no cumplido: delta {delta:+.6f} (>= {GATE_DELTA}), "
            f"pliegues {mejorados}/5 (>= {GATE_PLIEGUES}), IC bajo {ic_bajo:+.6f} (> 0)"
        )
    if not gate_efectivo:
        motivos.append(
            f"D-057: delta {delta:+.6f} < umbral efectivo {GATE_DELTA_EFECTIVO} "
            "para candidatos que aumentan capacidad"
        )
    if not estable:
        motivos.append("estabilidad entre semillas no cumplida")
    if not brecha_ok:
        motivos.append(
            f"D-057 brecha: candidato {d057['brecha_candidato_media']:.6f} - control "
            f"{d057['brecha_control_media']:.6f} = {d057['diferencia']:+.6f} > "
            f"{D057_BRECHA_MAXIMA}"
        )
    return {
        "evaluable": True,
        "delta_primario": delta,
        "pliegues_mejorados": mejorados,
        "ic95_primario": lectura["bootstrap"]["ic95"],
        "gate_base": gate_base,
        "gate_d057_umbral_efectivo": gate_efectivo,
        "estable": estable,
        "brecha_ok": brecha_ok,
        "promueve": promueve,
        "motivos": motivos,
        "decision": (
            "PROMOVER A GATE 2 (segunda particion, sin reajuste)"
            if promueve
            else "CERRAR DL-01 COMO ARTEFACTO DE PROTOCOLO"
        ),
    }


# ===== SECCION 8: salidas (parciales, informes, npz, checkpoints y CSV) =====


def escribir_parcial(resultado: ResultadoCombinacion, humo: bool) -> Path:
    """Escribe el parcial JSON de una combinacion (pliegue, semilla).

    Args:
        resultado: resultado de la combinacion.
        humo: si True el parcial lleva el sufijo _humo.

    Returns:
        Ruta escrita.
    """
    directorio: Path = CONFIG["parciales_dir"]
    directorio.mkdir(parents=True, exist_ok=True)
    sufijo = "_humo" if humo else ""
    ruta = directorio / f"p84_01_p{resultado.pliegue}_s{resultado.semilla}{sufijo}.json"
    carga = {
        "experiment_id": "P84-01",
        "modo_humo": humo,
        "pliegue": resultado.pliegue,
        "semilla": resultado.semilla,
        "E_j": resultado.e_j,
        "E_estrella": resultado.e_estrella,
        "dispersion_E_j": round(resultado.dispersion_e_j, 6),
        "splits_internos": resultado.info_splits,
        "umbrales_anidados": resultado.umbrales_anidados,
        "lecturas_fold": resultado.f1_por_lectura,
        "f1_train_05": resultado.f1_train_05,
        "f1_val_05": resultado.f1_val_05,
        "brecha": resultado.brecha,
        "contador_prediccion": resultado.contador,
        "incidencias": resultado.incidencias,
        "ruta_checkpoint": resultado.ruta_checkpoint,
        "segundos": resultado.segundos,
    }
    escribir_json(ruta, carga)
    return ruta


def upsert_experiments_csv(fila: dict[str, str]) -> dict[str, Any]:
    """Inserta o reemplaza la fila P84-01 en experiments.csv (23 columnas).

    Valida la cabecera, el ancho de todas las filas, la ausencia de ids
    duplicados y que la fila defina exactamente las 23 columnas antes de
    escribir. Escribe utf-8-sig y no duplica ids.

    Args:
        fila: valores por columna de la fila P84-01.

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
    nueva = [fila.get(columna, "") for columna in COLUMNAS_EXPERIMENTS]
    faltantes = [columna for columna in COLUMNAS_EXPERIMENTS if columna not in fila]
    if faltantes:
        raise SystemExit(f"STOP: faltan columnas en la fila P84-01: {faltantes}")
    sobrantes = sorted(set(fila) - set(COLUMNAS_EXPERIMENTS))
    if sobrantes:
        raise SystemExit(f"STOP: columnas desconocidas en la fila P84-01: {sobrantes}")
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
    resumen_primario: dict[str, Any],
    gate: dict[str, Any],
    segundos_totales: float,
) -> dict[str, str]:
    """Construye la fila P84-01 para experiments.csv.

    Args:
        resumen_primario: resumen de la semilla primaria.
        gate: resultado del gate.
        segundos_totales: tiempo total de la corrida.

    Returns:
        Diccionario columna -> valor.
    """
    lectura = leer_ruta(resumen_primario, "lecturas.primaria_umbral_0_5")
    metricas = leer_ruta(lectura, "metricas")
    comparacion = leer_ruta(lectura, "comparacion")
    f1_por_etiqueta = leer_ruta(metricas, "f1_por_etiqueta")
    resultado = {
        "lectura_primaria_0_5": leer_ruta(comparacion, "f1_global"),
        "lectura_secundaria_anidada": leer_ruta(
            resumen_primario, "lecturas.secundaria_anidada.comparacion.f1_global"
        ),
        "lectura_congelada_v03": leer_ruta(
            resumen_primario, "lecturas.control_congelado_v03.comparacion.f1_global"
        ),
        "delta_global_0_5": leer_ruta(comparacion, "delta_global"),
        "delta_medio_pliegues": leer_ruta(comparacion, "delta_medio_pliegues"),
        "delta_por_pliegue": leer_ruta(comparacion, "delta_por_pliegue"),
        "pliegues_mejorados": leer_ruta(comparacion, "pliegues_mejorados"),
        "ic95_bootstrap": leer_ruta(lectura, "bootstrap.ic95"),
        "semilla_bootstrap": SEMILLA_BOOTSTRAP,
        "ensemble_gain": leer_ruta(resumen_primario, "ensemble_gain"),
        "rescate_celdas": leer_ruta(comparacion, "rescate_celdas"),
        "dano_celdas": leer_ruta(comparacion, "dano_celdas"),
        "neto_celdas": leer_ruta(comparacion, "neto_celdas"),
        "tasa_positivos": leer_ruta(metricas, "tasa_positivos"),
        "f1_por_etiqueta": f1_por_etiqueta,
        "precision_micro": leer_ruta(metricas, "precision_micro"),
        "recall_micro": leer_ruta(metricas, "recall_micro"),
        "brecha_media": leer_ruta(resumen_primario, "brecha.media"),
        "E_estrella_por_pliegue": leer_ruta(resumen_primario, "E_estrella_por_pliegue"),
        "umbrales_anidados_por_pliegue": leer_ruta(
            resumen_primario, "umbrales_anidados_por_pliegue"
        ),
        "deltas_por_semilla": gate.get("deltas_por_semilla", {}),
        "estabilidad": gate.get("estabilidad", {}),
        "d057": gate.get("d057", {}),
        "gate": gate,
    }
    return {
        "experiment_id": "P84-01",
        "competition": str(CONFIG["competencia"]),
        "question": (
            "El MLP DL-01 (TF-IDF union -> SVD300 -> 256-128) medido sin contaminacion de "
            "pool ni de early stopping supera a V03 (0.7772114525165721) en las 1483 filas "
            "de desarrollo?"
        ),
        "hypothesis": (
            "La cifra historica 0.794854 (pool 1731, epoca elegida sobre el pliegue "
            "puntuado) no es comparable; la re-medicion limpia puede quedar por debajo de "
            "V03 porque la compresion a 300 componentes descarta n-gramas poco frecuentes."
        ),
        "preprocessing": (
            "texto crudo en minusculas; vectorizador, IDF, SVD y scaler ajustados SOLO con "
            "outer_train (los pliegues internos solo eligen E* y umbrales); sin embeddings "
            "preentrenados; holdout y test nunca usados"
        ),
        "feature_groups": "WORD_NGRAM+CHAR_NGRAM (proyeccion densa)",
        "representation": (
            "TF-IDF union word (1,2) + char_wb (3,5), min_df=2, sublinear_tf -> "
            "TruncatedSVD(300, rs=42) -> StandardScaler"
        ),
        "model": "DL-MLP (TF-IDF union -> SVD 300 -> MLP 256-128, dropout 0.3)",
        "hyperparameters": json.dumps(
            {
                "arquitectura": "300-256-128-3",
                "activacion": "ReLU",
                "dropout": 0.3,
                "optimizador": "AdamW",
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 32,
                "max_epocas": 30,
                "paciencia": 5,
                "clip_grad_norm": 1.0,
                "perdida": "BCEWithLogitsLoss con pos_weight = n_neg/max(n_pos,1)",
                "amp": False,
                "semillas": list(CONFIG["semillas_por_defecto"]),
                "E_estrella": "mediana de 3 splits internos por semilla y pliegue",
            },
            ensure_ascii=True,
        ),
        "validation": (
            "pliegues_corregidos de 5 pliegues sobre desarrollo (1483, grupos fusionados); "
            "E* en splits internos agrupados 80/20 (semillas 20261101+100k+7j); refit sobre "
            "outer_train E* epocas; una sola prediccion de outer_val; umbral primario 0.5"
        ),
        "f1": f"{leer_ruta(comparacion, 'f1_global'):.6f}",
        "macro_f1": f"{leer_ruta(metricas, 'f1_macro6'):.6f}",
        "weighted_f1": "",
        "precision": f"{leer_ruta(metricas, 'precision_micro'):.6f}",
        "recall": f"{leer_ruta(metricas, 'recall_micro'):.6f}",
        "training_time": f"{segundos_totales:.2f}",
        "dimension": str(N_PARAMETROS_MLP),
        "result": json.dumps(resultado, ensure_ascii=True, default=_json_default),
        "interpretation": (
            f"F1 primario {leer_ruta(comparacion, 'f1_global'):.6f} vs V03 congelado "
            f"{leer_ruta(comparacion, 'f1_v03_global'):.6f}; delta "
            f"{leer_ruta(comparacion, 'delta_global'):+.6f}; pliegues mejorados "
            f"{leer_ruta(comparacion, 'pliegues_mejorados')}/5; brecha media "
            f"{leer_ruta(resumen_primario, 'brecha.media'):+.4f} vs V03 "
            f"{gate.get('d057', {}).get('brecha_control_media', float('nan')):+.4f}"
        ),
        "decision": str(gate.get("decision", "NO EVALUABLE")),
        "date": str(CONFIG["fecha"]),
        "script": str(CONFIG["script"]),
        "notes": (
            "EXP-01 del preregistro P84 (Enmienda 1). La cifra historica 0.794854 queda "
            "marcada como no comparable; identidad V03 verificada con tolerancia 1e-12; "
            "contadores dev 1483 / holdout 0 / test 0."
        ),
    }


def construir_informe_md(carga: dict[str, Any]) -> str:
    """Construye el informe Markdown ASCII de la corrida.

    Todas las lecturas anidadas pasan por ``leer_ruta``: si el JSON cambiara de
    forma, el fallo indica la ruta exacta en lugar de un KeyError opaco.

    Args:
        carga: informe JSON de la corrida.

    Returns:
        Texto Markdown.

    Raises:
        SystemExit: si el informe no contiene alguna ruta esperada.
    """
    lineas: list[str] = []
    lineas.append("# P84-01: EXP-01 (DL-01 corregido), medicion limpia\n\n")
    lineas.append(
        f"Fecha: {leer_ruta(carga, 'fecha')} | modo humo: {leer_ruta(carga, 'modo_humo')} | "
        f"device final: {leer_ruta(carga, 'dispositivo.final')} | "
        f"{leer_ruta(carga, 'tiempo_total_segundos'):.1f} s\n\n"
    )
    identidad = leer_ruta(carga, "gate0.identidad_v03")
    contadores = leer_ruta(carga, "gate0.evidencia.contadores")
    particiones = leer_ruta(carga, "gate0.evidencia.particiones")
    lineas.append("## Gate 0\n\n```text\n")
    lineas.append(
        "identidad V03: max|delta proba| {:.3e} | F1 congelado {:.10f} | F1@0.5 {:.10f}\n".format(
            leer_ruta(identidad, "max_delta_proba_vs_npz"),
            leer_ruta(identidad, "f1_congelado_reconstruido"),
            leer_ruta(identidad, "f1_at_05_recalculado"),
        )
    )
    lineas.append(
        "contadores: dev {} | holdout {} | test {} | toxic_test abierto {}\n".format(
            leer_ruta(contadores, "dev_filas_leidas"),
            leer_ruta(contadores, "holdout_filas_usadas"),
            leer_ruta(contadores, "test_filas_usadas"),
            leer_ruta(contadores, "toxic_test_abierto"),
        )
    )
    lineas.append(
        f"particiones: dev {leer_ruta(particiones, 'dev')} | holdout "
        f"{leer_ruta(particiones, 'holdout')} | "
        f"pliegues {leer_ruta(particiones, 'pliegues_tamanos')}\n"
    )
    lineas.append("```\n\n## Determinismo\n\n```text\n")
    det = leer_ruta(carga, "determinismo")
    lineas.append(
        f"doble pasada pliegue 1 semilla 42: max|delta p| {leer_ruta(det, 'max_delta_proba'):.3e} "
        f"<= {leer_ruta(det, 'tolerancia'):g} | aprobado {leer_ruta(det, 'aprobado')} | "
        f"fallback CPU {leer_ruta(det, 'fallback_cpu')}\n"
    )
    lineas.append("```\n\n## Lecturas por semilla (semilla 42 primaria)\n\n")
    lineas.append(
        "| semilla | lectura | F1 | delta vs V03 | IC95 | pliegues | rescue | damage |\n"
    )
    lineas.append("|---|---|---|---|---|---|---|---|\n")
    por_semilla = leer_ruta(carga, "por_semilla")
    for clave, resumen in por_semilla.items():
        for nombre in ("primaria_umbral_0_5", "secundaria_anidada", "control_congelado_v03"):
            lectura = leer_ruta(resumen, f"lecturas.{nombre}")
            comparacion = leer_ruta(lectura, "comparacion")
            ic = leer_ruta(lectura, "bootstrap.ic95")
            lineas.append(
                f"| {clave} | {nombre} | {leer_ruta(comparacion, 'f1_global'):.6f} | "
                f"{leer_ruta(comparacion, 'delta_global'):+.6f} | [{ic[0]:+.6f}, {ic[1]:+.6f}] | "
                f"{leer_ruta(comparacion, 'pliegues_mejorados')}/5 | "
                f"{leer_ruta(comparacion, 'rescate_celdas')} | "
                f"{leer_ruta(comparacion, 'dano_celdas')} |\n"
            )
    lineas.append("\n## Detalle por pliegue (semilla primaria)\n\n```text\n")
    primaria = str(CONFIG["semilla_primaria"])
    if primaria in por_semilla:
        resumen = por_semilla[primaria]
        lineas.append(
            f"E* por pliegue {leer_ruta(resumen, 'E_estrella_por_pliegue')} | dispersion E_j "
            f"{leer_ruta(resumen, 'dispersion_E_j_por_pliegue')}\n"
        )
        comparacion = leer_ruta(resumen, "lecturas.primaria_umbral_0_5.comparacion")
        lineas.append(
            f"F1@0.5 por pliegue       {leer_ruta(comparacion, 'f1_por_pliegue')}\n"
        )
        lineas.append(
            f"F1 V03 por pliegue       {leer_ruta(comparacion, 'f1_por_pliegue_v03')}\n"
        )
        lineas.append(
            f"delta por pliegue        {leer_ruta(comparacion, 'delta_por_pliegue')}\n"
        )
        lineas.append(
            f"brecha train-validacion  {leer_ruta(resumen, 'brecha')}\n"
        )
        lineas.append(
            f"ensemble_gain a 0.5      {leer_ruta(resumen, 'ensemble_gain')}\n"
        )
    lineas.append("```\n\n## Estabilidad, D-057 y decision\n\n```text\n")
    lineas.append(f"estabilidad: {leer_ruta(carga, 'estabilidad')}\n")
    lineas.append(f"d057: {leer_ruta(carga, 'd057')}\n")
    lineas.append(f"gate: {leer_ruta(carga, 'gate')}\n")
    lineas.append("```\n\n## Incidencias\n\n```text\n")
    incidencias = leer_ruta(carga, "incidencias")
    if incidencias:
        for incidencia in incidencias:
            lineas.append(f"{incidencia}\n")
    else:
        lineas.append("ninguna\n")
    lineas.append("```\n")
    return "".join(lineas)


def guardar_oof_npz(
    semillas: tuple[int, ...],
    probas: dict[int, np.ndarray],
    pliegues: tuple[int, ...],
    datos: DatosGate0,
    e_estrella: dict[int, list[int]],
    umbrales_anidados: dict[int, list[list[float]]],
    humo: bool,
) -> Path | None:
    """Guarda features/p84_01_oof_mlp.npz con las probabilidades por semilla.

    En modo humo no se escribe nada.

    Args:
        semillas: semillas ejecutadas.
        probas: probabilidades OOF por semilla.
        pliegues: pliegues ejecutados.
        datos: datos de Gate 0.
        e_estrella: E* por semilla y pliegue.
        umbrales_anidados: umbrales anidados por semilla y pliegue.
        humo: si True no se escribe.

    Returns:
        Ruta escrita o None en modo humo.
    """
    if humo:
        return None
    proba = np.full((len(semillas), FILAS_DEV, len(ETIQUETAS)), -1.0, dtype=np.float64)
    for indice, semilla in enumerate(semillas):
        proba[indice] = probas[semilla]
    np.savez_compressed(
        CONFIG["oof_npz"],
        proba=proba,
        semillas=np.array(semillas, dtype=np.int64),
        pliegues=datos.folde,
        y_true=datos.y,
        ids=datos.ids,
        E_estrella=np.array([e_estrella[s] for s in semillas], dtype=np.int64),
        pliegues_ejecutados=np.array(pliegues, dtype=np.int64),
        umbrales_anidados=np.array([umbrales_anidados[s] for s in semillas], dtype=np.float64),
        nota_contador=np.array(
            [
                (
                    "una pasada oficial de outer_val por (semilla, pliegue); la guardia "
                    "torch.equal no produce lecturas; las tres lecturas de umbral "
                    "reutilizan la misma matriz"
                )
            ]
        ),
    )
    return CONFIG["oof_npz"]


# ===== SECCION 9: CLI y orquestacion =====


def parsear_lista(texto: str, nombre: str) -> tuple[int, ...]:
    """Parsea una lista de enteros separados por comas.

    Args:
        texto: cadena de entrada.
        nombre: nombre del argumento para mensajes.

    Returns:
        Tupla de enteros sin duplicados.

    Raises:
        SystemExit: si el formato es invalido o hay duplicados.
    """
    try:
        valores = tuple(int(parte) for parte in texto.split(",") if parte.strip())
    except ValueError as error:
        raise SystemExit(f"STOP: --{nombre} invalido: {texto!r}") from error
    if not valores:
        raise SystemExit(f"STOP: --{nombre} no puede estar vacio")
    if len(set(valores)) != len(valores):
        raise SystemExit(f"STOP: --{nombre} tiene valores duplicados: {valores}")
    return valores


def parsear_argumentos(argv: Sequence[str] | None) -> argparse.Namespace:
    """Define y parsea los argumentos de la linea de comandos.

    Args:
        argv: argumentos (None usa sys.argv[1:]).

    Returns:
        Namespace con humo, cpu, semillas y pliegues.
    """
    analizador = argparse.ArgumentParser(
        description="P84-01: EXP-01 (DL-01 corregido) con protocolo anidado."
    )
    analizador.add_argument("--humo", action="store_true", help="modo humo: pliegue 1, semilla 42")
    analizador.add_argument("--cpu", action="store_true", help="fuerza CPU sin pre-vuelo GPU")
    analizador.add_argument(
        "--semillas", default="42,43,44", help="semillas separadas por comas (defecto 42,43,44)"
    )
    analizador.add_argument(
        "--pliegues", default="1,2,3,4,5", help="pliegues separados por comas (defecto 1,2,3,4,5)"
    )
    return analizador.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta P84-01 completo.

    Args:
        argv: argumentos de linea de comandos.

    Returns:
        Codigo de salida 0 si todo termino bien.
    """
    inicio = time.perf_counter()
    argumentos = parsear_argumentos(argv)
    semillas = parsear_lista(argumentos.semillas, "semillas")
    pliegues = parsear_lista(argumentos.pliegues, "pliegues")
    humo = bool(argumentos.humo)
    if humo:
        semillas = (int(CONFIG["semilla_primaria"]),)
        pliegues = (1,)
        log("MODO HUMO: pliegue 1, semilla 42; sin experiments.csv, npz ni checkpoints finales")
    for pliegue in pliegues:
        if pliegue not in CONFIG["pliegues_por_defecto"]:
            raise SystemExit(f"STOP: pliegue invalido {pliegue}")
    for semilla in semillas:
        if semilla <= 0:
            raise SystemExit(f"STOP: semilla invalida {semilla}")

    torch.set_num_threads(32)
    log(f"semillas {semillas} | pliegues {pliegues} | humo {humo} | cpu {argumentos.cpu}")

    evidencia_gpu: dict[str, Any] | None = None
    if not argumentos.cpu:
        evidencia_gpu = verificar_gpu_libre(CONFIG["gpu_baseline_json"], CONFIG["gpu_preflight_log"])
    dispositivo = resolver_dispositivo(argumentos.cpu)
    log(f"dispositivo inicial: {nombre_dispositivo(dispositivo)}")

    # ---------- Gate 0 ----------
    log("Gate 0: hashes congelados, particiones, contadores e identidad V03")
    datos = cargar_gate0()
    log(
        f"   dev {len(datos.des)} | holdout {datos.evidencia['particiones']['holdout']} usado 0 | "
        f"test usado 0 | pliegues {datos.evidencia['particiones']['pliegues_tamanos']}"
    )
    identidad, brecha_v03 = control_identidad_v03(datos)
    log(
        f"   identidad V03: max|delta proba| {identidad['max_delta_proba_vs_npz']:.3e} "
        f"(<= {TOL_IDENTIDAD:g}) | F1 congelado {identidad['f1_congelado_reconstruido']:.10f} | "
        f"F1@0.5 {identidad['f1_at_05_recalculado']:.12f} vs {V03_F1_AT_05:.12f}"
    )
    arquitectura = verificar_arquitectura(int(CONFIG["svd_componentes"]))
    log(
        f"   arquitectura: {arquitectura['n_parametros']} parametros + dim densa "
        f"{arquitectura['dim_entrada']}"
    )

    # ---------- pre-vuelo de determinismo ----------
    log("pre-vuelo de determinismo: doble pasada en (semilla 42, pliegue 1)")
    dispositivo, determinismo = preflight_determinismo(datos, dispositivo)
    log(f"dispositivo final: {nombre_dispositivo(dispositivo)}")

    # ---------- bucle principal ----------
    incidencias: list[str] = []
    probas: dict[int, np.ndarray] = {}
    pred_primaria: dict[int, np.ndarray] = {}
    pred_anidada: dict[int, np.ndarray] = {}
    pred_congelada: dict[int, np.ndarray] = {}
    e_estrella: dict[int, list[int]] = {}
    umbrales_anidados: dict[int, list[list[float]]] = {}
    resultados_por_semilla: dict[int, list[ResultadoCombinacion]] = {}
    resumenes: dict[int, dict[str, Any]] = {}
    brechas_candidato: list[float] = []
    rutas_parciales: list[str] = []

    for semilla in semillas:
        proba = np.full((FILAS_DEV, len(ETIQUETAS)), -1.0, dtype=np.float64)
        p_primaria = np.zeros((FILAS_DEV, len(ETIQUETAS)), dtype=np.int64)
        p_anidada = np.zeros_like(p_primaria)
        p_congelada = np.zeros_like(p_primaria)
        resultados: list[ResultadoCombinacion] = []
        e_estrella[semilla] = []
        umbrales_anidados[semilla] = []
        log(f"semilla {semilla}: inicio")
        for pliegue in sorted(pliegues):
            resultado = entrenar_combinacion(
                pliegue=pliegue,
                semilla=semilla,
                datos=datos,
                dispositivo=dispositivo,
                guardar_checkpoint=not humo,
            )
            _, indice_val = datos.pliegues[pliegue - 1]
            b = np.asarray(indice_val)
            proba[b] = resultado.proba_val
            p_primaria[b] = resultado.pred_primaria
            p_anidada[b] = resultado.pred_anidada
            p_congelada[b] = resultado.pred_congelada
            resultados.append(resultado)
            brechas_candidato.append(resultado.brecha)
            e_estrella[semilla].append(resultado.e_estrella)
            umbrales_anidados[semilla].append(list(resultado.umbrales_anidados))
            incidencias.extend(resultado.incidencias)
            rutas_parciales.append(str(escribir_parcial(resultado, humo)))
        probas[semilla] = proba
        pred_primaria[semilla] = p_primaria
        pred_anidada[semilla] = p_anidada
        pred_congelada[semilla] = p_congelada
        resultados_por_semilla[semilla] = resultados
        resumenes[semilla] = resumir_semilla(
            semilla, datos, proba, p_primaria, p_anidada, p_congelada, resultados
        )
        log(
            f"semilla {semilla}: F1@0.5 {resumenes[semilla]['delta_primario'] + V03_CCV:.6f} | "
            f"delta {resumenes[semilla]['delta_primario']:+.6f} | "
            f"pliegues {resumenes[semilla]['lecturas']['primaria_umbral_0_5']['comparacion']['pliegues_mejorados']}/5 | "
            f"brecha {resumenes[semilla]['brecha']['media']:+.4f}"
        )

    # ---------- estabilidad, D-057 y gate ----------
    deltas = {semilla: float(resumenes[semilla]["delta_primario"]) for semilla in semillas}
    estabilidad = evaluar_estabilidad(deltas, int(CONFIG["semilla_primaria"]))
    brecha_v03_media = list(brecha_v03)
    d057 = evaluar_d057(brechas_candidato, brecha_v03_media)
    gate = evaluar_gate(resumenes, estabilidad, d057, semillas, pliegues)
    gate["deltas_por_semilla"] = deltas
    gate["estabilidad"] = estabilidad
    gate["d057"] = d057
    log(f"estabilidad: {estabilidad['estable']} | mismo signo {estabilidad['mismo_signo']} | "
        f"sd {estabilidad['sd_poblacional_ddof0']:.6f} < {estabilidad['umbral_sd']:.6f}")
    log(f"D-057: brecha candidato {d057['brecha_candidato_media']:.6f} vs control "
        f"{d057['brecha_control_media']:.6f} | ok {d057['ok']}")
    log(f"GATE: {gate['decision']}")
    for motivo in gate["motivos"]:
        log(f"   motivo: {motivo}")
    if humo:
        gate["motivos"].append("modo humo: decision no evaluable por diseno")

    # ---------- salidas finales ----------
    tiempo_total = round(time.perf_counter() - inicio, 2)
    ruta_npz: str | None = None
    if not humo:
        ruta_npz_path = guardar_oof_npz(
            semillas, probas, pliegues, datos, e_estrella, umbrales_anidados, humo
        )
        ruta_npz = str(ruta_npz_path) if ruta_npz_path else None
    carga: dict[str, Any] = {
        "experiment_id": "P84-01",
        "script": str(CONFIG["script"]),
        "fecha": str(CONFIG["fecha"]),
        "modo_humo": humo,
        "preregistro": {
            "referencia": "docs/p84_spec.md seccion 2 (EXP-01) + docs/p84_preregistro.md Enmienda 1",
            "hipotesis": (
                "la cifra historica 0.794854 no es comparable (pool 1731 y epoca elegida "
                "sobre el pliegue puntuado); la re-medicion limpia decide con Gate 1"
            ),
            "criterio_cierre": (
                "si no pasa Gate 1 con estabilidad y freno D-057: cerrar DL-01 como "
                "artefacto de protocolo y no citar 0.794854 como base"
            ),
            "gate": {
                "delta_global_minimo": GATE_DELTA,
                "umbral_efectivo_d057": GATE_DELTA_EFECTIVO,
                "pliegues_minimos": GATE_PLIEGUES,
                "ic95_excluye_cero": True,
                "estabilidad": "mismo signo en 3 semillas y sd < delta_primario / 2",
                "brecha_maxima_sobre_control": D057_BRECHA_MAXIMA,
            },
        },
        "seleccion": {"semillas": list(semillas), "pliegues": list(pliegues)},
        "gate0": {
            "evidencia": datos.evidencia,
            "identidad_v03": identidad,
            "brecha_train_validacion_v03_media": identidad[
                "brecha_train_validacion_v03_media"
            ],
            "arquitectura": arquitectura,
        },
        "gpu": {"preflight": evidencia_gpu, "amp": False, "amp_evidencia": CONFIG["amp_evidencia"]},
        "determinismo": determinismo,
        "dispositivo": {
            "inicial": determinismo["dispositivo_inicial"],
            "final": determinismo["dispositivo_final"],
        },
        "versiones": versiones_entorno(),
        "por_semilla": {str(semilla): resumenes[semilla] for semilla in semillas},
        "estabilidad": estabilidad,
        "d057": d057,
        "gate": gate,
        "capacidad": {
            "n_parametros": N_PARAMETROS_MLP,
            "dimension_densa": int(CONFIG["svd_componentes"]),
            "aumenta_capacidad": True,
        },
        "parciales": rutas_parciales,
        "oof_npz": ruta_npz,
        "incidencias": incidencias,
        "tiempo_total_segundos": tiempo_total,
    }

    if not humo:
        escribir_json(CONFIG["informe_json"], carga)
        escribir_texto(CONFIG["informe_md"], construir_informe_md(carga))
        resumen_primario = resumenes[int(CONFIG["semilla_primaria"])] if int(CONFIG["semilla_primaria"]) in resumenes else next(iter(resumenes.values()))
        fila = construir_fila_experiments(resumen_primario, gate, tiempo_total)
        if 42 in semillas:
            upsert = upsert_experiments_csv(fila)
            carga["experiments_csv"] = upsert
            escribir_json(CONFIG["informe_json"], carga)
        else:
            log("AVISO: sin semilla 42 no se escribe experiments.csv")
        log(f"informes: {CONFIG['informe_json'].name} y {CONFIG['informe_md'].name}")

    # ---------- resumen final ----------
    log("=" * 78)
    log(f"P84-01 {'HUMO' if humo else 'OK'} | device {determinismo['dispositivo_final']} | "
        f"{tiempo_total:.1f}s")
    log(f"   Gate 0: identidad V03 max|delta| {identidad['max_delta_proba_vs_npz']:.3e} | "
        f"F1 congelado {identidad['f1_congelado_reconstruido']:.10f}")
    primaria = str(CONFIG["semilla_primaria"])
    if humo:
        log("   modo humo: OOF incompleto (solo pliegue 1, semilla 42); metricas no interpretables")
    elif primaria in carga["por_semilla"]:
        lectura = carga["por_semilla"][primaria]["lecturas"]["primaria_umbral_0_5"]
        log(f"   semilla 42: F1@0.5 {lectura['comparacion']['f1_global']:.6f} | delta "
            f"{lectura['comparacion']['delta_global']:+.6f} | pliegues "
            f"{lectura['comparacion']['pliegues_mejorados']}/5 | IC95 "
            f"[{lectura['bootstrap']['ic95'][0]:+.6f}, {lectura['bootstrap']['ic95'][1]:+.6f}]")
        log(f"   brecha MLP {d057['brecha_candidato_media']:.4f} vs V03 "
            f"{d057['brecha_control_media']:.4f} | estable {estabilidad['estable']} | "
            f"decision {gate['decision']}")
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
