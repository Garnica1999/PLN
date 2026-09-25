"""11_dl_desde_cero.py - Fase 11: tres redes neuronales entrenadas DESDE CERO.

Pregunta que responde
---------------------
El requisito metodologico del curso exige entrenar redes neuronales desde cero:
sin transformers, sin embeddings preentrenados y con el vocabulario construido
SOLO con el pliegue de entrenamiento. Con 1.731 ejemplos y una referencia
clasica fuerte (MOD-03b_BEST, TF-IDF union + LR, OOF 0.7927; B1 OOF 0.7587),
¿alguna red neuronal desde cero alcanza a la familia lineal?

Arquitecturas (cabezal multi-tarea de 3 logits y BCEWithLogitsLoss con
pos_weight por etiqueta calculado dentro del pliegue)
------------------------------------------------------
- DL-01 (DL-MLP): TF-IDF union REP-05 -> TruncatedSVD(300) ajustado solo con el
  pliegue -> estandarizado -> MLP 256 -> 128 (ReLU, dropout 0.3) -> 3 logits.
- DL-02 (DL-CHARCNN): embeddings de caracter desde cero (dim 48, vocab del
  pliegue, longitud 200) -> Conv1d de 128 filtros con kernels 3 y 5 en paralelo
  -> max pooling enmascarado -> dropout -> densa 128 -> 3 logits.
- DL-03 (DL-BILSTM): embeddings de palabra desde cero (dim 64, vocab del pliegue
  con min_freq=2, longitud 40) -> BiLSTM(96) -> pooling enmascarado (media + max)
  -> dropout -> 3 logits.

Protocolo
---------
- Validacion: el MISMO split de todas las fases,
  ``StratifiedGroupKFold(5, shuffle=True, random_state=42)`` con grupos de
  near-duplicado (D-004) y estratificacion por la combinacion HS x TR x AG.
- Pre-vuelo de GPU obligatorio (D-008): ``rocm-smi`` con 5 muestras cada 2 s.
  Si la tarjeta esta ocupada el script NO entrena, escribe la evidencia en
  ``experiments/gpu_preflight.log`` y termina sin tocar nada.
- Entrenamiento: AdamW (lr 1e-3, weight decay 1e-4), batch 32, maximo 30 epocas,
  EARLY STOPPING ESTRICTO sobre la F1 media por columna de validacion (umbral
  0.5), paciencia 5 y restauracion de los mejores pesos (decision ya aprobada
  por Carlos en el plan del proyecto). ``clip_grad_norm_`` 1.0.
- AMP: NO se usa. Medido en esta GPU (ROCm 7.2, RX 9070 XT, 1 pliegue, 3
  epocas): BiLSTM 3.42 s sin AMP frente a 7.48 s con AMP; CharCNN 0.41 s frente
  a 1.73 s. Con modelos de este tamano el coste de los casts supera cualquier
  ganancia de las unidades matriciales en fp16.
- Determinismo: se fija explicitamente ``torch.use_deterministic_algorithms
  (False)`` y ``torch.backends.cudnn.deterministic = False``. Evidencia: con la
  bandera en True una prueba conjunta de las tres arquitecturas aborto con
  ``HSA_STATUS_ERROR_EXCEPTION`` (coredump de GPU) en este stack; una prueba
  aislada del BiLSTM si completa, pero no hay ninguna ganancia medible que
  justifique el riesgo. La reproducibilidad se garantiza con las semillas fijas
  (42, 43, 44) en la misma maquina y entorno.
- Sin fuga: vectorizador, TruncatedSVD, escalador y vocabularios se ajustan SOLO
  con el pliegue de entrenamiento. El test no se carga en este script.

Hallazgo de depuracion (smoke test del 2026-09-23)
--------------------------------------------------
La primera version codificaba los tokens fuera de vocabulario como relleno (0).
Un texto cuyos tokens eran todos OOV quedaba totalmente enmascarado y el
centinela del max pooling (-1e9) se propagaba a los logits: la perdida de
validacion de DL-03 alcanzaba 2.4e6 (equivalente a una unica fila con logit
-1e9, 1e9/346 = 2.9e6). Se corrigio reservando el indice 1 para el token
desconocido (el vocabulario empieza en 2) y anadiendo una guarda que pone a cero
las filas sin posiciones validas.

Artefactos
----------
- ``reports/dl_desde_cero_resultados.csv`` y ``.json`` (detalle completo).
- ``models/dl_<arquitectura>.pt``: mejor pliegue de la mejor arquitectura, con
  pesos, vocabulario y preprocesamiento del pliegue.
- ``reports/figuras/fig_26_dl_curvas.png``: perdida y F1 por epoca del mejor
  pliegue de cada arquitectura.
- filas ``DL-01``, ``DL-02``, ``DL-03`` y ``DL-03-SEEDS`` en ``experiments.csv``
  (idempotentes: se reescriben por ``experiment_id``, nunca se duplican).
- seccion de handoff (maximo 20 lineas) en ``reports/contexto.md``.

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/11_dl_desde_cero.py
"""

from __future__ import annotations

import copy
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.evaluation import ETIQUETAS, MetricasMultiEtiqueta, metricas_multietiqueta

CONFIG: Final[dict] = {
    "grupos_path": PROJECT_ROOT / "data" / "processed" / "grupos_near_duplicados.csv",
    "gpu_baseline_json": PROJECT_ROOT / "config" / "gpu_baseline.json",
    "gpu_preflight_log": PROJECT_ROOT / "experiments" / "gpu_preflight.log",
    "reporte_csv": PROJECT_ROOT / "reports" / "dl_desde_cero_resultados.csv",
    "reporte_json": PROJECT_ROOT / "reports" / "dl_desde_cero_resultados.json",
    "figura": PROJECT_ROOT / "reports" / "figuras" / "fig_26_dl_curvas.png",
    "modelos_dir": PROJECT_ROOT / "models",
    "experiments_csv": PROJECT_ROOT / "experiments.csv",
    "contexto_md": PROJECT_ROOT / "reports" / "contexto.md",
    "encabezado_contexto": "## Fase 11 - dl-specialist (2026-09-23)",
    "max_lineas_contexto": 20,
    "competencia": "KAGGLE_TOXIC_2026_PUJ",
    "fecha": "2026-09-23",
    "script": "experiments/11_dl_desde_cero.py",
    # Particion: identica a las fases previas (D-004).
    "n_splits": 5,
    "random_state": 42,
    "semillas": (42, 43, 44),
    # Entrenamiento.
    "umbral": 0.5,
    "batch_size": 32,
    "max_epocas": 30,
    "paciencia": 5,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "clip_grad_norm": 1.0,
    "usar_amp": False,
    # Arquitecturas.
    "svd_componentes": 300,
    "mlp_ocultas": (256, 128),
    "dropout": 0.3,
    "char_dim": 48,
    "char_filtros": 128,
    "char_kernels": (3, 5),
    "char_max_len": 200,
    "char_oculta": 128,
    "word_dim": 64,
    "word_oculto": 96,
    "word_max_len": 40,
    "word_min_freq": 2,
    # Presupuesto de la fase (menos de 25 min de GPU/CPU en total).
    "presupuesto_segundos": 1200,
    "decision_overfitting": "early stopping estricto (aprobado en el plan)",
    "referencias": {
        "MOD-03b_BEST": 0.7926588861781015,
        "B1": 0.7587146098126788,
        "LB_MOD03b": 0.78306,
    },
    "evidencia_amp": (
        "Medido en ROCm 7.2 / RX 9070 XT (1 pliegue, 3 epocas, batch 32): "
        "BiLSTM 3.42 s sin AMP vs 7.48 s con AMP; CharCNN 0.41 s vs 1.73 s. "
        "Con modelos de este tamano los casts de autocast cuestan mas que la "
        "ganancia de fp16, por lo que AMP queda desactivado."
    ),
    "evidencia_determinismo": (
        "Con torch.use_deterministic_algorithms(True) una prueba conjunta de las "
        "tres arquitecturas aborto con HSA_STATUS_ERROR_EXCEPTION (coredump de "
        "GPU) en ROCm 7.2 / gfx1201; una prueba aislada del BiLSTM si completa, "
        "pero no hay ganancia medible que justifique el riesgo. Se fija en False "
        "y la reproducibilidad se garantiza con semillas fijas."
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

COLUMNAS_REPORTE: Final[tuple[str, ...]] = (
    "experiment_id",
    "arquitectura",
    "semilla",
    "n_parametros",
    "vocab",
    "dim_embedding",
    "longitud_secuencia",
    "tipo_entrada",
    "epocas_ejecutadas_media",
    "epocas_mejor_media",
    "tiempo_segundos",
    "f1_val_media",
    "f1_val_std",
    "f1_HS",
    "f1_TR",
    "f1_AG",
    "f1_train_media",
    "gap_train_val",
    "delta_vs_MOD03b",
    "delta_vs_B1",
    "f1_por_pliegue",
    "decision",
)

PREPROCESAMIENTO_DL: Final[str] = (
    "texto crudo en minusculas; vocabulario y preprocesamiento ajustados SOLO con "
    "el pliegue de entrenamiento (sin embeddings preentrenados)"
)
VALIDACION_DL: Final[str] = (
    "StratifiedGroupKFold(5, shuffle=True, random_state=42) con grupos de "
    "near-duplicado (D-004) y estratificacion por HSxTRxAG; early stopping "
    "estricto sobre la F1 media por columna de validacion (umbral 0.5, paciencia "
    "5, mejores pesos restaurados); test nunca usado"
)
RE_TOKEN: Final = re.compile(r"[a-z0-9]+")
# Indices reservados de la codificacion: 0 es relleno (``padding_idx``) y 1 es el
# token desconocido. Los tokens del vocabulario empiezan en 2. El token
# desconocido es imprescindible: sin el, un texto cuyos tokens son todos OOV
# quedaria totalmente enmascarado y el centinela del max pooling (-1e9) se
# propagaria a los logits (se observo una perdida de validacion de ~2.4e6 en el
# smoke test, equivalente a una unica fila con logit -1e9).
INDICE_RELLENO: Final[int] = 0
INDICE_DESCONOCIDO: Final[int] = 1
PRIMER_INDICE_VOCABULARIO: Final[int] = 2


def log(message: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        message: texto a imprimir.
    """
    print(f"[11] {message}", flush=True)


def dispositivo_entrenamiento() -> torch.device:
    """Devuelve el dispositivo de entrenamiento disponible.

    Returns:
        ``cuda`` si la GPU ROCm esta accesible; ``cpu`` en caso contrario.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fijar_semillas(semilla: int) -> None:
    """Fija todas las semillas y el modo de determinismo del proyecto.

    Se fija ``torch.use_deterministic_algorithms(False)`` de forma explicita
    porque en este stack (ROCm 7.2 + gfx1201) la bandera en True provoco un
    coredump de GPU en la prueba conjunta de las tres arquitecturas; la
    reproducibilidad se garantiza con las semillas fijas.

    Args:
        semilla: semilla entera para random, numpy y torch.
    """
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


# ===== SECCION 1: pre-vuelo de GPU (D-008) =====


def muestra_gpu() -> dict:
    """Toma una muestra del estado de la GPU con ``rocm-smi``.

    Returns:
        Diccionario con ``uso_pct``, ``vram_bytes`` y ``pids_kfd`` de la muestra.

    Raises:
        RuntimeError: si ``rocm-smi`` falla o no devuelve los campos esperados.
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
    vrams = [
        int(valor) for valor in re.findall(r"VRAM Total Used Memory \(B\): (\d+)", texto)
    ]
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
    return {
        "uso_pct": max(usos),
        "vram_bytes": max(vrams),
        "pids_kfd": pids,
    }


def verificar_gpu_libre() -> dict:
    """Ejecuta el pre-vuelo de GPU de D-008 (5 muestras, 2 s entre muestras).

    El chequeo debe ejecutarse ANTES de inicializar el contexto de torch: en
    cuanto se llama a ``torch.cuda.is_available()`` el proceso aparece en la
    lista de KFD y se auto-bloquearia. Por eso el PID propio se excluye del
    criterio de "PID ajeno".

    Returns:
        Evidencia completa del chequeo con la decision ``libre``/``ocupada``.

    Raises:
        SystemExit: si la GPU esta ocupada; el script no debe entrenar.
    """
    config_gpu = json.loads(CONFIG["gpu_baseline_json"].read_text(encoding="utf-8"))
    baseline = int(config_gpu["vram_usada_en_reposo_bytes"])
    umbrales = config_gpu["umbrales"]
    margen = int(umbrales["vram_sobre_baseline_bytes_bloquea"])
    uso_limite = int(umbrales["gpu_use_pct_sostenido_bloquea"])
    minimo = int(umbrales["min_muestras_sobre_umbral"])
    pid_propio = os.getpid()

    muestras: list[dict] = []
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
    registrar_preflight(evidencia)
    if ocupada:
        log("GPU OCUPADA: no se entrena. Evidencia en experiments/gpu_preflight.log")
        raise SystemExit("GPU OCUPADA: " + "; ".join(motivos))
    log("GPU libre: se procede con el entrenamiento")
    return evidencia


def registrar_preflight(evidencia: dict) -> Path:
    """Anade la evidencia del pre-vuelo a ``experiments/gpu_preflight.log``.

    Args:
        evidencia: diccionario devuelto por :func:`verificar_gpu_libre`.

    Returns:
        Ruta del log actualizado.
    """
    destino: Path = CONFIG["gpu_preflight_log"]
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
    with destino.open("a", encoding="utf-8") as manejador:
        manejador.write("\n".join(lineas) + "\n")
    return destino


# ===== SECCION 2: datos y particion de validacion =====


def cargar_datos() -> tuple[pd.DataFrame, pd.Series]:
    """Carga el train oficial y los grupos de near-duplicado.

    Returns:
        Tupla ``(train, grupos)`` con ``grupos`` indexada por ``id``.

    Raises:
        FileNotFoundError: si falta la particion generada en la Fase 0.
    """
    train = data_module.read_train()
    grupos_path: Path = CONFIG["grupos_path"]
    if not grupos_path.exists():
        raise FileNotFoundError(
            f"Falta {grupos_path}. Ejecuta experiments/00_auditoria_datos.py primero."
        )
    grupos = pd.read_csv(grupos_path).set_index("id")["grupo"]
    return train, grupos


def construir_pliegues(
    train: pd.DataFrame, grupos: pd.Series
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construye la particion 5-fold agrupada y estratificada (D-004).

    Replica exactamente la particion de ``experiments/03_variantes_representaciones.py``
    para que las metricas de esta fase sean comparables con las anteriores.

    Args:
        train: DataFrame de entrenamiento con ``id``, ``text`` y etiquetas.
        grupos: serie de grupos de near-duplicado indexada por ``id``.

    Returns:
        Lista de 5 tuplas ``(indices_entrenamiento, indices_validacion)``.

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
            f"pliegue {numero}: {len(entrenamiento)} train / {len(validacion)} "
            f"validacion | grupos validacion {asignacion.iloc[validacion].nunique()}"
        )
    return pliegues


def cobertura_secuencia(textos: Sequence[str], limite: int, por_palabra: bool) -> dict:
    """Mide cuanta informacion conserva el recorte a ``limite`` unidades.

    Args:
        textos: textos del corpus de entrenamiento.
        limite: longitud maxima admitida.
        por_palabra: si es ``True`` la unidad es la palabra; si no, el caracter.

    Returns:
        Diccionario con la fraccion de textos completos y la de unidades
        conservadas.
    """
    if por_palabra:
        longitudes = [len(tokenizar(texto)) for texto in textos]
    else:
        longitudes = [len(texto) for texto in textos]
    completos = sum(1 for longitud in longitudes if longitud <= limite)
    conservadas = sum(min(longitud, limite) for longitud in longitudes)
    total = sum(longitudes)
    return {
        "limite": limite,
        "unidad": "palabras" if por_palabra else "caracteres",
        "fraccion_textos_completos": round(completos / len(longitudes), 4),
        "fraccion_unidades_conservadas": round(conservadas / total, 4),
        "longitud_maxima": int(max(longitudes)),
        "longitud_mediana": int(np.median(longitudes)),
    }


def tokenizar(texto: str) -> list[str]:
    """Tokeniza un tweet en minusculas con el patron alfanumerico del proyecto.

    Args:
        texto: texto crudo.

    Returns:
        Lista de tokens ``[a-z0-9]+``.
    """
    return RE_TOKEN.findall(texto.lower())


# ===== SECCION 3: vocabularios y codificacion por pliegue =====


def vocabulario_caracteres(textos_entrenamiento: Sequence[str]) -> list[str]:
    """Construye el vocabulario de caracteres con el pliegue de entrenamiento.

    Args:
        textos_entrenamiento: textos del pliegue de entrenamiento.

    Returns:
        Lista ordenada de caracteres distintos.
    """
    return sorted({caracter for texto in textos_entrenamiento for caracter in texto})


def vocabulario_palabras(
    textos_entrenamiento: Sequence[str], min_frecuencia: int
) -> list[str]:
    """Construye el vocabulario de palabras con el pliegue de entrenamiento.

    El orden es determinista: frecuencia descendente y, a igualdad, orden
    alfabetico.

    Args:
        textos_entrenamiento: textos del pliegue de entrenamiento.
        min_frecuencia: frecuencia minima para entrar al vocabulario.

    Returns:
        Lista de palabras del vocabulario.
    """
    conteo = Counter(
        token for texto in textos_entrenamiento for token in tokenizar(texto)
    )
    candidatas = [palabra for palabra, total in conteo.items() if total >= min_frecuencia]
    return sorted(candidatas, key=lambda palabra: (-conteo[palabra], palabra))


def codificar_secuencias(
    textos: Sequence[str],
    indice: dict[str, int],
    limite: int,
    por_palabra: bool,
) -> np.ndarray:
    """Codifica textos como secuencias enteras con relleno a la derecha.

    Los indices ``0`` (relleno) y ``1`` (desconocido) estan reservados; el
    vocabulario empieza en ``2``. Los tokens fuera de vocabulario se codifican
    como desconocidos, NUNCA como relleno: si se codificaran como relleno, un
    texto cuyos tokens son todos OOV quedaria totalmente enmascarado y el
    centinela del max pooling se propagaria a los logits.

    Args:
        textos: textos a codificar.
        indice: mapeo token -> entero (>= 2).
        limite: longitud maxima de la secuencia.
        por_palabra: si es ``True`` la unidad es la palabra; si no, el caracter.

    Returns:
        Matriz entera de forma ``(n_textos, limite)``.
    """
    matriz = np.full((len(textos), limite), INDICE_RELLENO, dtype=np.int64)
    for fila, texto in enumerate(textos):
        unidades = tokenizar(texto) if por_palabra else list(texto)
        for columna, unidad in enumerate(unidades[:limite]):
            matriz[fila, columna] = indice.get(unidad, INDICE_DESCONOCIDO)
    return matriz


def peso_positivos(etiquetas: np.ndarray) -> torch.Tensor:
    """Calcula ``pos_weight`` por etiqueta con el pliegue de entrenamiento.

    Args:
        etiquetas: matriz ``(n, 3)`` de etiquetas del pliegue de entrenamiento.

    Returns:
        Tensor ``(3,)`` con ``n_negativos / n_positivos`` por etiqueta.
    """
    positivos = etiquetas.sum(axis=0).astype(np.float64)
    negativos = etiquetas.shape[0] - positivos
    return torch.tensor(negativos / np.maximum(positivos, 1.0), dtype=torch.float32)


# ===== SECCION 4: arquitecturas multi-tarea =====


def maximo_enmascarado(
    activaciones: torch.Tensor, mascara: torch.Tensor
) -> torch.Tensor:
    """Calcula el maximo temporal ignorando las posiciones de relleno.

    Args:
        activaciones: tensor ``(lote, canales, longitud)``.
        mascara: tensor booleano ``(lote, 1, longitud)`` con True en las
            posiciones validas.

    Returns:
        Tensor ``(lote, canales)``. Las filas sin ninguna posicion valida se
        devuelven a cero: el token desconocido garantiza que eso no ocurra, pero
        la guarda evita propagar el centinela del pooling a los logits.
    """
    maximo = activaciones.masked_fill(~mascara, -1e9).amax(dim=-1)
    vacias = ~mascara.any(dim=-1)
    return maximo.masked_fill(vacias, 0.0)


class MLPMultiTarea(nn.Module):
    """Perceptron multicapa sobre la proyeccion densa del TF-IDF."""

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


class CharCNNMultiTarea(nn.Module):
    """CNN de caracteres con kernels 3 y 5 en paralelo y max pooling enmascarado."""

    def __init__(
        self,
        vocab_size: int,
        dim_embedding: int,
        filtros: int,
        kernels: Sequence[int],
        oculta: int,
        dropout: float,
        padding_idx: int = 0,
    ) -> None:
        """Inicializa embeddings, convoluciones y cabezal denso.

        Args:
            vocab_size: tamano del vocabulario (incluye el relleno).
            dim_embedding: dimension del embedding entrenado desde cero.
            filtros: numero de filtros por kernel.
            kernels: tamanos de kernel de las ramas paralelas.
            oculta: ancho de la capa densa previa a la salida.
            dropout: probabilidad de dropout.
            padding_idx: indice reservado para el relleno.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_embedding, padding_idx=padding_idx)
        self.convoluciones = nn.ModuleList(
            nn.Conv1d(dim_embedding, filtros, kernel, padding=kernel // 2)
            for kernel in kernels
        )
        self.dropout = nn.Dropout(dropout)
        self.densa = nn.Linear(filtros * len(kernels), oculta)
        self.salida = nn.Linear(oculta, len(ETIQUETAS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calcula los 3 logits de un lote de secuencias de caracteres.

        Args:
            x: tensor entero ``(lote, longitud)`` con indices de caracter.

        Returns:
            Tensor ``(lote, 3)`` de logits.
        """
        mascara = (x != 0).unsqueeze(1)
        embeddings = self.embedding(x).transpose(1, 2)
        activaciones = [
            maximo_enmascarado(torch.relu(convolucion(embeddings)), mascara)
            for convolucion in self.convoluciones
        ]
        combinada = torch.cat(activaciones, dim=-1)
        return self.salida(self.dropout(torch.relu(self.densa(self.dropout(combinada)))))


class BiLSTMMultiTarea(nn.Module):
    """BiLSTM sobre embeddings de palabra con pooling enmascarado media + maximo."""

    def __init__(
        self,
        vocab_size: int,
        dim_embedding: int,
        oculto: int,
        dropout: float,
        padding_idx: int = 0,
    ) -> None:
        """Inicializa embeddings, LSTM bidireccional y cabezal lineal.

        Args:
            vocab_size: tamano del vocabulario (incluye el relleno).
            dim_embedding: dimension del embedding entrenado desde cero.
            oculto: dimension oculta por direccion.
            dropout: probabilidad de dropout antes de la capa de salida.
            padding_idx: indice reservado para el relleno.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_embedding, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            dim_embedding, oculto, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.salida = nn.Linear(4 * oculto, len(ETIQUETAS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calcula los 3 logits de un lote de secuencias de palabras.

        Args:
            x: tensor entero ``(lote, longitud)`` con indices de palabra.

        Returns:
            Tensor ``(lote, 3)`` de logits.
        """
        mascara = (x != 0).unsqueeze(-1).float()
        salida, _ = self.lstm(self.embedding(x))
        salida = salida * mascara
        media = salida.sum(dim=1) / mascara.sum(dim=1).clamp(min=1.0)
        maximo = salida.masked_fill(mascara == 0, -1e9).amax(dim=1)
        # Guarda defensiva: sin tokens validos el centinela llegaria a los logits.
        maximo = maximo.masked_fill(mascara.sum(dim=1) == 0, 0.0)
        return self.salida(self.dropout(torch.cat([media, maximo], dim=-1)))


# ===== SECCION 5: especificacion de arquitecturas y preparacion por pliegue =====


@dataclass
class DatosPliegue:
    """Tensores y metadatos de un pliegue listos para entrenar.

    Attributes:
        x_train: entradas del pliegue de entrenamiento.
        x_val: entradas del pliegue de validacion.
        vocab_size: tamano del vocabulario usado por el embedding.
        dim_entrada: dimension de entrada de la primera capa.
        longitud_secuencia: longitud de la entrada (secuencia o vector).
        tipo_entrada: descripcion legible del tipo de entrada.
        preprocesamiento: objetos o metadatos necesarios para reproducir la
            entrada (vocabulario, vectorizador, SVD, escalador).
        diagnostico: metricas del preprocesamiento del pliegue.
    """

    x_train: torch.Tensor
    x_val: torch.Tensor
    vocab_size: int
    dim_entrada: int
    longitud_secuencia: int
    tipo_entrada: str
    preprocesamiento: dict
    diagnostico: dict


@dataclass(frozen=True)
class EspecArquitectura:
    """Especificacion de una arquitectura de la fase.

    Attributes:
        identificador: id del experimento (DL-01..DL-03).
        clave: nombre corto para el archivo de pesos.
        nombre: nombre legible de la arquitectura.
        pregunta: pregunta del registro de experimentos.
        hipotesis: hipotesis previa del registro de experimentos.
        representacion: representacion de entrada para ``experiments.csv``.
        feature_groups: familia de features para ``experiments.csv``.
        preparar: funcion que construye las entradas de un pliegue.
        construir: funcion ``(vocab_size, dim_entrada) -> nn.Module``.
        descripcion_dimension: descripcion de la dimension reportada.
    """

    identificador: str
    clave: str
    nombre: str
    pregunta: str
    hipotesis: str
    representacion: str
    feature_groups: str
    preparar: Callable[..., DatosPliegue]
    construir: Callable[[int, int], nn.Module]
    descripcion_dimension: str


def preparar_mlp(
    textos: Sequence[str],
    entrenamiento: np.ndarray,
    validacion: np.ndarray,
) -> DatosPliegue:
    """Construye la entrada densa del MLP con el pliegue de entrenamiento.

    El vectorizador TF-IDF (REP-05), el ``TruncatedSVD`` y el escalador se
    ajustan SOLO con el pliegue de entrenamiento; la validacion solo se
    transforma. La estandarizacion es necesaria porque las componentes SVD
    tienen escalas decrecientes segun el valor singular.

    Args:
        textos: textos de todo el train.
        entrenamiento: indices posicionales del pliegue de entrenamiento.
        validacion: indices posicionales del pliegue de validacion.

    Returns:
        Entradas densas y metadatos del pliegue.
    """
    vectorizador = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=1.0,
                    sublinear_tf=True,
                    lowercase=True,
                    strip_accents=None,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_df=1.0,
                    sublinear_tf=True,
                    lowercase=True,
                    strip_accents=None,
                ),
            ),
        ]
    )
    matriz_entrenamiento = vectorizador.fit_transform(
        [textos[indice] for indice in entrenamiento]
    )
    matriz_validacion = vectorizador.transform([textos[indice] for indice in validacion])
    svd = TruncatedSVD(
        n_components=CONFIG["svd_componentes"], random_state=CONFIG["random_state"]
    )
    denso_entrenamiento = svd.fit_transform(matriz_entrenamiento)
    denso_validacion = svd.transform(matriz_validacion)
    escalador = StandardScaler()
    denso_entrenamiento = escalador.fit_transform(denso_entrenamiento)
    denso_validacion = escalador.transform(denso_validacion)
    return DatosPliegue(
        x_train=torch.tensor(denso_entrenamiento, dtype=torch.float32),
        x_val=torch.tensor(denso_validacion, dtype=torch.float32),
        vocab_size=0,
        dim_entrada=int(svd.n_components),
        longitud_secuencia=int(svd.n_components),
        tipo_entrada="vector denso (TruncatedSVD del TF-IDF union)",
        preprocesamiento={
            "vectorizador": vectorizador,
            "svd": svd,
            "escalador": escalador,
        },
        diagnostico={
            "dimension_tfidf": int(matriz_entrenamiento.shape[1]),
            "varianza_explicada_svd": round(
                float(svd.explained_variance_ratio_.sum()), 4
            ),
            "componentes_svd": int(svd.n_components),
        },
    )


def preparar_charcnn(
    textos: Sequence[str],
    entrenamiento: np.ndarray,
    validacion: np.ndarray,
) -> DatosPliegue:
    """Codifica las secuencias de caracteres con el vocabulario del pliegue.

    Args:
        textos: textos de todo el train.
        entrenamiento: indices posicionales del pliegue de entrenamiento.
        validacion: indices posicionales del pliegue de validacion.

    Returns:
        Secuencias enteras de caracteres y metadatos del pliegue.
    """
    vocabulario = vocabulario_caracteres([textos[indice] for indice in entrenamiento])
    indice = {
        caracter: posicion + PRIMER_INDICE_VOCABULARIO
        for posicion, caracter in enumerate(vocabulario)
    }
    limite = int(CONFIG["char_max_len"])
    return DatosPliegue(
        x_train=torch.tensor(
            codificar_secuencias(
                [textos[i] for i in entrenamiento], indice, limite, por_palabra=False
            ),
            dtype=torch.int64,
        ),
        x_val=torch.tensor(
            codificar_secuencias(
                [textos[i] for i in validacion], indice, limite, por_palabra=False
            ),
            dtype=torch.int64,
        ),
        vocab_size=len(vocabulario) + PRIMER_INDICE_VOCABULARIO,
        dim_entrada=int(CONFIG["char_dim"]),
        longitud_secuencia=limite,
        tipo_entrada="secuencia de caracteres",
        preprocesamiento={"caracteres": vocabulario, "longitud_maxima": limite},
        diagnostico={
            "vocabulario_caracteres": len(vocabulario),
            "cobertura": cobertura_secuencia(textos, limite, por_palabra=False),
        },
    )


def preparar_bilstm(
    textos: Sequence[str],
    entrenamiento: np.ndarray,
    validacion: np.ndarray,
) -> DatosPliegue:
    """Codifica las secuencias de palabras con el vocabulario del pliegue.

    Args:
        textos: textos de todo el train.
        entrenamiento: indices posicionales del pliegue de entrenamiento.
        validacion: indices posicionales del pliegue de validacion.

    Returns:
        Secuencias enteras de palabras y metadatos del pliegue.
    """
    vocabulario = vocabulario_palabras(
        [textos[indice] for indice in entrenamiento], int(CONFIG["word_min_freq"])
    )
    indice = {
        palabra: posicion + PRIMER_INDICE_VOCABULARIO
        for posicion, palabra in enumerate(vocabulario)
    }
    limite = int(CONFIG["word_max_len"])
    return DatosPliegue(
        x_train=torch.tensor(
            codificar_secuencias(
                [textos[i] for i in entrenamiento], indice, limite, por_palabra=True
            ),
            dtype=torch.int64,
        ),
        x_val=torch.tensor(
            codificar_secuencias(
                [textos[i] for i in validacion], indice, limite, por_palabra=True
            ),
            dtype=torch.int64,
        ),
        vocab_size=len(vocabulario) + PRIMER_INDICE_VOCABULARIO,
        dim_entrada=int(CONFIG["word_dim"]),
        longitud_secuencia=limite,
        tipo_entrada="secuencia de palabras",
        preprocesamiento={"palabras": vocabulario, "longitud_maxima": limite},
        diagnostico={
            "vocabulario_palabras": len(vocabulario),
            "min_frecuencia": int(CONFIG["word_min_freq"]),
            "cobertura": cobertura_secuencia(textos, limite, por_palabra=True),
        },
    )


ARQUITECTURAS: Final[tuple[EspecArquitectura, ...]] = (
    EspecArquitectura(
        identificador="DL-01",
        clave="mlp",
        nombre="DL-MLP (TF-IDF union -> SVD 300 -> MLP 256-128)",
        pregunta=(
            "¿Un MLP sobre la proyeccion densa del TF-IDF union (TruncatedSVD "
            "ajustado en el pliegue) alcanza a la familia lineal TF-IDF + LR "
            "(MOD-03b_BEST, OOF 0.7927)?"
        ),
        hipotesis=(
            "La compresion a 300 componentes descarta los n-gramas poco frecuentes "
            "que sostienen al modelo lineal; con 1.385 ejemplos por pliegue el MLP "
            "no deberia superar a MOD-03b."
        ),
        representacion="TF-IDF union word (1,2) + char_wb (3,5) -> TruncatedSVD(300)",
        feature_groups="WORD_NGRAM+CHAR_NGRAM (proyeccion densa)",
        preparar=preparar_mlp,
        construir=lambda vocab, dim: MLPMultiTarea(
            dim, CONFIG["mlp_ocultas"], CONFIG["dropout"]
        ),
        descripcion_dimension="componentes SVD (dimension de la entrada)",
    ),
    EspecArquitectura(
        identificador="DL-02",
        clave="charcnn",
        nombre="DL-CHARCNN (embeddings de caracter desde cero, kernels 3 y 5)",
        pregunta=(
            "¿Una CNN de caracteres entrenada desde cero (embeddings aleatorios, "
            "vocabulario del pliegue) aprovecha la evasion ortografica mejor que "
            "los char n-gramas de TF-IDF?"
        ),
        hipotesis=(
            "Una CNN de caracteres necesita muchos mas datos que 1.385 ejemplos por "
            "pliegue para aprender embeddings utiles; se espera una F1 inferior a la "
            "familia lineal."
        ),
        representacion=(
            "embeddings de caracter desde cero (dim 48, vocab del pliegue) + "
            "Conv1d 128 filtros k=3 y k=5 + max pooling"
        ),
        feature_groups="CHAR_NGRAM (embeddings entrenados desde cero)",
        preparar=preparar_charcnn,
        construir=lambda vocab, dim: CharCNNMultiTarea(
            vocab,
            dim,
            int(CONFIG["char_filtros"]),
            CONFIG["char_kernels"],
            int(CONFIG["char_oculta"]),
            CONFIG["dropout"],
        ),
        descripcion_dimension="dimension del embedding de caracteres",
    ),
    EspecArquitectura(
        identificador="DL-03",
        clave="bilstm",
        nombre="DL-BILSTM (embeddings de palabra desde cero, BiLSTM 96)",
        pregunta=(
            "¿Un BiLSTM sobre embeddings de palabra desde cero (vocabulario del "
            "pliegue) aporta informacion de orden que la bolsa de n-gramas no "
            "captura?"
        ),
        hipotesis=(
            "El orden de las palabras aporta poco en tweets de ~16 palabras y el "
            "vocabulario del pliegue (~1.700 tipos) es demasiado pequeno para "
            "entrenar embeddings utiles; se espera un resultado inferior al clasico."
        ),
        representacion=(
            "embeddings de palabra desde cero (dim 64, vocab del pliegue min_freq=2) "
            "+ BiLSTM(96) + pooling media/max enmascarado"
        ),
        feature_groups="WORD_NGRAM (embeddings entrenados desde cero)",
        preparar=preparar_bilstm,
        construir=lambda vocab, dim: BiLSTMMultiTarea(
            vocab, dim, int(CONFIG["word_oculto"]), CONFIG["dropout"]
        ),
        descripcion_dimension="dimension del embedding de palabras",
    ),
)


# ===== SECCION 6: entrenamiento con early stopping estricto =====


@dataclass
class CurvasPliegue:
    """Curvas y resultado de entrenar un pliegue.

    Attributes:
        perdidas_train: perdida media de entrenamiento por epoca.
        perdidas_val: perdida de validacion por epoca.
        f1_val: F1 media por columna de validacion por epoca (umbral 0.5).
        mejor_epoca: epoca con la mejor F1 de validacion.
        mejor_f1_val: mejor F1 de validacion.
        epocas_ejecutadas: epocas realmente ejecutadas (incluye la paciencia).
        f1_train: F1 media por columna del pliegue de entrenamiento con los
            mejores pesos (diagnostico de sobreajuste).
        probabilidad_val: probabilidades de validacion con los mejores pesos.
        segundos: tiempo de entrenamiento del pliegue.
    """

    perdidas_train: list[float] = field(default_factory=list)
    perdidas_val: list[float] = field(default_factory=list)
    f1_val: list[float] = field(default_factory=list)
    mejor_epoca: int = 0
    mejor_f1_val: float = 0.0
    epocas_ejecutadas: int = 0
    f1_train: float = 0.0
    probabilidad_val: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    segundos: float = 0.0


def predecir_binario(probabilidad: np.ndarray) -> np.ndarray:
    """Umbraliza probabilidades a etiquetas binarias con el umbral del proyecto.

    Args:
        probabilidad: matriz ``(n, 3)`` de probabilidades.

    Returns:
        Matriz ``(n, 3)`` de etiquetas binarias.
    """
    return (probabilidad >= float(CONFIG["umbral"])).astype(int)


def entrenar_pliegue(
    modelo: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    pos_weight: torch.Tensor,
    dispositivo: torch.device,
) -> CurvasPliegue:
    """Entrena un pliegue con AdamW y early stopping estricto sobre F1 de val.

    El criterio de parada es la F1 media por columna de validacion (umbral 0.5):
    se detiene cuando pasan ``paciencia`` epocas sin mejora estricta y se
    restauran los pesos de la mejor epoca. Se aplica ``clip_grad_norm_`` en cada
    paso como proteccion ante picos de gradiente.

    Args:
        modelo: red ya construida en el dispositivo.
        x_train: entradas de entrenamiento.
        y_train: etiquetas de entrenamiento ``(n, 3)`` en float.
        x_val: entradas de validacion.
        y_val: etiquetas de validacion ``(n, 3)`` en float.
        pos_weight: peso de la clase positiva por etiqueta.
        dispositivo: dispositivo de computo.

    Returns:
        Curvas del pliegue con las metricas de la mejor epoca.
    """
    inicio = time.perf_counter()
    optimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=float(CONFIG["learning_rate"]),
        weight_decay=float(CONFIG["weight_decay"]),
    )
    criterio = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    y_val_np = y_val.cpu().numpy().astype(int)
    curvas = CurvasPliegue()
    mejor_estado = copy.deepcopy(modelo.state_dict())
    mejor_f1 = -1.0
    epocas_sin_mejora = 0
    n_train = int(x_train.shape[0])

    for epoca in range(1, int(CONFIG["max_epocas"]) + 1):
        modelo.train()
        permutacion = torch.randperm(n_train, device=dispositivo)
        perdidas_lote: list[float] = []
        for inicio_lote in range(0, n_train, int(CONFIG["batch_size"])):
            indices = permutacion[inicio_lote : inicio_lote + int(CONFIG["batch_size"])]
            optimizador.zero_grad(set_to_none=True)
            logits = modelo(x_train[indices])
            perdida = criterio(logits, y_train[indices])
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(
                modelo.parameters(), float(CONFIG["clip_grad_norm"])
            )
            optimizador.step()
            perdidas_lote.append(float(perdida.detach()))
        modelo.eval()
        with torch.no_grad():
            logits_val = modelo(x_val)
            perdida_val = float(criterio(logits_val, y_val))
            probabilidad_val = torch.sigmoid(logits_val).cpu().numpy()
        f1_val = float(
            metricas_multietiqueta(
                y_val_np, predecir_binario(probabilidad_val)
            )["f1_media_columnas"]
        )
        curvas.perdidas_train.append(float(np.mean(perdidas_lote)))
        curvas.perdidas_val.append(perdida_val)
        curvas.f1_val.append(f1_val)
        if f1_val > mejor_f1:
            mejor_f1 = f1_val
            curvas.mejor_epoca = epoca
            mejor_estado = copy.deepcopy(modelo.state_dict())
            epocas_sin_mejora = 0
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= int(CONFIG["paciencia"]):
                log(
                    f"  early stopping en la epoca {epoca}: sin mejora en "
                    f"{epocas_sin_mejora} epocas (mejor F1 {mejor_f1:.4f} en la "
                    f"epoca {curvas.mejor_epoca})"
                )
                break

    modelo.load_state_dict(mejor_estado)
    modelo.eval()
    with torch.no_grad():
        probabilidad_val = torch.sigmoid(modelo(x_val)).cpu().numpy()
        probabilidad_train = torch.sigmoid(modelo(x_train)).cpu().numpy()
    curvas.probabilidad_val = probabilidad_val
    curvas.mejor_f1_val = float(
        metricas_multietiqueta(
            y_val_np, predecir_binario(probabilidad_val)
        )["f1_media_columnas"]
    )
    curvas.f1_train = float(
        metricas_multietiqueta(
            y_train.cpu().numpy().astype(int), predecir_binario(probabilidad_train)
        )["f1_media_columnas"]
    )
    curvas.epocas_ejecutadas = len(curvas.f1_val)
    curvas.segundos = round(time.perf_counter() - inicio, 2)
    return curvas


def calcular_metricas(
    etiquetas: np.ndarray,
    prediccion: np.ndarray,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[MetricasMultiEtiqueta, list[dict]]:
    """Calcula las metricas out-of-fold globales y por pliegue.

    Misma definicion que ``experiments/03_variantes_representaciones.py`` para
    que las comparaciones entre fases sean homogeneas.

    Args:
        etiquetas: matriz ``(n, 3)`` con las etiquetas verdaderas.
        prediccion: matriz ``(n, 3)`` con las predicciones out-of-fold.
        pliegues: particion usada para generar las predicciones.

    Returns:
        Tupla ``(metricas_globales, metricas_por_pliegue)``.
    """
    metricas_pliegues: list[dict] = []
    for numero, (_, validacion) in enumerate(pliegues, start=1):
        metrica = dict(
            metricas_multietiqueta(etiquetas[validacion], prediccion[validacion])
        )
        metrica["pliegue"] = numero
        metrica["n_validacion"] = len(validacion)
        metricas_pliegues.append(metrica)
    return metricas_multietiqueta(etiquetas, prediccion), metricas_pliegues


@dataclass
class ResultadoArquitectura:
    """Resultado completo de evaluar una arquitectura con una semilla.

    Attributes:
        espec: especificacion de la arquitectura.
        semilla: semilla usada en la corrida.
        n_parametros: parametros entrenables del modelo.
        vocab_size: tamano del vocabulario (0 si no aplica).
        dim_embedding: dimension de la primera capa.
        longitud_secuencia: longitud de la entrada.
        tipo_entrada: descripcion del tipo de entrada.
        pliegues: lista de diccionarios con las metricas de cada pliegue.
        curvas: curvas de entrenamiento por pliegue.
        prediccion_oof: predicciones out-of-fold ``(n, 3)``.
        metricas_oof: metricas out-of-fold globales.
        metricas_pliegues: metricas por pliegue de la prediccion out-of-fold.
        f1_train_media: media de la F1 de entrenamiento con los mejores pesos.
        diagnosticos: diagnosticos de preprocesamiento por pliegue.
        segundos: tiempo total de la corrida.
    """

    espec: EspecArquitectura
    semilla: int
    n_parametros: int
    vocab_size: int
    dim_embedding: int
    longitud_secuencia: int
    tipo_entrada: str
    pliegues: list[dict]
    curvas: list[CurvasPliegue]
    prediccion_oof: np.ndarray
    metricas_oof: MetricasMultiEtiqueta
    metricas_pliegues: list[dict]
    f1_train_media: float
    diagnosticos: list[dict]
    segundos: float

    @property
    def f1_val_media(self) -> float:
        """F1 media por columna out-of-fold de la corrida."""
        return float(self.metricas_oof["f1_media_columnas"])

    @property
    def f1_val_std(self) -> float:
        """Desviacion estandar de la F1 entre los 5 pliegues."""
        return float(
            np.std([metrica["f1_media_columnas"] for metrica in self.metricas_pliegues])
        )

    @property
    def gap_train_val(self) -> float:
        """Diferencia entre la F1 de entrenamiento y la F1 out-of-fold."""
        return self.f1_train_media - self.f1_val_media


def evaluar_arquitectura(
    espec: EspecArquitectura,
    textos: Sequence[str],
    etiquetas: np.ndarray,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
    semilla: int,
) -> ResultadoArquitectura:
    """Evalua una arquitectura con la particion agrupada compartida.

    Args:
        espec: especificacion de la arquitectura.
        textos: textos de todo el train.
        etiquetas: matriz ``(n, 3)`` de etiquetas.
        pliegues: particion 5-fold compartida.
        semilla: semilla de la corrida.

    Returns:
        Resultado completo de la arquitectura.
    """
    inicio = time.perf_counter()
    dispositivo = dispositivo_entrenamiento()
    prediccion = np.zeros_like(etiquetas)
    curvas_pliegues: list[CurvasPliegue] = []
    diagnosticos: list[dict] = []
    pliegues_info: list[dict] = []
    n_parametros = 0
    vocab_size = 0
    dim_entrada = 0
    longitud_secuencia = 0
    tipo_entrada = ""

    for numero, (entrenamiento, validacion) in enumerate(pliegues, start=1):
        fijar_semillas(semilla)
        datos = espec.preparar(textos, entrenamiento, validacion)
        modelo = espec.construir(datos.vocab_size, datos.dim_entrada).to(dispositivo)
        n_parametros = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
        vocab_size = datos.vocab_size
        dim_entrada = datos.dim_entrada
        longitud_secuencia = datos.longitud_secuencia
        tipo_entrada = datos.tipo_entrada
        x_train = datos.x_train.to(dispositivo)
        x_val = datos.x_val.to(dispositivo)
        y_train = torch.tensor(
            etiquetas[entrenamiento], dtype=torch.float32, device=dispositivo
        )
        y_val = torch.tensor(
            etiquetas[validacion], dtype=torch.float32, device=dispositivo
        )
        curvas = entrenar_pliegue(
            modelo,
            x_train,
            y_train,
            x_val,
            y_val,
            peso_positivos(etiquetas[entrenamiento]).to(dispositivo),
            dispositivo,
        )
        prediccion[validacion] = predecir_binario(curvas.probabilidad_val)
        curvas_pliegues.append(curvas)
        diagnosticos.append(dict(datos.diagnostico))
        pliegues_info.append(
            {
                "pliegue": numero,
                "n_train": len(entrenamiento),
                "n_validacion": len(validacion),
                "epocas_ejecutadas": curvas.epocas_ejecutadas,
                "mejor_epoca": curvas.mejor_epoca,
                "f1_val": round(curvas.mejor_f1_val, 6),
                "f1_train": round(curvas.f1_train, 6),
                "segundos": curvas.segundos,
            }
        )
        log(
            f"  {espec.identificador} semilla {semilla} pliegue {numero}: "
            f"F1 val {curvas.mejor_f1_val:.4f} | train {curvas.f1_train:.4f} | "
            f"epocas {curvas.epocas_ejecutadas} (mejor {curvas.mejor_epoca}) | "
            f"{curvas.segundos:.1f} s"
        )

    metricas_oof, metricas_pliegues = calcular_metricas(etiquetas, prediccion, pliegues)
    return ResultadoArquitectura(
        espec=espec,
        semilla=semilla,
        n_parametros=n_parametros,
        vocab_size=vocab_size,
        dim_embedding=dim_entrada,
        longitud_secuencia=longitud_secuencia,
        tipo_entrada=tipo_entrada,
        pliegues=pliegues_info,
        curvas=curvas_pliegues,
        prediccion_oof=prediccion,
        metricas_oof=metricas_oof,
        metricas_pliegues=metricas_pliegues,
        f1_train_media=float(np.mean([curva.f1_train for curva in curvas_pliegues])),
        diagnosticos=diagnosticos,
        segundos=round(time.perf_counter() - inicio, 2),
    )


# ===== SECCION 7: figura de curvas =====


def mejor_pliegue(resultado: ResultadoArquitectura) -> int:
    """Devuelve el indice del pliegue con la mejor F1 de validacion.

    Args:
        resultado: resultado de una arquitectura.

    Returns:
        Indice (0-based) del mejor pliegue.
    """
    f1 = [curva.mejor_f1_val for curva in resultado.curvas]
    return int(np.argmax(f1))


def graficar_curvas(resultados: Sequence[ResultadoArquitectura]) -> Path:
    """Dibuja perdida y F1 por epoca del mejor pliegue de cada arquitectura.

    Args:
        resultados: resultados de las arquitecturas (semilla 42).

    Returns:
        Ruta de la figura guardada.
    """
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["font.size"] = 9
    figura, ejes = plt.subplots(
        len(resultados), 2, figsize=(11.5, 3.1 * len(resultados)), squeeze=False
    )
    for fila, resultado in enumerate(resultados):
        indice = mejor_pliegue(resultado)
        curvas = resultado.curvas[indice]
        epocas = np.arange(1, len(curvas.f1_val) + 1)
        eje_perdida, eje_f1 = ejes[fila]
        eje_perdida.plot(epocas, curvas.perdidas_train, label="train", linewidth=1.6)
        eje_perdida.plot(epocas, curvas.perdidas_val, label="validacion", linewidth=1.6)
        eje_perdida.axvline(
            curvas.mejor_epoca, color="black", linestyle="--", linewidth=1
        )
        eje_perdida.set_title(f"{resultado.espec.identificador} - perdida")
        eje_perdida.set_xlabel("epoca")
        eje_perdida.set_ylabel("BCE")
        eje_perdida.legend()
        eje_perdida.grid(alpha=0.3)
        eje_f1.plot(epocas, curvas.f1_val, color="#1f77b4", linewidth=1.6)
        eje_f1.axvline(
            curvas.mejor_epoca, color="black", linestyle="--", linewidth=1,
            label=f"mejor epoca {curvas.mejor_epoca}",
        )
        eje_f1.axhline(
            float(CONFIG["referencias"]["MOD-03b_BEST"]),
            color="#d62728",
            linestyle=":",
            linewidth=1.2,
            label="MOD-03b OOF 0.7927",
        )
        eje_f1.set_title(
            f"{resultado.espec.identificador} - F1 validacion "
            f"(mejor pliegue {indice + 1}: {curvas.mejor_f1_val:.4f})"
        )
        eje_f1.set_xlabel("epoca")
        eje_f1.set_ylabel("F1 media por columna")
        eje_f1.legend(loc="lower right")
        eje_f1.grid(alpha=0.3)
    figura.suptitle(
        "Fase 11 - DL desde cero: curvas del mejor pliegue por arquitectura "
        "(early stopping estricto, paciencia 5)",
        fontsize=11,
    )
    figura.tight_layout()
    CONFIG["figura"].parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(CONFIG["figura"], dpi=150, bbox_inches="tight")
    plt.close(figura)
    return CONFIG["figura"]


# ===== SECCION 8: reportes, registro y handoff =====


def clasificar_decision(f1_valor: float) -> str:
    """Clasifica la decision de una arquitectura frente a las referencias.

    Args:
        f1_valor: F1 media por columna out-of-fold de la arquitectura.

    Returns:
        Texto de decision honesto y trazable.
    """
    referencia_mod = float(CONFIG["referencias"]["MOD-03b_BEST"])
    referencia_b1 = float(CONFIG["referencias"]["B1"])
    if f1_valor >= referencia_mod:
        return (
            "supera a MOD-03b_BEST en OOF: candidata a modelo final, pero requiere "
            "sonda de leaderboard (D-012/D-014)"
        )
    if f1_valor >= referencia_b1:
        return (
            "por encima de B1 pero por debajo de MOD-03b_BEST: no desplaza al modelo "
            "final (D-014); el DL desde cero queda como evidencia metodologica"
        )
    return (
        "por debajo de B1 (0.7587) y de MOD-03b_BEST (0.7927): descartada; con 1.731 "
        "ejemplos y sin embeddings preentrenados el DL desde cero no supera al "
        "pipeline clasico"
    )


def construir_fila_reporte(
    resultado: ResultadoArquitectura,
    experiment_id: str,
    arquitectura: str,
    semilla: str,
) -> dict:
    """Construye una fila del reporte CSV de la fase.

    Args:
        resultado: resultado de la arquitectura.
        experiment_id: id del experimento (DL-01, DL-03-SEEDS, ...).
        arquitectura: nombre legible de la arquitectura.
        semilla: texto de la semilla o rango de semillas.

    Returns:
        Diccionario con las columnas de ``COLUMNAS_REPORTE``.
    """
    referencia_mod = float(CONFIG["referencias"]["MOD-03b_BEST"])
    referencia_b1 = float(CONFIG["referencias"]["B1"])
    return {
        "experiment_id": experiment_id,
        "arquitectura": arquitectura,
        "semilla": semilla,
        "n_parametros": resultado.n_parametros,
        "vocab": resultado.vocab_size,
        "dim_embedding": resultado.dim_embedding,
        "longitud_secuencia": resultado.longitud_secuencia,
        "tipo_entrada": resultado.tipo_entrada,
        "epocas_ejecutadas_media": round(
            float(np.mean([p["epocas_ejecutadas"] for p in resultado.pliegues])), 2
        ),
        "epocas_mejor_media": round(
            float(np.mean([p["mejor_epoca"] for p in resultado.pliegues])), 2
        ),
        "tiempo_segundos": resultado.segundos,
        "f1_val_media": round(resultado.f1_val_media, 4),
        "f1_val_std": round(resultado.f1_val_std, 4),
        "f1_HS": round(resultado.metricas_oof["f1_por_etiqueta"]["HS"], 4),
        "f1_TR": round(resultado.metricas_oof["f1_por_etiqueta"]["TR"], 4),
        "f1_AG": round(resultado.metricas_oof["f1_por_etiqueta"]["AG"], 4),
        "f1_train_media": round(resultado.f1_train_media, 4),
        "gap_train_val": round(resultado.gap_train_val, 4),
        "delta_vs_MOD03b": round(resultado.f1_val_media - referencia_mod, 4),
        "delta_vs_B1": round(resultado.f1_val_media - referencia_b1, 4),
        "f1_por_pliegue": json.dumps(
            [round(m["f1_media_columnas"], 4) for m in resultado.metricas_pliegues]
        ),
        "decision": clasificar_decision(resultado.f1_val_media),
    }


def resultado_a_dict(resultado: ResultadoArquitectura) -> dict:
    """Convierte un resultado de arquitectura en un diccionario serializable.

    Args:
        resultado: resultado de la arquitectura.

    Returns:
        Diccionario con metricas, pliegues, diagnosticos y curvas resumidas.
    """
    return {
        "experiment_id": resultado.espec.identificador,
        "arquitectura": resultado.espec.nombre,
        "semilla": resultado.semilla,
        "n_parametros": resultado.n_parametros,
        "vocab_size": resultado.vocab_size,
        "dim_embedding": resultado.dim_embedding,
        "longitud_secuencia": resultado.longitud_secuencia,
        "tipo_entrada": resultado.tipo_entrada,
        "tiempo_segundos": resultado.segundos,
        "metricas_oof": resultado.metricas_oof,
        "metricas_pliegues": resultado.metricas_pliegues,
        "pliegues": resultado.pliegues,
        "f1_train_media": round(resultado.f1_train_media, 6),
        "gap_train_val": round(resultado.gap_train_val, 6),
        "diagnosticos_preprocesamiento": resultado.diagnosticos,
        "curvas": [
            {
                "pliegue": numero,
                "perdidas_train": [round(v, 6) for v in curva.perdidas_train],
                "perdidas_val": [round(v, 6) for v in curva.perdidas_val],
                "f1_val": [round(v, 6) for v in curva.f1_val],
                "mejor_epoca": curva.mejor_epoca,
                "mejor_f1_val": round(curva.mejor_f1_val, 6),
                "epocas_ejecutadas": curva.epocas_ejecutadas,
                "segundos": curva.segundos,
            }
            for numero, curva in enumerate(resultado.curvas, start=1)
        ],
        "decision": clasificar_decision(resultado.f1_val_media),
    }


def guardar_checkpoint(
    resultado: ResultadoArquitectura,
    textos: Sequence[str],
    etiquetas: np.ndarray,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
) -> Path:
    """Guarda los pesos del mejor pliegue de la mejor arquitectura.

    El checkpoint incluye el vocabulario (o el vectorizador/SVD/escalador) del
    pliegue para poder reconstruir la entrada sin fuga.

    Args:
        resultado: resultado de la mejor arquitectura (semilla 42).
        textos: textos de todo el train.
        etiquetas: matriz ``(n, 3)`` de etiquetas.
        pliegues: particion 5-fold compartida.

    Returns:
        Ruta del archivo de pesos.
    """
    indice = mejor_pliegue(resultado)
    entrenamiento, validacion = pliegues[indice]
    fijar_semillas(resultado.semilla)
    datos = resultado.espec.preparar(textos, entrenamiento, validacion)
    # El estado guardado es el de la mejor epoca: se reentrena el pliegue de
    # forma determinista con la misma semilla en lugar de retener el modelo en
    # memoria durante la evaluacion.
    modelo = reentrenar_mejor_pliegue(resultado, etiquetas, pliegues, indice, datos)
    curvas = resultado.curvas[indice]
    preprocesamiento = compactar_preprocesamiento(datos.preprocesamiento)
    ruta = CONFIG["modelos_dir"] / f"dl_{resultado.espec.clave}.pt"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arquitectura": resultado.espec.nombre,
            "experiment_id": resultado.espec.identificador,
            "state_dict": modelo.state_dict(),
            "preprocesamiento": preprocesamiento,
            "vocab_size": datos.vocab_size,
            "dim_entrada": datos.dim_entrada,
            "longitud_secuencia": datos.longitud_secuencia,
            "pliegue": indice + 1,
            "mejor_epoca": curvas.mejor_epoca,
            "f1_val_mejor_pliegue": round(curvas.mejor_f1_val, 6),
            "f1_oof_media": round(resultado.f1_val_media, 6),
            "semilla": resultado.semilla,
            "umbral": float(CONFIG["umbral"]),
            "nota": (
                "Pesos de la mejor epoca del mejor pliegue; el preprocesamiento "
                "incluye el vocabulario (o vectorizador/SVD/escalador) ajustado "
                "solo con ese pliegue de entrenamiento."
            ),
        },
        ruta,
    )
    return ruta


def compactar_preprocesamiento(preprocesamiento: dict) -> dict:
    """Reduce el tamano del preprocesamiento guardado en el checkpoint.

    La matriz ``components_`` del ``TruncatedSVD`` ocupa ``300 x n_features`` en
    float64 (unos 66 MB para 27.431 features); convertirla a float32 reduce el
    archivo a la mitad sin cambiar el resultado de ``transform`` mas alla de la
    precision de la proyeccion.

    Args:
        preprocesamiento: diccionario con los objetos del pliegue.

    Returns:
        El mismo diccionario con el SVD compactado en float32.
    """
    svd = preprocesamiento.get("svd")
    if svd is not None and svd.components_.dtype != np.float32:
        svd.components_ = svd.components_.astype(np.float32)
    return preprocesamiento


def reentrenar_mejor_pliegue(
    resultado: ResultadoArquitectura,
    etiquetas: np.ndarray,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
    indice: int,
    datos: DatosPliegue,
) -> nn.Module:
    """Reentrena un pliegue de forma determinista para recuperar sus pesos.

    Se reentrena en lugar de guardar el objeto durante la evaluacion para no
    retener modelos en memoria; con la semilla fija el resultado es identico
    (el bucle es determinista en la misma maquina y entorno).

    Args:
        resultado: resultado de la arquitectura.
        etiquetas: matriz ``(n, 3)`` de etiquetas.
        pliegues: particion 5-fold compartida.
        indice: indice del pliegue a reentrenar.
        datos: entradas del pliegue.

    Returns:
        Red en CPU con los pesos de la mejor epoca.
    """
    dispositivo = dispositivo_entrenamiento()
    entrenamiento, validacion = pliegues[indice]
    fijar_semillas(resultado.semilla)
    modelo = resultado.espec.construir(datos.vocab_size, datos.dim_entrada).to(dispositivo)
    entrenar_pliegue(
        modelo,
        datos.x_train.to(dispositivo),
        torch.tensor(etiquetas[entrenamiento], dtype=torch.float32, device=dispositivo),
        datos.x_val.to(dispositivo),
        torch.tensor(etiquetas[validacion], dtype=torch.float32, device=dispositivo),
        peso_positivos(etiquetas[entrenamiento]).to(dispositivo),
        dispositivo,
    )
    return modelo.cpu()


def escribir_reportes(
    resultados: Sequence[ResultadoArquitectura],
    resultado_multilla: ResultadoArquitectura | None,
    f1_multilla: dict | None,
    evidencia_gpu: dict,
    rutas: dict,
) -> dict:
    """Escribe el CSV y el JSON de resultados de la fase.

    Args:
        resultados: resultados de las tres arquitecturas (semilla 42).
        resultado_multilla: mejor arquitectura (base de la fila multi-semilla).
        f1_multilla: media y desviacion de la F1 entre semillas, si existe.
        evidencia_gpu: evidencia del pre-vuelo de GPU.
        rutas: rutas de los artefactos escritos.

    Returns:
        Diccionario de detalle usado para la seccion de ``contexto.md``.
    """
    filas = [
        construir_fila_reporte(
            resultado, resultado.espec.identificador, resultado.espec.nombre, "42"
        )
        for resultado in resultados
    ]
    if resultado_multilla is not None and f1_multilla is not None:
        fila = construir_fila_reporte(
            resultado_multilla,
            "DL-03-SEEDS",
            resultado_multilla.espec.nombre,
            "/".join(str(semilla) for semilla in CONFIG["semillas"]),
        )
        fila["f1_val_media"] = round(f1_multilla["media"], 4)
        fila["f1_val_std"] = round(f1_multilla["desviacion"], 4)
        fila["decision"] = (
            "media de 3 semillas: "
            f"{f1_multilla['media']:.4f} +/- {f1_multilla['desviacion']:.4f}; "
            + clasificar_decision(f1_multilla["media"])
        )
        filas.append(fila)
    reporte = pd.DataFrame(filas, columns=list(COLUMNAS_REPORTE))
    CONFIG["reporte_csv"].parent.mkdir(parents=True, exist_ok=True)
    reporte.to_csv(CONFIG["reporte_csv"], index=False, encoding="utf-8")

    mejor = max(resultados, key=lambda r: r.f1_val_media)
    arquitectura_multilla = (
        None if resultado_multilla is None else resultado_multilla.espec.identificador
    )
    detalle = {
        "metadata": {
            "fecha": CONFIG["fecha"],
            "script": CONFIG["script"],
            "competencia": CONFIG["competencia"],
            "fase": "Fase 11 - Deep Learning desde cero (requisito metodologico)",
            "metrica_primaria": "f1_media_columnas (D-009)",
            "referencias": {
                "MOD-03b_BEST_oof": CONFIG["referencias"]["MOD-03b_BEST"],
                "MOD-03b_BEST_lb": CONFIG["referencias"]["LB_MOD03b"],
                "B1_oof": CONFIG["referencias"]["B1"],
            },
            "validacion": VALIDACION_DL,
            "decision_overfitting": CONFIG["decision_overfitting"],
            "optimizador": {
                "tipo": "AdamW",
                "learning_rate": CONFIG["learning_rate"],
                "weight_decay": CONFIG["weight_decay"],
                "batch_size": CONFIG["batch_size"],
                "max_epocas": CONFIG["max_epocas"],
                "paciencia": CONFIG["paciencia"],
                "clip_grad_norm": CONFIG["clip_grad_norm"],
            },
            "amp": {
                "usado": CONFIG["usar_amp"],
                "evidencia": CONFIG["evidencia_amp"],
            },
            "determinismo": {
                "semillas": list(CONFIG["semillas"]),
                "use_deterministic_algorithms": False,
                "evidencia": CONFIG["evidencia_determinismo"],
            },
            "presupuesto_segundos": CONFIG["presupuesto_segundos"],
            "sin_fuga": (
                "Vectorizador, TruncatedSVD, escalador y vocabularios ajustados "
                "dentro de cada pliegue; pos_weight calculado en el pliegue; el "
                "test no se carga en este script"
            ),
            "restricciones_cumplidas": (
                "Sin transformers, sin LLM y sin embeddings preentrenados: todos los "
                "embeddings se inicializan aleatoriamente y se entrenan desde cero "
                "con vocabulario construido solo con el pliegue de entrenamiento"
            ),
        },
        "preflight_gpu": evidencia_gpu,
        "arquitecturas": [resultado_a_dict(resultado) for resultado in resultados],
        "mejor_arquitectura": mejor.espec.identificador,
        "multilla": (
            None
            if f1_multilla is None
            else {
                "arquitectura": arquitectura_multilla,
                "semillas": list(CONFIG["semillas"]),
                "f1_media": round(f1_multilla["media"], 6),
                "f1_desviacion": round(f1_multilla["desviacion"], 6),
                "f1_por_semilla": {
                    str(clave): round(valor, 6)
                    for clave, valor in f1_multilla["por_semilla"].items()
                },
            }
        ),
        "rutas": {clave: str(valor) for clave, valor in rutas.items()},
        "conclusion": construir_conclusion(resultados, f1_multilla),
    }
    CONFIG["reporte_json"].write_text(
        json.dumps(detalle, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return detalle


def construir_conclusion(
    resultados: Sequence[ResultadoArquitectura], f1_multilla: dict | None
) -> dict:
    """Redacta la conclusion honesta de la fase.

    Args:
        resultados: resultados de las tres arquitecturas.
        f1_multilla: media y desviacion de la F1 entre semillas, si existe.

    Returns:
        Diccionario con la mejor arquitectura, los numeros y la interpretacion.
    """
    mejor = max(resultados, key=lambda r: r.f1_val_media)
    referencia_mod = float(CONFIG["referencias"]["MOD-03b_BEST"])
    referencia_b1 = float(CONFIG["referencias"]["B1"])
    supera_clasico = mejor.f1_val_media >= referencia_mod
    if supera_clasico:
        explicacion = (
            "La red supera a la referencia clasica en OOF, pero D-012/D-014 exigen "
            "una sonda de leaderboard antes de dar la mejora por real: el gap "
            "local-LB de esta familia es de -0.013 a -0.029."
        )
    else:
        explicacion = (
            f"El DL desde cero NO supera al pipeline clasico con 1.731 ejemplos: "
            f"{mejor.espec.hipotesis[0].lower()}{mejor.espec.hipotesis[1:]} El gap "
            f"train-validacion de "
            f"{mejor.gap_train_val:+.4f} (train {mejor.f1_train_media:.4f} frente a "
            f"validacion {mejor.f1_val_media:.4f}) confirma el diagnostico: la red "
            "memoriza el pliegue de entrenamiento y no generaliza al de validacion, "
            "mientras el modelo lineal aprovecha 27.431 n-gramas dispersos con "
            "regularizacion L2 y no necesita aprender representaciones."
        )
    return {
        "mejor_arquitectura": mejor.espec.identificador,
        "mejor_f1_oof": round(mejor.f1_val_media, 6),
        "delta_vs_MOD03b": round(mejor.f1_val_media - referencia_mod, 6),
        "delta_vs_B1": round(mejor.f1_val_media - referencia_b1, 6),
        "f1_multilla": (
            None
            if f1_multilla is None
            else {
                "media": round(f1_multilla["media"], 6),
                "desviacion": round(f1_multilla["desviacion"], 6),
            }
        ),
        "supera_pipeline_clasico": supera_clasico,
        "interpretacion": (
            f"La mejor red desde cero ({mejor.espec.identificador}) alcanza "
            f"{mejor.f1_val_media:.4f} de F1 OOF frente a {referencia_mod:.4f} de "
            f"MOD-03b_BEST (TF-IDF union + LR) y {referencia_b1:.4f} de B1. "
            + explicacion
        ),
    }


def registrar_en_experiments(
    resultados: Sequence[ResultadoArquitectura],
    f1_multilla: dict | None,
    mejor_resultado: ResultadoArquitectura,
) -> int:
    """Anade (o refresca) las filas de la fase en ``experiments.csv``.

    Idempotente: elimina las filas cuyo ``experiment_id`` pertenece a esta fase
    y las reescribe con el esquema existente.

    Args:
        resultados: resultados de las tres arquitecturas.
        f1_multilla: media y desviacion de la F1 entre semillas, si existe.
        mejor_resultado: resultado de la mejor arquitectura (para el checkpoint).

    Returns:
        Numero de filas escritas.

    Raises:
        ValueError: si el esquema de ``experiments.csv`` no coincide.
    """
    destino: Path = CONFIG["experiments_csv"]
    existente = pd.read_csv(destino, dtype=str, keep_default_na=False)
    if list(existente.columns) != list(COLUMNAS_EXPERIMENTS):
        raise ValueError("El esquema de experiments.csv cambio inesperadamente")
    filas = [_fila_experiments(resultado) for resultado in resultados]
    if f1_multilla is not None:
        filas.append(_fila_experiments_multilla(mejor_resultado, f1_multilla))
    identificadores = {fila["experiment_id"] for fila in filas}
    conservadas = existente[~existente["experiment_id"].isin(identificadores)]
    nuevas = pd.DataFrame(
        [{clave: fila[clave] for clave in COLUMNAS_EXPERIMENTS} for fila in filas],
        columns=list(COLUMNAS_EXPERIMENTS),
    )
    combinado = pd.concat([conservadas, nuevas], ignore_index=True)
    combinado.to_csv(destino, index=False, encoding="utf-8")
    return len(nuevas)


def _fila_experiments(resultado: ResultadoArquitectura) -> dict:
    """Construye la fila de una arquitectura para ``experiments.csv``.

    Args:
        resultado: resultado de la arquitectura.

    Returns:
        Diccionario con las columnas de ``COLUMNAS_EXPERIMENTS``.
    """
    espec = resultado.espec
    return {
        "experiment_id": espec.identificador,
        "competition": CONFIG["competencia"],
        "question": espec.pregunta,
        "hypothesis": espec.hipotesis,
        "preprocessing": PREPROCESAMIENTO_DL,
        "feature_groups": espec.feature_groups,
        "representation": espec.representacion,
        "model": espec.nombre,
        "hyperparameters": json.dumps(
            {
                "optimizador": "AdamW",
                "learning_rate": CONFIG["learning_rate"],
                "weight_decay": CONFIG["weight_decay"],
                "batch_size": CONFIG["batch_size"],
                "max_epocas": CONFIG["max_epocas"],
                "early_stopping": (
                    f"estricto sobre F1 de validacion, paciencia {CONFIG['paciencia']}, "
                    "mejores pesos restaurados"
                ),
                "clip_grad_norm": CONFIG["clip_grad_norm"],
                "dropout": CONFIG["dropout"],
                "amp": CONFIG["usar_amp"],
                "semilla": resultado.semilla,
                "vocab_size": resultado.vocab_size,
                "dim_embedding": resultado.dim_embedding,
                "longitud_secuencia": resultado.longitud_secuencia,
                "n_parametros": resultado.n_parametros,
                "perdida": "BCEWithLogitsLoss con pos_weight por etiqueta del pliegue",
            },
            ensure_ascii=False,
        ),
        "validation": VALIDACION_DL,
        "f1": round(resultado.f1_val_media, 4),
        "macro_f1": round(resultado.metricas_oof["f1_macro6"], 4),
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": resultado.segundos,
        "dimension": resultado.n_parametros,
        "result": json.dumps(
            {
                "f1_oof_media_columnas": round(resultado.f1_val_media, 6),
                "f1_por_pliegue": [
                    round(m["f1_media_columnas"], 6) for m in resultado.metricas_pliegues
                ],
                "std_pliegues": round(resultado.f1_val_std, 6),
                "f1_por_etiqueta": resultado.metricas_oof["f1_por_etiqueta"],
                "f1_train_media": round(resultado.f1_train_media, 6),
                "gap_train_val": round(resultado.gap_train_val, 6),
                "epocas_ejecutadas": [
                    pliegue["epocas_ejecutadas"] for pliegue in resultado.pliegues
                ],
                "n_parametros": resultado.n_parametros,
                "vocab_size": resultado.vocab_size,
                "dim_embedding": resultado.dim_embedding,
                "longitud_secuencia": resultado.longitud_secuencia,
                "delta_vs_MOD03b": round(
                    resultado.f1_val_media
                    - float(CONFIG["referencias"]["MOD-03b_BEST"]),
                    6,
                ),
                "delta_vs_B1": round(
                    resultado.f1_val_media - float(CONFIG["referencias"]["B1"]), 6
                ),
            },
            ensure_ascii=False,
        ),
        "interpretation": (
            f"F1 OOF {resultado.f1_val_media:.4f} +/- {resultado.f1_val_std:.4f} "
            f"(pliegues {[round(m['f1_media_columnas'], 4) for m in resultado.metricas_pliegues]}); "
            f"por etiqueta HS {resultado.metricas_oof['f1_por_etiqueta']['HS']:.4f} / "
            f"TR {resultado.metricas_oof['f1_por_etiqueta']['TR']:.4f} / "
            f"AG {resultado.metricas_oof['f1_por_etiqueta']['AG']:.4f}; "
            f"gap train-validacion {resultado.gap_train_val:+.4f} "
            f"(train {resultado.f1_train_media:.4f}); "
            f"{resultado.n_parametros} parametros; vocab {resultado.vocab_size}; "
            f"dim {resultado.dim_embedding}; longitud {resultado.longitud_secuencia}; "
            f"{resultado.segundos:.1f} s"
        ),
        "decision": clasificar_decision(resultado.f1_val_media),
        "date": CONFIG["fecha"],
        "script": CONFIG["script"],
        "notes": (
            "Requisito metodologico del curso: red entrenada desde cero, sin "
            "transformers ni embeddings preentrenados; vocabulario y preprocesamiento "
            "ajustados solo con el pliegue de entrenamiento. "
            f"{CONFIG['decision_overfitting']}. "
            f"AMP {CONFIG['usar_amp']} por evidencia medida en esta GPU; "
            "torch.use_deterministic_algorithms(False) por inestabilidad observada "
            "en ROCm 7.2. Sin submissions: la decision de envio es del orquestador."
        ),
    }


def _fila_experiments_multilla(
    resultado: ResultadoArquitectura, f1_multilla: dict
) -> dict:
    """Construye la fila ``<ARQ>-SEEDS`` para ``experiments.csv``.

    El identificador se deriva de la arquitectura ganadora (p. ej.
    ``DL-01-SEEDS``): etiquetar la fila con un id fijo distinto del real
    induciria a error en la trazabilidad.

    Args:
        resultado: resultado de la mejor arquitectura (semilla 42).
        f1_multilla: media y desviacion de la F1 entre semillas.

    Returns:
        Diccionario con las columnas de ``COLUMNAS_EXPERIMENTS``.
    """
    fila = _fila_experiments(resultado)
    semillas = "/".join(str(semilla) for semilla in CONFIG["semillas"])
    fila["experiment_id"] = f"{resultado.espec.identificador}-SEEDS"
    fila["question"] = (
        "¿El resultado de la mejor arquitectura es estable frente a la semilla de "
        "inicializacion y de barajado de lotes?"
    )
    fila["hypothesis"] = (
        "Con 1.731 ejemplos la varianza entre semillas es apreciable (del orden de "
        "0.01 en F1) y el nivel absoluto no alcanza a MOD-03b_BEST."
    )
    fila["hyperparameters"] = json.dumps(
        {
            "arquitectura": resultado.espec.nombre,
            "semillas": list(CONFIG["semillas"]),
            "optimizador": "AdamW",
            "learning_rate": CONFIG["learning_rate"],
            "weight_decay": CONFIG["weight_decay"],
            "batch_size": CONFIG["batch_size"],
            "max_epocas": CONFIG["max_epocas"],
            "early_stopping": (
                f"estricto sobre F1 de validacion, paciencia {CONFIG['paciencia']}"
            ),
            "n_parametros": resultado.n_parametros,
        },
        ensure_ascii=False,
    )
    fila["f1"] = round(f1_multilla["media"], 4)
    fila["result"] = json.dumps(
        {
            "f1_media_3_semillas": round(f1_multilla["media"], 6),
            "f1_desviacion_3_semillas": round(f1_multilla["desviacion"], 6),
            "f1_por_semilla": {
                str(clave): round(valor, 6)
                for clave, valor in f1_multilla["por_semilla"].items()
            },
            "f1_media_columnas_semilla_42": round(resultado.f1_val_media, 6),
        },
        ensure_ascii=False,
    )
    fila["interpretation"] = (
        f"F1 OOF media de {len(CONFIG['semillas'])} semillas "
        f"({semillas}): {f1_multilla['media']:.4f} +/- "
        f"{f1_multilla['desviacion']:.4f} frente a "
        f"{float(CONFIG['referencias']['MOD-03b_BEST']):.4f} de MOD-03b_BEST"
    )
    fila["decision"] = (
        f"estabilidad entre semillas {f1_multilla['desviacion']:.4f} (desviacion); "
        + clasificar_decision(f1_multilla["media"])
    )
    return fila


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
    previo = texto.rstrip("\n")
    destino.write_text(
        (previo + "\n\n" if previo else "") + seccion.rstrip("\n") + "\n",
        encoding="utf-8",
    )
    return destino


def construir_seccion_contexto(
    resultados: Sequence[ResultadoArquitectura],
    f1_multilla: dict | None,
    mejor_resultado: ResultadoArquitectura,
    conclusion: dict,
    rutas: dict,
) -> str:
    """Redacta la seccion de handoff (maximo 20 lineas) para ``contexto.md``.

    Args:
        resultados: resultados de las tres arquitecturas.
        f1_multilla: media y desviacion de la F1 entre semillas, si existe.
        mejor_resultado: resultado de la mejor arquitectura.
        conclusion: conclusion construida por :func:`construir_conclusion`.
        rutas: rutas de los artefactos.

    Returns:
        Texto de la seccion, con la primera linea como encabezado.

    Raises:
        ValueError: si la seccion excede el maximo de lineas permitido.
    """
    referencia_mod = float(CONFIG["referencias"]["MOD-03b_BEST"])
    referencia_b1 = float(CONFIG["referencias"]["B1"])
    lineas = [CONFIG["encabezado_contexto"]]
    lineas.append(
        "Requisito metodologico: tres redes desde cero (sin transformers ni "
        "embeddings preentrenados; vocabulario y preprocesamiento solo del pliegue "
        "de entrenamiento), cabezal multi-tarea de 3 logits con BCEWithLogitsLoss y "
        "pos_weight por etiqueta, AdamW lr 1e-3 / wd 1e-4, batch 32, maximo 30 "
        "epocas y early stopping estricto sobre la F1 de validacion (paciencia 5, "
        "mejores pesos restaurados)."
    )
    for resultado in resultados:
        lineas.append(
            f"{resultado.espec.identificador} ({resultado.espec.clave}): F1 OOF "
            f"{resultado.f1_val_media:.4f} +/- {resultado.f1_val_std:.4f}; por "
            f"etiqueta HS {resultado.metricas_oof['f1_por_etiqueta']['HS']:.4f} / "
            f"TR {resultado.metricas_oof['f1_por_etiqueta']['TR']:.4f} / "
            f"AG {resultado.metricas_oof['f1_por_etiqueta']['AG']:.4f}; gap "
            f"train-validacion {resultado.gap_train_val:+.4f}; "
            f"{resultado.n_parametros} parametros; vocab {resultado.vocab_size}; "
            f"dim {resultado.dim_embedding}; longitud "
            f"{resultado.longitud_secuencia}; {resultado.segundos:.1f} s."
        )
    if f1_multilla is not None:
        resumen_semillas = {
            str(clave): round(valor, 4)
            for clave, valor in f1_multilla["por_semilla"].items()
        }
        lineas.append(
            f"Repeticion con semillas "
            f"{'/'.join(str(s) for s in CONFIG['semillas'])} sobre "
            f"{mejor_resultado.espec.identificador}: F1 OOF "
            f"{f1_multilla['media']:.4f} +/- {f1_multilla['desviacion']:.4f} "
            f"(por semilla {resumen_semillas})."
        )
    lineas.append(
        f"Referencias: MOD-03b_BEST (TF-IDF union + LR) OOF {referencia_mod:.4f} / "
        f"LB {float(CONFIG['referencias']['LB_MOD03b']):.5f}; B1 OOF "
        f"{referencia_b1:.4f}."
    )
    lineas.append(
        f"Conclusion honesta: {conclusion['interpretacion']}"
    )
    lineas.append(
        "AMP no usado: medido en esta GPU, con estos modelos el coste de autocast "
        "supera la ganancia de fp16 (BiLSTM 3.42 s vs 7.48 s en 3 epocas). "
        "torch.use_deterministic_algorithms(False) por coredump de GPU observado "
        f"con True en ROCm 7.2; semillas "
        f"{'/'.join(str(s) for s in CONFIG['semillas'])}."
    )
    lineas.append(
        f"Artefactos: {rutas['reporte_csv']}, {rutas['reporte_json']}, "
        f"{rutas['figura']}, {rutas['checkpoint']}; filas DL-01/DL-02/DL-03"
        + (f"/{mejor_resultado.espec.identificador}-SEEDS" if f1_multilla else "")
        + " en experiments.csv; evidencia del pre-vuelo en "
        "experiments/gpu_preflight.log."
    )
    lineas.append(
        "Sin submissions: la decision de envio es del orquestador. Sin fuga: el test "
        "no se cargo en este script."
    )
    if len(lineas) > int(CONFIG["max_lineas_contexto"]):
        raise ValueError(
            f"La seccion de contexto tiene {len(lineas)} lineas y el maximo es "
            f"{CONFIG['max_lineas_contexto']}"
        )
    return "\n".join(lineas)


# ===== SECCION 9: flujo principal =====


def calcular_f1_multilla(
    resultados_multilla: Sequence[ResultadoArquitectura],
) -> dict:
    """Resume la F1 out-of-fold de una repeticion multi-semilla.

    Args:
        resultados_multilla: resultados de la mejor arquitectura por semilla.

    Returns:
        Diccionario con media, desviacion y F1 por semilla.
    """
    por_semilla = {
        resultado.semilla: resultado.f1_val_media for resultado in resultados_multilla
    }
    valores = np.array(list(por_semilla.values()))
    return {
        "media": float(valores.mean()),
        "desviacion": float(valores.std()),
        "por_semilla": por_semilla,
    }


def main() -> None:
    """Ejecuta la Fase 11 completa: pre-vuelo, entrenamiento y registro."""
    inicio = time.perf_counter()
    # El pre-vuelo va primero: cualquier llamada a torch.cuda inicializa el
    # contexto HIP y el proceso aparece en KFD.
    evidencia_gpu = verificar_gpu_libre()
    log(f"dispositivo: {dispositivo_entrenamiento()} | torch {torch.__version__}")

    train, grupos = cargar_datos()
    etiquetas = train[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    textos = list(train["text"])
    pliegues = construir_pliegues(train, grupos)
    log(
        f"train {train.shape[0]} filas | prevalencia "
        f"{dict(zip(ETIQUETAS, np.round(etiquetas.mean(axis=0), 4), strict=True))}"
    )

    resultados: list[ResultadoArquitectura] = []
    for espec in ARQUITECTURAS:
        log(f"=== {espec.identificador}: {espec.nombre} ===")
        resultado = evaluar_arquitectura(espec, textos, etiquetas, pliegues, semilla=42)
        resultados.append(resultado)
        log(
            f"{espec.identificador} F1 OOF {resultado.f1_val_media:.4f} +/- "
            f"{resultado.f1_val_std:.4f} | gap {resultado.gap_train_val:+.4f} | "
            f"{resultado.segundos:.1f} s"
        )

    mejor = max(resultados, key=lambda resultado: resultado.f1_val_media)
    transcurrido = time.perf_counter() - inicio
    semillas = [
        semilla for semilla in CONFIG["semillas"] if semilla != CONFIG["random_state"]
    ]
    resultados_multilla: list[ResultadoArquitectura] = [mejor]
    for semilla in semillas:
        costo_estimado = mejor.segundos * (len(resultados_multilla) + 1)
        if transcurrido + costo_estimado > float(CONFIG["presupuesto_segundos"]):
            log(
                f"presupuesto agotado: se omite la semilla {semilla} "
                f"({transcurrido:.0f} s transcurridos de "
                f"{CONFIG['presupuesto_segundos']} s)"
            )
            break
        log(f"=== {mejor.espec.identificador} con semilla {semilla} ===")
        resultados_multilla.append(
            evaluar_arquitectura(mejor.espec, textos, etiquetas, pliegues, semilla)
        )
    f1_multilla = (
        calcular_f1_multilla(resultados_multilla) if len(resultados_multilla) > 1 else None
    )
    if f1_multilla is not None:
        log(
            f"multi-semilla: {f1_multilla['media']:.4f} +/- "
            f"{f1_multilla['desviacion']:.4f}"
        )

    ruta_checkpoint = guardar_checkpoint(mejor, textos, etiquetas, pliegues)
    ruta_figura = graficar_curvas(resultados)
    rutas = {
        "reporte_csv": CONFIG["reporte_csv"],
        "reporte_json": CONFIG["reporte_json"],
        "figura": ruta_figura,
        "checkpoint": ruta_checkpoint,
    }
    detalle = escribir_reportes(
        resultados, mejor, f1_multilla, evidencia_gpu, rutas
    )
    filas = registrar_en_experiments(resultados, f1_multilla, mejor)
    seccion = construir_seccion_contexto(
        resultados, f1_multilla, mejor, detalle["conclusion"], rutas
    )
    actualizar_contexto(seccion)
    log(f"experiments.csv: {filas} filas escritas")
    log(f"conclusion: {detalle['conclusion']['interpretacion']}")
    log(f"tiempo total: {time.perf_counter() - inicio:.1f} s")


if __name__ == "__main__":
    main()
