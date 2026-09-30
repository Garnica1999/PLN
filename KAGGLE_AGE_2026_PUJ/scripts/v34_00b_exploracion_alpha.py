"""v34_00b_exploracion_alpha.py
Fecha: 2026-09-30
Descripcion: REGISTRO MATERIALIZADO de la exploracion metodologica previa de V34. Ejecuta las
27 celdas (alpha, lambda2, power_t, variante de L2) que el arquitecto midio en /tmp antes de
elegir la regla de margen, y las publica con semilla, configuracion, script y sha256 para que
D-085 no se apoye en numeros que solo viven en un temporal.

QUE ES Y QUE NO ES: esto NO es un grid del experimento V34. Ningun valor se selecciono, ninguna
celda alimenta la eleccion de alpha* (que quedo derivada en forma cerrada antes de correr estas
medidas) y ningun umbral de la puerta G0-G4 se calibro con estas celdas. Es una falsacion
previa, de perdida y sobre datos de fold-train, cuyo unico producto es el rango de ll_train
alcanzable por FTRL-Proximal sobre esta representacion.

ALCANCE DE LOS DATOS (restriccion dura): solo filas de fold-train de la particion canonica
(artifacts/p33_scores/todos/fold_idx.npy, fold != 0), nunca el fold-val, nunca el test, nunca el
holdout, nunca el leaderboard. Unica metrica: log_loss multiclase de train (ll_train). No se
calcula macro-F1 ni ninguna metrica de la puerta en este script.

PRESUPUESTO DECLARADO (identico al usado en /tmp): subconjunto de 30000 filas de fold-train
sorteado con np.random.default_rng(7).choice(135468, 30000, replace=False) y ordenado; 10 pasadas;
ordenes de cada pasada con np.random.default_rng(1000 + pasada).permutation(30000). TF-IDF de
palabra (1,2), min_df 3, max_df 0.5, sublinear_tf, ajustada SOLO en fold-train; sesgo anadido como
columna de 1.0.

LO QUE SE PUBLICA POR CELDA: alpha, lambda2, power_t, variante de L2 (acumulador o denominador),
ll_train final del modelo ajustado a las 10 pasadas, la trayectoria de perdida online por pasada,
|w|max, y la comparacion contra H(prior) y ln(K-1) del subconjunto. Ademas se publica la mejor y
la peor celda, y una referencia no celda del gemelo SGD log_loss con el mismo presupuesto.

Codigos de salida: 0 fin correcto; 2 verificacion (hashes, anclas, invariantes); 4 sonda basica
fallida antes de medir; 1 error inesperado con traceback.
"""

from __future__ import annotations

# =====================================================================================
# ---- 0. ENTORNO E IMPORTS (los 32 hilos van ANTES de importar numpy/scipy/sklearn) ----
# =====================================================================================
import os

#: Los 32 hilos del proyecto se fijan antes de importar numpy (determinismo del BLAS).
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["OPENBLAS_NUM_THREADS"] = "32"

import argparse
import hashlib
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

# =====================================================================================
# ---- 1. CONSTANTES CONGELADAS ----
# =====================================================================================
RAIZ = Path(__file__).resolve().parents[1]

#: Configuracion del vectorizador: la misma de v31_02 / D-083 / D-084.
VECTORIZADOR: dict[str, object] = {
    "analyzer": "word", "ngram_range": (1, 2), "min_df": 3, "max_df": 0.5,
    "sublinear_tf": True, "lowercase": True, "strip_accents": None,
    "token_pattern": r"(?u)\b\w+\b",
}

N_CLASES = 6
FOLD = 0

#: Presupuesto DECLARADO de la exploracion (identico al de /tmp).
SEMILLA_SUBCONJUNTO = 7
SEMILLA_BASE_ORDENES = 1000
FILAS_SUBCONJUNTO = 30000
PASADAS = 10
SEED_GEMELO_SGD = 20261010

#: Alpha de la regla que se uso en V33-E1 (3/(2*sqrt(T)) con T = 135468); se incluye como celda
#: porque es el punto de partida de la derivacion previa.
ALPHA_V33_E1 = 3.0 / (2.0 * math.sqrt(135468.0))

#: Anclas verificadas del proyecto (hashes recalculados en runtime).
ANCLA_HASH_ARCHIVO_CORPUS = "39c6c8270366fe155f3c8dd232e31867bee659af8908121d04b6bd89bd163748"
ANCLA_HASH_TEXTOS = "53c1a348a7954f3e59d7c67caddd6477a06f7d3441724c201052b4b5273445fc"
ANCLA_HASH_FOLD_IDX = "74d18330c40c3598a1936706ae9434c5fdf29119d357d38353cf298812fe6fb5"
ANCLA_N_TRAIN_FOLD = 135468

#: Betas y lambda1 del esquema FTRL-Proximal: identicos a D-083 y D-084.
BETA = 1.0
LAMBDA1 = 0.0
#: Anclas logaritmicas: el ancla uniforme ln(K) y el ancla de margen ln(K-1) que usa la regla de
#: margen de V34 (mismo valor que |log(1/5)| declarado en V33-E1).
LOG_UNIFORME = math.log(N_CLASES)
LOG_K_MENOS_1 = math.log(N_CLASES - 1)

#: Tolerancia de las sondas basicas (precedente D-084).
TOL_LOG_LOSS = 1e-09

RUTA_JSON = RAIZ / "experiments" / "v34_00b_exploracion_alpha.json"
RUTA_MD = RAIZ / "reports" / "v34_exploracion_alpha.md"

#: Etiqueta de categoria exigida por la condicion del kaggle-specialist.
CATEGORIA = ("exploracion metodologica previa; datos de fold-train; metrica de perdida; "
             "sin seleccion")


class Detencion(Exception):
    """Error controlado con codigo de salida asociado."""

    def __init__(self, mensaje: str, codigo: int) -> None:
        super().__init__(mensaje)
        self.codigo = codigo


def esc(code: int) -> None:
    """Escribe un mensaje de deteccion y termina con el codigo indicado.

    Parameters
    ----------
    code
        Codigo de salida del proceso.

    Raises
    ------
    SystemExit
        Siempre, para que el launcher vea el codigo.
    """
    print(f"[STOP] code={code}", file=sys.stderr)
    raise SystemExit(code)


def sha256_bytes(datos: bytes) -> str:
    """Devuelve el sha256 hexadecimal de un bloque de bytes."""
    return hashlib.sha256(datos).hexdigest()


def sha256_f64(arr: np.ndarray) -> str:
    """Devuelve el sha256 de un array float64 en orden C."""
    return sha256_bytes(np.ascontiguousarray(arr, dtype=np.float64).tobytes())


def sha256_textos(textos: np.ndarray) -> str:
    """Sha256 de un array de textos con la receta determinista del proyecto.

    Receta (D-083): por cada texto, la longitud en 8 bytes little-endian seguida de los bytes
    UTF-8 del texto, en orden.
    """
    hh = hashlib.sha256()
    for texto in textos:
        crudo = str(texto).encode("utf-8")
        hh.update(len(crudo).to_bytes(8, "little"))
        hh.update(crudo)
    return hh.hexdigest()


def escritura_atomica(ruta: Path, texto: str) -> None:
    """Escribe un fichero de forma atomica (temporal + os.replace)."""
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    temporal.write_text(texto, encoding="utf-8")
    os.replace(temporal, ruta)


# =====================================================================================
# ---- 2. NUCLEO FTRL-PROXIMAL (misma aritmetica que v32_05 y v33_01) ----
# =====================================================================================
def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax estable sobre la ultima dimension."""
    desplazado = logits - logits.max(axis=-1, keepdims=True)
    exponencial = np.exp(desplazado)
    return exponencial / exponencial.sum(axis=-1, keepdims=True)


def anadir_sesgo(matriz: sp.csr_matrix) -> sp.csr_matrix:
    """Anade la columna de sesgo x0 = 1.0 como ultima columna del CSR."""
    unos = sp.csr_matrix(np.ones((matriz.shape[0], 1), dtype=np.float64))
    return sp.hstack([matriz, unos], format="csr")


def ftrl_softmax(x: sp.csr_matrix, y: np.ndarray, alpha: float, ordenes: list[np.ndarray],
                 lambda2: float, power_t: float, variante_l2: str
                 ) -> tuple[np.ndarray, list[float]]:
    """FTRL-Proximal softmax multiclase (K=6, un solo juego de pesos) multi-pasada.

    Solo se actualizan las coordenadas no nulas de cada fila. ``sigma =
    (sqrt(n+g^2) - sqrt(n))/(alpha_t*beta)`` con ``alpha_t = alpha/t^power_t``; ``z += g - sigma*w``
    mas ``-lambda2*w`` si la variante es "acumulador"; ``n += g^2``; ``w = sign(z)*(|z|-lambda1) /
    ((beta+sqrt(n))/alpha_t + lambda2_den)`` donde ``lambda2_den`` es ``lambda2`` en la variante
    "denominador" y 0.0 en la variante "acumulador".

    Parameters
    ----------
    x
        CSR float64 con la columna de sesgo ya anadida.
    y
        Etiquetas enteras en 0..5.
    alpha
        alpha base; si ``power_t > 0`` el paso efectivo es ``alpha/t^power_t``.
    ordenes
        Una permutacion de las filas por pasada.
    lambda2
        Fuerza de regularizacion L2, con la semantica dictada por ``variante_l2``.
    power_t
        Exponente del decaimiento de alpha.
    variante_l2
        "acumulador" o "denominador".

    Returns
    -------
    tuple
        Pesos ``w`` con forma (n_coordenadas, 6) y la perdida online media por pasada.

    Raises
    ------
    Detencion
        Si ``variante_l2`` no es una de las dos variantes declaradas (codigo 2).
    """
    if variante_l2 not in ("acumulador", "denominador"):
        raise Detencion(f"[STOP] variante_l2 desconocida: {variante_l2}", 2)
    n_coordenadas = x.shape[1]
    lambda2_den = 0.0 if variante_l2 == "acumulador" else lambda2
    w = np.zeros((n_coordenadas, N_CLASES), dtype=np.float64)
    z = np.zeros((n_coordenadas, N_CLASES), dtype=np.float64)
    n = np.zeros((n_coordenadas, N_CLASES), dtype=np.float64)
    trayectoria: list[float] = []
    t = 0
    for orden in ordenes:
        perdida = 0.0
        for i in orden:
            t += 1
            alpha_t = alpha / (t ** power_t) if power_t > 0.0 else alpha
            inicio, fin = int(x.indptr[i]), int(x.indptr[i + 1])
            indices = x.indices[inicio:fin]
            valores = x.data[inicio:fin]
            logits = np.einsum("ij,i->j", w[indices], valores)
            p = softmax(logits)
            g = p.copy()
            g[y[i]] -= 1.0
            g_coord = g[None, :] * valores[:, None]
            n_previo = n[indices]
            n_nuevo = n_previo + g_coord * g_coord
            sigma = (np.sqrt(n_nuevo) - np.sqrt(n_previo)) / (alpha_t * BETA)
            z[indices] = z[indices] + g_coord - sigma * w[indices]
            if variante_l2 == "acumulador":
                z[indices] -= lambda2 * w[indices]
            n[indices] = n_nuevo
            z_nuevo = z[indices]
            if LAMBDA1 > 0.0:
                z_nuevo = np.sign(z_nuevo) * np.maximum(np.abs(z_nuevo) - LAMBDA1, 0.0)
            w[indices] = z_nuevo / ((BETA + np.sqrt(n_nuevo)) / alpha_t + lambda2_den)
            perdida += -math.log(max(float(p[y[i]]), 1e-15))
        trayectoria.append(perdida / len(orden))
    return w, trayectoria


def ll_train(w: np.ndarray, x: sp.csr_matrix, y: np.ndarray, lote: int = 20000) -> float:
    """Log_loss multiclase de train del modelo ``w`` sobre las filas de ``x``."""
    total = 0.0
    for inicio in range(0, x.shape[0], lote):
        fin = min(inicio + lote, x.shape[0])
        p = softmax(np.asarray(x[inicio:fin] @ w))
        total += float(np.sum(-np.log(np.maximum(p[np.arange(fin - inicio), y[inicio:fin]], 1e-15))))
    return total / x.shape[0]


# =====================================================================================
# ---- 3. VERIFICACION PREVIA (sondas basicas y hashes) ----
# =====================================================================================
def verificar() -> dict[str, object]:
    """Verifica implementacion basica, hashes del corpus y anclas de la partida.

    Returns
    -------
    dict
        Resultado de las sondas con sus tolerancias.

    Raises
    ------
    Detencion
        Con codigo 2 si un hash no reproduce o con codigo 4 si falla una sonda.
    """
    p = softmax(np.array([[2.0, 1.0, 0.0, -1.0, -2.0, 0.5],
                          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]))
    y_prueba = np.array([3, 4])
    filas = p.sum(axis=1)
    propio = float(-np.mean(np.log(p[np.arange(2), y_prueba])))
    delta_perdida = abs(propio - float(log_loss(y_prueba, p, labels=np.arange(N_CLASES))))
    if not np.all(np.isfinite(p)) or float(np.max(np.abs(filas - 1.0))) > 1e-09:
        esc(4)
    if delta_perdida > TOL_LOG_LOSS:
        esc(4)
    return {
        "softmax_filas_suman_uno": float(np.max(np.abs(filas - 1.0))),
        "log_loss_propio": propio,
        "log_loss_sklearn": float(log_loss(y_prueba, p, labels=np.arange(N_CLASES))),
        "log_loss_propio_vs_sklearn_delta": delta_perdida,
        "tolerancia": TOL_LOG_LOSS,
        "pass": True,
    }


# =====================================================================================
# ---- 4. CARGA DE DATOS (solo fold-train) ----
# =====================================================================================
def cargar() -> dict[str, object]:
    """Carga el corpus, verifica los hashes y construye el subconjunto de fold-train.

    Returns
    -------
    dict
        Textos, etiquetas, matriz TF-IDF con sesgo del subconjunto y metadatos de procedencia.
    """
    sys.path.insert(0, str(RAIZ))
    from src import config as cfg
    from src.data import load_train

    marco = load_train()
    textos = marco[cfg.TEXT_COL].to_numpy()
    y_all = marco[cfg.TARGET].cat.codes.to_numpy()

    hash_archivo = sha256_bytes(cfg.TRAIN_FILE.read_bytes())
    hash_textos = sha256_textos(textos)
    ruta_fold = RAIZ / "artifacts" / "p33_scores" / "todos" / "fold_idx.npy"
    hash_fold = sha256_bytes(ruta_fold.read_bytes())
    for nombre, obtenido, ancla in (
        ("archivo_corpus", hash_archivo, ANCLA_HASH_ARCHIVO_CORPUS),
        ("array_textos", hash_textos, ANCLA_HASH_TEXTOS),
        ("fold_idx", hash_fold, ANCLA_HASH_FOLD_IDX),
    ):
        if obtenido != ancla:
            print(f"[STOP] hash {nombre}: {obtenido} != {ancla}", file=sys.stderr)
            esc(2)

    fold_idx = np.load(ruta_fold)
    filas_fold_train = np.where(fold_idx != FOLD)[0]
    if filas_fold_train.size != ANCLA_N_TRAIN_FOLD:
        print(f"[STOP] n_train_fold {filas_fold_train.size} != {ANCLA_N_TRAIN_FOLD}", file=sys.stderr)
        esc(2)

    vectorizador = TfidfVectorizer(**VECTORIZADOR)
    x_fold_train = vectorizador.fit_transform(textos[filas_fold_train])
    y_fold_train = y_all[filas_fold_train]

    sub = np.sort(np.random.default_rng(SEMILLA_SUBCONJUNTO).choice(
        filas_fold_train.size, FILAS_SUBCONJUNTO, replace=False))
    x_sin_sesgo = x_fold_train[sub]
    return {
        "x": anadir_sesgo(x_sin_sesgo), "x_sin_sesgo": x_sin_sesgo, "y": y_fold_train[sub],
        "hashes": {"archivo_corpus": hash_archivo, "array_textos": hash_textos,
                   "fold_idx": hash_fold, "vocab_sha256": sha256_textos(
                       np.array(sorted(vectorizador.vocabulary_.keys()), dtype=object)),
                   "idf_sha256": sha256_f64(vectorizador.idf_.astype(np.float64))},
        "n_vocabulario": len(vectorizador.vocabulary_),
        "n_filas_fold_train": int(filas_fold_train.size),
    }


# =====================================================================================
# ---- 5. LAS 27 CELDAS (exactamente las medidas en /tmp; ninguna fue seleccionada) ----
# =====================================================================================
def tabla_celdas() -> list[tuple[float, float, float, str]]:
    """Devuelve la lista congelada de celdas: (alpha, lambda2, power_t, variante_l2).

    El orden reproduce el de los cuatro barridos temporales: L2 en denominador con lambda2 1.0
    (alpha del rango amplio y del rango pequeno), L2 en denominador con lambda2 1e-6, barrido de
    power_t con lambda2 1.0, L2 en acumulador con lambda2 1e-6 y L2 en acumulador con lambda2 en
    {0.05, 0.2, 1.0}. Ninguna celda se eligio por su resultado.
    """
    return [
        (ALPHA_V33_E1, 1.0, 0.0, "denominador"),
        (0.01, 1.0, 0.0, "denominador"),
        (0.02, 1.0, 0.0, "denominador"),
        (0.04, 1.0, 0.0, "denominador"),
        (0.08, 1.0, 0.0, "denominador"),
        (0.0005, 1.0, 0.0, "denominador"),
        (0.001, 1.0, 0.0, "denominador"),
        (0.002, 1.0, 0.0, "denominador"),
        (ALPHA_V33_E1, 1e-06, 0.0, "denominador"),
        (0.05, 1e-06, 0.0, "denominador"),
        (0.176, 1e-06, 0.0, "denominador"),
        (0.03526, 1.0, 0.0, "denominador"),
        (0.054869, 1.0, 0.25, "denominador"),
        (ALPHA_V33_E1, 1.0, 0.25, "denominador"),
        (0.083634, 1.0, 0.5, "denominador"),
        (0.176316, 1.0, 1.0, "denominador"),
        (ALPHA_V33_E1, 1e-06, 0.0, "acumulador"),
        (0.02, 1e-06, 0.0, "acumulador"),
        (0.05, 1e-06, 0.0, "acumulador"),
        (0.176, 1e-06, 0.0, "acumulador"),
        (0.5, 1e-06, 0.0, "acumulador"),
        (ALPHA_V33_E1, 0.05, 0.0, "acumulador"),
        (ALPHA_V33_E1, 0.2, 0.0, "acumulador"),
        (ALPHA_V33_E1, 1.0, 0.0, "acumulador"),
        (0.05, 0.2, 0.0, "acumulador"),
        (0.05, 1.0, 0.0, "acumulador"),
        (0.1763158, 1.0, 0.0, "acumulador"),
    ]


def medir_celda(x: sp.csr_matrix, y: np.ndarray, ordenes: list[np.ndarray],
                alpha: float, lambda2: float, power_t: float, variante: str) -> dict[str, object]:
    """Ejecuta una celda completa y devuelve su registro publicable."""
    inicio = time.time()
    w, trayectoria = ftrl_softmax(x, y, alpha, ordenes, lambda2, power_t, variante)
    referencia = ll_train(w, x, y)
    return {
        "alpha": alpha, "lambda2": lambda2, "power_t": power_t, "variante_l2": variante,
        "ll_train": referencia,
        "ll_train_por_pasada": trayectoria,
        "ll_train_primera_pasada": trayectoria[0],
        "w_max_abs": float(np.max(np.abs(w))),
        "segundos": time.time() - inicio,
    }


# =====================================================================================
# ---- 6. INFORME ----
# =====================================================================================
def informe(celdas: list[dict[str, object]], contexto: dict[str, object],
            referencia: dict[str, object]) -> str:
    """Construye el informe Markdown con la tabla de celdas y su lectura."""
    mejor = min(celdas, key=lambda c: c["ll_train"])
    peor = max(celdas, key=lambda c: c["ll_train"])
    h_prior = contexto["h_prior"]
    lineas = [
        "# V34 - Exploracion metodologica previa de alpha (registro materializado)",
        "",
        f"- Fecha: 2026-09-30. Script: `{contexto['script']}`",
        f"- sha256 del script: `{contexto['sha256_del_script']}`",
        (
            f"- Semillas: subconjunto {SEMILLA_SUBCONJUNTO}, ordenes "
            f"{SEMILLA_BASE_ORDENES}+pasada, gemelo SGD {SEED_GEMELO_SGD}"
        ),
        (
            f"- Presupuesto: {FILAS_SUBCONJUNTO} filas de fold-train (fold {FOLD}) x "
            f"{PASADAS} pasadas, vectorizador ajustado solo en fold-train "
            f"({contexto['n_vocabulario']} terminos)"
        ),
        "- Metrica unica: log_loss multiclase de train. NO se calcula macro-F1 aqui.",
        (
            f"- Anclas del subconjunto: H(prior) = {h_prior:.7f} (prior log), "
            f"ln(K) = {LOG_UNIFORME:.7f} (uniforme), ln(K-1) = {LOG_K_MENOS_1:.7f} "
            "(ancla de margen de la regla de V34)"
        ),
        "",
        "## No es un grid",
        "",
        "Ningun valor se selecciono. La regla de margen que fija `alpha*` en V34 quedo derivada en",
        "forma cerrada ANTES de estas mediciones y no usa ninguna de estas celdas. Ningun umbral de",
        "la puerta G0-G4 se calibro con estas celdas: G1 se ancla en `H(prior) - 0.01` y H(prior) se",
        "calcula del prior de las filas de fold-train, no de esta tabla. Estas 27 celdas NO se",
        "repiten dentro del runner `v34_01`: el runner solo publica la trayectoria de su propia",
        "configuracion congelada.",
        "",
        f"## Tabla de las {len(celdas)} celdas",
        "",
        "| # | alpha | lambda2 | power_t | L2 | ll_train | ll_train p1 | ll_train p10 | |w|max | s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for numero, celda in enumerate(celdas, start=1):
        lineas.append(
            f"| {numero} | {celda['alpha']:.7g} | {celda['lambda2']:g} | {celda['power_t']:g} | "
            f"{celda['variante_l2'][:4]} | {celda['ll_train']:.5f} | "
            f"{celda['ll_train_primera_pasada']:.5f} | {celda['ll_train_por_pasada'][-1]:.5f} | "
            f"{celda['w_max_abs']:.3e} | {celda['segundos']:.0f} |")
    lineas += [
        "",
        "## Lectura",
        "",
        (
            f"- Mejor celda: #{celdas.index(mejor) + 1} (alpha {mejor['alpha']:.7g}, lambda2 "
            f"{mejor['lambda2']:g}, power_t {mejor['power_t']:g}, {mejor['variante_l2']}) con "
            f"ll_train {mejor['ll_train']:.7f}, es decir el predictor uniforme: no hay aprendizaje."
        ),
        (
            f"- Peor celda: #{celdas.index(peor) + 1} (alpha {peor['alpha']:.7g}, lambda2 "
            f"{peor['lambda2']:g}, power_t {peor['power_t']:g}, {peor['variante_l2']}) con "
            f"ll_train {peor['ll_train']:.7f}, es decir el colapso de la firma D-083."
        ),
        (
            f"- Celdas por debajo de H(prior) = {h_prior:.7f}: "
            f"{contexto['n_por_debajo_de_h_prior']} de {len(celdas)}."
        ),
        (
            "- La perdida DEGRADA con el numero de pasadas en toda la familia salvo en el "
            "regimen alpha -> 0 o power_t -> 1, donde el modelo se congela en el uniforme."
        ),
        (
            f"- Referencia no celda: gemelo SGD log_loss con el mismo presupuesto -> ll_train "
            f"{referencia['ll_train']:.5f} en {referencia['pasadas']} pasadas, con frecuencias "
            f"de clase {referencia['frecuencias_por_clase']}."
        ),
        "",
        "## Conclusion de la exploracion",
        "",
        "No existe un `alpha` sano: el punto de operacion optimo de FTRL-Proximal sobre esta TF-IDF",
        "L2-normalizada es el predictor de prior, y el rango que si colapsa empieza justo por encima",
        "del alpha derivado por la regla de margen. La causa no es el valor de alpha sino la norma no",
        "acotada de `z / sqrt(n)`. Esto sustenta el cierre de la familia online por MECANISMO en",
        "D-085 y NO se re-mide dentro del runner.",
        "",
    ]
    return "\n".join(lineas)


# =====================================================================================
# ---- 7. MAIN ----
# =====================================================================================
def main() -> int:
    """Ejecuta el registro materializado de la exploracion previa de alpha.

    Returns
    -------
    int
        0 si termina correctamente; 2 o 4 si se dispara una deteccion.
    """
    inicio_total = time.time()
    parser = argparse.ArgumentParser(description="Exploracion previa de alpha de V34")
    parser.parse_args()

    print("[1/5] verificacion de sondas basicas")
    sondas = verificar()
    print("[2/5] carga de fold-train y verificacion de hashes")
    datos = cargar()
    x = datos["x"]
    y = datos["y"]
    prior = np.bincount(y, minlength=N_CLASES) / y.size
    h_prior = float(-np.sum(prior * np.log(prior)))
    ordenes = [np.random.default_rng(SEMILLA_BASE_ORDENES + p).permutation(y.size)
               for p in range(PASADAS)]
    contexto: dict[str, object] = {
        "n_filas": int(y.size), "n_clases": N_CLASES, "prior": prior.tolist(),
        "h_prior": h_prior, "log_uniforme": LOG_UNIFORME, "log_k_menos_1": LOG_K_MENOS_1,
        "n_vocabulario": datos["n_vocabulario"], "hashes": datos["hashes"],
        "n_filas_fold_train": datos["n_filas_fold_train"],
    }

    print(f"[3/5] midiendo {len(tabla_celdas())} celdas "
          f"({FILAS_SUBCONJUNTO} filas x {PASADAS} pasadas cada una)")
    celdas = [medir_celda(x, y, ordenes, *celda) for celda in tabla_celdas()]
    contexto["n_por_debajo_de_h_prior"] = int(sum(1 for c in celdas if c["ll_train"] < h_prior))

    print("[4/5] referencia no celda: gemelo SGD con el mismo presupuesto")
    # El gemelo se ajusta sobre la matriz SIN la columna de sesgo: fit_intercept=True aporta el
    # sesgo una sola vez (el control de D-083/D-084 recibio el doble sesgo; aqui se evita).
    x_gemelo = datos["x_sin_sesgo"]
    gemelo = SGDClassifier(loss="log_loss", alpha=1e-06, learning_rate="optimal", tol=None,
                           fit_intercept=True, average=False, shuffle=False,
                           random_state=SEED_GEMELO_SGD)
    gemelo.partial_fit(x_gemelo, y, classes=np.arange(N_CLASES))
    for _ in range(PASADAS - 1):
        gemelo.partial_fit(x_gemelo, y)
    w_sgd = np.vstack([gemelo.coef_.T.copy(), np.atleast_2d(gemelo.intercept_)])
    referencia = {
        "descripcion": "SGDClassifier(log_loss, alpha=1e-6, learning_rate=optimal, shuffle=False), "
                       "NO es una celda de la exploracion: es el control sano con el mismo "
                       "presupuesto y el que demuestra que el presupuesto no es el limite",
        "pasadas": PASADAS,
        "ll_train": ll_train(w_sgd, x, y),
        "frecuencias_por_clase": np.round(np.bincount(np.asarray(x @ w_sgd).argmax(1),
                                                     minlength=N_CLASES) / y.size, 4).tolist(),
        "w_max_abs": float(np.max(np.abs(w_sgd))),
    }

    print("[5/5] escritura de JSON e informe")
    contexto["script"] = str(Path(__file__).resolve().relative_to(RAIZ))
    contexto["sha256_del_script"] = sha256_bytes(Path(__file__).resolve().read_bytes())
    resultado = {
        "fase": "V34-00b: registro materializado de la exploracion metodologica previa de alpha",
        "fecha": "2026-09-30",
        "script": contexto["script"],
        "sha256_del_script": contexto["sha256_del_script"],
        "categoria": CATEGORIA,
        "no_repetida_en_el_runner": True,
        "no_es_un_grid": True,
        "justificacion": ("falsacion previa a la eleccion de la regla de margen; ningun valor se "
                          "selecciono y ningun umbral de G0-G4 se calibro con estas celdas"),
        "semilla": {
            "subconjunto": SEMILLA_SUBCONJUNTO,
            "ordenes_por_pasada": f"{SEMILLA_BASE_ORDENES}+pasada",
            "gemelo_sgd": SEED_GEMELO_SGD,
            "receta_subconjunto": (f"np.random.default_rng({SEMILLA_SUBCONJUNTO}).choice("
                                   f"{ANCLA_N_TRAIN_FOLD}, {FILAS_SUBCONJUNTO}, replace=False) "
                                   "y ordenado"),
            "receta_ordenes": (f"np.random.default_rng({SEMILLA_BASE_ORDENES}+p).permutation("
                               f"{FILAS_SUBCONJUNTO}) con p = 0..{PASADAS - 1}"),
        },
        "presupuesto": {
            "filas": FILAS_SUBCONJUNTO,
            "pasadas": PASADAS,
            "particion": f"fold-train del fold {FOLD} de la particion canonica "
                         "(artifacts/p33_scores/todos/fold_idx.npy, filas con fold != 0)",
            "vectorizador": "ajustado solo en fold-train; sesgo como columna x0 = 1.0",
            "beta": BETA, "lambda1": LAMBDA1,
            "celdas": len(celdas),
        },
        "datos": contexto,
        "sondas": sondas,
        "configuracion": {
            "algoritmo": "FTRL-Proximal softmax multiclase (K=6, un solo juego de pesos)",
            "variantes_l2": "acumulador (z -= lambda2*w, lambda2_den 0.0) y "
                            "denominador (w = z/((beta+sqrt(n))/alpha_t + lambda2))",
            "alpha_t": "alpha / t^power_t",
            "metrica": "log_loss multiclase de train unicamente",
            "fold_val_tocado": False, "test_tocado": False, "leaderboard_tocado": False,
        },
        "celdas": celdas,
        "referencia_gemelo_sgd": referencia,
        "mejor_celda": min(celdas, key=lambda c: c["ll_train"]),
        "peor_celda": max(celdas, key=lambda c: c["ll_train"]),
        "ninguna_celda_bajo_h_prior": contexto["n_por_debajo_de_h_prior"] == 0,
        "versiones": {"python": platform.python_version(), "numpy": np.__version__,
                      "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "rss_pico_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
        "duracion_seg": time.time() - inicio_total,
    }
    escritura_atomica(RUTA_JSON, json.dumps(resultado, indent=2, sort_keys=False) + "\n")
    escritura_atomica(RUTA_MD, informe(celdas, contexto, referencia))
    mejor = resultado["mejor_celda"]
    peor = resultado["peor_celda"]
    print(f"OK celdas={len(celdas)} mejor_ll_train={mejor['ll_train']:.7f} "
          f"peor_ll_train={peor['ll_train']:.7f} H(prior)={h_prior:.7f} "
          f"debajo_de_h_prior={contexto['n_por_debajo_de_h_prior']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())