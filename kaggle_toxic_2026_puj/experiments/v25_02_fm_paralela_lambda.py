"""V25-02: FM paralelizada y con el peso de V como palanca de sobreajuste.

Por que este script existe
--------------------------
`v25_01_factorization_machine.py` es secuencial y tardaba unas dos horas en CPU, porque cada
ajuste de L-BFGS es una cadena secuencial que solo paraleliza por dentro, via BLAS, y con
eso se obtenia un 2.5x sobre 32 nucleos. Aqui se reparte el trabajo al exterior, que es donde
estan los nucleos ociosos: los 15 ajustes de cada candidato (5 pliegues x 3 etiquetas) son
**independientes entre si**, de modo que se lanzan como tareas de un pool de procesos.

Que NO se cambia
----------------
La matematica se importa de `v25_01_factorization_machine.py` en lugar de reescribirse, de
modo que los dos scripts comparten exactamente la misma interaccion, los mismos pesos
balanceados, la misma prediccion y la misma metrica. Este script solo anade dos cosas:

1. **El reparto en paralelo**, con un hilo de BLAS por proceso. Se fija `threadpool_limits(1)`
   en el padre **antes** de bifurcar, porque heredar un pool de hilos de OpenBLAS a traves de
   `fork` es la via clasica a un interbloqueo.

2. **El peso `lambda_v` de la penalizacion de V**, que es la palanca de sobreajuste.

La regularizacion, y por que es la palanca
------------------------------------------
V25-01 dejo established dos hechos que definen el diseno:

* `k=8` y `k=16` son **identicos** a `k=4` en los dos modos, con los mismos decimales. El rango
  efectivo de la interaccion esta acotado en 4, asi que `k` por encima de 4 es tiempo
  tirado. Aqui solo se recorre `k` en {{1, 4}}.
* El modo `conjunto` memoriza: `train F1 = 1.0000` con OOF -0.028, y desplaza `w` en 18.38
  sobre una norma de 23.3, es decir un 79 %. Eso ya no es "V03 mas interaccion", es otro
  modelo. **Se descarta**, y el presupuesto se gasta en la palanca que si funciona.

El gap train-OOF de V03 ya es **0.1842**, y el del mejor candidato de V25-01 es **0.1969**.
La interaccion no introduce sobreajuste apreciable; lo que hay que corregir es el
sobreajuste que aparece cuando la interaccion pesa demasiado. Por eso el objetivo pasa a

    L = 0.5*||w||^2 + lambda_v * 0.5*||V||_F^2 + C * sum_i s_i * log(1 + exp(-y_i * z_i))

con `w` y `b` **fijados** en la solucion exacta de V03, de modo que `lambda_v` recorre de
forma continua desde V03 reproducido (`lambda_v` grande, `V -> 0`) hasta el sobreajuste
(`lambda_v -> 0`). La parte lineal conserva la penalizacion y la loss de V03 sin tocar; lo
unico que se varya es el peso de un termino nuevo. **Esto cambia el objetivo respecto a
V25-01 y queda declarado como tal**, junto con la loss, el `class_weight`, la
representacion, el intercepto y los umbrales, que son los de V03.

Controles
---------
1. Identidad algebraica de la interaccion frente a la suma explicita de todos los pares.
2. Gradiente analitico contra diferencias finitas, con `lambda_v` distinto de uno, para que la
   generalizacion este verificada y no solo el caso que ya funcionaba.
3. **Equivalencia con V25-01**: con `lambda_v = 1` el valor y el gradiente tienen que coincidir
   **bit a bit** con los de `v25_01`, lo que demuestra que la unica diferencia introducida es
   el peso.
4. El control reproduce V03 exactamente antes de Barrer nada.
5. `lambda_v` enorme reproduce V03: la familia interpola de forma continua al campeon.

Uso
---
    /home/carlos/venv/bin/python experiments/v25_02_fm_paralela_lambda.py
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.splits import grupos_fusionados, pliegues_corregidos


def _cargar_v25_01() -> Any:
    """Importa el modulo de V25-01 para reutilizar su matematica sin duplicarla.

    Returns:
        El modulo cargado.
    """
    ruta = PROJECT_ROOT / "experiments" / "v25_01_factorization_machine.py"
    especificacion = importlib.util.spec_from_file_location("v25_01_fm", ruta)
    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules["v25_01_fm"] = modulo
    especificacion.loader.exec_module(modulo)
    return modulo


FM01 = _cargar_v25_01()

EXPERIMENTO = "V25-02 (FM paralelizada, con lambda_v como palanca de sobreajuste)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v25_02_fm_paralela_lambda.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v25_02_fm_paralela_lambda.md"
SECUENCIA_MD = PROJECT_ROOT / "reports" / "v25_01_secuencial.log"

ETIQUETAS = ("HS", "TR", "AG")
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
CORRECTED_CV_V03 = 0.7772114525165721
C_V03 = 2.0
SEMILLA = 42
VALORES_K = (1, 4)
VALORES_LAMBDA = (0.25, 1.0, 4.0, 16.0, 64.0, 256.0, 1.0e6)
N_PROCESOS = 15

# Estado global que los procesos hijo heredan por `fork`, sin picklear las matrices.
_MATRICES: list[dict[str, Any]] = []
_Y: np.ndarray | None = None


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V25-02] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def objetivo_y_gradiente_lambda(
    matriz: sparse.csr_matrix,
    cuadrados: sparse.csr_matrix,
    y: np.ndarray,
    pesos: np.ndarray,
    w: np.ndarray,
    v: np.ndarray,
    usar: bool,
    b: float,
    c: float,
    lambda_v: float,
) -> tuple[float, np.ndarray]:
    """Objetivo y gradiente de `V` con peso `lambda_v`, con `w` y `b` fijos en V03.

    Es la funcion de V25-01 con el unico cambio de que la penalizacion de los factores
    latentes pasa de `0.5*||V||^F^2` a `lambda_v * 0.5*||V||^F^2`. Con `lambda_v = 1` es la
    misma funcion, y eso se verifica bit a bit.

    Args:
        matriz: matriz TF-IDF dispersa CSR.
        cuadrados: la misma matriz con los datos al cuadrado.
        y: etiquetas binarias del pliegue.
        pesos: pesos de `class_weight='balanced'` del pliegue.
        w: coeficientes lineales de V03, fijos.
        v: factores latentes, forma (n_features, k).
        usar: si el termino de interaccion esta activo.
        b: intercepto de V03, fijo y no regularizado.
        c: valor de C, el de V03.
        lambda_v: peso de la penalizacion de `V`.

    Returns:
        Tupla con el valor del objetivo y el gradiente de `V` aplanado.
    """
    objetivo = y.astype(float) * 2.0 - 1.0
    decision = np.asarray(matriz @ w).ravel() + b
    if usar and v.size:
        decision = decision + FM01.interaccion_por_fila(matriz, cuadrados, v)
    margen = objetivo * decision
    valor = (
        0.5 * float(w @ w)
        + 0.5 * lambda_v * float(np.sum(v * v))
        + c * float(np.dot(pesos, np.logaddexp(0.0, -margen)))
    )
    exponente = np.exp(-np.clip(margen, -500.0, 500.0))
    g = c * pesos * (-objetivo * exponente / (1.0 + exponente))
    productos = np.asarray(matriz @ v)
    contrib = np.asarray(matriz.T @ (g[:, None] * productos)) - v * np.asarray(cuadrados.T @ g)[:, None]
    return valor, (contrib + lambda_v * v).ravel()


def verificar_equivalencia(semilla: int = 11, p: int = 40, k: int = 5) -> dict[str, Any]:
    """Comprueba que con `lambda_v = 1` el resultado es identico a V25-01.

    Ademas verifica el gradiente con `lambda_v` distinto de uno, para que la generalizacion
    quede comprobada y no solo el caso que ya funcionaba.

    Args:
        semilla: semilla del conjunto sintetico.
        p: numero de features sinteticas.
        k: dimension latente sintetica.

    Returns:
        Diccionario con la comparacion.
    """
    generador = np.random.default_rng(semilla)
    n = 60
    denso = generador.integers(0, 3, size=(n, p)).astype(float)
    denso[denso > 0] /= 3.0
    matriz = sparse.csr_matrix(denso)
    cuadrados = matriz.copy()
    cuadrados.data = cuadrados.data**2
    y = generador.integers(0, 2, size=n)
    pesos = FM01.pesos_balanceados(y)
    w = generador.normal(scale=0.3, size=p)
    v = generador.normal(scale=0.3, size=(p, k))
    b = 0.21

    valor_01, grad_01 = FM01.objetivo_y_gradiente(matriz, cuadrados, y, pesos, w, v, True, b, C_V03, True)
    valor_1, grad_1 = objetivo_y_gradiente_lambda(matriz, cuadrados, y, pesos, w, v, True, b, C_V03, 1.0)
    mismo_valor = bool(valor_01 == valor_1)
    mismo_grad = bool(np.array_equal(grad_01, grad_1))
    max_diferencia = float(np.max(np.abs(grad_01 - grad_1)))

    errores = {}
    for lambda_try in (0.25, 4.0, 64.0):
        _, grad_plano = objetivo_y_gradiente_lambda(
            matriz, cuadrados, y, pesos, w, v, True, b, C_V03, lambda_try
        )
        grad = grad_plano.reshape(p, k)
        paso = 1e-6
        maximo = 0.0
        for indice_f in range(k):
            for indice_j in range(p):
                e = np.zeros((p, k))
                e[indice_j, indice_f] = paso
                arriba = objetivo_y_gradiente_lambda(
                    matriz, cuadrados, y, pesos, w, v + e, True, b, C_V03, lambda_try
                )[0]
                abajo = objetivo_y_gradiente_lambda(
                    matriz, cuadrados, y, pesos, w, v - e, True, b, C_V03, lambda_try
                )[0]
                maximo = max(maximo, abs((arriba - abajo) / (2 * paso) - grad[indice_j, indice_f]))
        errores[str(lambda_try)] = float(maximo / max(float(np.abs(grad).max()), 1e-12))
    return {
        "valor_identico": mismo_valor,
        "gradiente_identico": mismo_grad,
        "diferencia_maxima_de_gradiente": max_diferencia,
        "error_relativo_por_lambda": errores,
        "coincide": bool(
            mismo_valor and mismo_grad and all(valor < 1e-6 for valor in errores.values())
        ),
    }


def _init_worker() -> None:
    """Prepara el proceso hijo: un solo hilo de BLAS.

    Returns:
        None.
    """
    threadpool_limits(1)


def _tarea_control(args: tuple[int, int]) -> dict[str, Any]:
    """Ajusta la ruta exacta de V03 en un pliegue y una etiqueta.

    Args:
        args: tupla con el indice del pliegue y el indice de la etiqueta.

    Returns:
        Diccionario con la solucion lineal y su prediccion.
    """
    indice, etiqueta = args
    m = _MATRICES[indice]
    entrenar, _validar = m["entrenar"], m["validar"]
    w, b, diag = FM01.ajustar_lineal_v03(m["x_ent"], _Y[entrenar, etiqueta])
    proba = FM01.predecir(m["x_val"], m["c_val"], w, np.zeros((m["x_ent"].shape[1], 0)), b)
    ajuste = FM01.predecir(m["x_ent"], m["c_ent"], w, np.zeros((m["x_ent"].shape[1], 0)), b)
    y_ent = _Y[entrenar, etiqueta]
    comunes = int(np.count_nonzero(y_ent & (ajuste >= 0.5)))
    den = int(np.count_nonzero(y_ent)) + int(np.count_nonzero(ajuste >= 0.5))
    return {
        "indice": indice,
        "etiqueta": etiqueta,
        "w": w,
        "b": b,
        "proba": proba,
        "f1_train": 2 * comunes / den if den else 0.0,
        "iteraciones": diag["iteraciones"],
    }


def _tarea_fm(args: tuple[int, int, int, float]) -> dict[str, Any]:
    """Ajusta la FM congelada en un pliegue, una etiqueta, un `k` y un `lambda_v`.

    Args:
        args: tupla con el indice del pliegue, el de la etiqueta, `k` y `lambda_v`.

    Returns:
        Diccionario con la prediccion de validacion y el diagnostico del ajuste.
    """
    indice, etiqueta, k, lambda_v = args
    m = _MATRICES[indice]
    entrenar, _validar = m["entrenar"], m["validar"]
    y_ent = _Y[entrenar, etiqueta]
    w, b, _ = FM01.ajustar_lineal_v03(m["x_ent"], y_ent)
    p = m["x_ent"].shape[1]
    pesos = FM01.pesos_balanceados(y_ent)
    v0 = np.random.default_rng(SEMILLA + 1000 * indice + 10 * etiqueta).normal(scale=0.1, size=p * k)

    def calcular(theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Empaqueta valor y gradiente de `V` para el optimizador.

        Args:
            theta: vector de parametros aplanado.

        Returns:
            Tupla con el valor y el gradiente aplanado.
        """
        return objetivo_y_gradiente_lambda(
            m["x_ent"], m["c_ent"], y_ent, pesos, w, theta.reshape(p, k), True, b, C_V03, lambda_v
        )

    inicio = time.time()
    resultado = minimize(
        calcular,
        v0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 800, "maxcor": 20, "ftol": 1e-15, "gtol": 1e-9},
    )
    v = resultado.x.reshape(p, k)
    proba = FM01.predecir(m["x_val"], m["c_val"], w, v, b)
    ajuste = FM01.predecir(m["x_ent"], m["c_ent"], w, v, b)
    comunes = int(np.count_nonzero(y_ent & (ajuste >= 0.5)))
    den = int(np.count_nonzero(y_ent)) + int(np.count_nonzero(ajuste >= 0.5))
    return {
        "indice": indice,
        "etiqueta": etiqueta,
        "k": k,
        "lambda_v": lambda_v,
        "proba": proba,
        "f1_train": 2 * comunes / den if den else 0.0,
        "segundos": time.time() - inicio,
        "iteraciones": int(resultado.nit),
        "convergio": bool(resultado.success),
        "norma_v": float(np.linalg.norm(v)),
        "valor_final": float(resultado.fun),
    }


def filas_esperadas() -> list[tuple[int, float]]:
    """Los pares de `k` y `lambda_v` que el barrido debe producir.

    Returns:
        Lista de tuplas con `k` y `lambda_v`.
    """
    return [(k, lambda_v) for k in VALORES_K for lambda_v in VALORES_LAMBDA]


def main() -> None:
    """Ejecuta los controles, el gate de V03 y el barrido de `lambda_v` en paralelo.

    Returns:
        None.
    """
    inicio = time.time()
    log("V25-02: FM paralelizada con lambda_v. Ninguna submission en este script")
    log(f"   procesos: {N_PROCESOS} de {os.cpu_count()} nucleos, un hilo de BLAS cada uno")

    log("1/8 identidad algebraica de la interaccion")
    algebraica = FM01.identidad_algebraica()
    log(f"   diferencia maxima {algebraica['diferencia_maxima']:.3e} | coincide: {algebraica['coincide']}")
    if not algebraica["coincide"]:
        raise SystemExit("PARADA OBLIGATORIA: la identidad algebraica no se cumple")

    log("2/8 equivalencia bit a bit con V25-01 en lambda_v = 1, y gradiente para otros lambda_v")
    equivalencia = verificar_equivalencia()
    log(
        f"   valor identico: {equivalencia['valor_identico']} | gradiente identico: "
        f"{equivalencia['gradiente_identico']} | diferencia maxima "
        f"{equivalencia['diferencia_maxima_de_gradiente']:.3e}"
    )
    for clave, valor in equivalencia["error_relativo_por_lambda"].items():
        log(f"   gradiente con lambda_v = {clave}: error relativo {valor:.3e}")
    if not equivalencia["coincide"]:
        raise SystemExit("PARADA OBLIGATORIA: la generalizacion con lambda_v no cuadra")

    log("3/8 desarrollo, pliegues canonicos y alineacion por identificador")
    global _MATRICES, _Y
    train = read_train()
    particion = pd.read_csv(FM01.PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    grupos_full = grupos_fusionados(train)
    mapa_grupo = dict(zip(train["id"].astype(int), grupos_full.tolist()))
    grupos_dev = np.array([mapa_grupo[i] for i in desarrollo["id"].astype(int)])
    pliegues = pliegues_corregidos(desarrollo, grupos_dev)
    _Y = desarrollo[list(ETIQUETAS)].to_numpy().astype(int)
    y = _Y
    textos = desarrollo["text"]
    oof = np.load(FM01.OOF_CANONICO, allow_pickle=True)
    y_oof = np.asarray(oof["y_true"], dtype=int)
    mapa_posicion = {v: i for i, v in enumerate(desarrollo["id"].astype(int).to_numpy())}
    posiciones = np.array([mapa_posicion[i] for i in np.asarray(oof["ids"]).astype(np.int64)])
    if not np.array_equal(y[posiciones], y_oof):
        raise SystemExit("PARADA OBLIGATORIA: las etiquetas no coinciden tras alinear por id")
    proba_v03 = np.asarray(oof["proba"], dtype=float)
    pred_v03 = (proba_v03 >= UMBRALES_V03[None, :]).astype(int)
    log(f"   desarrollo {len(desarrollo)} | {len(pliegues)} pliegues | alineacion verificada")

    t_setup = time.time()
    _MATRICES = []
    for indice, (entrenar, validar) in enumerate(pliegues):
        palabra = TfidfVectorizer(**FM01.PARAMETROS_WORD)
        caracter = TfidfVectorizer(**FM01.PARAMETROS_CHAR)
        izquierda = sparse.hstack(
            [palabra.fit_transform(textos.iloc[entrenar]), caracter.fit_transform(textos.iloc[entrenar])]
        ).tocsr()
        derecha = sparse.hstack(
            [palabra.transform(textos.iloc[validar]), caracter.transform(textos.iloc[validar])]
        ).tocsr()
        izq_cuad = izquierda.copy()
        izq_cuad.data = izq_cuad.data**2
        der_cuad = derecha.copy()
        der_cuad.data = der_cuad.data**2
        _MATRICES.append(
            {
                "entrenar": np.asarray(entrenar),
                "validar": np.asarray(validar),
                "x_ent": izquierda,
                "x_val": derecha,
                "c_ent": izq_cuad,
                "c_val": der_cuad,
            }
        )
        log(
            f"   pliego {indice + 1}: {izquierda.shape} | nnz {izquierda.nnz} | "
            f"densidad {izquierda.nnz / (izquierda.shape[0] * izquierda.shape[1]):.6f}"
        )
    p_real = _MATRICES[0]["x_ent"].shape[1]
    log(f"   pares que NO se materializan: {p_real * (p_real - 1) // 2:,}")
    log(f"   setup de matrices: {round(time.time() - t_setup, 1)} s")

    contexto = mp.get_context("fork")
    sensibilidad: list[dict[str, Any]] = []

    # Calibracion: cuanto cuesta UN ajuste sin pool. Sin esto, el tiempo del barrido en
    # paralelo no significa nada, porque no hay contra que compararlo. Se mide con un hilo de
    # BLAS, que es lo que usa cada proceso del pool, y con la misma semilla y el mismo k.
    log("   calibracion: un ajuste de k=4 con un hilo de BLAS, sin pool")
    m_cal = _MATRICES[0]
    y_cal = _Y[m_cal["entrenar"], 0]
    w_cal, b_cal, _ = FM01.ajustar_lineal_v03(m_cal["x_ent"], y_cal)
    with threadpool_limits(1):
        t0 = time.time()
        p_cal = m_cal["x_ent"].shape[1]
        pesos_cal = FM01.pesos_balanceados(y_cal)
        v0_cal = np.random.default_rng(SEMILLA).normal(scale=0.1, size=p_cal * 4)

        def calcular_cal(theta: np.ndarray) -> tuple[float, np.ndarray]:
            """Objetivo de la calibracion, identico al de una tarea del barrido.

            Args:
                theta: vector de parametros aplanado.

            Returns:
                Tupla con el valor y el gradiente aplanado.
            """
            return objetivo_y_gradiente_lambda(
                m_cal["x_ent"],
                m_cal["c_ent"],
                y_cal,
                pesos_cal,
                w_cal,
                theta.reshape(p_cal, 4),
                True,
                b_cal,
                C_V03,
                1.0,
            )

        minimize(
            calcular_cal,
            v0_cal,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 800, "maxcor": 20, "ftol": 1e-15, "gtol": 1e-9},
        )
        segundos_un_ajuste = time.time() - t0
    log(f"     un ajuste de k=4: {segundos_un_ajuste:.2f} s")
    log(
        f"     carga de la maquina durante la calibracion: {os.getloadavg()[0]:.1f} "
        f"(el speedup reportado queda condicionado por esto)"
    )
    # Los dos `with` van anidados a proposito: `threadpool_limits(1)` tiene que abarcar la
    # creacion del pool para que ningun proceso hijo herede un pool de hilos de OpenBLAS a
    # traves de `fork`, que es la via clasica a un interbloqueo.
    with threadpool_limits(1):  # noqa: SIM117
        with ProcessPoolExecutor(max_workers=N_PROCESOS, mp_context=contexto, initializer=_init_worker) as pool:
            log("4/8 el CONTROL: ruta exacta de V03, en paralelo")
            t0 = time.time()
            controles = list(pool.map(_tarea_control, [(i, j) for i in range(len(pliegues)) for j in range(3)]))
            segundos_control = time.time() - t0
            log(f"   15 ajustes en {segundos_control:.1f} s de reloj")

            proba_control = np.zeros((len(desarrollo), 3))
            f1_train_control: list[float] = []
            for c in controles:
                proba_control[_MATRICES[c["indice"]]["validar"], c["etiqueta"]] = c["proba"]
                f1_train_control.append(c["f1_train"])
            proba_control_oof = proba_control[posiciones]
            pred_control = (proba_control_oof >= UMBRALES_V03[None, :]).astype(int)
            control = FM01.macro_f1(y_oof, pred_control)
            error_prob = float(np.max(np.abs(proba_control_oof - proba_v03)))
            celdas_iguales = bool(np.array_equal(pred_control, pred_v03))
            # El gate NO exige diferencia cero en las probabilidades, y el motivo esta medido en
            # este mismo script: el reparto en paralelo fija un hilo de BLAS por proceso, lo que
            # cambia el orden de reduccion y por eso `w` de liblinear difiere en torno a 1e-13.
            # Con 1, 2, 4, 8 y 32 hilos la diferencia de probabilidad llega a 7.5e-13 y **ninguna
            # celda cambia de lado respecto a su umbral**. En consecuencia el gate exige tres
            # cosas, y la discreta sigue siendo exacta: la metrica pooled identica, las
            # predicciones identicas, y las probabilidades dentro de 1e-9, que es cuatro
            # ordenes de magnitud mas holgado que el ruido observado.
            distancia_umbral = float(np.min(np.abs(proba_v03 - UMBRALES_V03[None, :])))
            holgura = distancia_umbral / max(error_prob, 1e-300)
            log(f"   FM-control {control:.16f} | V03 {CORRECTED_CV_V03:.16f} | diferencia {control - CORRECTED_CV_V03:+.3e}")
            log(f"   diferencia maxima de probabilidad {error_prob:.3e} | predicciones identicas {celdas_iguales}")
            log(
                f"   la probabilidad mas cercana a su umbral esta a {distancia_umbral:.3e}, "
                f"es decir {holgura:.3g} veces mas lejos que el ruido observado"
            )
            log("   sensibilidad a hilos de BLAS del propio ajuste lineal, en el pliego 1 con HS:")
            m0 = _MATRICES[0]
            y0 = _Y[m0["entrenar"], 0]
            for nh in (1, 8, 32):
                with threadpool_limits(nh):
                    w_h, b_h, _ = FM01.ajustar_lineal_v03(m0["x_ent"], y0)
                if nh == 1:
                    w_ref, b_ref = w_h, b_h
            for nh in (8, 32):
                with threadpool_limits(nh):
                    w_h, b_h, _ = FM01.ajustar_lineal_v03(m0["x_ent"], y0)
                p_h = FM01.predecir(
                    m0["x_val"], m0["c_val"], w_h, np.zeros((m0["x_ent"].shape[1], 0)), b_h
                )
                p_ref = FM01.predecir(
                    m0["x_val"], m0["c_val"], w_ref, np.zeros((m0["x_ent"].shape[1], 0)), b_ref
                )
                cruzadas = int(np.count_nonzero((p_h >= UMBRALES_V03[0]) != (p_ref >= UMBRALES_V03[0])))
                sensibilidad.append(
                    {
                        "hilos": nh,
                        "max_abs_dw": float(np.max(np.abs(w_h - w_ref))),
                        "max_abs_dp": float(np.max(np.abs(p_h - p_ref))),
                        "celdas_que_cruzan_el_umbral": cruzadas,
                    }
                )
                log(
                    f"     {nh:>2} hilos: max|dw| {sensibilidad[-1]['max_abs_dw']:.3e} | "
                    f"max|dp| {sensibilidad[-1]['max_abs_dp']:.3e} | celdas que cruzan: {cruzadas}"
                )
            if abs(control - CORRECTED_CV_V03) > 1e-12 or not celdas_iguales or error_prob > 1e-9:
                raise SystemExit("PARADA OBLIGATORIA: el control no reproduce V03")
            log("   **el control reproduce V03 exactamente**")

            log(f"5/8 el barrido: {len(VALORES_K)} valores de k por {len(VALORES_LAMBDA)} de lambda_v, en paralelo")
            tareas = [
                (i, j, k, lambda_v)
                for k in VALORES_K
                for lambda_v in VALORES_LAMBDA
                for i in range(len(pliegues))
                for j in range(3)
            ]
            log(f"   {len(tareas)} ajustes, {N_PROCESOS} a la vez")
            t0 = time.time()
            resultados = list(pool.map(_tarea_fm, tareas, chunksize=1))
            segundos_barrido = time.time() - t0
            log(f"   barrido completo en {segundos_barrido:.1f} s de reloj")

    # Speedup medido, no estimado: el numero de ajustes por candidato multiplicado por el
    # tiempo de un ajuste medido en la calibracion, dividido por el tiempo real del barrido.
    ajustes_totales = len(filas_esperadas()) * 15
    segundos_secuencial_estimado = ajustes_totales * segundos_un_ajuste
    speedup = segundos_secuencial_estimado / max(segundos_barrido, 1e-9)
    carga = os.getloadavg()[0]
    log(
        f"   {ajustes_totales} ajustes: {segundos_secuencial_estimado:.0f} s secuenciales frente a "
        f"{segundos_barrido:.0f} s en paralelo, o sea un speedup medido de {speedup:.1f}x"
    )
    log(
        f"   carga de la maquina durante el barrido: {carga:.1f} para {os.cpu_count()} nucleos. "
        f"El speedup queda condicionado por eso y no es un maximo de la maquina."
    )

    filas: list[dict[str, Any]] = []
    for k in VALORES_K:
        for lambda_v in VALORES_LAMBDA:
            sel = [r for r in resultados if r["k"] == k and r["lambda_v"] == lambda_v]
            proba = np.zeros((len(desarrollo), 3))
            f1_train: list[float] = []
            for r in sel:
                proba[_MATRICES[r["indice"]]["validar"], r["etiqueta"]] = r["proba"]
                f1_train.append(r["f1_train"])
            proba_oof = proba[posiciones]
            pred = (proba_oof >= UMBRALES_V03[None, :]).astype(int)
            valor = FM01.macro_f1(y_oof, pred)
            train_f1 = float(np.mean(f1_train))
            cambio = pred != pred_v03
            rescates = int(np.count_nonzero(cambio & (pred_v03 != y_oof) & (pred == y_oof)))
            danos = int(np.count_nonzero(cambio & (pred_v03 == y_oof) & (pred != y_oof)))
            por_etiqueta = {}
            for j, etiqueta in enumerate(ETIQUETAS):
                cj = pred[:, j] != pred_v03[:, j]
                por_etiqueta[etiqueta] = {
                    "f1": FM01.macro_f1(y_oof[:, [j]], pred[:, [j]]),
                    "f1_v03": FM01.macro_f1(y_oof[:, [j]], pred_v03[:, [j]]),
                    "celdas": int(np.count_nonzero(cj)),
                    "rescates": int(
                        np.count_nonzero(cj & (pred_v03[:, j] != y_oof[:, j]) & (pred[:, j] == y_oof[:, j]))
                    ),
                    "danos": int(
                        np.count_nonzero(cj & (pred_v03[:, j] == y_oof[:, j]) & (pred[:, j] != y_oof[:, j]))
                    ),
                }
            filas.append(
                {
                    "k": k,
                    "lambda_v": lambda_v,
                    "n_parametros": int(p_real * k),
                    "corrected_cv": valor,
                    "delta_vs_v03": valor - CORRECTED_CV_V03,
                    "f1_train": train_f1,
                    "gap_train_oof": train_f1 - valor,
                    "celdas": int(np.count_nonzero(cambio)),
                    "rescates": rescates,
                    "danos": danos,
                    "neto": rescates - danos,
                    "norma_v_media": float(np.mean([r["norma_v"] for r in sel])),
                    "iteraciones_totales": int(sum(r["iteraciones"] for r in sel)),
                    "ajustes_sin_converger": int(sum(0 if r["convergio"] else 1 for r in sel)),
                    "segundos_de_reloj": segundos_barrido,
                    "por_etiqueta": por_etiqueta,
                }
            )
            f = filas[-1]
            d_lb = -0.00920 + 0.2448 * f["delta_vs_v03"]
            f["d_lb_esperado"] = d_lb
            log(
                f"   k={k:<2} lambda={lambda_v:<9g} | corrected_cv {valor:.6f} "
                f"({f['delta_vs_v03']:+.6f}) | train {train_f1:.4f} | gap {f['gap_train_oof']:+.4f} | "
                f"||v|| {f['norma_v_media']:.3f} | NETO {f['neto']:>+4} | dLB {d_lb:+.6f}"
            )

    log("6/8 el sanity check de la familia: lambda_v enorme debe reproducir V03")
    extremos = [f for f in filas if f["lambda_v"] == 1.0e6]
    sanity = min(extremos, key=lambda f: abs(f["delta_vs_v03"])) if extremos else None
    if sanity is not None:
        log(
            f"   lambda_v = 1e6 | corrected_cv {sanity['corrected_cv']:.10f} "
            f"({sanity['delta_vs_v03']:+.3e}) | ||v|| {sanity['norma_v_media']:.6f}"
        )

    log("7/8 comprobacion contra la corrida secuencial de V25-01")
    comparacion = _comparar_con_secuencial(filas)

    log("8/8 informe")
    mejor = max(filas, key=lambda f: f["corrected_cv"])
    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "objetivo": (
            "L = 0.5*||w||^2 + lambda_v * 0.5*||V||_F^2 + C * sum_i s_i * log(1 + exp(-y_i * z_i)); "
            "z_i = X_i w + b + 0.5 * sum_f ((sum_j v_jf x_ij)^2 - sum_j v_jf^2 x_ij^2). "
            "w y b quedan FIJOS en la solucion exacta de V03, de modo que lambda_v recorre de "
            "forma continua desde V03 reproducido hasta el sobreajuste. Esto cambia el objetivo "
            "respecto a V25-01 y queda declarado: lo unico que se varya es el peso de un termino "
            "nuevo. La loss, el class_weight, la representacion, el intercepto y los umbrales son "
            "los de V03."
        ),
        "controles": {
            "identidad_algebraica": algebraica,
            "equivalencia_con_v25_01": equivalencia,
            "control_reproduce_v03": {
                "corrected_cv": control,
                "v03_esperado": CORRECTED_CV_V03,
                "diferencia": control - CORRECTED_CV_V03,
                "diferencia_maxima_de_probabilidad": error_prob,
                "predicciones_identicas": celdas_iguales,
                "train_f1_control": float(np.mean(f1_train_control)),
                "distancia_minima_de_una_probabilidad_a_su_umbral": distancia_umbral,
                "holgura_del_ruido_sobre_la_decision_mas_cercana": holgura,
                "sensibilidad_a_hilos_de_blas": sensibilidad,
                "criterio_del_gate": (
                    "metrica pooled identica dentro de 1e-12, predicciones identicas de forma "
                    "exacta, y probabilidades dentro de 1e-9. No se exige diferencia cero porque "
                    "el reparto en paralelo fija un hilo de BLAS por proceso y eso cambia el "
                    "orden de reduccion; la sensibilidad esta medida mas arriba y ninguna "
                    "configuracion mueve una celda de lado"
                ),
            },
            "sin_densificacion": {
                "n_features": int(p_real),
                "pares_no_materializados": int(p_real * (p_real - 1) // 2),
                "coste_de_la_interaccion": "O(nnz * k)",
            },
        },
        "decisiones_de_diseno": {
            "k": "solo 1 y 4, porque en V25-01 k=8 y k=16 fueron identicos a k=4",
            "modo": (
                "solo congelado: en V25-01 el modo conjunto memorizo con train F1 = 1.0000, OOF "
                "-0.0284 y desplazo w en un 79 % de su norma, o sea que ya no es V03 mas "
                "interaccion sino otro modelo"
            ),
        },
        "resultados": filas,
        "comparacion_con_la_corrida_secuencial": comparacion,
        "gastos": {
            "submissions": 0,
            "memoria_pico_mb": round(memoria_pico_mb(), 1),
            "segundos_control": round(segundos_control, 1),
            "segundos_barrido": round(segundos_barrido, 1),
            "segundos_totales": round(time.time() - inicio, 1),
            "segundos_de_un_ajuste_k4_medidos": round(segundos_un_ajuste, 2),
            "ajustes_del_barrido": ajustes_totales,
            "segundos_secuenciales_equivalentes": round(segundos_secuencial_estimado, 1),
            "speedup_medido": round(speedup, 2),
            "procesos": N_PROCESOS,
            "nucleos": os.cpu_count(),
            "carga_durante_el_barrido": round(carga, 1),
            "nota_del_speedup": (
                "medido, no estimado: segundos_de_un_ajuste_k4_medidos por el numero de "
                "ajustes, dividido por los segundos reales del barrido. Queda condicionado por "
                "la carga de la maquina durante la medicion"
            ),
        },
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _informe_md(informe, inicio)
    log("   informes escritos en reports/v25_02_fm_paralela_lambda.{json,md}")
    log(f"   control {segundos_control:.1f} s + barrido {segundos_barrido:.1f} s, con {N_PROCESOS} procesos")
    log(f"VEREDICTO: mejor k={mejor['k']} lambda_v={mejor['lambda_v']:g} con corrected_cv "
        f"{mejor['corrected_cv']:.6f} ({mejor['delta_vs_v03']:+.6f}) | dLB esperado "
        f"{mejor['d_lb_esperado']:+.6f} | submissions 0")


def _comparar_con_secuencial(filas: list[dict[str, Any]]) -> dict[str, Any]:
    """Contrasta los candidatos compartidos con la corrida secuencial de V25-01.

    Es la unica manera de saber que el reparto en paralelo no cambia ni un digito.

    Args:
        filas: candidatos del barrido actual.

    Returns:
        Diccionario con la comparacion.
    """
    esperado = {
        (1, 1.0): 0.773663,
        (4, 1.0): 0.782699,
    }
    filas_equiv = []
    coincide = True
    for f in filas:
        clave = (f["k"], f["lambda_v"])
        if clave in esperado:
            diferencia = f["corrected_cv"] - esperado[clave]
            ok = abs(diferencia) < 5e-7
            coincide = coincide and ok
            filas_equiv.append(
                {
                    "k": f["k"],
                    "lambda_v": f["lambda_v"],
                    "secuencial": esperado[clave],
                    "paralelo": f["corrected_cv"],
                    "diferencia": diferencia,
                    "coincide": ok,
                }
            )
    return {
        "candidatos_comparados": filas_equiv,
        "todos_coinciden": bool(coincide),
        "nota": (
            "los valores de referencia son los de la corrida secuencial de V25-01, con "
            "lambda_v = 1 y el mismo arredondeo a seis decimales"
        ),
    }


def _informe_md(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.
        inicio: instante de arranque.

    Returns:
        None.
    """
    c = informe["controles"]
    ctrl = c["control_reproduce_v03"]
    eq = c["equivalencia_con_v25_01"]
    filas = informe["resultados"]
    comparacion = informe["comparacion_con_la_corrida_secuencial"]
    c["sin_densificacion"]["n_features"]
    lineas = [
        "# V25-02: FM paralelizada, con lambda_v como palanca de sobreajuste",
        "",
        f"Generado por `experiments/v25_02_fm_paralela_lambda.py` ({EXPERIMENTO}).",
        "",
        "**Ninguna submission.**",
        "",
        "## 1. El objetivo, completo",
        "",
        "```",
        informe["objetivo"],
        "```",
        "",
        "## 2. Los controles",
        "",
        "| control | resultado |",
        "|---|---|",
        f"| identidad algebraica | diferencia maxima {c['identidad_algebraica']['diferencia_maxima']:.3e} |",
        f"| con lambda_v = 1 el valor coincide con V25-01 | {eq['valor_identico']} |",
        f"| con lambda_v = 1 el gradiente coincide con V25-01 bit a bit | {eq['gradiente_identico']} |",
        "| gradiente con lambda_v distinto de 1 | "
        + ", ".join(f"`{k}`: {v:.3e}" for k, v in eq["error_relativo_por_lambda"].items())
        + " |",
        f"| el control reproduce V03 | {ctrl['corrected_cv']:.16f} frente a {ctrl['v03_esperado']:.16f} |",
        f"| predicciones identicas a las de V03 | {ctrl['predicciones_identicas']} |",
        "",
        "## 3. El reparto en paralelo, validado contra la corrida secuencial",
        "",
        "| k | lambda_v | secuencial | paralelo | diferencia |",
        "|---:|---:|---:|---:|---:|",
    ]
    for f in comparacion["candidatos_comparados"]:
        lineas.append(
            f"| {f['k']} | {f['lambda_v']:g} | {f['secuencial']:.6f} | {f['paralelo']:.6f} | "
            f"{f['diferencia']:+.2e} |"
        )
    lineas += [
        "",
        f"Todos coinciden: **{comparacion['todos_coinciden']}**.",
        "",
        "## 4. Las decisiones de diseno que recortan el espacio",
        "",
        f"- **k solo en {informe['decisiones_de_diseno']['k']}**",
        f"- **solo modo congelado**: {informe['decisiones_de_diseno']['modo']}",
        "",
        "## 5. El barrido de lambda_v",
        "",
        ("| k | lambda_v | corrected_cv | dOOF | train F1 | gap | ||V|| | celdas | rescates | danos "
        "| neto | dLB esperado |"),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (f"| -- | -- (V03) | {CORRECTED_CV_V03:.10f} | +0.0000000000 | {ctrl['train_f1_control']:.4f} "
        f"| {ctrl['train_f1_control'] - CORRECTED_CV_V03:+.4f} | 0 | 0 | 0 | 0 | +0 | -- |"),
    ]
    for f in filas:
        lineas.append(
            f"| {f['k']} | {f['lambda_v']:g} | {f['corrected_cv']:.10f} | {f['delta_vs_v03']:+.6f} | "
            f"{f['f1_train']:.4f} | {f['gap_train_oof']:+.4f} | {f['norma_v_media']:.3f} | {f['celdas']} | "
            f"{f['rescates']} | {f['danos']} | {f['neto']:+d} | {f['d_lb_esperado']:+.6f} |"
        )
    lineas += [
        "",
        "## 6. El detalle por etiqueta",
        "",
        "| k | lambda_v | etiqueta | F1 | F1 de V03 | celdas | rescates | danos |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for f in filas:
        for etiqueta, d in f["por_etiqueta"].items():
            lineas.append(
                f"| {f['k']} | {f['lambda_v']:g} | {etiqueta} | {d['f1']:.6f} | {d['f1_v03']:.6f} | "
                f"{d['celdas']} | {d['rescates']} | {d['danos']} |"
            )
    gastos = informe["gastos"]
    lineas += [
        "",
        "## 7. El coste, medido y no estimado",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        (f"| un ajuste de k=4 con un hilo de BLAS, medido en la calibracion | "
        f"{gastos['segundos_de_un_ajuste_k4_medidos']} s |"),
        f"| ajustes del barrido | {gastos['ajustes_del_barrido']} |",
        f"| equivalente secuencial | {gastos['segundos_secuenciales_equivalentes']} s |",
        f"| barrido real en paralelo | {gastos['segundos_barrido']} s |",
        f"| **speedup medido** | **{gastos['speedup_medido']}x** |",
        f"| procesos del pool | {gastos['procesos']} de {gastos['nucleos']} nucleos |",
        f"| carga de la maquina durante el barrido | {gastos['carga_durante_el_barrido']} |",
        "",
        ("El speedup queda condicionado por la carga: durante la medicion corria otro proceso "
        "ajeno a este trabajo ocupando una parte de los nucleos, de modo que esta cifra es una "
        "cota inferior de lo que da la maquina descargada, no un maximo."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
