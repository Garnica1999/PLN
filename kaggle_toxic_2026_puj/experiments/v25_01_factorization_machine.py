"""V25-01: Factorization Machine sobre la representacion exacta de V03.

Que se implementa y por que
---------------------------
V03 es un modelo **lineal** sobre la union TF-IDF word(1,2) + char_wb(3,5). V23 demostro que
la no linealidad por kernel da un corrected_cv alto (0.790409) y transferencia mala (LB
0.77938, -0.00368), con 70.9 % de las muestras como vectores de soporte. V25 prueba la otra
via de no linealidad: interacciones de **segundo orden de bajo rango**, que nunca enumeran
pares y cuya complejidad crece con `k`, no con el numero de muestras.

    interaction(x) = 0.5 * sum_f ( (sum_i v_if x_i)^2 - sum_i (v_if^2 x_i^2) )

identica a `sum_{i<j} <v_i, v_j> x_i x_j`, y evaluada en `O(nnz * k)`.

El objetivo, completo y exacto
------------------------------
Se minimiza **la misma funcion que V03**, con el mismo `C`, el mismo `class_weight`, el mismo
intercepto y los mismos umbrales. Para `LogisticRegression` con `liblinear` el objetivo es

    L(w) = 0.5 * ||w||_2^2  +  C * sum_i s_i * log(1 + exp(-y_i * z_i))

con `s_i` el peso de `class_weight='balanced'` calculado **dentro del pliegue de
entrenamiento**, `z = X w + b` la decision, y el intercepto `b` **no** regularizado. V25
minimiza esa misma funcion con un unico termino anadido:

    L(w, V) = 0.5 * ||w||_2^2  +  0.5 * ||V||_F^2
            +  C * sum_i s_i * log(1 + exp(-y_i * z_i))
    z_i     = X_i w + b + interaction(x_i)

**No se cambia la regularizacion.** Los factores latentes reciben la misma penalizacion L2
que los coeficientes lineales, `0.5 * ||V||_F^2`, y no se ha explorado ninguna variante. El
intercepto no se regulariza, igual que en V03. El objetivo queda especificado antes de
interpretar ningun resultado.

El camino de control
--------------------
`interacciones=False` es una **rama explicita del codigo** que omite el termino de
interaccion y delega el ajuste en la ruta exacta de V03, `LogisticRegression` con
`liblinear`, `C=2.0`, `class_weight="balanced"`, `fit_intercept=True` y `tol=1e-4`. Esa
delegacion es necesaria y no es un atajo: la inspeccion del artefacto congelado
`model_v03.joblib` muestra `n_iter_ = [5]` y `[6]`, o sea que liblinear **se detiene en 5 o 6
iteraciones** y su solucion no es el optimo del objetivo: un L-BFGS completo llega a un
valor **menor** del mismo objetivo (845.861649 frente a 852.508395 en el pliego 1 de HS) y
cae en otro punto de una superficie de F1 plana. Como el encargo exige que el control
reproduzca V03 exactamente, la unica via honesta es usar el mismo solver y la misma
tolerancia que V03. El L-BFGS propio se conserva unicamente como diagnostico, y su
diferencia con liblinear queda reportada, no corregida.

No se usa `k=1` como control: con un factor latente por feature, el termino
`sum_{i<j} v_i v_j x_i x_j` ya es de segundo orden y **no** es un modelo lineal.

Los dos modos de la FM
---------------------
* `congelado`: `w` y `b` quedan **fijados** en la solucion de V03 y solo se optimiza `V`.
  Es el modo con mecanismo mas limpio: lo unico que cambia respecto a V03 es la existencia
  del termino de interaccion.
* `conjunto`: `w`, `b` y `V` se optimizan a la vez, partiendo de la solucion de V03 con
  `V = 0` (arranque en caliente). Alcanza un valor menor del objetivo.

Controles obligatorios
----------------------
1. **Identidad algebraica**: en un conjunto sintetico, el termino eficiente se compara
   contra `sum_{i<j} <v_i, v_j> x_i x_j` materializando **todos** los pares.
2. **Gradiente analitico contra diferencias finitas**, en el mismo conjunto sintetico, para
   `w`, para `V` y para `b`.
3. **El control reproduce V03**: probabilidades, predicciones, umbrales y metricas.
4. **Ninguna densificacion**: no existe en ningun punto una matriz densa `N x p` ni `p x p`,
   y la interaccion se evalua en `O(nnz * k)`.
5. Tiempo y memoria registrados.

Uso
---
    /home/carlos/venv/bin/python experiments/v25_01_factorization_machine.py
"""

from __future__ import annotations

import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.splits import grupos_fusionados, pliegues_corregidos

EXPERIMENTO = "V25-01 (Factorization Machine sobre la representacion exacta de V03)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v25_01_factorization_machine.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v25_01_factorization_machine.md"
OOF_CANONICO = PROJECT_ROOT / "features" / "oof_v03_final.npz"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"

ETIQUETAS = ("HS", "TR", "AG")
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
CORRECTED_CV_V03 = 0.7772114525165721
C_V03 = 2.0
TOL_V03 = 1e-4
SEMILLA = 42
PARAMETROS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMETROS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
PARAMETROS_V03 = {
    "C": C_V03,
    "class_weight": "balanced",
    "solver": "liblinear",
    "fit_intercept": True,
    "tol": TOL_V03,
    "max_iter": 1000,
    "random_state": SEMILLA,
}
VALORES_K = (1, 4, 8, 16)
MODOS = ("congelado", "conjunto")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V25-01] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def interaccion_por_fila(matriz: sparse.csr_matrix, cuadrados: sparse.csr_matrix, v: np.ndarray) -> np.ndarray:
    """Termino de interaccion de la FM para cada fila, en `O(nnz * k)`.

    La segunda mitad de la identidad se factoriza como
    `sum_j v_jf^2 x_ij^2 = x_i^2 . (sum_f v_jf^2)`, de modo que hace falta solo un producto
    disperso por vector en lugar de un recorrido elemento a elemento.

    Args:
        matriz: matriz TF-IDF dispersa CSR, nunca densificada.
        cuadrados: la misma matriz con los datos al cuadrado, para el termino `x^2`.
        v: factores latentes, forma (n_features, k).

    Returns:
        Vector de longitud `n` con la interaccion de cada fila.
    """
    productos = np.asarray(matriz @ v)
    por_feature = np.asarray(cuadrados @ np.sum(v * v, axis=1)).ravel()
    return 0.5 * (np.sum(productos * productos, axis=1) - por_feature)


def interaccion_explicita(v: np.ndarray, x: np.ndarray) -> float:
    """`sum_{i<j} <v_i, v_j> x_i x_j` materializando todos los pares.

    Solo se usa en el conjunto sintetico pequeno, para verificar la identidad algebraica.

    Args:
        v: factores latentes.
        x: vector denso de una fila.

    Returns:
        El valor de la suma sobre todos los pares.
    """
    activas = np.flatnonzero(x)
    total = 0.0
    for a in range(len(activas)):
        i = int(activas[a])
        for b in range(a + 1, len(activas)):
            j = int(activas[b])
            total += float(v[i] @ v[j]) * float(x[i]) * float(x[j])
    return total


def identidad_algebraica(semilla: int = 0) -> dict[str, Any]:
    """Verifica la identidad algebraica de la interaccion.

    Args:
        semilla: semilla del conjunto sintetico.

    Returns:
        Diccionario con la comparacion.
    """
    generador = np.random.default_rng(semilla)
    n_features, k, n_filas = 11, 3, 40
    v = generador.normal(size=(n_features, k))
    x = generador.integers(0, 4, size=(n_features, n_filas)).astype(float)
    maximo = 0.0
    for i in range(n_filas):
        vector = x[:, i]
        producto = v.T @ vector
        eficiente = 0.5 * float(np.sum(producto**2 - np.sum(v**2 * (vector**2)[:, None], axis=0)))
        maximo = max(maximo, abs(eficiente - interaccion_explicita(v, vector)))
    return {
        "n_features": n_features,
        "k": k,
        "n_filas": n_filas,
        "diferencia_maxima": maximo,
        "coincide": bool(maximo < 1e-9),
    }


def pesos_balanceados(y: np.ndarray) -> np.ndarray:
    """Peso de `class_weight='balanced'` calculado dentro del pliegue.

    Args:
        y: etiquetas binarias del entrenamiento del pliegue.

    Returns:
        Vector de pesos, uno por fila.
    """
    n = len(y)
    positivos = int(np.count_nonzero(y == 1))
    negativos = n - positivos
    pesos = np.ones(n, dtype=float)
    if positivos:
        pesos[y == 1] = n / (2.0 * positivos)
    if negativos:
        pesos[y == 0] = n / (2.0 * negativos)
    return pesos


def objetivo_y_gradiente(
    matriz: sparse.csr_matrix,
    cuadrados: sparse.csr_matrix,
    y: np.ndarray,
    pesos: np.ndarray,
    w: np.ndarray,
    v: np.ndarray,
    usar: bool,
    b: float = 0.0,
    c: float = C_V03,
    congelado: bool = False,
) -> tuple[float, np.ndarray]:
    """Valor del objetivo y gradientes analiticos respecto de `w`, `b` y `V`.

    Es la funcion que minimiza la FM, y tambien la que usa la verificacion por diferencias
    finitas. Objetivo, completo:

        L = 0.5*||w||^2 + 0.5*||V||_F^2 + C * sum_i s_i * log(1 + exp(-y_i * z_i))
        z_i = X_i w + b + interaction(x_i)

    con `b` **no** regularizado, como en V03, y con la misma penalizacion L2 en forma para
    `w` y para `V`, sin variantes exploradas.

    Args:
        matriz: matriz TF-IDF dispersa CSR.
        cuadrados: la misma matriz con los datos al cuadrado.
        y: etiquetas binarias del pliegue.
        pesos: pesos de `class_weight='balanced'` del pliegue.
        w: coeficientes lineales.
        v: factores latentes, forma (n_features, k).
        usar: si el termino de interaccion esta activo.
        b: intercepto, no regularizado.
        c: valor de C, el de V03.
        congelado: si es True, `w` y `b` estan fijos y solo se devuelve el gradiente de `V`.

    Returns:
        Tupla con el valor del objetivo y el vector de gradientes aplanado.
    """
    matriz.shape[1]
    k = v.shape[1] if v.size else 0
    objetivo = y.astype(float) * 2.0 - 1.0
    decision = np.asarray(matriz @ w).ravel() + b
    if usar and k > 0:
        decision = decision + interaccion_por_fila(matriz, cuadrados, v)
    margen = objetivo * decision
    valor = (
        0.5 * float(w @ w)
        + 0.5 * float(np.sum(v * v))
        + c * float(np.dot(pesos, np.logaddexp(0.0, -margen)))
    )
    exponente = np.exp(-np.clip(margen, -500.0, 500.0))
    # `g` es dL/dz SIN el factor C: el factor se aplica una sola vez, al proyectar con X^T.
    # Llevarlo aqui y otra vez en la proyeccion multiplicaria el gradiente por C^2, que es
    # justo lo que la verificacion por diferencias finitas detecto antes de entrenar nada.
    g = pesos * (-objetivo * exponente / (1.0 + exponente))
    g = c * g

    if not usar or k == 0:
        if congelado:
            return valor, np.zeros(0)
        grad_w = np.asarray(matriz.T @ g).ravel() + w
        return valor, np.concatenate([grad_w, np.array([float(np.sum(g))])])

    productos = np.asarray(matriz @ v)
    contrib = np.asarray(matriz.T @ (g[:, None] * productos)) - v * np.asarray(cuadrados.T @ g)[:, None]
    grad_v = contrib + v
    if congelado:
        return valor, grad_v.ravel()
    grad_w = np.asarray(matriz.T @ g).ravel() + w
    grad_b = np.array([float(np.sum(g))])
    return valor, np.concatenate([grad_w, grad_b, grad_v.ravel()])


def verificar_gradiente_finito(semilla: int = 1, paso: float = 1e-6) -> dict[str, Any]:
    """Compara los gradientes analiticos con diferencias finitas centrales.

    Args:
        semilla: semilla del conjunto sintetico.
        paso: tamano del paso de la diferencia finita.

    Returns:
        Diccionario con la comparacion de `w`, `b` y `V`.
    """
    generador = np.random.default_rng(semilla)
    n, p, k, c = 40, 9, 3, 2.0
    denso = generador.integers(0, 3, size=(n, p)).astype(float)
    denso[denso > 0] /= 3.0
    matriz = sparse.csr_matrix(denso)
    cuadrados = matriz.copy()
    cuadrados.data = cuadrados.data**2
    y = generador.integers(0, 2, size=n)
    pesos = pesos_balanceados(y)
    w = generador.normal(scale=0.3, size=p)
    v = generador.normal(scale=0.3, size=(p, k))
    b = 0.37
    valor, grad = objetivo_y_gradiente(matriz, cuadrados, y, pesos, w, v, True, b, c)
    grad_w, grad_b, grad_v = grad[:p], grad[p : p + 1], grad[p + 1 :].reshape(p, k)

    def valor_en(w_try: np.ndarray, b_try: float, v_try: np.ndarray) -> float:
        """Valor del objetivo para una perturbacion.

        Args:
            w_try: coeficientes de prueba.
            b_try: intercepto de prueba.
            v_try: factores de prueba.

        Returns:
            El valor del objetivo.
        """
        return objetivo_y_gradiente(matriz, cuadrados, y, pesos, w_try, v_try, True, b_try, c)[0]

    max_w = 0.0
    for indice in range(p):
        e = np.zeros(p)
        e[indice] = paso
        num = (valor_en(w + e, b, v) - valor_en(w - e, b, v)) / (2 * paso)
        max_w = max(max_w, abs(num - grad_w[indice]))
    arriba = valor_en(w, b + paso, v)
    abajo = valor_en(w, b - paso, v)
    error_b = abs((arriba - abajo) / (2 * paso) - float(grad_b[0]))
    max_v = 0.0
    for indice_f in range(k):
        for indice_j in range(p):
            e = np.zeros((p, k))
            e[indice_j, indice_f] = paso
            num = (valor_en(w, b, v + e) - valor_en(w, b, v - e)) / (2 * paso)
            max_v = max(max_v, abs(num - grad_v[indice_j, indice_f]))
    escala_w = max(float(np.abs(grad_w).max()), 1e-12)
    escala_v = max(float(np.abs(grad_v).max()), 1e-12)
    escala_b = max(abs(float(grad_b[0])), 1e-12)
    return {
        "n": n,
        "p": p,
        "k": k,
        "c": c,
        "intercepto": b,
        "paso": paso,
        "valor": valor,
        "error_relativo_grad_w": float(max_w / escala_w),
        "error_relativo_grad_b": float(error_b / escala_b),
        "error_relativo_grad_v": float(max_v / escala_v),
        "coincide": bool(max_w / escala_w < 1e-6 and error_b / escala_b < 1e-6 and max_v / escala_v < 1e-6),
    }


def ajustar_lineal_v03(matriz: sparse.csr_matrix, y: np.ndarray) -> tuple[np.ndarray, float, dict[str, Any]]:
    """El camino de control: la ruta exacta de V03, sin termino de interaccion.

    Es la rama explicita que omite la interaccion. Delega en `LogisticRegression` con los
    parametros exactos de V03, porque el artefacto congelado `model_v03.joblib` registra
    `n_iter_ = [5]`, es decir que liblinear se detiene en 5 o 6 iteraciones y su solucion no
    es el optimo del objetivo. Sin esta delegacion el control no puede ser identico a V03.

    Args:
        matriz: matriz TF-IDF dispersa del pliegue.
        y: etiquetas binarias.

    Returns:
        Tupla con `w`, el intercepto `b` y un diccionario de diagnostico.
    """
    inicio = time.time()
    modelo = LogisticRegression(**PARAMETROS_V03)
    modelo.fit(matriz, y)
    diagnostico = {
        "segundos": round(time.time() - inicio, 4),
        "iteraciones": int(modelo.n_iter_[0]),
        "n_parametros": int(matriz.shape[1] + 1),
    }
    return modelo.coef_.ravel().astype(float), float(modelo.intercept_[0]), diagnostico


def ajustar_fm(
    matriz: sparse.csr_matrix,
    cuadrados: sparse.csr_matrix,
    y: np.ndarray,
    w_inicial: np.ndarray,
    b_inicial: float,
    k: int,
    congelado: bool,
    c: float = C_V03,
    iteraciones: int = 800,
    semilla: int = SEMILLA,
    escala_inicial: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Ajusta la FM sobre un pliegue, partiendo de la solucion de V03.

    Con `k=0` el termino de interaccion se omite por una rama explicita del codigo, y con
    `congelado=True` `w` y `b` quedan fijos en la solucion de V03 y solo se optimiza `V`,
    de modo que lo unico que cambia respecto a V03 es la existencia del termino.

    Args:
        matriz: matriz TF-IDF dispersa del pliegue.
        cuadrados: la misma matriz con los datos al cuadrado.
        y: etiquetas binarias.
        w_inicial: coeficientes lineales de partida, los de V03.
        b_inicial: intercepto de partida, el de V03.
        k: dimension latente.
        congelado: si `w` y `b` quedan fijos.
        c: valor de C, el de V03.
        iteraciones: maximo de iteraciones del optimizador.
        semilla: semilla del arranque aleatorio de `V`.
        escala_inicial: desviacion tipica del arranque de `V`.

    Returns:
        Tupla con `w`, `V`, el intercepto y el diagnostico.
    """
    p = matriz.shape[1]
    usar = k > 0
    pesos = pesos_balanceados(y)
    # `v = 0` es un punto estacionario EXACTO de este objetivo: la interaccion es par en `v`,
    # asi que su derivada en el origen vale cero y un optimizador de gradiente que arranque
    # ahi no se mueve nunca. Por eso `v` se inicializa al azar, como en la practica estandar
    # de las Factorization Machines. La semilla es fija, de modo que el arranque es
    # reproducible; `w` y `b` si arrancan en la solucion exacta de V03.
    if usar:
        v0 = np.random.default_rng(semilla).normal(scale=escala_inicial, size=p * k)
    else:
        v0 = np.zeros(0)
    if congelado:
        x0 = v0
    elif usar:
        x0 = np.concatenate([w_inicial, [b_inicial], v0])
    else:
        x0 = np.concatenate([w_inicial, [b_inicial]])

    def calcular(theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Empaqueta valor y gradiente aplanado para el optimizador.

        Args:
            theta: vector de parametros.

        Returns:
            Tupla con el valor y el gradiente aplanado.
        """
        if not usar:
            if congelado:
                return objetivo_y_gradiente(matriz, cuadrados, y, pesos, w_inicial, np.zeros((p, 0)), False, b_inicial, c, True)
            return objetivo_y_gradiente(matriz, cuadrados, y, pesos, theta[:p], np.zeros((p, 0)), False, float(theta[p]), c, False)
        inicio_v = 0 if congelado else p + 1
        v = theta[inicio_v : inicio_v + p * k].reshape(p, k)
        w = w_inicial if congelado else theta[:p]
        b = b_inicial if congelado else float(theta[p])
        return objetivo_y_gradiente(matriz, cuadrados, y, pesos, w, v, True, b, c, congelado)

    inicio = time.time()
    resultado = minimize(
        calcular,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": iteraciones, "maxcor": 20, "ftol": 1e-15, "gtol": 1e-9},
    )
    if not usar:
        w = w_inicial if congelado else resultado.x[:p]
        b = b_inicial if congelado else float(resultado.x[p])
        v = np.zeros((p, 0))
    elif congelado:
        w, b = w_inicial, b_inicial
        v = resultado.x.reshape(p, k)
    else:
        w = resultado.x[:p]
        b = float(resultado.x[p])
        v = resultado.x[p + 1 : p + 1 + p * k].reshape(p, k)
    diagnostico = {
        "valor_final": float(resultado.fun),
        "iteraciones": int(resultado.nit),
        "convergio": bool(resultado.success),
        "norma_gradiente": float(np.linalg.norm(resultado.jac)),
        "segundos": round(time.time() - inicio, 3),
        "norma_w": float(np.linalg.norm(w)),
        "norma_v": float(np.linalg.norm(v)) if usar else 0.0,
        "n_parametros": int(p * k + (0 if congelado else p + 1)),
        "desplazamiento_w": float(np.linalg.norm(w - w_inicial)) if usar else 0.0,
    }
    return w, v, b, diagnostico


def predecir(
    matriz: sparse.csr_matrix,
    cuadrados: sparse.csr_matrix,
    w: np.ndarray,
    v: np.ndarray,
    b: float,
) -> np.ndarray:
    """Decision y probabilidad con la sigmoidal logistica.

    Args:
        matriz: matriz dispersa.
        cuadrados: la misma matriz con los datos al cuadrado.
        w: coeficientes lineales.
        v: factores latentes, vacios si las interacciones estan desactivadas.
        b: intercepto.

    Returns:
        Vector de probabilidades.
    """
    decision = np.asarray(matriz @ w).ravel() + b
    if v.size:
        decision = decision + interaccion_por_fila(matriz, cuadrados, v)
    return 1.0 / (1.0 + np.exp(-decision))


def macro_f1(verdad: np.ndarray, predicho: np.ndarray) -> float:
    """Macro-F1 de las tres clases positivas.

    Args:
        verdad: matriz verdadera binaria.
        predicho: matriz predicha binaria.

    Returns:
        La media de las tres F1.
    """
    valores = []
    for j in range(verdad.shape[1]):
        comunes = int(np.count_nonzero(verdad[:, j] & predicho[:, j]))
        denominador = int(np.count_nonzero(verdad[:, j])) + int(np.count_nonzero(predicho[:, j]))
        valores.append(2 * comunes / denominador if denominador > 0 else 0.0)
    return float(np.mean(valores))


def main() -> None:
    """Ejecuta los controles obligatorios, el control de V03 y la progresion de k.

    Returns:
        None.
    """
    inicio = time.time()
    log("V25-01: Factorization Machine. Ninguna submission en este script")

    log("1/8 identidad algebraica de la interaccion, con todos los pares materializados")
    algebraica = identidad_algebraica()
    log(
        f"   {algebraica['n_features']} features x {algebraica['k']} factores x "
        f"{algebraica['n_filas']} filas | diferencia maxima "
        f"{algebraica['diferencia_maxima']:.3e} | coincide: {algebraica['coincide']}"
    )
    if not algebraica["coincide"]:
        raise SystemExit("PARADA OBLIGATORIA: la identidad algebraica no se cumple")

    log("2/8 gradiente analitico contra diferencias finitas, para w, b y V")
    gradiente = verificar_gradiente_finito()
    log(
        f"   w {gradiente['error_relativo_grad_w']:.3e} | b "
        f"{gradiente['error_relativo_grad_b']:.3e} | V "
        f"{gradiente['error_relativo_grad_v']:.3e} | coincide: {gradiente['coincide']}"
    )
    if not gradiente["coincide"]:
        raise SystemExit("PARADA OBLIGATORIA: el gradiente analitico no cuadra con el numerico")

    log("3/8 desarrollo, pliegues canonicos y alineacion por identificador")
    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    grupos_full = grupos_fusionados(train)
    mapa_grupo = dict(zip(train["id"].astype(int), grupos_full.tolist()))
    grupos_dev = np.array([mapa_grupo[i] for i in desarrollo["id"].astype(int)])
    pliegues = pliegues_corregidos(desarrollo, grupos_dev)
    y = desarrollo[list(ETIQUETAS)].to_numpy().astype(int)
    textos = desarrollo["text"]
    oof = np.load(OOF_CANONICO, allow_pickle=True)
    y_oof = np.asarray(oof["y_true"], dtype=int)
    mapa_posicion = {v: i for i, v in enumerate(desarrollo["id"].astype(int).to_numpy())}
    posiciones = np.array([mapa_posicion[i] for i in np.asarray(oof["ids"]).astype(np.int64)])
    if not np.array_equal(y[posiciones], y_oof):
        raise SystemExit("PARADA OBLIGATORIA: las etiquetas no coinciden tras alinear por id")
    proba_v03 = np.asarray(oof["proba"], dtype=float)
    pred_v03 = (proba_v03 >= UMBRALES_V03[None, :]).astype(int)
    log(f"   desarrollo {len(desarrollo)} | {len(pliegues)} pliegues | alineacion verificada")

    matrices: list[dict[str, Any]] = []
    for indice, (entrenar, validar) in enumerate(pliegues):
        palabra = TfidfVectorizer(**PARAMETROS_WORD)
        caracter = TfidfVectorizer(**PARAMETROS_CHAR)
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
        matrices.append(
            {
                "entrenar": entrenar,
                "validar": validar,
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
    p_real = matrices[0]["x_ent"].shape[1]
    pares = p_real * (p_real - 1) // 2
    log(f"   nunca se crea una matriz densa N x p ({len(desarrollo)} x {p_real}) ni p x p")
    log(f"   pares que NO se materializan: {pares:,}")
    log(f"   memoria pico hasta aqui: {memoria_pico_mb():.0f} MB")

    log("4/8 el CONTROL: ruta de V03 con las interacciones desactivadas")
    proba_control = np.zeros((len(desarrollo), len(ETIQUETAS)))
    lineales: list[dict[str, Any]] = []
    f1_train_control: list[float] = []
    segundos_control = 0.0
    objetivo_control = 0.0
    objetivo_lbfgs = 0.0
    n_iter_liblinear: list[int] = []
    for indice, m in enumerate(matrices):
        entrenar, validar = m["entrenar"], m["validar"]
        for j in range(len(ETIQUETAS)):
            w, b, diag = ajustar_lineal_v03(m["x_ent"], y[entrenar, j])
            segundos_control += diag["segundos"]
            n_iter_liblinear.append(diag["iteraciones"])
            proba_control[validar, j] = predecir(m["x_val"], m["c_val"], w, np.zeros((p_real, 0)), b)
            pesos = pesos_balanceados(y[entrenar, j])
            valor, _ = objetivo_y_gradiente(m["x_ent"], m["c_ent"], y[entrenar, j], pesos, w, np.zeros((p_real, 0)), False, b)
            objetivo_control += valor
            ajuste = predecir(m["x_ent"], m["c_ent"], w, np.zeros((p_real, 0)), b)
            comunes = int(np.count_nonzero(y[entrenar, j] & (ajuste >= 0.5)))
            den = int(np.count_nonzero(y[entrenar, j])) + int(np.count_nonzero(ajuste >= 0.5))
            f1_train_control.append(2 * comunes / den if den else 0.0)
            # Diagnostico: el mismo objetivo resuelto con L-BFGS completo, sin regularizar el
            # intercepto, para medir cuanto se aleja liblinear del optimo real.
            w_alt, v_alt, b_alt, _ = ajustar_fm(
                m["x_ent"], m["c_ent"], y[entrenar, j], w, b, 0, False, iteraciones=3000
            )
            objetivo_lbfgs += objetivo_y_gradiente(
                m["x_ent"], m["c_ent"], y[entrenar, j], pesos, w_alt, v_alt, False, b_alt
            )[0]
            lineales.append({"w": w, "b": b})
    proba_control_oof = proba_control[posiciones]
    pred_control = (proba_control_oof >= UMBRALES_V03[None, :]).astype(int)
    control = macro_f1(y_oof, pred_control)
    error_prob = float(np.max(np.abs(proba_control_oof - proba_v03)))
    celdas_iguales = bool(np.array_equal(pred_control, pred_v03))
    log(f"   FM-control {control:.16f} | V03 esperado {CORRECTED_CV_V03:.16f}")
    log(f"   diferencia de corrected_cv: {control - CORRECTED_CV_V03:+.3e}")
    log(f"   diferencia maxima de probabilidad: {error_prob:.3e} | predicciones identicas: {celdas_iguales}")
    log(f"   train F1 medio a 0.5: {float(np.mean(f1_train_control)):.4f}")
    log(f"   iteraciones de liblinear por ajuste: min {min(n_iter_liblinear)} max {max(n_iter_liblinear)}")
    log(f"   objetivo del control {objetivo_control:.6f} | con L-BFGS completo {objetivo_lbfgs:.6f}")
    log(f"  Diagnostico: liblinear queda {objetivo_lbfgs - objetivo_control:+.6f} por encima del optimo real")
    log(f"   segundos del control: {segundos_control:.1f}")
    log(f"   memoria pico: {memoria_pico_mb():.0f} MB")
    if abs(control - CORRECTED_CV_V03) > 1e-12 or not celdas_iguales or error_prob > 1e-12:
        REPORTE_JSON.write_text(
            json.dumps(
                {
                    "experimento": EXPERIMENTO,
                    "estado": "PARADA_EL_CONTROL_NO_REPRODUCE_V03",
                    "control": control,
                    "v03_esperado": CORRECTED_CV_V03,
                    "diferencia_maxima_de_probabilidad": error_prob,
                    "predicciones_identicas": celdas_iguales,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        log(f"VEREDICTO: control incorrecto ({control:.10f} frente a {CORRECTED_CV_V03:.10f})")
        return
    log("   **el control reproduce V03 exactamente**: se puede continuar con k > 0")

    log("5/8 la progresion de k, en modo congelado y en modo conjunto")
    filas: list[dict[str, Any]] = []
    for modo in MODOS:
        for k in VALORES_K:
            proba = np.zeros((len(desarrollo), len(ETIQUETAS)))
            f1_train: list[float] = []
            segundos = 0.0
            iteraciones = 0
            n_params = 0
            norma_v = 0.0
            no_convergieron = 0
            for indice, m in enumerate(matrices):
                entrenar, validar = m["entrenar"], m["validar"]
                for j in range(len(ETIQUETAS)):
                    ini = lineales[indice * len(ETIQUETAS) + j]
                    w, v, b, diag = ajustar_fm(
                        m["x_ent"],
                        m["c_ent"],
                        y[entrenar, j],
                        ini["w"],
                        ini["b"],
                        k,
                        modo == "congelado",
                        semilla=SEMILLA + 1000 * indice + 10 * j,
                    )
                    segundos += diag["segundos"]
                    iteraciones += diag["iteraciones"]
                    n_params = diag["n_parametros"]
                    norma_v += diag["norma_v"]
                    no_convergieron += 0 if diag["convergio"] else 1
                    proba[validar, j] = predecir(m["x_val"], m["c_val"], w, v, b)
                    ajuste = predecir(m["x_ent"], m["c_ent"], w, v, b)
                    comunes = int(np.count_nonzero(y[entrenar, j] & (ajuste >= 0.5)))
                    den = int(np.count_nonzero(y[entrenar, j])) + int(np.count_nonzero(ajuste >= 0.5))
                    f1_train.append(2 * comunes / den if den else 0.0)
            proba_oof = proba[posiciones]
            pred = (proba_oof >= UMBRALES_V03[None, :]).astype(int)
            valor = macro_f1(y_oof, pred)
            train_f1 = float(np.mean(f1_train))
            cambio = pred != pred_v03
            rescates = int(np.count_nonzero(cambio & (pred_v03 != y_oof) & (pred == y_oof)))
            danos = int(np.count_nonzero(cambio & (pred_v03 == y_oof) & (pred != y_oof)))
            por_etiqueta = {}
            for j, etiqueta in enumerate(ETIQUETAS):
                cj = pred[:, j] != pred_v03[:, j]
                por_etiqueta[etiqueta] = {
                    "f1": macro_f1(y_oof[:, [j]], pred[:, [j]]),
                    "f1_v03": macro_f1(y_oof[:, [j]], pred_v03[:, [j]]),
                    "celdas": int(np.count_nonzero(cj)),
                    "rescates": int(np.count_nonzero(cj & (pred_v03[:, j] != y_oof[:, j]) & (pred[:, j] == y_oof[:, j]))),
                    "danos": int(np.count_nonzero(cj & (pred_v03[:, j] == y_oof[:, j]) & (pred[:, j] != y_oof[:, j]))),
                }
            filas.append(
                {
                    "modo": modo,
                    "k": k,
                    "n_parametros": n_params,
                    "corrected_cv": valor,
                    "delta_vs_v03": valor - CORRECTED_CV_V03,
                    "f1_train": train_f1,
                    "gap_train_oof": train_f1 - valor,
                    "celdas": int(np.count_nonzero(cambio)),
                    "rescates": rescates,
                    "danos": danos,
                    "neto": rescates - danos,
                    "norma_v_media": norma_v / (len(matrices) * len(ETIQUETAS)),
                    "por_etiqueta": por_etiqueta,
                    "segundos": round(segundos, 1),
                    "iteraciones_totales": iteraciones,
                    "ajustes_sin_converger": no_convergieron,
                }
            )
            f = filas[-1]
            log(
                f"   {modo:<9} k={k:<3} params {f['n_parametros']:>8} | corrected_cv {valor:.6f} "
                f"({f['delta_vs_v03']:+.6f}) | train {train_f1:.4f} | gap {f['gap_train_oof']:+.4f} | "
                f"celdas {f['celdas']:>4} | rescates {rescates:>4} | danos {danos:>4} | "
                f"NETO {f['neto']:>+4} | {f['segundos']:.0f} s | sin converger {no_convergieron}"
            )

    log("6/8 la prediccion de transferencia del modelo de V24")
    predicciones = []
    for f in filas:
        esperado = -0.00920 + 0.2448 * f["delta_vs_v03"]
        f["d_lb_esperado"] = esperado
        predicciones.append(
            {
                "modo": f["modo"],
                "k": f["k"],
                "d_oof": f["delta_vs_v03"],
                "d_lb_esperado": esperado,
                "supera_a_v03": bool(esperado > 0.0),
            }
        )
        log(
            f"   {f['modo']:<9} k={f['k']:<3} dOOF {f['delta_vs_v03']:+.6f} -> dLB esperado "
            f"{esperado:+.6f} | supera a V03: {esperado > 0.0}"
        )

    log("7/8 tabla pedida antes de interpretar")
    log("   modelo     interactions  params   corrected_cv      train F1      gap")
    log(f"   V03        OFF              {p_real:>6}  {CORRECTED_CV_V03:.10f}    0.9614    +0.1842")
    log(
        f"   FM-control OFF              {p_real:>6}  {control:.10f}    "
        f"{float(np.mean(f1_train_control)):.4f}    {float(np.mean(f1_train_control)) - control:+.4f}"
    )
    for f in filas:
        if f["k"] in (1, 4):
            log(
                f"   FM {f['modo']} k={f['k']:<2} ON     {f['n_parametros']:>9}  "
                f"{f['corrected_cv']:.10f}    {f['f1_train']:.4f}    {f['gap_train_oof']:+.4f}"
            )

    log("8/8 informe")
    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "objetivo": (
            "L = 0.5*||w||^2 + 0.5*||V||_F^2 + C * sum_i s_i * log(1 + exp(-y_i * z_i)); "
            "z_i = X_i w + b + 0.5 * sum_f ((sum_j v_jf x_ij)^2 - sum_j v_jf^2 x_ij^2). "
            "El intercepto b NO se regulariza, igual que en V03. La penalizacion L2 de V es "
            "identica en forma a la de w y no se ha explorado ninguna variante. class_weight, C, "
            "representacion, intercepto y umbrales son los de V03."
        ),
        "controles": {
            "identidad_algebraica": algebraica,
            "gradiente_por_diferencias_finitas": gradiente,
            "control_reproduce_v03": {
                "corrected_cv": control,
                "v03_esperado": CORRECTED_CV_V03,
                "diferencia": control - CORRECTED_CV_V03,
                "diferencia_maxima_de_probabilidad": error_prob,
                "predicciones_identicas": celdas_iguales,
                "train_f1_control": float(np.mean(f1_train_control)),
                "iteraciones_liblinear": {
                    "min": min(n_iter_liblinear),
                    "max": max(n_iter_liblinear),
                },
                "objetivo_del_control": objetivo_control,
                "objetivo_con_lbfgs_completo": objetivo_lbfgs,
                "distancia_de_liblinear_al_optimo": objetivo_lbfgs - objetivo_control,
            },
            "sin_densificacion": {
                "n_features": int(p_real),
                "pares_no_materializados": int(pares),
                "coste_de_la_interaccion": "O(nnz * k)",
                "matriz_densa_Nx_p": "nunca se construye",
                "matriz_densa_p_x_p": "nunca se construye",
            },
        },
        "resultados": filas,
        "prediccion_de_transferencia_v24": predicciones,
        "gastos": {"submissions": 0, "memoria_pico_mb": round(memoria_pico_mb(), 1)},
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _informe_md(informe, inicio)
    log("   informes escritos en reports/v25_01_factorization_machine.{json,md}")
    log(f"   memoria pico: {memoria_pico_mb():.0f} MB")
    log(f"completado en {round(time.time() - inicio, 1)} s")
    mejor = max(filas, key=lambda f: f["corrected_cv"])
    log(
        f"VEREDICTO: mejor {mejor['modo']} k={mejor['k']} con corrected_cv {mejor['corrected_cv']:.6f} "
        f"({mejor['delta_vs_v03']:+.6f}) | dLB esperado {mejor['d_lb_esperado']:+.6f} | submissions 0"
    )


def _informe_md(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.
        inicio: instante de arranque.

    Returns:
        None.
    """
    c = informe["controles"]
    filas = informe["resultados"]
    ctrl = c["control_reproduce_v03"]
    p_real = c["sin_densificacion"]["n_features"]
    lineas = [
        "# V25-01: Factorization Machine sobre la representacion exacta de V03",
        "",
        f"Generado por `experiments/v25_01_factorization_machine.py` ({EXPERIMENTO}).",
        "",
        "**Ninguna submission.**",
        "",
        "## 1. El objetivo, completo",
        "",
        "```",
        informe["objetivo"],
        "```",
        "",
        "## 2. Los controles obligatorios",
        "",
        "| control | resultado |",
        "|---|---|",
        (f"| identidad algebraica, frente a la suma sobre todos los pares | diferencia maxima "
        f"{c['identidad_algebraica']['diferencia_maxima']:.3e} |"),
        (f"| gradiente analitico contra diferencias finitas | `w` "
        f"{c['gradiente_por_diferencias_finitas']['error_relativo_grad_w']:.3e}, `b` "
        f"{c['gradiente_por_diferencias_finitas']['error_relativo_grad_b']:.3e}, `V` "
        f"{c['gradiente_por_diferencias_finitas']['error_relativo_grad_v']:.3e} |"),
        f"| el control reproduce V03 | {ctrl['corrected_cv']:.16f} frente a {ctrl['v03_esperado']:.16f} |",
        f"| predicciones identicas a las de V03 | {ctrl['predicciones_identicas']} |",
        f"| diferencia maxima de probabilidad | {ctrl['diferencia_maxima_de_probabilidad']:.3e} |",
        f"| pares que NO se materializan | {c['sin_densificacion']['pares_no_materializados']:,} |",
        f"| coste de la interaccion | {c['sin_densificacion']['coste_de_la_interaccion']} |",
        "",
        "## 3. La distancia de liblinear al optimo del objetivo",
        "",
        ("El artefacto congelado `model_v03.joblib` registra `n_iter_` de "
        f"{ctrl['iteraciones_liblinear']['min']} a {ctrl['iteraciones_liblinear']['max']} iteraciones: "
        "liblinear se detiene muy pronto y su solucion **no** es el optimo del objetivo que "
        "minimiza. Sumando el objetivo sobre los quince ajustes:"),
        "",
        "| optimizador | valor del objetivo |",
        "|---|---:|",
        f"| liblinear, la ruta de V03 | {ctrl['objetivo_del_control']:.6f} |",
        f"| L-BFGS completo, mismo objetivo | {ctrl['objetivo_con_lbfgs_completo']:.6f} |",
        f"| diferencia | {ctrl['distancia_de_liblinear_al_optimo']:+.6f} |",
        "",
        ("Por eso el camino de control **delega** en la ruta exacta de V03: el L-BFGS propio "
        "alcanza un valor menor del mismo objetivo, pero cae en otro punto de una superficie de "
        "F1 plana y no puede reproducir V03. El L-BFGS queda como diagnostico, no como ruta."),
        "",
        "## 4. La progresion de k",
        "",
        ("| modo | k | interactions | parametros | corrected_cv | dOOF | train F1 | gap | celdas "
        "| rescates | danos | neto | dLB esperado |"),
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (f"| V03 | -- | OFF | {p_real} | {CORRECTED_CV_V03:.10f} | +0.0000000000 | 0.9614 | +0.1842 "
        "| 0 | 0 | 0 | +0 | -- |"),
        (f"| FM-control | -- | OFF | {p_real} | {ctrl['corrected_cv']:.10f} | "
        f"{ctrl['corrected_cv'] - CORRECTED_CV_V03:+.10f} | {ctrl['train_f1_control']:.4f} | "
        f"{ctrl['train_f1_control'] - ctrl['corrected_cv']:+.4f} | 0 | 0 | 0 | +0 | -- |"),
    ]
    for f in filas:
        lineas.append(
            f"| FM {f['modo']} | {f['k']} | ON | {f['n_parametros']} | {f['corrected_cv']:.10f} | "
            f"{f['delta_vs_v03']:+.6f} | {f['f1_train']:.4f} | {f['gap_train_oof']:+.4f} | "
            f"{f['celdas']} | {f['rescates']} | {f['danos']} | {f['neto']:+d} | "
            f"{f['d_lb_esperado']:+.6f} |"
        )
    lineas += [
        "",
        "## 5. El detalle por etiqueta",
        "",
        "| modo | k | etiqueta | F1 | F1 de V03 | celdas | rescates | danos |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for f in filas:
        for etiqueta, d in f["por_etiqueta"].items():
            lineas.append(
                f"| {f['modo']} | {f['k']} | {etiqueta} | {d['f1']:.6f} | {d['f1_v03']:.6f} | "
                f"{d['celdas']} | {d['rescates']} | {d['danios']} |"
            )
    lineas += [
        "",
        "## 6. El umbral de transferencia",
        "",
        ("El modelo de V24 fija `dLB = -0.00920 + 0.2448 * dOOF`. Para superar a V03 hacen "
        "falta `dOOF = +0.03756`, y el techo historico es `+0.0132` (V23)."),
        "",
        (f"Memoria maxima residente: {informe['gastos']['memoria_pico_mb']:.0f} MB. "
        f"Completado en {round(time.time() - inicio, 1)} s."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
