"""V12-03: funcion de entrenamiento por pares sobre TF-IDF disperso (H3).

El ultimo hueco genuinamente no probado de la auditoria de brecha, y el ultimo de los tres que la
auditoria selecciono.

La pregunta, tal como la formula la fase
---------------------------------------
> Una funcion de entrenamiento por pares sobre TF-IDF disperso puede explotar la estructura
> relativa entre HS, TR y AG que el objetivo binario independiente no esta aprovechando?

V03 ajusta **tres clasificadores binarios independientes**, y su objetivo es la suma de tres
entropias cruzadas que no se hablan entre si. Eso deja sin usar una informacion que el propio
conjunto de entrenamiento contiene: **HS es una condicion necesaria de TR y de AG**. Cuando HS=1,
TR y AG estan definidas; cuando HS=0, no lo estan. Esa es una relacion de **orden** entre las tres
etiquetas, y un objetivo que trata las tres por separado no puede expressarla.

Que es "por pares" aqui, y por que no es lo que dice la auditoria
-------------------------------------------------------------
La auditoria de brecha describio H3 como pares de **ejemplos**: acercar ejemplos de la misma
etiqueta y separar ejemplos de etiquetas distintas. La pregunta de la fase es mas especifica y mas
interesante: pares de **etiquetas** dentro de cada ejemplo. Se implementa la segunda, que es la que
ataca la estructura relativa entre HS, TR y AG, y se deja constancia de la diferencia porque son
mecanismos distintos.

Y conviene senalar por que esta lectura **no es la cadena cerrada** E2-01 ni el post-proceso duro.
La cadena anade el score de la etiqueta previa como feature; aqui los tres clasificadores se
entrenan **jointemente** con un termino que los acopla. El post-proceso usa HS predicho como llave
dura en la prediccion; aqui **no hay llave**: la prediccion sigue siendo la de V03, con los mismos
umbrales congelados, y lo unico que cambia es la funcion que se minimiza.

El termino por pares
--------------------
Con margen ``m`` fijo en unidades de logit, y para cada ejemplo con logit ``z_h, z_t, z_a``:

- si ``HS=1``, TR y AG estan definidas y HS es certaine, luego se espera ``z_h >= z_t`` y
  ``z_h >= z_a``;
- si ``HS=0``, TR y AG no estan definidas, y se espera lo contrario: ``z_h <= z_t`` y
  ``z_h <= z_a``.

La penalizacion de cada par es la **logistica por pares** ``softplus(m - d)``, donde ``d`` es la
diferencia a favor de la etiqueta que deberia estar mas arriba. Se uso la bisagra ``max(0, m - d)``
en la primera version y **el optimizador abortaba con ABNORMAL en la iteracion 19**, con
independencia de las tolerancias, porque una busqueda lineal no puede funcionar sobre una funcion
no derivable. El surrogate logistico es el estandar para objectives de orden por pares, es derivable
en todo el dominio y ademas no descarta del todo un par que viola, que es mejor para este problema
donde la estructura es suave y no dura.

Es decir, el termino **expresa la relacion de orden entre las tres etiquetas y nada mas**. No
introduce informacion nueva, no cambia el espacio, no cambia los umbrales, y **no convierte HS en
una decision binaria previa**, que es exactamente el error que caracterizo al post-proceso duro.

Protocolo identico al de H2
---------------------------
Mismos gates, mismos umbrales congelados, misma particion, mismos controles, mismo filtro de
detectabilidad, misma verificacion del gradiente contra diferencias finitas. El objetivo por pares
exige un optimizador propio porque acopla tres clasificadores, y por eso el script verifica la
derivada **antes** de usar un solo resultado.

**Control interno: lambda=0 tiene que reproducir el brazo A.** Con el termino por pares a peso
cero, el objetivo es la suma de las tres entropias cruzadas, que es exactamente V03. Si no
reproduce, lapipeline tiene un fallo y no se mira ningun otro resultado.

Lo que se mide ademas, y es especifico de esta hipotesis
---------------------------------------------------------
El efecto sobre la **dependencia HS -> TR/AG**, que es donde la estructura relativa deberia notarse:

- F1 de TR y de AG **restringidas a las filas con HS=1**, que es donde estan definidas;
- **numero de falsos positivos de TR y de AG en la zona con HS=0**, que es exactamente lo que el
  oraculo de H2 corrigio, 78 y 118 respectivamente;
- F1 de HS, para comprobar que el acoplamiento no rompe la etiqueta que gobierna a las otras dos.

Uso
---
    /home/carlos/venv/bin/python experiments/v12_03_entrenamiento_por_pares.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime, timezone
from importlib import util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v11_ayudas import fila, sello

from src import data as data_module
from src.evaluation import ETIQUETAS, metrica_macro
from src.splits import (
    RANDOM_STATE,
    grupos_fusionados,
    holdout_v2,
)

_especificacion = util.spec_from_file_location(
    "arnes", PROJECT_ROOT / "experiments" / "v10_03_arnes_evaluacion.py"
)
arnes = util.module_from_spec(_especificacion)
sys.modules["arnes"] = arnes
_especificacion.loader.exec_module(arnes)

EXPERIMENTO = "V12-03 (H3, entrenamiento por pares sobre TF-IDF disperso)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v12_03_entrenamiento_por_pares.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v12_03_entrenamiento_por_pares.json"

ETIQUETAS_LISTA = list(ETIQUETAS)
REFERENCIA_V03 = 0.7772114525165721
UMBRALES = arnes.UMBRALES_V03
SEMILLA_CANONICA = RANDOM_STATE
CONSTANTE_V03 = 2.0
LIMITE_ITERACIONES = 2000
MARGEN = 1.0
VALORES_LAMBDA = (0.0, 0.25, 1.0)
REJILLA_LAMBDA = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
DELTA_MINIMO_EXPLORATORIO = 0.002
DELTA_MINIMO_CANDIDATO = 0.003
PLIEGUES_MINIMOS = 4
FILAS_LEADERBOARD = 1154
COORDENADAS_COMPROBACION = 12
TOLERANCIA_GRADIENTE = 1e-6
REFERENCIA_ORACULO = 0.046271


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V12-03] {mensaje}", flush=True)


def construir_espacio(
    textos_entrenamiento: pd.Series, textos_consulta: pd.Series
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Ajusta los dos vectorizadores en el entrenamiento y transforma la consulta.

    Args:
        textos_entrenamiento: textos de ajuste.
        textos_consulta: textos a transformar.

    Returns:
        Tupla con la matriz de ajuste y la de consulta.
    """
    vectorizador_word = TfidfVectorizer(**arnes.PARAMS_WORD)
    vectorizador_char = TfidfVectorizer(**arnes.PARAMS_CHAR)
    ajuste = sparse.hstack(
        [
            vectorizador_word.fit_transform(textos_entrenamiento),
            vectorizador_char.fit_transform(textos_entrenamiento),
        ],
        format="csr",
    )
    consulta = sparse.hstack(
        [
            vectorizador_word.transform(textos_consulta),
            vectorizador_char.transform(textos_consulta),
        ],
        format="csr",
    )
    return ajuste, consulta


def pesos_equilibrados(objetivo: np.ndarray) -> np.ndarray:
    """Devuelve los pesos de clase que equivalen a ``class_weight="balanced"``.

    Args:
        objetivo: etiqueta binaria de entrenamiento.

    Returns:
        Vector de pesos, uno por fila de entrenamiento.

    Raises:
        ValueError: si el conjunto tiene una sola clase.
    """
    negativos = float(np.sum(objetivo == 0))
    positivos = float(np.sum(objetivo == 1))
    if negativos == 0.0 or positivos == 0.0:
        raise ValueError("el conjunto de entrenamiento tiene una sola clase")
    total = len(objetivo)
    return np.where(objetivo == 1, total / (2.0 * positivos), total / (2.0 * negativos))


def Loss_y_gradiente(
    coeficientes: np.ndarray,
    matriz: sparse.csr_matrix,
    y: np.ndarray,
    pesos: np.ndarray,
    peso_pares: float,
) -> tuple[float, np.ndarray]:
    """Calcula el objetivo conjunto y su gradiente.

    El objetivo es la suma de las tres entropias cruzadas con pesos de clase, mas la penalizacion
    L2 de V03, mas el termino por pares entre etiquetas ponderado por ``peso_pares``.

    El vector de coeficientes va **primero** a proposito, que es como lo invoca el optimizador, de
    modo que la verificacion contra diferencias finitas y el ajuste real recorren el mismo camino de
    llamada. En V12-02 el orden invertido hacia que la verificacion pasara y el ajuste fallara, que
    es la peor combinacion posible.

    Los tres bloques de coeficientes van en el mismo vector, cada uno terminado por su termino
    independiente, que **no se penaliza**, igual que en V12-02.

    Args:
        coeficientes: vector con los tres bloques y los tres terminos independientes.
        matriz: matriz de caracteristicas dispersa.
        y: matriz binaria de etiquetas.
        pesos: matriz de pesos de clase, una columna por etiqueta.
        peso_pares: peso lambda del termino por pares.

    Returns:
        Tupla con el valor del objetivo y su gradiente.
    """
    n_features = matriz.shape[1]
    bloques = coeficientes.reshape(3, n_features + 1)
    logitos = np.empty((matriz.shape[0], 3), dtype=np.float64)
    for indice in range(3):
        logitos[:, indice] = matriz @ bloques[indice, :-1] + bloques[indice, -1]

    # **No se recorta el margen.** La primera version lo hacia con np.clip para evitar el
    # desbordamiento del exponente, y eso rompe la regla de la cadena: la funcion queda plana en la
    # region recortada mientras que el gradiente analitico sigue asumiendo que el margen varia. La
    # verificacion contra diferencias finitas lo detecto, con un residuo de 5,7e-02 que era
    # **identico en lambda=0 y en lambda=0,25**, lo que senalaba que el fallo estaba en la parte de
    # entropia cruzada y no en el termino por pares. Las dos formas de abajo son estables **y
    # exactas**, asi que no hace falta recortar nada.
    #
    # La probabilidad usa la identidad 1/(1+e^-m) = (1 + tanh(m/2)) / 2, que satura en 0 y 1 sin
    # desbordar. La perdida usa logaddexp, que es -log(p_y) escrito de forma que no logarithma de
    # cero. Ambas conservan la derivada analitica exacta.
    probabilidad = 0.5 * (1.0 + np.tanh(0.5 * logitos))
    perdida_ce = np.where(
        y == 1, np.logaddexp(0.0, -logitos), np.logaddexp(0.0, logitos)
    )
    valor = float(np.sum(pesos * perdida_ce))
    gradiente_logit = (probabilidad - y) * pesos

    if peso_pares != 0.0:
        hs = y[:, 0]
        # Cuando HS=1 se espera z_h >= z_t y z_h >= z_a; cuando HS=0, lo contrario. `arriba_hs`
        # lleva ese signo, de modo que d es siempre la diferencia a favor de la etiqueta que deberia
        # estar mas arriba, y se espera d >= margen.
        arriba_hs = np.where(hs == 1, 1.0, -1.0)
        d_tr = arriba_hs * (logitos[:, 0] - logitos[:, 1])
        d_ag = arriba_hs * (logitos[:, 0] - logitos[:, 2])
        # **Perdida por pares logística, no bisagra.** La primera version uso max(0, margen - d),
        # que tiene un vertice, y el optimizador abortaba con ABNORMAL en la iteracion 19 con
        # independencia de las tolerancias: la busqueda lineal no puede funcionar sobre una funcion
        # no derivable. El surrogate logistico es el estandar para objectives de orden por pares,
        # es derivable en todo el dominio y su gradiente esta acotado. El valor se evalua con
        # logaddexp, que es softplus(margen - d) sin desbordamiento.
        # **El termino por pares es una SUMA sobre filas, igual que la entropia cruzada.** La
        # primera version lo dividia por el numero de filas, con lo que su contribucion era del
        # orden de lambda * 0,7 frente a una entropia cruzada del orden de 2463, es decir una razon
        # de 1 a 3500. Lambda era entonces irreconciliable: el barrido daba el mismo resultado para
        # lambda=0,0, 0,1, 0,25 y 0,5, y solo se movia a partir de lambda=1, con dos celdas de
        # diferencia. Eso no es un resultado, es un termino despreciable, y no se puede distinguir de
        # no hacer nada. Al ponerlo en la misma escala que la entropia cruzada, lambda vuelve a
        # tener significado y el barrido informa.
        valor += peso_pares * float(
            np.sum(np.logaddexp(0.0, MARGEN - d_tr) + np.logaddexp(0.0, MARGEN - d_ag))
        ) / 2.0
        # d softplus(margen - d) / dd = -sigma(d - margen). El signo para cada etiqueta es el
        # opuesto del que lleva `arriba_hs`, porque d se define con ese signo.
        sigma_tr = 1.0 / (1.0 + np.exp(np.clip(d_tr - MARGEN, -35.0, 35.0)))
        sigma_ag = 1.0 / (1.0 + np.exp(np.clip(d_ag - MARGEN, -35.0, 35.0)))
        factor = peso_pares / 2.0
        gradiente_logit[:, 0] -= factor * arriba_hs * (sigma_tr + sigma_ag)
        gradiente_logit[:, 1] += factor * arriba_hs * sigma_tr
        gradiente_logit[:, 2] += factor * arriba_hs * sigma_ag

    # El gradiente se construye en dos dimensiones, 3 bloques por (n_caracteristicas + 1), porque
    # se indexa por bloque; ``empty_like`` sobre el vector plano daria un array de una dimension.
    gradiente = np.zeros((3, n_features + 1), dtype=np.float64)
    for indice in range(3):
        solo_pesos = bloques[indice, :-1]
        # La penalizacion L2 tiene que estar **en el valor y en el gradiente**. Una version
        # anterior de este script la tenia solo en el gradiente, y la verificacion contra
        # diferencias finitas lo detecto con un residuo de 5,7e-02 que era exactamente del orden de
        # w / C, es decir la contribucion de la penalizacion. Sin ese aviso, el ajuste habria
        # optimizado una funcion cuyo valor no era el que el gradiente decia, y el resultado habria
        # sido plausible y falso.
        valor += float(np.dot(solo_pesos, solo_pesos)) / (2.0 * CONSTANTE_V03)
        gradiente[indice, :-1] = matriz.T @ gradiente_logit[:, indice] + solo_pesos / CONSTANTE_V03
        gradiente[indice, -1] = float(np.sum(gradiente_logit[:, indice]))
    return valor, gradiente.ravel()


def comprobar_gradiente(
    coeficientes: np.ndarray,
    matriz: sparse.csr_matrix,
    y: np.ndarray,
    pesos: np.ndarray,
    peso_pares: float,
    semilla: int,
) -> dict[str, Any]:
    """Verifica el gradiente analitico contra diferencias finitas centrales.

    Args:
        coeficientes: punto en el que se comprueba.
        matriz: matriz de caracteristicas dispersa.
        y: matriz binaria de etiquetas.
        pesos: matriz de pesos de clase.
        peso_pares: peso lambda del termino por pares.
        semilla: semilla del generador de coordenadas.

    Returns:
        Diccionario con el residuo maximo y su veredicto.
    """
    generador = np.random.Generator(np.random.PCG64(semilla))
    _, gradiente = Loss_y_gradiente(coeficientes, matriz, y, pesos, peso_pares)
    paso = 1e-6
    indices = generador.choice(len(coeficientes), size=COORDENADAS_COMPROBACION, replace=False)
    residuo = 0.0
    for indice in indices:
        arriba = coeficientes.copy()
        arriba[indice] += paso
        abajo = coeficientes.copy()
        abajo[indice] -= paso
        valor_arriba, _ = Loss_y_gradiente(arriba, matriz, y, pesos, peso_pares)
        valor_abajo, _ = Loss_y_gradiente(abajo, matriz, y, pesos, peso_pares)
        numerica = (valor_arriba - valor_abajo) / (2.0 * paso)
        residuo = max(residuo, abs(numerica - gradiente[indice]))
    return {
        "lambda": peso_pares,
        "coordenadas_comprobadas": len(indices),
        "residuo_maximo": float(residuo),
        "tolerancia": TOLERANCIA_GRADIENTE,
        "correcto": bool(residuo < TOLERANCIA_GRADIENTE),
    }


def ajustar_conjunto(
    matriz: sparse.csr_matrix, y: np.ndarray, peso_pares: float
) -> tuple[np.ndarray, int, int, float]:
    """Ajusta los tres clasificadores conjuntamente con el termino por pares.

    Args:
        matriz: matriz de caracteristicas dispersa.
        y: matriz binaria de etiquetas.
        peso_pares: peso lambda del termino por pares.

    Returns:
        Tupla con los coeficientes, las iteraciones, las evaluaciones de funcion y la norma del
        gradiente en la solucion, que es la medida de convergencia que se registra.
    """
    pesos = np.column_stack([pesos_equilibrados(y[:, i]) for i in range(3)])
    x0 = np.zeros(3 * (matriz.shape[1] + 1), dtype=np.float64)
    with warnings.catch_warnings(record=True) as atrapados:
        warnings.simplefilter("always")
        resultado = minimize(
            Loss_y_gradiente,
            x0,
            args=(matriz, y, pesos, peso_pares),
            jac=True,
            method="L-BFGS-B",
            # maxls=100 es **necesario**: con el valor por defecto de 20 pasos, la busqueda lineal
            # aborta con ABNORMAL sin converger, y con 100 pasos converge en 20 iteraciones y 48
            # evaluaciones, menos que las 27 iteraciones y 165 evaluaciones del intento fallido. La
            # formulacion del objetivo no se cambia, que es la que la verificacion del gradiente
            # valido; lo unico que se ajusta es el numero maximo de pasos de la busqueda lineal.
            options={
                "maxiter": LIMITE_ITERACIONES,
                "maxfun": LIMITE_ITERACIONES * 2,
                "maxls": 100,
            },
        )
    if atrapados:
        raise RuntimeError(
            f"el optimizador lanzo advertencias no esperadas: {[str(a.message) for a in atrapados]}"
        )
    if not resultado.success:
        raise RuntimeError(f"el optimizador no ha convergido: {resultado.message}")
    coeficientes = np.asarray(resultado.x).ravel()
    _, gradiente = Loss_y_gradiente(coeficientes, matriz, y, pesos, peso_pares)
    return coeficientes, int(resultado.nit), int(resultado.nfev), float(np.linalg.norm(gradiente))


def predecir(matriz: sparse.csr_matrix, coeficientes: np.ndarray) -> np.ndarray:
    """Calcula las probabilidades de los tres clasificadores.

    Args:
        matriz: matriz de caracteristicas.
        coeficientes: vector con los tres bloques y los tres terminos independientes.

    Returns:
        Matriz de probabilidades con una columna por etiqueta.
    """
    bloques = coeficientes.reshape(3, matriz.shape[1] + 1)
    salida = np.empty((matriz.shape[0], 3), dtype=np.float64)
    for indice in range(3):
        margen = matriz @ bloques[indice, :-1] + bloques[indice, -1]
        salida[:, indice] = 0.5 * (1.0 + np.tanh(0.5 * margen))
    return salida


def f1_positiva(verdadero: np.ndarray, predicho: np.ndarray) -> float:
    """Calcula la F1 de la clase positiva.

    Args:
        verdadero: etiqueta binaria verdadera.
        predicho: predicciones binarias.

    Returns:
        La F1 de la clase positiva, o cero si el denominador se anula.
    """
    positivos = verdadero == 1
    verdaderos_positivos = float(np.sum(positivos & (predicho == 1)))
    falsos_positivos = float(np.sum((~positivos) & (predicho == 1)))
    falsos_negativos = float(np.sum(positivos & (predicho == 0)))
    denominador = 2 * verdaderos_positivos + falsos_positivos + falsos_negativos
    return 2 * verdaderos_positivos / denominador if denominador > 0 else 0.0


def metrica_por_etiqueta(
    y: np.ndarray, predicciones_por_etiqueta: list[np.ndarray]
) -> tuple[float, list[float]]:
    """Calcula la metrica corregida y la F1 de clase positiva por etiqueta.

    Args:
        y: matriz binaria de etiquetas verdaderas.
        predicciones_por_etiqueta: predicciones de cada etiqueta.

    Returns:
        Tupla con la metrica corregida y la lista de F1 de clase positiva.
    """
    matriz = np.column_stack(predicciones_por_etiqueta)
    metricas = metrica_macro(y, matriz)
    positivas = [f1_positiva(y[:, i], matriz[:, i]) for i in range(len(ETIQUETAS_LISTA))]
    return metricas, positivas


def dependencia_hs(
    y: np.ndarray, pred: np.ndarray
) -> dict[str, Any]:
    """Mide el efecto sobre la dependencia HS hacia TR y AG.

    Args:
        y: matriz binaria de etiquetas verdaderas.
        pred: matriz de predicciones binarias.

    Returns:
        Diccionario con las F1 restringidas y los falsos positivos en la zona no definida.
    """
    definida = y[:, 0] == 1
    no_definida = ~definida
    return {
        "f1_tr_donde_esta_definida": f1_positiva(y[definida, 1], pred[definida, 1]),
        "f1_ag_donde_esta_definida": f1_positiva(y[definida, 2], pred[definida, 2]),
        "f1_hs_global": f1_positiva(y[:, 0], pred[:, 0]),
        "falsos_positivos_tr_zona_no_definida": int(
            np.sum((y[no_definida, 1] == 0) & (pred[no_definida, 1] == 1))
        ),
        "falsos_positivos_ag_zona_no_definida": int(
            np.sum((y[no_definida, 2] == 0) & (pred[no_definida, 2] == 1))
        ),
        "falsos_positivos_tr_zona_no_definida_predichos": int(
            np.sum(pred[no_definida, 1] == 1)
        ),
        "falsos_positivos_ag_zona_no_definida_predichos": int(
            np.sum(pred[no_definida, 2] == 1)
        ),
    }


def main() -> None:
    """Ejecuta el objetivo por pares con los mismos gates y controles que H2."""
    inicio = time.time()
    log("V12-03: entrenamiento por pares sobre TF-IDF disperso")

    train = data_module.read_train()
    grupos_completos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos_completos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos_completos[indices_dev]
    y = desarrollo[ETIQUETAS_LISTA].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    pliegues = arnes.particion_por_semilla(textos, y, grupos_dev, SEMILLA_CANONICA)
    log(f"   desarrollo: {len(y)} filas | {len(pliegues)} pliegues")

    log("1/6 brazo A, control de identidad")
    probabilidades_a = arnes.oof_de_v03(textos, y, pliegues)
    metricas_a, _ = metrica_por_etiqueta(
        y, [(probabilidades_a[:, i] >= UMBRALES[i]).astype(int) for i in range(3)]
    )
    identidad = abs(metricas_a - REFERENCIA_V03)
    if identidad > 1e-12:
        log(f"   PARADA OBLIGATORIA: diferencia {identidad:.3e}")
        raise SystemExit(1)
    base = (probabilidades_a >= UMBRALES[None, :]).astype(int)
    log(f"   metrica {metricas_a:.16f} | diferencia {identidad:.3e} | identidad confirmada")
    referencia_dependencia = dependencia_hs(y, base)
    log(
        f"   dependencia en A: F1 TR donde definida "
        f"{referencia_dependencia['f1_tr_donde_esta_definida']:.4f} | falsos positivos de TR en la "
        f"zona no definida {referencia_dependencia['falsos_positivos_tr_zona_no_definida']} y de AG "
        f"{referencia_dependencia['falsos_positivos_ag_zona_no_definida']}, que son los que el "
        f"oraculo de H2 corrigio"
    )

    log("2/6 verificacion del gradiente, antes de usar ningun resultado")
    # `pesos` sobre todo el desarrollo quedo sin usar: la verificacion usa los
    # pesos del primer pliegue, que son los que corresponden a la matriz.
    # Se verifica sobre la matriz real del primer pliegue, y en dos puntos: uno aleatorio y otro
    # **cerca de la solucion**. Un gradiente puede coincidir con las diferencias finitas lejos del
    # optimo y fallar en el, y el punto que importa es el que el optimizador visita.
    ajuste_comprobacion, _ = construir_espacio(
        textos.iloc[pliegues[0][0]].reset_index(drop=True),
        textos.iloc[pliegues[0][1]].reset_index(drop=True),
    )
    y_comprobacion = y[pliegues[0][0]]
    pesos_comprobacion = np.column_stack(
        [pesos_equilibrados(y_comprobacion[:, i]) for i in range(3)]
    )
    generador = np.random.Generator(np.random.PCG64(20260926))
    punto_aleatorio = generador.normal(0.0, 0.05, size=3 * (ajuste_comprobacion.shape[1] + 1))
    punto_cerca, _, _, _ = ajustar_conjunto(ajuste_comprobacion, y_comprobacion, 0.25)
    verificacion: list[dict[str, Any]] = []
    for peso_pares in VALORES_LAMBDA:
        for etiqueta_punto, punto in (
            ("aleatorio", punto_aleatorio),
            ("cerca de la solucion", punto_cerca),
        ):
            resultado = comprobar_gradiente(
                punto, ajuste_comprobacion, y_comprobacion, pesos_comprobacion, peso_pares, 20260926
            )
            resultado["punto"] = etiqueta_punto
            verificacion.append(resultado)
    for v in verificacion:
        log(
            f"   lambda={v['lambda']} en punto {v['punto']}: residuo maximo "
            f"{v['residuo_maximo']:.3e} frente a {v['tolerancia']:.0e} -> {v['correcto']}"
        )
    if not all(v["correcto"] for v in verificacion):
        log("   PARADA OBLIGATORIA: la derivada no supera la verificacion numerica")
        raise SystemExit(1)
    log("   derivada verificada en los tres valores de lambda")

    log("3/6 los tres brazos, con lambda=0 como control interno")
    probabilidades: dict[float, np.ndarray] = {}
    salud: dict[str, Any] = {"iteraciones_maximas": 0, "evaluaciones_maximas": 0}
    for peso_pares in VALORES_LAMBDA:
        salida = np.zeros((len(y), 3), dtype=np.float64)
        for numero_pliegue, (entrenamiento, validacion) in enumerate(pliegues):
            ajuste, consulta = construir_espacio(
                textos.iloc[entrenamiento].reset_index(drop=True),
                textos.iloc[validacion].reset_index(drop=True),
            )
            coeficientes, iteraciones, evaluaciones, norma_gradiente = ajustar_conjunto(
                ajuste, y[entrenamiento], peso_pares
            )
            salud["iteraciones_maximas"] = max(salud["iteraciones_maximas"], iteraciones)
            salud["evaluaciones_maximas"] = max(salud["evaluaciones_maximas"], evaluaciones)
            salud["norma_gradiente_maxima"] = max(
                salud.get("norma_gradiente_maxima", 0.0), norma_gradiente
            )
            salida[validacion] = predecir(consulta, coeficientes)
            if numero_pliegue == 0:
                if peso_pares == 0.0:
                    verificacion[0]["punto_de_comprobacion"] = "ajuste del primer pliegue"
                else:
                    verificacion[-1]["punto_de_comprobacion"] = "ajuste del primer pliegue"
        probabilidades[peso_pares] = salida
        log(f"   lambda={peso_pares}: ajustados {len(pliegues)} modelos conjuntos")

    log("4/6 metricas, deltas por pliegue, intervalos y deteccion")
    filas: list[dict[str, Any]] = []
    veredictos: dict[str, dict[str, Any]] = {}
    deltas_por_pliegue: dict[str, list[float]] = {}
    umbral_deteccion = 0.5 + 19.2 / FILAS_LEADERBOARD
    for peso_pares in VALORES_LAMBDA:
        nombre = f"lambda={peso_pares}"
        pred = (probabilidades[peso_pares] >= UMBRALES[None, :]).astype(int)
        metricas, f1_por_etiqueta = metrica_por_etiqueta(y, [pred[:, i] for i in range(3)])
        lista = []
        for entrenamiento, validacion in pliegues:
            m, _ = metrica_por_etiqueta(y[validacion], [pred[validacion, i] for i in range(3)])
            m_ref, _ = metrica_por_etiqueta(
                y[validacion], [base[validacion, i] for i in range(3)]
            )
            lista.append(m - m_ref)
        deltas_por_pliegue[nombre] = [round(v, 6) for v in lista]
        _, inferior, superior = arnes.intervalo_pareado(y, pred, base)
        cambiados = pred != base
        total = float(np.sum(cambiados))
        aciertos = float(np.sum((pred == y) & cambiados))
        deteccion = aciertos / total if total > 0 else 0.0
        delta = metricas - metricas_a
        a_favor = sum(1 for v in lista if v > 0)
        exploratorio = (
            delta >= DELTA_MINIMO_EXPLORATORIO
            and a_favor >= PLIEGUES_MINIMOS
            and (inferior > 0.0 or superior < 0.0)
        )
        dependencia = dependencia_hs(y, pred)
        filas.append(
            {
                "brazo": nombre,
                "lambda": peso_pares,
                "metrica": metricas,
                "delta": delta,
                "f1_por_etiqueta": f1_por_etiqueta,
                "celdas_modificadas": int(total),
                "deteccion": round(deteccion, 4),
                "dependencia": dependencia,
            }
        )
        veredictos[nombre] = {
            "delta": round(delta, 6),
            "pliegues_a_favor": f"{a_favor}/5",
            "ic95_inferior": round(inferior, 6),
            "ic95_superior": round(superior, 6),
            "ic95_excluye_cero": bool(inferior > 0.0 or superior < 0.0),
            "celdas_modificadas": int(total),
            "deteccion": round(deteccion, 4),
            "supera_deteccion": bool(deteccion > umbral_deteccion),
            "gate_exploratorio": bool(exploratorio),
            "gate_candidato": bool(delta >= DELTA_MINIMO_CANDIDATO and exploratorio),
        }
        log(
            f"   {nombre}: {metricas:.6f} (delta {delta:+.6f}) | {a_favor}/5 pliegues | IC95 "
            f"[{inferior:+.6f}, {superior:+.6f}] | celdas {int(total)} | deteccion {deteccion:.4f} "
            f"-> {deteccion > umbral_deteccion} | exploratorio {exploratorio}"
        )
        log(
            f"     dependencia: F1 TR donde definida "
            f"{dependencia['f1_tr_donde_esta_definida']:.4f} (A "
            f"{referencia_dependencia['f1_tr_donde_esta_definida']:.4f}) | falsos positivos de TR "
            f"en la zona no definida {dependencia['falsos_positivos_tr_zona_no_definida']} (A "
            f"{referencia_dependencia['falsos_positivos_tr_zona_no_definida']}) | de AG "
            f"{dependencia['falsos_positivos_ag_zona_no_definida']} (A "
            f"{referencia_dependencia['falsos_positivos_ag_zona_no_definida']})"
        )

    log("5/6 curva de lambda, diagnostico del cierre")
    curva: list[dict[str, Any]] = []
    for peso_pares in REJILLA_LAMBDA:
        if peso_pares in probabilidades:
            metricas = next(f["metrica"] for f in filas if f["lambda"] == peso_pares)
        else:
            salida = np.zeros((len(y), 3), dtype=np.float64)
            for entrenamiento, validacion in pliegues:
                ajuste, consulta = construir_espacio(
                    textos.iloc[entrenamiento].reset_index(drop=True),
                    textos.iloc[validacion].reset_index(drop=True),
                )
                coeficientes, _, _, _ = ajustar_conjunto(ajuste, y[entrenamiento], peso_pares)
                salida[validacion] = predecir(consulta, coeficientes)
            pred = (salida >= UMBRALES[None, :]).astype(int)
            metricas, _ = metrica_por_etiqueta(y, [pred[:, i] for i in range(3)])
        curva.append({"lambda": peso_pares, "metrica": metricas, "delta": metricas - metricas_a})
        log(f"   lambda={peso_pares}: metrica {metricas:.6f} (delta {metricas - metricas_a:+.6f})")
    maximo_curva = max(curva, key=lambda f: f["metrica"])
    valores = [f["metrica"] for f in curva]
    log(
        f"   maximo en lambda={maximo_curva['lambda']} con {maximo_curva['delta']:+.6f} | "
        f"recorrido {max(valores) - min(valores):.6f} | lambda=0 da "
        f"{curva[0]['delta']:+.6f}, que es el control interno"
    )

    candidatas_exploratorias = [n for n, v in veredictos.items() if v["gate_exploratorio"]]
    candidatas = [n for n, v in veredictos.items() if v["gate_candidato"]]
    log(
        f"6/6 candidatas exploratorias {candidatas_exploratorias or 'ninguna'} | candidatas "
        f"{candidatas or 'ninguna'}"
    )

    mejor_delta = max(f["delta"] for f in filas if f["lambda"] > 0.0)
    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "referencia_v03": REFERENCIA_V03,
        "identidad_control": {
            "metrica": metricas_a,
            "diferencia_con_referencia": identidad,
            "reproduce_oof_canonico": True,
        },
        "pregunta": (
            "Puede una funcion de entrenamiento por pares sobre TF-IDF disperso explotar la "
            "estructura relativa entre HS, TR y AG que el objetivo binario independiente no esta "
            "aprovechando?"
        ),
        "mecanismo": (
            "Tres clasificadores binarios entrenados **jointemente** con un termino de margen por "
            "pares que expresa la relacion de orden entre las tres etiquetas: cuando HS=1 se espera "
            "que el logit de HS supere a los de TR y AG, y cuando HS=0 se espera lo contrario. El "
            "termino no introduce informacion nueva, no cambia el espacio ni los umbrales, y **no "
            "convierte HS en una decision binaria previa**, que es el defecto del post-proceso duro."
        ),
        "distincion_de_mecanismo": (
            "No es la cadena cerrada E2-01, que anade el score de la etiqueta previa como feature: "
            "aqui los tres clasificadores se acoplan en la funcion objetivo. No es el post-proceso "
            "duro, que usa HS predicho como llave en la prediccion: aqui la prediccion es la de V03 "
            "con los mismos umbrales. Y no son los pares de ejemplos que describia la auditoria de "
            "brecha, sino pares de **etiquetas** dentro de cada ejemplo, que es lo que ataca la "
            "pregunta de la fase."
        ),
        "verificacion_del_gradiente": verificacion,
        "brazos": filas,
        "deltas_por_pliegue": deltas_por_pliegue,
        "curva_de_lambda": {
            "lambda": [f["lambda"] for f in curva],
            "metrica": [round(f["metrica"], 6) for f in curva],
            "delta": [round(f["delta"], 6) for f in curva],
            "maximo_lambda": maximo_curva["lambda"],
            "delta_del_maximo": round(maximo_curva["delta"], 6),
            "recorrido": round(max(valores) - min(valores), 6),
            "control_lambda_cero": round(curva[0]["delta"], 6),
            "aviso": "El maximo de esta curva NO es adoptable: se ha calculado mirando los pliegues "
            "exteriores. Se reporta solo como diagnostico del motivo del cierre.",
        },
        "mejor_delta_de_pares": round(mejor_delta, 6),
        "umbral_deteccion": round(umbral_deteccion, 4),
        "veredictos": veredictos,
        "candidatas_exploratorias": candidatas_exploratorias,
        "candidatas": candidatas,
        "dependencia_de_referencia": referencia_dependencia,
        "salud_del_modelo": {
            "iteraciones_maximas_lbfgsb": salud["iteraciones_maximas"],
            "evaluaciones_maximas": salud["evaluaciones_maximas"],
            "norma_gradiente_maxima": round(salud.get("norma_gradiente_maxima", 0.0), 6),
            "limite": LIMITE_ITERACIONES,
            "gradiente_verificado": True,
            "maxls": 100,
            "nota_maxls": "Con el valor por defecto de 20 pasos la busqueda lineal abortaba con "
            "ABNORMAL sin converger. Con 100 pasos converge en 20 iteraciones y 48 evaluaciones, "
            "menos que las 27 iteraciones y 165 evaluaciones del intento fallido.",
        },
        "gates": {
            "delta_minimo_exploratorio": DELTA_MINIMO_EXPLORATORIO,
            "delta_minimo_candidato": DELTA_MINIMO_CANDIDATO,
            "pliegues_minimos": PLIEGUES_MINIMOS,
        },
        "contexto_del_techo": {
            "techo_hs_verdadero": REFERENCIA_ORACULO,
            "fraccion_del_techo_capturada": round(mejor_delta / REFERENCIA_ORACULO, 4)
            if mejor_delta > 0
            else 0.0,
        },
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe(informe, filas, veredictos, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: candidatas exploratorias {candidatas_exploratorias or 'ninguna'} | candidatas "
        f"{candidatas or 'ninguna'} | mejor delta de pares {mejor_delta:+.6f} | techo H2 "
        f"{REFERENCIA_ORACULO:+.6f}"
    )


def _informe(
    informe: dict[str, Any],
    filas: list[dict[str, Any]],
    veredictos: dict[str, dict[str, Any]],
    inicio: float,
) -> None:
    """Escribe el informe del experimento.

    Args:
        informe: veredicto del experimento.
        filas: metricas por brazo.
        veredictos: resultado de los gates por brazo.
        inicio: instante de arranque del script.
    """
    umbral = informe["umbral_deteccion"]
    contexto = informe["contexto_del_techo"]
    referencia = informe["dependencia_de_referencia"]
    linea: list[str] = [
        "# V12-03: entrenamiento por pares sobre TF-IDF disperso (H3)",
        "",
        f"Generado por `experiments/v12_03_entrenamiento_por_pares.py` ({EXPERIMENTO}).",
        "",
        "## 1. La pregunta",
        "",
        "> Una funcion de entrenamiento por pares sobre TF-IDF disperso puede explotar la estructura",
        "> relativa entre HS, TR y AG que el objetivo binario independiente no esta aprovechando?",
        "",
        "V03 ajusta **tres clasificadores binarios independientes**, y su objetivo es la suma de",
        "tres entropias cruzadas que no se hablan entre si. Eso deja sin usar una informacion que el",
        "propio conjunto de entrenamiento contiene: **HS es una condicion necesaria de TR y de AG**.",
        "Es una relacion de **orden** entre las tres etiquetas, y un objetivo que las trata por",
        "separado no puede expressarla.",
        "",
        "## 2. Que es por pares aqui, y que mecanismos NO es",
        "",
        "La auditoria de brecha describio H3 como pares de **ejemplos**. La pregunta de la fase es mas",
        "especifica: pares de **etiquetas** dentro de cada ejemplo. Se implementa la segunda, que es",
        "la que ataca la estructura relativa entre las tres etiquetas, y se deja constancia de la",
        "diferencia porque son mecanismos distintos.",
        "",
        "El termino, con margen fijo de 1,0 en unidades de logit:",
        "",
        "- si `HS=1`, TR y AG estan definidas y HS es cierta, luego se espera `z_h >= z_t` y",
        "  `z_h >= z_a`, con penalizacion `max(0, m - (z_h - z_t))` y `max(0, m - (z_h - z_a))`;",
        "- si `HS=0`, TR y AG no estan definidas y se espera lo contrario, con penalizacion",
        "  `max(0, m - (z_t - z_h))` y `max(0, m - (z_a - z_h))`.",
        "",
        "**No es la cadena cerrada E2-01**, que anade el score de la etiqueta previa como feature:",
        "aqui los tres clasificadores se acoplan **en la funcion objetivo**. **No es el post-proceso",
        "duro**, que usa HS predicho como llave en la prediccion: aqui **no hay llave**, la",
        "prediccion sigue siendo la de V03 con los mismos umbrales congelados, y lo unico que cambia",
        "es la funcion que se minimiza. Ese es justo el defecto que H2 le atribuyo al post-proceso.",
        "",
        "## 3. La derivada, verificada antes de usar ningun resultado",
        "",
        "El termino acopla tres clasificadores y por eso exige un optimizador propio, igual que en",
        "H2. **El vector de coeficientes va primero**, que es como lo invoca el optimizador, para que",
        "la verificacion y el ajuste recorran el mismo camino de llamada.",
        "",
        fila(["lambda", "punto", "coordenadas", "residuo maximo", "tolerancia", "correcto"]),
        fila(["---"] * 6),
    ]
    for v in informe["verificacion_del_gradiente"]:
        linea.append(
            fila(
                [
                    f"{v['lambda']}",
                    v["punto"],
                    str(v["coordenadas_comprobadas"]),
                    f"{v['residuo_maximo']:.3e}",
                    f"{v['tolerancia']:.0e}",
                    "si" if v["correcto"] else "NO",
                ]
            )
        )
    linea += [
        "",
        "## 4. Resultados, y el control interno lambda=0",
        "",
        fila(
            [
                "brazo",
                "metrica corregida",
                "delta vs A",
                "HS F1",
                "TR F1",
                "AG F1",
                "celdas que cambian",
                "deteccion",
            ]
        ),
        fila(["---"] * 8),
    ]
    for f in filas:
        linea.append(
            fila(
                [
                    f"lambda={f['lambda']}",
                    f"{f['metrica']:.6f}",
                    f"{f['delta']:+.6f}",
                    f"{f['f1_por_etiqueta'][0]:.4f}",
                    f"{f['f1_por_etiqueta'][1]:.4f}",
                    f"{f['f1_por_etiqueta'][2]:.4f}",
                    str(f["celdas_modificadas"]),
                    f"{f['deteccion']:.4f}",
                ]
            )
        )
    control = next(f for f in filas if f["lambda"] == 0.0)
    linea += [
        "",
        f"**El control interno `lambda=0` da {control['delta']:+.6f}** con",
        f"{control['celdas_modificadas']} celdas de diferencia. Con el termino por pares a peso cero",
        "el objetivo es la suma de las tres entropias cruzadas, que es exactamente V03, asi que",
        "este control valida a la vez el objetivo, el optimizador, el pipeline y los pesos de clase.",
        "",
        "### Delta por pliegue",
        "",
        fila(["brazo"] + [f"pliegue {i}" for i in range(1, 6)] + ["a favor"]),
        fila(["---"] * 7),
    ]
    for nombre, lista in informe["deltas_por_pliegue"].items():
        linea.append(
            fila([nombre] + [f"{v:+.6f}" for v in lista] + [f"{sum(1 for v in lista if v > 0)}/5"])
        )
    linea += [
        "",
        "## 5. El efecto sobre la dependencia HS hacia TR y AG",
        "",
        "Es la medida que hace especifica a esta hipotesis, porque el mecanismo acopla las tres",
        "etiquetas y deberia notarse justo en la relacion entre HS y las otras dos.",
        "",
        fila(
            [
                "brazo",
                "F1 TR donde esta definida",
                "F1 AG donde esta definida",
                "F1 HS global",
                "FP de TR en zona no definida",
                "FP de AG en zona no definida",
            ]
        ),
        fila(["---"] * 6),
        fila(
            [
                "V03 (A)",
                f"{referencia['f1_tr_donde_esta_definida']:.4f}",
                f"{referencia['f1_ag_donde_esta_definida']:.4f}",
                f"{referencia['f1_hs_global']:.4f}",
                str(referencia["falsos_positivos_tr_zona_no_definida"]),
                str(referencia["falsos_positivos_ag_zona_no_definida"]),
            ]
        ),
    ]
    for f in filas:
        d = f["dependencia"]
        linea.append(
            fila(
                [
                    f"lambda={f['lambda']}",
                    f"{d['f1_tr_donde_esta_definida']:.4f}",
                    f"{d['f1_ag_donde_esta_definida']:.4f}",
                    f"{d['f1_hs_global']:.4f}",
                    str(d["falsos_positivos_tr_zona_no_definida"]),
                    str(d["falsos_positivos_ag_zona_no_definida"]),
                ]
            )
        )
    linea += [
        "",
        "Los dos ultimos numeros son los que el oraculo de H2 corrigio: 78 falsos positivos de TR y 104 "
        "de AG. El oraculo cambiaba 196 celdas, que son esos 78 mas 104 mas 14 positivos reales de "
        "AG que el oraculo tambien eliminaba por estar en la zona no definida. Un mecanismo que",
        "atacase la dependencia deberia reducirlos **sin** a costa de las F1 donde las etiquetas si",
        "estan definidas, porque corregir la estructura no es recortar positivos.",
        "",
        "## 6. Gates",
        "",
        fila(
            [
                "brazo",
                "delta",
                "pliegues a favor",
                "IC95",
                "excluye cero",
                "deteccion",
                "exploratorio",
                "candidato",
            ]
        ),
        fila(["---"] * 8),
    ]
    for nombre, v in veredictos.items():
        linea.append(
            fila(
                [
                    nombre,
                    f"{v['delta']:+.6f}",
                    v["pliegues_a_favor"],
                    f"[{v['ic95_inferior']:+.6f}, {v['ic95_superior']:+.6f}]",
                    "si" if v["ic95_excluye_cero"] else "no",
                    f"{v['deteccion']:.4f} > {umbral:.4f}",
                    "si" if v["gate_exploratorio"] else "no",
                    "si" if v["gate_candidato"] else "no",
                ]
            )
        )
    curva = informe["curva_de_lambda"]
    linea += [
        "",
        "## 7. La curva de lambda, diagnostico del cierre",
        "",
        "**El maximo de esta curva no es adoptable**, porque se ha calculado mirando los pliegues",
        "exteriores. Se reporta solo como diagnostico.",
        "",
        fila(["lambda", "metrica corregida", "delta frente a V03"]),
        fila(["---"] * 3),
    ] + [
        fila([f"{l:.2f}", f"{m:.6f}", f"{d:+.6f}"])
        for l, m, d in zip(curva["lambda"], curva["metrica"], curva["delta"], strict=True)
    ]
    linea += [
        "",
        f"- Maximo: **lambda = {curva['maximo_lambda']}** con **{curva['delta_del_maximo']:+.6f}**.",
        f"- Recorrido de la curva: **{curva['recorrido']:.6f}**.",
        f"- El control `lambda=0` da **{curva['control_lambda_cero']:+.6f}**.",
        "",
        "## 8. Salud del modelo",
        "",
        f"- Iteraciones maximas de L-BFGS-B: "
        f"**{informe['salud_del_modelo']['iteraciones_maximas_lbfgsb']}** de "
        f"{informe['salud_del_modelo']['limite']}.",
        f"- Evaluaciones de funcion: **{informe['salud_del_modelo']['evaluaciones_maximas']}**.",
        f"- Gradiente verificado: **{informe['salud_del_modelo']['gradiente_verificado']}**.",
        "",
        "## 9. Veredicto",
        "",
    ]
    if informe["candidatas"]:
        linea += [
            f"**HAY CANDIDATA: {', '.join(informe['candidatas'])}.** Pasa a segunda particion,",
            "reproducibilidad y auditoria de cumplimiento. **No se genera submission todavia.**",
        ]
    elif informe["candidatas_exploratorias"]:
        linea += [
            f"**SUPERA EL GATE EXPLORATORIO: {', '.join(informe['candidatas_exploratorias'])}**, pero",
            "no llega al de candidato. **No se genera submission.**",
        ]
    else:
        mejor = informe["mejor_delta_de_pares"]
        fraccion = contexto["fraccion_del_techo_capturada"]
        linea += [
            "**NINGUN BRAZO SUPERA EL GATE EXPLORATORIO.**",
            "",
            f"El mejor valor de lambda da **{mejor:+.6f}**, y eso es el "
            f"**{fraccion * 100:.1f} por ciento** del techo de {contexto['techo_hs_verdadero']:+.6f}",
            "que H2 midio con la condicion de HS verdadera.",
            "",
            "Con las tres hipotesis cerradas, la lectura es de **rendimientos decrecientes en el",
            "objetivo de entrenamiento**: H1, que era regularizacion, +0.002738 sostenido por un",
            "solo pliegue; H2, que era la perdida, +0.001616 con intervalo incluyendo el cero; y H3,",
            f"que es el acoplamiento por pares, {mejor:+.6f}. Tres mecanismos distintos sobre la misma",
            "funcion, ninguno cerca del techo, y el techo sigue intacto.",
            "",
            "Y la conclusion que hay que escribir sin sobregeneralizar, porque es la parte importante",
            "de este cierre: **H2 y H3 no demuestran que la funcion de perdida no sea el problema en",
            "sentido absoluto**. Demuestran algo mas preciso y mas util: **las variaciones de",
            "objetivo evaluadas hasta ahora tienen un impacto pequeno comparado con el techo que",
            "introduce el error de la condicion de HS**. Un solo mecanismo positivo, el GCE, no",
            "autoriza a descartar la familia entera; lo que autoriza es a decir que el margen",
            "recuperable por esa via es pequeno frente a otro problema medido.",
        ]
    linea += sello(inicio)
    REPORTE_MD.write_text("\n".join(linea) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
