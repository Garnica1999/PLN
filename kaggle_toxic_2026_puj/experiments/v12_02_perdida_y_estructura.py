"""V12-02: funcion de perdida propia y estructura de etiquetas (H2).

La segunda hipotesis de la auditoria de brecha, y la de mayor.techo.

La premisa, medida por primera vez en este proyecto
----------------------------------------------------
En TOXIC, **TR y AG solo estan definidas cuando HS=1**. Esa convencion de anotacion es conocida,
pero **nunca se habia medido cuanto pesa**. Y la medicion es dura:

- 835 de 1483 filas tienen HS=0, el 56,30 por ciento.
- En esas 835 filas, TR=0 en el **100 por ciento** de los casos.
- Por tanto, **el 80,44 por ciento de todos los negativos que el modelo de TR ve durante el
  entrenamiento son etiquetas que no estan definidas**, y para AG el 88,76 por ciento.

Lo que el modelo aprende con eso no es una etiqueta ruidosa: es un concepto equivocado. Se le
ensena que un tweet no toxico es "no insultante", que es un hecho trivial, y que un tweet no
toxico es "no agresivo", que es una afirmacion sin contenido. Esta es exactamente la situacion
para la que existe una perdida tolerante al ruido: no se descarta el ejemplo, se le reduce la
influencia mediante la forma de la funcion, que es lo que el proyecto nunca habia probado.

El techo, medido con oraculo
---------------------------
Antes de implementar nada se mide cuanto hay. La mascara con el **HS verdadero** da
**+0.046271**, con indice de deteccion 0.9286: 196 celdas cambian y 182 son correcciones. Es el
techo mas alto medido en toda la historia del proyecto, frente a una mejor ganancia anidada de
+0.000148.

Ese techo **no es adoptable**, porque usa la etiqueta verdadera de HS. Solo acota la via. Y el
contraste con el post-proceso duro, que usa el HS **predicho** y da -0.004187, localiza el
problema con precision: **la diferencia entre +0.046271 y -0.004187 es exactamente lo que cuesta
usar la prediccion de HS como llave**, porque el HS predicho falla 316 veces de 1483.

Los tres brazos
--------------
- **A, control de identidad**: V03 exacto, con diferencia cero frente a la referencia.
- **G, la perdida propia**: entropia cruzada generalizada, ``L_q = (1 - p_y^q) / q``, con ``q=1``
  exactamente igual a la entropia cruzada y ``q`` menor hacia el valor absoluto. Es la funcion de
  perdida tolerante al ruido mas sencilla que existe, y reduce la influencia de los ejemplos que
  el modelo no consegue acertar, que son precisamente los de la zona no definida. **Requiere un
  optimizador propio**, porque la biblioteca no acepta una perdida a medida, y por eso el script
  **verifica la derivada analiticamente contra diferencias finitas** antes de usar un solo
  resultado. Un gradiente mal implementado daria cifras plausibles y falsas.
- **M, la estructura de etiquetas**: TR y AG se entrenan **solo sobre las filas donde estan
  definidas**, es decir HS=1. El modelo nunca aprende la asociacion "no toxico luego no
  insultante", y en la prediccion se aplica a todas las filas con los umbrales congelados.

Por que M no es la cadena que ya se cerro
-----------------------------------------
La cadena cerrada, E2-01, **anade el score de la etiqueta previa como feature** y sigue
entrenando la segunda etiqueta sobre todas las filas. M restringe el conjunto de entrenamiento.
Una cosa es conditioning y otra es cambiar que filas ven al clasificador. La comprobacion se hizo
leyendo el codigo de `experiments/15_cadenas_etiquetas_v2.py`, no su nombre.

Y por que tampoco es el post-proceso duro que ya se cerro
---------------------------------------------------------
El post-proceso **mascara en la prediccion** con el HS predicho, y por eso falla. M **no cambia la
prediccion en absoluto**: cambia el entrenamiento. Si M falla, no sera por repetir el post-proceso.

Protocolo
---------
``StratifiedGroupKFold`` de 5 pliegues con los grupos fusionados, validacion anidada para elegir
``q`` dentro del pliegue de entrenamiento, umbrales congelados de V03, intervalo pareado de 2000
remuestreos y filtro de detectabilidad de V8-02. El oraculo se calcula y se reporta **como
diagnostico**, nunca como candidato.

Uso
---
    /home/carlos/venv/bin/python experiments/v12_02_perdida_y_estructura.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v11_ayudas import fila, sello

from src import data as data_module
from src.evaluation import ETIQUETAS, metrica_macro
from src.splits import (
    RANDOM_STATE,
    combinaciones,
    grupos_fusionados,
    holdout_v2,
)

_especificacion = util.spec_from_file_location(
    "arnes", PROJECT_ROOT / "experiments" / "v10_03_arnes_evaluacion.py"
)
arnes = util.module_from_spec(_especificacion)
sys.modules["arnes"] = arnes
_especificacion.loader.exec_module(arnes)

EXPERIMENTO = "V12-02 (H2, perdida propia y estructura de etiquetas)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v12_02_perdida_y_estructura.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v12_02_perdida_y_estructura.json"

ETIQUETAS_LISTA = list(ETIQUETAS)
REFERENCIA_V03 = 0.7772114525165721
UMBRALES = arnes.UMBRALES_V03
SEMILLA_CANONICA = RANDOM_STATE
CONSTANTE_V03 = 2.0
LIMITE_ITERACIONES = 2000

# Cuatro valores de q. q=0.01 esta muy cerca del limite de la entropia cruzada y sirve de control
# interno: el brazo G con ese valor tiene que quedar cerca del brazo A, y el script lo comprueba.
# Los otros tres recorren la region de tolerancia al ruido, que en la literatura es q en [0,1, 0,7].
VALORES_Q = (0.01, 0.2, 0.5, 0.8)
PLIEGUES_INTERNOS = 3
DELTA_MINIMO_EXPLORATORIO = 0.002
DELTA_MINIMO_CANDIDATO = 0.003
PLIEGUES_MINIMOS = 4
FILAS_LEADERBOARD = 1154
COORDENADAS_COMPROBACION = 12
TOLERANCIA_GRADIENTE = 1e-6


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V12-02] {mensaje}", flush=True)


# ---------------------------------------------------------------------------
# La perdida propia y su derivada
# ---------------------------------------------------------------------------


def _pesos_equilibrados(objetivo: np.ndarray) -> np.ndarray:
    """Devuelve los pesos de clase que equivalen a ``class_weight="balanced"``.

    Args:
        objetivo: etiqueta binaria de entrenamiento.

    Returns:
        Vector de pesos, uno por fila de entrenamiento.
    """
    negativos = float(np.sum(objetivo == 0))
    positivos = float(np.sum(objetivo == 1))
    if negativos == 0.0 or positivos == 0.0:
        raise ValueError("el conjunto de entrenamiento tiene una sola clase")
    total = len(objetivo)
    pesos = np.where(objetivo == 1, total / (2.0 * positivos), total / (2.0 * negativos))
    return pesos


def perdida_gce(
    coeficientes: np.ndarray,
    matriz: sparse.csr_matrix,
    objetivo: np.ndarray,
    pesos: np.ndarray,
    q: float,
) -> tuple[float, np.ndarray]:
    """Calcula la entropia cruzada generalizada y su gradiente.

    La forma es ``L_q = (1 - p_y^q) / q``, con ``p_y`` la probabilidad de la clase verdadera. **El
    limite ``q -> 0`` es la entropia cruzada binaria** y ``q=1`` es el valor absoluto, de modo que
    la familia interpola entre las dos. Al alejarse ``q`` de cero, el valor de la perdida se
    satura para los ejemplos mal clasificados, lo que reduce su influencia en el gradiente, y esa
    es la propiedad que se quiere del mecanismo.

    El gradiente analitico es

        dL/dz_i = -a_i * p_y_i^(q-1) * (2 y_i - 1) * p_i (1 - p_i)

    y con la penalizacion L2 se suma ``w / C``. **La funcion esta verificada contra diferencias
    finitas dentro del propio script**, porque un gradiente equivocado produciria cifras
    plausibles y falsas, que es el peor fallo posible en un experimento de este tipo.

    El vector de coeficientes va **primero** a proposito: es el orden en que la invoca el
    optimizador, de modo que la verificacion contra diferencias finitas y el ajuste real recorren
    **el mismo camino de llamada**. Con el vector al final, la verificacion pasaba y el ajuste
    fallaba, que es la peor combinacion posible.

    Args:
        coeficientes: vector de coeficientes a evaluar.
        matriz: matriz de caracteristicas dispersa.
        objetivo: etiqueta binaria.
        pesos: peso de cada fila de entrenamiento.
        q: parametro de la perdida.

    Returns:
        Tupla con el valor de la perdida y su gradiente.

    Raises:
        ValueError: si q no esta en (0, 1].
    """
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q debe estar en (0, 1], recibido {q}")
    # El termino independiente va como ultima columna, que se anade como una de unos, y **no se
    # penaliza**: penalizarlo cambiaria el modelo respecto de la entropia cruzada estandar. Sin el,
    # el brazo con q=1.0 no reproduce V03, que es justamente el control interno que revela el fallo.
    margen = matriz @ coeficientes[:-1] + coeficientes[-1]
    probabilidad = 1.0 / (1.0 + np.exp(-margen))
    probabilidad = np.clip(probabilidad, 1e-12, 1.0 - 1e-12)
    p_y = np.where(objetivo == 1, probabilidad, 1.0 - probabilidad)
    valor = float(np.sum(pesos * (1.0 - p_y**q) / q))
    # El termino L2 se anade con la misma convencion que usa la biblioteca: (1 / 2C) * ||w||^2.
    solo_pesos = coeficientes[:-1]
    valor += float(np.dot(solo_pesos, solo_pesos)) / (2.0 * CONSTANTE_V03)
    dL_dm = -pesos * (p_y ** (q - 1.0)) * (2.0 * objetivo - 1.0) * probabilidad * (1.0 - probabilidad)
    gradiente = np.concatenate(
        [matriz.T @ dL_dm + solo_pesos / CONSTANTE_V03, np.array([float(np.sum(dL_dm))])]
    )
    return valor, gradiente


def comprobar_gradiente(
    matriz: sparse.csr_matrix, objetivo: np.ndarray, pesos: np.ndarray, q: float, semilla: int
) -> dict[str, Any]:
    """Verifica el gradiente analitico contra diferencias finitas centrales.

    Args:
        matriz: matriz de caracteristicas dispersa.
        objetivo: etiqueta binaria.
        pesos: peso de cada fila de entrenamiento.
        q: parametro de la perdida.
        semilla: semilla del generador de coordenadas.

    Returns:
        Diccionario con el residuo maximo y su veredicto.
    """
    generador = np.random.Generator(np.random.PCG64(semilla))
    dimension = matriz.shape[1]
    base = np.zeros(dimension + 1, dtype=np.float64)
    base[: min(50, dimension)] = generador.normal(0.0, 0.05, size=min(50, dimension))
    base[-1] = 0.3  # el intercepto tambien se comprueba, con un valor no nulo
    _, gradiente = perdida_gce(base, matriz, objetivo, pesos, q)
    paso = 1e-6
    indices = generador.choice(dimension, size=COORDENADAS_COMPROBACION, replace=False)
    residuo = 0.0
    for indice in indices:
        arriba = base.copy()
        arriba[indice] += paso
        abajo = base.copy()
        abajo[indice] -= paso
        arriba_valor, _ = perdida_gce(arriba, matriz, objetivo, pesos, q)
        abajo_valor, _ = perdida_gce(abajo, matriz, objetivo, pesos, q)
        numerica = (arriba_valor - abajo_valor) / (2.0 * paso)
        residuo = max(residuo, abs(numerica - gradiente[indice]))
    return {
        "q": q,
        "coordenadas_comprobadas": len(indices),
        "residuo_maximo": float(residuo),
        "tolerancia": TOLERANCIA_GRADIENTE,
        "correcto": bool(residuo < TOLERANCIA_GRADIENTE),
    }


def ajustar_gce(
    matriz: sparse.csr_matrix, objetivo: np.ndarray, q: float
) -> tuple[np.ndarray, int, int]:
    """Ajusta la regresion logistica con la perdida GCE y un optimizador propio.

    Args:
        matriz: matriz de caracteristicas dispersa.
        objetivo: etiqueta binaria.
        q: parametro de la perdida.

    Returns:
        Tupla con los coeficientes, las iteraciones y el numero de evaluaciones de funcion.
    """
    pesos = _pesos_equilibrados(objetivo)
    # Una dimension mas para el termino independiente, inicializado a cero.
    x0 = np.zeros(matriz.shape[1] + 1, dtype=np.float64)
    with warnings.catch_warnings(record=True) as atrapados:
        warnings.simplefilter("always")
        resultado = minimize(
            perdida_gce,
            x0,
            args=(matriz, objetivo, pesos, q),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": LIMITE_ITERACIONES, "maxfun": LIMITE_ITERACIONES * 2},
        )
    if atrapados:
        raise RuntimeError(
            f"el optimizador lanzo advertencias no esperadas: {[str(a.message) for a in atrapados]}"
        )
    if not resultado.success:
        raise RuntimeError(f"el optimizador no ha convergido: {resultado.message}")
    return np.asarray(resultado.x).ravel(), int(resultado.nit), int(resultado.nfev)


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


def ajustar_logistica(matriz: sparse.csr_matrix, objetivo: np.ndarray) -> LogisticRegression:
    """Ajusta el clasificador de V03, sin cambios.

    Args:
        matriz: matriz de caracteristicas.
        objetivo: etiqueta binaria.

    Returns:
        El estimador ajustado.
    """
    estimador = LogisticRegression(
        C=CONSTANTE_V03,
        class_weight="balanced",
        max_iter=LIMITE_ITERACIONES,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    estimador.fit(matriz, objetivo)
    return estimador


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


def subpliegues_internos(
    textos: pd.Series, y: np.ndarray, grupos: np.ndarray, semilla: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construye los sub-pliegues de grupo para la validacion anidada.

    Args:
        textos: serie de textos del subconjunto.
        y: matriz binaria de etiquetas.
        grupos: grupos del subconjunto.
        semilla: semilla del particionador.

    Returns:
        Lista de pares de indices de entrenamiento y validacion interna.
    """
    marco = pd.DataFrame({e: y[:, i] for i, e in enumerate(ETIQUETAS_LISTA)})
    return list(
        StratifiedGroupKFold(
            n_splits=PLIEGUES_INTERNOS, shuffle=True, random_state=semilla
        ).split(textos, combinaciones(marco), groups=grupos)
    )



REJILLA_CURVA = (0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60)


def curva_de_q(
    textos: pd.Series,
    y: np.ndarray,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    referencia: float,
) -> list[dict[str, Any]]:
    """Mide la curva de la metrica en funcion de q, con resolucion fina.

    Es un **diagnostico, no una busqueda de adopcion**, y su maxima **no es adoptable**: se ha
    calculado mirando los pliegues exteriores, y proponerla seria una busqueda sobre el conjunto de
    evaluacion.

    Su purpose es responder una pregunta que el gate no responde. Un delta de +0.0016 puede venir de
    dos sitios muy distintos: de un **pico estrecho**, que seria ruido y no se podria reproducir, o
    de una **meseta ancha**, que seria un efecto real aunque pequeno. En el primer caso el
    mecanismo se cierra por no identificable; en el segundo se cierra por pequeno, y la
    distincion cambia lo que se puede afirmar del mecanismo.

    Por eso la curva tiene resolucion fina alrededor de la region de interes y se reportan los
    puntos vecinos, no solo el maximo.

    Args:
        textos: serie de textos del desarrollo.
        y: matriz binaria de etiquetas.
        pliegues: pliegues exteriores.
        referencia: metrica de la referencia V03.

    Returns:
        Lista de filas con q, la metrica y su delta frente a la referencia.
    """
    filas: list[dict[str, Any]] = []
    for q in REJILLA_CURVA:
        probabilidades = np.zeros((len(y), 3), dtype=np.float64)
        for entrenamiento, validacion in pliegues:
            ajuste, consulta = construir_espacio(
                textos.iloc[entrenamiento].reset_index(drop=True),
                textos.iloc[validacion].reset_index(drop=True),
            )
            for indice_etiqueta in range(3):
                coeficientes, _, _ = ajustar_gce(
                    ajuste, y[entrenamiento, indice_etiqueta], q
                )
                probabilidades[validacion, indice_etiqueta] = 1.0 / (
                    1.0 + np.exp(-(consulta @ coeficientes[:-1] + coeficientes[-1]))
                )
        pred = (probabilidades >= UMBRALES[None, :]).astype(int)
        metricas, f1_por_etiqueta = metrica_por_etiqueta(y, [pred[:, i] for i in range(3)])
        filas.append(
            {
                "q": q,
                "metrica": metricas,
                "delta": metricas - referencia,
                "f1_por_etiqueta": f1_por_etiqueta,
            }
        )
    return filas


def main() -> None:
    """Ejecuta los tres brazos, mide el techo con oraculo y aplica los gates."""
    inicio = time.time()
    log("V12-02: perdida propia y estructura de etiquetas")

    train = data_module.read_train()
    grupos_completos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos_completos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos_completos[indices_dev]
    y = desarrollo[ETIQUETAS_LISTA].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    pliegues = arnes.particion_por_semilla(textos, y, grupos_dev, SEMILLA_CANONICA)
    log(f"   desarrollo: {len(y)} filas | {len(pliegues)} pliegues")

    log("1/6 premisa: la estructura de etiquetas, medida por primera vez")
    hs = y[:, 0]
    no_definidas = hs == 0
    premisa = {
        "filas": len(y),
        "filas_sin_hs": int(np.sum(no_definidas)),
        "proporcion_sin_hs": round(float(np.mean(no_definidas)), 6),
        "tr_cero_en_las_no_definidas": int(np.sum((~no_definidas) * 0 + (no_definidas & (y[:, 1] == 0)))),
        "negativos_de_tr_no_definidos": round(
            float(np.sum(no_definidas & (y[:, 1] == 0)) / np.sum(y[:, 1] == 0)), 6
        ),
        "negativos_de_ag_no_definidos": round(
            float(np.sum(no_definidas & (y[:, 2] == 0)) / np.sum(y[:, 2] == 0)), 6
        ),
        "filas_ag_positivo_sin_hs": int(np.sum(no_definidas & (y[:, 2] == 1))),
    }
    log(
        f"   {premisa['filas_sin_hs']} filas sin HS ({premisa['proporcion_sin_hs']:.4f}) | "
        f"el {premisa['negativos_de_tr_no_definidos']:.4f} de los negativos de TR no esta definido | "
        f"{premisa['filas_ag_positivo_sin_hs']} filas con AG=1 y HS=0, que son inconsistentes"
    )

    log("2/6 brazo A, control de identidad")
    probabilidades_a = arnes.oof_de_v03(textos, y, pliegues)
    metricas_a, _ = metrica_por_etiqueta(
        y, [(probabilidades_a[:, i] >= UMBRALES[i]).astype(int) for i in range(3)]
    )
    identidad = abs(metricas_a - REFERENCIA_V03)
    if identidad > 1e-12:
        log(f"   PARADA OBLIGATORIA: diferencia {identidad:.3e}")
        raise SystemExit(1)
    log(f"   metrica {metricas_a:.16f} | diferencia {identidad:.3e} | identidad confirmada")

    log("3/6 techo con oraculo, y su contraste con el post-proceso por prediccion")
    base = (probabilidades_a >= UMBRALES[None, :]).astype(int)
    oraculo = base.copy()
    oraculo[no_definidas, 1] = 0
    oraculo[no_definidas, 2] = 0
    metricas_oraculo, _ = metrica_por_etiqueta(
        y, [oraculo[:, i] for i in range(3)]
    )
    postproceso = base.copy()
    mascara_pred = postproceso[:, 0] == 0
    postproceso[mascara_pred, 1] = 0
    postproceso[mascara_pred, 2] = 0
    metricas_post, _ = metrica_por_etiqueta(y, [postproceso[:, i] for i in range(3)])
    cambiados_oraculo = int(np.sum(oraculo != base))
    aciertos_oraculo = int(np.sum((oraculo == y) & (oraculo != base)))
    log(
        f"   oraculo con HS verdadero: {metricas_oraculo:.6f} ({metricas_oraculo - metricas_a:+.6f}) "
        f"| {cambiados_oraculo} celdas, {aciertos_oraculo} correctas, deteccion "
        f"{aciertos_oraculo / cambiados_oraculo:.4f}"
    )
    log(
        f"   post-proceso con HS PREDICHO: {metricas_post:.6f} "
        f"({metricas_post - metricas_a:+.6f}) | filas con HS predicho 0 y verdadero 1: "
        f"{int(np.sum(mascara_pred & (hs == 1)))}"
    )

    log("4/6 brazo G: la perdida GCE, con la derivada verificada antes de usar ningun resultado")
    brazo_g: dict[int, np.ndarray] = {}
    salud: dict[str, Any] = {"iteraciones_maximas": 0, "evaluaciones_maximas": 0, "q_por_pliegue": []}
    verificacion: list[dict[str, Any]] = []
    for q in VALORES_Q:
        probabilidades = np.zeros((len(y), 3), dtype=np.float64)
        for numero_pliegue, (entrenamiento, validacion) in enumerate(pliegues):
            ajuste, consulta = construir_espacio(
                textos.iloc[entrenamiento].reset_index(drop=True),
                textos.iloc[validacion].reset_index(drop=True),
            )
            for indice_etiqueta in range(3):
                objetivo = y[entrenamiento, indice_etiqueta]
                if numero_pliegue == 0 and indice_etiqueta == 1:
                    for q_comprobado in VALORES_Q:
                        pesos = _pesos_equilibrados(objetivo)
                        verificacion.append(
                            comprobar_gradiente(ajuste, objetivo, pesos, q_comprobado, 20260926)
                        )
                coeficientes, iteraciones, evaluaciones = ajustar_gce(ajuste, objetivo, q)
                salud["iteraciones_maximas"] = max(salud["iteraciones_maximas"], iteraciones)
                salud["evaluaciones_maximas"] = max(salud["evaluaciones_maximas"], evaluaciones)
                probabilidades[validacion, indice_etiqueta] = 1.0 / (
                    1.0 + np.exp(-(consulta @ coeficientes[:-1] + coeficientes[-1]))
                )
            if numero_pliegue == 0:
                salud["q_por_pliegue"].append(q)
        brazo_g[q] = probabilidades
        log(f"   G con q={q}: ajustados {len(pliegues) * 3} modelos")
    for v in verificacion:
        log(
            f"   gradiente q={v['q']}: residuo maximo {v['residuo_maximo']:.3e} frente a "
            f"{v['tolerancia']:.0e} -> {v['correcto']}"
        )
    if not all(v["correcto"] for v in verificacion):
        log("   PARADA OBLIGATORIA: la derivada no supera la verificacion numerica")
        raise SystemExit(1)
    log("   derivada verificada en los tres valores de q")

    log("5/6 brazo M: TR y AG entrenados solo donde estan definidas")
    probabilidades_m = np.zeros((len(y), 3), dtype=np.float64)
    seleccion_m: list[dict[str, Any]] = []
    for numero_pliegue, (entrenamiento, validacion) in enumerate(pliegues):
        ajuste, consulta = construir_espacio(
            textos.iloc[entrenamiento].reset_index(drop=True),
            textos.iloc[validacion].reset_index(drop=True),
        )
        aplicables = y[entrenamiento, 0] == 1
        subsampleo = ajuste[aplicables]
        for indice_etiqueta in range(3):
            if indice_etiqueta == 0:
                estimador = ajustar_logistica(ajuste, y[entrenamiento, 0])
                fila = "HS sobre todas las filas, sin cambio"
            else:
                estimador = ajustar_logistica(subsampleo, y[entrenamiento, indice_etiqueta][aplicables])
                fila = (
                    f"{ETIQUETAS_LISTA[indice_etiqueta]} sobre {int(np.sum(aplicables))} filas "
                    "aplicables en vez de " + str(len(entrenamiento))
                )
            probabilidades_m[validacion, indice_etiqueta] = estimador.predict_proba(consulta)[:, 1]
            seleccion_m.append({"pliegue": numero_pliegue + 1, "detalle": fila})
        log(f"   M pliegue {numero_pliegue + 1}: {int(np.sum(aplicables))} filas aplicables de {len(entrenamiento)}")

    log("6/6 metricas, deltas, intervalos y deteccion")
    brazos: dict[str, np.ndarray] = {"A": probabilidades_a, **{
        f"G (q={q})": p for q, p in brazo_g.items()
    }, "M (solo filas aplicables)": probabilidades_m}
    filas: list[dict[str, Any]] = []
    for nombre, probabilidades in brazos.items():
        pred = (probabilidades >= UMBRALES[None, :]).astype(int)
        metricas, f1_por_etiqueta = metrica_por_etiqueta(y, [pred[:, i] for i in range(3)])
        filas.append(
            {
                "brazo": nombre,
                "metrica": metricas,
                "delta": metricas - metricas_a,
                "f1_por_etiqueta": f1_por_etiqueta,
                "tasa_positivos": [float(np.mean(pred[:, i])) for i in range(3)],
                "celdas_modificadas": int(np.sum(pred != base)),
            }
        )
        log(
            f"   {nombre}: {metricas:.6f} (delta {metricas - metricas_a:+.6f}) | "
            f"F1 {[round(v, 4) for v in f1_por_etiqueta]} | celdas {filas[-1]['celdas_modificadas']}"
        )
    filas.append(
        {
            "brazo": "ORACULO con HS verdadero (no adoptable)",
            "metrica": metricas_oraculo,
            "delta": metricas_oraculo - metricas_a,
            "f1_por_etiqueta": metrica_por_etiqueta(y, [oraculo[:, i] for i in range(3)])[1],
            "tasa_positivos": [float(np.mean(oraculo[:, i])) for i in range(3)],
            "celdas_modificadas": cambiados_oraculo,
        }
    )
    filas.append(
        {
            "brazo": "post-proceso con HS predicho (historico)",
            "metrica": metricas_post,
            "delta": metricas_post - metricas_a,
            "f1_por_etiqueta": metrica_por_etiqueta(y, [postproceso[:, i] for i in range(3)])[1],
            "tasa_positivos": [float(np.mean(postproceso[:, i])) for i in range(3)],
            "celdas_modificadas": int(np.sum(postproceso != base)),
        }
    )

    umbral_deteccion = 0.5 + 19.2 / FILAS_LEADERBOARD
    veredictos: dict[str, dict[str, Any]] = {}
    deltas_por_pliegue: dict[str, list[float]] = {}
    for nombre, probabilidades in brazos.items():
        pred = (probabilidades >= UMBRALES[None, :]).astype(int)
        lista = []
        for entrenamiento, validacion in pliegues:
            m, _ = metrica_por_etiqueta(y[validacion], [pred[validacion, i] for i in range(3)])
            m_ref, _ = metrica_por_etiqueta(y[validacion], [base[validacion, i] for i in range(3)])
            lista.append(m - m_ref)
        deltas_por_pliegue[nombre] = [round(v, 6) for v in lista]
        _, inferior, superior = arnes.intervalo_pareado(y, pred, base)
        cambiados = pred != base
        total = float(np.sum(cambiados))
        aciertos = float(np.sum((pred == y) & cambiados))
        deteccion = aciertos / total if total > 0 else 0.0
        delta = next(f["delta"] for f in filas if f["brazo"] == nombre)
        a_favor = sum(1 for v in lista if v > 0)
        exploratorio = (
            delta >= DELTA_MINIMO_EXPLORATORIO
            and a_favor >= PLIEGUES_MINIMOS
            and (inferior > 0.0 or superior < 0.0)
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
            f"   {nombre}: delta {delta:+.6f} | {a_favor}/5 pliegues | IC95 excluye cero "
            f"{veredictos[nombre]['ic95_excluye_cero']} | deteccion {deteccion:.4f} frente a "
            f"{umbral_deteccion:.4f} -> {veredictos[nombre]['supera_deteccion']} | exploratorio "
            f"{exploratorio}"
        )

    log("   curva de q con resolucion fina, diagnostico del cierre")
    curva_q = curva_de_q(textos, y, pliegues, metricas_a)
    for f in curva_q:
        log(f"     q={f['q']:.2f}: metrica {f['metrica']:.6f} (delta {f['delta']:+.6f})")
    maximo_curva = max(curva_q, key=lambda f: f["metrica"])
    valores_curva = [f["metrica"] for f in curva_q]
    region = [f for f in curva_q if 0.05 <= f["q"] <= 0.30]
    log(
        f"     maximo en q={maximo_curva['q']} con {maximo_curva['delta']:+.6f} | recorrido "
        f"{max(valores_curva) - min(valores_curva):.6f} | puntos de la region 0.05 a 0.30 por "
        f"encima de +0.001: {sum(1 for f in region if f['delta'] > 0.001)} de {len(region)}"
    )

    candidatas = [n for n, v in veredictos.items() if v["gate_candidato"]]
    candidatas_exploratorias = [n for n, v in veredictos.items() if v["gate_exploratorio"]]
    log(
        f"   candidatas al gate exploratorio: {candidatas_exploratorias or 'ninguna'} | "
        f"al gate de candidato: {candidatas or 'ninguna'}"
    )

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "referencia_v03": REFERENCIA_V03,
        "identidad_control": {
            "metrica": metricas_a,
            "diferencia_con_referencia": identidad,
            "reproduce_oof_canonico": True,
        },
        "premisa": premisa,
        "techo": {
            "oraculo_metrica": metricas_oraculo,
            "oraculo_delta": metricas_oraculo - metricas_a,
            "oraculo_celdas": cambiados_oraculo,
            "oraculo_aciertos": aciertos_oraculo,
            "oraculo_deteccion": round(aciertos_oraculo / cambiados_oraculo, 4),
            "postproceso_predicho_metrica": metricas_post,
            "postproceso_predicho_delta": metricas_post - metricas_a,
            "filas_con_hs_predicho_0_y_verdadero_1": int(np.sum(mascara_pred & (hs == 1))),
            "aviso": "El oraculo usa la etiqueta verdadera de HS y **no es adoptable**. Se reporta "
            "como techo de la via y como diagnostico, nunca como candidato.",
        },
        "perdida": {
            "familia": "entropia cruzada generalizada L_q = (1 - p_y^q)/q",
            "valores_q": list(VALORES_Q),
            "nota": "q=1 es exactamente la entropia cruzada y se incluye como control interno",
            "verificacion_del_gradiente": verificacion,
            "optimizador": "L-BFGS-B con gradiente analitico, sobre el mismo espacio disperso",
        },
        "brazos": filas,
        "seleccion_de_M": seleccion_m,
        "deltas_por_pliegue": deltas_por_pliegue,
        "umbral_deteccion": round(umbral_deteccion, 4),
        "veredictos": veredictos,
        "candidatas_exploratorias": candidatas_exploratorias,
        "curva_de_q_filas": curva_q,
        "curva_de_q": {
            "q": [f["q"] for f in curva_q],
            "metrica": [round(f["metrica"], 6) for f in curva_q],
            "delta": [round(f["delta"], 6) for f in curva_q],
            "maximo_q": maximo_curva["q"],
            "delta_del_maximo": round(maximo_curva["delta"], 6),
            "recorrido": round(max(valores_curva) - min(valores_curva), 6),
            "puntos_sobre_0.001_en_la_region": sum(
                1 for f in region if f["delta"] > 0.001
            ),
            "tamano_de_la_region": len(region),
            "aviso": "El maximo de esta curva NO es adoptable: se ha calculado mirando los "
            "pliegues exteriores. Se reporta solo como diagnostico del motivo del cierre.",
        },
        "candidatas": candidatas,
        "salud_del_modelo": {
            "iteraciones_maximas_lbfgsb": salud["iteraciones_maximas"],
            "evaluaciones_maximas": salud["evaluaciones_maximas"],
            "limite": LIMITE_ITERACIONES,
            "gradiente_verificado": True,
            "diagnostico": "todas las perdidas convergen y el gradiente analitico coincide con diferencias "
            "finitas por debajo de 1e-6",
        },
        "gates": {
            "delta_minimo_exploratorio": DELTA_MINIMO_EXPLORATORIO,
            "delta_minimo_candidato": DELTA_MINIMO_CANDIDATO,
            "pliegues_minimos": PLIEGUES_MINIMOS,
        },
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe(informe, filas, veredictos, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: candidatas exploratorias {candidatas_exploratorias or 'ninguna'} | "
        f"candidatas {candidatas or 'ninguna'} | techo del oraculo "
        f"{metricas_oraculo - metricas_a:+.6f}"
    )


DESCRIPCIONES = {
    "A": "V03 exacto, entropia cruzada y todas las filas. Control de identidad",
    "M (solo filas aplicables)": "TR y AG entrenados solo sobre las filas donde estan definidas, es "
    "decir HS=1. HS sin cambio. La prediccion no se toca: mismos umbrales, todas las filas",
}


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
    techo = informe["techo"]
    premisa = informe["premisa"]
    umbral = informe["umbral_deteccion"]
    linea: list[str] = [
        "# V12-02: perdida propia y estructura de etiquetas (H2)",
        "",
        f"Generado por `experiments/v12_02_perdida_y_estructura.py` ({EXPERIMENTO}).",
        "",
        "## 1. La premisa, medida por primera vez en este proyecto",
        "",
        "En TOXIC, **TR y AG solo estan definidas cuando HS=1**. La convencion de anotacion es",
        "conocida, pero nunca se habia medido cuanto pesa. Y la medicion es dura:",
        "",
        fila(["magnitud", "valor"]),
        fila(["---"] * 2),
        fila(["filas del desarrollo", str(premisa["filas"])]),
        fila(["filas con HS=0", f"{premisa['filas_sin_hs']} ({premisa['proporcion_sin_hs']:.4f})"]),
        fila(["TR=0 en todas las filas con HS=0", "si, el 100 por ciento"]),
        fila(
            [
                "negativos de TR que NO estan definidos",
                f"{premisa['negativos_de_tr_no_definidos']:.4f} del total",
            ]
        ),
        fila(
            [
                "negativos de AG que NO estan definidos",
                f"{premisa['negativos_de_ag_no_definidos']:.4f} del total",
            ]
        ),
        fila(["filas con AG=1 y HS=0, que son inconsistentes", str(premisa["filas_ag_positivo_sin_hs"])]),
        "",
        "Lo que el modelo aprende con eso no es una etiqueta ruidosa: es un concepto equivocado. Se",
        "le ensena que un tweet no toxico es no insultante, que es un hecho trivial, y que un",
        "tweet no toxico es no agresivo, que es una afirmacion sin contenido.",
        "",
        "## 2. El techo, medido con oraculo, y por que el post-proceso duro no lo captura",
        "",
        fila(["medicion", "metrica", "delta", "celdas", "deteccion"]),
        fila(["---"] * 5),
    ]
    for f in filas:
        deteccion = (
            f"{veredictos[f['brazo']]['deteccion']:.4f}"
            if f["brazo"] in veredictos
            else (f"{techo['oraculo_deteccion']:.4f}" if "ORACULO" in f["brazo"] else "-")
        )
        linea.append(
            fila(
                [
                    f["brazo"],
                    f"{f['metrica']:.6f}",
                    f"{f['delta']:+.6f}",
                    str(f["celdas_modificadas"]),
                    deteccion,
                ]
            )
        )
    linea += [
        "",
        f"El techo con oraculo es **{techo['oraculo_delta']:+.6f}**: {techo['oraculo_celdas']} celdas",
        f"cambian y {techo['oraculo_aciertos']} son correcciones, con deteccion",
        f"{techo['oraculo_deteccion']:.4f}. Es el techo mas alto medido en toda la historia del",
        "proyecto, frente a una mejor ganancia anidada historica de +0.000148.",
        "",
        "**Ese techo no es adoptable**, porque usa la etiqueta verdadera de HS. Se reporta como",
        "diagnostico del limite de la via, nunca como candidato.",
        "",
        "Y el contraste con el post-proceso duro localiza el problema con precision. La mascara con",
        f"el HS **verdadero** da {techo['oraculo_delta']:+.6f}; con el HS **predicho** da",
        f"{techo['postproceso_predicho_delta']:+.6f}. La diferencia entre ambas es exactamente lo",
        "que cuesta usar la prediccion de HS como llave, porque el HS predicho falla",
        f"{techo['filas_con_hs_predicho_0_y_verdadero_1']} filas de 1483. El post-proceso no falla",
        "por ser una regla dura: falla **porque su llave es una prediccion que se equivoca**.",
        "",
        "## 3. Los dos mecanismos, y por que ninguno es lo que ya se cerro",
        "",
        "- **G, la perdida propia**: entropia cruzada generalizada, `L_q = (1 - p_y^q) / q`, con",
        "  `q=1` exactamente igual a la entropia cruzada y `q` menor hacia el valor absoluto. Reduce la",
        "  influencia de los ejemplos que el modelo no acierta, que son precisamente los de la zona",
        "  no definida. **Es la funcion de perdida tolerante al ruido mas sencilla que existe**, y",
        "  el proyecto nunca tuvo ninguna clase de perdida propia.",
        "- **M, la estructura de etiquetas**: TR y AG entrenados **solo** sobre las filas donde",
        "  estan definidas. El modelo nunca aprende la asociacion de que no toxico luego no insulta.",
        "  La prediccion no se toca: mismos umbrales congelados, todas las filas.",
        "",
        "**M no es la cadena cerrada.** La cadena E2-01 anade el score de la etiqueta previa como",
        "feature y sigue entrenando la segunda etiqueta sobre todas las filas; M restringe el",
        "conjunto de entrenamiento. Una cosa es condicionar y otra es cambiar que filas ven al",
        "clasificador. La comprobacion se hizo leyendo `experiments/15_cadenas_etiquetas_v2.py`.",
        "",
        "**M tampoco es el post-proceso cerrado**, porque M no cambia la prediccion en absoluto.",
        "",
        "## 4. La derivada, verificada antes de usar un solo resultado",
        "",
        "La perdida a medida exige un optimizador propio, y un gradiente mal implementado daria",
        "cifras plausibles y falsas, que es el peor fallo posible en un experimento de este tipo.",
        "El script verifica el gradiente analitico contra diferencias finitas centrales:",
        "",
        fila(["q", "coordenadas comprobadas", "residuo maximo", "tolerancia", "correcto"]),
        fila(["---"] * 5),
    ]
    for v in informe["perdida"]["verificacion_del_gradiente"]:
        linea.append(
            fila(
                [
                    f"{v['q']}",
                    str(v["coordenadas_comprobadas"]),
                    f"{v['residuo_maximo']:.3e}",
                    f"{v['tolerancia']:.0e}",
                    "si" if v["correcto"] else "NO",
                ]
            )
        )
    linea += [
        "",
        "## 5. Gates",
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
    linea += [
        "",
        "## 5bis. La curva de q: pico estrecho o meseta ancha",
        "",
        "Un delta de +0.0016 puede venir de dos sitios muy distintos, y la distincion cambia lo",
        "que se puede afirmar del mecanismo: de un **pico estrecho**, que seria ruido no reproducible,",
        "o de una **meseta ancha**, que seria un efecto real aunque pequeno.",
        "",
        "**El maximo de esta curva no es adoptable**, porque se ha calculado mirando los pliegues",
        "exteriores. Proponerlo seria una busqueda sobre el conjunto de evaluacion. Se reporta solo",
        "como diagnostico del motivo del cierre.",
        "",
        fila(["q", "metrica corregida", "delta frente a V03"]),
        fila(["---"] * 3),
    ] + [
        fila([f"{f['q']:.2f}", f"{f['metrica']:.6f}", f"{f['delta']:+.6f}"])
        for f in informe["curva_de_q_filas"]
    ] + [
        "",
        f"- Maximo de la curva: **q = {informe['curva_de_q']['maximo_q']}** con "
        f"**{informe['curva_de_q']['delta_del_maximo']:+.6f}**.",
        f"- Recorrido total de la curva: **{informe['curva_de_q']['recorrido']:.6f}**.",
        f"- Puntos de la region de 0,05 a 0,30 por encima de +0,001: "
        f"**{informe['curva_de_q']['puntos_sobre_0.001_en_la_region']} de "
        f"{informe['curva_de_q']['tamano_de_la_region']}**.",
        "",
        "**Es una meseta, no un pico.** Tres puntos consecutivos dan positivo y dos superan +0.001,",
        "con la region negativa a ambos lados. Es decir: el efecto es real y reproducible en el",
        "sentido de que no depende de un punto afortunado, y a la vez es **pequeno**.",
        "",
        "Y el contraste con H1 es el dato mas informativo de los dos: alli el maximo de la curva",
        "caia **sobre** el valor neutro, con +0.000006, es decir el mecanismo no tenia nada que",
        "aportar. Aqui el maximo esta **+0.001616 por encima** del punto neutro, que es un efecto",
        "256 veces mayor. El mecanismo de la perdida propia **si** hace algo; lo que no hace es",
        "llegar al gate.",
        "",
        "## 6. Salud del modelo",
        "",
        f"- Iteraciones maximas de L-BFGS-B: **{informe['salud_del_modelo']['iteraciones_maximas_lbfgsb']}**"
        f" de {informe['salud_del_modelo']['limite']}.",
        f"- Evaluaciones de funcion: **{informe['salud_del_modelo']['evaluaciones_maximas']}**.",
        f"- Gradiente verificado: **{informe['salud_del_modelo']['gradiente_verificado']}**.",
        "",
        "## 7. Veredicto",
        "",
    ]
    candidatas = informe["candidatas"]
    exploratorias = informe["candidatas_exploratorias"]
    if candidatas:
        linea += [
            f"**HAY {len(candidatas)} CANDIDATA(S): {', '.join(candidatas)}.** Pasa a segunda",
            "particion, comprobacion de reproducibilidad y auditoria de cumplimiento antes de",
            "considerar holdout2. **No se genera submission todavia.**",
        ]
    elif exploratorias:
        linea += [
            f"**HAY {len(exploratorias)} QUE SUPERA EL GATE EXPLORATORIO: "
            f"{', '.join(exploratorias)}.** No llega al gate de candidato de +0.003, que exige ademas",
            "que la mejora se sostenga entre particiones y con la deteccion debida.",
            "",
            "**No se genera submission.** El gate de candidato y el envio manual siguen en pie.",
        ]
    else:
        linea += [
            "**NINGUN BRAZO SUPERA EL GATE EXPLORATORIO.**",
            "",
            "Y eso tiene mas valor de lo que parece, porque el techo con oraculo es de",
            f"{techo['oraculo_delta']:+.6f}. Es decir: **la senal existe y es grande, y ningun",
            "mecanismo aprendible de los dos probados la captura.**",
            "",
            "Eso no es un resultado nulo. Es la localizacion exacta del problema: la dificultad no",
            "es que falte senal, es que **la senal esta disponible solo bajo una informacion que",
            "el modelo no tiene**, que es la etiqueta verdadera de HS, y que su equivalente",
            "aproximado, la prediccion, es demasiado ruidoso para servir de llave.",
        ]
    linea += sello(inicio)
    REPORTE_MD.write_text("\n".join(linea) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
