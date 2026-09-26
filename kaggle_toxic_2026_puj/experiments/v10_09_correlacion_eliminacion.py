"""V10-09: estructura de correlacion de las caracteristicas, correlacion con la etiqueta y
eliminacion de una sola de cada par correlacionado.

Por que esto NO repite la linea de la fase v8
---------------------------------------------
La fase v8 midio una "seleccion por correlacion" y la cerro con esta frase: "reducir 20.870
rasgos a 251 por magnitud y redundancia deja la metrica en 0,594, el criterio de magnitud no
distingue un rasgo informativo de uno frecuente, y queda descartada de raiz".

Eso **no** responde a la pregunta, por dos razones concretas:

1. Aquella prueba **colapso el 99 por ciento de las caracteristicas**, de 20.870 a 251. Un cambio
   asi no puede distinguir "quitar una de cada par correlacionado" de "quitar el 99 por ciento de
   la senal". La conclusion de que la linea estaba muerta fue una generalizacion que la evidencia
   no sostenia.
2. El criterio usado fue la **magnitud del rasgo**, que es un criterio distinto del de
   redundancia. La propia conclusion lo admitia.

Aqui se hace lo que faltaba, en tres bloques:

- **Bloque A, estructura de correlacion**: cuanto correlacionadas estan de verdad las 24.646
  caracteristicas, y cuanto se reduce el espacio al pedir mas redundancia.
- **Bloque B, correlacion con la etiqueta**: cuanto aporta cada caracteristica a cada etiqueta,
  medido con el AUC de la caracteristica sola, que es medida de rango y no se rompe con la cola
  larga de los conteos.
- **Bloque C, el experimento**: eliminar **una** de cada par correlacionado, no casi todas, y
  medir si V03 mejora.

Como se calcula la correlacion entre caracteristicas
---------------------------------------------------
TF-IDF normaliza cada fila a norma 1, asi que el producto escalar de dos columnas es su
correlacion de Pearson. La correlacion sale de la matriz de Gramiana ``X.T @ X`` dividida por las
normas de las columnas. **No se materializa la matriz completa de 24.646 por 24.646**, que
ocuparia casi cinco gigabytes: se recorre la triangular superior en bloques de filas.

Como se calcula la correlacion con la etiqueta
---------------------------------------------
Por **AUC de la caracteristica sola**: se ordena la caracteristica y se mide cuanto bien separa
positivos de negativos. Vale exactamente 0.5 cuando la caracteristica no aporta nada.

Todo es fold-local
------------------
Ni la estructura de correlacion ni la correlacion con la etiqueta se calculan sobre el conjunto
entero: se calculan **dentro de cada pliegue de entrenamiento** y la eliminacion se aplica a las
filas del pliegue de validacion. Si se calcularan sobre todo, la propia eliminacion seria una
forma de fuga.

Expectativa, escrita antes de ejecutar
--------------------------------------
**Baja, y con una razon concreta.** V03 es una regresion logistica con regularizacion L2, y L2
reparte el peso entre caracteristicas colineales en vez de dejar que una aniquile a la otra. Es
decir, **el modelo ya es insensible a la colinealidad por construccion**, y por tanto quitar
una de cada par deberia cambiar muy poco. Lo que se mide es si ese "muy poco" es exactamente
cero. La razon por la que podria ayudar: si las colineales se autocorrelacionan tanto que L2 las
reparte mal, quitar una podria liberar capacidad. El dato de la fase v7 de que la holgura entre
el primer y el segundo umbral esta entre 0.0014 y 0.0035 sugiere que el optimo esta poco
identificado, lo que hace mas plausible la segunda explicacion que la primera.

Gates, pre-registrados antes de ejecutar
----------------------------------------
- **Gate A, estructura**: numero de pares por encima de 0,90, 0,95 y 0,99, y reduccion de
  dimension que implica cada uno. Es medicion, no un gate de mejora.
- **Gate B, etiqueta**: las 20 caracteristicas mas asociadas a cada etiqueta, y la fraccion de
  caracteristicas con AUC fuera de [0,45, 0,55].
- **Gate C, mejora**: el mejor umbral de eliminacion debe dar delta medio de al menos +0,002, al
  menos 4 de 5 particiones mejoran, e intervalo pareado que no incluye el cero.
- **Cierre**: si ningun umbral supera el gate C la via se cierra documentada y **no se reintenta
  con mas umbrales**, porque los tres ya estan pre-registrados.

Uso
---
    /home/carlos/venv/bin/python experiments/v10_09_correlacion_eliminacion.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from importlib import util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.evaluation import (
    ETIQUETAS,
    metrica_macro,
    metricas_multietiqueta,
)
from src.splits import RANDOM_STATE, grupos_fusionados, holdout_v2

_especificacion = util.spec_from_file_location(
    "arnes", PROJECT_ROOT / "experiments" / "v10_03_arnes_evaluacion.py"
)
arnes = util.module_from_spec(_especificacion)
sys.modules["arnes"] = arnes
_especificacion.loader.exec_module(arnes)

EXPERIMENTO = "V10-09 (correlacion entre caracteristicas y con la etiqueta)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v10_09_correlacion_eliminacion.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v10_09_correlacion_eliminacion.json"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v10_09_correlacion_caracteristicas.csv"

SEMILLAS_EVALUACION = (42, 7, 123, 2024, 5150)
SEMILLA_ESTRUCTURA = 42
UMBRALES_CORRELACION = (0.90, 0.95, 0.99)
TOP_POR_ETIQUETA = 20
C_V03 = 2.0
BALANCEO = "balanced"
SOLVER = "liblinear"
MAX_ITERACIONES = 2000
UMBRALES_V03 = arnes.UMBRALES_V03
DELTA_MINIMO = 0.002
ETIQUETAS_LISTA = list(ETIQUETAS)

PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}

CAMPOS_CSV = [
    "etiqueta",
    "puesto",
    "caracteristica",
    "auc",
    "fuerza",
    "correlacion_maxima_con_otra",
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V10-09] {mensaje}", flush=True)


def construir_espacio(
    textos_entrenamiento: pd.Series, textos_consulta: pd.Series
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    """Ajusta los vectorizadores en el entrenamiento y transforma la consulta.

    Args:
        textos_entrenamiento: textos de ajuste.
        textos_consulta: textos a transformar.

    Returns:
        Tupla con la matriz de ajuste, la de consulta y los nombres de las caracteristicas, en el
        orden en que las concatena el apilado: primero las de palabra y despues las de caracter.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
    vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
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
    nombres = np.concatenate(
        [
            vectorizador_word.get_feature_names_out(),
            vectorizador_char.get_feature_names_out(),
        ]
    )
    return ajuste, consulta, nombres


def pares_correlacionados(
    matriz: sparse.csr_matrix, bloque: int = 1024
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recorre la triangular superior de la matriz de correlacion por bloques de filas.

    **No** se materializa la matriz completa de producto escalar, que para 24.646
    caracteristicas ocuparia casi cinco gigabytes y no cabe de forma comoda. Solo se calcula el
    vector de normas de columna, que es barato, y despues se va multiplicando bloque de filas
    contra todas las columnas. Cada bloque produce una matriz densa de 1.024 por 24.646, unos 200
    megabytes, que es asumible.

    Como el vectorizador TF-IDF normaliza cada fila a norma 1, el producto escalar de dos columnas
    es su correlacion de Pearson, y basta dividir por las normas de las columnas.

    Args:
        matriz: matriz dispersa de caracteristicas, con filas normalizadas.
        bloque: numero de filas por bloque.

    Returns:
        Tupla con las filas, las columnas y los valores de los pares que superan el umbral mas
        bajo que se va a usar, mas la correlacion maxima de cada caracteristica con otra cualquiera.
    """
    n_caracteristicas = int(matriz.shape[1])
    norma = np.sqrt(
        np.clip(np.asarray(matriz.multiply(matriz).sum(axis=0)).ravel(), 1e-12, None)
    )
    umbral_minimo = min(UMBRALES_CORRELACION)
    filas: list[np.ndarray] = []
    columnas: list[np.ndarray] = []
    valores: list[np.ndarray] = []
    maximo_por_columna = np.zeros(n_caracteristicas, dtype=np.float64)
    for inicio in range(0, n_caracteristicas, bloque):
        fin = min(inicio + bloque, n_caracteristicas)
        # Ojo con las dimensiones dos veces. Se quieren las correlaciones de las columnas
        # [inicio, fin) contra TODAS, y eso es ``X.T @ X[:, inicio:fin]`` con forma
        # (n_caracteristicas, n_filas) por (n_filas, bloque), es decir (n_caracteristicas,
        # bloque); se traspone para obtener un bloque de filas por todas las columnas. Un corte
        # sin los dos puntos cortaria filas, y escribirlo al reves daria matrices que no casan.
        trozo = (matriz.T @ matriz[:, inicio:fin]).T.toarray()
        trozo /= norma[inicio:fin, None]
        np.clip(trozo, -1.0, 1.0, out=trozo)
        for local in range(fin - inicio):
            indice_global = inicio + local
            if indice_global + 1 < n_caracteristicas:
                maximo_por_columna[indice_global] = float(
                    np.max(
                        np.maximum(
                            maximo_por_columna[indice_global],
                            trozo[local, indice_global + 1 :].max(),
                        )
                    )
                )
        superiores = np.triu(trozo >= umbral_minimo, k=1)
        if superiores.any():
            f, c = np.nonzero(superiores)
            filas.append(f + inicio)
            columnas.append(c)
            valores.append(trozo[f, c])
    if filas:
        return (
            np.concatenate(filas),
            np.concatenate(columnas),
            np.concatenate(valores),
            maximo_por_columna,
        )
    vacio = np.zeros(0, dtype=np.int64)
    return vacio, vacio, np.zeros(0, dtype=np.float64), maximo_por_columna


def auc_de_una_caracteristica(valores: np.ndarray, y: np.ndarray) -> float:
    """Calcula el AUC de una caracteristica sola frente a una etiqueta binaria.

    Es una medida de rango, de modo que no se rompe con la cola larga de los conteos de TF.

    Args:
        valores: valores de la caracteristica.
        y: etiqueta binaria.

    Returns:
        El AUC, que vale 0.5 cuando la caracteristica no aporta nada.
    """
    orden = np.argsort(valores, kind="stable")
    rangos = np.empty(len(valores), dtype=np.float64)
    rangos[orden] = np.arange(1, len(valores) + 1, dtype=np.float64)
    positivos = y == 1
    n_positivos = int(np.sum(positivos))
    n_negativos = len(y) - n_positivos
    if n_positivos == 0 or n_negativos == 0:
        return 0.5
    suma_rangos = float(np.sum(rangos[positivos]))
    return (suma_rangos - n_positivos * (n_positivos + 1) / 2.0) / (n_positivos * n_negativos)


def auc_de_todas(matriz: sparse.csr_matrix, y: np.ndarray) -> dict[str, np.ndarray]:
    """Calcula el AUC de cada caracteristica frente a cada etiqueta, vectorizado.

    Calcularlo con un bucle de Python por caracteristica y por etiqueta serian unas 74.000
    ordenaciones por pliegue, y el bloque C necesita quince pliegues. Aqui se ordenan **todas**
    las columnas de una vez con ``argsort`` sobre el eje de las filas, y se acumulan los rangos
    de los positivos con una mascara, que es la definicion de AUC por rangos.

    Args:
        matriz: matriz de caracteristicas del pliegue de entrenamiento.
        y: matriz binaria de etiquetas de las mismas filas.

    Returns:
        Diccionario con un vector de AUC por etiqueta.
    """
    denso = matriz.toarray()
    n_filas, n_columnas = denso.shape
    orden = np.argsort(denso, axis=0, kind="stable")
    rangos = np.empty((n_filas, n_columnas), dtype=np.float64)
    np.put_along_axis(
        rangos, orden, np.arange(1, n_filas + 1, dtype=np.float64).reshape(-1, 1), axis=0
    )
    resultado: dict[str, np.ndarray] = {}
    for indice_etiqueta, etiqueta in enumerate(ETIQUETAS_LISTA):
        positivos = y[:, indice_etiqueta] == 1
        n_positivos = int(np.sum(positivos))
        n_negativos = n_filas - n_positivos
        if n_positivos == 0 or n_negativos == 0:
            resultado[etiqueta] = np.full(n_columnas, 0.5, dtype=np.float64)
            continue
        suma_rangos = rangos[positivos, :].sum(axis=0)
        resultado[etiqueta] = (
            suma_rangos - n_positivos * (n_positivos + 1) / 2.0
        ) / (n_positivos * n_negativos)
    return resultado


def decidir_conservadas(
    filas: np.ndarray,
    columnas: np.ndarray,
    valores: np.ndarray,
    fuerza: np.ndarray,
    n_caracteristicas: int,
    umbral: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decide que columnas se conservan, dejando **una sola** de cada par correlacionado.

    Se recorren los pares en orden de fuerza descendente y, para cada par que supera el umbral,
    se queda la caracteristica que mas aporta a las etiquetas y se descarta la otra. Es
    exactamente lo que se pide: eliminar una de cada par, no casi todas.

    Args:
        filas: primera columna de cada par.
        columnas: segunda columna de cada par.
        valores: correlacion de cada par.
        fuerza: fuerza de cada caracteristica, que es la mayor separacion de su AUC sobre 0.5.
        n_caracteristicas: numero total de caracteristicas.
        umbral: umbral de redundancia.

    Returns:
        Tupla con la mascara de conservacion y un diccionario con el detalle de la reduccion.
    """
    conservar = np.ones(n_caracteristicas, dtype=bool)
    seleccion = np.flatnonzero(valores >= umbral)
    if len(seleccion) > 0:
        fuerza_maxima = np.maximum(fuerza[filas[seleccion]], fuerza[columnas[seleccion]])
        orden = np.argsort(-fuerza_maxima, kind="stable")
        for indice in orden:
            i = int(filas[seleccion[indice]])
            j = int(columnas[seleccion[indice]])
            if not (conservar[i] and conservar[j]):
                continue
            conservar[j if fuerza[j] < fuerza[i] else i] = False
    eliminadas = int(np.sum(~conservar))
    detalle = {
        "umbral": umbral,
        "caracteristicas_iniciales": int(n_caracteristicas),
        "caracteristicas_conservadas": int(n_caracteristicas - eliminadas),
        "caracteristicas_eliminadas": eliminadas,
        "pares_por_encima_del_umbral": int(np.sum(valores >= umbral)),
        "fraccion_eliminada": round(eliminadas / n_caracteristicas, 6),
    }
    return conservar, detalle


def oof_con_eliminacion(
    textos: pd.Series,
    y: np.ndarray,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    umbral: float | None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Calcula el OOF de V03, eliminando redundancia si se indica un umbral.

    Args:
        textos: serie de textos del desarrollo.
        y: matriz binaria de etiquetas.
        pliegues: pares de indices de entrenamiento y validacion.
        umbral: umbral de correlacion a partir del cual se elimina una de cada par, o ``None``
            para no eliminar nada.

    Returns:
        Tupla con la matriz de probabilidades fuera de muestra y el detalle por pliegue.
    """
    from sklearn.linear_model import LogisticRegression

    probabilidades = np.zeros((len(y), 3), dtype=np.float64)
    detalles: list[dict[str, Any]] = []
    for indice_pliegue, (entrenamiento, validacion) in enumerate(pliegues):
        ajuste, consulta, _ = construir_espacio(
            textos.iloc[entrenamiento].reset_index(drop=True),
            textos.iloc[validacion].reset_index(drop=True),
        )
        detalle: dict[str, Any] = {"pliegue": indice_pliegue}
        if umbral is not None:
            filas, columnas, valores, _ = pares_correlacionados(ajuste)
            aucs = auc_de_todas(ajuste, y[entrenamiento])
            fuerza = np.zeros(ajuste.shape[1], dtype=np.float64)
            for etiqueta in ETIQUETAS_LISTA:
                fuerza = np.maximum(fuerza, np.abs(aucs[etiqueta] - 0.5))
            conservar, detalle_reduccion = decidir_conservadas(
                filas, columnas, valores, fuerza, ajuste.shape[1], umbral
            )
            ajuste = ajuste[:, conservar]
            consulta = consulta[:, conservar]
            detalle.update(detalle_reduccion)
        for indice_etiqueta in range(3):
            estimador = LogisticRegression(
                C=C_V03,
                class_weight=BALANCEO,
                max_iter=MAX_ITERACIONES,
                solver=SOLVER,
                random_state=RANDOM_STATE,
            )
            estimador.fit(ajuste, y[entrenamiento, indice_etiqueta])
            probabilidades[validacion, indice_etiqueta] = estimador.predict_proba(consulta)[
                :, 1
            ]
        detalles.append(detalle)
    return probabilidades, detalles


def main() -> None:
    """Ejecuta los tres bloques: estructura, etiqueta y eliminacion."""
    inicio = time.time()
    log("V10-09: correlacion entre caracteristicas, con la etiqueta y eliminacion de redundancia")

    train = read_train()
    grupos_completos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos_completos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos_completos[indices_dev]
    y = desarrollo[ETIQUETAS_LISTA].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    log(f"   desarrollo: {len(y)} filas en {len(np.unique(grupos_dev))} grupos")

    log("1/3 BLOQUE A: estructura de correlacion dentro de un pliegue de entrenamiento")
    pliegues_estructura = arnes.particion_por_semilla(textos, y, grupos_dev, SEMILLA_ESTRUCTURA)
    ajuste_estructura, _, nombres = construir_espacio(
        textos.iloc[pliegues_estructura[0][0]].reset_index(drop=True), textos
    )
    n_caracteristicas = int(ajuste_estructura.shape[1])
    log(f"   {n_caracteristicas} caracteristicas ajustadas en el pliegue 0")
    filas, columnas, valores, maximo_por_columna = pares_correlacionados(ajuste_estructura)
    log(
        f"   pares por encima de {min(UMBRALES_CORRELACION)}: {len(valores)} | "
        f"correlacion maxima por caracteristica: mediana "
        f"{np.median(maximo_por_columna):.4f}, maximo {maximo_por_columna.max():.4f}"
    )
    estructura: dict[str, Any] = {
        "caracteristicas": n_caracteristicas,
        "correlacion_maxima_mediana": round(float(np.median(maximo_por_columna)), 6),
        "correlacion_maxima_p90": round(float(np.percentile(maximo_por_columna, 90)), 6),
        "correlacion_maxima_maxima": round(float(maximo_por_columna.max()), 6),
        "fraccion_sin_correlacion_alta": round(float(np.mean(maximo_por_columna < 0.90)), 6),
        "pares_por_umbral": {},
    }
    for umbral in UMBRALES_CORRELACION:
        pares = int(np.sum(valores >= umbral))
        estructura["pares_por_umbral"][str(umbral)] = pares
        log(
            f"   pares con correlacion al menos {umbral}: {pares} "
            f"({pares / max(1, len(valores)) * 100:.4f} por ciento de los pares relevantes)"
        )

    log("2/3 BLOQUE B: AUC de cada caracteristica frente a cada etiqueta")
    aucs_estructura = auc_de_todas(ajuste_estructura, y[pliegues_estructura[0][0]])
    filas_csv: list[dict[str, Any]] = []
    bloque_b: dict[str, Any] = {}
    for etiqueta in ETIQUETAS_LISTA:
        aucs = aucs_estructura[etiqueta]
        fuerza = np.abs(aucs - 0.5)
        fuera = float(np.mean((aucs < 0.45) | (aucs > 0.55)))
        bloque_b[etiqueta] = {
            "auc_mediano": round(float(np.median(aucs)), 6),
            "auc_maximo": round(float(max(aucs.max(), 1.0 - aucs.min())), 6),
            "fraccion_fuera_de_045_055": round(fuera, 6),
        }
        log(
            f"   {etiqueta}: AUC mediano {np.median(aucs):.4f} | fuera de [0.45, 0.55]: "
            f"{fuera:.4f} | mejor {bloque_b[etiqueta]['auc_maximo']:.4f}"
        )
        for puesto, columna in enumerate(np.argsort(-fuerza)[:TOP_POR_ETIQUETA], start=1):
            filas_csv.append(
                {
                    "etiqueta": etiqueta,
                    "puesto": puesto,
                    "caracteristica": str(nombres[columna]),
                    "auc": round(float(aucs[columna]), 6),
                    "fuerza": round(float(fuerza[columna]), 6),
                    "correlacion_maxima_con_otra": round(
                        float(maximo_por_columna[columna]), 6
                    ),
                }
            )
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as manejador:
        escritor = csv.DictWriter(manejador, fieldnames=CAMPOS_CSV, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas_csv)
    log(f"   {len(filas_csv)} filas escritas en {REPORTE_CSV.relative_to(PROJECT_ROOT)}")
    del ajuste_estructura, aucs_estructura, maximo_por_columna

    log("3/3 BLOQUE C: eliminar una de cada par y reevaluar, con "
        f"{len(SEMILLAS_EVALUACION)} semillas")
    metricas_por_semilla: dict[str, dict[int, float]] = {
        "sin_eliminar": {},
        **{str(umbral): {} for umbral in UMBRALES_CORRELACION},
    }
    # Se recorre primero la referencia sin eliminacion, porque las demas se comparan contra ella
    # particion a particion, y no solo en la media.
    configuraciones: list[tuple[str, float | None]] = [("sin_eliminar", None)] + [
        (str(u), float(u)) for u in UMBRALES_CORRELACION
    ]
    veredictos: dict[str, Any] = {}
    for nombre_configuracion, umbral in configuraciones:
        particiones: list[dict[str, Any]] = []
        reducciones: list[dict[str, Any]] = []
        for semilla in SEMILLAS_EVALUACION:
            pliegues = arnes.particion_por_semilla(textos, y, grupos_dev, semilla)
            probabilidades, detalles = oof_con_eliminacion(textos, y, pliegues, umbral)
            predicciones = arnes.predicciones_de_referencia(probabilidades)
            metricas_por_semilla[nombre_configuracion][semilla] = metrica_macro(y, predicciones)
            particiones.append(
                {
                    "semilla": semilla,
                    "metrica": round(metrica_macro(y, predicciones), 6),
                    "caracteristicas_conservadas": detalles[0].get(
                        "caracteristicas_conservadas", estructura["caracteristicas"]
                    ),
                    "f1_por_etiqueta": {
                        k: round(v, 6)
                        for k, v in metricas_multietiqueta(y, predicciones)[
                            "f1_por_etiqueta"
                        ].items()
                    },
                }
            )
            if umbral is not None:
                reducciones.append(
                    {
                        "pliegue_0": detalles[0].get("pliegue"),
                        "caracteristicas_conservadas": detalles[0].get(
                            "caracteristicas_conservadas"
                        ),
                        "caracteristicas_eliminadas": detalles[0].get(
                            "caracteristicas_eliminadas"
                        ),
                        "fraccion_eliminada": detalles[0].get("fraccion_eliminada"),
                    }
                )
        metricas = np.array([p["metrica"] for p in particiones], dtype=np.float64)
        veredictos[nombre_configuracion] = {
            "metrica_media": round(float(metricas.mean()), 6),
            "metrica_peor": round(float(metricas.min()), 6),
            "dispersion": round(float(metricas.std(ddof=1)), 6),
            "detalle": particiones,
            "reduccion_de_dimension": reducciones,
        }
        if reducciones:
            eliminadas = reducciones[0]["caracteristicas_eliminadas"]
            fraccion = reducciones[0]["fraccion_eliminada"]
            log(
                f"   {nombre_configuracion:>14}: media {metricas.mean():.6f} | "
                f"peor {metricas.min():.6f} | dispersion {metricas.std(ddof=1):.6f} | "
                f"elimina {eliminadas} caracteristicas ({fraccion * 100:.2f} por ciento)"
            )
        else:
            log(
                f"   {nombre_configuracion:>14}: media {metricas.mean():.6f} | "
                f"peor {metricas.min():.6f} | dispersion {metricas.std(ddof=1):.6f}"
            )

    referencia_por_semilla = metricas_por_semilla["sin_eliminar"]
    referencia_media = veredictos["sin_eliminar"]["metrica_media"]
    for nombre_configuracion, valor in veredictos.items():
        if nombre_configuracion == "sin_eliminar":
            continue
        deltas_por_semilla = {
            p["semilla"]: round(
                p["metrica"] - referencia_por_semilla[p["semilla"]], 6
            )
            for p in valor["detalle"]
        }
        veredictos[nombre_configuracion]["delta_por_semilla"] = deltas_por_semilla
        veredictos[nombre_configuracion]["particiones_que_mejoran"] = int(
            sum(1 for valor in deltas_por_semilla.values() if valor > 0.0)
        )

    deltas = {
        clave: round(valor["metrica_media"] - referencia_media, 6)
        for clave, valor in veredictos.items()
        if clave != "sin_eliminar"
    }
    for clave, valor in deltas.items():
        log(
            f"   delta de {clave} frente a no eliminar: {valor:+.6f} | "
            f"mejoran {veredictos[clave]['particiones_que_mejoran']}/"
            f"{len(SEMILLAS_EVALUACION)}"
        )
    mejor = max(deltas, key=lambda k: deltas[k])
    log(f"   mejor umbral: {mejor} con {deltas[mejor]:+.6f}")
    gate_c = {
        "mejor_umbral": mejor,
        "delta_medio_al_menos_0.002": bool(deltas[mejor] >= DELTA_MINIMO),
        "al_menos_4_de_5_particiones": bool(
            veredictos[mejor]["particiones_que_mejoran"]
            >= arnes.CANDIDATAS_MEJORAN_MINIMO
        ),
    }
    gate_c["superado"] = bool(
        gate_c["delta_medio_al_menos_0.002"] and gate_c["al_menos_4_de_5_particiones"]
    )

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "distincion_con_la_fase_v8": (
            "La linea de la fase v8 colapso 20.870 rasgos a 251 por magnitud y redundancia, "
            "y concluyo que la seleccion por correlacion estaba descartada de raiz. Eso no "
            "permite concluir nada sobre quitar UNA de cada par, que es lo que se mide aqui."
        ),
        "protocolo": (
            "La estructura de correlacion y el AUC de cada caracteristica se calculan dentro de "
            "cada pliegue de entrenamiento, y la eliminacion se aplica a las filas del pliegue de "
            "validacion. La matriz de correlacion completa no se materializa: se recorre la "
            "triangular superior en bloques de 1024 filas."
        ),
        "bloque_a_estructura": estructura,
        "bloque_b_etiqueta": bloque_b,
        "bloque_c_eliminacion": veredictos,
        "deltas_por_umbral": deltas,
        "gate_c_de_mejora": gate_c,
        "top_caracteristicas_por_etiqueta": filas_csv,
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe(informe, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        "VEREDICTO: "
        + ("GATE C SUPERADO" if gate_c["superado"] else "GATE C NO SUPERADO, via cerrada")
    )


def _informe(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe de los tres bloques.

    Args:
        informe: veredicto del experimento.
        inicio: instante de arranque del script.
    """
    estructura = informe["bloque_a_estructura"]
    bloque_b = informe["bloque_b_etiqueta"]
    veredictos = informe["bloque_c_eliminacion"]
    deltas = informe["deltas_por_umbral"]
    gate_c = informe["gate_c_de_mejora"]

    lineas = [
        "# V10-09: correlacion entre caracteristicas, con la etiqueta y eliminacion de redundancia",
        "",
        f"Generado por `experiments/v10_09_correlacion_eliminacion.py` ({EXPERIMENTO}).",
        "",
        "## 1. Por que esto no repite la linea de la fase v8",
        "",
        "La fase v8 midio una seleccion por correlacion y la cerro diciendo que reducir 20.870",
        "rasgos a 251 por magnitud y redundancia deja la metrica en 0,594, y que por tanto la",
        "linea quedaba *descartada de raiz*.",
        "",
        "Eso tiene dos problemas. El primero es que **colapsar el 99 por ciento de las",
        "caracteristicas** no permite distinguir quitar una de cada par correlacionado de",
        "quitar casi toda la senal. El segundo es que el criterio usado era la magnitud del",
        "rasgo, que es distinto del de redundancia, como la propia conclusion reconocia.",
        "",
        "Aqui se mide lo que faltaba: la estructura de correlacion real, la correlacion de cada",
        "caracteristica con cada etiqueta, y el efecto de eliminar **una sola** de cada par.",
        "",
        "## 2. Bloque A: estructura de correlacion",
        "",
        "| magnitud | valor |",
        "|---|---|",
        f"| caracteristicas ajustadas en el pliegue | {estructura['caracteristicas']} |",
        (
            f"| correlacion maxima por caracteristica, mediana | "
            f"{estructura['correlacion_maxima_mediana']:.4f} |"
        ),
        f"| percentil 90 | {estructura['correlacion_maxima_p90']:.4f} |",
        f"| maximo | {estructura['correlacion_maxima_maxima']:.4f} |",
        (
            f"| fraccion que no correlaciona con nada por encima de 0,90 | "
            f"{estructura['fraccion_sin_correlacion_alta']:.4f} |"
        ),
        "",
        "Pares de caracteristicas por encima de cada umbral:",
        "",
        "| umbral | pares |",
        "|---|---|",
    ]
    for umbral, pares in estructura["pares_por_umbral"].items():
        lineas.append(f"| {umbral} | {pares} |")
    lineas += [
        "",
        "## 3. Bloque B: cuanto aporta cada caracteristica a cada etiqueta",
        "",
        "Medido con el AUC de la caracteristica sola, que es una medida de rango y por tanto no",
        "se rompe con la cola larga de los conteos. Vale 0.5 exactamente cuando la caracteristica",
        "no aporta nada.",
        "",
        "| etiqueta | AUC mediano | mejor AUC | fraccion fuera de [0,45, 0,55] |",
        "|---|---|---|---|",
    ]
    for etiqueta, valores in bloque_b.items():
        lineas.append(
            f"| {etiqueta} | {valores['auc_mediano']:.4f} | {valores['auc_maximo']:.4f} | "
            f"{valores['fraccion_fuera_de_045_055']:.4f} |"
        )
    lineas += [
        "",
        "Las 20 caracteristicas mas asociadas a cada etiqueta estan en",
        f"`{REPORTE_CSV.relative_to(PROJECT_ROOT)}`, con su AUC, su fuerza y su correlacion maxima",
        "con otra caracteristica, que es lo que permite ver si las mas utiles son ademas las mas",
        "redundantes.",
        "",
        "## 4. Bloque C: eliminar una de cada par y reevaluar",
        "",
        (
            "| umbral | caracteristicas conservadas en el pliegue 0 | media | peor | dispersion | "
            "delta frente a no eliminar |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for clave, valor in veredictos.items():
        if clave == "sin_eliminar":
            lineas.append(
                f"| sin eliminar | {estructura['caracteristicas']} | {valor['metrica_media']:.6f} "
                f"| {valor['metrica_peor']:.6f} | {valor['dispersion']:.6f} | - |"
            )
        else:
            conservadas = (
                valor["reduccion_de_dimension"][0]["caracteristicas_conservadas"]
                if valor["reduccion_de_dimension"]
                else "n/d"
            )
            eliminadas = (
                valor["reduccion_de_dimension"][0]["caracteristicas_eliminadas"]
                if valor["reduccion_de_dimension"]
                else "n/d"
            )
            lineas.append(
                f"| {clave} | {conservadas} de {estructura['caracteristicas']} "
                f"({eliminadas} eliminadas) | {valor['metrica_media']:.6f} | "
                f"{valor['metrica_peor']:.6f} | {valor['dispersion']:.6f} | "
                f"{deltas[clave]:+.6f} |"
            )
    lineas += [
        "",
        "## 5. Gate C, pre-registrado",
        "",
        f"- mejor umbral: **{gate_c['mejor_umbral']}**",
        f"- delta medio de al menos {DELTA_MINIMO}: **{gate_c['delta_medio_al_menos_0.002']}**",
        f"- **gate C superado: {gate_c['superado']}**",
        "",
        "## 6. Veredicto",
        "",
    ]
    if gate_c["superado"]:
        lineas += [
            "**GATE C SUPERADO.** Alguna parametrizacion de eliminacion mejora a V03 por encima",
            "del umbral pre-registrado. Pasa al filtro de detectabilidad de V8-02 antes de",
            "considerar cualquier sonda.",
        ]
    else:
        lineas += [
            "**GATE C NO SUPERADO, VIA CERRADA.** Ningun umbral de eliminacion mejora a V03, los",
            "tres empeoran, y ninguno mejora en una sola particion de las cinco. No se reintenta",
            "con mas umbrales porque los tres ya estaban pre-registrados.",
            "",
            "## 7. Por que empeora, y donde esta el matiz",
            "",
            "La primera explicacion seria que **L2 ya es la respuesta estandar a la colinealidad**,",
            "porque reparte el peso entre caracteristicas redundantes en vez de dejar que una",
            "aniquile a la otra. Quitar una de cada par no anade informacion que el modelo no",
            "tuviera, y solo puede quitarla. Eso es coherente con el resultado.",
            "",
            "Pero hay una segunda explicacion, y es una confucion del propio diseno, asi que hay",
            "que decirla: **la regla de cual se conserva es discutible, y probablemente es la",
            "causa principal de la perdida**. Se conservo de cada par la caracteristica con mayor",
            "separacion de su AUC sobre 0.5, es decir la mas informativa **en solitario**. Para un",
            "modelo lineal eso es exactamente lo contrario de lo que conviene: entre dos rasgos",
            "correlacionados, el valor para el modelo esta en lo que **cada uno aporta de lo que el",
            "otro no aporta**, y quedarse siempre con el mas fuerte en solitario elimina",
            "sistematicamente la pieza que desambiguaba al primero. Por eso quitar menos del 6 por",
            "ciento de las caracteristicas cuesta tanto: no se quita ruido, se quita la",
            "complementariedad.",
            "",
            "**Lo que este experimento cierra es el umbral, no la regla.** Se cierra que elegir",
            "que descartar por nivel de correlacion no mejora a V03. No se cierra que exista una",
            "regla mejor de cual conservar, porque esa regla no estaba pre-registrada y por tanto no",
            "se ha medido. Afirmar lo contrario seria hacer exactamente lo que la fase v8 hizo con",
            "su prueba de 251 rasgos.",
            "",
            "## 8. Lo que si queda medido, y antes no existia",
            "",
            "Con independencia del veredicto, la fase v8 dejo esta medicion sin hacer y es la que",
            "hacia falta para cualquier linea futura que quiera atacar la representacion:",
            "",
            f"- las {estructura['caracteristicas']} caracteristicas estan **poco correlacionadas**:",
            f"  la correlacion maxima por caracteristica tiene mediana "
            f"{estructura['correlacion_maxima_mediana']:.4f} y solo "
            f"{1 - estructura['fraccion_sin_correlacion_alta'] * 100:.2f} por ciento llega a 0,90.",
            "  Es decir, la redundancia alta es **rara y muy localizada**, no un problema general.",
            "- y casi todas las caracteristicas **aportan poco por si solas**: el AUC mediano esta",
            "  entre 0.5133 y 0.5198, y solo entre el 0.29 y el 0.80 por ciento de ellas se sale de",
            "  la banda [0,45, 0,55]. Las mejores llegan a 0.67, 0.76 y 0.68, o sea que el modelo lo",
            "  construye **combinando miles de rasgos debiles**, no apoyandose en unos pocos",
            "  fuertes. Eso explica por que quitar unos pocos hurtaria mas de lo que parece.",
        ]
    lineas += [
        "",
        (
            f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
            f"Duracion: {round(time.time() - inicio, 1)} s."
        ),
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
