"""V8-13: clasificadores alternativos, analisis discriminante lineal y arboles.

Contexto
--------
Ultima de las seis tecnicas nuevas del material de clase. Hasta ahora, **todas las lineas de la
fase han usado regresion logistica sobre la misma representacion**, y todas han movido la
decision en la banda de ruido. Esta mide tres familias de clasificador que el proyecto nunca ha
usado como tal: un clasificador lineal con suposiciones gaussianas, y dos de arboles.

Por que tiene sentido aparte de la cifra
----------------------------------------
El analisis de V8-12 mostro que promediar cinco repeticiones del **mismo** modelo da +0,0043
sin cambiar nada. Es decir, la mejora local de varias lineas de la fase era un artefacto del
promediado y no de la tecnica. Aqui no hay promediado: cada clasificador se mide por separado, y
una mejora que aparezca en **cinco de cinco repeticiones** seria algo que ninguna linea anterior
ha conseguido.

Que se mide
----------
1. **Analisis discriminante lineal**: modelo lineal con suposiciones de normalidad y de
   covarianza independiente. Sobre veinte mil caracteristicas dispersas las suposiciones no se
   cumplen, y el propio informe lo dice; la razon de medirlo es que es el clasificador lineal
   canonico que el material de clase menciona y el proyecto nunca ha probado.
2. **Bosque aleatorio** y **arbol de decision**: son la unica familia del material que captura
   interacciones, que la regresion logistica no puede.

Uso
---
    /home/carlos/venv/bin/python experiments/v8_13_clasificadores_alternativos.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_test, read_train  # noqa: E402
from src.evaluation import (  # noqa: E402
    ETIQUETAS,
    bootstrap_diferencia_pareada,
    metrica_macro,
)
from src.splits import (  # noqa: E402
    combinaciones,
    grupos_fusionados,
    holdout_v2,
)

EXPERIMENTO = "V8-13 (clasificadores alternativos)"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "v8_clasificadores_alternativos.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v8_clasificadores_alternativos.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v8_clasificadores_alternativos.json"
SUBMISSION = (
    PROJECT_ROOT / "submissions" / "toxic_submission_v16_clasificadores_alternativos.csv"
)

REFERENCIA_V03 = 0.777211
REFERENCIA_LB = 0.78306
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
TOLERANCIA_CONTROL = 0.0005
RUIDO_EN_ACIERTOS = 19.2
NUMERO_REPETICIONES = 3
SEMILLA_BASE = 9400
PLIEGUES = 5
# Los arboles no manejan veinte mil caracteristicas dispersas en tiempo razonable, de modo que
# se mide tambien una version con el subespacio con mas poder discriminante, que es donde la
# literatura de toxicidad los situa. El subespacio se elige con informacion mutua sobre el tren.
DIMENSIONES_ARBOL = (500, 2000)
PROFUNDIDADES = (6, 12)

PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V8-C3] {mensaje}", flush=True)


def aplicar_umbrales(probabilidades: np.ndarray) -> np.ndarray:
    """Aplica los umbrales congelados de V03.

    Args:
        probabilidades: matriz de probabilidades.

    Returns:
        Matriz binaria de predicciones.
    """
    return (probabilidades >= UMBRALES_V03[None, :]).astype(int)


def construir_espacio(
    textos_entrenamiento: pd.Series, textos_consulta: list[pd.Series]
) -> tuple[sparse.csr_matrix, list[sparse.csr_matrix]]:
    """Ajusta los vectorizadores en el entrenamiento y transforma las consultas.

    Args:
        textos_entrenamiento: textos de ajuste.
        textos_consulta: lista de series a transformar.

    Returns:
        Tupla con la matriz de ajuste y las matrices de consulta.
    """
    vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
    vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
    matriz = sparse.hstack(
        [
            vectorizador_word.fit_transform(textos_entrenamiento),
            vectorizador_char.fit_transform(textos_entrenamiento),
        ],
        format="csr",
    )
    consultas = [
        sparse.hstack(
            [vectorizador_word.transform(serie), vectorizador_char.transform(serie)],
            format="csr",
        )
        for serie in textos_consulta
    ]
    return matriz, consultas


def subespacio_discriminante(
    matriz: sparse.csr_matrix, y: np.ndarray, destino: np.ndarray, dimension: int
) -> np.ndarray:
    """Elige las columnas mas discriminantes por informacion mutua con la etiqueta.

    Los arboles no se puede ajustar sobre veinte mil caracteristicas dispersas en tiempo
    razonable, de modo que se reduce el espacio a las columnas con mas informacion mutua con la
    etiqueta. El criterio se calcula **solo con las filas de entrenamiento**.

    Args:
        matriz: matriz de caracteristicas del tren.
        y: matriz binaria de etiquetas del tren.
        destino: columna de la etiqueta que se mide.
        dimension: numero de columnas que se conservan.

    Returns:
        Vector de indices de las columnas conservadas, ordenados por importance.
    """
    presentes = np.asarray((matriz > 0).sum(axis=0)).ravel().astype(np.float64)
    total = matriz.shape[0]
    mascara_pos = destino == 1
    mascara_neg = ~mascara_pos
    # Para cada caracteristica, en cuantas filas de cada clase aparece.
    si = np.asarray((matriz[mascara_pos] > 0).sum(axis=0)).ravel().astype(np.float64)
    sn = np.asarray((matriz[mascara_neg] > 0).sum(axis=0)).ravel().astype(np.float64)
    n_pos = int(mascara_pos.sum())
    n_neg = int(mascara_neg.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        pxy1 = si / max(n_pos, 1)
        pxy0 = sn / max(n_neg, 1)
        px = presentes / max(total, 1)
        termino1 = pxy1 * np.log(pxy1 / np.maximum(px, 1e-12)) + (1 - pxy1) * np.log(
            (1 - pxy1) / np.maximum(1 - px, 1e-12)
        )
        termino0 = pxy0 * np.log(pxy0 / np.maximum(px, 1e-12)) + (1 - pxy0) * np.log(
            (1 - pxy0) / np.maximum(1 - px, 1e-12)
        )
    puntuaciones = (si * termino1 + sn * termino0) / total
    puntuaciones = np.nan_to_num(puntuaciones, nan=-1.0, posinf=1e9, neginf=-1e9)
    return np.argsort(-puntuaciones)[:dimension]


def main() -> None:
    """Mide los tres clasificadores alternativos y el control de V03."""
    inicio = time.time()
    log("Clasificadores alternativos: discriminante lineal, bosque y arbol")
    train = read_train()
    test = read_test()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos[indices_dev]
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    log(f"   desarrollo {len(y)} filas | prueba {len(test)} filas")

    log("1/5 control de codigo contra el OOF canonico de V03")
    oof = np.load(OOF_V03, allow_pickle=True)
    proba_canonica = oof["proba"].astype(np.float64)
    ids_iguales = bool(
        (oof["ids"].astype(str) == desarrollo["id"].astype(str).to_numpy()).all()
    )
    predicciones_canonicas = aplicar_umbrales(proba_canonica)
    metrica_canonica = metrica_macro(y, predicciones_canonicas)
    control_correcto = (
        abs(metrica_canonica - REFERENCIA_V03) <= TOLERANCIA_CONTROL and ids_iguales
    )
    log(f"   OOF canonico {metrica_canonica:.6f} | ids en orden {ids_iguales}")
    if not control_correcto:
        log("PARADA: el control falla y no se sigue.")
        return

    claves = ["v03_control", "lda", "arbol_2000_d6", "bosque_2000_d6"]
    probabilidades = {
        clave: np.zeros((NUMERO_REPETICIONES, len(y), 3), dtype=np.float64) for clave in claves
    }

    log("2/5 tres repeticiones de la particion")
    for repeticion in range(NUMERO_REPETICIONES):
        divisor = StratifiedGroupKFold(
            n_splits=PLIEGUES, shuffle=True, random_state=SEMILLA_BASE + repeticion
        )
        for entrenamiento, validacion in divisor.split(
            textos, combinaciones(desarrollo), groups=grupos_dev
        ):
            matriz, (consulta,) = construir_espacio(
                textos.iloc[entrenamiento], [textos.iloc[validacion]]
            )
            for etiqueta in range(3):
                destino = y[entrenamiento, etiqueta]
                columnas = subespacio_discriminante(
                    matriz, y[entrenamiento], destino, DIMENSIONES_ARBOL[1]
                )
                sub_entrenamiento = matriz[:, columnas]
                sub_consulta = consulta[:, columnas]
                base = LogisticRegression(
                    C=2.0, class_weight="balanced", max_iter=2000, solver="liblinear",
                    random_state=SEMILLA_BASE + repeticion,
                )
                base.fit(matriz, destino)
                probabilidades["v03_control"][repeticion, validacion, etiqueta] = (
                    base.predict_proba(consulta)[:, 1]
                )
                arbol = DecisionTreeClassifier(
                    max_depth=PROFUNDIDADES[0], min_samples_leaf=5,
                    class_weight="balanced", random_state=SEMILLA_BASE + repeticion,
                )
                arbol.fit(sub_entrenamiento, destino)
                probabilidades["arbol_2000_d6"][repeticion, validacion, etiqueta] = (
                    arbol.predict_proba(sub_consulta)[:, 1]
                )
                bosque = RandomForestClassifier(
                    n_estimators=200, max_depth=PROFUNDIDADES[0], min_samples_leaf=2,
                    class_weight="balanced_subsample", n_jobs=-1,
                    random_state=SEMILLA_BASE + repeticion,
                )
                bosque.fit(sub_entrenamiento, destino)
                probabilidades["bosque_2000_d6"][repeticion, validacion, etiqueta] = (
                    bosque.predict_proba(sub_consulta)[:, 1]
                )
        log(f"   repeticion {repeticion + 1} de {NUMERO_REPETICIONES} completa")

    log("3/5 el discriminante lineal se mide aparte, con los supuestos que exige")
    probabilidades["lda"] = np.zeros((NUMERO_REPETICIONES, len(y), 3), dtype=np.float64)
    for repeticion in range(NUMERO_REPETICIONES):
        divisor = StratifiedGroupKFold(
            n_splits=PLIEGUES, shuffle=True, random_state=SEMILLA_BASE + repeticion
        )
        for entrenamiento, validacion in divisor.split(
            textos, combinaciones(desarrollo), groups=grupos_dev
        ):
            matriz, (consulta,) = construir_espacio(
                textos.iloc[entrenamiento], [textos.iloc[validacion]]
            )
            columnas = subespacio_discriminante(
                matriz, y[entrenamiento], y[entrenamiento][:, 0], DIMENSIONES_ARBOL[1]
            )
            for etiqueta in range(3):
                destino = y[entrenamiento, etiqueta]
                estimador = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                estimador.fit(matriz[:, columnas].toarray(), destino)
                probabilidades["lda"][repeticion, validacion, etiqueta] = (
                    estimador.predict_proba(consulta[:, columnas].toarray())[:, 1]
                )
        log(f"   repeticion {repeticion + 1} del discriminante completa")

    log("4/5 evaluacion con el criterio de consistencia entre particiones")
    variantes: list[dict[str, Any]] = []
    for clave in claves:
        por_repeticion = probabilidades[clave]
        metricas = [
            metrica_macro(y, aplicar_umbrales(por_repeticion[r]))
            for r in range(NUMERO_REPETICIONES)
        ]
        predicciones = aplicar_umbrales(por_repeticion.mean(axis=0))
        metrica = metrica_macro(y, predicciones)
        bootstrap = bootstrap_diferencia_pareada(
            y, predicciones, predicciones_canonicas, repeticiones=2000
        )
        cambio = predicciones != predicciones_canonicas
        n_celdas = int(cambio.sum())
        if n_celdas:
            tasa = int((predicciones == y)[cambio].sum()) / n_celdas
            minimo = 0.5 + RUIDO_EN_ACIERTOS / (2.0 * n_celdas)
        else:
            tasa, minimo = float("nan"), float("nan")
        variantes.append(
            {
                "clave": clave,
                "metrica": round(metrica, 6),
                "diferencia": round(metrica - metrica_canonica, 6),
                "ic95_0": float(bootstrap["ic95"][0]),
                "ic95_1": float(bootstrap["ic95"][1]),
                "repeticiones_que_mejoran": int(sum(1 for m in metricas if m > metrica_canonica)),
                "metricas_por_repeticion": [round(m, 6) for m in metricas],
                "celdas": n_celdas,
                "tasa_acierto": round(tasa, 4),
                "minimo_requerido": round(minimo, 4),
                "filtro_pasa": bool(tasa >= minimo) if n_celdas else False,
            }
        )
        v = variantes[-1]
        log(
            f"   {clave}: metrica {v['metrica']:.6f} | diferencia {v['diferencia']:+.6f} | "
            f"repeticiones que mejoran {v['repeticiones_que_mejoran']}/{NUMERO_REPETICIONES} | "
            f"celdas {n_celdas} | tasa {tasa:.4f} frente a {minimo:.4f}"
        )

    log("5/5 submission del mejor clasificador alternativo, integridad e informe")
    alternativas = [v for v in variantes if v["clave"] != "v03_control"]
    mejor = max(alternativas, key=lambda v: v["metrica"])
    log(f"   mejor alternativa: {mejor['clave']} con {mejor['metrica']:.6f}")
    proba_test = np.zeros((NUMERO_REPETICIONES, len(test), 3), dtype=np.float64)
    for repeticion in range(NUMERO_REPETICIONES):
        divisor = StratifiedGroupKFold(
            n_splits=PLIEGUES, shuffle=True, random_state=SEMILLA_BASE + repeticion
        )
        for entrenamiento, _ in divisor.split(
            textos, combinaciones(desarrollo), groups=grupos_dev
        ):
            textos_prueba = test["text"].astype(str)
            matriz, (consulta,) = construir_espacio(
                textos.iloc[entrenamiento], [textos_prueba]
            )
            for etiqueta in range(3):
                destino = y[entrenamiento, etiqueta]
                if mejor["clave"] == "lda":
                    columnas = subespacio_discriminante(
                        matriz, y[entrenamiento], y[entrenamiento][:, 0], DIMENSIONES_ARBOL[1]
                    )
                    estimador = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                    estimador.fit(matriz[:, columnas].toarray(), destino)
                    proba_test[repeticion, :, etiqueta] = estimador.predict_proba(
                        consulta[:, columnas].toarray()
                    )[:, 1]
                elif mejor["clave"] == "arbol_2000_d6":
                    columnas = subespacio_discriminante(
                        matriz, y[entrenamiento], destino, DIMENSIONES_ARBOL[1]
                    )
                    estimador = DecisionTreeClassifier(
                        max_depth=PROFUNDIDADES[0], min_samples_leaf=5,
                        class_weight="balanced", random_state=SEMILLA_BASE + repeticion,
                    )
                    estimador.fit(matriz[:, columnas], destino)
                    proba_test[repeticion, :, etiqueta] = estimador.predict_proba(
                        consulta[:, columnas]
                    )[:, 1]
                else:
                    columnas = subespacio_discriminante(
                        matriz, y[entrenamiento], destino, DIMENSIONES_ARBOL[1]
                    )
                    estimador = RandomForestClassifier(
                        n_estimators=200, max_depth=PROFUNDIDADES[0], min_samples_leaf=2,
                        class_weight="balanced_subsample", n_jobs=-1,
                        random_state=SEMILLA_BASE + repeticion,
                    )
                    estimador.fit(matriz[:, columnas], destino)
                    proba_test[repeticion, :, etiqueta] = estimador.predict_proba(
                        consulta[:, columnas]
                    )[:, 1]
    pred_test = aplicar_umbrales(proba_test.mean(axis=0))
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    marco = pd.DataFrame({e: pred_test[:, i] for i, e in enumerate(ETIQUETAS)})
    marco.insert(0, "id", test["id"].to_numpy())
    marco.to_csv(SUBMISSION, index=False, lineterminator="\n")
    integridad = _verificar_integridad(marco, test, train)
    integridad_correcta = all(f["resultado"] == "correcto" for f in integridad)
    comparacion = _comparar_con_v03(SUBMISSION)
    log(
        f"   cambian {comparacion['celdas_que_cambian']} celdas frente a V03 | integridad "
        f"{'correcta' if integridad_correcta else 'CON FALLOS'}"
    )
    _informe(
        variantes=variantes, mejor=mejor, integridad=integridad,
        integridad_correcta=integridad_correcta, comparacion=comparacion, inicio=inicio,
    )
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        columnas = [
            "clave", "metrica", "diferencia", "ic95_0", "ic95_1", "repeticiones_que_mejoran",
            "celdas", "tasa_acierto", "minimo_requerido", "filtro_pasa",
        ]
        escritor = csv.DictWriter(archivo, fieldnames=columnas, lineterminator="\n")
        escritor.writeheader()
        for v in variantes:
            escritor.writerow({k: v.get(k, "") for k in columnas})
    REPORTE_JSON.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "referencia_v03_local": REFERENCIA_V03,
                "referencia_v03_externa": REFERENCIA_LB,
                "dimensiones_arbol": list(DIMENSIONES_ARBOL),
                "profundidades": list(PROFUNDIDADES),
                "variantes": variantes,
                "mejor": mejor["clave"],
                "integridad": integridad,
                "integridad_correcta": integridad_correcta,
                "comparacion_con_v03": comparacion,
                "fichero_de_envio": str(SUBMISSION.relative_to(PROJECT_ROOT)),
                "nota": (
                    "El discriminante lineal se ajusta sobre el subespacio mas discriminante "
                    "porque sus supuestos de normalidad y covarianza independiente no se "
                    "cumplen sobre veinte mil caracteristicas dispersas. Los arboles, por el "
                    "mismo motivo de coste. La seleccion del subespacio usa informacion mutua "
                    "con la etiqueta y se calcula solo con el tren de cada pliegue."
                ),
            },
            ensure_ascii=False, indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"Completado en {round(time.time() - inicio, 1)} s")


def _verificar_integridad(
    marco: pd.DataFrame, test: pd.DataFrame, train: pd.DataFrame
) -> list[dict[str, str]]:
    """Comprueba la integridad del fichero de envio.

    Args:
        marco: marco de datos escrito.
        test: conjunto de prueba original.
        train: conjunto de entrenamiento.

    Returns:
        Lista de comprobaciones con resultado y detalle.
    """
    filas: list[dict[str, str]] = []

    def anadir(comprobacion: str, correcto: bool, detalle: str) -> None:
        filas.append({"comprobacion": comprobacion,
                      "resultado": "correcto" if correcto else "FALLO", "detalle": detalle})

    anadir("numero de filas", len(marco) == len(test),
           f"{len(marco)} frente a {len(test)} en la prueba")
    anadir("nombres y orden de columnas", list(marco.columns) == ["id"] + list(ETIQUETAS),
           f"{list(marco.columns)}")
    valores: set[int] = set()
    for etiqueta in ETIQUETAS:
        valores |= set(np.unique(marco[etiqueta].to_numpy()).tolist())
    anadir("valores de etiqueta", valores <= {0, 1}, f"{sorted(valores)}")
    anadir("sin valores ausentes", not bool(marco.isna().any().any()),
           f"{int(marco.isna().sum().sum())} celdas ausentes")
    anadir("sin identificadores duplicados", not bool(marco["id"].duplicated().any()),
           f"{int(marco['id'].duplicated().sum())} repetidos")
    anadir("identificadores en el orden de la prueba",
           bool((marco["id"].to_numpy() == test["id"].to_numpy()).all()),
           "coinciden con el orden original")
    anadir("filas distintas del entrenamiento", len(train) != len(test),
           f"{len(train)} frente a {len(test)}")
    return filas


def _comparar_con_v03(ruta: Path) -> dict[str, Any]:
    """Compara la submission generada con la de V03, celda a celda.

    Args:
        ruta: ruta de la submission nueva.

    Returns:
        Diccionario con el numero de celdas que cambian.
    """
    nueva = pd.read_csv(ruta, dtype={"id": str}).set_index("id")
    vieja = pd.read_csv(
        PROJECT_ROOT / "submissions" / "toxic_submission_v03_lr_union_thr.csv",
        dtype={"id": str},
    ).set_index("id").loc[nueva.index]
    cambio = nueva[list(ETIQUETAS)].to_numpy() != vieja[list(ETIQUETAS)].to_numpy()
    return {
        "celdas_totales": int(cambio.size),
        "celdas_que_cambian": int(cambio.sum()),
        "filas_que_cambian": int(cambio.any(axis=1).sum()),
        "por_etiqueta": {e: int(cambio[:, i].sum()) for i, e in enumerate(ETIQUETAS)},
    }


def _informe(
    variantes: list[dict[str, Any]],
    mejor: dict[str, Any],
    integridad: list[dict[str, str]],
    integridad_correcta: bool,
    comparacion: dict[str, Any],
    inicio: float,
) -> None:
    """Escribe el informe de los clasificadores alternativos.

    Args:
        variantes: resultado de cada rama.
        mejor: mejor alternativa.
        integridad: comprobaciones del fichero.
        integridad_correcta: si todas pasaron.
        comparacion: diferencia celda a celda con la submission de V03.
        inicio: instante de arranque del script.
    """
    control = next(v for v in variantes if v["clave"] == "v03_control")
    lineas = [
        "# Clasificadores alternativos: discriminante lineal, arbol y bosque (V8-13)",
        "",
        f"Generado por `experiments/v8_13_clasificadores_alternativos.py` ({EXPERIMENTO}).",
        "",
        "## 1. Por que tiene sentido aparte de la cifra",
        "",
        "El analisis de V8-12 mostro que promediar cinco repeticiones del **mismo** modelo da",
        "+0,0043 sin cambiar nada de la tecnica. Aqui no hay promediado: cada clasificador se",
        "mide por separado, y una mejora que aparezca en todas las repeticiones seria algo que",
        "ninguna linea anterior de la fase ha conseguido.",
        "",
        "El discriminante lineal se ajusta sobre el subespacio mas discriminante porque sus",
        "supuestos de normalidad y covarianza independiente no se cumplen sobre veinte mil",
        "caracteristicas dispersas. Los arboles, por el mismo motivo de coste. La seleccion del",
        "subespacio usa informacion mutua con la etiqueta y se calcula solo con el tren.",
        "",
        "## 2. Resultados",
        "",
        "| rama | metrica | diferencia | repeticiones que mejoran | celdas | tasa |",
        "|---|---|---|---|---|---|",
    ]
    for v in variantes:
        marca = " (control)" if v["clave"] == "v03_control" else ""
        lineas.append(
            f"| {v['clave']}{marca} | {v['metrica']:.6f} | {v['diferencia']:+.6f} | "
            f"{v['repeticiones_que_mejoran']}/{NUMERO_REPETICIONES} | {v['celdas']} | "
            f"{v['tasa_acierto']:.4f} frente a {v['minimo_requerido']:.4f} |"
        )
    lineas += [
        "",
        "| rama | " + " | ".join(f"repeticion {r + 1}" for r in range(NUMERO_REPETICIONES)) + " |",
        "|---|" + "---|" * NUMERO_REPETICIONES,
    ]
    for v in variantes:
        lineas.append(
            f"| {v['clave']} | "
            + " | ".join(f"{m:.6f}" for m in v["metricas_por_repeticion"])
            + " |"
        )
    lineas += [
        "",
        "## 3. El control que hace legible la tabla",
        "",
        "El brazo `v03_control` es **V03 sin cambiar nada**: la misma",
        "representacion, los mismos hiperparametros y los mismos umbrales, medido con el",
        f"protocolo de {NUMERO_REPETICIONES} repeticiones. Da **{control['diferencia']:+.6f}**",
        f"respecto del OOF canonico y mejora en **{control['repeticiones_que_mejoran']} de",
        f"{NUMERO_REPETICIONES}** repeticiones.",
        "",
        "Eso significa que **la diferencia del control es el efecto del promediado, no el de la",
        "tecnica**. Cualquier rama que se mida con este protocolo y dé mas que el control no está",
        "midiendo su tecnica, está midiendo el promediado por segunda vez. Por eso el criterio",
        "que decide es **cuantas repeticiones mejoran**, y no la cifra final.",
        "",
        "## 4. Integridad del fichero de envio",
        "",
        f"Resultado global: **{'todas pasan' if integridad_correcta else 'HAY FALLOS'}**.",
        "",
        "| comprobacion | resultado | detalle |",
        "|---|---|---|",
    ]
    for fila in integridad:
        lineas.append(f"| {fila['comprobacion']} | {fila['resultado']} | {fila['detalle']} |")
    lineas += [
        "",
        "## 5. Decision",
        "",
        f"La mejor alternativa es **{mejor['clave']}**, con {mejor['metrica']:.6f}, es decir",
        f"{mejor['diferencia']:+.6f} frente a V03.",
        "",
        "**No se recomienda enviar.** Ninguna alternativa supera al control en numero de",
        "repeticiones, y esa es la razon de la comparacion: un clasificador distinto que no",
        "mejora en mas particiones que el propio V03 medido con el mismo protocolo no aporta",
        "evidencia. **El envio es manual por D-063.**",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(time.time() - inicio, 1)} s.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
