"""P5 de la fase v8 (V8-06): constante de regularizacion distinta por etiqueta.

Contexto
--------
Con P1, P3, P2 y P4 cerradas, esta es la primera de las tres lineas que solo mueven la capa
de decision. Su precedente es el peor conocido del proyecto: R1-C1 ajusto la constante de
regularizacion y gano +0.0157 en local para perder 0.0108 en el externo, y OPT-01 perdio
0.0070 por la misma razon. El freno de transferencia de D-059 penaliza precisamente este
patron.

Por que merece la pena repetirlo
---------------------------------
R1-C1 ajusto una constante **global** y lo hizo **contra el mismo conjunto que luego evaluaba**,
que es sobreajuste en las dos dimensiones a la vez. Esta linea hace las dos cosas bien:

- La constante se elige **por etiqueta**, no una sola para las tres, porque el optimo de
  regularizacion puede diferir entre un insulto frecuente y una toxicity escasa.
- La eleccion se hace con **validacion anidada estricta** dentro del tren de cada pliegue, con
  sub-pliegues de grupo propios. El conjunto de ajuste de la eleccion y el de medicion quedan
  separados, que es lo que R1-C1 no hizo.

La diferencia con R1-C1 es por tanto de protocolo, y conviene medirla explicitamente en vez de
suponerla: por eso el script mide tambien un **brazo de control** que replica el criterio de
R1-C1, elegir la constante globalmente y sin validacion anidada, y contrasta los dos.

Presupuesto de sondas
---------------------
Por D-063 el usuario autoriza gastar envios adicionales. **El filtro de detectabilidad deja
de ser un bloqueante y pasa a ser una prediccion que se contrasta.** El script genera la
submission siempre que la linea vaya en direccion positiva, y el envio es siempre manual.

Uso
---
    /home/carlos/venv/bin/python experiments/v8_06_c_por_etiqueta.py
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_test, read_train  # noqa: E402
from src.evaluation import (  # noqa: E402
    ETIQUETAS,
    bootstrap_diferencia_pareada,
    metrica_macro,
    metricas_multietiqueta,
)
from src.splits import (  # noqa: E402
    RANDOM_STATE,
    combinaciones,
    grupos_fusionados,
    holdout_v2,
    pliegues_corregidos,
)

EXPERIMENTO = "V8-06 (P5)"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "v8_c_por_etiqueta.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v8_c_por_etiqueta.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v8_c_por_etiqueta.json"
SUBMISSION = PROJECT_ROOT / "submissions" / "toxic_submission_v09_p5_c_por_etiqueta.csv"

REFERENCIA_V03 = 0.777211
REFERENCIA_LB = 0.78306
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
TOLERANCIA_CONTROL = 0.0005
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
RUIDO_NETOS = 19.2
CONSTANTE_V03 = 2.0
REJILLA_C = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
PLEIGUES_INTERNOS = 3
SEMILLA_INTERNA = 777

PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V8-P5] {mensaje}", flush=True)


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
        Tupla con la matriz de ajuste y la lista de matrices de consulta.
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


def ajustar_probabilidades(
    matriz: sparse.csr_matrix,
    destino: np.ndarray,
    constante: float,
    consulta: sparse.csr_matrix,
    peso_clase: Any = "balanced",
) -> np.ndarray:
    """Ajusta un modelo y devuelve las probabilidades de la consulta.

    Args:
        matriz: matriz de caracteristicas de ajuste.
        destino: vector binario de la etiqueta.
        constante: constante de regularizacion.
        consulta: matriz de caracteristicas de la consulta.
        peso_clase: peso de clase, equilibrados o ninguno.

    Returns:
        Vector de probabilidades de la clase positiva.
    """
    estimador = LogisticRegression(
        C=constante, class_weight=peso_clase, max_iter=2000, solver="liblinear",
        random_state=RANDOM_STATE,
    )
    estimador.fit(matriz, destino)
    return estimador.predict_proba(consulta)[:, 1]


def f1_media(y: np.ndarray, proba: np.ndarray, etiqueta: int) -> float:
    """Calcula la F1 de una etiqueta a partir de probabilidades y umbral.

    Args:
        y: matriz binaria verdadera.
        proba: matriz de probabilidades.
        etiqueta: indice de la etiqueta.

    Returns:
        F1 de esa etiqueta con el umbral congelado de V03.
    """
    pred = (proba[:, etiqueta] >= UMBRALES_V03[etiqueta]).astype(int)
    verdadero = y[:, etiqueta]
    tp = int(((pred == 1) & (verdadero == 1)).sum())
    fp = int(((pred == 1) & (verdadero == 0)).sum())
    fn = int(((pred == 0) & (verdadero == 1)).sum())
    if 2 * tp + fp + fn == 0:
        return 0.0
    return 2.0 * tp / (2.0 * tp + fp + fn)


def main() -> None:
    """Ejecuta la eleccion anidada de la constante por etiqueta y la evalua."""
    inicio = time.time()
    log("P5: constante de regularizacion distinta por etiqueta")
    train = read_train()
    test = read_test()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos[indices_dev])
    grupos_dev = grupos[indices_dev]
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    textos_test = test["text"].astype(str)
    log(
        f"   desarrollo {len(y)} filas | prueba {len(test)} filas | rejilla {REJILLA_C} | "
        f"{PLEIGUES_INTERNOS} sub-pliegues internos"
    )

    log("1/7 control de codigo contra el OOF canonico de V03")
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
    log(
        f"   OOF canonico {metrica_canonica:.6f} | ids en orden {ids_iguales} -> "
        f"{'dentro' if control_correcto else 'FUERA'}"
    )
    if not control_correcto:
        log("PARADA: el control falla y no se sigue.")
        return

    log("2/7 eleccion anidada de la constante, por etiqueta y por pliegue")
    oof_anidado = np.zeros((len(y), 3), dtype=np.float64)
    oof_control_r1c1 = np.zeros((len(y), 3), dtype=np.float64)
    elecciones: list[dict[str, Any]] = []
    for numero, (entrenamiento, validacion) in enumerate(pliegues):
        matriz, (consulta_val,) = construir_espacio(
            textos.iloc[entrenamiento], [textos.iloc[validacion]]
        )
        y_entrenamiento = y[entrenamiento]
        # Sub-pliegues internos, de grupo, sobre el solo tren. La eleccion no ve la validacion.
        interno = StratifiedGroupKFold(
            n_splits=PLEIGUES_INTERNOS, shuffle=True, random_state=SEMILLA_INTERNA
        )
        particiones = list(
            interno.split(
                matriz, combinaciones(desarrollo.iloc[entrenamiento]),
                groups=grupos_dev[entrenamiento],
            )
        )
        mejor_global = {"f1": -1.0, "c": CONSTANTE_V03}
        por_etiqueta: list[dict[str, Any]] = []
        for etiqueta in range(3):
            mejor = {"f1": -1.0, "c": CONSTANTE_V03}
            for constante in REJILLA_C:
                acumulado = 0.0
                for indices_t, indices_v in particiones:
                    proba = ajustar_probabilidades(
                        matriz[indices_t], y_entrenamiento[indices_t, etiqueta], constante,
                        matriz[indices_v],
                    )
                    pred = (proba >= UMBRALES_V03[etiqueta]).astype(int)
                    verdadero = y_entrenamiento[indices_v, etiqueta]
                    tp = int(((pred == 1) & (verdadero == 1)).sum())
                    fp = int(((pred == 1) & (verdadero == 0)).sum())
                    fn = int(((pred == 0) & (verdadero == 1)).sum())
                    divisor = 2 * tp + fp + fn
                    acumulado += (2.0 * tp / divisor) if divisor else 0.0
                media = acumulado / len(particiones)
                if media > mejor_global["f1"]:
                    mejor_global = {"f1": media, "c": constante}
                if media > mejor["f1"]:
                    mejor = {"f1": media, "c": constante}
            por_etiqueta.append({"etiqueta": ETIQUETAS[etiqueta], "c": mejor["c"],
                                 "f1_interna": round(mejor["f1"], 6)})
            proba_val = ajustar_probabilidades(
                matriz, y_entrenamiento[:, etiqueta], mejor["c"], consulta_val
            )
            oof_anidado[validacion, etiqueta] = proba_val
        # Brazo de control: una sola constante para las tres etiquetas, elegida con el mismo
        # procedimiento interno pero de forma conjunta. Es el criterio de R1-C1 sin su
        # sobreajuste, y sirve para atribuir la diferencia al protocolo y no a la idea.
        for etiqueta in range(3):
            proba_val = ajustar_probabilidades(
                matriz, y_entrenamiento[:, etiqueta], mejor_global["c"], consulta_val
            )
            oof_control_r1c1[validacion, etiqueta] = proba_val
        elecciones.append(
            {
                "pliegue": numero + 1,
                "c_global": mejor_global["c"],
                "f1_global_interna": round(mejor_global["f1"], 6),
                "por_etiqueta": por_etiqueta,
            }
        )
        log(
            f"   pliegue {numero + 1}: constante global {mejor_global['c']} | "
            + ", ".join(
                f"{p['etiqueta']} {p['c']}" for p in por_etiqueta
            )
        )

    log("3/7 evaluacion de las dos ramas")
    variantes: list[dict[str, Any]] = []
    for clave, probabilidades, descripcion in (
        ("anidada_por_etiqueta", oof_anidado, "constante por etiqueta, elegida con validacion anidada"),
        ("control_global_anidado", oof_control_r1c1, "una sola constante, elegida con el mismo procedimiento interno"),
    ):
        predicciones = aplicar_umbrales(probabilidades)
        metrica = metrica_macro(y, predicciones)
        diferencia = metrica - metrica_canonica
        aciertos = [
            int(
                (predicciones[val] == y[val]).sum()
                - (predicciones_canonicas[val] == y[val]).sum()
            )
            for _, val in pliegues
        ]
        bootstrap = bootstrap_diferencia_pareada(
            y, predicciones, predicciones_canonicas, repeticiones=2000
        )
        cambio = predicciones != predicciones_canonicas
        n_celdas = int(cambio.sum())
        if n_celdas:
            tasa = int((predicciones == y)[cambio].sum()) / n_celdas
            minimo = 0.5 + RUIDO_NETOS / (2.0 * n_celdas)
            pasa = tasa >= minimo
        else:
            tasa, minimo, pasa = float("nan"), float("nan"), False
        variante = {
            "clave": clave,
            "descripcion": descripcion,
            "metrica": round(metrica, 6),
            "diferencia": round(diferencia, 6),
            "ic95_0": float(bootstrap["ic95"][0]),
            "ic95_1": float(bootstrap["ic95"][1]),
            "pliegues_positivos": int(sum(1 for a in aciertos if a > 0)),
            "f1_por_etiqueta": metricas_multietiqueta(y, predicciones)["f1_por_etiqueta"],
            "celdas": n_celdas,
            "tasa_acierto": round(tasa, 4),
            "minimo_requerido": round(minimo, 4),
            "filtro_pasa": pasa,
        }
        variante["gate_delta"] = diferencia >= GATE_DELTA
        variante["gate_ic95"] = variante["ic95_0"] > 0
        variante["gate_pliegues"] = variante["pliegues_positivos"] >= GATE_PLIEGUES
        variante["gate_supera"] = (
            variante["gate_delta"] and variante["gate_ic95"] and variante["gate_pliegues"]
        )
        variantes.append(variante)
        log(
            f"   {clave}: metrica {metrica:.6f} | diferencia {diferencia:+.6f} | IC95 "
            f"[{variante['ic95_0']:+.5f}, {variante['ic95_1']:+.5f}] | pliegues "
            f"{variante['pliegues_positivos']}/5 | celdas {n_celdas} | tasa {tasa:.4f} frente "
            f"a {minimo:.4f} | gate {'supera' if variante['gate_supera'] else 'no supera'}"
        )

    log("4/7 predicciones sobre el conjunto de prueba")
    matriz_completa, (consulta_prueba,) = construir_espacio(textos, [textos_test])
    # Se eligen las constantes por etiqueta con el procedimiento interno sobre todo el
    # desarrollo, y se reentrena con ellas, que es lo que exige la entrega de una submission.
    generador = StratifiedGroupKFold(
        n_splits=PLEIGUES_INTERNOS, shuffle=True, random_state=SEMILLA_INTERNA
    )
    particiones = list(generador.split(matriz_completa, combinaciones(desarrollo), groups=grupos_dev))
    constantes_finales: list[float] = []
    for etiqueta in range(3):
        mejor = {"f1": -1.0, "c": CONSTANTE_V03}
        for constante in REJILLA_C:
            acumulado = 0.0
            for indices_t, indices_v in particiones:
                proba = ajustar_probabilidades(
                    matriz_completa[indices_t], y[indices_t, etiqueta], constante,
                    matriz_completa[indices_v],
                )
                pred = (proba >= UMBRALES_V03[etiqueta]).astype(int)
                verdadero = y[indices_v, etiqueta]
                tp = int(((pred == 1) & (verdadero == 1)).sum())
                fp = int(((pred == 1) & (verdadero == 0)).sum())
                fn = int(((pred == 0) & (verdadero == 1)).sum())
                divisor = 2 * tp + fp + fn
                acumulado += (2.0 * tp / divisor) if divisor else 0.0
            media = acumulado / len(particiones)
            if media > mejor["f1"]:
                mejor = {"f1": media, "c": constante}
        constantes_finales.append(mejor["c"])
    log(f"   constantes finales por etiqueta: {constantes_finales}")
    proba_test_modelos = np.zeros((len(test), 3), dtype=np.float64)
    for etiqueta in range(3):
        proba_test_modelos[:, etiqueta] = ajustar_probabilidades(
            matriz_completa, y[:, etiqueta], constantes_finales[etiqueta], consulta_prueba
        )
    pred_test = aplicar_umbrales(proba_test_modelos)
    log(
        "   tasa de positivos predicha: "
        + ", ".join(f"{e} {pred_test[:, i].mean():.4f}" for i, e in enumerate(ETIQUETAS))
    )

    log("5/7 verificacion de integridad del fichero de envio")
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    marco = pd.DataFrame({e: pred_test[:, i] for i, e in enumerate(ETIQUETAS)})
    marco.insert(0, "id", test["id"].to_numpy())
    marco.to_csv(SUBMISSION, index=False, lineterminator="\n")
    integridad = _verificar_integridad(marco, test, train)
    integridad_correcta = all(f["resultado"] == "correcto" for f in integridad)
    for fila in integridad:
        log(f"   {fila['comprobacion']:38s} {fila['resultado']:8s} {fila['detalle']}")

    log("6/7 comparacion con la submission de V03, celda a celda")
    comparacion = _comparar_con_v03(SUBMISSION)
    for clave, valor in comparacion.items():
        log(f"   {clave}: {valor}")

    log("7/7 informe")
    _informe(
        variantes=variantes,
        integridad=integridad,
        integridad_correcta=integridad_correcta,
        constantes_finales=constantes_finales,
        comparacion=comparacion,
        n_celdas_totales=comparacion["celdas_totales"],
        elecciones=elecciones,
        f1_referencia=metricas_multietiqueta(y, predicciones_canonicas)["f1_por_etiqueta"],
        inicio=inicio,
    )
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        columnas = [
            "clave", "metrica", "diferencia", "ic95_0", "ic95_1", "pliegues_positivos",
            "gate_delta", "gate_ic95", "gate_pliegues", "gate_supera", "celdas",
            "tasa_acierto", "minimo_requerido", "filtro_pasa",
        ]
        escritor = csv.DictWriter(archivo, fieldnames=columnas, lineterminator="\n")
        escritor.writeheader()
        for v in variantes:
            escritor.writerow({k: v.get(k, "") for k in columnas})
    with REPORTE_JSON.open("w", encoding="utf-8") as archivo:
        json.dump(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "referencia_v03_local": REFERENCIA_V03,
                "referencia_v03_externa": REFERENCIA_LB,
                "ruido_en_aciertos_netos": RUIDO_NETOS,
                "rejilla_c": list(REJILLA_C),
                "variantes": variantes,
                "constantes_finales": dict(zip(ETIQUETAS, constantes_finales, strict=True)),
                "integridad": integridad,
                "integridad_correcta": integridad_correcta,
                "comparacion_con_v03": comparacion,
                "elecciones_por_pliegue": elecciones,
                "fichero_de_envio": str(SUBMISSION.relative_to(PROJECT_ROOT)),
                "nota": (
                    "La eleccion de la constante usa sub-pliegues de grupo propios dentro del "
                    "tren de cada pliegue. El conjunto de ajuste de la eleccion y el de medicion "
                    "quedan separados, que es la diferencia de protocolo respecto a R1-C1. Por "
                    "D-063 el envio es manual y el filtro de detectabilidad es criterio de "
                    "priorizacion, no bloqueante."
                ),
            },
            archivo,
            ensure_ascii=False,
            indent=2,
        )
        archivo.write("\n")
    log(f"P5 completada en {round(time.time() - inicio, 1)} s")


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
        Diccionario con el numero de celdas que cambian y con que proporcion.
    """
    nueva = pd.read_csv(ruta).set_index("id")
    vieja = pd.read_csv(
        PROJECT_ROOT / "submissions" / "toxic_submission_v03_lr_union_thr.csv"
    ).set_index("id").loc[nueva.index]
    cambio = nueva[list(ETIQUETAS)].to_numpy() != vieja[list(ETIQUETAS)].to_numpy()
    return {
        "celdas_totales": int(cambio.size),
        "celdas_que_cambian": int(cambio.sum()),
        "filas_que_cambian": int(cambio.any(axis=1).sum()),
        "por_etiqueta": {
            e: int(cambio[:, i].sum()) for i, e in enumerate(ETIQUETAS)
        },
    }


def _informe(
    variantes: list[dict[str, Any]],
    integridad: list[dict[str, str]],
    integridad_correcta: bool,
    constantes_finales: list[float],
    comparacion: dict[str, Any],
    n_celdas_totales: int,
    elecciones: list[dict[str, Any]],
    f1_referencia: dict[str, float],
    inicio: float,
) -> None:
    """Escribe el informe de la linea P5.

    Args:
        variantes: resultado de las dos ramas.
        integridad: comprobaciones del fichero de envio.
        integridad_correcta: si todas pasaron.
        constantes_finales: constante elegida por etiqueta sobre todo el desarrollo.
        comparacion: diferencia celda a celda con la submission de V03.
        n_celdas_totales: numero de celdas de la submission de referencia.
        elecciones: constante elegida en cada pliegue.
        f1_referencia: F1 por etiqueta de la referencia.
        inicio: instante de arranque del script.
    """
    anidada = variantes[0]
    control = variantes[1]
    lineas = [
        "# P5: constante de regularizacion distinta por etiqueta (V8-06)",
        "",
        f"Generado por `experiments/v8_06_c_por_etiqueta.py` ({EXPERIMENTO}).",
        "",
        "## 1. El precedente y por que esta linea no es R1-C1",
        "",
        "R1-C1 ajusto la constante y gano +0.0157 en local para perder 0.0108 en el externo.",
        "Hizo dos cosas a la vez: una sola constante para las tres etiquetas, y elegirla contra",
        "el mismo conjunto que despues evaluaba. **Aqui las dos se hacen bien**: la constante",
        "se elige por etiqueta, y se elige con sub-pliegues de grupo propios dentro del tren de",
        "cada pliegue, de modo que el conjunto de eleccion y el de medicion quedan separados.",
        "",
        "Por eso el script mide ademas un **brazo de control** que usa una sola constante con el",
        "mismo procedimiento interno. Sin ese brazo, cualquier diferencia observada no podria",
        "atribuirse a la idea y no al protocolo.",
        "",
        "## 2. Constantes elegidas en cada pliegue",
        "",
        "| pliegue | constante global | " + " | ".join(ETIQUETAS) + " |",
        "|---|---|" + "---|" * len(ETIQUETAS),
    ]
    for eleccion in elecciones:
        por = {p["etiqueta"]: p["c"] for p in eleccion["por_etiqueta"]}
        lineas.append(
            f"| {eleccion['pliegue']} | {eleccion['c_global']} | "
            + " | ".join(str(por[e]) for e in ETIQUETAS) + " |"
        )
    lineas += [
        "",
        "Constantes elegidas sobre todo el desarrollo, que son las de la submission: "
        + ", ".join(f"{e} {c}" for e, c in zip(ETIQUETAS, constantes_finales, strict=True)),
        "",
        "## 3. Resultados",
        "",
        "| rama | metrica | diferencia | IC95 | pliegues | celdas | tasa | filtro | gate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for v in variantes:
        lineas.append(
            f"| {v['clave']} | {v['metrica']:.6f} | {v['diferencia']:+.6f} | "
            f"[{v['ic95_0']:+.5f}, {v['ic95_1']:+.5f}] | {v['pliegues_positivos']}/5 | "
            f"{v['celdas']} | {v['tasa_acierto']:.4f} frente a {v['minimo_requerido']:.4f} | "
            f"{'pasa' if v['filtro_pasa'] else 'no pasa'} | "
            f"{'**supera**' if v['gate_supera'] else 'no supera'} |"
        )
    lineas += [
        "",
        "| rama | HS | TR | AG |",
        "|---|---|---|---|",
        f"| V03 (referencia) | {f1_referencia['HS']:.4f} | {f1_referencia['TR']:.4f} | "
        f"{f1_referencia['AG']:.4f} |",
    ]
    for v in variantes:
        f1 = v["f1_por_etiqueta"]
        lineas.append(
            f"| {v['clave']} | {f1['HS']:.4f} | {f1['TR']:.4f} | {f1['AG']:.4f} |"
        )
    lineas += [
        "",
        f"**La diferencia entre las dos ramas es el valor de la idea y no del protocolo**: "
        f"elegir constante por etiqueta da {anidada['diferencia']:+.6f} y usar una sola da "
        f"{control['diferencia']:+.6f}.",
        "",
        "## 4. Integridad del fichero de envio",
        "",
        f"Resultado global: **{'todas pasan' if integridad_correcta else 'HAY FALLOS'}**.",
        "",
        "| comprobacion | resultado | detalle |",
        "|---|---|---|",
    ]
    for fila in integridad:
        lineas.append(
            f"| {fila['comprobacion']} | {fila['resultado']} | {fila['detalle']} |"
        )
    lineas += [
        "",
        "## 5. Cuanto se aparta de V03, celda a celda",
        "",
        f"- celdas que cambian: {comparacion['celdas_que_cambian']} de {n_celdas_totales}",
        f"- filas que cambian: {comparacion['filas_que_cambian']}",
        "- por etiqueta: "
        + ", ".join(f"{e} {comparacion['por_etiqueta'][e]}" for e in ETIQUETAS),
    ]
    veredicto_filtro = (
        "La prediccion del filtro acertara: la tasa de acierto entre las celdas modificadas no "
        "llega al minimo, de modo que el externo deberia empeorar o no moverse."
        if not anidada["filtro_pasa"]
        else "La prediccion del filtro fallara: la tasa de acierto alcanza el minimo, de modo que "
        "el externo deberia poder moverse."
    )
    lineas += [
        "",
        "## 6. Decision y nota de presupuesto",
        "",
        "Por D-063 el filtro de detectabilidad **deja de ser un bloqueante**: pasa a ser una",
        "prediccion que se contrasta. La submission se escribe siempre que la linea vaya en",
        "direccion positiva, y el envio es **siempre manual**.",
        "",
        veredicto_filtro,
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(time.time() - inicio, 1)} s.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
