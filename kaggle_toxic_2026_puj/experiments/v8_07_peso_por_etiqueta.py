"""P6 de la fase v8 (V8-07): peso de clase distinto por etiqueta.

Contexto
--------
P5 mostro que elegir la constante por etiqueta vale +0,0023 sobre elegir una sola, y que la
tasa de acierto entre las celdas modificadas subio a 0,5185, **la primera vez por encima del
azar en la fase**. Eso rompe el patron de P1, P3 y P4, en el que las tres lineas movian la
decision hacia peor. P6 es el paso siguiente en la misma direccion: si el optimo de
regularizacion difiere por etiqueta, es razonable que el **balanceo optimo** tambien.

Que se busca
------------
Con ``class_weight='balanced'`` el peso de cada clase es el cociente de filas entre el doble
de los positivos de esa etiqueta. Eso da un balanceo por etiqueta ya, pero el mismo criterio
para las tres. Lo que aqui se busca es un **multiplicador por etiqueta** sobre ese peso
equilibrado, elegido con validacion anidada dentro del tren de cada pliegue, con un intervalo
alrededor de uno para poder apartarse del balanceo si el dato lo pide.

Los brazos
----------
- ``anidado_por_etiqueta``: un multiplicador por etiqueta, elegido con sub-pliegues internos.
- ``control_global``: un unico multiplicador para las tres, con el mismo procedimiento. Es el
  criterio de R2-MINDF5 y MOD-01 sin su sobreajuste, y sirve para atribuir la diferencia a la
  idea y no al protocolo.
- ``sin_equilibrar``: ninguna ponderacion. Es el extremo del multiplicador, y muestra cuanto
  aporta el balanceo en si.

Presupuesto de sondas
---------------------
Por D-063 el usuario autoriza gastar envios adicionales y el filtro de detectabilidad pasa a
ser criterio de priorizacion. **La submission se escribe siempre que la linea vaya en
direccion positiva, y el envio es siempre manual.**

Uso
---
    /home/carlos/venv/bin/python experiments/v8_07_peso_por_etiqueta.py
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

EXPERIMENTO = "V8-07 (P6)"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "v8_peso_por_etiqueta.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v8_peso_por_etiqueta.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v8_peso_por_etiqueta.json"
SUBMISSION = PROJECT_ROOT / "submissions" / "toxic_submission_v10_p6_peso_por_etiqueta.csv"

REFERENCIA_V03 = 0.777211
REFERENCIA_LB = 0.78306
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
TOLERANCIA_CONTROL = 0.0005
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
RUIDO_NETOS = 19.2
CONSTANTE_V03 = 2.0
# Multiplicadores sobre el peso equilibrado. El uno es el balanceo de V03.
REJILLA_PESO = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
PLEIGUES_INTERNOS = 3
SEMILLA_INTERNA = 778

PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V8-P6] {mensaje}", flush=True)


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


def pesos_por_clase(n_filas: int, destino: np.ndarray, multiplicador: float) -> np.ndarray:
    """Construye los pesos de clase a partir del balanceo y un multiplicador.

    Args:
        n_filas: numero de filas del ajuste.
        destino: vector binario de la etiqueta.
        multiplicador: factor sobre el peso equilibrado.

    Returns:
        Vector de pesos por fila, con el peso de los positivos multiplicado.
    """
    positivos = max(int(destino.sum()), 1)
    negativos = max(n_filas - int(destino.sum()), 1)
    peso_positivo = multiplicador * n_filas / (2.0 * positivos)
    peso_negativo = n_filas / (2.0 * negativos)
    return np.where(destino == 1, peso_positivo, peso_negativo)


def ajustar_probabilidades(
    matriz: sparse.csr_matrix,
    destino: np.ndarray,
    consulta: sparse.csr_matrix,
    multiplicador: float,
    constante: float = CONSTANTE_V03,
) -> np.ndarray:
    """Ajusta un modelo con el peso de clase dado y devuelve probabilidades de la consulta.

    Args:
        matriz: matriz de caracteristicas de ajuste.
        destino: vector binario de la etiqueta.
        consulta: matriz de caracteristicas de la consulta.
        multiplicador: factor sobre el peso equilibrado.
        constante: constante de regularizacion.

    Returns:
        Vector de probabilidades de la clase positiva.
    """
    estimador = LogisticRegression(
        C=constante, max_iter=2000, solver="liblinear", random_state=RANDOM_STATE
    )
    estimador.fit(matriz, destino, sample_weight=pesos_por_clase(matriz.shape[0], destino, multiplicador))
    return estimador.predict_proba(consulta)[:, 1]


def f1_de(probabilidades: np.ndarray, verdadero: np.ndarray, etiqueta: int) -> float:
    """Calcula la F1 de una etiqueta con su umbral congelado.

    Args:
        probabilidades: vector de probabilidades.
        verdadero: vector binario verdadero.
        etiqueta: indice de la etiqueta, para su umbral.

    Returns:
        Valor de la F1, cero si el denominador es cero.
    """
    pred = (probabilidades >= UMBRALES_V03[etiqueta]).astype(int)
    tp = int(((pred == 1) & (verdadero == 1)).sum())
    fp = int(((pred == 1) & (verdadero == 0)).sum())
    fn = int(((pred == 0) & (verdadero == 1)).sum())
    divisor = 2 * tp + fp + fn
    return (2.0 * tp / divisor) if divisor else 0.0


def main() -> None:
    """Ejecuta la eleccion anidada del peso de clase por etiqueta y la evalua."""
    inicio = time.time()
    log("P6: peso de clase distinto por etiqueta")
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
    log(f"   desarrollo {len(y)} filas | prueba {len(test)} filas | rejilla {REJILLA_PESO}")

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
    log(f"   OOF canonico {metrica_canonica:.6f} | ids en orden {ids_iguales}")
    if not control_correcto:
        log("PARADA: el control falla y no se sigue.")
        return

    log("2/7 eleccion anidada del multiplicador de peso, por etiqueta y por pliegue")
    oof_anidado = np.zeros((len(y), 3), dtype=np.float64)
    oof_control = np.zeros((len(y), 3), dtype=np.float64)
    oof_sin_equilibrar = np.zeros((len(y), 3), dtype=np.float64)
    elecciones: list[dict[str, Any]] = []
    for numero, (entrenamiento, validacion) in enumerate(pliegues):
        matriz, (consulta_val,) = construir_espacio(
            textos.iloc[entrenamiento], [textos.iloc[validacion]]
        )
        y_entrenamiento = y[entrenamiento]
        interno = StratifiedGroupKFold(
            n_splits=PLEIGUES_INTERNOS, shuffle=True, random_state=SEMILLA_INTERNA
        )
        particiones = list(
            interno.split(
                matriz, combinaciones(desarrollo.iloc[entrenamiento]),
                groups=grupos_dev[entrenamiento],
            )
        )
        mejor_global = {"f1": -1.0, "m": 1.0}
        por_etiqueta: list[dict[str, Any]] = []
        for etiqueta in range(3):
            mejor = {"f1": -1.0, "m": 1.0}
            for multiplicador in REJILLA_PESO:
                acumulado = 0.0
                for indices_t, indices_v in particiones:
                    proba = ajustar_probabilidades(
                        matriz[indices_t], y_entrenamiento[indices_t, etiqueta],
                        matriz[indices_v], multiplicador,
                    )
                    media = f1_de(proba, y_entrenamiento[indices_v, etiqueta], etiqueta)
                    acumulado += media
                    if media > mejor_global["f1"]:
                        mejor_global = {"f1": media, "m": multiplicador}
                    if media > mejor["f1"]:
                        mejor = {"f1": media, "m": multiplicador}
            por_etiqueta.append(
                {"etiqueta": ETIQUETAS[etiqueta], "multiplicador": mejor["m"],
                 "f1_interna": round(mejor["f1"], 6)}
            )
            oof_anidado[validacion, etiqueta] = ajustar_probabilidades(
                matriz, y_entrenamiento[:, etiqueta], consulta_val, mejor["m"]
            )
        for etiqueta in range(3):
            oof_control[validacion, etiqueta] = ajustar_probabilidades(
                matriz, y_entrenamiento[:, etiqueta], consulta_val, mejor_global["m"]
            )
            oof_sin_equilibrar[validacion, etiqueta] = ajustar_probabilidades(
                matriz, y_entrenamiento[:, etiqueta], consulta_val, 1e-9
            )
        elecciones.append(
            {
                "pliegue": numero + 1,
                "multiplicador_global": mejor_global["m"],
                "por_etiqueta": por_etiqueta,
            }
        )
        log(
            f"   pliegue {numero + 1}: multiplicador global {mejor_global['m']} | "
            + ", ".join(f"{p['etiqueta']} {p['multiplicador']}" for p in por_etiqueta)
        )

    log("3/7 evaluacion de las tres ramas")
    variantes: list[dict[str, Any]] = []
    for clave, probabilidades, descripcion in (
        ("anidado_por_etiqueta", oof_anidado,
         "multiplicador de peso por etiqueta, elegido con validacion anidada"),
        ("control_global", oof_control,
         "un unico multiplicador para las tres, con el mismo procedimiento interno"),
        ("sin_equilibrar", oof_sin_equilibrar,
         "ninguna ponderacion de clase, el extremo del multiplicador"),
    ):
        predicciones = aplicar_umbrales(probabilidades)
        metrica = metrica_macro(y, predicciones)
        diferencia = metrica - metrica_canonica
        aciertos = [
            int((predicciones[val] == y[val]).sum()
                - (predicciones_canonicas[val] == y[val]).sum())
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
            "clave": clave, "descripcion": descripcion,
            "metrica": round(metrica, 6), "diferencia": round(diferencia, 6),
            "ic95_0": float(bootstrap["ic95"][0]), "ic95_1": float(bootstrap["ic95"][1]),
            "pliegues_positivos": int(sum(1 for a in aciertos if a > 0)),
            "f1_por_etiqueta": metricas_multietiqueta(y, predicciones)["f1_por_etiqueta"],
            "celdas": n_celdas, "tasa_acierto": round(tasa, 4),
            "minimo_requerido": round(minimo, 4), "filtro_pasa": pasa,
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
    generador = StratifiedGroupKFold(
        n_splits=PLEIGUES_INTERNOS, shuffle=True, random_state=SEMILLA_INTERNA
    )
    particiones = list(
        generador.split(matriz_completa, combinaciones(desarrollo), groups=grupos_dev)
    )
    multiplicadores_finales: list[float] = []
    for etiqueta in range(3):
        mejor = {"f1": -1.0, "m": 1.0}
        for multiplicador in REJILLA_PESO:
            acumulado = 0.0
            for indices_t, indices_v in particiones:
                proba = ajustar_probabilidades(
                    matriz_completa[indices_t], y[indices_t, etiqueta],
                    matriz_completa[indices_v], multiplicador,
                )
                acumulado += f1_de(proba, y[indices_v, etiqueta], etiqueta)
            media = acumulado / len(particiones)
            if media > mejor["f1"]:
                mejor = {"f1": media, "m": multiplicador}
        multiplicadores_finales.append(mejor["m"])
    log(f"   multiplicadores finales por etiqueta: {multiplicadores_finales}")
    proba_test = np.zeros((len(test), 3), dtype=np.float64)
    for etiqueta in range(3):
        proba_test[:, etiqueta] = ajustar_probabilidades(
            matriz_completa, y[:, etiqueta], consulta_prueba, multiplicadores_finales[etiqueta]
        )
    pred_test = aplicar_umbrales(proba_test)
    log("   tasa de positivos predicha: "
        + ", ".join(f"{e} {pred_test[:, i].mean():.4f}" for i, e in enumerate(ETIQUETAS)))

    log("5/7 verificacion de integridad del fichero de envio")
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    marco = pd.DataFrame({e: pred_test[:, i] for i, e in enumerate(ETIQUETAS)})
    marco.insert(0, "id", test["id"].to_numpy())
    marco.to_csv(SUBMISSION, index=False, lineterminator="\n")
    integridad = _verificar_integridad(marco, test, train)
    integridad_correcta = all(f["resultado"] == "correcto" for f in integridad)
    for fila in integridad:
        log(f"   {fila['comprobacion']:38s} {fila['resultado']:8s} {fila['detalle']}")

    log("6/7 comparacion con la submission de V03")
    comparacion = _comparar_con_v03(SUBMISSION)
    log(f"   celdas que cambian: {comparacion['celdas_que_cambian']} de "
        f"{comparacion['celdas_totales']} | filas {comparacion['filas_que_cambian']} | "
        + ", ".join(f"{e} {comparacion['por_etiqueta'][e]}" for e in ETIQUETAS))

    log("7/7 informe")
    _informe(
        variantes=variantes, integridad=integridad, integridad_correcta=integridad_correcta,
        multiplicadores_finales=multiplicadores_finales, comparacion=comparacion,
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
                "rejilla_peso": list(REJILLA_PESO),
                "variantes": variantes,
                "multiplicadores_finales": dict(
                    zip(ETIQUETAS, multiplicadores_finales, strict=True)
                ),
                "integridad": integridad, "integridad_correcta": integridad_correcta,
                "comparacion_con_v03": comparacion, "elecciones_por_pliegue": elecciones,
                "fichero_de_envio": str(SUBMISSION.relative_to(PROJECT_ROOT)),
                "nota": (
                    "El multiplicador se aplica sobre el peso de clase equilibrado, no en su "
                    "lugar, de modo que el valor uno reproduce exactamente el balanceo de V03. "
                    "La eleccion usa sub-pliegues de grupo propios dentro del tren de cada "
                    "pliegue. Por D-063 el envio es manual y el filtro es priorizacion."
                ),
            },
            archivo, ensure_ascii=False, indent=2,
        )
        archivo.write("\n")
    log(f"P6 completada en {round(time.time() - inicio, 1)} s")


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
        "celdas_totales": int(cambio.size), "celdas_que_cambian": int(cambio.sum()),
        "filas_que_cambian": int(cambio.any(axis=1).sum()),
        "por_etiqueta": {e: int(cambio[:, i].sum()) for i, e in enumerate(ETIQUETAS)},
    }


def _informe(
    variantes: list[dict[str, Any]],
    integridad: list[dict[str, str]],
    integridad_correcta: bool,
    multiplicadores_finales: list[float],
    comparacion: dict[str, Any],
    elecciones: list[dict[str, Any]],
    f1_referencia: dict[str, float],
    inicio: float,
) -> None:
    """Escribe el informe de la linea P6.

    Args:
        variantes: resultado de las tres ramas.
        integridad: comprobaciones del fichero de envio.
        integridad_correcta: si todas pasaron.
        multiplicadores_finales: multiplicador elegido por etiqueta sobre todo el desarrollo.
        comparacion: diferencia celda a celda con la submission de V03.
        elecciones: multiplicador elegido en cada pliegue.
        f1_referencia: F1 por etiqueta de la referencia.
        inicio: instante de arranque del script.
    """
    anidada = variantes[0]
    control = variantes[1]
    sin_equilibrar = variantes[2]
    lineas = [
        "# P6: peso de clase distinto por etiqueta (V8-07)",
        "",
        f"Generado por `experiments/v8_07_peso_por_etiqueta.py` ({EXPERIMENTO}).",
        "",
        "## 1. Que se busca y por que es el paso siguiente a P5",
        "",
        "P5 mostro que elegir la constante por etiqueta vale +0,0023 sobre elegir una sola, y que",
        "la tasa de acierto entre las celdas modificadas subio a 0,5185, **la primera vez por",
        "encima del azar en la fase**. Eso rompe el patron de P1, P3 y P4, en el que las tres",
        "lineas movian la decision hacia peor. Si el optimo de regularizacion difiere por",
        "etiqueta, es razonable que el balanceo optimo tambien.",
        "",
        "Con el balanceo de V03 el peso de cada clase es el cociente de filas entre el doble de",
        "los positivos **de esa etiqueta**, de modo que ya hay un balanceo por etiqueta, pero",
        "con el mismo criterio para las tres. Aqui se busca un **multiplicador por etiqueta**",
        "sobre ese peso, con un intervalo alrededor de uno para poder apartarse del balanceo si",
        "el dato lo pide. El multiplicador se aplica **sobre** el peso equilibrado y no en su",
        "lugar, de modo que el valor uno reproduce exactamente el balanceo de V03.",
        "",
        "## 2. Multiplicadores elegidos en cada pliegue",
        "",
        "| pliegue | multiplicador global | " + " | ".join(ETIQUETAS) + " |",
        "|---|---|" + "---|" * len(ETIQUETAS),
    ]
    for eleccion in elecciones:
        por = {p["etiqueta"]: p["multiplicador"] for p in eleccion["por_etiqueta"]}
        lineas.append(
            f"| {eleccion['pliegue']} | {eleccion['multiplicador_global']} | "
            + " | ".join(str(por[e]) for e in ETIQUETAS) + " |"
        )
    lineas += [
        "",
        "Multiplicadores sobre todo el desarrollo, que son los de la submission: "
        + ", ".join(
            f"{e} {m}" for e, m in zip(ETIQUETAS, multiplicadores_finales, strict=True)
        ),
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
        f"**El valor de la idea, separandola del protocolo**, es "
        f"{anidada['diferencia'] - control['diferencia']:+.6f}: por etiqueta frente a un",
        "multiplicador unico. Y **cuanto aporta el balanceo en si** lo dice la rama sin",
        f"equilibrar, que queda en {sin_equilibrar['diferencia']:+.6f}.",
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
        "## 5. Cuanto se aparta de V03, celda a celda",
        "",
        f"- celdas que cambian: {comparacion['celdas_que_cambian']} de "
        f"{comparacion['celdas_totales']}, en {comparacion['filas_que_cambian']} filas",
        "- por etiqueta: "
        + ", ".join(f"{e} {comparacion['por_etiqueta'][e]}" for e in ETIQUETAS),
        "",
        "## 6. Decision y nota de presupuesto",
        "",
        "Por D-063 el filtro de detectabilidad deja de ser un bloqueante y la submission se",
        "escribe siempre que la linea vaya en direccion positiva. **El envio es manual.**",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(time.time() - inicio, 1)} s.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
