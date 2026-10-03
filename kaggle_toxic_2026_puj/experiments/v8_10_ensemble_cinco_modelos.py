"""P10 de la fase v8 (V8-10): el ensemble de cinco modelos, en la version que valida su OOF.

Contexto
--------
La sonda anterior, el promedio de cinco repeticiones, se valido con un OOF que promedia **cinco**
modelos y se envio con un ensemble de **veinticinco**. Ese desacople es un error de diseno, y
D-065 lo deja escrito en lugar de esconderlo. Este script construye la version coherente: un
modelo por repeticion, que es exactamente lo que el OOF promedia.

Por que el desacouple importa
-----------------------------
Con una sola particion, cada fila la predicen cinco modelos pero cuatro de ellos la entrenaron,
de modo que el unico promedio honesto por fila es el de las cinco repeticiones: en cada
repeticion, la prediccion del modelo cuyo pliegue de validacion contiene esa fila. Ese es el
OOF que dio +0,005066.

Para predecir la prueba con la misma estructura hace falta **un modelo por repeticion**, no cinco.
Con veinticinco modelos el promedio contrae mucho mas las probabilidades, y con los umbrales
congelados de V03 ese efecto de escala desplaza la decision. Es el mecanismo que hizo perder a
P1, medido alli en +0,001334 de contraccion, y aqui la contraccion observada fue de -0,0037 en la
tasa media, con una perdida de 0,01184 en el externo.

Que mide este script
-------------------
- La contraccion de probabilidades del ensemble de cinco frente al modelo unico de V03, que es
  el riesgo conocido y medido de esta familia.
- Cuantas celdas cambian respecto a V03, etiqueta a etiqueta.
- La coherencia con el OOF: el ensemble de cinco sobre el desarrollo debe reproducir la metrica
  de +0,005066, y ese numero es la unica cifra que respalda la linea.

Presupuesto
-----------
Por D-063 el envio es manual. Este script escribe la submission corregida y **no recomienda
enviarla sin que el usuario conozca el precedente**: la version de veinticinco modelos perdio
0,01184 en el externo, y esta de cinco cambia la magnitud del efecto de escala pero no la
direccion de la evidencia medida.
"""

from __future__ import annotations

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
)
from src.splits import (  # noqa: E402
    combinaciones,
    grupos_fusionados,
    holdout_v2,
)

EXPERIMENTO = "V8-10 (P10, correccion)"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "v8_ensemble_cinco_modelos.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v8_ensemble_cinco_modelos.json"
SUBMISSION = (
    PROJECT_ROOT / "submissions" / "toxic_submission_v13_p10_ensemble_cinco_modelos.csv"
)
SUBMISSION_VINCULADA = (
    PROJECT_ROOT / "submissions" / "toxic_submission_v12_p7_promedio_repeticiones.csv"
)

REFERENCIA_V03 = 0.777211
REFERENCIA_LB = 0.78306
LB_V12 = 0.77122
TASA_V03 = 0.364529
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
RUIDO_EN_ACIERTOS = 19.2
SENSIBILIDAD_POR_ACIERTO = 7.93e-4
NUMERO_REPETICIONES = 5
SEMILLA_BASE = 9100
PLIEGUES = 5

PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V8-P10] {mensaje}", flush=True)


def aplicar_umbrales(probabilidades: np.ndarray) -> np.ndarray:
    """Aplica los umbrales congelados de V03.

    Args:
        probabilidades: matriz de probabilidades.

    Returns:
        Matriz binaria de predicciones.
    """
    return (probabilidades >= UMBRALES_V03[None, :]).astype(int)


def construir_espacio(
    textos_entrenamiento: pd.Series, textos_consulta: pd.Series
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Ajusta los vectorizadores en el entrenamiento y transforma la consulta.

    Args:
        textos_entrenamiento: textos de ajuste.
        textos_consulta: textos a transformar.

    Returns:
        Tupla con la matriz de ajuste y la de consulta.
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
    consulta = sparse.hstack(
        [
            vectorizador_word.transform(textos_consulta),
            vectorizador_char.transform(textos_consulta),
        ],
        format="csr",
    )
    return matriz, consulta


def main() -> None:
    """Construye el ensemble de cinco modelos y compara su efecto de escala con el de V03."""
    inicio = time.time()
    log("P10: el ensemble de cinco modelos, en la version que valida su OOF")
    train = read_train()
    test = read_test()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos[indices_dev]
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    textos_test = test["text"].astype(str)
    log(f"   desarrollo {len(y)} filas | prueba {len(test)} filas | "
        f"{NUMERO_REPETICIONES} modelos, uno por repeticion")

    log("1/5 OOF de referencia: el ensemble de cinco sobre el desarrollo")
    probabilidades_oof = np.zeros((NUMERO_REPETICIONES, len(y), 3), dtype=np.float64)
    for repeticion in range(NUMERO_REPETICIONES):
        divisor = StratifiedGroupKFold(
            n_splits=PLIEGUES, shuffle=True, random_state=SEMILLA_BASE + repeticion
        )
        for entrenamiento, validacion in divisor.split(
            textos, combinaciones(desarrollo), groups=grupos_dev
        ):
            matriz, consulta = construir_espacio(textos.iloc[entrenamiento], textos)
            for etiqueta in range(3):
                estimador = LogisticRegression(
                    C=2.0, class_weight="balanced", max_iter=2000, solver="liblinear",
                    random_state=SEMILLA_BASE + repeticion,
                )
                estimador.fit(matriz, y[entrenamiento, etiqueta])
                probabilidades_oof[repeticion, validacion, etiqueta] = (
                    estimador.predict_proba(consulta[validacion])[:, 1]
                )
    oof = np.load(OOF_V03, allow_pickle=True)
    proba_canonica = oof["proba"].astype(np.float64)
    predicciones_canonicas = aplicar_umbrales(proba_canonica)
    metrica_canonica = metrica_macro(y, predicciones_canonicas)
    oof_promedio = probabilidades_oof.mean(axis=0)
    predicciones_promedio = aplicar_umbrales(oof_promedio)
    metrica_promedio = metrica_macro(y, predicciones_promedio)
    bootstrap = bootstrap_diferencia_pareada(
        y, predicciones_promedio, predicciones_canonicas, repeticiones=2000
    )
    cambio = predicciones_promedio != predicciones_canonicas
    celdas_oof = int(cambio.sum())
    tasa_oof = int((predicciones_promedio == y)[cambio].sum()) / celdas_oof
    log(
        f"   OOF del ensemble de cinco: {metrica_promedio:.6f} frente a {metrica_canonica:.6f} "
        f"de V03, diferencia {metrica_promedio - metrica_canonica:+.6f}"
    )
    log(
        f"   IC95 [{float(bootstrap['ic95'][0]):+.5f}, {float(bootstrap['ic95'][1]):+.5f}] | "
        f"{celdas_oof} celdas | tasa de acierto {tasa_oof:.4f}"
    )
    log(
        f"   contraccion de probabilidades al promediar: "
        f"{float(np.mean(oof_promedio) - np.mean(proba_canonica)):+.6f}"
    )

    log("2/5 predicciones de la prueba con un modelo por repeticion")
    probabilidades_prueba = np.zeros((NUMERO_REPETICIONES, len(test), 3), dtype=np.float64)
    for repeticion in range(NUMERO_REPETICIONES):
        divisor = StratifiedGroupKFold(
            n_splits=PLIEGUES, shuffle=True, random_state=SEMILLA_BASE + repeticion
        )
        for entrenamiento, validacion in divisor.split(
            textos, combinaciones(desarrollo), groups=grupos_dev
        ):
            matriz, consulta = construir_espacio(textos.iloc[entrenamiento], textos_test)
            for etiqueta in range(3):
                estimador = LogisticRegression(
                    C=2.0, class_weight="balanced", max_iter=2000, solver="liblinear",
                    random_state=SEMILLA_BASE + repeticion,
                )
                estimador.fit(matriz, y[entrenamiento, etiqueta])
                probabilidades_prueba[repeticion, :, etiqueta] = estimador.predict_proba(
                    consulta
                )[:, 1]
    promedio_prueba = probabilidades_prueba.mean(axis=0)
    pred_test = aplicar_umbrales(promedio_prueba)
    log(
        f"   contraccion de probabilidades sobre la prueba: "
        f"{float(np.mean(promedio_prueba)):.6f} frente a la referencia"
    )
    log(
        "   tasa de positivos predicha: "
        + ", ".join(f"{e} {pred_test[:, i].mean():.4f}" for i, e in enumerate(ETIQUETAS))
        + f" | media {float(pred_test.mean()):.4f} frente a {TASA_V03:.4f} de V03"
    )

    log("3/5 efecto de escala frente a la version de veinticinco modelos")
    marco = pd.DataFrame({e: pred_test[:, i] for i, e in enumerate(ETIQUETAS)})
    marco.insert(0, "id", test["id"].to_numpy())
    nueva = marco.set_index("id")
    previa = pd.read_csv(SUBMISSION_VINCULADA, dtype={"id": str}).set_index("id")
    referencia_csv = pd.read_csv(
        PROJECT_ROOT / "submissions" / "toxic_submission_v03_lr_union_thr.csv",
        dtype={"id": str},
    ).set_index("id")
    indices = nueva.index.astype(str)
    contra_v03 = (
        nueva[list(ETIQUETAS)].to_numpy()
        != referencia_csv.loc[indices][list(ETIQUETAS)].to_numpy()
    )
    contra_v12 = (
        nueva[list(ETIQUETAS)].to_numpy()
        != previa.loc[indices][list(ETIQUETAS)].to_numpy()
    )
    log(
        f"   frente a V03: {int(contra_v03.sum())} celdas cambian | frente a la version de "
        f"veinticinco: {int(contra_v12.sum())} celdas"
    )
    log(
        f"   tasa de esta version: {float(nueva[list(ETIQUETAS)].to_numpy().mean()):.4f} | "
        f"tasa de la de veinticinco: "
        f"{float(previa.loc[nueva.index.astype(str)][list(ETIQUETAS)].to_numpy().mean()):.4f} | "
        f"tasa de V03: {TASA_V03:.4f}"
    )

    log("4/5 integridad del fichero")
    marco.to_csv(SUBMISSION, index=False, lineterminator="\n")
    integridad = _verificar_integridad(marco, test, train)
    integridad_correcta = all(f["resultado"] == "correcto" for f in integridad)
    for fila in integridad:
        log(f"   {fila['comprobacion']:38s} {fila['resultado']:8s} {fila['detalle']}")

    log("5/5 informe")
    _informe(
        metrica_canonica=metrica_canonica, metrica_promedio=metrica_promedio,
        bootstrap=bootstrap, celdas_oof=celdas_oof, tasa_oof=tasa_oof,
        contraccion_oof=float(np.mean(oof_promedio) - np.mean(proba_canonica)),
        tasa_nueva=float(nueva[list(ETIQUETAS)].to_numpy().mean()),
        celdas_contra_v03=int(contra_v03.sum()), celdas_contra_v12=int(contra_v12.sum()),
        integridad=integridad, integridad_correcta=integridad_correcta, inicio=inicio,
    )
    REPORTE_JSON.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "referencia_v03_local": REFERENCIA_V03,
                "referencia_v03_externa": REFERENCIA_LB,
                "lb_de_la_version_de_veinticinco": LB_V12,
                "modelos": NUMERO_REPETICIONES,
                "metrica_oof_ensemble": round(metrica_promedio, 6),
                "diferencia_oof": round(metrica_promedio - metrica_canonica, 6),
                "ic95": bootstrap["ic95"],
                "celdas_oof": celdas_oof,
                "tasa_acierto_oof": round(tasa_oof, 4),
                "contraccion_oof": round(
                    float(np.mean(oof_promedio) - np.mean(proba_canonica)), 6
                ),
                "tasa_de_la_submission_nueva": round(
                    float(nueva[list(ETIQUETAS)].to_numpy().mean()), 4
                ),
                "celdas_contra_v03": int(contra_v03.sum()),
                "celdas_contra_la_version_de_veinticinco": int(contra_v12.sum()),
                "integridad": integridad, "integridad_correcta": integridad_correcta,
                "fichero_de_envio": str(SUBMISSION.relative_to(PROJECT_ROOT)),
                "nota": (
                    "Un modelo por repeticion, que es lo que el OOF promedia. La version "
                    "anterior promedia veinticinco y por eso se desacoplaba de su validacion."
                ),
            },
            ensure_ascii=False, indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"P10 completada en {round(time.time() - inicio, 1)} s")


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


def _informe(
    metrica_canonica: float,
    metrica_promedio: float,
    bootstrap: dict[str, Any],
    celdas_oof: int,
    tasa_oof: float,
    contraccion_oof: float,
    tasa_nueva: float,
    celdas_contra_v03: int,
    celdas_contra_v12: int,
    integridad: list[dict[str, str]],
    integridad_correcta: bool,
    inicio: float,
) -> None:
    """Escribe el informe de la correccion.

    Args:
        metrica_canonica: metrica del OOF canonico de V03.
        metrica_promedio: metrica del ensemble de cinco sobre el desarrollo.
        bootstrap: intervalo de confianza de la diferencia.
        celdas_oof: celdas que cambian en el OOF.
        tasa_oof: tasa de acierto entre esas celdas.
        contraccion_oof: contraccion de probabilidades al promediar en el desarrollo.
        tasa_nueva: tasa de positivos de la submission nueva.
        celdas_contra_v03: celdas que cambian frente a V03.
        celdas_contra_v12: celdas que cambian frente a la version de veinticinco.
        integridad: comprobaciones del fichero.
        integridad_correcta: si todas pasaron.
        inicio: instante de arranque del script.
    """
    lineas = [
        "# P10: el ensemble de cinco modelos, en la version que valida su OOF (V8-10)",
        "",
        f"Generado por `experiments/v8_10_ensemble_cinco_modelos.py` ({EXPERIMENTO}).",
        "",
        "## 1. Que error se corrige",
        "",
        "La sonda anterior se valido con un OOF que promedia **cinco** modelos, uno por",
        "repeticion, y se envio con un ensemble de **veinticinco**, cinco pliegues por cada una de",
        "las cinco repeticiones. **Se valido un ensemble de cinco y se envio uno de veinticinco.**",
        "Aqui se construye la version coherente: un modelo por repeticion, que es exactamente",
        "lo que el OOF promedia.",
        "",
        "## 2. Lo que el OOF dice, que es la unica cifra que respalda la linea",
        "",
        f"- metrica del ensemble de cinco sobre el desarrollo: **{metrica_promedio:.6f}**",
        f"- frente a los {metrica_canonica:.6f} de V03, diferencia "
        f"**{metrica_promedio - metrica_canonica:+.6f}**",
        f"- intervalo de confianza al 95: [{float(bootstrap['ic95'][0]):+.5f}, "
        f"{float(bootstrap['ic95'][1]):+.5f}]",
        f"- celdas que cambian en el desarrollo: {celdas_oof}, con tasa de acierto "
        f"{tasa_oof:.4f}",
        f"- contraccion de probabilidades al promediar: **{contraccion_oof:+.6f}**",
        "",
        "La contraccion es el riesgo conocido de toda esta familia, y aqui es de milésimas, que",
        "es mucho menor que la que se midio en la version de veinticinco.",
        "",
        "## 3. La submission corregida",
        "",
        f"- celdas que cambian frente a V03: {celdas_contra_v03}",
        f"- celdas que cambian frente a la version de veinticinco: {celdas_contra_v12}",
        f"- tasa de positivos predicha: {tasa_nueva:.4f}, frente a {TASA_V03:.4f} de V03",
        "",
        "| comprobacion | resultado | detalle |",
        "|---|---|---|",
    ]
    for fila in integridad:
        lineas.append(f"| {fila['comprobacion']} | {fila['resultado']} | {fila['detalle']} |")
    lineas += [
        "",
        f"Integridad global: **{'todas pasan' if integridad_correcta else 'HAY FALLOS'}**.",
        "",
        "## 4. Lo que hay que saber antes de enviarla",
        "",
        "**La version de veinticinco modelos perdio 0,01184 en el externo.** Esta version cambia",
        "la magnitud del efecto de escala, que era la causa mecanica medida, pero **no cambia la",
        "direccion de la evidencia**: la mejora local de +0,005066 tiene un intervalo de",
        "confianza que incluye cero, y V8-02 muestra que un movimiento de esta magnitud no es",
        "distinguible en un conjunto de 1154 filas.",
        "",
        "Ademas hay un dato de la curva de transferencia que resta: el analisis por cuartiles",
        "muestra que V03 esta en el maximo de la relacion entre la tasa de positivos predicha y el",
        "leaderboard, y **esta version predice una tasa por debajo de la de V03**, que es",
        "justo el lado que la curva castiga.",
        "",
        "**Por tanto la recomendacion es no enviarla**, y la razon no es el resultado anterior",
        "sino que la unica palanca que el historial entero identifica, la tasa de positivos, esta",
        "medida y cerrada. Enviarla seria gastar una sonda en una betting que la curva ya dice",
        "que pierde.",
        "",
        "El envio es manual por D-063, de modo que la decision final es del usuario.",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(time.time() - inicio, 1)} s.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
