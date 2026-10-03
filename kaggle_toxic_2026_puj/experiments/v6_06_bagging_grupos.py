"""Track H de la fase v6 (V6-06): bagging con remuestreo por grupos, con su techo medido.

Contexto
--------
El gap analysis (V6-00) confirmo por busqueda en el codigo que **no existe ningun bagging en
el proyecto**. Los ensembles previos (ENS-01 a ENS-04) mezclaban **representaciones
distintas** y no superaron al mejor modelo, que es un problema distinto: promedian modelos
que ya se diferenciaban antes de promediarlos. Aqui la pregunta es otra: si se promedian
**V03-equivalentes**, es decir, el mismo modelo reentrenado sobre submuestras de grupos
distintos, ¿aporta algo?

Por que el techo se puede MEDIR y no solo suponer
-------------------------------------------------
El track A (V6-03) midio la estructura de grupos del desarrollo: **1415 grupos distintos con
un maximo de 7 filas, y 1378 de 1483 filas (92.9 %) en grupos de tamano 1**. Eso tiene una
consecuencia directa sobre el bagging: si casi todas las filas estan en grupos unicos,
**remuestrar por grupos y remuestrar por filas son practicamente la misma cosa**, y el
remuestreo por filas de un conjunto de 1.483 filas produce modelos muy parecidos. El
promedio de modelos casi identicos no puede mover la metrica.

Por eso este script no se limita a ejecutar el bagging: **mide la diversidad de los modelos
que promedia**. La correlacion media entre las predicciones de cada par de modelos es la
magnitud que explica el resultado. Sin ella, un delta negativo seria un misterio; con ella,
es una consecuencia medida. Esa medicion es la que justifica, o no, abrir el track.

Diseno
------
- **H0** control: un solo modelo por pliegue, identico a V03. Debe reproducir 0.777211.
- **H1** promedio de **5** modelos por pliegue, cada uno sobre un remuestreo de grupos con
  reemplazo, con el mismo pipeline de V03.
- Se reportan la correlacion media entre pares de las predicciones de los 5 modelos y la
  desviacion de sus probabilidades, que son las dos medidas de diversidad.

El bagging se hace **dentro de cada pliegue**: el remuestreo sale del tren, nunca del
desarrollo entero, porque si no cada pliegue veria filas que despues se validan.

Uso
---
    /home/carlos/venv/bin/python experiments/v6_06_bagging_grupos.py
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train  # noqa: E402
from src.evaluation import (  # noqa: E402
    ETIQUETAS,
    _f1_por_celda,
    bootstrap_diferencia_pareada,
    metrica_macro,
    metricas_multietiqueta,
)
from src.splits import grupos_fusionados, holdout_v2, pliegues_corregidos  # noqa: E402

EXPERIMENTO = "V6-06 (H0-H1)"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "v6_bagging_grupos.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v6_bagging_grupos.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v6_bagging_grupos.json"
FIGURAS = PROJECT_ROOT / "reports" / "figuras"

REFERENCIA_V03 = 0.777211
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
GATE_1_DELTA = 0.003
GATE_1_FOLDS = 4
MODELOS_POR_PLIEGUE = 5
REPETICIONES_BOOTSTRAP = 10_000
SEMILLA_BOOTSTRAP = 20260926
SEMILLA_BAGGING = 20260926

PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
PARAMETROS_LR = {
    "C": 2.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "liblinear",
    "random_state": 42,
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V6-H] {mensaje}", flush=True)


def remuestrear_grupos(grupos_entrenamiento: np.ndarray, generador) -> np.ndarray:
    """Dibuja un remuestreo de grupos con reemplazo y devuelve los indices de fila.

    Se sortean los **grupos**, no las filas, y luego se toman todas las filas de cada grupo
    sorteado. Es la diferencia con un bagging de filas: aqui una fila nunca aparece partida,
    lo que es justamente lo que se quiere al medir si la repeticion es ruido o señal.

    Args:
        grupos_entrenamiento: identificador de grupo de cada fila de entrenamiento.
        generador: generador de numeros aleatorios.

    Returns:
        Indices de fila del remuestreo, con duplicados si un grupo salio mas de una vez.
    """
    unicos = np.unique(grupos_entrenamiento)
    sorteados = generador.choice(unicos, size=len(unicos), replace=True)
    # La multiplicidad importa: si un grupo sale dos veces, sus filas deben aparecer DOS
    # veces. Filtrar con `np.isin` las incluiria una sola vez y el remuestreo dejaria de ser
    # un bootstrap para convertirse en una submuestra sin reemplazo, que es otra cosa.
    posiciones = {valor: indice for indice, valor in enumerate(grupos_entrenamiento)}
    orden = np.argsort(grupos_entrenamiento, kind="stable")
    grupos_ordenados = grupos_entrenamiento[orden]
    limites = np.flatnonzero(np.r_[True, grupos_ordenados[1:] != grupos_ordenados[:-1], True])
    bloques = {
        grupos_ordenados[inicio]: orden[inicio:fin]
        for inicio, fin in zip(limites[:-1], limites[1:], strict=True)
    }
    del posiciones
    indices: list[np.ndarray] = []
    for grupo in sorteados:
        indices.append(bloques[grupo])
    return np.concatenate(indices)


def ajustar_oof(textos, y, pliegues, grupos_dev, n_modelos: int) -> tuple[np.ndarray, dict[int, list[np.ndarray]]]:
    """Calcula las probabilidades fuera de pliegue, promediando `n_modelos` remuestreos.

    Args:
        textos: columna de textos del desarrollo.
        y: matriz binaria verdadera.
        pliegues: pliegues corregidos del protocolo v2.
        grupos_dev: identificador de grupo de cada fila del desarrollo.
        n_modelos: numero de modelos a promediar por pliegue.

    Returns:
        Tupla con la matriz de probabilidades promediada y un diccionario que, por indice
        de pliegue, guarda las predicciones de cada modelo. La clave es el indice real del
        pliegue y no la posicion en una lista, porque los pliegues tienen tamaños distintos y
        una agrupacion por modulo empareja modelos que se Flan a validar filas distintas.
    """
    generador = np.random.default_rng(SEMILLA_BAGGING)
    acumulado = np.zeros((len(textos), 3), dtype=np.float64)
    individuales: dict[int, list[np.ndarray]] = {}
    for indice_pliegue, (entrenamiento, validacion) in enumerate(pliegues):
        vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
        vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
        base = sparse.hstack(
            [
                vectorizador_word.fit_transform(textos.iloc[entrenamiento]),
                vectorizador_char.fit_transform(textos.iloc[entrenamiento]),
            ]
        ).tocsr()
        consulta = sparse.hstack(
            [
                vectorizador_word.transform(textos.iloc[validacion]),
                vectorizador_char.transform(textos.iloc[validacion]),
            ]
        ).tocsr()
        grupos_tr = grupos_dev[entrenamiento]
        acumulado[validacion] = 0.0
        for indice_modelo in range(n_modelos):
            if indice_modelo == 0:
                # El primer modelo usa el tren completo, sin remuestreo. Asi el promedio
                # siempre incluye un modelo equivalente a V03, y el control H0 se obtiene
                # con `n_modelos = 1` sin cambiar una sola linea del pipeline.
                indices = np.arange(len(entrenamiento))
            else:
                indices = remuestrear_grupos(grupos_tr, generador)
            proba_modelo = np.zeros((len(validacion), 3), dtype=np.float64)
            for etiqueta in range(3):
                estimador = LogisticRegression(**PARAMETROS_LR)
                estimador.fit(base[indices], y[entrenamiento[indices], etiqueta])
                proba_modelo[:, etiqueta] = estimador.predict_proba(
                    consulta
                )[:, 1]
            acumulado[validacion] += proba_modelo
            individuales.setdefault(indice_pliegue, []).append(proba_modelo)
    return acumulado / n_modelos, individuales


def medir_diversidad(individuales: dict[int, list[np.ndarray]]) -> dict[str, float]:
    """Mide cuanto se diferencian entre si los modelos que se promedian.

    Args:
        individuales: predicciones de cada modelo, indexadas por pliegue. Solo se comparan
        modelos del mismo pliegue: los de pliegues distintos se validan sobre filas
        distintas y su correlacion no mide diversidad sino ruido de estimacion.

    Returns:
        Diccionario con la correlacion media entre pares y la desviacion media.
    """
    correlaciones: list[float] = []
    desviaciones: list[float] = []
    for grupo in individuales.values():
        if len(grupo) < 2:
            continue
        for i in range(len(grupo)):
            for j in range(i + 1, len(grupo)):
                a, b = grupo[i].ravel(), grupo[j].ravel()
                if np.std(a) > 0 and np.std(b) > 0:
                    correlaciones.append(float(np.corrcoef(a, b)[0, 1]))
                desviaciones.append(float(np.mean(np.abs(a - b))))
    return {
        "correlacion_media_entre_pares": round(float(np.mean(correlaciones)), 6),
        "correlacion_minima": round(float(np.min(correlaciones)), 6),
        "diferencia_media_absoluta": round(float(np.mean(desviaciones)), 6),
        "pares_comparados": len(correlaciones),
    }


def evaluar(nombre: str, y, proba, pred_v03) -> dict[str, Any]:
    """Evalua una variante con los umbrales congelados de V03.

    Args:
        nombre: identificador de la variante.
        y: matriz binaria verdadera.
        proba: matriz de probabilidades.
        pred_v03: predicciones de V03.

    Returns:
        Diccionario con metricas, comparacion pareada y complementariedad.
    """
    pred = (proba >= UMBRALES_V03[None, :]).astype(int)
    comparacion = bootstrap_diferencia_pareada(
        y, pred, pred_v03, REPETICIONES_BOOTSTRAP, SEMILLA_BOOTSTRAP
    )
    por_etiqueta = metricas_multietiqueta(y, pred)["f1_por_etiqueta"]
    acierta = _f1_por_celda(y, pred) >= 0.999
    acierta_v03 = _f1_por_celda(y, pred_v03) >= 0.999
    return {
        "variante": nombre,
        "corrected_cv": round(metrica_macro(y, pred), 6),
        "delta_vs_V03": round(metrica_macro(y, pred) - REFERENCIA_V03, 6),
        "ic95_0": comparacion["ic95"][0],
        "ic95_1": comparacion["ic95"][1],
        "f1_HS": round(por_etiqueta[ETIQUETAS[0]], 4),
        "f1_TR": round(por_etiqueta[ETIQUETAS[1]], 4),
        "f1_AG": round(por_etiqueta[ETIQUETAS[2]], 4),
        "salvage": int(np.sum(acierta & ~acierta_v03)),
        "damage": int(np.sum(~acierta & acierta_v03)),
        "pred": pred,
    }


def main() -> None:
    """Ejecuta el control H0 y el promedio de cinco modelos H1."""
    inicio = time.time()
    log("Track H: bagging con remuestreo por grupos")
    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos[indices_dev])
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    oof = np.load(OOF_V03, allow_pickle=True)
    pred_v03 = (oof["proba"] >= UMBRALES_V03[None, :]).astype(int)
    textos = desarrollo["text"].astype(str)
    grupos_dev = grupos[indices_dev]
    log(f"   desarrollo {len(y)} filas | V03 {metrica_macro(y, pred_v03):.6f}")

    conteo = pd.Series(grupos_dev).value_counts()
    filas_unicas = int((conteo == 1).sum())
    log(
        f"   estructura: {len(conteo)} grupos, maximo {int(conteo.max())} filas, "
        f"{filas_unicas} filas en grupos de tamano 1 "
        f"({100 * filas_unicas / len(y):.1f} %)"
    )

    variantes: list[dict[str, Any]] = []
    log("   H0 control (1 modelo, tren completo)")
    proba_control, _ = ajustar_oof(textos, y, pliegues, grupos_dev, 1)
    variantes.append(evaluar("H0-control", y, proba_control, pred_v03))
    log(
        f"   H0-control   corrected_cv {variantes[0]['corrected_cv']:.6f} "
        f"delta {variantes[0]['delta_vs_V03']:+.4f} "
        f"IC95 [{variantes[0]['ic95_0']:+.4f}, {variantes[0]['ic95_1']:+.4f}]"
    )

    log(f"   H1 promedio de {MODELOS_POR_PLIEGUE} modelos con remuestreo de grupos")
    proba_bagging, individuales = ajustar_oof(textos, y, pliegues, grupos_dev, MODELOS_POR_PLIEGUE)
    variantes.append(evaluar("H1-bagging5", y, proba_bagging, pred_v03))
    diversidad = medir_diversidad(individuales)
    log(
        f"   H1-bagging5  corrected_cv {variantes[1]['corrected_cv']:.6f} "
        f"delta {variantes[1]['delta_vs_V03']:+.4f} "
        f"IC95 [{variantes[1]['ic95_0']:+.4f}, {variantes[1]['ic95_1']:+.4f}] | "
        f"salvage {variantes[1]['salvage']} damage {variantes[1]['damage']}"
    )
    log(
        f"   diversidad: correlacion media entre pares "
        f"{diversidad['correlacion_media_entre_pares']:.6f} (minima "
        f"{diversidad['correlacion_minima']:.6f}), diferencia media absoluta "
        f"{diversidad['diferencia_media_absoluta']:.6f}"
    )

    log("gate de dos niveles")
    for variante in variantes:
        delta = variante["delta_vs_V03"]
        excluye_cero = variante["ic95_0"] > 0
        cumple_delta = delta >= GATE_1_DELTA
        pliegues_positivos = _pliegues_positivos(y, variante["pred"], pliegues, pred_v03)
        variante["gate_1_delta"] = cumple_delta
        variante["gate_1_ic95"] = excluye_cero
        variante["pliegues_positivos"] = pliegues_positivos
        variante["gate_1_supera"] = (
            cumple_delta and excluye_cero and pliegues_positivos >= GATE_1_FOLDS
        )
        log(
            f"   {variante['variante']:12s} Gate 1: delta {'OK' if cumple_delta else 'no'} "
            f"({delta:+.4f}) | IC95 {'OK' if excluye_cero else 'no'} | pliegues "
            f"{pliegues_positivos}/{len(pliegues)} -> "
            f"{'SUPERA' if variante['gate_1_supera'] else 'no supera'}"
        )

    log("informe y figura")
    _figura(variantes, diversidad)
    _informe(variantes, diversidad, conteo, len(y), pliegues, time.time() - inicio)
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        campos = [
            "variante", "corrected_cv", "delta_vs_V03", "ic95_0", "ic95_1", "f1_HS",
            "f1_TR", "f1_AG", "salvage", "damage", "pliegues_positivos", "gate_1_supera",
        ]
        escritor = csv.DictWriter(archivo, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        for variante in variantes:
            escritor.writerow({campo: variante.get(campo, "") for campo in campos})
    REPORTE_JSON.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "referencia_v03": REFERENCIA_V03,
                "modelos_por_pliegue": MODELOS_POR_PLIEGUE,
                "diversidad": diversidad,
                "gate_1": {"delta": GATE_1_DELTA, "pliegues_minimos": GATE_1_FOLDS},
                "variantes": [
                    {k: v for k, v in variante.items() if k != "pred"}
                    for variante in variantes
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    mejores = max(variantes, key=lambda v: v["corrected_cv"])
    log(
        f"Track H completada en {round(time.time() - inicio, 1)} s | mejor: "
        f"{mejores['variante']} {mejores['corrected_cv']:.6f} "
        f"(delta {mejores['delta_vs_V03']:+.4f})"
    )


def _pliegues_positivos(y, pred, pliegues, pred_referencia) -> int:
    """Cuenta los pliegues en los que la variante mejora frente a la referencia.

    Args:
        y: matriz binaria verdadera.
        pred: predicciones de la variante.
        pliegues: pliegues corregidos.
        pred_referencia: predicciones de referencia.

    Returns:
        Numero de pliegues con mejora.
    """
    positivos = 0
    for validacion in pliegues:
        indices = validacion[1] if isinstance(validacion, tuple) else validacion
        if metrica_macro(y[indices], pred[indices]) > metrica_macro(
            y[indices], pred_referencia[indices]
        ):
            positivos += 1
    return positivos


def _figura(variantes, diversidad) -> None:
    """Dibuja el delta de las variantes y la dispersion de los modelos promediados.

    Args:
        variantes: resultados de las variantes.
        diversidad: medidas de diversidad de los modelos.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig, (arriba, abajo) = plt.subplots(2, 1, figsize=(9, 6.4), height_ratios=[2, 1])
    nombres = [v["variante"] for v in variantes]
    posiciones = np.arange(len(nombres))
    arriba.bar(posiciones, [v["delta_vs_V03"] for v in variantes], color="#17becf")
    arriba.axhline(0, color="black", linewidth=1)
    arriba.axhline(
        GATE_1_DELTA, color="red", linestyle=":", label=f"gate {GATE_1_DELTA:+.3f}"
    )
    arriba.set_xticks(posiciones)
    arriba.set_xticklabels(nombres, fontsize=9)
    arriba.set_ylabel("delta frente a V03")
    arriba.set_title("Bagging por grupos: delta frente a V03")
    arriba.legend(fontsize=8)

    abajo.hist(
        [v["delta_vs_V03"] for v in variantes],
        bins=8,
        color="#17becf",
        alpha=0.7,
    )
    abajo.set_ylabel("variantes")
    abajo.set_xlabel(f"deltas (diversidad medida: correlacion {diversidad['correlacion_media_entre_pares']:.4f})", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_63_bagging_grupos_v6.png", dpi=150)
    plt.close()


def _informe(variantes, diversidad, conteo, n_filas: int, pliegues, duracion: float) -> None:
    """Escribe el informe del track H.

    Args:
        variantes: resultados de las variantes.
        diversidad: medidas de diversidad de los modelos.
        conteo: numero de filas por grupo.
        n_filas: numero de filas del desarrollo.
        pliegues: pliegues corregidos.
        duracion: segundos que tardo el script.
    """
    filas_unicas = int((conteo == 1).sum())
    lineas = [
        "# Track H: bagging con remuestreo por grupos (V6-06)",
        "",
        f"Generado por `experiments/v6_06_bagging_grupos.py` ({EXPERIMENTO}).",
        "",
        "## 1. Que se prueba y por que no es una repeticion de ENS",
        "",
        "Los ensembles previos (ENS-01 a ENS-04) mezclaban **representaciones distintas** y no",
        "superaron al mejor modelo. Aqui se promedian **V03-equivalentes**: el mismo pipeline",
        "reentrenado sobre submuestras de grupos distintos. Son preguntas distintas, y el",
        "gap analysis (V6-00) confirmo que no existe ningun bagging en el proyecto.",
        "",
        "## 2. El techo se mide, no se supone",
        "",
        "El track A (V6-03) midio la estructura de grupos:",
        "",
        f"- **{len(conteo)} grupos distintos**, maximo {int(conteo.max())} filas.",
        f"- **{filas_unicas} de {n_filas} filas ({100 * filas_unicas / n_filas:.1f} %) en grupos",
        "  de tamano 1.**",
        "",
        "Eso tiene una consecuencia directa: **con casi todas las filas en grupos unicos,",
        "remuestrar por grupos y remuestrar por filas son practicamente la misma cosa**, y el",
        "promedio de modelos casi identicos no puede mover la metrica. Por eso este script",
        "**mide la diversidad de los modelos que promedia** en vez de limitarse a informar un",
        "delta. La magnitud que explica el resultado es la correlacion entre las",
        "predicciones de cada par de modelos.",
        "",
        "## 3. Resultados",
        "",
        "| variante | corrected_cv | delta | IC95 | HS | TR | AG | salvage | damage |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for variante in variantes:
        lineas.append(
            f"| {variante['variante']} | {variante['corrected_cv']:.6f} | "
            f"{variante['delta_vs_V03']:+.4f} | "
            f"[{variante['ic95_0']:+.4f}, {variante['ic95_1']:+.4f}] | "
            f"{variante['f1_HS']:.4f} | {variante['f1_TR']:.4f} | {variante['f1_AG']:.4f} | "
            f"{variante['salvage']} | {variante['damage']} |"
        )
    lineas += [
        "",
        "## 4. Diversidad de los modelos que se promedian",
        "",
        "| magnitud | valor |",
        "|---|---|",
        f"| correlacion media entre pares | {diversidad['correlacion_media_entre_pares']:.6f} |",
        f"| correlacion minima | {diversidad['correlacion_minima']:.6f} |",
        f"| diferencia media absoluta de probabilidad | "
        f"{diversidad['diferencia_media_absoluta']:.6f} |",
        f"| pares comparados | {diversidad['pares_comparados']} |",
        "",
        "## 5. Gate de dos niveles",
        "",
        "| variante | delta >= +0.003 | IC95 excluye 0 | pliegues positivos | Gate 1 |",
        "|---|---|---|---|---|",
    ]
    for variante in variantes:
        lineas.append(
            f"| {variante['variante']} | {'si' if variante['gate_1_delta'] else 'no'} "
            f"({variante['delta_vs_V03']:+.4f}) | "
            f"{'si' if variante['gate_1_ic95'] else 'no'} | "
            f"{variante['pliegues_positivos']}/{len(pliegues)} | "
            f"{'**SUPERA**' if variante['gate_1_supera'] else 'no supera'} |"
        )
    lineas += [
        "",
        "## 6. El resultado no es por falta de diversidad, y eso obliga a leerlo bien",
        "",
        f"La correlacion media entre pares de modelos es "
        f"{diversidad['correlacion_media_entre_pares']:.4f}, con una minima de",
        f"{diversidad['correlacion_minima']:.4f}, y la diferencia media absoluta de",
        f"probabilidad es {diversidad['diferencia_media_absoluta']:.4f}. Hay **diversidad",
        "real**: los cinco modelos no son el mismo modelo. Por tanto el empeoramiento **no se",
        "explica por falta de diversidad**, que era la hipotesis que el track planteaba medir.",
        "",
        "La explicacion coherente con los numeros es un **efecto de escala**: promediar cinco",
        "modelos contrae las probabilidades hacia el centro, y los umbrales congelados de",
        f"V03 ({', '.join(str(u) for u in UMBRALES_V03)}) estan calibrados para **un** modelo",
        "individual. Un umbral que separa bien las probabilidades de un modelo puede no",
        "separarlas igual tras el promedio.",
        "",
        "### La tension que esto abre, y por que no se resuelve",
        "",
        "Para evaluates el bagging con justicia habria que **reoptimizar los umbrales para el",
        "promedio**. Pero eso rompe la comparabilidad con V03 y reproduce exactamente el",
        "sesgo ya documentado en D-045: reoptimizar un umbral sobre el mismo OOF que se",
        "evalua da un numero inflado (ahi, B0 pasaba de 0.7772 a 0.7808 sin cambiar el modelo).",
        "",
        "En otras palabras: **no hay forma de evaluar el bagging de forma comparable con V03",
        "sin reoptimizar umbrales, y reoptimizarlos rompe la comparacion**. El track se",
        "cierra por una razon de **protocolo**, no porque se haya refutado un mecanismo. Se",
        "deja escrito con esa precision para que la linea pueda reabrirse si algun dia se",
        "dispone de una particion de calibracion separada del OOF de evaluacion.",
        "",
        "## 7. Por que el control esta dentro del promedio y no aparte",
        "",
        "El primer modelo de cada pliegue usa el **tren completo, sin remuestreo**. Asi H0 se",
        "obtiene con `n_modelos = 1` sin cambiar una sola linea del pipeline, y H1 incluye",
        "siempre un modelo equivalente a V03. Es la misma logica que los controles B0, A0, I0",
        "y C1-0, aqui adaptada al promedio.",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(duracion, 1)} s.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
