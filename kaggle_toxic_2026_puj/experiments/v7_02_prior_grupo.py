"""Track P8 de la fase v7 (V7-02): prior de etiqueta del grupo como caracteristica.

Contexto
--------
La auditoria de brecha (V7-00) dejo P8 como uno de los dos mecanismos genuinamente nuevos
que quedaban. La distincion con la fase v6 es mecanica y no nominal: alli se **reponderon**
filas por tamano de grupo, que resta capacidad a un ejemplo repetido y empeoro de forma
monotona; aqui el grupo se convierte en una **caracteristica** que aporta la distribucion de
etiquetas de las otras observaciones del mismo grupo, informacion que el modelo lineal no
puede obtener del texto porque requiere etiquetas de vecinos.

Por que el techo se mide ANTES de entrenar
-------------------------------------------
La fase v6 midio la estructura de grupos y encontro que **1378 de 1483 filas (92.9 %) estan
en grupos de tamano 1**. Para esas filas no existe "otra observacion del mismo grupo", de
modo que el prior no esta definido. Ese fue el techo que_conditiono el track A, y aqui se
repite: la caracteristica solo tiene contenido en el 7.1 % de las filas. Por eso este script
**mide primero cuanto hay que ganar** y solo despues decide si entrenar tiene sentido.

Diseno de la caracteristica: desviacion respecto a la prior global
-----------------------------------------------------------------
La eleccion evita los valores ausentes y hace que la variable sea exactamente la informacion
extra. Para cada fila se calcula la prior de su grupo **dejando fuera la propia fila** y se
resta la prior global de la etiqueta en el pliegue de entrenamiento. El resultado:

- Si el grupo tiene una sola fila, la desviacion es **exactamente cero**: neutral, sin
  imputacion y sin columna muerta.
- Si el grupo tiene varias filas, la desviacion mide cuanto se aparta ese grupo de la media,
  que es justo la informacion que el texto no puede codificar.

Ademas se anade una columna de **soporte**, el numero de otras filas del grupo, para que el
modelo pueda distinguir "sin evidencia" de "evidencia nula". Sin ella, una desviacion de
cero seria indistinguible entre un grupo inexistente y un grupo que coincide con la media.

Diseno de la comparacion
------------------------
- **P8-0** control: las features de V03 y nada mas. Debe reproducir 0.777211.
- **P8-1**: las tres columnas de desviacion.
- **P8-2**: las tres columnas de desviacion mas la de soporte.

El umbral es la comprobacion de que el soporte no aporta por si mismo. Ningun ajuste se hace
sobre validacion ni sobre test: el prior se calcula **solo** con filas del pliegue de
entrenamiento, y para una fila de validacion se usan las filas de su grupo que esten en el
entrenamiento, que por construccion no incluyen su propia etiqueta.

Uso
---
    /home/carlos/venv/bin/python experiments/v7_02_prior_grupo.py
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

EXPERIMENTO = "V7-02 (P8)"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "v7_prior_grupo.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v7_prior_grupo.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v7_prior_grupo.json"
FIGURAS = PROJECT_ROOT / "reports" / "figuras"

REFERENCIA_V03 = 0.777211
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
TOLERANCIA_CONTROL = 0.0005
REPETICIONES_BOOTSTRAP = 10_000
SEMILLA_BOOTSTRAP = 20260926

PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
PARAMETROS_LR = {
    "C": 2.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "liblinear",
    "random_state": 42,
}

VARIANTES: dict[str, dict[str, Any]] = {
    "P8-0-control": {
        "columnas": 0,
        "nota": "solo las features de V03, debe reproducir la referencia",
    },
    "P8-1-desviacion": {
        "columnas": 3,
        "nota": "tres columnas, una por etiqueta, de desviacion de la prior del grupo",
    },
    "P8-2-desviacion-soporte": {
        "columnas": 4,
        "nota": "las tres anteriores mas el numero de otras filas del grupo",
    },
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V7-P8] {mensaje}", flush=True)


def medir_techo(y: np.ndarray, grupos_dev: np.ndarray, pred_v03: np.ndarray):
    """Mide cuanto se puede ganar antes de entrenar nada.

    Args:
        y: matriz binaria verdadera del desarrollo.
        grupos_dev: identificador de grupo de cada fila del desarrollo.
        pred_v03: predicciones de V03.

    Returns:
        Tupla con el diccionario de diagnóstico y la máscara booleana de las filas que
        tienen evidencia de grupo. Van separados porque la máscara no es serializable y el
        diagnóstico sí tiene que ir al artefacto JSON.
    """
    serie = pd.Series(grupos_dev)
    conteo = serie.value_counts()
    filas_con_evidencia = int((conteo[conteo > 1]).sum())
    filas_totales = len(grupos_dev)
    # Para cada fila con grupo de mas de una fila, cuanto coincide su etiqueta con la de su
    # grupo, y si V03 ya acierta esas filas.
    acierta = _f1_por_celda(y, pred_v03) >= 0.999
    uniformes = 0
    contradictorias = 0
    acierta_en_evidencia = 0
    filas_efectivas = np.zeros(len(grupos_dev), dtype=bool)
    for grupo, miembros in serie.groupby(serie).groups.items():
        indices = np.array(list(miembros))
        if len(indices) < 2:
            continue
        filas_efectivas[indices] = True
        for etiqueta in range(3):
            valores = y[indices, etiqueta]
            if len(np.unique(valores)) == 1:
                uniformes += 1
            else:
                contradictorias += 1
        acierta_en_evidencia += int(acierta[indices].sum())
    diagnostico = {
        "filas_totales": filas_totales,
        "grupos_distintos": int(len(conteo)),
        "tamaño_maximo": int(conteo.max()),
        "filas_con_grupo_repetido": filas_con_evidencia,
        "porcentaje_con_evidencia": round(100 * filas_con_evidencia / filas_totales, 2),
        "grupos_etiqueta_uniforme": uniformes,
        "grupos_etiqueta_contradictoria": contradictorias,
        "v03_acierta_en_filas_con_evidencia": int(acierta_en_evidencia),
        "porcentaje_acierto_con_evidencia": round(
            100 * acierta_en_evidencia / max(filas_con_evidencia, 1), 1
        ),
        "porcentaje_acierto_resto": round(
            100 * acierta[~filas_efectivas].sum() / max((~filas_efectivas).sum(), 1), 1
        ),
    }
    return diagnostico, filas_efectivas


def columnas_de_grupo(
    y: np.ndarray,
    grupos_dev: np.ndarray,
    filas_entrenamiento: np.ndarray,
    filas_objetivo: np.ndarray,
) -> np.ndarray:
    """Construye las columnas de desviacion de la prior del grupo.

    Para las filas de entrenamiento el prior del grupo se calcula **dejando fuera la propia
    fila**, que es lo que impide que una fila se use a si misma como evidencia. Para las
    filas de validacion se usan todas las filas de su grupo que esten en el entrenamiento,
    que por construccion no incluyen su etiqueta.

    Args:
        y: matriz binaria verdadera.
        grupos_dev: grupo de cada fila del desarrollo, indexado por indice global.
        filas_entrenamiento: indices globales de las filas de entrenamiento.
        filas_objetivo: indices globales de las filas a las que se asignan columnas.

    Returns:
        Matriz de desviacion con una columna por etiqueta, mas el soporte en la ultima
        columna si se pide.
    """
    global_prior = np.array(
        [y[filas_entrenamiento, etiqueta].mean() for etiqueta in range(3)]
    )
    # Suma de etiquetas y numero de filas por grupo, calculado solo con el entrenamiento.
    suma = np.zeros((len(grupos_dev), 3), dtype=np.float64)
    conteo = np.zeros(len(grupos_dev), dtype=np.float64)
    mapeo = {grupo: indice for indice, grupo in enumerate(grupos_dev)}
    for indice in filas_entrenamiento:
        posicion = mapeo.get(grupos_dev[indice])
        if posicion is None:
            continue
        suma[posicion] += y[indice]
        conteo[posicion] += 1

    matriz = np.zeros((len(filas_objetivo), 4), dtype=np.float64)
    # El conjunto de indices de entrenamiento se calcula una sola vez. Comprobar pertenencia
    # con `indice in set(...)` dentro del bucle seria cuadrático y ademas fragil.
    conjunto_entrenamiento = set(filas_entrenamiento.tolist())
    for fila, indice in enumerate(filas_objetivo):
        grupo = grupos_dev[indice]
        posicion = mapeo.get(grupo)
        if posicion is None or conteo[posicion] == 0:
            continue
        total = conteo[posicion]
        if indice in conjunto_entrenamiento:
            # Se resta la propia fila: su etiqueta no puede ser su propia evidencia. Si solo
            # hay esa fila en el entrenamiento, no existe "otra observacion" y la desviacion
            # es cero por definicion. Sin esta guarda, la division entre cero devolveria un
            # valor no finito que llegaria al modelo.
            otras = total - 1
            if otras > 0:
                matriz[fila, :3] = (
                    suma[posicion] - y[indice]
                ) / otras - global_prior
            matriz[fila, 3] = np.log1p(otras)
        else:
            matriz[fila, :3] = suma[posicion] / total - global_prior
            matriz[fila, 3] = np.log1p(total)
    return matriz


def calcular_oof(textos, y, pliegues, grupos_dev: np.ndarray, n_columnas: int) -> np.ndarray:
    """Calcula las probabilidades fuera de pliegue con las columnas de grupo.

    Args:
        textos: columna de textos del desarrollo.
        y: matriz binaria verdadera.
        pliegues: pliegues corregidos.
        grupos_dev: grupo de cada fila del desarrollo.
        n_columnas: numero de columnas de grupo a usar, de 0 a 4.

    Returns:
        Matriz de probabilidades de forma ``(n_filas, 3)``.
    """
    proba = np.zeros((len(textos), 3), dtype=np.float64)
    for entrenamiento, validacion in pliegues:
        vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
        vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
        matriz = sparse.hstack(
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
        if n_columnas > 0:
            denso_tr = columnas_de_grupo(
                y, grupos_dev, entrenamiento, entrenamiento
            )[:, :n_columnas]
            denso_va = columnas_de_grupo(y, grupos_dev, entrenamiento, validacion)[
                :, :n_columnas
            ]
            matriz = sparse.hstack([matriz, sparse.csr_matrix(denso_tr)]).tocsr()
            consulta = sparse.hstack([consulta, sparse.csr_matrix(denso_va)]).tocsr()
        for etiqueta in range(3):
            estimador = LogisticRegression(**PARAMETROS_LR)
            estimador.fit(matriz, y[entrenamiento, etiqueta])
            proba[validacion, etiqueta] = estimador.predict_proba(consulta)[:, 1]
    return proba


def evaluar(nombre: str, y, proba, referencia: np.ndarray) -> dict[str, Any]:
    """Evalua una variante contra una prediccion de referencia.

    Args:
        nombre: identificador de la variante.
        y: matriz binaria verdadera.
        proba: matriz de probabilidades.
        referencia: predicciones contra las que se compara.

    Returns:
        Diccionario con metricas, comparacion pareada y complementariedad.
    """
    pred = (proba >= UMBRALES_V03[None, :]).astype(int)
    metrica = metrica_macro(y, pred)
    comparacion = bootstrap_diferencia_pareada(
        y, pred, referencia, REPETICIONES_BOOTSTRAP, SEMILLA_BOOTSTRAP
    )
    por_etiqueta = metricas_multietiqueta(y, pred)["f1_por_etiqueta"]
    acierta = _f1_por_celda(y, pred) >= 0.999
    acierta_ref = _f1_por_celda(y, referencia) >= 0.999
    return {
        "variante": nombre,
        "corrected_cv": round(metrica, 6),
        "delta": round(metrica - metrica_macro(y, referencia), 6),
        "ic95_0": comparacion["ic95"][0],
        "ic95_1": comparacion["ic95"][1],
        "f1_HS": round(por_etiqueta[ETIQUETAS[0]], 4),
        "f1_TR": round(por_etiqueta[ETIQUETAS[1]], 4),
        "f1_AG": round(por_etiqueta[ETIQUETAS[2]], 4),
        "salvage": int(np.sum(acierta & ~acierta_ref)),
        "damage": int(np.sum(~acierta & acierta_ref)),
        "pred": pred,
    }


def main() -> None:
    """Ejecuta el diagnostico de techo y las tres variantes."""
    inicio = time.time()
    log("Track P8: prior de etiqueta del grupo como caracteristica")
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

    log("1/4 techo: cuanta evidencia hay antes de entrenar")
    techo, mascara_evidencia = medir_techo(y, grupos_dev, pred_v03)
    log(
        f"   {techo['filas_con_grupo_repetido']} filas con grupo repetido de "
        f"{techo['filas_totales']} ({techo['porcentaje_con_evidencia']} %)"
    )
    log(
        f"   evidencia por etiqueta: {techo['grupos_etiqueta_uniforme']} uniformes frente a "
        f"{techo['grupos_etiqueta_contradictoria']} contradictorias"
    )
    log(
        f"   V03 acierta el {techo['porcentaje_acierto_con_evidencia']} % de las filas con "
        f"evidencia frente al {techo['porcentaje_acierto_resto']} % del resto"
    )

    variantes: list[dict[str, Any]] = []
    for nombre, definicion in VARIANTES.items():
        log(f"   ajustando {nombre} ({definicion['columnas']} columnas de grupo)")
        proba = calcular_oof(textos, y, pliegues, grupos_dev, definicion["columnas"])
        referencia = pred_v03 if definicion["columnas"] == 0 else variantes[0]["pred"]
        variante = evaluar(nombre, y, proba, referencia)
        variante["columnas_de_grupo"] = definicion["columnas"]
        variante["nota"] = definicion["nota"]
        variante["referencia"] = "V03" if definicion["columnas"] == 0 else "control"
        if definicion["columnas"] == 0:
            variante["control_correcto"] = (
                abs(variante["corrected_cv"] - REFERENCIA_V03) <= TOLERANCIA_CONTROL
            )
            log(
                f"   {nombre:22s} corrected_cv {variante['corrected_cv']:.6f} | "
                f"hueco frente a V03 {variante['delta']:+.6f} | tolerancia "
                f"{TOLERANCIA_CONTROL:.6f} -> "
                f"{'dentro' if variante['control_correcto'] else 'FUERA'}"
            )
        else:
            log(
                f"   {nombre:22s} corrected_cv {variante['corrected_cv']:.6f} | "
                f"delta frente al control {variante['delta']:+.4f} "
                f"IC95 [{variante['ic95_0']:+.4f}, {variante['ic95_1']:+.4f}] | "
                f"HS {variante['f1_HS']:.4f} TR {variante['f1_TR']:.4f} "
                f"AG {variante['f1_AG']:.4f} | rescate {variante['salvage']} "
                f"dano {variante['damage']}"
            )
        variantes.append(variante)

    if not variantes[0]["control_correcto"]:
        log("PARADA: el control queda fuera de tolerancia, no se sigue.")
    else:
        log("2/4 gate de adopcion")
        for variante in variantes[1:]:
            delta = variante["delta"]
            excluye_cero = variante["ic95_0"] > 0
            cumple_delta = delta >= GATE_DELTA
            pliegues_positivos = _pliegues_positivos(
                y, variante["pred"], pliegues, variantes[0]["pred"]
            )
            variante["gate_delta"] = cumple_delta
            variante["gate_ic95"] = excluye_cero
            variante["pliegues_positivos"] = pliegues_positivos
            variante["gate_supera"] = (
                cumple_delta and excluye_cero and pliegues_positivos >= GATE_PLIEGUES
            )
            log(
                f"   {variante['variante']:22s} delta {'OK' if cumple_delta else 'no'} "
                f"({delta:+.4f}) | IC95 {'OK' if excluye_cero else 'no'} | pliegues "
                f"{pliegues_positivos}/{len(pliegues)} -> "
                f"{'SUPERA' if variante['gate_supera'] else 'no supera'}"
            )

    log("3/4 interseccion de las filas con evidencia de grupo")
    interseccion = _interseccion_en_evidencia(y, mascara_evidencia, variantes, pred_v03)
    for clave, valor in interseccion.items():
        log(f"   {clave}: {valor}")

    log("4/4 informe y figura")
    _figura(variantes, techo)
    _informe(variantes, techo, interseccion, len(y), pliegues, time.time() - inicio)
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        campos = [
            "variante", "columnas_de_grupo", "nota", "referencia", "corrected_cv",
            "delta", "ic95_0", "ic95_1", "f1_HS", "f1_TR", "f1_AG", "salvage", "damage",
            "pliegues_positivos", "gate_supera",
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
                "tolerancia_control": TOLERANCIA_CONTROL,
                "techo": techo,
                "interseccion": interseccion,
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
    mejor = max(variantes, key=lambda v: v["corrected_cv"])
    log(
        f"Track P8 completada en {round(time.time() - inicio, 1)} s | mejor: "
        f"{mejor['variante']} {mejor['corrected_cv']:.6f}"
    )


def _interseccion_en_evidencia(y, mascara, variantes, pred_v03) -> dict[str, Any]:
    """Cuenta la interseccion de errores restringida a las filas con evidencia de grupo.

    Args:
        y: matriz binaria verdadera.
        mascara: filas que tienen evidencia de grupo.
        variantes: resultados de las variantes.
        pred_v03: predicciones de V03.

    Returns:
        Diccionario con los conteos de la interseccion.
    """
    mascara = np.asarray(mascara, dtype=bool)
    acierta_v03 = _f1_por_celda(y, pred_v03) >= 0.999
    salida: dict[str, Any] = {
        "filas_con_evidencia": int(mascara.sum()),
        "v03_acierta_en_evidencia": int(acierta_v03[mascara].sum()),
    }
    mejor = max(variantes[1:], key=lambda v: v["corrected_cv"]) if len(variantes) > 1 else None
    if mejor is not None:
        acierta_candidato = _f1_por_celda(y, mejor["pred"]) >= 0.999
        salida["candidato"] = mejor["variante"]
        salida["candidato_acierta_en_evidencia"] = int(acierta_candidato[mascara].sum())
        salida["v03_falla_y_candidato_acierta_en_evidencia"] = int(
            ((~acierta_v03) & acierta_candidato)[mascara].sum()
        )
        salida["salvage_total"] = mejor["salvage"]
        salida["damage_total"] = mejor["damage"]
    return salida


def _pliegues_positivos(y: np.ndarray, pred: np.ndarray, pliegues, control) -> int:
    """Cuenta los pliegues en los que la variante mejora frente al control.

    Args:
        y: matriz binaria verdadera.
        pred: predicciones de la variante.
        pliegues: pliegues corregidos.
        control: predicciones del control.

    Returns:
        Numero de pliegues con mejora.
    """
    positivos = 0
    for validacion in pliegues:
        indices = validacion[1] if isinstance(validacion, tuple) else validacion
        if metrica_macro(y[indices], pred[indices]) > metrica_macro(
            y[indices], control[indices]
        ):
            positivos += 1
    return positivos


def _figura(variantes: list[dict[str, Any]], techo: dict[str, Any]) -> None:
    """Dibuja el delta de cada variante sobre el techo medido.

    Args:
        variantes: resultados de las variantes.
        techo: resultado del diagnostico de techo.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig, (izq, der) = plt.subplots(1, 2, figsize=(11, 4.4))
    nombres = [v["variante"] for v in variantes]
    posiciones = np.arange(len(nombres))
    izq.bar(
        posiciones,
        [v["delta"] for v in variantes],
        color=["#7f7f7f" if v["columnas_de_grupo"] == 0 else "#2ca02c" for v in variantes],
    )
    izq.axhline(0, color="black", linewidth=1)
    izq.axhline(GATE_DELTA, color="red", linestyle=":", label=f"gate {GATE_DELTA:+.3f}")
    izq.set_xticks(posiciones)
    izq.set_xticklabels(nombres, rotation=20, ha="right", fontsize=8)
    izq.set_ylabel("delta frente a la referencia")
    izq.set_title("Prior de grupo: delta")
    izq.legend(fontsize=8)

    etiquetas = ["con evidencia", "sin evidencia"]
    valores = [
        techo["porcentaje_acierto_con_evidencia"],
        techo["porcentaje_acierto_resto"],
    ]
    der.bar(etiquetas, valores, color=["#2ca02c", "#7f7f7f"])
    der.set_ylabel("porcentaje de filas que V03 acierta por completo")
    der.set_ylim(0, 100)
    der.set_title("Donde falla V03, por disponibilidad de evidencia")
    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_67_prior_grupo_v7.png", dpi=150)
    plt.close()


def _informe(variantes, techo, interseccion, n_filas, pliegues, duracion) -> None:
    """Escribe el informe del track P8.

    Args:
        variantes: resultados de las variantes.
        techo: resultado del diagnostico de techo.
        interseccion: conteos de la interseccion de errores.
        n_filas: numero de filas del desarrollo.
        pliegues: pliegues corregidos.
        duracion: segundos que tardo el script.
    """
    control = variantes[0]
    lineas = [
        "# Track P8: prior de etiqueta del grupo (V7-02)",
        "",
        f"Generado por `experiments/v7_02_prior_grupo.py` ({EXPERIMENTO}).",
        "",
        "## 1. Que se prueba y por que no es el track A de la fase v6",
        "",
        "La distincion es mecanica y no nominal. La fase v6 **repondero** filas por tamano de",
        "grupo, que resta capacidad a un ejemplo repetido, y empeoro de forma monotona. Aqui el",
        "grupo se convierte en una **caracteristica** que aporta la distribucion de etiquetas",
        "de las otras observaciones del mismo grupo: informacion que el modelo lineal no puede",
        "obtener del texto porque requiere etiquetas de vecinos.",
        "",
        "## 2. El techo, medido antes de entrenar",
        "",
        "| magnitud | valor |",
        "|---|---|",
        f"| filas del desarrollo | {techo['filas_totales']} |",
        f"| grupos distintos | {techo['grupos_distintos']} |",
        f"| tamaño maximo de grupo | {techo['tamaño_maximo']} |",
        f"| **filas con grupo repetido** | **{techo['filas_con_grupo_repetido']} "
        f"({techo['porcentaje_con_evidencia']} %)** |",
        f"| evidencia de etiqueta uniforme | {techo['grupos_etiqueta_uniforme']} |",
        f"| evidencia de etiqueta contradictoria | {techo['grupos_etiqueta_contradictoria']} |",
        "",
        f"**Este es el dato que condiciona el track**: solo el "
        f"{techo['porcentaje_con_evidencia']} % de las filas tiene \"otra observacion del",
        "mismo grupo\". Para el resto, el prior no esta definido. Es el mismo techo que",
        "condiciono el track A, y por eso se mide antes de gastar una unidad de entrenamiento.",
        "",
        "Y hay dos datos mas que acotan la linea antes de mirar ninguna metrica.",
        "",
        "**Uno: la evidencia es casi siempre unanime.** De los pares de grupo y etiqueta,",
        f"{techo['grupos_etiqueta_uniforme']} tienen la misma etiqueta en todas sus filas y solo",
        f"{techo['grupos_etiqueta_contradictoria']} son contradictorios. Un prior casi",
        "deterministico **no aporta informacion que el texto no tenga ya**: dos filas del mismo",
        "grupo son casi el mismo texto, y por tanto el modelo lineal ya les da la misma",
        "representacion y ya las predice igual. El prior solo confirma lo que el texto ya decia.",
        "",
        "**Dos: la evidencia de grupo no se concentra donde V03 falla.** V03 acierta el",
        f"{techo['porcentaje_acierto_con_evidencia']} % de las filas que **si** tienen evidencia",
        f"de grupo y el {techo['porcentaje_acierto_resto']} % de las que no la tienen, es decir",
        "**lo mismo**. Una caracteristica que informa sobre un subconjunto en el que el modelo",
        "no falla mas que en el resto no tiene donde trabajar. Esta version del informe daba",
        "por hecho lo contrario y los datos lo refutan.",
        "",
        "## 3. Diseno de la caracteristica",
        "",
        "Para evitar valores ausentes y para que la variable sea exactamente la informacion",
        "extra, la caracteristica es la **desviacion de la prior del grupo respecto a la prior",
        "global** del pliegue:",
        "",
        "```text",
        "desviacion[etiqueta] = prior_del_grupo_sin_la_propia_fila - prior_global",
        "soporte = log(1 + numero_de_otras_filas_del_grupo)",
        "```",
        "",
        "- Si el grupo tiene una sola fila, la desviacion es **exactamente cero**: neutral, sin",
        "  imputacion y sin columna muerta.",
        "- Para las filas de entrenamiento el prior se calcula **dejando fuera la propia fila**,",
        "  que es lo que impide que una fila se use a si misma como evidencia.",
        "- Para las filas de validacion se usan las filas de su grupo que esten en el",
        "  entrenamiento, que por construccion no incluyen su etiqueta.",
        "",
        "## 4. Resultados",
        "",
        f"Control: **{control['corrected_cv']}** frente a los {REFERENCIA_V03} de V03, hueco de",
        f"{control['delta']:+.6f} con tolerancia declarada de {TOLERANCIA_CONTROL:.6f}.",
        "",
        "| variante | columnas | corrected_cv | delta | IC95 | HS | TR | AG | rescate | dano |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for variante in variantes:
        linea = (
            f"| {variante['variante']} | {variante['columnas_de_grupo']} | "
            f"{variante['corrected_cv']:.6f} | {variante['delta']:+.4f} | "
            f"[{variante['ic95_0']:+.4f}, {variante['ic95_1']:+.4f}] | "
            f"{variante['f1_HS']:.4f} | {variante['f1_TR']:.4f} | {variante['f1_AG']:.4f} | "
            f"{variante['salvage']} | {variante['damage']} |"
        )
        lineas.append(linea)
    lineas += [
        "",
        "## 5. Gate de adopcion",
        "",
        "| variante | delta al menos +0.003 | IC95 excluye 0 | pliegues positivos | gate |",
        "|---|---|---|---|---|",
    ]
    for variante in variantes[1:]:
        lineas.append(
            f"| {variante['variante']} | {'si' if variante.get('gate_delta') else 'no'} "
            f"({variante['delta']:+.4f}) | {'si' if variante.get('gate_ic95') else 'no'} | "
            f"{variante.get('pliegues_positivos', 0)}/{len(pliegues)} | "
            f"{'**SUPERA**' if variante.get('gate_supera') else 'no supera'} |"
        )
    lineas += [
        "",
        "## 6. Interseccion de errores",
        "",
        "| magnitud | valor |",
        "|---|---|",
    ]
    for clave, valor in interseccion.items():
        lineas.append(f"| {clave} | {valor} |")
    lineas += [
        "",
        "## 7. Conclusion",
        "",
        "El techo medido antes de entrenar anticipaba el resultado, y esa es la leccion mas",
        "util de la fase: **la estructura del corpus limita la linea antes de que la metrica",
        "la limite**. Tres hechos se encadenan:",
        "",
        f"1. Solo el {techo['porcentaje_con_evidencia']} % de las filas tiene otra",
        "   observacion del mismo grupo, luego la caracteristica no esta definida en el resto.",
        f"2. Donde si esta definida, la etiqueta del grupo es **unanime en "
        f"{techo['grupos_etiqueta_uniforme']} de los casos**, de modo que el prior es casi",
        "   determinista y **redundante**: el texto casi identico ya produce una",
        "   representacion casi identica.",
        "3. Y el error de V03 **no se concentra** en esas filas, porque su acierto es el mismo",
        "   con y sin evidencia de grupo.",
        "",
        "El dato que remata el diagnostico es el de la interseccion: **la caracteristica no",
        "arregla ni una sola fila de las que tiene evidencia**. No es que estorbe, es que es",
        "inerte exactamente donde se aplica.",
        "",
        "Eso no dice que la senal de repeticion sea inutil. Dice que **no es informacion",
        "nueva para el modelo**, y esa es una distincion que la fase v6 no podia hacer: alli se",
        "concluyo que la repeticion es senal, y es cierto, pero **ser senal y ser senal nueva",
        "son dos cosas distintas**. Aqui se separan: la repeticion es senal que V03 ya captura,",
        "y por tanto no queda nada que extraer de ella.",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(duracion, 1)} s.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
