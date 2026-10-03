"""V32-06: Fase 7 de V32. Context-gated n-grams, con medicion previa de dimensionalidad.

Que responde
------------
La Fase 7 de la directiva V32. La hipotesis es mas potente que la de las features
estructurales:

> un mismo n-grama puede tener distinto valor dependiendo de la longitud del tweet.

En lugar de anadir features estructurales, se construye una version controlada de
``n-grama x length_bin``: para los K n-gramas mas frecuentes y estables del training fold, se
crean tres columnas (una por bin de longitud), de modo que el modelo puede aprender coeficientes
distintos para el mismo n-grama segun la longitud del texto.

La medicion que manda la directiva
-----------------------------------
Antes de ejecutar se mide el numero de columnas, el sparsity y el porcentaje del vocabulario,
y la regla de parada es clara: **si la dimensionalidad explota, no se ejecuta**.

El subconjunto de n-gramas
--------------------------
Se toman los K n-gramas que aparecen en al menos ``MIN_DF`` documentos del training fold y que
aparecen en documentos de **mas de un tramo de longitud**. Sin esa segunda condicion el gating no
tiene sentido: un n-grama que solo existe en textos cortos no puede tener un efecto distinto en
textos largos.

Uso
---
    /home/carlos/venv/bin/python experiments/v32_06_context_gated.py
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.splits import grupos_fusionados, pliegues_corregidos
from src.structural_eval import (
    ETIQUETAS,
    LR_PARAMS,
    UMBRALES_V03,
    comparar,
    evaluar_bloque,
    matriz_v03,
    metricas_multietiqueta,
    resumen,
    verificar_control,
)
from src.structural_features import columnas, extraer, interacciones

EXPERIMENTO = "V32-06 (Fase 7 de V32: context-gated n-grams)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v32_06_context_gated.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v32_06_context_gated.md"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
BLOQUE = "url"
K_TOP = 100
MIN_DF = 5
N_BINS = 3


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V32-06] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def seleccionar_ngramas(
    textos_train: list[str], k: int, min_df: int
) -> tuple[list[str], tuple[float, ...]]:
    """Selecciona los K n-gramas mas frecuentes que aparecen en mas de un tramo de longitud.

    Args:
        textos_train: textos del training fold.
        k: numero de n-gramas a seleccionar.
        min_df: frecuencia minima de documento.

    Returns:
        Tupla con los nombres de los n-gramas y los cortes de longitud.
    """
    vectorizador = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=min_df, sublinear_tf=True
    )
    matriz = vectorizador.fit_transform(textos_train)
    nombres = vectorizador.get_feature_names_out().tolist()
    tokens = np.array([len(t.split()) for t in textos_train], dtype=float)
    cortes = tuple(float(x) for x in np.percentile(tokens, [33.33, 66.67]))
    tramos = np.digitize(tokens, cortes)
    estables = []
    for indice in range(matriz.shape[1]):
        columna = np.asarray(matriz[:, indice].todense()).ravel() > 0
        if len(set(tramos[columna].tolist())) >= 2:
            estables.append(indice)
    estables.sort(key=lambda i: -float((matriz[:, indice] > 0).sum()))
    elegidos = [nombres[i] for i in estables[:k]]
    return elegidos, cortes


def main() -> None:
    """Ejecuta la Fase 7 de V32.

    Returns:
        None.
    """
    inicio = time.time()
    log(f"{EXPERIMENTO}. NINGUNA submission")

    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos_fusionados(desarrollo))
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    textos = desarrollo["text"].astype(str).tolist()
    tokens = np.array([len(t.split()) for t in textos], dtype=float)
    log(f"   desarrollo {len(desarrollo)} filas | {len(pliegues)} pliegues")

    log("1/4 control de V03")
    control = resumen(evaluar_bloque(desarrollo, pliegues, bloque=None))
    verificar_control(control["corrected_cv"])
    log(f"   control exacto: {control['corrected_cv']:.10f}")

    log("2/4 medicion de dimensionalidad, antes de ejecutar")
    idx_tr, _ = pliegues[0]
    nombres, cortes = seleccionar_ngramas([textos[i] for i in idx_tr], K_TOP, MIN_DF)
    n_columnas = len(nombres) * N_BINS
    vocabulario = len(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        .fit([textos[i] for i in idx_tr])
        .get_feature_names_out()
    )
    porcentaje = 100.0 * n_columnas / max(vocabulario, 1)
    log(f"   K={K_TOP} n-gramas | {N_BINS} bins -> {n_columnas} columnas nuevas")
    log(f"   vocabulario del training {vocabulario} | el bloque es el {porcentaje:.2f} %")
    log(f"   sparsity del bloque nueva {100.0 * (1.0 - n_columnas / max(vocabulario, 1)):.2f} %")
    log(f"   cortes de longitud {cortes}")
    if porcentaje > 50.0:
        log("   PARADA: la dimensionalidad explota y la directiva manda no ejecutar")
        informe = _informe_no_ejecutado(control, n_columnas, vocabulario, porcentaje, cortes)
        REPORTE_JSON.write_text(
            json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _informe_md(informe)
        log("   informe escrito en reports/v32_06_context_gated.{json,md}")
        log("VEREDICTO: no ejecutado por dimensionalidad, 0 submissions")
        return

    log("3/4 evaluacion de context-gated n-grams")
    proba = np.zeros((len(desarrollo), 3), dtype=float)
    metricas = []
    train_f1s = []
    base = pd.DataFrame(extraer(BLOQUE, textos))[columnas(BLOQUE)].reset_index(drop=True)
    for numero, (idx_tr, idx_va) in enumerate(pliegues, start=1):
        nombres, cortes = seleccionar_ngramas([textos[i] for i in idx_tr], K_TOP, MIN_DF)
        tramos = np.digitize(tokens, cortes)
        filas = [
            {f"int_{k}": v for k, v in interacciones(BLOQUE, fila, tok).items()}
            for fila, tok in zip(base.to_dict("records"), tokens, strict=True)
        ]
        inter = pd.DataFrame(filas).reset_index(drop=True)
        vectorizador = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, vocabulary=nombres
        )
        # sklearn exige fit aunque el vocabulario este fijo: el ajuste se hace dentro del
        # training fold, como en V03.
        vectorizador.fit([textos[i] for i in idx_tr])
        freq = vectorizador.transform(textos).toarray()
        gated = np.zeros((len(textos), len(nombres) * N_BINS), dtype=np.float32)
        for b in range(N_BINS):
            gated[:, b * len(nombres):(b + 1) * len(nombres)] = (
                freq * (tramos == b).astype(np.float32)[:, None]
            )
        completa = pd.concat(
            [base, inter, pd.DataFrame(gated)], axis=1
        ).reset_index(drop=True)
        train, val = matriz_v03(
            desarrollo["text"].iloc[idx_tr], desarrollo["text"].iloc[idx_va]
        )
        train = sparse.hstack([train, completa.iloc[idx_tr].to_numpy()]).tocsr()
        val = sparse.hstack([val, completa.iloc[idx_va].to_numpy()]).tocsr()
        pred_train = np.zeros((len(idx_tr), 3), dtype=np.int64)
        for j, etiqueta in enumerate(ETIQUETAS):
            modelo = LogisticRegression(**LR_PARAMS)
            modelo.fit(train, desarrollo[etiqueta].iloc[idx_tr])
            proba[idx_va, j] = modelo.predict_proba(val)[:, 1]
            pred_train[:, j] = (
                modelo.predict_proba(train)[:, 1] >= UMBRALES_V03[etiqueta]
            ).astype(int)
        pred_pliegue = np.column_stack(
            [(proba[idx_va, j] >= UMBRALES_V03[et]).astype(int) for j, et in enumerate(ETIQUETAS)]
        )
        metrica = metricas_multietiqueta(y[idx_va], pred_pliegue)
        metrica["pliegue"] = numero
        metricas.append(metrica)
        train_f1s.append(metricas_multietiqueta(y[idx_tr], pred_train)["f1_media_columnas"])
        log(f"   pliegue {numero}: F1 media {metrica['f1_media_columnas']:.4f}")
    pred = np.column_stack(
        [(proba[:, j] >= UMBRALES_V03[et]).astype(int) for j, et in enumerate(ETIQUETAS)]
    )
    m = metricas_multietiqueta(y, pred)
    ch = {
        "corrected_cv": float(m["f1_media_columnas"]),
        "f1_por_etiqueta": {et: float(m["f1_por_etiqueta"][et]) for et in ETIQUETAS},
        "train_f1": float(np.mean(train_f1s)),
        "metricas_por_pliegues": metricas,
    }
    ch["gap_train_oof"] = ch["train_f1"] - ch["corrected_cv"]
    comp = comparar(y, evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"], pred)
    log(f"   corrected_cv {ch['corrected_cv']:.10f} | "
        f"delta {ch['corrected_cv'] - control['corrected_cv']:+.10f}")
    log(f"   rescates {comp['rescates']} danios {comp['danios']} neto {comp['neto']:+d}")

    log("4/4 informe")
    informe = {
        "familia": "STRUCTURAL_TEXT_FEATURES",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "OOF_EVALUATED",
        "fase": "7_CONTEXT_GATED_NGRAMS",
        "pregunta": "un mismo n-grama tiene distinto valor segun la longitud del tweet",
        "alcance": (
            "version controlada de n-grama x length_bin para los K n-gramas mas frecuentes y "
            "estables del training fold"
        ),
        "medicion_previa": {
            "k_top": K_TOP,
            "min_df": MIN_DF,
            "n_bins": N_BINS,
            "nuevas_columnas": n_columnas,
            "vocabulario_training": vocabulario,
            "porcentaje_del_vocabulario": porcentaje,
            "cortes_de_longitud": list(cortes),
        },
        "regla_de_seleccion": (
            "los n-gramas aparecen en al menos MIN_DF documentos y en documentos de mas de un "
            "tramo de longitud"
        ),
        "control": control,
        "challenger": ch,
        "delta_oof": ch["corrected_cv"] - control["corrected_cv"],
        "comparacion": comp,
        "gastos": {"submissions": 0, "modelos_entrenados": 3,
                   "memoria_pico_mb": round(memoria_pico_mb(), 1),
                   "segundos": round(time.time() - inicio, 1)},
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe_md(informe)
    log("   informe escrito en reports/v32_06_context_gated.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: Fase 7 evaluada, 0 submissions")


def _informe_no_ejecutado(
    control: dict[str, Any],
    n_columnas: int,
    vocabulario: int,
    porcentaje: float,
    cortes: tuple[float, ...],
) -> dict[str, Any]:
    """Informe cuando la dimensionalidad impide ejecutar.

    Args:
        control: metricas del control.
        n_columnas: columnas nuevas que se hubieran creado.
        vocabulario: vocabulario del training.
        porcentaje: porcentaje del vocabulario.
        cortes: cortes de longitud.

    Returns:
        El informe como diccionario.
    """
    return {
        "familia": "STRUCTURAL_TEXT_FEATURES",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "NOT_EXECUTED",
        "fase": "7_CONTEXT_GATED_NGRAMS",
        "pregunta": "un mismo n-grama tiene distinto valor segun la longitud del tweet",
        "alcance": "medicion de dimensionalidad; la directiva manda no ejecutar si explota",
        "medicion_previa": {
            "k_top": K_TOP,
            "min_df": MIN_DF,
            "n_bins": N_BINS,
            "nuevas_columnas": n_columnas,
            "vocabulario_training": vocabulario,
            "porcentaje_del_vocabulario": porcentaje,
            "cortes_de_longitud": list(cortes),
        },
        "regla_de_parada": "si las nuevas columnas superan el 50 % del vocabulario, no se ejecuta",
        "control": control,
        "challenger": None,
        "gastos": {"submissions": 0, "modelos_entrenados": 1,
                   "memoria_pico_mb": round(memoria_pico_mb(), 1), "segundos": 0.0},
    }


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    c = informe["control"]
    m = informe["medicion_previa"]
    ch = informe.get("challenger")
    g = informe["gastos"]
    lineas = [
        "# V32-06: Fase 7 de V32, context-gated n-grams",
        "",
        f"Generado por `experiments/v32_06_context_gated.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: {informe['alcance']}**",
        "",
        "## 1. La medicion previa, que es la que decide si se ejecuta",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| K n-gramas seleccionados | {m['k_top']} |",
        f"| min_df | {m['min_df']} |",
        f"| bins de longitud | {m['n_bins']} |",
        f"| nuevas columnas | {m['nuevas_columnas']} |",
        f"| vocabulario del training | {m['vocabulario_training']} |",
        f"| porcentaje del vocabulario | {m['porcentaje_del_vocabulario']:.2f} % |",
        f"| cortes de longitud | {m['cortes_de_longitud']} |",
        "",
        f"{informe['regla_de_seleccion']}.",
        "",
    ]
    if ch is not None:
        lineas += [
            "## 2. Resultado",
            "",
            "| etiqueta | F1 control | F1 challenger | delta |",
            "|---|---:|---:|---:|",
        ]
        for et in ETIQUETAS:
            lineas.append(
                f"| {et} | {c['f1_por_etiqueta'][et]:.4f} | {ch['f1_por_etiqueta'][et]:.4f} | "
                f"{ch['f1_por_etiqueta'][et] - c['f1_por_etiqueta'][et]:+.4f} |"
            )
        lineas += [
            "",
            (f"**corrected_cv {c['corrected_cv']:.10f} -> {ch['corrected_cv']:.10f} | "
            f"delta {informe['delta_oof']:+.10f}**"),
            "",
            f"train F1 {ch['train_f1']:.4f} | gap {ch['gap_train_oof']:+.4f}",
            "",
        ]
        if "comparacion" in informe:
            comp = informe["comparacion"]
            lineas += [
                f"rescates {comp['rescates']} | danios {comp['danios']} | neto {comp['neto']:+d}.",
                "",
            ]
    else:
        lineas += [
            "## 2. No ejecutado",
            "",
            f"{informe['regla_de_parada']}.",
            "",
        ]
    lineas += [
        "## 3. Lo que este script no hace",
        "",
        "- No aplica el gating a todo el vocabulario.",
        "- No genera submission. No lee el test.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
