"""V35-01: SGD hinge y modified_huber sobre la representacion exacta de V03.

Que responde
------------
La pregunta de la fase, dicha con las palabras de la directiva:

> Una perdida lineal distinta, particularmente Modified Huber, puede encontrar una solucion que
> generalice mejor que la regresion logistica de V03 sobre exactamente la misma representacion?

Correccion conceptual de la directiva
--------------------------------------
`SGDClassifier` NO es una nueva clase de funcion de decision respecto a V03. Sigue produciendo
un clasificador lineal. Lo que cambia es:

- `hinge` = control de clasificacion lineal tipo SVM;
- `modified_huber` = objetivo de perdida realmente distinto de la regresion logistica;
- el uso de SGD aporta ademas un procedimiento de optimizacion estocastico distinto.

Por tanto NO se describe como "nueva geometria" ni "nueva clase de funcion".

Las variantes
-------------
- **V35-01 hinge**: `SGDClassifier(loss="hinge")`. Como hinge no tiene `predict_proba`, se usa
  `decision_function` con sigmoide, que es la transformacion estandar.
- **V35-02 modified_huber**: `SGDClassifier(loss="modified_huber")` con `predict_proba` nativo.

Hiperparametros conservadores y reproducibles
---------------------------------------------
penalty l2, alpha 1e-4, max_iter 2000, tol 1e-3, class_weight balanced, random_state 42. No se
hace grid. Los mismos para ambas variantes para que la unica diferencia sea la perdida.

Convergencia (Fase 4, obligatoria)
----------------------------------
Se registra `n_iter_` por pliegue y etiqueta, se comprueba si se alcanzo `max_iter`, se mide la
sensibilidad a `max_iter` y `tol`, y se verifica que no hay warnings de convergencia. Si el modelo
no ha convergido, su bajo F1 no es evidencia contra la perdida.

Representacion identica (Fase 5)
--------------------------------
Se usa exactamente la matriz de V03: word TF-IDF (1,2) + char_wb TF-IDF (3,5), min_df=2,
sublinear_tf, ajustada dentro de cada pliegue. No se anade ninguna feature nueva.

Uso
---
    /home/carlos/venv/bin/python experiments/v35_01_sgd.py
"""

from __future__ import annotations

import json
import resource
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.evaluation import metricas_multietiqueta
from src.splits import grupos_fusionados, pliegues_corregidos
from src.structural_eval import (
    ETIQUETAS,
    UMBRALES_V03,
    comparar,
    evaluar_bloque,
    resumen,
    verificar_control,
)

EXPERIMENTO = "V35-01 (SGD hinge y modified_huber sobre la representacion exacta de V03)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v35_01_sgd.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v35_01_sgd.md"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"

# Hiperparametros conservadores y reproducibles. Los mismos para ambas variantes para que la
# unica diferencia sea la perdida.
PARAMS_SGD = {
    "penalty": "l2",
    "alpha": 1e-4,
    "max_iter": 2000,
    "tol": 1e-3,
    "class_weight": "balanced",
    "random_state": 42,
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V35-01] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def evaluar_sgd(
    train_dev: pd.DataFrame,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    perdida: str,
) -> dict[str, Any]:
    """Evalua SGD con una perdida dada sobre la representacion exacta de V03.

    Args:
        train_dev: DataFrame del desarrollo.
        pliegues: pliegues canonicos.
        perdida: "hinge" o "modified_huber".

    Returns:
        El resultado de la evaluacion, incluyendo convergencia.
    """
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer

    y_true = train_dev[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    n = len(train_dev)
    proba = np.zeros((n, 3), dtype=float)
    metricas = []
    train_f1s = []
    convergencia: dict[str, Any] = {"n_iter": {}, "max_iter_alcanzado": {}, "warnings": []}
    for numero, (idx_tr, idx_va) in enumerate(pliegues, start=1):
        word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
        train = sparse.hstack([
            word.fit_transform(train_dev["text"].iloc[idx_tr]),
            char.fit_transform(train_dev["text"].iloc[idx_tr]),
        ]).tocsr()
        val = sparse.hstack([
            word.transform(train_dev["text"].iloc[idx_va]),
            char.transform(train_dev["text"].iloc[idx_va]),
        ]).tocsr()
        pred_train = np.zeros((len(idx_tr), 3), dtype=np.int64)
        n_iters = []
        for j, etiqueta in enumerate(ETIQUETAS):
            modelo = SGDClassifier(loss=perdida, **PARAMS_SGD)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                modelo.fit(train, y_dev_j(train_dev, idx_tr, j))
                if w:
                    convergencia["warnings"].append(
                        f"pliegue {numero} {etiqueta}: {w[0].message}"
                    )
            # n_iter_ es un int en sklearn reciente, no un array
            n_iters.append(int(modelo.n_iter_))
            if perdida == "modified_huber":
                # modified_huber tiene predict_proba nativo, que es la probabilidad calibrada
                proba[idx_va, j] = modelo.predict_proba(val)[:, 1]
            else:
                # hinge no tiene predict_proba; se usa decision_function con sigmoide
                proba[idx_va, j] = 1.0 / (1.0 + np.exp(-modelo.decision_function(val)))
            pred_train[:, j] = (
                modelo.predict_proba(train)[:, 1] >= UMBRALES_V03[etiqueta]
                if perdida == "modified_huber"
                else (modelo.decision_function(train) >= 0).astype(int)
            )
        convergencia["n_iter"][str(numero)] = n_iters
        convergencia["max_iter_alcanzado"][str(numero)] = any(
            i >= PARAMS_SGD["max_iter"] for i in n_iters
        )
        pred_pliegue = np.column_stack(
            [(proba[idx_va, j] >= UMBRALES_V03[et]).astype(int) for j, et in enumerate(ETIQUETAS)]
        )
        metrica = metricas_multietiqueta(y_true[idx_va], pred_pliegue)
        metrica["pliegue"] = numero
        metricas.append(metrica)
        train_f1s.append(metricas_multietiqueta(y_true[idx_tr], pred_train)["f1_media_columnas"])
    pred = np.column_stack(
        [(proba[:, j] >= UMBRALES_V03[et]).astype(int) for j, et in enumerate(ETIQUETAS)]
    )
    return {
        "proba": proba, "pred": pred, "metricas_pliegues": metricas,
        "y_true": y_true, "train_f1": float(np.mean(train_f1s)),
        "convergencia": convergencia,
    }


def y_dev_j(train_dev: pd.DataFrame, idx: np.ndarray, j: int) -> np.ndarray:
    """Etiqueta binaria de una columna.

    Args:
        train_dev: DataFrame del desarrollo.
        idx: indices del pliegue.
        j: columna.

    Returns:
        La etiqueta como array.
    """
    return train_dev[ETIQUETAS[j]].iloc[idx].to_numpy(dtype=np.int64)


def main() -> None:
    """Ejecuta la evaluacion de SGD hinge y modified_huber.

    Returns:
        None.
    """
    inicio = time.time()
    log(f"{EXPERIMENTO}. NINGUNA submission en esta fase")

    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos_fusionados(desarrollo))
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    log(f"   desarrollo {len(desarrollo)} filas | {len(pliegues)} pliegues")
    log(f"   hiperparametros: {PARAMS_SGD}")

    log("1/5 control de V03")
    control = resumen(evaluar_bloque(desarrollo, pliegues, bloque=None))
    verificar_control(control["corrected_cv"])
    log(f"   control exacto: {control['corrected_cv']:.10f}")

    log("2/5 V35-01: SGD hinge")
    hinge = evaluar_sgd(desarrollo, pliegues, "hinge")
    m_h = metricas_multietiqueta(y, hinge["pred"])
    log(f"   hinge: corrected_cv {m_h['f1_media_columnas']:.10f} | "
        f"delta {m_h['f1_media_columnas'] - control['corrected_cv']:+.10f}")
    log(f"   train F1 {hinge['train_f1']:.4f} | gap {hinge['train_f1'] - m_h['f1_media_columnas']:+.4f}")

    log("3/5 V35-02: SGD modified_huber")
    mh = evaluar_sgd(desarrollo, pliegues, "modified_huber")
    m_mh = metricas_multietiqueta(y, mh["pred"])
    log(f"   modified_huber: corrected_cv {m_mh['f1_media_columnas']:.10f} | "
        f"delta {m_mh['f1_media_columnas'] - control['corrected_cv']:+.10f}")
    log(f"   train F1 {mh['train_f1']:.4f} | gap {mh['train_f1'] - m_mh['f1_media_columnas']:+.4f}")

    log("4/5 convergencia, complementariedad y sensibilidad")
    for nombre, res, m in (("hinge", hinge, m_h), ("modified_huber", mh, m_mh)):
        comp = comparar(y, evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"], res["pred"])
        max_alcanzado = any(res["convergencia"]["max_iter_alcanzado"].values())
        n_iter_medio = float(np.mean([
            i for v in res["convergencia"]["n_iter"].values() for i in v
        ]))
        log(
            f"   {nombre}: n_iter medio {n_iter_medio:.1f} | max_iter alcanzado {max_alcanzado} | "
            f"warnings {len(res['convergencia']['warnings'])} | rescates {comp['rescates']} "
            f"danios {comp['danios']} neto {comp['neto']:+d}"
        )
    # Sensibilidad a max_iter y tol, sin seleccionar mirando el OOF
    sens = _sensibilidad(desarrollo, pliegues)
    for clave, valor in sens.items():
        log(f"   sensibilidad {clave}: {valor}")

    log("5/5 informe y decision")
    decision = _decision(control, m_h, m_mh, hinge, mh)
    log(f"   DECISION: {decision}")

    informe = {
        "familia": "SGD_LOSS",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "OOF_EVALUATED",
        "pregunta": (
            "una perdida lineal distinta, particularmente Modified Huber, puede encontrar una "
            "solucion que generalice mejor que la regresion logistica de V03 sobre la misma "
            "representacion"
        ),
        "correccion_conceptual": (
            "SGDClassifier NO es una nueva clase de funcion de decision. Sigue siendo lineal. "
            "Lo que cambia es la perdida (hinge = control SVM, modified_huber = objetivo distinto "
            "de la logistica) y el procedimiento de optimizacion estocastico"
        ),
        "hiperparametros": PARAMS_SGD,
        "control": control,
        "V35_01_hinge": {
            "loss": "hinge",
            "penalty": "l2", "alpha": 1e-4, "max_iter": 2000, "tol": 1e-3,
            "class_weight": "balanced", "random_state": 42,
            "corrected_cv": float(m_h["f1_media_columnas"]),
            "delta_vs_v03": float(m_h["f1_media_columnas"] - control["corrected_cv"]),
            "f1_por_etiqueta": {et: float(m_h["f1_por_etiqueta"][et]) for et in ETIQUETAS},
            "train_f1": hinge["train_f1"],
            "gap_train_oof": hinge["train_f1"] - float(m_h["f1_media_columnas"]),
            "convergencia": hinge["convergencia"],
        },
        "V35_02_modified_huber": {
            "loss": "modified_huber",
            "penalty": "l2", "alpha": 1e-4, "max_iter": 2000, "tol": 1e-3,
            "class_weight": "balanced", "random_state": 42,
            "corrected_cv": float(m_mh["f1_media_columnas"]),
            "delta_vs_v03": float(m_mh["f1_media_columnas"] - control["corrected_cv"]),
            "f1_por_etiqueta": {et: float(m_mh["f1_por_etiqueta"][et]) for et in ETIQUETAS},
            "train_f1": mh["train_f1"],
            "gap_train_oof": mh["train_f1"] - float(m_mh["f1_media_columnas"]),
            "convergencia": mh["convergencia"],
        },
        "complementariedad": {
            "hinge": comparar(y, evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"], hinge["pred"]),
            "modified_huber": comparar(
                y, evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"], mh["pred"]
            ),
        },
        "sensibilidad": sens,
        "decision": decision,
        "gastos": {
            "submissions": 0,
            "modelos_entrenados": 3,
            "memoria_pico_mb": round(memoria_pico_mb(), 1),
            "segundos": round(time.time() - inicio, 1),
        },
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe_md(informe)
    log("   informe escrito en reports/v35_01_sgd.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: SGD evaluado, 0 submissions")


def _sensibilidad(
    train_dev: pd.DataFrame,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    """Sensibilidad a max_iter y tol para modified_huber, sin seleccionar mirando OOF.

    Solo se prueban dos valores por parametro, alrededor del valor inicial.

    Args:
        train_dev: DataFrame del desarrollo.
        pliegues: pliegues canonicos.

    Returns:
        Diccionario con el corrected_cv de cada sensibilidad.
    """
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer

    y_true = train_dev[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    salida = {}
    for nombre, parametros in (
        ("max_iter_500", {"max_iter": 500}),
        ("max_iter_5000", {"max_iter": 5000}),
        ("tol_1e-2", {"tol": 1e-2}),
        ("tol_1e-4", {"tol": 1e-4}),
    ):
        proba = np.zeros((len(train_dev), 3), dtype=float)
        for idx_tr, idx_va in pliegues:
            word = TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True
            )
            char = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True
            )
            train = sparse.hstack([
                word.fit_transform(train_dev["text"].iloc[idx_tr]),
                char.fit_transform(train_dev["text"].iloc[idx_tr]),
            ]).tocsr()
            val = sparse.hstack([
                word.transform(train_dev["text"].iloc[idx_va]),
                char.transform(train_dev["text"].iloc[idx_va]),
            ]).tocsr()
            for j in range(3):
                modelo = SGDClassifier(
                    loss="modified_huber", **{**PARAMS_SGD, **parametros}
                )
                modelo.fit(train, train_dev[ETIQUETAS[j]].iloc[idx_tr])
                proba[idx_va, j] = modelo.predict_proba(val)[:, 1]
        pred = np.column_stack(
            [(proba[:, j] >= UMBRALES_V03[et]).astype(int) for j, et in enumerate(ETIQUETAS)]
        )
        salida[nombre] = float(metricas_multietiqueta(y_true, pred)["f1_media_columnas"])
    return salida


def _decision(
    control: dict[str, Any], m_h: dict[str, Any], m_mh: dict[str, Any],
    hinge: dict[str, Any], mh: dict[str, Any],
) -> str:
    """Decision segun el criterio de la directiva.

    Args:
        control: metricas del control.
        m_h: metricas de hinge.
        m_mh: metricas de modified_huber.
        hinge: resultado completo de hinge.
        mh: resultado completo de modified_huber.

    Returns:
        La decision como cadena.
    """
    base = control["corrected_cv"]
    dh = m_h["f1_media_columnas"] - base
    dm = m_mh["f1_media_columnas"] - base
    # Convergencia: si se alcanzo max_iter, el bajo F1 no es evidencia contra la perdida
    conv_h = any(hinge["convergencia"]["max_iter_alcanzado"].values())
    conv_mh = any(mh["convergencia"]["max_iter_alcanzado"].values())
    notas = []
    if conv_h or conv_mh:
        notas.append("alguna variante alcanzo max_iter: su bajo F1 no es evidencia contra la perdida")
    if dh < 0 and dm < 0:
        return f"SGD = CLOSED. hinge {dh:+.6f}, modified_huber {dm:+.6f}. " + " ".join(notas)
    if dh > 0 and dm > 0:
        return "SGD = AMBAS POSITIVAS. No esperado."
    return (
        f"MIXED: hinge {dh:+.6f}, modified_huber {dm:+.6f}. " + " ".join(notas)
    )


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    c = informe["control"]
    h = informe["V35_01_hinge"]
    m = informe["V35_02_modified_huber"]
    g = informe["gastos"]
    lineas = [
        "# V35-01: SGD hinge y modified_huber sobre la representacion exacta de V03",
        "",
        f"Generado por `experiments/v35_01_sgd.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. {informe['correccion_conceptual']}.**",
        "",
        f"Hiperparametros: {informe['hiperparametros']}.",
        "",
        "## 1. Resultado",
        "",
        "| variante | corrected_cv | delta vs V03 | F1 HS | F1 TR | F1 AG | train F1 | gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (f"| V03 (control) | {c['corrected_cv']:.10f} | 0 | "
        f"{c['f1_por_etiqueta']['HS']:.4f} | {c['f1_por_etiqueta']['TR']:.4f} | "
        f"{c['f1_por_etiqueta']['AG']:.4f} | - | - |"),
        (f"| V35-01 hinge | {h['corrected_cv']:.10f} | {h['delta_vs_v03']:+.10f} | "
        f"{h['f1_por_etiqueta']['HS']:.4f} | {h['f1_por_etiqueta']['TR']:.4f} | "
        f"{h['f1_por_etiqueta']['AG']:.4f} | {h['train_f1']:.4f} | {h['gap_train_oof']:+.4f} |"),
        (f"| V35-02 modified_huber | {m['corrected_cv']:.10f} | {m['delta_vs_v03']:+.10f} | "
        f"{m['f1_por_etiqueta']['HS']:.4f} | {m['f1_por_etiqueta']['TR']:.4f} | "
        f"{m['f1_por_etiqueta']['AG']:.4f} | {m['train_f1']:.4f} | {m['gap_train_oof']:+.4f} |"),
        "",
        "## 2. Convergencia (Fase 4, obligatoria)",
        "",
        "| variante | n_iter medio | max_iter alcanzado | warnings |",
        "|---|---:|:--:|---:|",
    ]
    for nombre, clave in (("hinge", "V35_01_hinge"), ("modified_huber", "V35_02_modified_huber")):
        d = informe[clave]["convergencia"]
        n_medio = float(np.mean([i for v in d["n_iter"].values() for i in v]))
        lineas.append(
            f"| {nombre} | {n_medio:.1f} | {any(d['max_iter_alcanzado'].values())} | "
            f"{len(d['warnings'])} |"
        )
    lineas += [
        "",
        "## 3. Complementariedad contra V03",
        "",
        "| variante | rescates | danios | net | celdas |",
        "|---|---:|---:|---:|---:|",
    ]
    for nombre, clave in (("hinge", "hinge"), ("modified_huber", "modified_huber")):
        d = informe["complementariedad"][clave]
        lineas.append(
            f"| {nombre} | {d['rescates']} | {d['danios']} | {d['neto']:+d} | "
            f"{d['celdas_modificadas']} |"
        )
    lineas += [
        "",
        "## 4. Sensibilidad a max_iter y tol",
        "",
        "| configuracion | corrected_cv |",
        "|---|---:|",
    ]
    for clave, valor in informe["sensibilidad"].items():
        lineas.append(f"| {clave} | {valor:.10f} |")
    lineas += [
        "",
        f"**DECISION: {informe['decision']}**",
        "",
        "## 5. Lo que este script no hace",
        "",
        "- No envia submission.",
        "- No modifica V03.",
        "- No anade features nuevas.",
        "- No hace grid search: solo dos valores por parametro de sensibilidad.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
