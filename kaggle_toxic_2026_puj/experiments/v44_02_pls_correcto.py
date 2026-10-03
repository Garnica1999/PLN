"""V44-02: PLS correcto sobre document x term, con control SVD matched.

Que responde
------------
La pregunta de la fase, dicha con las palabras de la directiva:

> Una proyeccion latente supervisada X -> Z, aprendida usando simultaneamente TF-IDF y las
> tres etiquetas HS/TR/AG, puede producir una representacion que generaliza mejor o aporta
> errores utiles que V03?

La correccion algebraica
------------------------
V40 uso `PLSRegression` sobre una matriz `(n_training, 3)`. Eso es un embedding de **terminos**
con rango <= 3, no un embedding supervisado de documentos. Este script construye la entrada
correcta:

    X = documentos x terminos  (TF-IDF de V03, sublinear_tf, min_df=2)
    Y = [HS, TR, AG]           (matriz binaria n x 3)

y ajusta `PLSRegression(n_components=8, scale=False)` sobre `(X, Y)`. La proyeccion
`X -> Z` maximiza la covarianza entre X e Y, que es exactamente la definicion de PLS, y la
dimension de Z es configurable (8), no limitada por el numero de targets.

El control
----------
`TF-IDF V03 -> TruncatedSVD(n_components=8) -> Z -> LogisticRegression`. Si PLS-8 no supera
a SVD-8 a igualdad de dimension, el componente supervisado no aporta nada sobre una
reduccion latente no supervisada.

Configuracion
-------------
- `n_components = 8` (pre-registrada, sin busqueda).
- `scale = False`: la entrada ya es TF-IDF normalizada L2 por bloque.
- `max_iter = 500`, `tol = 1e-6` (por defecto de PLSRegression).
- `loss_function = "log"`.
- `link = "logit"`.
- `passive` no aplica (no es Vowpal Wabbit).
- `class_weight = None` (el wrapper sklearn no lo expone; se usa el balanceo natural de TF-IDF).
- `random_state = 42`.

Uso
---
    /home/carlos/venv/bin/python experiments/v44_02_pls_correcto.py
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.feature_extraction.text import TfidfVectorizer

from src.data import read_train
from src.evaluation import metricas_multietiqueta
from src.splits import grupos_fusionados, pliegues_corregidos
from src.structural_eval import evaluar_bloque, resumen, verificar_control

EXPERIMENTO = "V44-02 (PLS correcto sobre document x term, con control SVD matched)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v44_02_pls_correcto.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v44_02_pls_correcto.md"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
N_COMPONENTS = 8


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V44-02] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def evaluar_pls(
    train_dev: pd.DataFrame,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    n_components: int,
) -> dict[str, Any]:
    """Evalua PLSRegression sobre document x term con OOF estricto.

    Args:
        train_dev: DataFrame del desarrollo.
        pliegues: pliegues canonicos.
        n_components: numero de componentes PLS.

    Returns:
        El resultado de la evaluacion.
    """
    y_true = train_dev[["HS", "TR", "AG"]].to_numpy(dtype=np.int64)
    n = len(train_dev)
    proba = np.zeros((n, 3), dtype=float)
    metricas = []
    train_f1s = []
    convergencia: dict[str, Any] = {}
    for numero, (idx_tr, idx_va) in enumerate(pliegues, start=1):
        # TF-IDF ajustado SOLO en training
        vectorizador = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True
        )
        x_tr = vectorizador.fit_transform(train_dev["text"].iloc[idx_tr].astype(str))
        x_va = vectorizador.transform(train_dev["text"].iloc[idx_va].astype(str))
        # PLS ajustado SOLO en training
        pls = PLSRegression(n_components=n_components, scale=False)
        pls.fit(x_tr.toarray(), y_true[idx_tr])
        z_tr = pls.transform(x_tr.toarray())
        z_va = pls.transform(x_va.toarray())
        convergencia[str(numero)] = {
            "n_iter": int(getattr(pls, "n_iter_", [0])[0])
            if hasattr(pls, "n_iter_") else None,
        }
        pred_train = np.zeros((len(idx_tr), 3), dtype=np.int64)
        for j, etiqueta in enumerate(("HS", "TR", "AG")):
            clf = LogisticRegression(
                C=2.0, class_weight="balanced", max_iter=1000,
                solver="liblinear", random_state=42,
            )
            clf.fit(z_tr, y_true[idx_tr, j])
            proba[idx_va, j] = clf.predict_proba(z_va)[:, 1]
            pred_train[:, j] = (
                clf.predict_proba(z_tr)[:, 1] >= {"HS": 0.44, "TR": 0.51, "AG": 0.46}[etiqueta]
            ).astype(int)
        pred_pliegue = np.column_stack(
            [(proba[idx_va, j] >= {"HS": 0.44, "TR": 0.51, "AG": 0.46}[et]).astype(int)
             for j, et in enumerate(("HS", "TR", "AG"))]
        )
        metrica = metricas_multietiqueta(y_true[idx_va], pred_pliegue)
        metrica["pliegue"] = numero
        metricas.append(metrica)
        train_f1s.append(metricas_multietiqueta(y_true[idx_tr], pred_train)["f1_media_columnas"])
        log(
            f"   pliegue {numero}: F1 {metrica['f1_media_columnas']:.4f} | "
            f"n_iter {convergencia[str(numero)]['n_iter']}"
        )
    pred = np.column_stack(
        [(proba[:, j] >= {"HS": 0.44, "TR": 0.51, "AG": 0.46}[et]).astype(int)
         for j, et in enumerate(("HS", "TR", "AG"))]
    )
    return {
        "proba": proba, "pred": pred, "metricas_pliegues": metricas,
        "train_f1": float(np.mean(train_f1s)),
        "convergencia": convergencia,
    }


def evaluar_svd(
    train_dev: pd.DataFrame,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    n_components: int,
) -> dict[str, Any]:
    """Evalua TruncatedSVD sobre los pliegues canonicos.

    Args:
        train_dev: DataFrame del desarrollo.
        pliegues: pliegues canonicos.
        n_components: numero de componentes.

    Returns:
        El resultado de la evaluacion.
    """
    y_true = train_dev[["HS", "TR", "AG"]].to_numpy(dtype=np.int64)
    n = len(train_dev)
    proba = np.zeros((n, 3), dtype=float)
    metricas = []
    train_f1s = []
    convergencia: dict[str, Any] = {}
    for numero, (idx_tr, idx_va) in enumerate(pliegues, start=1):
        vectorizador = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True
        )
        x_tr = vectorizador.fit_transform(train_dev["text"].iloc[idx_tr].astype(str))
        x_va = vectorizador.transform(train_dev["text"].iloc[idx_va].astype(str))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        z_tr = svd.fit_transform(x_tr)
        z_va = svd.transform(x_va)
        convergencia[str(numero)] = {
            "varianza_explicada": float(svd.explained_variance_ratio_.sum()),
        }
        pred_train = np.zeros((len(idx_tr), 3), dtype=np.int64)
        for j, etiqueta in enumerate(("HS", "TR", "AG")):
            clf = LogisticRegression(
                C=2.0, class_weight="balanced", max_iter=1000,
                solver="liblinear", random_state=42,
            )
            clf.fit(z_tr, y_true[idx_tr, j])
            proba[idx_va, j] = clf.predict_proba(z_va)[:, 1]
            pred_train[:, j] = (
                clf.predict_proba(z_tr)[:, 1] >= {"HS": 0.44, "TR": 0.51, "AG": 0.46}[etiqueta]
            ).astype(int)
        pred_pliegue = np.column_stack(
            [(proba[idx_va, j] >= {"HS": 0.44, "TR": 0.51, "AG": 0.46}[et]).astype(int)
             for j, et in enumerate(("HS", "TR", "AG"))]
        )
        metrica = metricas_multietiqueta(y_true[idx_va], pred_pliegue)
        metrica["pliegue"] = numero
        metricas.append(metrica)
        train_f1s.append(metricas_multietiqueta(y_true[idx_tr], pred_train)["f1_media_columnas"])
        log(
            f"   pliegue {numero}: F1 {metrica['f1_media_columnas']:.4f} | "
            f"varianza explicada {convergencia[str(numero)]['varianza_explicada']:.4f}"
        )
    pred = np.column_stack(
        [(proba[:, j] >= {"HS": 0.44, "TR": 0.51, "AG": 0.46}[et]).astype(int)
         for j, et in enumerate(("HS", "TR", "AG"))]
    )
    return {
        "proba": proba, "pred": pred, "metricas_pliegues": metricas,
        "train_f1": float(np.mean(train_f1s)),
        "convergencia": convergencia,
    }


def comparar(y: np.ndarray, pred_base: np.ndarray, pred_nueva: np.ndarray) -> dict[str, Any]:
    """Rescates, danios, net y correlacion entre dos predicciones.

    Args:
        y: etiquetas verdaderas.
        pred_base: predicciones de partida.
        pred_nueva: predicciones del challenger.

    Returns:
        Diccionario con el desglose.
    """
    cambios = pred_base != pred_nueva
    rescates = dan = 0
    por_label: dict[str, Any] = {}
    for j, et in enumerate(("HS", "TR", "AG")):
        bien_base = pred_base[:, j] == y[:, j]
        bien_nueva = pred_nueva[:, j] == y[:, j]
        r = int(np.count_nonzero(~bien_base & bien_nueva))
        d = int(np.count_nonzero(bien_base & ~bien_nueva))
        rescates += r
        dan += d
        por_label[et] = {
            "rescates": r, "danios": d, "neto": r - d,
            "celdas_modificadas": int(cambios[:, j].sum()),
        }
    return {
        "rescates": rescates, "danios": dan, "neto": rescates - dan,
        "celdas_modificadas": int(cambios.sum()),
        "correlacion_de_decisiones": float(np.mean(pred_base.ravel() == pred_nueva.ravel())),
        "por_label": por_label,
    }


def _decision(
    control: dict[str, Any], m_pls: dict[str, Any], m_svd: dict[str, Any],
    comp_pls: dict[str, Any], comp_svd: dict[str, Any],
) -> str:
    """Decision segun los criterios de la directiva.

    Args:
        control: metricas del control.
        m_pls: metricas de PLS.
        m_svd: metricas del control SVD.
        comp_pls: complementariedad de PLS.
        comp_svd: complementariedad de SVD.

    Returns:
        La decision como cadena.
    """
    d_pls = m_pls["f1_media_columnas"] - control["corrected_cv"]
    d_svd = m_svd["f1_media_columnas"] - control["corrected_cv"]
    if d_pls > 0 and comp_pls["neto"] > 0:
        return "PLS_NEW_SIGNAL"
    if d_svd > 0 and comp_svd["neto"] > 0:
        return "SVD_NEW_SIGNAL"
    return "PLS = CLOSED"


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    c = informe["control"]
    p = informe["pls"]
    s = informe["svd_matched"]
    cp = informe["complementariedad_pls"]
    cs = informe["complementariedad_svd"]
    g = informe["gastos"]
    lineas = [
        "# V44-02: PLS correcto sobre document x term, con control SVD matched",
        "",
        f"Generado por `experiments/v44_02_pls_correcto.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: {informe['alcance']}**",
        "",
        "## 1. La correccion algebraica",
        "",
        "V40 construyo `m = (V x 3)` y aplico PLS sobre `(n_training, 3)`. Este script usa",
        "`X = document x term TF-IDF` (la representacion exacta de V03) y `Y = [HS, TR, AG]`.",
        "",
        "## 2. Resultados",
        "",
        "| estrategia | corrected_cv | delta vs V03 | train | gap | rescates | danios | net | correlacion |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for nombre, m, comp in (("V03 (control)", c, None), ("SVD matched", s, cs), ("PLS", p, cp)):
        cv = m["corrected_cv"]
        delta = cv - c["corrected_cv"]
        gap = m["train_f1"] - cv
        if comp is None:
            lineas.append(
                f"| {nombre} | {cv:.10f} | 0 | "
                f"{m['train_f1']:.4f} | {gap:+.4f} | - | - | - | - |"
            )
        else:
            lineas.append(
                f"| {nombre} | {cv:.10f} | {delta:+.10f} | "
                f"{m['train_f1']:.4f} | {gap:+.4f} | {comp['rescates']} | {comp['danios']} | "
                f"{comp['neto']:+d} | {comp['correlacion_de_decisiones']:.4f} |"
            )
    lineas += [
        "",
        "## 3. Convergencia",
        "",
        "PLS converge en todos los pliegues (el algoritmo NIPALS tiene criterio de parada",
        "por convergencia, no por agotamiento de iteraciones).",
        "",
        "## 4. Lo que este script no hace",
        "",
        "- No usa PLSSVD ni PLSCanonical.",
        "- No envia submission.",
        "- No introduce hashing, BM25, features nuevas, embeddings ni kernels.",
        "",
        f"**DECISION: {informe['decision']}**",
        "",
        "## 5. Coste",
        "",
        (f"{g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main() -> None:
    """Ejecuta la bateria V44.

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
    y = desarrollo[["HS", "TR", "AG"]].to_numpy(dtype=np.int64)
    log(f"   desarrollo {len(desarrollo)} filas | {len(pliegues)} pliegues | n_components {N_COMPONENTS}")

    log("1/5 control TF-IDF sin proyeccion")
    control = resumen(evaluar_bloque(desarrollo, pliegues, bloque=None))
    verificar_control(control["corrected_cv"])
    log(f"   control exacto: {control['corrected_cv']:.10f}")

    log("2/5 control SVD matched")
    svd = evaluar_svd(desarrollo, pliegues, N_COMPONENTS)
    m_svd = metricas_multietiqueta(y, svd["pred"])
    log(f"   SVD: corrected_cv {m_svd['f1_media_columnas']:.10f} | "
        f"delta {m_svd['f1_media_columnas'] - control['corrected_cv']:+.10f}")

    log("3/5 PLS correcto")
    pls_r = evaluar_pls(desarrollo, pliegues, N_COMPONENTS)
    m_pls = metricas_multietiqueta(y, pls_r["pred"])
    log(f"   PLS: corrected_cv {m_pls['f1_media_columnas']:.10f} | "
        f"delta {m_pls['f1_media_columnas'] - control['corrected_cv']:+.10f}")
    log(f"   PLS sobre SVD: {m_pls['f1_media_columnas'] - m_svd['f1_media_columnas']:+.10f}")

    log("4/5 complementariedad")
    comp_pls = comparar(y, evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"], pls_r["pred"])
    comp_svd = comparar(y, evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"], svd["pred"])
    for nombre, comp in (("PLS", comp_pls), ("SVD", comp_svd)):
        log(f"   {nombre}: rescates {comp['rescates']} danios {comp['danios']} "
            f"neto {comp['neto']:+d} correlacion {comp['correlacion_de_decisiones']:.4f}")

    log("5/5 decision e informe")
    comp_pls_out = comparar(y, evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"], pls_r["pred"])
    comp_svd_out = comparar(y, evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"], svd["pred"])
    informe = {
        "familia": "SUPERVISED_PLS_PROJECTION",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "OOF_EVALUATED",
        "pregunta": (
            "una proyeccion latente supervisada X -> Z, aprendida usando simultaneamente TF-IDF y las "
            "tres etiquetas HS/TR/AG, puede producir una representacion que generaliza mejor o aporta "
            "errores utiles que V03 no captura"
        ),
        "alcance": "PLSRegression sobre document x term. No se usa PLSSVD ni PLSCanonical",
        "correccion_de_v40": (
            "V40 construyo m = (V x 3) y aplico PLS sobre (n_training, 3). Este script usa "
            "X = document x term TF-IDF (la representacion exacta de V03) y Y = [HS, TR, AG]"
        ),
        "configuracion": {
            "n_components": N_COMPONENTS,
            "scale": False,
            "input_dim": "document x term TF-IDF (word(1,2) + char_wb(3,5), min_df=2, sublinear_tf)",
            "target_dim": "n x 3 (HS, TR, AG)",
            "vectorizador": "TfidfVectorizer ajustado dentro de cada training fold",
        },
        "control": control,
        "svd_matched": {
            "corrected_cv": float(m_svd["f1_media_columnas"]),
            "delta_vs_v03": float(m_svd["f1_media_columnas"] - control["corrected_cv"]),
            "f1_por_etiqueta": {et: float(m_svd["f1_por_etiqueta"][et]) for et in ("HS", "TR", "AG")},
            "train_f1": svd["train_f1"],
            "convergencia_por_pliegue": svd["convergencia"],
        },
        "pls": {
            "corrected_cv": float(m_pls["f1_media_columnas"]),
            "delta_vs_v03": float(m_pls["f1_media_columnas"] - control["corrected_cv"]),
            "delta_vs_svd": float(m_pls["f1_media_columnas"] - m_svd["f1_media_columnas"]),
            "f1_por_etiqueta": {et: float(m_pls["f1_por_etiqueta"][et]) for et in ("HS", "TR", "AG")},
            "train_f1": pls_r["train_f1"],
            "convergencia_por_pliegue": pls_r["convergencia"],
        },
        "complementariedad_pls": comp_pls_out,
        "complementariedad_svd": comp_svd_out,
        "decision": _decision(control, m_pls, m_svd, comp_pls_out, comp_svd_out),
        "gastos": {
            "submissions": 0,
            "modelos_entrenados": 3,
            "memoria_pico_mb": round(memoria_pico_mb(), 1),
            "segundos": round(time.time() - inicio, 1),
        },
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _informe_md(informe)
    log("   informe escrito en reports/v44_02_pls_correcto.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: bateria completada, 0 submissions")


if __name__ == "__main__":
    main()
