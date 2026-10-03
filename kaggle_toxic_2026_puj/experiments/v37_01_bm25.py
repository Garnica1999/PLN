"""V37-01: BM25 sobre el bloque word de V03, con char TF-IDF intacto.

Que responde
------------
La pregunta de la fase, dicha con las palabras de la directiva:

> Cambiar exclusivamente el weighting de los word n-grams desde TF-IDF hacia BM25, manteniendo
> char TF-IDF y el clasificador de V03, mejora la generalizacion aprovechando mejor la longitud y
> la frecuencia de terminos?

La formula
----------
BM25 clasico de Robertson, con los parametros estandar `k1 = 1.2` y `b = 0.75`:

    idf  = log(1 + (N - df + 0.5) / (df + 0.5))
    tf'  = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

donde `tf` es el conteo del termino, `dl` la longitud del documento en terminos, `avgdl` la longitud
media del training fold, `df` el numero de documentos del training fold que contienen el termino y `N`
el numero de documentos del training fold.

**Todas las estadisticas se calculan exclusivamente dentro del training fold.** Para validation se
usan las estadisticas aprendidas en training, se calcula la longitud de cada documento de validation y
se transforma con los parametros de training. Ninguna estadistica de validation influye.

Lo que se mantiene intacto
--------------------------
- char_wb TF-IDF (3,5), min_df=2, sublinear_tf, norm l2.
- LogisticRegression C=2.0, class_weight balanced, liblinear, max_iter 1000, random_state 42.
- umbrales congelados 0.44, 0.51, 0.46.
- los pliegues canonicos de V03.

Unico cambio sustancial: el weighting del bloque word.

Uso
---
    /home/carlos/venv/bin/python experiments/v37_01_bm25.py
"""

from __future__ import annotations

import json
import math
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.evaluation import metricas_multietiqueta
from src.splits import grupos_fusionados, pliegues_corregidos

EXPERIMENTO = "V37-01 (BM25 sobre el bloque word de V03, con char TF-IDF intacto)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v37_01_bm25.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v37_01_bm25.md"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
K1 = 1.2
B = 0.75
UMBRALES = {"HS": 0.44, "TR": 0.51, "AG": 0.46}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V37-01] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def bm25_transformar(
    matriz: sparse.csr_matrix,
    avgdl: float,
    idf: np.ndarray,
    k1: float = K1,
    b: float = B,
) -> sparse.csr_matrix:
    """Aplica la formula BM25 sobre una matriz de conteos.

    Args:
        matriz: matriz de conteos dispersa.
        avgdl: longitud media de documento del training fold.
        idf: vector de IDF BM25 aprendido en el training fold.
        k1: saturacion de la frecuencia de termino.
        b: normalizacion por longitud de documento.

    Returns:
        La matriz ponderada, normalizada en L2.
    """
    coo = matriz.tocoo().astype(np.float64)
    largos = np.asarray(matriz.sum(axis=1)).ravel()[coo.row]
    largos = np.where(largos <= 0.0, 1.0, largos)
    datos = coo.data
    denominador = datos + k1 * (1.0 - b + b * largos / max(avgdl, 1.0))
    salida = matriz.copy().astype(np.float64)
    salida.data = datos * (k1 + 1.0) / denominador * idf[coo.col]
    return normalize(salida, norm="l2")


def controles_matematicos(k1: float = K1, b: float = B) -> dict[str, Any]:
    """Verifica la formula BM25 con casos pequenos y controlados.

    Args:
        k1: saturacion de la frecuencia de termino.
        b: normalizacion por longitud.

    Returns:
        Diccionario con los resultados de los cuatro controles.
    """
    resultados: dict[str, Any] = {}
    # Caso 1: el mismo documento repetido. Al aumentar tf, el peso debe saturarse y no crecer
    # linealmente. Con k1=1.2, pasar de tf=1 a tf=4 debe multiplicar por menos de 4.
    def peso(tf: float, dl: float, avgdl: float, idf: float,
             k1: float = k1, b: float = b) -> float:
        """Peso BM25 de un termino.

        Args:
            tf: frecuencia del termino.
            dl: longitud del documento.
            avgdl: longitud media.
            idf: factor idf.
            k1: saturacion de la frecuencia de termino.
            b: normalizacion por longitud.

        Returns:
            El peso BM25.
        """
        return idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl / max(avgdl, 1.0)))

    base = peso(1.0, 10.0, 10.0, 2.0)
    saturacion = peso(4.0, 10.0, 10.0, 2.0) / base
    resultados["1_saturacion_tf"] = {
        "peso_tf1": base,
        "peso_tf4": peso(4.0, 10.0, 10.0, 2.0),
        "multiplicador": saturacion,
        "esperado": "menor que 4: la TF se satura y no crece linealmente",
        "cumple": saturacion < 4.0,
    }
    # Caso 2: documento corto frente a largo con el mismo termino. El largo debe pesar menos por
    # termino, porque la normalizacion por longitud lo penaliza.
    corto = peso(1.0, 5.0, 20.0, 2.0)
    largo = peso(1.0, 40.0, 20.0, 2.0)
    resultados["2_longitud"] = {
        "peso_corto_dl5": corto,
        "peso_largo_dl40": largo,
        "esperado": "el documento largo pesa menos por termino",
        "cumple": largo < corto,
    }
    # Caso 3: aumento de tf del mismo termino. Debe crecer pero no linealmente.
    tf1 = peso(1.0, 10.0, 10.0, 2.0)
    tf2 = peso(2.0, 10.0, 10.0, 2.0)
    resultados["3_crecimiento_tf"] = {
        "de_tf1_a_tf2": tf2 / tf1,
        "esperado": "entre 1 y 2: crece pero no linealmente",
        "cumple": 1.0 < tf2 / tf1 < 2.0,
    }
    # Caso 4: termino raro frente a termino frecuente. El raro debe tener mayor idf.
    idf_raro = math.log(1.0 + (100 - 1 + 0.5) / (1 + 0.5))
    idf_frecuente = math.log(1.0 + (100 - 90 + 0.5) / (90 + 0.5))
    resultados["4_idf"] = {
        "idf_raro_df1": idf_raro,
        "idf_frecuente_df90": idf_frecuente,
        "esperado": "el termino raro tiene mayor idf",
        "cumple": idf_raro > idf_frecuente,
    }
    # Caso 5: comportamiento de k1 y b. k1 alto satura menos; b alto penaliza mas la longitud.
    # k1 alto satura menos: con k1=4, pasar de tf=1 a tf=4 multiplica por menos que con k1=1.2
    base_k1_alto = peso(1.0, 10.0, 10.0, 2.0, k1=4.0)
    sat_k1_alto = peso(4.0, 10.0, 10.0, 2.0, k1=4.0) / base_k1_alto
    # b alto penaliza mas los largos: con b=0.75, un documento de dl=40 con avgdl=20 pesa menos
    # que con b=0 (sin normalizacion por longitud)
    penal_b_alto = peso(1.0, 40.0, 20.0, 2.0, b=0.75)
    penal_b_cero = peso(1.0, 40.0, 20.0, 2.0, b=0.0)
    resultados["5_k1_y_b"] = {
        "k1_alto_satura_menos": sat_k1_alto < 2.0,
        "multiplicador_con_k1_alto": sat_k1_alto,
        "b_alto_penaliza_largos": penal_b_alto < penal_b_cero,
        "nota": "k1 controla la saturacion de TF; b controla la penalizacion por longitud",
    }
    return resultados


def evaluar_bm25(
    train_dev: pd.DataFrame,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    k1: float = K1,
    b: float = B,
) -> dict[str, Any]:
    """Evalua word BM25 + char TF-IDF sobre los pliegues canonicos.

    Args:
        train_dev: DataFrame del desarrollo.
        pliegues: pliegues canonicos.
        k1: saturacion de la frecuencia de termino.
        b: normalizacion por longitud.

    Returns:
        El resultado de la evaluacion.
    """
    y_true = train_dev[["HS", "TR", "AG"]].to_numpy(dtype=np.int64)
    n = len(train_dev)
    proba = np.zeros((n, 3), dtype=float)
    metricas = []
    train_f1s = []
    stats_por_pliegue: dict[str, Any] = {}
    for numero, (idx_tr, idx_va) in enumerate(pliegues, start=1):
        # Bloque word con BM25, calculado exclusivamente con el training fold.
        conteo_word = CountVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, lowercase=True
        )
        mw_tr = conteo_word.fit_transform(train_dev["text"].iloc[idx_tr])
        mw_va = conteo_word.transform(train_dev["text"].iloc[idx_va])
        largos_tr = np.asarray(mw_tr.sum(axis=1)).ravel()
        avgdl = float(largos_tr.mean()) or 1.0
        df = np.asarray((mw_tr > 0).sum(axis=0)).ravel()
        n_docs = mw_tr.shape[0]
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        w_tr = bm25_transformar(mw_tr, avgdl, idf, k1, b)
        w_va = bm25_transformar(mw_va, avgdl, idf, k1, b)
        # Bloque char con TF-IDF intacto.
        char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True
        )
        c_tr = char.fit_transform(train_dev["text"].iloc[idx_tr])
        c_va = char.transform(train_dev["text"].iloc[idx_va])
        train = sparse.hstack([w_tr, c_tr]).tocsr()
        val = sparse.hstack([w_va, c_va]).tocsr()
        stats_por_pliegue[str(numero)] = {
            "avgdl_training": avgdl,
            "factor_normalizacion_min": float(
                np.min(1.0 - b + b * np.asarray(w_va.sum(axis=1)).ravel() / avgdl)
            ) if b > 0 else 1.0,
            "factor_normalizacion_max": float(
                np.max(1.0 - b + b * np.asarray(w_va.sum(axis=1)).ravel() / avgdl)
            ) if b > 0 else 1.0,
        }
        pred_train = np.zeros((len(idx_tr), 3), dtype=np.int64)
        for j, etiqueta in enumerate(("HS", "TR", "AG")):
            modelo = LogisticRegression(
                C=2.0, class_weight="balanced", max_iter=1000,
                solver="liblinear", random_state=42,
            )
            modelo.fit(train, train_dev[etiqueta].iloc[idx_tr])
            proba[idx_va, j] = modelo.predict_proba(val)[:, 1]
            pred_train[:, j] = (
                modelo.predict_proba(train)[:, 1] >= UMBRALES[etiqueta]
            ).astype(int)
        pred_pliegue = np.column_stack(
            [(proba[idx_va, j] >= UMBRALES[et]).astype(int) for j, et in enumerate(("HS", "TR", "AG"))]
        )
        metrica = metricas_multietiqueta(y_true[idx_va], pred_pliegue)
        metrica["pliegue"] = numero
        metricas.append(metrica)
        train_f1s.append(metricas_multietiqueta(y_true[idx_tr], pred_train)["f1_media_columnas"])
    pred = np.column_stack(
        [(proba[:, j] >= UMBRALES[et]).astype(int) for j, et in enumerate(("HS", "TR", "AG"))]
    )
    return {
        "proba": proba, "pred": pred, "metricas_pliegues": metricas,
        "y_true": y_true, "train_f1": float(np.mean(train_f1s)),
        "stats_por_pliegue": stats_por_pliegue,
    }


def main() -> None:
    """Ejecuta la evaluacion de BM25 sobre el bloque word.

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
    log(f"   desarrollo {len(desarrollo)} filas | {len(pliegues)} pliegues")
    log(f"   k1 {K1} | b {B}")

    log("1/6 controles matematicos")
    controles = controles_matematicos()
    for nombre, d in controles.items():
        if isinstance(d, dict) and "cumple" in d:
            log(f"   {nombre}: cumple {d['cumple']}")

    log("2/6 control TF-IDF sin hashing: debe reproducir V03")
    from src.structural_eval import evaluar_bloque, resumen, verificar_control
    control = resumen(evaluar_bloque(desarrollo, pliegues, bloque=None))
    verificar_control(control["corrected_cv"])
    log(f"   control exacto: {control['corrected_cv']:.10f}")

    log("3/6 OOF con word BM25 + char TF-IDF")
    r = evaluar_bm25(desarrollo, pliegues)
    m = metricas_multietiqueta(y, r["pred"])
    cv = float(m["f1_media_columnas"])
    log(f"   corrected_cv {cv:.10f} | delta {cv - control['corrected_cv']:+.10f}")
    log(f"   train F1 {r['train_f1']:.4f} | gap {r['train_f1'] - cv:+.4f}")
    for et in ("HS", "TR", "AG"):
        log(
            f"      {et}: control {control['f1_por_etiqueta'][et]:.4f} -> bm25 "
            f"{m['f1_por_etiqueta'][et]:.4f} ({m['f1_por_etiqueta'][et] - control['f1_por_etiqueta'][et]:+.4f})"
        )

    log("4/6 complementariedad contra V03")
    pred_v03 = evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"]
    cambios = pred_v03 != r["pred"]
    rescates = dan = 0
    por_label: dict[str, Any] = {}
    for j, et in enumerate(("HS", "TR", "AG")):
        bien_v03 = pred_v03[:, j] == y[:, j]
        bien_bm25 = r["pred"][:, j] == y[:, j]
        rr = int(np.count_nonzero(~bien_v03 & bien_bm25))
        dd = int(np.count_nonzero(bien_v03 & ~bien_bm25))
        rescates += rr
        dan += dd
        por_label[et] = {
            "rescates": rr, "danios": dd, "neto": rr - dd,
            "celdas_modificadas": int(cambios[:, j].sum()),
        }
    comp = {
        "rescates": rescates, "danios": dan, "neto": rescates - dan,
        "celdas_modificadas": int(cambios.sum()),
        "correlacion_de_decisiones": float(np.mean(pred_v03.ravel() == r["pred"].ravel())),
        "por_label": por_label,
    }
    log(
        f"   rescates {rescates} | danios {dan} | neto {rescates - dan:+d} | "
        f"correlacion {comp['correlacion_de_decisiones']:.4f}"
    )

    log("5/6 comportamiento por longitud")
    tokens = np.array([len(t.split()) for t in desarrollo["text"].astype(str)], dtype=float)
    t1, t2 = np.percentile(tokens, [33.33, 66.67])
    regiones = {"corto": tokens <= t1, "medio": (tokens > t1) & (tokens <= t2), "largo": tokens > t2}
    por_longitud: dict[str, Any] = {}
    for nombre, mask in regiones.items():
        mm = metricas_multietiqueta(y[mask], r["pred"][mask])
        mc = metricas_multietiqueta(y[mask], pred_v03[mask])
        por_longitud[nombre] = {
            "n_filas": int(mask.sum()),
            "bm25_macro_f1": float(mm["f1_media_columnas"]),
            "v03_macro_f1": float(mc["f1_media_columnas"]),
            "delta": float(mm["f1_media_columnas"] - mc["f1_media_columnas"]),
        }
        log(
            f"   {nombre:<7} {int(mask.sum()):>4} filas | V03 {mc['f1_media_columnas']:.4f} -> "
            f"bm25 {mm['f1_media_columnas']:.4f} ({mm['f1_media_columnas'] - mc['f1_media_columnas']:+.4f})"
        )

    log("6/6 decision")
    decision = _decision(control, cv, comp, r)
    log(f"   DECISION: {decision}")

    informe = {
        "familia": "BM25_WEIGHTING",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "OOF_EVALUATED",
        "pregunta": (
            "cambiar exclusivamente el weighting de los word n-grams desde TF-IDF hacia BM25, "
            "manteniendo char TF-IDF y el clasificador de V03, mejora la generalizacion "
            "aprovechando mejor la longitud y la frecuencia de terminos"
        ),
        "alcance": (
            "solo cambia el weighting del bloque word. char TF-IDF, clasificador, umbrales y "
            "pliegues intactos"
        ),
        "formula": "idf * tf * (k1+1) / (tf + k1*(1-b+b*dl/avgdl))",
        "parametros": {"k1": K1, "b": B, "idf": "log(1 + (N - df + 0.5)/(df + 0.5))",
                       "document_length": "numero de terminos del documento",
                       "avgdl": "longitud media del training fold"},
        "controles_matematicos": controles,
        "control": control,
        "bm25": {
            "corrected_cv": cv,
            "delta_vs_v03": cv - control["corrected_cv"],
            "f1_por_etiqueta": {et: float(m["f1_por_etiqueta"][et]) for et in ("HS", "TR", "AG")},
            "train_f1": r["train_f1"],
            "gap_train_oof": r["train_f1"] - cv,
            "metricas_por_pliegue": r["metricas_pliegues"],
            "stats_por_pliegue": r["stats_por_pliegue"],
        },
        "complementariedad": comp,
        "por_longitud": por_longitud,
        "evidencia_historica": (
            "experiments/30_supervised_weighting_v4.py ya probo BM25 con resultado 0.7748-0.7755, "
            "por debajo de V03. No era equivalente porque aplicaba BM25 a word y char, pero es "
            "evidencia previa negativa"
        ),
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
    log("   informe escrito en reports/v37_01_bm25.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: BM25 evaluado, 0 submissions")


def _decision(
    control: dict[str, Any], cv: float, comp: dict[str, Any], r: dict[str, Any]
) -> str:
    """Decision segun los criterios de la directiva.

    Args:
        control: metricas del control.
        cv: corrected_cv de BM25.
        comp: complementariedad contra V03.
        r: resultado completo de BM25.

    Returns:
        La decision como cadena.
    """
    delta = cv - control["corrected_cv"]
    gap = r["train_f1"] - cv
    if delta > 0 and comp["neto"] > 0 and gap < 0.15:
        return "NEW_SIGNAL"
    if delta > 0 or comp["neto"] > 0:
        return "WEAK_SIGNAL"
    if abs(delta) < 0.001 and comp["neto"] == 0:
        return "NO_SIGNAL"
    if gap > 0.15 and delta < 0:
        return "OVERFIT"
    return "REJECTED"


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    c = informe["control"]
    b = informe["bm25"]
    comp = informe["complementariedad"]
    g = informe["gastos"]
    lineas = [
        "# V37-01: BM25 sobre el bloque word de V03",
        "",
        f"Generado por `experiments/v37_01_bm25.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: {informe['alcance']}**",
        "",
        f"Formula: `{informe['formula']}`.",
        f"Parametros: {informe['parametros']}.",
        "",
        "## 1. Controles matematicos",
        "",
    ]
    for nombre, d in informe["controles_matematicos"].items():
        if isinstance(d, dict) and "cumple" in d:
            lineas.append(f"- {nombre}: cumple {d['cumple']}")
    lineas += [
        "",
        "## 2. Resultado OOF",
        "",
        "| variante | corrected_cv | delta vs V03 |",
        "|---|---:|---:|",
        f"| V03 (control) | {c['corrected_cv']:.10f} | 0 |",
        f"| word BM25 + char TF-IDF | {b['corrected_cv']:.10f} | {b['delta_vs_v03']:+.10f} |",
        "",
        "| etiqueta | V03 | BM25 | delta |",
        "|---|---:|---:|---:|",
    ]
    for et in ("HS", "TR", "AG"):
        lineas.append(
            f"| {et} | {c['f1_por_etiqueta'][et]:.4f} | {b['f1_por_etiqueta'][et]:.4f} | "
            f"{b['f1_por_etiqueta'][et] - c['f1_por_etiqueta'][et]:+.4f} |"
        )
    lineas += [
        "",
        f"train F1 {b['train_f1']:.4f} | gap {b['gap_train_oof']:+.4f}",
        "",
        "## 3. Complementariedad contra V03",
        "",
        "| label | rescates | danios | neto | celdas |",
        "|---|---:|---:|---:|---:|",
    ]
    for et in ("HS", "TR", "AG"):
        d = comp["por_label"][et]
        lineas.append(
            f"| {et} | {d['rescates']} | {d['danios']} | {d['neto']:+d} | "
            f"{d['celdas_modificadas']} |"
        )
    lineas += [
        "",
        (f"**rescates {comp['rescates']} | danios {comp['danios']} | neto {comp['neto']:+d} | "
        f"correlacion {comp['correlacion_de_decisiones']:.4f}**"),
        "",
        "## 4. Comportamiento por longitud",
        "",
        "| region | filas | V03 | BM25 | delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for nombre, d in informe["por_longitud"].items():
        lineas.append(
            f"| {nombre} | {d['n_filas']} | {d['v03_macro_f1']:.4f} | "
            f"{d['bm25_macro_f1']:.4f} | {d['delta']:+.4f} |"
        )
    lineas += [
        "",
        f"**DECISION: {informe['decision']}**",
        "",
        f"{informe['evidencia_historica']}.",
        "",
        "## 5. Lo que este script no hace",
        "",
        "- No envia submission.",
        "- No hace grid de k1/b.",
        "- No aplica BM25 a char.",
        "- No modifica V03.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
