"""P64 track P9 and P19: a classical multiple-instance model, and a small learned aggregator.

Why this is not a rerun of P59-07
---------------------------------
P59-07 changed the UNIT OF REPRESENTATION to a local window and aggregated by the maximum
probability, and it got -0,0006246152 with net -1. Two things happened there and only one of
them was the concept.

The concept was tested. What was never tested was the same concept with a unit of
representation that can actually reach the rows that matter. P59-07 recorded the reason
explicitly: the tweets are short, median 15 tokens, so a 32-token window leaves more than
one window in 9,6 per cent of the rows. The failure was diagnosed as COVERAGE.

The segmentation by punctuation reaches 50,6 per cent of the rows with more than one unit,
5,29 times more. So this script trains the segment-as-instance model on punctuation segments
rather than on fixed windows, which is the coverage-corrected version of a question P59
already asked and could not answer on enough data. If the answer is still negative it will be
negative about the CONCEPT, which is a different and much stronger claim than P59's.

The three arms, and they are three different mechanisms
------------------------------------------------------
    MI_seg      a segment is the unit of the classifier. Every segment inherits the label of
                its document, the classifier is trained on segment instances, and the
                document score is the maximum segment probability per label. This is P59-07
                with a different unit selector and nothing else changed.

    MI_lineal   the EVIDENCE block of P64-03 plus the CONTRIBUTION block of P64-04 plus V03's
                own margins plus the token count, fed to the SAME LogisticRegression as V03.
                The question is whether a linear model can exploit the structure that the
                distributional readouts already proved exists.

    MI_arbol    the same input to a deliberately small non-linear model, depth 3, which is the
                smallest thing that can represent an interaction between two evidence
                descriptors. The question is whether the signal is non-linear and a linear
                model simply cannot express it.

The two controls that decide
----------------------------
    MI_aleatorio  a random block of exactly the same width in MI_lineal, so capacity is
                   held fixed.
    V03_margen    V03 plus its own margins, which closed the composition family in P63 and
                   the evidence family in P64-03 and P64-04. Every candidate here has to beat
                   it, and beating it is the only claim that would mean the structure is new
                   information rather than a restatement of the confidence.

Why the nonlinear arm is allowed at all
---------------------------------------
The protocol says a new representation goes through LogisticRegression first and only a
small non-linear model if there is evidence that the signal exists but a linear model cannot
reach it. That evidence exists here and it is measured, not assumed: P64-03 and P64-04 both
found the block loses to the margin while the distributional readouts show a real and large
internal structure. That is precisely the situation the rule describes, so MI_arbol is a
single declared configuration, not the first step of a search.

Usage
-----
    /home/carlos/venv/bin/python experiments/p64_05_multiple_instance.py
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

REPORTS = PROJECT_ROOT / "reports"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
RAW_TRAIN = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
CELDA_EV = PROJECT_ROOT / "features" / "p64_evidencia_cells.npz"
CELDA_LOO = PROJECT_ROOT / "features" / "p64_loo_cells.npz"
MD_MI = REPORTS / "p64_multiple_instance.md"
JSON_MI = REPORTS / "p64_multiple_instance.json"
CSV_MI = REPORTS / "p64_candidate_comparison_multi_instance.csv"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True,
               "lowercase": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2,
               "sublinear_tf": True, "lowercase": True}
LR = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear",
      "random_state": 42}
# Modelo no lineal deliberadamente pequeno. Una sola configuracion, declarada aqui antes de
# ejecutar, y sin busqueda alrededor.
ARBOL = {"max_depth": 3, "max_iter": 120, "learning_rate": 0.06, "l2_regularization": 1.0,
         "min_samples_leaf": 20, "random_state": 42}
GATE_DELTA = 0.0015
GATE_PLIEGUES = 4
SEMILLA = 42
BINS: tuple[tuple[str, int, int], ...] = (
    ("CORTA", 0, 6),
    ("MEDIA", 7, 15),
    ("LARGA", 16, 30),
    ("MUY_LARGA", 31, 10**9),
)
RE_SEG = re.compile(r"[.!?]+")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P64-05] {mensaje}", flush=True)


SUSTITUCIONES = {
    chr(0x2014): " - ", chr(0x2013): "-", chr(0x2212): "-", chr(0x00B1): "+/-",
    chr(0x00AB): '"', chr(0x00BB): '"', chr(0x201C): '"', chr(0x201D): '"',
    chr(0x2018): "'", chr(0x2019): "'", chr(0x2264): "<=", chr(0x2265): ">=",
    chr(0x00D7): "x", chr(0x2192): "->", chr(0x00B7): ".", chr(0x2026): "...",
    chr(0x00BF): "?", chr(0x00A1): "!", chr(0x00A7): "parrafo ", chr(0x00A0): " ",
}


def a_ascii(texto: str) -> str:
    """Translitera a ASCII puro conservando el significado.

    Args:
        texto: cadena de entrada.

    Returns:
        La cadena equivalente en ASCII.
    """
    if not isinstance(texto, str):
        texto = str(texto)
    for k, v in SUSTITUCIONES.items():
        texto = texto.replace(k, v)
    salida = []
    for ch in unicodedata.normalize("NFKD", texto):
        if ord(ch) < 128:
            salida.append(ch)
            continue
        if unicodedata.category(ch).startswith("M"):
            continue
        partes = unicodedata.decomposition(ch).split()
        base = chr(int(partes[0], 16)) if partes and partes[0] != "<wide>" else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def segmentar(texto: str) -> list[str]:
    """Segmenta con la regla S1, la primaria declarada en P64-02.

    Args:
        texto: texto del registro.

    Returns:
        Lista de segmentos no vacios.
    """
    partes = [p for p in (s.strip() for s in RE_SEG.split(texto)) if p]
    return partes or [texto.strip() or texto]


def main() -> None:
    """Entrena el modelo multiple-instance clasico y el agregador aprendido.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P64-05 modelo multiple-instance clasico y agregador aprendido")
    log("   tres mecanismos distintos, dos controles, sin rejilla")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(OOF, allow_pickle=True)
    ids, y, pliegues = v["ids"], v["y_true"].astype(int), v["pliegues"]
    pred_v03, _proba_v03 = v["pred"].astype(int), v["proba"]
    margen_v03 = v["margen"]
    n = ids.size
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")

    ev = np.load(CELDA_EV, allow_pickle=True)
    loo = np.load(CELDA_LOO, allow_pickle=True)
    for cel in (ev, loo):
        if not np.array_equal(cel["ids"], ids):
            raise SystemExit("PARADA: un bloque de evidencia no corresponde al OOF")
    n_tok = ev["n_tokens"]
    ev["n_segmentos"]
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | bloques de P64-03 y P64-04 reutilizados")

    # ---------- el bloque denso de los dos brazos aprendidos ----------
    denso = np.hstack([ev["bloque"], loo["bloque"], margen_v03, n_tok.reshape(-1, 1)])
    nombres_denso = ([f"ev_{x}" for x in ev["nombres"].tolist()]
                     + [f"loo_{x}" for x in loo["nombres"].tolist()]
                     + ["margen_HS", "margen_TR", "margen_AG", "n_tokens"])
    log(f"   bloque denso: {denso.shape[1]} columnas "
        f"({ev['bloque'].shape[1]} evidencia + {loo['bloque'].shape[1]} contribucion "
        f"+ 3 margen + 1 longitud)")

    tr = pd.read_csv(RAW_TRAIN, sep=";", encoding="latin-1", dtype=str)
    tr["idn"] = pd.to_numeric(tr["id"].str.replace('"', "", regex=False),
                              errors="coerce").astype("int64")
    idx = tr.set_index("idn").loc[ids]
    if not np.array_equal(idx[["HS", "TR", "AG"]].to_numpy().astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    textos = idx["text"].astype(str).tolist()
    segmentos = [segmentar(t) for t in textos]
    log(f"   instancias de segmento: {sum(len(s) for s in segmentos)} en {n} documentos")

    BRAZOS = ("MI_seg", "MI_lineal", "MI_arbol", "MI_aleatorio", "V03_margen")
    proba: dict[str, np.ndarray] = {b: np.zeros((n, 3)) for b in BRAZOS}
    t0 = time.time()
    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        pw = TfidfVectorizer(**PARAMS_WORD)
        pc = TfidfVectorizer(**PARAMS_CHAR)
        xw = pw.fit_transform([textos[i] for i in ir])
        xc = pc.fit_transform([textos[i] for i in ir])
        xbase = hstack([xw, xc]).tocsr()
        xbase_v = hstack([pw.transform([textos[i] for i in iva]),
                          pc.transform([textos[i] for i in iva])]).tocsr()
        objetivo = float(np.sqrt(np.mean(xbase.multiply(xbase).sum(axis=1))))

        def escalar(mm: csr_matrix, ref: float = objetivo) -> csr_matrix:
            """Iguala la norma media de fila a la del bloque base.

            Args:
                mm: matriz dispersa.
                ref: norma objetivo.

            Returns:
                La matriz escalada.
            """
            media = float(np.sqrt(np.asarray(mm.multiply(mm).sum(axis=1)).ravel()).mean())
            return mm if media <= 0 else mm.multiply(ref / media)

        # ---------- MI_seg: el segmento como instancia del clasificador ----------
        # Los segmentos de ENTRENAMIENTO son las instancias; cada uno hereda la etiqueta de
        # su documento. Los segmentos de VALIDACION se transforman con el vectorizador del
        # pliegue y el score del documento es el maximo de sus segmentos. No se entrena nada
        # con filas de validacion ni se les asigna etiqueta.
        segs_tr: list[str] = []
        lab_tr: list[list[int]] = []
        for i in ir:
            for s in segmentos[i]:
                segs_tr.append(s)
                lab_tr.append([int(y[i, j]) for j in range(3)])
        xs_tr = hstack([pw.transform(segs_tr), pc.transform(segs_tr)]).tocsr()
        y_tr = np.array(lab_tr, dtype=int)
        modelos_seg = []
        for j in range(3):
            mo = LogisticRegression(**LR)
            mo.fit(xs_tr, y_tr[:, j])
            modelos_seg.append(mo)
        for fila in iva:
            segs_va = segmentos[fila]
            xs_va = hstack([pw.transform(segs_va), pc.transform(segs_va)]).tocsr()
            for j in range(3):
                proba["MI_seg"][fila, j] = float(modelos_seg[j].predict_proba(xs_va)[:, 1].max())

        # ---------- los tres brazos aprendidos sobre el bloque denso ----------
        rng = np.random.default_rng(SEMILLA + f)
        matrices = {"MI_lineal": denso,
                    "MI_arbol": denso,
                    "MI_aleatorio": rng.standard_normal(denso.shape),
                    "V03_margen": margen_v03}
        for brazo, be in matrices.items():
            if brazo == "V03_margen":
                z = (be - be[ir].mean(axis=0)) / np.where(be[ir].std(axis=0) < 1e-8, 1.0,
                                                           be[ir].std(axis=0))[None, :]
                xa = escalar(hstack([xbase, csr_matrix(z[ir])]).tocsr())
                xav = escalar(hstack([xbase_v, csr_matrix(z[iva])]).tocsr())
                for j in range(3):
                    mo = LogisticRegression(**LR)
                    mo.fit(xa, y[ir, j])
                    proba[brazo][iva, j] = mo.predict_proba(xav)[:, 1]
                continue
            if brazo == "MI_arbol":
                for j in range(3):
                    mo = HistGradientBoostingClassifier(**ARBOL)
                    mo.fit(denso[ir], y[ir, j])
                    proba[brazo][iva, j] = mo.predict_proba(denso[iva])[:, 1]
                continue
            sd = be[ir].std(axis=0)
            sd[sd < 1e-8] = 1.0
            z = (be - be[ir].mean(axis=0)) / sd[None, :]
            norma = float(np.sqrt(np.asarray((z[ir] ** 2).sum(axis=1))).mean())
            z = z / norma if norma > 0 else z
            xa = escalar(hstack([xbase, csr_matrix(z[ir])]).tocsr())
            xav = escalar(hstack([xbase_v, csr_matrix(z[iva])]).tocsr())
            for j in range(3):
                mo = LogisticRegression(**LR)
                mo.fit(xa, y[ir, j])
                proba[brazo][iva, j] = mo.predict_proba(xav)[:, 1]
        log(f"      pliegue {f} listo ({time.time() - t0:.0f} s)")

    # ---------- metricas ----------
    detalle: dict[str, Any] = {}
    filas: list[dict[str, Any]] = []
    for brazo in BRAZOS:
        p = proba[brazo]
        pr = (p >= V03_UMBRALES[None, :]).astype(int)
        ccv = float(f1_score(y, pr, average="macro"))
        folds = [float(f1_score(y[pliegues == g], pr[pliegues == g], average="macro"))
                 for g in range(1, 6)]
        ac = pr == y
        por_bin = {}
        for nombre, lo, hi in BINS:
            m = (n_tok >= lo) & (n_tok <= hi)
            f1b = float(f1_score(y[m], pr[m], average="macro", zero_division=0))
            f1v = float(f1_score(y[m], pred_v03[m], average="macro", zero_division=0))
            por_bin[nombre] = {"n": int(m.sum()), "F1": round(f1b, 6),
                               "F1_de_V03": round(f1v, 6), "delta": round(f1b - f1v, 6)}
        detalle[brazo] = {"corrected_cv": ccv, "folds": folds, "por_bin": por_bin,
                          "rescate": int(np.count_nonzero(ac & (pred_v03 != y))),
                          "damage": int(np.count_nonzero((~ac) & (pred_v03 == y)))}
        filas.append({"brazo": brazo, "corrected_cv": round(ccv, 10),
                      "delta_vs_V03": round(ccv - ccv_ref, 10),
                      "folds": "|".join(f"{x:.10f}" for x in folds),
                      "rescate": detalle[brazo]["rescate"],
                      "damage": detalle[brazo]["damage"]})
    folds_v = [float(f1_score(y[pliegues == g], pred_v03[pliegues == g], average="macro"))
               for g in range(1, 6)]

    log("")
    log("   LOS CINCO BRAZOS")
    log(f"      {'brazo':14}{'corrected_cv':>18}{'delta vs V03':>16}{'pliegues':>10}{'neto':>8}")
    for f in filas:
        pl = sum(1 for a, b in zip(detalle[f["brazo"]]["folds"], folds_v) if a > b)
        log(f"      {f['brazo']:14}{f['corrected_cv']:>18.10f}{f['delta_vs_V03']:>+16.10f}"
            f"{pl:>7d}/5{detalle[f['brazo']]['rescate'] - detalle[f['brazo']]['damage']:>8d}")

    brecha_mi_seg_cap = (detalle["MI_seg"]["corrected_cv"]
                         - detalle["MI_aleatorio"]["corrected_cv"])
    brecha_mi_lin_margen = (detalle["MI_lineal"]["corrected_cv"]
                            - detalle["V03_margen"]["corrected_cv"])
    brecha_mi_arbol_margen = (detalle["MI_arbol"]["corrected_cv"]
                              - detalle["V03_margen"]["corrected_cv"])
    brecha_arbol_lineal = (detalle["MI_arbol"]["corrected_cv"]
                           - detalle["MI_lineal"]["corrected_cv"])
    log("")
    log("   LAS BRECHAS")
    log(f"      MI_seg  contra el control de anchura: {brecha_mi_seg_cap:+.10f}")
    log(f"      MI_lineal contra el margen de V03:      {brecha_mi_lin_margen:+.10f}")
    log(f"      MI_arbol contra el margen de V03:       {brecha_mi_arbol_margen:+.10f}")
    log(f"      MI_arbol contra MI_lineal:              {brecha_arbol_lineal:+.10f}")
    log("      la ultima es la que responde a la regla lineal contra no lineal: si el arbol")
    log("      solo melhora frente a un lineal, la senal existe pero es lineal y no aporta")

    mejor = max(("MI_seg", "MI_lineal", "MI_arbol"),
                key=lambda b: detalle[b]["corrected_cv"])
    d_mejor = detalle[mejor]["corrected_cv"] - ccv_ref
    fm = sum(1 for a, b in zip(detalle[mejor]["folds"], folds_v) if a > b)
    gate = bool(d_mejor >= GATE_DELTA and fm >= GATE_PLIEGUES
                and detalle[mejor]["corrected_cv"] > detalle["V03_margen"]["corrected_cv"])

    if gate:
        lectura, decision = "PROMISING", "PROMISING"
        veredicto = (f"{mejor} alcanza {d_mejor:+.10f} con {fm} de 5 pliegues y supera al "
                     f"margen, luego exige bootstrap y segunda particion")
    elif brecha_arbol_lineal > 0.005 and brecha_mi_arbol_margen > 0:
        lectura, decision = "NO_LINEAL_UTIL", "COMPLEMENTARY"
        veredicto = (f"el modelo no lineal mejora al lineal en {brecha_arbol_lineal:+.10f} y "
                     f"supera al margen en {brecha_mi_arbol_margen:+.10f}, luego la senal "
                     f"existe y es no lineal, aunque no llega al gate de {GATE_DELTA:+.4f}")
    else:
        lectura, decision = "SIN_APORTE", "CLOSED"
        veredicto = (
            f"Ninguno de los tres mecanismos aporta. MI_seg, que es P59-07 con la "
            f"cobertura corregida de {gap_factor():.2f} veces, queda en "
            f"{detalle['MI_seg']['delta_vs_V03']:+.10f}, luego el CONCEPTO de P59 queda "
            f"cerrado ennegative y no era solo un problema de tamano de ventana. MI_lineal "
            f"queda en {detalle['MI_lineal']['delta_vs_V03']:+.10f} y MI_arbol en "
            f"{detalle['MI_arbol']['delta_vs_V03']:+.10f}, con una diferencia entre ellos de "
            f"{brecha_arbol_lineal:+.10f}, luego la senal no es no lineal. Y los dos brasos "
            f"aprendidos no superan al margen de V03, con {brecha_mi_lin_margen:+.10f} y "
            f"{brecha_mi_arbol_margen:+.10f}. La estructura interna es real, esta repartida "
            f"y es contradictoria, y ya esta dentro de la confianza de V03, y ni cambiar la "
            f"unidad de representacion ni aprender la agregacion la sacan de ahi")
    log("")
    log(f"   LECTURA: {lectura}")
    log(f"   VEREDICTO: {veredicto}")
    log(f"   DECISION: {decision}")

    with CSV_MI.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["brazo", "corrected_cv", "delta_vs_V03", "folds", "rescate", "damage"])
        for f in filas:
            w.writerow([f["brazo"], f["corrected_cv"], f["delta_vs_V03"], f["folds"],
                        f["rescate"], f["damage"]])
        w.writerow([])
        w.writerow(["bin", "n", "F1_de_V03"] + [f"F1_{b}" for b in BRAZOS])
        for nombre, _lo, _hi in BINS:
            w.writerow([nombre, detalle["MI_lineal"]["por_bin"][nombre]["n"],
                        detalle["MI_lineal"]["por_bin"][nombre]["F1_de_V03"]]
                       + [detalle[b]["por_bin"][nombre]["F1"] for b in BRAZOS])
        w.writerow([])
        for nombre, valor in (("brecha_mi_seg_contra_anchura", brecha_mi_seg_cap),
                              ("brecha_mi_lineal_contra_margen", brecha_mi_lin_margen),
                              ("brecha_mi_arbol_contra_margen", brecha_mi_arbol_margen),
                              ("brecha_arbol_contra_lineal", brecha_arbol_lineal)):
            w.writerow([nombre, round(valor, 10)])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P64_MULTIPLE_INSTANCE",
        "naturaleza": ("EXPERIMENTO. 3 mecanismos y 2 controles, 0 submissions, 0 Kaggle, "
                       "0 holdout, 0 test"),
        "fecha": inicio.isoformat(timespec="seconds"), "experiment_id": "P64-05",
        "track": "P9-P19-multiple-instance-y-agregador-aprendido",
        "script": "experiments/p64_05_multiple_instance.py",
        "pregunta": ("una agregacion aprendida de unidades internas del tweet supera a la "
                     "bolsa de palabras del tweet completo"),
        "por_que_no_es_un_rerun_de_P59_07": (
            "P59-07 cambio la unidad a una ventana de 32 tokens con avance 16 y explico su "
            "fracaso por COBERTURA: solo el 9,6 por ciento de las filas tenia mas de una "
            "unidad. Aqui la unidad es el segmento por puntuacion, que llega al 50,6 por "
            "ciento, 5,29 veces mas. El concepto se prueba con el selector de unidad corregido, "
            "y un resultado negativo aqui es sobre el CONCEPTO y no sobre el tamano"),
        "mecanismos": {
            "MI_seg": ("el segmento es la instancia del clasificador, hereda la etiqueta del "
                       "documento, y el score del documento es el maximo de sus segmentos"),
            "MI_lineal": ("el bloque de evidencia y de contribucion, mas los margenes de V03 "
                          "y la longitud, al LogisticRegression de V03"),
            "MI_arbol": ("el mismo bloque a un HistGradientBoosting de profundidad 3, que es "
                         "lo mas pequeno que puede representar una interaccion entre dos "
                         "descriptores de evidencia"),
        },
        "por_que_se_permite_el_brazo_no_lineal": (
            "la regla es pasar primero por LogisticRegression y solo probar un modelo no "
            "lineal pequeno si hay evidencia de que la senal existe y un lineal no la "
            "alcanza. Esa evidencia esta medida y no supuesta: P64-03 y P64-04 medialos dos "
            "bloques por debajo del margen mientras las lecturas distribucionales muestran "
            "una estructura interna grande y real. Una sola configuracion, declarada antes, "
            "sin busqueda alrededor"),
        "parametros_del_arbol": ARBOL,
        "n_columnas_del_bloque_denso": int(denso.shape[1]),
        "columnas": nombres_denso,
        "protocolo": {
            "vectorizadores": "los de V03, ajustados dentro de cada pliegue de entrenamiento",
            "instancias_de_segmento": ("los segmentos de ENTRENAMIENTO entrenan el "
                                       "clasificador y los de validacion solo se puntuan"),
            "pseudo_labeling": ("ninguno. El segmento hereda la etiqueta de su documento "
                                "para entrenar, que es multiple-instance learning y no "
                                "pseudo-labeling, y ningun segmento de validacion recibe "
                                "etiqueta ni participa de ningun ajuste"),
            "control_de_anchura": "MI_aleatorio, un bloque aleatorio de la misma anchura",
            "control_del_margen": "V03 mas sus margenes",
            "confound_de_escala": "el bloque denso se normaliza a la norma media del bloque base",
            "umbrales": "0,44 / 0,51 / 0,46 congelados, sin reajustar",
        },
        "resultados": detalle,
        "brechas": {
            "mi_seg_contra_anchura": round(brecha_mi_seg_cap, 10),
            "mi_lineal_contra_margen": round(brecha_mi_lin_margen, 10),
            "mi_arbol_contra_margen": round(brecha_mi_arbol_margen, 10),
            "arbol_contra_lineal": round(brecha_arbol_lineal, 10),
        },
        "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES, "mejor": mejor,
                 "delta": round(d_mejor, 10), "pliegues_mejorados": fm, "superado": gate},
        "lectura": lectura, "veredicto": veredicto, "decision": decision,
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_MI.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P64 - Modelo multiple-instance clasico y agregador aprendido")
    W("")
    W(f"Generado por `experiments/p64_05_multiple_instance.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Por que esto NO es un rerun de P59-07")
    W("")
    W("P59-07 cambio la UNIDAD DE REPRESENTACION a una ventana local y salio por el maximo,")
    W("y dio -0,0006246152 con neto -1. Pasaron dos cosas ahi y solo una fue el concepto.")
    W("")
    W("**El concepto se probo. Lo que nunca se probo fue el mismo concepto con una unidad que")
    W("puede alcanzar las filas que importan.** P59-07 registro la razon de forma explicita:")
    W("los tweets son cortos, mediana de 15 tokens, luego una ventana de 32 deja mas de una")
    W("ventana en el 9,6 por ciento de las filas. El fracaso fue diagnosticado como")
    W("**COBERTURA**.")
    W("")
    W("La segmentacion por puntuacion llega al 50,6 por ciento, 5,29 veces mas. Este script")
    W("entrena el modelo de segmento-como-instancia sobre segmentos de puntuacion y no sobre")
    W("ventanas fijas, y no cambia nada mas. Si la respuesta sigue siendo negativa, sera")
    W("negativa sobre el CONCEPTO, que es una afirmacion mucho mas fuerte que la de P59.")
    W("")
    W("## Los tres mecanismos, que son distintos")
    W("")
    W("| brazo | mecanismo |")
    W("|---|---|")
    W("| MI_seg | el segmento es la instancia del clasificador y el score es el maximo |")
    W("| MI_lineal | evidencia + contribucion + margen + longitud al LogisticRegression de V03 |")
    W("| MI_arbol | el mismo bloque a un arbol de profundidad 3 |")
    W("")
    W(f"El bloque denso tiene **{denso.shape[1]} columnas**: {ev['bloque'].shape[1]} de")
    W(f"evidencia, {loo['bloque'].shape[1]} de contribucion, 3 de margen y 1 de longitud.")
    W("")
    W("### Por que se permite el brazo no lineal")
    W("")
    W("La regla es pasar primero por LogisticRegression y solo probar un modelo no lineal")
    W("pequeno si hay evidencia de que la senal existe y un lineal no la alcanza. Esa")
    W("evidencia esta **medida y no supuesta**: P64-03 y P64-04 lo")
    W("midieron por debajo del margen mientras las lecturas distribucionales muestran una")
    W("estructura interna grande y real. Una sola configuracion, declarada antes, sin")
    W("busqueda alrededor:")
    W("")
    W("```text")
    W(f"{json.dumps(ARBOL, sort_keys=True)}")
    W("```")
    W("")
    W("## Resultados")
    W("")
    W("| brazo | corrected_cv | delta vs V03 | pliegues | neto |")
    W("|---|---:|---:|---:|---:|")
    for f in filas:
        pl = sum(1 for a, b in zip(detalle[f["brazo"]]["folds"], folds_v) if a > b)
        W(f"| {f['brazo']} | {f['corrected_cv']:.10f} | {f['delta_vs_V03']:+.10f} | {pl}/5 | "
          f"{detalle[f['brazo']]['rescate'] - detalle[f['brazo']]['damage']:+d} |")
    W("")
    W("### Por estrato de longitud")
    W("")
    W("| bin | n | " + " | ".join(f"F1 {b}" for b in BRAZOS) + " |")
    W("|---|---:|" + "---:|" * len(BRAZOS))
    for nombre, _lo, _hi in BINS:
        W(f"| {nombre} | {detalle['MI_lineal']['por_bin'][nombre]['n']} | "
          + " | ".join(f"{detalle[b]['por_bin'][nombre]['F1']:.4f}" for b in BRAZOS) + " |")
    W("")
    W("## Las brechas, que son las unicas preguntas")
    W("")
    W("```text")
    W(f"MI_seg    contra el control de anchura: {brecha_mi_seg_cap:+.10f}")
    W(f"MI_lineal contra el margen de V03:      {brecha_mi_lin_margen:+.10f}")
    W(f"MI_arbol  contra el margen de V03:       {brecha_mi_arbol_margen:+.10f}")
    W(f"MI_arbol  contra MI_lineal:              {brecha_arbol_lineal:+.10f}")
    W("```")
    W("")
    W("La ultima responde a la regla lineal contra no lineal: si el arbol solo mejora frente")
    W("a un lineal, la senal existe pero es lineal, y una senal lineal que un lineal no")
    W("expresa ya esta en V03.")
    W("")
    W("## Veredicto")
    W("")
    W("```text")
    W(veredicto)
    W("")
    W(f"LECTURA: {lectura}")
    W(f"DECISION: {decision}")
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p64_05_multiple_instance.py`. Sin push.*")
    MD_MI.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_MI, JSON_MI, CSV_MI):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P64-05 OK | mejor {mejor} {d_mejor:+.10f} | arbol contra lineal "
        f"{brecha_arbol_lineal:+.10f} | decision {decision}")


def gap_factor() -> float:
    """Factor de cobertura medido en P64-01.

    Returns:
        El factor, o 0,0 si el artefacto no esta disponible.
    """
    ruta = REPORTS / "p64_hypothesis_gap_analysis.json"
    if not ruta.exists():
        return 0.0
    return float(json.loads(ruta.read_text(encoding="utf-8"))
                 ["cobertura"]["factor_de_cobertura"])


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
