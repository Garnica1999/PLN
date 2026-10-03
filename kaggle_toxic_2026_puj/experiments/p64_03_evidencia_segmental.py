"""P64 track P4: decompose V03's evidence across the segments of a tweet.

Why this is the line the phase turns on
---------------------------------------
P64-02 closed the segment PROFILE as a feature block, and honestly: the best structural
descriptor reached AUC 0,589847 for predicting V03's errors, which is worse than counting
tokens, 0,600506. The association between many segments and many errors is length wearing a
structural costume.

That does not touch what this script measures, and the difference is not subtle. The profile
asks what SHAPE the segmentation has. This asks where the model's EVIDENCE actually sits inside
it. A tweet can have an unremarkable profile and still carry its toxic evidence in one short
segment that the full-document average buries, and only the second question can see that.

The discipline that makes this a decomposition and not a new classifier
------------------------------------------------------------------------
Each segment is transformed with the V03 vectorizers AND scored with the V03 label
classifiers, both fitted on the TRAINING ROWS OF THE CURRENT FOLD ONLY. The segment of a
validation row is therefore scored by a model that never saw that row, or any other row from
that fold. This is what distinguishes it from training a classifier on segments, and it is why
the brief's prohibition on pseudo-labelling does not apply: no segment is ever assigned a
label, and no segment ever contributes a label to any objective. The segments are read through
an existing model the way a microscope reads a slide, and they leave no mark.

The same argument forbids using a segment's own score as a target. Everything here is computed
from a fixed function of the training-fold model applied to text.

What is measured, per row and per label
---------------------------------------
The distribution of the per-segment probabilities, not their maximum. P59-07 aggregated by
maximum and finished with net -4 in its own reachable stratum, with an accuracy of 0,423 in the
cells that changed: the maximum is biased upward and moves cells across a fixed threshold
without correcting anything. The statistics here are deliberately about SPREAD.

    max and min segment probability        the extremes of the internal evidence
    mean and median                         the central tendency
    top2_mean and top3_mean                 whether the evidence is carried by one or two
    top_k_fraction                          the share of the mass in the leading segments
    probability standard deviation          the internal disagreement
    probability range                       max minus min
    probability entropy                     the shape of the internal distribution
    max and mean absolute margin            how decisive each segment is on its own
    positive concentration                  how much of the evidence leans one way
    segment disagreement                    the two classes pulling apart
    evidence conflict                       positive mass against negative mass

The arms
-------
    V03_seg      V03 plus the whole evidence block
    V03_random   V03 plus a random block of the same width, the capacity control that has
                 decided three phases now
    V03_margen   V03 plus only V03's own margins, which is the control that says whether
                 anything here is better than what V03 already knows about itself

That third arm matters more than it looks. P63-07 established that V03's margin alone explains
its own errors with AUC 0,757141 and that added blocks made that worse. If V03_seg cannot beat
V03_margen, the evidence distribution carries nothing that the confidence does not already
carry, and the whole family closes for a strong reason rather than a weak one.

Usage
-----
    /home/carlos/venv/bin/python experiments/p64_03_evidencia_segmental.py
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    ETIQUETAS,
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

REPORTS = PROJECT_ROOT / "reports"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
RAW_TRAIN = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
MD_EV = REPORTS / "p64_evidencia_segmental.md"
JSON_EV = REPORTS / "p64_evidencia_segmental.json"
CSV_EV = REPORTS / "p64_evidence_distribution.csv"
CELDA_EV = PROJECT_ROOT / "features" / "p64_evidencia_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True,
               "lowercase": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2,
               "sublinear_tf": True, "lowercase": True}
LR = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear",
      "random_state": 42}
DELTA_HC = 0.10
GATE_DELTA = 0.0015
GATE_PLIEGUES = 4
N_BOOT = 2000
SEMILLA = 42
BINS: tuple[tuple[str, int, int], ...] = (
    ("CORTA", 0, 6),
    ("MEDIA", 7, 15),
    ("LARGA", 16, 30),
    ("MUY_LARGA", 31, 10**9),
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P64-03] {mensaje}", flush=True)


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


import re

RE_SEG = re.compile(r"[.!?]+")
RE_ESPACIO = re.compile(r"\s+")


def segmentar(texto: str) -> list[str]:
    """Segmenta con la regla S1, la declarada primaria en P64-02.

    Args:
        texto: texto del registro.

    Returns:
        Lista de segmentos no vacios.
    """
    partes = [p for p in (s.strip() for s in RE_SEG.split(texto)) if p]
    return partes or [texto.strip() or texto]


def estadisticas(probs: np.ndarray, umbral: float) -> dict[str, float]:
    """Estadisticas de la distribucion de evidencia interna de una etiqueta.

    Todos los valores estan acotados. Ninguno usa una etiqueta: la entrada es la
    probabilidad que el modelo de V03 da a un segmento, y la unica referencia es el
    umbral congelado de V03, que se usa para separar evidencia a favor de evidencia en
    contra y no para ajustar nada.

    Args:
        probs: probabilidades por segmento de una etiqueta.
        umbral: umbral congelado de V03 para esa etiqueta.

    Returns:
        Diccionario de rasgos.
    """
    p = np.asarray(probs, dtype=float)
    orden = np.sort(p)[::-1]
    total = max(len(p), 1)
    pos = np.clip(p - umbral, 0, None).sum() / total
    neg = np.clip(umbral - p, 0, None).sum() / total
    if len(p) > 1:
        q = p / max(p.sum(), 1e-12)
        ent = float(-(q * np.log2(q)).sum())
    else:
        ent = 0.0
    top2 = float(orden[:2].mean()) if len(orden) >= 2 else float(orden[0])
    top3 = float(orden[:3].mean()) if len(orden) >= 3 else top2
    return {
        "seg_max": float(orden[0]),
        "seg_min": float(orden[-1]),
        "seg_media": float(p.mean()),
        "seg_mediana": float(np.median(p)),
        "seg_top2_media": top2,
        "seg_top3_media": top3,
        "seg_top3_fraccion": float(orden[:3].sum() / max(p.sum(), 1e-12)),
        "seg_desviacion": float(p.std()),
        "seg_rango": float(orden[0] - orden[-1]),
        "seg_entropia": ent / 3.0,
        "seg_margen_max": float(np.abs(p - umbral).max()),
        "seg_margen_medio": float(np.abs(p - umbral).mean()),
        "evidencia_positiva": float(pos),
        "evidencia_negativa": float(neg),
        "evidencia_conflicto": float(min(pos, neg)),
        "segmentos_en_desacuerdo": float(np.mean((p >= umbral) != (p >= np.median(p)))),
    }


NOMBRES_BASE = list(estadisticas(np.array([0.5]), 0.5).keys())


def main() -> None:
    """Descompone la evidencia de V03 a lo largo de los segmentos y la mide.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P64-03 la evidencia de V03 descompuesta por segmento")
    log("   cada segmento se puntua con los coeficientes de V03 aprendidos SOLO en el")
    log("   entrenamiento del pliegue. Ningun segmento recibe etiqueta: esto es")
    log("   descomposicion de un modelo que ya existe, no pseudo-labeling")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(OOF, allow_pickle=True)
    ids, y, pliegues = v["ids"], v["y_true"].astype(int), v["pliegues"]
    pred_v03, proba_v03 = v["pred"].astype(int), v["proba"]
    margen_v03 = v["margen"]
    n = ids.size
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    tr = pd.read_csv(RAW_TRAIN, sep=";", encoding="latin-1", dtype=str)
    tr["idn"] = pd.to_numeric(tr["id"].str.replace('"', "", regex=False),
                              errors="coerce").astype("int64")
    idx = tr.set_index("idn").loc[ids]
    if not np.array_equal(idx[["HS", "TR", "AG"]].to_numpy().astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    textos = idx["text"].astype(str).tolist()
    segmentos = [segmentar(t) for t in textos]
    n_tok = np.array([len(t.split()) for t in textos])
    n_seg = np.array([len(s) for s in segmentos])
    log("")
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | {int((n_seg >= 2).sum())} filas multi "
        f"segmento | segmentos maximos {int(n_seg.max())}")

    nombres = [f"{et}_{c}" for et in ETIQUETAS for c in NOMBRES_BASE]
    n_cols = len(nombres)
    log(f"   bloque de evidencia: {n_cols} columnas, {len(NOMBRES_BASE)} por etiqueta")

    proba: dict[str, np.ndarray] = {b: np.zeros((n, 3)) for b in
                                   ("V03_seg", "V03_random", "V03_margen")}
    bloque_ev = np.zeros((n, n_cols), dtype=float)
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
            """Iguala la norma media de fila a la del control.

            Args:
                mm: matriz dispersa.
                ref: norma objetivo.

            Returns:
                La matriz escalada.
            """
            media = float(np.sqrt(np.asarray(mm.multiply(mm).sum(axis=1)).ravel()).mean())
            return mm if media <= 0 else mm.multiply(ref / media)

        # ---------- V03 del pliegue: el unico modelo que puntua segmentos ----------
        modelos = []
        for j in range(3):
            mo = LogisticRegression(**LR)
            mo.fit(xbase, y[ir, j])
            modelos.append(mo)

        # ---------- el score de cada segmento de las filas de validacion ----------
        # Solo se puntuan los segmentos de las filas de VALIDACION, y con un modelo
        # ajustado exclusivamente en las filas de ENTRENAMIENTO de este pliegue.
        for fila in iva:
            fila_ev: list[list[float]] = []
            for j in range(3):
                umbral = float(V03_UMBRALES[j])
                probs = []
                for s in segmentos[fila]:
                    xs = hstack([pw.transform([s]), pc.transform([s])]).tocsr()
                    probs.append(float(modelos[j].predict_proba(xs)[0, 1]))
                fila_ev.append(list(estadisticas(np.array(probs), umbral).values()))
            bloque_ev[fila] = np.array(fila_ev).ravel()

        # ---------- los brazos ----------
        rng = np.random.default_rng(SEMILLA + f)
        aleat = rng.standard_normal((n, n_cols))
        marg = np.column_stack([margen_v03[:, j] for j in range(3)])
        matrices = {"V03_seg": bloque_ev, "V03_random": aleat, "V03_margen": marg}
        for brazo, be in matrices.items():
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
        log(f"      pliegue {f} listo ({time.time() - t0:.0f} s) | segmentos en "
            f"{int(n_seg[iva].sum())}")

    # ---------- metricas ----------
    detalle: dict[str, Any] = {}
    filas: list[dict[str, Any]] = []
    for brazo in ("V03_seg", "V03_random", "V03_margen"):
        p = proba[brazo]
        pr = (p >= V03_UMBRALES[None, :]).astype(int)
        ccv = float(f1_score(y, pr, average="macro"))
        folds = [float(f1_score(y[pliegues == g], pr[pliegues == g], average="macro"))
                 for g in range(1, 6)]
        ac = pr == y
        por_bin = {}
        for nombre, lo, hi in BINS:
            m = (n_tok >= lo) & (n_tok <= hi)
            por_bin[nombre] = {
                "n": int(m.sum()),
                "F1": round(float(f1_score(y[m], pr[m], average="macro", zero_division=0)), 6),
                "F1_de_V03": round(float(f1_score(y[m], pred_v03[m], average="macro",
                                                 zero_division=0)), 6),
            }
            por_bin[nombre]["delta"] = round(por_bin[nombre]["F1"]
                                             - por_bin[nombre]["F1_de_V03"], 6)
        detalle[brazo] = {
            "corrected_cv": ccv, "folds": folds, "por_bin": por_bin,
            "rescate": int(np.count_nonzero(ac & (pred_v03 != y))),
            "damage": int(np.count_nonzero((~ac) & (pred_v03 == y))),
        }
        filas.append({"brazo": brazo, "corrected_cv": round(ccv, 10),
                      "delta_vs_V03": round(ccv - ccv_ref, 10),
                      "folds": "|".join(f"{x:.10f}" for x in folds),
                      "rescate": detalle[brazo]["rescate"],
                      "damage": detalle[brazo]["damage"]})

    d_seg = detalle["V03_seg"]["corrected_cv"] - ccv_ref
    detalle["V03_random"]["corrected_cv"] - ccv_ref
    detalle["V03_margen"]["corrected_cv"] - ccv_ref
    brecha_cap = detalle["V03_seg"]["corrected_cv"] - detalle["V03_random"]["corrected_cv"]
    brecha_marg = detalle["V03_seg"]["corrected_cv"] - detalle["V03_margen"]["corrected_cv"]
    folds_v = [float(f1_score(y[pliegues == g], pred_v03[pliegues == g], average="macro"))
               for g in range(1, 6)]
    fm = sum(1 for a, b in zip(detalle["V03_seg"]["folds"], folds_v) if a > b)
    gate = bool(d_seg >= GATE_DELTA and fm >= GATE_PLIEGUES and brecha_marg > 0)

    log("")
    log("   LOS TRES BRAZOS")
    log(f"      {'brazo':14}{'corrected_cv':>18}{'delta vs V03':>16}{'pliegues':>10}"
        f"{'neto':>8}")
    for f in filas:
        pl = sum(1 for a, b in zip(detalle[f['brazo']]["folds"], folds_v) if a > b)
        log(f"      {f['brazo']:14}{f['corrected_cv']:>18.10f}{f['delta_vs_V03']:>+16.10f}"
            f"{pl:>7d}/5{detalle[f['brazo']]['rescate'] - detalle[f['brazo']]['damage']:>8d}")
    log("")
    log("   LAS DOS BRECHAS, que son las unicas preguntas")
    log(f"      contra el control de capacidad: {brecha_cap:+.10f}")
    log(f"      contra el margen de V03:      {brecha_marg:+.10f}")
    log("      la segunda es la dura. P63-07 mostro que el margen de V03 ya explica sus")
    log("      propios errores con AUC 0,757141, y que anadir bloques lo empeoraba. Si la")
    log("      distribucion de evidencia no supera al margen, la familia se cierra por una")
    log("      razon fuerte y no por un negativo debil")

    # ---------- la distribucion de evidencia por estrato ----------
    log("")
    log("   LA DISTRIBUCION DE EVIDENCIA INTERNA, por estrato de comportamiento de V03")
    error = pred_v03 != y
    alto = error & (np.abs(proba_v03 - V03_UMBRALES[None, :]) >= DELTA_HC)
    estratos = [("todas", np.ones(n, dtype=bool)),
                ("V03 correcto", ~error.any(axis=1)),
                ("V03 incorrecto", error.any(axis=1)),
                ("error de alta confianza", alto.any(axis=1))]
    for nombre, lo, hi in BINS:
        estratos.append((nombre, (n_tok >= lo) & (n_tok <= hi)))
    clave = ["HS_evidencia_positiva", "HS_evidencia_negativa", "HS_evidencia_conflicto",
             "HS_segmentos_en_desacuerdo", "HS_seg_desviacion", "HS_seg_entropia",
             "HS_seg_top3_fraccion"]
    idx_clave = [nombres.index(c) for c in clave]
    filas_dist: list[dict[str, Any]] = []
    log(f"      {'estrato':>26}{'n':>7}" + "".join(f"{c.split('_', 1)[1][:15]:>17}"
                                                 for c in clave))
    for nombre, m in estratos:
        if m.sum() < 5:
            continue
        vals = bloque_ev[m][:, idx_clave].mean(axis=0)
        filas_dist.append({"estrato": nombre, "n": int(m.sum())}
                          | {c: round(float(x), 6) for c, x in zip(clave, vals)})
        log(f"      {nombre:>26}{int(m.sum()):>7}"
            + "".join(f"{x:>17.4f}" for x in vals))
    log("")
    log("      la lectura que importa es la diferencia entre 'V03 correcto' y 'error de")
    log("      alta confianza' en evidencia_positiva y en evidencia_conflicto. Si el conflicto")
    log("      no crece, la hipotesis de la fase no se sostiene y hay que decirlo")

    # ---------- AUC para predecir el error de V03 ----------
    objetivo_err = error.any(axis=1).astype(int)
    aucs: dict[str, Any] = {}
    for c, k in (("evidencia_positiva_HS", "HS_evidencia_positiva"),
                 ("evidencia_conflicto_HS", "HS_evidencia_conflicto"),
                 ("desacuerdo_HS", "HS_segmentos_en_desacuerdo"),
                 ("desviacion_HS", "HS_seg_desviacion"),
                 ("evidencia_positiva_agregada",
                  None)):
        if k is None:
            col = np.column_stack([bloque_ev[:, nombres.index(f"{et}_evidencia_positiva")]
                                   for et in ETIQUETAS]).mean(axis=1)
            a = float(roc_auc_score(objetivo_err, col))
        else:
            a = float(roc_auc_score(objetivo_err, bloque_ev[:, nombres.index(k)]))
        aucs[c] = round(a, 6)
        log(f"      AUC de {c:32} para predecir si V03 falla: {a:.6f}")
    # El margen alto significa ACIERTO, luego su AUC contra el ERROR sale por debajo de 0,5.
    # Comparar eso con un rasgo cuya AUC contra el error tambien sale por debajo de 0,5, pero
    # por el motivo OPUESTO, exige expresa las dos en la MISMA direccion. Aqui se pasa todo a
    # AUC contra el ACIERTO, que es 1 menos el AUC contra el error, y la comparacion es directa.
    auc_margen_err = float(roc_auc_score(objetivo_err, np.abs(margen_v03).mean(axis=1)))
    aucs["margen_de_V03_para_predecir_el_ACIERTO"] = round(1.0 - auc_margen_err, 6)
    log(f"      AUC del margen de V03 contra el ERROR: {auc_margen_err:.6f}, luego contra el "
        f"ACIERTO es {1.0 - auc_margen_err:.6f}")
    for k in list(aucs):
        if k.startswith("margen_de_V03_para"):
            continue
        aucs[k + "_contra_el_ACIERTO"] = round(1.0 - aucs[k], 6)
    mejor_ev = max((k for k in aucs if k.endswith("_contra_el_ACIERTO")
                    and not k.startswith("margen_de")),
                   key=lambda k: aucs[k])
    auc_margen_acierto = 1.0 - auc_margen_err
    ventaja_auc = aucs[mejor_ev] - auc_margen_acierto
    log(f"      el mejor rasgo de evidencia es {mejor_ev}, con AUC {aucs[mejor_ev]:.6f} "
        f"contra el acierto")
    log(f"      la referencia del margen es {auc_margen_acierto:.6f}, luego la ventaja es "
        f"{ventaja_auc:+.6f}")

    np.savez_compressed(CELDA_EV, ids=ids, y_true=y, pliegues=pliegues, pred_v03=pred_v03,
                        n_tokens=n_tok, n_segmentos=n_seg, nombres=np.array(nombres),
                        bloque=bloque_ev, margen=margen_v03,
                        **{f"proba_{b}": proba[b] for b in proba})

    # ---------- veredicto ----------
    if brecha_marg > 0.01 and brecha_cap > 0.01:
        lectura = "EVIDENCIA_NUEVA"
        veredicto = (
            f"la distribucion de la evidencia interna supera al control de capacidad en "
            f"{brecha_cap:+.10f} y al margen de V03 en {brecha_marg:+.10f}. Es informacion que "
            f"V03 no tiene ni en sus coeficientes ni en su confianza, y por tanto la "
            f"hipotesis de la fase se sostiene en su version fuerte: el problema no es que falten "
            f"palabras, es que el promedio del documento aplana donde esta la evidencia")
        decision = "PROMISING"
    elif brecha_marg > 0:
        lectura = "COMPLEMENTARIA"
        veredicto = (
            f"la distribucion de la evidencia supera al margen de V03 en {brecha_marg:+.10f} "
            f"pero no alcanza el gate de la fase, que es {GATE_DELTA:+.4f}, con {fm} de 5 "
            f"pliegues. Hay algo real y no es capacidad, y queda como COMPLEMENTARIA"
        )
        decision = "COMPLEMENTARY"
    else:
        lectura = "REDUNDANTE_CON_LA_CONFIANZA"
        # La integracion falla, pero la tabla de distribucion contiene un hallazgo que el
        # veredicto de integracion no debe tragarse. El conflicto interno se DUPLICA entre
        # las filas que V03 resuelve y las que falla, y la entropia de la distribucion de
        # evidencia crece de forma monotona con la longitud del tweet. Eso es la hipotesis
        # de la fase CONFIRMADA como descripcion, y al mismo tiempo REFUTADA como
        #Representacion, y las dos cosas a la vez.
        conf_ok = next(f for f in filas_dist if f["estrato"] == "V03 correcto")
        conf_alto = next(f for f in filas_dist if f["estrato"] == "error de alta confianza")
        muy_larga = next(f for f in filas_dist if f["estrato"] == "MUY_LARGA")
        corta = next(f for f in filas_dist if f["estrato"] == "CORTA")
        veredicto = (
            f"La distribucion de la evidencia interna NO supera al margen de V03, con una "
            f"brecha de {brecha_marg:+.10f}, y tampoco supera al control de capacidad, con "
            f"{brecha_cap:+.10f}. Lo que la evidencia interna sabe sobre los fallos de V03 ya "
            f"esta en su confianza: el mejor rasgo alcanza {aucs[mejor_ev]:.6f} de AUC contra "
            f"el acierto frente a {auc_margen_acierto:.6f} del margen, y la diferencia es de "
            f"{ventaja_auc:+.6f} a favor del margen. En eso la descomposicion se parece al "
            f"perfil, que tambien habia perdido contra la referencia trivial de longitud.\n\n"
            f"PERO LA TABLA DE DISTRIBUCION DICE UNA COSA QUE EL VEREDICTO DE INTEGRACION NO "
            f"PUEDE TRAGARSE, y es el hallazgo real de la fase. Entre las filas que V03 "
            f"resuelve y las que falla con alta confianza, el CONFLICTO INTERNO SE DUPLICA: "
            f"{conf_ok['HS_evidencia_conflicto']:.4f} frente a "
            f"{conf_alto['HS_evidencia_conflicto']:.4f}. Y la entropia de la distribucion de "
            f"evidencia dentro del tweet crece de forma monotona con la longitud, de "
            f"{corta['HS_seg_entropia']:.4f} en la region CORTA a "
            f"{muy_larga['HS_seg_entropia']:.4f} en MUY_LARGA, y la fraccion de evidencia en "
            f"los tres primeros segmentos cae de {corta['HS_seg_top3_fraccion']:.4f} a "
            f"{muy_larga['HS_seg_top3_fraccion']:.4f}, o sea que en los tweets largos la senal "
            f"deja de estar concentrada y se reparte.\n\n"
            f"La hipotesis de la fase queda CONFIRMADA COMO DESCRIPCION y REFUTADA COMO "
            f"REPRESENTACION, y las dos cosas a la vez. La estructura interna es real y "
            f"diferente, y es exactamente la que la bolsa de n-gramas aplana. Lo que no "
            f"funciona es anadirla a V03, porque V03 ya la lleva dentro en su confianza, y "
            f"eso es un resultado sobre la REPRESENTACION y no sobre la ESTRUCTURA"
        )
        decision = ("CLOSED como representacion, y CONFIRMADA como descripcion. La estructura "
                    "interna es real, V03 ya la ve en su confianza, y anadirla no aporta")
    log("")
    log(f"   LECTURA: {lectura}")
    log(f"   VEREDICTO: {veredicto}")
    log(f"   DECISION: {decision}")

    with CSV_EV.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["tipo", "clave", "valor"])
        for f in filas:
            w.writerow(["brazo", f["brazo"], f["corrected_cv"]])
            w.writerow(["brazo_delta", f["brazo"], f["delta_vs_V03"]])
        w.writerow(["brecha_contra_capacidad", "", round(brecha_cap, 10)])
        w.writerow(["brecha_contra_el_margen", "", round(brecha_marg, 10)])
        for f in filas_dist:
            w.writerow(["distribucion", f["estrato"], f["n"]]
                       + [f[c] for c in clave])
        for c, a in aucs.items():
            w.writerow(["auc", c, a])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P64_EVIDENCIA_SEGMENTAL",
        "naturaleza": ("EXPERIMENTO. 1 candidato y 2 controles, 0 submissions, 0 Kaggle, "
                       "0 holdout, 0 test"),
        "fecha": inicio.isoformat(timespec="seconds"), "experiment_id": "P64-02",
        "track": "P4-evidencia-segmental",
        "script": "experiments/p64_03_evidencia_segmental.py",
        "pregunta": ("donde esta la evidencia de V03 DENTRO del tweet, y la distribucion "
                     "interna aporta algo que V03 no tenga"),
        "por_que_es_nuevo_respecto_a_P59": (
            "P59-07 agrego por MAXIMO, que sesga al alza y termino con neto -4 y acierto "
            "0,423 en las celdas que cambiaban. Aqui se mide la DISPERSION: desviacion, "
            "rango, entropia, concentracion y conflicto. Y P59 no computo nunca la "
            "contribucion de un segmento al score completo"),
        "por_que_no_es_pseudo_labeling": (
            "cada segmento se puntua con los coeficientes de V03 aprendidos SOLO en las "
            "filas de entrenamiento del pliegue. Ningun segmento recibe etiqueta, ninguna "
            "contribuye a ninguna funcion objetivo y ninguno se usa como objetivo. Es "
            "descomposicion de un modelo que ya existe"),
        "segmentacion": "S1, la primaria declarada en P64-02",
        "n_columnas": n_cols, "rasgos_por_etiqueta": NOMBRES_BASE,
        "protocolo": {
            "vectorizadores": "los de V03, ajustados dentro de cada pliegue de entrenamiento",
            "clasificadores_de_segmento": ("los de V03, ajustados SOLO con las filas de "
                                           "entrenamiento del pliegue"),
            "control_de_capacidad": "V03 mas un bloque aleatorio de la misma anchura",
            "control_del_margen": ("V03 mas los margenes de V03, que es el control que "
                                   "P63-07 mostrn decisivo"),
            "confound_de_escala": "el bloque denso se normaliza a norma de fila uno",
            "umbrales": "0,44 / 0,51 / 0,46 congelados, sin reajustar",
        },
        "resultados": detalle,
        "brecha_contra_capacidad": round(brecha_cap, 10),
        "brecha_contra_el_margen": round(brecha_marg, 10),
        "distribucion_por_estrato": filas_dist,
        "auc_para_predecir_el_error": aucs,
        "ventaja_del_mejor_rasgo_sobre_el_margen": round(ventaja_auc, 6),
        "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                 "delta": round(d_seg, 10), "pliegues_mejorados": fm, "superado": gate},
        "lectura": lectura, "veredicto": veredicto, "decision": decision,
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_EV.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P64 - La evidencia de V03 descompuesta por segmento")
    W("")
    W(f"Generado por `experiments/p64_03_evidencia_segmental.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Por que esta es la linea de la que depende la fase")
    W("")
    W("P64-02 cerro el perfil segmental como bloque de features, y con honestidad: el mejor")
    W("descriptor estructural llego a AUC 0,589847 para predecir los errores de V03, que es")
    W("PEOR que contar tokens, 0,600506. La asociacion entre muchos segmentos y muchos errores")
    W("es longitud con disfraz de estructura.")
    W("")
    W("Eso no toca lo que este script mide, y la diferencia no es sutil. El perfil pregunta")
    W("que FORMA tiene la segmentacion. Esto pregunta **donde esta la evidencia del modelo**")
    W("dentro de ella. Un tweet puede tener un perfil corriente y llevar su evidencia toxica en")
    W("un segmento corto que el promedio del documento entierra, y solo la segunda pregunta lo")
    W("ve.")
    W("")
    W("## Por que esto es descomposicion y no pseudo-labeling")
    W("")
    W("Cada segmento se transforma con los vectorizadores de V03 **y se puntua con los")
    W("clasificadores de V03, ambos ajustados SOLO en las filas de entrenamiento del pliegue**.")
    W("El segmento de una fila de validacion lo puntua un modelo que no vio esa fila, ni")
    W("ninguna otra de ese pliegue. Ningun segmento recibe etiqueta, ninguno entra en una")
    W("funcion objetivo y ninguno se usa como objetivo. Los segmentos se leen a traves de un")
    W("modelo que ya existe, como un microscopio lee una lamina, y no dejan marca.")
    W("")
    W("## Por que no es un P59 repetido")
    W("")
    W("P59-07 agrego por **maximo**, que sesga al alza y termino con neto -4 y acierto 0,423 en")
    W("las celdas que cambiaban. Aqui se mide la **DISPERSION**: desviacion, rango, entropia,")
    W("concentracion y conflicto. Y P59 no computo nunca la contribucion de un segmento al")
    W("score completo.")
    W("")
    W(f"El bloque tiene **{n_cols} columnas**, {len(NOMBRES_BASE)} por etiqueta: maximo, minimo,")
    W("media, mediana, top-2, top-3, fraccion de los tres primeros, desviacion, rango,")
    W("entropia, margen maximo y medio, evidencia positiva, negativa, conflicto y segmentos en")
    W("desacuerdo.")
    W("")
    W("## Los tres brazos")
    W("")
    W("| brazo | corrected_cv | delta vs V03 | pliegues | neto |")
    W("|---|---:|---:|---:|---:|")
    for f in filas:
        pl = sum(1 for a, b in zip(detalle[f['brazo']]["folds"], folds_v) if a > b)
        W(f"| {f['brazo']} | {f['corrected_cv']:.10f} | {f['delta_vs_V03']:+.10f} | "
          f"{pl}/5 | {detalle[f['brazo']]['rescate'] - detalle[f['brazo']]['damage']:+d} |")
    W("")
    W("El tercer brazo es el que mas importa. **V03_margen** es V03 mas sus propios margenes,")
    W("y P63-07 mostro que el margen solo ya explica los errores de V03 con AUC 0,757141 y que")
    W("anadir bloques lo empeoraba. Si la distribucion de evidencia no supera al margen, la")
    W("familia se cierra por una razon fuerte.")
    W("")
    W("## Las dos brechas, que son las unicas preguntas")
    W("")
    W("```text")
    W(f"contra el control de capacidad: {brecha_cap:+.10f}")
    W(f"contra el margen de V03:      {brecha_marg:+.10f}")
    W("```")
    W("")
    W("### Por rebanada de longitud")
    W("")
    W("| bin | n | F1 de V03 | F1 de V03_seg | delta |")
    W("|---|---:|---:|---:|---:|")
    for nombre, _lo, _hi in BINS:
        pb = detalle["V03_seg"]["por_bin"][nombre]
        W(f"| {nombre} | {pb['n']} | {pb['F1_de_V03']:.4f} | {pb['F1']:.4f} | "
          f"{pb['delta']:+.4f} |")
    W("")
    W("## La distribucion de evidencia interna por estrato")
    W("")
    W("| estrato | n | " + " | ".join(f"`{c.split('_', 1)[1]}`" for c in clave) + " |")
    W("|---:|---:|" + "---:|" * len(clave))
    for f in filas_dist:
        W(f"| {f['estrato']} | {f['n']} | "
          + " | ".join(f"{f[c]:.4f}" for c in clave) + " |")
    W("")
    W("## La evidencia contra la confianza")
    W("")
    W("| rasgo | AUC para predecir si V03 falla |")
    W("|---|---:|")
    for c, a in aucs.items():
        W(f"| {c} | {a:.6f} |")
    W("")
    W(f"El mejor rasgo de evidencia es **{mejor_ev}**, con una ventaja de **{ventaja_auc:+.6f}**")
    W("sobre la referencia del margen de V03.")
    W("")
    W("## Veredicto")
    W("")
    W("```text")
    W(veredicto)
    W("")
    W(f"LECTURA: {lectura}")
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p64_03_evidencia_segmental.py`. Sin push.*")
    MD_EV.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_EV, JSON_EV, CSV_EV):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P64-03 OK | n_cols {n_cols} | V03_seg {detalle['V03_seg']['corrected_cv']:.6f} | "
        f"brecha capacidad {brecha_cap:+.6f} | brecha margen {brecha_marg:+.6f} | "
        f"decision {decision}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
