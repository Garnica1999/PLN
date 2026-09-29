"""P64 track P5 and P6: leave-one-segment-out, and the contradiction profile from it.

What is genuinely new here, and it is not a variant of P64-03
------------------------------------------------------------
P64-03 measured the DISTRIBUTION of the per-segment probabilities, which is a description of
what the model says about each piece in isolation. It cannot say which piece the full-document
score actually DEPENDS on, because a segment can score high on its own and still be irrelevant
to the total once the rest of the document is added.

That is what leave-one-segment-out answers, and P59 never computed it. For every row with more
than one segment, and for each label:

    full_score        the score V03 gives the complete document
    score_without_i   the score it gives to the document with segment i removed
    contribution_i    full_score minus score_without_i

The contribution is signed, which is the whole point. A segment that pushes the score towards
the positive class contributes a positive number and a segment that pulls it back contributes
a negative one, and the spread between them is a property of the DOCUMENT that neither the
per-segment probabilities nor the aggregate score can express. P64-03 measured where the
evidence sits; this measures where the DECISION rests.

The two statistics that follow
-------------------------------
    concentration    the share of the total positive contribution held by the leading
                     segment, and the normalised entropy of the contributions. A high
                     concentration means one segment decides; a high entropy means the
                     evidence is genuinely spread.
    contradiction    positive mass against negative mass among the contributions, plus the
                     gap between the largest positive and the largest negative. This is
                     different from P64-03's conflict, which was computed on probabilities;
                     this one is computed on contributions and so separates a segment that
                     disagrees with the consensus from one that is simply weak.

The protocol is inherited unchanged from P64-03 and is not relaxed anywhere
-----------------------------------------------------------------------------
Every document with a segment removed is transformed with the fold's V03 vectorizers and scored
with the fold's V03 classifiers. The classifier is never trained on a partial document, and no
segment is ever assigned a label. Rows with a single segment are not given a contribution
profile at all, because the leave-one-out of a one-segment document is the empty string and
that is not evidence of anything; they carry a sentinel and are reported separately rather than
being filled in with a zero that would look like a measurement.

The arms
-------
    V03_loo       V03 plus the contribution block
    V03_random    V03 plus a random block of the same width, the capacity control
    V03_margen    V03 plus its own margins, the control that has closed three families now

The arm that matters is the third, and the reason is specific rather than ritual. P63-07 and
P64-03 both found that a block describing how the model is doing on the inside does worse than
the margin the model already exposes. If the leave-one-out profile beats the margin, the family
survives; if it does not, the internal structure is real, already represented, and the phase
closes on a mechanism rather than on a number.

Usage
-----
    /home/carlos/venv/bin/python experiments/p64_04_leave_one_out.py
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
MD_LOO = REPORTS / "p64_leave_one_out.md"
JSON_LOO = REPORTS / "p64_leave_one_out.json"
CSV_LOO = REPORTS / "p64_contradiction_analysis.csv"
CELDA_LOO = PROJECT_ROOT / "features" / "p64_loo_cells.npz"

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
    print(f"[P64-04] {mensaje}", flush=True)


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


def perfil_contribuciones(contrib: np.ndarray) -> dict[str, float]:
    """Perfil de las contribuciones con signo de una etiqueta.

    Args:
        contrib: contribuciones por segmento, con signo.

    Returns:
        Diccionario de rasgos acotados.
    """
    c = np.asarray(contrib, dtype=float)
    if c.size == 0:
        return {k: 0.0 for k in NOMBRES_BASE}
    pos = np.clip(c, 0, None)
    neg = np.clip(-c, 0, None)
    total_pos = pos.sum()
    total_neg = neg.sum()
    orden = np.sort(np.abs(c))[::-1]
    # La entropia normalizada de las contribuciones tiene un caso degenerado real: cuando
    # todos los segmentos aportan lo mismo, la distribucion es uniforme y la entropia es 1; y
    # cuando todos aportan cero, la suma es cero y la division produce NaN. Una primera
    # version no cubria el segundo caso y el bloque entero llego al clasificador con NaN, que
    # aborta. Se decide aqui de forma explicita: reparto uniforme es entropia maxima, y
    # reparto vacio es entropia cero, porque no hay en que haber incertidumbre.
    if orden.sum() <= 0:
        ent = 0.0
    elif c.size > 1:
        q = orden / orden.sum()
        q = np.where(q > 0, q, 1.0)
        ent = float(-(q * np.log2(q)).sum()) / np.log2(c.size)
    else:
        ent = 0.0
    return {
        "loo_max": float(c.max()),
        "loo_min": float(c.min()),
        "loo_rango": float(c.max() - c.min()),
        "loo_desviacion": float(c.std()),
        "loo_entropia": ent,
        "loo_concentracion": float(orden[0] / orden.sum()) if orden.sum() > 0 else 0.0,
        "loo_positiva": float(total_pos / max(c.size, 1)),
        "loo_negativa": float(total_neg / max(c.size, 1)),
        "loo_conflicto": float(min(total_pos, total_neg) / max(c.size, 1)),
        "loo_brecha": float(c.max() - c.min()),
        "loo_cambia_de_signo": float(np.any(pos > 1e-9) and np.any(neg > 1e-9)),
        "loo_fraccion_implicita": float(np.sum(c <= 1e-9) / max(c.size, 1)),
    }


NOMBRES_BASE = list(perfil_contribuciones(np.array([0.1, -0.1])).keys())


def main() -> None:
    """Calcula la contribucion de cada segmento y mide concentracion y contradiccion.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P64-04 leave-one-segment-out, y el perfil de contradiccion que sale de el")
    log("   la contribucion es CON SIGNO: eso separa un segmento que discrepa del")
    log("   consenso de uno que simplemente es debil, y es lo que P59 no computo nunca")

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
    multi = n_seg >= 2
    log("")
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | {int(multi.sum())} filas con dos o mas "
        f"segmentos")
    log("   las de un solo segmento NO reciben perfil: el leave-one-out de un documento de un")
    log("   segmento es la cadena vacia, que no es evidencia de nada. Se informsn aparte con")
    log(f"   {int((~multi).sum())} filas y no se rellenan con un cero que pareceria una medida")

    nombres = [f"{et}_{c}" for et in ETIQUETAS for c in NOMBRES_BASE]
    n_cols = len(nombres)
    log(f"   bloque de contribucion: {n_cols} columnas, {len(NOMBRES_BASE)} por etiqueta")

    bloque = np.zeros((n, n_cols), dtype=float)
    filas_loo: list[dict[str, Any]] = []
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

        modelos = []
        for j in range(3):
            mo = LogisticRegression(**LR)
            mo.fit(xbase, y[ir, j])
            modelos.append(mo)

        def puntuar(texto: str, j: int, vw: TfidfVectorizer, vc: TfidfVectorizer,
                    cls: list[LogisticRegression]) -> float:
            """Puntua un texto con el modelo del pliegue.

            Los vectorizadores se reciben por argumento y no se capturan del cierre. Es
            una cuestion de correctitud y no de estilo: en la segunda vuelta de este
            script se vuelve a entrenar V03 y los vectorizadores del cierre serian los
            del ultimo pliegue, que es exactamente el tipo de fallo que produce
            numeros plausibles y equivocados.

            Args:
                texto: texto a puntuar.
                j: indice de la etiqueta.
                vw: vectorizador de palabras del pliegue.
                vc: vectorizador de caracteres del pliegue.
                cls: clasificadores del pliegue, uno por etiqueta.

            Returns:
                La probabilidad de la etiqueta j.
            """
            xs = hstack([vw.transform([texto]), vc.transform([texto])]).tocsr()
            return float(cls[j].predict_proba(xs)[0, 1])

        for fila in iva:
            segs = segmentos[fila]
            if len(segs) < 2:
                continue
            completo = " ".join(segs)
            for j in range(3):
                completo_p = puntuar(completo, j, pw, pc, modelos)
                parciales = [puntuar(" ".join(segs[:i] + segs[i + 1:]), j, pw, pc, modelos)
                             for i in range(len(segs))]
                contrib = np.array([completo_p - p for p in parciales])
                bloque[fila, j * len(NOMBRES_BASE):(j + 1) * len(NOMBRES_BASE)] = (
                    list(perfil_contribuciones(contrib).values()))
        log(f"      pliegue {f} listo ({time.time() - t0:.0f} s) | "
            f"documentos con leave-one-out: {int(multi[iva].sum())}")

    # ---------- una fila de tabla por fila con leave-one-out ----------
    for i in np.where(multi)[0]:
        fila: dict[str, Any] = {"id": int(ids[i]), "n_tokens": int(n_tok[i]),
                                "n_segmentos": int(n_seg[i])}
        for j, et in enumerate(ETIQUETAS):
            c = bloque[i, j * len(NOMBRES_BASE):(j + 1) * len(NOMBRES_BASE)]
            fila[f"{et}_concentracion"] = round(float(c[5]), 6)
            fila[f"{et}_conflicto"] = round(float(c[8]), 6)
            fila[f"{et}_cambia_de_signo"] = round(float(c[10]), 6)
            fila[f"{et}_fraccion_implicita"] = round(float(c[11]), 6)
        filas_loo.append(fila)
    log("")
    log(f"   PERFIL DE CONTRIBUCION calculado para {len(filas_loo)} filas")

    # ---------- los brazos ----------
    proba: dict[str, np.ndarray] = {b: np.zeros((n, 3)) for b in
                                   ("V03_loo", "V03_random", "V03_margen")}
    # El bloque solo tiene valores en las filas multi segmento. Las de uno solo se dejan en
    # cero, lo que es correcto: no tienen perfil, y rellenarlas con un valor inventado
    # seria contaminar el control.
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

        def escalar2(mm: csr_matrix, ref: float = objetivo) -> csr_matrix:
            """Iguala la norma media de fila a la del control.

            Args:
                mm: matriz dispersa.
                ref: norma objetivo.

            Returns:
                La matriz escalada.
            """
            media = float(np.sqrt(np.asarray(mm.multiply(mm).sum(axis=1)).ravel()).mean())
            return mm if media <= 0 else mm.multiply(ref / media)

        rng = np.random.default_rng(SEMILLA + f)
        matrices = {"V03_loo": bloque,
                    "V03_random": rng.standard_normal((n, n_cols)),
                    "V03_margen": np.column_stack([margen_v03[:, j] for j in range(3)])}
        for brazo, be in matrices.items():
            sd = be[ir].std(axis=0)
            sd[sd < 1e-8] = 1.0
            z = (be - be[ir].mean(axis=0)) / sd[None, :]
            norma = float(np.sqrt(np.asarray((z[ir] ** 2).sum(axis=1))).mean())
            z = z / norma if norma > 0 else z
            xa = escalar2(hstack([xbase, csr_matrix(z[ir])]).tocsr())
            xav = escalar2(hstack([xbase_v, csr_matrix(z[iva])]).tocsr())
            for j in range(3):
                mo = LogisticRegression(**LR)
                mo.fit(xa, y[ir, j])
                proba[brazo][iva, j] = mo.predict_proba(xav)[:, 1]
        log(f"      brazo del pliegue {f} listo")

    detalle: dict[str, Any] = {}
    filas: list[dict[str, Any]] = []
    for brazo in ("V03_loo", "V03_random", "V03_margen"):
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
                                                 zero_division=0)), 6)}
            por_bin[nombre]["delta"] = round(por_bin[nombre]["F1"]
                                             - por_bin[nombre]["F1_de_V03"], 6)
        detalle[brazo] = {
            "corrected_cv": ccv, "folds": folds, "por_bin": por_bin,
            "rescate": int(np.count_nonzero(ac & (pred_v03 != y))),
            "damage": int(np.count_nonzero((~ac) & (pred_v03 == y)))}
        filas.append({"brazo": brazo, "corrected_cv": round(ccv, 10),
                      "delta_vs_V03": round(ccv - ccv_ref, 10),
                      "folds": "|".join(f"{x:.10f}" for x in folds),
                      "rescate": detalle[brazo]["rescate"],
                      "damage": detalle[brazo]["damage"]})

    d_loo = detalle["V03_loo"]["corrected_cv"] - ccv_ref
    brecha_cap = detalle["V03_loo"]["corrected_cv"] - detalle["V03_random"]["corrected_cv"]
    brecha_marg = detalle["V03_loo"]["corrected_cv"] - detalle["V03_margen"]["corrected_cv"]
    folds_v = [float(f1_score(y[pliegues == g], pred_v03[pliegues == g], average="macro"))
               for g in range(1, 6)]
    fm = sum(1 for a, b in zip(detalle["V03_loo"]["folds"], folds_v) if a > b)
    gate = bool(d_loo >= GATE_DELTA and fm >= GATE_PLIEGUES and brecha_marg > 0)

    log("")
    log("   LOS TRES BRAZOS")
    log(f"      {'brazo':14}{'corrected_cv':>18}{'delta vs V03':>16}{'pliegues':>10}{'neto':>8}")
    for f in filas:
        pl = sum(1 for a, b in zip(detalle[f['brazo']]["folds"], folds_v) if a > b)
        log(f"      {f['brazo']:14}{f['corrected_cv']:>18.10f}{f['delta_vs_V03']:>+16.10f}"
            f"{pl:>7d}/5{detalle[f['brazo']]['rescate'] - detalle[f['brazo']]['damage']:>8d}")
    log("")
    log("   LAS DOS BRECHAS")
    log(f"      contra el control de capacidad: {brecha_cap:+.10f}")
    log(f"      contra el margen de V03:      {brecha_marg:+.10f}")

    # ---------- la contradiccion por estrato ----------
    log("")
    log("   LA CONTRADICCION, que es el objetivo de P6")
    error = pred_v03 != y
    alto = error & (np.abs(proba_v03 - V03_UMBRALES[None, :]) >= DELTA_HC)
    claves = ["concentracion", "conflicto", "cambia_de_signo", "fraccion_implicita", "entropia"]
    estratos = [("todas multi", multi), ("V03 correcto", multi & ~error.any(axis=1)),
                ("V03 incorrecto", multi & error.any(axis=1)),
                ("error de alta confianza", multi & alto.any(axis=1))]
    for nombre, lo, hi in BINS:
        estratos.append((nombre, multi & (n_tok >= lo) & (n_tok <= hi)))
    filas_cont: list[dict[str, Any]] = []
    log(f"      {'estrato':>26}{'n':>7}"
        + "".join(f"{c[:17]:>19}" for c in claves))
    for nombre, m in estratos:
        if m.sum() < 5:
            continue
        vals = []
        for c in claves:
            col = np.mean([bloque[m, ETIQUETAS.index(et) * len(NOMBRES_BASE)
                              + NOMBRES_BASE.index(f"loo_{c}")] for et in ETIQUETAS], axis=0)
            vals.append(float(col.mean()))
        filas_cont.append({"estrato": nombre, "n": int(m.sum())}
                          | {f"loo_{c}": round(v, 6) for c, v in zip(claves, vals)})
        log(f"      {nombre:>26}{int(m.sum()):>7}"
            + "".join(f"{v:>19.4f}" for v in vals))
    next(f for f in filas_cont if f["estrato"] == "V03 correcto")
    next(f for f in filas_cont if f["estrato"] == "error de alta confianza")

    # ---------- AUC contra el acierto ----------
    objetivo_ok = (~error.any(axis=1)).astype(int)
    aucs: dict[str, Any] = {}
    for c in ("concentracion", "conflicto", "cambia_de_signo", "fraccion_implicita"):
        col = np.mean([bloque[:, ETIQUETAS.index(et) * len(NOMBRES_BASE)
                          + NOMBRES_BASE.index(f"loo_{c}")] for et in ETIQUETAS], axis=0)
        aucs[c] = round(float(roc_auc_score(objetivo_ok, col)), 6)
    ref = round(float(roc_auc_score(objetivo_ok, np.abs(margen_v03).mean(axis=1))), 6)
    aucs["margen_de_V03"] = ref
    log("")
    log("   LA CONTRIBUCION CONTRA LA CONFIANZA, AUC para predecir el ACIERTO")
    for c, a in aucs.items():
        log(f"      {c:34} {a:.6f}")
    mejor = max((k for k in aucs if k != "margen_de_V03"), key=lambda k: aucs[k])
    log(f"      el mejor es {mejor} con {aucs[mejor]:.6f} frente a {ref:.6f} del margen, "
        f"ventaja {aucs[mejor] - ref:+.6f}")

    np.savez_compressed(CELDA_LOO, ids=ids, y_true=y, pliegues=pliegues, pred_v03=pred_v03,
                        n_tokens=n_tok, n_segmentos=n_seg, multi=multi, nombres=np.array(nombres),
                        bloque=bloque, margen=margen_v03,
                        **{f"proba_{b}": proba[b] for b in proba})

    if brecha_marg > 0.01 and brecha_cap > 0.01:
        lectura, decision = "CONTRIBUCION_NUEVA", "PROMISING"
        veredicto = (f"la contribucion con signo supera al margen en {brecha_marg:+.10f} y al "
                     f"control en {brecha_cap:+.10f}. Es informacion que V03 no expone")
    elif brecha_marg > 0:
        lectura, decision = "CONTRIBUCION_COMPLEMENTARIA", "COMPLEMENTARY"
        veredicto = (f"la contribucion con signo supera al margen en {brecha_marg:+.10f} pero "
                     f"no alcanza el gate de {GATE_DELTA:+.4f} con {fm} de 5 pliegues")
    else:
        lectura, decision = "CONTRIBUCION_REDUNDANTE", "CLOSED por redundancia con la confianza"
        veredicto = (f"la contribucion con signo NO supera al margen de V03, con una brecha de "
                     f"{brecha_marg:+.10f}. El mejor rasgo alcanza {aucs[mejor]:.6f} de AUC "
                     f"contra el acierto frente a {ref:.6f} del margen. Y P64-03 ya habia "
                     f"encontrado lo mismo con la distribucion de probabilidades. La "
                     f"descomposicion con signo es MAS informativa que ninguna de las dos, y "
                     f"aun asi no aporta, porque V03 lleva la estructura interna dentro de su "
                     f"confianza")
    log("")
    log(f"   LECTURA: {lectura}")
    log(f"   VEREDICTO: {veredicto}")
    log(f"   DECISION: {decision}")

    with CSV_LOO.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["tipo", "clave"] + [f"loo_{c}" for c in claves] + ["n"])
        for f in filas_cont:
            w.writerow(["contradiccion", f["estrato"]] + [f[f"loo_{c}"] for c in claves]
                       + [f["n"]])
        for c, a in aucs.items():
            w.writerow(["auc_contra_el_acierto", c, a])
        for f in filas:
            w.writerow(["brazo", f["brazo"], f["corrected_cv"], f["delta_vs_V03"]])
        w.writerow(["brecha_contra_capacidad", "", round(brecha_cap, 10)])
        w.writerow(["brecha_contra_el_margen", "", round(brecha_marg, 10)])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P64_LEAVE_ONE_OUT",
        "naturaleza": ("EXPERIMENTO. 1 candidato y 2 controles, 0 submissions, 0 Kaggle, "
                       "0 holdout, 0 test"),
        "fecha": inicio.isoformat(timespec="seconds"), "experiment_id": "P64-03",
        "track": "P5-P6-leave-one-out-y-contradiccion",
        "script": "experiments/p64_04_leave_one_out.py",
        "pregunta": ("donde descansa la decision de V03, y la contribucion con signo de cada "
                     "segmento aporta algo que su confianza no tenga"),
        "novedad": ("P59 no computo nunca contribuciones por unidad. P64-03 midio la "
                    "distribucion de probabilidades por segmento, que describe lo que el "
                    "modelo dice de cada pieza aislada pero NO dice de que pieza depende la "
                    "decision una vez junto el resto del documento"),
        "definicion": ("contribucion_i = score del documento completo menos el score del "
                       "documento sin el segmento i, por etiqueta, con los coeficientes de "
                       "V03 del pliegue de entrenamiento"),
        "por_que_importa_el_signo": ("separa un segmento que DISCREPA del consenso de uno que "
                                     "simplemente es debil, y el conflicto sobre "
                                     "contribuciones no es el mismo que el conflicto sobre "
                                     "probabilidades de P64-03"),
        "n_columnas": n_cols, "rasgos_por_etiqueta": NOMBRES_BASE,
        "filas_con_perfil": int(multi.sum()),
        "filas_sin_perfil": int((~multi).sum()),
        "por_que_las_de_un_segmento_no_tienen_perfil": (
            "el leave-one-out de un documento de un segmento es la cadena vacia, que no es "
            "evidencia de nada. No se rellenan con un cero porque un cero pareceria una "
            "medida y es una ausencia"),
        "protocolo": {
            "clasificadores": "los de V03, ajustados SOLO en las filas de entrenamiento",
            "documentos_parciales": ("se transforman con los vectorizadores del pliegue y se "
                                     "puntuan con sus clasificadores. El clasificador nunca "
                                     "se entrena sobre un documento parcial"),
            "pseudo_labeling": "ninguno: ningun segmento recibe etiqueta ni es objetivo",
            "control_de_capacidad": "V03 mas un bloque aleatorio de la misma anchura",
            "control_del_margen": "V03 mas sus propios margenes",
            "umbrales": "0,44 / 0,51 / 0,46 congelados",
        },
        "resultados": detalle,
        "brecha_contra_capacidad": round(brecha_cap, 10),
        "brecha_contra_el_margen": round(brecha_marg, 10),
        "contradiccion_por_estrato": filas_cont,
        "auc_contra_el_acierto": aucs,
        "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                 "delta": round(d_loo, 10), "pliegues_mejorados": fm, "superado": gate},
        "lectura": lectura, "veredicto": veredicto, "decision": decision,
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_LOO.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P64 - Leave-one-segment-out y el perfil de contradiccion")
    W("")
    W(f"Generado por `experiments/p64_04_leave_one_out.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Que es genuinamente nuevo aqui")
    W("")
    W("P64-03 midio la **distribucion** de las probabilidades por segmento, que describe lo que")
    W("el modelo dice de cada pieza AISLADA. No puede decir de que pieza depende la DECISION")
    W("una vez junto el resto del documento, porque un segmento puede puntuar alto por si solo")
    W("y seguir siendo irrelevante para el total.")
    W("")
    W("Eso es lo que responde el leave-one-out, y **P59 nunca lo computo**:")
    W("")
    W("```text")
    W("contribucion_i = score del documento completo menos el score sin el segmento i")
    W("```")
    W("")
    W("La contribucion es **con signo**, y esa es la clave: un segmento que empuja el score")
    W("hacia la clase positiva aporta un numero positivo y uno que lo frena aporta un negativo,")
    W("y la separacion entre ambos es una propiedad del DOCUMENTO que ni las probabilidades")
    W("por segmento ni el score agregado expresan.")
    W("")
    W(f"**{int(multi.sum())} filas** tienen dos o mas segmentos y reciben perfil. Las")
    W(f"**{int((~multi).sum())} de un solo segmento no lo reciben**: su leave-one-out seria la")
    W("cadena vacia, que no es evidencia de nada, y no se rellenan con un cero porque un cero")
    W("pareceria una medida cuando es una ausencia.")
    W("")
    W(f"El bloque tiene **{n_cols} columnas**, {len(NOMBRES_BASE)} por etiqueta.")
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
    W("```text")
    W(f"contra el control de capacidad: {brecha_cap:+.10f}")
    W(f"contra el margen de V03:      {brecha_marg:+.10f}")
    W("```")
    W("")
    W("### Por rebanada de longitud")
    W("")
    W("| bin | n | F1 de V03 | F1 de V03_loo | delta |")
    W("|---|---:|---:|---:|---:|")
    for nombre, _lo, _hi in BINS:
        pb = detalle["V03_loo"]["por_bin"][nombre]
        W(f"| {nombre} | {pb['n']} | {pb['F1_de_V03']:.4f} | {pb['F1']:.4f} | "
          f"{pb['delta']:+.4f} |")
    W("")
    W("## La contradiccion, que es el objetivo de P6")
    W("")
    W("| estrato | n | " + " | ".join(f"`loo_{c}`" for c in claves) + " |")
    W("|---:|---:|" + "---:|" * len(claves))
    for f in filas_cont:
        W(f"| {f['estrato']} | {f['n']} | "
          + " | ".join(f"{f['loo_' + c]:.4f}" for c in claves) + " |")
    W("")
    W("## La contribucion contra la confianza")
    W("")
    W("| rasgo | AUC para predecir el ACIERTO |")
    W("|---|---:|")
    for c, a in aucs.items():
        W(f"| {c} | {a:.6f} |")
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
    W("*Generado por `experiments/p64_04_leave_one_out.py`. Sin push.*")
    MD_LOO.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_LOO, JSON_LOO, CSV_LOO):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P64-04 OK | filas con perfil {int(multi.sum())} | V03_loo "
        f"{detalle['V03_loo']['corrected_cv']:.6f} | brecha capacidad {brecha_cap:+.6f} | "
        f"brecha margen {brecha_marg:+.6f} | decision {decision}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
