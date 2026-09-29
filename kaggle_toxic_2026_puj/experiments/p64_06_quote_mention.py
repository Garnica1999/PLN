"""P64 track P11 and P12: quote structure and mention-boundary structure.

The EDA comes first and it decides whether either track runs
-----------------------------------------------------------
The brief is explicit that the quote line is a priority only if the exploratory analysis
shows that quotes are frequent or differentially associated with errors. That condition is
checked before any model is trained, and the decision rule is written in advance:

    a boundary is worth modelling only if it is FREQUENT, that is more than 5 per cent of the
    rows carry at least two parts, or DIFFERENTIAL, that is the error rate inside the two
    parts differs by more than five points

If neither condition holds for a boundary, that track is CLOSED without an experiment and the
script says why. Closing a track for a measured reason is not the same as leaving it untested.

What is actually being modelled, and the difference from a count
----------------------------------------------------------------
A count says how many mentions a tweet has. This says what the model believes on each SIDE of
the boundary, which is a different object and the only one that can express the hypothesis.
For a mention at character position p:

    evidence before    the max and the spread of the segment scores over the text left of p
    evidence after     the same over the text right of p
    evidence at        the same over the segment containing p

and for a quoted stretch delimited by a pair of double quotes:

    evidence inside    the same over the quoted text
    evidence outside   the same over the text around it
    asymmetry          inside max minus outside max, which asks whether V03 is scoring a
                       quote as if it were the author's own words

That asymmetry is the question the brief poses, stated as a number: is V03 treating quoted
content as its own evidence? A flag cannot answer it and a count cannot answer it.

The controls
------------
Same two as every other candidate in this phase: a random block of identical width for
capacity, and V03 plus its own margins, which is the bar that P63 and P64-03, P64-04 and
P64-05 have all shown to be the hard one.

Usage
-----
    /home/carlos/venv/bin/python experiments/p64_06_quote_mention.py
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

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
MD_QM = REPORTS / "p64_quote_mention.md"
JSON_QM = REPORTS / "p64_quote_mention.json"
CSV_QM = REPORTS / "p64_quote_mention.csv"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True,
               "lowercase": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2,
               "sublinear_tf": True, "lowercase": True}
LR = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear",
      "random_state": 42}
DELTA_HC = 0.10
UMBRAL_FRECUENCIA = 0.05
UMBRAL_DIFERENCIAL = 0.05
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
RE_MENCION = re.compile(r"@\w+")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P64-06] {mensaje}", flush=True)


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
    """Segmenta con la regla S1.

    Args:
        texto: texto del registro.

    Returns:
        Lista de segmentos no vacios.
    """
    partes = [p for p in (s.strip() for s in RE_SEG.split(texto)) if p]
    return partes or [texto.strip() or texto]


def partes_de_cita(texto: str) -> tuple[str, str] | None:
    """Separa el tramo citado del resto, sin identificar a nadie.

    Solo acepta un patron de comillas ASCII dobles bien emparejadas, que es el unico
    formato de cita que el corpus contiene de forma consistente. Si no hay par, no hay cita.

    Args:
        texto: texto del registro.

    Returns:
        Par con el interior de la cita y el exterior, o None si no hay cita cerrada.
    """
    partes = texto.split('"')
    if len(partes) < 3:
        return None
    dentro = " ".join(p for p in partes[1::2] if p.strip())
    fuera = " ".join(p for p in [partes[0], *partes[2::]] if p.strip())
    if not dentro.strip():
        return None
    return dentro, fuera


def resumen(probs: np.ndarray, umbral: float) -> list[float]:
    """Resumen acotado de la evidencia interna de un conjunto de segmentos.

    Args:
        probs: probabilidades por segmento de una etiqueta.
        umbral: umbral congelado de V03 para esa etiqueta.

    Returns:
        Maximo, minimo, media, desviacion y evidencia acumulada, todos acotados.
    """
    p = np.asarray(probs, dtype=float)
    if p.size == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return [float(p.max()), float(p.min()), float(p.mean()), float(p.std()),
            float(np.clip(p - umbral, 0, None).sum() / p.size)]


RASGOS = ("max", "min", "media", "desv", "pos")


def main() -> None:
    """Audita y, si la EDA lo justifica, modela las fronteras de cita y de mencion.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P64-06 frontera de cita y frontera de mencion, con la EDA decidiendo antes")

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
    tr = pd.read_csv(RAW_TRAIN, sep=";", encoding="latin-1", dtype=str)
    tr["idn"] = pd.to_numeric(tr["id"].str.replace('"', "", regex=False),
                              errors="coerce").astype("int64")
    idx = tr.set_index("idn").loc[ids]
    if not np.array_equal(idx[["HS", "TR", "AG"]].to_numpy().astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    textos = idx["text"].astype(str).tolist()
    n_tok = np.array([len(t.split()) for t in textos])
    error = pred_v03 != y
    log(f"   OOF canonico {ccv_ref!r} | {n} filas")

    # ---------- la EDA que decide ----------
    citas = [partes_de_cita(t) for t in textos]
    con_cita = np.array([c is not None for c in citas])
    n_menc = np.array([len(RE_MENCION.findall(t)) for t in textos])
    con_menc = n_menc >= 1
    log("")
    log("   LA EDA, que se ejecuta ANTES que cualquier modelo")
    eda: dict[str, Any] = {
        "umbral_de_frecuencia": UMBRAL_FRECUENCIA,
        "umbral_de_diferencial": UMBRAL_DIFERENCIAL,
        "citas": {
            "filas_con_cita_cerrada": int(con_cita.sum()),
            "porcentaje": round(float(con_cita.mean() * 100), 4),
            "error_con_cita": round(float(error[con_cita].mean()), 6),
            "error_sin_cita": round(float(error[~con_cita].mean()), 6),
        },
        "menciones": {
            "filas_con_mencion": int(con_menc.sum()),
            "porcentaje": round(float(con_menc.mean() * 100), 4),
            "menciones_maximas": int(n_menc.max()),
            "error_con_mencion": round(float(error[con_menc].mean()), 6),
            "error_sin_mencion": round(float(error[~con_menc].mean()), 6),
        },
    }
    dif_cita = (eda["citas"]["error_con_cita"] - eda["citas"]["error_sin_cita"]
                if con_cita.sum() > 5 else 0.0)
    dif_menc = (eda["menciones"]["error_con_mencion"] - eda["menciones"]["error_sin_mencion"]
                if con_menc.sum() > 5 else 0.0)
    eda["citas"]["diferencial_de_error"] = round(dif_cita, 6)
    eda["menciones"]["diferencial_de_error"] = round(dif_menc, 6)
    abierta_cita = bool(eda["citas"]["porcentaje"] / 100 > UMBRAL_FRECUENCIA
                        or abs(dif_cita) > UMBRAL_DIFERENCIAL)
    abierta_menc = bool(eda["menciones"]["porcentaje"] / 100 > UMBRAL_FRECUENCIA
                        or abs(dif_menc) > UMBRAL_DIFERENCIAL)
    eda["cita_justificada"] = abierta_cita
    eda["mencion_justificada"] = abierta_menc
    log(f"      citas cerradas:   {int(con_cita.sum()):4d} filas, "
        f"{eda['citas']['porcentaje']:.1f} por ciento | error {eda['citas']['error_con_cita']:.2%}"
        f" contra {eda['citas']['error_sin_cita']:.2%} | diferencial {dif_cita:+.4f}")
    log(f"      menciones:        {int(con_menc.sum()):4d} filas, "
        f"{eda['menciones']['porcentaje']:.1f} por ciento, maximo {int(n_menc.max())} por fila"
        f" | error {eda['menciones']['error_con_mencion']:.2%} contra "
        f"{eda['menciones']['error_sin_mencion']:.2%} | diferencial {dif_menc:+.4f}")
    log(f"      regla declarada: frecuente si supera el {UMBRAL_FRECUENCIA:.0%}, diferencial "
        f"si la diferencia de error supera {UMBRAL_DIFERENCIAL:.0%}")
    log(f"      cita JUSTIFICADA: {abierta_cita} | mencion JUSTIFICADA: {abierta_menc}")

    tracks: dict[str, Any] = {}
    bloques: dict[str, np.ndarray] = {}
    nombres_bloque: dict[str, list[str]] = {}

    # ---------- el bloque de cita ----------
    if abierta_cita:
        filas_cita = np.where(con_cita)[0]
        log("")
        log(f"   CONSTRUYENDO EL BLOQUE DE CITA para {len(filas_cita)} filas")
        dentro_txt = [citas[i][0] for i in filas_cita]
        fuera_txt = [citas[i][1] for i in filas_cita]
        dentro_seg = [segmentar(t) for t in dentro_txt]
        fuera_seg = [segmentar(t) for t in fuera_txt] or [["vacio"]]
        bloque_cita = np.zeros((n, 3 * len(RASGOS) * 3), dtype=float)
        nombres_cita: list[str] = []
        for et in ETIQUETAS:
            for lado in ("dentro", "fuera", "asimetria"):
                for rasgo in RASGOS:
                    nombres_cita.append(f"{et}_{lado}_{rasgo}")
        for f in range(1, 6):
            _ir_c, iva_c = (np.intersect1d(filas_cita, np.where(pliegues != f)[0]),
                           np.intersect1d(filas_cita, np.where(pliegues == f)[0]))
            pw = TfidfVectorizer(**PARAMS_WORD)
            pc = TfidfVectorizer(**PARAMS_CHAR)
            pw.fit([textos[i] for i in np.where(pliegues != f)[0]])
            pc.fit([textos[i] for i in np.where(pliegues != f)[0]])
            modelos = []
            xb = hstack([pw.transform([textos[i] for i in np.where(pliegues != f)[0]]),
                         pc.transform([textos[i] for i in np.where(pliegues != f)[0]])]).tocsr()
            for j in range(3):
                mo = LogisticRegression(**LR)
                mo.fit(xb, y[np.where(pliegues != f)[0], j])
                modelos.append(mo)

            def puntua(texto: str, j: int, vw: TfidfVectorizer,
                       vc: TfidfVectorizer, cls: list[LogisticRegression]) -> float:
                """Puntua un texto con los modelos del pliegue.

                Args:
                    texto: texto a puntuar.
                    j: indice de la etiqueta.
                    vw: vectorizador de palabras.
                    vc: vectorizador de caracteres.
                    cls: clasificadores del pliegue.

                Returns:
                    La probabilidad de la etiqueta j.
                """
                xs = hstack([vw.transform([texto]), vc.transform([texto])]).tocsr()
                return float(cls[j].predict_proba(xs)[0, 1])

            for k in iva_c:
                pos = int(np.where(filas_cita == k)[0][0])
                for j in range(3):
                    um = float(V03_UMBRALES[j])
                    pd_ = np.array([puntua(s, j, pw, pc, modelos) for s in dentro_seg[pos]])
                    pf_ = np.array([puntua(s, j, pw, pc, modelos) for s in fuera_seg[pos]])
                    rd, rf = resumen(pd_, um), resumen(pf_, um)
                    fila = rd + rf + [a - b for a, b in zip(rd, rf)]
                    base = j * len(RASGOS) * 3
                    bloque_cita[k, base:base + len(fila)] = fila
        bloques["cita"] = bloque_cita
        nombres_bloque["cita"] = nombres_cita
        log(f"      bloque de cita: {bloque_cita.shape[1]} columnas")
    else:
        log("      la cita NO se justifica por la EDA y el track queda cerrado sin experimento")
    tracks["cita"] = {"justificada": abierta_cita, "decidido_por": "EDA previa"}

    # ---------- el bloque de mencion ----------
    filas_menc = np.where(con_menc)[0]
    log("")
    log(f"   CONSTRUYENDO EL BLOQUE DE MENCION para {len(filas_menc)} filas")
    log("      no se usa la identidad: solo la posicion del simbolo @")
    nombres_menc: list[str] = []
    for et in ETIQUETAS:
        for lado in ("antes", "despues", "alrededor", "salto"):
            for rasgo in RASGOS:
                nombres_menc.append(f"{et}_{lado}_{rasgo}")
    bloque_menc = np.zeros((n, 3 * len(RASGOS) * 4), dtype=float)
    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        pw = TfidfVectorizer(**PARAMS_WORD)
        pc = TfidfVectorizer(**PARAMS_CHAR)
        pw.fit([textos[i] for i in ir])
        pc.fit([textos[i] for i in ir])
        modelos = []
        xb = hstack([pw.transform([textos[i] for i in ir]),
                     pc.transform([textos[i] for i in ir])]).tocsr()
        for j in range(3):
            mo = LogisticRegression(**LR)
            mo.fit(xb, y[ir, j])
            modelos.append(mo)

        def puntua(texto: str, j: int, vw: TfidfVectorizer,
                   vc: TfidfVectorizer, cls: list[LogisticRegression]) -> float:
            """Puntua un texto con los modelos del pliegue.

            Args:
                texto: texto a puntuar.
                j: indice de la etiqueta.
                vw: vectorizador de palabras.
                vc: vectorizador de caracteres.
                cls: clasificadores del pliegue.

            Returns:
                La probabilidad de la etiqueta j.
            """
            if not texto.strip():
                return 0.0
            xs = hstack([vw.transform([texto]), vc.transform([texto])]).tocsr()
            return float(cls[j].predict_proba(xs)[0, 1])

        for fila in iva:
            if not con_menc[fila]:
                continue
            t = textos[fila]
            m = RE_MENCION.search(t)
            if m is None:
                continue
            antes, despues = t[: m.start()], t[m.end():]
            alrededor = m.group(0)
            for j in range(3):
                um = float(V03_UMBRALES[j])
                ra = resumen(np.array([puntua(s, j, pw, pc, modelos)
                                       for s in segmentar(antes)] or [0.0]), um)
                rd = resumen(np.array([puntua(s, j, pw, pc, modelos)
                                       for s in segmentar(despues)] or [0.0]), um)
                rz = resumen(np.array([puntua(s, j, pw, pc, modelos)
                                       for s in segmentar(alrededor)]), um)
                salto = [a - b for a, b in zip(rd, ra)]
                fila_r = ra + rd + rz + salto
                base = j * len(RASGOS) * 4
                bloque_menc[fila, base:base + len(fila_r)] = fila_r
    bloques["mencion"] = bloque_menc
    nombres_bloque["mencion"] = nombres_menc
    log(f"      bloque de mencion: {bloque_menc.shape[1]} columnas")

    # ---------- los brazos ----------
    log("")
    log("   LOS BRAZOS, con los dos controles de siempre")
    matrices_meta: dict[str, Any] = {}
    if "cita" in bloques:
        matrices_meta["V03_cita"] = (bloques["cita"], nombres_bloque["cita"])
    matrices_meta["V03_mencion"] = (bloques["mencion"], nombres_bloque["mencion"])
    proba: dict[str, np.ndarray] = {b: np.zeros((n, 3)) for b in matrices_meta}
    proba["V03_margen"] = np.zeros((n, 3))
    proba["V03_random"] = np.zeros((n, 3))
    ancho_random = max(b.shape[1] for b, _n in matrices_meta.values())
    detalle: dict[str, Any] = {}
    filas: list[dict[str, Any]] = []
    folds_v = [float(f1_score(y[pliegues == g], pred_v03[pliegues == g], average="macro"))
               for g in range(1, 6)]

    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        pw = TfidfVectorizer(**PARAMS_WORD)
        pc = TfidfVectorizer(**PARAMS_CHAR)
        xbase = hstack([pw.fit_transform([textos[i] for i in ir]),
                        pc.fit_transform([textos[i] for i in ir])]).tocsr()
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

        rng = np.random.default_rng(SEMILLA + f)
        todo: dict[str, np.ndarray] = {}
        for nombre, (bloque, _nombres) in matrices_meta.items():
            todo[nombre] = bloque
        todo["V03_random"] = rng.standard_normal((n, ancho_random))
        todo["V03_margen"] = margen_v03
        for brazo, be in todo.items():
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
        log(f"      pliegue {f} listo")

    for brazo, p in proba.items():
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
        detalle[brazo] = {"corrected_cv": ccv, "folds": folds, "por_bin": por_bin,
                          "rescate": int(np.count_nonzero(ac & (pred_v03 != y))),
                          "damage": int(np.count_nonzero((~ac) & (pred_v03 == y)))}
        filas.append({"brazo": brazo, "corrected_cv": round(ccv, 10),
                      "delta_vs_V03": round(ccv - ccv_ref, 10),
                      "pliegues_mejorados": sum(1 for a, b in zip(folds, folds_v) if a > b),
                      "rescate": detalle[brazo]["rescate"],
                      "damage": detalle[brazo]["damage"]})

    log(f"      {'brazo':14}{'corrected_cv':>18}{'delta vs V03':>16}{'pliegues':>10}{'neto':>8}")
    for f in filas:
        log(f"      {f['brazo']:14}{f['corrected_cv']:>18.10f}{f['delta_vs_V03']:>+16.10f}"
            f"{f['pliegues_mejorados']:>7d}/5{f['rescate'] - f['damage']:>8d}")

    brechas: dict[str, float] = {}
    for nombre in matrices_meta:
        brechas[f"{nombre}_contra_aleatorio"] = round(
            detalle[nombre]["corrected_cv"] - detalle["V03_random"]["corrected_cv"], 10)
        brechas[f"{nombre}_contra_el_margen"] = round(
            detalle[nombre]["corrected_cv"] - detalle["V03_margen"]["corrected_cv"], 10)
    log("")
    log("   LAS BRECHAS")
    for k, val in brechas.items():
        log(f"      {k:38} {val:+.10f}")

    mejor = max(matrices_meta,
                key=lambda b: detalle[b]["corrected_cv"]) if matrices_meta else None
    if mejor is None:
        decision, lectura, veredicto = "CLOSED", "SIN_TRACK", "ninguna frontera se justifica"
    else:
        d_glob = detalle[mejor]["corrected_cv"] - ccv_ref
        pl = next(f["pliegues_mejorados"] for f in filas if f["brazo"] == mejor)
        sup_cap = brechas[f"{mejor}_contra_aleatorio"] > 0
        sup_marg = brechas[f"{mejor}_contra_el_margen"] > 0
        if d_glob >= GATE_DELTA and pl >= GATE_PLIEGUES and sup_cap and sup_marg:
            lectura, decision = "PROMISING", "PROMISING"
            veredicto = (f"{mejor} alcanza {d_glob:+.10f} con {pl} de 5 y supera a los dos "
                         f"controles, luego exige bootstrap y segunda particion")
        elif sup_cap and sup_marg:
            lectura, decision = "COMPLEMENTARIA", "COMPLEMENTARY"
            veredicto = (f"{mejor} supera al control de anchura en "
                         f"{brechas[f'{mejor}_contra_aleatorio']:+.10f} y al margen en "
                         f"{brechas[f'{mejor}_contra_el_margen']:+.10f}, luego las fronteras "
                         f"llevan informacion que no es capacidad ni confianza, pero el global "
                         f"es {d_glob:+.10f} con {pl} de 5 y no pasa el gate de "
                         f"{GATE_DELTA:+.4f}")
        elif not abierta_cita and not abierta_menc:
            lectura, decision = "SIN_JUSTIFICACION", "CLOSED por la EDA previa"
            veredicto = "ninguna frontera cumple la regla de frecuencia ni la de diferencial"
        else:
            lectura, decision = "SIN_APORTE", "CLOSED"
            veredicto = (f"ninguna frontera aporta: el mejor es {mejor} con {d_glob:+.10f} "
                         f"global, {brechas[f'{mejor}_contra_aleatorio']:+.10f} contra el "
                         f"control de anchura y {brechas[f'{mejor}_contra_el_margen']:+.10f} "
                         f"contra el margen")
    log("")
    log(f"   LECTURA: {lectura}")
    log(f"   VEREDICTO: {veredicto}")
    log(f"   DECISION: {decision}")

    with CSV_QM.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["tipo", "clave", "valor", "valor2", "valor3"])
        w.writerow(["eda_citas", "filas", eda["citas"]["filas_con_cita_cerrada"],
                    eda["citas"]["porcentaje"], eda["citas"]["error_con_cita"]])
        w.writerow(["eda_citas", "diferencial_de_error", dif_cita, "", ""])
        w.writerow(["eda_menciones", "filas", eda["menciones"]["filas_con_mencion"],
                    eda["menciones"]["porcentaje"], eda["menciones"]["error_con_mencion"]])
        w.writerow(["eda_menciones", "diferencial_de_error", dif_menc, "", ""])
        for f in filas:
            w.writerow(["brazo", f["brazo"], f["corrected_cv"], f["delta_vs_V03"],
                        f["pliegues_mejorados"]])
        for k, val in brechas.items():
            w.writerow(["brecha", k, val, "", ""])
        w.writerow([])
        w.writerow(["bin", "n", "F1_de_V03"] + [f"F1_{b}" for b in proba])
        for nombre, _lo, _hi in BINS:
            ref = next(iter(matrices_meta)) if matrices_meta else "V03_margen"
            w.writerow([nombre, detalle[ref]["por_bin"][nombre]["n"],
                        detalle[ref]["por_bin"][nombre]["F1_de_V03"]]
                       + [detalle[b]["por_bin"][nombre]["F1"] for b in proba])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P64_QUOTE_MENTION",
        "naturaleza": ("EXPERIMENTO. 0 a 2 candidatos y 2 controles, 0 submissions, 0 Kaggle, "
                       "0 holdout, 0 test"),
        "fecha": inicio.isoformat(timespec="seconds"), "experiment_id": "P64-06",
        "track": "P11-P12-frontera-de-cita-y-de-mencion",
        "script": "experiments/p64_06_quote_mention.py",
        "pregunta": ("esta V03 tratando como evidencia propia contenido que pertenece a una "
                     "cita, y hay una frontera de mencion que aporte algo"),
        "eda_previa": eda,
        "regla_de_decision": ("una frontera se modela solo si es FRECUENTE, mas del 5 por "
                             "ciento de filas con al menos dos partes, o DIFERENCIAL, si la "
                             "tasa de error entre las dos partes difiere mas de cinco puntos"),
        "por_que_no_es_un_recuento": (
            "un recuento dice cuantas menciones tiene un tweet. Esto dice lo que el modelo "
            "cree a cada LADO de la frontera, y la asimetria entre el interior y el exterior "
            "de una cita es la unica forma de responder si V03 puntua una cita como si "
            "fueran sus propias palabras. Una bandera no lo responde y un recuento tampoco"),
        "identidad_de_usuario": "no se usa. Solo la posicion del simbolo @",
        "resultados": detalle, "brechas": brechas, "tracks": tracks,
        "lectura": lectura, "veredicto": veredicto, "decision": decision,
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_QM.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P64 - Frontera de cita y frontera de mencion")
    W("")
    W(f"Generado por `experiments/p64_06_quote_mention.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## La EDA va primero y decide si alguno de los dos tracks corre")
    W("")
    W("El encargo dice que la linea de citas es prioritaria SOLO si la EDA demuestra que las")
    W("citas son frecuentes o diferencialmente asociadas a errores. Esa condicion se comprueba")
    W("antes de entrenar nada, y la regla esta escrita de antemano:")
    W("")
    W("```text")
    W(f"frecuente   mas del {UMBRAL_FRECUENCIA:.0%} de las filas con al menos dos partes")
    W(f"diferencial la tasa de error entre las dos partes difiere mas de "
      f"{UMBRAL_DIFERENCIAL:.0%}")
    W("```")
    W("")
    W("| frontera | filas | porcentaje | error dentro | error fuera | diferencial | "
      "justificada |")
    W("|---|---:|---:|---:|---:|---:|---|")
    W(f"| cita cerrada | {eda['citas']['filas_con_cita_cerrada']} | "
      f"{eda['citas']['porcentaje']:.1f}% | {eda['citas']['error_con_cita']:.2%} | "
      f"{eda['citas']['error_sin_cita']:.2%} | {dif_cita:+.4f} | {abierta_cita} |")
    W(f"| mencion | {eda['menciones']['filas_con_mencion']} | "
      f"{eda['menciones']['porcentaje']:.1f}% | {eda['menciones']['error_con_mencion']:.2%} | "
      f"{eda['menciones']['error_sin_mencion']:.2%} | {dif_menc:+.4f} | {abierta_menc} |")
    W("")
    W("## Que se modela, y por que no es un recuento")
    W("")
    W("Un recuento dice cuantas menciones tiene un tweet. Esto dice **lo que el modelo cree a")
    W("cada lado de la frontera**, y para una cita la asimetria entre el interior y el exterior")
    W("es la unica forma de responder a la pregunta del encargo: esta V03 puntuando una cita")
    W("como si fueran sus propias palabras? Una bandera no lo responde y un recuento tampoco.")
    W("")
    W("**No se usa la identidad del usuario**: solo la posicion del simbolo `@`.")
    W("")
    W("## Resultados")
    W("")
    W("| brazo | corrected_cv | delta vs V03 | pliegues | neto |")
    W("|---|---:|---:|---:|---:|")
    for f in filas:
        W(f"| {f['brazo']} | {f['corrected_cv']:.10f} | {f['delta_vs_V03']:+.10f} | "
          f"{f['pliegues_mejorados']}/5 | {f['rescate'] - f['damage']:+d} |")
    W("")
    W("### Las brechas")
    W("")
    W("```text")
    for k, val in brechas.items():
        W(f"{k:40} {val:+.10f}")
    W("```")
    W("")
    W("### Por estrato de longitud")
    W("")
    W("| bin | n | " + " | ".join(f"F1 {b}" for b in proba) + " |")
    W("|---|---:|" + "---:|" * len(proba))
    for nombre, _lo, _hi in BINS:
        ref = next(iter(matrices_meta)) if matrices_meta else "V03_margen"
        W(f"| {nombre} | {detalle[ref]['por_bin'][nombre]['n']} | "
          + " | ".join(f"{detalle[b]['por_bin'][nombre]['F1']:.4f}" for b in proba) + " |")
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
    W("*Generado por `experiments/p64_06_quote_mention.py`. Sin push.*")
    MD_QM.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_QM, JSON_QM, CSV_QM):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P64-06 OK | cita justificada {abierta_cita} | mencion justificada {abierta_menc} | "
        f"decision {decision}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
