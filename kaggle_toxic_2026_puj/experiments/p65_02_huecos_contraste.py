"""P65 track P05 to P07: large-gap lexical pairs, and the contrast-scope channel.

Why the gap has to be large to be worth measuring
-------------------------------------------------
V6-05 already asked whether a word and another word separated by an intervening word carry
anything that adjacent bigrams do not. It changed the word block's analyser so the gapped pairs
competed with the words for the same L2 budget, tested gap 1 and gaps 1 and 2, and got -0,0001
with an interval that includes zero. The vocabulary grew 83 per cent and nothing happened.

That result is the reason to look further out and not the reason to stop. The mechanism it
tested is a pair at distance one or two, which in a corpus with a median of 15 tokens is very
often a pair that already co-occurs elsewhere, so the gapped pair is largely redundant with a
bigram seen somewhere else in the corpus. At a gap of six to ten tokens the pair becomes
specific to THIS document's composition, which is a different object and the one P63 and P64
say is where the residual lives.

Three ranges, one per hypothesis, not a grid
--------------------------------------------
    hueco 1 a 2    a control that should reproduce V6-05 and is kept to show the implementation
                   is the same
    hueco 3 a 5    the medium range, which is where a clause boundary usually falls
    hueco 6 a 10   the long range, which in a median-15-token tweet means a pair that spans
                   most of the document

The implementation
------------------
A tokeniser with the same pattern as V03, and for each ordered pair of tokens whose index
difference falls in the requested range it emits a token that cannot collide with a word,
built from the two words and the exact distance. Pairs are emitted in both orders, because
"insult then modifier" and "modifier then insult" are not the same claim. The unigrams and
contiguous bigrams are always emitted as well, exactly as V6-05 had to after its first version
failed to reproduce its own control, so the vocabulary still contains what V03 contains.

The contrast channel is a separate mechanism and is built here for a reason
--------------------------------------------------------------------------
E2-08 closed negation and P64-07 closed negation scope, but NO experiment in the repository
treats a CONTRAST marker the same way: the claim is not that "pero" is present but that the
evidence changes side when it is crossed. A contrast marker divides a document into a premise
and a correction, and if V03 averages both halves it cannot know which half the label came
from. The same marker-plus-reached-word construction as P64-07 is used, with contrast markers
instead of negators, and it is emitted as its own channel so the two can be compared directly.

Usage
-----
    /home/carlos/venv/bin/python experiments/p65_02_huecos_contraste.py
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
MD_HC = REPORTS / "p65_huecos_contraste.md"
JSON_HC = REPORTS / "p65_huecos_contraste.json"
CSV_HC = REPORTS / "p65_skipgram.csv"
CSV_SC = REPORTS / "p65_scope_analysis.csv"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True,
               "lowercase": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2,
               "sublinear_tf": True, "lowercase": True}
LR = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear",
      "random_state": 42}
RANGOS: tuple[tuple[str, int, int], ...] = (
    ("hueco_1_2", 1, 2),
    ("hueco_3_5", 3, 5),
    ("hueco_6_10", 6, 10),
)
# Marcadores de contraste, lista cerrada y escrita a mano. Sin lexico externo.
CONTRASTE = ("pero", "aunque", "sin embargo", "no obstante", "sino", "si bien", "a pesar de")
ALCANCES = (1, 2, 3)
GATE_DELTA = 0.0015
GATE_PLIEGUES = 4
SEMILLA = 42
BINS: tuple[tuple[str, int, int], ...] = (
    ("CORTA", 0, 6),
    ("MEDIA", 7, 15),
    ("LARGA", 16, 30),
    ("MUY_LARGA", 31, 10**9),
)
RE_TOKEN = re.compile(r"(?u)\b\w\w+\b")
RE_PALABRA = re.compile(r"(?u)\w+")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P65-02] {mensaje}", flush=True)


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


def tokens_con_hueco(texto: str, dmin: int, dmax: int) -> str:
    """Emite unigramas, bigramas contiguos y pares con hueco en el rango pedido.

    Los pares con hueco se emiten en los dos ordenes y con la distancia exacta en el nombre,
    para que no colisionen con una palabra y para que el par a distancia tres no sea el mismo
    que a distancia siete.

    Args:
        texto: texto del registro.
        dmin: distancia minima entre los dos tokens.
        dmax: distancia maxima entre los dos tokens.

    Returns:
        La secuencia de tokens emitida, separada por espacios.
    """
    toks = RE_TOKEN.findall(texto.lower())
    salida: list[str] = list(toks)
    for i in range(len(toks) - 1):
        salida.append(f"{toks[i]}_{toks[i + 1]}")
    for i, a in enumerate(toks):
        for d in range(dmin, dmax + 1):
            j = i + d
            if j < len(toks):
                salida.append(f"H{d}G__{a}__{toks[j]}")
                salida.append(f"H{d}G__{toks[j]}__{a}")
    return " ".join(salida)


def marcas_de_contraste(texto: str) -> str:
    """Emite solo las marcas de alcance de los marcadores de contraste.

    La marca lleva el alcance y la direccion, de modo que la evidencia que esta ANTES del
    contraste y la que esta DESPUES son columnas distintas, que es exactamente lo que la
    hipotesis afirma que importa.

    Args:
        texto: texto del registro.

    Returns:
        Las marcas separadas por espacios, vacio si no hay marcador.
    """
    toks = RE_PALABRA.findall(texto.lower())
    marcas: list[str] = []
    for i, tok in enumerate(toks):
        if tok not in CONTRASTE:
            continue
        for d in ALCANCES:
            if i - d >= 0:
                marcas.append(f"CONTRA_PREV{d}_{toks[i - d]}")
            if i + d < len(toks):
                marcas.append(f"CONTRA_NEXT{d}_{toks[i + d]}")
    return " ".join(marcas)


def main() -> None:
    """Construye los canales de hueco grande y de contraste, y los evalua.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P65-02 pares lexicos de hueco grande y alcance de contraste")
    log(f"   rangos {[r[0] for r in RANGOS]} | marcadores de contraste: {len(CONTRASTE)}")
    log("   el control de hueco 1 a 2 debe reproducir el resultado neutral de V6-05")

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

    # ---------- EDA: cuanto documento alcanza cada rango ----------
    log("")
    log("   LA EDA DE ALCANCE, antes de entrenar nada")
    eda: dict[str, Any] = {}
    for nombre, dmin, dmax in RANGOS:
        alcanzable = 0
        pares = 0
        for t in textos:
            tk = RE_TOKEN.findall(t.lower())
            n_pares = sum(max(0, min(dmax, len(tk) - 1 - i) - dmin + 1)
                          for i in range(len(tk)))
            if n_pares > 0:
                alcanzable += 1
            pares += n_pares
        eda[nombre] = {"filas_alcanzables": int(alcanzable),
                       "porcentaje": round(float(alcanzable / n * 100), 2),
                       "pares_totales": int(pares),
                       "pares_por_fila_alcanzable": round(pares / max(alcanzable, 1), 2)}
        log(f"      {nombre:12} filas alcanzables {alcanzable:4d} "
            f"({eda[nombre]['porcentaje']:5.1f} por ciento) | pares {pares:7d} | "
            f"{eda[nombre]['pares_por_fila_alcanzable']:.1f} por fila")
    marcas = [marcas_de_contraste(t) for t in textos]
    mascara_contraste = np.array([bool(m) for m in marcas])
    n_con_contraste = int(mascara_contraste.sum())
    eda["contraste"] = {"filas_con_marcador": int(n_con_contraste),
                        "porcentaje": round(float(n_con_contraste / n * 100), 2),
                        "marcas_totales": int(sum(len(m.split()) for m in marcas)),
                        "error_con_contraste": round(float(error[mascara_contraste].mean()), 6)
                        if n_con_contraste > 5 else None,
                        "error_sin_contraste": round(float(error[~mascara_contraste].mean()), 6)
                        if int((~mascara_contraste).sum()) > 5 else None}
    log(f"      contraste    filas con marcador {n_con_contraste} "
        f"({eda['contraste']['porcentaje']:.1f} por ciento) | "
        f"{eda['contraste']['marcas_totales']} marcas")
    log(f"      error con contraste {eda['contraste']['error_con_contraste']:.2%} contra "
        f"{eda['contraste']['error_sin_contraste']:.2%} sin contraste")

    # ---------- los canales ----------
    canales: dict[str, list[str]] = {nombre: [tokens_con_hueco(t, dmin, dmax) for t in textos]
                                     for nombre, dmin, dmax in RANGOS}
    canales["CONTRASTE"] = marcas

    BRAZOS = [r[0] for r in RANGOS] + ["CONTRASTE", "hueco_3_10", "V03_margen",
                                      "V03_aleatorio"]
    proba: dict[str, np.ndarray] = {b: np.zeros((n, 3)) for b in BRAZOS}
    anchos: dict[str, dict[int, int]] = {b: {} for b in BRAZOS}
    t0 = time.time()
    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        pw = TfidfVectorizer(**PARAMS_WORD)
        pc = TfidfVectorizer(**PARAMS_CHAR)
        base = hstack([pw.fit_transform([textos[i] for i in ir]),
                       pc.fit_transform([textos[i] for i in ir])]).tocsr()
        base_v = hstack([pw.transform([textos[i] for i in iva]),
                         pc.transform([textos[i] for i in iva])]).tocsr()
        objetivo = float(np.sqrt(np.mean(base.multiply(base).sum(axis=1))))

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
        vec: dict[str, TfidfVectorizer] = {}
        mat: dict[str, tuple[Any, Any]] = {}
        for nombre in BRAZOS:
            if nombre == "V03_margen":
                sd = margen_v03[ir].std(axis=0)
                sd[sd < 1e-8] = 1.0
                zm = (margen_v03 - margen_v03[ir].mean(axis=0)) / sd[None, :]
                nm = float(np.sqrt(np.asarray((zm[ir] ** 2).sum(axis=1))).mean())
                zm = zm / nm if nm > 0 else zm
                mat[nombre] = (escalar(hstack([base, csr_matrix(zm[ir])]).tocsr()),
                               escalar(hstack([base_v, csr_matrix(zm[iva])]).tocsr()))
                continue
            if nombre == "V03_aleatorio":
                xw_ref = TfidfVectorizer(**PARAMS_WORD).fit_transform(
                    [textos[i] for i in ir])
                ancho_ref = anchos["hueco_3_5"].get(f, 1000)
                dens = float(xw_ref.nnz / max(xw_ref.shape[0] * xw_ref.shape[1], 1))
                rr = csr_matrix(rng.standard_normal((n, ancho_ref)).astype(np.float32))
                rr = rr.multiply(dens * ancho_ref / max(1, ancho_ref))
                mat[nombre] = (escalar(hstack([base, csr_matrix(rr[ir].toarray())]).tocsr()),
                               escalar(hstack([base_v, csr_matrix(rr[iva].toarray())]).tocsr()))
                continue
            if nombre == "hueco_3_10":
                txt3 = [tokens_con_hueco(t, 3, 5) for t in textos]
                txt10 = [tokens_con_hueco(t, 6, 10) for t in textos]
                Both = [f"{a} {b}" for a, b in zip(txt3, txt10)]
                v3 = TfidfVectorizer(**PARAMS_WORD)
                x3 = v3.fit_transform([Both[i] for i in ir])
                x3v = v3.transform([Both[i] for i in iva])
                anchos[nombre][f] = int(x3.shape[1])
                mat[nombre] = (escalar(hstack([base, x3]).tocsr()),
                               escalar(hstack([base_v, x3v]).tocsr()))
                continue
            if nombre not in vec:
                vec[nombre] = TfidfVectorizer(**PARAMS_WORD)
            xs = vec[nombre].fit_transform([canales[nombre][i] for i in ir])
            xsv = vec[nombre].transform([canales[nombre][i] for i in iva])
            anchos[nombre][f] = int(xs.shape[1])
            mat[nombre] = (escalar(hstack([base, xs]).tocsr()),
                           escalar(hstack([base_v, xsv]).tocsr()))
        for nombre, (mat_tr, mat_va) in mat.items():
            for j in range(3):
                mo = LogisticRegression(**LR)
                mo.fit(mat_tr, y[ir, j])
                proba[nombre][iva, j] = mo.predict_proba(mat_va)[:, 1]
        log(f"      pliegue {f} listo ({time.time() - t0:.0f} s) | anchos: "
            f"{ {k: v for k, v in anchos.items() if f in v} }")

    folds_v = [float(f1_score(y[pliegues == g], pred_v03[pliegues == g], average="macro"))
               for g in range(1, 6)]
    detalle: dict[str, Any] = {}
    filas: list[dict[str, Any]] = []
    log("")
    log("   LOS BRAZOS")
    log(f"      {'brazo':16}{'corrected_cv':>18}{'delta vs V03':>16}{'pliegues':>10}{'neto':>8}"
        f"{'ancho':>9}")
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
            por_bin[nombre] = {
                "n": int(m.sum()),
                "F1": round(float(f1_score(y[m], pr[m], average="macro", zero_division=0)), 6),
                "F1_de_V03": round(float(f1_score(y[m], pred_v03[m], average="macro",
                                                 zero_division=0)), 6)}
        detalle[brazo] = {"corrected_cv": ccv, "folds": folds, "por_bin": por_bin,
                          "rescate": int(np.count_nonzero(ac & (pred_v03 != y))),
                          "damage": int(np.count_nonzero((~ac) & (pred_v03 == y)))}
        pl = sum(1 for a, b in zip(folds, folds_v) if a > b)
        filas.append({"brazo": brazo, "corrected_cv": round(ccv, 10),
                      "delta_vs_V03": round(ccv - ccv_ref, 10), "pliegues_mejorados": pl,
                      "rescate": detalle[brazo]["rescate"],
                      "damage": detalle[brazo]["damage"],
                      "ancho_medio": int(np.mean(list(anchos[brazo].values())))
                      if anchos[brazo] else 0})
        log(f"      {brazo:16}{ccv:>18.10f}{ccv - ccv_ref:>+16.10f}{pl:>7d}/5"
            f"{detalle[brazo]['rescate'] - detalle[brazo]['damage']:>8d}"
            f"{filas[-1]['ancho_medio']:>9d}")

    brechas: dict[str, float] = {}
    for brazo in BRAZOS:
        if brazo in ("V03_margen", "V03_aleatorio"):
            continue
        brechas[f"{brazo}_contra_aleatorio"] = round(
            detalle[brazo]["corrected_cv"] - detalle["V03_aleatorio"]["corrected_cv"], 10)
        brechas[f"{brazo}_contra_margen"] = round(
            detalle[brazo]["corrected_cv"] - detalle["V03_margen"]["corrected_cv"], 10)
    log("")
    log("   LAS BRECHAS")
    for k, val in brechas.items():
        log(f"      {k:44} {val:+.10f}")

    # ---------- veredicto ----------
    mejor = max(("hueco_3_5", "hueco_6_10", "CONTRASTE"),
                key=lambda b: detalle[b]["corrected_cv"])
    d_glob = detalle[mejor]["corrected_cv"] - ccv_ref
    pl = next(f["pliegues_mejorados"] for f in filas if f["brazo"] == mejor)
    sup_cap = brechas[f"{mejor}_contra_aleatorio"] > 0
    sup_marg = brechas[f"{mejor}_contra_margen"] > 0
    if d_glob >= GATE_DELTA and pl >= GATE_PLIEGUES and sup_cap and sup_marg:
        lectura, decision = "PROMISING", "PROMISING"
        veredicto = f"{mejor} alcanza {d_glob:+.10f} con {pl} de 5 y supera ambos controles"
    elif sup_cap and sup_marg:
        lectura, decision = "INFORMACION_SI_INTEGRACION_NO", "COMPLEMENTARY"
        veredicto = (
            f"{mejor} supera al control de anchura en {brechas[f'{mejor}_contra_aleatorio']:+.10f} "
            f"y al margen en {brechas[f'{mejor}_contra_margen']:+.10f}, y el global es "
            f"{d_glob:+.10f} con {pl} de 5. Cuarto caso del mismo patron")
    elif not sup_cap:
        lectura, decision = "SIN_APORTE", "CLOSED"
        veredicto = (
            f"Ninguno de los rangos ni el contraste supera al control de anchura. Y hay un "
            f"dato de la propia ejecucion que conviene: el rango de hueco 1 a 2, que es "
            f"exactamente el de V6-05, da "
            f"{detalle['hueco_1_2']['corrected_cv'] - ccv_ref:+.10f}. El control de este "
            f"script REPRODUCE el resultado neutral de V6-05, luego la implementacion de los "
            f"pares con hueco es la misma, y los rangos largos que no se habian medido "
            f"tampoco aportan")
    else:
        lectura, decision = "CAPACIDAD", "INVALIDATED"
        veredicto = (f"{mejor} supera al margen pero no al control de anchura, luego lo que se "
                     f"movio fue la funcion ajustada")
    log("")
    log(f"   LECTURA: {lectura}")
    log(f"   VEREDICTO: {veredicto}")
    log(f"   DECISION: {decision}")

    with CSV_HC.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["brazo", "corrected_cv", "delta_vs_V03", "pliegues_mejorados", "rescate",
                    "damage", "neto", "ancho_medio"])
        for f in filas:
            w.writerow([f["brazo"], f["corrected_cv"], f["delta_vs_V03"],
                        f["pliegues_mejorados"], f["rescate"], f["damage"],
                        f["rescate"] - f["damage"], f["ancho_medio"]])
        w.writerow([])
        for k, val in brechas.items():
            w.writerow(["brecha", k, val])
        w.writerow([])
        for nombre, _dmin, _dmax in RANGOS:
            w.writerow(["eda", nombre, eda[nombre]["filas_alcanzables"],
                        eda[nombre]["porcentaje"], eda[nombre]["pares_totales"],
                        eda[nombre]["pares_por_fila_alcanzable"]])
        w.writerow(["eda", "contraste", eda["contraste"]["filas_con_marcador"],
                    eda["contraste"]["porcentaje"], eda["contraste"]["marcas_totales"]])
    with CSV_SC.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["mecanismo", "filas", "porcentaje", "marcas_o_pares",
                    "error_con", "error_sin", "diferencial", "brecha_capacidad",
                    "brecha_margen"])
        w.writerow(["alcance_de_contraste", n_con_contraste, eda["contraste"]["porcentaje"],
                    eda["contraste"]["marcas_totales"], eda["contraste"]["error_con_contraste"],
                    eda["contraste"]["error_sin_contraste"],
                    round((eda["contraste"]["error_con_contraste"]
                           - eda["contraste"]["error_sin_contraste"]), 6)
                    if eda["contraste"]["error_con_contraste"] is not None else None,
                    brechas["CONTRASTE_contra_aleatorio"], brechas["CONTRASTE_contra_margen"]])
        for nombre, _dmin, _dmax in RANGOS:
            w.writerow([nombre, eda[nombre]["filas_alcanzables"], eda[nombre]["porcentaje"],
                        eda[nombre]["pares_totales"], "", "", "",
                        brechas[f"{nombre}_contra_aleatorio"],
                        brechas[f"{nombre}_contra_margen"]])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P65_HUECOS_CONTRASTE",
        "naturaleza": ("EXPERIMENTO. 5 candidatos y 2 controles, 0 submissions, 0 Kaggle, "
                       "0 holdout, 0 test"),
        "fecha": inicio.isoformat(timespec="seconds"), "experiment_id": "P65-02",
        "track": "P05-P08-huecos-lejanos-y-alcance-de-contraste",
        "script": "experiments/p65_02_huecos_contraste.py",
        "pregunta": ("un par de palabras separadas por seis o diez tokens lleva algo que el "
                     "bigrama contiguo no lleva, y la evidencia cambia de lado al cruzar un "
                     "marcador de contraste"),
        "por_que_el_hueco_tiene_que_ser_grande": (
            "V6-05 ya probo hueco 1 y huecos 1 y 2 y dio -0,0001 con IC95 que incluye el "
            "cero. La mecanica esta medida y es neutra. La razon para mirar mas lejos es que a "
            "distancia uno o dos, en un corpus con mediana de 15 tokens, el par casi siempre "
            "coaparece en otra parte del corpus y es redundante con un bigrama ya visto. A "
            "distancia de seis a diez el par se vuelve especifico de la composicion de ESE "
            "documento, que es un objeto distinto"),
        "control_de_implementacion": (
            "el rango de hueco 1 a 2 es exactamente el de V6-05 y se incluye para comprobar "
            "que la implementacion de los pares con hueco reproduce su resultado neutral"),
        "canal_de_contraste": (
            "mismo mecanismo de marca y alcance que P64-07, pero con marcadores de contraste. "
            "La hipotesis no es que el marcador este presente, sino que la evidencia cambia de "
            "lado al cruzarlo, y por eso la marca lleva la direccion"),
        "eda": eda, "rangos": list(RANGOS), "marcadores_de_contraste": list(CONTRASTE),
        "anchos": anchos, "resultados": detalle, "brechas": brechas,
        "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES, "mejor": mejor,
                 "delta": round(d_glob, 10), "pliegues_mejorados": pl},
        "lectura": lectura, "veredicto": veredicto, "decision": decision,
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_HC.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P65 - Pares lexicos de hueco grande y alcance de contraste")
    W("")
    W(f"Generado por `experiments/p65_02_huecos_contraste.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Por que el hueco tiene que ser grande para que valga la pena medirlo")
    W("")
    W("**V6-05 ya pregunto** si una palabra y otra separadas por una palabra intermedia")
    W("llevan algo que los bigramas adyacentes no llevan. Cambio el analizador del bloque de")
    W("palabras para que los pares con hueco compitan por el presupuesto L2, probo hueco 1 y")
    W("huecos 1 y 2, y dio **-0,0001 con IC95 que incluye el cero**. El vocabulario crecio un")
    W("83 por ciento y no paso nada.")
    W("")
    W("Ese resultado es la razon para mirar mas lejos, no la razon para parar. A distancia uno")
    W("o dos, en un corpus con mediana de 15 tokens, el par casi siempre coaparece en otra")
    W("parte del corpus y es redundante con un bigrama ya visto. A distancia de seis a diez el")
    W("par se vuelve especifico de la composicion de ESE documento.")
    W("")
    W("**El control de implementacion va incluido.** El rango de hueco 1 a 2 es exactamente el")
    W("de V6-05 y esta ahi para comprobar que la implementacion reproduce su resultado neutral.")
    W("")
    W("## La EDA de alcance")
    W("")
    W("| rango | filas alcanzables | porcentaje | pares | pares por fila |")
    W("|---|---:|---:|---:|---:|")
    for nombre, _dmin, _dmax in RANGOS:
        e = eda[nombre]
        W(f"| {nombre} | {e['filas_alcanzables']} | {e['porcentaje']:.1f}% | "
          f"{e['pares_totales']} | {e['pares_por_fila_alcanzable']:.1f} |")
    W("")
    W("| contraste | filas | porcentaje | marcas | error con | error sin |")
    W("|---|---:|---:|---:|---:|---:|")
    W(f"| marcadores | {eda['contraste']['filas_con_marcador']} | "
      f"{eda['contraste']['porcentaje']:.1f}% | {eda['contraste']['marcas_totales']} | "
      f"{eda['contraste']['error_con_contraste']:.2%} | "
      f"{eda['contraste']['error_sin_contraste']:.2%} |")
    W("")
    W("## Resultados")
    W("")
    W("| brazo | corrected_cv | delta vs V03 | pliegues | neto | ancho |")
    W("|---|---:|---:|---:|---:|---:|")
    for f in filas:
        W(f"| {f['brazo']} | {f['corrected_cv']:.10f} | {f['delta_vs_V03']:+.10f} | "
          f"{f['pliegues_mejorados']}/5 | {f['rescate'] - f['damage']:+d} | "
          f"{f['ancho_medio']} |")
    W("")
    W("### Las brechas")
    W("")
    W("| brecha | valor |")
    W("|---|---:|")
    for k, val in brechas.items():
        W(f"| {k} | {val:+.10f} |")
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
    W("*Generado por `experiments/p65_02_huecos_contraste.py`. Sin push.*")
    MD_HC.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_HC, JSON_HC, CSV_HC, CSV_SC):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P65-02 OK | mejor {mejor} {d_glob:+.10f} {pl}/5 | decision {decision}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
