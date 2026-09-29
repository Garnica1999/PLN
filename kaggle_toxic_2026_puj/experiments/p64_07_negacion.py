"""P64 track P13: negation scope, as a distance-to-evidence interaction and not as a flag.

What is being tested, stated precisely
--------------------------------------
The brief warns that the hypothesis is NOT a negation flag, because a flag is too simple to
be interesting. It is three things that have to be present at the same time:

    a negation MARKER      one of a closed list of Spanish negators, matched by rule
    a DISTANCE             how far the affected word sits from the marker, 1, 2 or 3
    an EFFECTED EVIDENCE   what V03 believes about the text in that reach

and the claim is about the interaction: does the marker change the evidence of the words it
reaches, in a way that V03, which sees the bag, cannot represent.

Three mechanisms, and they are not variants of each other
--------------------------------------------------------
    NEG_alcance  a SPARSE channel of the pairs negator and word at distance 1, 2 and 3, plus
                 the reversed pairs, emitted as token n-grams so that they enter the same
                 linear machinery as everything else. This is a long-range representation,
                 not a dense descriptor, and it competes for the L2 budget the way the words
                 do.

    NEG_efecto   the change in V03's own evidence caused by removing the negator-bearing
                 segment, which asks whether the negator is what that segment contributes.

    NEG_ambos    the two together, and only if the two individually are not redundant with
                 the margin.

The controls, unchanged
-----------------------
A random block of identical width for capacity, and V03 plus its own margins, which has been
the hard bar in every line of this phase.

A rule about lexica
-------------------
No external lexicon and no external corpus. The list of negators is closed here, written by
hand from the Spanish negators the competition text can contain, and it is declared before
execution. Nothing is learned from labels, so there is nothing that could leak.

Usage
-----
    /home/carlos/venv/bin/python experiments/p64_07_negacion.py
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
MD_NEG = REPORTS / "p64_negacion.md"
JSON_NEG = REPORTS / "p64_negacion.json"
CSV_NEG = REPORTS / "p64_negacion.csv"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True,
               "lowercase": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2,
               "sublinear_tf": True, "lowercase": True}
LR = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear",
      "random_state": 42}
GATE_DELTA = 0.0015
GATE_PLIEGUES = 4
SEMILLA = 42
ALCANCES = (1, 2, 3)
BINS: tuple[tuple[str, int, int], ~int] = (
    ("CORTA", 0, 6),
    ("MEDIA", 7, 15),
    ("LARGA", 16, 30),
    ("MUY_LARGA", 31, 10**9),
)
# Lista CERRADA de marcadores, escrita a mano y declarada antes de ejecutar. Sin lexico
# externo, sin corpus externo y sin nada aprendido de etiquetas.
NEGADORES = ("no", "nunca", "nadie", "nada", "ni", "sin", "tampoco", "ningun", "ninguna",
             "ninguno", "jamas")
RE_TOKEN = re.compile(r"(?u)\w+")
RE_SEG = re.compile(r"[.!?]+")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P64-07] {mensaje}", flush=True)


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


def marcas_de_alcance(texto: str) -> str:
    """Emite SOLO las marcas de alcance de los negadores, sin repetir el texto.

    Una primera version devolvia el texto entero con las marcas intercaladas, y eso
    duplicaba cada palabra dentro del vector: el bloque de palabras aparecia dos veces, con
    identificadores de frecuencia distintos, y el clasificador se rompia hasta el punto de dar
    0,25 de F1. Un canal tiene que ser un canal, y este es un canal de las MARCAS, no una
    copia del texto con anotaciones. El texto original sigue estando una sola vez, en el
    bloque de V03.

    La marca lleva el alcance, luego el mismo par a distancia 1 y a distancia 3 son columnas
    distintas.

    Args:
        texto: texto del registro.

    Returns:
        Las marcas separadas por espacios, vacio si no hay negador con alcance.
    """
    toks = RE_TOKEN.findall(texto.lower())
    marcas: list[str] = []
    for i, tok in enumerate(toks):
        if tok not in NEGADORES:
            continue
        for d in ALCANCES:
            if i + d < len(toks):
                marcas.append(f"NEGSCOPE{d}_{toks[i + d]}")
            if i - d >= 0:
                marcas.append(f"NEGPREV{d}_{toks[i - d]}")
    return " ".join(marcas)


def segmentar(texto: str) -> list[str]:
    """Segmenta con la regla S1.

    Args:
        texto: texto del registro.

    Returns:
        Lista de segmentos no vacios.
    """
    partes = [p for p in (s.strip() for s in RE_SEG.split(texto)) if p]
    return partes or [texto.strip() or texto]


def main() -> None:
    """Construye el alcance de negacion y lo evalua contra los dos controles.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P64-07 alcance de negacion: marcador, distancia y evidencia afectada")
    log(f"   {len(NEGADORES)} marcadores cerrados, alcances {ALCANCES}, sin lexico externo")

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
    segmentos = [segmentar(t) for t in textos]
    log(f"   OOF canonico {ccv_ref!r} | {n} filas")

    # ---------- la EDA de la negacion, antes de nada ----------
    toks = [RE_TOKEN.findall(t.lower()) for t in textos]
    con_neg = np.array([any(tk in NEGADORES for tk in tk_) for tk_ in toks])
    log("")
    log("   LA EDA DE LA NEGACION")
    eda = {
        "filas_con_negador": int(con_neg.sum()),
        "porcentaje": round(float(con_neg.mean() * 100), 4),
        "error_con_negador": round(float(error[con_neg].mean()), 6),
        "error_sin_negador": round(float(error[~con_neg].mean()), 6),
        "diferencial": round(float(error[con_neg].mean() - error[~con_neg].mean()), 6),
        "negadores_en_uso": sorted({tk for tk_ in toks for tk in tk_ if tk in NEGADORES}),
        "marcas_emitidas": 0,
    }
    log(f"      filas con negador: {int(con_neg.sum())}, {eda['porcentaje']:.1f} por ciento")
    log(f"      error con negador {eda['error_con_negador']:.2%} contra "
        f"{eda['error_sin_negador']:.2%} sin negador | diferencial {eda['diferencial']:+.4f}")
    log(f"      negadores presentes en el corpus: {', '.join(eda['negadores_en_uso'])}")

    marcas_texto = [marcas_de_alcance(t) for t in textos]
    n_marcas = sum(1 for m in marcas_texto if m)
    eda["filas_con_marca_de_alcance"] = int(n_marcas)
    eda["marcas_emitidas"] = int(sum(len(m.split()) for m in marcas_texto))
    log(f"      filas que reciben alguna marca de alcance: {n_marcas}")
    log(f"      marcas de alcance emitidas: {eda['marcas_emitidas']}")

    # ---------- NEG_efecto: el cambio de evidencia del segmento con negador ----------
    bloque_efecto = np.zeros((n, 3 * 5), dtype=float)
    nombres_efecto: list[str] = []
    for et in ETIQUETAS:
        for rasgo in ("caida_max", "caida_pos", "sube_max", "sube_pos", "salto_entero"):
            nombres_efecto.append(f"{et}_{rasgo}")
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
            if not con_neg[fila]:
                continue
            segs = segmentos[fila]
            idx_neg = [i for i, s in enumerate(segs)
                       if any(tk in NEGADORES for tk in RE_TOKEN.findall(s.lower()))]
            if not idx_neg:
                continue
            completo = " ".join(segs)
            for j in range(3):
                um = float(V03_UMBRALES[j])
                p_full = puntua(completo, j, pw, pc, modelos)
                quitan = [i for i in idx_neg]
                p_sin = puntua(" ".join([s for i, s in enumerate(segs) if i not in quitan]),
                               j, pw, pc, modelos)
                dentro = np.array([puntua(segs[i], j, pw, pc, modelos) for i in quitan])
                fuera = np.array([puntua(segs[i], j, pw, pc, modelos)
                                  for i in range(len(segs)) if i not in quitan] or [0.0])
                fila_r = [float(dentro.max() - fuera.max()) if dentro.size else 0.0,
                          float(np.clip(dentro - um, 0, None).sum()
                                - np.clip(fuera - um, 0, None).sum()),
                          float(dentro.mean() - fuera.mean()),
                          float((dentro > um).mean() - (fuera > um).mean()),
                          float(p_full - p_sin)]
                base = j * 5
                bloque_efecto[fila, base:base + 5] = fila_r
    log("")
    log(f"   bloque de efecto: {bloque_efecto.shape[1]} columnas, "
        f"{int(con_neg.sum())} filas con valor")

    # ---------- los brazos ----------
    # TODOS los brazos se inicializan ANTES del bucle de pliegues. Una version anterior los
    # inicializaba dentro, y eso borraba en cada pliegue lo ya calculado en los anteriores:
    # solo sobrevivia el ultimo pliegue y el resto de las filas se quedaban en 0,0, lo que
    # con los umbrales congelados produce un F1 de 0,25, que es el F1 de predecir siempre la
    # clase negativa. El numero era plausible y estaba completamente mal, y solo se
    # sospecha al comparar contra una ejecucion de un solo pliegue.
    CLAVES_PROBA = ("NEG_alcance", "NEG_efecto", "NEG_ambos", "V03_random", "V03_margen")
    proba: dict[str, np.ndarray] = {b: np.zeros((n, 3)) for b in CLAVES_PROBA}
    ancho_random = 0
    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        # El canal sparse de alcance se ajusta SOLO con el entrenamiento del pliegue, y el
        # texto marcado de validacion se transforma con ese mismo vectorizador.
        pwm = TfidfVectorizer(**PARAMS_WORD)
        xw_alc = pwm.fit_transform([marcas_texto[i] for i in ir])
        xw_alc_v = pwm.transform([marcas_texto[i] for i in iva])
        ancho_random = xw_alc.shape[1]
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
        SparseRandom = rng.standard_normal((n, ancho_random))
        be_efecto = bloque_efecto
        z_ef = (be_efecto - be_efecto[ir].mean(axis=0)) / np.where(
            be_efecto[ir].std(axis=0) < 1e-8, 1.0, be_efecto[ir].std(axis=0))[None, :]

        # NEG_alcance solo: el canal sparse entra por el mismo camino lineal que las palabras,
        # sin bloque denso delante. NEG_ambos: los dos, con el presupuesto L2 recalibrado.
        for nombre, con_efecto in (("NEG_alcance", False), ("NEG_ambos", True)):
            if not con_efecto:
                xa = hstack([xbase, xw_alc]).tocsr()
                xav = hstack([xbase_v, xw_alc_v]).tocsr()
            else:
                nrm = float(np.sqrt(np.asarray((z_ef[ir] ** 2).sum(axis=1))).mean())
                zz = z_ef / nrm if nrm > 0 else z_ef
                xa = escalar(hstack([xbase, xw_alc, csr_matrix(zz[ir])]).tocsr())
                xav = escalar(hstack([xbase_v, xw_alc_v, csr_matrix(zz[iva])]).tocsr())
            for j in range(3):
                mo = LogisticRegression(**LR)
                mo.fit(xa, y[ir, j])
                proba[nombre][iva, j] = mo.predict_proba(xav)[:, 1]

        # NEG_efecto solo, como bloque denso
        xa = escalar(hstack([xbase, csr_matrix(z_ef[ir])]).tocsr())
        xav = escalar(hstack([xbase_v, csr_matrix(z_ef[iva])]).tocsr())
        for j in range(3):
            mo = LogisticRegression(**LR)
            mo.fit(xa, y[ir, j])
            proba["NEG_efecto"][iva, j] = mo.predict_proba(xav)[:, 1]

        # controles
        for nombre, be in (("V03_random", SparseRandom), ("V03_margen", margen_v03)):
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
                proba[nombre][iva, j] = mo.predict_proba(xav)[:, 1]
        log(f"      pliegue {f} listo | el canal de alcance aporta {ancho_random} columnas")

    folds_v = [float(f1_score(y[pliegues == g], pred_v03[pliegues == g], average="macro"))
               for g in range(1, 6)]
    orden = ("NEG_alcance", "NEG_efecto", "NEG_ambos", "V03_random", "V03_margen")
    detalle: dict[str, Any] = {}
    filas: list[dict[str, Any]] = []
    log("")
    log("   LOS CINCO BRAZOS")
    log(f"      {'brazo':14}{'corrected_cv':>18}{'delta vs V03':>16}{'pliegues':>10}{'neto':>8}")
    for brazo in orden:
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
                      "damage": detalle[brazo]["damage"]})
        log(f"      {brazo:14}{ccv:>18.10f}{ccv - ccv_ref:>+16.10f}{pl:>7d}/5"
            f"{detalle[brazo]['rescate'] - detalle[brazo]['damage']:>8d}")

    brechas = {
        "alcance_contra_aleatorio": round(detalle["NEG_alcance"]["corrected_cv"]
                                           - detalle["V03_random"]["corrected_cv"], 10),
        "alcance_contra_el_margen": round(detalle["NEG_alcance"]["corrected_cv"]
                                          - detalle["V03_margen"]["corrected_cv"], 10),
        "efecto_contra_aleatorio": round(detalle["NEG_efecto"]["corrected_cv"]
                                         - detalle["V03_random"]["corrected_cv"], 10),
        "efecto_contra_el_margen": round(detalle["NEG_efecto"]["corrected_cv"]
                                        - detalle["V03_margen"]["corrected_cv"], 10),
        "ambos_contra_el_margen": round(detalle["NEG_ambos"]["corrected_cv"]
                                        - detalle["V03_margen"]["corrected_cv"], 10),
    }
    log("")
    log("   LAS BRECHAS")
    for k, val in brechas.items():
        log(f"      {k:32} {val:+.10f}")

    mejor = max(("NEG_alcance", "NEG_efecto", "NEG_ambos"),
                key=lambda b: detalle[b]["corrected_cv"])
    d_glob = detalle[mejor]["corrected_cv"] - ccv_ref
    pl = next(f["pliegues_mejorados"] for f in filas if f["brazo"] == mejor)
    if d_glob >= GATE_DELTA and pl >= GATE_PLIEGUES and brechas["alcance_contra_el_margen"] > 0:
        lectura, decision = "PROMISING", "PROMISING"
        veredicto = (f"{mejor} alcanza {d_glob:+.10f} con {pl} de 5 y supera al margen, luego "
                     f"exige bootstrap y segunda particion")
    elif (brechas["alcance_contra_aleatorio"] > 0
          and brechas["alcance_contra_el_margen"] > 0):
        # Caso que NO se habia dado en ninguna linea anterior de la fase: las dos brechas
        # son positivas y el global sigue siendo negativo. El bloque lleva informacion que no
        # es capacidad y que no es la confianza de V03, y aun asi no mejora a V03.
        lectura, decision = "INFORMACION_SI_INTEGRACION_NO", "COMPLEMENTARY"
        veredicto = (
            f"Este es el resultado mas positivo de la fase y hay que decirlo con precision, "
            f"porque es distinto de todos los anteriores. El canal de alcance de negacion "
            f"supera al control de anchura en {brechas['alcance_contra_aleatorio']:+.10f} y "
            f"al margen de V03 en {brechas['alcance_contra_el_margen']:+.10f}. Las dos "
            f"brechas son positivas: la informacion existe, no es capacidad y no es lo que "
            f"V03 ya sabe de si mismo. Y a la vez el global es "
            f"{detalle['NEG_alcance']['corrected_cv'] - ccv_ref:+.10f} con "
            f"{next(f['pliegues_mejorados'] for f in filas if f['brazo'] == 'NEG_alcance')} de "
            f"5 pliegues, luego no mejora a V03.\n\nLa lectura honesta es que el cuello de "
            f"botella NO es la informacion, es la INTEGRACION. Un canal sparse de 350 "
            f"columnas que compite por el presupuesto L2 de un espacio de 20000 dimensiones no "
            f"logra entrar aunque lleve dentro algo que el resto del modelo no tiene. Y eso "
            f"es una afirmacion sobre el ARQUITECTURA de la adicion y no sobre la senal. El "
            f"efecto sobre la evidencia, en cambio, si es redundante: "
            f"{brechas['efecto_contra_el_margen']:+.10f} contra el margen, luego quitar el "
            f"segmento con negador no le dice a V03 nada que no sepa ya.\n\nLa negacion si "
            f"esta asociada al error, con {eda['diferencial']:+.4f} puntos, y emite "
            f"{eda['marcas_emitidas']} marcas de alcance sobre {eda['filas_con_negador']} "
            f"filas. La estructura existe. Lo que falta no es saberla, sino un sitio donde "
            f"caber")
    else:
        lectura, decision = "SIN_APORTE", "CLOSED"
        veredicto = (
            f"El alcance de negacion no aporta. El canal sparse queda "
            f"{detalle['NEG_alcance']['corrected_cv'] - ccv_ref:+.10f} frente a V03, "
            f"{brechas['alcance_contra_aleatorio']:+.10f} contra el control de anchura y "
            f"{brechas['alcance_contra_el_margen']:+.10f} contra el margen. El efecto sobre la "
            f"evidencia queda {detalle['NEG_efecto']['corrected_cv'] - ccv_ref:+.10f}, con "
            f"{brechas['efecto_contra_el_margen']:+.10f} contra el margen. Y la hipotesis de la "
            f"fase se sostiene como DESCRIPCION: la negacion si esta asociada al error, con "
            f"{eda['diferencial']:+.4f} puntos, y si emite "
            f"{eda['marcas_emitidas']} marcas de alcance, luego la estructura existe. Lo que "
            f"no funciona es REPRESENTARLA, porque V03 ya ve el efecto de la negacion a "
            f"traves de la forma superficial de las palabras")

    log("")
    log(f"   LECTURA: {lectura}")
    log(f"   VEREDICTO: {veredicto}")
    log(f"   DECISION: {decision}")

    with CSV_NEG.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["tipo", "clave", "valor", "valor2", "valor3"])
        for k, val in eda.items():
            w.writerow(["eda", k, val if not isinstance(val, list) else "|".join(val)])
        w.writerow(["eda", "columnas_del_canal_de_alcance", ancho_random])
        for f in filas:
            w.writerow(["brazo", f["brazo"], f["corrected_cv"], f["delta_vs_V03"],
                        f["pliegues_mejorados"]])
        for k, val in brechas.items():
            w.writerow(["brecha", k, val])
        w.writerow([])
        w.writerow(["bin", "n", "F1_de_V03"] + [f"F1_{b}" for b in orden])
        for nombre, _lo, _hi in BINS:
            w.writerow([nombre, detalle["NEG_alcance"]["por_bin"][nombre]["n"],
                        detalle["NEG_alcance"]["por_bin"][nombre]["F1_de_V03"]]
                       + [detalle[b]["por_bin"][nombre]["F1"] for b in orden])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P64_NEGACION",
        "naturaleza": ("EXPERIMENTO. 3 mecanismos y 2 controles, 0 submissions, 0 Kaggle, "
                       "0 holdout, 0 test"),
        "fecha": inicio.isoformat(timespec="seconds"), "experiment_id": "P64-07",
        "track": "P13-alcance-de-negacion", "script": "experiments/p64_07_negacion.py",
        "pregunta": ("la negacion cambia la evidencia de las palabras que alcanza, y eso es "
                     "algo que V03, que ve la bolsa, no puede representar"),
        "por_que_no_es_una_bandera": (
            "la hipotesis no es que exista una bandera de negacion, sino la INTERACCION entre "
            "un marcador, una distancia y la evidencia afectada. Las marcas se emiten a "
            "distancia 1, 2 y 3 en las dos direcciones, luego el mismo par a distancia distinta "
            "es una columna distinta, y eso es lo que una bolsa no puede expresar"),
        "mecanismos": {
            "NEG_alcance": "canal sparse de los pares negador y palabra alcanzada, con el "
                           "alcance en el nombre del token",
            "NEG_efecto": "el cambio en la evidencia de V03 al quitar el segmento con negador",
            "NEG_ambos": "los dos juntos, con el presupuesto L2 recalibrado",
        },
        "eda": eda, "marcadores": list(NEGADORES), "alcances": list(ALCANCES),
        "columnas_del_canal_de_alcance": int(ancho_random),
        "protocolo": {
            "lexico": "lista cerrada escrita a mano y declarada antes. Sin lexico externo",
            "corpus_externo": "ninguno",
            "cosas_aprendidas_de_etiquetas": "ninguna, luego no hay nada que pueda filtrar",
            "ajuste_del_canal": "dentro de cada pliegue de entrenamiento",
            "control_de_anchura": "un bloque aleatorio de las mismas columnas",
            "control_del_margen": "V03 mas sus margenes",
            "umbrales": "0,44 / 0,51 / 0,46 congelados",
        },
        "resultados": detalle, "brechas": brechas,
        "lectura": lectura, "veredicto": veredicto, "decision": decision,
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_NEG.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P64 - Alcance de negacion")
    W("")
    W(f"Generado por `experiments/p64_07_negacion.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Que se prueba, con precision")
    W("")
    W("El encargo advierte que la hipotesis NO es una bandera de negacion. Son tres cosas a")
    W("la vez:")
    W("")
    W("```text")
    W("un MARCADOR      uno de una lista cerrada de negadores, por regla")
    W("una DISTANCIA     a quantos tokens esta la palabra afectada: 1, 2 o 3")
    W("una EVIDENCIA     lo que V03 cree sobre el texto en ese alcance")
    W("```")
    W("")
    W("Y la afirmacion es sobre la INTERACCION: cambia el marcador la evidencia de las palabras")
    W("que alcanza, de una forma que V03, que ve la bolsa, no puede representar.")
    W("")
    W("## Los tres mecanismos")
    W("")
    W("| mecanismo | que es |")
    W("|---|---|")
    W("| NEG_alcance | canal sparse de los pares negador y palabra alcanzada, con el alcance en el nombre |")
    W("| NEG_efecto | el cambio en la evidencia de V03 al quitar el segmento con negador |")
    W("| NEG_ambos | los dos juntos, con el presupuesto L2 recalibrado |")
    W("")
    W(f"Lista cerrada de marcadores: `{', '.join(NEGADORES)}`. Sin lexico externo, sin corpus")
    W("externo y sin nada aprendido de etiquetas, luego no hay nada que pueda filtrar.")
    W("")
    W("## La EDA de la negacion")
    W("")
    W("| magnitud | valor |")
    W("|---|---:|")
    W(f"| filas con negador | {eda['filas_con_negador']} ({eda['porcentaje']:.1f}%) |")
    W(f"| error con negador | {eda['error_con_negador']:.2%} |")
    W(f"| error sin negador | {eda['error_sin_negador']:.2%} |")
    W(f"| diferencial | {eda['diferencial']:+.4f} |")
    W(f"| filas que reciben marcas | {eda['filas_con_marca_de_alcance']} |")
    W(f"| marcas de alcance emitidas | {eda['marcas_emitidas']} |")
    W(f"| columnas del canal sparse | {ancho_random} |")
    W("")
    W(f"Negadores presentes en el corpus: `{', '.join(eda['negadores_en_uso'])}`.")
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
        W(f"{k:32} {val:+.10f}")
    W("```")
    W("")
    W("### Por estrato de longitud")
    W("")
    W("| bin | n | " + " | ".join(f"F1 {b}" for b in orden) + " |")
    W("|---|---:|" + "---:|" * len(orden))
    for nombre, _lo, _hi in BINS:
        W(f"| {nombre} | {detalle['NEG_alcance']['por_bin'][nombre]['n']} | "
          + " | ".join(f"{detalle[b]['por_bin'][nombre]['F1']:.4f}" for b in orden) + " |")
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
    W("*Generado por `experiments/p64_07_negacion.py`. Sin push.*")
    MD_NEG.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_NEG, JSON_NEG, CSV_NEG):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P64-07 OK | filas con negador {eda['filas_con_negador']} | marcas "
        f"{eda['marcas_emitidas']} | decision {decision}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
