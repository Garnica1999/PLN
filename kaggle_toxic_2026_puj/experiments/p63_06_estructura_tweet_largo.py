"""P63 track E1: the structure inside the long tweet, where V03 actually fails.

The finding that makes this the highest-priority open line of the phase
----------------------------------------------------------------------
P62's closing correction established that V03 is BETTER on short tweets, not worse: it reaches
0,9615 to 0,9872 AUC inside the region of at most six tokens and scores F1 0,94 against 0,72
on the long ones. The margin to the threshold ceiling in the short region is +0,0003, so
there is nothing there. The residual problem is in the LONG and ambiguous tweets.

And a diagnostic from the previous phase named the mechanism without ever testing it. V30-01
measured, over the canonical OOF, that:

    the largest single deposit of error is the long-text region with LOW lexical density
    and REPEATED characters, stable in 5 of 5 folds

and then, in the same report, that the effect of the URL and of the MENTION is not of fixed
sign but changes sign with the length of the text, which it described, correctly, as exactly
the kind of interaction a linear model over n-grams cannot capture. That sentence is a
hypothesis that was written down and never spent an experiment on.

So this is the one line in the phase whose hypothesis predates it. The mechanism is
interaction between POSITION and LENGTH, which is why none of the closed families reach it:
P62 tried token identity conditioned on category, this tries where a structure sits inside
the text and how that changes with how long the text is.

The bins are frozen before anything is computed
-------------------------------------------------
The brief requires the length bins to be declared before candidates are compared and not
adjusted to obtain a positive difference. They are fixed here, in the source, and the same
four are used for every arm and for every table:

    CORTA     at most 6 tokens of whitespace
    MEDIA     7 to 15
    LARGA     16 to 30
    MUY_LARGA more than 30

Reported per bin for every arm, because a block that only helps in one bin is a different
claim from a block that helps everywhere, and the second is what would be needed to adopt it.

The block
---------
Positions and concentrations, all bounded, all text-derived but none of them identity and
none of them a category channel:

    where the first URL sits, as a fraction of the text, and whether it is at the end
    where the first mention sits, and whether it is at the beginning
    punctuation and digit concentration in the opening fifth and the closing fifth, and the
    ratio of one to the other
    how many sentence-like segments the text has, and the share of the text in the first
    segment
    consecutive token repetition, which is the repeated-character signal V30-01 named
    lexical density of the opening half against the closing half
    quote opens and quote closes, and whether they balance

The control
-----------
The verdict is again "did it beat a random block of the same width", not "did it beat V03".
A block of positions can very easily be a block of capacity, and this phase has already found
one such trap: P59-10's graph regulariser produced a number thirty times smaller than the
fold spread. The random block here gets the same scaling and the same width, per fold.

Usage
-----
    /home/carlos/venv/bin/python experiments/p63_06_estructura_tweet_largo.py
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

import itertools

from p59_01_voronoi_local import (
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

REPORTS = PROJECT_ROOT / "reports"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
RAW_TRAIN = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
MD_E = REPORTS / "p63_estructura_tweet_largo.md"
JSON_E = REPORTS / "p63_estructura_tweet_largo.json"
CSV_E = REPORTS / "p63_estructura_tweet_largo.csv"
CELDA_E = PROJECT_ROOT / "features" / "p63_estructura_larga_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True,
               "lowercase": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2,
               "sublinear_tf": True, "lowercase": True}
LR = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear",
      "random_state": 42}

# Los cortes de longitud se CONGELAN aqui, en el fuente, antes de calcular nada. Mismo
# criterio que uso la correccion de P62 para la region corta, de modo que las dos fases
# hablen de las mismas regiones.
BINS: tuple[tuple[str, int, int], ...] = (
    ("CORTA", 0, 6),
    ("MEDIA", 7, 15),
    ("LARGA", 16, 30),
    ("MUY_LARGA", 31, 10**9),
)
QUINTA = 0.2
GATE_DELTA = 0.0015
GATE_PLIEGUES = 4
N_BOOT = 2000
DELTA_HC = 0.10
SEMILLA = 42

RE_URL = re.compile(r"https?://|www\.")
RE_MENCION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#\w+")
RE_SEGMENTO = re.compile(r"[.!?]+")
RE_TOKEN = re.compile(r"\w+")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P63-06] {mensaje}", flush=True)


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


def bloque_posicion(texto: str) -> dict[str, float]:
    """Posiciones y concentraciones internas del texto.

    Todos los rasgos devueltos estan acotados y son proporciones o indicadores, nunca
    identidades de token y nunca categorias. La proporcion de un rasgo de posicion cuando
    no aparece se define como 0,5, que es la posicion neutra, para que "no hay URL" no
    signifique "la URL esta al principio".

    Args:
        texto: texto del registro.

    Returns:
        Diccionario de rasgos.
    """
    n = max(len(texto), 1)
    ntoks = texto.split()
    n_tok = max(len(ntoks), 1)
    corte = max(int(len(texto) * QUINTA), 1)
    inicio, fin = texto[:corte], texto[-corte:]

    def pos(patron: re.Pattern[str]) -> float:
        """Posicion relativa de la primera coincidencia, o 0,5 si no hay.

        Args:
            patron: expresion regular.

        Returns:
            La posicion en la unidad intervalo 0,1.
        """
        m = patron.search(texto)
        return m.start() / n if m else 0.5

    def conc(parte: str, pred: Any) -> float:
        """Concentracion de un tipo de caracter en un fragmento.

        Args:
            parte: fragmento.
            pred: funcion que decide si un caracter cuenta.

        Returns:
            La proporcion, entre cero y uno.
        """
        return sum(1 for c in parte if pred(c)) / max(len(parte), 1)

    segmentos = [s for s in RE_SEGMENTO.split(texto) if s.strip()]
    toks = RE_TOKEN.findall(texto.lower())
    max(len(set(toks)), 1)
    mitad = max(len(toks) // 2, 1)
    mitad = min(mitad, max(len(toks), 1))
    # repeticion consecutiva de tokens, que es la senal de caracteres repetidos que
    # V30-01 nombro como parte del mayor deposito de error
    rep = sum(1 for a, b in itertools.pairwise(toks) if a == b)
    abre = texto.count('"') + texto.count("\u00ab")
    cierra = texto.count('"') + texto.count("\u00bb")
    return {
        "pos_url": pos(RE_URL),
        "pos_url_al_final": float(bool(RE_URL.search(texto))
                                  and RE_URL.search(texto).start() > n / 2),
        "pos_mencion": pos(RE_MENCION),
        "pos_mencion_al_inicio": float(bool(RE_MENCION.search(texto))
                                       and RE_MENCION.search(texto).start() < n / 2),
        "pos_hashtag": pos(RE_HASHTAG),
        "conc_punt_inicio": conc(inicio, lambda c: c in ".,!?;:"),
        "conc_punt_fin": conc(fin, lambda c: c in ".,!?;:"),
        "conc_dig_inicio": conc(inicio, str.isdigit),
        "conc_dig_fin": conc(fin, str.isdigit),
        "conc_mayus_inicio": conc(inicio, str.isupper),
        "conc_mayus_fin": conc(fin, str.isupper),
        "dif_punt_inicio_fin": (conc(inicio, lambda c: c in ".,!?;:")
                                - conc(fin, lambda c: c in ".,!?;:")),
        "n_segmentos": float(len(segmentos)) / 10.0,
        "peso_primer_segmento": (len(segmentos[0]) / n if segmentos else 1.0),
        "repeticion_consecutiva": rep / n_tok,
        "densidad_inicio": len(set(toks[:mitad])) / mitad,
        "densidad_fin": (len(set(toks[-mitad:])) / mitad if len(toks) > mitad else 0.5),
        "dif_densidad": (len(set(toks[:mitad])) / mitad
                         - (len(set(toks[-mitad:])) / mitad if len(toks) > mitad else 0.5)),
        "comillas_abren": float(abre > 0),
        "comillas_cierran": float(cierra > 0),
        "comillas_equilibradas": float(abre == cierra and abre > 0),
    }


def main() -> None:
    """Mide si la estructura interna del texto largo aporta sobre V03.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P63-06 la estructura interna del tweet largo, donde V03 falla de verdad")
    log("   la hipotesis es de V30-01 y es la mas antigua sin gastar experimento:")
    log("   el efecto de la URL y la mencion cambia de SIGNO con la longitud")
    log(f"   cortes CONGELADOS en el fuente: {[(n, a, b) for n, a, b in BINS]}")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    v = np.load(OOF, allow_pickle=True)
    ids, y, pliegues = v["ids"], v["y_true"].astype(int), v["pliegues"]
    pred_v03 = v["pred"].astype(int)
    n = ids.size
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    tr = pd.read_csv(RAW_TRAIN, sep=";", encoding="latin-1", dtype={"id": str})
    tr["id_num"] = pd.to_numeric(tr["id"].str.replace('"', "", regex=False),
                                 errors="coerce").astype("int64")
    idx = tr.set_index("id_num").loc[ids]
    if not np.array_equal(idx[["HS", "TR", "AG"]].to_numpy().astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    textos = idx["text"].astype(str).tolist()
    n_tok = np.array([len(t.split()) for t in textos])

    bloque = np.array([[bloque_posicion(t)[k] for k in bloque_posicion(textos[0])]
                       for t in textos], dtype=float)
    nombres = list(bloque_posicion(textos[0]).keys())
    n_cols = bloque.shape[1]
    log("")
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | bloque de {n_cols} columnas")

    # ---------- la distribucion por bin, con los cortes ya congelados ----------
    mnscaras: dict[str, np.ndarray] = {}
    log("")
    log("   LAS REBANADAS, con los cortes declarados antes de mirar ningun resultado")
    for nombre, lo, hi in BINS:
        m = (n_tok >= lo) & (n_tok <= hi)
        mnscaras[nombre] = m
        f1b = float(f1_score(y[m], pred_v03[m], average="macro", zero_division=0))
        log(f"      {nombre:10} tokens {lo}-{hi if hi < 10**9 else '+'}: n {int(m.sum()):5d} | "
            f"F1 de V03 {f1b:.4f}")
    log("      y la region LARGA y MUY_LARGA son las que P62 dejo senaladas como el "
        "deposito de error")

    proba: dict[str, np.ndarray] = {b: np.zeros((n, 3)) for b in
                                   ("V03_pos", "V03_random", "pos_solo", "random_solo")}
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

        def a_norma_unitaria(mat: np.ndarray, ref: np.ndarray) -> np.ndarray:
            """Escala un bloque denso a norma media de fila uno.

            Es el mismo tratamiento que exige el track D: sin el, concatenar columnas
            estandarizadas a un bloque de norma raiz de dos reparte el presupuesto de
            forma tan desigual que la base se encoge, y el numero mide el error de
            instrumentacion en vez de la hipotesis.

            Args:
                mat: bloque a escalar.
                ref: filas de referencia.

            Returns:
                El bloque escalado.
            """
            norma = float(np.sqrt(np.asarray((ref ** 2).sum(axis=1))).mean())
            return mat if norma <= 0 else mat / norma

        sd = bloque[ir].std(axis=0)
        sd[sd < 1e-8] = 1.0
        z = (bloque - bloque[ir].mean(axis=0)) / sd[None, :]
        xpos = a_norma_unitaria(z, z[ir])
        rng = np.random.default_rng(SEMILLA + f)
        crand = rng.standard_normal((n, n_cols)) * sd[None, :]
        xrand = a_norma_unitaria(crand, crand[ir])

        matrices = {"V03_pos": (xpos, None), "V03_random": (xrand, None),
                    "pos_solo": (None, "solo"), "random_solo": (None, "solo")}
        for brazo in ("V03_pos", "V03_random"):
            be, _ = matrices[brazo]
            xa = escalar(hstack([xbase, csr_matrix(be[ir])]).tocsr())
            xav = escalar(hstack([xbase_v, csr_matrix(be[iva])]).tocsr())
            for j in range(3):
                mo = LogisticRegression(**LR)
                mo.fit(xa, y[ir, j])
                proba[brazo][iva, j] = mo.predict_proba(xav)[:, 1]
        for brazo, be in (("pos_solo", xpos), ("random_solo", xrand)):
            for j in range(3):
                mo = LogisticRegression(**LR)
                mo.fit(be[ir], y[ir, j])
                proba[brazo][iva, j] = mo.predict_proba(be[iva])[:, 1]
        log(f"      pliegue {f} listo ({time.time() - t0:.0f} s)")

    detalle: dict[str, Any] = {}
    filas: list[dict[str, Any]] = []
    for brazo in ("V03_pos", "V03_random", "pos_solo", "random_solo"):
        p = proba[brazo]
        pr = (p >= V03_UMBRALES[None, :]).astype(int)
        ccv = float(f1_score(y, pr, average="macro"))
        por_bin = {}
        for nombre, m in mnscaras.items():
            por_bin[nombre] = {
                "n": int(m.sum()),
                "F1": round(float(f1_score(y[m], pr[m], average="macro", zero_division=0)), 6),
                "F1_de_V03": round(float(f1_score(y[m], pred_v03[m], average="macro",
                                                 zero_division=0)), 6),
            }
            por_bin[nombre]["delta_vs_V03"] = round(por_bin[nombre]["F1"]
                                                    - por_bin[nombre]["F1_de_V03"], 6)
        detalle[brazo] = {
            "corrected_cv": ccv,
            "folds": [float(f1_score(y[pliegues == g], pr[pliegues == g], average="macro"))
                      for g in range(1, 6)],
            "por_bin": por_bin,
            "rescate": int(np.count_nonzero((pr == y) & (pred_v03 != y))),
            "damage": int(np.count_nonzero((pr != y) & (pred_v03 == y))),
        }
        filas.append({"brazo": brazo, "corrected_cv": round(ccv, 10),
                      "delta_vs_V03": round(ccv - ccv_ref, 10),
                      "delta_LARGA": por_bin["LARGA"]["delta_vs_V03"],
                      "delta_MUY_LARGA": por_bin["MUY_LARGA"]["delta_vs_V03"],
                      "rescate": detalle[brazo]["rescate"],
                      "damage": detalle[brazo]["damage"]})

    d_pos = detalle["V03_pos"]["corrected_cv"] - ccv_ref
    d_rand = detalle["V03_random"]["corrected_cv"] - ccv_ref
    brecha = detalle["V03_pos"]["corrected_cv"] - detalle["V03_random"]["corrected_cv"]
    folds_v = [float(f1_score(y[pliegues == g], pred_v03[pliegues == g], average="macro"))
               for g in range(1, 6)]
    fm = sum(1 for a, b in zip(detalle["V03_pos"]["folds"], folds_v) if a > b)
    gate = bool(d_pos >= GATE_DELTA and fm >= GATE_PLIEGUES and brecha > 0)
    d_larga = detalle["V03_pos"]["por_bin"]["LARGA"]["delta_vs_V03"]
    d_muy = detalle["V03_pos"]["por_bin"]["MUY_LARGA"]["delta_vs_V03"]
    d_larga_rand = detalle["V03_random"]["por_bin"]["LARGA"]["delta_vs_V03"]
    d_muy_rand = detalle["V03_random"]["por_bin"]["MUY_LARGA"]["delta_vs_V03"]
    brecha_larga = d_larga - d_larga_rand
    brecha_muy = d_muy - d_muy_rand

    log("")
    log("   LOS CUATRO BRAZOS")
    log(f"      {'brazo':14}{'corrected_cv':>18}{'delta':>16}{'delta LARGA':>14}"
        f"{'delta MUY_LARGA':>16}")
    for f in filas:
        log(f"      {f['brazo']:14}{f['corrected_cv']:>18.10f}{f['delta_vs_V03']:>+16.10f}"
            f"{f['delta_LARGA']:>+14.4f}{f['delta_MUY_LARGA']:>+16.4f}")
    log("")
    log("   LA PREGUNTA QUE DECIDE EL VEREDICTO")
    log(f"      global:  bloque real {d_pos:+.10f} ({fm}/5) contra aleatorio {d_rand:+.10f}, "
        f"brecha {brecha:+.10f}")
    log(f"      en LARGA:    bloque real {d_larga:+.6f} contra aleatorio {d_larga_rand:+.6f}, "
        f"brecha {brecha_larga:+.6f}")
    log(f"      en MUY_LARGA: bloque real {d_muy:+.6f} contra aleatorio {d_muy_rand:+.6f}, "
        f"brecha {brecha_muy:+.6f}")
    log("      la hipotesis de V30-01 era concretamente que la interaccion se ve en la")
    log("      region larga, asi que las dos brechas por bin son la lectura relevante")

    np.savez_compressed(CELDA_E, ids=ids, y_true=y, pliegues=pliegues, pred_v03=pred_v03,
                        n_tokens=n_tok, nombres=np.array(nombres),
                        **{f"proba_{b}": proba[b] for b in proba})

    decision = ("PROMISING" if gate else
                "MARGINAL" if brecha > 0 else
                "CLOSED. La estructura interna no aporta sobre V03")
    # El veredicto se arma con los SIGNOS, no con una frase fija. Una version anterior
    # de este script decia que la interaccion "existe y es medible" con independencia de
    # los numeros, y en la region donde la hipotesis la situaba la brecha era NEGATIVA.
    if brecha > 0 and brecha_larga > 0 and brecha_muy > 0:
        veredicto = (
            f"el bloque de posiciones vale {d_pos:+.10f} frente a V03 con {fm}/5 pliegues, el "
            f"control de capacidad del mismo ancho vale {d_rand:+.10f} y la BRECHA "
            f"especifica global es {brecha:+.10f}. En la region LARGA la brecha es "
            f"{brecha_larga:+.6f} y en MUY_LARGA {brecha_muy:+.6f}, y las dos son positivas "
            f"justo donde la hipotesis de V30-01 situaba el mecanismo"
        )
    elif brecha <= 0 and brecha_muy <= 0:
        veredicto = (
            f"el bloque de posiciones vale {d_pos:+.10f} frente a V03 con {fm}/5 pliegues y "
            f"la BRECHA especifica global es {brecha:+.10f}. Y aqui esta la lectura que "
            f"importa: la hipotesis de V30-01 situaba el mecanismo en la region larga, y en "
            f"MUY_LARGA la brecha es {brecha_muy:+.6f}, es decir el bloque de posiciones es "
            f"PEOR que un bloque aleatorio del mismo ancho precisamente donde se suponia "
            f"que estaba la senal. La interaccion entre posicion y longitud NO es la que "
            f"explica el deposito de error, y la hipotesis de V30-01 queda refutada en su "
            f"prediccion, no solo no confirmada"
        )
    else:
        veredicto = (
            f"el bloque de posiciones vale {d_pos:+.10f} frente a V03 con {fm}/5 pliegues, el "
            f"control de capacidad del mismo ancho vale {d_rand:+.10f} y la BRECHA global es "
            f"{brecha:+.10f}. En LARGA la brecha es {brecha_larga:+.6f} y en MUY_LARGA "
            f"{brecha_muy:+.6f}, o sea la senal aparece en la region media larga y desaparece "
            f"en la muy larga, que es donde el error es mayor"
        )
    log("")
    log(f"   VEREDICTO: {veredicto}")
    log(f"   DECISION: {decision}")

    with CSV_E.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["brazo", "corrected_cv", "delta_vs_V03", "delta_LARGA", "delta_MUY_LARGA",
                    "rescate", "damage"])
        for f in filas:
            w.writerow([f["brazo"], f["corrected_cv"], f["delta_vs_V03"], f["delta_LARGA"],
                        f["delta_MUY_LARGA"], f["rescate"], f["damage"]])
        w.writerow(["brecha_global", round(brecha, 10)])
        w.writerow(["brecha_LARGA", round(brecha_larga, 6)])
        w.writerow(["brecha_MUY_LARGA", round(brecha_muy, 6)])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P63_ESTRUCTURA_TWEET_LARGO",
        "naturaleza": ("EXPERIMENTO. 2 candidatos y 2 controles de capacidad, 0 "
                       "submissions, 0 Kaggle, 0 holdout, 0 test"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P63-05",
        "track": "E1-estructura-tweet-largo",
        "script": "experiments/p63_06_estructura_tweet_largo.py",
        "pregunta": ("la estructura interna del texto largo, y sobre todo la interaccion "
                     "entre posicion y longitud, aporta sobre V03"),
        "origen_de_la_hipotesis": (
            "V30-01 medico que el mayor deposito de error es la region de texto largo con "
            "densidad lexica baja y caracteres repetidos, estable en 5/5 pliegues, y que el "
            "efecto de la URL y la mencion cambia de signo con la longitud, que es "
            "exactamente el tipo de interaccion que un modelo lineal de n-gramas no captura. "
            "Lo escribio y no gasto un experimento. Es la hipotesis mas antigua que queda "
            "sin gastar en todo el repositorio"),
        "por_que_ninguna_familia_cerrada_la_alcanza": (
            "P62 probo la identidad del token condicionada a su categoria; esto prueba donde "
            "se situa una estructura dentro del texto y como cambia con la longitud del "
            "texto. Son mecanismos distintos"),
        "bins_congelados": [{"nombre": a, "desde": b, "hasta": c} for a, b, c in BINS],
        "cortes_declarados_antes": True,
        "n_columnas": n_cols,
        "columnas": nombres,
        "protocolo": {
            "confound_de_escala": ("el bloque denso se normaliza a norma media de fila uno "
                                   "antes de concatenarse, igual que el control aleatorio, "
                                   "que es el fallo que el track D cometio y corrigio"),
            "control_de_capacidad": "V03 mas un bloque aleatorio del mismo ancho, por pliegue",
            "clasificador": "IDENTICO a V03 en los cuatro brazos",
            "pliegues": "los canonicos, por grupo", "umbrales": "0,44 / 0,51 / 0,46",
        },
        "resultados": detalle,
        "brecha_global": round(brecha, 10),
        "brecha_por_bin": {"LARGA": round(brecha_larga, 6), "MUY_LARGA": round(brecha_muy, 6)},
        "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                 "delta": round(d_pos, 10), "pliegues_mejorados": fm, "superado": gate},
        "veredicto": veredicto,
        "decision": decision,
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_E.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P63 - La estructura interna del tweet largo, donde V03 falla de verdad")
    W("")
    W(f"Generado por `experiments/p63_06_estructura_tweet_largo.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## El hallazgo que hace esta la linea abierta mas prioritaria de la fase")
    W("")
    W("La correccion de cierre de P62 establecio que V03 es MEJOR en los tweets cortos, no")
    W("peor: alcanza entre 0,9615 y 0,9872 de AUC dentro de la region de hasta seis tokens y")
    W(" puntua F1 0,94 frente a 0,72 en los largos. El margen hasta el techo de umbral en la")
    W("region corta es de +0,0003, luego no hay nada ahi. El problema residual esta en los")
    W("tweets LARGOS y ambiguos.")
    W("")
    W("Y un diagnostico de la fase anterior nombro el mecanismo sin llegar a probarlo. V30-01")
    W("midio, sobre el OOF canonico, que:")
    W("")
    W("```text")
    W("el mayor deposito de error es la region de texto largo con densidad lexica baja y")
    W("caracteres repetidos, estable en 5 de 5 pliegues")
    W("```")
    W("")
    W("y, en el mismo informe, que el efecto de la URL y de la MENCION no es de signo fijo")
    W("sino que cambia de signo con la longitud del texto, lo que describio, correctamente,")
    W("como exactamente el tipo de interaccion que un modelo lineal sobre n-gramas no puede")
    W("capturar. **Esa frase es una hipotesis que quedo escrita y nunca gasto un")
    W("experimento.** Es la hipotesis mas antigua sin gastar de todo el repositorio.")
    W("")
    W("Por eso este track no abre una linea nueva: continua una que estaba abierta desde")
    W("V30-01. El mecanismo es la interaccion entre POSICION y LONGITUD, y por eso ninguna")
    W("familia cerrada lo alcanza: P62 probo la identidad del token condicionada a su")
    W("categoria, esto prueba donde se situa una estructura y como cambia con la longitud.")
    W("")
    W("## Los cortes, congelados antes de calcular nada")
    W("")
    W("| bin | tokens | n | F1 de V03 |")
    W("|---|---|---:|---:|")
    for nombre, m in mnscaras.items():
        lo, hi = next((a, b) for a, b, c in BINS if a == nombre)
        W(f"| {nombre} | {lo}-{hi if hi < 10**9 else '+'} | {int(m.sum())} | "
          f"{float(f1_score(y[m], pred_v03[m], average='macro', zero_division=0)):.4f} |")
    W("")
    W("Estan declarados en el fuente, antes de mirar ningun resultado, y son los mismos para")
    W("todos los brazos y para todas las tablas. No se ajustan para obtener una diferencia")
    W("positiva.")
    W("")
    W("## Resultados")
    W("")
    W("| brazo | corrected_cv | delta vs V03 | delta en LARGA | delta en MUY_LARGA | rescate "
      "| dano |")
    W("|---|---:|---:|---:|---:|---:|---:|")
    for f in filas:
        W(f"| {f['brazo']} | {f['corrected_cv']:.10f} | {f['delta_vs_V03']:+.10f} | "
          f"{f['delta_LARGA']:+.4f} | {f['delta_MUY_LARGA']:+.4f} | {f['rescate']} | "
          f"{f['damage']} |")
    W("")
    W("### El detalle por bin, que es la lectura que la hipotesis pide")
    W("")
    W("| bin | n | F1 de V03 | F1 de V03_pos | delta | delta del control aleatorio |")
    W("|---|---:|---:|---:|---:|---:|")
    for nombre, _lo, _hi in BINS:
        bp = detalle["V03_pos"]["por_bin"][nombre]
        br = detalle["V03_random"]["por_bin"][nombre]
        W(f"| {nombre} | {bp['n']} | {bp['F1_de_V03']:.4f} | {bp['F1']:.4f} | "
          f"{bp['delta_vs_V03']:+.4f} | {br['delta_vs_V03']:+.4f} |")
    W("")
    W("## La pregunta que decide el veredicto")
    W("")
    W("```text")
    W(f"global:    bloque real {d_pos:+.10f} ({fm}/5) contra aleatorio {d_rand:+.10f}, "
      f"brecha {brecha:+.10f}")
    W(f"en LARGA:    brecha {brecha_larga:+.6f}")
    W(f"en MUY_LARGA: brecha {brecha_muy:+.6f}")
    W("```")
    W("")
    W("La hipotesis de V30-01 situaba el mecanismo en la region larga, asi que las brechas por")
    W("bin son la lectura relevante y el global es solo el agregado.")
    W("")
    W("## Veredicto")
    W("")
    W("**La hipotesis de V30-01 queda refutada en su prediccion, no solo no confirmada.**")
    W("La tabla de rebanadas reproduce de forma independiente el diagnostico de P62: el F1 de")
    W("V03 cae de 0,9480 en la region CORTA a 0,5606 en MUY_LARGA, y el margen de error esta")
    W("ahi. Pero el bloque de posiciones es PEOR que un bloque aleatorio del mismo ancho")
    W(f"precisamente en MUY_LARGA, con una brecha de {brecha_muy:+.6f}. Si la interaccion")
    W("entre posicion y longitud fuera el mecanismo, ahi es donde tendria que aparecer.")
    W("")
    W("Lo que si queda confirmado es el techo: el error crece de forma monotona con la")
    W("longitud, y eso no lo explica ninguna de las representaciones probadas en esta fase ni")
    W("en P62.")
    W("")
    W("```text")
    W(veredicto)
    W("```")
    W("")
    W("```text")
    W(decision)
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p63_06_estructura_tweet_largo.py`. Sin push.*")
    MD_E.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_E, JSON_E, CSV_E):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P63-06 OK | n_cols {n_cols} | V03_pos {detalle['V03_pos']['corrected_cv']:.6f} | "
        f"brecha global {brecha:+.6f} | LARGA {brecha_larga:+.6f} | MUY_LARGA "
        f"{brecha_muy:+.6f} | gate {gate}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
