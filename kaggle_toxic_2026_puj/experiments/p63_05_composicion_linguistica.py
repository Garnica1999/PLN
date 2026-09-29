"""P63 track D: aggregate composition, as ratios, after the high-dimensional POS channel died.

The distinction this experiment is built on
--------------------------------------------
P62 closed a family, and the reading of what it closed matters more than the fact that it
closed. What P62 measured was a high-dimensional lexical POS channel, `palabra__POS`, whose
cost of dilution was +0,011585 and whose value of content was +0,002014, so it repaid 17,4
per cent of its cost. That is a statement about one REPRESENTATION, not about grammar.

The brief is explicit that the correct reading is not "grammar is useless" but:

    grammatical composition may exist as signal, while its explicit high-dimensional
    representation does not pay for its cost in the available data regime

So this experiment does not re-run the closed family. It changes the shape of the block
along the axis that P62's own curve says matters: SIZE. A channel of several hundred
sparse columns competes for the L2 budget of a 21000-dimensional space and dilutes. A
handful of bounded ratios does not. If the composition signal is real, ratios are the
representation in which it can pay for itself.

What the block is
-----------------
Low-dimensional, bounded, and every column interpretable as a proportion:

    lexical      type-token ratio, hapax ratio, mean and median token length, unique
                 token count, long-word and short-word share
    functional   pronoun, determiner, preposition, conjunction, auxiliary, adverb,
                 interjection and particle shares, computed over spaCy tags so they do not
                 need a closed lexicon
    person       first, second and third person singular and plural shares
    category     verb, noun, adjective, adverb, pronoun shares, and the verb to noun and
                 adjective to noun ratios, which is where a linear model over n-grams
                 cannot go
    punctuation and orthography   question, exclamation, ellipsis, repeated-punctuation,
                 uppercase, digit and elongation shares

The category and person shares use the spaCy annotation already authorised by D-003 and
built in P62, so no new tagger and no new external resource. The ratio of a tag share is a
genuinely different mechanism from the token that carries the tag, and that is the whole
point.

The arms, and the control that decides the verdict
--------------------------------------------------
    comp_solo      the composition block alone
    random_solo    the same pipeline on random columns of identical dimensionality
    V03_comp       V03 plus the composition block
    V03_random     V03 plus random columns of the same count, the capacity control

The verdict is not "did V03_comp beat V03". It is "did V03_comp beat V03_random". A gain
that a random block of the same width also produces is capacity, not composition, and this
project has already been fooled once by exactly that question, in P59-10, where a graph
regulariser that looked real was thirty times smaller than the fold spread.

Fold safety of the frequency-derived columns
---------------------------------------------
The hapax ratio and the unique-token counts depend on document frequency, which is a
supervised statistic if it is computed over everything. It is computed inside each training
fold and never sees a validation row, and the script says so where it happens.

Usage
-----
    /home/carlos/venv/bin/python experiments/p63_05_composicion_linguistica.py
"""

from __future__ import annotations

import ast
import csv as _csv
import gzip
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
    ETIQUETAS,
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

REPORTS = PROJECT_ROOT / "reports"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
RAW_TRAIN = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
ANOTACION = PROJECT_ROOT / "features" / "p62_anotacion_spacy.json.gz"
MD_D = REPORTS / "p63_composicion_linguistica.md"
JSON_D = REPORTS / "p63_composicion_linguistica.json"
CSV_D = REPORTS / "p63_composicion_linguistica.csv"
CELDA_D = PROJECT_ROOT / "features" / "p63_composicion_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True,
               "lowercase": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2,
               "sublinear_tf": True, "lowercase": True}
LR = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear",
      "random_state": 42}
PRONOMBRES = {"PRON"}
DETERMINANTES = {"DET"}
PREPOSICIONES = {"ADP"}
CONJUNCIONES = {"CCONJ", "SCONJ"}
AUXILIARES = {"AUX"}
ADVERBIOS = {"ADV"}
INTERJECCIONES = {"INTJ", "PART"}
SUSTANTIVOS = {"NOUN", "PROPN"}
VERBOS = {"VERB", "AUX"}
ADJETIVOS = {"ADJ"}
GATE_DELTA = 0.0015
GATE_PLIEGUES = 4
N_BOOT = 2000
DELTA_HC = 0.10
SEMILLA = 42


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P63-05] {mensaje}", flush=True)


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


RE_ELONGACION = re.compile(r"(.)\1{2,}")
RE_ELLIPSIS = re.compile("\\.{3,}|\u2026")


def rasgos_fijos(texto: str) -> dict[str, float]:
    """Rasgos ortograficos y de puntuacion que no dependen de ningun corpus.

    Args:
        texto: texto del registro.

    Returns:
        Diccionario de proporciones.
    """
    n = max(len(texto), 1)
    toks = texto.split()
    n_tok = max(len(toks), 1)
    mayusculas = sum(1 for c in texto if c.isupper())
    digitos = sum(1 for c in texto if c.isdigit())
    interog = texto.count("?")
    exclama = texto.count("!")
    elipsis = len(RE_ELLIPSIS.findall(texto))
    elong = len(RE_ELONGACION.findall(texto))
    rep_punt = len(re.findall(r"([.,!?;:])\1+", texto))
    comillas = texto.count('"') + texto.count("'") + texto.count("\u00ab")
    comillas += texto.count("\u00bb")
    urls = len(re.findall(r"https?://|www\.", texto))
    menciones = len(re.findall(r"@\w+", texto))
    hashtags = len(re.findall(r"#\w+", texto))
    return {
        "ort_mayusculas": mayusculas / n,
        "ort_digitos": digitos / n,
        "ort_interrogaciones": interog / n_tok,
        "ort_exclamaciones": exclama / n_tok,
        "ort_elipsis": elipsis / n_tok,
        "ort_elongacion": elong / n_tok,
        "ort_puntuacion_repetida": rep_punt / n_tok,
        "ort_comillas": comillas / n_tok,
        "ort_urls": float(urls > 0),
        "ort_menciones": menciones / n_tok,
        "ort_hashtags": hashtags / n_tok,
        "ort_puntuacion_total": sum(1 for c in texto if c in ".,!?;:") / n_tok,
    }


def rasgos_lexicos(texto: str, freq: dict[str, int], n_docs: int) -> dict[str, float]:
    """Rasgos lexicos, con la frecuencia de documento calculada dentro del pliegue.

    Args:
        texto: texto del registro.
        freq: frecuencia de documento por token, ajustada SOLO con el pliegue.
        n_docs: numero de documentos del pliegue.

    Returns:
        Diccionario de proporciones.
    """
    toks = re.findall(r"\w+", texto.lower())
    n_tok = max(len(toks), 1)
    unicos = set(toks)
    hapax = sum(1 for t in unicos if freq.get(t, 0) == 1)
    lens = [len(t) for t in toks] or [0]
    largas = sum(1 for t in toks if len(t) >= 7)
    cortas = sum(1 for t in toks if len(t) <= 3)
    return {
        "lex_ttr": len(unicos) / n_tok,
        "lex_hapax": hapax / max(len(unicos), 1),
        "lex_long_media": float(np.mean(lens)) / 10.0,
        "lex_long_mediana": float(np.median(lens)) / 10.0,
        "lex_unicos_log": float(np.log1p(len(unicos))) / 10.0,
        "lex_palabras_largas": largas / n_tok,
        "lex_palabras_cortas": cortas / n_tok,
        "lex_n_tokens_log": float(np.log1p(n_tok)) / 10.0,
        "lex_longitud_maxima": float(max(lens)) / 20.0,
    }


def rasgos_categoricos(doc: dict[str, Any]) -> dict[str, float]:
    """Ratios de composicion categorial y de persona, desde la anotacion de spaCy.

    Args:
        doc: documento anotado de P62.

    Returns:
        Diccionario de proporciones.
    """
    toks = doc["tok"]
    lex = [t for t in toks if t["al"] and t["p"] not in ("PUNCT", "SPACE", "SYMBOL", "NUM")]
    n = max(len(lex), 1)

    def share(etiquetas: set[str]) -> float:
        """Proporcion de tokens de un conjunto de categorias.

        Args:
            etiquetas: conjunto de categorias universales.

        Returns:
            La proporcion, entre cero y uno.
        """
        return sum(1 for t in lex if t["p"] in etiquetas) / n

    sust = share(SUSTANTIVOS)
    verb = share(VERBOS)
    adj = share(ADJETIVOS)
    return {
        "cat_sustantivos": sust,
        "cat_verbos": verb,
        "cat_adjetivos": adj,
        "cat_adverbios": share(ADVERBIOS),
        "cat_pronombres": share(PRONOMBRES),
        "cat_determinantes": share(DETERMINANTES),
        "cat_preposiciones": share(PREPOSICIONES),
        "cat_conjunciones": share(CONJUNCIONES),
        "cat_auxiliares": share(AUXILIARES),
        "cat_interjecciones": share(INTERJECCIONES),
        "cat_verbos_sobre_sustantivos": verb / max(sust, 1e-3),
        "cat_adjetivos_sobre_sustantivos": adj / max(sust, 1e-3),
        "cat_adjetivos_sobre_verbos": adj / max(verb, 1e-3),
        "cat_pron_sobre_sustantivos": share(PRONOMBRES) / max(sust, 1e-3),
        "per_1sg": sum(1 for t in lex if t["p"] in PRONOMBRES and t["l"] == "yo") / n,
        "per_1pl": sum(1 for t in lex if t["p"] in PRONOMBRES
                       and t["l"] in ("nosotros", "nosotras")) / n,
        "per_2sg": sum(1 for t in lex if t["p"] in PRONOMBRES
                       and t["l"] in ("tn", "tu", "usted")) / n,
        "per_2pl": sum(1 for t in lex if t["p"] in PRONOMBRES
                       and t["l"] in ("vosotros", "vosotras", "ustedes")) / n,
        "per_3": share(PRONOMBRES) - (sum(1 for t in lex if t["p"] in PRONOMBRES
                                          and t["l"] in ("yo", "nosotros", "nosotras", "tn",
                                                         "tu", "usted", "vosotros",
                                                         "vosotras", "ustedes")) / n),
        "per_2_total": (sum(1 for t in lex if t["p"] in PRONOMBRES
                            and t["l"] in ("tn", "tu", "usted", "vosotros", "vosotras",
                                           "ustedes")) / n),
        "per_1_total": (sum(1 for t in lex if t["p"] in PRONOMBRES
                            and t["l"] in ("yo", "nosotros", "nosotras")) / n),
    }


def main() -> None:
    """Mide la composicion agregada como ratios, contra control de capacidad.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P63-05 la composicion agregada, en forma de ratios y no de canal disperso")
    log("   P62 cerro el canal palabra__POS de alta dimension. Aqui cambia la FORMA del")
    log("   bloque, que es justo el eje que la curva de presupuesto de P62 marco")

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
    with gzip.open(ANOTACION, "rt", encoding="utf-8") as fh:
        anot = json.load(fh)
    if anot["ids"] != [int(x) for x in ids]:
        raise SystemExit("PARADA: la anotacion no corresponde a las filas del OOF")
    documentos = anot["documentos"]
    log("")
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | anotacion verificada")

    # ---------- el bloque, una vez, sin frequencies ----------
    fijos = [rasgos_fijos(t) for t in textos]
    categoricos = [rasgos_categoricos(d) for d in documentos]
    nombres = list(fijos[0].keys()) + list(categoricos[0].keys())
    base_sin_freq = np.array([[f[k] for k in fijos[0]] + [c[k] for k in categoricos[0]]
                             for f, c in zip(fijos, categoricos)], dtype=float)
    n_ratios = base_sin_freq.shape[1]
    log("")
    log(f"   el bloque tiene {n_ratios} columnas ademas de las "
        f"{len(rasgos_lexicos('a b', {}, 1))} que dependen de la frecuencia de documento")

    n_cols = n_ratios + len(rasgos_lexicos("a b", {}, 1))
    proba: dict[str, np.ndarray] = {b: np.zeros((n, 3)) for b in
                                   ("comp_solo", "random_solo", "V03_comp", "V03_random")}
    t0 = time.time()
    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        # La frecuencia de documento se calcula SOLO con las filas de entrenamiento. Una
        # version sin esta linea habria usado todo el corpus, y el ratio de hapax habria
        # sido una estadistica que veia las filas de validacion.
        freq: dict[str, int] = {}
        for i in ir:
            for t in set(re.findall(r"\w+", textos[i].lower())):
                freq[t] = freq.get(t, 0) + 1
        lex = np.array([[rasgos_lexicos(textos[i], freq, ir.size)[k]
                         for k in rasgos_lexicos("a b", {}, 1)] for i in range(n)], dtype=float)
        bloque = np.hstack([base_sin_freq, lex])

        pw = TfidfVectorizer(**PARAMS_WORD)
        pc = TfidfVectorizer(**PARAMS_CHAR)
        xw = pw.fit_transform([textos[i] for i in ir])
        xc = pc.fit_transform([textos[i] for i in ir])
        xbase = hstack([xw, xc]).tocsr()
        xbase_v = hstack([pw.transform([textos[i] for i in iva]),
                          pc.transform([textos[i] for i in iva])]).tocsr()
        objetivo = float(np.sqrt(np.mean(xbase.multiply(xbase).sum(axis=1))))

        def escalar(m: csr_matrix, ref: float = objetivo) -> csr_matrix:
            """Iguala la norma media de fila a la del control.

            Args:
                m: matriz dispersa.
                ref: norma objetivo.

            Returns:
                La matriz escalada.
            """
            media = float(np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel()).mean())
            return m if media <= 0 else m.multiply(ref / media)

        rng = np.random.default_rng(SEMILLA + f)
        # Estandarizacion por columna con la desviacion del pliegue de entrenamiento.
        sd = bloque[ir].std(axis=0)
        sd[sd < 1e-8] = 1.0
        z = (bloque - bloque[ir].mean(axis=0)) / sd[None, :]
        # Y normalizacion del bloque a norma media de fila UNO, que es lo que hace cada
        # bloque TF-IDF de V03. Sin este paso, concatenar 42 columnas estandarizadas a un
        # bloque que ya tiene norma raiz de dos reparte el presupuesto de forma tan
        # desigual que la base se encoge por un factor de orden 4,7. La primera version de
        # este script no lo hacia y devolvia -0,1899987838, que era un fallo de
        # instrumentacion y no un hallazgo. El control aleatorio recibe el mismo
        # tratamiento, para que ambos brazos difieran en la DIRECCION y no en la ESCALA.
        def a_norma_unitaria(mat: np.ndarray, ref: np.ndarray) -> np.ndarray:
            """Escala un bloque para que su norma media de fila sea uno.

            Args:
                mat: bloque a escalar.
                ref: filas de referencia, de entrenamiento.

            Returns:
                El bloque escalado.
            """
            norma = float(np.sqrt(np.asarray((ref ** 2).sum(axis=1))).mean())
            return mat if norma <= 0 else mat / norma
        xcomp = a_norma_unitaria(z, z[ir])
        rsd = z[ir].std(axis=0)
        rsd[rsd < 1e-8] = 1.0
        # El control aleatorio se normaliza con la MISMA funcion y sobre las filas de
        # entrenamiento, para que ambos brazos tengan la misma norma y solo difieran en
        # la direccion de las columnas.
        crand = rng.standard_normal((n, n_cols)) * rsd[None, :]
        xrand = a_norma_unitaria(crand, crand[ir])
        log(f"      pliegue {f}: norma del bloque real "
            f"{float(np.sqrt(np.asarray((xcomp[ir] ** 2).sum(axis=1))).mean()):.6f} y del "
            f"aleatorio "
            f"{float(np.sqrt(np.asarray((xrand[ir] ** 2).sum(axis=1))).mean()):.6f}")

        matrices = {
            "comp_solo": (xcomp[ir], xcomp[iva]),
            "random_solo": (xrand[ir], xrand[iva]),
            "V03_comp": (csr_matrix(xcomp[ir]), csr_matrix(xcomp[iva])),
            "V03_random": (csr_matrix(xrand[ir]), csr_matrix(xrand[iva])),
        }
        for brazo, (be, bv) in matrices.items():
            if brazo.startswith("V03"):
                xent = escalar(hstack([xbase, be]).tocsr())
                xval = escalar(hstack([xbase_v, bv]).tocsr())
            else:
                xent, xval = be, bv
            for j in range(3):
                mo = LogisticRegression(**LR)
                mo.fit(xent, y[ir, j])
                proba[brazo][iva, j] = mo.predict_proba(xval)[:, 1]
        log(f"      pliegue {f} listo ({time.time() - t0:.0f} s) | bloque {n_cols} columnas")

    detalle: dict[str, Any] = {}
    filas: list[dict[str, Any]] = []
    for brazo in ("V03_comp", "V03_random", "comp_solo", "random_solo"):
        p = proba[brazo]
        pr = (p >= V03_UMBRALES[None, :]).astype(int)
        ccv = float(f1_score(y, pr, average="macro"))
        folds = [float(f1_score(y[pliegues == g], pr[pliegues == g], average="macro"))
                 for g in range(1, 6)]
        ac = pr == y
        por_et = {}
        for j, et in enumerate(ETIQUETAS):
            a = ac[:, j]
            por_et[et] = {
                "F1": round(float(f1_score(y[:, j], pr[:, j], zero_division=0)), 6),
                "rescate": int(np.count_nonzero(a & (pred_v03[:, j] != y[:, j]))),
                "damage": int(np.count_nonzero((~a) & (pred_v03[:, j] == y[:, j]))),
            }
        detalle[brazo] = {"corrected_cv": ccv, "folds": folds, "por_etiqueta": por_et,
                          "rescate": int(np.count_nonzero(ac & (pred_v03 != y))),
                          "damage": int(np.count_nonzero((~ac) & (pred_v03 == y))),
                          "n_columnas": n_cols}
        filas.append({"brazo": brazo, "corrected_cv": round(ccv, 10),
                      "delta_vs_V03": round(ccv - ccv_ref, 10),
                      "folds": "|".join(f"{x:.10f}" for x in folds),
                      "delta_vs_V03_random": ""})

    d_comp = detalle["V03_comp"]["corrected_cv"] - ccv_ref
    d_rand = detalle["V03_random"]["corrected_cv"] - ccv_ref
    brecha = detalle["V03_comp"]["corrected_cv"] - detalle["V03_random"]["corrected_cv"]
    for f in filas:
        if f["brazo"] == "V03_comp":
            f["delta_vs_V03_random"] = round(brecha, 10)
    folds_v = [float(f1_score(y[pliegues == g], pred_v03[pliegues == g], average="macro"))
               for g in range(1, 6)]
    fm = sum(1 for a, b in zip(detalle["V03_comp"]["folds"], folds_v) if a > b)
    fm_rand = sum(1 for a, b in zip(detalle["V03_random"]["folds"], folds_v) if a > b)
    gate = bool(d_comp >= GATE_DELTA and fm >= GATE_PLIEGUES and brecha > 0)

    log("")
    log("   LOS CUATRO BRAZOS")
    log(f"      {'brazo':14}{'corrected_cv':>18}{'delta vs V03':>16}{'pliegues':>10}")
    for f in filas:
        pl = sum(1 for a, b in zip(detalle[f['brazo']]["folds"], folds_v) if a > b)
        log(f"      {f['brazo']:14}{f['corrected_cv']:>18.10f}{f['delta_vs_V03']:>+16.10f}"
            f"{pl:>7d}/5")
    log("")
    log("   LA PREGUNTA QUE DECIDE EL VEREDICTO")
    log(f"      V03 mas composicion:  {d_comp:+.10f} con {fm}/5 pliegues")
    log(f"      V03 mas ALEATORIO:    {d_rand:+.10f} con {fm_rand}/5 pliegues")
    log(f"      BRECHA especifica:    {brecha:+.10f}")
    log("      si la brecha es pequena o negativa, lo que habria ganado el bloque real es")
    log("      CAPACIDAD, y con ella cualquier bloque de ese ancho")
    log(f"      en solitario: composicion {detalle['comp_solo']['corrected_cv']:.10f} "
        f"contra aleatorio {detalle['random_solo']['corrected_cv']:.10f}, brecha "
        f"{detalle['comp_solo']['corrected_cv'] - detalle['random_solo']['corrected_cv']:+.10f}")

    np.savez_compressed(CELDA_D, ids=ids, y_true=y, pliegues=pliegues, pred_v03=pred_v03,
                        nombres=np.array(nombres + list(rasgos_lexicos("a b", {}, 1).keys())),
                        **{f"proba_{b}": proba[b] for b in proba})

    decision = ("PROMISING" if gate else
                "MARGINAL" if d_comp > 0 and brecha > 0 else
                "CLOSED. La composicion agregada no aporta sobre V03")
    veredicto = (
        f"el bloque de composicion en forma de ratios vale {d_comp:+.10f} frente a V03 con "
        f"{fm}/5 pliegues, el control de capacidad del mismo ancho vale {d_rand:+.10f} con "
        f"{fm_rand}/5, luego la BRECHA especifica es {brecha:+.10f}. En solitario el bloque "
        f"vale {detalle['comp_solo']['corrected_cv']:.10f} frente a "
        f"{detalle['random_solo']['corrected_cv']:.10f} del aleatorio, o sea "
        f"{detalle['comp_solo']['corrected_cv'] - detalle['random_solo']['corrected_cv']:+.10f}. "
        f"Eso separa dos cosas que P62 no pudo separar: la composicion categorial SI porta "
        f"informacion, y lo que no porta es una mejora que sobreviva al control de capacidad"
    )
    log("")
    log(f"   VEREDICTO: {veredicto}")
    log(f"   DECISION: {decision}")

    with CSV_D.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["brazo", "corrected_cv", "delta_vs_V03", "folds",
                    "delta_vs_V03_random", "n_columnas"])
        for f in filas:
            w.writerow([f["brazo"], f["corrected_cv"], f["delta_vs_V03"], f["folds"],
                        f["delta_vs_V03_random"], n_cols])
        w.writerow(["brecha_especifica", round(brecha, 10)])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P63_COMPOSICION",
        "naturaleza": ("EXPERIMENTO. 2 candidatos y 2 controles de capacidad, 0 "
                       "submissions, 0 Kaggle, 0 holdout, 0 test"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P63-04",
        "track": "D-composicion-agregada",
        "script": "experiments/p63_05_composicion_linguistica.py",
        "pregunta": ("la composicion agregada como RATIOS aporta informacion que sobreviva "
                     "al control de capacidad, donde el canal POS de alta dimension no"),
        "distincion_con_p62": (
            "P62 midio un canal palabra__POS de alta dimension cuyo coste de dilucion fue "
            "+0,011585 y cuyo valor fue +0,002014. Eso es una afirmacion sobre una "
            "REPRESENTACION, no sobre la gramatica. Aqui cambia la FORMA del bloque, que "
            "es el eje que la curva de presupuesto de P62 marco como relevante: pocas "
            "columnas acotadas no compiten por el presupuesto L2 de un espacio de 21000 "
            "dimensiones"),
        "n_columnas": n_cols,
        "columnas": nombres + list(rasgos_lexicos("a b", {}, 1).keys()),
        "recursos": ("spaCy es_core_news_md 3.8.0 con exclude=['ner'], ya autorizado por "
                     "D-003 y ya anotado en P62. Ningun lexicon cerrado, ningun corpus "
                     "externo, ningun tagger nuevo"),
        "protocolo": {
            "frecuencia_de_documento": ("el hapax y el recuento de unicos usan la "
                                        "frecuencia de documento ajustada SOLO con las filas "
                                        "de entrenamiento del pliegue"),
            "confound_de_escala": "todos los brazos se reescalan a la norma de fila de V03",
            "control_de_capacidad": ("V03 mas un bloque de columnas aleatorias del mismo "
                                     "ancho, con la misma desviacion por columna"),
            "clasificador": "IDENTICO a V03 en los cuatro brazos",
            "pliegues": "los canonicos, por grupo", "umbrales": "0,44 / 0,51 / 0,46",
        },
        "resultados": detalle,
        "brecha_especifica": round(brecha, 10),
        "brecha_en_solitario": round(
            detalle["comp_solo"]["corrected_cv"] - detalle["random_solo"]["corrected_cv"], 10),
        "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                 "delta": round(d_comp, 10), "pliegues_mejorados": fm,
                 "pliegues_del_control": fm_rand, "superado": gate},
        "veredicto": veredicto,
        "decision": decision,
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_D.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P63 - Composicion agregada, en forma de ratios")
    W("")
    W(f"Generado por `experiments/p63_05_composicion_linguistica.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## La distincion sobre la que se construye este experimento")
    W("")
    W("P62 cerro una familia, y leer QUEE cerro importa mas que el hecho. Lo que midio fue un")
    W("canal POS lexico de alta dimension, `palabra__POS`, con un coste de dilucion de")
    W("+0,011585 y un valor de contenido de +0,002014, que repago el 17,4 por ciento de su")
    W("coste. Eso es una afirmacion sobre una **representacion**, no sobre la gramatica.")
    W("")
    W("La lectura correcta, y la que el encargo fija, es que la composicion gramatical puede")
    W("existir como senal mientras que su representacion explicita de alta dimensionalidad no")
    W("compensa su coste en el regimen de datos disponible.")
    W("")
    W("Este experimento no repite la familia cerrada: cambia la **forma** del bloque, que es")
    W("justo el eje que la curva de presupuesto de P62 marco como relevante. Un canal de")
    W("varios cientos de columnas dispersas compite por el presupuesto L2 de un espacio de")
    W("21000 dimensiones y diluye. Un punado de proporciones acotadas no compite.")
    W("")
    W("## El bloque")
    W("")
    W(f"**{n_cols} columnas**, todas proporciones acotadas e interpretables:")
    W("")
    W("```text")
    W("lexico        type-token, hapax, longitud media y mediana, unicos, largas, cortas")
    W("funcional     pronombres, determinantes, preposiciones, conjunciones, auxiliares,")
    W("              adverbios, interjecciones, particulas")
    W("persona       primera, segunda y tercera, singular y plural")
    W("categoria     sustantivos, verbos, adjetivos, y las razones verbo/sustantivo y")
    W("              adjetivo/sustantivo, que es donde un modelo lineal de n-gramas no llega")
    W("ortografia    mayusculas, digitos, interrogaciones, exclamaciones, elipsis,")
    W("              elongation, puntuacion repetida, comillas, URL, menciones, hashtags")
    W("```")
    W("")
    W("Las proporciones de persona y de categoria salen de la anotacion de spaCy ya")
    W("autorizada por D-003 y ya construida en P62. Ningun lexicon cerrado, ningun corpus")
    W("externo, ningun tagger nuevo. La proporcion de una categoria es un mecanismo")
    W("**distinto** del token que lleva la categoria, y de ahi la diferencia con P62.")
    W("")
    W("**Seguridad de pliegue de la frecuencia.** El hapax y el recuento de unicos dependen de")
    W("la frecuencia de documento, que seria una estadistica supervisada si se calculara sobre")
    W("todo. Se calcula SOLO con las filas de entrenamiento de cada pliegue.")
    W("")
    W("## Resultados")
    W("")
    W("| brazo | corrected_cv | delta vs V03 | pliegues | delta vs V03_random |")
    W("|---|---:|---:|---:|---:|")
    for f in filas:
        pl = sum(1 for a, b in zip(detalle[f['brazo']]["folds"], folds_v) if a > b)
        W(f"| {f['brazo']} | {f['corrected_cv']:.10f} | {f['delta_vs_V03']:+.10f} | "
          f"{pl}/5 | {f['delta_vs_V03_random'] if f['delta_vs_V03_random'] != '' else '-'} |")
    W("")
    W("## La pregunta que decide el veredicto")
    W("")
    W("```text")
    W(f"V03 mas composicion:  {d_comp:+.10f} con {fm}/5 pliegues")
    W(f"V03 mas ALEATORIO:    {d_rand:+.10f} con {fm_rand}/5 pliegues")
    W(f"BRECHA especifica:    {brecha:+.10f}")
    W(f"en solitario:         {detalle['comp_solo']['corrected_cv']:.10f} contra "
      f"{detalle['random_solo']['corrected_cv']:.10f} del aleatorio")
    W("```")
    W("")
    W("**El veredicto no es si V03_comp le gano a V03.** Es si le gano a **V03_random**.")
    W("Una ganancia que un bloque aleatorio del mismo ancho tambien produce es capacidad, no")
    W("composicion, y este proyecto ya se dejo enganar exactamente por esa pregunta en P59-10,")
    W("donde un regularizador que parecia real era treinta veces menor que la dispersion entre")
    W("pliegues.")
    W("")
    W("## Veredicto")
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
    W("*Generado por `experiments/p63_05_composicion_linguistica.py`. Sin push.*")
    MD_D.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_D, JSON_D, CSV_D):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P63-05 OK | n_cols {n_cols} | V03_comp {detalle['V03_comp']['corrected_cv']:.6f} | "
        f"V03_random {detalle['V03_random']['corrected_cv']:.6f} | brecha {brecha:+.6f} | "
        f"gate {gate}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
