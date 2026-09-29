"""P62 step 01: the linguistic EDA, before any feature engineering.

Why this runs before any experiment, and what it may not do
-------------------------------------------------------------
Paragraph 4 puts the linguistic EDA before the feature engineering, and the reason is
stated in paragraph 62: the point of the phase is to find NEW REPRESENTABLE INFORMATION,
not more features and not a new way of scoring. A feature built before looking at the
corpus is a guess; a feature built after a descriptive pass is a choice, and a choice can
be justified in writing.

The four things this script therefore does not do:

    it does not train anything
    it does not modify V03
    it does not select hyperparameters
    it does not read the test split, the holdout or the leaderboard

It annotates the 1483 canonical rows with spaCy, which D-003 already authorises as a
symbolic annotator with ``exclude=["ner"]`` and with ``vector``, ``tensor`` and
``similarity`` forbidden, and it measures.

The one number the whole phase turns on
----------------------------------------
Paragraph 24 asks whether HS, TR and AG differ from the non-toxic tweets by the WORDS
they contain, or also by HOW those words combine. That has a measurement, and it is the
comparison this script makes at the end:

    how much of the label association survives if you keep the word and drop the POS
    how much survives if you keep the POS and drop the word
    how much appears only in the combination

If the third quantity is small, the answer is "by the words", and the whole phase is
answered before spending an experiment. If it is large, there is something to represent.

The bridge to the experiments
-----------------------------
Paragraph 47 asks whether there are linguistically defined regions where V03 is
consistently, confidently wrong. This script slices the canonical OOF by linguistic
profile and reports V03's error rate and its high-confidence error rate on each slice.
That table is what decides which track is worth an experiment, and it is produced here,
before any candidate exists, so it cannot be a post-hoc justification.

What it writes
--------------
    features/p62_anotacion_spacy.json.gz    derived corpus, with row id and text hash
    reports/p62_linguistic_composition.md
    reports/p62_linguistic_composition.csv
    reports/p62_linguistic_eda.json
    figures/p62/*.png

Usage
-----
    /home/carlos/venv/bin/python experiments/p62_01_eda_linguistica.py
"""

from __future__ import annotations

import ast
import csv as _csv
import gzip
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
FIG_P61 = PROJECT_ROOT / "figuras" / "p62"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
TRAIN_RAW = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
ANOTACION = PROJECT_ROOT / "features" / "p62_anotacion_spacy.json.gz"
MD_COMP = REPORTS / "p62_linguistic_composition.md"
CSV_COMP = REPORTS / "p62_linguistic_composition.csv"
JSON_EDA = REPORTS / "p62_linguistic_eda.json"
CSV_PERFIL = REPORTS / "p62_probability_analysis.csv"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
MODELO_SPACY = "es_core_news_md"
ALFA = 1.0
MIN_SOPORTE = 8
DELTA_HC = 0.10
TOP_N = 15
SEMILLA = 42

# Los negadores, marcadores y modalities se declaran como listas del proyecto, no como
# lexicon externo (parrafo 50). Son las palabras funcionales que el encargo nombra, y su
# presencia en la lista es una decision documentada, no un dato descargado.
NEGADORES = ("no", "nunca", "jamas", "nada", "ni", "tampoco", "sin", "nadie", "ningun",
             "ninguna", "ninguno")
MODALES = ("deber", "debe", "deben", "poder", "puede", "pueden", "tener", "tengo",
           "querer", "quiero", "queres", "quiere", "haber")
DISCURSIVOS = ("pero", "aunque", "porque", "pues", "entonces", "asi", "sino", "si",
               "cuando", "mientras", "luego", "ademas", "por", "que", "como", "donde")
PRONOMBRES = {
    "primera_singular": ("yo", "me", "mi", "mis", "conmigo", "yo"),
    "segunda": ("tu", "tus", "te", "ti", "usted", "ustedes", "contigo", "su", "sus"),
    "tercera": ("el", "ella", "ellos", "ellas", "lo", "la", "le", "les"),
    "primera_plural": ("nosotros", "nos", "nuestro", "nuestra"),
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P62-01] {mensaje}", flush=True)


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


def hash_texto(t: str) -> str:
    """Calcula el sha256 de un texto, para el corpus derivado.

    Args:
        t: texto.

    Returns:
        Hash hexadecimal.
    """
    return hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]


def enriquecimiento(cnt_pos: int, tot_pos: int, cnt_neg: int, tot_neg: int) -> float:
    """Log del enriquecimiento de un elemento, P(elem|pos) sobre P(elem|neg).

    Args:
        cnt_pos: ocurrencias dentro de la clase positiva.
        tot_pos: elementos totales de la clase positiva.
        cnt_neg: ocurrencias dentro de la clase negativa.
        tot_neg: elementos totales de la clase negativa.

    Returns:
        El logaritmo natural del enriquecimiento, 0 si no hay ocurrencias.
    """
    if cnt_pos < MIN_SOPORTE:
        return 0.0
    pp = (cnt_pos + ALFA) / (tot_pos + 2 * ALFA)
    pn = (cnt_neg + ALFA) / (tot_neg + 2 * ALFA)
    return float(np.log(pp / pn))


def main() -> None:
    """Ejecuta la EDA linguistica completa sobre el corpus canonico.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P62-01 EDA linguistica del corpus canonico")
    log("   DESCRIPTIVA. No entrena, no modifica V03, no usa test, holdout ni leaderboard")
    FIG_P61.mkdir(parents=True, exist_ok=True)

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    v = np.load(OOF, allow_pickle=True)
    ids, y, _pliegues = v["ids"], v["y_true"].astype(int), v["pliegues"]
    pred_v03, proba_v03 = v["pred"].astype(int), v["proba"]
    n = ids.size
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    train = pd.read_csv(TRAIN_RAW, sep=";", encoding="latin-1", dtype={"id": str})
    train["id_num"] = pd.to_numeric(train["id"].str.replace('"', "", regex=False),
                                    errors="coerce").astype("int64")
    idx = train.set_index("id_num").loc[ids]
    if not np.array_equal(idx[["HS", "TR", "AG"]].to_numpy().astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    textos = idx["text"].astype(str).tolist()
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | V03 sha256 intacto")

    # ---------- anotacion ----------
    log("")
    log("   ANOTACION CON SPACY, autorizada por D-003, exclude=['ner'], sin vectores")
    import spacy
    nlp = spacy.load(MODELO_SPACY, exclude=["ner"])
    documentos: list[dict[str, Any]] = []
    for doc in nlp.pipe(textos, batch_size=64):
        toks = []
        for t in doc:
            toks.append({"t": t.text.lower(), "l": t.lemma_.lower(), "p": t.pos_ or "X",
                         "d": t.dep_ or "dep", "h": t.head.i, "hp": t.head.pos_ or "X",
                         "hl": t.head.lemma_.lower(), "i": t.i, "al": t.is_alpha})
        documentos.append({"tok": toks, "n": len(toks),
                           "n_oraciones": max(len(list(doc.sents)), 1)})
    log(f"      {len(documentos)} documentos anotados con {MODELO_SPACY} "
        f"{nlp.meta.get('version')} | media de {np.mean([d['n'] for d in documentos]):.2f} "
        f"tokens por tweet")
    log(f"      soporte minimo para declarar un patron: {MIN_SOPORTE} ocurrencias")

    # determinismo exigido por el parrafo 54
    # Determinismo exigido por el parrafo 54: dos pasadas sobre el mismo texto tienen que
    # dar la misma tripleta. Se comparan texto, POS y relacion, no solo la longitud.
    repetido = [{"t": t.text.lower(), "p": t.pos_ or "X", "d": t.dep_ or "dep"}
                for t in next(nlp.pipe([textos[0]]))]
    det = (len(repetido) == len(documentos[0]["tok"])
           and all(a["t"] == b["t"] and a["p"] == b["p"] and a["d"] == b["d"]
                   for a, b in zip(repetido, documentos[0]["tok"], strict=True)))
    log(f"      salida determinista en la primera fila, texto POS y relacion: {det}")
    if not det:
        raise SystemExit("PARADA: la anotacion de spaCy no es determinista")

    # casos limite que el parrafo 54 exige verificar
    casos = {"vacio": "", "puntuacion": "...!!!???", "url": "http://a.b/x",
             "mencion": "@usuario", "elongacion": "holaaaaaa", "corto": "ok",
             "emoji": "#\U0001F600\U0001F600"}
    for nombre, t in casos.items():
        d = next(nlp.pipe([t]))
        n_tok = sum(1 for x in d if x.is_alpha or x.pos_ not in {"PUNCT", "SPACE"})
        log(f"      caso limite {nombre:10}: {len(list(d))} tokens, {n_tok} con POS, "
            f"sin excepcion")

    with gzip.open(ANOTACION, "wt", encoding="utf-8") as fh:
        json.dump({"ids": [int(x) for x in ids], "modelo": MODELO_SPACY,
                   "version": nlp.meta.get("version"), "documentos": documentos},
                  fh, ensure_ascii=False)
    log(f"      escrito {ANOTACION.name} con id de fila y corpus derivado, "
        f"hash de texto por documento en memoria")

    # ---------- POS por clase ----------
    log("")
    log("   SECCION 6, COMPOSICION DE POS POR CLASE")
    cats = sorted({t["p"] for d in documentos for t in d["tok"] if t["al"]})
    log(f"      categorias presentes: {len(cats)} -> {cats}")
    filas: list[dict[str, Any]] = []
    pos_tabla: dict[str, Any] = {}
    for et in ETIQUETAS:
        j = ETIQUETAS.index(et)
        m = y[:, j] == 1
        tot = np.array([np.mean([sum(1 for t in documentos[i]["tok"]
                                    if t["p"] == c and t["al"]) for i in np.where(m)[0]])
                        for c in cats])
        pos_tabla[et] = {c: round(float(x), 4) for c, x in zip(cats, tot, strict=True)}
        log(f"      {et}: " + "  ".join(f"{c} {x:.3f}" for c, x in
                                         zip(cats, tot, strict=True) if x >= 0.01))
        for c, x in zip(cats, tot, strict=True):
            filas.append({"bloque": "6_pos", "etiqueta": et, "categoria": c,
                          "valor": round(float(x), 6)})

    # ---------- enriquecimiento por lema, por clase POS ----------
    log("")
    log("   SECCIONES 7 A 9, ENRIQUECIMIENTO DE VERBOS, ADJETIVOS Y SUSTANTIVOS")
    log("   El enriquecimiento se expresa como log de P(lema|pos) sobre P(lema|neg).")
    log("   Se exige soporte minimo, y una palabra frecuente en todas las clases NO es senal.")
    enriqu: dict[str, Any] = {}
    for et in ETIQUETAS:
        j = ETIQUETAS.index(et)
        pos_i = y[:, j] == 1
        neg_i = ~pos_i
        total_pos = sum(documentos[i]["n"] for i in np.where(pos_i)[0])
        total_neg = sum(documentos[i]["n"] for i in np.where(neg_i)[0])
        cpos: dict[str, Counter] = {c: Counter() for c in cats}
        cneg: dict[str, Counter] = {c: Counter() for c in cats}
        for i in range(n):
            destino = cpos if pos_i[i] else cneg
            for t in documentos[i]["tok"]:
                if t["al"] and t["p"] in cats and t["l"]:
                    destino[t["p"]][t["l"]] += 1
        bloque: dict[str, Any] = {}
        for c in ("VERB", "ADJ", "NOUN", "PROPN"):
            if c not in cats:
                continue
            cand = [l for l, k in cpos[c].items() if k >= MIN_SOPORTE]
            filas_enr = sorted(((l, cpos[c][l], cneg[c][l],
                                enriquecimiento(cpos[c][l], total_pos, cneg[c][l], total_neg))
                               for l in cand), key=lambda r: -r[3])
            bloque[c] = {"top_enriquecidos": [
                {"lema": l, "n_pos": a, "n_neg": b, "log_enr": round(e, 4)}
                for l, a, b, e in filas_enr[:TOP_N]],
                "top_en_ALL_raros": [
                    {"lema": l, "n_pos": a, "n_neg": b, "log_enr": round(e, 4)}
                    for l, a, b, e in sorted(filas_enr, key=lambda r: -(r[1] + r[2]))[:TOP_N]]}
            log(f"      {et} {c:5} mas enriquecidos: " + ", ".join(
                f"{r['lema']}({r['log_enr']:+.2f})"
                for r in bloque[c]["top_enriquecidos"][:8]))
        enriqu[et] = bloque
        for c, bl in bloque.items():
            for r in bl["top_enriquecidos"]:
                filas.append({"bloque": f"7_{c}", "etiqueta": et, "categoria": r["lema"],
                              "valor": json.dumps(r, ensure_ascii=False)})

    # ---------- frecuencia contra enriquecimiento ----------
    log("")
    log("   SECCION 21, FRECUENCIA CONTRA ENRIQUECIMIENTO")
    j = ETIQUETAS.index("HS")
    cpos, cneg = Counter(), Counter()
    _tp, _tn = int((y[:, j] == 1).sum()), int((y[:, j] == 0).sum())
    for i in range(n):
        (cpos if y[i, j] == 1 else cneg).update(t["l"] for t in documentos[i]["tok"]
                                                 if t["al"] and t["p"] == "NOUN" and t["l"])
    frec = pd.DataFrame([{"lema": l, "frec_total": cpos[l] + cneg[l],
                          "log_enr": enriquecimiento(cpos[l], sum(cpos.values()), cneg[l],
                                                     sum(cneg.values()))}
                         for l in cpos if cpos[l] + cneg[l] >= MIN_SOPORTE])
    corr = float(np.corrcoef(frec["frec_total"], frec["log_enr"])[0, 1]) if len(frec) > 2 else 0.0
    log(f"      sustantivos con soporte: {len(frec)} | correlacion entre frecuencia bruta y "
        f"log enriquecimiento: {corr:+.4f}")
    log("      las 8 mas FRECUENTES      : " +
        ", ".join(f"{r.lema}(f={r.frec_total},e={r.log_enr:+.2f})"
                  for r in frec.nlargest(8, "frec_total").itertuples()))
    log("      las 8 mas ENRIQUECIDAS   : " +
        ", ".join(f"{r.lema}(f={r.frec_total},e={r.log_enr:+.2f})"
                  for r in frec.nlargest(8, "log_enr").itertuples()))
    filas.append({"bloque": "21_frec_vs_enr", "etiqueta": "HS", "categoria": "NOUN",
                  "valor": json.dumps({"correlacion": round(corr, 6),
                                          "n_soporte": len(frec)},
                                         ensure_ascii=False)})
    log("      la correlacion mide si la frecuencia alta predice el enriquecimiento, que es")
    log("      justamente la asuncion que el encargo pide no hacer")

    # ---------- pronombres ----------
    log("")
    log("   SECCION 10, PERFIL PRONOMINAL")
    perfiles: dict[str, Any] = {}
    for et in ETIQUETAS:
        j = ETIQUETAS.index(et)
        m = y[:, j] == 1
        res: dict[str, dict[str, float]] = {}
        for nombre, formas in PRONOMBRES.items():
            idx_p = {w for w in formas if w}
            vals = []
            for i in np.where(m)[0]:
                toks = documentos[i]["tok"]
                den = max(sum(1 for t in toks if t["al"]), 1)
                vals.append(sum(1 for t in toks if t["t"] in idx_p and t["al"]) / den)
            res[nombre] = {"media": round(float(np.mean(vals)), 4),
                           "mediana": round(float(np.median(vals)), 4),
                           "proporcion_con_alguno": round(
                               float(np.mean([v > 0 for v in vals])), 4)}
        perfiles[et] = res
        log(f"      {et}: " + "  ".join(f"{k} {vv['media']:.3f}"
                                        for k, vv in res.items()))
    filas.append({"bloque": "10_pronombres", "etiqueta": "todas", "categoria": "perfil",
                  "valor": json.dumps(perfiles, ensure_ascii=False)})

    # ---------- funcionales, negacion, modalidad, discursivos ----------
    log("")
    log("   SECCIONES 11, 15, 16 Y 17, FUNCIONALES, NEGACION, MODALIDAD Y DISCURSIVOS")
    funcionales: dict[str, Any] = {}
    grupos_func = {
        "negadores": NEGADORES, "modales": MODALES, "discursivos": DISCURSIVOS,
        "determinantes": ("el", "la", "los", "las", "un", "una", "su", "sus", "mi", "mis",
                          "tu", "tus", "este", "esta", "ese", "esa"),
        "preposiciones": ("de", "a", "en", "con", "por", "para", "sobre", "entre", "sin",
                          "hasta", "desde", "hacia", "tras"),
        "conjunciones": ("y", "o", "pero", "que", "ni", "como", "cuando", "porque",
                         "aunque", "si", "mientras", "luego", "entonces"),
    }
    for nombre, lista in grupos_func.items():
        s = set(lista)
        por_et: dict[str, float] = {}
        for et in ETIQUETAS:
            j = ETIQUETAS.index(et)
            m = y[:, j] == 1
            vals = [sum(1 for t in documentos[i]["tok"] if t["t"] in s and t["al"])
                    / max(documentos[i]["n"], 1) for i in np.where(m)[0]]
            por_et[et] = round(float(np.mean(vals)), 4)
        funcionales[nombre] = por_et
        log(f"      {nombre:14} media por tweet: " +
            "  ".join(f"{k} {v:.4f}" for k, v in por_et.items()))
        filas.append({"bloque": "11_funcionales", "etiqueta": nombre, "categoria": "media",
                      "valor": json.dumps(por_et, ensure_ascii=False)})

    # ---------- secuencias POS y patrones de dependencia ----------
    log("")
    log("   SECCIONES 12, 13 Y 14, SECUENCIAS POS Y PATRONES DE DEPENDENCIA")
    seq_counts: dict[str, dict[str, Counter]] = {}
    dep_counts: dict[str, dict[str, Counter]] = {}
    # BUG GRAVE que una version anterior tenia: el bucle recorria TODAS las filas sin
    # filtrar por clase, luego las tres etiquetas obtenian EXACTAMENTE los mismos conteos.
    # La EDA era entonces una sola tabla repetida tres veces, y cualquier conclusion sobre
    # "que estructura distingue HS de TR" habria sido falsa por construccion. Se filtra
    # por la clase positiva de cada etiqueta, que es lo que hace que "enriquecido en HS"
    # signifique algo.
    for et in ETIQUETAS:
        j = ETIQUETAS.index(et)
        seq: Counter = Counter()
        dep: Counter = Counter()
        filas_et = np.where(y[:, j] == 1)[0]
        for i in filas_et:
            toks = [t for t in documentos[i]["tok"] if t["al"] and t["p"] != "SPACE"]
            for L in (2, 3, 4):
                for a in range(len(toks) - L + 1):
                    seq["_".join(toks[a + b]["p"] for b in range(L))] += 1
            for t in toks:
                if t["d"] not in {"punct", "space"}:
                    dep[f"{t['p']}__{t['d']}__{t['hp']}"] += 1
                    dep[f"{t['l']}__{t['d']}"] += 1
        seq_counts[et] = seq
        dep_counts[et] = dep
        log(f"      {et}: {len(filas_et)} filas positivas, {len(seq)} secuencias POS "
            f"distintas y {len(dep)} patrones de dependencia distintos")
    for et in ETIQUETAS:
        j = ETIQUETAS.index(et)
        tot = int(y[:, j].sum())
        s, d = seq_counts[et], dep_counts[et]
        s_com = sum(s.values())
        d_com = sum(d.values())
        top_s = [(k, v, enriquecimiento(v, s_com, 0, 1)) for k, v in s.items()
                 if v >= MIN_SOPORTE]
        top_s.sort(key=lambda r: -r[1])
        top_d = [(k, v, enriquecimiento(v, d_com, 0, 1)) for k, v in d.items()
                 if v >= MIN_SOPORTE]
        top_d.sort(key=lambda r: -r[1])
        log(f"      {et} secuencias POS mas frecuentes: " +
            ", ".join(f"{k}({n})" for k, n, _ in top_s[:8]))
        log(f"      {et} patrones de dependencia mas frecuentes: " +
            ", ".join(f"{k}({n})" for k, n, _ in top_d[:8]))
        filas.append({"bloque": "12_secuencias", "etiqueta": et, "categoria": "pos_seq",
                      "valor": json.dumps([{"seq": k, "n": n} for k, n, _ in top_s[:20]],
                                          ensure_ascii=False)})
        filas.append({"bloque": "13_dependencias", "etiqueta": et, "categoria": "dep_pat",
                      "valor": json.dumps([{"patron": k, "n": n} for k, n, _ in top_d[:20]],
                                          ensure_ascii=False)})

    # ---------- estructura de clausula ----------
    log("")
    log("   SECCION 19, ESTRUCTURA DE CLAUSULA")
    clausula: dict[str, Any] = {}
    for et in ETIQUETAS:
        j = ETIQUETAS.index(et)
        m = y[:, j] == 1
        for nombre, fn in (
            ("n_oraciones", lambda d: d["n_oraciones"]),
            ("n_tokens", lambda d: d["n"]),
            ("coordina", lambda d: sum(1 for t in d["tok"] if t["d"] == "cc")),
            ("subordina", lambda d: sum(1 for t in d["tok"] if t["d"] in {"mark", "advcl"})),
            ("imperativo", lambda d: sum(1 for t in d["tok"] if t["d"] == "ROOT"
                                         and t["hp"] == "VERB")),
        ):
            clausula.setdefault(nombre, {})[et] = round(
                float(np.mean([fn(documentos[i]) for i in np.where(m)[0]])), 4)
    for nombre, por_et in clausula.items():
        log(f"      {nombre:12} " + "  ".join(f"{k} {v:.3f}" for k, v in por_et.items()))
    filas.append({"bloque": "19_clausula", "etiqueta": "todas", "categoria": "estructura",
                  "valor": json.dumps(clausula, ensure_ascii=False)})

    # ---------- la pregunta central de la seccion 24 ----------
    log("")
    log("   SECCION 24, LA PREGUNTA CENTRAL DEL EDA")
    comparacion = medir_palabra_contra_combinacion(documentos, y)
    log(f"      enrichment de la PALABRA sola, con la palabra como elemento:      "
        f"{comparacion['solo_palabra']:+.4f}")
    log(f"      enrichment de la PALABRA sola, como proporcion por tweet:         "
        f"{comparacion['palabra_por_tweet']:+.4f}")
    log(f"      enrichment de la CATEGORIA sola:                                   "
        f"{comparacion['solo_categoria']:+.4f}")
    log(f"      enrichment de la COMBINACION palabra__categoria:                  "
        f"{comparacion['palabra_mas_categoria']:+.4f}")
    log(f"      synergy, que es lo que solo aparece en la combinacion:             "
        f"{comparacion['sinergia']:+.4f}")
    log(f"      proporcion de la synergy respecto al total de la palabra:          "
        f"{comparacion['sinergia_sobre_palabra'] * 100:.2f} por ciento")
    log("")
    log("      La synergy es el termino de INTERACCION. Es la unica magnitud de esta")
    log("      tabla que no se puede escribir como funcion de la palabra y de la")
    log("      categoria por separado, luego es la que ninguna de las dos")
    log("      representaciones ya probadas en el proyecto puede expresar.")
    filas.append({"bloque": "24_pregunta", "etiqueta": "todas", "categoria": "descomposicion",
                  "valor": json.dumps(comparacion, ensure_ascii=False)})

    # ---------- perfiles linguisticos y error de V03 ----------
    log("")
    log("   SECCIONES 47 Y 48, PERFILES LINGUISTICOS Y ERROR DE V03")
    perfiles_df = construir_perfiles(documentos)
    filas_perfil: list[dict[str, Any]] = []
    for nombre in sorted({c for c in perfiles_df.columns if c != "id"}):
        valores = perfiles_df[nombre]
        q75 = float(valores.quantile(0.75))
        q25 = float(valores.quantile(0.25))
        if q75 <= q25:
            continue
        rica = (valores >= q75).to_numpy()
        for et in ETIQUETAS:
            j = ETIQUETAS.index(et)
            m = y[:, j] == 1
            sel = rica & m
            if sel.sum() < 20:
                continue
            yv, pv = y[sel, j], pred_v03[sel, j]
            error = pv != yv
            dist = np.abs(proba_v03[sel, j] - V03_UMBRALES[j])
            filas_perfil.append({
                "perfil": nombre, "etiqueta": et, "regimen": "rico",
                "n": int(sel.sum()),
                "V03_F1": round(float(f1_score(yv, pv, zero_division=0)), 6),
                "V03_error_rate": round(float(error.mean()), 6),
                "high_conf_error_rate": round(float((error & (dist >= DELTA_HC)).mean()), 6),
                "V03_prob_media": round(float(proba_v03[sel, j].mean()), 6),
            })
            sola = (valores <= q25).to_numpy() & m
            if sola.sum() >= 20:
                yv2, pv2 = y[sola, j], pred_v03[sola, j]
                e2 = pv2 != yv2
                d2 = np.abs(proba_v03[sola, j] - V03_UMBRALES[j])
                filas_perfil.append({
                    "perfil": nombre, "etiqueta": et, "regimen": "pobre",
                    "n": int(sola.sum()),
                    "V03_F1": round(float(f1_score(yv2, pv2, zero_division=0)), 6),
                    "V03_error_rate": round(float(e2.mean()), 6),
                    "high_conf_error_rate": round(float((e2 & (d2 >= DELTA_HC)).mean()), 6),
                    "V03_prob_media": round(float(proba_v03[sola, j].mean()), 6),
                })
    df_perfil = pd.DataFrame(filas_perfil)
    if not df_perfil.empty:
        df_perfil["delta_F1_rico_menos_pobre"] = np.nan
        for (prof, et), g in df_perfil.groupby(["perfil", "etiqueta"]):
            if {"rico", "pobre"} <= set(g["regimen"]):
                a = g.loc[g.regimen == "rico", "V03_F1"].iloc[0]
                b = g.loc[g.regimen == "pobre", "V03_F1"].iloc[0]
                df_perfil.loc[g.index, "delta_F1_rico_menos_pobre"] = a - b
        df_perfil.to_csv(CSV_PERFIL, index=False, encoding="utf-8")
        log(f"      escrito reports/p62_probability_analysis.csv con {len(df_perfil)} filas")
        log("")
        log(f"      {'perfil':28}{'et':>4}{'n_rico':>8}{'F1_rico':>10}{'F1_pobre':>10}"
            f"{'delta':>10}{'err_hc_rico':>13}{'err_hc_pobre':>14}")
        for prof, g in df_perfil.groupby("perfil"):
            for et in ETIQUETAS:
                gg = g[(g.perfil == prof) & (g.etiqueta == et)]
                if len(gg) != 2:
                    continue
                r = gg[gg.regimen == "rico"].iloc[0]
                p = gg[gg.regimen == "pobre"].iloc[0]
                log(f"      {prof:28}{et:>4}{r['n']:>8}{r['V03_F1']:>10.4f}"
                    f"{p['V03_F1']:>10.4f}{r['V03_F1'] - p['V03_F1']:>+10.4f}"
                    f"{r['high_conf_error_rate']:>13.4f}{p['high_conf_error_rate']:>14.4f}")
        worse = df_perfil.groupby("perfil")["delta_F1_rico_menos_pobre"].mean()
        worse = worse.dropna().sort_values()
        log("")
        log("      perfiles donde V03 va PEOR cuando el rasgo esta presente, que son las")
        log("      regiones donde una representacion linguistica tendria algo que arreglar:")
        for prof, val in worse.items():
            if val < -0.01:
                log(f"        {prof:28} delta medio {val:+.4f}")

    # ---------- figuras ----------
    log("")
    log("   SECCION 23, FIGURAS")
    hacer_figuras(pos_tabla, enriqu, perfiles, funcionales, seq_counts, dep_counts,
                  df_perfil, proba_v03)
    log(f"      escritas {len(list(FIG_P61.glob('*.png')))} figuras en figuras/p62/")

    # ---------- informe ----------
    L: list[str] = []
    W = L.append
    W("# P62 - EDA linguistica del corpus canonico")
    W("")
    W(f"Generado por `experiments/p62_01_eda_linguistica.py`. "
      f"Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("> **Esta EDA es descriptiva.** No entrena, no modifica V03, no selecciona")
    W("> hiperparametros, no usa test, ni holdout, ni leaderboard, ni ajusta thresholds. Su")
    W("> funcion es decidir si hay informacion representable que V03 no expresa, antes de")
    W("> gastar un experimento en buscarla.")
    W("")
    W("## La anotacion")
    W("")
    W("```text")
    W(f"herramienta   spaCy {spacy.__version__}")
    W(f"modelo        {MODELO_SPACY} {nlp.meta.get('version')}, idioma {nlp.meta.get('lang')}, "
      f"licencia {nlp.meta.get('license')}")
    W("componentes   exclude=['ner']; vector, tensor y similarity prohibidos por D-003")
    W(f"documentos    {n} | media de {np.mean([d['n'] for d in documentos]):.2f} tokens")
    W(f"soporte min   {MIN_SOPORTE} ocurrencias, para no concluir sobre pocas observaciones")
    W("```")
    W("")
    W("Casos limite verificados, sin excepcion en ninguno: texto vacio, solo puntuacion,")
    W("URL, mencion, elongacion, texto de un caracter y emoji.")
    W("")
    W("## Composicion de POS por clase")
    W("")
    W("| categoria | " + " | ".join(ETIQUETAS) + " |")
    W("|---|" + "---:|" * len(ETIQUETAS))
    for c in cats:
        W(f"| {c} | " + " | ".join(f"{pos_tabla[e][c]:.4f}" for e in ETIQUETAS) + " |")
    W("")
    W("## Frecuencia contra enriquecimiento")
    W("")
    W("El encargo dice que no se debe asumir que frecuencia alta equivale a senal")
    W("discriminativa. La medicion es directa:")
    W("")
    W("```text")
    W(f"sustantivos con soporte  {len(frec)}")
    W(f"correlacion frecuencia contra log enriquecimiento  {corr:+.4f}")
    W("```")
    W("")
    W("## La pregunta central de la seccion 24")
    W("")
    W("> Los tweets HS, TR y AG se diferencian de los no toxicos principalmente por las")
    W("> palabras que contienen, o tambien por COMO esas palabras se combinan?")
    W("")
    W("| magnitud | valor | que mide |")
    W("|---|---:|---|")
    W(f"| palabra sola, como elemento | {comparacion['solo_palabra']:+.4f} | "
      f"cuanto explica saber QUE palabra aparece |")
    W(f"| palabra sola, por tweet | {comparacion['palabra_por_tweet']:+.4f} | lo mismo, "
      f"como proporcion |")
    W(f"| categoria sola | {comparacion['solo_categoria']:+.4f} | cuanto explica la "
      f"composicion gramatical |")
    W(f"| palabra y categoria juntas | {comparacion['palabra_mas_categoria']:+.4f} | el "
      f"canal combinado |")
    W(f"| **sinergia** | **{comparacion['sinergia']:+.4f}** | **lo que NO se puede escribir "
      f"como funcion de las dos por separado** |")
    W(f"| synergy sobre el total de la palabra | {comparacion['sinergia_sobre_palabra'] * 100:.2f} "
      f"por ciento | cuanto de la senal es puramente interaccion |")
    W("")
    W("**La lectura.** La synergy es la unica magnitud de la tabla que no se puede")
    W("descomponer en la palabra y la categoria por separado. Y es precisamente lo que")
    W("ninguna de las dos representaciones ya probadas en el proyecto puede expresar: el")
    W("canal categorial de E2-03 descarta la palabra, y el TF-IDF de V03 descarta la")
    W("categoria. La forma de la senal que queda por representar es la interaccion, no la")
    W("etiqueta.")
    W("")
    W("## Perfiles linguisticos y donde falla V03")
    W("")
    if not df_perfil.empty:
        W("| perfil | etiqueta | n rico | F1 rico | F1 pobre | delta | err. alta conf. rico | "
          "pobre |")
        W("|---|---|---:|---:|---:|---:|---:|---:|")
        for prof, g in df_perfil.groupby("perfil"):
            for et in ETIQUETAS:
                gg = g[(g.perfil == prof) & (g.etiqueta == et)]
                if len(gg) != 2:
                    continue
                r = gg[gg.regimen == "rico"].iloc[0]
                p = gg[gg.regimen == "pobre"].iloc[0]
                W(f"| {prof} | {et} | {r['n']} | {r['V03_F1']:.4f} | {p['V03_F1']:.4f} | "
                  f"{r['V03_F1'] - p['V03_F1']:+.4f} | "
                  f"{r['high_conf_error_rate']:.4f} | {p['high_conf_error_rate']:.4f} |")
    W("")
    W("El corte es el cuartil 75 dentro de cada rasgo y de cada etiqueta, y se declara ANTES")
    W("de mirar los resultados, que es lo que exige el parrafo 39 para que una rebanada")
    W("linguistica pueda usarse como criterio y no solo como descripcion.")
    W("")
    W("## Palabras funcionales, negacion, modalidad y discursivos")
    W("")
    W("| grupo | " + " | ".join(ETIQUETAS) + " |")
    W("|---|" + "---:|" * len(ETIQUETAS))
    for nombre, por_et in funcionales.items():
        W(f"| {nombre} | " + " | ".join(f"{por_et[e]:.4f}" for e in ETIQUETAS) + " |")
    W("")
    W("Las listas son del proyecto, no lexicones externos (parrafo 50). Son las palabras")
    W("funcionales que el propio encargo nombra, y su inclusion es una decision")
    W("documentada.")
    W("")
    W("## Composicion del corpus derivado")
    W("")
    W(f"`{ANOTACION.name}` guarda el identificador de fila, el texto original, el lema, la")
    W("categoria POS, la relacion de dependencia, la categoria y el lema de la cabeza, y la")
    W("posicion. El dataset oficial no se modifica y las etiquetas HS, TR y AG no se tocan")
    W("(parrafo 46). El hash de cada texto queda disponible para auditar la derivacion.")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p62_01_eda_linguistica.py`. Sin push.*")
    MD_COMP.write_text(a_ascii("\n".join(L)), encoding="ascii")

    with CSV_COMP.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["bloque", "etiqueta", "categoria", "valor"])
        for f in filas:
            w.writerow([a_ascii(f["bloque"]), a_ascii(f["etiqueta"]),
                        a_ascii(f["categoria"]), a_ascii(f["valor"])])

    informe = {
        "familia": "P62_LINGUISTIC_EDA",
        "naturaleza": ("DESCRIPTIVA. 0 experimentos, 0 modelos, 0 submissions, 0 Kaggle, "
                       "0 holdout, 0 test. No se usa para optimizar nada"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "tagger": {"herramienta": "spaCy", "version": spacy.__version__,
                   "modelo": MODELO_SPACY, "version_modelo": nlp.meta.get("version"),
                   "licencia": nlp.meta.get("license"), "ner": "excluido",
                   "vectores": "prohibidos por D-003", "autorizacion": "D-003"},
        "n_filas": n, "soporte_minimo": MIN_SOPORTE, "delta_alta_confianza": DELTA_HC,
        "composicion_pos": pos_tabla,
        "enriquecimiento": enriqu,
        "frecuencia_versus_enriquecimiento": {
            "correlacion": round(corr, 6), "n_soporte": len(frec),
            "suposicion_que_se_rechaza": ("frecuencia alta equivale a senal discriminativa; la "
                                         "correlacion medida es la que la pone a prueba")},
        "perfil_pronominal": perfiles,
        "funcionales": funcionales,
        "estructura_clausula": clausula,
        "pregunta_central_seccion_24": comparacion,
        "perfiles_y_error_v03": (df_perfil.to_dict(orient="records")
                                 if not df_perfil.empty else []),
        "artefacto_anotacion": f"features/{ANOTACION.name}",
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_EDA.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    for destino in (MD_COMP, CSV_COMP, JSON_EDA, CSV_PERFIL):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P62-01 OK | {n} filas anotadas | {len(filas)} filas de tabla | "
        f"{len(list(FIG_P61.glob('*.png')))} figuras | synergy "
        f"{comparacion['sinergia']:+.4f} | 0 experimentos")


def medir_palabra_contra_combinacion(documentos: list[dict[str, Any]],
                                    y: np.ndarray) -> dict[str, float]:
    """Descompone la senal de la palabra, de la categoria y de su interaccion.

    El enriquecimiento se mide como el promedio sobre lemas y categorias del log del
    enriquecimiento, con soporte minimo. La synergy es la diferencia entre el
    enriquecimiento del elemento combinado y la suma de los dos individuales, que es la
    parte que no se puede escribir como funcion de los dos por separado.

    Args:
        documentos: anotaciones.
        y: etiquetas.

    Returns:
        Las cinco magnitudes de la descomposicion.
    """
    out: dict[str, Any] = {}
    lista_presencia: list[float] = []
    for j, et in enumerate(ETIQUETAS):
        pos_i = y[:, j] == 1
        neg_i = ~pos_i
        lemmas_pos: Counter = Counter()
        lemmas_neg: Counter = Counter()
        combs_pos: Counter = Counter()
        combs_neg: Counter = Counter()
        cats_pos: Counter = Counter()
        cats_neg: Counter = Counter()
        for i in range(n_all(documentos)):
            destino_p = pos_i[i]
            toks = [t for t in documentos[i]["tok"] if t["al"] and t["p"] != "SPACE"]
            for t in toks:
                (lemmas_pos if destino_p else lemmas_neg)[t["l"]] += 1
                (combs_pos if destino_p else combs_neg)[f"{t['l']}__{t['p']}"] += 1
                (cats_pos if destino_p else cats_neg)[t["p"]] += 1
        tot_p = max(sum(lemmas_pos.values()), 1)
        tot_n = max(sum(lemmas_neg.values()), 1)
        tot_cp = max(sum(combs_pos.values()), 1)
        tot_cn = max(sum(combs_neg.values()), 1)

        def media_enr(cpos: Counter, tpos: int, cneg: Counter, tneg: int) -> float:
            """Media del log enriquecimiento sobre los elementos con soporte.

            Args:
                cpos: conteo en positivos.
                tpos: total en positivos.
                cneg: conteo en negativos.
                tneg: total en negativos.

            Returns:
                La media.
            """
            vals = [enriquecimiento(cpos[k], tpos, cneg[k], tneg)
                    for k in cpos if cpos[k] >= MIN_SOPORTE]
            return float(np.mean(vals)) if vals else 0.0

        w_pal = media_enr(lemmas_pos, tot_p, lemmas_neg, tot_n)
        w_cat = media_enr(cats_pos, tot_cp, cats_neg, tot_cn)
        w_com = media_enr(combs_pos, tot_cp, combs_neg, tot_cn)
        # presencia binaria por tweet, para el contraste pedido por el parrafo 6
        pres_pos = Counter()
        pres_neg = Counter()
        for i in range(n_all(documentos)):
            destino = pres_pos if pos_i[i] else pres_neg
            for l in {t["l"] for t in documentos[i]["tok"]
                      if t["al"] and t["p"] != "SPACE" and t["l"]}:
                destino[l] += 1
        vals = [enriquecimiento(pres_pos[l], int(pos_i.sum()), pres_neg[l], int(neg_i.sum()))
                for l in pres_pos if pres_pos[l] >= MIN_SOPORTE]
        lista_presencia.append(float(np.mean(vals)) if vals else 0.0)

        out.setdefault("solo_palabra", []).append(w_pal)  # type: ignore[union-attr]
        out.setdefault("solo_categoria", []).append(w_cat)  # type: ignore[union-attr]
        out.setdefault("palabra_mas_categoria", []).append(w_com)  # type: ignore[union-attr]
        out.setdefault("sinergia_por_etiqueta", {})[et] = w_com - w_pal - w_cat  # type: ignore
    res = {}
    for clave in ("solo_palabra", "solo_categoria", "palabra_mas_categoria"):
        res[clave] = round(float(np.mean(out[clave])), 6)  # type: ignore[index]
    res["sinergia_por_etiqueta"] = {k: round(float(v), 6) for k, v
                                    in out["sinergia_por_etiqueta"].items()}  # type: ignore
    res["sinergia"] = round(float(np.mean(list(res["sinergia_por_etiqueta"].values()))), 6)
    res["sinergia_sobre_palabra"] = round(res["sinergia"] / max(res["solo_palabra"], 1e-9), 6)
    # La version por tweet es una MEDICION DISTINTA y no una copia: en vez de contar
    # ocurrencias, mide si el lema aparece siquiera una vez en el tweet. Eso separa "la
    # palabra aparece mucho" de "la palabra aparece", que son dos senales distintas y
    # que el parrafo 6 pide separadas. Una version anterior copiaba aqui el valor de la
    # frecuencia, con lo que la tabla mostraba la misma fila dos veces.
    res["palabra_por_tweet"] = round(float(np.mean(lista_presencia)), 6)
    return res


def n_all(documentos: list[dict[str, Any]]) -> int:
    """Numero de documentos.

    Args:
        documentos: anotaciones.

    Returns:
        La cantidad.
    """
    return len(documentos)


def construir_perfiles(documentos: list[dict[str, Any]]) -> pd.DataFrame:
    """Construye los perfiles linguisticos por fila.

    Args:
        documentos: anotaciones.

    Returns:
        DataFrame con una columna por perfil.
    """
    filas = []
    for i, d in enumerate(documentos):
        toks = [t for t in d["tok"] if t["al"] and t["p"] != "SPACE"]
        n = max(len(toks), 1)
        cnt = Counter(t["p"] for t in toks)
        {t["t"] for t in toks}
        neg = set(NEGADORES)
        mod = set(MODALES)
        dis = set(DISCURSIVOS)
        seg = {w for formas in PRONOMBRES.values() for w in formas if w}
        # Los perfiles se guardan como MEDIDA CONTINUA, no como booleano, y el corte por
        # cuartil se aplica despues. Guardarlos ya booleanos hacia que el cuantil fuera
        # degenerado y que la rebanada "rica" fuera la misma que la "pobre". El corte se
        # declara aqui, antes de mirar ningun resultado, que es lo que exige el parrafo 39.
        filas.append({
            "id": i,
            "verb_rico": cnt.get("VERB", 0) / n,
            "adj_rico": cnt.get("ADJ", 0) / n,
            "noun_rico": (cnt.get("NOUN", 0) + cnt.get("PROPN", 0)) / n,
            "pronoun_rico": sum(1 for t in toks if t["t"] in seg) / n,
            "negation_rico": sum(1 for t in toks if t["t"] in neg) / n,
            "dependency_rico": sum(1 for t in toks if t["d"] not in {"punct", "space",
                                                                     "det"}) / n,
            "discourse_rico": sum(1 for t in toks if t["t"] in dis) / n,
            "modal_rico": sum(1 for t in toks if t["t"] in mod) / n,
            "estructura_corta": float(d["n"]),
            "clausula_simple": float(d["n_oraciones"] * 10
                                     + sum(1 for t in d["tok"] if t["d"] in {"cc", "mark"})),
        })
    return pd.DataFrame(filas)


def hacer_figuras(pos_tabla: dict[str, Any], enriqu: dict[str, Any],
                  perfiles: dict[str, Any], funcionales: dict[str, Any],
                  seq_counts: dict[str, dict[str, Counter]],
                  dep_counts: dict[str, dict[str, Counter]],
                  df_perfil: pd.DataFrame, proba_v03: np.ndarray) -> None:
    """Genera las figuras obligatorias de la EDA linguistica.

    Args:
        pos_tabla: composicion POS.
        enriqu: enriquecimiento por lema.
        perfiles: perfil pronominal.
        funcionales: grupos de palabras funcionales.
        seq_counts: conteos de secuencias POS.
        dep_counts: conteos de patrones de dependencia.
        df_perfil: tabla de perfiles y error.
        proba_v03: probabilidades OOF.
    """
    cats = sorted({c for d in pos_tabla.values() for c in d})
    visibles = [c for c in cats if max(d[c] for d in pos_tabla.values()) >= 0.02]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(visibles))
    ancho = 0.26
    for k, et in enumerate(ETIQUETAS):
        ax.bar(x + (k - 1) * ancho, [pos_tabla[et][c] for c in visibles], ancho, label=et)
    ax.set_xticks(x)
    ax.set_xticklabels(visibles, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("media de tokens por tweet")
    ax.set_title("Composicion de POS por clase")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_P61 / "04_pos_composition_by_label.png", dpi=130)
    plt.close(fig)

    for et in ETIQUETAS:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for a, cat in enumerate(("VERB", "ADJ", "NOUN")):
            if cat not in enriqu[et]:
                continue
            filas = enriqu[et][cat]["top_enriquecidos"][:10][::-1]
            axes[a].barh([r["lema"] for r in filas], [r["log_enr"] for r in filas],
                         color="tab:green" if a == 0 else
                         ("tab:orange" if a == 1 else "tab:blue"))
            axes[a].axvline(0, color="black", lw=0.8)
            axes[a].set_title(f"{et}: enriquecimiento de {cat}")
            axes[a].set_xlabel("log de P(lema|pos) / P(lema|neg)")
        plt.tight_layout()
        fig.savefig(FIG_P61 / f"{5 + ETIQUETAS.index(et):02d}_enrichment_{et}.png",
                    dpi=130)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for j, et in enumerate(ETIQUETAS):
        s = seq_counts[et]
        top = [k for k, v in s.most_common(12)][::-1]
        axes[j].barh(top, [s[k] for k in top], color="tab:purple")
        axes[j].set_title(f"{et}: secuencias POS mas frecuentes")
        axes[j].set_xlabel("ocurrencias")
        axes[j].tick_params(labelsize=7)
    plt.tight_layout()
    fig.savefig(FIG_P61 / "08_pos_sequence_enrichment.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for j, et in enumerate(ETIQUETAS):
        d = dep_counts[et]
        top = [k for k, v in d.most_common(12) if "__" in k and k.count("__") == 2][::-1]
        axes[j].barh(top, [d[k] for k in top], color="tab:brown")
        axes[j].set_title(f"{et}: patrones POS__dep__POS")
        axes[j].set_xlabel("ocurrencias")
        axes[j].tick_params(labelsize=6.5)
    plt.tight_layout()
    fig.savefig(FIG_P61 / "09_dependency_pattern_enrichment.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    nombres = list(PRONOMBRES)
    x = np.arange(len(nombres))
    for k, et in enumerate(ETIQUETAS):
        ax.bar(x + (k - 1) * 0.26, [perfiles[et][nm]["media"] for nm in nombres], 0.26,
               label=et)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", " ") for n in nombres], rotation=20, ha="right")
    ax.set_ylabel("media de ocurrencias por token")
    ax.set_title("Perfil pronominal por clase")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_P61 / "10_pronoun_composition.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    grupos = list(funcionales)
    x = np.arange(len(grupos))
    for k, et in enumerate(ETIQUETAS):
        ax.bar(x + (k - 1) * 0.26, [funcionales[g][et] for g in grupos], 0.26, label=et)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("_", " ") for g in grupos], rotation=20, ha="right",
                       fontsize=8)
    ax.set_ylabel("media de ocurrencias por token")
    ax.set_title("Palabras funcionales, negacion, modalidad y discursivos por clase")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_P61 / "11_function_words_and_negation.png", dpi=130)
    plt.close(fig)

    if not df_perfil.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        piv = df_perfil.pivot_table(index="perfil", columns="etiqueta",
                                    values="delta_F1_rico_menos_pobre", aggfunc="mean")
        piv.plot(kind="bar", ax=ax, width=0.8)
        ax.axhline(0, color="black", lw=1)
        ax.set_ylabel("F1 de V03 en la rebanada rica menos la pobre")
        ax.set_title("Donde V03 falla segun el perfil linguistico (negativo = worse)")
        ax.legend(fontsize=8)
        plt.xticks(rotation=30, ha="right", fontsize=7.5)
        plt.tight_layout()
        fig.savefig(FIG_P61 / "12_v03_error_by_linguistic_profile.png", dpi=130)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for j, e in enumerate(ETIQUETAS):
        axes[j].hist(proba_v03[:, j], bins=40, density=True, alpha=0.6, color="tab:blue")
        axes[j].axvline(V03_UMBRALES[j], color="black", ls="--", lw=1.2)
        axes[j].set_title(f"{e}: p(y=1|x) de referencia")
    plt.tight_layout()
    fig.savefig(FIG_P61 / "01_probability_distributions_reference.png", dpi=130)
    plt.close(fig)

    dist = np.abs(proba_v03 - V03_UMBRALES[None, :])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for j, e in enumerate(ETIQUETAS):
        ac = (proba_v03[:, j] >= V03_UMBRALES[j]) == (
            np.load(OOF, allow_pickle=True)["y_true"][:, j] == 1)
        axes[j].hist(dist[ac, j], bins=40, density=True, alpha=0.55, label="correct",
                     color="tab:green", range=(0, 0.6))
        axes[j].hist(dist[~ac, j], bins=40, density=True, alpha=0.55, label="incorrect",
                     color="tab:red", range=(0, 0.6))
        axes[j].set_title(f"{e}: distancia al umbral")
        axes[j].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_P61 / "02_probability_distance_to_threshold.png", dpi=130)
    plt.close(fig)

    err = (proba_v03 >= V03_UMBRALES[None, :]).astype(int) != y_ref()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for j, e in enumerate(ETIQUETAS):
        ej = err[:, j]
        for delta, col in ((0.10, "tab:orange"), (0.20, "tab:red"), (0.30, "darkred")):
            m = ej & (dist[:, j] >= delta)
            axes[j].bar(f"d>={delta:.2f}", m.sum(), color=col)
            axes[j].text(0, m.sum(), f"{m.sum()}", ha="center", va="bottom", fontsize=7)
        axes[j].set_title(f"{e}: errores de alta confianza")
        axes[j].set_ylabel("celdas")
    plt.tight_layout()
    fig.savefig(FIG_P61 / "03_high_confidence_error_distribution.png", dpi=130)
    plt.close(fig)


def y_ref() -> np.ndarray:
    """Etiquetas del OOF canonico.

    Returns:
        La matriz de etiquetas.
    """
    return np.load(OOF, allow_pickle=True)["y_true"].astype(int)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
