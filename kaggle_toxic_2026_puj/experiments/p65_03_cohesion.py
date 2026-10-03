"""P65 track P14 and P15: within-document lexical cohesion, first as a diagnostic.

Why this family is genuinely open, after the near misses
------------------------------------------------------
Two things looked like antecedents and are not. V10-07 measures covariate shift BETWEEN the
training and the test population, which is a different object from cohesion WITHIN a document.
And 16_targeting_v2 measures token distances between a target and an insult using an induced
lexicon, which is a distance between two semantic categories, not a sparse pair of arbitrary
words. Neither of them measures how much a document repeats itself.

The question, and it is not the question of P63 or P64
-----------------------------------------------------
P63 measured the composition of a document as RATIOS of morphological features, and found that
the block beat a random control by 0,1055 in isolation. P64 measured how the model's evidence is
distributed INSIDE a document, and found that the internal conflict doubles in the rows V03
gets wrong. Neither of those asks how much the document's own vocabulary repeats.

A long tweet that restates itself, quotes itself, or pivots to a new topic has a different
shape from one that develops a single thread, and a bag of n-grams cannot see the difference:
it sees the words twice in both cases. Repetition and overlap are properties of the DOCUMENT,
not of any token, which is exactly the kind of thing this project keeps finding and not being
able to use.

Why the diagnostic comes before the features, without exception
--------------------------------------------------------------
The brief is explicit that this is a diagnostic first and never automatically a feature, and
this script obeys that. Section one measures six quantities and reports them across the strata
that have mattered since P63. Section two only builds the feature block, and it builds it from
the SAME six quantities that were already computed and reported, so nothing new is invented
after seeing the diagnostic.

The reference the diagnostic is compared against
-----------------------------------------------
A structural descriptor is worthless if it only measures length, and P64-02 already proved how
that happens: the segment profile reached AUC 0,589847 against a token-count reference of
0,600506, and it had looked like signal until the reference was computed. So every cohesion
descriptor is scored against the length reference here too, before anything is built.

The controls
------------
A random block of identical width for capacity, and V03 plus its own margins. Both are
mandatory, and the margin one has decided every line since P63.

Usage
-----
    /home/carlos/venv/bin/python experiments/p65_03_cohesion.py
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
from sklearn.metrics import f1_score, roc_auc_score

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
MD_CO = REPORTS / "p65_cohesion.md"
JSON_CO = REPORTS / "p65_cohesion.json"
CSV_CO = REPORTS / "p65_cohesion.csv"

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
RE_TOKEN = re.compile(r"(?u)\b\w\w+\b")
RASGOS = ("coseno_medio", "coseno_minimo", "coseno_desviacion", "jaccard_medio",
          "repeticion_cruzada", "cambio_lexico")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P65-03] {mensaje}", flush=True)


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


def cohesion(segmentos: list[str]) -> dict[str, float]:
    """Seis cantidades de cohesion documental, todas acotadas.

    Los cosenos y los jaccard se calculan sobre vectores de conteo dentro del documento, sin
   TF-IDF global y sin ningun modelo, de modo que la cantidad mide el documento y no el
    corpus. Un documento de un solo segmento tiene por definicion los seis valores neutros, y
    no se le inventa una cohesion que no puede tener.

    Args:
        segmentos: segmentos del documento.

    Returns:
        Diccionario con las seis cantidades.
    """
    conjuntos = [set(RE_TOKEN.findall(s.lower())) for s in segmentos]
    conjuntos = [v for v in conjuntos if v]
    total = sum(len(v) for v in conjuntos)
    repeticion = 1.0 - (len(set().union(*conjuntos)) / total) if total > 0 else 0.0
    if len(conjuntos) < 2:
        return {"coseno_medio": 1.0, "coseno_minimo": 1.0, "coseno_desviacion": 0.0,
                "jaccard_medio": 1.0, "repeticion_cruzada": repeticion,
                "cambio_lexico": 0.0}
    cosenos: list[float] = []
    jaccards: list[float] = []
    for a in range(len(conjuntos)):
        for b in range(a + 1, len(conjuntos)):
            u, s = conjuntos[a], conjuntos[b]
            inter = len(u & s)
            jaccards.append(inter / max(len(u | s), 1))
            na = np.array([1.0 if t in u else 0.0 for t in u | s])
            nb = np.array([1.0 if t in s else 0.0 for t in u | s])
            den = float(np.sqrt((na * na).sum()) * np.sqrt((nb * nb).sum()))
            cosenos.append(float((na * nb).sum() / den) if den > 0 else 0.0)
    return {
        "coseno_medio": float(np.mean(cosenos)),
        "coseno_minimo": float(np.min(cosenos)),
        "coseno_desviacion": float(np.std(cosenos)),
        "jaccard_medio": float(np.mean(jaccards)),
        "repeticion_cruzada": repeticion,
        "cambio_lexico": 1.0 - float(np.mean(cosenos)),
    }


def main() -> None:
    """Mide la cohesion documental como diagnostico y, solo despues, como bloque.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P65-03 cohesion documental interna, diagnostico antes que bloque")

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
    n_tok = np.array([len(t.split()) for t in textos])
    error = pred_v03 != y
    alto = error & (np.abs(proba_v03 - V03_UMBRALES[None, :]) >= DELTA_HC)
    log(f"   OOF canonico {ccv_ref!r} | {n} filas")

    segmentos = [segmentar(t) for t in textos]
    n_seg = np.array([len(s) for s in segmentos])
    filas_coh = [cohesion(s) for s in segmentos]
    bloque = np.array([[f[r] for r in RASGOS] for f in filas_coh], dtype=float)
    log("")
    log("   LA EDA DE COHESION, que es un DIAGNOSTICO y no un experimento")
    log(f"      documentos con dos o mas segmentos: {int((n_seg >= 2).sum())} de {n}")
    log("      un documento de un solo segmento tiene los seis valores neutros por")
    log("      definicion, y no se le inventa una cohesion que no puede tener")

    estratos: list[tuple[str, np.ndarray]] = [("todas", np.ones(n, dtype=bool))]
    for nombre, lo, hi in BINS:
        estratos.append((nombre, (n_tok >= lo) & (n_tok <= hi)))
    estratos.append(("V03 correcto", ~error.any(axis=1)))
    estratos.append(("V03 incorrecto", error.any(axis=1)))
    estratos.append(("error de alta confianza", alto.any(axis=1)))
    multi = n_seg >= 2
    estratos.append(("solo multi segmento", multi))
    log("")
    log(f"      {'estrato':>26}{'n':>7}" + "".join(f"{r[:15]:>17}" for r in RASGOS))
    filas_eda: list[dict[str, Any]] = []
    for nombre, m in estratos:
        if m.sum() < 5:
            continue
        vals = bloque[m].mean(axis=0)
        filas_eda.append({"estrato": nombre, "n": int(m.sum())}
                         | {r: round(float(x), 6) for r, x in zip(RASGOS, vals)})
        log(f"      {nombre:>26}{int(m.sum()):>7}"
            + "".join(f"{x:>17.4f}" for x in vals))

    # ---------- la referencia de longitud, que es la que evita el falso positivo ----------
    objetivo_err = error.any(axis=1).astype(int)
    aucs: dict[str, Any] = {}
    for j, r in enumerate(RASGOS):
        if objetivo_err.sum() < 10:
            continue
        aucs[r] = round(float(roc_auc_score(objetivo_err, bloque[:, j])), 6)
    aucs["n_tokens_como_referencia"] = round(float(roc_auc_score(objetivo_err, n_tok)), 6)
    log("")
    log("   LA SEPARACION CONTRA LA REFERENCIA TRIVIAL DE LONGITUD")
    log("   un descriptor estructural que solo mide longitud no aporta nada, y P64-02 ya")
    log("   demostro ese exacto falso positivo")
    ref = aucs["n_tokens_como_referencia"]
    for r in RASGOS:
        log(f"      {r:22} AUC {aucs[r]:.6f}")
    log(f"      {'n_tokens, REFERENCIA':22} AUC {ref:.6f}")
    mejor = max((r for r in RASGOS), key=lambda r: abs(aucs[r] - 0.5))
    ventaja = abs(aucs[mejor] - 0.5) - abs(ref - 0.5)
    log(f"      mejor rasgo: {mejor}, ventaja sobre la referencia {ventaja:+.6f}")

    # ---------- el bloque ----------
    if ventaja > 0.01:
        log("")
        log(f"   DECISION DEL DIAGNOSTICO: el descriptor GANA a la referencia de longitud "
            f"({ventaja:+.6f}),")
        log("   luego la cohesion es un rasgo con senal propia y se construye el bloque")
    else:
        log("")
        log(f"   DECISION DEL DIAGNOSTICO: el descriptor NO gana a la referencia de longitud "
            f"({ventaja:+.6f}),")
        log("   la cohesion interna se comporta como la longitud con otro disfraz")
    log("")
    log("   EL BLOQUE, con los dos controles")

    proba: dict[str, np.ndarray] = {b: np.zeros((n, 3)) for b in
                                   ("V03_cohesion", "V03_aleatorio", "V03_margen")}
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
            cuadrados = np.asarray(mm.multiply(mm).sum(axis=1)).ravel()
            media = float(np.sqrt(cuadrados).mean())
            return mm if media <= 0 else mm.multiply(ref / media)

        rng = np.random.default_rng(SEMILLA + f)
        matrices = {"V03_cohesion": bloque,
                    "V03_aleatorio": rng.standard_normal(bloque.shape),
                    "V03_margen": margen_v03}
        for brazo, be in matrices.items():
            sd = be[ir].std(axis=0)
            sd[sd < 1e-8] = 1.0
            z = (be - be[ir].mean(axis=0)) / sd[None, :]
            norma = float(np.sqrt(np.asarray((z[ir] ** 2).sum(axis=1))).mean())
            z = z / norma if norma > 0 else z
            xa = escalar(hstack([base, csr_matrix(z[ir])]).tocsr())
            xav = escalar(hstack([base_v, csr_matrix(z[iva])]).tocsr())
            for j in range(3):
                mo = LogisticRegression(**LR)
                mo.fit(xa, y[ir, j])
                proba[brazo][iva, j] = mo.predict_proba(xav)[:, 1]
        log(f"      pliegue {f} listo")

    folds_v = [float(f1_score(y[pliegues == g], pred_v03[pliegues == g], average="macro"))
               for g in range(1, 6)]
    detalle: dict[str, Any] = {}
    filas: list[dict[str, Any]] = []
    log("")
    log(f"      {'brazo':16}{'corrected_cv':>18}{'delta vs V03':>16}{'pliegues':>10}{'neto':>8}")
    for brazo in ("V03_cohesion", "V03_aleatorio", "V03_margen"):
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
        log(f"      {brazo:16}{ccv:>18.10f}{ccv - ccv_ref:>+16.10f}{pl:>7d}/5"
            f"{detalle[brazo]['rescate'] - detalle[brazo]['damage']:>8d}")

    brecha_cap = detalle["V03_cohesion"]["corrected_cv"] - detalle["V03_aleatorio"]["corrected_cv"]
    brecha_marg = detalle["V03_cohesion"]["corrected_cv"] - detalle["V03_margen"]["corrected_cv"]
    d_glob = detalle["V03_cohesion"]["corrected_cv"] - ccv_ref
    pl = next(f["pliegues_mejorados"] for f in filas if f["brazo"] == "V03_cohesion")
    log("")
    log("   LAS BRECHAS")
    log(f"      contra el control de anchura: {brecha_cap:+.10f}")
    log(f"      contra el margen de V03:      {brecha_marg:+.10f}")
    if d_glob >= GATE_DELTA and pl >= GATE_PLIEGUES and brecha_cap > 0 and brecha_marg > 0:
        lectura, decision = "PROMISING", "PROMISING"
        veredicto = (f"la cohesion aporta {d_glob:+.10f} con {pl} de 5 y supera ambos "
                     f"controles, luego exige bootstrap y segunda particion")
    elif brecha_cap > 0 and brecha_marg > 0:
        lectura, decision = "INFORMACION_SI_INTEGRACION_NO", "COMPLEMENTARY"
        veredicto = (
            f"la cohesion documental lleva informacion que no es capacidad ni confianza: "
            f"{brecha_cap:+.10f} contra el control de anchura y {brecha_marg:+.10f} contra el "
            f"margen. Y el global es {d_glob:+.10f} con {pl} de 5, luego no entra. Quinto caso "
            f"del mismo patron, y esta vez la lectura es mas nitida porque el diagnostico y el "
            f"bloque usan LAS MISMAS seis cantidades: no hay ninguna representacion nueva que "
            f"descartar, solo una que no cabe")
    else:
        lectura, decision = "SIN_APORTE", "CLOSED"
        veredicto = (
            f"la cohesion documental no aporta: {brecha_cap:+.10f} contra el control de "
            f"anchura y {brecha_marg:+.10f} contra el margen. Y el diagnostico lo dice antes "
            f"de hacerlo: el mejor descriptor, {mejor}, da {aucs[mejor]:.6f} frente a "
            f"{ref:.6f} de contar tokens, luego la cohesion interna se comporta como la "
            f"longitud con otro disfraz, igual que el perfil segmental de P64-02")
    log("")
    log(f"   LECTURA: {lectura}")
    log(f"   VEREDICTO: {veredicto}")
    log(f"   DECISION: {decision}")

    with CSV_CO.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["tipo", "clave", "n"] + list(RASGOS))
        for f in filas_eda:
            w.writerow(["eda", f["estrato"], f["n"]] + [f[r] for r in RASGOS])
        for r in RASGOS:
            w.writerow(["auc_contra_el_error", r, "", aucs[r]])
        w.writerow(["auc_contra_el_error", "n_tokens_REFERENCIA", "", ref])
        w.writerow(["brecha", "contra_anchura", "", round(brecha_cap, 10)])
        w.writerow(["brecha", "contra_margen", "", round(brecha_marg, 10)])
        w.writerow([])
        w.writerow(["brazo", "corrected_cv", "delta_vs_V03", "pliegues_mejorados", "rescate",
                    "damage", "neto"])
        for f in filas:
            w.writerow([f["brazo"], f["corrected_cv"], f["delta_vs_V03"],
                        f["pliegues_mejorados"], f["rescate"], f["damage"],
                        f["rescate"] - f["damage"]])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P65_COHESION",
        "naturaleza": ("DIAGNOSTICO Y EXPERIMENTO. 1 candidato y 2 controles, 0 submissions, "
                       "0 Kaggle, 0 holdout, 0 test"),
        "fecha": inicio.isoformat(timespec="seconds"), "experiment_id": "P65-03",
        "track": "P14-P15-cohesion-documental", "script": "experiments/p65_03_cohesion.py",
        "pregunta": ("los tweets largos que V03 no resuelve se distinguen por repetir su propia "
                     "evidencia o por cambiar de vocabulario a mitad"),
        "por_que_es_nueva": (
            "V10-07 es shift de covariables ENTRE tren y prueba, que es otro objeto, y "
            "16_targeting_v2 mide distancias entre un target y un insulto con un lexico "
            "inducido, que es una distancia entre categorias semanticas y no un par lexico de "
            "anchura desconocida. Ninguno de los dos mide cuanto se repite un documento"),
        "rasgos": list(RASGOS),
        "definiciones": {
            "coseno_medio": "coseno medio entre todas las parejas de segmentos, sobre vectores "
                            "de conteo dentro del documento",
            "coseno_minimo": "el peor par",
            "coseno_desviacion": "dispersion de esos cosenos, que es cuanto varia la cohesion",
            "jaccard_medio": "solapamiento de vocabulario medio entre pares de segmentos",
            "repeticion_cruzada": "uno menos el cociente de tipos distintos sobre ocurrencias",
            "cambio_lexico": "uno menos el coseno medio, que es la lectura directa de la "
                             "hipotesis de cambio de tema",
        },
        "documentos_de_un_segmento": ("tienen los seis valores neutros por definicion y no se "
                                      "les inventa una cohesion que no pueden tener"),
        "eda_por_estrato": filas_eda,
        "auc_contra_el_error": aucs,
        "referencia_de_longitud": ref,
        "mejor_rasgo": mejor,
        "ventaja_sobre_la_referencia": round(ventaja, 6),
        "el_bloque_usa_las_mismas_cantidades": True,
        "resultados": detalle,
        "brechas": {"contra_anchura": round(brecha_cap, 10), "contra_margen": round(brecha_marg,
                                                                                     10)},
        "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                 "delta": round(d_glob, 10), "pliegues_mejorados": pl},
        "lectura": lectura, "veredicto": veredicto, "decision": decision,
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_CO.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P65 - Cohesion documental interna")
    W("")
    W(f"Generado por `experiments/p65_03_cohesion.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Por que esta familia estaba abierta")
    W("")
    W("Dos cosas parecian antecedentes y no lo son. **V10-07** mide shift de covariables")
    W("**entre** entrenamiento y prueba, que es otro objeto que la cohesion **dentro** del")
    W("documento. Y **16_targeting_v2** mide distancias entre un target y un insulto con un")
    W("lexico inducido, que es una distancia entre categorias semanticas y no un par lexico de")
    W("anchura desconocida. Ninguno de los dos mide cuanto se repite un documento.")
    W("")
    W("La pregunta, que no es la de P63 ni la de P64")
    W("-----------------------------------------------")
    W("P63 midio la composicion como **ratios** morfologicos, y el bloque batio a un control")
    W("aleatorio en 0,1055 aislado. P64 midio como se **distribuye la evidencia** dentro del")
    W("documento, y el conflicto interno se duplica en las filas que V03 falla. Ninguna de las")
    W("dos pregunta cuanto se repite el vocabulario del propio documento.")
    W("")
    W("Un tweet largo que se repite, se cita o pivota a otro tema tiene otra forma que uno que")
    W("desarrolla un solo hilo, y una bolsa de n-gramas no ve la diferencia: ve las palabras dos")
    W("veces en los dos casos. La repeticion y el solapamiento son propiedades del DOCUMENTO y no")
    W("de ningun token, que es exactamente el tipo de cosa que este proyecto lleva dos fases")
    W("encontrando y sin poder usar.")
    W("")
    W("## El diagnostico va antes que las features, sin excepcion")
    W("")
    W(f"Documentos con dos o mas segmentos: **{int((n_seg >= 2).sum())} de {n}**. Un documento de")
    W("un solo segmento tiene los seis valores neutros por definicion y no se le inventa una")
    W("cohesion que no puede tener.")
    W("")
    W("| estrato | n | " + " | ".join(f"`{r}`" for r in RASGOS) + " |")
    W("|---|---:|" + "---:|" * len(RASGOS))
    for f in filas_eda:
        W(f"| {f['estrato']} | {f['n']} | "
          + " | ".join(f"{f[r]:.4f}" for r in RASGOS) + " |")
    W("")
    W("## La referencia de longitud, que es la que evita el falso positivo")
    W("")
    W("Un descriptor estructural no vale nada si solo mide longitud, y **P64-02 ya demostro**")
    W("como pasa eso: el perfil segmental llego a AUC 0,589847 frente a una referencia de")
    W("contar tokens de 0,600506, y habia parecido senal hasta que se computo la referencia.")
    W("")
    W("| rasgo | AUC para predecir si V03 falla |")
    W("|---|---:|")
    for r in RASGOS:
        W(f"| {r} | {aucs[r]:.6f} |")
    W(f"| **n_tokens, REFERENCIA TRIVIAL** | {ref:.6f} |")
    W("")
    W(f"Mejor rasgo: **{mejor}**, ventaja sobre la referencia **{ventaja:+.6f}**.")
    W("")
    W("## El bloque, con los dos controles")
    W("")
    W("| brazo | corrected_cv | delta vs V03 | pliegues | neto |")
    W("|---|---:|---:|---:|---:|")
    for f in filas:
        W(f"| {f['brazo']} | {f['corrected_cv']:.10f} | {f['delta_vs_V03']:+.10f} | "
          f"{f['pliegues_mejorados']}/5 | {f['rescate'] - f['damage']:+d} |")
    W("")
    W("```text")
    W(f"contra el control de anchura: {brecha_cap:+.10f}")
    W(f"contra el margen de V03:      {brecha_marg:+.10f}")
    W("```")
    W("")
    W("Y el bloque usa **las mismas seis cantidades** que el diagnostico, de modo que no hay")
    W("ninguna representacion nueva que descartar despues de mirar: hay una que ya se midio y")
    W("que no cabe.")
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
    W("*Generado por `experiments/p65_03_cohesion.py`. Sin push.*")
    MD_CO.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_CO, JSON_CO, CSV_CO):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P65-03 OK | mejor rasgo {mejor} {aucs[mejor]:.6f} contra referencia {ref:.6f} "
        f"({ventaja:+.6f}) | bloque {d_glob:+.10f} | decision {decision}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
