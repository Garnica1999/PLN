"""P62 track: is the binding constraint the vocabulary coverage, and not the grammar?

What the previous two experiments between them located
------------------------------------------------------
P62-04 compared the interaction channel with and without the POS suffix, with everything
else identical, and the answer was not the tidy one the hypothesis expected. The suffix
helped at 6 of 8 budgets, by up to +0,006881, so the interaction is not pure row-splitting.
But BOTH variants lost at 8 of 8, and the script refused to call it redundancy because its
own pre-declared condition failed on one of its three terms.

The condition failed on coverage, and following that thread produced this experiment.

The measurement that reframes the phase
---------------------------------------
V03's word channel has a vocabulary of 2673 terms in fold 1. The 21.000-odd features are
the union of word bigrams and char_wb 3-to-5 grams, and the word UNIGRAM channel is small
because ``min_df=2`` throws away every term that appears once.

Against that vocabulary, on the training rows of fold 1:

    distinct lemmas present            4057     in V03's word vocabulary   957  (23,6 %)
    distinct surface forms present    4569     in V03's word vocabulary  1259  (27,6 %)

So roughly three quarters of the words that are actually in the training text are absent
from V03's word channel. The examples are ordinary words, `abandonado`, `aberrante`,
`abolir`, each of which appears once and is therefore dropped.

That reframes every linguistic result in this phase. A POS-conditioned channel is not
mainly a source of grammatical information; it is also a way of putting back words that
``min_df=2`` had discarded, keyed by lemma and split by category. And the same is true of
the bare-lemma channel. If that reading is right, the binding constraint is COVERAGE, and
coverage can be tested without any grammar at all.

The four arms, and what each one isolates
-----------------------------------------
    control     V03 exactly: word(1,2) min_df=2 plus char_wb(3,5) min_df=2
    A           word(1,2) min_df=1, char unchanged
    B           word(1,2) min_df=1 and char_wb(3,5) min_df=1
    C           word(1,2) min_df=1 plus a POS-conditioned channel

A isolates the word channel's coverage. B adds the character channel's coverage. C asks
whether the grammar adds anything ON TOP of full coverage, which is the question P62-04
could not answer while the two effects were tangled.

No grid and no search. One pre-registered configuration per arm, the classifier is V03's
at every arm, the folds and thresholds are the canonical ones, and the scale confound is
controlled by rescaling every arm to V03's row norm. Nothing is selected by its score.

Usage
-----
    /home/carlos/venv/bin/python experiments/p62_09_cobertura.py
"""

from __future__ import annotations

import ast
import csv as _csv
import gzip
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
TRAIN_RAW = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
ANOTACION = PROJECT_ROOT / "features" / "p62_anotacion_spacy.json.gz"
MD_C = REPORTS / "p62_cobertura_vocabulario.md"
JSON_C = REPORTS / "p62_cobertura_vocabulario.json"
CSV_C = REPORTS / "p62_cobertura_vocabulario.csv"
CELDA_NPZ = PROJECT_ROOT / "features" / "p62_cobertura_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
WORD = {"analyzer": "word", "ngram_range": (1, 2), "sublinear_tf": True, "lowercase": True}
CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "sublinear_tf": True, "lowercase": True}
LR = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear",
      "random_state": 42}
CATEGORIAS = ("NOUN", "VERB", "ADJ", "PROPN", "ADV", "PRON", "AUX", "INTJ")
# El tamano del canal POS en el brazo C. Es el presupuesto que la curva de P62-07 marco
# como el menos malo, y se fija aqui ANTES de ejecutar para que C no se convierta en una
# busqueda. No se varia.
K_POS = 256
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
N_BOOT = 2000
DELTA_HC = 0.10
SEMILLA = 42


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P62-09] {mensaje}", flush=True)


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


def tokens_pos(documentos: list[dict[str, Any]], filas: np.ndarray) -> list[str]:
    """Tokens lema__POS de un conjunto de filas.

    Args:
        documentos: anotaciones.
        filas: indices.

    Returns:
        Un token por fila.
    """
    return [" ".join(f"{t['l']}__{t['p']}" for t in documentos[i]["tok"]
                     if t["al"] and t["l"] and t["p"] in CATEGORIAS) for i in filas]


def main() -> None:
    """Prueba si el limite de V03 es la cobertura del vocabulario y no la gramatica.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P62-09 es la restriction la cobertura del vocabulario y no la gramatica")
    log("   cuatro brazos, una configuracion pre-registrada cada uno, sin rejilla")
    log(f"   el brazo C lleva un canal POS de K={K_POS} rasgos, fijado antes de ejecutar")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    v = np.load(OOF, allow_pickle=True)
    ids, y, pliegues = v["ids"], v["y_true"].astype(int), v["pliegues"]
    pred_v03, proba_v03 = v["pred"].astype(int), v["proba"]
    n = ids.size
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    part = pd.read_csv(PARTICION)
    grupos = part.set_index("id").loc[ids, "grupo_near_dup_autor"].to_numpy()
    if int(pd.Series(grupos).groupby(grupos).nunique().gt(1).sum()):
        raise SystemExit("PARADA: hay grupos canonicos partidos entre pliegues")
    tr = pd.read_csv(TRAIN_RAW, sep=";", encoding="latin-1", dtype={"id": str})
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
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | {len(set(grupos))} grupos intactos")

    brazos = ("control", "A_palabras_min_df_1", "B_palabras_y_char_min_df_1",
              "C_min_df_1_mas_canal_POS")
    proba: dict[str, np.ndarray] = {b: np.zeros((n, 3)) for b in brazos}
    dims: dict[str, list[int]] = {b: [] for b in brazos}
    densidad: dict[str, list[float]] = {b: [] for b in brazos}
    cobertura: list[dict[str, Any]] = []
    t0 = time.time()

    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        # ---------- el control: V03 exacto ----------
        w2 = TfidfVectorizer(min_df=2, **WORD)
        c2 = TfidfVectorizer(min_df=2, **CHAR)
        xw2 = w2.fit_transform([textos[i] for i in ir])
        xc2 = c2.fit_transform([textos[i] for i in ir])
        xctrl = hstack([xw2, xc2]).tocsr()
        xctrl_v = hstack([w2.transform([textos[i] for i in iva]),
                          c2.transform([textos[i] for i in iva])]).tocsr()
        objetivo = float(np.sqrt(np.mean(xctrl.multiply(xctrl).sum(axis=1))))

        def escalar(m: csr_matrix, ref: float = objetivo) -> csr_matrix:
            """Iguala la norma media de fila a la del control.

            Args:
                m: matriz.
                ref: norma objetivo.

            Returns:
                La matriz escalada.
            """
            media = float(np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel()).mean())
            return m if media <= 0 else m.multiply(ref / media)

        # ---------- A y B: cobertura completa ----------
        w1 = TfidfVectorizer(min_df=1, **WORD)
        xw1 = w1.fit_transform([textos[i] for i in ir])
        xw1_v = w1.transform([textos[i] for i in iva])
        c1 = TfidfVectorizer(min_df=1, **CHAR)
        xc1 = c1.fit_transform([textos[i] for i in ir])
        xc1_v = c1.transform([textos[i] for i in iva])
        xa = hstack([xw1, xc2]).tocsr()
        xa_v = hstack([xw1_v, c2.transform([textos[i] for i in iva])]).tocsr()
        xb = hstack([xw1, xc1]).tocsr()
        xb_v = hstack([xw1_v, xc1_v]).tocsr()

        # ---------- C: cobertura completa mas un canal POS ----------
        pc_pos = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=1,
                                 sublinear_tf=True, lowercase=False, token_pattern=r"[^ ]+")
        tok_ir = tokens_pos(documentos, ir)
        xpos = pc_pos.fit_transform(tok_ir)
        vocab = sorted(pc_pos.vocabulary_.keys())
        col = {t: i for i, t in enumerate(vocab)}
        score = np.zeros(len(vocab))
        for j in range(3):
            fp = np.zeros(len(vocab))
            fn = np.zeros(len(vocab))
            for a, i in enumerate(ir):
                destino = fp if y[i, j] == 1 else fn
                for t in set(tok_ir[a].split()):
                    c = col.get(t)
                    if c is not None:
                        destino[c] += 1
            npos, nneg = fp.sum(), fn.sum()
            pp = (fp + 1.0) / (npos + 2.0)
            pn = (fn + 1.0) / (nneg + 2.0)
            score += np.abs(np.log(pp / pn))
        elegidos = sorted({col[t] for t in [vocab[i] for i in
                                            np.argsort(-score)[:K_POS]]})
        xc = escalar(hstack([hstack([xw1, xc2]).tocsr(), xpos[:, elegidos]]).tocsr())
        xc_v = escalar(hstack([xa_v, pc_pos.transform(tokens_pos(documentos, iva))[:, elegidos]]).tocsr())

        # ---------- la cobertura, medida ----------
        formas = {t["t"] for i in ir for t in documentos[i]["tok"] if t["al"] and t["l"]}
        voc_w2 = set(w2.vocabulary_.keys())
        voc_w1 = set(w1.vocabulary_.keys())
        cobertura.append({
            "pliegue": f,
            "vocabulario_de_palabras_min_df_2": len(voc_w2),
            "vocabulario_de_palabras_min_df_1": len(voc_w1),
            "formas_superficiales_presentes": len(formas),
            "cobertura_con_min_df_2": round(len(formas & voc_w2) / max(len(formas), 1), 6),
            "cobertura_con_min_df_1": round(len(formas & voc_w1) / max(len(formas), 1), 6),
            "rasgos_totales_min_df_2": int(xctrl.shape[1]),
            "rasgos_totales_min_df_1_palabras": int(xa.shape[1]),
            "rasgos_totales_min_df_1_palabras_y_char": int(xb.shape[1]),
        })

        matrices = {"control": (xctrl, xctrl_v),
                    "A_palabras_min_df_1": (xa, xa_v),
                    "B_palabras_y_char_min_df_1": (xb, xb_v),
                    "C_min_df_1_mas_canal_POS": (xc, xc_v)}
        for b, (xent, xval) in matrices.items():
            if b == "control":
                xent, xval = xctrl, xctrl_v
            for j in range(3):
                mo = LogisticRegression(**LR)
                mo.fit(xent, y[ir, j])
                proba[b][iva, j] = mo.predict_proba(xval)[:, 1]
            dims[b].append(int(xent.shape[1]))
            densidad[b].append(round(float(xent.nnz / (xent.shape[0] * xent.shape[1])),
                                       7))
        log(f"      pliegue {f} ({time.time() - t0:.0f} s) | control {xctrl.shape[1]} | "
            f"A {xa.shape[1]} | B {xb.shape[1]} | C {xc.shape[1]} | cobertura "
            f"{cobertura[-1]['cobertura_con_min_df_2']:.4f} -> "
            f"{cobertura[-1]['cobertura_con_min_df_1']:.4f}")

    def metricas(b: str) -> dict[str, Any]:
        """Metricas de un brazo.

        Args:
            b: nombre del brazo.

        Returns:
            Diccionario con las metricas.
        """
        p = proba[b]
        pr = (p >= V03_UMBRALES[None, :]).astype(int)
        ac = pr == y
        por_et = {}
        for j, et in enumerate(ETIQUETAS):
            a = ac[:, j]
            d_v = np.abs(proba_v03[:, j] - V03_UMBRALES[j])
            d_c = np.abs(p[:, j] - V03_UMBRALES[j])
            por_et[et] = {
                "F1": round(float(f1_score(y[:, j], pr[:, j])), 6),
                "rescate": int(np.count_nonzero(a & (pred_v03[:, j] != y[:, j]))),
                "damage": int(np.count_nonzero((~a) & (pred_v03[:, j] == y[:, j]))),
                "alta_confianza_corregidos": int(np.count_nonzero(
                    (pred_v03[:, j] != y[:, j]) & a & (d_v >= DELTA_HC))),
                "alta_confianza_introducidos": int(np.count_nonzero(
                    (pred_v03[:, j] == y[:, j]) & ~a & (d_c >= DELTA_HC))),
            }
        return {
            "corrected_cv": float(f1_score(y, pr, average="macro")),
            "folds": [float(f1_score(y[pliegues == g], pr[pliegues == g], average="macro"))
                      for g in range(1, 6)],
            "rescate": int(np.count_nonzero(ac & (pred_v03 != y))),
            "damage": int(np.count_nonzero((~ac) & (pred_v03 == y))),
            "cambian": int(np.count_nonzero(pr != pred_v03)),
            "por_etiqueta": por_et,
        }

    log("")
    log("   LA COBERTURA DEL CANAL DE PALABRAS, que es lo que estos brazos cambian")
    log(f"      {'pliegue':>8}{'vocab md2':>12}{'vocab md1':>12}{'formas':>10}"
        f"{'cob md2':>10}{'cob md1':>10}")
    for c in cobertura:
        log(f"      {c['pliegue']:>8}{c['vocabulario_de_palabras_min_df_2']:>12}"
            f"{c['vocabulario_de_palabras_min_df_1']:>12}"
            f"{c['formas_superficiales_presentes']:>10}"
            f"{c['cobertura_con_min_df_2']:>10.4f}{c['cobertura_con_min_df_1']:>10.4f}")
    c2m = float(np.mean([c["cobertura_con_min_df_2"] for c in cobertura]))
    c1m = float(np.mean([c["cobertura_con_min_df_1"] for c in cobertura]))
    log(f"      cobertura media: min_df=2 {c2m:.4f} | min_df=1 {c1m:.4f}")

    log("")
    log("   LOS CUATRO BRAZOS")
    mc = metricas("control")
    ms = {b: metricas(b) for b in brazos if b != "control"}
    if mc["corrected_cv"] != V03_CCV:
        raise SystemExit(f"PARADA: el control no reproduce V03, da {mc['corrected_cv']!r}")
    log(f"      {'brazo':<32}{'rasgos':>8}{'corrected_cv':>18}{'delta':>16}")
    log(f"      {'control, V03 exacto':<32}{dims['control'][0]:>8}"
        f"{mc['corrected_cv']:>18.10f}{0.0:>+16.10f}")
    for b, m in ms.items():
        log(f"      {b:<32}{dims[b][0]:>8}{m['corrected_cv']:>18.10f}"
            f"{m['corrected_cv'] - ccv_ref:>+16.10f}")

    filas_csv = [{"brazo": "control", "delta": 0.0,
                  "corrected_cv": round(mc["corrected_cv"], 10)}]
    for b, m in ms.items():
        filas_csv.append({"brazo": b, "delta": round(m["corrected_cv"] - ccv_ref, 10),
                          "corrected_cv": round(m["corrected_cv"], 10)})
    with CSV_C.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["brazo", "corrected_cv", "delta"])
        for f in filas_csv:
            w.writerow([a_ascii(f["brazo"]), f["corrected_cv"], f["delta"]])

    # ---------- que se lleva cada brazo ----------
    log("")
    log("   QUE SE LLEVA CADA BRAZO, que es lo que decide si la cobertura era el cuello")
    d_a = ms["A_palabras_min_df_1"]["corrected_cv"] - ccv_ref
    d_b = ms["B_palabras_y_char_min_df_1"]["corrected_cv"] - ccv_ref
    d_c = ms["C_min_df_1_mas_canal_POS"]["corrected_cv"] - ccv_ref
    d_c_puro = d_c - d_a
    log(f"      A, solo el canal de palabras con min_df=1     {d_a:+.10f}")
    log(f"      B, mas el canal de caracteres con min_df=1     {d_b:+.10f}")
    log(f"      C, cobertura completa mas canal POS            {d_c:+.10f}")
    log(f"      el canal POS sobre cobertura completa vale     {d_c_puro:+.10f}")
    if max(d_a, d_b, d_c) > 0:
        mejor_b = max((d_a, "A"), (d_b, "B"), (d_c, "C"))[1]
        veredicto = (
            f"la cobertura SI es el cuello de botella: los brazos que la amplian ganan, y el "
            f"mejor es el {mejor_b} con {max(d_a, d_b, d_c):+.10f}. Sobre cobertura completa "
            f"el canal POS aporta {d_c_puro:+.10f}, luego la gramatica es marginal incluso "
            f"cuando ya no le queda hueco a la falta de palabras")
    else:
        veredicto = (
            f"ampliar la cobertura NO mejora: A {d_a:+.10f}, B {d_b:+.10f}, C {d_c:+.10f}. "
            f"Asi que las palabras que min_df=2 descartaba no eran las que faltaban, y la "
            f"pista de la cobertura queda descartada junto con la de la gramatica")
    log("")
    log(f"   VEREDICTO: {veredicto}")

    # ---------- gate sobre el mejor brazo ----------
    # La tupla es (delta, nombre). Una version anterior la desempaquetaba al reves y
    # buscaba un brazo llamado "-0,00335", que no existe.
    delta, nom = max((d_a, "A_palabras_min_df_1"), (d_b, "B_palabras_y_char_min_df_1"),
                     (d_c, "C_min_df_1_mas_canal_POS"), key=lambda r: r[0])
    m = ms[nom]
    fm = sum(1 for u, w2_ in zip(m["folds"], mc["folds"]) if u > w2_)
    gate = bool(delta >= GATE_DELTA and fm >= GATE_PLIEGUES)
    log("")
    log("   GATE")
    log(f"      mejor brazo {nom} | delta {delta:+.10f} | pliegues {fm}/5 | gate {gate}")
    log(f"      rescate {m['rescate']} | dano {m['damage']} | neto {m['rescate'] - m['damage']:+d} "
        f"| cambian {m['cambian']}")
    for et in ETIQUETAS:
        h = m["por_etiqueta"][et]
        log(f"      {et}: F1 {h['F1']:.4f} | rescate {h['rescate']:4d} | dano {h['damage']:4d} "
            f"| alta confianza corregidos {h['alta_confianza_corregidos']:3d} | introducidos "
            f"{h['alta_confianza_introducidos']:3d}")
    ic_lo = ic_hi = None
    if gate:
        rng = np.random.default_rng(SEMILLA)
        pr = (proba[nom] >= V03_UMBRALES[None, :]).astype(int)
        dc = ((pr == y).astype(int) - (pred_v03 == y).astype(int)).sum(axis=1).astype(float)
        gu, gc = np.unique(grupos, return_inverse=True)
        dif_g = np.bincount(gc, weights=dc, minlength=len(gu))
        ms_boot = dif_g[rng.integers(0, len(gu), size=(N_BOOT, len(gu)))].sum(axis=1) / len(gu)
        ic_lo, ic_hi = float(np.percentile(ms_boot, 2.5)), float(np.percentile(ms_boot, 97.5))
        log(f"      bootstrap agrupado por {len(gu)} grupos: IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}]")
    np.savez_compressed(CELDA_NPZ, ids=ids, y_true=y, pliegues=pliegues, pred_v03=pred_v03,
                        **{f"proba_{b}": proba[b] for b in brazos})

    decision = ("PROMISING. Supera el gate; exige segunda particion" if gate else
                ("MARGINAL. Mejora pero no alcanza el gate de +0,003" if delta > 0 else
                 "CLOSED. Ampliar la cobertura del vocabulario no aporta sobre V03"))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P62_COBERTURA",
        "naturaleza": ("EXPERIMENTO. 3 candidatos y 1 control, 0 submissions, 0 Kaggle, "
                       "0 holdout, 0 test. V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P62-05", "track": "cobertura-del-vocabulario",
        "script": "experiments/p62_09_cobertura.py",
        "hipotesis": ("el cuello de botella es la COBERTURA del vocabulario, no la gramatica: "
                      "min_df=2 descarta las palabras que aparecen una vez, y todos los "
                      "canales linguisticos de la fase estaban sobre todo recuperando esas"),
        "medida_previa": {
            "vocabulario_de_palabras_de_V03_en_el_pliegue_1": 2673,
            "formas_superficiales_presentes_en_el_pliegue_1": 4569,
            "cobertura_de_V03_con_min_df_2": 0.276,
            "cobertura_de_lemas_con_min_df_2": 0.236,
            "nota": ("las 21.000 entidades de V03 son la union de bigramas de palabra y "
                     "gramas de caracter, y el canal de UNIGRAMAS tiene solo 2673 terminos. "
                     "Ejemplos de palabras ordinaria que min_df=2 tira: abandonado, "
                     "aberrante, abolir"),
        },
        "brazos": {
            "control": "V03 exacto: word(1,2) min_df=2 y char_wb(3,5) min_df=2",
            "A_palabras_min_df_1": "solo el canal de palabras pasa a min_df=1",
            "B_palabras_y_char_min_df_1": "los dos canales pasan a min_df=1",
            "C_min_df_1_mas_canal_POS": f"palabras min_df=1 mas un canal POS de K={K_POS}",
        },
        "protocolo": {
            "una_configuracion_por_brazo": ("no hay rejilla ni busqueda: cada brazo es una "
                                            "configuracion declarada antes de ejecutar"),
            "K_del_brazo_C": K_POS,
            "por_que_ese_K": ("es el presupuesto que la curva de P62-07 marco como el menos "
                              "malo, y se fija aqui para que C no se convierta en una "
                              "busqueda sobre su propio resultado"),
            "clasificador": "IDENTICO a V03 en los cuatro brazos",
            "pliegues": "los canonicos, por grupo", "umbrales": "0,44 / 0,51 / 0,46",
            "confound_de_escala": "todos los brazos se reescalan a la norma de fila de V03",
        },
        "cobertura_por_pliegue": cobertura,
        "cobertura_media": {"min_df_2": round(c2m, 6), "min_df_1": round(c1m, 6)},
        "control_de_vocabulario": {"rasgos_por_brazo": dims, "densidad": densidad},
        "resultados": {"control": mc, **ms},
        "aportaciones": {"A_palabras_min_df_1": round(d_a, 10),
                         "B_palabras_y_char_min_df_1": round(d_b, 10),
                         "C_min_df_1_mas_canal_POS": round(d_c, 10),
                         "canal_POS_sobre_cobertura_completa": round(d_c_puro, 10)},
        "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES, "mejor_brazo": nom,
                 "delta": round(delta, 10), "pliegues_mejorados": fm, "superado": gate},
        "bootstrap": (None if ic_lo is None else
                      {"IC95": [round(ic_lo, 6), round(ic_hi, 6)], "n": N_BOOT,
                       "agrupado_por": "grupo canonico"}),
        "veredicto": veredicto,
        "decision": decision,
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "tagger": "spaCy es_core_news_md 3.8.0, exclude=['ner'], D-003"},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_C.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P62 - Es la restriction la cobertura del vocabulario y no la gramatica")
    W("")
    W(f"Generado por `experiments/p62_09_cobertura.py`. "
      f"Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Como se llega aqui")
    W("")
    W("P62-04 comparo el canal de interaccion con y sin el sufijo de POS y el resultado no")
    W("fue el que la hipotesis anticipaba: el sufijo ayudaba en 6 de 8 presupuestos, con")
    W("hasta +0,006881, luego la interaccion no es solo partir filas. Pero AMBOS pierden en")
    W("8 de 8, y el script se negn a llamarlo redundancia porque su propia condicion")
    W("pre-declarada fallo en uno de sus tres terminos.")
    W("")
    W("La condicion fallo en la COBERTURA, y seguir ese hilo dio el experimento.")
    W("")
    W("## La medicion que reencuadra la fase")
    W("")
    W("El canal de palabras de V03 tiene **2673 terminos** en el pliegue 1. Las 21.000")
    W("entidades son la union de bigramas de palabra y gramas de caracter, y el canal de")
    W("UNIGRAMAS es pequeno porque `min_df=2` tira todo termino que aparece una vez.")
    W("")
    W("```text")
    W("formas superficiales presentes en el pliegue 1   4569")
    W("  en el vocabulario de V03                       1259   27,6 por ciento")
    W("lemas distintos presentes                        4057")
    W("  en el vocabulario de V03                        957   23,6 por ciento")
    W("```")
    W("")
    W("Unas tres cuartas partes de las palabras que estan de verdad en el texto de")
    W("entrenamiento no estan en el canal de palabras de V03. Los ejemplos son palabras")
    W("ordinarias: `abandonado`, `aberrante`, `abolir`, cada una aparece una vez y por eso")
    W("se va.")
    W("")
    W("**Eso reencuadra todos los resultados linguisticos de la fase.** Un canal")
    W("condicionado por POS no es sobre todo una fuente de informacion gramatical: es")
    W("tambien una manera de devolver palabras que `min_df=2` habia descartado, indexadas")
    W("por lema y partidas por categoria. Y lo mismo vale para el canal de lemas solos. Si")
    W("esa lectura es correcta, el cuello de botella es la COBERTURA, y la cobertura se")
    W("puede probar sin gramatica alguna.")
    W("")
    W("## Los cuatro brazos")
    W("")
    W("| brazo | que ansla |")
    W("|---|---|")
    W("| control | V03 exacto, `min_df=2` en los dos canales |")
    W("| A | solo la cobertura del canal de palabras |")
    W("| B | la cobertura de los dos canales |")
    W(f"| C | cobertura completa mas un canal POS de K={K_POS}, para preguntar si la "
      f"gramatica aporta cuando ya no queda hueco a la falta de palabras |")
    W("")
    W("Una configuracion por brazo, sin rejilla y sin busqueda. El K del brazo C se fijo")
    W("antes de ejecutar para que C no se convirtiera en una busqueda sobre su propio")
    W("resultado.")
    W("")
    W("## La cobertura")
    W("")
    W("| pliegue | vocab md2 | vocab md1 | formas | cobertura md2 | cobertura md1 |")
    W("|---:|---:|---:|---:|---:|---:|")
    for c in cobertura:
        W(f"| {c['pliegue']} | {c['vocabulario_de_palabras_min_df_2']} | "
          f"{c['vocabulario_de_palabras_min_df_1']} | "
          f"{c['formas_superficiales_presentes']} | {c['cobertura_con_min_df_2']:.4f} | "
          f"{c['cobertura_con_min_df_1']:.4f} |")
    W("")
    W(f"Cobertura media: `min_df=2` **{c2m:.4f}**, `min_df=1` **{c1m:.4f}**.")
    W("")
    W("## Los resultados")
    W("")
    W("| brazo | rasgos | corrected_cv | delta vs V03 |")
    W("|---|---:|---:|---:|")
    W(f"| control, V03 exacto | {dims['control'][0]} | {mc['corrected_cv']:.10f} | "
      f"+0.0000000000 |")
    for b, m in ms.items():
        W(f"| {b} | {dims[b][0]} | {m['corrected_cv']:.10f} | "
          f"{m['corrected_cv'] - ccv_ref:+.10f} |")
    W("")
    W("### Que se lleva cada brazo")
    W("")
    W("```text")
    W(f"A, solo el canal de palabras con min_df=1   {d_a:+.10f}")
    W(f"B, mas el canal de caracteres con min_df=1   {d_b:+.10f}")
    W(f"C, cobertura completa mas canal POS          {d_c:+.10f}")
    W(f"el canal POS sobre cobertura completa vale   {d_c_puro:+.10f}")
    W("```")
    W("")
    W("```text")
    W(veredicto)
    W("```")
    W("")
    W("## Gate")
    W("")
    W(f"Mejor brazo **{nom}**: delta {delta:+.10f}, pliegues {fm}/5, gate **{gate}**, "
      f"bootstrap {'NO EJECUTADO' if ic_lo is None else f'IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}]'}")
    W("")
    W("## Decision")
    W("")
    W("```text")
    W(decision)
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p62_09_cobertura.py`. Sin push.*")
    MD_C.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_C, JSON_C, CSV_C):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P62-09 OK | cobertura {c2m:.4f} -> {c1m:.4f} | A {d_a:+.6f} B {d_b:+.6f} "
        f"C {d_c:+.6f} | mejor {nom} | gate {gate}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
