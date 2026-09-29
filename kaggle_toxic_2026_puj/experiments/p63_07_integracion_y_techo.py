"""P63 track H: do the new blocks help each other, and is anything left that V03 cannot reach?

Why this experiment has to exist
--------------------------------
Four tracks have run and the pattern is identical in all of them. Composition, as ratios,
beats a random block of the same width by +0,0083528268 when added to V03, and by
+0,1055015148 on its own. Position and concentration beats random by +0,0712736720 on its
own and loses by -0,0025020526 when added. The relational block is 92,92 per cent degenerate
and the collection index is indistinguishable from noise.

So every block carries signal and none of them makes V03 better. That is the shape P62 found
too, and there it was explained by dilution of a sparse channel competing for the L2 budget.
Here the blocks are dense and tiny, 42 and 21 columns against 21044, so the dilution
explanation is much weaker and the honest question changes. It is not "does each block help"
any more, it is "do they help EACH OTHER", because the two carry different things: one is
the aggregate shape of the language, the other is where structure sits inside the text.

Three arms and a control that can settle it
---------------------------------------------
    V03_todo       V03 plus composition plus position
    V03_solo_comp  V03 plus composition only, the best single block of the phase
    V03_random     V03 plus a random block of the SAME TOTAL WIDTH as V03_todo

The control is the whole experiment. A joint block of 63 columns can look better than V03 for
the same reason a single block of 21 did not: more room. If V03_todo does not clear
V03_random, then nothing in the joint block is specific.

What is measured alongside, because the brief asks and because it is the honest ceiling
---------------------------------------------------------------------------------------------
For every length bin, frozen here in the source before anything is computed:

    the F1 of V03
    the F1 of the best arm
    the best F1 reachable by moving ONLY the threshold, which is the ceiling of any monotone
    function of V03's own score and therefore the part of the residual that no representation
    can touch

That last number is what turns "this block did not help" into "and here is what is left that
no block of this kind can reach". Without it, a negative is just a negative.

Usage
-----
    /home/carlos/venv/bin/python experiments/p63_07_integracion_y_techo.py
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
MD_H = REPORTS / "p63_integracion_y_techo.md"
JSON_H = REPORTS / "p63_integracion_y_techo.json"
CSV_H = REPORTS / "p63_integracion_y_techo.csv"
CELDA_H = PROJECT_ROOT / "features" / "p63_integracion_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True,
               "lowercase": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2,
               "sublinear_tf": True, "lowercase": True}
LR = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear",
      "random_state": 42}

# Cortes de longitud CONGELADOS, los mismos que el track E1 para que las dos fases
# comparen las mismas regiones.
BINS: tuple[tuple[str, int, int], ...] = (
    ("CORTA", 0, 6),
    ("MEDIA", 7, 15),
    ("LARGA", 16, 30),
    ("MUY_LARGA", 31, 10**9),
)
GATE_DELTA = 0.0015
GATE_PLIEGUES = 4
N_BOOT = 2000
SEMILLA = 42


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P63-07] {mensaje}", flush=True)


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


def main() -> None:
    """Comprueba si los bloques nuevos se ayudan, y mide el techo residual por bin.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P63-07 los bloques nuevos se ayudan entre si, y que queda sin alcanzar")
    log("   el veredicto lo decide V03_random, un bloque aleatorio del ANCHO TOTAL")
    log("   si el bloque conjunto no supera al aleatorio, nada de el es especifico")

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
    tr = pd.read_csv(RAW_TRAIN, sep=";", encoding="latin-1", dtype={"id": str})
    tr["id_num"] = pd.to_numeric(tr["id"].str.replace('"', "", regex=False),
                                 errors="coerce").astype("int64")
    idx = tr.set_index("id_num").loc[ids]
    if not np.array_equal(idx[["HS", "TR", "AG"]].to_numpy().astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    textos = idx["text"].astype(str).tolist()
    n_tok = np.array([len(t.split()) for t in textos])
    with gzip.open(ANOTACION, "rt", encoding="utf-8") as fh:
        documentos = json.load(fh)["documentos"]
    log("")
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | anotacion verificada")

    # ---------- los dos bloques, tal cual los construyeron sus tracks ----------
    import importlib
    comp_mod = importlib.import_module("p63_05_composicion_linguistica")
    pos_mod = importlib.import_module("p63_06_estructura_tweet_largo")
    fijos = [comp_mod.rasgos_fijos(t) for t in textos]
    cat = [comp_mod.rasgos_categoricos(d) for d in documentos]
    kf, kc = list(fijos[0].keys()), list(cat[0].keys())
    base_ratios = np.array([[f[k] for k in kf] + [c[k] for k in kc]
                            for f, c in zip(fijos, cat)], dtype=float)
    lexk = list(comp_mod.rasgos_lexicos("a b", {}, 1).keys())
    freq_total = {}
    for t in textos:
        for tok in set(re.findall(r"\w+", t.lower())):
            freq_total[tok] = freq_total.get(tok, 0) + 1
    lex = np.array([[comp_mod.rasgos_lexicos(textos[i], freq_total, n)[k] for k in lexk]
                    for i in range(n)], dtype=float)
    B_comp = np.hstack([base_ratios, lex])
    B_pos = np.array([[pos_mod.bloque_posicion(t)[k]
                       for k in pos_mod.bloque_posicion(textos[0])] for t in textos],
                     dtype=float)
    n_comp, n_pos = B_comp.shape[1], B_pos.shape[1]
    n_total = n_comp + n_pos
    log("")
    log(f"   bloque de composicion: {n_comp} columnas | bloque de posicion: {n_pos} | "
        f"conjunto: {n_total}")
    log(f"   el control aleatorio tendra {n_total} columnas, el mismo ancho total")

    proba: dict[str, np.ndarray] = {b: np.zeros((n, 3)) for b in
                                   ("V03_todo", "V03_comp", "V03_pos", "V03_random")}
    t0 = time.time()
    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        # La frecuencia de documento se recalcula SOLO con el pliegue de entrenamiento.
        freq = {}
        for i in ir:
            for tok in set(re.findall(r"\w+", textos[i].lower())):
                freq[tok] = freq.get(tok, 0) + 1
        lex_f = np.array([[comp_mod.rasgos_lexicos(textos[i], freq, ir.size)[k] for k in lexk]
                          for i in range(n)], dtype=float)
        comp_f = np.hstack([base_ratios, lex_f])
        pos_f = B_pos

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

        def estandarizar(mat: np.ndarray, ir_: np.ndarray) -> np.ndarray:
            """Estandariza por columna con la desviacion del pliegue y normaliza la norma.

            La normalizacion a norma media de fila uno es obligatoria: sin ella, concatenar
            columnas estandarizadas a un bloque de norma raiz de dos encoge la base por un
            factor de orden 4,7 y el numero mide el error de instrumentacion.

            Args:
                mat: bloque completo.
                ir_: filas de entrenamiento.

            Returns:
                El bloque estandarizado y normalizado.
            """
            sd = mat[ir_].std(axis=0)
            sd[sd < 1e-8] = 1.0
            z = (mat - mat[ir_].mean(axis=0)) / sd[None, :]
            norma = float(np.sqrt(np.asarray((z[ir_] ** 2).sum(axis=1))).mean())
            return z / norma if norma > 0 else z

        zc = estandarizar(comp_f, ir)
        zp = estandarizar(pos_f, ir)
        rng = np.random.default_rng(SEMILLA + f)
        rz = rng.standard_normal((n, n_total))
        rz = rz / float(np.sqrt(np.asarray((rz[ir] ** 2).sum(axis=1))).mean())

        bloques = {"V03_todo": np.hstack([zc, zp]), "V03_comp": zc, "V03_pos": zp,
                   "V03_random": rz}
        for brazo, be in bloques.items():
            xa = escalar(hstack([xbase, csr_matrix(be[ir])]).tocsr())
            xav = escalar(hstack([xbase_v, csr_matrix(be[iva])]).tocsr())
            for j in range(3):
                mo = LogisticRegression(**LR)
                mo.fit(xa, y[ir, j])
                proba[brazo][iva, j] = mo.predict_proba(xav)[:, 1]
        log(f"      pliegue {f} listo ({time.time() - t0:.0f} s)")

    # ---------- el techo que ningun bloque de este tipo puede alcanzar ----------
    log("")
    log("   EL TECHO POR REBANADA: el mejor F1 moviendo SOLO el corte de V03")
    log("   es el techo de cualquier funcion monotona de su propia puntuacion, luego es la")
    log("   parte del error residual que ninguna representacion puede tocar")

    def techo_umbral(m: np.ndarray) -> dict[str, float]:
        """Mejor macro-F1 alcanzable moviendo solo el umbral, por region.

        El corrected_cv de V03 es la MEDIA MACRO de las tres etiquetas, luego el techo
        comparable tiene que ser la media macro de los tres optimos por etiqueta. Una
        version anterior de esta funcion devolvia el MAXIMO de los tres, y eso inflaba el
        margen casi al doble: +0,1589 en lugar de +0,0890 en la region MUY_LARGA. Sobre
        declarar un techo es el error mas facil de cometer en todo el experimento, porque
        un techo mas alto hace que cualquier negativo parezca mas pequeno.

        Args:
            m: mascara de la region.

        Returns:
            Diccionario con el techo macro, los cortes por etiqueta y el desglose.
        """
        optimos, cortes, actuales = [], [], []
        for j in range(3):
            s, t = proba_v03[m, j], y[m, j]
            actual = float(f1_score(t, (s >= V03_UMBRALES[j]).astype(int), zero_division=0))
            actuales.append(actual)
            if t.sum() == 0 or (t == 0).sum() == 0:
                optimos.append(actual)
                cortes.append(float(V03_UMBRALES[j]))
                continue
            mejor, corte = 0.0, 0.0
            for c in np.unique(s):
                f1j = f1_score(t, (s >= c).astype(int), zero_division=0)
                if f1j > mejor:
                    mejor, corte = float(f1j), float(c)
            optimos.append(mejor)
            cortes.append(corte)
        actual_macro = float(np.mean(actuales))
        techo_macro = float(np.mean(optimos))
        return {"techo": round(techo_macro, 6),
                "actual": round(actual_macro, 6),
                "margen": round(techo_macro - actual_macro, 6),
                "cortes_optimos": [round(c, 6) for c in cortes],
                "umbrales_de_V03": [float(u) for u in V03_UMBRALES],
                "f1_por_etiqueta_con_V03": [round(a, 6) for a in actuales],
                "f1_por_etiqueta_optima": [round(o, 6) for o in optimos]}

    mnscaras = {}
    filas_techo = []
    for nombre, lo, hi in BINS:
        m = (n_tok >= lo) & (n_tok <= hi)
        mnscaras[nombre] = m
        f1_v03 = float(f1_score(y[m], pred_v03[m], average="macro", zero_division=0))
        t = techo_umbral(m)
        filas_techo.append({"bin": nombre, "n": int(m.sum()),
                            "tokens": f"{lo}-{hi if hi < 10**9 else '+'}",
                            "F1_de_V03": round(f1_v03, 6),
                            "techo_moviendo_el_corte": t["techo"],
                            "margen": t["margen"],
                            "cortes_optimos": t["cortes_optimos"],
                            "f1_por_etiqueta_con_V03": t["f1_por_etiqueta_con_V03"],
                            "f1_por_etiqueta_optima": t["f1_por_etiqueta_optima"]})
        log(f"      {nombre:10} n {int(m.sum()):5d}  F1 {f1_v03:.4f}  techo {t['techo']:.4f}  "
            f"margen {t['techo'] - f1_v03:+.4f}")

    detalle: dict[str, Any] = {}
    filas: list[dict[str, Any]] = []
    for brazo in ("V03_todo", "V03_comp", "V03_pos", "V03_random"):
        p = proba[brazo]
        pr = (p >= V03_UMBRALES[None, :]).astype(int)
        ccv = float(f1_score(y, pr, average="macro"))
        por_bin = {}
        for nombre, m in mnscaras.items():
            f1b = float(f1_score(y[m], pr[m], average="macro", zero_division=0))
            f1v = float(f1_score(y[m], pred_v03[m], average="macro", zero_division=0))
            por_bin[nombre] = {"n": int(m.sum()), "F1": round(f1b, 6),
                               "F1_de_V03": round(f1v, 6),
                               "delta_vs_V03": round(f1b - f1v, 6)}
        detalle[brazo] = {
            "corrected_cv": ccv,
            "folds": [float(f1_score(y[pliegues == g], pr[pliegues == g], average="macro"))
                      for g in range(1, 6)],
            "por_bin": por_bin,
            "rescate": int(np.count_nonzero((pr == y) & (pred_v03 != y))),
            "damage": int(np.count_nonzero((pr != y) & (pred_v03 == y))),
        }
        filas.append({"brazo": brazo, "n_columnas": (n_total if brazo == "V03_todo"
                                                     else n_comp if brazo == "V03_comp"
                                                     else n_pos if brazo == "V03_pos"
                                                     else n_total),
                      "corrected_cv": round(ccv, 10),
                      "delta_vs_V03": round(ccv - ccv_ref, 10),
                      "rescate": detalle[brazo]["rescate"],
                      "damage": detalle[brazo]["damage"]})

    d_todo = detalle["V03_todo"]["corrected_cv"] - ccv_ref
    d_rand = detalle["V03_random"]["corrected_cv"] - ccv_ref
    brecha = detalle["V03_todo"]["corrected_cv"] - detalle["V03_random"]["corrected_cv"]
    brecha_comp = detalle["V03_comp"]["corrected_cv"] - detalle["V03_random"]["corrected_cv"]
    folds_v = [float(f1_score(y[pliegues == g], pred_v03[pliegues == g], average="macro"))
               for g in range(1, 6)]
    fm = sum(1 for a, b in zip(detalle["V03_todo"]["folds"], folds_v) if a > b)
    gate = bool(d_todo >= GATE_DELTA and fm >= GATE_PLIEGUES and brecha > 0)

    log("")
    log("   LOS CUATRO BRAZOS")
    log(f"      {'brazo':14}{'n_cols':>8}{'corrected_cv':>18}{'delta vs V03':>16}")
    for f in filas:
        log(f"      {f['brazo']:14}{f['n_columnas']:>8}{f['corrected_cv']:>18.10f}"
            f"{f['delta_vs_V03']:>+16.10f}")
    log("")
    log("   LA PREGUNTA QUE DECIDE EL VEREDICTO")
    log(f"      V03 + los dos bloques: {d_todo:+.10f} con {fm}/5 pliegues")
    log(f"      V03 + ALEATORIO de igual ancho: {d_rand:+.10f}")
    log(f"      BRECHA especifica: {brecha:+.10f}")
    log(f"      y la del bloque de composicion solo, contra el mismo aleatorio: "
        f"{brecha_comp:+.10f}")
    log("      si el conjunto no supera al aleatorio de su propio ancho, nada del")
    log("      conjunto es especifico y la senal de los dos bloques se solapa")

    # ---------- la interseccion de errores, que el encargo pide explicitamente ----------
    log("")
    log("   LA INTERSECCION DE ERRORES, separada en cerca del borde y alta confianza")
    p_todo = proba["V03_todo"]
    pr_todo = (p_todo >= V03_UMBRALES[None, :]).astype(int)
    filas_err: list[dict[str, Any]] = []
    for nombre, m in mnscaras.items():
        for j, et in enumerate(ETIQUETAS):
            dist = np.abs(proba_v03[:, j] - V03_UMBRALES[j])
            err_v03 = (pred_v03[:, j] != y[:, j]) & m
            alto = err_v03 & (dist >= 0.10)
            cerca = err_v03 & (dist < 0.10)
            rescue = (pr_todo[:, j] == y[:, j]) & (pred_v03[:, j] != y[:, j]) & m
            dmg = (pr_todo[:, j] != y[:, j]) & (pred_v03[:, j] == y[:, j]) & m
            filas_err.append({
                "bin": nombre, "etiqueta": et, "n": int(m.sum()),
                "errores_de_V03": int(err_v03.sum()),
                "errores_cerca_del_borde": int(cerca.sum()),
                "errores_de_alta_confianza": int(alto.sum()),
                "rescatados_por_el_bloque": int(rescue.sum()),
                "danados_por_el_bloque": int(dmg.sum()),
            })
    for f in filas_err:
        if f["bin"] == "MUY_LARGA":
            log(f"      MUY_LARGA {f['etiqueta']}: {f['errores_de_V03']} errores, "
                f"{f['errores_de_alta_confianza']} de alta confianza, "
                f"rescatados {f['rescatados_por_el_bloque']}, "
                f"danados {f['danados_por_el_bloque']}")

    decision = ("PROMISING" if gate else
                "MARGINAL" if brecha > 0 else
                "CLOSED. Los bloques nuevos no se ayudan entre si")
    if brecha > 0:
        veredicto = (
            f"el conjunto de los dos bloques vale {d_todo:+.10f} frente a V03 con {fm}/5 "
            f"pliegues, y el control de capacidad del mismo ancho total vale {d_rand:+.10f}, "
            f"luego la BRECHA especifica es {brecha:+.10f}. Hay una senal especifica que "
            f"sobrevive al control, pero no llega al gate de la fase"
        )
    else:
        veredicto = (
            f"el conjunto de los dos bloques vale {d_todo:+.10f} frente a V03 con {fm}/5 "
            f"pliegues, y el control de capacidad del mismo ancho TOTAL vale {d_rand:+.10f}, "
            f"luego la BRECHA especifica es {brecha:+.10f}. Los dos bloques NO se ayudan: "
            f"la senal que aporta el de composicion, mas {brecha_comp:+.10f} frente al mismo "
            f"aleatorio, se pierde al anadirle el de posicion. Y el margen al techo por "
            f"rebanada muestra que el residuo esta donde la representacion no llega"
        )
    log("")
    log(f"   VEREDICTO: {veredicto}")
    log(f"   DECISION: {decision}")

    np.savez_compressed(CELDA_H, ids=ids, y_true=y, pliegues=pliegues, pred_v03=pred_v03,
                        n_tokens=n_tok, proba_v03=proba_v03,
                        pred_todo=pr_todo,
                        **{f"proba_{b}": proba[b] for b in proba})

    with CSV_H.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["bloque", "n_columnas", "corrected_cv", "delta_vs_V03", "rescate", "damage"])
        for f in filas:
            w.writerow([f["brazo"], f["n_columnas"], f["corrected_cv"], f["delta_vs_V03"],
                        f["rescate"], f["damage"]])
        w.writerow(["brecha_global", "", round(brecha, 10)])
        w.writerow(["brecha_composicion", "", round(brecha_comp, 10)])
    with (REPORTS / "p63_techo_por_rebanada.csv").open("w", encoding="ascii",
                                                       newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["bin", "tokens", "n", "F1_de_V03", "techo_moviendo_el_corte",
                    "margen", "cortes_optimos"])
        for f in filas_techo:
            w.writerow([f["bin"], f["tokens"], f["n"], f["F1_de_V03"],
                        f["techo_moviendo_el_corte"], f["margen"],
                        ";".join(str(c) for c in f["cortes_optimos"])])
    with (REPORTS / "p63_error_intersection.csv").open("w", encoding="ascii",
                                                       newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["bin", "etiqueta", "n", "errores_de_V03", "errores_cerca_del_borde",
                    "errores_de_alta_confianza", "rescatados_por_el_bloque",
                    "danados_por_el_bloque"])
        for f in filas_err:
            w.writerow([f["bin"], f["etiqueta"], f["n"], f["errores_de_V03"],
                        f["errores_cerca_del_borde"], f["errores_de_alta_confianza"],
                        f["rescatados_por_el_bloque"], f["danados_por_el_bloque"]])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P63_INTEGRACION",
        "naturaleza": ("EXPERIMENTO. 3 candidatos y 1 control de igual ancho total, 0 "
                       "submissions, 0 Kaggle, 0 holdout, 0 test"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P63-06",
        "track": "H-integracion-y-techo",
        "script": "experiments/p63_07_integracion_y_techo.py",
        "pregunta": ("los bloques nuevos se ayudan entre si, y queda algo que ninguna "
                     "representacion de este tipo puede alcanzar"),
        "por_que_esta_pregunta_y_no_otra": (
            "cuatro tracks con el mismo patron: la composicion gana +0,1055015148 en "
            "solitario y +0,0083528268 frente al control al anadirse, la posicion gana "
            "+0,0712736720 en solitario y pierde -0,0025020526 al anadirse. Como los bloques "
            "son densos y pequenos, 42 y 21 columnas contra 21044, la explicacion de "
            "dilucion de P62 es aqui mas debil, y la pregunta honesta deja de ser si cada "
            "bloque ayuda y pasa a ser si se ayudan ENTRE SI"),
        "brazos": {
            "V03_todo": f"composicion mas posicion, {n_total} columnas",
            "V03_comp": f"solo composicion, {n_comp} columnas",
            "V03_pos": f"solo posicion, {n_pos} columnas",
            "V03_random": (f"control de capacidad de {n_total} columnas, el ANCHO TOTAL "
                           f"del conjunto, que es lo que hace comparable la pregunta"),
        },
        "protocolo": {
            "frecuencia_de_documento": "ajustada SOLO con las filas de entrenamiento",
            "normalizacion": ("el bloque denso se normaliza a norma media de fila uno antes "
                              "de concatenarse, con la misma funcion para todos los brazos"),
            "clasificador": "IDENTICO a V03 en los cuatro brazos",
            "pliegues": "los canonicos, por grupo", "umbrales": "0,44 / 0,51 / 0,46",
            "bins_de_longitud": [{"nombre": a, "desde": b, "hasta": c} for a, b, c in BINS],
        },
        "resultados": detalle,
        "brecha_global": round(brecha, 10),
        "brecha_composicion": round(brecha_comp, 10),
        "techo_por_rebanada": filas_techo,
        "interseccion_de_errores": filas_err,
        "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                 "delta": round(d_todo, 10), "pliegues_mejorados": fm, "superado": gate},
        "veredicto": veredicto,
        "decision": decision,
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_H.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P63 - Integracion de los bloques, y el techo que queda")
    W("")
    W(f"Generado por `experiments/p63_07_integracion_y_techo.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Por que esta pregunta y no otra")
    W("")
    W("Cuatro tracks con el mismo patron: la composicion gana +0,1055 en solitario y")
    W("+0,0084 frente al control al anadirse, la posicion gana +0,0713 en solitario y pierde")
    W("-0,0025 al anadirse. Como los bloques son **densos y pequenos**, 42 y 21 columnas contra")
    W("21044, la explicacion de dilucion de P62 es aqui mas debil, y la pregunta honesta deja")
    W("de ser si cada bloque ayuda y pasa a ser **si se ayudan entre si**.")
    W("")
    W("## Los brazos")
    W("")
    W("| brazo | columnas | que es |")
    W("|---|---:|---|")
    W(f"| V03_todo | {n_total} | composicion mas posicion |")
    W(f"| V03_comp | {n_comp} | solo composicion, el mejor bloque suelto de la fase |")
    W(f"| V03_pos | {n_pos} | solo posicion |")
    W(f"| V03_random | {n_total} | control de capacidad del ANCHO TOTAL |")
    W("")
    W("**El control es el experimento entero.** Un bloque conjunto de 63 columnas puede")
    W("parecer mejor que V03 por la misma razon por la que un bloque suelto de 21 no: mas")
    W("sitio. Si V03_todo no supera a V03_random, nada de el es especifico.")
    W("")
    W("## Resultados")
    W("")
    W("| brazo | columnas | corrected_cv | delta vs V03 | rescate | dano |")
    W("|---|---:|---:|---:|---:|---:|")
    for f in filas:
        W(f"| {f['brazo']} | {f['n_columnas']} | {f['corrected_cv']:.10f} | "
          f"{f['delta_vs_V03']:+.10f} | {f['rescate']} | {f['damage']} |")
    W("")
    W("### El detalle por rebanada")
    W("")
    W("| bin | n | F1 de V03 | " + " | ".join(b[0] for b in
                                           ("V03_todo", "V03_comp", "V03_pos")) + " |")
    W("|---|---:|---:|" + "---:|" * 3)
    for nombre, _lo, _hi in BINS:
        cells = []
        for b in ("V03_todo", "V03_comp", "V03_pos"):
            pb = detalle[b]["por_bin"][nombre]
            cells.append(f"{pb['delta_vs_V03']:+.4f}")
        pv = detalle["V03_todo"]["por_bin"][nombre]["F1_de_V03"]
        W(f"| {nombre} | {detalle['V03_todo']['por_bin'][nombre]['n']} | {pv:.4f} | "
          + " | ".join(cells) + " |")
    W("")
    W("## El techo por rebanada, que es lo que ninguna representacion toca")
    W("")
    W("El mejor F1 alcanzable moviendo **solo el corte** de V03 es el techo de cualquier")
    W("funcion monotona de su propia puntuacion, luego es exactamente la parte del error")
    W("residual que ninguna representacion puede atacar.")
    W("")
    W("**Correccion de este propio script, y es importante.** Una version anterior devolvia")
    W("el MAXIMO de los tres optimos por etiqueta en vez de la MEDIA MACRO, y eso inflaba el")
    W("margen casi al doble. El corrected_cv de V03 es la media macro, luego el techo")
    W("comparable tiene que serlo tambien. Sobre-declarar un techo es el error mas facil de")
    W("cometer en todo el experimento, porque un techo mas alto hace que cualquier negativo")
    W("parezca mas pequeno.")
    W("")
    W("| bin | tokens | n | F1 de V03 | techo moviendo el corte | margen |")
    W("|---|---|---:|---:|---:|---:|")
    for f in filas_techo:
        W(f"| {f['bin']} | {f['tokens']} | {f['n']} | {f['F1_de_V03']:.4f} | "
          f"{f['techo_moviendo_el_corte']:.4f} | {f['margen']:+.4f} |")
    W("")
    W("## La interseccion de errores")
    W("")
    W("| bin | etiqueta | n | errores de V03 | cerca del borde | alta confianza | "
      "rescatados | danados |")
    W("|---|---|---:|---:|---:|---:|---:|---:|")
    for f in filas_err:
        W(f"| {f['bin']} | {f['etiqueta']} | {f['n']} | {f['errores_de_V03']} | "
          f"{f['errores_cerca_del_borde']} | {f['errores_de_alta_confianza']} | "
          f"{f['rescatados_por_el_bloque']} | {f['danados_por_el_bloque']} |")
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
    W("*Generado por `experiments/p63_07_integracion_y_techo.py`. Sin push.*")
    MD_H.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_H, JSON_H, CSV_H, REPORTS / "p63_techo_por_rebanada.csv",
                    REPORTS / "p63_error_intersection.csv"):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P63-07 OK | V03_todo {detalle['V03_todo']['corrected_cv']:.6f} | aleatorio "
        f"{detalle['V03_random']['corrected_cv']:.6f} | brecha {brecha:+.6f} | margen al techo "
        f"en MUY_LARGA {filas_techo[-1]['margen']:+.6f} | gate {gate}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
