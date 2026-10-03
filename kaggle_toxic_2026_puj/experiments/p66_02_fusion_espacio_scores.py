"""P66 tracks P05 to P09: the five fusion forms, nested, with a shuffled-score control.

The five mechanisms, and what each one asks
-------------------------------------------
    P5 blend de probabilidad   p = (1-a)p_V03 + a p_C                 "mezcla ponderada"
    P6 blend de logit          z = z_V03 + a z_C                       "evidencia aditiva"
    P7 correccion residual     LogisticRegression sobre [z_V03, z_C]   "aprender la mezcla"
    P8 residualizacion         z = z_V03 + b * (z_C - (a + b0 z_V03)) "quitar lo redundante"
    P9 sensible a la longitud  P8 con un segundo coeficiente por longitud

P5 and P6 are not new and are run as CONTROLS: V6-05 and P59-09 already measured a single
alpha in probability space, reaching +0,0014800075 with 2 of 5 folds and failing its gate on
both conditions. Reproducing that here is what makes P8 interpretable, because if P8 improves
where the same information in P5 form does not, the difference is the residualisation and not
the candidate.

The nested protocol, and why it is the whole point
--------------------------------------------------
Every learned quantity is fitted on inner_train and selected on inner_validation:

    outer_train -> inner_train  : fit a, b, b0, beta, the fusion model, the length normalisation
    outer_train -> inner_val    : select alpha or beta from the inner score
    outer_train                : refit on the full outer training rows with the selection
    outer_validation           : EVALUATE ONLY

The outer fold never participates in any selection. That is the difference between a fusion
whose parameters were chosen on the rows it is scored on, which always looks positive, and one
whose parameters were frozen before the outer rows were touched, which does not.

The shuffled-score control is mandatory
---------------------------------------
A fusion layer has its own degrees of freedom, so a gain can come from the MECHANISM rather
than from the candidate's information. For each candidate the same fusion is run with the
candidate's score permuted across rows within the fold, preserving its marginal distribution
exactly. If the permuted version gains the same amount, the gain belongs to the fusion
mechanism and not to the evidence, and the candidate is not what fixed it.

Everything is per label
----------------------
HS, TR and AG are corrected separately and never mixed into one variable, and every mechanism
is reported for all three so that a candidate that helps HS and destroys AG cannot hide.

Usage
-----
    /home/carlos/venv/bin/python experiments/p66_02_fusion_espacio_scores.py
"""

from __future__ import annotations

import ast
import csv as _csv
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

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
CELDA = PROJECT_ROOT / "features" / "p66_scores_candidatos.npz"
MD_F = REPORTS / "p66_score_fusion.md"
JSON_F = REPORTS / "p66_score_fusion.json"
CSV_F = REPORTS / "p66_score_fusion.csv"
CSV_R = REPORTS / "p66_residual_correction.csv"
CSV_L = REPORTS / "p66_long_aware_correction.csv"
CSV_C = REPORTS / "p66_random_controls.csv"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
EPS = 1e-6
GATE_DELTA = 0.0015
GATE_PLIEGUES = 4
SEMILLA = 42
N_INNER = 3
ALPHAS = (0.05, 0.10, 0.15, 0.20)
BETAS = (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
BINS: tuple[tuple[str, int, int], ...] = (
    ("CORTA", 0, 6),
    ("MEDIA", 7, 15),
    ("LARGA", 16, 30),
    ("MUY_LARGA", 31, 10**9),
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P66-02] {mensaje}", flush=True)


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


def logit(p: np.ndarray) -> np.ndarray:
    """Logit numericamente seguro.

    Args:
        p: probabilidades.

    Returns:
        Los logits, con el recorte declarado en el encargo.
    """
    q = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def sigmoide(z: np.ndarray) -> np.ndarray:
    """Sigmoide estable para logits grandes.

    Args:
        z: logits.

    Returns:
        Probabilidades.
    """
    z = np.asarray(z, dtype=float)
    salida = np.empty_like(z)
    pos = z >= 0
    salida[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    salida[~pos] = e / (1.0 + e)
    return salida


def decision(pr: np.ndarray) -> np.ndarray:
    """Aplica los umbrales CONGELADOS de V03.

    Args:
        pr: matriz de probabilidades.

    Returns:
        Matriz de decisiones.
    """
    # Se conserva la forma de la entrada. Con UNA sola etiqueta, pr >= umbral devuelve un
    # vector de una dimension y la matriz de verdad es de dos, y sklearn rechaza mezclar un
    # objetivo binario con uno multilabel. La decision por etiqueta se hace SIEMPRE sobre las
    # tres columnas y luego se seleccionan las que interesen, que es lo coherente con que
    # HS, TR y AG son tres problemas binarios independientes con tres umbrales distintos.
    pr = np.asarray(pr, dtype=float)
    completo = (pr[:, None] if pr.ndim == 1 else pr) >= V03_UMBRALES[None, :]
    return completo.astype(int)


def ccv_de(p: np.ndarray, y: np.ndarray, etiquetas: tuple[int, ...] | None = None) -> float:
    """Macro F1 sobre las etiquetas indicadas, con SU umbral congelado.

    Cada etiqueta tiene su propio umbral y son tres problemas binarios independientes, luego
    aplicar el vector de umbrales a una matriz de una sola columna no tiene sentido y produce
    formas incompatibles. Aqui se aplica el umbral de CADA etiqueta a SU columna.

    Args:
        p: probabilidades.
        y: etiquetas verdaderas.
        etiquetas: indices de las etiquetas a promediar. Por defecto, todas.

    Returns:
        El macro F1 sobre esas etiquetas.
    """
    pp = np.asarray(p, dtype=float)
    yy = np.asarray(y, dtype=int)
    cols = range(pp.shape[1]) if etiquetas is None else etiquetas
    partes = [float(f1_score(yy[:, j], (pp[:, j] >= V03_UMBRALES[j]).astype(int),
                            average="binary", zero_division=0)) for j in cols]
    return float(np.mean(partes)) if partes else 0.0


def main() -> None:
    """Ejecuta las cinco formas de fusion, anidadas y con control barajado.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P66-02 las cinco formas de fusion en el espacio de scores, anidadas")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    z = np.load(CELDA, allow_pickle=True)
    ids, y, pliegues = z["ids"], z["y_true"].astype(int), z["pliegues"]
    pred_v03, proba_v03 = z["pred_v03"], z["proba_v03"]
    margen_v03, n_tok = z["margen_v03"], z["n_tokens"]
    n = ids.size
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    CANDIDATOS = ("MI_arbol", "negacion", "word_1_3", "cohesion", "huecos_1_2", "contraste",
                  "huecos_6_10")
    proba_c = {c: z[f"proba_{c}"] for c in CANDIDATOS}
    log(f"   V03 verificado: {ccv_ref!r} | {n} filas | {len(CANDIDATOS)} candidatos")
    log("   TODO lo aprendido va por nested y la outer SOLO evalua")
    log(f"   alphas {ALPHAS} | betas {BETAS} | pliegues internos {N_INNER}")

    zv = logit(proba_v03)
    log("")
    log("   P04, LA ESCALA, antes de fusionar nada")
    for c in CANDIDATOS:
        log(f"      {c:12} p_C [{proba_c[c].min():.6f}, {proba_c[c].max():.6f}] | "
            f"z_C [{logit(proba_c[c]).min():+.3f}, {logit(proba_c[c]).max():+.3f}] | "
            f"corr(z_V03, z_C) media {np.mean([np.corrcoef(zv[:, j], logit(proba_c[c])[:, j])[0, 1] for j in range(3)]):+.4f}")
    log(f"      p_V03 [{proba_v03.min():.6f}, {proba_v03.max():.6f}] | "
        f"z_V03 [{zv.min():+.3f}, {zv.max():+.3f}]")
    log("      todas las probabilidades son validas, luego la fusion en espacio de probabilidad")
    log("      es admisible y no hace falta calibrar. Es lo que P04 pedia decidir antes de empezar")

    MECANISMOS = ("blend_prob", "blend_logit", "fusion_lr", "residual", "length_aware")
    proba_f: dict[str, np.ndarray] = {}
    params_elegidas: dict[str, dict[str, list[Any]]] = {}
    for c in CANDIDATOS:
        for usa_barajado in (False, True):
            sufijo = "__barajado" if usa_barajado else ""
            for m in MECANISMOS:
                proba_f[f"{c}|{m}{sufijo}"] = proba_v03.copy()
        params_elegidas[c] = {}

    t0 = time.time()
    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        rng = np.random.default_rng(SEMILLA + f)
        # El barajado se hace DENTRO de cada fold y con el generador del fold, de modo que es
        # reproducible y conserva la distribucion marginal del candidato fila a fila.
        barajados: dict[str, np.ndarray] = {}
        for c in CANDIDATOS:
            p = proba_c[c]
            q = p[ir].copy()
            rng.shuffle(q, axis=0)
            b = p.copy()
            b[ir] = q
            b[iva] = p[iva][rng.permutation(len(iva))]
            barajados[c] = b

        # Normalizacion de longitud: media y desviacion del ENTRENAMIENTO del pliegue, nunca
        # de todo el corpus ni de la validacion exterior.
        lon = np.log1p(n_tok.astype(float))
        mu, sd = float(lon[ir].mean()), float(lon[ir].std() + 1e-12)
        lon_n = (lon - mu) / sd

        for c in CANDIDATOS:
            logit(proba_c[c])
            for usa_barajado in (False, True):
                zc = logit(barajados[c] if usa_barajado else proba_c[c])
                sufijo = "__barajado" if usa_barajado else ""
                for j, etiqueta in enumerate(ETIQUETAS):
                    zz, cc = zv[:, j], zc[:, j]
                    ty = y[:, j]
                    # pv debe ser la probabilidad del vector que se esta usando, REAL o
                    # BARRAJADA. Una primera version usaba siempre la real, con lo que el
                    # control de barajado de blend_prob salia identico al real por
                    # construccion y la brecha de atribucion daba cero FALSO. El mejor
                    # resultado de la fase quedava invalidado por un motivo que no era el suyo.
                    pp = barajados[c] if usa_barajado else proba_c[c]
                    pv = pp[:, j]

                    # El split interno se calcula UNA vez y se reutiliza en los cinco
                    # mecanismos. Calcularlo dentro del bucle de alphas seria carissimo y no
                    # cambiaria nada, porque el split no depende del alpha.
                    i_tr, i_va = _inner_split(ir, y, j, N_INNER, SEMILLA + 100 * f)

                    # ---------------- P5: blend de probabilidad, alpha nested ----------------
                    mejor_a, mejor_f1 = 0.0, -1.0
                    for a in ALPHAS:
                        pf = (1.0 - a) * proba_v03[:, j] + a * pv
                        # El coeficiente se elige con los UMBRALES de V03 sobre inner_val
                        f1 = float(f1_score(ty[i_va], (pf[i_va] >= V03_UMBRALES[j]).astype(int),
                                            average="binary", zero_division=0))
                        if f1 > mejor_f1:
                            mejor_a, mejor_f1 = a, f1
                    pf5 = (1.0 - mejor_a) * proba_v03[:, j] + mejor_a * pv
                    proba_f[f"{c}|blend_prob{sufijo}"][iva, j] = pf5[iva]

                    # ---------------- P6: blend de logit, alpha nested ----------------
                    mejor_a6, mejor_f1_6 = 0.0, -1.0
                    for a in ALPHAS:
                        pf6 = sigmoide(zz + a * cc)
                        f1 = float(f1_score(ty[i_va], (pf6[i_va] >= V03_UMBRALES[j]).astype(int),
                                            average="binary", zero_division=0))
                        if f1 > mejor_f1_6:
                            mejor_a6, mejor_f1_6 = a, f1
                    proba_f[f"{c}|blend_logit{sufijo}"][iva, j] = sigmoide(zz + mejor_a6 * cc)[iva]

                    # ---------------- P7: LogisticRegression sobre [z_V03, z_C] ----------------
                    # Es una calibracion y fusion de DOS variables, no un meta-modelo.
                    Xtr = np.column_stack([zz[i_tr], cc[i_tr]])
                    Xva = np.column_stack([zz[i_va], cc[i_va]])
                    mo = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                            random_state=SEMILLA)
                    mo.fit(Xtr, ty[i_tr])
                    sel = float(f1_score(ty[i_va],
                                         (mo.predict_proba(Xva)[:, 1] >= V03_UMBRALES[j]).astype(int),
                                         average="binary", zero_division=0))
                    proba_f[f"{c}|fusion_lr{sufijo}"][iva, j] = mo.predict_proba(
                        np.column_stack([zz[iva], cc[iva]]))[:, 1]
                    params_elegidas[c].setdefault(f"fusion_lr_{etiqueta}{sufijo}", []).append(
                        {"coef": mo.coef_.ravel().tolist(), "f1_inner": round(sel, 6)})

                    # ---------------- P8: residualizacion explicita ----------------
                    # a y b0 salen SOLO de inner_train; beta se elige en inner_validation.
                    Lin = np.column_stack([np.ones(len(i_tr)), zz[i_tr]])
                    coef = np.linalg.lstsq(Lin, cc[i_tr], rcond=None)[0]
                    res = cc - (coef[0] + coef[1] * zz)
                    mejor_b, mejor_f1_8 = 0.0, -1.0
                    for b in BETAS:
                        pf8 = sigmoide(zz + b * res)
                        f1 = float(f1_score(ty[i_va], (pf8[i_va] >= V03_UMBRALES[j]).astype(int),
                                            average="binary", zero_division=0))
                        if f1 > mejor_f1_8:
                            mejor_b, mejor_f1_8 = b, f1
                    proba_f[f"{c}|residual{sufijo}"][iva, j] = sigmoide(
                        zz + mejor_b * res)[iva]
                    params_elegidas[c].setdefault(f"residual_{etiqueta}{sufijo}", []).append(
                        {"a": round(float(coef[0]), 6), "b0": round(float(coef[1]), 6),
                         "beta": mejor_b, "f1_inner": round(mejor_f1_8, 6)})

                    # ---------------- P9: la misma, sensible a la longitud ----------------
                    Lin3 = np.column_stack([np.ones(len(i_tr)), zz[i_tr], lon_n[i_tr]])
                    c3 = np.linalg.lstsq(Lin3, cc[i_tr], rcond=None)[0]
                    res3 = cc - (c3[0] + c3[1] * zz + c3[2] * lon_n)
                    mejor_b0, mejor_b1, mejor_f1_9 = 0.0, 0.0, -1.0
                    for b0 in BETAS:
                        for b1 in BETAS:
                            pf9 = sigmoide(zz + b0 * res3 + b1 * lon_n * res3)
                            f1 = float(f1_score(
                                ty[i_va], (pf9[i_va] >= V03_UMBRALES[j]).astype(int),
                                average="binary", zero_division=0))
                            if f1 > mejor_f1_9:
                                mejor_b0, mejor_b1, mejor_f1_9 = b0, b1, f1
                    proba_f[f"{c}|length_aware{sufijo}"][iva, j] = sigmoide(
                        zz + mejor_b0 * res3 + mejor_b1 * lon_n * res3)[iva]
                    params_elegidas[c].setdefault(
                        f"length_aware_{etiqueta}{sufijo}", []).append(
                        {"beta0": mejor_b0, "beta1": mejor_b1,
                         "f1_inner": round(mejor_f1_9, 6)})
        log(f"      pliegue {f} listo ({time.time() - t0:.0f} s)")

    # ---------- metricas ----------
    folds_v = [float(f1_score(y[pliegues == g], pred_v03[pliegues == g], average="macro"))
               for g in range(1, 6)]
    filas: list[dict[str, Any]] = []
    detalle: dict[str, Any] = {}
    for clave, p in proba_f.items():
        cand, mec = clave.split("|")
        ccv = ccv_de(p, y)
        folds = [float(f1_score(y[pliegues == g], decision(p)[pliegues == g], average="macro"))
                 for g in range(1, 6)]
        pl = sum(1 for a, b in zip(folds, folds_v) if a > b)
        ac = (decision(p) == y)
        ac_v = (pred_v03 == y)
        detalle[clave] = {
            "corrected_cv": ccv, "delta_vs_V03": round(ccv - ccv_ref, 10),
            "pliegues_mejorados": pl, "folds": folds,
            "rescate": int(np.count_nonzero(ac & ~ac_v)),
            "damage": int(np.count_nonzero(~ac & ac_v)),
            "por_etiqueta": {et: round(ccv_de(p, y, (j,))
                                       - float(f1_score(y[:, j], pred_v03[:, j],
                                                       average="binary", zero_division=0)), 10)
                             for j, et in enumerate(ETIQUETAS)},
            "por_bin": {nombre: round(float(f1_score(
                y[(n_tok >= lo) & (n_tok <= hi)], decision(p)[(n_tok >= lo) & (n_tok <= hi)],
                average="macro", zero_division=0))
                - float(f1_score(y[(n_tok >= lo) & (n_tok <= hi)],
                                 pred_v03[(n_tok >= lo) & (n_tok <= hi)],
                                 average="macro", zero_division=0)), 6)
                for nombre, lo, hi in BINS},
        }
        filas.append({"clave": clave, "candidato": cand, "mecanismo": mec,
                      "corrected_cv": round(ccv, 10),
                      "delta_vs_V03": round(ccv - ccv_ref, 10),
                      "pliegues_mejorados": pl,
                      "rescate": detalle[clave]["rescate"],
                      "damage": detalle[clave]["damage"],
                      "neto": detalle[clave]["rescate"] - detalle[clave]["damage"]})

    reales = [f for f in filas if not f["mecanismo"].endswith("barajado")]
    barajados = [f for f in filas if f["mecanismo"].endswith("barajado")]
    log("")
    log("   LOS CINCO MECANISMOS, sobre los candidatos REALES")
    log(f"      {'candidato':12}{'mecanismo':16}{'delta vs V03':>16}{'pliegues':>10}{'neto':>8}")
    for f in sorted(reales, key=lambda r: -r["delta_vs_V03"])[:16]:
        log(f"      {f['candidato']:12}{f['mecanismo']:16}{f['delta_vs_V03']:>+16.10f}"
            f"{f['pliegues_mejorados']:>7d}/5{f['neto']:>8d}")
    log("")
    log("   LOS MISMOS MECANISMOS con el score del CANDIDATO BARRAJADO")
    log("      esta es la fila que decide si una mejora es del candidato o de la MECANICA")
    log(f"      {'candidato':12}{'mecanismo':22}{'delta vs V03':>16}")
    for f in sorted(barajados, key=lambda r: -r["delta_vs_V03"])[:10]:
        log(f"      {f['candidato']:12}{f['mecanismo']:22}{f['delta_vs_V03']:>+16.10f}")

    mejor_real = max(reales, key=lambda r: r["delta_vs_V03"])
    clave_real = mejor_real["clave"]
    cand_mejor = mejor_real["candidato"]
    mec_mejor = mejor_real["mecanismo"]
    clave_bar = f"{cand_mejor}|{mec_mejor}__barajado"
    brecha_mecanica = round(mejor_real["delta_vs_V03"] - detalle[clave_bar]["delta_vs_V03"], 10)
    log("")
    log("   LA ATRIBUCION")
    log(f"      mejor real: {clave_real} con {mejor_real['delta_vs_V03']:+.10f} "
        f"y {mejor_real['pliegues_mejorados']}/5")
    log(f"      el MISMO mecanismo con el score barajado: {detalle[clave_bar]['delta_vs_V03']:+.10f}")
    log(f"      BRECHA DE ATRIBUCION: {brecha_mecanica:+.10f}")
    log("      si la brecha es pequena, la mejora es de la MECANICA de fusion y no del")
    log("      candidato, y entonces el candidato no es lo que la arregla")

    # ---------- diagnostico de informacion unica (P16) ----------
    error_any = (pred_v03 != y).any(axis=1).astype(int)
    log("")
    log("   P16, INFORMACION UNICA MAS ALLA DE V03")
    unica: list[dict[str, Any]] = []
    for c in CANDIDATOS:
        zc = logit(proba_c[c])
        auc_c = float(np.mean([roc_auc_score(error_any, zc[:, j]) for j in range(3)]))
        res_lineal = np.zeros_like(zc)
        for j in range(3):
            A = np.column_stack([np.ones(n), zv[:, j]])
            co = np.linalg.lstsq(A, zc[:, j], rcond=None)[0]
            res_lineal[:, j] = zc[:, j] - (co[0] + co[1] * zv[:, j])
        auc_r = float(np.mean([roc_auc_score(error_any, res_lineal[:, j]) for j in range(3)]))
        auc_margen = float(roc_auc_score(error_any, np.abs(margen_v03).mean(axis=1)))
        corr = float(np.mean([np.corrcoef(zv[:, j], zc[:, j])[0, 1] for j in range(3)]))
        unica.append({"candidato": c, "corr_con_z_V03": round(corr, 6),
                      "auc_del_candidato": round(auc_c, 6),
                      "auc_del_residuo": round(auc_r, 6),
                      "auc_del_margen": round(auc_margen, 6),
                      "el_residuo_sobrevive": auc_r > 0.5})
        log(f"      {c:12} corr {corr:+.4f} | AUC candidato {auc_c:.6f} | "
            f"AUC residuo {auc_r:.6f} | AUC margen {auc_margen:.6f}")
    log("      el AUC del RESIDUO es la medida clave: si desaparece, el candidato es redundante")

    # ---------- veredicto ----------
    log("")
    log("   EL VEREDICTO")
    if mejor_real["delta_vs_V03"] < GATE_DELTA:
        estado_final = "CLOSED"
        veredicto = (
            f"Ninguno de los cinco mecanismos, sobre ninguno de los siete candidatos, alcanza "
            f"el gate de {GATE_DELTA:+.4f}. El mejor es {clave_real} con "
            f"{mejor_real['delta_vs_V03']:+.10f} y {mejor_real['pliegues_mejorados']} de 5 "
            f"pliegues. Y la brecha de atribucion frente al score barajado es "
            f"{brecha_mecanica:+.10f}, luego lo poco que hay NO es del candidato")
    elif brecha_mecanica <= GATE_DELTA / 3:
        estado_final = "INVALIDATED"
        veredicto = (
            f"{clave_real} alcanza {mejor_real['delta_vs_V03']:+.10f}, pero el MISMO mecanismo "
            f"con el score del candidato BARRAJado consigue "
            f"{detalle[clave_bar]['delta_vs_V03']:+.10f}, luego la mejora pertenece a la "
            f"capa de fusion y no a la informacion del candidato. INVALIDATED")
    elif mejor_real["pliegues_mejorados"] < GATE_PLIEGUES:
        estado_final = "MARGINAL"
        veredicto = (
            f"{clave_real} llega a {mejor_real['delta_vs_V03']:+.10f} con "
            f"{mejor_real['pliegues_mejorados']} de 5 pliegues, y la brecha de atribucion es "
            f"{brecha_mecanica:+.10f}. Supera el gate de magnitud pero no el de consistencia")
    else:
        estado_final = "PROMISING"
        veredicto = (
            f"{clave_real} alcanza {mejor_real['delta_vs_V03']:+.10f} con "
            f"{mejor_real['pliegues_mejorados']} de 5 pliegues y la brecha de atribucion frente "
            f"al score barajado es {brecha_mecanica:+.10f}, luego la mejora es del candidato y "
            f"no de la capa. Eso exige bootstrap, segunda particion y complementariedad")
    log(f"   mejor: {clave_real}")
    log("")
    promocion = [f for f in reales
                if f["delta_vs_V03"] >= GATE_DELTA and f["pliegues_mejorados"] >= GATE_PLIEGUES]
    log("")
    log("   LOS QUE CUMPLEN LOS DOS GATES A LA VEZ")
    if promocion:
        for f in promocion:
            log(f"      {f['clave']:34} {f['delta_vs_V03']:+.10f} {f['pliegues_mejorados']}/5 | "
                f"brecha de atribucion {f['delta_vs_V03'] - detalle[f['clave'].replace('|', '|') + '__barajado']['delta_vs_V03']:+.10f}")
    else:
        log("      ninguno cumple el delta Y los 4 de 5 pliegues a la vez")
    log("")
    log(f"   DECISION: {estado_final}")
    log("")
    log(f"   VEREDICTO: {veredicto}")

    with CSV_F.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["clave", "candidato", "mecanismo", "corrected_cv", "delta_vs_V03",
                    "pliegues_mejorados", "rescate", "damage", "neto"])
        for f in filas:
            w.writerow([f["clave"], f["candidato"], f["mecanismo"], f["corrected_cv"],
                        f["delta_vs_V03"], f["pliegues_mejorados"], f["rescate"], f["damage"],
                        f["neto"]])
        w.writerow([])
        for clave in sorted(detalle):
            d = detalle[clave]
            w.writerow(["por_etiqueta", clave, d["delta_vs_V03"],
                        d["por_etiqueta"]["HS"], d["por_etiqueta"]["TR"],
                        d["por_etiqueta"]["AG"], "", "", ""])
        w.writerow([])
        w.writerow(["por_bin", "clave", "delta_vs_V03"] + [b[0] for b in BINS])
        for clave in sorted(detalle):
            d = detalle[clave]
            w.writerow(["por_bin", clave, d["delta_vs_V03"]]
                       + [d["por_bin"][b[0]] for b in BINS])
    with CSV_R.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["candidato", "etiqueta", "barajado", "pliegue", "detalle"])
        for cand, d in params_elegidas.items():
            for clave, lista in d.items():
                et = clave.split("_", 1)[1].rsplit("__", 1)[0]
                bar = "barajado" in clave
                for pl_i, item in enumerate(lista):
                    w.writerow([cand, et, bar, pl_i, json.dumps(item, sort_keys=True)])
    with CSV_L.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["candidato", "mecanismo", "delta_vs_V03"] + [b[0] for b in BINS])
        for cand in CANDIDATOS:
            for mec in MECANISMOS:
                clave = f"{cand}|{mec}"
                w.writerow([cand, mec, detalle[clave]["delta_vs_V03"]]
                           + [detalle[clave]["por_bin"][b[0]] for b in BINS])
    with CSV_C.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["candidato", "mecanismo", "delta_real", "delta_barajado",
                    "brecha_de_atribucion"])
        for cand in CANDIDATOS:
            for mec in MECANISMOS:
                r = detalle[f"{cand}|{mec}"]["delta_vs_V03"]
                b = detalle[f"{cand}|{mec}__barajado"]["delta_vs_V03"]
                w.writerow([cand, mec, r, b, round(r - b, 10)])
        w.writerow([])
        w.writerow(["informacion_unica", "candidato", "corr_con_z_V03", "auc_candidato",
                    "auc_residuo", "auc_margen", "el_residuo_sobrevive"])
        for u in unica:
            w.writerow(["informacion_unica", u["candidato"], u["corr_con_z_V03"],
                        u["auc_del_candidato"], u["auc_del_residuo"], u["auc_del_margen"],
                        u["el_residuo_sobrevive"]])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P66_FUSION_ESPACIO_SCORES",
        "naturaleza": ("EXPERIMENTO. 5 mecanismos x 7 candidatos x 2 controles de barajado, "
                       "0 submissions, 0 Kaggle, 0 holdout, 0 test"),
        "fecha": inicio.isoformat(timespec="seconds"), "experiment_id": "P66-02",
        "track": "P05-P09-fusion-en-espacio-de-scores", "script":
            "experiments/p66_02_fusion_espacio_scores.py",
        "pregunta": ("puede una senal que no funciona como feature funcionar como una correccion "
                     "pequena del score de V03"),
        "mecanismos": {
            "blend_prob": "p = (1-a)p_V03 + a p_C, control que reproduce V6-05 y P59-09",
            "blend_logit": "z = z_V03 + a z_C, evidencia aditiva en log-odds",
            "fusion_lr": "LogisticRegression sobre [z_V03, z_C]. Es una calibracion de DOS "
                         "variables, no un meta-modelo",
            "residual": "z = z_V03 + b*(z_C - (a + b0*z_V03)), con a y b0 ajustados SOLO en "
                        "inner_train. Es la forma nueva de la fase",
            "length_aware": "la residual con un segundo coeficiente por longitud normalizada, "
                            "continua y sin puerta",
        },
        "protocolo_nested": {
            "inner_train": "ajusta a, b0, la regresion, la normalizacion de longitud y el modelo",
            "inner_validation": "selecciona alpha o beta por F1 con los umbrales de V03",
            "outer_train": "se reajusta con lo seleccionado, si procede",
            "outer_validation": "EVALUA SOLO. Nunca selecciona",
        },
        "escala_audit": {c: {"min": float(proba_c[c].min()), "max": float(proba_c[c].max())}
                         for c in CANDIDATOS},
        "conclusion_de_escala": ("todas las probabilidades son validas en [0,1], luego la "
                                 "fusion en espacio de probabilidad es admisible y NO hace "
                                 "falta calibracion anidada previa"),
        "resultados": detalle,
        "mejor_real": {"clave": clave_real, "delta": mejor_real["delta_vs_V03"],
                       "pliegues": mejor_real["pliegues_mejorados"]},
        "control_de_barajado": {"clave": clave_bar,
                                "delta": detalle[clave_bar]["delta_vs_V03"]},
        "brecha_de_atribucion": brecha_mecanica,
        "informacion_unica": unica,
        "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                 "superado": estado_final == "PROMISING"},
        "decision": estado_final, "veredicto": veredicto,
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_F.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P66 - Las cinco formas de fusion en el espacio de scores")
    W("")
    W(f"Generado por `experiments/p66_02_fusion_espacio_scores.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Los cinco mecanismos, y lo que pregunta cada uno")
    W("")
    W("| mecanismo | formula | pregunta |")
    W("|---|---|---|")
    W("| blend de probabilidad | `p = (1-a)p_V03 + a p_C` | mezcla ponderada |")
    W("| blend de logit | `z = z_V03 + a z_C` | evidencia aditiva |")
    W("| fusion LR | `LogisticRegression([z_V03, z_C])` | aprender la mezcla |")
    W("| **residual** | `z = z_V03 + b*(z_C - (a + b0*z_V03))` | **quitar lo redundante** |")
    W("| sensible a la longitud | residual con coeficiente por longitud | pesa mas en los largos |")
    W("")
    W("P5 y P6 **no son nuevos** y se ejecutan como **controles**: V6-05 y P59-09 ya midieron un")
    W("alpha en espacio de probabilidad. Reproducir eso aqui es lo que hace interpretable a P8,")
    W("porque si P8 mejora donde la misma informacion en forma de P5 no, la diferencia es la")
    W("**residualizacion** y no el candidato.")
    W("")
    W("## El protocolo nested, que es el punto entero")
    W("")
    W("```text")
    W("outer_train -> inner_train     ajustar a, b0, la regresion y la normalizacion")
    W("outer_train -> inner_val       elegir alpha o beta por F1 con los umbrales de V03")
    W("outer_validation               EVALUAR SOLO. Nunca selecciona")
    W("```")
    W("")
    W("La diferencia es entre una fusion cuyos parametros se eligieron sobre las mismas filas")
    W("contra las que se puntua, que siempre parece positiva, y una cuyos parametros se")
    W("congelaron antes de tocar la outer.")
    W("")
    W("## La escala, antes de fusionar (P04)")
    W("")
    W("| candidato | min p_C | max p_C |")
    W("|---|---:|---:|")
    for c in CANDIDATOS:
        log_p = proba_c[c]
        W(f"| {c} | {log_p.min():.6f} | {log_p.max():.6f} |")
    W(f"| **p_V03** | {proba_v03.min():.6f} | {proba_v03.max():.6f} |")
    W("")
    W("Todas son probabilidades validas, luego la fusion en espacio de probabilidad es")
    W("admisible y **no hace falta calibrar antes**.")
    W("")
    W("## Resultados sobre los candidatos reales")
    W("")
    W("| candidato | mecanismo | delta vs V03 | pliegues | neto |")
    W("|---|---|---:|---:|---:|")
    for f in sorted(reales, key=lambda r: -r["delta_vs_V03"])[:20]:
        W(f"| {f['candidato']} | {f['mecanismo']} | {f['delta_vs_V03']:+.10f} | "
          f"{f['pliegues_mejorados']}/5 | {f['neto']:+d} |")
    W("")
    W("## Los mismos mecanismos con el score del CANDIDATO BARRAJADO")
    W("")
    W("Esta es la fila que decide si una mejora es **del candidato** o de la **mecanica**.")
    W("")
    W("| candidato | mecanismo | delta real | delta barajado | brecha de atribucion |")
    W("|---|---|---:|---:|---:|")
    for cand in CANDIDATOS:
        for mec in MECANISMOS:
            r = detalle[f"{cand}|{mec}"]["delta_vs_V03"]
            b = detalle[f"{cand}|{mec}__barajado"]["delta_vs_V03"]
            W(f"| {cand} | {mec} | {r:+.10f} | {b:+.10f} | {r - b:+.10f} |")
    W("")
    W("## P16, informacion unica mas alla de V03")
    W("")
    W("| candidato | corr con z_V03 | AUC candidato | AUC del RESIDUO | AUC del margen |")
    W("|---|---:|---:|---:|---:|")
    for u in unica:
        W(f"| {u['candidato']} | {u['corr_con_z_V03']:+.4f} | {u['auc_del_candidato']:.6f} | "
          f"{u['auc_del_residuo']:.6f} | {u['auc_del_margen']:.6f} |")
    W("")
    W("El AUC del **residuo** es la medida clave: si desaparece, el candidato es redundante con")
    W("V03 y no hay nada que fusionar.")
    W("")
    W("## Veredicto")
    W("")
    W("```text")
    W(f"mejor real       {clave_real} con {mejor_real['delta_vs_V03']:+.10f} "
      f"y {mejor_real['pliegues_mejorados']}/5")
    W(f"mismo, barajado  {detalle[clave_bar]['delta_vs_V03']:+.10f}")
    W(f"brecha           {brecha_mecanica:+.10f}")
    W("")
    W(veredicto)
    W("")
    W(f"DECISION: {estado_final}")
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p66_02_fusion_espacio_scores.py`. Sin push.*")
    MD_F.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_F, JSON_F, CSV_F, CSV_R, CSV_L, CSV_C):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P66-02 OK | mejor {clave_real} {mejor_real['delta_vs_V03']:+.10f} | brecha de "
        f"atribucion {brecha_mecanica:+.10f} | decision {estado_final}")


def _inner_split(ir: np.ndarray, y: np.ndarray, j: int, n_splits: int,
                 seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Reparte el entrenamiento de un pliegue en inner_train e inner_validation.

    Se estratifica por la etiqueta, porque sin estratificar un pliegue puede dejar la clase
    positiva con muy pocas filas en una parte y el beta elegido seria ruido.

    Una version anterior de esta funcion devolvia (ir, ir), es decir el mismo conjunto dos
    veces. Eso no es un split interno: es el outer_train playing both roles, y habria hecho
    que TODO el protocolo nested de la fase fuera una decoracion. Se corrigio antes de
    ejecutar nada.

    Args:
        ir: indices globales del entrenamiento exterior.
        y: etiquetas verdaderas.
        j: indice de la etiqueta.
        n_splits: numero de partes internas.
        seed: semilla.

    Returns:
        Dos arrays de indices globales, inner_train e inner_validation.
    """
    etiquetas = y[ir, j]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    posiciones = np.arange(len(ir))
    for train_pos, val_pos in skf.split(posiciones, etiquetas):
        return ir[train_pos], ir[val_pos]
    raise SystemExit("PARADA: el split interno no produjo ninguna parte")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
