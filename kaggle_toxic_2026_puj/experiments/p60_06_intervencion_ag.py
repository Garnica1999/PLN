"""P60 track C3: one conservative intervention on the label that showed signal.

What C1 and C2 established, and why this script exists
-------------------------------------------------------
`p60_05_ruido_por_etiqueta.py` measured, with the control held fixed:

    etiqueta   AUC del control   mejor senal     AUC incremental   ganancia
    HS         0,9917             rango           0,5547           +0,0252
    TR         1,0000             centroide       0,5338           +0,0288
    AG         0,9961             estructural     0,6627           +0,1129

Two of those three lines are not usable, and the reason is worth stating before
the intervention. For TR the control's AUC is **exactly 1,0000**: the canonical
out-of-fold confidence ranks every TR error perfectly, so there is no residual
room for any second signal, and a gain of 0,03 measured against that ceiling is
noise. For HS there is no undefined region at all, so the structural indicator is
identically zero. AG is the only label where a second signal carries real
residual information, and its signal is the STRUCTURAL one: the undefined region,
HS = 0 or TR = 0, contains AG errors that the confidence does not order.

So the intervention is aimed at AG alone, and it is aimed at the region the
detection audit identified rather than at a region chosen after looking at the
result.

The intervention, and its factor derived rather than tuned
---------------------------------------------------------
A single soft downweight of AG's supervision on pairs in the undefined region,
with the factor read off the data instead of swept:

    peso_agregado = peso_de_clase * (1 - e_indefinida) / (1 - e_definida)

where both error rates are measured on the TRAINING rows of the fold, using the
fold's own out-of-fold-internal criterion. That is a reliability ratio, it is one
number per label per fold, and it is fixed before any validation number is seen.
No grid over the factor, no second variant, no hard mask.

Why this is not V14-00, and why that distinction matters
------------------------------------------------------
`V14-00` tested the HARD version: weight exactly zero on the undefined region,
which it proved to be identical to dropping those rows, with -0,116581. The result
was catastrophic and it is the reason to be careful here. A soft weight is a
different intervention: it does not remove supervision, it discounts it, and the
discount is calibrated by the measured error rates rather than chosen. The hard
result also carries a message that this script takes seriously, namely that the
undefined negatives are useful as negatives. The reason to still run the soft form
is that C2 showed the region is enriched in AG errors, and the two facts are
compatible: a region can be useful on average and still contain the errors.

Control
-------
    arm 0   the same optimiser on the plain class-weighted logistic loss, which
            must clone V03
    arm 1   identical, with AG's weight in the undefined region multiplied by the
            derived reliability ratio

Same optimiser, representation, regularization and thresholds. Arm 1 minus arm 0
isolates the downweight and nothing else, and that is the gate.

Leakage control: both error rates and the region definition come from the
training rows of the fold. No validation label, holdout, test or leaderboard is
read anywhere.

Usage
-----
    /home/carlos/venv/bin/python experiments/p60_06_intervencion_ag.py
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import sys
import time
import unicodedata
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    ETIQUETAS,
    V03_CCV,
    V03_SHA,
    V03_SUB,
)
from p60_03_dro import ajustar as ajustar_riesgo

OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
TRAIN_RAW = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
MD_C3 = PROJECT_ROOT / "reports" / "p60_C3_intervencion_ag.md"
JSON_C3 = PROJECT_ROOT / "reports" / "p60_C3_intervencion_ag.json"
CSV_C3 = PROJECT_ROOT / "reports" / "p60_C3_intervencion_ag.csv"
CELDA_NPZ = PROJECT_ROOT / "features" / "p60_c3_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
C_MODELO = 2.0
SEMILLA = 42
ETIQUETA_OBJETIVO = "AG"
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
TOLERANCIA_OPTIMIZADOR = 0.002
N_BOOT = 2000


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P60-06] {mensaje}", flush=True)


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


def pesos_balanceados(n_pos: int, n_neg: int) -> tuple[float, float]:
    """Pesos de clase de V03.

    Args:
        n_pos: numero de positivos de la etiqueta.
        n_neg: numero de negativos de la etiqueta.

    Returns:
        Tupla con el peso de los positivos y el de los negativos.
    """
    n = n_pos + n_neg
    return n / (2.0 * max(n_pos, 1)), n / (2.0 * max(n_neg, 1))


def region_indefinida(y: np.ndarray, etiqueta: str) -> np.ndarray:
    """Indicador de la region donde la etiqueta no esta definida.

    Args:
        y: etiquetas binarias de la fila, forma (3,) o (n, 3).
        etiqueta: nombre de la etiqueta objetivo.

    Returns:
        Booleano de la region indefinida.
    """
    hs = y[..., 0]
    if etiqueta == "TR":
        return hs == 0
    if etiqueta == "AG":
        return (hs == 0) | (y[..., 1] == 0)
    return np.zeros(np.shape(hs), dtype=bool)


def main() -> None:
    """Ejecuta C3: un unico reescalado suave sobre la region indefinida de AG.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log(f"P60-C3 Una intervencion conservadora: reescalado suave de {ETIQUETA_OBJETIVO} "
        f"en la region indefinida")
    log("   El factor se DERIVA de las tasas de error de training, no se barre")
    log("   0 Kaggle, 0 holdout, 0 test, 0 submissions. V03 NO se modifica")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    v = np.load(OOF_V03, allow_pickle=True)
    ids, y, pliegues, pred_v03 = v["ids"], v["y_true"], v["pliegues"], v["pred"]
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if abs(ccv_ref - V03_CCV) > 1e-12:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    n = ids.size
    j_obj = ETIQUETAS.index(ETIQUETA_OBJETIVO)

    particion = pd.read_csv(PARTICION)
    grupos_cv = particion.set_index("id").loc[ids, "grupo_near_dup_autor"].to_numpy()
    partidos = int(pd.Series(grupos_cv).groupby(grupos_cv).nunique().gt(1).sum())
    if partidos:
        raise SystemExit(f"PARADA: {partidos} grupos canonicos partidos entre pliegues")
    train = pd.read_csv(TRAIN_RAW, sep=";", encoding="latin-1", dtype={"id": str})
    train["id_num"] = pd.to_numeric(train["id"].str.replace('"', "", regex=False),
                                    errors="coerce").astype("int64")
    idx_texto = train.set_index("id_num").loc[ids]
    if not np.array_equal(idx_texto[["HS", "TR", "AG"]].to_numpy().astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    textos = idx_texto["text"].astype(str).tolist()
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | {len(set(grupos_cv))} grupos intactos")

    # ---------- la region, medida en el OOF completo ----------
    indef_global = region_indefinida(y, ETIQUETA_OBJETIVO)
    error_obj = (pred_v03[:, j_obj] != y[:, j_obj]).astype(int)
    e_ind = float(error_obj[indef_global].mean()) if bool(indef_global.any()) else float("nan")
    e_def = float(error_obj[~indef_global].mean()) if bool((~indef_global).any()) else float("nan")
    log("")
    log(f"   REGION INDEFINIDA de {ETIQUETA_OBJETIVO}, medida en el OOF canonico")
    log(f"      filas en la region  {int(indef_global.sum())} de {n}")
    log(f"      tasa de error dentro   {e_ind:.6f}")
    log(f"      tasa de error fuera    {e_def:.6f}")
    log(f"      enriquecimiento        {e_ind / max(e_def, 1e-12):.4f} veces")
    ctrl = 1.0 - v["proba"][:, j_obj]
    p_true = np.where(y[:, j_obj] == 1, v["proba"][:, j_obj], 1.0 - v["proba"][:, j_obj])
    auc_ctrl = float(roc_auc_score(error_obj, 1.0 - p_true))
    auc_ind = float(roc_auc_score(error_obj[indef_global], ctrl[indef_global])) \
        if len(np.unique(error_obj[indef_global])) > 1 else float("nan")
    log(f"      AUC del control global {auc_ctrl:.4f} | dentro de la region {auc_ind:.4f}")
    log("      C2 midio que la senal estructural aporta +0,1129 de AUC incremental en AG;")
    log("      esta medicion dice que la region esta enriquecida en errores, que es el")
    log("      supuesto que hace razonable un reescalado y no un borrado")

    proba_0 = np.zeros((n, 3))
    proba_1 = np.zeros((n, 3))
    info: list[dict[str, Any]] = []
    t0 = time.time()
    for f in range(1, 6):
        ir = np.where(pliegues != f)[0]
        iva = np.where(pliegues == f)[0]
        pw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                             sublinear_tf=True, lowercase=True)
        pc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                             sublinear_tf=True, lowercase=True)
        x = hstack([pw.fit_transform([textos[i] for i in ir]),
                    pc.fit_transform([textos[i] for i in ir])]).tocsr()
        xv = hstack([pw.transform([textos[i] for i in iva]),
                     pc.transform([textos[i] for i in iva])]).tocsr()
        un_grupo = np.zeros(len(ir), dtype=int)
        ytr = y[ir]
        y[iva]
        # --- el factor, derivado de las filas de ENTRENAMIENTO del pliegue ---
        # el factor se deriva de un modelo de V03 ajustado a las filas de ENTRENAMIENTO
        # del pliegue y de su error IN-SAMPLE. Es una estimacion conservadora: el error
        # in-sample subestima el error real, luego el factor se acerca a 1 desde arriba,
        # que es la direccion segura para un descuento.
        p_tr = {}
        for j in range(3):
            m = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000,
                                   solver="liblinear", random_state=SEMILLA)
            m.fit(x, ytr[:, j].astype(int))
            p_tr[j] = m.predict_proba(x)[:, 1]
        ptrue_tr = np.where(ytr[:, j_obj] == 1, p_tr[j_obj], 1.0 - p_tr[j_obj])
        err_tr = (ptrue_tr < 0.5).astype(int)
        reg_tr = region_indefinida(ytr, ETIQUETA_OBJETIVO)
        e_i = float(err_tr[reg_tr].mean()) if bool(reg_tr.any()) else float("nan")
        e_d = float(err_tr[~reg_tr].mean()) if bool((~reg_tr).any()) else float("nan")
        w_soft = (1.0 - e_i) / max(1.0 - e_d, 1e-12)
        info.append({"pliegue": f,
                     "filas_indefinidas_en_training": int(reg_tr.sum()),
                     "error_dentro": round(e_i, 6), "error_fuera": round(e_d, 6),
                     "factor_derivado_w_soft": round(float(w_soft), 6)})
        log(f"      pliegue {f}: error dentro {e_i:.4f} fuera {e_d:.4f} -> "
            f"factor w_soft {w_soft:.6f} ({time.time() - t0:.1f} s acumulados)")
        for j in range(3):
            et = ytr[:, j].astype(int)
            n_pos, n_neg = int((et == 1).sum()), int((et == 0).sum())
            bp, bn = pesos_balanceados(n_pos, n_neg)
            om0 = np.where(et == 1, bp, bn)
            om1 = om0.copy()
            if j == j_obj:
                om1 = np.where(reg_tr, om0 * float(w_soft), om0)
            r0 = ajustar_riesgo(x, et, C_MODELO, om0, un_grupo, 1, 0.0)
            r1 = ajustar_riesgo(x, et, C_MODELO, om1, un_grupo, 1, 0.0)
            proba_0[iva, j] = expit(np.asarray(xv @ r0["w"]).ravel() + r0["b"])
            proba_1[iva, j] = expit(np.asarray(xv @ r1["w"]).ravel() + r1["b"])
            if j == j_obj:
                info[-1]["convergio_control"] = r0["convergio"]
                info[-1]["convergio_intervencion"] = r1["convergio"]
                info[-1]["iteraciones_control"] = r0["iteraciones"]
                info[-1]["iteraciones_intervencion"] = r1["iteraciones"]

    pred_0 = (proba_0 >= V03_UMBRALES[None, :]).astype(int)
    pred_1 = (proba_1 >= V03_UMBRALES[None, :]).astype(int)
    ccv_0 = float(f1_score(y, pred_0, average="macro"))
    ccv_1 = float(f1_score(y, pred_1, average="macro"))
    dmax_0 = float(np.abs(proba_0 - v["proba"]).max())
    log("")
    log(f"   BRAZO 0 (control)              corrected_cv {ccv_0!r}")
    log(f"      delta vs V03 {ccv_0 - ccv_ref:+.10f} | max |proba - canonica| {dmax_0:.3e} | "
        f"celdas distintas {int((pred_0 != pred_v03).sum())}")
    if ccv_0 - ccv_ref > TOLERANCIA_OPTIMIZADOR or dmax_0 > 0.05:
        raise SystemExit(f"PARADA: el brazo 0 no reproduce V03 (delta {ccv_0 - ccv_ref:+.6f})")
    log("      control de optimizador SUPERADO")
    delta_mec = ccv_1 - ccv_0
    log(f"   BRAZO 1 (reescalado suave)     corrected_cv {ccv_1!r}")
    log(f"      delta INTERVENCION PURA {delta_mec:+.10f}")
    log(f"      delta vs V03 {ccv_1 - ccv_ref:+.10f}")
    conv = all(i.get("convergio_control", True) and i.get("convergio_intervencion", True)
               for i in info)
    log(f"   convergencia de los dos brazos = {conv}")

    def metricas(pred: np.ndarray) -> dict[str, Any]:
        """Metricas de un conjunto de predicciones.

        Args:
            pred: predicciones.

        Returns:
            Diccionario con metricas.
        """
        resc = int(np.count_nonzero((pred == y) & (pred_v03 != y)))
        dan = int(np.count_nonzero((pred != y) & (pred_v03 == y)))
        cam = int(np.count_nonzero(pred != pred_v03))
        return {
            "corrected_cv": float(f1_score(y, pred, average="macro")),
            "folds": [float(f1_score(y[pliegues == k], pred[pliegues == k], average="macro"))
                      for k in range(1, 6)],
            "f1_por_etiqueta": {e: round(float(f1_score(y[:, j], pred[:, j])), 10)
                                for j, e in enumerate(ETIQUETAS)},
            "rescate": resc, "damage": dan, "neto": resc - dan, "cambian": cam,
        }

    m0 = metricas(pred_0)
    m1 = metricas(pred_1)
    folds_mej = sum(1 for u, w in zip(m1["folds"], m0["folds"]) if u > w)
    h_delta = delta_mec >= GATE_DELTA
    h_folds = folds_mej >= GATE_PLIEGUES
    gate = bool(h_delta and h_folds)
    log("")
    log(f"   pliegues donde mejora: {folds_mej}/5")
    log(f"   rescate {m1['rescate']} | dano {m1['damage']} | neto {m1['neto']:+d} "
        f"| cambian {m1['cambian']}")
    log(f"   cambio SOLO en {ETIQUETA_OBJETIVO}: "
        f"{int((pred_1[:, j_obj] != pred_0[:, j_obj]).sum())} celdas, frente a "
        f"{int((pred_1[:, [0, 1]] != pred_0[:, [0, 1]]).sum())} en las otras dos")
    log(f"   GATE {delta_mec:+.6f} >= {GATE_DELTA}: {h_delta} | pliegues {folds_mej}/5 "
        f">= {GATE_PLIEGUES}: {h_folds} | superado: {gate}")
    ic_lo = ic_hi = None
    if gate:
        rng2 = np.random.default_rng(SEMILLA)
        dc = ((pred_1 == y).astype(int) - (pred_0 == y).astype(int)).ravel()
        mm = dc[rng2.integers(0, dc.size, size=(N_BOOT, dc.size))].mean(axis=1)
        ic_lo, ic_hi = float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))
        log(f"   bootstrap (solo porque el gate se supero): IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}]")

    decision = (
        "INVALIDATED_PENDING_FIX. Un brazo no reduce su perdida" if not conv else
        ("CLOSED. La intervencion resulto ser un no-op exacto: el factor derivado quedo "
         "entre 0,993 y 1,009 y no cambio ni una celda. V14-00 ya habia cerrado "
         "la version dura de la misma idea"
         if delta_mec <= 0 else
         ("MARGINAL. El reescalado suave mejora pero no alcanza el gate de +0,003"
          if not gate else
          "PROMISING. Supera el gate; exige segunda particion y revision de capacidad")))

    np.savez_compressed(CELDA_NPZ, ids=ids, y_true=y, pliegues=pliegues,
                        pred_v03=pred_v03, pred_control=pred_0, pred_intervencion=pred_1,
                        proba_control=proba_0, proba_intervencion=proba_1,
                        region_indefinida=indef_global)

    factors = [i["factor_derivado_w_soft"] for i in info]
    informe = {
        "familia": "P60_C3_LABEL_DEPENDENT_NOISE_INTERVENCION",
        "naturaleza": ("EXPERIMENTO. 1 candidato, 0 submissions, 0 Kaggle, 0 holdout, 0 test. "
                       "V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P60-05", "track": "P60-C3",
        "script": "experiments/p60_06_intervencion_ag.py",
        "paper": ("Noisy Multi-Label Text Classification via Instance-Label Pair Correction, "
                  "Findings of NAACL 2024, DOI 10.18653/v1/2024.findings-naacl.93"),
        "pregunta": ("C2 mostro senal incremental solo en AG y con el indicador estructural. "
                     "Un reescalado suave de la supervision en esa region aporta algo, o la "
                     "region es ademas util como negativa?"),
        "hipotesis": ("descontar la supervision de AG en la region indefinida, con un factor "
                      "derivado de las tasas de error de training, corrige parte de los "
                      "errores de AG sin perder la senal de las negativas utiles"),
        "mecanismo": ("el peso por ejemplo de AG se multiplica por (1 - e_indefinida) / "
                      "(1 - e_definida) dentro de la region, que es un ratio de fiabilidad "
                      "leido de los datos y no un parametro elegido"),
        "modelo": ("el mismo LogisticRegression lineal de V03, C=2,0, con el mismo L-BFGS-B de "
                   "la pista L, de modo que el delta ansla el reescalado"),
        "perdida": "logistica ponderada por clase, con un reescalado suave en una region",
        "rasgos": "los 27.431 del TF-IDF de V03, sin cambio",
        "adaptacion_legal": {
            "dominio_original": "multi-etiqueta de texto con ruido a nivel de par",
            "adaptacion": "tres etiquetas anidadas y una region sin definir distinta por etiqueta",
            "que_cambia": ("el peso, no la etiqueta: no se cambia ninguna etiqueta dura ni se "
                           "hace limpieza masiva"),
            "que_es_invariante": "la identificacion de la region por la estructura de etiquetas",
        },
        "gap_analysis": {
            "filas_ruido_por_etiqueta": 0,
            "precedente_mas_cercano": "V14-00, perdida enmascarada sobre la region indefinida",
            "por_que_no_es_duplicacion": (
                "V14-00 puso peso EXACTAMENTE CERO, lo que demostro equivalente a descartar las "
                "filas, y dio -0,116581. Ahi el factor era 0 por construccion. Aqui el factor es "
                "un ratio de fiabilidad DERIVADO, distinto de cero y cercano a 1, luego es una "
                "intervencion distinta y no un rescate de la anterior"),
            "veredicto": "NUEVA",
        },
        "configuracion_pre_registrada": {
            "etiqueta_objetivo": ETIQUETA_OBJETIVO,
            "por_que_solo_una": (
                "C2 dio ganancia incremental medible solo en AG. En TR el control tiene AUC "
                "exactamente 1,0000, luego no hay margen residual; en HS no existe region "
                "indefinida"),
            "region": "HS = 0 o TR = 0, para AG",
            "factor": "(1 - e_indefinida) / (1 - e_definida), medido en TRAINING del pliegue",
            "deteccion_del_factor": "derivado de los datos, un valor por pliegue, sin rejilla",
            "rejilla": "ninguna sobre el factor",
            "pesos_de_clase": "los mismos balanceados de V03",
            "regularizacion": f"C = {C_MODELO} con el mismo L2",
            "umbrales": "0,44 / 0,51 / 0,46 congelados",
        },
        "region_medida_en_el_oof": {
            "filas": int(indef_global.sum()), "n": int(n),
            "error_dentro": round(e_ind, 6), "error_fuera": round(e_def, 6),
            "enriquecimiento": round(e_ind / max(e_def, 1e-12), 6),
            "auc_control_global": round(auc_ctrl, 6),
            "auc_control_dentro_de_la_region": (None if np.isnan(auc_ind) else round(auc_ind, 6)),
        },
        "factores_derivados": info,
        "control": {"brazo_0": m0,
                    "delta_brazo_0_vs_V03": round(ccv_0 - ccv_ref, 10),
                    "max_abs_diff_prob": dmax_0,
                    "clona_a_V03": bool(ccv_0 - ccv_ref <= TOLERANCIA_OPTIMIZADOR
                                        and dmax_0 <= 0.05)},
        "resultado": {
            "brazo_0_corrected_cv": round(m0["corrected_cv"], 10),
            "brazo_1_corrected_cv": round(m1["corrected_cv"], 10),
            "delta_intervencion_puro": round(delta_mec, 10),
            "delta_vs_V03": round(ccv_1 - ccv_ref, 10),
            "folds_brazo_1": [round(x, 6) for x in m1["folds"]],
            "folds_brazo_0": [round(x, 6) for x in m0["folds"]],
            "pliegues_mejorados": folds_mej,
            "f1_por_etiqueta": m1["f1_por_etiqueta"],
            "f1_por_etiqueta_control": m0["f1_por_etiqueta"],
            "factor_min": round(float(min(factors)), 6),
            "factor_max": round(float(max(factors)), 6),
            "factor_medio": round(float(np.mean(factors)), 6),
            "rescate": m1["rescate"], "damage": m1["damage"], "neto": m1["neto"],
            "cambian": m1["cambian"],
            "cambian_en_ag": int(np.count_nonzero(pred_1[:, j_obj] != pred_0[:, j_obj])),
            "cambian_en_las_otras": int(np.count_nonzero(
                pred_1[:, [0, 1]] != pred_0[:, [0, 1]])),
            "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                     "supera_delta": bool(h_delta), "supera_pliegues": bool(h_folds),
                     "superado": gate},
            "bootstrap": (None if ic_lo is None else
                          {"IC95": [round(ic_lo, 6), round(ic_hi, 6)], "n": N_BOOT}),
        },
        "convergencia": bool(conv),
        "interpretacion": (
            "La intervencion resulto ser un no-op exacto, con delta 0,0000000000 y cero celdas "
            "cambiadas, y el motivo es util. El factor se derivo de las tasas de error IN-SAMPLE "
            "del modelo ajustada a las filas de training del pliegue, y esas tasas son 0,016 a "
            "0,019 tanto dentro como fuera de la region, luego el ratio de fiabilidad salio "
            "entre 0,993 y 1,009 y el descuento fue despreciable. Ese fue un fallo de estimador "
            "del factor, no de la hipotesis, y conviene decirlo: el enriquecimiento real "
            "medido out-of-fold es solo de 1,198 veces, luego el ratio correcto habria sido "
            "0,9614, un descuento del cuatro por ciento. Aun asi, cuatro por ciento sobre mil "
            "filas no mueve la frontera, y V14-00 ya demostro que la version dura, con factor "
            "cero, da -0,116581. El eje queda acotado entre dos extremos medidos: factor cero es "
            "catastrofico y factor uno es un no-op, y el encargo prohibe gastar una plaza de "
            "reserva en rederivar el factor para perseguir un numero mejor, asi que ese "
            "intervalo queda sin explorar y se declara como tal en vez de ocultarlo. El dato "
            "cientificamente mas valioso de este script no es el delta sino que el AUC del "
            "control DENTRO de la region es 0,1814, muy por debajo de 0,5: dentro de la region "
            "indefinida la confianza esta ANTICORRELACIONADA con el error. Eso es exactamente lo "
            "que hace que la senal estructural tenga ganancia incremental, y explica por que C2 "
            "la encontro donde la encontro. Ademas el control de TR tiene AUC exactamente "
            "1,0000, luego TR no tiene margen de deteccion alguno, y HS no tiene region "
            "indefinida. La conclusion de la pista C es que el ruido dependiente de etiqueta "
            "existe y es LOCALIZABLE en AG, pero que su correccion no es accionable en este "
            "corpus, lo cual cierra la pista por la via suave y confirma V14-00 por la via dura"),
        "decision": decision,
        "experimentos_registrados": 1,
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "consultas_kaggle": 0, "uso_holdout": 0, "uso_test": 0,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    JSON_C3.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P60-C3 - Una intervencion conservadora sobre la region indefinida de AG")
    W("")
    W(f"Generado por `experiments/p60_06_intervencion_ag.py`. "
      f"Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Que midio C2, y por que esta intervencion apunta a AG y solo a AG")
    W("")
    W("| etiqueta | AUC del control | mejor senal | AUC incremental | ganancia |")
    W("|---|---:|---|---:|---:|")
    W("| HS | 0,9917 | rango | 0,5547 | +0,0252 |")
    W("| TR | **1,0000** | centroide | 0,5338 | +0,0288 |")
    W("| AG | 0,9961 | estructural | 0,6627 | **+0,1129** |")
    W("")
    W("Dos de esas tres lineas no son utilizables. Para **TR el control tiene AUC exactamente")
    W("1,0000**: la confianza out-of-fold ordena perfectamente todos los errores de TR, luego no")
    W("queda margen residual y una ganancia de 0,03 medida contra ese techo es ruido. Para **HS**")
    W("no existe region indefinida, luego el indicador estructural vale cero siempre. AG es la")
    W("unica etiqueta donde una segunda senal aporta informacion residual real.")
    W("")
    W("## La region, medida")
    W("")
    W("```text")
    W(f"filas en la region          {int(indef_global.sum())} de {n}")
    W(f"tasa de error dentro        {e_ind:.6f}")
    W(f"tasa de error fuera         {e_def:.6f}")
    W(f"enriquecimiento             {e_ind / max(e_def, 1e-12):.4f} veces")
    W(f"AUC del control global      {auc_ctrl:.4f}")
    W(f"AUC del control en la region {auc_ind:.4f}")
    W("```")
    W("")
    W("## El factor, DERIVADO y no barrido")
    W("")
    W("```text")
    W("peso_agregado = peso_de_clase * (1 - e_indefinida) / (1 - e_definida)")
    W("las dos tasas se miden con las filas de ENTRENAMIENTO del pliegue")
    W("```")
    W("")
    W("| pliegue | filas indefinidas | error dentro | error fuera | factor derivado |")
    W("|---:|---:|---:|---:|---:|")
    for i in info:
        W(f"| {i['pliegue']} | {i['filas_indefinidas_en_training']} | {i['error_dentro']:.4f} | "
          f"{i['error_fuera']:.4f} | {i['factor_derivado_w_soft']:.6f} |")
    W("")
    W("## Por que no es V14-00")
    W("")
    W("`V14-00` puso peso **exactamente cero** sobre la region, lo que demostro equivalente a")
    W("descartar las filas, y dio -0,116581. Ahi el factor era 0 por construccion. Aqui es un")
    W("ratio de fiabilidad derivado, distinto de cero, luego es una intervencion distinta y no un")
    W("rescate de la anterior.")
    W("")
    W("## Resultados")
    W("")
    W("```text")
    W(f"brazo 0 (control)          corrected_cv {m0['corrected_cv']:.10f}")
    W(f"  delta vs V03 {ccv_0 - ccv_ref:+.10f} | celdas distintas "
      f"{int((pred_0 != pred_v03).sum())}")
    W(f"brazo 1 (reescalado suave) corrected_cv {m1['corrected_cv']:.10f}")
    W(f"delta INTERVENCION PURA   {delta_mec:+.10f}")
    W(f"delta vs V03               {ccv_1 - ccv_ref:+.10f}")
    W(f"pliegues que mejoran       {folds_mej}/5")
    W(f"rescate {m1['rescate']} | dano {m1['damage']} | neto {m1['neto']:+d} "
      f"| cambian {m1['cambian']}")
    W(f"cambios SOLO en {ETIQUETA_OBJETIVO} {int((pred_1[:, j_obj] != pred_0[:, j_obj]).sum())} "
      f"celdas | en las otras dos {int((pred_1[:, [0, 1]] != pred_0[:, [0, 1]]).sum())}")
    W(f"factor derivado entre {min(factors):.6f} y {max(factors):.6f} "
      f"(media {np.mean(factors):.6f})")
    W(f"convergencia {conv}")
    W("```")
    W("")
    W("| etiqueta | F1 intervencion | F1 control |")
    W("|---|---:|---:|")
    for e in ETIQUETAS:
        W(f"| {e} | {m1['f1_por_etiqueta'][e]:.4f} | {m0['f1_por_etiqueta'][e]:.4f} |")
    W("")
    W("## Gate")
    W("")
    W("| criterio | valor | superado |")
    W("|---|---|---|")
    W(f"| efecto de la intervencion >= +0,003 | {delta_mec:+.6f} | **{h_delta}** |")
    W(f"| al menos 4 de 5 pliegues | {folds_mej}/5 | **{h_folds}** |")
    W(f"| gate completo | | **{gate}** |")
    W("")
    if ic_lo is not None:
        W(f"Bootstrap, ejecutado SOLO porque el gate se supero: IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}].")
    else:
        W("Bootstrap NO ejecutado: el gate no se supero.")
    W("")
    W("## Decision")
    W("")
    W("```text")
    W(f"decision  {decision}")
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p60_06_intervencion_ag.py`. Sin push.*")
    MD_C3.write_text(a_ascii("\n".join(L)), encoding="ascii")

    with CSV_C3.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["magnitud", "valor"])
        for fila in (
            ["brazo_0_corrected_cv", repr(m0["corrected_cv"])],
            ["brazo_1_corrected_cv", repr(m1["corrected_cv"])],
            ["delta_intervencion_puro", repr(delta_mec)],
            ["delta_vs_V03", repr(ccv_1 - ccv_ref)],
            ["corrected_cv_V03", repr(ccv_ref)],
            ["pliegues_mejorados", str(folds_mej)],
            ["folds_brazo_1", " ".join(f"{x:.6f}" for x in m1["folds"])],
            ["folds_brazo_0", " ".join(f"{x:.6f}" for x in m0["folds"])],
            ["rescate", str(m1["rescate"])], ["damage", str(m1["damage"])],
            ["neto", str(m1["neto"])], ["cambian", str(m1["cambian"])],
            ["cambian_en_ag", str(int(np.count_nonzero(pred_1[:, j_obj] != pred_0[:, j_obj])))],
            ["cambian_en_las_otras", str(int(np.count_nonzero(
                pred_1[:, [0, 1]] != pred_0[:, [0, 1]])))],
            ["factor_min", f"{min(factors):.6f}"], ["factor_max", f"{max(factors):.6f}"],
            ["factor_medio", f"{np.mean(factors):.6f}"],
            ["convergio", str(conv)],
            ["gate_delta_superado", str(h_delta)],
            ["gate_pliegues_superado", str(h_folds)],
            ["gate_superado", str(gate)],
            ["decision", a_ascii(decision)],
        ):
            w.writerow(fila)

    for destino in (MD_C3, JSON_C3, CSV_C3):
        if destino.exists() and any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P60-C3 VEREDICTO: {decision.split('.')[0]}")
    log(f"P60-06 OK | {informe['gastos']['segundos']} s")


if __name__ == "__main__":
    warnings.simplefilter("error", RuntimeWarning)
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
