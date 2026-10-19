#!/usr/bin/env python
"""
P102-02 (C1-corner): promedio V03 + LinearSVC + ComplementNB con umbrales
recalibrados a la escala del promedio.

Pre-registro: reports/p102_preregistro.md (seccion 1, item 2). Pregunta: el
promedio de tres modelos lineales (LR de V03, LinearSVC y ComplementNB) con
umbrales RECALIBRADOS tras el promedio supera a V03 segun el gate A3?

Contexto medido: v08_p1 (promedio de modelos de pliegue) perdio en el LB por
la contraccion de probabilidades al promediar (+0.0013) con umbrales de V03;
aqui se corrige ese defecto (recalibracion sobre el OOF del promedio) y se
anade ComplementNB como tercer miembro diverso.

Metodo:
  - Folds canonicos congelados; vectorizador+modelos dentro del pliegue.
  - Miembros: LR C=2 balanced (M2 de V03), LinearSVC C=1 balanced
    (sigmoide del decision_function), ComplementNB.
  - Umbrales recalibrados por etiqueta (rejilla 0.20-0.80) sobre el OOF del
    promedio en la particion canonica; se TRANSFIEREN tal cual a la 47.
  - Control obligatorio: M2 debe reproducir el OOF congelado.
  - Gate A3 identico a P102-01.

Sin test; sin holdouts; sin leaderboard. Salidas:
  reports/p102_02_promedio_recalibrado.json y .md

Uso:
    /home/carlos/venv/bin/python experiments/p102_02_promedio_recalibrado.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta
from src.splits import pliegues_corregidos

OOF_FINAL = RAIZ / "features" / "oof_v03_final.npz"
OOF_BASE = RAIZ / "features" / "v2_oof_base_corrected.npz"
OOF_47 = RAIZ / "features" / "p98_14_oof_b.npz"
PRE_47 = RAIZ / "reports" / "p98_14_preregistro.json"
IDS_PATH = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
SAL_JSON = RAIZ / "reports" / "p102_02_promedio_recalibrado.json"
SAL_MD = RAIZ / "reports" / "p102_02_promedio_recalibrado.md"

V03_CCV = 0.7772114525165721
UMBRALES_V03 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
SEMILLA_BOOT = 20260925
LR_PARAMS = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000,
             "solver": "liblinear", "random_state": 42}
SVC_PARAMS = {"C": 1.0, "class_weight": "balanced", "max_iter": 5000,
              "random_state": 42}
CNB_PARAMS = {"alpha": 1.0}
BLOQUES = [("word", (1, 2)), ("char_wb", (3, 5))]


def log(mensaje: str) -> None:
    """Progreso."""
    print(f"[P102-02] {mensaje}", flush=True)


def sha256(ruta: Path) -> str:
    """sha256 completo."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()


def construir_matrices(textos_tr, textos_va):
    """Union TF-IDF word(1,2)+char_wb(3,5) ajustada dentro del pliegue."""
    mats_tr, mats_va = [], []
    for analyzer, ngram in BLOQUES:
        vectorizador = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram,
                                       min_df=2, sublinear_tf=True)
        mats_tr.append(vectorizador.fit_transform(textos_tr))
        mats_va.append(vectorizador.transform(textos_va))
    return (sparse.hstack(mats_tr).tocsr(), sparse.hstack(mats_va).tocsr())


def oof_miembros(train_dev: pd.DataFrame, pliegues: list) -> dict:
    """OOF (n,3) por miembro y del promedio simple de los tres."""
    n = len(train_dev)
    proba = {nombre: np.zeros((n, len(ETIQUETAS)), dtype=float)
             for nombre in ("lr", "svc", "cnb")}
    for numero, (idx_tr, idx_va) in enumerate(pliegues, start=1):
        matriz_tr, matriz_va = construir_matrices(
            train_dev["text"].iloc[idx_tr], train_dev["text"].iloc[idx_va])
        for j, etiqueta in enumerate(ETIQUETAS):
            y_tr = train_dev[etiqueta].iloc[idx_tr]
            lr = LogisticRegression(**LR_PARAMS).fit(matriz_tr, y_tr)
            svc = LinearSVC(**SVC_PARAMS).fit(matriz_tr, y_tr)
            cnb = ComplementNB(**CNB_PARAMS).fit(matriz_tr, y_tr)
            proba["lr"][idx_va, j] = lr.predict_proba(matriz_va)[:, 1]
            proba["svc"][idx_va, j] = expit(svc.decision_function(matriz_va))
            proba["cnb"][idx_va, j] = cnb.predict_proba(matriz_va)[:, 1]
        log(f"  pliegue {numero} ajustado")
    proba["promedio"] = (proba["lr"] + proba["svc"] + proba["cnb"]) / 3.0
    return proba


def umbrales_recalibrados(proba: np.ndarray, y: np.ndarray) -> dict:
    """Rejilla 0.20-0.80 por etiqueta maximizando la F1 binaria sobre el OOF."""
    umbrales = {}
    for j, etiqueta in enumerate(ETIQUETAS):
        mejor_u, mejor_f1 = 0.5, -1.0
        for u in np.arange(0.20, 0.801, 0.01):
            f1 = f1_score(y[:, j], (proba[:, j] >= u).astype(int),
                          zero_division=0)
            if f1 > mejor_f1:
                mejor_f1, mejor_u = float(f1), round(float(u), 2)
        umbrales[etiqueta] = mejor_u
    return umbrales


def pred_con(proba: np.ndarray, umbrales: dict) -> np.ndarray:
    """Aplica umbrales por etiqueta."""
    return np.column_stack([
        (proba[:, j] >= umbrales[etiqueta]).astype(int)
        for j, etiqueta in enumerate(ETIQUETAS)])


def f1_pliegues(y: np.ndarray, pred: np.ndarray, pliegues: np.ndarray) -> list:
    """F1 media por columna en cada pliegue."""
    return [float(metricas_multietiqueta(
        y[pliegues == k], pred[pliegues == k])["f1_media_columnas"])
        for k in sorted(set(pliegues.tolist()))]


def bootstrap_clusters_delta(y, pred_a, pred_b, grupos, repeticiones=2000):
    """IC95 del delta pooled (a-b) por clusters (maquinaria de p98_13)."""
    generador = np.random.default_rng(SEMILLA_BOOT)
    indices = {int(g): np.flatnonzero(grupos == g) for g in np.unique(grupos)}
    claves = np.array(sorted(indices))
    dif = np.empty(repeticiones)
    for r in range(repeticiones):
        muestra = generador.integers(0, len(claves), len(claves))
        idx = np.concatenate([indices[int(claves[i])] for i in muestra])
        dif[r] = (metricas_multietiqueta(y[idx], pred_a[idx])
                  ["f1_media_columnas"]
                  - metricas_multietiqueta(y[idx], pred_b[idx])
                  ["f1_media_columnas"])
    c = np.quantile(dif, [0.025, 0.975])
    return {"ic95": [round(float(c[0]), 6), round(float(c[1]), 6)],
            "pct_mejora": round(float((dif > 0).mean()), 6)}


def folds_part47(y: np.ndarray, grupos: np.ndarray) -> np.ndarray:
    """Recrea la particion 47 y verifica su hash contra el pre-registro."""
    combinacion = np.array([f"{a}{b}{c}" for a, b, c in y])
    divisor = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
    pliegues = np.full(len(y), -1, dtype=int)
    for numero, (_, val) in enumerate(divisor.split(
            np.arange(len(y)).reshape(-1, 1), combinacion, groups=grupos)):
        pliegues[val] = numero + 1
    sha = hashlib.sha256(pliegues.astype("int64").tobytes()).hexdigest()
    previo = json.loads(PRE_47.read_text(encoding="utf-8"))
    if previo["particion"]["sha256_pliegues"] != sha:
        raise SystemExit("la particion 47 no reproduce el pre-registro")
    return pliegues


def iter_folds(pliegues: np.ndarray):
    """Itera (train, val) desde un vector de pliegues."""
    for k in sorted(set(pliegues.tolist())):
        yield np.flatnonzero(pliegues != k), np.flatnonzero(pliegues == k)


def gate_a3(delta_can, folds_pos, ic95, delta_47, celdas) -> dict:
    """Gate A3 del pre-registro P102 (todas las condiciones obligatorias)."""
    condiciones = {
        "delta_can_ge_0003": delta_can >= 0.003,
        "pliegues_ge_4": folds_pos >= 4,
        "ic95_excluye_cero": ic95[0] > 0.0,
        "delta_47_positivo": delta_47 > 0.0,
        "celdas_ge_200": celdas >= 200,
    }
    return {"condiciones": condiciones,
            "veredicto": "GATE_PASSED" if all(condiciones.values())
            else "GATE_FAILED"}


def main() -> None:
    """Ejecuta P102-02 completo."""
    for ruta in (SAL_JSON, SAL_MD):
        if ruta.exists():
            raise SystemExit(f"Ya existe {ruta}; no se sobrescribe.")
    inicio = time.perf_counter()

    zf = np.load(OOF_FINAL, allow_pickle=True)
    zb = np.load(OOF_BASE, allow_pickle=True)
    pliegues_can = zf["pliegues"].astype(int)
    y = zf["y_true"].astype(int)
    pred_v03 = zf["pred"].astype(int)
    grupos = zb["grupos"].astype(np.int64)
    modelos = [str(m) for m in zb["modelos"]]
    idx_m2 = modelos.index("M2_union_lr_c2")
    proba_m2_hist = zb["proba"][:, idx_m2, :]

    train = data_module.read_train()
    particion = pd.read_csv(IDS_PATH)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(
        int).tolist()
    train_dev = train[train["id"].astype(int).isin(ids_dev)].reset_index(
        drop=True)
    log(f"desarrollo: {len(train_dev)} filas")

    pliegues_frescos = pliegues_corregidos(train_dev, grupos)
    etiquetas_fold = np.empty(len(train_dev), dtype=int)
    for num, (_, va) in enumerate(pliegues_frescos, start=1):
        etiquetas_fold[va] = num
    if not np.array_equal(etiquetas_fold, pliegues_can):
        raise SystemExit("los folds frescos no coinciden con el congelado")

    log("miembros en la particion canonica")
    proba_can = oof_miembros(train_dev, pliegues_frescos)
    diff_control = float(np.abs(proba_can["lr"] - proba_m2_hist).max())
    if diff_control > 1e-6:
        raise SystemExit(f"control NO reproduce: diff {diff_control}")
    log(f"control OK (diff max {diff_control:.2e})")

    umbrales_recal = umbrales_recalibrados(proba_can["promedio"], y)
    pred_prom_frozen = pred_con(proba_can["promedio"], UMBRALES_V03)
    pred_prom_recal = pred_con(proba_can["promedio"], umbrales_recal)
    ccv_frozen = float(metricas_multietiqueta(y, pred_prom_frozen)
                       ["f1_media_columnas"])
    ccv_recal = float(metricas_multietiqueta(y, pred_prom_recal)
                      ["f1_media_columnas"])
    folds_prom = f1_pliegues(y, pred_prom_recal, pliegues_can)
    folds_v03 = f1_pliegues(y, pred_v03, pliegues_can)
    folds_pos = int(sum(a > b + 1e-12 for a, b in zip(folds_prom, folds_v03)))
    celdas = int((pred_prom_recal != pred_v03).sum())
    boot = bootstrap_clusters_delta(y, pred_prom_recal, pred_v03, grupos)
    log(f"promedio recalibrado: ccv {ccv_recal:.6f} "
        f"(delta {ccv_recal - V03_CCV:+.6f}; frozen {ccv_frozen:.6f})")

    log("miembros en la particion 47")
    proba_47 = oof_miembros(train_dev, list(iter_folds(folds_part47(y, grupos))))
    oof47 = np.load(OOF_47, allow_pickle=True)
    pred_v03_47 = oof47["pred_v03"].astype(int)
    ccv_v03_47 = float(metricas_multietiqueta(y, pred_v03_47)
                       ["f1_media_columnas"])
    pred_47_recal = pred_con(proba_47["promedio"], umbrales_recal)
    pred_47_frozen = pred_con(proba_47["promedio"], UMBRALES_V03)
    ccv_47_recal = float(metricas_multietiqueta(y, pred_47_recal)
                         ["f1_media_columnas"])
    ccv_47_frozen = float(metricas_multietiqueta(y, pred_47_frozen)
                          ["f1_media_columnas"])

    contexto_solo = {}
    for nombre in ("svc", "cnb"):
        contexto_solo[nombre] = {
            "ccv_frozen": round(float(metricas_multietiqueta(
                y, pred_con(proba_can[nombre], UMBRALES_V03))
                ["f1_media_columnas"]), 6),
            "umbrales_propios": umbrales_recalibrados(proba_can[nombre], y)}

    salida = {
        "experiment_id": "P102-02",
        "fase": "P102 sprint (triage): C1-corner promedio recalibrado",
        "fecha": "2026-10-14",
        "control": {"diff_max_proba_lr": diff_control, "reproduce_v03": True},
        "promedio": {
            "miembros": ["lr_M2", "svc_C1", "cnb_alpha1"],
            "umbrales_recalibrados": umbrales_recal,
            "corrected_cv_recalibrado": ccv_recal,
            "corrected_cv_frozen_v03": ccv_frozen,
            "delta_vs_v03": round(ccv_recal - V03_CCV, 6),
            "folds_canonicos": [round(x, 6) for x in folds_prom],
            "pliegues_mejorados": folds_pos,
            "celdas_cambiadas": celdas,
            "bootstrap_vs_v03": boot,
            "particion_47": {
                "recalibrado_transferido": round(ccv_47_recal, 6),
                "frozen": round(ccv_47_frozen, 6),
                "v03_equivalente": round(ccv_v03_47, 6),
                "delta_recal_vs_v03": round(ccv_47_recal - ccv_v03_47, 6)},
            "gate": gate_a3(ccv_recal - V03_CCV, folds_pos, boot["ic95"],
                            ccv_47_recal - ccv_v03_47, celdas),
        },
        "contexto_miembros_solos": contexto_solo,
        "sha_oof_final": sha256(OOF_FINAL)[:16],
        "tiempo_s": round(time.perf_counter() - inicio, 1),
    }
    SAL_JSON.write_text(json.dumps(salida, indent=2, ensure_ascii=True)
                        + "\n", encoding="utf-8")

    p = salida["promedio"]
    ic = p["bootstrap_vs_v03"]["ic95"]
    md = [("# P102-02 - Promedio V03+LinearSVC+ComplementNB con umbrales "
           "recalibrados"), "",
          (f"Fecha: 2026-10-14. Control reproduce V03 exacto "
           f"(diff {diff_control:.2e}). Umbrales recalibrados: "
           f"{umbrales_recal}."), "",
          "| Metrica | Valor |", "|---|---|",
          f"| corrected_cv (recalibrado) | {p['corrected_cv_recalibrado']:.6f} |",
          f"| corrected_cv (umbrales V03 frozen) | {p['corrected_cv_frozen_v03']:.6f} |",
          f"| delta vs V03 | {p['delta_vs_v03']:+.6f} |",
          f"| pliegues mejorados | {p['pliegues_mejorados']}/5 |",
          f"| celdas cambiadas | {p['celdas_cambiadas']} |",
          f"| IC95 vs V03 | [{ic[0]:+.6f},{ic[1]:+.6f}] |",
          (f"| particion 47 (recal transferido) | {p['particion_47']['recalibrado_transferido']:.6f} "
           f"({p['particion_47']['delta_recal_vs_v03']:+.6f} vs v03) |"),
          f"| gate | {p['gate']['veredicto']} |", "",
          ("Gate A3: delta >= +0.003; >=4/5 pliegues; IC95 excluye cero; "
           "delta 47 > 0; >= 200 celdas."), ""]
    SAL_MD.write_text("\n".join(md), encoding="ascii")

    log(f"resultados: {SAL_JSON.relative_to(RAIZ)} y "
        f"{SAL_MD.relative_to(RAIZ)} | tiempo {salida['tiempo_s']} s")


if __name__ == "__main__":
    main()
