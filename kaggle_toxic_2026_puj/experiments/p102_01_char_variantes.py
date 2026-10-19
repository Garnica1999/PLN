#!/usr/bin/env python
"""
P102-01 (D4): variantes char sin 'wb' sobre el arnes REP (control = V03).

Pre-registro: reports/p102_preregistro.md (seccion 3). Pregunta: aportan las
variantes char SIN 'wb' (con/sin case) algo que sobreviva al gate A3?

Metodo:
  - Folds canonicos congelados (pliegues de features/oof_v03_final.npz,
    verificados contra src.splits.pliegues_corregidos).
  - Vectorizador y LR ajustados DENTRO del pliegue; LR balanced C=2.0
    liblinear seed 42 por etiqueta; umbrales V03 congelados.
  - Control obligatorio: M2 debe reproducir el OOF congelado de
    features/v2_oof_base_corrected.npz (si no, se detiene).
  - Particion 47: recreada (seed 43) y verificada por hash contra
    reports/p98_14_preregistro.json; control contra pred_v03 de
    features/p98_14_oof_b.npz.
  - Gate A3: delta >= +0.003; >=4/5 pliegues; IC95 por clusters excluye cero;
    delta 47 > 0; >= 200 celdas cambiadas.

Sin test; sin holdouts; sin leaderboard. Salidas:
  reports/p102_01_char_variantes.json y .md

Uso:
    /home/carlos/venv/bin/python experiments/p102_01_char_variantes.py
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

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
SAL_JSON = RAIZ / "reports" / "p102_01_char_variantes.json"
SAL_MD = RAIZ / "reports" / "p102_01_char_variantes.md"

V03_CCV = 0.7772114525165721
UMBRALES_V03 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
SEMILLA_BOOT = 20260925
LR_PARAMS = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000,
             "solver": "liblinear", "random_state": 42}

# Bloques (analyzer, ngram_range, lowercase) por variante.
VARIANTES = {
    "control_M2": [("word", (1, 2), True), ("char_wb", (3, 5), True)],
    "V1_char_no_wb": [("word", (1, 2), True), ("char", (1, 6), True)],
    "V2_triple": [("word", (1, 2), True), ("char_wb", (3, 5), True),
                  ("char", (1, 6), True)],
    "V3_char_no_wb_case": [("word", (1, 2), False), ("char", (1, 6), False)],
}


def log(mensaje: str) -> None:
    """Progreso."""
    print(f"[P102-01] {mensaje}", flush=True)


def sha256(ruta: Path) -> str:
    """sha256 completo."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()


def construir_matrices(textos_tr, textos_va, bloques):
    """Ajusta los vectorizadores dentro del pliegue y transforma validacion."""
    mats_tr, mats_va = [], []
    for analyzer, ngram, lower in bloques:
        vectorizador = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram,
                                       min_df=2, sublinear_tf=True,
                                       lowercase=lower)
        mats_tr.append(vectorizador.fit_transform(textos_tr))
        mats_va.append(vectorizador.transform(textos_va))
    return (sparse.hstack(mats_tr).tocsr(), sparse.hstack(mats_va).tocsr())


def oof_proba(train_dev: pd.DataFrame, pliegues: list, bloques) -> np.ndarray:
    """Probabilidades OOF (n, 3) con vectorizador+LR dentro de cada pliegue."""
    proba = np.zeros((len(train_dev), len(ETIQUETAS)), dtype=float)
    for numero, (idx_tr, idx_va) in enumerate(pliegues, start=1):
        matriz_tr, matriz_va = construir_matrices(
            train_dev["text"].iloc[idx_tr], train_dev["text"].iloc[idx_va],
            bloques)
        for j, etiqueta in enumerate(ETIQUETAS):
            modelo = LogisticRegression(**LR_PARAMS)
            modelo.fit(matriz_tr, train_dev[etiqueta].iloc[idx_tr])
            proba[idx_va, j] = modelo.predict_proba(matriz_va)[:, 1]
        log(f"  pliegue {numero} ajustado")
    return proba


def pred_umbrales(proba: np.ndarray) -> np.ndarray:
    """Aplica los umbrales congelados de V03."""
    return np.column_stack([
        (proba[:, j] >= UMBRALES_V03[etiqueta]).astype(int)
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
    """Ejecuta P102-01 completo."""
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
    pred_desde_m2 = pred_umbrales(proba_m2_hist)
    if not np.array_equal(pred_desde_m2, pred_v03):
        raise SystemExit("pred de M2 con umbrales V03 != pred congelado")

    train = data_module.read_train()
    particion = pd.read_csv(IDS_PATH)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(
        int).tolist()
    train_dev = train[train["id"].astype(int).isin(ids_dev)].reset_index(
        drop=True)
    if not np.array_equal(train_dev["id"].astype(int).to_numpy(),
                          zf["ids"].astype(int)):
        raise SystemExit("orden de dev distinto al congelado")
    log(f"desarrollo: {len(train_dev)} filas")

    pliegues_frescos = pliegues_corregidos(train_dev, grupos)
    etiquetas_fold = np.empty(len(train_dev), dtype=int)
    for num, (_, va) in enumerate(pliegues_frescos, start=1):
        etiquetas_fold[va] = num
    if not np.array_equal(etiquetas_fold, pliegues_can):
        log("AVISO: los folds frescos no coinciden 1:1 con el congelado")

    log("control M2 (debe reproducir el OOF congelado)")
    proba_control = oof_proba(train_dev, pliegues_frescos,
                              VARIANTES["control_M2"])
    diff_control = float(np.abs(proba_control - proba_m2_hist).max())
    ccv_control = float(metricas_multietiqueta(
        y, pred_umbrales(proba_control))["f1_media_columnas"])
    if diff_control > 1e-6 or abs(ccv_control - V03_CCV) > 1e-12:
        raise SystemExit(f"control NO reproduce: diff {diff_control}, "
                         f"ccv {ccv_control}")
    log(f"control OK (diff max {diff_control:.2e}; ccv {ccv_control:.16f})")

    oof47 = np.load(OOF_47, allow_pickle=True)
    pred_v03_47 = oof47["pred_v03"].astype(int)
    pliegues_47 = folds_part47(y, grupos)
    log("particion 47 verificada por hash")

    resultados = {}
    for nombre, bloques in VARIANTES.items():
        log(f"variante {nombre}")
        proba = oof_proba(train_dev, pliegues_frescos, bloques)
        pred = pred_umbrales(proba)
        ccv = float(metricas_multietiqueta(y, pred)["f1_media_columnas"])
        folds_can = f1_pliegues(y, pred, pliegues_can)
        folds_v03 = f1_pliegues(y, pred_v03, pliegues_can)
        folds_pos = int(sum(a > b + 1e-12 for a, b in zip(folds_can, folds_v03)))
        celdas = int((pred != pred_v03).sum())
        boot = bootstrap_clusters_delta(y, pred, pred_v03, grupos)
        proba_47 = oof_proba(train_dev, list(_folds_iter(pliegues_47)),
                             bloques)
        pred_47 = pred_umbrales(proba_47)
        ccv_47 = float(metricas_multietiqueta(y, pred_47)
                       ["f1_media_columnas"])
        ccv_v03_47 = float(metricas_multietiqueta(y, pred_v03_47)
                           ["f1_media_columnas"])
        diff_47 = int((pred_47 != pred_v03_47).sum()) if nombre == \
            "control_M2" else None
        resultados[nombre] = {
            "corrected_cv": ccv,
            "delta_vs_v03": round(ccv - V03_CCV, 6),
            "folds_canonicos": [round(x, 6) for x in folds_can],
            "pliegues_mejorados": folds_pos,
            "celdas_cambiadas": celdas,
            "bootstrap_vs_v03": boot,
            "particion_47": {"corrected_cv": round(ccv_47, 6),
                             "v03_equivalente": round(ccv_v03_47, 6),
                             "delta_vs_v03": round(ccv_47 - ccv_v03_47, 6)},
            "control_47_pred_identica": diff_47,
        }
        if nombre != "control_M2":
            resultados[nombre]["gate"] = gate_a3(
                ccv - V03_CCV, folds_pos, boot["ic95"],
                ccv_47 - ccv_v03_47, celdas)
        log(f"  ccv {ccv:.6f} | delta {ccv - V03_CCV:+.6f} | "
            f"pliegues {folds_pos}/5 | celdas {celdas}")

    salida = {
        "experiment_id": "P102-01",
        "fase": "P102 sprint (triage): D4 char sin wb",
        "fecha": "2026-10-14",
        "control": {"diff_max_proba": diff_control, "ccv": ccv_control,
                    "reproduce_v03": True},
        "v03_referencia": {"corrected_cv": V03_CCV,
                           "umbrales": UMBRALES_V03},
        "variantes": resultados,
        "sha_oof_final": sha256(OOF_FINAL)[:16],
        "sha_oof_base": sha256(OOF_BASE)[:16],
        "tiempo_s": round(time.perf_counter() - inicio, 1),
    }
    SAL_JSON.write_text(json.dumps(salida, indent=2, ensure_ascii=True)
                        + "\n", encoding="utf-8")

    md = ["# P102-01 - Variantes char sin 'wb' (control = V03)", "",
          (f"Fecha: 2026-10-14. Control reproduce V03 exacto "
           f"(diff proba {diff_control:.2e}; ccv {ccv_control:.16f})."), "",
          ("| Variante | corrected_cv | delta V03 | pliegues | celdas | "
           "IC95 vs V03 | delta 47 | Gate |"),
          "|---|---|---|---|---|---|---|---|"]
    for nombre, r in resultados.items():
        ic = r["bootstrap_vs_v03"]["ic95"]
        gate = r.get("gate", {}).get("veredicto", "-")
        md.append(
            f"| {nombre} | {r['corrected_cv']:.6f} | "
            f"{r['delta_vs_v03']:+.6f} | {r['pliegues_mejorados']}/5 | "
            f"{r['celdas_cambiadas']} | "
            f"[{ic[0]:+.6f},{ic[1]:+.6f}] | "
            f"{r['particion_47']['delta_vs_v03']:+.6f} | {gate} |")
    md += ["", "Gate A3: delta >= +0.003; >=4/5 pliegues; IC95 excluye cero;",
           "delta 47 > 0; >= 200 celdas cambiadas.",
           "Sin test; sin holdouts; sin leaderboard; umbrales V03 congelados.",
           ""]
    SAL_MD.write_text("\n".join(md), encoding="ascii")

    log(f"resultados: {SAL_JSON.relative_to(RAIZ)} y "
        f"{SAL_MD.relative_to(RAIZ)} | tiempo "
        f"{salida['tiempo_s']} s")


def _folds_iter(pliegues: np.ndarray):
    """Itera (train, val) desde un vector de pliegues."""
    for k in sorted(set(pliegues.tolist())):
        val = np.flatnonzero(pliegues == k)
        tr = np.flatnonzero(pliegues != k)
        yield tr, val


if __name__ == "__main__":
    main()
