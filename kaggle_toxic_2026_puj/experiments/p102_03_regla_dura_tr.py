#!/usr/bin/env python
"""
P102-03 (B2-corner): regla dura TR=0 si HS=0, aislada.

Pre-registro: reports/p102_preregistro.md (seccion 1, item 3). En train la
invariante TR=1 => HS=1 es exacta (0 excepciones; V6-01/V12-02/P45); AG tiene
26 excepciones. Pregunta: aplicar la regla dura SOLO a las predicciones de V03
(canonica y particion 47) mejora la metrica segun el gate A3?

No entrena nada: usa las probabilidades y predicciones congeladas
(oof_v03_final.npz y p98_14_oof_b.npz). Sin test; sin holdouts; sin leaderboard.
Contexto adicional (sin decision): celdas que la regla cambiaria en la
submission congelada de V03 (artefacto propio, solo predicciones).

Salidas: reports/p102_03_regla_dura_tr.json y .md

Uso:
    /home/carlos/venv/bin/python experiments/p102_03_regla_dura_tr.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import metricas_multietiqueta

OOF_FINAL = RAIZ / "features" / "oof_v03_final.npz"
OOF_47 = RAIZ / "features" / "p98_14_oof_b.npz"
SUBMISSION_V03 = RAIZ / "submissions" / "toxic_submission_v03_lr_union_thr.csv"
SAL_JSON = RAIZ / "reports" / "p102_03_regla_dura_tr.json"
SAL_MD = RAIZ / "reports" / "p102_03_regla_dura_tr.md"

V03_CCV = 0.7772114525165721
SEMILLA_BOOT = 20260925


def log(mensaje: str) -> None:
    """Progreso."""
    print(f"[P102-03] {mensaje}", flush=True)


def sha256(ruta: Path) -> str:
    """sha256 completo."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()


def regla_dura(pred: np.ndarray) -> np.ndarray:
    """TR=0 donde HS=0 (columnas HS=0, TR=1, AG=2)."""
    salida = pred.copy()
    salida[(salida[:, 0] == 0) & (salida[:, 1] == 1), 1] = 0
    return salida


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


def main() -> None:
    """Ejecuta P102-03 completo."""
    for ruta in (SAL_JSON, SAL_MD):
        if ruta.exists():
            raise SystemExit(f"Ya existe {ruta}; no se sobrescribe.")
    inicio = time.perf_counter()

    train = data_module.read_train()
    hs0_tr1 = int(((train["HS"] == 0) & (train["TR"] == 1)).sum())
    hs0_ag1 = int(((train["HS"] == 0) & (train["AG"] == 1)).sum())
    log(f"invariantes en train: TR=1&HS=0 -> {hs0_tr1}; AG=1&HS=0 -> {hs0_ag1}")

    zf = np.load(OOF_FINAL, allow_pickle=True)
    y = zf["y_true"].astype(int)
    pliegues = zf["pliegues"].astype(int)
    pred_v03 = zf["pred"].astype(int)
    grupos = np.load(RAIZ / "features" / "v2_oof_base_corrected.npz",
                     allow_pickle=True)["grupos"].astype(np.int64)

    viola_can = int(((pred_v03[:, 0] == 0) & (pred_v03[:, 1] == 1)).sum())
    pred_dura = regla_dura(pred_v03)
    ccv_dura = float(metricas_multietiqueta(y, pred_dura)["f1_media_columnas"])
    folds_dura = f1_pliegues(y, pred_dura, pliegues)
    folds_v03 = f1_pliegues(y, pred_v03, pliegues)
    folds_pos = int(sum(a > b + 1e-12 for a, b in zip(folds_dura, folds_v03)))
    celdas_can = int((pred_dura != pred_v03).sum())
    boot = bootstrap_clusters_delta(y, pred_dura, pred_v03, grupos)
    log(f"canonica: violaciones {viola_can}; celdas {celdas_can}; "
        f"delta {ccv_dura - V03_CCV:+.6f}")

    zb = np.load(OOF_47, allow_pickle=True)
    pred_v03_47 = zb["pred_v03"].astype(int)
    viola_47 = int(((pred_v03_47[:, 0] == 0) & (pred_v03_47[:, 1] == 1)).sum())
    pred_dura_47 = regla_dura(pred_v03_47)
    ccv_dura_47 = float(metricas_multietiqueta(y, pred_dura_47)
                        ["f1_media_columnas"])
    ccv_v03_47 = float(metricas_multietiqueta(y, pred_v03_47)
                       ["f1_media_columnas"])
    celdas_47 = int((pred_dura_47 != pred_v03_47).sum())
    log(f"particion 47: violaciones {viola_47}; celdas {celdas_47}; "
        f"delta {ccv_dura_47 - ccv_v03_47:+.6f}")

    sub = pd.read_csv(SUBMISSION_V03)
    sub_dura = regla_dura(sub[["HS", "TR", "AG"]].to_numpy().astype(int))
    celdas_sub = int((sub_dura != sub[["HS", "TR", "AG"]].to_numpy()).sum())

    delta_can = ccv_dura - V03_CCV
    delta_47 = ccv_dura_47 - ccv_v03_47
    condiciones = {
        "delta_can_ge_0003": delta_can >= 0.003,
        "pliegues_ge_4": folds_pos >= 4,
        "ic95_excluye_cero": boot["ic95"][0] > 0.0,
        "delta_47_positivo": delta_47 > 0.0,
        "celdas_ge_200": celdas_can >= 200,
    }
    salida = {
        "experiment_id": "P102-03",
        "fase": "P102 sprint (triage): B2-corner regla dura TR",
        "fecha": "2026-10-14",
        "invariantes_train": {"TR1_HS0": hs0_tr1, "AG1_HS0": hs0_ag1,
                              "filas": len(train)},
        "canonica": {
            "violaciones_en_v03": viola_can,
            "corrected_cv_dura": ccv_dura,
            "delta_vs_v03": round(delta_can, 6),
            "folds_canonicos": [round(x, 6) for x in folds_dura],
            "pliegues_mejorados": folds_pos,
            "celdas_cambiadas": celdas_can,
            "bootstrap_vs_v03": boot},
        "particion_47": {
            "violaciones_en_v03": viola_47,
            "corrected_cv_dura": round(ccv_dura_47, 6),
            "v03_equivalente": round(ccv_v03_47, 6),
            "delta_vs_v03": round(delta_47, 6),
            "celdas_cambiadas": celdas_47},
        "submission_v03_contexto": {"celdas_que_cambiaria": celdas_sub},
        "gate": {"condiciones": condiciones,
                 "veredicto": "GATE_PASSED" if all(condiciones.values())
                 else "GATE_FAILED"},
        "sha_submission_v03": sha256(SUBMISSION_V03)[:16],
        "tiempo_s": round(time.perf_counter() - inicio, 1),
    }
    SAL_JSON.write_text(json.dumps(salida, indent=2, ensure_ascii=True)
                        + "\n", encoding="utf-8")

    ic = boot["ic95"]
    md = [("# P102-03 - Regla dura TR=0 si HS=0 (aislada sobre V03)"), "",
          (f"Fecha: 2026-10-14. Invariante en train: TR=1 & HS=0 -> "
           f"{hs0_tr1} casos; AG=1 & HS=0 -> {hs0_ag1} casos."), "",
          ("| Particion | violaciones V03 | celdas cambiadas | "
           "delta vs V03 |"), "|---|---|---|---|",
          (f"| Canonica | {viola_can} | {celdas_can} | {delta_can:+.6f} |"),
          (f"| Particion 47 | {viola_47} | {celdas_47} | {delta_47:+.6f} |"),
          "", (f"Bootstrap vs V03 (canonica): IC95 "
               f"[{ic[0]:+.6f},{ic[1]:+.6f}]; pliegues "
               f"{folds_pos}/5."),
          (f"Submission V03: la regla cambiaria {celdas_sub} celdas "
           f"(contexto, sin decision)."),
          "", (f"Gate: {salida['gate']['veredicto']} "
               f"{salida['gate']['condiciones']}"), ""]
    SAL_MD.write_text("\n".join(md), encoding="ascii")

    log(f"resultados: {SAL_JSON.relative_to(RAIZ)} y "
        f"{SAL_MD.relative_to(RAIZ)} | tiempo {salida['tiempo_s']} s")


if __name__ == "__main__":
    main()
