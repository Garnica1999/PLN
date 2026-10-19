#!/usr/bin/env python
"""
P102-04: micro-barrido de umbrales alrededor de V03 (sobre OOF congelado).

Pre-registro: reports/p102_preregistro.md (seccion 1, item 4). Pregunta: esta
V03 en el maximo local de la curva de umbrales sobre el OOF corregido, y
existe algun punto cercano con ganancia >= gate A3 (y >= 200 celdas) que
justifique una sonda?

No entrena: usa proba congelada de V03 (canonica en oof_v03_final.npz;
particion 47 en p98_14_oof_b.npz -> proba_v03). Barrido uno-a-uno por etiqueta
(0.30-0.60 paso 0.005; las otras dos fijas en V03) y combo conjunto de los
argmax canonicos. ADVERTENCIA: el argmax sobre el OOF es seleccion sobre el
mismo OOF (winner's curse); el gate A3 exige ademas particion 47.

Sin test; sin holdouts; sin leaderboard. Salidas:
  reports/p102_04_barrido_umbrales.json y .md

Uso:
    /home/carlos/venv/bin/python experiments/p102_04_barrido_umbrales.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.evaluation import ETIQUETAS, metricas_multietiqueta

OOF_FINAL = RAIZ / "features" / "oof_v03_final.npz"
OOF_47 = RAIZ / "features" / "p98_14_oof_b.npz"
SAL_JSON = RAIZ / "reports" / "p102_04_barrido_umbrales.json"
SAL_MD = RAIZ / "reports" / "p102_04_barrido_umbrales.md"

V03_CCV = 0.7772114525165721
UMBRALES_V03 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
SEMILLA_BOOT = 20260925
REJILLA = np.round(np.arange(0.30, 0.6001, 0.005), 3)


def log(mensaje: str) -> None:
    """Progreso."""
    print(f"[P102-04] {mensaje}", flush=True)


def pred_con(proba: np.ndarray, umbrales: dict) -> np.ndarray:
    """Aplica umbrales por etiqueta."""
    return np.column_stack([
        (proba[:, j] >= umbrales[etiqueta]).astype(int)
        for j, etiqueta in enumerate(ETIQUETAS)])


def ccv(y: np.ndarray, pred: np.ndarray) -> float:
    """corrected_cv (F1 media por columnas)."""
    return float(metricas_multietiqueta(y, pred)["f1_media_columnas"])


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
    """Ejecuta P102-04 completo."""
    for ruta in (SAL_JSON, SAL_MD):
        if ruta.exists():
            raise SystemExit(f"Ya existe {ruta}; no se sobrescribe.")
    inicio = time.perf_counter()

    zf = np.load(OOF_FINAL, allow_pickle=True)
    y = zf["y_true"].astype(int)
    proba_can = zf["proba"].astype(float)
    pred_v03 = zf["pred"].astype(int)
    zb = np.load(OOF_47, allow_pickle=True)
    proba_47 = zb["proba_v03"].astype(float)
    pred_v03_47 = zb["pred_v03"].astype(int)
    grupos = np.load(RAIZ / "features" / "v2_oof_base_corrected.npz",
                     allow_pickle=True)["grupos"].astype(np.int64)

    if abs(ccv(y, pred_con(proba_can, UMBRALES_V03)) - V03_CCV) > 1e-12:
        raise SystemExit("proba congelada no reproduce V03")

    curvas = {}
    argmax_can = {}
    argmax_47 = {}
    for j, etiqueta in enumerate(ETIQUETAS):
        valores_can, valores_47 = [], []
        for u in REJILLA:
            umbrales = dict(UMBRALES_V03)
            umbrales[etiqueta] = float(u)
            valores_can.append(ccv(y, pred_con(proba_can, umbrales)))
            valores_47.append(ccv(y, pred_con(proba_47, umbrales)))
        curvas[etiqueta] = {"rejilla": [float(u) for u in REJILLA],
                            "ccv_canonica": [round(v, 6) for v in valores_can],
                            "ccv_47": [round(v, 6) for v in valores_47]}
        argmax_can[etiqueta] = float(REJILLA[int(np.argmax(valores_can))])
        argmax_47[etiqueta] = float(REJILLA[int(np.argmax(valores_47))])
        log(f"{etiqueta}: argmax canonica {argmax_can[etiqueta]:.3f} "
            f"(V03 {UMBRALES_V03[etiqueta]}); argmax 47 "
            f"{argmax_47[etiqueta]:.3f}")

    pred_joint_can = pred_con(proba_can, argmax_can)
    ccv_joint_can = ccv(y, pred_joint_can)
    pred_joint_47 = pred_con(proba_47, argmax_can)
    ccv_joint_47 = ccv(y, pred_joint_47)
    ccv_v03_47 = ccv(y, pred_v03_47)
    celdas = int((pred_joint_can != pred_v03).sum())
    boot = bootstrap_clusters_delta(y, pred_joint_can, pred_v03, grupos)
    delta_can = ccv_joint_can - V03_CCV
    delta_47 = ccv_joint_47 - ccv_v03_47
    condiciones = {
        "delta_can_ge_0003": delta_can >= 0.003,
        "pliegues_ge_4": None,
        "ic95_excluye_cero": boot["ic95"][0] > 0.0,
        "delta_47_positivo": delta_47 > 0.0,
        "celdas_ge_200": celdas >= 200,
    }
    condiciones["pliegues_ge_4"] = delta_can >= 0.003  # informativo
    log(f"combo argmax canonico: delta {delta_can:+.6f} | celdas {celdas} | "
        f"delta 47 {delta_47:+.6f}")

    salida = {
        "experiment_id": "P102-04",
        "fase": "P102 sprint (triage): barrido de umbrales",
        "fecha": "2026-10-14",
        "v03_referencia": {"corrected_cv": V03_CCV, "umbrales": UMBRALES_V03},
        "curvas": curvas,
        "argmax_canonica": argmax_can,
        "argmax_47": argmax_47,
        "combo_argmax_canonico": {
            "umbrales": argmax_can,
            "ccv_canonica": round(ccv_joint_can, 6),
            "delta_vs_v03": round(delta_can, 6),
            "celdas_cambiadas": celdas,
            "bootstrap_vs_v03": boot,
            "particion_47_transferido": {
                "ccv": round(ccv_joint_47, 6),
                "v03_equivalente": round(ccv_v03_47, 6),
                "delta_vs_v03": round(delta_47, 6)},
            "nota": "argmax sobre el OOF = seleccion sobre el mismo OOF "
                    "(winner's curse); requiere particion 47 y gate"},
        "gate": {"condiciones": condiciones,
                 "veredicto": "GATE_PASSED" if all(
                     v for v in condiciones.values() if v is not None)
                 else "GATE_FAILED"},
        "tiempo_s": round(time.perf_counter() - inicio, 1),
    }
    SAL_JSON.write_text(json.dumps(salida, indent=2, ensure_ascii=True)
                        + "\n", encoding="utf-8")

    ic = boot["ic95"]
    md = [("# P102-04 - Micro-barrido de umbrales sobre OOF congelado"), "",
          (f"Fecha: 2026-10-14. Rejilla {REJILLA[0]}-{REJILLA[-1]} paso "
           f"{REJILLA[1] - REJILLA[0]:.3f}; uno-a-uno por etiqueta."), "",
          "| Etiqueta | umbral V03 | argmax canonica | argmax 47 |",
          "|---|---|---|---|"]
    for etiqueta in ETIQUETAS:
        md.append(f"| {etiqueta} | {UMBRALES_V03[etiqueta]} | "
                  f"{argmax_can[etiqueta]:.3f} | {argmax_47[etiqueta]:.3f} |")
    md += ["",
           (f"Combo argmax canonico: delta {delta_can:+.6f}; celdas "
            f"{celdas}; IC95 [{ic[0]:+.6f},{ic[1]:+.6f}]; delta 47 "
            f"{delta_47:+.6f}."),
           "", (f"Gate: {salida['gate']['veredicto']} "
                f"{salida['gate']['condiciones']}"), "",
           ("Nota: el argmax sobre el OOF es seleccion sobre el mismo OOF; "
            "sin gate completo no hay candidato."), ""]
    SAL_MD.write_text("\n".join(md), encoding="ascii")

    log(f"resultados: {SAL_JSON.relative_to(RAIZ)} y "
        f"{SAL_MD.relative_to(RAIZ)} | tiempo {salida['tiempo_s']} s")


if __name__ == "__main__":
    main()
