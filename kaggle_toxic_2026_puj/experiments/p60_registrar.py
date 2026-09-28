"""P60 helper: register a track's outcome in every registry that must agree.

Why a helper and not one block per track
---------------------------------------
Eight tracks produce the same thirteen numbers and each has to land in four
places: ``experiments.csv`` (only if an experiment was really trained), the P60
hypothesis registry, the P60 master results table, and the gap registry. Doing
that by hand eight times is eight chances to mistype a fold, and this project has
already paid twice inside P59 for transcription errors in a control.

So each track script computes its own JSON artifact, and this helper reads that
artifact and propagates the numbers. No track script writes a number twice, and no
number is ever retyped.

It also enforces the project rule that is easiest to break by accident: a track
closed without training must NOT create a row in ``experiments.csv``.

The delta is recorded with its DENOMINATOR. A track may gate on the effect of its
mechanism against its own control, or against V03, and those two numbers can have
different signs. Reading a delta without knowing what it is measured against is
how a marginal result gets promoted, so the table carries both.

Usage
-----
    /home/carlos/venv/bin/python experiments/p60_registrar.py P60-G
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REG = PROJECT_ROOT / "experiments.csv"
REPORTS = PROJECT_ROOT / "reports"
CSV_HIPO = REPORTS / "p60_hypothesis_registry.csv"
CSV_MASTER = REPORTS / "p60_master_results.csv"
CSV_GAP = REPORTS / "p60_gap_registry.csv"

# Pista -> (json, md), en el orden en que se ejecutan.
ARTEFACTOS: dict[str, tuple[str, str]] = {
    "P60-G": ("p60_G_contrastive.json", "p60_G_contrastive.md"),
    "P60-A": ("p60_A_asl.json", "p60_A_asl.md"),
    "P60-L": ("p60_L_dro.json", "p60_L_dro.md"),
    "P60-D": ("p60_D_distribucion.json", "p60_D_distribucion.md"),
    "P60-C": ("p60_C3_intervencion_ag.json", "p60_C3_intervencion_ag.md"),
    "P60-C1C2": ("p60_C_ruido_por_etiqueta.json", "p60_C_ruido_por_etiqueta.md"),
    "P60-H": ("p60_H_transporte.json", "p60_H_transporte.md"),
    "P60-I": ("p60_I_metareweight.json", "p60_I_metareweight.md"),
    "P60-J": ("p60_J_polyloss.json", "p60_J_polyloss.md"),
}

CLAVES_F1 = ("f1_por_etiqueta_brazo_1", "f1_por_etiqueta_candidato", "f1_por_etiqueta")
CLAVES_FOLDS = ("folds_brazo_1", "folds_candidato", "folds")
CLAVES_DELTA_MEC = ("delta_mecanismo_puro", "delta_contraste_puro", "delta_perdida_puro",
                    "delta_agrupado_puro", "delta_intervencion_puro",
                    "delta_divergencia_puro", "delta_meta_puro",
                    "delta_vs_control", "delta")
CLAVES_DELTA_V03 = ("delta_vs_V03", "delta_vs_V03_brazo_1", "delta_vs_V03_candidato")


def primero(d: dict[str, Any], claves: tuple[str, ...]) -> Any:
    """Devuelve el primer valor no nulo entre varias claves.

    Args:
        d: diccionario de la pista.
        claves: claves candidatas, en orden de prioridad.

    Returns:
        El primer valor encontrado, o None.
    """
    for c in claves:
        if d.get(c) is not None:
            return d[c]
    return None


def commit_actual() -> str:
    """Devuelve el commit del HEAD.

    Returns:
        Hash abreviado del commit actual.
    """
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                       text=True, cwd=PROJECT_ROOT, check=False)
    return r.stdout.strip() or "desconocido"


def main() -> None:
    """Propaga el resultado de una pista a los cuatro registros.

    Returns:
        None.
    """
    warnings.simplefilter("ignore", FutureWarning)
    if len(sys.argv) != 2:
        raise SystemExit("uso: p60_registrar.py <P60-X>")
    pista = sys.argv[1].strip().upper()
    if pista not in ARTEFACTOS:
        raise SystemExit(f"PARADA: no hay artefacto declarado para {pista}")
    js_name, md_name = ARTEFACTOS[pista]
    js = REPORTS / js_name
    if not js.exists():
        raise SystemExit(f"PARADA: falta {js_name}")
    d = json.loads(js.read_text(encoding="utf-8"))

    res = d.get("resultado", {})
    decision = d.get("decision", "PENDIENTE")
    entreno = int(d.get("experimentos_registrados", 0)) > 0
    cfg = d.get("configuracion_pre_registrada", {})
    gap = d.get("gap_analysis", {})
    f1 = primero(res, CLAVES_F1) or {}
    folds = primero(res, CLAVES_FOLDS) or []
    delta_mec = primero(res, CLAVES_DELTA_MEC)
    delta_v03 = primero(res, CLAVES_DELTA_V03)
    ccv = res.get("brazo_1_corrected_cv") or res.get("corrected_cv_candidato") \
        or res.get("corrected_cv")

    # ---------- 1. experiments.csv, SOLO si hubo entrenamiento real ----------
    if entreno:
        df = pd.read_csv(REG)
        eid = str(d.get("experiment_id", pista))
        if eid in set(df.experiment_id):
            print(f"  {eid} ya esta en experiments.csv, no se duplica")
        else:
            fila: dict[str, Any] = {
                "experiment_id": eid,
                "competition": "toxic-2026-puj",
                "question": str(d.get("pregunta", d.get("paper", "")))[:400],
                "hypothesis": str(d.get("hipotesis", ""))[:600],
                "preprocessing": str(cfg)[:600],
                "feature_groups": str(d.get("rasgos", "NINGUNO NUEVO"))[:300],
                "representation": str(d.get("adaptacion_legal", ""))[:500],
                "model": str(d.get("modelo", ""))[:400],
                "hyperparameters": json.dumps(cfg, ensure_ascii=False)[:1800],
                "validation": ("StratifiedGroupKFold 5 con los grupos canonicos de TOXIC; 1415 "
                               "grupos, 0 partidos. 0 holdout, 0 test, 0 leaderboard, "
                               "0 submissions"),
                "f1": round(float(f1["HS"]), 6) if f1.get("HS") is not None else None,
                "macro_f1": ccv,
                "training_time": d.get("segundos"),
                "dimension": str(d.get("dimension", "ver configuracion"))[:200],
                "result": json.dumps(res, ensure_ascii=False)[:3000],
                "interpretation": str(d.get("interpretacion", ""))[:1500],
                "decision": decision[:900],
                "date": str(d.get("fecha", ""))[:10],
                "script": str(d.get("script", "")),
                "notes": (f"git_commit {commit_actual()} | seed 42 | fase P60 | "
                          f"paper: {d.get('paper', 'n/a')} | "
                          f"delta_mecanismo {delta_mec} | delta_vs_V03 {delta_v03} | "
                          f"0 transformers, 0 BERT, 0 RoBERTa, 0 LLM, 0 embeddings preentrenados, "
                          f"0 clasificadores externos, 0 etiquetas externas, 0 pseudo-etiquetas, "
                          f"0 test, 0 holdout, 0 leaderboard, 0 submissions | "
                          f"artefactos: {js_name}, {md_name}"),
            }
            nuevo = pd.DataFrame([fila])
            for c in df.columns:
                if c not in nuevo.columns:
                    nuevo[c] = "NINGUNO"
            pd.concat([df, nuevo[df.columns]], ignore_index=True).to_csv(
                REG, index=False, encoding="utf-8")
            print(f"  experiments.csv: alta {eid}, ahora {len(df) + 1} filas")

    # ---------- 2. registro de hipotesis ----------
    if CSV_HIPO.exists():
        hi = pd.read_csv(CSV_HIPO)
        m = hi["ID"] == pista
        if bool(m.any()):
            hi.loc[m, "Experiment"] = d.get("experiment_id", pista) if entreno else "NINGUNO"
            hi.loc[m, "Result"] = json.dumps(
                {"corrected_cv": ccv, "delta_mecanismo": delta_mec, "delta_vs_V03": delta_v03,
                 "folds": folds, "neto": res.get("neto")}, ensure_ascii=False)[:400]
            hi.loc[m, "Status"] = decision.split(".")[0][:140]
            hi.loc[m, "Decision"] = decision[:600]
            hi.to_csv(CSV_HIPO, index=False, encoding="utf-8")
            print(f"  p60_hypothesis_registry.csv actualizada para {pista}")

    # ---------- 3. tabla maestra ----------
    if CSV_MASTER.exists():
        ma = pd.read_csv(CSV_MASTER)
        if pista not in set(ma["ID"]):
            fila_m: dict[str, Any] = {
                "ID": pista,
                "Track": pista.replace("P60-", ""),
                "Paper_or_reference": str(d.get("paper", "")),
                "Hypothesis": str(d.get("hipotesis", ""))[:300],
                "Novelty": gap.get("veredicto", "n/a"),
                "Mechanism": str(d.get("mecanismo", ""))[:400],
                "Representation": str(d.get("adaptacion_legal", ""))[:200],
                "Model": str(d.get("modelo", ""))[:200],
                "Loss": str(d.get("perdida", ""))[:250],
                "Configuration": str(cfg)[:300],
                "Corrected_CV": ccv,
                "Fold_1": folds[0] if len(folds) > 0 else None,
                "Fold_2": folds[1] if len(folds) > 1 else None,
                "Fold_3": folds[2] if len(folds) > 2 else None,
                "Fold_4": folds[3] if len(folds) > 3 else None,
                "Fold_5": folds[4] if len(folds) > 4 else None,
                "Delta_vs_V03": delta_v03,
                "Delta_mecanismo": delta_mec,
                "Denominador_del_delta": ("efecto del mecanismo contra su control"
                                          if delta_mec is not None else "contra V03"),
                "F1_HS": f1.get("HS"), "F1_TR": f1.get("TR"), "F1_AG": f1.get("AG"),
                "Rescue": res.get("rescate"), "Damage": res.get("damage"),
                "Net": res.get("neto"),
                "Bootstrap_CI": "NO EJECUTADO (gate no superado)",
                "Decision": decision[:500],
            }
            nuevo_m = pd.DataFrame([fila_m])
            for c in ma.columns:
                if c not in nuevo_m.columns:
                    nuevo_m[c] = None
            ma = pd.concat([ma, nuevo_m[ma.columns]], ignore_index=True)
            ma.to_csv(CSV_MASTER, index=False, encoding="utf-8")
            print(f"  p60_master_results.csv: alta {pista}, ahora {len(ma)} filas")
        else:
            print(f"  {pista} ya esta en p60_master_results.csv")

    # ---------- 4. gap registry: estado final ----------
    if CSV_GAP.exists():
        ga = pd.read_csv(CSV_GAP)
        m = ga["track"] == pista.replace("P60-", "")
        if bool(m.any()):
            ga.loc[m, "abierto"] = False
            ga.loc[m, "motivo"] = decision[:500]
            ga.to_csv(CSV_GAP, index=False, encoding="utf-8")
            print(f"  p60_gap_registry.csv: {pista} marcada como ejecutada")

    print(f"  registro de {pista} completado | decision: {decision.split('.')[0][:90]}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
