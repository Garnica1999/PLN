#!/usr/bin/env python
"""P109-99: Verificacion de cierre de P109 (challenger V2_triple min_df=5).

Solo lectura. Comprueba:
1. Existencia de entregables, scripts y submission de P109.
2. Metricas y controles de generalizacion en el reporte JSON.
3. Registro de P109-01 en experiments.csv.
4. Compatibilidad ASCII pura en todos los ficheros de P109.
5. Integridad de hashes de los artefactos canonicos congelados.
6. Validez formal de la submission generada (formato, filas, columnas, IDs).
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
R = RAIZ / "reports"
EXP = RAIZ / "experiments"
SUB = RAIZ / "submissions"

ENTREGABLES = [
    "p109_01_challenger_mdf5_v2.json",
    "p109_01_challenger_mdf5_v2.md",
]
SCRIPTS = [
    "p109_01_challenger_mdf5_v2.py",
    "p109_90_registrar.py",
    "p109_99_verificacion.py",
]
SUBMISSIONS = [
    "toxic_submission_p109_mdf5_v2_asc.csv",
]

CANONICOS = {
    "data/raw/toxic_train.csv": "3d40e7345b588860",
    "features/oof_v03_final.npz": "559265816ce4f631",
    "experiments/baseline_v03_frozen/submission_v03.csv": "7d7478f271e6edfd",
    "experiments/baseline_v03_frozen/model_v03.joblib": "263ed5bd0fc0de7c",
}


def sha16(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()[:16]


def puro_ascii(ruta: Path) -> bool:
    with open(ruta, "rb") as fh:
        return all(b <= 127 for b in fh.read())


def main() -> None:
    chequeos: dict[str, bool] = {}

    # 1. Existencia de entregables, scripts y submissions
    for e in ENTREGABLES:
        chequeos[f"existe_reporte_{e}"] = (R / e).exists()
    for s in SCRIPTS:
        chequeos[f"existe_script_{s}"] = (EXP / s).exists()
    for sub in SUBMISSIONS:
        chequeos[f"existe_submission_{sub}"] = (SUB / sub).exists()

    # 2. Control y metricas en reporte JSON
    f_json = R / "p109_01_challenger_mdf5_v2.json"
    if f_json.exists():
        data = json.loads(f_json.read_text(encoding="utf-8"))
        can = data["metricas_dev_canonica"]
        p47 = data["metricas_dev_particion47"]
        chequeos["canonica_f1_ge_0.79"] = can["p109_ccv"] >= 0.790
        chequeos["delta_canonica_gt_0.007"] = can["delta_vs_p108"] > 0.007
        chequeos["ganados_canonica_5_5"] = can["folds_ganados_vs_p108"] == "5/5"
        chequeos["bootstrap_canonica_win_gt_95"] = can["bootstrap_clusters_vs_p108"]["pct_mejora"] > 0.95
        chequeos["part47_f1_ge_0.788"] = p47["p109_part47"] >= 0.788
        chequeos["delta_part47_gt_0.005"] = p47["delta_vs_p108"] > 0.005
        chequeos["ganados_part47_5_5"] = p47["folds_ganados_vs_p108"] == "5/5"
        chequeos["bootstrap_part47_win_gt_95"] = p47["bootstrap_clusters_vs_p108"]["pct_mejora"] > 0.95
        chequeos["features_parsimoniosas_le_35k"] = data["hiperparametros"]["num_features_train_completo"] <= 35000

    # 3. Registro en experiments.csv
    f_csv = RAIZ / "experiments.csv"
    chequeos["existe_experiments_csv"] = f_csv.exists()
    if f_csv.exists():
        with open(f_csv, encoding="utf-8-sig", newline="") as fh:
            filas = list(csv.DictReader(fh))
        p109_filas = [f for f in filas if f["experiment_id"] == "P109-01"]
        chequeos["p109_registrado"] = len(p109_filas) == 1

    # 4. Pureza ASCII
    for e in ENTREGABLES:
        p = R / e
        if p.exists():
            chequeos[f"ascii_{e}"] = puro_ascii(p)
    for s in SCRIPTS:
        p = EXP / s
        if p.exists():
            chequeos[f"ascii_{s}"] = puro_ascii(p)

    # 5. Integridad de los artefactos canonicos congelados
    for rel_path, hash_esp in CANONICOS.items():
        p = RAIZ / rel_path
        if p.exists():
            chequeos[f"canonico_{p.name}"] = sha16(p) == hash_esp
        else:
            chequeos[f"canonico_{p.name}"] = False

    # 6. Validez formal de la submission
    sub_path = SUB / "toxic_submission_p109_mdf5_v2_asc.csv"
    plantilla_path = RAIZ / "data" / "raw" / "submission.csv"
    if sub_path.exists() and plantilla_path.exists():
        sub_df = pd.read_csv(sub_path)
        plantilla_df = pd.read_csv(plantilla_path)
        chequeos["sub_columnas_exactas"] = list(sub_df.columns) == list(plantilla_df.columns)
        chequeos["sub_filas_exactas"] = len(sub_df) == len(plantilla_df)
        chequeos["sub_ids_coinciden"] = (sub_df["id"].to_numpy() == plantilla_df["id"].to_numpy()).all()
        chequeos["sub_sin_nulos"] = sub_df.isna().sum().sum() == 0
        chequeos["sub_valores_binarios"] = bool(np.isin(sub_df[["HS", "TR", "AG"]].to_numpy(), [0, 1]).all())
        # Regla ascendente: no debe haber casos TR=1 & HS=0
        violaciones = ((sub_df["TR"] == 1) & (sub_df["HS"] == 0)).sum()
        chequeos["sub_cero_violaciones_ascendentes"] = violaciones == 0

    total = len(chequeos)
    pasados = sum(1 for v in chequeos.values() if v)
    fallados = total - pasados

    print("=" * 60)
    print("VERIFICACION DE CIERRE P109 (Challenger V2_triple min_df=5)")
    print("=" * 60)
    for nombre, ok in chequeos.items():
        simbolo = "[OK]" if ok else "[FALLO]"
        print(f"  {simbolo} {nombre}")
    print("-" * 60)
    print(f"Total chequeos: {total} | Pasados: {pasados} | Fallados: {fallados}")
    print("=" * 60)

    if fallados > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
