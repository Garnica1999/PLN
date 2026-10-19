#!/usr/bin/env python
"""P108-99: Verificacion de cierre de P108 (challenger V2_triple_asc).

Solo lectura. Comprueba:
1. Existencia de entregables y scripts de P108.
2. Controles canonicos reproducidos en el reporte JSON.
3. Registro de P108-01 en experiments.csv.
4. Compatibilidad ASCII pura en todos los ficheros de P108.
5. Integridad de hashes de los artefactos canonicos congelados.
6. Validez formal de la submission generada (formato, filas, columnas, IDs).
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
R = RAIZ / "reports"
EXP = RAIZ / "experiments"
SUB = RAIZ / "submissions"

ENTREGABLES = [
    "p108_01_challenger_v2_triple_asc.json",
    "p108_01_challenger_v2_triple_asc.md",
]
SCRIPTS = [
    "p108_01_challenger_v2_triple_asc.py",
    "p108_90_registrar.py",
    "p108_99_verificacion.py",
]
SUBMISSIONS = [
    "toxic_submission_p108_v2_triple_asc.csv",
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

    # 2. Control canonico en el reporte JSON
    f_json = R / "p108_01_challenger_v2_triple_asc.json"
    if f_json.exists():
        data = json.loads(f_json.read_text(encoding="utf-8"))
        chequeos["control_canonica_ok"] = bool(data["control_canonica"]["ok"])
        chequeos["diff_max_cero"] = data["control_canonica"]["diff_max"] == 0.0
        chequeos["ccv_exacto"] = data["control_canonica"]["ccv"] == 0.777211
        chequeos["delta_canonica_positivo"] = data["metricas_dev_canonica"]["delta"] > 0.005
        chequeos["delta_part47_positivo"] = data["metricas_dev_part47"]["delta"] > 0.005
        chequeos["ganados_canonica_ge_4"] = data["metricas_dev_canonica"]["pliegues_ganados"] >= 4
        chequeos["ganados_part47_5"] = data["metricas_dev_part47"]["pliegues_ganados"] == 5

    # 3. Registro en experiments.csv
    f_csv = RAIZ / "experiments.csv"
    chequeos["existe_experiments_csv"] = f_csv.exists()
    if f_csv.exists():
        with open(f_csv, encoding="utf-8-sig", newline="") as fh:
            filas = list(csv.DictReader(fh))
        p108_filas = [f for f in filas if f["experiment_id"] == "P108-01"]
        chequeos["p108_registrado"] = len(p108_filas) == 1

    # 4. Pureza ASCII
    for e in ENTREGABLES:
        p = R / e
        if p.exists():
            chequeos[f"ascii_{e}"] = puro_ascii(p)
    for s in SCRIPTS:
        p = EXP / s
        if p.exists():
            chequeos[f"ascii_{s}"] = puro_ascii(p)

    # 5. Hashes canonicos
    for rel, hash_esp in CANONICOS.items():
        ruta = RAIZ / rel
        chequeos[f"canonico_{rel.replace('/', '_')}"] = (
            ruta.exists() and sha16(ruta) == hash_esp
        )

    # 6. Formato de la submission generada
    sub_path = SUB / "toxic_submission_p108_v2_triple_asc.csv"
    template_path = RAIZ / "data" / "raw" / "submission.csv"
    if sub_path.exists() and template_path.exists():
        import pandas as pd
        df_sub = pd.read_csv(sub_path)
        df_tmp = pd.read_csv(template_path)
        chequeos["sub_columnas_ok"] = list(df_sub.columns) == list(df_tmp.columns)
        chequeos["sub_filas_ok"] = len(df_sub) == len(df_tmp)
        chequeos["sub_ids_ok"] = (df_sub["id"].to_numpy() == df_tmp["id"].to_numpy()).all()
        chequeos["sub_valores_binarios"] = bool(
            np.isin(df_sub[["HS", "TR", "AG"]].to_numpy(), [0, 1]).all()
        )
        chequeos["sub_sin_violaciones_asc"] = int(
            ((df_sub["TR"] == 1) & (df_sub["HS"] == 0)).sum()
        ) == 0

    total = len(chequeos)
    superados = sum(1 for v in chequeos.values() if v)
    fallos = [k for k, v in chequeos.items() if not v]

    print(f"P108-99 Verificacion: {superados}/{total} chequeos superados.")
    if fallos:
        print(f"FALLOS ({len(fallos)}): {fallos}")
        sys.exit(1)
    else:
        print("TODOS LOS CHEQUEOS DE P108 SUPERADOS EXITOSAMENTE.")


if __name__ == "__main__":
    import numpy as np
    main()
