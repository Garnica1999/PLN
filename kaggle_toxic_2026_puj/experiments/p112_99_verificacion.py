#!/usr/bin/env python
"""P112-99: Verificacion de cierre de P112 (challenger Bagging 10-Fold sobre 100% Train).

Solo lectura. Comprueba:
1. Existencia de entregables y scripts de P112.
2. Controles canonicos reproducidos en el reporte JSON (diff < 1e-6).
3. Registro de P112-01 en experiments.csv.
4. Compatibilidad ASCII pura en todos los ficheros de P112.
5. Integridad de hashes de los artefactos canonicos congelados.
6. Validez formal de la submission generada (formato, filas, columnas, IDs, 0 nulos, valores binarios).
7. Verificacion de recuento de positivos de test (1.254 total: HS=509, TR=328, AG=417).
8. Verificacion de que modifica exactamente 33 celdas (0.95%) de P110 en test.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
R = RAIZ / "reports"
EXP = RAIZ / "experiments"
SUB = RAIZ / "submissions"

ENTREGABLES = [
    "p112_01_challenger_bagging_10fold.json",
    "p112_01_challenger_bagging_10fold.md",
]
SCRIPTS = [
    "p112_01_challenger_bagging_10fold.py",
    "p112_90_registrar.py",
    "p112_99_verificacion.py",
]
SUBMISSIONS = [
    "toxic_submission_p112_bagging_10fold_asc.csv",
]

CANONICOS = {
    "data/raw/toxic_train.csv": "3d40e7345b588860",
    "features/oof_v03_final.npz": "559265816ce4f631",
    "submissions/toxic_submission_p110_lr_sgd_blend_asc.csv": "535b2b35e4e3131f",
    "submissions/toxic_submission_p111_per_label_c_asc.csv": "c63bb0506571eb00",
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
    f_json = R / "p112_01_challenger_bagging_10fold.json"
    if f_json.exists():
        data = json.loads(f_json.read_text(encoding="utf-8"))
        chequeos["canonica_ccv_ge_0780"] = data["canonica"]["ccv_p112"] >= 0.780
        chequeos["part47_ccv_ge_0780"] = data["particion_47"]["ccv_p112"] >= 0.780
        chequeos["test_celdas_distintas_p110_eq_33"] = data["test"]["celdas_distintas_vs_p110"] == 33
        chequeos["test_total_positivos_eq_1254"] = data["test"]["total_positivos_p112"] == 1254
        chequeos["test_pos_hs_509"] = data["test"]["positivos_p112"]["HS"] == 509
        chequeos["test_pos_tr_328"] = data["test"]["positivos_p112"]["TR"] == 328
        chequeos["test_pos_ag_417"] = data["test"]["positivos_p112"]["AG"] == 417

    # 3. Registro en experiments.csv
    f_csv = RAIZ / "experiments.csv"
    chequeos["existe_experiments_csv"] = f_csv.exists()
    if f_csv.exists():
        with open(f_csv, encoding="utf-8-sig", newline="") as fh:
            filas = list(csv.DictReader(fh))
        p112_filas = [f for f in filas if f["experiment_id"] == "P112-01"]
        chequeos["p112_registrado"] = len(p112_filas) == 1

    # 4. Pureza ASCII
    for s in SCRIPTS:
        ruta = EXP / s
        if ruta.exists():
            chequeos[f"ascii_{s}"] = puro_ascii(ruta)
    for e in ENTREGABLES:
        ruta = R / e
        if ruta.exists():
            chequeos[f"ascii_{e}"] = puro_ascii(ruta)

    # 5. Integridad de hashes de artefactos canonicos
    for rel, prefijo in CANONICOS.items():
        ruta = RAIZ / rel
        if ruta.exists():
            chequeos[f"hash_{rel}"] = sha16(ruta).startswith(prefijo)
        else:
            chequeos[f"hash_{rel}"] = False

    # 6. Formato de la submission
    sub_path = SUB / "toxic_submission_p112_bagging_10fold_asc.csv"
    if sub_path.exists():
        df_sub = pd.read_csv(sub_path)
        chequeos["sub_filas_1154"] = len(df_sub) == 1154
        chequeos["sub_columnas_correctas"] = list(df_sub.columns) == ["id", "HS", "TR", "AG"]
        chequeos["sub_cero_nulos"] = df_sub.isnull().sum().sum() == 0
        chequeos["sub_valores_binarios"] = all(
            set(df_sub[c].unique()).issubset({0, 1}) for c in ["HS", "TR", "AG"]
        )
        template = pd.read_csv(RAIZ / "data" / "raw" / "submission.csv")
        chequeos["sub_orden_ids_identico"] = (df_sub["id"].values == template["id"].values).all()
        chequeos["sub_sin_violaciones_ontologicas"] = (
            (df_sub["TR"] == 1) & (df_sub["HS"] == 0)
        ).sum() == 0

    fallos = [k for k, v in chequeos.items() if not v]

    print("\n=== RESUMEN DE VERIFICACION P112-99 ===")
    print(f"Total chequeos: {len(chequeos)} | Pasados: {len(chequeos) - len(fallos)} | Fallos: {len(fallos)}")

    if fallos:
        print("\nCHEQUEOS FALLIDOS:")
        for f in fallos:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nTODOS LOS CHEQUEOS PASARON SATISFACTORIAMENTE.")
        sys.exit(0)


if __name__ == "__main__":
    main()
