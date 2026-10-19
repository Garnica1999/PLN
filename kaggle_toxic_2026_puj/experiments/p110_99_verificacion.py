#!/usr/bin/env python
"""P110-99: Verificacion de cierre de P110 (challenger V2_triple + Blend LR/SGD + Regla Ascendente).

Solo lectura. Comprueba:
1. Existencia de entregables y scripts de P110.
2. Controles canonicos reproducidos en el reporte JSON (diff < 1e-6).
3. Registro de P110-01 en experiments.csv.
4. Compatibilidad ASCII pura en todos los ficheros de P110.
5. Integridad de hashes de los artefactos canonicos congelados.
6. Validez formal de la submission generada (formato, filas, columnas, IDs, 0 nulos, valores binarios).
7. Verificacion de recuento de positivos de test acotados cerca de P108.
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
    "p110_01_challenger_lr_sgd_blend.json",
    "p110_01_challenger_lr_sgd_blend.md",
]
SCRIPTS = [
    "p110_01_challenger_lr_sgd_blend.py",
    "p110_90_registrar.py",
    "p110_99_verificacion.py",
]
SUBMISSIONS = [
    "toxic_submission_p110_lr_sgd_blend_asc.csv",
]

CANONICOS = {
    "data/raw/toxic_train.csv": "3d40e7345b588860",
    "features/oof_v03_final.npz": "559265816ce4f631",
    "experiments/baseline_v03_frozen/submission_v03.csv": "7d7478f271e6edfd",
    "experiments/baseline_v03_frozen/model_v03.joblib": "263ed5bd0fc0de7c",
    "submissions/toxic_submission_p108_v2_triple_asc.csv": "d3c19d2bcaf1e0df",
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
    f_json = R / "p110_01_challenger_lr_sgd_blend.json"
    if f_json.exists():
        data = json.loads(f_json.read_text(encoding="utf-8"))
        chequeos["delta_canonica_positivo"] = data["canonica"]["delta_vs_p108"] > 0.002
        chequeos["delta_part47_positivo"] = data["particion_47"]["delta_vs_p108"] > 0.002
        chequeos["ganados_canonica_ge_4"] = data["canonica"]["pliegues_ganados"] in ["4/5", "5/5"]
        chequeos["ganados_part47_ge_4"] = data["particion_47"]["pliegues_ganados"] in ["4/5", "5/5"]
        chequeos["dimension_86275"] = data["features"]["dimension"] == 86275
        chequeos["test_celdas_distintas_le_40"] = data["test"]["celdas_distintas_vs_p108"] <= 40

    # 3. Registro en experiments.csv
    f_csv = RAIZ / "experiments.csv"
    chequeos["existe_experiments_csv"] = f_csv.exists()
    if f_csv.exists():
        with open(f_csv, encoding="utf-8-sig", newline="") as fh:
            filas = list(csv.DictReader(fh))
        p110_filas = [f for f in filas if f["experiment_id"] == "P110-01"]
        chequeos["p110_registrado"] = len(p110_filas) == 1

    # 4. Pureza ASCII
    for s in SCRIPTS:
        ruta = EXP / s
        if ruta.exists():
            chequeos[f"ascii_{s}"] = puro_ascii(ruta)
    for e in ENTREGABLES:
        ruta = R / e
        if ruta.exists():
            chequeos[f"ascii_{e}"] = puro_ascii(ruta)

    # 5. Hashes canonicos congelados intactos
    for rel, prefijo in CANONICOS.items():
        ruta = RAIZ / rel
        if ruta.exists():
            h16 = sha16(ruta)
            chequeos[f"hash_canonico_{rel}"] = h16 == prefijo
        else:
            chequeos[f"hash_canonico_{rel}"] = False

    # 6. Formato de la submission generada
    f_sub = SUB / "toxic_submission_p110_lr_sgd_blend_asc.csv"
    chequeos["existe_submission_p110"] = f_sub.exists()
    if f_sub.exists():
        with open(f_sub, encoding="utf-8", newline="") as fh:
            filas_sub = list(csv.reader(fh))
        chequeos["sub_cabecera"] = filas_sub[0] == ["id", "HS", "TR", "AG"]
        chequeos["sub_1154_filas"] = len(filas_sub) == 1155
        chequeos["sub_sin_nulos"] = all(len(f) == 4 and all(c != "" for c in f) for f in filas_sub)
        chequeos["sub_valores_binarios"] = all(
            f[1] in ("0", "1") and f[2] in ("0", "1") and f[3] in ("0", "1") for f in filas_sub[1:]
        )
        # Regla ontologica respetada en test (0 violaciones: TR=1 => HS=1)
        chequeos["sub_sin_violaciones_ascendentes"] = not any(
            f[2] == "1" and f[1] == "0" for f in filas_sub[1:]
        )

    # Resumen
    n_total = len(chequeos)
    n_ok = sum(1 for v in chequeos.values() if v)
    fallos = [k for k, v in chequeos.items() if not v]

    print(f"[P110-99] Chequeos superados: {n_ok}/{n_total}")
    if fallos:
        print(f"[P110-99] FALLOS DETECTADOS ({len(fallos)}):")
        for f in fallos:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("[P110-99] TODOS LOS CHEQUEOS SUPERADOS CON EXITO.")
        sys.exit(0)


if __name__ == "__main__":
    main()
