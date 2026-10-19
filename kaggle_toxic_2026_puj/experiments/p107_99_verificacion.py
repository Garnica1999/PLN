#!/usr/bin/env python
"""P107-99: Verificacion de cierre de P107 (arquitecturas, aumento de datos, ensembles).

Solo lectura. Comprueba:
1. Existencia de entregables y scripts de P107.
2. Controles canonicos reproducidos en el reporte JSON.
3. Registro de P107-01 en experiments.csv.
4. Compatibilidad ASCII pura en todos los ficheros de P107.
5. Integridad de hashes de los artefactos canonicos congelados.
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

ENTREGABLES = [
    "p107_01_arquitecturas_aumento_ensemble.json",
    "p107_01_arquitecturas_aumento_ensemble.md",
]
SCRIPTS = [
    "p107_01_arquitecturas_aumento_ensemble.py",
    "p107_90_registrar.py",
    "p107_99_verificacion.py",
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

    # 1. Existencia de entregables y scripts
    for e in ENTREGABLES:
        chequeos[f"existe_reporte_{e}"] = (R / e).exists()
    for s in SCRIPTS:
        chequeos[f"existe_script_{s}"] = (EXP / s).exists()

    # 2. Validacion del reporte JSON
    json_path = R / "p107_01_arquitecturas_aumento_ensemble.json"
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        chequeos["fase_p107"] = data.get("fase") == "P107"
        chequeos["control_canonica_ok"] = data["control_canonica"]["ok"] is True
        chequeos["control_canonica_ccv"] = abs(data["control_canonica"]["ccv"] - 0.7772114525165721) < 1e-12
        chequeos["control_part47_ok"] = data["control_particion_47"]["ok"] is True
        chequeos["control_part47_ccv"] = abs(data["control_particion_47"]["ccv"] - 0.7756319868803105) < 1e-12
        chequeos["control_part47_celdas0"] = data["control_particion_47"]["celdas_distintas"] == 0
        chequeos["variantes_4"] = len(data.get("variantes", {})) == 4
    else:
        chequeos["json_presente"] = False

    # 3. Registro en experiments.csv
    with open(RAIZ / "experiments.csv", encoding="utf-8-sig", newline="") as fh:
        filas = list(csv.DictReader(fh))
    ids = [f["experiment_id"] for f in filas]
    chequeos["p107_en_experiments_csv"] = "P107-01" in ids

    # 4. Pure ASCII en ficheros P107
    todos_ficheros = [R / e for e in ENTREGABLES] + [EXP / s for s in SCRIPTS]
    for f in todos_ficheros:
        if f.exists():
            chequeos[f"ascii_{f.name}"] = puro_ascii(f)

    # 5. Hashes canonicos
    for rel_path, prefijo_esperado in CANONICOS.items():
        ruta = RAIZ / rel_path
        if ruta.exists():
            h = sha16(ruta)
            chequeos[f"hash_{rel_path}"] = h == prefijo_esperado
        else:
            chequeos[f"existe_{rel_path}"] = False

    fallos = [k for k, v in chequeos.items() if not v]
    print(f"P107-99 Verificacion: {len(chequeos) - len(fallos)}/{len(chequeos)} chequeos superados.")
    if fallos:
        print(f"FALLOS DETECTADOS ({len(fallos)}): {fallos}")
        sys.exit(1)
    else:
        print("TODOS LOS CHEQUEOS DE P107 SUPERADOS EXITOSAMENTE.")


if __name__ == "__main__":
    main()
