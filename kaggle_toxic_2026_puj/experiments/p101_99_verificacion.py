#!/usr/bin/env python
"""P101-99: verificacion de cierre de P101 (auditoria de supervision).

Solo lectura. Comprueba entregables, tabla de procedencia, registros
(experiments.csv, RUNLOG, DECISIONS, PROJECT_STATE), notebook seccion 164,
addendum retrospectivo, hashes canonicos V03/P98/P99/P100, ASCII,
verificador oficial y Git.

Uso:
    /home/carlos/venv/bin/python experiments/p101_99_verificacion.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
R = RAIZ / "reports"

ENTREGABLES = ["p101_provenance_audit.md", "p101_provenance_table.csv",
               "p101_materials_inventory.csv", "p101_governance_note.md",
               "p101_compliance.md", "p101_final_report.md",
               "p101_reproducibility.md"]
SCRIPTS = ["p101_00_auditoria_procedencia.py", "p101_90_registrar.py",
           "p101_95_integrar_notebook.py", "p101_99_verificacion.py"]

CANONICOS = {
    "data/raw/toxic_train.csv": "3d40e7345b588860",
    "features/oof_v03_final.npz": "559265816ce4f631",
    "experiments/baseline_v03_frozen/submission_v03.csv": "7d7478f271e6edfd",
    "experiments/baseline_v03_frozen/model_v03.joblib": "263ed5bd0fc0de7c",
    "data/processed/holdout_v2_ids.csv": "e5b62e5f22c4c63a",
    "data/processed/holdout2_ids.csv": "f6c6ed958abf728b",
    "data/external/corpora/p98_c98_1/tokens_30m.txt": "9f562718ef4e7d45",
    "features/p98_c98_1_vocab.json": "92fecce611f164b7",
    "models/p98_c98_1_ltr.pt": "472891a7598a1daa",
    "models/p98_c98_1_rtl.pt": "cff68d45bf683cad",
    "features/p98_13_reps_dev.npz": "8f1aeeecaf346a80",
    "data/external/corpora/p99_s99_01/tokens_60m.txt": "e627a7f33886c0a1",
    "models/p99_s99_01_ltr.pt": "559309e0042d8c44",
    "models/p99_s99_01_rtl.pt": "dde2147f1147d142",
    "features/p99_reps_dev.npz": "3e72755bb7f03f48",
    "data/external/corpora/p100_b1/tokens_30m.txt": "5111a1832fe9567b",
    "models/p100_b1_ltr.pt": "a4c057a762500f54",
    "models/p100_b1_rtl.pt": "fb38bbbdac055732",
    "features/p100_b1_reps_dev.npz": "6524eae950b242e4",
}


def sha256(ruta: Path) -> str:
    """sha256 completo."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()


def puro_ascii(ruta: Path) -> bool:
    """True si no hay bytes > 127."""
    with open(ruta, "rb") as fh:
        return all(b <= 127 for b in fh.read())


def main() -> None:
    """Ejecuta la verificacion de cierre de P101."""
    chequeos: dict[str, bool] = {}
    for nombre in ENTREGABLES:
        chequeos[f"existe_{nombre}"] = (R / nombre).exists()
    for nombre in SCRIPTS:
        chequeos[f"existe_{nombre}"] = (RAIZ / "experiments" / nombre).exists()

    with open(R / "p101_provenance_table.csv", encoding="utf-8") as fh:
        tabla = list(csv.DictReader(fh))
    chequeos["tabla_1327"] = len(tabla) == 1327
    chequeos["elegibles_971"] = sum(
        1 for r in tabla
        if r["eligibility_new_audit"] == "True") == 971
    chequeos["main_250_cal_15"] = (
        sum(1 for r in tabla if r["v2_1_main_sample"] == "True") == 250
        and sum(1 for r in tabla
                if r["v2_1_calibration_no_response"] == "True") == 15)
    chequeos["ronda2_127"] = sum(
        1 for r in tabla if int(r["r2_c12_cells"]) > 0) == 127
    chequeos["sin_reservados_main"] = all(
        r["source_partition"] != "test"
        for r in tabla if r["v2_1_main_sample"] == "True")

    texto_final = (R / "p101_final_report.md").read_text(encoding="utf-8")
    texto_gob = (R / "p101_governance_note.md").read_text(encoding="utf-8")
    chequeos["final_closed_blocked"] = "CLOSED_BLOCKED" in texto_final
    chequeos["gob_ambiguous"] = "AMBIGUOUS" in texto_gob

    texto_runlog = (RAIZ / "RUNLOG.md").read_text(encoding="utf-8")
    texto_dec = (RAIZ / "DECISIONS.md").read_text(encoding="utf-8")
    texto_estado = (RAIZ / "PROJECT_STATE.md").read_text(encoding="utf-8")
    marca = "Addendum retrospectivo (2026-10-14): trazabilidad de P91-P93"
    chequeos["addendum_runlog"] = marca in texto_runlog
    chequeos["addendum_decisions"] = marca in texto_dec
    chequeos["decisions_d305"] = "## D-305" in texto_dec
    chequeos["estado_p101"] = ("P101 = CLOSED_BLOCKED" in texto_estado
                               and "P101" in texto_runlog)

    with open(RAIZ / "experiments.csv", encoding="utf-8-sig") as fh:
        filas = list(csv.DictReader(fh))
    ids = [f["experiment_id"] for f in filas]
    chequeos["experiments_492"] = len(filas) == 492
    chequeos["ids_unicos"] = len(set(ids)) == len(ids)
    chequeos["p101_00"] = "P101-00" in ids
    fila = next(f for f in filas if f["experiment_id"] == "P101-00")
    chequeos["sync_p101"] = (json.loads(fila["result"])["estado"]
                             == "CLOSED_BLOCKED_NO_REVIEWERS")

    nb = json.loads((RAIZ / "notebooks" / "10_final_pipeline.ipynb").read_text(
        encoding="utf-8"))
    chequeos["notebook_164"] = any(
        "".join(c.get("source", [])).startswith("## 164. P101")
        for c in nb["cells"])
    chequeos["notebook_codigo_164"] = any(
        "Verificacion de la seccion 164 (P101" in "".join(c.get("source", []))
        for c in nb["cells"])

    for nombre in ENTREGABLES:
        chequeos[f"ascii_{nombre}"] = puro_ascii(R / nombre)
    for nombre in SCRIPTS:
        chequeos[f"ascii_{nombre}"] = puro_ascii(RAIZ / "experiments" / nombre)

    for ruta, pref in CANONICOS.items():
        chequeos[f"canonico:{ruta}"] = sha256(RAIZ / ruta).startswith(pref)

    verif = subprocess.run(
        [sys.executable, str(RAIZ / "tools" / "verificar_caracteres.py")],
        capture_output=True, text=True, check=False)
    chequeos["verificador_oficial_ok"] = verif.returncode == 0

    git_st = subprocess.run(["git", "-C", str(RAIZ), "status", "--short"],
                            capture_output=True, text=True,
                            check=False).stdout.strip()
    lineas_git = git_st.splitlines()
    chequeos["test_holdouts_sin_cambios_git"] = not any(
        (l[:2] in (" M", "M ", "MM"))
        and ("toxic_test" in l or "holdout" in l) for l in lineas_git)

    n_untracked = sum(1 for l in lineas_git if l.startswith("??"))
    n_modified = sum(1 for l in lineas_git if l[:2] in (" M", "M ", "MM"))
    fallos = [k for k, v in chequeos.items() if not v]
    print("P101-99 verificacion de cierre: ejecutada")
    print(f"  chequeos: {sum(chequeos.values())}/{len(chequeos)} | "
          f"fallos: {fallos if fallos else 'ninguno'}")
    print(f"  verificador oficial: {'OK' if verif.returncode == 0 else 'FALLO'}")
    print(f"  git: {n_untracked} sin rastrear, {n_modified} modificados "
          f"(solo lectura)")
    print()
    print("  limites declarados (por regla de no acceso):")
    print("   - toxic_test: consultado SOLO por columna de id (1154 ids);")
    print("     sin texto, labels ni scores.")
    print("   - holdout/holdout2: solo columnas de id / particion.")
    print("   - sin anotacion ejecutada ni simulada; P6 no ejecutado.")


if __name__ == "__main__":
    main()
