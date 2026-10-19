#!/usr/bin/env python
"""
P94-02: verificacion de la fase P94 (investigacion y diseno).

Comprueba (sin entrenar, sin predicciones, sin tocar recursos reservados):
- existencias y estructura de los entregables P94;
- coherencia del inventario (24 familias) y del registro (17 propuestas, 0 recomendadas);
- ASCII byte a byte de lo generado;
- integridad canonica (V03, OOF, submission, modelo, particion, respuestas, P93-R5);
- hashes completos de entrada y salida de P94;
- estado de Git (solo lectura).

Uso:
    /home/carlos/venv/bin/python experiments/p94_02_verificacion.py
"""

from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
R = RAIZ / "reports"

CLASES = {"GENUINELY_NEW", "ALREADY_TESTED", "DUPLICATE", "NOT_FEASIBLE",
          "INSUFFICIENT_EVIDENCE"}


def h(ruta: Path) -> str:
    """sha256 completo."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            x.update(b)
    return x.hexdigest()


def puro_ascii(ruta: Path) -> bool:
    """True si el archivo no tiene bytes > 127."""
    with open(ruta, "rb") as fh:
        return all(b <= 127 for b in fh.read())


def main() -> None:
    fallos: list[str] = []
    chequeos: dict[str, bool] = {}

    # ---------------------------------------------------- entregables P94
    rutas = {
        "inventario": R / "p94_experiment_inventory.csv",
        "sota": R / "p94_sota_review.md",
        "registro": R / "p94_hypothesis_registry.csv",
        "recomendacion": R / "p94_recommended_experiment.md",
        "cumplimiento": R / "p94_reproducibility_and_compliance.md",
    }
    for nombre, ruta in rutas.items():
        chequeos[f"existe_{nombre}"] = ruta.exists()

    inv = list(csv.DictReader(open(rutas["inventario"], encoding="utf-8", newline="")))
    chequeos["inventario_24_familias"] = len(inv) == 24
    chequeos["inventario_ancho_8"] = all(len(f) == 8 for f in inv)
    estados = {}
    for f in inv:
        estados[f["class_status"]] = estados.get(f["class_status"], 0) + 1
    chequeos["inventario_estados_21_1_2"] = (
        estados.get("CLOSED") == 21 and estados.get("MARGINAL") == 1
        and estados.get("BLOCKED") == 2)

    reg = list(csv.DictReader(open(rutas["registro"], encoding="utf-8", newline="")))
    chequeos["registro_17_propuestas"] = len(reg) == 17
    chequeos["registro_clases_validas"] = all(f["classification"] in CLASES for f in reg)
    chequeos["registro_ninguna_genuinamente_nueva"] = not any(
        f["classification"] == "GENUINELY_NEW" for f in reg)
    chequeos["registro_0_recomendadas"] = not any(f["recommendation"] == "SI" for f in reg)
    cortas = [f for f in reg if f["p3_shortlisted"] == "true"]
    chequeos["registro_shortlist_3"] = len(cortas) == 3

    rec = rutas["recomendacion"].read_text(encoding="utf-8")
    chequeos["veredicto_no_new_en_md"] = "NO_NEW_MODELING_HYPOTHESIS" in rec
    chequeos["veredicto_training_no_autorizado"] = "TRAINING_NOT_AUTHORIZED" in rec

    # ---------------------------------------------------- ASCII de lo generado
    for nombre in ("inventario", "sota", "registro", "recomendacion"):
        chequeos[f"ascii_{nombre}"] = puro_ascii(rutas[nombre])
    chequeos["ascii_script_00"] = puro_ascii(RAIZ / "experiments" / "p94_00_inventario.py")
    chequeos["ascii_script_01"] = puro_ascii(RAIZ / "experiments" / "p94_01_registro_hipotesis.py")
    chequeos["ascii_script_02"] = puro_ascii(Path(__file__))

    # ---------------------------------------------------- integridad canonica
    canonicos = {
        "data/raw/toxic_train.csv": "3d40e7345b588860",
        "features/oof_v03_final.npz": "559265816ce4f631",
        "experiments/baseline_v03_frozen/submission_v03.csv": "7d7478f271e6edfd",
        "experiments/baseline_v03_frozen/model_v03.joblib": "263ed5bd0fc0de7c",
        "data/processed/holdout_v2_ids.csv": "e5b62e5f22c4c63a",
        "data/processed/holdout2_ids.csv": "f6c6ed958abf728b",
        "artifacts/p93_adjudication_v1/responses/response_C1.csv": "f89f74cdabd9649d",
        "artifacts/p93_adjudication_v1/responses/response_C2.csv": "e0449a811fa8b64b",
        "artifacts/p93_adjudication_v2/responses/response_C1_round2.csv": "91911ab70f6d70cc",
        "artifacts/p93_adjudication_v2/responses/response_C2_round2_correcciones.csv":
            "5da8fc1b33135489",
        "reports/p93r4_final_decisions.csv": "98e4264b0fb98b7a",
        "reports/p93r5_gold_comparison.csv": "cc7ab96b90155980",
        "reports/p93r5_label_correction_candidates.csv": "a884cf91467aa3ea",
        "reports/p93r5_manifest.json": "107355c774b15d8e",
    }
    for ruta, pref in canonicos.items():
        chequeos[f"canonico:{ruta}"] = h(RAIZ / ruta).startswith(pref)

    # ---------------------------------------------------- verificador oficial
    verif = subprocess.run(
        [sys.executable, str(RAIZ / "tools" / "verificar_caracteres.py")],
        capture_output=True, text=True)
    chequeos["verificador_oficial_ok"] = verif.returncode == 0

    # ---------------------------------------------------- git (solo lectura)
    git_st = subprocess.run(["git", "-C", str(RAIZ.parent), "status", "--short"],
                            capture_output=True, text=True).stdout.strip().splitlines()
    n_untracked = sum(1 for l in git_st if l.startswith("??"))
    n_modified = sum(1 for l in git_st if l.startswith(" M") or l.startswith("M "))
    git_diff = subprocess.run(["git", "-C", str(RAIZ.parent), "diff", "--stat"],
                              capture_output=True, text=True).stdout.strip().splitlines()

    # ---------------------------------------------------- hashes
    artefactos = [
        rutas["inventario"], rutas["sota"], rutas["registro"], rutas["recomendacion"],
        rutas["cumplimiento"],
        RAIZ / "experiments" / "p94_00_inventario.py",
        RAIZ / "experiments" / "p94_01_registro_hipotesis.py",
        RAIZ / "experiments" / "p94_02_verificacion.py",
    ]

    print("P94-02 verificacion de la fase: ejecutada")
    for clave, valor in chequeos.items():
        if not valor:
            fallos.append(clave)
    print(f"  chequeos: {sum(chequeos.values())}/{len(chequeos)} | "
          f"fallos: {fallos if fallos else 'ninguno'}")
    print(f"  verificador oficial: {'OK' if verif.returncode == 0 else 'FALLO'}")
    print(f"  git: {n_untracked} sin rastrear, {n_modified} modificados (solo lectura)")
    print()
    print("  === hashes completos de artefactos P94 ===")
    for ruta in artefactos:
        if ruta.exists():
            print(f"  {h(ruta)}  {ruta.relative_to(RAIZ)}")
        else:
            print(f"  (pendiente: {ruta.relative_to(RAIZ)})")
    print()
    print("  === git diff --stat (ultimas 3 lineas) ===")
    for l in git_diff[-3:]:
        print("  " + l)

    if fallos:
        sys.exit(1)


if __name__ == "__main__":
    main()
