#!/usr/bin/env python
"""
P95-01: verificacion de la fase P95 (cierre de frontera y plan de reapertura).

Comprueba (sin entrenar, sin predicciones, sin tocar recursos reservados):
- existencias y estructura de los entregables P95;
- coherencia de las tablas de gobernanza y de evidencia nueva;
- presencia del bloque de decision y de las cinco limitaciones en los informes;
- ASCII byte a byte de lo generado;
- integridad canonica (V03, OOF, etiquetas, respuestas humanas, P93-R5, P94);
- hashes completos y estado de Git (solo lectura).

Uso:
    /home/carlos/venv/bin/python experiments/p95_01_verificacion.py
"""

from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
R = RAIZ / "reports"

CLASES_GOV = {"currently_allowed", "allowed_with_explicit_approval", "ambiguous",
              "prohibited"}


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

    rutas = {
        "assessment": R / "p95_reopening_assessment.md",
        "gov": R / "p95_governance_options.csv",
        "evi": R / "p95_new_evidence_options.csv",
        "gate": R / "p95_decision_gate.md",
    }
    for nombre, ruta in rutas.items():
        chequeos[f"existe_{nombre}"] = ruta.exists()

    gov = list(csv.DictReader(open(rutas["gov"], encoding="utf-8", newline="")))
    chequeos["gov_7_vias"] = len(gov) == 7
    chequeos["gov_ancho_13"] = all(len(f) == 13 for f in gov)
    chequeos["gov_clases_validas"] = all(f["class"] in CLASES_GOV for f in gov)
    por_clase: dict[str, int] = {}
    for f in gov:
        por_clase[f["class"]] = por_clase.get(f["class"], 0) + 1
    chequeos["gov_reparto_clases"] = (
        por_clase.get("currently_allowed") == 1
        and por_clase.get("allowed_with_explicit_approval") == 4
        and por_clase.get("prohibited") == 1
        and por_clase.get("ambiguous") == 1)

    evi = list(csv.DictReader(open(rutas["evi"], encoding="utf-8", newline="")))
    chequeos["evi_4_filas"] = len(evi) == 4
    chequeos["evi_ancho_10"] = all(len(f) == 10 for f in evi)
    chequeos["evi_sintesis_no_actionable"] = any(
        "NO_ACTIONABLE_NEW_EVIDENCE_SOURCE" in f["expected_value_for_score"]
        for f in evi)

    ass = rutas["assessment"].read_text(encoding="utf-8")
    chequeos["assessment_5_limitaciones"] = all(
        s in ass for s in ("### A. Representacion insuficiente",
                           "### B. Objetivo de entrenamiento o decision",
                           "### C. Variabilidad y transferencia de la validacion",
                           "### D. Calidad y definicion de las etiquetas",
                           "### E. Restricciones academicas y de la competicion"))
    chequeos["assessment_14_candidatas"] = "REQUIRES_MORE_HUMAN_EVIDENCE" in ass

    gate = rutas["gate"].read_text(encoding="utf-8")
    chequeos["gate_bloque_decision"] = all(
        s in gate for s in ("MODEL_EXPLORATION_PAUSED", "V03 = FROZEN",
                            "NO_TRAINING_AUTHORIZED", "NO_SUBMISSIONS_AUTHORIZED",
                            "REOPENING_CRITERIA_DOCUMENTED"))
    chequeos["gate_sin_via_prioritaria"] = (
        "Via prioritaria seleccionada: NINGUNA" in gate)

    for nombre in ("assessment", "gov", "evi", "gate"):
        chequeos[f"ascii_{nombre}"] = puro_ascii(rutas[nombre])
    chequeos["ascii_script_00"] = puro_ascii(RAIZ / "experiments" / "p95_00_opciones.py")
    chequeos["ascii_script_01"] = puro_ascii(Path(__file__))

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
        "reports/p94_experiment_inventory.csv": "11f376e1d4c1545c",
        "reports/p94_sota_review.md": "245cf119df408b4b",
        "reports/p94_hypothesis_registry.csv": "17917ce2a1a132d9",
        "reports/p94_recommended_experiment.md": "7a09af6710b863af",
        "reports/p94_reproducibility_and_compliance.md": "588c7664e8b7ddf4",
    }
    for ruta, pref in canonicos.items():
        chequeos[f"canonico:{ruta}"] = h(RAIZ / ruta).startswith(pref)

    verif = subprocess.run(
        [sys.executable, str(RAIZ / "tools" / "verificar_caracteres.py")],
        capture_output=True, text=True)
    chequeos["verificador_oficial_ok"] = verif.returncode == 0

    git_st = subprocess.run(["git", "-C", str(RAIZ.parent), "status", "--short"],
                            capture_output=True, text=True).stdout.strip().splitlines()
    n_untracked = sum(1 for l in git_st if l.startswith("??"))
    n_modified = sum(1 for l in git_st if l.startswith(" M") or l.startswith("M "))

    print("P95-01 verificacion de la fase: ejecutada")
    for clave, valor in chequeos.items():
        if not valor:
            fallos.append(clave)
    print(f"  chequeos: {sum(chequeos.values())}/{len(chequeos)} | "
          f"fallos: {fallos if fallos else 'ninguno'}")
    print(f"  verificador oficial: {'OK' if verif.returncode == 0 else 'FALLO'}")
    print(f"  git: {n_untracked} sin rastrear, {n_modified} modificados (solo lectura)")
    print()
    print("  === hashes completos de artefactos P95 ===")
    for ruta in (rutas["assessment"], rutas["gov"], rutas["evi"], rutas["gate"],
                 RAIZ / "experiments" / "p95_00_opciones.py",
                 RAIZ / "experiments" / "p95_01_verificacion.py"):
        print(f"  {h(ruta)}  {ruta.relative_to(RAIZ)}")

    if fallos:
        sys.exit(1)


if __name__ == "__main__":
    main()
