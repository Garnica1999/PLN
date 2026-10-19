#!/usr/bin/env python
"""
P96-04: verificacion de cierre de la fase forense P96.

Solo lectura. Comprueba: entregables existentes y bien formados; matriz de
integridad (459 filas); hallazgos (17, sin P0/P1); candidatos de reproduccion;
matriz de reproducibilidad (sin NUMERIC_MISMATCH); manifest (canonicos
intactos); recomputo exacto de V03; evidencia de folds; ASCII de lo generado;
hashes canonicos; verificador oficial de caracteres y estado de Git.

Uso:
    /home/carlos/venv/bin/python experiments/p96_04_verificacion.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.evaluation import metricas_multietiqueta  # noqa: E402

R = RAIZ / "reports"
CANONICO = 0.7772114525165721


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
    """Ejecuta la verificacion de cierre."""
    chequeos: dict[str, bool] = {}
    entregables = [
        "p96_forensic_audit.md", "p96_experiment_integrity_matrix.csv",
        "p96_code_and_metric_findings.csv", "p96_reproducibility_matrix.csv",
        "p96_validation_and_leakage_audit.md",
        "p96_historical_conclusions_review.md",
        "p96_reproduction_candidates.csv", "p96_manifest.json",
    ]
    for nombre in entregables:
        chequeos[f"existe_{nombre}"] = (R / nombre).exists()
    for nombre in ("p96_folds_evidence.json", "p96_static_scans.txt"):
        chequeos[f"evidencia_{nombre}"] = (R / nombre).exists()

    integridad = list(csv.DictReader(open(R / "p96_experiment_integrity_matrix.csv",
                                          encoding="utf-8")))
    chequeos["integridad_459"] = len(integridad) == 459
    chequeos["integridad_ids_unicos"] = len({f["experiment_id"] for f in integridad}) == 459

    hallazgos = list(csv.DictReader(open(R / "p96_code_and_metric_findings.csv",
                                         encoding="utf-8")))
    chequeos["hallazgos_17"] = len(hallazgos) == 17
    chequeos["hallazgos_sin_p0p1"] = all(f["severity"] not in ("P0", "P1")
                                         for f in hallazgos)
    ids_h = {f["finding_id"] for f in hallazgos}
    chequeos["hallazgos_clave"] = {"F96-05", "F96-06", "F96-13", "F96-17"} <= ids_h

    repro = list(csv.DictReader(open(R / "p96_reproduction_candidates.csv",
                                     encoding="utf-8")))
    chequeos["repro_2"] = len(repro) == 2
    chequeos["repro_condicionales"] = all("condicional" in f["experiment_id"]
                                          for f in repro)

    matriz = list(csv.DictReader(open(R / "p96_reproducibility_matrix.csv",
                                      encoding="utf-8")))
    estados = Counter(f["status"] for f in matriz)
    chequeos["repro_sin_mismatch"] = estados.get("NUMERIC_MISMATCH", 0) == 0
    chequeos["repro_curadas_20"] = estados.get("COMPARED_WITH_DOCUMENTED_EXACTLY",
                                               0) == 20
    chequeos["repro_200"] = len(matriz) == 200

    manifest = json.load(open(R / "p96_manifest.json", encoding="utf-8"))
    chequeos["manifest_canonicos"] = all(manifest["canonicos_intactos"].values())
    chequeos["manifest_bootstrap_4"] = len(manifest.get("bootstrap", {})) == 4
    chequeos["manifest_v03"] = manifest.get("v03_reproducido_exacto") is True

    folds = json.load(open(R / "p96_folds_evidence.json", encoding="utf-8"))
    chequeos["folds_particion_identica"] = bool(
        folds["pliegues"]["particion_identica_tras_traducir_numeracion"])
    chequeos["folds_sin_fugas"] = folds["pliegues"]["pliegues_con_grupos_compartidos"] == []
    chequeos["folds_grupos_exactos"] = bool(
        folds["grupos"]["iguales_etiqueta_a_etiqueta"])

    # recomputo de V03 (chequeo funcional rapido)
    z = np.load(RAIZ / "features" / "oof_v03_final.npz", allow_pickle=True)
    valor = float(metricas_multietiqueta(
        z["y_true"].astype(int),
        (z["proba"] >= z["umbrales"]).astype(int))["f1_media_columnas"])
    chequeos["v03_exacto"] = abs(valor - CANONICO) == 0.0

    for nombre in entregables + ["p96_folds_evidence.json", "p96_static_scans.txt"]:
        chequeos[f"ascii_{nombre}"] = puro_ascii(R / nombre)
    for nombre in ("p96_00_registro_audit.py", "p96_01_metric_recompute.py",
                   "p96_02_folds_leakage.py", "p96_03_informes.py",
                   "p96_04_verificacion.py", "p96_05_integrar_notebook.py"):
        ruta = RAIZ / "experiments" / nombre
        chequeos[f"ascii_{nombre}"] = puro_ascii(ruta) if ruta.exists() else False

    canonicos = {
        "data/raw/toxic_train.csv": "3d40e7345b588860",
        "features/oof_v03_final.npz": "559265816ce4f631",
        "experiments/baseline_v03_frozen/submission_v03.csv": "7d7478f271e6edfd",
        "experiments/baseline_v03_frozen/model_v03.joblib": "263ed5bd0fc0de7c",
        "data/processed/holdout_v2_ids.csv": "e5b62e5f22c4c63a",
        "artifacts/p93_adjudication_v1/responses/response_C1.csv": "f89f74cdabd9649d",
        "artifacts/p93_adjudication_v1/responses/response_C2.csv": "e0449a811fa8b64b",
        "artifacts/p93_adjudication_v2/responses/response_C1_round2.csv": "91911ab70f6d70cc",
        "artifacts/p93_adjudication_v2/responses/response_C2_round2_correcciones.csv":
            "5da8fc1b33135489",
        "reports/p93r5_label_correction_candidates.csv": "a884cf91467aa3ea",
        "reports/p94_hypothesis_registry.csv": "17917ce2a1a132d9",
    }
    for ruta, pref in canonicos.items():
        chequeos[f"canonico:{ruta}"] = sha256(RAIZ / ruta).startswith(pref)

    verif = subprocess.run(
        [sys.executable, str(RAIZ / "tools" / "verificar_caracteres.py")],
        capture_output=True, text=True)
    chequeos["verificador_oficial_ok"] = verif.returncode == 0

    git_st = subprocess.run(["git", "-C", str(RAIZ.parent), "status", "--short"],
                            capture_output=True, text=True).stdout.strip().splitlines()
    n_untracked = sum(1 for l in git_st if l.startswith("??"))
    n_modified = sum(1 for l in git_st if l[:2] in (" M", "M ", "MM"))

    fallos = [k for k, v in chequeos.items() if not v]
    print("P96-04 verificacion forense: ejecutada")
    print(f"  chequeos: {sum(chequeos.values())}/{len(chequeos)} | "
          f"fallos: {fallos if fallos else 'ninguno'}")
    print(f"  verificador oficial: {'OK' if verif.returncode == 0 else 'FALLO'}")
    print(f"  git: {n_untracked} sin rastrear, {n_modified} modificados (solo lectura)")
    print()
    print("  === hashes completos de artefactos P96 ===")
    for nombre in entregables + ["p96_folds_evidence.json", "p96_static_scans.txt"]:
        print(f"  {sha256(R / nombre)}  reports/{nombre}")
    for nombre in ("p96_00_registro_audit.py", "p96_01_metric_recompute.py",
                   "p96_02_folds_leakage.py", "p96_03_informes.py",
                   "p96_04_verificacion.py", "p96_05_integrar_notebook.py"):
        ruta = RAIZ / "experiments" / nombre
        if ruta.exists():
            print(f"  {sha256(ruta)}  experiments/{nombre}")
    nb = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
    print(f"  {sha256(nb)}  notebooks/10_final_pipeline.ipynb")
    if fallos:
        sys.exit(1)


if __name__ == "__main__":
    main()
