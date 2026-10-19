#!/usr/bin/env python
"""
P97-99: verificacion de cierre de la fase P97.

Solo lectura. Comprueba entregables, veredictos pre-registrados, registro
(471 filas, 12 unidades P97), matrices del registro de metodos, ASCII de lo
generado, fuentes canonicas intactas, verificador oficial de caracteres y
estado de Git. Imprime hashes completos.

Uso:
    /home/carlos/venv/bin/python experiments/p97_99_verificacion.py
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

TEXTO_ESPERADO = [
    "p97_independent_research.md", "p97_method_registry.csv",
    "p97_novelty_matrix.csv",
    "p97_01_reglas_por_segmento.json", "p97_01_reglas_por_segmento.md",
    "p97_02_group_dro.json", "p97_02_group_dro.md",
    "p97_02_group_dro_escala_invalida.json", "p97_02_group_dro_escala_invalida.md",
    "p97_03_nnpu_tr.json", "p97_03_nnpu_tr.md",
    "p97_03_nnpu_tr_l2_escala_invalida.json", "p97_03_nnpu_tr_l2_escala_invalida.md",
    "p97_03_nnpu_tr_pn_idx_invalido.json", "p97_03_nnpu_tr_pn_idx_invalido.md",
    "p97_04_replicacion.json", "p97_04_replicacion.md",
]
NPZ_ESPERADOS = [
    "features/p97_01_pred_segmentos.npz", "features/p97_02_oof_dro.npz",
    "features/p97_02_oof_dro_escala_invalida.npz",
    "features/p97_03_oof_nnpu.npz",
    "features/p97_03_oof_nnpu_l2_escala_invalida.npz",
    "features/p97_03_oof_nnpu_pn_idx_invalido.npz",
    "features/p97_04_oof_replica.npz",
]
FIGURAS = [
    "reports/figuras/p97_01_segmentos.png",
    "reports/figuras/p97_02_group_dro.png",
    "reports/figuras/p97_02_group_dro_escala_invalida.png",
    "reports/figuras/p97_03_nnpu_tr.png",
    "reports/figuras/p97_03_nnpu_tr_l2_escala_invalida.png",
    "reports/figuras/p97_03_nnpu_tr_pn_idx_invalido.png",
    "reports/figuras/p97_04_replicacion.png",
]
SCRIPTS = ["p97_00_generador_registro.py", "p97_01_reglas_por_segmento.py",
           "p97_02_group_dro.py", "p97_03_nnpu_tr.py", "p97_04_replicacion.py",
           "p97_90_registrar.py", "p97_95_integrar_notebook.py",
           "p97_99_verificacion.py"]


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
    """Ejecuta la verificacion de cierre de P97."""
    chequeos: dict[str, bool] = {}
    for nombre in TEXTO_ESPERADO:
        chequeos[f"existe_{nombre}"] = (R / nombre).exists()
    for rel in NPZ_ESPERADOS + FIGURAS:
        chequeos[f"existe_{rel}"] = (RAIZ / rel).exists()
    for nombre in SCRIPTS:
        chequeos[f"existe_{nombre}"] = (RAIZ / "experiments" / nombre).exists()

    reg = list(csv.DictReader(open(R / "p97_method_registry.csv", encoding="utf-8")))
    mat = list(csv.DictReader(open(R / "p97_novelty_matrix.csv", encoding="utf-8")))
    chequeos["registro_25_mecanismos"] = len(reg) == 25
    chequeos["novelty_8_candidatos"] = len(mat) == 8
    chequeos["novelty_3_seleccionados"] = sum(
        1 for f in mat if f["decision"].startswith("SELECCIONADA")) == 3

    v01 = json.load(open(R / "p97_01_reglas_por_segmento.json", encoding="utf-8"))
    v02 = json.load(open(R / "p97_02_group_dro.json", encoding="utf-8"))
    v03 = json.load(open(R / "p97_03_nnpu_tr.json", encoding="utf-8"))
    v04 = json.load(open(R / "p97_04_replicacion.json", encoding="utf-8"))
    chequeos["veredicto_01_falsificada"] = (
        v01["veredicto_pre_registrado"] == "FALSIFICADA")
    chequeos["veredicto_02_falsificada"] = (
        v02["veredicto_pre_registrado"] == "FALSIFICADA")
    chequeos["veredicto_03_senal"] = v03["veredicto_pre_registrado"] == "SENAL"
    chequeos["veredicto_04_falsificada"] = (
        v04["veredicto_pre_registrado"] == "FALSIFICADA")
    chequeos["control_A0_exacto"] = bool(v02["control_A0"]["reproduce_v03_exacto"])
    chequeos["nnpu_delta_positivo"] = v03["comparacion"]["delta_pooled"] > 0
    chequeos["nnpu_5_de_5"] = v03["comparacion"]["pliegues_mejorados"] == 5
    chequeos["replica_delta_negativo"] = v04["comparacion"]["delta_pooled"] < 0

    filas = list(csv.DictReader(open(RAIZ / "experiments.csv", encoding="utf-8-sig")))
    chequeos["registro_471"] = len(filas) == 471
    chequeos["registro_12_p97"] = sum(
        1 for f in filas if f["experiment_id"].startswith("P97")) == 12
    lineas = (RAIZ / "experiments.csv").read_text(encoding="utf-8").splitlines()
    nuevas = [l for l in lineas if l.startswith("P97-")]
    chequeos["filas_nuevas_ascii"] = len(nuevas) == 12 and all(
        all(ord(c) <= 127 for c in l) for l in nuevas)

    for nombre in TEXTO_ESPERADO:
        chequeos[f"ascii_{nombre}"] = puro_ascii(R / nombre)
    for nombre in SCRIPTS:
        chequeos[f"ascii_{nombre}"] = puro_ascii(RAIZ / "experiments" / nombre)

    canonicos = {
        "data/raw/toxic_train.csv": "3d40e7345b588860",
        "features/oof_v03_final.npz": "559265816ce4f631",
        "experiments/baseline_v03_frozen/submission_v03.csv": "7d7478f271e6edfd",
        "experiments/baseline_v03_frozen/model_v03.joblib": "263ed5bd0fc0de7c",
        "data/processed/holdout2_ids.csv": "f6c6ed958abf728b",
        "reports/p94_hypothesis_registry.csv": "17917ce2a1a132d9",
    }
    for ruta, pref in canonicos.items():
        chequeos[f"canonico:{ruta}"] = sha256(RAIZ / ruta).startswith(pref)

    verif = subprocess.run(
        [sys.executable, str(RAIZ / "tools" / "verificar_caracteres.py")],
        capture_output=True, text=True)
    chequeos["verificador_oficial_ok"] = verif.returncode == 0

    git_st = subprocess.run(["git", "-C", str(RAIZ), "status", "--short"],
                            capture_output=True, text=True).stdout.strip().splitlines()
    n_untracked = sum(1 for l in git_st if l.startswith("??"))
    n_modified = sum(1 for l in git_st if l[:2] in (" M", "M ", "MM"))

    fallos = [k for k, v in chequeos.items() if not v]
    print("P97-99 verificacion de cierre: ejecutada")
    print(f"  chequeos: {sum(chequeos.values())}/{len(chequeos)} | "
          f"fallos: {fallos if fallos else 'ninguno'}")
    print(f"  verificador oficial: {'OK' if verif.returncode == 0 else 'FALLO'}")
    print(f"  git: {n_untracked} sin rastrear, {n_modified} modificados "
          f"(solo lectura)")
    print()
    print("  === hashes completos de artefactos P97 ===")
    for rel in (["reports/" + n for n in TEXTO_ESPERADO] + NPZ_ESPERADOS
                + ["experiments/" + n for n in SCRIPTS]
                + ["notebooks/10_final_pipeline.ipynb", "experiments.csv"]):
        print(f"  {sha256(RAIZ / rel)}  {rel}")
    if fallos:
        sys.exit(1)


if __name__ == "__main__":
    main()
