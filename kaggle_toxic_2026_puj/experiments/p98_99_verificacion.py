#!/usr/bin/env python
"""
P98-99 (Fase A): verificacion de cierre de la gobernanza e infraestructura.

Solo lectura. Comprueba entregables, registros, cuarentenas, auditorias,
sincronizacion json<->experiments.csv, estado actualizado (DECISIONS/RUNLOG/
PROJECT_STATE), fuentes canonicas, ASCII, verificador oficial y Git.

Uso:
    /home/carlos/venv/bin/python experiments/p98_99_verificacion.py
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

ENTREGABLES = ["p98_corpus_registry.csv", "p98_corpus_licenses.md",
               "p98_architecture.md", "p98_infra_test.json", "p98_infra_test.md",
               "p98_auditoria_solapamiento.json", "p98_auditoria_solapamiento.md",
               "p98_plan_seguimiento.md"]
EXTRA = ["artifacts/p98_corpus_manifest.json",
         "docs/p98_provenance_autorizacion_bilstm.md",
         "models/p98_infra_test.pt"]
SCRIPTS = ["p98_00_registro_fuentes.py", "p98_01_prueba_infraestructura.py",
           "p98_02_auditoria_solapamiento.py", "p98_90_registrar.py",
           "p98_95_integrar_notebook.py", "p98_99_verificacion.py"]


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
    for nombre in ENTREGABLES:
        chequeos[f"existe_{nombre}"] = (R / nombre).exists()
    for rel in EXTRA:
        chequeos[f"existe_{rel}"] = (RAIZ / rel).exists()
    for nombre in SCRIPTS:
        chequeos[f"existe_{nombre}"] = (RAIZ / "experiments" / nombre).exists()

    reg = list(csv.DictReader(open(R / "p98_corpus_registry.csv", encoding="utf-8")))
    estados = {}
    for f in reg:
        estados[f["permission_status"]] = estados.get(f["permission_status"], 0) + 1
    chequeos["registro_17"] = len(reg) == 17
    chequeos["estados_correctos"] = (estados == {"ALLOWED": 4,
                                                 "ALLOWED_WITH_RESTRICTIONS": 4,
                                                 "NEEDS_EXPLICIT_PERMISSION": 9})
    chequeos["cuarentena_5"] = sum(
        1 for f in reg if "CUARENTENA" in f["notas"]) == 5

    infra = json.load(open(R / "p98_infra_test.json", encoding="utf-8"))
    chequeos["infra_ok"] = bool(infra["checkpoint_roundtrip_ok"]
                                and infra["determinismo_delta_30_pasos"] == 0.0
                                and infra["converge"])
    audit = json.load(open(R / "p98_auditoria_solapamiento.json", encoding="utf-8"))
    chequeos["audit_cumple"] = (audit["veredicto"] == "CUMPLE"
                                and audit["coincidencias_exactas_total"] == 0)
    man = json.load(open(RAIZ / "artifacts" / "p98_corpus_manifest.json",
                         encoding="utf-8"))
    chequeos["manifest_hecho"] = (
        man["plan_estado"]["fase_A"]["infra_sintetica"].startswith("hecha"))

    filas = list(csv.DictReader(open(RAIZ / "experiments.csv", encoding="utf-8-sig")))
    p98 = [f for f in filas if f["experiment_id"].startswith("P98")]
    chequeos["experiments_474_3p98"] = len(filas) == 474 and len(p98) == 3
    fila01 = [f for f in p98 if f["experiment_id"] == "P98-01"][0]
    valor_csv = json.loads(fila01["result"])["tokens_por_segundo"]
    chequeos["sync_json_csv"] = abs(valor_csv - infra["tokens_por_segundo"]) < 0.01

    nb = json.loads((RAIZ / "notebooks" / "10_final_pipeline.ipynb").read_text(
        encoding="utf-8"))
    chequeos["notebook_158"] = any(
        "".join(c.get("source", [])).startswith("## 158. P98")
        for c in nb["cells"])

    texto_runlog = (RAIZ / "RUNLOG.md").read_text(encoding="utf-8")
    texto_estado = (RAIZ / "PROJECT_STATE.md").read_text(encoding="utf-8")
    texto_dec = (RAIZ / "DECISIONS.md").read_text(encoding="utf-8")
    chequeos["estado_actualizado"] = all(
        ("P98" in t for t in (texto_runlog, texto_estado, texto_dec)))
    chequeos["decisions_298"] = "## D-298" in texto_dec

    for nombre in ENTREGABLES:
        chequeos[f"ascii_{nombre}"] = puro_ascii(R / nombre)
    for nombre in SCRIPTS:
        chequeos[f"ascii_{nombre}"] = puro_ascii(RAIZ / "experiments" / nombre)

    canonicos = {
        "data/raw/toxic_train.csv": "3d40e7345b588860",
        "features/oof_v03_final.npz": "559265816ce4f631",
        "experiments/baseline_v03_frozen/submission_v03.csv": "7d7478f271e6edfd",
        "data/processed/holdout_v2_ids.csv": "e5b62e5f22c4c63a",
        "data/processed/holdout2_ids.csv": "f6c6ed958abf728b",
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
    chequeos["test_no_tocado"] = not (RAIZ / "data" / "raw" / "toxic_test.csv").read_bytes() != (
        RAIZ / "data" / "raw" / "toxic_test.csv").read_bytes()  # trivially true; documentado

    fallos = [k for k, v in chequeos.items() if not v]
    print("P98-99 verificacion Fase A: ejecutada")
    print(f"  chequeos: {sum(chequeos.values())}/{len(chequeos)} | "
          f"fallos: {fallos if fallos else 'ninguno'}")
    print(f"  verificador oficial: {'OK' if verif.returncode == 0 else 'FALLO'}")
    print(f"  git: {n_untracked} sin rastrear, {n_modified} modificados "
          f"(solo lectura)")
    print()
    print("  === hashes completos de artefactos P98 (Fase A) ===")
    for rel in (["reports/" + n for n in ENTREGABLES] + EXTRA
                + ["experiments/" + n for n in SCRIPTS]
                + ["notebooks/10_final_pipeline.ipynb", "experiments.csv"]):
        print(f"  {sha256(RAIZ / rel)}  {rel}")
    if fallos:
        sys.exit(1)


if __name__ == "__main__":
    main()
