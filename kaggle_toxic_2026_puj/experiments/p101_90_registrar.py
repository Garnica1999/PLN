#!/usr/bin/env python
"""P101-90: registra la fila P101-00 en experiments.csv (idempotente).

Registra el cierre de P101 (auditoria de supervision P0/P1; rama humana
bloqueada por ausencia de revisores; P6 no ejecutado).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA = RAIZ / "experiments.csv"
COLUMNAS = ["experiment_id", "competition", "question", "hypothesis",
            "preprocessing", "feature_groups", "representation", "model",
            "hyperparameters", "validation", "f1", "macro_f1", "weighted_f1",
            "precision", "recall", "training_time", "dimension", "result",
            "interpretation", "decision", "date", "script", "notes"]


def sha16(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def jc(valor) -> str:
    return json.dumps(valor, ensure_ascii=True, separators=(",", ":"))


def leer():
    with open(RUTA, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def anexar(nuevas) -> None:
    with open(RUTA, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS, lineterminator="\n")
        for f in nuevas:
            w.writerow(f)


def main() -> int:
    filas = leer()
    ids = {f["experiment_id"] for f in filas}
    nuevas = []
    if "P101-00" not in ids:
        nuevas.append({
            "experiment_id": "P101-00",
            "competition": "KAGGLE_TOXIC_2026_PUJ",
            "question": "Existe una fuente de senal supervisada permitida "
                        "(errores de anotacion verificables) que justifique "
                        "objetivos alternativos de entrenamiento?",
            "hypothesis": "P101-H1 (calidad de objetivos): solo ejecutable "
                          "con evidencia humana independiente y gobernanza "
                          "valida",
            "preprocessing": "n/a (auditoria documental)",
            "feature_groups": "n/a", "representation": "n/a",
            "model": "n/a (sin entrenamiento)", "hyperparameters": "n/a",
            "validation": "n/a", "f1": "", "macro_f1": "", "weighted_f1": "",
            "precision": "", "recall": "", "training_time": "",
            "dimension": "",
            "result": jc({
                "estado": "CLOSED_BLOCKED_NO_REVIEWERS",
                "p0": {"tabla_filas": 1327, "elegibles": 971, "dev2": 1236,
                       "revisados": 265, "reservados_v2_1": 0},
                "p1": "AMBIGUOUS",
                "p6": "NOT_EXECUTED",
                "revisores": "no disponibles (decision del propietario)"}),
            "interpretation": "",
            "decision": "CERRAR la rama humana por bloqueo; reabrible solo "
                        "con revisores reales; labels originales intactas; "
                        "P6 no ejecutado.",
            "date": "2026-10-14",
            "script": "experiments/p101_90_registrar.py",
            "notes": (
                f"P0 {sha16(RAIZ / 'reports' / 'p101_provenance_audit.md')} | "
                f"P1 {sha16(RAIZ / 'reports' / 'p101_governance_note.md')} | "
                f"final {sha16(RAIZ / 'reports' / 'p101_final_report.md')} | "
                f"compliance "
                f"{sha16(RAIZ / 'reports' / 'p101_compliance.md')}"),
        })
    if nuevas:
        anexar(nuevas)
    print(f"P101-90: {len(nuevas)} fila(s) anadida(s); "
          f"total={len(filas) + len(nuevas)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
