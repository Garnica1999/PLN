#!/usr/bin/env python
"""P108-90: Registra o actualiza la fila P108-01 en experiments.csv (idempotente)."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA = RAIZ / "experiments.csv"
COLUMNAS = [
    "experiment_id", "competition", "question", "hypothesis",
    "preprocessing", "feature_groups", "representation", "model",
    "hyperparameters", "validation", "f1", "macro_f1", "weighted_f1",
    "precision", "recall", "training_time", "dimension", "result",
    "interpretation", "decision", "date", "script", "notes"
]


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


def escribir(filas) -> None:
    with open(RUTA, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS, lineterminator="\n")
        w.writeheader()
        for f in filas:
            w.writerow(f)


def main() -> None:
    f_json = RAIZ / "reports" / "p108_01_challenger_v2_triple_asc.json"
    if not f_json.exists():
        print("No existe el reporte json de P108-01")
        return

    data = json.loads(f_json.read_text(encoding="utf-8"))
    c_can = data["metricas_dev_canonica"]
    c_47 = data["metricas_dev_part47"]
    c_te = data["comparacion_test_vs_v03"]

    filas = leer()

    nueva = {
        "experiment_id": "P108-01",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Supera el challenger V2_triple + Regla Ascendente a V03 en ambas particiones de CV y produce submission valida?",
        "hypothesis": "La union triple TF-IDF (word(1,2) + char_wb(3,5) + char(1,6)) enriquece el espacio n-grama y la regla ontologica TR=1=>HS=1 elimina incoherencias estructurales sin riesgo de sobreajuste",
        "preprocessing": "texto oficial; union triple ajustada fold-local en CV y sobre train completo para test",
        "feature_groups": "word(1,2) + char_wb(3,5) + char(1,6) con min_df=2 y sublinear_tf=True",
        "representation": "union TF-IDF triple (86.275 dimensiones sobre train completo)",
        "model": "LogisticRegression C=2.0 balanced liblinear x3 + Regla Ascendente TR=1=>HS=1",
        "hyperparameters": jc({"umbrales": "V03 congelados 0.44/0.51/0.46", "regla": "TR=1 => HS=1"}),
        "validation": "folds canonicos + particion 47",
        "f1": f"{c_can['v2_triple_asc_ccv']}",
        "macro_f1": f"{c_can['v2_triple_asc_ccv']}",
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": f"{data['duracion_segundos']} s",
        "dimension": "86275",
        "result": jc({
            "control_reproduce": data["control_canonica"]["ok"],
            "canonica": {
                "ccv": c_can["v2_triple_asc_ccv"],
                "delta": c_can["delta"],
                "pliegues": f"{c_can['pliegues_ganados']}/5",
                "celdas": c_can["celdas_cambiadas"],
                "ic95": c_can["bootstrap_ic95"],
                "pct_mejora": c_can["bootstrap_pct_mejora"],
            },
            "particion_47": {
                "ccv": c_47["v2_triple_asc_ccv"],
                "delta": c_47["delta"],
                "pliegues": f"{c_47['pliegues_ganados']}/5",
                "celdas": c_47["celdas_cambiadas"],
                "ic95": c_47["bootstrap_ic95"],
                "pct_mejora": c_47["bootstrap_pct_mejora"],
            },
            "test_vs_v03": {
                "celdas_distintas": c_te["total_celdas_distintas"],
                "pct_distintas": c_te["pct_celdas_distintas"],
                "violaciones_corregidas": c_te["violaciones_corregidas_test"],
                "sha256": data["submission_sha256"][:16],
            },
        }),
        "interpretation": f"Challenger formalmente validado. Mejora consistente en ambas particiones: delta canonica {c_can['delta']:+.6f} ({c_can['pliegues_ganados']}/5 pliegues; IC95 {c_can['bootstrap_ic95']}; {c_can['bootstrap_pct_mejora']*100:.1f}% mejora); delta 47 {c_47['delta']:+.6f} ({c_47['pliegues_ganados']}/5 pliegues; IC95 {c_47['bootstrap_ic95']}; {c_47['bootstrap_pct_mejora']*100:.1f}% mejora). En test modifica 66 celdas ({c_te['pct_celdas_distintas']}%) y corrige {c_te['violaciones_corregidas_test']} violaciones ontologicas.",
        "decision": "ADOPCION RECOMENDADA COMO CHALLENGER: V03 se mantiene como ancla en Slot 1 (LB 0.78306) y V2_triple_asc se recomienda para Slot 2 de Kaggle.",
        "date": "2026-10-14",
        "script": "experiments/p108_01_challenger_v2_triple_asc.py",
        "notes": f"sub {data['submission_sha256'][:16]} | json {sha16(f_json)}",
    }

    idx = next((i for i, f in enumerate(filas) if f["experiment_id"] == "P108-01"), None)
    if idx is not None:
        filas[idx] = nueva
        print("P108-01 actualizado in-place en experiments.csv")
    else:
        filas.append(nueva)
        print("P108-01 anexado exitosamente a experiments.csv")

    escribir(filas)


if __name__ == "__main__":
    main()
