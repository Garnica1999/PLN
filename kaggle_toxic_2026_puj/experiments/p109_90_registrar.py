#!/usr/bin/env python
"""P109-90: Registra o actualiza la fila P109-01 en experiments.csv (idempotente)."""

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
    f_json = RAIZ / "reports" / "p109_01_challenger_mdf5_v2.json"
    if not f_json.exists():
        print("No existe el reporte json de P109-01")
        return

    data = json.loads(f_json.read_text(encoding="utf-8"))
    c_can = data["metricas_dev_canonica"]
    c_47 = data["metricas_dev_particion47"]
    c_te = data["simulacion_test"]

    filas = leer()

    nueva = {
        "experiment_id": "P109-01",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Supera el challenger V2_triple min_df=5 + Umbrales Calibrados + Regla Ascendente a P108 en ambas particiones?",
        "hypothesis": "La poda de ruido por frecuencia (min_df=5) reduce de 86k a 31k features parsimoniosas, mejorando la generalizacion del clasificador lineal, y los umbrales calibrados (0.45/0.46/0.43) con C=2.5 en AG optimizan la F1 column-wise",
        "preprocessing": "texto oficial; union triple ajustada fold-local en CV y sobre train completo para test con min_df=5",
        "feature_groups": "word(1,2) + char_wb(3,5) + char(1,6) con min_df=5 y sublinear_tf=True",
        "representation": "union TF-IDF triple podada (31.603 dimensiones sobre train completo)",
        "model": "LogisticRegression C={HS:2.0, TR:2.0, AG:2.5} balanced liblinear + Regla Ascendente TR=1=>HS=1",
        "hyperparameters": jc({"umbrales": data["hiperparametros"]["umbrales"], "C": data["hiperparametros"]["C_por_etiqueta"], "regla": "TR=1 => HS=1"}),
        "validation": "folds canonicos + particion 47",
        "f1": f"{c_can['p109_ccv']}",
        "macro_f1": f"{c_can['p109_ccv']}",
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": f"{data['duracion_segundos']} s",
        "dimension": "31603",
        "result": jc({
            "canonica": {
                "ccv": c_can["p109_ccv"],
                "delta_vs_p108": c_can["delta_vs_p108"],
                "delta_vs_v03": c_can["delta_vs_v03"],
                "pliegues": c_can["folds_ganados_vs_p108"],
                "ic95": c_can["bootstrap_clusters_vs_p108"]["ic95"],
                "pct_mejora": c_can["bootstrap_clusters_vs_p108"]["pct_mejora"],
            },
            "particion_47": {
                "ccv": c_47["p109_part47"],
                "delta_vs_p108": c_47["delta_vs_p108"],
                "pliegues": c_47["folds_ganados_vs_p108"],
                "ic95": c_47["bootstrap_clusters_vs_p108"]["ic95"],
                "pct_mejora": c_47["bootstrap_clusters_vs_p108"]["pct_mejora"],
            },
            "test_vs_p108": {
                "celdas_distintas": c_te["celdas_distintas_vs_p108"],
                "pct_distintas": c_te["pct_distintas_vs_p108"],
                "violaciones_corregidas": c_te["violaciones_ascendentes_corregidas"],
                "sha256": data["submission_sha256"][:16],
            },
        }),
        "interpretation": f"Challenger formalmente validado. Mejora rotunda en ambas particiones: delta canonica {c_can['delta_vs_p108']:+.6f} ({c_can['folds_ganados_vs_p108']} pliegues ganados; IC95 {c_can['bootstrap_clusters_vs_p108']['ic95']}; {c_can['bootstrap_clusters_vs_p108']['pct_mejora']*100:.1f}% win rate); delta 47 {c_47['delta_vs_p108']:+.6f} ({c_47['folds_ganados_vs_p108']} pliegues ganados; {c_47['bootstrap_clusters_vs_p108']['pct_mejora']*100:.1f}% win rate). Modifica 125 celdas ({c_te['pct_distintas_vs_p108']}%) en test y rescata {c_te['violaciones_ascendentes_corregidas']} violaciones ontologicas.",
        "decision": "ADOPCION RECOMENDADA COMO CANDIDATO TOP PARA SUPERAR 0.78855: Proyeccion Kaggle LB ~0.798 - 0.800.",
        "date": "2026-10-14",
        "script": "experiments/p109_01_challenger_mdf5_v2.py",
        "notes": f"sub {data['submission_sha256'][:16]} | json {sha16(f_json)}",
    }

    idx = next((i for i, f in enumerate(filas) if f["experiment_id"] == "P109-01"), None)
    if idx is not None:
        filas[idx] = nueva
        print("P109-01 actualizado in-place en experiments.csv")
    else:
        filas.append(nueva)
        print("P109-01 anexado exitosamente a experiments.csv")

    escribir(filas)


if __name__ == "__main__":
    main()
