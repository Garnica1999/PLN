#!/usr/bin/env python
"""P106-90: registra la fila P106-01 en experiments.csv (idempotente)."""

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


def anexar(nuevas) -> None:
    with open(RUTA, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS, lineterminator="\n")
        for f in nuevas:
            w.writerow(f)


def main() -> None:
    f_json = RAIZ / "reports" / "p106_01_nuevos_experimentos.json"
    if not f_json.exists():
        print("No existe el reporte json")
        return

    data = json.loads(f_json.read_text(encoding="utf-8"))
    v = data["variantes"]

    filas = leer()
    existentes = {f["experiment_id"] for f in filas}
    if "P106-01" in existentes:
        print("P106-01 ya esta registrado en experiments.csv")
        return

    nueva = {
        "experiment_id": "P106-01",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Supera alguna de las 4 nuevas tecnicas (boost TR->HS, desactivacion afecto, umbrales 3D anidados, fusion logits LinearSVC) el gate A3 frente a V03?",
        "hypothesis": "P106: exploracion de dependencias suaves, atenuacion afectiva residual y umbrales conjuntos anidados para cerrar la brecha hacia 0.80",
        "preprocessing": "texto oficial; representacion V03 dentro del pliegue",
        "feature_groups": "word(1,2) + char_wb(3,5) de V03",
        "representation": "union TF-IDF M2",
        "model": "LogisticRegression balanced C=2.0 liblinear x3",
        "hyperparameters": jc({"umbrales": "V03 congelados 0.44/0.51/0.46"}),
        "validation": "folds canonicos + particion 47",
        "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "", "recall": "",
        "training_time": f"{data['duracion_segundos']} s",
        "dimension": "",
        "result": jc({
            "control_reproduce": data["control_canonica"]["ok"],
            "V1_boost_tr_hs": {
                "delta": v["V1_boost_tr_hs"]["canonica"]["delta_vs_v03"],
                "pliegues": v["V1_boost_tr_hs"]["canonica"]["pliegues_mejorados"],
                "celdas": v["V1_boost_tr_hs"]["canonica"]["celdas_cambiadas"],
                "delta_47": v["V1_boost_tr_hs"]["particion_47"]["delta_vs_v03"],
                "gate": v["V1_boost_tr_hs"]["gate"]["veredicto"],
            },
            "V2_desactivacion_afecto": {
                "delta": v["V2_desactivacion_afecto"]["canonica"]["delta_vs_v03"],
                "pliegues": v["V2_desactivacion_afecto"]["canonica"]["pliegues_mejorados"],
                "celdas": v["V2_desactivacion_afecto"]["canonica"]["celdas_cambiadas"],
                "delta_47": v["V2_desactivacion_afecto"]["particion_47"]["delta_vs_v03"],
                "gate": v["V2_desactivacion_afecto"]["gate"]["veredicto"],
            },
            "V3_umbrales_3d_anidados": {
                "delta": v["V3_umbrales_3d_anidados"]["canonica"]["delta_vs_v03"],
                "pliegues": v["V3_umbrales_3d_anidados"]["canonica"]["pliegues_mejorados"],
                "celdas": v["V3_umbrales_3d_anidados"]["canonica"]["celdas_cambiadas"],
                "delta_47": v["V3_umbrales_3d_anidados"]["particion_47"]["delta_vs_v03"],
                "gate": v["V3_umbrales_3d_anidados"]["gate"]["veredicto"],
            },
            "V4_fusion_logits_linearsvc": {
                "delta": v["V4_fusion_logits_linearsvc"]["canonica"]["delta_vs_v03"],
                "pliegues": v["V4_fusion_logits_linearsvc"]["canonica"]["pliegues_mejorados"],
                "celdas": v["V4_fusion_logits_linearsvc"]["canonica"]["celdas_cambiadas"],
                "delta_47": v["V4_fusion_logits_linearsvc"]["particion_47"]["delta_vs_v03"],
                "gate": v["V4_fusion_logits_linearsvc"]["gate"]["veredicto"],
            },
        }),
        "interpretation": "Las 4 tecnicas fallan el gate A3. V3 mejora levemente en canonica (+0.001667, 4/5) pero degrada en particion 47 (-0.002058). V1 da senal marginal indetectable (+0.000239).",
        "decision": "CERRADO sin candidato: V03 se mantiene como la solucion verificada.",
        "date": "2026-10-14",
        "script": "experiments/p106_01_nuevos_experimentos.py",
        "notes": f"json {sha16(f_json)}",
    }
    anexar([nueva])
    print("P106-01 registrado correctamente en experiments.csv")


if __name__ == "__main__":
    main()
