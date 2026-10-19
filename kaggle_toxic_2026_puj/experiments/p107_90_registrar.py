#!/usr/bin/env python
"""P107-90: registra o actualiza la fila P107-01 en experiments.csv (idempotente)."""

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
    f_json = RAIZ / "reports" / "p107_01_arquitecturas_aumento_ensemble.json"
    if not f_json.exists():
        print("No existe el reporte json")
        return

    data = json.loads(f_json.read_text(encoding="utf-8"))
    v = data["variantes"]

    filas = leer()

    nueva = {
        "experiment_id": "P107-01",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Supera alguna de las 4 tecnicas (aumento caracteres, mixup continuo, 1D CharCNN PyTorch, FastText scratch) el gate A3 frente a V03?",
        "hypothesis": "P107: exploracion de arquitecturas no lineales desde cero (FastText, CharCNN), aumento ortografico y mixup continuo hacia 0.80",
        "preprocessing": "texto oficial; aumento y codificacion estrictamente dentro del pliegue sin fuga",
        "feature_groups": "word(1,2) + char_wb(3,5) de V03, secuencias de caracteres y subpalabras",
        "representation": "union TF-IDF M2, indices de caracteres y subpalabras n-gramas",
        "model": "LogisticRegression C=2.0, Ridge continuo, CharCNN1D PyTorch, FastText1D PyTorch",
        "hyperparameters": jc({"umbrales": "V03 congelados 0.44/0.51/0.46"}),
        "validation": "folds canonicos + particion 47",
        "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "", "recall": "",
        "training_time": f"{data['duracion_segundos']} s",
        "dimension": "",
        "result": jc({
            "control_reproduce": data["control_canonica"]["ok"],
            "V1_aumento_caracteres": {
                "delta": v["V1_aumento_caracteres"]["canonica"]["delta_vs_v03"],
                "pliegues": v["V1_aumento_caracteres"]["canonica"]["pliegues_mejorados"],
                "celdas": v["V1_aumento_caracteres"]["canonica"]["celdas_cambiadas"],
                "delta_47": v["V1_aumento_caracteres"]["particion_47"]["delta_vs_v03"],
                "gate": v["V1_aumento_caracteres"]["gate"]["veredicto"],
            },
            "V2_mixup_features": {
                "delta": v["V2_mixup_features"]["canonica"]["delta_vs_v03"],
                "pliegues": v["V2_mixup_features"]["canonica"]["pliegues_mejorados"],
                "celdas": v["V2_mixup_features"]["canonica"]["celdas_cambiadas"],
                "delta_47": v["V2_mixup_features"]["particion_47"]["delta_vs_v03"],
                "gate": v["V2_mixup_features"]["gate"]["veredicto"],
            },
            "V3_char_cnn_lightweight": {
                "delta": v["V3_char_cnn_lightweight"]["canonica"]["delta_vs_v03"],
                "pliegues": v["V3_char_cnn_lightweight"]["canonica"]["pliegues_mejorados"],
                "celdas": v["V3_char_cnn_lightweight"]["canonica"]["celdas_cambiadas"],
                "delta_47": v["V3_char_cnn_lightweight"]["particion_47"]["delta_vs_v03"],
                "gate": v["V3_char_cnn_lightweight"]["gate"]["veredicto"],
            },
            "V4_fasttext_scratch": {
                "delta": v["V4_fasttext_scratch"]["canonica"]["delta_vs_v03"],
                "pliegues": v["V4_fasttext_scratch"]["canonica"]["pliegues_mejorados"],
                "celdas": v["V4_fasttext_scratch"]["canonica"]["celdas_cambiadas"],
                "delta_47": v["V4_fasttext_scratch"]["particion_47"]["delta_vs_v03"],
                "gate": v["V4_fasttext_scratch"]["gate"]["veredicto"],
            },
        }),
        "interpretation": "Las 4 tecnicas fallan el gate A3 de forma concluyente. V1 aumento con reajuste degrada al diluir TF-IDF; V2 mixup continuo degrada severamente la separabilidad dispersa; V3 CharCNN leak-free y V4 FastText desde cero PyTorch no superan la regularizacion L2 analitica de V03.",
        "decision": "CERRADO sin candidato: V03 se mantiene firmemente congelado como champion verificado.",
        "date": "2026-10-14",
        "script": "experiments/p107_01_arquitecturas_aumento_ensemble.py",
        "notes": f"json {sha16(f_json)}",
    }

    actualizadas = []
    encontrado = False
    for f in filas:
        if f["experiment_id"] == "P107-01":
            actualizadas.append(nueva)
            encontrado = True
        else:
            actualizadas.append(f)
    if not encontrado:
        actualizadas.append(nueva)

    escribir(actualizadas)
    print("P107-01 registrado/actualizado correctamente en experiments.csv")


if __name__ == "__main__":
    main()
