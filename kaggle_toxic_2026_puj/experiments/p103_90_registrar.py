#!/usr/bin/env python
"""P103-90: registra las filas P103 en experiments.csv (idempotente).

Registra P103-01 (N5 vectorizadores por etiqueta) y P103-02 (N6 word dropout
conservador) cuando existan sus artefactos en reports/.
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
    """sha256 truncado a 16 hex."""
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def jc(valor) -> str:
    """JSON compacto ASCII."""
    return json.dumps(valor, ensure_ascii=True, separators=(",", ":"))


def leer():
    """Lee las filas actuales del registro."""
    with open(RUTA, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def anexar(nuevas) -> None:
    """Anade filas al final del registro."""
    with open(RUTA, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS, lineterminator="\n")
        for f in nuevas:
            w.writerow(f)


def fila_p103_01(d: dict) -> dict:
    """Fila P103-01 (N5) desde su JSON."""
    c = d["canonica"]
    p47 = d["particion_47"]
    return {
        "experiment_id": "P103-01",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Elige cada etiqueta (HS/TR/AG) su propio espacio TF-IDF "
                    "(min_df, max_df, n-gramas) con seleccion anidada y "
                    "supera el gate A3 frente a V03?",
        "hypothesis": "N5 (RONDA 2): la libertad de espacio por etiqueta, "
                      "seleccionada sin mirar el OOF global, aporta senal",
        "preprocessing": "texto oficial (ASCII); vectorizador y seleccion "
                         "dentro del pliegue; umbrales V03 congelados",
        "feature_groups": "TF-IDF word + char_wb con malla por etiqueta",
        "representation": "word {(1,1),(1,2),(1,3)} + char_wb {(2,5),(3,5),"
                          "(3,6)}; min_df {1,2,3}; max_df {0.9,1.0}",
        "model": "LogisticRegression balanced C=2.0 liblinear x3 "
                 "(espacio elegido por etiqueta)",
        "hyperparameters": jc({"malla": 20,
                               "seleccion": "CV interna 3 pliegues seed 42; "
                                            "criterio F1 de la etiqueta @0.5"}),
        "validation": "folds canonicos + particion 47 (nested en ambas)",
        "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "",
        "recall": "", "training_time": f"{d['tiempo_s']} s", "dimension": "",
        "result": jc({
            "control_reproduce": d["control"]["reproduce_v03"],
            "delta_canonica": c["delta_vs_v03"],
            "pliegues": c["pliegues_mejorados"],
            "celdas": c["celdas_cambiadas"],
            "ic95": c["bootstrap"]["ic95"],
            "pct_mejora": c["bootstrap"]["pct_mejora"],
            "piso_deteccion": c["deteccion"]["delta_min_detectable_95"],
            "delta_47": p47["delta_vs_v03"],
            "celdas_47": p47["celdas_cambiadas"],
            "gate": d["gate"]["veredicto"]}),
        "interpretation": "",
        "decision": "N5 CERRADO sin candidato: GATE_FAILED; delta +0.002389 "
                    "por debajo del piso de deteccion (+0.004828) e IC95 "
                    "incluye cero; 64 celdas (<200). Direccion observada no "
                    "adoptada: HS max_df 0.9 + word (1,3); AG char (2,5); TR "
                    "inestable. Sin senal detectable.",
        "date": "2026-10-14",
        "script": "experiments/p103_01_vectorizadores_por_etiqueta.py",
        "notes": f"json {sha16(RAIZ / 'reports' / 'p103_01_vectorizadores_por_etiqueta.json')}"
                 f" | preregistro {sha16(RAIZ / 'reports' / 'p103_preregistro.md')}",
    }


def fila_p103_02(d: dict) -> dict:
    """Fila P103-02 (N6) desde su JSON."""
    c = d["canonica"]
    p47 = d["particion_47"]
    return {
        "experiment_id": "P103-02",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Aumenta el word dropout conservador (p 0.05-0.15; 1-2 "
                    "copias) solo el train del pliegue y supera el gate A3 "
                    "frente a V03?",
        "hypothesis": "N6 (RONDA 2): el unico hueco de augmentation "
                      "reconocido por V41-01 (word dropout/masking) aporta",
        "preprocessing": "texto oficial; aumento solo del train del pliegue; "
                         "vectorizador y LR dentro del pliegue",
        "feature_groups": "TF-IDF word(1,2) + char_wb(3,5) (M2 fijo)",
        "representation": "M2 (V03) sobre train aumentado con copias de "
                          "dropout de tokens",
        "model": "LogisticRegression balanced C=2.0 liblinear x3",
        "hyperparameters": jc({"malla": [[0.05, 1], [0.05, 2], [0.1, 1],
                                         [0.1, 2], [0.15, 1], [0.15, 2]],
                               "semilla_aug": 20261014}),
        "validation": "folds canonicos + particion 47 (nested en ambas)",
        "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "",
        "recall": "", "training_time": f"{d['tiempo_s']} s", "dimension": "",
        "result": jc({
            "control_reproduce": d["control"]["reproduce_v03"],
            "delta_canonica": c["delta_vs_v03"],
            "pliegues": c["pliegues_mejorados"],
            "celdas": c["celdas_cambiadas"],
            "ic95": c["bootstrap"]["ic95"],
            "pct_mejora": c["bootstrap"]["pct_mejora"],
            "piso_deteccion": c["deteccion"]["delta_min_detectable_95"],
            "delta_47": p47["delta_vs_v03"],
            "celdas_47": p47["celdas_cambiadas"],
            "gate": d["gate"]["veredicto"]}),
        "interpretation": "",
        "decision": "N6 CERRADO sin candidato: GATE_FAILED; el word dropout "
                    "PERJUDICA (-0.006873 canonica, 1/5 pliegues; -0.004265 "
                    "en la 47); coherente con V6-08 (aumento negativo) y "
                    "P94-R16. Sin senal detectable.",
        "date": "2026-10-14",
        "script": "experiments/p103_02_word_dropout.py",
        "notes": f"json {sha16(RAIZ / 'reports' / 'p103_02_word_dropout.json')}"
                 f" | preregistro {sha16(RAIZ / 'reports' / 'p103_preregistro.md')}",
    }


def main() -> int:
    """Registra las filas P103 pendientes."""
    filas = leer()
    ids = {f["experiment_id"] for f in filas}
    nuevas = []
    pendientes = [
        ("P103-01", "p103_01_vectorizadores_por_etiqueta.json", fila_p103_01),
        ("P103-02", "p103_02_word_dropout.json", fila_p103_02),
    ]
    for eid, nombre, constructor in pendientes:
        ruta = RAIZ / "reports" / nombre
        if eid not in ids and ruta.exists():
            nuevas.append(constructor(json.loads(ruta.read_text(
                encoding="utf-8"))))
    if nuevas:
        anexar(nuevas)
    print(f"P103-90: {len(nuevas)} fila(s) anadida(s); "
          f"total={len(filas) + len(nuevas)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
