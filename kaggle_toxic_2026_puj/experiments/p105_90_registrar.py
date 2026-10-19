#!/usr/bin/env python
"""P105-90: registra las filas P105 en experiments.csv (idempotente).

Registra P105-01 (S2/S7a normalizacion), P105-02 (S4/S7b fonetica) y
P105-03 (S7c pesos por etiqueta) cuando existan sus artefactos en reports/.
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


def _resumen(r: dict) -> dict:
    """Resumen compacto de una variante."""
    c, p = r["canonica"], r["particion_47"]
    return {"delta": c["delta_vs_v03"],
            "pliegues": c["pliegues_mejorados"],
            "celdas": c["celdas_cambiadas"],
            "ic95": c["bootstrap"]["ic95"],
            "delta_47": p["delta_vs_v03"],
            "gate": r["gate"]["veredicto"]}


def fila_p105_01(d: dict) -> dict:
    """Fila P105-01 (S2/S7a) desde su JSON."""
    v = d["variantes"]
    return {
        "experiment_id": "P105-01",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Supera alguna normalizacion alternativa de la union "
                    "word+char (L2 por bloque, L2 conjunta, norma media) el "
                    "gate A3?",
        "hypothesis": "S2/S7a: la normalizacion por bloque vs conjunta cambia "
                      "el balance word/char y puede aportar",
        "preprocessing": "texto oficial; bloques TF-IDF crudos (norm=None) + "
                         "normalizacion de la variante; LR dentro del pliegue",
        "feature_groups": "word(1,2) + char_wb(3,5) (union de V03)",
        "representation": "V1 L2 por bloque (identidad); V2 L2 conjunta; "
                          "V3 reescalado a norma media",
        "model": "LogisticRegression balanced C=2.0 liblinear x3",
        "hyperparameters": jc({"umbrales": "V03 congelados"}),
        "validation": "folds canonicos + particion 47",
        "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "",
        "recall": "", "training_time": f"{d['tiempo_s']} s", "dimension": "",
        "result": jc({
            "control_reproduce": d["control"]["reproduce_v03"],
            "V1_identidad": {"confirmada": d["v1_identidad_confirmada"],
                             "delta": 0.0, "celdas": 0},
            "V2_l2_conjunta": _resumen(v["V2_l2_conjunta"]),
            "V3_norma_media": _resumen(v["V3_norma_media"]),
            "gate": "GATE_FAILED"}),
        "interpretation": "",
        "decision": "S2/S7a CERRADO sin candidato: V1 (L2 por bloque) es "
                    "identidad matematica con V03 (diff 0.0; no compite); "
                    "V2 (-0.017517, IC95 [-0.028724,-0.006328]) y V3 "
                    "(-0.034369) PERJUDICAN con efecto negativo detectable. "
                    "La union cruda de V03 queda validada como optima entre "
                    "las tres.",
        "date": "2026-10-14",
        "script": "experiments/p105_01_normalizacion_bloques.py",
        "notes": f"json {sha16(RAIZ / 'reports' / 'p105_01_normalizacion_bloques.json')}"
                 f" | preregistro {sha16(RAIZ / 'reports' / 'p105_preregistro.md')}",
    }


def fila_p105_02(d: dict) -> dict:
    """Fila P105-02 (S4/S7b) desde su JSON."""
    v = d["variantes"]["F_fonetica_canal"]
    return {
        "experiment_id": "P105-02",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Anade el folding fonetico espanol (b/v, ll/y, h muda, "
                    "c/k/qu, s/z) como canal char adicional algo que supere "
                    "el gate A3?",
        "hypothesis": "S4/S7b: el canal fonetico captura variacion de "
                      "escritura que char_wb no pliega",
        "preprocessing": "texto oficial + texto plegado (reglas "
                         "deterministas, sin ajuste); vectorizadores dentro "
                         "del pliegue",
        "feature_groups": "word(1,2) + char_wb(3,5) + char_wb(3,5) sobre "
                          "texto plegado",
        "representation": "canal fonetico adicional sobre el bloque char",
        "model": "LogisticRegression balanced C=2.0 liblinear x3",
        "hyperparameters": jc({"reglas": "qu->k; ce/ci->se/si; z->s; c->k; "
                                         "ll->y; v->b; h muda salvo tras c"}),
        "validation": "folds canonicos + particion 47",
        "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "",
        "recall": "", "training_time": f"{d['tiempo_s']} s", "dimension": "",
        "result": jc({
            "control_reproduce": d["control"]["reproduce_v03"],
            "F_fonetica_canal": _resumen(v),
            "piso_deteccion": v["canonica"]["deteccion"]
                               ["delta_min_detectable_95"],
            "gate": "GATE_FAILED"}),
        "interpretation": "",
        "decision": "S4/S7b CERRADO sin candidato: sin senal detectable "
                    "(delta -0.000363 bajo el piso +0.005798 e IC95 "
                    "[-0.006092,+0.005504] que incluye cero; 96 celdas); 47 "
                    "+0.000414 sin celdas suficientes. El canal fonetico no "
                    "aporta sobre char_wb.",
        "date": "2026-10-14",
        "script": "experiments/p105_02_fonetica.py",
        "notes": f"json {sha16(RAIZ / 'reports' / 'p105_02_fonetica.json')}"
                 f" | preregistro {sha16(RAIZ / 'reports' / 'p105_preregistro.md')}",
    }


def fila_p105_03(d: dict) -> dict:
    """Fila P105-03 (S7c) desde su JSON."""
    c, p = d["canonica"], d["particion_47"]
    return {
        "experiment_id": "P105-03",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Elegir el peso del bloque char POR ETIQUETA (malla "
                    "0.5-1.5, seleccion anidada) supera el gate A3?",
        "hypothesis": "S7c: la libertad de peso word/char por etiqueta aporta "
                      "senal",
        "preprocessing": "texto oficial; bloques estandar de V03; peso char "
                         "elegido dentro del pliegue",
        "feature_groups": "word(1,2) + w * char_wb(3,5), w por etiqueta",
        "representation": "union ponderada por etiqueta (malla 0.5-1.5)",
        "model": "LogisticRegression balanced C=2.0 liblinear x3",
        "hyperparameters": jc({"malla": list(d["malla"]),
                               "seleccion": "CV interna 3 pliegues; F1 "
                                            "@0.5 por etiqueta"}),
        "validation": "folds canonicos + particion 47",
        "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "",
        "recall": "", "training_time": f"{d['tiempo_s']} s", "dimension": "",
        "result": jc({
            "control_reproduce": d["control"]["reproduce_v03"],
            "delta_canonica": c["delta_vs_v03"],
            "pliegues": c["pliegues_mejorados"],
            "celdas": c["celdas_cambiadas"],
            "ic95": c["bootstrap"]["ic95"],
            "piso_deteccion": c["deteccion"]["delta_min_detectable_95"],
            "delta_47": p["delta_vs_v03"],
            "estabilidad": d["estabilidad_canonica"],
            "gate": d["gate"]["veredicto"]}),
        "interpretation": "",
        "decision": "S7c CERRADO sin candidato: sin senal detectable (delta "
                    "-0.001125; IC95 [-0.005996,+0.003508] incluye cero; 67 "
                    "celdas); seleccion inestable (HS 1.25/1.5; TR 0.5-1.25; "
                    "AG 0.75-1.5); GATE_FAILED.",
        "date": "2026-10-14",
        "script": "experiments/p105_03_pesos_bloque_etiqueta.py",
        "notes": f"json {sha16(RAIZ / 'reports' / 'p105_03_pesos_bloque_etiqueta.json')}"
                 f" | preregistro {sha16(RAIZ / 'reports' / 'p105_preregistro.md')}",
    }


def main() -> int:
    """Registra las filas P105 pendientes."""
    filas = leer()
    ids = {f["experiment_id"] for f in filas}
    nuevas = []
    pendientes = [
        ("P105-01", "p105_01_normalizacion_bloques.json", fila_p105_01),
        ("P105-02", "p105_02_fonetica.json", fila_p105_02),
        ("P105-03", "p105_03_pesos_bloque_etiqueta.json", fila_p105_03),
    ]
    for eid, nombre, constructor in pendientes:
        ruta = RAIZ / "reports" / nombre
        if eid not in ids and ruta.exists():
            nuevas.append(constructor(json.loads(ruta.read_text(
                encoding="utf-8"))))
    if nuevas:
        anexar(nuevas)
    print(f"P105-90: {len(nuevas)} fila(s) anadida(s); "
          f"total={len(filas) + len(nuevas)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
