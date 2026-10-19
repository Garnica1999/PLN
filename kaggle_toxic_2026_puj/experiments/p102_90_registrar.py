#!/usr/bin/env python
"""P102-90: registra las filas P102 en experiments.csv (idempotente).

Registra P102-01 (D4 char sin wb) y las filas siguientes cuando existan sus
artefactos en reports/.
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


def fila_p102_01(d: dict) -> dict:
    v = d["variantes"]
    return {
        "experiment_id": "P102-01",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Aportan las variantes char sin wb (con/sin case) sobre el "
                    "arnes REP algo que sobreviva al gate A3?",
        "hypothesis": "Triage D4: la perdida de fronteras de palabra y de "
                      "senal tipografica explica parte del error en tweets "
                      "ruidosos",
        "preprocessing": "texto oficial (ASCII, 0 emojis); vectorizador "
                         "ajustado dentro del pliegue",
        "feature_groups": "TF-IDF word(1,2) + char (variantes)",
        "representation": "union word+char_wb (control) vs +char sin wb "
                          "(1,6) y triple union",
        "model": "LogisticRegression balanced C=2.0 liblinear x3",
        "hyperparameters": jc({"umbrales": "V03 congelados 0.44/0.51/0.46"}),
        "validation": "folds canonicos congelados + particion 47 (hash)",
        "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "",
        "recall": "", "training_time": f"{d['tiempo_s']} s", "dimension": "",
        "result": jc({
            "control_reproduce": d["control"]["reproduce_v03"],
            "V1_char_no_wb": {"delta": v["V1_char_no_wb"]["delta_vs_v03"],
                              "pliegues": v["V1_char_no_wb"]["pliegues_mejorados"],
                              "celdas": v["V1_char_no_wb"]["celdas_cambiadas"]},
            "V2_triple": {"delta": v["V2_triple"]["delta_vs_v03"],
                          "pliegues": v["V2_triple"]["pliegues_mejorados"],
                          "celdas": v["V2_triple"]["celdas_cambiadas"],
                          "ic95": v["V2_triple"]["bootstrap_vs_v03"]["ic95"],
                          "delta_47": v["V2_triple"]["particion_47"]["delta_vs_v03"]},
            "V3_case": {"delta": v["V3_char_no_wb_case"]["delta_vs_v03"],
                        "pliegues": v["V3_char_no_wb_case"]["pliegues_mejorados"]},
            "gate": "GATE_FAILED (mejor: V2 marginal)"}),
        "interpretation": "",
        "decision": "D4 CERRADA sin candidato: ninguna variante pasa el gate "
                    "A3; V2 (triple union) queda como senal marginal "
                    "+0.004466 (3/5 pliegues; IC95 [-0.001664,+0.011226] "
                    "incluye cero; 91 celdas < 200); case-sensible "
                    "descartado (-0.022374).",
        "date": "2026-10-14",
        "script": "experiments/p102_01_char_variantes.py",
        "notes": f"json {sha16(RAIZ / 'reports' / 'p102_01_char_variantes.json')}",
    }


def fila_p102_02(d: dict) -> dict:
    p = d["promedio"]
    return {
        "experiment_id": "P102-02",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "El promedio V03+LinearSVC+ComplementNB con umbrales "
                    "recalibrados tras el promedio supera a V03 (gate A3)?",
        "hypothesis": "Triage C1-corner: la contraccion del promedio (v08_p1) "
                      "se corrige recalibrando umbrales",
        "preprocessing": "texto oficial; vectorizador dentro del pliegue",
        "feature_groups": "union TF-IDF word(1,2)+char_wb(3,5)",
        "representation": "promedio simple de probabilidades LR+SVC+CNB",
        "model": "LR C=2 + LinearSVC C=1 + ComplementNB",
        "hyperparameters": jc({"umbrales_recalibrados":
                               p["umbrales_recalibrados"]}),
        "validation": "folds canonicos + particion 47 (umbrales transferidos)",
        "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "",
        "recall": "", "training_time": f"{d['tiempo_s']} s", "dimension": "",
        "result": jc({"ccv_recalibrado": p["corrected_cv_recalibrado"],
                      "ccv_frozen": p["corrected_cv_frozen_v03"],
                      "delta_vs_v03": p["delta_vs_v03"],
                      "pliegues": p["pliegues_mejorados"],
                      "celdas": p["celdas_cambiadas"],
                      "ic95": p["bootstrap_vs_v03"]["ic95"],
                      "delta_47": p["particion_47"]["delta_recal_vs_v03"],
                      "gate": p["gate"]["veredicto"]}),
        "interpretation": "",
        "decision": "C1-corner CERRADO sin candidato: GATE_FAILED; la "
                    "recalibracion recupera la contraccion de v08_p1 (+0.0299 "
                    "vs umbrales congelados) pero el promedio queda -0.003841 "
                    "bajo V03 (47: -0.009448); SVC solo 0.7715 y CNB 0.6866.",
        "date": "2026-10-14",
        "script": "experiments/p102_02_promedio_recalibrado.py",
        "notes": f"json {sha16(RAIZ / 'reports' / 'p102_02_promedio_recalibrado.json')}",
    }


def fila_p102_03(d: dict) -> dict:
    c = d["canonica"]
    return {
        "experiment_id": "P102-03",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Aplicar la regla dura TR=0 si HS=0 a las predicciones de "
                    "V03 mejora la metrica (gate A3)?",
        "hypothesis": "Triage B2-corner: la invariante TR=1 => HS=1 es exacta "
                      "en train (0/1731)",
        "preprocessing": "n/a (post-proceso sobre predicciones congeladas)",
        "feature_groups": "n/a", "representation": "n/a",
        "model": "regla determinista sobre la salida de V03",
        "hyperparameters": jc({"regla": "TR=0 donde HS=0"}),
        "validation": "folds canonicos + particion 47 (predicciones congeladas)",
        "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "",
        "recall": "", "training_time": f"{d['tiempo_s']} s", "dimension": "",
        "result": jc({"violaciones_canonica": c["violaciones_en_v03"],
                      "delta_canonica": c["delta_vs_v03"],
                      "ic95": c["bootstrap_vs_v03"]["ic95"],
                      "delta_47": d["particion_47"]["delta_vs_v03"],
                      "celdas_submission":
                          d["submission_v03_contexto"]["celdas_que_cambiaria"],
                      "gate": d["gate"]["veredicto"]}),
        "interpretation": "",
        "decision": "B2-corner CERRADO: la regla dura PERJUDICA (-0.000646 "
                    "canonica; -0.004097 en 47): destruye celdas TR correctas "
                    "cuando HS es falso negativo. GATE_FAILED.",
        "date": "2026-10-14",
        "script": "experiments/p102_03_regla_dura_tr.py",
        "notes": f"json {sha16(RAIZ / 'reports' / 'p102_03_regla_dura_tr.json')}",
    }


def fila_p102_04(d: dict) -> dict:
    c = d["combo_argmax_canonico"]
    return {
        "experiment_id": "P102-04",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Esta V03 en el maximo local de umbrales sobre el OOF; "
                    "existe punto cercano que pase el gate A3?",
        "hypothesis": "Triage: densificar la curva de umbrales (v16 la midio "
                      "con 4 puntos de LB)",
        "preprocessing": "n/a (proba congelada)",
        "feature_groups": "n/a", "representation": "n/a",
        "model": "umbrales sobre la proba de V03",
        "hyperparameters": jc({"umbrales_argmax": c["umbrales"]}),
        "validation": "folds canonicos + particion 47 (proba_v03 congelada)",
        "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "",
        "recall": "", "training_time": f"{d['tiempo_s']} s", "dimension": "",
        "result": jc({"delta_canonica": c["delta_vs_v03"],
                      "celdas": c["celdas_cambiadas"],
                      "ic95": c["bootstrap_vs_v03"]["ic95"],
                      "delta_47": c["particion_47_transferido"]["delta_vs_v03"],
                      "argmax_47": d["argmax_47"],
                      "gate": d["gate"]["veredicto"]}),
        "interpretation": "",
        "decision": "Barrido CERRADO: el argmax canonico gana +0.003895 pero "
                    "IC incluye cero, la 47 es negativa (-0.002002) y solo "
                    "cambia 83 celdas; los optimos de la 47 no coinciden (HS "
                    "0.365/TR 0.505/AG 0.490). V03 sigue en el maximo estable. "
                    "GATE_FAILED.",
        "date": "2026-10-14",
        "script": "experiments/p102_04_barrido_umbrales.py",
        "notes": f"json {sha16(RAIZ / 'reports' / 'p102_04_barrido_umbrales.json')}",
    }


def main() -> int:
    filas = leer()
    ids = {f["experiment_id"] for f in filas}
    nuevas = []
    pendientes = [
        ("P102-01", "p102_01_char_variantes.json", fila_p102_01),
        ("P102-02", "p102_02_promedio_recalibrado.json", fila_p102_02),
        ("P102-03", "p102_03_regla_dura_tr.json", fila_p102_03),
        ("P102-04", "p102_04_barrido_umbrales.json", fila_p102_04),
    ]
    for eid, nombre, constructor in pendientes:
        ruta = RAIZ / "reports" / nombre
        if eid not in ids and ruta.exists():
            nuevas.append(constructor(json.loads(ruta.read_text(
                encoding="utf-8"))))
    if nuevas:
        anexar(nuevas)
    print(f"P102-90: {len(nuevas)} fila(s) anadida(s); "
          f"total={len(filas) + len(nuevas)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
