#!/usr/bin/env python
"""
P98-90: registra las unidades de la Fase A de P98 en experiments.csv.

Unidades: P98-00 (registro de fuentes y licencias), P98-01 (prueba tecnica de
infraestructura con datos sinteticos; no es experimento principal), P98-02
(auditoria de solapamiento). Solo anade filas nuevas.

Uso:
    /home/carlos/venv/bin/python experiments/p98_90_registrar.py
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
    """sha256 truncado a 16 hexdigitos."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()[:16]


def jc(valor: object) -> str:
    """JSON compacto ASCII."""
    return json.dumps(valor, ensure_ascii=True, separators=(",", ":"))


def main() -> None:
    """Anade las unidades de Fase A."""
    with open(RUTA, encoding="utf-8-sig", newline="") as fh:
        filas = list(csv.DictReader(fh))
    if any(f["experiment_id"].startswith("P98") for f in filas):
        raise SystemExit("Ya hay filas P98; no se duplica.")
    n_antes = len(filas)
    reg = list(csv.DictReader(open(RAIZ / "reports" / "p98_corpus_registry.csv",
                                   encoding="utf-8")))
    infra = json.load(open(RAIZ / "reports" / "p98_infra_test.json",
                           encoding="utf-8"))
    audit = json.load(open(RAIZ / "reports" / "p98_auditoria_solapamiento.json",
                           encoding="utf-8"))
    s00 = sha16(RAIZ / "experiments" / "p98_00_registro_fuentes.py")
    s01 = sha16(RAIZ / "experiments" / "p98_01_prueba_infraestructura.py")
    s02 = sha16(RAIZ / "experiments" / "p98_02_auditoria_solapamiento.py")

    def fila(eid, question, hypothesis, preprocessing, representation, model,
             validation, result, decision, script, notes):
        """Fila con el esquema de experiments.csv."""
        return {
            "experiment_id": eid, "competition": "KAGGLE_TOXIC_2026_PUJ",
            "question": question, "hypothesis": hypothesis,
            "preprocessing": preprocessing, "feature_groups": "no aplica",
            "representation": representation, "model": model,
            "hyperparameters": "", "validation": validation, "f1": "",
            "macro_f1": "", "weighted_f1": "", "precision": "", "recall": "",
            "training_time": "", "dimension": "", "result": result,
            "interpretation": "", "decision": decision, "date": "2026-10-13",
            "script": script, "notes": notes,
        }

    from collections import Counter
    estados = Counter(f["permission_status"] for f in reg)
    nuevas = [
        fila("P98-00",
             "Que fuentes externas pueden usarse legalmente para el LM "
             "contextual desde cero y bajo que licencia?",
             "Registro de fuentes con clasificacion ALLOWED / "
             "ALLOWED_WITH_RESTRICTIONS / NEEDS_EXPLICIT_PERMISSION / "
             "PROHIBITED",
             "auditoria documental y de hashes (sin descarga)",
             "no aplica (gobernanza)",
             "no aplica (gobernanza)",
             "verificacion de 6 hashes de corpus contra manifiesto v5 y P73",
             jc({"fuentes_registradas": len(reg), "por_estado": dict(estados),
                 "autorizados_operativos": ["C98-SRC-01", "C98-SRC-02",
                                            "C98-SRC-03", "C98-SRC-04",
                                            "C98-SRC-05", "C98-SRC-06"],
                 "cuarentena_odio": ["C98-SRC-40", "C98-SRC-41", "C98-SRC-42",
                                     "C98-SRC-43", "C98-SRC-44"]}),
             "REGISTRO: 17 fuentes; 4 ALLOWED + 4 ALLOWED_WITH_RESTRICTIONS + "
             "9 NEEDS_EXPLICIT_PERMISSION (5 de odio en cuarentena activa); "
             "familia Common Crawl tratada como una sola fuente.",
             "experiments/p98_00_registro_fuentes.py",
             f"git_head e43c594 | autorizacion: docs/p98_provenance_"
             f"autorizacion_bilstm.md | sha16 {s00} | salidas: "
             f"reports/p98_corpus_registry.csv, p98_corpus_licenses.md, "
             f"artifacts/p98_corpus_manifest.json"),
        fila("P98-01",
             "La infraestructura AMD/ROCm soporta el pipeline del LM char-aware "
             "con determinismo y checkpoint reproducibles? (datos sinteticos)",
             "Prueba tecnica de Fase A: sin corpus reales",
             "corpus sintetico zipf (1.5M tokens) con formas elongadas",
             "char-aware (conv 3/5) + LSTM + softmax atado",
             "torch 2.14.0+rocm7.2; Adam lr 3e-3; 300 pasos",
             "sintetica (no aplica a la competencia)",
             jc({"device": infra["device"], "n_params": infra["n_params"],
                 "loss": [round(infra["perdida_primera"], 3),
                          round(infra["perdida_ultima"], 3)],
                 "val_loss": round(infra["val_loss"], 3),
                 "tokens_por_segundo": infra["tokens_por_segundo"],
                 "vram_pico_gb": infra["vram_pico_gb"],
                 "checkpoint_roundtrip_ok": infra["checkpoint_roundtrip_ok"],
                 "determinismo_delta": infra["determinismo_delta_30_pasos"]}),
             "INFRA_OK: ROCm estable, convergencia, checkpoint identico y "
             "determinismo exacto (delta 0.0). No es experimento principal.",
             "experiments/p98_01_prueba_infraestructura.py",
             f"seed 20260925 | config sintetica ~50x menor que la objetivo | "
             f"sha16 {s01} | json {sha16(RAIZ/'reports'/'p98_infra_test.json')}"),
        fila("P98-02",
             "El corpus externo autorizado comparte texto con el train de TOXIC?",
             "Condicion de la autorizacion: corpus distinto de los datos de la "
             "competicion",
             "normalizacion conservadora (minusculas, sin tildes)",
             "no aplica (auditoria)",
             "no aplica (auditoria)",
             "exacto por documento en 4 corpus + 8-gramas (twitter completo; "
             "wiki y mC4 muestreados a 1.5M tokens)",
             jc({"coincidencias_exactas": audit["coincidencias_exactas_total"],
                 "coincidencias_8gramas": audit["coincidencias_8gramas_total"],
                 "jaccard_max_pares": audit["jaccard_max_pares"],
                 "categoria_unica": "puntuacion (ocho puntos consecutivos)",
                 "veredicto": audit["veredicto"]}),
             "CUMPLE: 0 coincidencias exactas; la unica coincidencia de "
             "8-gramas resulto ser puntuacion (sin contenido compartido); "
             "frente a toxic_test no se escanea (regla vigente).",
             "experiments/p98_02_auditoria_solapamiento.py",
             f"linea base P73-01 citada (max Jaccard 0.142857 < 0.2) | "
             f"sha16 {s02} | json "
             f"{sha16(RAIZ/'reports'/'p98_auditoria_solapamiento.json')}"),
    ]
    with open(RUTA, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS, lineterminator="\n")
        for f in nuevas:
            w.writerow(f)
    with open(RUTA, encoding="utf-8-sig", newline="") as fh:
        filas2 = list(csv.DictReader(fh))
    lineas = (RAIZ / "experiments.csv").read_text(encoding="utf-8").splitlines()
    nuevas_ascii = [l for l in lineas if l.startswith("P98-")]
    assert all(all(ord(c) <= 127 for c in l) for l in nuevas_ascii)
    print("P98-90 registro Fase A: ejecutado")
    print(f"  filas: {n_antes} -> {len(filas2)} (+{len(filas2) - n_antes}) | "
          f"P98: {len([f for f in filas2 if f['experiment_id'].startswith('P98')])}")


if __name__ == "__main__":
    main()
