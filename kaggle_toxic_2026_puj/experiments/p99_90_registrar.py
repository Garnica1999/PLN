#!/usr/bin/env python
"""P99-90: registra las filas P99 en experiments.csv (idempotente).

Registra P99-00 (pre-registro) y, si existen sus artefactos, P99-11/12
(entrenamiento LTR/RTL) y P99-13/14 (downstream canonica/particion 47).
"""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[1]
RUTA = RAIZ / "experiments.csv"
COLUMNAS = ["experiment_id", "competition", "question", "hypothesis",
            "preprocessing", "feature_groups", "representation", "model",
            "hyperparameters", "validation", "f1", "macro_f1", "weighted_f1",
            "precision", "recall", "training_time", "dimension", "result",
            "interpretation", "decision", "date", "script", "notes"]
def sha16(r):
    x = hashlib.sha256()
    with open(r, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""): x.update(b)
    return x.hexdigest()[:16]
def jc(v): return json.dumps(v, ensure_ascii=True, separators=(",", ":"))
def leer():
    with open(RUTA, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))
def anexar(nuevas):
    with open(RUTA, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS, lineterminator="\n")
        for f in nuevas: w.writerow(f)
def main():
    filas = leer()
    ids = {f["experiment_id"] for f in filas}
    nuevas = []
    if "P99-00" not in ids:
        nuevas.append({
            "experiment_id": "P99-00", "competition": "KAGGLE_TOXIC_2026_PUJ",
            "question": "El corpus del LM contextual escala a 60M tokens unicos "
                        "con material local autorizado, manteniendo arquitectura "
                        "y exposicion?",
            "hypothesis": "H99-01 pre-registrada en reports/p99_preregistration.md",
            "preprocessing": "n/a (pre-registro)",
            "feature_groups": "n/a", "representation": "n/a",
            "model": "n/a (pre-registro)", "hyperparameters": "n/a",
            "validation": "n/a", "f1": "", "macro_f1": "", "weighted_f1": "",
            "precision": "", "recall": "", "training_time": "", "dimension": "",
            "result": jc({"nominal": {"wiki": 36000000, "mc4": 15000000,
                                      "twitter": 9000000},
                          "efectivo_planificado": {"wiki": 41000000,
                                                   "mc4": 10000000,
                                                   "twitter": 9000000},
                          "justificacion": "mC4 local tope 10M; sin descargas",
                          "gate": "delta>=+0.003 canonica, >=4/5 pliegues, "
                                  "positivo vs 30M, positivo en particion 47, "
                                  "neto>0"}),
            "interpretation": "",
            "decision": "PRE-REGISTRADO antes de construir; H99-01 y sus "
                        "criterios de falsacion quedan fijados en "
                        "p99_preregistration.md y p99_corpus_scaling_plan.md.",
            "date": "2026-10-13", "script": "experiments/p99_90_registrar.py",
            "notes": f"plan {sha16(RAIZ/'reports'/'p99_corpus_scaling_plan.md')} | "
                     f"preregistro {sha16(RAIZ/'reports'/'p99_preregistration.md')}",
        })
    pre = RAIZ / "reports" / "p99_pretraining.json"
    if pre.exists():
        d = json.loads(pre.read_text(encoding="utf-8"))
        for eid, dir_ in (("P99-11", "ltr"), ("P99-12", "rtl")):
            if eid in ids: continue
            c = d["checkpoints"][dir_]
            hist = d["direcciones"][dir_]["historial"]
            nuevas.append({
                "experiment_id": eid, "competition": "KAGGLE_TOXIC_2026_PUJ",
                "question": f"S99-01 {dir_.upper()}: 60M tokens unicos x 2 "
                            f"epocas converge y con que perplexity?",
                "hypothesis": "H99-01: mas documentos unicos, misma arquitectura "
                              "y exposicion",
                "preprocessing": "tokens EMB-NORMAL de S99-01; EOS; vocab "
                                 "congelado de C98-1 (92fecce6...)",
                "feature_groups": "n/a",
                "representation": "char-aware + word emb top-50k (MISMO vocab "
                                  "congelado)",
                "model": f"LMReal {dir_.upper()} (arquitectura congelada de C98-1)",
                "hyperparameters": jc({"lr": "1e-3 hasta 75.1% de pasos, luego 5e-4",
                                       "batch_tokens": 8192, "epocas": 2}),
                "validation": "mismos 3320 documentos de P98 (por fuente)",
                "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "",
                "recall": "",
                "training_time": f"{sum(h['segundos'] for h in hist)} s",
                "dimension": "",
                "result": jc({"val_ppl_por_epoca": [round(h.get("val_perplexity_corregida", h["val_perplexity"]), 1)
                                                    for h in hist],
                              "val_ppl_final": round(c["val_ppl"], 1),
                              "tokens_por_epoca": hist[0]["tokens"],
                              "checkpoint": c["ruta"],
                              "checkpoint_sha256": c["sha256"]}),
                "interpretation": "",
                "decision": f"ENTRENADO (S99-01): {dir_.upper()} val ppl "
                            f"{c['val_ppl']:.1f} en {c['epocas']} epoca(s).",
                "date": "2026-10-13",
                "script": "experiments/p99_11_preentrenar_s99_01.py",
                "notes": f"corpus sha {d['corpus']['sha256'][:16]} | GPU-h "
                         f"{d['gpu_horas_totales']} (tope 6) | json "
                         f"{sha16(pre)}",
            })
    for eid, ruta_json, etiqueta in (
            ("P99-13", RAIZ / "reports" / "p99_downstream_canonica.json",
             "canonica"),
            ("P99-14", RAIZ / "reports" / "p99_downstream_part47.json",
             "particion 47")):
        if eid in ids or not ruta_json.exists(): continue
        d = json.loads(ruta_json.read_text(encoding="utf-8"))
        nuevas.append({
            "experiment_id": eid, "competition": "KAGGLE_TOXIC_2026_PUJ",
            "question": f"60M vs 30M vs V03 en la {etiqueta}: hay mejora "
                        f"downstream?",
            "hypothesis": "H99-01 (gate de escala)",
            "preprocessing": "identico a D98-1 (reps congeladas + fusion v5.1)",
            "feature_groups": "bloque denso 2048",
            "representation": "reps S99-01 y P98 (congeladas)",
            "model": "LR por pliegue (fusion C=2; emb C=1)",
            "hyperparameters": jc({"umbrales": "V03 congelados"}),
            "validation": f"pliegues {etiqueta}",
            "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "",
            "recall": "", "training_time": f"{d.get('tiempo_total_s', 0)} s",
            "dimension": "2048",
            "result": jc(d["comparacion"]),
            "interpretation": "",
            "decision": d["gate"]["veredicto"],
            "date": "2026-10-13",
            "script": "experiments/p99_12_downstream_s99_01.py",
            "notes": f"sha16 {sha16(ruta_json)}",
        })
    if nuevas:
        anexar(nuevas)
    n1 = len(leer())
    print(f"P99-90: registradas {len(nuevas)} filas nuevas | total {n1}")
if __name__ == "__main__":
    main()
