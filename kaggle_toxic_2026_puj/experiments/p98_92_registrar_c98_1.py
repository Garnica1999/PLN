#!/usr/bin/env python
"""P98-92: registra P98-12 (LTR) y P98-13 (RTL) en experiments.csv."""
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
def main():
    with open(RUTA, encoding="utf-8-sig", newline="") as fh:
        filas = list(csv.DictReader(fh))
    if any(f["experiment_id"] in ("P98-12", "P98-13") for f in filas):
        raise SystemExit("ya existen P98-12/13")
    n0 = len(filas)
    d = json.load(open(RAIZ / "reports" / "p98_c98_1_pretraining.json", encoding="utf-8"))
    s = sha16(RAIZ / "experiments" / "p98_12_preentrenar_c98_1.py")
    nuevas = []
    for eid, direccion in (("P98-12", "ltr"), ("P98-13", "rtl")):
        c = d["checkpoints"][direccion]
        hist = d["direcciones"][direccion]["historial"]
        nuevas.append({
            "experiment_id": eid, "competition": "KAGGLE_TOXIC_2026_PUJ",
            "question": f"Preentrenamiento {direccion.upper()} del LM contextual "
                        f"desde cero sobre C98-1 (30M tokens): converge?",
            "hypothesis": "C98-1: unidad de entrenamiento del encoder contextual "
                          "(dos direcciones independientes)",
            "preprocessing": "tokens EMB-NORMAL de C98-1; EOS entre documentos",
            "feature_groups": "no aplica",
            "representation": "char-aware + word emb top-50k (desde cero)",
            "model": f"LMReal {direccion.upper()} (LSTM 1024x2 + softmax atado)",
            "hyperparameters": jc({"lr": 0.001, "lr_desde_ep4": 0.0005,
                                   "batch_tokens": 8192, "seq_len": 96,
                                   "clip": 5.0, "dropout": 0.2}),
            "validation": "3320 documentos de validacion (2% por fuente)",
            "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "",
            "recall": "",
            "training_time": f"{sum(h['segundos'] for h in hist)} s",
            "dimension": "", "result": jc({
                "val_ppl_por_epoca": [round(h["val_perplexity"], 1) for h in hist],
                "val_ppl_final": round(c["val_ppl"], 1),
                "epocas": c["epocas"],
                "tokens_por_epoca": hist[0]["tokens"],
                "tok_s_medio": round(sum(h["tokens_por_segundo"] for h in hist) / len(hist), 1),
                "checkpoint": c["ruta"], "checkpoint_sha256": c["sha256"],
                "cobertura_top50k": d["vocab"]["cobertura_top50k"]}),
            "interpretation": "",
            "decision": f"ENTRENADO: {direccion.upper()} val ppl "
                        f"{c['val_ppl']:.1f} en {c['epocas']} epocas "
                        f"(curva monotona); la perplexity no es criterio de "
                        f"adopcion (P98 seccion 10).",
            "date": "2026-10-13",
            "script": "experiments/p98_12_preentrenar_c98_1.py",
            "notes": f"C98-1 | corpus sha {d['corpus']['sha256'][:16]} | "
                     f"GPU-h {d['gpu_horas_totales_c98_1']} (tope 6) | "
                     f"script sha16 {s} | json "
                     f"{sha16(RAIZ/'reports'/'p98_c98_1_pretraining.json')}",
        })
    with open(RUTA, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS, lineterminator="\n")
        for f in nuevas: w.writerow(f)
    with open(RUTA, encoding="utf-8-sig", newline="") as fh:
        n1 = len(list(csv.DictReader(fh)))
    print(f"P98-92: {n0} -> {n1} filas")
if __name__ == "__main__":
    main()
