#!/usr/bin/env python
"""P98-93: registra P98-14 (D98-1 downstream) en experiments.csv."""
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
    if any(f["experiment_id"] == "P98-14" for f in filas):
        raise SystemExit("ya existe P98-14")
    n0 = len(filas)
    d = json.load(open(RAIZ / "reports" / "p98_13_downstream.json", encoding="utf-8"))
    r = d["reglas"]; c = d["comparaciones"]
    s = sha16(RAIZ / "experiments" / "p98_13_downstream_d98_1.py")
    fila = {
        "experiment_id": "P98-14", "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Las representaciones contextuales congeladas de C98-1 "
                    "aportan senal downstream frente a V03?",
        "hypothesis": "D98-1: reps (media L2 LTR+RTL, 2048d) + LR por pliegue; "
                      "fusion estilo v5.1",
        "preprocessing": "normalizar_emb_normal + tokenizar (identico al corpus)",
        "feature_groups": "bloque denso 2048 (contextual)",
        "representation": "reps congeladas C98-1 (LTR+RTL)",
        "model": "LR binarias por etiqueta (emb: C=1; fusion: C=2, TF-IDF fold)",
        "hyperparameters": jc({"emb_C": 1.0, "fusion_C": 2.0, "umbrales": "V03 congelados"}),
        "validation": "StratifiedGroupKFold canonico (5 pliegues, grupos fusionados)",
        "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "", "recall": "",
        "training_time": f"{d['tiempo_total_s']} s",
        "dimension": "2048",
        "result": jc({"v03": round(r["v03_frozen"]["corrected_cv"], 10),
                      "embedding_only": round(r["embedding_only"]["corrected_cv"], 10),
                      "fusion": round(r["fusion_v51_style"]["corrected_cv"], 10),
                      "delta_fusion_vs_v03": c["fusion_vs_v03"]["delta"],
                      "pliegues_mejorados": c["fusion_vs_v03"]["pliegues_mejorados"],
                      "ic95_clusters": c["fusion_vs_v03"]["bootstrap_clusters"]["ic95"],
                      "pct_mejora": c["fusion_vs_v03"]["bootstrap_clusters"]["pct_mejora"],
                      "neto_celdas": r["fusion_v51_style"]["correccion_vs_v03"]["neto"],
                      "f1_por_etiqueta_fusion": {k: round(v, 4) for k, v in
                                                 r["fusion_v51_style"]["f1_por_etiqueta"].items()},
                      "referencia_estatica_v51": {"emb": 0.6523, "fusion": 0.7818},
                      "slices_clave": [f for f in d["slices"]
                                       if f["slice"] in ("longitud_q4", "longitud_q1",
                                                         "hashtag", "url")]}),
        "interpretation": "",
        "decision": "MARGINAL_SENAL: delta +0.005055 con 4/5 pliegues y neto +11, "
                    "pero IC95 de clusters [-0.004058,+0.014721] incluye cero y el "
                    "embedding-only (0.6522) no supera a la referencia estatica "
                    "v5.1 (0.6523); la fusion contextual (+0.0005 sobre la fusion "
                    "estatica) cae en la clase historica de +0.005 que nunca "
                    "transfirio. Candidato: replicar particion (D98-1b) y evaluar "
                    "D98-2 dentro del presupuesto (autorizado si la senal se "
                    "considera prometedora pero insuficiente).",
        "date": "2026-10-13",
        "script": "experiments/p98_13_downstream_d98_1.py",
        "notes": f"reps sha {sha16(RAIZ/'features'/'p98_13_reps_dev.npz')} | "
                 f"oof sha {sha16(RAIZ/'features'/'p98_13_oof.npz')} | "
                 f"script sha16 {s} | json "
                 f"{sha16(RAIZ/'reports'/'p98_13_downstream.json')}",
    }
    with open(RUTA, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS, lineterminator="\n")
        w.writerow(fila)
    with open(RUTA, encoding="utf-8-sig", newline="") as fh:
        n1 = len(list(csv.DictReader(fh)))
    print(f"P98-93: {n0} -> {n1} filas")
if __name__ == "__main__":
    main()
