#!/usr/bin/env python
"""P98-94: registra P98-15 (D98-1b replica) en experiments.csv."""
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
    if any(f["experiment_id"] == "P98-15" for f in filas):
        raise SystemExit("ya existe P98-15")
    n0 = len(filas)
    d = json.load(open(RAIZ / "reports" / "p98_14_replica_d98_1b.json", encoding="utf-8"))
    r = d["reglas"]; c = d["comparacion"]
    s = sha16(RAIZ / "experiments" / "p98_14_replica_d98_1b.py")
    fila = {
        "experiment_id": "P98-15", "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "La senal de la fusion contextual de D98-1 sobrevive al "
                    "cambio de particion agrupada (seed 43)?",
        "hypothesis": "D98-1b: replica de robustez con todo congelado (reps, "
                      "fusion, umbrales); pre-registro de folds previo",
        "preprocessing": "normalizar_emb_normal + tokenizar (identico a D98-1)",
        "feature_groups": "bloque denso 2048 (congelado)",
        "representation": "reps C98-1 congeladas (LTR+RTL)",
        "model": "V03-equivalente (TF-IDF+LR C=2) y fusion (TF-IDF fold + denso)",
        "hyperparameters": jc({"C": 2.0, "umbrales": "V03 congelados",
                               "semilla_particion": 43, "bootstrap": 20260925}),
        "validation": "StratifiedGroupKFold(5, seed 43) sobre dev; sin fuga de "
                      "grupos; pre-registro p98_14_preregistro.json",
        "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "", "recall": "",
        "training_time": f"{d['tiempo_total_s']} s",
        "dimension": "2048",
        "result": jc({"v03_equivalente": round(r["v03_equivalente_part47"]["corrected_cv"], 10),
                      "fusion": round(r["fusion_contextual_part47"]["corrected_cv"], 10),
                      "delta": c["delta_pooled"],
                      "pliegues_positivos": c["pliegues_positivos"],
                      "ic95_clusters": c["bootstrap_clusters"]["ic95"],
                      "pct_mejora": c["bootstrap_clusters"]["pct_mejora"],
                      "neto_celdas": c["neto_celdas"],
                      "referencia_canonica": c["referencia_canonica_D98_1"]}),
        "interpretation": "",
        "decision": "FALSIFICADA: la senal canonica (+0.005055, 4/5) no "
                    "sobrevive a la particion 47 (+0.001151, 2/5 pliegues, neto "
                    "+2, IC95 [-0.007491,+0.009919]). Por los criterios "
                    "pre-registrados del propietario se CIERRA la linea "
                    "contextual actual: sin escalar corpus, sin D98-2, sin mas "
                    "replicas. Consistente con el freno historico (deltas < "
                    "+0.02 no transfieren).",
        "date": "2026-10-13",
        "script": "experiments/p98_14_replica_d98_1b.py",
        "notes": f"pre-registro {sha16(RAIZ/'reports'/'p98_14_preregistro.json')} | "
                 f"oof sha {sha16(RAIZ/'features'/'p98_14_oof_b.npz')} | "
                 f"script sha16 {s} | json {sha16(RAIZ/'reports'/'p98_14_replica_d98_1b.json')}",
    }
    with open(RUTA, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS, lineterminator="\n")
        w.writerow(fila)
    with open(RUTA, encoding="utf-8-sig", newline="") as fh:
        n1 = len(list(csv.DictReader(fh)))
    print(f"P98-94: {n0} -> {n1} filas")
if __name__ == "__main__":
    main()
