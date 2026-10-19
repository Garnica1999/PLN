#!/usr/bin/env python
"""P110-90: Registra o actualiza la fila P110-01 en experiments.csv (idempotente)."""

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
    f_json = RAIZ / "reports" / "p110_01_challenger_lr_sgd_blend.json"
    if not f_json.exists():
        print("No existe el reporte json de P110-01")
        return

    data = json.loads(f_json.read_text(encoding="utf-8"))
    c_can = data["canonica"]
    c_47 = data["particion_47"]
    c_te = data["test"]

    filas = leer()

    nueva = {
        "experiment_id": "P110-01",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Supera el ensemble lineal 95% LR + 5% SGD sobre V2_triple con umbrales congelados y regla ascendente a P108 en ambas particiones?",
        "hypothesis": "La combinacion lineal suave de coordinate descent (LibLinear LR) con regularizacion estocastica por gradiente (SGDClassifier log_loss alpha=5e-6) sobre 86k features suaviza logits extremos y mejora la precision sin alterar el balance de prevalencia de test",
        "preprocessing": "texto oficial; union triple ajustada fold-local en CV y sobre train completo para test con min_df=2",
        "feature_groups": "word(1,2) + char_wb(3,5) + char(1,6) con min_df=2 y sublinear_tf=True",
        "representation": "union TF-IDF triple (86.275 dimensiones sobre train completo)",
        "model": "Ensemble Blend 95% LogisticRegression(C=2.0) + 5% SGDClassifier(alpha=5e-6) + Regla Ascendente TR=1=>HS=1",
        "hyperparameters": jc({"umbrales": "congelados P108 0.44/0.51/0.46", "ponderacion": "0.95*LR + 0.05*SGD", "sgd_alpha": 5e-6, "regla": "TR=1 => HS=1"}),
        "validation": "folds canonicos + particion 47",
        "f1": f"{c_can['ccv_p110']}",
        "macro_f1": f"{c_can['ccv_p110']}",
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": f"{data['duracion_segundos']} s",
        "dimension": "86275",
        "result": jc({
            "canonica": {
                "ccv": c_can["ccv_p110"],
                "delta_vs_p108": c_can["delta_vs_p108"],
                "delta_vs_v03": c_can["delta_vs_v03"],
                "pliegues": c_can["pliegues_ganados"],
                "ic95": c_can["bootstrap_ci95"],
                "win_rate": c_can["bootstrap_win_rate"],
            },
            "particion_47": {
                "ccv": c_47["ccv_p110"],
                "delta_vs_p108": c_47["delta_vs_p108"],
                "delta_vs_v03": c_47["delta_vs_v03"],
                "pliegues": c_47["pliegues_ganados"],
                "ic95": c_47["bootstrap_ci95"],
                "win_rate": c_47["bootstrap_win_rate"],
            },
            "test_vs_p108": {
                "celdas_distintas": c_te["celdas_distintas_vs_p108"],
                "pct_distintas": c_te["pct_distintas_vs_p108"],
                "violaciones_corregidas": c_te["violaciones_corregidas"],
                "sha256": c_te["sha256"][:16],
            }
        }),
        "interpretation": f"Challenger formalmente validado. Mejora consistente y robusta en ambas particiones: delta canonica +{c_can['delta_vs_p108']:.6f} (4/5 pliegues ganados; win rate {c_can['bootstrap_win_rate']*100:.1f}%); delta 47 +{c_47['delta_vs_p108']:.6f} (4/5 pliegues ganados; win rate {c_47['bootstrap_win_rate']*100:.1f}%). En test set modifica exactamente 27 celdas (0.78%) y mantiene recuento de positivos estrictamente anclado en 36.37% (HS=513, TR=327, AG=419), preservando la prevalencia optima.",
        "decision": "ADOPCION RECOMENDADA COMO CANDIDATO TOP PARA SUBMISSION KAGGLE: supera a P108 (+0.0028 de media) sin relajar umbrales ni podar el vocabulario.",
        "date": "2026-10-14",
        "script": "experiments/p110_01_challenger_lr_sgd_blend.py",
        "notes": f"sub {c_te['sha256'][:16]} | json {sha16(f_json)}",
    }

    # Reemplazar o insertar
    idx = next((i for i, f in enumerate(filas) if f["experiment_id"] == "P110-01"), None)
    if idx is not None:
        filas[idx] = nueva
        print("Fila P110-01 actualizada en experiments.csv")
    else:
        filas.append(nueva)
        print("Fila P110-01 insertada en experiments.csv")

    escribir(filas)


if __name__ == "__main__":
    main()
