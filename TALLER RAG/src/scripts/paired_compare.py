"""Comparación pareada entre dos corridas de recuperación.

Sirve para E1 (chunking) y E2 (k). Lee dos JSONL, calcula métricas por pregunta
y el bootstrap pareado de la diferencia, con IC 95%.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.scripts.metrics import row_metrics

METRICS = ["hit", "recall", "p_at_1", "p_at_5", "mrr", "ndcg"]
BOOT = 1000


def load(path: str) -> pd.DataFrame:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(row_metrics(json.loads(line)))
    frame = pd.DataFrame(rows)
    frame["question_id"] = frame["question_id"].astype(str)
    return frame


def per_question(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("question_id", as_index=False)[METRICS + ["retrieval_ms"]].mean()


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(BOOT)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="Corrida base")
    parser.add_argument("--b", required=True, help="Corrida comparada")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--out", required=True, help="Prefijo de salida en reports/tablas")
    parser.add_argument("--figure", default=None)
    args = parser.parse_args()

    a = per_question(load(args.a))
    b = per_question(load(args.b))
    common = a.merge(b, on="question_id", suffixes=("_a", "_b"))
    if common.empty:
        raise SystemExit("las corridas no comparten preguntas")

    rng = np.random.default_rng(42)
    # Bonferroni: una comparacion prueba todas las metricas mas la latencia.
    alpha = 0.05 / (len(METRICS) + 1)
    rows = []
    for label, frame, suffix in ((args.label_a, common, "_a"), (args.label_b, common, "_b")):
        row = {"label": label, "n_preguntas": len(common)}
        for metric in METRICS + ["retrieval_ms"]:
            values = frame[f"{metric}{suffix}"].to_numpy(dtype=float)
            low, high = bootstrap_ci(values, rng)
            row[metric] = round(float(np.mean(values)), 4)
            row[f"{metric}_lo"] = round(low, 4)
            row[f"{metric}_hi"] = round(high, 4)
        rows.append(row)
    summary = pd.DataFrame(rows)
    Path(f"reports/tablas/{args.out}.csv").parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(f"reports/tablas/{args.out}.csv", index=False)

    paired = []
    for metric in METRICS + ["retrieval_ms"]:
        diff = (common[f"{metric}_b"] - common[f"{metric}_a"]).to_numpy(dtype=float)
        means = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(BOOT)]
        low = float(np.percentile(means, 100 * alpha / 2))
        high = float(np.percentile(means, 100 * (1 - alpha / 2)))
        paired.append(
            {
                "metric": metric,
                "diff": round(float(diff.mean()), 4),
                "ci_lo": round(low, 4),
                "ci_hi": round(high, 4),
                "significativo": bool(low > 0 or high < 0),
            }
        )
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(f"reports/tablas/{args.out}_paired.csv", index=False)

    if args.figure:
        metrics = ["p_at_1", "p_at_5", "recall", "ndcg"]
        positions = np.arange(len(metrics))
        fig, axis = plt.subplots(figsize=(7, 4))
        for offset, (label, suffix, color) in enumerate(
            ((args.label_a, "_a", "#335c81"), (args.label_b, "_b", "#8a5a44"))
        ):
            values = [common[f"{m}{suffix}"].mean() for m in metrics]
            axis.bar(positions + (offset - 0.5) * 0.36, values, width=0.36, label=label, color=color)
        axis.set_xticks(positions, metrics)
        axis.set_ylim(0, 1)
        axis.set_ylabel("Valor medio")
        axis.set_title(f"{args.label_a} vs {args.label_b}")
        axis.legend()
        fig.tight_layout()
        fig.savefig(args.figure, dpi=150)
        plt.close(fig)

    print(json.dumps({"preguntas": len(common), "significativos": int(paired_frame["significativo"].sum())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
