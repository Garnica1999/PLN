"""Métricas por variante de limpieza: desglose y bootstrap pareado contra V0."""
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

METRICS = ["hit", "recall", "p_at_1", "p_at_5", "mrr", "map", "ndcg"]
BOOT = 1000


def load_records(paths: list[str]) -> pd.DataFrame:
    rows = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(row_metrics(json.loads(line)))
    return pd.DataFrame(rows)


def load_silver(paths: list[str]) -> pd.DataFrame:
    rows = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if not record.get("excluded"):
                    rows.append(
                        {
                            "question_id": record["id"],
                            "lang": record.get("lang"),
                            "type": record.get("type"),
                            "source": record["reference_doc_ids"][0].split(":")[0] if record.get("reference_doc_ids") else "sin_respuesta",
                        }
                    )
    return pd.DataFrame(rows)


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int = BOOT) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap(variant: pd.Series, baseline: pd.Series, rng: np.random.Generator, alpha: float) -> tuple[float, float, float]:
    diff = (variant - baseline).dropna().to_numpy(dtype=float)
    if len(diff) == 0:
        return float("nan"), float("nan"), float("nan")
    means = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(BOOT)]
    return (
        float(diff.mean()),
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--silver", nargs="+", default=["eval/silver/questions_smoke.jsonl", "eval/silver/questions_dev.jsonl"])
    parser.add_argument("--outdir", default="reports/tablas")
    parser.add_argument("--figdir", default="reports/figuras")
    args = parser.parse_args()

    frame = load_records(args.results)
    silver = load_silver(args.silver)
    frame["variant"] = frame["run"].str.extract(r"(v[0-4])$")[0].fillna(frame["run"])
    merged = frame.merge(silver, on="question_id", how="left")
    outdir = Path(args.outdir)
    figdir = Path(args.figdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    per_question = merged.groupby(["variant", "question_id"], as_index=False)[METRICS].mean()
    rng = np.random.default_rng(42)

    summary_rows = []
    for variant, group in per_question.groupby("variant"):
        row = {"variant": variant, "n_preguntas": len(group)}
        for metric in METRICS:
            row[metric] = round(float(group[metric].mean()), 4)
            low, high = bootstrap_mean(group[metric].to_numpy(dtype=float), rng)
            row[f"{metric}_lo"] = round(low, 4)
            row[f"{metric}_hi"] = round(high, 4)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("variant")
    summary.to_csv(outdir / "limpieza_metricas.csv", index=False)

    breakdown_rows = []
    for dimension in ["lang", "source", "type"]:
        for (variant, value), group in merged.groupby(["variant", dimension]):
            breakdown_rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "variant": variant,
                    "n_registros": len(group),
                    **{metric: round(float(group[metric].mean()), 4) for metric in METRICS},
                }
            )
    pd.DataFrame(breakdown_rows).to_csv(outdir / "limpieza_breakdown.csv", index=False)

    baseline = per_question[per_question["variant"] == "v0"].set_index("question_id")
    n_variants = per_question["variant"].nunique()
    # Bonferroni sobre todas las pruebas de la tabla.
    alpha = 0.05 / (len(METRICS) * max(n_variants - 1, 1))
    paired_rows = []
    for variant in sorted(per_question["variant"].unique()):
        if variant == "v0":
            continue
        current = per_question[per_question["variant"] == variant].set_index("question_id")
        common = current.index.intersection(baseline.index)
        for metric in METRICS:
            diff, low, high = paired_bootstrap(current.loc[common, metric], baseline.loc[common, metric], rng, alpha)
            paired_rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "n": len(common),
                    "diff": round(diff, 4),
                    "ci_lo": round(low, 4),
                    "ci_hi": round(high, 4),
                    "significativo": bool(low > 0 or high < 0),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(outdir / "limpieza_paired.csv", index=False)

    fig, axis = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(summary))
    axis.bar(positions, summary["ndcg"], color="#335c81", label="nDCG@10")
    axis.errorbar(
        positions,
        summary["ndcg"],
        yerr=[summary["ndcg"] - summary["ndcg_lo"], summary["ndcg_hi"] - summary["ndcg"]],
        fmt="none",
        ecolor="#1b1b1b",
        capsize=4,
    )
    axis.set_xticks(positions, summary["variant"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("nDCG@10")
    axis.set_title("Limpieza del texto y recuperación")
    fig.tight_layout()
    fig.savefig(figdir / "limpieza_comparacion.png", dpi=150)
    plt.close(fig)

    print(json.dumps({"variantes": len(summary), "pares": len(paired)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
