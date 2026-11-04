"""Métricas de recuperación, generación y rendimiento a partir de los JSONL crudos."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

RETRIEVAL_METRICS = [
    "hit",
    "recall",
    "p_at_1",
    "p_at_5",
    "mrr",
    "map",
    "ndcg",
    "chunk_recall",
    "span_hit",
    "context_precision",
    "context_recall",
]
CI_METRICS = ["recall", "mrr", "ndcg"]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def dcg(gains: list[int]) -> float:
    return sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    gains = [1 if doc in relevant else 0 for doc in retrieved[:k]]
    ideal = [1] * min(len(relevant), k)
    denominator = dcg(ideal)
    return dcg(gains) / denominator if denominator else 0.0


def average_precision(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for index, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant:
            hits += 1
            total += hits / index
    return total / min(len(relevant), k)


def ranked_items(record: dict) -> list[dict]:
    response = record.get("response") or {}
    items = response.get("chunks") if record.get("mode") == "retrieve" else response.get("evidence")
    return items or []


def ranked_docs(items: list[dict]) -> list[str]:
    seen: set[str] = set()
    ordered = []
    for item in items:
        doc = item.get("document_id")
        if doc and doc not in seen:
            seen.add(doc)
            ordered.append(doc)
    return ordered


def row_metrics(record: dict) -> dict:
    items = ranked_items(record)
    retrieved = ranked_docs(items)
    relevant = set(record.get("reference_doc_ids") or [])
    reference_chunks = set(record.get("reference_chunk_ids") or [])
    reference_span = normalize(record.get("reference_span") or "")
    k = int(record.get("k") or 5)
    top = retrieved[:k]

    first_hit = next((i for i, doc in enumerate(top, start=1) if doc in relevant), None)
    chunk_ids = [item.get("chunk_id") for item in items if item.get("chunk_id")]
    span_hit = bool(reference_span) and any(reference_span in normalize(item.get("text", "")) for item in items)
    context_docs = set(retrieved)
    context_relevant = sum(1 for item in items if item.get("document_id") in relevant)
    response = record.get("response") or {}

    return {
        "run": record.get("run"),
        "question_id": record.get("question_id"),
        "split": record.get("split"),
        "model": record.get("model") or "n/a",
        "mode": record.get("mode"),
        "k": k,
        "rep": record.get("rep"),
        "error": bool(record.get("error")),
        "has_refs": bool(relevant),
        "hit": int(first_hit is not None),
        "recall": len(relevant & set(top)) / len(relevant) if relevant else 0.0,
        "p_at_1": int(bool(top) and top[0] in relevant),
        "p_at_5": int(any(doc in relevant for doc in top[:5])),
        "mrr": 1.0 / first_hit if first_hit else 0.0,
        "map": average_precision(top, relevant, k),
        "ndcg": ndcg_at_k(top, relevant, k),
        "chunk_recall": len(reference_chunks & set(chunk_ids)) / len(reference_chunks) if reference_chunks else None,
        "span_hit": int(span_hit) if reference_span else None,
        "context_precision": context_relevant / len(items) if items else 0.0,
        "context_recall": len(relevant & context_docs) / len(relevant) if relevant else 0.0,
        "unique_docs": len(context_docs),
        "duplicate_rate": 1.0 - len(context_docs) / len(items) if items else 0.0,
        "retrieval_ms": response.get("retrieval_ms"),
        "generation_ms": response.get("generation_ms"),
        "total_ms": response.get("total_ms"),
        "tokens_per_second": response.get("tokens_per_second"),
        "abstained": bool(response.get("abstained")),
        "invalid_citations": len(response.get("invalid_citations") or []),
        "citations": len(response.get("citations") or []),
        "missing_citations": bool(response.get("missing_citations")),
    }


def bootstrap_ci(values: pd.Series, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    array = values.dropna().to_numpy(dtype=float)
    if len(array) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [rng.choice(array, size=len(array), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[(~frame["error"]) & frame["has_refs"]]
    rows = []
    for keys, group in frame.groupby(["run", "mode", "model", "k", "split"]):
        row = dict(zip(["run", "mode", "model", "k", "split"], keys))
        row["n_registros"] = len(group)
        row["n_preguntas"] = int(group["question_id"].nunique())
        for metric in RETRIEVAL_METRICS:
            row[metric] = float(group[metric].mean())
            if metric in CI_METRICS:
                low, high = bootstrap_ci(group[metric])
                row[f"{metric}_lo"] = low
                row[f"{metric}_hi"] = high
        for metric, func in [("retrieval_ms", 50), ("retrieval_ms", 95), ("generation_ms", 50), ("generation_ms", 95)]:
            series = group[metric].dropna()
            row[f"{metric.split('_')[0]}_p{func}"] = float(np.percentile(series, func)) if len(series) else float("nan")
        row["tokens_por_segundo"] = float(group["tokens_per_second"].mean())
        row["abstencion"] = float(group["abstained"].mean())
        row["citas_invalidas"] = float(group["invalid_citations"].mean())
        row["sin_citas"] = float(group["missing_citations"].mean())
        row["duplicados"] = float(group["duplicate_rate"].mean())
        rows.append(row)
    return pd.DataFrame(rows).round(4)


def to_markdown(summary: pd.DataFrame) -> str:
    columns = [
        "run",
        "mode",
        "model",
        "k",
        "split",
        "n_registros",
        "n_preguntas",
        "hit",
        "recall",
        "recall_lo",
        "recall_hi",
        "p_at_1",
        "p_at_5",
        "mrr",
        "map",
        "ndcg",
        "ndcg_lo",
        "ndcg_hi",
        "chunk_recall",
        "span_hit",
        "context_precision",
        "context_recall",
        "duplicados",
        "retrieval_p50",
        "generation_p50",
        "tokens_por_segundo",
        "abstencion",
        "citas_invalidas",
        "sin_citas",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, row in summary.iterrows():
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--annotations", nargs="*", default=[])
    parser.add_argument("--silver", default=None)
    parser.add_argument("--outdir", default="reports/tablas")
    args = parser.parse_args()

    silver = {}
    if args.silver:
        silver = {
            row["id"]: row
            for row in (
                json.loads(line)
                for line in Path(args.silver).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
    records = []
    for path in args.results:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("question_id") in silver:
                reference = silver[record["question_id"]]
                record["reference_span"] = reference.get("reference_span") or record.get("reference_span")
                record["reference_doc_ids"] = reference.get("reference_doc_ids") or record.get("reference_doc_ids")
            records.append(row_metrics(record))
    frame = pd.DataFrame(records)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / "resultados_detalle.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(outdir / "metricas.csv", index=False)
    (outdir / "metricas.md").write_text(to_markdown(summary), encoding="utf-8")

    if args.annotations:
        parts = [pd.read_csv(path) for path in args.annotations]
        annotations = pd.concat(parts, ignore_index=True)
        keys = [key for key in ["question_id", "model", "rep"] if key in annotations.columns]
        merged = frame.merge(annotations, on=keys, how="inner")
        if not merged.empty:
            rubric = (
                merged.groupby(["model"])[
                    [c for c in ["correccion", "fidelidad", "citas", "abstencion"] if c in merged.columns]
                ]
                .mean()
                .reset_index()
                .round(3)
            )
            rubric.to_csv(outdir / "rubrica.csv", index=False)

    print(json.dumps({"registros": len(frame), "resumen": len(summary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
