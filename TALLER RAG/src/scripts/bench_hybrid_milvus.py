"""Evalúa dense, BM25 sparse e híbrido 0,3/0,7 en el corpus completo con Milvus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.rag.pipeline import retrieve, retrieve_sparse, weighted_rrf
from src.scripts.metrics import row_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=Path("eval/gold/silver_standard.jsonl"))
    parser.add_argument("--outdir", type=Path, default=Path("reports/tablas"))
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    gold = [json.loads(line) for line in args.gold.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold = [row for row in gold if row.get("reference_doc_ids")]
    results = []
    for question in gold:
        for rep in range(1, args.reps + 1):
            dense, dense_ms = retrieve(question["question"], k=50)
            sparse, sparse_ms = retrieve_sparse(question["question"], k=50)
            variants = (
                ("dense", dense[:10], dense_ms),
                ("sparse", sparse[:10], sparse_ms),
                ("hybrid", weighted_rrf(dense, sparse, 0.3, 0.7, top=10), dense_ms + sparse_ms),
            )
            for method, hits, elapsed in variants:
                record = {
                    "run": f"bench_milvus_{method}",
                    "question_id": question["id"],
                    "split": question.get("split"),
                    "model": method,
                    "mode": "retrieve",
                    "k": 10,
                    "rep": rep,
                    "collection": "rag_full",
                    "reference_doc_ids": question.get("reference_doc_ids"),
                    "reference_span": question.get("reference_span"),
                    "response": {"chunks": hits},
                    "retrieval_ms": elapsed,
                    "error": None,
                }
                results.append(
                    {
                        **row_metrics(record),
                        "method": method,
                        "lang": question.get("lang"),
                        "split": question.get("split"),
                        "retrieval_ms": elapsed,
                    }
                )
    frame = pd.DataFrame(results)
    frame.to_csv(args.outdir / "bench_milvus_raw.csv", index=False)
    summary = frame.groupby(["method", "lang"]).agg(
        n=("question_id", "nunique"),
        recall=("recall", "mean"),
        ndcg=("ndcg", "mean"),
        p_at_1=("p_at_1", "mean"),
        span_hit=("span_hit", "mean"),
        p50_ms=("retrieval_ms", "median"),
    ).reset_index()
    summary.to_csv(args.outdir / "bench_milvus.csv", index=False)
    print(summary.round(3).to_string(index=False))
    global_summary = frame.groupby("method").agg(
        n=("question_id", "nunique"), recall=("recall", "mean"), ndcg=("ndcg", "mean"),
        p_at_1=("p_at_1", "mean"), p50_ms=("retrieval_ms", "median"),
    ).reset_index()
    print("\nGlobal")
    print(global_summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
