"""Reranker pequeño sobre el top-50 de BM25 sparse, en GPU."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sentence_transformers import CrossEncoder

from src.rag.pipeline import retrieve_sparse
from src.scripts.metrics import row_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=Path("eval/gold/gold_standard.jsonl"))
    parser.add_argument("--model", default="BAAI/bge-reranker-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outdir", type=Path, default=Path("reports/tablas"))
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    gold = [json.loads(line) for line in args.gold.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold = [row for row in gold if row.get("reference_doc_ids")]
    model = CrossEncoder(args.model, device=args.device)
    print(f"device={model.device}", flush=True)

    results = []
    for question in gold:
        for rep in range(1, args.reps + 1):
            hits, sparse_ms = retrieve_sparse(question["question"], k=50)
            started = time.perf_counter()
            scores = model.predict([(question["question"], hit["text"]) for hit in hits], batch_size=32)
            rerank_ms = round((time.perf_counter() - started) * 1000, 2)
            order = sorted(range(len(hits)), key=lambda i: -scores[i])[:10]
            reranked = [{**hits[i], "rerank_score": float(scores[i])} for i in order]
            record = {
                "run": "sparse_rerank",
                "question_id": question["id"],
                "split": question.get("split"),
                "model": args.model,
                "mode": "retrieve",
                "k": 10,
                "rep": rep,
                "collection": "rag_full",
                "reference_doc_ids": question.get("reference_doc_ids"),
                "reference_span": question.get("reference_span"),
                "response": {"chunks": reranked},
                "retrieval_ms": sparse_ms,
                "error": None,
            }
            results.append(
                {
                    **row_metrics(record),
                    "lang": question.get("lang"),
                    "split": question.get("split"),
                    "sparse_ms": sparse_ms,
                    "rerank_ms": rerank_ms,
                    "total_ms": sparse_ms + rerank_ms,
                }
            )
    frame = pd.DataFrame(results)
    frame.to_csv(args.outdir / "bench_rerank_sparse_raw.csv", index=False)
    summary = frame.groupby("lang").agg(
        n=("question_id", "nunique"),
        recall=("recall", "mean"),
        ndcg=("ndcg", "mean"),
        p_at_1=("p_at_1", "mean"),
        p_at_5=("p_at_5", "mean"),
        span_hit=("span_hit", "mean"),
        rerank_p50=("rerank_ms", "median"),
        total_p50=("total_ms", "median"),
    ).reset_index()
    summary.to_csv(args.outdir / "bench_rerank_sparse.csv", index=False)
    print(summary.round(3).to_string(index=False))
    print("\nGlobal")
    print(
        frame[["recall", "ndcg", "p_at_1", "p_at_5", "span_hit", "rerank_ms", "total_ms"]]
        .mean()
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
