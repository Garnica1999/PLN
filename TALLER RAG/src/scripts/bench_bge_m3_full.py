"""Evalúa bge-m3 sobre el corpus completo y compara con e5 dense y BM25 sparse."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

from src.scripts.metrics import row_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver", type=Path, default=Path("eval/silver/silver_standard.jsonl"))
    parser.add_argument("--collection", default="rag_full_bge_m3_d1024_v1")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--outdir", type=Path, default=Path("reports/tablas"))
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    silver = [json.loads(line) for line in args.silver.read_text(encoding="utf-8").splitlines() if line.strip()]
    silver = [row for row in silver if row.get("reference_doc_ids")]
    model = SentenceTransformer(args.model, device="cuda")
    model.half()
    client = MilvusClient(uri="http://localhost:19530")

    results = []
    for question in silver:
        for rep in range(1, args.reps + 1):
            vector = model.encode([question["question"]], normalize_embeddings=True)[0]
            response = client.search(
                collection_name=args.collection,
                data=[vector.tolist()],
                anns_field="vector",
                limit=10,
                output_fields=["chunk_id", "document_id", "text"],
            )
            hits = [
                {
                    "chunk_id": hit.get("chunk_id") or hit.get("id"),
                    "document_id": (hit.get("entity") or {}).get("document_id"),
                    "text": (hit.get("entity") or {}).get("text"),
                    "score": float(hit.get("distance", 0.0)),
                }
                for hit in response[0]
            ]
            record = {
                "run": "bge_m3_full",
                "question_id": question["id"],
                "split": question.get("split"),
                "model": args.model,
                "mode": "retrieve",
                "k": 10,
                "rep": rep,
                "collection": args.collection,
                "reference_doc_ids": question.get("reference_doc_ids"),
                "reference_span": question.get("reference_span"),
                "response": {"chunks": hits},
                "retrieval_ms": 0.0,
                "error": None,
            }
            results.append({**row_metrics(record), "lang": question.get("lang"), "split": question.get("split")})
    frame = pd.DataFrame(results)
    frame.to_csv(args.outdir / "bench_bge_m3_full_raw.csv", index=False)
    summary = frame.groupby("lang").agg(
        n=("question_id", "nunique"), recall=("recall", "mean"), ndcg=("ndcg", "mean"),
        p_at_1=("p_at_1", "mean"), p_at_5=("p_at_5", "mean"), span_hit=("span_hit", "mean"),
    ).reset_index()
    summary.to_csv(args.outdir / "bench_bge_m3_full.csv", index=False)
    print(summary.round(3).to_string(index=False))
    print("\nGlobal")
    print(frame[["recall", "ndcg", "p_at_1", "p_at_5", "span_hit"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
