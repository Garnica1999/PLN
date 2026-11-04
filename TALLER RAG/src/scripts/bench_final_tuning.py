"""Ajuste final: ef de HNSW, hibrido bge-m3 + BM25 y reranker sobre bge-m3."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from pymilvus import MilvusClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.rag.pipeline import retrieve_sparse, weighted_rrf
from src.scripts.metrics import row_metrics

EFS = [200, 400, 800, 1600]
WEIGHTS = [(0.5, 0.5), (0.3, 0.7)]
FIELDS = ["chunk_id", "document_id", "text", "title", "url"]


def search_dense(client: MilvusClient, collection: str, vector, ef: int, limit: int = 50) -> list[dict]:
    response = client.search(
        collection_name=collection,
        data=[vector.tolist()],
        anns_field="vector",
        limit=limit,
        output_fields=FIELDS,
        search_params={"metric_type": "COSINE", "params": {"ef": ef}},
    )
    return [
        {
            "chunk_id": hit.get("chunk_id") or hit.get("id"),
            "document_id": (hit.get("entity") or {}).get("document_id"),
            "text": (hit.get("entity") or {}).get("text"),
            "title": (hit.get("entity") or {}).get("title"),
            "url": (hit.get("entity") or {}).get("url"),
            "score": float(hit.get("distance", 0.0)),
        }
        for hit in response[0]
    ]


def metrics(hits: list[dict], question: dict, variant: str, ef: int, elapsed: float) -> dict:
    record = {
        "run": variant,
        "question_id": question["id"],
        "split": question.get("split"),
        "model": variant,
        "mode": "retrieve",
        "k": 10,
        "rep": 1,
        "collection": "rag_full",
        "reference_doc_ids": question.get("reference_doc_ids"),
        "reference_span": question.get("reference_span"),
        "response": {"chunks": hits[:10]},
        "retrieval_ms": elapsed,
        "error": None,
    }
    return {**row_metrics(record), "variant": variant, "ef": ef, "lang": question.get("lang"), "ms": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=Path("eval/gold/silver_standard.jsonl"))
    parser.add_argument("--collection", default="rag_full_bge_m3_d1024_v1")
    parser.add_argument("--outdir", type=Path, default=Path("reports/tablas"))
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    gold = [json.loads(line) for line in args.gold.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold = [row for row in gold if row.get("reference_doc_ids")]
    embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")
    embedder.half()
    reranker = CrossEncoder("BAAI/bge-reranker-base", device="cuda")
    client = MilvusClient(uri="http://localhost:19530")

    results = []
    for question in gold:
        for _ in range(args.reps):
            vector = embedder.encode([question["question"]], normalize_embeddings=True)[0]
            dense = {}
            dense_ms = {}
            for ef in EFS:
                started = time.perf_counter()
                hits = search_dense(client, args.collection, vector, ef)
                elapsed = (time.perf_counter() - started) * 1000
                dense[ef] = hits
                dense_ms[ef] = elapsed
                results.append(metrics(hits, question, f"dense_ef{ef}", ef, elapsed))
            started = time.perf_counter()
            sparse, _ = retrieve_sparse(question["question"], k=50)
            sparse_ms = (time.perf_counter() - started) * 1000
            for ef in (200, 800):
                for w_dense, w_sparse in WEIGHTS:
                    started = time.perf_counter()
                    fused = weighted_rrf(dense[ef], sparse, w_dense, w_sparse, top=10)
                    elapsed = dense_ms[ef] + sparse_ms + (time.perf_counter() - started) * 1000
                    results.append(metrics(fused, question, f"hybrid_ef{ef}_{w_dense}_{w_sparse}", ef, elapsed))
            started = time.perf_counter()
            scores = reranker.predict([(question["question"], hit["text"]) for hit in dense[800]], batch_size=32)
            rerank_ms = (time.perf_counter() - started) * 1000
            order = sorted(range(len(dense[800])), key=lambda i: -scores[i])[:10]
            reranked = [dense[800][i] for i in order]
            results.append(metrics(reranked, question, "rerank_ef800", 800, rerank_ms))

    frame = pd.DataFrame(results)
    frame.to_csv(args.outdir / "bench_final_tuning_raw.csv", index=False)
    summary = frame.groupby(["variant", "lang"]).agg(
        n=("question_id", "nunique"), recall=("recall", "mean"), ndcg=("ndcg", "mean"),
        p_at_1=("p_at_1", "mean"), p_at_5=("p_at_5", "mean"), span_hit=("span_hit", "mean"),
        p50_ms=("ms", "median"),
    ).reset_index()
    summary.to_csv(args.outdir / "bench_final_tuning.csv", index=False)
    global_summary = frame.groupby("variant").agg(
        recall=("recall", "mean"), ndcg=("ndcg", "mean"), p_at_1=("p_at_1", "mean"),
        p_at_5=("p_at_5", "mean"), span_hit=("span_hit", "mean"), p50_ms=("ms", "median"),
    ).reset_index().sort_values("recall", ascending=False)
    print(global_summary.round(4).to_string(index=False))
    print("\nPor idioma")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
