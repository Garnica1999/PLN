"""Prueba de concepto: BM25 en memoria y fusión RRF con el dense sobre el corpus completo.

No reindexa Milvus. Descarga los chunks, construye el índice BM25 y compara
dense, BM25 y RRF con las mismas preguntas y el mismo gold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import bm25s
import pandas as pd

from src.rag.config import settings
from src.rag.store import get_client
from src.scripts.metrics import row_metrics


def load_chunks(collection: str, limit: int | None = None) -> list[dict]:
    client = get_client()
    rows: list[dict] = []
    iterator = client.query_iterator(
        collection_name=collection,
        batch_size=5000,
        filter="",
        output_fields=["chunk_id", "text", "document_id", "language", "position"],
    )
    while True:
        batch = iterator.next()
        if not batch:
            break
        rows.extend(batch)
        if limit and len(rows) >= limit:
            rows = rows[:limit]
            break
        if len(rows) % 50000 < 5000:
            print(f"  {len(rows)} chunks", flush=True)
    iterator.close()
    return rows


def load_dense(path: Path) -> dict[tuple[str, int], list[dict]]:
    dense = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        dense[(record["question_id"], int(record["rep"]))] = record["response"]["chunks"]
    return dense


def rrf(dense_ids: list[str], bm25_ids: list[str], k: int = 60, top: int = 10) -> list[str]:
    scores: dict[str, float] = {}
    for rank, chunk_id in enumerate(dense_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
    for rank, chunk_id in enumerate(bm25_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)[:top]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=Path("eval/gold/silver_standard.jsonl"))
    parser.add_argument("--dense", type=Path, default=Path("reports/corridas/bench_ef200_k10.jsonl"))
    parser.add_argument("--collection", default=settings.collection)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("reports/tablas"))
    args = parser.parse_args()

    gold = [json.loads(line) for line in args.gold.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold = [row for row in gold if row.get("reference_doc_ids")]
    dense = load_dense(args.dense)

    print(f"Descargando chunks de {args.collection}", flush=True)
    rows = load_chunks(args.collection, args.limit)
    print(f"{len(rows)} chunks", flush=True)
    texts = [row["text"] for row in rows]
    print("Tokenizando", flush=True)
    corpus_tokens = bm25s.tokenize(texts, stopwords=None, stemmer=None, return_ids=True, show_progress=False)
    retriever = bm25s.BM25()
    print("Indexando", flush=True)
    retriever.index(corpus_tokens)

    results = []
    for question in gold:
        for rep in (1, 2, 3):
            dense_hits = dense.get((question["id"], rep), [])
            query_tokens = bm25s.tokenize([question["question"]], stopwords=None, return_ids=True, show_progress=False)
            indices, _ = retriever.retrieve(query_tokens, k=10, show_progress=False)
            bm25_hits = [rows[int(index)] for index in indices[0]]
            fused_ids = rrf([hit["chunk_id"] for hit in dense_hits], [hit["chunk_id"] for hit in bm25_hits])
            by_id = {hit["chunk_id"]: hit for hit in dense_hits + bm25_hits}
            fused_hits = [by_id[chunk_id] for chunk_id in fused_ids if chunk_id in by_id]
            for method, hits in (("bm25", bm25_hits), ("rrf", fused_hits)):
                record = {
                    "run": f"bench_{method}",
                    "question_id": question["id"],
                    "split": question.get("split"),
                    "model": "bm25",
                    "mode": "retrieve",
                    "k": 10,
                    "rep": rep,
                    "collection": args.collection,
                    "reference_doc_ids": question.get("reference_doc_ids"),
                    "reference_span": question.get("reference_span"),
                    "response": {"chunks": hits},
                    "retrieval_ms": 0.0,
                    "error": None,
                    "lang": question.get("lang"),
                    "source": question.get("source"),
                    "type": question.get("type"),
                }
                results.append({**row_metrics(record), "method": method, "lang": record["lang"], "split": record["split"]})
    frame = pd.DataFrame(results)
    frame.to_csv(args.outdir / "bench_bm25_raw.csv", index=False)
    summary = frame.groupby(["method", "lang"]).agg(
        n=("question_id", "nunique"),
        recall=("recall", "mean"),
        ndcg=("ndcg", "mean"),
        p_at_1=("p_at_1", "mean"),
        span_hit=("span_hit", "mean"),
    ).reset_index()
    summary.to_csv(args.outdir / "bench_bm25.csv", index=False)
    print(summary.round(3).to_string(index=False))

    dense_path = args.outdir / "e6_breakdown_raw.csv"
    if dense_path.exists():
        dense_frame = pd.read_csv(dense_path)
        dense_summary = dense_frame.groupby("lang").agg(
            n=("question_id", "nunique"), recall=("recall", "mean"), ndcg=("ndcg", "mean"),
            p_at_1=("p_at_1", "mean"), span_hit=("span_hit", "mean"),
        ).reset_index()
    print("\nDense, referencia")
    print(dense_summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
