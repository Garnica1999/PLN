"""Benchmark de fusión dense + BM25 con pesos y RRF sobre top-50.

Compara dense solo, BM25 solo y tres pesos de fusión, en el gold original y en
un conjunto de preguntas parafraseadas para medir el sobreajuste al enunciado.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import bm25s
import pandas as pd

from src.rag.config import settings
from src.rag.pipeline import retrieve
from src.scripts.bench_bm25 import load_chunks
from src.scripts.metrics import row_metrics

WEIGHTS = [(1.0, 0.0), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.0, 1.0)]


def weighted_rrf(dense_ids: list[str], bm25_ids: list[str], w_dense: float, w_bm25: float, k: int = 60, top: int = 10) -> list[str]:
    scores: dict[str, float] = {}
    for rank, chunk_id in enumerate(dense_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + w_dense / (k + rank)
    for rank, chunk_id in enumerate(bm25_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + w_bm25 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)[:top]


def load_dense(path: Path) -> dict[tuple[str, int], list[dict]]:
    dense = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        dense[(record["question_id"], int(record["rep"]))] = record["response"]["chunks"]
    return dense


def evaluate(
    gold: list[dict],
    dense: dict[tuple[str, int], list[dict]],
    rows: list[dict],
    retriever,
    collection: str,
) -> pd.DataFrame:
    results = []
    for question in gold:
        for rep in (1, 2, 3):
            dense_hits = dense.get((question["id"], rep))
            if dense_hits is None:
                dense_hits, _ = retrieve(question["question"], k=50, collection=collection)
            query_tokens = bm25s.tokenize([question["question"]], stopwords=None, return_ids=True, show_progress=False)
            indices, _ = retriever.retrieve(query_tokens, k=50, show_progress=False)
            bm25_hits = [rows[int(index)] for index in indices[0]]
            for w_dense, w_bm25 in WEIGHTS:
                if w_dense == 1.0:
                    hits = dense_hits[:10]
                elif w_bm25 == 1.0:
                    hits = bm25_hits[:10]
                else:
                    fused = weighted_rrf(
                        [hit["chunk_id"] for hit in dense_hits],
                        [hit["chunk_id"] for hit in bm25_hits],
                        w_dense,
                        w_bm25,
                    )
                    by_id = {hit["chunk_id"]: hit for hit in dense_hits + bm25_hits}
                    hits = [by_id[chunk_id] for chunk_id in fused if chunk_id in by_id]
                record = {
                    "run": f"hybrid_{w_dense}_{w_bm25}",
                    "question_id": question["id"],
                    "split": question.get("split"),
                    "model": "hybrid",
                    "mode": "retrieve",
                    "k": 10,
                    "rep": rep,
                    "collection": collection,
                    "reference_doc_ids": question.get("reference_doc_ids"),
                    "reference_span": question.get("reference_span"),
                    "response": {"chunks": hits},
                    "retrieval_ms": 0.0,
                    "error": None,
                    "lang": question.get("lang"),
                    "source": question.get("source"),
                }
                results.append(
                    {
                        **row_metrics(record),
                        "weight_dense": w_dense,
                        "weight_bm25": w_bm25,
                        "lang": record["lang"],
                        "split": record["split"],
                    }
                )
    return pd.DataFrame(results)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["weight_dense", "weight_bm25", "lang"]).agg(
        n=("question_id", "nunique"),
        recall=("recall", "mean"),
        ndcg=("ndcg", "mean"),
        p_at_1=("p_at_1", "mean"),
        span_hit=("span_hit", "mean"),
    ).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=Path("eval/gold/gold_standard.jsonl"))
    parser.add_argument("--paraphrases", type=Path, default=Path("eval/gold/paraphrases.jsonl"))
    parser.add_argument("--dense", type=Path, default=Path("reports/corridas/bench_ef200_k50.jsonl"))
    parser.add_argument("--collection", default=settings.collection)
    parser.add_argument("--outdir", type=Path, default=Path("reports/tablas"))
    args = parser.parse_args()

    gold = [json.loads(line) for line in args.gold.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold = [row for row in gold if row.get("reference_doc_ids")]
    dense = load_dense(args.dense)
    rows = load_chunks(args.collection)
    print(f"{len(rows)} chunks", flush=True)
    corpus_tokens = bm25s.tokenize([row["text"] for row in rows], stopwords=None, return_ids=True, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    print("BM25 indexado", flush=True)

    original = evaluate(gold, dense, rows, retriever, args.collection)
    original.to_csv(args.outdir / "bench_hybrid_raw.csv", index=False)
    summary = summarize(original)
    summary.to_csv(args.outdir / "bench_hybrid.csv", index=False)
    print("\nGold original")
    print(summary.round(3).to_string(index=False))

    if args.paraphrases.exists():
        paraphrases = [
            json.loads(line) for line in args.paraphrases.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        paraphrases = [row for row in paraphrases if row.get("reference_doc_ids")]
        para = evaluate(paraphrases, {}, rows, retriever, args.collection)
        para.to_csv(args.outdir / "bench_paraphrases_raw.csv", index=False)
        para_summary = summarize(para)
        para_summary.to_csv(args.outdir / "bench_paraphrases.csv", index=False)
        print("\nParafrasis")
        print(para_summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
