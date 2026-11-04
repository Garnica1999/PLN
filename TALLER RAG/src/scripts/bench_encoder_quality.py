"""Compara encoders en el piloto: calidad de recuperación y throughput.

Codifica los chunks del piloto y las consultas, calcula similitud coseno y mide
Recall, nDCG y span_hit con el gold original y las paráfrasis. No reindexa.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src.rag.store import get_client
from src.scripts.metrics import row_metrics


def load_chunks(collection: str) -> list[dict]:
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
    iterator.close()
    return rows


def prefixes(model: str) -> tuple[str, str]:
    if "e5" in model.lower():
        return "query: ", "passage: "
    return "", ""


def encode(model_name: str, rows: list[dict], queries: list[str], device: str, half: bool = False, batch: int = 128) -> tuple[np.ndarray, np.ndarray, float]:
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    if half:
        model.half()
    print(f"  device={model.device}", flush=True)
    q_prefix, p_prefix = prefixes(model_name)
    started = time.perf_counter()
    passages = model.encode(
        [p_prefix + row["text"] for row in rows],
        normalize_embeddings=True,
        batch_size=batch,
        show_progress_bar=False,
    )
    elapsed = time.perf_counter() - started
    vectors = model.encode(
        [q_prefix + query for query in queries],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )
    return passages, vectors, len(rows) / elapsed


def evaluate(
    gold: list[dict],
    rows: list[dict],
    passages: np.ndarray,
    vectors: np.ndarray,
    model_name: str,
    dataset: str,
    offset: int = 0,
) -> pd.DataFrame:
    results = []
    for index, question in enumerate(gold):
        scores = passages @ vectors[offset + index]
        top = np.argsort(-scores)[:10]
        hits = [
            {
                "chunk_id": rows[int(i)]["chunk_id"],
                "document_id": rows[int(i)]["document_id"],
                "text": rows[int(i)]["text"],
                "score": float(scores[int(i)]),
            }
            for i in top
        ]
        record = {
            "run": f"{model_name}_{dataset}",
            "question_id": question["id"],
            "split": question.get("split"),
            "model": model_name,
            "mode": "retrieve",
            "k": 10,
            "rep": 1,
            "collection": "pilot",
            "reference_doc_ids": question.get("reference_doc_ids"),
            "reference_span": question.get("reference_span"),
            "response": {"chunks": hits},
            "retrieval_ms": 0.0,
            "error": None,
        }
        results.append({**row_metrics(record), "lang": question.get("lang"), "dataset": dataset})
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="rag_pilot_e5base_d768_v2_v1")
    parser.add_argument("--gold", type=Path, default=Path("eval/gold/silver_standard.jsonl"))
    parser.add_argument("--paraphrases", type=Path, default=Path("eval/gold/paraphrases.jsonl"))
    parser.add_argument("--models", default="intfloat/multilingual-e5-base,BAAI/bge-m3,Snowflake/snowflake-arctic-embed-m-v2.0,Alibaba-NLP/gte-multilingual-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--outdir", type=Path, default=Path("reports/tablas"))
    args = parser.parse_args()

    gold = [json.loads(line) for line in args.gold.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold = [row for row in gold if row.get("reference_doc_ids")]
    paraphrases = [
        json.loads(line) for line in args.paraphrases.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    paraphrases = [row for row in paraphrases if row.get("reference_doc_ids")]
    rows = load_chunks(args.collection)
    print(f"{len(rows)} chunks del piloto", flush=True)

    frames = []
    for model_name in [name.strip() for name in args.models.split(",") if name.strip()]:
        print(f"Codificando {model_name}", flush=True)
        queries = [row["question"] for row in gold + paraphrases]
        passages, vectors, throughput = encode(model_name, rows, queries, args.device, args.half, args.batch)
        print(f"  {throughput:.1f} chunks/s", flush=True)
        frames.append(evaluate(gold, rows, passages, vectors, model_name, "gold"))
        frames.append(evaluate(paraphrases, rows, passages, vectors, model_name, "paraphrases", offset=len(gold)))
        frames[-1]["throughput"] = throughput
        frames[-2]["throughput"] = throughput
        del passages, vectors
        torch.cuda.empty_cache()

    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(args.outdir / "bench_encoder_quality_raw.csv", index=False)
    summary = frame.groupby(["model", "dataset", "lang"]).agg(
        n=("question_id", "nunique"),
        recall=("recall", "mean"),
        ndcg=("ndcg", "mean"),
        p_at_1=("p_at_1", "mean"),
        span_hit=("span_hit", "mean"),
        chunks_por_segundo=("throughput", "max"),
    ).reset_index()
    summary.to_csv(args.outdir / "bench_encoder_quality.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
