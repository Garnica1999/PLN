"""E7: recupera top-N con e5 y reordena con un cross-encoder."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from sentence_transformers import CrossEncoder

from src.rag.embeddings import get_embedder
from src.rag.pipeline import dedupe
from src.rag.store import get_client, search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="eval/gold/gold_standard.jsonl")
    parser.add_argument("--split", default="all", choices=["dev", "test", "all"])
    parser.add_argument("--collection", default="rag_full_e5base_d768_v2_v1")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--device", default=os.getenv("EMBEDDING_DEVICE", "cpu"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--run", default="e7_rerank_full")
    parser.add_argument("--outdir", default="reports/corridas")
    args = parser.parse_args()

    questions = [
        json.loads(line)
        for line in Path(args.gold).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    questions = [
        q
        for q in questions
        if (args.split == "all" or q.get("split") == args.split)
        and not q.get("excluded")
        and q.get("reference_doc_ids")
    ]
    if not questions:
        raise SystemExit("no hay preguntas respondibles para ese split")

    embedder = get_embedder()
    client = get_client()
    reranker = CrossEncoder(args.model, device=args.device, max_length=args.max_length)
    out = Path(args.outdir) / f"{args.run}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out.open("w", encoding="utf-8") as handle:
        for question in questions:
            for rep in range(1, args.reps + 1):
                started = time.perf_counter()
                vector = embedder.embed_queries([question["question"]])[0]
                hits = dedupe(search(client, args.collection, vector, args.top_n))
                search_ms = (time.perf_counter() - started) * 1000
                rerank_started = time.perf_counter()
                if hits:
                    scores = reranker.predict(
                        [(question["question"], hit["text"]) for hit in hits],
                        batch_size=args.batch_size,
                        show_progress_bar=False,
                    )
                    order = sorted(range(len(hits)), key=lambda index: scores[index], reverse=True)[: args.k]
                    top = [hits[index] for index in order]
                else:
                    top = []
                rerank_ms = (time.perf_counter() - rerank_started) * 1000
                total_ms = (time.perf_counter() - started) * 1000
                record = {
                    "run": args.run,
                    "question_id": question["id"],
                    "split": args.split,
                    "model": f"e5+{args.model.split('/')[-1]}",
                    "rep": rep,
                    "mode": "retrieve",
                    "k": args.k,
                    "top_n": args.top_n,
                    "collection": args.collection,
                    "reference_doc_ids": question.get("reference_doc_ids", []),
                    "reference_chunk_ids": question.get("reference_chunk_ids", []),
                    "reference_span": question.get("reference_span", ""),
                    "response": {
                        "chunks": top,
                        "retrieval_ms": round(total_ms, 2),
                        "search_ms": round(search_ms, 2),
                        "rerank_ms": round(rerank_ms, 2),
                    },
                    "error": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                written += 1
            print(f"{question['id']} listo ({written})", flush=True)

    print(json.dumps({"run": args.run, "registros": written, "salida": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
