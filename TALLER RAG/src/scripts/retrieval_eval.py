"""Evaluación de recuperación con un encoder explícito, sin pasar por la API.

Sirve para comparar encoders (E3) sin reiniciar el servicio ni cambiar el .env.
Escribe el mismo formato JSONL que consume metrics.py.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.rag.config import settings
from src.rag.embeddings import OllamaEmbedder, SentenceTransformerEmbedder
from src.rag.pipeline import dedupe
from src.rag.store import count, get_client, search


def build_embedder(name: str, url: str | None, dims: int | None):
    if name == "e5":
        return SentenceTransformerEmbedder()
    return OllamaEmbedder(
        url=url or settings.embeddings_url,
        dims=dims or settings.embeddings_dims,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver", default="eval/silver/questions_dev.jsonl")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "clean", "smoke", "all"])
    parser.add_argument("--collection", required=True)
    parser.add_argument("--embedder", choices=["e5", "ollama"], default="e5")
    parser.add_argument("--embeddings-url", default=None)
    parser.add_argument("--dims", type=int, default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--run", required=True)
    parser.add_argument("--outdir", default="reports/corridas")
    args = parser.parse_args()

    questions = [json.loads(line) for line in Path(args.silver).read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = [
        q
        for q in questions
        if (args.split == "all" or q.get("split") == args.split)
        and not q.get("excluded")
        and q.get("reference_doc_ids")
    ]
    if not questions:
        raise SystemExit("no hay preguntas para ese split")

    embedder = build_embedder(args.embedder, args.embeddings_url, args.dims)
    client = get_client()
    total = count(client, args.collection)
    out = Path(args.outdir) / f"{args.run}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out.open("w", encoding="utf-8") as handle:
        for question in questions:
            for rep in range(1, args.reps + 1):
                started = time.perf_counter()
                vector = embedder.embed_queries([question["question"]])[0]
                hits = dedupe(search(client, args.collection, vector, args.k))
                elapsed = round((time.perf_counter() - started) * 1000, 2)
                record = {
                    "run": args.run,
                    "question_id": question["id"],
                    "split": args.split,
                    "model": args.embedder,
                    "rep": rep,
                    "mode": "retrieve",
                    "k": args.k,
                    "collection": args.collection,
                    "collection_rows": total,
                    "reference_doc_ids": question.get("reference_doc_ids", []),
                    "reference_chunk_ids": question.get("reference_chunk_ids", []),
                    "reference_span": question.get("reference_span", ""),
                    "response": {"chunks": hits, "retrieval_ms": elapsed},
                    "error": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                written += 1
            print(f"{question['id']} listo ({written})", flush=True)

    print(json.dumps({"run": args.run, "registros": written, "filas": total, "salida": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
