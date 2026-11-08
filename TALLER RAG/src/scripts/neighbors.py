"""Vecinos semánticos de consultas de ejemplo, en español e inglés."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.rag.config import settings
from src.rag.pipeline import retrieve

QUERIES = [
    "resultados de las elecciones en Colombia",
    "protestas y derechos humanos",
    "cambio climático y sequías",
    "what did the president say about the economy",
    "fake news and social media",
    "salud pública y vacunación",
    "migración en la frontera",
    "energy transition and fossil fuels",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=settings.collection)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default="reports/tablas/vecinos.json")
    args = parser.parse_args()

    output = []
    for query in QUERIES:
        hits, elapsed = retrieve(query, k=args.k, collection=args.collection)
        rows = [
            {
                "rank": index,
                "score": round(hit["score"], 4),
                "document_id": hit["document_id"],
                "title": hit.get("title"),
                "language": hit.get("language"),
                "url": hit.get("url"),
                "snippet": hit["text"][:220].replace("\n", " "),
            }
            for index, hit in enumerate(hits, start=1)
        ]
        output.append({"query": query, "retrieval_ms": elapsed, "neighbors": rows})
        print(f"\n{query} ({elapsed} ms)")
        for row in rows:
            print(f"  {row['rank']}. {row['score']:.3f} [{row['language']}] {row['snippet'][:110]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
