"""Indexa los chunks de Milvus en Solr para el camino RAG-Solr."""
from __future__ import annotations

import argparse
import time

import requests

from src.rag.config import settings
from src.rag.store import OUTPUT_FIELDS, get_client


def to_doc(row: dict) -> dict:
    chunk_id = row.get("chunk_id") or f"{row.get('document_id', '')}:{row.get('position', 0)}"
    return {
        "id": chunk_id,
        "chunk_id": chunk_id,
        "text": row.get("text") or "",
        "document_id": row.get("document_id") or "",
        "title": row.get("title") or "",
        "url": row.get("url") or "",
        "collection_name": row.get("collection_name") or "",
        "source": row.get("source") or "",
        "language": row.get("language") or "",
        "created_at": str(row.get("created_at") or ""),
        "position": int(row.get("position") or 0),
        "end_offset": int(row.get("end_offset") or 0),
        "chunk_strategy": row.get("chunk_strategy") or "",
        "text_sha256": row.get("text_sha256") or "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solr", default="http://192.168.1.100:8983/solr/noticias")
    parser.add_argument("--collection", default=settings.collection)
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--commit-every", type=int, default=50000)
    args = parser.parse_args()

    client = get_client()
    iterator = client.query_iterator(
        collection_name=args.collection,
        batch_size=args.batch,
        filter="",
        output_fields=["chunk_id", *OUTPUT_FIELDS],
        limit=args.limit if args.limit else -1,
    )
    total = 0
    since_commit = 0
    started = time.perf_counter()
    try:
        while True:
            rows = iterator.next()
            if not rows:
                break
            docs = [to_doc(row) for row in rows]
            requests.post(f"{args.solr}/update", params={"commit": "false"}, json=docs, timeout=120).raise_for_status()
            total += len(docs)
            since_commit += len(docs)
            if since_commit >= args.commit_every:
                requests.post(f"{args.solr}/update", params={"commit": "true"}, json=[], timeout=120).raise_for_status()
                since_commit = 0
            if total % 50000 < args.batch:
                elapsed = time.perf_counter() - started
                print(f"{total} docs, {elapsed:.1f} s, {total / elapsed:.0f} docs/s", flush=True)
    finally:
        iterator.close()
    requests.post(f"{args.solr}/update", params={"commit": "true"}, json=[], timeout=120).raise_for_status()
    print(f"listo: {total} docs en {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
