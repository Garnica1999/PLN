"""Agrega los embeddings bge-m3 a los documentos ya indexados en Solr."""
from __future__ import annotations

import argparse
import time

import requests

from src.rag.config import settings
from src.rag.store import get_client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solr", default="http://192.168.1.100:8983/solr/noticias")
    parser.add_argument("--collection", default=settings.collection)
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--commit-every", type=int, default=50000)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    client = get_client()
    iterator = client.query_iterator(
        collection_name=args.collection,
        batch_size=args.batch,
        filter="",
        output_fields=["chunk_id", "vector"],
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
            payload = [
                {"id": row["chunk_id"], "vector": {"set": [float(x) for x in row["vector"]]}}
                for row in rows
            ]
            requests.post(
                f"{args.solr}/update",
                params={"commit": "false"},
                json=payload,
                timeout=180,
            ).raise_for_status()
            total += len(rows)
            since_commit += len(rows)
            if since_commit >= args.commit_every:
                requests.post(
                    f"{args.solr}/update",
                    params={"commit": "true"},
                    json=[],
                    timeout=180,
                ).raise_for_status()
                since_commit = 0
                elapsed = time.perf_counter() - started
                print(f"{total} vectores, {elapsed:.1f} s, {total / elapsed:.0f} docs/s", flush=True)
    finally:
        iterator.close()
    requests.post(f"{args.solr}/update", params={"commit": "true"}, json=[], timeout=180).raise_for_status()
    print(f"listo: {total} vectores en {time.perf_counter() - started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
