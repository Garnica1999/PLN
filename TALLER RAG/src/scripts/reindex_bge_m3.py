"""Reindexa los chunks existentes con bge-m3 en una colección nueva, sin re-chunkear."""
from __future__ import annotations

import argparse
import time

from pymilvus import DataType, MilvusClient
from sentence_transformers import SentenceTransformer

FIELDS = [
    ("document_id", DataType.VARCHAR, {"max_length": 256}),
    ("title", DataType.VARCHAR, {"max_length": 1024}),
    ("url", DataType.VARCHAR, {"max_length": 2048}),
    ("collection_name", DataType.VARCHAR, {"max_length": 64}),
    ("source", DataType.VARCHAR, {"max_length": 128}),
    ("language", DataType.VARCHAR, {"max_length": 16}),
    ("created_at", DataType.VARCHAR, {"max_length": 32}),
    ("position", DataType.INT64, {}),
    ("end_offset", DataType.INT64, {}),
    ("chunk_strategy", DataType.VARCHAR, {"max_length": 32}),
    ("text_sha256", DataType.VARCHAR, {"max_length": 64}),
]
OUTPUT = ["chunk_id", "text", *[name for name, _, _ in FIELDS]]


def create(client: MilvusClient, name: str, dim: int) -> None:
    schema = client.create_schema()
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    for field, dtype, params in FIELDS:
        schema.add_field(field, dtype, **params)
    index = client.prepare_index_params()
    index.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 200})
    client.create_collection(name, schema=schema, index_params=index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="rag_full_e5base_d768_v2_v1")
    parser.add_argument("--target", default="rag_full_bge_m3_d1024_v1")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--encode-batch", type=int, default=128)
    parser.add_argument("--uri", default="http://localhost:19530")
    args = parser.parse_args()

    client = MilvusClient(uri=args.uri)
    if not client.has_collection(args.target):
        create(client, args.target, 1024)

    existing: set[str] = set()
    iterator = client.query_iterator(collection_name=args.target, batch_size=10000, filter="", output_fields=["chunk_id"])
    while True:
        batch = iterator.next()
        if not batch:
            break
        existing.update(row["chunk_id"] for row in batch)
    iterator.close()
    if existing:
        print(f"Reanudando con {len(existing)} chunks ya indexados", flush=True)

    model = SentenceTransformer(args.model, device="cuda")
    model.half()

    iterator = client.query_iterator(collection_name=args.source, batch_size=args.batch, filter="", output_fields=OUTPUT)
    total = 0
    started = time.perf_counter()
    while True:
        batch = iterator.next()
        if not batch:
            break
        pending = [row for row in batch if row["chunk_id"] not in existing]
        if not pending:
            continue
        vectors = model.encode(
            [row["text"] for row in pending],
            batch_size=args.encode_batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        records = [{**row, "vector": vector.tolist()} for row, vector in zip(pending, vectors)]
        client.insert(args.target, records)
        total += len(records)
        elapsed = time.perf_counter() - started
        print(f"  {total} nuevos, {total/elapsed:.1f} chunks/s", flush=True)
    iterator.close()
    client.flush(args.target)
    print(f"{total} chunks en {(time.perf_counter()-started)/60:.1f} min")
    print(client.get_collection_stats(args.target))


if __name__ == "__main__":
    main()
