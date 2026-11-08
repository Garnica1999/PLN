"""Crea la colección sparse BM25 a partir de la colección dense, sin recalcular embeddings."""
from __future__ import annotations

import argparse
import time

from pymilvus import DataType, Function, FunctionType, MilvusClient

FIELDS = [
    ("document_id", DataType.VARCHAR, {"max_length": 256}),
    ("title", DataType.VARCHAR, {"max_length": 1024}),
    ("url", DataType.VARCHAR, {"max_length": 2048}),
    ("collection_name", DataType.VARCHAR, {"max_length": 128}),
    ("source", DataType.VARCHAR, {"max_length": 64}),
    ("language", DataType.VARCHAR, {"max_length": 16}),
    ("created_at", DataType.VARCHAR, {"max_length": 32}),
    ("position", DataType.INT64, {}),
    ("end_offset", DataType.INT64, {}),
    ("chunk_strategy", DataType.VARCHAR, {"max_length": 64}),
    ("text_sha256", DataType.VARCHAR, {"max_length": 64}),
]
OUTPUT = ["chunk_id", "text", *[name for name, _, _ in FIELDS]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="rag_full_e5base_d768_v2_v1")
    parser.add_argument("--target", default="rag_full_e5base_d768_v2_bm25_v1")
    parser.add_argument("--batch", type=int, default=5000)
    parser.add_argument("--uri", default="http://localhost:19530")
    args = parser.parse_args()

    client = MilvusClient(uri=args.uri)
    if client.has_collection(args.target):
        raise SystemExit(f"{args.target} ya existe")

    schema = client.create_schema()
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("text", DataType.VARCHAR, max_length=65535, enable_analyzer=True)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    for name, dtype, params in FIELDS:
        schema.add_field(name, dtype, **params)
    schema.add_function(
        Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse"],
        )
    )
    index = client.prepare_index_params()
    index.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    client.create_collection(args.target, schema=schema, index_params=index)

    iterator = client.query_iterator(
        collection_name=args.source,
        batch_size=args.batch,
        filter="",
        output_fields=OUTPUT,
    )
    total = 0
    started = time.perf_counter()
    while True:
        batch = iterator.next()
        if not batch:
            break
        client.insert(args.target, batch)
        total += len(batch)
        print(f"  {total} chunks", flush=True)
    iterator.close()
    client.flush(args.target)
    elapsed = time.perf_counter() - started
    print(f"{total} chunks en {elapsed/60:.1f} min")
    print(client.get_collection_stats(args.target))


if __name__ == "__main__":
    main()
