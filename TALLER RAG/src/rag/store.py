"""Acceso a Milvus: esquema, índice, escritura y búsqueda."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from pymilvus import DataType, MilvusClient

from .config import settings

OUTPUT_FIELDS = [
    "text",
    "document_id",
    "title",
    "url",
    "collection_name",
    "source",
    "language",
    "created_at",
    "position",
    "end_offset",
    "chunk_strategy",
    "text_sha256",
]


@lru_cache(maxsize=1)
def get_client() -> MilvusClient:
    return MilvusClient(uri=settings.milvus_uri)


def collection_dim(client: MilvusClient, name: str) -> int:
    info = client.describe_collection(name)
    for field in info.get("fields", []):
        if field.get("name") == "vector":
            return int((field.get("params") or {}).get("dim", 0))
    return 0


def ensure_collection(client: MilvusClient, name: str, dim: int) -> None:
    if client.has_collection(name):
        existing = collection_dim(client, name)
        if existing and existing != dim:
            raise ValueError(f"la colección {name} tiene dimensión {existing}, se esperaba {dim}")
        return
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("document_id", DataType.VARCHAR, max_length=256)
    schema.add_field("title", DataType.VARCHAR, max_length=1024)
    schema.add_field("url", DataType.VARCHAR, max_length=2048)
    schema.add_field("collection_name", DataType.VARCHAR, max_length=64)
    schema.add_field("source", DataType.VARCHAR, max_length=128)
    schema.add_field("language", DataType.VARCHAR, max_length=16)
    schema.add_field("created_at", DataType.VARCHAR, max_length=32)
    schema.add_field("position", DataType.INT64)
    schema.add_field("end_offset", DataType.INT64)
    schema.add_field("chunk_strategy", DataType.VARCHAR, max_length=32)
    schema.add_field("text_sha256", DataType.VARCHAR, max_length=64)

    index = client.prepare_index_params()
    index.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    client.create_collection(collection_name=name, schema=schema, index_params=index)


def upsert(client: MilvusClient, name: str, rows: list[dict], batch: int = 500) -> int:
    written = 0
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        client.upsert(collection_name=name, data=chunk)
        written += len(chunk)
    return written


def search(
    client: MilvusClient,
    name: str,
    vector: np.ndarray,
    k: int,
    expr: str | None = None,
    ef: int | None = None,
) -> list[dict]:
    if not client.has_collection(name):
        return []
    results = client.search(
        collection_name=name,
        data=[vector.tolist()],
        limit=k,
        output_fields=OUTPUT_FIELDS,
        search_params={"metric_type": "COSINE", "params": {"ef": ef or settings.search_ef}},
        filter=expr or "",
    )
    hits = []
    for hit in results[0]:
        entity = hit.get("entity", {}) or {}
        hits.append({"chunk_id": hit.get("chunk_id") or hit.get("id"), "score": float(hit.get("distance", 0.0)), **entity})
    return hits


def count(client: MilvusClient, name: str, exact: bool = False) -> int:
    if not client.has_collection(name):
        return 0
    if exact:
        client.flush(name)
    rows = client.query(collection_name=name, filter="", output_fields=["count(*)"], limit=None)
    return int(rows[0].get("count(*)", 0)) if rows else 0


def collections(client: MilvusClient) -> list[str]:
    return sorted(client.list_collections())


def drop(client: MilvusClient, name: str) -> None:
    if client.has_collection(name):
        client.drop_collection(collection_name=name)
