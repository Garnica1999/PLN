"""Ingesta de una variante de limpieza. Es el único bucle de ingesta del proyecto."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.rag.chunking import chunk_fixed, chunk_identity, chunk_sentences
from src.rag.cleaning import VARIANTS, clean_chunk, dedupes_chunks, skips_document
from src.rag.config import settings
from src.rag.corpus import normalize_date, normalize_language, truncate_bytes
from src.rag.embeddings import get_embedder
from src.rag.store import count, drop, ensure_collection, get_client, upsert
from src.rag.textometry import simhash64

COLUMNS = [
    "document_id",
    "text",
    "title",
    "url",
    "collection_name",
    "source",
    "language",
    "created_at",
]


class SimHashIndex:
    """Indice por bandas para descartar chunks con distancia de Hamming <= 3."""

    def __init__(self, max_distance: int = 3, bands: int = 4) -> None:
        self.max_distance = max_distance
        self.bands = bands
        self.width = 64 // bands
        self.buckets: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(bands)]
        self.hashes: list[int] = []

    def add_if_new(self, value: int) -> bool:
        candidates: set[int] = set()
        for band in range(self.bands):
            key = (value >> (band * self.width)) & ((1 << self.width) - 1)
            candidates.update(self.buckets[band].get(key, []))
        for index in candidates:
            if (self.hashes[index] ^ value).bit_count() <= self.max_distance:
                return False
        index = len(self.hashes)
        self.hashes.append(value)
        for band in range(self.bands):
            key = (value >> (band * self.width)) & ((1 << self.width) - 1)
            self.buckets[band][key].append(index)
        return True


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iter_chunks(text: str, strategy: str, args) -> list:
    if strategy == "fixed":
        return list(chunk_fixed(text, args.chunk_size, args.overlap))
    if strategy == "sentences":
        return list(chunk_sentences(text, args.max_tokens, args.overlap_sentences))
    raise ValueError(f"estrategia desconocida: {strategy}")


def run(
    variant: str,
    default_collection: str,
    default_log: str = "reports/tablas/ingesta_limpieza.jsonl",
) -> None:
    if variant not in VARIANTS:
        raise SystemExit(f"variante desconocida: {variant}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=settings.parquet_path)
    parser.add_argument("--sample", default="eval/pilot_docs.parquet")
    parser.add_argument("--collection", default=default_collection)
    parser.add_argument("--strategy", choices=["fixed", "sentences"], default="fixed")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--overlap-sentences", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--drop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log", default=default_log)
    args = parser.parse_args()

    selected = set(pd.read_parquet(args.sample)["document_id"].tolist())
    embedder = get_embedder()
    client = get_client()
    if not args.dry_run:
        if args.drop:
            drop(client, args.collection)
        ensure_collection(client, args.collection, embedder.dims)

    pending: list[dict] = []
    ready: list[dict] = []
    dedup_index = SimHashIndex() if dedupes_chunks(variant) else None
    seen_docs = 0
    indexed_docs = 0
    chunks = 0
    junk = 0
    near_dup = 0
    started = time.perf_counter()

    def flush() -> None:
        nonlocal chunks
        if args.dry_run or not pending:
            return
        vectors = embedder.embed_documents([row["text"] for row in pending])
        for row, vector in zip(pending, vectors):
            ready.append({**row, "vector": vector.tolist()})
        pending.clear()
        if len(ready) >= 500:
            upsert(client, args.collection, ready, 500)
            chunks += len(ready)
            ready.clear()

    stop = False
    parquet = pq.ParquetFile(args.input)
    for batch in parquet.iter_batches(batch_size=200, columns=COLUMNS):
        for record in batch.to_pylist():
            document_id = record["document_id"]
            if document_id not in selected:
                continue
            if seen_docs < args.start:
                seen_docs += 1
                continue
            seen_docs += 1
            text = record["text"] or ""
            if skips_document(variant, text):
                junk += 1
                continue
            seen: set[str] = set()
            for chunk in iter_chunks(text, args.strategy, args):
                cleaned = clean_chunk(variant, chunk.text)
                if not cleaned.strip():
                    continue
                text_hash = sha(cleaned)
                if text_hash in seen:
                    continue
                seen.add(text_hash)
                if dedup_index is not None and not dedup_index.add_if_new(simhash64(cleaned)):
                    near_dup += 1
                    continue
                if args.dry_run:
                    chunks += 1
                    continue
                chunk_id = chunk_identity(
                    variant,
                    document_id,
                    args.strategy,
                    args.chunk_size,
                    args.max_tokens,
                    args.overlap,
                    args.overlap_sentences,
                    chunk.start,
                    text_hash,
                )
                pending.append(
                    {
                        "chunk_id": chunk_id,
                        "text": truncate_bytes(cleaned, 65535),
                        "document_id": document_id,
                        "title": truncate_bytes(record["title"] or "", 1024),
                        "url": truncate_bytes(record["url"] or "", 2048),
                        "collection_name": truncate_bytes(record["collection_name"] or "", 64),
                        "source": truncate_bytes(record["source"] or "", 128),
                        "language": normalize_language(record["language"]) or "",
                        "created_at": normalize_date(record["created_at"]),
                        "position": int(chunk.start),
                        "end_offset": int(chunk.end),
                        "chunk_strategy": truncate_bytes(f"{args.strategy}_{variant}", 32),
                        "text_sha256": text_hash,
                    }
                )
                if len(pending) >= args.batch_size:
                    flush()
            indexed_docs += 1
            if args.limit and indexed_docs >= args.limit:
                stop = True
                break
            if indexed_docs % 200 == 0:
                elapsed = time.perf_counter() - started
                print(
                    json.dumps(
                        {
                            "variante": variant,
                            "docs": indexed_docs,
                            "chunks": chunks + len(ready),
                            "chunks_por_segundo": round((chunks + len(ready)) / max(elapsed, 1e-6), 1),
                        }
                    ),
                    flush=True,
                )
        if stop:
            break

    if not args.dry_run:
        flush()
        if ready:
            upsert(client, args.collection, ready, 500)
            chunks += len(ready)
            ready.clear()

    elapsed = round(time.perf_counter() - started, 1)
    summary = {
        "variante": variant,
        "coleccion": args.collection,
        "estrategia": args.strategy,
        "documentos_muestra": len(selected),
        "documentos_indexados": indexed_docs,
        "omitidos_por_start": min(args.start, seen_docs),
        "descartados_por_calidad": junk,
        "chunks": chunks,
        "descartados_near_dup": near_dup,
        "segundos": elapsed,
        "chunks_por_segundo": round(chunks / max(elapsed, 1e-6), 1),
        "filas_en_milvus": 0 if args.dry_run else count(client, args.collection, exact=True),
        "encoder": getattr(embedder, "name", settings.embeddings_model),
        "dimensiones": embedder.dims,
        "dry_run": args.dry_run,
        "fecha": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.log).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
