"""Construye el grafo de documentos y entidades en Neo4j.

Uso local (el grafo vive en el servidor):
    NEO4J_PASSWORD=... python -m src.scripts.build_graph --limit 2000

Sin NER, solo metadatos:
    python -m src.scripts.build_graph --skip-ner
"""
from __future__ import annotations

import argparse
import math
import os
import time

import pandas as pd
from neo4j import GraphDatabase

from src.rag.entities import extract

COLUMNS = [
    "document_id",
    "text",
    "title",
    "url",
    "publisher",
    "keyword_family",
    "language",
    "created_at",
    "source",
]

SCHEMA = [
    (
        "CREATE CONSTRAINT document_id_unique IF NOT EXISTS "
        "FOR (d:Document) REQUIRE d.document_id IS UNIQUE"
    ),
    (
        "CREATE CONSTRAINT entity_key_unique IF NOT EXISTS "
        "FOR (e:Entity) REQUIRE e.key IS UNIQUE"
    ),
    (
        "CREATE FULLTEXT INDEX entity_names IF NOT EXISTS "
        "FOR (e:Entity) ON EACH [e.name]"
    ),
]

WRITE_DOCUMENTS = """
UNWIND $docs AS row
MERGE (d:Document {document_id: row.document_id})
SET d.title = row.title,
    d.url = row.url,
    d.publisher = row.publisher,
    d.topic = row.topic,
    d.language = row.language,
    d.published_at = row.published_at,
    d.source = row.source
WITH d, row
UNWIND row.entities AS ent
MERGE (e:Entity {key: ent.key})
ON CREATE SET e.name = ent.name, e.type = ent.type
MERGE (d)-[m:MENTIONS]->(e)
SET m.count = ent.count
"""


def clean(value) -> str:
    """Convierte un valor de pandas en texto corto y seguro para Neo4j."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)[:1000]


def document_row(row, skip_ner: bool) -> dict:
    """Arma el documento con sus entidades extraídas."""
    text = clean(row.text)
    title = clean(row.title)
    entities = []
    if not skip_ner and text:
        payload = f"{title}. {text[:1500]}"
        entities = list(extract(payload, clean(row.language)).values())
    return {
        "document_id": clean(row.document_id),
        "title": title[:500],
        "url": clean(row.url),
        "publisher": clean(row.publisher)[:200],
        "topic": clean(row.keyword_family)[:100],
        "language": clean(row.language)[:16],
        "published_at": clean(row.created_at)[:32],
        "source": clean(row.source)[:64],
        "entities": entities,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default="Dataset/documentos_rag.parquet")
    parser.add_argument("--uri", default="bolt://192.168.1.100:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", ""))
    parser.add_argument("--batch", type=int, default=400)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress", type=int, default=5000)
    parser.add_argument("--skip-ner", action="store_true")
    args = parser.parse_args()

    if not args.password:
        raise SystemExit("falta NEO4J_PASSWORD o --password")

    frame = pd.read_parquet(args.parquet, columns=COLUMNS)
    if args.limit:
        frame = frame.head(args.limit)
    total = len(frame)
    print(f"documentos: {total}", flush=True)

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    driver.verify_connectivity()
    with driver.session() as session:
        for statement in SCHEMA:
            session.run(statement)

    started = time.perf_counter()
    buffer: list[dict] = []
    written = 0
    mentions = 0
    with driver.session() as session:
        for row in frame.itertuples(index=False):
            item = document_row(row, args.skip_ner)
            mentions += len(item["entities"])
            buffer.append(item)
            if len(buffer) >= args.batch:
                session.run(WRITE_DOCUMENTS, docs=buffer)
                written += len(buffer)
                buffer.clear()
                if written % args.progress == 0:
                    elapsed = time.perf_counter() - started
                    rate = written / elapsed
                    eta = (total - written) / rate if rate else 0
                    print(
                        f"{written}/{total} docs, {mentions} menciones, "
                        f"{rate:.0f} docs/s, eta {eta/60:.1f} min",
                        flush=True,
                    )
        if buffer:
            session.run(WRITE_DOCUMENTS, docs=buffer)
            written += len(buffer)

        counts = session.run(
            "MATCH (d:Document) WITH count(d) AS docs "
            "MATCH (e:Entity) RETURN docs, count(e) AS entities"
        ).single()
        print(
            f"listo: {counts['docs']} documentos, {counts['entities']} entidades, "
            f"{mentions} menciones en {time.perf_counter() - started:.1f} s",
            flush=True,
        )
    driver.close()


if __name__ == "__main__":
    main()
