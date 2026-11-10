"""Recuperación asistida por el grafo de entidades en Neo4j."""
from __future__ import annotations

import json
import os
import time
from functools import lru_cache

from .config import settings
from .entities import extract_query, normalize
from .store import OUTPUT_FIELDS, get_client

QUERY_FIELDS = OUTPUT_FIELDS + ["chunk_id"]


@lru_cache(maxsize=1)
def get_driver():
    """Driver perezoso; None si el grafo no está configurado."""
    uri = os.getenv("NEO4J_URI", "")
    if not uri:
        return None
    try:
        from neo4j import GraphDatabase

        return GraphDatabase.driver(
            uri,
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
        )
    except Exception:  # noqa: BLE001
        return None


def entity_keys(query: str) -> list[str]:
    """Claves de las entidades nombradas en la consulta."""
    return [item["key"] for item in extract_query(query)][:8]


def documents_for_entities(session, keys: list[str], limit: int) -> list[tuple[str, float]]:
    rows = session.run(
        """
        MATCH (d:Document)-[m:MENTIONS]->(e:Entity)
        WHERE e.key IN $keys
        RETURN d.document_id AS doc_id, sum(m.count) AS score
        ORDER BY score DESC
        LIMIT $limit
        """,
        keys=keys,
        limit=limit,
    )
    return [(row["doc_id"], float(row["score"])) for row in rows]


def expand_documents(session, docs: list[str], limit: int) -> list[tuple[str, float]]:
    """Segundo salto acotado: entidades semilla y sus documentos más frecuentes."""
    rows = session.run(
        """
        MATCH (d1:Document)-[m1:MENTIONS]->(e:Entity)
        WHERE d1.document_id IN $docs
        WITH e, sum(m1.count) AS seed_weight
        ORDER BY seed_weight DESC
        LIMIT 12
        CALL (e) {
            MATCH (d2:Document)-[m2:MENTIONS]->(e)
            WHERE NOT d2.document_id IN $docs
            RETURN d2, m2.count AS c
            ORDER BY c DESC
            LIMIT 15
        }
        RETURN d2.document_id AS doc_id, sum(c * seed_weight) AS score
        ORDER BY score DESC
        LIMIT $limit
        """,
        docs=docs,
        limit=limit,
    )
    return [(row["doc_id"], float(row["score"])) for row in rows]


def chunks_for_documents(doc_ids: list[str], limit: int, collection: str | None = None) -> list[dict]:
    """Trae fragmentos de Milvus filtrando por documento, sin búsqueda vectorial."""
    if not doc_ids:
        return []
    client = get_client()
    name = collection or settings.collection
    filter_expr = f"document_id in {json.dumps(doc_ids)}"
    rows = client.query(
        collection_name=name,
        filter=filter_expr,
        output_fields=QUERY_FIELDS,
        limit=limit,
    )
    return [dict(row) for row in rows]


def rank_chunks(query: str, chunks: list[dict], keys: list[str]) -> list[dict]:
    """Puntúa cada fragmento por entidades de la consulta y deja uno por documento."""
    best: dict[str, dict] = {}
    for chunk in chunks:
        text = normalize(chunk.get("text", ""))
        overlap = float(sum(1 for key in keys if key in text))
        chunk["graph_score"] = overlap
        doc_id = chunk.get("document_id") or chunk.get("chunk_id")
        current = best.get(doc_id)
        if current is None or overlap > current["graph_score"]:
            best[doc_id] = chunk
    ordered = sorted(best.values(), key=lambda item: (-item["graph_score"], item.get("position") or 0))
    for chunk in ordered:
        chunk["score"] = chunk["graph_score"]
    return ordered


def search_graph(
    query: str,
    k: int,
    collection: str | None = None,
    hops: int = 1,
) -> tuple[list[dict], float]:
    """Recupera por entidades del grafo y devuelve fragmentos con su puntaje."""
    started = time.perf_counter()
    driver = get_driver()
    keys = entity_keys(query)
    if driver is None or not keys:
        return [], round((time.perf_counter() - started) * 1000, 2)
    with driver.session() as session:
        seeds = documents_for_entities(session, keys, limit=max(60, k * 10))
        seed_docs = [doc_id for doc_id, _ in seeds]
        expanded = expand_documents(session, seed_docs, limit=max(60, k * 10)) if hops and seed_docs else []
    scores: dict[str, float] = {}
    for doc_id, value in seeds:
        scores[doc_id] = scores.get(doc_id, 0.0) + value
    for doc_id, value in expanded:
        scores[doc_id] = scores.get(doc_id, 0.0) + 0.5 * value
    top_docs = sorted(scores, key=scores.get, reverse=True)[: max(20, k * 4)]
    chunks = chunks_for_documents(top_docs, limit=max(200, k * 20), collection=collection)
    hits = rank_chunks(query, chunks, keys)[: max(k, 10)]
    return hits, round((time.perf_counter() - started) * 1000, 2)


def boost_hits(query: str, hits: list[dict], weight: float = 0.15) -> list[dict]:
    """Suma al rerank_score un pequeño bono por entidades presentes en el fragmento."""
    keys = entity_keys(query)
    if not keys:
        return hits
    for hit in hits:
        text = normalize(hit.get("text", ""))
        overlap = sum(1 for key in keys if key in text)
        hit["graph_overlap"] = overlap
        if "rerank_score" in hit:
            hit["rerank_score"] = float(hit["rerank_score"]) + weight * overlap
    return sorted(hits, key=lambda item: -float(item.get("rerank_score", 0.0)))
