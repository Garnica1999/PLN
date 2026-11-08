"""Recuperación sobre Solr para el camino RAG-Solr."""
from __future__ import annotations

import time

import requests

from .config import settings

FIELDS = (
    "chunk_id,text,document_id,title,url,collection_name,source,language,"
    "created_at,position,end_offset,chunk_strategy,text_sha256,score"
)


def search_solr(query: str, k: int) -> tuple[list[dict], float]:
    """Busca en Solr con edismax y devuelve los hits con la forma del pipeline."""
    started = time.perf_counter()
    params = {
        "q": f"text:({query})",
        "defType": "edismax",
        "qf": "text title^2",
        "rows": k,
        "fl": FIELDS,
    }
    response = requests.get(f"{settings.solr_url}/select", params=params, timeout=settings.request_timeout)
    response.raise_for_status()
    docs = response.json().get("response", {}).get("docs", [])
    hits = []
    for doc in docs:
        score = float(doc.pop("score", 0.0))
        for key in ("title", "language", "text"):
            value = doc.get(key)
            if isinstance(value, list):
                doc[key] = value[0] if value else ""
        hits.append({"chunk_id": doc.get("chunk_id") or doc.get("id"), "score": score, **doc})
    return hits, round((time.perf_counter() - started) * 1000, 2)
