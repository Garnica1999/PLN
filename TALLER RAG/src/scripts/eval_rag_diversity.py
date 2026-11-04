"""Evalua diversidad (MMR) sobre el top-20 para subir la precision del contexto.

Variantes: baseline top-5 con reranker, MMR con similitud de embeddings y MMR con
score del reranker. Tambien diagnostica el rango del documento de referencia en los
dos fallos conocidos (g-031 y g-046).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.rag.config import settings
from src.rag.embeddings import get_embedder
from src.rag.pipeline import rerank_hits, search_hits
from src.scripts.metrics import ranked_docs, row_metrics

GOLD = Path("eval/gold/silver_standard_ampliado.jsonl")
OUT = Path("reports/tablas/mejoras_diversidad.csv")
COLLECTION = settings.collection
MISSES = ("g-031", "g-046")


def unit(vector: np.ndarray) -> np.ndarray:
    """Normaliza un vector para que el producto punto sea coseno."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def mmr(
    query_vector: np.ndarray,
    doc_vectors: np.ndarray,
    relevance: np.ndarray,
    k: int,
    lam: float,
) -> list[int]:
    """Selecciona k indices maximizando relevancia y penalizando redundancia."""
    selected: list[int] = []
    candidates = list(range(len(doc_vectors)))
    while candidates and len(selected) < k:
        best_index = None
        best_score = -np.inf
        for index in candidates:
            redundancy = (
                max(float(doc_vectors[index] @ doc_vectors[other]) for other in selected)
                if selected
                else 0.0
            )
            score = lam * float(relevance[index]) - (1.0 - lam) * redundancy
            if score > best_score:
                best_index, best_score = index, score
        selected.append(best_index)
        candidates.remove(best_index)
    return selected


def scale(values: np.ndarray) -> np.ndarray:
    """Lleva los valores a [0, 1] para que lambda pese igual en todas las variantes."""
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.ones_like(values)
    return (values - low) / (high - low)


def variants(question: str, embedder) -> dict[str, tuple[list[dict], float]]:
    out: dict[str, tuple[list[dict], float]] = {}

    started = time.perf_counter()
    hits5, _ = search_hits(question, k=5, collection=COLLECTION)
    out["rerank5"] = (rerank_hits(question, hits5)[:5], round((time.perf_counter() - started) * 1000, 2))

    started = time.perf_counter()
    hits20, _ = search_hits(question, k=20, collection=COLLECTION)
    query_vector = unit(np.asarray(embedder.embed_queries([question])[0], dtype=float))
    doc_vectors = np.asarray(
        [unit(np.asarray(vector, dtype=float)) for vector in embedder.embed_documents([hit["text"] for hit in hits20])]
    )
    cosine = doc_vectors @ query_vector
    for lam in (0.5, 0.7):
        order = mmr(query_vector, doc_vectors, scale(cosine), 5, lam)
        out[f"mmr_cos_{lam}"] = ([hits20[index] for index in order], round((time.perf_counter() - started) * 1000, 2))

    started = time.perf_counter()
    reranked = rerank_hits(question, hits20)
    scores = np.asarray([hit.get("rerank_score", 0.0) for hit in reranked], dtype=float)
    vector_by_id = {hit["chunk_id"]: doc_vectors[index] for index, hit in enumerate(hits20)}
    reranked_vectors = np.asarray([vector_by_id[hit["chunk_id"]] for hit in reranked])
    for lam in (0.5, 0.7):
        order = mmr(query_vector, reranked_vectors, scale(scores), 5, lam)
        out[f"mmr_rerank_{lam}"] = ([reranked[index] for index in order], round((time.perf_counter() - started) * 1000, 2))
    out["rerank20_5"] = (reranked[:5], round((time.perf_counter() - started) * 1000, 2))
    return out


def diagnose(question: str, reference: set[str], embedder) -> str:
    """Busca el documento de referencia en el top-50 denso y tras el reranker."""
    hits50, _ = search_hits(question, k=50, collection=COLLECTION)
    dense_rank = next(
        (index + 1 for index, doc in enumerate(ranked_docs(hits50)) if doc in reference),
        None,
    )
    reranked = rerank_hits(question, hits50)
    rerank_rank = next(
        (index + 1 for index, doc in enumerate(ranked_docs(reranked)) if doc in reference),
        None,
    )
    return f"denso top-50: {dense_rank} | reranker top-50: {rerank_rank}"


def main() -> None:
    gold = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold = [row for row in gold if row.get("reference_doc_ids")]
    embedder = get_embedder()
    rows = []
    for index, row in enumerate(gold, start=1):
        found = variants(row["question"], embedder)
        for variant, (hits, elapsed) in found.items():
            record = {
                "run": variant,
                "question_id": row["id"],
                "split": row.get("split"),
                "model": variant,
                "mode": "retrieve",
                "k": 5,
                "rep": 1,
                "collection": COLLECTION,
                "reference_doc_ids": row.get("reference_doc_ids"),
                "reference_span": row.get("reference_span"),
                "response": {"chunks": hits, "retrieval_ms": elapsed},
            }
            metrics = row_metrics(record)
            metrics["variant"] = variant
            metrics["ms"] = elapsed
            rows.append(metrics)
        if row["id"] in MISSES:
            print(f"{row['id']} {row['question'][:60]}", flush=True)
            print("  " + diagnose(row["question"], set(row["reference_doc_ids"]), embedder), flush=True)
        print(f"{index}/{len(gold)}", flush=True)

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("variant")
        .agg(
            n=("question_id", "nunique"),
            recall=("recall", "mean"),
            ndcg=("ndcg", "mean"),
            p_at_1=("p_at_1", "mean"),
            p_at_5=("p_at_5", "mean"),
            span_hit=("span_hit", "mean"),
            p50_ms=("ms", "median"),
        )
        .reset_index()
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT, index=False)
    print(summary.round(4).to_string(index=False))
    print(f"escrito {OUT}")


if __name__ == "__main__":
    main()
