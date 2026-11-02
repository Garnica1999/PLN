"""Compara variantes de recuperacion para la fase 6.

Variantes: denso top-5, reranker top-5, reranker top-20 a top-5, filtros por
score del reranker y expansion de consulta con fusion RRF.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from src.rag.config import settings
from src.rag.pipeline import call_llm, rerank_hits, search_hits, weighted_rrf
from src.scripts.metrics import ranked_docs, row_metrics

GOLD = Path("eval/gold/gold_standard_ampliado.jsonl")
OUT = Path("reports/tablas/mejoras_recuperacion.csv")
OUT_LANG = Path("reports/tablas/mejoras_recuperacion_por_idioma.csv")
COLLECTION = settings.collection
EXPANSION_MODEL = os.getenv("EXPANSION_MODEL", "gemma2:2b")


def expand_query(question: str) -> str:
    prompt = (
        "Reformula la pregunta para buscar en el corpus de prensa colombiana y Bluesky. "
        "Incluye sinonimos y entidades concretas. Devuelve solo la consulta, sin explicacion.\n"
        f"Pregunta: {question}"
    )
    try:
        data = call_llm(prompt, EXPANSION_MODEL, 0.0, timeout=30)
    except Exception:  # noqa: BLE001
        return question
    return (data.get("response") or "").strip() or question


def variants(question: str) -> dict[str, tuple[list[dict], float]]:
    out: dict[str, tuple[list[dict], float]] = {}

    started = time.perf_counter()
    hits5, _ = search_hits(question, k=5, collection=COLLECTION)
    out["dense5"] = (hits5, round((time.perf_counter() - started) * 1000, 2))

    started = time.perf_counter()
    hits5, _ = search_hits(question, k=5, collection=COLLECTION)
    out["rerank5"] = (rerank_hits(question, hits5)[:5], round((time.perf_counter() - started) * 1000, 2))

    started = time.perf_counter()
    hits20, _ = search_hits(question, k=20, collection=COLLECTION)
    reranked20 = rerank_hits(question, hits20)[:5]
    elapsed20 = round((time.perf_counter() - started) * 1000, 2)
    out["rerank20_5"] = (reranked20, elapsed20)
    for ratio in (0.8, 0.9):
        threshold = reranked20[0]["rerank_score"] * ratio if reranked20 else 0.0
        filtered = [hit for hit in reranked20 if hit.get("rerank_score", 0.0) >= threshold]
        out[f"rerank20_5_r{ratio}"] = (filtered or reranked20[:1], elapsed20)

    started = time.perf_counter()
    expansion = expand_query(question)
    original, _ = search_hits(question, k=10, collection=COLLECTION)
    expanded, _ = search_hits(expansion, k=10, collection=COLLECTION)
    fused = weighted_rrf(original, expanded, 0.5, 0.5, top=5)
    out["expansion"] = (fused, round((time.perf_counter() - started) * 1000, 2))
    out["expansion_rerank"] = (
        rerank_hits(question, fused)[:5],
        round((time.perf_counter() - started) * 1000, 2),
    )
    return out


def main() -> None:
    gold = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold = [row for row in gold if row.get("reference_doc_ids")]
    rows = []
    for index, row in enumerate(gold, start=1):
        found = variants(row["question"])
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
            metrics["lang"] = row.get("lang")
            metrics["ms"] = elapsed
            rows.append(metrics)
        if row["id"] in ("g-031", "g-046"):
            print(f"{row['id']} {row['question'][:60]}")
            for variant, (hits, _) in found.items():
                rank = next((i + 1 for i, doc in enumerate(ranked_docs(hits)) if doc in set(row["reference_doc_ids"])), None)
                print(f"  {variant:20} rank={rank}")
        print(f"{index}/{len(gold)}", flush=True)

    frame = pd.DataFrame(rows)
    summary = frame.groupby("variant").agg(
        n=("question_id", "nunique"),
        recall=("recall", "mean"),
        ndcg=("ndcg", "mean"),
        p_at_1=("p_at_1", "mean"),
        p_at_5=("p_at_5", "mean"),
        span_hit=("span_hit", "mean"),
        p50_ms=("ms", "median"),
    ).reset_index()
    by_lang = frame.groupby(["variant", "lang"]).agg(
        n=("question_id", "nunique"),
        recall=("recall", "mean"),
        ndcg=("ndcg", "mean"),
        p_at_1=("p_at_1", "mean"),
        p_at_5=("p_at_5", "mean"),
        span_hit=("span_hit", "mean"),
    ).reset_index()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT, index=False)
    by_lang.to_csv(OUT_LANG, index=False)
    print(summary.round(4).to_string(index=False))
    print(f"escrito {OUT} y {OUT_LANG}")


if __name__ == "__main__":
    main()
