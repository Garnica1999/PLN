"""Benchmark de recuperación sobre el corpus completo: curvas de ef y k, breakdown y fallos.

Corre dentro del contenedor api, sin pasar por la API HTTP. Escribe un JSONL por
configuración y un CSV con las métricas agregadas.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.rag.config import settings
from src.rag.pipeline import retrieve
from src.rag.store import get_client
from src.scripts.metrics import normalize, row_metrics


def load_gold(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("reference_doc_ids")]


def run_config(
    gold: list[dict],
    collection: str,
    ef: int,
    k: int,
    reps: int,
    outdir: Path,
) -> pd.DataFrame:
    run = f"bench_ef{ef}_k{k}"
    records = []
    for question in gold:
        for rep in range(1, reps + 1):
            hits, elapsed = retrieve(question["question"], k=k, collection=collection, ef=ef)
            record = {
                "run": run,
                "question_id": question["id"],
                "split": question.get("split"),
                "model": "e5",
                "mode": "retrieve",
                "k": k,
                "rep": rep,
                "collection": collection,
                "reference_doc_ids": question.get("reference_doc_ids"),
                "reference_span": question.get("reference_span"),
                "response": {"chunks": hits},
                "retrieval_ms": elapsed,
                "error": None,
                "lang": question.get("lang"),
                "source": question.get("source"),
                "type": question.get("type"),
            }
            records.append(record)
    out = outdir / f"{run}.jsonl"
    out.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")
    frame = pd.DataFrame([{**row_metrics(record), "lang": record["lang"], "source": record["source"],
                           "type": record["type"], "retrieval_ms": record["retrieval_ms"]}
                          for record in records])
    return frame


def summarize(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "n_preguntas": frame["question_id"].nunique(),
            "recall": frame["recall"].mean(),
            "p_at_1": frame["p_at_1"].mean(),
            "p_at_5": frame["p_at_5"].mean(),
            "mrr": frame["mrr"].mean(),
            "ndcg": frame["ndcg"].mean(),
            "span_hit": frame["span_hit"].mean(),
            "p50_ms": frame["retrieval_ms"].median(),
            "p95_ms": frame["retrieval_ms"].quantile(0.95),
        }
    )


def check_failures(gold: list[dict], collection: str) -> None:
    client = get_client()
    failures = []
    for question in gold:
        hits, _ = retrieve(question["question"], k=10, collection=collection, ef=200)
        docs = {hit.get("document_id") for hit in hits}
        if not (docs & set(question["reference_doc_ids"])):
            failures.append(question)
    print(f"\nFallos de documento en top-10: {len(failures)}")
    for question in failures:
        span = normalize(question.get("reference_span") or "")
        in_doc = False
        in_chunk = False
        for doc in question["reference_doc_ids"]:
            chunks = client.query(
                collection_name=collection,
                filter=f'document_id == "{doc}"',
                output_fields=["text", "position"],
                limit=200,
            )
            ordered = sorted(chunks, key=lambda chunk: chunk.get("position", 0))
            in_doc = in_doc or bool(span) and span in normalize(" ".join(chunk["text"] for chunk in ordered))
            in_chunk = in_chunk or any(span in normalize(chunk["text"]) for chunk in chunks)
        estado = "partido" if in_doc and not in_chunk else ("en_chunk" if in_chunk else "fuera_del_indice")
        print(f"{question['id']} | {question.get('lang')} | {question.get('source')} | {estado} | {question['question'][:60]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=Path("eval/gold/silver_standard.jsonl"))
    parser.add_argument("--collection", default=settings.collection)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--outdir", type=Path, default=Path("reports/tablas"))
    parser.add_argument("--failures", action="store_true")
    args = parser.parse_args()

    gold = load_gold(args.gold)
    args.outdir.mkdir(parents=True, exist_ok=True)
    configs = [(200, 10), (400, 10), (800, 10), (1600, 10), (200, 20), (200, 50)]
    rows = []
    for ef, k in configs:
        frame = run_config(gold, args.collection, ef, k, args.reps, args.outdir.parent / "corridas")
        summary = summarize(frame)
        summary["ef"] = ef
        summary["k"] = k
        rows.append(summary)
        print(f"ef={ef:<5} k={k:<3} recall={summary['recall']:.4f} ndcg={summary['ndcg']:.4f} "
              f"p_at_1={summary['p_at_1']:.4f} span={summary['span_hit']:.4f} p50={summary['p50_ms']:.1f}ms")
        if ef == 200 and k == 10:
            frame.to_csv(args.outdir / "e6_breakdown_raw.csv", index=False)
            breakdown = frame.groupby(["lang"]).agg(
                n=("question_id", "nunique"), recall=("recall", "mean"), ndcg=("ndcg", "mean"),
                p_at_1=("p_at_1", "mean"), span_hit=("span_hit", "mean"),
            ).reset_index()
            breakdown.to_csv(args.outdir / "e6_breakdown_lang.csv", index=False)
            by_source = frame.groupby(["source"]).agg(
                n=("question_id", "nunique"), recall=("recall", "mean"), ndcg=("ndcg", "mean"),
                p_at_1=("p_at_1", "mean"), span_hit=("span_hit", "mean"),
            ).reset_index()
            by_source.to_csv(args.outdir / "e6_breakdown_source.csv", index=False)
            print("\nPor idioma")
            print(breakdown.round(3).to_string(index=False))
            print("\nPor fuente")
            print(by_source.round(3).to_string(index=False))

    summary = pd.DataFrame(rows)[["ef", "k", "n_preguntas", "recall", "p_at_1", "p_at_5", "mrr", "ndcg", "span_hit", "p50_ms", "p95_ms"]]
    summary.to_csv(args.outdir / "bench_retrieval.csv", index=False)
    print("\n" + summary.round(4).to_string(index=False))

    if args.failures:
        check_failures(gold, args.collection)


if __name__ == "__main__":
    main()
