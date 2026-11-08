"""Evalua la recuperacion de Solr sobre el silver de 38 preguntas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.rag.solr import search_solr, search_solr_dense
from src.scripts.metrics import row_metrics

SILVER = Path("eval/silver/silver_standard.jsonl")
OUT_CSV = Path("reports/tablas/bench_solr.csv")
OUT_JSONL = Path("reports/corridas/bench_solr.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bm25", "dense"], default="bm25")
    args = parser.parse_args()
    search = search_solr if args.mode == "bm25" else search_solr_dense
    out_csv = OUT_CSV if args.mode == "bm25" else Path(f"reports/tablas/bench_solr_{args.mode}.csv")
    out_jsonl = OUT_JSONL if args.mode == "bm25" else Path(f"reports/corridas/bench_solr_{args.mode}.jsonl")
    rows = [json.loads(line) for line in SILVER.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("reference_doc_ids")]

    records = []
    for row in rows:
        hits, elapsed = search(row["question"], k=10)
        records.append(
            {
                "run": "solr",
                "question_id": row["id"],
                "split": row.get("split"),
                "model": "solr",
                "mode": "retrieve",
                "k": 10,
                "rep": 1,
                "reference_doc_ids": row.get("reference_doc_ids"),
                "reference_span": row.get("reference_span"),
                "response": {"chunks": hits},
                "retrieval_ms": elapsed,
                "error": None,
                "lang": row.get("lang"),
                "source": row.get("source"),
                "type": row.get("type"),
            }
        )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    frame = pd.DataFrame(
        [
            {
                **row_metrics(record),
                "lang": record["lang"],
                "source": record["source"],
                "type": record["type"],
                "retrieval_ms": record["retrieval_ms"],
            }
            for record in records
        ]
    )
    summary = pd.Series(
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
    summary.to_frame().T.to_csv(out_csv, index=False)
    print(summary.to_string())
    print(f"escrito {out_csv} y {out_jsonl}")


if __name__ == "__main__":
    main()
