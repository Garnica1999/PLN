"""Evaluacion del conjunto reservado con recuperacion hibrida.

Configuracion: union de denso top-10 y BM25 top-20, reordenada con el
cross-encoder en GPU, k=10 para el contexto. Mide recuperacion, respuesta,
abstencion y compuerta en una sola corrida.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.rag.chat import chat
from src.rag.domain import domain_probability
from src.rag.pipeline import retrieve_union, score_hits
from src.scripts.eval_configuracion_final import contains_reference, token_f1
from src.scripts.metrics import row_metrics

GOLD = Path("eval/gold/gold_reservado.jsonl")
RAW = Path("reports/corridas/reservado_hibrida_raw.jsonl")
SUMMARY = Path("reports/tablas/reservado_hibrida_resumen.csv")
PER_QUESTION = Path("reports/tablas/reservado_hibrida_por_pregunta.csv")
THRESHOLD = 0.4


def main() -> None:
    rows = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    for index, row in enumerate(rows, start=1):
        output = chat(row["question"])
        evidence = output.get("evidence") or []
        metrics = row_metrics(
            {
                "reference_doc_ids": row.get("reference_doc_ids") or [],
                "reference_chunk_ids": row.get("reference_chunk_ids") or [],
                "reference_span": row.get("reference_span") or "",
                "mode": "retrieve",
                "k": 10,
                "response": {"chunks": evidence},
            }
        )
        answer = output.get("answer") or ""
        abstained = bool(output.get("abstained"))
        strict = contains_reference(row.get("reference_answer") or "", answer) if not abstained else False
        results.append(
            {
                "id": row["id"],
                "type": row["type"],
                "lang": row["lang"],
                "question": row["question"],
                "reference_answer": row.get("reference_answer") or "",
                "answer": answer,
                "abstained": abstained,
                "abstain_reason": output.get("abstain_reason"),
                "citations": len(output.get("citations") or []),
                "strict_match": strict,
                "token_f1": token_f1(row.get("reference_answer") or "", answer) if not abstained else 0.0,
                "retrieval_recall": metrics["recall"],
                "retrieval_ndcg": metrics["ndcg"],
                "retrieval_p_at_1": metrics["p_at_1"],
                "retrieval_p_at_5": metrics["p_at_5"],
                "retrieval_span_hit": metrics["span_hit"],
                "retrieval_ms": output.get("retrieval_ms"),
                "generation_ms": output.get("generation_ms"),
                "total_ms": output.get("total_ms"),
                "context_tokens": output.get("context_tokens"),
            }
        )
        print(
            f"{index}/{len(rows)} {row['id']} abst={abstained} "
            f"recall={metrics['recall']:.2f} match={strict}",
            flush=True,
        )

    # Compuerta: se evalua sobre la misma union, sin generar respuestas.
    for row, result in zip(rows, results):
        hits, _ = retrieve_union(row["question"])
        hits = score_hits(row["question"], hits)
        probability = domain_probability(row["question"], hits)
        result["domain_probability"] = round(probability, 4)
        result["gate_domain"] = probability >= THRESHOLD

    frame = pd.DataFrame(results)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    PER_QUESTION.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PER_QUESTION, index=False)
    with RAW.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    direct = frame[frame["type"] == "directa"]
    negative = frame[frame["type"] == "sin_respuesta"]
    with_refs = direct[direct["retrieval_recall"].notna()]
    delivered = direct[~direct["abstained"]]
    summary = {
        "n_total": len(frame),
        "n_direct": len(direct),
        "n_negative": len(negative),
        "recall_at_10": float(with_refs["retrieval_recall"].mean()),
        "ndcg_at_10": float(with_refs["retrieval_ndcg"].mean()),
        "p_at_1": float(with_refs["retrieval_p_at_1"].mean()),
        "p_at_5": float(with_refs["retrieval_p_at_5"].mean()),
        "span_hit": float(with_refs["retrieval_span_hit"].dropna().mean()),
        "delivered_direct": int(len(delivered)),
        "abstained_direct": int(direct["abstained"].sum()),
        "abstained_negative": int(negative["abstained"].sum()),
        "strict_match_direct": int(delivered["strict_match"].sum()),
        "strict_match_rate": float(delivered["strict_match"].mean()),
        "token_f1_direct": float(delivered["token_f1"].mean()),
        "citation_rate": float(delivered["citations"].gt(0).mean()),
        "gate_false_rejection": float((~direct["gate_domain"]).mean()),
        "gate_false_acceptance": float(negative["gate_domain"].mean()),
        "total_ms_mean": float(frame["total_ms"].mean()),
        "total_ms_median": float(frame["total_ms"].median()),
        "total_ms_p95": float(frame["total_ms"].quantile(0.95)),
    }
    pd.DataFrame([summary]).to_csv(SUMMARY, index=False)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)


if __name__ == "__main__":
    main()
