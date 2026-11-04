"""Evalua el flujo hibrido con la compuerta estricta sobre el conjunto reservado.

Mide el sistema completo: recuperacion hibrida, compuerta con strict=True y
generacion. No sustituye la corrida sin compuerta; la complementa.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import requests

from src.scripts.eval_configuracion_final import contains_reference

API = os.getenv("API_URL", "http://localhost:8010").rstrip("/")
SILVER = Path("eval/silver/silver_reservado.jsonl")
RAW = Path("reports/corridas/reservado_hibrido_estricto_raw.jsonl")
SUMMARY = Path("reports/tablas/reservado_hibrido_estricto_resumen.csv")


def post(question: str) -> dict:
    response = requests.post(
        f"{API}/chat",
        json={"query": question, "k": 10, "strict": True, "model": "gemma4:e4b", "ambiguity_policy": "off"},
        timeout=900,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    rows = [json.loads(line) for line in SILVER.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = []
    for index, row in enumerate(rows, start=1):
        data = post(row["question"])
        answer = data.get("answer") or ""
        abstained = bool(data.get("abstained"))
        direct = row.get("type") == "directa"
        coverage = bool(direct and not abstained and contains_reference(row.get("reference_answer") or "", answer))
        record = {
            "id": row["id"],
            "type": row.get("type"),
            "question": row["question"],
            "reference_answer": row.get("reference_answer") or "",
            "abstained": abstained,
            "abstain_reason": data.get("abstain_reason"),
            "domain_probability": data.get("domain_probability"),
            "answer": answer,
            "citations": data.get("citations") or [],
            "token_coverage": coverage,
            "total_ms": data.get("total_ms"),
            "retrieval_ms": data.get("retrieval_ms"),
            "generation_ms": data.get("generation_ms"),
        }
        records.append(record)
        print(
            f"{index}/{len(rows)} {row['id']} abst={abstained} "
            f"motivo={record['abstain_reason']} cobertura={coverage}",
            flush=True,
        )

    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")

    direct = [record for record in records if record["type"] == "directa"]
    negatives = [record for record in records if record["type"] != "directa"]
    delivered = [record for record in direct if not record["abstained"]]
    summary = {
        "n": len(records),
        "directas": len(direct),
        "negativas": len(negatives),
        "entregadas": len(delivered),
        "abstenciones_directas": sum(1 for record in direct if record["abstained"]),
        "abstenciones_negativas": sum(1 for record in negatives if record["abstained"]),
        "rechazos_compuerta": sum(1 for record in direct if record["abstain_reason"] == "fuera_dominio"),
        "cobertura_tokens_entregadas": round(
            sum(1 for record in delivered if record["token_coverage"]) / len(delivered), 3
        )
        if delivered
        else 0.0,
        "cobertura_tokens_directas": round(
            sum(1 for record in direct if record["token_coverage"]) / len(direct), 3
        )
        if direct
        else 0.0,
        "latencia_media_ms": round(
            sum(record["total_ms"] or 0 for record in records) / len(records), 1
        ),
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary), delimiter=";", lineterminator="\n")
        writer.writeheader()
        writer.writerow(summary)
    print(summary)


if __name__ == "__main__":
    main()
