"""Exporta el dataset de RAGAS con la configuracion final del RAG."""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests

API = os.getenv("API_URL", "http://localhost:8010").rstrip("/")
COLLECTION = os.getenv("COLLECTION", "rag_full_bge_m3_d1024_v1")
GOLD = Path("eval/gold/silver_standard_ampliado.jsonl")
OUT = Path("reports/ragas/dataset.jsonl")
NEGATIVES = Path("reports/ragas/negatives.jsonl")


def post(question: str) -> dict:
    response = requests.post(
        f"{API}/chat",
        json={
            "query": question,
            "model": "gemma4:e4b",
            "k": 5,
            "collection": COLLECTION,
            "strict": False,
            "think": False,
        },
        timeout=900,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    gold = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = []
    negatives = []
    for index, row in enumerate(gold, start=1):
        data = post(row["question"])
        item = {
            "id": row["id"],
            "type": row["type"],
            "user_input": row["question"],
            "response": data.get("answer", ""),
            "retrieved_contexts": [hit.get("text", "") for hit in data.get("evidence", [])],
            "reference": row.get("reference_answer") or "",
            "abstained": bool(data.get("abstained")),
            "citations": data.get("citations") or [],
        }
        if row.get("type") == "directa":
            samples.append(item)
        else:
            negatives.append(item)
        print(f"{index}/{len(gold)} {row['id']} abst={item['abstained']} ctx={len(item['retrieved_contexts'])}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for item in samples:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with NEGATIVES.open("w", encoding="utf-8") as handle:
        for item in negatives:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    abstained = sum(1 for item in negatives if item["abstained"])
    print(f"directas {len(samples)} -> {OUT}")
    print(f"sin respuesta {len(negatives)} -> {NEGATIVES}; se abstienen {abstained}/{len(negatives)}")


if __name__ == "__main__":
    main()
