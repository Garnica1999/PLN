"""Diagnostico: en que puesto aparece el documento de referencia.

Recupera 100 fragmentos con bge-m3 y mide el rango del primer fragmento del
documento de referencia para cada pregunta directa del conjunto reservado.
Compara tres valores de `ef` para separar un fallo de ranking de un fallo de
representacion.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.rag.pipeline import retrieve

GOLD = Path("eval/gold/gold_reservado.jsonl")
OUT = Path("reports/tablas/reservado_ranking.csv")
COLLECTION = "rag_full_bge_m3_d1024_v1"
EF_VALUES = (800, 1600, 3200)
K = 100


def main() -> None:
    questions = [
        row
        for row in (json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip())
        if row["type"] == "directa"
    ]
    rows = []
    for ef in EF_VALUES:
        for question in questions:
            hits, elapsed = retrieve(question["question"], k=K, collection=COLLECTION, ef=ef)
            doc_ids = [hit.get("document_id") for hit in hits]
            reference = (question.get("reference_doc_ids") or [None])[0]
            rank = doc_ids.index(reference) + 1 if reference in doc_ids else None
            rows.append(
                {
                    "id": question["id"],
                    "lang": question["lang"],
                    "ef": ef,
                    "rank": rank,
                    "retrieval_ms": elapsed,
                    "reference": reference,
                }
            )
            print(f"ef={ef} {question['id']} rank={rank} ms={elapsed}", flush=True)
    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    for ef in EF_VALUES:
        subset = frame[frame["ef"] == ef]
        ranks = subset["rank"].dropna()
        filled = subset["rank"].fillna(10**6)
        print(
            f"ef={ef}: encontrado {len(ranks)}/{len(subset)} | "
            f"recall@10={float((filled <= 10).mean()):.3f} "
            f"recall@20={float((filled <= 20).mean()):.3f} "
            f"recall@50={float((filled <= 50).mean()):.3f} "
            f"recall@100={float((filled <= 100).mean()):.3f} "
            f"p50={subset['retrieval_ms'].median():.1f} ms",
            flush=True,
        )
    print(frame[frame["rank"].isna()][["id", "ef"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
