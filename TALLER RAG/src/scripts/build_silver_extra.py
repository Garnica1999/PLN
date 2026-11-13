"""Genera preguntas de ampliación desde documentos aleatorios del corpus.

El span literal del documento es el gold. La revisión humana marca cuáles
se quedan; el resto se descarta.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
import requests

OLLAMA = "http://localhost:11435"
MODEL = "gemma4:e4b"
PROMPT = """Lee el documento y escribe UNA pregunta autosuficiente en el mismo idioma: debe nombrar el evento, proyecto, lugar o persona concreta usando el título o el texto; no uses "el proyecto", "el encuentro" ni "la iniciativa" sin nombrarlos. La respuesta debe estar literalmente en el texto. Devuelve solo JSON: {{"question": "...", "answer": "respuesta corta", "span": "fragmento literal del documento que contiene la respuesta"}}

TÍTULO: {title}
DOCUMENTO:
{text}
"""


def call(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_ctx": 8192, "seed": 42},
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json().get("response", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default="Dataset/documentos_rag.parquet")
    parser.add_argument("--out", default="eval/silver/silver_extra.jsonl")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_parquet(
        args.parquet, columns=["document_id", "text", "title", "url", "language", "source"]
    )
    frame = frame[frame["source"] == "media_cloud"]
    frame = frame[frame["text"].str.len().between(700, 3000)]
    used: set[str] = set()
    for path in (
        "eval/silver/silver_standard.jsonl",
        "eval/silver/silver_reservado.jsonl",
        "eval/silver/silver_multihop.jsonl",
    ):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                used.update(json.loads(line).get("reference_doc_ids") or [])
    frame = frame[~frame["document_id"].isin(used)]
    sample = frame.sample(n=min(len(frame), args.n * 6), random_state=args.seed)

    rows: list[dict] = []
    seen: set[str] = set()
    for _, doc in sample.iterrows():
        if len(rows) >= args.n:
            break
        text = str(doc["text"])[:3000]
        language = str(doc["language"])
        try:
            data = json.loads(call(PROMPT.format(title=str(doc.get('title') or '')[:300], text=text)))
        except Exception:  # noqa: BLE001
            continue
        question = str(data.get("question") or "").strip()
        answer = str(data.get("answer") or "").strip()
        span = str(data.get("span") or "").strip()
        if not question or not answer or not span:
            continue
        if span not in text or answer.lower() not in span.lower():
            continue
        key = re.sub(r"\W+", "", question.lower())[:80]
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "id": f"x-{len(rows)+1:03d}",
                "question": question,
                "lang": "es" if language.startswith("es") else "en",
                "type": "ampliacion",
                "split": "extra",
                "reference_doc_ids": [str(doc["document_id"])],
                "reference_chunk_ids": [],
                "reference_span": span,
                "reference_answer": answer,
                "criteria": "La respuesta debe contener el dato del span.",
                "validated": False,
                "source": "media_cloud",
                "reviewer": "",
                "reference_url": str(doc.get("url") or ""),
            }
        )

    Path(args.out).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )
    print(f"escritas {len(rows)} preguntas en {args.out}")


if __name__ == "__main__":
    main()
