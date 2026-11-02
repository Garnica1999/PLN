"""Propone candidatos de preguntas para el gold. La validación final es humana."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.rag.config import settings
from src.rag.corpus import normalize_language
from src.rag.pipeline import call_llm
from src.rag.quality import is_junk

PROMPT = """Lee el documento y escribe UNA pregunta factual cuya respuesta este en el texto.
Devuelve solo JSON con las claves question, answer, span y type.
- question: la pregunta, en el idioma del documento.
- answer: la respuesta en una frase.
- span: una cita LITERAL del documento, de 20 a 300 caracteres, que responda la pregunta.
- type: directa, comparativa o ambigua.
No inventes datos que no esten en el documento.

DOCUMENTO:
{text}
"""


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


PROMPT_STRICT = """Lee el documento y escribe UNA pregunta factual cuya respuesta este en el texto.
La pregunta DEBE nombrar la entidad, el lugar, la fecha o el hecho exacto que aparece en el documento,
de modo que solo ese documento la responda. Prohibido usar "el pais", "la medida", "el sitio",
"la iniciativa", "el programa" o "la empresa" sin nombrarlos. Prohibido preguntar cifras globales
o hechos que aparezcan en muchos documentos.
Devuelve solo JSON con las claves question, answer, span y type.
- span: cita LITERAL del documento, de 20 a 300 caracteres, que responda la pregunta.
- type: directa, comparativa o ambigua.
No inventes datos que no esten en el documento.

DOCUMENTO:
{text}
"""


def parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not data.get("question") or not data.get("answer") or not data.get("span"):
        return None
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=settings.parquet_path)
    parser.add_argument("--sample", default="eval/pilot_docs.parquet")
    parser.add_argument("--out", default="eval/gold/candidates.jsonl")
    parser.add_argument("--gold-out", default=None, help="Escribe un gold listo para evaluar")
    parser.add_argument("--split", default="smoke")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--model", default=settings.llm_model)
    parser.add_argument("--langs", default="es,en")
    parser.add_argument("--collection", default="")
    parser.add_argument("--min-chars", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true", help="Exige preguntas que identifiquen el documento")
    args = parser.parse_args()

    selected = None
    if Path(args.sample).exists():
        selected = set(pd.read_parquet(args.sample)["document_id"].tolist())

    table = pq.read_table(
        args.input,
        columns=["document_id", "text", "title", "url", "collection_name", "language", "created_at", "text_sha256"],
    ).to_pandas()
    table = table[table["text"].str.len().between(args.min_chars, 3000)]
    table = table.drop_duplicates("text_sha256")
    langs = {item.strip() for item in args.langs.split(",") if item.strip()}
    if langs:
        table = table[table["language"].map(normalize_language).isin(langs)]
    if args.collection:
        table = table[table["collection_name"] == args.collection]
    if selected:
        table = table[table["document_id"].isin(selected)]
    sample = table.sample(min(args.n, len(table)), random_state=args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    gold_handle = Path(args.gold_out).open("w", encoding="utf-8") if args.gold_out else None  # noqa: SIM115  # se cierra en el finally
    written = 0
    rejected = 0
    try:
        with Path(args.out).open("w", encoding="utf-8") as handle:
            for _, row in sample.iterrows():
                if is_junk(row["text"]):
                    rejected += 1
                    continue
                text = row["text"][:4000]
                prompt = PROMPT_STRICT if args.strict else PROMPT
                data = call_llm(prompt.format(text=text), args.model, 0.0)
                parsed = parse_json(data.get("response", ""))
                span = (parsed or {}).get("span", "").strip()
                if not parsed or normalize_text(span) not in normalize_text(row["text"]):
                    rejected += 1
                    continue
                written += 1
                record = {
                    "document_id": row["document_id"],
                    "title": row["title"],
                    "url": row["url"],
                    "collection_name": row["collection_name"],
                    "language": row["language"],
                    "created_at": row["created_at"],
                    "question": parsed["question"].strip(),
                    "answer": parsed["answer"].strip(),
                    "span": span,
                    "type": parsed.get("type", "directa"),
                    "validated": False,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if gold_handle:
                    gold = {
                        "id": f"{args.split}-{written:03d}",
                        "question": record["question"],
                        "lang": normalize_language(row["language"]),
                        "type": record["type"],
                        "split": args.split,
                        "reference_doc_ids": [row["document_id"]],
                        "reference_chunk_ids": [],
                        "reference_span": span,
                        "reference_answer": record["answer"],
                        "criteria": "La respuesta debe coincidir con el span de referencia.",
                        "validated": False,
                    }
                    gold_handle.write(json.dumps(gold, ensure_ascii=False) + "\n")
                print(f"{written}: {record['question'][:100]}")
    finally:
        if gold_handle:
            gold_handle.close()
    print(json.dumps({"candidatos": written, "rechazados": rejected, "salida": args.out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
