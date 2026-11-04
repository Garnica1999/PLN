"""Construye el gold v2 de ambiguedad con tipos nuevos y controles.

La anotacion es asistida por el LLM y validada contra la evidencia. No sustituye
una anotacion humana doble; se documenta como gold exploratorio v2.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.rag.ambiguity import detect
from src.rag.chat import condense_query
from src.rag.pipeline import rerank_hits, retrieve_union, score_hits

GOLD_A = Path("eval/gold/gold_ambiguedad.jsonl")
GOLD_B = Path("eval/gold/gold_ambiguedad_otros.jsonl")
RESERVED = Path("eval/gold/gold_reservado.jsonl")
STANDARD = Path("eval/gold/silver_standard.jsonl")
OUT = Path("eval/gold/gold_ambiguedad_v2.jsonl")

CANDIDATES = [
    {"id": "v-001", "type": "unidad", "question": "¿Cuánto cuesta el proyecto de energía eólica?"},
    {"id": "v-002", "type": "unidad", "question": "¿Cuál es el porcentaje de inflación en Colombia?"},
    {"id": "v-003", "type": "unidad", "question": "¿Cuánto cuesta el pasaje de TransMilenio?"},
    {"id": "v-004", "type": "comparacion", "question": "¿Qué país tiene la mayor inflación?"},
    {"id": "v-005", "type": "comparacion", "question": "¿Qué ciudad tiene el mejor clima?"},
    {"id": "v-006", "type": "comparacion", "question": "¿Qué empresa tiene más ingresos?"},
    {"id": "v-007", "type": "conversacional", "question": "¿Y el de agosto?",
     "history": [{"role": "user", "content": "¿Cuántos municipios afectó el terremoto?"}]},
    {"id": "v-008", "type": "conversacional", "question": "¿Y Duque?",
     "history": [{"role": "user", "content": "¿Qué dijo Petro sobre la reforma tributaria?"}]},
    {"id": "v-009", "type": "conversacional", "question": "¿Y la COP29?",
     "history": [{"role": "user", "content": "¿Cuándo se celebró la COP16?"}]},
    {"id": "v-010", "type": "referencial", "question": "¿Qué empresa lidera el proyecto de energía eólica?"},
    {"id": "v-011", "type": "temporal", "question": "¿Cuándo se aprobó la reforma pensional?"},
    {"id": "v-012", "type": "alcance", "question": "¿Cuántos votos obtuvo la reforma pensional?"},
    {"id": "v-013", "type": "referencial", "question": "¿Qué universidad realizó el estudio?"},
    {"id": "v-014", "type": "temporal", "question": "¿Cuándo se aprobó la reforma tributaria?"},
    {"id": "v-015", "type": "alcance", "question": "¿Cuántos empleos creó el proyecto?"},
]


def span_for(text: str, answer: str) -> str:
    index = text.lower().find(answer.lower())
    if index < 0:
        return text[:300]
    return text[max(0, index - 150) : index + len(answer) + 150]


def main() -> None:
    existing = []
    for path in (GOLD_A, GOLD_B):
        existing.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    ambiguous = [row for row in existing if row["type"] == "ambigua"]
    controls = []
    seen = set()
    for row in existing:
        if row["type"] != "no_ambigua" or row["question"] in seen:
            continue
        seen.add(row["question"])
        controls.append(row)

    new_rows = []
    for candidate in CANDIDATES:
        history = candidate.get("history")
        query = condense_query(history, candidate["question"]) if history else candidate["question"]
        hits, _ = retrieve_union(query, dense_k=50, sparse_k=50)
        hits = rerank_hits(query, score_hits(query, hits))[:10]
        detection = detect(query, hits, history=history, force=True)
        if not detection.get("ambiguous") or len(detection.get("options", [])) < 2:
            print("descartada", candidate["id"], candidate["question"], "|", detection.get("ambiguous"))
            continue
        options = detection["options"]
        intended = options[0]["answer"]
        evidence_index = options[0]["evidence"][0] if options[0]["evidence"] else 1
        hit = hits[evidence_index - 1]
        new_rows.append(
            {
                "id": candidate["id"],
                "question": candidate["question"],
                "lang": "es",
                "type": "ambigua",
                "subtype": candidate["type"],
                "split": "dev" if len(new_rows) % 3 else "test",
                "history": history,
                "intended_answer": intended,
                "intended_aliases": [intended],
                "options": options,
                "conditional_answers": [{"label": option["label"], "answer": option["answer"]} for option in options],
                "reference_doc_ids": [hit.get("document_id")],
                "reference_span": span_for(hit.get("text") or "", intended),
                "source": "v2_llm",
            }
        )
        print("ok", candidate["id"], candidate["type"], "|", intended, "|", len(options), "opciones")

    # controles adicionales del gold reservado y del estandar
    reserved = [json.loads(line) for line in RESERVED.read_text(encoding="utf-8").splitlines() if line.strip()]
    standard = [json.loads(line) for line in STANDARD.read_text(encoding="utf-8").splitlines() if line.strip()]
    used = {row["question"] for row in controls}
    for row in reserved + standard:
        if len(controls) >= 21:
            break
        if row.get("type") != "directa" or row["question"] in used:
            continue
        used.add(row["question"])
        controls.append(
            {
                "id": f"v-{row['id']}",
                "question": row["question"],
                "lang": row.get("lang", "es"),
                "type": "no_ambigua",
                "split": "dev" if len(controls) % 3 else "test",
                "reference_answer": row.get("reference_answer") or "",
                "reference_doc_ids": row.get("reference_doc_ids") or [],
                "reference_span": row.get("reference_span") or "",
                "source": "control",
            }
        )

    rows = ambiguous + new_rows + controls
    OUT.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print("total", len(rows), "ambiguas", len(ambiguous) + len(new_rows), "controles", len(controls))


if __name__ == "__main__":
    main()
