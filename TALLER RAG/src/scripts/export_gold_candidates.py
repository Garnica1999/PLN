"""Genera el Excel de candidatos para ampliar el gold con más documentos relevantes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=Path("eval/gold/gold_standard.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eval/gold/ampliacion_candidatos.xlsx"))
    parser.add_argument("--collection", default="rag_full_bge_m3_d1024_v1")
    parser.add_argument("--candidates", type=int, default=2)
    args = parser.parse_args()

    gold = [json.loads(line) for line in args.gold.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold = [row for row in gold if row.get("reference_doc_ids")]
    model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    model.half()
    client = MilvusClient(uri="http://localhost:19530")

    rows = []
    for question in gold:
        vector = model.encode([question["question"]], normalize_embeddings=True)[0]
        response = client.search(
            collection_name=args.collection,
            data=[vector.tolist()],
            anns_field="vector",
            limit=20,
            output_fields=["document_id", "text", "title", "url"],
        )
        seen = set(question["reference_doc_ids"])
        picked = 0
        for hit in response[0]:
            entity = hit.get("entity") or {}
            document_id = entity.get("document_id")
            if not document_id or document_id in seen:
                continue
            seen.add(document_id)
            rows.append(
                {
                    "id": question["id"],
                    "pregunta": question["question"],
                    "respuesta de referencia": question.get("reference_answer", ""),
                    "documento": document_id,
                    "titulo": entity.get("title", ""),
                    "url": entity.get("url", ""),
                    "fragmento": (entity.get("text") or "")[:800],
                    "responde": None,
                    "nota": None,
                }
            )
            picked += 1
            if picked >= args.candidates:
                break

    book = Workbook()
    guide = book.active
    guide.title = "Instrucciones"
    guide.column_dimensions["A"].width = 110
    lines = [
        ("Ampliación del gold: documentos candidatos", True),
        ("", False),
        ("Para cada pregunta se proponen dos documentos recuperados por bge-m3 que no están en el gold actual.", False),
        ("Marca 'Sí' en la columna 'responde' si el documento contiene la respuesta a la pregunta.", False),
        ("Marca 'No' si solo menciona el tema, una entidad relacionada o un dato distinto.", False),
        ("Si dudas, usa la nota. No cambies el id ni el documento.", False),
        ("", False),
        ("Al terminar, guarda el archivo y devuélvelo. Tiempo estimado: 30 a 45 minutos.", False),
    ]
    for row, (text, bold) in enumerate(lines, start=1):
        cell = guide.cell(row=row, column=1, value=text)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        if bold:
            cell.font = Font(bold=True, size=12)

    sheet = book.create_sheet("Candidatos")
    headers = ["id", "pregunta", "respuesta de referencia", "documento", "titulo", "url", "fragmento", "responde", "nota"]
    widths = [10, 45, 35, 30, 35, 45, 70, 12, 30]
    for col, (header, width) in enumerate(zip(headers, widths), start=1):
        cell = sheet.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="D9D9D9")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        sheet.column_dimensions[cell.column_letter].width = width
    for row, item in enumerate(rows, start=2):
        values = [item[h] for h in headers]
        for col, value in enumerate(values, start=1):
            cell = sheet.cell(row=row, column=col, value=value)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        sheet.row_dimensions[row].height = 110
    sheet.freeze_panes = "A2"
    validation = DataValidation(type="list", formula1='"Sí,No"', allow_blank=True)
    sheet.add_data_validation(validation)
    validation.add(f"H2:H{len(rows) + 1}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    book.save(args.out)
    print(f"{len(rows)} candidatos en {args.out}")


if __name__ == "__main__":
    main()
