"""Genera el Excel ciego de anotación de E4 para un anotador externo.

El libro no incluye el modelo que respondió. El mapeo queda en
eval/annotations/key.csv y se usa después para calcular kappa y medias.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from src.rag.annotation import EJES, blind_id, load_items, write_key

HEADERS = [
    "item_id",
    "pregunta",
    "respuesta de referencia",
    "span de referencia",
    "criterio",
    "respuesta del modelo",
    "evidencias",
    *EJES,
    "nota",
]
WIDTHS = [12, 45, 35, 35, 30, 55, 60, 12, 12, 12, 12, 30]
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def safe(value: object, limit: int | None = None) -> str:
    text = CONTROL.sub(" ", str(value or ""))
    if limit:
        text = text[:limit]
    return f"'{text}" if text.startswith("=") else text


def evidence_text(response: dict) -> str:
    parts = []
    for i, hit in enumerate(response.get("evidence", []), start=1):
        parts.append(
            f"[{i}] {hit.get('title')} ({hit.get('language')}, {hit.get('created_at')})\n"
            f"{safe(hit.get('text'), 600)}\n{hit.get('url')}"
        )
    return safe("\n\n".join(parts), 3000)


def build_workbook(gold: dict, records: list[dict]) -> Workbook:
    book = Workbook()
    guide = book.active
    guide.title = "Instrucciones"
    guide.column_dimensions["A"].width = 110
    lines = [
        ("Anotación humana de respuestas RAG", True),
        ("", False),
        ("Este libro tiene 76 respuestas de un asistente RAG. No se indica qué modelo respondió cada una.", False),
        ("Califica cada respuesta en la hoja 'Anotacion', en las cuatro columnas finales.", False),
        ("Cada eje va de 0 a 3. Usa la lista desplegable; no escribas otro valor.", False),
        ("", False),
        ("Rúbrica", True),
        ("Corrección: 3 responde lo que pide la pregunta; 2 correcta con imprecisión; 1 parcial o con error relevante; 0 incorrecta o no responde.", False),
        ("Fidelidad: 3 todo el contenido sale de la evidencia; 2 algún añadido menor; 1 inventa o contradice parte; 0 inventa el núcleo.", False),
        ("Citas: 3 todas válidas y apuntan al pasaje que sostiene la respuesta; 2 válidas pero no al pasaje; 1 parciales; 0 sin citas o todas inválidas.", False),
        ("Abstención: 3 responde cuando hay evidencia; 2 responde con duda razonable; 1 se abstiene teniendo evidencia clara; 0 se abstiene sin motivo.", False),
        ("", False),
        ("Reglas", True),
        ("Juzga contra la evidencia y la respuesta de referencia, no contra tu conocimiento del tema.", False),
        ("No compares respuestas entre sí ni intentes adivinar el modelo.", False),
        ("No cambies el orden de las filas ni el item_id.", False),
        ("Si dudas, usa la nota y elige el valor que mejor describa el texto.", False),
        ("", False),
        ("Al terminar, guarda el archivo con tu nombre y devuélvelo. Tiempo estimado: 45 a 60 minutos.", False),
    ]
    for row, (text, bold) in enumerate(lines, start=1):
        cell = guide.cell(row=row, column=1, value=text)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        if bold:
            cell.font = Font(bold=True, size=12)

    sheet = book.create_sheet("Anotacion")
    header_fill = PatternFill("solid", fgColor="D9D9D9")
    for col, (header, width) in enumerate(zip(HEADERS, WIDTHS), start=1):
        cell = sheet.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        sheet.column_dimensions[cell.column_letter].width = width

    for row, record in enumerate(records, start=2):
        question = gold.get(record["question_id"], {})
        response = record["response"]
        values = [
            record["item_id"],
            question.get("question", record["question_id"]),
            question.get("reference_answer", ""),
            question.get("reference_span", ""),
            question.get("criteria", ""),
            response.get("answer", ""),
            evidence_text(response),
            None,
            None,
            None,
            None,
            None,
        ]
        for col, value in enumerate(values, start=1):
            cell = sheet.cell(row=row, column=col, value=safe(value) if value is not None else None)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        sheet.row_dimensions[row].height = 120

    sheet.freeze_panes = "A2"
    validation = DataValidation(type="list", formula1='"0,1,2,3"', allow_blank=True)
    sheet.add_data_validation(validation)
    first = 8
    validation.add(f"{sheet.cell(row=2, column=first).column_letter}2:{sheet.cell(row=len(records) + 1, column=first + 3).column_letter}{len(records) + 1}")
    return book


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("eval/annotations/Julian_Carrillo.xlsx"))
    parser.add_argument("--run", type=Path, default=Path("reports/corridas/e4_test.jsonl"))
    parser.add_argument("--gold", type=Path, default=Path("eval/gold/silver_standard.jsonl"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("eval/annotations"))
    args = parser.parse_args()

    gold, records = load_items(args.run, args.gold)
    for record in records:
        record["item_id"] = blind_id(record["model"], record["question_id"], int(record["rep"]))
    write_key(records, args.annotations_dir)

    book = build_workbook(gold, records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    book.save(args.out)
    print(f"{len(records)} respuestas en {args.out}")


if __name__ == "__main__":
    main()
