"""Exporta el silver standard a Excel con una sola columna de revisor.

Cada persona genera su propia copia con --reviewer. Así conviven varias
versiones del mismo archivo sin columnas duplicadas.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.properties import PageSetupProperties

SILVER = Path("eval/silver/silver_standard.jsonl")
OUT = Path("eval/silver/silver_standard.xlsx")

HEADER_FILL = PatternFill("solid", fgColor="335C81")
HEADER_FONT = Font(bold=True, color="FFFFFF", size=11)
BODY_FONT = Font(size=11)
LINK_FONT = Font(size=11, color="0563C1", underline="single")
MUTED_FONT = Font(size=11, color="6B6B6B", italic=True)
THIN = Side(style="thin", color="D0D0D0")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP_TOP = Alignment(wrap_text=True, vertical="top")
CENTER = Alignment(horizontal="center", vertical="top")
SIN_RESPUESTA_FILL = PatternFill("solid", fgColor="FFF2CC")
OK_FILL = PatternFill("solid", fgColor="D9EAD3")
FIX_FILL = PatternFill("solid", fgColor="FCE5CD")
NO_FILL = PatternFill("solid", fgColor="E7E6E6")

HEADERS = [
    ("id", 9),
    ("split", 8),
    ("tipo", 13),
    ("idioma", 8),
    ("fuente", 12),
    ("pregunta", 55),
    ("respuesta de referencia", 45),
    ("documento", 17),
    ("medio", 22),
    ("fragmento en el documento", 60),
    ("revisor", 16),
    ("validado", 11),
    ("comentario", 35),
]

SOURCE_LABEL = {"url_crawls": "prensa", "bluesky_posts": "Bluesky", "sin_respuesta": "sin respuesta"}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def build_instructions(ws, reviewer: str, records: list[dict]) -> None:
    ws.column_dimensions["A"].width = 112
    subtitle = (
        f"Copia de {reviewer}. Escribe tu nombre en la columna revisor si aún no está."
        if reviewer
        else "Una sola columna revisor. Cada persona genera su propia copia del archivo."
    )
    total = len(records)
    dev = sum(1 for record in records if record["split"] == "dev")
    test = sum(1 for record in records if record["split"] == "test")
    answerable = sum(1 for record in records if record["type"] != "sin_respuesta")
    no_answer = total - answerable
    press = sum(1 for record in records if record["source"] == "url_crawls")
    bluesky = sum(1 for record in records if record["source"] == "bluesky_posts")
    spanish = sum(1 for record in records if record["lang"] == "es")
    english = sum(1 for record in records if record["lang"] == "en")
    rows = [
        ("Silver standard del Taller RAG", Font(bold=True, size=16)),
        (f"{total} preguntas para revisar. {subtitle}", BODY_FONT),
        ("No modifiques la columna id.", BODY_FONT),
        ("", BODY_FONT),
        ("Qué hay en la hoja Silver", Font(bold=True, size=12)),
        (f"{dev} preguntas de desarrollo y {test} de prueba. {answerable} respondibles y {no_answer} sin respuesta.", BODY_FONT),
        (f"{spanish} en español, {english} en inglés. {press} de prensa y {bluesky} de Bluesky.", BODY_FONT),
        ("", BODY_FONT),
        ("Cómo revisar cada fila", Font(bold=True, size=12)),
        ("1. Abre el enlace de la columna documento.", BODY_FONT),
        ("2. Comprueba que la pregunta se responde con ese documento y nada más.", BODY_FONT),
        ("3. Comprueba que la respuesta de referencia contesta la pregunta.", BODY_FONT),
        ("4. Comprueba que el fragmento aparece literal en el documento.", BODY_FONT),
        ("5. Escribe tu nombre en revisor y marca validado con Sí, No o Corregir.", BODY_FONT),
        ("", BODY_FONT),
        ("Qué significa cada marca", Font(bold=True, size=12)),
        ("Sí: la pregunta, la respuesta, el fragmento y el enlace coinciden.", BODY_FONT),
        ("Corregir: la fila sirve pero la respuesta o el fragmento deben cambiarse.", BODY_FONT),
        ("No: la pregunta es ambigua, no se responde con el documento o está duplicada.", BODY_FONT),
        ("", BODY_FONT),
        ("Preguntas sin respuesta", Font(bold=True, size=12)),
        ("Las 5 filas con tipo sin_respuesta no tienen documento. Verifica que ningún documento del corpus las responda.", BODY_FONT),
        ("No se evalúan con Recall@k: se evalúan por abstención.", BODY_FONT),
        ("", BODY_FONT),
        ("Varias revisiones", Font(bold=True, size=12)),
        ("Cada persona crea su copia con python -m src.scripts.export_silver_excel --reviewer NOMBRE.", BODY_FONT),
        ("Así hay un archivo por revisor, todos con la misma estructura y una sola columna revisor.", BODY_FONT),
        ("Al final se comparan las copias: una fila es válida cuando dos revisores marcan Sí.", BODY_FONT),
        ("", BODY_FONT),
        ("Colores", Font(bold=True, size=12)),
        ("Amarillo: pregunta sin respuesta (no tiene documento).", BODY_FONT),
        ("Verde: validado con Sí. Naranja: Corregir. Gris: No.", BODY_FONT),
        ("", BODY_FONT),
        ("Regenerar", Font(bold=True, size=12)),
        ("python -m src.scripts.build_silver_standard ... y luego python -m src.scripts.export_silver_excel --reviewer NOMBRE", BODY_FONT),
    ]
    for index, (text, font) in enumerate(rows, start=1):
        cell = ws.cell(row=index, column=1, value=text)
        cell.font = font
        cell.alignment = Alignment(wrap_text=True, vertical="top")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer", default="", help="Nombre del revisor para esta copia")
    parser.add_argument("--silver", default=str(SILVER))
    parser.add_argument("--ignore-validation", action="store_true", help="Deja validado vacío para otra persona")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.silver).read_text(encoding="utf-8").splitlines() if line.strip()]
    reviewer = args.reviewer.strip()
    out = Path(args.out) if args.out else (
        Path(f"eval/silver/silver_standard_{slug(reviewer)}.xlsx") if reviewer else OUT
    )

    workbook = Workbook()
    instructions = workbook.active
    instructions.title = "Instrucciones"
    build_instructions(instructions, reviewer, records)

    sheet = workbook.create_sheet("Silver")
    for column, (title, width) in enumerate(HEADERS, start=1):
        cell = sheet.cell(row=1, column=column, value=title)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER
        sheet.column_dimensions[cell.column_letter].width = width
    sheet.row_dimensions[1].height = 30

    for row_index, record in enumerate(records, start=2):
        url = record.get("reference_url") or ""
        domain = urlparse(url).netloc.removeprefix("www.") if url else ""
        source = SOURCE_LABEL.get(record.get("source", ""), record.get("source", ""))
        values = [
            record["id"],
            record["split"],
            record["type"],
            record["lang"],
            source,
            record["question"],
            record["reference_answer"],
            "abrir documento" if url else "sin documento",
            domain,
            record["reference_span"],
            reviewer or record.get("reviewer", "") or "",
            "" if args.ignore_validation else ("Sí" if record.get("validated") else ""),
            record.get("review_note", ""),
        ]
        for column, value in enumerate(values, start=1):
            cell = sheet.cell(row=row_index, column=column, value=value)
            cell.font = BODY_FONT
            cell.border = BORDER
            cell.alignment = WRAP_TOP if column in (6, 7, 10, 13) else CENTER
        link = sheet.cell(row=row_index, column=8)
        if url:
            link.hyperlink = url
            link.font = LINK_FONT
        else:
            link.font = MUTED_FONT
        if record["type"] == "sin_respuesta":
            for column in range(1, len(HEADERS) + 1):
                sheet.cell(row=row_index, column=column).fill = SIN_RESPUESTA_FILL
        sheet.row_dimensions[row_index].height = 78

    last_row = len(records) + 1
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = f"A1:M{last_row}"

    validation = DataValidation(type="list", formula1='"Sí,No,Corregir"', allow_blank=True)
    validation.error = "Elige Sí, No o Corregir."
    validation.errorTitle = "Valor no válido"
    sheet.add_data_validation(validation)
    validation.add(f"L2:L{last_row}")

    for value, fill in (("Sí", OK_FILL), ("Corregir", FIX_FILL), ("No", NO_FILL)):
        sheet.conditional_formatting.add(
            f"L2:L{last_row}",
            CellIsRule(operator="equal", formula=[f'"{value}"'], fill=fill),
        )

    summary = workbook.create_sheet("Resumen")
    summary.column_dimensions["A"].width = 32
    summary.column_dimensions["B"].width = 12
    summary["A1"] = "Resumen del silver"
    summary["A1"].font = Font(bold=True, size=14)
    summary["A2"] = "Revisor"
    summary["A2"].font = Font(bold=True)
    summary["B2"] = reviewer or "(sin asignar)"
    counts = {
        "Total": len(records),
        "Desarrollo": sum(1 for r in records if r["split"] == "dev"),
        "Prueba": sum(1 for r in records if r["split"] == "test"),
        "Respondibles": sum(1 for r in records if r["type"] != "sin_respuesta"),
        "Sin respuesta": sum(1 for r in records if r["type"] == "sin_respuesta"),
        "Directas": sum(1 for r in records if r["type"] == "directa"),
        "Ambiguas": sum(1 for r in records if r["type"] == "ambigua"),
        "Español": sum(1 for r in records if r["lang"] == "es"),
        "Inglés": sum(1 for r in records if r["lang"] == "en"),
        "Prensa": sum(1 for r in records if r["source"] == "url_crawls"),
        "Bluesky": sum(1 for r in records if r["source"] == "bluesky_posts"),
    }
    for index, (label, value) in enumerate(counts.items(), start=4):
        summary.cell(row=index, column=1, value=label).font = BODY_FONT
        summary.cell(row=index, column=2, value=value).font = Font(bold=True)
    start = len(counts) + 6
    summary.cell(row=start, column=1, value="Avance de esta copia").font = Font(bold=True, size=12)
    summary.cell(row=start + 1, column=1, value="Filas asignadas").font = BODY_FONT
    summary.cell(row=start + 1, column=2, value=f'=COUNTA(Silver!K2:K{last_row})')
    summary.cell(row=start + 2, column=1, value="Validadas con Sí").font = BODY_FONT
    summary.cell(row=start + 2, column=2, value=f'=COUNTIF(Silver!L2:L{last_row},"Sí")')
    summary.cell(row=start + 3, column=1, value="Pendientes").font = BODY_FONT
    summary.cell(row=start + 3, column=2, value=f'=COUNTA(Silver!A2:A{last_row})-COUNTIF(Silver!L2:L{last_row},"Sí")')

    for worksheet in (instructions, sheet, summary):
        worksheet.page_setup.orientation = "landscape"
        worksheet.page_setup.fitToWidth = 1
        worksheet.page_setup.fitToHeight = 0
        worksheet.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(out)
    print(json.dumps({"archivo": str(out), "revisor": reviewer, "filas": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
