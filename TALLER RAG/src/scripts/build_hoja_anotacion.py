"""Genera el libro de Excel para la doble anotacion del silver v2.

El libro tiene dos hojas de anotacion con lista desplegable, una hoja que
calcula el acuerdo y el kappa de Cohen con formulas, y una hoja consolidada.
No requiere ejecutar nada para anotar ni para ver el acuerdo.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

SILVER = Path("eval/silver/silver_ambiguedad_v2.jsonl")
SALIDA = Path("eval/anotacion/anotacion_v2.xlsx")

CABECERA = PatternFill("solid", fgColor="1F3864")
CABECERA_FUENTE = Font(color="FFFFFF", bold=True)
BORDE = Border(bottom=Side(style="thin", color="BFBFBF"))
AMBAR = PatternFill("solid", fgColor="FCE4D6")
VERDE = PatternFill("solid", fgColor="E2EFDA")
NARANJA = PatternFill("solid", fgColor="F8CBAD")

COLUMNAS = [
    ("n", 5),
    ("id", 10),
    ("pregunta", 58),
    ("historial", 46),
    ("es_ambigua", 12),
    ("interpretacion_1", 32),
    ("respuesta_1", 24),
    ("interpretacion_2", 32),
    ("respuesta_2", 24),
    ("notas", 30),
]


def orden(qid: str) -> tuple[int, int, str]:
    if qid.startswith("v-r-"):
        return (4, int(qid[4:]), qid)
    if qid.startswith("v-g-"):
        return (5, int(qid[4:]), qid)
    if qid.startswith("c-"):
        return (3, int(qid[2:]), qid)
    if qid.startswith("v-"):
        return (2, int(qid[2:]), qid)
    return (1, 0, qid)


def filas() -> list[dict]:
    rows = [
        json.loads(line)
        for line in SILVER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: orden(row["id"]))
    return rows


def alto_texto(texto: str, ancho: int) -> float:
    """Altura de fila suficiente para el texto ajustado al ancho de la columna."""
    if not texto:
        return 15.0
    lineas = max(1, math.ceil(len(str(texto)) / max(1, ancho - 3)))
    return max(15.0, 13.5 * lineas + 3)


def ajustar_impresion(sheet) -> None:
    sheet.page_setup.orientation = "landscape"
    sheet.page_setup.fitToWidth = 1
    sheet.page_setup.fitToHeight = 0
    sheet.sheet_properties.pageSetUpPr.fitToPage = True
    sheet.print_title_rows = "1:1"


def hoja_anotacion(book: Workbook, nombre: str, rows: list[dict]) -> None:
    sheet = book.create_sheet(nombre)
    for index, (titulo, ancho) in enumerate(COLUMNAS, start=1):
        cell = sheet.cell(row=1, column=index, value=titulo)
        cell.fill = CABECERA
        cell.font = CABECERA_FUENTE
        cell.alignment = Alignment(horizontal="center", vertical="center")
        sheet.column_dimensions[get_column_letter(index)].width = ancho
    sheet.freeze_panes = "C2"
    sheet.row_dimensions[1].height = 22
    ajustar_impresion(sheet)

    # contador de avance, fuera de la tabla
    sheet["L1"] = "Etiquetadas"
    sheet["M1"] = f'=COUNTIF(E2:E{len(rows) + 1},"si")+COUNTIF(E2:E{len(rows) + 1},"no")'
    sheet["L2"] = "Faltan"
    sheet["M2"] = f"={len(rows)}-M1"
    for cell in ("L1", "L2"):
        sheet[cell].font = Font(bold=True)
    sheet.column_dimensions["L"].width = 12
    sheet.column_dimensions["M"].width = 8

    for numero, row in enumerate(rows, start=2):
        history = " | ".join(
            f"{'usuario' if turn.get('role') == 'user' else 'asistente'}: {turn.get('content')}"
            for turn in (row.get("history") or [])
        )
        valores = [numero - 1, row["id"], row["question"], history]
        for index, value in enumerate(valores, start=1):
            cell = sheet.cell(row=numero, column=index, value=value)
            cell.border = BORDE
            if index in (3, 4):
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        for index in range(5, len(COLUMNAS) + 1):
            cell = sheet.cell(row=numero, column=index)
            cell.border = BORDE
            if index in (6, 7, 8, 9, 10):
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        sheet.row_dimensions[numero].height = max(
            alto_texto(row["question"], 58), alto_texto(history, 46)
        )

    validacion = DataValidation(
        type="list",
        formula1='"si,no"',
        allow_blank=True,
        showDropDown=False,
        showErrorMessage=True,
        showInputMessage=True,
        errorStyle="stop",
        errorTitle="Valor no valido",
        error="Escribe si o no.",
        promptTitle="Es ambigua",
        prompt="si o no",
    )
    sheet.add_data_validation(validacion)
    validacion.add(f"E2:E{len(rows) + 1}")



def hoja_acuerdo(book: Workbook, rows: list[dict]) -> None:
    sheet = book.create_sheet("Acuerdo")
    ultimo = len(rows) + 1
    for index, (titulo, ancho) in enumerate(
        [("n", 5), ("id", 10), ("etiqueta_1", 12), ("etiqueta_2", 12), ("coincide", 10)], start=1
    ):
        cell = sheet.cell(row=1, column=index, value=titulo)
        cell.fill = CABECERA
        cell.font = CABECERA_FUENTE
        sheet.column_dimensions[get_column_letter(index)].width = ancho
    sheet.freeze_panes = "A2"
    ajustar_impresion(sheet)

    for numero in range(2, ultimo + 1):
        sheet.cell(row=numero, column=1, value=f"='Anotador 1'!A{numero}")
        sheet.cell(row=numero, column=2, value=f"='Anotador 1'!B{numero}")
        sheet.cell(row=numero, column=3, value=f"='Anotador 1'!E{numero}")
        sheet.cell(row=numero, column=4, value=f"='Anotador 2'!E{numero}")
        sheet.cell(
            row=numero,
            column=5,
            value=f'=IF(OR(C{numero}="",D{numero}=""),"",IF(C{numero}=D{numero},"si","no"))',
        )
        for columna in range(1, 6):
            sheet.cell(row=numero, column=columna).border = BORDE

    resumen = [
        ("Preguntas", f"=COUNTA(B2:B{ultimo})", None),
        ("Etiquetadas anotador 1", f'=COUNTIF(\'Anotador 1\'!E2:E{ultimo},"si")+COUNTIF(\'Anotador 1\'!E2:E{ultimo},"no")', None),
        ("Etiquetadas anotador 2", f'=COUNTIF(\'Anotador 2\'!E2:E{ultimo},"si")+COUNTIF(\'Anotador 2\'!E2:E{ultimo},"no")', None),
        ("Coinciden", f'=COUNTIF(E2:E{ultimo},"si")', None),
        ("Acuerdo observado", "=IFERROR(H5/H2,\"\")", "0.000"),
        ("«si» anotador 1", f'=COUNTIF(\'Anotador 1\'!E2:E{ultimo},"si")', None),
        ("«si» anotador 2", f'=COUNTIF(\'Anotador 2\'!E2:E{ultimo},"si")', None),
        ("Acuerdo esperado", "=(H7/H2)*(H8/H2)+((H2-H7)/H2)*((H2-H8)/H2)", "0.000"),
        ("Kappa de Cohen", '=IFERROR((H6-H9)/(1-H9),"")', "0.000"),
    ]
    sheet["G1"] = "Resumen"
    sheet["G1"].font = Font(bold=True, size=12)
    sheet["G2"], sheet["G3"], sheet["G4"], sheet["G5"] = "Preguntas", "Etiquetadas anotador 1", "Etiquetadas anotador 2", "Coinciden"
    for offset, (titulo, formula, formato) in enumerate(resumen):
        fila = 2 + offset
        sheet.cell(row=fila, column=7, value=titulo).font = Font(bold=False)
        celda = sheet.cell(row=fila, column=8, value=formula)
        if formato:
            celda.number_format = formato
    sheet.column_dimensions["G"].width = 26
    sheet.column_dimensions["H"].width = 10

    sheet.conditional_formatting.add(
        f"E2:E{ultimo}",
        CellIsRule(operator="equal", formula=['"no"'], fill=AMBAR),
    )


def hoja_consolidado(book: Workbook, rows: list[dict]) -> None:
    sheet = book.create_sheet("Consolidado")
    ultimo = len(rows) + 1
    cabeceras = [
        ("n", 5),
        ("id", 10),
        ("pregunta", 58),
        ("etiqueta", 12),
        ("interpretacion_1", 32),
        ("respuesta_1", 24),
        ("interpretacion_2", 32),
        ("respuesta_2", 24),
    ]
    for index, (titulo, ancho) in enumerate(cabeceras, start=1):
        cell = sheet.cell(row=1, column=index, value=titulo)
        cell.fill = CABECERA
        cell.font = CABECERA_FUENTE
        sheet.column_dimensions[get_column_letter(index)].width = ancho
    sheet.freeze_panes = "C2"
    ajustar_impresion(sheet)

    for numero in range(2, ultimo + 1):
        sheet.cell(row=numero, column=1, value=f"='Acuerdo'!A{numero}")
        sheet.cell(row=numero, column=2, value=f"='Acuerdo'!B{numero}")
        sheet.cell(row=numero, column=3, value=f"='Anotador 1'!C{numero}")
        sheet.cell(
            row=numero,
            column=4,
            value=f'=IF(Acuerdo!E{numero}="si",Acuerdo!C{numero},IF(Acuerdo!E{numero}="no","revisar",""))',
        )
        for columna, origen in ((5, "F"), (6, "G"), (7, "H"), (8, "I")):
            sheet.cell(row=numero, column=columna, value=f"='Anotador 1'!{origen}{numero}")
        for columna in range(1, 9):
            cell = sheet.cell(row=numero, column=columna)
            cell.border = BORDE
            if columna in (3, 5, 6, 7, 8):
                cell.alignment = Alignment(wrap_text=True, vertical="top")

    sheet.conditional_formatting.add(
        f"D2:D{ultimo}",
        CellIsRule(operator="equal", formula=['"revisar"'], fill=NARANJA),
    )


def hoja_instrucciones(book: Workbook) -> None:
    sheet = book.create_sheet("Instrucciones", 0)
    sheet.column_dimensions["A"].width = 110
    ajustar_impresion(sheet)
    lineas = [
        ("Anotación humana del silver v2 de ambigüedad", True),
        ("", False),
        ("Una pregunta es ambigua si, con el corpus del proyecto, admite más de una respuesta válida o más de una interpretación razonable.", False),
        ("No es ambigua si una persona con acceso al corpus puede responderla sin preguntar nada más.", False),
        ("", False),
        ("Cómo anotar", True),
        ("1. Abre tu hoja: «Anotador 1» o «Anotador 2». No mires la del otro.", False),
        ("2. En la columna es_ambigua elige si o no en la lista desplegable.", False),
        ("3. Si es ambigua, escribe dos interpretaciones y una respuesta corta para cada una.", False),
        ("4. Si no es ambigua, deja las interpretaciones vacías.", False),
        ("5. Usa notas solo para dudas que el otro anotador deba conocer.", False),
        ("", False),
        ("Las columnas de respuesta empiezan vacías a propósito: el libro es el formulario. El contador «Etiquetadas» de la derecha muestra tu avance.", False),
        ("La hoja «Acuerdo» calcula sola las coincidencias, el acuerdo observado y el kappa de Cohen.", False),
        ("La hoja «Consolidado» marca «revisar» en las preguntas donde las dos etiquetas no coinciden.", False),
        ("", False),
        ("Cuentan como ambiguas: referencia genérica sin entidad, unidad o escala sin especificar, comparación sin criterio, sujeto omitido en una conversación y alcance temporal o institucional.", False),
        ("No cuentan: la pregunta nombra la entidad con un nombre propio, una fecha o un lugar que la identifica.", False),
    ]
    for fila, (texto, negrita) in enumerate(lineas, start=1):
        cell = sheet.cell(row=fila, column=1, value=texto)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        sheet.row_dimensions[fila].height = alto_texto(texto, 110)
        if negrita:
            cell.font = Font(bold=True, size=13 if fila == 1 else 11)


def main() -> None:
    rows = filas()
    book = Workbook()
    book.remove(book.active)
    hoja_instrucciones(book)
    hoja_anotacion(book, "Anotador 1", rows)
    hoja_anotacion(book, "Anotador 2", rows)
    hoja_acuerdo(book, rows)
    hoja_consolidado(book, rows)
    book.active = 0
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    book.save(SALIDA)
    print(f"escrito {SALIDA} con {len(rows)} preguntas y {len(book.sheetnames)} hojas")


if __name__ == "__main__":
    main()
