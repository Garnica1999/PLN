"""Aplica la segunda revisión del gold (Julián) al JSONL consolidado.

Solo aplica cambios respaldados por el texto indexado: los siete spans y la
reformulación de g-017. g-036 y g-045 quedan como están, con nota, porque el
documento respalda el valor actual. Las tres preguntas con URL 404 se marcan
como no verificadas.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from openpyxl import load_workbook

GOLD = Path("eval/gold/silver_standard.jsonl")
BACKUP = Path("eval/gold/silver_standard_pre_julian.jsonl")
REVIEW = Path("eval/gold/silver_standard_julian.xlsx")

SPAN_IDS = {"g-009", "g-012", "g-016", "g-031", "g-040", "g-042", "g-047"}
REFORMULADA = {
    "g-017": "¿Qué porcentaje del territorio marítimo de la Unión Europea está sometido a un régimen de protección estricto?"
}
NO_APLICADAS = {
    "g-036": "El span respalda 'entre 1996 y 2002'; la propuesta de pregunta y respuesta es inconsistente.",
    "g-045": "El documento indexado dice 'near Kendal' y no menciona Tebay.",
}
SIN_VERIFICAR = {"g-022", "g-025", "g-032"}


def main() -> None:
    sheet = load_workbook(REVIEW, data_only=True)["Gold"]
    headers = [cell.value for cell in sheet[1]]
    review = {
        row[0]: dict(zip(headers, row))
        for row in sheet.iter_rows(min_row=2, values_only=True)
        if row[0]
    }
    rows = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not BACKUP.exists():
        shutil.copy2(GOLD, BACKUP)

    for row in rows:
        qid = row["id"]
        propuesta = review[qid]
        row["revisor_2"] = "Julian"
        row["segunda_revision"] = {
            "Sí": "si",
            "No": "no_aplicada",
            "Corregir": "corregida",
        }.get(propuesta["validado"], "sin_verificar")
        row["segunda_revision_nota"] = propuesta["comentario"] or ""
        if qid in SPAN_IDS:
            row["reference_span"] = propuesta["fragmento en el documento"]
            row["segunda_revision"] = "corregida"
        elif qid in REFORMULADA:
            row["question"] = REFORMULADA[qid]
            row["segunda_revision"] = "corregida"
        elif qid in NO_APLICADAS:
            row["segunda_revision"] = "no_aplicada"
            row["segunda_revision_nota"] = NO_APLICADAS[qid]
        elif qid in SIN_VERIFICAR:
            row["segunda_revision"] = "sin_verificar"

    GOLD.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"{len(rows)} preguntas actualizadas")
    for estado in ("si", "corregida", "no_aplicada", "sin_verificar"):
        print(estado, sum(1 for row in rows if row["segunda_revision"] == estado))


if __name__ == "__main__":
    main()
