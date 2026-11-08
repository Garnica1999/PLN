"""Consolida la tercera revision del silver (Oscar).

Aplica solo los spans que se pueden verificar literalmente en el texto indexado.
Las propuestas sin respaldo o ya resueltas por revisiones anteriores quedan con
nota. Tambien registra el estado de las cinco preguntas sin respuesta.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from openpyxl import load_workbook

SILVER_FILES = [
    Path("eval/silver/silver_standard.jsonl"),
    Path("eval/silver/silver_standard_ampliado.jsonl"),
]
REVIEW = Path("eval/silver/silver_standard_oscar.xlsx")
CORPUS = Path("Dataset/documentos_rag.parquet")

SPANS = {
    "g-009": "Costa Rica recibirá también un insumo económico de unos 1,4 millones de dólares.",
    "g-012": (
        "La implementación de estos dispositivos tiene como objetivo mitigar el impacto de los "
        "residuos sólidos flotantes, mejorar el control ambiental y aportar a la protección del "
        "manglar, uno de los ecosistemas estratégicos de Barranquilla."
    ),
    "g-028": (
        "Thanks to NHSF for inviting me to speak at today’s Strategic Framework for Heritage "
        "Science in the UK launch event"
    ),
    "g-031": "En el sur del Tolima, Chaparral se impone como un municipio de gran extensión y tradición agropecuaria.",
    "g-042": (
        "En su cierre, Petro reiteró su llamado a la movilización permanente durante el 2025 y "
        "enfatizó que el proceso de transformación social no ha concluido."
    ),
}

NO_APLICADAS = {
    "g-003": "El span actual ya sostiene la respuesta: habla de las opciones que estudiarán, no de medidas tomadas.",
    "g-011": "El span es literal y dice «dieciséis rutas de avistamiento»; la respuesta usa el número 16.",
    "g-016": "El span sostiene «oficialmente, solo 24 horas»; la pregunta es telegráfica pero respondible.",
    "g-017": "El 11% es el estatus general de protección; el 1% es el régimen estricto que pide la pregunta.",
    "g-048": "Pregunta sin respuesta: no tiene documento de referencia que validar.",
    "g-049": "Pregunta sin respuesta: no tiene documento de referencia que validar.",
    "g-050": "Pregunta sin respuesta: no tiene documento de referencia que validar.",
    "g-051": "Pregunta sin respuesta: no tiene documento de referencia que validar.",
    "g-052": "Pregunta sin respuesta: no tiene documento de referencia que validar.",
}

SIN_VERIFICAR = {
    "g-022": "El enlace de archivo no abre; el documento sí está indexado y el span es literal.",
    "g-025": "El enlace de archivo no abre; el documento sí está indexado y el span es literal.",
    "g-032": "El enlace de archivo no abre; el documento sí está indexado y el span es literal.",
}


def load_review() -> dict[str, dict]:
    sheet = load_workbook(REVIEW, data_only=True)["Silver"]
    headers = [cell.value for cell in sheet[1]]
    return {
        row[0]: dict(zip(headers, row))
        for row in sheet.iter_rows(min_row=2, values_only=True)
        if row[0]
    }


def verify_spans(rows: list[dict]) -> None:
    docs = {row["id"]: row["reference_doc_ids"][0] for row in rows if row["id"] in SPANS}
    table = pq.read_table(
        CORPUS, columns=["document_id", "text"], filters=[("document_id", "in", list(docs.values()))]
    )
    by_id = dict(zip(table["document_id"].to_pylist(), table["text"].to_pylist()))
    for gid, span in SPANS.items():
        text = by_id[docs[gid]]
        if span not in text:
            raise SystemExit(f"el span de {gid} no aparece literal en el documento")


def main() -> None:
    review = load_review()
    first = [json.loads(line) for line in SILVER_FILES[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    verify_spans(first)

    for path in SILVER_FILES:
        backup = path.with_name(path.stem + "_pre_oscar.jsonl")
        if not backup.exists():
            shutil.copy2(path, backup)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            propuesta = review[row["id"]]
            row["revisor_3"] = "Oscar"
            estado = {
                "Sí": "si",
                "No": "no_aplicada",
                "Corregir": "corregida",
            }.get(propuesta.get("validado"), "sin_verificar")
            row["tercera_revision"] = estado
            row["tercera_revision_nota"] = propuesta.get("comentario") or ""
            if row["id"] in SPANS:
                row["reference_span"] = SPANS[row["id"]]
                row["tercera_revision"] = "corregida"
            elif row["id"] in NO_APLICADAS:
                row["tercera_revision"] = "no_aplicada"
                row["tercera_revision_nota"] = NO_APLICADAS[row["id"]]
            elif row["id"] in SIN_VERIFICAR:
                row["tercera_revision"] = "sin_verificar"
                row["tercera_revision_nota"] = SIN_VERIFICAR[row["id"]]
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        print(f"{path}: {len(rows)} preguntas")
        for estado in ("si", "corregida", "no_aplicada", "sin_verificar"):
            print(f"  {estado}: {sum(1 for row in rows if row['tercera_revision'] == estado)}")


if __name__ == "__main__":
    main()
