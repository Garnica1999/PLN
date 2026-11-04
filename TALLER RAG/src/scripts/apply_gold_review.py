"""Consolida la revisión de Carlos sobre el silver standard.

Aplica las 12 correcciones que Carlos dejó en la columna comentario de
silver_standard_carlos.xlsx, marca las 43 filas como revisadas y regenera
JSONL, CSV y Markdown. No vuelve a generar el gold desde los archivos fuente.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from openpyxl import load_workbook

CORRECTIONS: dict[str, dict] = {
    "g-003": {
        "question": "¿Qué opciones estudiarán los especialistas de Vantara para manejar el crecimiento de la población de hipopótamos?",
        "reference_answer": "El traslado de algunos ejemplares a India y la creación de un santuario de conservación.",
        "reference_span": "Entre las opciones que estudiarán figuran el traslado de algunos ejemplares a ese país y la creación de un santuario de conservación para el manejo de esta especie invasora.",
    },
    "g-009": {
        "reference_span": "Costa Rica recibirá también un insumo económico de unos 1,4 millones de dólares.",
    },
    "g-012": {
        "reference_span": "La implementación de estos dispositivos tiene como objetivo mitigar el impacto de los residuos sólidos flotantes, mejorar el control ambiental y aportar a la protección del manglar, uno de los ecosistemas estratégicos de Barranquilla.",
    },
    "g-022": {
        "reference_url": "http://web.archive.org/web/20250615013203/https://extra.com.co/onu-aprueba-resolucion-para-proteger-oceanos-internacionales/",
    },
    "g-025": {
        "reference_url": "http://web.archive.org/web/20250524005725/https://extra.com.co/el-valle-del-cauca-lanza-observatorio-contra-la-discriminacion-etnico-racial-y-red-etnica-de-ciencia-e-innovacion/",
    },
    "g-031": {
        "reference_span": "En el sur del Tolima, Chaparral se impone como un municipio de gran extensión y tradición agropecuaria.",
    },
    "g-032": {
        "reference_url": "http://web.archive.org/web/20250930155243/https://extra.com.co/cientificos-modifican-plantas-geneticamente-para-aumentar-la-absorcion-de-co%e2%82%82/",
    },
    "g-036": {
        "question": "¿Entre qué años tuvo su primera circulación la moneda de 1 000 pesos?",
        "reference_answer": "Entre 1996 y 2002.",
        "type": "directa",
    },
    "g-039": {
        "lang": "es",
    },
    "g-040": {
        "reference_span": "La semana pasada, la estructura Comandos de la Frontera, una estructura de esa disidencia, atacó con drones cargados con explosivos a los uniformados del Grupo de Caballería Meridiano #3 del Ejército.",
    },
    "g-042": {
        "question": "¿En qué año Petro reiteró su llamado a la movilización permanente y dijo que el proceso de transformación social no ha concluido?",
        "reference_span": "En su cierre, Petro reiteró su llamado a la movilización permanente durante el 2025 y enfatizó que el proceso de transformación social no ha concluido.",
    },
    "g-047": {
        "reference_span": "El Centro Wilson anunció hoy martes la incorporación como miembro distinguido del expresidente colombiano Iván Duque, a quien describió como un defensor de la democracia y la biodiversidad.",
    },
}

COLUMNS = [
    "id",
    "split",
    "type",
    "lang",
    "source",
    "question",
    "reference_answer",
    "reference_doc_ids",
    "reference_url",
    "reference_span",
    "criteria",
    "reviewer",
    "validated",
    "review_note",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer-xlsx", default="eval/gold/silver_standard_carlos.xlsx")
    parser.add_argument("--gold", default="eval/gold/silver_standard.jsonl")
    parser.add_argument("--parquet", default="data/documentos_rag.parquet")
    parser.add_argument("--outdir", default="eval/gold")
    parser.add_argument("--reviewer", default="Carlos")
    args = parser.parse_args()

    sheet = load_workbook(args.reviewer_xlsx)["Gold"]
    marks = {}
    for row in sheet.iter_rows(min_row=2, values_only=True):
        if row[0]:
            marks[row[0]] = (row[11] or "", row[12] or "")

    records = [json.loads(line) for line in Path(args.gold).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {record["id"]: record for record in records}
    doc_ids = {record["id"]: record["reference_doc_ids"][0] for record in records if record["reference_doc_ids"]}
    meta = pq.read_table(args.parquet, columns=["document_id", "text"]).to_pandas()
    texts = dict(zip(meta["document_id"], meta["text"]))

    corrected = 0
    for gid, changes in CORRECTIONS.items():
        record = by_id[gid]
        record.update(changes)
        if "reference_span" in changes:
            assert normalize(changes["reference_span"]) in normalize(texts[doc_ids[gid]]), f"span no encontrado en {gid}"
        corrected += 1

    validated = 0
    for record in records:
        mark, comment = marks.get(record["id"], ("", ""))
        record["reviewer"] = args.reviewer
        record["validated"] = mark in ("Sí", "Corregir")
        if comment:
            record["review_note"] = comment
        if record["validated"]:
            validated += 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "silver_standard.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    flat = pd.DataFrame(records)
    flat["reference_doc_ids"] = flat["reference_doc_ids"].map(lambda ids: ";".join(ids) if isinstance(ids, list) else "")
    flat[COLUMNS].to_csv(outdir / "silver_standard.csv", index=False)

    lines = [
        "# Silver standard",
        "",
        f"Revisión de {args.reviewer}: {validated} de {len(records)} filas validadas. La segunda revisión está pendiente.",
        "",
        "| id | split | tipo | idioma | pregunta | respuesta de referencia | documento | revisor | validado |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for record in records:
        document = record["reference_doc_ids"][0] if record["reference_doc_ids"] else ""
        lines.append(
            f"| {record['id']} | {record['split']} | {record['type']} | {record['lang']} | "
            f"{record['question']} | {record['reference_answer']} | {document} | "
            f"{record['reviewer']} | {'sí' if record['validated'] else ''} |"
        )
    (outdir / "silver_standard.md").write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "validadas": validated,
                "corregidas": corrected,
                "total": len(records),
                "sin_marca": sum(1 for record in records if record["id"] not in marks),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
