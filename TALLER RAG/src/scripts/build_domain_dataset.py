"""Construye el set de 120 preguntas para la compuerta de dominio.

60 de dominio: 38 directas del gold, 11 variantes de entidades del gold y 11
nuevas sacadas de titulos del corpus. 60 de fuera: 4 del chat, 5 sin respuesta
del gold y 51 de conocimiento general.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import requests

API = os.getenv("API_URL", "http://localhost:8010").rstrip("/")
COLLECTION = os.getenv("COLLECTION", "rag_full_bge_m3_d1024_v1")
GOLD = Path("eval/gold/gold_standard_ampliado.jsonl")
OUT_JSONL = Path("eval/domain/domain_questions.jsonl")
OUT_CSV = Path("eval/domain/domain_questions.csv")

VARIANTS = [
    ("¿Qué especie invasora afecta los páramos de Boyacá?", "retamo"),
    ("¿Entre qué años tuvo su primera circulación la moneda de 1.000 pesos?", "moneda"),
    ("¿Cuántos hipopótamos hay en la cuenca del Magdalena?", "hipopotamos"),
    ("¿Qué empresa colombiana invertirá en proyectos piloto de fracking?", "ecopetrol"),
    ("¿Qué país aprobó el Convenio Internacional del Cacao?", "cacao"),
    ("¿Qué argumentan los senadores demócratas sobre los precios de McDonald's?", "mcdonalds"),
    ("¿Qué busca el Observatorio Contra la Discriminación Étnico-Racial del Valle del Cauca?", "observatorio"),
    ("¿Por qué México prohibió la siembra de maíz transgénico blanco?", "maiz"),
    ("¿Quién era el guardaparque asesinado en Güican, Boyacá?", "guican"),
    ("¿Qué es Extinction Rebellion y dónde planea protestar?", "extinction"),
    ("¿Qué es el Tratado de Alta Mar y cuándo entra en vigor?", "tratado"),
]

NEW_DOMAIN = [
    ("¿Qué aprobó la Cámara de Representantes sobre la Paz Total?", "paz_total"),
    ("¿Cuál es la cotización del euro en Colombia?", "euro"),
    ("¿Cuánto se aceleró la inflación en Colombia?", "inflacion"),
    ("¿Qué ofrece Colombia como destino turístico?", "turismo"),
    ("¿Qué polémica arbitral hubo en el partido Junior vs Fortaleza?", "futbol"),
    ("¿Qué se dice sobre la inversión en ciencia en Colombia?", "ciencia"),
    ("¿Dónde se inició el encuentro de energías renovables?", "energia"),
    ("¿Qué exigen los manifestantes en Colombia?", "protestas"),
    ("¿Qué amenaza las elecciones en Colombia?", "elecciones"),
    ("¿Cómo afecta el cambio climático a la salud en Colombia?", "salud"),
    ("¿Qué se dice en Bluesky sobre políticas de IA en Colombia?", "ia"),
]

CHAT_OUTSIDE = [
    ("¿Cuál es la capital de Francia?", "capital_francia"),
    ("¿Cuál es la población de Tokio?", "poblacion_tokio"),
    ("¿Quién escribió Cien años de soledad?", "literatura_cien_anos"),
    ("¿Quién ganó el mundial de fútbol de 2022?", "deportes_mundial"),
]

OUTSIDE = [
    ("¿Cuál es la capital de Italia?", "capitales"),
    ("¿Cuál es la capital de Japón?", "capitales"),
    ("¿Cuál es la capital de Australia?", "capitales"),
    ("¿Cuál es la capital de Canadá?", "capitales"),
    ("¿Cuál es la capital de Egipto?", "capitales"),
    ("¿Cuál es la capital de Brasil?", "capitales"),
    ("¿Cuál es la capital de India?", "capitales"),
    ("¿Cuál es la capital de Noruega?", "capitales"),
    ("¿Cuántos habitantes tiene Ciudad de México?", "poblacion"),
    ("¿Cuántos habitantes tiene El Cairo?", "poblacion"),
    ("¿Cuántos habitantes tiene Lagos?", "poblacion"),
    ("¿Cuántos habitantes tiene Londres?", "poblacion"),
    ("¿Cuántos habitantes tiene Nueva York?", "poblacion"),
    ("¿Cuántos habitantes tiene São Paulo?", "poblacion"),
    ("¿Cuántos habitantes tiene Delhi?", "poblacion"),
    ("¿Cuántos habitantes tiene Shanghái?", "poblacion"),
    ("¿Quién escribió Don Quijote?", "literatura"),
    ("¿Quién escribió La Odisea?", "literatura"),
    ("¿Quién escribió Hamlet?", "literatura"),
    ("¿Quién escribió Orgullo y prejuicio?", "literatura"),
    ("¿Quién escribió El Principito?", "literatura"),
    ("¿Quién escribió Rayuela?", "literatura"),
    ("¿Quién escribió Fahrenheit 451?", "literatura"),
    ("¿Quién escribió 1984?", "literatura"),
    ("¿Cuál es la fórmula del agua?", "ciencia_general"),
    ("¿Cuál es la velocidad de la luz?", "ciencia_general"),
    ("¿Cuál es el planeta más grande del sistema solar?", "ciencia_general"),
    ("¿Qué es un gas noble?", "ciencia_general"),
    ("¿Qué es el ADN?", "ciencia_general"),
    ("¿Qué es la gravedad?", "ciencia_general"),
    ("¿Cuántos elementos tiene la tabla periódica?", "ciencia_general"),
    ("¿Qué es un agujero negro?", "ciencia_general"),
    ("¿Cuándo cayó el Muro de Berlín?", "historia"),
    ("¿Cuándo empezó la Revolución Francesa?", "historia"),
    ("¿Cuándo se independizó Colombia?", "historia"),
    ("¿Cuándo terminó la Segunda Guerra Mundial?", "historia"),
    ("¿Cuándo cayó el Imperio Romano?", "historia"),
    ("¿Cuándo cayó Constantinopla?", "historia"),
    ("¿Cuándo terminó la Guerra Fría?", "historia"),
    ("¿Quién ganó el Balón de Oro 2023?", "deportes"),
    ("¿Quién ganó la NBA 2024?", "deportes"),
    ("¿Quién ganó el Tour de Francia 2024?", "deportes"),
    ("¿Quién ganó Wimbledon 2024?", "deportes"),
    ("¿Quién ganó el Super Bowl 2024?", "deportes"),
    ("¿Quién ganó la Champions League 2024?", "deportes"),
    ("¿Cuál es el río más largo del mundo?", "geografia"),
    ("¿Cuál es el océano más profundo?", "geografia"),
    ("¿Cuál es el desierto más grande del mundo?", "geografia"),
    ("¿Cuál es la montaña más alta del mundo?", "geografia"),
    ("¿Cuál es el país más grande del mundo?", "geografia"),
    ("¿Cuál es la isla más grande del mundo?", "geografia"),
]


def retrieve_evidence(question: str) -> tuple[str, str]:
    """Devuelve (document_id, title) del primer hit, o vacios si la API falla."""
    try:
        response = requests.post(
            f"{API}/retrieve",
            json={"query": question, "k": 3, "collection": COLLECTION},
            timeout=120,
        )
        response.raise_for_status()
        chunks = response.json().get("chunks", [])
    except requests.RequestException:
        return "", ""
    if not chunks:
        return "", ""
    return str(chunks[0].get("document_id") or ""), str(chunks[0].get("title") or "")


def main() -> None:
    gold = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    direct = [row for row in gold if row.get("type") == "directa"]
    negative = [row for row in gold if row.get("type") == "sin_respuesta"]
    if len(direct) != 38 or len(negative) != 5:
        raise SystemExit(f"gold inesperado: {len(direct)} directas y {len(negative)} sin respuesta")

    rows: list[dict] = []
    for index, row in enumerate(direct):
        rows.append(
            {
                "question": row["question"],
                "label": "dominio",
                "subtype": "directa",
                "source": "gold",
                "group": f"gold_{row['id']}",
                "split": "train" if index < 28 else "dev",
                "reference_answer": row.get("reference_answer") or "",
                "reference_doc_ids": row.get("reference_doc_ids") or [],
                "evidence_doc_id": "",
                "evidence_title": "",
            }
        )
    for row in negative:
        rows.append(
            {
                "question": row["question"],
                "label": "fuera",
                "subtype": "sin_respuesta",
                "source": "gold",
                "group": f"sin_respuesta_{row['id']}",
                "split": "test",
                "reference_answer": "",
                "reference_doc_ids": [],
                "evidence_doc_id": "",
                "evidence_title": "",
            }
        )
    for question, group in VARIANTS:
        rows.append(
            {
                "question": question,
                "label": "dominio",
                "subtype": "hard_positive",
                "source": "variant",
                "group": group,
                "split": "train",
                "reference_answer": "",
                "reference_doc_ids": [],
                "evidence_doc_id": "",
                "evidence_title": "",
            }
        )
    for question, group in NEW_DOMAIN:
        doc_id, title = retrieve_evidence(question)
        rows.append(
            {
                "question": question,
                "label": "dominio",
                "subtype": "hard_positive",
                "source": "corpus",
                "group": group,
                "split": "test",
                "reference_answer": "",
                "reference_doc_ids": [],
                "evidence_doc_id": doc_id,
                "evidence_title": title,
            }
        )
    for question, group in CHAT_OUTSIDE:
        rows.append(
            {
                "question": question,
                "label": "fuera",
                "subtype": "fuera",
                "source": "chat",
                "group": group,
                "split": "test",
                "reference_answer": "",
                "reference_doc_ids": [],
                "evidence_doc_id": "",
                "evidence_title": "",
            }
        )
    for index, (question, group) in enumerate(OUTSIDE):
        rows.append(
            {
                "question": question,
                "label": "fuera",
                "subtype": "hard_negative",
                "source": "general",
                "group": group,
                "split": "train" if index < 36 else "dev",
                "reference_answer": "",
                "reference_doc_ids": [],
                "evidence_doc_id": "",
                "evidence_title": "",
            }
        )

    for index, row in enumerate(rows, start=1):
        row["id"] = f"d-{index:03d}"

    if len(rows) != 120:
        raise SystemExit(f"se esperaban 120 preguntas y hay {len(rows)}")

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "reference_doc_ids": ";".join(row["reference_doc_ids"])})

    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["split"], row["label"])
        counts[key] = counts.get(key, 0) + 1
    print(f"escrito {OUT_JSONL} y {OUT_CSV}")
    for key in sorted(counts):
        print(f"  {key[0]:5} {key[1]:7} {counts[key]}")
    print("nuevas de dominio con evidencia:")
    for row in rows:
        if row["source"] == "corpus":
            print(f"  {row['group']:12} {row['evidence_title'][:70]}")


if __name__ == "__main__":
    main()
