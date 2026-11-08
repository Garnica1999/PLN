"""Selección ciega de respuestas para la anotación humana de E4."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

SEED = 20261031
EJES = ["correccion", "fidelidad", "citas", "abstencion"]
KEY_FIELDS = ["item_id", "model", "question_id", "rep"]


def blind_id(model: str, question_id: str, rep: int) -> str:
    return hashlib.sha1(f"{model}|{question_id}|{rep}".encode()).hexdigest()[:10]


def load_items(run_path: Path, silver_path: Path) -> tuple[dict, list[dict]]:
    silver = {
        row["id"]: row
        for row in (json.loads(line) for line in silver_path.read_text(encoding="utf-8").splitlines() if line.strip())
    }
    answerable = {key for key, row in silver.items() if row.get("reference_doc_ids")}
    records = [json.loads(line) for line in run_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [
        r
        for r in records
        if r.get("mode") == "ask"
        and not r.get("error")
        and r.get("response")
        and int(r.get("rep", 1)) == 1
        and r.get("question_id") in answerable
    ]
    records.sort(key=lambda r: (r["question_id"], r["model"]))
    random.Random(SEED).shuffle(records)
    return silver, records


def write_key(records: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "key.csv"
    expected = {blind_id(r["model"], r["question_id"], int(r["rep"])) for r in records}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            existing = {row["item_id"] for row in csv.DictReader(handle)}
        if existing != expected:
            raise ValueError("key.csv no coincide con la corrida actual")
        return path
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=KEY_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "item_id": blind_id(r["model"], r["question_id"], int(r["rep"])),
                    "model": r["model"],
                    "question_id": r["question_id"],
                    "rep": int(r["rep"]),
                }
            )
    return path
