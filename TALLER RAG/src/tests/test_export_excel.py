"""El Excel debe reflejar el estado validado del JSONL."""
from __future__ import annotations

import json
import sys

from openpyxl import load_workbook

from src.scripts.export_gold_excel import main


def test_validado_se_refleja(tmp_path, monkeypatch) -> None:
    gold = tmp_path / "gold.jsonl"
    record = {
        "id": "g-001",
        "split": "dev",
        "type": "directa",
        "lang": "es",
        "source": "url_crawls",
        "question": "pregunta",
        "reference_answer": "respuesta",
        "reference_doc_ids": ["url_crawls:1"],
        "reference_url": "",
        "reference_span": "respuesta",
        "criteria": "criterio",
        "reviewer": "",
        "validated": True,
    }
    gold.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    out = tmp_path / "gold.xlsx"
    monkeypatch.setattr(sys, "argv", ["export", "--gold", str(gold), "--out", str(out)])
    main()
    sheet = load_workbook(out)["Gold"]
    assert sheet["L2"].value == "Sí"
