"""El Excel debe reflejar el estado validado del JSONL."""
from __future__ import annotations

import json
import sys

from openpyxl import load_workbook

from src.scripts.export_silver_excel import main


def test_validado_se_refleja(tmp_path, monkeypatch) -> None:
    silver = tmp_path / "silver.jsonl"
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
    silver.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    out = tmp_path / "silver.xlsx"
    monkeypatch.setattr(sys, "argv", ["export", "--silver", str(silver), "--out", str(out)])
    main()
    sheet = load_workbook(out)["Silver"]
    assert sheet["L2"].value == "Sí"
