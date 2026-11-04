"""El split del gold no debe repetir una pregunta."""
from __future__ import annotations

import pandas as pd

from src.scripts.build_gold_standard import assign_split


def test_split_balanceado_y_sin_solapamiento() -> None:
    frame = pd.DataFrame(
        [
            {"id": "a", "lang": "es", "reference_doc_ids": ["url_crawls:1"]},
            {"id": "b", "lang": "es", "reference_doc_ids": ["url_crawls:2"]},
            {"id": "c", "lang": "en", "reference_doc_ids": ["bluesky_posts:1"]},
            {"id": "d", "lang": "en", "reference_doc_ids": ["bluesky_posts:2"]},
        ]
    )
    output = assign_split(frame)
    assert set(output["split"]) == {"dev", "test"}
    assert output["id"].is_unique
    assert len(output) == 4


def test_builder_preserva_el_split(tmp_path, monkeypatch) -> None:
    import json
    import sys

    from src.scripts.build_gold_standard import main

    source = tmp_path / "gold.jsonl"
    records = [
        {"id": "x1", "question": "q1", "lang": "es", "type": "directa", "reference_doc_ids": ["url_crawls:1"], "reference_answer": "a", "reference_span": "a"},
        {"id": "x2", "question": "q2", "lang": "es", "type": "directa", "reference_doc_ids": ["url_crawls:2"], "reference_answer": "b", "reference_span": "b"},
        {"id": "x3", "question": "q3", "lang": "en", "type": "directa", "reference_doc_ids": ["bluesky_posts:1"], "reference_answer": "c", "reference_span": "c"},
        {"id": "x4", "question": "q4", "lang": "en", "type": "directa", "reference_doc_ids": ["bluesky_posts:2"], "reference_answer": "d", "reference_span": "d"},
    ]
    source.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    argv = ["builder", "--gold", str(source), "--outdir", str(out), "--parquet", str(tmp_path / "none.parquet")]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    first = {
        json.loads(line)["question"]: json.loads(line)["split"]
        for line in (out / "silver_standard.jsonl").read_text(encoding="utf-8").splitlines()
    }
    changed = []
    for line in (out / "silver_standard.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        record["split"] = "test" if record["split"] == "dev" else "dev"
        changed.append(record)
    flipped = {record["question"]: record["split"] for record in changed}
    (out / "silver_standard.jsonl").write_text(
        "\n".join(json.dumps(r) for r in changed) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(sys, "argv", argv)
    main()
    second = {
        json.loads(line)["question"]: json.loads(line)["split"]
        for line in (out / "silver_standard.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert second == flipped
    assert first != second
