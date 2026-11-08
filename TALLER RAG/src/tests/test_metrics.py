"""Pruebas de las métricas de recuperación con casos calculados a mano."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.scripts.metrics import average_precision, ndcg_at_k, row_metrics, to_markdown


def test_ndcg_relevante_primero() -> None:
    assert ndcg_at_k(["a", "b"], {"a"}, 2) == pytest.approx(1.0)


def test_ndcg_relevante_segundo() -> None:
    assert ndcg_at_k(["b", "a"], {"a"}, 2) == pytest.approx(1 / math.log2(3))


def test_average_precision() -> None:
    assert average_precision(["a", "b"], {"a"}, 2) == pytest.approx(1.0)
    assert average_precision(["b", "a"], {"a"}, 2) == pytest.approx(0.5)


def test_sin_referencias_marca_has_refs() -> None:
    record = {
        "question_id": "q1",
        "mode": "retrieve",
        "k": 5,
        "response": {"chunks": []},
        "reference_doc_ids": [],
    }
    row = row_metrics(record)
    assert row["has_refs"] is False
    assert row["recall"] == 0.0


def test_markdown_incluye_p_at_1_y_p_at_5() -> None:
    row = {
        "run": "r", "mode": "retrieve", "model": "e5", "k": 10, "split": "test",
        "n_registros": 3, "n_preguntas": 1, "hit": 1, "recall": 1, "recall_lo": 1,
        "recall_hi": 1, "p_at_1": 1, "p_at_5": 1, "mrr": 1, "map": 1, "ndcg": 1,
        "ndcg_lo": 1, "ndcg_hi": 1, "chunk_recall": 1, "span_hit": 1,
        "context_precision": 1, "context_recall": 1, "duplicados": 0,
        "retrieval_p50": 1, "generation_p50": 0, "tokens_por_segundo": 0,
        "abstencion": 0, "citas_invalidas": 0, "sin_citas": 0,
    }
    markdown = to_markdown(pd.DataFrame([row]))
    assert "p_at_1" in markdown
    assert "p_at_5" in markdown
