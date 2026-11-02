"""Prueba de humo del render de la interfaz, sin red."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

import requests
from streamlit.testing.v1 import AppTest

UI = Path(__file__).resolve().parents[1] / "rag" / "ui.py"

PAYLOAD = {
    "status": "ok",
    "models": ["gemma4:e4b"],
    "collections": {"rag_full_bge_m3_d1024_v1": 618286},
}


class FakeResponse:
    """Respuesta minima para los tres GET de arranque."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.status_code = 200

    def json(self) -> dict:
        return self.payload

    def raise_for_status(self) -> None:
        return None


def test_la_ui_renderiza_sin_excepciones(monkeypatch) -> None:
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: FakeResponse(PAYLOAD))
    app = AppTest.from_file(str(UI), default_timeout=60)
    app.run()

    assert not app.exception
    assert any("Asistente RAG" in block.value for block in app.markdown)
    assert [toggle.value for toggle in app.toggle] == [False, True, True]
