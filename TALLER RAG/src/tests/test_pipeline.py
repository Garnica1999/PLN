"""El guard de encoder debe aceptar la familia activa y rechazar otra."""
from __future__ import annotations

import pytest

from src.rag.config import settings
from src.rag.pipeline import (
    call_llm,
    model_limits,
    supports_native_thinking,
    validate_collection,
)


def test_acepta_la_familia_activa() -> None:
    validate_collection(f"rag_pilot_{settings.encoder_tag}_d768_v2_v1")


def test_rechaza_otra_familia() -> None:
    with pytest.raises(ValueError):
        validate_collection("rag_pilot_sin_encoder_v1")


def test_none_no_falla() -> None:
    validate_collection(None)


def test_weighted_rrf_prioriza_el_peso() -> None:
    from src.rag.pipeline import weighted_rrf

    dense = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    sparse = [{"chunk_id": "b"}, {"chunk_id": "c"}]
    result = weighted_rrf(dense, sparse, 0.3, 0.7, top=2)
    assert result[0]["chunk_id"] == "b"


def test_retrieve_sparse_sin_coleccion(monkeypatch) -> None:
    from src.rag import pipeline

    class FakeClient:
        def has_collection(self, name: str) -> bool:
            return False

    monkeypatch.setattr(pipeline, "get_client", lambda: FakeClient())
    with pytest.raises(ValueError):
        pipeline.retrieve_sparse("una consulta")


def test_extract_citations_multiples() -> None:
    from src.rag.pipeline import extract_citations

    assert extract_citations("La moneda circuló entre 1996 y 2002 [2, 3, 4].") == [2, 3, 4]
    assert extract_citations("22 municipios [1].") == [1]
    assert extract_citations("sin cita") == []


def test_call_llm_usa_el_maximo_de_salida(monkeypatch) -> None:
    captured = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"response": "ok"}

    def fake_post(url, json, timeout):
        captured.update(json)
        return Response()

    monkeypatch.setattr("src.rag.pipeline.requests.post", fake_post)
    call_llm("hola", "gemma2:2b", 0.0)
    assert captured["options"]["num_predict"] == settings.max_output_tokens
    assert captured["options"]["num_ctx"] == 8192


def test_limites_de_gemma4() -> None:
    assert model_limits("gemma4:e4b") == (116384, 100000)
    assert supports_native_thinking("gemma4:e4b")


def test_limites_de_modelos_pequenos() -> None:
    assert model_limits("gemma2:2b") == (8192, 8192)
    assert not supports_native_thinking("gemma2:2b")
