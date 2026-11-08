"""Pruebas de la API con la generación simulada."""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.rag import api

client = TestClient(api.app)


def test_retrieve(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "search_hits",
        lambda *args, **kwargs: ([{"chunk_id": "c1", "document_id": "d1", "score": 0.9}], 12.5),
    )
    response = client.post("/retrieve", json={"query": "una consulta", "k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_ms"] == 12.5
    assert body["chunks"][0]["chunk_id"] == "c1"


def test_ask(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "generate",
        lambda *args, **kwargs: {
            "answer": "respuesta [1]",
            "evidence": [],
            "citations": [1],
            "invalid_citations": [],
            "abstained": False,
        },
    )
    response = client.post("/ask", json={"query": "una consulta", "model": "qwen2.5:1.5b"})
    assert response.status_code == 200
    assert response.json()["answer"] == "respuesta [1]"


def test_rechaza_idioma_con_comillas() -> None:
    response = client.post("/retrieve", json={"query": "prueba", "language": 'es" or true'})
    assert response.status_code == 422


def test_rechaza_coleccion_de_otro_encoder() -> None:
    response = client.post(
        "/retrieve",
        json={"query": "prueba", "collection": "rag_pilot_qwen3emb8b_d1024_fixed_v1"},
    )
    assert response.status_code == 400


def test_chat(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "chat_answer",
        lambda *args, **kwargs: {
            "answer": "Hola.",
            "intent": "charla",
            "guardrail": "charla",
            "citations": [],
            "invalid_citations": [],
            "evidence": [],
            "abstained": False,
            "retrieval_ms": 0.0,
            "generation_ms": 1.0,
            "total_ms": 1.0,
            "tokens_per_second": None,
            "model": "gemma2:2b",
            "k": 5,
            "collection": "rag_full_bge_m3_d1024_v1",
            "prompt_version": "v2",
            "prompt_hash": "x",
            "context_tokens": 0,
        },
    )
    response = client.post("/chat", json={"query": "Hola"})
    assert response.status_code == 200
    assert response.json()["intent"] == "charla"
