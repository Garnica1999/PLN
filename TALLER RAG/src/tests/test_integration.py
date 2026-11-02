"""Pruebas de integración contra la API real. Requieren el stack levantado."""
from __future__ import annotations

import os

import pytest
import requests

API = os.getenv("API_URL", "http://localhost:8010").rstrip("/")


def api_disponible() -> bool:
    try:
        return requests.get(f"{API}/health", timeout=5).ok
    except requests.RequestException:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not api_disponible(), reason="API no disponible"),
]


def post(path: str, payload: dict) -> dict:
    response = requests.post(f"{API}{path}", json=payload, timeout=300)
    assert response.status_code == 200, response.text
    return response.json()


def test_health() -> None:
    body = requests.get(f"{API}/health", timeout=10).json()
    assert body["status"] == "ok"


def test_retrieve() -> None:
    body = post("/retrieve", {"query": "¿Cuántos municipios han reportado la presencia del Retamo Espinoso?", "k": 5})
    assert body["chunks"]


def test_chat_saludo() -> None:
    body = post("/chat", {"query": "Hola", "model": "gemma2:2b", "k": 5})
    assert body["intent"] == "charla"
    assert body["abstained"] is False


def test_chat_dominio() -> None:
    body = post("/chat", {"query": "¿Cuántos municipios han reportado la presencia del Retamo Espinoso?", "model": "gemma2:2b", "k": 5})
    assert body["intent"] == "dominio"
    assert body["abstained"] is False
    assert body["citations"]


def test_chat_sin_evidencia() -> None:
    body = post("/chat", {"query": "¿Cuántos colonos marcianos viven en la base de Marte?", "model": "gemma2:2b", "k": 5})
    assert body["abstained"] is True


def test_tool_calling() -> None:
    body = post("/chat", {"query": "¿Cuántos municipios han reportado la presencia del Retamo Espinoso?", "model": "llama3.2:3b", "k": 5})
    assert "buscar_corpus" in body["tool_calls"]


def test_chat_con_razonamiento() -> None:
    body = post(
        "/chat",
        {
            "query": "¿Cuántos municipios han reportado la presencia del Retamo Espinoso?",
            "model": "gemma2:2b",
            "k": 5,
            "think": True,
        },
    )
    assert body["reasoning"].strip()
    assert body["reasoning_ms"] > 0
    assert body["citations"]
    assert body["abstained"] is False


def test_chat_sin_razonamiento_por_defecto() -> None:
    body = post(
        "/chat",
        {"query": "¿Cuántos municipios han reportado la presencia del Retamo Espinoso?", "model": "gemma2:2b", "k": 5},
    )
    assert body["reasoning"] == ""
    assert body["reasoning_ms"] == 0.0


def test_chat_gemma4_razonamiento_nativo() -> None:
    body = post(
        "/chat",
        {
            "query": "¿Cuántos municipios han reportado la presencia del Retamo Espinoso?",
            "model": "gemma4:e4b",
            "k": 5,
            "think": True,
        },
    )
    assert body["native_thinking"] is True
    assert body["reasoning"].strip()
    assert body["citations"]
    assert body["abstained"] is False
