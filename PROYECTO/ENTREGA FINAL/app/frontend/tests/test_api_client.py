"""test_api_client.py
Fecha: 2026-10-31
Descripción: Pruebas del cliente HTTP del frontend con httpx.MockTransport.
    Cubren el parseo de cada endpoint, la forma del payload, los códigos de
    error y los fallos de red sin levantar el backend. La prueba de
    integración es opcional y se activa con JURIS_API_TESTS=1.
Estado del arte aplicado: dobles de transporte en proceso, aserciones sobre
    el contrato exacto y separación entre pruebas unitarias e integración.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pytest

from api_client import ApiClient, ApiError


def make_client(
    handler: Any,
    *,
    base_url: str = "http://api.test",
    timeout: float = 1.0,
) -> ApiClient:
    """Crea un cliente con transporte simulado.

    Args:
        handler: Función que recibe la petición y devuelve una respuesta.
        base_url: URL base de prueba.
        timeout: Timeout de prueba.

    Returns:
        Cliente listo para usar.
    """
    return ApiClient(
        base_url=base_url,
        timeout=timeout,
        transport=httpx.MockTransport(handler),
    )


def test_health_returns_payload() -> None:
    """GET /health devuelve el estado sin transformaciones."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok", "ready": True})

    with make_client(handler) as client:
        assert client.health() == {"status": "ok", "ready": True}


def test_validate_sends_id_only() -> None:
    """La validación por referencia envía solo la clave id."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "es_sentencia": True,
                "score_validacion": 1.0,
                "canon_detectado": "T-293-1997",
            },
        )

    with make_client(handler) as client:
        result = client.validate(sentence_id="2195")

    assert captured == {"id": "2195"}
    assert result["canon_detectado"] == "T-293-1997"


def test_validate_sends_text_only() -> None:
    """La validación por texto envía solo la clave texto."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"es_sentencia": False})

    with make_client(handler) as client:
        client.validate(text="Providencia de prueba")

    assert captured == {"texto": "Providencia de prueba"}


def test_link_sends_parameters_and_parses_articles() -> None:
    """El vínculo conserva k y rerank y devuelve las listas del contrato."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/link"
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "2195",
                "canon_detectado": "T-293-1997",
                "origen": "id",
                "articulos_extraidos": [
                    {"numero": "86", "score": None, "bloque": "tutela"},
                ],
                "articulos_recuperados": [
                    {"numero": "86", "score": 1.6064, "bloque": "tutela"},
                ],
                "rerank": False,
                "k": 3,
                "ms": 120,
            },
        )

    with make_client(handler) as client:
        result = client.link(sentence_id="2195", k=3)

    assert captured == {"id": "2195", "k": 3, "rerank": False}
    assert result["articulos_recuperados"][0]["score"] == 1.6064


def test_models_returns_catalog() -> None:
    """El catálogo de modelos conserva nombre, defecto y tool_calling."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "disponible": True,
                "defecto": "qwen3.8:diario-8k-q3_k_s",
                "modelos": [
                    {
                        "nombre": "qwen3.8:diario-8k-q3_k_s",
                        "tool_calling": True,
                        "defecto": True,
                    }
                ],
            },
        )

    with make_client(handler) as client:
        result = client.models()

    assert result["modelos"][0]["tool_calling"] is True


def test_article_uses_number_in_path() -> None:
    """El detalle de artículo arma la ruta con el número recibido."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/article/86"
        return httpx.Response(200, json={"numero": "86", "texto_150chars": "Texto"})

    with make_client(handler) as client:
        result = client.article("86")

    assert result["numero"] == "86"


def test_evidencia_sends_articles_and_parses_fragments() -> None:
    """La evidencia envía artículos y devuelve fragmentos con su score."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/evidencia"
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "2195",
                "canon_detectado": "T-293-1997",
                "articulos": [
                    {
                        "numero": "86",
                        "fragmentos": [{"texto": "fragmento", "score": 149.1889}],
                    }
                ],
                "rerank": False,
                "ms": 12,
            },
        )

    with make_client(handler) as client:
        result = client.evidencia(sentence_id="2195", articles=["86", "13"])

    assert captured == {
        "id": "2195",
        "articulos": ["86", "13"],
        "rerank": False,
        "max_fragmentos": 3,
    }
    assert result["articulos"][0]["fragmentos"][0]["score"] == 149.1889


def test_jurisprudencia_uses_number_and_top() -> None:
    """La jurisprudencia arma la ruta con el número y el tope del ranking."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/article/86/jurisprudencia"
        assert request.url.params["top"] == "10"
        return httpx.Response(
            200,
            json={
                "numero": "86",
                "total": 16405,
                "por_ano": {"1992": 167, "2021": 15},
                "top": [
                    {
                        "id": "224",
                        "canon": "C-543-1992",
                        "ano": 1992,
                        "tipo": "C",
                        "pagerank": 0.004,
                        "area_silver": "derechos_fundamentales",
                    }
                ],
                "aviso": "Etiquetas debiles de M2.",
            },
        )

    with make_client(handler) as client:
        result = client.jurisprudencia("86")

    assert result["total"] == 16405
    assert result["top"][0]["canon"] == "C-543-1992"


def test_evidencia_404_is_actionable() -> None:
    """Un artículo fuera del índice se reporta con su código y detalle."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404, json={"detail": "El articulo '999' no esta en el indice."}
        )

    with make_client(handler) as client, pytest.raises(ApiError) as info:
        client.evidencia(sentence_id="2195", articles=["999"])

    assert info.value.status_code == 404
    assert "999" in str(info.value)


def test_ask_omits_model_when_none() -> None:
    """Sin modelo explícito, el payload no incluye la clave model."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"respuesta": "Hola", "modo": "small_talk"})

    with make_client(handler) as client:
        client.ask("Hola")

    assert captured == {
        "pregunta": "Hola",
        "rerank": False,
        "k": 6,
        "pensar": False,
    }


def test_ask_sends_selected_model() -> None:
    """Con modelo explícito, el payload lo incluye."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"respuesta": "Hola"})

    with make_client(handler) as client:
        client.ask("Hola", model="alia-legal:latest")

    assert captured["model"] == "alia-legal:latest"


def test_ask_envia_pensamiento() -> None:
    """Con razonamiento activo, el payload lo declara."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"respuesta": "Hola"})

    with make_client(handler) as client:
        client.ask("Hola", pensar=True)

    assert captured["pensar"] is True


def test_eval_summary_returns_provenance() -> None:
    """El resumen de evaluación conserva métricas y procedencia."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "fuentes": ["resultados/metricas.json"],
                "metricas": {"m0_accuracy": 1.0},
                "procedencia": {"m0_accuracy": "resultados/metricas.json:accuracy"},
            },
        )

    with make_client(handler) as client:
        result = client.eval_summary()

    assert result["metricas"]["m0_accuracy"] == 1.0
    assert result["procedencia"]["m0_accuracy"].endswith(":accuracy")


def test_similar_uses_id_and_k() -> None:
    """La similitud arma la ruta con el id y envía k como query string."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/similar/2195"
        assert request.url.params["k"] == "5"
        return httpx.Response(
            200,
            json={
                "id": "2195",
                "canon": "T-293-1997",
                "similares": [
                    {
                        "id": "5778",
                        "canon": "T-826-1999",
                        "ano": 1999,
                        "tipo": "T",
                        "pagerank": 0.0001,
                        "score": 0.29,
                    }
                ],
                "k": 5,
                "ms": 95,
                "aviso": "Similitud TF-IDF.",
            },
        )

    with make_client(handler) as client:
        result = client.similares("2195", k=5)

    assert result["similares"][0]["canon"] == "T-826-1999"
    assert result["similares"][0]["score"] == 0.29


def test_http_error_includes_status_and_detail() -> None:
    """Un 404 se convierte en ApiError con el detalle del backend."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "No existe el artículo 999."})

    with make_client(handler) as client, pytest.raises(ApiError) as info:
        client.article("999")

    assert info.value.status_code == 404
    assert "No existe el artículo 999." in str(info.value)


def test_validation_error_joins_messages() -> None:
    """Un 422 de Pydantic se resume con los mensajes de la lista."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "detail": [
                    {"msg": "Indique exactamente uno de 'texto' o 'id'."},
                    {"msg": "Value error, entrada inválida."},
                ]
            },
        )

    with make_client(handler) as client, pytest.raises(ApiError) as info:
        client.validate()

    assert info.value.status_code == 422
    assert "Indique exactamente uno" in str(info.value)
    assert "entrada inválida" in str(info.value)


def test_connection_error_is_actionable() -> None:
    """Un fallo de conexión nombra la URL y sugiere revisar API_URL."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("conexión rechazada", request=request)

    with (
        make_client(handler, base_url="http://127.0.0.1:9") as client,
        pytest.raises(ApiError) as info,
    ):
        client.health()

    assert "http://127.0.0.1:9" in str(info.value)
    assert "API_URL" in str(info.value)


def test_timeout_is_actionable() -> None:
    """Un timeout informa el límite y la ruta."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("sin respuesta", request=request)

    with (
        make_client(handler, timeout=2.0) as client,
        pytest.raises(ApiError) as info,
    ):
        client.health()

    assert "2 s" in str(info.value)
    assert "/health" in str(info.value)


def test_non_json_response_is_rejected() -> None:
    """Un cuerpo que no es JSON se reporta como error de protocolo."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>error</html>")

    with make_client(handler) as client, pytest.raises(ApiError) as info:
        client.health()

    assert "no es JSON" in str(info.value)


def test_base_url_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """API_URL define la URL base y se recorta la barra final."""
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9999/")
    client = ApiClient()
    try:
        assert client.base_url == "http://127.0.0.1:9999"
    finally:
        client.close()


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("JURIS_API_TESTS") != "1",
    reason="Requiere el backend real; active JURIS_API_TESTS=1.",
)
def test_real_backend_smoke() -> None:
    """Comprueba salud y un vínculo real con el id 2195."""
    with ApiClient(timeout=60.0) as client:
        health = client.health()
        assert health["status"] == "ok"

        link = client.link(sentence_id="2195", k=3)
        assert link["canon_detectado"] == "T-293-1997"
        assert link["articulos_recuperados"]
