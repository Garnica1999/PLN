"""api_client.py
Fecha: 2026-10-31
Descripción: Cliente HTTP del frontend Juris-PLN. Habla con la API FastAPI
    por httpx y convierte los fallos de red, de estado o de formato en
    ApiError con un mensaje accionable. No contiene lógica de negocio: no
    valida providencias, no recupera artículos y no interpreta el corpus.
Estado del arte aplicado: httpx.Client reutilizable con timeout explícito,
    transporte inyectable (httpx.MockTransport en pruebas) y errores tipados
    con el código HTTP original.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8123"
DEFAULT_TIMEOUT = 30.0
LINK_TIMEOUT = 120.0
EVIDENCIA_TIMEOUT = 120.0
ASK_TIMEOUT = 300.0
SIMILAR_TIMEOUT = 300.0


class ApiError(RuntimeError):
    """Fallo al comunicarse con la API.

    Attributes:
        status_code: Código HTTP de la respuesta, o None si la petición no
            llegó a completarse.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        """Inicializa el error con un mensaje listo para mostrar.

        Args:
            message: Texto que explica qué falló y qué hacer.
            status_code: Código HTTP asociado, si existe.
        """
        super().__init__(message)
        self.status_code = status_code


def _error_detail(response: httpx.Response) -> str:
    """Extrae el detalle legible de una respuesta de error.

    Args:
        response: Respuesta HTTP con código 4xx o 5xx.

    Returns:
        Mensaje del campo 'detail' o el cuerpo recortado si no es JSON.
    """
    try:
        data = response.json()
    except ValueError:
        return response.text.strip()[:300] or "sin detalle"
    detail = data.get("detail") if isinstance(data, dict) else None
    if isinstance(detail, list):
        messages = [
            str(item.get("msg", item)) if isinstance(item, dict) else str(item)
            for item in detail
        ]
        return "; ".join(messages) or "sin detalle"
    if detail:
        return str(detail)
    return response.text.strip()[:300] or "sin detalle"


class ApiClient:
    """Cliente de la API Juris-PLN.

    Args:
        base_url: URL base de la API. Si se omite, se usa la variable de
            entorno API_URL y, en su defecto, DEFAULT_API_URL.
        timeout: Timeout en segundos de las peticiones normales.
        transport: Transporte de httpx inyectable para pruebas.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("API_URL") or DEFAULT_API_URL).rstrip(
            "/"
        )
        self._timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> ApiClient:
        """Devuelve el cliente para usarlo como context manager."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Cierra las conexiones al salir del context manager."""
        self.close()

    def close(self) -> None:
        """Cierra el cliente y libera las conexiones abiertas."""
        self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Ejecuta una petición y normaliza los errores.

        Args:
            method: Verbo HTTP.
            path: Ruta relativa a la URL base.
            payload: Cuerpo JSON de la petición, si aplica.
            timeout: Timeout específico; si se omite se usa el del cliente.

        Returns:
            El objeto JSON de la respuesta.

        Raises:
            ApiError: Si la API no responde, devuelve un error HTTP o un
                cuerpo que no es un objeto JSON.
        """
        try:
            response = self._client.request(
                method,
                path,
                json=payload,
                timeout=timeout if timeout is not None else self._timeout,
            )
        except httpx.TimeoutException as exc:
            limit = timeout if timeout is not None else self._timeout
            raise ApiError(
                f"La API no respondió en {limit:.0f} s ({path}). "
                "Reintente o aumente el timeout."
            ) from exc
        except httpx.ConnectError as exc:
            raise ApiError(
                f"No se pudo conectar con la API en {self.base_url}. "
                "Inicie el backend o revise la variable API_URL."
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiError(f"Fallo de red al llamar a {path}: {exc}.") from exc

        if response.status_code >= 400:
            raise ApiError(
                f"La API respondió {response.status_code} en {path}: "
                f"{_error_detail(response)}",
                status_code=response.status_code,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ApiError(
                f"La API devolvió una respuesta que no es JSON en {path}."
            ) from exc
        if not isinstance(data, dict):
            raise ApiError(f"La API devolvió un JSON inesperado en {path}.")
        return data

    def health(self) -> dict[str, Any]:
        """Consulta el estado del servicio y de sus componentes.

        Returns:
            Respuesta de /health.
        """
        return self._request("GET", "/health")

    def models(self) -> dict[str, Any]:
        """Lista los modelos de ollama y su soporte estimado de tool calling.

        Returns:
            Respuesta de /v1/models.
        """
        return self._request("GET", "/v1/models")

    def validate(
        self,
        *,
        text: str | None = None,
        sentence_id: str | None = None,
    ) -> dict[str, Any]:
        """Valida si una entrada es una providencia (heurística M0).

        Args:
            text: Texto completo de la providencia.
            sentence_id: Id del corpus o canon de la sentencia.

        Returns:
            Respuesta de /v1/validate.
        """
        payload = _drop_none({"texto": text, "id": sentence_id})
        return self._request("POST", "/v1/validate", payload=payload)

    def link(
        self,
        *,
        sentence_id: str | None = None,
        text: str | None = None,
        k: int = 10,
        rerank: bool = False,
    ) -> dict[str, Any]:
        """Vincula una sentencia con artículos de la Constitución (M2+M7).

        Args:
            sentence_id: Id del corpus o canon de la sentencia.
            text: Texto de la sentencia, alternativo al id.
            k: Número de artículos a recuperar.
            rerank: Activa la ruta híbrida con rerank (GPU, más lenta).

        Returns:
            Respuesta de /v1/link.
        """
        payload = _drop_none(
            {
                "id": sentence_id,
                "texto": text,
                "k": k,
                "rerank": rerank,
            }
        )
        return self._request(
            "POST",
            "/v1/link",
            payload=payload,
            timeout=LINK_TIMEOUT,
        )

    def article(self, number: str) -> dict[str, Any]:
        """Consulta el texto y el bloque de un artículo de la CP.

        Args:
            number: Número del artículo, por ejemplo "86".

        Returns:
            Respuesta de /v1/article/{numero}.
        """
        return self._request("GET", f"/v1/article/{number}")

    def jurisprudencia(self, number: str, *, top: int = 10) -> dict[str, Any]:
        """Lista las sentencias del corpus que citan un artículo de la CP.

        Args:
            number: Número del artículo, por ejemplo "86".
            top: Tamaño del ranking por PageRank (1-50).

        Returns:
            Respuesta de /v1/article/{numero}/jurisprudencia.
        """
        return self._request(
            "GET", f"/v1/article/{number}/jurisprudencia?top={top}"
        )

    def evidencia(
        self,
        *,
        sentence_id: str | None = None,
        text: str | None = None,
        articles: list[str],
        rerank: bool = False,
        max_fragments: int = 3,
    ) -> dict[str, Any]:
        """Recupera los fragmentos de una sentencia para cada artículo.

        Args:
            sentence_id: Id del corpus o canon de la sentencia.
            text: Texto de la sentencia, alternativo al id.
            articles: Artículos de la CP a evidenciar (1 a 5).
            rerank: Activa la ruta B2+B1 para repuntuar los fragmentos.
            max_fragments: Fragmentos por artículo (1 a 3).

        Returns:
            Respuesta de /v1/evidencia.
        """
        payload = _drop_none(
            {
                "id": sentence_id,
                "texto": text,
                "articulos": articles,
                "rerank": rerank,
                "max_fragmentos": max_fragments,
            }
        )
        return self._request(
            "POST",
            "/v1/evidencia",
            payload=payload,
            timeout=EVIDENCIA_TIMEOUT,
        )

    def ask(
        self,
        question: str,
        *,
        model: str | None = None,
        rerank: bool = False,
        k: int = 6,
        pensar: bool = False,
    ) -> dict[str, Any]:
        """Formula una consulta al agente conversacional.

        Args:
            question: Pregunta en lenguaje natural.
            model: Modelo de ollama; si se omite se usa el del backend.
            rerank: Activa la ruta de recuperación con rerank.
            k: Artículos que recupera la herramienta de vínculo.
            pensar: Habilita el razonamiento del modelo; la respuesta tarda
                más y devuelve el pensamiento aparte.

        Returns:
            Respuesta de /v1/ask.
        """
        payload = _drop_none(
            {
                "pregunta": question,
                "model": model,
                "rerank": rerank,
                "k": k,
                "pensar": pensar,
            }
        )
        return self._request(
            "POST",
            "/v1/ask",
            payload=payload,
            timeout=ASK_TIMEOUT,
        )

    def eval_summary(self) -> dict[str, Any]:
        """Lee las métricas clave con su procedencia.

        Returns:
            Respuesta de /v1/eval/summary.
        """
        return self._request("GET", "/v1/eval/summary")

    def similares(self, sentence_id: str, *, k: int = 5) -> dict[str, Any]:
        """Lista las sentencias del corpus más parecidas a una dada.

        Args:
            sentence_id: Id del corpus de la sentencia consultada.
            k: Número de vecinos a devolver (1-20).

        Returns:
            Respuesta de /v1/similar/{id}.
        """
        return self._request(
            "GET",
            f"/v1/similar/{sentence_id}?k={k}",
            timeout=SIMILAR_TIMEOUT,
        )


def _drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    """Elimina las claves con valor None de un payload.

    Args:
        payload: Diccionario candidato.

    Returns:
        Copia sin las claves cuyo valor es None.
    """
    return {key: value for key, value in payload.items() if value is not None}
