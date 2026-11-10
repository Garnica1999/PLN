"""Trazas de Langfuse. Si no hay llaves configuradas, todo es no-op."""
from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache

DEFAULT_HOST = "http://localhost:3000"


@lru_cache(maxsize=1)
def get_client():
    """Crea el cliente una vez; None si faltan llaves o el SDK falla."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    if not public_key or not secret_key:
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            host=os.getenv("LANGFUSE_HOST", DEFAULT_HOST),
            public_key=public_key,
            secret_key=secret_key,
        )
    except Exception:  # noqa: BLE001  # la observabilidad no puede tumbar la API
        return None


def trace_id(seed: str | None = None) -> str | None:
    """Identificador estable y consultable desde la respuesta HTTP."""
    client = get_client()
    if client is None:
        return None
    try:
        return client.create_trace_id(seed=seed)
    except Exception:  # noqa: BLE001
        return None


@contextmanager
def observation(name: str, as_type: str = "span", trace: str | None = None, **kwargs):
    """Abre un span o generación; si Langfuse no está, entrega None."""
    client = get_client()
    if client is None:
        yield None
        return
    context = {"trace_id": trace} if trace else None
    try:
        manager = client.start_as_current_observation(
            name=name, as_type=as_type, trace_context=context, **kwargs
        )
    except Exception:  # noqa: BLE001
        yield None
        return
    with manager as observation:
        yield observation


def score(name: str, value: float | str, trace: str | None = None, comment: str | None = None) -> None:
    """Adjunta un score al trace actual o a uno explícito."""
    client = get_client()
    if client is None:
        return
    try:
        client.create_score(name=name, value=value, trace_id=trace, comment=comment)
    except Exception:  # noqa: BLE001
        return


def flush() -> None:
    """Fuerza el envío de eventos pendientes."""
    client = get_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:  # noqa: BLE001
        return
