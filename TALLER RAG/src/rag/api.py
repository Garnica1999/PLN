"""API FastAPI: salud, recuperación y respuesta con evidencia."""
from __future__ import annotations

from typing import Literal

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .chat import chat as chat_answer
from .config import settings
from .pipeline import LLMError, generate, search_hits
from .store import collections, count, get_client
from .tracing import observation
from .tracing import trace_id as make_trace_id

app = FastAPI(title="Taller RAG Milvus", version="1.0")


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=3)
    k: int = Field(default=5, ge=1, le=20)
    collection: str | None = None
    language: str | None = Field(default=None, pattern=r"^[a-z]{2}$")
    date_from: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    date_to: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    hybrid: bool | None = None
    retriever: str | None = None


class AskRequest(RetrieveRequest):
    model: str | None = None
    temperature: float = Field(default=0.0, ge=0, le=1)
    think: bool = False
    strict: bool = False
    ambiguity_policy: str = "auto"


def build_expr(request: RetrieveRequest) -> str | None:
    parts = []
    if request.language:
        parts.append(f'language == "{request.language}"')
    if request.date_from:
        parts.append(f'created_at >= "{request.date_from}"')
    if request.date_to:
        parts.append(f'created_at <= "{request.date_to}"')
    return " and ".join(parts) if parts else None


def _traced(request: AskRequest, producer, name: str = "ask") -> dict:
    """Ejecuta una respuesta dentro de un trace de Langfuse."""
    trace = make_trace_id()
    with observation(
        name,
        trace=trace,
        input={"query": request.query, "k": request.k, "collection": request.collection},
        metadata={
            "retriever": request.retriever or settings.retriever,
            "llm": settings.llm_model,
            "embeddings": settings.embeddings_model,
        },
    ) as root:
        result = producer()
        if root is not None:
            root.update(
                output={
                    "answer": result.get("answer"),
                    "citations": result.get("citations"),
                    "abstained": result.get("abstained"),
                }
            )
            root.score_trace(
                name="citation_valid",
                value=0.0 if result.get("invalid_citations") else 1.0,
            )
            grounded = (result.get("grounding") or {}).get("ungrounded") or []
            root.score_trace(name="grounded", value=0.0 if grounded else 1.0)
            root.score_trace(name="latency_ms", value=float(result.get("total_ms") or 0.0))
        result["trace_id"] = trace
        return result


@app.get("/health")
def health() -> dict:
    client = get_client()
    milvus_ok = True
    try:
        client.list_collections()
    except Exception:  # noqa: BLE001
        milvus_ok = False

    llm_ok = False
    try:
        llm_ok = requests.get(f"{settings.llm_url.rstrip('/')}/api/tags", timeout=5).ok
    except requests.RequestException:
        llm_ok = False

    if settings.embedder == "ollama":
        try:
            embeddings_ok = requests.get(f"{settings.embeddings_url.rstrip('/')}/api/tags", timeout=5).ok
        except requests.RequestException:
            embeddings_ok = False
    else:
        embeddings_ok = True
    status = "ok" if milvus_ok and llm_ok and embeddings_ok else "degraded"
    return {
        "status": status,
        "milvus": milvus_ok,
        "llm": llm_ok,
        "embeddings": embeddings_ok,
        "collection": settings.collection,
        "llm_model": settings.llm_model,
        "embeddings_model": settings.embeddings_model,
    }


@app.get("/models")
def models() -> dict:
    import requests

    try:
        data = requests.get(f"{settings.llm_url.rstrip('/')}/api/tags", timeout=10).json()
    except Exception as error:
        raise HTTPException(503, detail=f"Ollama no responde: {error}") from error
    return {"models": [item["name"] for item in data.get("models", [])]}


@app.get("/stats")
def stats() -> dict:
    client = get_client()
    names = collections(client)
    return {"collections": {name: count(client, name) for name in names}}


@app.post("/retrieve")
def search_endpoint(request: RetrieveRequest) -> dict:
    try:
        hits, elapsed = search_hits(
            request.query,
            k=request.k,
            collection=request.collection,
            expr=build_expr(request),
            hybrid=request.hybrid,
        )
    except ValueError as error:
        raise HTTPException(400, detail=str(error)) from error
    return {"chunks": hits, "retrieval_ms": elapsed}


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(AskRequest):
    history: list[ChatTurn] = Field(default_factory=list, max_length=12)


@app.post("/chat")
def chat_endpoint(request: ChatRequest) -> dict:
    try:
        return _traced(
            request,
            lambda: chat_answer(
                request.query,
                model=request.model,
                k=request.k,
                temperature=request.temperature,
                collection=request.collection,
                expr=build_expr(request),
                hybrid=request.hybrid,
                history=[turn.model_dump() for turn in request.history],
                think=request.think,
                strict=request.strict,
                ambiguity_policy=request.ambiguity_policy,
            ),
            name="chat",
        )
    except ValueError as error:
        raise HTTPException(400, detail=str(error)) from error
    except LLMError as error:
        raise HTTPException(503, detail=str(error)) from error


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    try:
        return _traced(
            request,
            lambda: generate(
                request.query,
                model=request.model,
                k=request.k,
                temperature=request.temperature,
                collection=request.collection,
                expr=build_expr(request),
                hybrid=request.hybrid,
                retriever=request.retriever,
                think=request.think,
                domain_gate=request.strict,
                ambiguity_policy=request.ambiguity_policy,
                interactive=False,
            ),
        )
    except ValueError as error:
        raise HTTPException(400, detail=str(error)) from error
    except LLMError as error:
        raise HTTPException(503, detail=str(error)) from error
