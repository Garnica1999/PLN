"""Orquestación RAG: recuperar, armar contexto, generar y verificar citas."""
from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from functools import lru_cache

import requests

from .ambiguity import clarification_payload, conditional_payload, needs_detection
from .ambiguity import detect as detect_ambiguity
from .chunking import estimate_tokens
from .config import settings
from .domain import domain_probability as domain_probability_score
from .embeddings import get_embedder
from .graph import boost_hits, search_graph
from .grounding import check as check_grounding
from .prompts import REASONING_PROMPT, get_prompt, prompt_hash
from .solr import search_solr, search_solr_dense
from .store import OUTPUT_FIELDS, get_client, search
from .tracing import observation

CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
ABSTENTION_TEXT = "No hay evidencia suficiente en el corpus para responder."


class LLMError(RuntimeError):
    """El servicio de generación no respondió."""


def validate_collection(collection: str | None) -> None:
    """Evita consultar una colección indexada con otro encoder."""
    if collection and settings.encoder_tag not in collection:
        raise ValueError(
            f"la colección '{collection}' no corresponde al encoder activo "
            f"({settings.encoder_tag}). Usa una colección con '{settings.encoder_tag}' en el nombre."
        )


def dedupe(hits: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for hit in hits:
        key = hit.get("text_sha256") or hit.get("chunk_id")
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    return unique


@lru_cache(maxsize=1)
def get_reranker():
    """Carga perezosa del cross-encoder de reordenamiento."""
    from sentence_transformers import CrossEncoder

    kwargs = {}
    if settings.rerank_device == "cuda":
        import torch

        kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    return CrossEncoder(
        settings.rerank_model,
        device=settings.rerank_device,
        max_length=512,
        **kwargs,
    )


def score_hits(query: str, hits: list[dict]) -> list[dict]:
    """Anota rerank_score sin reordenar, para compartirlo con la compuerta."""
    if not hits:
        return hits
    missing = [index for index, hit in enumerate(hits) if "rerank_score" not in hit]
    if missing:
        scores = get_reranker().predict(
            [(query, hits[index].get("text", "")) for index in missing],
            batch_size=settings.rerank_batch,
            show_progress_bar=False,
        )
        for index, score in zip(missing, scores):
            hits[index]["rerank_score"] = float(score)
    return hits


def rerank_hits(query: str, hits: list[dict]) -> list[dict]:
    """Reordena los fragmentos recuperados con un cross-encoder multilingüe."""
    if len(hits) < 2:
        return hits
    hits = score_hits(query, hits)
    order = sorted(range(len(hits)), key=lambda index: -float(hits[index]["rerank_score"]))
    return [dict(hits[index]) for index in order]


def retrieve(
    query: str,
    k: int | None = None,
    collection: str | None = None,
    expr: str | None = None,
    ef: int | None = None,
) -> tuple[list[dict], float]:
    k = k or settings.top_k
    validate_collection(collection)
    name = collection or settings.collection
    started = time.perf_counter()
    vector = get_embedder().embed_queries([query])[0]
    hits = search(get_client(), name, vector, k, expr, ef=ef)
    hits = dedupe(hits)
    return hits, round((time.perf_counter() - started) * 1000, 2)


def retrieve_sparse(
    query: str,
    k: int | None = None,
    collection: str | None = None,
    expr: str | None = None,
) -> tuple[list[dict], float]:
    """Busqueda BM25 sobre la coleccion sparse, sin embeddings."""
    k = k or settings.top_k
    name = collection or settings.sparse_collection
    if not get_client().has_collection(name):
        raise ValueError(f"la colección sparse '{name}' no existe")
    started = time.perf_counter()
    results = get_client().search(
        collection_name=name,
        data=[query],
        anns_field="sparse",
        limit=k,
        output_fields=OUTPUT_FIELDS,
        filter=expr or "",
    )
    hits = []
    for hit in results[0]:
        entity = hit.get("entity", {}) or {}
        hits.append({"chunk_id": hit.get("chunk_id") or hit.get("id"), "score": float(hit.get("distance", 0.0)), **entity})
    return dedupe(hits), round((time.perf_counter() - started) * 1000, 2)


def retrieve_union(
    query: str,
    k: int | None = None,
    collection: str | None = None,
    expr: str | None = None,
    dense_k: int | None = None,
    sparse_k: int | None = None,
) -> tuple[list[dict], float]:
    """Union densa + BM25 sin fusionar: conserva el top denso y anade entidades."""
    dense_k = dense_k or settings.union_dense_k
    sparse_k = sparse_k or settings.union_sparse_k
    dense_hits, dense_ms = retrieve(query, k=dense_k, collection=collection, expr=expr)
    sparse_hits, sparse_ms = retrieve_sparse(query, k=sparse_k, expr=expr)
    return dedupe(dense_hits + sparse_hits), round(dense_ms + sparse_ms, 2)


def weighted_rrf(
    dense_hits: list[dict],
    sparse_hits: list[dict],
    w_dense: float,
    w_sparse: float,
    rrf_k: int = 60,
    top: int = 10,
) -> list[dict]:
    scores: dict[str, float] = {}
    for rank, hit in enumerate(dense_hits, start=1):
        chunk_id = hit.get("chunk_id")
        scores[chunk_id] = scores.get(chunk_id, 0.0) + w_dense / (rrf_k + rank)
    for rank, hit in enumerate(sparse_hits, start=1):
        chunk_id = hit.get("chunk_id")
        scores[chunk_id] = scores.get(chunk_id, 0.0) + w_sparse / (rrf_k + rank)
    by_id = {hit["chunk_id"]: hit for hit in dense_hits + sparse_hits}
    return [by_id[chunk_id] for chunk_id in sorted(scores, key=scores.get, reverse=True)[:top] if chunk_id in by_id]


def retrieve_hybrid(
    query: str,
    k: int | None = None,
    collection: str | None = None,
    expr: str | None = None,
    weights: tuple[float, float] | None = None,
) -> tuple[list[dict], float]:
    k = k or settings.top_k
    w_dense, w_sparse = weights or (settings.hybrid_dense, settings.hybrid_sparse)
    top = max(50, k)
    dense_hits, dense_ms = retrieve(query, k=top, collection=collection, expr=expr)
    sparse_hits, sparse_ms = retrieve_sparse(query, k=top, expr=expr)
    return weighted_rrf(dense_hits, sparse_hits, w_dense, w_sparse, top=k), round(dense_ms + sparse_ms, 2)


def search_hits(
    query: str,
    k: int | None = None,
    collection: str | None = None,
    expr: str | None = None,
    hybrid: bool | None = None,
    retriever: str | None = None,
) -> tuple[list[dict], float]:
    """Elige dense, sparse o hibrido segun el flag del request o la configuracion."""
    validate_collection(collection)
    mode = retriever or settings.retriever
    if hybrid is True:
        return retrieve_hybrid(query, k=k, collection=collection, expr=expr)
    if hybrid is False:
        return retrieve(query, k=k, collection=collection, expr=expr)
    if mode == "solr":
        return search_solr(query, k or settings.top_k)
    if mode == "solr_dense":
        return search_solr_dense(query, k or settings.top_k)
    if mode == "graph":
        return search_graph(query, k or settings.top_k, collection=collection)
    if mode == "hybrid_boost":
        return retrieve_union(query, k=k, collection=collection, expr=expr)
    if mode == "graph_hybrid":
        top_k = k or settings.top_k
        dense_hits, dense_ms = retrieve(query, k=max(20, top_k * 2), collection=collection, expr=expr)
        graph_hits, graph_ms = search_graph(query, k=max(20, top_k * 2), collection=collection)
        return dedupe(dense_hits + graph_hits), round(dense_ms + graph_ms, 2)
    if mode == "hybrid_union":
        return retrieve_union(query, k=k, collection=collection, expr=expr)
    if mode == "sparse":
        return retrieve_sparse(query, k=k, expr=expr)
    if mode == "hybrid":
        return retrieve_hybrid(query, k=k, collection=collection, expr=expr)
    return retrieve(query, k=k, collection=collection, expr=expr)



def expand_parents(
    hits: list[dict],
    name: str,
    top_docs: int = 3,
    max_chunks: int = 8,
) -> list[dict]:
    """Small-to-big: reemplaza el mejor fragmento de cada documento por todos sus chunks."""
    doc_ids: list[str] = []
    for hit in hits:
        doc_id = hit.get("document_id")
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
        if len(doc_ids) >= top_docs:
            break
    if not doc_ids:
        return hits
    client = get_client()
    rows = client.query(
        collection_name=name,
        filter=f"document_id in {json.dumps(doc_ids)}",
        output_fields=OUTPUT_FIELDS + ["chunk_id"],
        limit=top_docs * max_chunks,
    )
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("document_id") in doc_ids:
            by_doc[row["document_id"]].append(dict(row))
    for chunks in by_doc.values():
        chunks.sort(key=lambda item: item.get("position") or 0)
    expanded: list[dict] = []
    seen: set[str] = set()
    done: set[str] = set()
    for hit in hits:
        doc_id = hit.get("document_id")
        if doc_id in doc_ids and doc_id not in done:
            done.add(doc_id)
            for row in by_doc.get(doc_id, [])[:max_chunks]:
                if row["chunk_id"] in seen:
                    continue
                seen.add(row["chunk_id"])
                expanded.append(
                    {
                        **row,
                        "score": hit.get("score", 0.0),
                        "rerank_score": hit.get("rerank_score", 0.0),
                        "parent": True,
                    }
                )
            continue
        chunk_id = hit.get("chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        expanded.append(hit)
    return expanded

def build_context(hits: list[dict], max_tokens: int) -> tuple[str, list[dict]]:
    blocks = []
    used = []
    spent = 0
    for index, hit in enumerate(hits, start=1):
        header = f"[{index}] ({hit.get('title') or 'sin titulo'} | {hit.get('url')} | {hit.get('created_at')})"
        block = f"{header}\n{hit.get('text', '')}"
        cost = estimate_tokens(block)
        if used and spent + cost > max_tokens:
            break
        blocks.append(block)
        used.append(hit)
        spent += cost
    return "\n\n".join(blocks), used


def extract_citations(answer: str) -> list[int]:
    found: set[int] = set()
    for group in CITATION_RE.findall(answer):
        for part in group.split(","):
            part = part.strip()
            if part.isdigit():
                found.add(int(part))
    return sorted(found)


def model_limits(model: str) -> tuple[int, int]:
    """Devuelve (num_ctx, num_predict) para el modelo elegido."""
    if model.startswith("gemma4"):
        return settings.gemma4_input_tokens + settings.gemma4_output_tokens, settings.gemma4_output_tokens
    return settings.model_context_tokens, settings.max_output_tokens


def supports_native_thinking(model: str) -> bool:
    """gemma4 expone el campo thinking de Ollama."""
    return model.startswith("gemma4")


def call_llm(
    prompt: str,
    model: str,
    temperature: float,
    timeout: int | None = None,
    num_predict: int | None = None,
    num_ctx: int | None = None,
    think: bool = False,
    format: str | None = None,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": temperature,
            "seed": 42,
            "num_ctx": num_ctx or settings.model_context_tokens,
            "num_predict": num_predict or settings.max_output_tokens,
        },
    }
    payload["think"] = think
    if format:
        payload["format"] = format
    try:
        response = requests.post(
            f"{settings.llm_url.rstrip('/')}/api/generate",
            json=payload,
            timeout=timeout or settings.request_timeout,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        raise LLMError(f"No se pudo consultar Ollama: {error}") from error


def generate(
    query: str,
    model: str | None = None,
    k: int | None = None,
    temperature: float = 0.0,
    collection: str | None = None,
    expr: str | None = None,
    prompt_version: str | None = None,
    hybrid: bool | None = None,
    think: bool = False,
    domain_gate: bool = False,
    ambiguity_policy: str = "off",
    interactive: bool = False,
    history: list[dict] | None = None,
    original_query: str | None = None,
    retriever: str | None = None,
    parent_expansion: bool | None = None,
) -> dict:
    model = model or settings.llm_model
    prompt_version = prompt_version or settings.prompt_version
    k = k or settings.top_k
    search_k = settings.rerank_top_n if settings.rerank and settings.rerank_top_n > k else k
    with observation(
        "retrieve",
        as_type="retriever",
        input={"query": query, "k": search_k},
        metadata={"collection": collection or settings.collection},
    ) as retrieve_span:
        hits, retrieval_ms = search_hits(
            query, k=search_k, collection=collection, expr=expr, hybrid=hybrid, retriever=retriever
        )
        if retrieve_span is not None:
            retrieve_span.update(
                output={
                    "ms": retrieval_ms,
                    "chunks": [
                        {
                            "chunk_id": hit.get("chunk_id"),
                            "doc_id": hit.get("document_id"),
                            "score": round(float(hit.get("score", 0.0)), 4),
                        }
                        for hit in hits[:20]
                    ],
                }
            )
    base = {
        "query": query,
        "model": model,
        "k": k,
        "collection": collection or settings.collection,
        "retrieval_ms": retrieval_ms,
        "generation_ms": 0.0,
        "total_ms": retrieval_ms,
        "tokens_per_second": None,
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash(prompt_version),
        "citations": [],
        "invalid_citations": [],
        "context_tokens": 0,
        "context_chunks": 0,
        "reasoning": "",
        "reasoning_ms": 0.0,
        "native_thinking": False,
        "domain_probability": None,
        "type": "answer",
    }
    if settings.rerank and len(hits) > 1:
        hits = score_hits(query, hits)
    domain_probability = None
    if domain_gate and hits:
        domain_probability = domain_probability_score(query, hits)
        if domain_probability < settings.domain_threshold:
            return {
                **base,
                "answer": ABSTENTION_TEXT,
                "abstained": True,
                "abstain_reason": "fuera_dominio",
                "evidence": hits,
                "domain_probability": domain_probability,
                "total_ms": retrieval_ms,
            }
    if settings.rerank and len(hits) > 1:
        with observation(
            "rerank",
            input={"query": query, "candidates": len(hits)},
            metadata={"model": settings.rerank_model},
        ) as rerank_span:
            rerank_started = time.perf_counter()
            hits = rerank_hits(query, hits)
            if os.getenv("GRAPH_BOOST") == "1" or retriever == "hybrid_boost":
                hits = boost_hits(query, hits)
            hits = hits[:k]
            if settings.rerank_score_ratio > 0 and hits:
                threshold = hits[0]["rerank_score"] * settings.rerank_score_ratio
                filtered = [hit for hit in hits if hit["rerank_score"] >= threshold]
                hits = filtered or hits[:1]
            retrieval_ms = round(retrieval_ms + (time.perf_counter() - rerank_started) * 1000, 2)
            base["retrieval_ms"] = retrieval_ms
            base["total_ms"] = retrieval_ms
            if rerank_span is not None:
                rerank_span.update(output={"kept": len(hits)})

    if not hits:
        return {**base, "answer": ABSTENTION_TEXT, "abstained": True, "abstain_reason": "sin_evidencia", "evidence": []}

    best = hits[0]["score"]
    if settings.min_score > 0 and best < settings.min_score:
        return {
            **base,
            "answer": ABSTENTION_TEXT,
            "abstained": True,
            "abstain_reason": f"bajo_umbral:{best:.3f}",
            "evidence": hits,
        }

    if ambiguity_policy != "off" and hits:
        policy = ambiguity_policy
        if policy == "auto":
            policy = "clarify" if interactive else "conditional"
        ambiguity_started = time.perf_counter()
        ambiguity_hits = hits
        detect_query = original_query or query
        if needs_detection(detect_query, history):
            deep_hits, deep_ms = retrieve_union(query, dense_k=50, sparse_k=50)
            ambiguity_hits = rerank_hits(query, score_hits(query, deep_hits))[:10]
            retrieval_ms = round(retrieval_ms + deep_ms, 2)
        detection = detect_ambiguity(detect_query, ambiguity_hits, model, history)
        if detection.get("ambiguous"):
            if policy == "clarify":
                payload = clarification_payload(detection, ambiguity_hits)
            else:
                payload = conditional_payload(detection, ambiguity_hits)
            if payload is not None:
                ambiguity_ms = round((time.perf_counter() - ambiguity_started) * 1000, 2)
                return {
                    **base,
                    **payload,
                    "generation_ms": ambiguity_ms,
                    "total_ms": round(retrieval_ms + ambiguity_ms, 2),
                }

    expand = parent_expansion
    if expand is None:
        expand = os.getenv("PARENT_EXPANSION", "1") == "1"
    if expand:
        hits = expand_parents(hits, collection or settings.collection)
    context, used = build_context(hits, settings.max_context_tokens)
    num_ctx, num_predict = model_limits(model)
    native_thinking = think and supports_native_thinking(model)
    reasoning = ""
    reasoning_ms = 0.0
    if think and not native_thinking:
        reasoning_started = time.perf_counter()
        try:
            reasoning_data = call_llm(
                REASONING_PROMPT.format(context=context, question=query),
                model,
                temperature,
                num_predict=settings.reasoning_tokens,
                num_ctx=num_ctx,
            )
            reasoning = (reasoning_data.get("response") or "").strip()
        except LLMError:
            reasoning = ""
        reasoning_ms = round((time.perf_counter() - reasoning_started) * 1000, 2)

    prompt = get_prompt(prompt_version).format(context=context, question=query)
    started = time.perf_counter()
    with observation(
        "generate",
        as_type="generation",
        model=model,
        input={"prompt": prompt[:8000]},
        model_parameters={"temperature": temperature, "num_ctx": num_ctx},
    ) as generation_span:
        data = call_llm(
            prompt,
            model,
            temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
            think=native_thinking,
        )
    generation_ms = round((time.perf_counter() - started) * 1000, 2)

    if native_thinking:
        reasoning = (data.get("thinking") or "").strip()
    answer = (data.get("response") or "").strip()
    eval_count = data.get("eval_count") or 0
    eval_duration = data.get("eval_duration") or 0
    tokens_per_second = round(eval_count / (eval_duration / 1e9), 2) if eval_duration else None
    if generation_span is not None:
        generation_span.update(
            output=answer,
            usage_details={
                "input": int(data.get("prompt_eval_count") or 0),
                "output": int(eval_count),
            },
        )

    citations = extract_citations(answer)
    invalid = [c for c in citations if c < 1 or c > len(used)]
    abstained = answer.lower().startswith(ABSTENTION_TEXT[:40].lower())
    missing_citations = not abstained and not citations

    with observation("grounding", input={"answer": answer[:2000]}) as grounding_span:
        grounding = check_grounding(answer, [hit.get("text", "") for hit in used])
        if grounding_span is not None:
            grounding_span.update(output=grounding)

    return {
        **base,
        "answer": answer,
        "grounding": grounding,
        "abstained": abstained,
        "abstain_reason": "modelo" if abstained else None,
        "missing_citations": missing_citations,
        "evidence": used,
        "generation_ms": generation_ms,
        "total_ms": round(retrieval_ms + reasoning_ms + generation_ms, 2),
        "tokens_per_second": tokens_per_second,
        "reasoning": reasoning,
        "reasoning_ms": reasoning_ms,
        "native_thinking": native_thinking,
        "domain_probability": domain_probability,
        "citations": citations,
        "invalid_citations": invalid,
        "context_tokens": estimate_tokens(context),
        "context_chunks": len(used),
    }
