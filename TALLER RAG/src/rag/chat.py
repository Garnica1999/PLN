"""Camino conversacional con tool calling y guardarraíles.

El modelo decide si la pregunta requiere buscar en el corpus y reformula la
consulta. La respuesta la redacta el pipeline validado, con citas. Un guardarraíl
de salida exige citas válidas y que las entidades y números de la respuesta
aparezcan en la evidencia. Los saludos no pasan por el corpus.
"""
from __future__ import annotations

import os
import re
import time
import unicodedata

from .config import settings
from .pipeline import (
    ABSTENTION_TEXT,
    LLMError,
    call_llm,
    extract_citations,
    generate,
    model_limits,
)
from .prompts import prompt_hash

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama
except ImportError:  # la API sigue arrancando sin las dependencias nuevas
    ChatOllama = None

ROUTER_PROMPT = """Eres un enrutador de preguntas sobre un corpus de prensa colombiana y Bluesky.
Si la pregunta trata sobre ese corpus, llama a buscar_corpus con una consulta breve.
Si es un saludo, una charla o una pregunta general que no trata del corpus, responde sin llamar la herramienta."""

JUDGE_PROMPT = """Eres un verificador de respuestas. Responde solo SI o NO.
Responde SI si la respuesta se puede sustentar con la evidencia, aunque requiera una inferencia simple.
Responde NO si la evidencia no contiene el dato o contradice la respuesta.

Evidencia:
{context}

Pregunta: {question}
Respuesta: {answer}"""

CONDENSE_PROMPT = """Eres un reformulador de preguntas. No respondas la pregunta.
Dada la conversacion y la ultima pregunta, escribe una sola pregunta completa que incluya el tema del que se hablaba.
No pidas disculpas. No expliques nada. Devuelve solo la pregunta.

Ejemplo:
Conversacion:
usuario: Cuantos municipios han reportado el Retamo Espinoso?
asistente: 22 [1].
Ultima pregunta: Y en que departamento?
Pregunta completa: En que departamento han reportado la presencia del Retamo Espinoso?"""

CHAT_PROMPT = """Eres el asistente de un corpus de prensa colombiana y Bluesky.
El usuario te saluda o hace charla. Responde en una o dos frases, en español, y ofrécele buscar en el corpus si tiene una pregunta.
No busques en el corpus y no inventes datos."""

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "buscar_corpus",
            "description": "Busca en el corpus de prensa colombiana y Bluesky.",
            "parameters": {
                "type": "object",
                "properties": {
                    "consulta": {"type": "string", "description": "Consulta breve para la búsqueda."},
                },
                "required": ["consulta"],
            },
        },
    }
]

SMALL_TALK_WORDS = {
    "hola", "holi", "buenas", "buenos", "buen", "dia", "dias", "tarde", "tardes",
    "noche", "noches", "hey", "saludos", "tal", "que", "como", "estas", "esta",
    "estoy", "te", "va", "vas", "andas", "haces", "cuentas", "todo", "bien",
    "muy", "y", "tu", "tus", "usted", "me", "le", "les", "la", "el", "los",
    "las", "un", "una", "por", "favor", "gracias", "mucho", "gusto", "encantado",
    "puedes", "puede", "ayudar", "ayuda", "ayudas", "ayudarme", "ahi", "espero",
}
FAREWELL_RE = re.compile(r"\b(gracias|adios|hasta luego|chao|nos vemos)\b")
META_RE = re.compile(r"\b(quien eres|que puedes hacer|como funcionas|que sabes|para que sirves|ayuda)\b")
ENTITY_RE = re.compile(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}\b")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
STOP_WORDS = {
    "para", "como", "donde", "cuando", "cuanto", "cuantos", "cuanta", "cuantas", "sobre", "entre",
    "desde", "hasta", "este", "esta", "estos", "estas", "pero", "porque", "aunque", "tambien",
    "solo", "todo", "toda", "todos", "todas", "hace", "hacia", "segun", "tras", "ante", "bajo",
    "contra", "durante", "mediante", "unos", "unas", "otro", "otra", "otros", "otras", "mismo",
    "misma", "tiene", "tienen", "dicen", "dice", "dijo", "sera", "seran", "fueron", "puede",
}
STOP_ENTITIES = {
    "la", "el", "los", "las", "un", "una", "unos", "unas", "no", "si", "en", "de", "del", "al",
    "segun", "este", "esta", "estos", "estas", "para", "por", "con", "sin", "que", "como",
    "donde", "cuando", "cuanto", "cuantos", "cuanta", "cuantas", "hay", "habia", "hubo",
    "existe", "existen", "son", "es", "era", "eran", "fue", "fueron", "estan",
    "queda", "quedan", "tiene", "tienen", "actualmente", "ademas", "tambien", "pero",
    "mientras", "tras", "durante", "entre", "sobre", "bajo", "desde", "hasta", "se", "su",
    "sus", "lo", "le", "les", "nos", "ya", "solo", "mas", "muy", "todo", "toda", "todos",
    "todas", "otro", "otra", "otros", "otras", "primer", "primera", "gran", "grande",
    "grandes", "nuevo", "nueva", "mayor", "menor", "mejor", "peor",
    "adicionalmente", "especificamente", "garantizar", "promover", "ser", "reiterar",
    "finalmente", "asimismo", "igualmente", "principalmente", "posteriormente",
    "inicialmente", "concretamente", "particularmente", "efectivamente", "ciertamente",
    "destacar", "cabe", "senalar", "mencionar", "indicar", "explicar", "confirmar",
    "asegurar", "afirmar", "sostener", "agregar", "anadir", "concluir", "resaltar",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip()


def is_small_talk(text: str) -> bool:
    """True si la consulta solo tiene saludo, cortesía o preguntas de cortesía."""
    words = re.findall(r"[a-z0-9]+", text)
    return bool(words) and all(word in SMALL_TALK_WORDS for word in words)


def classify(query: str) -> str:
    """Separa charla de pregunta de dominio con reglas, sin llamar al modelo."""
    text = normalize(query)
    words = text.split()
    if META_RE.search(text):
        return "charla"
    if is_small_talk(text):
        return "charla"
    if FAREWELL_RE.search(text) and len(words) <= 6 and "?" not in query:
        return "charla"
    return "dominio"


def unsupported_terms(answer: str, context: str, question: str = "") -> list[str]:
    """Entidades y números de la respuesta que no aparecen en la evidencia ni en la pregunta."""
    text = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "", answer)
    context_norm = normalize(context)
    question_norm = normalize(question)
    entities = [
        entity
        for entity in ENTITY_RE.findall(text)
        if normalize(entity) not in STOP_ENTITIES
        and normalize(entity) not in context_norm
        and normalize(entity) not in question_norm
    ]
    numbers = [
        number
        for number in NUMBER_RE.findall(text)
        if number not in context_norm
        and re.sub(r"[.,]", "", number) not in context_norm
        and number not in question_norm
    ]
    return entities + numbers


SPANISH_HINTS = {"el", "la", "de", "que", "en", "se", "por", "con", "para", "una", "del", "los", "las"}


def _is_spanish(text: str) -> bool:
    return len(set(normalize(text).split()) & SPANISH_HINTS) >= 2


def content_coverage(answer: str, support: str) -> float:
    """Proporcion de palabras de contenido de la respuesta que aparecen en el chunk citado."""
    words = [word for word in re.findall(r"\w{4,}", normalize(answer)) if word not in STOP_WORDS]
    if not words:
        return 1.0
    support_norm = normalize(support)
    return sum(1 for word in words if word in support_norm) / len(words)


def apply_guardrail(
    answer: str, intent: str, hits: list[dict], context: str = "", question: str = ""
) -> tuple[str, list[int], list[int], str | None, bool]:
    """Una respuesta de dominio sin citas válidas o sin respaldo no se entrega."""
    answer = (answer or "").strip()
    citations = extract_citations(answer)
    invalid = [citation for citation in citations if citation < 1 or citation > len(hits)]
    abstained = answer.lower().startswith(ABSTENTION_TEXT[:40].lower())
    if intent == "charla":
        return answer, citations, invalid, None, False
    if abstained:
        return answer, citations, invalid, None, True
    if not citations:
        return ABSTENTION_TEXT, [], [], "sin_citas", True
    if invalid:
        return ABSTENTION_TEXT, [], invalid, "citas_invalidas", True
    cited_hits = [hits[citation - 1] for citation in citations if 1 <= citation <= len(hits)]
    cited = [hit.get("text", "") for hit in cited_hits]
    meta = [
        " ".join(str(hit.get(field, "")) for field in ("title", "created_at", "url"))
        for hit in cited_hits
    ]
    text_support = "\n".join(cited) or context
    support = "\n".join(part for part in [text_support, *meta] if part.strip()) if cited_hits else context
    if support:
        cross_language = _is_spanish(answer) and any(
            hit.get("language") not in (None, "", "es") for hit in cited_hits
        )
        if not cross_language:
            missing = unsupported_terms(answer, support, question)
            if missing:
                return ABSTENTION_TEXT, [], [], f"sin_respaldo:{','.join(missing[:3])}", True
            if content_coverage(answer, support) < 0.5:
                return ABSTENTION_TEXT, [], [], "sin_respaldo:cobertura", True
    return answer, citations, invalid, None, False


def judge(answer: str, question: str, context: str, model: str) -> bool:
    """Segunda llamada: la evidencia debe sostener la respuesta."""
    prompt = JUDGE_PROMPT.format(question=question, context=context[:3000], answer=answer)
    data = call_llm(prompt, os.getenv("JUDGE_MODEL", "gemma2:2b"), 0.0, timeout=30)
    verdict = (data.get("response") or "").strip().lower()
    return re.match(r"^s[ií]\b", verdict) is not None


def condense_query(history: list[dict], query: str) -> str:
    """Convierte un seguimiento conversacional en una consulta autonoma."""
    if not history or ChatOllama is None:
        return query
    messages = [SystemMessage(content=CONDENSE_PROMPT)]
    for turn in history[-6:]:
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=query))
    try:
        llm = ChatOllama(model=os.getenv("CONDENSE_MODEL", "gemma2:2b"), base_url=settings.llm_url, temperature=0)
        response = llm.invoke(messages)
    except Exception:  # noqa: BLE001
        return query
    candidate = _text(response.content).strip()
    first_user = next((turn.get("content", "") for turn in history if turn.get("role") == "user"), "")
    topic = {word for word in re.findall(r"\w{4,}", normalize(first_user))}
    proposed = {word for word in re.findall(r"\w{4,}", normalize(candidate))}
    bad = (
        not candidate
        or len(candidate.split()) > 40
        or candidate.lower().startswith(("lo siento", "no tengo", "no puedo", "no hay"))
        or not (topic & proposed)
    )
    if bad:
        return f"{first_user} {query}".strip() if first_user else query
    return candidate


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content or "")


def _no_tools(error: Exception) -> bool:
    return "does not support tools" in str(error).lower()


def _chat_answer(query: str, model: str, temperature: float) -> tuple[str, float]:
    if ChatOllama is None:
        return "Hola. Puedo buscar en el corpus de prensa y Bluesky. ¿Qué quieres consultar?", 0.0
    started = time.perf_counter()
    try:
        llm = ChatOllama(model=model, base_url=settings.llm_url, temperature=temperature)
        response = llm.invoke([SystemMessage(content=CHAT_PROMPT), HumanMessage(content=query)])
    except Exception as error:
        raise LLMError(f"No se pudo consultar Ollama: {error}") from error
    return _text(response.content).strip(), round((time.perf_counter() - started) * 1000, 2)


def route_query(query: str, model: str, temperature: float) -> dict | None:
    """Devuelve la consulta reformulada por el modelo, o None si no usa tools."""
    if ChatOllama is None:
        return None
    try:
        num_ctx, num_predict = model_limits(model)
        llm = ChatOllama(
            model=model,
            base_url=settings.llm_url,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
        )
        bound = llm.bind_tools(TOOL_SCHEMA)
        response = bound.invoke([SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=query)])
    except Exception as error:
        if _no_tools(error):
            return None
        raise LLMError(f"No se pudo consultar Ollama: {error}") from error
    calls = getattr(response, "tool_calls", None) or []
    if not calls:
        return None
    return {"tool_calls": [call["name"] for call in calls]}


def chat(
    query: str,
    model: str | None = None,
    k: int | None = None,
    temperature: float = 0.0,
    collection: str | None = None,
    expr: str | None = None,
    hybrid: bool | None = None,
    history: list[dict] | None = None,
    think: bool = False,
    strict: bool = False,
    ambiguity_policy: str = "off",
) -> dict:
    model = model or settings.llm_model
    k = k or settings.top_k
    intent = classify(query)
    started = time.perf_counter()
    base = {
        "query": query,
        "model": model,
        "k": k,
        "collection": collection or settings.collection,
        "intent": intent,
        "guardrail": None,
        "citations": [],
        "invalid_citations": [],
        "evidence": [],
        "tool_calls": [],
        "retrieval_ms": 0.0,
        "generation_ms": 0.0,
        "total_ms": 0.0,
        "tokens_per_second": None,
        "prompt_version": settings.prompt_version,
        "prompt_hash": prompt_hash(settings.prompt_version),
        "context_tokens": 0,
        "context_chunks": 0,
        "reasoning": "",
        "reasoning_ms": 0.0,
        "native_thinking": False,
        "domain_probability": None,
    }

    if intent == "charla":
        answer, generation_ms = _chat_answer(query, model, temperature)
        return {
            **base,
            "answer": answer,
            "abstained": False,
            "abstain_reason": None,
            "generation_ms": generation_ms,
            "total_ms": round((time.perf_counter() - started) * 1000, 2),
            "guardrail": "charla",
        }

    search_query = condense_query(history or [], query)
    route = None if strict else route_query(search_query, model, temperature)
    result = generate(
        search_query,
        model=model,
        k=k,
        temperature=temperature,
        collection=collection,
        expr=expr,
        hybrid=hybrid,
        think=think,
        domain_gate=strict,
        ambiguity_policy=ambiguity_policy,
        interactive=history is not None,
        history=history,
    )
    if result.get("type") in ("clarification", "conditional"):
        return {
            **result,
            "query": query,
            "search_query": search_query,
            "intent": intent,
            "total_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    context = "\n".join(hit.get("text", "") for hit in result["evidence"])
    answer, citations, invalid, guardrail, abstained = apply_guardrail(
        result["answer"], intent, result["evidence"], context, search_query
    )
    cited = [result["evidence"][citation - 1] for citation in citations if 1 <= citation <= len(result["evidence"])]
    judge_context = "\n".join(
        f"{hit.get('title', '')} {hit.get('created_at', '')}\n{hit.get('text', '')}" for hit in cited
    ) or context
    if not abstained and citations and not judge(answer, search_query, judge_context, model):
        answer, citations, invalid, guardrail, abstained = ABSTENTION_TEXT, [], [], "juez_no", True
    reason = result.get("abstain_reason") or guardrail if abstained else None
    return {
        **result,
        "query": query,
        "search_query": search_query,
        "answer": answer,
        "citations": citations,
        "invalid_citations": invalid,
        "abstained": abstained,
        "abstain_reason": reason,
        "guardrail": guardrail or ("tool_calling" if route else "fallback_sin_tools"),
        "intent": intent,
        "tool_calls": route["tool_calls"] if route else [],
        "total_ms": round((time.perf_counter() - started) * 1000, 2),
    }
