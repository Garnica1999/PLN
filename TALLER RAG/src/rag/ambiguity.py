"""Deteccion y tratamiento de ambiguedad referencial en preguntas RAG."""
from __future__ import annotations

import json
import re
import unicodedata

from .config import settings

GENERIC_REFERENCES = (
    "la iniciativa", "el concierto", "la reforma", "el proyecto", "el acuerdo",
    "la cumbre", "la cop", "el partido", "la empresa", "el presidente",
    "el ministro", "la ley", "el programa", "el plan", "la campana",
    "el terremoto", "el sismo", "la cumbre", "el tratado", "el estudio", "el informe",
    "the initiative", "the concert", "the reform", "the project", "the agreement",
    "the summit", "the cop", "the match", "the company", "the president",
    "the minister", "the law", "the program", "the plan", "the campaign",
    "the study", "the report",
)

DETECTION_PROMPT = """Eres un analista de ambiguedad en preguntas sobre un corpus de noticias.
Recibes una pregunta y hasta diez evidencias numeradas.
Decide si la pregunta tiene mas de una interpretacion posible segun las evidencias.
Devuelve solo JSON con esta forma:
{{
  "ambiguous": true,
  "type": "referencial",
  "clarifying_question": "pregunta breve para que el usuario elija",
  "options": [
    {{"label": "opcion corta", "query": "pregunta reescrita y especifica", "answer": "respuesta de esa interpretacion", "evidence": [1, 2]}}
  ]
}}
Reglas:
- Si la pregunta tiene una sola interpretacion, devuelve {{"ambiguous": false, "options": []}}.
- Si hay ambiguedad, devuelve entre 2 y 3 opciones, todas respaldadas por las evidencias.
- La query debe resolver la ambiguedad, ser una pregunta completa y no mencionar "opcion", "el texto", "la evidencia" ni numeros de evidencia.
- No inventes interpretaciones que no aparezcan en las evidencias.
- La respuesta debe ser corta, de 1 a 12 palabras, e incluir la cifra o entidad exacta que responde, no una parafrasis.
- Si la pregunta pide una unidad o medida, incluye la unidad y la escala en la answer.
- Si la pregunta compara, incluye el ano, la region o la fuente en la label y en la query.
- Si hay historial, usalo para resolver el sujeto omitido.
- Si el historial fija un tema, cada opcion debe responder a ese mismo tema para el nuevo sujeto, no a un tema lateral de la evidencia.

PREGUNTA: {question}
HISTORIAL:
{history}
EVIDENCIAS:
{evidence}
"""

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip()


def needs_detection(question: str, history: list[dict] | None = None) -> bool:
    """Solo gasta una llamada al LLM cuando hay una senal de ambiguedad."""
    text = normalize(question)
    if any(reference in text for reference in GENERIC_REFERENCES):
        return True
    if re.search(r"\b(mayor|menor|mejor|peor|mas|menos|highest|lowest|best|worst)\b", text) and len(text.split()) <= 16:
        return True
    if re.search(r"\b(cuanto|cuanta|how much)\b", text) and any(
        unit in text for unit in ("cuesta", "precio", "hectarea", "kilo", "dolar", "peso", "euro")
    ):
        return True
    if history and (len(text.split()) <= 8 or text.startswith("y ")):
        return True
    return False


def evidence_block(hits: list[dict], limit: int = 10, chars: int = 600) -> str:
    lines = []
    for index, hit in enumerate(hits[:limit], start=1):
        title = str(hit.get("title") or "sin titulo")
        text = str(hit.get("text") or "").replace("\n", " ")[:chars]
        lines.append(f"[{index}] ({title}) {text}")
    return "\n".join(lines)


def parse_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _llm(prompt: str, model: str) -> dict:
    from .pipeline import call_llm

    return call_llm(prompt, model, 0.0, num_predict=500, format="json")




QUANTITY_PATTERN = re.compile(
    r"(\d[\d.,]*)\s*(votos?|municipios?|personas?|kilos?|hectáreas?|millones?|billones?)",
    re.IGNORECASE,
)


def _question_unit(question: str) -> str | None:
    text = normalize(question)
    for unit in ("municipios", "votos", "personas", "kilos", "hectareas", "millones", "billones"):
        if unit.rstrip("s") in text:
            return unit.rstrip("s")
    return None


def _quantity_mentions(question: str, hits: list[dict]) -> list[dict]:
    text = normalize(question)
    if not any(word in text for word in ("cuantos", "cuantas", "cuanto", "cuanta", "how many", "votos", "voto")):
        return []
    mentions = []
    for index, hit in enumerate(hits[:10], start=1):
        body = f"{hit.get('title') or ''} {hit.get('text') or ''}"
        for match in QUANTITY_PATTERN.finditer(body):
            number = match.group(1).rstrip(".,")
            unit = match.group(2).lower()
            mentions.append(
                {
                    "number": number,
                    "unit": unit,
                    "index": index,
                    "title": str(hit.get("title") or "")[:90],
                }
            )
    return mentions


def augment_quantity_options(question: str, hits: list[dict], options: list[dict]) -> list[dict]:
    """Anade la primera cifra de votos que el LLM no propuso."""
    if len(options) >= 6:
        return options
    existing = {
        number.rstrip(".,")
        for option in options
        for number in re.findall(r"\d[\d.,]*", option["answer"])
    }
    unit_root = _question_unit(question)
    for mention in _quantity_mentions(question, hits):
        if unit_root and not mention["unit"].startswith(unit_root):
            continue
        answer = f"{mention['number']} {mention['unit']}"
        if mention["number"] in existing:
            continue
        title = mention["title"] or f"iniciativa con {mention['number']} votos"
        clean_title = re.split(r"\s[-|]\s", title)[0]
        if mention["unit"] in ("voto", "votos"):
            query = f"¿Cuántos votos a favor obtuvo la iniciativa sobre {clean_title}?"
        else:
            query = f"{question.rstrip('?')} según {clean_title}?"
        options.append(
            {
                "label": title,
                "query": query,
                "answer": answer,
                "evidence": [mention["index"]],
            }
        )
        existing.add(mention["number"])
    return options


def augment_quantity_conditions(question: str, hits: list[dict], conditions: list[dict]) -> list[dict]:
    """Anade la primera cifra de votos que el LLM no incluyo como condicion."""
    if len(conditions) >= 4:
        return conditions
    existing = {
        number.rstrip(".,")
        for condition in conditions
        for number in re.findall(r"\d[\d.,]*", condition["answer"])
    }
    unit_root = _question_unit(question)
    for mention in _quantity_mentions(question, hits):
        if unit_root and not mention["unit"].startswith(unit_root):
            continue
        answer = f"{mention['number']} {mention['unit']}"
        if mention["number"] in existing:
            continue
        conditions.append(
            {
                "label": mention["title"] or f"iniciativa con {mention['number']} votos",
                "answer": answer,
                "citations": [mention["index"]],
            }
        )
        existing.add(mention["number"])
        break
    return conditions



AMOUNT_PATTERN = re.compile(
    r"\$?\s*(\d[\d.,]*)\s*(billones|millones|mil millones)",
    re.IGNORECASE,
)


def _amount_mentions(question: str, hits: list[dict]) -> list[dict]:
    text = normalize(question)
    if not (re.search(r"\bcuanto\b", text) or "que dijo" in text or "what did" in text or "how much" in text):
        return []
    mentions = []
    for index, hit in enumerate(hits[:10], start=1):
        body = f"{hit.get('title') or ''} {hit.get('text') or ''}"
        for match in AMOUNT_PATTERN.finditer(body):
            number = match.group(1).rstrip(".,")
            unit = match.group(2).lower()
            mentions.append({"number": number, "unit": unit, "index": index, "title": str(hit.get("title") or "")[:90]})
    return mentions


def augment_amount_options(question: str, hits: list[dict], options: list[dict]) -> list[dict]:
    """Anade la primera cifra monetaria que el LLM no propuso."""
    if len(options) >= 6:
        return options
    existing = {
        number.rstrip(".,")
        for option in options
        for number in re.findall(r"\d[\d.,]*", option["answer"])
    }
    for mention in _amount_mentions(question, hits):
        if mention["number"] in existing:
            continue
        amount = f"{mention['number']} {mention['unit']}"
        options.append(
            {
                "label": f"Cifra de {amount}",
                "query": f"{question.rstrip('?')} de {amount}?",
                "answer": amount,
                "evidence": [mention["index"]],
            }
        )
        existing.add(mention["number"])
    return options

def detect(question: str, hits: list[dict], model: str | None = None, history: list[dict] | None = None, force: bool = False) -> dict:
    """Devuelve la ambiguedad detectada y las opciones, o una respuesta vacia."""
    if not hits or (not force and not needs_detection(question, history)):
        return {"ambiguous": False, "options": []}
    model = model or settings.llm_model
    history_text = ""
    if history:
        turns = history[-3:]
        history_text = "\n".join(f"{turn.get('role')}: {turn.get('content')}" for turn in turns)
    prompt = DETECTION_PROMPT.format(question=question, evidence=evidence_block(hits), history=history_text or "(sin historial)")
    try:
        data = parse_json(_llm(prompt, model).get("response") or "")
    except Exception:
        return {"ambiguous": False, "options": []}
    if not data.get("ambiguous"):
        return {"ambiguous": False, "options": []}
    options = []
    for option in data.get("options") or []:
        label = str(option.get("label") or "").strip()
        query = str(option.get("query") or "").strip()
        answer = str(option.get("answer") or "").strip()
        evidence = option.get("evidence") or []
        if not label or not query or not answer:
            continue
        valid = [int(index) for index in evidence if isinstance(index, int) and 1 <= index <= len(hits)]
        options.append({"label": label, "query": query, "answer": answer, "evidence": valid})
    options = augment_quantity_options(question, hits, options)
    options = augment_amount_options(question, hits, options)
    if len(options) < 2:
        return {"ambiguous": False, "options": []}
    return {
        "ambiguous": True,
        "type": str(data.get("type") or "referencial"),
        "clarifying_question": str(data.get("clarifying_question") or "La pregunta tiene varias interpretaciones. ¿A cuál te refieres?"),
        "options": options[:6],
    }


def clarification_payload(detection: dict, hits: list[dict]) -> dict:
    return {
        "type": "clarification",
        "answer": detection.get("clarifying_question") or "La pregunta tiene varias interpretaciones. ¿A cuál te refieres?",
        "clarification": detection,
        "evidence": hits[:10],
        "abstained": False,
        "abstain_reason": None,
        "citations": [],
    }


def conditional_payload(detection: dict, hits: list[dict]) -> dict | None:
    """Respuesta condicional construida con las mismas opciones de la clarificacion."""
    options = detection.get("options") or []
    if len(options) < 2:
        return None
    conditions = [
        {
            "label": option["label"],
            "answer": option["answer"],
            "citations": option.get("evidence") or [1],
        }
        for option in options
    ]

    lines = []
    lines.append("La pregunta tiene varias interpretaciones:")
    for condition in conditions:
        cites = ", ".join(f"[{index}]" for index in condition["citations"])
        lines.append(f"- Para {condition['label']}: {condition['answer']} {cites}")
    citations = sorted({index for condition in conditions for index in condition["citations"]})
    return {
        "type": "conditional",
        "answer": "\n".join(lines),
        "conditions": conditions,
        "recommendation": None,
        "evidence": hits[:10],
        "abstained": False,
        "abstain_reason": None,
        "citations": citations,
    }
