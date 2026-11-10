"""Verificación de anclaje: entidades de la respuesta presentes en el contexto."""
from __future__ import annotations

from .entities import extract_query, normalize


def check(answer: str, contexts: list[str]) -> dict:
    """Devuelve las entidades de la respuesta que no aparecen en el contexto."""
    if not answer:
        return {"entities": [], "ungrounded": []}
    entities = extract_query(answer)
    context = normalize(" ".join(contexts))
    names = [item["name"] for item in entities]
    ungrounded = [name for name in names if normalize(name) not in context]
    return {"entities": names, "ungrounded": ungrounded}
