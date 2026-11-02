"""Compuerta de dominio sin LLM: embeddings bge-m3 y features de retrieval."""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import settings
from .embeddings import get_embedder

STOPWORDS = {
    "cual", "cuales", "que", "quien", "quienes", "cuando", "donde", "cuanto", "cuantos",
    "cuanta", "cuantas", "como", "por", "para", "con", "sin", "del", "las", "los", "una",
    "unos", "unas", "este", "esta", "estos", "estas", "sobre", "entre", "desde", "hasta",
}
ENTITY_STOP = STOPWORDS | {"segun", "tras", "durante", "mientras", "ademas", "tambien"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", normalize(text)) if token not in STOPWORDS]


def question_type(question: str) -> list[float]:
    text = normalize(question)
    patterns = [
        r"\bcapital\b",
        r"\b(poblacion|habitantes)\b",
        r"\b(quien|escribio|autor|autora)\b",
        r"\b(cuando|ano|fecha)\b",
        r"\b(donde|ubicad|encuentra)\b",
        r"\b(cuantos|cuantas|cuanto|cuanta)\b",
    ]
    vector = [1.0 if re.search(pattern, text) else 0.0 for pattern in patterns]
    vector.append(1.0 if sum(vector) == 0 else 0.0)
    return vector


def retrieval_features(question: str, hits: list[dict]) -> list[float]:
    """Mismas features que el entrenamiento, calculadas sobre el top-5 denso."""
    scores = [float(hit.get("score") or 0.0) for hit in hits[:5]]
    if not scores:
        return [0.0] * 5
    title = str(hits[0].get("title") or "")
    question_tokens = set(tokens(question))
    title_tokens = set(tokens(title))
    overlap = len(question_tokens & title_tokens) / len(question_tokens) if question_tokens else 0.0
    entities = {
        word
        for word in re.findall(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}\b", question)
        if normalize(word) not in ENTITY_STOP
    }
    entity_hits = sum(1 for entity in entities if normalize(entity) in normalize(title))
    return [
        scores[0],
        float(np.mean(scores)),
        scores[0] - (scores[1] if len(scores) > 1 else scores[0]),
        overlap,
        float(entity_hits),
    ]


@lru_cache(maxsize=1)
def _model() -> dict | None:
    path = Path(settings.domain_classifier_path)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        "mean": np.asarray(raw["mean"], dtype=np.float32),
        "scale": np.asarray(raw["scale"], dtype=np.float32),
        "coef": np.asarray(raw["coef"], dtype=np.float32),
        "intercept": float(raw["intercept"]),
        "threshold": float(raw.get("threshold", settings.domain_threshold)),
    }


def domain_probability(question: str, hits: list[dict]) -> float:
    """Probabilidad de que la pregunta sea del dominio del corpus."""
    model = _model()
    if model is None:
        return 1.0
    embedding = get_embedder().embed_queries([question])[0].astype(np.float32)
    features = np.concatenate(
        [
            embedding,
            np.asarray(retrieval_features(question, hits), dtype=np.float32),
            np.asarray(question_type(question), dtype=np.float32),
        ]
    )
    scaled = (features - model["mean"]) / model["scale"]
    logit = float(scaled @ model["coef"] + model["intercept"])
    return float(1.0 / (1.0 + np.exp(-logit)))


def is_domain(question: str, hits: list[dict]) -> tuple[bool, float]:
    model = _model()
    threshold = model["threshold"] if model else settings.domain_threshold
    probability = domain_probability(question, hits)
    return probability >= threshold, probability
