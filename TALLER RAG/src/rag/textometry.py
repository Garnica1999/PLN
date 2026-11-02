"""Textometría y detección de near-duplicates sobre el corpus."""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter

import numpy as np

WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
HTML_RE = re.compile(r"<[^>]{1,200}>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")
REPEAT_RE = re.compile(r"([!?.,])\1{2,}")
SPACE_RE = re.compile(r"[ \t]+")
NEWLINE_RE = re.compile(r"\n{3,}")

NAV_TOKENS = {
    "inicio",
    "menu",
    "menú",
    "suscribete",
    "suscríbete",
    "compartir",
    "publicidad",
    "cookie",
    "cookies",
    "newsletter",
    "home",
    "subscribe",
    "share",
    "advertisement",
    "login",
}

ES_STOP = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para",
    "con", "no", "una", "su", "al", "lo", "como", "mas", "más", "pero", "sus", "le", "ya",
    "o", "este", "si", "porque", "esta", "entre", "cuando", "muy", "sin", "sobre", "tambien",
}
EN_STOP = {
    "the", "of", "and", "to", "in", "a", "is", "that", "for", "it", "as", "was", "with",
    "be", "by", "on", "not", "he", "i", "this", "are", "or", "his", "from", "at", "which",
    "but", "have", "an", "they", "you", "were", "their", "said", "we", "she",
}


def tokenize(text: str) -> list[str]:
    return [word.lower() for word in WORD_RE.findall(text or "")]


def simhash64(text: str, shingle_size: int = 5) -> int:
    tokens = tokenize(text)
    if len(tokens) < shingle_size:
        shingles = [text[i : i + 4] for i in range(max(0, len(text) - 3))]
    else:
        shingles = [" ".join(tokens[i : i + shingle_size]) for i in range(len(tokens) - shingle_size + 1)]
    if not shingles:
        return 0
    vector = [0] * 64
    for shingle in shingles:
        digest = int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            vector[bit] += 1 if (digest >> bit) & 1 else -1
    result = 0
    for bit in range(64):
        if vector[bit] > 0:
            result |= 1 << bit
    return result


POPCOUNT = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def hamming_neighbors(hashes: np.ndarray, max_distance: int = 3) -> list[tuple[int, int]]:
    pairs = []
    total = len(hashes)
    for index in range(total - 1):
        xor = hashes[index + 1 :] ^ hashes[index]
        distances = POPCOUNT[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)
        for offset in np.where(distances <= max_distance)[0]:
            pairs.append((index, index + 1 + int(offset)))
    return pairs


def _syllables_ratio(tokens: list[str], stopwords: set[str]) -> float:
    if not tokens:
        return 0.0
    return sum(1 for token in tokens if token in stopwords) / len(tokens)


def document_metrics(text: str) -> dict:
    raw = text or ""
    tokens = tokenize(raw)
    counts = Counter(tokens)
    types = len(counts)
    hapax = sum(1 for value in counts.values() if value == 1)
    words = len(tokens) or 1
    nav_hits = sum(1 for token in tokens if token in NAV_TOKENS)
    es_ratio = _syllables_ratio(tokens, ES_STOP)
    en_ratio = _syllables_ratio(tokens, EN_STOP)
    stripped = raw.rstrip()
    truncated = bool(stripped) and stripped[-1] not in ".!?\"'”’)]}…"
    return {
        "chars": len(raw),
        "words": len(tokens),
        "tokens_est": max(1, (len(raw) + 3) // 4),
        "types": types,
        "ttr": types / words,
        "hapax": hapax,
        "hapax_ratio": hapax / max(types, 1),
        "guiraud": types / math.sqrt(words) if words else 0.0,
        "html": len(HTML_RE.findall(raw)),
        "urls": len(URL_RE.findall(raw)),
        "emails": len(EMAIL_RE.findall(raw)),
        "emojis": len(EMOJI_RE.findall(raw)),
        "newlines": raw.count("\n"),
        "repeat_punct": len(REPEAT_RE.findall(raw)),
        "nav": nav_hits,
        "nav_ratio": nav_hits / words,
        "es_stop_ratio": round(es_ratio, 4),
        "en_stop_ratio": round(en_ratio, 4),
        "code_switching": int(es_ratio > 0.08 and en_ratio > 0.08),
        "truncated": int(truncated),
        "empty": int(not raw.strip()),
    }


def conservative_clean(text: str) -> str:
    """V2: NFKC, quita HTML y URLs, colapsa espacios. No toca mayúsculas ni acentos."""
    value = unicodedata.normalize("NFKC", text or "")
    value = HTML_RE.sub(" ", value)
    value = URL_RE.sub(" ", value)
    value = SPACE_RE.sub(" ", value)
    value = NEWLINE_RE.sub("\n\n", value)
    return value.strip()


def aggressive_clean(text: str) -> str:
    """V4: control negativo. Minúsculas, sin acentos y sin puntuación."""
    value = unicodedata.normalize("NFKD", conservative_clean(text))
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.lower()
    value = re.sub(r"[^\w\s]", " ", value)
    return SPACE_RE.sub(" ", value).strip()


def top_terms(texts: list[str], limit: int = 20) -> list[tuple[str, int]]:
    counter = Counter()
    for text in texts:
        counter.update(token for token in tokenize(text) if token not in ES_STOP and token not in EN_STOP)
    return counter.most_common(limit)


def exclusive_terms(texts_a: list[str], texts_b: list[str], limit: int = 20, min_count: int = 10) -> list[tuple[str, float, int, int]]:
    counter_a = Counter()
    counter_b = Counter()
    for text in texts_a:
        counter_a.update(token for token in tokenize(text) if token not in ES_STOP and token not in EN_STOP)
    for text in texts_b:
        counter_b.update(token for token in tokenize(text) if token not in ES_STOP and token not in EN_STOP)
    total_a = sum(counter_a.values()) or 1
    total_b = sum(counter_b.values()) or 1
    scored = []
    for term, count in counter_a.items():
        if count < min_count:
            continue
        rate_a = count / total_a
        rate_b = counter_b.get(term, 0) / total_b
        scored.append((term, rate_a / (rate_b + 1e-9), count, counter_b.get(term, 0)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]
