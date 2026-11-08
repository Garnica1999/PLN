"""Segmentación del texto en fragmentos con offsets sobre el texto original."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    start: int
    end: int
    text: str


def chunk_identity(
    variant: str,
    document_id: str,
    strategy: str,
    chunk_size: int,
    max_tokens: int,
    overlap: int,
    overlap_sentences: int,
    start: int,
    text_hash: str,
) -> str:
    """Hash estable que incluye todos los parametros de segmentacion."""
    raw = (
        f"{variant}|{document_id}|{strategy}|{chunk_size}|{max_tokens}|"
        f"{overlap}|{overlap_sentences}|{start}|{text_hash}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def estimate_tokens(text: str) -> int:
    """Estimación gruesa para cuando no hay tokenizador disponible (4 caracteres por token)."""
    return max(1, (len(text) + 3) // 4)


SENTENCE_RE = re.compile(r"[^.!?\u2026\n]+[.!?\u2026]*[ \t]*\n?")


def split_sentences(text: str) -> list[tuple[int, int]]:
    spans = []
    for match in SENTENCE_RE.finditer(text):
        start, end = match.span()
        if text[start:end].strip():
            spans.append((start, end))
    return spans


def chunk_fixed(text: str, size: int = 1000, overlap: int = 150) -> Iterator[Chunk]:
    if size <= 0:
        raise ValueError("size debe ser positivo")
    if not 0 <= overlap < size:
        raise ValueError("overlap debe ser menor que size")
    pos = 0
    total = len(text)
    while pos < total:
        end = min(pos + size, total)
        yield Chunk(pos, end, text[pos:end])
        if end >= total:
            break
        pos = end - overlap


def chunk_sentences(
    text: str,
    max_tokens: int = 384,
    overlap_sentences: int = 1,
    count_tokens: Callable[[str], int] = estimate_tokens,
    chars_per_token: int = 4,
) -> Iterator[Chunk]:
    if max_tokens <= 0:
        raise ValueError("max_tokens debe ser positivo")
    overlaps = max(0, overlap_sentences)
    spans = split_sentences(text)
    if not spans:
        if text.strip():
            yield Chunk(0, len(text), text)
        return

    total = len(spans)
    index = 0
    max_chars = max_tokens * chars_per_token
    while index < total:
        start = spans[index][0]
        end = spans[index][1]
        last = index
        while last + 1 < total:
            candidate_end = spans[last + 1][1]
            if count_tokens(text[start:candidate_end]) > max_tokens:
                break
            last += 1
            end = candidate_end

        if last == index and count_tokens(text[start:end]) > max_tokens:
            for cut in range(start, end, max_chars):
                yield Chunk(cut, min(cut + max_chars, end), text[cut:min(cut + max_chars, end)])
        else:
            yield Chunk(start, end, text[start:end])

        if last + 1 >= total:
            break
        index = max(last + 1 - overlaps, index + 1)
