"""Clusters de near-duplicados (porte fiel de src/cv.py:33-229).

Construye los clusters de duplicado exacto o near-duplicado sobre el conjunto completo de textos
(train + test, sin usar etiquetas) con *blocking* por tokens raros y union-find. Es el soporte del
protocolo congelado de folds (D-003). ASCII puro. Sin efectos en import.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

#: Longitud minima del texto normalizado para participar en el agrupamiento.
MIN_LEN = 20
#: Umbral de Jaccard sobre conjuntos de tokens para considerar near-duplicado.
JACCARD_THRESHOLD = 0.8
#: Numero de tokens raros usados como clave de bloqueo.
BLOCK_TOKENS = 3
#: Frecuencia documental maxima para que un token sirva como clave de bloqueo.
MAX_DF_FOR_KEY = 200
#: Tope de documentos comparados dentro de un mismo bloque (evita costo cuadratico).
MAX_BLOCK_SIZE = 400

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_RT_RE = re.compile(r"^\s*rt\s*:?\s*", flags=re.IGNORECASE)
_NOISE_RE = re.compile("[^0-9a-z\\u00e1\\u00e9\\u00ed\\u00f3\\u00fa\\u00fc\\u00f1 ]+")
_SPACES_RE = re.compile(r"\s+")


def normalize_for_dedup(series: pd.Series) -> pd.Series:
    """Normaliza el texto para detectar duplicados y near-duplicados (src/cv.py:51-73)."""
    text = series.fillna("").astype(str).str.lower()
    text = text.str.replace(_URL_RE, " ", regex=True)
    text = text.str.replace(_MENTION_RE, " ", regex=True)
    text = text.str.replace(_RT_RE, "", regex=True)
    text = text.str.replace(_NOISE_RE, " ", regex=True)
    return text.str.replace(_SPACES_RE, " ", regex=True).str.strip()


def dedup_signature(normalized: pd.Series, chars: int = 60) -> pd.Series:
    """Firma informativa de prefijo (solo metadato, no agrupa; src/cv.py:76-78)."""
    return normalized.str.slice(0, chars)


def jaccard(left: set, right: set) -> float:
    """Similitud de Jaccard entre dos conjuntos de tokens (src/cv.py:81-93)."""
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    if len(left) / len(right) < JACCARD_THRESHOLD and len(left) < JACCARD_THRESHOLD * len(right):
        return 0.0
    intersection = len(left & right)
    if intersection == 0:
        return 0.0
    return intersection / (len(left) + len(right) - intersection)


class _UnionFind:
    """Union-find con compresion de caminos (src/cv.py:96-115)."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        """Raiz del conjunto de ``item``."""
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, left: int, right: int) -> None:
        """Une los conjuntos de ``left`` y ``right``."""
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def construir_clusters(textos: pd.Series, min_len: int = MIN_LEN,
                       threshold: float = JACCARD_THRESHOLD,
                       block_tokens: int = BLOCK_TOKENS,
                       max_df_for_key: int = MAX_DF_FOR_KEY,
                       max_block_size: int = MAX_BLOCK_SIZE,
                       pair_budget: int = 20_000_000,
                       stats: dict | None = None) -> pd.DataFrame:
    """Agrupa textos en clusters de duplicado exacto o near-duplicado (src/cv.py:118-229).

    Dos filas caen en el mismo cluster si su texto normalizado es identico o si la similitud de
    Jaccard entre sus conjuntos de tokens es >= ``threshold``. El blocking por los ``block_tokens``
    tokens mas raros evita el costo cuadratico.
    """
    normalized = normalize_for_dedup(textos)
    signature = dedup_signature(normalized)
    normalized_array = normalized.to_numpy()
    valid = normalized.str.len().to_numpy() >= min_len
    union = _UnionFind(len(normalized))

    exact_groups: dict = defaultdict(list)
    for position, key in enumerate(normalized_array):
        if valid[position]:
            exact_groups[key].append(position)
    for positions in exact_groups.values():
        anchor = positions[0]
        for position in positions[1:]:
            union.union(anchor, position)

    token_sets = [frozenset(tokens.split()) for tokens in normalized_array]
    document_frequency: Counter = Counter()
    for tokens in token_sets:
        document_frequency.update(tokens)

    blocks: dict = defaultdict(list)
    for position, tokens in enumerate(token_sets):
        if not valid[position]:
            continue
        rare = sorted((document_frequency[token], token) for token in tokens
                      if document_frequency[token] <= max_df_for_key)
        for _, token in rare[:block_tokens]:
            blocks[token].append(position)

    comparisons = 0
    budget_exhausted = False
    for documents in sorted(blocks.values(), key=len):
        if len(documents) < 2:
            continue
        if len(documents) > max_block_size:
            documents = documents[:max_block_size]
        for left_index in range(len(documents)):
            left = documents[left_index]
            left_tokens = token_sets[left]
            if len(left_tokens) < 2:
                continue
            for right_index in range(left_index + 1, len(documents)):
                right = documents[right_index]
                if union.find(left) == union.find(right):
                    continue
                comparisons += 1
                if comparisons > pair_budget:
                    budget_exhausted = True
                    break
                if jaccard(left_tokens, token_sets[right]) >= threshold:
                    union.union(left, right)
            if budget_exhausted:
                break
        if budget_exhausted:
            break

    roots = np.array([union.find(index) for index in range(len(normalized))])
    _, cluster = np.unique(roots, return_inverse=True)

    if stats is not None:
        stats.update({"blocks": len(blocks), "comparisons": comparisons,
                      "pair_budget_exhausted": budget_exhausted, "threshold": threshold,
                      "max_df_for_key": max_df_for_key, "block_tokens": block_tokens})
    return pd.DataFrame({"norm": normalized_array, "sig": signature.to_numpy(),
                         "cluster": cluster})
