"""Clústeres de near-duplicados y esquemas de validación cruzada (decisión D-003).

El dataset no tiene identificador de autor, por lo que no es posible agrupar por usuario.
El riesgo real de generalización es el **duplicado / near-duplicado**: retuits y variantes
del mismo texto (que solo cambian URLs o menciones) aparecen en train y en test.

Este módulo construye clústeres de near-duplicados sobre el **conjunto completo** de textos
(train + test, sin usar etiquetas) y genera particiones:

- ``scheme="group"``: ``StratifiedGroupKFold`` agrupando por clúster (validación primaria).
- ``scheme="stratified"``: ``StratifiedKFold`` ingenuo (control para cuantificar el leakage
  por duplicados como diferencia entre ambos esquemas).

Método de agrupamiento (corregido el 2026-09-22): igualdad de texto normalizado **o**
similitud de Jaccard ≥ ``jaccard_threshold`` entre conjuntos de tokens, con *blocking* por
los tokens más raros para evitar el costo cuadrático. Una primera versión usó la firma de
los primeros 60 caracteres y produjo clústeres espurios (tweets distintos que comparten una
apertura genérica quedaban unidos), lo que sesgaba a la baja la pureza de etiqueta; el
método actual solo une textos realmente similares.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src import config as cfg

#: Longitud mínima del texto normalizado para participar en el agrupamiento.
MIN_LEN = 20
#: Umbral de Jaccard sobre conjuntos de tokens para considerar near-duplicado.
JACCARD_THRESHOLD = 0.8
#: Número de tokens raros usados como clave de bloqueo.
BLOCK_TOKENS = 3
#: Frecuencia documental máxima para que un token sirva como clave de bloqueo.
MAX_DF_FOR_KEY = 200
#: Tope de documentos comparados dentro de un mismo bloque (evita costo cuadrático).
MAX_BLOCK_SIZE = 400

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_RT_RE = re.compile(r"^\s*rt\s*:?\s*", flags=re.IGNORECASE)
_NOISE_RE = re.compile(r"[^0-9a-záéíóúüñ ]+")
_SPACES_RE = re.compile(r"\s+")


def normalize_for_dedup(series: pd.Series) -> pd.Series:
    """Normaliza el texto para detectar duplicados y near-duplicados.

    Elimina URLs, menciones y el prefijo ``RT``; pasa a minúsculas; descarta puntuación y
    colapsa espacios. El objetivo es que dos variantes del mismo contenido (que solo
    difieren en enlaces o menciones) colapsen al mismo texto normalizado.

    Parameters
    ----------
    series : pandas.Series
        Textos crudos.

    Returns
    -------
    pandas.Series
        Textos normalizados.
    """
    text = series.fillna("").astype(str).str.lower()
    text = text.str.replace(_URL_RE, " ", regex=True)
    text = text.str.replace(_MENTION_RE, " ", regex=True)
    text = text.str.replace(_RT_RE, "", regex=True)
    text = text.str.replace(_NOISE_RE, " ", regex=True)
    return text.str.replace(_SPACES_RE, " ", regex=True).str.strip()


def dedup_signature(normalized: pd.Series, chars: int = 60) -> pd.Series:
    """Firma informativa de prefijo (solo se guarda como metadato, no agrupa)."""
    return normalized.str.slice(0, chars)


def jaccard(left: set[str], right: set[str]) -> float:
    """Similitud de Jaccard entre dos conjuntos de tokens."""
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    # Cota superior: si ni siquiera el conjunto pequeño cabe en la unión mínima, se descarta.
    if len(left) / len(right) < JACCARD_THRESHOLD and len(left) < JACCARD_THRESHOLD * len(right):
        return 0.0
    intersection = len(left & right)
    if intersection == 0:
        return 0.0
    return intersection / (len(left) + len(right) - intersection)


class _UnionFind:
    """Estructura union-find con compresión de caminos."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        """Encuentra la raíz del conjunto de ``item`` con compresión de caminos."""
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


def build_dup_clusters(texts: pd.Series, min_len: int = MIN_LEN,
                       threshold: float = JACCARD_THRESHOLD,
                       block_tokens: int = BLOCK_TOKENS,
                       max_df_for_key: int = MAX_DF_FOR_KEY,
                       max_block_size: int = MAX_BLOCK_SIZE,
                       pair_budget: int = 20_000_000,
                       stats: dict | None = None) -> pd.DataFrame:
    """Agrupa textos en clústeres de duplicado exacto o near-duplicado.

    Dos filas caen en el mismo clúster si su texto normalizado es idéntico **o** si la
    similitud de Jaccard entre sus conjuntos de tokens es ≥ ``threshold``. Para evitar el
    costo cuadrático se usa *blocking*: cada documento se indexa por sus ``block_tokens``
    tokens más raros (frecuencia documental ≤ ``max_df_for_key``) y solo se comparan pares
    dentro del mismo bloque.

    Parameters
    ----------
    texts : pandas.Series
        Textos (típicamente ``concat([train.text, test.text])``).
    min_len : int
        Longitud mínima del texto normalizado para agrupar (evita unir textos vacíos).
    threshold : float
        Umbral de Jaccard para considerar near-duplicado.
    block_tokens : int
        Número de tokens raros usados como clave de bloqueo por documento.
    max_df_for_key : int
        Frecuencia documental máxima para que un token sirva como clave.
    max_block_size : int
        Tope de documentos comparados por bloque.
    pair_budget : int
        Presupuesto máximo de comparaciones de pares (corta la búsqueda si se agota).
    stats : dict, optional
        Si se pasa, se rellena con diagnósticos del proceso (bloques, comparaciones).

    Returns
    -------
    pandas.DataFrame
        Columnas ``norm`` (texto normalizado), ``sig`` (firma informativa) y ``cluster``.
    """
    normalized = normalize_for_dedup(texts)
    signature = dedup_signature(normalized)
    normalized_array = normalized.to_numpy()
    valid = normalized.str.len().to_numpy() >= min_len
    union = _UnionFind(len(normalized))

    # 1) Duplicado exacto sobre el texto normalizado.
    exact_groups: dict[str, list[int]] = defaultdict(list)
    for position, key in enumerate(normalized_array):
        if valid[position]:
            exact_groups[key].append(position)
    for positions in exact_groups.values():
        anchor = positions[0]
        for position in positions[1:]:
            union.union(anchor, position)

    # 2) Near-duplicados por Jaccard con blocking por tokens raros.
    token_sets = [frozenset(tokens.split()) for tokens in normalized_array]
    document_frequency: Counter = Counter()
    for tokens in token_sets:
        document_frequency.update(tokens)

    blocks: dict[str, list[int]] = defaultdict(list)
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
    # Renumeración compacta y determinista de los clústeres.
    _, cluster = np.unique(roots, return_inverse=True)

    if stats is not None:
        stats.update({
            "blocks": len(blocks),
            "comparisons": comparisons,
            "pair_budget_exhausted": budget_exhausted,
            "threshold": threshold,
            "max_df_for_key": max_df_for_key,
            "block_tokens": block_tokens,
        })
    return pd.DataFrame({"norm": normalized_array, "sig": signature.to_numpy(),
                         "cluster": cluster})


def cluster_leakage_report(clusters: pd.DataFrame, n_train: int,
                           y_train: pd.Series | None = None) -> dict:
    """Cuantifica la exposición a leakage por duplicados entre train y test.

    Parameters
    ----------
    clusters : pandas.DataFrame
        Salida de :func:`build_dup_clusters` sobre ``train + test`` concatenados.
    n_train : int
        Número de filas que pertenecen a train (las primeras del concatenado).
    y_train : pandas.Series, optional
        Etiquetas de train para medir pureza de los clústeres.

    Returns
    -------
    dict
        Métricas de cobertura (test con clúster visto en train), pureza (etiqueta única) y
        número de clústeres que cruzan train/test.
    """
    cluster = clusters["cluster"].to_numpy()
    train_cluster = cluster[:n_train]
    test_cluster = cluster[n_train:]

    train_lookup: dict[int, Counter] = defaultdict(Counter)
    if y_train is not None:
        labels = np.asarray(y_train.astype(str))
        for cluster_id, label in zip(train_cluster, labels, strict=True):
            train_lookup[int(cluster_id)][label] += 1

    seen = set(train_cluster.tolist())
    covered = np.isin(test_cluster, list(seen))
    report = {
        "n_clusters_total": int(cluster.max()) + 1,
        "n_clusters_train": int(len(np.unique(train_cluster))),
        "n_clusters_test": int(len(np.unique(test_cluster))),
        "n_clusters_crossing": int(len(seen.intersection(set(test_cluster.tolist())))),
        "test_rows_covered": int(covered.sum()),
        "test_coverage_pct": round(float(covered.mean() * 100), 4),
    }
    if y_train is not None:
        pure = 0
        for cluster_id in np.unique(test_cluster[covered]):
            if len(train_lookup[int(cluster_id)]) == 1:
                pure += 1
        report["covered_clusters_pure"] = int(pure)
        report["covered_clusters"] = int(len(np.unique(test_cluster[covered])))
        report["purity_pct"] = round(100 * pure / max(1, report["covered_clusters"]), 3)
    return report


def duplicate_memorization_baseline(cluster: np.ndarray, y: np.ndarray, folds: list[dict],
                                    classes: list[str] | None = None) -> dict:
    """Mide cuánto puntaje se puede obtener "de gratis" memorizando duplicados.

    Para cada fold predice, en las filas de validación cuyo clúster tiene miembros en el
    train del fold, la etiqueta mayoritaria observada en ese clúster; las filas no
    cubiertas reciben un centinela que nunca acierta. El macro-F1 resultante es una cota
    de cuánto del desempeño proviene de memorizar duplicados y no de generalizar estilo.

    Returns
    -------
    dict
        ``coverage_pct``, ``accuracy_covered``, ``macro_f1_memorization_only`` (media y
        desviación entre folds).
    """
    from sklearn.metrics import f1_score

    classes = classes or cfg.CLASSES
    sentinel = "__no_cubierto__"
    covered_total = 0
    correct_total = 0
    rows_total = 0
    scores: list[float] = []

    for fold in folds:
        train_idx, val_idx = fold["train_idx"], fold["val_idx"]
        lookup: dict[int, Counter] = defaultdict(Counter)
        for index in train_idx:
            lookup[int(cluster[index])][str(y[index])] += 1
        reference = np.asarray([str(value) for value in np.asarray(y)[val_idx]])
        predicted = np.array([
            lookup[int(cluster[index])].most_common(1)[0][0] if lookup.get(int(cluster[index]))
            else sentinel
            for index in val_idx
        ])
        mask = predicted != sentinel
        covered_total += int(mask.sum())
        correct_total += int((predicted[mask] == reference[mask]).sum())
        rows_total += len(val_idx)
        scores.append(float(f1_score(reference, predicted, average="macro", labels=classes,
                                     zero_division=0)))

    return {
        "coverage_pct": round(100 * covered_total / max(1, rows_total), 4),
        "accuracy_covered": round(correct_total / max(1, covered_total), 4),
        "macro_f1_memorization_only_mean": round(float(np.mean(scores)), 4),
        "macro_f1_memorization_only_std": round(float(np.std(scores)), 4),
    }


def make_folds(y: pd.Series | np.ndarray, groups: np.ndarray | None = None,
               n_splits: int = cfg.N_SPLITS, n_repeats: int = cfg.N_REPEATS,
               seed: int = cfg.SEED, scheme: str = "group") -> list[dict]:
    """Genera particiones de validación cruzada repetidas.

    Parameters
    ----------
    y : array-like
        Etiquetas (se usan como variable de estratificación).
    groups : numpy.ndarray, optional
        Identificador de clúster por fila. Obligatorio si ``scheme="group"``.
    n_splits : int
        Número de folds por repetición.
    n_repeats : int
        Número de repeticiones (semillas distintas).
    seed : int
        Semilla base.
    scheme : {"group", "stratified"}
        Esquema de partición: agrupado por clúster o estratificado ingenuo.

    Returns
    -------
    list of dict
        Cada elemento contiene ``repeat``, ``fold``, ``train_idx`` y ``val_idx``.
    """
    y_array = np.asarray(y)
    folds: list[dict] = []
    for repeat in range(n_repeats):
        random_state = seed + repeat
        if scheme == "group":
            if groups is None:
                raise ValueError("scheme='group' requiere el argumento groups")
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                            random_state=random_state)
            iterator = splitter.split(np.zeros(len(y_array)), y_array, groups=np.asarray(groups))
        elif scheme == "stratified":
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            iterator = splitter.split(np.zeros(len(y_array)), y_array)
        else:
            raise ValueError(f"scheme desconocido: {scheme}")
        for fold, (train_idx, val_idx) in enumerate(iterator):
            folds.append({"repeat": repeat, "fold": fold, "train_idx": train_idx,
                          "val_idx": val_idx, "scheme": scheme})
    return folds
