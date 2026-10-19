"""Features supervisadas in-fold de la cadena AGE (porte fiel del proyecto original).

Reproduce los 5 bloques supervisados con el orden EXACTO de columnas (73):
mention 17 + hashtag 17 + rt_source 17 + rarity 12 + lexicon 10.

- Perfil de clase por entidad: P(clase|entidad) con suavizado ``smoothing=20.0`` sobre el
  prior del train recibido; sin entidad conocida ``[prior, prior, 0, 0, 0, 0, 0]``
  (src/advanced_features.py:326-393, linea 377).
- Rareza lexica: frecuencias solo del train recibido; OOV con log10 centinela ``-6.0``
  (src/advanced_features.py:396-434, linea 419).
- Lexico por clase: log-odds de Monroe et al. 2008 con ``alpha0=100.0, min_count=5,
  top_k=3000`` (src/advanced_features.py:437-490).
- Regexes ``MENTION_RE``/``HASHTAG_RE``/``RT_SOURCE_RE``/``WORD_RE`` de
  src/advanced_features.py:28-32.

Lado OOF (LOFO estricto): scripts/advanced_02_features.py:205-234; cada fold ajusta los
5 bloques con el train de los otros 4 y transforma su validacion, reensamblando por
indice. Lado test: train completo como fuente
(scripts/phase25_submission_fresca.py:69-75).

ASCII puro. Sin efectos en import. Solo numpy/pandas/stdlib (+ cadena.comun).
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from cadena import comun

# --------------------------------------------------------------------------------------
# Patrones de entidades (src/advanced_features.py:28-32)
# --------------------------------------------------------------------------------------
#: En WORD_RE los caracteres acentuados van como escapes \uXXXX para mantener el fichero
#: en ASCII puro; el patron compilado es identico al literal del original.
MENTION_RE = re.compile(r"@(\w+)")
HASHTAG_RE = re.compile(r"#(\w+)")
RT_SOURCE_RE = re.compile(r"^\s*RT\s+@(\w+)", flags=re.IGNORECASE)
WORD_RE = re.compile("[A-Za-z\u00c1\u00c9\u00cd\u00d3\u00da\u00dc\u00d1\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f10-9]+")


def _safe_entropy(counts: np.ndarray) -> float:
    """Entropia de Shannon (base 2) segura para vectores vacios (src/advanced_features.py:56-62)."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def _entity_documents(texts: pd.Series, kind: str) -> list[list[str]]:
    """Lista de entidades por documento segun el tipo pedido (src/advanced_features.py:312-323)."""
    if kind == "mention":
        return [[m.lower() for m in MENTION_RE.findall(value)]
                for value in texts.fillna("").astype(str)]
    if kind == "hashtag":
        return [[h.lower() for h in HASHTAG_RE.findall(value)]
                for value in texts.fillna("").astype(str)]
    if kind == "rt_source":
        return [[m.group(1).lower()] if (m := RT_SOURCE_RE.match(value)) else []
                for value in texts.fillna("").astype(str)]
    raise ValueError(f"tipo de entidad desconocido: {kind}")


def supervised_entity_features(train_texts: pd.Series, y_train: np.ndarray,
                               target_texts: pd.Series, kind: str = "mention",
                               smoothing: float = 20.0) -> pd.DataFrame:
    """Perfil de clase supervisado de las entidades (src/advanced_features.py:326-393).

    Aprende P(clase|entidad) con suavizado empirico-bayesiano solo con el train recibido
    (por tanto es seguro dentro de un fold) y resume cada documento con el perfil medio,
    el perfil de su entidad mas informativa, el margen, la entropia y la cobertura. Sin
    entidad conocida devuelve ``[prior, prior, 0, 0, 0, 0, 0]`` (17 columnas).

    Parameters
    ----------
    train_texts : pandas.Series
        Textos de train del fold.
    y_train : numpy.ndarray
        Etiquetas de train del fold.
    target_texts : pandas.Series
        Textos a transformar (validacion o test).
    kind : {"mention", "hashtag", "rt_source"}
        Tipo de entidad.
    smoothing : float
        Peso del prior de clase (cuanto mayor, mas se acerca a la distribucion global).

    Returns
    -------
    pandas.DataFrame
        17 columnas: ``{kind}_mean_c0..c5``, ``{kind}_best_c0..c5``, ``{kind}_margin``,
        ``{kind}_entropy``, ``{kind}_known_count``, ``{kind}_unknown_ratio`` y
        ``{kind}_dominance``.
    """
    classes = comun.CLASSES
    class_index = {label: index for index, label in enumerate(classes)}
    train_entities = _entity_documents(train_texts, kind)
    target_entities = _entity_documents(target_texts, kind)

    prior = np.bincount([class_index[label] for label in y_train],
                        minlength=len(classes)).astype(float)
    prior /= prior.sum()

    counts: dict[str, np.ndarray] = {}
    for entities, label in zip(train_entities, y_train, strict=True):
        for entity in set(entities):
            counts.setdefault(entity, np.zeros(len(classes)))[class_index[label]] += 1

    profiles: dict[str, np.ndarray] = {}
    for entity, vector in counts.items():
        total = vector.sum()
        profiles[entity] = (vector + smoothing * prior) / (total + smoothing)

    rows: list[np.ndarray] = []
    for entities in target_entities:
        known = [profiles[entity] for entity in entities if entity in profiles]
        if not known:
            rows.append(np.concatenate([prior, prior, [0.0, 0.0, 0.0, 0.0, 0.0]]))
            continue
        stack = np.vstack(known)
        mean_profile = stack.mean(axis=0)
        best = stack[np.argmax(stack.max(axis=1))]
        ordered = np.sort(best)[::-1]
        margin = float(ordered[0] - ordered[1])
        entropy = _safe_entropy(best)
        rows.append(np.concatenate([mean_profile, best,
                                    [margin, entropy, float(len(stack)),
                                     float(1 - len(stack) / max(len(entities), 1)),
                                     float((np.array(entities) == Counter(entities).most_common(1)[0][0]).mean())]]))
    columns = ([f"{kind}_mean_c{i}" for i in range(len(classes))]
               + [f"{kind}_best_c{i}" for i in range(len(classes))]
               + [f"{kind}_margin", f"{kind}_entropy", f"{kind}_known_count",
                  f"{kind}_unknown_ratio", f"{kind}_dominance"])
    return pd.DataFrame(np.vstack(rows), columns=columns)


def lexical_rarity_features(train_texts: pd.Series, target_texts: pd.Series) -> pd.DataFrame:
    """Rareza y novedad lexica respecto del vocabulario del train recibido
    (src/advanced_features.py:396-434).

    Nunca usa estadisticas del conjunto objetivo (regla 74 del brief): las frecuencias se
    aprenden solo de ``train_texts``. Los OOV reciben el log10 centinela ``-6.0``
    (linea 419). Devuelve 12 columnas: ``rarity_mean_logfreq``, ``rarity_median_logfreq``,
    ``rarity_min_logfreq``, ``rarity_max_logfreq``, ``rarity_mean_freq``,
    ``rare_token_ratio``, ``very_rare_token_ratio``, ``oov_ratio``, ``hapax_token_ratio``,
    ``top_frequency_ratio``, ``lexical_novelty`` y ``lexical_familiarity``.
    """
    frequency: Counter = Counter()
    for tokens in train_texts.fillna("").str.lower().str.findall(WORD_RE.pattern):
        frequency.update(tokens)
    total = sum(frequency.values()) or 1
    log_frequencies = {token: np.log10(count / total) for token, count in frequency.items()}

    rows: list[dict] = []
    for tokens in target_texts.fillna("").str.lower().str.findall(WORD_RE.pattern):
        if not tokens:
            rows.append({"rarity_mean_logfreq": 0.0, "rarity_median_logfreq": 0.0,
                         "rarity_min_logfreq": 0.0, "rarity_max_logfreq": 0.0,
                         "rarity_mean_freq": 0.0, "rare_token_ratio": 0.0,
                         "very_rare_token_ratio": 0.0, "oov_ratio": 0.0,
                         "hapax_token_ratio": 0.0, "top_frequency_ratio": 0.0,
                         "lexical_novelty": 0.0, "lexical_familiarity": 0.0})
            continue
        frequencies = np.array([frequency.get(token, 0) for token in tokens], dtype=float)
        log_freqs = np.array([log_frequencies.get(token, -6.0) for token in tokens])
        rows.append({
            "rarity_mean_logfreq": float(log_freqs.mean()),
            "rarity_median_logfreq": float(np.median(log_freqs)),
            "rarity_min_logfreq": float(log_freqs.min()),
            "rarity_max_logfreq": float(log_freqs.max()),
            "rarity_mean_freq": float(frequencies.mean()),
            "rare_token_ratio": float((frequencies <= 5).mean()),
            "very_rare_token_ratio": float((frequencies <= 2).mean()),
            "oov_ratio": float((frequencies == 0).mean()),
            "hapax_token_ratio": float((frequencies == 1).mean()),
            "top_frequency_ratio": float(np.mean(log_freqs >= np.percentile(log_freqs, 90))),
            "lexical_novelty": float((frequencies <= 2).mean()),
            "lexical_familiarity": float(np.mean(log_freqs)),
        })
    return pd.DataFrame(rows)


def class_lexicon_scores(train_texts: pd.Series, y_train: np.ndarray,
                         target_texts: pd.Series, alpha0: float = 100.0,
                         min_count: int = 5, top_k: int = 3000) -> pd.DataFrame:
    """Distancia al lexico de cada clase mediante log-odds (Monroe et al. 2008)
    (src/advanced_features.py:437-490).

    Aprende, solo con el train recibido, los ``top_k`` terminos mas distintivos de cada
    clase (frecuencia minima ``min_count``, prior ``alpha0``) y resume cada documento con
    la puntuacion media de sus tokens frente a cada lexico, mas el margen, la entropia, la
    clase ganadora y la desviacion. Devuelve 10 columnas: ``lexicon_c0..c5``,
    ``lexicon_margin``, ``lexicon_entropy``, ``lexicon_best_class`` y ``lexicon_std``.
    """
    classes = comun.CLASSES
    class_index = {label: index for index, label in enumerate(classes)}
    train_tokens = train_texts.fillna("").str.lower().str.findall(WORD_RE.pattern)
    counter_global: Counter = Counter()
    counters_by_class = [Counter() for _ in classes]
    for tokens, label in zip(train_tokens, y_train, strict=True):
        position = class_index[label]
        counter_global.update(tokens)
        counters_by_class[position].update(tokens)

    total = sum(counter_global.values()) or 1
    prior = {token: alpha0 * count / total for token, count in counter_global.items()}
    class_scores: list[dict[str, float]] = []
    totals_by_class = [sum(counter.values()) for counter in counters_by_class]
    for position in range(len(classes)):
        scores: dict[str, float] = {}
        other_total = total - totals_by_class[position]
        for token, count in counter_global.items():
            if count < min_count:
                continue
            in_class = counters_by_class[position].get(token, 0)
            out_class = count - in_class
            if in_class == 0:
                continue
            log_odds = (np.log((in_class + prior[token])
                               / (totals_by_class[position] + alpha0 - in_class - prior[token]))
                        - np.log((out_class + prior[token])
                                 / (other_total + alpha0 - out_class - prior[token])))
            scores[token] = float(log_odds)
        top_items = sorted(scores.items(), key=lambda item: -item[1])[:top_k]
        class_scores.append({token: value for token, value in top_items})

    rows: list[dict] = []
    for tokens in target_texts.fillna("").str.lower().str.findall(WORD_RE.pattern):
        vector = np.zeros(len(classes))
        for position, scores in enumerate(class_scores):
            if tokens:
                vector[position] = float(np.mean([scores.get(token, 0.0) for token in tokens]))
        ordered = np.sort(vector)[::-1]
        rows.append({**{f"lexicon_c{i}": vector[i] for i in range(len(classes))},
                     "lexicon_margin": float(ordered[0] - ordered[1]),
                     "lexicon_entropy": _safe_entropy(vector - vector.min() + 1e-9),
                     "lexicon_best_class": float(np.argmax(vector)),
                     "lexicon_std": float(vector.std())})
    return pd.DataFrame(rows)


def supervisadas_infold(textos: pd.Series, y: np.ndarray, fold_idx: np.ndarray) -> np.ndarray:
    """Features supervisadas LOFO estricto, (n, 73) float64.

    Porte de scripts/advanced_02_features.py:205-234. Para cada fold se ajustan los 5
    bloques con el train de los otros 4 folds y se transforma la validacion; el resultado
    se reensambla por indice (posicional) en el orden original. Orden de columnas:
    mention 17 + hashtag 17 + rt_source 17 + rarity 12 + lexicon 10.

    Parameters
    ----------
    textos : pandas.Series
        Textos del train completo (columna de texto, ya limpia).
    y : numpy.ndarray
        Etiquetas del train completo (strings de CLASSES).
    fold_idx : numpy.ndarray
        Protocolo de folds congelado: valor 0-4 por fila.

    Returns
    -------
    numpy.ndarray
        Matriz (n, 73) float64 en el orden de filas de ``textos``.
    """
    partes: list[pd.DataFrame] = []
    for fold in sorted(np.unique(fold_idx)):
        train_idx = np.flatnonzero(fold_idx != fold)
        valid_idx = np.flatnonzero(fold_idx == fold)
        train_texts = textos.iloc[train_idx]
        valid_texts = textos.iloc[valid_idx]
        y_train = y[train_idx]
        # Mismo orden que el dict `blocks` del original: mencion, hashtag, fuente_rt,
        # rareza, lexico (scripts/advanced_02_features.py:215-221).
        block = pd.concat([
            supervised_entity_features(train_texts, y_train, valid_texts, "mention"),
            supervised_entity_features(train_texts, y_train, valid_texts, "hashtag"),
            supervised_entity_features(train_texts, y_train, valid_texts, "rt_source"),
            lexical_rarity_features(train_texts, valid_texts),
            class_lexicon_scores(train_texts, y_train, valid_texts),
        ], axis=1)
        block.index = valid_idx
        partes.append(block)
    matriz = pd.concat(partes).sort_index().to_numpy(float)
    assert matriz.shape[1] == 73, matriz.shape
    return matriz


def supervisadas_test(textos_train: pd.Series, y: np.ndarray,
                      textos_test: pd.Series) -> np.ndarray:
    """Features supervisadas con el train completo como fuente, (m, 73) float64.

    Porte de scripts/phase25_submission_fresca.py:69-75: misma funcion que in-fold pero
    ajustando los 5 bloques con todo el train y transformando el test. Orden de columnas:
    mention 17 + hashtag 17 + rt_source 17 + rarity 12 + lexicon 10.

    Parameters
    ----------
    textos_train : pandas.Series
        Textos del train completo.
    y : numpy.ndarray
        Etiquetas del train completo (strings de CLASSES).
    textos_test : pandas.Series
        Textos del test a transformar.

    Returns
    -------
    numpy.ndarray
        Matriz (m, 73) float64 en el orden de filas de ``textos_test``.
    """
    matriz = pd.concat([
        supervised_entity_features(textos_train, y, textos_test, "mention"),
        supervised_entity_features(textos_train, y, textos_test, "hashtag"),
        supervised_entity_features(textos_train, y, textos_test, "rt_source"),
        lexical_rarity_features(textos_train, textos_test),
        class_lexicon_scores(textos_train, y, textos_test),
    ], axis=1).to_numpy(float)
    assert matriz.shape[1] == 73, matriz.shape
    return matriz
