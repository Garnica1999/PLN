"""Features de segunda generación (fase de investigación avanzada).

Implementa las familias del brief con dos garantías metodológicas:

1. **Features locales/deterministas** (distribuciones, forma, posición, entropía, compresión):
   se calculan una vez por documento y no dependen de etiquetas ni del corpus de referencia.
2. **Features supervisadas** (perfil de clase de menciones, hashtags, cuenta fuente de RT, léxico
   por clase, rareza de corpus): se calculan **siempre dentro del fold**, aprendiendo las
   estadísticas solo con el train del fold (regla §74 del brief: nunca `fit` en todo train y
   `transform` en validación).

Cada función es determinista, testeable e independiente (familia correspondiente del brief).
"""

from __future__ import annotations

import re
import zlib
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

from src import config as cfg

# --- Patrones de entidades -------------------------------------------------------------
MENTION_RE = re.compile(r"@(\w+)")
HASHTAG_RE = re.compile(r"#(\w+)")
RT_SOURCE_RE = re.compile(r"^\s*RT\s+@(\w+)", flags=re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+")
WORD_RE = re.compile(r"[a-záéíóúüñ0-9]+")
TOKEN_RE = re.compile(r"@\w+|#\w+|https?://\S+|[a-záéíóúüñ]+|\d+|[^\s\w]", flags=re.UNICODE)

#: Familias del brief cubiertas por este módulo.
FAMILIES = {
    "A_distribucion_longitud_palabra": "percentiles, asimetría, curtosis, CV, IQR, MAD",
    "B_distribucion_caracteres": "ratios por tipo y concentración (Simpson, HHI)",
    "C_longitud_palabra_colas": "proporciones por rango de longitud",
    "D_riqueza_lexica": "MATTR, Yule's K, Simpson, Maas, root TTR",
    "H_word_shape": "ALLCAPS, capitalización, alfanuméricos, repetición",
    "I_posicion": "primera/última posición normalizada de cada entidad",
    "M_urls": "conteo, densidad, longitud, posición",
    "N_estructura": "puntuación por tipo, transiciones, diversidad",
    "R_entropia": "entropía de caracteres, tokens, puntuación y entidades",
    "S_compresibilidad": "ratio de compresión zlib",
    "T_repeticion": "repetición de palabras, bigramas y runs",
    "U_posicion_lexica": "primera/última palabra (longitud, frecuencia, tipo)",
}


def _safe_entropy(counts: np.ndarray) -> float:
    """Entropía de Shannon (base 2) segura para vectores vacíos."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def _percentiles(values: np.ndarray, points: tuple[int, ...] = (10, 25, 50, 75, 90)) -> dict:
    """Percentiles de un vector (0 si está vacío)."""
    if values.size == 0:
        return {f"p{p}": 0.0 for p in points}
    return {f"p{p}": float(np.percentile(values, p)) for p in points}


# --------------------------------------------------------------------------------------
# Familias A, C — distribución de la longitud de palabra
# --------------------------------------------------------------------------------------
def distribution_features(texts: pd.Series) -> pd.DataFrame:
    """Familia A/C: distribución interna de la longitud de palabra y de los tokens."""
    tokens = texts.fillna("").str.lower().str.findall(WORD_RE.pattern)
    rows: list[dict] = []
    for token_list in tokens:
        lengths = np.array([len(token) for token in token_list], dtype=float)
        if lengths.size == 0:
            rows.append({"wlen_mean": 0.0, "wlen_median": 0.0, "wlen_std": 0.0, "wlen_cv": 0.0,
                         "wlen_skew": 0.0, "wlen_kurtosis": 0.0, "wlen_iqr": 0.0, "wlen_mad": 0.0,
                         "wlen_range": 0.0, "wlen_p10": 0.0, "wlen_p25": 0.0, "wlen_p50": 0.0,
                         "wlen_p75": 0.0, "wlen_p90": 0.0, "wlen_tail_ratio": 0.0,
                         "wlen_head_ratio": 0.0, "prop_len1": 0.0, "prop_len2": 0.0,
                         "prop_len3": 0.0, "prop_len_ge8": 0.0, "prop_len_ge10": 0.0})
            continue
        percentiles = {f"wlen_{key}": value for key, value in _percentiles(lengths).items()}
        mean = float(lengths.mean())
        std = float(lengths.std())
        has_variance = lengths.std() > 0
        rows.append({
            "wlen_mean": mean, "wlen_median": float(np.median(lengths)), "wlen_std": std,
            "wlen_cv": std / mean if mean > 0 else 0.0,
            # Asimetría y curtosis requieren varianza no nula (si todos los tokens miden igual,
            # la forma no está definida y se reporta 0 en lugar de NaN).
            "wlen_skew": float(stats.skew(lengths)) if lengths.size > 2 and has_variance else 0.0,
            "wlen_kurtosis": (float(stats.kurtosis(lengths))
                              if lengths.size > 3 and has_variance else 0.0),
            "wlen_iqr": float(np.percentile(lengths, 75) - np.percentile(lengths, 25)),
            "wlen_mad": float(np.mean(np.abs(lengths - mean))),
            "wlen_range": float(lengths.max() - lengths.min()),
            **percentiles,
            "wlen_tail_ratio": float((lengths >= np.percentile(lengths, 90)).mean()),
            "wlen_head_ratio": float((lengths <= np.percentile(lengths, 10)).mean()),
            "prop_len1": float((lengths == 1).mean()), "prop_len2": float((lengths == 2).mean()),
            "prop_len3": float((lengths == 3).mean()), "prop_len_ge8": float((lengths >= 8).mean()),
            "prop_len_ge10": float((lengths >= 10).mean()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Familia B, R — distribución de caracteres y entropías
# --------------------------------------------------------------------------------------
def char_distribution_features(texts: pd.Series) -> pd.DataFrame:
    """Familia B/R: composición y concentración del alfabeto usado en el tweet."""
    clean = texts.fillna("").astype(str)
    lowered = clean.str.lower()
    total = clean.str.len().replace(0, np.nan)
    vowels = lowered.str.count(r"[aeiouáéíóúü]")
    consonants = lowered.str.count(r"[bcdfghjklmnpqrstvwxyzñ]")
    letters = vowels + consonants
    rows = pd.DataFrame({
        "char_vowel_ratio": (vowels / letters.replace(0, np.nan)).fillna(0.0),
        "char_consonant_ratio": (consonants / letters.replace(0, np.nan)).fillna(0.0),
        "char_alpha_ratio": (letters / total).fillna(0.0),
        "char_space_ratio": (clean.str.count(r"\s") / total).fillna(0.0),
        "char_symbol_ratio": (clean.str.count(r"[^\w\s]") / total).fillna(0.0),
    })
    character_entropy, simpson, hhi, dominant, unique_ratio = [], [], [], [], []
    for value in clean:
        if not value:
            character_entropy.append(0.0)
            simpson.append(0.0)
            hhi.append(0.0)
            dominant.append(0.0)
            unique_ratio.append(0.0)
            continue
        counts = np.array(list(Counter(value).values()), dtype=float)
        probabilities = counts / counts.sum()
        character_entropy.append(_safe_entropy(counts))
        simpson.append(float(1 - (probabilities ** 2).sum()))
        hhi.append(float((probabilities ** 2).sum()))
        dominant.append(float(probabilities.max()))
        unique_ratio.append(len(set(value)) / len(value))
    rows["char_entropy"] = character_entropy
    rows["char_simpson"] = simpson
    rows["char_hhi"] = hhi
    rows["char_dominant_ratio"] = dominant
    rows["char_unique_ratio"] = unique_ratio
    return rows


# --------------------------------------------------------------------------------------
# Familia D — riqueza léxica avanzada
# --------------------------------------------------------------------------------------
def lexical_richness_features(texts: pd.Series) -> pd.DataFrame:
    """Familia D: diversidad léxica estable para textos cortos (MATTR, Yule's K, Simpson)."""
    tokens = texts.fillna("").str.lower().str.findall(WORD_RE.pattern)
    rows: list[dict] = []
    for token_list in tokens:
        n = len(token_list)
        if n == 0:
            rows.append({"mattr_25": 0.0, "root_ttr": 0.0, "log_ttr": 0.0, "maas": 0.0,
                         "yule_k": 0.0, "simpson_d": 0.0, "dis_legomena_ratio": 0.0,
                         "repeated_token_ratio": 0.0, "top_token_concentration": 0.0})
            continue
        counts = Counter(token_list)
        window = min(25, n)
        windows = [token_list[start:start + window] for start in range(0, max(1, n - window + 1))]
        mattr = float(np.mean([len(set(w)) / len(w) for w in windows if w]))
        types = len(counts)
        frequencies = np.array(list(counts.values()), dtype=float)
        yule_k = float(1e4 * ((frequencies ** 2).sum() - n) / (n ** 2)) if n > 1 else 0.0
        rows.append({
            "mattr_25": mattr,
            "root_ttr": types / np.sqrt(n),
            "log_ttr": np.log(types) / np.log(n) if n > 1 and types > 1 else 0.0,
            "maas": (np.log(n) - np.log(types)) / (np.log(n) ** 2) if n > 1 and types > 1 else 0.0,
            "yule_k": yule_k,
            "simpson_d": float((frequencies * (frequencies - 1)).sum() / (n * (n - 1)))
            if n > 1 else 0.0,
            "dis_legomena_ratio": float((frequencies == 2).sum() / types) if types else 0.0,
            "repeated_token_ratio": float((n - types) / n),
            "top_token_concentration": float(frequencies.max() / n),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Familia H, T — forma de palabra y repetición
# --------------------------------------------------------------------------------------
def word_shape_features(texts: pd.Series) -> pd.DataFrame:
    """Familia H/T: patrones de escritura de las palabras y repetición interna."""
    tokens = texts.fillna("").str.findall(TOKEN_RE.pattern)
    rows: list[dict] = []
    for token_list in tokens:
        words = [token for token in token_list if WORD_RE.fullmatch(token)]
        n = len(words) or 1
        all_caps = sum(1 for token in words if token.isupper() and len(token) > 1)
        capitalized = sum(1 for token in words if token[:1].isupper() and token[1:].islower())
        numeric = sum(1 for token in words if token.isdigit())
        alphanumeric = sum(1 for token in words if any(c.isdigit() for c in token)
                           and any(c.isalpha() for c in token))
        repeated = sum(1 for token in words if re.search(r"(\w)\1{1,}", token))
        bigrams = list(zip(token_list, token_list[1:])) if len(token_list) > 1 else []
        trigrams = (list(zip(token_list, token_list[1:], token_list[2:]))
                    if len(token_list) > 2 else [])
        rows.append({
            "shape_allcaps_ratio": all_caps / n,
            "shape_capitalized_ratio": capitalized / n,
            "shape_lower_ratio": sum(1 for token in words if token.islower()) / n,
            "shape_numeric_ratio": numeric / n,
            "shape_alphanumeric_ratio": alphanumeric / n,
            "shape_repeated_letter_token_ratio": repeated / n,
            "repeat_token_ratio": float(1 - len(set(token_list)) / len(token_list))
            if token_list else 0.0,
            "repeat_bigram_ratio": float(1 - len(set(bigrams)) / len(bigrams)) if bigrams else 0.0,
            "repeat_trigram_ratio": float(1 - len(set(trigrams)) / len(trigrams)) if trigrams else 0.0,
            "adjacent_repeat_ratio": float(np.mean([a == b for a, b in bigrams])) if bigrams else 0.0,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Familia I, M, U — posición de elementos, URLs y posición léxica
# --------------------------------------------------------------------------------------
def position_features(texts: pd.Series) -> pd.DataFrame:
    """Familias I/M/U: posición normalizada de entidades y de la primera/última palabra."""
    clean = texts.fillna("").astype(str)
    rows: list[dict] = []
    for value in clean:
        length = max(len(value), 1)
        tokens = TOKEN_RE.findall(value)

        def positions(pattern: re.Pattern) -> tuple[float, float, float]:
            matches = [m.start() / length for m in pattern.finditer(value)]
            if not matches:
                return 0.0, 0.0, 0.0
            return matches[0], matches[-1], float(np.mean(matches))

        mention_first, mention_last, mention_mean = positions(MENTION_RE)
        hashtag_first, hashtag_last, hashtag_mean = positions(HASHTAG_RE)
        url_first, url_last, url_mean = positions(URL_RE)
        first_word = tokens[0] if tokens else ""
        last_word = tokens[-1] if tokens else ""
        rows.append({
            "pos_mention_first": mention_first, "pos_mention_last": mention_last,
            "pos_mention_mean": mention_mean, "pos_hashtag_first": hashtag_first,
            "pos_hashtag_last": hashtag_last, "pos_hashtag_mean": hashtag_mean,
            "pos_url_first": url_first, "pos_url_last": url_last, "pos_url_mean": url_mean,
            "first_token_is_mention": int(bool(tokens) and tokens[0].startswith("@")),
            "last_token_is_mention": int(bool(tokens) and tokens[-1].startswith("@")),
            "first_token_is_hashtag": int(bool(tokens) and tokens[0].startswith("#")),
            "last_token_is_hashtag": int(bool(tokens) and tokens[-1].startswith("#")),
            "first_token_is_url": int(bool(tokens) and tokens[0].startswith("http")),
            "last_token_is_url": int(bool(tokens) and tokens[-1].startswith("http")),
            "first_word_len": len(first_word), "last_word_len": len(last_word),
            "first_word_is_word": int(bool(WORD_RE.fullmatch(first_word))),
            "last_word_is_word": int(bool(WORD_RE.fullmatch(last_word))),
            "first_word_is_upper": int(first_word[:1].isupper() and len(first_word) > 1),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Familia N, R, S — estructura, entropía y compresibilidad
# --------------------------------------------------------------------------------------
def structure_entropy_features(texts: pd.Series) -> pd.DataFrame:
    """Familias N/R/S: puntuación, entropías por tipo y ratio de compresión."""
    clean = texts.fillna("").astype(str)
    total = clean.str.len().replace(0, np.nan)
    punctuation = {name: clean.str.count(pattern) for name, pattern in
                   (("comma", ","), ("period", r"\."), ("colon", ":"), ("semicolon", ";"),
                    ("parenthesis", r"[()]"), ("quote", r"[\"']"), ("dash", r"[-–—]"),
                    ("excl", "!"), ("quest", r"\?"))}
    rows = pd.DataFrame({
        f"punct_{name}_density": (count / total).fillna(0.0)
        for name, count in punctuation.items()
    })
    rows["punct_type_diversity"] = clean.apply(
        lambda value: len(set(re.findall(r"[^\w\s]", value))))
    rows["punct_transitions"] = clean.apply(
        lambda value: len(re.findall(r"[^\w\s]{2,}", value)))
    rows["token_entropy"] = clean.apply(
        lambda value: _safe_entropy(np.array(list(Counter(TOKEN_RE.findall(value)).values()),
                                             dtype=float)) if value else 0.0)
    rows["punct_entropy"] = clean.apply(
        lambda value: _safe_entropy(np.array(list(Counter(re.findall(r"[^\w\s]", value)).values()),
                                             dtype=float)) if value else 0.0)
    rows["entity_count"] = clean.apply(lambda value: len(MENTION_RE.findall(value))
                                       + len(HASHTAG_RE.findall(value)))
    rows["entity_entropy"] = clean.apply(
        lambda value: _safe_entropy(np.array(list(Counter(
            [m.lower() for m in MENTION_RE.findall(value)]
            + [h.lower() for h in HASHTAG_RE.findall(value)]).values()), dtype=float))
        if value else 0.0)
    rows["compression_ratio"] = clean.apply(
        lambda value: len(zlib.compress(value.encode("utf-8"), 6)) / max(len(value), 1))
    rows["compression_gain"] = clean.apply(
        lambda value: 1 - len(zlib.compress(value.encode("utf-8"), 6)) / max(len(value), 1))
    return rows


# --------------------------------------------------------------------------------------
# Features supervisadas (in-fold)
# --------------------------------------------------------------------------------------
def _entity_documents(texts: pd.Series, kind: str) -> list[list[str]]:
    """Extrae la lista de entidades por documento según el tipo pedido."""
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
    """Perfil de clase supervisado de las entidades (familias J, K, L).

    Aprende ``P(clase | entidad)`` con suavizado empírico-bayesiano **solo con el train recibido**
    (por tanto es seguro dentro de un fold) y resume cada documento con el perfil medio, el
    perfil de su entidad más informativa, el margen, la entropía y la cobertura.

    Parameters
    ----------
    train_texts : pandas.Series
        Textos de train del fold.
    y_train : numpy.ndarray
        Etiquetas de train del fold.
    target_texts : pandas.Series
        Textos a transformar (validación o test).
    kind : {"mention", "hashtag", "rt_source"}
        Tipo de entidad.
    smoothing : float
        Peso del prior de clase (cuanto mayor, más se acerca a la distribución global).

    Returns
    -------
    pandas.DataFrame
        Columnas ``{kind}_profile_c`` para cada clase, más margen, entropía, coberturas y
        concentración de la entidad dominante.
    """
    classes = cfg.CLASSES
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
    """Familia E/Y: rareza y novedad léxica respecto del vocabulario **del train recibido**.

    Nunca usa estadísticas del conjunto objetivo (regla §74): las frecuencias se aprenden solo
    de ``train_texts``.
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
    """Familias F/X: distancia al léxico de cada clase mediante log-odds (Monroe et al. 2008).

    Aprende, **solo con el train recibido**, los términos más distintivos de cada clase y resume
    cada documento con la puntuación de sus tokens frente a cada léxico de clase, más el margen,
    la entropía y la clase ganadora.
    """
    classes = cfg.CLASSES
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
