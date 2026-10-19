"""Features estructuradas de segunda generacion (87 columnas) - porte literal.

Porte fiel de las seis familias locales de ``src/advanced_features.py``, en el orden de
ensamblado de ``scripts/advanced_02_features.py:38-45,94-104`` (``LOCAL_BUILDERS`` /
``build_local_features``) y de ``scripts/phase3b_recorregido.py:54-69`` (``build_local``):

    1. ``distribution_features``       (src/advanced_features.py:75-111)  -> 21 cols wlen_*/prop_*
    2. ``char_distribution_features``  (src/advanced_features.py:117-153) -> 10 cols char_*
    3. ``lexical_richness_features``   (src/advanced_features.py:159-189) -> 9 cols de riqueza lexica
    4. ``word_shape_features``         (src/advanced_features.py:195-226) -> 10 cols shape_*/repeat_*
    5. ``position_features``           (src/advanced_features.py:232-267) -> 20 cols pos_*/first_*/last_*
    6. ``structure_entropy_features``  (src/advanced_features.py:273-306) -> 17 cols punct_*/entropia

mas la correccion vigente de ``scripts/phase3b_recorregido.py:42-51`` (``ALNUM_RE`` y recalculo
de ``shape_alphanumeric_ratio``), que es la version que produjo el artefacto
``features/age_structured_features.parquet``.

Orden y conteo verificados contra ese parquet: 21 + 10 + 9 + 10 + 20 + 17 = 87 columnas
(73 float64 + 14 int64). Nota: el plan cita 19/18 para las familias 5/6, pero el codigo literal
y el artefacto confirman 20/17 con el mismo total de 87.

Entrada: DataFrame con columna ``text`` limpia (se procesa en orden de fila). Salida: DataFrame
de 87 columnas con el mismo indice que la entrada. Determinista, sin RNG. Solo
numpy/pandas/scipy/stdlib; sin efectos en import. ASCII puro: los caracteres acentuados de las
regex se escriben con escapes unicode (``\\uXXXX``) para mantener el fuente en ASCII sin
cambiar el patron compilado.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

# --- Patrones (src/advanced_features.py:28-37) ------------------------------------------
MENTION_RE = re.compile(r"@(\w+)")
HASHTAG_RE = re.compile(r"#(\w+)")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
#: Alternativa de palabra con vocales acentuadas (mismo patron que el original; el fuente usa
#: escapes unicode para quedar en ASCII).
WORD_RE = re.compile(r"[A-Za-z"
                     "\u00c1\u00c9\u00cd\u00d3\u00da\u00dc\u00d1"
                     "\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1"
                     r"0-9]+")
#: Tokenizador propio de tweets. La alternativa de palabra **debe** aceptar mayusculas: si no,
#: las palabras capitalizadas o en mayusculas se fragmentan y las features de forma quedan en
#: cero (bug detectado en la auditoria de alineacion de la Fase 3).
TOKEN_RE = re.compile(r"@\w+|#\w+|https?://\S+|[A-Za-z"
                      "\u00c1\u00c9\u00cd\u00d3\u00da\u00dc\u00d1"
                      "\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1"
                      r"]+|\d+|[^\s\w]", flags=re.UNICODE)
#: Tokens que contienen letras **y** digitos (p. ej. "abc123"), que el tokenizador separa
#: (``scripts/phase3b_recorregido.py:42``).
ALNUM_RE = re.compile(r"\b(?=\w*\d)(?=\w*[A-Za-z])\w+\b")


def _safe_entropy(counts: np.ndarray) -> float:
    """Entropia de Shannon (base 2) segura para vectores vacios (src/advanced_features.py:56-62)."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def _percentiles(values: np.ndarray, points: tuple[int, ...] = (10, 25, 50, 75, 90)) -> dict:
    """Percentiles de un vector (0 si esta vacio) (src/advanced_features.py:65-69)."""
    if values.size == 0:
        return {f"p{p}": 0.0 for p in points}
    return {f"p{p}": float(np.percentile(values, p)) for p in points}


# --------------------------------------------------------------------------------------
# Familias A, C - distribucion de la longitud de palabra (src/advanced_features.py:75-111)
# --------------------------------------------------------------------------------------
def distribution_features(texts: pd.Series) -> pd.DataFrame:
    """Familia A/C: distribucion interna de la longitud de palabra y de los tokens."""
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
            # Asimetria y curtosis requieren varianza no nula (si todos los tokens miden igual,
            # la forma no esta definida y se reporta 0 en lugar de NaN).
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
# Familia B, R - distribucion de caracteres y entropias (src/advanced_features.py:117-153)
# --------------------------------------------------------------------------------------
def char_distribution_features(texts: pd.Series) -> pd.DataFrame:
    """Familia B/R: composicion y concentracion del alfabeto usado en el tweet."""
    clean = texts.fillna("").astype(str)
    lowered = clean.str.lower()
    total = clean.str.len().replace(0, np.nan)
    vowels = lowered.str.count(r"[aeiou" "\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc" r"]")
    consonants = lowered.str.count(r"[bcdfghjklmnpqrstvwxyz" "\u00f1" r"]")
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
# Familia D - riqueza lexica avanzada (src/advanced_features.py:159-189)
# --------------------------------------------------------------------------------------
def lexical_richness_features(texts: pd.Series) -> pd.DataFrame:
    """Familia D: diversidad lexica estable para textos cortos (MATTR, Yule's K, Simpson)."""
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
# Familia H, T - forma de palabra y repeticion (src/advanced_features.py:195-226)
# --------------------------------------------------------------------------------------
def word_shape_features(texts: pd.Series) -> pd.DataFrame:
    """Familia H/T: patrones de escritura de las palabras y repeticion interna."""
    tokens = texts.fillna("").str.findall(TOKEN_RE.pattern)
    rows: list[dict] = []
    for token_list in tokens:
        # ``WORD_RE`` acepta mayusculas: si no, las features de forma de palabra quedarian en
        # cero (bug detectado en la auditoria de alineacion de la Fase 3).
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
# Familia I, M, U - posicion de elementos, URLs y posicion lexica
# (src/advanced_features.py:232-267)
# --------------------------------------------------------------------------------------
def position_features(texts: pd.Series) -> pd.DataFrame:
    """Familias I/M/U: posicion normalizada de entidades y de la primera/ultima palabra."""
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
# Familia N, R, S - estructura, entropia y compresibilidad (src/advanced_features.py:273-306)
# --------------------------------------------------------------------------------------
def structure_entropy_features(texts: pd.Series) -> pd.DataFrame:
    """Familias N/R/S: puntuacion, entropias por tipo y ratio de compresion."""
    clean = texts.fillna("").astype(str)
    total = clean.str.len().replace(0, np.nan)
    punctuation = {name: clean.str.count(pattern) for name, pattern in
                   (("comma", ","), ("period", r"\."), ("colon", ":"), ("semicolon", ";"),
                    ("parenthesis", r"[()]"), ("quote", r"[\"']"), ("dash", "[-\u2013\u2014]"),
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
# Correccion vigente (scripts/phase3b_recorregido.py:42-51)
# --------------------------------------------------------------------------------------
def add_alphanumeric_ratio(frame: pd.DataFrame, texts: pd.Series) -> pd.DataFrame:
    """Recalcula ``shape_alphanumeric_ratio`` sobre el texto crudo (phase3b:45-51).

    El tokenizador separa letras y digitos, asi que el conteo interno de ``word_shape_features``
    es siempre cero; aqui se cuentan los tokens mixtos con ``ALNUM_RE`` y se dividen por el
    numero de tokens de ``TOKEN_RE`` (denominador con piso 1).
    """
    counts = texts.fillna("").astype(str).apply(lambda value: len(ALNUM_RE.findall(value)))
    words = texts.fillna("").astype(str).str.findall(TOKEN_RE.pattern).apply(len).replace(0, 1)
    frame = frame.copy()
    frame["shape_alphanumeric_ratio"] = (counts / words).to_numpy()
    return frame


# --------------------------------------------------------------------------------------
# Ensamblado (scripts/advanced_02_features.py:94-104 / scripts/phase3b_recorregido.py:54-69)
# --------------------------------------------------------------------------------------
def extraer_estructuradas(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula las 87 features estructuradas de segunda generacion.

    Ensambla las seis familias en el orden original (``scripts/advanced_02_features.py:38-45,
    94-104``; ``scripts/phase3b_recorregido.py:54-69``) y aplica la correccion vigente de
    ``scripts/phase3b_recorregido.py:42-51`` (recalculo de ``shape_alphanumeric_ratio`` con
    ``ALNUM_RE`` sobre el texto crudo).

    Parameters
    ----------
    df : pd.DataFrame
        Marco con columna ``text`` (texto limpio, sin padding).

    Returns
    -------
    pd.DataFrame
        87 columnas (73 float64 + 14 int64) con el mismo indice que ``df``.
    """
    # Los textos se procesan en orden de fila con indice neutro (los builders construyen sus
    # marcos de forma posicional); al final se restituye el indice de la entrada.
    textos = df["text"].reset_index(drop=True)
    builders = (("A_distribucion", distribution_features),
                ("B_char", char_distribution_features),
                ("D_riqueza", lexical_richness_features),
                ("HT_forma", word_shape_features),
                ("IMU_posicion", position_features),
                ("NRS_estructura", structure_entropy_features))
    frames = [builder(textos) for _name, builder in builders]
    combined = pd.concat(frames, axis=1)
    combined = add_alphanumeric_ratio(combined, textos)
    combined.index = df.index
    return combined
