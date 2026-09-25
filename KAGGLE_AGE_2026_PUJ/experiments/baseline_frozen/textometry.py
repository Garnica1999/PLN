"""Textometría y features estilométricas a nivel de documento.

Extrae, para cada tweet, medidas de tamaño, riqueza léxica, puntuación, mayúsculas, dígitos,
repeticiones, elongaciones, emojis, emoticonos, hashtags, menciones, URLs y retuits.

Todas las features son **deterministas y locales al documento**: no aprenden parámetros del
corpus, por lo que pueden calcularse sobre train y test sin riesgo de leakage (el riesgo de
leakage aparece al *seleccionar* features con etiquetas o al ajustar vocabularios, y eso se
hace siempre dentro de los folds — ver D-003).

Las features se agrupan en familias (`GROUP_D_TEXTOMETRY`, `GROUP_C_STYLE`) para la ablación
de la Fase 18.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.preprocessing import _EMOJI_RE, _URL_RE  # patrones compartidos

#: Emoticonos textuales clásicos (con fronteras de palabra para evitar falsos positivos).
EMOTICON_RE = re.compile(
    r"(?<![\w:/])(?:[:;=8][-o*']?[\)\]\(\[dDpP/:}{@|]|x[dDpP](?!\w)|XD(?!\w)|<3)(?![\w])")
#: Palabras con elongación (3 o más repeticiones de un carácter).
ELONGATED_RE = re.compile(r"\b\w*(\w)\1{2,}\w*\b", flags=re.UNICODE)
#: Secuencias de puntuación repetida (!!, ??, ..., !?, etc.).
MULTI_PUNCT_RE = re.compile(r"[!?.,;:]{2,}")
#: Signos de apertura del español.
OPEN_PUNCT_RE = re.compile(r"[¡¿]")
#: Palabras completamente en mayúsculas (2+ letras).
ALL_CAPS_RE = re.compile(r"\b[A-ZÁÉÍÓÚÜÑ]{2,}\b")
#: Separador de frases.
SENTENCE_SPLIT_RE = re.compile(r"[.!?…]+|\n+")

#: Lista fija de stopwords en español (recurso léxico cerrado, sin parámetros aprendidos).
SPANISH_STOPWORDS = frozenset("""
a al algo algunas alguno algunos ambos ante antes aquel aquella aquellas aquellos aqui asi
aun aunque bien cada casi como con contra cual cuales cuando cuanto cuenta da de del desde
donde dos el ella ellas ello ellos en entonces entre era eran es esa esas ese eso esos esta
estaba estado estan estar estas este esto estos fue fueron ha habia han hasta hay la las le
les lo los mas me mi mientras mis mucho muchos muy nada ni no nos nosotros nuestra nuestro
nuestros o os otra otras otro otros para pero poco por porque que quien quienes se sea segun
ser si sin sobre solo son su sus tal tambien tanto te tener tiene tienen todo todos tu tus un
una uno unos usted ustedes va vamos van varias varios ver y ya yo
""".split())

#: Familias de features para la ablación.
TEXTOMETRY_FEATURES = [
    "n_chars", "n_words", "n_unique_words", "ttr", "cttr", "hapax_ratio", "lexical_entropy",
    "avg_word_len", "max_word_len", "std_word_len", "n_sentences", "avg_sentence_len_words",
    "stopword_ratio", "content_ratio", "long_word_ratio",
]
STYLE_FEATURES = [
    "upper_ratio", "digit_ratio", "punct_ratio", "n_excl", "n_quest", "n_open_punct",
    "multi_punct_count", "elongated_count", "all_caps_count", "has_all_caps",
    "n_emoji", "emoji_diversity", "has_emoji", "n_emoticon", "has_emoticon",
    "n_hashtag", "n_mention", "n_url", "is_rt", "starts_mention", "n_repeated_char_runs",
    "punct_per_word", "unique_char_ratio",
]


def _entropy(counts: np.ndarray) -> float:
    """Entropía de Shannon (base 2) de un vector de conteos."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def _word_stats(tokens: list[str]) -> tuple[int, int, float, float]:
    """Número de palabras únicas, hapax, longitud media y desviación de longitud."""
    if not tokens:
        return 0, 0, 0.0, 0.0
    lengths = np.array([len(token) for token in tokens])
    unique, counts = np.unique(tokens, return_counts=True)
    return int(unique.size), int((counts == 1).sum()), float(lengths.mean()), float(lengths.std())


def extract_textometry(frame: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Calcula las features textométricas y estilométricas de cada documento.

    Parameters
    ----------
    frame : pandas.DataFrame
        DataFrame con la columna de texto (ya sin padding).
    text_col : str
        Nombre de la columna de texto.

    Returns
    -------
    pandas.DataFrame
        Un DataFrame con una fila por documento y las columnas de
        :data:`TEXTOMETRY_FEATURES` y :data:`STYLE_FEATURES`.
    """
    text = frame[text_col].fillna("").astype(str)
    lowered = text.str.lower()
    words = lowered.str.split()
    # Los emoticonos se cuentan sobre el texto sin URLs: "http://" contiene ":/" y
    # contaminaba el conteo (detectado en el EDA de la Fase 2).
    text_without_urls = text.str.replace(_URL_RE, " ", regex=True)

    n_chars = text.str.len()
    n_words = words.str.len()
    n_unique = words.apply(lambda tokens: len(set(tokens)) if tokens else 0)
    n_hapax = words.apply(
        lambda tokens: int(sum(1 for count in pd.Series(tokens).value_counts() if count == 1))
        if tokens else 0)
    entropy = words.apply(lambda tokens: _entropy(np.unique(tokens, return_counts=True)[1])
                          if tokens else 0.0)
    word_len_mean = words.apply(lambda tokens: float(np.mean([len(t) for t in tokens]))
                                if tokens else 0.0)
    word_len_max = words.apply(lambda tokens: int(max((len(t) for t in tokens), default=0)))
    word_len_std = words.apply(lambda tokens: float(np.std([len(t) for t in tokens]))
                               if tokens else 0.0)

    letters = text.str.count(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]")
    upper = text.str.count(r"[A-ZÁÉÍÓÚÜÑ]")
    digits = text.str.count(r"\d")
    punct = text.str.count(r"[^\w\s]")
    sentences = text.apply(lambda value: len([s for s in SENTENCE_SPLIT_RE.split(value) if s.strip()]))
    stopword_hits = words.apply(lambda tokens: sum(1 for token in tokens
                                                   if token.strip(".,;:!?¡¿\"'()") in SPANISH_STOPWORDS))
    long_words = words.apply(lambda tokens: sum(1 for token in tokens if len(token) >= 8))

    features = pd.DataFrame({
        # --- textometría ---
        "n_chars": n_chars,
        "n_words": n_words,
        "n_unique_words": n_unique,
        "ttr": (n_unique / n_words.replace(0, np.nan)).fillna(0.0),
        "cttr": (n_unique / np.sqrt(2 * n_words.replace(0, np.nan))).fillna(0.0),
        "hapax_ratio": (n_hapax / n_words.replace(0, np.nan)).fillna(0.0),
        "lexical_entropy": entropy,
        "avg_word_len": word_len_mean,
        "max_word_len": word_len_max,
        "std_word_len": word_len_std,
        "n_sentences": sentences,
        "avg_sentence_len_words": (n_words / sentences.replace(0, np.nan)).fillna(0.0),
        "stopword_ratio": (stopword_hits / n_words.replace(0, np.nan)).fillna(0.0),
        "content_ratio": (1 - (stopword_hits / n_words.replace(0, np.nan))).fillna(0.0),
        "long_word_ratio": (long_words / n_words.replace(0, np.nan)).fillna(0.0),
        # --- estilo ---
        "upper_ratio": (upper / letters.replace(0, np.nan)).fillna(0.0),
        "digit_ratio": (digits / n_chars.replace(0, np.nan)).fillna(0.0),
        "punct_ratio": (punct / n_chars.replace(0, np.nan)).fillna(0.0),
        "n_excl": text.str.count("!"),
        "n_quest": text.str.count(r"\?"),
        "n_open_punct": text.str.count(OPEN_PUNCT_RE.pattern),
        "multi_punct_count": text.str.count(MULTI_PUNCT_RE.pattern),
        "elongated_count": text.str.count(ELONGATED_RE.pattern),
        "all_caps_count": text.str.count(ALL_CAPS_RE.pattern),
        "has_all_caps": text.str.contains(ALL_CAPS_RE.pattern).astype(int),
        "n_emoji": text.str.count(_EMOJI_RE.pattern),
        "emoji_diversity": text.apply(
            lambda value: len(set(_EMOJI_RE.findall(value))) if value else 0),
        "has_emoji": text.str.contains(_EMOJI_RE.pattern).astype(int),
        "n_emoticon": text_without_urls.str.count(EMOTICON_RE.pattern),
        "has_emoticon": text_without_urls.str.contains(EMOTICON_RE.pattern).astype(int),
        "n_hashtag": text.str.count("#"),
        "n_mention": text.str.count("@"),
        "n_url": text.str.count(r"https?://|www\."),
        "is_rt": text.str.match(r"^\s*RT\b", case=False).astype(int),
        "starts_mention": text.str.match(r"^\s*@").astype(int),
        "n_repeated_char_runs": text.str.count(r"(\w)\1{1,}"),
        "punct_per_word": (punct / n_words.replace(0, np.nan)).fillna(0.0),
        "unique_char_ratio": (text.apply(lambda value: len(set(value)))
                              / n_chars.replace(0, np.nan)).fillna(0.0),
    })
    return features


def textometry_summary(features: pd.DataFrame, target: pd.Series,
                       columns: list[str] | None = None) -> pd.DataFrame:
    """Media de cada feature por clase objetivo."""
    columns = columns or (TEXTOMETRY_FEATURES + STYLE_FEATURES)
    working = features[columns].copy()
    working["_target"] = np.asarray(target.astype(str))
    return working.groupby("_target").mean().T


def constant_feature_report(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Detecta features constantes o casi constantes (varianza nula o casi nula).

    Una feature constante no aporta información y debe excluirse antes de modelar; además
    delata propiedades del corpus (por ejemplo, la ausencia total de emojis).

    Returns
    -------
    pandas.DataFrame
        ``feature, n_unique, top_value, top_share, varianza, es_constante``.
    """
    rows: list[dict] = []
    total = len(features)
    for column in columns:
        series = features[column]
        counts = series.value_counts(dropna=False)
        top_value = counts.index[0] if len(counts) else np.nan
        top_share = float(counts.iloc[0] / total) if len(counts) else 0.0
        rows.append({
            "feature": column,
            "n_unique": int(series.nunique(dropna=False)),
            "top_value": top_value,
            "top_share": round(top_share, 4),
            "varianza": float(series.var()),
            "es_constante": bool(series.nunique(dropna=False) <= 1),
        })
    return pd.DataFrame(rows).sort_values("n_unique").reset_index(drop=True)
