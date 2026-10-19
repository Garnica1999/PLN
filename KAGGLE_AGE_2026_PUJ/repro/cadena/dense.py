"""Textometria y features estilometricas a nivel de documento (porte literal).

Porte de ``src/textometry.py`` (lineas 1-173) a ``repro/cadena/dense.py``. La funcion
publica ``extraer_textometria`` reproduce exactamente el original: 38 columnas en el
orden ``TEXTOMETRY_FEATURES + STYLE_FEATURES``. Las regex ``_EMOJI_RE`` y ``_URL_RE``
se copian de ``src/preprocessing.py:18,25-32`` como constantes locales del modulo.

Todas las features son deterministas y locales al documento: no aprenden parametros del
corpus, por lo que pueden calcularse sobre train y test sin riesgo de leakage.

Sin dependencias del proyecto (solo stdlib + numpy/pandas), ASCII puro y sin efectos en
import.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Emoticonos textuales clasicos (con fronteras de palabra para evitar falsos positivos).
# src/textometry.py:24-26
EMOTICON_RE = re.compile(
    r"(?<![\w:/])(?:[:;=8][-o*']?[\)\]\(\[dDpP/:}{@|]|x[dDpP](?!\w)|XD(?!\w)|<3)(?![\w])")
# Palabras con elongacion (3 o mas repeticiones de un caracter). src/textometry.py:27-28
ELONGATED_RE = re.compile(r"\b\w*(\w)\1{2,}\w*\b", flags=re.UNICODE)
# Secuencias de puntuacion repetida (!!, ??, ..., !?, etc.). src/textometry.py:29-30
MULTI_PUNCT_RE = re.compile(r"[!?.,;:]{2,}")
# Signos de apertura del espanol. src/textometry.py:31-32
OPEN_PUNCT_RE = re.compile(r"[\u00a1\u00bf]")
# Palabras completamente en mayusculas (2+ letras). src/textometry.py:33-34
ALL_CAPS_RE = re.compile(r"\b[A-Z\u00c1\u00c9\u00cd\u00d3\u00da\u00dc\u00d1]{2,}\b")
# Separador de frases. src/textometry.py:35-36
SENTENCE_SPLIT_RE = re.compile(r"[.!?\u2026]+|\n+")

# Lista fija de stopwords en espanol (recurso lexico cerrado, sin parametros aprendidos).
# src/textometry.py:38-48
SPANISH_STOPWORDS = frozenset(["a", "al", "algo", "algunas", "alguno", "algunos", "ambos", "ante", "antes", "aquel", "aquella", "aquellas", "aquellos", "aqui", "asi", "aun", "aunque", "bien", "cada", "casi", "como", "con", "contra", "cual", "cuales", "cuando", "cuanto", "cuenta", "da", "de", "del", "desde", "donde", "dos", "el", "ella", "ellas", "ello", "ellos", "en", "entonces", "entre", "era", "eran", "es", "esa", "esas", "ese", "eso", "esos", "esta", "estaba", "estado", "estan", "estar", "estas", "este", "esto", "estos", "fue", "fueron", "ha", "habia", "han", "hasta", "hay", "la", "las", "le", "les", "lo", "los", "mas", "me", "mi", "mientras", "mis", "mucho", "muchos", "muy", "nada", "ni", "no", "nos", "nosotros", "nuestra", "nuestro", "nuestros", "o", "os", "otra", "otras", "otro", "otros", "para", "pero", "poco", "por", "porque", "que", "quien", "quienes", "se", "sea", "segun", "ser", "si", "sin", "sobre", "solo", "son", "su", "sus", "tal", "tambien", "tanto", "te", "tener", "tiene", "tienen", "todo", "todos", "tu", "tus", "un", "una", "uno", "unos", "usted", "ustedes", "va", "vamos", "van", "varias", "varios", "ver", "y", "ya", "yo"])

# Familias de features (orden exacto del original). src/textometry.py:50-62
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

# Patrones compartidos copiados de src/preprocessing.py (_URL_RE en la linea 18;
# _EMOJI_RE en las lineas 25-32).
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF\U0001FA00-\U0001FAFF\U0001F1E6-\U0001F1FF"
    "\U00002600-\U000027BF\U00002B00-\U00002BFF\U0000FE0F"
    "]"
)


def _entropy(counts: np.ndarray) -> float:
    """Entropia de Shannon (base 2) de un vector de conteos (src/textometry.py:65-71)."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def extraer_textometria(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula las features textometricas y estilometricas de cada documento.

    Porte literal de ``src/textometry.py:83-173`` (helper ``_entropy`` en 65-71). El
    original acepta ``text_col: str = "text"``; en este porte la columna es fija
    ``"text"``, segun la interfaz de ``repro/cadena``. La entrada debe traer el texto
    ya limpio (strip) y la salida conserva el indice de ``df``, con 38 columnas en el
    orden ``TEXTOMETRY_FEATURES + STYLE_FEATURES``.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con la columna ``text`` (ya sin padding).

    Returns
    -------
    pandas.DataFrame
        Una fila por documento, con las columnas de ``TEXTOMETRY_FEATURES`` y
        ``STYLE_FEATURES``.
    """
    text = df["text"].fillna("").astype(str)
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

    letters = text.str.count(r"[A-Za-z\u00c1\u00c9\u00cd\u00d3\u00da\u00dc\u00d1\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1]")
    upper = text.str.count(r"[A-Z\u00c1\u00c9\u00cd\u00d3\u00da\u00dc\u00d1]")
    digits = text.str.count(r"\d")
    punct = text.str.count(r"[^\w\s]")
    sentences = text.apply(lambda value: len([s for s in SENTENCE_SPLIT_RE.split(value) if s.strip()]))
    stopword_hits = words.apply(lambda tokens: sum(1 for token in tokens
                                                   if token.strip(".,;:!?\u00a1\u00bf\"'()") in SPANISH_STOPWORDS))
    long_words = words.apply(lambda tokens: sum(1 for token in tokens if len(token) >= 8))

    features = pd.DataFrame({
        # --- textometria ---
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
