"""Normalización de texto y variantes de preprocesamiento.

Regla del proyecto: **no se asume que limpiar sea mejor**. Cada variante de normalización
(URLs, menciones, hashtags, emojis, mayúsculas, stopwords, lematización) se evalúa como un
experimento controlado en la Fase 4; este módulo solo ofrece las transformaciones y la
variante "ligera" que usa el EDA para contar términos de forma interpretable.

Todas las transformaciones son deterministas y no aprenden parámetros del corpus (salvo las
que usan listas de stopwords, que son recursos fijos).
"""

from __future__ import annotations

import re

import pandas as pd

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_RT_RE = re.compile(r"^\s*rt\s*:?\s*", flags=re.IGNORECASE)
_HTML_RE = re.compile(r"&amp;|&lt;|&gt;|&quot;|&#\d+;")
_SPACES_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ]")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF\U0001FA00-\U0001FAFF\U0001F1E6-\U0001F1FF"
    "\U00002600-\U000027BF\U00002B00-\U00002BFF\U0000FE0F"
    "]"
)


def light_normalize(series: pd.Series, keep_case: bool = False,
                    keep_punctuation: bool = False) -> pd.Series:
    """Normaliza para análisis de términos: sin URLs, menciones ni prefijo RT.

    Parameters
    ----------
    series : pandas.Series
        Texto crudo.
    keep_case : bool
        Si es ``False`` (por defecto) pasa a minúsculas.
    keep_punctuation : bool
        Si es ``False`` (por defecto) elimina signos conservando acentos y ñ.

    Returns
    -------
    pandas.Series
        Texto normalizado y con espacios colapsados.
    """
    text = series.fillna("").astype(str)
    if not keep_case:
        text = text.str.lower()
    text = text.str.replace(_URL_RE, " ", regex=True)
    text = text.str.replace(_MENTION_RE, " ", regex=True)
    text = text.str.replace(_RT_RE, "", regex=True)
    text = text.str.replace(_HTML_RE, " ", regex=True)
    text = text.str.replace(_HASHTAG_RE, r"\1", regex=True)  # conserva la palabra del hashtag
    if not keep_punctuation:
        text = text.str.replace(_PUNCT_RE, " ", regex=True)
    return text.str.replace(_SPACES_RE, " ", regex=True).str.strip()


def strip_urls(series: pd.Series) -> pd.Series:
    """Reemplaza URLs por un marcador ``<url>`` (preserva la señal de su presencia)."""
    return series.fillna("").astype(str).str.replace(_URL_RE, " <url> ", regex=True)


def strip_mentions(series: pd.Series) -> pd.Series:
    """Reemplaza menciones por ``<user>``."""
    return series.fillna("").astype(str).str.replace(_MENTION_RE, " <user> ", regex=True)


def strip_hashtags(series: pd.Series, keep_word: bool = True) -> pd.Series:
    """Normaliza hashtags: conserva la palabra (por defecto) o los marca como ``<hash>``."""
    if keep_word:
        return series.fillna("").astype(str).str.replace(_HASHTAG_RE, r"\1", regex=True)
    return series.fillna("").astype(str).str.replace(_HASHTAG_RE, " <hash> ", regex=True)


def strip_rt_prefix(series: pd.Series) -> pd.Series:
    """Elimina el prefijo de retuit."""
    return series.fillna("").astype(str).str.replace(_RT_RE, "", regex=True)


def replace_emojis(series: pd.Series, replacement: str = " <emoji> ") -> pd.Series:
    """Sustituye emojis por un marcador."""
    return series.fillna("").astype(str).str.replace(_EMOJI_RE, replacement, regex=True)


def remove_emojis(series: pd.Series) -> pd.Series:
    """Elimina emojis del texto."""
    return series.fillna("").astype(str).str.replace(_EMOJI_RE, " ", regex=True)


def collapse_spaces(series: pd.Series) -> pd.Series:
    """Colapsa espacios repetidos y recorta extremos."""
    return series.fillna("").astype(str).str.replace(_SPACES_RE, " ", regex=True).str.strip()


def remove_urls(series: pd.Series) -> pd.Series:
    """Elimina por completo las URLs (variante de la descomposición de normalización)."""
    return collapse_spaces(series.fillna("").astype(str).str.replace(_URL_RE, " ", regex=True))


def remove_mentions(series: pd.Series) -> pd.Series:
    """Elimina por completo las menciones."""
    return collapse_spaces(series.fillna("").astype(str).str.replace(_MENTION_RE, " ", regex=True))


def remove_punctuation(series: pd.Series) -> pd.Series:
    """Elimina la puntuación conservando letras, dígitos y espacios."""
    return collapse_spaces(series.fillna("").astype(str).str.replace(_PUNCT_RE, " ", regex=True))


def url_mention_markers(series: pd.Series) -> pd.Series:
    """Sustituye URLs y menciones por marcadores ``<url>`` y ``<user>``.

    Conserva la señal de *presencia y posición* (que en este corpus es informativa) sin
    arrastrar el contenido específico del enlace o del usuario.
    """
    text = series.fillna("").astype(str)
    text = text.str.replace(_URL_RE, " <url> ", regex=True)
    text = text.str.replace(_MENTION_RE, " <user> ", regex=True)
    return collapse_spaces(text)


def emoji_pattern() -> re.Pattern:
    """Devuelve el patrón de emojis usado por el proyecto."""
    return _EMOJI_RE
