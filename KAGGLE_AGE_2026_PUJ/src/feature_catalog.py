"""Catálogo de features del proyecto (entregable de la Fase 10).

Cada feature declara: nombre, grupo, descripción, fórmula, tipo, fuente, hipótesis que la
justifica y riesgo de leakage. Los grupos siguen la nomenclatura del plan (A-H).

Nota sobre grupos sin contenido: `GROUP_G_EMOTION` quedó vacío porque el corpus es ASCII puro
(cero emojis, D-009) y `GROUP_E_POS` está pendiente de decisión (el POS tagger de spaCy es un
recurso preentrenado de zona gris, D-001); se documentan igualmente para dejar constancia.
"""

from __future__ import annotations

import pandas as pd

from src import textometry

GROUP_A_CONTENT = "A - contenido (word n-grams)"
GROUP_B_CHARACTER = "B - caracteres (char n-grams)"
GROUP_C_STYLE = "C - estilo"
GROUP_D_TEXTOMETRY = "D - textometría"
GROUP_E_POS = "E - POS/sintaxis"
GROUP_F_SENTIMENT = "F - sentimiento"
GROUP_G_EMOTION = "G - emoción/emoji"
GROUP_H_LEXICON = "H - léxico específico"

#: Descripciones y fórmulas de las features densas calculadas en ``src/textometry.py``.
DENSE_DESCRIPTIONS: dict[str, tuple[str, str, str]] = {
    # nombre: (descripción, fórmula, hipótesis)
    "n_chars": ("Número de caracteres del tweet sin padding", "len(text)", "H1"),
    "n_words": ("Número de palabras separadas por espacios", "len(text.split())", "H1"),
    "n_unique_words": ("Tipos léxicos distintos en el tweet", "|set(words)|", "H1"),
    "ttr": ("Type-Token Ratio", "n_unique_words / n_words", "H1"),
    "cttr": ("TTR corregido por longitud", "n_unique / sqrt(2 * n_words)", "H1"),
    "hapax_ratio": ("Proporción de palabras que aparecen una sola vez",
                    "hapax / n_words", "H1"),
    "lexical_entropy": ("Entropía de Shannon del tweet", "-sum(p log2 p) sobre frecuencias",
                        "H1"),
    "avg_word_len": ("Longitud media de palabra", "mean(len(w))", "H1"),
    "max_word_len": ("Longitud máxima de palabra", "max(len(w))", "H1"),
    "std_word_len": ("Desviación de la longitud de palabra", "std(len(w))", "H1"),
    "n_sentences": ("Número de oraciones", "split por [.!?…]", "H1"),
    "avg_sentence_len_words": ("Palabras por oración", "n_words / n_sentences", "H1"),
    "stopword_ratio": ("Proporción de palabras funcionales", "stopwords / n_words", "H1"),
    "content_ratio": ("Proporción de palabras de contenido", "1 - stopword_ratio", "H1"),
    "long_word_ratio": ("Proporción de palabras de 8+ caracteres",
                        "palabras_largas / n_words", "H1"),
    "upper_ratio": ("Proporción de letras mayúsculas", "mayúsculas / letras", "H2"),
    "digit_ratio": ("Proporción de dígitos", "dígitos / n_chars", "H2"),
    "punct_ratio": ("Proporción de signos de puntuación", "puntuación / n_chars", "H2"),
    "n_excl": ("Signos de exclamación", "count('!')", "H2"),
    "n_quest": ("Signos de interrogación", "count('?')", "H2"),
    "n_open_punct": ("Signos de apertura del español", "count('¡') + count('¿')", "H2"),
    "multi_punct_count": ("Secuencias de puntuación repetida", "count([!?.,;:]{2,})", "H2"),
    "elongated_count": ("Palabras con 3+ letras repetidas", "count(\\w*(\\w)\\1{2,}\\w*)", "H8"),
    "all_caps_count": ("Palabras completamente en mayúsculas", "count([A-ZÁÉÍÓÚÑ]{2,})", "H2"),
    "has_all_caps": ("Indicador de palabra en mayúsculas", "all_caps_count > 0", "H2"),
    "n_emoji": ("Emojis (constante 0 en este corpus)", "count(emoji)", "H9 (no medible)"),
    "emoji_diversity": ("Emojis distintos (constante 0)", "|set(emojis)|", "H9 (no medible)"),
    "has_emoji": ("Indicador de emoji (constante 0)", "n_emoji > 0", "H9 (no medible)"),
    "n_emoticon": ("Emoticonos textuales, contados sin URLs",
                   "count(emoticon) sobre texto sin URLs", "H2"),
    "has_emoticon": ("Indicador de emoticono textual", "n_emoticon > 0", "H2"),
    "n_hashtag": ("Número de hashtags", "count('#')", "H9"),
    "n_mention": ("Número de menciones", "count('@')", "H10"),
    "n_url": ("Número de URLs", "count(https?://|www\\.)", "H10"),
    "is_rt": ("Indicador de retuit", "text startswith RT", "H3"),
    "starts_mention": ("El tweet empieza con mención (respuesta)", "text startswith @", "H10"),
    "n_repeated_char_runs": ("Runs de caracteres repetidos", "count((\\w)\\1{1,})", "H8"),
    "punct_per_word": ("Puntuación por palabra", "puntuación / n_words", "H2"),
    "unique_char_ratio": ("Diversidad de caracteres", "|set(chars)| / n_chars", "H1"),
}

DROPPED_FEATURES = {"n_emoji", "emoji_diversity", "has_emoji"}


def build_catalog() -> pd.DataFrame:
    """Construye el catálogo completo de features del proyecto.

    Returns
    -------
    pandas.DataFrame
        Columnas: ``name, group, description, formula, dtype, source, hypothesis, leakage_risk``.
    """
    rows: list[dict] = []

    # --- Grupo A: word n-grams ---
    rows.append({
        "name": "word_tfidf_1_2", "group": GROUP_A_CONTENT,
        "description": "TF-IDF sublinear de unigramas y bigramas de palabra (min_df=3, 300k)",
        "formula": "tf * (1 + log(tf)) * idf", "dtype": "sparse float32",
        "source": "src/representations (scripts/experimentos_representaciones.py)",
        "hypothesis": "H7 (léxico temático)", "leakage_risk": "bajo: vectorizador ajustado en fold",
    })

    # --- Grupo B: char n-grams ---
    rows.append({
        "name": "char_wb_tfidf_3_6", "group": GROUP_B_CHARACTER,
        "description": "TF-IDF sublinear de n-gramas de carácter 3-6 con frontera de palabra "
                       "(min_df=5, 300k). Representación principal tras la Etapa B.",
        "formula": "tf * (1 + log(tf)) * idf sobre char_wb(3,6)", "dtype": "sparse float32",
        "source": "src/representations (scripts/experimentos_representaciones.py)",
        "hypothesis": "H6 (ortografía/morfología)", "leakage_risk": "bajo: ajustado en fold",
    })

    # --- Grupo C/D: features densas de estilo y textometría ---
    for name, (description, formula, hypothesis) in DENSE_DESCRIPTIONS.items():
        if name in DROPPED_FEATURES:
            group = GROUP_G_EMOTION
            dropped = True
        elif name in textometry.TEXTOMETRY_FEATURES:
            group = GROUP_D_TEXTOMETRY
            dropped = False
        else:
            group = GROUP_C_STYLE
            dropped = False
        rows.append({
            "name": name, "group": group, "description": description, "formula": formula,
            "dtype": "float64", "source": "src/textometry.py",
            "hypothesis": hypothesis,
            "leakage_risk": ("descartada por varianza nula (D-009)" if dropped
                             else "nulo: local al documento, no aprende del corpus"),
        })

    # --- Grupo H: canal de menciones y hashtags ---
    rows.append({
        "name": "mention_tfidf_1gram", "group": GROUP_H_LEXICON,
        "description": "TF-IDF de unigramas sobre SOLO los tokens de mención (@cuenta). "
                       "Aportó +0,0144 de macro-F1 sobre char n-grams (5/5 folds).",
        "formula": "TF-IDF sobre tokens '@\\w+' extraídos del tweet", "dtype": "sparse float32",
        "source": "src/representations (scripts/experimentos_representaciones.py)",
        "hypothesis": "H10 (interacción social)", "leakage_risk": "bajo: ajustado en fold",
    })
    rows.append({
        "name": "hashtag_tokens", "group": GROUP_H_LEXICON,
        "description": "Tokens de hashtag analizados con log-odds (EDA); candidatos a canal "
                       "propio si la ablación lo respalda.",
        "formula": "tokens de '#\\w+'", "dtype": "sparse float32",
        "source": "src/eda.py (análisis) — canal pendiente de evaluar",
        "hypothesis": "H9", "leakage_risk": "bajo",
    })

    # --- Grupos declarados sin contenido ---
    rows.append({
        "name": "(sin features)", "group": GROUP_E_POS,
        "description": "POS/sintaxis pendiente: el POS tagger de spaCy es un recurso "
                       "preentrenado de zona gris (D-001) y el corpus ASCII degrada su calidad.",
        "formula": "-", "dtype": "-", "source": "pendiente", "hypothesis": "-",
        "leakage_risk": "no aplica",
    })
    rows.append({
        "name": "(sin features)", "group": GROUP_F_SENTIMENT,
        "description": "Sentimiento: los lexicones externos en español no casan con el texto "
                       "transliterado a ASCII; alternativa planificada: léxicos internos por "
                       "regresión L2 (Sap et al. 2014, D-008).",
        "formula": "-", "dtype": "-", "source": "pendiente (Fase 9)", "hypothesis": "-",
        "leakage_risk": "no aplica",
    })

    return pd.DataFrame(rows)
