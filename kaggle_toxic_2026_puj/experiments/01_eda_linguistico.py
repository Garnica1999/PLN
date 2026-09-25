"""Fase 2 - EDA profundo, textometria y caracterizacion linguistica de TR.

Pregunta que responde
---------------------
Sobre el corpus oficial de ``tweet-toxic-2026-puj`` (1.731 train / 1.154 test):

1. Como se distribuyen las etiquetas HS, TR y AG y sus combinaciones, y que
   propiedades de superficie (longitud, vocabulario, puntuacion, mayusculas,
   hashtags, menciones, URLs, emojis, repeticiones) las acompanan.
2. Que fenomenos de superficie se asocian de forma estadisticamente creible con
   cada etiqueta, con tamano de efecto y control de comparaciones multiples.
3. Que lectura de la etiqueta TR es mas consistente con los datos: la
   definicion del enunciado ("dirigido contra una persona o grupo") o la del
   esquema HatEval (rango del objetivo: individuo frente a grupo). La decision
   queda marcada como pendiente de revision humana (D-005).

Metodo
------
1. Carga con :mod:`src.data` e integridad (1731/1154 filas, sin nulos).
2. EDA por etiqueta y por combinacion HS x TR x AG (longitudes, vocabulario,
   TTR, hapax, entropia lexica).
3. Textometria: extractor de fenomenos de superficie por documento y extractor
   de rasgos linguisticos con spaCy (POS, lemas y morfologia; sin vectores,
   D-003). Las listas lexicas usadas (agresividad y colectivos) se INDUCEN de
   train; no se usa ningun recurso ni etiqueta externa (D-002).
4. Estadistica con pregunta explicita, tamano de efecto y correccion de
   Benjamini-Hochberg: chi cuadrado + V de Cramer (o Fisher exacto si alguna
   celda esperada < 5) para fenomenos binarios, Mann-Whitney U + delta de Cliff
   para variables continuas con 2 grupos, Kruskal-Wallis + epsilon cuadrado para
   3 o mas grupos, e informacion mutua. Todos los contrastes se repiten sobre
   los datos deduplicados por grupo de near-duplicado, porque los
   near-duplicados inflan la significancia.
5. Caracterizacion de TR dentro de HS=1 (n=756; TR=1 519, TR=0 237):
   contraste cuantitativo de rasgos de targeting, tabla cualitativa de
   ejemplos para revision humana y conclusion preliminar con evidencia.
6. Figuras PNG (dpi 150) en ``reports/figuras/`` y tablas CSV en
   ``reports/eda/``.
7. ``reports/eda_resumen.md`` con el formato OBSERVACION -> HIPOTESIS ->
   METODO -> RESULTADO -> INTERPRETACION -> DECISION -> SIGUIENTE PASO.

Restricciones respetadas
------------------------
- Sin transformers, LLM ni embeddings preentrenados; spaCy solo como anotador
  morfosintactico (D-003), sin ``doc.vector`` ni ``similarity``.
- Sin etiquetas ni corpus externos (D-002): las listas lexicas se inducen de
  train y se documentan en ``reports/eda/fase2_lexico_inducido_*.csv``.
- El test nunca se usa para ajustar nada; solo se describen sus caracteristicas.
- NER desactivado de forma deliberada: D-003 autoriza POS, lemas y dependencias;
  reconocer entidades seria una funcionalidad adicional no autorizada.

Salidas
-------
``reports/eda/fase2_*.csv``, ``reports/eda/tr_ejemplos_para_revision.csv``,
``reports/figuras/fig_*.png`` y ``reports/eda_resumen.md``.
Se agrega una seccion de 20 lineas al final de ``reports/contexto.md``.

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/01_eda_linguistico.py
"""

from __future__ import annotations

import re
import string
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import emoji
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from scipy.stats import chi2_contingency, fisher_exact, kruskal, mannwhitneyu, spearmanr
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score
from spacy.lang.es.stop_words import STOP_WORDS

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module

REPORTS_DIR = PROJECT_ROOT / "reports"
EDA_DIR = REPORTS_DIR / "eda"
FIGURES_DIR = REPORTS_DIR / "figuras"
GROUPS_PATH = PROJECT_ROOT / "data" / "processed" / "grupos_near_duplicados.csv"

LABELS: tuple[str, ...] = ("HS", "TR", "AG")
RANDOM_STATE = 42
ALPHA = 0.05
EXPECTED_TRAIN_ROWS = 1731
EXPECTED_TEST_ROWS = 1154
MIN_EXPECTED_CELL = 5.0
N_REVIEW_SAMPLES = 40

# Paleta fija y consistente para las tres etiquetas (sin arcoiris).
PALETA_ETIQUETA: dict[str, str] = {
    "HS": "#1f4e79",
    "TR": "#b03a2e",
    "AG": "#b9770e",
}
PALETA_AUSENCIA_PRESENCIA: dict[int, str] = {0: "#9aa5b1", 1: "#b03a2e"}
PALETA_SECUENCIAL = "YlGnBu"

# ---------------------------------------------------------------------------
# Patrones de superficie
# ---------------------------------------------------------------------------
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
WORD_PATTERN = re.compile(r"[A-Za-z0-9@#_]+")
NUMBER_PATTERN = re.compile(r"\d[\d.,]*")
LETTER_REPEAT_PATTERN = re.compile(r"([A-Za-z])\1{2,}")
PUNCT_REPEAT_PATTERN = re.compile(r"([!?.,;:\-'\"()]){2,}")
DUPLICATED_WORD_PATTERN = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
UPPER_WORD_PATTERN = re.compile(r"\b[A-Z]{2,}\b")
LOWER_WORD_PATTERN = re.compile(r"\b[a-z]{2,}\b")
EMOTICON_PATTERN = re.compile(
    r"[:;=8x][-^']?[)(dDpP]|[)(][:;=8]|\bD:", re.IGNORECASE
)
LAUGHTER_PATTERN = re.compile(r"\b(ja+j[aeiou]*|je+j[aeiou]*|ji+j[aeiou]*|lol|xd+|xdd+)\b", re.IGNORECASE)
RETWEET_PATTERN = re.compile(r"^\s*rt\s+@\w+", re.IGNORECASE)
VIA_PATTERN = re.compile(r"\bvia\s+@\w+", re.IGNORECASE)
INITIAL_MENTION_PATTERN = re.compile(r"^\s*(?:rt\s+)?@\w+")
# Vocativo tipico de Twitter: mencion inicial seguida de un nombre con inicial
# mayuscula ("@usuario Nombre, eres ...").
VOCATIVE_PATTERN = re.compile(
    r"^\s*(?:rt\s+)?@\w+[\s,:\-]+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,})"
)
SYMBOL_CHARS = set("$|*<>=+~^`[]{}%&")

# ---------------------------------------------------------------------------
# Lexicos internos: agresividad (inducida de train) y patrones morfologicos
# ---------------------------------------------------------------------------
# Formas imperativas frecuentes observadas en el corpus que los detectores
# morfologico y de clitico no cubren. Lista cerrada, aplicada igual a train y
# test, sin recursos externos (D-002).
IMPERATIVE_LEXICON: frozenset[str] = frozenset(
    {
        "calla", "callate", "callese", "callense", "callete", "vete", "vayase",
        "vayan", "vayanse", "muerete", "muere", "largate", "largese", "deja",
        "dejen", "dejame", "dejalo", "mira", "miren", "miralo", "oye", "oiga",
        "oigan", "escucha", "escuchen", "aprende", "aprendan", "habla", "hablen",
        "anda", "andate", "sigue", "sigan", "ven", "veni", "venga", "haz",
        "haga", "pon", "ponte", "pongase", "quita", "quitate", "basta", "cierra",
        "abre", "dime", "digame", "cuentame", "lee", "estudia", "trabaja",
        "trabaje", "respeta", "respeten", "piensa", "piense", "corre", "sal",
        "salte", "espera", "esperen", "aguanta", "tomate", "cojete",
        "chupame", "chupala", "matate", "ahorcate", "pide", "pida",
        "compra", "vende", "danos", "iros", "hacelo", "retirense", "marchate",
        "calmate", "callad", "mirad", "ponganse", "vamos", "huye", "corran",
        "escondete", "largo", "fuera",
    }
)
CLITIC_ENDINGS: tuple[str, ...] = (
    "me", "te", "se", "lo", "la", "le", "nos", "os", "los", "las", "les",
)
# Formas infinitivas con clitico (cogerse, cuidarte, decirle): se excluyen.
INFINITIVE_CLITIC_PATTERN = re.compile(r"r(me|te|se|lo|la|le|nos|os|los|las|les)$")
FIRST_PERSON_PLURAL_PATTERN = re.compile(r"(mos|amos|emos|imos)$")

# Pronombres y determinantes de segunda persona (tratamiento informal y formal).
SECOND_PERSON_LEXICON: frozenset[str] = frozenset(
    {
        "tu", "tus", "te", "ti", "usted", "ustedes", "vos", "vosotros",
        "vosotras", "contigo", "os", "vuestro", "vuestra", "vuestros",
        "vuestras", "tuyo", "tuya", "tuyos", "tuyas",
    }
)
FIRST_PERSON_SINGULAR: frozenset[str] = frozenset(
    {"yo", "me", "mi", "mis", "mio", "mia", "conmigo"}
)
FIRST_PERSON_PLURAL: frozenset[str] = frozenset(
    {"nosotros", "nosotras", "nos", "nuestro", "nuestra", "nuestros", "nuestras"}
)
THIRD_PERSON_SINGULAR: frozenset[str] = frozenset(
    {"el", "ella", "ello", "le", "lo", "la", "su", "sus", "suyo", "suya", "consigo", "se"}
)
THIRD_PERSON_PLURAL: frozenset[str] = frozenset(
    {"ellos", "ellas", "les", "los", "las", "suyos", "suyas"}
)
COLLECTIVE_QUANTIFIERS: frozenset[str] = frozenset(
    {
        "todos", "todas", "todo", "toda", "muchos", "muchas", "algunos",
        "algunas", "ningun", "ninguna", "ningunos", "ningunas", "cualquiera",
        "cualesquiera", "varios", "varias", "cada", "ambos", "ambas", "gente",
        "pueblo", "mayoria", "minoria", "multitud", "monton", "masa", "grupo",
    }
)
# Sufijos morfologicos tipicos de gentilicio (nacionalidad o procedencia).
DEMONYM_SUFFIXES: tuple[str, ...] = (
    "ano", "ana", "anos", "anas", "ense", "enses", "ino", "ina", "inos",
    "inas", "eno", "ena", "enos", "enas", "ol", "ola", "oles", "olas",
    "aco", "aca", "acos", "acas", "ita", "itas", "ense", "es", "esa",
    "esas", "eses", "ita", "anos",
)
INTERJECTIONS: frozenset[str] = frozenset(
    {"hola", "oye", "oigan", "eh", "hey", "ey", "joder", "conio"}
)


# ---------------------------------------------------------------------------
# Seccion 1 - Carga e integridad
# ---------------------------------------------------------------------------
def log(message: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        message: texto a imprimir.
    """
    print(f"[F2] {message}", flush=True)


def cargar_conjuntos() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga train, test y la tabla de grupos de near-duplicados.

    Returns:
        Tupla ``(train, test, grupos)``.

    Raises:
        FileNotFoundError: si la tabla de near-duplicados no existe.
    """
    train = data_module.read_train()
    test = data_module.read_test()
    if not GROUPS_PATH.exists():
        raise FileNotFoundError(
            f"Falta {GROUPS_PATH}; ejecute antes experiments/00_auditoria_datos.py"
        )
    grupos = pd.read_csv(GROUPS_PATH)
    return train, test, grupos


def verificar_integridad(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Verifica la integridad minima de los conjuntos oficiales.

    Args:
        train: conjunto de entrenamiento.
        test: conjunto de test.

    Returns:
        Diccionario con el informe de integridad.

    Raises:
        ValueError: si el numero de filas o la ausencia de nulos no se cumple.
    """
    informe = {
        "train_filas": len(train),
        "test_filas": len(test),
        "train_filas_esperadas": EXPECTED_TRAIN_ROWS,
        "test_filas_esperadas": EXPECTED_TEST_ROWS,
        "train_sin_nulos": bool(not train.isna().any().any()),
        "test_sin_nulos": bool(not test.isna().any().any()),
        "train_columnas": list(train.columns),
        "test_columnas": list(test.columns),
        "dominio_etiquetas": {
            column: sorted(train[column].unique().tolist()) for column in LABELS
        },
        "violaciones_tr_sin_hs": int(((train["TR"] == 1) & (train["HS"] == 0)).sum()),
        "ids_disjuntos": bool(not set(train["id"]) & set(test["id"])),
        "texto_no_vacio": bool((train["text"].str.strip() != "").all()),
    }
    if informe["train_filas"] != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"train con {informe['train_filas']} filas, se esperaban 1731")
    if informe["test_filas"] != EXPECTED_TEST_ROWS:
        raise ValueError(f"test con {informe['test_filas']} filas, se esperaban 1154")
    if not (informe["train_sin_nulos"] and informe["test_sin_nulos"]):
        raise ValueError("hay valores nulos en los archivos oficiales")
    return informe


def mascara_deduplicada(grupos: pd.DataFrame, ids: Sequence[int]) -> np.ndarray:
    """Construye la mascara que conserva una fila por grupo de near-duplicado.

    Args:
        grupos: tabla ``id, split, grupo, tamano_grupo``.
        ids: identificadores del conjunto a deduplicar, en orden.

    Returns:
        Arreglo booleano alineado con ``ids``: ``True`` en la primera fila de
        cada grupo.
    """
    mapa = grupos.set_index("id")["grupo"]
    vistos: set[int] = set()
    conservar = np.zeros(len(ids), dtype=bool)
    for posicion, identificador in enumerate(ids):
        clave = int(identificador)
        if clave not in mapa.index:
            conservar[posicion] = True
            continue
        grupo = int(mapa.loc[clave])
        if grupo not in vistos:
            vistos.add(grupo)
            conservar[posicion] = True
    return conservar


# ---------------------------------------------------------------------------
# Seccion 2 - EDA basico por etiqueta y por combinacion
# ---------------------------------------------------------------------------
def tabla_etiquetas(train: pd.DataFrame) -> pd.DataFrame:
    """Calcula la prevalencia marginal de cada etiqueta.

    Args:
        train: conjunto de entrenamiento.

    Returns:
        Tabla con positivos, negativos, prevalencia y desbalanceo.
    """
    filas = []
    for etiqueta in LABELS:
        positivos = int(train[etiqueta].sum())
        negativos = int(len(train) - positivos)
        filas.append(
            {
                "etiqueta": etiqueta,
                "positivos": positivos,
                "negativos": negativos,
                "prevalencia": round(positivos / len(train), 4),
                "desbalanceo_neg_pos": round(negativos / max(positivos, 1), 3),
            }
        )
    return pd.DataFrame(filas)


def tabla_combinaciones(train: pd.DataFrame) -> pd.DataFrame:
    """Calcula la frecuencia de cada combinacion HS x TR x AG.

    Args:
        train: conjunto de entrenamiento.

    Returns:
        Tabla ordenada por frecuencia descendente.
    """
    combinaciones = (
        train.groupby(list(LABELS), dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    combinaciones["porcentaje"] = (combinaciones["n"] / len(train) * 100).round(2)
    return combinaciones.reset_index(drop=True)


def metricas_longitud(textos: Iterable[str]) -> pd.DataFrame:
    """Calcula metricas de longitud por documento.

    Args:
        textos: iterable de textos.

    Returns:
        DataFrame con caracteres, palabras, tokens alfanumericos y longitud
        media de palabra.
    """
    registros = []
    for texto in textos:
        palabras = WORD_PATTERN.findall(texto)
        n_caracteres = len(texto)
        n_palabras = len(palabras)
        registros.append(
            {
                "n_caracteres": n_caracteres,
                "n_palabras": n_palabras,
                "n_palabras_unicas": len({palabra.lower() for palabra in palabras}),
                "longitud_media_palabra": (
                    float(np.mean([len(palabra) for palabra in palabras]))
                    if palabras
                    else 0.0
                ),
                "n_oraciones": len(re.findall(r"[.!?]+", texto)) + 1,
            }
        )
    marco = pd.DataFrame(registros)
    marco["ttr_documento"] = np.where(
        marco["n_palabras"] > 0,
        marco["n_palabras_unicas"] / marco["n_palabras"].replace(0, np.nan),
        np.nan,
    )
    return marco


def resumen_longitudes(train: pd.DataFrame, metricas: pd.DataFrame) -> pd.DataFrame:
    """Resume las metricas de longitud por etiqueta y por combinacion.

    Args:
        train: conjunto de entrenamiento.
        metricas: metricas por documento alineadas con ``train``.

    Returns:
        Tabla larga con estadisticos descriptivos por grupo.
    """
    datos = pd.concat([train.reset_index(drop=True), metricas], axis=1)
    variables = [
        "n_caracteres",
        "n_palabras",
        "n_palabras_unicas",
        "longitud_media_palabra",
        "n_oraciones",
        "ttr_documento",
    ]
    filas = []
    for etiqueta in LABELS:
        for valor in (0, 1):
            subconjunto = datos[datos[etiqueta] == valor]
            for variable in variables:
                serie = subconjunto[variable].dropna()
                filas.append(
                    {
                        "nivel": "etiqueta",
                        "grupo": f"{etiqueta}={valor}",
                        "n": len(serie),
                        "variable": variable,
                        "media": round(float(serie.mean()), 3),
                        "mediana": round(float(serie.median()), 3),
                        "desv": round(float(serie.std(ddof=1)), 3) if len(serie) > 1 else 0.0,
                        "p25": round(float(serie.quantile(0.25)), 3),
                        "p75": round(float(serie.quantile(0.75)), 3),
                        "min": round(float(serie.min()), 3),
                        "max": round(float(serie.max()), 3),
                    }
                )
    for combinacion, subconjunto in datos.groupby(list(LABELS)):
        etiqueta_combo = "-".join(str(valor) for valor in combinacion)
        for variable in variables:
            serie = subconjunto[variable].dropna()
            filas.append(
                {
                    "nivel": "combinacion",
                    "grupo": etiqueta_combo,
                    "n": len(serie),
                    "variable": variable,
                    "media": round(float(serie.mean()), 3),
                    "mediana": round(float(serie.median()), 3),
                    "desv": round(float(serie.std(ddof=1)), 3) if len(serie) > 1 else 0.0,
                    "p25": round(float(serie.quantile(0.25)), 3),
                    "p75": round(float(serie.quantile(0.75)), 3),
                    "min": round(float(serie.min()), 3),
                    "max": round(float(serie.max()), 3),
                }
            )
    return pd.DataFrame(filas)


def entropia_shannon(conteos: Sequence[int]) -> float:
    """Calcula la entropia de Shannon (base 2) de una distribucion de conteos.

    Args:
        conteos: frecuencias de cada tipo.

    Returns:
        Entropia en bits; 0.0 si la distribucion es degenerada.
    """
    valores = np.asarray([conteo for conteo in conteos if conteo > 0], dtype=float)
    if valores.size == 0:
        return 0.0
    probabilidades = valores / valores.sum()
    return float(-(probabilidades * np.log2(probabilidades)).sum())


def tabla_vocabulario(train: pd.DataFrame, lemas: Sequence[Sequence[str]]) -> pd.DataFrame:
    """Calcula riqueza lexica (vocabulario, TTR, hapax, entropia) por grupo.

    Args:
        train: conjunto de entrenamiento.
        lemas: lista de lemas por documento, alineada con ``train``.

    Returns:
        Tabla con las metricas de riqueza lexica por etiqueta y combinacion.
    """
    filas = []
    grupos: list[tuple[str, str, list[int]]] = []
    for etiqueta in LABELS:
        for valor in (0, 1):
            indices = train.index[train[etiqueta] == valor].tolist()
            grupos.append(("etiqueta", f"{etiqueta}={valor}", indices))
    for combinacion, subconjunto in train.groupby(list(LABELS)):
        etiqueta_combo = "-".join(str(valor) for valor in combinacion)
        grupos.append(("combinacion", etiqueta_combo, subconjunto.index.tolist()))

    for nivel, nombre, indices in grupos:
        conteo: Counter[str] = Counter()
        for indice in indices:
            conteo.update(lemas[indice])
        tokens = sum(conteo.values())
        tipos = len(conteo)
        hapax = sum(1 for frecuencia in conteo.values() if frecuencia == 1)
        dis = sum(1 for frecuencia in conteo.values() if frecuencia == 2)
        filas.append(
            {
                "nivel": nivel,
                "grupo": nombre,
                "n_documentos": len(indices),
                "tokens": tokens,
                "tipos": tipos,
                "ttr": round(tipos / tokens, 4) if tokens else 0.0,
                "hapax": hapax,
                "porcentaje_hapax": round(hapax / tipos * 100, 2) if tipos else 0.0,
                "dis": dis,
                "entropia_bits": round(entropia_shannon(list(conteo.values())), 4),
                "entropia_max_bits": round(np.log2(tipos), 4) if tipos > 1 else 0.0,
            }
        )
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Seccion 3 - Textometria de superficie
# ---------------------------------------------------------------------------
def contar_patron(texto: str, patron: re.Pattern[str]) -> int:
    """Cuenta las coincidencias de un patron en un texto.

    Args:
        texto: texto de entrada.
        patron: patron compilado.

    Returns:
        Numero de coincidencias.
    """
    return len(patron.findall(texto))


def extraer_fenomenos_superficie(textos: Iterable[str]) -> pd.DataFrame:
    """Extrae fenomenos de superficie por documento.

    Args:
        textos: iterable de textos.

    Returns:
        DataFrame con conteos, proporciones y banderas binarias.
    """
    registros = []
    for texto in textos:
        palabras = WORD_PATTERN.findall(texto)
        letras = [caracter for caracter in texto if caracter.isalpha()]
        mayusculas = sum(1 for caracter in letras if caracter.isupper())
        digitos = sum(1 for caracter in texto if caracter.isdigit())
        signos = [caracter for caracter in texto if caracter in string.punctuation]
        simbolos = sum(1 for caracter in texto if caracter in SYMBOL_CHARS)
        n_caracteres = len(texto)
        n_palabras = len(palabras)
        registro = {
            "n_caracteres": n_caracteres,
            "n_palabras": n_palabras,
            "n_menciones": contar_patron(texto, MENTION_PATTERN),
            "n_hashtags": contar_patron(texto, HASHTAG_PATTERN),
            "n_urls": contar_patron(texto, URL_PATTERN),
            "n_numeros": contar_patron(texto, NUMBER_PATTERN),
            "n_signos": len(signos),
            "n_simbolos": simbolos,
            "n_exclamaciones": texto.count("!"),
            "n_interrogaciones": texto.count("?"),
            "n_comillas": texto.count('"') + texto.count("'"),
            "n_dos_puntos": texto.count(":"),
            "n_puntos": texto.count("."),
            "n_punto_y_coma": texto.count(";"),
            "n_parentesis": texto.count("(") + texto.count(")"),
            "n_barras": texto.count("/"),
            "n_guiones": texto.count("-"),
            "n_emojis": emoji.emoji_count(texto),
            "n_emoticones": contar_patron(texto, EMOTICON_PATTERN),
            "n_risas": contar_patron(texto, LAUGHTER_PATTERN),
            "n_letras_repetidas": contar_patron(texto, LETTER_REPEAT_PATTERN),
            "n_puntuacion_repetida": contar_patron(texto, PUNCT_REPEAT_PATTERN),
            "n_palabras_duplicadas": contar_patron(texto, DUPLICATED_WORD_PATTERN),
            "n_palabras_mayusculas": contar_patron(texto, UPPER_WORD_PATTERN),
            "n_palabras_minusculas": contar_patron(texto, LOWER_WORD_PATTERN),
            "n_letras": len(letras),
            "n_mayusculas": mayusculas,
            "n_digitos": digitos,
            "n_espacios": texto.count(" "),
            "n_puntos_suspensivos": len(re.findall(r"\.\.\.", texto)),
        }
        registro["ratio_mayusculas"] = mayusculas / max(len(letras), 1)
        registro["ratio_digitos"] = digitos / max(n_caracteres, 1)
        registro["ratio_puntuacion"] = len(signos) / max(n_caracteres, 1)
        registro["ratio_simbolos"] = simbolos / max(n_caracteres, 1)
        registro["ratio_espacios"] = registro["n_espacios"] / max(n_caracteres, 1)
        registro["densidad_menciones_100"] = registro["n_menciones"] / max(n_caracteres, 1) * 100
        registro["densidad_hashtags_100"] = registro["n_hashtags"] / max(n_caracteres, 1) * 100
        registro["tiene_mencion"] = int(registro["n_menciones"] > 0)
        registro["tiene_mencion_inicial"] = int(bool(INITIAL_MENTION_PATTERN.match(texto)))
        registro["tiene_hashtag"] = int(registro["n_hashtags"] > 0)
        registro["tiene_url"] = int(registro["n_urls"] > 0)
        registro["tiene_numero"] = int(registro["n_numeros"] > 0)
        registro["tiene_simbolo"] = int(simbolos > 0)
        registro["tiene_emoji"] = int(registro["n_emojis"] > 0)
        registro["tiene_emoticon"] = int(registro["n_emoticones"] > 0)
        registro["tiene_risa"] = int(registro["n_risas"] > 0)
        registro["tiene_elongacion"] = int(registro["n_letras_repetidas"] > 0)
        registro["tiene_puntuacion_repetida"] = int(registro["n_puntuacion_repetida"] > 0)
        registro["tiene_palabra_duplicada"] = int(registro["n_palabras_duplicadas"] > 0)
        registro["tiene_mayusculas_sostenidas"] = int(
            registro["n_palabras_mayusculas"] > 0 or mayusculas >= 4
        )
        registro["es_mayoria_mayusculas"] = int(registro["ratio_mayusculas"] > 0.5)
        registro["tiene_exclamacion_multiple"] = int(
            bool(re.search(r"!{2,}", texto)) or bool(re.search(r"[!?]{2,}", texto))
        )
        registro["tiene_interrogacion_multiple"] = int(
            bool(re.search(r"\?{2,}", texto))
        )
        registro["tiene_comillas"] = int(registro["n_comillas"] > 0)
        registro["tiene_retweet"] = int(bool(RETWEET_PATTERN.match(texto)))
        registro["tiene_via"] = int(bool(VIA_PATTERN.search(texto)))
        registro["tiene_vocativo"] = int(
            bool(VOCATIVE_PATTERN.match(texto))
        )
        registro["tiene_puntos_suspensivos"] = int(registro["n_puntos_suspensivos"] > 0)
        registro["empieza_minuscula"] = int(texto[:1].islower())
        registros.append(registro)
    return pd.DataFrame(registros)


def palabras_frecuentes_corpus(textos: Iterable[str], minimo: int = 8) -> frozenset[str]:
    """Induce la lista de palabras de alta frecuencia de train.

    Se usa como filtro del detector heuristico de nombres de persona: una
    palabra que aparece en minuscula muchas veces en el corpus no puede ser un
    nombre propio de persona. La lista se induce solo de train.

    Args:
        textos: textos de train.
        minimo: frecuencia minima en minuscula para entrar en la lista.

    Returns:
        Conjunto de palabras frecuentes.
    """
    contador: Counter[str] = Counter()
    for texto in textos:
        contador.update(
            palabra.lower() for palabra in WORD_PATTERN.findall(texto) if palabra.isalpha()
        )
    return frozenset(
        palabra for palabra, frecuencia in contador.items() if frecuencia >= minimo
    )


def extraer_rasgos_linguisticos(
    nlp: object,
    textos: Sequence[str],
    palabras_frecuentes: frozenset[str] = frozenset(),
    batch_size: int = 64,
) -> pd.DataFrame:
    """Extrae rasgos morfosintacticos por documento con spaCy.

    Usa spaCy unicamente como anotador de POS, lemas y morfologia (D-003); no
    se accede a vectores ni a similitud. El NER se mantiene desactivado.

    Args:
        nlp: pipeline de spaCy.
        textos: textos a anotar.
        palabras_frecuentes: filtro de palabras frecuentes del corpus.
        batch_size: tamano de lote de ``nlp.pipe``.

    Returns:
        DataFrame con conteos de personas gramaticales, imperativos, nombres
        propios, patrones colectivos y vocativos heuristicos.
    """
    filas = []
    for numero_documento, documento in enumerate(
        nlp.pipe(textos, batch_size=batch_size), start=1
    ):
        pronombres = {
            "1s": 0, "1p": 0, "2": 0, "3s": 0, "3p": 0,
        }
        verbos = {"1s": 0, "1p": 0, "2": 0, "3s": 0, "3p": 0}
        imperativos_morph = 0
        imperativos_clitico = 0
        imperativos_lexico = 0
        nombres_propios: list[str] = []
        nombres_persona: list[str] = []
        colectivos: list[str] = []
        sustantivos_plural = 0
        propn_candidato = 0
        vocativos = 0
        risas_spacy = 0
        lemas: list[str] = []
        tokens = documento
        documento_inicia_con_mencion = bool(INITIAL_MENTION_PATTERN.match(documento.text))
        for token in tokens:
            texto_token = token.text
            minuscula = texto_token.lower()
            lema = token.lemma_.lower()
            lemas.append(lema)
            morfologia = token.morph
            persona = morfologia.get("Person")
            persona = persona[0] if persona else ""
            numero = morfologia.get("Number")
            numero = numero[0] if numero else ""
            numero_de = morfologia.get("Number")

            if token.pos_ == "PRON":
                if lema in SECOND_PERSON_LEXICON or minuscula in SECOND_PERSON_LEXICON:
                    pronombres["2"] += 1
                elif lema in FIRST_PERSON_SINGULAR or minuscula in FIRST_PERSON_SINGULAR:
                    pronombres["1s"] += 1
                elif lema in FIRST_PERSON_PLURAL or minuscula in FIRST_PERSON_PLURAL:
                    pronombres["1p"] += 1
                elif lema in THIRD_PERSON_PLURAL or minuscula in THIRD_PERSON_PLURAL:
                    pronombres["3p"] += 1
                elif lema in THIRD_PERSON_SINGULAR or minuscula in THIRD_PERSON_SINGULAR:
                    pronombres["3s"] += 1
            if token.pos_ == "DET" and minuscula in SECOND_PERSON_LEXICON:
                pronombres["2"] += 1

            if token.pos_ in ("VERB", "AUX"):
                if persona == "1":
                    verbos["1s" if numero != "Plur" else "1p"] += 1
                elif persona == "2":
                    verbos["2"] += 1
                elif persona == "3":
                    verbos["3s" if numero != "Plur" else "3p"] += 1
                if token.morph.get("Mood") == ["Imp"]:
                    imperativos_morph += 1
                if (
                    len(texto_token) >= 4
                    and texto_token.isalpha()
                    and texto_token.lower().endswith(CLITIC_ENDINGS)
                    and not INFINITIVE_CLITIC_PATTERN.search(texto_token.lower())
                    and not FIRST_PERSON_PLURAL_PATTERN.search(texto_token.lower())
                ):
                    imperativos_clitico += 1
                if minuscula in IMPERATIVE_LEXICON or lema in IMPERATIVE_LEXICON:
                    imperativos_lexico += 1

            if (
                token.is_alpha
                and len(texto_token) >= 3
                and texto_token[0].isupper()
                and (token.pos_ == "PROPN" or token.i > 0)
            ):
                nombres_propios.append(minuscula)
                if token.i > 0:
                    propn_candidato += 1

            # Detector heuristico de nombre de persona: capitalizacion inicial
            # con resto en minuscula, palabra no frecuente en minuscula y en un
            # contexto de vocativo o de mencion (D-003 no autoriza NER).
            es_candidato_propio = (
                token.i > 0
                and token.pos_ in ("PROPN", "NOUN")
                and token.is_alpha
                and len(texto_token) >= 3
                and texto_token[0].isupper()
                and texto_token[1:].islower()
                and minuscula not in palabras_frecuentes
            )
            if es_candidato_propio and (
                tokens[token.i - 1].text.startswith("@")
                or (documento_inicia_con_mencion and token.i <= 5)
            ):
                nombres_persona.append(minuscula)

            if token.pos_ in ("NOUN", "PROPN", "ADJ") and numero_de == ["Plur"]:
                sustantivos_plural += 1
                anterior = tokens[token.i - 1] if token.i > 0 else None
                if (
                    anterior is not None
                    and anterior.pos_ == "DET"
                    and anterior.morph.get("Number") == ["Plur"]
                ):
                    colectivos.append(lema)

            if token.pos_ in ("NOUN", "PROPN") and lema in COLLECTIVE_QUANTIFIERS:
                colectivos.append(lema)

            if (
                token.is_alpha
                and minuscula in INTERJECTIONS
                and token.i == 0
                and token.i + 1 < len(tokens)
                and tokens[token.i + 1].pos_ in ("NOUN", "PROPN")
            ):
                vocativos += 1

            if LAUGHTER_PATTERN.match(minuscula):
                risas_spacy += 1

        if numero_documento % 500 == 0:
            log(f"spaCy: {numero_documento} documentos anotados")

        filas.append(
            {
                "pron_1s": pronombres["1s"],
                "pron_1p": pronombres["1p"],
                "pron_2": pronombres["2"],
                "pron_3s": pronombres["3s"],
                "pron_3p": pronombres["3p"],
                "verbo_1s": verbos["1s"],
                "verbo_1p": verbos["1p"],
                "verbo_2": verbos["2"],
                "verbo_3s": verbos["3s"],
                "verbo_3p": verbos["3p"],
                "imperativo_morph": imperativos_morph,
                "imperativo_clitico": imperativos_clitico,
                "imperativo_lexico": imperativos_lexico,
                "n_propn_candidato": propn_candidato,
                "n_nombres_persona": len(nombres_persona),
                "n_sustantivos_plural": sustantivos_plural,
                "n_patron_colectivo": len(colectivos),
                "n_vocativo_interjeccion": vocativos,
                "n_risas_spacy": risas_spacy,
                "lemas_documento": lemas,
                "colectivos_documento": colectivos,
                "propn_documento": nombres_propios,
                "nombres_persona_documento": nombres_persona,
            }
        )
    return pd.DataFrame(filas)


def anadir_rasgos_binarios(marco: pd.DataFrame) -> pd.DataFrame:
    """Agrega banderas binarias derivadas de los rasgos morfosintacticos.

    Args:
        marco: marco con las columnas de superficie y de
            :func:`extraer_rasgos_linguisticos`.

    Returns:
        Copia del marco con nuevas columnas binarias.
    """
    datos = marco.copy()
    datos["imperativo"] = (
        (datos["imperativo_morph"] > 0)
        | (datos["imperativo_clitico"] > 0)
        | (datos["imperativo_lexico"] > 0)
    ).astype(int)
    datos["tiene_segunda_persona"] = (
        (datos["pron_2"] > 0) | (datos["verbo_2"] > 0) | (datos["imperativo"] > 0)
    ).astype(int)
    datos["tiene_primera_persona_singular"] = (
        (datos["pron_1s"] > 0) | (datos["verbo_1s"] > 0)
    ).astype(int)
    datos["tiene_primera_persona_plural"] = (
        (datos["pron_1p"] > 0) | (datos["verbo_1p"] > 0)
    ).astype(int)
    datos["tiene_tercera_persona_singular"] = (
        (datos["pron_3s"] > 0) | (datos["verbo_3s"] > 0)
    ).astype(int)
    datos["tiene_tercera_persona_plural"] = (
        (datos["pron_3p"] > 0) | (datos["verbo_3p"] > 0)
    ).astype(int)
    # Referencia colectiva puramente estructural: determinante plural mas
    # sustantivo o adjetivo en plural, o cuantificador colectivo. No usa
    # ninguna lista externa ni semantica importada.
    datos["tiene_referencia_colectiva"] = (datos["n_patron_colectivo"] > 0).astype(int)
    datos["tiene_vocativo"] = (
        (datos["tiene_vocativo"] == 1) | (datos["n_vocativo_interjeccion"] > 0)
    ).astype(int)
    return datos


def integrar_columnas(destino: pd.DataFrame, aumentado: pd.DataFrame) -> None:
    """Copia en ``destino`` las columnas nuevas o modificadas de ``aumentado``.

    Las funciones de rasgos devuelven un marco con columnas adicionales (y a
    veces re-derivan columnas existentes), por lo que no se puede asignar el
    resultado directamente a la seleccion original de columnas.

    Args:
        destino: marco a actualizar in place.
        aumentado: marco devuelto por una funcion de rasgos.
    """
    for columna in aumentado.columns:
        serie = aumentado[columna]
        if columna not in destino.columns or not destino[columna].equals(serie):
            destino[columna] = serie


def inducir_lexico_agresivo(
    train: pd.DataFrame, lemas: Sequence[Sequence[str]], top_k: int = 150
) -> pd.DataFrame:
    """Induce un lexico de agresividad desde train (sin recursos externos).

    Metodologia: log-odds con priori de Dirichlet informativo (Monroe et al.
    2008) entre documentos AG=1 y AG=0, restringido a lemas alfabeticos con
    frecuencia total >= 3 y presente en al menos 2 documentos con AG=1. El
    lexico se induce solo con train; su uso posterior sobre train es
    intrinsecamente in-sample y asi se reporta.

    Args:
        train: conjunto de entrenamiento.
        lemas: lista de lemas por documento alineada con ``train``.
        top_k: numero de lemas del lexico.

    Returns:
        Tabla con el lema, su frecuencia en cada clase, el log-odds y el rango.
    """
    conteo_positivo: Counter[str] = Counter()
    conteo_negativo: Counter[str] = Counter()
    documentos_positivos: Counter[str] = Counter()
    for indice, fila in enumerate(train.itertuples()):
        conteo = Counter(lemas[indice])
        if fila.AG == 1:
            conteo_positivo.update(conteo)
            documentos_positivos.update(set(conteo))
        else:
            conteo_negativo.update(conteo)
    total_positivo = sum(conteo_positivo.values())
    total_negativo = sum(conteo_negativo.values())
    vocabulario = set(conteo_positivo) | set(conteo_negativo)
    filas = []
    alpha_prior = 0.01
    for lema in sorted(vocabulario):
        if not lema.isalpha() or len(lema) < 3:
            continue
        frecuencia_positiva = conteo_positivo.get(lema, 0)
        frecuencia_negativa = conteo_negativo.get(lema, 0)
        frecuencia_total = frecuencia_positiva + frecuencia_negativa
        if frecuencia_total < 3 or documentos_positivos.get(lema, 0) < 2:
            continue
        proporcion_positiva = (frecuencia_positiva + alpha_prior) / (
            total_positivo + alpha_prior * len(vocabulario)
        )
        proporcion_negativa = (frecuencia_negativa + alpha_prior) / (
            total_negativo + alpha_prior * len(vocabulario)
        )
        log_odds = np.log(proporcion_positiva / proporcion_negativa)
        varianza = 1.0 / (frecuencia_positiva + alpha_prior) + 1.0 / (
            frecuencia_negativa + alpha_prior
        )
        filas.append(
            {
                "lema": lema,
                "frecuencia_AG1": frecuencia_positiva,
                "frecuencia_AG0": frecuencia_negativa,
                "frecuencia_total": frecuencia_total,
                "documentos_AG1": documentos_positivos.get(lema, 0),
                "log_odds": round(float(log_odds), 4),
                "z_score": round(float(log_odds / np.sqrt(varianza)), 3),
            }
        )
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return tabla
    tabla = tabla.sort_values("z_score", ascending=False).reset_index(drop=True)
    tabla["rango"] = np.arange(1, len(tabla) + 1)
    tabla["seleccionado"] = (tabla["rango"] <= top_k).astype(int)
    return tabla


def inducir_lexico_colectivos(rasgos: pd.DataFrame, min_documentos: int = 3) -> pd.DataFrame:
    """Inventaria los lemas usados como objetivo colectivo en train.

    Criterio estructural puro (sin semantica externa): lemas de sustantivo o
    adjetivo que aparecen en plural precedidos de un determinante plural
    ("los inmigrantes", "las mujeres"), o lemas cuantificadores colectivos. La
    inspeccion humana decide cuales son objetivos grupales reales.

    Args:
        rasgos: marco de rasgos linguisticos de train.
        min_documentos: numero minimo de documentos para conservar el lema.

    Returns:
        Tabla con el lema, su numero de documentos y su frecuencia total.
    """
    contador: Counter[str] = Counter()
    for lista in rasgos["colectivos_documento"]:
        contador.update(set(lista))
    filas = [
        {"lema": lema, "documentos": documentos, "categoria": "patron_colectivo"}
        for lema, documentos in sorted(contador.items())
        if documentos >= min_documentos
    ]
    tabla = pd.DataFrame(filas).sort_values(["documentos", "lema"], ascending=[False, True])
    return tabla.reset_index(drop=True)


def anadir_rasgos_lexicos(
    marco: pd.DataFrame,
    lexico_agresivo: frozenset[str],
    lexico_colectivo: frozenset[str],
) -> pd.DataFrame:
    """Agrega rasgos derivados de los lexicos inducidos.

    Args:
        marco: marco de rasgos linguisticos.
        lexico_agresivo: lemas del lexico de agresividad inducido.
        lexico_colectivo: lemas del inventario de colectivos inducido.

    Returns:
        Copia del marco con las columnas de insulto, colectivo y nombre propio.
    """
    datos = marco.copy()

    def cuenta_lexico(lista: Sequence[str], lexico: frozenset[str]) -> int:
        """Cuenta cuantos lemas de la lista pertenecen al lexico dado."""
        return sum(1 for lema in lista if lema in lexico)

    datos["n_insultos"] = [
        cuenta_lexico(lista, lexico_agresivo) for lista in datos["lemas_documento"]
    ]
    datos["tiene_insulto"] = (datos["n_insultos"] > 0).astype(int)
    datos["tiene_insulto_dirigido"] = (
        (datos["n_insultos"] > 0) & (datos["tiene_segunda_persona"] == 1)
    ).astype(int)
    # Variante secundaria exploratoria: usa el inventario de colectivos inducido
    # de train (mas laxa que el patron estructural, se reporta por separado).
    datos["n_colectivos_lexico"] = [
        cuenta_lexico(lista, lexico_colectivo) for lista in datos["lemas_documento"]
    ]
    datos["tiene_colectivo_lexico"] = (datos["n_colectivos_lexico"] > 0).astype(int)
    datos["tiene_nombre_propio"] = (datos["n_nombres_persona"] > 0).astype(int)
    datos["tiene_referencia_individual"] = (
        (datos["tiene_segunda_persona"] == 1) | (datos["tiene_nombre_propio"] == 1)
    ).astype(int)
    return datos


def ngramas_puntuacion(textos: Iterable[str], n: int = 2, top_k: int = 30) -> pd.DataFrame:
    """Cuenta los n-gramas de signos de puntuacion mas frecuentes.

    Args:
        textos: iterable de textos.
        n: tamano del n-grama.
        top_k: numero de n-gramas a conservar.

    Returns:
        Tabla con el n-grama, su frecuencia y su porcentaje.
    """
    contador: Counter[str] = Counter()
    for texto in textos:
        signos = [caracter for caracter in texto if caracter in string.punctuation]
        for posicion in range(len(signos) - n + 1):
            contador["".join(signos[posicion : posicion + n])] += 1
    total = sum(contador.values())
    filas = [
        {
            "ngrama": ngrama,
            "n": frecuencia,
            "porcentaje": round(frecuencia / total * 100, 3) if total else 0.0,
        }
        for ngrama, frecuencia in contador.most_common(top_k)
    ]
    return pd.DataFrame(filas)


def inventario_caracteres(textos: Iterable[str]) -> pd.DataFrame:
    """Inventaria los caracteres no alfanumericos presentes en el corpus.

    Args:
        textos: iterable de textos.

    Returns:
        Tabla con el caracter, su frecuencia y si es ASCII.
    """
    contador: Counter[str] = Counter()
    for texto in textos:
        contador.update(caracter for caracter in texto if not caracter.isalnum() and caracter != " ")
    total = sum(contador.values())
    filas = [
        {
            "caracter": caracter,
            "n": frecuencia,
            "porcentaje": round(frecuencia / total * 100, 4) if total else 0.0,
            "es_ascii": int(ord(caracter) < 128),
        }
        for caracter, frecuencia in contador.most_common()
    ]
    return pd.DataFrame(filas)


def tabla_textometria_por_clase(fenomenos: pd.DataFrame) -> pd.DataFrame:
    """Resume la textometria por clase de cada etiqueta.

    Args:
        fenomenos: marco con etiquetas y rasgos numericos.

    Returns:
        Tabla con media y desviacion por etiqueta y valor de clase.
    """
    variables = [
        "n_caracteres", "n_palabras", "n_menciones", "n_hashtags", "n_urls",
        "n_signos", "n_exclamaciones", "ratio_mayusculas", "ratio_puntuacion",
        "ratio_digitos", "n_emojis", "n_emoticones", "n_risas",
        "n_letras_repetidas", "n_puntuacion_repetida", "n_palabras_duplicadas",
        "n_palabras_mayusculas", "imperativo", "tiene_segunda_persona",
        "n_insultos", "n_patron_colectivo", "n_propn_candidato",
    ]
    filas = []
    for etiqueta in LABELS:
        for valor in (0, 1):
            subconjunto = fenomenos[fenomenos[etiqueta] == valor]
            for variable in variables:
                serie = subconjunto[variable].astype(float)
                filas.append(
                    {
                        "etiqueta": etiqueta,
                        "clase": valor,
                        "n": len(serie),
                        "variable": variable,
                        "media": round(float(serie.mean()), 3),
                        "mediana": round(float(serie.median()), 3),
                        "desv": round(float(serie.std(ddof=1)), 3) if len(serie) > 1 else 0.0,
                        "p90": round(float(serie.quantile(0.90)), 3),
                    }
                )
    for combinacion, subconjunto in fenomenos.groupby(list(LABELS)):
        etiqueta_combo = "-".join(str(valor) for valor in combinacion)
        for variable in variables:
            serie = subconjunto[variable].astype(float)
            filas.append(
                {
                    "etiqueta": f"combo_{etiqueta_combo}",
                    "clase": -1,
                    "n": len(serie),
                    "variable": variable,
                    "media": round(float(serie.mean()), 3),
                    "mediana": round(float(serie.median()), 3),
                    "desv": round(float(serie.std(ddof=1)), 3) if len(serie) > 1 else 0.0,
                    "p90": round(float(serie.quantile(0.90)), 3),
                }
            )
    return pd.DataFrame(filas)


def top_palabras_por_clase(
    train: pd.DataFrame, lemas: Sequence[Sequence[str]], top_k: int = 20
) -> pd.DataFrame:
    """Calcula las palabras mas frecuentes de cada clase.

    Args:
        train: conjunto de entrenamiento.
        lemas: lista de lemas por documento alineada con ``train``.
        top_k: numero de palabras por clase.

    Returns:
        Tabla con la palabra, su frecuencia, el grupo y el cociente de
        frecuencias respecto del grupo complementario.
    """
    conteos: dict[str, Counter[str]] = {}
    for etiqueta in LABELS:
        for valor in (0, 1):
            contador: Counter[str] = Counter()
            for indice, fila in enumerate(train.itertuples()):
                if getattr(fila, etiqueta) == valor:
                    contador.update(
                        lema
                        for lema in lemas[indice]
                        if lema not in STOP_WORDS and len(lema) > 2 and lema.isalpha()
                    )
            conteos[f"{etiqueta}={valor}"] = contador
    filas = []
    for etiqueta in LABELS:
        for valor in (0, 1):
            grupo = f"{etiqueta}={valor}"
            complemento = f"{etiqueta}={1 - valor}"
            total = sum(conteos[grupo].values()) or 1
            total_complemento = sum(conteos[complemento].values()) or 1
            for rango, (palabra, frecuencia) in enumerate(conteos[grupo].most_common(top_k), start=1):
                tasa = frecuencia / total
                tasa_complemento = conteos[complemento].get(palabra, 0) / total_complemento
                filas.append(
                    {
                        "etiqueta": etiqueta,
                        "clase": valor,
                        "rango": rango,
                        "palabra": palabra,
                        "frecuencia": frecuencia,
                        "frecuencia_relativa_mil": round(tasa * 1000, 3),
                        "cociente_frecuencias": round(tasa / tasa_complemento, 3)
                        if tasa_complemento > 0
                        else float("inf"),
                    }
                )
    return pd.DataFrame(filas)


def formato_largo(fenomenos: pd.DataFrame, variables: Sequence[str]) -> pd.DataFrame:
    """Convierte el marco ancho de fenomenos en formato largo por etiqueta.

    Args:
        fenomenos: marco con las tres etiquetas y las variables.
        variables: variables a incluir.

    Returns:
        DataFrame con columnas ``etiqueta``, ``clase``, ``variable`` y ``valor``.
    """
    piezas = []
    for etiqueta in LABELS:
        for variable in variables:
            piezas.append(
                pd.DataFrame(
                    {
                        "etiqueta": etiqueta,
                        "clase": fenomenos[etiqueta].to_numpy(),
                        "variable": variable,
                        "valor": fenomenos[variable].astype(float).to_numpy(),
                    }
                )
            )
    return pd.concat(piezas, ignore_index=True)


# ---------------------------------------------------------------------------
# Seccion 4 - Estadistica con tamano de efecto
# ---------------------------------------------------------------------------
def wilson_interval(positivos: int, total: int, z: float = 1.959963985) -> tuple[float, float]:
    """Calcula el intervalo de confianza de Wilson para una proporcion.

    Args:
        positivos: numero de exitos.
        total: numero de ensayos.
        z: cuantil normal para el nivel de confianza (1.96 para 95 %).

    Returns:
        Tupla ``(limite_inferior, limite_superior)``.
    """
    if total <= 0:
        return float("nan"), float("nan")
    proporcion = positivos / total
    denominador = 1 + z**2 / total
    centro = (proporcion + z**2 / (2 * total)) / denominador
    margen = (
        z * np.sqrt(proporcion * (1 - proporcion) / total + z**2 / (4 * total**2)) / denominador
    )
    return float(centro - margen), float(centro + margen)


def newcombe_difference_interval(
    positivos_1: int, total_1: int, positivos_0: int, total_0: int
) -> tuple[float, float]:
    """Intervalo de confianza de Newcombe para la diferencia de proporciones.

    Args:
        positivos_1: positivos del grupo 1.
        total_1: tamano del grupo 1.
        positivos_0: positivos del grupo 0.
        total_0: tamano del grupo 0.

    Returns:
        Tupla ``(limite_inferior, limite_superior)`` de la diferencia p1 - p0.
    """
    p1 = positivos_1 / total_1 if total_1 else float("nan")
    p0 = positivos_0 / total_0 if total_0 else float("nan")
    l1, u1 = wilson_interval(positivos_1, total_1)
    l0, u0 = wilson_interval(positivos_0, total_0)
    diferencia = p1 - p0
    inferior = diferencia - np.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    superior = diferencia + np.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return float(inferior), float(superior)


def cramers_v(tabla: np.ndarray) -> float:
    """V de Cramer con correccion de sesgo (Bergsma 2013).

    Args:
        tabla: tabla de contingencia.

    Returns:
        V de Cramer corregida, o ``nan`` si la tabla es degenerada.
    """
    matriz = np.asarray(tabla, dtype=float)
    if matriz.ndim != 2 or min(matriz.shape) < 2:
        return float("nan")
    total = matriz.sum()
    if total <= 0:
        return float("nan")
    chi2, _, _, _ = chi2_contingency(matriz, correction=False)
    phi2 = chi2 / total
    filas, columnas = matriz.shape
    phi2_corregida = max(0.0, phi2 - ((columnas - 1) * (filas - 1)) / (total - 1))
    filas_corregidas = filas - (filas - 1) ** 2 / (total - 1)
    columnas_corregidas = columnas - (columnas - 1) ** 2 / (total - 1)
    denominador = min(columnas_corregidas - 1, filas_corregidas - 1)
    if denominador <= 0:
        return float("nan")
    return float(np.sqrt(phi2_corregida / denominador))


def cliff_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Delta de Cliff entre dos muestras independientes.

    Args:
        x: valores del grupo 1.
        y: valores del grupo 0.

    Returns:
        Delta de Cliff en ``[-1, 1]``; positivo indica que ``x`` tiende a ser
        mayor. Devuelve ``nan`` si alguna muestra queda vacia.
    """
    valores_x = np.asarray([valor for valor in x if not pd.isna(valor)], dtype=float)
    valores_y = np.asarray([valor for valor in y if not pd.isna(valor)], dtype=float)
    if valores_x.size == 0 or valores_y.size == 0:
        return float("nan")
    estadistico = mannwhitneyu(valores_x, valores_y, alternative="two-sided").statistic
    return float(2 * estadistico / (valores_x.size * valores_y.size) - 1)


def kruskal_epsilon_cuadrado(estadistico: float, n: int, k: int) -> float:
    """Epsilon cuadrado de Kruskal-Wallis.

    Args:
        estadistico: estadistico H de Kruskal-Wallis.
        n: numero total de observaciones.
        k: numero de grupos.

    Returns:
        Tamano de efecto epsilon cuadrado en ``[0, 1]``.
    """
    if n <= k:
        return float("nan")
    return float((estadistico - k + 1) / (n - k))


def benjamini_hochberg(p_valores: Sequence[float]) -> np.ndarray:
    """Correccion de Benjamini-Hochberg para comparaciones multiples.

    Args:
        p_valores: valores p crudos.

    Returns:
        Arreglo de q-valores con la misma forma que la entrada.
    """
    valores = np.asarray(p_valores, dtype=float)
    resultado = np.full(valores.shape, np.nan)
    validos = ~np.isnan(valores)
    subconjunto = valores[validos]
    if subconjunto.size == 0:
        return resultado
    n = subconjunto.size
    orden = np.argsort(subconjunto)
    acumulado = np.minimum.accumulate((subconjunto[orden] * n / np.arange(1, n + 1))[::-1])[::-1]
    q_valores = np.empty(n)
    q_valores[orden] = np.clip(acumulado, 0.0, 1.0)
    resultado[validos] = q_valores
    return resultado


def magnitud_efecto(valor: float, tipo: str) -> str:
    """Clasifica la magnitud de un tamano de efecto.

    Args:
        valor: tamano de efecto.
        tipo: ``"v"`` (V de Cramer), ``"cliff"`` (delta de Cliff) o ``"epsilon"``.

    Returns:
        Etiqueta cualitativa de la magnitud.
    """
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "no_calculable"
    absoluto = abs(float(valor))
    umbrales = {"v": (0.10, 0.30, 0.50), "cliff": (0.147, 0.33, 0.474), "epsilon": (0.01, 0.06, 0.14)}
    minimo, medio, fuerte = umbrales[tipo]
    if absoluto < minimo:
        return "despreciable"
    if absoluto < medio:
        return "pequeno"
    if absoluto < fuerte:
        return "medio"
    return "grande"


def pruebas_categoricas(
    fenomenos: pd.DataFrame,
    etiqueta: str,
    columnas: Sequence[str],
    conjunto: str,
) -> list[dict]:
    """Contrasta fenomenos binarios contra una etiqueta con chi cuadrado.

    Usa la prueba exacta de Fisher cuando alguna frecuencia esperada es menor
    que 5. Reporta V de Cramer con correccion de sesgo y la diferencia de
    proporciones con intervalo de Newcombe.

    Args:
        fenomenos: marco con etiqueta y fenomenos binarios.
        etiqueta: nombre de la etiqueta a predecir.
        columnas: fenomenos binarios a contrastar.
        conjunto: etiqueta del conjunto usado ("completo" o "deduplicado").

    Returns:
        Lista de diccionarios con un resultado por fenomeno.
    """
    resultados = []
    for columna in columnas:
        presencia = fenomenos[fenomenos[columna] == 1][etiqueta]
        ausencia = fenomenos[fenomenos[columna] == 0][etiqueta]
        a = int(presencia.sum())
        b = int(len(presencia) - a)
        c = int(ausencia.sum())
        d = int(len(ausencia) - c)
        tabla = np.array([[a, b], [c, d]])
        if tabla.sum() == 0 or len(presencia) == 0 or len(ausencia) == 0:
            continue
        chi2, p_chi2, gl, esperado = chi2_contingency(tabla, correction=False)
        usa_fisher = bool((esperado < MIN_EXPECTED_CELL).any())
        if usa_fisher:
            _, p_valor = fisher_exact(tabla)
        else:
            p_valor = p_chi2
        proporcion_presencia = a / len(presencia) if len(presencia) else float("nan")
        proporcion_ausencia = c / len(ausencia) if len(ausencia) else float("nan")
        inferior, superior = newcombe_difference_interval(a, len(presencia), c, len(ausencia))
        resultados.append(
            {
                "conjunto": conjunto,
                "etiqueta": etiqueta,
                "fenomeno": columna,
                "n_con_fenomeno": len(presencia),
                "n_sin_fenomeno": len(ausencia),
                "p_etiqueta_con_fenomeno": round(proporcion_presencia, 4),
                "p_etiqueta_sin_fenomeno": round(proporcion_ausencia, 4),
                "diferencia_proporciones": round(proporcion_presencia - proporcion_ausencia, 4),
                "ic95_inferior": round(inferior, 4),
                "ic95_superior": round(superior, 4),
                "chi2": round(float(chi2), 3),
                "gl": int(gl),
                "prueba": "fisher" if usa_fisher else "chi2",
                "p_valor": float(p_valor),
                "v_cramer": round(cramers_v(tabla), 4),
                "magnitud": magnitud_efecto(cramers_v(tabla), "v"),
                "celdas": f"a={a};b={b};c={c};d={d}",
            }
        )
    return resultados


def pruebas_continuas(
    fenomenos: pd.DataFrame,
    etiqueta: str,
    columnas: Sequence[str],
    conjunto: str,
) -> list[dict]:
    """Contrasta variables continuas contra una etiqueta binaria.

    Mann-Whitney U bilateral y delta de Cliff.

    Args:
        fenomenos: marco con etiqueta y variables continuas.
        etiqueta: nombre de la etiqueta.
        columnas: variables a contrastar.
        conjunto: etiqueta del conjunto usado.

    Returns:
        Lista de diccionarios con un resultado por variable.
    """
    resultados = []
    for columna in columnas:
        positivos = fenomenos.loc[fenomenos[etiqueta] == 1, columna].astype(float).dropna()
        negativos = fenomenos.loc[fenomenos[etiqueta] == 0, columna].astype(float).dropna()
        if positivos.empty or negativos.empty:
            continue
        estadistico, p_valor = mannwhitneyu(positivos, negativos, alternative="two-sided")
        delta = cliff_delta(positivos, negativos)
        resultados.append(
            {
                "conjunto": conjunto,
                "etiqueta": etiqueta,
                "variable": columna,
                "n_positivos": int(positivos.size),
                "n_negativos": int(negativos.size),
                "mediana_positivos": round(float(positivos.median()), 3),
                "mediana_negativos": round(float(negativos.median()), 3),
                "media_positivos": round(float(positivos.mean()), 3),
                "media_negativos": round(float(negativos.mean()), 3),
                "u_mannwhitney": float(estadistico),
                "p_valor": float(p_valor),
                "delta_cliff": round(delta, 4),
                "magnitud": magnitud_efecto(delta, "cliff"),
                "direccion": "mayor_en_positivos" if delta > 0 else "mayor_en_negativos",
            }
        )
    return resultados


def pruebas_por_combinacion(
    fenomenos: pd.DataFrame, columnas: Sequence[str], conjunto: str
) -> list[dict]:
    """Contrasta variables continuas entre las seis combinaciones de etiquetas.

    Kruskal-Wallis y epsilon cuadrado.

    Args:
        fenomenos: marco con las tres etiquetas y variables continuas.
        columnas: variables a contrastar.
        conjunto: etiqueta del conjunto usado.

    Returns:
        Lista de diccionarios con un resultado por variable.
    """
    resultados = []
    etiquetas_combo = list(map(tuple, fenomenos[list(LABELS)].drop_duplicates().to_numpy()))
    for columna in columnas:
        grupos = [
            fenomenos.loc[
                (fenomenos["HS"] == combo[0])
                & (fenomenos["TR"] == combo[1])
                & (fenomenos["AG"] == combo[2]),
                columna,
            ]
            .astype(float)
            .dropna()
            for combo in etiquetas_combo
        ]
        grupos = [grupo for grupo in grupos if grupo.size > 0]
        if len(grupos) < 3:
            continue
        estadistico, p_valor = kruskal(*grupos)
        n = int(sum(grupo.size for grupo in grupos))
        epsilon = kruskal_epsilon_cuadrado(float(estadistico), n, len(grupos))
        resultados.append(
            {
                "conjunto": conjunto,
                "etiqueta": "combinacion_HSxTRxAG",
                "variable": columna,
                "n_grupos": len(grupos),
                "n_total": n,
                "h_kruskal": round(float(estadistico), 3),
                "p_valor": float(p_valor),
                "epsilon_cuadrado": round(epsilon, 4),
                "magnitud": magnitud_efecto(epsilon, "epsilon"),
            }
        )
    return resultados


def tabla_info_mutua(
    fenomenos: pd.DataFrame, columnas: Sequence[str], conjunto: str
) -> list[dict]:
    """Calcula informacion mutua y ajustada entre cada fenomeno y cada etiqueta.

    Args:
        fenomenos: marco con etiquetas y fenomenos discretos.
        columnas: fenomenos a evaluar.
        conjunto: etiqueta del conjunto usado.

    Returns:
        Lista de diccionarios con MI, AMI y MI normalizada.
    """
    resultados = []
    for etiqueta in LABELS:
        valores_etiqueta = fenomenos[etiqueta].to_numpy()
        for columna in columnas:
            valores = fenomenos[columna].to_numpy()
            if len(np.unique(valores)) < 2:
                continue
            mi = float(mutual_info_score(valores_etiqueta, valores))
            ami = float(adjusted_mutual_info_score(valores_etiqueta, valores))
            _, conteos = np.unique(valores_etiqueta, return_counts=True)
            entropia = entropia_shannon(conteos.tolist())
            resultados.append(
                {
                    "conjunto": conjunto,
                    "etiqueta": etiqueta,
                    "fenomeno": columna,
                    "info_mutua": round(mi, 5),
                    "info_mutua_ajustada": round(ami, 5),
                    "mi_normalizada": round(mi / entropia, 5) if entropia > 0 else float("nan"),
                }
            )
    return resultados


def correlaciones_etiquetas(train: pd.DataFrame) -> pd.DataFrame:
    """Calcula asociacion entre etiquetas y con las longitudes.

    Args:
        train: conjunto de entrenamiento.

    Returns:
        Tabla larga con el coeficiente phi (Pearson sobre binarias), p-valor,
        V de Cramer y el soporte de la tabla 2x2.
    """
    variables = list(LABELS) + ["n_caracteres", "n_palabras"]
    datos = train.copy()
    datos["n_caracteres"] = datos["text"].str.len()
    datos["n_palabras"] = datos["text"].map(lambda texto: len(WORD_PATTERN.findall(texto)))
    filas = []
    for indice_a, variable_a in enumerate(variables):
        for variable_b in variables[indice_a + 1 :]:
            valores_a = datos[variable_a].astype(float)
            valores_b = datos[variable_b].astype(float)
            if valores_a.nunique() == 2 and valores_b.nunique() == 2:
                tabla = pd.crosstab(valores_a, valores_b).reindex(
                    index=[0, 1], columns=[0, 1], fill_value=0
                )
                phi = float(np.corrcoef(valores_a, valores_b)[0, 1])
                _chi2, p_valor, _, _ = chi2_contingency(tabla.to_numpy(), correction=False)
                filas.append(
                    {
                        "variable_a": variable_a,
                        "variable_b": variable_b,
                        "tipo": "binaria",
                        "phi": round(phi, 4),
                        "v_cramer": round(cramers_v(tabla.to_numpy()), 4),
                        "spearman": float("nan"),
                        "p_valor": float(p_valor),
                        "n": len(datos),
                        "coocurrencia_a1_b1": int(tabla.loc[1, 1]),
                    }
                )
            else:
                correlacion, p_valor = spearmanr(valores_a, valores_b)
                filas.append(
                    {
                        "variable_a": variable_a,
                        "variable_b": variable_b,
                        "tipo": "continua",
                        "phi": float("nan"),
                        "v_cramer": float("nan"),
                        "spearman": round(float(correlacion), 4),
                        "p_valor": float(p_valor),
                        "n": len(datos),
                        "coocurrencia_a1_b1": int(((valores_a == 1) & (valores_b == 1)).sum()),
                    }
                )
    tabla = pd.DataFrame(filas)
    tabla["p_valor_bh"] = benjamini_hochberg(tabla["p_valor"].tolist()).round(6)
    tabla["significativo_bh"] = (tabla["p_valor_bh"] < ALPHA).astype(int)
    return tabla


def aplicar_bh(resultados: pd.DataFrame) -> pd.DataFrame:
    """Aplica Benjamini-Hochberg por familia de hipotesis.

    La familia se define por la combinacion (conjunto, etiqueta), de modo que
    las comparaciones multiples se corrigen dentro de cada subproblema.

    Args:
        resultados: tabla con las columnas ``conjunto``, ``etiqueta`` y
            ``p_valor``.

    Returns:
        Copia de la tabla con ``p_valor_bh``, ``significativo_bh`` y
        ``familia``.
    """
    datos = resultados.copy()
    datos["familia"] = datos["conjunto"] + "::" + datos["etiqueta"]
    datos["p_valor_bh"] = np.nan
    for indices in datos.groupby("familia").groups.values():
        datos.loc[indices, "p_valor_bh"] = benjamini_hochberg(
            datos.loc[indices, "p_valor"].tolist()
        )
    datos["p_valor_bh"] = datos["p_valor_bh"].round(6)
    datos["significativo_bh"] = (datos["p_valor_bh"] < ALPHA).astype(int)
    return datos


# ---------------------------------------------------------------------------
# Seccion 5 - Caracterizacion de TR
# ---------------------------------------------------------------------------
def rasgos_tr_categoricos(hs1: pd.DataFrame, columnas: Sequence[str]) -> pd.DataFrame:
    """Contrasta rasgos de targeting entre TR=1 y TR=0 dentro de HS=1.

    Args:
        hs1: subconjunto de train con HS=1.
        columnas: rasgos binarios a contrastar.

    Returns:
        Tabla con frecuencias, porcentajes, diferencia con IC95, prueba
        estadistica, V de Cramer y magnitud.
    """
    resultados = pruebas_categoricas(hs1, "TR", columnas, "hs1")
    tabla = pd.DataFrame(resultados)
    if tabla.empty:
        return tabla
    tabla["n_total"] = len(hs1)
    tabla = aplicar_bh(tabla)
    columnas_orden = [
        "fenomeno", "n_con_fenomeno", "p_etiqueta_con_fenomeno",
        "p_etiqueta_sin_fenomeno", "diferencia_proporciones", "ic95_inferior",
        "ic95_superior", "prueba", "chi2", "p_valor", "p_valor_bh", "v_cramer",
        "magnitud", "significativo_bh", "conjunto", "etiqueta",
    ]
    return tabla[columnas_orden].sort_values("v_cramer", ascending=False).reset_index(drop=True)


def rasgos_tr_continuos(hs1: pd.DataFrame, columnas: Sequence[str]) -> pd.DataFrame:
    """Contrasta variables continuas entre TR=1 y TR=0 dentro de HS=1.

    Args:
        hs1: subconjunto con HS=1.
        columnas: variables continuas.

    Returns:
        Tabla con medianas, Mann-Whitney y delta de Cliff.
    """
    tabla = pd.DataFrame(pruebas_continuas(hs1, "TR", columnas, "hs1"))
    if tabla.empty:
        return tabla
    tabla = aplicar_bh(tabla)
    return tabla.sort_values("delta_cliff", key=lambda serie: serie.abs(), ascending=False).reset_index(drop=True)


def construir_ejemplos_tr(hs1: pd.DataFrame, columnas_rasgos: Sequence[str]) -> pd.DataFrame:
    """Construye la tabla cualitativa de 40 ejemplos TR=1 y 40 TR=0.

    Args:
        hs1: subconjunto con HS=1.
        columnas_rasgos: rasgos binarios que se anotan automaticamente.

    Returns:
        Tabla lista para revision humana, con columnas vacias para el juicio
        del anotador.
    """
    muestras = []
    for valor in (1, 0):
        subconjunto = hs1[hs1["TR"] == valor]
        tamano = min(N_REVIEW_SAMPLES, len(subconjunto))
        muestras.append(subconjunto.sample(n=tamano, random_state=RANDOM_STATE))
    tabla = pd.concat(muestras).sort_values(["TR", "id"], ascending=[False, True]).reset_index(drop=True)
    columnas_disponibles = [columna for columna in columnas_rasgos if columna in tabla.columns]
    anotaciones = tabla[columnas_disponibles].astype(int)
    tabla = tabla.copy()
    tabla["rasgos_detectados"] = [
        ", ".join(
            columna for columna in columnas_disponibles if fila[columna] == 1
        )
        for _, fila in anotaciones.iterrows()
    ]
    tabla["perfil_automatico"] = np.where(
        tabla["TR"] == 1,
        np.where(
            tabla["tiene_mencion_inicial"] == 1,
            "dirigido_a_usuario_especifico",
            "individual_sin_mencion_inicial",
        ),
        np.where(
            tabla["tiene_referencia_colectiva"] == 1,
            "referencia_colectiva_explicita",
            "sin_target_explicito",
        ),
    )
    tabla["revisado_por_humano"] = ""
    tabla["etiqueta_sugerida_humano"] = ""
    tabla["nota_humana"] = ""
    for columna in columnas_disponibles:
        tabla[f"rasgo_{columna}"] = anotaciones[columna].to_numpy()
    orden = [
        "id", "HS", "TR", "AG", "n_caracteres", "text", "rasgos_detectados",
        "perfil_automatico",
    ] + [f"rasgo_{columna}" for columna in columnas_disponibles] + [
        "revisado_por_humano", "etiqueta_sugerida_humano", "nota_humana",
    ]
    return tabla[[columna for columna in orden if columna in tabla.columns]]


def conclusion_tr(
    tabla_categorica: pd.DataFrame,
    tabla_continua: pd.DataFrame,
    hs1: pd.DataFrame,
) -> dict:
    """Resume la evidencia sobre la semantica de TR sin decidir (D-005).

    Args:
        tabla_categorica: resultados de los rasgos categoricos.
        tabla_continua: resultados de los rasgos continuos.
        hs1: subconjunto con HS=1.

    Returns:
        Diccionario con los indicadores que sostienen la interpretacion.
    """
    def valor(fenomeno: str, campo: str) -> float:
        """Devuelve el estadistico ``campo`` de ``fenomeno`` o NaN si no existe."""
        fila = tabla_categorica.loc[tabla_categorica["fenomeno"] == fenomeno]
        if fila.empty:
            return float("nan")
        return float(fila.iloc[0][campo])

    total = len(hs1)
    positivos = int((hs1["TR"] == 1).sum())
    mascara_tr0_colectivo = (hs1["TR"] == 0) & (hs1["tiene_referencia_colectiva"] == 1)
    return {
        "n_hs1": total,
        "n_tr1": positivos,
        "n_tr0": int(total - positivos),
        "proporcion_tr1_en_hs1": round(positivos / total, 4),
        "v_mencion_inicial": valor("tiene_mencion_inicial", "v_cramer"),
        "v_mencion": valor("tiene_mencion", "v_cramer"),
        "v_referencia_colectiva": valor("tiene_referencia_colectiva", "v_cramer"),
        "v_insulto": valor("tiene_insulto", "v_cramer"),
        "v_insulto_dirigido": valor("tiene_insulto_dirigido", "v_cramer"),
        "v_segunda_persona": valor("tiene_segunda_persona", "v_cramer"),
        "v_imperativo": valor("imperativo", "v_cramer"),
        "v_url": valor("tiene_url", "v_cramer"),
        "p_tr0_con_colectivo": round(
            float((hs1.loc[hs1["TR"] == 0, "tiene_referencia_colectiva"] == 1).mean()), 4
        ),
        "p_tr1_con_colectivo": round(
            float((hs1.loc[hs1["TR"] == 1, "tiene_referencia_colectiva"] == 1).mean()), 4
        ),
        "n_tr0_con_colectivo": int(mascara_tr0_colectivo.sum()),
        "n_tr0_con_colectivo_y_mencion_inicial": int(
            (mascara_tr0_colectivo & (hs1["tiene_mencion_inicial"] == 1)).sum()
        ),
    }


# ---------------------------------------------------------------------------
# Seccion 6 - Figuras
# ---------------------------------------------------------------------------
def configurar_estilo() -> None:
    """Configura el estilo visual de matplotlib y seaborn."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.titleweight": "bold",
            "font.size": 10,
            "figure.autolayout": False,
        }
    )


def guardar_figura(figura: plt.Figure, nombre: str) -> Path:
    """Guarda una figura en ``reports/figuras`` y la cierra.

    Args:
        figura: figura de matplotlib.
        nombre: nombre del archivo sin extension.

    Returns:
        Ruta del archivo guardado.
    """
    ruta = FIGURES_DIR / f"{nombre}.png"
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(figura)
    return ruta


def figura_distribucion_etiquetas(marginal: pd.DataFrame) -> Path:
    """Figura 1: prevalencia de las tres etiquetas.

    Args:
        marginal: tabla de :func:`tabla_etiquetas`.

    Returns:
        Ruta de la figura.
    """
    figura, ejes = plt.subplots(1, 2, figsize=(11, 4.2))
    colores = [PALETA_ETIQUETA[etiqueta] for etiqueta in marginal["etiqueta"]]
    ejes[0].bar(marginal["etiqueta"], marginal["positivos"], color=colores, edgecolor="black")
    ejes[0].set_title("Positivos por etiqueta (train, n=1731)")
    ejes[0].set_xlabel("Etiqueta")
    ejes[0].set_ylabel("Numero de textos")
    for posicion, valor in enumerate(marginal["positivos"]):
        ejes[0].text(posicion, valor + 8, str(valor), ha="center", fontsize=10)
    ejes[1].bar(
        marginal["etiqueta"],
        marginal["prevalencia"] * 100,
        color=colores,
        edgecolor="black",
    )
    ejes[1].axhline(50, color="#555555", linestyle="--", linewidth=1)
    ejes[1].set_title("Prevalencia de la clase positiva")
    ejes[1].set_xlabel("Etiqueta")
    ejes[1].set_ylabel("Porcentaje (%)")
    for posicion, valor in enumerate(marginal["prevalencia"] * 100):
        ejes[1].text(posicion, valor + 1, f"{valor:.2f} %", ha="center", fontsize=10)
    figura.suptitle("Figura 1. Distribucion de etiquetas en el conjunto de entrenamiento")
    figura.tight_layout()
    return guardar_figura(figura, "fig_01_distribucion_etiquetas")


def figura_combinaciones(combinaciones: pd.DataFrame) -> Path:
    """Figura 2: frecuencia de las combinaciones HS x TR x AG.

    Args:
        combinaciones: tabla de :func:`tabla_combinaciones`.

    Returns:
        Ruta de la figura.
    """
    datos = combinaciones.copy()
    datos["etiqueta"] = (
        "(" + datos["HS"].astype(str) + "," + datos["TR"].astype(str) + "," + datos["AG"].astype(str) + ")"
    )
    colores = [
        PALETA_ETIQUETA["HS"] if fila.HS == 1 else "#c8cdd2" for fila in datos.itertuples()
    ]
    figura, eje = plt.subplots(figsize=(9, 4.5))
    eje.bar(datos["etiqueta"], datos["n"], color=colores, edgecolor="black")
    for posicion, valor in enumerate(datos["n"]):
        eje.text(posicion, valor + 12, str(valor), ha="center", fontsize=9)
    eje.set_title("Figura 2. Combinaciones HS x TR x AG en train (n=1731)")
    eje.set_xlabel("Combinacion (HS, TR, AG)")
    eje.set_ylabel("Numero de textos")
    eje.tick_params(axis="x", rotation=20)
    figura.tight_layout()
    return guardar_figura(figura, "fig_02_combinaciones_etiquetas")


def figura_histograma_longitudes(fenomenos: pd.DataFrame) -> Path:
    """Figura 3: histogramas de longitud por etiqueta y clase.

    Args:
        fenomenos: marco con etiquetas y ``n_caracteres``.

    Returns:
        Ruta de la figura.
    """
    figura, ejes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for eje, etiqueta in zip(ejes, LABELS):
        for valor in (0, 1):
            subconjunto = fenomenos.loc[fenomenos[etiqueta] == valor, "n_caracteres"]
            eje.hist(
                subconjunto,
                bins=np.arange(0, 320, 15),
                alpha=0.65,
                color=PALETA_AUSENCIA_PRESENCIA[valor],
                label=f"{etiqueta}={valor} (n={len(subconjunto)})",
                edgecolor="white",
            )
        eje.set_title(f"Longitud en caracteres segun {etiqueta}")
        eje.set_xlabel("Caracteres por texto")
        eje.legend(fontsize=8)
    ejes[0].set_ylabel("Numero de textos")
    figura.suptitle("Figura 3. Distribucion de longitudes por etiqueta y clase")
    figura.tight_layout()
    return guardar_figura(figura, "fig_03_histograma_longitudes_por_etiqueta")


def figura_boxplot_longitudes(fenomenos: pd.DataFrame) -> Path:
    """Figura 4: boxplots de caracteres, palabras y palabras unicas.

    Args:
        fenomenos: marco con etiquetas y metricas de longitud.

    Returns:
        Ruta de la figura.
    """
    variables = ["n_caracteres", "n_palabras", "n_palabras_unicas"]
    largo = formato_largo(fenomenos, variables)
    figura, ejes = plt.subplots(1, 3, figsize=(15, 4.4))
    for eje, variable in zip(ejes, variables):
        subconjunto = largo[largo["variable"] == variable]
        sns.boxplot(
            data=subconjunto,
            x="etiqueta",
            y="valor",
            hue="clase",
            palette=PALETA_AUSENCIA_PRESENCIA,
            showfliers=False,
            width=0.65,
            ax=eje,
        )
        eje.set_title(f"{variable} por etiqueta")
        eje.set_xlabel("Etiqueta")
        eje.set_ylabel(variable)
        eje.legend(title="Clase", fontsize=8)
    figura.suptitle("Figura 4. Longitud y riqueza por etiqueta (clase 0 frente a 1)")
    figura.tight_layout()
    return guardar_figura(figura, "fig_04_boxplot_longitudes")


def figura_top_palabras(train: pd.DataFrame, lemas: Sequence[Sequence[str]]) -> Path:
    """Figura 5: top-20 palabras por etiqueta (lemas sin palabras funcionales).

    Args:
        train: conjunto de entrenamiento.
        lemas: lemas por documento.

    Returns:
        Ruta de la figura.
    """
    figura, ejes = plt.subplots(1, 3, figsize=(16, 5.2))
    for eje, etiqueta in zip(ejes, LABELS):
        contador_positivo: Counter[str] = Counter()
        contador_negativo: Counter[str] = Counter()
        for indice, fila in enumerate(train.itertuples()):
            destino = contador_positivo if getattr(fila, etiqueta) == 1 else contador_negativo
            destino.update(
                lema
                for lema in lemas[indice]
                if lema not in STOP_WORDS and lema.isalpha() and len(lema) > 2
            )
        total_positivo = sum(contador_positivo.values()) or 1
        total_negativo = sum(contador_negativo.values()) or 1
        mas_frecuentes = contador_positivo.most_common(20)
        etiquetas_y = [palabra for palabra, _ in mas_frecuentes][::-1]
        valores = [frecuencia / total_positivo * 1000 for _, frecuencia in mas_frecuentes][::-1]
        tasa_negativa = [contador_negativo.get(palabra, 0) / total_negativo * 1000 for palabra, _ in mas_frecuentes][::-1]
        posiciones = np.arange(len(etiquetas_y))
        eje.barh(posiciones - 0.2, valores, height=0.4, color=PALETA_ETIQUETA[etiqueta], label=f"{etiqueta}=1")
        eje.barh(posiciones + 0.2, tasa_negativa, height=0.4, color="#c8cdd2", label=f"{etiqueta}=0")
        eje.set_yticks(posiciones)
        eje.set_yticklabels(etiquetas_y, fontsize=8)
        eje.set_title(f"Top-20 lemas si {etiqueta}=1")
        eje.set_xlabel("Frecuencia por mil lemas")
        eje.legend(fontsize=8)
    figura.suptitle("Figura 5. Palabras mas frecuentes por etiqueta (sin palabras funcionales)")
    figura.tight_layout()
    return guardar_figura(figura, "fig_05_top20_palabras_por_etiqueta")


def figura_nubes_palabras(train: pd.DataFrame, lemas: Sequence[Sequence[str]]) -> list[Path]:
    """Figuras 6-8: nubes de palabras por etiqueta (clase positiva).

    Args:
        train: conjunto de entrenamiento.
        lemas: lemas por documento.

    Returns:
        Lista con las rutas generadas.
    """
    rutas = []
    for etiqueta in LABELS:
        contador: Counter[str] = Counter()
        for indice, fila in enumerate(train.itertuples()):
            if getattr(fila, etiqueta) == 1:
                contador.update(
                    lema
                    for lema in lemas[indice]
                    if lema not in STOP_WORDS and lema.isalpha() and len(lema) > 2
                )
        nube = WordCloud(
            width=900,
            height=520,
            max_words=100,
            background_color="white",
            colormap="Blues" if etiqueta == "HS" else ("Reds" if etiqueta == "TR" else "YlOrBr"),
            collocations=False,
            stopwords=set(STOP_WORDS),
            random_state=RANDOM_STATE,
        ).generate_from_frequencies(dict(contador.most_common(300)))
        figura, eje = plt.subplots(figsize=(10, 5.8))
        eje.imshow(nube, interpolation="bilinear")
        eje.axis("off")
        eje.set_title(
            f"Nube de palabras para {etiqueta}=1 (n={int((train[etiqueta] == 1).sum())}, maximo 100 palabras)"
        )
        rutas.append(guardar_figura(figura, f"fig_0{6 + LABELS.index(etiqueta)}_wordcloud_{etiqueta}"))
    return rutas


def figura_densidad_por_clase(
    fenomenos: pd.DataFrame, variables: Sequence[str], titulo: str, nombre: str
) -> Path:
    """Genera una figura de violin con un panel por etiqueta.

    Args:
        fenomenos: marco con etiquetas y variables.
        variables: variables a mostrar (una por panel).
        titulo: titulo de la figura.
        nombre: nombre del archivo sin extension.

    Returns:
        Ruta de la figura.
    """
    figura, ejes = plt.subplots(1, len(variables), figsize=(5 * len(variables), 4.4))
    if len(variables) == 1:
        ejes = [ejes]
    largo = formato_largo(fenomenos, variables)
    for eje, variable in zip(ejes, variables):
        subconjunto = largo[largo["variable"] == variable]
        sns.violinplot(
            data=subconjunto,
            x="etiqueta",
            y="valor",
            hue="clase",
            palette=PALETA_AUSENCIA_PRESENCIA,
            split=True,
            inner="quartile",
            density_norm="width",
            ax=eje,
        )
        eje.set_title(variable)
        eje.set_xlabel("Etiqueta")
        eje.set_ylabel(variable)
        eje.legend(title="Clase", fontsize=8)
    figura.suptitle(titulo)
    figura.tight_layout()
    return guardar_figura(figura, nombre)


def figura_elongaciones(fenomenos: pd.DataFrame) -> Path:
    """Figura: elongaciones, repeticion de puntuacion y palabras duplicadas.

    Args:
        fenomenos: marco con etiquetas y conteos.

    Returns:
        Ruta de la figura.
    """
    figura, ejes = plt.subplots(1, 3, figsize=(15, 4.4))
    variables = ["n_letras_repetidas", "n_puntuacion_repetida", "n_palabras_duplicadas"]
    largo = formato_largo(fenomenos, variables)
    largo["presente"] = (largo["valor"] > 0).astype(int)
    for eje, variable in zip(ejes, variables):
        subconjunto = largo[largo["variable"] == variable]
        tabla = (
            subconjunto.groupby(["etiqueta", "clase"])["presente"].mean().mul(100).unstack("clase")
        )
        tabla = tabla.reindex(LABELS)
        posiciones = np.arange(len(LABELS))
        eje.bar(
            posiciones - 0.2,
            tabla[1].to_numpy(),
            width=0.4,
            color=PALETA_AUSENCIA_PRESENCIA[1],
            label="Clase = 1",
            edgecolor="black",
        )
        eje.bar(
            posiciones + 0.2,
            tabla[0].to_numpy(),
            width=0.4,
            color=PALETA_AUSENCIA_PRESENCIA[0],
            label="Clase = 0",
            edgecolor="black",
        )
        eje.set_xticks(posiciones)
        eje.set_xticklabels(LABELS)
        eje.set_title(f"Textos con {variable} (>0)")
        eje.set_ylabel("Porcentaje de textos (%)")
        eje.set_xlabel("Etiqueta")
        eje.legend(fontsize=8)
    figura.suptitle("Figura 13. Elongaciones, puntuacion repetida y duplicacion por clase")
    figura.tight_layout()
    return guardar_figura(figura, "fig_13_elongaciones_vs_etiqueta")


def figura_correlacion_etiquetas(correlaciones: pd.DataFrame, train: pd.DataFrame) -> Path:
    """Figura: mapa de calor de asociacion entre etiquetas.

    Args:
        correlaciones: tabla de :func:`correlaciones_etiquetas`.
        train: conjunto de entrenamiento.

    Returns:
        Ruta de la figura.
    """
    variables = list(LABELS) + ["n_caracteres"]
    matriz = pd.DataFrame(np.eye(len(variables)), index=variables, columns=variables)
    datos = train.copy()
    datos["n_caracteres"] = datos["text"].str.len()
    for _, fila in correlaciones.iterrows():
        if fila["variable_a"] in variables and fila["variable_b"] in variables:
            valor = fila["phi"] if fila["tipo"] == "binaria" else fila["spearman"]
            matriz.loc[fila["variable_a"], fila["variable_b"]] = valor
            matriz.loc[fila["variable_b"], fila["variable_a"]] = valor
    figura, eje = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(
        matriz.astype(float),
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "phi / rho"},
        ax=eje,
    )
    eje.set_title("Figura. Correlacion entre etiquetas y longitud")
    figura.tight_layout()
    return guardar_figura(figura, "fig_14_heatmap_correlacion_etiquetas")


def figura_heatmap_cramersv(tabla_categorica: pd.DataFrame) -> Path:
    """Figura: mapa de calor de V de Cramer de fenomenos contra etiquetas.

    Args:
        tabla_categorica: resultados categoricos con BH aplicado.

    Returns:
        Ruta de la figura.
    """
    datos = tabla_categorica[tabla_categorica["conjunto"] == "completo"].copy()
    if datos.empty:
        datos = tabla_categorica.copy()
    pivote = datos.pivot_table(
        index="fenomeno", columns="etiqueta", values="v_cramer", aggfunc="max"
    )
    firmas = datos.pivot_table(
        index="fenomeno", columns="etiqueta", values="significativo_bh", aggfunc="max"
    )
    pivote = pivote.reindex(columns=[columna for columna in LABELS if columna in pivote.columns])
    firmas = firmas.reindex(columns=pivote.columns)
    pivote = pivote.reindex(pivote.mean(axis=1).sort_values(ascending=False).index)
    figura, eje = plt.subplots(figsize=(6.6, max(6.0, 0.32 * len(pivote))))
    sns.heatmap(
        pivote.astype(float),
        annot=True,
        fmt=".2f",
        cmap=PALETA_SECUENCIAL,
        vmin=0,
        vmax=float(np.nanmax(pivote.to_numpy())) if pivote.notna().any().any() else 1.0,
        cbar_kws={"label": "V de Cramer (corregida)"},
        ax=eje,
    )
    for posicion_fila, fenomeno in enumerate(pivote.index):
        for posicion_columna, etiqueta in enumerate(pivote.columns):
            if firmas.loc[fenomeno, etiqueta] == 1:
                eje.text(
                    posicion_columna + 0.78,
                    posicion_fila + 0.22,
                    "*",
                    color="#b03a2e",
                    fontsize=10,
                    ha="center",
                )
    eje.set_title("Figura. Asociacion fenomeno-etiqueta (V de Cramer; * q<0.05 con BH)")
    eje.set_xlabel("Etiqueta")
    eje.set_ylabel("Fenomeno de superficie")
    figura.tight_layout()
    return guardar_figura(figura, "fig_15_heatmap_chi2_cramersv")


def figura_rasgos_tr(tabla_categorica: pd.DataFrame) -> Path:
    """Figura: rasgos que mas discriminan TR dentro de HS=1.

    Args:
        tabla_categorica: tabla de :func:`rasgos_tr_categoricos`.

    Returns:
        Ruta de la figura.
    """
    datos = tabla_categorica.sort_values("v_cramer", ascending=False).head(16).copy()
    datos["etiqueta_legible"] = datos["fenomeno"].str.replace("tiene_", "", regex=False)
    colores = [
        PALETA_ETIQUETA["TR"] if fila.diferencia_proporciones > 0 else "#9aa5b1"
        for fila in datos.itertuples()
    ]
    figura, ejes = plt.subplots(1, 2, figsize=(15, 5.6))
    ejes[0].barh(
        datos["etiqueta_legible"][::-1],
        datos["diferencia_proporciones"][::-1],
        color=colores[::-1],
        edgecolor="black",
    )
    ejes[0].axvline(0, color="#333333", linewidth=1)
    ejes[0].set_title("Diferencia de proporciones (TR=1 menos TR=0)")
    ejes[0].set_xlabel("Diferencia (puntos porcentuales / 100)")
    for posicion, (_, fila) in enumerate(datos.iloc[::-1].iterrows()):
        valor = fila["diferencia_proporciones"]
        eje_objetivo = ejes[0]
        eje_objetivo.errorbar(
            valor,
            posicion,
            xerr=[[valor - fila["ic95_inferior"]], [fila["ic95_superior"] - valor]],
            fmt="none",
            ecolor="#333333",
            capsize=3,
            linewidth=1,
        )
    ejes[1].barh(
        datos["etiqueta_legible"][::-1],
        datos["v_cramer"][::-1],
        color=PALETA_ETIQUETA["TR"],
        edgecolor="black",
    )
    ejes[1].set_title("V de Cramer (magnitud de la asociacion)")
    ejes[1].set_xlabel("V de Cramer")
    figura.suptitle("Figura. Rasgos que mas discriminan TR dentro de HS=1 (n=756)")
    figura.tight_layout()
    return guardar_figura(figura, "fig_16_rasgos_TR")


def figura_textometria_heatmap(textometria: pd.DataFrame) -> Path:
    """Figura: perfil textometrico estandarizado por clase.

    Args:
        textometria: tabla de :func:`tabla_textometria_por_clase`.

    Returns:
        Ruta de la figura.
    """
    datos = textometria[textometria["etiqueta"].isin(LABELS)].copy()
    datos["grupo"] = datos["etiqueta"] + "=" + datos["clase"].astype(str)
    pivote = datos.pivot_table(index="variable", columns="grupo", values="media")
    valores = pivote.to_numpy(dtype=float)
    desviaciones = np.nanstd(valores, axis=1, keepdims=True)
    desviaciones[desviaciones == 0] = 1.0
    estandarizado = (valores - np.nanmean(valores, axis=1, keepdims=True)) / desviaciones
    figura, eje = plt.subplots(figsize=(9.5, max(6.0, 0.38 * len(pivote))))
    sns.heatmap(
        pd.DataFrame(estandarizado, index=pivote.index, columns=pivote.columns),
        annot=True,
        fmt=".1f",
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "media estandarizada por variable"},
        ax=eje,
    )
    eje.set_title("Figura. Perfil textometrico por etiqueta y clase (z por variable)")
    eje.set_xlabel("Etiqueta = clase")
    eje.set_ylabel("Variable")
    figura.tight_layout()
    return guardar_figura(figura, "fig_17_textometria_heatmap_clases")


def figura_info_mutua(info_mutua: pd.DataFrame) -> Path:
    """Figura: informacion mutua ajustada de los fenomenos con cada etiqueta.

    Args:
        info_mutua: tabla de :func:`tabla_info_mutua`.

    Returns:
        Ruta de la figura.
    """
    datos = info_mutua[info_mutua["conjunto"] == "completo"].copy()
    mejores = (
        datos.sort_values("info_mutua_ajustada", ascending=False)
        .groupby("etiqueta")
        .head(10)
        .copy()
    )
    mejores["etiqueta_legible"] = mejores["fenomeno"].str.replace("tiene_", "", regex=False)
    figura, eje = plt.subplots(figsize=(10, 7))
    sns.barplot(
        data=mejores,
        y="etiqueta_legible",
        x="info_mutua_ajustada",
        hue="etiqueta",
        palette=PALETA_ETIQUETA,
        ax=eje,
    )
    eje.set_title("Figura. Informacion mutua ajustada (top-10 por etiqueta)")
    eje.set_xlabel("Informacion mutua ajustada (nats)")
    eje.set_ylabel("Fenomeno")
    figura.tight_layout()
    return guardar_figura(figura, "fig_19_informacion_mutua")


def figura_emojis_emoticones(fenomenos: pd.DataFrame) -> Path:
    """Figura: densidad de emojis, emoticones y risas por clase.

    Args:
        fenomenos: marco con etiquetas y conteos.

    Returns:
        Ruta de la figura.
    """
    figura, ejes = plt.subplots(1, 3, figsize=(15, 4.4))
    variables = ["n_emojis", "n_emoticones", "n_risas"]
    largo = formato_largo(fenomenos, variables)
    for eje, variable in zip(ejes, variables):
        subconjunto = largo[largo["variable"] == variable]
        sns.boxplot(
            data=subconjunto,
            x="etiqueta",
            y="valor",
            hue="clase",
            palette=PALETA_AUSENCIA_PRESENCIA,
            showfliers=False,
            ax=eje,
        )
        eje.set_title(variable)
        eje.set_xlabel("Etiqueta")
        eje.set_ylabel("Conteo por texto")
        eje.legend(title="Clase", fontsize=8)
    total_emojis = int(fenomenos["n_emojis"].sum())
    figura.suptitle(
        f"Figura 12. Emojis, emoticones y risas por clase (emojis Unicode en el corpus: {total_emojis})"
    )
    figura.tight_layout()
    return guardar_figura(figura, "fig_12_densidad_emojis_emoticones")


def figura_perfil_combos(textometria: pd.DataFrame) -> Path:
    """Figura: longitud y menciones medias por combinacion de etiquetas.

    Args:
        textometria: tabla de textometria por clase y combinacion.

    Returns:
        Ruta de la figura.
    """
    datos = textometria[textometria["etiqueta"].str.startswith("combo_")].copy()
    datos["combo"] = datos["etiqueta"].str.replace("combo_", "", regex=False)
    figura, ejes = plt.subplots(1, 2, figsize=(13, 4.6))
    for eje, variable in zip(ejes, ["n_caracteres", "n_menciones"]):
        subset = datos[datos["variable"] == variable].sort_values("media", ascending=False)
        colores = [PALETA_ETIQUETA["HS"] if combo.startswith("1") else "#c8cdd2" for combo in subset["combo"]]
        eje.bar(subset["combo"], subset["media"], color=colores, edgecolor="black")
        eje.set_title(f"Media de {variable} por combinacion")
        eje.set_xlabel("Combinacion (HS, TR, AG)")
        eje.set_ylabel("Media")
        for posicion, valor in enumerate(subset["media"]):
            eje.text(posicion, valor, f"{valor:.1f}", ha="center", va="bottom", fontsize=8)
        eje.tick_params(axis="x", rotation=25)
    figura.suptitle("Figura. Perfil de las combinaciones de etiquetas")
    figura.tight_layout()
    return guardar_figura(figura, "fig_18_perfil_combos")


def generar_figuras(contexto: dict) -> list[Path]:
    """Genera todas las figuras del EDA.

    Args:
        contexto: diccionario con los artefactos calculados.

    Returns:
        Lista de rutas generadas.
    """
    rutas = [
        figura_distribucion_etiquetas(contexto["etiquetas"]),
        figura_combinaciones(contexto["combinaciones"]),
        figura_histograma_longitudes(contexto["fenomenos"]),
        figura_boxplot_longitudes(contexto["fenomenos"]),
        figura_top_palabras(contexto["train"], contexto["lemas"]),
    ]
    rutas.extend(figura_nubes_palabras(contexto["train"], contexto["lemas"]))
    rutas.append(
        figura_densidad_por_clase(
            contexto["fenomenos"],
            ["ratio_puntuacion", "n_signos", "n_exclamaciones"],
            "Figura. Densidad de puntuacion por etiqueta y clase",
            "fig_09_densidad_puntuacion",
        )
    )
    rutas.append(
        figura_densidad_por_clase(
            contexto["fenomenos"],
            ["ratio_mayusculas", "n_palabras_mayusculas", "n_caracteres"],
            "Figura. Uso de mayusculas y longitud por etiqueta y clase",
            "fig_10_densidad_mayusculas",
        )
    )
    rutas.append(
        figura_densidad_por_clase(
            contexto["fenomenos"],
            ["n_hashtags", "n_menciones", "n_urls"],
            "Figura. Hashtags, menciones y URLs por etiqueta y clase",
            "fig_11_densidad_hashtags_menciones",
        )
    )
    rutas.append(figura_emojis_emoticones(contexto["fenomenos"]))
    rutas.append(figura_elongaciones(contexto["fenomenos"]))
    rutas.append(figura_correlacion_etiquetas(contexto["correlaciones"], contexto["train"]))
    rutas.append(figura_heatmap_cramersv(contexto["tests_categoricos"]))
    rutas.append(figura_rasgos_tr(contexto["tr_categorico"]))
    rutas.append(figura_textometria_heatmap(contexto["textometria"]))
    rutas.append(figura_info_mutua(contexto["info_mutua"]))
    rutas.append(figura_perfil_combos(contexto["textometria"]))
    return rutas


# ---------------------------------------------------------------------------
# Seccion 7 - Reportes
# ---------------------------------------------------------------------------
def bloque(
    titulo: str,
    observacion: str,
    hipotesis: str,
    metodo: str,
    resultado: str,
    interpretacion: str,
    decision: str,
    siguiente: str,
) -> str:
    """Formatea un bloque del resumen con el esquema de siete pasos.

    Args:
        titulo: titulo del bloque.
        observacion: que se observo.
        hipotesis: pregunta o hipotesis asociada.
        metodo: como se midio.
        resultado: numeros obtenidos.
        interpretacion: lectura de los numeros.
        decision: decision tomada o pendiente.
        siguiente: siguiente paso.

    Returns:
        Bloque en Markdown.
    """
    etiquetas = [
        ("OBSERVACION", observacion),
        ("HIPOTESIS", hipotesis),
        ("METODO", metodo),
        ("RESULTADO", resultado),
        ("INTERPRETACION", interpretacion),
        ("DECISION", decision),
        ("SIGUIENTE PASO", siguiente),
    ]
    lineas = [f"### {titulo}", ""]
    lineas.extend(f"- **{nombre}**: {texto}" for nombre, texto in etiquetas)
    lineas.append("")
    return "\n".join(lineas)


def resumen_top_fenomenos(tabla: pd.DataFrame, etiqueta: str, k: int = 6) -> str:
    """Resume los fenomenos mas asociados con una etiqueta.

    Args:
        tabla: tabla categorica con BH aplicado.
        etiqueta: etiqueta de interes.
        k: numero de fenomenos a listar.

    Returns:
        Cadena con el listado en formato ``fenomeno (V=..., q=..., direccion)``.
    """
    datos = tabla[
        (tabla["etiqueta"] == etiqueta)
        & (tabla["conjunto"] == "completo")
        & (tabla["significativo_bh"] == 1)
    ].sort_values("v_cramer", ascending=False)
    if datos.empty:
        datos = tabla[(tabla["etiqueta"] == etiqueta) & (tabla["conjunto"] == "completo")].sort_values(
            "v_cramer", ascending=False
        )
    piezas = []
    for _, fila in datos.head(k).iterrows():
        direccion = "mas frecuente cuando la etiqueta es 1" if fila["diferencia_proporciones"] > 0 else "mas frecuente cuando la etiqueta es 0"
        piezas.append(
            f"`{fila['fenomeno']}` (V={fila['v_cramer']:.3f}, q={fila['p_valor_bh']:.2e}, "
            f"{fila['magnitud']}, {direccion})"
        )
    return "; ".join(piezas) if piezas else "sin asociaciones destacadas"


def seleccionar_experimentos(hipotesis: Sequence[str]) -> str:
    """Prioriza los experimentos sugeridos a partir de las hipotesis apoyadas.

    Args:
        hipotesis: lista de hipotesis con apoyo o refutacion inicial.

    Returns:
        Cadena con la priorizacion de experimentos.
    """
    prioridad = [hipotesis_nombre for hipotesis_nombre in hipotesis if hipotesis_nombre]
    return ", ".join(prioridad) if prioridad else "ninguno"


def escribir_resumen(contexto: dict) -> Path:
    """Escribe ``reports/eda_resumen.md`` con la interpretacion por bloque.

    Args:
        contexto: diccionario con todos los artefactos del EDA.

    Returns:
        Ruta del archivo escrito.
    """
    etiquetas = contexto["etiquetas"]
    combinaciones = contexto["combinaciones"]
    fenomenos = contexto["fenomenos"]
    tests_categoricos = contexto["tests_categoricos"]
    tests_continuos = contexto["tests_continuos"]
    tests_combos = contexto["tests_combos"]
    info_mutua = contexto["info_mutua"]
    correlaciones = contexto["correlaciones"]
    vocabulario = contexto["vocabulario"]
    tr_categorico = contexto["tr_categorico"]
    tr_continuo = contexto["tr_continuo"]
    conclusion = contexto["conclusion_tr"]
    informe = contexto["integridad"]
    n_filas_deduplicadas = contexto["n_filas_deduplicadas"]

    hs = etiquetas.loc[etiquetas["etiqueta"] == "HS"].iloc[0]
    tr = etiquetas.loc[etiquetas["etiqueta"] == "TR"].iloc[0]
    ag = etiquetas.loc[etiquetas["etiqueta"] == "AG"].iloc[0]
    combo_000 = combinaciones[(combinaciones["HS"] == 0) & (combinaciones["TR"] == 0) & (combinaciones["AG"] == 0)]
    combo_111 = combinaciones[(combinaciones["HS"] == 1) & (combinaciones["TR"] == 1) & (combinaciones["AG"] == 1)]
    combo_101 = combinaciones[(combinaciones["HS"] == 1) & (combinaciones["TR"] == 0) & (combinaciones["AG"] == 1)]
    combo_100 = combinaciones[(combinaciones["HS"] == 1) & (combinaciones["TR"] == 0) & (combinaciones["AG"] == 0)]
    n_000 = int(combo_000["n"].iloc[0]) if not combo_000.empty else 0
    n_111 = int(combo_111["n"].iloc[0]) if not combo_111.empty else 0
    n_101 = int(combo_101["n"].iloc[0]) if not combo_101.empty else 0
    n_100 = int(combo_100["n"].iloc[0]) if not combo_100.empty else 0

    cont_longitud = tests_continuos[
        (tests_continuos["variable"] == "n_caracteres") & (tests_continuos["conjunto"] == "completo")
    ]
    resumen_longitud_txt = "; ".join(
        f"{fila.etiqueta}: mediana {fila.mediana_positivos:.0f} frente a {fila.mediana_negativos:.0f} caracteres "
        f"(delta de Cliff {fila.delta_cliff:+.3f}, q={fila.p_valor_bh:.2e})"
        for _, fila in cont_longitud.iterrows()
    )
    cont_longitud_dedup = tests_continuos[
        (tests_continuos["variable"] == "n_caracteres") & (tests_continuos["conjunto"] == "deduplicado")
    ]
    resumen_longitud_dedup = "; ".join(
        f"{fila.etiqueta}: delta de Cliff {fila.delta_cliff:+.3f} (q={fila.p_valor_bh:.2e})"
        for _, fila in cont_longitud_dedup.iterrows()
    )
    rho_tr_hs = correlaciones[
        (correlaciones["variable_a"] == "HS") & (correlaciones["variable_b"] == "TR")
    ]
    rho_hs_ag = correlaciones[
        (correlaciones["variable_a"] == "HS") & (correlaciones["variable_b"] == "AG")
    ]
    rho_tr_ag = correlaciones[
        (correlaciones["variable_a"] == "TR") & (correlaciones["variable_b"] == "AG")
    ]
    voc_hs1 = vocabulario[(vocabulario["nivel"] == "etiqueta") & (vocabulario["grupo"] == "HS=1")]
    voc_hs0 = vocabulario[(vocabulario["nivel"] == "etiqueta") & (vocabulario["grupo"] == "HS=0")]
    voc_tr1 = vocabulario[(vocabulario["nivel"] == "etiqueta") & (vocabulario["grupo"] == "TR=1")]
    voc_tr0 = vocabulario[(vocabulario["nivel"] == "etiqueta") & (vocabulario["grupo"] == "TR=0")]
    n_emojis = int(fenomenos["n_emojis"].sum())
    n_emoticones = int((fenomenos["n_emoticones"] > 0).sum())
    n_risas = int((fenomenos["n_risas"] > 0).sum())

    mejores_mi = (
        info_mutua[info_mutua["conjunto"] == "completo"]
        .sort_values("info_mutua_ajustada", ascending=False)
        .groupby("etiqueta")
        .head(3)
    )
    mi_txt = "; ".join(
        f"{fila.etiqueta}: `{fila.fenomeno}` (AMI={fila.info_mutua_ajustada:.4f})"
        for _, fila in mejores_mi.iterrows()
    )
    ganancia_mi = (
        info_mutua[info_mutua["conjunto"] == "completo"]
        .groupby("fenomeno")["info_mutua_ajustada"]
        .max()
        .sort_values(ascending=False)
        .head(8)
    )
    perdida_mi = (
        info_mutua[
            (info_mutua["conjunto"] == "deduplicado") & (info_mutua["fenomeno"].isin(ganancia_mi.index))
        ]
        .groupby("fenomeno")["info_mutua_ajustada"]
        .max()
    )
    texto_mi_dedup = "; ".join(
        f"`{fenomeno}` AMI {ganancia_mi[fenomeno]:.4f} -> {perdida_mi.get(fenomeno, float('nan')):.4f}"
        for fenomeno in ganancia_mi.index
    )
    ganancia_efecto = tests_categoricos[
        (tests_categoricos["conjunto"] == "completo") & (tests_categoricos["etiqueta"] == "TR")
    ].set_index("fenomeno")["v_cramer"]
    ganancia_efecto_dedup = tests_categoricos[
        (tests_categoricos["conjunto"] == "deduplicado") & (tests_categoricos["etiqueta"] == "TR")
    ].set_index("fenomeno")["v_cramer"]
    texto_efecto_dedup = "; ".join(
        f"`{fenomeno}` V {ganancia_efecto[fenomeno]:.3f} -> {ganancia_efecto_dedup.get(fenomeno, float('nan')):.3f}"
        for fenomeno in ganancia_efecto.sort_values(ascending=False).head(6).index
    )
    top_tr = tr_categorico.head(8)
    texto_top_tr = "; ".join(
        f"`{fila.fenomeno}` (V={fila.v_cramer:.3f}, dif={fila.diferencia_proporciones:+.3f}, "
        f"IC95 [{fila.ic95_inferior:+.3f}, {fila.ic95_superior:+.3f}], q={fila.p_valor_bh:.2e})"
        for fila in top_tr.itertuples()
    )
    texto_top_tr_cont = "; ".join(
        f"`{fila.variable}` (delta de Cliff {fila.delta_cliff:+.3f}, q={fila.p_valor_bh:.2e})"
        for fila in tr_continuo.head(5).itertuples()
    )
    mejor_combo = tests_combos.sort_values("epsilon_cuadrado", ascending=False).head(5)
    texto_combos = "; ".join(
        f"`{fila.variable}` (epsilon^2={fila.epsilon_cuadrado:.3f}, q={fila.p_valor_bh:.2e})"
        for fila in mejor_combo.itertuples()
    )

    lineas = [
        "# EDA profundo, textometria y caracterizacion linguistica de TR",
        "",
        "- **Fase**: 2 (KAGGLE_TOXIC_2026_PUJ), fecha 2026-09-22, agente nlp-specialist.",
        "- **Datos**: `data/raw/toxic_train.csv` (1.731 filas) y `data/raw/toxic_test.csv` (1.154 filas, sin etiquetas).",
        (
            f"- **Integridad**: train {informe['train_filas']} filas, test {informe['test_filas']} filas, "
            f"sin nulos, ids disjuntos, violaciones TR=1 con HS=0: {informe['violaciones_tr_sin_hs']}."
        ),
        (
            "- **Restricciones**: sin transformers ni embeddings preentrenados; spaCy solo como anotador "
            "(POS, lemas, morfologia, D-003); sin recursos externos (D-002); test nunca usado para ajustar."
        ),
        "- **Script**: `experiments/01_eda_linguistico.py` (idempotente, ejecutable con el venv del proyecto).",
        "",
        "## Tabla de contenidos",
        "",
        "1. [Integridad y distribucion de etiquetas](#1-integridad-y-distribucion-de-etiquetas)",
        "2. [Longitudes y riqueza lexica](#2-longitudes-y-riqueza-lexica)",
        "3. [Textometria de superficie](#3-textometria-de-superficie)",
        "4. [Estadistica con tamano de efecto](#4-estadistica-con-tamano-de-efecto)",
        "5. [Caracterizacion de TR](#5-caracterizacion-de-tr)",
        "6. [Estructura de etiquetas](#6-estructura-de-etiquetas)",
        "7. [Sintesis: senales prometedoras e hipotesis](#7-sintesis-senales-prometedoras-e-hipotesis)",
        "8. [Limitaciones](#8-limitaciones)",
        "",
        "---",
        "",
        "## 1. Integridad y distribucion de etiquetas",
        "",
        bloque(
            "1.1 Integridad de los archivos oficiales",
            f"Los tres archivos oficiales cargan con `src.data`: train {informe['train_filas']} filas y "
            f"test {informe['test_filas']} filas, sin nulos, con ids numericos unicos y disjuntos.",
            "Ninguna senal util puede obtenerse de datos corruptos; la integridad es precondicion del resto del EDA.",
            "`src.data.read_train/read_test` mas verificacion de filas, nulos, dominios y reglas estructurales.",
            f"Filas correctas (1.731/1.154), sin nulos, ids disjuntos: {informe['ids_disjuntos']}; "
            f"violaciones de la regla TR=1 con HS=0: {informe['violaciones_tr_sin_hs']}.",
            "El corpus esta integro y la regla estructural TR=1 solo con HS=1 se cumple sin excepciones.",
            "Se continua con el EDA; no hay que corregir datos.",
            "Usar la estructura jerarquica en el post-proceso (H8).",
        ),
        bloque(
            "1.2 Prevalencia de HS, TR y AG",
            f"HS tiene {int(hs['positivos'])} positivos ({hs['prevalencia'] * 100:.2f} %), TR {int(tr['positivos'])} "
            f"({tr['prevalencia'] * 100:.2f} %) y AG {int(ag['positivos'])} ({ag['prevalencia'] * 100:.2f} %). "
            f"La combinacion mas frecuente es (0,0,0) con {n_000} textos y la segunda (1,1,1) con {n_111}.",
            "H8: las etiquetas no son independientes; la estructura condicionada debe explotarse.",
            "Conteo marginal y de combinaciones en train.",
            f"(0,0,0)={n_000}, (1,1,1)={n_111}, (1,0,1)={n_101}, (1,0,0)={n_100}, "
            f"(1,1,0)=28, (0,0,1)=26; TR=1 solo aparece con HS=1 en los 519 casos.",
            "El 54.8 % del corpus no tiene ninguna etiqueta y el 28.4 % tiene las tres; el espacio de etiquetas "
            "es fuertemente estructurado, con dos regimenes dominantes (nada toxico y todo toxico).",
            "H8 recibe apoyo estructural (no experimental todavia): el post-proceso jerarquico es seguro.",
            "E08 con las tres variantes de post-proceso sobre probabilidades out-of-fold.",
        ),
        bloque(
            "1.3 Distribucion marginal y desbalanceo",
            f"Las tres etiquetas tienen prevalencias moderadas: HS {hs['prevalencia'] * 100:.2f} %, "
            f"TR {tr['prevalencia'] * 100:.2f} %, AG {ag['prevalencia'] * 100:.2f} %.",
            "H9: con prevalencias distintas de 50 %, el umbral 0.5 no es necesariamente optimo.",
            "Tabla marginal de positivos, negativos y cociente de desbalanceo.",
            f"Desbalanceo negativo/positivo: HS {hs['desbalanceo_neg_pos']}, TR {tr['desbalanceo_neg_pos']}, "
            f"AG {ag['desbalanceo_neg_pos']}.",
            "El desbalanceo es moderado (menos de 2.4 a 1 en todos los casos), lo que descarta la necesidad de "
            "remuestreo agresivo y favorece class weights y umbral optimizado.",
            "Se descarta SMOTE/ADASYN en la primera iteracion; se usaran pesos de clase y umbral.",
            "E09 con umbral optimizado por etiqueta y validacion anidada.",
        ),
        "",
        "## 2. Longitudes y riqueza lexica",
        "",
        bloque(
            "2.1 Longitud en caracteres por etiqueta",
            "Las longitudes van de 6 a 306 caracteres con mediana cercana a 90; se contrasto si las etiquetas "
            "se asocian con textos mas largos o mas cortos.",
            "H6: la estilometria (incluida la longitud) aporta senal en textos cortos.",
            "Mann-Whitney U bilateral y delta de Cliff sobre `n_caracteres` por etiqueta, con y sin near-duplicados.",
            f"Completo: {resumen_longitud_txt}. Deduplicado por grupo de near-duplicado: {resumen_longitud_dedup}.",
            "Las diferencias existen pero el tamano de efecto es despreciable o pequeno: la longitud por si sola "
            "no separa las etiquetas, aunque los textos con HS y AG tienden a ser algo mas largos.",
            "La longitud se usara solo como feature auxiliar dentro del bloque estilometrico, no como senal primaria.",
            "E06 con el sub-bloque de medidas de forma, incluida longitud normalizada.",
        ),
        bloque(
            "2.2 Riqueza lexica por clase",
            f"HS=1 tiene {int(voc_hs1['tipos'].iloc[0])} tipos y {int(voc_hs1['tokens'].iloc[0])} tokens "
            f"(TTR {voc_hs1['ttr'].iloc[0]:.4f}); HS=0 tiene {int(voc_hs0['tipos'].iloc[0])} tipos y "
            f"{int(voc_hs0['tokens'].iloc[0])} tokens (TTR {voc_hs0['ttr'].iloc[0]:.4f}). TR=1: "
            f"{int(voc_tr1['tipos'].iloc[0])} tipos, TTR {voc_tr1['ttr'].iloc[0]:.4f}; TR=0: "
            f"{int(voc_tr0['tipos'].iloc[0])} tipos, TTR {voc_tr0['ttr'].iloc[0]:.4f}.",
            "H6: si las clases toxicas usaran un vocabulario mas repetitivo, el TTR seria una senal de estilo.",
            "Vocabulario, TTR, hapax y entropia de Shannon por clase y por combinacion.",
            f"Entropia lexica por clase: HS=1 {voc_hs1['entropia_bits'].iloc[0]:.3f} bits frente a HS=0 "
            f"{voc_hs0['entropia_bits'].iloc[0]:.3f} bits; TR=1 {voc_tr1['entropia_bits'].iloc[0]:.3f} frente a "
            f"TR=0 {voc_tr0['entropia_bits'].iloc[0]:.3f} bits.",
            "Las diferencias de entropia son menores a 0.3 bits: el vocabulario toxico no es drasticamente mas "
            "repetitivo, pero el lexico concreto si cambia (ver top-20 y nubes de palabras).",
            "El TTR global no se usara como feature; si se usaran las densidades lexicas por documento.",
            "E06 con densidades y n-gramas de palabras funcionales.",
        ),
        "",
        "## 3. Textometria de superficie",
        "",
        bloque(
            "3.1 Menciones, hashtags y URLs",
            "El 47.5 % de los textos de train contiene al menos una mencion, el 9.3 % un hashtag y el 31.2 % "
            "una URL; las proporciones cambian fuertemente por etiqueta.",
            "H3: las menciones y hashtags son marcadores de targeting.",
            "Conteo por texto y contraste chi cuadrado con V de Cramer por etiqueta.",
            f"Fenomenos mas asociados con cada etiqueta (conjunto completo, q<0.05): {resumen_top_fenomenos(tests_categoricos, 'TR', 6)}.",
            "La mencion inicial es el fenomeno mas asociado con TR: las respuestas a un usuario concentran la "
            "etiqueta. Las URLs se asocian al regimen de noticia (HS=1 con TR=0).",
            "Las menciones, hashtags y URLs entran como features de targeting en E03 y como features "
            "estructurales en E06.",
            "E03 con los seis sub-bloques de targeting ablacionados.",
        ),
        bloque(
            "3.2 Emojis, emoticones y risas",
            f"El corpus contiene {n_emojis} emojis Unicode en train (y 0 en test): la normalizacion ASCII de los "
            f"organizadores elimino los emojis. Si hay {n_emoticones} textos con emoticones ASCII y {n_risas} "
            "con risas del tipo jaja/lol.",
            "H6: la intensidad afectiva suele manifestarse en emojis y risas.",
            "Conteo con la libreria `emoji` y patrones ASCII; boxplots por clase.",
            f"Emojis: {n_emojis}. Emoticones: {n_emoticones} textos. Risas: {n_risas} textos.",
            "La senal de emoji es inexistente por construccion del corpus (hallazgo relevante y no obvio): "
            "cualquier hipotesis basada en emojis es inaplicable aqui. Los emoticones y las risas si existen y "
            "se conservan como features de estilo.",
            "Se elimina la familia de features de emoji del plan; se mantienen emoticonos y risas.",
            "Documentar en el notebook que el corpus es ASCII puro (D-010) y que esa via esta cerrada.",
        ),
        bloque(
            "3.3 Elongaciones, repeticion y duplicacion",
            "Existen letras repetidas (por ejemplo 'siiii'), signos repetidos ('!!!') y palabras duplicadas "
            "adyacentes ('que que').",
            "H1 y H6: la evasion ortografica y la intensidad se manifiestan en repeticiones que los n-gramas de "
            "caracter capturan mejor que los de palabra.",
            "Conteo de patrones de repeticion por documento y contraste por etiqueta.",
            f"Fenomenos mas asociados con AG: {resumen_top_fenomenos(tests_categoricos, 'AG', 6)}.",
            "Las repeticiones acompanan sobre todo a AG (intensidad expresiva), con efectos pequenos pero "
            "estadisticamente solidos, y son casi independientes de TR.",
            "Se conservan como features de estilo dentro de E06 y como justificacion de H1.",
            "E01 (char frente a word n-grams) y E06 (estilometria).",
        ),
        "",
        "## 4. Estadistica con tamano de efecto",
        "",
        bloque(
            "4.1 Contraste de fenomenos binarios contra cada etiqueta",
            "Se contrastaron los fenomenos de superficie binarios contra HS, TR y AG.",
            "Pregunta explicita por fenomeno: es la presencia del fenomeno independiente de la etiqueta?",
            "Chi cuadrado con V de Cramer corregida por sesgo; Fisher exacto si alguna frecuencia esperada < 5; "
            "diferencia de proporciones con intervalo de Newcombe; correccion de Benjamini-Hochberg por familia "
            "(conjunto x etiqueta).",
            f"Informacion mutua mas alta por etiqueta: {mi_txt}.",
            "Los efectos mas grandes son de targeting (mencion, mencion inicial) y de lexico agresivo; los "
            "efectos de estilo son pequenos pero sistematicos.",
            "Se priorizan las familias de features con V de Cramer alto y q<0.05.",
            "E03 (targeting) y E02 (lexico agresivo) antes que E06 (estilo).",
        ),
        bloque(
            "4.2 Control por near-duplicados",
            "La auditoria de Fase 0 encontro 4 grupos de near-duplicados con 17 filas afectadas entre train y test.",
            "Si los near-duplicados inflan la significancia, los efectos deben atenuarse al deduplicar.",
            "Repeticion de todos los contrastes conservando una sola fila por grupo de near-duplicado.",
            f"Informacion mutua ajustada (completo -> deduplicado): {texto_mi_dedup}. V de Cramer para TR: {texto_efecto_dedup}.",
            "El orden de importancia de los fenomenos no cambia y las magnitudes se mantienen: la deduplicacion "
            f"(que solo elimina {n_filas_deduplicadas} filas de train) no altera las conclusiones, lo que confirma que las senales no "
            "provienen de los near-duplicados.",
            "Los tests sobre el conjunto completo se reportan como principales y la version deduplicada como control.",
            "Mantener `StratifiedGroupKFold` con los grupos de near-duplicado en la validacion (D-004).",
        ),
        bloque(
            "4.3 Variables continuas y combinaciones",
            f"Contraste de variables continuas por etiqueta y por combinacion HS x TR x AG. "
            f"Combinaciones mas discriminadas: {texto_combos}.",
            "Pregunta: que metricas continuas distinguen las etiquetas y las seis combinaciones?",
            "Mann-Whitney U con delta de Cliff para 2 grupos; Kruskal-Wallis con epsilon cuadrado para las seis "
            "combinaciones.",
            f"Variables mas discriminantes por combinacion: {texto_combos}.",
            "Las variables de superficie separan mejor las combinaciones que las etiquetas sueltas, lo que "
            "confirma que el espacio de etiquetas tiene estructura (dos regimenes) y no es ruido independiente.",
            "Se usaran features de superficie en el bloque estilometrico, sin esperar gran aporte individual.",
            "E06 con ablacion de sub-bloques y E04 para medir la union con n-gramas.",
        ),
        "",
        "## 5. Caracterizacion de TR",
        "",
        bloque(
            "5.1 Rasgos que separan TR dentro de HS=1",
            f"Dentro de HS=1 (n={conclusion['n_hs1']}: TR=1 {conclusion['n_tr1']}, TR=0 {conclusion['n_tr0']}), "
            "los textos con TR=1 parecen respuestas directas a un usuario y los TR=0 parecen noticia o mencion "
            "de grupo.",
            "H3: el targeting es aprendible con features de superficie; D-005: la semantica de la etiqueta esta "
            "en discusion.",
            "Contraste chi cuadrado con V de Cramer, diferencia de proporciones con IC95 de Newcombe, "
            "Mann-Whitney con delta de Cliff; solo train y solo HS=1.",
            f"Rasgos categoricos: {texto_top_tr}. Rasgos continuos: {texto_top_tr_cont}.",
            f"TR=1 tiene mencion inicial en el {conclusion['v_mencion_inicial']:.3f} de V de Cramer y referencia "
            f"colectiva en solo el {conclusion['p_tr1_con_colectivo'] * 100:.1f} % de los casos, mientras que "
            f"TR=0 tiene referencia colectiva en el {conclusion['p_tr0_con_colectivo'] * 100:.1f} %.",
            "EVIDENCIA PRELIMINAR, DECISION PENDIENTE DE REVISION HUMANA (D-005). Ver 5.3.",
            "Carlos revisa `reports/eda/tr_ejemplos_para_revision.csv` y confirma la lectura de TR.",
        ),
        bloque(
            "5.2 Tabla cualitativa para revision humana",
            "Se generaron 40 ejemplos TR=1 y 40 TR=0 dentro de HS=1, con anotacion automatica de rasgos.",
            "D-005 exige revision humana antes de optimizar TR.",
            "Muestreo aleatorio con semilla fija 42 y anotacion de mencion, 2a persona, imperativo, insulto, "
            "colectivo y nombre propio.",
            "Artefacto: `reports/eda/tr_ejemplos_para_revision.csv` con columnas "
            "`perfil_automatico`, `rasgos_detectados` y tres columnas vacias para el anotador humano.",
            f"Perfiles automaticos: TR=1 con mencion inicial y TR=0 con referencia colectiva "
            f"({conclusion['n_tr0_con_colectivo']} casos TR=0 con colectivo explicito en todo HS=1).",
            "El material de revision esta listo; no se decide nada automaticamente.",
            "Carlos completa `revisado_por_humano` y `etiqueta_sugerida_humano`.",
        ),
        bloque(
            "5.3 Lectura de TR: individuo frente a grupo, o dirigido frente a no dirigido",
            f"TR=1 se concentra en respuestas directas (menciones) y TR=0 en textos con referencia colectiva "
            f"({conclusion['p_tr0_con_colectivo'] * 100:.1f} % frente a {conclusion['p_tr1_con_colectivo'] * 100:.1f} %).",
            "Que lectura del enunciado es consistente con los datos?",
            "Contraste de rasgos de targeting y de referencia colectiva dentro de HS=1, mas inspeccion de los "
            "casos TR=0 con referencia colectiva explicita.",
            f"V de Cramer: mencion inicial {conclusion['v_mencion_inicial']:.3f}, mencion {conclusion['v_mencion']:.3f}, "
            f"referencia colectiva {conclusion['v_referencia_colectiva']:.3f}, insulto {conclusion['v_insulto']:.3f}, "
            f"insulto dirigido {conclusion['v_insulto_dirigido']:.3f}, segunda persona {conclusion['v_segunda_persona']:.3f}, "
            f"imperativo {conclusion['v_imperativo']:.3f}, URL {conclusion['v_url']:.3f}.",
            "Ver la seccion de evidencia detallada mas abajo (5.4) y la tabla de casos para revision.",
            "PENDIENTE DE REVISION HUMANA (D-005): no se decide unilateralmente.",
            "Revision manual de la tabla de ejemplos y de los casos TR=0 con colectivo antes de fijar la "
            "estrategia de E03.",
        ),
        "",
        "### 5.4 Evidencia detallada sobre TR",
        "",
        "**Evidencia a favor de la lectura 'rango del objetivo (individuo frente a grupo)'**",
        "",
        (
            f"1. `TR=1` es casi siempre una respuesta directa a un usuario: mencion inicial presente en el "
            f"{conclusion['p_tr1_con_mencion_inicial'] * 100:.1f} % de los TR=1 frente al "
            f"{conclusion['p_tr0_con_mencion_inicial'] * 100:.1f} % de los TR=0 (V de Cramer "
            f"{conclusion['v_mencion_inicial']:.3f})."
        ),
        (
            f"2. `TR=1` usa segunda persona y lexico agresivo dirigido (V de Cramer "
            f"{conclusion['v_segunda_persona']:.3f} y {conclusion['v_insulto_dirigido']:.3f}), lo que caracteriza "
            "agresion interpersonal."
        ),
        (
            f"3. `TR=0` dentro de HS=1 contiene referencias colectivas explicitas en el "
            f"{conclusion['p_tr0_con_colectivo'] * 100:.1f} % de los casos frente al "
            f"{conclusion['p_tr1_con_colectivo'] * 100:.1f} % de los TR=1 (V de Cramer "
            f"{conclusion['v_referencia_colectiva']:.3f}), y concentra las URLs "
            f"(V de Cramer {conclusion['v_url']:.3f}): el perfil es 'odio sobre un grupo' o noticia, no 'ataque a "
            "una persona'."
        ),
        "",
        "**Evidencia a favor de la lectura 'dirigido frente a no dirigido' (definicion del enunciado)**",
        "",
        (
            f"1. El {conclusion['n_tr0_con_colectivo_y_mencion_inicial']} de los casos TR=0 con referencia colectiva "
            "tambien tiene mencion inicial: parte de ellos podria ser discurso dirigido a un usuario que menciona a "
            "un grupo, en cuyo caso la definicion del enunciado ('dirigido contra una persona o grupo') los haria TR=1."
        ),
        (
            f"2. Solo el {conclusion['proporcion_tr1_en_hs1'] * 100:.1f} % de los textos con HS=1 tiene TR=1: bajo "
            "la lectura literal 'dirigido contra una persona o grupo' el valor esperado seria cercano al 100 %, salvo "
            "que exista una clase de odio no dirigido (noticia, cita, comentario general)."
        ),
        "",
        "**Conflictos y limites de la evidencia**",
        "",
        (
            "- Las dos lecturas predicen la misma direccion para menciones y segunda persona, por lo que la mencion "
            "no puede discriminar entre ellas; el discriminador real es el tratamiento de los textos con objetivo "
            "colectivo explicito."
        ),
        (
            "- La distincion empirica entre 'dirigido a un grupo' y 'habla sobre un grupo' no es separable con "
            "features de superficie y requiere juicio humano sobre los ejemplos."
        ),
        (
            "- La lista de casos TR=0 con referencia colectiva esta en "
            "`reports/eda/fase2_tr0_con_referencia_colectiva.csv` (top 40 por numero de colectivos detectados) para "
            "que Carlos decida caso por caso."
        ),
        "",
        "",
        "## 6. Estructura de etiquetas",
        "",
        bloque(
            "6.1 Asociacion entre etiquetas y con la longitud",
            f"Se calculo phi entre las tres etiquetas y su asociacion con la longitud: "
            f"HS-TR phi={rho_tr_hs['phi'].iloc[0]:.3f}, HS-AG phi={rho_hs_ag['phi'].iloc[0]:.3f}, "
            f"TR-AG phi={rho_tr_ag['phi'].iloc[0]:.3f}.",
            "H8: la jerarquia estructural debe modelarse explicitamente.",
            "Coeficiente phi y V de Cramer sobre tablas 2x2, mas correlacion de Spearman con longitudes.",
            f"HS-TR phi={rho_tr_hs['phi'].iloc[0]:.3f} (V={rho_tr_hs['v_cramer'].iloc[0]:.3f}), "
            f"HS-AG phi={rho_hs_ag['phi'].iloc[0]:.3f} (V={rho_hs_ag['v_cramer'].iloc[0]:.3f}), "
            f"TR-AG phi={rho_tr_ag['phi'].iloc[0]:.3f} (V={rho_tr_ag['v_cramer'].iloc[0]:.3f}).",
            "HS-TR tiene la asociacion mas fuerte por construccion (TR=1 implica HS=1); la asociacion HS-AG es "
            "fuerte; TR-AG es la mas debil. Las longitudes apenas correlacionan con las etiquetas.",
            "El modelado debe respetar la jerarquia y explotar la correlacion HS-AG; la longitud no necesita "
            "tratamiento especial.",
            "E08 (post-proceso) y E04 (union de representaciones) con estas tres etiquetas.",
        ),
        "",
        "## 7. Sintesis: senales prometedoras e hipotesis",
        "",
        "### 7.1 Senales prometedoras por etiqueta",
        "",
        f"- **HS**: {resumen_top_fenomenos(tests_categoricos, 'HS', 6)}.",
        f"- **TR**: {resumen_top_fenomenos(tests_categoricos, 'TR', 6)}.",
        f"- **AG**: {resumen_top_fenomenos(tests_categoricos, 'AG', 6)}.",
        "",
        "Ademas, en el contraste por combinacion las variables mas discriminantes son: " + texto_combos + ".",
        "",
        "### 7.2 Estado inicial de las hipotesis H1-H10",
        "",
        "| Hipotesis | Estado inicial en Fase 2 | Evidencia de este EDA |",
        "|---|---|---|",
        (
            "| H1 (char n-grams 3-6 > word n-grams) | Apoyada (indirecta) | Hay elongaciones, repeticiones y "
            "evasion ortografica medibles; hay texto degradado sin diacriticos (D-010), lo que favorece la "
            "representacion por caracteres. No es una prueba directa: se confirma en E01. |"
        ),
        (
            "| H2 (lexico agresivo + 2a persona separa AG) | Apoyada (parcial) | El lexico inducido de train y la "
            "segunda persona muestran asociacion con AG y con HS; falta la ablacion en E02. |"
        ),
        (
            "| H3 (features de targeting separan TR) | Apoyada con reservas | Efectos grandes y significativos de "
            "mencion inicial, segunda persona e insulto dirigido sobre TR dentro de HS=1; el riesgo es que la senal "
            "sea 'es una respuesta' y no 'es individuo'. |"
        ),
        "| H4 (word+char > cada una sola) | No evaluada | Requiere E04; el EDA no la puede decidir. |",
        (
            "| H5 (sentimiento/emocion complementa) | No evaluada (indicio) | Hay risas y emoticones con senal "
            "debil; sin lexico afectivo no se puede medir todavia. |"
        ),
        (
            "| H6 (estilometria aporta en textos cortos) | Apoyada (parcial) | Puntuacion, mayusculas, menciones, "
            "URLs y repeticiones muestran asociaciones significativas con q<0.05 y efectos pequenos a medios. |"
        ),
        (
            "| H7 (lematizacion no aporta) | No evaluada | El corpus sin diacriticos es un indicio a favor; se "
            "prueba en E07. |"
        ),
        (
            "| H8 (post-proceso jerarquico) | Apoyada (estructural) | 0 violaciones de TR=1 con HS=0 y solo 26 "
            "casos con AG=1 y HS=0 (1.5 %). |"
        ),
        (
            "| H9 (umbral por etiqueta != 0.5) | Apoyada (indirecta) | Prevalencias distintas (43.67/29.98/38.24) "
            "con desbalanceo moderado; el umbral merece optimizacion. |"
        ),
        "| H10 (stacking > mejor individual) | No evaluada | Sin modelos base entrenados. |",
        "",
        "### 7.3 Priorizacion sugerida de experimentos",
        "",
        (
            f"Secuencia sugerida: {seleccionar_experimentos(['E01', 'E04', 'E08', 'E09', 'E03', 'E02', 'E06', 'E05', 'E07', 'E10'])} "
            "(se mantiene la priorizacion de `reports/hipotesis.md`: primero lo de bajo riesgo y alto retorno)."
        ),
        "",
        "## 8. Limitaciones",
        "",
        (
            "1. Los contrastes se calculan sobre train; sus tamanos de efecto son descriptivos y no una estimacion "
            "del rendimiento out-of-sample."
        ),
        (
            "2. El lexico de agresividad y el inventario de colectivos se inducen de train y se evaluan sobre train: "
            "sus tamanos de efecto estan sesgados de forma optimista. E02/E03 deben re-verificarlos con validacion cruzada."
        ),
        (
            "3. Los detectores de imperativo y de vocativo son heuristicos: spaCy asigna `Mood=Imp` a solo 63 tokens "
            "y el modelo no emite la etiqueta de dependencia `vocative`. Se combinan tres detectores y se reporta cada "
            "componente por separado."
        ),
        (
            "4. El NER de spaCy esta deliberadamente desactivado (D-003 no lo autoriza); la deteccion de nombres de "
            "persona usa una heuristica de POS, mayusculas y contexto de segunda persona."
        ),
        (
            "5. Los near-duplicados entre train y test (10 filas de test) hacen que el rendimiento de validacion sea "
            "una cota ligeramente optimista; la particion agrupada lo mitiga."
        ),
        "6. La lectura de TR queda pendiente de revision humana (D-005).",
        "",
        "## Artefactos generados",
        "",
        "- Tablas: `reports/eda/fase2_*.csv` y `reports/eda/tr_ejemplos_para_revision.csv`.",
        "- Figuras: `reports/figuras/fig_*.png`.",
        "- Este informe: `reports/eda_resumen.md`.",
        f"- Total de figuras generadas: {len(contexto['figuras'])}.",
        "",
    ]
    contenido = "\n".join(lineas)
    ruta = REPORTS_DIR / "eda_resumen.md"
    ruta.write_text(contenido, encoding="utf-8")
    return ruta


def agregar_seccion_contexto(contexto: dict) -> Path:
    """Anade la seccion de Fase 2 al final de ``reports/contexto.md``.

    Si la seccion ya existe (re-ejecucion), se reemplaza para mantener el
    script idempotente.

    Args:
        contexto: diccionario con los artefactos del EDA.

    Returns:
        Ruta del archivo actualizado.
    """
    ruta = REPORTS_DIR / "contexto.md"
    encabezado = "## Fase 2 - nlp-specialist (2026-09-22)"
    contenido_previo = ruta.read_text(encoding="utf-8") if ruta.exists() else ""
    if encabezado in contenido_previo:
        contenido_previo = contenido_previo.split(encabezado)[0].rstrip() + "\n"
    conclusion = contexto["conclusion_tr"]
    n_figuras = len(contexto["figuras"])
    seccion = f"""
{encabezado}

EDA profundo, textometria y caracterizacion linguistica de TR. Script `experiments/01_eda_linguistico.py`
(idempotente); salidas en `reports/eda/` ({len(list(EDA_DIR.glob("fase2_*.csv")))} tablas), `reports/figuras/`
({n_figuras} PNG) y `reports/eda_resumen.md` (formato OBSERVACION-HIPOTESIS-METODO-RESULTADO-INTERPRETACION-DECISION-SIGUIENTE PASO).

Hallazgos clave: (1) el corpus no tiene NINGUN emoji Unicode (normalizacion ASCII) y test tiene 6 bytes no ASCII;
(2) la mencion inicial y la segunda persona dominan la asociacion con TR (dentro de HS=1, n=756: TR=1 {conclusion['n_tr1']},
TR=0 {conclusion['n_tr0']}); (3) el odio con objetivo colectivo explicito es casi siempre TR=0
({conclusion['p_tr0_con_colectivo'] * 100:.1f} % de los TR=0 frente a {conclusion['p_tr1_con_colectivo'] * 100:.1f} % de los TR=1),
y esas filas concentran las URLs; (4) prevalencias moderadas (desbalanceo <= 2.4:1), sin necesidad de remuestreo agresivo;
(5) la deduplicacion por near-duplicado no cambia el orden ni la magnitud de los efectos.

TR (D-005): la evidencia es compatible con la lectura HatEval de rango del objetivo (TR=1 = individuo, TR=0 = grupo/generico)
y deja abierta la lectura literal del enunciado. NO se decide: `reports/eda/tr_ejemplos_para_revision.csv` (40 TR=1 + 40 TR=0
con rasgos anotados) y `reports/eda/fase2_tr0_con_referencia_colectiva.csv` (casos limite) quedan para revision humana.

Hipotesis: H1, H3, H6, H8 y H9 reciben apoyo inicial (indirecto o estructural); H4, H5, H7 y H10 quedan sin evaluar.
Riesgo detectado: la senal de TR puede ser principalmente "es una respuesta a un usuario" (mencion inicial V de Cramer
{conclusion['v_mencion_inicial']:.3f}) y no "el objetivo es una persona"; E03 debe ablacionar la mencion para separarlo.
"""
    ruta.write_text(contenido_previo.rstrip() + "\n" + seccion, encoding="utf-8")
    return ruta


def imprimir_resumen(contexto: dict) -> None:
    """Imprime el resumen compacto de hallazgos por consola.

    Args:
        contexto: diccionario con los artefactos del EDA.
    """
    conclusion = contexto["conclusion_tr"]
    etiquetas = contexto["etiquetas"]
    figuras = contexto["figuras"]
    print("\n" + "=" * 78)
    print("FASE 2 - RESUMEN COMPACTO (EDA, textometria y caracterizacion de TR)")
    print("=" * 78)
    for fila in etiquetas.itertuples():
        print(
            f"Etiqueta {fila.etiqueta}: {fila.positivos} positivos "
            f"({fila.prevalencia * 100:.2f} %), desbalanceo {fila.desbalanceo_neg_pos}:1"
        )
    print(
        "Emojis Unicode en train/test: 0 / 0 (normalizacion ASCII); "
        f"emoticones ASCII en {int((contexto['fenomenos']['n_emoticones'] > 0).sum())} textos"
    )
    print(
        "Top senales por etiqueta -> "
        + " | ".join(
            f"{etiqueta}: {resumen_top_fenomenos(contexto['tests_categoricos'], etiqueta, 3)}"
            for etiqueta in ("HS", "TR", "AG")
        )
    )
    print(
        f"TR dentro de HS=1 (n={conclusion['n_hs1']}): mencion inicial V="
        f"{conclusion['v_mencion_inicial']:.3f}, segunda persona V={conclusion['v_segunda_persona']:.3f}, "
        f"insulto dirigido V={conclusion['v_insulto_dirigido']:.3f}, colectivo V="
        f"{conclusion['v_referencia_colectiva']:.3f}, URL V={conclusion['v_url']:.3f}"
    )
    print(
        f"TR=0 con referencia colectiva: {conclusion['p_tr0_con_colectivo'] * 100:.1f} % "
        f"({conclusion['n_tr0_con_colectivo']} casos en HS=1) frente a TR=1: "
        f"{conclusion['p_tr1_con_colectivo'] * 100:.1f} %"
    )
    print("Lectura de TR: PENDIENTE DE REVISION HUMANA (D-005); evidencia en reports/eda_resumen.md seccion 5.4")
    print(f"Figuras: {len(figuras)} PNG en reports/figuras/")
    print("Tablas: reports/eda/fase2_*.csv y reports/eda/tr_ejemplos_para_revision.csv")
    print("Informe: reports/eda_resumen.md | Seccion anadida a reports/contexto.md")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Seccion 8 - Orquestacion
# ---------------------------------------------------------------------------
def escribir_tablas(contexto: dict) -> list[Path]:
    """Escribe todas las tablas CSV del EDA en ``reports/eda``.

    Args:
        contexto: diccionario con los artefactos del EDA.

    Returns:
        Lista de rutas escritas.
    """
    tablas = {
        "fase2_distribucion_etiquetas.csv": contexto["etiquetas"],
        "fase2_combinaciones_etiquetas.csv": contexto["combinaciones"],
        "fase2_longitudes_por_clase.csv": contexto["longitudes"],
        "fase2_vocabulario_por_clase.csv": contexto["vocabulario"],
        "fase2_textometria_por_clase.csv": contexto["textometria"],
        "fase2_top_palabras_por_clase.csv": contexto["top_palabras"],
        "fase2_puntuacion_ngramas.csv": contexto["ngramas_puntuacion"],
        "fase2_inventario_caracteres.csv": contexto["inventario"],
        "fase2_tests_categoricos.csv": contexto["tests_categoricos"],
        "fase2_tests_continuos.csv": contexto["tests_continuos"],
        "fase2_tests_combinaciones.csv": contexto["tests_combos"],
        "fase2_info_mutua.csv": contexto["info_mutua"],
        "fase2_correlaciones_etiquetas.csv": contexto["correlaciones"],
        "fase2_lexico_inducido_agresividad.csv": contexto["lexico_agresivo"],
        "fase2_colectivos_detectados.csv": contexto["colectivos"],
        "fase2_rasgos_TR.csv": contexto["tr_categorico"],
        "fase2_rasgos_TR_continuos.csv": contexto["tr_continuo"],
        "fase2_tr0_con_referencia_colectiva.csv": contexto["tr0_colectivo"],
        "fase2_resumen_tr_semantica.csv": pd.DataFrame([contexto["conclusion_tr"]]),
        "fase2_hallazgos_clave.csv": contexto["hallazgos"],
        "fase2_perfil_test.csv": contexto["perfil_test"],
    }
    rutas = []
    for nombre, tabla in tablas.items():
        ruta = EDA_DIR / nombre
        tabla.to_csv(ruta, index=False)
        rutas.append(ruta)
    ruta_ejemplos = EDA_DIR / "tr_ejemplos_para_revision.csv"
    contexto["ejemplos_tr"].to_csv(ruta_ejemplos, index=False)
    rutas.append(ruta_ejemplos)
    return rutas


def perfil_test(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Compara la distribucion de fenomenos de superficie entre train y test.

    Solo describe el test; no se usa para ajustar nada.

    Args:
        train: conjunto de entrenamiento.
        test: conjunto de test.

    Returns:
        Tabla comparativa con medias por conjunto y prueba Mann-Whitney.
    """
    train_metricas = extraer_fenomenos_superficie(train["text"])
    test_metricas = extraer_fenomenos_superficie(test["text"])
    variables = [
        "n_caracteres", "n_palabras", "n_menciones", "n_hashtags", "n_urls",
        "ratio_mayusculas", "ratio_puntuacion", "n_exclamaciones", "n_emoticones",
    ]
    filas = []
    for variable in variables:
        estadistico, p_valor = mannwhitneyu(
            train_metricas[variable], test_metricas[variable], alternative="two-sided"
        )
        filas.append(
            {
                "variable": variable,
                "media_train": round(float(train_metricas[variable].mean()), 4),
                "media_test": round(float(test_metricas[variable].mean()), 4),
                "mediana_train": round(float(train_metricas[variable].median()), 4),
                "mediana_test": round(float(test_metricas[variable].median()), 4),
                "delta_cliff": round(
                    cliff_delta(train_metricas[variable], test_metricas[variable]), 4
                ),
                "u_mannwhitney": float(estadistico),
                "p_valor": float(p_valor),
            }
        )
    tabla = pd.DataFrame(filas)
    tabla["p_valor_bh"] = benjamini_hochberg(tabla["p_valor"].tolist()).round(6)
    tabla["descripcion"] = "test: solo descriptivo, sin etiquetas y sin uso para ajustar"
    return tabla


def construir_hallazgos(contexto: dict) -> pd.DataFrame:
    """Construye la tabla de hallazgos clave con su trazabilidad.

    Args:
        contexto: diccionario con los artefactos del EDA.

    Returns:
        Tabla con hallazgo, evidencia, artefacto e hipotesis relacionada.
    """
    conclusion = contexto["conclusion_tr"]
    return pd.DataFrame(
        [
            {
                "hallazgo": "El corpus no contiene emojis Unicode (normalizacion ASCII).",
                "evidencia": "0 emojis en train y en test con la libreria emoji; 6 bytes no ASCII en test.",
                "artefacto": "reports/eda/fase2_textometria_por_clase.csv",
                "hipotesis": "H6 (descarta la subfamilia de emojis)",
            },
            {
                "hallazgo": "La mencion inicial es el fenomeno mas asociado con TR dentro de HS=1.",
                "evidencia": f"V de Cramer {conclusion['v_mencion_inicial']:.3f}",
                "artefacto": "reports/eda/fase2_rasgos_TR.csv",
                "hipotesis": "H3",
            },
            {
                "hallazgo": "El odio con objetivo colectivo explicito es casi siempre TR=0.",
                "evidencia": (
                    f"{conclusion['p_tr0_con_colectivo'] * 100:.1f} % de TR=0 frente a "
                    f"{conclusion['p_tr1_con_colectivo'] * 100:.1f} % de TR=1"
                ),
                "artefacto": "reports/eda/fase2_tr0_con_referencia_colectiva.csv",
                "hipotesis": "D-005 (semantica de TR)",
            },
            {
                "hallazgo": "La deduplicacion por near-duplicado no altera el orden de los efectos.",
                "evidencia": "mismas magnitudes y significancias en los tests con y sin deduplicar",
                "artefacto": "reports/eda/fase2_tests_categoricos.csv",
                "hipotesis": "Transversal (D-004)",
            },
            {
                "hallazgo": "Desbalanceo moderado en las tres etiquetas (maximo 2.34:1).",
                "evidencia": "prevalencias 43.67 %, 29.98 % y 38.24 %",
                "artefacto": "reports/eda/fase2_distribucion_etiquetas.csv",
                "hipotesis": "H9",
            },
            {
                "hallazgo": "Existe estructura jerarquica fuerte: 0 violaciones de TR=1 con HS=0.",
                "evidencia": "519 de 519 casos cumplen TR=1 => HS=1; 26 casos con AG=1 y HS=0",
                "artefacto": "reports/eda/fase2_combinaciones_etiquetas.csv",
                "hipotesis": "H8",
            },
        ]
    )


def main() -> None:
    """Ejecuta el EDA completo de Fase 2 y escribe todos los artefactos."""
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    configurar_estilo()

    log("seccion 1: carga e integridad")
    train, test, grupos = cargar_conjuntos()
    informe = verificar_integridad(train, test)
    log(f"integridad verificada: {informe['train_filas']} train, {informe['test_filas']} test")

    log("seccion 2: EDA por etiqueta y combinacion")
    etiquetas = tabla_etiquetas(train)
    combinaciones = tabla_combinaciones(train)
    longitud_train = metricas_longitud(train["text"])
    longitud_test = metricas_longitud(test["text"])
    longitudes = resumen_longitudes(train, longitud_train)

    log("seccion 3: textometria de superficie y rasgos linguisticos")
    nlp = spacy.load("es_core_news_md", exclude=["ner"])
    superficie_train = extraer_fenomenos_superficie(train["text"])
    superficie_test = extraer_fenomenos_superficie(test["text"])
    palabras_frecuentes = palabras_frecuentes_corpus(train["text"])
    log("spaCy: anotando train")
    rasgos_train = extraer_rasgos_linguisticos(
        nlp, train["text"].tolist(), palabras_frecuentes
    )
    log("spaCy: anotando test")
    rasgos_test = extraer_rasgos_linguisticos(
        nlp, test["text"].tolist(), palabras_frecuentes
    )
    lemas = rasgos_train["lemas_documento"].tolist()

    log("seccion 3: induccion de lexicos desde train (sin recursos externos)")
    lexico_agresivo_tabla = inducir_lexico_agresivo(train, lemas)
    lexico_agresivo = frozenset(
        lexico_agresivo_tabla.loc[lexico_agresivo_tabla["seleccionado"] == 1, "lema"].tolist()
    )
    colectivos_tabla = inducir_lexico_colectivos(rasgos_train)
    lemas_colectivos = frozenset(colectivos_tabla["lema"].tolist())

    # Las metricas lexicas aportan columnas que no estan en el extractor de
    # superficie (palabras unicas, TTR por documento, oraciones).
    metricas_lexicas = longitud_train.drop(
        columns=["n_caracteres", "n_palabras"], errors="ignore"
    )
    fenomenos = pd.concat(
        [
            train.reset_index(drop=True),
            superficie_train,
            rasgos_train,
            metricas_lexicas,
        ],
        axis=1,
    )
    fenomenos_test = pd.concat(
        [test.reset_index(drop=True), superficie_test, rasgos_test], axis=1
    )
    sin_etiquetas = [columna for columna in fenomenos.columns if columna not in LABELS]
    aumentado = anadir_rasgos_binarios(fenomenos[sin_etiquetas])
    aumentado = anadir_rasgos_lexicos(aumentado, lexico_agresivo, lemas_colectivos)
    integrar_columnas(fenomenos, aumentado)
    sin_etiquetas_test = [
        columna for columna in fenomenos_test.columns if columna not in LABELS
    ]
    aumentado_test = anadir_rasgos_binarios(fenomenos_test[sin_etiquetas_test])
    aumentado_test = anadir_rasgos_lexicos(
        aumentado_test, lexico_agresivo, lemas_colectivos
    )
    integrar_columnas(fenomenos_test, aumentado_test)
    mascara_train = mascara_deduplicada(grupos, train["id"].tolist())
    log(f"filas de train conservadas al deduplicar: {int(mascara_train.sum())}")

    vocabulario = tabla_vocabulario(train, [list(fila) for fila in lemas])
    textometria = tabla_textometria_por_clase(fenomenos)
    top_palabras = top_palabras_por_clase(train, lemas)
    ngramas_puntuacion_tabla = ngramas_puntuacion(train["text"])
    inventario = inventario_caracteres(pd.concat([train["text"], test["text"]]))

    log("seccion 4: contraste estadistico")
    columnas_binarias = [
        columna
        for columna in fenomenos.columns
        if columna.startswith(("tiene_", "es_mayoria"))
    ]
    columnas_continuas = [
        "n_caracteres",
        "n_palabras",
        "n_palabras_unicas",
        "longitud_media_palabra",
        "ttr_documento",
        "n_oraciones",
        "ratio_mayusculas",
        "ratio_puntuacion",
        "ratio_digitos",
        "ratio_simbolos",
        "n_signos",
        "n_exclamaciones",
        "n_interrogaciones",
        "n_letras_repetidas",
        "n_puntuacion_repetida",
        "n_palabras_duplicadas",
        "n_palabras_mayusculas",
        "n_insultos",
        "n_patron_colectivo",
        "n_propn_candidato",
        "imperativo",
        "pron_2",
        "verbo_2",
    ]
    columnas_continuas = [columna for columna in columnas_continuas if columna in fenomenos.columns]

    registros_categoricos: list[dict] = []
    registros_continuos: list[dict] = []
    registros_combos: list[dict] = []
    registros_mi: list[dict] = []
    for conjunto, datos in (
        ("completo", fenomenos),
        ("deduplicado", fenomenos.loc[mascara_train].reset_index(drop=True)),
    ):
        for etiqueta in LABELS:
            registros_categoricos.extend(
                pruebas_categoricas(datos, etiqueta, columnas_binarias, conjunto)
            )
            registros_continuos.extend(
                pruebas_continuas(datos, etiqueta, columnas_continuas, conjunto)
            )
            registros_mi.extend(tabla_info_mutua(datos, columnas_binarias, conjunto))
        registros_combos.extend(pruebas_por_combinacion(datos, columnas_continuas, conjunto))
    tests_categoricos = aplicar_bh(pd.DataFrame(registros_categoricos))
    tests_continuos = aplicar_bh(pd.DataFrame(registros_continuos))
    tests_combos = aplicar_bh(pd.DataFrame(registros_combos))
    info_mutua = pd.DataFrame(registros_mi)
    correlaciones = correlaciones_etiquetas(train)

    log("seccion 5: caracterizacion de TR dentro de HS=1")
    hs1 = fenomenos[fenomenos["HS"] == 1].copy()
    rasgos_tr_binarios = [
        columna
        for columna in hs1.columns
        if columna.startswith("tiene_") or columna == "imperativo"
    ]
    tr_categorico = rasgos_tr_categoricos(hs1, rasgos_tr_binarios)
    tr_continuo = rasgos_tr_continuos(hs1, columnas_continuas)
    ejemplos_tr = construir_ejemplos_tr(hs1, rasgos_tr_binarios)
    tr0_colectivo = (
        hs1[(hs1["TR"] == 0) & (hs1["tiene_referencia_colectiva"] == 1)]
        .assign(n_colectivos=lambda marco: marco["n_patron_colectivo"] + marco["n_colectivos_lexico"])
        .sort_values(["n_colectivos", "n_caracteres"], ascending=False)
        .head(40)
    )
    columnas_tr0 = [
        "id", "HS", "TR", "AG", "n_caracteres", "text", "n_patron_colectivo",
        "n_colectivos_lexico", "colectivos_documento", "tiene_mencion_inicial",
        "tiene_segunda_persona", "tiene_insulto",
    ]
    tr0_colectivo = tr0_colectivo[[columna for columna in columnas_tr0 if columna in tr0_colectivo.columns]]

    contexto: dict = {
        "train": train,
        "test": test,
        "integridad": informe,
        "etiquetas": etiquetas,
        "combinaciones": combinaciones,
        "longitudes": longitudes,
        "longitud_train": longitud_train,
        "longitud_test": longitud_test,
        "vocabulario": vocabulario,
        "textometria": textometria,
        "top_palabras": top_palabras,
        "ngramas_puntuacion": ngramas_puntuacion_tabla,
        "inventario": inventario,
        "fenomenos": fenomenos,
        "fenomenos_test": fenomenos_test,
        "lemas": lemas,
        "lexico_agresivo": lexico_agresivo_tabla,
        "colectivos": colectivos_tabla,
        "tests_categoricos": tests_categoricos,
        "tests_continuos": tests_continuos,
        "tests_combos": tests_combos,
        "info_mutua": info_mutua,
        "correlaciones": correlaciones,
        "tr_categorico": tr_categorico,
        "tr_continuo": tr_continuo,
        "ejemplos_tr": ejemplos_tr,
        "tr0_colectivo": tr0_colectivo,
        "figuras": [],
        "n_filas_deduplicadas": int((~mascara_train).sum()),
    }
    contexto["conclusion_tr"] = conclusion_tr(tr_categorico, tr_continuo, hs1)
    contexto["conclusion_tr"]["p_tr1_con_mencion_inicial"] = round(
        float((hs1.loc[hs1["TR"] == 1, "tiene_mencion_inicial"] == 1).mean()), 4
    )
    contexto["conclusion_tr"]["p_tr0_con_mencion_inicial"] = round(
        float((hs1.loc[hs1["TR"] == 0, "tiene_mencion_inicial"] == 1).mean()), 4
    )

    log("seccion 6: figuras")
    contexto["figuras"] = generar_figuras(contexto)
    contexto["hallazgos"] = construir_hallazgos(contexto)
    contexto["perfil_test"] = perfil_test(train, test)

    log("seccion 7: tablas y reportes")
    rutas_tablas = escribir_tablas(contexto)
    contexto["tablas"] = rutas_tablas
    escribir_resumen(contexto)
    agregar_seccion_contexto(contexto)

    imprimir_resumen(contexto)


if __name__ == "__main__":
    main()
