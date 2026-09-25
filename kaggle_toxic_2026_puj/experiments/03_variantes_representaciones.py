"""03_variantes_representaciones.py - Fases 4 y 5: preprocesamiento, representaciones y modelos.

Pregunta que responde
---------------------
Con la validacion agrupada por near-duplicado y la metrica oficial
(``f1_media_columnas``, D-009), que variante de preprocesamiento, que
representacion TF-IDF y que clasificador lineal maximizan el F1 en este corpus
de tweets en espanol?

Estructura del barrido (una fila por configuracion en
``reports/variantes_representaciones.csv``)
----------------------------------------------------------------------
- PARTE A (PRE-00..PRE-09): ablacion de transformaciones deterministas del texto
  (case, stopwords, URLs, menciones, hashtags, elongaciones, puntuacion,
  digitos, lematizacion). Modelo y representacion fijos: TF-IDF word (1,2),
  min_df=2, sublinear_tf + LogisticRegression balanced C=1.0 liblinear x3.
  Cada variante aplica UNA transformacion sobre el texto crudo, no son
  acumulativas: la ablacion aísla el efecto de cada decision y las
  combinaciones de las ganadoras se prueban en una fase posterior.
- PARTE B (REP-01..REP-06): representaciones TF-IDF alternativas (char_wb,
  word, union word+char) con el modelo fijo de la Parte A.
- PARTE C (MOD-01..MOD-04): clasificadores alternativos (LinearSVC,
  ComplementNB, LR con C=0.5 y C=2.0) sobre la mejor representacion de la
  Parte B y sobre word (1,2), mas el post-proceso jerarquico (D-001) aplicado a
  la mejor configuracion global. Los ids se desdoblan por representacion
  (sufijos ``_BEST`` y ``_W12``) y MOD-03 por valor de ``C`` (``a``/``b``) para
  mantener ids unicos en ``experiments.csv``.

Protocolo (identico al baseline B1, sin cambios)
-----------------------------------------------
- Validacion: ``StratifiedGroupKFold(5, shuffle=True, random_state=42)`` con
  grupos = componentes de near-duplicado (D-004) y estratificacion por la
  combinacion HS x TR x AG. Una sola particion compartida por todas las
  configuraciones, calculada una vez.
- Sin fuga: todos los vectorizadores se ajustan con el pliegue de entrenamiento
  y las transformaciones de texto son deterministas y sin parametros ajustados
  (D-010), por lo que se precalculan una vez sobre el texto crudo.
- Test nunca se usa: solo se inspecciona para documentar la ausencia de emojis.
- Regla de decision pre-registrada: una variante se adopta solo si mejora
  >= +0.005 absoluto en ``f1_media_columnas`` out-of-fold **y** mejora en >= 3
  de los 5 pliegues respecto de la referencia PRE-00 (= B1).

Idempotencia
------------
Los pliegues, las metricas, las dimensiones y las decisiones son deterministas
con la misma version de las librerias. La unica columna no determinista de
``reports/variantes_representaciones.csv`` es ``tiempo_segundos`` (ruido de
maquina, redondeado a 2 decimales): todo lo demas es reproducible byte a byte,
igual que ``reports/variantes_representaciones.json``.

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/03_variantes_representaciones.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Final

import emoji
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.evaluation import (
    ETIQUETAS,
    MetricasMultiEtiqueta,
    aplicar_postproceso_jerarquico,
    metricas_multietiqueta,
)

CONFIG: Final[dict] = {
    "grupos_path": PROJECT_ROOT / "data" / "processed" / "grupos_near_duplicados.csv",
    "b1_metricas_path": PROJECT_ROOT / "reports" / "baseline_b1_metricas.json",
    "reporte_csv": PROJECT_ROOT / "reports" / "variantes_representaciones.csv",
    "reporte_json": PROJECT_ROOT / "reports" / "variantes_representaciones.json",
    "experiments_csv": PROJECT_ROOT / "experiments.csv",
    "contexto_md": PROJECT_ROOT / "reports" / "contexto.md",
    "random_state": 42,
    "n_splits": 5,
    "umbral_probabilidad": 0.5,
    "referencia_id": "PRE-00",
    "umbral_mejora": 0.005,
    "pliegues_minimos_mejora": 3,
    "spacy_model": "es_core_news_md",
    "momento_contexto": "2026-09-22",
    "fecha": "2026-09-22",
    "script": "experiments/03_variantes_representaciones.py",
}

PARTE_A: Final[str] = "A - preprocesamiento"
PARTE_B: Final[str] = "B - representacion"
PARTE_C: Final[str] = "C - modelo"

# Parametros TF-IDF comunes a todas las variantes. min_df=2 y max_df=1.0
# (sin recorte) replican la decision del baseline B1: con 1.385 documentos de
# entrenamiento por pliegue, min_df=2 elimina hapax sin perder senal y
# sublinear_tf amortigua la repeticion de terminos. strip_accents=None es
# explicito porque el corpus ya esta normalizado a ASCII (D-010) y no se
# "inventan" diacriticos.
TFIDF_PARAMS: Final[dict] = {
    "min_df": 2,
    "max_df": 1.0,
    "sublinear_tf": True,
    "lowercase": True,
    "strip_accents": None,
}

# Stopwords de Twitter que no estan en la lista NLTK (ver `patron_stopwords`):
# rt (retuit), via (procedencia), xq/xd (abreviaturas de "porque"/risa), co,
# http, https, www y amp (restos de URL y de entidades HTML que sobreviven
# cuando la variante PRE-03 no se aplica). Se incluyen ademas q, d, x y t
# (monogramas y abreviaturas de una letra): el patron de tokens por defecto de
# scikit-learn exige >= 2 caracteres, por lo que son inocuos, pero se documentan
# para dejar la lista completa.
STOPWORDS_TWITTER: Final[tuple[str, ...]] = (
    "rt",
    "via",
    "xq",
    "xd",
    "co",
    "http",
    "https",
    "www",
    "amp",
    "q",
    "d",
    "x",
    "t",
)

RE_URL: Final = re.compile(r"(?:https?://|www\.)\S*")
RE_MENCION: Final = re.compile(r"@\w+")
RE_HASHTAG: Final = re.compile(r"#(\w+)")
RE_CAMEL: Final = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
RE_ELONGACION: Final = re.compile(r"(.)\1{2,}")
RE_PUNTUACION: Final = re.compile(r"[^\w\s]")
RE_DIGITOS: Final = re.compile(r"\d+")
RE_ESPACIOS: Final = re.compile(r"\s+")

COLUMNAS_REPORTE: Final[tuple[str, ...]] = (
    "experiment_id",
    "parte",
    "descripcion",
    "representacion",
    "modelo",
    "f1_media_columnas",
    "f1_macro6",
    "f1_micro",
    "f1_samples",
    "exact_match",
    "f1_HS",
    "f1_TR",
    "f1_AG",
    "std_pliegues",
    "mejora_vs_B1",
    "mejora_en_pliegues",
    "dimension",
    "tiempo_segundos",
    "decision",
)

def log(message: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        message: texto a imprimir.
    """
    print(f"[03] {message}", flush=True)


# ===== SECCION 1: carga de datos y particion de validacion =====


def cargar_datos() -> tuple[pd.DataFrame, pd.Series]:
    """Carga el train oficial y la asignacion de grupos de near-duplicado.

    Returns:
        Tupla ``(train, grupos)`` donde ``grupos`` es una serie indexada por
        ``id`` con el numero de grupo de near-duplicado.

    Raises:
        FileNotFoundError: si falta la particion generada en la Fase 0.
    """
    train = data_module.read_train()
    grupos_path: Path = CONFIG["grupos_path"]
    if not grupos_path.exists():
        raise FileNotFoundError(
            f"Falta {grupos_path}. Ejecuta experiments/00_auditoria_datos.py primero."
        )
    grupos = pd.read_csv(grupos_path).set_index("id")["grupo"]
    return train, grupos


def construir_pliegues(
    train: pd.DataFrame, grupos: pd.Series
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construye la particion 5-fold agrupada y estratificada (D-004).

    La particion se calcula una sola vez y se reutiliza en todas las
    configuraciones para que las diferencias de F1 sean atribuibles a la
    variante y no al reparto de los datos.

    Args:
        train: DataFrame de entrenamiento con ``id``, ``text`` y etiquetas.
        grupos: serie de grupos de near-duplicado indexada por ``id``.

    Returns:
        Lista de 5 tuplas ``(indices_entrenamiento, indices_validacion)`` con
        indices posicionales.

    Raises:
        ValueError: si algun ``id`` de train no tiene grupo asignado.
    """
    combinacion = (
        train["HS"].astype(str) + train["TR"].astype(str) + train["AG"].astype(str)
    )
    asignacion = train["id"].astype(int).map(grupos)
    if asignacion.isna().any():
        faltantes = int(asignacion.isna().sum())
        raise ValueError(
            f"{faltantes} ids de train sin grupo de near-duplicado; "
            "regenera data/processed/grupos_near_duplicados.csv"
        )
    divisor = StratifiedGroupKFold(
        n_splits=CONFIG["n_splits"],
        shuffle=True,
        random_state=CONFIG["random_state"],
    )
    pliegues = [
        (entrenamiento, validacion)
        for entrenamiento, validacion in divisor.split(
            train, combinacion, groups=asignacion.to_numpy()
        )
    ]
    for numero, (entrenamiento, validacion) in enumerate(pliegues, start=1):
        log(
            f"pliegue {numero}: {len(entrenamiento)} train / {len(validacion)} validacion "
            f"| grupos validacion {asignacion.iloc[validacion].nunique()}"
        )
    return pliegues


def recolectar_evidencia(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Documenta la evidencia sobre emojis y caracteres no ASCII del corpus.

    Motivo: la hipotesis H6 y la lista de variantes A-L incluyen emojis; si el
    corpus no tiene ninguno, esa variante no aplica y hay que demostrarlo con un
    conteo reproducible. El test solo se INSPECCIONA (nunca se usa para
    entrenar, ajustar ni decidir).

    Args:
        train: DataFrame de entrenamiento con la columna ``text``.
        test: DataFrame de test con la columna ``text`` (solo inspeccion).

    Returns:
        Diccionario con los conteos y la conclusion.
    """
    textos_train = list(train["text"])
    textos_test = list(test["text"])
    no_ascii_test = [
        f"U+{ord(caracter):04X}"
        for texto in textos_test
        for caracter in texto
        if ord(caracter) > 127
    ]
    return {
        "n_emojis_train": int(sum(emoji.emoji_count(texto) for texto in textos_train)),
        "n_emojis_test": int(sum(emoji.emoji_count(texto) for texto in textos_test)),
        "n_caracteres_no_ascii_train": int(
            sum(1 for texto in textos_train for caracter in texto if ord(caracter) > 127)
        ),
        "n_caracteres_no_ascii_test": len(no_ascii_test),
        "caracteres_no_ascii_test": {
            codigo: no_ascii_test.count(codigo) for codigo in sorted(set(no_ascii_test))
        },
        "conclusion": (
            "El corpus no contiene ningun emoji Unicode (0 en train y 0 en test); "
            "las variantes de emojis no aplican. Train es 100 % ASCII (D-010) y el "
            "test solo contiene el caracter '?' (U+00BF) heredado del cp1252."
        ),
    }


# ===== SECCION 2: variantes de preprocesamiento (Parte A) =====


def _colapsar_espacios(texto: str) -> str:
    """Reduce secuencias de espacios a uno solo y recorta los extremos.

    Args:
        texto: texto de entrada.

    Returns:
        Texto con espacios normalizados.
    """
    return RE_ESPACIOS.sub(" ", texto).strip()


def normalizar_urls(texto: str) -> str:
    """Sustituye URLs (http, https o www) por el token generico ``url``.

    Args:
        texto: texto de entrada.

    Returns:
        Texto con las URLs enmascaradas como ``url``.
    """
    return _colapsar_espacios(RE_URL.sub(" url ", texto))


def normalizar_menciones(texto: str) -> str:
    """Sustituye menciones ``@usuario`` por el token generico ``usuario``.

    Anonimizar las menciones evita que el modelo memorice nombres propios de
    pseudo-autores y deja solo la senal estructural "va dirigido a alguien".

    Args:
        texto: texto de entrada.

    Returns:
        Texto con las menciones enmascaradas como ``usuario``.
    """
    return _colapsar_espacios(RE_MENCION.sub(" usuario ", texto))


def partir_hashtags(texto: str) -> str:
    """Separa los hashtags en sus palabras conservando el contenido lexico.

    Se elimina ``#`` y se insertan espacios en las fronteras camelCase y en los
    guiones bajos (``#OdioALosInmigrantes`` -> ``Odio A Los Inmigrantes``). Un
    hashtag todo en minusculas (``#inmigrantes``) no tiene fronteras y se
    conserva como una sola palabra.

    Args:
        texto: texto de entrada.

    Returns:
        Texto con los hashtags descompuestos en palabras.
    """

    def _expandir(coincidencia: re.Match[str]) -> str:
        palabra = coincidencia.group(1).replace("_", " ")
        return f" {RE_CAMEL.sub(' ', palabra)} "

    return _colapsar_espacios(RE_HASHTAG.sub(_expandir, texto))


def colapsar_elongaciones(texto: str) -> str:
    """Reduce repeticiones de 3 o mas caracteres identicos a dos.

    Convierte alargamientos voca­licos y de signos (``jajajajaja``,
    ``noooooo``, ``!!!!``) a una forma canonica corta sin perder la senal de
    enfasis (se conserva la repeticion doble).

    Args:
        texto: texto de entrada.

    Returns:
        Texto con las elongaciones colapsadas.
    """
    return RE_ELONGACION.sub(r"\1\1", texto)


def quitar_puntuacion(texto: str) -> str:
    """Elimina los signos de puntuacion y deja solo letras, digitos y espacios.

    Args:
        texto: texto de entrada.

    Returns:
        Texto sin signos de puntuacion.
    """
    return _colapsar_espacios(RE_PUNTUACION.sub(" ", texto))


def normalizar_digitos(texto: str) -> str:
    """Sustituye cualquier secuencia de digitos por el token ``num``.

    Args:
        texto: texto de entrada.

    Returns:
        Texto con los digitos enmascarados como ``num``.
    """
    return _colapsar_espacios(RE_DIGITOS.sub(" num ", texto))


@lru_cache(maxsize=1)
def cargar_spacy():
    """Carga el modelo linguistico de spaCy sin el componente NER.

    Se usa exclusivamente como anotador linguistico: solo se leen
    ``token.lemma_`` y ``token.is_space``. Prohibido ``doc.vector``,
    ``token.vector`` y ``doc.tensor`` (D-003: los vectores son embeddings
    preentrenados y estan vetados por las reglas de la competencia).

    Returns:
        Pipeline de spaCy con ``es_core_news_md`` y ``ner`` excluido.

    Raises:
        OSError: si el modelo no esta instalado en el entorno.
    """
    import spacy

    return spacy.load(CONFIG["spacy_model"], exclude=["ner"])


def lematizar(textos: list[str]) -> list[str]:
    """Lematiza cada texto con spaCy (sin vectores, D-003).

    Args:
        textos: lista de textos crudos.

    Returns:
        Lista de textos donde cada palabra se sustituyo por su lema.
    """
    procesador = cargar_spacy()
    return [
        " ".join(
            token.lemma_ for token in documento if not token.is_space and token.lemma_
        )
        for documento in procesador.pipe(textos, batch_size=32)
    ]


def _por_texto(transformacion: Callable[[str], str]) -> Callable[[list[str]], list[str]]:
    """Adapta una transformacion de un texto a una transformacion por lote.

    Args:
        transformacion: funcion que transforma un unico texto.

    Returns:
        Funcion que aplica ``transformacion`` a cada elemento de la lista.
    """
    return lambda textos: [transformacion(texto) for texto in textos]


def _identidad(textos: list[str]) -> list[str]:
    """Devuelve los textos sin cambios (variante de referencia).

    Args:
        textos: lista de textos.

    Returns:
        La misma lista de textos.
    """
    return list(textos)


@lru_cache(maxsize=1)
def patron_stopwords() -> re.Pattern[str]:
    """Construye el patron de stopwords del espanol normalizadas y de Twitter.

    La lista NLTK de espanol trae diacriticos (``mas``, ``el``, ``esta``) que no
    aparecen en este corpus (D-010), por lo que se normaliza con
    :func:`src.data.strip_accents` y se deduplica antes de compilar el patron.
    Se anaden las stopwords de Twitter documentadas en ``STOPWORDS_TWITTER``.
    Se documenta como decision explicita que la lista NLTK incluye ``no``, ``si``
    y ``ya``: eliminarlas puede afectar la negacion y el resultado de esta
    variante lo mide.

    Returns:
        Patron compilado que reconoce cualquier stopword como palabra completa.
    """
    from nltk.corpus import stopwords

    normalizadas = {
        data_module.strip_accents(palabra).lower()
        for palabra in stopwords.words("spanish")
    }
    vocabulario = sorted(normalizadas.union(STOPWORDS_TWITTER), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(p) for p in vocabulario) + r")\b")


def quitar_stopwords(texto: str) -> str:
    """Elimina las stopwords del espanol y de Twitter del texto.

    Args:
        texto: texto de entrada.

    Returns:
        Texto sin stopwords.
    """
    return _colapsar_espacios(patron_stopwords().sub(" ", texto))


@dataclass(frozen=True)
class VariantePreprocesamiento:
    """Especificacion de una variante de preprocesamiento (Parte A).

    Attributes:
        experiment_id: identificador del experimento (PRE-00..PRE-09).
        descripcion: descripcion corta de la transformacion.
        transformacion: funcion que aplica la variante a una lista de textos.
        lowercase: valor de ``lowercase`` que se pasa al vectorizador TF-IDF.
        notas: nota metodologica para ``experiments.csv``.
    """

    experiment_id: str
    descripcion: str
    transformacion: Callable[[list[str]], list[str]]
    lowercase: bool = True
    notas: str = ""


VARIANTES_PREPROCESAMIENTO: Final[tuple[VariantePreprocesamiento, ...]] = (
    VariantePreprocesamiento(
        "PRE-00",
        "texto crudo (referencia = baseline B1)",
        _identidad,
        notas="Configuracion de referencia; debe reproducir el B1 registrado.",
    ),
    VariantePreprocesamiento(
        "PRE-01",
        "sin lowercase: se preserva el uso de mayusculas/minusculas",
        _identidad,
        lowercase=False,
        notas="El tensor de mayusculas es senal de enfasis (H6); el vectorizador recibe lowercase=False.",
    ),
    VariantePreprocesamiento(
        "PRE-02",
        "stopwords eliminadas (NLTK espanol sin diacriticos + lista de Twitter)",
        _por_texto(quitar_stopwords),
        notas="La lista NLTK se normaliza con strip_accents porque el corpus no tiene tildes (D-010).",
    ),
    VariantePreprocesamiento(
        "PRE-03",
        "URLs sustituidas por el token 'url'",
        _por_texto(normalizar_urls),
        notas="Reduce vocabulario de enlaces acortados y conserva la senal de cita de fuente.",
    ),
    VariantePreprocesamiento(
        "PRE-04",
        "menciones @usuario sustituidas por el token 'usuario'",
        _por_texto(normalizar_menciones),
        notas="Anonimiza pseudo-autores y conserva la senal estructural de direccion (H3).",
    ),
    VariantePreprocesamiento(
        "PRE-05",
        "hashtags partidos en camelCase conservando las palabras",
        _por_texto(partir_hashtags),
        notas="El corpus tiene 161 documentos con hashtag; los compuestos camelCase se separan.",
    ),
    VariantePreprocesamiento(
        "PRE-06",
        "elongaciones colapsadas: (.)\\1{2,} -> dos repeticiones",
        _por_texto(colapsar_elongaciones),
        notas="Normaliza alargamientos vocalicos y de signos conservando la repeticion doble.",
    ),
    VariantePreprocesamiento(
        "PRE-07",
        "puntuacion eliminada (solo letras, digitos y espacios)",
        _por_texto(quitar_puntuacion),
        notas="Contraste con las variantes que conservan puntuacion; afecta a la mencion y al hashtag.",
    ),
    VariantePreprocesamiento(
        "PRE-08",
        "digitos sustituidos por el token 'num'",
        _por_texto(normalizar_digitos),
        notas="726 documentos de train contienen algun digito (numeros de telefono, fechas, cantidades).",
    ),
    VariantePreprocesamiento(
        "PRE-09",
        "lematizado con spaCy es_core_news_md (exclude=['ner'], sin vectores)",
        lematizar,
        notas="H7: la lematizacion suele no aportar en tweets; D-003 prohibe doc.vector/token.vector/doc.tensor.",
    ),
)


# ===== SECCION 3: representaciones (Parte B) =====


@dataclass(frozen=True)
class EspecRepresentacion:
    """Especificacion de una representacion TF-IDF.

    Attributes:
        identificador: id corto de la representacion (REP-01..REP-06) o
            ``WORD_12`` para la representacion del baseline.
        nombre: etiqueta descriptiva de la representacion (columna
            ``representacion`` del reporte).
        descripcion: razon de ser o contenido de la representacion.
        partes: parametros de ``TfidfVectorizer``; mas de una parte construye una
            ``FeatureUnion`` que concatena matrices sparse.
        grupo_features: familia de features para ``experiments.csv``.
    """

    identificador: str
    nombre: str
    descripcion: str
    partes: tuple[dict, ...]
    grupo_features: str


def _tfidf(**overrides: object) -> dict:
    """Combina los parametros TF-IDF comunes con los de una variante.

    Args:
        **overrides: parametros que sobreescriben ``TFIDF_PARAMS``.

    Returns:
        Diccionario de parametros completo para ``TfidfVectorizer``.
    """
    return {**TFIDF_PARAMS, **overrides}


REPRESENTACION_WORD_12: Final[EspecRepresentacion] = EspecRepresentacion(
    identificador="WORD_12",
    nombre="TF-IDF word (1,2), min_df=2, sublinear_tf",
    descripcion="n-gramas de palabra 1-2 (representacion del baseline B1)",
    partes=(_tfidf(analyzer="word", ngram_range=(1, 2)),),
    grupo_features="WORD_NGRAM",
)

REPRESENTACIONES: Final[tuple[EspecRepresentacion, ...]] = (
    EspecRepresentacion(
        identificador="REP-01",
        nombre="TF-IDF char_wb (3,5), min_df=2, sublinear_tf",
        descripcion="n-gramas de caracter char_wb 3-5 (H1: evasion ortografica)",
        partes=(_tfidf(analyzer="char_wb", ngram_range=(3, 5)),),
        grupo_features="CHAR_NGRAM",
    ),
    EspecRepresentacion(
        identificador="REP-02",
        nombre="TF-IDF char_wb (2,5), min_df=2, sublinear_tf",
        descripcion="n-gramas de caracter char_wb 2-5 (rango mas corto que REP-01)",
        partes=(_tfidf(analyzer="char_wb", ngram_range=(2, 5)),),
        grupo_features="CHAR_NGRAM",
    ),
    EspecRepresentacion(
        identificador="REP-03",
        nombre="TF-IDF word (1,1), min_df=2, sublinear_tf",
        descripcion="unigramas de palabra (control de n-gramas de orden 1)",
        partes=(_tfidf(analyzer="word", ngram_range=(1, 1)),),
        grupo_features="WORD_NGRAM",
    ),
    EspecRepresentacion(
        identificador="REP-04",
        nombre="TF-IDF word (1,3), min_df=2, sublinear_tf",
        descripcion="n-gramas de palabra 1-3 (control de orden mayor que B1)",
        partes=(_tfidf(analyzer="word", ngram_range=(1, 3)),),
        grupo_features="WORD_NGRAM",
    ),
    EspecRepresentacion(
        identificador="REP-05",
        nombre="union TF-IDF word (1,2) + char_wb (3,5), min_df=2, sublinear_tf",
        descripcion="union de palabras y caracteres (H4: complementariedad)",
        partes=(
            _tfidf(analyzer="word", ngram_range=(1, 2)),
            _tfidf(analyzer="char_wb", ngram_range=(3, 5)),
        ),
        grupo_features="WORD_NGRAM+CHAR_NGRAM",
    ),
    EspecRepresentacion(
        identificador="REP-06",
        nombre="union TF-IDF word (1,2) + char_wb (3,6), min_df=2, sublinear_tf",
        descripcion="union de palabras y caracteres con rango 3-6 (variante de REP-05)",
        partes=(
            _tfidf(analyzer="word", ngram_range=(1, 2)),
            _tfidf(analyzer="char_wb", ngram_range=(3, 6)),
        ),
        grupo_features="WORD_NGRAM+CHAR_NGRAM",
    ),
)


def con_lowercase(
    base: EspecRepresentacion, lowercase: bool, sufijo: str
) -> EspecRepresentacion:
    """Devuelve la representacion con el parametro ``lowercase`` fijado.

    Args:
        base: representacion de partida.
        lowercase: valor que se asigna a ``lowercase`` en todos los bloques.
        sufijo: texto que se anade al nombre para dejar constancia en el reporte.

    Returns:
        Nueva especificacion de representacion.
    """
    partes = tuple({**parte, "lowercase": lowercase} for parte in base.partes)
    return replace(base, nombre=base.nombre + sufijo, partes=partes)


def construir_vectorizador(espec: EspecRepresentacion):
    """Crea el vectorizador (o la union de vectorizadores) de una representacion.

    Args:
        espec: especificacion de la representacion.

    Returns:
        ``TfidfVectorizer`` si hay un solo bloque o ``FeatureUnion`` con la
        concatenacion sparse de varios bloques.
    """
    if len(espec.partes) == 1:
        return TfidfVectorizer(**espec.partes[0])
    return FeatureUnion(
        [
            (f"bloque_{indice}", TfidfVectorizer(**parametros))
            for indice, parametros in enumerate(espec.partes)
        ]
    )


# ===== SECCION 4: modelos (Parte C) =====


@dataclass(frozen=True)
class EspecModelo:
    """Especificacion de un clasificador lineal o probabilistico.

    Attributes:
        nombre: etiqueta corta para el reporte.
        descripcion: descripcion legible con los hiperparametros.
        tipo: ``logistic``, ``svc`` o ``complement_nb``.
        parametros: hiperparametros que se pasan al constructor de sklearn.
        modo_score: ``proba`` (umbraliza ``predict_proba``) o ``decision``
            (umbraliza ``decision_function`` en 0).
    """

    nombre: str
    descripcion: str
    tipo: str
    parametros: dict
    modo_score: str


def construir_modelo(espec: EspecModelo):
    """Instancia un clasificador nuevo (sin estado compartido entre pliegues).

    Args:
        espec: especificacion del modelo.

    Returns:
        Clasificador de scikit-learn sin ajustar.

    Raises:
        ValueError: si ``tipo`` no es uno de los soportados.
    """
    if espec.tipo == "logistic":
        return LogisticRegression(**espec.parametros)
    if espec.tipo == "svc":
        return LinearSVC(**espec.parametros)
    if espec.tipo == "complement_nb":
        return ComplementNB(**espec.parametros)
    raise ValueError(f"Tipo de modelo no soportado: {espec.tipo}")


def predecir(modelo, matriz, espec: EspecModelo) -> np.ndarray:
    """Convierte la salida del modelo en etiquetas binarias.

    Args:
        modelo: clasificador ya ajustado.
        matriz: matriz sparse de validacion.
        espec: especificacion del modelo (define el modo de score).

    Returns:
        Vector binario de forma ``(n_filas,)``.
    """
    if espec.modo_score == "proba":
        return (
            modelo.predict_proba(matriz)[:, 1] >= CONFIG["umbral_probabilidad"]
        ).astype(int)
    return (modelo.decision_function(matriz) > 0).astype(int)


MODELO_FIJO_AB: Final[EspecModelo] = EspecModelo(
    nombre="LogisticRegression(class_weight=balanced, C=1.0, liblinear) x3",
    descripcion="LR balanced C=1.0 liblinear, umbral 0.5 (modelo fijo de las partes A y B)",
    tipo="logistic",
    parametros={
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 1000,
        "solver": "liblinear",
        "random_state": CONFIG["random_state"],
    },
    modo_score="proba",
)

MODELOS_PARTE_C: Final[tuple[EspecModelo, ...]] = (
    EspecModelo(
        nombre="LinearSVC(class_weight=balanced, C=1.0) x3",
        descripcion="SVM lineal balanced; prediccion por decision_function > 0",
        tipo="svc",
        parametros={
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 5000,
            "random_state": CONFIG["random_state"],
        },
        modo_score="decision",
    ),
    EspecModelo(
        nombre="ComplementNB(alpha=1.0) x3",
        descripcion="ComplementNB alpha=1.0, umbral 0.5 (apto para TF-IDF no negativo)",
        tipo="complement_nb",
        parametros={"alpha": 1.0},
        modo_score="proba",
    ),
    EspecModelo(
        nombre="LogisticRegression(class_weight=balanced, C=0.5, liblinear) x3",
        descripcion="LR balanced C=0.5 liblinear, umbral 0.5 (mas regularizado)",
        tipo="logistic",
        parametros={
            "C": 0.5,
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "liblinear",
            "random_state": CONFIG["random_state"],
        },
        modo_score="proba",
    ),
    EspecModelo(
        nombre="LogisticRegression(class_weight=balanced, C=2.0, liblinear) x3",
        descripcion="LR balanced C=2.0 liblinear, umbral 0.5 (menos regularizado)",
        tipo="logistic",
        parametros={
            "C": 2.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "liblinear",
            "random_state": CONFIG["random_state"],
        },
        modo_score="proba",
    ),
)


# ===== SECCION 5: motor de evaluacion =====


@dataclass(frozen=True)
class ResultadoConfig:
    """Resultado completo de una configuracion evaluada.

    Attributes:
        experiment_id: identificador de la configuracion.
        parte: parte del barrido (A, B o C).
        descripcion: descripcion legible de la configuracion.
        espec_representacion: representacion TF-IDF usada.
        espec_modelo: clasificador usado.
        metricas_oof: metricas sobre las predicciones out-of-fold completas.
        metricas_pliegues: lista de diccionarios con las metricas de cada pliegue.
        dimension: numero de features del vectorizador ajustado con todo el train
            (diagnostico; no interviene en la validacion).
        segundos: tiempo de evaluacion de la configuracion.
    """

    experiment_id: str
    parte: str
    descripcion: str
    espec_representacion: EspecRepresentacion
    espec_modelo: EspecModelo
    metricas_oof: MetricasMultiEtiqueta
    metricas_pliegues: list[dict]
    dimension: int
    segundos: float

    @property
    def representacion(self) -> str:
        """Nombre legible de la representacion TF-IDF."""
        return self.espec_representacion.nombre

    @property
    def modelo(self) -> str:
        """Nombre legible del clasificador."""
        return self.espec_modelo.nombre

    @property
    def grupo_features(self) -> str:
        """Familia de features de la representacion."""
        return self.espec_representacion.grupo_features


@dataclass
class EstadoBarrido:
    """Estado acumulado del barrido de configuraciones.

    Attributes:
        etiquetas: matriz ``(n, 3)`` con las etiquetas verdaderas de train.
        pliegues: particion agrupada compartida por todas las configuraciones.
        resultados: resultados completos en orden de ejecucion.
        predicciones: predicciones out-of-fold por ``experiment_id``.
        filas: filas del reporte CSV en orden de ejecucion.
        referencia_oof: F1 primaria de la configuracion de referencia (PRE-00).
        referencia_pliegues: F1 primaria por pliegue de la referencia.
    """

    etiquetas: np.ndarray
    pliegues: list[tuple[np.ndarray, np.ndarray]]
    resultados: list[ResultadoConfig] = field(default_factory=list)
    predicciones: dict[str, np.ndarray] = field(default_factory=dict)
    filas: list[dict] = field(default_factory=list)
    referencia_oof: float | None = None
    referencia_pliegues: list[float] | None = None


def calcular_metricas(
    etiquetas: np.ndarray,
    prediccion: np.ndarray,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[MetricasMultiEtiqueta, list[dict]]:
    """Calcula las metricas out-of-fold globales y por pliegue.

    Args:
        etiquetas: matriz ``(n, 3)`` con las etiquetas verdaderas.
        prediccion: matriz ``(n, 3)`` con las predicciones out-of-fold.
        pliegues: particion usada para generar las predicciones.

    Returns:
        Tupla ``(metricas_globales, metricas_por_pliegue)``; cada diccionario de
        pliegue incluye ``pliegue`` y ``n_validacion``.
    """
    metricas_pliegues: list[dict] = []
    for numero, (_, validacion) in enumerate(pliegues, start=1):
        metrica = dict(
            metricas_multietiqueta(etiquetas[validacion], prediccion[validacion])
        )
        metrica["pliegue"] = numero
        metrica["n_validacion"] = len(validacion)
        metricas_pliegues.append(metrica)
    return metricas_multietiqueta(etiquetas, prediccion), metricas_pliegues


def evaluar_configuracion(
    experiment_id: str,
    parte: str,
    descripcion: str,
    textos: pd.Series,
    espec_representacion: EspecRepresentacion,
    espec_modelo: EspecModelo,
    estado: EstadoBarrido,
) -> tuple[ResultadoConfig, np.ndarray]:
    """Evalua una configuracion con la particion agrupada compartida.

    El vectorizador se ajusta dentro de cada pliegue (nunca con validacion) y se
    entrena un clasificador binario por etiqueta. El paso de dimension ajusta el
    vectorizador con todo el train solo como diagnostico: no interviene en las
    predicciones out-of-fold.

    Args:
        experiment_id: identificador de la configuracion.
        parte: parte del barrido (A, B o C).
        descripcion: descripcion legible para el reporte.
        textos: serie de textos ya preprocesados (misma longitud que el train).
        espec_representacion: representacion TF-IDF a usar.
        espec_modelo: clasificador a usar.
        estado: estado del barrido con las etiquetas y los pliegues.

    Returns:
        Tupla ``(resultado, prediccion_oof)``.
    """
    inicio = time.perf_counter()
    prediccion = np.zeros_like(estado.etiquetas)
    for entrenamiento, validacion in estado.pliegues:
        vectorizador = construir_vectorizador(espec_representacion)
        matriz_entrenamiento = vectorizador.fit_transform(textos.iloc[entrenamiento])
        matriz_validacion = vectorizador.transform(textos.iloc[validacion])
        for indice, etiqueta in enumerate(ETIQUETAS):
            modelo = construir_modelo(espec_modelo)
            modelo.fit(matriz_entrenamiento, estado.etiquetas[entrenamiento, indice])
            prediccion[validacion, indice] = predecir(
                modelo, matriz_validacion, espec_modelo
            )
    dimension = int(
        construir_vectorizador(espec_representacion).fit_transform(textos).shape[1]
    )
    metricas_oof, metricas_pliegues = calcular_metricas(
        estado.etiquetas, prediccion, estado.pliegues
    )
    resultado = ResultadoConfig(
        experiment_id=experiment_id,
        parte=parte,
        descripcion=descripcion,
        espec_representacion=espec_representacion,
        espec_modelo=espec_modelo,
        metricas_oof=metricas_oof,
        metricas_pliegues=metricas_pliegues,
        dimension=dimension,
        segundos=round(time.perf_counter() - inicio, 2),
    )
    return resultado, prediccion


def contar_pliegues_mejorados(
    candidato: Sequence[dict], referencia: Sequence[float]
) -> int:
    """Cuenta en cuantos pliegues la configuracion supera a la referencia.

    Args:
        candidato: metricas por pliegue de la configuracion candidata.
        referencia: F1 primaria por pliegue de la configuracion de referencia.

    Returns:
        Numero de pliegues con mejora estricta.
    """
    return int(
        sum(
            metrica["f1_media_columnas"] > valor
            for metrica, valor in zip(candidato, referencia, strict=True)
        )
    )


def decidir(mejora: float, pliegues_mejorados: int, es_referencia: bool) -> str:
    """Aplica la regla de decision pre-registrada del proyecto.

    Una variante se adopta solo si mejora al menos ``umbral_mejora`` en la
    metrica primaria y si mejora en al menos ``pliegues_minimos_mejora`` de los
    5 pliegues. Cualquier otro caso se documenta y se descarta.

    Args:
        mejora: diferencia de ``f1_media_columnas`` frente a la referencia.
        pliegues_mejorados: numero de pliegues con mejora estricta.
        es_referencia: True si la configuracion es la de referencia.

    Returns:
        ``referencia``, ``adoptada`` o ``descartada``.
    """
    if es_referencia:
        return "referencia"
    cumple = (
        mejora >= CONFIG["umbral_mejora"]
        and pliegues_mejorados >= CONFIG["pliegues_minimos_mejora"]
    )
    return "adoptada" if cumple else "descartada"


def construir_fila(
    resultado: ResultadoConfig, referencia_oof: float, referencia_pliegues: Sequence[float]
) -> dict:
    """Construye una fila del reporte CSV a partir de un resultado.

    Args:
        resultado: resultado de la configuracion.
        referencia_oof: F1 primaria de la referencia (PRE-00).
        referencia_pliegues: F1 primaria por pliegue de la referencia.

    Returns:
        Diccionario con las columnas de ``COLUMNAS_REPORTE``.
    """
    metricas = resultado.metricas_oof
    mejora = metricas["f1_media_columnas"] - referencia_oof
    pliegues_mejorados = contar_pliegues_mejorados(
        resultado.metricas_pliegues, referencia_pliegues
    )
    return {
        "experiment_id": resultado.experiment_id,
        "parte": resultado.parte,
        "descripcion": resultado.descripcion,
        "representacion": resultado.representacion,
        "modelo": resultado.modelo,
        # Clave auxiliar (fuera de COLUMNAS_REPORTE): la consume experiments.csv.
        "grupo_features": resultado.grupo_features,
        "f1_media_columnas": round(metricas["f1_media_columnas"], 4),
        "f1_macro6": round(metricas["f1_macro6"], 4),
        "f1_micro": round(metricas["f1_micro"], 4),
        "f1_samples": round(metricas["f1_samples"], 4),
        "exact_match": round(metricas["exact_match"], 4),
        "f1_HS": round(metricas["f1_por_etiqueta"]["HS"], 4),
        "f1_TR": round(metricas["f1_por_etiqueta"]["TR"], 4),
        "f1_AG": round(metricas["f1_por_etiqueta"]["AG"], 4),
        "std_pliegues": round(
            float(
                np.std(
                    [metrica["f1_media_columnas"] for metrica in resultado.metricas_pliegues]
                )
            ),
            4,
        ),
        "mejora_vs_B1": round(mejora, 6),
        "mejora_en_pliegues": pliegues_mejorados,
        "dimension": resultado.dimension,
        "tiempo_segundos": resultado.segundos,
        "decision": decidir(
            mejora,
            pliegues_mejorados,
            es_referencia=resultado.experiment_id == CONFIG["referencia_id"],
        ),
    }


def evaluar_y_registrar(
    estado: EstadoBarrido,
    experiment_id: str,
    parte: str,
    descripcion: str,
    textos: pd.Series,
    espec_representacion: EspecRepresentacion,
    espec_modelo: EspecModelo,
) -> ResultadoConfig:
    """Evalua una configuracion, la registra en el estado y la reporta en consola.

    Args:
        estado: estado del barrido.
        experiment_id: identificador de la configuracion.
        parte: parte del barrido (A, B o C).
        descripcion: descripcion legible para el reporte.
        textos: textos preprocesados que alimentan al vectorizador.
        espec_representacion: representacion TF-IDF.
        espec_modelo: clasificador.

    Returns:
        El resultado de la configuracion evaluada.

    Raises:
        RuntimeError: si la referencia PRE-00 no se ha evaluado todavia.
    """
    resultado, prediccion = evaluar_configuracion(
        experiment_id,
        parte,
        descripcion,
        textos,
        espec_representacion,
        espec_modelo,
        estado,
    )
    estado.predicciones[experiment_id] = prediccion
    estado.resultados.append(resultado)
    if estado.referencia_oof is None:
        if experiment_id != CONFIG["referencia_id"]:
            raise RuntimeError(
                f"La referencia {CONFIG['referencia_id']} debe evaluarse primero"
            )
        estado.referencia_oof = resultado.metricas_oof["f1_media_columnas"]
        estado.referencia_pliegues = [
            metrica["f1_media_columnas"] for metrica in resultado.metricas_pliegues
        ]
    fila = construir_fila(
        resultado, estado.referencia_oof, estado.referencia_pliegues or []
    )
    estado.filas.append(fila)
    log(
        f"{experiment_id:12s} f1_media_columnas={fila['f1_media_columnas']:.4f} "
        f"mejora={fila['mejora_vs_B1']:+.4f} pliegues={fila['mejora_en_pliegues']}/5 "
        f"dim={fila['dimension']} {fila['decision']}"
    )
    return resultado


def ajustar_decision_postproceso(estado: EstadoBarrido) -> None:
    """Corrige la decision de la fila MOD-04 comparandola con su origen.

    La regla general compara contra la referencia B1, pero el post-proceso
    jerarquico debe juzgarse contra la configuracion sobre la que se aplica.

    Args:
        estado: estado del barrido con las filas ya construidas.
    """
    fila_mod04 = next(
        (fila for fila in estado.filas if fila["experiment_id"] == "MOD-04"), None
    )
    if fila_mod04 is None:
        return
    origen = next(
        (fila for fila in estado.filas if fila["experiment_id"] == "MOD-03b_BEST"),
        None,
    )
    if origen is None:
        candidatas = [
            fila for fila in estado.filas if fila["experiment_id"] != "MOD-04"
        ]
        if not candidatas:
            return
        origen = max(candidatas, key=lambda fila: fila["f1_media_columnas"])
    if fila_mod04["f1_media_columnas"] <= origen["f1_media_columnas"]:
        fila_mod04["decision"] = (
            f"descartada (empeora a su origen {origen['experiment_id']})"
        )


def mejor_resultado(resultados: Sequence[ResultadoConfig]) -> ResultadoConfig:
    """Devuelve el resultado con mayor F1 primaria (empate: el primero).

    Args:
        resultados: lista de resultados en orden de ejecucion.

    Returns:
        Resultado ganador.
    """
    ganador = resultados[0]
    for resultado in resultados[1:]:
        if (
            resultado.metricas_oof["f1_media_columnas"]
            > ganador.metricas_oof["f1_media_columnas"]
        ):
            ganador = resultado
    return ganador


# ===== SECCION 6: reportes =====


def comparacion_postproceso(estado: EstadoBarrido) -> dict | None:
    """Resume el efecto del post-proceso jerarquico sobre su configuracion origen.

    Args:
        estado: estado del barrido con todos los resultados.

    Returns:
        Diccionario con el origen, la F1 con y sin post-proceso y el delta; o
        ``None`` si MOD-04 no se ejecuto.
    """
    mod04 = next(
        (resultado for resultado in estado.resultados if resultado.experiment_id == "MOD-04"),
        None,
    )
    if mod04 is None:
        return None
    origen_id = mod04.espec_modelo.parametros.get("origen")
    origen = next(
        (resultado for resultado in estado.resultados if resultado.experiment_id == origen_id),
        None,
    )
    if origen is None:
        return None
    f1_origen = float(origen.metricas_oof["f1_media_columnas"])
    f1_post = float(mod04.metricas_oof["f1_media_columnas"])
    return {
        "origen": origen_id,
        "f1_origen": round(f1_origen, 6),
        "f1_con_postproceso": round(f1_post, 6),
        "delta": round(f1_post - f1_origen, 6),
        "f1_por_etiqueta_origen": origen.metricas_oof["f1_por_etiqueta"],
        "f1_por_etiqueta_postproceso": mod04.metricas_oof["f1_por_etiqueta"],
        "conclusion": (
            "el post-proceso jerarquico empeora la mejor configuracion"
            if f1_post < f1_origen
            else "el post-proceso jerarquico mejora la mejor configuracion"
        ),
    }


def escribir_reporte(estado: EstadoBarrido, mejor: ResultadoConfig, evidencia: dict,
                     detalle_variantes: dict, referencia_publicada: float | None) -> dict:
    """Escribe el CSV y el JSON de resultados del barrido.

    Args:
        estado: estado con las filas y los resultados completos.
        mejor: mejor configuracion global del barrido.
        evidencia: conteos de emojis y caracteres no ASCII.
        detalle_variantes: numero de textos modificados por cada variante PRE.
        referencia_publicada: F1 de B1 registrada por el script 02, si existe.

    Returns:
        Diccionario de metadatos usado para la seccion de ``reports/contexto.md``.
    """
    reporte = pd.DataFrame(estado.filas, columns=list(COLUMNAS_REPORTE))
    orden = reporte.sort_values(
        "f1_media_columnas", ascending=False, kind="stable"
    ).reset_index(drop=True)
    CONFIG["reporte_csv"].parent.mkdir(parents=True, exist_ok=True)
    orden.to_csv(CONFIG["reporte_csv"], index=False, encoding="utf-8")

    adoptadas = [
        fila["experiment_id"] for fila in estado.filas if fila["decision"] == "adoptada"
    ]
    referencia_oof = (
        None if estado.referencia_oof is None else float(estado.referencia_oof)
    )
    detalle = {
        "metadata": {
            "fecha": CONFIG["fecha"],
            "script": CONFIG["script"],
            "competencia": "KAGGLE_TOXIC_2026_PUJ",
            "fase": (
                "Fases 4-5 del plan (variantes de preprocesamiento, representaciones "
                "y modelos)"
            ),
            "metrica_primaria": "f1_media_columnas",
            "referencia": CONFIG["referencia_id"],
            "referencia_f1_oof": referencia_oof,
            "referencia_b1_registrada": referencia_publicada,
            "diferencia_vs_b1_registrada": (
                None
                if (referencia_publicada is None or referencia_oof is None)
                else round(referencia_oof - referencia_publicada, 6)
            ),
            "regla_decision": (
                f"adoptada si mejora >= {CONFIG['umbral_mejora']} absoluto en "
                f"f1_media_columnas y mejora en >= {CONFIG['pliegues_minimos_mejora']} "
                "de 5 pliegues"
            ),
            "validacion": (
                f"StratifiedGroupKFold({CONFIG['n_splits']}, shuffle=True, "
                f"random_state={CONFIG['random_state']}), grupos = near-duplicados, "
                "estratificado por combinacion HSxTRxAG"
            ),
            "tamano_pliegues": [
                {
                    "pliegue": numero,
                    "n_train": len(entrenamiento),
                    "n_validacion": len(validacion),
                }
                for numero, (entrenamiento, validacion) in enumerate(
                    estado.pliegues, start=1
                )
            ],
            "sin_fuga": (
                "Vectorizadores ajustados dentro de cada pliegue; transformaciones de "
                "texto deterministas y sin parametros ajustados; el test solo se "
                "inspecciono para documentar la ausencia de emojis"
            ),
            "idempotencia": (
                "Determinista salvo la columna tiempo_segundos (ruido de maquina, "
                "redondeada a 2 decimales) en variantes_representaciones.csv"
            ),
        },
        "evidencia_emojis": evidencia,
        "textos_modificados_por_variante": detalle_variantes,
        "comparacion_postproceso": comparacion_postproceso(estado),
        "mejor_representacion_parte_b": None,
        "mejor_configuracion_global": mejor.experiment_id,
        "configuraciones_adoptadas": adoptadas,
        "configuraciones": [
            {
                "experiment_id": resultado.experiment_id,
                "parte": resultado.parte,
                "descripcion": resultado.descripcion,
                "representacion": resultado.representacion,
                "modelo": resultado.modelo,
                "grupo_features": resultado.grupo_features,
                "parametros_vectorizador": list(resultado.espec_representacion.partes),
                "parametros_modelo": resultado.espec_modelo.parametros,
                "dimension": resultado.dimension,
                "tiempo_segundos": resultado.segundos,
                "metricas_oof": resultado.metricas_oof,
                "metricas_pliegues": resultado.metricas_pliegues,
                "mejora_vs_B1": round(
                    float(resultado.metricas_oof["f1_media_columnas"])
                    - float(estado.referencia_oof or 0.0),
                    6,
                ),
                "mejora_en_pliegues": contar_pliegues_mejorados(
                    resultado.metricas_pliegues, estado.referencia_pliegues or []
                ),
                "decision": next(
                    fila["decision"]
                    for fila in estado.filas
                    if fila["experiment_id"] == resultado.experiment_id
                ),
            }
            for resultado in estado.resultados
        ],
    }
    return detalle


def escribir_detalle_json(detalle: dict, mejor_representacion: str) -> None:
    """Escribe el JSON de detalle con las metricas por pliegue.

    Args:
        detalle: diccionario de detalle construido en :func:`escribir_reporte`.
        mejor_representacion: ganadora de la Parte B.
    """
    detalle["mejor_representacion_parte_b"] = mejor_representacion
    CONFIG["reporte_json"].write_text(
        json.dumps(detalle, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def registrar_en_experiments(filas: Sequence[dict]) -> int:
    """Anade (o refresca) las filas del barrido en ``experiments.csv``.

    La operacion es idempotente: primero elimina las filas cuyo
    ``experiment_id`` pertenece a este barrido y despues las reescribe, de modo
    que dos ejecuciones seguidas dejan el archivo con el mismo contenido (salvo
    ``training_time``, que depende de la maquina).

    Args:
        filas: filas del reporte de variantes.

    Returns:
        Numero de filas escritas.

    Raises:
        ValueError: si el esquema del archivo no coincide con el esperado.
    """
    destino: Path = CONFIG["experiments_csv"]
    existente = pd.read_csv(destino, dtype=str, keep_default_na=False)
    identificadores = {fila["experiment_id"] for fila in filas}
    conservadas = existente[~existente["experiment_id"].isin(identificadores)]
    nuevas = [
        construir_fila_experiments(fila, parte=fila["parte"]) for fila in filas
    ]
    combinado = pd.concat(
        [conservadas, pd.DataFrame(nuevas, columns=list(existente.columns))],
        ignore_index=True,
    )
    if list(combinado.columns) != list(existente.columns):
        raise ValueError("El esquema de experiments.csv cambio inesperadamente")
    combinado.to_csv(destino, index=False, encoding="utf-8")
    return len(nuevas)


def _pregunta_de_parte(parte: str) -> tuple[str, str]:
    """Devuelve la pregunta e hipotesis asociadas a una parte del barrido.

    Args:
        parte: parte del barrido (A, B o C).

    Returns:
        Tupla ``(pregunta, hipotesis)``.
    """
    if parte == PARTE_A:
        return (
            "Que transformacion determinista del texto mejora el F1 con modelo y representacion fijos?",
            "Alguna normalizacion explicita del ruido de Twitter supera el texto crudo (H6, H7)",
        )
    if parte == PARTE_B:
        return (
            "Que representacion TF-IDF obtiene el mejor F1 con el clasificador fijo?",
            "Los n-gramas de caracter capturan evasion ortografica mejor que los de palabra (H1) y su union aporta (H4)",
        )
    return (
        "Que clasificador lineal aprovecha mejor la representacion ganadora?",
        "Un margen maximo (SVM) o un modelo probabilistico distinto mejora la LR del baseline",
    )


def construir_fila_experiments(fila: dict, parte: str) -> dict:
    """Adapta una fila del reporte al esquema de ``experiments.csv``.

    Args:
        fila: fila del reporte de variantes.
        parte: parte del barrido (A, B o C).

    Returns:
        Diccionario con las columnas del registro de experimentos.
    """
    pregunta, hipotesis = _pregunta_de_parte(parte)
    return {
        "experiment_id": fila["experiment_id"],
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": pregunta,
        "hypothesis": hipotesis,
        "preprocessing": fila["descripcion"],
        "feature_groups": fila["grupo_features"],
        "representation": fila["representacion"],
        "model": fila["modelo"],
        "hyperparameters": "",
        "validation": (
            f"StratifiedGroupKFold(5) grupos=near-duplicados, estratificado por combo HSxTRxAG; "
            f"referencia {CONFIG['referencia_id']}"
        ),
        "f1": fila["f1_media_columnas"],
        "macro_f1": fila["f1_macro6"],
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": fila["tiempo_segundos"],
        "dimension": fila["dimension"],
        "result": json.dumps(
            {
                "f1_media_columnas": fila["f1_media_columnas"],
                "f1_macro6": fila["f1_macro6"],
                "f1_micro": fila["f1_micro"],
                "f1_samples": fila["f1_samples"],
                "exact_match": fila["exact_match"],
                "f1_por_etiqueta": {
                    "HS": fila["f1_HS"],
                    "TR": fila["f1_TR"],
                    "AG": fila["f1_AG"],
                },
                "std_pliegues": fila["std_pliegues"],
                "mejora_vs_B1": fila["mejora_vs_B1"],
                "mejora_en_pliegues": fila["mejora_en_pliegues"],
            },
            ensure_ascii=False,
        ),
        "interpretation": f"Variante {fila['experiment_id']} de la {parte}: decision {fila['decision']}",
        "decision": fila["decision"],
        "date": CONFIG["fecha"],
        "script": CONFIG["script"],
        "notes": (
            "Barrido de Fases 4-5; sin fuga (vectorizador por pliegue); "
            f"std_pliegues={fila['std_pliegues']}; dimension={fila['dimension']}"
        ),
    }


def construir_seccion_contexto(
    estado: EstadoBarrido, detalle: dict, mejor_representacion: ResultadoConfig
) -> str:
    """Redacta la seccion de handoff (maximo 20 lineas) para ``reports/contexto.md``.

    Args:
        estado: estado del barrido con las filas en orden de ejecucion.
        detalle: detalle del barrido escrito en JSON.
        mejor_representacion: ganadora de la Parte B.

    Returns:
        Texto de la seccion, con encabezado y sin linea final en blanco.

    Raises:
        RuntimeError: si la seccion generada excede las 20 lineas permitidas.
    """
    orden = sorted(
        estado.filas, key=lambda fila: fila["f1_media_columnas"], reverse=True
    )
    mejor = orden[0]
    por_parte = {
        parte: [fila for fila in orden if fila["parte"] == parte]
        for parte in (PARTE_A, PARTE_B, PARTE_C)
    }
    adoptadas = [fila for fila in orden if fila["decision"] == "adoptada"]
    registrada = detalle["metadata"]["referencia_b1_registrada"]
    if estado.referencia_oof is None:
        linea_referencia = "Referencia no disponible (error de ejecucion)."
    elif registrada is None:
        linea_referencia = (
            f"Referencia `{CONFIG['referencia_id']}` (B1) reproducida en "
            f"{estado.referencia_oof:.4f}; no habia valor de B1 registrado para comparar."
        )
    else:
        linea_referencia = (
            f"Referencia `{CONFIG['referencia_id']}` (B1) reproducida en "
            f"{estado.referencia_oof:.4f} frente al B1 registrado {float(registrada):.4f} "
            f"(diferencia {estado.referencia_oof - float(registrada):+.6f})."
        )
    lista_adoptadas = (
        ", ".join(
            f"{fila['experiment_id']} ({fila['f1_media_columnas']:.4f})"
            for fila in adoptadas
        )
        if adoptadas
        else "ninguna"
    )
    mod04 = next((fila for fila in orden if fila["experiment_id"] == "MOD-04"), None)
    if mod04 is None:
        linea_postproceso = "MOD-04 (post-proceso jerarquico) no se ejecuto."
    else:
        origen = next(
            (fila for fila in estado.filas if fila["experiment_id"] == "MOD-03b_BEST"),
            mejor,
        )
        delta = mod04["f1_media_columnas"] - origen["f1_media_columnas"]
        linea_postproceso = (
            f"MOD-04 (post-proceso jerarquico sobre el ganador): {mod04['f1_media_columnas']:.4f} "
            f"frente a {origen['f1_media_columnas']:.4f} de su origen {origen['experiment_id']} "
            f"-> {delta:+.4f}; resultado "
            f"{'negativo' if delta <= 0 else 'positivo'} de H8 con esta representacion, por lo que "
            f"{'no ' if delta <= 0 else ''}se adopta como configuracion final."
        )
    lineas = [
        f"## Fase 4-5 - ml-coder ({CONFIG['momento_contexto']})",
        "",
        (
            f"Barrido de {len(estado.filas)} configuraciones de las Fases 4-5: PARTE A "
            "preprocesamiento (PRE-00..PRE-09), PARTE B representaciones (REP-01..REP-06) y "
            "PARTE C modelos (MOD-01..MOD-04) con 5-fold."
        ),
        (
            "Validacion comun `StratifiedGroupKFold(shuffle=True, random_state=42)` agrupada por "
            "near-duplicado y estratificada por la combinacion HSxTRxAG; modelo fijo LR balanced "
            "C=1 liblinear en A y B; metrica primaria `f1_media_columnas`."
        ),
        linea_referencia,
        (
            "Artefactos: `reports/variantes_representaciones.csv` (ranking), "
            "`reports/variantes_representaciones.json` (metricas por pliegue), "
            "`src/evaluation.py` (metricas reutilizables) y "
            f"`{CONFIG['script']}`."
        ),
        "",
        "PARTE A: " + _resumir_parte(por_parte[PARTE_A]),
        "PARTE B: " + _resumir_parte(por_parte[PARTE_B]),
        "PARTE C: " + _resumir_parte(por_parte[PARTE_C]),
        "",
        (
            f"Mejor representacion de la Parte B: {mejor_representacion.experiment_id} "
            f"({mejor_representacion.metricas_oof['f1_media_columnas']:.4f})."
        ),
        (
            f"Mejor configuracion global: {mejor['experiment_id']} "
            f"({mejor['f1_media_columnas']:.4f}); variantes adoptadas por la regla "
            f"pre-registrada (+0.005 absoluto y >= 3/5 pliegues): {lista_adoptadas}."
        ),
        linea_postproceso,
        (
            "Prueba de emojis: 0 emojis Unicode en train y test (train 100 % ASCII), por lo que "
            "la variante de emojis no aplica; evidencia en el JSON."
        ),
        (
            "Todas las variantes no adoptadas quedan documentadas y descartadas con su evidencia "
            "en el CSV/JSON; ningun script previo se modifico (02 se conserva por trazabilidad)."
        ),
    ]
    while lineas and lineas[-1] == "":
        lineas.pop()
    if len(lineas) > 20:
        raise RuntimeError(f"La seccion de contexto excede 20 lineas ({len(lineas)})")
    return "\n".join(lineas)


def _resumir_parte(filas: Sequence[dict]) -> str:
    """Resume una parte del barrido en una frase con ganadora y descartadas.

    Args:
        filas: filas de la parte, ordenadas de mejor a peor F1.

    Returns:
        Frase resumen.
    """
    mejor = filas[0]
    segunda = filas[1] if len(filas) > 1 else None
    texto = f"gana {mejor['experiment_id']} ({mejor['f1_media_columnas']:.4f}, {mejor['mejora_vs_B1']:+.4f} vs B1, pliegues {mejor['mejora_en_pliegues']}/5, {mejor['decision']})"
    if segunda is not None:
        texto += (
            f"; siguiente {segunda['experiment_id']} ({segunda['f1_media_columnas']:.4f}); "
            f"descartadas {len(filas) - 1} de {len(filas)}"
        )
    return texto


def actualizar_contexto(seccion: str) -> Path:
    """Insertar la seccion de handoff al final de ``reports/contexto.md``.

    Idempotente: si la seccion de esta fase ya existe, se reemplaza en lugar de
    duplicarse.

    Args:
        seccion: texto de la seccion (sin linea final en blanco).

    Returns:
        Ruta del archivo actualizado.
    """
    destino: Path = CONFIG["contexto_md"]
    encabezado = f"## Fase 4-5 - ml-coder ({CONFIG['momento_contexto']})"
    texto = destino.read_text(encoding="utf-8") if destino.exists() else ""
    if encabezado in texto:
        texto = texto[: texto.index(encabezado)]
    destino.write_text(
        texto.rstrip("\n") + "\n\n" + seccion.rstrip("\n") + "\n", encoding="utf-8"
    )
    return destino


def leer_referencia_publicada() -> float | None:
    """Lee la F1 out-of-fold de B1 registrada por el script 02.

    Returns:
        Valor de ``f1_media_columnas`` sin post-proceso, o ``None`` si el archivo
        no existe o no tiene el campo esperado.
    """
    ruta: Path = CONFIG["b1_metricas_path"]
    if not ruta.exists():
        return None
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    return float(datos["oof_sin_postproceso"]["f1_media_columnas"])


# ===== SECCION 7: flujo principal =====


def preparar_textos(
    train: pd.DataFrame, variante: VariantePreprocesamiento
) -> tuple[pd.Series, int]:
    """Precalcula los textos de una variante de preprocesamiento.

    Las transformaciones son deterministas y no tienen parametros ajustados, por
    lo que aplicarlas una sola vez sobre el texto crudo no introduce fuga
    (D-010) y ahorra tiempo al repetirlas en los 5 pliegues.

    Args:
        train: DataFrame de entrenamiento.
        variante: variante de preprocesamiento.

    Returns:
        Tupla ``(textos, n_textos_modificados)``.
    """
    textos = pd.Series(
        variante.transformacion(list(train["text"])), index=train.index, name="text"
    )
    modificados = int((textos != train["text"]).sum())
    return textos, modificados


def ejecutar_parte_a(
    train: pd.DataFrame, estado: EstadoBarrido, detalle_variantes: dict
) -> None:
    """Ejecuta las variantes de preprocesamiento (PRE-00..PRE-09).

    Args:
        train: DataFrame de entrenamiento.
        estado: estado del barrido.
        detalle_variantes: acumulador de textos modificados por variante.
    """
    log("PARTE A: variantes de preprocesamiento (modelo fijo LR balanced C=1)")
    for variante in VARIANTES_PREPROCESAMIENTO:
        textos, modificados = preparar_textos(train, variante)
        detalle_variantes[variante.experiment_id] = modificados
        sufijo = "" if variante.lowercase else " [lowercase=False]"
        representacion = con_lowercase(REPRESENTACION_WORD_12, variante.lowercase, sufijo)
        evaluar_y_registrar(
            estado,
            variante.experiment_id,
            PARTE_A,
            f"{variante.descripcion} (transformacion unica sobre texto crudo; "
            f"{modificados} textos modificados)",
            textos,
            representacion,
            MODELO_FIJO_AB,
        )


def ejecutar_parte_b(
    train: pd.DataFrame, estado: EstadoBarrido
) -> list[ResultadoConfig]:
    """Ejecuta las representaciones alternativas (REP-01..REP-06).

    Args:
        train: DataFrame de entrenamiento.
        estado: estado del barrido.

    Returns:
        Resultados de la Parte B en orden de ejecucion.
    """
    log("PARTE B: representaciones (modelo fijo LR balanced C=1)")
    textos = pd.Series(train["text"], index=train.index, name="text")
    resultados = [
        evaluar_y_registrar(
            estado,
            espec.identificador,
            PARTE_B,
            f"{espec.descripcion}; parametros TF-IDF comunes min_df=2, max_df=1.0, sublinear_tf",
            textos,
            espec,
            MODELO_FIJO_AB,
        )
        for espec in REPRESENTACIONES
    ]
    return resultados


def ejecutar_parte_c(
    train: pd.DataFrame,
    estado: EstadoBarrido,
    mejor_representacion: ResultadoConfig,
) -> None:
    """Evalua los clasificadores alternativos y el post-proceso jerarquico.

    Args:
        train: DataFrame de entrenamiento.
        estado: estado del barrido.
        mejor_representacion: ganadora de la Parte B (representacion ``_BEST``).
    """
    log(
        f"PARTE C: modelos sobre {mejor_representacion.experiment_id} (mejor de B) "
        "y sobre word(1,2)"
    )
    textos = pd.Series(train["text"], index=train.index, name="text")
    representaciones = {
        "_BEST": mejor_representacion.espec_representacion,
        "_W12": REPRESENTACION_WORD_12,
    }
    identificadores = {
        MODELOS_PARTE_C[0].nombre: "MOD-01",
        MODELOS_PARTE_C[1].nombre: "MOD-02",
        MODELOS_PARTE_C[2].nombre: "MOD-03a",
        MODELOS_PARTE_C[3].nombre: "MOD-03b",
    }
    for espec_modelo in MODELOS_PARTE_C:
        for sufijo, representacion in representaciones.items():
            evaluar_y_registrar(
                estado,
                f"{identificadores[espec_modelo.nombre]}{sufijo}",
                PARTE_C,
                f"{espec_modelo.descripcion}; representacion {representacion.nombre}",
                textos,
                representacion,
                espec_modelo,
            )


def aplicar_mod04(estado: EstadoBarrido, mejor: ResultadoConfig) -> ResultadoConfig:
    """Aplica el post-proceso jerarquico a la mejor configuracion global.

    Args:
        estado: estado del barrido con las predicciones out-of-fold.
        mejor: mejor configuracion global.

    Returns:
        Resultado de la configuracion con post-proceso ``HS=0 => TR=0, AG=0``.
    """
    inicio = time.perf_counter()
    prediccion = aplicar_postproceso_jerarquico(estado.predicciones[mejor.experiment_id])
    metricas_oof, metricas_pliegues = calcular_metricas(
        estado.etiquetas, prediccion, estado.pliegues
    )
    delta = float(metricas_oof["f1_media_columnas"]) - float(
        mejor.metricas_oof["f1_media_columnas"]
    )
    resultado = ResultadoConfig(
        experiment_id="MOD-04",
        parte=PARTE_C,
        descripcion=(
            f"post-proceso jerarquico HS=0 => TR=0, AG=0 aplicado a la mejor configuracion "
            f"global ({mejor.experiment_id}); efecto {delta:+.4f} frente a su origen "
            f"({'no supera al origen' if delta <= 0 else 'supera al origen'})"
        ),
        espec_representacion=replace(
            mejor.espec_representacion,
            nombre=(
                f"post-proceso jerarquico sobre {mejor.experiment_id} "
                f"({mejor.representacion})"
            ),
        ),
        espec_modelo=EspecModelo(
            nombre=f"post-proceso jerarquico sobre {mejor.modelo}",
            descripcion="post-proceso jerarquico HS=0 => TR=0, AG=0 (D-001)",
            tipo="postproceso",
            parametros={
                "postproceso": "HS=0 => TR=0, AG=0",
                "origen": mejor.experiment_id,
            },
            modo_score="proba",
        ),
        metricas_oof=metricas_oof,
        metricas_pliegues=metricas_pliegues,
        dimension=mejor.dimension,
        segundos=round(time.perf_counter() - inicio, 2),
    )
    estado.predicciones["MOD-04"] = prediccion
    estado.resultados.append(resultado)
    fila = construir_fila(resultado, estado.referencia_oof or 0.0, estado.referencia_pliegues or [])
    estado.filas.append(fila)
    log(
        f"{'MOD-04':12s} f1_media_columnas={fila['f1_media_columnas']:.4f} "
        f"mejora={fila['mejora_vs_B1']:+.4f} pliegues={fila['mejora_en_pliegues']}/5 "
        f"{fila['decision']}"
    )
    return resultado


def mostrar_ranking(estado: EstadoBarrido, top: int = 5) -> None:
    """Imprime el ranking de las mejores configuraciones del barrido.

    Args:
        estado: estado del barrido.
        top: numero de configuraciones a imprimir.
    """
    orden = sorted(
        estado.filas, key=lambda fila: fila["f1_media_columnas"], reverse=True
    )
    log(f"ranking top {top} (f1_media_columnas out-of-fold):")
    for posicion, fila in enumerate(orden[:top], start=1):
        log(
            f"  {posicion}. {fila['experiment_id']:12s} {fila['f1_media_columnas']:.4f} "
            f"({fila['mejora_vs_B1']:+.4f} vs B1, pliegues {fila['mejora_en_pliegues']}/5) "
            f"{fila['decision']}"
        )


def main() -> None:
    """Ejecuta el barrido completo de las Fases 4-5 y escribe los artefactos."""
    inicio = time.perf_counter()
    train, grupos = cargar_datos()
    test = data_module.read_test()
    log(f"train {train.shape} | test {test.shape} (solo inspeccion) | grupos {grupos.nunique()}")
    pliegues = construir_pliegues(train, grupos)
    evidencia = recolectar_evidencia(train, test)
    log(f"emojis train/test: {evidencia['n_emojis_train']}/{evidencia['n_emojis_test']}")

    estado = EstadoBarrido(
        etiquetas=train[list(ETIQUETAS)].to_numpy(), pliegues=pliegues
    )
    detalle_variantes: dict[str, int] = {}
    ejecutar_parte_a(train, estado, detalle_variantes)
    resultados_b = ejecutar_parte_b(train, estado)
    mejor_representacion = mejor_resultado(resultados_b)
    ejecutar_parte_c(train, estado, mejor_representacion)
    mejor = mejor_resultado(estado.resultados)
    aplicar_mod04(estado, mejor)
    ajustar_decision_postproceso(estado)

    referencia_publicada = leer_referencia_publicada()
    detalle = escribir_reporte(
        estado, mejor, evidencia, detalle_variantes, referencia_publicada
    )
    escribir_detalle_json(detalle, mejor_representacion.experiment_id)
    escritas = registrar_en_experiments(estado.filas)
    contexto = actualizar_contexto(construir_seccion_contexto(estado, detalle, mejor_representacion))

    mostrar_ranking(estado)
    log(f"referencia {CONFIG['referencia_id']}: {estado.referencia_oof:.6f} | B1 registrado: {referencia_publicada}")
    log(f"mejor representacion B: {mejor_representacion.experiment_id} | mejor global: {mejor.experiment_id}")
    log(f"filas escritas: {escritas} en {CONFIG['experiments_csv'].name}")
    log(f"artefactos: {CONFIG['reporte_csv']} | {CONFIG['reporte_json']} | {contexto}")
    log(f"duracion total: {time.perf_counter() - inicio:.2f} s")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - el punto de entrada CLI reporta todo
        import traceback

        traceback.print_exc()
        sys.exit(1)
