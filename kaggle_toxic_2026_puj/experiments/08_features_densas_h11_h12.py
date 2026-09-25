"""experiments/08_features_densas_h11_h12.py - Fases 6-8: features densas para H11, H12 y H14.

Pregunta que responde
---------------------
El analisis de error de la Fase 12 (`reports/error_analysis.md`) mostro que el
error del mejor modelo (`MOD-03b_BEST`, union TF-IDF word (1,2) + char_wb (3,5)
con LR balanced C=2.0, F1 OOF 0.7927) se concentra en dos regimenes: el insulto
interpersonal en registro de broma (falsos positivos de HS y AG) y el comentario
o noticia con objetivo colectivo (falsos negativos de AG y de TR). Un modelo
lineal sobre bolsas de n-gramas no puede representar conjunciones
(insulto x marca de afecto, objetivo individual frente a colectivo) ni pesos de
lexico especificos de una etiqueta. Este script construye tres bloques densos
que representan esas conjunciones y mide si aportan:

- BLOQUE_H11 (registro bromista e interpersonal, hipotesis H11).
- BLOQUE_H12 (objetivo individual frente a colectivo, hipotesis H12).
- BLOQUE_H14 (deshumanizacion: lexico agresivo inducido por etiqueta, H14).

Hipotesis evaluadas
-------------------
- H11: los falsos positivos de HS y AG viven en el insulto interpersonal con
  marcas de broma o afiliacion; un bloque explicito de interacciones de registro
  los reduce.
- H12: el error de TR vive en los textos sin destinatario explicito; un bloque
  que separe objetivo individual de objetivo colectivo mejora TR.
- H14: los falsos negativos de AG son agresion implicita contra grupos; un
  lexico de agresividad inducido dentro del pliegue y su cruce con la referencia
  colectiva los recupera.

Protocolo (heredado, sin cambios)
---------------------------------
- Validacion: ``StratifiedGroupKFold(5, shuffle=True, random_state=42)`` con
  grupos = componentes de near-duplicado (D-004) y estratificacion por la
  combinacion HS x TR x AG. La particion se recalcula igual que en
  ``experiments/03_variantes_representaciones.py`` y se verifica contra las
  probabilidades OOF publicadas de ``MOD-03b_BEST`` antes de decidir nada.
- Metrica primaria: ``f1_media_columnas`` (D-009). Secundarias: macro6, micro,
  samples, exact match y F1 por etiqueta.
- Sin fuga: los vectorizadores TF-IDF y el ``StandardScaler`` se ajustan solo
  con el pliegue de entrenamiento; los lexicos inducidos y la lista de palabras
  frecuentes que filtra el detector de nombres propios se INDUCEN dentro de cada
  pliegue usando exclusivamente las etiquetas y los textos de entrenamiento de
  ese pliegue. Las features restantes son deterministas y sin parametros
  ajustados (regex y listas cerradas), por lo que se anotan una sola vez
  (D-010). El test nunca se lee.
- spaCy se usa como anotador de POS, lemas y morfologia con ``exclude=["ner"]``
  (D-003); prohibido ``doc.vector``, ``token.vector``, ``doc.tensor`` y
  ``similarity()``.

Regla de decision (pre-registrada)
----------------------------------
Una configuracion se adopta solo si mejora >= +0.005 absoluto en
``f1_media_columnas`` OOF **y** mejora en >= 3 de los 5 pliegues frente a
`MOD-03b_BEST`. Por D-012, en la familia union (TF-IDF + bloque denso) se exige
ademas >= +0.010 local antes de gastar una sonda de leaderboard, porque el gap
local-LB medido de esa familia es ~0.013-0.016.

Configuraciones evaluadas
-------------------------
1. `MOD-03b_REF`: control recalculado (union REP-05 + LR C=2.0) para comprobar
   que la particion reproduce la F1 publicada y para tener sus rebanadas.
2. `B5-DENSE`: LR C=1.0 balanced sobre los tres bloques densos escalados.
3. `HIB-UNION`: hstack(TF-IDF REP-05, bloques densos escalados) + LR C=2.0.
4. `HIB-H11`, `HIB-H12`, `HIB-H14`: ablacion por bloque sobre la union.

Artefactos
----------
- ``reports/features_densas_resultados.csv`` y ``.json``.
- ``reports/eda/features_densas_slices.csv`` (rebanadas debiles de la Fase 12).
- Filas B5-DENSE, HIB-UNION, HIB-H11, HIB-H12 y HIB-H14 en ``experiments.csv``
  (idempotente: al re-ejecutar se reemplazan, nunca se duplican).
- Seccion de handoff de hasta 20 lineas en ``reports/contexto.md``.

Idempotencia
------------
Todo es determinista salvo la columna de tiempos (ruido de maquina, redondeada a
2 decimales). Dos ejecuciones seguidas dejan los mismos CSV, el mismo JSON, las
mismas filas en ``experiments.csv`` y la misma seccion en ``contexto.md``.

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/08_features_densas_h11_h12.py
"""

from __future__ import annotations

import json
import re
import sys
import time
import traceback
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import hstack as sparse_hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.evaluation import ETIQUETAS, MetricasMultiEtiqueta, metricas_multietiqueta

# ===== SECCION 1: configuracion =====

CONFIG: Final[dict[str, Any]] = {
    "grupos_path": PROJECT_ROOT / "data" / "processed" / "grupos_near_duplicados.csv",
    "oof_referencia_path": PROJECT_ROOT / "features" / "oof_mod03b.npz",
    "reporte_csv": PROJECT_ROOT / "reports" / "features_densas_resultados.csv",
    "reporte_json": PROJECT_ROOT / "reports" / "features_densas_resultados.json",
    "slices_csv": PROJECT_ROOT / "reports" / "eda" / "features_densas_slices.csv",
    "experiments_csv": PROJECT_ROOT / "experiments.csv",
    "contexto_md": PROJECT_ROOT / "reports" / "contexto.md",
    "random_state": 42,
    "n_splits": 5,
    "umbral_probabilidad": 0.5,
    "spacy_model": "es_core_news_md",
    "batch_size_spacy": 64,
    "referencia_id": "MOD-03b_BEST",
    "referencia_f1": 0.7926588861781015,
    "umbral_adopcion": 0.005,
    "umbral_union_realista": 0.010,
    "pliegues_minimos": 3,
    "tolerancia_referencia": 1e-3,
    "min_frecuencia_palabra_frecuente": 8,
    "top_k_lexico": 150,
    "top_k_lexico_colectivo": 100,
    "z_nucleo_insulto": 2.5,
    "min_frecuencia_lexico": 3,
    "min_documentos_positivos": 2,
    "min_longitud_lema": 3,
    "alpha_prior": 0.01,
    "fecha": "2026-09-22",
    "script": "experiments/08_features_densas_h11_h12.py",
    "momento_contexto": "2026-09-22",
}

# Parametros TF-IDF de REP-05 (identicos a los de la Fase 4-5, sin cambios).
TFIDF_PARAMS: Final[dict[str, Any]] = {
    "min_df": 2,
    "max_df": 1.0,
    "sublinear_tf": True,
    "lowercase": True,
    "strip_accents": None,
}

COLUMNAS_EXPERIMENTS: Final[tuple[str, ...]] = (
    "experiment_id",
    "competition",
    "question",
    "hypothesis",
    "preprocessing",
    "feature_groups",
    "representation",
    "model",
    "hyperparameters",
    "validation",
    "f1",
    "macro_f1",
    "weighted_f1",
    "precision",
    "recall",
    "training_time",
    "dimension",
    "result",
    "interpretation",
    "decision",
    "date",
    "script",
    "notes",
)

COLUMNAS_REPORTE: Final[tuple[str, ...]] = (
    "experiment_id",
    "descripcion",
    "familia",
    "bloques",
    "usa_tfidf",
    "C",
    "f1_media_columnas",
    "f1_macro6",
    "f1_micro",
    "f1_samples",
    "exact_match",
    "f1_HS",
    "f1_TR",
    "f1_AG",
    "std_pliegues",
    "mejora_vs_MOD03b",
    "mejora_en_pliegues",
    "f1_train_medio",
    "brecha_train_oof",
    "dimension",
    "n_features_densas",
    "tiempo_segundos",
    "decision",
)


def log(message: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        message: texto a imprimir.
    """
    print(f"[08] {message}", flush=True)


# ===== SECCION 2: patrones y listas cerradas (features deterministas) =====

URL_PATTERN: Final = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN: Final = re.compile(r"@\w+")
HASHTAG_PATTERN: Final = re.compile(r"#\w+")
INITIAL_MENTION_PATTERN: Final = re.compile(r"^\s*(?:rt\s+)?@\w+", re.IGNORECASE)
WORD_PATTERN: Final = re.compile(r"[A-Za-z0-9@#_]+")
PUNCT_REPEAT_PATTERN: Final = re.compile(r"([!?.,;:\-'\"()]){2,}")

# Risas: patron laxo de subcadena, identico al de la Fase 12 (el corpus escribe
# series como "JAJAJAJAJJAJAJAJA" que un patron con frontera de palabra no
# captura). Se anade la forma "risas"/"risa" que el EDA tambien documenta.
RE_RISA: Final = re.compile(r"(?:ja){2,}|(?:je){2,}|(?:ji){2,}|\blol\b|\bxd+\b|\brisas?\b", re.IGNORECASE)

# Emoticones de texto (dos puntos, punto y coma, igual, equis, ocho mas
# parentesis o letra). Mismo patron que el EDA (Fase 2).
EMOTICON_PATTERN: Final = re.compile(
    r"[:;=8x][-^']?[)(dDpP]|[)(][:;=8]|\bD:", re.IGNORECASE
)

# Terminos afectivos y de afiliacion (lista cerrada: apelativos y posesivos de
# primera persona que acompanan al insulto en el registro de broma). Incluye
# las formas del enunciado de H11 y variantes inmediatas documentadas.
RE_AFECTIVO: Final = re.compile(
    r"\b(?:bebe|amor|carino|mi vida|mi amor|mi reina|mi rey|guapa|guapo|hermosa|"
    r"hermoso|linda|lindo|preciosa|bella|bello|bonita|bonito|corazon|te amo|"
    r"te quiero|besos|abrazos|socia|amig\w+)\b",
    re.IGNORECASE,
)

# Diminutivos apreciativos (-ito/-ita y plurales) sobre formas alfabeticas.
RE_DIMINUTIVO: Final = re.compile(r"\b[a-z]{3,}(?:ito|ita|itos|itas)\b")

# Vocativo con posesivo de primera persona: mencion (o saludo) inicial seguida
# de un posesivo en los primeros caracteres del texto ("@usuario mi amor...").
RE_VOCATIVO_POSESIVO: Final = re.compile(
    r"^\s*(?:rt\s+)?@\w+[\s,:;\-]+(?:\w+[\s,:;\-]+){0,4}\b(?:mi|mis|mio|mia)\b",
    re.IGNORECASE,
)

# Marcadores de atribucion a terceros (verbos de reporte y fuentes de noticia).
RE_ATRIBUCION: Final = re.compile(
    r"\b(?:segun|dijo|dicen|dice|declaro|declara|declararon|afirma|afirmo|asegura|"
    r"aseguro|publica|publico|informa|informo|senala|senalo|denuncia|denuncio|"
    r"reporta|reporto|cita|revela|revelo|confirma|confirmo|advirtio|advierte)\b",
    re.IGNORECASE,
)

# Listas funcionales cerradas heredadas del EDA (Fase 2). No son recursos
# externos: son formas funcionales del espanol usadas como anotacion.
SECOND_PERSON_LEXICON: Final[frozenset[str]] = frozenset(
    {
        "tu", "tus", "te", "ti", "usted", "ustedes", "vos", "vosotros",
        "vosotras", "contigo", "os", "vuestro", "vuestra", "vuestros",
        "vuestras", "tuyo", "tuya", "tuyos", "tuyas",
    }
)
FIRST_PERSON_SINGULAR: Final[frozenset[str]] = frozenset(
    {"yo", "me", "mi", "mis", "mio", "mia", "conmigo"}
)
FIRST_PERSON_PLURAL: Final[frozenset[str]] = frozenset(
    {"nosotros", "nosotras", "nos", "nuestro", "nuestra", "nuestros", "nuestras"}
)
THIRD_PERSON_SINGULAR: Final[frozenset[str]] = frozenset(
    {"el", "ella", "ello", "le", "lo", "la", "su", "sus", "suyo", "suya", "consigo"}
)
THIRD_PERSON_PLURAL: Final[frozenset[str]] = frozenset(
    {"ellos", "ellas", "les", "los", "las", "suyos", "suyas"}
)
COLLECTIVE_QUANTIFIERS: Final[frozenset[str]] = frozenset(
    {
        "todos", "todas", "todo", "toda", "muchos", "muchas", "algunos",
        "algunas", "ningun", "ninguna", "ningunos", "ningunas", "cualquiera",
        "cualesquiera", "varios", "varias", "cada", "ambos", "ambas", "gente",
        "pueblo", "mayoria", "minoria", "multitud", "monton", "masa", "grupo",
    }
)
IMPERATIVE_LEXICON: Final[frozenset[str]] = frozenset(
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
CLITIC_ENDINGS: Final[tuple[str, ...]] = (
    "me", "te", "se", "lo", "la", "le", "nos", "os", "los", "las", "les",
)
INFINITIVE_CLITIC_PATTERN: Final = re.compile(r"r(me|te|se|lo|la|le|nos|os|los|las|les)$")
FIRST_PERSON_PLURAL_PATTERN: Final = re.compile(r"(mos|amos|emos|imos)$")

COLUMNAS_H11: Final[tuple[str, ...]] = (
    "n_caracteres",
    "texto_corto_40",
    "n_risas",
    "tiene_risa",
    "n_afectivos",
    "tiene_afectivo",
    "n_diminutivos",
    "tiene_diminutivo",
    "cooc_primera_segunda",
    "vocativo_posesivo",
    "n_marcas_cita",
    "tiene_comillas",
    "n_emoticones",
    "tiene_emoticon",
    "n_exclamaciones",
    "n_interrogaciones",
    "n_puntuacion_repetida",
    "insulto_x_risa",
    "insulto_x_afectivo",
    "insulto_x_vocativo",
    "insulto_x_sin_colectiva",
    "insulto_x_segunda_persona",
)

COLUMNAS_H12: Final[tuple[str, ...]] = (
    "tiene_mencion_inicial",
    "n_nombres_persona",
    "tiene_nombre_persona",
    "tiene_segunda_persona",
    "tiene_imperativo",
    "tiene_referencia_colectiva",
    "n_lexico_colectivo",
    "ratio_lexico_colectivo",
    "tiene_lexico_colectivo",
    "tiene_url",
    "n_marcadores_atribucion",
    "tiene_atribucion",
    "tiene_tercera_persona_singular",
    "tiene_tercera_persona_plural",
    "mencion_multiple",
)

COLUMNAS_H14: Final[tuple[str, ...]] = (
    "n_lexico_agresivo",
    "ratio_lexico_agresivo",
    "tiene_lexico_agresivo",
    "n_lexico_agresivo_hs",
    "ratio_lexico_agresivo_hs",
    "tiene_lexico_agresivo_hs",
    "n_lexico_agresivo_exclusivo",
    "tiene_lexico_agresivo_exclusivo",
    "lexico_agresivo_x_colectiva",
)

# Columnas que dependen del pliegue de entrenamiento (lexicos inducidos, lista
# de palabras frecuentes y las interacciones que usan el lexico inducido). El
# resto de columnas son deterministas y se anotan una sola vez.
COLUMNAS_DEPENDIENTES_PLIEGUE: Final[tuple[str, ...]] = (
    "n_nombres_persona",
    "tiene_nombre_persona",
    "n_lexico_colectivo",
    "ratio_lexico_colectivo",
    "tiene_lexico_colectivo",
    "n_lexico_agresivo",
    "ratio_lexico_agresivo",
    "tiene_lexico_agresivo",
    "n_lexico_agresivo_hs",
    "ratio_lexico_agresivo_hs",
    "tiene_lexico_agresivo_hs",
    "n_lexico_agresivo_exclusivo",
    "tiene_lexico_agresivo_exclusivo",
    "lexico_agresivo_x_colectiva",
    "insulto_x_risa",
    "insulto_x_afectivo",
    "insulto_x_vocativo",
    "insulto_x_sin_colectiva",
    "insulto_x_segunda_persona",
)

# ===== SECCION 3: especificacion de configuraciones =====


@dataclass(frozen=True)
class Configuracion:
    """Especificacion de una configuracion evaluada en las Fases 6-8.

    Attributes:
        identificador: id del experimento (fila de ``experiments.csv``).
        descripcion: descripcion legible para el reporte.
        bloques: bloques densos incluidos ("H11", "H12", "H14").
        usa_tfidf: si la configuracion concatena la representacion REP-05.
        c: valor de ``C`` de la regresion logistica.
        familia: "referencia", "densa" o "union" (define el umbral de D-012).
        pregunta: pregunta que responde la configuracion.
        hipotesis: hipotesis asociada.
        registrar: si la configuracion se registra en ``experiments.csv``.
    """

    identificador: str
    descripcion: str
    bloques: tuple[str, ...]
    usa_tfidf: bool
    c: float
    familia: str
    pregunta: str
    hipotesis: str
    registrar: bool = True


PREGUNTA_H11: Final[str] = (
    "Reduce el bloque de registro bromista e interpersonal los falsos positivos "
    "de HS y AG del modelo union?"
)
PREGUNTA_H12: Final[str] = (
    "Mejora el bloque de objetivo individual frente a colectivo la F1 de TR en "
    "los textos sin destinatario explicito?"
)
PREGUNTA_H14: Final[str] = (
    "Recupera el lexico agresivo inducido dentro del pliegue los falsos "
    "negativos de AG contra grupos?"
)

CONFIGURACIONES: Final[tuple[Configuracion, ...]] = (
    Configuracion(
        identificador="MOD-03b_REF",
        descripcion=(
            "control recalculado: union TF-IDF REP-05 + LR C=2.0 balanced (mismo "
            "split); no se registra en experiments.csv porque MOD-03b_BEST ya esta"
        ),
        bloques=(),
        usa_tfidf=True,
        c=2.0,
        familia="referencia",
        pregunta="Reproduce la particion la F1 OOF publicada de MOD-03b_BEST?",
        hipotesis="La particion de D-004 reconstruida es identica a la de la Fase 4-5",
        registrar=False,
    ),
    Configuracion(
        identificador="B5-DENSE",
        descripcion=(
            "baseline denso B5: LR C=1.0 balanced sobre los bloques H11+H12+H14 "
            "escalados con StandardScaler (sin TF-IDF)"
        ),
        bloques=("H11", "H12", "H14"),
        usa_tfidf=False,
        c=1.0,
        familia="densa",
        pregunta=(
            "Cuanto F1 aportan por si solas las features densas de registro, "
            "objetivo y deshumanizacion?"
        ),
        hipotesis=(
            "Las conjunciones explicitas de registro y objetivo aportan senal que "
            "los n-gramas no representan (H11, H12, H14)"
        ),
    ),
    Configuracion(
        identificador="HIB-UNION",
        descripcion=(
            "hibrido union: hstack(TF-IDF REP-05, bloques densos H11+H12+H14 "
            "escalados) + LR C=2.0 balanced"
        ),
        bloques=("H11", "H12", "H14"),
        usa_tfidf=True,
        c=2.0,
        familia="union",
        pregunta="Aportan los tres bloques densos sobre la representacion ganadora?",
        hipotesis=(
            "La union de n-gramas con conjunciones densas reduce los falsos "
            "positivos de broma y los falsos negativos de odio implicito"
        ),
    ),
    Configuracion(
        identificador="HIB-H11",
        descripcion=(
            "ablacion H11: hstack(TF-IDF REP-05, bloque H11 escalado) + LR C=2.0"
        ),
        bloques=("H11",),
        usa_tfidf=True,
        c=2.0,
        familia="union",
        pregunta=PREGUNTA_H11,
        hipotesis=(
            "El registro bromista (risa, afecto, diminutivos, vocativo con "
            "posesivo, cita) modula la lectura del insulto (H11)"
        ),
    ),
    Configuracion(
        identificador="HIB-H12",
        descripcion=(
            "ablacion H12: hstack(TF-IDF REP-05, bloque H12 escalado) + LR C=2.0"
        ),
        bloques=("H12",),
        usa_tfidf=True,
        c=2.0,
        familia="union",
        pregunta=PREGUNTA_H12,
        hipotesis=(
            "Separar objetivo individual de objetivo colectivo (nombre propio, "
            "segunda persona, referencia colectiva, atribucion) mejora TR (H12)"
        ),
    ),
    Configuracion(
        identificador="HIB-H14",
        descripcion=(
            "ablacion H14: hstack(TF-IDF REP-05, bloque H14 escalado) + LR C=2.0"
        ),
        bloques=("H14",),
        usa_tfidf=True,
        c=2.0,
        familia="union",
        pregunta=PREGUNTA_H14,
        hipotesis=(
            "El lexico agresivo inducido dentro del pliegue y su cruce con la "
            "referencia colectiva recuperan la agresion implicita de AG (H14)"
        ),
    ),
)

# ===== SECCION 4: carga de datos y particion de validacion =====


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

    Replica exactamente la particion de ``experiments/03_variantes_representaciones.py``
    (mismo estimador, misma semilla y mismo orden de filas) para que las
    diferencias de F1 sean atribuibles a las features y no al reparto.

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
            f"pliegue {numero}: {len(entrenamiento)} train / {len(validacion)} validacion"
        )
    return pliegues


# ===== SECCION 5: anotacion spaCy y rasgos deterministas =====


def cargar_anotador():
    """Carga el anotador linguistico de spaCy sin el componente NER.

    Se usa exclusivamente para leer POS, lemas y morfologia (D-003); los
    vectores preentrenados del modelo estan vetados y no se consultan.

    Returns:
        Pipeline de spaCy con ``es_core_news_md`` y ``ner`` excluido.

    Raises:
        OSError: si el modelo no esta instalado en el entorno.
    """
    import spacy

    return spacy.load(CONFIG["spacy_model"], exclude=["ner"])


def _es_candidato_persona(token: Any, documento: Any, inicia_con_mencion: bool) -> bool:
    """Aplica la heuristica de nombre propio de persona del EDA (Fase 2).

    Criterio: capitalizacion inicial con resto en minuscula, token alfabetico de
    3 o mas caracteres, POS de nombre propio o sustantivo, y contexto de vocativo
    o de mencion (va inmediatamente despues de una mencion o el documento empieza
    con mencion y el token esta entre los cinco primeros). El filtro por palabra
    frecuente del corpus no se aplica aqui porque depende del pliegue: se aplica
    despues con las palabras frecuentes del pliegue de entrenamiento.

    Args:
        token: token de spaCy.
        documento: documento de spaCy al que pertenece el token.
        inicia_con_mencion: True si el documento empieza con una mencion.

    Returns:
        True si el token es candidato a nombre propio de persona.
    """
    if token.i == 0 or token.pos_ not in ("PROPN", "NOUN") or not token.is_alpha:
        return False
    if len(token.text) < 3 or not token.text[0].isupper() or not token.text[1:].islower():
        return False
    anterior = documento[token.i - 1].text.startswith("@")
    return bool(anterior or (inicia_con_mencion and token.i <= 5))


def _contar_personas(documento: Any) -> dict[str, Any]:
    """Cuenta personas gramaticales, imperativos, colectivos y candidatos.

    Args:
        documento: documento anotado por spaCy (D-003, sin vectores).

    Returns:
        Diccionario con los conteos por persona, el numero de imperativos, el
        numero de referencias colectivas estructurales, los lemas, las formas
        alfabeticas en minuscula y los candidatos a nombre propio de persona.
    """
    pronombres = {"1s": 0, "1p": 0, "2": 0, "3s": 0, "3p": 0}
    verbos = {"1s": 0, "1p": 0, "2": 0, "3s": 0, "3p": 0}
    imperativos = 0
    colectivos = 0
    candidatos: list[str] = []
    lemas: list[str] = []
    palabras: list[str] = []
    inicia_con_mencion = bool(INITIAL_MENTION_PATTERN.match(documento.text))
    for token in documento:
        forma = token.text.lower()
        lema = token.lemma_.lower()
        lemas.append(lema)
        if token.is_alpha:
            palabras.append(forma)
        persona = token.morph.get("Person")
        persona = persona[0] if persona else ""
        numero = token.morph.get("Number")
        numero = numero[0] if numero else ""
        if token.pos_ == "PRON":
            if lema in SECOND_PERSON_LEXICON or forma in SECOND_PERSON_LEXICON:
                pronombres["2"] += 1
            elif lema in FIRST_PERSON_SINGULAR or forma in FIRST_PERSON_SINGULAR:
                pronombres["1s"] += 1
            elif lema in FIRST_PERSON_PLURAL or forma in FIRST_PERSON_PLURAL:
                pronombres["1p"] += 1
            elif lema in THIRD_PERSON_PLURAL or forma in THIRD_PERSON_PLURAL:
                pronombres["3p"] += 1
            elif lema in THIRD_PERSON_SINGULAR or forma in THIRD_PERSON_SINGULAR:
                pronombres["3s"] += 1
        if token.pos_ == "DET" and forma in SECOND_PERSON_LEXICON:
            pronombres["2"] += 1
        if token.pos_ in ("VERB", "AUX"):
            if persona == "1":
                verbos["1s" if numero != "Plur" else "1p"] += 1
            elif persona == "2":
                verbos["2"] += 1
            elif persona == "3":
                verbos["3s" if numero != "Plur" else "3p"] += 1
            if token.morph.get("Mood") == ["Imp"]:
                imperativos += 1
            if (
                len(token.text) >= 4
                and token.text.isalpha()
                and forma.endswith(CLITIC_ENDINGS)
                and not INFINITIVE_CLITIC_PATTERN.search(forma)
                and not FIRST_PERSON_PLURAL_PATTERN.search(forma)
            ):
                imperativos += 1
            if forma in IMPERATIVE_LEXICON or lema in IMPERATIVE_LEXICON:
                imperativos += 1
        if token.pos_ in ("NOUN", "PROPN", "ADJ") and numero == "Plur":
            anterior = documento[token.i - 1] if token.i > 0 else None
            if (
                anterior is not None
                and anterior.pos_ == "DET"
                and anterior.morph.get("Number") == ["Plur"]
            ):
                colectivos += 1
        if token.pos_ in ("NOUN", "PROPN") and lema in COLLECTIVE_QUANTIFIERS:
            colectivos += 1
        if _es_candidato_persona(token, documento, inicia_con_mencion):
            candidatos.append(forma)
    return {
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
        "n_imperativos": imperativos,
        "n_colectivos": colectivos,
        "candidatos_persona": candidatos,
        "lemas": lemas,
        "palabras": palabras,
    }


def _rasgos_superficie(texto: str) -> dict[str, Any]:
    """Extrae los conteos y banderas deterministas de superficie de un texto.

    Args:
        texto: texto crudo del documento.

    Returns:
        Diccionario con conteos y banderas binarias (sin ningun parametro
        ajustado).
    """
    n_marcas_cita = (
        texto.count('"') + texto.count("'") + texto.count("\u00ab") + texto.count("\u00bb")
    )
    return {
        "n_caracteres": len(texto),
        "n_palabras": len(WORD_PATTERN.findall(texto)),
        "texto_corto_40": int(len(texto) < 40),
        "n_menciones": len(MENTION_PATTERN.findall(texto)),
        "tiene_mencion": int(bool(MENTION_PATTERN.search(texto))),
        "tiene_mencion_inicial": int(bool(INITIAL_MENTION_PATTERN.match(texto))),
        "tiene_hashtag": int(bool(HASHTAG_PATTERN.search(texto))),
        "tiene_url": int(bool(URL_PATTERN.search(texto))),
        "n_exclamaciones": texto.count("!"),
        "n_interrogaciones": texto.count("?"),
        "tiene_pregunta": int("?" in texto),
        "n_puntuacion_repetida": len(PUNCT_REPEAT_PATTERN.findall(texto)),
        "n_risas": len(RE_RISA.findall(texto)),
        "tiene_risa": int(bool(RE_RISA.search(texto))),
        "n_afectivos": len(RE_AFECTIVO.findall(texto)),
        "tiene_afectivo": int(bool(RE_AFECTIVO.search(texto))),
        "n_diminutivos": len(RE_DIMINUTIVO.findall(texto)),
        "tiene_diminutivo": int(bool(RE_DIMINUTIVO.search(texto))),
        "n_marcas_cita": n_marcas_cita,
        "tiene_comillas": int(n_marcas_cita > 0),
        "n_emoticones": len(EMOTICON_PATTERN.findall(texto)),
        "tiene_emoticon": int(bool(EMOTICON_PATTERN.search(texto))),
        "n_marcadores_atribucion": len(RE_ATRIBUCION.findall(texto)),
        "tiene_atribucion": int(bool(RE_ATRIBUCION.search(texto))),
        "mencion_multiple": int(len(MENTION_PATTERN.findall(texto)) >= 2),
        "vocativo_posesivo": int(bool(RE_VOCATIVO_POSESIVO.match(texto))),
    }


def _rasgos_documento(documento: Any) -> dict[str, Any]:
    """Combina los rasgos de superficie y los morfosintacticos de un documento.

    Args:
        documento: documento anotado por spaCy.

    Returns:
        Diccionario con todos los rasgos deterministas del documento.
    """
    personas = _contar_personas(documento)
    registro = {**personas, **_rasgos_superficie(documento.text)}
    tiene_primera = int(
        personas["pron_1s"]
        + personas["pron_1p"]
        + personas["verbo_1s"]
        + personas["verbo_1p"]
        > 0
    )
    tiene_segunda = int(
        personas["pron_2"] > 0
        or personas["verbo_2"] > 0
        or personas["n_imperativos"] > 0
    )
    registro["tiene_primera_persona"] = tiene_primera
    registro["tiene_segunda_persona"] = tiene_segunda
    registro["cooc_primera_segunda"] = int(tiene_primera and tiene_segunda)
    registro["tiene_imperativo"] = int(personas["n_imperativos"] > 0)
    registro["tiene_referencia_colectiva"] = int(personas["n_colectivos"] > 0)
    registro["tiene_tercera_persona_singular"] = int(
        personas["pron_3s"] > 0 or personas["verbo_3s"] > 0
    )
    registro["tiene_tercera_persona_plural"] = int(
        personas["pron_3p"] > 0 or personas["verbo_3p"] > 0
    )
    return registro


def anotar_documentos(anotador: Any, textos: Sequence[str]) -> pd.DataFrame:
    """Anota todos los textos una sola vez (rasgos deterministas).

    Args:
        anotador: pipeline de spaCy.
        textos: lista de textos crudos.

    Returns:
        DataFrame con un registro por documento y los rasgos deterministas.
    """
    filas: list[dict[str, Any]] = []
    for numero, documento in enumerate(
        anotador.pipe(list(textos), batch_size=CONFIG["batch_size_spacy"]), start=1
    ):
        filas.append(_rasgos_documento(documento))
        if numero % 500 == 0:
            log(f"spaCy: {numero} documentos anotados")
    return pd.DataFrame(filas)


def palabras_frecuentes(anotaciones: pd.DataFrame, minimo: int) -> frozenset[str]:
    """Induce las palabras de alta frecuencia del pliegue de entrenamiento.

    Se usa como filtro del detector heuristico de nombre de persona: una palabra
    que aparece en minuscula muchas veces en el pliegue no puede ser un nombre
    propio. La lista se induce solo con los documentos de entrenamiento del
    pliegue y nunca con validacion.

    Args:
        anotaciones: rasgos anotados de los documentos de entrenamiento.
        minimo: frecuencia minima en minuscula para entrar en la lista.

    Returns:
        Conjunto de palabras frecuentes.
    """
    contador: Counter[str] = Counter()
    for palabras in anotaciones["palabras"]:
        contador.update(palabras)
    return frozenset(
        palabra for palabra, frecuencia in contador.items() if frecuencia >= minimo
    )


# ===== SECCION 6: lexicos inducidos dentro del pliegue =====


def inducir_lexico_log_odds(
    lemas_documento: Sequence[Sequence[str]], positivo: np.ndarray
) -> pd.DataFrame:
    """Induce un lexico por log-odds con priori de Dirichlet (Monroe et al. 2008).

    Metodologia identica a la de ``experiments/01_eda_linguistico.py``: log-odds
    entre documentos de la clase positiva y el resto, restringido a lemas
    alfabeticos de 3 o mas caracteres con frecuencia total >= 3 y presentes en al
    menos 2 documentos positivos. Toda la informacion proviene del pliegue de
    entrenamiento: ni validacion ni test entran en la induccion.

    Args:
        lemas_documento: lista de lemas por documento del pliegue de entrenamiento.
        positivo: vector booleano alineado con ``lemas_documento`` que marca la
            clase positiva de la induccion.

    Returns:
        Tabla con lema, frecuencias por clase, documentos positivos, log-odds,
        z-score, rango y bandera de seleccion; ordenada por z-score descendente.
    """
    conteo_positivo: Counter[str] = Counter()
    conteo_negativo: Counter[str] = Counter()
    documentos_positivos: Counter[str] = Counter()
    for indice, lemas in enumerate(lemas_documento):
        conteo = Counter(lemas)
        if positivo[indice]:
            conteo_positivo.update(conteo)
            documentos_positivos.update(set(conteo))
        else:
            conteo_negativo.update(conteo)
    total_positivo = sum(conteo_positivo.values())
    total_negativo = sum(conteo_negativo.values())
    vocabulario = sorted(set(conteo_positivo) | set(conteo_negativo))
    if not vocabulario:
        return pd.DataFrame(
            columns=[
                "lema",
                "frecuencia_positivo",
                "frecuencia_negativo",
                "documentos_positivos",
                "log_odds",
                "z_score",
                "rango",
                "seleccionado",
            ]
        )
    filas: list[dict[str, Any]] = []
    for lema in vocabulario:
        if not lema.isalpha() or len(lema) < CONFIG["min_longitud_lema"]:
            continue
        frecuencia_positiva = conteo_positivo.get(lema, 0)
        frecuencia_negativa = conteo_negativo.get(lema, 0)
        if frecuencia_positiva + frecuencia_negativa < CONFIG["min_frecuencia_lexico"]:
            continue
        if documentos_positivos.get(lema, 0) < CONFIG["min_documentos_positivos"]:
            continue
        alpha = CONFIG["alpha_prior"]
        proporcion_positiva = (frecuencia_positiva + alpha) / (
            total_positivo + alpha * len(vocabulario)
        )
        proporcion_negativa = (frecuencia_negativa + alpha) / (
            total_negativo + alpha * len(vocabulario)
        )
        log_odds = float(np.log(proporcion_positiva / proporcion_negativa))
        varianza = 1.0 / (frecuencia_positiva + alpha) + 1.0 / (
            frecuencia_negativa + alpha
        )
        filas.append(
            {
                "lema": lema,
                "frecuencia_positivo": frecuencia_positiva,
                "frecuencia_negativo": frecuencia_negativa,
                "documentos_positivos": documentos_positivos.get(lema, 0),
                "log_odds": round(log_odds, 4),
                "z_score": round(float(log_odds / np.sqrt(varianza)), 3),
            }
        )
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return tabla
    tabla = tabla.sort_values(
        ["z_score", "frecuencia_positivo", "lema"], ascending=[False, False, True]
    ).reset_index(drop=True)
    tabla["rango"] = np.arange(1, len(tabla) + 1)
    tabla["seleccionado"] = (tabla["rango"] <= CONFIG["top_k_lexico"]).astype(int)
    return tabla


@dataclass(frozen=True)
class LexicosPliegue:
    """Lexicos inducidos con los datos del pliegue de entrenamiento.

    Attributes:
        agresivo: lemas del lexico de agresividad (log-odds AG).
        agresivo_nucleo: subconjunto de ``agresivo`` con z-score alto.
        hs: lemas del lexico inducido por log-odds HS.
        colectivo: lemas inducidos por log-odds de la referencia colectiva
            estructural (sin usar etiquetas de toxicidad).
    """

    agresivo: frozenset[str]
    agresivo_nucleo: frozenset[str]
    hs: frozenset[str]
    colectivo: frozenset[str]


def _lemas_seleccionados(tabla: pd.DataFrame) -> frozenset[str]:
    """Extrae los lemas seleccionados de una tabla de lexico inducido.

    Args:
        tabla: tabla devuelta por :func:`inducir_lexico_log_odds`.

    Returns:
        Conjunto de lemas con ``seleccionado == 1``.
    """
    if tabla.empty:
        return frozenset()
    return frozenset(tabla.loc[tabla["seleccionado"] == 1, "lema"].astype(str))


def _lemas_nucleo(tabla: pd.DataFrame, z_minimo: float) -> frozenset[str]:
    """Extrae el nucleo de un lexico inducido (z-score por encima del umbral).

    Args:
        tabla: tabla devuelta por :func:`inducir_lexico_log_odds`.
        z_minimo: umbral de z-score para entrar en el nucleo.

    Returns:
        Conjunto de lemas del nucleo.
    """
    if tabla.empty:
        return frozenset()
    return frozenset(tabla.loc[tabla["z_score"] >= z_minimo, "lema"].astype(str))


def inducir_lexicos_pliegue(
    anotaciones: pd.DataFrame, etiquetas: np.ndarray, train_idx: np.ndarray
) -> tuple[LexicosPliegue, dict[str, Any]]:
    """Induce los tres lexicos del pliegue con los datos de entrenamiento.

    Args:
        anotaciones: rasgos anotados (incluye los lemas por documento).
        etiquetas: matriz ``(n, 3)`` de etiquetas en el orden HS, TR, AG.
        train_idx: indices posicionales del pliegue de entrenamiento.

    Returns:
        Tupla ``(lexicos, diagnostico)``; el diagnostico guarda el numero de
        lemas inducidos y los primeros lemas por z-score para el reporte.
    """
    lemas_train = anotaciones["lemas"].iloc[train_idx].tolist()
    etiquetas_train = etiquetas[train_idx]
    colectiva_train = anotaciones["tiene_referencia_colectiva"].iloc[train_idx].to_numpy()
    tabla_agresivo = inducir_lexico_log_odds(lemas_train, etiquetas_train[:, 2] == 1)
    tabla_hs = inducir_lexico_log_odds(lemas_train, etiquetas_train[:, 0] == 1)
    tabla_colectivo = inducir_lexico_log_odds(lemas_train, colectiva_train == 1)
    tabla_colectivo = tabla_colectivo.head(CONFIG["top_k_lexico_colectivo"]).copy()
    tabla_colectivo["seleccionado"] = 1
    lexicos = LexicosPliegue(
        agresivo=_lemas_seleccionados(tabla_agresivo),
        agresivo_nucleo=_lemas_nucleo(tabla_agresivo, CONFIG["z_nucleo_insulto"]),
        hs=_lemas_seleccionados(tabla_hs),
        colectivo=_lemas_seleccionados(tabla_colectivo),
    )
    diagnostico = {
        "n_agresivo": len(lexicos.agresivo),
        "n_agresivo_nucleo": len(lexicos.agresivo_nucleo),
        "n_hs": len(lexicos.hs),
        "n_colectivo": len(lexicos.colectivo),
        "top_agresivo": tabla_agresivo.head(15)["lema"].astype(str).tolist(),
        "top_agresivo_nucleo": sorted(lexicos.agresivo_nucleo)[:15],
        "top_colectivo": tabla_colectivo.head(15)["lema"].astype(str).tolist(),
    }
    return lexicos, diagnostico


# ===== SECCION 7: construccion de los bloques densos por pliegue =====


def _contar_lemas(lista_lemas: Sequence[str], lexico: frozenset[str]) -> int:
    """Cuenta cuantos lemas de una lista pertenecen a un lexico.

    Args:
        lista_lemas: lemas del documento.
        lexico: conjunto de lemas del lexico.

    Returns:
        Numero de coincidencias.
    """
    return sum(1 for lema in lista_lemas if lema in lexico)


def construir_denso_pliegue(
    anotaciones: pd.DataFrame, etiquetas: np.ndarray, train_idx: np.ndarray
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Construye la matriz densa del pliegue (columnas deterministas e inducidas).

    Las columnas que dependen del pliegue usan exclusivamente el pliegue de
    entrenamiento: los lexicos inducidos y la lista de palabras frecuentes
    (filtro del detector de nombres propios). La matriz cubre los 1.731
    documentos; el llamador selecciona las filas de entrenamiento y validacion,
    de modo que las filas de validacion se calculan con parametros que no las
    vieron.

    Args:
        anotaciones: rasgos anotados de todos los documentos.
        etiquetas: matriz ``(n, 3)`` de etiquetas.
        train_idx: indices posicionales del pliegue de entrenamiento.

    Returns:
        Tupla ``(datos, diagnostico)`` con la matriz densa (todas las columnas
        de los tres bloques) y el diagnostico de los lexicos inducidos.
    """
    frecuentes = palabras_frecuentes(
        anotaciones.iloc[train_idx], CONFIG["min_frecuencia_palabra_frecuente"]
    )
    lexicos, diagnostico = inducir_lexicos_pliegue(anotaciones, etiquetas, train_idx)
    agresivo_exclusivo = lexicos.agresivo.difference(lexicos.hs)
    palabras = anotaciones["n_palabras"].to_numpy(dtype=float)
    datos = pd.DataFrame(index=anotaciones.index)
    for columna in anotaciones.columns:
        if columna not in ("lemas", "palabras", "candidatos_persona"):
            datos[columna] = anotaciones[columna].to_numpy()
    datos["n_nombres_persona"] = [
        sum(1 for candidato in candidatos if candidato not in frecuentes)
        for candidatos in anotaciones["candidatos_persona"]
    ]
    datos["tiene_nombre_persona"] = (datos["n_nombres_persona"] > 0).astype(int)
    conteos = {
        "n_lexico_colectivo": lexicos.colectivo,
        "n_lexico_agresivo": lexicos.agresivo,
        "n_lexico_agresivo_hs": lexicos.hs,
        "n_lexico_agresivo_exclusivo": agresivo_exclusivo,
    }
    for columna, lexico in conteos.items():
        datos[columna] = [
            _contar_lemas(lista, lexico) for lista in anotaciones["lemas"]
        ]
        datos[f"ratio_{columna.removeprefix('n_')}"] = (
            datos[columna].to_numpy(dtype=float) / np.maximum(palabras, 1.0)
        )
        datos[f"tiene_{columna.removeprefix('n_')}"] = (datos[columna] > 0).astype(int)
    datos["insulto"] = [
        int(_contar_lemas(lista, lexicos.agresivo_nucleo) > 0)
        for lista in anotaciones["lemas"]
    ]
    datos["insulto_x_risa"] = datos["insulto"] * datos["tiene_risa"]
    datos["insulto_x_afectivo"] = datos["insulto"] * datos["tiene_afectivo"]
    datos["insulto_x_vocativo"] = datos["insulto"] * datos["vocativo_posesivo"]
    datos["insulto_x_sin_colectiva"] = datos["insulto"] * (
        1 - datos["tiene_referencia_colectiva"]
    )
    datos["insulto_x_segunda_persona"] = datos["insulto"] * datos["tiene_segunda_persona"]
    datos["lexico_agresivo_x_colectiva"] = (
        datos["n_lexico_agresivo"] * datos["tiene_referencia_colectiva"]
    )
    return datos, diagnostico


def columnas_de_bloques(bloques: Sequence[str]) -> list[str]:
    """Devuelve las columnas densas de los bloques pedidos.

    Args:
        bloques: subconjunto de ("H11", "H12", "H14").

    Returns:
        Lista de columnas en el orden de los bloques.

    Raises:
        ValueError: si un bloque no esta definido.
    """
    mapa = {"H11": COLUMNAS_H11, "H12": COLUMNAS_H12, "H14": COLUMNAS_H14}
    columnas: list[str] = []
    for bloque in bloques:
        if bloque not in mapa:
            raise ValueError(f"Bloque denso no definido: {bloque}")
        columnas.extend(mapa[bloque])
    return columnas


# ===== SECCION 8: motor de evaluacion =====


@dataclass(frozen=True)
class ResultadoConfig:
    """Resultado completo de una configuracion evaluada.

    Attributes:
        configuracion: especificacion evaluada.
        metricas_oof: metricas sobre las predicciones out-of-fold completas.
        metricas_pliegues: metricas por pliegue.
        metricas_entrenamiento: metricas por pliegue sobre el propio pliegue de
            entrenamiento (diagnostico anti-sobreajuste; no es una estimacion de
            generalizacion).
        prediccion: matriz ``(n, 3)`` de predicciones binarias out-of-fold.
        dimension: numero de features (TF-IDF diagnostico + bloque denso).
        n_features_densas: numero de columnas densas usadas.
        segundos: tiempo de evaluacion.
    """

    configuracion: Configuracion
    metricas_oof: MetricasMultiEtiqueta
    metricas_pliegues: list[dict[str, Any]]
    metricas_entrenamiento: list[dict[str, Any]]
    prediccion: np.ndarray
    dimension: int
    n_features_densas: int
    segundos: float

    @property
    def f1_train_medio(self) -> float:
        """Media de ``f1_media_columnas`` sobre el pliegue de entrenamiento."""
        return float(
            np.mean([metrica["f1_media_columnas"] for metrica in self.metricas_entrenamiento])
        )


def construir_vectorizador_rep05() -> Any:
    """Crea la union TF-IDF de REP-05 (word (1,2) + char_wb (3,5), D-004).

    Returns:
        Vectorizador TF-IDF si solo hay un bloque o concatenacion sparse de
        varios bloques (misma definicion que la Fase 4-5).
    """
    return FeatureUnion(
        [
            ("bloque_0", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), **TFIDF_PARAMS)),
            (
                "bloque_1",
                TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **TFIDF_PARAMS),
            ),
        ]
    )


def construir_modelo(c: float) -> LogisticRegression:
    """Instancia la regresion logistica binaria de la configuracion.

    Args:
        c: valor de ``C`` (inverso de la regularizacion).

    Returns:
        Clasificador sin ajustar (LR balanced, liblinear), igual que el mejor
        modelo de la Fase 4-5.
    """
    return LogisticRegression(
        C=c,
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
        random_state=CONFIG["random_state"],
    )


def evaluar_configuracion(
    configuracion: Configuracion,
    textos: pd.Series,
    etiquetas: np.ndarray,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
    densos: Sequence[tuple[pd.DataFrame, dict[str, Any]]],
) -> ResultadoConfig:
    """Evalua una configuracion con la particion agrupada compartida.

    En cada pliegue: se seleccionan las columnas densas de los bloques pedidos,
    se ajusta ``StandardScaler`` solo con el pliegue de entrenamiento, se
    concatena (si aplica) la matriz TF-IDF ajustada tambien solo con el pliegue
    de entrenamiento y se entrena un clasificador binario por etiqueta.

    Args:
        configuracion: configuracion a evaluar.
        textos: serie de textos crudos (misma longitud que el train).
        etiquetas: matriz ``(n, 3)`` con las etiquetas verdaderas.
        pliegues: particion agrupada compartida.
        densos: matriz densa y diagnostico por pliegue (en el orden de pliegues).

    Returns:
        Resultado de la configuracion con metricas OOF y por pliegue.
    """
    inicio = time.perf_counter()
    columnas = columnas_de_bloques(configuracion.bloques)
    prediccion = np.zeros_like(etiquetas)
    metricas_entrenamiento: list[dict[str, Any]] = []
    for numero, ((entrenamiento, validacion), (denso, _)) in enumerate(
        zip(pliegues, densos, strict=True), start=1
    ):
        if columnas:
            escalador = StandardScaler().fit(denso[columnas].iloc[entrenamiento])
            denso_train = escalador.transform(denso[columnas].iloc[entrenamiento])
            denso_validacion = escalador.transform(denso[columnas].iloc[validacion])
        else:
            denso_train = np.empty((len(entrenamiento), 0))
            denso_validacion = np.empty((len(validacion), 0))
        if configuracion.usa_tfidf:
            vectorizador = construir_vectorizador_rep05()
            matriz_train = vectorizador.fit_transform(textos.iloc[entrenamiento])
            matriz_validacion = vectorizador.transform(textos.iloc[validacion])
            if columnas:
                matriz_train = sparse_hstack(
                    [matriz_train, csr_matrix(denso_train)], format="csr"
                )
                matriz_validacion = sparse_hstack(
                    [matriz_validacion, csr_matrix(denso_validacion)], format="csr"
                )
        else:
            matriz_train = csr_matrix(denso_train)
            matriz_validacion = csr_matrix(denso_validacion)
        etiquetas_entrenamiento = etiquetas[entrenamiento]
        prediccion_entrenamiento = np.zeros_like(etiquetas_entrenamiento)
        for indice in range(len(ETIQUETAS)):
            modelo = construir_modelo(configuracion.c)
            modelo.fit(matriz_train, etiquetas_entrenamiento[:, indice])
            probabilidad = modelo.predict_proba(matriz_validacion)[:, 1]
            prediccion[validacion, indice] = (
                probabilidad >= CONFIG["umbral_probabilidad"]
            ).astype(int)
            probabilidad_entrenamiento = modelo.predict_proba(matriz_train)[:, 1]
            prediccion_entrenamiento[:, indice] = (
                probabilidad_entrenamiento >= CONFIG["umbral_probabilidad"]
            ).astype(int)
        metricas_entrenamiento.append(
            dict(metricas_multietiqueta(etiquetas_entrenamiento, prediccion_entrenamiento))
        )
        log(
            f"{configuracion.identificador}: pliegue {numero} "
            f"({matriz_train.shape[1]} features, {len(entrenamiento)} train)"
        )
    if configuracion.usa_tfidf:
        dimension_tfidf = int(construir_vectorizador_rep05().fit_transform(textos).shape[1])
    else:
        dimension_tfidf = 0
    metricas_oof, metricas_pliegues = calcular_metricas(etiquetas, prediccion, pliegues)
    return ResultadoConfig(
        configuracion=configuracion,
        metricas_oof=metricas_oof,
        metricas_pliegues=metricas_pliegues,
        metricas_entrenamiento=metricas_entrenamiento,
        prediccion=prediccion,
        dimension=dimension_tfidf + len(columnas),
        n_features_densas=len(columnas),
        segundos=round(time.perf_counter() - inicio, 2),
    )


def calcular_metricas(
    etiquetas: np.ndarray,
    prediccion: np.ndarray,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[MetricasMultiEtiqueta, list[dict[str, Any]]]:
    """Calcula las metricas out-of-fold globales y por pliegue.

    Args:
        etiquetas: matriz ``(n, 3)`` con las etiquetas verdaderas.
        prediccion: matriz ``(n, 3)`` con las predicciones out-of-fold.
        pliegues: particion usada para generar las predicciones.

    Returns:
        Tupla ``(metricas_globales, metricas_por_pliegue)``.
    """
    metricas_pliegues: list[dict[str, Any]] = []
    for numero, (_, validacion) in enumerate(pliegues, start=1):
        metrica = dict(metricas_multietiqueta(etiquetas[validacion], prediccion[validacion]))
        metrica["pliegue"] = numero
        metrica["n_validacion"] = len(validacion)
        metricas_pliegues.append(metrica)
    return metricas_multietiqueta(etiquetas, prediccion), metricas_pliegues


def contar_pliegues_mejorados(
    candidato: Sequence[dict[str, Any]], referencia: Sequence[dict[str, Any]]
) -> int:
    """Cuenta en cuantos pliegues la configuracion supera a la referencia.

    Args:
        candidato: metricas por pliegue de la configuracion candidata.
        referencia: metricas por pliegue de MOD-03b_BEST.

    Returns:
        Numero de pliegues con mejora estricta.
    """
    return int(
        sum(
            metrica["f1_media_columnas"] > base["f1_media_columnas"]
            for metrica, base in zip(candidato, referencia, strict=True)
        )
    )


def decidir(mejora: float, pliegues_mejorados: int, familia: str, es_referencia: bool) -> str:
    """Aplica la regla de decision pre-registrada del proyecto.

    Args:
        mejora: diferencia de ``f1_media_columnas`` frente a MOD-03b_BEST.
        pliegues_mejorados: numero de pliegues con mejora estricta.
        familia: "referencia", "densa" o "union".
        es_referencia: True si la configuracion es el control recalculado.

    Returns:
        Etiqueta de la decision.
    """
    if es_referencia:
        return "referencia (control)"
    cumple_base = (
        mejora >= CONFIG["umbral_adopcion"]
        and pliegues_mejorados >= CONFIG["pliegues_minimos"]
    )
    if not cumple_base:
        return "descartada"
    if familia == "union" and mejora < CONFIG["umbral_union_realista"]:
        return (
            "descartada (D-012: la familia union exige >= +0.010 local antes de "
            "gastar sonda de leaderboard)"
        )
    return "adoptada"


def verificar_referencia(
    prediccion: np.ndarray, etiquetas: np.ndarray, train: pd.DataFrame
) -> dict[str, Any]:
    """Verifica la particion y la reproducibilidad contra las OOF publicadas.

    Compara las predicciones recalculadas con las probabilidades publicadas de
    ``features/oof_mod03b.npz`` (alineadas por ``id``) y con la F1 publicada.

    Args:
        prediccion: predicciones OOF recalculadas de MOD-03b.
        etiquetas: matriz ``(n, 3)`` de etiquetas verdaderas.
        train: DataFrame de train (para alinear por ``id``).

    Returns:
        Diccionario con las F1 comparadas, la coincidencia binaria y si la
        verificacion paso.
    """
    ruta: Path = CONFIG["oof_referencia_path"]
    if not ruta.exists():
        return {"ok": False, "motivo": f"no existe {ruta}"}
    datos = np.load(ruta)
    ids_publicados = [int(valor) for valor in datos["ids"]]
    posicion = {identificador: indice for indice, identificador in enumerate(ids_publicados)}
    orden = [posicion[int(identificador)] for identificador in train["id"].to_numpy()]
    probabilidad = datos["proba"][orden]
    pred_publicada = (probabilidad >= CONFIG["umbral_probabilidad"]).astype(int)
    f1_publicada_npz = float(metricas_multietiqueta(etiquetas, pred_publicada)["f1_media_columnas"])
    f1_recalculada = float(metricas_multietiqueta(etiquetas, prediccion)["f1_media_columnas"])
    coincidencia = float((pred_publicada == prediccion).mean())
    return {
        "etiquetas_identicas": bool(np.array_equal(datos["y_true"], etiquetas)),
        "f1_publicada_npz": round(f1_publicada_npz, 6),
        "f1_publicada_config": round(float(CONFIG["referencia_f1"]), 6),
        "f1_recalculada": round(f1_recalculada, 6),
        "coincidencia_binaria": round(coincidencia, 6),
        "diferencia_f1": round(f1_recalculada - f1_publicada_npz, 6),
        "ok": bool(
            abs(f1_recalculada - float(CONFIG["referencia_f1"]))
            <= CONFIG["tolerancia_referencia"]
            and coincidencia >= 0.999
        ),
    }


# ===== SECCION 9: rebanadas de la Fase 12 =====


@dataclass(frozen=True)
class MascaraRebanada:
    """Rebanada evaluable definida por una mascara booleana.

    Attributes:
        rebanada: nombre de la rebanada.
        grupo: valor de la rebanada (por ejemplo "1" o "Q4").
        mascara: mascara booleana sobre las filas de train.
    """

    rebanada: str
    grupo: str
    mascara: np.ndarray


def construir_mascaras_rebanadas(anotaciones: pd.DataFrame) -> list[MascaraRebanada]:
    """Define las rebanadas debiles de la Fase 12 y dos controles.

    Los cuartiles de longitud se calculan con ``pd.qcut`` sobre el total de
    train, igual que en ``experiments/06_error_analysis.py``.

    Args:
        anotaciones: rasgos anotados de todos los documentos.

    Returns:
        Lista de mascaras de rebanada.
    """
    cuartiles = pd.qcut(
        anotaciones["n_caracteres"], 4, labels=["Q1", "Q2", "Q3", "Q4"]
    ).astype(str)
    tiene_mencion = anotaciones["tiene_mencion"].to_numpy() == 1
    colectiva = anotaciones["tiene_referencia_colectiva"].to_numpy() == 1
    mencion_inicial = anotaciones["tiene_mencion_inicial"].to_numpy() == 1
    tiene_url = anotaciones["tiene_url"].to_numpy() == 1
    tiene_hashtag = anotaciones["tiene_hashtag"].to_numpy() == 1
    return [
        MascaraRebanada("tiene_hashtag", "1", tiene_hashtag),
        MascaraRebanada("tiene_pregunta", "1", anotaciones["tiene_pregunta"].to_numpy() == 1),
        MascaraRebanada("tiene_referencia_colectiva", "1", colectiva),
        MascaraRebanada("cuartil_longitud", "Q4", (cuartiles == "Q4").to_numpy()),
        MascaraRebanada("tiene_mencion", "0", ~tiene_mencion),
        MascaraRebanada("region_regimen_noticia", "1", tiene_url | tiene_hashtag),
        MascaraRebanada("region_no_responde_con_colectiva", "1", (~mencion_inicial) & colectiva),
        MascaraRebanada("tiene_mencion_inicial", "1", mencion_inicial),
        MascaraRebanada("cuartil_longitud", "Q1", (cuartiles == "Q1").to_numpy()),
    ]


def evaluar_rebanadas(
    modelo: str,
    prediccion: np.ndarray,
    etiquetas: np.ndarray,
    mascaras: Sequence[MascaraRebanada],
) -> list[dict[str, Any]]:
    """Evalua las rebanadas de un modelo con sus predicciones OOF.

    Args:
        modelo: identificador del modelo evaluado.
        prediccion: matriz ``(n, 3)`` de predicciones binarias.
        etiquetas: matriz ``(n, 3)`` de etiquetas verdaderas.
        mascaras: rebanadas a evaluar.

    Returns:
        Lista de filas con el F1 por etiqueta, la media de columnas y el delta
        frente a la F1 global del modelo.
    """
    global_f1 = float(metricas_multietiqueta(etiquetas, prediccion)["f1_media_columnas"])
    filas: list[dict[str, Any]] = []
    for definicion in mascaras:
        metrica = metricas_multietiqueta(etiquetas[definicion.mascara], prediccion[definicion.mascara])
        filas.append(
            {
                "modelo": modelo,
                "rebanada": definicion.rebanada,
                "grupo": definicion.grupo,
                "n": int(definicion.mascara.sum()),
                "f1_HS": round(metrica["f1_por_etiqueta"]["HS"], 4),
                "f1_TR": round(metrica["f1_por_etiqueta"]["TR"], 4),
                "f1_AG": round(metrica["f1_por_etiqueta"]["AG"], 4),
                "f1_media_columnas": round(metrica["f1_media_columnas"], 4),
                "delta_vs_global_modelo": round(
                    float(metrica["f1_media_columnas"]) - global_f1, 4
                ),
            }
        )
    return filas


# ===== SECCION 10: reportes =====


def construir_fila_reporte(
    resultado: ResultadoConfig, referencia: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Construye una fila del reporte de resultados.

    Args:
        resultado: resultado de la configuracion.
        referencia: metricas por pliegue de MOD-03b_BEST (control).

    Returns:
        Diccionario con las columnas de ``COLUMNAS_REPORTE``.
    """
    metricas = resultado.metricas_oof
    mejora = float(metricas["f1_media_columnas"]) - float(CONFIG["referencia_f1"])
    pliegues_mejorados = contar_pliegues_mejorados(resultado.metricas_pliegues, referencia)
    return {
        "experiment_id": resultado.configuracion.identificador,
        "descripcion": resultado.configuracion.descripcion,
        "familia": resultado.configuracion.familia,
        "bloques": "+".join(resultado.configuracion.bloques) or "ninguno",
        "usa_tfidf": int(resultado.configuracion.usa_tfidf),
        "C": resultado.configuracion.c,
        "f1_media_columnas": round(float(metricas["f1_media_columnas"]), 4),
        "f1_macro6": round(float(metricas["f1_macro6"]), 4),
        "f1_micro": round(float(metricas["f1_micro"]), 4),
        "f1_samples": round(float(metricas["f1_samples"]), 4),
        "exact_match": round(float(metricas["exact_match"]), 4),
        "f1_HS": round(float(metricas["f1_por_etiqueta"]["HS"]), 4),
        "f1_TR": round(float(metricas["f1_por_etiqueta"]["TR"]), 4),
        "f1_AG": round(float(metricas["f1_por_etiqueta"]["AG"]), 4),
        "std_pliegues": round(
            float(np.std([m["f1_media_columnas"] for m in resultado.metricas_pliegues])), 4
        ),
        "mejora_vs_MOD03b": round(mejora, 6),
        "mejora_en_pliegues": pliegues_mejorados,
        "f1_train_medio": round(resultado.f1_train_medio, 4),
        "brecha_train_oof": round(
            resultado.f1_train_medio - float(metricas["f1_media_columnas"]), 4
        ),
        "dimension": resultado.dimension,
        "n_features_densas": resultado.n_features_densas,
        "tiempo_segundos": resultado.segundos,
        "decision": decidir(
            mejora,
            pliegues_mejorados,
            resultado.configuracion.familia,
            es_referencia=resultado.configuracion.familia == "referencia",
        ),
    }


def construir_fila_experiments(
    fila: dict[str, Any], resultados_detalle: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Adapta una fila del reporte al esquema de ``experiments.csv``.

    Args:
        fila: fila del reporte de resultados.
        resultados_detalle: metricas por pliegue de las configuraciones (para
            adjuntar las del propio id).

    Returns:
        Diccionario con las columnas de ``COLUMNAS_EXPERIMENTS``.
    """
    configuracion = next(
        config
        for config in CONFIGURACIONES
        if config.identificador == fila["experiment_id"]
    )
    detalle = next(
        item for item in resultados_detalle if item["experiment_id"] == fila["experiment_id"]
    )
    return {
        "experiment_id": fila["experiment_id"],
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": configuracion.pregunta,
        "hypothesis": configuracion.hipotesis,
        "preprocessing": fila["descripcion"],
        "feature_groups": (
            "TFIDF_UNION+DENSAS" if configuracion.usa_tfidf else "DENSAS"
        ),
        "representation": (
            ("TF-IDF REP-05 (word 1-2 + char_wb 3-5) + " if configuracion.usa_tfidf else "")
            + f"bloques densos {fila['bloques']}"
        ),
        "model": (
            f"LogisticRegression(class_weight=balanced, C={fila['C']}, liblinear) x3"
        ),
        "hyperparameters": json.dumps(
            {
                "C": fila["C"],
                "class_weight": "balanced",
                "max_iter": 1000,
                "solver": "liblinear",
                "random_state": CONFIG["random_state"],
                "bloques_densos": list(configuracion.bloques),
                "estandarizacion": "StandardScaler ajustado por pliegue",
                "lexicos_inducidos_por_pliegue": True,
            },
            ensure_ascii=False,
        ),
        "validation": (
            "StratifiedGroupKFold(5, shuffle=True, random_state=42) grupos=near-duplicados, "
            "estratificado por combo HSxTRxAG; referencia MOD-03b_BEST 0.7927"
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
                "mejora_vs_MOD03b": fila["mejora_vs_MOD03b"],
                "mejora_en_pliegues": fila["mejora_en_pliegues"],
                "n_features_densas": fila["n_features_densas"],
                "metricas_pliegues": detalle["metricas_pliegues"],
            },
            ensure_ascii=False,
        ),
        "interpretation": (
            f"Fases 6-8, configuracion {fila['experiment_id']} "
            f"(bloques {fila['bloques']}): {fila['decision']} con delta "
            f"{fila['mejora_vs_MOD03b']:+.4f} y {fila['mejora_en_pliegues']}/5 pliegues "
            "frente a MOD-03b_BEST"
        ),
        "decision": fila["decision"],
        "date": CONFIG["fecha"],
        "script": CONFIG["script"],
        "notes": (
            "Sin fuga: TF-IDF, StandardScaler y lexicos inducidos (log-odds AG, HS y "
            "referencia colectiva) se ajustan solo con el pliegue de entrenamiento; "
            "el resto de features son regex y listas cerradas. Umbral 0.5. El test "
            "nunca se leyo."
        ),
    }


def registrar_en_experiments(filas: Sequence[dict[str, Any]]) -> int:
    """Anade (o refresca) las filas de la fase en ``experiments.csv``.

    Idempotente: elimina las filas cuyo ``experiment_id`` pertenece a esta fase
    y despues las reescribe con el esquema existente.

    Args:
        filas: filas completas con las columnas de ``COLUMNAS_EXPERIMENTS``.

    Returns:
        Numero de filas escritas.

    Raises:
        ValueError: si el esquema del archivo no coincide con el esperado.
    """
    destino: Path = CONFIG["experiments_csv"]
    existente = pd.read_csv(destino, dtype=str, keep_default_na=False)
    if list(existente.columns) != list(COLUMNAS_EXPERIMENTS):
        raise ValueError("El esquema de experiments.csv cambio inesperadamente")
    identificadores = {fila["experiment_id"] for fila in filas}
    conservadas = existente[~existente["experiment_id"].isin(identificadores)]
    nuevas = pd.DataFrame(
        [{clave: fila[clave] for clave in COLUMNAS_EXPERIMENTS} for fila in filas],
        columns=list(COLUMNAS_EXPERIMENTS),
    )
    combinado = pd.concat([conservadas, nuevas], ignore_index=True)
    combinado.to_csv(destino, index=False, encoding="utf-8")
    return len(nuevas)


def escribir_reporte(filas: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Escribe el CSV de resultados ordenado por F1 descendente.

    Args:
        filas: filas del reporte.

    Returns:
        DataFrame escrito.
    """
    reporte = pd.DataFrame(filas, columns=list(COLUMNAS_REPORTE))
    orden = reporte.sort_values(
        "f1_media_columnas", ascending=False, kind="stable"
    ).reset_index(drop=True)
    CONFIG["reporte_csv"].parent.mkdir(parents=True, exist_ok=True)
    orden.to_csv(CONFIG["reporte_csv"], index=False, encoding="utf-8")
    return orden


def escribir_slices(filas: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Escribe el CSV de rebanadas comparadas.

    Args:
        filas: filas de rebanadas de todos los modelos comparados.

    Returns:
        DataFrame escrito.
    """
    tabla = pd.DataFrame(filas)
    CONFIG["slices_csv"].parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(CONFIG["slices_csv"], index=False, encoding="utf-8")
    return tabla


def escribir_json(detalle: dict[str, Any]) -> Path:
    """Escribe el JSON de resultados de las Fases 6-8.

    Args:
        detalle: diccionario completo del reporte.

    Returns:
        Ruta del archivo escrito.
    """
    CONFIG["reporte_json"].write_text(
        json.dumps(detalle, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return CONFIG["reporte_json"]


def construir_comparacion_rebanadas(
    tabla: pd.DataFrame, mejor_identificador: str
) -> dict[str, Any]:
    """Compara las rebanadas del mejor candidato contra el control MOD-03b.

    Args:
        tabla: tabla de rebanadas (una fila por modelo y rebanada).
        mejor_identificador: id del mejor candidato de la fase.

    Returns:
        Diccionario con el delta por rebanada y el numero de rebanadas mejoradas.
    """
    referencia = tabla[tabla["modelo"] == "MOD-03b_REF"].set_index(
        ["rebanada", "grupo"]
    )["f1_media_columnas"]
    candidato = tabla[tabla["modelo"] == mejor_identificador].set_index(
        ["rebanada", "grupo"]
    )["f1_media_columnas"]
    deltas = {
        f"{rebanada}|{grupo}": round(float(candidato.loc[(rebanada, grupo)] - valor), 4)
        for (rebanada, grupo), valor in referencia.items()
        if (rebanada, grupo) in candidato.index
    }
    return {
        "mejor_candidato": mejor_identificador,
        "referencia": "MOD-03b_REF",
        "deltas": deltas,
        "rebanadas_mejoradas": int(sum(delta > 0 for delta in deltas.values())),
        "rebanadas_empeoradas": int(sum(delta < 0 for delta in deltas.values())),
    }


def construir_conclusion(
    filas: Sequence[dict[str, Any]], mejor_identificador: str, comparacion: dict[str, Any]
) -> dict[str, Any]:
    """Redacta la conclusion con la regla de decision del proyecto.

    Args:
        filas: filas del reporte ordenadas por F1 descendente.
        mejor_identificador: id del mejor candidato.
        comparacion: comparacion de rebanadas del mejor candidato.

    Returns:
        Diccionario con la decision, la evidencia y el texto de conclusion.
    """
    mejor = filas[0]
    if mejor_identificador == "MOD-03b_REF":
        decision = "ninguna configuracion de la fase supera el control MOD-03b_BEST"
    else:
        decision = mejor["decision"]
    texto = (
        f"Mejor configuracion de las Fases 6-8: {mejor['experiment_id']} con "
        f"f1_media_columnas {mejor['f1_media_columnas']:.4f} "
        f"(delta {mejor['mejora_vs_MOD03b']:+.4f} frente a MOD-03b_BEST 0.7927, "
        f"{mejor['mejora_en_pliegues']}/5 pliegues): {decision}. "
        f"Rebanadas mejoradas del mejor candidato frente al control: "
        f"{comparacion['rebanadas_mejoradas']} de {len(comparacion['deltas'])}."
    )
    return {
        "mejor_configuracion": mejor["experiment_id"],
        "f1_mejor": mejor["f1_media_columnas"],
        "delta_vs_MOD03b": mejor["mejora_vs_MOD03b"],
        "pliegues_mejorados": mejor["mejora_en_pliegues"],
        "decision": decision,
        "rebanadas_mejoradas": comparacion["rebanadas_mejoradas"],
        "texto": texto,
    }


def construir_seccion_contexto(
    filas: Sequence[dict[str, Any]], detalle: dict[str, Any]
) -> str:
    """Redacta la seccion de handoff (maximo 20 lineas) para ``contexto.md``.

    Args:
        filas: filas del reporte ordenadas por F1 descendente.
        detalle: diccionario completo del reporte (con la verificacion y la
            comparacion de rebanadas).

    Returns:
        Texto de la seccion, con encabezado y sin linea final en blanco.

    Raises:
        RuntimeError: si la seccion generada excede las 20 lineas permitidas.
    """
    mejor = filas[0]
    verificacion = detalle["verificacion_referencia"]
    comparacion = detalle["comparacion_rebanadas"]
    linea_bloques = "; ".join(
        f"{nombre} {info['n_features']} features" for nombre, info in detalle["bloques"].items()
    )
    deltas_relevantes = {
        clave: valor
        for clave, valor in comparacion["deltas"].items()
        if valor != 0.0
    }
    texto_deltas = (
        f"; deltas: {json.dumps(deltas_relevantes, ensure_ascii=False)}"
        if deltas_relevantes
        else ""
    )
    lineas = [
        f"## Fase 6-8 - ml-coder ({CONFIG['momento_contexto']})",
        "",
        (
            "Features densas para H11 (registro bromista), H12 (objetivo individual frente "
            "a colectivo) y H14 (deshumanizacion) en "
            f"`{CONFIG['script']}`, con el split de D-004 "
            "(StratifiedGroupKFold(5, shuffle=True, random_state=42), grupos de "
            "near-duplicado, estratificado por combo HSxTRxAG)."
        ),
        (
            f"Bloques densos: {linea_bloques}; escalado con StandardScaler ajustado por "
            "pliegue. Los lexicos inducidos (log-odds AG, HS y referencia colectiva) y la "
            "lista de palabras frecuentes del detector de nombres propios se inducen DENTRO "
            "de cada pliegue de entrenamiento (hallazgo de la revision de la Fase C); el "
            "resto de features son deterministas (regex y listas cerradas)."
        ),
        (
            f"Control MOD-03b_REF recalculado: {verificacion['f1_recalculada']:.6f} frente a "
            f"{verificacion['f1_publicada_npz']:.6f} de `features/oof_mod03b.npz` "
            f"(coincidencia binaria {verificacion['coincidencia_binaria']:.4f}); "
            f"verificacion {'OK' if verificacion['ok'] else 'FALLIDA'}."
        ),
        (
            "Resultados (f1_media_columnas OOF, delta vs MOD-03b 0.7927, pliegues mejorados): "
            + "; ".join(
                f"{fila['experiment_id']} {fila['f1_media_columnas']:.4f} "
                f"({fila['mejora_vs_MOD03b']:+.4f}, {fila['mejora_en_pliegues']}/5)"
                for fila in filas
            )
            + "."
        ),
        (
            f"Mejor configuracion: {mejor['experiment_id']} ({mejor['f1_media_columnas']:.4f}, "
            f"{mejor['decision']}); mejor candidato de la fase "
            f"{detalle['conclusion']['mejor_configuracion']} con "
            f"{detalle['conclusion']['delta_vs_MOD03b']:+.4f}."
        ),
        (
            "Evidencia anti-sobreajuste (F1 del pliegue de entrenamiento frente a OOF): "
            + "; ".join(
                f"{fila['experiment_id']} {fila['f1_train_medio']:.4f}/{fila['f1_media_columnas']:.4f} "
                f"(brecha {fila['brecha_train_oof']:+.4f})"
                for fila in filas
            )
            + "."
        ),
        (
            "Control de pipeline: HIB-UNION con las "
            f"{sum(info['n_features'] for info in detalle['bloques'].values())} columnas densas "
            f"puestas a cero da "
            f"{detalle['controles']['HIB-UNION_denso_cero']['f1_media_columnas']:.6f} "
            f"({detalle['controles']['HIB-UNION_denso_cero']['diferencia_vs_MOD03b']:+.6f} "
            "vs MOD-03b), por lo que la perdida del hibrido es atribuible a las features y "
            "no a la concatenacion."
        ),
        (
            f"Rebanadas debiles de la Fase 12 (hashtag, pregunta, referencia colectiva, Q4 y "
            f"sin mencion) frente al control: {comparacion['rebanadas_mejoradas']} de "
            f"{len(comparacion['deltas'])} mejoran"
            + texto_deltas
            + "."
        ),
        (
            "Nota de lectura: las rebanadas usan umbral 0.5 (los valores del control difieren "
            "de la tabla de la Fase 12, que uso los umbrales por etiqueta 0.44/0.51/0.46) y "
            "sirven para comparar control y candidato con el mismo criterio."
        ),
        detalle["conclusion"]["texto"],
        (
            "Artefactos: `reports/features_densas_resultados.csv` y `.json`, "
            "`reports/eda/features_densas_slices.csv` y filas B5-DENSE, HIB-UNION, HIB-H11, "
            "HIB-H12 y HIB-H14 en `experiments.csv`."
        ),
        (
            "El test nunca se leyo; ninguna decision de la fase usa etiquetas de test. "
            "Ningun script previo se modifico."
        ),
    ]
    while lineas and lineas[-1] == "":
        lineas.pop()
    if len(lineas) > 20:
        raise RuntimeError(f"La seccion de contexto excede 20 lineas ({len(lineas)})")
    return "\n".join(lineas)


def actualizar_contexto(seccion: str) -> Path:
    """Inserta la seccion de handoff al final de ``reports/contexto.md``.

    Idempotente: si la seccion de esta fase ya existe, se reemplaza en lugar de
    duplicarse.

    Args:
        seccion: texto de la seccion (sin linea final en blanco).

    Returns:
        Ruta del archivo actualizado.
    """
    destino: Path = CONFIG["contexto_md"]
    encabezado = f"## Fase 6-8 - ml-coder ({CONFIG['momento_contexto']})"
    texto = destino.read_text(encoding="utf-8") if destino.exists() else ""
    if encabezado in texto:
        texto = texto[: texto.index(encabezado)]
    destino.write_text(
        texto.rstrip("\n") + "\n\n" + seccion.rstrip("\n") + "\n", encoding="utf-8"
    )
    return destino


# ===== SECCION 11: flujo principal =====


def preparar_densos(
    anotaciones: pd.DataFrame,
    etiquetas: np.ndarray,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[pd.DataFrame, dict[str, Any]]]:
    """Construye la matriz densa de cada pliegue (una vez para todas las configs).

    Args:
        anotaciones: rasgos anotados de todos los documentos.
        etiquetas: matriz ``(n, 3)`` de etiquetas.
        pliegues: particion agrupada compartida.

    Returns:
        Lista de tuplas ``(matriz_densa, diagnostico_lexicos)`` por pliegue.
    """
    return [
        construir_denso_pliegue(anotaciones, etiquetas, entrenamiento)
        for entrenamiento, _ in pliegues
    ]


def ejecutar_control_denso_cero(
    configuracion: Configuracion,
    textos: pd.Series,
    etiquetas: np.ndarray,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
    densos: Sequence[tuple[pd.DataFrame, dict[str, Any]]],
) -> dict[str, Any]:
    """Control de pipeline: repite el hibrido con el bloque denso puesto a cero.

    Aisla el efecto de las features del efecto de la ruta de codigo
    (``StandardScaler`` mas ``sparse_hstack``). Si con el bloque en ceros la F1
    reproduce el control, cualquier diferencia del hibrido es atribuible a las
    features y no a un error de concatenacion o de alineacion de filas.

    Args:
        configuracion: configuracion hibrida de referencia para el control.
        textos: serie de textos crudos.
        etiquetas: matriz ``(n, 3)`` de etiquetas.
        pliegues: particion agrupada compartida.
        densos: matriz densa y diagnostico por pliegue.

    Returns:
        Diccionario con la F1 del control y su diferencia frente a MOD-03b.
    """
    densos_cero = [
        (pd.DataFrame(0.0, index=denso.index, columns=denso.columns), diagnostico)
        for denso, diagnostico in densos
    ]
    resultado = evaluar_configuracion(
        configuracion, textos, etiquetas, pliegues, densos_cero
    )
    f1_control = float(resultado.metricas_oof["f1_media_columnas"])
    return {
        "descripcion": (
            f"{configuracion.identificador} con las {resultado.n_features_densas} columnas "
            "densas puestas a cero: misma ruta de codigo (StandardScaler + sparse_hstack), "
            "cero senal en el bloque"
        ),
        "f1_media_columnas": round(f1_control, 6),
        "diferencia_vs_MOD03b": round(f1_control - float(CONFIG["referencia_f1"]), 6),
        "conclusion": (
            "el pipeline reproduce el control con el bloque en ceros, por lo que las "
            "diferencias del hibrido son atribuibles a las features densas"
            if abs(f1_control - float(CONFIG["referencia_f1"])) < 1e-6
            else "el pipeline NO reproduce el control con el bloque en ceros: revisar"
        ),
    }


def ejecutar_principal() -> None:
    """Ejecuta el pipeline completo de las Fases 6-8.

    Raises:
        RuntimeError: si la verificacion de la particion contra las probabilidades
            publicadas falla, porque las comparaciones dejarian de ser validas.
    """
    train, grupos = cargar_datos()
    etiquetas = train[list(ETIQUETAS)].to_numpy()
    textos = pd.Series(train["text"], index=train.index, name="text")
    log(f"train: {train.shape} | prevalencias {train[list(ETIQUETAS)].mean().round(4).to_dict()}")
    pliegues = construir_pliegues(train, grupos)

    log(f"anotando {len(textos)} documentos con spaCy {CONFIG['spacy_model']} (sin vectores)")
    anotaciones = anotar_documentos(cargar_anotador(), textos.tolist())
    log(f"anotacion lista: {anotaciones.shape[1]} columnas de rasgos")

    densos = preparar_densos(anotaciones, etiquetas, pliegues)
    log(
        "lexicos inducidos por pliegue: "
        + "; ".join(
            f"pliegue {indice}: AG {diag['n_agresivo']} (nucleo {diag['n_agresivo_nucleo']}), "
            f"HS {diag['n_hs']}, colectivo {diag['n_colectivo']}"
            for indice, (_, diag) in enumerate(densos, start=1)
        )
    )

    resultados = [
        evaluar_configuracion(configuracion, textos, etiquetas, pliegues, densos)
        for configuracion in CONFIGURACIONES
    ]
    referencia = next(
        resultado for resultado in resultados if resultado.configuracion.familia == "referencia"
    )
    verificacion = verificar_referencia(referencia.prediccion, etiquetas, train)
    if not verificacion["ok"]:
        raise RuntimeError(
            "La particion recalculada no reproduce MOD-03b_BEST; las comparaciones "
            f"no serian validas: {verificacion}"
        )
    log(f"verificacion del control: {verificacion}")

    filas_con_referencia = [
        construir_fila_reporte(resultado, referencia.metricas_pliegues)
        for resultado in resultados
    ]
    control = ejecutar_control_denso_cero(
        next(
            config
            for config in CONFIGURACIONES
            if config.identificador == "HIB-UNION"
        ),
        textos,
        etiquetas,
        pliegues,
        densos,
    )
    log(f"control denso en ceros: {control['f1_media_columnas']} ({control['conclusion']})")
    candidatas = [
        fila for fila in filas_con_referencia if fila["experiment_id"] != "MOD-03b_REF"
    ]
    orden = sorted(candidatas, key=lambda fila: fila["f1_media_columnas"], reverse=True)
    reporte = escribir_reporte(filas_con_referencia)
    for _, fila in reporte.iterrows():
        log(
            f"{fila['experiment_id']:12s} f1={fila['f1_media_columnas']:.4f} "
            f"delta={fila['mejora_vs_MOD03b']:+.4f} pliegues={fila['mejora_en_pliegues']}/5 "
            f"dim={fila['dimension']} {fila['tiempo_segundos']:.2f}s {fila['decision']}"
        )

    mascaras = construir_mascaras_rebanadas(anotaciones)
    predicciones = {
        resultado.configuracion.identificador: resultado.prediccion for resultado in resultados
    }
    mejor_identificador = orden[0]["experiment_id"]
    modelos_comparados = ["MOD-03b_REF", mejor_identificador]
    filas_slices = [
        fila
        for modelo in dict.fromkeys(modelos_comparados)
        for fila in evaluar_rebanadas(modelo, predicciones[modelo], etiquetas, mascaras)
    ]
    tabla_slices = escribir_slices(filas_slices)
    comparacion = construir_comparacion_rebanadas(tabla_slices, mejor_identificador)

    detalle: dict[str, Any] = {
        "metadata": {
            "fecha": CONFIG["fecha"],
            "script": CONFIG["script"],
            "competencia": "KAGGLE_TOXIC_2026_PUJ",
            "fase": "Fases 6-8 (features densas H11/H12/H14, baseline denso B5 e hibrido)",
            "metrica_primaria": "f1_media_columnas",
            "referencia": CONFIG["referencia_id"],
            "referencia_f1": round(float(CONFIG["referencia_f1"]), 6),
            "regla_decision": (
                f"adoptada si mejora >= +{CONFIG['umbral_adopcion']} absoluto en "
                f"f1_media_columnas y mejora en >= {CONFIG['pliegues_minimos']} de 5 "
                "pliegues; la familia union exige ademas "
                f">= +{CONFIG['umbral_union_realista']} local (D-012)"
            ),
            "validacion": (
                f"StratifiedGroupKFold({CONFIG['n_splits']}, shuffle=True, "
                f"random_state={CONFIG['random_state']}), grupos = near-duplicados, "
                "estratificado por combinacion HSxTRxAG"
            ),
            "sin_fuga": (
                "TF-IDF y StandardScaler por pliegue; lexicos inducidos y palabras "
                "frecuentes inducidos solo con el pliegue de entrenamiento; features "
                "deterministas anotadas una vez; el test no se leyo"
            ),
            "idempotencia": (
                "Determinista salvo las columnas de tiempo (ruido de maquina, redondeadas "
                "a 2 decimales)"
            ),
        },
        "verificacion_referencia": verificacion,
        "controles": {"HIB-UNION_denso_cero": control},
        "bloques": {
            nombre: {"n_features": len(columnas), "columnas": list(columnas)}
            for nombre, columnas in (
                ("H11", COLUMNAS_H11),
                ("H12", COLUMNAS_H12),
                ("H14", COLUMNAS_H14),
            )
        },
        "lexicos_inducidos": [
            {"pliegue": indice, **diagnostico}
            for indice, (_, diagnostico) in enumerate(densos, start=1)
        ],
        "configuraciones": [
            {
                "experiment_id": fila["experiment_id"],
                "descripcion": fila["descripcion"],
                "familia": fila["familia"],
                "bloques": fila["bloques"],
                "usa_tfidf": bool(fila["usa_tfidf"]),
                "C": fila["C"],
                "dimension": fila["dimension"],
                "n_features_densas": fila["n_features_densas"],
                "tiempo_segundos": fila["tiempo_segundos"],
                "mejora_vs_MOD03b": fila["mejora_vs_MOD03b"],
                "mejora_en_pliegues": fila["mejora_en_pliegues"],
                "f1_train_medio": fila["f1_train_medio"],
                "brecha_train_oof": fila["brecha_train_oof"],
                "decision": fila["decision"],
                "metricas_oof": resultado.metricas_oof,
                "metricas_pliegues": resultado.metricas_pliegues,
                "metricas_entrenamiento": resultado.metricas_entrenamiento,
            }
            for fila, resultado in zip(
                filas_con_referencia, resultados, strict=True
            )
        ],
        "rebanadas": tabla_slices.to_dict(orient="records"),
        "comparacion_rebanadas": comparacion,
        "limitaciones": [
            (
                "Los bloques densos se concatenan con ~23.400 features TF-IDF sin re-ajustar "
                "C ni seleccionar features: la perdida del hibrido es compatible con "
                "sobreajuste por capacidad anadida (la brecha train-OOF crece respecto del "
                "control) y no demuestra que las features sean inutiles, solo que no aportan "
                "con este contrato."
            ),
            (
                "StandardScaler sobre banderas binarias raras produce valores estandarizados "
                "de magnitud alta; no se probo otra escala ni un escalado por bloque porque "
                "queda fuera del alcance pre-registrado de la fase."
            ),
            (
                "Las rebanadas usan predicciones OOF con umbral 0.5; los umbrales por etiqueta "
                "de D-012 no se re-optimizan aqui."
            ),
            (
                "Los rasgos de POS, lemas y morfologia dependen de spaCy es_core_news_md como "
                "anotador (D-003) sobre texto ASCII sin diacriticos: hay ruido de anotacion."
            ),
            (
                "Los lexicos inducidos dentro del pliegue son mas pequenos que el lexico "
                "in-sample del EDA (150 lemas por pliegue, unos 18 en el nucleo), lo que "
                "reduce su cobertura frente a la version in-sample."
            ),
        ],
        "siguiente_paso": (
            "Si el orquestador quiere insistir en la familia hibrida, el siguiente paso "
            "controlado es re-ajustar C y/o dar al bloque denso su propia regularizacion "
            "(escala reducida, seleccion de features o promedio de dos modelos), siempre con "
            "el mismo split y la misma regla de decision; con el contrato pre-registrado de "
            "esta fase la concatenacion directa no supera al single."
        ),
    }
    detalle["conclusion"] = construir_conclusion(orden, mejor_identificador, comparacion)
    escribir_json(detalle)

    filas_experiments = [
        construir_fila_experiments(fila, detalle["configuraciones"])
        for fila in orden
        if next(
            config.registrar
            for config in CONFIGURACIONES
            if config.identificador == fila["experiment_id"]
        )
    ]
    registradas = registrar_en_experiments(filas_experiments)
    log(f"experiments.csv: {registradas} filas registradas")

    seccion = construir_seccion_contexto(orden, detalle)
    actualizar_contexto(seccion)

    log("resumen final")
    print(f"\n{'configuracion':14s} {'f1_media':>9s} {'delta':>8s} {'pliegues':>9s} {'f1_train':>9s} "
          f"{'brecha':>7s} {'dim':>7s} {'f1_HS':>7s} {'f1_TR':>7s} {'f1_AG':>7s}  decision")
    for _, fila in reporte.iterrows():
        print(
            f"{fila['experiment_id']:14s} {fila['f1_media_columnas']:9.4f} "
            f"{fila['mejora_vs_MOD03b']:+8.4f} {fila['mejora_en_pliegues']:7d}/5 "
            f"{fila['f1_train_medio']:9.4f} {fila['brecha_train_oof']:7.4f} "
            f"{fila['dimension']:7d} {fila['f1_HS']:7.4f} {fila['f1_TR']:7.4f} "
            f"{fila['f1_AG']:7.4f}  {fila['decision']}"
        )
    print(
        f"\nControl (denso en ceros): {control['f1_media_columnas']:.6f} "
        f"({control['diferencia_vs_MOD03b']:+.6f} vs MOD-03b) -> {control['conclusion']}"
    )
    print(f"\nRebanadas debiles (MOD-03b_REF -> {mejor_identificador}):")
    print(
        f"{'rebanada':34s} {'grupo':>5s} {'n':>5s} {'control':>8s} {'candidato':>10s} {'delta':>7s}"
    )
    vista = tabla_slices.pivot_table(
        index=["rebanada", "grupo", "n"], columns="modelo", values="f1_media_columnas"
    ).reset_index()
    for _, fila in vista.iterrows():
        print(
            f"{fila['rebanada']:34s} {fila['grupo']:>5s} {int(fila['n']):5d} "
            f"{fila['MOD-03b_REF']:8.4f} {fila[mejor_identificador]:10.4f} "
            f"{fila[mejor_identificador] - fila['MOD-03b_REF']:+7.4f}"
        )
    print(f"\nConclusion: {detalle['conclusion']['texto']}")


def main() -> None:
    """Punto de entrada con manejo de errores y codigo de salida."""
    try:
        ejecutar_principal()
    except Exception:  # noqa: BLE001 - se reporta el traceback completo y se sale con 1
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
