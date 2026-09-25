"""experiments/16_targeting_v2.py

Fecha: 2026-09-25
Descripcion: P2 (E2-05/E2-06) del plan v2 - targeting estructural. Construye un
lexico de insulto inducido fold-local, un inventario de terminos colectivos, las
distancias target-insulto (token/caracter, orden, densidad) y un canal simbolico
abstracto (TARGET, INSULT, COLLECTIVE, SECOND_PERSON, NEG, QUOTE) para evaluar
cuatro variantes con el protocolo v2 sobre el DESARROLLO (holdout congelado
intacto).
Estado del arte aplicado: log-odds con prior informativo de Dirichlet (Monroe,
Colaresi y Quinn, 2008) para la induccion fold-local de lexicos; sustitucion de
papeles por simbolos (slots) como representacion sparse de la relacion
target-insulto; regla estructural de referencia colectiva del script 06
(determinante plural + nombre/adjetivo plural + gentilicios por sufijo).

Pregunta que responde
---------------------
La brecha de targeting (gap analysis v2, brechas #1, #2, #5 y #6; FN de AG con
referencia colectiva 45.5 % frente a 13.9 % de los TP) se puede representar con
features explicitas de la RELACION target-insulto y con un canal simbolico
abstracto que el modelo de texto no aislaba? Aporta sobre V03 (0.7772) o sobre el
candidato v2-A de P1 (meta-modelo 0.7853)?

Metodo (sin fuga)
-----------------
- Trabajo exclusivo sobre el desarrollo (n=1483); el holdout congelado
  (`data/processed/holdout_v2_ids.csv`) se usa solo para excluir filas.
- Pliegues corregidos v2 (D-016): `StratifiedGroupKFold(5)`, grupos = near
  duplicado UNION pseudo-autor, seed 42 (`src.splits.pliegues_corregidos`).
- Anotacion morfosintactica con spaCy (`es_core_news_md`, sin NER y sin
  vectores: POS, lema, morfologia y fronteras de oracion; D-003) una sola vez y
  cacheada en `features/v2_targeting_annotations.json.gz`. Si spaCy no estuviera
  disponible se cae a una heuristica de mayusculas documentada.
- Lexico de insulto, lista de adjetivos negativos e inventario colectivo se
  INDUCEN EN CADA PLIEGUE con las etiquetas y los textos del pliegue de
  entrenamiento (log-odds con prior informativo de Dirichlet; min 5 apariciones,
  presente en >= 3 documentos; umbral de log-odds 1.0 para el insulto, ~2.7x de
  odds; lemas normalizados a ASCII, sin PROPN, sin stopwords y restringidos a
  ADJ/NOUN/VERB/INTJ). Limitacion honesta: el lexico es informativo (asociado a
  las etiquetas), no una lista curada de profanidades, asi que incluye terminos
  del contexto de agresion (verbos frecuentes con HS/AG) ademas de los insultos.
- Targets fold-locales: mencion inicial, menciones internas, PROPN (POS de
  spaCy), segunda persona, imperativo y colectivos (patron estructural + sufijo
  de gentilicio, regla sin etiquetas).
- Variantes con LR balanced: CH-TGT (TF-IDF simbolico solo); CH-TGT+DIST
  (simbolico + numericas E2-05/E2-06 estandarizadas con StandardScaler ajustado
  SOLO con el pliegue de entrenamiento); TXT+CH (word+char de V03 + simbolico +
  numericas); TXT+TGT-SPARSE (word+char de V03 + n-gramas 1-2 simbolicos). La
  representacion de texto se importa de `experiments/14_oof_corregido_v2.py`
  (`construir_matrices`) para que sea exactamente la de V03.
- C en {1.0, 2.0} para todas las variantes; se reporta cada C y se elige el
  mejor por corrected_cv (seleccion interna sobre el mismo OOF: cota optimista
  documentada, misma convencion que P1).
- Referencias: BASE (M2 a 0.5, debe reproducir 0.7718) y V03 con umbrales
  0.44/0.51/0.46 (debe reproducir 0.7772). Se verifica ademas que las
  probabilidades de BASE coinciden con `features/v2_oof_base_corrected.npz`.
- Regla de adopcion (D-016): corrected_cv >= 0.7772 + 0.005 con >= 3/5 pliegues
  mejorados, o ensemble_gain >= 0.0015 con IC bootstrap pareado (2000
  remuestreos a nivel de tweet) estrictamente positivo.
- Diagnostico E2-06: recall de AG en la rebanada de referencia colectiva (regla
  de `experiments/06_error_analysis.py`) frente a V03.

Salidas
-------
`reports/targeting_v2.csv`, `reports/targeting_v2.json`,
`reports/figuras/fig_28_targeting_v2.png`, `features/v2_oof_targeting.npz`
(probabilidades OOF de CH-TGT+DIST y TXT+CH para el stacking de P6), filas
`E2-05-*`/`E2-06-*` en `experiments.csv`, seccion en `reports/contexto.md` y
actualizacion de `tasks/todo.md` (P2.1-P2.4 y Gate P2).

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/16_targeting_v2.py
"""

from __future__ import annotations

import gzip
import importlib.util
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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta
from src.splits import (
    RANDOM_STATE,
    grupos_fusionados,
    pliegues_corregidos,
)

IDS_PATH = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
OOF_BASE_PATH = PROJECT_ROOT / "features" / "v2_oof_base_corrected.npz"
OOF_TARGETING_PATH = PROJECT_ROOT / "features" / "v2_oof_targeting.npz"
CACHE_ANOTACIONES = PROJECT_ROOT / "features" / "v2_targeting_annotations.json.gz"
REPORTE_CSV = PROJECT_ROOT / "reports" / "targeting_v2.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "targeting_v2.json"
FIGURA_PATH = PROJECT_ROOT / "reports" / "figuras" / "fig_28_targeting_v2.png"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"
CONTEXTO_MD = PROJECT_ROOT / "reports" / "contexto.md"
TODO_MD = PROJECT_ROOT / "tasks" / "todo.md"

CONFIG: Final[dict[str, Any]] = {
    "fecha": "2026-09-25",
    "referencia_m2_umbral_05": 0.7718,
    "referencia_v03_umbrales": 0.7772,
    "referencia_v2a_meta": 0.7853,
    "umbrales_v03": {"HS": 0.44, "TR": 0.51, "AG": 0.46},
    "v03_por_etiqueta": {"HS": 0.762, "TR": 0.816, "AG": 0.754},
    "C_variantes": (1.0, 2.0),
    "min_frecuencia_lexico": 5,
    "min_documentos_lexico": 3,
    "min_documentos_colectivo": 2,
    "alpha_prior_log_odds": 50.0,
    "umbral_log_odds_insulto": 1.0,
    "umbral_log_odds_adjetivo": 1.0,
    "max_iter_lr": 1000,
    "n_bootstrap": 2000,
    "mejora_minima": 0.005,
    "ensemble_gain_minimo": 0.0015,
    "pliegues_minimos_mejorados": 3,
    "modelo_spacy": "es_core_news_md",
    "batch_spacy": 64,
    "tolerancia_replicacion_base": 1e-3,
    "tolerancia_replicacion_f1": 1e-4,
}

# Representacion de texto de V03 (identica a experiments/14, modelo M2).
PARAMS_WORD: Final[dict[str, Any]] = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
}
PARAMS_CHAR: Final[dict[str, Any]] = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "min_df": 2,
    "sublinear_tf": True,
}
# El vocabulario simbolico tiene 8 simbolos (<= 64 bigramas): min_df=1 es viable y
# evita perder patrones raros pero informativos (por ejemplo "QUOTE INSULT").
PARAMS_SIMBOLICO: Final[dict[str, Any]] = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "min_df": 1,
    "sublinear_tf": True,
    "lowercase": False,
    "token_pattern": r"(?u)\S+",
}
PARAMS_LR: Final[dict[str, Any]] = {
    "class_weight": "balanced",
    "max_iter": CONFIG["max_iter_lr"],
    "solver": "liblinear",
    "random_state": RANDOM_STATE,
}
MODELOS_BASE_ESPERADOS: Final[tuple[str, ...]] = (
    "M1_word_lr",
    "M2_union_lr_c2",
    "M3_union_svc",
    "M4_char_lr",
)

SIMBOLO_TARGET: Final[str] = "TARGET"
SIMBOLO_INSULTO: Final[str] = "INSULT"
SIMBOLO_COLECTIVO: Final[str] = "COLLECTIVE"
SIMBOLO_SEGUNDA: Final[str] = "SECOND_PERSON"
SIMBOLO_NEG: Final[str] = "NEG"
SIMBOLO_CITA: Final[str] = "QUOTE"
SIMBOLO_RESTO: Final[str] = "O"
SIMBOLO_PUNTUACION: Final[str] = "SEP"

POS_LEXICO: Final[tuple[str, ...]] = ("ADJ", "NOUN", "VERB", "INTJ")
POS_COLECTIVO: Final[tuple[str, ...]] = ("NOUN", "ADJ", "PROPN")
SUFIJOS_GENTILICIO: Final[tuple[str, ...]] = ("ano", "ana", "ense", "eno", "ena", "i")
# Falsos amigos del sufijo -ano/-ana documentados (nombres comunes que no son
# gentilicios ni colectivos).
GENTILICIO_EXCLUIDOS: Final[frozenset[str]] = frozenset(
    {"hermano", "hermana", "temprano", "temprana", "verano", "semana", "ventana", "manzano"}
)
NEGACION: Final[frozenset[str]] = frozenset(
    {"no", "ni", "nunca", "jamas", "tampoco", "sin", "nada", "nadie"}
)
CUANTIFICADORES_UNIVERSALES: Final[frozenset[str]] = frozenset(
    {
        "todo",
        "todos",
        "toda",
        "todas",
        "cualquier",
        "cualquiera",
        "cualesquiera",
        "ningun",
        "ninguno",
        "ninguna",
        "ningunos",
        "ningunas",
        "nadie",
        "nada",
        "siempre",
        "jamas",
        "totalidad",
    }
)
# Stopwords de respaldo si spaCy no estuviera disponible; con spaCy se usa su
# lista de stopwords del espanol (documentado).
STOPWORDS_RESPALDO: Final[frozenset[str]] = frozenset(
    {
        "de",
        "la",
        "el",
        "los",
        "las",
        "un",
        "una",
        "unos",
        "unas",
        "y",
        "o",
        "que",
        "en",
        "a",
        "al",
        "del",
        "por",
        "para",
        "con",
        "sin",
        "se",
        "su",
        "sus",
        "es",
        "son",
        "ser",
        "esta",
        "este",
        "esto",
        "estos",
        "estas",
        "como",
        "mas",
        "pero",
        "porque",
        "cuando",
        "donde",
        "muy",
        "ya",
        "tambien",
        "solo",
        "hay",
        "lo",
        "le",
        "les",
    }
)

NOMBRES_E2_05: Final[tuple[str, ...]] = (
    "ti_tiene_par",
    "ti_n_pares",
    "ti_n_targets",
    "ti_n_insultos",
    "ti_n_mencion_inicial",
    "ti_n_mencion_interna",
    "ti_n_propn",
    "ti_densidad",
    "ti_dist_tok_min",
    "ti_dist_tok_media",
    "ti_dist_tok_mediana",
    "ti_dist_tok_max",
    "ti_dist_car_min",
    "ti_dist_car_media",
    "ti_dist_car_mediana",
    "ti_dist_car_max",
    "ti_orden_media",
    "ti_frac_antes",
    "ti_frac_despues",
    "ti_par_misma_oracion",
    "ti_par_le3",
    "ti_par_le5",
    "ti_par_adyacente",
)
NOMBRES_E2_06: Final[tuple[str, ...]] = (
    "col_presente",
    "col_conteo",
    "col_intensidad",
    "col_densidad",
    "col_insulto",
    "col_insulto_conteo",
    "col_insulto_le3",
    "col_adj_neg",
    "col_adj_neg_conteo",
    "cuant_universal",
    "plural_densidad",
    "plural_generico",
    "col_segunda_persona",
)
NOMBRES_NUMERICOS: Final[tuple[str, ...]] = NOMBRES_E2_05 + NOMBRES_E2_06
NOMBRES_VARIANTES: Final[tuple[str, ...]] = (
    "CH-TGT",
    "CH-TGT+DIST",
    "TXT+CH",
    "TXT+TGT-SPARSE",
)
VARIANTES_TEXTO: Final[frozenset[str]] = frozenset({"TXT+CH", "TXT+TGT-SPARSE"})
PATRON_MENCION: Final[re.Pattern[str]] = re.compile(r"@\w+")
PATRON_MAYUSCULA_INICIAL: Final[re.Pattern[str]] = re.compile(r"^[A-Z][a-z]{2,}$")


def log(mensaje: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        mensaje: texto a imprimir.
    """
    print(f"[P2] {mensaje}", flush=True)


# ===== SECCION 1: carga de datos y pliegues del desarrollo =====


def cargar_desarrollo() -> tuple[pd.DataFrame, NDArray[np.int_], list[tuple[NDArray[np.int_], NDArray[np.int_]]]]:
    """Carga el desarrollo y sus pliegues corregidos v2.

    Returns:
        Tupla ``(desarrollo, grupos, pliegues)``. El holdout congelado queda
        excluido por construccion (solo se lee su lista de ids).
    """
    train = data_module.read_train()
    particion = pd.read_csv(IDS_PATH)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)

    grupos_completos = grupos_fusionados(train)
    mapa_grupo = dict(zip(train["id"].astype(int), grupos_completos.tolist()))
    grupos = np.array([mapa_grupo[identificador] for identificador in desarrollo["id"].astype(int)])
    pliegues = pliegues_corregidos(desarrollo, grupos)
    return desarrollo, grupos, pliegues


def verificar_datos(desarrollo: pd.DataFrame, grupos: NDArray[np.int_]) -> dict:
    """Comprueba integridad basica y resume el desarrollo.

    Args:
        desarrollo: DataFrame del conjunto de desarrollo.
        grupos: grupos fusionados alineados con ``desarrollo``.

    Returns:
        Diccionario con el resumen de verificacion.

    Raises:
        RuntimeError: si hay nulos, folds vacios o grupos compartidos.
    """
    nulos = int(desarrollo[list(ETIQUETAS) + ["text"]].isna().sum().sum())
    resumen = {
        "n_filas": len(desarrollo),
        "n_grupos": int(np.unique(grupos).size),
        "nulos": nulos,
        "prevalencia": {etiqueta: float(desarrollo[etiqueta].mean()) for etiqueta in ETIQUETAS},
    }
    if nulos:
        raise RuntimeError(f"El desarrollo tiene {nulos} nulos")
    log(
        f"desarrollo n={resumen['n_filas']} grupos={resumen['n_grupos']} nulos={nulos} "
        f"prevalencia={ {clave: round(valor, 3) for clave, valor in resumen['prevalencia'].items()} }"
    )
    return resumen


# ===== SECCION 2: anotacion morfosintactica (spaCy cacheada) =====


@dataclass(frozen=True)
class TokenAnotado:
    """Token con la informacion morfosintactica usada por el canal de targeting.

    Attributes:
        texto: forma original.
        lema: lema en minusculas y sin diacriticos.
        pos: categoria POS de spaCy (o ``WORD`` en el modo heuristico).
        numero: valor de la categoria morfologica ``Number`` (``Plur``, ``Sing`` o ``""``).
        persona: valor de ``Person`` (``1``, ``2``, ``3`` o ``""``).
        modo: valor de ``Mood`` (``Imp``, etc., o ``""``).
        oracion: indice de oracion dentro del documento.
        inicio: desplazamiento en caracteres del token en el texto.
        en_comillas: si el token cae dentro de un tramo entre comillas dobles.
    """

    texto: str
    lema: str
    pos: str
    numero: str
    persona: str
    modo: str
    oracion: int
    inicio: int
    en_comillas: bool


@dataclass(frozen=True)
class DocumentoAnotado:
    """Documento del desarrollo con sus tokens anotados.

    Attributes:
        tokens: lista de tokens en orden de aparicion.
        n_caracteres: longitud del texto original.
    """

    tokens: tuple[TokenAnotado, ...]
    n_caracteres: int


@dataclass(frozen=True)
class LexicosPliegue:
    """Lexicos inducidos SOLO con el pliegue de entrenamiento.

    Attributes:
        insulto: lemas (ASCII, minusculas) con evidencia de insulto en alguna etiqueta.
        adjetivos_negativos: lemas de adjetivos con log-odds positivo de agresion o HS.
        colectivos: inventario de nombres/adjetivos vistos como colectivo plural.
        tamanos: conteo de terminos por lista y por etiqueta.
    """

    insulto: frozenset[str]
    adjetivos_negativos: frozenset[str]
    colectivos: frozenset[str]
    tamanos: dict[str, int]


def cargar_constantes_error_analysis() -> tuple[dict[str, Any], str]:
    """Importa las constantes estructurales de ``experiments/06_error_analysis.py``.

    Reutilizarlas garantiza que la rebanada de referencia colectiva de E2-06 sea
    exactamente la del analisis de error (45.5 % de FN con colectiva). El script
    06 no se modifica: solo se leen sus constantes. Si la importacion falla se
    usa una copia local documentada.

    Returns:
        Tupla ``(constantes, origen)`` con las listas lexicas y el origen usado.
    """
    respaldo = {
        "segunda_persona": frozenset(
            {
                "contigo",
                "os",
                "te",
                "ti",
                "tu",
                "tus",
                "tuya",
                "tuyas",
                "tuyo",
                "tuyos",
                "usted",
                "ustedes",
                "vos",
                "vosotras",
                "vosotros",
                "vuestra",
                "vuestras",
                "vuestro",
                "vuestros",
            }
        ),
        "imperativos": frozenset(),
        "colectivos": frozenset(
            {
                "algunas",
                "algunos",
                "ambas",
                "ambos",
                "cada",
                "cualesquiera",
                "cualquiera",
                "gente",
                "grupo",
                "masa",
                "mayoria",
                "minoria",
                "monton",
                "muchas",
                "muchos",
                "multitud",
                "ningun",
                "ninguna",
                "pueblo",
                "toda",
                "todas",
                "todo",
                "todos",
                "varias",
                "varios",
            }
        ),
        "patron_mencion_inicial": r"^\s*(?:rt\s+)?@\w+",
    }
    ruta = PROJECT_ROOT / "experiments" / "06_error_analysis.py"
    try:
        especificacion = importlib.util.spec_from_file_location("experimento_06_constantes", ruta)
        if especificacion is None or especificacion.loader is None:
            raise ImportError(f"No se pudo cargar {ruta}")
        modulo = importlib.util.module_from_spec(especificacion)
        sys.modules[especificacion.name] = modulo
        especificacion.loader.exec_module(modulo)
        constantes = {
            "segunda_persona": frozenset(modulo.SECOND_PERSON_LEXICON),
            "imperativos": frozenset(modulo.IMPERATIVE_LEXICON),
            "colectivos": frozenset(modulo.COLLECTIVE_QUANTIFIERS),
            "patron_mencion_inicial": modulo.INITIAL_MENTION_PATTERN.pattern,
        }
        origen = "experiments/06_error_analysis.py"
    except Exception as error:  # noqa: BLE001 - respaldo defensivo si el script 06 no importa
        log(f"aviso: no se pudo importar el script 06 ({error}); se usa respaldo local")
        constantes = respaldo
        origen = "respaldo local"
    return constantes, origen


def normalizar_ascii(texto: str) -> str:
    """Normaliza un token a minusculas ASCII sin diacriticos.

    Args:
        texto: token de entrada.

    Returns:
        Texto normalizado.
    """
    return data_module.strip_accents(texto).lower().strip()


def _tramos_comillas(texto: str) -> list[tuple[int, int]]:
    """Localiza los tramos entre comillas dobles de un texto.

    Args:
        texto: texto del documento.

    Returns:
        Lista de ``(inicio, fin)`` por cada par de comillas dobles cerrado.
    """
    posiciones = [indice for indice, caracter in enumerate(texto) if caracter == '"']
    tramos = []
    for indice in range(0, len(posiciones) - 1, 2):
        tramos.append((posiciones[indice], posiciones[indice + 1]))
    return tramos


def anotar_con_spacy(textos: Sequence[str]) -> list[DocumentoAnotado]:
    """Anota los documentos con spaCy (POS, lema, morfologia, oraciones).

    spaCy se usa solo como anotador morfosintactico sin NER ni vectores (D-003).

    Args:
        textos: textos del desarrollo.

    Returns:
        Lista de documentos anotados.
    """
    import spacy

    anotador = spacy.load(CONFIG["modelo_spacy"], exclude=["ner"])
    documentos = []
    for documento in anotador.pipe(list(textos), batch_size=CONFIG["batch_spacy"]):
        tramos = _tramos_comillas(documento.text)
        tokens = []
        for token in documento:
            en_comillas = any(inicio < token.idx < fin for inicio, fin in tramos)
            tokens.append(
                TokenAnotado(
                    texto=token.text,
                    lema=normalizar_ascii(token.lemma_),
                    pos=token.pos_ or "X",
                    numero=(token.morph.get("Number") or [""])[0],
                    persona=(token.morph.get("Person") or [""])[0],
                    modo=(token.morph.get("Mood") or [""])[0],
                    oracion=token.sent.start,
                    inicio=int(token.idx),
                    en_comillas=en_comillas,
                )
            )
        documentos.append(DocumentoAnotado(tokens=tuple(tokens), n_caracteres=len(documento.text)))
    return documentos


def anotar_heuristico(textos: Sequence[str]) -> list[DocumentoAnotado]:
    """Anotacion de respaldo sin spaCy (solo mayusculas y lexicos).

    Args:
        textos: textos del desarrollo.

    Returns:
        Lista de documentos anotados con POS ``PROPN``/``WORD`` y sin morfologia.
    """
    patron = re.compile(r"@\w+|#\w+|[A-Za-z][A-Za-z0-9']*|\d+|[^\sA-Za-z0-9]")
    documentos = []
    for texto in textos:
        tramos = _tramos_comillas(texto)
        tokens = []
        for coincidencia in patron.finditer(texto):
            forma = coincidencia.group(0)
            lema = normalizar_ascii(forma)
            if lema in NEGACION or lema in CUANTIFICADORES_UNIVERSALES:
                pos = "ADV" if lema in NEGACION else "PRON"
            elif PATRON_MAYUSCULA_INICIAL.match(forma) and coincidencia.start() > 0:
                pos = "PROPN"
            else:
                pos = "WORD"
            tokens.append(
                TokenAnotado(
                    texto=forma,
                    lema=lema,
                    pos=pos,
                    numero="",
                    persona="",
                    modo="",
                    oracion=0,
                    inicio=int(coincidencia.start()),
                    en_comillas=any(
                        inicio < coincidencia.start() < fin for inicio, fin in tramos
                    ),
                )
            )
        documentos.append(DocumentoAnotado(tokens=tuple(tokens), n_caracteres=len(texto)))
    return documentos


def _serializar(documentos: Sequence[DocumentoAnotado]) -> list[list[dict[str, Any]]]:
    """Serializa los documentos anotados a estructuras JSON.

    Args:
        documentos: documentos anotados.

    Returns:
        Estructura serializable.
    """
    return [
        [
            {
                "t": token.texto,
                "l": token.lema,
                "p": token.pos,
                "n": token.numero,
                "pe": token.persona,
                "m": token.modo,
                "o": token.oracion,
                "i": token.inicio,
                "c": int(token.en_comillas),
            }
            for token in documento.tokens
        ]
        # El numero de caracteres no se cachea: se recalcula con el texto.
        for documento in documentos
    ]


def _deserializar(carga: list[list[dict[str, Any]]], textos: Sequence[str]) -> list[DocumentoAnotado]:
    """Reconstruye los documentos anotados desde la cache.

    Args:
        carga: estructura leida de la cache.
        textos: textos originales (para la longitud).

    Returns:
        Lista de documentos anotados.
    """
    documentos = []
    for fila, texto in zip(carga, textos, strict=True):
        tokens = tuple(
            TokenAnotado(
                texto=registro["t"],
                lema=registro["l"],
                pos=registro["p"],
                numero=registro["n"],
                persona=registro["pe"],
                modo=registro["m"],
                oracion=int(registro["o"]),
                inicio=int(registro["i"]),
                en_comillas=bool(registro["c"]),
            )
            for registro in fila
        )
        documentos.append(DocumentoAnotado(tokens=tokens, n_caracteres=len(texto)))
    return documentos


def cargar_o_crear_anotaciones(desarrollo: pd.DataFrame) -> tuple[list[DocumentoAnotado], str]:
    """Devuelve la anotacion del desarrollo, reutilizando la cache si existe.

    Args:
        desarrollo: DataFrame del desarrollo.

    Returns:
        Tupla ``(documentos, modo)`` con el modo de anotacion usado
        (``spacy-cache``, ``spacy`` o ``heuristico``).
    """
    ids = desarrollo["id"].astype(int).tolist()
    textos = desarrollo["text"].tolist()
    if CACHE_ANOTACIONES.exists():
        with gzip.open(CACHE_ANOTACIONES, "rt", encoding="utf-8") as archivo:
            carga = json.load(archivo)
        if carga.get("ids") == ids and carga.get("modo") == "spacy":
            documentos = _deserializar(carga["documentos"], textos)
            log(f"anotacion spaCy reutilizada de la cache ({len(documentos)} documentos)")
            return documentos, "spacy-cache"
        log("la cache de anotaciones no coincide con el desarrollo: se reanota")
    try:
        documentos = anotar_con_spacy(textos)
        modo = "spacy"
    except Exception as error:  # noqa: BLE001 - respaldo defensivo si spaCy no esta disponible
        log(f"aviso: spaCy no disponible ({error}); se usa la anotacion heuristica")
        documentos = anotar_heuristico(textos)
        modo = "heuristico"
    if modo == "spacy":
        CACHE_ANOTACIONES.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(CACHE_ANOTACIONES, "wt", encoding="utf-8") as archivo:
            json.dump({"ids": ids, "modo": modo, "documentos": _serializar(documentos)}, archivo)
    log(f"anotacion completada: modo={modo} documentos={len(documentos)}")
    return documentos, modo


def es_mencion(token: TokenAnotado) -> bool:
    """Indica si el token es una mencion social.

    Args:
        token: token anotado.

    Returns:
        ``True`` si el token empieza por ``@``.
    """
    return token.texto.startswith("@")


def es_colectivo_estructural(
    tokens: Sequence[TokenAnotado],
    indice: int,
    constantes: dict[str, Any],
) -> bool:
    """Aplica la regla estructural de referencia colectiva (script 06).

    La regla es: lema cuantificador colectivo (NOUN/PROPN) o nombre/adjetivo en
    plural precedido por un determinante en plural. No depende de etiquetas.

    Args:
        tokens: tokens del documento.
        indice: posicion del token a evaluar.
        constantes: constantes estructurales importadas del script 06.

    Returns:
        ``True`` si el token es una referencia colectiva.
    """
    token = tokens[indice]
    if token.pos in ("NOUN", "PROPN") and token.lema in constantes["colectivos"]:
        return True
    return bool(
        token.pos in POS_COLECTIVO
        and token.numero == "Plur"
        and indice > 0
        and tokens[indice - 1].pos == "DET"
        and tokens[indice - 1].numero == "Plur"
    )


def rebanada_colectiva_referencia(
    documentos: Sequence[DocumentoAnotado], constantes: dict[str, Any]
) -> NDArray[np.bool_]:
    """Calcula la rebanada binaria de referencia colectiva del script 06.

    Args:
        documentos: documentos anotados del desarrollo.
        constantes: constantes estructurales importadas del script 06.

    Returns:
        Arreglo booleano alineado con ``documentos``.
    """
    mascara = np.zeros(len(documentos), dtype=bool)
    for posicion, documento in enumerate(documentos):
        for indice in range(len(documento.tokens)):
            if es_colectivo_estructural(documento.tokens, indice, constantes):
                mascara[posicion] = True
                break
    return mascara


# ===== SECCION 3: induccion fold-local de lexicos e inventarios =====


def log_odds_informativo(
    documentos_positivos: Counter,
    documentos_negativos: Counter,
    n_positivos: int,
    n_negativos: int,
    frecuencia: Counter,
    total_frecuencia: int,
) -> dict[str, float]:
    """Calcula log-odds con prior informativo de Dirichlet (Monroe et al., 2008).

    Args:
        documentos_positivos: documentos por lema con la etiqueta activa.
        documentos_negativos: documentos por lema con la etiqueta inactiva.
        n_positivos: numero de documentos con la etiqueta activa.
        n_negativos: numero de documentos con la etiqueta inactiva.
        frecuencia: apariciones totales por lema en el pliegue.
        total_frecuencia: apariciones totales del pliegue.
        Prior: ``alpha0 * frecuencia[w] / total_frecuencia`` con
        ``alpha0 = CONFIG['alpha_prior_log_odds']``.

    Returns:
        Diccionario ``lema -> log-odds``; positivo si el lema es mas frecuente
        en los documentos con la etiqueta activa.
    """
    vocabulario = sorted(set(documentos_positivos) | set(documentos_negativos))
    alpha0 = CONFIG["alpha_prior_log_odds"]
    prior = {lema: alpha0 * frecuencia[lema] / max(total_frecuencia, 1) for lema in vocabulario}
    total_prior = sum(prior.values())
    scores = {}
    for lema in vocabulario:
        valor = prior[lema]
        p_positivo = (documentos_positivos.get(lema, 0) + valor) / (
            n_positivos + total_prior - documentos_positivos.get(lema, 0) - valor
        )
        p_negativo = (documentos_negativos.get(lema, 0) + valor) / (
            n_negativos + total_prior - documentos_negativos.get(lema, 0) - valor
        )
        scores[lema] = float(np.log(max(p_positivo, 1e-12)) - np.log(max(p_negativo, 1e-12)))
    return scores


def _stopwords() -> frozenset[str]:
    """Devuelve las stopwords del espanol de spaCy (o el respaldo local).

    Returns:
        Conjunto de stopwords.
    """
    try:
        from spacy.lang.es.stop_words import STOP_WORDS

        return frozenset(STOP_WORDS)
    except Exception:  # noqa: BLE001 - respaldo defensivo si spaCy no esta disponible
        return STOPWORDS_RESPALDO


def inducir_lexicos(
    documentos: Sequence[DocumentoAnotado],
    etiquetas_pliegue: NDArray[np.int_],
) -> LexicosPliegue:
    """Induce los lexicos del pliegue de entrenamiento (sin ver validacion).

    El lexico de insulto se restringe a lemas de contenido (ADJ, NOUN, VERB,
    INTJ), sin stopwords y sin PROPN: los nombres propios entran por el canal de
    target, no por el de insulto. El inventario colectivo se mina del mismo
    pliegue.

    Args:
        documentos: documentos anotados SOLO del pliegue de entrenamiento.
        etiquetas_pliegue: matriz ``(n, 3)`` de etiquetas del pliegue.

    Returns:
        Lexicos inducidos con sus tamanos.
    """
    stopwords = _stopwords()
    df_lema: Counter = Counter()
    frecuencia: Counter = Counter()
    adjetivos_vistos: set[str] = set()
    df_positivo: list[Counter] = [Counter() for _ in ETIQUETAS]
    df_negativo: list[Counter] = [Counter() for _ in ETIQUETAS]
    for posicion, documento in enumerate(documentos):
        vistos: set[str] = set()
        for token in documento.tokens:
            if token.pos not in POS_LEXICO or token.lema in stopwords or len(token.lema) < 3:
                continue
            frecuencia[token.lema] += 1
            vistos.add(token.lema)
            if token.pos == "ADJ":
                adjetivos_vistos.add(token.lema)
        for lema in vistos:
            df_lema[lema] += 1
        for indice_etiqueta in range(len(ETIQUETAS)):
            destino = df_positivo if etiquetas_pliegue[posicion, indice_etiqueta] else df_negativo
            destino[indice_etiqueta].update(vistos)
    total_frecuencia = sum(frecuencia.values())
    scores_por_etiqueta = []
    for indice_etiqueta in range(len(ETIQUETAS)):
        n_positivos = int(etiquetas_pliegue[:, indice_etiqueta].sum())
        n_negativos = int(len(etiquetas_pliegue) - n_positivos)
        scores_por_etiqueta.append(
            log_odds_informativo(
                df_positivo[indice_etiqueta],
                df_negativo[indice_etiqueta],
                n_positivos,
                n_negativos,
                frecuencia,
                total_frecuencia,
            )
        )
    candidatos = {
        lema
        for lema, apariciones in frecuencia.items()
        if apariciones >= CONFIG["min_frecuencia_lexico"]
        and df_lema[lema] >= CONFIG["min_documentos_lexico"]
    }
    insulto: set[str] = set()
    for scores in scores_por_etiqueta:
        insulto.update(
            lema
            for lema in candidatos
            if scores.get(lema, 0.0) > CONFIG["umbral_log_odds_insulto"]
        )
    indice_ag = list(ETIQUETAS).index("AG")
    indice_hs = list(ETIQUETAS).index("HS")
    adjetivos_negativos = {
        lema
        for lema in candidatos
        if lema in adjetivos_vistos
        and max(
            scores_por_etiqueta[indice_ag].get(lema, 0.0),
            scores_por_etiqueta[indice_hs].get(lema, 0.0),
        )
        > CONFIG["umbral_log_odds_adjetivo"]
    }
    colectivos = inventario_colectivo(documentos)
    tamanos = {
        "insulto": len(insulto),
        "adjetivos_negativos": len(adjetivos_negativos),
        "colectivos": len(colectivos),
        "vocabulario_candidato": len(candidatos),
    }
    return LexicosPliegue(
        insulto=frozenset(insulto),
        adjetivos_negativos=frozenset(adjetivos_negativos),
        colectivos=frozenset(colectivos),
        tamanos=tamanos,
    )


def es_gentilicio(lema: str) -> bool:
    """Detecta gentilicios por sufijo (regla morfologica documentada).

    Args:
        lema: lema normalizado a ASCII.

    Returns:
        ``True`` si el lema termina en un sufijo de gentilicio conocido y no es
        un falso amigo documentado.
    """
    return (
        len(lema) >= 6
        and lema.isalpha()
        and lema.endswith(SUFIJOS_GENTILICIO)
        and lema not in GENTILICIO_EXCLUIDOS
    )


def inventario_colectivo(documentos: Sequence[DocumentoAnotado]) -> set[str]:
    """Minera el inventario de terminos colectivos del pliegue de entrenamiento.

    Un termino entra si aparece al menos ``min_documentos_colectivo`` veces como
    nombre/adjetivo en plural precedido por un determinante en plural.

    Args:
        documentos: documentos anotados del pliegue de entrenamiento.

    Returns:
        Conjunto de lemas colectivos.
    """
    conteo: Counter = Counter()
    vistos: set[str] = set()
    for documento in documentos:
        vistos.clear()
        for indice in range(1, len(documento.tokens)):
            token = documento.tokens[indice]
            if (
                token.pos in POS_COLECTIVO
                and token.numero == "Plur"
                and documento.tokens[indice - 1].pos == "DET"
                and documento.tokens[indice - 1].numero == "Plur"
            ):
                vistos.add(token.lema)
        conteo.update(vistos)
    return {
        lema for lema, apariciones in conteo.items() if apariciones >= CONFIG["min_documentos_colectivo"]
    }


# ===== SECCION 4: detecciones de target y features E2-05 / E2-06 =====


@dataclass(frozen=True)
class AnalisisDocumento:
    """Representacion del documento para el canal de targeting.

    Attributes:
        secuencia: secuencia simbolica separada por espacios.
        numericos: vector de features E2-05 + E2-06 en el orden de
            :data:`NOMBRES_NUMERICOS`.
        pos_targets: posiciones de token de los targets detectados.
        pos_insultos: posiciones de token de los insultos detectados.
    """

    secuencia: str
    numericos: NDArray[np.float64]
    pos_targets: tuple[int, ...]
    pos_insultos: tuple[int, ...]


def es_segunda_persona(token: TokenAnotado, constantes: dict[str, Any]) -> bool:
    """Detecta marcas de segunda persona (pronombre, verbo conjugado o imperativo).

    Args:
        token: token anotado.
        constantes: constantes estructurales del script 06.

    Returns:
        ``True`` si el token es de segunda persona.
    """
    if token.lema in constantes["segunda_persona"]:
        return True
    if token.persona == "2":
        return True
    return es_imperativo(token, constantes)


def es_imperativo(token: TokenAnotado, constantes: dict[str, Any]) -> bool:
    """Detecta imperativos por morfologia o por el lexico del script 06.

    Args:
        token: token anotado.
        constantes: constantes estructurales del script 06.

    Returns:
        ``True`` si el token es un imperativo.
    """
    return token.modo == "Imp" or token.lema in constantes["imperativos"]


def _agregados_distancias(
    tokens: Sequence[TokenAnotado],
    pos_targets: Sequence[int],
    pos_insultos: Sequence[int],
) -> list[float]:
    """Calcula los agregados de distancia, orden y rango de los pares target-insulto.

    Las distancias se miden en tokens (diferencia de posiciones) y en caracteres
    (diferencia absoluta de desplazamientos). Cuando no hay pares, las columnas
    de distancia usan el centinela ``-1.0`` para distinguir "sin par" de "par
    adyacente" (distancia 0).

    Args:
        tokens: tokens del documento.
        pos_targets: posiciones de los targets.
        pos_insultos: posiciones de los insultos.

    Returns:
        Lista de 15 valores: 4 distancias en tokens, 4 en caracteres, 3 de orden
        y 4 binarias de rango (misma oracion, <= 3, <= 5 y <= 1 tokens).
    """
    if not pos_targets or not pos_insultos:
        return [-1.0] * 8 + [0.0] * 7
    dist_tok = []
    dist_car = []
    orden = []
    misma_oracion = 0
    le3 = 0
    le5 = 0
    adyacente = 0
    for posicion_target in pos_targets:
        for posicion_insulto in pos_insultos:
            distancia = abs(posicion_target - posicion_insulto)
            dist_tok.append(float(distancia))
            dist_car.append(
                float(abs(tokens[posicion_target].inicio - tokens[posicion_insulto].inicio))
            )
            orden.append(float(np.sign(posicion_insulto - posicion_target)))
            if tokens[posicion_target].oracion == tokens[posicion_insulto].oracion:
                misma_oracion = 1
            if distancia <= 3:
                le3 = 1
            if distancia <= 5:
                le5 = 1
            if distancia <= 1:
                adyacente = 1
    arreglo_tok = np.array(dist_tok)
    arreglo_car = np.array(dist_car)
    arreglo_orden = np.array(orden)
    return [
        float(arreglo_tok.min()),
        float(arreglo_tok.mean()),
        float(np.median(arreglo_tok)),
        float(arreglo_tok.max()),
        float(arreglo_car.min()),
        float(arreglo_car.mean()),
        float(np.median(arreglo_car)),
        float(arreglo_car.max()),
        float(arreglo_orden.mean()),
        float(np.mean(arreglo_orden < 0)),
        float(np.mean(arreglo_orden > 0)),
        float(misma_oracion),
        float(le3),
        float(le5),
        float(adyacente),
    ]


def _agregados_colectivo(
    tokens: Sequence[TokenAnotado],
    pos_colectivos: Sequence[int],
    pos_insultos: Sequence[int],
    pos_adj_negativos: Sequence[int],
    pos_universales: Sequence[int],
    pos_segunda: Sequence[int],
) -> list[float]:
    """Calcula el bloque E2-06 (colectivo + agresion).

    Args:
        tokens: tokens del documento.
        pos_colectivos: posiciones de terminos colectivos.
        pos_insultos: posiciones de insultos.
        pos_adj_negativos: posiciones de adjetivos negativos inducidos.
        pos_universales: posiciones de cuantificadores universales.
        pos_segunda: posiciones de marcas de segunda persona.

    Returns:
        Lista de valores en el orden de :data:`NOMBRES_E2_06`.
    """
    n_tokens = max(len(tokens), 1)
    n_colectivos = len(pos_colectivos)
    pares_colectivo_insulto = [
        abs(posicion - posicion_insulto)
        for posicion in pos_colectivos
        for posicion_insulto in pos_insultos
    ]
    pares_colectivo_adj = [
        abs(posicion - posicion_adj)
        for posicion in pos_colectivos
        for posicion_adj in pos_adj_negativos
    ]
    plurales = [
        token
        for token in tokens
        if token.pos in POS_COLECTIVO and token.numero == "Plur" and token.pos != "PROPN"
    ]
    plural_generico = int(len(plurales) >= 2 and not pos_segunda and not any(
        es_mencion(token) for token in tokens
    ))
    return [
        float(n_colectivos > 0),
        float(n_colectivos),
        float(np.log1p(n_colectivos)),
        n_colectivos / n_tokens,
        float(bool(pares_colectivo_insulto)),
        float(len(pares_colectivo_insulto)),
        float(any(distancia <= 3 for distancia in pares_colectivo_insulto)),
        float(bool(pares_colectivo_adj)),
        float(len(pares_colectivo_adj)),
        float(bool(pos_universales)),
        len(plurales) / n_tokens,
        float(plural_generico),
        float(n_colectivos > 0 and bool(pos_segunda)),
    ]


def features_e2_05(
    tokens: Sequence[TokenAnotado],
    pos_targets: Sequence[int],
    pos_insultos: Sequence[int],
    pos_mencion_inicial: Sequence[int],
    pos_mencion_interna: Sequence[int],
    pos_propn: Sequence[int],
) -> list[float]:
    """Construye el bloque E2-05 completo (conteos + distancias + orden).

    Args:
        tokens: tokens del documento.
        pos_targets: posiciones de los targets.
        pos_insultos: posiciones de los insultos.
        pos_mencion_inicial: posiciones de menciones iniciales.
        pos_mencion_interna: posiciones de menciones internas.
        pos_propn: posiciones de PROPN no mencion.

    Returns:
        Lista de valores en el orden de :data:`NOMBRES_E2_05`.
    """
    n_pares = len(pos_targets) * len(pos_insultos)
    cabecera = [
        float(bool(pos_targets) and bool(pos_insultos)),
        float(n_pares),
        float(len(pos_targets)),
        float(len(pos_insultos)),
        float(len(pos_mencion_inicial)),
        float(len(pos_mencion_interna)),
        float(len(pos_propn)),
        n_pares / max(len(pos_targets), 1),
    ]
    return cabecera + _agregados_distancias(tokens, pos_targets, pos_insultos)


def simbolizar(
    tokens: Sequence[TokenAnotado],
    lexicos: LexicosPliegue,
    constantes: dict[str, Any],
) -> list[str]:
    """Construye la secuencia simbolica del documento.

    Prioridad documentada por token: TARGET (mencion o PROPN), INSULT,
    COLLECTIVE, SECOND_PERSON, NEG; la puntuacion es ``SEP`` y el resto ``O``.
    Los tramos entre comillas se marcan con un simbolo ``QUOTE`` en cada
    frontera para conservar la estructura de cita sin perder el contenido.

    Args:
        tokens: tokens del documento.
        lexicos: lexicos inducidos del pliegue.
        constantes: constantes estructurales del script 06.

    Returns:
        Lista de simbolos.
    """
    simbolos: list[str] = []
    en_comillas_previo = False
    for indice, token in enumerate(tokens):
        if token.en_comillas != en_comillas_previo:
            simbolos.append(SIMBOLO_CITA)
            en_comillas_previo = token.en_comillas
        if es_mencion(token) or token.pos == "PROPN":
            simbolos.append(SIMBOLO_TARGET)
        elif token.lema in lexicos.insulto:
            simbolos.append(SIMBOLO_INSULTO)
        elif es_colectivo_estructural(tokens, indice, constantes) or (
            token.lema in lexicos.colectivos and token.numero == "Plur"
        ) or (token.pos in POS_COLECTIVO and es_gentilicio(token.lema)):
            simbolos.append(SIMBOLO_COLECTIVO)
        elif es_segunda_persona(token, constantes):
            simbolos.append(SIMBOLO_SEGUNDA)
        elif token.lema in NEGACION:
            simbolos.append(SIMBOLO_NEG)
        elif token.texto and not any(caracter.isalnum() for caracter in token.texto):
            simbolos.append(SIMBOLO_PUNTUACION)
        else:
            simbolos.append(SIMBOLO_RESTO)
    if en_comillas_previo:
        simbolos.append(SIMBOLO_CITA)
    return simbolos


def analizar_documento(
    documento: DocumentoAnotado,
    lexicos: LexicosPliegue,
    constantes: dict[str, Any],
) -> AnalisisDocumento:
    """Calcula simbolos, features numericas y detecciones de un documento.

    Args:
        documento: documento anotado.
        lexicos: lexicos inducidos con el pliegue de entrenamiento.
        constantes: constantes estructurales del script 06.

    Returns:
        Analisis con secuencia simbolica y vector numerico.
    """
    tokens = documento.tokens
    pos_mencion_inicial = []
    pos_mencion_interna = []
    pos_propn = []
    for indice, token in enumerate(tokens):
        if es_mencion(token):
            primero = 0
            if tokens and tokens[0].texto.lower() == "rt" and len(tokens) > 1:
                primero = 1
            if indice == primero:
                pos_mencion_inicial.append(indice)
            else:
                pos_mencion_interna.append(indice)
        elif token.pos == "PROPN":
            pos_propn.append(indice)
    pos_targets = sorted(pos_mencion_inicial + pos_mencion_interna + pos_propn)
    pos_insultos = [indice for indice, token in enumerate(tokens) if token.lema in lexicos.insulto]
    pos_colectivos = [
        indice
        for indice, token in enumerate(tokens)
        if es_colectivo_estructural(tokens, indice, constantes)
        or (token.lema in lexicos.colectivos and token.numero == "Plur")
        or (token.pos in POS_COLECTIVO and es_gentilicio(token.lema))
    ]
    pos_adj_negativos = [
        indice for indice, token in enumerate(tokens) if token.lema in lexicos.adjetivos_negativos
    ]
    pos_segunda = [
        indice for indice, token in enumerate(tokens) if es_segunda_persona(token, constantes)
    ]
    pos_universales = [
        indice
        for indice, token in enumerate(tokens)
        if token.lema in CUANTIFICADORES_UNIVERSALES
    ]
    numericos = np.array(
        features_e2_05(
            tokens, pos_targets, pos_insultos, pos_mencion_inicial, pos_mencion_interna, pos_propn
        )
        + _agregados_colectivo(
            tokens, pos_colectivos, pos_insultos, pos_adj_negativos, pos_universales, pos_segunda
        ),
        dtype=float,
    )
    if numericos.shape[0] != len(NOMBRES_NUMERICOS):
        raise RuntimeError(
            f"Features numericas mal alineadas: {numericos.shape[0]} vs {len(NOMBRES_NUMERICOS)}"
        )
    return AnalisisDocumento(
        secuencia=" ".join(simbolizar(tokens, lexicos, constantes)),
        numericos=numericos,
        pos_targets=tuple(pos_targets),
        pos_insultos=tuple(pos_insultos),
    )


# ===== SECCION 5: variantes y evaluacion out-of-fold =====


@dataclass(frozen=True)
class ResultadoOOF:
    """Salida de la evaluacion por pliegues.

    Attributes:
        oof: diccionario ``clave -> probabilidades (n, 3)``; las claves son
            ``(variante, C)`` y ``("BASE", 2.0)``.
        pliegue_de_fila: numero de pliegue (1..5) de cada fila.
        diagnostico: estadisticas de lexicos y detecciones por pliegue.
    """

    oof: dict[tuple[str, float], NDArray[np.float64]]
    pliegue_de_fila: NDArray[np.int_]
    diagnostico: list[dict[str, Any]]


def cargar_receta_v03() -> Any:
    """Importa ``experiments/14_oof_corregido_v2.py`` para reutilizar su receta.

    Returns:
        Modulo del experimento 14 (``construir_matrices`` y ``MODELOS``).

    Raises:
        ImportError: si no se puede cargar el modulo.
        RuntimeError: si la receta no coincide con la esperada (M2 con C=2.0).
    """
    ruta = PROJECT_ROOT / "experiments" / "14_oof_corregido_v2.py"
    especificacion = importlib.util.spec_from_file_location("experimento_14_base_p2", ruta)
    if especificacion is None or especificacion.loader is None:
        raise ImportError(f"No se pudo cargar {ruta}")
    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = modulo
    especificacion.loader.exec_module(modulo)
    if tuple(modulo.MODELOS) != MODELOS_BASE_ESPERADOS:
        raise RuntimeError("Los modelos base de experiments/14 no coinciden con el OOF congelado")
    if modulo.MODELOS["M2_union_lr_c2"]["params"]["C"] != 2.0:
        raise RuntimeError("La receta M2 de experiments/14 no es C=2.0")
    if modulo.PARAMS_WORD != PARAMS_WORD or modulo.PARAMS_CHAR != PARAMS_CHAR:
        raise RuntimeError("La representacion word/char de experiments/14 no coincide con la esperada")
    return modulo


def ajustar_lr(
    nombre_variante: str,
    valor_c: float,
    x_entrenamiento: sparse.csr_matrix,
    etiquetas_entrenamiento: NDArray[np.int_],
    x_prediccion: sparse.csr_matrix,
) -> NDArray[np.float64]:
    """Ajusta una regresion logistica balanceada y devuelve probabilidades.

    Args:
        nombre_variante: nombre de la variante (para trazas).
        valor_c: valor de ``C``.
        x_entrenamiento: matriz del pliegue de entrenamiento.
        etiquetas_entrenamiento: etiqueta binaria del pliegue.
        x_prediccion: matriz a predecir.

    Returns:
        Probabilidades de la clase positiva.
    """
    del nombre_variante
    modelo = LogisticRegression(C=valor_c, **PARAMS_LR)
    modelo.fit(x_entrenamiento, etiquetas_entrenamiento)
    return modelo.predict_proba(x_prediccion)[:, 1]


def evaluar_pliegues(
    desarrollo: pd.DataFrame,
    documentos: Sequence[DocumentoAnotado],
    y_verdadera: NDArray[np.int_],
    pliegues: list[tuple[NDArray[np.int_], NDArray[np.int_]]],
    constantes: dict[str, Any],
    receta_v03: Any,
) -> ResultadoOOF:
    """Evalua las cuatro variantes (y la base V03) fold a fold.

    Dentro de cada pliegue: se inducen los lexicos con el entrenamiento, se
    anotan las features, se ajustan los vectorizadores y los modelos; nada de
    validacion entra en ningun ajuste.

    Args:
        desarrollo: DataFrame del desarrollo.
        documentos: documentos anotados del desarrollo.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        pliegues: pliegues corregidos ``(entrenamiento, validacion)``.
        constantes: constantes estructurales del script 06.
        receta_v03: modulo de ``experiments/14``.

    Returns:
        Probabilidades OOF por clave, el pliegue de cada fila y diagnosticos.
    """
    textos = desarrollo["text"]
    claves = [(variante, valor_c) for variante in NOMBRES_VARIANTES for valor_c in CONFIG["C_variantes"]]
    claves.append(("BASE", 2.0))
    oof = {clave: np.zeros((len(desarrollo), len(ETIQUETAS)), dtype=float) for clave in claves}
    pliegue_de_fila = np.zeros(len(desarrollo), dtype=int)
    diagnostico: list[dict[str, Any]] = []
    for numero, (idx_entrenamiento, idx_validacion) in enumerate(pliegues, start=1):
        pliegue_de_fila[idx_validacion] = numero
        lexicos = inducir_lexicos(
            [documentos[indice] for indice in idx_entrenamiento], y_verdadera[idx_entrenamiento]
        )
        analisis_entrenamiento = [
            analizar_documento(documentos[indice], lexicos, constantes) for indice in idx_entrenamiento
        ]
        analisis_validacion = [
            analizar_documento(documentos[indice], lexicos, constantes) for indice in idx_validacion
        ]
        numericas_entrenamiento = np.vstack([analisis.numericos for analisis in analisis_entrenamiento])
        numericas_validacion = np.vstack([analisis.numericos for analisis in analisis_validacion])
        escalador = StandardScaler().fit(numericas_entrenamiento)
        numericas_entrenamiento = escalador.transform(numericas_entrenamiento)
        numericas_validacion = escalador.transform(numericas_validacion)

        m_word_ent, m_word_val = receta_v03.construir_matrices(
            textos.iloc[idx_entrenamiento], textos.iloc[idx_validacion], "word"
        )
        m_char_ent, m_char_val = receta_v03.construir_matrices(
            textos.iloc[idx_entrenamiento], textos.iloc[idx_validacion], "char"
        )
        vectorizador_simbolico = TfidfVectorizer(**PARAMS_SIMBOLICO)
        m_sim_ent = vectorizador_simbolico.fit_transform(
            [analisis.secuencia for analisis in analisis_entrenamiento]
        )
        m_sim_val = vectorizador_simbolico.transform(
            [analisis.secuencia for analisis in analisis_validacion]
        )
        bloque_texto_ent = sparse.hstack([m_word_ent, m_char_ent], format="csr")
        bloque_texto_val = sparse.hstack([m_word_val, m_char_val], format="csr")
        bloques = {
            "CH-TGT": (m_sim_ent, m_sim_val),
            "CH-TGT+DIST": (
                sparse.hstack([m_sim_ent, sparse.csr_matrix(numericas_entrenamiento)], format="csr"),
                sparse.hstack([m_sim_val, sparse.csr_matrix(numericas_validacion)], format="csr"),
            ),
            "TXT+CH": (
                sparse.hstack(
                    [bloque_texto_ent, m_sim_ent, sparse.csr_matrix(numericas_entrenamiento)],
                    format="csr",
                ),
                sparse.hstack(
                    [bloque_texto_val, m_sim_val, sparse.csr_matrix(numericas_validacion)],
                    format="csr",
                ),
            ),
            "TXT+TGT-SPARSE": (
                sparse.hstack([bloque_texto_ent, m_sim_ent], format="csr"),
                sparse.hstack([bloque_texto_val, m_sim_val], format="csr"),
            ),
        }
        for nombre_variante, (x_entrenamiento, x_validacion) in bloques.items():
            for valor_c in CONFIG["C_variantes"]:
                for indice_etiqueta in range(len(ETIQUETAS)):
                    oof[(nombre_variante, valor_c)][idx_validacion, indice_etiqueta] = ajustar_lr(
                        nombre_variante,
                        valor_c,
                        x_entrenamiento,
                        y_verdadera[idx_entrenamiento, indice_etiqueta],
                        x_validacion,
                    )
        for indice_etiqueta in range(len(ETIQUETAS)):
            oof[("BASE", 2.0)][idx_validacion, indice_etiqueta] = ajustar_lr(
                "BASE",
                2.0,
                bloque_texto_ent,
                y_verdadera[idx_entrenamiento, indice_etiqueta],
                bloque_texto_val,
            )
        diagnostico.append(
            {
                "pliegue": numero,
                "n_entrenamiento": len(idx_entrenamiento),
                "n_validacion": len(idx_validacion),
                "lexicos": lexicos.tamanos,
                "dims": {
                    "simbolico": int(m_sim_ent.shape[1]),
                    "texto_union": int(bloque_texto_ent.shape[1]),
                },
            }
        )
        log(
            f"pliegue {numero}: lexico insulto={lexicos.tamanos['insulto']} "
            f"adj_neg={lexicos.tamanos['adjetivos_negativos']} "
            f"colectivos={lexicos.tamanos['colectivos']} dim_simbolico={m_sim_ent.shape[1]}"
        )
    return ResultadoOOF(oof=oof, pliegue_de_fila=pliegue_de_fila, diagnostico=diagnostico)


def vector_umbrales(umbrales: dict[str, float]) -> NDArray[np.float64]:
    """Convierte el diccionario de umbrales en vector en el orden de etiquetas.

    Args:
        umbrales: umbral por etiqueta.

    Returns:
        Arreglo ``(3,)`` en el orden de :data:`ETIQUETAS`.
    """
    return np.array([umbrales[etiqueta] for etiqueta in ETIQUETAS])


def predecir(proba: NDArray[np.float64], umbrales: dict[str, float]) -> NDArray[np.int_]:
    """Binariza las probabilidades con umbrales por etiqueta.

    Args:
        proba: probabilidades ``(n, 3)``.
        umbrales: umbral por etiqueta.

    Returns:
        Matriz binaria ``(n, 3)``.
    """
    return (proba >= vector_umbrales(umbrales)).astype(int)


def f1_media_rapida(y_verdadera: NDArray[np.int_], prediccion: NDArray[np.int_]) -> float:
    """Calcula la F1 media por columna de forma vectorizada.

    Args:
        y_verdadera: matriz binaria verdadera ``(n, k)``.
        prediccion: matriz binaria predicha ``(n, k)``.

    Returns:
        Media de la F1 binaria de la clase positiva por columna.
    """
    valores = []
    for posicion in range(y_verdadera.shape[1]):
        verdad = y_verdadera[:, posicion].astype(bool)
        aciertos = prediccion[:, posicion].astype(bool)
        verdaderos_positivos = int(np.count_nonzero(verdad & aciertos))
        falsos_positivos = int(np.count_nonzero(~verdad & aciertos))
        falsos_negativos = int(np.count_nonzero(verdad & ~aciertos))
        denominador = 2 * verdaderos_positivos + falsos_positivos + falsos_negativos
        valores.append(2 * verdaderos_positivos / denominador if denominador else 0.0)
    return float(np.mean(valores))


def recall_columna(
    y_columna: NDArray[np.int_], prediccion: NDArray[np.int_], mascara: NDArray[np.bool_]
) -> float:
    """Calcula el recall de una columna sobre una rebanada.

    Args:
        y_columna: etiqueta verdadera de la columna.
        prediccion: prediccion binaria de la columna.
        mascara: rebanada sobre la que evaluar.

    Returns:
        Recall (``0.0`` si no hay positivos verdaderos en la rebanada).
    """
    verdad = y_columna[mascara].astype(bool)
    aciertos = prediccion[mascara].astype(bool)
    positivos = int(np.count_nonzero(verdad))
    if positivos == 0:
        return 0.0
    return float(np.count_nonzero(verdad & aciertos) / positivos)


def f1_por_pliegue(
    proba: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
    umbrales: dict[str, float],
) -> list[float]:
    """Calcula la F1 media por columna en cada pliegue.

    Args:
        proba: probabilidades ``(n, 3)``.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        pliegue_de_fila: numero de pliegue de cada fila.
        umbrales: umbral por etiqueta.

    Returns:
        Lista de F1 por pliegue en orden creciente de pliegue.
    """
    return [
        metricas_multietiqueta(
            y_verdadera[pliegue_de_fila == numero],
            predecir(proba[pliegue_de_fila == numero], umbrales),
        )["f1_media_columnas"]
        for numero in sorted(np.unique(pliegue_de_fila).tolist())
    ]


def bootstrap_delta(
    proba_referencia: NDArray[np.float64],
    proba_variante: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    umbrales_referencia: dict[str, float],
    umbrales_variante: dict[str, float],
) -> dict:
    """Calcula el IC bootstrap pareado del delta de F1 a nivel de tweet.

    Args:
        proba_referencia: probabilidades de la referencia.
        proba_variante: probabilidades de la variante.
        y_verdadera: etiquetas verdaderas.
        umbrales_referencia: umbrales de la referencia.
        umbrales_variante: umbrales de la variante.

    Returns:
        Diccionario con media, cuantiles 2.5/97.5 y fraccion positiva.
    """
    generador = np.random.default_rng(RANDOM_STATE)
    n_filas = len(y_verdadera)
    prediccion_ref = predecir(proba_referencia, umbrales_referencia)
    prediccion_var = predecir(proba_variante, umbrales_variante)
    deltas = np.empty(CONFIG["n_bootstrap"], dtype=float)
    for iteracion in range(CONFIG["n_bootstrap"]):
        muestra = generador.integers(0, n_filas, size=n_filas)
        verdad = y_verdadera[muestra]
        deltas[iteracion] = f1_media_rapida(verdad, prediccion_var[muestra]) - f1_media_rapida(
            verdad, prediccion_ref[muestra]
        )
    return {
        "media": float(np.mean(deltas)),
        "ic_inferior": float(np.quantile(deltas, 0.025)),
        "ic_superior": float(np.quantile(deltas, 0.975)),
        "fraccion_positiva": float(np.mean(deltas > 0)),
        "remuestreos": int(CONFIG["n_bootstrap"]),
    }


def resumir_clave(
    clave: tuple[str, float],
    proba: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
    proba_base: NDArray[np.float64],
    f1_pliegues_base: list[float],
    f1_pliegues_v03: list[float],
    mascara_colectiva: NDArray[np.bool_],
) -> dict:
    """Resume una clave (variante, C) frente a las referencias del protocolo v2.

    Args:
        clave: ``(variante, C)``.
        proba: probabilidades OOF de la clave.
        y_verdadera: etiquetas verdaderas.
        pliegue_de_fila: numero de pliegue de cada fila.
        proba_base: probabilidades de la base M2 (V03 sin umbrales).
        f1_pliegues_base: F1 por pliegue de la base a 0.5.
        f1_pliegues_v03: F1 por pliegue de la base con los umbrales de V03.
        mascara_colectiva: rebanada de referencia colectiva.

    Returns:
        Diccionario con F1 global, por etiqueta, por pliegue, deltas,
        complementariedad y el recall de AG en la rebanada colectiva.
    """
    sin_umbral = {etiqueta: 0.5 for etiqueta in ETIQUETAS}
    variante, valor_c = clave
    metrica = metricas_multietiqueta(y_verdadera, predecir(proba, sin_umbral))
    f1_pliegues = f1_por_pliegue(proba, y_verdadera, pliegue_de_fila, sin_umbral)
    prediccion = predecir(proba, sin_umbral)
    prediccion_base_05 = predecir(proba_base, sin_umbral)
    prediccion_base_v03 = predecir(proba_base, CONFIG["umbrales_v03"])
    resumen = {
        "clave": f"{variante}@C={valor_c}",
        "variante": variante,
        "C": valor_c,
        "corrected_cv": metrica["f1_media_columnas"],
        "f1_por_etiqueta": metrica["f1_por_etiqueta"],
        "f1_pliegues": f1_pliegues,
        "std_pliegues": float(np.std(f1_pliegues)),
        "delta_vs_m2_05": metrica["f1_media_columnas"] - CONFIG["referencia_m2_umbral_05"],
        "delta_vs_v03_umbrales": metrica["f1_media_columnas"] - CONFIG["referencia_v03_umbrales"],
        "delta_vs_v2a_meta": metrica["f1_media_columnas"] - CONFIG["referencia_v2a_meta"],
        "pliegues_mejorados_vs_m2_05": int(np.sum(np.array(f1_pliegues) > np.array(f1_pliegues_base))),
        "pliegues_mejorados_vs_v03": int(np.sum(np.array(f1_pliegues) > np.array(f1_pliegues_v03))),
        "ag_recall_slice": recall_columna(y_verdadera[:, 2], prediccion[:, 2], mascara_colectiva),
        "ag_recall_slice_v03": recall_columna(
            y_verdadera[:, 2], prediccion_base_v03[:, 2], mascara_colectiva
        ),
        "ag_recall_fuera_slice_v03": recall_columna(
            y_verdadera[:, 2], prediccion_base_v03[:, 2], ~mascara_colectiva
        ),
        "ag_recall_fuera_slice": recall_columna(y_verdadera[:, 2], prediccion[:, 2], ~mascara_colectiva),
        "ag_recall_global": recall_columna(y_verdadera[:, 2], prediccion[:, 2], np.ones(len(y_verdadera), dtype=bool)),
        "ag_recall_global_v03": recall_columna(
            y_verdadera[:, 2], prediccion_base_v03[:, 2], np.ones(len(y_verdadera), dtype=bool)
        ),
        "ag_precision_global": float(
            np.count_nonzero((prediccion[:, 2] == 1) & (y_verdadera[:, 2] == 1))
            / max(np.count_nonzero(prediccion[:, 2] == 1), 1)
        ),
        "base_precision_ag": float(
            np.count_nonzero((prediccion_base_05[:, 2] == 1) & (y_verdadera[:, 2] == 1))
            / max(np.count_nonzero(prediccion_base_05[:, 2] == 1), 1)
        ),
    }
    if variante != "BASE":
        proba_ensemble = (proba + proba_base) / 2.0
        metrica_ensemble = metricas_multietiqueta(y_verdadera, predecir(proba_ensemble, sin_umbral))
        ganancia = metrica_ensemble["f1_media_columnas"] - CONFIG["referencia_m2_umbral_05"]
        resumen["ensemble_gain"] = ganancia
        resumen["corrected_cv_ensemble"] = metrica_ensemble["f1_media_columnas"]
        if ganancia > 0:
            resumen["bootstrap"] = bootstrap_delta(
                proba_base, proba_ensemble, y_verdadera, sin_umbral, sin_umbral
            )
    return resumen


def decidir(resumenes: dict[str, dict]) -> dict:
    """Aplica la regla de adopcion pre-registrada en D-016.

    Args:
        resumenes: resumen por clave; la clave ``BASE@C=2.0`` se excluye.

    Returns:
        Diccionario con la mejor variante y la decision.
    """
    candidatos = {
        clave: resumen for clave, resumen in resumenes.items() if resumen["variante"] != "BASE"
    }
    clave, fila = max(candidatos.items(), key=lambda par: par[1]["corrected_cv"])
    cumple_nivel = fila["corrected_cv"] >= CONFIG["referencia_v03_umbrales"] + CONFIG["mejora_minima"]
    cumple_pliegues = fila["pliegues_mejorados_vs_v03"] >= CONFIG["pliegues_minimos_mejorados"]
    ganancia = fila.get("ensemble_gain", 0.0)
    bootstrap = fila.get("bootstrap", {})
    cumple_complementariedad = bool(
        ganancia >= CONFIG["ensemble_gain_minimo"] and bootstrap.get("ic_inferior", 0.0) > 0
    )
    adoptado = bool((cumple_nivel and cumple_pliegues) or cumple_complementariedad)
    return {
        "mejor_clave": clave,
        "corrected_cv": fila["corrected_cv"],
        "criterio_nivel": {
            "objetivo": CONFIG["referencia_v03_umbrales"] + CONFIG["mejora_minima"],
            "cumple": bool(cumple_nivel),
        },
        "criterio_pliegues": {
            "objetivo": CONFIG["pliegues_minimos_mejorados"],
            "pliegues_mejorados": fila["pliegues_mejorados_vs_v03"],
            "cumple": bool(cumple_pliegues),
        },
        "criterio_complementariedad": {
            "objetivo_gain": CONFIG["ensemble_gain_minimo"],
            "ensemble_gain": ganancia,
            "ic_bootstrap": [bootstrap.get("ic_inferior"), bootstrap.get("ic_superior")],
            "cumple": cumple_complementariedad,
        },
        "adoptado": adoptado,
        "justificacion": (
            "Adoptada: supera el nivel exigido con 3/5 pliegues o demuestra complementariedad con IC > 0."
            if adoptado
            else "No adoptada: no alcanza el nivel exigido con 3/5 pliegues ni la complementariedad "
            "exigida; la linea de targeting estructural se cierra documentando el resultado."
        ),
    }


# ===== SECCION 6: figura, artefactos, contexto y main =====


def generar_figura(
    resumenes: dict[str, dict],
    mejores_por_variante: dict[str, str],
    f1_slice: dict[str, float],
) -> None:
    """Genera la figura comparativa de variantes, referencias y rebanada colectiva.

    Args:
        resumenes: resumen por clave.
        mejores_por_variante: clave con mejor corrected_cv por variante.
        f1_slice: recall de AG en la rebanada colectiva (y fuera de ella) por
            etiqueta de barra.
    """
    figura, (eje_a, eje_b) = plt.subplots(1, 2, figsize=(14, 5.5))
    nombres = [mejores_por_variante[variante] for variante in NOMBRES_VARIANTES]
    valores = [resumenes[clave]["corrected_cv"] for clave in nombres]
    nombres = [resumenes[clave]["clave"] for clave in nombres]
    nombres.append("BASE (M2)")
    valores.append(resumenes["BASE@C=2.0"]["corrected_cv"])
    colores = ["#4c72b0"] * (len(nombres) - 1) + ["#8c8c8c"]
    posiciones = list(range(len(nombres)))
    eje_a.barh(posiciones, valores, color=colores)
    eje_a.axvline(
        CONFIG["referencia_m2_umbral_05"],
        color="#d62728",
        linestyle="--",
        linewidth=1.1,
        label=f"M2 a 0.5 ({CONFIG['referencia_m2_umbral_05']:.4f})",
    )
    eje_a.axvline(
        CONFIG["referencia_v03_umbrales"],
        color="#2ca02c",
        linestyle="-.",
        linewidth=1.1,
        label=f"V03 umbrales ({CONFIG['referencia_v03_umbrales']:.4f})",
    )
    eje_a.axvline(
        CONFIG["referencia_v2a_meta"],
        color="#9467bd",
        linestyle=":",
        linewidth=1.1,
        label=f"v2-A meta ({CONFIG['referencia_v2a_meta']:.4f})",
    )
    eje_a.set_yticks(posiciones, nombres, fontsize=9)
    eje_a.set_xlabel("corrected_cv (F1 media por columna, OOF, umbral 0.5)")
    eje_a.set_title("P2 - Targeting estructural (desarrollo n=1483)")
    eje_a.legend(loc="lower right", fontsize=8)
    for posicion, valor in zip(posiciones, valores, strict=True):
        eje_a.text(valor + 0.0004, posicion, f"{valor:.4f}", va="center", fontsize=8)
    eje_a.set_xlim(min(valores) - 0.006, max(valores) + 0.006)

    etiquetas_b = list(f1_slice)
    valores_b = [f1_slice[etiqueta] for etiqueta in etiquetas_b]
    colores_b = ["#8c8c8c" if "V03" in etiqueta else "#4c72b0" for etiqueta in etiquetas_b]
    eje_b.bar(etiquetas_b, valores_b, color=colores_b)
    eje_b.set_ylabel("recall de AG")
    eje_b.set_title("E2-06: recall de AG en la rebanada colectiva de referencia")
    eje_b.tick_params(axis="x", labelrotation=15)
    for posicion, valor in enumerate(valores_b):
        eje_b.text(posicion, valor + 0.01, f"{valor:.3f}", ha="center", fontsize=8)
    eje_b.set_ylim(0, 1.0)
    figura.tight_layout()
    FIGURA_PATH.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(FIGURA_PATH, dpi=150)
    plt.close(figura)


def escribir_csv(resumenes: dict[str, dict]) -> pd.DataFrame:
    """Escribe el CSV comparativo de variantes y referencias.

    Args:
        resumenes: resumen por clave.

    Returns:
        DataFrame escrito.
    """
    filas = []
    for resumen in resumenes.values():
        filas.append(
            {
                "clave": resumen["clave"],
                "variante": resumen["variante"],
                "C": resumen["C"],
                "corrected_cv": round(resumen["corrected_cv"], 4),
                "f1_HS": round(resumen["f1_por_etiqueta"]["HS"], 4),
                "f1_TR": round(resumen["f1_por_etiqueta"]["TR"], 4),
                "f1_AG": round(resumen["f1_por_etiqueta"]["AG"], 4),
                "std_pliegues": round(resumen["std_pliegues"], 4),
                "delta_vs_m2_05": round(resumen["delta_vs_m2_05"], 4),
                "delta_vs_v03_umbrales": round(resumen["delta_vs_v03_umbrales"], 4),
                "delta_vs_v2a_meta": round(resumen["delta_vs_v2a_meta"], 4),
                "pliegues_mejorados_vs_v03": resumen["pliegues_mejorados_vs_v03"],
                "f1_pliegues": "|".join(f"{valor:.4f}" for valor in resumen["f1_pliegues"]),
                "ag_recall_slice": round(resumen["ag_recall_slice"], 4),
                "ag_recall_slice_v03": round(resumen["ag_recall_slice_v03"], 4),
                "ensemble_gain": round(resumen.get("ensemble_gain", float("nan")), 4),
            }
        )
    tabla = pd.DataFrame(filas)
    REPORTE_CSV.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(REPORTE_CSV, index=False)
    return tabla


def guardar_oof_targeting(
    desarrollo: pd.DataFrame,
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
    oof: dict[tuple[str, float], NDArray[np.float64]],
    mejores: dict[str, tuple[str, float]],
) -> None:
    """Guarda las probabilidades OOF de los canales de targeting para P6.

    Args:
        desarrollo: DataFrame del desarrollo.
        y_verdadera: etiquetas verdaderas.
        pliegue_de_fila: pliegue de cada fila.
        oof: probabilidades por clave.
        mejores: mejor clave y C por variante (solo CH-TGT+DIST y TXT+CH).
    """
    nombres = ("CH-TGT+DIST", "TXT+CH")
    OOF_TARGETING_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OOF_TARGETING_PATH,
        ids=desarrollo["id"].astype(int).to_numpy(),
        y_true=y_verdadera,
        pliegue=pliegue_de_fila,
        proba=np.stack(
            [oof[(nombre, mejores[nombre][1])] for nombre in nombres], axis=1
        ),
        variantes=np.array(nombres),
        C=np.array([mejores[nombre][1] for nombre in nombres]),
        etiquetas=np.array(list(ETIQUETAS)),
    )
    log(f"OOF de targeting guardado en {OOF_TARGETING_PATH.name} (variantes {list(nombres)})")


def registrar_experimentos(
    resumenes: dict[str, dict],
    mejores: dict[str, tuple[str, float]],
    decision: dict,
    tamanos_lexicos: dict[str, float],
    duracion: float,
) -> None:
    """Anade las filas ``E2-05-*`` y ``E2-06-*`` a ``experiments.csv``.

    Args:
        resumenes: resumen por clave.
        mejores: mejor clave y C por variante.
        decision: decision del gate.
        tamanos_lexicos: medias de tamanos de lexicos por pliegue.
        duracion: segundos totales del experimento.
    """
    definiciones = {
        "CH-TGT": (
            "E2-05-CH-TGT",
            "Aporta el canal simbolico abstracto (TARGET/INSULT/COLLECTIVE/SECOND_PERSON/NEG/QUOTE) solo?",
        ),
        "CH-TGT+DIST": (
            "E2-05-CH-TGT-DIST",
            "Aportan las distancias y el orden target-insulto sobre el canal simbolico?",
        ),
        "TXT+CH": (
            "E2-06-TXT-CH",
            "Aporta el canal de targeting (simbolico + colectivo + distancias) sobre la representacion de V03?",
        ),
        "TXT+TGT-SPARSE": (
            "E2-06-TXT-TGT-SPARSE",
            "Aportan los n-gramas 1-2 del canal simbolico sobre la representacion de V03?",
        ),
    }
    filas = []
    for variante in NOMBRES_VARIANTES:
        clave, valor_c = mejores[variante]
        resumen = resumenes[clave]
        identificador, pregunta = definiciones[variante]
        filas.append(
            {
                "experiment_id": identificador,
                "competition": "KAGGLE_TOXIC_2026_PUJ",
                "question": pregunta,
                "hypothesis": (
                    "La relacion target-insulto (tipo de target, distancia, orden) y el colectivo "
                    "explican confusiones de targeting y los FN de AG (brechas #1, #2, #5, #6)"
                ),
                "preprocessing": (
                    "texto crudo + anotacion spaCy (POS/lema/morfologia/oraciones) + lexico de insulto "
                    "y adjetivos negativos inducidos fold-local (log-odds con prior de Dirichlet) + "
                    "inventario colectivo fold-local"
                ),
                "feature_groups": "TARGET_PROFILING+INSULT_DISTANCE+COLLECTIVE_PATTERN+FUNCTION_PATH"
                if variante in VARIANTES_TEXTO
                else "FUNCTION_PATH",
                "representation": (
                    "word(1,2)+char_wb(3,5) de V03 + "
                    if variante in VARIANTES_TEXTO
                    else ""
                )
                + "TF-IDF simbolico (1,2)"
                + (" + numericas E2-05/E2-06 estandarizadas" if "DIST" in variante or variante == "TXT+CH" else ""),
                "model": f"LogisticRegression C={valor_c} balanced",
                "hyperparameters": json.dumps({"C": valor_c, **PARAMS_LR}, ensure_ascii=False),
                "validation": "StratifiedGroupKFold(5) sobre desarrollo, grupos fusionados (D-016)",
                "f1": round(resumen["corrected_cv"], 4),
                "macro_f1": "",
                "weighted_f1": "",
                "precision": "",
                "recall": "",
                "training_time": round(duracion, 2),
                "dimension": "",
                "result": json.dumps(
                    {
                        "corrected_cv": resumen["corrected_cv"],
                        "f1_por_etiqueta": resumen["f1_por_etiqueta"],
                        "pliegues": resumen["f1_pliegues"],
                        "delta_vs_v03_umbrales": resumen["delta_vs_v03_umbrales"],
                        "delta_vs_v2a_meta": resumen["delta_vs_v2a_meta"],
                        "pliegues_mejorados_vs_v03": resumen["pliegues_mejorados_vs_v03"],
                        "ag_recall_slice": resumen["ag_recall_slice"],
                        "ag_recall_slice_v03": resumen["ag_recall_slice_v03"],
                        "ag_recall_fuera_slice": resumen["ag_recall_fuera_slice"],
                        "ensemble_gain": resumen.get("ensemble_gain"),
                        "bootstrap": resumen.get("bootstrap"),
                        "tamanos_lexicos_medios": tamanos_lexicos,
                        "por_C": {
                            f"C={otro_c}": round(
                                resumenes[f"{variante}@C={otro_c}"]["corrected_cv"], 4
                            )
                            for otro_c in CONFIG["C_variantes"]
                        },
                    },
                    ensure_ascii=False,
                ),
                "interpretation": (
                    f"Mejor C={valor_c} elegido por corrected_cv interno (cota optimista de seleccion); "
                    f"delta vs V03 {resumen['delta_vs_v03_umbrales']:+.4f} y vs v2-A {resumen['delta_vs_v2a_meta']:+.4f}"
                ),
                "decision": (
                    f"ADOPTADA como canal de P6 ({decision['justificacion']})"
                    if resumen["clave"] == decision["mejor_clave"] and decision["adoptado"]
                    else (
                        f"NO adoptada: {resumen['delta_vs_v03_umbrales']:+.4f} vs V03 "
                        f"({resumen['pliegues_mejorados_vs_v03']}/5 pliegues); canal guardado solo "
                        "para el stacking de P6"
                    )
                ),
                "date": CONFIG["fecha"],
                "script": "experiments/16_targeting_v2.py",
                "notes": "holdout congelado no utilizado; sin fuga (lexicos y escaladores solo del pliegue de entrenamiento)",
            }
        )
    existente = pd.read_csv(EXPERIMENTS_CSV)
    identificadores = {fila["experiment_id"] for fila in filas}
    existente = existente[~existente["experiment_id"].isin(identificadores)]
    existente.to_csv(EXPERIMENTS_CSV, index=False)
    pd.DataFrame(filas).to_csv(EXPERIMENTS_CSV, mode="a", header=False, index=False)
    log(f"filas registradas: {sorted(identificadores)}")


def escribir_contexto(lineas: list[str]) -> None:
    """Reemplaza o anade la seccion de P2 en ``reports/contexto.md``.

    Args:
        lineas: lineas de la seccion (sin encabezado).
    """
    encabezado = "## Fase v2 - P2 (2026-09-25)"
    bloque = "\n".join([encabezado, *lineas])
    texto = CONTEXTO_MD.read_text(encoding="utf-8")
    if encabezado in texto:
        inicio = texto.index(encabezado)
        fin = texto.find("\n## ", inicio + len(encabezado))
        fin = len(texto) if fin == -1 else fin + 1
        texto = texto[:inicio] + bloque + "\n" + texto[fin:]
    else:
        texto = texto.rstrip("\n") + "\n\n" + bloque + "\n"
    CONTEXTO_MD.write_text(texto, encoding="utf-8")
    log("seccion de P2 escrita en reports/contexto.md")


def actualizar_todo(resumenes: dict[str, dict], mejores: dict[str, tuple[str, float]], decision: dict) -> None:
    """Actualiza P2.1-P2.4 y el Gate P2 en ``tasks/todo.md``.

    Args:
        resumenes: resumen por clave.
        mejores: mejor clave y C por variante.
        decision: decision del gate.
    """
    def linea_mejor(variante: str) -> str:
        """Devuelve el texto resumido de la mejor clave de una variante."""
        resumen = resumenes[mejores[variante][0]]
        return (
            f"corrected_cv {resumen['corrected_cv']:.4f} "
            f"(delta vs V03 {resumen['delta_vs_v03_umbrales']:+.4f}, "
            f"pliegues {resumen['pliegues_mejorados_vs_v03']}/5; "
            f"AG recall colectiva {resumen['ag_recall_slice']:.3f} vs {resumen['ag_recall_slice_v03']:.3f} de V03)"
        )

    reemplazos = [
        (
            "- [pendiente] P2.1 Distancias target-insulto (token/char, orden, densidad, minimos)",
            (
                "- [hecho] P2.1 Distancias target-insulto (token/char, orden, densidad, minimos): "
                f"{linea_mejor('CH-TGT+DIST')}"
            ),
        ),
        (
            "- [pendiente] P2.2 Target profiling (persona/grupo/colectivo/gentilicio/2a persona)",
            (
                "- [hecho] P2.2 Target profiling (persona/grupo/colectivo/gentilicio/2a persona): canal "
                f"simbolico + PROPN/segunda persona/imperativo fold-locales; CH-TGT {linea_mejor('CH-TGT')}"
            ),
        ),
        (
            "- [pendiente] P2.3 Colectivo + agresion (linea de los FN de AG: 45.5 % vs 13.9 %)",
            (
                "- [hecho] P2.3 Colectivo + agresion (linea de los FN de AG: 45.5 % vs 13.9 %): "
                f"{linea_mejor('TXT+CH')}"
            ),
        ),
        (
            "- [pendiente] P2.4 Gate P2",
            (
                f"- [hecho] P2.4 Gate P2: {'ADOPTADO' if decision['adoptado'] else 'CERRADO sin mejora'} "
                f"({decision['mejor_clave']}, corrected_cv {decision['corrected_cv']:.4f})"
            ),
        ),
    ]
    texto = TODO_MD.read_text(encoding="utf-8")
    for viejo, nuevo in reemplazos:
        if viejo in texto:
            texto = texto.replace(viejo, nuevo)
    fila_gate = (
        f"| P2 | 2026-09-25 | {'SUPERADO' if decision['adoptado'] else 'CERRADO (sin mejora)'}. "
        f"E2-05/E2-06 targeting estructural: mejor variante {decision['mejor_clave']} con corrected_cv "
        f"{decision['corrected_cv']:.4f} "
        f"({decision['corrected_cv'] - CONFIG['referencia_v03_umbrales']:+.4f} vs V03 0.7772, "
        f"{decision['corrected_cv'] - CONFIG['referencia_v2a_meta']:+.4f} vs v2-A 0.7853, "
        f"{resumenes[decision['mejor_clave']]['pliegues_mejorados_vs_v03']}/5 pliegues). "
        f"Evidencia: `experiments/16_targeting_v2.py`, `reports/targeting_v2.json`, "
        f"`reports/figuras/fig_28_targeting_v2.png` |"
    )
    if "| P2 | 2026-09-25 |" not in texto:
        # La tabla de gates es la ultima seccion: la fila se inserta tras la de P1.
        lineas = texto.split("\n")
        indice = max(
            posicion
            for posicion, linea in enumerate(lineas)
            if linea.startswith("| P1 | 2026-09-25 |")
        )
        lineas.insert(indice + 1, fila_gate)
        texto = "\n".join(lineas)
    TODO_MD.write_text(texto, encoding="utf-8")
    log("tasks/todo.md actualizado (P2.1-P2.4 y Gate P2)")


def escribir_json(
    resumenes: dict[str, dict],
    mejores: dict[str, tuple[str, float]],
    decision: dict,
    diagnostico: list[dict[str, Any]],
    verificaciones: dict[str, Any],
    rebanada: dict[str, Any],
    modo_anotacion: str,
    origen_constantes: str,
    duracion: float,
) -> None:
    """Escribe el reporte JSON completo del experimento.

    Args:
        resumenes: resumen por clave.
        mejores: mejor clave y C por variante.
        decision: decision del gate.
        diagnostico: diagnosticos por pliegue.
        verificaciones: comprobaciones de replicacion.
        rebanada: informacion de la rebanada colectiva.
        modo_anotacion: modo de anotacion usado.
        origen_constantes: origen de las constantes estructurales.
        duracion: segundos totales.
    """
    reporte = {
        "fecha": CONFIG["fecha"],
        "protocolo": "v2 (D-016): grupos = near-duplicado UNION pseudo-autor, StratifiedGroupKFold(5)",
        "referencias": {
            "m2_umbral_05": CONFIG["referencia_m2_umbral_05"],
            "v03_umbrales": CONFIG["referencia_v03_umbrales"],
            "v03_por_etiqueta": CONFIG["v03_por_etiqueta"],
            "candidato_v2a_meta": CONFIG["referencia_v2a_meta"],
        },
        "modo_anotacion": modo_anotacion,
        "origen_constantes_estructurales": origen_constantes,
        "config": {clave: valor for clave, valor in CONFIG.items()},
        "verificaciones": verificaciones,
        "rebanada_colectiva_referencia": rebanada,
        "diagnostico_por_pliegue": diagnostico,
        "variantes": {
            variante: {
                "mejor_clave": mejores[variante][0],
                "mejor_C": mejores[variante][1],
                "por_C": {
                    f"C={valor_c}": round(resumenes[f"{variante}@C={valor_c}"]["corrected_cv"], 4)
                    for valor_c in CONFIG["C_variantes"]
                },
                "resumen": resumenes[mejores[variante][0]],
            }
            for variante in NOMBRES_VARIANTES
        },
        "base": resumenes["BASE@C=2.0"],
        "decision": decision,
        "duracion_segundos": round(duracion, 2),
        "artefactos": [
            str(REPORTE_CSV.relative_to(PROJECT_ROOT)),
            str(REPORTE_JSON.relative_to(PROJECT_ROOT)),
            str(FIGURA_PATH.relative_to(PROJECT_ROOT)),
            str(OOF_TARGETING_PATH.relative_to(PROJECT_ROOT)),
            str(CACHE_ANOTACIONES.relative_to(PROJECT_ROOT)),
        ],
    }
    REPORTE_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORTE_JSON.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"reporte JSON escrito: {REPORTE_JSON.name}")


def verificar_replicacion(
    oof: dict[tuple[str, float], NDArray[np.float64]],
    y_verdadera: NDArray[np.int_],
    ids: Sequence[int],
) -> dict[str, Any]:
    """Comprueba que BASE reproduce el OOF congelado y las referencias publicadas.

    Args:
        oof: probabilidades por clave.
        y_verdadera: etiquetas verdaderas.
        ids: identificadores del desarrollo en el mismo orden que las filas.

    Returns:
        Diccionario con las comprobaciones.

    Raises:
        RuntimeError: si los ids no coinciden o si BASE no reproduce el OOF
            congelado de M2.
    """
    congelado = np.load(OOF_BASE_PATH, allow_pickle=True)
    ids_congelados = [int(identificador) for identificador in congelado["ids"]]
    if ids_congelados != list(ids):
        raise RuntimeError("Los ids del desarrollo no coinciden con el OOF congelado")
    modelos = [str(nombre) for nombre in congelado["modelos"]]
    indice_m2 = modelos.index("M2_union_lr_c2")
    proba_m2 = congelado["proba"][:, indice_m2, :]
    diferencia = float(np.max(np.abs(proba_m2 - oof[("BASE", 2.0)])))
    if diferencia > CONFIG["tolerancia_replicacion_base"]:
        raise RuntimeError(
            f"BASE no reproduce el OOF congelado de M2 (max |diff| = {diferencia:.2e})"
        )
    f1_base = metricas_multietiqueta(y_verdadera, predecir(oof[("BASE", 2.0)], {e: 0.5 for e in ETIQUETAS}))
    f1_v03 = metricas_multietiqueta(
        y_verdadera, predecir(oof[("BASE", 2.0)], CONFIG["umbrales_v03"])
    )
    verificaciones = {
        "max_abs_diff_vs_oof_congelado": diferencia,
        "f1_base_0_5": f1_base["f1_media_columnas"],
        "delta_f1_base_vs_referencia": f1_base["f1_media_columnas"] - CONFIG["referencia_m2_umbral_05"],
        "f1_v03_umbrales": f1_v03["f1_media_columnas"],
        "delta_f1_v03_vs_referencia": f1_v03["f1_media_columnas"] - CONFIG["referencia_v03_umbrales"],
        "f1_v03_por_etiqueta": f1_v03["f1_por_etiqueta"],
        "delta_v03_por_etiqueta": {
            etiqueta: f1_v03["f1_por_etiqueta"][etiqueta] - CONFIG["v03_por_etiqueta"][etiqueta]
            for etiqueta in ETIQUETAS
        },
    }
    for nombre, valor, referencia in (
        ("f1_base_0_5", verificaciones["f1_base_0_5"], CONFIG["referencia_m2_umbral_05"]),
        ("f1_v03_umbrales", verificaciones["f1_v03_umbrales"], CONFIG["referencia_v03_umbrales"]),
    ):
        if abs(valor - referencia) > CONFIG["tolerancia_replicacion_f1"]:
            log(f"aviso: {nombre} = {valor:.4f} difiere de la referencia {referencia:.4f}")
    log(
        f"replicacion: BASE max |diff| vs OOF congelado = {diferencia:.2e}; "
        f"BASE 0.5 = {verificaciones['f1_base_0_5']:.4f}; V03 umbrales = {verificaciones['f1_v03_umbrales']:.4f}"
    )
    return verificaciones


def main() -> None:
    """Ejecuta P2 (E2-05/E2-06) y guarda todos los artefactos."""
    inicio = time.perf_counter()
    desarrollo, grupos, pliegues = cargar_desarrollo()
    verificacion_datos = verificar_datos(desarrollo, grupos)
    y_verdadera = desarrollo[list(ETIQUETAS)].to_numpy()

    constantes, origen_constantes = cargar_constantes_error_analysis()
    documentos, modo_anotacion = cargar_o_crear_anotaciones(desarrollo)
    if modo_anotacion == "heuristico":
        log("advertencia: sin POS de spaCy, PROPN por heuristica de mayusculas")
    mascara_colectiva = rebanada_colectiva_referencia(documentos, constantes)
    rebanada = {
        "n": int(mascara_colectiva.sum()),
        "n_ag_positivos": int(y_verdadera[mascara_colectiva, 2].sum()),
        "n_ag_positivos_fuera": int(y_verdadera[~mascara_colectiva, 2].sum()),
        "definicion": "regla estructural del script 06 (cuantificador colectivo o DET plural + NOUN/ADJ plural)",
    }
    log(f"rebanada colectiva de referencia: n={rebanada['n']} AG positivos={rebanada['n_ag_positivos']}")

    receta_v03 = cargar_receta_v03()
    resultado = evaluar_pliegues(desarrollo, documentos, y_verdadera, pliegues, constantes, receta_v03)
    verificaciones = verificar_replicacion(
        resultado.oof, y_verdadera, desarrollo["id"].astype(int).tolist()
    )

    sin_umbral = {etiqueta: 0.5 for etiqueta in ETIQUETAS}
    f1_pliegues_base = f1_por_pliegue(resultado.oof[("BASE", 2.0)], y_verdadera, resultado.pliegue_de_fila, sin_umbral)
    f1_pliegues_v03 = f1_por_pliegue(
        resultado.oof[("BASE", 2.0)], y_verdadera, resultado.pliegue_de_fila, CONFIG["umbrales_v03"]
    )
    resumenes = {}
    for clave, proba in resultado.oof.items():
        resumenes[f"{clave[0]}@C={clave[1]}"] = resumir_clave(
            clave, proba, y_verdadera, resultado.pliegue_de_fila, resultado.oof[("BASE", 2.0)],
            f1_pliegues_base, f1_pliegues_v03, mascara_colectiva,
        )
    mejores = {
        variante: max(
            ((clave, resumenes[clave]["corrected_cv"]) for clave in resumenes if resumenes[clave]["variante"] == variante),
            key=lambda par: par[1],
        )[0]
        for variante in NOMBRES_VARIANTES
    }
    mejores_por_variante = {variante: clave for variante, clave in mejores.items()}
    mejores_tuplas = {
        variante: (clave, resumenes[clave]["C"]) for variante, clave in mejores.items()
    }
    decision = decidir(resumenes)

    tabla = escribir_csv(resumenes)
    guardar_oof_targeting(desarrollo, y_verdadera, resultado.pliegue_de_fila, resultado.oof, mejores_tuplas)
    tamanos_medios = {
        clave: float(np.mean([fila["lexicos"][clave] for fila in resultado.diagnostico]))
        for clave in ("insulto", "adjetivos_negativos", "colectivos", "vocabulario_candidato")
    }
    generar_figura(
        resumenes,
        mejores_por_variante,
        {
            "V03 umbrales (slice)": resumenes["BASE@C=2.0"]["ag_recall_slice_v03"],
            f"{decision['mejor_clave']} (slice)": resumenes[decision["mejor_clave"]]["ag_recall_slice"],
            "V03 umbrales (fuera)": resumenes["BASE@C=2.0"]["ag_recall_fuera_slice_v03"],
            f"{decision['mejor_clave']} (fuera)": resumenes[decision["mejor_clave"]]["ag_recall_fuera_slice"],
        },
    )
    duracion = time.perf_counter() - inicio
    escribir_json(
        resumenes, mejores_tuplas, decision, resultado.diagnostico, verificaciones, rebanada,
        modo_anotacion, origen_constantes, duracion,
    )
    registrar_experimentos(resumenes, mejores_tuplas, decision, tamanos_medios, duracion)

    lineas_contexto = [
        (
            f"P2 (E2-05/E2-06) targeting estructural sobre el desarrollo (n={verificacion_datos['n_filas']}), "
            "pliegues corregidos v2, holdout congelado intacto."
        ),
        (
            f"BASE reproducido: corrected_cv {verificaciones['f1_base_0_5']:.4f} a 0.5 (max |diff| vs OOF congelado "
            f"{verificaciones['max_abs_diff_vs_oof_congelado']:.1e}) y V03 con umbrales "
            f"{verificaciones['f1_v03_umbrales']:.4f} (HS {verificaciones['f1_v03_por_etiqueta']['HS']:.3f} / "
            f"TR {verificaciones['f1_v03_por_etiqueta']['TR']:.3f} / AG {verificaciones['f1_v03_por_etiqueta']['AG']:.3f})."
        ),
        (
            f"Lexico de insulto inducido fold-local (log-odds con prior de Dirichlet, freq>={CONFIG['min_frecuencia_lexico']}, "
            f"df>={CONFIG['min_documentos_lexico']}; media {tamanos_medios['insulto']:.0f} terminos por pliegue), "
            f"adjetivos negativos {tamanos_medios['adjetivos_negativos']:.0f}, inventario colectivo "
            f"{tamanos_medios['colectivos']:.0f} (regla estructural + gentilicios por sufijo, sin etiquetas)."
        ),
        f"Anotacion: {modo_anotacion}; constantes estructurales: {origen_constantes}.",
    ]
    for variante in NOMBRES_VARIANTES:
        clave = mejores_tuplas[variante][0]
        resumen = resumenes[clave]
        lineas_contexto.append(
            f"{resumen['clave']}: corrected_cv {resumen['corrected_cv']:.4f} "
            f"(delta vs V03 {resumen['delta_vs_v03_umbrales']:+.4f}; vs v2-A {resumen['delta_vs_v2a_meta']:+.4f}; "
            f"pliegues {resumen['pliegues_mejorados_vs_v03']}/5; HS {resumen['f1_por_etiqueta']['HS']:.3f} / "
            f"TR {resumen['f1_por_etiqueta']['TR']:.3f} / AG {resumen['f1_por_etiqueta']['AG']:.3f}; "
            f"ensemble_gain {resumen.get('ensemble_gain', float('nan')):+.4f})."
        )
    conjunto_variantes = {
        resumenes[f"{variante}@C={valor_c}"]["ag_recall_slice"]: f"{variante}@C={valor_c}"
        for variante in NOMBRES_VARIANTES
        for valor_c in CONFIG["C_variantes"]
    }
    mejor_rebanada = max(conjunto_variantes)
    lineas_contexto.extend(
        [
            (
                f"Rebanada de referencia colectiva: n={rebanada['n']} (AG positivos {rebanada['n_ag_positivos']}); "
                f"recall AG V03 {resumenes['BASE@C=2.0']['ag_recall_slice_v03']:.3f} vs "
                f"{resumenes[decision['mejor_clave']]['ag_recall_slice']:.3f} de la mejor variante global; fuera de la "
                f"rebanada V03 {resumenes['BASE@C=2.0']['ag_recall_fuera_slice_v03']:.3f} vs "
                f"{resumenes[decision['mejor_clave']]['ag_recall_fuera_slice']:.3f} "
                f"(brecha dentro-fuera de V03 "
                f"{resumenes['BASE@C=2.0']['ag_recall_slice_v03'] - resumenes['BASE@C=2.0']['ag_recall_fuera_slice_v03']:+.3f}, "
                f"que confirma que el FN de AG se concentra en la referencia colectiva)."
            ),
            (
                f"Senal localizada para P6: el canal CH-TGT+DIST alcanza el mayor recall de AG en la rebanada "
                f"({mejor_rebanada:.3f} con {conjunto_variantes[mejor_rebanada]}) frente a 0.452 de V03, aunque su F1 "
                f"global es muy inferior; el promedio simple de probabilidades con M2 degrada (gain negativo), asi que "
                f"solo el stacking regularizado de P6 puede explotarlo. Sus probabilidades OOF quedan en "
                f"features/v2_oof_targeting.npz."
            ),
            (
                f"Gate P2: {'SUPERADO' if decision['adoptado'] else 'CERRADO SIN MEJORA'}. {decision['justificacion']} "
                f"Nivel exigido {decision['criterio_nivel']['objetivo']:.4f} "
                f"(cumple={decision['criterio_nivel']['cumple']}), pliegues "
                f"{decision['criterio_pliegues']['pliegues_mejorados']}/5 (cumple={decision['criterio_pliegues']['cumple']}), "
                f"complementariedad gain {decision['criterio_complementariedad']['ensemble_gain']:+.4f} "
                f"(cumple={decision['criterio_complementariedad']['cumple']}; IC bootstrap no aplica con gain <= 0)."
            ),
            (
                "Sin fuga: lexicos, escaladores, vectorizadores y modelos se ajustan solo con el pliegue "
                "de entrenamiento; umbral y seleccion de C sobre OOF del desarrollo."
            ),
            (
                f"Duracion: {duracion:.1f} s. Script: experiments/16_targeting_v2.py. Artefactos: reports/targeting_v2.csv "
                "y .json, reports/figuras/fig_28_targeting_v2.png, features/v2_oof_targeting.npz (canales para P6)."
            ),
        ]
    )
    escribir_contexto(lineas_contexto)
    actualizar_todo(resumenes, mejores_tuplas, decision)

    log("resumen de variantes (corrected_cv, delta vs V03, delta vs v2-A, pliegues mejorados):")
    for variante in NOMBRES_VARIANTES:
        resumen = resumenes[mejores_tuplas[variante][0]]
        log(
            f"  {resumen['clave']:<20} {resumen['corrected_cv']:.4f} "
            f"({resumen['delta_vs_v03_umbrales']:+.4f}, {resumen['delta_vs_v2a_meta']:+.4f}, "
            f"{resumen['pliegues_mejorados_vs_v03']}/5)"
        )
    log(f"decision: {'ADOPTADA' if decision['adoptado'] else 'CERRADA SIN MEJORA'} ({decision['mejor_clave']})")
    log(f"filas={len(tabla)} duracion={duracion:.1f} s")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - reporte explicito y salida con codigo 1
        traceback.print_exc()
        sys.exit(1)
