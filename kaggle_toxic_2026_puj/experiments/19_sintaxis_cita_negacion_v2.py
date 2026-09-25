"""experiments/19_sintaxis_cita_negacion_v2.py
Fecha: 2026-09-25
Descripcion: P3 (sintaxis sparse) y P4 (cita/atribucion/negacion) del plan v2,
evaluados sobre el desarrollo (n=1483) con los pliegues corregidos de D-016
(grupos fusionados = near-duplicado UNION pseudo-autor, StratifiedGroupKFold(5),
seed 42). El holdout congelado no se toca en ningun momento.
Estado del arte aplicado: P3.1-P3.3 del plan v2 (POS 1-4, arcos de dependencia
1-3, morfologia 1-2 como canales sparse simbolicos) y P4.1-P4.3 (canales
autor/citado, insulto/target dentro-fuera de cita y negacion sintactica),
con LR balanced de V03 (C en {1.0, 2.0}), lexico de insulto inducido
fold-local de P2 y la regla de adopcion pre-registrada de D-016.

Preguntas que responde
----------------------
P3 (E2-03/E2-04): aportan las representaciones sintacticas sparse (POS,
dependencias y morfologia, todas SIMBOLICAS via spaCy) senal que word/char de
V03 no capturan, ya sea solas o concatenadas?
P4 (E2-07/E2-08): aportan la separacion cita/atribucion (comillas, RT,
marcadores) y la negacion sintactica (neg + insulto/target), solas o
concatenadas a V03?

Metodo
------
1. Anotacion spaCy SIMBOLICA (POS, lema, morfologia, dep_, cabeza) cacheada en
   `features/v2_syntax_annotations.json.gz` (la cache de P2 no tiene
   dependencias, por eso se crea una propia). Prohibido el uso de vectores
   (`doc.vector`, `token.vector`, `doc.tensor`, similarity): todo es simbolico.
2. Canales sparse POR PLIEGUE (vectorizador y modelo ajustados solo con el
   entrenamiento del pliegue):
   - POS: secuencia de etiquetas POS -> TF-IDF (1,4).
   - DEPS: por cada arco (hijo, cabeza) se emiten los tokens
     `POS_HIJO_POS_CABEZA` (PRON_VERB, NOUN_ADJ) y `DEP_CABEZA_POS`
     (NSUBJ_VERB, NEG_VERB, AMOD_...); si la cabeza es la raiz se emite ademas
     `DEP_HEAD` (AMOD_HEAD) -> TF-IDF (1,3).
   - MORPH: por token, un token `Rasgo=Valor` por cada rasgo presente
     (Number/Gender/Mood/Tense/Person/Polarity) -> TF-IDF (1,2).
   - CITA: dos representaciones word(1,2) separadas (texto dentro de comillas y
     texto fuera) concatenadas + bloque numerico de cita/atribucion.
   - NEG: secuencia de patrones (`NEG_INSULT`, `NEG_VERB`, `NEG_TARGET`) ->
     TF-IDF (1,2) + bloque numerico de negacion.
   Nota de implementacion: `es_core_news_md` NO etiqueta `dep_ == "neg"` (usa
   `advmod` con `Polarity=Neg`), asi que un negador es `dep_ == "neg"` O
   `advmod` con polaridad negativa O lema del inventario de negacion de P2;
   la feature `neg_dep_tag` cuenta la etiqueta literal y queda en 0.
3. Lexico de insulto INDUCIDO fold-local con el mismo procedimiento de P2
   (`experiments/16_targeting_v2.py`, log-odds con prior de Dirichlet); los
   targets reutilizan la definicion estructural de P2 (mencion, PROPN, segunda
   persona, imperativo, colectivo/gentilicio).
4. Variantes con LR balanced (liblinear) y C en {1.0, 2.0}; todas se comparan
   contra la referencia v2 (V03 con umbrales = 0.7772) y contra la base M2 a 0.5
   (0.7718). Regla de adopcion D-016: nivel >= 0.7822 con >= 3/5 pliegues, o
   complementariedad (ensemble_gain >= 0.0015 con IC bootstrap > 0).

Expectativa declarada (plan v2): BAJA. Los canales sintacticos no aportaron al
meta y P2 ya mostro que la concatenacion simbolica degrada; el objetivo es
evaluar y documentar el cierre de linea con numeros.

Pruebas de robustez
-------------------
Texto vacio, solo URL, solo mencion, RT, comillas desbalanceadas y texto sin
insultos: ninguna feature nueva produce NaN ni infinito, y la extraccion no
depende del orden de las filas (se verifica permutando el desarrollo).

Salidas
-------
`reports/sintaxis_cita_negacion_v2.csv` y `.json`,
`reports/figuras/fig_30_sintaxis_cita_v2.png`,
`features/v2_oof_syntax.npz` (canales P3) y
`features/v2_oof_cita_negacion.npz` (canales P4), filas `E2-03-*`, `E2-04-*`,
`E2-07-*` y `E2-08-*` en `experiments.csv`, seccion
"## Fase v2 - P3+P4 (2026-09-25)" en `reports/contexto.md` y actualizacion de
`tasks/todo.md` (P3.1-P3.4, P4.1-P4.4 y sus gates).

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/19_sintaxis_cita_negacion_v2.py
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import re
import sys
import time
import traceback
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

from src.evaluation import ETIQUETAS, metricas_multietiqueta

IDS_PATH = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
OOF_BASE_PATH = PROJECT_ROOT / "features" / "v2_oof_base_corrected.npz"
CACHE_SINTAXIS = PROJECT_ROOT / "features" / "v2_syntax_annotations.json.gz"
OOF_SYNTAX_PATH = PROJECT_ROOT / "features" / "v2_oof_syntax.npz"
OOF_CITA_NEG_PATH = PROJECT_ROOT / "features" / "v2_oof_cita_negacion.npz"
REPORTE_CSV = PROJECT_ROOT / "reports" / "sintaxis_cita_negacion_v2.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "sintaxis_cita_negacion_v2.json"
FIGURA_PATH = PROJECT_ROOT / "reports" / "figuras" / "fig_30_sintaxis_cita_v2.png"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"
CONTEXTO_MD = PROJECT_ROOT / "reports" / "contexto.md"
TODO_MD = PROJECT_ROOT / "tasks" / "todo.md"


def _cargar_experimento_p2() -> Any:
    """Importa ``experiments/16_targeting_v2.py`` para reutilizar su logica.

    Se reutilizan: la carga del desarrollo y sus pliegues, la induccion
    fold-local del lexico de insulto (log-odds con prior de Dirichlet), la
    deteccion estructural de targets (P2), la receta V03 de word/char, el
    bootstrap pareado y las utilidades de umbrales.

    Returns:
        Modulo del experimento 16.

    Raises:
        ImportError: si el modulo no se puede cargar.
    """
    ruta = PROJECT_ROOT / "experiments" / "16_targeting_v2.py"
    especificacion = importlib.util.spec_from_file_location("experimento_16_p2", ruta)
    if especificacion is None or especificacion.loader is None:
        raise ImportError(f"No se pudo cargar {ruta}")
    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = modulo
    especificacion.loader.exec_module(modulo)
    return modulo


P2: Final[Any] = _cargar_experimento_p2()

CONFIG: Final[dict[str, Any]] = {
    "fecha": "2026-09-25",
    "referencia_m2_umbral_05": 0.7718,
    "referencia_v03_umbrales": 0.7772,
    "referencia_v2a_meta": 0.7853,
    "umbrales_v03": {"HS": 0.44, "TR": 0.51, "AG": 0.46},
    "v03_por_etiqueta": {"HS": 0.762, "TR": 0.816, "AG": 0.754},
    "C_variantes": (1.0, 2.0),
    "mejora_minima": 0.005,
    "ensemble_gain_minimo": 0.0015,
    "pliegues_minimos_mejorados": 3,
    "n_bootstrap": 2000,
    "modelo_spacy": "es_core_news_md",
    "batch_spacy": 64,
    "tolerancia_replicacion": 1e-3,
    "tolerancia_f1_replicacion": 1e-4,
}

# Vectorizadores simbolicos: vocabulario pequeno, min_df=1 (convencion del canal
# simbolico de P2), tokens separados por espacios.
PARAMS_POS: Final[dict[str, Any]] = {**P2.PARAMS_SIMBOLICO, "ngram_range": (1, 4)}
PARAMS_DEPS: Final[dict[str, Any]] = {**P2.PARAMS_SIMBOLICO, "ngram_range": (1, 3)}
PARAMS_MORPH: Final[dict[str, Any]] = {**P2.PARAMS_SIMBOLICO, "ngram_range": (1, 2)}
PARAMS_NEG: Final[dict[str, Any]] = {**P2.PARAMS_SIMBOLICO, "ngram_range": (1, 2)}
# Canal de cita dentro/fuera: misma representacion word(1,2) de V03 por separado.
PARAMS_CITA: Final[dict[str, Any]] = dict(P2.PARAMS_WORD)

FEATURES_MORFOLOGICAS: Final[tuple[str, ...]] = (
    "Number",
    "Gender",
    "Mood",
    "Tense",
    "Person",
    "Polarity",
)
MARCADORES_ATRIBUCION: Final[frozenset[str]] = frozenset(
    {
        "decir",
        "segun",
        "afirmar",
        "declarar",
        "comentar",
        "publicar",
        "acusar",
        "llamar",
    }
)
NOMBRES_CITA: Final[tuple[str, ...]] = (
    "cita_presente",
    "n_tramos_cita",
    "comillas_dobles",
    "comillas_simples",
    "comillas_angulares",
    "rt_presente",
    "atribucion_presente",
    "n_marcadores_atribucion",
    "frac_chars_en_cita",
    "insulto_dentro_de_cita",
    "insulto_fuera_de_cita",
    "target_dentro_de_cita",
    "target_fuera_de_cita",
)
NOMBRES_NEG: Final[tuple[str, ...]] = (
    "negacion_presente",
    "negacion_count",
    "neg_dep_tag",
    "insulto_bajo_negacion",
    "target_bajo_negacion",
    "neg_verbo",
)
PATRON_RT: Final[re.Pattern[str]] = re.compile(r"^\s*rt\s+@", re.IGNORECASE)
PATRON_TOKEN_RESPALDO: Final[re.Pattern[str]] = re.compile(
    r"@\w+|#\w+|[A-Za-z][A-Za-z0-9']*|\d+|[^\sA-Za-z0-9]"
)
# Comillas soportadas: rectas (las unicas presentes en el desarrollo ASCII) y
# tipograficas/angulares por robustez si el corpus se re-decodifica.
COMILLAS: Final[dict[str, str]] = {
    '"': '"',
    "'": "'",
    "«": "»",
    "‹": "›",
    "“": "”",
    "‘": "’",
}

VARIANTES: Final[dict[str, tuple[str, ...]]] = {
    "SYN-POS": ("pos",),
    "SYN-DEPS": ("deps",),
    "SYN-MORPH": ("morph",),
    "SYN-3": ("pos", "deps", "morph"),
    "TXT+POS": ("word", "char", "pos"),
    "TXT+DEPS": ("word", "char", "deps"),
    "TXT+MORPH": ("word", "char", "morph"),
    "TXT+SYN3": ("word", "char", "pos", "deps", "morph"),
    "CITA": ("cita_en", "cita_fuera", "num_cita"),
    "NEG": ("neg_pat", "num_neg"),
    "TXT+CITA": ("word", "char", "cita_en", "cita_fuera", "num_cita"),
    "TXT+NEG": ("word", "char", "neg_pat", "num_neg"),
    "TXT+CITA+NEG": (
        "word",
        "char",
        "cita_en",
        "cita_fuera",
        "num_cita",
        "neg_pat",
        "num_neg",
    ),
    "BASE": ("word", "char"),
}
VARIANTES_P3: Final[tuple[str, ...]] = (
    "SYN-POS",
    "SYN-DEPS",
    "SYN-MORPH",
    "SYN-3",
    "TXT+POS",
    "TXT+DEPS",
    "TXT+MORPH",
    "TXT+SYN3",
)
VARIANTES_P4: Final[tuple[str, ...]] = ("CITA", "NEG", "TXT+CITA", "TXT+NEG", "TXT+CITA+NEG")


def log(mensaje: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        mensaje: texto a imprimir.
    """
    print(f"[P3+P4] {mensaje}", flush=True)


# ===== SECCION 1: carga del desarrollo y pliegues corregidos =====


def cargar_datos() -> tuple[pd.DataFrame, Any, list[tuple[NDArray[np.int_], NDArray[np.int_]]]]:
    """Carga el desarrollo (holdout excluido) y sus pliegues corregidos.

    La logica se delega en ``experiments/16_targeting_v2.py`` para garantizar
    que la particion sea identica a la de P2.

    Returns:
        Tupla ``(desarrollo, grupos, pliegues)``.
    """
    desarrollo, grupos, pliegues = P2.cargar_desarrollo()
    tabla = P2.verificar_datos(desarrollo, grupos)
    log(f"desarrollo verificado: n={tabla['n_filas']} grupos={tabla['n_grupos']}")
    return desarrollo, grupos, pliegues


# ===== SECCION 2: anotacion sintactica simbolica (spaCy cacheada) =====


@dataclass(frozen=True)
class TokenSintaxis:
    """Token con informacion morfosintactica y de dependencias.

    Attributes:
        texto: forma original.
        lema: lema en minusculas sin diacriticos.
        pos: categoria POS de spaCy (``WORD`` en modo heuristico).
        numero: valor de ``Number`` (``Plur``, ``Sing`` o ``""``).
        persona: valor de ``Person`` (``1``, ``2``, ``3`` o ``""``).
        modo: valor de ``Mood`` (``Imp``, ``Ind``, ... o ``""``).
        genero: valor de ``Gender`` (``Fem``, ``Masc`` o ``""``).
        tiempo: valor de ``Tense`` (``Past``, ``Pres``, ... o ``""``).
        polaridad: valor de ``Polarity`` (``Neg`` o ``""``).
        oracion: indice de oracion dentro del documento.
        inicio: desplazamiento en caracteres.
        en_comillas: si el token cae dentro de un tramo entre comillas.
        dep: etiqueta de dependencia del token.
        head_idx: indice del token cabeza (incluye tokens SPACE).
        head_pos: POS de la cabeza.
        head_dep: etiqueta de dependencia de la cabeza.
    """

    texto: str
    lema: str
    pos: str
    numero: str
    persona: str
    modo: str
    genero: str
    tiempo: str
    polaridad: str
    oracion: int
    inicio: int
    en_comillas: bool
    dep: str
    head_idx: int
    head_pos: str
    head_dep: str


@dataclass(frozen=True)
class DocumentoSintaxis:
    """Documento con sus tokens sintacticos.

    Attributes:
        tokens: tokens en orden de aparicion.
        texto: texto original del documento.
    """

    tokens: tuple[TokenSintaxis, ...]
    texto: str


def tramos_comillas(texto: str) -> list[tuple[int, int]]:
    """Localiza los tramos entre comillas con un escaner de pila.

    Soporta comillas rectas dobles/simples y angulares/tipograficas. Las
    comillas sin pareja se ignoran (no producen tramo), de modo que un texto
    con comillas desbalanceadas no rompe la extraccion.

    Args:
        texto: texto del documento.

    Returns:
        Lista de ``(inicio, fin)`` ordenada por posicion de apertura.
    """
    pila: list[tuple[str, int]] = []
    tramos: list[tuple[int, int]] = []
    for posicion, caracter in enumerate(texto):
        if caracter not in COMILLAS:
            continue
        if pila and pila[-1][0] == caracter:
            _, inicio = pila.pop()
            tramos.append((inicio, posicion))
        elif not pila:
            pila.append((caracter, posicion))
        elif COMILLAS[pila[-1][0]] == caracter:
            _, inicio = pila.pop()
            tramos.append((inicio, posicion))
    return sorted(tramos)


def segmentar_cita(texto: str, tramos: Sequence[tuple[int, int]]) -> tuple[str, str]:
    """Separa el texto en el contenido citado y el texto propio.

    Args:
        texto: texto del documento.
        tramos: tramos de comillas devueltos por :func:`tramos_comillas`.

    Returns:
        Tupla ``(dentro, fuera)``; cada segmento se limpia y los vacios se
        descartan.
    """
    dentro: list[str] = []
    fuera: list[str] = []
    anterior = 0
    for inicio, fin in tramos:
        fuera.append(texto[anterior:inicio])
        dentro.append(texto[inicio + 1 : fin])
        anterior = fin + 1
    fuera.append(texto[anterior:])
    return (
        " ".join(segmento.strip() for segmento in dentro if segmento.strip()),
        " ".join(segmento.strip() for segmento in fuera if segmento.strip()),
    )


def _en_tramo(inicio: int, fin_token: int, tramos: Sequence[tuple[int, int]]) -> bool:
    """Indica si un intervalo de caracteres cae dentro de un tramo de comillas.

    Args:
        inicio: posicion inicial del token.
        fin_token: posicion final (exclusiva) del token.
        tramos: tramos de comillas.

    Returns:
        ``True`` si el token esta contenido en algun tramo.
    """
    return any(inicio > tramo_inicio and fin_token <= tramo_fin for tramo_inicio, tramo_fin in tramos)


def anotar_con_spacy(textos: Sequence[str]) -> list[DocumentoSintaxis]:
    """Anota los documentos con spaCy (POS, lema, morfologia, dependencias).

    spaCy se usa solo como anotador simbolico: no se accede a vectores ni a
    similitud. Cada token guarda el POS y la etiqueta de dependencia de su
    cabeza para construir los patrones de arco.

    Args:
        textos: textos a anotar.

    Returns:
        Lista de documentos anotados.
    """
    import spacy

    anotador = spacy.load(CONFIG["modelo_spacy"], exclude=["ner"])
    documentos = []
    for documento in anotador.pipe(list(textos), batch_size=CONFIG["batch_spacy"]):
        tramos = tramos_comillas(documento.text)
        tokens = []
        for token in documento:
            morfologia = token.morph
            tokens.append(
                TokenSintaxis(
                    texto=token.text,
                    lema=P2.normalizar_ascii(token.lemma_),
                    pos=token.pos_ or "X",
                    numero=(morfologia.get("Number") or [""])[0],
                    persona=(morfologia.get("Person") or [""])[0],
                    modo=(morfologia.get("Mood") or [""])[0],
                    genero=(morfologia.get("Gender") or [""])[0],
                    tiempo=(morfologia.get("Tense") or [""])[0],
                    polaridad=(morfologia.get("Polarity") or [""])[0],
                    oracion=token.sent.start,
                    inicio=int(token.idx),
                    en_comillas=_en_tramo(int(token.idx), int(token.idx) + len(token.text), tramos),
                    dep=token.dep_ or "",
                    head_idx=int(token.head.i),
                    head_pos=token.head.pos_ or "X",
                    head_dep=token.head.dep_ or "",
                )
            )
        documentos.append(DocumentoSintaxis(tokens=tuple(tokens), texto=documento.text))
    return documentos


def anotar_heuristico(textos: Sequence[str]) -> list[DocumentoSintaxis]:
    """Anotacion de respaldo sin spaCy (sin dependencias ni morfologia).

    Args:
        textos: textos a anotar.

    Returns:
        Lista de documentos anotados con POS heuristico (``PROPN``/``WORD``) y
        sin etiquetas de dependencia (los canales DEPS/NEG quedan vacios).
    """
    documentos = []
    for texto in textos:
        tramos = tramos_comillas(texto)
        tokens = []
        for coincidencia in PATRON_TOKEN_RESPALDO.finditer(texto):
            forma = coincidencia.group(0)
            lema = P2.normalizar_ascii(forma)
            if lema in P2.NEGACION:
                pos = "ADV"
            elif forma[:1].isupper() and coincidencia.start() > 0:
                pos = "PROPN"
            else:
                pos = "WORD"
            tokens.append(
                TokenSintaxis(
                    texto=forma,
                    lema=lema,
                    pos=pos,
                    numero="",
                    persona="",
                    modo="",
                    genero="",
                    tiempo="",
                    polaridad="",
                    oracion=0,
                    inicio=int(coincidencia.start()),
                    en_comillas=_en_tramo(
                        int(coincidencia.start()),
                        int(coincidencia.end()),
                        tramos,
                    ),
                    dep="",
                    head_idx=len(tokens),
                    head_pos=pos,
                    head_dep="",
                )
            )
        documentos.append(DocumentoSintaxis(tokens=tuple(tokens), texto=texto))
    return documentos


def anotar_textos(textos: Sequence[str]) -> tuple[list[DocumentoSintaxis], str]:
    """Anota textos con spaCy y cae al modo heuristico si no esta disponible.

    Args:
        textos: textos a anotar.

    Returns:
        Tupla ``(documentos, modo)``.
    """
    try:
        return anotar_con_spacy(textos), "spacy"
    except Exception as error:  # noqa: BLE001 - respaldo defensivo si spaCy falla
        log(f"aviso: spaCy no disponible ({error}); se usa la anotacion heuristica")
        return anotar_heuristico(textos), "heuristico"


def _serializar(documentos: Sequence[DocumentoSintaxis]) -> list[list[dict[str, Any]]]:
    """Serializa los documentos anotados a estructuras JSON.

    Args:
        documentos: documentos anotados.

    Returns:
        Estructura serializable (el texto original no se duplica).
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
                "g": token.genero,
                "te": token.tiempo,
                "po": token.polaridad,
                "o": token.oracion,
                "i": token.inicio,
                "c": int(token.en_comillas),
                "d": token.dep,
                "h": token.head_idx,
                "hp": token.head_pos,
                "hd": token.head_dep,
            }
            for token in documento.tokens
        ]
        for documento in documentos
    ]


def _deserializar(
    carga: list[list[dict[str, Any]]], textos: Sequence[str]
) -> list[DocumentoSintaxis]:
    """Reconstruye los documentos anotados desde la cache.

    Args:
        carga: estructura leida de la cache.
        textos: textos originales.

    Returns:
        Lista de documentos anotados.
    """
    documentos = []
    for fila, texto in zip(carga, textos, strict=True):
        tokens = tuple(
            TokenSintaxis(
                texto=registro["t"],
                lema=registro["l"],
                pos=registro["p"],
                numero=registro["n"],
                persona=registro["pe"],
                modo=registro["m"],
                genero=registro["g"],
                tiempo=registro["te"],
                polaridad=registro["po"],
                oracion=int(registro["o"]),
                inicio=int(registro["i"]),
                en_comillas=bool(registro["c"]),
                dep=registro["d"],
                head_idx=int(registro["h"]),
                head_pos=registro["hp"],
                head_dep=registro["hd"],
            )
            for registro in fila
        )
        documentos.append(DocumentoSintaxis(tokens=tokens, texto=texto))
    return documentos


def cargar_o_crear_anotaciones(
    desarrollo: pd.DataFrame,
) -> tuple[list[DocumentoSintaxis], str]:
    """Devuelve la anotacion sintactica, reutilizando la cache si es valida.

    Args:
        desarrollo: DataFrame del desarrollo.

    Returns:
        Tupla ``(documentos, modo)``.
    """
    ids = desarrollo["id"].astype(int).tolist()
    textos = desarrollo["text"].tolist()
    if CACHE_SINTAXIS.exists():
        with gzip.open(CACHE_SINTAXIS, "rt", encoding="utf-8") as archivo:
            carga = json.load(archivo)
        if carga.get("ids") == ids and carga.get("modo") == "spacy":
            documentos = _deserializar(carga["documentos"], textos)
            log(f"anotacion sintactica reutilizada de la cache ({len(documentos)} documentos)")
            return documentos, "spacy-cache"
        log("la cache sintactica no coincide con el desarrollo: se reanota")
    documentos, modo = anotar_textos(textos)
    if modo == "spacy":
        CACHE_SINTAXIS.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(CACHE_SINTAXIS, "wt", encoding="utf-8") as archivo:
            json.dump({"ids": ids, "modo": modo, "documentos": _serializar(documentos)}, archivo)
    log(f"anotacion completada: modo={modo} documentos={len(documentos)}")
    return documentos, modo


# ===== SECCION 3: features de cita/atribucion y de negacion =====


@dataclass(frozen=True)
class AnalisisCita:
    """Analisis de cita y atribucion de un documento.

    Attributes:
        dentro: texto citado (dentro de comillas).
        fuera: texto propio (fuera de comillas).
        numericos: bloque de features de cita en el orden de :data:`NOMBRES_CITA`.
    """

    dentro: str
    fuera: str
    numericos: NDArray[np.float64]


@dataclass(frozen=True)
class AnalisisNegacion:
    """Analisis de negacion sintactica de un documento.

    Attributes:
        patrones: secuencia de patrones (``NEG_INSULT``, ``NEG_VERB``,
            ``NEG_TARGET``) en orden de aparicion.
        numericos: bloque de features de negacion en el orden de
            :data:`NOMBRES_NEG`.
    """

    patrones: str
    numericos: NDArray[np.float64]


def posiciones_target(
    tokens: Sequence[TokenSintaxis], lexicos: Any, constantes: dict[str, Any]
) -> list[int]:
    """Detecta las posiciones de target con la definicion estructural de P2.

    Args:
        tokens: tokens del documento.
        lexicos: lexicos fold-locales inducidos por P2.
        constantes: constantes estructurales del script 06 (via P2).

    Returns:
        Posiciones ordenadas de los targets (mencion, PROPN, segunda persona,
        imperativo, colectivo o gentilicio).
    """
    posiciones = []
    for indice, token in enumerate(tokens):
        if token.pos == "SPACE":
            continue
        if (
            P2.es_mencion(token)
            or token.pos == "PROPN"
            or P2.es_segunda_persona(token, constantes)
            or P2.es_colectivo_estructural(tokens, indice, constantes)
            or (token.lema in lexicos.colectivos and token.numero == "Plur")
            or (token.pos in P2.POS_COLECTIVO and P2.es_gentilicio(token.lema))
        ):
            posiciones.append(indice)
    return posiciones


def analizar_cita(
    documento: DocumentoSintaxis, lexicos: Any, constantes: dict[str, Any]
) -> AnalisisCita:
    """Construye el analisis de cita/atribucion de un documento.

    Args:
        documento: documento anotado.
        lexicos: lexicos fold-locales inducidos por P2.
        constantes: constantes estructurales del script 06.

    Returns:
        Analisis con los textos dentro/fuera de comillas y el bloque numerico.
    """
    tokens = documento.tokens
    tramos = tramos_comillas(documento.texto)
    dentro, fuera = segmentar_cita(documento.texto, tramos)
    pos_insultos = [indice for indice, token in enumerate(tokens) if token.lema in lexicos.insulto]
    pos_targets = posiciones_target(tokens, lexicos, constantes)
    insultos_en_cita = sum(1 for indice in pos_insultos if tokens[indice].en_comillas)
    targets_en_cita = sum(1 for indice in pos_targets if tokens[indice].en_comillas)
    n_atribucion = sum(1 for token in tokens if token.lema in MARCADORES_ATRIBUCION)
    caracteres_en_cita = sum(fin - inicio - 1 for inicio, fin in tramos)
    numericos = np.array(
        [
            float(bool(tramos)),
            float(len(tramos)),
            float('"' in documento.texto),
            float("'" in documento.texto),
            float(any(caracter in documento.texto for caracter in "«»‹›")),
            float(bool(PATRON_RT.match(documento.texto))),
            float(n_atribucion > 0),
            float(n_atribucion),
            caracteres_en_cita / max(len(documento.texto), 1),
            float(insultos_en_cita),
            float(len(pos_insultos) - insultos_en_cita),
            float(targets_en_cita),
            float(len(pos_targets) - targets_en_cita),
        ],
        dtype=float,
    )
    if numericos.shape[0] != len(NOMBRES_CITA):
        raise RuntimeError(f"Features de cita mal alineadas: {numericos.shape[0]}")
    return AnalisisCita(dentro=dentro, fuera=fuera, numericos=numericos)


def posiciones_negacion(tokens: Sequence[TokenSintaxis]) -> list[int]:
    """Detecta los negadores del documento.

    El modelo ``es_core_news_md`` no emite la etiqueta ``dep_ == "neg"``
    (usa ``advmod`` con ``Polarity=Neg``), por lo que se aceptan las tres vias
    documentadas: ``dep_ == "neg"``, ``advmod`` con polaridad negativa, o lema
    del inventario de negacion de P2 con POS de particula/pronombre.

    Args:
        tokens: tokens del documento.

    Returns:
        Posiciones de los negadores.
    """
    posiciones = []
    for indice, token in enumerate(tokens):
        if token.pos == "SPACE":
            continue
        es_negador = (
            token.dep == "neg"
            or (token.dep == "advmod" and token.polaridad == "Neg")
            or (token.lema in P2.NEGACION and token.pos in ("ADV", "CCONJ", "ADP", "PRON"))
        )
        if es_negador:
            posiciones.append(indice)
    return posiciones


def alcance_negacion(tokens: Sequence[TokenSintaxis], pos_negacion: Sequence[int]) -> set[int]:
    """Calcula el alcance local de la negacion en el arbol de dependencias.

    Alcance de un negador: el propio negador, su cabeza sintactica y los hijos
    directos de ambos. Es una aproximacion de vecindad documentada (2 saltos
    maximos), suficiente para "no eres tonto" y "no te quiero".

    Args:
        tokens: tokens del documento.
        pos_negacion: posiciones de los negadores.

    Returns:
        Conjunto de posiciones bajo negacion.
    """
    hijos: dict[int, list[int]] = {}
    for indice, token in enumerate(tokens):
        if 0 <= token.head_idx < len(tokens):
            hijos.setdefault(token.head_idx, []).append(indice)
    alcance: set[int] = set()
    for indice in pos_negacion:
        cabeza = tokens[indice].head_idx
        cabeza = cabeza if 0 <= cabeza < len(tokens) else indice
        alcance.add(indice)
        alcance.add(cabeza)
        alcance.update(hijos.get(indice, []))
        alcance.update(hijos.get(cabeza, []))
    return alcance


def analizar_negacion(
    documento: DocumentoSintaxis, lexicos: Any, constantes: dict[str, Any]
) -> AnalisisNegacion:
    """Construye el analisis de negacion sintactica de un documento.

    Args:
        documento: documento anotado.
        lexicos: lexicos fold-locales inducidos por P2.
        constantes: constantes estructurales del script 06.

    Returns:
        Analisis con la secuencia de patrones y el bloque numerico.
    """
    tokens = documento.tokens
    pos_negacion = posiciones_negacion(tokens)
    alcance = alcance_negacion(tokens, pos_negacion)
    pos_insultos = {indice for indice, token in enumerate(tokens) if token.lema in lexicos.insulto}
    pos_targets = set(posiciones_target(tokens, lexicos, constantes))
    partes: list[str] = []
    for indice, token in enumerate(tokens):
        if token.pos == "SPACE" or indice in pos_negacion or indice not in alcance:
            continue
        if indice in pos_insultos:
            partes.append("NEG_INSULT")
        if indice in pos_targets:
            partes.append("NEG_TARGET")
        if token.pos in ("VERB", "AUX"):
            partes.append("NEG_VERB")
    numericos = np.array(
        [
            float(bool(pos_negacion)),
            float(len(pos_negacion)),
            float(sum(1 for indice in pos_negacion if tokens[indice].dep == "neg")),
            float(len([indice for indice in pos_insultos if indice in alcance])),
            float(len([indice for indice in pos_targets if indice in alcance])),
            float(
                sum(
                    1
                    for indice, token in enumerate(tokens)
                    if indice in alcance
                    and indice not in pos_negacion
                    and token.pos in ("VERB", "AUX")
                )
            ),
        ],
        dtype=float,
    )
    if numericos.shape[0] != len(NOMBRES_NEG):
        raise RuntimeError(f"Features de negacion mal alineadas: {numericos.shape[0]}")
    return AnalisisNegacion(patrones=" ".join(partes), numericos=numericos)


# ===== SECCION 4: canales sparse por pliegue =====


@dataclass(frozen=True)
class BloquesPliegue:
    """Bloques de representacion de un pliegue (entrenamiento y validacion).

    Attributes:
        entrenamiento: mapa ``nombre -> matriz CSR`` del entrenamiento.
        validacion: mapa ``nombre -> matriz CSR`` de la validacion.
        diagnostico: dimensiones y tamanos de lexico del pliegue.
    """

    entrenamiento: dict[str, sparse.csr_matrix]
    validacion: dict[str, sparse.csr_matrix]
    diagnostico: dict[str, Any]


def _matriz_tfidf(
    textos_entrenamiento: Sequence[str], textos_validacion: Sequence[str], params: dict[str, Any]
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Ajusta un TF-IDF con el entrenamiento y transforma la validacion.

    Args:
        textos_entrenamiento: textos del pliegue de entrenamiento.
        textos_validacion: textos del pliegue de validacion.
        params: parametros del vectorizador.

    Returns:
        Tupla ``(matriz_entrenamiento, matriz_validacion)``.
    """
    vectorizador = TfidfVectorizer(**params)
    return (
        vectorizador.fit_transform(textos_entrenamiento).tocsr(),
        vectorizador.transform(textos_validacion).tocsr(),
    )


def _bloque_numerico(
    numericos_entrenamiento: NDArray[np.float64], numericos_validacion: NDArray[np.float64]
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Estandariza un bloque numerico con el escalador ajustado en entrenamiento.

    Args:
        numericos_entrenamiento: matriz numerica del entrenamiento.
        numericos_validacion: matriz numerica de la validacion.

    Returns:
        Tupla de matrices CSR.
    """
    escalador = StandardScaler().fit(numericos_entrenamiento)
    return (
        sparse.csr_matrix(escalador.transform(numericos_entrenamiento)),
        sparse.csr_matrix(escalador.transform(numericos_validacion)),
    )


def secuencia_pos(documento: DocumentoSintaxis) -> str:
    """Construye la secuencia de etiquetas POS del documento.

    Args:
        documento: documento anotado.

    Returns:
        Etiquetas POS separadas por espacios (sin tokens SPACE).
    """
    return " ".join(token.pos for token in documento.tokens if token.pos != "SPACE")


def secuencia_deps(documento: DocumentoSintaxis) -> str:
    """Construye la secuencia de patrones de dependencia del documento.

    Por cada arco (hijo, cabeza) se emiten dos tokens: ``POS_hijo_POS_cabeza``
    (PRON_VERB, NOUN_ADJ) y ``DEP_cabeza_POS`` (NSUBJ_VERB, NEG_VERB,
    AMOD_NOUN); si la cabeza es la raiz de la oracion se emite ademas
    ``DEP_HEAD`` (AMOD_HEAD). Los tokens se emiten en orden de aparicion.

    Args:
        documento: documento anotado.

    Returns:
        Secuencia de patrones separada por espacios.
    """
    partes: list[str] = []
    for token in documento.tokens:
        if token.pos == "SPACE":
            continue
        dep = token.dep.upper() if token.dep else "DEP"
        partes.append(f"{token.pos}_{token.head_pos}")
        partes.append(f"{dep}_{token.head_pos}")
        if token.head_dep == "ROOT":
            partes.append(f"{dep}_HEAD")
    return " ".join(partes)


def secuencia_morph(documento: DocumentoSintaxis) -> str:
    """Construye la secuencia de rasgos morfologicos del documento.

    Args:
        documento: documento anotado.

    Returns:
        Tokens ``Rasgo=Valor`` de los rasgos presentes, separados por espacios.
    """
    partes: list[str] = []
    for token in documento.tokens:
        if token.pos == "SPACE":
            continue
        valores = {
            "Number": token.numero,
            "Gender": token.genero,
            "Mood": token.modo,
            "Tense": token.tiempo,
            "Person": token.persona,
            "Polarity": token.polaridad,
        }
        for rasgo in FEATURES_MORFOLOGICAS:
            valor = valores[rasgo]
            if valor:
                partes.append(f"{rasgo}={valor}")
    return " ".join(partes)


def preparar_pliegue(
    desarrollo: pd.DataFrame,
    documentos: Sequence[DocumentoSintaxis],
    y_verdadera: NDArray[np.int_],
    idx_entrenamiento: NDArray[np.int_],
    idx_validacion: NDArray[np.int_],
    constantes: dict[str, Any],
    receta_v03: Any,
) -> BloquesPliegue:
    """Construye todos los bloques de representacion de un pliegue.

    Los lexicos, vectorizadores y escaladores se ajustan SOLO con el
    entrenamiento del pliegue; la validacion unicamente se transforma.

    Args:
        desarrollo: DataFrame del desarrollo.
        documentos: documentos anotados del desarrollo.
        y_verdadera: etiquetas ``(n, 3)`` del desarrollo.
        idx_entrenamiento: posiciones del pliegue de entrenamiento.
        idx_validacion: posiciones del pliegue de validacion.
        constantes: constantes estructurales del script 06.
        receta_v03: modulo de ``experiments/14`` (representacion word/char).

    Returns:
        Bloques del pliegue.
    """
    lexicos = P2.inducir_lexicos(
        [documentos[indice] for indice in idx_entrenamiento], y_verdadera[idx_entrenamiento]
    )
    analisis_cita = [analizar_cita(documento, lexicos, constantes) for documento in documentos]
    analisis_neg = [analizar_negacion(documento, lexicos, constantes) for documento in documentos]

    textos = desarrollo["text"]
    word_ent, word_val = receta_v03.construir_matrices(
        textos.iloc[idx_entrenamiento], textos.iloc[idx_validacion], "word"
    )
    char_ent, char_val = receta_v03.construir_matrices(
        textos.iloc[idx_entrenamiento], textos.iloc[idx_validacion], "char"
    )
    pos_ent, pos_val = _matriz_tfidf(
        [secuencia_pos(documentos[indice]) for indice in idx_entrenamiento],
        [secuencia_pos(documentos[indice]) for indice in idx_validacion],
        PARAMS_POS,
    )
    deps_ent, deps_val = _matriz_tfidf(
        [secuencia_deps(documentos[indice]) for indice in idx_entrenamiento],
        [secuencia_deps(documentos[indice]) for indice in idx_validacion],
        PARAMS_DEPS,
    )
    morph_ent, morph_val = _matriz_tfidf(
        [secuencia_morph(documentos[indice]) for indice in idx_entrenamiento],
        [secuencia_morph(documentos[indice]) for indice in idx_validacion],
        PARAMS_MORPH,
    )
    cita_en_ent, cita_en_val = _matriz_tfidf(
        [analisis_cita[indice].dentro for indice in idx_entrenamiento],
        [analisis_cita[indice].dentro for indice in idx_validacion],
        PARAMS_CITA,
    )
    cita_fuera_ent, cita_fuera_val = _matriz_tfidf(
        [analisis_cita[indice].fuera for indice in idx_entrenamiento],
        [analisis_cita[indice].fuera for indice in idx_validacion],
        PARAMS_CITA,
    )
    neg_pat_ent, neg_pat_val = _matriz_tfidf(
        [analisis_neg[indice].patrones for indice in idx_entrenamiento],
        [analisis_neg[indice].patrones for indice in idx_validacion],
        PARAMS_NEG,
    )
    num_cita_ent, num_cita_val = _bloque_numerico(
        np.vstack([analisis_cita[indice].numericos for indice in idx_entrenamiento]),
        np.vstack([analisis_cita[indice].numericos for indice in idx_validacion]),
    )
    num_neg_ent, num_neg_val = _bloque_numerico(
        np.vstack([analisis_neg[indice].numericos for indice in idx_entrenamiento]),
        np.vstack([analisis_neg[indice].numericos for indice in idx_validacion]),
    )
    entrenamiento = {
        "word": word_ent,
        "char": char_ent,
        "pos": pos_ent,
        "deps": deps_ent,
        "morph": morph_ent,
        "cita_en": cita_en_ent,
        "cita_fuera": cita_fuera_ent,
        "neg_pat": neg_pat_ent,
        "num_cita": num_cita_ent,
        "num_neg": num_neg_ent,
    }
    validacion = {
        "word": word_val,
        "char": char_val,
        "pos": pos_val,
        "deps": deps_val,
        "morph": morph_val,
        "cita_en": cita_en_val,
        "cita_fuera": cita_fuera_val,
        "neg_pat": neg_pat_val,
        "num_cita": num_cita_val,
        "num_neg": num_neg_val,
    }
    diagnostico = {
        "lexico_insulto": len(lexicos.insulto),
        "dimensiones": {
            nombre: int(matriz.shape[1]) for nombre, matriz in entrenamiento.items()
        },
    }
    return BloquesPliegue(entrenamiento=entrenamiento, validacion=validacion, diagnostico=diagnostico)


def _concatenar(bloques: dict[str, sparse.csr_matrix], nombres: Sequence[str]) -> sparse.csr_matrix:
    """Concatena horizontalmente los bloques pedidos.

    Args:
        bloques: mapa de bloques del pliegue.
        nombres: nombres de los bloques a concatenar, en orden.

    Returns:
        Matriz CSR concatenada.
    """
    return sparse.hstack([bloques[nombre] for nombre in nombres], format="csr")


def _ajustar_lr(
    x_entrenamiento: sparse.csr_matrix,
    y_entrenamiento: NDArray[np.int_],
    x_prediccion: sparse.csr_matrix,
    valor_c: float,
) -> NDArray[np.float64]:
    """Ajusta una regresion logistica balanceada y devuelve probabilidades.

    Args:
        x_entrenamiento: matriz del pliegue de entrenamiento.
        y_entrenamiento: etiqueta binaria del pliegue de entrenamiento.
        x_prediccion: matriz a predecir.
        valor_c: regularizacion inversa.

    Returns:
        Probabilidades de la clase positiva.
    """
    modelo = LogisticRegression(C=valor_c, **P2.PARAMS_LR)
    modelo.fit(x_entrenamiento, y_entrenamiento)
    return modelo.predict_proba(x_prediccion)[:, 1]


# ===== SECCION 5: evaluacion out-of-fold, resumen y decision =====


@dataclass(frozen=True)
class ResultadoOOF:
    """Resultado de la evaluacion out-of-fold.

    Attributes:
        oof: mapa ``(variante, C) -> probabilidades (n, 3)``.
        pliegue_de_fila: numero de pliegue de cada fila.
        diagnostico: diagnosticos por pliegue.
    """

    oof: dict[tuple[str, float], NDArray[np.float64]]
    pliegue_de_fila: NDArray[np.int_]
    diagnostico: list[dict[str, Any]]


def evaluar_pliegues(
    desarrollo: pd.DataFrame,
    documentos: Sequence[DocumentoSintaxis],
    y_verdadera: NDArray[np.int_],
    pliegues: list[tuple[NDArray[np.int_], NDArray[np.int_]]],
    constantes: dict[str, Any],
    receta_v03: Any,
) -> ResultadoOOF:
    """Evalua todas las variantes fold a fold y devuelve sus OOF.

    Args:
        desarrollo: DataFrame del desarrollo.
        documentos: documentos anotados del desarrollo.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        pliegues: pliegues corregidos ``(entrenamiento, validacion)``.
        constantes: constantes estructurales del script 06.
        receta_v03: modulo de ``experiments/14``.

    Returns:
        Resultado con las probabilidades OOF por clave.
    """
    claves = [
        (variante, valor_c)
        for variante in VARIANTES
        if variante != "BASE"
        for valor_c in CONFIG["C_variantes"]
    ]
    claves.append(("BASE", 2.0))
    oof = {clave: np.zeros((len(desarrollo), len(ETIQUETAS)), dtype=float) for clave in claves}
    pliegue_de_fila = np.zeros(len(desarrollo), dtype=int)
    diagnostico = []
    for numero, (idx_entrenamiento, idx_validacion) in enumerate(pliegues, start=1):
        pliegue_de_fila[idx_validacion] = numero
        bloques = preparar_pliegue(
            desarrollo,
            documentos,
            y_verdadera,
            idx_entrenamiento,
            idx_validacion,
            constantes,
            receta_v03,
        )
        for variante, nombres in VARIANTES.items():
            x_entrenamiento = _concatenar(bloques.entrenamiento, nombres)
            x_validacion = _concatenar(bloques.validacion, nombres)
            valores_c = (2.0,) if variante == "BASE" else CONFIG["C_variantes"]
            for valor_c in valores_c:
                for indice_etiqueta in range(len(ETIQUETAS)):
                    oof[(variante, valor_c)][idx_validacion, indice_etiqueta] = _ajustar_lr(
                        x_entrenamiento,
                        y_verdadera[idx_entrenamiento, indice_etiqueta],
                        x_validacion,
                        valor_c,
                    )
        fila = dict(bloques.diagnostico)
        fila["pliegue"] = numero
        fila["n_entrenamiento"] = len(idx_entrenamiento)
        fila["n_validacion"] = len(idx_validacion)
        diagnostico.append(fila)
        log(
            f"pliegue {numero}: insulto={bloques.diagnostico['lexico_insulto']} "
            f"dim_word={bloques.diagnostico['dimensiones']['word']} "
            f"dim_char={bloques.diagnostico['dimensiones']['char']} "
            f"dim_pos={bloques.diagnostico['dimensiones']['pos']} "
            f"dim_deps={bloques.diagnostico['dimensiones']['deps']} "
            f"dim_morph={bloques.diagnostico['dimensiones']['morph']} "
            f"dim_neg={bloques.diagnostico['dimensiones']['neg_pat']}"
        )
    return ResultadoOOF(oof=oof, pliegue_de_fila=pliegue_de_fila, diagnostico=diagnostico)


def f1_pliegues_v03(
    proba_base: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
) -> list[float]:
    """Calcula la F1 por pliegue de la referencia V03 con sus umbrales.

    Args:
        proba_base: probabilidades OOF de la base M2.
        y_verdadera: etiquetas verdaderas.
        pliegue_de_fila: numero de pliegue de cada fila.

    Returns:
        Lista de F1 por pliegue.
    """
    return P2.f1_por_pliegue(proba_base, y_verdadera, pliegue_de_fila, CONFIG["umbrales_v03"])


def resumir_clave(
    clave: tuple[str, float],
    proba: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
    proba_base: NDArray[np.float64],
    f1_v03: list[float],
) -> dict[str, Any]:
    """Resume una clave (variante, C) frente a las referencias del protocolo v2.

    Args:
        clave: ``(variante, C)``.
        proba: probabilidades OOF de la clave.
        y_verdadera: etiquetas verdaderas.
        pliegue_de_fila: numero de pliegue de cada fila.
        proba_base: probabilidades OOF de la base M2 (V03 a 0.5).
        f1_v03: F1 por pliegue de V03 con sus umbrales.

    Returns:
        Diccionario con F1 global, por etiqueta, por pliegue, deltas y
        complementariedad.
    """
    variante, valor_c = clave
    sin_umbral = {etiqueta: 0.5 for etiqueta in ETIQUETAS}
    metrica = metricas_multietiqueta(y_verdadera, P2.predecir(proba, sin_umbral))
    f1_pliegues = P2.f1_por_pliegue(proba, y_verdadera, pliegue_de_fila, sin_umbral)
    resumen: dict[str, Any] = {
        "clave": f"{variante}@C={valor_c}",
        "variante": variante,
        "seccion": "P3" if variante in VARIANTES_P3 else "P4" if variante in VARIANTES_P4 else "BASE",
        "C": valor_c,
        "corrected_cv": metrica["f1_media_columnas"],
        "f1_por_etiqueta": metrica["f1_por_etiqueta"],
        "f1_pliegues": f1_pliegues,
        "std_pliegues": float(np.std(f1_pliegues)),
        "delta_vs_m2_05": metrica["f1_media_columnas"] - CONFIG["referencia_m2_umbral_05"],
        "delta_vs_v03_umbrales": metrica["f1_media_columnas"] - CONFIG["referencia_v03_umbrales"],
        "delta_vs_v2a_meta": metrica["f1_media_columnas"] - CONFIG["referencia_v2a_meta"],
        "pliegues_mejorados_vs_v03": int(np.sum(np.array(f1_pliegues) > np.array(f1_v03))),
    }
    if variante != "BASE":
        proba_ensemble = (proba + proba_base) / 2.0
        metrica_ensemble = metricas_multietiqueta(y_verdadera, P2.predecir(proba_ensemble, sin_umbral))
        ganancia = metrica_ensemble["f1_media_columnas"] - CONFIG["referencia_m2_umbral_05"]
        resumen["ensemble_gain"] = ganancia
        resumen["corrected_cv_ensemble"] = metrica_ensemble["f1_media_columnas"]
        if ganancia > 0:
            resumen["bootstrap"] = P2.bootstrap_delta(
                proba_base, proba_ensemble, y_verdadera, sin_umbral, sin_umbral
            )
    return resumen


def decidir_seccion(resumenes: dict[str, dict[str, Any]], variantes: Sequence[str]) -> dict[str, Any]:
    """Aplica la regla de adopcion pre-registrada en D-016 a una seccion.

    Args:
        resumenes: resumen por clave.
        variantes: variantes que pertenecen a la seccion.

    Returns:
        Diccionario con la mejor variante y la decision.
    """
    candidatos = {
        clave: resumen
        for clave, resumen in resumenes.items()
        if resumen["variante"] in variantes
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
            "exigida; la linea se cierra documentando el resultado."
        ),
    }


# ===== SECCION 6: pruebas de robustez de las features nuevas =====


def probar_robustez(
    documentos: Sequence[DocumentoSintaxis],
    lexicos: Any,
    constantes: dict[str, Any],
) -> dict[str, Any]:
    """Prueba las features nuevas ante casos limite y al orden del dataset.

    Args:
        documentos: documentos anotados del desarrollo.
        lexicos: lexicos fold-locales (de un pliegue; sin fuga de validacion).
        constantes: constantes estructurales del script 06.

    Returns:
        Diccionario con el resultado de cada prueba.

    Raises:
        AssertionError: si alguna feature produce NaN/infinito o si la
        extraccion depende del orden del dataset.
    """
    casos = {
        "texto_vacio": "",
        "solo_url": "https://t.co/AbCdEf1234",
        "solo_mencion": "@usuario",
        "rt": "RT @usuario: eres un tonto",
        "comillas_desbalanceadas": 'dijo "eres un tonto',
        "sin_insultos": "hoy hace sol en la ciudad y me voy a caminar",
        "control_cita_balanceada": 'dijo: "eres un tonto" y nada mas',
        "control_negacion": "no eres tonto y nunca lo seras",
    }
    documentos_casos, modo_casos = anotar_textos(list(casos.values()))
    resultado: dict[str, Any] = {"modo_anotacion": modo_casos, "casos": {}}
    for nombre, documento in zip(casos, documentos_casos, strict=True):
        cita = analizar_cita(documento, lexicos, constantes)
        negacion = analizar_negacion(documento, lexicos, constantes)
        finito = bool(np.isfinite(cita.numericos).all() and np.isfinite(negacion.numericos).all())
        if not finito:
            raise AssertionError(f"Features no finitas en el caso {nombre}")
        resultado["casos"][nombre] = {
            "finite": finito,
            "cita_presente": float(cita.numericos[0]),
            "tramos_cita": float(cita.numericos[1]),
            "texto_dentro_no_vacio": bool(cita.dentro),
            "negacion_count": float(negacion.numericos[1]),
            "patrones_neg": negacion.patrones,
        }
    if not any(caso["cita_presente"] > 0 for caso in resultado["casos"].values()):
        raise AssertionError("La senal de cita no se activa en ningun caso de prueba")
    if not any(caso["negacion_count"] > 0 for caso in resultado["casos"].values()):
        raise AssertionError("La senal de negacion no se activa en ningun caso de prueba")

    # Independencia del orden: permutar filas no debe cambiar las features.
    generador = np.random.default_rng(42)
    orden = generador.permutation(len(documentos))
    normal_cita = np.vstack([analizar_cita(documento, lexicos, constantes).numericos for documento in documentos])
    normal_neg = np.vstack([analizar_negacion(documento, lexicos, constantes).numericos for documento in documentos])
    perm_cita = np.vstack(
        [analizar_cita(documentos[indice], lexicos, constantes).numericos for indice in orden]
    )
    perm_neg = np.vstack(
        [analizar_negacion(documentos[indice], lexicos, constantes).numericos for indice in orden]
    )
    independiente = bool(
        np.array_equal(normal_cita[orden], perm_cita) and np.array_equal(normal_neg[orden], perm_neg)
    )
    if not independiente:
        raise AssertionError("La extraccion de features depende del orden del dataset")
    resultado["independiente_del_orden"] = independiente

    # Los vectorizadores transforman los casos limite sin NaN.
    textos_dentro = [analizar_cita(documento, lexicos, constantes).dentro for documento in documentos]
    textos_fuera = [analizar_cita(documento, lexicos, constantes).fuera for documento in documentos]
    patrones_neg = [analizar_negacion(documento, lexicos, constantes).patrones for documento in documentos]
    casos_dentro = [analizar_cita(documento, lexicos, constantes).dentro for documento in documentos_casos]
    casos_fuera = [analizar_cita(documento, lexicos, constantes).fuera for documento in documentos_casos]
    casos_patrones = [
        analizar_negacion(documento, lexicos, constantes).patrones for documento in documentos_casos
    ]
    finitos_vectorizador = True
    for params, textos, casos_texto in (
        (PARAMS_CITA, textos_dentro, casos_dentro),
        (PARAMS_CITA, textos_fuera, casos_fuera),
        (PARAMS_NEG, patrones_neg, casos_patrones),
    ):
        # min_df=1 en la prueba: solo interesa que transformar los casos limite
        # produzca valores finitos, no el vocabulario del pliegue.
        vectorizador = TfidfVectorizer(**{**params, "min_df": 1}).fit(textos)
        matriz = vectorizador.transform(casos_texto)
        finitos_vectorizador = finitos_vectorizador and bool(np.isfinite(matriz.data).all())
    if not finitos_vectorizador:
        raise AssertionError("Algun vectorizador produce valores no finitos en los casos limite")
    resultado["vectorizadores_finitos"] = finitos_vectorizador
    return resultado


# ===== SECCION 7: artefactos, registro y main =====


def verificar_replicacion(
    proba_base: NDArray[np.float64],
    oof_congelado: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
) -> dict[str, Any]:
    """Verifica que la base y V03 reproducen las referencias v2 congeladas.

    Args:
        proba_base: probabilidades OOF de la base calculadas en este script.
        oof_congelado: probabilidades de M2 del OOF congelado ``(n, 3)``.
        y_verdadera: etiquetas verdaderas.
        pliegue_de_fila: numero de pliegue de cada fila.

    Returns:
        Diccionario con las diferencias y F1 de referencia.
    """
    diferencia = float(np.max(np.abs(proba_base - oof_congelado)))
    f1_base = metricas_multietiqueta(y_verdadera, P2.predecir(proba_base, {e: 0.5 for e in ETIQUETAS}))[
        "f1_media_columnas"
    ]
    f1_v03 = metricas_multietiqueta(
        y_verdadera, P2.predecir(proba_base, CONFIG["umbrales_v03"])
    )["f1_media_columnas"]
    ok_diferencia = diferencia <= CONFIG["tolerancia_replicacion"]
    ok_f1_base = abs(f1_base - CONFIG["referencia_m2_umbral_05"]) <= CONFIG["tolerancia_f1_replicacion"]
    ok_f1_v03 = abs(f1_v03 - CONFIG["referencia_v03_umbrales"]) <= CONFIG["tolerancia_f1_replicacion"]
    if not (ok_diferencia and ok_f1_base and ok_f1_v03):
        raise RuntimeError(
            f"Replicacion fallida: dif={diferencia:.2e} base={f1_base:.4f} v03={f1_v03:.4f}"
        )
    log(
        f"replicacion OK: dif_max={diferencia:.2e} base={f1_base:.4f} v03={f1_v03:.4f} "
        f"(pliegues {pliegue_de_fila.min()}-{pliegue_de_fila.max()})"
    )
    return {
        "diferencia_maxima_oof_base": diferencia,
        "f1_base_umbral_05": f1_base,
        "f1_v03_umbrales": f1_v03,
        "ok": True,
    }


def guardar_oof(
    ruta: Path,
    desarrollo: pd.DataFrame,
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
    probabilidades: dict[str, NDArray[np.float64]],
    mejores: dict[str, str],
) -> None:
    """Guarda las probabilidades OOF de los canales de una seccion.

    Args:
        ruta: ruta del archivo ``.npz``.
        desarrollo: DataFrame del desarrollo.
        y_verdadera: etiquetas verdaderas.
        pliegue_de_fila: pliegue de cada fila.
        probabilidades: mapa ``clave -> probabilidades (n, 3)``.
        mejores: clave ganadora por variante.
    """
    variantes = list(mejores)
    valores_c = [
        2.0 if "@C=" not in mejores[variante] else float(mejores[variante].split("@C=")[1])
        for variante in variantes
    ]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ruta,
        ids=desarrollo["id"].astype(int).to_numpy(),
        y_true=y_verdadera,
        pliegue=pliegue_de_fila,
        proba=np.stack([probabilidades[mejores[variante]] for variante in variantes], axis=1),
        variantes=np.array(variantes),
        C=np.array(valores_c),
        etiquetas=np.array(list(ETIQUETAS)),
    )
    log(f"OOF guardado en {ruta.name} (variantes {variantes})")


def generar_figura(mejores: dict[str, str], resumenes: dict[str, dict[str, Any]]) -> None:
    """Genera la figura comparativa de P3 y P4 frente a las referencias.

    Args:
        mejores: clave ganadora por variante.
        resumenes: resumen por clave.
    """
    figura, (eje_p3, eje_p4) = plt.subplots(1, 2, figsize=(15, 6))
    base_etiqueta = f"BASE (M2, {CONFIG['referencia_m2_umbral_05']:.4f})"
    for eje, variantes, titulo in (
        (eje_p3, VARIANTES_P3, "P3 - Sintaxis sparse (POS / deps / morfologia)"),
        (eje_p4, VARIANTES_P4, "P4 - Cita / atribucion / negacion"),
    ):
        nombres = [resumenes[mejores[variante]]["clave"] for variante in variantes]
        valores = [resumenes[mejores[variante]]["corrected_cv"] for variante in variantes]
        nombres.append(base_etiqueta)
        valores.append(resumenes["BASE@C=2.0"]["corrected_cv"])
        colores = ["#4c72b0"] * (len(nombres) - 1) + ["#8c8c8c"]
        posiciones = list(range(len(nombres)))
        eje.barh(posiciones, valores, color=colores)
        eje.axvline(
            CONFIG["referencia_v03_umbrales"],
            color="#2ca02c",
            linestyle="-.",
            linewidth=1.1,
            label=f"V03 umbrales ({CONFIG['referencia_v03_umbrales']:.4f})",
        )
        eje.axvline(
            CONFIG["referencia_v03_umbrales"] + CONFIG["mejora_minima"],
            color="#d62728",
            linestyle="--",
            linewidth=1.1,
            label=f"nivel exigido ({CONFIG['referencia_v03_umbrales'] + CONFIG['mejora_minima']:.4f})",
        )
        eje.set_yticks(posiciones, nombres, fontsize=8)
        eje.set_xlabel("corrected_cv (F1 media por columna, OOF, umbral 0.5)")
        eje.set_title(titulo)
        eje.legend(loc="lower right", fontsize=7)
        for posicion, valor in zip(posiciones, valores, strict=True):
            eje.text(valor + 0.0004, posicion, f"{valor:.4f}", va="center", fontsize=7)
        eje.set_xlim(min(valores) - 0.008, max(valores) + 0.008)
    figura.tight_layout()
    FIGURA_PATH.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(FIGURA_PATH, dpi=150)
    plt.close(figura)
    log(f"figura escrita: {FIGURA_PATH.name}")


def escribir_csv(resumenes: dict[str, dict[str, Any]]) -> None:
    """Escribe el CSV comparativo de todas las claves evaluadas.

    Args:
        resumenes: resumen por clave.
    """
    filas = []
    for resumen in resumenes.values():
        bootstrap = resumen.get("bootstrap") or {}
        filas.append(
            {
                "clave": resumen["clave"],
                "seccion": resumen["seccion"],
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
                "ensemble_gain": round(resumen.get("ensemble_gain", float("nan")), 4),
                "corrected_cv_ensemble": round(resumen.get("corrected_cv_ensemble", float("nan")), 4),
                "bootstrap_ic": (
                    f"[{bootstrap.get('ic_inferior'):.4f},{bootstrap.get('ic_superior'):.4f}]"
                    if bootstrap.get("ic_inferior") is not None
                    else ""
                ),
            }
        )
    tabla = pd.DataFrame(filas)
    REPORTE_CSV.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(REPORTE_CSV, index=False)
    log(f"CSV escrito: {REPORTE_CSV.name} ({len(tabla)} filas)")


DEFINICIONES_EXPERIMENTO: Final[dict[str, tuple[str, str, str]]] = {
    "SYN-POS": (
        "E2-03-POS",
        "Aporta la secuencia POS (n-gramas 1-4) como TF-IDF sparse por si sola?",
        "POS n-gramas como TF-IDF aportan estructura gramatical que word/char no capturan (E2-03)",
    ),
    "TXT+POS": (
        "E2-03-POS-TXT",
        "Aporta la secuencia POS sobre la representacion word+char de V03?",
        "POS n-gramas como TF-IDF aportan estructura gramatical que word/char no capturan (E2-03)",
    ),
    "SYN-DEPS": (
        "E2-04-DEPS",
        "Aportan los patrones de arco (POS/DEP) como tokens simbolicos solos?",
        "Los patrones de dependencia (PRON_VERB, NEG_VERB, AMOD_HEAD) son senal complementaria (E2-04, H12)",
    ),
    "SYN-MORPH": (
        "E2-04-MORPH",
        "Aporta la morfologia (numero/genero/tiempo/modo/persona) como tokens sparse sola?",
        "La morfologia de persona/tiempo/modo distingue agresion (E2-04, H12)",
    ),
    "SYN-3": (
        "E2-04-SYN3",
        "Aporta la union POS+DEPS+MORPH como representacion sintactica sola?",
        "La sintaxis completa aporta estructura que word/char no capturan (E2-04, H12)",
    ),
    "TXT+DEPS": (
        "E2-04-DEPS-TXT",
        "Aportan los patrones de arco sobre la representacion word+char de V03?",
        "Los patrones de dependencia (PRON_VERB, NEG_VERB, AMOD_HEAD) son senal complementaria (E2-04, H12)",
    ),
    "TXT+MORPH": (
        "E2-04-MORPH-TXT",
        "Aporta la morfologia sparse sobre la representacion word+char de V03?",
        "La morfologia de persona/tiempo/modo distingue agresion (E2-04, H12)",
    ),
    "TXT+SYN3": (
        "E2-04-SYN3-TXT",
        "Aporta la sintaxis completa (POS+DEPS+MORPH) sobre la representacion word+char?",
        "La sintaxis completa aporta estructura que word/char no capturan (E2-04, H12)",
    ),
    "CITA": (
        "E2-07-CITA",
        "Aporta el canal cita/atribucion (dentro vs fuera de comillas + RT + marcadores) solo?",
        "Separar autor de citado mejora HS/AG en regimen noticia (E2-07, H13)",
    ),
    "TXT+CITA": (
        "E2-07-CITA-TXT",
        "Aporta el canal cita/atribucion sobre la representacion word+char de V03?",
        "Separar autor de citado mejora HS/AG en regimen noticia (E2-07, H13)",
    ),
    "TXT+CITA+NEG": (
        "E2-07-CITA-NEG-TXT",
        "Aportan cita+negacion juntas sobre la representacion word+char de V03?",
        "Cita y negacion juntas explican el regimen de reporte y la inversion de polaridad (E2-07/E2-08)",
    ),
    "NEG": (
        "E2-08-NEG",
        "Aporta la negacion sintactica (patrones NEG_* + conteos) por si sola?",
        "La negacion cambia la etiqueta ('no es X' vs 'es X') (E2-08)",
    ),
    "TXT+NEG": (
        "E2-08-NEG-TXT",
        "Aporta la negacion sintactica sobre la representacion word+char de V03?",
        "La negacion cambia la etiqueta ('no es X' vs 'es X') (E2-08)",
    ),
}


def registrar_experimentos(
    resumenes: dict[str, dict[str, Any]],
    mejores: dict[str, str],
    decisiones: dict[str, dict[str, Any]],
    duracion: float,
) -> None:
    """Anade las filas E2-03/E2-04/E2-07/E2-08 a ``experiments.csv``.

    Args:
        resumenes: resumen por clave.
        mejores: clave ganadora por variante.
        decisiones: decision por seccion (``P3``, ``P4``).
        duracion: segundos totales del experimento.
    """
    filas = []
    for variante, (identificador, pregunta, hipotesis) in DEFINICIONES_EXPERIMENTO.items():
        clave = mejores[variante]
        resumen = resumenes[clave]
        decision = decisiones["P3" if resumen["seccion"] == "P3" else "P4"]
        filas.append(
            {
                "experiment_id": identificador,
                "competition": "KAGGLE_TOXIC_2026_PUJ",
                "question": pregunta,
                "hypothesis": hipotesis,
                "preprocessing": (
                    "texto crudo + anotacion spaCy SIMBOLICA (POS, lema, morfologia, dep y cabeza) "
                    "cacheada; sin vectores ni similitud"
                ),
                "feature_groups": (
                    ("SYNTAX_POS" if ("POS" in variante or variante == "SYN-3" or variante.endswith("SYN3")) else "")
                    + ("+SYNTAX_DEPENDENCY" if ("DEPS" in variante or variante == "SYN-3" or variante.endswith("SYN3")) else "")
                    + ("+SYNTAX_MORPHOLOGY" if ("MORPH" in variante or variante == "SYN-3" or variante.endswith("SYN3")) else "")
                    + ("+QUOTE_ATTRIBUTION" if "CITA" in variante else "")
                    + ("+NEGATION_PATTERN" if "NEG" in variante else "")
                ),
                "representation": (
                    ("word(1,2)+char_wb(3,5) de V03 + " if variante.startswith("TXT+") else "")
                    + {
                        "SYN-POS": "TF-IDF POS (1,4)",
                        "SYN-DEPS": "TF-IDF arcos POS/DEP (1,3)",
                        "SYN-MORPH": "TF-IDF morfologia (1,2)",
                        "SYN-3": "TF-IDF POS (1,4) + arcos (1,3) + morfologia (1,2)",
                        "TXT+POS": "TF-IDF POS (1,4)",
                        "TXT+DEPS": "TF-IDF arcos POS/DEP (1,3)",
                        "TXT+MORPH": "TF-IDF morfologia (1,2)",
                        "TXT+SYN3": "TF-IDF POS (1,4) + arcos (1,3) + morfologia (1,2)",
                        "CITA": "TF-IDF word(1,2) dentro y fuera de comillas + numericas estandarizadas",
                        "TXT+CITA": "TF-IDF word(1,2) dentro y fuera de comillas + numericas estandarizadas",
                        "NEG": "TF-IDF patrones NEG_* (1,2) + numericas estandarizadas",
                        "TXT+NEG": "TF-IDF patrones NEG_* (1,2) + numericas estandarizadas",
                        "TXT+CITA+NEG": "TF-IDF word(1,2) dentro/fuera de comillas + patrones NEG_* (1,2) + numericas",
                    }[variante]
                ),
                "model": f"LogisticRegression C={resumen['C']} balanced",
                "hyperparameters": json.dumps({"C": resumen["C"], **P2.PARAMS_LR}, ensure_ascii=False),
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
                        "ensemble_gain": resumen.get("ensemble_gain"),
                        "bootstrap": resumen.get("bootstrap"),
                        "por_C": {
                            f"C={valor_c}": round(
                                resumenes[f"{variante}@C={valor_c}"]["corrected_cv"], 4
                            )
                            for valor_c in CONFIG["C_variantes"]
                        },
                    },
                    ensure_ascii=False,
                ),
                "interpretation": (
                    f"Mejor C={resumen['C']} elegido por corrected_cv interno (cota optimista de "
                    f"seleccion); delta vs V03 {resumen['delta_vs_v03_umbrales']:+.4f} y vs v2-A "
                    f"{resumen['delta_vs_v2a_meta']:+.4f}"
                ),
                "decision": (
                    f"ADOPTADA: {decision['justificacion']}"
                    if resumen["clave"] == decision["mejor_clave"] and decision["adoptado"]
                    else (
                        f"NO adoptada: {resumen['delta_vs_v03_umbrales']:+.4f} vs V03 "
                        f"({resumen['pliegues_mejorados_vs_v03']}/5 pliegues); linea de sintaxis/cita/"
                        "negacion cerrada sin mejora"
                    )
                ),
                "date": CONFIG["fecha"],
                "script": "experiments/19_sintaxis_cita_negacion_v2.py",
                "notes": (
                    "holdout congelado no utilizado; sin fuga (lexicos, vectorizadores y escaladores "
                    "solo del pliegue de entrenamiento)"
                ),
            }
        )
    existente = pd.read_csv(EXPERIMENTS_CSV)
    identificadores = {fila["experiment_id"] for fila in filas}
    existente = existente[~existente["experiment_id"].isin(identificadores)]
    existente.to_csv(EXPERIMENTS_CSV, index=False)
    pd.DataFrame(filas).to_csv(EXPERIMENTS_CSV, mode="a", header=False, index=False)
    log(f"filas registradas: {sorted(identificadores)}")


def escribir_contexto(lineas: list[str]) -> None:
    """Reemplaza o anade la seccion de P3+P4 en ``reports/contexto.md``.

    Args:
        lineas: lineas de la seccion (sin encabezado); maximo 20.
    """
    if len(lineas) > 20:
        raise ValueError("La seccion de contexto no puede superar 20 lineas")
    encabezado = "## Fase v2 - P3+P4 (2026-09-25)"
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
    log("seccion P3+P4 escrita en reports/contexto.md")


def actualizar_todo(
    resumenes: dict[str, dict[str, Any]],
    mejores: dict[str, str],
    decisiones: dict[str, dict[str, Any]],
) -> None:
    """Actualiza P3.1-P3.4, P4.1-P4.4 y los gates P3/P4 en ``tasks/todo.md``.

    Args:
        resumenes: resumen por clave.
        mejores: clave ganadora por variante.
        decisiones: decision por seccion.
    """

    def linea(variante: str) -> str:
        """Devuelve el texto resumido de la mejor clave de una variante."""
        resumen = resumenes[mejores[variante]]
        return (
            f"{resumen['clave']} corrected_cv {resumen['corrected_cv']:.4f} "
            f"(delta {resumen['delta_vs_v03_umbrales']:+.4f} vs V03, "
            f"pliegues {resumen['pliegues_mejorados_vs_v03']}/5)"
        )

    reemplazos = [
        (
            "- [pendiente] P3.1 POS n-gramas sparse (1-4) como CountVectorizer/TF-IDF",
            (
                "- [hecho] P3.1 POS n-gramas sparse (1-4) como CountVectorizer/TF-IDF: "
                f"SYN-POS {linea('SYN-POS')}; TXT+POS {linea('TXT+POS')}"
            ),
        ),
        (
            "- [pendiente] P3.2 Dependencias como tokens simbolicos (PRON_VERB, NEG_VERB, TARGET_ADJ)",
            (
                "- [hecho] P3.2 Dependencias como tokens simbolicos (PRON_VERB, NEG_VERB, "
                f"AMOD_HEAD): SYN-DEPS {linea('SYN-DEPS')}; TXT+DEPS {linea('TXT+DEPS')}"
            ),
        ),
        (
            "- [pendiente] P3.3 Morfologia (numero/genero/tiempo/modo/persona) sparse",
            (
                "- [hecho] P3.3 Morfologia (numero/genero/tiempo/modo/persona) sparse: "
                f"SYN-MORPH {linea('SYN-MORPH')}; TXT+MORPH {linea('TXT+MORPH')}; "
                f"SYN-3 {linea('SYN-3')}"
            ),
        ),
        (
            "- [pendiente] P3.4 Gate P3",
            (
                f"- [hecho] P3.4 Gate P3: {'ADOPTADO' if decisiones['P3']['adoptado'] else 'CERRADO sin mejora'} "
                f"({decisiones['P3']['mejor_clave']}, corrected_cv {decisiones['P3']['corrected_cv']:.4f})"
            ),
        ),
        (
            "- [pendiente] P4.1 Canales autor vs citado (comillas, RT, segun/dijo/afirmo)",
            (
                "- [hecho] P4.1 Canales autor vs citado (comillas, RT, segun/dijo/afirmo): "
                f"CITA {linea('CITA')}; TXT+CITA {linea('TXT+CITA')}"
            ),
        ),
        (
            "- [pendiente] P4.2 Insulto dentro/fuera de cita; target dentro/fuera",
            (
                "- [hecho] P4.2 Insulto dentro/fuera de cita; target dentro/fuera: features del canal "
                f"CITA (lexico inductivo fold-local); TXT+CITA {linea('TXT+CITA')}"
            ),
        ),
        (
            "- [pendiente] P4.3 Negacion sintactica (neg + insulto, neg + target)",
            (
                "- [hecho] P4.3 Negacion sintactica (neg + insulto, neg + target): "
                f"NEG {linea('NEG')}; TXT+NEG {linea('TXT+NEG')}; "
                f"TXT+CITA+NEG {linea('TXT+CITA+NEG')}"
            ),
        ),
        (
            "- [pendiente] P4.4 Gate P4",
            (
                f"- [hecho] P4.4 Gate P4: {'ADOPTADO' if decisiones['P4']['adoptado'] else 'CERRADO sin mejora'} "
                f"({decisiones['P4']['mejor_clave']}, corrected_cv {decisiones['P4']['corrected_cv']:.4f})"
            ),
        ),
    ]
    texto = TODO_MD.read_text(encoding="utf-8")
    for viejo, nuevo in reemplazos:
        if viejo in texto:
            texto = texto.replace(viejo, nuevo)
    filas_gate = []
    for seccion, evidencia in (
        ("P3", "`experiments/19_sintaxis_cita_negacion_v2.py`, `reports/sintaxis_cita_negacion_v2.json`"),
        ("P4", "`experiments/19_sintaxis_cita_negacion_v2.py`, `reports/sintaxis_cita_negacion_v2.json`"),
    ):
        decision = decisiones[seccion]
        mejor = resumenes[decision["mejor_clave"]]
        filas_gate.append(
            f"| {seccion} | 2026-09-25 | "
            f"{'SUPERADO' if decision['adoptado'] else 'CERRADO (sin mejora)'}. "
            f"{seccion} sintaxis/cita/negacion: mejor variante {decision['mejor_clave']} con "
            f"corrected_cv {decision['corrected_cv']:.4f} "
            f"({decision['corrected_cv'] - CONFIG['referencia_v03_umbrales']:+.4f} vs V03 0.7772, "
            f"{mejor['pliegues_mejorados_vs_v03']}/5 pliegues, ensemble_gain "
            f"{mejor.get('ensemble_gain', float('nan')):+.4f}). Evidencia: {evidencia} |"
        )
    for indice, fila_gate in enumerate(filas_gate):
        seccion = ("P3", "P4")[indice]
        if f"| {seccion} | 2026-09-25 |" not in texto:
            ancla = "| P2 | 2026-09-25 |" if seccion == "P3" else "| P3 | 2026-09-25 |"
            lineas = texto.split("\n")
            posicion = max(
                posicion
                for posicion, linea in enumerate(lineas)
                if linea.startswith(ancla)
            )
            lineas.insert(posicion + 1, fila_gate)
            texto = "\n".join(lineas)
    TODO_MD.write_text(texto, encoding="utf-8")
    log("tasks/todo.md actualizado (P3.1-P3.4, P4.1-P4.4 y gates)")


def escribir_json(
    resumenes: dict[str, dict[str, Any]],
    mejores: dict[str, str],
    decisiones: dict[str, dict[str, Any]],
    verificaciones: dict[str, Any],
    diagnostico: list[dict[str, Any]],
    robustez: dict[str, Any],
    modo_anotacion: str,
    duracion: float,
) -> None:
    """Escribe el reporte JSON completo del experimento.

    Args:
        resumenes: resumen por clave.
        mejores: clave ganadora por variante.
        decisiones: decision por seccion.
        verificaciones: comprobaciones de replicacion.
        diagnostico: diagnosticos por pliegue.
        robustez: resultado de las pruebas de robustez.
        modo_anotacion: modo de anotacion usado.
        duracion: segundos totales.
    """
    variantes = list(DEFINICIONES_EXPERIMENTO) + ["BASE"]
    reporte = {
        "fecha": CONFIG["fecha"],
        "protocolo": "v2 (D-016): grupos = near-duplicado UNION pseudo-autor, StratifiedGroupKFold(5)",
        "referencias": {
            "m2_umbral_05": CONFIG["referencia_m2_umbral_05"],
            "v03_umbrales": CONFIG["referencia_v03_umbrales"],
            "v03_por_etiqueta": CONFIG["v03_por_etiqueta"],
            "candidato_v2a_meta": CONFIG["referencia_v2a_meta"],
            "nivel_exigido": CONFIG["referencia_v03_umbrales"] + CONFIG["mejora_minima"],
        },
        "modo_anotacion": modo_anotacion,
        "verificaciones": verificaciones,
        "robustez": robustez,
        "config": {clave: valor for clave, valor in CONFIG.items()},
        "diagnostico_por_pliegue": diagnostico,
        "variantes": {
            variante: {
                "mejor_clave": mejores[variante],
                "mejor_C": resumenes[mejores[variante]]["C"],
                "por_C": (
                    {
                        f"C={valor_c}": round(resumenes[f"{variante}@C={valor_c}"]["corrected_cv"], 4)
                        for valor_c in CONFIG["C_variantes"]
                    }
                    if variante != "BASE"
                    else {"C=2.0": round(resumenes["BASE@C=2.0"]["corrected_cv"], 4)}
                ),
                "resumen": resumenes[mejores[variante]],
            }
            for variante in variantes
        },
        "base": resumenes["BASE@C=2.0"],
        "decision_p3": decisiones["P3"],
        "decision_p4": decisiones["P4"],
        "duracion_segundos": round(duracion, 2),
        "artefactos": [
            str(REPORTE_CSV.relative_to(PROJECT_ROOT)),
            str(REPORTE_JSON.relative_to(PROJECT_ROOT)),
            str(FIGURA_PATH.relative_to(PROJECT_ROOT)),
            str(OOF_SYNTAX_PATH.relative_to(PROJECT_ROOT)),
            str(OOF_CITA_NEG_PATH.relative_to(PROJECT_ROOT)),
            str(CACHE_SINTAXIS.relative_to(PROJECT_ROOT)),
        ],
    }
    REPORTE_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORTE_JSON.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"reporte JSON escrito: {REPORTE_JSON.name}")


def main() -> None:
    """Ejecuta P3+P4 y guarda todos los artefactos."""
    inicio = time.perf_counter()
    desarrollo, _, pliegues = cargar_datos()
    documentos, modo_anotacion = cargar_o_crear_anotaciones(desarrollo)
    y_verdadera = desarrollo[list(ETIQUETAS)].to_numpy()
    constantes, origen_constantes = P2.cargar_constantes_error_analysis()
    log(f"constantes estructurales: {origen_constantes}")

    congelado = np.load(OOF_BASE_PATH)
    # El OOF congelado apila los 4 bases en el eje 1: M2 = indice 1.
    proba_m2_congelada = congelado["proba"][:, 1, :]
    if congelado["ids"].tolist() != desarrollo["id"].astype(int).tolist():
        raise RuntimeError("Los ids del OOF congelado no coinciden con el desarrollo")

    receta_v03 = P2.cargar_receta_v03()
    resultado = evaluar_pliegues(
        desarrollo, documentos, y_verdadera, pliegues, constantes, receta_v03
    )
    proba_base = resultado.oof[("BASE", 2.0)]
    verificaciones = verificar_replicacion(
        proba_base, proba_m2_congelada, y_verdadera, resultado.pliegue_de_fila
    )

    f1_v03 = f1_pliegues_v03(proba_base, y_verdadera, resultado.pliegue_de_fila)
    resumenes: dict[str, dict[str, Any]] = {}
    probabilidades_oof: dict[str, NDArray[np.float64]] = {}
    for clave, proba in resultado.oof.items():
        resumen = resumir_clave(
            clave, proba, y_verdadera, resultado.pliegue_de_fila, proba_base, f1_v03
        )
        resumenes[resumen["clave"]] = resumen
        probabilidades_oof[resumen["clave"]] = proba
    mejores = {
        variante: max(
            (
                resumen["clave"]
                for resumen in resumenes.values()
                if resumen["variante"] == variante
            ),
            key=lambda clave: resumenes[clave]["corrected_cv"],
        )
        for variante in VARIANTES
    }
    decisiones = {
        "P3": decidir_seccion(resumenes, VARIANTES_P3),
        "P4": decidir_seccion(resumenes, VARIANTES_P4),
    }

    lexicos_robustez = P2.inducir_lexicos(
        [documentos[indice] for indice in pliegues[0][0]], y_verdadera[pliegues[0][0]]
    )
    robustez = probar_robustez(documentos, lexicos_robustez, constantes)
    log(f"robustez: {json.dumps(robustez, ensure_ascii=False)[:200]}")

    generar_figura(mejores, resumenes)
    escribir_csv(resumenes)
    guardar_oof(
        OOF_SYNTAX_PATH,
        desarrollo,
        y_verdadera,
        resultado.pliegue_de_fila,
        probabilidades_oof,
        {variante: mejores[variante] for variante in VARIANTES_P3},
    )
    guardar_oof(
        OOF_CITA_NEG_PATH,
        desarrollo,
        y_verdadera,
        resultado.pliegue_de_fila,
        probabilidades_oof,
        {variante: mejores[variante] for variante in VARIANTES_P4},
    )
    registrar_experimentos(resumenes, mejores, decisiones, time.perf_counter() - inicio)

    duracion = time.perf_counter() - inicio
    escribir_json(
        resumenes,
        mejores,
        decisiones,
        verificaciones,
        resultado.diagnostico,
        robustez,
        modo_anotacion,
        duracion,
    )
    lineas_contexto = [
        (
            "P3 (E2-03/E2-04) sintaxis sparse y P4 (E2-07/E2-08) cita/atribucion/negacion sobre el "
            "desarrollo (n=1483), pliegues corregidos v2, holdout congelado intacto."
        ),
        (
            f"Anotacion spaCy SIMBOLICA (POS/lema/morfologia/dep/cabeza) en cache propia "
            f"({CACHE_SINTAXIS.name}); prohibido el uso de vectores y similitud."
        ),
        (
            f"BASE reproducido: corrected_cv 0.7718 a 0.5 (dif max vs OOF congelado "
            f"{verificaciones['diferencia_maxima_oof_base']:.1e}) y V03 con umbrales "
            f"{verificaciones['f1_v03_umbrales']:.4f}."
        ),
        (
            f"P3 - mejor POS: {resumenes[mejores['SYN-POS']]['clave']} "
            f"corrected_cv {resumenes[mejores['SYN-POS']]['corrected_cv']:.4f} "
            f"(delta {resumenes[mejores['SYN-POS']]['delta_vs_v03_umbrales']:+.4f} vs V03, "
            f"pliegues {resumenes[mejores['SYN-POS']]['pliegues_mejorados_vs_v03']}/5); TXT+POS "
            f"{resumenes[mejores['TXT+POS']]['corrected_cv']:.4f} "
            f"({resumenes[mejores['TXT+POS']]['delta_vs_v03_umbrales']:+.4f})."
        ),
        (
            f"P3 - DEPS: SYN-DEPS {resumenes[mejores['SYN-DEPS']]['corrected_cv']:.4f} "
            f"({resumenes[mejores['SYN-DEPS']]['delta_vs_v03_umbrales']:+.4f}); TXT+DEPS "
            f"{resumenes[mejores['TXT+DEPS']]['corrected_cv']:.4f} "
            f"({resumenes[mejores['TXT+DEPS']]['delta_vs_v03_umbrales']:+.4f})."
        ),
        (
            f"P3 - MORPH: SYN-MORPH {resumenes[mejores['SYN-MORPH']]['corrected_cv']:.4f} "
            f"({resumenes[mejores['SYN-MORPH']]['delta_vs_v03_umbrales']:+.4f}); TXT+MORPH "
            f"{resumenes[mejores['TXT+MORPH']]['corrected_cv']:.4f} "
            f"({resumenes[mejores['TXT+MORPH']]['delta_vs_v03_umbrales']:+.4f})."
        ),
        (
            f"P3 - union de los tres: SYN-3 {resumenes[mejores['SYN-3']]['corrected_cv']:.4f} "
            f"({resumenes[mejores['SYN-3']]['delta_vs_v03_umbrales']:+.4f}); TXT+SYN3 "
            f"{resumenes[mejores['TXT+SYN3']]['corrected_cv']:.4f} "
            f"({resumenes[mejores['TXT+SYN3']]['delta_vs_v03_umbrales']:+.4f})."
        ),
        (
            f"P4 - CITA: CITA {resumenes[mejores['CITA']]['corrected_cv']:.4f} "
            f"({resumenes[mejores['CITA']]['delta_vs_v03_umbrales']:+.4f}); TXT+CITA "
            f"{resumenes[mejores['TXT+CITA']]['corrected_cv']:.4f} "
            f"({resumenes[mejores['TXT+CITA']]['delta_vs_v03_umbrales']:+.4f})."
        ),
        (
            f"P4 - NEG: NEG {resumenes[mejores['NEG']]['corrected_cv']:.4f} "
            f"({resumenes[mejores['NEG']]['delta_vs_v03_umbrales']:+.4f}); TXT+NEG "
            f"{resumenes[mejores['TXT+NEG']]['corrected_cv']:.4f} "
            f"({resumenes[mejores['TXT+NEG']]['delta_vs_v03_umbrales']:+.4f})."
        ),
        (
            f"P4 - union: TXT+CITA+NEG {resumenes[mejores['TXT+CITA+NEG']]['corrected_cv']:.4f} "
            f"({resumenes[mejores['TXT+CITA+NEG']]['delta_vs_v03_umbrales']:+.4f})."
        ),
        (
            f"Complementariedad: mejor ensemble_gain "
            f"{max(resumen.get('ensemble_gain', float('-inf')) for resumen in resumenes.values()):+.4f} "
            "sobre M2 a 0.5 (objetivo >= +0.0015 con IC > 0)."
        ),
        (
            f"Gate P3: {'ADOPTADO' if decisiones['P3']['adoptado'] else 'CERRADO SIN MEJORA'} "
            f"({decisiones['P3']['mejor_clave']} {decisiones['P3']['corrected_cv']:.4f}); "
            f"nivel exigido {'cumple' if decisiones['P3']['criterio_nivel']['cumple'] else 'no cumple'}, "
            f"pliegues {decisiones['P3']['criterio_pliegues']['pliegues_mejorados']}/5."
        ),
        (
            f"Gate P4: {'ADOPTADO' if decisiones['P4']['adoptado'] else 'CERRADO SIN MEJORA'} "
            f"({decisiones['P4']['mejor_clave']} {decisiones['P4']['corrected_cv']:.4f}); "
            f"nivel exigido {'cumple' if decisiones['P4']['criterio_nivel']['cumple'] else 'no cumple'}, "
            f"pliegues {decisiones['P4']['criterio_pliegues']['pliegues_mejorados']}/5."
        ),
        (
            "Robustez: texto vacio, solo URL, solo mencion, RT, comillas desbalanceadas y sin insultos "
            "sin NaN/infinito; features independientes del orden del dataset."
        ),
        (
            "Sin fuga: holdout congelado no utilizado; lexicos, vectorizadores, escaladores y modelos "
            "solo con el entrenamiento de cada pliegue."
        ),
        (
            f"Duracion: {duracion:.1f} s. Script: experiments/19_sintaxis_cita_negacion_v2.py. "
            "Artefactos: reports/sintaxis_cita_negacion_v2.csv y .json, "
            "reports/figuras/fig_30_sintaxis_cita_v2.png, features/v2_oof_syntax.npz y "
            "features/v2_oof_cita_negacion.npz."
        ),
    ]
    escribir_contexto(lineas_contexto)
    actualizar_todo(resumenes, mejores, decisiones)

    log("resumen final")
    log(
        f"BASE {resumenes['BASE@C=2.0']['corrected_cv']:.4f} | "
        f"V03 umbrales {verificaciones['f1_v03_umbrales']:.4f}"
    )
    log(f"mejor P3: {decisiones['P3']['mejor_clave']} = {decisiones['P3']['corrected_cv']:.4f}")
    log(f"mejor P4: {decisiones['P4']['mejor_clave']} = {decisiones['P4']['corrected_cv']:.4f}")
    log(
        f"decisiones: P3 {'ADOPTADO' if decisiones['P3']['adoptado'] else 'cerrado'} / "
        f"P4 {'ADOPTADO' if decisiones['P4']['adoptado'] else 'cerrado'}"
    )
    log(f"artefactos: {REPORTE_CSV.name}, {REPORTE_JSON.name}, {FIGURA_PATH.name}")
    log(f"duracion total: {duracion:.1f} s")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - trazabilidad completa y salida con error
        traceback.print_exc()
        sys.exit(1)
