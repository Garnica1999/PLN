"""Núcleo de la fase de lexicones: carga, transliteración y agregados.

Reconstruido tras el incidente git del 2026-09-25 (ver ``INCIDENTE_GIT_2026-09-25.md``). La lógica
es la misma que se ejecutó en P1-P5; la reconstrucción se verifica comparando el hash de las
matrices generadas con el de las matrices que sobrevivieron a la pérdida.

Decisiones de diseño que implementa (ver ``plan/AGE_LEXICON_PLAN.md``):

DD-2  Los ficheros se guardan en español correcto con tildes y se transliteran en carga, porque el
      corpus es ASCII puro (96 caracteres distintos, sin tildes ni eñes; solo ``¿`` no ASCII).
DD-4  Tres agregados por subcategoría (``count``, ``ratio``, ``presence``) y uno por familia.
      ``density_per_100_words`` es una transformación afín de ``ratio_over_words`` y
      ``count_unique`` colapsa en ``presence`` con tweets de 140 caracteres, de modo que las cinco
      estadísticas del §11 serían redundantes por construcción.
DD-5  Un token presente en más de una subcategoría se **elimina**: evita el doble conteo.

Pérdida por transliteración (documentada, medida y asumida): ``mas`` = "más"|"mas", ``esta`` =
"esta"|"está", ``solo`` = "solo"|"sólo", ``sino`` = "sí no". Ninguna lista usa esas formas.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

LEXICONES = cfg.ROOT / "lexicons"
FAMILIAS = ("registro", "vida", "temas")
#: Tokenizador de palabras: ASCII en minúscula, dos o más caracteres y con límites de palabra
#: (convención de los vectorizadores del proyecto, ``\b\w\w+\b``, sobre el corpus transliterado).
TOKEN_RE = re.compile(r"\b[a-z]{2,}\b")
#: Una subcategoría con menos entradas efectivas que esto se retira: emitir columnas con ratio ~0
#: solo añade ruido y falsea el agregado de familia.
MIN_ENTRADAS_SUBCATEGORIA = 5
COMENTARIO_RE = re.compile(r"^\s*#")
ESTADISTICAS = ("count", "ratio", "presence")
#: Formas ambiguas al transliterar: no se admiten en ninguna lista.
FORMAS_AMBIGUAS = frozenset({"mas", "esta", "estan", "estas", "solo", "sino", "digo", "alla",
                             "mio", "mia", "tuyo", "tia", "dia", "dias", "ves", "sera"})


def transliterar(texto: str) -> str:
    """Pasa un texto a ASCII minúsculo igual que hace el corpus (``ñ`` -> ``n``, sin tildes)."""
    descompuesto = unicodedata.normalize("NFD", texto.lower())
    return "".join(caracter for caracter in descompuesto
                   if unicodedata.category(caracter) != "Mn")


def tokeniza(texto: str) -> list[str]:
    """Tokens alfanuméricos en ASCII de longitud >= 2, con límites de palabra."""
    return TOKEN_RE.findall(transliterar(texto))


def sha256_de(ruta: Path) -> str:
    """Huella SHA256 de un fichero de lexicón."""
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def lee_fichero(ruta: Path) -> tuple[list[str], dict[str, str]]:
    """Lee un ``.txt`` de lexicon: devuelve entradas y metadatos del encabezado."""
    entradas, metadatos = [], {}
    for linea in Path(ruta).read_text(encoding="utf-8").splitlines():
        if COMENTARIO_RE.match(linea):
            cuerpo = linea.lstrip("#").strip()
            if ":" in cuerpo:
                clave, _, valor = cuerpo.partition(":")
                metadatos[clave.strip()] = valor.strip()
            continue
        if linea.strip():
            entradas.append(linea.strip())
    return entradas, metadatos


def carga_lexicones(directorio: Path | None = None) -> dict:
    """Carga los ``.txt`` de las tres familias y resuelve la ambigüedad (DD-5) y el mínimo.

    Returns
    -------
    dict
        ``familias``, ``token_a_subcategoria`` (un único destino por token), ``ambiguos``,
        ``metadatos``, ``rutas`` y ``descartadas_por_minimo``.
    """
    base = Path(directorio) if directorio is not None else LEXICONES
    familias: dict[str, dict[str, list[str]]] = {}
    metadatos: dict[str, dict[str, str]] = {}
    rutas: dict[str, Path] = {}
    descartadas_minimo: dict[str, int] = {}
    apariciones: dict[str, set[tuple[str, str]]] = {}
    for familia in FAMILIAS:
        familias[familia] = {}
        for ruta in sorted((base / familia).glob("*.txt")):
            entradas, cabecera = lee_fichero(ruta)
            subcategoria = cabecera.get("subcategoria", ruta.stem)
            tokens = []
            for palabra in entradas:
                ascii_palabra = transliterar(palabra)
                if len(ascii_palabra) < 2 or ascii_palabra in FORMAS_AMBIGUAS:
                    continue
                tokens.append(ascii_palabra)
                apariciones.setdefault(ascii_palabra, set()).add((familia, subcategoria))
            familias[familia][subcategoria] = sorted(set(tokens))
            metadatos[f"{familia}/{subcategoria}"] = cabecera
            rutas[f"{familia}/{subcategoria}"] = ruta
    ambiguos = {token: sorted(destinos) for token, destinos in apariciones.items()
                if len(destinos) > 1}
    destino_unico = {token: list(destinos) for token, destinos in apariciones.items()
                     if len(destinos) == 1}
    for token, destinos in ambiguos.items():
        for familia, subcategoria in destinos:
            familias[familia][subcategoria] = [t for t in familias[familia][subcategoria]
                                               if t != token]
    for familia, subcategorias in familias.items():
        for subcategoria, lista in list(subcategorias.items()):
            if len(lista) < MIN_ENTRADAS_SUBCATEGORIA:
                descartadas_minimo[f"{familia}/{subcategoria}"] = len(lista)
                del subcategorias[subcategoria]
                rutas.pop(f"{familia}/{subcategoria}", None)
    return {"familias": familias, "token_a_subcategoria": destino_unico, "ambiguos": ambiguos,
            "metadatos": metadatos, "rutas": rutas, "descartadas_por_minimo": descartadas_minimo}


def _resuelve_destino(token_a_subcategoria: dict, familia: str, subcategoria: str) -> dict[str, int]:
    """Mapa token -> 1 para las entradas de una subcategoría concreta."""
    return {token: 1 for token, destinos in token_a_subcategoria.items()
            if destinos[0] == (familia, subcategoria)}


def matriz_lexica(textos: pd.Series, lexicon: dict,
                  familias: tuple[str, ...] = FAMILIAS) -> tuple[np.ndarray, list[str]]:
    """Construye la matriz de agregados léxicos de los tweets (count, ratio, presence + familia).

    La transformación es una función determinista del texto y **no usa etiquetas** (§13): por eso la
    matriz se calcula una sola vez para todo el corpus, sin recalcular in-fold.
    """
    filas = textos.fillna("").astype(str).str.strip()
    documentos = [tokeniza(valor) for valor in filas]
    n_palabras = np.array([max(len(documento), 1) for documento in documentos], dtype=float)
    nombres: list[str] = []
    bloques: list[np.ndarray] = []
    ratios_familia: dict[str, list[np.ndarray]] = {}
    for familia in familias:
        ratios_familia[familia] = []
        for subcategoria in sorted(lexicon["familias"][familia]):
            vocabulario = _resuelve_destino(lexicon["token_a_subcategoria"], familia, subcategoria)
            if not vocabulario:
                continue
            conteo = np.zeros(len(documentos), dtype=np.float32)
            for indice, documento in enumerate(documentos):
                if documento:
                    conteo[indice] = sum(1 for token in documento if token in vocabulario)
            ratio = conteo / n_palabras
            presence = (conteo > 0).astype(np.float32)
            prefijo = f"lex_{familia}_{subcategoria}"
            nombres += [f"{prefijo}_count", f"{prefijo}_ratio", f"{prefijo}_presence"]
            bloques += [conteo[:, None], ratio[:, None], presence[:, None]]
            ratios_familia[familia].append(ratio)
        if ratios_familia[familia]:
            nombres.append(f"lex_{familia}_ratio_medio")
            bloques.append(np.mean(np.column_stack(ratios_familia[familia]), axis=1)[:, None])
    return np.column_stack(bloques).astype(np.float32), nombres


def cobertura(tokens_documento: list[list[str]], lexicon: dict, familia: str) -> dict:
    """Fracción de tokens del corpus que pertenecen a alguna subcategoría de la familia."""
    vocabulario = {token for token, destinos in lexicon["token_a_subcategoria"].items()
                   for destino in destinos if destino[0] == familia}
    total = sum(len(documento) for documento in tokens_documento) or 1
    cubiertos = sum(1 for documento in tokens_documento for token in documento
                    if token in vocabulario)
    return {"tokens_corpus": total, "tokens_cubiertos": cubiertos,
            "cobertura_pct": round(100.0 * cubiertos / total, 4)}
