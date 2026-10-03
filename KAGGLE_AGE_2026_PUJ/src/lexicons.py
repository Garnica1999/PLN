"""Núcleo de la fase de lexicones: carga, transliteración y agregados.

Este módulo es la única fuente de verdad del pipeline léxico. No aprende nada de las etiquetas:
la transformación ``tweet -> categorías léxicas`` es una función determinista del texto
(§13 del brief), por lo que la matriz de features se calcula **una sola vez** y no necesita
recalcularse in-fold como el canal supervisado de entidades.

Decisiones de diseño que implementa (ver ``plan/AGE_LEXICON_PLAN.md``):

DD-2  Los ficheros se guardan en español correcto con tildes y se transliteran en carga, porque el
      corpus es ASCII puro (96 caracteres distintos, sin tildes ni eñes; solo ``¿`` no ASCII).
DD-4  Tres agregados por subcategoría (``count``, ``ratio``, ``presence``) y uno por familia.
      ``density_per_100_words`` es una transformación afín de ``ratio_over_words`` y
      ``count_unique`` colapsa en ``presence`` con tweets de 140 caracteres, de modo que las cinco
      estadísticas del §11 serían redundantes por construcción.
DD-5  Un token presente en más de una subcategoría se **elimina**: evita el doble conteo.

Notas de compatibilidad con la transliteración
---------------------------------------------
La pérdida de tildes crea homófonos que se documentan aquí y se miden en P1:

* ``mas``  = "más" | "mas" (conjunción) | "más" (macho)
* ``esta`` = "esta" (demostrativo) | "está" (verbo)
* ``solo`` = "solo" | "sólo"
* ``sino`` = "sí no"
* ``digo`` = "digo" | "di/go"

Por eso ninguna lista usa esas formas ambiguas: se prefieren tokens inequívocos y se deja constancia
del coste en el catálogo.
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
#: Tokenizador: el del proyecto para texto (``advanced_features.TOKEN_RE``) restringido a palabras
#: de dos o más caracteres, que es la convención de los vectorizadores (``\\b\\w\\w+\\b``).
TOKEN_RE = re.compile(r"\b[a-z]{2,}\b")
#: Una subcategoría con menos entradas efectivas que esto se retira: emitir columnas
#: con ratio ~0 solo añade ruido y falsea el agregado de familia.
MIN_ENTRADAS_SUBCATEGORIA = 5
COMENTARIO_RE = re.compile(r"^\s*#")

#: Formas ambiguas al transliterar (documentadas arriba): no se admiten en ninguna lista.
FORMAS_AMBIGUAS = frozenset({"mas", "esta", "estan", "estas", "solo", "sino", "digo", "alla",
                             "mio", "mia", "tuyo", "tia", "dia", "dias", "ves", "sera"})

ESTADISTICAS = ("count", "ratio", "presence")


def transliterar(texto: str) -> str:
    """Pasa un texto a ASCII minúsculo igual que hace el corpus (``ñ`` -> ``n``, sin tildes)."""
    descompuesto = unicodedata.normalize("NFD", texto.lower())
    sin_acentos = "".join(caracter for caracter in descompuesto
                          if unicodedata.category(caracter) != "Mn")
    return sin_acentos.replace("ñ", "n").replace("Ñ", "n")


def tokeniza(texto: str) -> list[str]:
    """Tokens alfanuméricos en ASCII de longitud >= 2, como el vectorizador del proyecto."""
    return TOKEN_RE.findall(transliterar(texto))


def sha256_de(ruta: Path) -> str:
    """Huella SHA256 de un fichero de lexicón."""
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def lee_fichero(ruta: Path) -> tuple[list[str], dict[str, str]]:
    """Lee un ``.txt`` de lexicon: devuelve entradas y metadatos del encabezado."""
    entradas, metadatos = [], {}
    for linea in ruta.read_text(encoding="utf-8").splitlines():
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
    """Carga todos los ``.txt`` de las tres familias y resuelve la ambigüedad (DD-5).

    Returns
    -------
    dict
        ``familias``: ``{familia: {subcategoria: [tokens ascii]}}`` ya sin ambigüedades.
        ``token_a_subcategoria``: ``{token: [(familia, subcategoria), ...]}`` con un único destino
        por token (los ambiguos se descartan).
        ``ambiguos``: tokens presentes en más de una subcategoría, con sus destinos.
        ``metadatos``: encabezado de cada fichero.
        ``rutas``: ruta de cada fichero de lexicon.
    """
    base = Path(directorio) if directorio is not None else LEXICONES
    familias: dict[str, dict[str, list[str]]] = {}
    descartadas_minimo: dict[str, int] = {}
    metadatos: dict[str, dict[str, str]] = {}
    rutas: dict[str, Path] = {}
    apariciones: dict[str, set[tuple[str, str]]] = {}
    for familia in FAMILIAS:
        familias[familia] = {}
        for ruta in sorted((base / familia).glob("*.txt")):
            entradas, cabecera = lee_fichero(ruta)
            subcategoria = cabecera.get("subcategoria", ruta.stem)
            tokens = []
            for palabra in entradas:
                ascii_palabra = transliterar(palabra)
                if len(ascii_palabra) < 2:
                    continue
                if ascii_palabra in FORMAS_AMBIGUAS:
                    # Defensa en profundidad: el cribado principal está en el escritor, pero una
                    # edición manual de un .txt no debe reintroducir formas que colisionan.
                    continue
                tokens.append(ascii_palabra)
                apariciones.setdefault(ascii_palabra, set()).add((familia, subcategoria))
            familias[familia][subcategoria] = sorted(set(tokens))
            metadatos[f"{familia}/{subcategoria}"] = cabecera
            rutas[f"{familia}/{subcategoria}"] = ruta
    ambiguos = {token: sorted(destinos) for token, destinos in apariciones.items()
                if len(destinos) > 1}
    destino_unico: dict[str, list[tuple[str, str]]] = {
        token: list(destinos) for token, destinos in apariciones.items() if len(destinos) == 1}
    for token, destinos in ambiguos.items():
        for familia, subcategoria in destinos:
            familias[familia][subcategoria] = [t for t in familias[familia][subcategoria]
                                               if t != token]
    # Una subcategoría que se queda sin vocabulario utilizable se retira: emitir columnas con
    # ratio ~0 solo añade ruido al agregado de familia y falsea su media.
    for familia, subcategorias in familias.items():
        for subcategoria, lista in list(subcategorias.items()):
            if len(lista) < MIN_ENTRADAS_SUBCATEGORIA:
                descartadas_minimo[f"{familia}/{subcategoria}"] = len(lista)
                del subcategorias[subcategoria]
                rutas.pop(f"{familia}/{subcategoria}", None)
    return {"familias": familias, "token_a_subcategoria": destino_unico, "ambiguos": ambiguos,
            "metadatos": metadatos, "rutas": rutas,
            "descartadas_por_minimo": descartadas_minimo}


def _resuelve_destino(token_a_subcategoria: dict, familia: str, subcategoria: str) -> dict[str, int]:
    """Mapa token -> 1 para las entradas de una subcategoría concreta."""
    return {token: 1 for token, destinos in token_a_subcategoria.items()
            if destinos[0] == (familia, subcategoria)}


def matriz_lexica(textos: pd.Series, lexicon: dict, familias: tuple[str, ...] = FAMILIAS
                  ) -> tuple[np.ndarray, list[str]]:
    """Construye la matriz de agregados léxicos de los tweets.

    Parámetros
    ----------
    textos : pandas.Series
        Texto crudo (se aplica ``clean_text`` si viene con padding).
    lexicon : dict
        Salida de :func:`carga_lexicones`.
    familias : tuple of str
        Familias a incluir.

    Returns
    -------
    tuple
        ``(matriz, nombres)`` con una columna por subcategoría y estadística más un agregado por
        familia (media de ratios), en el orden de columnas de ``nombres``.
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
    matriz = np.column_stack(bloques).astype(np.float32)
    return matriz, nombres


def cobertura(tokens_documento: list[list[str]], lexicon: dict, familia: str) -> dict:
    """Fracción de tokens del corpus que pertenecen a alguna subcategoría de la familia."""
    vocabulario = {token for token, destinos in lexicon["token_a_subcategoria"].items()
                   for destino in destinos if destino[0] == familia}
    total = sum(len(documento) for documento in tokens_documento) or 1
    cubiertos = sum(1 for documento in tokens_documento for token in documento
                    if token in vocabulario)
    return {"tokens_corpus": total, "tokens_cubiertos": cubiertos,
            "cobertura_pct": round(100.0 * cubiertos / total, 4)}
