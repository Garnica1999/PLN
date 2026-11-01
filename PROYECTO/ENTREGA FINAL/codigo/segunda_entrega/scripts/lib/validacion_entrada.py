"""validacion_entrada.py
Fecha: 2026-09-08
Descripcion: Modulo M0 de validacion de entrada del producto Juris-PLN.
    Decide si un texto es una providencia de la Corte Constitucional mediante
    la heuristica del anexo 25: (1) encabezado con canon de sentencia
    (regex ^(sentencia )?(C|T|SU) NUM ANO), (2) marcador "corte
    constitucional" y (3) marcador "magistrado ponente", todos insensibles a
    tildes y mayusculas. El score es reglas cumplidas / total y el umbral de
    aceptacion es >= 2/3.
Estado del arte aplicado: reglas lexicas del genero sentencia de la Corte
    Constitucional con normalizacion Unicode (NFD + eliminacion de
    diacriticos); el encabezado se evalúa solo en la zona inicial del texto.
"""

import re
import unicodedata

CONFIG = {
    "zona_encabezado_chars": 3000,
    "total_reglas": 3,
    "umbral_reglas": 2,
}

RE_ENCABEZADO = re.compile(r"^(?:sentencia\s+)?([cts])(u)?\s*\d+\s+\d+", re.MULTILINE)
MARCADOR_CORTE = "corte constitucional"
MARCADOR_PONENTE = "magistrado ponente"
SEPARADOR = re.compile(r"[^a-z0-9\n]+")
ESPACIOS = re.compile(r"[ \t]+")


def normalizar_texto(texto: str) -> str:
    """Normaliza un texto a minusculas y sin marcas diacriticas.

    Conserva saltos de linea y reemplaza la puntuacion y otros separadores por
    espacios simples para que el matching por marcadores y encabezado sea
    insensible a tildes, mayusculas, guiones y espacios no separables.

    Args:
        texto: Texto original en espanol.

    Returns:
        Texto normalizado en una unica escala (una linea por linea original).
    """
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFD", texto) if not unicodedata.combining(c)
    )
    sin_tildes = sin_tildes.lower()
    con_separadores = SEPARADOR.sub(" ", sin_tildes)
    return ESPACIOS.sub(" ", con_separadores)


def _zona_encabezado(texto_normalizado: str) -> str:
    """Recorta la zona inicial donde debe vivir el encabezado.

    La cabecera de una providencia (canon, republica de colombia, corte
    constitucional y magistrado ponente) aparece en los primeros cientos de
    caracteres; restringir la busqueda alli evita falsos positivos por citas
    de sentencias en el cuerpo (p. ej. "C-004-92" como referencia).

    Args:
        texto_normalizado: Texto ya normalizado por normalizar_texto().

    Returns:
        Porcion normalizada del encabezado (primeros CONFIG chars).
    """
    return texto_normalizado[: CONFIG["zona_encabezado_chars"]]


def evaluar_reglas(texto: str) -> dict:
    """Evalua las tres reglas de la heuristica M0 sobre un texto.

    Args:
        texto: Texto completo del documento a validar (crudo, con tildes).

    Returns:
        Dict con el cumplimiento de cada regla (encabezado, corte, ponente),
        el numero de reglas cumplidas y la decision binaria es_sentencia
        (reglas cumplidas >= 2/3).
    """
    normalizado = normalizar_texto(texto)
    encabezado = RE_ENCABEZADO.search(_zona_encabezado(normalizado)) is not None
    plano = re.sub(r"\s+", " ", normalizado)
    marcador_corte = MARCADOR_CORTE in plano
    marcador_ponente = MARCADOR_PONENTE in plano
    reglas_cumplidas = sum((encabezado, marcador_corte, marcador_ponente))
    return {
        "encabezado": encabezado,
        "marcador_corte": marcador_corte,
        "marcador_ponente": marcador_ponente,
        "reglas_cumplidas": reglas_cumplidas,
        "total_reglas": CONFIG["total_reglas"],
        "es_sentencia": reglas_cumplidas >= CONFIG["umbral_reglas"],
    }


def es_sentencia_corte(texto: str) -> bool:
    """Clasifica un texto como providencia de la Corte Constitucional.

    Args:
        texto: Texto completo del documento a validar.

    Returns:
        True si cumple al menos 2 de las 3 reglas (umbral 2/3), False si no.
    """
    return evaluar_reglas(texto)["es_sentencia"]
