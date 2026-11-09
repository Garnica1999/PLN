"""tools.py
Fecha: 2026-10-31
Descripcion: Herramientas LangChain del agente Juris-PLN. Cada @tool envuelve
    una funcion de core.py y devuelve JSON compacto en texto: el LLM lee la
    salida, por eso se recortan los campos largos (texto de articulos, listas)
    para no agotar el contexto ni inflar la latencia. Los errores se devuelven
    como {"error": ...} en lugar de excepciones, para que el agente pueda
    corregir la llamada o responder sin evidencia.
Estado del arte aplicado: tool calling de LangChain (langchain_core.tools) con
    docstrings descriptivos; evidencia estructurada y acotada.
"""

import json

from langchain_core.tools import tool

import core

MAX_ARTICULO_CHARS = 140
MAX_RECUPERADOS = 6
MAX_EXTRAIDOS = 8
MAX_CITAS = 10


def _json(datos: dict) -> str:
    """Serializa un dict a JSON compacto en una linea."""
    return json.dumps(datos, ensure_ascii=False, separators=(",", ":"))


def _recortar(texto: str | None, limite: int = MAX_ARTICULO_CHARS) -> str:
    """Recorta un texto a `limite` caracteres agregando elipsis."""
    if not texto:
        return ""
    limpio = " ".join(texto.split())
    if len(limpio) <= limite:
        return limpio
    return limpio[:limite].rstrip() + "..."


def _error(exc: Exception) -> str:
    """Empaqueta una excepcion como salida de herramienta."""
    return _json({"error": str(exc)})


# ===== SECCION 1: herramientas =====
@tool
def validar_sentencia(texto: str) -> str:
    """Valida si un texto es una providencia de la Corte Constitucional.

    Aplica la heuristica M0 (encabezado con canon, marcador "corte
    constitucional" y marcador "magistrado ponente"; umbral 2 de 3 reglas).
    Usala solo cuando el usuario pegue el texto de la providencia o pida
    validar un documento completo.

    Args:
        texto: Texto completo del documento a validar.

    Returns:
        JSON con es_sentencia, score_validacion, reglas y canon_detectado.
    """
    try:
        resultado = core.validar(texto)
    except Exception as exc:  # noqa: BLE001 - el agente lee el error
        return _error(exc)
    return _json(resultado)


@tool
def vincular_cp(referencia: str, k: int = 6, rerank: bool = False) -> str:
    """Identifica los articulos de la Constitucion Politica vinculados.

    Devuelve dos grupos: los articulos citados explicitamente en la sentencia
    y los recuperados por el producto (BM25 con prior de hubs o hibrido
    B2+B1). Acepta el id del corpus, el canon de la sentencia o el texto
    literal; si el usuario describe un tema sin nombrar la sentencia, pasa su
    consulta como texto.

    Args:
        referencia: Id del corpus ("2195"), canon ("T-293/97") o texto.
        k: Numero de articulos a recuperar (1-20). Por defecto 6.
        rerank: True usa la ruta hibrida B2+B1 (GPU, mas lenta); False usa
            BM25 con prior de hubs.

    Returns:
        JSON con id, canon_detectado, articulos_extraidos y
        articulos_recuperados (numero, score, bloque y texto recortado en
        ambos grupos).
    """
    try:
        resultado = core.vincular(referencia, k=k, rerank=rerank)
    except Exception as exc:  # noqa: BLE001 - el agente lee el error
        return _error(exc)
    extraidos = [
        {
            "numero": fila["numero"],
            "bloque": fila.get("bloque"),
            "texto": core.texto_constitucional(fila["numero"])
            or _recortar(fila.get("texto_150chars")),
        }
        for fila in resultado["articulos_extraidos"][:MAX_EXTRAIDOS]
    ]
    recuperados = [
        {
            "numero": fila["numero"],
            "score": fila.get("score"),
            "bloque": fila.get("bloque"),
            "texto": core.texto_constitucional(fila["numero"])
            or _recortar(fila.get("texto_150chars")),
        }
        for fila in resultado["articulos_recuperados"][:MAX_RECUPERADOS]
    ]
    return _json(
        {
            "id": resultado["id"],
            "canon_detectado": resultado["canon_detectado"],
            "origen": resultado["origen"],
            "rerank": resultado["rerank"],
            "articulos_extraidos": extraidos,
            "articulos_recuperados": recuperados,
        }
    )


@tool
def contexto_citas(referencia: str) -> str:
    """Lista y contextualiza las sentencias que cita una providencia.

    Extrae las citas del texto (M2) y las resuelve contra el corpus (M5) con
    ano, area tematica y PageRank de la red de citas. La autocita del
    encabezado se excluye. Acepta id del corpus, canon o texto.

    Args:
        referencia: Id del corpus ("2195"), canon ("T-293/97") o texto.

    Returns:
        JSON con canon_detectado, n_resueltas y citas (canon, id, ano,
        pagerank y area si resuelven).
    """
    try:
        resultado = core.contexto(texto=referencia)
    except Exception as exc:  # noqa: BLE001 - el agente lee el error
        return _error(exc)
    citas = [
        {
            "canon": fila["canon"],
            "id": fila["id"],
            "resuelta": fila["resuelta"],
            "ano": fila.get("ano"),
            "pagerank": fila.get("pagerank"),
            "area": fila.get("area"),
        }
        for fila in resultado["citas"][:MAX_CITAS]
    ]
    return _json(
        {
            "id": resultado["id"],
            "canon_detectado": resultado["canon_detectado"],
            "n_citas": resultado["n_citas"],
            "n_resueltas": resultado["n_resueltas"],
            "citas": citas,
        }
    )


@tool
def explorar_articulo(numero: str) -> str:
    """Consulta el texto y el bloque tematico de un articulo de la CP.

    Args:
        numero: Numero del articulo sin la palabra "articulo" (por ejemplo
            "13", "22A", "Transitorio 68").

    Returns:
        JSON con numero, texto de la primera frase o frases (hasta 600
        caracteres) y bloque tematico.
    """
    try:
        resultado = core.detalle_articulo(numero)
    except Exception as exc:  # noqa: BLE001 - el agente lee el error
        return _error(exc)
    return _json(
        {
            "numero": resultado["numero"],
            "texto": core.texto_constitucional(resultado["numero"], 600)
            or _recortar(resultado["texto_150chars"], 600),
            "bloque": resultado["bloque"],
        }
    )


TOOLS = [validar_sentencia, vincular_cp, contexto_citas, explorar_articulo]
MAPA_TOOLS = {herramienta.name: herramienta for herramienta in TOOLS}
