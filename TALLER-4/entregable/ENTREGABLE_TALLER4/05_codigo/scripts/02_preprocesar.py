"""02_preprocesar.py

Fecha: 2026-09-04
Descripcion: Fase 4 del TALLER-4 (IR sobre corpus CEV). Preprocesado de los dos
    corpus siguiendo ADR-002, ADR-003 y las secciones (c) y (d) de
    plan/arquitectura.md:

    Parte 1 (normalizacion estructural de entrevistas, regex, sin spaCy):
        - limpieza de encabezados de sistema (codigos 001-VI-XXXXX_(NNNNN)_,
          lineas AUDIO, codigos embebidos con timecode),
        - eliminacion de anotaciones entre corchetes (con inventario por tipo),
        - mapeo de anonimizados a categorias canonicas (ADR-002):
          organizacion_publica, organizacion_social, lugar, familiar, conocido,
          institucion_educativa, partido_politico y anonimo (generico),
        - eliminacion de etiquetas de hablante al inicio de linea y de tokens
          de etiqueta residuales (TEST, ENT, ...), y de guiones (>=3),
        - inventario de patrones (variantes reales y conteos) y estadisticas
          por documento para el informe EDA.
        Entrada : data/raw/entrevistas_all_2023-03-21_14-24_05.json
        Salidas : data/corpus/entrevistas_normalizadas.jsonl
                  resultados/preprocesado/inventario_patrones_entrevistas.json

    Parte 2 (preprocesado linguistico spaCy es_core_news_md, CPU):
        - pipeline unico para libros y entrevistas: tokenizacion, minusculas,
          lematizacion; stopwords = spacy.lang.es.STOP_WORDS + muletillas del
          habla (config/stopwords_habla.txt, lista inicial verificada por
          frecuencia en una muestra de 30 entrevistas); filtro no alfabetico
          conservando anios 1900-2025 y las categorias canonicas como un solo
          token.
        - salidas 1 documento por linea (lemas y tokens crudos) con indice de
          metadatos alineado por posicion; checkpoint por documento
          (reanudable, automatico por numero de lineas o --desde N).
        Salidas : data/corpus/preprocesado/<libros|entrevistas>_lemas.txt
                  data/corpus/preprocesado/<libros|entrevistas>_tokens.txt
                  data/corpus/preprocesado/<libros|entrevistas>_docs.json

    Estado del arte aplicado (herencia del plan):
        - ADR-002: categorias canonicas en minusculas y subrayado que se
          comportan como un solo token lexico; el resto de tokens en mayusculas
          con sufijo numerico -> anonimo. Reglas exactas verificadas sobre el
          corpus real: variantes ortograficas (O/Ó, U/Ú), numero pegado o
          separado por espacios/nuevas lineas, numero opcional en organizaciones
          cuando el siguiente caracter no es una minuscula.
        - ADR-003: n_process=8-16 validado (8 es el optimo medido, 4.1x vs 1),
          batch_size pequeno para documentos muy largos; fork como metodo de
          arranque multiproceso (requerido por Python 3.14).
        - arquitectura (c)6: normalizacion estructural sin lowercase ni
          lematizacion; el linguistico va en la parte 2.

Uso:
    python scripts/02_preprocesar.py                          # todo: parte 1 + parte 2 (libros y entrevistas)
    python scripts/02_preprocesar.py normalizar               # solo parte 1 (entrevistas)
    python scripts/02_preprocesar.py spacy --corpus libros    # solo parte 2 libros
    python scripts/02_preprocesar.py spacy --corpus entrevistas --muestra 200
    python scripts/02_preprocesar.py spacy --corpus entrevistas --desde 500
    python scripts/02_preprocesar.py spacy --corpus entrevistas --force
    python scripts/02_preprocesar.py --solo libros            # atajo de spacy --corpus libros
    python scripts/02_preprocesar.py --solo entrevistas       # normalizar + spacy entrevistas

    Flags: --force (regenerar aunque exista), --desde N (reanudar desde el
    documento N, 0-based), --muestra N (corrida de validacion sobre una muestra
    fija de N documentos: escribe en preprocesado/validacion_<corpus>_*.txt y
    NO toca las salidas de produccion), --procesos K (workers spaCy).

Notas de implementacion:
    - El corpus de entrevistas es oral transcrito con marcas de sistema
      (encabezados, separadores, etiquetas de turno) que no son habla; la
      limpieza estructural es una decision de dominio explicita (Castillejo
      Cuellar 2023 en plan/estado_del_arte.md), no solo higiene.
    - Los codigos 001-VI-XXXXX embebidos en el cuerpo son encabezados de
      pagina/segmento con timecode (no citas internas): se eliminan en todas
      sus apariciones (verificado en la corrida completa de produccion:
      896 ocurrencias en 476 documentos; ver
      resultados/preprocesado/inventario_patrones_entrevistas.json).
    - La categorizacion de anonimizados es una transformacion interna del
      corpus (no re-identificacion); los ids/roles originales no se publican.
    - En esta fase NO se congelan las muletillas: la lista inicial con
      evidencia queda en config/stopwords_habla.txt y el EDA (Fase 5) la
      congela. Tokens borderline ('ve', 'cierto') se documentan en
      resultados/preprocesado/muletillas_muestra.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import random
import re
import resource
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ============================================================================
# SECCION 0: CONFIGURACION GLOBAL
# ============================================================================

RUTA_PROYECTO = Path(__file__).resolve().parents[1]

CONFIG: dict[str, Any] = {
    "rutas": {
        "raw_entrevistas": RUTA_PROYECTO / "data" / "raw" / "entrevistas_all_2023-03-21_14-24_05.json",
        "segmentos_libros": RUTA_PROYECTO / "data" / "corpus" / "segmentos_libros.jsonl",
        "entrevistas_normalizadas": RUTA_PROYECTO / "data" / "corpus" / "entrevistas_normalizadas.jsonl",
        "preprocesado": RUTA_PROYECTO / "data" / "corpus" / "preprocesado",
        "resultados_preprocesado": RUTA_PROYECTO / "resultados" / "preprocesado",
        "stopwords_habla": RUTA_PROYECTO / "config" / "stopwords_habla.txt",
    },
    "modelo_spacy": "es_core_news_md",
    "n_procesos": 8,  # validado: optimo medido (4.1x vs 1; 16 no mejora)
    "batch_libros": 16,
    "batch_entrevistas": 4,
    "min_anio": 1900,
    "max_anio": 2025,
    "semilla_muletillas": 42,
    "semilla_validacion": 20260904,
    "n_muestra_muletillas": 30,
    "min_docs_muletilla": 5,    # de 30 documentos de la muestra
    "min_total_muletilla": 20,  # ocurrencias totales en la muestra
    "min_total_nucleo": 2,      # interjecciones puras con presencia minima
}

# Categorias canonicas ADR-002 (texto tal como queda en el corpus normalizado).
CATEGORIAS_CANONICAS: frozenset[str] = frozenset(
    {
        "organizacion_publica",
        "organizacion_social",
        "lugar",
        "familiar",
        "conocido",
        "institucion_educativa",
        "partido_politico",
        "anonimo",
    }
)

# Tokens de etiqueta de hablante residuales (sin dos puntos) que se eliminan
# en cualquier posicion; los marcadores de turno llevan dos puntos y se tratan
# aparte al inicio de linea.
TOKENS_ETIQUETA: tuple[str, ...] = (
    "ACOMP", "ACOM", "PRUEBA", "ENT", "TES", "TEST", "TETS", "TET", "TSET",
    "TESTS", "ENTE", "ENTT", "ENTS", "TENT", "INF", "ETN", "TRAD", "FAC",
    "MEM", "EST", "TODOS", "VARIOS",
)


def _compilar_patrones_anonimizados() -> list[tuple[str, re.Pattern[str]]]:
    """Compila los patrones de anonimizacion de ADR-002.

    Orden importante: primero las categorias especificas (multi-palabra y
    unarias) y al final el generico 'anonimo', para que el generico solo
    capture lo que no cubrieron las especificas. Los patrones admiten la
    variante ortografica O/Ó y U/Ú y el numero pegado o separado por espacios
    o nuevas lineas. En las organizaciones el numero es opcional solo cuando
    el siguiente caracter no es una minuscula: la alternancia intenta primero
    consumir el numero (p.ej. 'PUBLICA 1 y ...') y solo si no hay numero exige
    que no le siga una minuscula ('PUBLICA es...' no es una marca y no se toca).

    Returns: lista de (categoria, patron_compilado).
    """
    e = r"[ \t\n]"
    num = rf"(?:{e}*[0-9]+)"
    no_minuscula = r"(?![ \t]*[a-záéíóúñ])"
    org_pub = re.compile(
        rf"ORGANIZACI[ÓO]N{e}+P[ÚU]BLICA(?:{num}|{no_minuscula})"
    )
    org_soc = re.compile(
        rf"ORGANIZACI[ÓO]N{e}+SOCIAL(?:{num}|{no_minuscula})"
    )
    inst_edu = re.compile(
        rf"INSTITUCI[ÓO]N{e}+EDUCATIVA(?:{num}|{no_minuscula})"
    )
    partido = re.compile(
        rf"PARTIDO{e}+POLITICO(?:{num}|{no_minuscula})"
    )
    lugar = re.compile(r"\bLUGAR[ \t\n]*[0-9]+\b")
    familiar = re.compile(r"\bFAMILIAR[ \t\n]*[0-9]+\b")
    conocido = re.compile(r"\bCONOCIDO[ \t\n]*[0-9]+\b")
    generico = re.compile(
        r"(?<![A-ZÑÁÉÍÓÚa-zñáéíóú0-9])[A-ZÑÁÉÍÓÚ]{2,}[0-9]+(?![A-ZÑÁÉÍÓÚa-zñáéíóú0-9])"
    )
    return [
        ("organizacion_publica", org_pub),
        ("organizacion_social", org_soc),
        ("institucion_educativa", inst_edu),
        ("partido_politico", partido),
        ("lugar", lugar),
        ("familiar", familiar),
        ("conocido", conocido),
        ("anonimo", generico),
    ]


ANONIMIZADOS: list[tuple[str, re.Pattern[str]]] = _compilar_patrones_anonimizados()

# Encabezado de sistema al inicio del documento: 001-VI-00019_(48201)_
RE_CODIGO_LIDER = re.compile(
    r"\A[0-9]{3}-[A-Z]{2}-[0-9]{4,6}(?:[ _-]?\(?[0-9]{3,8}\)?_?)?[ \t]*(?=\n|$)"
)

# Lineas AUDIO (marcadores de audio del transcriptor; solo mayusculas,
# digitos, guiones, parentesis y dos puntos; nunca contenido con minusculas).
RE_LINEA_AUDIO = re.compile(r"AUDIO[0-9A-ZÑÁÉÍÓÚ _\-–():]*")

# Codigo de sistema embebido en el cuerpo, con timecode opcional:
# 001-VI-00011_(48078): 00:45:25
RE_CODIGO_EMBEBIDO = re.compile(
    r"\b[0-9]{3}-[A-Z]{2}-[0-9]{4,6}(?:[ _-]?\(?[0-9]{3,8}\)?_?)?"
    r"(?::[ \t]*[0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?)?"
)

# Anotaciones entre corchetes (no son habla; se eliminan con su contenido).
RE_BRACKET = re.compile(r"\[[^\[\]]*\]")

# Etiqueta de hablante con dos puntos al inicio de linea (TEST:, ENT1:, X:,
# PRUEBA:, ACOMPANANTE:, ...). Rango [A-ZÑ]{1,16} validado contra el corpus
# (la etiqueta mas larga observada es ENTREVISTADO con 12 letras).
RE_LABEL_LINEA = re.compile(r"(?m)^[ \t]*[A-ZÑ]{1,16}[0-9]{0,3}:[ \t]*")

# Linea que queda compuesta solo por una etiqueta sin dos puntos
# ('TEST', 'ENT1' en su propia linea).
RE_LINEA_SOLO_ETIQUETA = re.compile(r"[A-ZÑ]{1,16}[0-9]{0,3}")

# Tokens de etiqueta residuales en medio del texto (marcas sin dos puntos).
RE_ETIQUETA_AISLADA = re.compile(
    r"(?<![A-ZÑÁÉÍÓÚa-zñáéíóú0-9])(?:"
    + "|".join(TOKENS_ETIQUETA)
    + r")[0-9]{0,3}(?![A-ZÑÁÉÍÓÚa-zñáéíóú0-9])"
)

# Guiones separadores de sistema (>=3 guiones, tambien rayas largas).
RE_GUIONES = re.compile(r"[ \t]*[\-–—]{3,}[ \t]*")

# Tokenizacion ligera para evidencias de frecuencia (sin spaCy).
RE_PALABRA = re.compile(r"[a-záéíóúüñ]+")

# Candidatos a muletilla medidos en la muestra de 30 entrevistas (union de la
# lista del enunciado y ADR-003; 'o sea'/'no sé' y otras frases se reportan
# aparte en el CSV porque el archivo de stopwords es de un token por linea).
MULETILLAS_CANDIDATAS: tuple[str, ...] = (
    "eh", "ehh", "este", "sea", "o", "no", "pues", "bueno", "entonces",
    "digamos", "listo", "cierto", "ajá", "aja", "ah", "ahh", "mmm", "mm",
    "mhm", "umm", "um", "mirá", "vea", "ve", "jum", "uy", "ujum", "mjm",
    "mmh", "uf", "mire", "sabe",
)

# Interjecciones puras: se incluyen con solo presencia minima en la muestra
# (son inequivocas pero poco frecuentes en 30 documentos).
NUCLEO_INTERJECCIONES: frozenset[str] = frozenset(
    {"mmm", "mm", "umm", "um", "ahh", "aja", "jum", "ehh", "mjm", "mmh",
     "mhm", "ujum", "uf"}
)

# Frases multipalabra reportadas como evidencia (no aplicables a un archivo de
# stopwords de un token; sus componentes ya son stopwords de spaCy o quedan
# como lemas de contenido).
FRASES_MULETILLA: tuple[str, ...] = (
    "o sea", "no sé", "bueno pues", "digamos que", "es decir", "a ver",
)


def fecha_iso() -> str:
    """Devuelve la fecha/hora actual en formato ISO 8601 (UTC)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def contar_lineas(ruta: Path) -> int:
    """Cuenta las lineas completas de un archivo (0 si no existe)."""
    if not ruta.exists():
        return 0
    with ruta.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def truncar_a_n_lineas(ruta: Path, n: int) -> None:
    """Deja solo las primeras n lineas de un archivo (reescritura en temp)."""
    if not ruta.exists() or n <= 0:
        return
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    with ruta.open("r", encoding="utf-8") as fuente, temporal.open(
        "w", encoding="utf-8"
    ) as destino:
        for i, linea in enumerate(fuente):
            if i >= n:
                break
            destino.write(linea)
    temporal.replace(ruta)


def leer_jsonl(ruta: Path) -> list[dict[str, Any]]:
    """Lee un archivo JSONL completo y devuelve la lista de registros."""
    with ruta.open("r", encoding="utf-8") as fh:
        return [json.loads(linea) for linea in fh]


def es_anio(texto: str) -> bool:
    """True si el token es un anio del rango [1900, 2025] (4 digitos)."""
    return (
        len(texto) == 4
        and texto.isdigit()
        and CONFIG["min_anio"] <= int(texto) <= CONFIG["max_anio"]
    )


# ============================================================================
# SECCION 1: PARTE 1 - NORMALIZACION ESTRUCTURAL DE ENTREVISTAS (regex)
# ============================================================================


def _clasificar_bracket(contenido: str) -> str:
    """Clasifica el contenido de una anotacion [..] en un tipo canonico.

    Args:
        contenido: texto crudo entre corchetes (sin los corchetes).

    Returns:
        Tipo canonico: inc_dud_inaudible, pausa, corte, interrupcion,
        continuacion, emocion, audio, marca_hablante, timbre, vacio u otro.
    """
    c = contenido.strip()
    if not c:
        return "vacio"
    bajo = c.lower()
    if bajo.startswith(("inc:", "dud:", "inad", "inaudible", "indescifrable")):
        return "inc_dud_inaudible"
    if bajo.startswith("pausa"):
        return "pausa"
    if bajo.startswith(("corte", "cortad", "corto")):
        return "corte"
    if bajo.startswith(("interrup", "interru")):
        return "interrupcion"
    if bajo.startswith(("cont", "continua")):
        return "continuacion"
    if re.search(r"risa|llanto|silencio|suspiro|tose|tos|llor|grit|aplauso", bajo):
        return "emocion"
    if "audio" in bajo:
        return "audio"
    if re.fullmatch(r"(test|ent|x|e|p|inf|acomp|prueba)\d*", bajo):
        return "marca_hablante"
    if re.search(r"[0-9]{1,2}:[0-9]{2}", bajo):
        return "timbre"
    return "otro"


def _nuevos_contadores_normalizacion() -> dict[str, Any]:
    """Crea un dict de contadores para la normalizacion de una entrevista."""
    return {
        "chars_antes": 0,
        "chars_despues": 0,
        "n_codigos_lider": 0,
        "n_lineas_audio": 0,
        "n_codigos_embebidos": 0,
        "n_brackets": 0,
        "brackets_por_tipo": Counter(),
        "n_hablantes_linea": 0,
        "n_lineas_solo_etiqueta": 0,
        "n_etiquetas_aisladas": 0,
        "n_guiones": 0,
        "anonimizados": Counter(),
        "variantes_anonimizados": {},
        "ejemplos_brackets": {},
        "ejemplos_residuales": Counter(),
        "residuales_caps_digit": 0,
    }


def normalizar_texto_entrevista(texto: str, contadores: dict[str, Any]) -> str:
    """Aplica la normalizacion estructural de ADR-002/seccion (c) a un texto.

    Secuencia: codigo lider -> lineas AUDIO -> codigos embebidos -> corchetes
    -> etiquetas de hablante (linea, linea sola, aisladas) -> anonimizados
    (especificos y generico) -> guiones -> colapso de espacios. NO aplica
    lowercase ni lematizacion (eso va en la parte 2).

    Args:
        texto: transcripcion cruda de una entrevista.
        contadores: dict mutado con las estadisticas de la normalizacion
            (totales y por documento; ver normalizar_entrevistas_corpus).

    Returns:
        Texto normalizado en una unica linea (flujo de turnos unido).
    """
    t = texto.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")

    # (1) Codigo de sistema al inicio del documento.
    m = RE_CODIGO_LIDER.match(t)
    if m:
        contadores["n_codigos_lider"] += 1
        t = t[m.end():]

    # (2) Lineas AUDIO (marcadores de audio) -> linea vacia.
    lineas = t.split("\n")
    for i, linea in enumerate(lineas):
        s = linea.strip()
        if s and RE_LINEA_AUDIO.fullmatch(s):
            contadores["n_lineas_audio"] += 1
            lineas[i] = ""
    t = "\n".join(lineas)

    # (3) Codigos de sistema embebidos en el cuerpo (con timecode opcional).
    contadores["n_codigos_embebidos"] += len(list(RE_CODIGO_EMBEBIDO.finditer(t)))
    t = RE_CODIGO_EMBEBIDO.sub("", t)

    # (4) Anotaciones entre corchetes: eliminar marca y contenido (default
    # experimento 1, seccion (c)2); clasificar por tipo para el inventario.
    for m in RE_BRACKET.finditer(t):
        contadores["n_brackets"] += 1
        tipo = _clasificar_bracket(m.group(0)[1:-1])
        contadores["brackets_por_tipo"][tipo] += 1
        ejemplos = contadores["ejemplos_brackets"].setdefault(tipo, Counter())
        contenido = re.sub(r"\s+", " ", m.group(0)[1:-1].strip())
        ejemplos[contenido[:40]] += 1
    t = RE_BRACKET.sub("", t)

    # (5) Etiquetas de hablante con dos puntos al inicio de linea (se elimina
    # la etiqueta y el turno se une al flujo del texto, seccion (c)4).
    contadores["n_hablantes_linea"] += len(list(RE_LABEL_LINEA.finditer(t)))
    t = RE_LABEL_LINEA.sub("", t)

    # (6) Lineas que quedaron solo con la etiqueta (sin dos puntos).
    lineas = t.split("\n")
    for i, linea in enumerate(lineas):
        s = linea.strip()
        if s and RE_LINEA_SOLO_ETIQUETA.fullmatch(s):
            contadores["n_lineas_solo_etiqueta"] += 1
            lineas[i] = ""
    t = "\n".join(lineas)

    # (7) Tokens de etiqueta residuales en medio del texto. Si la etiqueta
    # iba entre comas ('bueno, TEST, gracias') queda ', ,': se colapsa.
    contadores["n_etiquetas_aisladas"] += len(list(RE_ETIQUETA_AISLADA.finditer(t)))
    t = RE_ETIQUETA_AISLADA.sub(" ", t)
    t = re.sub(r",[ \t]*,", ",", t)

    # (8) Anonimizados -> categorias canonicas (especificas primero y
    # generico al final, para no pisar las especificas).
    for categoria, patron in ANONIMIZADOS:
        variantes = contadores["variantes_anonimizados"].setdefault(categoria, Counter())
        for m in patron.finditer(t):
            contadores["anonimizados"][categoria] += 1
            variante = re.sub(r"\s+", " ", m.group(0).strip())
            variantes[variante] += 1
        t = patron.sub(categoria, t)

    # (9) Guiones separadores (>=3) -> un espacio.
    contadores["n_guiones"] += len(list(RE_GUIONES.finditer(t)))
    t = RE_GUIONES.sub(" ", t)

    # (10) Colapso de todo espacio en blanco a un unico espacio.
    t = re.sub(r"\s+", " ", t).strip()

    # Control: tokens en mayusculas con digito que aun sobreviven (deberian
    # ser ~0; se reportan para revision humana en el inventario).
    residuales = re.findall(
        r"(?<![A-ZÑÁÉÍÓÚa-zñáéíóú0-9])[A-ZÑÁÉÍÓÚ]{2,}[0-9]+(?![A-ZÑÁÉÍÓÚa-zñáéíóú0-9])",
        t,
    )
    contadores["residuales_caps_digit"] += len(residuales)
    for r_ in residuales[:20]:
        contadores["ejemplos_residuales"][r_] += 1
    return t


def normalizar_entrevistas_corpus(force: bool = False) -> dict[str, Any]:
    """Ejecuta la parte 1 completa sobre las 2.486 entrevistas crudas.

    Args:
        force: si True, regenera la salida aunque ya exista.

    Returns:
        Dict resumen con totales (tiempo, chars, conteos agregados).
    """
    rutas = CONFIG["rutas"]
    salida = rutas["entrevistas_normalizadas"]
    inventario = rutas["resultados_preprocesado"] / "inventario_patrones_entrevistas.json"

    if salida.exists() and not force:
        print(f"[normalizar] ya existe {salida.name}; usar --force para regenerar.")
        return {"skipped": True, "salida": str(salida)}

    rutas["resultados_preprocesado"].mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with rutas["raw_entrevistas"].open("r", encoding="utf-8") as fh:
        registros = json.load(fh)
    print(f"[normalizar] {len(registros)} registros cargados desde data/raw/.")

    totales = {
        "chars_antes": 0,
        "chars_despues": 0,
        "n_codigos_lider": 0,
        "n_lineas_audio": 0,
        "n_codigos_embebidos": 0,
        "n_brackets": 0,
        "n_hablantes_linea": 0,
        "n_lineas_solo_etiqueta": 0,
        "n_etiquetas_aisladas": 0,
        "n_guiones": 0,
        "residuales_caps_digit": 0,
        "brackets_por_tipo": Counter(),
        "anonimizados": Counter(),
        "variantes_anonimizados": {},
        "ejemplos_brackets": {},
        "ejemplos_residuales": Counter(),
    }
    por_doc: list[dict[str, Any]] = []
    CLAVES_NUMERICAS = (
        "chars_antes", "chars_despues", "n_codigos_lider", "n_lineas_audio",
        "n_codigos_embebidos", "n_brackets", "n_hablantes_linea",
        "n_lineas_solo_etiqueta", "n_etiquetas_aisladas", "n_guiones",
        "residuales_caps_digit",
    )

    with salida.open("w", encoding="utf-8") as fh:
        for i, reg in enumerate(registros):
            contadores = _nuevos_contadores_normalizacion()
            texto = reg.get("text", "")
            contadores["chars_antes"] = len(texto)
            texto_norm = normalizar_texto_entrevista(texto, contadores)
            contadores["chars_despues"] = len(texto_norm)
            fh.write(
                json.dumps(
                    {
                        "id_doc": reg.get("id_doc"),
                        "pages": reg.get("pages"),
                        "text_normalizado": texto_norm,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            for clave, valor in contadores.items():
                if isinstance(valor, Counter):
                    if clave in totales and isinstance(totales[clave], Counter):
                        totales[clave].update(valor)
                elif isinstance(valor, dict) and clave in (
                    "variantes_anonimizados", "ejemplos_brackets",
                ):
                    destino = totales.setdefault(clave, {})
                    for sub, ctr in valor.items():
                        destino.setdefault(sub, Counter()).update(ctr)
                elif clave in CLAVES_NUMERICAS:
                    totales[clave] += valor
            por_doc.append(
                {
                    "id_doc": reg.get("id_doc"),
                    "chars_antes": contadores["chars_antes"],
                    "chars_despues": contadores["chars_despues"],
                    "n_codigos_lider": contadores["n_codigos_lider"],
                    "n_lineas_audio": contadores["n_lineas_audio"],
                    "n_codigos_embebidos": contadores["n_codigos_embebidos"],
                    "n_brackets": contadores["n_brackets"],
                    "brackets_por_tipo": dict(contadores["brackets_por_tipo"]),
                    "n_hablantes_linea": contadores["n_hablantes_linea"],
                    "n_lineas_solo_etiqueta": contadores["n_lineas_solo_etiqueta"],
                    "n_etiquetas_aisladas": contadores["n_etiquetas_aisladas"],
                    "n_guiones": contadores["n_guiones"],
                    "anonimizados": dict(contadores["anonimizados"]),
                    "residuales_caps_digit": contadores["residuales_caps_digit"],
                }
            )
            if (i + 1) % 500 == 0:
                print(f"[normalizar] {i + 1}/{len(registros)} documentos...")

    duracion = time.time() - t0
    totales["brackets_por_tipo"] = {
        k: int(v)
        for k, v in sorted(totales["brackets_por_tipo"].items(), key=lambda x: -x[1])
    }
    totales["anonimizados"] = {
        k: int(v)
        for k, v in sorted(totales["anonimizados"].items(), key=lambda x: -x[1])
    }
    totales["variantes_anonimizados"] = {
        cat: {k: int(v) for k, v in ctr.most_common(60)}
        for cat, ctr in sorted(totales["variantes_anonimizados"].items())
    }
    totales["ejemplos_brackets"] = {
        tipo: {k: int(v) for k, v in ctr.most_common(20)}
        for tipo, ctr in sorted(totales["ejemplos_brackets"].items())
    }
    totales["ejemplos_residuales"] = {
        k: int(v) for k, v in totales["ejemplos_residuales"].most_common(30)
    }
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    inventario_doc = {
        "fecha": fecha_iso(),
        "fuente": str(rutas["raw_entrevistas"]),
        "n_docs": len(registros),
        "totales": totales,
        "reduccion_pct": round(
            100.0 * (1.0 - totales["chars_despues"] / totales["chars_antes"]), 2
        ),
        "pico_ram_mb": round(rss_kb / 1024.0, 1),
        "por_doc": por_doc,
    }
    with inventario.open("w", encoding="utf-8") as fh:
        json.dump(inventario_doc, fh, ensure_ascii=False, indent=1)

    print(
        f"[normalizar] {len(registros)} docs en {duracion:.1f}s | "
        f"chars {totales['chars_antes']:,} -> {totales['chars_despues']:,} "
        f"({inventario_doc['reduccion_pct']}%) | "
        f"pico RAM {inventario_doc['pico_ram_mb']} MB"
    )
    return {
        "skipped": False,
        "duracion_s": duracion,
        "n_docs": len(registros),
        "totales": totales,
        "reduccion_pct": inventario_doc["reduccion_pct"],
        "pico_ram_mb": inventario_doc["pico_ram_mb"],
    }


# ============================================================================
# SECCION 2: MULETILLAS (evidencia en muestra de 30 entrevistas) y archivo
# ============================================================================


def _contar_muletillas_muestra(
    textos: list[str],
) -> tuple[dict[str, int], dict[str, int], int]:
    """Cuenta candidatos a muletilla en la muestra fija de entrevistas.

    Args:
        textos: lista de textos normalizados de la muestra.

    Returns:
        Tupla (ocurrencias_totales, docs_presentes, n_tokens_totales).
    """
    total = {c: 0 for c in MULETILLAS_CANDIDATAS}
    por_doc = {c: 0 for c in MULETILLAS_CANDIDATAS}
    n_tokens = 0
    for texto in textos:
        palabras = RE_PALABRA.findall(texto.lower())
        n_tokens += len(palabras)
        presentes = set(palabras)
        for c in MULETILLAS_CANDIDATAS:
            total[c] += palabras.count(c)
            if c in presentes:
                por_doc[c] += 1
    return total, por_doc, n_tokens


def _decision_muletilla(
    token: str, total: int, docs: int, en_spacy: bool
) -> tuple[str, str]:
    """Decide si un candidato entra en config/stopwords_habla.txt.

    Regla documentada (calibrada con la evidencia de la muestra):
        1. Si ya es stopword de spaCy -> no hace falta (se reporta).
        2. Evidencia de frecuencia: docs >= 5 (de 30) y total >= 20.
        3. Interjeccion pura del nucleo con presencia minima (total >= 2).
        4. Resto -> excluida con motivo (el EDA de la Fase 5 congelara).

    Args:
        token: candidato.
        total: ocurrencias en la muestra de 30 documentos.
        docs: documentos de la muestra donde aparece.
        en_spacy: True si spacy.lang.es.STOP_WORDS ya lo cubre.

    Returns:
        Tupla (decision, motivo).
    """
    if en_spacy:
        return "ya_en_spacy", "cubierta por spacy.lang.es.STOP_WORDS"
    cfg = CONFIG
    if docs >= cfg["min_docs_muletilla"] and total >= cfg["min_total_muletilla"]:
        return "incluir", (
            f"evidencia: docs={docs}/30 >= {cfg['min_docs_muletilla']} y "
            f"total={total} >= {cfg['min_total_muletilla']}"
        )
    if token in NUCLEO_INTERJECCIONES and total >= cfg["min_total_nucleo"]:
        return "incluir", (
            f"nucleo de interjecciones con total={total} >= {cfg['min_total_nucleo']}"
        )
    return "excluir", "no supera umbrales de evidencia en la muestra"


def generar_stopwords_habla() -> dict[str, Any]:
    """Genera config/stopwords_habla.txt y la evidencia en CSV (muestra 30).

    La muestra fija (semilla 42) se toma de las entrevistas NORMALIZADAS, por
    lo que requiere la parte 1. El CSV evidencia ademas las frases
    multipalabra ('o sea', 'no sé'...) que no aplican como stopword de un
    token, para que el EDA (Fase 5) decida la lista final.

    Returns:
        Dict resumen (incluidas, ruta del archivo, frases multipalabra).
    """
    rutas = CONFIG["rutas"]
    norm_path = rutas["entrevistas_normalizadas"]
    if not norm_path.exists():
        raise FileNotFoundError(
            f"No existe {norm_path}. Ejecuta primero: python scripts/02_preprocesar.py normalizar"
        )
    rutas["resultados_preprocesado"].mkdir(parents=True, exist_ok=True)

    from spacy.lang.es import STOP_WORDS

    stopwords_spacy = set(STOP_WORDS)

    registros = leer_jsonl(norm_path)
    semilla = CONFIG["semilla_muletillas"]
    rng = random.Random(semilla)
    indices = sorted(rng.sample(range(len(registros)), CONFIG["n_muestra_muletillas"]))
    textos = [registros[i]["text_normalizado"] for i in indices]

    totales, docs_pres, n_tokens = _contar_muletillas_muestra(textos)

    filas = []
    incluidos: list[str] = []
    for token in MULETILLAS_CANDIDATAS:
        total = totales[token]
        docs = docs_pres[token]
        en_spacy = token in stopwords_spacy
        decision, motivo = _decision_muletilla(token, total, docs, en_spacy)
        tasa = total * 100_000.0 / n_tokens if n_tokens else 0.0
        filas.append(
            {
                "token": token,
                "ocurrencias_muestra": total,
                "docs_presentes_30": docs,
                "tasa_por_100k": round(tasa, 1),
                "en_stopwords_spacy": en_spacy,
                "decision": decision,
                "motivo": motivo,
            }
        )
        if decision == "incluir":
            incluidos.append(token)

    csv_path = rutas["resultados_preprocesado"] / "muletillas_muestra.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)

    frases = []
    for frase in FRASES_MULETILLA:
        ocurrencias = 0
        for texto in textos:
            ocurrencias += len(re.findall(re.escape(frase), texto.lower()))
        frases.append({"frase": frase, "ocurrencias_muestra": ocurrencias})

    # Archivo de stopwords (un token por linea, con cabecera de procedencia).
    lineas_txt = [
        "# stopwords_habla.txt - muletillas del habla colombiana transcrita",
        "# Generado el " + fecha_iso() + " por scripts/02_preprocesar.py (Fase 4, ADR-003).",
        "# Evidencia: muestra fija de 30 entrevistas normalizadas (semilla 42).",
        "# Regla: docs>=5/30 y total>=20, o interjeccion pura del nucleo con total>=2.",
        "# Nota: este/o/sea/no/pues/bueno/entonces ya estan en spacy.lang.es.STOP_WORDS.",
        "# La lista FINAL se congela en el EDA (Fase 5); borderline documentado en el CSV.",
    ]
    lineas_txt.extend(incluidos)
    rutas["stopwords_habla"].write_text(
        "\n".join(lineas_txt) + "\n", encoding="utf-8"
    )

    print(
        f"[muletillas] muestra={len(textos)} docs, {n_tokens:,} tokens | "
        f"{len(incluidos)} incluidas -> {rutas['stopwords_habla'].name}"
    )
    return {
        "incluidas": incluidos,
        "n_muestra": len(textos),
        "n_tokens_muestra": n_tokens,
        "csv": str(csv_path),
        "config": str(rutas["stopwords_habla"]),
        "frases_multipalabra": frases,
    }


def cargar_stopwords_habla() -> set[str]:
    """Carga las muletillas de config/stopwords_habla.txt (ignora cabeceras '#').

    Si el archivo no existe, devuelve vacio (los libros no requieren
    muletillas y el pipeline unico funciona igual).

    Returns:
        Set de tokens muletilla.
    """
    ruta = CONFIG["rutas"]["stopwords_habla"]
    if not ruta.exists():
        print("[spacy] aviso: no existe stopwords_habla.txt; solo stopwords de spaCy.")
        return set()
    resultado = set()
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#"):
            resultado.add(linea)
    return resultado


# ============================================================================
# SECCION 3: PARTE 2 - PREPROCESADO LINGUISTICO spaCy
# ============================================================================


def construir_nlp() -> Any:
    """Carga el pipeline spaCy es_core_news_md sin parser ni ner.

    El modelo es_core_news_md (v3.8) usa tok2vec+morphologizer+attribute_ruler
    para el POS (no tiene tagger): el lematizador espanol necesita el POS y la
    concordancia de genero/numero, por eso se conservan tok2vec, morphologizer,
    attribute_ruler y lemmatizer, y solo se descartan parser y ner (no aportan
    a la lematizacion). Verificado: deshabilitar attribute_ruler deja el POS
    vacio y empeora las lemas.

    Returns:
        Objeto Language de spaCy listo para nlp.pipe.
    """
    import spacy

    return spacy.load(CONFIG["modelo_spacy"], disable=["parser", "ner"])


def filtrar_doc(doc: Any, muletillas: set[str]) -> tuple[list[str], list[str], dict[str, int]]:
    """Aplica los filtros linguisticos de ADR-003 a un Doc ya procesado.

    Para cada token no-espacio se emite su forma en minusculas al archivo de
    tokens crudos (EDA). Para el archivo de lemas se aplica: categorias
    canonicas de ADR-002 y anios 1900-2025 se conservan como un solo token;
    numeros sueltos, tokens no alfabeticos, stopwords de spaCy, muletillas y
    tokens de 1 caracter se eliminan.

    Args:
        doc: documento ya corrido por el pipeline spaCy (nlp.pipe).
        muletillas: conjunto extra de stopwords del habla (config).

    Returns:
        Tupla (tokens_crudos, lemas, conteos) con los conteos por motivo de
        eliminacion: stopword, muletilla, no_alfa, num_no_anio, corto.
    """
    tokens_crudos: list[str] = []
    lemas: list[str] = []
    conteos = {
        "raw": 0, "lemas": 0, "stopword": 0, "muletilla": 0,
        "no_alfa": 0, "num_no_anio": 0, "corto": 0,
    }
    for tok in doc:
        if tok.is_space:
            continue
        bajo = tok.text.lower()
        tokens_crudos.append(bajo)
        conteos["raw"] += 1

        if bajo in CATEGORIAS_CANONICAS:
            lemas.append(bajo)
            conteos["lemas"] += 1
            continue
        if es_anio(tok.text):
            lemas.append(tok.text)
            conteos["lemas"] += 1
            continue
        if tok.text.isdigit():
            conteos["num_no_anio"] += 1
            continue
        if not tok.text.isalpha():
            conteos["no_alfa"] += 1
            continue
        if tok.is_stop:
            conteos["stopword"] += 1
            continue
        if bajo in muletillas:
            conteos["muletilla"] += 1
            continue
        lem = tok.lemma_.lower() if tok.lemma_ else bajo
        if len(lem) <= 1:
            conteos["corto"] += 1
            continue
        lemas.append(lem)
        conteos["lemas"] += 1
    return tokens_crudos, lemas, conteos


def _cargar_corpus(nombre: str) -> list[dict[str, Any]]:
    """Carga los documentos de un corpus con sus metadatos.

    Args:
        nombre: 'libros' (segmentos_libros.jsonl) o 'entrevistas'
            (entrevistas_normalizadas.jsonl; requiere la parte 1).

    Returns:
        Lista de dicts con 'id', 'texto' y 'meta' (metadatos propios).
    """
    rutas = CONFIG["rutas"]
    if nombre == "libros":
        registros = leer_jsonl(rutas["segmentos_libros"])
        return [
            {
                "id": r.get("id_segmento"),
                "texto": r.get("texto", ""),
                "meta": {
                    "tomo": r.get("tomo"),
                    "titulo": r.get("titulo"),
                    "nivel_toc": r.get("nivel_toc"),
                    "pagina_inicio": r.get("pagina_inicio"),
                    "pagina_fin": r.get("pagina_fin"),
                },
            }
            for r in registros
        ]
    if nombre == "entrevistas":
        ruta = rutas["entrevistas_normalizadas"]
        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe {ruta}. Ejecuta primero: python scripts/02_preprocesar.py normalizar"
            )
        registros = leer_jsonl(ruta)
        return [
            {"id": r.get("id_doc"), "texto": r.get("text_normalizado", ""),
             "meta": {"pages": r.get("pages")}}
            for r in registros
        ]
    raise ValueError(f"corpus desconocido: {nombre}")


def _resumen_estadisticas(estadisticas: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrega las estadisticas por documento en un resumen de saneamiento."""
    if not estadisticas:
        return {}
    raw = sum(e["n_raw"] for e in estadisticas)
    lemas = sum(e["n_lemas"] for e in estadisticas)
    ratios = [e["ratio"] for e in estadisticas if e["n_raw"] > 0]
    motivos = Counter()
    for e in estadisticas:
        motivos.update(e["eliminados"])
    return {
        "n_docs": len(estadisticas),
        "tokens_crudos": raw,
        "lemas_emitidos": lemas,
        "eliminados_pct": round(100.0 * (1.0 - lemas / raw), 2) if raw else 0.0,
        "ratio_lemas_tokens_medio": round(sum(ratios) / len(ratios), 4) if ratios else 0.0,
        "docs_sin_lemas": sum(1 for e in estadisticas if e["n_lemas"] == 0),
        "eliminados_por_motivo": {k: int(v) for k, v in motivos.items()},
        "lemas_min_max": (
            min((e["n_lemas"] for e in estadisticas), default=0),
            max((e["n_lemas"] for e in estadisticas), default=0),
        ),
    }


def _escribir_docs_json(
    nombre: str, docs: list[dict[str, Any]], sufijo: str = ""
) -> None:
    """Escribe el indice de metadatos alineado por posicion con los .txt."""
    ruta = CONFIG["rutas"]["preprocesado"] / f"{sufijo}{nombre}_docs.json"
    contenido = {
        "corpus": nombre,
        "n_docs": len(docs),
        "docs": [{"id": d["id"], **d["meta"]} for d in docs],
    }
    ruta.write_text(json.dumps(contenido, ensure_ascii=False, indent=1), encoding="utf-8")


def preprocesar_corpus_spacy(
    nombre: str,
    nlp: Any,
    muletillas: set[str],
    *,
    desde: int = 0,
    force: bool = False,
    muestra: int = 0,
    n_procesos: int | None = None,
) -> dict[str, Any]:
    """Ejecuta la parte 2 sobre un corpus con checkpoint por documento.

    Produccion (muestra=0): escribe <corpus>_lemas.txt y <corpus>_tokens.txt
    en data/corpus/preprocesado/, reanudable: si el archivo ya tiene L lineas
    completas y faltan documentos, continua en L; si L == total, avisa y
    termina; --desde N reanuda desde el documento N (trunca el archivo si ya
    tiene mas lineas; si tiene menos, continua en L porque no puede haber
    huecos); --force borra y empieza de cero. Validacion (muestra>0):
    procesa una muestra fija (semilla fija) y escribe en
    validacion_<corpus>_*.txt sin tocar la produccion, y reporta velocidad +
    extrapolacion del tiempo total del corpus.

    Args:
        nombre: 'libros' o 'entrevistas'.
        nlp: pipeline spaCy cargado.
        muletillas: stopwords extra del habla.
        desde: indice del documento por el que reanudar (0 = automatico).
        force: regenerar desde cero ignorando el checkpoint.
        muestra: tamano de la muestra de validacion (0 = produccion).
        n_procesos: workers de spaCy (default: CONFIG['n_procesos']).

    Returns:
        Dict resumen (docs procesados, tiempo, velocidad, saneamiento).
    """
    n_procesos = n_procesos or CONFIG["n_procesos"]
    docs_corpus = _cargar_corpus(nombre)
    total_corpus = len(docs_corpus)
    chars_corpus = sum(len(d["texto"]) for d in docs_corpus)
    preproc = CONFIG["rutas"]["preprocesado"]
    preproc.mkdir(parents=True, exist_ok=True)
    CONFIG["rutas"]["resultados_preprocesado"].mkdir(parents=True, exist_ok=True)

    es_validacion = muestra > 0
    sufijo = "validacion_" if es_validacion else ""
    if es_validacion:
        rng = random.Random(CONFIG["semilla_validacion"])
        indices = sorted(rng.sample(range(total_corpus), min(muestra, total_corpus)))
        docs = [docs_corpus[i] for i in indices]
        total = len(docs)
    else:
        docs = docs_corpus
        total = total_corpus

    ruta_lemas = preproc / f"{sufijo}{nombre}_lemas.txt"
    ruta_tokens = preproc / f"{sufijo}{nombre}_tokens.txt"

    ruta_docs = preproc / f"{sufijo}{nombre}_docs.json"
    if not es_validacion:
        existentes = contar_lineas(ruta_lemas)
        if existentes >= total and not force:
            if not ruta_docs.exists():
                _escribir_docs_json(nombre, docs_corpus)
            print(f"[spacy] {nombre}: ya completo ({existentes}/{total} docs).")
            return {"skipped": True, "corpus": nombre}
        if force:
            existentes = 0
            desde = 0
        if 0 < desde < existentes:
            print(f"[spacy] {nombre}: truncando a {desde} docs (--desde).")
            truncar_a_n_lineas(ruta_lemas, desde)
            truncar_a_n_lineas(ruta_tokens, desde)
            existentes = desde
        inicio = desde if desde > 0 else existentes
        if desde > existentes:
            # El archivo tiene menos lineas que el punto pedido: no puede
            # haber huecos, se continua donde estan las lineas completas.
            inicio = existentes
    else:
        inicio = 0
        if ruta_lemas.exists():
            ruta_lemas.unlink()
        if ruta_tokens.exists():
            ruta_tokens.unlink()

    if inicio >= total:
        print(f"[spacy] {nombre}: no hay documentos pendientes (inicio={inicio}).")
        return {"skipped": True, "corpus": nombre}

    print(
        f"[spacy] {nombre}: {total} docs | chars {chars_corpus:,} | "
        f"procesando desde {inicio} | {'validacion' if es_validacion else 'produccion'}"
    )
    textos = (d["texto"] for d in docs[inicio:])
    modo_escritura = "a" if (inicio > 0 and not es_validacion) else "w"
    batch = CONFIG["batch_entrevistas"] if nombre == "entrevistas" else CONFIG["batch_libros"]

    estadisticas: list[dict[str, Any]] = []
    t0 = time.time()
    n_chars = 0
    with ruta_lemas.open(modo_escritura, encoding="utf-8") as fh_lemas, ruta_tokens.open(
        modo_escritura, encoding="utf-8"
    ) as fh_tokens:
        for pos, doc in enumerate(
            nlp.pipe(textos, batch_size=batch, n_process=n_procesos)
        ):
            crudos, lemas_doc, conteos = filtrar_doc(doc, muletillas)
            fh_lemas.write(" ".join(lemas_doc) + "\n")
            fh_tokens.write(" ".join(crudos) + "\n")
            fh_lemas.flush()
            fh_tokens.flush()
            n_chars += len(doc.text)
            estadisticas.append(
                {
                    "idx": inicio + pos,
                    "id": docs[inicio + pos]["id"],
                    "n_raw": conteos["raw"],
                    "n_lemas": conteos["lemas"],
                    "ratio": (
                        round(conteos["lemas"] / conteos["raw"], 4)
                        if conteos["raw"] else 0.0
                    ),
                    "eliminados": {
                        k: v for k, v in conteos.items() if k not in ("raw", "lemas")
                    },
                }
            )
            if (pos + 1) % 200 == 0:
                print(f"[spacy] {nombre}: {inicio + pos + 1}/{total} docs...")

    duracion = time.time() - t0
    n_procesados = len(estadisticas)
    vel_chars = n_chars / duracion if duracion > 0 else 0.0
    resumen = _resumen_estadisticas(estadisticas)
    resumen.update(
        {
            "corpus": nombre,
            "modo": "validacion" if es_validacion else "produccion",
            "desde": inicio,
            "docs_procesados": n_procesados,
            "duracion_s": round(duracion, 1),
            "chars_procesados": n_chars,
            "velocidad_chars_s": round(vel_chars, 0),
            "velocidad_docs_s": round(n_procesados / duracion, 2) if duracion > 0 else 0.0,
            "extrapolacion_total_s": (
                round(chars_corpus / vel_chars, 0)
                if es_validacion and vel_chars > 0 else None
            ),
        }
    )

    nombre_stats = (
        f"sanidad_{nombre}.json" if not es_validacion else f"sanidad_{nombre}_validacion.json"
    )
    ruta_stats = CONFIG["rutas"]["resultados_preprocesado"] / nombre_stats
    ruta_stats.write_text(
        json.dumps({"resumen": resumen, "por_doc": estadisticas},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    if not es_validacion and inicio + n_procesados >= total_corpus:
        _escribir_docs_json(nombre, docs_corpus)
    elif es_validacion:
        _escribir_docs_json(nombre, docs, sufijo="validacion_")

    extrapolacion = resumen["extrapolacion_total_s"]
    extra_txt = (
        f" | extrapolacion total {extrapolacion}s" if extrapolacion is not None else ""
    )
    print(
        f"[spacy] {nombre} ({resumen['modo']}): {n_procesados} docs en "
        f"{duracion:.1f}s | {vel_chars / 1e6:.2f} Mchars/s | "
        f"lemas/tokens medio {resumen['ratio_lemas_tokens_medio']}{extra_txt}"
    )
    return {"skipped": False, "corpus": nombre, "resumen": resumen}


# ============================================================================
# SECCION 4: CLI
# ============================================================================


def _construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos de la linea de comandos."""
    parser = argparse.ArgumentParser(
        prog="02_preprocesar.py",
        description="Fase 4 TALLER-4: normalizacion (parte 1) y preprocesado "
                    "linguistico spaCy (parte 2) de libros y entrevistas CEV.",
    )
    parser.add_argument(
        "--solo", choices=["libros", "entrevistas"], default=None,
        help="atajo: libros -> parte 2 libros; entrevistas -> parte 1 + parte 2",
    )
    sub = parser.add_subparsers(dest="comando")

    p_norm = sub.add_parser("normalizar", help="solo parte 1 (entrevistas)")
    p_norm.add_argument("--force", action="store_true",
                        help="regenerar aunque exista la salida")

    p_spacy = sub.add_parser("spacy", help="solo parte 2 (linguistico)")
    p_spacy.add_argument("--corpus", choices=["libros", "entrevistas", "ambos"],
                         default="ambos", help="corpus a procesar (default: ambos)")
    p_spacy.add_argument("--force", action="store_true",
                         help="ignorar checkpoint y empezar de cero")
    p_spacy.add_argument("--desde", type=int, default=0,
                         help="reanudar desde el documento N (0-based)")
    p_spacy.add_argument("--muestra", type=int, default=0,
                         help="corrida de validacion sobre N docs (no toca produccion)")
    p_spacy.add_argument("--procesos", type=int, default=CONFIG["n_procesos"],
                         help="workers de spaCy (default: 8)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada principal con manejo de errores."""
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass  # metodo ya fijado por el entorno
    parser = _construir_parser()
    args = parser.parse_args(argv)
    n_procesos = args.procesos if args.comando == "spacy" else CONFIG["n_procesos"]
    desde = args.desde if args.comando == "spacy" else 0

    try:
        if args.solo == "libros":
            nlp = construir_nlp()
            preprocesar_corpus_spacy("libros", nlp, cargar_stopwords_habla(),
                                     desde=desde, n_procesos=n_procesos)
        elif args.solo == "entrevistas":
            normalizar_entrevistas_corpus(force=False)
            generar_stopwords_habla()
            nlp = construir_nlp()
            preprocesar_corpus_spacy("entrevistas", nlp, cargar_stopwords_habla(),
                                     desde=desde, n_procesos=n_procesos)
        elif args.comando == "normalizar":
            normalizar_entrevistas_corpus(force=args.force)
        elif args.comando == "spacy":
            if not CONFIG["rutas"]["stopwords_habla"].exists() and CONFIG["rutas"][
                "entrevistas_normalizadas"
            ].exists():
                generar_stopwords_habla()
            nlp = construir_nlp()
            if args.corpus in ("libros", "ambos"):
                preprocesar_corpus_spacy("libros", nlp, cargar_stopwords_habla(),
                                         desde=desde, force=args.force,
                                         muestra=args.muestra,
                                         n_procesos=n_procesos)
            if args.corpus in ("entrevistas", "ambos"):
                preprocesar_corpus_spacy("entrevistas", nlp, cargar_stopwords_habla(),
                                         desde=desde, force=args.force,
                                         muestra=args.muestra,
                                         n_procesos=n_procesos)
        else:
            # Sin subcomando: pipeline completo (parte 1 + parte 2 ambos).
            normalizar_entrevistas_corpus(force=False)
            generar_stopwords_habla()
            nlp = construir_nlp()
            preprocesar_corpus_spacy("libros", nlp, cargar_stopwords_habla())
            preprocesar_corpus_spacy("entrevistas", nlp, cargar_stopwords_habla())
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        return 1
    print("[fin] 02_preprocesar.py terminado sin errores.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
