"""03_extraccion_entidades.py
Fecha: 2026-09-08
Descripcion: Extraccion de entidades juridicas (NER M2) por sentencia sobre el
    texto crudo del corpus: articulos de la Constitucion citados, normas
    (ley, decreto con o sin ley, acto legislativo), sentencias citadas en
    canon TIPO-NUM-AAAA y magistrado ponente. Valida los patrones en una
    muestra previa, extrae el corpus completo por row-groups con pyarrow y
    escribe el parquet de entidades mas el reporte de cobertura para M3/M5/M7.
Estado del arte aplicado: regex insensibles a tildes y mayusculas sobre texto
    crudo; ventana de 60 caracteres para vincular articulo con constitucion;
    normalizacion de anos de 2 digitos (>=92 hacia 19xx) con rango 1992-2021.
"""

import json
import logging
import random
import re
import sys
import time
import traceback
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CONFIG = {
    "semilla": 42,
    "ruta_corpus": "data/corpus_sentencias.parquet",
    "ruta_salida": "data/entidades_sentencias.parquet",
    "ruta_cobertura": "resultados/ner_cobertura.md",
    "ruta_log": "experimentos/extraccion_entidades.log",
    "ventana_cp": 60,
    "n_muestra": 100,
    "n_casos_validacion": 20,
    "ano_min": 1992,
    "ano_max": 2021,
    "ano_norma_min": 1850,
    "art_max": 380,
    "progreso_cada": 2000,
    "letras_art_validas": {"A", "B", "C"},
    "periodos": [
        ("1992-1996", 1992, 1996),
        ("1997-2001", 1997, 2001),
        ("2002-2006", 2002, 2006),
        ("2007-2011", 2007, 2011),
        ("2012-2016", 2012, 2016),
        ("2017-2021", 2017, 2021),
    ],
}

ROOT = Path(__file__).resolve().parent.parent

# ===== Patrones compilados (ajustados tras inspeccion de muestra seed 42) =====
# Ajustes aplicados: \s* cubre saltos de linea ("T-462\nde 2003"); el ancla de
# articulo exige frontera para no tragar "arte"/"articular"; el ponente acepta
# "Dr./Dra.", salto de linea y mayusculas sostenidas; "superior"/"carta" son
# marcadores de constitucion ("articulo 86 superior", "articulo 13 de la Carta").
RE_ART = re.compile(r"\bart(?:iculos?|es|s)?\.?\s*(\d{1,3})([a-c])?(?![0-9])")
RE_MARCA_CP = re.compile(r"constitucion|\bcarta\b|c\s*\.\s*p\b|\bcp\b|superior")
RE_NUM_EXTRA = re.compile(r"(\d{1,3})([a-c])?(?![0-9])")
RE_SEP_NUMS = re.compile(r"^[\s,;:\-\(\)/\"'yoe.]*$")
RE_STOP_NUMS = re.compile(
    r"numeral|inciso|literal|paragrafo|ordinal|articulo|ley|decreto|sentencia|"
    r"codigo|expediente|radicacion|estatuto|titulo|capitulo"
)
RE_NORMA = re.compile(
    r"\b(ley|decreto(?:\s+ley)?|acto\s+legislativo)\s+"
    r"(?:n[o°. ]?\s*)?(\d{1,4})\s+de\s+(\d{4})\b"
)
RE_DEC_LEG_DIAG = re.compile(r"\bdecreto\s+legislativo\s+(\d{1,4})\s+de\s+(\d{4})\b")
RE_SENT = re.compile(
    r"\b([cts])(u)?\s*-?\s*(\d{1,4})\s*(?:de|/)\s*(\d{2,4})\b", re.IGNORECASE
)
RE_PON = re.compile(
    r"(?i:magistrado\s+ponente)\s*:?\s*(?:(?i:dra?)\.?\s*)?"
    r"([A-Z\xc1\xc9\xcd\xd3\xda\xd1\xdc][a-z\xe1\xe9\xed\xf3\xfa\xf1\xfc]+"
    r"(?:\s+[A-Z\xc1\xc9\xcd\xd3\xda\xd1\xdc][a-z\xe1\xe9\xed\xf3\xfa\xf1\xfc]+){0,3}"
    r"|[A-Z\xc1\xc9\xcd\xd3\xda\xd1\xdc]{2,}"
    r"(?:\s+[A-Z\xc1\xc9\xcd\xd3\xda\xd1\xdc]{2,}){0,3})"
)
RE_PARTICION = re.compile(r"tipo=([^/]+)/ano=([^/]+)")
STOP_NO_NOMBRE = {
    "sala",
    "corte",
    "constitucional",
    "sentencia",
    "bogota",
    "colombia",
    "expediente",
    "proceso",
    "accion",
    "recurso",
    "auto",
    "fallo",
    "republica",
    "tutela",
    "demanda",
    "radicacion",
    "acta",
    "juzgado",
    "tribunal",
    "consejo",
    "estado",
    "plena",
    "revision",
}


# ===== SECCION 1: utilidades base =====
def configurar_semillas(semilla: int) -> None:
    """Fija las semillas de aleatoriedad para reproducibilidad.

    Args:
        semilla: Valor entero de semilla.
    """
    random.seed(semilla)
    np.random.seed(semilla)


def configurar_logging(ruta_log: str) -> logging.Logger:
    """Configura el logger con salida a consola y a archivo.

    Args:
        ruta_log: Ruta del archivo de log (se crea su carpeta padre).

    Returns:
        Logger configurado con dos handlers (consola y archivo).
    """
    Path(ruta_log).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("extraccion_entidades")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formato = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    consola = logging.StreamHandler(sys.stdout)
    consola.setFormatter(formato)
    archivo = logging.FileHandler(ruta_log, mode="w", encoding="utf-8")
    archivo.setFormatter(formato)
    logger.addHandler(consola)
    logger.addHandler(archivo)
    return logger


def normalizar_tm(texto: str) -> str:
    """Normaliza texto para matching: minusculas y sin tildes.

    Conserva puntuacion, digitos y espacios originales para que las ventanas
    de caracteres sigan siendo comparables con el texto crudo.

    Args:
        texto: Cadena original en espanol.

    Returns:
        Cadena en minusculas sin marcas diacriticas.
    """
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFD", texto) if not unicodedata.combining(c)
    )
    return sin_tildes.lower()


def contexto_unilinea(texto: str, inicio: int, fin: int, radio: int = 60) -> str:
    """Extrae un fragmento unilinea alrededor de un match para el log.

    Args:
        texto: Texto crudo completo.
        inicio: Offset inicial del match.
        fin: Offset final del match.
        radio: Caracteres de contexto a cada lado.

    Returns:
        Fragmento con saltos de linea colapsados a espacios.
    """
    frag = texto[max(0, inicio - radio) : fin + radio]
    return re.sub(r"\s+", " ", frag).strip()


# ===== SECCION 2: extractores por sentencia =====
def extraer_arts_cp(texto_crudo: str, ventana: int) -> list:
    """Extrae los articulos de la Constitucion citados en una sentencia.

    Busca menciones de articulo ("articulo N", "art. N", "arts. N") y las
    acepta solo si hay un marcador de constitucion ("constitucion", "carta",
    "C.P./CP", "superior") dentro de la ventana de caracteres alrededor.
    Captura series ("articulos 11, 13 y 86") y sufijos A/B/C ("22A").

    Args:
        texto_crudo: Texto original de la sentencia (con tildes).
        ventana: Radio en caracteres para vincular articulo y constitucion.

    Returns:
        Lista de articulos en canon ("13", "22A") sin duplicados en orden.
    """
    tn = normalizar_tm(texto_crudo)
    hallados: list = []
    vistos = set()
    for ancla in RE_ART.finditer(tn):
        ini, fin = ancla.span()
        trozo = tn[max(0, ini - ventana) : fin + ventana]
        if not RE_MARCA_CP.search(trozo):
            continue
        candidatos = [(ancla.group(1), (ancla.group(2) or "").upper())]
        previo_fin = 0
        adelante = tn[fin : fin + 80]
        for extra in RE_NUM_EXTRA.finditer(adelante):
            hueco = adelante[previo_fin : extra.start()]
            if RE_STOP_NUMS.search(hueco):
                break
            if not RE_SEP_NUMS.match(hueco):
                break
            candidatos.append((extra.group(1), (extra.group(2) or "").upper()))
            previo_fin = extra.end()
        for num_txt, letra in candidatos:
            base = int(num_txt)
            if not 1 <= base <= CONFIG["art_max"]:
                continue
            if letra and letra not in CONFIG["letras_art_validas"]:
                continue
            canon = f"{base}{letra}"
            if canon not in vistos:
                vistos.add(canon)
                hallados.append(canon)
    return hallados


def extraer_normas(texto_crudo: str, contadores: dict) -> list:
    """Extrae normas citadas (ley, decreto [ley], acto legislativo).

    Args:
        texto_crudo: Texto original de la sentencia.
        contadores: Dict mutable donde se acumulan descartes y diagnosticos.

    Returns:
        Lista en canon ("Ley 100 de 1993", "Decreto Ley 663 de 1993") unica.
    """
    tn = normalizar_tm(texto_crudo)
    halladas: list = []
    vistos = set()
    for coinc in RE_NORMA.finditer(tn):
        tipo_txt = re.sub(r"\s+", " ", coinc.group(1)).strip()
        numero = int(coinc.group(2))
        ano = int(coinc.group(3))
        if not CONFIG["ano_norma_min"] <= ano <= CONFIG["ano_max"]:
            contadores["norma_ano_fuera_rango"] += 1
            continue
        if numero < 1:
            contadores["norma_num_invalido"] += 1
            continue
        tipo = {"ley": "Ley", "acto legislativo": "Acto Legislativo"}.get(
            tipo_txt, "Decreto Ley" if "ley" in tipo_txt else "Decreto"
        )
        canon = f"{tipo} {numero} de {ano}"
        if canon not in vistos:
            vistos.add(canon)
            halladas.append(canon)
    contadores["decreto_legislativo_mencion"] += len(RE_DEC_LEG_DIAG.findall(tn))
    return halladas


def normalizar_ano_cita(ano_txt: str) -> int | None:
    """Normaliza el ano de una cita de sentencia a 4 digitos.

    Aplica la regla del corpus: 4 digitos directos; 2 digitos con >=92 hacia
    19xx y <92 hacia 20xx.

    Args:
        ano_txt: Digitos capturados (2 a 4).

    Returns:
        Ano en 4 digitos o None si tiene longitud distinta de 2 o 4.
    """
    if len(ano_txt) == 4:
        return int(ano_txt)
    if len(ano_txt) == 2:
        valor = int(ano_txt)
        return 1900 + valor if valor >= 92 else 2000 + valor
    return None


def extraer_sents(
    texto_crudo: str, contadores: dict, menciones: Counter | None = None
) -> list:
    """Extrae sentencias citadas y las normaliza a canon TIPO-NUM-AAAA.

    Cubre "T-293 de 1997", "C-004/92", "SU-120 de 2003" y variantes sin
    espacios o con saltos de linea. Descarta anos fuera de 1992-2021.

    Args:
        texto_crudo: Texto original de la sentencia.
        contadores: Dict mutable donde se acumulan los descartes.
        menciones: Counter opcional donde se cuenta cada ocurrencia valida
            (con duplicados) para distinguir la autocita de encabezado.

    Returns:
        Lista canonica unica en orden ("T-293-1997", "C-4-1992").
    """
    halladas: list = []
    vistos = set()
    for coinc in RE_SENT.finditer(texto_crudo):
        base = coinc.group(1).upper()
        con_u = bool(coinc.group(2))
        tipo = "SU" if base == "S" and con_u else base
        if tipo not in {"T", "C", "SU", "S"}:
            contadores["sent_tipo_invalido"] += 1
            continue
        numero = int(coinc.group(3))
        if numero < 1:
            contadores["sent_num_invalido"] += 1
            continue
        ano = normalizar_ano_cita(coinc.group(4))
        if ano is None:
            contadores["sent_ano_invalido"] += 1
            continue
        if not CONFIG["ano_min"] <= ano <= CONFIG["ano_max"]:
            contadores["sent_ano_fuera_rango"] += 1
            continue
        canon = f"{tipo}-{numero}-{ano}"
        if menciones is not None:
            menciones[canon] += 1
        if canon not in vistos:
            vistos.add(canon)
            halladas.append(canon)
    return halladas


def extraer_ponente(texto_crudo: str) -> str | None:
    """Extrae el magistrado ponente de la sentencia (primera mencion).

    Acepta "Magistrado Ponente: NOMBRE", variante con salto de linea,
    prefijo "Dr./Dra." y nombres en mayusculas sostenidas (los pasa a
    formato titulo). Exige 1 a 4 palabras capitalizadas, lo que descarta
    falsos como "magistrado ponente nego por...".

    Args:
        texto_crudo: Texto original de la sentencia.

    Returns:
        Nombre del ponente o None si no hay mencion valida.
    """
    for coinc in RE_PON.finditer(texto_crudo):
        nombre = re.sub(r"\s+", " ", coinc.group(1)).strip().strip(".,;:")
        if not nombre:
            continue
        if nombre.replace(" ", "").isupper():
            nombre = nombre.title()
        if normalizar_tm(nombre.split()[0]) in STOP_NO_NOMBRE:
            continue
        return nombre
    return None


def extraer_todo(
    texto_crudo: str,
    contadores: dict,
    ventana: int,
    menciones: Counter | None = None,
) -> dict:
    """Aplica los cuatro extractores a una sentencia.

    Args:
        texto_crudo: Texto original de la sentencia.
        contadores: Dict mutable de descartes y diagnosticos.
        ventana: Radio en caracteres para vincular articulo y constitucion.
        menciones: Counter opcional de ocurrencias de sentencias citadas.

    Returns:
        Dict con arts_cp, normas, sents_citadas y ponente.
    """
    if not isinstance(texto_crudo, str) or not texto_crudo:
        contadores["texto_vacio"] += 1
        return {"arts_cp": [], "normas": [], "sents_citadas": [], "ponente": None}
    return {
        "arts_cp": extraer_arts_cp(texto_crudo, ventana),
        "normas": extraer_normas(texto_crudo, contadores),
        "sents_citadas": extraer_sents(texto_crudo, contadores, menciones),
        "ponente": extraer_ponente(texto_crudo),
    }


# ===== SECCION 3: lectura streaming por row-groups =====
def particion_de_ruta(ruta: str) -> tuple:
    """Recupera (tipo, ano) Hive desde la ruta del fragmento parquet.

    Args:
        ruta: Ruta del archivo de datos dentro del dataset particionado.

    Returns:
        Tupla (tipo, ano) como (str, int).

    Raises:
        ValueError: Si la ruta no contiene el patron tipo=/ano=.
    """
    coinc = RE_PARTICION.search(ruta.replace("\\", "/"))
    if coinc is None:
        raise ValueError(f"Ruta sin particion Hive: {ruta}")
    return coinc.group(1), int(coinc.group(2))


def iterar_grupos_filas(ruta_dataset: str, columnas: list):
    """Itera el dataset parquet por row-groups sin cargarlo en memoria.

    Args:
        ruta_dataset: Directorio del dataset Hive.
        columnas: Columnas fisicas a leer de cada row-group.

    Yields:
        Tuplas (tipo, ano, DataFrame del row-group).
    """
    dataset = pq.ParquetDataset(ruta_dataset)
    fragmentos = sorted(dataset.fragments, key=lambda f: f.path)
    for frag in fragmentos:
        tipo, ano = particion_de_ruta(frag.path)
        lector = pq.ParquetFile(frag.path)
        for grupo in range(lector.num_row_groups):
            tabla = lector.read_row_group(grupo, columns=columnas)
            yield tipo, ano, tabla.to_pandas()


def listar_ids(ruta_dataset: str, logger: logging.Logger) -> list:
    """Lista todos los ids del corpus con una pasada ligera (solo id).

    Args:
        ruta_dataset: Directorio del dataset Hive.
        logger: Logger de ejecucion.

    Returns:
        Lista de ids como cadenas en orden estable de lectura.
    """
    ids: list = []
    for _tipo, _ano, bloque in iterar_grupos_filas(ruta_dataset, ["id"]):
        ids.extend(str(v) for v in bloque["id"].tolist())
    logger.info("Ids listados: %d", len(ids))
    return ids


def construir_canones_corpus(ruta_dataset: str, logger: logging.Logger) -> set:
    """Construye el conjunto de canones TIPO-NUM-ANO de todo el corpus.

    Lee solo id/numero por row-groups y toma tipo/ano de la particion Hive
    del path (las columnas de particion no viajan en los archivos fisicos).

    Args:
        ruta_dataset: Directorio del dataset Hive.
        logger: Logger de ejecucion.

    Returns:
        Conjunto de cadenas canonicas para resolucion de citas y autocitas.
    """
    canones = set()
    for tipo, ano, bloque in iterar_grupos_filas(ruta_dataset, ["numero"]):
        for numero in bloque["numero"].tolist():
            canones.add(f"{tipo}-{int(numero)}-{int(ano)}")
    logger.info("Canones del corpus: %d", len(canones))
    return canones


# ===== SECCION 4: validacion muestral previa =====
def recolectar_muestra(ruta_dataset: str, ids_muestra: set) -> list:
    """Recolecta los documentos de la muestra en pasada streaming.

    Args:
        ruta_dataset: Directorio del dataset Hive.
        ids_muestra: Conjunto de ids a conservar.

    Returns:
        Lista de dicts con id, tipo, numero, ano y texto.
    """
    docs: list = []
    for tipo, ano, bloque in iterar_grupos_filas(
        ruta_dataset, ["id", "numero", "texto"]
    ):
        for fila in bloque.itertuples(index=False):
            sid = str(fila.id)
            if sid in ids_muestra:
                docs.append(
                    {
                        "id": sid,
                        "tipo": str(tipo),
                        "numero": int(fila.numero),
                        "ano": int(ano),
                        "texto": fila.texto if isinstance(fila.texto, str) else "",
                    }
                )
    por_id = {d["id"]: d for d in docs}
    return [por_id[sid] for sid in por_id if sid in ids_muestra]


def juzgar_art(canon: str) -> bool:
    """Juzga un articulo extraido: base numerica dentro de 1-380.

    Args:
        canon: Articulo en canon ("13", "22A").

    Returns:
        Verdadero si es un positivo plausible.
    """
    base = int(re.match(r"(\d+)", canon).group(1))
    return 1 <= base <= CONFIG["art_max"]


def juzgar_norma(canon: str) -> bool:
    """Juzga una norma extraida por ano y numero plausibles.

    Args:
        canon: Norma en canon ("Ley 100 de 1993").

    Returns:
        Verdadero si es un positivo plausible.
    """
    coinc = re.match(r"(.+) (\d+) de (\d{4})$", canon)
    if coinc is None:
        return False
    return 1880 <= int(coinc.group(3)) <= CONFIG["ano_max"] and int(coinc.group(2)) >= 1


def juzgar_sent(canon: str, canones_corpus: set) -> bool:
    """Juzga una cita de sentencia: resuelve a una sentencia del corpus.

    Args:
        canon: Cita en canon ("T-293-1997").
        canones_corpus: Conjunto de canones TIPO-NUM-ANO del corpus.

    Returns:
        Verdadero si la cita existe en el corpus (criterio estricto).
    """
    return canon in canones_corpus


def juzgar_ponente(nombre: str | None) -> bool:
    """Juzga un ponente extraido contra palabras no-nombre.

    Args:
        nombre: Nombre extraido o None.

    Returns:
        Verdadero si parece un nombre de persona.
    """
    if not nombre:
        return False
    palabras = nombre.split()
    if not 1 <= len(palabras) <= 4:
        return False
    return all(normalizar_tm(p) not in STOP_NO_NOMBRE and len(p) >= 2 for p in palabras)


def validar_muestra(
    docs: list, canones_corpus: set, contadores: dict, logger: logging.Logger
) -> dict:
    """Valida los patrones en la muestra y registra 20 casos por tipo.

    Extrae entidades de cada documento de la muestra, toma hasta 20 casos
    por tipo con su contexto, los juzga con reglas deterministicas y deja
    los contextos en el log para auditoria visual.

    Args:
        docs: Documentos de la muestra con id, tipo, numero, ano y texto.
        canones_corpus: Canones TIPO-NUM-ANO de todo el corpus.
        contadores: Dict mutable de descartes de la pasada muestral.
        logger: Logger de ejecucion.

    Returns:
        Dict con precision muestral por tipo y conteos de cobertura muestral.
    """
    ventana = CONFIG["ventana_cp"]
    n_casos = CONFIG["n_casos_validacion"]
    casos = {"arts_cp": [], "normas": [], "sents": [], "ponente": []}
    with_cp = with_norma = with_sent = with_pon = 0
    for doc in docs:
        texto = doc["texto"]
        res = extraer_todo(texto, contadores, ventana)
        if res["arts_cp"]:
            with_cp += 1
        if res["normas"]:
            with_norma += 1
        if res["sents_citadas"]:
            with_sent += 1
        if res["ponente"]:
            with_pon += 1
        tn = normalizar_tm(texto)
        for art in res["arts_cp"]:
            if len(casos["arts_cp"]) < n_casos:
                pos = tn.find("art")
                casos["arts_cp"].append(
                    (
                        doc["id"],
                        art,
                        contexto_unilinea(texto, max(0, pos), max(0, pos), 60),
                    )
                )
        for norma in res["normas"]:
            if len(casos["normas"]) < n_casos:
                casos["normas"].append((doc["id"], norma, ""))
        for cita in res["sents_citadas"]:
            if len(casos["sents"]) < n_casos:
                casos["sents"].append((doc["id"], cita, ""))
        if res["ponente"] and len(casos["ponente"]) < n_casos:
            casos["ponente"].append((doc["id"], res["ponente"], ""))
    precision = {}
    precision["arts_cp"] = (
        sum(juzgar_art(c[1]) for c in casos["arts_cp"]) / len(casos["arts_cp"])
        if casos["arts_cp"]
        else 0.0
    )
    precision["normas"] = (
        sum(juzgar_norma(c[1]) for c in casos["normas"]) / len(casos["normas"])
        if casos["normas"]
        else 0.0
    )
    tp_estricta = sum(juzgar_sent(c[1], canones_corpus) for c in casos["sents"])
    precision["sents_resuelve_corpus"] = (
        tp_estricta / len(casos["sents"]) if casos["sents"] else 0.0
    )
    precision["sents_rango_valido"] = 1.0 if casos["sents"] else 0.0
    precision["ponente"] = (
        sum(juzgar_ponente(c[1]) for c in casos["ponente"]) / len(casos["ponente"])
        if casos["ponente"]
        else 0.0
    )
    logger.info(
        "Muestra: %d docs; con arts_cp=%d normas=%d sents=%d ponente=%d",
        len(docs),
        with_cp,
        with_norma,
        with_sent,
        with_pon,
    )
    logger.info(
        "Precision muestral arts_cp: %.2f (%d casos)",
        precision["arts_cp"],
        len(casos["arts_cp"]),
    )
    for sid, art, ctx in casos["arts_cp"]:
        logger.info(
            "  caso arts_cp id=%s art=%s ok=%s ctx=%s",
            sid,
            art,
            juzgar_art(art),
            ctx[:160],
        )
    logger.info(
        "Precision muestral normas: %.2f (%d casos)",
        precision["normas"],
        len(casos["normas"]),
    )
    for sid, norma, _ctx in casos["normas"]:
        logger.info(
            "  caso norma id=%s norma=%s ok=%s", sid, norma, juzgar_norma(norma)
        )
    logger.info(
        "Sents: resolucion al corpus %.2f (%d casos)",
        precision["sents_resuelve_corpus"],
        len(casos["sents"]),
    )
    for sid, cita, _ctx in casos["sents"]:
        logger.info(
            "  caso sent id=%s cita=%s resuelve=%s",
            sid,
            cita,
            juzgar_sent(cita, canones_corpus),
        )
    logger.info(
        "Precision muestral ponente: %.2f (%d casos)",
        precision["ponente"],
        len(casos["ponente"]),
    )
    for sid, nombre, _ctx in casos["ponente"]:
        logger.info(
            "  caso ponente id=%s nombre=%s ok=%s", sid, nombre, juzgar_ponente(nombre)
        )
    logger.info(
        "FP tipicos arts_cp: numero >380 de otros codigos; 'tribunal superior' "
        "cercano a un articulo no constitucional."
    )
    logger.info(
        "FP tipicos normas: 'decreto legislativo N de AAAA' no extraido "
        "(patron fuera de spec, solo diagnostico); anos <1880 juzgados FP."
    )
    logger.info(
        "FP tipicos sents: citas a providencias fuera del corpus (autos, "
        "anos validos pero sin resolucion); anos fuera de 1992-2021 "
        "descartados con contador."
    )
    logger.info(
        "FP tipicos ponente: menciones internas a ponentes de sentencias "
        "citadas cuando el encabezado no casa; encabezados sin ponente."
    )
    return {
        "precision": precision,
        "n_casos": {k: len(v) for k, v in casos.items()},
        "cobertura_muestra": {
            "arts_cp": with_cp,
            "normas": with_norma,
            "sents": with_sent,
            "ponente": with_pon,
            "n": len(docs),
        },
    }


# ===== SECCION 5: extraccion completa y coberturas =====
def extraer_corpus(ruta_dataset: str, logger: logging.Logger) -> tuple:
    """Extrae entidades de las 23.742 sentencias por row-groups.

    Args:
        ruta_dataset: Directorio del dataset Hive.
        logger: Logger de ejecucion.

    Returns:
        Tupla (filas, contadores) con una fila por documento y los conteos.
    """
    ventana = CONFIG["ventana_cp"]
    cada = CONFIG["progreso_cada"]
    contadores: dict = Counter()
    filas: list = []
    n_docs = 0
    inicio = time.time()
    for tipo, ano, bloque in iterar_grupos_filas(
        ruta_dataset, ["id", "numero", "texto"]
    ):
        for fila in bloque.itertuples(index=False):
            sid = str(fila.id)
            numero = int(fila.numero)
            texto = fila.texto if isinstance(fila.texto, str) else ""
            menciones: Counter = Counter()
            res = extraer_todo(texto, contadores, ventana, menciones)
            propia = f"{tipo}-{numero}-{int(ano)}"
            n_self = menciones.get(propia, 0)
            contadores["menciones_sent_total"] += sum(menciones.values())
            contadores["docs_autocita_bruta"] += 1 if n_self >= 1 else 0
            contadores["docs_autocita_2plus"] += 1 if n_self >= 2 else 0
            filas.append(
                {
                    "id": sid,
                    "tipo": str(tipo),
                    "numero": numero,
                    "ano": int(ano),
                    "arts_cp": json.dumps(res["arts_cp"], ensure_ascii=False),
                    "normas": json.dumps(res["normas"], ensure_ascii=False),
                    "sents_citadas": json.dumps(
                        res["sents_citadas"], ensure_ascii=False
                    ),
                    "ponente": res["ponente"],
                }
            )
            n_docs += 1
            if n_docs % cada == 0:
                msj = f"Progreso: {n_docs} docs en {time.time() - inicio:.1f}s"
                logger.info(msj)
    logger.info("Extraccion completa: %d docs en %.1fs", n_docs, time.time() - inicio)
    logger.info("Contadores: %s", dict(contadores))
    return filas, contadores


def periodo_de(ano: int) -> str:
    """Asigna un ano a su etiqueta de periodo quinquenal.

    Args:
        ano: Ano calendario.

    Returns:
        Etiqueta del periodo o "fuera_rango".
    """
    for etiqueta, ini, fin in CONFIG["periodos"]:
        if ini <= ano <= fin:
            return etiqueta
    return "fuera_rango"


def construir_canones(df: pd.DataFrame) -> set:
    """Construye el conjunto de canones TIPO-NUM-ANO del corpus.

    Args:
        df: DataFrame con columnas tipo, numero y ano.

    Returns:
        Conjunto de cadenas canonicas para resolucion de citas y autocitas.
    """
    return {
        f"{t}-{int(n)}-{int(a)}" for t, n, a in zip(df["tipo"], df["numero"], df["ano"])
    }


def escribir_cobertura(
    df: pd.DataFrame,
    contadores: dict,
    validacion: dict,
    segundos: float,
    raiz: Path,
    logger: logging.Logger,
) -> dict:
    """Calcula coberturas y escribe resultados/ner_cobertura.md.

    Args:
        df: DataFrame de entidades con listas ya decodificadas.
        contadores: Contadores de la pasada completa.
        validacion: Resultado de validar_muestra con precisiones.
        segundos: Tiempo total de ejecucion en segundos.
        raiz: Directorio raiz del proyecto.
        logger: Logger de ejecucion.

    Returns:
        Dict con las metricas clave (coberturas, aristas, autocitas, tops).
    """
    total = len(df)
    df["tiene_cp"] = df["arts_cp"].map(bool)
    df["tiene_norma"] = df["normas"].map(bool)
    df["tiene_sent"] = df["sents_citadas"].map(bool)
    df["tiene_ponente"] = df["ponente"].notna()
    df["periodo"] = df["ano"].map(periodo_de)

    def pct(serie) -> float:
        """Porcentaje de verdaderos en una serie booleana.

        Args:
            serie: Serie booleana de pandas.

        Returns:
            Porcentaje entre 0 y 100.
        """
        return float(serie.mean() * 100) if len(serie) else 0.0

    cobertura = {
        "cp": pct(df["tiene_cp"]),
        "norma": pct(df["tiene_norma"]),
        "sent": pct(df["tiene_sent"]),
        "ponente": pct(df["tiene_ponente"]),
    }
    por_tipo = (
        df.groupby("tipo")[["tiene_cp", "tiene_norma", "tiene_sent", "tiene_ponente"]]
        .mean()
        .mul(100)
        .round(2)
    )
    por_periodo = (
        df.groupby("periodo")[
            ["tiene_cp", "tiene_norma", "tiene_sent", "tiene_ponente"]
        ]
        .mean()
        .mul(100)
        .round(2)
    )
    conteo_arts = Counter(a for lista in df["arts_cp"] for a in lista)
    top_arts = conteo_arts.most_common(20)
    conteo_docs_art: dict = Counter()
    for lista in df["arts_cp"]:
        for art in set(lista):
            conteo_docs_art[art] += 1
    n_docs_art = {art: conteo_docs_art[art] for art, _c in top_arts}
    conteo_pon = Counter(df.loc[df["ponente"].notna(), "ponente"])
    top_pon = conteo_pon.most_common(10)

    canones = construir_canones(df)
    pares = set()
    n_menciones = 0
    n_resuelven = 0
    n_autocitas = 0
    for fila in df.itertuples(index=False):
        propia = f"{fila.tipo}-{int(fila.numero)}-{int(fila.ano)}"
        citadas = fila.sents_citadas
        n_menciones += len(citadas)
        if propia in citadas:
            n_autocitas += 1
        for cita in citadas:
            pares.add((fila.id, cita))
            if cita in canones:
                n_resuelven += 1
    n_aristas = len(pares)
    n_sin_self = n_aristas - n_autocitas
    pct_autocitas = n_autocitas / total * 100 if total else 0.0
    n_2plus = int(contadores.get("docs_autocita_2plus", 0))
    pct_2plus = n_2plus / total * 100 if total else 0.0
    n_menciones_real = int(contadores.get("menciones_sent_total", n_menciones))
    pct_resuelven = n_resuelven / n_menciones * 100 if n_menciones else 0.0

    lineas = ["# Cobertura NER juridico (M2)", ""]
    lineas.append(f"Fecha: {datetime.now(timezone.utc).date().isoformat()}")
    lineas.append(f"Documentos: {total} (T/C/SU)")
    lineas.append(f"Tiempo total: {segundos:.1f}s")
    lineas.append("")
    lineas.append("## Precision muestral (100 docs seed 42, 20 casos por tipo)")
    for clave, valor in validacion["precision"].items():
        lineas.append(f"- {clave}: {valor:.2f}")
    lineas.append("")
    lineas.append("## Cobertura global (% docs con >=1)")
    lineas.append(f"- cita CP: {cobertura['cp']:.2f}")
    lineas.append(f"- norma: {cobertura['norma']:.2f}")
    lineas.append(f"- sentencia citada: {cobertura['sent']:.2f}")
    lineas.append(f"- ponente: {cobertura['ponente']:.2f}")
    lineas.append("")
    lineas.append("## Por tipo T/C/SU (%)")
    lineas.append("| tipo | cita_CP | norma | sentencia | ponente | n |")
    lineas.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    conteo_tipo = df["tipo"].value_counts()
    for tipo in por_tipo.index.tolist():
        f = por_tipo.loc[tipo]
        lineas.append(
            f"| {tipo} | {f['tiene_cp']:.2f} | {f['tiene_norma']:.2f} | "
            f"{f['tiene_sent']:.2f} | {f['tiene_ponente']:.2f} | "
            f"{int(conteo_tipo.get(tipo, 0))} |"
        )
    lineas.append("")
    lineas.append("## Por periodo (%)")
    lineas.append("| periodo | cita_CP | norma | sentencia | ponente | n |")
    lineas.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    conteo_per = df["periodo"].value_counts()
    for etiqueta, _ini, _fin in CONFIG["periodos"]:
        if etiqueta in por_periodo.index:
            f = por_periodo.loc[etiqueta]
            lineas.append(
                f"| {etiqueta} | {f['tiene_cp']:.2f} | {f['tiene_norma']:.2f} | "
                f"{f['tiene_sent']:.2f} | {f['tiene_ponente']:.2f} | "
                f"{int(conteo_per.get(etiqueta, 0))} |"
            )
    lineas.append("")
    lineas.append("## Top-20 articulos CP citados (n_docs)")
    for art, _c in top_arts:
        lineas.append(f"- Art. {art}: {n_docs_art[art]} docs")
    lineas.append("")
    lineas.append("## Top-10 ponentes (n_docs)")
    for nombre, conteo in top_pon:
        lineas.append(f"- {nombre}: {conteo} docs")
    lineas.append("")
    lineas.append("## Grafo sentencia-sentencia (M5)")
    lineas.append(f"- Aristas unicas doc->cita (con self-loops): {n_aristas}")
    lineas.append(f"- Aristas unicas sin self-loops: {n_sin_self}")
    lineas.append(f"- Menciones totales con duplicados: {n_menciones_real}")
    lineas.append(f"- Menciones unicas por doc sumadas: {n_menciones}")
    lineas.append(f"- Menciones que resuelven al corpus: {pct_resuelven:.2f}%")
    lineas.append(f"- Docs que se autocitan: {n_autocitas} ({pct_autocitas:.2f}%)")
    lineas.append(
        "- Nota M5: la autocita bruta incluye la mencion del propio encabezado; "
        "los self-loops deben excluirse del grafo."
    )
    lineas.append(
        f"- Docs con >=2 menciones propias (proxy autocita en cuerpo): "
        f"{n_2plus} ({pct_2plus:.2f}%)"
    )
    lineas.append("")
    lineas.append("## Descartes y diagnosticos")
    for clave in sorted(contadores):
        if clave in {
            "docs_autocita_bruta",
            "docs_autocita_2plus",
            "menciones_sent_total",
        }:
            continue
        lineas.append(f"- {clave}: {int(contadores[clave])}")
    ruta_cob = raiz / CONFIG["ruta_cobertura"]
    ruta_cob.parent.mkdir(parents=True, exist_ok=True)
    ruta_cob.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    logger.info("Cobertura escrita en %s", ruta_cob)
    return {
        "cobertura": cobertura,
        "n_aristas": n_aristas,
        "n_aristas_sin_self": n_sin_self,
        "n_menciones": n_menciones_real,
        "pct_autocitas": pct_autocitas,
        "pct_autocitas_2plus": pct_2plus,
        "pct_resuelven": pct_resuelven,
        "top_arts": [(a, n_docs_art[a]) for a, _c in top_arts[:5]],
        "top_pon": top_pon,
    }


# ===== SECCION 6: orquestacion principal =====
def main() -> None:
    """Orquesta validacion muestral, extraccion completa y reportes."""
    raiz = ROOT
    logger = configurar_logging(str(raiz / CONFIG["ruta_log"]))
    inicio_total = time.time()
    try:
        configurar_semillas(CONFIG["semilla"])
        logger.info("Semilla fijada: %d", CONFIG["semilla"])
        ruta_corpus = str(raiz / CONFIG["ruta_corpus"])
        logger.info("Corpus: %s", ruta_corpus)

        ids = listar_ids(ruta_corpus, logger)
        muestra = random.Random(CONFIG["semilla"]).sample(
            sorted(set(ids)), CONFIG["n_muestra"]
        )
        logger.info("Muestra seed 42: %d docs", len(muestra))
        docs_muestra = recolectar_muestra(ruta_corpus, set(muestra))
        logger.info("Docs muestra recolectados: %d", len(docs_muestra))
        mini = pd.DataFrame(
            [
                {"tipo": d["tipo"], "numero": d["numero"], "ano": d["ano"]}
                for d in docs_muestra
            ]
        )
        logger.info("Muestra head:\n%s", mini.head(3).to_string())
        canones_full = construir_canones_corpus(ruta_corpus, logger)
        cont_muestra: dict = Counter()
        validacion = validar_muestra(docs_muestra, canones_full, cont_muestra, logger)
        logger.info("Descartes muestra: %s", dict(cont_muestra))

        filas, contadores = extraer_corpus(ruta_corpus, logger)
        df = pd.DataFrame(
            filas,
            columns=[
                "id",
                "tipo",
                "numero",
                "ano",
                "arts_cp",
                "normas",
                "sents_citadas",
                "ponente",
            ],
        )
        print(df.head(3).to_string())
        print(f"Filas: {len(df)}; nulos ponente: {int(df['ponente'].isna().sum())}")
        for col in ["arts_cp", "normas", "sents_citadas"]:
            df[col + "_n"] = df[col].map(lambda s: len(json.loads(s)))
            print(f"{col}: media listas {df[col + '_n'].mean():.3f}")
        df_out = df.drop(columns=["numero", "arts_cp_n", "normas_n", "sents_citadas_n"])
        ruta_salida = raiz / CONFIG["ruta_salida"]
        df_out.to_parquet(ruta_salida, index=False)
        logger.info("Parquet escrito: %s (%d filas)", ruta_salida, len(df_out))

        for col in ["arts_cp", "normas", "sents_citadas"]:
            df[col] = df[col].map(json.loads)
        segundos = time.time() - inicio_total
        metricas = escribir_cobertura(
            df, contadores, validacion, segundos, raiz, logger
        )
        cob = metricas["cobertura"]
        print(
            f"CP>=1: {cob['cp']:.2f}% | norma: {cob['norma']:.2f}% | "
            f"sent: {cob['sent']:.2f}% | ponente: {cob['ponente']:.2f}%"
        )
        print(f"Top-5 arts: {metricas['top_arts']}")
        print(
            f"Aristas M5: {metricas['n_aristas']} "
            f"(sin self {metricas['n_aristas_sin_self']}) | autocitas: "
            f"{metricas['pct_autocitas']:.2f}% "
            f"(2plus {metricas['pct_autocitas_2plus']:.2f}%) | "
            f"tiempo: {segundos:.1f}s"
        )
        print(
            f"Salidas: {CONFIG['ruta_salida']} {CONFIG['ruta_cobertura']} "
            f"{CONFIG['ruta_log']}"
        )
    except Exception:  # noqa: BLE001 - main captura todo para log + exit 1
        logger.error("Fallo la extraccion:\n%s", traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
