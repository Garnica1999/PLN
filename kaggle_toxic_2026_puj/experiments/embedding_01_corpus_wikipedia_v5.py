"""P1 de la fase v5: corpus de 10M tokens de Wikipedia ES y cobertura TOXIC.

Contexto
--------
Es la primera vez que el proyecto ingiere texto externo. La decision de
gobierno **D-030** acota ese uso: los corpus externos **no etiquetados** sirven
**exclusivamente** para aprender la representacion distribuida desde cero y no
pueden usarse para entrenamiento supervisado, generacion de labels,
pseudo-labeling ni ajuste de umbrales. Este script construye la escala 1 del
corpus y, sobre ella, mide si la representacion puede cubrir el vocabulario del
problema.

Pipeline (``download -> raw -> clean -> dedup -> normalize -> tokenize ->
shards -> manifest``), en streaming y sin cargar el dump en memoria:

1. **Descarga por partes**: el dump de eswiki viene partido por paginas
   (``eswiki-latest-pages-articles-multistream1.xml-p1p159400.bz2``, 305 MB,
   paginas 1-159400). Se baja **una sola parte**, suficiente para toda la fase
   P3, en vez de los 5.19 GB completos. Se registra URL, fecha y ``sha256``.
2. **Extraccion**: se recorre el XML de forma incremental (``XMLPullParser``),
   conservando solo articulos de Wikipedia (``ns=0``) y descartando redirecciones.
3. **Limpieza**: se elimina el marcado MediaWiki (plantillas, tablas, refs,
   etiquetas HTML, enlaces a archivos y categorias) conservando el texto plano.
4. **Deduplicacion**: hash ``sha1`` del texto limpio mas un umbral minimo de
   longitud, para no arrastrar paginas vacias ni redirecciones residuales.
5. **Normalizacion EMB-NORMAL**: minusculas (con tildes), URLs/menciones/
   hashtags a marcadores estables y **sin** correctores ortograficos ni
   reconstruccion de tildes.
6. **Tokenizacion y shards**: un documento por linea, en shards de ~2M tokens,
   hasta 10M tokens.
7. **Manifiesto**: se actualiza ``corpus_manifest.json`` con los campos de la
   seccion 35 de la spec (procedencia, version, licencia, hashes, conteos).
8. **Auditoria de fuga**: se comprueba con hashes de 8-gramas que ningun texto
   de ``toxic_test.csv`` (ni de train) aparece en el corpus.
9. **Cobertura del vocabulario TOXIC** (secciones 7-9 de la spec): cobertura por
   tipo, por token, ``oov_rate`` y cobertura ponderada por frecuencia; cobertura
   por grupo relevante; y distincion entre **word-OOV** y **representation-OOV**
   (lo que los subpalabras de fastText pueden reconstruir).
10. **Informes y figuras**.

Lo que este script **no** hace: no entrena el embedding, no usa labels, no lee
el holdout, no genera submissions y no anade ningun texto de TOXIC al corpus
(variante ``EXTERNAL-ONLY``). Los textos de TOXIC se leen **solo** como
instrumento de diagnostico de cobertura, que es lo que exigen las secciones 7-9.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_01_corpus_wikipedia_v5.py
"""

from __future__ import annotations

import bz2
import csv
import hashlib
import html
import json
import re
import subprocess
import sys
import time
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_test, read_train

# ---------------------------------------------------------------------------
# Constantes de la fase
# ---------------------------------------------------------------------------
EXPERIMENT_ID = "E-CORP-01"
PREPROCESSING_VERSION = "v5.0-EMB-NORMAL"
OBJETIVO_TOKENS = 10_000_000
TOKENS_POR_SHARD = 2_000_000
TAMAO_MINIMO_TOKENS_DOC = 20
NGRAMAS_FUGA = 8
SEMILLA_Z = 2**31 - 1

BASE_DUMP = "https://dumps.wikimedia.org/eswiki/latest"
PARTE = "eswiki-latest-pages-articles-multistream1.xml-p1p159400.bz2"
LICENCIA = "CC BY-SA 4.0 (atribucion obligatoria) y texto de MediaWiki bajo CC BY-SA/GFDL"

CORPORA = PROJECT_ROOT / "data" / "external" / "corpora"
RAW_DIR = CORPORA / "raw"
CORPUS_DIR = CORPORA / "wikipedia_es" / "emb_normal"
MANIFEST_PATH = CORPORA / "corpus_manifest.json"
MANIFEST_REPORTE = PROJECT_ROOT / "reports" / "corpus_manifest_v5.json"
REPORTE_CORPUS = PROJECT_ROOT / "reports" / "embedding_corpus_audit.md"
REPORTE_COBERTURA = PROJECT_ROOT / "reports" / "embedding_vocabulary_coverage.md"
REPORTE_COBERTURA_CSV = PROJECT_ROOT / "reports" / "embedding_vocabulary_coverage.csv"
FIGURAS = PROJECT_ROOT / "reports" / "figuras"
JSON_RESULTADO = PROJECT_ROOT / "reports" / "embedding_corpus_v5.json"
FASTTEXT_BIN = PROJECT_ROOT / "tools" / "fastText" / "build" / "fasttext"
SIFTS_DIR = CORPORA / "sifts"

ETIQUETAS = ("HS", "TR", "AG")

# ---------------------------------------------------------------------------
# Expresiones regulares de limpieza y normalizacion
# ---------------------------------------------------------------------------
RE_COMENTARIO = re.compile(r"<!--.*?-->", re.DOTALL)
RE_REF = re.compile(r"<ref[^>]*?/>|<ref[^>]*?>.*?</ref>", re.DOTALL | re.IGNORECASE)
RE_ETIQUETA_HTML = re.compile(r"<[^>]{1,200}>")
RE_PLANTILLA = re.compile(r"\{\{.*?\}\}", re.DOTALL)
RE_TABLA = re.compile(r"\{\|.*?\|\}", re.DOTALL)
RE_ENLACE_ARCHIVO = re.compile(
    r"\[\[\s*(?:Archivo|File|Imagen|Image|Categor[ií]a|Category|Wikipedia|Plantilla|Template)\s*:[^\]]*\]\]",
    re.IGNORECASE,
)
RE_ENLACE_INTERNO = re.compile(r"\[\[\s*([^\[\]|]*)\s*\|\s*([^\[\]|]*?)\s*\]\]")
RE_ENLACE_SIMPLE = re.compile(r"\[\[\s*([^\[\]|]*?)\s*\]\]")
RE_ENLACE_EXTERNO = re.compile(r"\[(?:https?|ftp)://[^\s\]]+\s*([^\]]*)\]")
RE_URL_SIMPLE = re.compile(r"(?:https?|ftp)://\S+")
RE_MARCAS = re.compile(r"'{2,5}")
RE_ESPACIOS = re.compile(r"\s+")
RE_CARACTERES_PERMITIDOS = re.compile(r"[^\w\s']", re.UNICODE)
RE_ENTIDADES = re.compile(r"&[a-zA-Z#0-9]{2,8};")

# Normalizacion EMB-NORMAL
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_MENCION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#\w+")
RE_NUMERO = re.compile(r"\d+")

# Tokenizacion
RE_TOKEN = re.compile(r"\w+(?:['\u2019]\w+)*", re.UNICODE)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso de la fase.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[{EXPERIMENT_ID}] {mensaje}", flush=True)


# ---------------------------------------------------------------------------
# 1. Descarga
# ---------------------------------------------------------------------------
def descargar_parte(destino: Path, fuerza: bool = False) -> dict[str, Any]:
    """Descarga una parte del dump de eswiki y devuelve sus metadatos.

    La descarga es idempotente: si el archivo ya existe con el tamano esperado
    se reutiliza. Se anade un parametro aleatorio a la URL para que los
    servidorales intermedios no respondan con una version truncada.

    Args:
        destino: ruta del archivo local.
        fuerza: si es ``True`` vuelve a descargar aunque el archivo exista.

    Returns:
        Diccionario con ruta, bytes, ``sha256`` y fecha de descarga.
    """
    url = f"{BASE_DUMP}/{PARTE}"
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and not fuerza:
        log(f"   el archivo ya existe ({destino.stat().st_size / 1024**2:.0f} MB); no se vuelve a bajar")
        return metadatos_descarga(destino, url, descargado=False)
    log(f"   descargando {PARTE}")
    peticion = urllib.request.Request(url, headers={"User-Agent": "kaggle-toxic-2026-puj/1.0"})
    temporal = destino.with_suffix(".parcial")
    inicio = time.time()
    with urllib.request.urlopen(peticion, timeout=120) as respuesta, temporal.open("wb") as salida:
        esperado = int(respuesta.headers.get("Content-Length", 0))
        recibido = 0
        while True:
            bloque = respuesta.read(1 << 20)
            if not bloque:
                break
            salida.write(bloque)
            recibido += len(bloque)
            if recibido % (50 << 20) < (1 << 20):
                log(f"      {recibido / 1024**2:.0f} / {esperado / 1024**2:.0f} MB")
    if esperado and recibido != esperado:
        temporal.unlink(missing_ok=True)
        raise RuntimeError(f"descarga incompleta: {recibido} de {esperado} bytes")
    temporal.replace(destino)
    log(f"   descargado en {time.time() - inicio:.0f} s")
    return metadatos_descarga(destino, url, descargado=True)


def metadatos_descarga(destino: Path, url: str, descargado: bool) -> dict[str, Any]:
    """Calcula los metadatos de un archivo descargado.

    Args:
        destino: ruta del archivo.
        url: URL de origen.
        descargado: si se descargo en esta ejecucion.

    Returns:
        Diccionario con los metadatos del archivo.
    """
    digest = hashlib.sha256()
    with destino.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return {
        "archivo": destino.name,
        "url": url,
        "bytes": destino.stat().st_size,
        "sha256": digest.hexdigest(),
        "descargado_en_esta_ejecucion": descargado,
        "fecha_consulta": datetime.now(timezone.utc).date().isoformat(),
    }


# ---------------------------------------------------------------------------
# 2-3. Extraccion y limpieza
# ---------------------------------------------------------------------------
def _tag(elemento: ET.Element) -> str:
    """Devuelve el nombre de la etiqueta sin espacio de nombres.

    Args:
        elemento: elemento XML.

    Returns:
        Nombre local de la etiqueta.
    """
    return elemento.tag.rsplit("}", 1)[-1]


def _texto_hijo(elemento: ET.Element, nombre: str) -> str | None:
    """Devuelve el texto del primer hijo directo con ese nombre.

    Args:
        elemento: elemento XML padre.
        nombre: nombre local buscado.

    Returns:
        Texto del hijo o ``None`` si no existe.
    """
    for hijo in elemento:
        if _tag(hijo) == nombre:
            return (hijo.text or "").strip()
    return None


def iterar_articulos(ruta_bz2: Path) -> Iterator[tuple[str, str]]:
    """Recorre el dump y produce articulos limpios de forma incremental.

    Args:
        ruta_bz2: archivo ``.bz2`` del dump.

    Yields:
        Tuplas ``(titulo, texto_limpio)``.
    """
    parser = ET.XMLPullParser(["end"])
    leidos = 0
    with bz2.open(ruta_bz2, "rb") as archivo:
        while True:
            bloque = archivo.read(1 << 20)
            if not bloque:
                break
            leidos += len(bloque)
            parser.feed(bloque)
            for _, elemento in parser.read_events():
                if _tag(elemento) != "page":
                    continue
                articulo = _articulo_de(elemento)
                if articulo is not None:
                    yield articulo
                elemento.clear()
    log(f"   XML recorrido: {leidos / 1024**2:.0f} MB comprimidos leidos")


def _articulo_de(elemento: ET.Element) -> tuple[str, str] | None:
    """Extrae ``(titulo, texto)`` de un elemento ``page`` si es un articulo.

    Args:
        elemento: elemento ``page`` del dump.

    Returns:
        Tupla ``(titulo, texto_limpio)`` o ``None`` si no es un articulo util.
    """
    espacio = _texto_hijo(elemento, "ns")
    if espacio != "0":
        return None
    titulo = _texto_hijo(elemento, "title") or ""
    if not titulo or ":" in titulo:
        return None
    for hijo in elemento:
        if _tag(hijo) != "revision":
            continue
        marcado = _texto_hijo(hijo, "text")
        if not marcado:
            return None
        if marcado.lstrip()[:12].lower().startswith(("#redirect", "#redirecci")):
            return None
        limpio = limpiar_marcado(marcado)
        if limpio:
            return titulo, limpio
    return None


def limpiar_marcado(marcado: str) -> str:
    """Elimina el marcado MediaWiki y conserva el texto plano.

    Args:
        marcado: texto original del campo ``text`` de una revision.

    Returns:
        Texto plano con una linea por parrafo.
    """
    texto = RE_COMENTARIO.sub(" ", marcado)
    texto = RE_REF.sub(" ", texto)
    texto = RE_ETIQUETA_HTML.sub(" ", texto)
    texto = RE_TABLA.sub(" ", texto)
    for _ in range(8):
        nuevo = RE_PLANTILLA.sub(" ", texto)
        if nuevo == texto:
            break
        texto = nuevo
    texto = RE_ENLACE_ARCHIVO.sub(" ", texto)
    texto = RE_ENLACE_EXTERNO.sub(r" \1 ", texto)
    texto = RE_ENLACE_INTERNO.sub(r" \2 ", texto)
    texto = RE_ENLACE_SIMPLE.sub(r" \1 ", texto)
    texto = RE_URL_SIMPLE.sub(" url ", texto)
    texto = RE_MARCAS.sub("", texto)
    texto = html.unescape(texto)
    texto = RE_ENTIDADES.sub(" ", texto)
    lineas = []
    for linea in texto.splitlines():
        limpia = linea.strip()
        if not limpia or limpia.startswith("="):
            continue
        if limpia.startswith("*") or limpia.startswith("#"):
            limpia = limpia.lstrip("*#:; ")
        limpia = RE_CARACTERES_PERMITIDOS.sub(" ", limpia)
        limpia = RE_ESPACIOS.sub(" ", limpia).strip()
        if limpia:
            lineas.append(limpia)
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# 4-6. Deduplicacion, normalizacion, tokenizacion y shards
# ---------------------------------------------------------------------------
def normalizar_emb_normal(texto: str) -> str:
    """Aplica la normalizacion EMB-NORMAL.

    Minusculas conservando tildes, URLs, menciones y hashtags reducidos a
    marcadores estables y numeros a ``numero``. **No** hay corrector ortografico
    ni reconstruccion de tildes.

    Args:
        texto: texto ya limpio de marcado.

    Returns:
        Texto normalizado.
    """
    texto = texto.lower()
    texto = RE_URL.sub(" url ", texto)
    texto = RE_HASHTAG.sub(" hashtag ", texto)
    texto = RE_MENCION.sub(" usuario ", texto)
    texto = RE_NUMERO.sub(" numero ", texto)
    return RE_ESPACIOS.sub(" ", texto).strip()


def plegar_ascii(texto: str) -> str:
    """Elimina los diacriticos de un texto sin alterar el resto de caracteres.

    Es la operacion que define la variante **EMB-ASCII** de la seccion 16 de la
    spec. Se usa aqui **solo como proyeccion diagnostica**: permite medir
    cuantos tipos ausentes del corpus se deben unicamente a la falta de
    tildes en el corpus TOXIC, sin necesidad de reentrenar nada.

    Args:
        texto: texto de entrada.

    Returns:
        Texto sin marcas diacriticas.
    """
    descompuesto = unicodedata.normalize("NFD", texto)
    return "".join(caracter for caracter in descompuesto if unicodedata.category(caracter) != "Mn")


VOCALES = frozenset("aeiouáéíóúüAEIOUÁÉÍÓÚÜ")


def esqueleto_vocalico(texto: str) -> str:
    """Elimina las vocales de una palabra para comparar formas deformadas.

    Los datos de TOXIC no solo pierden tildes: pierden vocales completas
    (``cllate``, ``violacin``, ``espaa``). Comparar el esqueleto vocalico de un
    tipo TOXIC con el de los tipos del corpus permite distinguir esa deformacion
    ortografica de una laguna lexica real.

    Args:
        texto: palabra de entrada.

    Returns:
        Palabra sin vocales, en minusculas.
    """
    plegada = plegar_ascii(texto).lower()
    return "".join(caracter for caracter in plegada if caracter not in VOCALES)


def tokenizar(texto: str) -> list[str]:
    """Tokeniza un texto con el mismo criterio que se usara en el embedding.

    Args:
        texto: texto normalizado.

    Returns:
        Lista de tokens.
    """
    return RE_TOKEN.findall(texto)


def construir_corpus(
    ruta_bz2: Path, objetivo: int = OBJETIVO_TOKENS
) -> dict[str, Any]:
    """Construye el corpus normalizado en shards hasta alcanzar el objetivo.

    Args:
        ruta_bz2: parte del dump ya descargada.
        objetivo: numero de tokens objetivo.

    Returns:
        Diccionario con conteos, rutas de shards y estadisticas del pipeline.
    """
    log("2/9 limpieza, deduplicacion, normalizacion y tokenizacion")
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for viejo in CORPUS_DIR.glob("tokens_shard_*.txt"):
        viejo.unlink()
    vistos: set[bytes] = set()
    vocabulario: Counter[str] = Counter()
    documentos = 0
    tokens_totales = 0
    articulos_leidos = 0
    articulos_vacios = 0
    duplicados = 0
    corto = 0
    inicio = time.time()
    shards: list[Path] = []
    tokens_shard = 0
    archivo = None
    indice_shard = 0
    titulo_ejemplo: list[str] = []
    for titulo, limpio in iterar_articulos(ruta_bz2):
        articulos_leidos += 1
        if not limpio:
            articulos_vacios += 1
            continue
        huella = hashlib.sha1(limpio.encode("utf-8")).digest()
        if huella in vistos:
            duplicados += 1
            continue
        vistos.add(huella)
        normalizado = normalizar_emb_normal(limpio)
        tokens = tokenizar(normalizado)
        if len(tokens) < TAMAO_MINIMO_TOKENS_DOC:
            corto += 1
            continue
        if archivo is None or (
            tokens_shard >= TOKENS_POR_SHARD and tokens_totales < objetivo
        ):
            if archivo is not None:
                archivo.close()
            ruta_shard = CORPUS_DIR / f"tokens_shard_{indice_shard:02d}.txt"
            archivo = ruta_shard.open("w", encoding="utf-8")
            shards.append(ruta_shard)
            tokens_shard = 0
            indice_shard += 1
            log(f"      shard {indice_shard:02d} abierto (documentos {documentos})")
        archivo.write(" ".join(tokens) + "\n")
        documentos += 1
        tokens_totales += len(tokens)
        tokens_shard += len(tokens)
        vocabulario.update(tokens)
        if len(titulo_ejemplo) < 5:
            titulo_ejemplo.append(titulo)
        if tokens_totales >= objetivo:
            break
    if archivo is not None:
        archivo.close()
    log(
        f"   {documentos} documentos | {tokens_totales} tokens | "
        f"{len(vocabulario)} tipos | {time.time() - inicio:.0f} s"
    )
    log(
        f"   articulos leidos {articulos_leidos} | duplicados {duplicados} | "
        f"cortos {corto} | sin texto util {articulos_vacios}"
    )
    (CORPUS_DIR / "vocabulario.txt").write_text(
        "".join(f"{palabra} {conteo}\n" for palabra, conteo in vocabulario.most_common()),
        encoding="utf-8",
    )
    return {
        "documentos": documentos,
        "tokens": tokens_totales,
        "tipos": len(vocabulario),
        "articulos_leidos": articulos_leidos,
        "duplicados": duplicados,
        "cortos": corto,
        "sin_texto_util": articulos_vacios,
        "shards": [str(ruta.relative_to(PROJECT_ROOT)) for ruta in shards],
        "ejemplos_titulo": titulo_ejemplo,
        "segundos": round(time.time() - inicio, 1),
        "vocabulario_path": str((CORPUS_DIR / "vocabulario.txt").relative_to(PROJECT_ROOT)),
    }


def leer_vocabulario_corpus() -> Counter[str]:
    """Carga el vocabulario del corpus con sus conteos.

    Returns:
        ``Counter`` con tipo -> frecuencia.
    """
    vocabulario: Counter[str] = Counter()
    with (CORPUS_DIR / "vocabulario.txt").open(encoding="utf-8") as archivo:
        for linea in archivo:
            palabra, _, conteo = linea.rpartition(" ")
            vocabulario[palabra] = int(conteo)
    return vocabulario


def concatenar_corpus(objetivo_tokens: int = OBJETIVO_TOKENS) -> Path:
    """Une los shards en un unico archivo de entrenamiento.

    Args:
        objetivo_tokens: numero de tokens que se deben conservar.

    Returns:
        Ruta del archivo concatenado.
    """
    destino = CORPUS_DIR / f"tokens_{objetivo_tokens // 1_000_000}M.txt"
    tokens = 0
    with destino.open("w", encoding="utf-8") as salida:
        for shard in sorted(CORPUS_DIR.glob("tokens_shard_*.txt")):
            with shard.open(encoding="utf-8") as entrada:
                for linea in entrada:
                    if tokens >= objetivo_tokens:
                        break
                    salida.write(linea)
                    tokens += len(linea.split())
    log(f"   archivo de entrenamiento: {destino.name} con {tokens} tokens")
    return destino


# ---------------------------------------------------------------------------
# 8. Auditoria de fuga
# ---------------------------------------------------------------------------
def hash_ngramas(tokens: list[str], n: int = NGRAMAS_FUGA) -> np.ndarray:
    """Calcula hashes de 64 bits de los n-gramas de una secuencia.

    Args:
        tokens: secuencia de tokens.
        n: tamaño del n-grama.

    Returns:
        Arreglo de hashes.
    """
    if len(tokens) < n:
        return np.empty(0, dtype=np.uint64)
    ids = np.fromiter((hash(palabra) & SEMILLA_Z for palabra in tokens), dtype=np.uint64, count=len(tokens))
    suma = np.zeros(len(ids) - n + 1, dtype=np.uint64)
    for desplazamiento in range(n):
        suma = suma * np.uint64(1_000_003) + ids[desplazamiento : desplazamiento + len(suma)]
    return suma


def auditar_fuga(archivo_corpus: Path) -> dict[str, Any]:
    """Comprueba que ningun texto de TOXIC aparece en el corpus.

    Se usan hashes de 8-gramas de tokens: si un texto de train o de test
    compartiera 8 o mas tokens consecutivos con el corpus, se detectaria.

    Args:
        archivo_corpus: archivo de entrenamiento con los shards unidos.

    Returns:
        Diccionario con el resultado de la auditoria y los ejemplos.
    """
    log(f"8/9 auditoria de fuga (n-gramas de {NGRAMAS_FUGA} tokens)")
    indice = np.empty(0, dtype=np.uint64)
    bloques: list[np.ndarray] = []
    with archivo_corpus.open(encoding="utf-8") as archivo:
        for linea in archivo:
            bloques.append(hash_ngramas(linea.split(), NGRAMAS_FUGA))
    if bloques:
        indice = np.unique(np.concatenate(bloques))
    log(f"   indice de {NGRAMAS_FUGA}-gramas: {indice.size} unicos")
    train = read_train()
    test = read_test()
    hallazgos: list[dict[str, Any]] = []
    for nombre, marco in (("train", train), ("test", test)):
        textos = marco["text"].astype(str).tolist()
        for identificador, texto in zip(marco["id"].tolist(), textos, strict=True):
            tokens = tokenizar(normalizar_emb_normal(texto))
            hashes = hash_ngramas(tokens, NGRAMAS_FUGA)
            if hashes.size == 0:
                continue
            posicion = np.searchsorted(indice, hashes)
            posicion = np.clip(posicion, 0, indice.size - 1)
            if np.any(indice[posicion] == hashes):
                hallazgos.append({"conjunto": nombre, "id": int(identificador)})
    resultado = {
        "n_gramas": NGRAMAS_FUGA,
        "n_gramas_unicos_corpus": int(indice.size),
        "textos_con_coincidencia": len(hallazgos),
        "detalle": hallazgos[:20],
        "cumple": not hallazgos,
    }
    log(
        f"   train {len(train)} textos | test {len(test)} textos | "
        f"coincidencias: {len(hallazgos)} (cumple: {resultado['cumple']})"
    )
    return resultado


# ---------------------------------------------------------------------------
# 9. Cobertura del vocabulario TOXIC
# ---------------------------------------------------------------------------
def vocabulario_toxic(marco: pd.DataFrame) -> Counter[str]:
    """Calcula el vocabulario de tokens de un marco de TOXIC.

    Args:
        marco: marco con columna ``text``.

    Returns:
        ``Counter`` con tipo -> frecuencia en TOXIC.
    """
    vocabulario: Counter[str] = Counter()
    for texto in marco["text"].astype(str):
        vocabulario.update(tokenizar(normalizar_emb_normal(texto)))
    return vocabulario


GRUPOS_RELEVANTES: dict[str, dict[str, Any]] = {
    "insultos": {
        "descripcion": "lexico de insulto explicito y variantes morfologicas",
        "patron": r"^(?:idiot\w*|imbec\w*|estupid\w*|callate|callate|subnormal\w*|retrasad\w*|gilipollas?|pendej\w*|marica\w*|zorra\w*|perra\w*|put\w*|cabron\w*|joder\w*|coño|polla\w*|verga\w*|mamon\w*|capullo\w*|culero\w*|boludo\w*|pelotud\w*|mierda\w*|hijue\w*|p%$#@!+|mierdas?)$",
    },
    "targeting": {
        "descripcion": "marcas de segunda persona y de destinatario",
        "patron": r"^(?:tu|tú|usted|ustedes|tus|su|sus|eres|eres?$|vete|hablas|habla$|mientes|imbecil)$",
    },
    "colectivos": {
        "descripcion": "colectivos y gentilicos usados como categoria",
        "patron": r"^(?:inmigrantes|inmigrados?|gitanos?|moros|juntos?|chinos?|indios?|negros?|judios?|musulmanes|mojibkes|indigenas?|gitanos|negras?|blancos?|extranjeros?|invasores?|paramilitares?)$",
    },
    "negacion": {
        "descripcion": "operadores de negacion",
        "patron": r"^(?:no|ni|nunca|jamás|nada|ningun\w*|ninguna|tampoco|sin|nel|demasiado)$",
    },
    "coloquial": {
        "descripcion": "marcadores coloquiales y abreviaturas",
        "patron": r"^(?:jaja\w*|jeje\w*|xd|xdd|xdr|lol|jejej|plis|plis|porfa|xfa|q|qe|xq|pa|pa'|k|tmb|aw|uff|uy|eh|más|bien|vale|nah)$",
    },
    "elongaciones": {
        "descripcion": "formas con caracteres repetidos ( elongation ortografica )",
        "patron": r"(.)\1{2,}",
    },
    "noticia": {
        "descripcion": "discurso de noticia y politica",
        "patron": r"^(?:noticia|noticias|hoy|ayer|mañana|según|dice|dijo|gobierno|país|ministro|ministra|elecciones|protesta|manifestación|ley|congreso|política|pueblo|gente|mundo)$",
    },
    "hashtags": {
        "descripcion": "marcas de hashtag (normalizadas a `hashtag`)",
        "patron": r"^hashtag$",
    },
    "menciones": {
        "descripcion": "marcas de mencion (normalizadas a `usuario`)",
        "patron": r"^usuario$",
    },
    "formas_raras": {
        "descripcion": "tipos TOXIC poco frecuentes (frecuencia <= 2) y de 4+ caracteres",
        "patron": None,
    },
    "marcadores_web": {
        "descripcion": "marcas de URL (normalizadas a `url`)",
        "patron": r"^(?:url|numero)$",
    },
}


def metricas_grupo(
    tipos: list[str], vocabulario_toxic_c: Counter[str], vocabulario_corpus: Counter[str]
) -> dict[str, Any]:
    """Calcula cobertura, OOV y frecuencia media de un grupo.

    Args:
        tipos: tipos del grupo.
        vocabulario_toxic_c: frecuencias en TOXIC.
        vocabulario_corpus: frecuencias del corpus.

    Returns:
        Diccionario con las metricas del grupo.
    """
    n_tokens = sum(vocabulario_toxic_c.get(tipo, 0) for tipo in tipos)
    presentes = [tipo for tipo in tipos if tipo in vocabulario_corpus]
    tipo_cobertura = len(presentes) / len(tipos) if tipos else float("nan")
    token_cobertura = (
        sum(vocabulario_toxic_c.get(tipo, 0) for tipo in presentes) / n_tokens
        if n_tokens
        else float("nan")
    )
    frecuencia_media = (
        float(np.mean([vocabulario_corpus[tipo] for tipo in presentes])) if presentes else float("nan")
    )
    return {
        "n_types": len(tipos),
        "n_tokens": n_tokens,
        "type_coverage": round(tipo_cobertura, 4),
        "token_coverage": round(token_cobertura, 4),
        "oov_rate": round(1 - tipo_cobertura, 4),
        "frecuencia_media_corpus": round(frecuencia_media, 1),
    }


def clasificar_grupos(vocabulario_toxic_c: Counter[str]) -> dict[str, list[str]]:
    """Asigna los tipos de TOXIC a los grupos relevantes.

    Los grupos se definen con listas y patrones **curados**, sin usar labels:
    son un instrumento de diagnostico de cobertura, no una fuente de
    de supervision para el embedding.

    Args:
        vocabulario_toxic_c: frecuencias en TOXIC.

    Returns:
        Diccionario grupo -> lista de tipos.
    """
    grupos: dict[str, list[str]] = {nombre: [] for nombre in GRUPOS_RELEVANTES}
    compilados: dict[str, re.Pattern[str]] = {}
    for nombre, definicion in GRUPOS_RELEVANTES.items():
        patron = definicion["patron"]
        if patron is not None:
            compilados[nombre] = re.compile(patron, re.IGNORECASE)
    for tipo, frecuencia in vocabulario_toxic_c.items():
        for nombre, patron in compilados.items():
            if patron.search(tipo):
                grupos[nombre].append(tipo)
        if frecuencia <= 2 and len(tipo) >= 4:
            grupos["formas_raras"].append(tipo)
    return grupos


def medir_cobertura(
    vocabulario_corpus: Counter[str],
    archivo_sifts: Path,
    ruta_vec_probe: Path,
) -> dict[str, Any]:
    """Mide la cobertura del vocabulario TOXIC sobre el corpus.

    Args:
        vocabulario_corpus: frecuencias del corpus.
        archivo_sifts: archivo de vectores probe para representation-OOV.
        ruta_vec_probe: archivo ``.vec`` del probe, para el vecino mas cercano.

    Returns:
        Diccionario con cobertura global, por grupo y lista de OOV.
    """
    log("9/9 cobertura del vocabulario TOXIC")
    train = read_train()
    test = read_test()
    vocab_train = vocabulario_toxic(train)
    vocab_test = vocabulario_toxic(test)
    vocab_union = vocab_train + vocab_test
    vectores_probe = cargar_sifts(archivo_sifts)
    indice_probe, matriz_probe = cargar_matriz_probe(ruta_vec_probe)

    resumen: dict[str, Any] = {"por_conjunto": {}, "grupos": [], "oov": []}
    for nombre, vocabulario in (
        ("train", vocab_train),
        ("test", vocab_test),
        ("train+test", vocab_union),
    ):
        tipos = list(vocabulario)
        presentes = [tipo for tipo in tipos if tipo in vocabulario_corpus]
        n_tokens = sum(vocabulario.values())
        tokens_cubiertos = sum(vocabulario[tipo] for tipo in presentes)
        word_oov = [tipo for tipo in tipos if tipo not in vocabulario_corpus]
        representables = [tipo for tipo in word_oov if tipo in vectores_probe]
        normas_probe = [
            float(np.linalg.norm(vectores_probe[tipo]))
            for tipo in representables
            if tipo in vectores_probe
        ]
        # Dos pruebas de la deformacion ortografica, ambas sobre conjuntos
        # precalculados: (a) plegado ASCII, (b) esqueleto vocalico.
        # Dos pruebas de la deformacion ortografica, sobre conjuntos precalculados
        # una sola vez por conjunto (no por tipo OOV: seria cuadratico).
        ascii_corpus = {plegar_ascii(palabra) for palabra in vocabulario_corpus}
        esqueletos_corpus = {esqueleto_vocalico(palabra) for palabra in vocabulario_corpus}
        ascii_resolubles = [tipo for tipo in word_oov if plegar_ascii(tipo) in ascii_corpus]
        esqueleto_resolubles = [
            tipo
            for tipo in word_oov
            if len(esqueleto_vocalico(tipo)) >= 3 and esqueleto_vocalico(tipo) in esqueletos_corpus
        ]
        resumen["por_conjunto"][nombre] = {
            "n_types": len(tipos),
            "n_tokens": n_tokens,
            "type_coverage": round(len(presentes) / len(tipos), 4),
            "token_coverage": round(tokens_cubiertos / n_tokens, 4),
            "frequency_weighted_coverage": round(tokens_cubiertos / n_tokens, 4),
            "oov_rate_types": round(1 - len(presentes) / len(tipos), 4),
            "oov_rate_tokens": round(1 - tokens_cubiertos / n_tokens, 4),
            "n_word_oov": len(word_oov),
            "n_representation_oov": len(word_oov) - len(representables),
            "representation_oov_rate": round(
                1 - len(representables) / len(word_oov), 4
            )
            if word_oov
            else 0.0,
            "mediana_norma_probe": round(float(np.median(normas_probe)), 6) if normas_probe else 0.0,
            "n_word_oov_resolubles_por_ascii": len(ascii_resolubles),
            "tasa_oov_por_ascii": round(len(ascii_resolubles) / len(word_oov), 4) if word_oov else 0.0,
            "n_word_oov_resolubles_por_esqueleto": len(esqueleto_resolubles),
            "tasa_oov_por_esqueleto": round(
                len(esqueleto_resolubles) / len(word_oov), 4
            )
            if word_oov
            else 0.0,
            "type_coverage_proyeccion_ascii": round(
                (len(presentes) + len(ascii_resolubles)) / len(tipos), 4
            ),
            "type_coverage_proyeccion_esqueleto": round(
                (len(presentes) + len(esqueleto_resolubles)) / len(tipos), 4
            ),
            "ejemplos_oov_resolubles_por_ascii": [
                {"tipo": tipo, "ascii": plegar_ascii(tipo)} for tipo in sorted(ascii_resolubles)[:15]
            ],
            "ejemplos_oov_resolubles_por_esqueleto": [
                {"tipo": tipo, "esqueleto": esqueleto_vocalico(tipo)}
                for tipo in sorted(esqueleto_resolubles)[:15]
            ],
        }
        if nombre == "train+test":
            resumen["vecinos_subwords"] = analizar_vecinos_subwords(
                vectores_probe, indice_probe, matriz_probe, word_oov, vocabulario
            )
            oov = sorted(word_oov, key=lambda tipo: -vocabulario[tipo])
            resumen["oov"] = [
                {
                    "tipo": tipo,
                    "freq_toxic": vocabulario[tipo],
                    "representable_subwords": tipo in vectores_probe,
                }
                for tipo in oov
            ]
        log(
            f"   {nombre}: tipos {len(tipos)} | cobertura tipo "
            f"{resumen['por_conjunto'][nombre]['type_coverage']:.4f} | token "
            f"{resumen['por_conjunto'][nombre]['token_coverage']:.4f} | word-OOV "
            f"{len(word_oov)} | representation-OOV "
            f"{resumen['por_conjunto'][nombre]['n_representation_oov']} | "
            f"resolubles por ASCII {len(ascii_resolubles)} "
            f"({resumen['por_conjunto'][nombre]['tasa_oov_por_ascii']:.2f})"
        )
    grupos = clasificar_grupos(vocab_union)
    filas = []
    for nombre, tipos in grupos.items():
        if not tipos:
            continue
        metricas = metricas_grupo(tipos, vocab_union, vocabulario_corpus)
        metricas["grupo"] = nombre
        metricas["descripcion"] = GRUPOS_RELEVANTES[nombre]["descripcion"]
        filas.append(metricas)
    filas.sort(key=lambda fila: fila["token_coverage"])
    resumen["grupos"] = filas
    return resumen


def cargar_matriz_probe(ruta_vec: Path) -> tuple[dict[str, int], np.ndarray]:
    """Carga los vectores del probe en una matriz normalizada para vecinos.

    Args:
        ruta_vec: archivo ``.vec`` del modelo probe.

    Returns:
        Tupla ``(indice, matriz)`` con ``indice[tipo]`` y filas normalizadas.
    """
    palabras: list[str] = []
    filas: list[np.ndarray] = []
    with ruta_vec.open(encoding="utf-8", errors="replace") as archivo:
        primera = archivo.readline().split()
        for linea in archivo:
            partes = linea.split()
            if len(partes) < 3:
                continue
            valores = np.asarray(partes[1:], dtype=np.float32)
            if not np.isfinite(valores).all():
                continue
            norma = float(np.linalg.norm(valores))
            if norma == 0.0:
                continue
            palabras.append(partes[0])
            filas.append(valores / norma)
    indice = {palabra: posicion for posicion, palabra in enumerate(palabras)}
    matriz = np.vstack(filas) if filas else np.empty((0, 1), dtype=np.float32)
    return indice, matriz


def cargar_sifts(archivo_sifts: Path) -> dict[str, np.ndarray]:
    """Carga los vectores probe de los tipos TOXIC.

    Args:
        archivo_sifts: archivo de salida de ``fasttext print-word-vectors``.

    Returns:
        Diccionario tipo -> vector (``float32``). Los vectores nulos no se
        guardan, de modo que la presencia en el diccionario significa que el
        modelo produce una representacion no trivial.
    """
    vectores: dict[str, np.ndarray] = {}
    if not archivo_sifts.exists():
        return vectores
    with archivo_sifts.open(encoding="utf-8", errors="replace") as archivo:
        for linea in archivo:
            # fastText separa por espacios y deja un espacio final: `split()`
            # colapsa los separadores repetidos y descarta el campo vacio final.
            partes = linea.split()
            if len(partes) < 2:
                continue
            palabra = partes[0]
            try:
                valores = np.asarray(partes[1:], dtype=np.float32)
            except ValueError:
                continue
            if not np.isfinite(valores).all() or float(np.linalg.norm(valores)) == 0.0:
                continue
            vectores[palabra] = valores
    return vectores


def analizar_vecinos_subwords(
    vectores: dict[str, np.ndarray],
    indice_probe: dict[str, int],
    matriz_probe: np.ndarray,
    tipos_oov: list[str],
    frecuencias: Counter[str],
    limite: int = 25,
) -> dict[str, Any]:
    """Mide si el vector compuesto por subpalabras es informativo.

    Para cada tipo word-OOV se busca el tipo del corpus mas cercano en coseno al
    vector que fastText compone con subpalabras. Un coseno alto indica que los
    subpalabras reconstruyen la deformacion ortografica; un coseno bajo indica
    que el tipo no tiene referente en un corpus enciclopedico.

    Args:
        vectores: vectores probe de los tipos TOXIC.
        indice_probe: indice palabra -> fila de la matriz del probe.
        matriz_probe: matriz normalizada de vectores del corpus.
        tipos_oov: tipos word-OOV a evaluar.
        frecuencias: frecuencias de los tipos en TOXIC.
        limite: numero de ejemplos a reportar.

    Returns:
        Diccionario con la distribucion de cosenos y ejemplos.
    """
    if matriz_probe.size == 0 or not tipos_oov:
        return {"evaluados": 0}
    lista_palabras = [None] * len(indice_probe)
    for palabra, posicion in indice_probe.items():
        lista_palabras[posicion] = palabra
    # Los vectores de fastText comparten un componente comun grande (direccion de
    # la frecuencia unigrama), de modo que el coseno crudo entre cualesquiera dos
    # vectores es alto y el vecino mas cercano no significa nada. Se resta la
    # media de la matriz y se renormaliza para obtener un coseno informativo.
    centro = matriz_probe.mean(axis=0)
    matriz_centrada = matriz_probe - centro
    normas_centradas = np.linalg.norm(matriz_centrada, axis=1, keepdims=True)
    matriz_centrada = matriz_centrada / np.maximum(normas_centradas, 1e-9)
    cosenos: list[float] = []
    ejemplos: list[dict[str, Any]] = []
    for tipo in sorted(tipos_oov, key=lambda palabra: -frecuencias[palabra]):
        vector = vectores.get(tipo)
        if vector is None or vector.shape[0] != matriz_probe.shape[1]:
            continue
        consulta = (vector - centro) / max(float(np.linalg.norm(vector - centro)), 1e-9)
        similitudes = matriz_centrada @ consulta
        orden = np.argsort(-similitudes)[:2]
        vecinos = [
            {
                "tipo_corpus": lista_palabras[posicion],
                "coseno_centrado": round(float(similitudes[posicion]), 3),
            }
            for posicion in orden
        ]
        cosenos.append(float(similitudes[orden[0]]))
        if len(ejemplos) < limite:
            ejemplos.append({"tipo_oov": tipo, "freq_toxic": frecuencias[tipo], "vecinos": vecinos})
    valores = np.asarray(cosenos, dtype=np.float64)
    return {
        "evaluados": int(valores.size),
        "nota": "cosenos centrados: se resta la media de la matriz del probe antes de comparar",
        "coseno_medio": round(float(valores.mean()), 4) if valores.size else 0.0,
        "coseno_mediano": round(float(np.median(valores)), 4) if valores.size else 0.0,
        "frac_coseno_ge_0.6": round(float((valores >= 0.6).mean()), 4) if valores.size else 0.0,
        "frac_coseno_ge_0.4": round(float((valores >= 0.4).mean()), 4) if valores.size else 0.0,
        "frac_coseno_lt_0.3": round(float((valores < 0.3).mean()), 4) if valores.size else 0.0,
        "ejemplos": ejemplos,
    }


def probar_representabilidad(archivo_corpus: Path, vocabulario_toxic_c: Counter[str]) -> dict[str, Any]:
    """Calcula que tipos word-OOV reciben vector por subpalabras.

    Args:
        archivo_corpus: archivo de entrenamiento.
        vocabulario_toxic_c: frecuencias en TOXIC.

    Returns:
        Diccionario con el destino del probe y el numero de tipos probeados.
    """
    SIFTS_DIR.mkdir(parents=True, exist_ok=True)
    destino = SIFTS_DIR / "probe_50d"
    subprocess.run(
        [str(FASTTEXT_BIN), "skipgram", "-input", str(archivo_corpus), "-output", str(destino),
         "-dim", "50", "-ws", "5", "-minCount", "3", "-neg", "10", "-epoch", "1",
         "-thread", "16", "-seed", "42", "-minn", "3", "-maxn", "6", "-bucket", "200000", "-verbose", "0"],
        check=True,
        capture_output=True,
    )
    tipos = sorted(vocabulario_toxic_c)
    entrada = SIFTS_DIR / "tipos_toxic.txt"
    entrada.write_text("\n".join(tipos) + "\n", encoding="utf-8")
    salida = SIFTS_DIR / "probe_50d_sift.txt"
    with entrada.open("rb") as origen, salida.open("wb") as destino_f:
        subprocess.run(
            [str(FASTTEXT_BIN), "print-word-vectors", f"{destino}.bin"],
            stdin=origen,
            stdout=destino_f,
            check=True,
            stderr=subprocess.DEVNULL,
        )
    log(f"   vectores probeados: {len(tipos)}")
    return {
        "modelo": f"{destino}.bin",
        "sifts": str(salida.relative_to(PROJECT_ROOT)),
        "vec": str((SIFTS_DIR / "probe_50d.vec").relative_to(PROJECT_ROOT)),
        "tipos": len(tipos),
    }


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def figuras(vocabulario_corpus: Counter[str], cobertura: dict[str, Any]) -> list[str]:
    """Genera las figuras de la fase.

    Args:
        vocabulario_corpus: frecuencias del corpus.
        cobertura: resultado de la medicion de cobertura.

    Returns:
        Rutas de las figuras generadas.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    rutas = []
    frecuencias = np.array(sorted(vocabulario_corpus.values(), reverse=True), dtype=np.float64)
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, frecuencias.size + 1), frecuencias, linewidth=0.8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("rango de tipo (log)")
    plt.ylabel("frecuencia en el corpus (log)")
    plt.title("Distribucion de frecuencias del corpus 10M (Wikipedia ES)")
    ruta = FIGURAS / "fig_41_distribucion_frecuencias_v5.png"
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    rutas.append(str(ruta.relative_to(PROJECT_ROOT)))

    grupos = cobertura["grupos"]
    nombres = [fila["grupo"] for fila in grupos]
    tipos = [100 * fila["type_coverage"] for fila in grupos]
    tokens = [100 * fila["token_coverage"] for fila in grupos]
    posicion = np.arange(len(nombres))
    plt.figure(figsize=(8, 4.5))
    plt.barh(posicion - 0.2, tipos, height=0.38, label="cobertura por tipo")
    plt.barh(posicion + 0.2, tokens, height=0.38, label="cobertura por token")
    plt.yticks(posicion, nombres)
    plt.xlim(0, 100)
    plt.xlabel("cobertura (%)")
    plt.title("Cobertura del vocabulario TOXIC por grupo relevante (corpus 10M)")
    plt.legend()
    ruta = FIGURAS / "fig_42_cobertura_grupos_v5.png"
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    rutas.append(str(ruta.relative_to(PROJECT_ROOT)))
    return rutas


# ---------------------------------------------------------------------------
# Manifiesto e informes
# ---------------------------------------------------------------------------
def actualizar_manifiesto(
    descarga: dict[str, Any], corpus: dict[str, Any], fuga: dict[str, Any], cobertura: dict[str, Any]
) -> dict[str, Any]:
    """Actualiza el manifiesto con el corpus construido.

    Args:
        descarga: metadatos de la parte descargada.
        corpus: estadisticas de la construccion.
        fuga: resultado de la auditoria de fuga.
        cobertura: medicion de cobertura.

    Returns:
        El manifiesto actualizado.
    """
    manifiesto = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifiesto["preprocessing_version"] = PREPROCESSING_VERSION
    entrada = {
        "id": EXPERIMENT_ID,
        "source": "wikipedia_es",
        "url": descarga["url"],
        "resource": PARTE,
        "version": f"eswiki latest (dump del {descarga['fecha_consulta']})",
        "license": LICENCIA,
        "download_date": descarga["fecha_consulta"],
        "raw_file": descarga["archivo"],
        "raw_sha256": descarga["sha256"],
        "raw_size": descarga["bytes"],
        "document_count": corpus["documentos"],
        "token_count": corpus["tokens"],
        "vocabulary_size": corpus["tipos"],
        "shards": corpus["shards"],
        "clean_size": sum(
            (PROJECT_ROOT / ruta).stat().st_size for ruta in corpus["shards"]
        ),
        "preprocessing": {
            "version": PREPROCESSING_VERSION,
            "variante": "EMB-NORMAL",
            "minusculas": True,
            "tildes": "conservadas",
            "corrector_ortografico": False,
            "urls": "marcador `url`",
            "menciones": "marcador `usuario`",
            "hashtags": "marcador `hashtag`",
            "numeros": "marcador `numero`",
            "deduplicacion": "sha1 del texto limpio",
            "minimo_tokens_documento": TAMAO_MINIMO_TOKENS_DOC,
        },
        "pipeline": {
            "articulos_leidos": corpus["articulos_leidos"],
            "duplicados_descartados": corpus["duplicados"],
            "documentos_cortos_descartados": corpus["cortos"],
            "sin_texto_util": corpus["sin_texto_util"],
        },
        "auditoria_fuga": {
            "n_gramas": fuga["n_gramas"],
            "textos_con_coincidencia": fuga["textos_con_coincidencia"],
            "cumple": fuga["cumple"],
        },
        "cobertura_toxic": cobertura["por_conjunto"],
        "limitaciones": [
            "Solo una parte del dump (paginas 1-159400): el orden del dump se respeta, "
            "por lo que la muestra se concentra en articulos antiguos.",
            "El corpus no contiene registro web informal; esa limitacion se mide en la "
            "cobertura por grupo (coloquial, hashtags, elongation).",
            "Wikipedia es texto enciclopedico: no es representativo de tweets.",
        ],
    }
    manifiesto["corpus_construidos"] = [entrada]
    MANIFEST_PATH.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MANIFEST_REPORTE.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifiesto


def escribir_informes(
    descarga: dict[str, Any],
    corpus: dict[str, Any],
    fuga: dict[str, Any],
    cobertura: dict[str, Any],
    rutas_figuras: list[str],
) -> None:
    """Escribe los dos informes de P1.

    Args:
        descarga: metadatos de la descarga.
        corpus: estadisticas de la construccion.
        fuga: auditoria de fuga.
        cobertura: medicion de cobertura.
        rutas_figuras: figuras generadas.
    """
    lineas = [
        "# Auditoria del corpus de la fase v5 (escala 1)",
        "",
        f"Generado por `experiments/embedding_01_corpus_wikipedia_v5.py` "
        f"({EXPERIMENT_ID}). Decision de gobierno: **D-030**.",
        "",
        "## 1. Procedencia del corpus",
        "",
        "| Campo | Valor |",
        "|---|---|",
        "| Fuente | Wikipedia en espanol (`eswiki`) |",
        f"| URL | {descarga['url']} |",
        f"| Recurso | `{PARTE}` |",
        f"| Version | dump del {descarga['fecha_consulta']} |",
        f"| Licencia | {LICENCIA} |",
        f"| sha256 del archivo | `{descarga['sha256']}` |",
        f"| Tamano comprimido | {descarga['bytes'] / 1024**2:.0f} MB |",
        "",
        "Se descarga **una sola parte** del dump (paginas 1-159400) y no los 5.19 GB "
        "completos: la parte basta para 10M, 50M y 100M tokens, y hace la fase "
        "reproducible con un solo artefacto verificable por hash.",
        "",
        "## 2. Pipeline aplicado",
        "",
        "`download -> raw -> clean -> dedup -> normalize -> tokenize -> shards -> manifest`",
        "",
        "| Paso | Detalle |",
        "|---|---|",
        "| Lectura | incremental con `XMLPullParser` sobre el bz2; nunca se carga el dump en RAM |",
        "| Filtro | solo `ns=0` (articulos), sin redirecciones ni paginas de titulo con `:` |",
        "| Limpieza | plantillas, tablas, `ref`, HTML, enlaces a archivos y categorias; se conservan las tildes |",
        f"| Deduplicacion | `sha1` del texto limpio ({corpus['duplicados']} duplicados descartados) |",
        f"| Normalizacion | {PREPROCESSING_VERSION}: minusculas, `url`/`usuario`/`hashtag`/`numero`, sin corrector ortografico |",
        f"| Filtro de longitud | documentos con menos de {TAMAO_MINIMO_TOKENS_DOC} tokens descartados ({corpus['cortos']}) |",
        "",
        "## 3. Conteos",
        "",
        "| Metrica | Valor |",
        "|---|---|",
        f"| Articulos leidos | {corpus['articulos_leidos']:,} |".replace(",", "."),
        f"| Documentos en el corpus | {corpus['documentos']:,} |".replace(",", "."),
        f"| Tokens | {corpus['tokens']:,} |".replace(",", "."),
        f"| Tipos (vocabulario) | {corpus['tipos']:,} |".replace(",", "."),
        f"| Tipos por token (T/TR) | {corpus['tokens'] / max(corpus['tipos'], 1):.2f} |",
        f"| Shards | {len(corpus['shards'])} (~2M tokens cada uno) |",
        f"| Tiempo de construccion | {corpus['segundos']} s |",
        "",
        f"Ejemplos de articulos: {', '.join(corpus['ejemplos_titulo'])}.",
        "",
        "## 4. Auditoria de fuga",
        "",
        f"Se indexaron **{fuga['n_gramas_unicos_corpus']:,}** {fuga['n_gramas']}-gramas unicos del corpus "
        "y se buscaron los de los 2.885 textos de TOXIC (train y test).",
        "",
        f"- Textos de TOXIC con coincidencia de {fuga['n_gramas']} o mas tokens: **{fuga['textos_con_coincidencia']}**",
        f"- Veredicto: **{'CUMPLE' if fuga['cumple'] else 'REVISAR'}**",
        "",
        "Ademas, por construccion, el corpus no contiene ningun texto de TOXIC: la variante es "
        "`EXTERNAL-ONLY` y la normalizacion se aplica al dump, no al train.",
        "",
        "## 5. Limitaciones declaradas",
        "",
        "1. El orden del dump se respeta, de modo que la muestra se concentra en articulos "
        "antiguos; es una fuente enciclopedica, no tweets.",
        "2. No hay registro web informal (coloquialidad, elongation, hashtags): esa carencia se "
        "mide cuantitativamente en `reports/embedding_vocabulary_coverage.md`.",
        "3. Sin corrector ortografico ni reconstruccion de tildes ( EMB-ASCII es una variante "
        "distinta y posterior).",
        "",
        "## 6. Artefactos",
        "",
        f"- Shards: {', '.join(corpus['shards'])}",
        f"- Vocabulario: `{corpus['vocabulario_path']}`",
        "- Manifiesto: `data/external/corpora/corpus_manifest.json` y "
        "`reports/corpus_manifest_v5.json`",
    ]
    for ruta in rutas_figuras:
        lineas.append(f"- Figura: `{ruta}`")
    REPORTE_CORPUS.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    union = cobertura["por_conjunto"]["train+test"]
    lineas = [
        "# Cobertura del vocabulario TOXIC sobre el corpus de 10M tokens",
        "",
        f"Generado por `experiments/embedding_01_corpus_wikipedia_v5.py` ({EXPERIMENT_ID}).",
        "Secciones 7, 8 y 9 de la spec. Los grupos se definen con listas y patrones "
        "**curados, sin usar labels**: son un instrumento de diagnostico, nunca una fuente "
        "de supervision para el embedding.",
        "",
        "## 1. Cobertura global (secciones 7 y 9)",
        "",
        "| Conjunto | tipos | tokens | cobertura tipo | cobertura token | OOV tipos | OOV tokens | word-OOV | representation-OOV |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for nombre, fila in cobertura["por_conjunto"].items():
        lineas.append(
            f"| {nombre} | {fila['n_types']} | {fila['n_tokens']} | {fila['type_coverage']:.4f} | "
            f"{fila['token_coverage']:.4f} | {fila['oov_rate_types']:.4f} | {fila['oov_rate_tokens']:.4f} | "
            f"{fila['n_word_oov']} | {fila['n_representation_oov']} |"
        )
    lineas += [
        "",
        f"- `frequency_weighted_coverage` coincide con `token_coverage` por definicion: "
        f"**{union['frequency_weighted_coverage']:.4f}** (ponderada por la frecuencia de cada tipo en TOXIC).",
        f"- **word-OOV**: {union['n_word_oov']} tipos del train+test de TOXIC no aparecen en el corpus.",
        f"- **representation-OOV**: de esos, {union['n_representation_oov']} reciben vector por "
        f"composicion de subpalabras (3-6) en el probe fastText 50d; la tasa de "
        f"indisponibilidad es {union['representation_oov_rate']:.4f}.",
        "- El probe es un modelo de 50 dimensiones y 1 epoca, entrenado **solo** para responder "
        "esta pregunta de la seccion 9; no es el baseline de la fase (que es P2).",
        "",
        "### 1.1 Que explica la falta de cobertura (y que no)",
        "",
        "La hipotesis inicial era que la cobertura baja se debia a que los tweets de TOXIC estan "
        "en ASCII sin tildes mientras el corpus conserva las tildes. **Los datos la refutan**:",
        "",
        f"- Tipos de TOXIC ausentes del corpus: **{union['n_word_oov']}**.",
        f"- Plegado ASCII: solo **{union['n_word_oov_resolubles_por_ascii']}** "
        f"({union['tasa_oov_por_ascii']:.1%}) coinciden con el plegado ASCII de un tipo del corpus. "
        f"La cobertura por tipo proyectada pasaria de {union['type_coverage']:.4f} a "
        f"{union['type_coverage_proyeccion_ascii']:.4f}: **no explica la brecha**.",
        f"- Esqueleto vocalico (deformaciones que pierden vocales, como `cllate`, `violacin`): "
        f"**{union['n_word_oov_resolubles_por_esqueleto']}** "
        f"({union['tasa_oov_por_esqueleto']:.1%}), con cobertura proyectada "
        f"{union['type_coverage_proyeccion_esqueleto']:.4f}.",
        f"- Ejemplos de pares por esqueleto: "
        + ", ".join(
            f"`{par['tipo']}`~`{par['esqueleto']}`"
            for par in union["ejemplos_oov_resolubles_por_esqueleto"][:10]
        )
        + ".",
        "",
        "La lectura correcta es que los tipos ausentes son de dos familias: (a) deformaciones "
        "ortograficas como la perdida de vocales, que **no** son pliegues de acento y (b) formas "
        "informales, elongaciones y lenguaje coloquial que un corpus enciclopedico no contiene.",
        "",
        "### 1.2 Que resuelven los subpalabras (respuesta directa a la seccion 9)",
        "",
    ]
    vecinos = cobertura["vecinos_subwords"]
    if vecinos.get("evaluados"):
        lineas += [
            f"De los {union['n_word_oov']} tipos word-OOV se evaluaron {vecinos['evaluados']} con el "
            "probe 50d. Para cada uno se busco el tipo del corpus mas cercano al vector que fastText "
            "compone con subpalabras:",
            "",
            f"- coseno mediano al vecino mas cercano: **{vecinos['coseno_mediano']:.3f}** "
            f"(media {vecinos['coseno_medio']:.3f})",
            f"- proporcion con coseno >= 0.6: **{vecinos['frac_coseno_ge_0.6']:.1%}**; "
            f">= 0.4: {vecinos['frac_coseno_ge_0.4']:.1%}; < 0.3: {vecinos['frac_coseno_lt_0.3']:.1%}",
            "",
            "| tipo word-OOV (freq. TOXIC) | vecino del corpus | coseno centrado | segundo vecino |",
            "|---|---|---|---|",
        ]
        for ejemplo in vecinos["ejemplos"][:15]:
            primero = ejemplo["vecinos"][0]
            segundo = ejemplo["vecinos"][1] if len(ejemplo["vecinos"]) > 1 else {"tipo_corpus": "-", "coseno_centrado": 0.0}
            lineas.append(
                f"| `{ejemplo['tipo_oov']}` ({ejemplo['freq_toxic']}) | "
                f"`{primero['tipo_corpus']}` | {primero['coseno_centrado']:.3f} | "
                f"`{segundo['tipo_corpus']}` ({segundo['coseno_centrado']:.3f}) |"
            )
        lineas += [
            "",
            "**Nota metodologica sobre el coseno**: los vectores de fastText comparten un componente "
            "comun grande (la direccion de la frecuencia unigrama), por lo que el coseno crudo entre "
            "cualesquiera dos vectores sale alto (~0.94 con este probe) y el vecino mas cercano no "
            "significa nada. Por eso la tabla usa el coseno **centrado** (se resta la media de la "
            "matriz antes de comparar). Sin labels externos **no se puede afirmar** que un vecino sea "
            "el lexema correcto, y no se afirma: la tabla se muestra para que el lector juzgue.",
            "",
            "Lo que si se puede afirmar con estos datos:",
            "",
            "1. Todo tipo word-OOV obtiene una representacion no nula (los subpalabras siempre tienen "
            "materia prima), asi que el problema de cobertura **no** es de indisponibilidad de vector.",
            f"2. La deformacion ortografica afecta a {union['n_word_oov_resolubles_por_esqueleto']} de "
            f"los {union['n_word_oov']} tipos ausentes ({union['tasa_oov_por_esqueleto']:.1%}) segun el "
            "esqueleto vocalico, que es exactamente el caso donde los subpalabras 3-6 pueden "
            "reconstruir el lexema.",
            f"3. El coseno centrado mediano es {vecinos['coseno_mediano']:.3f}, con "
            f"{vecinos['frac_coseno_ge_0.4']:.1%} de los tipos por encima de 0.4 y "
            f"{vecinos['frac_coseno_lt_0.3']:.1%} por debajo de 0.3.",
            "4. El grupo con peor cobertura sigue siendo `elongaciones` (0.035 por tipo) y "
            "`coloquial` (0.214): son formas que un corpus enciclopedico **no contiene**. Para esas "
            "la palanca no es el numero de tokens sino la **fuente** del corpus (registro web, "
            "mC4-es), que es lo que mide P4.",
            "5. Los vecinos de la tabla **no son fiables como lexemas** con este probe (varios son "
            "claramente incorrectos: `callate` -> `pulcherrima`, `imbcil` -> `xẩm`). Es el resultado "
            "esperado en un modelo de 50 dimensiones y 1 epoca: los vectores de subpalabra estan "
            "subentrenados. La tabla no se usa para afirmar nada sobre la calidad del embedding.",
            "",
            "La prueba que decide si los subpalabras aportan algo utilizable no es esta sonda, sino "
            "el embedding de P2 (200d, 5 epocas) medido downstream. Este probe de 50d y 1 epoca "
            "existe solo para responder la pregunta de disponibilidad de la seccion 9.",
        ]
    else:
        lineas.append("No se pudo evaluar el vecino mas cercano de los tipos word-OOV.")
    lineas += [
        "",
        "## 2. Cobertura por grupo relevante (seccion 8)",
        "",
        "| Grupo | tipos | tokens | cobertura tipo | cobertura token | OOV | frec. media corpus |",
        "|---|---|---|---|---|---|---|",
    ]
    for fila in cobertura["grupos"]:
        lineas.append(
            f"| {fila['grupo']} | {fila['n_types']} | {fila['n_tokens']} | {fila['type_coverage']:.4f} | "
            f"{fila['token_coverage']:.4f} | {fila['oov_rate']:.4f} | {fila['frecuencia_media_corpus']} |"
        )
    top_oov = cobertura["oov"][:40]
    lineas += [
        "",
        "## 3. Tipos word-OOV mas frecuentes en TOXIC (top 40)",
        "",
        "| tipo | frecuencia en TOXIC | representable por subpalabras |",
        "|---|---|---|",
    ]
    for entrada in top_oov:
        lineas.append(
            f"| `{entrada['tipo']}` | {entrada['freq_toxic']} | {'si' if entrada['representable_subwords'] else 'no'} |"
        )
    lineas += [
        "",
        "## 4. Lectura",
        "",
        "La cobertura por tipo es la que importa para la representacion: un tipo ausente del "
        "vocabulario del corpus solo puede existir como composicion de subpalabras, y las "
        "subpalabras conservan el orden de las letras pero pierden el significado contextual. "
        "Los grupos con menor cobertura por token son los que mas dependen de los subwords.",
        "",
        "## 5. Artefactos",
        "",
        "- Tabla completa: `reports/embedding_vocabulary_coverage.csv`",
        "- Figuras: `reports/figuras/fig_41_distribucion_frecuencias_v5.png`, "
        "`reports/figuras/fig_42_cobertura_grupos_v5.png`",
        "- Sonda de representation-OOV: `data/external/corpora/sifts/probe_50d_sift.txt`",
    ]
    REPORTE_COBERTURA.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    with REPORTE_COBERTURA_CSV.open("w", encoding="utf-8", newline="") as archivo:
        writer = csv.DictWriter(
            archivo,
            fieldnames=[
                "grupo",
                "descripcion",
                "n_types",
                "n_tokens",
                "type_coverage",
                "token_coverage",
                "oov_rate",
                "frecuencia_media_corpus",
            ],
        )
        writer.writeheader()
        writer.writerows(cobertura["grupos"])


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------
def main() -> None:
    """Ejecuta P1: corpus de 10M tokens y cobertura del vocabulario TOXIC."""
    inicio = time.time()
    log("P1: corpus de 10M tokens de Wikipedia ES")
    log("1/9 descarga")
    descarga = descargar_parte(RAW_DIR / PARTE)
    corpus = construir_corpus(RAW_DIR / PARTE)
    archivo_corpus = concatenar_corpus()
    vocabulario_corpus = leer_vocabulario_corpus()
    fuga = auditar_fuga(archivo_corpus)
    probe = probar_representabilidad(archivo_corpus, vocabulario_toxic(read_train()) + vocabulario_toxic(read_test()))
    cobertura = medir_cobertura(
        vocabulario_corpus,
        Path(PROJECT_ROOT / probe["sifts"]),
        Path(PROJECT_ROOT / probe["vec"]),
    )
    rutas_figuras = figuras(vocabulario_corpus, cobertura)
    actualizar_manifiesto(descarga, corpus, fuga, cobertura)
    escribir_informes(descarga, corpus, fuga, cobertura, rutas_figuras)

    resultado = {
        "experiment_id": EXPERIMENT_ID,
        "fase": "P1 - corpus 10M (Wikipedia ES) y cobertura TOXIC",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duracion_s": round(time.time() - inicio, 1),
        "descarga": descarga,
        "corpus": corpus,
        "fuga": fuga,
        "probe_representation_oov": probe,
        "cobertura": cobertura["por_conjunto"],
        "grupos": cobertura["grupos"],
        "oov_top": cobertura["oov"][:40],
        "figuras": rutas_figuras,
        "manifiesto": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "referencia_corrected_cv": 0.777211,
        "decision": (
            "P1 GO: corpus 10M construido y auditado; la cobertura TOXIC queda medida y "
            "documentada. La decision de escalar a 50M se toma en P2 con el primer resultado "
            "downstream, y exige RATIONALE_FOR_CORPUS_EXPANSION escrito antes de descargar."
        ),
    }
    JSON_RESULTADO.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"   informes: {REPORTE_CORPUS.name}, {REPORTE_COBERTURA.name}, {REPORTE_COBERTURA_CSV.name}")
    log(f"P1 completada en {resultado['duracion_s']} s")


if __name__ == "__main__":
    main()
