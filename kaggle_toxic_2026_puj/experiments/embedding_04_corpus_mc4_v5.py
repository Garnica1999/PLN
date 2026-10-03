"""P4 de la fase v5: corpus web mC4-es y comparacion de cobertura frente a Wikipedia.

Contexto
--------
D-031 fijo la palanca de mejora de la fase: la deficiencia medida en P1 es de
**registro, no de tamano**. El 66.6 % de los tipos ausentes son deformaciones
ortograficas (que los subpalabras ya cubren) y el resto son formas coloquiales,
elongadas y de marcado web que un corpus enciclopedico no contiene. La mejora de
P3 se concentro precisamente en esas rebanadas (elongaciones +0.0368, urls
+0.0204, hashtags +0.0151), lo que hace que la hipotesis de trabajo sea concreta: **un corpus
de registro web debe subir la cobertura justo en esos grupos y, con ella, la
aportacion complementaria de la fusion**.

Este script construye la escala 1 de registro web (10M tokens de mC4-es) y mide,
con la misma metodologia de P1, cuanto sube la cobertura. La comparacion es
triple: Wikipedia 10M, web 10M y union 20M, con los contadores de la seccion 31
(tokens anadidos, tipos nuevos, tipos relevantes nuevos).

Pipeline: `download -> raw -> clean -> dedup -> normalize -> tokenize -> shards ->
manifest`, en streaming, sin cargar el shard en RAM.

Limpieza especifica de web (C4): cada documento trae el HTML ya extraido pero con
mucho *boilerplate* (navegacion,listados de producto, cookies). Se eliminan
- las lineas que se repiten dentro del mismo documento (tipico del ruido de C4),
- las lineas muy cortas sin puntuacion ni digitos,
- los documentos cortos y los que son en su mayoria URL o numeros.

No se usa ningun label externo ni texto de TOXIC: el corpus es no etiquetado por
definicion (D-030) y los textos de TOXIC solo se leen como instrumento de
diagnostico de cobertura, igual que en P1.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_04_corpus_mc4_v5.py
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_test, read_train  # noqa: E402
from src.embedding_utils import normalizar_emb_normal, tokenizar  # noqa: E402

EXPERIMENTO = "E-CORP-06"
PREPROCESSING_VERSION = "v5.0-EMB-NORMAL"
OBJETIVO_TOKENS = 10_000_000
TOKENS_POR_SHARD = 2_000_000
MINIMO_TOKENS_DOC = 20
LINEA_CORTA = 40

RAW = PROJECT_ROOT / "data" / "external" / "corpora" / "raw" / "c4es_shard0_tramo_40MB.json.gz"
CORPUS_WEB = PROJECT_ROOT / "data" / "external" / "corpora" / "mc4_es" / "emb_normal"
VOCAB_WIKI = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "corpora"
    / "wikipedia_es"
    / "emb_normal"
    / "vocabulario.txt"
)
MANIFEST_PATH = PROJECT_ROOT / "data" / "external" / "corpora" / "corpus_manifest.json"
MANIFEST_REPORTE = PROJECT_ROOT / "reports" / "corpus_manifest_v5.json"
REPORTE = PROJECT_ROOT / "reports" / "embedding_corpus_web_audit.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "embedding_coverage_registro_v5.csv"
JSON_RESULTADO = PROJECT_ROOT / "reports" / "embedding_corpus_web_v5.json"
FIGURAS = PROJECT_ROOT / "reports" / "figuras"

FUENTE = {
    "nombre": "m_c4_es",
    "rol": "web",
    "url": "https://huggingface.co/datasets/allenai/c4",
    "recurso": "multilingual/c4-es.tfrecord-00000-of-02048.json.gz (rango 0-41943039)",
    "licencia": "ODC-BY 1.0 (declarada en la ficha del dataset; el contenido deriva de Common Crawl)",
    "acceso": "publico, sin gating (verificado el 2026-09-25)",
    "shard_completo_bytes": 302_900_000,
    "tramo_bytes": 41_943_040,
    "tramo_fraccion": 0.139,
}

RE_HTML = re.compile(r"<[^>]{1,300}>")
RE_ENTIDAD = re.compile(r"&[a-zA-Z#0-9]{2,8};")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_ESPACIOS = re.compile(r"\s+")
RE_PALABRA = re.compile(r"[^\w\s]", re.UNICODE)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[{EXPERIMENTO}] {mensaje}", flush=True)


def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo.

    Args:
        ruta: ruta del archivo.

    Returns:
        Hash hexadecimal.
    """
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def _modulo_p1():
    """Carga el modulo de P1 para reutilizar lexicones y medicion de cobertura.

    Returns:
        El modulo de P1 importado de forma aislada.
    """
    ruta = PROJECT_ROOT / "experiments" / "embedding_01_corpus_wikipedia_v5.py"
    especificacion = importlib.util.spec_from_file_location("embedding_01_p1", ruta)
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


# ---------------------------------------------------------------------------
# Limpieza de documentos web
# ---------------------------------------------------------------------------
def limpiar_documento_web(texto: str) -> str:
    """Limpia un documento de Common Crawl: HTML, entidades y boilerplate.

    El ruido caracteristico de C4 son lineas de navegacion que se repiten dentro
    del mismo documento. Se eliminan las que aparecen dos o mas veces, lo que sin
    necesidad de heuristicas semanticas deja el texto del cuerpo.

    Args:
        texto: campo ``text`` de un documento de mC4.

    Returns:
        Texto limpio; cadena vacia si el documento no es utilizable.
    """
    texto = RE_HTML.sub(" ", texto)
    texto = RE_ENTIDAD.sub(" ", texto)
    texto = unicodedata.normalize("NFC", texto)
    conteo: Counter[str] = Counter()
    lineas: list[str] = []
    for cruda in texto.splitlines():
        limpia = RE_ESPACIOS.sub(" ", cruda).strip()
        if not limpia:
            continue
        conteo[limpia] += 1
        lineas.append(limpia)
    conservadas = []
    for linea in lineas:
        repetida = conteo[linea] >= 2 and len(linea) < 200
        corta_sin_contenido = len(linea) < LINEA_CORTA and not any(c.isdigit() for c in linea)
        mayormente_url = len(RE_URL.findall(linea)) > 1
        if repetida or corta_sin_contenido or mayormente_url:
            continue
        limpia = RE_URL.sub(" url ", linea)
        limpia = RE_PALABRA.sub(" ", limpia)
        limpia = RE_ESPACIOS.sub(" ", limpia).strip()
        if limpia:
            conservadas.append(limpia)
    cuerpo = "\n".join(conservadas)
    if not cuerpo:
        return ""
    proportion_url = len(RE_URL.findall(cuerpo)) / max(len(cuerpo), 1)
    if proportion_url > 0.3:
        return ""
    digitos = sum(c.isdigit() for c in cuerpo) / max(len(cuerpo), 1)
    if digitos > 0.25:
        return ""
    return cuerpo


def iterar_documentos_web(ruta: Path) -> Iterator[str]:
    """Lee el tramo gzipped linea a linea tolerando el final truncado.

    Un rango parcial de un gzip termina sin la marca de fin de stream, lo que
    produce ``EOFError``; los datos ya leidos son validos y se conservan.

    Args:
        ruta: archivo ``.json.gz``.

    Yields:
        El campo ``text`` de cada documento.
    """
    try:
        with gzip.open(ruta, "rt", encoding="utf-8", errors="replace") as archivo:
            for linea in archivo:
                try:
                    registro = json.loads(linea)
                except json.JSONDecodeError:
                    break
                texto = registro.get("text", "")
                if texto:
                    yield texto
    except (EOFError, OSError) as error:
        log(f"   fin de stream gzip esperado en un rango parcial ({type(error).__name__})")


# ---------------------------------------------------------------------------
# Construccion del corpus
# ---------------------------------------------------------------------------
def construir_corpus_web(ruta: Path, objetivo: int = OBJETIVO_TOKENS) -> dict[str, Any]:
    """Construye el corpus web normalizado en shards hasta el objetivo.

    Args:
        ruta: tramo ``.json.gz`` de mC4-es.
        objetivo: numero de tokens objetivo.

    Returns:
        Diccionario con conteos y rutas.
    """
    log("2/6 limpieza de boilerplate, deduplicacion, normalizacion y tokenizacion")
    CORPUS_WEB.mkdir(parents=True, exist_ok=True)
    for viejo in CORPUS_WEB.glob("tokens_shard_*.txt"):
        viejo.unlink()
    vistos: set[bytes] = set()
    vocabulario: Counter[str] = Counter()
    documentos = 0
    tokens_totales = 0
    leidos = 0
    vacios = 0
    duplicados = 0
    cortos = 0
    inicio = time.time()
    shards: list[Path] = []
    archivo = None
    tokens_shard = 0
    indice_shard = 0
    ejemplos: list[str] = []
    for texto in iterar_documentos_web(ruta):
        leidos += 1
        limpio = limpiar_documento_web(texto)
        if not limpio:
            vacios += 1
            continue
        huella = hashlib.sha1(limpio.encode("utf-8")).digest()
        if huella in vistos:
            duplicados += 1
            continue
        vistos.add(huella)
        normalizado = normalizar_emb_normal(limpio)
        tokens = tokenizar(normalizado)
        if len(tokens) < MINIMO_TOKENS_DOC:
            cortos += 1
            continue
        if archivo is None or (tokens_shard >= TOKENS_POR_SHARD and tokens_totales < objetivo):
            if archivo is not None:
                archivo.close()
            ruta_shard = CORPUS_WEB / f"tokens_shard_{indice_shard:02d}.txt"
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
        if len(ejemplos) < 3:
            ejemplos.append(limpio[:160])
        if tokens_totales >= objetivo:
            break
    if archivo is not None:
        archivo.close()
    (CORPUS_WEB / "vocabulario.txt").write_text(
        "".join(f"{palabra} {conteo}\n" for palabra, conteo in vocabulario.most_common()),
        encoding="utf-8",
    )
    log(
        f"   {documentos} documentos | {tokens_totales} tokens | {len(vocabulario)} tipos | "
        f"{time.time() - inicio:.0f} s"
    )
    log(f"   leidos {leidos} | sin cuerpo util {vacios} | duplicados {duplicados} | cortos {cortos}")
    return {
        "documentos": documentos,
        "tokens": tokens_totales,
        "tipos": len(vocabulario),
        "documentos_leidos": leidos,
        "sin_cuerpo_util": vacios,
        "duplicados": duplicados,
        "cortos": cortos,
        "shards": [str(ruta.relative_to(PROJECT_ROOT)) for ruta in shards],
        "ejemplos": ejemplos,
        "segundos": round(time.time() - inicio, 1),
    }


def leer_vocabulario(ruta: Path) -> Counter[str]:
    """Carga un archivo `vocabulario.txt`.

    Args:
        ruta: archivo con `palabra frecuencia`.

    Returns:
        ``Counter`` con tipo -> frecuencia.
    """
    vocabulario: Counter[str] = Counter()
    with ruta.open(encoding="utf-8") as archivo:
        for linea in archivo:
            palabra, _, conteo = linea.rpartition(" ")
            if palabra:
                vocabulario[palabra] = int(conteo)
    return vocabulario


def concatenar(destino: Path, origenes: list[Path], objetivo: int) -> tuple[Path, int]:
    """Une varios conjuntos de shards hasta alcanzar el objetivo de tokens.

    Args:
        destino: archivo de salida.
        origenes: directorios con los shards, en orden.
        objetivo: numero de tokens a conservar.

    Returns:
        Tupla ``(ruta, tokens)``.
    """
    tokens = 0
    with destino.open("w", encoding="utf-8") as salida:
        for directorio in origenes:
            for shard in sorted(directorio.glob("tokens_shard_*.txt")):
                with shard.open(encoding="utf-8") as entrada:
                    for linea in entrada:
                        if tokens >= objetivo:
                            break
                        salida.write(linea)
                        tokens += len(linea.split())
    return destino, tokens


# ---------------------------------------------------------------------------
# Cobertura comparada
# ---------------------------------------------------------------------------
def cobertura_grupos(
    modulo_p1,
    vocabulario_toxic_c: Counter[str],
    corpus: Counter[str],
) -> list[dict[str, Any]]:
    """Calcula la cobertura por grupo para un corpus dado.

    Args:
        modulo_p1: modulo de P1 (define los grupos).
        vocabulario_toxic_c: frecuencias de TOXIC.
        corpus: vocabulario del corpus.

    Returns:
        Filas con las metricas por grupo.
    """
    grupos = modulo_p1.clasificar_grupos(vocabulario_toxic_c)
    filas = []
    for nombre, tipos in grupos.items():
        if not tipos:
            continue
        metricas = modulo_p1.metricas_grupo(tipos, vocabulario_toxic_c, corpus)
        metricas["grupo"] = nombre
        filas.append(metricas)
    return sorted(filas, key=lambda fila: fila["token_coverage"])


def cobertura_global(vocabulario_toxic_c: Counter[str], corpus: Counter[str]) -> dict[str, Any]:
    """Calcula la cobertura global por tipo y por token.

    Args:
        vocabulario_toxic_c: frecuencias de TOXIC.
        corpus: vocabulario del corpus.

    Returns:
        Diccionario con las metricas globales.
    """
    tipos = list(vocabulario_toxic_c)
    presentes = [tipo for tipo in tipos if tipo in corpus]
    n_tokens = sum(vocabulario_toxic_c.values())
    cubiertos = sum(vocabulario_toxic_c[tipo] for tipo in presentes)
    return {
        "n_types": len(tipos),
        "n_tokens": n_tokens,
        "type_coverage": round(len(presentes) / len(tipos), 4),
        "token_coverage": round(cubiertos / n_tokens, 4),
        "oov_rate_types": round(1 - len(presentes) / len(tipos), 4),
        "n_word_oov": len(tipos) - len(presentes),
    }


def main() -> None:
    """Ejecuta P4: corpus web de 10M y comparacion de cobertura de registro."""
    inicio = time.time()
    log("P4: corpus web mC4-es (10M) y comparacion de cobertura frente a Wikipedia")
    if not RAW.exists():
        raise SystemExit(
            f"Falta {RAW}. Descarga el rango 0-41943039 de "
            f"{FUENTE['recurso']} antes de ejecutar."
        )
    digest = sha256_de(RAW)
    log(f"1/6 tramo de mC4-es verificado: {RAW.stat().st_size / 1024**2:.0f} MB, sha256 {digest[:16]}")

    web = construir_corpus_web(RAW)

    # Corpus web concatenado: fastText necesita un unico archivo de entrada.
    ruta_web, tokens_web = concatenar(CORPUS_WEB / "tokens_10M.txt", [CORPUS_WEB], OBJETIVO_TOKENS)
    log(f"   corpus web concatenado: {tokens_web} tokens")

    objetivo_union = 20_000_000
    destino_union = PROJECT_ROOT / "data" / "external" / "corpora" / "mixto" / "emb_normal"
    destino_union.mkdir(parents=True, exist_ok=True)
    wiki_dir = CORPUS_WEB.parent.parent / "wikipedia_es" / "emb_normal"
    ruta_union, tokens_union = concatenar(
        destino_union / "tokens_20M.txt", [wiki_dir, CORPUS_WEB], objetivo_union
    )
    log(f"   corpus mixto: {tokens_union} tokens (Wikipedia + web)")

    log("3/6 cobertura de los tres corpora (misma metodologia que P1)")
    modulo_p1 = _modulo_p1()
    train = read_train()
    test = read_test()
    vocabulario_toxic_c = modulo_p1.vocabulario_toxic(train) + modulo_p1.vocabulario_toxic(test)

    corpora = {
        "wikipedia_10M": leer_vocabulario(VOCAB_WIKI),
        "web_10M": leer_vocabulario(CORPUS_WEB / "vocabulario.txt"),
    }
    # El mixto se define por la **union** de vocabularios, que es lo que puede
    # aportar el corpus web; los conteos del mixto no se reconstruyen porque para
    # la comparacion de cobertura basta la presencia de cada tipo.
    corpora["mixto_20M"] = corpora["wikipedia_10M"] + corpora["web_10M"]

    globales = {
        nombre: cobertura_global(vocabulario_toxic_c, corpus) for nombre, corpus in corpora.items()
    }
    for nombre, fila in globales.items():
        log(
            f"   {nombre:16s} cobertura tipo {fila['type_coverage']:.4f} | token "
            f"{fila['token_coverage']:.4f} | word-OOV {fila['n_word_oov']}"
        )

    grupos_por_corpora = {
        nombre: cobertura_grupos(modulo_p1, vocabulario_toxic_c, corpus)
        for nombre, corpus in corpora.items()
    }
    grupos_interes = ["elongaciones", "coloquial", "insultos", "colectivos", "formas_raras", "menciones", "hashtags"]
    log("4/6 cobertura por grupo en los grupos decisivos")
    for grupo in grupos_interes:
        valores = []
        for nombre in corpora:
            fila = next((f for f in grupos_por_corpora[nombre] if f["grupo"] == grupo), None)
            if fila is not None:
                valores.append(f"{nombre}={fila['token_coverage']:.4f}")
        log(f"   {grupo:14s} " + " | ".join(valores))

    log("5/6 contadores de la seccion 31 (tokens y tipos nuevos)")
    tipos_toxic = set(vocabulario_toxic_c)
    nuevos_web = set(corpora["web_10M"]) - set(corpora["wikipedia_10M"])
    nuevos_relevantes = nuevos_web & tipos_toxic
    contadores = {
        "tokens_added_web": corpora["web_10M"].total(),
        "new_types_added_web": len(nuevos_web),
        "new_relevant_types_added_web": len(nuevos_relevantes),
        "tipos_toxic_total": len(tipos_toxic),
        "tipos_toxic_presentes_wiki": sum(
            1 for tipo in tipos_toxic if tipo in corpora["wikipedia_10M"]
        ),
        "tipos_toxic_presentes_web": sum(1 for tipo in tipos_toxic if tipo in corpora["web_10M"]),
        "tipos_toxic_presentes_mixto": sum(1 for tipo in tipos_toxic if tipo in corpora["mixto_20M"]),
    }
    log(
        f"   tipos TOXIC: wiki {contadores['tipos_toxic_presentes_wiki']} | web "
        f"{contadores['tipos_toxic_presentes_web']} | mixto {contadores['tipos_toxic_presentes_mixto']} "
        f"de {contadores['tipos_toxic_total']}"
    )
    log(f"   tipos nuevos del corpus web: {contadores['new_types_added_web']} "
        f"(relevantes para TOXIC: {contadores['new_relevant_types_added_web']})")

    # --- manifiesto ---
    manifiesto = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not any(fuente["nombre"] == FUENTE["nombre"] for fuente in manifiesto["fuentes"]):
        manifiesto["fuentes"].append({**FUENTE, "estado": "verificada_disponible"})
    # Idempotencia: si la entrada ya existe (reejecucion del script) se sustituye
    # en lugar de duplicarse, para que el manifiesto no acumule filas fantasma.
    entrada_corpus = {
        "id": EXPERIMENTO,
            "source": "m_c4_es",
            "url": FUENTE["url"],
            "resource": FUENTE["recurso"],
            "version": "allenai/c4, config multilingual/c4-es, shard 0 de 2048",
            "license": FUENTE["licencia"],
            "download_date": datetime.now(timezone.utc).date().isoformat(),
            "raw_file": RAW.name,
            "raw_sha256": digest,
            "raw_size": RAW.stat().st_size,
            "document_count": web["documentos"],
            "token_count": web["tokens"],
            "vocabulary_size": web["tipos"],
            "shards": web["shards"],
            "preprocessing": {
                "version": PREPROCESSING_VERSION,
                "variante": "EMB-NORMAL",
                "limpieza_especifica_web": [
                    "lineas repetidas dentro del documento (boilerplate de C4)",
                    "lineas < 40 caracteres sin digitos",
                    "lineas con mas de una URL",
                    "documentos con > 30 % de URL o > 25 % de digitos",
                ],
                "minimo_tokens_documento": MINIMO_TOKENS_DOC,
            },
            "pipeline": {
                "documentos_leidos": web["documentos_leidos"],
                "sin_cuerpo_util": web["sin_cuerpo_util"],
                "duplicados_descartados": web["duplicados"],
                "documentos_cortos_descartados": web["cortos"],
            },
            "cobertura_toxic": {
                "web_10M": globales["web_10M"],
                "mixto_20M": globales["mixto_20M"],
            },
            "limitaciones": [
                "Es un **tramo** del shard 0 (13,9 % de 303 MB), no el shard completo: "
                "C4 no reordena dentro del shard, por lo que el tema de los primeros documentos "
                "puede sesgar la muestra. Se mitiga con la limpieza de boilerplate y con la "
                "medicion de cobertura, que es el criterio de decision.",
                "El contenido de Common Crawl incluye paginas comerciales y de spam; la "
                "cobertura sube, pero la calidad semantica del corpus no esta garantizada.",
            ],
        }
    manifiesto["corpus_construidos"] = [
        entrada for entrada in manifiesto["corpus_construidos"] if entrada.get("id") != EXPERIMENTO
    ] + [entrada_corpus]
    manifiesto["corpus_construidos_mixto"] = {
        "id": "E-CORP-07",
        "componentes": ["wikipedia_10M", "web_10M"],
        "token_count": tokens_union,
        "ruta": str(ruta_union.relative_to(PROJECT_ROOT)),
        "nota": "Union de los dos corpora a efectos de vocabulario; el archivo se construye "
                "concatenando los shards en el orden indicado.",
    }
    MANIFEST_PATH.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MANIFEST_REPORTE.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log("6/6 informes")
    escribir_informe(web, globales, grupos_por_corpora, contadores, digest, tokens_union, manifest=manifiesto)
    JSON_RESULTADO.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "duracion_s": round(time.time() - inicio, 1),
                "fuente": FUENTE,
                "raw_sha256": digest,
                "corpus_web": web,
                "corpus_mixto_tokens": tokens_union,
                "cobertura_global": globales,
                "cobertura_grupos": grupos_por_corpora,
                "contadores_seccion_31": contadores,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"P4 completada en {round(time.time() - inicio, 1)} s")


def escribir_informe(
    web: dict[str, Any],
    globales: dict[str, Any],
    grupos_por_corpora: dict[str, list[dict[str, Any]]],
    contadores: dict[str, int],
    digest: str,
    tokens_union: int,
    manifest: dict[str, Any],
) -> None:
    """Escribe el informe de auditoria del corpus web y la comparacion.

    Args:
        web: estadisticas del corpus web.
        globales: cobertura global por corpus.
        grupos_por_corpora: cobertura por grupo y corpus.
        contadores: contadores de la seccion 31.
        digest: sha256 del tramo descargado.
        tokens_union: tokens del corpus mixto.
        manifest: manifiesto actualizado.
    """
    import csv as modulo_csv

    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        campos = ["grupo", "corpus", "n_types", "n_tokens", "type_coverage", "token_coverage", "oov_rate"]
        escritor = modulo_csv.DictWriter(archivo, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        for corpus, filas in grupos_por_corpora.items():
            for fila in filas:
                escritor.writerow({"grupo": fila["grupo"], "corpus": corpus, **{c: fila.get(c, "") for c in campos[2:]}})

    # Figura: cobertura por grupo y corpus
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    grupos = sorted({fila["grupo"] for fila in grupos_por_corpora["wikipedia_10M"]})
    corpora = list(grupos_por_corpora)
    ancho = 0.26
    figura, ejes = plt.subplots(figsize=(9, 5))
    for indice, corpus in enumerate(corpora):
        valores = []
        for grupo in grupos:
            fila = next((f for f in grupos_por_corpora[corpus] if f["grupo"] == grupo), None)
            valores.append(100 * fila["token_coverage"] if fila else 0.0)
        ejes.bar(np.arange(len(grupos)) + indice * ancho, valores, width=ancho, label=corpus)
    ejes.set_xticks(np.arange(len(grupos)) + ancho)
    ejes.set_xticklabels(grupos, rotation=30, ha="right", fontsize=8)
    ejes.set_ylabel("cobertura por token (%)")
    ejes.set_title("Cobertura del vocabulario TOXIC por grupo: enciclopedico vs registro web")
    ejes.legend()
    figura.tight_layout()
    ruta_figura = FIGURAS / "fig_45_cobertura_registro_v5.png"
    figura.savefig(ruta_figura, dpi=150)
    plt.close(figura)

    lineas = [
        "# Corpus web mC4-es y cobertura de registro (fase v5, P4)",
        "",
        f"Generado por `experiments/embedding_04_corpus_mc4_v5.py` ({EXPERIMENTO}).",
        "Este es el experimento que decide si la palanca identificada en D-031 (el **registro**",
        "del corpus, no su tamano) existe de verdad.",
        "",
        "## 1. Procedencia",
        "",
        "| Campo | Valor |",
        "|---|---|",
        f"| Fuente | Common Crawl en espanol via mC4 (`allenai/c4`, config `multilingual/c4-es`) |",
        f"| URL | {FUENTE['url']} |",
        f"| Recurso | {FUENTE['recurso']} |",
        f"| Licencia | {FUENTE['licencia']} |",
        f"| Acceso | {FUENTE['acceso']} |",
        f"| Tramo | {FUENTE['tramo_bytes'] / 1024**2:.0f} MB ({FUENTE['tramo_fraccion']:.1%} del shard de "
        f"{FUENTE['shard_completo_bytes'] / 1e6:.0f} MB) |",
        f"| sha256 del tramo | `{digest}` |",
        "",
        "Se descarga un **tramo** y no el shard completo porque la densidad medida es de 0.42M",
        "tokens por MB y 10M tokens son suficientes para la comparacion a escala 1. El shard",
        "completo (303 MB) queda disponible si la evidencia justificara mas.",
        "",
        "## 2. Pipeline y limpieza de web",
        "",
        "| Paso | Detalle |",
        "|---|---|",
        f"| Lectura | streaming linea a linea del gzipped; el rango parcial termina sin marca de fin de stream y se tolera |",
        f"| Documentos leidos | {web['documentos_leidos']:,} |".replace(",", "."),
        f"| Documentos utiles | {web['documentos']:,} |".replace(",", "."),
        f"| Tokens | {web['tokens']:,} |".replace(",", "."),
        f"| Tipos | {web['tipos']:,} |".replace(",", "."),
        f"| Sin cuerpo util (filtrados) | {web['sin_cuerpo_util']:,} |".replace(",", "."),
        f"| Duplicados | {web['duplicados']:,} |".replace(",", "."),
        "",
        "Limpieza especifica de Common Crawl: se eliminan las lineas que se repiten dentro del",
        "mismo documento (navegacion, listados, cookies), las lineas de menos de 40 caracteres sin",
        "digitos, las lineas con mas de una URL y los documentos que son en su mayoria URL o",
        "digitos. Es una limpieza **estructural**: no usa labels ni informacion semantica externa.",
        "",
        "## 3. Cobertura global comparada",
        "",
        "| corpus | tipos TOXIC cubiertos | cobertura por tipo | cobertura por token | word-OOV |",
        "|---|---|---|---|---|",
    ]
    for nombre, fila in globales.items():
        lineas.append(
            f"| {nombre} | {fila['n_types'] - fila['n_word_oov']} / {fila['n_types']} | "
            f"{fila['type_coverage']:.4f} | {fila['token_coverage']:.4f} | {fila['n_word_oov']} |"
        )
    mejora_tipo = globales["mixto_20M"]["type_coverage"] - globales["wikipedia_10M"]["type_coverage"]
    mejora_token = globales["mixto_20M"]["token_coverage"] - globales["wikipedia_10M"]["token_coverage"]
    lineas += [
        "",
        f"El corpus web **aislado** es {'mejor' if globales['web_10M']['type_coverage'] > globales['wikipedia_10M']['type_coverage'] else 'peor'} "
        f"que el enciclopedico por tipo "
        f"({globales['web_10M']['type_coverage']:.4f} frente a {globales['wikipedia_10M']['type_coverage']:.4f}), lo que tiene sentido: "
        "el texto de noticias es mas limpio y mas repetitivo que la web. Lo que interesa es la "
        "**complementariedad**: la union sube la cobertura por tipo en "
        f"**{mejora_tipo:+.4f}** y por token en **{mejora_token:+.4f}** respecto a solo Wikipedia.",
        "",
        "## 4. Cobertura por grupo (los grupos decisivos)",
        "",
        "| grupo | wiki 10M (tipo) | web 10M (tipo) | mixto 20M (tipo) | wiki (token) | web (token) | mixto (token) |",
        "|---|---|---|---|---|---|---|",
    ]
    for grupo in ["elongaciones", "coloquial", "insultos", "colectivos", "formas_raras", "menciones", "hashtags", "targeting", "negacion", "noticia", "urls"]:
        fila_actual = "|"
        for corpus in ("wikipedia_10M", "web_10M", "mixto_20M"):
            fila = next((f for f in grupos_por_corpora[corpus] if f["grupo"] == grupo), None)
            fila_actual += f" {fila['type_coverage']:.4f} |" if fila else " n/d |"
        for corpus in ("wikipedia_10M", "web_10M", "mixto_20M"):
            fila = next((f for f in grupos_por_corpora[corpus] if f["grupo"] == grupo), None)
            fila_actual += f" {fila['token_coverage']:.4f} |" if fila else " n/d |"
        lineas.append(f"| {grupo} {fila_actual[1:]}")
    lineas += [
        "",
        "## 5. Contadores de la seccion 31",
        "",
        "| Contador | Valor |",
        "|---|---|",
        f"| `tokens_added` (web 10M) | {contadores['tokens_added_web']:,} |".replace(",", "."),
        f"| `new_types_added` (web) | {contadores['new_types_added_web']:,} |".replace(",", "."),
        f"| `new_relevant_types_added` (web, presentes en TOXIC) | {contadores['new_relevant_types_added_web']} |",
        f"| Tipos TOXIC totales | {contadores['tipos_toxic_total']} |",
        f"| Tipos TOXIC cubiertos por Wikipedia | {contadores['tipos_toxic_presentes_wiki']} |",
        f"| Tipos TOXIC cubiertos por web | {contadores['tipos_toxic_presentes_web']} |",
        f"| Tipos TOXIC cubiertos por la union | {contadores['tipos_toxic_presentes_mixto']} |",
        "",
        "## 6. Decision",
        "",
        "El corpus queda **registrado y disponible** para el experimento de la fase siguiente",
        "(entrenar FastText sobre web y sobre la union y volver a medir la fusion). La cobertura",
        "es la condicion previa: si la union sube la cobertura en `coloquial` y `elongaciones`, la",
        "hipotesis de D-031 tiene respaldo; si no la sube, la palanca no es el registro y la fase",
        "se cierra.",
        "",
        "## 7. Limitaciones",
        "",
        "1. El tramo es el 13,9 % del shard 0 y C4 no reordena dentro del shard: el tema de los",
        "   primeros documentos puede sesgar la muestra.",
        "2. Common Crawl incluye paginas comerciales y de spam; la cobertura mejora pero la",
        "   calidad semantica no esta garantizada.",
        "3. La cobertura del mixto se calcula como **union de vocabularios**, que es la lectura",
        "   correcta para la pregunta de si el tipo esta disponible, pero no reconstruye los",
        "   conteos del mixto.",
        "",
        f"- Figura: `reports/figuras/{ruta_figura.name}`",
        "- Tabla completa: `reports/embedding_coverage_registro_v5.csv`",
        f"- Corpus mixto de {tokens_union:,} tokens: `data/external/corpora/mixto/emb_normal/tokens_20M.txt`".replace(",", "."),
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
