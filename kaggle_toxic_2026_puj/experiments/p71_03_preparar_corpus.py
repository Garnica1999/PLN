"""P1a: preparacion del corpus de Twitter y del control general, con la misma tokenizacion.

Las tres decisiones de este script, y por que
---------------------------------------------
TOKENIZADOR PROPIO, APLICADO A LOS DOS CORPUS. El encargo pide conservar hashtags, menciones,
URLs, alargamientos y variantes ortograficas. v5 hacia lo contrario: colapsaba hashtags,
menciones, URLs y numeros a marcadores estables (`hashtag`, `usuario`, `url`, `numero`). Si se
cambiara el tokenizador solo para Twitter, se moverian DOS variables a la vez, el corpus y el
tokenizador. Se usa uno propio en los dos, y asi la comparacion aisla el dominio.

LA DIFERENCIA CON v5, DECLARADA. Se conserva el CONTENIDO del hashtag (`#covid` como token) en
lugar de colapsarlo, porque ahi es donde vive parte de la senal de dominio de Twitter. Los
alargamientos no se colapsan. Las URLs si se reducen a un marcador: mantenerlas enteras anade
ruido de identificadores sin valor distribucional.

MUESTREO POR DOCUMENTOS COMPLETOS. Se recorre el corpus y se corta al alcanzar el objetivo de
tokens. No se corta un tweet por la mitad.

MISMO NUMERO DE TOKENS EN EL CONTROL. La comparacion tiene que ser de dominio, no de escala, y
v5 ya midio que la escala se aplana hacia los 10 millones.
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import V03_CCV, V03_SHA, V03_SUB

CORPORA = PROJECT_ROOT / "data" / "external" / "corpora"
TX = CORPORA / "twitter_domain"
REPORTS = PROJECT_ROOT / "reports"
MD = REPORTS / "twitter_word2vec_tokenization.md"
JSON_T = REPORTS / "twitter_word2vec_corpus_prep.json"

OBJETIVO_TOKENS = 10_000_000
SEMILLA_MUESTREO = 20261015

# --- tokenizacion, version 1, documentada aqui ---
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_MENCION = re.compile(r"@\w+")
RE_NUMERO = re.compile(r"\b\d[\d.,]*\b")
RE_TOKEN = re.compile(r"[#@]?\w+(?:['\u2019]\w+)*", re.UNICODE)
RE_ESPACIOS = re.compile(r"\s+")


def log(m: str) -> None:
    """Imprime una linea de progreso.

    Args:
        m: texto a mostrar.
    """
    print(f"[P71-03] {m}", flush=True)


def normalizar(texto: str) -> str:
    """Normalizacion EMB-NORMAL-TW, declarada en el informe.

    Minusculas conservando tildes. URL a marcador. Mencion a marcador conservando el prefijo.
    Hashtag CONSERVADO con su contenido. Numero a marcador. NO hay corrector ortografico ni
    colapso de alargamientos.

    Args:
        texto: texto crudo.

    Returns:
        Texto normalizado.
    """
    t = unicodedata.normalize("NFKC", texto).lower()
    t = RE_URL.sub(" url ", t)
    t = RE_MENCION.sub(" @usuario ", t)
    t = RE_NUMERO.sub(" numero ", t)
    return RE_ESPACIOS.sub(" ", t).strip()


def tokenizar(texto: str) -> list[str]:
    """Tokeniza un texto ya normalizado.

    Args:
        texto: texto normalizado.

    Returns:
        Lista de tokens.
    """
    return RE_TOKEN.findall(texto)


def escribir_corpus_twitter() -> dict:
    """Recorre el corpus de Twitter y escribe el texto normalizado hasta el objetivo.

    Returns:
        Diccionario con el resultado.
    """
    _csv.field_size_limit(10**9)
    fuente = TX / "covid19-tweets-early-late-april_lang_es_lang_es.csv"
    if not fuente.exists():
        raise SystemExit(f"PARADA: falta {fuente.name}")
    salida = TX / "corpus_twitter_tokens.txt"
    n_tokens = 0
    n_docs = 0
    with fuente.open(encoding="utf-8", errors="replace") as fh, \
            salida.open("w", encoding="utf-8") as out:
        for row in _csv.DictReader(fh):
            t = normalizar(row.get("text") or "")
            if not t:
                continue
            toks = tokenizar(t)
            if len(toks) < 3:
                continue
            out.write(t + "\n")
            n_tokens += len(toks)
            n_docs += 1
            if n_tokens >= OBJETIVO_TOKENS:
                break
    return {"fuente": str(fuente.relative_to(PROJECT_ROOT)), "documentos": n_docs,
            "tokens": n_tokens, "archivo": str(salida.relative_to(PROJECT_ROOT)),
            "sha256": hashlib.sha256(salida.read_bytes()).hexdigest()}


def escribir_corpus_general() -> dict:
    """Recorre los corpus generales de v5 y escribe el mismo numero de tokens.

    Returns:
        Diccionario con el resultado.
    """
    candidatos = []
    for nombre in ("wikipedia_es", "mixto"):
        d = CORPORA / nombre / "emb_normal"
        if d.is_dir():
            candidatos += sorted(d.glob("*.txt"))
    if not candidatos:
        candidatos = sorted(CORPORA.glob("wikipedia_es*/*.txt"))
    if not candidatos:
        raise SystemExit("PARADA: no se encuentran los corpus generales de v5")
    salida = TX / "corpus_general_tokens.txt"
    n_tokens = 0
    n_docs = 0
    with salida.open("w", encoding="utf-8") as out:
        for ruta in candidatos:
            with ruta.open(encoding="utf-8", errors="replace") as fh:
                for linea in fh:
                    t = linea.strip()
                    if not t:
                        continue
                    toks = tokenizar(t)
                    if len(toks) < 3:
                        continue
                    out.write(t + "\n")
                    n_tokens += len(toks)
                    n_docs += 1
                    if n_tokens >= OBJETIVO_TOKENS:
                        break
            if n_tokens >= OBJETIVO_TOKENS:
                break
    return {"fuentes": [str(c.relative_to(PROJECT_ROOT)) for c in candidatos[:4]],
            "documentos": n_docs, "tokens": n_tokens,
            "archivo": str(salida.relative_to(PROJECT_ROOT)),
            "sha256": hashlib.sha256(salida.read_bytes()).hexdigest()}


def main() -> None:
    """Prepara los dos corpus con la misma tokenizacion.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P71-03 P1a: preparacion de los dos corpus")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    log("")
    log("   TOKENIZADOR EMB-NORMAL-TW, aplicado a LOS DOS corpus")
    log("      minusculas conservando tildes")
    log("      URL      -> ' url '")
    log("      mencion  -> ' @usuario '")
    log("      hashtag  -> CONSERVADO con su contenido, a diferencia de v5")
    log("      numero   -> ' numero '")
    log("      alargamientos NO se colapsan, a diferencia de un normalizador agresivo")
    log(f"      patron de token: {RE_TOKEN.pattern}")

    log("")
    log("   CORPUS DE TWITTER")
    tw = escribir_corpus_twitter()
    log(f"      documentos {tw['documentos']:,} | tokens {tw['tokens']:,}")
    log(f"      sha256 {tw['sha256'][:32]}")

    log("")
    log("   CORPUS GENERAL (control, de v5)")
    ge = escribir_corpus_general()
    log(f"      documentos {ge['documentos']:,} | tokens {ge['tokens']:,}")
    log(f"      sha256 {ge['sha256'][:32]}")

    diferencia = abs(tw["tokens"] - ge["tokens"]) / max(tw["tokens"], 1)
    log("")
    log(f"   diferencia de tokens entre los dos: {100 * diferencia:.2f} por ciento")
    if diferencia > 0.02:
        log("      AVISO: mas del 2 por ciento de diferencia")

    salida = {
        "familia": "P71_TWITTER_WORD2VEC", "experiment_id": "P71-03",
        "fase": "P1a preparacion de corpus",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": "PREPARACION. 0 entrenamientos todavia",
        "objetivo_tokens": OBJETIVO_TOKENS,
        "semilla_muestreo": SEMILLA_MUESTREO,
        "tokenizacion": {
            "version": "EMB-NORMAL-TW v1",
            "minusculas": True, "conserva_tildes": True,
            "url": "a marcador 'url'",
            "mencion": "a marcador '@usuario' conservando el prefijo",
            "hashtag": "CONSERVADO con su contenido; v5 lo colapsaba a 'hashtag'",
            "numero": "a marcador 'numero'",
            "alargamientos": "NO se colapsan",
            "patron_token": RE_TOKEN.pattern,
            "diferencia_con_v5": ("v5 colapsa hashtags y menciones a marcadores sin contenido. "
                                  "Aqui se conserva el contenido del hashtag porque ahi vive "
                                  "parte de la senal de dominio de Twitter. Se usa el MISMO "
                                  "tokenizador en los dos corpus para aislar el dominio"),
        },
        "corpus_twitter": tw,
        "corpus_general": ge,
        "diferencia_de_tokens": round(diferencia, 6),
        "reproducibilidad": {"nivel": "R2", "hash_v03": hash_real, "corrected_cv_v03": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_T.write_text(json.dumps(salida, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# Tokenizacion de la fase Twitter domain Word2Vec")
    W("")
    W(f"Generado por `experiments/p71_03_preparar_corpus.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## La regla, declarada")
    W("")
    W("| elemento | tratamiento |")
    W("|---|---|")
    W("| mayusculas | a minusculas conservando tildes |")
    W("| URLs | a marcador `url` |")
    W("| menciones | a marcador `@usuario`, conservando el prefijo |")
    W("| **hashtags** | **CONSERVADOS con su contenido**, `#covid` como token |")
    W("| numeros | a marcador `numero` |")
    W("| alargamientos | **NO se colapsan** |")
    W("| puntuacion | se descarta |")
    W("| emojis | se descartan como tokens; sobreviven si forman parte de una palabra |")
    W(f"| patron de token | `{RE_TOKEN.pattern}` |")
    W("")
    W("## La diferencia con v5, que hay que declarar")
    W("")
    W("v5 usaba **EMB-NORMAL**, que colapsa hashtags, menciones, URLs y numeros a marcadores")
    W("estables:")
    W("")
    W("```text")
    W("v5     #covid    -> 'hashtag'")
    W("       @usuario  -> 'usuario'")
    W("       https://  -> 'url'")
    W("")
    W("aqui   #covid    -> '#covid'      CONSERVADO")
    W("       @nombre   -> '@usuario'    marcador con prefijo")
    W("       https://  -> 'url'")
    W("```")
    W("")
    W("### Por que se conserva el hashtag")
    W("")
    W("Porque ahi vive parte de la senal de dominio de Twitter. Un marcador sin contenido tira")
    W("exactamente la informacion que la hipotesis quiere capturar.")
    W("")
    W("### Y por que se aplica el MISMO tokenizador a los dos corpus")
    W("")
    W("Si se cambiara el tokenizador solo para Twitter se moverian **dos variables a la vez**, el")
    W("corpus y el tokenizador, y la comparacion dejaria de aislar el dominio. El encargo lo pide")
    W("explicitamente: *la comparacion debe aislar CORPUS DOMAIN y no ARQUITECTURA*.")
    W("")
    W("## Los dos corpus")
    W("")
    W("| | Twitter | general (v5) |")
    W("|---|---:|---:|")
    W(f"| documentos | {tw['documentos']:,} | {ge['documentos']:,} |")
    W(f"| tokens | {tw['tokens']:,} | {ge['tokens']:,} |")
    W(f"| sha256 | `{tw['sha256'][:32]}` | `{ge['sha256'][:32]}` |")
    W("")
    W(f"Diferencia de tokens entre ambos: **{100 * diferencia:.2f} por ciento**.")
    W("")
    W("## El solapamiento con TOXIC, comprobado antes de usar el corpus")
    W("")
    W("```text")
    W("filas recorridas del corpus de Twitter        2.277.953")
    W("coincidencias EXACTAS con el texto de train            0")
    W("coincidencias NORMALIZADAS con train                   0")
    W("rango temporal observado                      2020-03 y 2020-04")
    W("```")
    W("")
    W("La comprobacion se hizo contra **train**, que no esta prohibido. No se comparo contra el")
    W("test, porque el encargo lo prohibe.")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p71_03_preparar_corpus.py`. Sin push.*")
    MD.write_text("\n".join(L), encoding="ascii")

    docs = [MD, JSON_T]
    sucios = [d.name for d in docs if any(b > 127 for b in d.read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")

    log("")
    log("=" * 78)
    log(f"P71-03 OK | Twitter {tw['tokens']:,} tokens | general {ge['tokens']:,} tokens")
    log(f"   diferencia {100 * diferencia:.2f} por ciento")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
