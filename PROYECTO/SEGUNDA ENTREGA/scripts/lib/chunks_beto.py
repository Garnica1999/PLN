"""chunks_beto.py

Utilidades compartidas de tokenizacion/troceo BETO sin importar torch
directamente. Fuente unica para scripts/07_transformer.py (entrenamiento)
y scripts/14_precache_chunks_beto.py (pre-tokenizacion a disco).

ciclo v2 - Fase V (cache de chunks). Extraido de scripts/07_transformer.py
el 2026-09-11 para que el pre-cache corra en un proceso SIN el stack de
entrenamiento (torch/CUDA) y aun asi use EXACTAMENTE el mismo enmascarado,
tokenizacion y troceo. Motivo: el DataLoader multiproceso de 07 se colgaba
siempre a escala completa (Python 3.14 + fork; evidencia en
experimentos/m4_transformer_chunks_stdout_faulthandler_evidencia.log).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

# Tokenizer oficial del experimento M4 (mismo que v1 truncado).
MODELO_BETO = "dccuchile/bert-base-spanish-wwm-cased"

# Parametros de chunking de la Fase V (Chalkidis 2022): ventanas de 256
# tokens con stride 128 (solapamiento 50 %), tope de 64 chunks por doc con
# submuestreo uniforme determinista si el documento excede el tope.
CHUNK_TOKENS = 256
CHUNK_STRIDE = 128
MAX_CHUNKS = 64
VERSION_CHUNKING = 1

# Enmascarado sobre TEXTO CRUDO (mas fiable que a nivel de lemas).
# v2 (2026-09-09): grupos secuenciales sin retroceso anomalo; series
# "11, 13 y 86" en una sola CITA; sin sufijos [A-C] (falso con C.P.).
RE_CITA_CRUDO = re.compile(
    r"(?:(?:art[ií]culos?|art\.?)\s*\d{1,4}(?:\s*(?:bis|ter|quáter)\b)?"
    r"(?:\s*,?\s*(?:y|,|e)\s*(?:el\s+)?(?:art[ií]culo\s*)?\d{1,4})*"
    r"(?:\s+(?:de|del)\s+(?:la\s+)?(?:Constituci[oó]n(?:\s+Pol[ií]tica)?|C\.?\s?P\.?))?"
    r")"
    r"|(?:(?:ley(?:es)?|decretos?(?:\s+ley)?|acto\s+legislativo)\s+\d{1,4}"
    r"(?:\s+de\s+(?:19|20)\d{2})?"
    r")"
    r"|\b[CTS]U?\s?-?\s?\d{1,4}\s+(?:de\s+|/)\s?(?:19|20)\d{2}\b",
    re.IGNORECASE,
)


def enmascarar_texto_crudo(texto: str) -> str:
    """Enmascara citas normativas en el texto crudo con [CITA].

    Args:
        texto: Texto crudo de la sentencia.

    Returns:
        Texto con las citas reemplazadas por [CITA].
    """
    return RE_CITA_CRUDO.sub("[CITA]", texto)


def trocear_doc(input_ids: list[int], chunk: int, stride: int, max_chunks: int) -> list[list[int]]:
    """Trocea una secuencia tokenizada en chunks con sliding window.

    Args:
        input_ids: Tokens del documento completo (sin truncar).
        chunk: Tamano de cada ventana en tokens (256).
        stride: Paso entre ventanas (128; solapamiento 50 %).
        max_chunks: Tope de chunks por doc; si hay mas ventanas, se
            submuestrean uniformemente (determinista, np.linspace).

    Returns:
        Lista de chunks (listas de input_ids, largo variable <= chunk).
    """
    if len(input_ids) <= chunk:
        return [input_ids]
    ventanas = [input_ids[i : i + chunk] for i in range(0, len(input_ids) - chunk + 1, stride)]
    if len(ventanas) > max_chunks:
        idx = np.linspace(0, len(ventanas) - 1, max_chunks).astype(int)
        ventanas = [ventanas[i] for i in idx]
    return ventanas


def hash_config_chunking(modelo: str, chunk: int, stride: int, max_chunks: int) -> str:
    """Huella determinista de la configuracion de chunking.

    La usan 14 (al escribir la cache) y 07 (al cargarla) para detectar
    caches generadas con parametros/tokenizer distintos.

    Args:
        modelo: Nombre del tokenizer (p. ej. dccuchile/bert-base-spanish-wwm-cased).
        chunk: Tamano de ventana en tokens.
        stride: Paso entre ventanas.
        max_chunks: Tope de chunks por documento.

    Returns:
        sha256 hex del payload canonico JSON.
    """
    payload = json.dumps(
        {
            "version": VERSION_CHUNKING,
            "modelo": modelo,
            "chunk_tokens": chunk,
            "chunk_stride": stride,
            "max_chunks": max_chunks,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_ruta(ruta: str | Path) -> str:
    """Calcula sha256 de un archivo o de un dataset parquet en directorio.

    En directorio (dataset hive, p. ej. data/corpus_sentencias.parquet/)
    recorre archivos en orden de ruta relativa e incorpora esa ruta al
    hash para que el resultado sea estable ante reordenamientos.

    Args:
        ruta: Archivo o directorio a hashear.

    Returns:
        sha256 hexdigest.

    Raises:
        FileNotFoundError: si la ruta no existe.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"no existe para hashear: {ruta}")
    digest = hashlib.sha256()
    archivos = sorted(p for p in ruta.rglob("*") if p.is_file()) if ruta.is_dir() else [ruta]
    for archivo in archivos:
        if ruta.is_dir():
            digest.update(str(archivo.relative_to(ruta)).encode("utf-8"))
            digest.update(b"\0")
        with open(archivo, "rb") as manejador:
            for bloque in iter(lambda: manejador.read(1 << 20), b""):
                digest.update(bloque)
    return digest.hexdigest()
