#!/usr/bin/env python
"""
P98-11 (Fase B): construccion del corpus C98-1 (30M tokens, 60/25/15).

Componentes (todos locales y autorizados; sin descargas):
  - wikipedia: prefijo POR DOCUMENTOS del archivo de 50M hasta >=18.0M tokens
  - mc4: prefijo POR DOCUMENTOS del archivo de 10M hasta >=7.5M tokens
  - twitter: muestreo uniforme POR DOCUMENTO (seed 20260925) hasta >=4.5M tokens

Deduplicacion: exacta por documento (sha1 del texto normalizado) dentro y
entre componentes; se conserva la primera aparicion.
Orden final: documentos barajados con la misma semilla (documentos intactos).
Salida: data/external/corpora/p98_c98_1/tokens_30m.txt + manifiesto propio en
artifacts/p98_c98_1_manifest.json. Auditoria exacta vs train incluida.

Uso:
    /home/carlos/venv/bin/python experiments/p98_11_construir_c98_1.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module  # noqa: E402

SAL_DIR = RAIZ / "data" / "external" / "corpora" / "p98_c98_1"
SAL_CORPUS = SAL_DIR / "tokens_30m.txt"
SAL_MANIFEST = RAIZ / "artifacts" / "p98_c98_1_manifest.json"

SEMILLA = 20260925
FUENTES = {
    "wikipedia": {
        "ruta": RAIZ / "data/external/corpora/wikipedia_es_50M/emb_normal/tokens_50M.txt",
        "objetivo": 18_000_000, "modo": "prefijo",
        "sha256_esperado": "8088bce2a9283fbc835ab62efe3fb25f40af346f9a8003fe3d101619bbc32463",
    },
    "mc4": {
        "ruta": RAIZ / "data/external/corpora/mc4_es/emb_normal/tokens_10M.txt",
        "objetivo": 7_500_000, "modo": "prefijo",
        "sha256_esperado": "361498bf3eca104d527759de68980ea358002aeeb27e57d0de8562fac79b4f97",
    },
    "twitter": {
        "ruta": RAIZ / "data/external/corpora/twitter_domain/corpus_twitter_tokens.txt",
        "objetivo": 4_500_000, "modo": "muestreo",
        "sha256_esperado": "1ef872c189d0f405bc1a3f8281cafe08e6b099958e6b8a3816db1460df91f8b9",
    },
}


def sha256(ruta: Path) -> str:
    """sha256 completo."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()


def normalizar(texto: str) -> str:
    """Minusculas y sin tildes (comparacion de duplicados)."""
    plano = unicodedata.normalize("NFKD", texto.lower())
    return " ".join("".join(c for c in plano if not unicodedata.combining(c)).split())


def leer_documentos(ruta: Path) -> list[tuple[str, int]]:
    """Lista de (documento, n_tokens) por linea."""
    docs = []
    with open(ruta, encoding="utf-8") as fh:
        for linea in fh:
            tokens = linea.split()
            if tokens:
                docs.append((linea.rstrip("\n"), len(tokens)))
    return docs


def main() -> None:
    """Construye el corpus C98-1 y su manifiesto."""
    for ruta in (SAL_CORPUS, SAL_MANIFEST):
        if ruta.exists():
            raise SystemExit(f"Ya existe {ruta}; no se sobrescribe.")
    inicio = time.perf_counter()
    rng = np.random.default_rng(SEMILLA)
    seleccion: dict[str, list[str]] = {}
    stats: dict[str, dict] = {}

    for nombre, spec in FUENTES.items():
        real = sha256(spec["ruta"])
        if real != spec["sha256_esperado"]:
            raise SystemExit(f"hash inesperado en componente {nombre}")
        docs = leer_documentos(spec["ruta"])
        if spec["modo"] == "prefijo":
            elegidos, total = [], 0
            for doc, n in docs:
                if total >= spec["objetivo"]:
                    break
                elegidos.append(doc)
                total += n
        else:
            orden = rng.permutation(len(docs))
            elegidos, total = [], 0
            for i in orden:
                doc, n = docs[i]
                if total >= spec["objetivo"]:
                    break
                elegidos.append(doc)
                total += n
        seleccion[nombre] = elegidos
        stats[nombre] = {"documentos_disponibles": len(docs),
                         "documentos_seleccionados": len(elegidos),
                         "tokens_seleccionados": total,
                         "modo": spec["modo"]}
        print(f"  {nombre}: {len(elegidos)} docs | {total:,} tokens")

    vistos: set[str] = set()
    dedup_stats = {}
    finales: list[str] = []
    orden_fuente: list[str] = []
    for nombre in ("wikipedia", "mc4", "twitter"):
        antes = len(seleccion[nombre])
        conservados = []
        for doc in seleccion[nombre]:
            clave = hashlib.sha1(normalizar(doc).encode("utf-8")).hexdigest()
            if clave in vistos:
                continue
            vistos.add(clave)
            conservados.append(doc)
        dedup_stats[nombre] = {"antes": antes, "despues": len(conservados),
                               "duplicados_removidos": antes - len(conservados)}
        finales.extend(conservados)
        orden_fuente.extend([nombre] * len(conservados))
        print(f"  dedup {nombre}: {antes} -> {len(conservados)}")

    perm = rng.permutation(len(finales))
    finales = [finales[i] for i in perm]
    orden_fuente = [orden_fuente[i] for i in perm]

    SAL_DIR.mkdir(parents=True, exist_ok=True)
    tokens_totales = 0
    with open(SAL_CORPUS, "w", encoding="utf-8") as fh:
        for doc in finales:
            fh.write(doc + "\n")
            tokens_totales += len(doc.split())

    train = data_module.read_train()
    textos_norm = {hashlib.sha1(normalizar(t).encode("utf-8")).hexdigest()
                   for t in train["text"]}
    coincidencias = sum(
        1 for doc in finales
        if hashlib.sha1(normalizar(doc).encode("utf-8")).hexdigest() in textos_norm)

    por_fuente = {}
    for origen in orden_fuente:
        por_fuente[origen] = por_fuente.get(origen, 0) + 1
    resultado = {
        "experiment_id": "P98-11",
        "etapa": "C98-1 (construccion de corpus)",
        "fecha": "2026-10-13",
        "seed": SEMILLA,
        "componentes": stats,
        "deduplicacion": dedup_stats,
        "documentos_finales": len(finales),
        "tokens_finales": tokens_totales,
        "proporcion_documentos": {k: round(v / len(finales), 4)
                                  for k, v in por_fuente.items()},
        "coincidencias_exactas_vs_train": coincidencias,
        "sha256": sha256(SAL_CORPUS),
        "ruta": str(SAL_CORPUS.relative_to(RAIZ)),
        "serie": "P98/c98_1",
        "criterios": "wikipedia >=18M prefijo por documentos; mc4 >=7.5M prefijo; "
                     "twitter >=4.5M muestreo uniforme por documento; dedup "
                     "exacta por documento entre componentes; barajado de "
                     "documentos con la semilla; sin descargas nuevas",
        "tiempo_s": round(time.perf_counter() - inicio, 1),
    }
    SAL_MANIFEST.write_text(json.dumps(resultado, indent=2, ensure_ascii=True) + "\n",
                            encoding="utf-8")
    print(f"C98-1 construido: {len(finales)} docs | {tokens_totales:,} tokens | "
          f"coincidencias exactas vs train: {coincidencias} | "
          f"sha256 {resultado['sha256'][:16]}")
    print(f"tiempo: {resultado['tiempo_s']} s")


if __name__ == "__main__":
    main()
