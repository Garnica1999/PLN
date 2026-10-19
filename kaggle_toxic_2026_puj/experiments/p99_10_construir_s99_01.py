#!/usr/bin/env python
"""
P99-10: construccion del corpus S99-01 (60M tokens unicos, local).

Plan pre-registrado: reports/p99_corpus_scaling_plan.md.
  - Replica y extiende la seleccion de C98-1 (p98_11) con la misma semilla.
  - wiki 41M (=60%) -> mC4 10M (tope del archivo) -> twitter 9.5M seleccion;
    dedup exacta por documento normalizado (orden wiki->mc4->twitter).
  - Holgura: si el total post-dedup < 60.0M, se extiende el prefijo de
    Wikipedia por pasos de 1M (techo del archivo) hasta >= 60.0M.
  - Verifica C98-1 subconjunto exacto, overlap vs train = 0, val de lenguaje
    de P98 presente; escribe corpus + manifiesto + resumen CSV.

Salidas: data/external/corpora/p99_s99_01/tokens_60m.txt,
artifacts/p99_s99_01_manifest.json, reports/p99_corpus_results.csv.

Uso:
    /home/carlos/venv/bin/python experiments/p99_10_construir_s99_01.py
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module  # noqa: E402

SAL_DIR = RAIZ / "data" / "external" / "corpora" / "p99_s99_01"
SAL_CORPUS = SAL_DIR / "tokens_60m.txt"
SAL_MANIFEST = RAIZ / "artifacts" / "p99_s99_01_manifest.json"
SAL_RES = RAIZ / "reports" / "p99_corpus_results.csv"

SEMILLA = 20260925
OBJETIVO_TOTAL = 60_000_000
TECHO_WIKI = 50_500_000


def _cargar(ruta: Path, nombre: str):
    """Carga un script del proyecto como modulo."""
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


m11 = _cargar(RAIZ / "experiments" / "p98_11_construir_c98_1.py", "p98_11")
m12 = _cargar(RAIZ / "experiments" / "p98_12_preentrenar_c98_1.py", "p98_12")


def sha256(ruta: Path) -> str:
    """sha256 completo."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()


def clave(doc: str) -> str:
    """sha1 del documento normalizado."""
    return hashlib.sha1(m11.normalizar(doc).encode("utf-8")).hexdigest()


def replicar_c98_1() -> dict:
    """Re-deriva la seleccion de C98-1 (docs conservados por fuente)."""
    rng = np.random.default_rng(SEMILLA)
    conservados: dict[str, list[str]] = {}
    tokens: dict[str, int] = {}
    vistos: set[str] = set()
    for nombre, spec in m11.FUENTES.items():
        docs = m11.leer_documentos(spec["ruta"])
        if spec["modo"] == "prefijo":
            elegidos, total = [], 0
            for doc, n in docs:
                if total >= spec["objetivo"]:
                    break
                elegidos.append(doc)
                total += n
        else:
            elegidos, total = [], 0
            for i in rng.permutation(len(docs)):
                doc, n = docs[i]
                if total >= spec["objetivo"]:
                    break
                elegidos.append(doc)
                total += n
        conservados[nombre], tokens[nombre] = [], 0
        for doc in elegidos:
            k = clave(doc)
            if k in vistos:
                continue
            vistos.add(k)
            conservados[nombre].append(doc)
            tokens[nombre] += len(doc.split())
    return {"docs": conservados, "tokens": tokens, "claves": vistos}


def seleccionar(wiki_objetivo: int) -> dict:
    """Selecciona y deduplica el corpus S99-01 (determinista)."""
    rng = np.random.default_rng(SEMILLA)
    wiki_docs = m11.leer_documentos(m11.FUENTES["wikipedia"]["ruta"])
    elegidos_w, total_w = [], 0
    for doc, n in wiki_docs:
        if total_w >= wiki_objetivo:
            break
        elegidos_w.append(doc)
        total_w += n
    mc4_docs = m11.leer_documentos(m11.FUENTES["mc4"]["ruta"])
    elegidos_m, total_m = [], 0
    for doc, n in mc4_docs:
        if total_m >= 10_000_000:
            break
        elegidos_m.append(doc)
        total_m += n
    tw_docs = m11.leer_documentos(m11.FUENTES["twitter"]["ruta"])
    elegidos_t, total_t = [], 0
    for i in rng.permutation(len(tw_docs)):
        doc, n = tw_docs[i]
        if total_t >= 9_500_000:
            break
        elegidos_t.append(doc)
        total_t += n
    vistos: set[str] = set()
    finales: list[tuple[str, str]] = []
    stats = {}
    for nombre, elegidos in (("wikipedia", elegidos_w), ("mc4", elegidos_m),
                             ("twitter", elegidos_t)):
        conservados = 0
        for doc in elegidos:
            k = clave(doc)
            if k in vistos:
                continue
            vistos.add(k)
            finales.append((doc, nombre))
            conservados += 1
        stats[nombre] = {"seleccionados": len(elegidos), "conservados": conservados,
                         "removidos": len(elegidos) - conservados}
    return {"finales": finales, "stats": stats, "claves": vistos}


def main() -> None:
    """Construye S99-01 completo."""
    for ruta in (SAL_CORPUS, SAL_MANIFEST, SAL_RES):
        if ruta.exists():
            raise SystemExit(f"Ya existe {ruta}; no se sobrescribe.")
    inicio = time.perf_counter()

    for nombre, rel in (("wikipedia", "data/external/corpora/wikipedia_es_50M/"
                         "emb_normal/tokens_50M.txt"),
                        ("mc4", "data/external/corpora/mc4_es/emb_normal/"
                         "tokens_10M.txt"),
                        ("twitter", "data/external/corpora/twitter_domain/"
                         "corpus_twitter_tokens.txt")):
        esperado = m11.FUENTES[nombre]["sha256_esperado"]
        real = sha256(RAIZ / rel)
        if real != esperado:
            raise SystemExit(f"hash inesperado en {nombre}")
    print("hashes de las 3 fuentes verificados", flush=True)

    ref = replicar_c98_1()
    ref_docs = sum(len(v) for v in ref["docs"].values())
    ref_tokens = sum(ref["tokens"].values())
    print(f"C98-1 replicado: {ref_docs} docs | {ref_tokens:,} tokens",
          flush=True)

    wiki_objetivo = 41_000_000
    while True:
        sel = seleccionar(wiki_objetivo)
        total = sum(len(d.split()) for d, _ in sel["finales"])
        faltan = OBJETIVO_TOTAL - total
        print(f"seleccion wiki={wiki_objetivo:,}: total={total:,} "
              f"(faltan {faltan:,})", flush=True)
        if total >= OBJETIVO_TOTAL or wiki_objetivo >= TECHO_WIKI:
            break
        wiki_objetivo += 1_000_000
    if total < OBJETIVO_TOTAL:
        raise SystemExit("No se alcanzan 60M con material local: detener y "
                         "pedir aprobacion (plan seccion 6).")

    faltantes = ref["claves"] - sel["claves"]
    if faltantes:
        raise SystemExit(f"C98-1 no es subconjunto exacto: {len(faltantes)} docs "
                         f"de referencia no aparecen en S99-01")
    print("C98-1 subconjunto verificado (0 documentos de referencia faltantes)",
          flush=True)

    rng_final = np.random.default_rng(SEMILLA)
    orden = rng_final.permutation(len(sel["finales"]))
    finales = [sel["finales"][i] for i in orden]

    SAL_DIR.mkdir(parents=True, exist_ok=True)
    tokens_por_fuente: dict[str, int] = {}
    docs_por_fuente: dict[str, int] = {}
    total_tokens = 0
    with open(SAL_CORPUS, "w", encoding="utf-8") as fh:
        for doc, origen in finales:
            fh.write(doc + "\n")
            n = len(doc.split())
            total_tokens += n
            tokens_por_fuente[origen] = tokens_por_fuente.get(origen, 0) + n
            docs_por_fuente[origen] = docs_por_fuente.get(origen, 0) + 1
    print(f"corpus escrito: {total_tokens:,} tokens | {len(finales):,} docs",
          flush=True)

    train = data_module.read_train()
    textos_norm = {hashlib.sha1(m11.normalizar(t).encode("utf-8")).hexdigest()
                   for t in train["text"]}
    coincidencias = sum(1 for doc, _ in finales if clave(doc) in textos_norm)
    claves_val = m12.val_set(m11, m12.hashes_por_fuente(m11))
    val_presentes = sum(1 for k in claves_val if k in sel["claves"])

    manifiesto = {
        "experiment_id": "P99-10", "fase": "S99-01 (construccion)",
        "fecha": "2026-10-13", "seed": SEMILLA,
        "objetivo": OBJETIVO_TOTAL, "wiki_objetivo_final": wiki_objetivo,
        "tokens_finales": total_tokens, "documentos_finales": len(finales),
        "tokens_por_fuente": tokens_por_fuente,
        "documentos_por_fuente": docs_por_fuente,
        "semilla_stats": sel["stats"],
        "referencia_c98_1": {"documentos": ref_docs, "tokens": ref_tokens,
                             "presente_integro": True},
        "tokens_nuevos_unicos": total_tokens - ref_tokens,
        "documentos_nuevos": len(finales) - ref_docs,
        "coincidencias_exactas_vs_train": coincidencias,
        "val_language_docs": len(claves_val), "val_presentes": val_presentes,
        "sha256": sha256(SAL_CORPUS),
        "ruta": str(SAL_CORPUS.relative_to(RAIZ)),
        "criterios": "pre-registrado en p99_corpus_scaling_plan.md; material "
                     "local; sin descargas; dedup exacta",
        "tiempo_s": round(time.perf_counter() - inicio, 1),
    }
    SAL_MANIFEST.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=True)
                            + "\n", encoding="utf-8")
    filas = ["fuente,documentos,tokens,tokens_referencia_c98_1,removidos_dedup,"
             "tokens_nuevos"]
    for nombre in ("wikipedia", "mc4", "twitter"):
        filas.append(f"{nombre},{docs_por_fuente.get(nombre,0)},"
                     f"{tokens_por_fuente.get(nombre,0)},"
                     f"{ref['tokens'][nombre]},{sel['stats'][nombre]['removidos']},"
                     f"{tokens_por_fuente.get(nombre,0) - ref['tokens'][nombre]}")
    SAL_RES.write_text("\n".join(filas) + "\n", encoding="utf-8")

    print(f"S99-01: {total_tokens:,} tokens | {len(finales):,} docs | nuevos "
          f"{manifiesto['tokens_nuevos_unicos']:,} tokens | val presentes "
          f"{val_presentes}/{len(claves_val)} | ovelap train {coincidencias}",
          flush=True)
    print(f"sha256 {manifiesto['sha256'][:16]} | tiempo "
          f"{manifiesto['tiempo_s']} s", flush=True)


if __name__ == "__main__":
    main()
