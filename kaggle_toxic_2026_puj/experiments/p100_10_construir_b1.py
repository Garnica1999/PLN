#!/usr/bin/env python
"""
P100-10 (S3): construccion del corpus B1 (composicion social+conversacional).

Pre-registro: reports/p100_preregistration.md; autorizado por el propietario
(2026-10-13) con objetivos EXACTOS y sin superar el total.

Componentes (objetivos exactos; total 29.727.450):
  - social:         4.227.450 tokens = seleccion twitter de C98-1 (replicada)
  - conversacional: 9.000.000 tokens = ChatSubs es (muestreo uniforme de
                    archivos JSONL, semilla 20260925)
  - wikipedia:     11.650.000 tokens = prefijo por documentos de la seleccion
                    wiki de C98-1
  - mc4:            4.850.000 tokens = prefijo por documentos de la seleccion
                    mC4 de C98-1

Convencion de exactitud: el documento limite de cada componente se trunca a
nivel token para alcanzar el objetivo exacto (registrado en el manifiesto;
para social no hace falta). Deduplicacion exacta por documento normalizado
(orden wikipedia->mc4->social->conversacional).

Controles antes de S4: replicacion exacta de C98-1 (claves y tokens contra el
corpus en disco); verificacion de subconjunto (prefijos limite incluidos);
val de lenguaje fijo (3.320 claves, presentes y excluibles); overlap exacto
vs train = 0; solapamiento del conversacional vs S99-01 = 0. No se escanea
toxic_test.

Salidas: data/external/corpora/p100_b1/tokens_30m.txt,
         artifacts/p100_b1_manifest.json.

Uso:
    /home/carlos/venv/bin/python experiments/p100_10_construir_b1.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module

SAL_DIR = RAIZ / "data" / "external" / "corpora" / "p100_b1"
SAL_CORPUS = SAL_DIR / "tokens_30m.txt"
SAL_MANIFEST = RAIZ / "artifacts" / "p100_b1_manifest.json"

CHATSUBS_DIR = (RAIZ / "data" / "external" / "corpora" / "p100_chatsubs"
                / "extracted" / "open_subtitles_es")
CHATSUBS_TAR = (RAIZ / "data" / "external" / "corpora" / "p100_chatsubs"
                / "ChatSubs.tar.gz")
RUTA_C98_1 = (RAIZ / "data" / "external" / "corpora" / "p98_c98_1"
              / "tokens_30m.txt")
RUTA_S99_01 = (RAIZ / "data" / "external" / "corpora" / "p99_s99_01"
               / "tokens_60m.txt")

SEMILLA = 20260925
MD5_TAR_CHATSUBS = "afd2d3967cf8cdf78883a7baa8d061e4"
SHA256_C98_1 = ("9f562718ef4e7d45622040cd9ab5973182cc5405a99bfbc1636fa57"
                "abc6b9f0e")
SHA256_S99_01 = ("e627a7f33886c0a177849c2b9e76dc2bb9665c8476c000b3ed4f5"
                 "10f97b6e956")

OBJETIVOS = {"social": 4_227_450, "conversacional": 9_000_000,
             "wikipedia": 11_650_000, "mc4": 4_850_000}
TOTAL_OBJETIVO = 29_727_450
C98_1_ESPERADO = {"wikipedia": (5191, 18_009_963), "mc4": (12418, 7_499_770),
                  "twitter": (148_413, 4_227_450)}


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


def md5(ruta: Path) -> str:
    """md5 completo."""
    x = hashlib.md5()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()


def clave(texto: str) -> str:
    """sha1 del documento normalizado (misma regla P98/P99)."""
    return hashlib.sha1(m11.normalizar(texto).encode("utf-8")).hexdigest()


def n_tokens(texto: str) -> int:
    """Tokens por split (misma convencion P98/P99)."""
    return len(texto.split())


def replicar_c98_1() -> tuple:
    """Re-deriva la seleccion+dedup de C98-1 por fuente (determinista)."""
    rng = np.random.default_rng(SEMILLA)
    docs_por: dict[str, list[str]] = {}
    vistos: set[str] = set()
    for nombre, spec in m11.FUENTES.items():
        docs = m11.leer_documentos(spec["ruta"])
        elegidos, total = [], 0
        if spec["modo"] == "prefijo":
            for doc, n in docs:
                if total >= spec["objetivo"]:
                    break
                elegidos.append(doc)
                total += n
        else:
            for i in rng.permutation(len(docs)):
                doc, n = docs[i]
                if total >= spec["objetivo"]:
                    break
                elegidos.append(doc)
                total += n
        conservados = []
        for doc in elegidos:
            k = clave(doc)
            if k in vistos:
                continue
            vistos.add(k)
            conservados.append(doc)
        docs_por[nombre] = conservados
    return docs_por, vistos


def truncar_a_objetivo(docs: list, objetivo: int) -> tuple:
    """Recorta la lista a exactamente `objetivo` tokens (trunca el limite)."""
    finales: list[str] = []
    total = 0
    for doc in docs:
        n = n_tokens(doc)
        if total + n <= objetivo:
            finales.append(doc)
            total += n
            continue
        faltan = objetivo - total
        if faltan > 0:
            finales.append(" ".join(doc.split()[:faltan]))
            info = {"truncado": True, "doc_sha1_original": clave(doc),
                    "tokens_originales": n, "tokens_conservados": faltan}
        else:
            info = {"truncado": False, "doc_sha1_original": None,
                    "tokens_originales": None, "tokens_conservados": None}
        return finales, info
    info = {"truncado": False, "doc_sha1_original": None,
            "tokens_originales": None, "tokens_conservados": None}
    return finales, info


def construir_conversacional(vistos: set, objetivo: int) -> tuple:
    """Muestreo uniforme de archivos JSONL de ChatSubs es hasta el objetivo."""
    archivos = sorted(CHATSUBS_DIR.rglob("*.jsonl"))
    rng = np.random.default_rng(SEMILLA)
    orden = rng.permutation(len(archivos))
    docs: list[str] = []
    total = 0
    leidos = 0
    errores = 0
    duplicados = 0
    info_truncado = {"truncado": False, "doc_sha1_original": None,
                     "tokens_originales": None, "tokens_conservados": None}
    for idx in orden:
        if total >= objetivo:
            break
        ruta = archivos[int(idx)]
        try:
            d = json.loads(ruta.read_text(encoding="utf-8"))
            dial = d.get("dialogues", []) if isinstance(d, dict) else []
            texto = " ".join(str(x).replace("\n", " ") for x in dial)
            texto = " ".join(texto.split())
        except (OSError, ValueError):
            errores += 1
            continue
        if not texto:
            errores += 1
            continue
        leidos += 1
        k = clave(texto)
        if k in vistos:
            duplicados += 1
            continue
        n = n_tokens(texto)
        if total + n <= objetivo:
            docs.append(texto)
            total += n
            vistos.add(k)
            continue
        faltan = objetivo - total
        if faltan > 0:
            docs.append(" ".join(texto.split()[:faltan]))
            info_truncado = {"truncado": True, "doc_sha1_original": k,
                             "tokens_originales": n,
                             "tokens_conservados": faltan}
        break
    stats = {"archivos_totales": len(archivos), "archivos_leidos": leidos,
             "errores_lectura": errores, "duplicados_removidos": duplicados,
             "truncamiento": info_truncado}
    return docs, total, stats


def main() -> None:
    """Construye B1 completo con controles."""
    for ruta in (SAL_CORPUS, SAL_MANIFEST):
        if ruta.exists():
            raise SystemExit(f"Ya existe {ruta}; no se sobrescribe.")
    inicio = time.perf_counter()

    for nombre, spec in m11.FUENTES.items():
        real = sha256(spec["ruta"])
        if real != spec["sha256_esperado"]:
            raise SystemExit(f"hash inesperado en fuente {nombre}")
    if md5(CHATSUBS_TAR) != MD5_TAR_CHATSUBS:
        raise SystemExit("md5 inesperado en ChatSubs.tar.gz")
    if sha256(RUTA_C98_1) != SHA256_C98_1:
        raise SystemExit("sha256 inesperado en C98-1")
    print("hashes de las 4 fuentes verificados", flush=True)

    ref_docs, ref_claves = replicar_c98_1()
    for nombre, (n_docs, n_tok) in C98_1_ESPERADO.items():
        real_docs = len(ref_docs[nombre])
        real_tok = sum(n_tokens(d) for d in ref_docs[nombre])
        if (real_docs, real_tok) != (n_docs, n_tok):
            raise SystemExit(f"replicacion C98-1 divergente en {nombre}: "
                             f"{real_docs}/{real_tok} != {n_docs}/{n_tok}")
    print(f"C98-1 replicado exacto: {len(ref_claves)} docs | "
          f"{sum(n_tokens(d) for v in ref_docs.values() for d in v):,} tokens",
          flush=True)

    claves_disco = set()
    with open(RUTA_C98_1, encoding="utf-8") as fh:
        for linea in fh:
            doc = linea.rstrip("\n")
            if doc:
                claves_disco.add(clave(doc))
    if claves_disco != ref_claves:
        raise SystemExit("las claves de C98-1 en disco NO coinciden con la "
                         "replicacion; detener")
    print("replicacion verificada contra el corpus C98-1 en disco "
          "(claves identicas)", flush=True)

    social = ref_docs["twitter"]
    if sum(n_tokens(d) for d in social) != OBJETIVOS["social"]:
        raise SystemExit("social no coincide con el objetivo exacto")
    wiki, info_w = truncar_a_objetivo(ref_docs["wikipedia"],
                                      OBJETIVOS["wikipedia"])
    mc4, info_m = truncar_a_objetivo(ref_docs["mc4"], OBJETIVOS["mc4"])
    conv, _tok_conv, stats_conv = construir_conversacional(set(ref_claves),
                                                          OBJETIVOS[
                                                              "conversacional"])
    for nombre, lista, objetivo in (("wikipedia", wiki, OBJETIVOS["wikipedia"]),
                                    ("mc4", mc4, OBJETIVOS["mc4"]),
                                    ("conversacional", conv,
                                     OBJETIVOS["conversacional"])):
        real = sum(n_tokens(d) for d in lista)
        if real != objetivo:
            raise SystemExit(f"{nombre}: {real} != objetivo {objetivo}")
        print(f"  {nombre}: {len(lista)} docs | {real:,} tokens (exacto)",
              flush=True)

    finales = ([(d, "wikipedia") for d in wiki]
               + [(d, "mc4") for d in mc4]
               + [(d, "social") for d in social]
               + [(d, "conversacional") for d in conv])
    rng_final = np.random.default_rng(SEMILLA)
    orden = rng_final.permutation(len(finales))
    finales = [finales[i] for i in orden]

    SAL_DIR.mkdir(parents=True, exist_ok=True)
    tokens_por_fuente: dict[str, int] = {}
    docs_por_fuente: dict[str, int] = {}
    total_tokens = 0
    with open(SAL_CORPUS, "w", encoding="utf-8") as fh:
        for doc, origen in finales:
            fh.write(doc + "\n")
            n = n_tokens(doc)
            total_tokens += n
            tokens_por_fuente[origen] = tokens_por_fuente.get(origen, 0) + n
            docs_por_fuente[origen] = docs_por_fuente.get(origen, 0) + 1
    if total_tokens != TOTAL_OBJETIVO:
        raise SystemExit(f"total {total_tokens} != {TOTAL_OBJETIVO}")
    print(f"corpus B1 escrito: {total_tokens:,} tokens | {len(finales):,} docs",
          flush=True)

    b1_claves = {clave(d) for d, _ in finales}
    mapa_original = {clave(d): d for d in ref_docs["wikipedia"]
                     + ref_docs["mc4"]}
    fallos_subset = 0
    prefijos_verificados = 0
    for info in (info_w, info_m):
        if not info["truncado"]:
            continue
        original = mapa_original[info["doc_sha1_original"]]
        prefijo = " ".join(original.split()[:info["tokens_conservados"]])
        if clave(prefijo) in b1_claves:
            prefijos_verificados += 1
        else:
            fallos_subset += 1
    faltantes = [d for d, o in finales
                 if o in ("wikipedia", "mc4", "social")
                 and clave(d) not in ref_claves]
    esperados_limite = sum(1 for info in (info_w, info_m)
                           if info["truncado"])
    if len(faltantes) != esperados_limite or fallos_subset:
        raise SystemExit("verificacion de subconjunto fallida: "
                         f"{len(faltantes)} faltantes, {fallos_subset} fallos")

    claves_val = m12.val_set(m11, m12.hashes_por_fuente(m11))
    val_por_componente = {}
    for origen in ("wikipedia", "mc4", "social", "conversacional"):
        claves_comp = {clave(d) for d, o in finales if o == origen}
        val_por_componente[origen] = len(claves_comp & claves_val)
    val_presentes = len(b1_claves & claves_val)

    train = data_module.read_train()
    claves_train = {hashlib.sha1(m11.normalizar(t).encode("utf-8")).hexdigest()
                    for t in train["text"]}
    coincidencias_train = len(b1_claves & claves_train)

    if sha256(RUTA_S99_01) != SHA256_S99_01:
        raise SystemExit("sha256 inesperado en S99-01")
    claves_s99 = set()
    with open(RUTA_S99_01, encoding="utf-8") as fh:
        for linea in fh:
            doc = linea.rstrip("\n")
            if doc:
                claves_s99.add(clave(doc))
    claves_conv = {clave(d) for d in conv}
    overlap_conv_s99 = len(claves_conv & claves_s99)

    manifiesto = {
        "experiment_id": "P100-10",
        "fase": "P100-B1 (construccion)",
        "fecha": "2026-10-13",
        "seed": SEMILLA,
        "objetivo_total": TOTAL_OBJETIVO,
        "tokens_finales": total_tokens,
        "documentos_finales": len(finales),
        "objetivos_nominales": OBJETIVOS,
        "tokens_por_fuente": tokens_por_fuente,
        "documentos_por_fuente": docs_por_fuente,
        "exactitud": "objetivos exactos por componente; documento limite "
                     "truncado a nivel token; sin relleno por duplicacion",
        "truncamientos": {"wikipedia": info_w, "mc4": info_m,
                          "conversacional": stats_conv["truncamiento"]},
        "conversacional_stats": stats_conv,
        "referencia_c98_1": {
            "documentos": len(ref_claves),
            "tokens": sum(n_tokens(d) for v in ref_docs.values() for d in v),
            "sha256": SHA256_C98_1,
            "replicacion_claves_identica": True,
            "por_fuente": {k: {"documentos": len(v),
                               "tokens": sum(n_tokens(d) for d in v)}
                           for k, v in ref_docs.items()}},
        "subset_verificacion": {
            "docs_social_general": len(wiki) + len(mc4) + len(social),
            "en_c98_1_exactos": len(wiki) + len(mc4) + len(social)
                                - esperados_limite,
            "prefijos_limite": esperados_limite,
            "prefijos_limite_verificados": prefijos_verificados,
            "fallos": fallos_subset},
        "val_language_docs": len(claves_val),
        "val_presentes_en_b1": val_presentes,
        "val_por_componente": val_por_componente,
        "val_exclusion": "p100_11 excluye las claves val del stream de "
                         "entrenamiento y evalua el medidor fijo de 3.320 "
                         "documentos materializados (regla p98_12)",
        "coincidencias_exactas_vs_train": coincidencias_train,
        "overlap_conversacional_vs_s99_01": overlap_conv_s99,
        "fuentes": {
            "wikipedia": "wikipedia_es_50M (CC BY-SA 4.0)",
            "mc4": "mc4_es (ODC-BY 1.0)",
            "social": "twitter_domain COVID-es (CC BY-SA 4.0; autorizacion "
                      "p73/P98)",
            "conversacional": "ChatSubs es (CC BY-NC 4.0; condiciones del "
                              "propietario 2026-10-13)"},
        "sha256": sha256(SAL_CORPUS),
        "ruta": str(SAL_CORPUS.relative_to(RAIZ)),
        "criterios": "pre-registrado en reports/p100_preregistration.md; "
                     "objetivos exactos autorizados 2026-10-13; dedup exacta; "
                     "sin toxic_test; sin descargas nuevas",
        "tiempo_s": round(time.perf_counter() - inicio, 1),
    }
    SAL_MANIFEST.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=True)
                            + "\n", encoding="utf-8")
    print(f"B1: {total_tokens:,} tokens | {len(finales):,} docs | val "
          f"presentes {val_presentes}/{len(claves_val)} | overlap train "
          f"{coincidencias_train} | conv vs S99-01 {overlap_conv_s99}",
          flush=True)
    print(f"sha256 {manifiesto['sha256'][:16]} | tiempo "
          f"{manifiesto['tiempo_s']} s", flush=True)


if __name__ == "__main__":
    main()
