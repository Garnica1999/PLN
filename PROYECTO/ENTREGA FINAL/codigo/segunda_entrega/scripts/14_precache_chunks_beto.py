"""14_precache_chunks_beto.py

Fecha: 2026-09-11
Descripcion: pre-tokeniza UNA vez a disco los chunks (256 tokens, stride
128, max 64) de las columnas texto y texto_mask de los 23.654 docs M4
(definidos por data/splits_m4.parquet). Shards .pkl de ~500 docs por
columna con reanudacion, Pool spawn sin torch/CUDA y auto-verificacion.

ciclo v2 - Fase V (cache de chunks). Contexto del incidente 2026-09-11:
los 4 runs completos de 07_transformer.py (chunk-mean BETO) se colgaban
SIEMPRE en el DataLoader multiproceso (main en dataloader._try_get_data
esperando lote, 8 workers idle esperando indices; Python 3.14 + fork;
evidencia en
experimentos/m4_transformer_chunks_stdout_faulthandler_evidencia.log).
Este script saca la tokenizacion de los workers de entrenamiento:
pre-tokeniza a la cache y 07 entrena con num_workers=0 (sin procesos
worker, sin deadlock posible) y arranques mas rapidos.

Estado del arte aplicado: chunk-mean de Chalkidis et al. 2022
(arXiv:2211.00974); los parametros de troceo viven en
scripts/lib/chunks_beto.py (fuente unica compartida con 07).

Uso:
    /home/carlos/venv/bin/python scripts/14_precache_chunks_beto.py
    /home/carlos/venv/bin/python scripts/14_precache_chunks_beto.py --limite 400
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import pickle
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from lib.chunks_beto import (
    CHUNK_STRIDE,
    CHUNK_TOKENS,
    MAX_CHUNKS,
    MODELO_BETO,
    enmascarar_texto_crudo,
    hash_config_chunking,
    sha256_ruta,
    trocear_doc,
)
from transformers import AutoTokenizer

# ===== SECCION 1: configuracion =====

CONFIG = {
    "semilla_muestra": 42,
    "semilla_verificacion": 7,
    "docs_verificacion": 5,
    "procesos": 12,
    "docs_por_tarea": 16,  # docs por tarea IPC (menos overhead de pickling)
    "docs_por_shard": 500,  # ~48 shards por columna (23.654 docs)
    "dir_cache": "data/cache_chunks_beto",
    "dir_smoke": "/tmp/opencode/cache_chunks_smoke",
    "ruta_splits": "data/splits_m4.parquet",
    "ruta_corpus": "data/corpus_sentencias.parquet",
    "ruta_log": "experimentos/14_precache_chunks.log",
    "ruta_log_smoke": "/tmp/opencode/14_precache_chunks_smoke.log",
    "espacio_min_gb": 3.0,
}

LOGGER = logging.getLogger("precache_chunks")

# Tokenizer por proceso worker (inicializado UNA vez en _init_worker).
TOKENIZER = None


# ===== SECCION 2: utilidades de directorio, muestra y shards =====


def configurar_logging(limite: int | None) -> logging.Logger:
    """Configura el logger del pre-cache.

    Args:
        limite: Limite de docs del modo smoke (define el log a /tmp).

    Returns:
        Logger configurado.
    """
    ruta = CONFIG["ruta_log_smoke"] if limite is not None else CONFIG["ruta_log"]
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(ruta, mode="w"),
            logging.StreamHandler(),
        ],
    )
    return LOGGER


def resolver_salida(args: argparse.Namespace) -> str:
    """Decide el directorio de cache segun args.

    Args:
        args: Namespace de parsear_args().

    Returns:
        Ruta de salida (smoke -> /tmp/opencode/cache_chunks_smoke).
    """
    if args.dir_salida is not None:
        return args.dir_salida
    return CONFIG["dir_smoke"] if args.limite is not None else CONFIG["dir_cache"]


def verificar_espacio(dir_salida: str, minimo_gb: float) -> None:
    """Verifica espacio libre suficiente antes de escribir la cache.

    Args:
        dir_salida: Directorio destino de la cache.
        minimo_gb: GB libres minimos exigidos.

    Raises:
        RuntimeError: si el filesystem destino tiene menos espacio.
    """
    ruta = Path(dir_salida)
    while not ruta.exists() and ruta != ruta.parent:
        ruta = ruta.parent
    libre_gb = shutil.disk_usage(ruta).free / 1e9
    if libre_gb < minimo_gb:
        raise RuntimeError(f"espacio insuficiente en {ruta}: {libre_gb:.1f} GB libres < {minimo_gb} GB")
    LOGGER.info("espacio libre en %s: %.1f GB (minimo %.1f GB)", ruta, libre_gb, minimo_gb)


def cargar_documentos(limite: int | None) -> pd.DataFrame:
    """Carga los 23.654 docs de splits con su texto crudo.

    Args:
        limite: Si no es None, toma una muestra estratificada determinista
            (seed 42) por particion y tipo para el smoke.

    Returns:
        DataFrame con id, particion, tipo y texto, ordenado por id.
    """
    splits = pd.read_parquet(CONFIG["ruta_splits"])
    ids = {int(i) for i in splits["id"]}
    corpus = pd.read_parquet(CONFIG["ruta_corpus"], columns=["id", "texto", "tipo"])
    corpus = corpus[corpus["id"].astype(np.int64).isin(ids)]
    df = splits.merge(corpus, on="id", how="inner").sort_values("id").reset_index(drop=True)
    if limite is not None:
        df = muestra_estratificada(df, limite, CONFIG["semilla_muestra"])
        LOGGER.info(
            "[smoke] muestra estratificada seed=%d: %d docs %s",
            CONFIG["semilla_muestra"],
            len(df),
            df["particion"].value_counts().to_dict(),
        )
    LOGGER.info("documentos objetivo: %d (columnas: texto, texto_mask)", len(df))
    return df


def muestra_estratificada(df: pd.DataFrame, limite: int, semilla: int) -> pd.DataFrame:
    """Muestra estratificada determinista por particion y tipo.

    Args:
        df: Documentos completos (id, particion, tipo, texto).
        limite: Numero objetivo de documentos.
        semilla: Semilla del muestreo.

    Returns:
        Subconjunto con <=limite docs y al menos uno por particion.
    """
    if limite >= len(df):
        return df.sort_values("id").reset_index(drop=True)
    frac = limite / len(df)
    sub = df.groupby(["particion", "tipo"], group_keys=False, observed=True).sample(frac=frac, random_state=semilla)
    for particion in sorted(df["particion"].unique()):
        if not (sub["particion"] == particion).any():
            extra = df[df["particion"] == particion].sample(n=1, random_state=semilla)
            sub = pd.concat([sub, extra], ignore_index=True)
    if len(sub) > limite:
        sub = sub.sample(n=limite, random_state=semilla)
    return sub.drop_duplicates(subset="id").sort_values("id").reset_index(drop=True)


def hash_config_actual() -> str:
    """Huella de la configuracion de chunking vigente (compartida con 07)."""
    return hash_config_chunking(MODELO_BETO, CHUNK_TOKENS, CHUNK_STRIDE, MAX_CHUNKS)


def comprobar_meta(dir_salida: str, ids: list[int], limite: int | None) -> None:
    """Valida que una cache existente corresponda a la configuracion actual.

    Args:
        dir_salida: Directorio de cache.
        ids: Ids objetivo (tamano esperado).
        limite: Limite del modo smoke (None en full).

    Raises:
        RuntimeError: si la cache existente fue generada con otra config.
    """
    ruta = Path(dir_salida) / "meta.json"
    if not ruta.exists():
        return
    meta = json.loads(ruta.read_text(encoding="utf-8"))
    if meta.get("config_hash") != hash_config_actual() or meta.get("n_docs") != len(ids) or meta.get("limite") != limite:
        raise RuntimeError(
            f"cache existente en {dir_salida} con configuracion distinta "
            f"(config_hash={meta.get('config_hash')} n_docs={meta.get('n_docs')} "
            f"limite={meta.get('limite')}); borra el directorio y reintenta"
        )
    LOGGER.info("cache existente compatible: reanudacion habilitada (%s)", dir_salida)


def leer_shard_valido(ruta: Path, esperados: list[int]) -> list | None:
    """Lee un shard y valida que contenga exactamente los ids esperados.

    Args:
        ruta: Ruta del shard .pkl.
        esperados: Ids que debe contener el shard.

    Returns:
        Lista [(id, chunks)] si el shard es valido; None si falta o no sirve.
    """
    if not ruta.exists():
        return None
    try:
        with open(ruta, "rb") as manejador:
            items = pickle.load(manejador)
    except Exception as exc:  # noqa: BLE001 - shard corrupto: se regenera sin abortar
        LOGGER.warning("shard %s ilegible (%s): se regenera", ruta, exc)
        return None
    if not isinstance(items, list) or any(not (isinstance(it, tuple) and len(it) == 2) for it in items):
        LOGGER.warning("shard %s con estructura invalida: se regenera", ruta)
        return None
    ids_leidos = sorted(int(it[0]) for it in items)
    if ids_leidos != sorted(int(i) for i in esperados):
        LOGGER.warning("shard %s con ids inesperados (%d vs %d): se regenera", ruta, len(ids_leidos), len(esperados))
        return None
    return items


def escribir_shard(ruta: Path, items: list[tuple[int, list[np.ndarray]]]) -> None:
    """Escribe un shard de forma atomica (tmp + os.replace).

    Args:
        ruta: Ruta destino del shard.
        items: Lista (id_doc, chunks uint16).
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".pkl.tmp")
    with open(temporal, "wb") as manejador:
        pickle.dump(items, manejador, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporal, ruta)


def escribir_meta(
    dir_salida: str,
    ids: list[int],
    hechos: dict[str, set[int]],
    args: argparse.Namespace,
    corpus_sha: str,
    tiempo_seg: float,
    docs_por_shard: int,
) -> None:
    """Escribe meta.json con la trazabilidad completa de la cache.

    Args:
        dir_salida: Directorio de cache.
        ids: Ids objetivo.
        hechos: Ids completados por columna.
        args: Namespace de parsear_args().
        corpus_sha: sha256 del corpus de origen.
        tiempo_seg: Tiempo transcurrido.
        docs_por_shard: Tamano de shard usado.
    """
    n_shards = (len(ids) + docs_por_shard - 1) // docs_por_shard
    meta = {
        "fecha": pd.Timestamp.now().isoformat(),
        "tokenizer": MODELO_BETO,
        "chunking": {
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_stride": CHUNK_STRIDE,
            "max_chunks": MAX_CHUNKS,
        },
        "config_hash": hash_config_actual(),
        "n_docs": len(ids),
        "limite": args.limite,
        "semilla_muestra": CONFIG["semilla_muestra"],
        "corpus": CONFIG["ruta_corpus"],
        "corpus_sha256": corpus_sha,
        "splits": CONFIG["ruta_splits"],
        "docs_por_shard": docs_por_shard,
        "tiempo_seg": tiempo_seg,
        "columnas": {},
    }
    for columna, hechos_col in hechos.items():
        meta["columnas"][columna] = {
            "n_docs": len(hechos_col),
            "n_shards_objetivo": n_shards,
            "shards": [f"shard_{i:04d}.pkl" for i in range(n_shards)],
        }
    ruta = Path(dir_salida) / "meta.json"
    temporal = ruta.with_suffix(".json.tmp")
    temporal.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporal, ruta)
    LOGGER.info("meta escrito: %s", ruta)


def verificar_cobertura(ids: list[int], hechos: dict[str, set[int]], dir_salida: str) -> None:
    """Verifica cobertura completa de ids en todas las columnas.

    Args:
        ids: Ids objetivo.
        hechos: Ids completados por columna.
        dir_salida: Directorio de cache (para el mensaje de error).

    Raises:
        RuntimeError: si falta algun doc en alguna columna.
    """
    faltantes = {columna: sorted(set(ids) - hechos_col) for columna, hechos_col in hechos.items()}
    total = sum(len(v) for v in faltantes.values())
    if total:
        detalle = "; ".join(f"{c}: {len(v)} faltantes (p. ej. {v[:5]})" for c, v in faltantes.items())
        raise RuntimeError(f"cobertura incompleta en {dir_salida}: {detalle}")
    LOGGER.info("cobertura OK: %d ids x %d columnas", len(ids), len(hechos))


# ===== SECCION 3: workers (Pool spawn, sin torch/CUDA) =====


def _init_worker(modelo: str) -> None:
    """Inicializa el tokenizer UNA vez por proceso worker.

    Args:
        modelo: Nombre del tokenizer en la cache HF local.
    """
    global TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    TOKENIZER = AutoTokenizer.from_pretrained(modelo)


def _tokenizar_chunks(texto: str) -> list[np.ndarray]:
    """Tokeniza el doc completo y lo trocea (identico a 07).

    Args:
        texto: Texto crudo o enmascarado del documento.

    Returns:
        Lista de chunks como np.ndarray uint16.

    Raises:
        RuntimeError: si el worker no inicializo el tokenizer.
        ValueError: si un token no cabe en uint16 (proteccion de cache).
    """
    if TOKENIZER is None:
        raise RuntimeError("worker sin tokenizer inicializado")
    enc = TOKENIZER(texto, truncation=False)
    input_ids = enc["input_ids"]
    if input_ids and max(input_ids) > np.iinfo(np.uint16).max:
        raise ValueError(f"token fuera de uint16: max={max(input_ids)}")
    chunks = trocear_doc(input_ids, CHUNK_TOKENS, CHUNK_STRIDE, MAX_CHUNKS)
    return [np.asarray(ch, dtype=np.uint16) for ch in chunks]


def _procesar_tarea(tarea: list[tuple[int, int, str]]) -> list[tuple]:
    """Procesa un lote de docs; reintenta una vez por doc ante fallos.

    Args:
        tarea: Lista de (posicion_global, id_doc, texto).

    Returns:
        Lista de (posicion_global, id_doc, chunks|None, error|None).
    """
    salida = []
    for posicion, id_doc, texto in tarea:
        chunks, error = None, None
        for _ in range(2):  # reintento unico por doc
            try:
                chunks = _tokenizar_chunks(texto)
                error = None
                break
            except Exception as exc:  # noqa: BLE001 - error por doc: se registra, reintenta y omite sin abortar el lote
                error = f"{type(exc).__name__}: {exc}"
        salida.append((posicion, id_doc, chunks, error))
    return salida


def _generar_tareas(
    ids_bloque: list[int],
    textos_map: dict[int, str],
    columna: str,
    pos_base: int,
    docs_por_tarea: int,
):
    """Genera tareas por lote (generador: solapa enmascarado con el Pool).

    Args:
        ids_bloque: Ids de un shard (<= docs_por_shard).
        textos_map: id -> texto crudo.
        columna: "texto" o "texto_mask".
        pos_base: Posicion global del primer id del bloque.
        docs_por_tarea: Docs por tarea IPC.

    Yields:
        Lista de (posicion_global, id_doc, texto_de_la_columna).
    """
    for ini in range(0, len(ids_bloque), docs_por_tarea):
        tarea = []
        for j, id_doc in enumerate(ids_bloque[ini : ini + docs_por_tarea]):
            texto = textos_map[int(id_doc)]
            if columna == "texto_mask":
                texto = enmascarar_texto_crudo(texto)
            tarea.append((pos_base + ini + j, int(id_doc), texto))
        yield tarea


# ===== SECCION 4: procesamiento de una columna por shards =====


def procesar_columna(
    pool: mp.pool.Pool,
    ids: list[int],
    textos_map: dict[int, str],
    columna: str,
    dir_salida: str,
    docs_por_shard: int,
    docs_por_tarea: int,
) -> tuple[set[int], list[tuple]]:
    """Pre-tokeniza una columna completa por shards reanudables.

    Args:
        pool: Pool spawn con workers inicializados.
        ids: Ids objetivo ordenados.
        textos_map: id -> texto crudo.
        columna: "texto" o "texto_mask".
        dir_salida: Directorio raiz de cache.
        docs_por_shard: Docs por shard.
        docs_por_tarea: Docs por tarea IPC.

    Returns:
        Tupla (ids completados, lista de fallos (posicion, id, None, error)).
    """
    dir_col = Path(dir_salida) / columna
    dir_col.mkdir(parents=True, exist_ok=True)
    n_shards = (len(ids) + docs_por_shard - 1) // docs_por_shard
    hechos: set[int] = set()
    errores: list[tuple] = []
    t0 = time.time()
    for shard in range(n_shards):
        esperados = ids[shard * docs_por_shard : (shard + 1) * docs_por_shard]
        ruta = dir_col / f"shard_{shard:04d}.pkl"
        items = leer_shard_valido(ruta, esperados)
        if items is not None:
            hechos.update(int(it[0]) for it in items)
            LOGGER.info(
                "[%s] shard %04d/%04d existente y valido (%d docs): reanudacion, se omite",
                columna,
                shard + 1,
                n_shards,
                len(items),
            )
            continue
        tareas = _generar_tareas(esperados, textos_map, columna, shard * docs_por_shard, docs_por_tarea)
        resultados: list[tuple] = []
        for parcial in pool.imap_unordered(_procesar_tarea, tareas):
            resultados.extend(parcial)
        resultados.sort(key=lambda resultado: resultado[0])
        fallidos = [resultado for resultado in resultados if resultado[2] is None]
        if fallidos:
            for _, id_doc, _, error in fallidos:
                LOGGER.error("[%s] doc %s fallo tras reintento: %s", columna, id_doc, error)
            errores.extend(fallidos)
            LOGGER.warning(
                "[%s] shard %04d NO se escribe (%d docs fallidos): se reintentara en la proxima ejecucion",
                columna,
                shard + 1,
                len(fallidos),
            )
            continue
        escribir_shard(ruta, [(int(id_doc), chunks) for _, id_doc, chunks, _ in resultados])
        hechos.update(int(id_doc) for _, id_doc, _, _ in resultados)
        transcurrido = max(time.time() - t0, 1e-9)
        velocidad = len(hechos) / transcurrido
        eta_min = (len(ids) - len(hechos)) / max(velocidad, 1e-9) / 60
        LOGGER.info(
            "[%s] shard %04d/%04d docs=%d/%d (%.1f%%) | %.2f docs/s | ETA %.1f min",
            columna,
            shard + 1,
            n_shards,
            len(hechos),
            len(ids),
            100 * len(hechos) / len(ids),
            velocidad,
            eta_min,
        )
    return hechos, errores


# ===== SECCION 5: auto-verificacion smoke =====


def verificar_cache_smoke(
    dir_salida: str,
    ids: list[int],
    textos_map: dict[int, str],
    docs_por_shard: int,
) -> None:
    """Auto-verifica 5 docs (seed 7) contra tokenizacion directa en proceso.

    Compara EXACTAMENTE la tokenizacion de 07
    (tokenizer(texto, truncation=False) + trocear_doc) contra los chunks
    persistidos, para ambas columnas. Falla ruidosamente si difieren.

    Args:
        dir_salida: Directorio de cache smoke.
        ids: Ids objetivo ordenados.
        textos_map: id -> texto crudo.
        docs_por_shard: Docs por shard (para localizar el shard del doc).

    Raises:
        RuntimeError: si algun doc verificado no coincide con la cache.
    """
    rng = random.Random(CONFIG["semilla_verificacion"])
    muestra = rng.sample(sorted(ids), min(CONFIG["docs_verificacion"], len(ids)))
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BETO)
    posiciones = {id_doc: i for i, id_doc in enumerate(ids)}
    LOGGER.info("[auto-verificacion] docs seed=%d: %s", CONFIG["semilla_verificacion"], muestra)
    for columna in ("texto", "texto_mask"):
        shards_leidos: dict[int, dict[int, list[np.ndarray]]] = {}
        for id_doc in muestra:
            indice = posiciones[id_doc] // docs_por_shard
            if indice not in shards_leidos:
                with open(Path(dir_salida) / columna / f"shard_{indice:04d}.pkl", "rb") as manejador:
                    shards_leidos[indice] = {int(it[0]): it[1] for it in pickle.load(manejador)}
        for id_doc in muestra:
            texto = textos_map[id_doc]
            if columna == "texto_mask":
                texto = enmascarar_texto_crudo(texto)
            esperado = trocear_doc(tokenizer(texto, truncation=False)["input_ids"], CHUNK_TOKENS, CHUNK_STRIDE, MAX_CHUNKS)
            cache = shards_leidos[posiciones[id_doc] // docs_por_shard][id_doc]
            coincide = len(cache) == len(esperado) and all(
                cache[i].dtype == np.uint16 and np.array_equal(cache[i], np.asarray(esperado[i], dtype=np.uint16))
                for i in range(len(cache))
            )
            if not coincide:
                raise RuntimeError(
                    f"[auto-verificacion] doc {id_doc} columna {columna}: los chunks cacheados "
                    f"({len(cache)}) no coinciden con tokenizer(texto, truncation=False) + trocear_doc ({len(esperado)})"
                )
    LOGGER.info(
        "[auto-verificacion] OK: %d docs x 2 columnas; cache == tokenizer(texto, truncation=False) + trocear_doc",
        len(muestra),
    )


# ===== SECCION 6: entrada principal =====


def parsear_args() -> argparse.Namespace:
    """Parsea los argumentos de linea de comandos.

    Returns:
        Namespace con limite, dir_salida, procesos y docs_por_shard.
    """
    parser = argparse.ArgumentParser(description="Pre-tokeniza chunks BETO a disco (ciclo v2 Fase V)")
    parser.add_argument("--limite", type=int, default=None, help="Smoke: muestra estratificada determinista de N docs")
    parser.add_argument("--dir-salida", type=str, default=None,
                        help="Directorio de cache (default: data/cache_chunks_beto; smoke: /tmp/opencode/cache_chunks_smoke)")
    parser.add_argument("--procesos", type=int, default=CONFIG["procesos"], help="Procesos del Pool spawn (default 12)")
    parser.add_argument("--docs-por-shard", type=int, default=CONFIG["docs_por_shard"], help="Docs por shard (default 500)")
    return parser.parse_args()


def main() -> None:
    """Ejecuta el pre-cache completo (o smoke) con reanudacion y verificacion."""
    args = parsear_args()
    dir_salida = resolver_salida(args)
    configurar_logging(args.limite)
    random.seed(CONFIG["semilla_muestra"])
    np.random.seed(CONFIG["semilla_muestra"])
    t0 = time.time()
    LOGGER.info(
        "inicio pre-cache chunks BETO: limite=%s dir_salida=%s procesos=%d docs_por_tarea=%d",
        args.limite, dir_salida, args.procesos, CONFIG["docs_por_tarea"],
    )
    verificar_espacio(dir_salida, CONFIG["espacio_min_gb"])
    df = cargar_documentos(args.limite)
    ids = sorted(int(i) for i in df["id"])
    comprobar_meta(dir_salida, ids, args.limite)
    textos_map = {int(i): t for i, t in zip(df["id"], df["texto"])}
    corpus_sha = sha256_ruta(CONFIG["ruta_corpus"])
    LOGGER.info("corpus %s sha256=%s... (%d docs objetivo)", CONFIG["ruta_corpus"], corpus_sha[:16], len(ids))

    hechos: dict[str, set[int]] = {}
    errores: list[tuple] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.procesos, initializer=_init_worker, initargs=(MODELO_BETO,)) as pool:
        for columna in ("texto", "texto_mask"):
            completados, fallos = procesar_columna(
                pool, ids, textos_map, columna, dir_salida, args.docs_por_shard, CONFIG["docs_por_tarea"]
            )
            hechos[columna] = completados
            errores.extend(fallos)

    escribir_meta(dir_salida, ids, hechos, args, corpus_sha, time.time() - t0, args.docs_por_shard)
    if errores:
        raise RuntimeError(f"{len(errores)} docs fallidos tras reintento; reejecuta el script (reanuda shards validos)")
    verificar_cobertura(ids, hechos, dir_salida)
    if args.limite is not None:
        verificar_cache_smoke(dir_salida, ids, textos_map, args.docs_por_shard)
    LOGGER.info("fin: %d docs x 2 columnas en %.1f min; cache en %s", len(ids), (time.time() - t0) / 60, dir_salida)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - protocolo: traceback completo y exit(1)
        traceback.print_exc()
        sys.exit(1)
