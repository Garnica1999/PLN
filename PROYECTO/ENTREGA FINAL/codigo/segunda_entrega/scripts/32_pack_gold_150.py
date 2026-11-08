"""32_pack_gold_150.py
Fecha: 2026-10-05
Descripcion: Genera el pack de anotacion C1 (gold estandar de 150 casos) del
ciclo v3: 40 casos restantes del benchmark M7 (los 90 de
resultados/vinculo_cp_benchmark_v2.json menos los 50 ya anotados en el
mini-gold B3) + 110 casos del corpus C3 2022-2026 (muestreo con semilla 42,
~22 por anio), con candidatos = union de los articulos de la CP citados por
NER y el top-10 de la ruta del producto v2.2 por caso, snippets del catalogo
de la CP, guia de anotacion y estadisticas del pack. NO anota: prepara el
pack para que scripts/16_anotador_gold.py (--pack gold_150) lo recorra,
calcule acuerdos y exporte.

Diseno:
    - Origen benchmark (40 casos, 1992-2021): ids de vinculo_cp_benchmark_v2
      no presentes en resultados/gold_mini/muestra_gold.jsonl. Candidatos =
      arts_cp NER (data/entidades_sentencias.parquet, con respaldo en
      arts_cp del benchmark) + top-10 de la ruta
      hibrido_b2_rerank_chunks_top50_b1 de vinculo_cp_benchmark_v13.json
      (ruta del producto v2.2). Anti-leakage: los pares de entrenamiento de
      B1/B2 (data/pares_b1, scripts/23 y 25) excluyen los 90 ids del
      benchmark y se construyen sobre el corpus 1992-2021; estos 40 casos
      son ids NUEVOS para anotacion (no repiten el mini-gold B3).
    - Origen C3 (110 casos, 2022-2026): universo = filas de
      data/corpus_2022_2026/corpus_2022_2026.parquet con estado "ok" e id
      presente en entidades_2022_2026.parquet (2330 casos); cuotas por anio
      ~22 (cuota equitativa por anio) y semilla 42. Candidatos = arts_cp NER del caso +
      top-10 de VinculoCP(usar_reranker=True).recuperar(texto_limpio, k=10)
      (hibrido BM25+B2, rerank B1, prior de hubs alpha=1.5). Anti-leakage:
      los pares de B1/B2 provienen del corpus 1992-2021, de modo que el
      corpus C3 no entra en su entrenamiento. Los rankings C3 con scores se
      guardan en rutas_c3_gold150.json (evaluacion futura) y se cachean por
      id en cache_rutas_c3.jsonl para reanudar sin repetir la GPU.
    - Muestra por caso (mismo formato del pack B3 mas "origen"): indice,
      id, tipo, ano, periodo, tercil_dificultad, n_citas_ner, extracto
      (~1500 chars) y origen ("benchmark" | "c3"). Para C3 el periodo es
      "2022-2026" y el extracto es el texto limpiado E1/E2 del parquet
      (minusculas, sin digitos ni puntuacion; documentado en la guia).
    - tercil_dificultad: terciles de n_citas_ner del pack (0 = menos citas,
      mas dificil). Proxy documentado: los 40 casos benchmark tienen
      nDCG@10 por caso en v13, pero los 110 de C3 no tienen ninguna ruta
      evaluada y n_citas_ner es la unica senal comparable; se avisa en la
      guia y en pack_stats.json.
    - Candidatos por caso (mismo formato y logica del pack B3): union de
      NER y top-10, ordenados por numero de articulo (22A entre 22 y 23;
      Preambulo y Transitorio al final), snippet de ~200 chars del catalogo
      data/constitucion_articulos.parquet y flag citado_ner. Se reutilizan
      las funciones del script 16 por importlib (modulo_16).
    - Salidas en resultados/gold_150/ (o --dir-salida): muestra_gold.jsonl,
      candidatos.jsonl, guia_anotacion_gold.md, rutas_c3_gold150.json,
      pack_stats.json y el cache reanudable cache_rutas_c3.jsonl.
    - Guardas: sin --forzar no se reescribe ninguna salida existente; la
      cache por id se reutiliza siempre (es el mecanismo de reanudacion).
      --limite N (pruebas) toma ceil(N/2) casos benchmark y el resto C3.

Reutilizacion (importlib, SIN modificar los originales):
    scripts/16_anotador_gold.py: normalizacion de articulos y orden natural,
    catalogo de la CP, construccion de candidatos (union NER + top-10),
    lectura de NER de entidades y del corpus benchmark, terciles, helpers de
    jsonl/md5/extractos/guarda y resumenes.
    scripts/lib/componentes_producto.py: VinculoCP (ruta del producto v2.2
    con B2+B1) para el top-10 de los casos C3.

Estado del arte aplicado:
    - Relevancia graduada y nDCG (Jarvelin y Kekalainen, 2002) como marco
      del gold (la guia y el anotador 16 usan la escala 2/1/0).
    - Juicios agrupados (pooling) con union de NER + top-10 de la ruta
      automatica (tradicion TREC; Sakai, 2007).
    - Muestreo estratificado por anio con cuota equitativa (110 -> 22 por
      anio) y semilla fija 42 para reproducibilidad.
    - Ruta del producto v2.2 (B2 retriever + B1 reranker, ADR-18) como
      segunda fuente de candidatos en los casos C3.

Uso:
    /home/carlos/venv/bin/python scripts/32_pack_gold_150.py --limite 6 --forzar
    /home/carlos/venv/bin/python scripts/32_pack_gold_150.py
    /home/carlos/venv/bin/python scripts/32_pack_gold_150.py --dir-salida /tmp/opencode/gold_150_prueba
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lib.componentes_producto import VinculoCP

# ===== SECCION 1: configuracion =====

RAIZ = Path(__file__).resolve().parent.parent

CONFIG = {
    "semilla": 42,
    "n_total": 150,
    "n_benchmark": 40,
    "n_c3": 110,
    "periodo_c3": "2022-2026",
    "chars_extracto": 1500,
    "chars_snippet_articulo": 200,
    "estrategia_benchmark": "hibrido_b2_rerank_chunks_top50_b1",
    "k_c3": 10,
    "version_pack": "gold_150_v1",
    "dir_salida_default": "resultados/gold_150",
    "rutas_datos": {
        "benchmark_v2": "resultados/vinculo_cp_benchmark_v2.json",
        "benchmark_v8": "resultados/vinculo_cp_benchmark_v8.json",
        "benchmark_v13": "resultados/vinculo_cp_benchmark_v13.json",
        "muestra_b3": "resultados/gold_mini/muestra_gold.jsonl",
        "entidades": "data/entidades_sentencias.parquet",
        "articulos": "data/constitucion_articulos.parquet",
        "corpus": "data/corpus_sentencias.parquet",
        "c3_corpus": "data/corpus_2022_2026/corpus_2022_2026.parquet",
        "c3_entidades": "data/corpus_2022_2026/entidades_2022_2026.parquet",
    },
    "nombres_salida": {
        "muestra": "muestra_gold.jsonl",
        "candidatos": "candidatos.jsonl",
        "guia": "guia_anotacion_gold.md",
        "rutas_c3": "rutas_c3_gold150.json",
        "stats": "pack_stats.json",
        "cache_c3": "cache_rutas_c3.jsonl",
    },
}

# Premisas de diseno congeladas en las salidas (trazabilidad).
PREMISAS = [
    (
        "Muestra: 40 casos del benchmark M7 (los 90 de vinculo_cp_benchmark_v2.json "
        "menos los 50 anotados en el mini-gold B3) + 110 casos del corpus C3 2022-2026 "
        "(universo: estado ok y entidades NER; cuota equitativa ~22/anio; semilla 42)."
    ),
    (
        "Anti-leakage B1/B2: los pares de entrenamiento (data/pares_b1, scripts 23 y 25) "
        "excluyen los 90 ids del benchmark y provienen del corpus 1992-2021; el corpus C3 "
        "2022-2026 no participa en ellos. Los 40 casos benchmark son ids nuevos (no "
        "repiten el pack B3)."
    ),
    (
        "Candidatos por caso: union de arts_cp NER y el top-10 de la ruta del producto "
        "v2.2 (benchmark: hibrido_b2_rerank_chunks_top50_b1 de v13; C3: "
        "VinculoCP(usar_reranker=True).recuperar(texto, k=10) con scores en "
        "rutas_c3_gold150.json); ordenados por numero de articulo; snippet ~200 chars del "
        "catalogo de la CP. citado_ner es referencia automatica, no verdad."
    ),
    (
        "tercil_dificultad: terciles de n_citas_ner DENTRO del pack (0 = menos citas, "
        "mas dificil). Proxy documentado: C3 no tiene nDCG@10 por caso y n_citas_ner es "
        "la unica senal comun; el benchmark conserva su ndcg en v13 pero el pack usa el "
        "mismo proxy para los dos origenes."
    ),
    (
        "Anotacion humana graduada de relevancia 2/1/0 (script 16 con --pack gold_150). "
        "Es diagnostico: no entrena nada y no toca test ni el holdout sellado. La "
        "evaluacion contra este gold queda pendiente de los benchmarks v14."
    ),
    (
        "El corpus y los parquet se leen SOLO por columnas/filtros; no se modifica ningun "
        "archivo de datos ni resultado previo (v2..v13 y gold_mini intactos)."
    ),
]

LOGGER_NOMBRE = "pack_gold_150_32"

# Cache de modulos reutilizados (sus nombres empiezan por digito).
_MODULOS: dict[str, object] = {}


# ===== SECCION 2: reutilizacion por importlib y utilidades =====


def _cargar_modulo(ruta: Path, nombre: str) -> object:
    """Importa un script por ruta (su nombre empieza por digito).

    Args:
        ruta: Ruta del archivo .py a cargar.
        nombre: Nombre interno del modulo en sys.modules.

    Returns:
        Modulo ya ejecutado.

    Raises:
        ImportError: Si el archivo no existe o no se puede cargar.
    """
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"no se pudo cargar {ruta}")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def modulo_16() -> object:
    """Importa scripts/16_anotador_gold.py por ruta (cacheado).

    Returns:
        Modulo 16 ya ejecutado (candidatos, terciles, catalogo, helpers).
    """
    if "16" not in _MODULOS:
        _MODULOS["16"] = _cargar_modulo(RAIZ / "scripts" / "16_anotador_gold.py", "anotador_gold_16")
    return _MODULOS["16"]


def modulo_21() -> object:
    """Importa scripts/21_pack_validacion_m4.py por ruta (cacheado).

    Returns:
        Modulo 21 ya ejecutado (jsonl, md5, guardas y extractos).
    """
    if "21" not in _MODULOS:
        _MODULOS["21"] = _cargar_modulo(RAIZ / "scripts" / "21_pack_validacion_m4.py", "pack_validacion_m4_21")
    return _MODULOS["21"]


def cargar_benchmark(ruta: str | Path) -> dict:
    """Carga un JSON del benchmark M7 (modulo 16).

    Args:
        ruta: Ruta del JSON.

    Returns:
        Dict completo del benchmark.
    """
    return modulo_16().cargar_benchmark(ruta)


def validar_ids_benchmark(muestra_v2: list[dict], muestra_v8: list[dict], logger: logging.Logger) -> None:
    """Verifica que los ids de v2 y v8 coincidan (modulo 16).

    Args:
        muestra_v2: Lista de casos de v2.
        muestra_v8: Lista de casos de v8.
        logger: Logger activo.
    """
    modulo_16().validar_ids_benchmark(muestra_v2, muestra_v8, logger)


def cargar_jsonl(ruta: str | Path) -> list[dict]:
    """Carga un JSON Lines validando linea por linea (modulo 21).

    Args:
        ruta: Ruta del archivo.

    Returns:
        Lista de registros.
    """
    return modulo_21().cargar_jsonl(ruta)


def escribir_jsonl(ruta: str | Path, registros: list[dict]) -> None:
    """Escribe una lista de registros como JSON Lines (modulo 21).

    Args:
        ruta: Ruta de salida.
        registros: Lista de dicts serializables.
    """
    modulo_21().escribir_jsonl(ruta, registros)


def escribir_jsonl_append(ruta: str | Path, registro: dict) -> None:
    """Agrega un registro al final de un JSON Lines con flush (modulo 21).

    Args:
        ruta: Ruta del archivo (se crea si no existe).
        registro: Dict serializable.
    """
    modulo_21().escribir_jsonl_append(ruta, registro)


def verificar_salida_nueva(ruta: str | Path, forzar: bool, smoke: bool, logger: logging.Logger) -> Path:
    """Aborta si la salida final ya existe (guarda del modulo 21).

    Args:
        ruta: Ruta de salida.
        forzar: True permite reescribir.
        smoke: True si es corrida smoke (siempre reescribible).
        logger: Logger activo.

    Returns:
        Path de la salida validada.
    """
    return modulo_21().verificar_salida_nueva(ruta, forzar, smoke, logger)


def md5_archivo(ruta: str | Path) -> str | None:
    """Calcula el md5 de un archivo (modulo 21).

    Args:
        ruta: Ruta del archivo.

    Returns:
        Hexdigest md5, o None si el archivo no existe.
    """
    return modulo_21().md5_archivo(ruta)


def extracto_texto(texto: str, n_chars: int) -> str:
    """Extrae el inicio de un texto normalizando finales de linea (modulo 21).

    Args:
        texto: Texto completo.
        n_chars: Maximo de caracteres a conservar.

    Returns:
        Extracto del inicio.
    """
    return modulo_21().extracto_texto(texto, n_chars)


def asignar_terciles(valores: list[float], ids: list[str]) -> list[int]:
    """Asigna terciles de dificultad (0 = mas dificil) (modulo 16).

    Args:
        valores: Valor por caso (menor = mas dificil).
        ids: Ids alineados con valores (desempate determinista).

    Returns:
        Lista de terciles en {0, 1, 2} en el orden de entrada.
    """
    return modulo_16().asignar_terciles(valores, ids)


def asignar_periodo(anio: int) -> str:
    """Asigna un anio a su periodo cronologico del benchmark (modulo 16).

    Args:
        anio: Anio de la sentencia.

    Returns:
        Nombre del periodo (p. ej. "1992-2001") o "desconocido".
    """
    return modulo_16().asignar_periodo(anio)


def normalizar_articulo(valor: object) -> str:
    """Normaliza un numero de articulo de la CP (modulo 16).

    Args:
        valor: Numero crudo.

    Returns:
        Numero normalizado ("86", "22A", "Transitorio 5", "Preambulo").
    """
    return modulo_16().normalizar_articulo(valor)


def cargar_catalogo_articulos(rutas: dict) -> dict[str, str]:
    """Carga el catalogo completo de articulos de la CP (modulo 16).

    Args:
        rutas: Dict de rutas activas.

    Returns:
        Dict numero normalizado -> texto del articulo.
    """
    return modulo_16().cargar_catalogo_articulos(rutas)


def construir_candidatos(
    elegidos: list[dict],
    ner_por_id: dict[str, list[str]],
    top10_por_id: dict,
    catalogo: dict[str, str],
    logger: logging.Logger,
) -> list[dict]:
    """Construye candidatos.jsonl: union NER + top-10 por caso (modulo 16).

    Args:
        elegidos: Casos elegidos (id, indice, periodo, tercil).
        ner_por_id: Articulos NER por id.
        top10_por_id: id -> estrategia -> lista de articulos.
        catalogo: Texto de cada articulo de la CP.
        logger: Logger activo.

    Returns:
        Lista de registros por caso con la lista de candidatos ordenada.
    """
    return modulo_16().construir_candidatos(elegidos, ner_por_id, top10_por_id, catalogo, logger)


def cargar_top10_archivo(ruta: str | Path, logger: logging.Logger) -> dict:
    """Carga y normaliza top10_por_caso de un benchmark (modulo 16).

    Args:
        ruta: Ruta del JSON del benchmark.
        logger: Logger activo.

    Returns:
        Dict id_caso -> estrategia -> lista de articulos normalizados.
    """
    return modulo_16().cargar_top10_archivo(ruta, logger)


def cargar_ner_por_id(rutas: dict, ids: list[str], logger: logging.Logger) -> dict[str, list[str]]:
    """Carga arts_cp NER del benchmark por id (modulo 16).

    Args:
        rutas: Dict de rutas activas.
        ids: Ids a leer.
        logger: Logger activo.

    Returns:
        Dict id -> lista de articulos normalizados (puede faltar un id).
    """
    return modulo_16().cargar_ner_por_id(rutas, ids, logger)


def cargar_textos_corpus(rutas: dict, ids: list[str], logger: logging.Logger) -> dict[str, str]:
    """Lee el texto del corpus benchmark por id (modulo 16).

    Args:
        rutas: Dict de rutas activas.
        ids: Ids a leer.
        logger: Logger activo.

    Returns:
        Dict id -> texto (vacio si el corpus no lo tiene).
    """
    return modulo_16().cargar_textos_corpus(rutas, ids, logger)


def leer_filas_por_id(ruta: str, columnas: list[str], ids: list[str]) -> pd.DataFrame:
    """Lee solo las filas con los ids dados de un parquet (modulo 16 -> 12).

    Args:
        ruta: Ruta del parquet (archivo o directorio particionado).
        columnas: Columnas a leer.
        ids: Lista de ids (strings) a conservar.

    Returns:
        DataFrame con las filas cuyos ids estan en la lista.
    """
    return modulo_16().leer_filas_por_id(ruta, columnas, ids)


def resumen_tamanos(valores: list[int]) -> dict:
    """Resume la distribucion de una lista de enteros (modulo 16).

    Args:
        valores: Lista de valores (p. ej. n candidatos por caso).

    Returns:
        Dict con n, min, max, media, mediana, p25, p75 y total.
    """
    return modulo_16().resumen_tamanos(valores)


def escribir_guia(destino: str | Path, contenido: str, logger: logging.Logger) -> None:
    """Escribe la guia de anotacion en Markdown (modulo 16).

    Args:
        destino: Ruta de salida.
        contenido: Texto completo de la guia.
        logger: Logger activo.
    """
    modulo_16().escribir_guia(destino, contenido, logger)


def configurar_logger() -> logging.Logger:
    """Configura el logger de consola del generador del pack.

    Returns:
        Logger con formato fecha | nivel | mensaje, solo consola.
    """
    logger = logging.getLogger(LOGGER_NOMBRE)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formato = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    manejador = logging.StreamHandler()
    manejador.setFormatter(formato)
    logger.addHandler(manejador)
    logger.propagate = False
    return logger


def ahora_iso() -> str:
    """Devuelve la marca de tiempo actual en ISO (segundos).

    Returns:
        Timestamp ISO-8601 sin zona.
    """
    return pd.Timestamp.now().isoformat(timespec="seconds")


def clave_id(valor: object) -> int | str:
    """Clave de orden determinista para ids (numericos primero).

    Args:
        valor: Id del caso.

    Returns:
        Entero si el id es numerico, o el string original.
    """
    texto = str(valor)
    return int(texto) if texto.isdigit() else texto


def rutas_datos_activas() -> dict:
    """Devuelve el mapa de rutas de datos de entrada.

    Returns:
        Dict con las rutas de benchmarks, entidades, corpus y catalogo.
    """
    return dict(CONFIG["rutas_datos"])


def rutas_salida(args: argparse.Namespace) -> dict[str, Path]:
    """Construye el mapa de rutas de salida del pack en --dir-salida.

    Args:
        args: Argumentos CLI (dir_salida).

    Returns:
        Dict nombre -> Path de cada salida del pack.
    """
    directorio = Path(args.dir_salida)
    return {clave: directorio / nombre for clave, nombre in CONFIG["nombres_salida"].items()}


def escribir_json(destino: str | Path, datos: dict, logger: logging.Logger) -> None:
    """Escribe un dict como JSON indentado (trazabilidad del pack).

    Args:
        destino: Ruta de salida.
        datos: Dict serializable.
        logger: Logger activo.
    """
    archivo = Path(destino)
    archivo.parent.mkdir(parents=True, exist_ok=True)
    with open(archivo, "w", encoding="utf-8") as manejador:
        json.dump(datos, manejador, ensure_ascii=False, indent=2, default=str)
    logger.info("json guardado: %s (md5 %s)", archivo, md5_archivo(archivo))


# ===== SECCION 3: universo y muestreo del benchmark =====


def cargar_casos_benchmark_restantes(rutas: dict, logger: logging.Logger) -> list[dict]:
    """Carga los casos del benchmark v2 que no estan en el pack B3.

    Verifica la integridad de los 90 ids contra v8 (modulo 16) y descuenta
    los ids ya anotados en resultados/gold_mini/muestra_gold.jsonl.

    Args:
        rutas: Dict de rutas activas.
        logger: Logger activo.

    Returns:
        Lista de casos de v2 (id, tipo, ano, arts_cp) no anotados en B3.

    Raises:
        FileNotFoundError: Si falta el benchmark o la muestra B3.
        RuntimeError: Si los ids de v2 no coinciden con v8.
    """
    benchmark = cargar_benchmark(rutas["benchmark_v2"])
    muestra = benchmark["muestra"]
    v8 = cargar_benchmark(rutas["benchmark_v8"])
    validar_ids_benchmark(muestra, v8["muestra"], logger)
    anotados = {str(registro["id"]) for registro in cargar_jsonl(rutas["muestra_b3"])}
    restantes = [caso for caso in muestra if str(caso["id"]) not in anotados]
    logger.info(
        "benchmark M7: %d casos | anotados en B3: %d | restantes: %d",
        len(muestra),
        len(anotados),
        len(restantes),
    )
    return restantes


def elegir_benchmark(restantes: list[dict], n_objetivo: int, logger: logging.Logger) -> list[dict]:
    """Elige casos del benchmark con semilla 42 y orden determinista.

    Args:
        restantes: Casos del benchmark no anotados en B3.
        n_objetivo: Numero de casos a elegir (40 en la corrida completa).
        logger: Logger activo.

    Returns:
        Lista de casos elegidos con id, tipo, ano, periodo, origen y arts_cp.

    Raises:
        RuntimeError: Si no hay suficientes casos para el objetivo.
    """
    ordenados = sorted(restantes, key=lambda caso: clave_id(str(caso["id"])))
    if n_objetivo > len(ordenados):
        raise RuntimeError(f"benchmark insuficiente: {len(ordenados)} casos para objetivo {n_objetivo}")
    rng = np.random.default_rng(CONFIG["semilla"])
    posiciones = np.sort(rng.choice(len(ordenados), size=n_objetivo, replace=False))
    elegidos = []
    for posicion in posiciones:
        caso = ordenados[int(posicion)]
        elegidos.append(
            {
                "id": str(caso["id"]),
                "tipo": str(caso["tipo"]),
                "ano": int(caso["ano"]),
                "periodo": asignar_periodo(int(caso["ano"])),
                "origen": "benchmark",
                "arts_cp_benchmark": [normalizar_articulo(articulo) for articulo in caso.get("arts_cp", [])],
            }
        )
    logger.info("benchmark elegido: %d casos (objetivo %d)", len(elegidos), n_objetivo)
    return elegidos


# ===== SECCION 4: universo y muestreo del corpus C3 =====


def cargar_meta_corpus_c3(rutas: dict) -> pd.DataFrame:
    """Carga la metadata liviana del corpus C3 (sin textos).

    Args:
        rutas: Dict de rutas activas.

    Returns:
        DataFrame con id, tipo, anio y estado de cada documento.
    """
    tabla = pq.read_table(rutas["c3_corpus"], columns=["id", "tipo", "anio", "estado"])
    return tabla.to_pandas()


def cargar_ids_entidades_c3(rutas: dict) -> set[str]:
    """Carga los ids con entidades NER del corpus C3.

    Args:
        rutas: Dict de rutas activas.

    Returns:
        Conjunto de ids presentes en entidades_2022_2026.parquet.
    """
    tabla = pq.read_table(rutas["c3_entidades"], columns=["id"])
    return {str(valor) for valor in tabla.column("id").to_pylist()}


def cuotas_equitativas(anios: list[str], objetivo: int) -> dict[str, int]:
    """Reparte un objetivo lo mas parejo posible entre anios (determinista).

    Con objetivo multiplo del numero de anios la cuota es exactamente igual
    (110 -> 22 por anio); en otro caso los primeros anios del orden reciben
    un caso extra (N=3 -> 1, 1, 1, 0, 0).

    Args:
        anios: Lista de anios (claves) en orden determinista.
        objetivo: Numero total de casos a repartir.

    Returns:
        Dict anio -> cuota asignada.
    """
    base, resto = divmod(int(objetivo), len(anios))
    return {anio: base + (1 if posicion < resto else 0) for posicion, anio in enumerate(anios)}


def muestrear_c3(meta: pd.DataFrame, ids_entidades: set[str], n_objetivo: int, logger: logging.Logger) -> list[dict]:
    """Muestrea casos C3 por anio con cuota equitativa y semilla 42.

    El universo son las filas con estado "ok" cuyo id tiene entidades NER.
    La cuota por anio es EQUITATIVA (~22 de 110; reparto parejo con los
    primeros anios del orden recibiendo el resto); con --limite el reparto
    parejo se aplica igual (N=3 -> 1 por anio en los primeros anios). Dentro
    de cada anio la eleccion es determinista sobre ids ordenados.

    Args:
        meta: Metadata del corpus C3 (id, tipo, anio, estado).
        ids_entidades: Ids con entidades NER.
        n_objetivo: Numero de casos a elegir (110 en la corrida completa).
        logger: Logger activo.

    Returns:
        Lista de casos elegidos con id, tipo, ano, periodo y origen.

    Raises:
        RuntimeError: Si el universo no alcanza el objetivo o una cuota
            excede los documentos disponibles de su anio.
    """
    elegible = meta[(meta["estado"] == "ok") & (meta["id"].isin(ids_entidades))].copy()
    anios = sorted({str(int(anio)) for anio in elegible["anio"]}, key=int)
    cuotas = cuotas_equitativas(anios, n_objetivo)
    logger.info(
        "universo C3 elegible: %d casos | cuotas equitativas por anio (objetivo %d): %s",
        len(elegible),
        n_objetivo,
        cuotas,
    )
    rng = np.random.default_rng(CONFIG["semilla"])
    elegidos: list[dict] = []
    for anio in sorted(cuotas, key=int):
        cuota = int(cuotas[anio])
        if cuota == 0:
            continue
        sub = elegible[elegible["anio"] == int(anio)].sort_values("id")
        if cuota > len(sub):
            raise RuntimeError(f"anio {anio}: cuota {cuota} > documentos {len(sub)}")
        posiciones = np.sort(rng.choice(len(sub), size=cuota, replace=False))
        for posicion in posiciones:
            fila = sub.iloc[int(posicion)]
            elegidos.append(
                {
                    "id": str(fila["id"]),
                    "tipo": str(fila["tipo"]),
                    "ano": int(fila["anio"]),
                    "periodo": CONFIG["periodo_c3"],
                    "origen": "c3",
                }
            )
    if len(elegidos) != n_objetivo:
        raise RuntimeError(f"muestreo C3 incompleto: {len(elegidos)} de {n_objetivo} casos")
    logger.info("C3 elegido: %d casos | por anio: %s", len(elegidos), dict(sorted(Counter(e["ano"] for e in elegidos).items())))
    return elegidos


# ===== SECCION 5: NER, textos y rutas C3 =====


def cargar_ner_benchmark(rutas: dict, elegidos: list[dict], logger: logging.Logger) -> dict[str, list[str]]:
    """Carga arts_cp NER de los casos benchmark (entidades con respaldo v2).

    Args:
        rutas: Dict de rutas activas.
        elegidos: Casos benchmark elegidos (con arts_cp_benchmark).
        logger: Logger activo.

    Returns:
        Dict id -> articulos NER normalizados.
    """
    ner_entidades = cargar_ner_por_id(rutas, [elegido["id"] for elegido in elegidos], logger)
    ner: dict[str, list[str]] = {}
    for elegido in elegidos:
        articulos = ner_entidades.get(elegido["id"])
        ner[elegido["id"]] = list(articulos) if articulos else list(elegido["arts_cp_benchmark"])
    return ner


def cargar_ner_c3(rutas: dict, ids: list[str], logger: logging.Logger) -> dict[str, list[str]]:
    """Carga arts_cp NER de los casos C3 por id.

    Args:
        rutas: Dict de rutas activas.
        ids: Ids elegidos de C3.
        logger: Logger activo.

    Returns:
        Dict id -> articulos NER normalizados.
    """
    tabla = leer_filas_por_id(rutas["c3_entidades"], ["id", "arts_cp"], ids)
    ner: dict[str, list[str]] = {}
    for fila in tabla.to_dict("records"):
        try:
            articulos = json.loads(fila["arts_cp"] or "[]")
        except (TypeError, ValueError):
            articulos = []
        ner[str(fila["id"])] = [normalizar_articulo(articulo) for articulo in articulos]
    faltantes = [id_caso for id_caso in ids if id_caso not in ner]
    if faltantes:
        logger.warning("sin arts_cp en entidades C3 para %d ids: %s", len(faltantes), faltantes[:10])
    return ner


def cargar_textos_c3(rutas: dict, ids: list[str], logger: logging.Logger) -> dict[str, str]:
    """Lee el texto limpiado E1/E2 de los ids C3 pedidos.

    Args:
        rutas: Dict de rutas activas.
        ids: Ids elegidos de C3.
        logger: Logger activo.

    Returns:
        Dict id -> texto (vacio si el corpus no lo tiene).
    """
    tabla = leer_filas_por_id(rutas["c3_corpus"], ["id", "texto"], ids)
    textos = {
        str(fila["id"]): (fila["texto"] if isinstance(fila["texto"], str) else "")
        for fila in tabla.to_dict("records")
    }
    faltantes = [id_caso for id_caso in ids if not textos.get(id_caso)]
    if faltantes:
        logger.warning("%d casos C3 sin texto en el corpus: %s", len(faltantes), faltantes[:10])
    return textos


def cargar_cache_rutas_c3(ruta_cache: str | Path, logger: logging.Logger) -> dict[str, list[dict]]:
    """Carga el cache por id de rankings C3 (ultimo registro gana).

    Args:
        ruta_cache: Ruta del JSON Lines de cache.
        logger: Logger activo.

    Returns:
        Dict id -> lista de {numero, score}.

    Raises:
        ValueError: Si alguna linea no es JSON valido.
    """
    cache: dict[str, list[dict]] = {}
    archivo = Path(ruta_cache)
    if not archivo.exists():
        return cache
    invalidos = 0
    with open(archivo, encoding="utf-8") as manejador:
        for numero, linea in enumerate(manejador, 1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                registro = json.loads(linea)
            except json.JSONDecodeError as exc:
                raise ValueError(f"jsonl de cache invalido en {archivo}:{numero}") from exc
            if "id" not in registro or not isinstance(registro.get("ranking"), list):
                invalidos += 1
                continue
            cache[str(registro["id"])] = registro["ranking"]
    if invalidos:
        logger.warning("%s: %d registros de cache invalidos ignorados", archivo, invalidos)
    logger.info("cache de rutas C3: %d ids disponibles en %s", len(cache), archivo)
    return cache


def recuperar_rutas_c3(
    elegidos: list[dict],
    textos: dict[str, str],
    ruta_cache: str | Path,
    logger: logging.Logger,
) -> dict[str, list[dict]]:
    """Recupera el top-10 de la ruta del producto v2.2 para los casos C3.

    Usa VinculoCP(usar_reranker=True): hibrido 0.5*BM25 + 0.5*B2, rerank B1
    por chunks, prior de hubs alpha=1.5 y k=10. Cada ranking se escribe en
    el cache por id (reanudable) antes de continuar con el siguiente caso;
    los ids ya cacheados no se vuelven a recuperar.

    Args:
        elegidos: Casos C3 elegidos.
        textos: Texto limpiado por id.
        ruta_cache: Ruta del JSON Lines de cache por id.
        logger: Logger activo.

    Returns:
        Dict id -> ranking normalizado [{numero, score}] de los elegidos.

    Raises:
        RuntimeError: Si VinculoCP no esta disponible.
    """
    cache = cargar_cache_rutas_c3(ruta_cache, logger)
    faltantes = [elegido for elegido in elegidos if elegido["id"] not in cache]
    logger.info("rutas C3: %d en cache | %d por recuperar", len(elegidos) - len(faltantes), len(faltantes))
    if faltantes:
        vinculo = VinculoCP(usar_reranker=True)
        logger.info("VinculoCP v2.2 cargado (ruta reranker B2+B1; carga lazy de modelos en el primer recuperar)")
        for posicion, elegido in enumerate(faltantes, 1):
            id_caso = elegido["id"]
            ranking = vinculo.recuperar(textos.get(id_caso, ""), k=CONFIG["k_c3"])
            normalizado = [
                {
                    "numero": normalizar_articulo(item.get("numero")),
                    "score": round(float(item.get("score", 0.0)), 4),
                }
                for item in ranking
            ]
            if not normalizado:
                logger.warning("ruta C3 vacia para %s (texto de %d chars)", id_caso, len(textos.get(id_caso, "")))
            escribir_jsonl_append(ruta_cache, {"id": id_caso, "fecha": ahora_iso(), "ranking": normalizado})
            cache[id_caso] = normalizado
            logger.info(
                "[%d/%d] ruta C3 %s: %d articulos | top1=%s (%.4f)",
                posicion,
                len(faltantes),
                id_caso,
                len(normalizado),
                normalizado[0]["numero"] if normalizado else "n/d",
                normalizado[0]["score"] if normalizado else 0.0,
            )
    return {elegido["id"]: cache.get(elegido["id"], []) for elegido in elegidos}


def escribir_rutas_c3(destino: str | Path, rutas_por_id: dict[str, list[dict]], logger: logging.Logger) -> None:
    """Persiste los rankings C3 con scores para la evaluacion futura.

    Args:
        destino: Ruta del JSON de salida.
        rutas_por_id: Dict id -> ranking [{numero, score}].
        logger: Logger activo.
    """
    contenido = {
        "meta": {
            "fecha": ahora_iso(),
            "script": "scripts/32_pack_gold_150.py",
            "descripcion": "top-10 de la ruta del producto v2.2 por caso C3 (evaluacion futura)",
            "componentes": (
                "VinculoCP(usar_reranker=True): hibrido 0.5*BM25+0.5*B2, rerank B1 por chunks, "
                "prior de hubs alpha=1.5, k=10; numeros normalizados"
            ),
            "k": CONFIG["k_c3"],
            "semilla": CONFIG["semilla"],
            "n_casos": len(rutas_por_id),
        },
        "por_id": rutas_por_id,
    }
    escribir_json(destino, contenido, logger)


# ===== SECCION 6: construccion del pack =====


def repartir_limite(limite: int | None, logger: logging.Logger) -> tuple[int, int]:
    """Reparte el tope --limite entre benchmark y C3 (pruebas).

    Sin --limite devuelve la composicion completa (40 + 110). Con --limite N
    toma ceil(N/2) casos benchmark y el resto de C3 (N=6 -> 3 + 3).

    Args:
        limite: Tope total de casos o None (corrida completa).
        logger: Logger activo.

    Returns:
        Tupla (n_benchmark, n_c3).
    """
    if limite is None:
        return CONFIG["n_benchmark"], CONFIG["n_c3"]
    if limite > CONFIG["n_total"]:
        logger.warning("--limite %d > %d: se recorta al pack completo", limite, CONFIG["n_total"])
    n_benchmark = min(CONFIG["n_benchmark"], (limite + 1) // 2)
    n_c3 = min(CONFIG["n_c3"], max(0, limite - n_benchmark))
    return n_benchmark, n_c3


def ordenar_elegidos(elegidos_benchmark: list[dict], elegidos_c3: list[dict]) -> list[dict]:
    """Ordena los casos (benchmark por id, C3 por anio e id) e indiza 1..N.

    Args:
        elegidos_benchmark: Casos benchmark elegidos.
        elegidos_c3: Casos C3 elegidos.

    Returns:
        Lista combinada y ordenada con indice 1..N y tercil pendiente.
    """
    ordenados = sorted(elegidos_benchmark, key=lambda elegido: clave_id(elegido["id"]))
    ordenados += sorted(elegidos_c3, key=lambda elegido: (int(elegido["ano"]), clave_id(elegido["id"])))
    for indice, elegido in enumerate(ordenados, 1):
        elegido["indice"] = indice
        elegido["tercil"] = 0
    return ordenados


def construir_top10_pack(
    elegidos: list[dict],
    top10_benchmark: dict,
    rutas_c3: dict[str, list[dict]],
) -> dict:
    """Construye el mapa top-10 por id con el formato que espera el 16.

    Args:
        elegidos: Casos del pack (con origen).
        top10_benchmark: top10 de v13 (id -> estrategia -> articulos).
        rutas_c3: Rankings C3 (id -> [{numero, score}]).

    Returns:
        Dict id -> estrategia -> lista de articulos.

    Raises:
        RuntimeError: Si la estrategia del 16 diverge de la configurada.
    """
    estrategia = CONFIG["estrategia_benchmark"]
    if modulo_16().CONFIG["estrategia_candidatos"] != estrategia:
        raise RuntimeError(
            "la estrategia del script 16 cambio: "
            f"{modulo_16().CONFIG['estrategia_candidatos']} != {estrategia}"
        )
    top10: dict = {}
    for elegido in elegidos:
        if elegido["origen"] == "benchmark":
            top10[elegido["id"]] = top10_benchmark.get(elegido["id"], {})
        else:
            top10[elegido["id"]] = {
                estrategia: [item["numero"] for item in rutas_c3.get(elegido["id"], [])]
            }
    return top10


def construir_registros_muestra(
    elegidos: list[dict],
    ner_por_id: dict[str, list[str]],
    textos: dict[str, str],
    logger: logging.Logger,
) -> tuple[list[dict], int]:
    """Construye muestra_gold.jsonl con el formato del pack B3 mas origen.

    Args:
        elegidos: Casos del pack (indice, id, tipo, ano, periodo, tercil, origen).
        ner_por_id: Articulos NER por id.
        textos: Texto completo por id.
        logger: Logger activo.

    Returns:
        Tupla (registros de la muestra, numero de casos sin texto).
    """
    registros = []
    sin_texto = 0
    for elegido in elegidos:
        id_caso = elegido["id"]
        texto = textos.get(id_caso, "")
        if not texto:
            sin_texto += 1
        registros.append(
            {
                "indice": int(elegido["indice"]),
                "id": id_caso,
                "tipo": elegido["tipo"],
                "ano": int(elegido["ano"]),
                "periodo": elegido["periodo"],
                "tercil_dificultad": int(elegido["tercil"]),
                "n_citas_ner": len(ner_por_id.get(id_caso, [])),
                "extracto": extracto_texto(texto, CONFIG["chars_extracto"]),
                "origen": elegido["origen"],
            }
        )
    if sin_texto:
        logger.warning("%d casos del pack sin texto en el corpus", sin_texto)
    return registros, sin_texto


def construir_stats(
    registros: list[dict],
    registros_candidatos: list[dict],
    rutas_c3: dict[str, list[dict]],
    n_sin_texto: int,
    args: argparse.Namespace,
    salidas: dict[str, Path],
) -> dict:
    """Construye pack_stats.json con la trazabilidad completa del pack.

    Args:
        registros: Registros de muestra_gold.jsonl.
        registros_candidatos: Registros de candidatos.jsonl.
        rutas_c3: Rankings C3 por id.
        n_sin_texto: Casos sin texto en el corpus.
        args: Argumentos CLI.
        salidas: Dict de rutas de salida.

    Returns:
        Dict de estadisticas del pack.
    """
    por_origen = Counter(registro["origen"] for registro in registros)
    por_periodo = Counter(registro["periodo"] for registro in registros)
    terciles = Counter(registro["tercil_dificultad"] for registro in registros)
    terciles_por_origen = {
        origen: dict(sorted(Counter(registro["tercil_dificultad"] for registro in registros if registro["origen"] == origen).items()))
        for origen in sorted(por_origen)
    }
    c3_por_anio = dict(
        sorted(Counter(registro["ano"] for registro in registros if registro["origen"] == "c3").items())
    )
    n_candidatos = [registro["n_candidatos"] for registro in registros_candidatos]
    n_sin_catalogo = sum(
        1 for registro in registros_candidatos for candidato in registro["candidatos"] if not candidato["snippet"]
    )
    n_items_c3 = sum(len(lista) for lista in rutas_c3.values())
    return {
        "meta": {
            "fecha": ahora_iso(),
            "script": "scripts/32_pack_gold_150.py",
            "version_pack": CONFIG["version_pack"],
            "semilla": CONFIG["semilla"],
            "limite": args.limite,
            "dir_salida": str(args.dir_salida),
            "estrategia_benchmark": CONFIG["estrategia_benchmark"],
            "k_c3": CONFIG["k_c3"],
            "n_total_esperado": CONFIG["n_total"],
        },
        "n_total": len(registros),
        "por_origen": dict(sorted(por_origen.items())),
        "c3_por_anio": c3_por_anio,
        "periodos": dict(sorted(por_periodo.items())),
        "tercil_dificultad": {
            "fuente": (
                "terciles de n_citas_ner dentro del pack (0 = menos citas, mas dificil); "
                "proxy documentado porque C3 no tiene nDCG@10 por caso"
            ),
            "distribucion": dict(sorted(terciles.items())),
            "por_origen": terciles_por_origen,
        },
        "n_citas_ner": resumen_tamanos([registro["n_citas_ner"] for registro in registros]),
        "candidatos_por_caso": resumen_tamanos(n_candidatos),
        "n_casos_sin_texto": int(n_sin_texto),
        "n_candidatos_sin_catalogo": int(n_sin_catalogo),
        "rutas_c3": {
            "n_casos": len(rutas_c3),
            "n_items": int(n_items_c3),
            "n_casos_vacios": sum(1 for lista in rutas_c3.values() if not lista),
        },
        "md5": {
            "muestra": md5_archivo(salidas["muestra"]),
            "candidatos": md5_archivo(salidas["candidatos"]),
            "guia": md5_archivo(salidas["guia"]),
            "rutas_c3": md5_archivo(salidas["rutas_c3"]),
        },
        "premisas": PREMISAS,
    }


# ===== SECCION 7: guia de anotacion =====


def texto_guia_gold150(
    n_muestra: int,
    n_benchmark: int,
    n_c3: int,
    total_candidatos: int,
    max_candidatos: int,
    dir_salida: str,
) -> str:
    """Construye la guia de anotacion del gold estandar C1 (Markdown).

    Args:
        n_muestra: Tamano de la muestra (150 en la corrida completa).
        n_benchmark: Casos del benchmark M7 en la muestra.
        n_c3: Casos del corpus C3 en la muestra.
        total_candidatos: Total de decisiones de grado de la muestra.
        max_candidatos: Maximo de candidatos por caso en la muestra.
        dir_salida: Directorio de salida del pack.

    Returns:
        Contenido completo de la guia.
    """
    fecha = pd.Timestamp.now().strftime("%Y-%m-%d")
    return f"""# Guia de anotacion - Gold estandar C1 (150 casos)

Fecha de generacion: {fecha}
Script: scripts/32_pack_gold_150.py (semilla 42)
Muestra: muestra_gold.jsonl ({n_muestra} casos: {n_benchmark} del benchmark M7
+ {n_c3} del corpus C3 2022-2026)
Directorio de trabajo: {dir_salida}/

## 1. Objetivo y alcance

Medir con juicio humano la relevancia de los articulos candidatos de cada caso
para el vinculo sentencia -> articulos de la Constitucion (CP). Este gold
estandar del Sprint C1 del ciclo v3 es la referencia humana para re-evaluar
las rutas del proyecto con relevancia graduada.

Es un diagnostico: NO entrena nada, no toca el test ni el holdout sellado y no
modifica ningun resultado previo del proyecto.

## 2. Dos origenes de la muestra

- Benchmark M7 ({n_benchmark} casos, 1992-2021): los 90 casos de
  resultados/vinculo_cp_benchmark_v2.json menos los 50 ya anotados en el
  mini-gold B3. Son ids NUEVOS para anotacion (no repiten el trabajo de B3).
- Corpus C3 2022-2026 ({n_c3} casos, ~22 por anio): muestreo con semilla 42
  sobre data/corpus_2022_2026/corpus_2022_2026.parquet (estado ok y con
  entidades NER).

Anti-leakage B1/B2: los pares de entrenamiento de los modelos de dominio
(data/pares_b1, scripts 23 y 25) excluyen los 90 ids del benchmark y se
construyen sobre el corpus 1992-2021; el corpus C3 2022-2026 no participa en
ellos. Por eso la anotacion de los dos origenes no evalua sobre datos vistos.

Cada caso indica su origen en el campo "origen" de muestra_gold.jsonl.

## 3. Que se mide: relevancia graduada

Para cada articulo candidato de un caso, asigne un grado de relevancia pensando
en la sentencia como un todo:

- 2 = fundamento: el articulo es base normativa directa de lo decidido o de la
  ratio analizada. La sentencia lo cita o lo desarrolla como sustento central.
- 1 = mencionado: el articulo aparece en el texto o se relaciona de forma
  tangencial (transcripciones de normas, listados, contexto, citas de paso),
  pero no es base del analisis.
- 0 = no relevante: el articulo no tiene relacion con lo que la sentencia decide.

Ejemplos:
- Una tutela sobre salud que funda el caso en el derecho a la vida: el articulo
  11 (vida) suele ser 2; el articulo 86 (tutela) puede ser 1 o 2 segun si el
  analisis lo desarrolla; un articulo de regimen economico citado de paso es 0.
- Una sentencia de constitucionalidad de una ley: los articulos de la CP que
  fijan la competencia o el parametro de control usado en la ratio son 2; los
  que solo figuran en el encabezado o en transcripciones son 1.

Casos limite: si duda entre 2 y 1, piense si el fallo o su ratio cambiarian sin
ese articulo; si la respuesta es no, use 1. Entre 1 y 0, piense si el articulo
tiene alguna conexion tematica real con lo decidido; si no, use 0.

## 4. Candidatos y el flag citado_ner

La lista de candidatos de cada caso es la union de (a) los articulos de la CP
citados detectados por NER y (b) el top-10 de la ruta automatica del producto
v2.2 por caso:

- Benchmark: top-10 de la ruta hibrido_b2_rerank_chunks_top50_b1 guardado en
  resultados/vinculo_cp_benchmark_v13.json.
- C3: top-10 de VinculoCP(usar_reranker=True).recuperar(texto, k=10) (hibrido
  BM25+B2 con rerank B1 y prior de hubs); los rankings con scores quedan en
  rutas_c3_gold150.json para la evaluacion futura.

Por eso la lista contiene articulos posiblemente no relevantes: su tarea es
precisamente graduarlos.

El flag citado_ner (que aparece al cerrar el caso) es una REFERENCIA automatica,
no una verdad: indica que el extractor de citas marco ese articulo. No lo use
para decidir el grado; se imprime al final para no sesgar la lectura.

## 5. Aviso sobre el tercil de dificultad (proxy)

El campo tercil_dificultad (0 = mas dificil, 2 = mas facil) se calculo con los
terciles de n_citas_ner DENTRO de este pack (menos citas NER = tercil 0). Es un
PROXY documentado: los 40 casos del benchmark tienen nDCG@10 por caso en v13,
pero los {n_c3} casos C3 no tienen ninguna ruta evaluada y n_citas_ner es la
unica senal comun comparable. No use el tercil para decidir grados: es solo
orientacion.

## 6. Extractos de la sentencia

- Benchmark: encabezado del texto del corpus (hasta 1500 chars).
- C3: el parquet 2022-2026 guarda texto limpiado E1/E2 (minusculas, sin
  digitos ni puntuacion); el extracto lo refleja. Para mas contexto abra el
  documento fuente del caso (URL de relatoria).

## 7. Regla de oro

1. Lea el extracto de la sentencia que acompana al caso.
2. Para cada articulo, lea el snippet; si necesita mas contexto, abra el texto
   completo del articulo en data/constitucion_articulos.json o el documento
   fuente del caso.
3. Decida el grado SOLO con la sentencia y el articulo.
4. No consulte rankings, scores, resultados de modelos ni anotaciones de otros
   anotadores. En particular NO abra rutas_c3_gold150.json (contiene los scores
   de la ruta C3) ni los resultados de la evaluacion futura.

## 8. Tiempo estimado

La muestra tiene {n_muestra} casos con {total_candidatos} decisiones de grado en
total (el caso mas grande tiene {max_candidatos} candidatos). A un ritmo de 10 a
20 segundos por candidato el rango tipico es de 4 a 8 horas. Se puede anotar en
tandas: el progreso se guarda despues de cada candidato.

## 9. Reanudacion y comandos

    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --pack gold_150 --resumen
    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --pack gold_150 --anotar --anotador carlos
    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --pack gold_150 --exportar-csv
    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --pack gold_150 --calcular-acuerdo

Durante --anotar, por candidato: 2, 1 o 0 para calificar; Enter deja el
candidato pendiente para otra sesion; s salta el caso completo (queda
pendiente); q guarda y sale; ? muestra la ayuda. Al reanudar, los candidatos ya
calificados se muestran marcados y no se vuelven a preguntar.

Cada linea de captura muestra el numero de articulo (por ejemplo
`[ 6/19] art. 13 | grado ...`) para seguir la lista sin subir. Tambien se puede
calificar de corrido con un lote: escriba varios grados separados por espacios
(por ejemplo `0 2 0 0 0 - 1`) y se aplican, en orden, a los candidatos
pendientes tal como aparecen en pantalla; `-` deja ese candidato pendiente. El
lote se valida completo: si algun token no es 2, 1, 0 o -, no se aplica nada.

## 10. Duplicados

La lista de candidatos ya esta deduplicada por numero de articulo y ordenada.
Articulos como 22 y 22A son distintos: califique cada uno. Si dos casos parecen
la misma sentencia, anote ambos: cada caso cuenta.

## 11. Acuerdo y evaluacion

Un segundo anotador puede correr --anotar con otro nombre (por ejemplo
--anotador ana, siempre con --pack gold_150) sobre la misma muestra; despues
--calcular-acuerdo reporta kappa de Cohen por pares y Fleiss.

La evaluacion contra este gold (nDCG@5/@10 graduado, precision y recall) queda
PENDIENTE de los benchmarks v14: --evaluar con --pack gold_150 falla a proposito
("benchmarks v14 pendientes") porque las rutas del benchmark v13 solo cubren
90 casos y las rutas C3 viven en rutas_c3_gold150.json. En la evaluacion, un
candidato anotado sin grado cuenta como 0 (documentado); por eso conviene
completar los casos antes de evaluar.

## 12. Recordatorio

Esta anotacion es humana y es diagnostico. No es una etiqueta automatica y no
se usa para entrenar; su unico fin es auditar las rutas con ojos humanos.
"""


# ===== SECCION 8: generacion del pack =====


def modo_generar(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Orquesta la generacion del pack C1 (muestra, candidatos, guia, stats).

    Args:
        args: Argumentos CLI (limite, forzar, dir_salida).
        logger: Logger activo.

    Raises:
        FileExistsError: Si una salida existe y no hay --forzar.
        RuntimeError: Si el benchmark no es integro, el muestreo falla o la
            composicion completa (40 + 110) no se cumple.
    """
    t0 = time.perf_counter()
    rutas = rutas_datos_activas()
    salidas = rutas_salida(args)
    for clave in ("muestra", "candidatos", "guia", "rutas_c3", "stats"):
        verificar_salida_nueva(salidas[clave], args.forzar, False, logger)
    n_benchmark, n_c3 = repartir_limite(args.limite, logger)
    logger.info("composicion objetivo: %d benchmark + %d C3 = %d casos", n_benchmark, n_c3, n_benchmark + n_c3)
    restantes = cargar_casos_benchmark_restantes(rutas, logger)
    if args.limite is None and len(restantes) != CONFIG["n_benchmark"]:
        raise RuntimeError(
            f"se esperaban {CONFIG['n_benchmark']} casos benchmark restantes, hay {len(restantes)}; "
            "revisar resultados/gold_mini/muestra_gold.jsonl"
        )
    elegidos_benchmark = elegir_benchmark(restantes, n_benchmark, logger)
    meta_c3 = cargar_meta_corpus_c3(rutas)
    ids_entidades = cargar_ids_entidades_c3(rutas)
    elegidos_c3 = muestrear_c3(meta_c3, ids_entidades, n_c3, logger) if n_c3 > 0 else []
    ner_benchmark = cargar_ner_benchmark(rutas, elegidos_benchmark, logger)
    ner_c3 = cargar_ner_c3(rutas, [elegido["id"] for elegido in elegidos_c3], logger) if elegidos_c3 else {}
    ner_por_id = {**ner_benchmark, **ner_c3}
    textos_benchmark = cargar_textos_corpus(rutas, [elegido["id"] for elegido in elegidos_benchmark], logger)
    textos_c3 = cargar_textos_c3(rutas, [elegido["id"] for elegido in elegidos_c3], logger) if elegidos_c3 else {}
    textos = {**textos_benchmark, **textos_c3}
    rutas_c3 = recuperar_rutas_c3(elegidos_c3, textos, salidas["cache_c3"], logger)
    escribir_rutas_c3(salidas["rutas_c3"], rutas_c3, logger)
    elegidos = ordenar_elegidos(elegidos_benchmark, elegidos_c3)
    valores = [len(ner_por_id.get(elegido["id"], [])) for elegido in elegidos]
    terciles = asignar_terciles(valores, [elegido["id"] for elegido in elegidos])
    for elegido, tercil in zip(elegidos, terciles):
        elegido["tercil"] = int(tercil)
    registros, n_sin_texto = construir_registros_muestra(elegidos, ner_por_id, textos, logger)
    top10_benchmark = cargar_top10_archivo(rutas["benchmark_v13"], logger)
    top10_pack = construir_top10_pack(elegidos, top10_benchmark, rutas_c3)
    catalogo = cargar_catalogo_articulos(rutas)
    registros_candidatos = construir_candidatos(elegidos, ner_por_id, top10_pack, catalogo, logger)
    escribir_jsonl(salidas["muestra"], registros)
    escribir_jsonl(salidas["candidatos"], registros_candidatos)
    stats_candidatos = resumen_tamanos([registro["n_candidatos"] for registro in registros_candidatos])
    escribir_guia(
        salidas["guia"],
        texto_guia_gold150(
            len(registros),
            len(elegidos_benchmark),
            len(elegidos_c3),
            stats_candidatos["total"],
            stats_candidatos["max"],
            str(args.dir_salida),
        ),
        logger,
    )
    stats = construir_stats(registros, registros_candidatos, rutas_c3, n_sin_texto, args, salidas)
    escribir_json(salidas["stats"], stats, logger)
    logger.info(
        "candidatos por caso: min=%d max=%d media=%.1f mediana=%.1f p25=%.0f p75=%.0f | total=%d",
        stats_candidatos["min"],
        stats_candidatos["max"],
        stats_candidatos["media"],
        stats_candidatos["mediana"],
        stats_candidatos["p25"],
        stats_candidatos["p75"],
        stats_candidatos["total"],
    )
    logger.info("muestra guardada: %s (%d casos, md5 %s)", salidas["muestra"], len(registros), md5_archivo(salidas["muestra"]))
    logger.info("candidatos guardados: %s (md5 %s)", salidas["candidatos"], md5_archivo(salidas["candidatos"]))
    logger.info("--generar completado en %.1f s", time.perf_counter() - t0)


# ===== SECCION 9: CLI y main =====


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y parsea los argumentos de linea de comandos.

    Args:
        argv: Lista de argumentos (None usa sys.argv).

    Returns:
        Namespace con limite, forzar y dir_salida.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Pack de anotacion C1 (gold estandar de 150 casos): 40 casos del benchmark M7 "
            "restantes + 110 del corpus C3 2022-2026, candidatos NER + top-10 por caso y "
            "guia de anotacion. No anota: el recorrido lo hace el script 16 con "
            "--pack gold_150."
        )
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help="tope de casos para pruebas (ceil(N/2) del benchmark y el resto C3; N=6 -> 3 + 3)",
    )
    parser.add_argument("--forzar", action="store_true", help="permite reescribir salidas existentes")
    parser.add_argument(
        "--dir-salida",
        dest="dir_salida",
        default=CONFIG["dir_salida_default"],
        help="directorio de salida del pack (default resultados/gold_150)",
    )
    args = parser.parse_args(argv)
    if args.limite is not None and args.limite < 1:
        parser.error("--limite debe ser >= 1")
    return args


def main(argv: list[str] | None = None) -> None:
    """Orquesta la generacion del pack C1 (gold estandar de 150 casos).

    Args:
        argv: Argumentos CLI (None usa sys.argv).
    """
    args = parse_args(argv)
    os.chdir(RAIZ)
    logger = configurar_logger()
    np.random.seed(CONFIG["semilla"])
    logger.info(
        "=== INICIO pack gold_150 | limite=%s forzar=%s dir_salida=%s ===",
        args.limite,
        args.forzar,
        args.dir_salida,
    )
    t0 = time.perf_counter()
    modo_generar(args, logger)
    logger.info("=== FIN pack gold_150 en %.1f s ===", time.perf_counter() - t0)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - protocolo: traza completa y salida 1 ante cualquier fallo
        traceback.print_exc()
        sys.exit(1)
