"""16_anotador_gold.py
Fecha: 2026-10-04
Descripcion: Herramienta del mini-gold humano del Sprint B3 (control cruzado del
gate B del ciclo v3). Genera un subconjunto estratificado de 50 casos del
benchmark M7 (los 90 casos de vinculo_cp_benchmark_v2.json, ids verificados
contra v8), un conjunto de articulos candidatos por caso (union de los articulos
de la CP citados por NER y el top-10 de la ruta B2 de v13), la guia de anotacion
y las utilidades para anotar relevancia GRADUADA (2 = fundamento, 1 = mencionado,
0 = no relevante) de forma interactiva y reanudable. Con >= 1 anotador evalua
las rutas guardadas del benchmark contra el gold humano (nDCG@5/@10 graduado,
precision/recall, IC bootstrap 1000 seed 42 y comparaciones pareadas B2 vs v12
vs v8) y calcula el acuerdo inter-anotador (kappa de Cohen/Fleiss).

Esta anotacion es DIAGNOSTICO: no entrena nada, no toca test ni el holdout
sellado y no modifica ningun artefacto previo del proyecto (v2..v13 intactos).

Modos (mutuamente excluyentes):
    --generar            construye muestra_gold.jsonl + candidatos.jsonl + guia
    --anotar             recorrido interactivo reanudable (requiere --anotador)
    --resumen            avance global y por caso de cada anotador
    --exportar-csv       CSV plano (id, articulo, grado, citado_ner, anotador)
    --evaluar            evalua las rutas guardadas contra el gold humano
    --calcular-acuerdo   acuerdo inter-anotador (Cohen por pares y Fleiss)
    --smoke              variante de 10 casos con TODO en /tmp/opencode/gold_mini_smoke/
    --forzar             permite reescribir salidas existentes (no aplica a smoke)

Diseno de la muestra:
    50 casos muestreados de los 90 del benchmark v2 con semilla 42, estratificado
    por periodo (3 periodos, cuotas ~17/17/16) y, dentro de cada periodo, por
    tercil de dificultad (terciles del nDCG@10 por caso de la ruta B2 en v13,
    campo comparacion_b2.metricas_por_caso[contexto_ganador] alineado con la
    muestra; 0 = tercil mas dificil, 2 = mas facil). Si no hubiera metricas por
    caso alineadas se usa un proxy documentado (nDCG@10 binario del top-10 B2
    contra los arts_cp NER). Cuotas con resto mayor (Hamilton) y eleccion
    determinista (rng choice sobre ids ordenados).

Diseno de candidatos:
    Por caso, union de (a) articulos de la CP citados por NER (arts_cp de
    data/entidades_sentencias.parquet) y (b) el top-10 de la ruta B2 de v13
    (top10_por_caso[id]['hibrido_b2_rerank_chunks_top50_b1']). Ordenados por
    numero de articulo (22A entre 22 y 23; Preambulo y Transitorio al final);
    cada candidato guarda numero, snippet de ~200 chars del texto del articulo
    (data/constitucion_articulos.parquet) y el flag citado_ner (referencia
    automatica, NO verdad; se muestra al anotador solo al cerrar el caso).

Evaluacion contra el gold humano:
    Rutas evaluadas: hibrido_b2_rerank_chunks_top50_b1 (v13),
    hibrido_rerank_chunks_top50_b1 (v12), hibrido_rerank_chunks_top50 (v8) y
    cita_exacta (v13; referencia circular NER). nDCG@5/@10 con ganancia
    exponencial 2^g - 1; P@k y R@k binarios con umbral grado >= 1; recall
    graduado @10; IC bootstrap 1000/seed 42 de la media por caso y bootstrap
    PAREADO de las diferencias B2 vs v12 vs v8. Un candidato anotado sin grado
    cuenta como 0 (documentado). Se reportan DOS variantes por ruta:
    'completo' (items del top-10 fuera del pool de candidatos = 0, estandar de
    no juzgados) y 'pool' (ranking restringido a los candidatos anotados,
    sensibilidad ante juicios incompletos); la segunda evita sesgar la
    comparacion cuando una ruta recupera articulos que el anotador no vio.
    Sin >= 2 anotadores no hay acuerdo; con >= 2, kappa de Cohen por pares sobre
    pares (caso, articulo) comunes y Fleiss sobre items calificados por todos.

Reutilizacion (importlib, SIN modificar los originales):
    scripts/21_pack_validacion_m4.py: jsonl (escritura/append/lectura), guardas
    de no-sobrescritura, extracto de texto, cuotas Hamilton, kappa de Cohen y
    Fleiss, normalizacion de anotador, md5 y versiones de entorno.
    scripts/12_evaluar_vinculo_cp_v2.py: lectura de parquet por id con filtros
    pyarrow, bootstrap de la media y bootstrap pareado, nDCG binario (fallback
    de dificultad).

Estado del arte aplicado:
    - Relevancia graduada y nDCG (Jarvelin y Kekalainen, 2002) con ganancia
      exponencial.
    - Juicios humanos agrupados (pooling) con manejo documentado de no juzgados
      (tradicion TREC; Sakai, 2007: listas condensadas para juicios incompletos).
    - IC bootstrap no parametrico (Efron, 1979) de la media y pareado por caso
      (mismo generador y semilla, protocolo del proyecto, ADR-16).
    - Acuerdo inter-anotador: Cohen (1960), Fleiss (1971), Landis y Koch (1977).
    - Muestreo estratificado con cuotas por resto mayor (Hamilton) y semilla 42.

Uso:
    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --generar
    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --anotar --anotador carlos
    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --resumen
    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --exportar-csv
    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --evaluar
    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --calcular-acuerdo
    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --smoke --generar
"""

import argparse
import importlib.util
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ===== SECCION 1: configuracion =====

RAIZ = Path(__file__).resolve().parent.parent

CONFIG = {
    "semilla": 42,
    "n_muestra": 50,
    "n_muestra_smoke": 10,
    "chars_extracto": 1500,
    "chars_snippet_articulo": 200,
    "lote_lectura_corpus": 64,
    "n_remuestreos_bootstrap": 1000,
    "semilla_bootstrap": 42,
    "umbral_relevante": 1,
    "estrategia_candidatos": "hibrido_b2_rerank_chunks_top50_b1",
    "dir_smoke": "/tmp/opencode/gold_mini_smoke",
    "periodos": [
        ("1992-2001", 1992, 2001),
        ("2002-2011", 2002, 2011),
        ("2012-2021", 2012, 2021),
    ],
    "rutas_datos": {
        "benchmark_v2": "resultados/vinculo_cp_benchmark_v2.json",
        "benchmark_v8": "resultados/vinculo_cp_benchmark_v8.json",
        "benchmark_v12": "resultados/vinculo_cp_benchmark_v12.json",
        "benchmark_v13": "resultados/vinculo_cp_benchmark_v13.json",
        "entidades": "data/entidades_sentencias.parquet",
        "articulos": "data/constitucion_articulos.parquet",
        "corpus": "data/corpus_sentencias.parquet",
    },
    "rutas_salida": {
        "muestra": "resultados/gold_mini/muestra_gold.jsonl",
        "candidatos": "resultados/gold_mini/candidatos.jsonl",
        "guia": "resultados/gold_mini/guia_anotacion_gold.md",
        "respuestas": "resultados/gold_mini/respuestas_{anotador}.jsonl",
        "export_csv": "resultados/gold_mini/anotaciones_gold_export.csv",
        "evaluacion": "resultados/gold_mini/evaluacion_gold_mini.json",
        "acuerdo": "resultados/gold_mini/acuerdo_gold_mini.json",
    },
    "rutas_salida_smoke": {
        "muestra": "/tmp/opencode/gold_mini_smoke/muestra_gold.jsonl",
        "candidatos": "/tmp/opencode/gold_mini_smoke/candidatos.jsonl",
        "guia": "/tmp/opencode/gold_mini_smoke/guia_anotacion_gold.md",
        "respuestas": "/tmp/opencode/gold_mini_smoke/respuestas_{anotador}.jsonl",
        "export_csv": "/tmp/opencode/gold_mini_smoke/anotaciones_gold_export.csv",
        "evaluacion": "/tmp/opencode/gold_mini_smoke/evaluacion_gold_mini.json",
        "acuerdo": "/tmp/opencode/gold_mini_smoke/acuerdo_gold_mini.json",
    },
    "rutas_evaluables": [
        {
            "clave": "b2_v13",
            "etiqueta": "B2 v13 (retriever denso de dominio)",
            "archivo": "benchmark_v13",
            "estrategia": "hibrido_b2_rerank_chunks_top50_b1",
        },
        {
            "clave": "v12_b1",
            "etiqueta": "v12 (reranker B1 de dominio)",
            "archivo": "benchmark_v12",
            "estrategia": "hibrido_rerank_chunks_top50_b1",
        },
        {
            "clave": "v8_top50",
            "etiqueta": "v8 (hibrido E5 top-50)",
            "archivo": "benchmark_v8",
            "estrategia": "hibrido_rerank_chunks_top50",
        },
        {
            "clave": "cita_exacta",
            "etiqueta": "cita exacta NER (referencia circular)",
            "archivo": "benchmark_v13",
            "estrategia": "cita_exacta",
        },
    ],
    "pares_comparacion": [
        ("b2_v13", "v12_b1"),
        ("b2_v13", "v8_top50"),
        ("v12_b1", "v8_top50"),
    ],
}

# Premisas de diseno congeladas en las salidas (trazabilidad).
PREMISAS = [
    (
        "Muestra: 50 de los 90 casos del benchmark M7 (vinculo_cp_benchmark_v2.json), "
        "ids verificados contra v8; estratificada por periodo (cuotas ~17/17/16) y por "
        "tercil de dificultad dentro de cada periodo (terciles del nDCG@10 por caso de la "
        "ruta B2 de v13, contexto ganador; 0 = mas dificil). Semilla 42."
    ),
    (
        "Candidatos por caso: union de arts_cp NER (data/entidades_sentencias.parquet) y "
        "el top-10 de la ruta B2 de v13; ordenados por numero de articulo; snippet de "
        "~200 chars del texto de la CP. El flag citado_ner es una referencia automatica "
        "(no verdad) y se muestra al anotador solo al cerrar el caso."
    ),
    (
        "Anotacion humana graduada de relevancia: 2 = fundamento, 1 = mencionado, "
        "0 = no relevante. Enter deja el candidato pendiente; la anotacion se guarda "
        "por candidato (append) y es reanudable. Es diagnostico: no entrena nada y no "
        "toca test ni el holdout sellado."
    ),
    (
        "Evaluacion: rutas guardadas del benchmark (no se re-ejecuta retrieval); nDCG@5/"
        "@10 con ganancia exponencial 2^g - 1; P@k y R@k binarios con umbral grado >= 1; "
        "recall graduado @10; IC bootstrap 1000 remuestreos seed 42 de la media por caso "
        "y bootstrap PAREADO de B2 vs v12 vs v8. Un candidato anotado sin grado cuenta "
        "como 0. Variante 'completo': items del top-10 fuera del pool de candidatos = 0 "
        "(estandar de no juzgados). Variante 'pool': ranking restringido a candidatos "
        "anotados (sensibilidad ante juicios incompletos, Sakai 2007)."
    ),
    (
        "Acuerdo inter-anotador (solo con >= 2 anotadores): kappa de Cohen por pares "
        "sobre los pares (caso, articulo) calificados por ambos y Fleiss sobre items "
        "calificados por todos; interpretacion Landis y Koch (1977). El gold agrupado "
        "(pooled) conserva el ultimo registro por fecha."
    ),
    (
        "El corpus y los parquet se leen SOLO por id con filtros pyarrow; no se carga el "
        "corpus completo ni se modifica ningun archivo de datos o resultado previo."
    ),
]

LOGGER_NOMBRE = "anotador_gold_16"

# Caches de modulos reutilizados y de top10 por archivo (nombre empieza por digito).
_MODULO_21 = None
_MODULO_12 = None
_CACHE_TOP10: dict[str, dict] = {}


# ===== SECCION 2: utilidades y reutilizacion por importlib =====


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


def cargar_modulo_21() -> object:
    """Importa scripts/21_pack_validacion_m4.py por ruta (cacheado).

    Returns:
        Modulo 21 ya ejecutado (jsonl, guardas, cuotas, kappa, md5, versiones).
    """
    global _MODULO_21
    if _MODULO_21 is None:
        _MODULO_21 = _cargar_modulo(RAIZ / "scripts" / "21_pack_validacion_m4.py", "pack_validacion_m4_21")
    return _MODULO_21


def cargar_modulo_12() -> object:
    """Importa scripts/12_evaluar_vinculo_cp_v2.py por ruta (cacheado).

    Returns:
        Modulo 12 ya ejecutado (lectura parquet por id, bootstrap, nDCG binario).
    """
    global _MODULO_12
    if _MODULO_12 is None:
        _MODULO_12 = _cargar_modulo(RAIZ / "scripts" / "12_evaluar_vinculo_cp_v2.py", "evaluar_vinculo_cp_v2_12")
    return _MODULO_12


def configurar_logger() -> logging.Logger:
    """Configura el logger de consola del anotador.

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


def md5_archivo(ruta: str | Path) -> str | None:
    """Calcula el md5 de un archivo (delegado al modulo 21).

    Args:
        ruta: Ruta del archivo.

    Returns:
        Hexdigest md5, o None si el archivo no existe.
    """
    return cargar_modulo_21().md5_archivo(str(ruta))


def versiones_entorno() -> dict:
    """Recolecta versiones de interprete y librerias clave (modulo 21).

    Returns:
        Dict con python, plataforma y versiones de librerias clave.
    """
    return cargar_modulo_21().versiones_entorno()


def escribir_jsonl(ruta: str | Path, registros: list[dict]) -> None:
    """Escribe una lista de registros como JSON Lines (modulo 21).

    Args:
        ruta: Ruta de salida.
        registros: Lista de dicts serializables.
    """
    cargar_modulo_21().escribir_jsonl(ruta, registros)


def escribir_jsonl_append(ruta: str | Path, registro: dict) -> None:
    """Agrega un registro al final de un JSON Lines (modulo 21).

    Args:
        ruta: Ruta del archivo (se crea si no existe).
        registro: Dict serializable.
    """
    cargar_modulo_21().escribir_jsonl_append(ruta, registro)


def cargar_jsonl(ruta: str | Path) -> list[dict]:
    """Carga un JSON Lines validando linea por linea (modulo 21).

    Args:
        ruta: Ruta del archivo.

    Returns:
        Lista de registros.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si alguna linea no es JSON valido o el archivo esta vacio.
    """
    return cargar_modulo_21().cargar_jsonl(ruta)


def verificar_salida_nueva(ruta: str | Path, forzar: bool, smoke: bool, logger: logging.Logger) -> Path:
    """Aborta si la salida final ya existe (guarda del modulo 21).

    Args:
        ruta: Ruta de salida.
        forzar: True permite reescribir (siempre permitido en smoke).
        smoke: True si es corrida smoke (scratch reescribible).
        logger: Logger activo.

    Returns:
        Path de la salida validada.

    Raises:
        FileExistsError: Si la salida existe y no es smoke ni forzar.
    """
    return cargar_modulo_21().verificar_salida_nueva(ruta, forzar, smoke, logger)


def extracto_texto(texto: str, n_chars: int) -> str:
    """Extrae el inicio del texto normalizando finales de linea (modulo 21).

    Args:
        texto: Texto completo.
        n_chars: Maximo de caracteres a conservar.

    Returns:
        Extracto del inicio (sin CR, con strip).
    """
    return cargar_modulo_21().extracto_texto(texto, n_chars)


def normalizar_anotador(nombre: str) -> str:
    """Normaliza el nombre del anotador para usarlo como archivo (modulo 21).

    Args:
        nombre: Nombre crudo.

    Returns:
        Nombre con solo letras, digitos, guion y guion bajo.

    Raises:
        ValueError: Si el nombre queda vacio.
    """
    return cargar_modulo_21().normalizar_anotador(nombre)


def asignar_cuotas(conteos: dict[str, int], objetivo: int, minimo: int) -> dict[str, int]:
    """Cuotas por resto mayor (Hamilton) con minimo por clase (modulo 21).

    Args:
        conteos: Dict clase -> documentos disponibles.
        objetivo: Tamano total deseado.
        minimo: Cuota minima por clase.

    Returns:
        Dict clase -> cuota asignada.
    """
    return cargar_modulo_21().asignar_cuotas(conteos, objetivo, minimo)


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> dict:
    """Kappa de Cohen sobre dos listas alineadas de etiquetas (modulo 21).

    Args:
        labels_a: Etiquetas del primer anotador.
        labels_b: Etiquetas del segundo anotador (misma longitud).

    Returns:
        Dict con n, po, pe, kappa y nivel.
    """
    return cargar_modulo_21().cohen_kappa(labels_a, labels_b)


def fleiss_kappa(items: list[list[str]], categorias: list[str]) -> dict | None:
    """Kappa de Fleiss sobre items con igual numero de evaluadores (modulo 21).

    Args:
        items: Lista de items; cada item es la lista de etiquetas de todos los
            evaluadores (misma longitud en todos los items).
        categorias: Categorias posibles.

    Returns:
        Dict con kappa, p_bar, p_e, n_items y n_evaluadores; o None.
    """
    return cargar_modulo_21().fleiss_kappa(items, categorias)


def formatear_numero(valor: float | None, decimales: int = 3) -> str:
    """Formatea un numero opcional para consola (modulo 21).

    Args:
        valor: Numero o None.
        decimales: Decimales a mostrar.

    Returns:
        Texto del numero o "n/d".
    """
    return cargar_modulo_21().formatear_numero(valor, decimales)


def leer_filas_por_id(ruta: str, columnas: list[str], ids: list[str]) -> pd.DataFrame:
    """Lee solo las filas con los ids dados de un parquet (modulo 12).

    Args:
        ruta: Ruta del parquet.
        columnas: Columnas a leer.
        ids: Lista de ids (strings) a conservar.

    Returns:
        DataFrame con las filas cuyos ids estan en la lista.
    """
    return cargar_modulo_12().leer_filas_por_id(ruta, columnas, ids)


def abrir_corpus(ruta_corpus: str | Path) -> object:
    """Abre el corpus particionado (tipo/ano) sin cargarlo (modulo 21).

    Args:
        ruta_corpus: Ruta del dataset parquet particionado.

    Returns:
        Dataset pyarrow listo para leer por id con filtros.
    """
    return cargar_modulo_21().abrir_corpus(ruta_corpus)


def leer_textos_por_ids(dataset: object, ids: list[str], lote: int, logger: logging.Logger) -> dict[str, str]:
    """Lee SOLO el texto de los ids pedidos por lotes (modulo 21).

    Args:
        dataset: Dataset del corpus ya abierto.
        ids: Ids a leer (strings).
        lote: Tamano de lote del filtro isin.
        logger: Logger activo.

    Returns:
        Dict id -> texto (string vacio si el texto viniera nulo).
    """
    return cargar_modulo_21().leer_textos_por_ids(dataset, ids, lote, logger)


def bootstrap_ic_media(valores: np.ndarray, n_remuestreos: int, semilla: int) -> dict:
    """IC 95% bootstrap del promedio por caso (modulo 12).

    Args:
        valores: Array (n,) con la medida por caso.
        n_remuestreos: Numero de remuestreos bootstrap.
        semilla: Semilla del generador (reproducible).

    Returns:
        Dict {"n", "media", "p2_5", "p97_5"}.
    """
    return cargar_modulo_12().bootstrap_ic_media(valores, n_remuestreos, semilla)


def bootstrap_ic_pareado(nuevo: np.ndarray, referencia: np.ndarray, n_remuestreos: int, semilla: int) -> dict:
    """IC 95% bootstrap pareado de la diferencia nuevo - referencia (modulo 12).

    Args:
        nuevo: Array (n,) con la medida por caso de la ruta nueva.
        referencia: Array (n,) con la medida por caso de la referencia.
        n_remuestreos: Numero de remuestreos bootstrap.
        semilla: Semilla del generador (reproducible).

    Returns:
        Dict {"n", "media_delta", "p2_5", "p97_5", "prob_mejora", "gana"}.
    """
    return cargar_modulo_12().bootstrap_ic_pareado(nuevo, referencia, n_remuestreos, semilla)


def rutas_activas(smoke: bool) -> dict:
    """Construye el mapa de rutas de entrada/salida segun el modo smoke.

    Args:
        smoke: True para usar salidas exclusivamente en /tmp/opencode/gold_mini_smoke.

    Returns:
        Dict combinado de rutas de datos y de salidas.
    """
    rutas = {}
    rutas.update(CONFIG["rutas_datos"])
    rutas.update(CONFIG["rutas_salida_smoke"] if smoke else CONFIG["rutas_salida"])
    return rutas


def ahora_iso() -> str:
    """Devuelve la marca de tiempo actual en ISO (segundos).

    Returns:
        Timestamp ISO-8601 sin zona.
    """
    return pd.Timestamp.now().isoformat(timespec="seconds")


def clave_id(id_caso: str) -> int | str:
    """Clave de orden determinista para ids (numericos primero).

    Args:
        id_caso: Id del caso (string).

    Returns:
        Entero si el id es numerico, o el string original.
    """
    texto = str(id_caso)
    return int(texto) if texto.isdigit() else texto


# ===== SECCION 3: benchmark, dificultad y muestreo =====


def cargar_benchmark(ruta: str | Path) -> dict:
    """Carga un JSON del benchmark M7.

    Args:
        ruta: Ruta del JSON.

    Returns:
        Dict completo del benchmark.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el JSON no tiene el campo 'muestra'.
    """
    archivo = Path(ruta)
    if not archivo.exists():
        raise FileNotFoundError(f"no existe el benchmark: {archivo}")
    with open(archivo, encoding="utf-8") as manejador:
        datos = json.load(manejador)
    if "muestra" not in datos:
        raise ValueError(f"benchmark sin campo 'muestra': {archivo}")
    return datos


def validar_ids_benchmark(muestra_v2: list[dict], muestra_v8: list[dict], logger: logging.Logger) -> None:
    """Verifica que los ids de v2 y v8 coincidan (integridad de la muestra).

    Args:
        muestra_v2: Lista de casos de v2 (referencia).
        muestra_v8: Lista de casos de v8 (control).
        logger: Logger activo.

    Raises:
        RuntimeError: Si los conjuntos de ids no son identicos.
    """
    ids_v2 = {str(caso["id"]) for caso in muestra_v2}
    ids_v8 = {str(caso["id"]) for caso in muestra_v8}
    if ids_v2 != ids_v8:
        raise RuntimeError(
            f"ids del benchmark v2 != v8 (v2-v8={sorted(ids_v2 - ids_v8)[:5]}, "
            f"v8-v2={sorted(ids_v8 - ids_v2)[:5]})"
        )
    logger.info("ids del benchmark verificados contra v8: %d/%d coinciden", len(ids_v2), len(ids_v8))


def asignar_periodo(anio: int) -> str:
    """Asigna un anio a su periodo cronologico de CONFIG.

    Args:
        anio: Anio de la sentencia.

    Returns:
        Nombre del periodo (p. ej. "1992-2001") o "desconocido".
    """
    for nombre, inicio, fin in CONFIG["periodos"]:
        if inicio <= anio <= fin:
            return nombre
    return "desconocido"


def cargar_dificultad_b2(rutas: dict, muestra: list[dict], logger: logging.Logger) -> tuple[list[float], str, str]:
    """Carga el nDCG@10 por caso de la ruta B2 (v13) como proxy de dificultad.

    Fuente primaria: comparacion_b2.metricas_por_caso[contexto_ganador]
    ['hibrido_b2_rerank_chunks_top50_b1']['ndcg_at_10'], alineado con la muestra
    de v13 (que coincide con v2). Si no estuviera disponible o alineado, se usa
    un proxy documentado: nDCG@10 binario del top-10 B2 contra los arts_cp NER.

    Args:
        rutas: Dict de rutas activas.
        muestra: Muestra de v2 (90 casos, orden de referencia).
        logger: Logger activo.

    Returns:
        Tupla (valores por caso en el orden de la muestra, fuente, contexto).
    """
    v13 = cargar_benchmark(rutas["benchmark_v13"])
    ids_v13 = [str(caso["id"]) for caso in v13.get("muestra", [])]
    ids_muestra = [str(caso["id"]) for caso in muestra]
    contexto = str(v13.get("comparacion_b2", {}).get("contexto_ganador", "con_hubs"))
    serie = (
        v13.get("comparacion_b2", {})
        .get("metricas_por_caso", {})
        .get(contexto, {})
        .get(CONFIG["estrategia_candidatos"], {})
        .get("ndcg_at_10")
    )
    if isinstance(serie, list) and len(serie) == len(ids_muestra) and ids_v13 == ids_muestra:
        logger.info(
            "dificultad B2: nDCG@10 por caso de comparacion_b2 (%s), %d valores alineados con la muestra",
            contexto,
            len(serie),
        )
        return [float(valor) for valor in serie], f"comparacion_b2.metricas_por_caso.{contexto}.ndcg_at_10 (v13)", contexto
    logger.warning("v13 sin metricas por caso alineadas; se usa el proxy binario B2 vs NER")
    top10 = cargar_top10_archivo(rutas["benchmark_v13"], logger)
    modulo_12 = cargar_modulo_12()
    valores = []
    for caso in muestra:
        id_caso = str(caso["id"])
        ranking = top10.get(id_caso, {}).get(CONFIG["estrategia_candidatos"], [])
        relevantes = [normalizar_articulo(articulo) for articulo in caso["arts_cp"]]
        valores.append(float(modulo_12.ndcg_at_k(relevantes, ranking, 10)))
    return valores, "proxy_binario: nDCG@10 (ganancia binaria) del top-10 B2 vs NER (v13)", contexto


def asignar_terciles(valores: list[float], ids: list[str]) -> list[int]:
    """Asigna cada caso a un tercil de dificultad (0 = mas dificil).

    Los terciles parten la muestra ordenada por (valor, id) en tres grupos lo mas
    iguales posible: con n multiplo de 3 son de n/3; en otro caso los primeros
    reciben un caso extra.

    Args:
        valores: nDCG@10 por caso (menor = mas dificil).
        ids: Ids alineados con valores (desempate determinista).

    Returns:
        Lista de terciles en {0, 1, 2} en el orden de entrada.
    """
    n = len(valores)
    orden = sorted(range(n), key=lambda indice: (float(valores[indice]), ids[indice]))
    terciles = [0] * n
    for posicion, indice in enumerate(orden):
        terciles[indice] = min(2, posicion * 3 // n) if n > 0 else 0
    return terciles


def muestrear_estratificado(
    muestra: list[dict], terciles: list[int], objetivo: int, semilla: int, logger: logging.Logger
) -> list[dict]:
    """Muestrea casos por periodo y tercil de dificultad con cuotas Hamilton.

    Args:
        muestra: Muestra de v2 (90 casos).
        terciles: Tercil de dificultad por caso (orden de la muestra).
        objetivo: Tamano de la submuestra (50 real, 10 smoke).
        semilla: Semilla del generador (42).
        logger: Logger activo.

    Returns:
        Lista de casos elegidos con indice, id, tipo, ano, periodo y tercil, en
        orden barajado determinista.

    Raises:
        RuntimeError: Si el muestreo no alcanza el objetivo (datos insuficientes).
    """
    universo = []
    for posicion, caso in enumerate(muestra):
        universo.append(
            {
                "id": str(caso["id"]),
                "tipo": str(caso["tipo"]),
                "ano": int(caso["ano"]),
                "periodo": asignar_periodo(int(caso["ano"])),
                "tercil": int(terciles[posicion]),
            }
        )
    conteo_periodo = Counter(registro["periodo"] for registro in universo)
    cuotas_periodo = asignar_cuotas(conteo_periodo, objetivo, 0)
    logger.info("cuotas por periodo (objetivo=%d): %s", objetivo, cuotas_periodo)
    rng = np.random.default_rng(semilla)
    elegidos: list[dict] = []
    for periodo in sorted(cuotas_periodo):
        sub = [registro for registro in universo if registro["periodo"] == periodo]
        conteo_tercil = Counter(str(registro["tercil"]) for registro in sub)
        cuotas_tercil = asignar_cuotas(conteo_tercil, cuotas_periodo[periodo], 0)
        logger.info(
            "  %s: cuota=%d | por tercil (0=dificil) %s",
            periodo,
            cuotas_periodo[periodo],
            cuotas_tercil,
        )
        for tercil in sorted(cuotas_tercil):
            cuota = int(cuotas_tercil[tercil])
            if cuota == 0:
                continue
            candidatos = sorted(
                [registro for registro in sub if str(registro["tercil"]) == tercil],
                key=lambda registro: registro["id"],
            )
            posiciones = np.sort(rng.choice(len(candidatos), size=cuota, replace=False))
            for posicion in posiciones:
                elegidos.append(dict(candidatos[int(posicion)]))
    if len(elegidos) != objetivo:
        raise RuntimeError(f"muestreo incompleto: {len(elegidos)} de {objetivo} casos")
    orden = rng.permutation(len(elegidos))
    elegidos = [elegidos[int(indice)] for indice in orden]
    for indice, registro in enumerate(elegidos, 1):
        registro["indice"] = indice
    tabla = Counter((registro["periodo"], registro["tercil"]) for registro in elegidos)
    logger.info(
        "muestra elegida: %d casos | crosstab (periodo, tercil) = %s",
        len(elegidos),
        {f"{periodo}|T{tercil}": conteo for (periodo, tercil), conteo in sorted(tabla.items())},
    )
    return elegidos


def cargar_ner_por_id(rutas: dict, ids: list[str], logger: logging.Logger) -> dict[str, list[str]]:
    """Carga los articulos CP citados por NER de los ids pedidos.

    Lee SOLO los ids pedidos de data/entidades_sentencias.parquet con filtros
    pyarrow (modulo 12) y normaliza los numeros de articulo.

    Args:
        rutas: Dict de rutas activas.
        ids: Ids de la submuestra.
        logger: Logger activo.

    Returns:
        Dict id -> lista de articulos normalizados (puede faltar un id).
    """
    tabla = leer_filas_por_id(rutas["entidades"], ["id", "arts_cp"], ids)
    ner: dict[str, list[str]] = {}
    for fila in tabla.to_dict("records"):
        try:
            articulos = json.loads(fila["arts_cp"] or "[]")
        except (TypeError, ValueError):
            articulos = []
        ner[str(fila["id"])] = [normalizar_articulo(articulo) for articulo in articulos]
    faltantes = [id_caso for id_caso in ids if id_caso not in ner]
    if faltantes:
        logger.warning("sin arts_cp en entidades para %d ids: %s", len(faltantes), faltantes[:10])
    return ner


def cargar_textos_corpus(rutas: dict, ids: list[str], logger: logging.Logger) -> dict[str, str]:
    """Lee el texto completo de los ids pedidos del corpus particionado.

    Args:
        rutas: Dict de rutas activas.
        ids: Ids de la submuestra.
        logger: Logger activo.

    Returns:
        Dict id -> texto (vacio si el corpus no lo tiene).
    """
    dataset = abrir_corpus(rutas["corpus"])
    return leer_textos_por_ids(dataset, ids, CONFIG["lote_lectura_corpus"], logger)


def construir_registros_muestra(
    elegidos: list[dict],
    dificultad_por_id: dict[str, float],
    ner_por_id: dict[str, list[str]],
    textos: dict[str, str],
    fuente_dificultad: str,
    logger: logging.Logger,
) -> list[dict]:
    """Construye los registros de muestra_gold.jsonl con extracto del encabezado.

    Args:
        elegidos: Casos elegidos por el muestreo (con indice, periodo, tercil).
        dificultad_por_id: nDCG@10 B2 por id (trazabilidad del tercil).
        ner_por_id: Articulos NER por id.
        textos: Texto completo del corpus por id.
        fuente_dificultad: Descripcion de la fuente del proxy de dificultad.
        logger: Logger activo.

    Returns:
        Lista de registros para el pack de anotacion.
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
                "tipo": str(elegido["tipo"]),
                "ano": int(elegido["ano"]),
                "periodo": str(elegido["periodo"]),
                "tercil_dificultad": int(elegido["tercil"]),
                "ndcg_b2_caso": round(float(dificultad_por_id.get(id_caso, 0.0)), 6),
                "fuente_dificultad": fuente_dificultad,
                "n_citas_ner": len(ner_por_id.get(id_caso, [])),
                "extracto": extracto_texto(texto, CONFIG["chars_extracto"]),
            }
        )
    if sin_texto:
        logger.warning("%d casos de la muestra sin texto en el corpus", sin_texto)
    return registros


def cargar_catalogo_articulos(rutas: dict) -> dict[str, str]:
    """Carga el catalogo completo de articulos de la CP (497 filas, liviano).

    Args:
        rutas: Dict de rutas activas.

    Returns:
        Dict numero normalizado -> texto del articulo.
    """
    filas = pq.read_table(rutas["articulos"], columns=["numero", "texto"]).to_pylist()
    return {normalizar_articulo(fila["numero"]): (fila["texto"] or "") for fila in filas}


def normalizar_articulo(valor: object) -> str:
    """Normaliza un numero de articulo de la CP.

    Acepta "86", "22A", "Transitorio 5" y "Preambulo"; el resto se conserva tal
    cual (strip) para no perder ids raros.

    Args:
        valor: Numero crudo (string o numerico).

    Returns:
        Numero normalizado.
    """
    texto = str(valor).strip()
    coincidencia = re.match(r"^(\d+)([A-Za-z]*)$", texto)
    if coincidencia:
        return f"{int(coincidencia.group(1))}{coincidencia.group(2).upper()}"
    coincidencia = re.match(r"(?i)^transitorio\s+(\d+)$", texto)
    if coincidencia:
        return f"Transitorio {int(coincidencia.group(1))}"
    if texto.lower() in ("preambulo", "preámbulo"):
        return "Preambulo"
    return texto


def clave_orden_articulo(articulo: str) -> tuple:
    """Clave de orden natural por numero de articulo.

    Args:
        articulo: Numero normalizado.

    Returns:
        Tupla (grupo, numero, sufijo, texto) con numericos primero (22A entre 22
        y 23), luego Preambulo, luego Transitorio N y finalmente ids raros.
    """
    coincidencia = re.match(r"^(\d+)([A-Z]*)$", articulo)
    if coincidencia:
        return (0, int(coincidencia.group(1)), coincidencia.group(2), "")
    if articulo == "Preambulo":
        return (1, 0, "", "")
    coincidencia = re.match(r"^Transitorio (\d+)$", articulo)
    if coincidencia:
        return (2, int(coincidencia.group(1)), "", "")
    return (3, 0, "", articulo)


def construir_candidatos(
    elegidos: list[dict],
    ner_por_id: dict[str, list[str]],
    top10_v13: dict,
    catalogo: dict[str, str],
    logger: logging.Logger,
) -> list[dict]:
    """Construye candidatos.jsonl: union NER + top-10 B2 por caso.

    Args:
        elegidos: Casos elegidos por el muestreo.
        ner_por_id: Articulos NER por id.
        top10_v13: top10_por_caso de v13 (id -> estrategia -> articulos).
        catalogo: Texto de cada articulo de la CP.
        logger: Logger activo.

    Returns:
        Lista de registros por caso con la lista de candidatos ordenada.
    """
    registros = []
    sin_catalogo = 0
    for elegido in elegidos:
        id_caso = elegido["id"]
        ner = set(ner_por_id.get(id_caso, []))
        estrategias = top10_v13.get(id_caso, {})
        top_b2 = [normalizar_articulo(item) for item in estrategias.get(CONFIG["estrategia_candidatos"], [])]
        union = sorted(ner | set(top_b2), key=clave_orden_articulo)
        candidatos = []
        for indice, articulo in enumerate(union, 1):
            texto = catalogo.get(articulo, "")
            if not texto:
                sin_catalogo += 1
            candidatos.append(
                {
                    "indice": indice,
                    "articulo": articulo,
                    "snippet": extracto_texto(texto, CONFIG["chars_snippet_articulo"]),
                    "citado_ner": articulo in ner,
                }
            )
        registros.append(
            {
                "indice": int(elegido["indice"]),
                "id": id_caso,
                "periodo": str(elegido["periodo"]),
                "tercil_dificultad": int(elegido["tercil"]),
                "n_citas_ner": len(ner),
                "n_candidatos": len(candidatos),
                "candidatos": candidatos,
            }
        )
    if sin_catalogo:
        logger.warning("%d candidatos sin texto en el catalogo de articulos", sin_catalogo)
    return registros


def resumen_tamanos(valores: list[int]) -> dict:
    """Resume la distribucion de tamanos de los conjuntos candidatos.

    Args:
        valores: Lista de tamanos (n candidatos por caso).

    Returns:
        Dict con n, min, max, media, mediana y percentiles 25/75.
    """
    arreglo = np.asarray(valores, dtype=np.float64)
    return {
        "n": int(arreglo.size),
        "min": int(arreglo.min()) if arreglo.size else 0,
        "max": int(arreglo.max()) if arreglo.size else 0,
        "media": float(arreglo.mean()) if arreglo.size else 0.0,
        "mediana": float(np.median(arreglo)) if arreglo.size else 0.0,
        "p25": float(np.percentile(arreglo, 25)) if arreglo.size else 0.0,
        "p75": float(np.percentile(arreglo, 75)) if arreglo.size else 0.0,
        "total": int(arreglo.sum()) if arreglo.size else 0,
    }


def modo_generar(args: argparse.Namespace, logger: logging.Logger, rutas: dict) -> None:
    """Construye la submuestra estratificada, los candidatos y la guia.

    Args:
        args: Argumentos CLI (smoke, forzar).
        logger: Logger activo.
        rutas: Dict de rutas activas.

    Raises:
        FileExistsError: Si una salida existe y no hay --forzar (salvo smoke).
        RuntimeError: Si el benchmark no es integro o el muestreo falla.
    """
    t0 = time.perf_counter()
    destino_muestra = verificar_salida_nueva(rutas["muestra"], args.forzar, args.smoke, logger)
    destino_candidatos = verificar_salida_nueva(rutas["candidatos"], args.forzar, args.smoke, logger)
    destino_guia = verificar_salida_nueva(rutas["guia"], args.forzar, args.smoke, logger)
    benchmark = cargar_benchmark(rutas["benchmark_v2"])
    muestra = benchmark["muestra"]
    v8 = cargar_benchmark(rutas["benchmark_v8"])
    validar_ids_benchmark(muestra, v8["muestra"], logger)
    dificultad, fuente_dificultad, contexto = cargar_dificultad_b2(rutas, muestra, logger)
    logger.info("proxy de dificultad: %s (contexto %s)", fuente_dificultad, contexto)
    terciles = asignar_terciles(dificultad, [str(caso["id"]) for caso in muestra])
    objetivo = CONFIG["n_muestra_smoke"] if args.smoke else CONFIG["n_muestra"]
    elegidos = muestrear_estratificado(muestra, terciles, objetivo, CONFIG["semilla"], logger)
    ids = [registro["id"] for registro in elegidos]
    ner_entidades = cargar_ner_por_id(rutas, ids, logger)
    ner_benchmark = {
        str(caso["id"]): [normalizar_articulo(articulo) for articulo in caso["arts_cp"]] for caso in muestra
    }
    discrepancias = [
        id_caso
        for id_caso in ids
        if id_caso in ner_entidades and sorted(ner_entidades[id_caso]) != sorted(ner_benchmark.get(id_caso, []))
    ]
    if discrepancias:
        logger.warning("NER de entidades != arts_cp del benchmark en %d casos: %s", len(discrepancias), discrepancias[:5])
    ner_por_id = {id_caso: ner_entidades.get(id_caso) or ner_benchmark.get(id_caso, []) for id_caso in ids}
    textos = cargar_textos_corpus(rutas, ids, logger)
    dificultad_por_id = {str(caso["id"]): float(valor) for caso, valor in zip(muestra, dificultad)}
    registros = construir_registros_muestra(
        elegidos, dificultad_por_id, ner_por_id, textos, fuente_dificultad, logger
    )
    top10_v13 = cargar_top10_archivo(rutas["benchmark_v13"], logger)
    catalogo = cargar_catalogo_articulos(rutas)
    registros_candidatos = construir_candidatos(elegidos, ner_por_id, top10_v13, catalogo, logger)
    escribir_jsonl(destino_muestra, registros)
    escribir_jsonl(destino_candidatos, registros_candidatos)
    stats = resumen_tamanos([registro["n_candidatos"] for registro in registros_candidatos])
    escribir_guia(
        destino_guia,
        texto_guia(len(registros), destino_muestra.name, args.smoke, stats["total"], stats["max"]),
        logger,
    )
    logger.info(
        "candidatos por caso: min=%d max=%d media=%.1f mediana=%.1f p25=%.0f p75=%.0f | total=%d",
        stats["min"],
        stats["max"],
        stats["media"],
        stats["mediana"],
        stats["p25"],
        stats["p75"],
        stats["total"],
    )
    logger.info("muestra guardada: %s (%d casos, md5 %s)", destino_muestra, len(registros), md5_archivo(destino_muestra))
    logger.info("candidatos guardados: %s (md5 %s)", destino_candidatos, md5_archivo(destino_candidatos))
    logger.info("--generar completado en %.1f s", time.perf_counter() - t0)


# ===== SECCION 4: guia de anotacion =====


def texto_guia(
    n_muestra: int,
    nombre_muestra: str,
    smoke: bool,
    total_candidatos: int,
    max_candidatos: int,
) -> str:
    """Construye la guia de anotacion en Markdown (espanol, sin emojis).

    Args:
        n_muestra: Tamano de la muestra.
        nombre_muestra: Nombre del archivo jsonl de la muestra.
        smoke: True si la guia corresponde a la corrida smoke.
        total_candidatos: Total de decisiones de grado de la muestra.
        max_candidatos: Maximo de candidatos por caso en la muestra.

    Returns:
        Contenido completo de la guia.
    """
    fecha = pd.Timestamp.now().strftime("%Y-%m-%d")
    nota_smoke = (
        "\nNOTA: esta guia se genero en modo smoke (10 casos) y todas sus salidas "
        "viven en /tmp/opencode/gold_mini_smoke/.\n"
        if smoke
        else ""
    )
    return f"""# Guia de anotacion - Mini-gold humano del vinculo sentencia -> articulos CP

Fecha de generacion: {fecha}
Script: scripts/16_anotador_gold.py (semilla 42)
Muestra: {nombre_muestra} ({n_muestra} casos del benchmark M7)
{nota_smoke}
## 1. Objetivo y alcance

Medir con juicio humano la relevancia de los articulos candidatos de cada caso
para el vinculo sentencia -> articulos de la Constitucion (CP). Este mini-gold es
el control cruzado del gate B del Sprint B3: contrasta las rutas automaticas del
benchmark (gold debil NER) con una lectura humana.

Es un diagnostico: NO entrena nada, no toca el test ni el holdout sellado y no
modifica ningun resultado previo del proyecto.

## 2. Que se mide: relevancia graduada

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

## 3. Candidatos y el flag citado_ner

La lista de candidatos de cada caso es la union de (a) los articulos citados
detectados por NER y (b) el top-10 de la ruta automatica B2 del benchmark. Por
eso la lista contiene articulos posiblemente no relevantes: su tarea es
precisamente graduarlos.

El flag citado_ner (que aparece al cerrar el caso) es una REFERENCIA automatica,
no una verdad: indica que el extractor de citas marco ese articulo. No lo use
para decidir el grado; se imprime al final para no sesgar la lectura.

## 4. Regla de oro

1. Lea el encabezado de la sentencia que acompana al caso.
2. Para cada articulo, lea el snippet; si necesita mas contexto, abra el texto
   completo del articulo en data/constitucion_articulos.json o el documento
   fuente del caso.
3. Decida el grado SOLO con la sentencia y el articulo.
4. No consulte rankings, scores, resultados de modelos ni anotaciones de otros
   anotadores. Tampoco consulte los resultados de --evaluar.

## 5. Tiempo estimado

La muestra tiene {n_muestra} casos con {total_candidatos} decisiones de grado en
total (el caso mas grande tiene {max_candidatos} candidatos). El plan estima 1,5
a 3 horas; a un ritmo de 10 a 20 segundos por candidato el rango tipico es de 2
a 4 horas. Se puede anotar en tandas: el progreso se guarda despues de cada
candidato.

## 6. Reanudacion y comandos

    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --resumen
    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --anotar --anotador carlos
    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --exportar-csv
    /home/carlos/venv/bin/python scripts/16_anotador_gold.py --evaluar

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

## 7. Duplicados

La lista de candidatos ya esta deduplicada por numero de articulo y ordenada.
Articulos como 22 y 22A son distintos: califique cada uno. Si dos casos parecen
la misma sentencia, anote ambos: cada caso cuenta.

## 8. Acuerdo y evaluacion

Un segundo anotador puede correr --anotar con otro nombre (por ejemplo
--anotador ana) sobre la misma muestra; despues --calcular-acuerdo reporta
kappa de Cohen por pares y Fleiss. --evaluar calcula nDCG@5/@10 graduado,
precision y recall de las rutas guardadas contra este gold. En la evaluacion,
un candidato anotado sin grado cuenta como 0 (documentado); por eso conviene
completar los casos antes de evaluar.

## 9. Recordatorio

Esta anotacion es humana y es diagnostico. No es una etiqueta automatica y no
se usa para entrenar; su unico fin es auditar las rutas del benchmark con ojos
humanos.
"""


def escribir_guia(destino: str | Path, contenido: str, logger: logging.Logger) -> None:
    """Escribe la guia de anotacion en Markdown.

    Args:
        destino: Ruta de salida.
        contenido: Texto completo de la guia.
        logger: Logger activo.
    """
    archivo = Path(destino)
    archivo.parent.mkdir(parents=True, exist_ok=True)
    with open(archivo, "w", encoding="utf-8") as manejador:
        manejador.write(contenido)
    logger.info("guia guardada: %s (%d lineas)", archivo, contenido.count("\n") + 1)


# ===== SECCION 5: anotacion interactiva, resumen y exportacion =====


def cargar_grados(ruta: str | Path, logger: logging.Logger | None = None) -> dict[tuple[str, str], dict]:
    """Carga las respuestas de un anotador (clave (id, articulo), ultimo gana).

    Args:
        ruta: Ruta del jsonl de respuestas (si no existe, dict vacio).
        logger: Logger activo (opcional) para avisar registros invalidos.

    Returns:
        Dict (id, articulo) -> registro valido.

    Raises:
        ValueError: Si alguna linea no es JSON valido.
    """
    grados: dict[tuple[str, str], dict] = {}
    archivo = Path(ruta)
    if not archivo.exists():
        return grados
    invalidos = 0
    with open(archivo, encoding="utf-8") as manejador:
        for numero, linea in enumerate(manejador, 1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                registro = json.loads(linea)
            except json.JSONDecodeError as exc:
                raise ValueError(f"jsonl invalido en {archivo}:{numero}") from exc
            grado = registro.get("nota_grado")
            if grado not in (0, 1, 2) or "id" not in registro or "articulo" not in registro:
                invalidos += 1
                continue
            grados[(str(registro["id"]), str(registro["articulo"]))] = registro
    if invalidos and logger is not None:
        logger.warning("%s: %d registros invalidos ignorados", archivo, invalidos)
    return grados


def cargar_respuestas_todas(rutas: dict, logger: logging.Logger) -> dict[str, dict[tuple[str, str], dict]]:
    """Carga todas las respuestas_*.jsonl del directorio de la muestra.

    Args:
        rutas: Dict de rutas activas.
        logger: Logger activo.

    Returns:
        Dict anotador -> ((id, articulo) -> registro), ordenado por nombre.
    """
    directorio = Path(rutas["muestra"]).parent
    prefijo = "respuestas_"
    resultado: dict[str, dict[tuple[str, str], dict]] = {}
    for archivo in sorted(directorio.glob(f"{prefijo}*.jsonl")):
        anotador = archivo.stem[len(prefijo) :]
        if anotador:
            resultado[anotador] = cargar_grados(archivo, logger)
    return resultado


def caso_completo(caso: dict, candidatos_por_id: dict[str, list[dict]], grados: dict[tuple[str, str], dict]) -> bool:
    """Indica si todos los candidatos de un caso tienen grado.

    Args:
        caso: Registro de la muestra.
        candidatos_por_id: Candidatos por id de caso.
        grados: Grados actuales (clave (id, articulo)).

    Returns:
        True si el caso tiene candidatos y todos estan calificados.
    """
    lista = candidatos_por_id.get(str(caso["id"]), [])
    return bool(lista) and all((str(caso["id"]), candidato["articulo"]) in grados for candidato in lista)


def interpretar_grado(entrada: str) -> tuple[str, int | list[int | None] | None, str | None]:
    """Interpreta la respuesta de grado del anotador (un grado o un lote).

    Un lote es una secuencia de grados separados por espacios o comas que se
    aplica, en orden, a los candidatos pendientes tal como aparecen en
    pantalla; dentro de un lote, "-" deja el candidato pendiente sin calificar.

    Args:
        entrada: Texto crudo del usuario.

    Returns:
        Tupla (accion, valor, mensaje) con accion en grado | lote | pendiente |
        saltar | salir | ayuda | reintentar. Para accion lote, valor es la lista
        de grados (int) y pendientes (None) en el orden escrito.
    """
    texto = " ".join(entrada.replace(",", " ").strip().lower().split())
    if texto == "":
        return "pendiente", None, None
    if texto in ("2", "1", "0"):
        return "grado", int(texto), None
    if texto == "-":
        return "pendiente", None, None
    if texto in ("s", "saltar", "skip"):
        return "saltar", None, None
    if texto in ("q", "salir", "quit"):
        return "salir", None, None
    if texto in ("?", "ayuda", "help", "menu"):
        return "ayuda", None, None
    tokens = texto.split()
    if len(tokens) > 1:
        invalidos = [token for token in tokens if token not in ("2", "1", "0", "-")]
        if invalidos:
            return (
                "reintentar",
                None,
                (
                    f"lote no valido: {', '.join(invalidos)} (use grados 2/1/0 y '-' para "
                    "pendiente, separados por espacios)"
                ),
            )
        return "lote", [None if token == "-" else int(token) for token in tokens], None
    return "reintentar", None, f"entrada no reconocida: {entrada!r} (use 2, 1, 0, -, Enter, s, q o ?)"


def imprimir_ayuda_grado() -> None:
    """Imprime la ayuda de la escala graduada y de los comandos."""
    print("Escala de relevancia graduada:")
    print("  2  fundamento  (base normativa directa de lo decidido o de la ratio)")
    print("  1  mencionado  (aparece o se relaciona tangencialmente, sin ser base)")
    print("  0  no relevante (sin relacion con lo que la sentencia decide)")
    print("  Enter = pendiente | s = saltar caso | q = guardar y salir | ? = ayuda")
    print("  lote = varios grados separados por espacios (p. ej. 0 2 0 0 - 1); se aplican")
    print("         a los candidatos pendientes en el orden en que aparecen en pantalla;")
    print("         '-' deja ese candidato pendiente")


def imprimir_caso(
    caso: dict,
    candidatos: list[dict],
    grados: dict[tuple[str, str], dict],
    indice: int,
    total: int,
) -> None:
    """Imprime un caso para anotar: metadata, encabezado y candidatos.

    Args:
        caso: Registro de la muestra.
        candidatos: Lista de candidatos del caso.
        grados: Grados actuales (los ya calificados se marcan).
        indice: Posicion 1..N de la sesion.
        total: Total de casos de la sesion.
    """
    print()
    print(f"===== CASO {indice}/{total} | id {caso['id']} =====")
    print(
        f"tipo: {caso['tipo']} | ano: {caso['ano']} | periodo: {caso['periodo']} | "
        f"tercil dificultad: {caso['tercil_dificultad']} (0=dificil, 2=facil) | "
        f"citas NER: {caso['n_citas_ner']}"
    )
    extracto = caso.get("extracto", "")
    print(f"--- encabezado de la sentencia ({len(extracto)} chars) ---")
    print(extracto if extracto else "(sin extracto disponible en el pack)")
    print("--- fin del encabezado ---")
    print(f"Candidatos ({len(candidatos)}; orden por numero de articulo):")
    for candidato in candidatos:
        registro = grados.get((str(caso["id"]), candidato["articulo"]))
        marca = f"  [ya: {registro['nota_grado']}]" if registro else ""
        print(f"  [{candidato['indice']:2d}/{len(candidatos)}] art. {candidato['articulo']}{marca}")
        snippet = candidato.get("snippet", "")
        print(f"      {snippet}" if snippet else "      (sin texto en el catalogo)")


def imprimir_resumen_caso(caso: dict, candidatos: list[dict], grados: dict[tuple[str, str], dict]) -> None:
    """Imprime el cierre de un caso: avance y referencia NER (no verdad).

    Args:
        caso: Registro de la muestra.
        candidatos: Lista de candidatos del caso.
        grados: Grados actuales.
    """
    calificados = [
        grados[(str(caso["id"]), candidato["articulo"])]
        for candidato in candidatos
        if (str(caso["id"]), candidato["articulo"]) in grados
    ]
    distribucion = Counter(int(registro["nota_grado"]) for registro in calificados)
    citados = [candidato["articulo"] for candidato in candidatos if candidato["citado_ner"]]
    print(
        f"  cierre caso {caso['id']}: grados {len(calificados)}/{len(candidatos)} | "
        f"distribucion {dict(sorted(distribucion.items()))}"
    )
    print(
        "  referencia (NO verdad, solo para verificacion posterior): citados por NER: "
        + (", ".join(citados) if citados else "ninguno")
    )


def registrar_grado(
    ruta_respuestas: Path,
    grados: dict[tuple[str, str], dict],
    anotador: str,
    caso_id: str,
    candidato: dict,
    nota: int,
    t_inicio: float,
    logger: logging.Logger,
    total_candidatos: int,
) -> None:
    """Escribe un grado nuevo en el JSONL de respuestas y lo agrega a memoria.

    Args:
        ruta_respuestas: Ruta del JSONL del anotador.
        grados: Grados actuales (se actualiza en sitio).
        anotador: Nombre normalizado del anotador.
        caso_id: Id del caso.
        candidato: Candidato calificado (dict del pack).
        nota: Grado 0, 1 o 2.
        t_inicio: Marca de tiempo del inicio de la respuesta.
        logger: Logger activo.
        total_candidatos: Total de candidatos de la muestra (para el log).
    """
    registro = {
        "anotador": anotador,
        "id": caso_id,
        "indice_candidato": int(candidato["indice"]),
        "articulo": candidato["articulo"],
        "nota_grado": int(nota),
        "fecha": ahora_iso(),
        "tiempo_seg": round(time.perf_counter() - t_inicio, 1),
    }
    escribir_jsonl_append(ruta_respuestas, registro)
    grados[(caso_id, candidato["articulo"])] = registro
    logger.info(
        "[%d/%d] caso %s | art. %s -> %d | %.1f s",
        len(grados),
        total_candidatos,
        caso_id,
        candidato["articulo"],
        int(nota),
        time.perf_counter() - t_inicio,
    )


def modo_anotar(nombre: str, logger: logging.Logger, rutas: dict) -> None:
    """Recorrido interactivo reanudable de anotacion graduada (grado individual o lote).

    Args:
        nombre: Nombre del anotador (se normaliza para el archivo).
        logger: Logger activo.
        rutas: Dict de rutas activas.

    Raises:
        FileNotFoundError: Si no existe la muestra (hay que ejecutar --generar).
    """
    t0 = time.perf_counter()
    anotador = normalizar_anotador(nombre)
    muestra = cargar_jsonl(rutas["muestra"])
    candidatos_por_id = {str(registro["id"]): registro["candidatos"] for registro in cargar_jsonl(rutas["candidatos"])}
    ruta_respuestas = Path(rutas["respuestas"].format(anotador=anotador))
    grados = cargar_grados(ruta_respuestas, logger)
    pendientes = [caso for caso in muestra if not caso_completo(caso, candidatos_por_id, grados)]
    logger.info(
        "anotador=%s | casos completos=%d/%d | grados registrados=%d | archivo=%s",
        anotador,
        len(muestra) - len(pendientes),
        len(muestra),
        len(grados),
        ruta_respuestas,
    )
    if not pendientes:
        logger.info("no hay casos pendientes: anotacion completa")
        return
    imprimir_ayuda_grado()
    total_candidatos = sum(len(lista) for lista in candidatos_por_id.values())
    nuevos = 0
    casos_cerrados = 0
    salir = False
    try:
        for posicion, caso in enumerate(pendientes, 1):
            candidatos = candidatos_por_id.get(str(caso["id"]), [])
            imprimir_caso(caso, candidatos, grados, posicion, len(pendientes))
            saltar_caso = False
            candidatos_pendientes = [
                candidato
                for candidato in candidatos
                if (str(caso["id"]), candidato["articulo"]) not in grados
            ]
            posicion_candidato = 0
            while posicion_candidato < len(candidatos_pendientes):
                candidato = candidatos_pendientes[posicion_candidato]
                t_candidato = time.perf_counter()
                while True:
                    entrada = input(
                        f"  [{candidato['indice']:2d}/{len(candidatos)}] art. {candidato['articulo']} | "
                        "grado (2/1/0, lote '2 1 0 ...', Enter=pendiente, s=saltar caso, q=salir, ?=ayuda): "
                    )
                    accion, valor, mensaje = interpretar_grado(entrada)
                    if mensaje:
                        print(mensaje)
                    if accion == "ayuda":
                        imprimir_ayuda_grado()
                        continue
                    if accion == "reintentar":
                        continue
                    if accion == "lote" and len(valor) > len(candidatos_pendientes) - posicion_candidato:
                        print(
                            f"  lote de {len(valor)} grados, pero solo quedan "
                            f"{len(candidatos_pendientes) - posicion_candidato} candidatos pendientes"
                        )
                        continue
                    break
                if accion == "salir":
                    salir = True
                    break
                if accion == "saltar":
                    saltar_caso = True
                    break
                if accion == "lote":
                    for desplazamiento, nota in enumerate(valor):
                        objetivo = candidatos_pendientes[posicion_candidato + desplazamiento]
                        if nota is None:
                            continue
                        registrar_grado(
                            ruta_respuestas,
                            grados,
                            anotador,
                            str(caso["id"]),
                            objetivo,
                            int(nota),
                            t_candidato,
                            logger,
                            total_candidatos,
                        )
                        nuevos += 1
                    posicion_candidato += len(valor)
                    continue
                if accion == "pendiente":
                    posicion_candidato += 1
                    continue
                registrar_grado(
                    ruta_respuestas,
                    grados,
                    anotador,
                    str(caso["id"]),
                    candidato,
                    int(valor),
                    t_candidato,
                    logger,
                    total_candidatos,
                )
                nuevos += 1
                posicion_candidato += 1
            if salir:
                break
            if saltar_caso:
                logger.info("caso %s saltado (queda pendiente)", caso["id"])
                continue
            if caso_completo(caso, candidatos_por_id, grados):
                casos_cerrados += 1
            imprimir_resumen_caso(caso, candidatos, grados)
    except EOFError:
        logger.warning("entrada terminada (EOF): se conserva lo respondido")
    except KeyboardInterrupt:
        logger.warning("interrumpido por el usuario: se conserva lo respondido")
    logger.info(
        "sesion: %d grados nuevos | casos cerrados=%d | grados totales=%d | %.1f s",
        nuevos,
        casos_cerrados,
        len(grados),
        time.perf_counter() - t0,
    )


def modo_resumen(logger: logging.Logger, rutas: dict) -> None:
    """Imprime el avance de la anotacion (global y por caso).

    Args:
        logger: Logger activo.
        rutas: Dict de rutas activas.
    """
    t0 = time.perf_counter()
    muestra = cargar_jsonl(rutas["muestra"])
    candidatos_por_id = {str(registro["id"]): registro["candidatos"] for registro in cargar_jsonl(rutas["candidatos"])}
    total_candidatos = sum(len(lista) for lista in candidatos_por_id.values())
    respuestas = cargar_respuestas_todas(rutas, logger)
    print(f"Muestra gold: {len(muestra)} casos | {total_candidatos} candidatos | {rutas['muestra']}")
    if not respuestas:
        print("Todavia no hay respuestas. Ejecute --anotar --anotador NOMBRE")
    for anotador in sorted(respuestas):
        grados = respuestas[anotador]
        completos = sum(1 for caso in muestra if caso_completo(caso, candidatos_por_id, grados))
        con_grado = len({clave[0] for clave in grados})
        print(
            f"\n[{anotador}] grados {len(grados)}/{total_candidatos} | casos con al menos un grado "
            f"{con_grado}/{len(muestra)} | casos completos {completos}/{len(muestra)}"
        )
        for caso in muestra:
            lista = candidatos_por_id.get(str(caso["id"]), [])
            marcados = sum(1 for candidato in lista if (str(caso["id"]), candidato["articulo"]) in grados)
            estado = "completo" if (lista and marcados == len(lista)) else ("parcial" if marcados else "pendiente")
            print(
                f"  {caso['id']:>6} | {caso['periodo']} | T{caso['tercil_dificultad']} | "
                f"{marcados:2d}/{len(lista):2d} | {estado}"
            )
    logger.info("--resumen completado en %.2f s", time.perf_counter() - t0)


def modo_exportar_csv(forzar: bool, smoke: bool, logger: logging.Logger, rutas: dict) -> None:
    """Exporta las anotaciones a CSV plano (una fila por candidato calificado).

    Args:
        forzar: True permite reescribir el CSV existente.
        smoke: True si es corrida smoke (scratch reescribible).
        logger: Logger activo.
        rutas: Dict de rutas activas.

    Raises:
        SystemExit: Si no hay respuestas que exportar.
    """
    t0 = time.perf_counter()
    muestra = cargar_jsonl(rutas["muestra"])
    indice_caso = {str(caso["id"]): int(caso.get("indice", 0)) for caso in muestra}
    candidatos_por_id = {str(registro["id"]): registro["candidatos"] for registro in cargar_jsonl(rutas["candidatos"])}
    respuestas = cargar_respuestas_todas(rutas, logger)
    filas = []
    huerfanos = 0
    for anotador in sorted(respuestas):
        for (id_caso, articulo), registro in respuestas[anotador].items():
            lista = candidatos_por_id.get(id_caso)
            if lista is None:
                huerfanos += 1
                continue
            candidato = next((item for item in lista if item["articulo"] == articulo), None)
            filas.append(
                {
                    "id": id_caso,
                    "articulo": articulo,
                    "grado": int(registro["nota_grado"]),
                    "citado_ner": bool(candidato["citado_ner"]) if candidato else False,
                    "anotador": anotador,
                    "_caso": indice_caso.get(id_caso, 0),
                    "_cand": int(registro.get("indice_candidato", 0)),
                }
            )
    if not filas:
        logger.error("no hay respuestas que coincidan con la muestra; ejecute --anotar primero")
        raise SystemExit(1)
    if huerfanos:
        logger.warning("%d respuestas de ids fuera de la muestra actual (ignoradas)", huerfanos)
    destino = verificar_salida_nueva(rutas["export_csv"], forzar, smoke, logger)
    destino.parent.mkdir(parents=True, exist_ok=True)
    columnas = ["id", "articulo", "grado", "citado_ner", "anotador"]
    tabla = pd.DataFrame(filas).sort_values(["anotador", "_caso", "_cand"])[columnas]
    tabla.to_csv(destino, index=False, encoding="utf-8")
    logger.info("CSV exportado: %s (%d filas) en %.2f s", destino, len(tabla), time.perf_counter() - t0)


# ===== SECCION 6: metricas graduadas y evaluacion de rutas =====


def cargar_top10_archivo(ruta: str | Path, logger: logging.Logger) -> dict:
    """Carga y normaliza el campo top10_por_caso de un benchmark (cacheado).

    Args:
        ruta: Ruta del JSON del benchmark.
        logger: Logger activo.

    Returns:
        Dict id_caso -> estrategia -> lista de articulos normalizados.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    clave = str(ruta)
    if clave not in _CACHE_TOP10:
        archivo = Path(ruta)
        if not archivo.exists():
            raise FileNotFoundError(f"no existe el benchmark: {archivo}")
        with open(archivo, encoding="utf-8") as manejador:
            datos = json.load(manejador)
        top10 = {
            str(id_caso): {
                estrategia: [normalizar_articulo(item.get("id")) for item in lista if item.get("id")]
                for estrategia, lista in estrategias.items()
            }
            for id_caso, estrategias in datos.get("top10_por_caso", {}).items()
        }
        _CACHE_TOP10[clave] = top10
        logger.info("top10 cargado: %s (%d casos)", archivo, len(top10))
    return _CACHE_TOP10[clave]


def detectar_rutas(top10_por_archivo: dict[str, dict], logger: logging.Logger) -> list[dict]:
    """Detecta las rutas evaluables disponibles en los benchmarks.

    Args:
        top10_por_archivo: Dict archivo -> top10 normalizado.
        logger: Logger activo.

    Returns:
        Lista de rutas detectadas (con n_casos_con_top10), en orden de CONFIG.
    """
    detectadas = []
    for ruta in CONFIG["rutas_evaluables"]:
        top10 = top10_por_archivo.get(ruta["archivo"])
        if top10 is None:
            logger.warning("ruta %s omitida: falta %s", ruta["clave"], ruta["archivo"])
            continue
        n_casos = sum(1 for estrategias in top10.values() if estrategias.get(ruta["estrategia"]))
        if n_casos == 0:
            logger.warning("ruta %s omitida: estrategia %s sin top10", ruta["clave"], ruta["estrategia"])
            continue
        detectadas.append({**ruta, "n_casos_con_top10": n_casos})
        logger.info("ruta evaluable %s (%s): %d casos con top10", ruta["clave"], ruta["estrategia"], n_casos)
    return detectadas


def ganancia_grado(grado: int) -> float:
    """Ganancia exponencial de un grado de relevancia (2^g - 1).

    Args:
        grado: Grado en {0, 1, 2}.

    Returns:
        Ganancia (0, 1 o 3).
    """
    return float((2 ** int(grado)) - 1)


def ndcg_graduado_at_k(grados: dict[str, int], ranking: list[str], k: int) -> float:
    """Calcula nDCG@k con relevancia graduada (ganancia exponencial).

    El IDCG usa los grados anotados ordenados de mayor a menor (tope k). Un
    articulo del ranking sin grado anotado cuenta como 0. Si el caso no tiene
    ningun grado positivo (IDCG = 0), devuelve 0.0.

    Args:
        grados: Dict articulo -> grado (0, 1, 2).
        ranking: Articulos ordenados por el retrieval (mejor primero).
        k: Profundidad de corte.

    Returns:
        nDCG@k en [0.0, 1.0].
    """
    dcg = sum(
        ganancia_grado(grados.get(articulo, 0)) / np.log2(posicion + 1)
        for posicion, articulo in enumerate(ranking[:k], start=1)
    )
    ideales = sorted(grados.values(), reverse=True)[:k]
    idcg = sum(ganancia_grado(grado) / np.log2(posicion + 1) for posicion, grado in enumerate(ideales, start=1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def metricas_binarias(grados: dict[str, int], ranking: list[str], k: int, umbral: int) -> dict:
    """Calcula P@k y R@k binarios con un umbral de grado.

    Args:
        grados: Dict articulo -> grado.
        ranking: Articulos ordenados por el retrieval.
        k: Profundidad de corte.
        umbral: Grado minimo para considerar relevante (1).

    Returns:
        Dict {"p": P@k, "r": R@k, "n_relevantes": total de relevantes}.
    """
    relevantes = {articulo for articulo, grado in grados.items() if int(grado) >= umbral}
    top = ranking[:k]
    aciertos = sum(1 for articulo in top if articulo in relevantes)
    precision = aciertos / k if k > 0 else 0.0
    recall = aciertos / len(relevantes) if relevantes else 0.0
    return {"p": float(precision), "r": float(recall), "n_relevantes": len(relevantes)}


def recall_graduado_at_k(grados: dict[str, int], ranking: list[str], k: int) -> float:
    """Calcula el recall graduado @k (ganancia recuperada / ganancia total).

    Args:
        grados: Dict articulo -> grado.
        ranking: Articulos ordenados por el retrieval.
        k: Profundidad de corte.

    Returns:
        Recall graduado en [0.0, 1.0]; 0.0 si el caso no tiene ganancia.
    """
    total = sum(ganancia_grado(grado) for grado in grados.values())
    if total <= 0:
        return 0.0
    recuperado = sum(ganancia_grado(grados.get(articulo, 0)) for articulo in ranking[:k])
    return float(recuperado / total)


def evaluar_ruta(gold_por_id: dict[str, dict[str, int]], ranking_por_id: dict[str, list[str]], variante: str) -> tuple[list[dict], int]:
    """Evalua una ruta por caso contra el gold humano en una variante.

    Args:
        gold_por_id: Dict id_caso -> (articulo -> grado).
        ranking_por_id: Dict id_caso -> ranking de la ruta (mejor primero).
        variante: 'completo' (fuera del pool = 0) o 'pool' (ranking restringido
            a los articulos anotados).

    Returns:
        Tupla (lista de metricas por caso, items del ranking fuera del pool).
    """
    por_caso = []
    n_fuera = 0
    for id_caso in sorted(gold_por_id, key=clave_id):
        ranking = ranking_por_id.get(id_caso)
        if ranking is None:
            continue
        grados = gold_por_id[id_caso]
        if variante == "pool":
            ranking = [articulo for articulo in ranking if articulo in grados]
        else:
            n_fuera += sum(1 for articulo in ranking if articulo not in grados)
        binario_5 = metricas_binarias(grados, ranking, 5, CONFIG["umbral_relevante"])
        binario_10 = metricas_binarias(grados, ranking, 10, CONFIG["umbral_relevante"])
        por_caso.append(
            {
                "id": id_caso,
                "ndcg_at_5": ndcg_graduado_at_k(grados, ranking, 5),
                "ndcg_at_10": ndcg_graduado_at_k(grados, ranking, 10),
                "p_at_5": binario_5["p"],
                "p_at_10": binario_10["p"],
                "r_at_5": binario_5["r"],
                "r_at_10": binario_10["r"],
                "r_graduado_at_10": recall_graduado_at_k(grados, ranking, 10),
                "n_relevantes": binario_10["n_relevantes"],
                "n_anotados": len(grados),
            }
        )
    return por_caso, n_fuera


MEDIDAS_GRADUADAS = (
    "ndcg_at_5",
    "ndcg_at_10",
    "p_at_5",
    "p_at_10",
    "r_at_5",
    "r_at_10",
    "r_graduado_at_10",
)


def resumir_metricas_ruta(por_caso: list[dict], logger: logging.Logger) -> dict:
    """Resume las metricas por caso de una ruta con IC bootstrap de la media.

    Args:
        por_caso: Lista de filas de metricas por caso.
        logger: Logger activo.

    Returns:
        Dict medida -> {n, media, p2_5, p97_5} mas n_casos_sin_relevantes.
    """
    metricas = {}
    for medida in MEDIDAS_GRADUADAS:
        valores = np.asarray([fila[medida] for fila in por_caso], dtype=np.float64)
        metricas[medida] = bootstrap_ic_media(valores, CONFIG["n_remuestreos_bootstrap"], CONFIG["semilla_bootstrap"])
    metricas["n_casos_sin_relevantes"] = sum(1 for fila in por_caso if fila["n_relevantes"] == 0)
    if metricas["n_casos_sin_relevantes"]:
        logger.warning("%d casos anotados sin ningun grado >= 1 (nDCG=0 por definicion)", metricas["n_casos_sin_relevantes"])
    return metricas


def construir_ranking_por_id(top10: dict, estrategia: str) -> dict[str, list[str]]:
    """Extrae el ranking de una estrategia por caso (solo casos con top10).

    Args:
        top10: Dict id_caso -> estrategia -> lista de articulos.
        estrategia: Nombre de la estrategia.

    Returns:
        Dict id_caso -> ranking.
    """
    return {
        id_caso: estrategias[estrategia]
        for id_caso, estrategias in top10.items()
        if estrategias.get(estrategia)
    }


def evaluar_gold(
    gold_por_id: dict[str, dict[str, int]],
    rutas_detectadas: list[dict],
    top10_por_archivo: dict[str, dict],
    logger: logging.Logger,
) -> dict:
    """Evalua todas las rutas detectadas contra un gold humano.

    Args:
        gold_por_id: Dict id_caso -> (articulo -> grado).
        rutas_detectadas: Rutas evaluables (salida de detectar_rutas).
        top10_por_archivo: top10 normalizado por archivo.
        logger: Logger activo.

    Returns:
        Dict con n_casos_anotados, rutas (variantes completo/pool) y
        comparaciones pareadas.
    """
    resultado = {"n_casos_anotados": len(gold_por_id), "rutas": {}}
    for ruta in rutas_detectadas:
        ranking_por_id = construir_ranking_por_id(top10_por_archivo[ruta["archivo"]], ruta["estrategia"])
        entrada = {
            "etiqueta": ruta["etiqueta"],
            "estrategia": ruta["estrategia"],
            "archivo": ruta["archivo"],
            "n_casos_con_top10": ruta["n_casos_con_top10"],
            "variantes": {},
        }
        for variante in ("completo", "pool"):
            por_caso, n_fuera = evaluar_ruta(gold_por_id, ranking_por_id, variante)
            entrada["variantes"][variante] = {
                "n_casos": len(por_caso),
                "n_items_fuera_pool": n_fuera,
                "metricas": resumir_metricas_ruta(por_caso, logger),
                "metricas_por_caso": por_caso,
            }
        resultado["rutas"][ruta["clave"]] = entrada
    resultado["comparaciones_pareadas"] = comparaciones_pareadas(resultado["rutas"], CONFIG["pares_comparacion"], logger)
    return resultado


def comparaciones_pareadas(rutas_resultado: dict, pares: list[tuple[str, str]], logger: logging.Logger) -> dict:
    """Bootstrap pareado de B2 vs v12 vs v8 por variante y medida.

    Args:
        rutas_resultado: Dict clave -> resultado de evaluar_gold.
        pares: Lista de pares (clave_nueva, clave_referencia).
        logger: Logger activo.

    Returns:
        Dict "nueva_vs_referencia" -> variante -> {n_casos_comunes, medidas}.
    """
    comparaciones = {}
    for clave_nueva, clave_referencia in pares:
        ruta_nueva = rutas_resultado.get(clave_nueva)
        ruta_referencia = rutas_resultado.get(clave_referencia)
        if ruta_nueva is None or ruta_referencia is None:
            logger.warning("comparacion %s vs %s omitida (ruta no disponible)", clave_nueva, clave_referencia)
            continue
        entrada = {}
        for variante in ("completo", "pool"):
            casos_nueva = {fila["id"]: fila for fila in ruta_nueva["variantes"][variante]["metricas_por_caso"]}
            casos_referencia = {fila["id"]: fila for fila in ruta_referencia["variantes"][variante]["metricas_por_caso"]}
            comunes = sorted(set(casos_nueva) & set(casos_referencia), key=clave_id)
            medidas = {}
            for medida in ("ndcg_at_5", "ndcg_at_10"):
                arreglo_nueva = np.asarray([casos_nueva[id_caso][medida] for id_caso in comunes], dtype=np.float64)
                arreglo_referencia = np.asarray(
                    [casos_referencia[id_caso][medida] for id_caso in comunes], dtype=np.float64
                )
                medidas[medida] = bootstrap_ic_pareado(
                    arreglo_nueva, arreglo_referencia, CONFIG["n_remuestreos_bootstrap"], CONFIG["semilla_bootstrap"]
                )
            entrada[variante] = {"n_casos_comunes": len(comunes), "medidas": medidas}
        comparaciones[f"{clave_nueva}_vs_{clave_referencia}"] = entrada
    return comparaciones


def cargar_gold_anotadores(rutas: dict, logger: logging.Logger) -> dict[str, dict[str, dict[str, int]]]:
    """Carga el gold humano de cada anotador (id -> articulo -> grado).

    Args:
        rutas: Dict de rutas activas.
        logger: Logger activo.

    Returns:
        Dict anotador -> (id_caso -> (articulo -> grado)).
    """
    respuestas = cargar_respuestas_todas(rutas, logger)
    gold: dict[str, dict[str, dict[str, int]]] = {}
    for anotador, grados in respuestas.items():
        por_id: dict[str, dict[str, int]] = {}
        for (id_caso, articulo), registro in grados.items():
            por_id.setdefault(id_caso, {})[articulo] = int(registro["nota_grado"])
        gold[anotador] = por_id
    return gold


def construir_gold_pooled(respuestas: dict[str, dict[tuple[str, str], dict]]) -> dict[str, dict[str, int]]:
    """Agrupa el gold de varios anotadores (ultimo registro por fecha gana).

    Args:
        respuestas: Dict anotador -> ((id, articulo) -> registro).

    Returns:
        Dict id_caso -> (articulo -> grado) del pool.
    """
    pooled: dict[tuple[str, str], dict] = {}
    for anotador in sorted(respuestas):
        for clave, registro in respuestas[anotador].items():
            previo = pooled.get(clave)
            if previo is None or str(registro.get("fecha", "")) >= str(previo.get("fecha", "")):
                pooled[clave] = registro
    por_id: dict[str, dict[str, int]] = {}
    for (id_caso, articulo), registro in pooled.items():
        por_id.setdefault(id_caso, {})[articulo] = int(registro["nota_grado"])
    return por_id


def imprimir_resumen_evaluacion(resultado: dict, logger: logging.Logger) -> None:
    """Imprime el resumen de la evaluacion por anotador, ruta y comparacion.

    Args:
        resultado: Dict con anotadores/pooled/acuerdo.
        logger: Logger activo.
    """
    for anotador in sorted(resultado["anotadores"]):
        evaluacion = resultado["anotadores"][anotador]
        logger.info("anotador %s: casos anotados=%d", anotador, evaluacion["n_casos_anotados"])
        for clave, ruta in evaluacion["rutas"].items():
            for variante in ("completo", "pool"):
                metricas = ruta["variantes"][variante]["metricas"]
                ndcg10 = metricas["ndcg_at_10"]
                logger.info(
                    "  %-12s [%s] nDCG@10=%s IC[%s; %s] n=%d | nDCG@5=%s | P@10=%s | R@10=%s",
                    clave,
                    variante,
                    formatear_numero(ndcg10["media"]),
                    formatear_numero(ndcg10["p2_5"]),
                    formatear_numero(ndcg10["p97_5"]),
                    ndcg10["n"],
                    formatear_numero(metricas["ndcg_at_5"]["media"]),
                    formatear_numero(metricas["p_at_10"]["media"]),
                    formatear_numero(metricas["r_at_10"]["media"]),
                )
        for par, datos in evaluacion["comparaciones_pareadas"].items():
            for variante in ("completo", "pool"):
                delta = datos[variante]["medidas"]["ndcg_at_10"]
                logger.info(
                    "  pareado %s [%s] nDCG@10: delta=%s IC[%s; %s] gana=%s (n=%d)",
                    par,
                    variante,
                    formatear_numero(delta["media_delta"]),
                    formatear_numero(delta["p2_5"]),
                    formatear_numero(delta["p97_5"]),
                    delta["gana"],
                    datos[variante]["n_casos_comunes"],
                )
    if resultado.get("acuerdo") is not None:
        acuerdo = resultado["acuerdo"]
        for par, datos in acuerdo["kappas_pares"].items():
            logger.info(
                "acuerdo %s: kappa=%s (%s) sobre %d pares",
                par,
                formatear_numero(datos["kappa"]),
                datos["nivel"],
                datos["n_comunes"],
            )
        if acuerdo["fleiss"] is not None:
            logger.info(
                "Fleiss global: kappa=%s (%s) sobre %d items",
                formatear_numero(acuerdo["fleiss"]["kappa"]),
                acuerdo["fleiss"]["nivel"],
                acuerdo["fleiss"]["n_items"],
            )


def modo_evaluar(forzar: bool, smoke: bool, logger: logging.Logger, rutas: dict) -> None:
    """Evalua las rutas guardadas contra el gold humano y persiste el JSON.

    Args:
        forzar: True permite reescribir la evaluacion existente.
        smoke: True si es corrida smoke (scratch reescribible).
        logger: Logger activo.
        rutas: Dict de rutas activas.

    Raises:
        SystemExit: Si no hay respuestas de ningun anotador.
    """
    t0 = time.perf_counter()
    muestra = cargar_jsonl(rutas["muestra"])
    candidatos_por_id = {str(registro["id"]): registro["candidatos"] for registro in cargar_jsonl(rutas["candidatos"])}
    respuestas = cargar_respuestas_todas(rutas, logger)
    if not respuestas:
        logger.error("no hay respuestas: ejecute --anotar antes de --evaluar")
        raise SystemExit(1)
    gold = cargar_gold_anotadores(rutas, logger)
    top10_por_archivo: dict[str, dict] = {}
    for ruta in CONFIG["rutas_evaluables"]:
        archivo = ruta["archivo"]
        if archivo in top10_por_archivo:
            continue
        ruta_json = rutas[archivo]
        if not Path(ruta_json).exists():
            logger.warning("benchmark ausente: %s", ruta_json)
            continue
        top10_por_archivo[archivo] = cargar_top10_archivo(ruta_json, logger)
    detectadas = detectar_rutas(top10_por_archivo, logger)
    resultado = {
        "meta": {
            "fecha": ahora_iso(),
            "script": "scripts/16_anotador_gold.py",
            "semilla_muestreo": CONFIG["semilla"],
            "semilla_bootstrap": CONFIG["semilla_bootstrap"],
            "n_remuestreos_bootstrap": CONFIG["n_remuestreos_bootstrap"],
            "n_muestra": len(muestra),
            "muestra": str(rutas["muestra"]),
            "md5_muestra": md5_archivo(rutas["muestra"]),
            "md5_candidatos": md5_archivo(rutas["candidatos"]),
            "md5_respuestas": {
                anotador: md5_archivo(Path(rutas["muestra"]).parent / f"respuestas_{anotador}.jsonl")
                for anotador in sorted(respuestas)
            },
            "anotadores": sorted(respuestas),
            "versiones": versiones_entorno(),
        },
        "premisas": PREMISAS,
        "anotadores": {},
        "pooled": {},
        "acuerdo": None,
    }
    for anotador in sorted(gold):
        evaluacion = evaluar_gold(gold[anotador], detectadas, top10_por_archivo, logger)
        evaluacion["n_casos_completos"] = sum(
            1 for caso in muestra if caso_completo(caso, candidatos_por_id, respuestas[anotador])
        )
        resultado["anotadores"][anotador] = evaluacion
    if len(gold) >= 2:
        resultado["pooled"] = evaluar_gold(construir_gold_pooled(respuestas), detectadas, top10_por_archivo, logger)
        resultado["acuerdo"] = calcular_acuerdo_gold(respuestas, logger)
    destino = verificar_salida_nueva(rutas["evaluacion"], forzar, smoke, logger)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as archivo:
        json.dump(resultado, archivo, ensure_ascii=False, indent=2, default=str)
    logger.info("evaluacion guardada: %s (md5 %s)", destino, md5_archivo(destino))
    imprimir_resumen_evaluacion(resultado, logger)
    logger.info("--evaluar completado en %.1f s", time.perf_counter() - t0)


# ===== SECCION 7: acuerdo inter-anotador =====


def calcular_acuerdo_gold(respuestas: dict[str, dict[tuple[str, str], dict]], logger: logging.Logger) -> dict:
    """Calcula cobertura, distribucion y acuerdo inter-anotador del gold.

    Args:
        respuestas: Dict anotador -> ((id, articulo) -> registro).
        logger: Logger activo.

    Returns:
        Dict con por_anotador (cobertura y distribucion), kappas_pares (Cohen
        sobre pares comunes) y fleiss (items calificados por todos).
    """
    grados_planos: dict[str, dict[tuple[str, str], int]] = {}
    por_anotador = {}
    for anotador in sorted(respuestas):
        planos = {
            clave: int(registro["nota_grado"])
            for clave, registro in respuestas[anotador].items()
            if registro.get("nota_grado") in (0, 1, 2)
        }
        grados_planos[anotador] = planos
        por_anotador[anotador] = {
            "n_grados": len(planos),
            "n_casos_con_grado": len({clave[0] for clave in planos}),
            "distribucion_grados": dict(sorted(Counter(planos.values()).items())),
        }
    anotadores = sorted(grados_planos)
    kappas_pares = {}
    for posicion in range(len(anotadores)):
        for siguiente in range(posicion + 1, len(anotadores)):
            anotador_a = anotadores[posicion]
            anotador_b = anotadores[siguiente]
            comunes = sorted(
                set(grados_planos[anotador_a]) & set(grados_planos[anotador_b]),
                key=lambda clave: (clave_id(clave[0]), clave[1]),
            )
            kappa = cohen_kappa(
                [str(grados_planos[anotador_a][clave]) for clave in comunes],
                [str(grados_planos[anotador_b][clave]) for clave in comunes],
            )
            kappas_pares[f"{anotador_a}|{anotador_b}"] = {
                "n_comunes": len(comunes),
                "po": kappa["po"],
                "kappa": kappa["kappa"],
                "nivel": kappa["nivel"],
            }
    fleiss = None
    if len(anotadores) >= 2:
        completos = sorted(
            set.intersection(*(set(grados_planos[anotador]) for anotador in anotadores)),
            key=lambda clave: (clave_id(clave[0]), clave[1]),
        )
        if completos:
            items = [[str(grados_planos[anotador][clave]) for anotador in anotadores] for clave in completos]
            fleiss = fleiss_kappa(items, ["0", "1", "2"])
            if fleiss is not None:
                fleiss["n_items_completos"] = len(completos)
        else:
            logger.warning("Fleiss no calculable: ningun par (caso, articulo) calificado por TODOS")
    return {"por_anotador": por_anotador, "kappas_pares": kappas_pares, "fleiss": fleiss}


def modo_calcular_acuerdo(forzar: bool, smoke: bool, logger: logging.Logger, rutas: dict) -> None:
    """Calcula y persiste el acuerdo inter-anotador del gold.

    Args:
        forzar: True permite reescribir el acuerdo existente.
        smoke: True si es corrida smoke (scratch reescribible).
        logger: Logger activo.
        rutas: Dict de rutas activas.

    Raises:
        SystemExit: Si no hay respuestas de ningun anotador.
    """
    t0 = time.perf_counter()
    respuestas = cargar_respuestas_todas(rutas, logger)
    if not respuestas:
        logger.error("no hay respuestas: ejecute --anotar antes de --calcular-acuerdo")
        raise SystemExit(1)
    acuerdo = calcular_acuerdo_gold(respuestas, logger)
    acuerdo["meta"] = {
        "fecha": ahora_iso(),
        "script": "scripts/16_anotador_gold.py",
        "anotadores": sorted(respuestas),
        "md5_respuestas": {
            anotador: md5_archivo(Path(rutas["muestra"]).parent / f"respuestas_{anotador}.jsonl")
            for anotador in sorted(respuestas)
        },
        "premisas": PREMISAS,
    }
    destino = verificar_salida_nueva(rutas["acuerdo"], forzar, smoke, logger)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as archivo:
        json.dump(acuerdo, archivo, ensure_ascii=False, indent=2, default=str)
    logger.info("acuerdo guardado: %s (md5 %s)", destino, md5_archivo(destino))
    for anotador, datos in sorted(acuerdo["por_anotador"].items()):
        logger.info(
            "anotador %s: grados=%d | casos=%d | distribucion=%s",
            anotador,
            datos["n_grados"],
            datos["n_casos_con_grado"],
            datos["distribucion_grados"],
        )
    for par, datos in acuerdo["kappas_pares"].items():
        logger.info(
            "kappa %s: %s (%s) sobre %d pares",
            par,
            formatear_numero(datos["kappa"]),
            datos["nivel"],
            datos["n_comunes"],
        )
    if acuerdo["fleiss"] is not None:
        logger.info(
            "Fleiss global: %s (%s) sobre %d items y %d evaluadores",
            formatear_numero(acuerdo["fleiss"]["kappa"]),
            acuerdo["fleiss"]["nivel"],
            acuerdo["fleiss"]["n_items"],
            acuerdo["fleiss"]["n_evaluadores"],
        )
    logger.info("--calcular-acuerdo completado en %.1f s", time.perf_counter() - t0)


# ===== SECCION 8: CLI y main =====


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y parsea los argumentos de linea de comandos.

    Args:
        argv: Lista de argumentos (None usa sys.argv).

    Returns:
        Namespace con el modo pedido, anotador, smoke y forzar.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Mini-gold humano del vinculo sentencia -> articulos CP (Sprint B3): "
            "muestra estratificada, anotacion graduada reanudable y evaluacion de "
            "rutas contra el gold humano. No entrena nada."
        )
    )
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--generar", action="store_true", help="construye muestra_gold.jsonl + candidatos.jsonl + guia")
    grupo.add_argument("--anotar", action="store_true", help="recorrido interactivo reanudable (requiere --anotador)")
    grupo.add_argument("--resumen", action="store_true", help="avance global y por caso de la anotacion")
    grupo.add_argument(
        "--exportar-csv", action="store_true", dest="exportar_csv", help="CSV plano (id, articulo, grado, citado_ner, anotador)"
    )
    grupo.add_argument("--evaluar", action="store_true", help="evalua las rutas guardadas contra el gold humano")
    grupo.add_argument(
        "--calcular-acuerdo", action="store_true", dest="calcular_acuerdo", help="acuerdo inter-anotador (Cohen/Fleiss)"
    )
    parser.add_argument("--anotador", default=None, help="nombre del anotador (requerido con --anotar)")
    parser.add_argument("--smoke", action="store_true", help="10 casos y todo en /tmp/opencode/gold_mini_smoke")
    parser.add_argument("--forzar", action="store_true", help="permite reescribir salidas existentes")
    args = parser.parse_args(argv)
    if args.anotar and not args.anotador:
        parser.error("--anotar requiere --anotador NOMBRE")
    return args


def nombre_modo(args: argparse.Namespace) -> str:
    """Devuelve el nombre legible del modo activo.

    Args:
        args: Argumentos CLI.

    Returns:
        Nombre del modo (generar, anotar, resumen, exportar_csv, evaluar,
        calcular_acuerdo).
    """
    if args.generar:
        return "generar"
    if args.anotar:
        return "anotar"
    if args.resumen:
        return "resumen"
    if args.exportar_csv:
        return "exportar_csv"
    if args.evaluar:
        return "evaluar"
    return "calcular_acuerdo"


def main(argv: list[str] | None = None) -> None:
    """Orquesta el modo pedido del anotador de mini-gold.

    Args:
        argv: Argumentos CLI (None usa sys.argv).
    """
    args = parse_args(argv)
    os.chdir(RAIZ)
    logger = configurar_logger()
    random.seed(CONFIG["semilla"])
    np.random.seed(CONFIG["semilla"])
    rutas = rutas_activas(args.smoke)
    logger.info(
        "=== INICIO anotador gold | modo=%s smoke=%s forzar=%s ===",
        nombre_modo(args),
        args.smoke,
        args.forzar,
    )
    if args.smoke:
        Path(CONFIG["dir_smoke"]).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if args.generar:
        modo_generar(args, logger, rutas)
    elif args.anotar:
        modo_anotar(args.anotador, logger, rutas)
    elif args.resumen:
        modo_resumen(logger, rutas)
    elif args.exportar_csv:
        modo_exportar_csv(args.forzar, args.smoke, logger, rutas)
    elif args.evaluar:
        modo_evaluar(args.forzar, args.smoke, logger, rutas)
    elif args.calcular_acuerdo:
        modo_calcular_acuerdo(args.forzar, args.smoke, logger, rutas)
    logger.info("=== FIN anotador gold en %.1f s ===", time.perf_counter() - t0)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - protocolo: traza completa y salida 1 ante cualquier fallo
        traceback.print_exc()
        sys.exit(1)
