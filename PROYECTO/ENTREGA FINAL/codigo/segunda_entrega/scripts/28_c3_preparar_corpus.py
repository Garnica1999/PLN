"""28_c3_preparar_corpus.py
Fecha: 2026-10-04
Descripcion: Sprint C3 - preparacion del corpus 2022-2026 (2.382 filas, 2.330
    con estado ok) con las tres capas del corpus original para poder medir la
    deriva temporal: NER juridico (M2, script 03), etiquetado de area (M3,
    script 04) y lematizacion (M1, script 05). Reutiliza las funciones de esos
    scripts via importlib (sin modificarlos ni ejecutarlos como subproceso) y
    escribe, replicando los esquemas exactos del corpus 1992-2021:
    - data/corpus_2022_2026/entidades_2022_2026.parquet (esquema de
      data/entidades_sentencias.parquet: id, tipo, ano, arts_cp, normas,
      sents_citadas, ponente);
    - data/corpus_2022_2026/etiquetas_2022_2026.parquet (esquema de
      data/etiquetas_areas.parquet: id, tipo, ano, area_primaria, top3_json,
      via, n_citas_cp, fallback_termino);
    - data/corpus_2022_2026/lemas_2022_2026.parquet (esquema de
      data/lemas_por_doc.parquet: id, lemas_texto, n_lemas);
    - resultados/c3_preparacion_stats.json (coberturas, aristas, tops, tiempos).

    NOTA DE ENTRADA (desviacion documentada): el parquet C3 guarda en `texto`
    el texto LIMPIADO E1/E2 (minusculas, sin digitos ni puntuacion, sin la
    abreviatura "art"), inservible para el NER regex de 03 (0 entidades) y no
    comparable con la lematizacion original de 05 (que corrio sobre texto
    crudo). Por eso NER y lemas reconstruyen el texto crudo desde los HTML ya
    descargados (data/corpus_2022_2026/raw/{anio}/*.html) con
    extraer_texto_html del script 26 (importlib); si un HTML faltara se usa el
    texto limpio como respaldo y se cuenta en stats. El parquet del corpus no
    se modifica. El fallback lexico de etiquetas (04) si usa el `texto` limpio
    del parquet, porque su regex trabaja en minusculas.

    Modos: --ner, --etiquetas, --lemas y --todo (secuencial ner -> etiquetas ->
    lemas). --smoke procesa 10 docs reales con todas las salidas en
    /tmp/opencode/c3_prep_smoke/ (estructura data/, resultados/ y experimentos/
    espejo de la real). --limite N acota los documentos procesados: en modo
    smoke redefine el tamano (default 10) y en modo real redirige las salidas a
    data/corpus_2022_2026/pruebas_limite_<N>/ y el stats/log a un archivo con
    sufijo, para que una prueba parcial nunca contamine las salidas canonicas.

    Reglas: solo se procesan los docs con estado ok; stub (48) y texto_corto
    (4) quedan excluidos y se reportan en stats. NO se sobrescribe ninguna
    salida existente: cada etapa reutiliza su parquet si ya existe (y recalcula
    sus stats desde el). Los checkpoints y la reanudacion por id de
    lematizacion viven en data/corpus_2022_2026/lematizacion_c3/; las salidas
    congeladas de v1/v2 (data/lemas_por_doc.parquet, resultados/lematizacion/)
    no se tocan. Tiempos por etapa impresos y registrados. Memoria consciente:
    NER streaming por filas, recursos spaCy elegidos por RAM disponible y
    lematizacion con el diseno de dos fases del script 05.

Estado del arte aplicado: NER regex del script 03 (ventana CP 60 chars, canon
    TIPO-NUM-AAAA, normalizacion de anos de 2 digitos) con rango temporal
    extendido a 1992-2026 para citas y normas del corpus nuevo; etiquetado
    ADR-1 del script 04 (bloque mayoritario de citas CP normalizadas, desempate
    por menor articulo, fallback "derecho a X"); lematizacion del script 05
    (es_core_news_md, tok.is_alpha -> lemma_.lower().strip(), dos fases
    cortos/largos con umbral 200k chars, checkpoints y reanudacion por id,
    watchdog anti-OOM).

Uso:
    /home/carlos/venv/bin/python scripts/28_c3_preparar_corpus.py --smoke --todo
    /home/carlos/venv/bin/python scripts/28_c3_preparar_corpus.py --todo
    /home/carlos/venv/bin/python scripts/28_c3_preparar_corpus.py --ner
    /home/carlos/venv/bin/python scripts/28_c3_preparar_corpus.py --lemas
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

CONFIG = {
    "semilla": 42,
    "smoke_dir": "/tmp/opencode/c3_prep_smoke",
    "smoke_n": 10,
    "estado_ok": "ok",
    "corpus": "data/corpus_2022_2026/corpus_2022_2026.parquet",
    "corpus_previo": "data/corpus_sentencias.parquet",
    "constitucion": "data/constitucion_articulos.parquet",
    "raw_dir": "data/corpus_2022_2026/raw",
    "salida_entidades": "entidades_2022_2026.parquet",
    "salida_etiquetas": "etiquetas_2022_2026.parquet",
    "salida_lemas": "lemas_2022_2026.parquet",
    "dir_ckpt_lemas": "lematizacion_c3",
    "stats": "resultados/c3_preparacion_stats.json",
    "log": "experimentos/28_c3_preparar.log",
    "modulos": {
        "descarga": "scripts/26_c3_descarga_2022_2026.py",
        "ner": "scripts/03_extraccion_entidades.py",
        "etiquetas": "scripts/04_etiquetado_areas.py",
        "lemas": "scripts/05_lematizacion.py",
    },
    "ventana_cp": 60,
    "ano_max_citas": 2026,
    "modelo_spacy": "es_core_news_md",
    # Recursos spaCy del smoke: multiproceso moderado (valida la ruta real de
    # nlp.pipe sin arriesgar RAM en una maquina compartida con llama-server).
    "spacy_smoke": {"n_process": 2, "batch_size": 4},
    # Umbrales de RAM disponible (GB) para el run real; se elige la primera
    # fila cuyo umbral se cumpla. Batch maximo 6 (no el 12 de 05): los docs
    # C3 son ~3x mas largos (raw medio ~160k chars vs ~50-60k del corpus
    # original), asi que 6 docs en vuelo por worker equivalen al lote 12 que
    # 05 valido; un lote mayor se OOM-killea en esta maquina compartida con
    # llama-server (verificado en el benchmark de 2026-10-04 con 3/10).
    "spacy_recursos_real": [
        (20.0, 4, 6),
        (10.0, 3, 6),
        (0.0, 2, 4),
    ],
    "top_arts": 20,
    "top_lemas": 20,
}

RE_ID_CORPUS = re.compile(r"^([A-Z]{1,2})-(\d{1,4})/(\d{2,4})$")


# ===== SECCION 1: CLI, configuracion y utilidades base =====
def parsear_argumentos() -> argparse.Namespace:
    """Define y parsea los argumentos de linea de comandos.

    Returns:
        Namespace con el modo (ner/etiquetas/lemas/todo), smoke y limite.

    Raises:
        SystemExit: Si argparse detecta un modo ausente o valores invalidos.
    """
    parser = argparse.ArgumentParser(
        description=(
            "C3: prepara el corpus 2022-2026 con NER (03), etiquetado de "
            "areas (04) y lematizacion (05) reutilizados via importlib."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python scripts/28_c3_preparar_corpus.py --smoke --todo\n"
            "  python scripts/28_c3_preparar_corpus.py --todo\n"
            "  python scripts/28_c3_preparar_corpus.py --lemas --limite 50\n"
        ),
    )
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument(
        "--ner",
        action="store_true",
        help="Extrae entidades juridicas (M2) del corpus 2022-2026.",
    )
    grupo.add_argument(
        "--etiquetas",
        action="store_true",
        help="Etiqueta area tematica (M3) desde las entidades nuevas.",
    )
    grupo.add_argument(
        "--lemas",
        action="store_true",
        help="Lematiza el corpus nuevo con spaCy (M1).",
    )
    grupo.add_argument(
        "--todo",
        action="store_true",
        help="Ejecuta ner, etiquetas y lemas en secuencia.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Prueba con 10 docs reales y todas las salidas en "
            "/tmp/opencode/c3_prep_smoke/."
        ),
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help=(
            "Maximo de documentos a procesar (los primeros en orden canonico "
            "de id). En smoke redefine el tamano; en modo real redirige las "
            "salidas a pruebas_limite_<N>."
        ),
    )
    args = parser.parse_args()
    if args.limite is not None and args.limite < 1:
        parser.error("--limite debe ser >= 1")
    return args


def construir_config(args: argparse.Namespace) -> dict:
    """Construye la configuracion efectiva del run (rutas y parametros).

    Args:
        args: Namespace parseado con los flags del CLI.

    Returns:
        Diccionario de configuracion con rutas absolutas; smoke escribe en
        /tmp/opencode/c3_prep_smoke/ y un --limite real en pruebas_limite_<N>
        (las entradas se leen siempre del corpus real, solo cambia la salida).
    """
    base = Path(__file__).resolve().parent.parent
    if args.smoke:
        raiz = Path(CONFIG["smoke_dir"])
        limite = args.limite if args.limite is not None else CONFIG["smoke_n"]
        dir_datos = raiz / "data" / "corpus_2022_2026"
        ruta_stats = raiz / "resultados" / "c3_preparacion_stats.json"
        ruta_log = raiz / "experimentos" / "28_c3_preparar.log"
        modo = "smoke"
    elif args.limite is not None:
        raiz = base
        limite = args.limite
        dir_datos = (
            base / "data" / "corpus_2022_2026" / f"pruebas_limite_{limite}"
        )
        ruta_stats = (
            base / "resultados" / f"c3_preparacion_stats_limite_{limite}.json"
        )
        ruta_log = base / "experimentos" / f"28_c3_preparar_limite_{limite}.log"
        modo = "prueba"
    else:
        raiz = base
        limite = None
        dir_datos = base / "data" / "corpus_2022_2026"
        ruta_stats = base / CONFIG["stats"]
        ruta_log = base / CONFIG["log"]
        modo = "real"
    cfg = {
        "modo": modo,
        "smoke": bool(args.smoke),
        "limite": limite,
        "base": base,
        "raiz": raiz,
        "dir_datos": dir_datos,
        "ruta_corpus": base / CONFIG["corpus"],
        "ruta_corpus_previo": base / CONFIG["corpus_previo"],
        "ruta_constitucion": base / CONFIG["constitucion"],
        "ruta_raw": base / CONFIG["raw_dir"],
        "ruta_entidades": dir_datos / CONFIG["salida_entidades"],
        "ruta_etiquetas": dir_datos / CONFIG["salida_etiquetas"],
        "ruta_lemas": dir_datos / CONFIG["salida_lemas"],
        "dir_ckpt_lemas": dir_datos / CONFIG["dir_ckpt_lemas"],
        "ruta_stats": ruta_stats,
        "ruta_log": ruta_log,
        "ruta_mod_descarga": base / CONFIG["modulos"]["descarga"],
        "ruta_mod_ner": base / CONFIG["modulos"]["ner"],
        "ruta_mod_etiquetas": base / CONFIG["modulos"]["etiquetas"],
        "ruta_mod_lemas": base / CONFIG["modulos"]["lemas"],
    }
    return cfg


def configurar_semillas(semilla: int) -> None:
    """Fija las semillas de aleatoriedad para reproducibilidad.

    Args:
        semilla: Valor entero de semilla.
    """
    random.seed(semilla)
    np.random.seed(semilla)


def configurar_logging(ruta_log: Path) -> logging.Logger:
    """Configura el logger con salida a consola y a archivo (append).

    Args:
        ruta_log: Ruta del archivo de log (se crea su carpeta padre).

    Returns:
        Logger configurado con dos handlers (consola y archivo).
    """
    ruta_log.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("c3_preparar")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formato = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    consola = logging.StreamHandler(sys.stdout)
    consola.setFormatter(formato)
    archivo = logging.FileHandler(ruta_log, mode="a", encoding="utf-8")
    archivo.setFormatter(formato)
    logger.addHandler(consola)
    logger.addHandler(archivo)
    return logger


def ahora_utc() -> str:
    """Devuelve la marca de tiempo UTC actual en formato ISO corto.

    Returns:
        Cadena "YYYY-MM-DDTHH:MM:SSZ".
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def porcentaje(numerador: int, denominador: int) -> float:
    """Calcula un porcentaje protegido contra division por cero.

    Args:
        numerador: Casos favorables.
        denominador: Casos totales.

    Returns:
        Porcentaje en 0-100 (0.0 si el denominador es cero).
    """
    if denominador == 0:
        return 0.0
    return round(100.0 * numerador / denominador, 2)


def leer_json(ruta: Path, defecto: dict) -> dict:
    """Lee un JSON; si no existe o esta corrupto devuelve el valor por defecto.

    Args:
        ruta: Ruta del archivo JSON.
        defecto: Valor a devolver si no se puede leer.

    Returns:
        Diccionario leido o el valor por defecto.
    """
    if not ruta.exists():
        return defecto
    try:
        with open(ruta, "r", encoding="utf-8") as manejador:
            return json.load(manejador)
    except (OSError, json.JSONDecodeError):
        return defecto


def escribir_json(ruta: Path, datos: dict) -> None:
    """Escribe un JSON de forma atomica (temporal + replace).

    Args:
        ruta: Ruta destino del JSON.
        datos: Objeto serializable a escribir.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = Path(str(ruta) + ".tmp")
    with open(temporal, "w", encoding="utf-8") as manejador:
        json.dump(datos, manejador, ensure_ascii=False, indent=2)
    os.replace(temporal, ruta)


def escribir_parquet(df: pd.DataFrame, ruta: Path, logger: logging.Logger) -> None:
    """Escribe un parquet de forma atomica (temporal + replace).

    Evita dejar un archivo a medias si el proceso muere durante la escritura:
    un parquet parcial bloquearia la reutilizacion por salida existente.

    Args:
        df: DataFrame a escribir.
        ruta: Ruta destino del parquet.
        logger: Logger de ejecucion.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = Path(str(ruta) + ".tmp")
    df.to_parquet(temporal, index=False)
    os.replace(temporal, ruta)
    logger.info("Parquet escrito: %s (%d filas)", ruta, len(df))


def cargar_modulo(ruta: Path, nombre: str):
    """Carga un script del proyecto como modulo importable via importlib.

    Los scripts 03/04/05 no se modifican: se cargan por ruta con un nombre
    propio y sus funciones se invocan directamente. La carga no ejecuta sus
    main() porque todos protegen con `if __name__ == "__main__"`.

    Args:
        ruta: Ruta del script a cargar.
        nombre: Nombre interno del modulo en sys.modules.

    Returns:
        Modulo cargado listo para invocar sus funciones.

    Raises:
        FileNotFoundError: Si la ruta del script no existe.
        ImportError: Si no se puede construir la especificacion del modulo.
    """
    if not ruta.exists():
        raise FileNotFoundError(f"Modulo no encontrado: {ruta}")
    especificacion = importlib.util.spec_from_file_location(nombre, ruta)
    if especificacion is None or especificacion.loader is None:
        raise ImportError(f"No se pudo cargar el modulo: {ruta}")
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


def numero_desde_id(doc_id: str) -> int | None:
    """Extrae el numero de sentencia del id canonico TIPO-NUM/AA.

    Args:
        doc_id: Identificador del corpus nuevo (p. ej. "T-001/22").

    Returns:
        Numero entero de la sentencia o None si el id no casa el patron.
    """
    coincidencia = RE_ID_CORPUS.match(str(doc_id))
    if coincidencia is None:
        return None
    return int(coincidencia.group(2))


def decodificar_lista_json(valor: object) -> list:
    """Decodifica una celda JSON de lista a lista de Python.

    Args:
        valor: Celda con lista JSON serializada, lista o nulo.

    Returns:
        Lista de elementos; vacia si el valor es nulo o invalido.
    """
    if valor is None:
        return []
    if isinstance(valor, list):
        return list(valor)
    try:
        datos = json.loads(valor)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(datos, list):
        return []
    return datos


# ===== SECCION 2: carga y filtrado del corpus 2022-2026 =====
def cargar_corpus_filtrado(cfg: dict, logger: logging.Logger) -> tuple:
    """Carga el corpus nuevo, excluye stub/texto_corto y aplica el limite.

    El orden canonico es por id ascendente; --limite conserva los primeros N
    ids de ese orden para que las pruebas sean reproducibles.

    Args:
        cfg: Configuracion efectiva del run.
        logger: Logger de ejecucion.

    Returns:
        Tupla (DataFrame filtrado, meta del corpus) con columnas originales
        mas id/texto normalizados.

    Raises:
        FileNotFoundError: Si el parquet del corpus no existe.
        ValueError: Si faltan columnas requeridas.
    """
    if not cfg["ruta_corpus"].exists():
        raise FileNotFoundError(f"Corpus no encontrado: {cfg['ruta_corpus']}")
    df = pd.read_parquet(cfg["ruta_corpus"])
    requeridas = {"id", "tipo", "anio", "texto", "estado"}
    faltantes = requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Columnas faltantes en el corpus: {sorted(faltantes)}")
    conteos_estado = {
        str(clave): int(valor)
        for clave, valor in df["estado"].value_counts().items()
    }
    ok = df[df["estado"] == CONFIG["estado_ok"]].copy()
    ok["id"] = ok["id"].astype(str)
    ok["texto"] = ok["texto"].fillna("").astype(str)
    ok = ok.sort_values("id").reset_index(drop=True)
    n_ok = len(ok)
    if cfg["limite"] is not None:
        ok = ok.head(cfg["limite"]).copy()
    meta = {
        "docs_corpus": len(df),
        "docs_ok": int(n_ok),
        "docs_procesados": len(ok),
        "excluidos_por_estado": conteos_estado,
        "limite": cfg["limite"],
    }
    logger.info(
        "Corpus: %d filas | ok=%d | procesados=%d | excluidos=%s | limite=%s",
        len(df),
        n_ok,
        len(ok),
        conteos_estado,
        cfg["limite"],
    )
    logger.info(
        "Head corpus filtrado:\n%s",
        ok.head(3)[["id", "tipo", "anio", "n_chars_texto"]].to_string(),
    )
    return ok, meta


def ruta_html_doc(ruta_raw: Path, doc_id: str, anio: int) -> Path:
    """Construye la ruta del HTML crudo descargado por el script 26.

    Convencion de 26: T/C usan {TIPO}-{NUM}-{AA}.html y SU omite el guion tras
    SU ({SU}{NUM}-{AA}.html, p. ej. SU020-22.html).

    Args:
        ruta_raw: Directorio raiz de los HTML crudos.
        doc_id: Id canonico del corpus (p. ej. "T-001/22", "SU-020/22").
        anio: Anio de la sentencia (carpeta de particion).

    Returns:
        Ruta esperada del HTML crudo.
    """
    nombre = str(doc_id).replace("/", "-")
    if nombre.startswith("SU-"):
        nombre = "SU" + nombre[3:]
    return ruta_raw / str(int(anio)) / f"{nombre}.html"


def cargar_textos_crudos(
    df_corpus: pd.DataFrame, cfg: dict, mod26, logger: logging.Logger
) -> pd.DataFrame:
    """Reconstruye el texto crudo de cada doc desde su HTML descargado.

    El parquet C3 solo guarda el texto limpiado E1/E2, mientras que el NER de
    03 y la lematizacion de 05 operan sobre texto crudo (digitos, mayusculas,
    puntuacion); aqui se recupera el texto visible del HTML con
    extraer_texto_html del script 26. Si un HTML faltara o viniera vacio, se
    cae al texto limpio del parquet y se cuenta el respaldo.

    Args:
        df_corpus: Corpus filtrado (id, anio, texto).
        cfg: Configuracion efectiva del run.
        mod26: Modulo del script 26 cargado via importlib.
        logger: Logger de ejecucion.

    Returns:
        Copia del corpus con la columna adicional texto_crudo; los metadatos
        de la reconstruccion quedan en attrs["info_texto_crudo"].
    """
    inicio = time.time()
    textos = []
    n_fallback = 0
    ids_fallback = []
    for fila in df_corpus.itertuples(index=False):
        ruta_html = ruta_html_doc(cfg["ruta_raw"], str(fila.id), int(fila.anio))
        texto = ""
        if ruta_html.exists():
            texto = mod26.extraer_texto_html(ruta_html, mod26.CONFIG)
        if not texto.strip():
            texto = fila.texto if isinstance(fila.texto, str) else ""
            n_fallback += 1
            ids_fallback.append(str(fila.id))
        textos.append(texto)
    df = df_corpus.copy()
    df["texto_crudo"] = textos
    segundos = time.time() - inicio
    info = {
        "docs": len(df),
        "docs_html": len(df) - n_fallback,
        "docs_fallback_limpio": n_fallback,
        "ids_fallback": ids_fallback[:20],
        "segundos": round(segundos, 2),
        "chars_crudos_total": int(sum(len(texto) for texto in textos)),
    }
    df.attrs["info_texto_crudo"] = info
    logger.info(
        "Texto crudo: %d docs desde HTML, %d fallback a texto limpio, %.1f s",
        info["docs_html"],
        n_fallback,
        segundos,
    )
    if n_fallback:
        logger.warning(
            "Docs sin HTML crudo utilizable (%d): %s",
            n_fallback,
            ids_fallback[:10],
        )
    return df


def construir_canones(cfg: dict, df_corpus: pd.DataFrame, logger) -> set:
    """Construye el conjunto de canones TIPO-NUM-AAAA para resolver citas.

    Une los canones del corpus nuevo (derivados del id TIPO-NUM/AA y la
    columna anio) con los del corpus 1992-2021, de modo que una cita del
    documento pueda resolverse contra cualquiera de las dos epocas.

    Args:
        cfg: Configuracion efectiva del run.
        df_corpus: Corpus nuevo filtrado (id, tipo, anio).
        logger: Logger de ejecucion.

    Returns:
        Conjunto de cadenas canonicas ("T-293-1997", "C-5-2026").
    """
    canones = set()
    for fila in df_corpus.itertuples(index=False):
        numero = numero_desde_id(str(fila.id))
        if numero is not None:
            canones.add(f"{fila.tipo}-{numero}-{int(fila.anio)}")
    logger.info("Canones corpus nuevo: %d", len(canones))
    if cfg["ruta_corpus_previo"].exists():
        previo = pd.read_parquet(
            cfg["ruta_corpus_previo"], columns=["tipo", "numero", "ano"]
        )
        for tipo, numero, anio in zip(
            previo["tipo"], previo["numero"], previo["ano"]
        ):
            canones.add(f"{tipo!s}-{int(numero)}-{int(anio)}")
        logger.info("Canones totales con corpus 1992-2021: %d", len(canones))
    else:
        logger.warning(
            "Corpus previo no encontrado (%s): resolucion solo contra 2022-2026",
            cfg["ruta_corpus_previo"],
        )
    return canones


# ===== SECCION 3: etapa NER (M2, reuso de 03) =====
def extraer_ner_corpus(
    df_corpus: pd.DataFrame, mod03, logger: logging.Logger
) -> tuple:
    """Aplica los extractores del script 03 a cada documento del corpus nuevo.

    Reutiliza extraer_todo (arts CP, normas, sentencias citadas, ponente) con
    Counter de diagnosticos y el rango temporal ya extendido a 2026 en
    mod03.CONFIG. Opera sobre texto_crudo (reconstruido del HTML), no sobre el
    texto limpiado del parquet. Escribe los campos de lista como JSON (mismo
    formato que data/entidades_sentencias.parquet).

    Args:
        df_corpus: Corpus filtrado con id, tipo, anio y texto_crudo.
        mod03: Modulo del script 03 cargado via importlib.
        logger: Logger de ejecucion.

    Returns:
        Tupla (DataFrame con esquema de entidades, contadores Counter,
        segundos de extraccion).
    """
    contadores: Counter = Counter()
    filas = []
    inicio = time.time()
    for fila in df_corpus.itertuples(index=False):
        texto = fila.texto_crudo if isinstance(fila.texto_crudo, str) else ""
        menciones: Counter = Counter()
        resultado = mod03.extraer_todo(
            texto, contadores, CONFIG["ventana_cp"], menciones
        )
        contadores["menciones_sent_total"] += sum(menciones.values())
        filas.append(
            {
                "id": str(fila.id),
                "tipo": str(fila.tipo),
                "ano": int(fila.anio),
                "arts_cp": json.dumps(resultado["arts_cp"], ensure_ascii=False),
                "normas": json.dumps(resultado["normas"], ensure_ascii=False),
                "sents_citadas": json.dumps(
                    resultado["sents_citadas"], ensure_ascii=False
                ),
                "ponente": resultado["ponente"],
            }
        )
    segundos = time.time() - inicio
    columnas = [
        "id",
        "tipo",
        "ano",
        "arts_cp",
        "normas",
        "sents_citadas",
        "ponente",
    ]
    df = pd.DataFrame(filas, columns=columnas)
    logger.info("NER: %d docs extraidos en %.1f s", len(df), segundos)
    logger.info("NER descartes: %s", dict(contadores))
    return df, contadores, segundos


def decodificar_entidades(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia del DataFrame de entidades con listas decodificadas.

    Args:
        df: DataFrame de entidades (listas JSON como texto).

    Returns:
        Copia con arts_cp, normas y sents_citadas como listas de Python.
    """
    df = df.copy()
    for columna in ("arts_cp", "normas", "sents_citadas"):
        df[columna] = df[columna].map(decodificar_lista_json)
    return df


def stats_ner(
    df: pd.DataFrame,
    contadores: Counter,
    canones: set,
    segundos: float,
    reusado: bool,
) -> dict:
    """Calcula coberturas, aristas y tops del NER sobre el corpus nuevo.

    Args:
        df: Entidades con listas decodificadas (id, tipo, ano, arts_cp,
            normas, sents_citadas, ponente).
        contadores: Descartes de la extraccion (vacio si se reutilizo salida).
        canones: Canones TIPO-NUM-AAAA de resolucion (ambas epocas).
        segundos: Duracion de la extraccion en segundos.
        reusado: True si el parquet existia y no se recalculo.

    Returns:
        Diccionario de estadisticas serializable a JSON.
    """
    total = len(df)
    tiene_cp = df["arts_cp"].map(bool)
    tiene_norma = df["normas"].map(bool)
    tiene_sent = df["sents_citadas"].map(bool)
    tiene_ponente = df["ponente"].notna()
    cobertura = {
        "cp": porcentaje(int(tiene_cp.sum()), total),
        "norma": porcentaje(int(tiene_norma.sum()), total),
        "sent": porcentaje(int(tiene_sent.sum()), total),
        "ponente": porcentaje(int(tiene_ponente.sum()), total),
    }
    por_tipo = {}
    for tipo, sub in df.groupby("tipo", sort=True):
        n = len(sub)
        por_tipo[str(tipo)] = {
            "n": int(n),
            "cp": porcentaje(int(sub["arts_cp"].map(bool).sum()), n),
            "norma": porcentaje(int(sub["normas"].map(bool).sum()), n),
            "sent": porcentaje(int(sub["sents_citadas"].map(bool).sum()), n),
            "ponente": porcentaje(int(sub["ponente"].notna().sum()), n),
        }
    por_anio = {}
    for anio, sub in df.groupby("ano", sort=True):
        n = len(sub)
        por_anio[str(int(anio))] = {
            "n": int(n),
            "cp": porcentaje(int(sub["arts_cp"].map(bool).sum()), n),
            "sent": porcentaje(int(sub["sents_citadas"].map(bool).sum()), n),
            "ponente": porcentaje(int(sub["ponente"].notna().sum()), n),
        }
    conteo_docs_art: Counter = Counter()
    for lista in df["arts_cp"]:
        for articulo in set(lista):
            conteo_docs_art[str(articulo)] += 1
    top_arts = [
        [articulo, int(n)]
        for articulo, n in conteo_docs_art.most_common(CONFIG["top_arts"])
    ]
    pares = set()
    pares_sin_self = set()
    menciones_unicas = 0
    n_resuelven = 0
    docs_autocita = 0
    docs_autocita_2plus = 0
    for fila in df.itertuples(index=False):
        numero = numero_desde_id(str(fila.id))
        propia = None
        if numero is not None:
            propia = f"{fila.tipo}-{numero}-{int(fila.ano)}"
        n_self = 0
        for cita in fila.sents_citadas:
            pares.add((fila.id, cita))
            if cita != propia:
                pares_sin_self.add((fila.id, cita))
            else:
                n_self += 1
            menciones_unicas += 1
            if cita in canones:
                n_resuelven += 1
        if n_self >= 1:
            docs_autocita += 1
        if n_self >= 2:
            docs_autocita_2plus += 1
    stats = {
        "fecha": ahora_utc(),
        "reusado": bool(reusado),
        "segundos": round(segundos, 2),
        "n_docs": total,
        "n_canones_resolucion": len(canones),
        "cobertura_pct": cobertura,
        "por_tipo_pct": por_tipo,
        "por_anio_pct": por_anio,
        "n_aristas": len(pares),
        "n_aristas_sin_self": len(pares_sin_self),
        "menciones_unicas_por_doc": menciones_unicas,
        "menciones_con_duplicados": (
            int(contadores.get("menciones_sent_total", 0)) if not reusado else None
        ),
        "pct_menciones_resuelven_corpus": porcentaje(n_resuelven, menciones_unicas),
        "docs_autocita": docs_autocita,
        "docs_autocita_2plus": docs_autocita_2plus,
        "top_arts_docs": top_arts,
        "ponente_nulos": int(df["ponente"].isna().sum()),
        "descartes": (
            {clave: int(valor) for clave, valor in sorted(contadores.items())}
            if not reusado
            else None
        ),
    }
    return stats


def etapa_ner(
    cfg: dict, df_corpus: pd.DataFrame, mod03, canones: set, logger
) -> tuple:
    """Ejecuta la etapa NER respetando salidas existentes.

    Args:
        cfg: Configuracion efectiva del run.
        df_corpus: Corpus filtrado con texto.
        mod03: Modulo del script 03 cargado.
        canones: Canones de resolucion de citas.
        logger: Logger de ejecucion.

    Returns:
        Tupla (DataFrame de entidades decodificado, stats de la etapa).
    """
    if cfg["ruta_entidades"].exists():
        logger.info(
            "NER: salida existente %s, se reutiliza sin sobrescribir",
            cfg["ruta_entidades"],
        )
        df = decodificar_entidades(pd.read_parquet(cfg["ruta_entidades"]))
        if len(df) != len(df_corpus):
            logger.warning(
                "NER: filas existentes (%d) != docs esperados (%d); se "
                "reportan las stats del archivo tal como esta",
                len(df),
                len(df_corpus),
            )
        stats = stats_ner(df, Counter(), canones, 0.0, True)
        return df, stats
    df_json, contadores, segundos = extraer_ner_corpus(df_corpus, mod03, logger)
    escribir_parquet(df_json, cfg["ruta_entidades"], logger)
    df = decodificar_entidades(df_json)
    stats = stats_ner(df, contadores, canones, segundos, False)
    return df, stats


# ===== SECCION 4: etapa de etiquetas de area (M3, reuso de 04) =====
def stats_etiquetas(
    df: pd.DataFrame, diagnostico: dict | None, segundos: float, reusado: bool
) -> dict:
    """Calcula coberturas por via, areas y criterio del etiquetado.

    Args:
        df: Etiquetas con esquema de data/etiquetas_areas.parquet.
        diagnostico: Contadores de normalizacion de citas (None si reusado).
        segundos: Duracion del etiquetado en segundos.
        reusado: True si el parquet existia y no se recalculo.

    Returns:
        Diccionario de estadisticas serializable a JSON.
    """
    total = len(df)
    vias = {str(k): int(v) for k, v in df["via"].value_counts().items()}
    n_cita = vias.get("cita_cp", 0)
    n_fallback = vias.get("fallback_derecho", 0)
    n_sin = vias.get("sin_area", 0)
    asignadas = n_cita + n_fallback
    por_tipo = {}
    for tipo, sub in df.groupby("tipo", sort=True):
        n = len(sub)
        por_tipo[str(tipo)] = {
            "n": int(n),
            "cita_pct": porcentaje(int((sub["via"] == "cita_cp").sum()), n),
            "fallback_pct": porcentaje(
                int((sub["via"] == "fallback_derecho").sum()), n
            ),
            "sin_area_pct": porcentaje(int((sub["via"] == "sin_area").sum()), n),
        }
    por_anio = {}
    for anio, sub in df.groupby("ano", sort=True):
        n = len(sub)
        por_anio[str(int(anio))] = {
            "n": int(n),
            "cita_pct": porcentaje(int((sub["via"] == "cita_cp").sum()), n),
            "fallback_pct": porcentaje(
                int((sub["via"] == "fallback_derecho").sum()), n
            ),
            "sin_area_pct": porcentaje(int((sub["via"] == "sin_area").sum()), n),
        }
    dist = df["area_primaria"].value_counts()
    areas = [[str(area), int(n)] for area, n in dist.items()]
    asignadas_pct = porcentaje(asignadas, total)
    stats = {
        "fecha": ahora_utc(),
        "reusado": bool(reusado),
        "segundos": round(segundos, 2),
        "n_docs": total,
        "vias_docs": vias,
        "via_cita_pct": porcentaje(n_cita, total),
        "via_fallback_pct": porcentaje(n_fallback, total),
        "via_sin_area_pct": porcentaje(n_sin, total),
        "asignadas_pct": asignadas_pct,
        "criterio_95_asignadas": bool(asignadas_pct >= 95.0),
        "areas_docs": areas,
        "n_citas_cp_media": round(float(df["n_citas_cp"].mean()), 3),
        "por_tipo_pct": por_tipo,
        "por_anio_pct": por_anio,
        "diagnostico_normalizacion": (
            {clave: int(valor) for clave, valor in sorted(diagnostico.items())}
            if diagnostico is not None
            else None
        ),
    }
    return stats


def etapa_etiquetas(cfg: dict, mod04, logger) -> tuple:
    """Ejecuta el etiquetado de areas respetando salidas existentes.

    Reutiliza mod04.construir_tabla, que ya recibe las tres rutas por
    argumento (entidades, constitucion, corpus) y aplica ADR-1: via primaria
    por citas CP normalizadas, fallback "derecho a X" y sin_area_definida.

    Args:
        cfg: Configuracion efectiva del run.
        mod04: Modulo del script 04 cargado.
        logger: Logger de ejecucion.

    Returns:
        Tupla (DataFrame de etiquetas, stats de la etapa).

    Raises:
        FileNotFoundError: Si no existe el parquet de entidades (--ner previo).
    """
    if cfg["ruta_etiquetas"].exists():
        logger.info(
            "Etiquetas: salida existente %s, se reutiliza sin sobrescribir",
            cfg["ruta_etiquetas"],
        )
        df = pd.read_parquet(cfg["ruta_etiquetas"])
        stats = stats_etiquetas(df, None, 0.0, True)
        return df, stats
    if not cfg["ruta_entidades"].exists():
        raise FileNotFoundError(
            f"Falta {cfg['ruta_entidades']}: ejecute primero --ner"
        )
    inicio = time.time()
    df, diagnostico = mod04.construir_tabla(
        cfg["ruta_entidades"],
        cfg["ruta_constitucion"],
        cfg["ruta_corpus"],
        logger,
    )
    segundos = time.time() - inicio
    escribir_parquet(df, cfg["ruta_etiquetas"], logger)
    logger.info("Etiquetas: %d filas en %.1f s", len(df), segundos)
    logger.info(
        "Etiquetas: head\n%s",
        df.head(3)[
            ["id", "area_primaria", "via", "n_citas_cp"]
        ].to_string(),
    )
    stats = stats_etiquetas(df, diagnostico, segundos, False)
    return df, stats


# ===== SECCION 5: etapa de lematizacion (M1, reuso de 05) =====
def recursos_spacy(cfg: dict, logger: logging.Logger) -> dict:
    """Elige n_process y batch_size de spaCy segun RAM disponible.

    En smoke usa recursos moderados fijos; en modo real aplica los umbrales de
    CONFIG["spacy_recursos_real"] (4/12 validado por el script 05 con ~7 GB
    RSS) y avisa si la RAM disponible es baja.

    Args:
        cfg: Configuracion efectiva del run.
        logger: Logger de ejecucion.

    Returns:
        Dict con claves n_process y batch_size.
    """
    if cfg["smoke"]:
        recursos = dict(CONFIG["spacy_smoke"])
    else:
        disponible_gb = psutil.virtual_memory().available / (1024**3)
        n_process, batch_size = 2, 6
        for umbral_gb, candidato_proc, candidato_batch in CONFIG[
            "spacy_recursos_real"
        ]:
            if disponible_gb >= umbral_gb:
                n_process, batch_size = candidato_proc, candidato_batch
                break
        if disponible_gb < 8.0:
            logger.warning(
                "RAM disponible baja (%.1f GB): spaCy con n_process=%d "
                "batch_size=%d; libere memoria si hay OOM",
                disponible_gb,
                n_process,
                batch_size,
            )
        recursos = {"n_process": n_process, "batch_size": batch_size}
    logger.info(
        "Lemas: recursos spaCy n_process=%d batch_size=%d (modelo %s)",
        recursos["n_process"],
        recursos["batch_size"],
        CONFIG["modelo_spacy"],
    )
    return recursos


def configurar_modulo_lemas(cfg: dict, mod05, recursos: dict) -> None:
    """Redirige las rutas globales del script 05 hacia las salidas C3.

    El pipeline de 05 resuelve OUT_DIR, LOG_PATH, OMITIDOS_LOG, ESTADO_PATH y
    CONTADOR_PARCIAL como globales de modulo en tiempo de llamada; se
    sobreescriben aqui para que checkpoints, estado y reanudacion por id
    queden en data/corpus_2022_2026/lematizacion_c3/ sin tocar las salidas
    congeladas de v1/v2 en resultados/lematizacion/. El archivo 05 no se
    modifica: es inyeccion de configuracion en un script legado con rutas
    fijas. Tambien ajusta n_process/batch_size segun el hardware.

    Args:
        cfg: Configuracion efectiva del run.
        mod05: Modulo del script 05 cargado.
        recursos: Dict con n_process y batch_size elegidos.
    """
    dir_ckpt = cfg["dir_ckpt_lemas"]
    dir_ckpt.mkdir(parents=True, exist_ok=True)
    mod05.OUT_DIR = str(dir_ckpt)
    mod05.CONTADOR_PARCIAL = str(dir_ckpt / "contador_lemas_parcial.parquet")
    mod05.ESTADO_PATH = str(dir_ckpt / "estado.json")
    mod05.LOG_PATH = str(cfg["ruta_log"])
    mod05.OMITIDOS_LOG = str(dir_ckpt / "omitidos.log")
    mod05.CONFIG["spacy_n_process"] = recursos["n_process"]
    mod05.CONFIG["spacy_batch_size"] = recursos["batch_size"]


def stats_lemas(
    df: pd.DataFrame,
    contador: Counter | None,
    segundos: float,
    reusado: bool,
    recursos: dict | None,
) -> dict:
    """Calcula volumen y vocabulario de la lematizacion.

    Args:
        df: Lemas con esquema id, lemas_texto, n_lemas.
        contador: Frecuencias de lemas (None si no hay contador disponible).
        segundos: Duracion de la lematizacion en segundos.
        reusado: True si el parquet existia y no se recalculo.
        recursos: Recursos spaCy usados (None si reusado).

    Returns:
        Diccionario de estadisticas serializable a JSON.
    """
    total = len(df)
    total_lemas = int(df["n_lemas"].sum())
    stats = {
        "fecha": ahora_utc(),
        "reusado": bool(reusado),
        "segundos": round(segundos, 2),
        "n_docs": total,
        "total_lemas": total_lemas,
        "media_lemas_doc": round(total_lemas / total, 1) if total else 0.0,
        "nulos_lemas_texto": int(df["lemas_texto"].isna().sum()),
    }
    if contador:
        stats["n_tipos_lemas"] = len(contador)
        stats["n_tipos_con_espacio"] = sum(
            1 for lema in contador if " " in str(lema)
        )
        stats["top_lemas"] = [
            [str(lema), int(n)]
            for lema, n in contador.most_common(CONFIG["top_lemas"])
        ]
        stats["integridad_contador_secuencias"] = bool(
            sum(contador.values()) == total_lemas
        )
    else:
        vocabulario = set()
        for texto in df["lemas_texto"].astype(str):
            vocabulario.update(texto.split())
        stats["n_tipos_lemas"] = len(vocabulario)
        stats["n_tipos_con_espacio"] = sum(
            1 for lema in vocabulario if " " in lema
        )
        stats["integridad_contador_secuencias"] = None
        stats["nota_vocabulario"] = (
            "calculado desde el parquet final (contador no disponible)"
        )
    if recursos is not None:
        stats["spacy_n_process"] = recursos["n_process"]
        stats["spacy_batch_size"] = recursos["batch_size"]
    return stats


def etapa_lemas(cfg: dict, df_corpus: pd.DataFrame, mod05, logger) -> tuple:
    """Ejecuta la lematizacion respetando salidas existentes.

    Sin parquet final, invoca el pipeline del script 05 con rutas C3: dos
    fases (cortos por nlp.pipe multiproceso, largos en monoproceso), checkpoints
    cada 500, reanudacion por id y watchdog. La entrada es texto_crudo
    (reconstruido del HTML), igual que la lematizacion original sobre `texto`
    crudo, para que el vocabulario sea comparable entre epocas. Con parquet
    final, solo recalcula stats (usa contador_lemas_parcial.parquet del
    directorio de checkpoints).

    Args:
        cfg: Configuracion efectiva del run.
        df_corpus: Corpus filtrado con id y texto_crudo.
        mod05: Modulo del script 05 cargado.
        logger: Logger de ejecucion.

    Returns:
        Tupla (DataFrame de lemas, stats de la etapa).
    """
    if cfg["ruta_lemas"].exists():
        logger.info(
            "Lemas: salida existente %s, se reutiliza sin sobrescribir",
            cfg["ruta_lemas"],
        )
        df = pd.read_parquet(cfg["ruta_lemas"])
        ruta_contador = cfg["dir_ckpt_lemas"] / "contador_lemas_parcial.parquet"
        contador = None
        if ruta_contador.exists():
            contador = mod05.parquet_a_contador(str(ruta_contador))
        stats = stats_lemas(df, contador, 0.0, True, None)
        return df, stats
    recursos = recursos_spacy(cfg, logger)
    configurar_modulo_lemas(cfg, mod05, recursos)
    df_lemas = df_corpus[["id", "texto_crudo"]].rename(
        columns={"texto_crudo": "texto"}
    )
    total_docs = len(df_lemas)
    estado = mod05.cargar_estado(total_docs)
    contador, segundos = mod05.pasada_lemas(df_lemas, estado, total_docs)
    df = mod05.concatenar_checkpoints(total_docs)
    escribir_parquet(df, cfg["ruta_lemas"], logger)
    mod05.guardar_estado(total_docs, total_docs)
    logger.info("Lemas: %d filas en %.1f s", len(df), segundos)
    logger.info(
        "Lemas: head\n%s",
        df.head(2)
        .assign(lemas_texto=lambda d: d["lemas_texto"].str[:80])
        .to_string(),
    )
    stats = stats_lemas(df, contador, segundos, False, recursos)
    return df, stats


# ===== SECCION 6: estadisticas, resumen y orquestacion =====
def actualizar_stats(
    cfg: dict, secciones: dict, meta: dict, tiempos: dict, logger
) -> None:
    """Actualiza el JSON de stats preservando secciones de runs anteriores.

    Args:
        cfg: Configuracion efectiva del run.
        secciones: Dict con las stats de las etapas ejecutadas.
        meta: Metadatos del run (corpus, modo, fecha).
        tiempos: Dict con los segundos por etapa ejecutada.
        logger: Logger de ejecucion.
    """
    datos = leer_json(cfg["ruta_stats"], {})
    datos.update(meta)
    datos["tiempos_seg"] = {**datos.get("tiempos_seg", {}), **tiempos}
    for clave, valor in secciones.items():
        datos[clave] = valor
    escribir_json(cfg["ruta_stats"], datos)
    logger.info("Stats actualizadas: %s", cfg["ruta_stats"])


def imprimir_resumen(
    cfg: dict, meta_corpus: dict, secciones: dict, tiempos: dict, total_seg: float
) -> None:
    """Imprime un resumen final compacto del run (maximo 10 lineas).

    Args:
        cfg: Configuracion efectiva del run.
        meta_corpus: Metadatos del corpus (docs ok/procesados/excluidos).
        secciones: Stats de las etapas ejecutadas.
        tiempos: Segundos por etapa.
        total_seg: Segundos totales del run.
    """
    lineas = [
        (
            f"Resumen 28_c3_preparar (modo={cfg['modo']}, "
            f"smoke={cfg['smoke']}, limite={cfg['limite']})"
        )
    ]
    lineas.append(
        f"- docs: procesados {meta_corpus['docs_procesados']} de "
        f"{meta_corpus['docs_ok']} ok; excluidos: "
        f"{meta_corpus['excluidos_por_estado']}"
    )
    ner = secciones.get("ner")
    if ner:
        cob = ner["cobertura_pct"]
        lineas.append(
            f"- NER: CP={cob['cp']:.2f}% norma={cob['norma']:.2f}% "
            f"sent={cob['sent']:.2f}% ponente={cob['ponente']:.2f}% | "
            f"aristas={ner['n_aristas']} (sin self {ner['n_aristas_sin_self']})"
        )
        lineas.append(f"- NER top-5 arts CP (docs): {ner['top_arts_docs'][:5]}")
    etiquetas = secciones.get("etiquetas")
    if etiquetas:
        criterio = "SI" if etiquetas["criterio_95_asignadas"] else "NO"
        lineas.append(
            f"- Etiquetas: cita={etiquetas['via_cita_pct']:.2f}% "
            f"fallback={etiquetas['via_fallback_pct']:.2f}% "
            f"sin_area={etiquetas['via_sin_area_pct']:.2f}% | "
            f"asignadas={etiquetas['asignadas_pct']:.2f}% (criterio 95: {criterio})"
        )
    lemas = secciones.get("lemas")
    if lemas:
        lineas.append(
            f"- Lemas: docs={lemas['n_docs']} tipos={lemas['n_tipos_lemas']} "
            f"total={lemas['total_lemas']} media={lemas['media_lemas_doc']}/doc"
        )
    lineas.append(
        f"- Tiempos: ner={tiempos.get('ner', 0.0)}s "
        f"etiquetas={tiempos.get('etiquetas', 0.0)}s "
        f"lemas={tiempos.get('lemas', 0.0)}s total={total_seg:.1f}s"
    )
    lineas.append(f"- Salidas: {cfg['dir_datos']}")
    lineas.append(f"- Stats: {cfg['ruta_stats']} | log: {cfg['ruta_log']}")
    for linea in lineas:
        print(linea)


def main() -> int:
    """Orquesta las etapas pedidas y actualiza stats y resumen final.

    Returns:
        Codigo de salida (0 ante exito, 1 ante error con traceback).
    """
    args = parsear_argumentos()
    cfg = construir_config(args)
    configurar_semillas(CONFIG["semilla"])
    logger = configurar_logging(cfg["ruta_log"])
    inicio = time.time()
    try:
        logger.info(
            "Inicio 28_c3_preparar: modo=%s smoke=%s limite=%s semilla=%d",
            cfg["modo"],
            cfg["smoke"],
            cfg["limite"],
            CONFIG["semilla"],
        )
        df_corpus, meta_corpus = cargar_corpus_filtrado(cfg, logger)
        info_crudo = None
        if args.ner or args.lemas or args.todo:
            mod26 = cargar_modulo(
                cfg["ruta_mod_descarga"], "c3_mod26_descarga"
            )
            df_corpus = cargar_textos_crudos(df_corpus, cfg, mod26, logger)
            info_crudo = df_corpus.attrs.get("info_texto_crudo")
        secciones: dict = {}
        tiempos: dict = {}
        if args.ner or args.todo:
            canones = construir_canones(cfg, df_corpus, logger)
            mod03 = cargar_modulo(cfg["ruta_mod_ner"], "c3_mod03_entidades")
            mod03.CONFIG["ano_max"] = CONFIG["ano_max_citas"]
            logger.info(
                "NER: rango temporal de citas/normas extendido a 1992-%d",
                CONFIG["ano_max_citas"],
            )
            t0 = time.time()
            _, stats = etapa_ner(cfg, df_corpus, mod03, canones, logger)
            tiempos["ner"] = round(time.time() - t0, 2)
            secciones["ner"] = stats
        if args.etiquetas or args.todo:
            mod04 = cargar_modulo(
                cfg["ruta_mod_etiquetas"], "c3_mod04_etiquetas"
            )
            t0 = time.time()
            _, stats = etapa_etiquetas(cfg, mod04, logger)
            tiempos["etiquetas"] = round(time.time() - t0, 2)
            secciones["etiquetas"] = stats
        if args.lemas or args.todo:
            mod05 = cargar_modulo(cfg["ruta_mod_lemas"], "c3_mod05_lemas")
            t0 = time.time()
            _, stats = etapa_lemas(cfg, df_corpus, mod05, logger)
            tiempos["lemas"] = round(time.time() - t0, 2)
            secciones["lemas"] = stats
        total_seg = time.time() - inicio
        meta = {
            "script": "28_c3_preparar_corpus.py",
            "fecha": ahora_utc(),
            "modo": cfg["modo"],
            "smoke": cfg["smoke"],
            "docs_corpus": meta_corpus["docs_corpus"],
            "docs_ok": meta_corpus["docs_ok"],
            "docs_procesados": meta_corpus["docs_procesados"],
            "excluidos_por_estado": meta_corpus["excluidos_por_estado"],
            "limite": meta_corpus["limite"],
            "texto_crudo_ner_lemas": info_crudo,
        }
        actualizar_stats(cfg, secciones, meta, tiempos, logger)
        imprimir_resumen(cfg, meta_corpus, secciones, tiempos, total_seg)
        return 0
    except Exception:  # noqa: BLE001 - main captura todo para log + exit 1
        logger.error("Fallo 28_c3_preparar:\n%s", traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
