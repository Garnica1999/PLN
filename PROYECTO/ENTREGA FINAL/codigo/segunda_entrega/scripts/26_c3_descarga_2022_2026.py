"""26_c3_descarga_2022_2026.py
Fecha: 2026-10-04
Descripcion: Sprint C3 del ciclo v3 - descarga y extraccion del corpus
2022-2026 de la Corte Constitucional de Colombia para validacion temporal.

    Modos (mutuamente excluyentes):
    - --listar: consulta la API SODA keyless de datos.gov.co
      (dataset v2k4-2t8s, metadatos de sentencias) con paginacion
      ($limit/$offset), filtro fecha_sentencia > '2021-12-31' y orden
      canonico (fecha, sentencia). Normaliza cada registro (id interno =
      cita de relatoria, p. ej. "T-280/26"), construye la url_texto
      (https://www.corteconstitucional.gov.co/relatoria/{anio}/{TIPO}-
      {NUM}-{AA}.htm para T/C, con guion; y .../SU{NUM}-{AA}.htm para
      SU, sin guion: p. ej. SU020-22.htm; verificado por el revisor
      2026-10-04) y guarda data/corpus_2022_2026/metadatos.jsonl.
      Deduplica por cita y marca duplicados; filtra sentencia_tipo en
      {T, C, SU} (excluye autos).
    - --descargar: para cada registro sin HTML local valido, descarga el
      .htm con User-Agent identificable, delay entre peticiones
      (--delay, default 1.0 s), timeout 30 s, 3 reintentos con backoff
      exponencial + jitter y respeto de Retry-After si aparece. Guarda
      crudo en data/corpus_2022_2026/raw/{anio}/{TIPO}-{NUM}-{AA}.html.
      Detecta stubs (HTTP 200 con pagina de redireccion Angular de
      ~8.6 KB o marcas "redireccion"/"buscador-jurisprudencia"): no se
      guardan como HTML valido, se registran como stub y se reintentan
      una vez al final del run. Reanudable: salta HTML existente.
      Progreso/ETA cada 50 documentos.
    - --texto: extrae texto limpio del HTML con las convenciones E1/E2
      (reutiliza construir_texto_limpio de scripts/00_csv_a_parquet.py
      via importlib; si no fuera importable, replica las regex
      documentadas de la cadena E1). Decodificacion robusta
      (utf-8 -> cp1252 -> latin-1) y eliminacion de scripts, estilos y
      navegacion. Guarda data/corpus_2022_2026/corpus_2022_2026.parquet
      (id, tipo, anio, fecha, magistrado, expediente, texto,
      n_chars_texto, n_palabras_clean, url, estado) y actualiza stats
      (ok/texto_corto/stub/error, palabras totales, docs con texto
      > 1.000 chars). El estado por fila es "texto_corto" cuando el
      texto limpio queda por debajo del umbral de 1.000 chars.
    - --todo: listar -> descargar -> texto.

    Salidas: data/corpus_2022_2026/ (metadatos.jsonl, raw/, parquet,
    estado_descarga.json), resultados/c3_descarga_stats.json y
    experimentos/26_c3_descarga.log. No se sobrescribe una salida previa
    sin --forzar (reanudable). --smoke usa 5 registros y escribe todo en
    /tmp/opencode/c3_smoke/. --limite N acota descarga/texto (en smoke
    tambien acota el listado). Sin subprocesos: scraping secuencial y
    educado. Memoria consciente (un documento en memoria a la vez).

Estado del arte aplicado: API SODA 2.1 (Socrata) keyless para metadatos
oficiales de relatoria; descarga secuencial educada (User-Agent
identificable, delay, backoff exponencial con jitter, Retry-After,
deteccion de stubs de redireccion SPA); extraccion E1/E2 exacta del
proyecto (minusculas, proteccion de enie, NFD sin diacriticos, filtro
[a-z], tokens de 1 caracter no gramaticales y abreviatura art).

Uso:
    /home/carlos/venv/bin/python scripts/26_c3_descarga_2022_2026.py --smoke --todo
    /home/carlos/venv/bin/python scripts/26_c3_descarga_2022_2026.py --todo
    /home/carlos/venv/bin/python scripts/26_c3_descarga_2022_2026.py --descargar
    /home/carlos/venv/bin/python scripts/26_c3_descarga_2022_2026.py --texto --forzar
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
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from bs4 import BeautifulSoup

CONFIG = {
    "semilla": 42,
    "socrata_url": "https://www.datos.gov.co/resource/v2k4-2t8s.json",
    "socrata_where": "fecha_sentencia > '2021-12-31'",
    "socrata_order": "fecha_sentencia ASC, sentencia ASC",
    "socrata_limit": 5000,
    "socrata_max_registros": 50000,
    "base_relatoria": "https://www.corteconstitucional.gov.co/relatoria",
    "tipos_validos": ("T", "C", "SU"),
    "user_agent": (
        "Juris-PLN-C3/1.0 (proyecto academico de investigacion; "
        "descarga secuencial educada)"
    ),
    "delay": 1.0,
    "timeout": 30.0,
    "reintentos": 3,
    "backoff_base": 2.0,
    "backoff_max": 120.0,
    "backoff_jitter": 0.5,
    "stub_max_bytes": 10000,
    "stub_bytes_muestra": 20000,
    "stub_marcas": ("redireccion", "buscador-jurisprudencia"),
    "etiquetas_ruido": ("script", "style", "noscript", "nav", "header", "footer"),
    "umbral_texto_min_chars": 1000,
    "max_duplicados_listados": 50,
    "log_cada": 50,
    "log_cada_texto": 100,
}

PATRON_CITA = re.compile(r"([TCS]U?)-(\d+)([A-Z]?)/(\d{2,4})")
PLACEHOLDER_ENIE = "\u0001"
ABREV_CITA = {"art"}

ESQUEMA_PARQUET = pa.schema(
    [
        ("id", pa.string()),
        ("tipo", pa.string()),
        ("anio", pa.int32()),
        ("fecha", pa.string()),
        ("magistrado", pa.string()),
        ("expediente", pa.string()),
        ("texto", pa.string()),
        ("n_chars_texto", pa.int64()),
        ("n_palabras_clean", pa.int64()),
        ("url", pa.string()),
        ("estado", pa.string()),
    ]
)


# ===== SECCION 1: CONFIGURACION, CLI Y UTILIDADES =====


def parsear_argumentos() -> argparse.Namespace:
    """Define y parsea los argumentos de linea de comandos.

    Returns:
        Namespace con modo, smoke, forzar, limite, delay, timeout y
        reintentos validados.

    Raises:
        SystemExit: Si argparse detecta un modo ausente o valores invalidos.
    """
    parser = argparse.ArgumentParser(
        description=(
            "C3: descarga y extraccion del corpus 2022-2026 de la Corte "
            "Constitucional para validacion temporal."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python scripts/26_c3_descarga_2022_2026.py --smoke --todo\n"
            "  python scripts/26_c3_descarga_2022_2026.py --todo\n"
            "  python scripts/26_c3_descarga_2022_2026.py --descargar --limite 10\n"
        ),
    )
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument(
        "--listar", action="store_true", help="Consulta Socrata y guarda metadatos."
    )
    grupo.add_argument(
        "--descargar", action="store_true", help="Descarga los HTML crudos."
    )
    grupo.add_argument(
        "--texto", action="store_true", help="Extrae texto limpio a parquet."
    )
    grupo.add_argument(
        "--todo", action="store_true", help="Ejecuta listar, descargar y texto."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Prueba con 5 registros y salidas en /tmp/opencode/c3_smoke/.",
    )
    parser.add_argument(
        "--forzar",
        action="store_true",
        help="Rehace las salidas del modo pedido (por defecto son reanudables).",
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help="Maximo de registros a descargar/procesar (pruebas parciales).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Segundos de espera entre peticiones HTTP (default 1.0).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout por peticion HTTP en segundos (default 30).",
    )
    parser.add_argument(
        "--reintentos",
        type=int,
        default=3,
        help="Intentos por documento ante error de red/HTTP (default 3).",
    )
    args = parser.parse_args()
    if args.limite is not None and args.limite < 1:
        parser.error("--limite debe ser >= 1")
    if args.delay < 0:
        parser.error("--delay debe ser >= 0")
    if args.timeout <= 0:
        parser.error("--timeout debe ser > 0")
    if args.reintentos < 1:
        parser.error("--reintentos debe ser >= 1")
    return args


def construir_config(args: argparse.Namespace) -> dict:
    """Construye la configuracion efectiva del run (rutas y parametros).

    Args:
        args: Namespace parseado con los flags del CLI.

    Returns:
        Diccionario de configuracion con rutas absolutas; en modo smoke
        la raiz es /tmp/opencode/c3_smoke y el limite por defecto es 5.
    """
    base = Path(__file__).resolve().parent.parent
    raiz = Path("/tmp/opencode/c3_smoke") if args.smoke else base
    dir_datos = raiz / "data" / "corpus_2022_2026"
    if args.limite is not None:
        limite = args.limite
    else:
        limite = 5 if args.smoke else None
    cfg = {
        **CONFIG,
        "smoke": args.smoke,
        "forzar": args.forzar,
        "limite": limite,
        "delay": args.delay,
        "timeout": args.timeout,
        "reintentos": args.reintentos,
        "raiz": raiz,
        "dir_datos": dir_datos,
        "dir_raw": dir_datos / "raw",
        "ruta_metadatos": dir_datos / "metadatos.jsonl",
        "ruta_estado": dir_datos / "estado_descarga.json",
        "ruta_parquet": dir_datos / "corpus_2022_2026.parquet",
        "ruta_stats": raiz / "resultados" / "c3_descarga_stats.json",
        "ruta_log": raiz / "experimentos" / "26_c3_descarga.log",
        "ruta_modulo_e1": base / "scripts" / "00_csv_a_parquet.py",
    }
    return cfg


def configurar_logging(ruta_log: Path) -> logging.Logger:
    """Configura el logger con salida a consola y a archivo (append).

    Args:
        ruta_log: Ruta del archivo de log (se crea su carpeta padre).

    Returns:
        Logger configurado con dos handlers (consola y archivo).
    """
    ruta_log.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("c3_descarga")
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


def formatear_duracion(segundos: float) -> str:
    """Formatea una duracion en segundos a texto legible.

    Args:
        segundos: Duracion en segundos.

    Returns:
        Cadena con segundos, minutos u horas segun magnitud.
    """
    if segundos < 60:
        return f"{segundos:.0f} s"
    if segundos < 3600:
        return f"{segundos / 60:.1f} min"
    return f"{segundos / 3600:.2f} h"


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


def escribir_json(ruta: Path, datos: dict, indent: int | None = 2) -> None:
    """Escribe un JSON de forma atomica (temporal + replace).

    Args:
        ruta: Ruta destino del JSON.
        datos: Objeto serializable a escribir.
        indent: Indentacion del JSON (None para compacto).
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = Path(str(ruta) + ".tmp")
    with open(temporal, "w", encoding="utf-8") as manejador:
        json.dump(datos, manejador, ensure_ascii=False, indent=indent)
    os.replace(temporal, ruta)


def escribir_jsonl(ruta: Path, registros: list) -> None:
    """Escribe una lista de diccionarios como JSON Lines de forma atomica.

    Args:
        ruta: Ruta destino del .jsonl.
        registros: Lista de diccionarios (un registro por linea).
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = Path(str(ruta) + ".tmp")
    with open(temporal, "w", encoding="utf-8") as manejador:
        manejador.writelines(
            json.dumps(registro, ensure_ascii=False) + "\n"
            for registro in registros
        )
    os.replace(temporal, ruta)


def cargar_metadatos(cfg: dict) -> list:
    """Carga los metadatos normalizados desde el JSON Lines.

    Args:
        cfg: Configuracion del run.

    Returns:
        Lista de registros de metadatos.

    Raises:
        FileNotFoundError: Si el JSON Lines no existe (falta --listar).
        ValueError: Si una linea no es JSON valido.
    """
    ruta = cfg["ruta_metadatos"]
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existen metadatos en {ruta}; ejecute primero --listar."
        )
    registros = []
    with open(ruta, "r", encoding="utf-8") as manejador:
        for numero_linea, linea in enumerate(manejador, start=1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                registros.append(json.loads(linea))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Linea {numero_linea} invalida en {ruta}: {exc}"
                ) from exc
    return registros


def cargar_estado(cfg: dict) -> dict:
    """Carga el estado por documento de la fase de descarga.

    Args:
        cfg: Configuracion del run.

    Returns:
        Diccionario id -> estado (ok/stub/error, intentos, bytes, razon).
    """
    estado = leer_json(cfg["ruta_estado"], {})
    return estado if isinstance(estado, dict) else {}


def guardar_estado(cfg: dict, estado: dict) -> None:
    """Persiste el estado por documento de la fase de descarga.

    Args:
        cfg: Configuracion del run.
        estado: Diccionario id -> estado a guardar.
    """
    escribir_json(cfg["ruta_estado"], estado, indent=None)


def actualizar_stats(cfg: dict, seccion: str, datos: dict, logger: logging.Logger) -> None:
    """Actualiza una seccion del JSON acumulativo de estadisticas.

    Args:
        cfg: Configuracion del run.
        seccion: Nombre de la seccion (listado, descarga o texto).
        datos: Contenido de la seccion.
        logger: Logger de ejecucion.
    """
    stats = leer_json(cfg["ruta_stats"], {})
    if not isinstance(stats, dict):
        stats = {}
    stats["generado_utc"] = ahora_utc()
    stats[seccion] = datos
    escribir_json(cfg["ruta_stats"], stats)
    logger.info("Stats actualizados: seccion '%s' -> %s", seccion, cfg["ruta_stats"])


# ===== SECCION 2: LISTADO DE METADATOS (--listar) =====


def peticion_con_reintentos(
    sesion: requests.Session,
    url: str,
    cfg: dict,
    logger: logging.Logger,
    params: dict | None = None,
    max_intentos: int | None = None,
) -> tuple:
    """Ejecuta un GET con reintentos, backoff exponencial y Retry-After.

    Args:
        sesion: Sesion requests con User-Agent configurado.
        url: URL a consultar.
        cfg: Configuracion del run.
        logger: Logger de ejecucion.
        params: Parametros de query string (Socrata) o None.
        max_intentos: Override del numero de intentos (default cfg).

    Returns:
        Tupla (respuesta | None, intentos_usados, ultimo_error). La
        respuesta es None si se agotaron los intentos.
    """
    intentos = max_intentos if max_intentos is not None else cfg["reintentos"]
    error = ""
    for intento in range(1, intentos + 1):
        respuesta = None
        try:
            respuesta = sesion.get(url, params=params, timeout=cfg["timeout"])
        except requests.RequestException as exc:
            error = f"{type(exc).__name__}: {exc}"
        else:
            if respuesta.status_code == 200:
                return respuesta, intento, ""
            error = f"HTTP {respuesta.status_code}"
        if intento < intentos:
            espera = calcular_espera(intento, respuesta, cfg)
            logger.warning(
                "Reintento %d/%d en %.1f s para %s (%s)",
                intento,
                intentos,
                espera,
                url,
                error,
            )
            time.sleep(espera)
    return None, intentos, error


def retry_after_segundos(valor: str) -> float | None:
    """Interpreta el encabezado Retry-After (segundos o fecha HTTP).

    Args:
        valor: Valor crudo del encabezado Retry-After.

    Returns:
        Segundos de espera, o None si el valor no es interpretable.
    """
    try:
        return max(0.0, float(valor))
    except ValueError:
        pass
    try:
        fecha = parsedate_to_datetime(valor)
    except (TypeError, ValueError):
        return None
    if fecha is None:
        return None
    if fecha.tzinfo is None:
        fecha = fecha.replace(tzinfo=timezone.utc)
    return max(0.0, (fecha - datetime.now(timezone.utc)).total_seconds())


def calcular_espera(intento: int, respuesta, cfg: dict) -> float:
    """Calcula la espera del reintento: backoff + jitter + Retry-After.

    Args:
        intento: Numero de intento fallido (1-based).
        respuesta: Respuesta HTTP del intento fallido (o None).
        cfg: Configuracion del run.

    Returns:
        Segundos de espera, acotados por cfg["backoff_max"].
    """
    espera = cfg["backoff_base"] * (2 ** (intento - 1))
    espera += random.uniform(0.0, cfg["backoff_jitter"])
    if respuesta is not None:
        valor = respuesta.headers.get("Retry-After")
        if valor:
            segundos = retry_after_segundos(valor)
            if segundos is not None:
                espera = max(espera, segundos)
    return min(espera, cfg["backoff_max"])


def consultar_socrata(
    sesion: requests.Session,
    cfg: dict,
    logger: logging.Logger,
    limite_total: int | None = None,
) -> list:
    """Consulta la API SODA paginando hasta agotar los registros.

    Args:
        sesion: Sesion requests.
        cfg: Configuracion del run.
        logger: Logger de ejecucion.
        limite_total: Corte opcional del total (usado por el smoke).

    Returns:
        Lista de registros crudos de la API.

    Raises:
        RuntimeError: Si la API falla o responde algo que no es JSON.
    """
    registros = []
    offset = 0
    tam_pagina = cfg["socrata_limit"]
    if limite_total is not None:
        tam_pagina = min(tam_pagina, limite_total)
    while True:
        params = {
            "$where": cfg["socrata_where"],
            "$order": cfg["socrata_order"],
            "$limit": tam_pagina,
            "$offset": offset,
        }
        respuesta, intentos, error = peticion_con_reintentos(
            sesion, cfg["socrata_url"], cfg, logger, params=params
        )
        if respuesta is None:
            raise RuntimeError(f"La API Socrata no respondio: {error}")
        try:
            lote = respuesta.json()
        except ValueError as exc:
            raise RuntimeError(f"Respuesta Socrata no es JSON: {exc}") from exc
        if not isinstance(lote, list):
            raise TypeError(f"Respuesta Socrata inesperada: {type(lote)}")
        registros.extend(lote)
        logger.info(
            "Socrata: offset=%d lote=%d acumulado=%d (intentos=%d)",
            offset,
            len(lote),
            len(registros),
            intentos,
        )
        if len(lote) < tam_pagina:
            break
        if limite_total is not None and len(registros) >= limite_total:
            break
        offset += len(lote)
        if offset > cfg["socrata_max_registros"]:
            raise RuntimeError("La paginacion Socrata excedio el maximo esperado")
        time.sleep(cfg["delay"])
    if limite_total is not None:
        registros = registros[:limite_total]
    return registros


def normalizar_registros(
    crudos: list, cfg: dict, logger: logging.Logger
) -> tuple:
    """Normaliza los registros Socrata y deduplica por cita de relatoria.

    Args:
        crudos: Registros crudos de la API.
        cfg: Configuracion del run.
        logger: Logger de ejecucion.

    Returns:
        Tupla (registros normalizados ordenados por fecha e id, resumen
        con conteos de deduplicacion y descartes).
    """
    unicos = {}
    duplicados = []
    descartados_tipo = 0
    descartados_formato = 0
    inconsistencias_tipo = 0
    for crudo in crudos:
        tipo_api = str(crudo.get("sentencia_tipo") or "").strip().upper()
        if tipo_api not in cfg["tipos_validos"]:
            descartados_tipo += 1
            continue
        cita = str(crudo.get("sentencia") or "").strip()
        coincidencia = PATRON_CITA.fullmatch(cita)
        if coincidencia is None:
            descartados_formato += 1
            logger.warning("Cita no parseable, se descarta: %r", cita)
            continue
        tipo, numero, sufijo, aa = coincidencia.groups()
        if cita in unicos:
            duplicados.append(cita)
            continue
        fecha_completa = str(crudo.get("fecha_sentencia") or "")
        if len(fecha_completa) < 10:
            descartados_formato += 1
            logger.warning("Fecha invalida para %s: %r", cita, fecha_completa)
            continue
        fecha = fecha_completa[:10]
        anio = int(fecha[:4])
        if tipo_api != tipo:
            inconsistencias_tipo += 1
        # Las URLs de SU van sin guion entre el tipo y el numero
        # (SU020-22.htm); T y C si lo llevan (T-280-26.htm, C-237A-22.htm).
        separador = "" if tipo == "SU" else "-"
        url_texto = (
            f"{cfg['base_relatoria']}/{anio}/"
            f"{tipo}{separador}{numero}{sufijo}-{aa}.htm"
        )
        unicos[cita] = {
            "id": cita,
            "tipo": tipo,
            "numero": int(numero),
            "aa": aa,
            "anio": anio,
            "fecha": fecha,
            "magistrado": str(crudo.get("magistrado_a") or "").strip(),
            "expediente": str(crudo.get("expediente_numero") or "").strip(),
            "proceso": str(crudo.get("proceso") or "").strip(),
            "sala": str(crudo.get("sala") or "").strip(),
            "url_texto": url_texto,
        }
    registros = sorted(unicos.values(), key=lambda r: (r["fecha"], r["id"]))
    resumen = {
        "n_api": len(crudos),
        "n_normalizados": len(registros),
        "n_duplicados": len(duplicados),
        "ids_duplicados": duplicados[: cfg["max_duplicados_listados"]],
        "n_descartados_tipo": descartados_tipo,
        "n_descartados_formato": descartados_formato,
        "n_inconsistencias_tipo": inconsistencias_tipo,
    }
    return registros, resumen


def calcular_stats_listado(
    registros: list, resumen: dict, cfg: dict, reutilizado: bool, tiempo: float
) -> dict:
    """Construye la seccion 'listado' del JSON de estadisticas.

    Args:
        registros: Registros normalizados.
        resumen: Conteos de normalizacion/deduplicacion.
        cfg: Configuracion del run.
        reutilizado: True si se reutilizo el JSON Lines existente.
        tiempo: Tiempo de la fase en segundos.

    Returns:
        Diccionario de estadisticas del listado.
    """
    por_tipo = Counter(registro["tipo"] for registro in registros)
    por_anio = Counter(str(registro["anio"]) for registro in registros)
    fechas = [registro["fecha"] for registro in registros]
    return {
        "reutilizado": reutilizado,
        "endpoint": cfg["socrata_url"],
        "where": cfg["socrata_where"],
        **resumen,
        "n_por_tipo": dict(sorted(por_tipo.items())),
        "n_por_anio": dict(sorted(por_anio.items())),
        "fecha_min": min(fechas) if fechas else None,
        "fecha_max": max(fechas) if fechas else None,
        "ruta_metadatos": str(cfg["ruta_metadatos"]),
        "tiempo_seg": round(tiempo, 2),
        "timestamp_utc": ahora_utc(),
    }


def listar(sesion: requests.Session, cfg: dict, logger: logging.Logger) -> list:
    """Ejecuta el modo --listar: consulta, normaliza y guarda metadatos.

    Args:
        sesion: Sesion requests.
        cfg: Configuracion del run.
        logger: Logger de ejecucion.

    Returns:
        Lista de registros normalizados.
    """
    t0 = time.time()
    if cfg["ruta_metadatos"].exists() and not cfg["forzar"]:
        registros = cargar_metadatos(cfg)
        resumen = {
            "n_api": None,
            "n_normalizados": len(registros),
            "n_duplicados": None,
            "ids_duplicados": [],
            "n_descartados_tipo": None,
            "n_descartados_formato": None,
            "n_inconsistencias_tipo": None,
        }
        reutilizado = True
        logger.info(
            "Metadatos existentes reutilizados: %d registros (%s); "
            "use --forzar para re-consultar",
            len(registros),
            cfg["ruta_metadatos"],
        )
    else:
        limite_total = cfg["limite"] if cfg["smoke"] else None
        crudos = consultar_socrata(sesion, cfg, logger, limite_total=limite_total)
        registros, resumen = normalizar_registros(crudos, cfg, logger)
        escribir_jsonl(cfg["ruta_metadatos"], registros)
        reutilizado = False
        logger.info(
            "Metadatos guardados: %d registros en %s (duplicados=%d "
            "descartados_tipo=%d descartados_formato=%d)",
            len(registros),
            cfg["ruta_metadatos"],
            resumen["n_duplicados"],
            resumen["n_descartados_tipo"],
            resumen["n_descartados_formato"],
        )
    stats = calcular_stats_listado(
        registros, resumen, cfg, reutilizado, time.time() - t0
    )
    actualizar_stats(cfg, "listado", stats, logger)
    logger.info(
        "Listado: %d sentencias (T=%d C=%d SU=%d) %s..%s en %s",
        len(registros),
        stats["n_por_tipo"].get("T", 0),
        stats["n_por_tipo"].get("C", 0),
        stats["n_por_tipo"].get("SU", 0),
        stats["fecha_min"],
        stats["fecha_max"],
        formatear_duracion(time.time() - t0),
    )
    return registros


# ===== SECCION 3: DESCARGA DE HTML CRUDO (--descargar) =====


def ruta_html(cfg: dict, registro: dict) -> Path:
    """Calcula la ruta local del HTML crudo de un registro.

    Args:
        cfg: Configuracion del run.
        registro: Registro normalizado de metadatos.

    Returns:
        Ruta raw/{anio}/{TIPO}-{NUM}-{AA}.html.
    """
    nombre = Path(registro["url_texto"]).stem + ".html"
    return cfg["dir_raw"] / str(registro["anio"]) / nombre


def html_valido_en_disco(registro: dict, cfg: dict) -> bool:
    """Indica si el HTML del registro ya existe y supera el umbral de stub.

    Args:
        registro: Registro normalizado de metadatos.
        cfg: Configuracion del run.

    Returns:
        True si el archivo existe con tamano >= stub_max_bytes.
    """
    ruta = ruta_html(cfg, registro)
    return ruta.exists() and ruta.stat().st_size >= cfg["stub_max_bytes"]


def escribir_html(ruta: Path, contenido: bytes) -> None:
    """Escribe el HTML crudo de forma atomica (temporal + os.replace).

    Un kill a mitad de escritura deja solo el .tmp, que el resume nunca
    acepta como HTML valido (solo mira archivos .html completos).

    Args:
        ruta: Ruta destino del .html.
        contenido: Bytes crudos del documento.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = Path(str(ruta) + ".tmp")
    temporal.write_bytes(contenido)
    os.replace(temporal, ruta)


def es_stub(contenido: bytes, cfg: dict) -> tuple:
    """Detecta la pagina stub de redireccion SPA servida con HTTP 200.

    Criterios (verificados 2026-10-04): los documentos reales pesan mas
    de 100 KB; las URLs inexistentes devuelven un shell Angular de
    ~8.6 KB. Se marca stub si el tamano es menor a stub_max_bytes o si
    la muestra inicial contiene alguna marca de redireccion.

    Args:
        contenido: Bytes de la respuesta HTTP.
        cfg: Configuracion del run.

    Returns:
        Tupla (es_stub, razon).
    """
    if len(contenido) < cfg["stub_max_bytes"]:
        return True, f"tamano {len(contenido)}B < {cfg['stub_max_bytes']}B"
    muestra = (
        contenido[: cfg["stub_bytes_muestra"]]
        .decode("latin-1", errors="replace")
        .lower()
    )
    for marca in cfg["stub_marcas"]:
        if marca in muestra:
            return True, f"marca de redireccion '{marca}'"
    return False, ""


def descargar_documento(
    sesion: requests.Session,
    registro: dict,
    cfg: dict,
    logger: logging.Logger,
    max_intentos: int | None = None,
) -> dict:
    """Descarga el HTML de un registro y clasifica el resultado.

    Args:
        sesion: Sesion requests.
        registro: Registro normalizado de metadatos.
        cfg: Configuracion del run.
        logger: Logger de ejecucion.
        max_intentos: Override del numero de intentos.

    Returns:
        Diccionario con estado (ok/stub/error), intentos, bytes, razon
        y contenido (solo si estado == "ok").
    """
    respuesta, intentos, error = peticion_con_reintentos(
        sesion,
        registro["url_texto"],
        cfg,
        logger,
        max_intentos=max_intentos,
    )
    if respuesta is None:
        return {
            "estado": "error",
            "intentos": intentos,
            "bytes": 0,
            "razon": error,
            "contenido": None,
        }
    stub, razon = es_stub(respuesta.content, cfg)
    if stub:
        return {
            "estado": "stub",
            "intentos": intentos,
            "bytes": len(respuesta.content),
            "razon": razon,
            "contenido": None,
        }
    return {
        "estado": "ok",
        "intentos": intentos,
        "bytes": len(respuesta.content),
        "razon": "",
        "contenido": respuesta.content,
    }


def calcular_acumulado_descarga(cfg: dict, registros: list) -> dict:
    """Calcula el estado acumulado del corpus en disco.

    Args:
        cfg: Configuracion del run.
        registros: Todos los registros de metadatos (sin recortes).

    Returns:
        Diccionario con validos, stub, error, pendientes y bytes.
    """
    estado = cargar_estado(cfg)
    n_validos = 0
    bytes_disco = 0
    n_stub = 0
    n_error = 0
    for registro in registros:
        if html_valido_en_disco(registro, cfg):
            n_validos += 1
            bytes_disco += ruta_html(cfg, registro).stat().st_size
            continue
        estado_doc = estado.get(registro["id"], {}).get("estado")
        if estado_doc == "stub":
            n_stub += 1
        elif estado_doc == "error":
            n_error += 1
    return {
        "n_registros": len(registros),
        "n_html_validos": n_validos,
        "n_stub": n_stub,
        "n_error": n_error,
        "n_pendientes": len(registros) - n_validos - n_stub - n_error,
        "bytes_html_en_disco": bytes_disco,
    }


def descargar(sesion: requests.Session, cfg: dict, logger: logging.Logger) -> None:
    """Ejecuta el modo --descargar: baja los HTML pendientes y los guarda.

    Args:
        sesion: Sesion requests.
        cfg: Configuracion del run.
        logger: Logger de ejecucion.
    """
    todos = cargar_metadatos(cfg)
    procesables = todos[: cfg["limite"]] if cfg["limite"] is not None else todos
    estado = cargar_estado(cfg)
    pendientes = []
    omitidos = 0
    for registro in procesables:
        if not cfg["forzar"] and html_valido_en_disco(registro, cfg):
            omitidos += 1
            previo = estado.get(registro["id"], {})
            if previo.get("estado") != "ok":
                ruta = ruta_html(cfg, registro)
                estado[registro["id"]] = {
                    "estado": "ok",
                    "intentos": previo.get("intentos", 0),
                    "bytes": ruta.stat().st_size,
                    "razon": "",
                    "ultimo_intento": previo.get("ultimo_intento", ahora_utc()),
                }
        else:
            pendientes.append(registro)
    logger.info(
        "Descarga: %d pendientes, %d omitidos (HTML valido existente), "
        "limite=%s, delay=%.1f s",
        len(pendientes),
        omitidos,
        cfg["limite"],
        cfg["delay"],
    )
    t0 = time.time()
    contadores = Counter()
    bytes_descargados = 0
    reintentos_red = 0
    stubs_del_run = []
    for indice, registro in enumerate(pendientes, start=1):
        resultado = descargar_documento(sesion, registro, cfg, logger)
        reintentos_red += max(0, resultado["intentos"] - 1)
        contadores[resultado["estado"]] += 1
        if resultado["estado"] == "ok":
            escribir_html(ruta_html(cfg, registro), resultado["contenido"])
            bytes_descargados += resultado["bytes"]
        elif resultado["estado"] == "stub":
            stubs_del_run.append(registro)
        previo = estado.get(registro["id"], {})
        estado[registro["id"]] = {
            "estado": resultado["estado"],
            "intentos": previo.get("intentos", 0) + resultado["intentos"],
            "bytes": resultado["bytes"],
            "razon": resultado["razon"],
            "ultimo_intento": ahora_utc(),
        }
        if indice % cfg["log_cada"] == 0 or indice == len(pendientes):
            transcurrido = time.time() - t0
            eta = transcurrido / indice * (len(pendientes) - indice)
            logger.info(
                "Descarga %d/%d (%.1f%%): ok=%d stub=%d error=%d | "
                "%.2f doc/s | ETA %s",
                indice,
                len(pendientes),
                100.0 * indice / len(pendientes),
                contadores["ok"],
                contadores["stub"],
                contadores["error"],
                indice / transcurrido if transcurrido > 0 else 0.0,
                formatear_duracion(eta),
            )
        if indice < len(pendientes):
            time.sleep(cfg["delay"])
    reintento_final_ok = 0
    if stubs_del_run:
        logger.info(
            "Reintento final de %d stubs (una vez, sin backoff)",
            len(stubs_del_run),
        )
    for registro in stubs_del_run:
        time.sleep(cfg["delay"])
        resultado = descargar_documento(
            sesion, registro, cfg, logger, max_intentos=1
        )
        previo = estado.get(registro["id"], {})
        if resultado["estado"] == "ok":
            escribir_html(ruta_html(cfg, registro), resultado["contenido"])
            bytes_descargados += resultado["bytes"]
            reintento_final_ok += 1
            estado[registro["id"]] = {
                "estado": "ok",
                "intentos": previo.get("intentos", 0) + 1,
                "bytes": resultado["bytes"],
                "razon": "ok en reintento final",
                "ultimo_intento": ahora_utc(),
            }
        else:
            razon = resultado["razon"]
            if resultado["estado"] == "error":
                razon = f"reintento final fallido: {resultado['razon']}"
            estado[registro["id"]] = {
                **previo,
                "estado": "stub",
                "intentos": previo.get("intentos", 0) + 1,
                "razon": razon,
                "ultimo_intento": ahora_utc(),
            }
    guardar_estado(cfg, estado)
    tiempo = time.time() - t0
    acumulado = calcular_acumulado_descarga(cfg, todos)
    stats = {
        "run": {
            "intentados": len(pendientes),
            "ok": contadores["ok"],
            "stub": contadores["stub"],
            "error": contadores["error"],
            "omitidos": omitidos,
            "reintentos_red": reintentos_red,
            "stubs_reintentados": len(stubs_del_run),
            "reintento_final_ok": reintento_final_ok,
            "bytes_descargados": bytes_descargados,
            "delay_seg": cfg["delay"],
            "tiempo_seg": round(tiempo, 2),
        },
        "acumulado": acumulado,
        "timestamp_utc": ahora_utc(),
    }
    actualizar_stats(cfg, "descarga", stats, logger)
    logger.info(
        "Descarga terminada: ok=%d stub=%d error=%d omitidos=%d | "
        "validos en disco=%d/%d | %.1f MB nuevos | %s",
        contadores["ok"],
        contadores["stub"],
        contadores["error"],
        omitidos,
        acumulado["n_html_validos"],
        acumulado["n_registros"],
        bytes_descargados / 1e6,
        formatear_duracion(tiempo),
    )


# ===== SECCION 4: EXTRACCION DE TEXTO LIMPIO (--texto) =====


def normalizar_fallback(texto: str, con_digitos: bool) -> str:
    """Replica de normalizar() de la cadena E1 (solo si falla el import).

    Args:
        texto: Cadena original.
        con_digitos: True conserva digitos; False los elimina.

    Returns:
        Cadena normalizada (minusculas, sin diacriticos).
    """
    cadena = texto.lower().replace("\u00f1", PLACEHOLDER_ENIE)
    cadena = unicodedata.normalize("NFD", cadena)
    cadena = re.sub(r"[\u0300-\u036f]", "", cadena)
    cadena = cadena.replace(PLACEHOLDER_ENIE, "\u00f1")
    patron = r"[^a-z\u00f10-9\s]" if con_digitos else r"[^a-z\u00f1\s]"
    cadena = re.sub(patron, " ", cadena)
    return re.sub(r"\s+", " ", cadena).strip()


def construir_texto_limpio_fallback(texto_crudo: str, stopwords: set) -> str:
    """Replica la cadena E1/E2 completa (normalizar, limpiar, filtrar).

    Args:
        texto_crudo: Texto original de la sentencia.
        stopwords: Stopwords gramaticales normalizadas.

    Returns:
        Texto limpio final (tokens filtrados unidos por espacios).
    """
    normalizado = normalizar_fallback(texto_crudo, con_digitos=True)
    limpio = re.sub(r"\s+", " ", re.sub(r"\d", " ", normalizado)).strip()
    conservados = []
    for token in limpio.split():
        if token in ABREV_CITA:
            continue
        if len(token) == 1 and token not in stopwords:
            continue
        conservados.append(token)
    return " ".join(conservados)


def stopwords_fallback(logger: logging.Logger) -> set:
    """Construye las stopwords gramaticales normalizadas via NLTK.

    Args:
        logger: Logger de ejecucion.

    Returns:
        Set de stopwords normalizadas (vacio si NLTK no esta disponible).
    """
    try:
        from nltk.corpus import stopwords

        conjunto = set()
        for palabra in stopwords.words("spanish"):
            normalizada = normalizar_fallback(palabra, con_digitos=False)
            if re.fullmatch(r"[a-z\u00f1]+", normalizada):
                conjunto.add(normalizada)
        return conjunto
    except Exception as exc:  # noqa: BLE001 - portabilidad del fallback
        logger.warning("NLTK no disponible para el fallback (%s)", exc)
        return set()


def cargar_limpiador(cfg: dict, logger: logging.Logger) -> tuple:
    """Carga la funcion de limpieza E1/E2 (importlib o fallback replicado).

    Args:
        cfg: Configuracion del run.
        logger: Logger de ejecucion.

    Returns:
        Tupla (funcion texto_crudo -> texto_limpio, origen).
    """
    ruta = cfg["ruta_modulo_e1"]
    try:
        especificacion = importlib.util.spec_from_file_location(
            "c3_modulo_e1", ruta
        )
        if especificacion is None or especificacion.loader is None:
            raise ImportError(f"no se pudo cargar {ruta}")
        modulo = importlib.util.module_from_spec(especificacion)
        especificacion.loader.exec_module(modulo)
        stopwords = modulo.construir_stopwords_gramaticales()
        logger.info(
            "Limpiador E1/E2 importado de %s (%d stopwords)",
            ruta,
            len(stopwords),
        )

        def limpiar(texto_crudo: str) -> str:
            return modulo.construir_texto_limpio(texto_crudo, stopwords)

        return limpiar, "importado_00_csv_a_parquet"
    except Exception as exc:  # noqa: BLE001 - se replica la cadena E1
        logger.warning(
            "No se pudo importar el limpiador E1 (%s); se replica la "
            "cadena de regex documentada",
            exc,
        )
        stopwords = stopwords_fallback(logger)

        def limpiar(texto_crudo: str) -> str:
            return construir_texto_limpio_fallback(texto_crudo, stopwords)

        return limpiar, "fallback_regex_documentadas"


def decodificar_html(datos: bytes) -> str:
    """Decodifica HTML de forma robusta: utf-8, cp1252 y latin-1.

    Las paginas reales de relatoria fallan utf-8 estricto (byte 0xED) y
    decodifican correctamente con cp1252/latin-1 (verificado 2026-10-04).

    Args:
        datos: Bytes crudos del HTML.

    Returns:
        Texto decodificado.
    """
    for codificacion in ("utf-8", "cp1252"):
        try:
            return datos.decode(codificacion)
        except UnicodeDecodeError:
            continue
    return datos.decode("latin-1")


def extraer_texto_html(ruta: Path, cfg: dict) -> str:
    """Extrae el texto visible del HTML eliminando ruido de navegacion.

    Args:
        ruta: Ruta del HTML crudo.
        cfg: Configuracion del run (etiquetas de ruido).

    Returns:
        Texto visible plano (sin scripts, estilos ni navegacion).
    """
    sopa = BeautifulSoup(decodificar_html(ruta.read_bytes()), "lxml")
    for etiqueta in cfg["etiquetas_ruido"]:
        for nodo in sopa.find_all(etiqueta):
            nodo.decompose()
    return sopa.get_text(" ", strip=True)


def construir_fila(registro: dict, texto: str, estado: str) -> dict:
    """Construye una fila del parquet con el esquema documentado.

    Args:
        registro: Registro normalizado de metadatos.
        texto: Texto limpio E1/E2 (vacio si el estado no es ok).
        estado: ok, stub o error.

    Returns:
        Diccionario con las columnas del parquet.
    """
    return {
        "id": registro["id"],
        "tipo": registro["tipo"],
        "anio": int(registro["anio"]),
        "fecha": registro["fecha"],
        "magistrado": registro["magistrado"],
        "expediente": registro["expediente"],
        "texto": texto,
        "n_chars_texto": len(texto),
        "n_palabras_clean": len(texto.split()),
        "url": registro["url_texto"],
        "estado": estado,
    }


def cargar_parquet_previo(cfg: dict, logger: logging.Logger) -> dict:
    """Carga el parquet previo para reanudar la extraccion sin rehacerla.

    Args:
        cfg: Configuracion del run.
        logger: Logger de ejecucion.

    Returns:
        Diccionario id -> fila previa (vacio si no hay parquet legible).
    """
    ruta = cfg["ruta_parquet"]
    if not ruta.exists():
        return {}
    try:
        tabla = pq.read_table(ruta)
    except Exception as exc:  # noqa: BLE001 - parquet corrupto: se reconstruye
        logger.warning("Parquet previo ilegible (%s); se reconstruye", exc)
        return {}
    previas = {fila["id"]: fila for fila in tabla.to_pylist()}
    logger.info("Parquet previo reutilizado: %d filas", len(previas))
    return previas


def escribir_parquet(cfg: dict, filas: list, logger: logging.Logger) -> None:
    """Escribe el parquet de forma atomica (temporal + replace).

    Args:
        cfg: Configuracion del run.
        filas: Filas con el esquema ESQUEMA_PARQUET.
        logger: Logger de ejecucion.
    """
    ruta = cfg["ruta_parquet"]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla = pa.Table.from_pylist(filas, schema=ESQUEMA_PARQUET)
    temporal = Path(str(ruta) + ".tmp")
    pq.write_table(tabla, temporal, compression="zstd")
    os.replace(temporal, ruta)
    logger.info(
        "Parquet escrito: %s (%d filas, %.2f MB)",
        ruta,
        tabla.num_rows,
        ruta.stat().st_size / 1e6,
    )


def procesar_texto(cfg: dict, logger: logging.Logger) -> None:
    """Ejecuta el modo --texto: extrae, limpia y guarda el parquet.

    Args:
        cfg: Configuracion del run.
        logger: Logger de ejecucion.
    """
    registros = cargar_metadatos(cfg)
    if cfg["limite"] is not None:
        registros = registros[: cfg["limite"]]
    estado_descarga = cargar_estado(cfg)
    previas = {} if cfg["forzar"] else cargar_parquet_previo(cfg, logger)
    limpiar, origen_limpiador = cargar_limpiador(cfg, logger)
    t0 = time.time()
    filas = []
    nuevos_ok = 0
    reusados = 0
    n_texto_corto = 0
    n_stub = 0
    n_error = 0
    for indice, registro in enumerate(registros, start=1):
        previa = previas.get(registro["id"])
        if previa is not None and previa.get("estado") in ("ok", "texto_corto"):
            filas.append(previa)
            reusados += 1
            continue
        ruta = ruta_html(cfg, registro)
        if ruta.exists() and ruta.stat().st_size >= cfg["stub_max_bytes"]:
            try:
                texto = limpiar(extraer_texto_html(ruta, cfg))
                if len(texto) >= cfg["umbral_texto_min_chars"]:
                    filas.append(construir_fila(registro, texto, "ok"))
                    nuevos_ok += 1
                else:
                    filas.append(construir_fila(registro, texto, "texto_corto"))
                    n_texto_corto += 1
            except Exception as exc:  # noqa: BLE001 - un doc no tumba el run
                logger.warning("Fallo la extraccion de %s: %s", registro["id"], exc)
                filas.append(construir_fila(registro, "", "error"))
                n_error += 1
        else:
            estado_previo = estado_descarga.get(registro["id"], {}).get(
                "estado", "error"
            )
            estado = estado_previo if estado_previo in ("stub", "error") else "error"
            filas.append(construir_fila(registro, "", estado))
            if estado == "stub":
                n_stub += 1
            else:
                n_error += 1
        if indice % cfg["log_cada_texto"] == 0 or indice == len(registros):
            logger.info(
                "Texto %d/%d (%.1f%%): nuevos_ok=%d texto_corto=%d reusados=%d "
                "stub=%d error=%d",
                indice,
                len(registros),
                100.0 * indice / len(registros),
                nuevos_ok,
                n_texto_corto,
                reusados,
                n_stub,
                n_error,
            )
    filas.sort(key=lambda fila: (fila["fecha"], fila["id"]))
    if not cfg["forzar"] and cfg["limite"] is not None:
        # Con --limite (prueba parcial) se preservan las filas previas
        # fuera del subconjunto procesado: nunca se pierde trabajo hecho.
        ids_procesados = {registro["id"] for registro in registros}
        for id_previa, fila_previa in previas.items():
            if id_previa not in ids_procesados:
                filas.append(fila_previa)
        filas.sort(key=lambda fila: (fila["fecha"], fila["id"]))
    escribir_parquet(cfg, filas, logger)
    n_ok = sum(1 for fila in filas if fila["estado"] == "ok")
    n_chars_total = sum(fila["n_chars_texto"] for fila in filas)
    n_palabras_total = sum(fila["n_palabras_clean"] for fila in filas)
    n_mas_1000 = sum(
        1 for fila in filas if fila["n_chars_texto"] > cfg["umbral_texto_min_chars"]
    )
    stats = {
        "run": {
            "procesados": len(registros),
            "nuevos_ok": nuevos_ok,
            "texto_corto": n_texto_corto,
            "reusados": reusados,
            "stub": n_stub,
            "error": n_error,
            "limpiador": origen_limpiador,
            "tiempo_seg": round(time.time() - t0, 2),
        },
        "acumulado": {
            "n_docs": len(filas),
            "n_ok": n_ok,
            "n_texto_corto": sum(
                1 for fila in filas if fila["estado"] == "texto_corto"
            ),
            "n_stub": sum(1 for fila in filas if fila["estado"] == "stub"),
            "n_error": sum(1 for fila in filas if fila["estado"] == "error"),
            "n_chars_total": n_chars_total,
            "n_palabras_total": n_palabras_total,
            "n_docs_mas_1000_chars": n_mas_1000,
            "media_chars_ok": round(n_chars_total / n_ok, 1) if n_ok else 0.0,
        },
        "ruta_parquet": str(cfg["ruta_parquet"]),
        "timestamp_utc": ahora_utc(),
    }
    actualizar_stats(cfg, "texto", stats, logger)
    logger.info(
        "Texto terminado: ok=%d texto_corto=%d stub=%d error=%d | chars=%d "
        "palabras=%d docs>1000=%d | %s",
        n_ok,
        stats["acumulado"]["n_texto_corto"],
        stats["acumulado"]["n_stub"],
        stats["acumulado"]["n_error"],
        n_chars_total,
        n_palabras_total,
        n_mas_1000,
        formatear_duracion(time.time() - t0),
    )


# ===== SECCION 5: ORQUESTACION =====


def main() -> None:
    """Punto de entrada: parsea argumentos y ejecuta el modo pedido."""
    args = parsear_argumentos()
    cfg = construir_config(args)
    logger = configurar_logging(cfg["ruta_log"])
    random.seed(cfg["semilla"])
    np.random.seed(cfg["semilla"])
    logger.info(
        "=== C3 descarga 2022-2026 | smoke=%s forzar=%s limite=%s delay=%.1fs "
        "reintentos=%d ===",
        cfg["smoke"],
        cfg["forzar"],
        cfg["limite"],
        cfg["delay"],
        cfg["reintentos"],
    )
    logger.info("Raiz de salidas: %s", cfg["raiz"])
    modos = []
    if args.todo:
        modos.append("todo")
    else:
        if args.listar:
            modos.append("listar")
        if args.descargar:
            modos.append("descargar")
        if args.texto:
            modos.append("texto")
    t0 = time.time()
    sesion = requests.Session()
    sesion.headers.update({"User-Agent": cfg["user_agent"]})
    try:
        if args.todo or args.listar:
            listar(sesion, cfg, logger)
        if args.todo or args.descargar:
            descargar(sesion, cfg, logger)
        if args.todo or args.texto:
            procesar_texto(cfg, logger)
        logger.info(
            "RESUMEN C3: modo=%s smoke=%s tiempo_total=%s",
            "+".join(modos),
            cfg["smoke"],
            formatear_duracion(time.time() - t0),
        )
    except Exception:  # noqa: BLE001 - main captura todo para log + exit 1
        logger.error("Fallo C3 descarga:\n%s", traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
