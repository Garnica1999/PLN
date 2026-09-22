"""01_descarga_constitucion.py
Fecha: 2026-09-08
Descripcion: Descarga el texto oficial de la Constitucion Politica de Colombia
    1991 desde el portal del Senado (secretariasenado.gov.co, Base de Datos
    Avance Juridico) y guarda el crudo en data/constitucion/ con un log
    trazable en experimentos/descarga_constitucion.log.
Estado del arte aplicado: fuente oficial primaria del Senado; reintentos con
    backoff y hash SHA-256 por pagina para trazabilidad E2 Juris-PLN.
"""

import hashlib
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

CONFIG = {
    "semilla": 42,
    "base_url": "http://www.secretariasenado.gov.co/senado/basedoc/",
    "pagina_principal": "constitucion_politica_1991.html",
    "n_paginas_secundarias": 15,
    "prefijo_secundaria": "constitucion_politica_1991_pr",
    "dir_crudo": "data/constitucion",
    "ruta_log": "experimentos/descarga_constitucion.log",
    "ruta_manifiesto": "data/constitucion/manifiesto.json",
    "timeout_seg": 60,
    "reintentos": 4,
    "pausa_seg": 2.0,
    "user_agent": "Juris-PLN-E2 (investigacion academica; contacto universidad)",
    "fuente_portada": "http://www.secretariasenado.gov.co/constitucion-politica",
    "fuente_espejo": "https://www.constitucioncolombia.com/titulo-1/capitulo-0/articulo-1",
}

ROOT = Path(__file__).resolve().parent.parent


def configurar_semillas(semilla: int) -> None:
    """Fija las semillas de aleatoriedad para reproducibilidad.

    Args:
        semilla: Valor entero de semilla.
    """
    random.seed(semilla)
    np.random.seed(semilla)


def construir_urls(config: dict) -> list:
    """Construye la lista ordenada de URLs oficiales a descargar.

    Args:
        config: Diccionario CONFIG con base_url y nombres de pagina.

    Returns:
        Lista de URLs (portada, pagina principal y pr001..pr015).
    """
    urls = [config["fuente_portada"]]
    urls.append(config["base_url"] + config["pagina_principal"])
    for i in range(1, config["n_paginas_secundarias"] + 1):
        nombre = f"{config['prefijo_secundaria']}{i:03d}.html"
        urls.append(config["base_url"] + nombre)
    return urls


def descargar_con_reintentos(
    sesion: requests.Session, url: str, config: dict
) -> requests.Response:
    """Descarga una URL con reintentos y pausa entre intentos.

    Args:
        sesion: Sesion de requests reutilizable.
        url: URL a descargar.
        config: Diccionario CONFIG (timeout_seg, reintentos, pausa_seg).

    Returns:
        Respuesta HTTP exitosa.

    Raises:
        RuntimeError: Si todos los intentos fallan, con el ultimo error.
    """
    ultimo_error = None
    for intento in range(1, config["reintentos"] + 1):
        try:
            respuesta = sesion.get(url, timeout=config["timeout_seg"])
            respuesta.raise_for_status()
            return respuesta
        except Exception as exc:  # noqa: BLE001
            ultimo_error = exc
            time.sleep(config["pausa_seg"] * intento)
    raise RuntimeError(
        f"Fallo {url} tras {config['reintentos']} intentos: {ultimo_error}"
    )


def nombre_archivo_para_url(url: str, indice: int) -> str:
    """Deriva un nombre de archivo local estable para una URL.

    Args:
        url: URL descargada.
        indice: Posicion en la lista de descarga (orden documental).

    Returns:
        Nombre de archivo como cp_<indice>_<ultimo_segmento>.
    """
    segmento = url.rstrip("/").split("/")[-1] or "portada"
    if "." not in segmento:
        segmento += ".html"
    return f"cp_{indice:02d}_{segmento}"


def descargar_todo(config: dict, raiz: Path) -> tuple:
    """Descarga todas las paginas oficiales y guarda el crudo en disco.

    Args:
        config: Diccionario CONFIG.
        raiz: Directorio raiz del proyecto (SEGUNDA ENTREGA).

    Returns:
        Tupla (registros, dir_crudo) donde registros es la lista de dicts
        con url, archivo, bytes y sha256 por pagina.

    Raises:
        RuntimeError: Si alguna descarga falla sin fallback disponible.
    """
    dir_crudo = raiz / config["dir_crudo"]
    dir_crudo.mkdir(parents=True, exist_ok=True)
    sesion = requests.Session()
    sesion.headers.update({"User-Agent": config["user_agent"]})
    registros = []
    for indice, url in enumerate(construir_urls(config)):
        respuesta = descargar_con_reintentos(sesion, url, config)
        contenido = respuesta.content
        nombre = nombre_archivo_para_url(url, indice)
        destino = dir_crudo / nombre
        destino.write_bytes(contenido)
        registros.append(
            {
                "url": url,
                "archivo": str(destino.relative_to(raiz)),
                "bytes": len(contenido),
                "sha256": hashlib.sha256(contenido).hexdigest(),
            }
        )
        print(f"OK {indice:02d} {url} ({len(contenido)} bytes -> {nombre})")
        time.sleep(config["pausa_seg"])
    return registros, dir_crudo


def escribir_log_y_manifiesto(
    registros: list, config: dict, raiz: Path, nota_fallback: str = ""
) -> None:
    """Escribe el log de descarga y el manifiesto JSON de trazabilidad.

    Args:
        registros: Lista de dicts con url, archivo, bytes y sha256.
        config: Diccionario CONFIG.
        raiz: Directorio raiz del proyecto.
        nota_fallback: Nota sobre uso de fuente espejo (vacio si no aplica).
    """
    import json

    fecha = datetime.now(timezone.utc).isoformat()
    total = sum(r["bytes"] for r in registros)
    lineas = [
        f"fecha_utc: {fecha}",
        f"fuente_primaria: {config['base_url']}",
        f"paginas: {len(registros)}",
        f"bytes_total: {total}",
    ]
    if nota_fallback:
        lineas.append(f"fallback: {nota_fallback}")
    for r in registros:
        lineas.append(
            f"URL={r['url']} | archivo={r['archivo']} | "
            f"bytes={r['bytes']} | sha256={r['sha256']}"
        )
    ruta_log = raiz / config["ruta_log"]
    ruta_log.parent.mkdir(parents=True, exist_ok=True)
    ruta_log.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    ruta_manifiesto = raiz / config["ruta_manifiesto"]
    ruta_manifiesto.write_text(
        json.dumps(
            {"fecha_utc": fecha, "paginas": registros, "fallback": nota_fallback},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Log escrito en {ruta_log} ({len(registros)} paginas, {total} bytes)")


def main() -> None:
    """Orquesta la descarga completa de la Constitucion y su trazabilidad."""
    configurar_semillas(CONFIG["semilla"])
    raiz = ROOT
    print(f"Raiz del proyecto: {raiz}")
    try:
        registros, _ = descargar_todo(CONFIG, raiz)
        escribir_log_y_manifiesto(registros, CONFIG, raiz)
    except RuntimeError as exc:
        print(f"ERROR en fuente primaria: {exc}", file=sys.stderr)
        print(
            f"Fallback espejo verificable: {CONFIG['fuente_espejo']}", file=sys.stderr
        )
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
