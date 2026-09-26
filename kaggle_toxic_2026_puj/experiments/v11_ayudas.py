"""Utilidades compartidas de la fase v11.

Se separan en su propio modulo porque los informes los repiten todos y porque asi ninguno de
los scripts de la fase tiene que reimplementar el formato de tabla ni el pie de fecha.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone


def fila(columnas: list[str]) -> str:
    """Construye una fila de tabla a partir de sus celdas.

    Se usa en vez de concatenar cadenas a mano porque una fila de tabla tiene varias celdas y
    escribirlas en una sola linea pasaria del limite de caracteres, mientras que partirlas en
    varias activa la concatenacion implicita, que la revision marca como error de estilo. El
    helper resuelve las dos cosas a la vez.

    Args:
        columnas: celdas de la fila.

    Returns:
        La fila formateada con separadores de Markdown.
    """
    return "| " + " | ".join(columnas) + " |"


def sello(inicio: float) -> list[str]:
    """Construye las dos lineas de pie de un informe.

    Args:
        inicio: instante de arranque del script.

    Returns:
        Lista con la linea en blanco de separacion y la de fecha y duracion.
    """
    marca = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return ["", f"Fecha: {marca}. Duracion: {round(time.time() - inicio, 1)} s."]
