"""V31+ - 09: ejecuta la seccion V31+ del notebook y comprueba que las aserciones quedan en verde.

Un notebook que se escribe pero no se ejecuta no es evidencia de nada: las cifras pueden haber
cambiado en los artefactos y el texto del notebook seguir quoting las viejas. Este script lo
comprueba de verdad, y por eso va aparte de `v31_08`, que solo lo anade.

Corre con el directorio de trabajo en `notebooks/`, porque las celdas usan rutas relativas.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import nbformat

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "99_entrega_consolidado.ipynb"
MARCADOR = "V31+ ADVANCED RESEARCH"


def main() -> int:
    """Ejecuta las celdas de codigo de la seccion V31+ una a una, en orden."""
    nb = nbformat.read(NOTEBOOK, as_version=4)
    inicio = None
    for i, celda in enumerate(nb.cells):
        if celda.cell_type == "markdown" and MARCADOR in celda.source:
            inicio = i
            break
    if inicio is None:
        print(f"[STOP] la seccion {MARCADOR!r} no esta en el notebook", flush=True)
        return 1
    celdas = [c for c in nb.cells[inicio:] if c.cell_type == "code"]
    print(f"seccion en la celda {inicio} | {len(celdas)} celdas de codigo | "
          f"{len(nb.cells)} celdas totales", flush=True)

    espacio: dict = {"__name__": "__main__"}
    fallos = []
    for i, celda in enumerate(celdas, 1):
        try:
            exec(compile(celda.source, f"<celda {i}>", "exec"), espacio)
            print(f"  celda {i}: OK", flush=True)
        except Exception as error:  # noqa: BLE001
            fallos.append((i, type(error).__name__, str(error)))
            print(f"  celda {i}: FALLO {type(error).__name__}: {error}", flush=True)
    print(f"\n{len(celdas) - len(fallos)}/{len(celdas)} celdas en verde", flush=True)
    if fallos:
        print("[STOP] la seccion del notebook no se ejecuta limpia", flush=True)
        return 1
    print("[OK] la seccion se ejecuta de principio a fin y las aserciones pasan", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
