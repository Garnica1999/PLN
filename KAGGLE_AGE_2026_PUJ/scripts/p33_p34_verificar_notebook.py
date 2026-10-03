"""P33+P34 - ejecuta la seccion del notebook y comprueba que pasa de principio a fin.

Un notebook escrito y no ejecutado no es evidencia: las cifras pueden haber cambiado en los
artefactos y el texto seguir quoting las viejas. Este script lo comprueba de verdad, y por eso va
aparte del que la anade.

Corre con el directorio de trabajo en `notebooks/`, porque las celdas usan rutas relativas.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import nbformat  # noqa: E402

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "99_entrega_consolidado.ipynb"
MARCADOR = "FASE P33 Y P34"


def main() -> int:
    """Ejecuta las celdas de codigo de la seccion una a una, en orden."""
    nb = nbformat.read(NOTEBOOK, as_version=4)
    inicio = None
    for i, c in enumerate(nb.cells):
        if c.cell_type == "markdown" and MARCADOR in c.source:
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
    for i, c in enumerate(celdas, 1):
        try:
            exec(compile(c.source, f"<celda {i}>", "exec"), espacio)
            print(f"  celda {i}: OK", flush=True)
        except Exception as error:  # noqa: BLE001
            fallos.append((i, type(error).__name__, str(error)))
            print(f"  celda {i}: FALLO {type(error).__name__}: {error}", flush=True)
    print(f"\n{len(celdas) - len(fallos)}/{len(celdas)} celdas en verde", flush=True)
    if fallos:
        print("[STOP] la seccion no se ejecuta limpia", flush=True)
        return 1
    print("[OK] la seccion se ejecuta y las aserciones pasan", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
