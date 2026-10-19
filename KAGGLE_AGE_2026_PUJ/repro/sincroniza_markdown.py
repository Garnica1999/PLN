"""Reconcilia la prosa del notebook generado sobre un notebook ejecutado.

El notebook de entrega debe llevar la prosa final (secciones documentales actualizadas) sin perder
las salidas ya ejecutadas. Este script toma el GENERADO como fuente de la prosa y el EJECUTADO como
fuente de las celdas de codigo y sus salidas, alineando ambos por las celdas de codigo (anclas con
fuente identica) para ser robusto a inserciones y borrados de celdas markdown.

Uso:
    python repro/sincroniza_markdown.py <generado.ipynb> <ejecutado.ipynb> <salida.ipynb>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    """Reconstruye el notebook de entrega: codigo del ejecutado, prosa del generado."""
    if len(sys.argv) != 4:
        print("uso: sincroniza_markdown.py <generado.ipynb> <ejecutado.ipynb> <salida.ipynb>")
        return 2
    ruta_generado, ruta_ejecutado, ruta_salida = (Path(arg) for arg in sys.argv[1:])
    generado = json.loads(ruta_generado.read_text(encoding="utf-8"))
    ejecutado = json.loads(ruta_ejecutado.read_text(encoding="utf-8"))

    celdas_generado = generado["cells"]
    celdas_ejecutado = ejecutado["cells"]
    salida: list[dict] = []
    puntero = 0
    codigo_emparejadas = 0
    prosa_generada = 0
    for celda in celdas_generado:
        if celda["cell_type"] != "code":
            salida.append(celda)
            prosa_generada += 1
            continue
        while puntero < len(celdas_ejecutado) and not (
            celdas_ejecutado[puntero]["cell_type"] == "code"
            and celdas_ejecutado[puntero]["source"] == celda["source"]
        ):
            puntero += 1
        if puntero >= len(celdas_ejecutado):
            print("no se encontro la celda de codigo ejecutada correspondiente:")
            print("".join(celda["source"])[:120])
            return 1
        salida.append(celdas_ejecutado[puntero])
        codigo_emparejadas += 1
        puntero += 1

    entregable = dict(ejecutado)
    entregable["cells"] = salida
    ruta_salida.write_text(json.dumps(entregable, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"celdas: {len(salida)} | codigo emparejadas: {codigo_emparejadas} | "
          f"prosa del generado: {prosa_generada} | salida: {ruta_salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
