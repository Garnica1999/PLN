"""Utilidades para construir y ejecutar notebooks de forma reproducible.

Los notebooks del proyecto se generan a partir de listas de celdas declaradas en
``scripts/`` y se ejecutan con ``nbclient`` usando el kernel del venv, de modo que el
artefacto entregable queda con salidas embebidas y es re-ejecutable de principio a fin.
"""

from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from src import config as cfg

KERNEL_NAME = "python3"


def md(text: str) -> dict:
    """Crea una celda markdown."""
    return nbformat.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> dict:
    """Crea una celda de código."""
    return nbformat.v4.new_code_cell(text.strip("\n"))


def build(path: Path, cells: list[dict], title: str | None = None) -> Path:
    """Construye (o reconstruye) un notebook a partir de una lista de celdas."""
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3 (venv)", "language": "python",
                       "name": KERNEL_NAME},
        "language_info": {"name": "python"},
        "title": title or path.stem,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)
    return path


def execute(path: Path, timeout: int = 3600, log_dir: Path | None = None) -> dict:
    """Ejecuta un notebook en sitio y guarda un log de texto.

    Returns
    -------
    dict
        ``{"notebook": str, "seconds": float, "cells": int, "errors": int, "log": str}``.
    """
    log_dir = log_dir or cfg.EXPERIMENTS
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{path.stem}.log"

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=timeout, kernel_name=KERNEL_NAME,
                            resources={"metadata": {"path": str(cfg.ROOT)}},
                            allow_errors=False)
    started = time.time()
    try:
        client.execute()
        errors = 0
    except Exception as exc:  # noqa: BLE001 - se registra y se propaga el detalle
        errors = 1
        nbformat.write(notebook, path)
        log_path.write_text(f"ERROR: {exc}\n", encoding="utf-8")
        raise
    seconds = time.time() - started
    nbformat.write(notebook, path)

    lines = [f"notebook={path.name} seconds={seconds:.1f} cells={len(notebook.cells)}"]
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                lines.append(output.get("text", ""))
            elif output.get("output_type") == "error":
                lines.append(f"ERROR {output.get('ename')}: {output.get('evalue')}")
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return {"notebook": path.name, "seconds": round(seconds, 1), "cells": len(notebook.cells),
            "errors": errors, "log": str(log_path)}
