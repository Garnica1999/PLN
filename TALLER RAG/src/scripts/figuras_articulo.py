"""Regenera las figuras del articulo a tamano de impresion."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.rag.plotting import INK, PALETTE, WARM, apply_theme, save_figure

TABLES = Path("reports/tablas")
FIGURES = Path("reports/figuras")


def _json(name: str) -> dict:
    return json.loads((TABLES / name).read_text(encoding="utf-8"))


def colecciones() -> None:
    """Dos barras horizontales para una columna."""
    perfil = _json("perfilado.json")
    etiquetas = {"url_crawls": "Rastreo web", "bluesky_posts": "Bluesky"}
    pares = [(etiquetas.get(k, k), v) for k, v in perfil["colecciones"].items()][::-1]
    nombres = [nombre for nombre, _ in pares]
    valores = [valor for _, valor in pares]
    fig, ax = plt.subplots(figsize=(3.5, 1.7))
    ax.barh(nombres, valores, color=[PALETTE[1], PALETTE[0]], height=0.55)
    for i, valor in enumerate(valores):
        ax.text(valor, i, f" {valor:,}".replace(",", "."), va="center", ha="left", fontsize=8, color=INK)
    ax.set_xlim(0, max(valores) * 1.18)
    ax.set_xlabel("Documentos")
    ax.tick_params(axis="y", length=0)
    save_figure(fig, FIGURES / "perfil_colecciones.png")


def near_dups() -> None:
    """Porcentaje de documentos con near-duplicates por muestra."""
    data = _json("textometria.json")["near_duplicates"]
    nombres = ["Piloto", "Muestra"]
    valores = [data["piloto"]["pct_documentos_con_par"], data["muestra"]["pct_documentos_con_par"]]
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    ax.bar(nombres, valores, color=WARM, width=0.5)
    for i, valor in enumerate(valores):
        ax.text(i, valor + 0.08, f"{valor:.2f} %".replace(".", ","), ha="center", fontsize=8, color=INK)
    ax.set_ylim(0, max(valores) * 1.25)
    ax.set_ylabel("% de documentos")
    save_figure(fig, FIGURES / "textometria_near_dups.png")


def mosaico() -> None:
    """Zipf y longitud en un solo PNG de ancho de pagina."""
    data = _json("textometria.json")
    frame = pd.read_csv(TABLES / "textometria.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))

    ax = axes[0]
    for label, points in (("Piloto", data["zipf"]["piloto"]), ("Muestra", data["zipf"]["muestra"])):
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", markersize=2, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rango")
    ax.set_ylabel("Frecuencia")
    ax.legend(loc="lower left")

    ax = axes[1]
    for label, conjunto in (("Piloto", "piloto"), ("Muestra", "muestra")):
        words = frame.loc[frame["set"] == conjunto, "words"].clip(upper=3000)
        ax.hist(words, bins=50, alpha=0.55, label=label)
    ax.set_xlabel("Palabras por documento (tope 3.000)")
    ax.set_ylabel("Documentos")
    ax.legend(loc="upper right")

    fig.tight_layout()
    save_figure(fig, FIGURES / "textometria_mosaico.png")


def main() -> None:
    apply_theme()
    colecciones()
    near_dups()
    mosaico()


if __name__ == "__main__":
    main()
