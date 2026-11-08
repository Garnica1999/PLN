"""Regenera las figuras de experimentos con el tema del informe."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.rag.plotting import ACCENT, INK, PALETTE, apply_theme, save_figure

TABLES = Path("reports/tablas")
FIGURES = Path("reports/figuras")
METRICS = [("p_at_1", "P@1"), ("p_at_5", "Hit@5"), ("recall", "Recall@10"), ("ndcg", "nDCG@10")]


def grouped(path: Path, title: str, out: str) -> None:
    frame = pd.read_csv(path)
    positions = np.arange(len(METRICS))
    width = 0.8 / len(frame)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for index, (_, row) in enumerate(frame.iterrows()):
        offset = (index - (len(frame) - 1) / 2) * width
        values = [row[metric] for metric, _ in METRICS]
        ax.bar(positions + offset, values, width, label=row["label"], color=PALETTE[index])
    ax.set_xticks(positions, [label for _, label in METRICS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Valor")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(frame))
    save_figure(fig, FIGURES / out)


def limpieza() -> None:
    frame = pd.read_csv(TABLES / "limpieza_metricas.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(frame["variant"].str.upper(), frame["ndcg"], color=ACCENT, width=0.55)
    ax.errorbar(
        frame["variant"].str.upper(),
        frame["ndcg"],
        yerr=[frame["ndcg"] - frame["ndcg_lo"], frame["ndcg_hi"] - frame["ndcg"]],
        fmt="none",
        ecolor=INK,
        capsize=3,
        lw=0.8,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("nDCG@10")
    ax.set_title("Limpieza del texto y recuperación")
    save_figure(fig, FIGURES / "limpieza_comparacion.png")


def escala() -> None:
    labels = ["Recall@10", "P@1", "Hit@5", "nDCG@10"]
    series = {
        "Piloto": [0.947, 0.842, 0.947, 0.909],
        "Corpus": [0.684, 0.474, 0.658, 0.594],
        "Corpus + rerank": [0.684, 0.632, 0.684, 0.661],
    }
    positions = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for index, (name, values) in enumerate(series.items()):
        offset = (index - 1) * width
        ax.bar(positions + offset, values, width, label=name, color=PALETTE[index])
    ax.set_xticks(positions, labels, fontsize=7)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=7,
              handlelength=1.2, columnspacing=0.8)
    save_figure(fig, FIGURES / "escala_e6_e7.png")


def colecciones() -> None:
    """Documentos por coleccion con etiquetas legibles, desde el perfilado."""
    import json

    perfil = json.loads((TABLES / "perfilado.json").read_text(encoding="utf-8"))
    etiquetas = {"url_crawls": "Rastreo web", "bluesky_posts": "Bluesky"}
    conteo = perfil["colecciones"]
    nombres = [etiquetas.get(k, k) for k in conteo]
    valores = list(conteo.values())
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.bar(nombres, valores, color=[PALETTE[0], PALETTE[1]], width=0.55)
    for i, v in enumerate(valores):
        ax.text(i, v, f"{v:,}".replace(",", "."), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Documentos")
    ax.set_title("Documentos por coleccion")
    ax.set_ylim(0, max(valores) * 1.12)
    save_figure(fig, FIGURES / "perfil_colecciones.png")


def main() -> None:
    apply_theme()
    limpieza()
    grouped(TABLES / "e1_chunking.csv", "Segmentación: fijo contra oraciones", "e1_chunking.png")
    grouped(TABLES / "e2_k3_vs_k10.csv", "Tamaño de contexto: k=3 contra k=10", "e2_k3_vs_k10.png")
    grouped(TABLES / "e2_k5_vs_k10.csv", "Tamaño de contexto: k=5 contra k=10", "e2_k5_vs_k10.png")
    escala()
    colecciones()


if __name__ == "__main__":
    main()
