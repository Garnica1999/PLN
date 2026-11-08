"""Genera la figura de resumen final del sistema."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.rag.plotting import INK, PALETTE, apply_theme, save_figure

TABLES = Path("reports/tablas")
OUT = Path("reports/figuras/resumen_final.png")


def media(archivo: str, metrica: str) -> float:
    frame = pd.read_csv(TABLES / archivo)
    return float(frame[metrica].mean())


def main() -> None:
    apply_theme()
    configs = [
        ("Híbrido k=10", "ragas_hibrido.csv"),
        ("Denso k=10", "ragas_denso_k10.csv"),
        ("Denso k=5", "ragas_gemma4-e4b_bge-m3.csv"),
        ("Anterior", "ragas_gemma2-2b.csv"),
    ]
    metricas = [
        ("faithfulness", "Fid."),
        ("answer_relevancy", "Rel."),
        ("context_precision", "Prec."),
        ("context_recall", "Cob."),
        ("answer_correctness", "Corr."),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.6, 1, 1]})

    ax = axes[0]
    ancho = 0.2
    for j, (nombre, archivo) in enumerate(configs):
        valores = [media(archivo, metrica) for metrica, _ in metricas]
        posiciones = [i + (j - (len(configs) - 1) / 2) * ancho for i in range(len(metricas))]
        ax.bar(posiciones, valores, ancho, label=nombre, color=PALETTE[j])
    ax.set_xticks(range(len(metricas)), [etiqueta for _, etiqueta in metricas])
    ax.set_ylim(0, 1)
    ax.set_title("RAGAS")
    ax.legend(ncol=2, fontsize=6, loc="lower left")

    ax = axes[1]
    amb = pd.read_csv(TABLES / "ambiguedad_v2_humano_resumen.csv").iloc[0]
    valores = [
        amb["baseline_accuracy"],
        amb["clarification_accuracy"],
        amb["conditional_accuracy"],
        amb["conditional_intended_covered"],
    ]
    etiquetas = ["Base", "Clar.", "Cond.", "Opciones"]
    ax.bar(range(4), valores, color=[PALETTE[3], PALETTE[2], PALETTE[0], PALETTE[1]])
    ax.set_xticks(range(4), etiquetas)
    ax.set_ylim(0, 1)
    ax.set_title("Ambigüedad (n=40)")
    for i, valor in enumerate(valores):
        ax.text(i, valor + 0.02, f"{valor:.2f}", ha="center", fontsize=7, color=INK)

    ax = axes[2]
    libre = pd.read_csv(TABLES / "reservado_hibrida_resumen.csv").iloc[0]
    estricto = pd.read_csv(TABLES / "reservado_hibrido_estricto_resumen.csv", sep=";").iloc[0]
    entregadas = [int(libre["delivered_direct"]), int(estricto["entregadas"])]
    abstenciones = [int(libre["abstained_direct"]), int(estricto["abstenciones_directas"])]
    ax.bar(["Sin compuerta", "Con compuerta"], entregadas, color=PALETTE[0], label="Entregadas")
    ax.bar(["Sin compuerta", "Con compuerta"], abstenciones, bottom=entregadas, color=PALETTE[1], label="Abstenciones")
    ax.set_ylim(0, 20)
    ax.set_title("Reservado (20 directas)")
    ax.legend(fontsize=6, loc="lower right")
    for i, (entregada, abstenida) in enumerate(zip(entregadas, abstenciones)):
        ax.text(i, entregada + abstenida + 0.4, f"{entregada}/20", ha="center", fontsize=7, color=INK)

    fig.tight_layout()
    save_figure(fig, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
