"""Genera la figura de resumen final del sistema."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.7, 1, 1]})
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.28, wspace=0.4)

    ax = axes[0]
    ancho = 0.2
    for j, (nombre, archivo) in enumerate(configs):
        valores = [media(archivo, metrica) for metrica, _ in metricas]
        posiciones = [i + (j - (len(configs) - 1) / 2) * ancho for i in range(len(metricas))]
        ax.bar(posiciones, valores, ancho, label=nombre, color=PALETTE[j])
    ax.set_xticks(range(len(metricas)), [etiqueta for _, etiqueta in metricas], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title("RAGAS", fontsize=9)

    ax = axes[1]
    amb = pd.read_csv(TABLES / "ambiguedad_v2_humano_resumen.csv").iloc[0]
    valores = [
        amb["baseline_accuracy"],
        amb["clarification_accuracy"],
        amb["conditional_accuracy"],
        amb["conditional_intended_covered"],
    ]
    etiquetas = ["Base", "Clar.", "Cond.", "Opc."]
    ax.bar(range(4), valores, color=[PALETTE[3], PALETTE[2], PALETTE[0], PALETTE[1]])
    ax.set_xticks(range(4), etiquetas, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title("Ambigüedad (n=40)", fontsize=9)
    for i, valor in enumerate(valores):
        ax.text(i, valor + 0.02, f"{valor:.2f}", ha="center", fontsize=7, color=INK)

    ax = axes[2]
    libre = pd.read_csv(TABLES / "reservado_hibrida_resumen.csv").iloc[0]
    estricto = pd.read_csv(TABLES / "reservado_hibrido_estricto_resumen.csv", sep=";").iloc[0]
    entregadas = [int(libre["delivered_direct"]), int(estricto["entregadas"])]
    abstenciones = [int(libre["abstained_direct"]), int(estricto["abstenciones_directas"])]
    ax.bar([0, 1], entregadas, color=PALETTE[0], width=0.55)
    ax.bar([0, 1], abstenciones, bottom=entregadas, color=PALETTE[1], width=0.55)
    ax.set_xticks([0, 1], ["Sin\ncompuerta", "Con\ncompuerta"], fontsize=7)
    ax.set_ylim(0, 20)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_title("Reservado (20 directas)", fontsize=9)
    for i, (entregada, abstenida) in enumerate(zip(entregadas, abstenciones)):
        ax.text(i, entregada / 2, str(entregada), ha="center", va="center",
                color="white", fontsize=7, fontweight="bold")
        if abstenida >= 3:
            ax.text(i, entregada + abstenida / 2, str(abstenida), ha="center", va="center",
                    color="white", fontsize=6.5, fontweight="bold")
        else:
            ax.text(i + 0.3, entregada + abstenida / 2, str(abstenida), ha="left", va="center",
                    color=PALETTE[1], fontsize=6.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 0.01))

    save_figure(fig, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
