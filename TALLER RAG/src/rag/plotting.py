"""Tema visual único para las figuras del informe y las diapositivas."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = "#ffffff"
INK = "#1d2420"
MUTED = "#6b726b"
LINE = "#dcd6c9"
ACCENT = "#1f6b57"
WARM = "#a64b2a"
BLUE = "#3f5d73"
PALETTE = [ACCENT, WARM, BLUE, "#8a8f88", "#7b5e83"]


def apply_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.labelsize": 9,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": LINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
