"""Genera las figuras nuevas del informe final AGE (competencia tweet-age-2026-puj).

Salidas (informe/figuras/):
- fig_evolucion.png      : evolucion CV local y LB por version (datos de CHAMPION.md)
- fig_arquitectura_v48.png: esquema del campeon V48 (PoE geometrico + plug-in)
- fig_v57_analisis.png   : sensibilidad del simplex y deciles de confiabilidad (V57)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RAIZ = Path(__file__).resolve().parents[1]
FIG = RAIZ / "informe" / "figuras"
AZUL = "#2b6cb0"
ROJO = "#c53030"
VERDE = "#2f855a"
GRIS = "#718096"

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
                     "figure.dpi": 200, "savefig.bbox": "tight"})


def figura_evolucion() -> None:
    """Evolucion de la CV local y del leaderboard por version (hitos del proyecto)."""
    versiones = ["V12", "V19", "V20", "V30", "V30+plug", "V38", "V39", "V40", "V41",
                 "V42", "V43", "V48", "V53", "V54", "V56"]
    cv = [0.6005, 0.6079388, 0.6097782, 0.6134471, 0.6159732, 0.6166579, 0.6168017,
          0.6175912, 0.6181119, 0.6184162, 0.6184783, 0.6186451, 0.6188259, 0.6188887,
          0.6189929]
    lb = [0.62182, 0.62578, 0.62689, 0.63302, 0.63575, 0.63590, 0.63588, 0.63685,
          0.63722, 0.63716, 0.63722, 0.63724, 0.63706, 0.63712, 0.63717]
    equis = np.arange(len(versiones))
    figura, eje_cv = plt.subplots(figsize=(7.0, 3.1))
    eje_lb = eje_cv.twinx()
    eje_cv.plot(equis, cv, marker="o", markersize=3.2, color=AZUL, linewidth=1.4,
                label="CV local (macro-F1, LOFO agrupado)")
    eje_lb.plot(equis, lb, marker="s", markersize=3.2, color=ROJO, linewidth=1.4,
                label="Leaderboard Kaggle")
    eje_lb.axhline(0.642, color=GRIS, linestyle="--", linewidth=1.0)
    eje_lb.annotate("techo declarado (D-087): 0.642", xy=(0.4, 0.6422), fontsize=7.5,
                    color=GRIS)
    eje_cv.annotate("V48\nLB 0.63724", xy=(11, cv[11]), xytext=(8.4, 0.6035),
                    fontsize=7.5, color=ROJO, arrowprops={"arrowstyle": "-", "color": ROJO,
                                                          "linewidth": 0.8})
    eje_lb.annotate("V56\nmejor CV 0.6189929", xy=(14, lb[14]), xytext=(12.0, 0.6295),
                    fontsize=7.5, color=VERDE, arrowprops={"arrowstyle": "-", "color": VERDE,
                                                           "linewidth": 0.8})
    eje_cv.set_ylabel("CV local (macro-F1)", color=AZUL)
    eje_lb.set_ylabel("Leaderboard Kaggle", color=ROJO)
    eje_cv.tick_params(axis="y", colors=AZUL)
    eje_lb.tick_params(axis="y", colors=ROJO)
    eje_cv.set_xticks(equis)
    eje_cv.set_xticklabels(versiones, fontsize=7.5, rotation=35, ha="right")
    eje_cv.set_ylim(0.598, 0.6215)
    eje_lb.set_ylim(0.618, 0.645)
    eje_cv.grid(alpha=0.25, linewidth=0.6)
    lineas = eje_cv.get_lines()[:1] + eje_lb.get_lines()[:1]
    eje_cv.legend(lineas, [linea.get_label() for linea in lineas], loc="lower right",
                  fontsize=7.5)
    figura.savefig(FIG / "fig_evolucion.png")
    plt.close(figura)


def figura_arquitectura() -> None:
    """Esquema del campeon V48: seis expertos, PoE geometrico en log-espacio y plug-in.

    Diseno con margenes amplios: buses de entrada/salida y cajas separadas con espacio claro.
    """
    figura, eje = plt.subplots(figsize=(7.2, 4.7))
    eje.set_xlim(0, 11.5)
    eje.set_ylim(0, 7.5)
    eje.axis("off")

    def caja(x, y, ancho, alto, texto, color, tam=7.5):
        eje.add_patch(FancyBboxPatch((x, y), ancho, alto, boxstyle="round,pad=0.10",
                                     linewidth=1.0, edgecolor=color, facecolor="white"))
        eje.text(x + ancho / 2, y + alto / 2, texto, ha="center", va="center",
                 fontsize=tam, color="#1a202c")

    def flecha(x1, y1, x2, y2):
        eje.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                      mutation_scale=9, linewidth=1.0, color=GRIS,
                                      shrinkA=1, shrinkB=1))

    # Entrada
    caja(0.05, 2.95, 2.10, 1.6, "Matriz 279\n169,336 x 279\nfeatures + 14 miembros\n+ ADV-07",
         AZUL, tam=6.8)
    # Bus de entrada
    eje.plot([2.35, 2.35], [1.25, 6.65], color=GRIS, linewidth=1.0)
    flecha(2.15, 3.75, 2.35, 3.75)
    # Expertos (columna unica, separacion vertical 0.90 con cajas de 0.66)
    expertos = [
        ("V30 - LightGBM (Optuna)\npeso 0.563", 6.35, True),
        ("V42 - TabNet suave + SWA\npeso 0.172", 5.45, False),
        ("V43 - TabNet Cram\u00e9r + SWA\npeso 0.083", 4.55, False),
        ("V40 - SE-TabNet\npeso 0.075", 3.65, False),
        ("V39-HGB - HistGradient\nBoosting\npeso 0.055", 2.75, False),
        ("V39-Ord - TabNet ordinal\npeso 0.051", 1.85, False),
    ]
    ancho_caja, alto_caja = 3.0, 0.66
    for texto, centro_y, destacado in expertos:
        caja(2.65, centro_y - alto_caja / 2, ancho_caja, alto_caja, texto,
             ROJO if destacado else VERDE, tam=6.8)
        flecha(2.35, centro_y, 2.65, centro_y)
    # Bus de salida
    eje.plot([5.95, 5.95], [1.45, 6.55], color=GRIS, linewidth=1.0)
    for _, centro_y, _ in expertos:
        flecha(5.65, centro_y, 5.95, centro_y)
    flecha(5.95, 5.50, 6.30, 5.50)
    # Cadena de consenso
    caja(6.30, 4.75, 2.75, 1.5, "PoE en log-espacio\nlog P $\\propto$ $\\Sigma$ wm log Pm",
         VERDE, tam=7.5)
    caja(6.30, 2.85, 2.75, 1.35, "Regla plug-in por clase\n(umbrales de F1 m\u00e1ximo)",
         "#b7791f", tam=7.2)
    caja(6.30, 0.65, 2.75, 1.35, "Submission\n112,892 filas\nLB 0.63724", ROJO, tam=7.2)
    flecha(7.68, 4.75, 7.68, 4.20)
    flecha(7.68, 2.85, 7.68, 2.00)
    eje.text(5.75, 0.10, "El miembro V30 de la evaluaci\u00f3n usa su predicci\u00f3n fuera de fold "
                         "de la segunda partici\u00f3n (esquema publicado del proyecto)",
             ha="center", fontsize=6.8, color=GRIS)
    figura.savefig(FIG / "fig_arquitectura_v48.png")
    plt.close(figura)


def figura_v57() -> None:
    """V57: sensibilidad del simplex 9D (+-20%) y calibracion por deciles."""
    datos = json.loads((RAIZ / "experiments" / "v57_analisis_matematico_estadistico.json")
                       .read_text(encoding="utf-8"))
    sensibilidad = datos["sensibilidad_simplex"]
    deciles = datos["deciles_confiabilidad"]
    figura, ejes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    nombres = [fila["modelo"].replace("_", " ") for fila in sensibilidad]
    menos = [fila["delta_minus_20pct"] for fila in sensibilidad]
    mas = [fila["delta_plus_20pct"] for fila in sensibilidad]
    equis = np.arange(len(nombres))
    ejes[0].bar(equis - 0.19, menos, width=0.36, color=AZUL, label="peso -20%")
    ejes[0].bar(equis + 0.19, mas, width=0.36, color=ROJO, label="peso +20%")
    ejes[0].axhline(0, color="#1a202c", linewidth=0.6)
    ejes[0].set_xticks(equis)
    ejes[0].set_xticklabels(nombres, rotation=45, ha="right", fontsize=6.2)
    ejes[0].set_ylabel("Delta macro-F1 (OOF)")
    ejes[0].set_title("Sensibilidad del simplex (+-20% por peso)")
    ejes[0].legend(fontsize=6.5)
    confianza = [fila["confianza_media"] for fila in deciles]
    exactitud = [fila["exactitud"] for fila in deciles]
    ejes[1].plot(confianza, exactitud, marker="o", markersize=3, color=AZUL,
                 linewidth=1.2, label="deciles (OOF V56)")
    ejes[1].plot([0.15, 0.95], [0.15, 0.95], linestyle="--", linewidth=0.9, color=GRIS,
                 label="calibracion perfecta")
    ejes[1].set_xlabel("confianza media del decil")
    ejes[1].set_ylabel("exactitud")
    ejes[1].set_title("Confiabilidad por deciles")
    ejes[1].legend(fontsize=6.5)
    ejes[1].grid(alpha=0.25, linewidth=0.6)
    figura.tight_layout()
    figura.savefig(FIG / "fig_v57_analisis.png")
    plt.close(figura)


if __name__ == "__main__":
    figura_evolucion()
    figura_arquitectura()
    figura_v57()
    print("figuras nuevas generadas en", FIG)
