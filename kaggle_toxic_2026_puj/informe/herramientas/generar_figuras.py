#!/usr/bin/env python
"""Genera las figuras del informe final TOXIC 2026 PUJ.

Todas las figuras se producen a partir de los artefactos congelados del proyecto
y de `informe/datos/estadisticas.json`. Se guardan en PDF vectorial (para LaTeX)
y en PNG a 300 dpi (para revisión rápida).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "informe" / "datos"
FIGURAS = RAIZ / "informe" / "figuras"
FIGURAS.mkdir(parents=True, exist_ok=True)

with open(DATOS / "estadisticas.json", encoding="utf-8") as archivo:
    CIFRAS = json.load(archivo)

ETIQUETAS = ("HS", "TR", "AG")
COLORES = {"HS": "#1f4e79", "TR": "#c55a11", "AG": "#548235"}
GRIS = "#595959"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def guardar(figura: plt.Figure, nombre: str) -> None:
    """Guarda la figura en PDF y PNG."""
    figura.savefig(FIGURAS / f"{nombre}.pdf")
    figura.savefig(FIGURAS / f"{nombre}.png", dpi=300)
    plt.close(figura)
    print("Figura:", nombre)


def leer_train() -> pd.DataFrame:
    """Lee el CSV de entrenamiento con identificadores limpios."""
    marco = pd.read_csv(
        RAIZ / "data" / "raw" / "toxic_train.csv",
        sep=";",
        dtype=str,
        keep_default_na=False,
        encoding="cp1252",
    )
    marco.columns = [columna.strip() for columna in marco.columns]
    marco["id"] = marco["id"].str.strip().str.strip('"')
    return marco


def figura_eda() -> None:
    """Prevalencias, perfiles multietiqueta y longitudes."""
    train = leer_train()
    figura, ejes = plt.subplots(1, 2, figsize=(7.0, 2.55), gridspec_kw={"width_ratios": [1.05, 1.0]})

    eje = ejes[0]
    perfiles = CIFRAS["perfiles"]
    claves = ["000", "001", "100", "101", "110", "111"]
    valores = [perfiles[clave] for clave in claves]
    colores = ["#7f7f7f", "#c55a11", "#1f4e79", "#2e75b6", "#548235", "#a61c00"]
    eje.bar(claves, valores, color=colores, width=0.72)
    for posicion, valor in enumerate(valores):
        eje.text(posicion, valor + 18, str(valor), ha="center", va="bottom", fontsize=7)
    eje.set_title("Perfiles HS·TR·AG (N = 1 731)")
    eje.set_xlabel("combinación de etiquetas")
    eje.set_ylabel("tuits")
    eje.set_ylim(0, max(valores) * 1.18)

    eje = ejes[1]
    longitudes = train["text"].str.len().to_numpy()
    eje.hist(longitudes, bins=42, color="#4472c4", alpha=0.85, edgecolor="white", linewidth=0.3)
    mediana = CIFRAS["longitudes"]["caracteres_mediana"]
    eje.axvline(mediana, color="#c00000", linestyle="--", linewidth=1.0)
    eje.text(
        mediana + 5,
        eje.get_ylim()[1] * 0.9,
        f"mediana = {mediana:.0f}",
        color="#c00000",
        fontsize=7,
    )
    eje.set_title("Longitud de los tuits (caracteres)")
    eje.set_xlabel("caracteres")
    eje.set_ylabel("frecuencia")

    figura.tight_layout(w_pad=1.4)
    guardar(figura, "fig_eda")


def figura_phi() -> None:
    """Matriz de asociación Phi entre etiquetas."""
    figura, eje = plt.subplots(figsize=(3.4, 2.75))
    matriz = np.array(
        [
            [1.0, CIFRAS["phi"]["HS-TR"]["phi"], CIFRAS["phi"]["HS-AG"]["phi"]],
            [CIFRAS["phi"]["HS-TR"]["phi"], 1.0, CIFRAS["phi"]["TR-AG"]["phi"]],
            [CIFRAS["phi"]["HS-AG"]["phi"], CIFRAS["phi"]["TR-AG"]["phi"], 1.0],
        ]
    )
    imagen = eje.imshow(matriz, cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(3):
        for j in range(3):
            eje.text(
                j,
                i,
                f"{matriz[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if matriz[i, j] > 0.62 else "black",
            )
    eje.set_xticks(range(3), ETIQUETAS)
    eje.set_yticks(range(3), ETIQUETAS)
    eje.set_title("Coeficiente Phi entre etiquetas")
    eje.grid(False)
    figura.colorbar(imagen, ax=eje, shrink=0.82, pad=0.03)
    guardar(figura, "fig_phi")


def figura_escalera() -> None:
    """Métrica local y leaderboard a lo largo de la serie final."""
    serie = CIFRAS["serie"]
    nombres = [fila["experimento"] for fila in serie]
    canonica = [fila["ccv_canonica"] for fila in serie]
    particion = [fila["ccv_47"] for fila in serie]
    lb = [fila["lb"] for fila in serie]
    x = np.arange(len(serie))

    figura, ejes = plt.subplots(2, 1, figsize=(7.0, 4.25), sharex=True)

    eje = ejes[0]
    eje.plot(x, canonica, marker="o", markersize=2.6, linewidth=1.2, color="#1f4e79", label="CCV canónica")
    eje.plot(x, particion, marker="s", markersize=2.2, linewidth=1.0, color="#c55a11", label="CCV partición 47")
    eje.set_ylabel("F1 local (OOF)")
    eje.set_ylim(0.777, 0.794)
    eje.legend(loc="upper right", ncol=2, frameon=False)
    eje.annotate(
        "P109: máximo local,\nmínimo en LB",
        xy=(1, canonica[1]),
        xytext=(3.4, 0.7798),
        fontsize=6.8,
        color=GRIS,
        arrowprops={"arrowstyle": "->", "lw": 0.6, "color": GRIS},
    )
    eje.set_title("La serie final: métrica local frente al leaderboard público")

    eje = ejes[1]
    eje.plot(x, lb, marker="o", markersize=2.8, linewidth=1.3, color="#2e75b6")
    eje.axhline(0.80, color=GRIS, linestyle=":", linewidth=0.8)
    eje.axhline(0.81, color=GRIS, linestyle=":", linewidth=0.8)
    eje.text(len(serie) - 0.6, 0.8003, "0.80", ha="right", fontsize=6.5, color=GRIS)
    eje.text(len(serie) - 0.6, 0.8103, "0.81", ha="right", fontsize=6.5, color=GRIS)
    for indice in (0, 6, 19, 22, 23, 24):
        desplazamientos = {
            0: (0, 8),
            6: (0, 8),
            19: (-30, -26),
            22: (-12, -28),
            23: (-6, 14),
            24: (6, 16),
        }
        eje.annotate(
            f"{nombres[indice]}\n{lb[indice]:.5f}",
            xy=(indice, lb[indice]),
            xytext=desplazamientos[indice],
            textcoords="offset points",
            ha="center",
            fontsize=6.4,
            color="#1f1f1f",
        )
    eje.set_ylabel("F1 público (LB)")
    eje.set_ylim(0.771, 0.819)
    eje.set_xticks(x, nombres, rotation=90, ha="center")
    eje.set_xlim(-0.8, len(serie) - 0.2)

    figura.tight_layout(h_pad=0.35)
    guardar(figura, "fig_escalera")


def figura_waterfall() -> None:
    """Descomposición de la ganancia del leaderboard por capas."""
    etapas = [
        ("V03", 0.78306, 0.0, "#7f7f7f"),
        ("Base\n(P114)", 0.78306, 0.00751, "#1f4e79"),
        ("Leyes de\ndecisión (P121)", 0.79057, 0.00038, "#2e75b6"),
        ("Pares\n(P128)", 0.79095, 0.01421, "#548235"),
        ("Pares finos\n(P133)", 0.80516, 0.00931, "#a61c00"),
    ]
    figura, eje = plt.subplots(figsize=(4.4, 2.55))
    acumulado = 0.0
    for posicion, (nombre, base, ganancia, color) in enumerate(etapas):
        if posicion == 0:
            eje.bar(posicion, base, bottom=0.0, color=color, width=0.62)
            eje.text(posicion, base + 0.0004, f"{base:.5f}", ha="center", fontsize=7)
            acumulado = base
        else:
            eje.bar(posicion, ganancia, bottom=acumulado, color=color, width=0.62)
            if posicion == len(etapas) - 1:
                eje.text(
                    posicion,
                    acumulado + ganancia / 2,
                    f"+{ganancia:.5f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                )
            else:
                eje.text(
                    posicion,
                    acumulado + ganancia + 0.0005,
                    f"+{ganancia:.5f}",
                    ha="center",
                    fontsize=7,
                )
            acumulado += ganancia
    eje.text(4, acumulado + 0.0010, f"{acumulado:.5f}", ha="center", fontsize=7.5, fontweight="bold")
    eje.set_xticks(range(len(etapas)), [etapa[0] for etapa in etapas])
    eje.set_ylim(0.780, 0.8175)
    eje.set_ylabel("F1 público acumulado")
    eje.set_title("De 0.78306 a 0.81447: qué aportó cada capa")
    eje.text(
        0.02,
        0.96,
        "eje truncado",
        transform=eje.transAxes,
        fontsize=6.5,
        color=GRIS,
        va="top",
    )
    figura.tight_layout()
    guardar(figura, "fig_waterfall")


def figura_umbrales() -> None:
    """Sensibilidad de la F1 al umbral de decisión por etiqueta."""
    rejilla = np.array(CIFRAS["umbrales"]["rejilla"])
    figura, eje = plt.subplots(figsize=(3.45, 2.45))
    for etiqueta in ETIQUETAS:
        curva = np.array(CIFRAS["umbrales"]["curvas"][etiqueta])
        eje.plot(rejilla, curva, linewidth=1.2, color=COLORES[etiqueta], label=etiqueta)
        congelado = CIFRAS["umbrales"]["congelados"][etiqueta]
        valor = np.interp(congelado, rejilla, curva)
        eje.plot([congelado], [valor], marker="o", markersize=3.6, color=COLORES[etiqueta])
        eje.axvline(congelado, color=COLORES[etiqueta], linestyle=":", linewidth=0.7, alpha=0.8)
    eje.set_xlabel("umbral de decisión")
    eje.set_ylabel("F1 de la clase positiva")
    eje.set_xlim(0.20, 0.80)
    eje.set_ylim(0.70, 0.85)
    eje.legend(loc="lower center", ncol=3, frameon=False)
    eje.set_title("F1 frente al umbral (OOF, campeón P133)")
    figura.tight_layout()
    guardar(figura, "fig_umbrales")


def figura_confusion() -> None:
    """Matrices de confusión del campeón por etiqueta."""
    figura, ejes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    for eje, etiqueta in zip(ejes, ETIQUETAS):
        confusion = CIFRAS["modelos"]["P133"]["confusion"][etiqueta]
        matriz = np.array(
            [[confusion["TN"], confusion["FP"]], [confusion["FN"], confusion["TP"]]]
        )
        eje.imshow(matriz, cmap="Blues")
        for i in range(2):
            for j in range(2):
                eje.text(
                    j,
                    i,
                    f"{matriz[i, j]:,}".replace(",", "."),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if matriz[i, j] > matriz.max() * 0.55 else "black",
                )
        eje.set_xticks([0, 1], ["pred 0", "pred 1"])
        eje.set_yticks([0, 1], ["real 0", "real 1"])
        eje.set_title(f"{etiqueta} (F1 = {CIFRAS['modelos']['P133']['f1_por_etiqueta'][etiqueta]:.3f})")
        eje.grid(False)
    figura.tight_layout(w_pad=1.0)
    guardar(figura, "fig_confusion")


def figura_bootstrap() -> None:
    """Intervalos de confianza de la mejora y rendimiento por longitud."""
    figura, ejes = plt.subplots(1, 2, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1.2, 1.0]})

    eje = ejes[0]
    filas = [
        ("vs V03", CIFRAS["bootstrap"]["P133-V03"], "#1f4e79"),
        ("vs P110", CIFRAS["bootstrap"]["P133-P110"], "#548235"),
        ("vs P114", CIFRAS["bootstrap"]["P133-P114"], "#c55a11"),
    ]
    for posicion, (nombre, datos, color) in enumerate(filas):
        inferior = datos["ic95_inferior"]
        superior = datos["ic95_superior"]
        centro = datos["diferencia_observada"]
        eje.plot([inferior, superior], [posicion, posicion], color=color, linewidth=2.2)
        eje.plot([centro], [posicion], marker="o", markersize=4, color=color)
        eje.text(
            superior + 0.0005,
            posicion,
            f"{centro:+.4f} [{inferior:+.4f}, {superior:+.4f}]",
            va="center",
            fontsize=6.8,
            color="#1f1f1f",
        )
    eje.axvline(0.0, color=GRIS, linestyle="--", linewidth=0.8)
    eje.set_yticks(range(len(filas)), [fila[0] for fila in filas])
    eje.set_xlim(-0.001, 0.0235)
    eje.set_xlabel("ΔF1 macro (IC95, bootstrap agrupado)")
    eje.set_title("Mejora local del campeón")
    eje.set_ylim(-0.6, len(filas) - 0.35)

    eje = ejes[1]
    franjas = ["corta", "media", "larga"]
    valores = [CIFRAS["longitud_franjas"][franja]["macro"] for franja in franjas]
    tamanos = [CIFRAS["longitud_franjas"][franja]["n"] for franja in franjas]
    barras = eje.bar(franjas, valores, color=["#548235", "#2e75b6", "#a61c00"], width=0.62)
    for barra, valor, tamano in zip(barras, valores, tamanos):
        eje.text(
            barra.get_x() + barra.get_width() / 2,
            valor + 0.012,
            f"{valor:.3f}\n(n={tamano})",
            ha="center",
            fontsize=6.8,
        )
    eje.set_ylim(0.55, 0.97)
    eje.set_ylabel("F1 macro")
    eje.set_xlabel("franja de longitud")
    eje.set_title("Desempeño por longitud (OOF)")
    figura.tight_layout(w_pad=1.2)
    guardar(figura, "fig_bootstrap")


def main() -> None:
    """Genera todas las figuras."""
    figura_eda()
    figura_phi()
    figura_escalera()
    figura_waterfall()
    figura_umbrales()
    figura_confusion()
    figura_bootstrap()
    print("Figuras generadas en", FIGURAS)


if __name__ == "__main__":
    main()
