"""fig_mtld_por_ano.py
Fecha: 2026-10-31
Descripcion: Genera la figura del deck de la Entrega 3 con el MTLD medio por
    ano del corpus (1992-2021). Sustituye la distribucion multicolor del EDA
    por una serie unica en la paleta del deck: un solo acento azul, grises
    para ejes y etiquetas.
Estado del arte aplicado: matplotlib con la paleta del DESIGN.md, sin
    titulo dentro de la figura (el titulo lo pone la diapositiva), etiquetas
    directas en los extremos y linea de tendencia anotada.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
DIACRONIA = BASE.parents[1] / "SEGUNDA ENTREGA" / "resultados" / "diacronia.json"
SALIDA = BASE / "fig_mtld_por_ano.png"

AZUL = "#2F75B5"
GRIS = "#425466"
REJILLA = "#E5E7EB"


def main() -> None:
    """Dibuja el MTLD medio por ano y guarda el PNG del deck."""
    datos = json.loads(DIACRONIA.read_text(encoding="utf-8"))["mtld"]["por_ano"]
    anios = sorted(int(anio) for anio in datos)
    valores = [float(datos[str(anio)]["mtld_media"]) for anio in anios]
    tendencia = float(
        json.loads(DIACRONIA.read_text(encoding="utf-8"))["mtld"][
            "tendencia_mtld_por_ano"
        ]
    )

    figura, eje = plt.subplots(figsize=(13.0, 4.4), dpi=200)
    figura.patch.set_facecolor("white")
    eje.set_facecolor("white")
    eje.plot(anios, valores, color=AZUL, linewidth=2.4, marker="o", markersize=3.2)
    eje.set_xlim(anios[0] - 1, anios[-1] + 1)
    eje.set_ylim(min(valores) - 1.2, max(valores) + 1.2)
    eje.set_xticks([1992, 1997, 2002, 2007, 2012, 2017, 2021])
    eje.set_ylabel("MTLD medio", color=GRIS, fontsize=13)
    eje.tick_params(colors=GRIS, labelsize=12, length=0)
    for lado in ("top", "right", "left"):
        eje.spines[lado].set_visible(False)
    eje.spines["bottom"].set_color(REJILLA)
    eje.yaxis.grid(True, color=REJILLA, linewidth=0.9)
    eje.set_axisbelow(True)

    eje.annotate(
        f"{valores[0]:.2f}".replace(".", ","),
        (anios[0], valores[0]),
        textcoords="offset points",
        xytext=(10, 12),
        color=AZUL,
        fontsize=14,
        fontweight="bold",
    )
    eje.annotate(
        f"{valores[-1]:.2f}".replace(".", ","),
        (anios[-1], valores[-1]),
        textcoords="offset points",
        xytext=(-42, -6),
        color=AZUL,
        fontsize=14,
        fontweight="bold",
    )
    eje.annotate(
        f"Tendencia {str(round(tendencia, 3)).replace('.', ',')} por ano",
        (anios[len(anios) // 2], min(valores)),
        textcoords="offset points",
        xytext=(-20, -34),
        color=GRIS,
        fontsize=12,
    )

    figura.tight_layout()
    figura.savefig(SALIDA, facecolor="white")
    print(f"Figura: {SALIDA} ({SALIDA.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
