"""Genera la Figura 1 del informe: ganancia media de F1 por tamano de ajuste.

Los valores se derivan de la Tabla 3 de Piperno et al. (2025): promedio de los
tres profesores (Bp, Bft, mBft) para cada lengua y tamano de datos.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BASE = Path(__file__).resolve().parent
TAMANOS = [10, 25, 50, 100]

with open(BASE / "datos_ganancia.csv", encoding="utf-8") as archivo:
    filas = list(csv.DictReader(archivo))

datos = {
    fila["lengua"]: [float(fila[str(tamano)]) for tamano in TAMANOS]
    for fila in filas
}

posiciones = range(len(TAMANOS))
colores = {"Español": "#1f77b4", "Francés": "#d95f02", "Alemán": "#2ca02c"}

fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200, constrained_layout=True)

for lengua in ["Español", "Francés", "Alemán"]:
    ax.plot(
        posiciones,
        datos[lengua],
        marker="o",
        markersize=5,
        linewidth=1.8,
        color=colores[lengua],
        label=lengua,
    )

ax.plot(
    posiciones,
    datos["Promedio"],
    marker="s",
    markersize=4,
    linewidth=1.6,
    linestyle="--",
    color="black",
    label="Promedio",
)

ax.axhline(0, color="gray", linewidth=0.8, zorder=0)
ax.set_xticks(list(posiciones), [f"{t} %" for t in TAMANOS])
ax.set_xlabel("Porcentaje de datos de ajuste en la lengua destino")
ax.set_ylabel("Ganancia media de F1 (puntos)")
ax.set_ylim(-2.5, 14.5)
ax.yaxis.set_major_formatter(FuncFormatter(lambda valor, _: f"{valor:g}".replace(".", ",")))
ax.grid(axis="y", color="#dddddd", linewidth=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=9)

for lado in ("top", "right"):
    ax.spines[lado].set_visible(False)

fig.savefig(BASE / "ganancia_media.png", bbox_inches="tight")
print("Figura guardada en", BASE / "ganancia_media.png")
