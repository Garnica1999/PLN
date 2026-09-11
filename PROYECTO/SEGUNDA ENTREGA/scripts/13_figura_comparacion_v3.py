"""13_figura_comparacion_v3.py
Fecha: 2026-09-10
Descripcion: Figura comparativa del ciclo v2 - Fase S1 (plan/
    PLAN_v2_ciclo_mejora.md): barras agrupadas de F1@5, MRR y nDCG@10 de
    la estrategia bm25 con el indice SUCIO (v3, Fase R) vs el indice
    LIMPIO de articulos permanentes (v3_limpio, Fase S1). Lee
    resultados/vinculo_cp_benchmark_v3.json y
    resultados/vinculo_cp_benchmark_v3_limpio.json SOLO en lectura
    (ADR-11: los benchmarks previos no se tocan) y genera
    figuras/comparacion_v3_sucio_vs_limpio.png. Imprime ademas la tabla
    comparativa de las tres estrategias de indice (bm25, e5, hibrido)
    para el reporte de la fase.
Estado del arte aplicado: evidencia cuantitativa de la limpieza del
    indice CP (ADR-12): el diagnostico de la Fase R midio 24.4% de
    articulos transitorios en top-5 contaminando el ranking; esta figura
    documenta la mejora de retrieval tras excluir transitorios y
    preambulo del indice.
"""

import json
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIG = {
    "ruta_sucio": "resultados/vinculo_cp_benchmark_v3.json",
    "ruta_limpio": "resultados/vinculo_cp_benchmark_v3_limpio.json",
    "ruta_figura": "figuras/comparacion_v3_sucio_vs_limpio.png",
    "estrategia_figura": "bm25",
    "medidas": ["f1_at_k", "mrr", "ndcg_at_10"],
    "etiquetas_medidas": {
        "f1_at_k": "F1@5",
        "mrr": "MRR",
        "ndcg_at_10": "nDCG@10",
    },
    "estrategias_tabla": ["bm25", "e5", "hibrido"],
    "colores": {"sucio": "#ff7f0e", "limpio": "#1f77b4"},
}

ROOT = Path(__file__).resolve().parent.parent


def cargar_globales(ruta: str) -> dict:
    """Lee el bloque global de un JSON de benchmark (solo lectura).

    Args:
        ruta: Ruta relativa a la raiz del proyecto del JSON de benchmark.

    Returns:
        Dict {estrategia: {medida: valor}} del campo global.

    Raises:
        FileNotFoundError: Si el JSON no existe (se exige haber corrido
            antes las Fases R y S1 del script 12).
    """
    archivo = ROOT / ruta
    if not archivo.is_file():
        raise FileNotFoundError(
            f"No existe {archivo}: ejecute antes "
            "scripts/12_evaluar_vinculo_cp_v2.py (Fases R y S1)."
        )
    with archivo.open("r", encoding="utf-8") as manejador:
        return json.load(manejador)["global"]


def graficar_comparacion(sucio: dict, limpio: dict) -> None:
    """Genera la figura de barras agrupadas sucio vs limpio (bm25).

    Args:
        sucio: Bloque global del benchmark v3 (indice sucio-v1).
        limpio: Bloque global del benchmark v3_limpio (permanentes-v3).
    """
    estrategia = CONFIG["estrategia_figura"]
    medidas = CONFIG["medidas"]
    valores_sucio = [sucio[estrategia][m] for m in medidas]
    valores_limpio = [limpio[estrategia][m] for m in medidas]

    posiciones = range(len(medidas))
    ancho = 0.35
    figura, eje = plt.subplots(figsize=(8, 5))
    barras_sucio = eje.bar(
        [p - ancho / 2 for p in posiciones],
        valores_sucio,
        ancho,
        label="Indice sucio-v1 (v3, Fase R)",
        color=CONFIG["colores"]["sucio"],
    )
    barras_limpio = eje.bar(
        [p + ancho / 2 for p in posiciones],
        valores_limpio,
        ancho,
        label="Indice permanentes-v3 (v3_limpio, Fase S1)",
        color=CONFIG["colores"]["limpio"],
    )
    eje.bar_label(barras_sucio, fmt="%.3f", fontsize=9, padding=2)
    eje.bar_label(barras_limpio, fmt="%.3f", fontsize=9, padding=2)
    eje.set_xticks(list(posiciones))
    eje.set_xticklabels([CONFIG["etiquetas_medidas"][m] for m in medidas])
    eje.set_ylabel("Valor de la metrica")
    eje.set_xlabel("Metrica de retrieval (k=5 / k=10)")
    eje.set_title(
        "BM25 sentencia->articulos CP: indice sucio vs limpio "
        "(90 casos, muestra semilla 42)",
        fontsize=11,
    )
    eje.set_ylim(0, max(valores_sucio + valores_limpio) * 1.2)
    eje.grid(True, axis="y", alpha=0.3)
    eje.legend(fontsize=9)
    figura.tight_layout()
    ruta = ROOT / CONFIG["ruta_figura"]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(figura)
    print(f"[ok] figura guardada: {ruta}")


def imprimir_tabla(sucio: dict, limpio: dict) -> None:
    """Imprime la tabla comparativa sucio vs limpio por estrategia.

    Args:
        sucio: Bloque global del benchmark v3 (indice sucio-v1).
        limpio: Bloque global del benchmark v3_limpio (permanentes-v3).
    """
    print("\n=== Comparativa v3 sucio vs v3 limpio (global) ===")
    for estrategia in CONFIG["estrategias_tabla"]:
        print(f"  {estrategia}:")
        for medida in CONFIG["medidas"]:
            vs = sucio[estrategia][medida]
            vl = limpio[estrategia][medida]
            delta = vl - vs
            print(
                f"    {CONFIG['etiquetas_medidas'][medida]:8s} "
                f"sucio={vs:.4f} limpio={vl:.4f} delta={delta:+.4f}"
            )


def main() -> None:
    """Orquesta la figura comparativa de la Fase S1 (solo lectura)."""
    sucio = cargar_globales(CONFIG["ruta_sucio"])
    limpio = cargar_globales(CONFIG["ruta_limpio"])
    graficar_comparacion(sucio, limpio)
    imprimir_tabla(sucio, limpio)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"Error fatal: {exc}")
        sys.exit(1)
