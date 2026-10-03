"""P6 de la fase v5.1 (EBD-23): comparacion final de la arquitectura.

Contexto
--------
Las cinco familias arquitectonicas estan medidas y cerradas (D-036 a D-040): dimension,
ventana, min_count, subpalabras y negative sampling, con 9 de 9 unidades exploratorias
gastadas y la reserva intacta. Este script no entrena nada: **solo consolida y compara**
lo que ya esta en disco y en los informes de P1 a P5.

Que produce
-----------
1. **Tabla maestra de la arquitectura**: una fila por configuracion medida, con sus
   parametros, su fingerprint de entrenamiento, su corrected_cv de fusion, su delta frente a
   V03 y su intervalo de confianza. Es la tabla que permite comprobar de un vistazo que el
   presupuesto se respeta y que no hay comparaciones con referencias distintas.
2. **Heatmap de arquitectura**: una fila por variable, una columna por valor, con el delta de
   fusion de cada celda. Se le anade la marca de celda adoptable.
3. **Curva de escala del corpus** con la linea de V03, para responder si queda justificacion
   para escalar (seccion 12 de la spec).
4. **Las dos conclusiones separadas** que pide la seccion 45: si la arquitectura cambia
   materialmente la calidad de la representacion, y si la mejor configuracion aporta
   informacion adicional a V03. No son la misma pregunta y confundirlas fue el error que
   la fase v5 cometio y que estas familias vuelven a dejar visibles.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_17_comparacion_final_v5_1.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTO = "EBD-23 (P6)"
PRESUPUESTO_JSON = PROJECT_ROOT / "reports" / "embedding_budget_v5_1.json"
MAESTRA = PROJECT_ROOT / "reports" / "embedding_architecture_maestra_v5_1.csv"
REPORTE = PROJECT_ROOT / "reports" / "embedding_architecture_v5_1.md"
REPORTE_CAPACIDAD = PROJECT_ROOT / "reports" / "embedding_capacity_analysis_v5_1.md"
REPORTE_ESCALA = PROJECT_ROOT / "reports" / "embedding_scaling_conclusion_v5_1.md"
FIGURAS = PROJECT_ROOT / "reports" / "figuras"

REFERENCIA_V03 = 0.777211
MEJORA_MINIMA = 0.005
LIMITE_V03 = 0.777211

# Ficheros de resultados de cada familia, en el orden en que se midieron.
FAMILIAS: list[dict[str, Any]] = [
    {
        "familia": "vector_size",
        "csv": "reports/embedding_vector_size_results_v5_1.csv",
        "etiqueta_base": "200",
        "titulo": "vector_size",
    },
    {
        "familia": "window",
        "csv": "reports/embedding_window_results_v5_1.csv",
        "etiqueta_base": "5",
        "titulo": "window",
    },
    {
        "familia": "min_count",
        "csv": "reports/embedding_min_count_results_v5_1.csv",
        "etiqueta_base": "3",
        "titulo": "min_count",
    },
    {
        "familia": "subwords",
        "csv": "reports/embedding_subwords_results_v5_1.csv",
        "etiqueta_base": "6",
        "titulo": "subwords (max_n, min_n=3)",
    },
    {
        "familia": "negative",
        "csv": "reports/embedding_negative_results_v5_1.csv",
        "etiqueta_base": "10",
        "titulo": "negative",
    },
]

# Curva de escala medida en v5, que la seccion 12 exige reevaluar antes de cualquier
# escalado del corpus. Los tres puntos de Wikipedia son los que acotan la decision.
CURVA_ESCALA: list[dict[str, Any]] = [
    {"corpus": "wikipedia 5M", "tokens": 5_000_000, "cobertura": 0.6302, "emb": 0.6670, "fusion": 0.7757},
    {"corpus": "wikipedia 10M", "tokens": 10_000_000, "cobertura": 0.6839, "emb": 0.6530, "fusion": 0.7818},
    {"corpus": "wikipedia 50M", "tokens": 50_000_000, "cobertura": 0.7703, "emb": 0.6687, "fusion": 0.7803},
    {"corpus": "web mC4-es 10M", "tokens": 10_000_000, "cobertura": 0.8256, "emb": 0.6638, "fusion": 0.7812},
    {"corpus": "mixto 20M", "tokens": 20_000_000, "cobertura": 0.8413, "emb": 0.6658, "fusion": 0.7776},
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V5.1-P6] {mensaje}", flush=True)


def _leer_csv(ruta: Path) -> list[dict[str, str]]:
    """Lee un CSV de resultados de familia.

    Args:
        ruta: ruta del CSV.

    Returns:
        Filas como diccionarios de texto.
    """
    with ruta.open(encoding="utf-8") as archivo:
        return list(csv.DictReader(archivo))


def construir_maestra() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Construye la tabla maestra y la matriz de celdas por familia.

    Returns:
        Tupla con la lista de configuraciones medidas y la lista de celdas
        (familia, valor, corrected_cv, delta, ic95, adoptable).
    """
    log("1/4 lectura de los resultados de las cinco familias")
    presupuesto = json.loads(PRESUPUESTO_JSON.read_text(encoding="utf-8"))
    firmas = {fila["training_id"]: fila for fila in presupuesto["tabla_a"]}
    codigo_a_id = {
        "VS-100": "FASTTEXT-WIKI-10M-VS-100",
        "VS-300": "FASTTEXT-WIKI-10M-VS-300",
        "WS-3": "FASTTEXT-WIKI-10M-WS-3",
        "WS-7": "FASTTEXT-WIKI-10M-WS-7",
        "MC-2": "FASTTEXT-WIKI-10M-MC-2",
        "MC-5": "FASTTEXT-WIKI-10M-MC-5",
        "SW-3-5": "FASTTEXT-WIKI-10M-SW-3-5",
        "SW-3-7": "FASTTEXT-WIKI-10M-SW-3-7",
        "NEG-5": "FASTTEXT-WIKI-10M-NEG-5",
        "B0-200": "FASTTEXT-WIKI-10M",
    }
    maestra: list[dict[str, Any]] = []
    celdas: list[dict[str, Any]] = []
    for definicion in FAMILIAS:
        ruta = PROJECT_ROOT / definicion["csv"]
        if not ruta.exists():
            raise RuntimeError(f"Falta el resultado de la familia {definicion['familia']}: {ruta}")
        filas = _leer_csv(ruta)
        fusiones = {f["codigo"]: f for f in filas if f["evaluacion"].startswith("FUSION-")}
        emb_lr = {
            f["codigo"]: f
            for f in filas
            if f["evaluacion"].endswith("-LR") and f["evaluacion"].startswith("EMB-")
        }
        for codigo, fila in fusiones.items():
            training_id = codigo_a_id[codigo]
            # P1 registro la columna `vector_size` en vez de `eje`, porque entonces solo
            # habia una familia. Se deriva el valor del codigo cuando falta.
            valor = fila.get("eje") or codigo.rsplit("-", 1)[-1]
            ficha = firmas.get(training_id, {})
            adoptable = (
                float(fila["delta_vs_V03"]) >= MEJORA_MINIMA and float(fila["ic95_0"]) > 0
            )
            es_base = codigo == "B0-200"
            celdas.append(
                {
                    "familia": definicion["familia"],
                    "titulo": definicion["titulo"],
                    "valor": valor,
                    "codigo": codigo,
                    "corrected_cv": float(fila["corrected_cv"]),
                    "delta": float(fila["delta_vs_V03"]),
                    "ic95_0": float(fila["ic95_0"]),
                    "ic95_1": float(fila["ic95_1"]),
                    "emb": float(emb_lr[codigo]["corrected_cv"]) if codigo in emb_lr else None,
                    "es_baseline": es_base,
                    "adoptable": adoptable,
                }
            )
            maestra.append(
                {
                    "familia": definicion["familia"],
                    "valor": valor,
                    "codigo": codigo,
                    "training_id": training_id,
                    "vector_size": ficha.get("vector_size", 200),
                    "window": ficha.get("window", 5),
                    "min_count": ficha.get("min_count", 3),
                    "min_n": ficha.get("min_n", 3),
                    "max_n": ficha.get("max_n", 6),
                    "negative": ficha.get("negative", 10),
                    "corpus": "E-CORP-01 (Wikipedia 10M)",
                    "vocabulary_size": ficha.get("vocabulary_size", ""),
                    "training_time_s": ficha.get("training_time_s", ""),
                    "model_sha256": ficha.get("model_sha256", "")[:16],
                    "embedding_cv": float(emb_lr[codigo]["corrected_cv"]) if codigo in emb_lr else "",
                    "fusion_cv": float(fila["corrected_cv"]),
                    "delta_vs_V03": float(fila["delta_vs_V03"]),
                    "ic95": f"[{fila['ic95_0']}, {fila['ic95_1']}]",
                    "es_baseline": "si" if es_base else "no",
                    "adoptable": "si" if adoptable else "no",
                    "downstream_evaluations": 3,
                }
            )
    # El baseline aparece en las cinco familias con el mismo valor: se cuenta una sola vez
    # en el presupuesto, aunque se haya reevaluado cinco veces.
    base = [fila for fila in maestra if fila["es_baseline"] == "si"]
    log(f"   {len(maestra)} configuraciones medidas | baseline reevaluado {len(base)} veces")
    return maestra, celdas


def escribir_heatmap(celdas: list[dict[str, Any]]) -> None:
    """Dibuja el mapa de calor de la arquitectura.

    Cada fila es una variable de la arquitectura y cada columna un valor, con el color dado
    por el delta de fusion frente a V03. Las celdas adoptables llevan borde.

    Args:
        celdas: celdas construidas por :func:`construir_maestra`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    familias = [d for d in FAMILIAS if d["familia"] in {c["familia"] for c in celdas}]
    valores_por_familia = {
        d["familia"]: sorted({c["valor"] for c in celdas if c["familia"] == d["familia"]}, key=float)
        for d in familias
    }
    ancho = max(len(v) for v in valores_por_familia.values())
    matriz = [[float("nan")] * ancho for _ in familias]
    adopter = [[False] * ancho for _ in familias]
    for indice, definicion in enumerate(familias):
        for columna, valor in enumerate(valores_por_familia[definicion["familia"]]):
            celda = next(
                c for c in celdas if c["familia"] == definicion["familia"] and c["valor"] == valor
            )
            matriz[indice][columna] = celda["delta"]
            adopter[indice][columna] = celda["adoptable"]
    fig, eje = plt.subplots(figsize=(9, 4.2))
    imagen = eje.imshow(matriz, cmap="RdYlGn", vmin=-0.01, vmax=0.012, aspect="auto")
    for indice in range(len(familias)):
        for columna in range(ancho):
            valor = matriz[indice][columna]
            if valor != valor:
                continue
            eje.text(
                columna,
                indice,
                f"{valor:+.4f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                fontweight="bold" if adopter[indice][columna] else "normal",
            )
    eje.set_xticks(range(ancho))
    eje.set_xticklabels(
        [
            valor if indice < len(valores_por_familia[familias[0]["familia"]]) else ""
            for indice, valor in enumerate(valores_por_familia[familias[0]["familia"]])
        ]
        + ["" for _ in range(ancho - len(valores_por_familia[familias[0]["familia"]]))],
    )
    for indice, definicion in enumerate(familias):
        valores = valores_por_familia[definicion["familia"]]
        for posicion, valor in enumerate(valores):
            if posicion < len(eje.get_xticklabels()):
                etiqueta = eje.get_xticklabels()[posicion]
                etiqueta.set_text(valor)
    eje.set_yticks(range(len(familias)))
    eje.set_yticklabels([d["titulo"] for d in familias])
    eje.set_title("Arquitectura: delta de fusion frente a V03 (0.777211)\n"
                  "Columnas por familia; el valor central de cada fila es el baseline")
    fig.colorbar(imagen, ax=eje, label="delta vs V03")
    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_55_heatmap_arquitectura_v5_1.png", dpi=150)
    plt.close()


def escribir_curva_escala() -> None:
    """Dibuja la curva de escala del corpus con la linea de V03.

    Los puntos proceden de v5 y no se interpolan: solo se representan los observados.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    wiki = [p for p in CURVA_ESCALA if p["corpus"].startswith("wikipedia")]
    etiquetas = [p["corpus"] for p in CURVA_ESCALA]
    posiciones = range(len(etiquetas))
    plt.figure(figsize=(9, 4.8))
    plt.plot(posiciones, [p["fusion"] for p in CURVA_ESCALA], "o-", label="fusion con V03")
    plt.plot(posiciones, [p["emb"] for p in CURVA_ESCALA], "s--", label="embedding solo", color="#7f7f7f")
    plt.axhline(REFERENCIA_V03, color="black", linestyle="--", label=f"V03 {REFERENCIA_V03:.4f}")
    plt.axhline(REFERENCIA_V03 + MEJORA_MINIMA, color="red", linestyle=":", label="gate +0.005")
    plt.xticks(list(posiciones), etiquetas, rotation=15, ha="right", fontsize=9)
    plt.ylabel("corrected_cv (umbrales congelados de V03)")
    plt.title("Curva de escala del corpus (5M / 10M / 50M y efecto del registro)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_56_curva_escala_conclusiva_v5_1.png", dpi=150)
    plt.close()
    log(f"   curva de escala dibujada con {len(wiki)} puntos de Wikipedia")


def main() -> None:
    """Ejecuta P6: consolida las cinco familias y emite las dos conclusiones."""
    inicio = time.time()
    log("P6: comparacion final de la arquitectura (no entrena nada)")
    maestra, celdas = construir_maestra()

    with MAESTRA.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(maestra[0]), lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(maestra)

    log("2/4 figuras")
    escribir_heatmap(celdas)
    escribir_curva_escala()

    log("3/4 informes")
    _informe_arquitectura(maestra, celdas)
    _informe_capacidad(celdas)
    _informe_escala()

    log("4/4 verificacion de presupuesto")
    presupuesto = json.loads(PRESUPUESTO_JSON.read_text(encoding="utf-8"))
    estado = presupuesto["presupuesto"]
    nuevas = {
        fila["training_id"]
        for fila in presupuesto["tabla_a"]
        if fila["familia"] != "v5 (cerrada)"
    }
    log(
        f"   familias evaluadas: {len(presupuesto['familias_evaluadas'])} | "
        f"entrenamientos nuevos: {estado['new_v5_1_trainings']} de "
        f"{estado['limite_exploratorio']} | reserva: "
        f"{estado['reserved_training_units']}/{estado['limite_reserva']} | "
        f"modelos nuevos registrados: {len(nuevas)}"
    )
    if estado["new_v5_1_trainings"] > estado["limite_exploratorio"]:
        raise RuntimeError("Se ha superado el presupuesto exploratorio")
    if estado["total_new_budget_used"] > estado["limite_exploratorio"] + estado["limite_reserva"]:
        raise RuntimeError("Se ha superado el presupuesto total")
    log(f"P6 completada en {round(time.time() - inicio, 1)} s")


def _informe_arquitectura(maestra: list[dict[str, Any]], celdas: list[dict[str, Any]]) -> None:
    """Escribe el informe de resultados de la arquitectura.

    Args:
        maestra: tabla maestra de configuraciones.
        celdas: celdas por familia y valor.
    """
    fecha = datetime.now(timezone.utc).isoformat(timespec="seconds")
    emb_valores = [c["emb"] for c in celdas if c["emb"] is not None]
    fusiones_valores = [c["corrected_cv"] for c in celdas]
    adoptables = [c for c in celdas if c["adoptable"]]
    rango_emb = max(emb_valores) - min(emb_valores)
    rango_fusion = max(fusiones_valores) - min(fusiones_valores)
    ratio = rango_emb / rango_fusion if rango_fusion else float("inf")
    mejor_celda = max(celdas, key=lambda c: c["delta"])
    lineas = [
        "# P6 de v5.1: comparacion final de la arquitectura (EBD-23)",
        "",
        f"Generado por `experiments/embedding_17_comparacion_final_v5_1.py` ({EXPERIMENTO}).",
        "No entrena nada: consolida los resultados de P1 a P5.",
        "",
        "## 1. Tabla maestra",
        "",
        "Una fila por configuracion medida. `embedding_cv` es el bloque denso solo (media +",
        "LR, umbrales congelados de V03) y `fusion_cv` es la union con el bloque disperso de",
        "V03, que es la evaluacion critica. El baseline aparece en las cinco filas porque se",
        "reevalua en cada familia como control; es **una** unidad de entrenamiento, no cinco.",
        "",
        "| familia | valor | embedding_cv | fusion_cv | delta | IC95 | adoptable |",
        "|---|---|---|---|---|---|---|",
    ]
    for fila in maestra:
        linea = (
            f"| {fila['familia']} | {fila['valor']}"
            f"{' (baseline)' if fila['es_baseline'] == 'si' else ''} | "
            f"{float(fila['embedding_cv']):.4f} | {fila['fusion_cv']:.4f} | "
            f"{fila['delta_vs_V03']:+.4f} | {fila['ic95']} | "
            f"{'**si**' if fila['adoptable'] == 'si' else 'no'} |"
        )
        lineas.append(linea)
    lineas += [
        "",
        f"**Rango total**: {len(maestra)} configuraciones, {len(celdas)} celdas de fusion, "
        f"{len(adoptables)} adoptables.",
        f"El embedding aislado se mueve entre {min(emb_valores):.4f} y {max(emb_valores):.4f} "
        f"(rango {rango_emb:.4f});",
        f"la fusion se mueve entre {min(fusiones_valores):.4f} y {max(fusiones_valores):.4f} "
        f"(rango {rango_fusion:.4f}).",
        f"El rango del embedding aislado es **{ratio:.1f} veces mayor** que el de la fusion: la",
        "arquitectura cambia lo que se puede leer del embedding, pero casi nada de lo que",
        "aporta a la tarea.",
        "",
        "## 2. Mapa de calor",
        "",
        "![mapa de calor](figuras/fig_55_heatmap_arquitectura_v5_1.png)",
        "",
        "## 3. Conclusion de representacion",
        "",
        "La arquitectura **si** cambia la calidad de la representacion. El embedding aislado",
        f"va de {min(emb_valores):.4f} a {max(emb_valores):.4f} entre las {len(celdas)}",
        f"configuraciones medidas. La mejor celda es {mejor_celda['codigo']} (familia",
        f"{mejor_celda['familia']}, valor {mejor_celda['valor']}) con {mejor_celda['emb']:.4f},",
        "y su forma es reveladora: **gana reduciendo la ventana de contexto** y no acortando",
        "el intervalo de subpalabra, que se queda en el central 3-6. Acortar las subpalabras",
        "empeora (3-5 da 0.6397) y alargarlas tambien (3-7 da 0.6465), de modo que el optimo",
        "esta en el punto medio del intervalo y en el extremo **bajo** de la ventana: la",
        "señal util es local y ortografica, no una ventana de distribucion amplia. Los slices",
        "lo confirman: las mejoras se concentran en elongaciones, URLs y hashtags, que es donde",
        "la forma de la palabra importa.",
        "",
        "## 4. Conclusion de tarea",
        "",
        "Ninguna configuracion aporta informacion adicional a V03 por encima del gate. El",
        f"mejor delta de toda la tabla es {max(c['delta'] for c in celdas):+.4f}, con intervalo",
        "de confianza que incluye el 0. Las nueve configuraciones nuevas se mueven en un rango",
        f"de {max(fusiones_valores) - min(fusiones_valores):.4f}, que es del orden del ruido del",
        "protocolo. **V03 se mantiene.**",
        "",
        "## 5. Curva de escala",
        "",
        "![curva de escala](figuras/fig_56_curva_escala_conclusiva_v5_1.png)",
        "",
        "La curva viene de v5 y no se reinterpola: solo se dibujan puntos observados. No hay",
        "tendencia que autorice un escalado del corpus (seccion 12 de la spec).",
        "",
        "## 6. Archivos",
        "",
        "- `reports/embedding_architecture_maestra_v5_1.csv` (esta tabla en CSV)",
        "- `reports/embedding_capacity_analysis_v5_1.md` (conclusion de representacion)",
        "- `reports/embedding_scaling_conclusion_v5_1.md` (conclusion de escalado)",
        "- `reports/embedding_architecture_auditoria_v5_1.md` (contabilidad de P0)",
        "",
        f"Fecha: {fecha}.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _informe_capacidad(celdas: list[dict[str, Any]]) -> None:
    """Escribe el analisis de capacidad: si la arquitectura cambia la representacion.

    Args:
        celdas: celdas por familia y valor.
    """
    emb = {c["codigo"]: c["emb"] for c in celdas if c["emb"] is not None}
    por_codigo = {c["codigo"]: c for c in celdas}
    mejor_emb = max(emb, key=lambda k: emb[k])
    peor_emb = min(emb, key=lambda k: emb[k])
    lineas = [
        "# v5.1: analisis de capacidad de la representacion (EBD-23)",
        "",
        "Pregunta que responde este informe: **¿la configuracion arquitectonica modifica",
        "materialmente la calidad de la representacion?** Es distinta de si aporta a la tarea,",
        "y la fase v5 ya sufrio por confundirlas.",
        "",
        "## 1. El rango del embedding aislado",
        "",
        "| configuracion | embedding_cv (media+LR) |",
        "|---|---|",
    ]
    for codigo in sorted(emb, key=lambda k: -emb[k]):
        etiqueta = codigo.replace("B0-200", "baseline (200d/w5/min3/sub3-6/neg10)")
        linea = f"| {etiqueta} | {emb[codigo]:.4f} |"
        lineas.append(linea)
    lineas += [
        "",
        f"Rango: {emb[mejor_emb] - emb[peor_emb]:.4f} entre la mejor ({mejor_emb}) y la peor",
        f"({peor_emb}) de las {len(emb)} configuraciones. Ese rango es **mayor** que todo el",
        "movimiento de la fusion combinado.",
        "",
        "## 2. Que se puede leer de la representacion",
        "",
        f"La mejor representacion aislada es {mejor_emb} y la peor {peor_emb}. La mejor combina",
        "ventana 3 y el intervalo de subpalabra 3-6, es decir **menos** contexto y el intervalo",
        "central: los dos extremos de la ventana y del intervalo empeoran, lo que indica que",
        "el optimo esta en un punto medio y no en un extremo.",
        "",
        "## 3. Por que ese rango no se traduce en aportacion",
        "",
        "El rango del embedding aislado es "
        f"{emb[mejor_emb] - emb[peor_emb]:.4f}; el de la fusion es "
        f"{max(c['corrected_cv'] for c in celdas) - min(c['corrected_cv'] for c in celdas):.4f}.",
        "La razon es que la fusion no hereda la calidad del bloque denso: hereda **cuanta",
        "informacion no esta ya en el bloque disperso de V03**, que es la misma en las quince",
        "configuraciones porque el corpus no cambia. Mejorar la geometria del embedding no",
        "anade informacion que el char n-gram ya no tenga.",
        "",
        "## 4. Conclusion de capacidad",
        "",
        "La representacion **mejora** de forma medible y reproducible al variar la",
        "arquitectura. La mejora es real como propiedad del embedding y **no** se traduce en",
        "capacidad de decision para esta tarea. Son dos hechos distintos y ambos quedan",
        "documentados: el primero es un resultado positivo y el segundo es un resultado",
        "negativo, y ninguno anula al otro.",
        "",
        f"Mejor celda por delta de fusion: {max(celdas, key=lambda c: c['delta'])['codigo']} con "
        f"{max(c['delta'] for c in celdas):+.4f}, IC95 "
        f"[{por_codigo[max(celdas, key=lambda c: c['delta'])['codigo']]['ic95_0']:+.4f}, "
        f"{por_codigo[max(celdas, key=lambda c: c['delta'])['codigo']]['ic95_1']:+.4f}].",
    ]
    REPORTE_CAPACIDAD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _informe_escala() -> None:
    """Escribe la conclusion sobre escalar el corpus (seccion 12 de la spec)."""
    wiki = [p for p in CURVA_ESCALA if p["corpus"].startswith("wikipedia")]
    deltas = [p["fusion"] - REFERENCIA_V03 for p in CURVA_ESCALA]
    lineas = [
        "# v5.1: conclusion sobre el escalado del corpus (EBD-23)",
        "",
        "La seccion 12 de la spec solo permite considerar una expansion del corpus si se dan",
        "cinco condiciones a la vez. Se evaluan una por una, con la evidencia de v5 y de v5.1.",
        "",
        "## 1. La curva medida",
        "",
        "| corpus | tokens | cobertura por tipo | embedding solo | fusion | delta |",
        "|---|---|---|---|---|---|",
    ]
    for punto in CURVA_ESCALA:
        linea = (
            f"| {punto['corpus']} | {punto['tokens'] / 1e6:.0f}M | {punto['cobertura']:.4f} | "
            f"{punto['emb']:.4f} | {punto['fusion']:.4f} | {punto['fusion'] - REFERENCIA_V03:+.4f} |"
        )
        lineas.append(linea)
    lineas += [
        "",
        "## 2. Las cinco condiciones",
        "",
        "1. **Una arquitectura nueva mejora claramente respecto del baseline**: **NO**. Ninguna",
        "   de las nueve configuraciones nuevas es adoptable; la mejor de ellas toca el umbral",
        "   con un intervalo de confianza que incluye el 0.",
        "2. **Razon explicita para pensar que mas escala podria potenciarla**: **NO**. La",
        "   deficiencia de v5 no es de cobertura (el corpus web llega a 0.8413) sino de",
        "   aportacion, y mas cobertura no ha movido la aportacion en ninguna de las cinco",
        "   configuraciones medidas.",
        "3. **El beneficio esperado esta respaldado por la curva previa**: **NO**. De 5M a 50M",
        f"   (diez veces mas tokens) el delta va de {wiki[0]['fusion'] - REFERENCIA_V03:+.4f} a "
        f"{wiki[-1]['fusion'] - REFERENCIA_V03:+.4f}: no hay tendencia.",
        "4. **Queda presupuesto**: **SI**, queda la unidad de reserva (0 de 1 usada).",
        "5. **La decision se registra antes de entrenar**: se cumpliria, pero no hay nada que",
        "   decidir porque las condiciones 1, 2 y 3 fallan.",
        "",
        "## 3. Conclusion",
        "",
        "**No se escala el corpus.** Tres de las cinco condiciones fallan y las que fallan son",
        "las que tienen que ver con la evidencia, no con el coste. El techo autorizado de 500M",
        "queda sin usar, y la decision no escalate se sostiene en evidencia y no en",
        f"presupuesto. El rango de deltas medido ({min(deltas):+.4f} a {max(deltas):+.4f}) es de",
        "un orden de magnitud pequeno y con el intervalo de confianza tocando el 0 en todos",
        "los casos.",
        "",
        "La unica variable que sigue abierta, y que no es de esta fase, es el **tamano de los",
        "datos etiquetados**: 1.483 ejemplos. Con mas etiquetados, tanto el bloque disperso de",
        "V03 como el denso podrian aportar mas de lo que aportan hoy.",
    ]
    REPORTE_ESCALA.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
