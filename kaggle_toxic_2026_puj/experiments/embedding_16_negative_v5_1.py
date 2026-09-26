"""P5 de la fase v5.1 (EBD-22): familia negative sampling (5 negativos).

Contexto
--------
Ultima familia arquitectonica y ultima unidad exploratoria del presupuesto (8 de 9
gastadas al empezar esta). Las cuatro anteriores quedaron cerradas por inercia: dimension
(D-036), ventana (D-037), min_count (D-038) y subpalabras (D-039).

Por que esta es distinta
------------------------
Las cuatro anteriores median parametros que cambian **cuantos datos** ve el modelo o **que
formas** reconstruye. ``negative`` cambia otra cosa: como se estiman los vectores de las
palabras que **no** aparecen en el contexto. Con 10 negativos (el valor de v5) la
distribucion de palabras negativas es una muestra de todo el vocabulario, que es
enorme; con 5 la estimacion es mas ruidosa pero el modelo se ajusta un poco mas rapido.

Hipotesis de la familia
-----------------------
``negative`` no deberia mover la aportacion a V03, porque el bloque de V03 que decide la
etiqueta es disperso y ortografico (char n-grams 3-5), no un vector de 200 numeros. Es la
familia con menos probabilidades de Surprise del plan, y se mide precisamente para cerrar
la lista y no dejarlo para el final.

Si apareciera una mejora clara, la unidad de **reserva** queda disponible para
confirmarla con una repeticion; si no, la fase se cierra con 8 de 9 unidades consumidas y
la reserva intacta, que es el uso correcto del presupuesto: gastar lo justo.

Protocolo
---------
Igual que en P1-P4: corpus E-CORP-01, vector_size 200, ventana 5, min_count 3, subpalabras
3-6, 5 epocas, seed 42, bucket 200.000. Cambia **solo** ``negative``. El baseline B0
(negative 10) se reevalua como control y debe reproducir 0.7818.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_16_negative_v5_1.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTO = "EBD-22 (P5)"
FAMILIA = "negative"
TITULO = "Familia negative sampling (muestras negativas por actualizacion)"
UNIDADES = 1
EJE_X = "negative"
FIGURA = "fig_54_negative_v5_1.png"
REPORTE = PROJECT_ROOT / "reports" / "embedding_negative_v5_1.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "embedding_negative_results_v5_1.csv"
REPORTE_REBANADAS = PROJECT_ROOT / "reports" / "embedding_negative_slices_v5_1.csv"
PROPUESTAS = [
    {"codigo": "NEG-5", "cambios": {"negativos": 5}, "etiqueta": "5"},
]
PUNTO_EXISTENTE = {
    "codigo": "B0-200",
    "training_id": "FASTTEXT-WIKI-10M",
    "etiqueta": "10",
}


def etiqueta_de_negative(ficha: dict[str, Any]) -> str:
    """Extrae la etiqueta del eje negative de una ficha.

    Args:
        ficha: ficha de un entrenamiento.

    Returns:
        El numero de negativos como texto.
    """
    return str(ficha.get("etiqueta", ficha["negative"]))


def main() -> None:
    """Ejecuta P5: familia negative sampling con 5 negativos."""
    modulo_p2 = _cargar_modulo(
        "embedding_13_p2", PROJECT_ROOT / "experiments" / "embedding_13_window_v5_1.py"
    )
    # El log de la maquinaria compartida imprime el nombre de la familia del modulo P2; se
    # reasigna para que la salida de P5 no quede etiquetada como window. No afecta a
    # ningun resultado, solo a la etiqueta de las lineas de progreso.
    modulo_p2.FAMILIA = FAMILIA
    veredicto = modulo_p2.ejecutar_familia(
        familia=FAMILIA,
        titulo=TITULO,
        unidades=UNIDADES,
        eje_x=EJE_X,
        figura=FIGURA,
        reporte=REPORTE,
        reporte_csv=REPORTE_CSV,
        reporte_rebanadas=REPORTE_REBANADAS,
        propuestas=PROPUESTAS,
        punto_existente=PUNTO_EXISTENTE,
        etiqueta_de=etiqueta_de_negative,
        lectura=(
            "Las cuatro familias anteriores median parametros que cambian cuantos datos ve el\n"
            "modelo o que formas reconstruye. `negative` cambia otra cosa: como se estiman los\n"
            "vectores de las palabras que **no** aparecen en el contexto.\n\n"
            "La hipotesis es que no deberia mover la aportacion a V03, porque el bloque de V03\n"
            "que decide la etiqueta es disperso y ortografico (char n-grams 3-5), no un vector de\n"
            "200 numeros. Es la familia con menos probabilidades de sorpresa del plan, y se\n"
            "mide precisamente para cerrar la lista y no dejarlo sin medir.\n\n"
            "Con dos puntos (5 frente a 10) no hay dispersion que interpretar: la comparacion\n"
            "es directa y se lee contra el control."
        ),
        experimento=EXPERIMENTO,
    )
    _anotar_cierre(veredicto)


def _cargar_modulo(nombre: str, ruta: Path):
    """Carga un modulo de experimento de forma aislada.

    Args:
        nombre: nombre del modulo en memoria.
        ruta: ruta del archivo.

    Returns:
        El modulo cargado.
    """
    especificacion = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


def _anotar_cierre(veredicto: dict[str, Any]) -> None:
    """Anade al informe el estado del presupuesto tras la ultima familia exploratoria.

    Args:
        veredicto: devuelto por ``ejecutar_familia``.
    """
    import json

    with REPORTE.open(encoding="utf-8") as archivo:
        lineas = archivo.read().splitlines()
    presupuesto = json.loads(
        (PROJECT_ROOT / "reports" / "embedding_budget_v5_1.json").read_text(encoding="utf-8")
    )
    estado = presupuesto["presupuesto"]
    lineas += [
        "",
        "## 7. Estado del presupuesto tras la ultima familia exploratoria",
        "",
        "| contador | valor |",
        "|---|---|",
        f"| entrenamientos existentes de v5 | {estado['existing_v5_trainings']} |",
        f"| entrenamientos nuevos de v5.1 | {estado['new_v5_1_trainings']} |",
        f"| unidades de reserva usadas | {estado['reserved_training_units']} |",
        f"| presupuesto total consumido | {estado['total_new_budget_used']} de "
        f"{estado['limite_exploratorio'] + estado['limite_reserva']} |",
        f"| presupuesto restante | {estado['budget_remaining']} |",
        "",
        "La familia `negative` es la ultima exploratoria del plan. Si queda cerrada por",
        "inercia, la fase no consume la unidad de reserva: la reserva existe para confirmar",
        "con una repeticion una mejora que ya se ha visto, no para abrir una familia nueva.",
        "Gastarla aqui seria gastar por obligacion y no por evidencia.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(
        f"[V5.1-{FAMILIA}] estado del presupuesto anadido: "
        f"{estado['new_v5_1_trainings']}/{estado['limite_exploratorio']} + "
        f"{estado['reserved_training_units']}/{estado['limite_reserva']} reserva"
    )


if __name__ == "__main__":
    main()
