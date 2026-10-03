"""P3 de la fase v5.1 (EBD-20): familia min_count (2 y 5).

Contexto
--------
P1 cerro ``vector_size`` inerte y P2 cerro ``window`` inerte (D-036, D-037). La tercera
familia es la mas interesante de las tres, porque ``min_count`` es el unico parametro de
la lista que cambia **cuantos tipos existen**, y por tanto los unicos dos que pueden
mover la cobertura del vocabulario TOXIC: los tipos que caen por debajo del umbral se
quedan sin vector propio y pasan a depender de la composicion por subpalabras.

Hipotesis de la familia
-----------------------
Hay dos efectos opuestos y el resultado dira cual pesa:

- **A favor de min_count = 2**: mas tipos entran en el vocabulario, luego mas palabras de
  TOXIC deformadas tienen vector propio en vez de uno compuesto. En v5 se midio que el
  66,6 % de los tipos ausentes del corpus son evasiones con perdida de vocales, y esas son
  justo subpalabra.
- **En contra de min_count = 2**: cada tipo con frecuencia 2 aporta una estimacion ruidosa
  de su vector, y con 1.483 ejemplos etiquetados el clasificador ya no es el cuello de
  botella; anadir ruido al bloque denso puede empeorar la fusion.

Si ninguno de los dos lados mueve la fusion mas de 0.005, la conclusion es que la
representacion esta saturada y que el limite es el lado supervisado, no el no supervisado.

Protocolo
---------
Igual que en P1 y P2: corpus E-CORP-01, vector_size 200, ventana 5, subpalabras 3-6,
negative 10, 5 epocas, seed 42, bucket 200.000. Cambia **solo** ``min_count``. El
baseline B0 (min_count 3) se reevalua como control y debe reproducir 0.7818.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_14_min_count_v5_1.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTO = "EBD-20 (P3)"
FAMILIA = "min_count"
TITULO = "Familia min_count (frecuencia minima para entrar en el vocabulario)"
UNIDADES = 2
EJE_X = "min_count"
FIGURA = "fig_52_min_count_v5_1.png"
REPORTE = PROJECT_ROOT / "reports" / "embedding_min_count_v5_1.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "embedding_min_count_results_v5_1.csv"
REPORTE_REBANADAS = PROJECT_ROOT / "reports" / "embedding_min_count_slices_v5_1.csv"
PROPUESTAS = [
    {"codigo": "MC-2", "cambios": {"min_count": 2}, "etiqueta": "2"},
    {"codigo": "MC-5", "cambios": {"min_count": 5}, "etiqueta": "5"},
]
PUNTO_EXISTENTE = {
    "codigo": "B0-200",
    "training_id": "FASTTEXT-WIKI-10M",
    "etiqueta": "3",
}


def etiqueta_de_min_count(ficha: dict[str, Any]) -> str:
    """Extrae la etiqueta del eje min_count de una ficha.

    Args:
        ficha: ficha de un entrenamiento.

    Returns:
        El valor de min_count como texto.
    """
    return str(ficha.get("etiqueta", ficha["min_count"]))


def main() -> None:
    """Ejecuta P3: familia min_count con umbral 2 y 5."""
    modulo_p2 = _cargar_modulo(
        "embedding_13_p2", PROJECT_ROOT / "experiments" / "embedding_13_window_v5_1.py"
    )
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
        etiqueta_de=etiqueta_de_min_count,
        lectura=(
            "`min_count` es el unico parametro de la lista que cambia **cuantos tipos\n"
            "existen**, y por tanto los unicos que pueden mover la cobertura del vocabulario\n"
            "TOXIC. Hay dos efectos opuestos y el resultado dice cual pesa. A favor de 2: mas\n"
            "tipos entran, luego mas palabras de TOXIC deformadas tienen vector propio en vez de\n"
            "uno compuesto por subpalabras, y en v5 se midio que el 66,6 % de los tipos ausentes\n"
            "son evasiones con perdida de vocales. En contra de 2: cada tipo con frecuencia 2\n"
            "aporta una estimacion ruidosa de su vector, y con 1.483 ejemplos el lado\n"
            "supervisado ya es el cuello de botella.\n\n"
            "El numero de tipos con vector es la cifra que hay que mirar en esta familia: es la\n"
            "unica de las tres que puede cambiarlo, porque con el corpus y la ventana fijos solo\n"
            "el umbral de frecuencia decide que palabras tienenrepresentacion propia."
        ),
        experimento=EXPERIMENTO,
    )
    _anotar_cobertura(veredicto)


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


def _anotar_cobertura(veredicto: dict[str, Any]) -> None:
    """Anade al informe la cifra que esta familia si mueve: el vocabulario del modelo.

    La cobertura del **corpus** (0.6839 por tipo) es identica en toda la familia, porque el
    corpus no cambia. Lo que ``min_count`` mueve es el numero de tipos que reciben vector
    **propio**, y con el la parte de la cobertura que depende del modelo. Para el baseline se
    usa el valor medido en v5 (6.059 de 6.059 con FastText y subpalabras 3-6), que no
    estaba en la Tabla A de P0.

    Args:
        veredicto: devuelto por ``ejecutar_familia``.
    """
    import json

    with REPORTE.open(encoding="utf-8") as archivo:
        lineas = archivo.read().splitlines()
    presupuesto = json.loads(
        (PROJECT_ROOT / "reports" / "embedding_budget_v5_1.json").read_text(encoding="utf-8")
    )
    por_id = {fila["training_id"]: fila for fila in presupuesto["tabla_a"]}
    esperados = [
        ("2", "FASTTEXT-WIKI-10M-MC-2"),
        ("3", "FASTTEXT-WIKI-10M"),
        ("5", "FASTTEXT-WIKI-10M-MC-5"),
    ]
    filas = [
        "",
        "## 6. Que mueve esta familia",
        "",
        "La cobertura del **corpus** (0.6839 por tipo, 0.8833 por token) es identica en toda",
        "la familia porque el corpus no cambia. Lo que `min_count` mueve es cuantos de esos",
        "tipos reciben vector **propio** del modelo, y con ello la parte de la cobertura que",
        "depende del modelo y no del corpus. Ese es el numero que hay que mirar aqui.",
        "",
        "| min_count | tipos en el vocabulario del modelo | tipos del train con vector |",
        "|---|---|---|",
    ]
    for etiqueta, training_id in esperados:
        fila = por_id.get(training_id)
        if fila is None:
            continue
        con_vector = fila.get("n_tipos_con_vector")
        if con_vector in (None, ""):
            con_vector = "6059 (medido en v5 con subpalabras 3-6)"
        filas.append(
            f"| {etiqueta} | {fila.get('vocabulary_size', '?')} | {con_vector} |"
        )
    filas += [
        "",
        "Un umbral mas bajo agranda el vocabulario, pero cada tipo con frecuencia 2 aporta una",
        "estimacion ruidosa de su vector. Un umbral mas alto hace lo contrario. La tabla de la",
        "seccion 2 dice cual de los dos efectos pesa en la fusion.",
        "",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n" + "\n".join(filas) + "\n", encoding="utf-8")
    print(f"[V5.1-{FAMILIA}] seccion de cobertura anadida al informe")


if __name__ == "__main__":
    main()
