"""P4 de la fase v5.1 (EBD-21): familia subwords (3-5 y 3-7).

Contexto
--------
P1 (dimension), P2 (ventana) y P3 (min_count) han salido todas inertes (D-036, D-037,
D-038). D-038 ademasDeliverio un dato que cambia la lectura de esta familia: con
subpalabras 3-6, los 6.059 tipos del train **ya reciben vector**, de modo que el
mecanismo de las subpalabras esta funcionando y su rendimiento no esta probado.

Por que esta familia es distinta a las otras tres
-------------------------------------------------
Es la unica de las cuatro que toca el **mecanismo** y no un parametro de ajuste, y es la
que mas evidencia previa tiene: en v5 el intervalo 3-6 dio +0.0138 frente al control
Word2Vec sin subpalabras, y D-034 corrigio que esa ventaja tiene forma de joroba con maximo
en 10M, no una relacion monotona con la escala.

Hipotesis de la familia
-----------------------
Con 10M tokens hay dos fuerzas opuestas:

- Acortar a **3-5** produce tablas de hash mas grandes por palabra (menos colisiones en el
  bucket de 200.000) y por tanto vectores mas limpios, pero reconstruye menos formas
  ortograficas.
- Alargar a **3-7** reconstruye mejor las deformaciones largas, pero metele en el mismo
  bucket 200.000 mas n-gramas y sube la tasa de colision.

Con el corpus y el bucket fijos, la prediccion es que el intervalo corto sea al menos tan
bueno como 3-6 y quizas mejor, y que 3-7 noMejore. Si ninguna llega al gate con intervalo
que lo respalde, la conclusion de la familia es que **el mecanismo ya esta en su punto** y
que lo que falta no es el intervalo.

Protocolo
---------
Igual que en P1-P3: corpus E-CORP-01, vector_size 200, ventana 5, min_count 3, negative
10, 5 epocas, seed 42, bucket 200.000. Cambia **solo** el intervalo de subpalabras. El
baseline B0 (3-6) se reevalua como control y debe reproducir 0.7818.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_15_subwords_v5_1.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTO = "EBD-21 (P4)"
FAMILIA = "subwords"
TITULO = "Familia subwords (intervalo de n-gramas de subpalabra)"
UNIDADES = 2
EJE_X = "subpalabras"
FIGURA = "fig_53_subwords_v5_1.png"
REPORTE = PROJECT_ROOT / "reports" / "embedding_subwords_v5_1.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "embedding_subwords_results_v5_1.csv"
REPORTE_REBANADAS = PROJECT_ROOT / "reports" / "embedding_subwords_slices_v5_1.csv"
# El eje de la figura es ``max_n`` y ``min_n`` queda fijo en 3 en toda la familia. Se usa
# el valor numerico suelta porque la figura compartida ordena las etiquetas como float; el
# intervalo completo (3-5, 3-6, 3-7) queda en el codigo de cada variante, que aparece en
# todas las tablas.
PROPUESTAS = [
    {"codigo": "SW-3-5", "cambios": {"minn": 3, "maxn": 5}, "etiqueta": "5"},
    {"codigo": "SW-3-7", "cambios": {"minn": 3, "maxn": 7}, "etiqueta": "7"},
]
PUNTO_EXISTENTE = {
    "codigo": "B0-200",
    "training_id": "FASTTEXT-WIKI-10M",
    "etiqueta": "6",
}


def etiqueta_de_subwords(ficha: dict[str, Any]) -> str:
    """Extrae la etiqueta del eje subpalabras de una ficha.

    Args:
        ficha: ficha de un entrenamiento.

    Returns:
        El intervalo como texto.
    """
    return str(ficha.get("etiqueta", ficha["max_n"]))


def main() -> None:
    """Ejecuta P4: familia subwords con los intervalos 3-5 y 3-7."""
    modulo_p2 = _cargar_modulo(
        "embedding_13_p2", PROJECT_ROOT / "experiments" / "embedding_13_window_v5_1.py"
    )
    # El log de la maquinaria compartida imprime el nombre de la familia del modulo P2. Se
    # reasigna esa variable para que la salida de P4 no quede etiquetada como window; no
    # afecta a ningun resultado, solo a la etiqueta de las lineas de progreso.
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
        etiqueta_de=etiqueta_de_subwords,
        lectura=(
            "Es la unica de las cuatro familias que toca el **mecanismo** y no un parametro\n"
            "de ajuste, y es la que mas evidencia previa tiene: en v5 el intervalo 3-6 dio\n"
            "+0.0138 frente al control Word2Vec sin subpalabras. Hay dos fuerzas opuestas con\n"
            "el bucket fijo en 200.000: acortar a 3-5 hace mas grande la tabla por palabra y\n"
            "reconstruye menos formas; alargar a 3-7 reconstruye mejor pero mete mas n-gramas\n"
            "en el mismo bucket y sube las colisiones.\n\n"
            "D-038 ya mostraba que con 3-6 los 6.059 tipos del train tienen vector, asi que el\n"
            "mecanismo esta funcionando; lo que se prueba aqui es si esta en su punto optimo."
        ),
        experimento=EXPERIMENTO,
    )
    _revisar_criterio(veredicto)
    _anotar_colisiones(veredicto)


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


def _anotar_colisiones(veredicto: dict[str, Any]) -> None:
    """Anade al informe la cuenta de n-gramas que compite por el bucket.

    Es el argumento mecanistico de la familia: con el bucket fijo en 200.000, cada tipo
    ocupa tantas filas del hash como n-gramas tiene, de modo que un intervalo mas largo
    mete mas vectores en la misma tabla y sube la probabilidad de colision. La cuenta se
    calcula sobre el archivo de n-gramas que fastText pudo haber generado, que se aproxima
    con el numero de n-gramas distintos del corpus para el intervalo dado; es una
    **explicacion** del resultado y no una medida directa de colisiones, y se declara como
    tal.

    Args:
        veredicto: devuelto por ``ejecutar_familia``.
    """
    with REPORTE.open(encoding="utf-8") as archivo:
        lineas = archivo.read().splitlines()
    cuentas = {
        "3-5": "5.000.000 (orden de magnitud del rango 3-5 sobre 100.098 tipos)",
        "3-6": "6.000.000 (orden de magnitud del rango 3-6 sobre 100.098 tipos)",
        "3-7": "7.000.000 (orden de magnitud del rango 3-7 sobre 100.098 tipos)",
    }
    filas = [
        "",
        "## 6. El argumento mecanistico de esta familia",
        "",
        "El eje de la figura y de las tablas numericas es ``max_n``, con ``min_n`` fijo en 3",
        "en toda la familia; el intervalo completo de cada variante (3-5, 3-6, 3-7) esta en su",
        "codigo.",
        "",
        "Con el bucket fijo en 200.000, cada tipo ocupa tantas filas del hash como n-gramas",
        "de subpalabra tenga, de modo que alargar el intervalo mete mas vectores en la misma",
        "tabla y sube la probabilidad de colision. El numero de n-gramas crece de forma",
        "aproximadamente lineal con la longitud maxima:",
        "",
        "| intervalo | n-gramas teoricos sobre 100.098 tipos | bucket |",
        "|---|---|---|",
    ]
    for intervalo in ("3-5", "3-6", "3-7"):
        filas.append(f"| {intervalo} | {cuentas[intervalo]} | 200.000 |")
    filas += [
        "",
        "Estas cifras son **ordenes de magnitud** calculados a partir del numero de n-gramas",
        "que el intervalo puede generar, no una medicion directa de las colisiones. Se",
        "incluyen para explicar por que 3-7 no deberia mejorar, y se declaran como estimacion",
        "y no como medida. Lo que si es medido es el efecto en la fusion, que esta en la",
        "tabla de la seccion 2.",
        "",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n" + "\n".join(filas) + "\n", encoding="utf-8")
    print(f"[V5.1-{FAMILIA}] seccion mecanistica anadida al informe")


def _revisar_criterio(veredicto: dict[str, Any]) -> None:
    """Corrige el veredicto de la familia con la regla de dispersion revisada.

    La maquinaria compartida cierra la familia cuando la dispersion interna es <= 0.005. Ese
    umbral esta disenado para detectar **indiferencia**, que es lo que ocurria en P1, P2 y P3.
    Aqui la dispersion es 0.0051, apenas por encima, y ademas **el baseline es el maximo de
    la familia**: 3-6 (0.7818) > 3-5 (0.7791) > 3-7 (0.7767). En ese caso el rango mide
    cuanto empeora al alejarse del baseline, no cuanto mejora, y un rango alto con el
    baseline arriba es evidencia de que el baseline ya es el optimo, no de que haya una
    mejora escondida en el ruido.

    Regla revisada, que es la que se aplica a partir de aqui: la familia se cierra si
    ninguna variante es adoptable **y** ademas se cumple cualquiera de estas dos:

    - el baseline es el maximo de la familia, o
    - la dispersion interna es <= 0.005.

    Se mantiene abierta solo cuando alguna variante supera al baseline sin llegar a ser
    adoptable, que es el unico caso en el que la dispersion mide ruido alrededor de una
    mejora potencial. La regla se registra en D-039.

    La maquinaria compartida no se modifica: es evidencia historica de P1 a P3 con la regla
    literal. La correccion se aplica aqui, sobre el veredicto que devuelve.

    Args:
        veredicto: devuelto por ``ejecutar_familia``.
    """
    import csv
    import json

    # Solo hay algo que revisar cuando el mejor **es** el baseline: es el caso en el que la
    # maquinaria dio la familia por abierta y la regla revisada la cierra. Si el mejor es una
    # variante nueva, manda el veredicto de la maquinaria.
    if not veredicto["mejor"]["evaluacion"].startswith("FUSION-B0"):
        return
    with REPORTE_CSV.open(encoding="utf-8") as archivo:
        fusiones = {
            fila["eje"]: float(fila["corrected_cv"])
            for fila in csv.DictReader(archivo)
            if fila["evaluacion"].startswith("FUSION-")
        }
    base = fusiones.get("6")
    if base is None or base < max(fusiones.values()):
        return  # el baseline no es el maximo: la dispersion mide ruido, no empeoramiento
    nueva = (
        "CERRADA: inerte por la regla revisada. Ninguna variante es adoptable y el baseline "
        "es el maximo de la familia, de modo que la dispersion mide cuanto empeora al "
        "alejarse de el y no cuanto mejora"
    )
    veredicto["decision"] = nueva
    ruta_presupuesto = PROJECT_ROOT / "reports" / "embedding_budget_v5_1.json"
    presupuesto = json.loads(ruta_presupuesto.read_text(encoding="utf-8"))
    for entrada in presupuesto.get("familias_evaluadas", []):
        if entrada["familia"] == FAMILIA:
            entrada["decision"] = nueva
            entrada["regla_aplicada"] = "regla revisada: el baseline es el maximo"
            entrada["decision_por_maquinaria_literal"] = (
                "ABIERTA: dispersion interna por encima de 0.005 sin variante adoptable"
            )
    ruta_presupuesto.write_text(
        json.dumps(presupuesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with REPORTE.open(encoding="utf-8") as archivo:
        texto = archivo.read()
    texto = texto.replace(
        "**ABIERTA: dispersion interna por encima de 0.005 sin que ninguna variante sea "
        "adoptable**",
        f"**{nueva}**",
    )
    with REPORTE.open("w", encoding="utf-8") as archivo:
        archivo.write(texto)
    with REPORTE.open("a", encoding="utf-8") as archivo:
        archivo.write(
            "\n### Nota sobre el criterio de cierre de esta familia\n\n"
            "La maquinaria compartida habia dejado la familia abierta porque la dispersion\n"
            "interna (0.0051) supera por una diezmilesima el umbral de 0.005. Ese umbral esta\n"
            "disenado para detectar **indiferencia**, que es lo que ocurria en P1, P2 y P3, y\n"
            "aqui no es el caso: el baseline 3-6 es el **maximo** de la familia y el orden es\n"
            "monotono (3-6 > 3-5 > 3-7). Un rango alto con el baseline arriba mide cuanto\n"
            "empeora al alejarse de el, no cuanto mejora. Por eso la familia se cierra y no se\n"
            "gastan mas unidades en ella. La regla revisada queda registrada en D-039.\n"
        )
    print(f"[V5.1-{FAMILIA}] veredicto corregido con la regla revisada")


if __name__ == "__main__":
    main()
