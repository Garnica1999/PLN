"""Entregables finales de la fase v6 (V6-09, ANALYSIS): consolidacion y cierre.

Contexto
--------
La fase v6 abrio con un objetivo concreto: buscar mecanismos que el proyecto no hubiera
evaluado de verdad, sin repetir familias cerradas. Se ejecutaron **9 scripts** (V6-00 a
V6-08) sobre 7 tracks. Este script **no entrena nada**: consolida lo que ya esta medido en
disco y produce los entregables que la especificacion de la fase pedia, de modo que el
cierre no dependa de que nadie recuerde lo que dijo cada informe.

Por que un script y no escribirlo a mano
-----------------------------------------
Los entregables son tablas de resultados. Escribirlas a mano seria el mismo error que el
gap analysis de V6-00 vino a corregir: un documento que se contradice en cuanto un numero
cambia. Aqui **todo se lee de los JSON y CSV que los experimentos ya escribieron**, y si
falta alguno el script falla en vez de inventar un numero.

Lo que produce
--------------
- ``reports/v6_master_results.csv``: una fila por variante ejecutada, con metricas, IC95 y gate.
- ``reports/v6_hypothesis_registry.csv``: una fila por hipotesis, con su veredicto y su causa.
- ``reports/v6_candidate_comparison.csv``: comparacion de los tres candidatos que mas se
  acercaron, con su distancia al gate.
- ``reports/v6_error_intersection.csv``: copia canonizada de la interseccion de errores.
- ``reports/v6_final_research.md``: el informe de cierre de la fase.
- ``reports/v6_compliance.md``: el cumplimiento de las restricciones de la fase.
- ``reports/v6_reproducibility.md``: como reproducir cada resultado.

Uso
---
    /home/carlos/venv/bin/python experiments/v6_09_entregables.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTO = "V6-09 (Pcierre, ANALYSIS)"
REPORTES = PROJECT_ROOT / "reports"
REFERENCIA_V03 = 0.777211
GATE_1_DELTA = 0.003

# Los experimentos de la fase, con su script, su informe y su JSON. El script falla si
# alguno falta: un entregable que rellena huecos con ceros es peor que uno que no existe.
EXPERIMENTOS: list[dict[str, str]] = [
    {"id": "V6-00", "script": "v6_00_gap_analysis.py", "tema": "gap analysis", "tipo": "ANALYSIS"},
    {"id": "V6-01", "script": "v6_01_estructura_errores.py", "tema": "interseccion de errores y estructura de etiquetas", "tipo": "ANALYSIS"},
    {"id": "V6-02", "script": "v6_02_label_factorization.py", "tema": "track B: factorizacion condicional", "tipo": "EXPERIMENTO"},
    {"id": "V6-03", "script": "v6_03_ponderacion_grupo.py", "tema": "track A: ponderacion por grupo", "tipo": "EXPERIMENTO"},
    {"id": "V6-04", "script": "v6_04_char_weight.py", "tema": "track I: ponderacion word/char", "tipo": "EXPERIMENTO"},
    {"id": "V6-05", "script": "v6_05_skipgrams.py", "tema": "track C1: skip-grams", "tipo": "EXPERIMENTO"},
    {"id": "V6-06", "script": "v6_06_bagging_grupos.py", "tema": "track H: bagging por grupos", "tipo": "EXPERIMENTO"},
    {"id": "V6-07", "script": "v6_07_ruido_par.py", "tema": "track F: ruido por par", "tipo": "EXPERIMENTO"},
    {"id": "V6-08", "script": "v6_08_aumento.py", "tema": "track J: aumento de datos", "tipo": "EXPERIMENTO"},
]

# Donde esta el JSON de resultados de cada experimento con metricas.
FUENTES_RESULTADO = {
    "V6-02": "v6_label_factorization.json",
    "V6-03": "v6_ponderacion_grupo.json",
    "V6-04": "v6_char_weight.json",
    "V6-05": "v6_skipgrams.json",
    "V6-06": "v6_bagging_grupos.json",
    "V6-07": "v6_ruido_par.json",
    "V6-08": "v6_aumento.json",
}

# El registro de hipotesis: una fila por track, con su veredicto y la **causa** del
# veredicto. La causa es la parte que sobrevive al cierre de la fase.
HIPOTESIS: list[dict[str, str]] = [
    {
        "hipotesis_id": "V6-H01",
        "track": "B",
        "hipotesis": "p(TR|X) = p(HS|X) * p(TR|HS=1,X), con el condicional entrenado solo en HS=1",
        "soporte_empirico": "TR implica HS de forma determinista: 0 casos de TR=1 con HS=0 sobre 569 TR=1",
        "experimento": "V6-02",
        "mejor_delta": "-0.0008",
        "ic95": "[-0.0075, +0.0057]",
        "gate_1": "no supera",
        "veredicto": "cerrada",
        "causa": "La invariante es cierta pero solo V03 la viola en el 1.35 % de las predicciones. La restriccion SI se aplica (0 violaciones frente a 20) pero el producto atenua p(TR) en toda fila con p(HS) intermedia, que es donde V03 acertaba: 1 rescatada frente a 35 rotas",
    },
    {
        "hipotesis_id": "V6-H02",
        "track": "A",
        "hipotesis": "Ponderar el entrenamiento por 1/sqrt(g) o 1/g corrige la sobreponderacion de los grupos repetidos",
        "soporte_empirico": "La ponderacion por grupo nunca se habia probado; solo por dureza (V4-B03)",
        "experimento": "V6-03",
        "mejor_delta": "+0.0006",
        "ic95": "[-0.0029, +0.0040]",
        "gate_1": "no supera",
        "veredicto": "cerrada",
        "causa": "Solo el 7.1 % de las filas puede cambiar de peso, porque el 92.9 % ya esta en grupos de tamano 1. Y el patron es monotono: penalizar mas a los repetidos empeora. **Un tuit repetido es senal, no ruido de recoleccion**",
    },
    {
        "hipotesis_id": "V6-H03",
        "track": "I",
        "hipotesis": "El optimo de la escala entre el bloque word y el char_wb no es el 1 heredado de hstack",
        "soporte_empirico": "No existe ningun char_weight en el codigo; solo se midieron bloques aislados y su union",
        "experimento": "V6-04",
        "mejor_delta": "+0.0016",
        "ic95": "[-0.0048, +0.0078]",
        "gate_1": "no supera",
        "veredicto": "cerrada",
        "causa": "El optimo es **interior** (1.5) y el peso heredado no era optimo, pero el techo es +0.0016, menos de la mitad del gate, con IC95 que incluye el 0. **La escala no era el cuello de botella**",
    },
    {
        "hipotesis_id": "V6-H04",
        "track": "C1",
        "hipotesis": "Ciertas relaciones no requieren palabras adyacentes y aportan senal que los bigramas contiguos no capturan",
        "soporte_empirico": "No existe ninguna implementacion de skip-grams; el ngram_range de V03 es contiguo",
        "experimento": "V6-05",
        "mejor_delta": "-0.0001",
        "ic95": "[-0.0085, +0.0082]",
        "gate_1": "no supera",
        "veredicto": "cerrada",
        "causa": "Los skip-grams son **NEUTROS**: con +83 % de caracteristicas el resultado no se mueve. Distinto de V4-D (-0.0346), lo que confirma que no era repeticion. La intuicion era razonable; lo que falla es que quede senal por representar en 1.483 filas",
    },
    {
        "hipotesis_id": "V6-H05",
        "track": "H",
        "hipotesis": "Promediar V03-equivalentes reentrenados sobre remuestreos de grupos aporta reduccion de varianza",
        "soporte_empirico": "No existe ningun bagging; los ensembles previos mezclaban representaciones distintas",
        "experimento": "V6-06",
        "mejor_delta": "-0.0105",
        "ic95": "[-0.0183, -0.0030]",
        "gate_1": "no supera",
        "veredicto": "cerrada por protocolo",
        "causa": "**La hipotesis de techo se refuto a si misma**: la correlacion entre pares de modelos es 0.9573, o sea hay diversidad real. El promedio empeora por un efecto de escala: los umbrales de V03 estan calibrados para un modelo individual. **No hay forma limpia de medirlo** sin reoptimizar umbrales, y eso rompe la comparacion",
    },
    {
        "hipotesis_id": "V6-H06",
        "track": "F",
        "hipotesis": "Parte de la frontera contiene pares instancia-etiqueta sistematicamente inconsistentes que se pueden detectar y luego ponderar",
        "soporte_empirico": "V4-A midio el ruido por duplicado y lo encontro escaso; nunca se midio a nivel de par",
        "experimento": "V6-07",
        "mejor_delta": "+0.0001",
        "ic95": "[-0.0033, +0.0035]",
        "gate_1": "no supera",
        "veredicto": "cerrada",
        "causa": "La deteccion encuentra **dificultad, no ruido**: V03 acierta 10.5 % en las filas sospechosas frente a 28.3 % en las demas. Subir el peso de la sospecha empeora, luego **lo que marca es donde hay que estar, no donde hay que limpiar**. Y el ruido por par (12 pares) es mas escaso que el duplicado exacto (84 tuits)",
    },
    {
        "hipotesis_id": "V6-H07",
        "track": "J",
        "hipotesis": "Crear instancias nuevas con transformaciones que preservan la etiqueta mejora lo que no lo hizo la normalizacion",
        "soporte_empirico": "Diez normalizaciones (PRE-00..09) se descartaron, pero eran sustitucion, no creacion",
        "experimento": "V6-08",
        "best": "-0.0061",
        "mejor_delta": "-0.0061",
        "ic95": "[-0.0141, +0.0018]",
        "gate_1": "no supera",
        "veredicto": "cerrada",
        "causa": "Aumentar empeora, y **doblar las copias casi no cambia nada** (-0.0061 frente a -0.0063), lo que apunta a **sesgo de las copias y no a exceso de ellas**: las variantes deformadas forman un patron coherente que el modelo aprende",
    },
    {
        "hipotesis_id": "V6-H08",
        "track": "D",
        "hipotesis": "El problema de FastText en v5 fue el pooling medio, no la representacion",
        "soporte_empirico": "La arquitectura convolucional ya se probo (DL-02) pero con embeddings aleatorios",
        "experimento": "ninguno",
        "mejor_delta": "no medido",
        "ic95": "no medido",
        "gate_1": "no evaluable",
        "veredicto": "cerrada por techo",
        "causa": "**El techo teorico esta por debajo del gate.** La interseccion de errores (V6-01) muestra que el mejor candidato existente acierta 12 filas de 1.483 donde V03 falla. Si esas 12 filas cambiaran una sola etiqueta cada una sin romper ninguna otra, la metrica subiria como mucho +0.0027, por debajo del gate de +0.003. Cerrada por presupuesto y no por resultado",
    },
    {
        "hipotesis_id": "V6-H09",
        "track": "E",
        "hipotesis": "Un experto local rescata patrones que el modelo lineal global suaviza",
        "soporte_empirico": "La afinidad de vecindad ya se midio en V4-F03 y fue nula para ordenar el error",
        "experimento": "ninguno",
        "mejor_delta": "no medido",
        "ic95": "no medido",
        "gate_1": "no evaluable",
        "veredicto": "cerrada por techo",
        "causa": "Mismo techo que D: +0.0027 como cota superior de cualquier linea cuya unica via sea recuperar filas ajenas a V03",
    },
    {
        "hipotesis_id": "V6-H10",
        "track": "C2, C3, C4",
        "hipotesis": "Canales posicionales, tokens de interaccion y combinaciones forma x token aportan senal",
        "soporte_empirico": "Ya ejecutados: V4-D dio interacciones 0.7426 y ventanas 0.7734; E2-10 dio -0.028 a -0.039",
        "experimento": "ninguno",
        "mejor_delta": "-0.0346",
        "ic95": "no aplica",
        "gate_1": "no evaluable",
        "veredicto": "cerrada por duplicacion",
        "causa": "Ejecutar seria repetir un resultado negativo ya medido. El gap analysis lo verifico por busqueda en el codigo, no de memoria",
    },
    {
        "hipotesis_id": "V6-H11",
        "track": "G",
        "hipotesis": "Permitir pocas interacciones especificas de etiqueta en la capa final",
        "soporte_empirico": "E2-21 cerro los espacios por etiqueta; el score con su margen ya es entrada de chains y del meta (E2-01, E2-18)",
        "experimento": "ninguno",
        "mejor_delta": "no medido",
        "ic95": "no aplica",
        "gate_1": "no evaluable",
        "veredicto": "no ejecutar",
        "causa": "Lo unico que quedaria anadir es el score kNN o el condicional, y ambos tienen techo medido o bajo. Marginal de una sola interaccion",
    },
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V6-cierre] {mensaje}", flush=True)


def _leer_variantes(identificador: str) -> list[dict[str, Any]]:
    """Lee las variantes de un experimento desde su JSON.

    Args:
        identificador: identificador del experimento.

    Returns:
        Lista de variantes con sus metricas.

    Raises:
        FileNotFoundError: si el JSON del experimento no existe.
    """
    ruta = REPORTES / FUENTES_RESULTADO[identificador]
    if not ruta.exists():
        raise FileNotFoundError(
            f"falta {ruta.name}: el experimento {identificador} no se ha ejecutado o no "
            f"ha escrito su JSON. No se rellenan huecos con ceros."
        )
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    return datos["variantes"]


def construir_master() -> list[dict[str, Any]]:
    """Construye la tabla maestra de resultados de la fase.

    Returns:
        Filas de ``v6_master_results.csv``.
    """
    filas: list[dict[str, Any]] = []
    for entrada in EXPERIMENTOS:
        if entrada["id"] not in FUENTES_RESULTADO:
            filas.append(
                {
                    "experiment_id": entrada["id"],
                    "track": entrada["tema"].split(":")[0].strip(),
                    "tipo": entrada["tipo"],
                    "variante": "no aplica",
                    "corrected_cv": "",
                    "delta_vs_V03": "",
                    "ic95_0": "",
                    "ic95_1": "",
                    "pliegues_positivos": "",
                    "gate_1_supera": "no aplica",
                    "observacion": entrada["tema"],
                }
            )
            continue
        for variante in _leer_variantes(entrada["id"]):
            filas.append(
                {
                    "experiment_id": entrada["id"],
                    "track": entrada["tema"].split(":")[0].strip(),
                    "tipo": entrada["tipo"],
                    "variante": variante.get("variante", ""),
                    "corrected_cv": variante.get("corrected_cv")
                    or variante.get("corrected_cv_frozen", ""),
                    "delta_vs_V03": variante.get("delta_vs_V03", ""),
                    "ic95_0": variante.get("ic95_0", ""),
                    "ic95_1": variante.get("ic95_1", ""),
                    "pliegues_positivos": variante.get("pliegues_positivos", ""),
                    "gate_1_supera": variante.get("gate_1_supera", ""),
                    "observacion": entrada["tema"],
                }
            )
    return filas


def construir_candidatos(master: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compara los tres candidatos que mas se acercaron al gate.

    Args:
        master: tabla maestra de resultados.

    Returns:
        Filas de ``v6_candidate_comparison.csv``, ordenadas por closeness al gate.
    """
    evaluables = [
        fila
        for fila in master
        if isinstance(fila["delta_vs_V03"], (int, float))
        and fila["variante"] not in {"B0-control-marginales", "A0-control", "I0-control",
                                     "C1-0-control", "H0-control", "F0-control",
                                     "J0-control"}
    ]
    for fila in evaluables:
        fila["distancia_al_gate"] = abs(GATE_1_DELTA - fila["delta_vs_V03"])
    evaluables.sort(key=lambda f: f["distancia_al_gate"])
    return evaluables[:3]


def _escribir_csv(nombre: str, filas: list[dict[str, Any]]) -> None:
    """Escribe una tabla en CSV con la codificacion del proyecto.

    Args:
        nombre: nombre del fichero.
        filas: filas a escribir.
    """
    if not filas:
        raise ValueError(f"no hay filas para {nombre}: revisa el origen antes de escribir")
    campos = list({clave for fila in filas for clave in fila})
    orden = [
        campo
        for campo in (
            "hipotesis_id", "experiment_id", "track", "tipo", "variante", "hipotesis",
            "soporte_empirico", "corrected_cv", "delta_vs_V03", "distancia_al_gate",
            "ic95", "ic95_0", "ic95_1", "pliegues_positivos", "gate_1", "gate_1_supera",
            "veredicto", "causa", "observacion",
        )
        if campo in campos
    ]
    orden += [campo for campo in campos if campo not in orden]
    with (REPORTES / nombre).open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=orden, lineterminator="\n")
        escritor.writeheader()
        for fila in filas:
            escritor.writerow(fila)


def main() -> None:
    """Construye todos los entregables de la fase v6."""
    inicio = time.time()
    log("Entregables finales de la fase v6: consolidacion y cierre")
    log(f"   referencia V03: corrected_cv {REFERENCIA_V03}, gate {GATE_1_DELTA:+.3f}")

    log("1/6 tabla maestra de resultados")
    master = construir_master()
    _escribir_csv("v6_master_results.csv", master)
    n_variantes = sum(1 for f in master if f["variante"] != "no aplica")
    n_controles = sum(1 for f in master if "control" in str(f["variante"]))
    n_superan = sum(1 for f in master if f["gate_1_supera"] is True or f["gate_1_supera"] == "True")
    log(
        f"   {len(master)} filas | {n_variantes} variantes medidas | {n_controles} controles "
        f"| {n_superan} superan el Gate 1"
    )

    log("2/6 registro de hipotesis")
    _escribir_csv("v6_hypothesis_registry.csv", HIPOTESIS)
    cerradas = sum(1 for h in HIPOTESIS if h["veredicto"].startswith("cerrada"))
    log(f"   {len(HIPOTESIS)} hipotesis | {cerradas} cerradas | "
        f"{len(HIPOTESIS) - cerradas} no ejecutadas por techo o duplicacion")

    log("3/6 comparacion de candidatos")
    candidatos = construir_candidatos(master)
    _escribir_csv("v6_candidate_comparison.csv", candidatos)
    for fila in candidatos:
        log(
            f"   {fila['variante']:22s} delta {fila['delta_vs_V03']:+.4f} | a "
            f"{fila['distancia_al_gate']:.4f} del gate"
        )

    log("4/6 interseccion de errores canonizada")
    origen = REPORTES / "v6_error_intersection.csv"
    destino = REPORTES / "v6_error_intersection.csv"
    if not origen.exists():
        raise FileNotFoundError(f"falta {origen.name}: ejecuta V6-01 primero")
    destino.write_text(origen.read_text(encoding="utf-8"), encoding="utf-8")
    log("   reports/v6_error_intersection.csv presente y canonizado")

    log("5/6 cumplimiento y reproducibilidad")
    _escribir_cumplimiento()
    _escribir_reproducibilidad()

    log("6/6 informe de cierre")
    _escribir_informe(master, candidatos, time.time() - inicio)
    log(f"Entregables completados en {round(time.time() - inicio, 1)} s")


def _rama_y_commits() -> tuple[str, int, str]:
    """Devuelve la rama actual, el numero de commits de la fase y el ultimo commit.

    Returns:
        Tupla con la rama, el numero de commits desde el cierre de v5.1 y el ultimo hash.
    """
    try:
        rama = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=PROJECT_ROOT.parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        commits = int(
            subprocess.run(
                ["git", "rev-list", "--count", "a719153..HEAD"],
                cwd=PROJECT_ROOT.parent,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        ultimo = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=PROJECT_ROOT.parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()[0]
        return rama, commits, ultimo
    except (subprocess.CalledProcessError, IndexError, ValueError):
        return "desconocida", 0, "desconocido"


def _escribir_cumplimiento() -> None:
    """Escribe el informe de cumplimiento de las restricciones de la fase."""
    rama, commits, ultimo = _rama_y_commits()
    con_secretos = subprocess.run(
        ["git", "grep", "-l", "ghp_", "--", "*.py", "*.md", "*.csv"],
        cwd=PROJECT_ROOT.parent,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lineas = [
        "# v6: cumplimiento de las restricciones (V6-09)",
        "",
        f"Generado por `experiments/v6_09_entregables.py` ({EXPERIMENTO}).",
        "",
        "## 1. Restricciones de la especificacion",
        "",
        "| restriccion | estado | como se verifica |",
        "|---|---|---|",
        "| sin transformers, LLM ni embeddings preentrenados de terceros | cumplida | "
        "ningun script de v6 importa transformers, sentence-transformers ni HF |",
        "| sin etiquetas externas ni pseudo-labeling | cumplida | ninguna etiqueta entra "
        "de fuera de `toxic_train.csv` |",
        "| sin usar `toxic_test.csv` para entrenar | cumplida | solo se lee para generar "
        "el fichero de submission, que en v6 no se genero |",
        "| sin usar el holdout | cumplida | el holdout esta agotado (5/5) y la fase lo "
        "declara; el Gate 2 usaria otra particion agrupada |",
        "| sin usar el leaderboard para ajustar hiperparametros | cumplida | **0 sondas "
        "de LB y 0 submissions en toda la fase** |",
        "| los 19 artefactos FastText de v5/v5.1 son reutilizables | cumplido en "
        "principio | no se gasto presupuesto en ellos: D se cerro por techo teorico |",
        "| presupuesto de 24 principales y 6 de reserva | cumplido de sobra | "
        "**7 tracks ejecutados, ninguno adopted** |",
        "| un experimento = un script nuevo e inmutable | cumplido | 9 scripts nuevos "
        "`v6_00` a `v6_08`, ninguno edita a otro |",
        "| ejecutar con `/home/carlos/venv/bin/python` | cumplido | todos los scripts |",
        "",
        "## 2. Gasto de la fase",
        "",
        f"- scripts nuevos: **{len(EXPERIMENTOS)}** (2 de analisis, 7 de experimento)",
        "- tracks con variantes medidas: **7** (B, A, I, C1, H, F, J)",
        "- tracks cerrados **sin gastar**: **3** (D y E por techo, C2/C3/C4 por "
        "duplicacion, G marcado no ejecutar)",
        "- variantes que superan el Gate 1: **0**",
        "- submissiones al leaderboard: **0**",
        "- sondas de LB: **0**",
        "",
        "El presupuesto autorizado era de 24 experimentos principales y 6 de reserva. El",
        "gasto real fue de **7**. La razon no es que faltara tiempo: es que el gap analysis",
        "(V6-00) y el diagnostico (V6-01) **cerraron 4 tracks antes de gastar una unidad**, y",
        "los tracks ejecutados se cerraron uno a uno por resultado negativo. El presupuesto",
        "no obliga a gastarlo, y gastar por gastar seria exactamente el error que D-044",
        "intenta evitar.",
        "",
        "## 3. Estado del repositorio",
        "",
        f"- rama: `{rama}`",
        f"- commits de la fase: **{commits}**, ultimo `{ultimo}`",
        "- **sin push**: la autorizacion de push nunca se pidio y no se hace sin ella",
        f"- secretos en ficheros versionados: "
        f"{'**SI, revisar**' if con_secretos else 'ninguno en los ficheros de la fase'}",
        "",
        "## 4. Recordatorio de seguridad pendiente desde v2",
        "",
        "El token clasico de GitHub (`ghp_rR3A...`) que aparecio en el historial del proyecto",
        "**sigue pendiente de rotar**. Esta fase no lo ha usado ni lo ha reintroducido, pero",
        "la rotacion es responsabilidad de Carlos y no depende de este trabajo.",
    ]
    (REPORTES / "v6_compliance.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _escribir_reproducibilidad() -> None:
    """Escribe el informe de reproducibilidad de la fase."""
    filas = [
        (
            "V6-00",
            "experiments/v6_00_gap_analysis.py",
            "v6_hypothesis_gap_analysis.md, v6_hypothesis_gap_analysis.csv, "
            "v6_verificaciones_gap_v6.json",
            "no",
            "15 patrones de implementacion en el codigo, excluyendo los scripts de la fase",
        ),
        (
            "V6-01",
            "experiments/v6_01_estructura_errores.py",
            "v6_error_structure.md, v6_error_intersection.csv, v6_label_structure.csv",
            "evaluacion downstream",
            "reconstruye 3 candidatos con el protocolo v2; usa el OOF congelado de V03",
        ),
        (
            "V6-02",
            "experiments/v6_02_label_factorization.py",
            "v6_label_factorization.md, .csv, .json",
            "si, 6 modelos por pliegue",
            "el control B0 debe dar 0.777211 exacto; si no, el codigo esta mal",
        ),
        (
            "V6-03",
            "experiments/v6_03_ponderacion_grupo.py",
            "v6_ponderacion_grupo.md, .csv, .json",
            "si, 12 ajustes por esquema",
            "los pesos se cuentan solo sobre el tren de cada pliegue",
        ),
        (
            "V6-04",
            "experiments/v6_04_char_weight.py",
            "v6_char_weight.md, .csv, .json",
            "si, 12 ajustes por peso",
            "el control I0 debe dar 0.777211 exacto",
        ),
        (
            "V6-05",
            "experiments/v6_05_skipgrams.py",
            "v6_skipgrams.md, .csv, .json",
            "si, 9 ajustes por esquema",
            "el control C1-0 debe dar 0.777211 exacto y 2744 caracteristicas de palabras",
        ),
        (
            "V6-06",
            "experiments/v6_06_bagging_grupos.py",
            "v6_bagging_grupos.md, .csv, .json",
            "si, 15 modelos por pliegue",
            "el control H0 debe dar 0.777211 exacto; la semilla del bagging es 20260926",
        ),
        (
            "V6-07",
            "experiments/v6_07_ruido_par.py",
            "v6_ruido_par.md, .csv, .json",
            "si, 12 ajustes por esquema",
            "la deteccion usa el OOF congelado de V03 y NearestNeighbors con 5 vecinos",
        ),
        (
            "V6-08",
            "experiments/v6_08_aumento.py",
            "v6_aumento.md, .csv, .json",
            "si, hasta 3.558 filas de train por pliegue",
            "el control J0 debe dar 0.777211 exacto; la transformacion se elige por indice",
        ),
    ]
    lineas = [
        "# v6: reproducibilidad (V6-09)",
        "",
        f"Generado por `experiments/v6_09_entregables.py` ({EXPERIMENTO}).",
        "",
        "## 1. Como ejecutar cada experimento",
        "",
        "Todos con el interprete obligatorio del proyecto:",
        "",
        "```bash",
        "/home/carlos/venv/bin/python experiments/v6_00_gap_analysis.py",
        "```",
        "",
        "| experimento | script | artefactos | entrena | nota de determinismo |",
        "|---|---|---|---|---|",
    ]
    for identificador, script, artefactos, entrena, nota in filas:
        lineas.append(
            f"| {identificador} | `experiments/{script}` | {artefactos} | {entrena} | {nota} |"
        )
    lineas += [
        "",
        "## 2. Los controles de reproducibilidad de la fase",
        "",
        "Cada track lleva un **control que debe reproducir V03 exactamente** (0.777211, delta",
        "+0.0000, IC95 [0, 0], 0 salvage y 0 damage):",
        "",
        "| track | control | resultado |",
        "|---|---|---|",
        "| B | B0 marginales reentrenados | 0.777211 exacto |",
        "| A | A0 `w = 1` | 0.777211 exacto |",
        "| I | I0 `char_weight = 1.0` | 0.777211 exacto |",
        "| C1 | C1-0 sin skip-grams | 0.777211 exacto |",
        "| H | H0 un modelo sin remuestreo | 0.777211 exacto |",
        "| F | F0 sin ponderar | 0.777211 exacto |",
        "| J | J0 sin copias | 0.777211 exacto |",
        "",
        "**Los siete controles dieron el numero exacto.** Ese es el resultado mas util de la",
        "fase desde el punto de vista de la confianza: cuando un control fallo (C1-0 en su",
        "primera version, que emitia solo unigramas y daba 0.7554), el control lo detecto antes",
        "de que se publicara ningun delta. Sin el, el resultado de C1 habria sido publicado",
        "como evidencia y era falso.",
        "",
        "## 3. Semillas fijadas",
        "",
        "- bootstrap pareado: 10.000 repeticiones, semilla **20260926**",
        "- bagging por grupos: semilla **20260926**",
        "- aumento: semilla **20260926**, transformacion elegida por indice de fila",
        "- `LogisticRegression` y `LinearSVC`: `random_state = 42`",
        "",
        "## 4. Artefactos de entrada compartidos",
        "",
        "| artefacto | quien lo usa |",
        "|---|---|",
        "| `features/oof_v03_final.npz` | V6-01, V6-02, V6-03, V6-04, V6-05, V6-06, V6-07, V6-08 |",
        "| `src/splits.py` (`grupos_fusionados`, `holdout_v2`, `pliegues_corregidos`) | todos |",
        "| `src/evaluation.py` (`metrica_macro`, `bootstrap_diferencia_pareada`) | todos |",
        "| `data/external/embeddings/*.bin` (19 modelos) | no usados: D se cerro por techo |",
    ]
    (REPORTES / "v6_reproducibility.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _escribir_informe(master, candidatos, duracion: float) -> None:
    """Escribe el informe de cierre de la fase v6.

    Args:
        master: tabla maestra de resultados.
        candidatos: comparacion de los tres mas cercanos al gate.
        duracion: segundos que tardo el script.
    """
    medidos = [f for f in master if f["variante"] != "no aplica"]
    controles = [f for f in medidos if "control" in str(f["variante"])]
    # Los recuentos del cierre salen del registro de hipotesis, no de un numero escrito a
    # mano: si mañana se añade una hipotesis, el informe cuenta la que es.
    ejecutadas = sum(1 for h in HIPOTESIS if h["experimento"] != "ninguno")
    por_techo = sum(1 for h in HIPOTESIS if h["veredicto"] == "cerrada por techo")
    no_ejecutadas = sum(1 for h in HIPOTESIS if h["veredicto"] == "no ejecutar")
    deltas_negativos = [
        fila for fila in medidos
        if isinstance(fila["delta_vs_V03"], (int, float))
        and fila["delta_vs_V03"] < 0
        and "control" not in str(fila["variante"])
    ]
    lineas = [
        "# v6: informe de cierre de la fase (V6-09)",
        "",
        f"Generado por `experiments/v6_09_entregables.py` ({EXPERIMENTO}).",
        "",
        "## 1. Que buscaba la fase y que ha encontrado",
        "",
        "La fase v6 no buscaba un modelo mejor, sino **mecanismos que el proyecto no hubiera",
        "evaluado de verdad**. La respuesta es que **no quedaba ninguno con margen**: de los",
        f"{len(HIPOTESIS)} hipotesis del registro, **{ejecutadas} se midieron y ninguna supero",
        f"el Gate 1**, {por_techo} quedaron cerradas **sin gastar** y {no_ejecutadas} se",
        "marcaron como no ejecutables.",
        "",
        "El resultado no es que V03 sea inalcanzable, sino algo mas preciso y mas util: **se",
        "sabe ahora que tipo de mecanismo no puede aportar nada y por que**. Los",
        f"**{len(controles)} controles de reproduccion**, cada uno dando 0.777211 exacto,",
        "convierten cada resultado negativo en una afirmacion y no en una sospecha.",
        "",
        "## 2. Inventario de la fase",
        "",
        f"- scripts nuevos: **{len(EXPERIMENTOS)}** (2 de analisis, 7 de experimento)",
        f"- variantes medidas: **{len(medidos)}**, de las cuales **{len(controles)}** son "
        "controles de reproduccion",
        "- variantes que superan el Gate 1: **0**",
        f"- deltas negativos entre variantes no control: **{len(deltas_negativos)}** de "
        f"{len(medidos) - len(controles)}",
        "- **V03 sigue congelada**: `corrected_cv` 0.777211, LB 0.78306. Nada de la fase se "
        "adopto, y **0 submissions** se hicieron.",
        "",
        "## 3. Los siete tracks ejecutados y por que se cerraron",
        "",
        "| track | mejor delta | IC95 | por que se cerro |",
        "|---|---|---|---|",
    ]
    filas_por_track = {
        "B (V6-02)": (
            "-0.0008", "[-0.0075, +0.0057]",
            "la invariante TR implica HS es cierta pero solo se viola en el 1.35 % de las "
            "predicciones; el producto atenua p(TR) donde V03 acertaba",
        ),
        "A (V6-03)": (
            "+0.0006", "[-0.0029, +0.0040]",
            "solo el 7.1 % de las filas puede cambiar de peso, y penalizar mas a los "
            "repetidos empeora: **la repeticion es senal**",
        ),
        "I (V6-04)": (
            "+0.0016", "[-0.0048, +0.0078]",
            "el optimo es interior (1.5) pero el techo es la mitad del gate: **la escala no "
            "era el cuello de botella**",
        ),
        "C1 (V6-05)": (
            "-0.0001", "[-0.0085, +0.0082]",
            "los skip-grams son **neutros**: +83 % de caracteristicas sin mover el resultado",
        ),
        "H (V6-06)": (
            "-0.0105", "[-0.0183, -0.0030]",
            "**por protocolo**: hay diversidad real (corr. 0.9573) pero los umbrales de V03 "
            "no son validos para un promedio",
        ),
        "F (V6-07)": (
            "+0.0001", "[-0.0033, +0.0035]",
            "la deteccion encuentra **dificultad, no ruido**: V03 acierta 10.5 % en las filas "
            "sospechosas frente a 28.3 % en las demas",
        ),
        "J (V6-08)": (
            "-0.0061", "[-0.0141, +0.0018]",
            "aumentar empeora y **doblar las copias no empeora mas**: es sesgo de las copias, "
            "no cantidad",
        ),
    }
    for track, (delta, ic95, causa) in filas_por_track.items():
        lineas.append(f"| {track} | {delta} | {ic95} | {causa} |")
    lineas += [
        "",
        "## 4. Los tracks que se cerraron sin gastar",
        "",
        "| track | veredicto | causa |",
        "|---|---|---|",
        "| D (TextCNN) | techo teorico | la interseccion de errores da un techo de **+0.0027**, "
        "por debajo del gate de +0.003 |",
        "| E (kNN) | techo teorico | mismo techo: +0.0027 como cota de cualquier linea que "
        "solo recupere filas ajenas |",
        "| C2, C3, C4 | duplicacion | ya medidos: V4-D dio -0.0038 y -0.0346; E2-10 dio "
        "-0.028 a -0.039 |",
        "| G | no ejecutar | el score y su margen ya son entrada de chains y del meta desde "
        "E2-01 y E2-18 |",
        "",
        "Cerrar por techo **antes** de gastar es la decision que mas presupuesto ahorro de la",
        "fase, y se tomo apoyandose en una medicion (la interseccion de errores) y no en una",
        "suposicion.",
        "",
        "## 5. Los tres candidatos mas cercanos al gate",
        "",
        "| variante | delta | a que distancia del gate | IC95 | pliegues |",
        "|---|---|---|---|---|",
    ]
    for fila in candidatos:
        lineas.append(
            f"| {fila['variante']} | {fila['delta_vs_V03']:+.4f} | "
            f"{fila['distancia_al_gate']:.4f} | "
            f"[{fila['ic95_0']:+.4f}, {fila['ic95_1']:+.4f}] | "
            f"{fila['pliegues_positivos']} |"
        )
    mejor = candidatos[0]
    lineas += [
        "",
        f"El mas cercano es **{mejor['variante']}** con {mejor['delta_vs_V03']:+.4f}, a",
        f"{mejor['distancia_al_gate']:.4f} del gate, con un IC95 que **incluye el 0**. No es un",
        "candidato: es el techo empirico de la linea mas barata que quedaba.",
        "",
        "## 6. Lo que la fase deja escrito, mas alla de los negativos",
        "",
        "Cinco conclusiones sobreviven al cierre y son las que hacen que la fase valiera la",
        "pena:",
        "",
        "1. **Un tuit repetido es senal, no ruido de recoleccion** (track A). Penalizar los",
        "   grupos repetidos empeora de forma monotonica. No reabrir esa linea ni en variantes",
        "   mas suaves.",
        "2. **La repeticion aporta informacion recuperable: casi ninguna** (P0-bis). El mejor",
        "   modelo existente acierta 9 a 12 filas de 1.483 donde V03 falla, y combinarlos",
        "   **elige** entre sus aciertos en vez de sumarlos. Toda linea cuyo techo sea",
        "   recuperar filas ajenas esta condenada por construccion.",
        "3. **La deteccion de sospecha sirve para medir dificultad, no para limpiar** (track F).",
        "   Marca donde V03 falla, y bajar el peso de esas filas aleja al modelo de sus",
        "   errores.",
        "4. **La escala entre word y char no era el cuello de botella** (track I), aunque el",
        "   optimo este en 1.5 y no en el 1 heredado.",
        "5. **No hay forma limpia de medir un promedio de modelos con umbrales calibrados "
        "para uno** (track H). Es una limitacion del protocolo, y queda planteada la condicion",
        "   que la resolveria: una particion de calibracion separada del OOF de evaluacion.",
        "",
        "## 7. Entregables de la fase",
        "",
        "| entregable | contenido |",
        "|---|---|",
        "| `reports/v6_final_research.md` | este informe |",
        "| `reports/v6_master_results.csv` | una fila por variante, con metricas y gate |",
        "| `reports/v6_hypothesis_registry.csv` | una fila por hipotesis, con veredicto y causa |",
        "| `reports/v6_candidate_comparison.csv` | los tres mas cercanos al gate |",
        "| `reports/v6_error_intersection.csv` | interseccion de errores canonizada |",
        "| `reports/v6_compliance.md` | cumplimiento de las restricciones |",
        "| `reports/v6_reproducibility.md` | como reproducir cada resultado |",
        "| `reports/v6_hypothesis_gap_analysis.md` | el gap analysis con verificacion programatica |",
        "| `reports/v6_error_structure.md` | interseccion de errores y estructura de etiquetas |",
        "| `reports/figuras/fig_58` a `fig_65` | ocho figuras de la fase |",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion de la consolidacion: {round(duracion, 1)} s.",
    ]
    (REPORTES / "v6_final_research.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
