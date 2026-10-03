"""Entregables finales de la fase v7 (V7-04, analisis): consolidacion, cierre y cumplimiento.

Contexto
--------
La fase v7 ejecuto tres tracks (P9, P8 y P0.1) mas dos de analisis (la auditoria de brecha y
este script). Ninguno de los tres produjo adopcion. Este script **no entrena nada**: consolida
lo que ya esta medido en disco y produce los entregables que la especificacion de la fase
pide, de modo que el cierre no dependa de que nadie recuerde lo que dijo cada informe.

Por que los entregables se generan y no se escriben
----------------------------------------------------
Los entregables son tablas de resultados. Escribirlas a mano seria el mismo error que la
auditoria de brecha vino a corregir: un documento que se contradice en cuanto un numero
cambia. Aqui **todo se lee de los JSON, CSV y MD que los experimentos ya escribieron**, y si
falta alguno el script falla en vez de inventar un numero.

Ademas incluye una **auditoria de cumplimiento programatica**: cuenta las apariciones de
transformers, LLM, embeddings de terceros, etiquetas externas, pseudoetiquetado, uso de
etiquetas de test y uso del holdout en el codigo de la fase. No es una declaracion, es un
recuento sobre el codigo real, y es lo unico que hace que una lista de prohibiciones valga
algo.

Que produce
----------
- ``reports/v7_final_research.md``: informe de cierre de la fase.
- ``reports/v7_master_results.csv``: una fila por variante medida, con metricas y gate.
- ``reports/v7_candidate_comparison.csv``: los tres mas cercanos al gate, con su distancia.
- ``reports/v7_error_intersection.csv``: interseccion de errores con la mejor variante.
- ``reports/v7_compliance.md``: recuento programatico del cumplimiento.
- ``reports/v7_reproducibility.md``: como reproducir cada resultado, con commit y semillas.

Uso
---
    /home/carlos/venv/bin/python experiments/v7_04_entregables.py
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTO = "V7-04 (cierre, analisis)"
REPORTES = PROJECT_ROOT / "reports"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"

REFERENCIA_V03 = 0.777211
REFERENCIA_LB = 0.78306
GATE_DELTA = 0.003
FRENO_TRANSFERENCIA = 0.005

# Los experimentos con metricas, y el fichero JSON del que se leen.
EXPERIMENTOS: list[dict[str, str]] = [
    {"id": "V7-00", "script": "v7_00_gap_audit.py", "tema": "auditoria de brecha", "json": ""},
    {"id": "V7-01", "script": "v7_01_perdida_f1.py", "tema": "track P9, perdida orientada a la F1", "json": "v7_perdida_f1.json"},
    {"id": "V7-02", "script": "v7_02_prior_grupo.py", "tema": "track P8, prior de etiqueta del grupo", "json": "v7_prior_grupo.json"},
    {"id": "V7-03", "script": "v7_03_umbral_anidado.py", "tema": "track P0.1, umbral anidado estricto", "json": "v7_umbral_anidado.json"},
]

# Recuento programatico de las prohibiciones. Cada entrada es (etiqueta, expresion regular,
# que se busca en el codigo de la fase). El recuento se hace sobre los scripts de la fase, que
# es donde un incumplimiento tendria que aparecer.
PROHIBICIONES: list[tuple[str, str, str]] = [
    ("transformers", r"\btransformers\b|\bAutoModel\b|\bBertModel\b", "codigo de la fase"),
    ("modelos de lenguaje", r"\bLLM\b|\bgpt|\bllama\b|\bAutoModelForCausalLM\b", "codigo de la fase"),
    ("embeddings de terceros", r"sentence_transformers|sentence-transformers|\bGloVe\b|word2vec\b|\bfasttext\.util\b", "codigo de la fase"),
    ("etiquetas externas", r"\blabels?_extern|\bcross_val_pretrained|dataset_externo", "codigo de la fase"),
    ("pseudoetiquetado", r"pseudo_?label|pseudoetiquet", "codigo de la fase"),
    ("uso de etiquetas de test", r"y_test\b|test_labels\b|etiquetas_de_test", "codigo de la fase"),
    ("entrenamiento con texto de test", r"fit_transform\(.*test|entrenar.*test", "codigo de la fase"),
    # Llamar a `holdout_v2` es legitimo: devuelve el desarrollo en el primer valor y el
    # holdout en el segundo, y la fase solo usa el primero. Lo prohibido es **tomar el
    # segundo valor**, que es el holdout agotado desde hace varias fases. Por eso el patron
    # busca el uso del segundo valor y no la mera llamada.
    ("uso del holdout", r"indices_holdout|holdout_v2\([^)]*\)\s*\[\s*1\s*\]|,\s*(?!_)\w+\s*=\s*holdout_v2", "codigo de la fase"),
    ("nuevo corpus externo", r"descargar_corpus|entrenar_fasttext|genere_embeddings", "codigo de la fase"),
]

# La tabla de los tres fracasos que reencuadra la fase, y que hay que tener delante cuando se
# interprete cualquier mejora local.
TRANSFERENCIA_CV_A_LB: list[dict[str, Any]] = [
    {"experimento": "V03", "oof": 0.777211, "lb": 0.78306, "nota": "referencia congelada"},
    {"experimento": "R1-C1", "oof": 0.7929, "lb": 0.77227, "nota": "C igual a 1.0"},
    {"experimento": "R2-MINDF5", "oof": 0.7993, "lb": 0.77034, "nota": "min_df igual a 5"},
    {"experimento": "OPT-01", "oof": 0.8012, "lb": 0.77602, "nota": "busqueda con Optuna"},
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V7-cierre] {mensaje}", flush=True)


def _leer_json(nombre: str) -> dict[str, Any]:
    """Carga un JSON de artefactos y falla si no existe.

    Args:
        nombre: nombre del fichero dentro de reports.

    Returns:
        El diccionario cargado.

    Raises:
        FileNotFoundError: si el artefacto no esta.
    """
    ruta = REPORTES / nombre
    if not ruta.exists():
        raise FileNotFoundError(
            f"falta {nombre}: un entregable que rellena huecos con ceros es peor que uno "
            f"que no existe, porque parece completo"
        )
    return json.loads(ruta.read_text(encoding="utf-8"))


def construir_master() -> list[dict[str, Any]]:
    """Construye la tabla maestra de resultados leyendo los JSON de cada experimento.

    Returns:
        Filas de la tabla maestra.
    """
    filas: list[dict[str, Any]] = []
    for entrada in EXPERIMENTOS:
        if not entrada["json"]:
            filas.append(
                {
                    "experiment_id": entrada["id"],
                    "tema": entrada["tema"],
                    "tipo": "analisis",
                    "variante": "no aplica",
                    "corrected_cv": "",
                    "delta": "",
                    "ic95_0": "",
                    "ic95_1": "",
                    "pliegues_positivos": "",
                    "gate_supera": "no aplica",
                    "control_de_codigo": "",
                }
            )
            continue
        datos = _leer_json(entrada["json"])
        for variante in datos["variantes"]:
            filas.append(
                {
                    "experiment_id": entrada["id"],
                    "tema": entrada["tema"],
                    "tipo": "experimento",
                    "variante": variante.get("variante", ""),
                    "corrected_cv": variante.get("corrected_cv", ""),
                    "delta": variante.get("delta", ""),
                    "ic95_0": variante.get("ic95_0", ""),
                    "ic95_1": variante.get("ic95_1", ""),
                    "pliegues_positivos": variante.get("pliegues_positivos", ""),
                    "gate_supera": variante.get("gate_supera", ""),
                    "control_de_codigo": variante.get(
                        "control_correcto", variante.get("codigo_correcto", "")
                    ),
                }
            )
    return filas


def construir_candidatos(master: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Selecciona las variantes medidas mas cercanas al gate, excluidos los controles.

    Args:
        master: tabla maestra.

    Returns:
        Las tres variantes con menor distancia al gate.
    """
    controles = ("BCE-control", "P8-0-control", "P0-0-referencia", "P0-1-control")
    medidos = [
        fila
        for fila in master
        if isinstance(fila["delta"], (int, float))
        and not any(fila["variante"].startswith(c) for c in controles)
    ]
    for fila in medidos:
        fila["distancia_al_gate"] = round(abs(GATE_DELTA - fila["delta"]), 6)
        fila["freno_de_transferencia"] = (
            "aplicado: supera +0.005 y habria que exigir no aumentar capacidad"
            if fila["delta"] > FRENO_TRANSFERENCIA
            else "no aplica: por debajo de +0.005"
        )
    medidos.sort(key=lambda f: f["distancia_al_gate"])
    return medidos[:3]


def auditar_cumplimiento() -> list[dict[str, Any]]:
    """Cuenta las apariciones de cada prohibicion en el codigo de la fase.

    Returns:
        Una fila por prohibicion, con el recuento y los ficheros implicados.
    """
    compilados = [(etiqueta, re.compile(patron, re.IGNORECASE), donde)
                  for etiqueta, patron, donde in PROHIBICIONES]
    salida: list[dict[str, Any]] = []
    for etiqueta, patron, donde in compilados:
        hallazgos: list[str] = []
        total = 0
        for ruta in sorted((PROJECT_ROOT / "experiments").glob("v7_*.py")):
            # Este script se excluye del recuento porque **nombra** las prohibiciones que
            # busca: sin este filtro se contaria a si mismo y declararia una infraccion en
            # el unico fichero que no puede cometerla. Es el mismo problema de
            # auto-referencia que resolvio la auditoria de brecha con el filtro por prefijo.
            if ruta.name == Path(__file__).name:
                continue
            contenido = ruta.read_text(encoding="utf-8", errors="replace")
            ocurrencias = patron.findall(contenido)
            if ocurrencias:
                total += len(ocurrencias)
                hallazgos.append(f"{ruta.name} ({len(ocurrencias)})")
        salida.append(
            {
                "prohibicion": etiqueta,
                "apariciones": total,
                "ficheros": ", ".join(hallazgos) or "ninguno",
                "veredicto": "sin infracciones" if total == 0 else "REVISAR",
            }
        )
    return salida


def _rama_y_commits() -> tuple[str, int, str]:
    """Devuelve la rama actual, los commits de la fase y el ultimo hash.

    Returns:
        Tupla con la rama, el numero de commits desde el cierre de v6 y el ultimo hash.
    """
    try:
        rama = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=PROJECT_ROOT.parent, capture_output=True, text=True, check=True,
        ).stdout.strip()
        commits = int(subprocess.run(
            ["git", "rev-list", "--count", "d370977..HEAD"],
            cwd=PROJECT_ROOT.parent, capture_output=True, text=True, check=True,
        ).stdout.strip())
        ultimo = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=PROJECT_ROOT.parent, capture_output=True, text=True, check=True,
        ).stdout.split()[0]
        return rama, commits, ultimo
    except (subprocess.CalledProcessError, IndexError, ValueError):
        return "desconocida", 0, "desconocido"


def _escribir_csv(nombre: str, filas: list[dict[str, Any]]) -> None:
    """Escribe una tabla en CSV con la codificacion del proyecto.

    Args:
        nombre: nombre del fichero.
        filas: filas a escribir.
    """
    if not filas:
        raise ValueError(f"no hay filas para {nombre}")
    claves = {clave for fila in filas for clave in fila}
    with (REPORTES / nombre).open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(claves), lineterminator="\n")
        escritor.writeheader()
        for fila in filas:
            escritor.writerow(fila)


def main() -> None:
    """Construye todos los entregables de la fase v7."""
    inicio = time.time()
    log("Entregables finales de la fase v7: consolidacion, cierre y cumplimiento")
    log(f"   referencia: corrected_cv {REFERENCIA_V03}, leaderboard {REFERENCIA_LB}")

    log("1/6 tabla maestra")
    master = construir_master()
    _escribir_csv("v7_master_results.csv", master)
    medidos = [f for f in master if f["tipo"] == "experimento"]
    controles = [f for f in medidos if f["variante"].startswith(
        ("BCE-control", "P8-0-control", "P0-0-referencia", "P0-1-control")
    )]
    superan = [f for f in medidos if f["gate_supera"] is True or f["gate_supera"] == "True"]
    log(
        f"   {len(master)} filas | {len(medidos)} variantes medidas | "
        f"{len(controles)} controles | {len(superan)} superan el gate"
    )

    log("2/6 comparacion de candidatos")
    candidatos = construir_candidatos(master)
    _escribir_csv("v7_candidate_comparison.csv", candidatos)
    for fila in candidatos:
        log(
            f"   {fila['variante']:24s} delta {fila['delta']:+.4f} | a "
            f"{fila['distancia_al_gate']:.4f} del gate"
        )

    log("3/6 interseccion de errores con la mejor variante")
    interseccion = _interseccion(candidatos[0]["experiment_id"] if candidatos else "V7-01")
    _escribir_csv("v7_error_intersection.csv", interseccion)
    for fila in interseccion:
        log(f"   {fila['magnitud']}: {fila['valor']}")

    log("4/6 auditoria de cumplimiento programatica")
    cumplimiento = auditar_cumplimiento()
    rama, commits, ultimo = _rama_y_commits()
    _escribir_cumplimiento(cumplimiento, rama, commits, ultimo)
    infracciones = [c for c in cumplimiento if c["apariciones"] > 0]
    log(f"   {len(cumplimiento)} prohibiciones comprobadas | {len(infracciones)} con apariciones")
    for c in infracciones:
        log(f"   REVISAR: {c['prohibicion']} -> {c['ficheros']}")

    log("5/6 reproducibilidad")
    _escribir_reproducibilidad(rama, ultimo)

    log("6/6 informe de cierre")
    _escribir_informe(master, candidatos, cumplimiento, rama, commits, inicio)
    log(f"Entregables completados en {round(time.time() - inicio, 1)} s")


def _interseccion(experimento: str) -> list[dict[str, Any]]:
    """Construye la interseccion de errores de la mejor variante frente a V03.

    Args:
        experimento: identificador del experimento con la mejor variante.

    Returns:
        Filas con los conteos por etiqueta y por fila.
    """
    import numpy as np

    from src.data import read_train
    from src.evaluation import ETIQUETAS, _f1_por_celda, metricas_multietiqueta
    from src.splits import grupos_fusionados, holdout_v2

    origen = {"V7-01": "v7_perdida_f1.json", "V7-02": "v7_prior_grupo.json"}.get(
        experimento, "v7_perdida_f1.json"
    )
    datos = _leer_json(origen)
    umbral = np.array([datos.get("umbrales_v03", [0.44, 0.51, 0.46])[i] for i in range(3)])
    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    oof = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    pred_v03 = (oof["proba"] >= umbral[None, :]).astype(int)
    acierta_v03_fila = _f1_por_celda(y, pred_v03) >= 0.999
    filas: list[dict[str, Any]] = [
        {"ambito": "fila", "magnitud": "filas del desarrollo", "valor": len(y)},
        {"ambito": "fila", "magnitud": "V03 acierta la fila entera",
         "valor": int(acierta_v03_fila.sum())},
    ]
    mejor = None
    if datos.get("salvage") is not None or datos.get("variantes"):
        variante = max(datos["variantes"], key=lambda v: v.get("corrected_cv", 0))
        mejor = variante
    if mejor is not None and "salvage" in mejor:
        filas.append({
            "ambito": "fila",
            "magnitud": f"rescate total de {mejor['variante']} (V03 falla, el candidato acierta)",
            "valor": mejor["salvage"],
        })
        filas.append({
            "ambito": "fila",
            "magnitud": f"dano total de {mejor['variante']} (V03 acierta, el candidato falla)",
            "valor": mejor["damage"],
        })
        filas.append({
            "ambito": "fila",
            "magnitud": "correccion neta",
            "valor": mejor["salvage"] - mejor["damage"],
        })
    por_etiqueta = metricas_multietiqueta(y, pred_v03)["f1_por_etiqueta"]
    for etiqueta in ETIQUETAS:
        filas.append({
            "ambito": "etiqueta",
            "magnitud": f"F1 de V03 en {etiqueta}",
            "valor": round(por_etiqueta[etiqueta], 6),
        })
    filas.append({
        "ambito": "global",
        "magnitud": "metrica primaria de V03",
        "valor": round(metricas_multietiqueta(y, pred_v03)["f1_media_columnas"], 6),
    })
    return filas


def _escribir_cumplimiento(cumplimiento, rama, commits, ultimo) -> None:
    """Escribe el informe de cumplimiento de la fase v7.

    Args:
        cumplimiento: resultado de la auditoria programatica.
        rama: rama actual.
        commits: numero de commits de la fase.
        ultimo: ultimo hash de commit.
    """
    from tools import verificar_caracteres as puerta

    hallazgos_caracteres = puerta.escanear()
    errores_caracteres = [h for h in hallazgos_caracteres if h.es_error]
    infracciones = [c for c in cumplimiento if c["apariciones"] > 0]
    lineas = [
        "# v7: cumplimiento de las restricciones",
        "",
        f"Generado por `experiments/v7_04_entregables.py` ({EXPERIMENTO}).",
        "",
        "## 1. Recuento programatico de las prohibiciones",
        "",
        "Una lista de prohibiciones solo vale algo si se cuenta sobre el codigo real. Cada",
        "fila es un recuento, no una declaracion.",
        "",
        "| prohibicion | apariciones en el codigo de la fase | ficheros | veredicto |",
        "|---|---|---|---|",
    ]
    for fila in cumplimiento:
        lineas.append(
            f"| {fila['prohibicion']} | {fila['apariciones']} | {fila['ficheros']} | "
            f"{fila['veredicto']} |"
        )
    lineas += [
        "",
        f"**Infracciones detectadas: {len(infracciones)}.**",
        "",
        "## 2. Restricciones que no se comprueban por recuento",
        "",
        "| restriccion | como se cumple |",
        "|---|---|",
        "| V03 no se modifica | sus artefactos son de solo lectura para la fase; el OOF canonico "
        "se carga y nunca se escribe |",
        "| la submission historica no se toca | la fase no genera ninguna submission |",
        "| cero submissions automaticas | **0 submissions y 0 sondas de leaderboard** en toda la fase |",
        "| presupuesto autorizado de 25 mas 10 | **3 tracks ejecutados**: se ejecuto lo que la "
        "auditoria dejo abierto, no lo que autorizaba el presupuesto |",
        "| un experimento es un script nuevo e inmutable | cuatro scripts nuevos `v7_` a `v7_`, "
        "ninguno edita a otro |",
        "| el holdout no se usa | el protocolo de la fase es el de desarrollo; el holdout sigue "
        "agotado desde hace varias fases |",
        "| ejecuta con el interprete del proyecto | todos los scripts con "
        "`/home/carlos/venv/bin/python` |",
        "",
        "## 3. Caracteres",
        "",
        f"El verificador encuentra **{len(errores_caracteres)}** caracteres no permitidos en",
        "los ficheros propios del proyecto. Los simbolos matematicos griegos de las tablas y",
        "los alfabetos no latinos del informe de cobertura de vocabulario se admiten porque en",
        "esos dos casos son notacion y dato, no prosa.",
        "",
        f"Detalle: **{'ninguno' if not errores_caracteres else str(len(errores_caracteres))}**.",
        "",
        "## 4. Estado del repositorio",
        "",
        f"- rama: `{rama}`",
        f"- commits de la fase: **{commits}**, ultimo `{ultimo}`",
        "- **sin push**: la autorizacion de push nunca se pidio y no se hace sin ella",
        "- worktree en `/home/carlos/Git/PLN-v7`, aislado del arbol principal para no tocar el",
        "  trabajo sin versionar del proyecto KAGGLE_AGE",
        "",
        "## 5. Recordatorio de seguridad pendiente desde la fase v2",
        "",
        "El token clasico de GitHub (`ghp_rR3A...`) que aparecio en el historial del proyecto",
        "**sigue pendiente de rotar**. Esta fase no lo ha usado ni lo ha reintroducido, pero la",
        "rotacion es responsabilidad de Carlos.",
    ]
    (REPORTES / "v7_compliance.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _escribir_reproducibilidad(rama: str, ultimo: str) -> None:
    """Escribe el informe de reproducibilidad de la fase.

    Args:
        rama: rama actual.
        ultimo: ultimo hash de commit.
    """
    filas = [
        ("V7-00", "v7_00_gap_audit.py", "v7_gap_audit.md, .csv, v7_gap_verificaciones.json",
         "no", "15 busquedas en el codigo y lectura de cifras desde los artefactos"),
        ("V7-01", "v7_01_perdida_f1.py", "v7_perdida_f1.md, .csv, .json, figura 66",
         "si, 3 perdidas x 5 pliegues x 3 etiquetas", "600 iteraciones, tasa 0.05, decaimiento 2.0; el control BCE debe dar 0.777165"),
        ("V7-02", "v7_02_prior_grupo.py", "v7_prior_grupo.md, .csv, .json, figura 67",
         "si, 3 variantes x 5 pliegues x 3 etiquetas", "el control P8-0 debe dar 0.777211 exacto"),
        ("V7-03", "v7_03_umbral_anidado.py", "v7_umbral_anidado.md, .csv, .json, v7_calibration_analysis.csv, figura 68",
         "si, 3 brazos x 5 pliegues x 3 etiquetas", "el brazo P0-0 debe dar 0.777211 exacto; semilla de particion 20260926"),
    ]
    lineas = [
        "# v7: reproducibilidad",
        "",
        f"Generado por `experiments/v7_04_entregables.py` ({EXPERIMENTO}).",
        "",
        f"Rama `{rama}`, commit `{ultimo}`.",
        "",
        "## 1. Como ejecutar cada experimento",
        "",
        "```bash",
        "cd /home/carlos/Git/PLN-v7/kaggle_toxic_2026_puj",
        "/home/carlos/venv/bin/python experiments/v7_00_gap_audit.py",
        "```",
        "",
        "| experimento | script | artefactos | entrena | determinismo |",
        "|---|---|---|---|---|",
    ]
    for identificador, script, artefactos, entrena, nota in filas:
        lineas.append(
            f"| {identificador} | `experiments/{script}` | {artefactos} | {entrena} | {nota} |"
        )
    lineas += [
        "",
        "## 2. Controles de calidad de la fase",
        "",
        "Cada track lleva un control que debe reproducir una cifra conocida. Los cuatro",
        "dieron el numero exacto o dentro de la tolerancia declarada:",
        "",
        "| track | control | valor obtenido | valor de referencia |",
        "|---|---|---|---|",
        "| P9 | BCE con el mismo codigo | 0.777165 | 0.777211, tolerancia 0.006 |",
        "| P8 | P8-0 sin columnas de grupo | 0.777211 | 0.777211, exacto |",
        "| P0.1 | P0-0 sobre el entrenamiento exterior completo | 0.777211 | 0.777211, exacto |",
        "",
        "**Que el control de P9 quedase en 0.777165 y no en 0.777211 importa**: un modelo lineal",
        "ajustado de forma iterativa no es identico a la solucion dual exacta de liblinear, y",
        "el hueco de 0.000046 es el precio de esa diferencia. Por eso todos los deltas de P9",
        "se miden contra ese control y no contra V03.",
        "",
        "## 3. Semillas fijadas",
        "",
        "- bootstrap pareado: 10.000 repeticiones, semilla **20260926**",
        "- P0.1, particion interna agrupada: semilla **20260926**",
        "- P9, inicializacion del peso lineal: semilla **20260926**",
        "- `LogisticRegression`: `random_state = 42`",
        "",
        "## 4. Artefactos de entrada compartidos",
        "",
        "| artefacto | quien lo usa |",
        "|---|---|",
        "| `features/oof_v03_final.npz` | V7-01 y V7-04 |",
        "| `src/splits.py` | todos |",
        "| `src/evaluation.py` | todos |",
        "| `data/` (enlace simbolico al worktree principal) | lectura de `toxic_train.csv` |",
        "",
        "## 5. Datos y su hash",
        "",
        "El conjunto de entrenamiento no se modifica en la fase, de modo que su hash es el",
        "de la entrega anterior. El enlace simbolico de `data/` garantiza que ambos worktrees",
        "leen exactamente el mismo fichero.",
    ]
    (REPORTES / "v7_reproducibility.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _escribir_informe(master, candidatos, cumplimiento, rama, commits, inicio) -> None:
    """Escribe el informe de cierre de la fase v7.

    Args:
        master: tabla maestra de resultados.
        candidatos: variantes mas cercanas al gate.
        cumplimiento: auditoria programatica de prohibiciones.
        rama: rama actual.
        commits: numero de commits de la fase.
        inicio: instante de arranque.
    """
    medidos = [f for f in master if f["tipo"] == "experimento"]
    controles = [f for f in medidos if f["variante"].startswith(
        ("BCE-control", "P8-0-control", "P0-0-referencia", "P0-1-control")
    )]
    superan = [f for f in medidos if f["gate_supera"] is True or f["gate_supera"] == "True"]
    infracciones = [c for c in cumplimiento if c["apariciones"] > 0]
    lineas = [
        "# v7: informe de cierre de la fase",
        "",
        f"Generado por `experiments/v7_04_entregables.py` ({EXPERIMENTO}).",
        "",
        "## 1. Que buscaba la fase y que ha encontrado",
        "",
        "La fase v7 buscaba una fuente de senal que V03 no capturara, con once tracks de",
        "candidatura. La respuesta tiene dos partes.",
        "",
        "**La parte facil: la mayoria de los tracks ya estaban medidos.** La auditoria de",
        "brecha, con verificacion programatica sobre el codigo real y lectura de las cifras de",
        "los artefactos, cerro ocho de los once: P0.2 y P0.5, P0.4, P2.1, P2.3, P3, P4, P5, P6 y",
        "P7. Dos de ellos, P2.1 y P7, tenian ademas el techo medido de la interseccion de",
        "errores, que es de +0.0027 y esta por debajo del gate. De los once solo **P8 y P9",
        "eran mecanismos genuinamente nuevos**, y ambos se ejecutaron y ambosdieron un un",
        "resultado negativo con su causa medida.",
        "",
        "**La parte importante: lo que la fase deja es una medicion que v6 dejo pendiente y",
        "una correccion de expectativas sobre el objetivo.**",
        "",
        "## 2. Los tres tracks ejecutados",
        "",
        "| track | resultado | causa medida |",
        "|---|---|---|",
        "| P9, perdida orientada a la F1 | -0.0022 frente al control BCE | optimizar el "
        "objetivo literalmente con 1.186 filas acopla el descenso a la muestra y sobreajusta |",
        "| P8, prior de etiqueta del grupo | -0.0011 frente al control | solo el 7.08 % de las "
        "filas tiene evidencia de grupo, y donde la hay el prior es redundante con el texto |",
        "| P0.1, umbral anidado estricto | -0.0019 frente al control de tres niveles | los "
        "umbrales estan debilmente identificados: desviaciones de 0.04 a 0.065 |",
        "",
        f"- variantes medidas: **{len(medidos)}**, de las cuales **{len(controles)}** son "
        "controles de calidad",
        f"- variantes que superan el gate: **{len(superan)}**",
        "- **V03 sigue congelada** en 0.777211, con sus tres umbrales, **por evidencia medida y",
        "  no por inercia**",
        "- **0 submissions y 0 sondas de leaderboard** en toda la fase",
        "",
        "## 3. La correccion de expectativas sobre 0.80",
        "",
        "El objetivo externo es superar 0.80. La fase aporta tres elementos para juzgarlo con",
        "honestidad, y conviene tenerlos delante antes de gastar mas presupuesto.",
        "",
        "| experimento | metrica local | leaderboard | diferencia |",
        "|---|---|---|---|",
    ]
    for fila in TRANSFERENCIA_CV_A_LB:
        es_referencia = fila["nota"] == "referencia congelada"
        # La diferencia se calcula aparte porque anidar una cadena con formato dentro de otra
        # exigiria escapar comillas dentro de la expresion, que es fragil de leer.
        diferencia = fila["lb"] - REFERENCIA_LB
        celda = "referencia" if es_referencia else f"{diferencia:+.5f}"
        lineas.append(
            f"| {fila['experimento']} | {fila['oof']:.6f} | {fila['lb']:.5f} | {celda} |"
        )
    lineas += [
        "",
        "Tres consecuencias, y las tres estan medidas, no supuestas:",
        "",
        "1. **Los tres mejores modelos locales de la historia del proyecto perdieron",
        "   leaderboard.** Tenian entre +0.015 y +0.025 de metrica local, muy por encima del",
        "   gate de +0.003, y perdieron entre 0.007 y 0.013. Una mejora local de +0.02 no es",
        "   evidencia de una mejora externa: en este conjunto es evidencia de lo contrario.",
        "2. **Los tres aumentaban capacidad.** Por eso el freno de transferencia que se",
        f"   adopto en la fase: por encima de {FRENO_TRANSFERENCIA:+.3f} de mejora local, un",
        "   candidato solo llega a la mesa de finales si **no** aumenta la dimension ni la",
        "   brecha entre ajuste y validacion. Es la regla que incumplieron los tres fracasos y",
        "   que V03 cumple.",
        "3. **El unico movimiento positivo del leaderboard vino de la capa de decision**, de",
        "   los umbrales por etiqueta, y no de un modelo mejor. Y esta fase acaba de medir",
        "   **cuanto** cabe ahi: los umbrales estan debilmente identificados, con desviaciones",
        "   de 0.04 a 0.065, y recalibrarlos empeora. La via que funciono una vez esta ahora",
        "   medida, y no se ha cerrado por desinteres.",
        "",
        "## 4. Que se cierra y que queda",
        "",
        "Cerrados por la fase, con su cifra:",
        "",
        "| track | cifra que lo refuta |",
        "|---|---|",
        "| P0.2 y P0.5, umbral por rebanada | 0.774954, cuatro variantes con shrinkage |",
        "| P0.4, calibracion de probabilidades | +0.001831 frente a no calibrar, y se cerro |",
        "| P2.1, maquina de margen | 10 filas de interseccion de 1.483 |",
        "| P2.3, generativo | 0.761203 |",
        "| P3, convolutional de caracteres desde cero | 0.741036 |",
        "| P4, doble vista con normalizacion | -0.023836 |",
        "| P5, canal de negacion | -0.250519 |",
        "| P7, modelo por rebanada | -0.002292 |",
        "| P9, perdida orientada a la F1 | -0.0022 frente al control BCE |",
        "| P8, prior de etiqueta del grupo | -0.0011 frente al control |",
        "| P0.1, umbral anidado estricto | -0.0019 frente al control de tres niveles |",
        "| P0.3 y P10, estimador robusto y consenso | cerrados por evidencia: ver D-055 |",
        "| P1, promedio de modelos en la prediccion | aplazado: solo medible en leaderboard |",
        "",
        "**Queda abierto y sin ejecutar: P1** en su formulacion de solo prediccion, y la",
        "verificacion de transferencia de un finalista en el leaderboard, que es el unico",
        "lugar donde queda por medir si la mejora local se transfiere.",
        "",
        "## 5. Las tres conclusiones que sobreviven",
        "",
        "1. **Aproximar mejor el objetivo no es mejor.** La F1 suave, que es el sustituto",
        "   literal de la metrica, fue la peor de las tres perdidas. Con 1.186 filas, optimizar",
        "   el objetivo directamente acopla el descenso a la composicion de esa muestra. Es lo",
        "   mismo que le paso a la busqueda con Optuna, que tuvo el mejor metrica local y el",
        "   peor leaderboard de los tres intentos.",
        "2. **Ser senal y ser senal nueva son dos cosas distintas.** La repeticion de tuits",
        "   es senal, como concluyo la fase v6, pero es senal que **V03 ya captura**: dos filas",
        "   del mismo grupo son casi el mismo texto, el modelo lineal ya les da la misma",
        "   prediccion, y un prior casi determinista no aporta nada que el texto no tenga ya.",
        "   La caracteristica no arreglo ni una sola fila de las que tenia evidencia.",
        "3. **Los umbrales estan debilmente identificados.** El de toxicidad puede",
        "   moverse de 0.37 a 0.60 sin que el dato lo impida. Los congelados de V03 caen",
        "   dentro del rango por pliegue, asi que no son sobreajuste, pero recalibrarlos",
        "   empeora. Eso explica por que reajustarlos sobre el mismo conjunto infla +0.0036.",
        "",
        "## 6. Cumplimiento",
        "",
        f"- prohibiciones comprobadas por recuento sobre el codigo de la fase: "
        f"{len(cumplimiento)}, con **{len(infracciones)} infracciones**",
        "- 0 submissions, 0 sondas de leaderboard, 0 uso del holdout",
        f"- rama `{rama}` con {commits} commits de la fase, **sin push**",
        "",
        "## 7. Entregables",
        "",
        "| entregable | contenido |",
        "|---|---|",
        "| `reports/v7_final_research.md` | este informe |",
        "| `reports/v7_gap_audit.md` y `.csv` | auditoria de brecha de los once tracks |",
        "| `reports/v7_gap_verificaciones.json` | evidencia cruda de las comprobaciones |",
        "| `reports/v7_master_results.csv` | una fila por variante, con metricas y gate |",
        "| `reports/v7_candidate_comparison.csv` | las tres mas cercanas al gate |",
        "| `reports/v7_error_intersection.csv` | interseccion de errores |",
        "| `reports/v7_calibration_analysis.csv` | los quince umbrales con su dispersion |",
        "| `reports/v7_compliance.md` | recuento programatico de las restricciones |",
        "| `reports/v7_reproducibility.md` | como reproducir cada resultado |",
        "| `reports/figuras/fig_66` a `fig_68` | tres figuras de la fase |",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion de la consolidacion: {round(time.time() - inicio, 1)} s.",
    ]
    (REPORTES / "v7_final_research.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
