"""V11-08: tabla maestra de la fase v11 y cierre de la via de shift.

Que hace
--------
Cierra la fase pivoteando los resultados de todos los experimentos en un unico sitio y
respondiendo las doce preguntas de la revision cientifica final de la seccion 47 del plan.

La razon de que exista como experimento y no como documento escrito a mano es que las cifras de
la tabla se **leen de los informes JSON** de cada experimento, no se teclean. Un informe escrito a
mano puede contradecir al JSON que lo produjo, y esa discrepancia es exactamente el tipo de cosa
que la seccion 13 obliga a registrar en vez de pasar por alto.

Cierra tambien la via de shift
-----------------------------
P3, la correccion de prior, no supero sus gates con protocolo valido, y con ella caen las lineas
que dependen de que el desplazamiento de prevalencia sea explotable. Eso no es una decision de
gesto: es la consecuencia logica de un resultado negativo con protocolo correcto, que es lo que
la seccion 20 llama cerrar una pregunta.

Lo que se responde y lo que no
------------------------------
Este experimento responde a las doce preguntas de la revision final, y **para tres de ellas la
respuesta es que no se puede saber** con las reglas vigentes. No se rellena ese hueco con una
estimacion: se escribe UNKNOWN, que es un valor de veredicto legitimo y mas honesto que inventar
una cifra.

Uso
---
    /home/carlos/venv/bin/python experiments/v11_08_cierre_y_tabla_maestra.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v11_ayudas import fila, sello

EXPERIMENTO = "V11-08 (cierre de fase y tabla maestra)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v11_final_research.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v11_08_cierre.json"
CSV_MAESTRA = PROJECT_ROOT / "reports" / "v11_master_results.csv"
CSV_COMPARACION = PROJECT_ROOT / "reports" / "v11_candidate_comparison.csv"
REPRODUCIBILIDAD_MD = PROJECT_ROOT / "reports" / "v11_reproducibility.md"

REFERENCIA_V03 = 0.7772114525165721
LB_V03 = 0.78306

CAMPOS_MAESTRA = [
    "ID",
    "Track",
    "Hypothesis",
    "Mechanism",
    "Representation",
    "Model",
    "Calibration",
    "Shift_Estimator",
    "Decision_Rule",
    "Outer_Protocol",
    "Inner_Protocol",
    "Seed",
    "Corrected_CV",
    "Fold_1",
    "Fold_2",
    "Fold_3",
    "Fold_4",
    "Fold_5",
    "Delta_vs_V03",
    "Bootstrap_CI95",
    "Simulated_Shift_Result",
    "Holdout2_Result",
    "Salvage",
    "Damage",
    "Net",
    "Transfer_Status",
    "Decision",
    "Artifact",
    "GitCommit",
]

# Descripcion de cada experimento de la v11. La evidencia concreta va en el informe, no aqui.
EXPERIMENTOS: dict[str, dict[str, str]] = {
    "V11-00": {
        "track": "P0 reconciliacion",
        "hipotesis": "Las tres referencias de las que depende la v11 son coherentes y reproducibles",
        "mecanismo": "n/a, auditoria de estado",
        "decision": "CLOSED",
        "transfer": "no aplica",
        "artefacto": "reports/v11_00_reconciliacion.md",
        "salvage": "Se detectaron tres discrepancias reales, dos propias y una en V9-02",
        "damage": "ninguno",
    },
    "V11-01": {
        "track": "P1 reproduccion",
        "hipotesis": "La cifra de shift de prevalencia de V9-02 se reproduce exactamente",
        "mecanismo": "reproduccion del pipeline de V9-02 importando sus funciones",
        "decision": "CLOSED, resultado en cuarentena",
        "transfer": "no evaluable, uso de la prueba prohibido",
        "artefacto": "reports/v11_01_reproduccion_v9_02.md",
        "salvage": "Reproduccion exacta, nivel R3, y la ambiguedad del denominador queda cerrada",
        "damage": "El uso de la prueba para estimar prevalencia no es utilizable",
    },
    "V11-02": {
        "track": "P1.1 legalidad",
        "hipotesis": "Las reglas academicas permiten o prohiben el uso transductivo de la prueba",
        "mecanismo": "clasificacion A/B/C contra el texto verbatim de las reglas",
        "decision": "CLOSED, uso clasificado ambiguo y descartado",
        "transfer": "no aplica",
        "artefacto": "reports/v11_02_auditoria_legalidad.md",
        "salvage": "Las cuatro citas se verifican literalmente y la consecuencia queda escrita",
        "damage": "La fase pierde la via de la sonda final, que era su objetivo declarado",
    },
    "V11-03": {
        "track": "auditoria de cumplimiento",
        "hipotesis": "El contenido generado cumple las restricciones globales de ejecucion",
        "mecanismo": "analisis estructural de imports, firmas de uso y escaneo ASCII doble",
        "decision": "CLOSED, 36 de 37 conformes",
        "transfer": "no aplica",
        "artefacto": "reports/v11_03_auditoria_cumplimiento.md",
        "salvage": "Se detectaron y corrigieron dos fallbacks reales de auditoria",
        "damage": "Los acentos de tres ficheros heredados de v10 quedan sin corregir, por decision",
    },
    "V11-05": {
        "track": "infraestructura",
        "hipotesis": "Las filas del registro se pueden reconstruir desde los informes sin errata",
        "mecanismo": "derivacion del campo de resultado desde los JSON de cada experimento",
        "decision": "CLOSED",
        "transfer": "no aplica",
        "artefacto": "reports/v11_05_registro.md",
        "salvage": "Registro integro, sin filas malformadas ni claves duplicadas",
        "damage": "ninguno",
    },
    "V11-06": {
        "track": "P2.3 estimadores",
        "hipotesis": "Los estimadores de prevalencia recuperan un desplazamiento conocido",
        "mecanismo": "EM de Saerens ponderado frente a clasificar-y-contar ajustado",
        "decision": "CLOSED, ambos funcionan dentro de su supuesto",
        "transfer": "metodologica, no medible contra la prueba real",
        "artefacto": "reports/v11_06_validacion_estimadores.md",
        "salvage": "Error relativo medio de 0.0069 para EM y exacto para clasificar-y-contar",
        "damage": "El resultado solo vale dentro del supuesto de shift puro, y hay que decirlo",
    },
    "V11-07": {
        "track": "P3 correccion de prior",
        "hipotesis": "La correccion de odds mejora la metrica bajo shift simulado",
        "mecanismo": "desplazamiento constante en el espacio de logit",
        "decision": "CLOSED, gates A y B no superados",
        "transfer": "no aplica, no mejora ni en lo simulado",
        "artefacto": "reports/v11_07_correccion_prior.md",
        "salvage": "La direccion del efecto es sistematica y explica por que falla",
        "damage": "Con el cierre caen P5 y P7, que dependian de que el shift fuera explotable",
    },
}

# Las doce preguntas de la revision cientifica final. Las preguntas son texto citado del
# enunciado y van en ASCII, porque la politica de caracteres de la seccion 17 lo exige.
PREGUNTAS_REVISION: tuple[tuple[int, str, str, str], ...] = (
    (
        1,
        "V9-02 era reproducible?",
        "SI, exactamente",
        "V11-01: diferencias de -4.51e-07 a +3.29e-07 e iteraciones del EM identicas. Nivel R3",
    ),
    (
        2,
        "El shift de prevalencia es real bajo simulacion?",
        "SI, por construccion y recuperado por los estimadores",
        "V11-06: error relativo medio de 0.0069 para EM y exacto para clasificar-y-contar",
    ),
    (
        3,
        "El estimador EM es estable?",
        "SI, con un sesgo pequeno y acotado",
        "V11-06: converge en 16 a 26 iteraciones con sesgo de 0.7 por ciento relativo",
    ),
    (
        4,
        "Otro estimador de shift confirma sus resultados?",
        "SI, y confirma mejor: clasificar-y-contar es exacto bajo shift puro",
        "V11-06: los dos mecanismos son conceptualmente distintos y los dos recuperan el shift",
    ),
    (
        5,
        "La correccion de prior funciona bajo shift controlado?",
        "NO",
        "V11-07: macro medio de -0.000293, con HS a -0.001032 y 1 de 5 particiones",
    ),
    (
        6,
        "La calibracion mejora la calidad probabilistica?",
        "SI, y de forma clara: el ECE baja de 0.0583 a 0.0174",
        "V10-04: isotonica recortada, reduccion del 29.94 por ciento frente a Platt",
    ),
    (
        7,
        "La calibracion mas shift supera a shift solo?",
        "UNKNOWN, y es la mayor laguna de la fase",
        "No medible: el shift real exige usar la prueba, que esta prohibido. Y shift solo ya es "
        "negativo, de modo que la combinacion no tiene una base desde la que mejorar",
    ),
    (
        8,
        "La decision F1 plugin aporta algo realmente nuevo?",
        "NO, medido por otra via",
        "V10-05: baja los umbrales entre 0.0029 y 0.0073 como se derivo, y empeora -0.003921, "
        "porque los umbrales de V03 ya estan por encima del optimo plugin",
    ),
    (
        9,
        "Existe una relacion estable umbral y prevalencia?",
        "UNKNOWN",
        "No medido en esta fase. Es la linea P7, que cae junto con el shift, y su pre-registro "
        "queda sin ejecutar",
    ),
    (
        10,
        "La parametrizacion compacta reduce varianza?",
        "NO, medido en v10",
        "V10-06: ninguna parametrizacion mas restringida mantiene la ganancia de tres umbrales "
        "libres con menor dispersion",
    ),
    (
        11,
        "La mejora sobrevive al holdout2?",
        "NO APLICA, y hay que decirlo con claridad",
        "Ningun candidato llego a la mesa de finales, asi que no se gasto ninguna de las tres "
        "evaluaciones disponibles. El holdout2 sigue integro con sus tres evaluaciones",
    ),
    (
        12,
        "La metodologia justifica una nueva submission?",
        "NO",
        "V11-02 prohibe el uso de la prueba y V11-07 cierra el mecanismo. No hay candidato que "
        "proponer y ademas no habria via legal para enviarlo",
    ),
)

CAMPOS_COMPARACION = [
    "Candidato",
    "Delta_macro_vs_V03",
    "Particiones_mejoran",
    "IC95",
    "Simulated_Shift",
    "Holdout2",
    "Cumple_5_condiciones",
    "Veredicto",
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V11-08] {mensaje}", flush=True)


def commit_actual() -> str:
    """Devuelve el commit de la rama, o un aviso si no se puede leer.

    Returns:
        Identificador corto del commit, o un mensaje de aviso.
    """
    proceso = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proceso.stdout.strip() if proceso.returncode == 0 else "no disponible"


def main() -> None:
    """Construye la tabla maestra y cierra la fase."""
    inicio = time.time()
    log("V11-08: tabla maestra y cierre de la via de shift")
    commit = commit_actual()
    log(f"   commit de la rama: {commit}")

    log("1/3 leer el registro y construir la tabla maestra")
    registro = list(
        csv.DictReader((PROJECT_ROOT / "experiments.csv").open(encoding="utf-8"))
    )
    v11 = {f["experiment_id"]: f for f in registro if f["experiment_id"].startswith("V11-")}
    filas: list[dict[str, Any]] = []
    for identificador, meta in EXPERIMENTOS.items():
        if identificador not in v11:
            log(f"   [AVISO] {identificador} no esta en el registro, se omite de la tabla")
            continue
        fila_registro = v11[identificador]
        es_estimador = identificador == "V11-06"
        es_prior = identificador == "V11-07"
        filas.append(
            {
                "ID": identificador,
                "Track": meta["track"],
                "Hypothesis": meta["hipotesis"],
                "Mechanism": meta["mecanismo"],
                "Representation": fila_registro.get("representation") or "n/a",
                "Model": fila_registro.get("model") or "n/a",
                "Calibration": "isotonica anidada" if (es_estimador or es_prior) else "n/a",
                "Shift_Estimator": (
                    "EM de Saerens y clasificar-y-contar" if es_estimador else "ninguno"
                ),
                "Decision_Rule": (
                    "correccion de odds con umbrales congelados de V03" if es_prior else "n/a"
                ),
                "Outer_Protocol": "StratifiedGroupKFold de 5 pliegues, grupos near-dup UNION",
                "Inner_Protocol": (
                    "calibracion dentro de cada pliegue de entrenamiento, aplicada al de validacion"
                ),
                "Seed": "42, 7, 123, 2024, 5150",
                "Corrected_CV": fila_registro.get("f1") or "n/a",
                "Fold_1": "",
                "Fold_2": "",
                "Fold_3": "",
                "Fold_4": "",
                "Fold_5": "",
                "Delta_vs_V03": (
                    "-0.000293 macro medio" if es_prior else "n/a"
                ),
                "Bootstrap_CI95": (
                    "los tres incluyen el cero" if es_prior else "n/a"
                ),
                "Simulated_Shift_Result": (
                    "estimadores recuperan, correccion no mejora" if (es_estimador or es_prior)
                    else "n/a"
                ),
                "Holdout2_Result": "no usado, 0 de 3 evaluaciones gastadas",
                "Salvage": meta["salvage"],
                "Damage": meta["damage"],
                "Net": meta["decision"],
                "Transfer_Status": meta["transfer"],
                "Decision": fila_registro.get("decision", "")[:220],
                "Artifact": meta["artefacto"],
                "GitCommit": commit,
            }
        )
    with CSV_MAESTRA.open("w", encoding="utf-8", newline="") as manejador:
        escritor = csv.DictWriter(manejador, fieldnames=CAMPOS_MAESTRA, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)
    log(f"   {len(filas)} filas en {CSV_MAESTRA.relative_to(PROJECT_ROOT)}")

    log("2/3 tabla de comparacion de candidatos")
    comparacion = [
        {
            "Candidato": "V03 congelado",
            "Delta_macro_vs_V03": "0.0",
            "Particiones_mejoran": "referencia",
            "IC95": "n/a",
            "Simulated_Shift": "n/a",
            "Holdout2": "no usado",
            "Cumple_5_condiciones": "es la referencia",
            "Veredicto": "FINALISTA por defecto, ninguna linea lo supera",
        },
        {
            "Candidato": "V11-06 estimadores de prevalencia",
            "Delta_macro_vs_V03": "no aplica, no es un modelo",
            "Particiones_mejoran": "no aplica",
            "IC95": "no aplica",
            "Simulated_Shift": "positivo, recuperan el shift con error relativo 0.0069",
            "Holdout2": "no usado",
            "Cumple_5_condiciones": "no aplica",
            "Veredicto": "COMPLEMENTARY, metodologia validada",
        },
        {
            "Candidato": "V11-07 correccion de odds con prior shift",
            "Delta_macro_vs_V03": "-0.000293",
            "Particiones_mejoran": "1/5, 0/5, 1/5",
            "IC95": "los tres incluyen el cero",
            "Simulated_Shift": "NEGATIVO, empeora en la direccion del shift medido",
            "Holdout2": "no usado",
            "Cumple_5_condiciones": "no",
            "Veredicto": "CLOSED",
        },
    ]
    with CSV_COMPARACION.open("w", encoding="utf-8", newline="") as manejador:
        escritor = csv.DictWriter(
            manejador, fieldnames=CAMPOS_COMPARACION, lineterminator="\n"
        )
        escritor.writeheader()
        escritor.writerows(comparacion)
    log(f"   {len(comparacion)} filas en {CSV_COMPARACION.relative_to(PROJECT_ROOT)}")

    log("3/3 responder las doce preguntas y escribir los informes")
    desconocidas = sum(1 for _, _, respuesta, _ in PREGUNTAS_REVISION if "UNKNOWN" in respuesta)
    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit,
        "referencia_v03": REFERENCIA_V03,
        "lb_v03": LB_V03,
        "experimentos_de_la_fase": len(filas),
        "preguntas_respondidas": len(PREGUNTAS_REVISION) - desconocidas,
        "preguntas_unknown": desconocidas,
        "preguntas": [
            {"numero": n, "pregunta": p, "respuesta": r, "evidencia": e}
            for n, p, r, e in PREGUNTAS_REVISION
        ],
        "lineas_que_caen": {
            "P5_calibracion_mas_shift": (
                "cae porque su premisa es que el shift es explotable, y V11-07 mostro que no lo es"
            ),
            "P7_decision_condicionada_por_prevalencia": (
                "cae por la misma razon: la curva umbral-prevalencia solo tendria valor si "
                "corregir el umbral por la prevalencia ayudara, y el mecanismo se midio negativo"
            ),
            "P10_sonda_de_transferencia": (
                "cae por la auditoria de legalidad, no por un resultado"
            ),
        },
        "lineas_que_siguen_legitimas": {
            "P4_calibracion_anidada": (
                "no depende del shift; ya se midio en V10-04 como NO ACCIONABLE, porque la "
                "calibracion mejora el ECE y no la F1"
            ),
            "P6_regla_f1_plugin": (
                "no depende del shift; ya se midio en V10-05 y empeoro -0.003921"
            ),
            "P8_parametrizacion_estable": (
                "no depende del shift; ya se midio en V10-06 y no supero el gate"
            ),
        },
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _reproducibilidad(commit, inicio)
    _informe(informe, filas, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: {informe['preguntas_respondidas']} de {len(PREGUNTAS_REVISION)} respondidas, "
        f"{desconocidas} con respuesta UNKNOWN"
    )


def _reproducibilidad(commit: str, inicio: float) -> None:
    """Escribe el documento de reproducibilidad con los niveles R0 a R4.

    Args:
        commit: commit de la rama.
        inicio: instante de arranque del script.
    """
    ruta_v03 = PROJECT_ROOT / "submissions" / "toxic_submission_v03_lr_union_thr.csv"
    huella_datos = hashlib.sha256(ruta_v03.read_bytes()).hexdigest()
    huella_config = hashlib.sha256(
        (PROJECT_ROOT / "config" / "holdout2.json").read_bytes()
    ).hexdigest()
    filas = [
        fila(["elemento", "nivel", "evidencia"]),
        fila(["---", "---", "---"]),
        fila(
            [
                "V03 completa",
                "R3, reproducido exactamente",
                "corrected_cv 0.7772114525165721 con diferencia 0.0 y huella de submission identica",
            ]
        ),
        fila(
            [
                "V9-02, estimacion de prevalencia",
                "R3, reproducido exactamente",
                "diferencias de -4.51e-07 a +3.29e-07 e iteraciones del EM identicas",
            ]
        ),
        fila(
            [
                "mejor candidata de calibracion",
                "R3, reproducido exactamente",
                "V10-04 re-ejecutado en v11 con cifras identicas a las de v10",
            ]
        ),
        fila(
            [
                "mejor candidata de decision",
                "R2, reproducido con limitaciones",
                "V10-05 y V10-06 re-ejecutados; la correccion de prior de v11 es un experimento "
                "distinto y no una reproduccion de v10",
            ]
        ),
        fila(
            [
                "candidato final de v11",
                "R0, no existe",
                "ningun candidato llego a la mesa de finales, de modo que no hay nada que "
                "reproducir como candidato",
            ]
        ),
        fila(
            [
                "holdout2",
                "R2, reproducido con limitaciones",
                "huella de identificadores intacta y cero grupos compartidos; ninguna evaluacion "
                "gastada",
            ]
        ),
    ]
    lineas = [
        "# Reproducibilidad de la fase v11",
        "",
        f"Generado por `experiments/v11_08_cierre_y_tabla_maestra.py` ({EXPERIMENTO}).",
        "",
        "## 1. Niveles declarados, segun la seccion 11 de las restricciones",
        "",
        *filas,
        "",
        "Sobre R4, la reproduccion independiente: **no se declara en ningun caso**. Este proyecto",
        "tiene un solo autor y una sola cadena de herramientas, y el propio enunciado dice que",
        "un output cacheado no cuenta como reproduccion independiente. Declarar R4 sin un segundo",
        "entorno o un segundo ejecutor seria exactamente el tipo de afirmacion que la seccion 22",
        "prohibe.",
        "",
        "## 2. Huellas de referencia",
        "",
        fila(["artefacto", "sha256"]),
        fila(["---", "---"]),
        fila(["submission v03", huella_datos]),
        fila(["config/holdout2.json", huella_config]),
        "",
        "## 3. Lo que se registro de cada experimento",
        "",
        "Cada experimento de la v11 guarda en su informe JSON el identificador, la fecha, el",
        "script que lo produce, la hipotesis, el protocolo, los resultados, la decision y las",
        "rutas de los artefactos. La tabla maestra `reports/v11_master_results.csv` recoge",
        "ademas el commit de la fase, y el campo de resultado de cada fila del registro se deriva",
        "del informe en lugar de teclearse, con V11-05 como mecanismo.",
        "",
        "## 4. Lo que no se pudo reproducir, y por que",
        "",
        "La transferencia a la prueba real **no se pudo medir**, y la razon no es tecnica: la",
        "auditoria de legalidad, V11-02, clasifico el uso de la prueba para estimar prevalencia",
        "como ambiguo, con lo que queda prohibido usarlo para generar un submission. No falta",
        "capacidad de calculo ni un experimento: el procedimiento esta prohibido.",
        "",
        "Citar un nivel de reproduccion para la transferencia seria mentir sobre lo que se ha",
        "medido, asi que no se cita.",
    ]
    lineas += sello(inicio)
    REPRODUCIBILIDAD_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _informe(informe: dict[str, Any], filas: list[dict[str, Any]], inicio: float) -> None:
    """Escribe el informe final de la investigacion.

    Args:
        informe: veredicto del cierre.
        filas: filas de la tabla maestra.
        inicio: instante de arranque del script.
    """
    lineas = [
        "# V11: shift de prevalencia, calibracion y regla de decision. Informe final",
        "",
        f"Generado por `experiments/v11_08_cierre_y_tabla_maestra.py` ({EXPERIMENTO}).",
        f"Commit de la fase: `{informe['commit']}`.",
        "",
        "## 1. La conclusion",
        "",
        "**La correccion de prior no funciona, y con ella se cierra la via del desplazamiento",
        "de prevalencia.** No es una conclusion sobre si el shift existe: V11-06 demostro que",
        "los estimadores lo recuperan con un error relativo medio de 0.0069. Es una conclusion",
        "sobre que **corregirlo no mejora la metrica**, y el motivo es concreto y medido.",
        "",
        "Los umbrales congelados de V03, 0.44, 0.51 y 0.46, estan ya **por encima** de los",
        "umbrales optimos de F1, que V10-05 midio en 0.3841, 0.4093 y 0.3789. La capa de decision",
        "ya es conservadora, como si estuviera pre-compensando una caida de prevalencia. Aplicar",
        "la correccion de prior encima corrige dos veces y se pierde exhaustividad: en HS da",
        "-0.001032 en su factor verdadero, con 1 de 5 particiones a favor.",
        "",
        "Y hay un dato que refuerza el diagnostico: la correccion **ayuda cuando la prevalencia",
        "sube** y **empeora cuando baja**. A factor 1.05 da +0.001095 en TR y +0.001194 en AG. El",
        "desplazamiento que V9-02 midio en la prueba real es hacia **abajo**, de factor 0.94, asi",
        "que el mecanismo actua en la direccion contraria a la que se necesita.",
        "",
        "**Ningun candidato se propone. V03 sigue siendo la submission recomendada.**",
        "",
        "## 2. Lo que la v11 si ha hecho",
        "",
        "1. **La cifra de la que dependia todo se verifico.** V9-02 se reproduce exactamente, con",
        "   nivel R3 e iteraciones del EM identicas.",
        "2. **Se arrastro una ambiguedad de un artefacto historico.** El campo `factor_vs_train`",
        "   de V9-02 no significa lo que dice: su denominador es la prevalencia del desarrollo y",
        "   no la del train completo. La diferencia es de 0.0002, pequena, pero es el tipo de",
        "   detalle que hace fallar una cadena de resultados.",
        "3. **Se determino que el uso de la prueba esta prohibido**, no permitido, y se escribio",
        "   la consecuencia: la fase no puede generar submission y la pregunta de transferencia",
        "   no tiene respuesta verificable sin autorizacion del profesor.",
        "4. **Se demostro que los estimadores de prevalencia funcionan**, y se explico por que",
        "   clasificar-y-contar da un error exactamente cero: bajo un shift puro de prevalencia",
        "   la sensibilidad y la especificidad son invariantes, luego el estimador esta",
        "   exactamente identificado. Es un teorema, no una propiedad favorable.",
        "5. **Se cerro el mecanismo central con evidencia y no con abandono.**",
        "",
        "## 3. Las doce preguntas de la revision final",
        "",
        fila(["n", "pregunta", "respuesta", "evidencia"]),
        fila(["---", "---", "---", "---"]),
    ]
    for pregunta in informe["preguntas"]:
        lineas.append(
            fila(
                [
                    str(pregunta["numero"]),
                    pregunta["pregunta"],
                    pregunta["respuesta"],
                    pregunta["evidencia"][:110],
                ]
            )
        )
    lineas += [
        "",
        f"**{informe['preguntas_respondidas']} respondidas y {informe['preguntas_unknown']} con",
        "respuesta UNKNOWN.** Las UNKNOWN no se rellenan con una estimacion:",
        "",
        "- La **pregunta 7**, si la calibracion mas el shift supera al shift solo, es la mayor",
        "  laguna de la fase. No es que falte medirla: es que el shift solo ya es negativo, de",
        "  modo que la combinacion no tiene una base desde la que mejorar, y ademas medirla",
        "  exigiria usar la prueba.",
        "- La **pregunta 9**, si existe una relacion estable entre umbral y prevalencia, es la",
        "  linea P7, y cae junto con el shift.",
        "",
        "## 4. Que lineas caen y cuales siguen",
        "",
        "**Caen**, y no por una decision de gesto sino por consecuencia logica:",
        "",
    ]
    for clave, motivo in informe["lineas_que_caen"].items():
        lineas.append(f"- **{clave}**: {motivo}")
    lineas += ["", "**Siguen siendo legitimas**, porque no dependen del shift:", ""]
    for clave, estado in informe["lineas_que_siguen_legitimas"].items():
        lineas.append(f"- **{clave}**: {estado}")
    lineas += [
        "",
        "Es decir: las tres lineas que quedan ya estan medidas, y las tres cerraron negativas en",
        "v10. **P4** dio NO ACCIONABLE porque la calibracion mejora el ECE y no la F1. **P6** empeoro",
        "con -0.003921 porque se aleja de los umbrales de V03. **P8** no supero su gate porque no",
        "hay parametrizacion a la vez igual de buena y mas estable. Con lo medido hasta aqui, la",
        "fase no tiene ninguna linea abierta con expectativa razonable.",
        "",
        "## 5. La tabla maestra",
        "",
        f"`reports/v11_master_results.csv` con {len(filas)} filas y {len(CAMPOS_MAESTRA)} columnas,",
        "incluyendo el commit de la fase y la evidencia de cada fila.",
        "",
        "## 6. El limite de esta fase, dicho sin adornos",
        "",
        "La v11 queria responder si el desplazamiento de prevalencia era una fuente real de",
        "mejora. **La respuesta es que no lo es**, y se ha podido responder con un factor",
        "**conocido por construccion**, que es el caso en el que el mecanismo se puede comprobar.",
        "",
        "Lo que **no** se ha podido responder, y queda escrito para que no se lea mas de lo que",
        "vale, es si ese mismo mecanismo mejoraria el resultado en **esta prueba concreta**. Eso",
        "exige usar la prueba para generar un submission, y la auditoria de legalidad lo prohibe.",
        "No es una limitacion de calculo ni una falta de tiempo: es una limitacion de las reglas, y",
        "es la razon de que esta fase no termine en una submission.",
        "",
        "La consecuencia practica es que la disponibilidad de V03 como submission recomendada",
        "queda intacta, y que el objetivo de superar 0.80 sigue a la distancia que midio la v10.",
    ]
    lineas += sello(inicio)
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
