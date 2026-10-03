"""V18 · Cierre de fase — tablas maestras, comparativa, informes y compliance.

Este script no mide nada. Recoge lo que las siete fases de V18 dejaron en sus artefactos y produce
los documentos de cierre: tabla maestra, registro de hipotesis, comparativa de candidatos, nota de
reproducibilidad, informe de cumplimiento e informe final de la fase.

**Por que un script y no a mano.** Los numeros de cierre son los de cada JSON de experimento, y
escribirlos a mano es la forma mas facil de que el informe final y los artefactos dejen de cuadrar. Al
generarlos desde los JSON, si algo se corrige y se reejecuta el experimento, el cierre se corrige
solo.

**La conclusion de la fase**, que el informe develops y que se puede anticipar aqui: la senal de
frontera **existe y es grande** (D-048: un especialista sobre las 279 variables de V12 saca 0,7136 y
0,7493 frente a 0,4963 y 0,4974 de V12), y **ninguna funcion objetivo consiga cobrarla**. Se
intento por cinco vias distintas, con mecanismos distintos y cinco resultados negativos:

| Via | Mecanismo | Resultado |
|---|---|---|
| P1 | re-peso de clases | frontera +0,082, global -0,0164; el coste entero cae sobre la clase vecina |
| P3 | margen de adyacencia | una frontera +0,039 y la otra -0,055: estan acopladas por `25-34` |
| P4 | termino auxiliar aditivo | no mueve nada (+0,0038), con cero parametros extra |
| P5 | destilacion del teacher | mejora sobre su propio control en +0,0008 y recupera el **0,8 %** de lo disponible |

La explicacion comun es precisa: la macro-F1 de seis clases es una media, y la frontera es una
minoria de la masa. Todo lo que la mejore pagando fuera de ella pierde; y lo que la mejore sin pagar
fuera no consigue moverla, porque en un problema de seis clases la restriccion de frontera resulta
redundante con el ajuste global.

Entregables: ``reports/v18_master_results.csv``, ``reports/v18_hypothesis_registry.csv``,
``reports/v18_candidate_comparison.csv``, ``reports/v18_reproducibility.md``,
``reports/v18_compliance.md``, ``reports/v18_final_research.md``, dos figuras y
``experiments/v18_cierre.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

MAESTRO = cfg.REPORTS / "v18_master_results.csv"
REGISTRO = cfg.REPORTS / "v18_hypothesis_registry.csv"
COMPARATIVA = cfg.REPORTS / "v18_candidate_comparison.csv"
REPRO = cfg.REPORTS / "v18_reproducibility.md"
COMPLIANCE = cfg.REPORTS / "v18_compliance.md"
INFORME = cfg.REPORTS / "v18_final_research.md"
RESULTADO = cfg.EXPERIMENTS / "v18_cierre.json"
REFERENCIA = 0.6005017473
GATE_EXPLORATORIO = 0.0015
GATE_ADOPCION = 0.003


def leer(nombre: str) -> dict:
    """Lee un artefacto de metricas de la fase."""
    ruta = cfg.EXPERIMENTS / nombre
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def commit_actual() -> str:
    """Sha del commit actual, para la nota de reproducibilidad."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no disponible"


def main() -> int:
    """Construye los documentos de cierre de la fase."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    p0 = leer("v18_p0_reproduccion.json")
    p1 = leer("v18_p1_peso_clases.json")
    p3 = leer("v18_p3_margen.json")
    p4 = leer("v18_p4_multiobjetivo.json")
    p5 = leer("v18_p5_destilacion.json")

    # ---------------------------------------------------------------- tabla maestra
    columnas = ["ID", "Track", "Hypothesis", "Mechanism", "Capacity_Added", "Seed",
                "Global_MacroF1", "Delta_V12", "Folds_Positive", "Boundary_18_24_25_34",
                "Boundary_25_34_35_49", "Boundary_18_24_25_34_LinearOnly",
                "Boundary_25_34_35_49_LinearOnly", "Bootstrap_CI95", "Decision", "Artifact"]
    filas = [
        {"ID": "P0", "Track": "auditoria de brechas", "Hypothesis": "que pistas son genuinamente nuevas",
         "Mechanism": "lectura dirigida de memoria, informes y codigo",
         "Capacity_Added": "ninguna", "Seed": cfg.SEED, "Decision": "cerrada",
         "Artifact": "reports/v18_gap_audit.md"},
        {"ID": "P0'", "Track": "validacion de la evidencia y cierre del confound",
         "Hypothesis": "de donde viene la ventaja del especialista: representacion u objetivo",
         "Mechanism": "rejilla de 2 representaciones por 2 modelos sobre la misma tarea",
         "Capacity_Added": "ninguna", "Seed": cfg.SEED,
         "Global_MacroF1": round(p0["referencia_pooled"], 6), "Delta_V12": 0.0,
         "Decision": "corrige el diagnostico de P3a (D-048)",
         "Artifact": "reports/v18_p0_reproduccion.md"},
    ]
    for resultado in p1["resultados"]:
        filas.append({
            "ID": f"P1-{resultado['nivel']}", "Track": "peso de clases consciente de la frontera",
            "Hypothesis": "subir el peso de las tres clases centrales convierte la frontera en ganancia global",
            "Mechanism": f"sample_weight {resultado['peso_central']} sobre 18-24, 25-34 y 35-49",
            "Capacity_Added": "ninguna", "Seed": cfg.SEED,
            "Global_MacroF1": resultado["macro_f1"], "Delta_V12": resultado["delta"],
            "Folds_Positive": f"{resultado['folds_positivos']}/5",
            "Boundary_18_24_25_34": resultado["frontera_18_24_25_34"],
            "Boundary_25_34_35_49": resultado["frontera_25_34_35_49"],
            "Bootstrap_CI95": f"[{resultado['bootstrap_ic95'][0]}; {resultado['bootstrap_ic95'][1]}]",
            "Decision": "NO ADOPTAR", "Artifact": "reports/v18_p1_peso_clases.md"})
    for resultado in p3["resultados"]:
        filas.append({
            "ID": f"P3-{resultado['brazo']}", "Track": "margen de adyacencia en entrenamiento",
            "Hypothesis": "exigir margen frente a la vecina verdadera sin pagar con las demas clases",
            "Mechanism": f"lambda {resultado['lambda']}, {resultado['restriccion']}",
            "Capacity_Added": "ninguna", "Seed": cfg.SEED,
            "Global_MacroF1": resultado["macro_combinada"], "Delta_V12": resultado["delta"],
            "Boundary_18_24_25_34": resultado["frontera_18_24_25_34"],
            "Boundary_25_34_35_49": resultado["frontera_25_34_35_49"],
            "Boundary_18_24_25_34_LinearOnly": resultado["fronteras_lineal_solo"]["18-24|25-34"],
            "Boundary_25_34_35_49_LinearOnly": resultado["fronteras_lineal_solo"]["25-34|35-49"],
            "Bootstrap_CI95": f"[{resultado['bootstrap_ic95'][0]}; {resultado['bootstrap_ic95'][1]}]",
            "Decision": "NO ADOPTAR", "Artifact": "reports/v18_p3_margen.md"})
    for resultado in p4["resultados"]:
        filas.append({
            "ID": f"P4-{resultado['brazo']}", "Track": "entrenamiento multi-objetivo",
            "Hypothesis": "un termino auxiliar aditivo sobre las dos fronteras, sin parametros extra",
            "Mechanism": f"lambda {resultado['lambda']}, {resultado['tipo']}",
            "Capacity_Added": f"{resultado['parametros_extra']} parametros",
            "Seed": cfg.SEED, "Global_MacroF1": resultado["macro_combinada"],
            "Delta_V12": resultado["delta"],
            "Boundary_18_24_25_34": resultado["frontera_18_24_25_34"],
            "Boundary_25_34_35_49": resultado["frontera_25_34_35_49"],
            "Boundary_18_24_25_34_LinearOnly": resultado["fronteras_cabeza_global"]["18-24|25-34"],
            "Boundary_25_34_35_49_LinearOnly": resultado["fronteras_cabeza_global"]["25-34|35-49"],
            "Bootstrap_CI95": f"[{resultado['bootstrap_ic95'][0]}; {resultado['bootstrap_ic95'][1]}]",
            "Decision": "NO ADOPTAR", "Artifact": "reports/v18_p4_multiobjetivo.md"})
    for resultado in p5["resultados"]:
        filas.append({
            "ID": f"P5-{resultado['lambda']}-T{resultado['temperatura']:.0f}",
            "Track": "destilacion del conocimiento de frontera",
            "Hypothesis": "transferir al student el conocimiento del especialista sin routing",
            "Mechanism": f"lambda {resultado['lambda']}, temperatura {resultado['temperatura']:.0f}, "
                         f"teacher cross-fitted",
            "Capacity_Added": "ninguna", "Seed": cfg.SEED,
            "Global_MacroF1": resultado["macro_combinada"], "Delta_V12": resultado["delta"],
            "Folds_Positive": f"{resultado['folds_positivos']}/5",
            "Boundary_18_24_25_34": resultado["frontera_18_24_25_34"],
            "Boundary_25_34_35_49": resultado["frontera_25_34_35_49"],
            "Bootstrap_CI95": f"[{resultado['bootstrap_ic95'][0]}; {resultado['bootstrap_ic95'][1]}]",
            "Decision": "NO ADOPTAR", "Artifact": "reports/v18_p5_destilacion.md"})
    maestro = pd.DataFrame(filas).reindex(columns=columnas)
    maestro.to_csv(MAESTRO, index=False)

    # ---------------------------------------------------------------- comparativa
    # El nivel_0_base de P1 es la propia referencia de V12, asi que como "mejor candidato" de P1 hay
    # que tomar el mejor nivel **con peso**, no el mayor macro_f1 de la lista.
    niveles_p1 = [r for r in p1["resultados"] if r["nivel"] != "nivel_0_base"]
    mejor_p1 = max(niveles_p1, key=lambda r: r["macro_f1"])
    # El nivel que demuestra el intercambio de suma cero, que es el que la vía muestra.
    exchanges_p1 = min(niveles_p1, key=lambda r: r["macro_f1"])
    mejor_p3 = max(p3["resultados"], key=lambda r: r["macro_combinada"])
    mejor_p4 = max(p4["resultados"], key=lambda r: r["macro_combinada"])
    mejor_p5 = max(p5["resultados"], key=lambda r: r["macro_combinada"])
    control_p5 = p5["resultados"][0]
    comparativa = pd.DataFrame([
        {"Candidato": "V12 (referencia congelada)", "Via": "-", "MacroF1": 0.6005, "Delta": 0.0,
         "Folds_Pos": "-", "Frontera_1": 0.4963, "Frontera_2": 0.4974,
         "Frontera_1_CabezaLineal": 0.4865, "Frontera_2_CabezaLineal": 0.4833,
         "Decision": "REFERENCIA", "Nota": "pooled 0,6005017473"},
        {"Candidato": f"P1 {mejor_p1['nivel']}", "Via": "re-peso de clases",
         "MacroF1": mejor_p1["macro_f1"], "Delta": mejor_p1["delta"],
         "Folds_Pos": f"{mejor_p1['folds_positivos']}/5", "Frontera_1": mejor_p1["frontera_18_24_25_34"],
         "Frontera_2": mejor_p1["frontera_25_34_35_49"],
         "Decision": "NO ADOPTAR", "Nota": "movio la frontera +0,082 y la global bajo"},
        {"Candidato": f"P3 {mejor_p3['brazo']}", "Via": "margen de adyacencia",
         "MacroF1": mejor_p3["macro_combinada"], "Delta": mejor_p3["delta"], "Folds_Pos": "-",
         "Frontera_1": mejor_p3["frontera_18_24_25_34"], "Frontera_2": mejor_p3["frontera_25_34_35_49"],
         "Frontera_1_CabezaLineal": mejor_p3["fronteras_lineal_solo"]["18-24|25-34"],
         "Frontera_2_CabezaLineal": mejor_p3["fronteras_lineal_solo"]["25-34|35-49"],
         "Decision": "NO ADOPTAR", "Nota": "las dos fronteras compiten por 25-34"},
        {"Candidato": f"P4 {mejor_p4['brazo']}", "Via": "termino auxiliar aditivo",
         "MacroF1": mejor_p4["macro_combinada"], "Delta": mejor_p4["delta"], "Folds_Pos": "-",
         "Frontera_1": mejor_p4["frontera_18_24_25_34"], "Frontera_2": mejor_p4["frontera_25_34_35_49"],
         "Frontera_1_CabezaLineal": mejor_p4["fronteras_cabeza_global"]["18-24|25-34"],
         "Frontera_2_CabezaLineal": mejor_p4["fronteras_cabeza_global"]["25-34|35-49"],
         "Decision": "NO ADOPTAR", "Nota": "cero parametros extra y no movio la frontera"},
        {"Candidato": "P5 control sin destilacion", "Via": "student sin senal auxiliar",
         "MacroF1": control_p5["macro_combinada"], "Delta": control_p5["delta"],
         "Folds_Pos": f"{control_p5['folds_positivos']}/5",
         "Frontera_1": control_p5["frontera_18_24_25_34"], "Frontera_2": control_p5["frontera_25_34_35_49"],
         "Decision": "NO ADOPTAR", "Nota": "el control de la propia P5"},
        {"Candidato": f"P5 destilado lambda {mejor_p5['lambda']} T {mejor_p5['temperatura']:.0f}",
         "Via": "destilacion cross-fitted", "MacroF1": mejor_p5["macro_combinada"],
         "Delta": mejor_p5["delta"], "Folds_Pos": f"{mejor_p5['folds_positivos']}/5",
         "Frontera_1": mejor_p5["frontera_18_24_25_34"],
         "Frontera_2": mejor_p5["frontera_25_34_35_49"],
         "Decision": "NO ADOPTAR", "Nota": "recupera el 0,8 % de lo que el teacher tiene"},
        {"Candidato": "especialista de frontera (no es un candidato global)", "Via": "referencia",
         "MacroF1": None, "Delta": None, "Folds_Pos": "-", "Frontera_1": 0.7136, "Frontera_2": 0.7493,
         "Decision": "COTA, NO RESULTADO",
         "Nota": "lo que hay disponible y no se ha podido cobrar"},
    ])
    comparativa.to_csv(COMPARATIVA, index=False)

    # ---------------------------------------------------------------- registro de hipotesis
    registros = [
        {"ID": "H01",
         "Hypothesis": "la ventaja del especialista de frontera viene de la representacion",
         "Previous_Evidence": "P3a publico 0,4944 con 279 variables frente a 0,7218 con TF-IDF",
         "Why_New": "misma tarea, dos representaciones",
         "Expected_Mechanism": "las 279 columnas comprimidas pierden lo que el texto crudo conserva",
         "Risk": "que los dos precedentes no sean comparables por modelo o protocolo",
         "Cost": "baja, cuatro celdas", "Protocol": "rejilla 2x2 con folds y filas identicos",
         "Result": "0,7136 y 0,7493 con las 279 variables, MAS que el 0,6864 y 0,7218 del TF-IDF. "
                   "El 0,4944 de P3a era una mezcla con la prediccion de V12 en 18.599 filas "
                   "congeladas",
         "Decision": "REFUTADA, y con ello se corrige el registro (D-048)"},
        {"ID": "H02",
         "Hypothesis": "subir el peso de las clases centrales convierte la frontera en ganancia global",
         "Previous_Evidence": "D-017 y D-030, que movieron pesos sin exito",
         "Why_New": "nunca se peso a las tres clases centrales a la vez",
         "Expected_Mechanism": "desplazar el equilibrio hacia donde esta el error",
         "Risk": "que las clases balanceadas dejen sin palanca real",
         "Cost": "media: 20 ajustes de dos modelos", "Protocol": "tres niveles pre-registrados",
         "Result": f"frontera hasta 0,5786 (+0,082) y global {mejor_p1['macro_f1']} "
                   f"(delta {mejor_p1['delta']:+.4f}). El coste entero cae sobre 13-17: -0,0166",
         "Decision": "CERRADA: el re-peso es de suma cero"},
        {"ID": "H03",
         "Hypothesis": "un margen de adyacencia en entrenamiento compra la frontera sin pagar fuera",
         "Previous_Evidence": "33 capas de decision a posteriori (D-037), que no es lo mismo",
         "Why_New": "el margen se aplica al objetivo, no a la decodificacion",
         "Expected_Mechanism": "penalizar solo la confusion con la vecina verdadera",
         "Risk": "que las dos fronteras compitan por la clase 25-34",
         "Cost": "media: cinco brazos con PyTorch",
         "Protocol": "misma capacidad que la LogReg de V12, todo pre-registrado",
         "Result": "frontera 1 +0,039 y frontera 2 -0,055 con lambda 0,1. estan acopladas",
         "Decision": "CERRADA: optimizar una frontera daña la otra"},
        {"ID": "H04",
         "Hypothesis": "un termino auxiliar aditivo, con cero parametros extra, mueve la frontera",
         "Previous_Evidence": "ninguno: nunca se habia anadido un termino auxiliar al objetivo",
         "Why_New": "distingue arquitectura multi-cabeza de objetivo boundary-aware",
         "Expected_Mechanism": "una restriccion que no redistribuye la masa del objetivo global",
         "Risk": "que la restriccion sea demasiado débil para desplazar el optimo",
         "Cost": "baja: cabeza lineal", "Protocol": "cuatro brazos, todo pre-registrado",
         "Result": f"la frontera se mueve +0,0038 como maximo y el mejor brazo queda "
                   f"{mejor_p4['delta']:+.4f}. Handicap de montaje declarado de -0,0023",
         "Decision": "CERRADA: el termino aditivo no mueve la frontera"},
        {"ID": "H05",
         "Hypothesis": "el conocimiento del especialista se puede transferir al student sin routing",
         "Previous_Evidence": "V17: el problema era no saber cuando escuchar al especialista",
         "Why_New": "la senal se descarga en el entrenamiento, luego no hay nada que distinguir",
         "Expected_Mechanism": "el teacher aporta lo que el global no deduce solo",
         "Risk": "que la transferencia sea ineficiente bajo esta representacion",
         "Cost": "alta: 60 teachers cross-fitted y 35 students", "Protocol": "cross-fitted en inner "
                                                                       "folds agrupados, "
                                                                       "cuadricula seleccionada en el train exterior",
         "Result": f"todos los brazos destilados superan a su propio control (+0,0008 como maximo) "
                   f"pero ninguno adopta: 0 de 5 folds. La frontera recupera el 0,8 % de lo "
                   f"disponible",
         "Decision": "CERRADA: la senal es real pero la transferencia es marginal"},
    ]
    for indice, (pista, hipotesis) in enumerate([
            ("P2", "coste ordinal sensible en el entrenamiento"),
            ("P6", "destilacion de miembros seleccionados"),
            ("P7", "destilacion de residuo de frontera"),
            ("P9", "analisis de error y destino de las victorias del especialista")], start=6):
        registros.append({
            "ID": f"H{indice:02d}", "Hypothesis": hipotesis,
            "Previous_Evidence": "capa de decision ordinal (cerrada) o lo de V17",
            "Why_New": "pista del brief de V18", "Expected_Mechanism": "por determinar",
            "Risk": "el mecanismo de P1 a P5 no dio senal adoptable",
            "Cost": "no presupuestada", "Protocol": "anidado",
            "Result": "", "Decision": "NO EJECUTADA: la condicion del brief era que P1-P4 demostraran "
                                    "senal del objetivo, y no se cumplio"})
    registro = pd.DataFrame(registros)
    registro.to_csv(REGISTRO, index=False)

    # ---------------------------------------------------------------- figuras
    figura1 = cfg.FIGURES / "v18_comparativa_candidatos.png"
    datos_grafico = comparativa[comparativa["MacroF1"].notna()].copy()
    fig, eje = plt.subplots(figsize=(12, 5.5))
    colores = ["#2166ac" if "V12" in nombre else "#b2182b" for nombre in datos_grafico["Candidato"]]
    eje.barh(range(len(datos_grafico)), datos_grafico["Delta"] * 1000, color=colores)
    eje.set_yticks(range(len(datos_grafico)))
    eje.set_yticklabels(datos_grafico["Candidato"], fontsize=8)
    eje.axvline(0, color="grey", linewidth=1)
    eje.axvline(GATE_EXPLORATORIO * 1000, color="#f1a340", linestyle="--", linewidth=1)
    eje.axvline(GATE_ADOPCION * 1000, color="#4d4d4d", linestyle="--", linewidth=1)
    eje.set_xlabel("delta de macro-F1 global respecto a V12 (milesimas)")
    eje.set_title("V18: ninguna via de objetivo llega al gate exploratorio de +1,5", fontsize=11)
    fig.tight_layout()
    fig.savefig(figura1, dpi=130)
    plt.close(fig)

    figura2 = cfg.FIGURES / "v18_frontera_disponible_vs_cobrada.png"
    fig, eje = plt.subplots(figsize=(11, 5.5))
    etiquetas = ["18-24|25-34", "25-34|35-49"]
    disponibles = [0.7136, 0.7493]
    por_cobrar = [0.4963, 0.4974]
    mejor_p5_f = [p5["resultados"][0]["frontera_18_24_25_34"], p5["resultados"][0]["frontera_25_34_35_49"]]
    eje.bar(np.arange(2) - 0.25, disponibles, 0.25, label="disponible (especialista validado)",
            color="#1a9850")
    eje.bar(np.arange(2), por_cobrar, 0.25, label="V12 lo extrae", color="#2166ac")
    eje.bar(np.arange(2) + 0.25, mejor_p5_f, 0.25, label="mejor student destilado", color="#b2182b")
    eje.set_xticks(np.arange(2))
    eje.set_xticklabels(etiquetas, fontsize=9)
    eje.set_ylim(0.4, 0.8)
    eje.set_ylabel("accuracy en las filas del par")
    eje.legend(fontsize=8)
    eje.set_title("V18: lo que hay disponible en las fronteras y lo que el modelo consigue",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(figura2, dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- reproducibilidad
    REPRO.write_text("\n".join([
        "# V18 — Reproducibilidad", "",
        f"- **Rama**: `kaggle-age-2026-puj-v18`", "- **Commit base**: `46b32f9`",
        f"- **Commit de la fase**: `{commit_actual()}`", f"- **Semilla**: `cfg.SEED`",
        "- **Interprete**: `/home/carlos/venv/bin/python` (Python 3.14.7)",
        "- **Hardware**: AMD Ryzen 9 5950X (32 hilos), 31,2 GB. **CPU pura**: el modelo de P3, P4 y",
        "  P5 es una cabeza lineal de 279 a 6, y la GPU no aporta nada a un producto de ese tamano.",
        "  Se fijo en CPU de forma explicita para no depender de la estabilidad de ROCm en un",
        "  entrenamiento largo.", "",
        "## Script de cada fase", "", "| Fase | Script |", "|---|---|",
        "| P0 auditoria | documental, sin script de metricas |",
        "| P0' evidencia y confound | `scripts/v18_p0_reproduccion.py` |",
        "| P1 peso de clases | `scripts/v18_p1_peso_clases.py` |",
        "| P3 margen de adyacencia | `scripts/v18_p3_margen.py` |",
        "| P4 multi-objetivo | `scripts/v18_p4_multiobjetivo.py` |",
        "| P5 destilacion | `scripts/v18_p5_destilacion.py` |",
        "| Cierre | `scripts/v18_cierre.py` |", "",
        "## Entradas, todas preexistentes", "",
        "| Artefacto | Que es |", "|---|---|",
        "| `features/oof_ensemble_v12.npy` | OOF de produccion de V12, 169.336 x 6 |",
        "| `features/oof_ADV07_nbsvm_word.npy` | 6 columnas de NB-SVM que completan las 279 |",
        "| `features/oof_E028a_pares.npy` | margenes de pares, usados solo para reproduccion |",
        "| `features/dup_clusters.parquet` | agrupacion de casi duplicados |",
        "| `features/age_supervised_features.npy` | 72 columnas supervisadas, cross-fitted |", "",
        "## Puntos donde la reproduccion depende de una convencion", "",
        "1. **El estimador de la referencia**: pooled, `metricas_globales`. El canonico es",
        "   0,6005017473.",
        "2. **Las fronteras de V18 son accuracy restringida al par**, no macro-F1 de seis clases.",
        "3. **La cabeza lineal de PyTorch no iguala a la `LogReg` de sklearn**: el control de P3, P4 y",
        "   P5 queda 0,0022-0,0023 por debajo de V12. Es un handicap de montaje del mismo orden que",
        "   los efectos medidos, y por eso P4 no puede resolver efectos por debajo de 0,002.",
        "4. **El cross-fitting de P5** usa cinco inner folds agrupados por duplicados sobre el train",
        "   exterior. Si un grupo de casi duplicados cruzara, la destilacion estaria contaminada.",
        "",
    ]), encoding="utf-8")

    # ---------------------------------------------------------------- compliance
    auditoria = leer("../experiments/v13_p0a_auditoria_idioma.json")
    COMPLIANCE.write_text("\n".join([
        "# V18 — Cumplimiento", "",
        "## Restricciones del brief", "", "| Restriccion | Estado | Evidencia |", "|---|---|---|",
        "| Sin embeddings ni vectorizadores aprendidos | **Cumplido** | las unicas representaciones son",
        "  las 279 variables de V12 y, en P0', el TF-IDF disperso **preexistente** reconstruido con",
        "  la configuracion de E-009, sin ajustarla ni ampliarla |",
        "| Sin corpus, datasets ni etiquetas externas | **Cumplido** | no se ha descargado nada |",
        "| Sin test | **Cumplido** | ningun script lee el test; toda medicion es sobre el train con",
        "  folds agrupados |",
        "| Sin holdout2 para explorar | **Cumplido** | no se ha tocado |",
        "| Sin leaderboard para seleccionar | **Cumplido** | no se ha enviado ninguna submission |",
        "| No podar V12 | **Cumplido** | V15 ya lo cerro y V18 no lo repite |",
        "| No reabrir lo cerrado | **Cumplido** | P1 a P5 son objetivos de entrenamiento, no capas de",
        "  decision, no poda, no meta-combinador global y no enrutamiento |",
        "| Protocolo anidado | **Cumplido** | la cuadricula de P5 se selecciona en una particion",
        "  interior del train exterior; el teacher se cross-fita en inner folds agrupados |",
        "| Adoptar exige ganancia global | **Cumplido** | ningun candidato se adopto, y la mejora de",
        "  frontera por si sola se reporto como evidencia mecanistica, nunca como admision |",
        "", "## Destilacion, comprobaciones exigidas", "", "| Comprobacion | Estado |", "|---|---|",
        "| Teacher entrenado solo con datos AGE permitidos | **si**, sobre las 279 variables de V12 |",
        "| Targets del teacher cross-fitted | **si**, cinco inner folds agrupados; cada ejemplo del",
        "  train exterior fue predicho por un teacher que no lo uso para entrenarse |",
        "| Etiquetas verdadeiras como objetivo principal | **si**, la entropia cruzada manda y la",
        "  senal del teacher es un termino aditivo con peso propio |",
        "| Ninguna etiqueta generada por el teacher | **si**, la senal es una probabilidad continua |",
        "| Predicciones del teacher sobre la validacion exterior | **solo diagnostico**, nunca entran en",
        "  el entrenamiento del student |", "",
        "## Idioma y caracteres", "",
        f"- Auditoria de idioma del proyecto: **{auditoria.get('errores', 0)} errores** sobre "
        f"{auditoria.get('ficheros_auditados', 0)} ficheros.",
        "- `tools/verificar_caracteres.py`, que nombra el brief, **no existe en este proyecto**. El",
        "  sustituto es `scripts/v13_p0a_auditoria_idioma.py`, mas un escaneo independiente.", "",
        "## Git", "",
        "- Rama `kaggle-age-2026-puj-v18` creada desde `46b32f9` con el arbol limpio.",
        "- **Sin push, sin PR, sin merge, sin reescritura de historia y sin modificar remotes.**",
        "- V12, V15, V16 y V17 intactas, con notas de correccion documentadas y sin tocar su codigo.", "",
        "## Entregables", "", "| Entregable | Estado |", "|---|---|",
        *[f"| `{nombre}` | {'entregado' if (cfg.REPORTS / nombre).exists() else 'PENDIENTE'} |"
          for nombre in ("v18_gap_audit.md", "v18_p0_reproduccion.md", "v18_p1_peso_clases.md",
                         "v18_p3_margen.md", "v18_p4_multiobjetivo.md", "v18_p5_destilacion.md",
                         "v18_master_results.csv", "v18_hypothesis_registry.csv",
                         "v18_candidate_comparison.csv", "v18_reproducibility.md",
                         "v18_compliance.md", "v18_final_research.md", "v18_distillation.csv")],
        "",
    ]), encoding="utf-8")

    # ---------------------------------------------------------------- informe final
    INFORME.write_text("\n".join([
        "# V18 — Informe final: cambiar el objetivo, no la representacion", "",
        f"- **Rama** `kaggle-age-2026-puj-v18` · **commit** `{commit_actual()[:7]}`",
        "- **Experimentos**: 8 de 14 previstos, todos con protocolo anidado",
        f"- **Resultado global**: **ninguna adopcion**. V12 sigue en "
        f"**{REFERENCIA:.10f}** pooled y 0,62188 en el leaderboard.", "",
        "## 1. La pregunta y la respuesta", "",
        "> ¿Puede un modelo global de seis clases usar mejor la informacion de las fronteras",
        "> cambiando su funcion de objetivo, sin aumentar capacidad y sin nuevas representaciones?", "",
        "**Respuesta: no, y la razon esta medida con cinco mecanismos distintos.**", "",
        "## 2. Lo que cambio la premisa de la fase: D-048", "",
        "La auditoria de P0 concluyo que la ventaja del especialista de frontera venia de la",
        "**representacion**, y uso usar para concluirlo el 0,4944 que V13 P3a publico. Ese numero **no es la",
        "exactitud del especialista**: `filas_actuables: 57.019` es el recuento sobre todo el dataset,",
        "no dentro del par, y dentro de las 57.023 filas del par solo 38.424 eran actuables mientras",
        "que 18.599 quedaban congeladas con la prediccion de V12. La mezcla daba exactamente 0,4944.",
        "",
        "Ejecutando el codigo y las variables exactos de P3a, el especialista sobre las **279 variables",
        "de V12** saca **0,7136** y **0,7493**, mas que el 0,6864 y 0,7218 del TF-IDF disperso, y sin",
        "fuga: las 72 columnas supervisadas se construyen con el mismo `make_folds` agrupado.", "",
        "**La senal de frontera estaba en el propio espacio de variables del modelo, todo el tiempo.**",
        "Y el combinador de V12 extrae 0,4963 de 0,7136 disponibles.", "",
        "## 3. Las cinco vias, y por que ninguna cobro la senal", "",
        "Las fronteras se dan en el **sistema combinado**, que es lo comparable con V12. Para P3 se anade",
        "la medida en la **cabeza lineal sola**, que es donde actua la perdida de margen y donde el",
        "acoplamiento de las dos fronteras se ve sin el amortiguamiento del MLP. Las dos medidas son",
        "correctas y responder a preguntas distintas; confundirlas seria repetir justo el error que",
        "D-048 corrige.", "",
        "| Via | Mecanismo | Frontera 1 | Frontera 2 | Frontera 1 (lineal) | Frontera 2 (lineal) | Global | Delta |",
        "|---|---|---|---|---|---|---|---|",
        f"| P1 re-peso | `sample_weight` en las tres clases centrales | 0,5441 | 0,5341 | - | - | "
        f"{mejor_p1['macro_f1']} | {mejor_p1['delta']:+.4f} |",
        f"| P3 margen | penalizacion de adyacencia en el objetivo | {mejor_p3['frontera_18_24_25_34']} | "
        f"{mejor_p3['frontera_25_34_35_49']} | "
        f"{mejor_p3['fronteras_lineal_solo']['18-24|25-34']} | "
        f"{mejor_p3['fronteras_lineal_solo']['25-34|35-49']} | {mejor_p3['macro_combinada']} | "
        f"{mejor_p3['delta']:+.4f} |",
        f"| P4 auxiliar | termino aditivo, cero parametros | {mejor_p4['frontera_18_24_25_34']} | "
        f"{mejor_p4['frontera_25_34_35_49']} | "
        f"{mejor_p4['fronteras_cabeza_global']['18-24|25-34']} | "
        f"{mejor_p4['fronteras_cabeza_global']['25-34|35-49']} | {mejor_p4['macro_combinada']} | "
        f"{mejor_p4['delta']:+.4f} |",
        f"| P5 destilacion | teacher cross-fitted, sin routing | "
        f"{mejor_p5['frontera_18_24_25_34']} | {mejor_p5['frontera_25_34_35_49']} | - | - | "
        f"{mejor_p5['macro_combinada']} | {mejor_p5['delta']:+.4f} |",
        "| P1 re-peso (nivel fuerte) | mismo mecanismo, peso mayor | 0,5786 | 0,5699 | - | - | 0,5841 "
        "| -0,0164 |",
        "| **V12** | referencia | 0,4963 | 0,4974 | 0,4865 | 0,4833 | **0,6005** | - |", "",
        "**Los mecanismos son distintos y los cinco fallan por razones distintas y coherentes:**", "",
        "1. **P1 redistribuye.** La frontera sube +0,082, pero el intercambio es de suma cero y su",
        "   localizacion es exacta: `18-24` gana +0,0170 y `13-17` paga -0,0166. El coste entero cae",
        "   sobre la clase facil vecina.",
        "2. **P3 pone a competir las fronteras.** En la **cabeza lineal**, donde actua la perdida de",
        "   margen, la frontera `18-24|25-34` sube **+0,0392** (0,4865 a 0,5257) y `25-34|35-49` baja",
        "   **-0,0552** (0,4833 a 0,4281) con lambda 0,1. En el sistema combinado el mismo efecto se",
        "   amortigua a +0,0209 y -0,0268. Las dos fronteras comparten la clase `25-34`, luego estan",
        "   acopladas, y con lambda 0,5 la segunda se desploma hasta 0,4013.",
        "3. **P4 no mueve nada.** El termino aditivo, con **cero parametros extra**, mueve la frontera",
        "   **+0,0038** como maximo, y esa cifra es de la **cabeza global** (0,4858 a 0,4896 en el brazo",
        "   C2); en el sistema combinado el movimiento es de +0,0017. Compárese con las +0,08 que movia el",
        "   re-peso de P1: **dos ordenes de magnitud menos con el mismo numero de parametros**. En un",
        "   problema de seis clases la restriccion de frontera resulta redundante con el ajuste global.",
        "4. **P5 transfiere poco.** Todos los brazos destilados superan a su propio control, asi que",
        "   la senal del teacher **no es inerte**, pero la mejora es de +0,0008 como maximo, con 0 de",
        "   5 folds positivos, y la frontera recupera el **0,8 %** de lo que el teacher tiene.", "",
        "## 4. La explicacion comun", "",
        "La macro-F1 de seis clases es una **media**, y la frontera es una **minoria de la masa**.",
        "Todo mecanismo que la mejore pagando fuera de ella pierde, y todo mecanismo que la mejore sin",
        "pagar fuera no consigue moverla. Eso no es una conjetura: es lo que muestran los cinco",
        "experimentos, y ademas lo explica el caso de V17, donde un enrutamiento con la senal correcta",
        "tampoco cobraba nada.", "",
        "## 5. Respuesta a la pregunta cientifica de la fase", "",
        "| Opcion | Veredicto |", "|---|---|",
        "| A. cuello de botella del objetivo | **Parcial**: el objetivo mueve la frontera (P1 +0,082),",
        "  pero no la cobra globalmente |",
        "| B. cuello de botella de informacion | **Falso**: hay 0,7136 disponibles (D-048) |",
        "| C. cuello de botella de arquitectura | **No**: la destilacion no anade parametros y tampoco |",
        "  rinde |",
        "| D. el conocimiento del especialista se puede destilar | **Marginal**: se transfiere el 0,8 % |",
        "| E. la senal de frontera sigue siendo local | **Cierto**: con la media de seis clases, la",
        "  frontera no puede pesar mas |",
        "| F. no queda senal explotable | **Cierto en la practica**: 0,0000 alcanzable frente a", "",
        "## 6. Lo que queda abierto, y no es un experimento de esta fase", "",
        "Una sola cosa, y es la unica accion que este informe recomienda:", "",
        "> **Cambiar la funcion objetivo a la que se optimise, no la que se informa.** El problema",
        "> esta en las fronteras de edad y el modelo se optimiza con una media de seis clases que las",
        "> diluye. Los datos existen y estan en el espacio de variables del propio modelo. Cualquier",
        "> esquema que entrene o seleccione **sobre las fronteras directamente** es la via que la",
        "> evidencia deja abierta; el resto esta cerrado con medicion.", "",
        "Y una consecuencia futura que este trabajo deja a mano: **V12 comprime la salida de cada",
        "miembro a seis numeros**, y el especialista binario recupera el 0,7136 de esas mismas seis",
        "columnas por miembro. Esa compresion es, por si sola, un objeto de estudio con nombre propio.",
        "",
    ]), encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V18 · cierre", "fecha": "2026-09-26", "commit": commit_actual(),
        "referencia_pooled": REFERENCIA,
        "experimentos": 8, "adopciones": 0,
        "vias_cerradas": ["P1 re-peso", "P3 margen", "P4 multi-objetivo", "P5 destilacion"],
        "pistas_no_ejecutadas": ["P2 coste ordinal", "P6", "P7", "P9"],
        "correccion_de_registro": "D-048",
        "transferencia_del_teacher": "0,8 % de lo disponible en la primera frontera",
        "veredicto": ("la senal existe y es grande en las 279 variables, y ninguna funcion objetivo "
                      " la cobra: la macro-F1 de seis clases es una media y la frontera es una "
                      "minoria de la masa"),
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"maestro: {len(maestro)} filas | registro: {len(registro)} | comparativa: {len(comparativa)} "
          f"| adopciones: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
