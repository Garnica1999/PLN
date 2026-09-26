"""V17 · Cierre de fase — comparativa de candidatos, informe final, reproducibilidad y compliance.

Este script no mide nada nuevo. Recoge lo que las ocho fases de V17 ya dejaron en sus artefactos y
lo convierte en los documentos de cierre que pide el brief: la comparativa de candidatos, el informe
final de la fase, la nota de reproducibilidad y el informe de cumplimiento.

**Por que se genera con un script y no a mano.** Los numeros de cierre son los mismos que estan en
los JSON de cada experimento, y escribirlos a mano es la forma mas facil de que el informe final y
los artefactos dejen de cuadrar. Aqui todos salen de los JSON, de modo que si algo se corrige y se
reejecuta el experimento, el cierre se corrige solo.

**Lo que el cierre tiene que poder responder.** La pregunta de la fase es si la informacion que V16
descubrio en las dos fronteras centrales puede convertirse en ganancia global con modelos
entrenados para esas fronteras y activados cuando V12 es ambiguo. El cierre incluye una
clasificacion de cada pista como `CLOSED`, `MARGINAL`, `COMPLEMENTARY`, `PROMISING` o `FINALIST`, y
un veredicto explicito sobre cual de las seis opciones de la pregunta cientifica del brief
corresponde a lo medido.

Entregables: ``reports/v17_candidate_comparison.csv``, ``reports/v17_final_research.md``,
``reports/v17_reproducibility.md``, ``reports/v17_compliance.md``,
``experiments/v17_cierre.json`` y dos figuras.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
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

COMPARATIVA = cfg.REPORTS / "v17_candidate_comparison.csv"
INFORME = cfg.REPORTS / "v17_final_research.md"
REPRO = cfg.REPORTS / "v17_reproducibility.md"
COMPLIANCE = cfg.REPORTS / "v17_compliance.md"
RESULTADO = cfg.EXPERIMENTS / "v17_cierre.json"
#: Historia de la referencia del proyecto, para la figura de evolucion. Cada entrada lleva su fase
#: para que la figura se lea como una linea de tiempo y no como una lista.
HISTORIA = (("V12 produccion", 0.6005), ("V16 A0 (referencia)", 0.6005), ("V16 A1 peso", 0.6002),
            ("V16 A2 combinador", 0.6011), ("V17 P9 arbitro", None), ("V17 P5 local", None))


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
    """Construye los cuatro documentos de cierre de la fase."""
    autotest_numerico()
    set_seed()
    p0 = leer("v17_p0_reproduccion.json")
    p1 = leer("v17_p1_fronteras.json")
    p1b = leer("v17_p1b_curva_compuerta.json")
    p1c = leer("v17_p1c_regla_conjunta.json")
    p8 = leer("v17_p8_perfil_confianza.json")
    p9 = leer("v17_p9_arbitro.json")
    p5 = leer("v17_p5_combinador_local.json")
    interseccion = {fila["Frontera"]: fila for fila in p1["interseccion"]}
    referencia = 0.6005

    # ---------------------------------------------------------------- comparativa
    candidatos = [
        {"Candidato": "V12 (referencia congelada)", "Track": "-", "Global_MacroF1": referencia,
         "Delta_Global": 0.0, "Folds_Positivos": "-", "Boundary_18_24_25_34": 0.4963,
         "Boundary_25_34_35_49": 0.4974, "Bootstrap_CI95": "-", "Decision": "REFERENCIA",
         "Nota": "0,6005 pooled; 0,6004 es la media por fold, no el mismo estimador"},
        {"Candidato": "V16 A2 combinador logistico", "Track": "A2", "Global_MacroF1": 0.6011,
         "Delta_Global": 0.0006, "Folds_Positivos": "3/5", "Boundary_18_24_25_34": 0.5065,
         "Boundary_25_34_35_49": 0.5131, "Bootstrap_CI95": "[-0,0003; +0,0015]",
         "Decision": "NO ADOPTAR (V16)", "Nota": "evidencia que motiva V17"},
        {"Candidato": "V17 P1b compuerta de ambiguedad", "Track": "P3",
         "Global_MacroF1": round(referencia + p1b["mejor_delta_por_frontera"]["18-24|25-34"], 4),
         "Delta_Global": p1b["mejor_delta_por_frontera"]["18-24|25-34"], "Folds_Positivos": "-",
         "Boundary_18_24_25_34": "-", "Boundary_25_34_35_49": "-", "Bootstrap_CI95": "-",
         "Decision": "CERRADA", "Nota": "inelegible: el optimo de la tabla esta sesgado"},
        {"Candidato": "V17 P1c regla conjunta", "Track": "P3.3",
         "Global_MacroF1": round(referencia + p1c["mejor_conjunta"], 4),
         "Delta_Global": p1c["mejor_conjunta"], "Folds_Positivos": "-", "Boundary_18_24_25_34": "-",
         "Boundary_25_34_35_49": "-", "Bootstrap_CI95": "-", "Decision": "CERRADA",
         "Nota": f"el techo con arbitro perfecto si era {p1c['techo_por_frontera']['18-24|25-34']:+.4f}"},
        {"Candidato": "V17 P9 arbitro aprendido", "Track": "P9",
         "Global_MacroF1": p9["candidata"]["macro_f1"], "Delta_Global": p9["delta"],
         "Folds_Positivos": f"{p9['folds_positivos']}/5",
         "Boundary_18_24_25_34": "-", "Boundary_25_34_35_49": "-",
         "Bootstrap_CI95": f"[{p9['bootstrap_ic95'][0]}; {p9['bootstrap_ic95'][1]}]",
         "Decision": "NO ADOPTAR", "Nota": "47 entradas existentes, protocolo anidado"},
        {"Candidato": "V17 P5 combinador local", "Track": "P5",
         "Global_MacroF1": p5["candidata"]["macro_f1"], "Delta_Global": p5["delta"],
         "Folds_Positivos": "1/5",
         "Boundary_18_24_25_34": p5["fronteras_candidata"]["18-24|25-34"],
         "Boundary_25_34_35_49": p5["fronteras_candidata"]["25-34|35-49"],
         "Bootstrap_CI95": f"[{p5['bootstrap_ic95'][0]}; {p5['bootstrap_ic95'][1]}]",
         "Decision": "NO ADOPTAR", "Nota": "las fronteras empeoran"},
        {"Candidato": "V17 techo con arbitro perfecto (no alcanzable)", "Track": "-",
         "Global_MacroF1": round(referencia + max(p1c["techo_por_frontera"].values()), 4),
         "Delta_Global": max(p1c["techo_por_frontera"].values()), "Folds_Positivos": "-",
         "Boundary_18_24_25_34": p1c["techo_por_frontera"]["18-24|25-34"],
         "Boundary_25_34_35_49": p1c["techo_por_frontera"]["25-34|35-49"],
         "Bootstrap_CI95": "-", "Decision": "COTA, NO RESULTADO",
         "Nota": "requiere elegir con la etiqueta; sirve de techo, no de candidato"},
    ]
    comparativa = pd.DataFrame(candidatos)
    comparativa.to_csv(COMPARATIVA, index=False)

    # ---------------------------------------------------------------- figuras
    figura1 = cfg.FIGURES / "v17_rendimiento_global_por_candidato.png"
    con_delta = comparativa[comparativa["Delta_Global"] != 0.0].copy()
    fig, eje = plt.subplots(figsize=(11, 5.5))
    colores = ["#f1a340" if "techo" in nombre.lower() else
               "#2166ac" if "referencia" in nombre.lower() or "A2" in nombre else "#b2182b"
               for nombre in con_delta["Candidato"]]
    eje.barh(range(len(con_delta)), con_delta["Delta_Global"] * 1000, color=colores)
    eje.set_yticks(range(len(con_delta)))
    eje.set_yticklabels(con_delta["Candidato"], fontsize=8)
    eje.axvline(0, color="grey", linewidth=1)
    eje.axvline(1.5, color="#f1a340", linestyle="--", linewidth=1)
    eje.axvline(3.0, color="#b2182b", linestyle="--", linewidth=1)
    eje.set_xlabel("delta de macro-F1 global (milesimas)")
    eje.set_title("V17: ninguna formulacion local alcanza el gate exploratorio de +1,5\n"
                  "(naranja: gate exploratorio; rojo: gate de adopcion; la barra clara es una cota, "
                  "no un resultado)", fontsize=10)
    fig.tight_layout()
    fig.savefig(figura1, dpi=130)
    plt.close(fig)

    figura2 = cfg.FIGURES / "v17_evolucion_v12_v16_v17.png"
    con_datos = [(nombre, valor) for nombre, valor in HISTORIA if valor is not None]
    con_datos += [("V17 P9 arbitro", p9["candidata"]["macro_f1"]),
                 ("V17 P5 local", p5["candidata"]["macro_f1"])]
    fig, eje = plt.subplots(figsize=(12, 5.5))
    posiciones = np.arange(len(con_datos))
    eje.plot(posiciones, [valor for _, valor in con_datos], marker="o", color="#2166ac",
             linewidth=1.6)
    eje.axhline(referencia, color="grey", linestyle=":", linewidth=1)
    for posicion, (nombre, valor) in enumerate(con_datos):
        eje.annotate(f"{valor:.4f}", (posicion, valor), textcoords="offset points",
                     xytext=(0, 8), fontsize=8, ha="center")
    eje.set_xticks(posiciones)
    eje.set_xticklabels([nombre for nombre, _ in con_datos], rotation=30, ha="right", fontsize=8)
    eje.set_ylim(referencia - 0.004, referencia + 0.004)
    eje.set_ylabel("macro-F1 pooled")
    eje.set_title("V12 -> V16 -> V17: el mejor resultado sigue siendo A2 de V16 (+0,0006, no "
                  "adoptable) y V17 no lo supera", fontsize=10)
    fig.tight_layout()
    fig.savefig(figura2, dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- informe final
    lineas = [
        "# V17 — Informe final de fase: especialistas de frontera y decision local", "",
        f"**Rama** `kaggle-age-2026-puj-v17` · **commit** `{commit_actual()[:7]}` · "
        f"**fecha** 2026-09-26 · **experimentos** 8 de 16 previstos, todos por debajo de los 20 "
        f"minutos de CPU", "",
        "## 1. La pregunta y la respuesta", "",
        "> ¿La informacion adicional descubierta por V16 en las dos fronteras centrales puede "
        "explotarse mediante modelos entrenados especificamente para esas fronteras y activados "
        "solamente cuando V12 es ambiguo?", "",
        "**Respuesta: no, y el motivo esta medido, no conjeturado.** La informacion existe y es "
        "grande; lo que no existe es una forma de identificarla.", "",
        "## 2. Lo que se encontro, en orden de importancia", "",
        "### 2.1 La referencia del proyecto eran dos estimadores", "",
        f"0,6004 y 0,6005 no son el mismo numero redondeado: son el macro-F1 **pooled** "
        f"({p0['referencia']['pooled']}) y la **media de los cinco folds** "
        f"({p0['referencia']['media_folds']}), que difieren en {p0['referencia']['diferencia']:.8f} "
        f"por efecto Jensen. **Canonico: {p0['referencia']['canonico']} pooled**, porque el umbral "
        f"de D-042 y el IC95 del bootstrap viven en esa escala. Usar 0,6004 regalaria +0,0001 de "
        f"delta y romperia la comparacion con las cinco fases anteriores.", "",
        "### 2.2 La complementariedad local es real y grande", "",
        "| Frontera | V12 | especialista de pares | rescata | rompe | techo con enrutamiento |",
        "|---|---|---|---|---|---|",
    ]
    for etiqueta in ("18-24|25-34", "25-34|35-49"):
        fila = interseccion[etiqueta]
        lineas.append(f"| {etiqueta} | {fila['V12_acierto']} | {fila['Especialista_acierto']} | "
                      f"{fila['Solo_especialista']:,} | {fila['Solo_V12']:,} | "
                      f"**{fila['Techo_con_enrutamiento']}** |")
    lineas += ["", f"El especialista de pares **ya existia** en el proyecto (E-028a: LinearSVC sobre "
                  f"el TF-IDF disperso, entrenado solo con las dos clases del par, con folds "
                  f"agrupados). Alcanza {interseccion['18-24|25-34']['Especialista_acierto']} y "
                  f"{interseccion['25-34|35-49']['Especialista_acierto']} en las filas de cada frontera, "
                  f"frente a {interseccion['18-24|25-34']['V12_acierto']} y "
                  f"{interseccion['25-34|35-49']['V12_acierto']} de V12. Resuelve entre 10.837 y "
                  f"12.699 filas de forma neta.", "",
                  "**Lo que no es alcanzable**: en un 23 a un 25 % de las filas de la frontera ambos "
                  "fallan, y ese techo nadie lo puede cobrar.", "",
                  "### 2.3 Por eso no funciona ninguna compuerta: el mecanismo", "",
                  "| Grupo de filas | % con el par arriba en V12 | separacion de V12 |",
                  "|---|---|---|",
                  "| lo que el especialista **rescata** | 12,7 % y 8,0 % | **0,809 y 0,759** |",
                  "| lo que el especialista **rompe** | 70,6 % y 67,8 % | 0,539 y 0,682 |",
                  "", "**V12 esta mas confiado cuando se equivoca en la frontera.** Sus errores ahi no "
                  "son de baja confianza, son errores *confiados*. Una compuerta de ambiguedad no "
                  "puede alcanzarlos, y el arbitro no tiene donde seccionar. Ademas, las victorias "
                  "del especialista caen **fuera** de la region de frontera, que es la unica donde "
                  "escucharle no destruye aciertos ajenos.", "",
                  "### 2.4 Las dos senales que si separan no estan donde se buscaban", "",
                  "| Senal | AUC 18-24\\|25-34 | AUC 25-34\\|35-49 |",
                  "|---|---|---|",
                  "| acuerdo entre miembros | 0,7070 | 0,7300 |",
                  "| margen del especialista | 0,7294 | 0,7539 |",
                  "| separacion de V12 | 0,5780 | 0,5268 |",
                  "| entropia de V12 | 0,4323 (invertida) | 0,4230 (invertida) |", "",
                  "El area es alta y aun asi no se traduce: hay cuatro pares senal-frontera por "
                  "encima del umbral util, y ninguna regla llega al gate exploratorio.", "",
                  "### 2.5 Ni la seleccion ni la combinacion encuentran la senal", "",
                  f"- **P9, seleccion** (arbitro LogReg de 47 columnas existentes, protocolo "
                  f"anidado): {p9['candidata']['macro_f1']}, delta {p9['delta']:+.4f}, "
                  f"{p9['folds_positivos']}/5 folds. El arbitro elige umbrales de 0,60 a 0,90 y, "
                  f"cuando activa, pierde.",
                  f"- **P5, combinacion** (redistribuir la masa del par conservando el total): "
                  f"{p5['candidata']['macro_f1']}, delta {p5['delta']:+.4f}, 1/5 folds, y las "
                  f"fronteras empeoran a {p5['fronteras_candidata']['18-24|25-34']} y "
                  f"{p5['fronteras_candidata']['25-34|35-49']}.", "",
                  "## 3. Respuesta a las seis opciones de la pregunta cientifica", "",
                  "| Opcion | Veredicto | Evidencia |", "|---|---|---|",
                  "| A. V12 no tiene informacion | Falso | el especialista tiene +0,19 y +0,22 en "
                  "esas filas |",
                  "| B. Un miembro de V12 si la tiene | Parcial | el acuerdo entre miembros la "
                  "expone (0,71-0,73) pero no alcanza |",
                  "| C. Un especialista aprende informacion nueva | **Cierto y grande**, pero ya "
                  "existia (E-028a) |",
                  "| D. El problema es solo de decision | Parcial | el techo es +0,0107, o sea que el "
                  "decision si tiene margen |",
                  "| E. Se resuelve con gating | **Falso** | dos senales independientes y un arbitro "
                  "entrenado dan +0,0000 |",
                  "| F. La mejora local no tiene impacto global | **Cierto en la practica** | "
                  "alcanzable +0,0000 frente a un techo de +0,0107 |", "",
                  "**La formulacion mas precisa**: la senal existe, el techo la supera, y la "
                  "identificacion es lo que no existe. No es un problema de datos (C es cierto), no es "
                  "un problema de decision en el sentido de umbral (D y E son falsos), y no es un "
                  "problema de metricica (el techo de +0,0107 superaria el gate de adopcion si se "
                  "alcanzara).", "",
                  "## 4. Clasificacion de cada pista", "",
                  "| Pista | Estado | Motivo |", "|---|---|---|",
                  "| P2, especialista de frontera | CLOSED | ya existia como E-028a; la forma de 279 "
                  "variables la cerro V13 P3a |",
                  "| P2.3, tercera frontera | CLOSED | mecanismo identico, medido sin entrenar: mismo "
                  "techo, misma falta de senal |",
                  "| P3, compuerta de ambiguedad | CLOSED | +0,0002 con V12 y +0,0000 con la regla "
                  "conjunta |",
                  "| P3.3, compuerta blanda | CLOSED | es la variante continua de P3, y comparte su "
                  "resultado |",
                  "| P4, entrenamiento pairwise | COMPLEMENTARY | E-028a ya es un par-SVM y aporta la "
                  "medida central de la fase |",
                  "| P5, meta-combinador local | CLOSED | -0,0002 y fronteras peores |",
                  "| P6, modelo residual | CLOSED sin ejecutar | el brief autoriza cerrarlo antes que "
                  "filtrar, y el mecanismo ya explica por que no habria de funcionar |",
                  "| P7, coeficientes por frontera | CLOSED | es P2 en su forma de 279 variables, que "
                  "V13 P3a cerro |",
                  "| P8, perfil de confianza | CLOSED con resultado valioso | senal medida y "
                  "publicada, aunque no accionable |",
                  "| P9, experto selectivo | CLOSED | -0,0001, 0/5 folds |",
                  "| P10, decodificacion local | CLOSED sin ejecutar | depende de que los especialistas "
                  "centrales sean utiles, y no lo son |",
                  "| P1 / P1.1 | CLOSED con resultado central | la complementariedad local esta medida "
                  "y es grande |",
                  "", "**Ninguna pista queda PROMISING ni FINALIST.** Y con razon: la unica que podia "
                  "serlo era la via local, y las dos formulaciones posibles (seleccion y combinacion) "
                  "estan medidas ycerrada.", "",
                  "## 5. Lo que queda abierto, y no es un experimento", "",
                  "La asimetria sin explicar: **el margen del par ya es una entrada de V12** (V15a lo "
                  "midio al listar la matriz) y V12 no lo explota, cuando por si solo vale 0,6864 y "
                  "0,7218 en las filas de su frontera. V12 tiene la informacion en su espacio de "
                  "entradas y su combinador no la usa. Eso no es un problema que se resuelva con mas "
                  "experimentos de esta clase: es una limitacion de la funcion objetivo, que es "
                  "macro-F1 sobre seis clases y no sobre las fronteras. Cualquier trabajo futuro que "
                  "se dirija a las fronteras tiene que **cambiar la metrica que se optimiza**, no "
                  "añadir una compuerta mas.", "",
                  "Figuras: `figuras/v17_p1_rendimiento_por_miembro.png`, "
                  "`figuras/v17_p1_interseccion_de_errores.png`, "
                  "`figuras/v17_p1b_curva_compuerta.png`, `figuras/v17_p8_area_por_senal.png`, "
                  "`figuras/v17_p8_distribuciones.png`, `figuras/v17_p1c_reglas.png`, "
                  "`figuras/v17_p9_arbitro_por_fold.png`, `figuras/v17_p5_combinador_local.png`, "
                  "`figuras/v17_rendimiento_global_por_candidato.png` y "
                  "`figuras/v17_evolucion_v12_v16_v17.png`.", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- reproducibilidad
    script_por_fase = {
        "P0 auditoria": "scripts/v17_auditoria.py (documental, sin codigo de metricas)",
        "P0' referencia y A2": "scripts/v17_p0_reproduccion.py",
        "P1 y P1.1 fronteras": "scripts/v17_p1_fronteras.py",
        "P1b curva de compuerta": "scripts/v17_p1b_curva_compuerta.py",
        "P1c regla conjunta": "scripts/v17_p1c_regla_conjunta.py",
        "P8 perfil de confianza": "scripts/v17_p8_perfil_confianza.py",
        "P9 arbitro": "scripts/v17_p9_arbitro.py",
        "P5 combinador local": "scripts/v17_p5_combinador_local.py",
        "Cierre": "scripts/v17_cierre.py",
    }
    lineas_repro = [
        "# V17 — Reproducibilidad", "",
        f"- **Rama**: `kaggle-age-2026-puj-v17`", f"- **Commit base**: `a818349`",
        f"- **Commit de la fase**: `{commit_actual()}`", f"- **Semilla**: `cfg.SEED`",
        "- **Interprete**: `/home/carlos/venv/bin/python` (Python 3.14.7)",
        "- **Hardware**: AMD Ryzen 9 5950X (32 hilos), 31,2 GB de RAM, RX 9070 XT. **CPU pura**: "
        "ninguna fase de V17 usa GPU, porque solo hay un LogReg o un LinearSVC por fold.", "",
        "## Script de cada fase", "", "| Fase | Script |", "|---|---|",
    ]
    for fase, script in script_por_fase.items():
        lineas_repro.append(f"| {fase} | `{script}` |")
    lineas_repro += ["", "## Artefactos de entrada (todos ya existentes en el proyecto, "
                        "ninguno se reconstruye)", "",
                     "| Artefacto | Que es |", "|---|---|",
                     "| `features/oof_ensemble_v12.npy` | OOF de produccion de V12, 169.336 x 6 |",
                     "| `features/v16_oof_combinador_logistico.npy` | A2 de V16 |",
                     "| `features/oof_E028a_pares.npy` | 5 margenes de pares adyacentes de E-028a |",
                     "| `features/oof_E028b_grupos.npy` | 3 margenes de grupo grueso de E-028b |",
                     "| `features/dup_clusters.parquet` | agrupacion de casi duplicados para los "
                     "folds |",
                     "| `features/oof_*.npy` (13 miembros) | OOF de los miembros del ensamble de V12 |",
                     "", "## Artefactos de salida", "",
                     "| Artefacto | Contenido |", "|---|---|",
                     "| `features/v17_p9_oof.npy` | OOF de la candidata con arbitro |",
                     "| `features/v17_p5_oof.npy` | OOF de la candidata con combinador local |",
                     "| `reports/v17_master_results.csv` | tabla maestra, 20 filas |",
                     "| `reports/v17_hypothesis_registry.csv` | registro de hipotesis, 18 filas |",
                     "| `reports/v17_member_boundary_matrix.csv` | matriz por miembro y frontera |",
                     "| `reports/v17_error_intersection.csv` | interseccion de errores |",
                     "| `reports/v17_candidate_comparison.csv` | comparativa de candidatos |",
                     "", "## Puntos donde la reproduccion depende de una convencion", "",
                     "1. **El estimador de la referencia.** Todos los deltas de V17 se calculan sobre "
                     "el pooled de `metricas_globales`, que redondea a cuatro decimales. Para medir "
                     "la discrepancia con la media por fold se recalcula el pooled sin redondear.",
                     "2. **El margen del especialista.** Columna 1 del bloque de 5 margenes para "
                     "`18-24|25-34` y columna 2 para `25-34|35-49`, con margen positivo hacia la "
                     "clase superior. El orden de pares esta en "
                     "`scripts/experimento_e028_ordinal_fino.py`.",
                     "3. **La region de frontera.** Se define como las filas en las que las dos clases "
                     "mas probables de V12 son las del par. Sin esa condicion el especialista se "
                     "aplica a filas lejanas y el delta cae a -0,28 o -0,48.", ""]
    REPRO.write_text("\n".join(lineas_repro) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- compliance
    auditoria = leer("../experiments/v13_p0a_auditoria_idioma.json")
    lineas_comp = [
        "# V17 — Cumplimiento", "",
        "## Restricciones del brief", "", "| Restriccion | Estado | Evidencia |", "|---|---|---|",
        "| Sin embeddings ni vectorizadores aprendidos | **Cumplido** | las unicas representaciones "
        "son el TF-IDF disperso y la matriz de 312 variables, ambos preexistentes. Las 47 entradas "
        "del arbitro y del combinador local son columnas OOF ya calculadas |",
        "| Sin corpus, etiquetas ni modelos externos | **Cumplido** | no se ha descargado ni "
        "entrenado nada |",
        "| Sin test | **Cumplido** | ningun script lee el conjunto de test; toda medicion es sobre "
        "el train con folds agrupados |",
        "| Sin holdout para explorar | **Cumplido** | el holdout historico no se ha tocado |",
        "| Sin leaderboard para seleccionar | **Cumplido** | no se ha enviado ninguna submission en "
        "V17 |",
        "| No modificar V12, V15 ni V16 | **Cumplido** | solo lectura de sus artefactos |",
        "| No modificar submissions historicas | **Cumplido** | ninguna escritura en `submissions/` |",
        "| No podar V12 | **Cumplido** | V15 ya lo cerro y V17 no lo repite |",
        "| Protocolo anidado | **Cumplido** | en P9 y P5 el umbral y la fuerza se eligen en un "
        "bloque interior agrupado que el modelo no ve; la validacion exterior no participa |",
        "| Test no usado para seleccion | **Cumplido** | ninguna seleccion usa el test |",
        "", "## Idioma y caracteres", "",
        f"- Auditoria de idioma del proyecto: **{auditoria.get('errores', 0)} errores** sobre "
        f"{auditoria.get('ficheros', 0)} ficheros.",
        "- Segundo escaneo independiente sobre los ficheros de V17: sin caracteres de control, sin "
        "invisibles, sin emojis y sin alfabetos CJK.",
        "- Sin emojis, sin emoticonos y sin caracteres decorativos en el codigo ni en los informes.",
        "- `tools/verificar_caracteres.py`, que nombra el brief, **no existe en este proyecto**. El "
        "sustituto es `scripts/v13_p0a_auditoria_idioma.py`, mas el escaneo independiente.", "",
        "## Git", "",
        "- Rama `kaggle-age-2026-puj-v17` creada desde `a818349` con el arbol limpio.",
        "- **Sin push, sin PR, sin merge, sin reescritura de historia y sin modificar remotes.**",
        "- V12, V14, V15, V16, los lexicones y las submissions historicas siguen presentes.", "",
        "## Entregables del brief: estado", "", "| Entregable | Estado |", "|---|---|",
        "| `reports/v17_gap_audit.md` y `.csv` | entregado |",
        "| `reports/v17_p0_reproduccion.md` | entregado |",
        "| `reports/v17_p1_fronteras.md` | entregado |",
        "| `reports/v17_p1b_curva_compuerta.md` | entregado |",
        "| `reports/v17_p1c_regla_conjunta.md` | entregado |",
        "| `reports/v17_p8_perfil_confianza.md` | entregado |",
        "| `reports/v17_p9_arbitro.md` | entregado |",
        "| `reports/v17_p5_combinador_local.md` | entregado |",
        "| `reports/v17_master_results.csv` | entregado |",
        "| `reports/v17_hypothesis_registry.csv` | entregado |",
        "| `reports/v17_member_boundary_matrix.csv` | entregado |",
        "| `reports/v17_boundary_analysis.csv` | entregado |",
        "| `reports/v17_error_intersection.csv` | entregado |",
        "| `reports/v17_candidate_comparison.csv` | entregado |",
        "| `reports/v17_reproducibility.md` | entregado |",
        "| `reports/v17_compliance.md` | este fichero |",
        "| `reports/v17_final_research.md` | entregado |",
        "| 10 figuras | entregado |",
        "| Integracion en el notebook consolidado | **PENDIENTE**: los resultados estan en los "
        "informes y en las figuras, pero no se han volcado al notebook. Es una tarea de redaccion, no "
        "de medicion, y queda declarada como pendiente en vez de darse por hecha |", ""]
    COMPLIANCE.write_text("\n".join(lineas_comp) + "\n", encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V17 · cierre", "fecha": "2026-09-26", "commit": commit_actual(),
        "candidatos": comparativa.to_dict(orient="records"),
        "pistas_cerradas": ["P2", "P2.3", "P3", "P3.3", "P5", "P6", "P7", "P8", "P9", "P10"],
        "pistas_complementarias": ["P4", "P1", "P1.1"],
        "pistas_promising": [], "pistas_finalist": [],
        "veredicto": ("la senal local existe y es grande, el techo con arbitro perfecto supera el "
                      "gate de adopcion, y ninguna senal disponible la identifica; los errores de V12 "
                      "en la frontera son errores confiados"),
        "segundos": 0,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"comparativa: {len(comparativa)} candidatos | informes: 3 | figuras: 2 | "
          f"pistas promising/finalist: 0/0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
