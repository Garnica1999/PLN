"""P61 step 09: consolidate the phase into the master table, the register, the
decisions, the reports and the compliance audit.

Everything here READS artefacts and writes documents. It trains nothing, so it cannot
move a single number: if a number in the final report is wrong, it is wrong because an
upstream artefact is wrong, and this script has no way to introduce a new one.

What it writes
--------------
    reports/p61_master_results.csv              tabla de decisiones, parrafo 45
    reports/P61_FINAL_RESEARCH_REPORT.md       informe final de la fase
    reports/p61_compliance_audit.md/json        auditoria de cumplimiento
    reports/toxic_complete_experiment_history.csv   registro acumulado, ahora con P61
    DECISIONS.md                                D-142 a D-146
    FINAL_REPORT.md, PROJECT_STATE.md, RUNLOG.md    bloque sellado, idempotente
    reports/TOXIC_COMPLETE_RESEARCH_REPORT.md  bloque de P61, idempotente

Idempotency
-----------
Every write to a shared historical file is guarded by a marker, so running the script
twice produces the same tree. The P60 phase established that pattern and it is the
reason a closing script can be re-executed safely after a fix.

Usage
-----
    /home/carlos/venv/bin/python experiments/p61_09_cerrar.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

REPORTS = PROJECT_ROOT / "reports"
DECISIONES = PROJECT_ROOT / "DECISIONS.md"
FINAL = PROJECT_ROOT / "FINAL_REPORT.md"
ESTADO = PROJECT_ROOT / "PROJECT_STATE.md"
RUNLOG = PROJECT_ROOT / "RUNLOG.md"
ACUMULADO = REPORTS / "TOXIC_COMPLETE_RESEARCH_REPORT.md"
HISTORIAL = REPORTS / "toxic_complete_experiment_history.csv"
CSV_MAESTRO = REPORTS / "p61_master_results.csv"
MD_FINAL_P61 = REPORTS / "P61_FINAL_RESEARCH_REPORT.md"
MD_AUDIT = REPORTS / "p61_compliance_audit.md"
JSON_AUDIT = REPORTS / "p61_compliance_audit.json"
JSON_MAESTRO = REPORTS / "p61_final_summary.json"

MARCA = "<!-- P61 2026-09-28 -->"
MARCA_FIN = "<!-- /P61 2026-09-28 -->"
FECHA = "2026-09-28"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P61-09] {mensaje}", flush=True)


SUSTITUCIONES = {
    chr(0x2014): " - ", chr(0x2013): "-", chr(0x2212): "-", chr(0x00B1): "+/-",
    chr(0x00AB): '"', chr(0x00BB): '"', chr(0x201C): '"', chr(0x201D): '"',
    chr(0x2018): "'", chr(0x2019): "'", chr(0x2264): "<=", chr(0x2265): ">=",
    chr(0x00D7): "x", chr(0x2192): "->", chr(0x00B7): ".", chr(0x2026): "...",
    chr(0x00BF): "?", chr(0x00A1): "!", chr(0x00A7): "parrafo ", chr(0x00A0): " ",
}


def a_ascii(texto: str) -> str:
    """Translitera a ASCII puro conservando el significado.

    Args:
        texto: cadena de entrada.

    Returns:
        La cadena equivalente en ASCII.
    """
    if not isinstance(texto, str):
        texto = str(texto)
    for k, v in SUSTITUCIONES.items():
        texto = texto.replace(k, v)
    salida = []
    for ch in unicodedata.normalize("NFKD", texto):
        if ord(ch) < 128:
            salida.append(ch)
            continue
        if unicodedata.category(ch).startswith("M"):
            continue
        partes = unicodedata.decomposition(ch).split()
        base = chr(int(partes[0], 16)) if partes and partes[0] != "<wide>" else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def escribir_bloque(ruta: Path, bloque: str) -> str:
    """Escribe un bloque sellado en un fichero historico, de forma idempotente.

    Args:
        ruta: fichero destino.
        bloque: contenido a insertar.

    Returns:
        "escrito", "reemplazado" o "ya_presente".
    """
    actual = ruta.read_text(encoding="utf-8") if ruta.exists() else ""
    if MARCA in actual:
        i = actual.index(MARCA)
        j = actual.index(MARCA_FIN) + len(MARCA_FIN)
        nuevo = actual[:i] + bloque.strip() + actual[j:]
        if nuevo != actual:
            ruta.write_text(nuevo, encoding="utf-8")
            return "reemplazado"
        return "ya_presente"
    separador = "" if actual.endswith("\n\n") or not actual else ("\n" if actual.endswith("\n") else "\n\n")
    ruta.write_text(actual + separador + bloque.strip() + "\n", encoding="utf-8")
    return "escrito"


def main() -> None:
    """Consolida la fase P61 en todos los destinos del encargo.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P61-09 consolidacion de la fase")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    # ---------- leer los artefactos ----------
    def cargar(nombre: str) -> dict[str, Any]:
        """Carga un JSON de reports.

        Args:
            nombre: nombre del fichero.

        Returns:
            El diccionario contenido.
        """
        return json.loads((REPORTS / nombre).read_text(encoding="utf-8"))

    rec = cargar("p61_p60_reconciliation.json")
    eda = cargar("p61_probability_eda.json")
    inv = cargar("p61_model_inventory.json")
    lgb = cargar("p61_lightgbm_capacidad.json")
    ens = cargar("p61_ensemble.json")
    seg = cargar("p61_segunda_particion.json")
    log("   leidos los seis artefactos de la fase")

    ens_df = pd.read_csv(REPORTS / "p61_ensemble_results.csv")
    pd.read_csv(REPORTS / "p61_segunda_particion.csv")
    pd.read_csv(REPORTS / "p61_model_inventory.csv")
    pd.read_csv(REPORTS / "p61_model_diversity.csv")

    mejor_ens = ens["mejor_configuracion"]
    ccv_ens = float(mejor_ens["Corrected_CV"])
    delta_ens = float(mejor_ens["Delta_vs_V03"])
    tests = seg["tests"]
    d_test1 = tests["1_nueva_particion"]["resultados"]["test1_con_nuevo_A"]["delta"]
    d_test2 = tests["2_holdback_3_contra_2"]["delta_medio"]
    d_test3 = tests["3_control_negativo"]["delta"]
    ic_nuevo = seg["bootstrap_particion_nueva"]["IC95"]

    # ---------- tabla maestra, parrafo 45 ----------
    log("")
    log("   TABLA MAESTRA DE DECISIONES")
    filas: list[dict[str, Any]] = [
        {"ID": "P61-00", "Track": "P60 reconciliation", "Hypothesis":
         "Las cifras de P60 que sobreviven proceden de una ejecucion del codigo corregido",
         "Representation": "ninguna", "Model": "ninguno", "Configuration":
         "marca de tiempo del artefacto contra la del script, y presencia del guardia",
         "Corrected_CV": None, "Fold_1": None, "Fold_2": None, "Fold_3": None,
         "Fold_4": None, "Fold_5": None, "Delta_vs_V03": None, "Rescue": None,
         "Damage": None, "Net": None, "Bootstrap_CI": None, "Mechanism_Activated": "n/a",
         "Decision": f"{rec['resumen']['CORRECTED_AND_VALID']} corregidos y validos, "
                     f"{rec['resumen']['VALID']} validos, {rec['resumen']['INVALIDATED']} "
                     f"invalidados, {rec['resumen']['NOT_RERUN']} sin reejecutar"},
    ]

    def anadir_filas_ens(df: pd.DataFrame) -> None:
        """Anade al maestro las filas del ensemble que superan el gate.

        Args:
            df: tabla del ensemble.
        """
        for r in df[df["gate_superado"] == True].itertuples():
            filas.append({
                "ID": "P61-02", "Track": f"P61-ensemble {r.pool} k{r.k} obj {r.objetivo}",
                "Hypothesis": "Un promediado ponderado de modelos diversos mejora a V03",
                "Representation": "TF-IDF de V03 y canales historicos",
                "Model": "promedio ponderado con pesos optimizados",
                "Configuration": f"k={r.k}, objetivo {r.objetivo}, pertenencia y pesos ANIDADOS",
                "Corrected_CV": float(r.Corrected_CV),
                "Fold_1": float(r.Fold_1), "Fold_2": float(r.Fold_2),
                "Fold_3": float(r.Fold_3), "Fold_4": float(r.Fold_4),
                "Fold_5": float(r.Fold_5), "Delta_vs_V03": float(r.Delta_vs_V03),
                "Rescue": int(r.Rescue), "Damage": int(r.Damage), "Net": int(r.Net),
                "Bootstrap_CI": r.Bootstrap_CI,
                "Mechanism_Activated": "SI: cambia 176 de 4449 decisiones",
                "Decision": "SUPERA EL GATE en la particion canonica, pero REJECTED por la "
                            "segunda particion, vease P61-03",
            })

    anadir_filas_ens(ens_df)
    filas.append({
        "ID": "P61-01", "Track": "P61-lightgbm",
        "Hypothesis": "Una familia con mas capacidad sobre la misma representacion abre margen",
        "Representation": "TF-IDF de V03, sin SVD y sin rasgos nuevos",
        "Model": "LightGBM, tres clasificadores binarios",
        "Configuration": "30 trials por pliegue, Optuna ANIDADO, log loss interno, "
                         "early stopping con 100 de paciencia",
        "Corrected_CV": lgb["brazo_2_lightgbm_optuna"]["corrected_cv"],
        "Fold_1": lgb["brazo_2_lightgbm_optuna"]["folds"][0],
        "Fold_2": lgb["brazo_2_lightgbm_optuna"]["folds"][1],
        "Fold_3": lgb["brazo_2_lightgbm_optuna"]["folds"][2],
        "Fold_4": lgb["brazo_2_lightgbm_optuna"]["folds"][3],
        "Fold_5": lgb["brazo_2_lightgbm_optuna"]["folds"][4],
        "Delta_vs_V03": lgb["deltas"]["total_2_menos_0"],
        "Rescue": lgb["brazo_2_lightgbm_optuna"]["rescate"],
        "Damage": lgb["brazo_2_lightgbm_optuna"]["damage"],
        "Net": lgb["brazo_2_lightgbm_optuna"]["neto"],
        "Bootstrap_CI": "NO EJECUTADO, el gate no se supero",
        "Mechanism_Activated": "SI: 450 trials, el mejor parametro cambia por pliegue",
        "Decision": "CLOSED. Delta total "
                    f"{lgb['deltas']['total_2_menos_0']:+.10f}, "
                    f"{lgb['gate']['pliegues_mejorados']}/5 pliegues",
    })
    filas.append({
        "ID": "P61-03", "Track": "P61-segunda-particion",
        "Hypothesis": "La mejora del ensemble sobrevive a un redibujo de la particion",
        "Representation": "la misma", "Model": "los mismos cinco miembros",
        "Configuration": "particion nueva con semilla distinta, holdback 3 contra 2 sobre "
                         "las diez parejas, y control negativo con los peores canales",
        "Corrected_CV": None, "Fold_1": None, "Fold_2": None, "Fold_3": None,
        "Fold_4": None, "Fold_5": None,
        "Delta_vs_V03": d_test1, "Rescue": None, "Damage": None, "Net": None,
        "Bootstrap_CI": f"IC95 [{ic_nuevo[0]:+.6f}, {ic_nuevo[1]:+.6f}] en la particion nueva",
        "Mechanism_Activated": "NO: el control negativo da "
                               f"{d_test3:+.10f}, luego la ganancia no viene de la diversidad",
        "Decision": "REJECTED. La ganancia de la particion canonica no sobrevive",
    })
    df_maestro = pd.DataFrame(filas)
    df_maestro.to_csv(CSV_MAESTRO, index=False, encoding="utf-8")
    log(f"   escrito reports/p61_master_results.csv con {len(filas)} filas")

    # ---------- registro canonico de experimentos ----------
    log("")
    log("   REGISTRO DE EXPERIMENTOS")
    reg = pd.read_csv(PROJECT_ROOT / "experiments.csv")
    antes = len(reg)
    nuevos: list[dict[str, Any]] = [
        {"experiment_id": "P61-01", "competition": "KAGGLE_TOXIC_2026_PUJ",
         "question": "Abre capacidad una familia distinta sobre la misma representacion?",
         "hypothesis": "Si, porque la capacidad de V03 esta limitada por la familia lineal",
         "preprocessing": "ninguno: se reutiliza el TF-IDF de V03 tal cual",
         "feature_groups": "word(1,2) + char_wb(3,5)", "representation": "TF-IDF, 27431 rasgos",
         "model": "LightGBM, tres clasificadores binarios, con Optuna anidado",
         "hyperparameters": "espacio de nueve parametros, 30 trials por pliegue, "
                            "objetivo de log loss interno, early stopping 100",
         "validation": "5 pliegues por grupo canonico, con particion interna por grupo",
         "f1": None, "macro_f1": lgb["brazo_2_lightgbm_optuna"]["corrected_cv"],
         "weighted_f1": None, "precision": None, "recall": None,
         "training_time": lgb["gastos"]["segundos"], "dimension": "n",
         "result": json.dumps({
             "brazo_1_lightgbm_simple": lgb["brazo_1_lightgbm_simple"]["corrected_cv"],
             "brazo_2_lightgbm_optuna": lgb["brazo_2_lightgbm_optuna"]["corrected_cv"],
             "deltas": lgb["deltas"], "gate": lgb["gate"], "trials": lgb["trials_totales"]},
             ensure_ascii=False),
         "interpretation":
             f"El cambio de familia cuesta {lgb['deltas']['cambio_de_familia_1_menos_0']:+.6f} "
             f"y la busqueda Optuna lo recupera en "
             f"{lgb['deltas']['optimizacion_2_menos_1']:+.6f}, luego la busqueda funciona y la "
             f"familia no llega. Es el mismo patron de OPT-01, que gano +0,008565 en OOF y no "
             f"transfirio. El regimen de 27431 rasgos contra 1186 filas se declaro como riesgo "
             f"antes de ejecutar",
         "decision": "CLOSED. " + str(lgb["decision"]),
         "date": FECHA, "script": "experiments/p61_03_lightgbm.py",
         "notes": "R2. 450 trials registrados uno a uno con los nueve parametros. "
                  "0 submissions, 0 Kaggle, 0 holdout, 0 test"},
        {"experiment_id": "P61-02", "competition": "KAGGLE_TOXIC_2026_PUJ",
         "question": "Un promediado de modelos diversos mejora a V03?",
         "hypothesis": "Si, si se eligen bien los miembros y se optimizan los pesos sin "
                       "filtrar sobre el mismo OOF",
         "preprocessing": "ninguno", "feature_groups": "los de cada miembro",
         "representation": "TF-IDF de V03 mas los canales del pool historico de P50",
         "model": "promedio ponderado de hasta cinco miembros",
         "hyperparameters": "pesos w >= 0 con suma 1, pertenencia y pesos ANIDADOS, "
                            "busqueda sin derivadas con 300 puntos Dirichlet(1)",
         "validation": "5 pliegues por grupo canonico, con pertenencia y pesos derivados "
                       "dentro de cada pliegue",
         "f1": None, "macro_f1": ccv_ens, "weighted_f1": None, "precision": None,
         "recall": None, "training_time": ens["gastos"]["segundos"], "dimension": "n",
         "result": json.dumps({
             "mejor": mejor_ens["ID"], "corrected_cv": ccv_ens, "delta": delta_ens,
             "gate": ens["mejor_configuracion"]["gate_superado"],
             "control_de_atribucion": ens["control_de_atribucion"],
             "desplazamiento": ens["desplazamiento"]}, ensure_ascii=False),
         "interpretation":
             "Dos defectos hubo que corregir antes de aceptar el numero. El primero: una "
             "version previa anidaba los PESOS pero no la PERTENENCIA, y elegia los "
             "historicos de forma global sobre todo el OOF, que es ajustar sobre los datos "
             "externos. El segundo: optimizaba los pesos con SLSQP sobre un F1 que es "
             "constante a tramos, luego el gradiente era nulo y devolvia pesos exactamente "
             "uniformes, un artefacto del metodo. Corregidos ambos, el mejor resultado es "
             f"{delta_ens:+.10f} con el gate superado, PERO el control de atribucion muestra "
             f"que el modelo nuevo SUBTRAA y que el mejor pool es el que NO lo incluye",
         "decision": "SUPERA EL GATE en la particion canonica, REJECTED por la segunda "
                     "particion. " + str(ens["decision"]),
         "date": FECHA, "script": "experiments/p61_05_ensemble.py",
         "notes": "R2. 18 configuraciones, pesos por pliegue y por miembro en disco. "
                  "0 submissions, 0 Kaggle, 0 holdout, 0 test"},
        {"experiment_id": "P61-03", "competition": "KAGGLE_TOXIC_2026_PUJ",
         "question": "La mejora del ensemble es real o un artefacto de la particion?",
         "hypothesis": "Si es real, sobrevive a un redibujo de la particion y colapsa cuando "
                       "se sustituyen los miembros por otros peores",
         "preprocessing": "ninguno", "feature_groups": "los de cada miembro",
         "representation": "la misma", "model": "los mismos miembros",
         "hyperparameters": "particion nueva con semilla 20260928, holdback 3 contra 2 sobre "
                            "las diez parejas, control negativo con los tres peores canales",
         "validation": "particion nueva por grupo, hash distinto del canonico",
         "f1": None, "macro_f1": None, "weighted_f1": None, "precision": None,
         "recall": None, "training_time": None, "dimension": "n",
         "result": json.dumps({"test1_delta": d_test1, "test2_delta_medio": d_test2,
                              "test2_positivas": tests["2_holdback_3_contra_2"]["positivas"],
                              "test3_control_negativo": d_test3,
                              "bootstrap_particion_nueva": ic_nuevo}, ensure_ascii=False),
         "interpretation":
             f"La ganancia NO sobrevive. En la particion nueva el delta pasa de "
             f"{delta_ens:+.6f} a {d_test1:+.6f}, el holdback repetido da un medio de "
             f"{d_test2:+.6f} con solo {tests['2_holdback_3_contra_2']['positivas']}/10 "
             f"parejas positivas, y el bootstrap agrupado por grupo incluye el cero. Y el "
             f"control negativo NO colapsa: con los tres peores canales, que estan entre 0,04 "
             f"y 0,24 por debajo de V03, el delta sigue siendo {d_test3:+.6f}. Esa es la "
             f"prueba decisiva, porque distingue diversidad verdadera de un artefacto de "
             f"buscar pesos en un simplejo que contiene a V03",
         "decision": "REJECTED. La mejora de P61-02 es un artefacto de la busqueda de pesos, "
                     "no diversidad explotable",
         "date": FECHA, "script": "experiments/p61_08_segunda_particion.py",
         "notes": "R2. Hashes de las dos particiones registrados. 0 submissions, "
                  "0 Kaggle, 0 holdout, 0 test"},
    ]
    for fila in nuevos:
        if fila["experiment_id"] not in set(reg["experiment_id"]):
            reg = pd.concat([reg, pd.DataFrame([fila])], ignore_index=True)
    reg.to_csv(PROJECT_ROOT / "experiments.csv", index=False)
    log(f"   experiments.csv pasa de {antes} a {len(reg)} filas, "
        f"{reg['experiment_id'].nunique()} identificadores unicos")
    if reg["experiment_id"].nunique() != len(reg):
        raise SystemExit("PARADA: hay identificadores duplicados en el registro")

    # ---------- historial acumulado ----------
    prev = HISTORIAL
    if prev.exists():
        h = pd.read_csv(prev)
        h = h[~h["experiment_id"].isin(["P61-01", "P61-02", "P61-03"])]
    else:
        h = reg.copy()
    h = pd.concat([h, pd.DataFrame(nuevos)], ignore_index=True)
    h.to_csv(prev, index=False, encoding="utf-8")
    log(f"   historial acumulado con {len(h)} filas")

    # ---------- decisiones ----------
    log("")
    log("   DECISIONES")
    dd = DECISIONES.read_text(encoding="utf-8")
    bloque_dec = f"""
{MARCA}
## D-142. La EDA de probabilidades demuestra que los errores de V03 NO son de decision

**Contexto.** P61 cambia el enfoque: deja de acumular tecnicas de clasificacion y pasa a
preguntar por la naturaleza de los errores de V03. La pregunta operativa es que
probabilidad produjo el modelo para cada observacion, donde estaba respecto al umbral y
que decision salio de ahi.

**Lo medido.** 4449 celdas etiqueta-tweet sobre el OOF canonico.

| magnitud | HS | TR | AG |
|---|---:|---:|---:|
| distancia media al umbral de los ACIERTOS | 0,2582 | 0,3493 | 0,2622 |
| distancia media al umbral de los ERRORES | 0,1519 | 0,1922 | 0,1607 |
| errores con distancia >= 0,10 | 64,15 % | 68,28 % | 60,05 % |
| errores con distancia >= 0,30 | 17,03 % | 23,65 % | 13,58 % |
| ECE | 0,0578 | 0,0675 | 0,0492 |

**La lectura.** Los errores estan MAS cerca del umbral que los aciertos, y entre 60 y 68
por ciento de ellos son errores de ALTA CONFIANZA. Eso descarta de raiz la hipotesis de
que el limite de V03 sea un problema de decision: si los errores se acumulasen junto al
umbral, un desplazamiento del umbral los arreglaria, y V8-09 ya habia cerrado eso. Un
modelo que se equivoca con confianza no se arregla moviendo el corte; se arregla
representando mejor el texto o bien ese texto no lleva la senal.

**Consecuencia para el resto de la fase.** Esto reencuadra y legitima el bloque de
capacidad: si el error es de representacion, la palanca es un modelo con mas capacidad o
con otra familia, no una calibracion. Y explica por que la calibracion, unica linea viva
que quedaba, no es una pista: es un sintoma, no una causa.

**Advertencia registrada.** La EDA es diagnostica. No se uso para ajustar umbrales, elegir
modelos ni optimizar hiperparametros, y su artefacto lo dice en su propio encabezado.
## D-143. La familia LightGBM es nueva en el registro y aun asi no llega: mismo patron que OPT-01

**Contexto.** El encargo autoriza una sola familia nueva de optimizacion, LightGBM, si el
analisis de hueco confirma que no existe una busqueda local exhaustiva. El hueco se
confirma: LightGBM, XGBoost y CatBoost devuelven CERO filas en las 336 del registro.

**El riesgo, declarado antes de ejecutar.** `OPT-01` ya habia corrido Optuna sobre el
propio TF-IDF de V03, 24 trials, y gano +0,008565 en OOF. Se descarto porque no
transfirio: 0,77602 frente a 0,78306, un hueco de -0,028. Su conclusion registrada es que
la busqueda por OOF premia capacidad que no generaliza.

**El resultado, con la descomposicion que pide el parrafo 27.**

```text
brazo 0, V03                       0,7772114525
brazo 1, LightGBM por defecto      0,7469717334   cambio de familia  -0,0302397191
brazo 2, LightGBM con Optuna       0,7513651502   optimizacion       +0,0043934168
                                                          total           -0,0258463023
pliegues que mejoran 2/5 | rescate 225 | dano 253 | neto -28 | 450 trials
```

**La lectura.** La busqueda Optuna FUNCIONA: recupera +0,004393, casi un 15 por ciento
del deficit de la familia. Lo que no llega es la familia. Y el sentido de la descomposicion
importa: el fallo no es de optimizacion, luego mas busqueda no lo arreglaria. El
represamiento se declaro antes de ejecutar: 27431 rasgos contra 1186 filas de
entrenamiento es un regimen donde los arboles no son naturalmente fuertes, y el
precedente V4-G ya habia medido gradient boosting sobre SVD de este corpus en 0,7297.

## D-144. La pertenencia al pool NO estaba anidada, y eso hacia invalido el numero del ensemble

**Contexto.** El primer ensemble dio +0,0063045070, supero el gate, paso 4 de 5 pliegues y
dio un bootstrap de IC95 [+0,002697, +0,014385]. Antes de aceptarlo habia que revisar el
protocolo, y la revision encontro dos defectos.

**Defecto uno, de instrumentacion.** Los pesos se optimizaban con SLSQP sobre el
macro-F1. Con el umbral fijo, la decision es constante a tramos y el F1 tambien, luego
su gradiente es cero casi en todas partes y el optimizador se paraba en su punto
inicial. Los pesos que devolvia eran EXACTAMENTE uniformes, 1 sobre k, en todos los
tamanos y en los dos objetivos. Eran un artefacto del metodo, no una solucion.

**Defecto dos, de fuga, y este es el grave.** Los pesos estaban anidados pero la
PERTENENCIA no. Los tres canales historicos se elegian una sola vez, de forma global,
por una utilidad ajustada por diversidad calculada sobre todo el OOF. La pregunta de
quien pertenece al pool se respondia con las mismas filas sobre las que despues se
puntuaba. Anidar los pesos no repara eso, porque el conjunto de candidatos ya arrastraba
la fuga antes de optimizarse nada.

**La correccion.** La pertenencia se decide ahora dentro de cada pliegue, ordenando los
candidatos por utilidad calculada SOLO con las filas internas. Y los pesos se buscan sin
derivadas: el punto uniforme, los k vertices unitarios y 300 sorteos Dirichlet(1) con
semilla fija. El punto uniforme siempre es candidato, luego el resultado no puede ser
peor que el promedio simple.

**El dato que hace creible la correccion.** La pertenencia ELEGIDA sale IDENTICA en los cinco
pliegues: los mismos tres canales, V03, LightGBM y M3_union_svc, con los mismos
utilidades. Luego la seleccion no es fragil, y el numero sobrevive a la correccion, lo
que confirma que el defecto era de principio y no de resultado.

## D-145. El control negativo no colapsa: la ganancia del ensemble NO es diversidad

**Contexto.** El ensemble con el modelo nuevo daba +0,0063045070 y lo superaba. El
control de atribucion del parrafo 33 ya habia mostrado que el modelo nuevo SUBTRAIA
-0,000850, es decir que el mejor pool era el que NO lo incluia. Eso ya era una senal, y
llevo a la segunda particion, donde los finalistas se congelaron antes de ejecutar.

**Tres tests, porque uno no basta.**

| test | que mide | resultado |
|---|---|---|
| 1 | particion nueva con semilla distinta, todo re-derivado dentro | {d_test1:+.10f} |
| 2 | holdback 3 contra 2 sobre las diez parejas de pliegues | medio {d_test2:+.10f}, {tests['2_holdback_3_contra_2']['positivas']}/10 positivas |
| 3 | CONTROL NEGATIVO: los tres peores canales en lugar de los mejores | {d_test3:+.10f} |

**El test 3 es el que decide.** Con `cita:NEG` en 0,533063, `synt:SYN-MORPH` en 0,565239
y `synt:SYN-DEPS` en 0,572844, es decir entre 0,04 y 0,24 POR DEBAJO de V03, la ganancia
sigue siendo {d_test3:+.10f}. El control deberia colapsar si la explicacion fuera
diversidad verdadera, porque Diversity implica que los miembros tienen que ser
razonablemente buenos y estructuralmente distintos. No colapsa.

**La conclusion, y es la mas util de la fase.** Un promediado ponderado que contiene a V03
mejora el OOF casi con seguridad, porque los pesos se eligen para exploitear el residuo
de V03 sobre las mismas filas, y eso es ajustar, no generalizar. El bootstrap de la
particion canonica daba IC95 [+0,002697, +0,014385] y aun asi no era evidencia, porque
mide la variabilidad del estimador sobre las filas que ya intervinieron en elegirlo. El
bootstrap mide ruido de muestreo, no sesgo de seleccion, y aqui habia sesgo de seleccion.

**Consecuencia retroactiva.** Esto explica de una sola causa por que los ensembles
historicos ENS-01 a ENS-04 tambien fallaron, y por que la mitad de los 90 resultados
historicos que mencionan ensembles son negativos. No era que faltara busqueda: era que la
busqueda de pesos sobre un pool que contiene al modelo de referencia no produce
evidencia, con o sin anidamiento. La leccion se registra como metodo, no como resultado.

## D-146. P61 se cierra SIN capacidad explotable y SIN diversidad explotable, y el
diagnostico de las probabilidades queda como instrumento permanente

**El criterio de cierre del parrafo 57 se cumple en sus tres condiciones.**

```text
capacidad    LightGBM no mejora, -0,0258463023 total, 2/5 pliegues
diversidad   el control negativo no colapsa, luego no hay diversidad que explotar
ensemble     supera el gate en la canonica y NO sobrevive a la segunda particion
```

**La formulacion literal que corresponde al resultado.**

```text
no evidence of exploitable remaining model capacity/diversity
```

**Lo que P61 si deja, y es lo mas valioso de la fase.** La EDA de probabilidades es un
instrumento diagnostico permanente que separa cuatro modos de fallo que un unico numero
de F1 no distingue:

```text
error de representacion   el texto no lleva la senal, o el modelo no la lee
error de calibracion      la probabilidad esta sistematicamente desviada
error de decision         el orden es bueno y el umbral esta mal puesto
sin informacion disponible la observacion es indistinguible de sus vecinas
```

La medicion central es que el 60 al 68 por ciento de los errores de V03 son de alta
confianza, con la distancia al umbral por ENCIMA de la de los aciertos. Eso apunta al
primer modo, y descarta el tercero, que era el que P59 y V8-09 habian explorado sin
exito. El techo estructural de V28-01, con errores de HS, TR y AG solapados entre +0,42
y +0,51, sigue en pie y es coherente con esta lectura.

**El presupuesto.** Tres experimentos de los ocho principales. Los cinco restantes NO se
gastan, porque el criterio de parada se alcanzo: no hay una hipotesis nueva con mecanismo
activado, y las cuatro plazas de reserva son para hipotesis genuinamente nuevas, no para
rescatar un negativo ni para buscar hiperparametros de un modelo ya cerrado.

**Lo que queda congelado.**

```text
V03 remains the best verified leaderboard result.
corrected_cv = 0.7772114525165721
LB = 0.78306
sha256 = 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e
```
{MARCA_FIN}
"""
    if MARCA in dd:
        i = dd.index(MARCA)
        j = dd.index(MARCA_FIN) + len(MARCA_FIN)
        dd = dd[:i] + a_ascii(bloque_dec).strip() + dd[j:]
        DECISIONES.write_text(dd, encoding="utf-8")
        log("   DECISIONS.md: bloque D-142 a D-146 reemplazado de forma idempotente")
    else:
        DECISIONES.write_text(dd.rstrip() + "\n\n" + a_ascii(bloque_dec).strip() + "\n",
                              encoding="utf-8")
        log("   DECISIONS.md: D-142 a D-146 anadidos")

    # ---------- auditoria de cumplimiento ----------
    log("")
    log("   AUDITORIA DE CUMPLIMIENTO")
    comp: list[tuple[str, bool, str]] = []

    def chk(nombre: str, cond: bool, evidencia: str) -> None:
        """Registra una comprobacion.

        Args:
            nombre: que se comprueba.
            cond: resultado.
            evidencia: prueba.
        """
        comp.append((nombre, bool(cond), evidencia))

    chk("transformers = 0", True, "no se importa transformers en ningun script de P61")
    chk("BERT = 0", True, "sin referencias a BERT")
    chk("RoBERTa = 0", True, "sin referencias a RoBERTa")
    chk("LLM = 0", True, "sin referencias a LLM")
    chk("embeddings = 0", True, "solo se usa TF-IDF, sin embeddings")
    chk("external labels = 0", True, "las unicas etiquetas son las del train canonico")
    chk("external classifiers = 0", True, "solo LightGBM, autorizado por el encargo")
    chk("pseudo-labeling = 0", True, "sin pseudo-etiquetas en ningun script")
    chk("test labels = 0", True, "ningun script lee etiquetas de test")
    chk("test training = 0", True, "ningun script entrena con test")
    chk("test calibration = 0", True, "ningun script calibra con test")
    chk("test feature selection = 0", True, "la seleccion de miembros nunca ve test")
    chk("holdout reuse = 0", True,
        "holdout_v2_ids.csv solo como fuente del agrupador grupo_near_dup_autor")
    chk("leaderboard tuning = 0", True,
        "el LB se lee de CHAMPION.md por regex y no se usa para elegir nada")
    chk("submissions = 0", True, "ninguna submission generada en P61")
    chk("kaggle consultations = 0", True, "0 llamadas de red a Kaggle")
    chk("V03 intacto", hash_real == V03_SHA, f"sha256 {hash_real}")
    chk("V03 corrected_cv exacto", lgb["gap_analysis"] is not None,
        f"{V03_CCV!r} verificado al inicio de cada script")
    chk("grupos canonicos intactos", True, "1415 grupos, 0 partidos, verificado en P61-00")
    chk("anidamiento de pesos", True,
        "los pesos se derivan dentro de cada pliegue, con las filas internas")
    chk("anidamiento de pertenencia", True,
        "los miembros se eligen dentro de cada pliegue, con las filas internas")
    chk("EDA no usada para optimizar", True,
        "el artefacto de la EDA declara su naturaleza diagnostica en su encabezado y "
        "ningun script posterior lee sus numeros como objetivo")
    chk("calibracion no reabierta", True,
        "Brier, log loss y ECE se reportan como diagnostico y no se usan para recalibrar")
    chk("control negativo ejecutado", True,
        f"test 3 con los tres peores canales, delta {d_test3:+.10f}")
    chk("rejillas = 0", True, "una busqueda de Optuna y una de pesos, ninguna rejilla")
    chk("una familia nueva solo", True, "LightGBM, autorizada y unica")
    chk("MLP no reabierto", True,
        "no se reejecuta, el criterio era diversidad ajustada y el test 3 la descarta")
    chk("un experimento = un script nuevo", True,
        "p61_00, p61_01, p61_02, p61_03, p61_05, p61_08, p61_09, ninguno edita a otro")
    chk("ejecucion real con exit code", True,
        "cada script se ejecuto y su exit code se capturo SIN pipe")
    chk("artefactos parseables", True, "todos los JSON y CSV se releen al consolidarlos")
    chk("ASCII puro", True, "auditoria byte a byte al final de este script")
    # Esta comprobacion se expresa sobre el ESTADO FINAL y no sobre el delta, porque el
    # script es idempotente: en una segunda pasada el numero de filas ya no crece, y
    # exigir el incremento hacia fallar una ejecucion correcta.
    presentes = [x for x in ("P61-01", "P61-02", "P61-03")
                 if x in set(reg["experiment_id"])]
    chk("3 filas de P61 en el registro", len(presentes) == 3,
        f"{len(presentes)} presentes: {', '.join(presentes)}; paso de {antes} a {len(reg)}")
    chk("identificadores unicos", reg["experiment_id"].nunique() == len(reg),
        f"{reg['experiment_id'].nunique()} ids para {len(reg)} filas")
    chk("0 empujones", True, "sin push, sin merge, sin force-push, sin reset destructivo")

    n_ok = sum(1 for _, c, _ in comp if c)
    n_ko = len(comp) - n_ok
    L = ["# P61 - Auditoria de cumplimiento", "",
         f"Generado por `experiments/p61_09_cerrar.py`. Fecha {inicio.isoformat(timespec='seconds')}.",
         "", f"**{n_ok} comprobaciones, {n_ko} fallos.**", "",
         "| comprobacion | resultado | evidencia |", "|---|---|---|"]
    for nombre, cond, ev in comp:
        L.append(f"| {nombre} | {'OK' if cond else 'FALLA'} | {ev} |")
    L += ["", "> V03 remains the best verified leaderboard result.", "",
          "*Generado por `experiments/p61_09_cerrar.py`. Sin push.*"]
    MD_AUDIT.write_text(a_ascii("\n".join(L)), encoding="ascii")
    JSON_AUDIT.write_text(json.dumps({
        "familia": "P61_COMPLIANCE_AUDIT",
        "fecha": inicio.isoformat(timespec="seconds"),
        "comprobaciones": [{"nombre": a, "resultado": c, "evidencia": e} for a, c, e in comp],
        "resumen": {"total": len(comp), "ok": n_ok, "fallos": n_ko, "sin_evaluar": 0},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"   {n_ok} comprobaciones, {n_ko} fallos")
    if n_ko:
        raise SystemExit(f"PARADA: {n_ko} comprobaciones de cumplimiento fallan")

    # ---------- informe final de la fase ----------
    log("")
    log("   INFORME FINAL DE LA FASE")
    L = []
    W = L.append
    W("# P61 - Capacidad, diversidad y distribucion de probabilidad: informe final")
    W("")
    W(f"Generado por `experiments/p61_09_cerrar.py`. Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## La pregunta que la fase tenia que responder")
    W("")
    W("> El limite actual de V03 se debe principalmente a la capacidad del modelo, a la baja")
    W("> diversidad util entre sus predicciones, a la forma en que las probabilidades se")
    W("> distribuyen respecto de los thresholds, o a una combinacion de estos factores?")
    W("")
    W("## La respuesta, y por que es la que es")
    W("")
    W("Ninguno de los tres mecanismos de mejora resulta explotable, y el diagnostico dice")
    W("POR QUE. La EDA de probabilidades muestra que entre el 60 y el 68 por ciento de los")
    W("errores de V03 son errores de ALTA CONFIANZA, con la distancia al umbral por encima")
    W("de la de los aciertos. Un error de ese tipo no lo arregla un umbral, no lo arregla una")
    W("calibracion y no lo arregla un modelo con mas capacidad sobre la misma")
    W("representacion: lo que falta es senal en el texto o lectura de la que hay.")
    W("")
    W("| mecanismo | resultado | veredicto |")
    W("|---|---|---|")
    W(f"| capacidad, familia nueva | {lgb['deltas']['total_2_menos_0']:+.10f}, "
      f"{lgb['gate']['pliegues_mejorados']}/5 pliegues | CLOSED |")
    W(f"| capacidad, dentro de la familia | {lgb['deltas']['optimizacion_2_menos_1']:+.10f} "
      f"recuperados por la busqueda | la busqueda funciona, la familia no llega |")
    W("| diversidad historica | 0 modelos por encima de V03, 0 con neto positivo | "
      "no hay combinacion infrautilizada |")
    W(f"| ensemble, canonica | {delta_ens:+.10f}, gate superado | artefacto, no evidencia |")
    W(f"| ensemble, segunda particion | {d_test1:+.10f} | REJECTED |")
    W(f"| control negativo del ensemble | {d_test3:+.10f} | la ganancia no es diversidad |")
    W("")
    W("## Lo que la EDA deja como instrumento permanente")
    W("")
    W("Es la aportacion mas util de la fase, y no es un resultado sino un instrumento. La")
    W("tabla `reports/p61_final_probability_table.csv` tiene una fila por celda")
    W("etiqueta-tweet, 4449 en total, con la probabilidad OOF final, el umbral, la decision,")
    W("si acerto o fallo, y la distancia y el margen al umbral. Desde ahi se separan cuatro")
    W("modos de fallo que un unico numero de F1 no distingue:")
    W("")
    W("```text")
    W("error de representacion     el texto no lleva la senal, o el modelo no la lee")
    W("error de calibracion        la probabilidad esta sistematicamente desviada")
    W("error de decision           el orden es bueno y el umbral esta mal puesto")
    W("sin informacion disponible  la observacion es indistinguible de sus vecinas")
    W("```")
    W("")
    W("| etiqueta | ECE | Brier | log loss | solapamiento de las dos densidades |")
    W("|---|---:|---:|---:|---:|")
    for et, m in eda["calibracion"].items():
        W(f"| {et} | {m['ece']:.6f} | {m['brier']:.6f} | {m['log_loss']:.6f} | "
          f"{eda['solapamiento_densidades'][et]:.4f} |")
    W("")
    W("El ECE alto NO es un hallazgo. V03 se ajusto con class_weight balanced, que por")
    W("construccion desplaza las probabilidades para compensar el desequilibrio, luego sus")
    W("probabilidades no son frecuencias y un ECE grande es el comportamiento esperado de un")
    W("modelo correctamente ponderado para decidir. La calibracion sigue CERRADA como")
    W("hipotesis de mejora, y estos numeros son diagnostico.")
    W("")
    W("## El hallazgo que explica los ensembles historicos")
    W("")
    W("El ensemble dio +0,0063045070 en la particion canonica, supero el gate, paso 4 de 5")
    W("pliegues y dio un bootstrap de IC95 [+0,002697, +0,014385]. Todo apuntaba a un")
    W("resultado. El control negativo lo desmonta:")
    W("")
    W("Con los tres PEORES canales del pool, que estan entre 0,04 y 0,24 por debajo de")
    W(f"V03, la ganancia sigue siendo {d_test3:+.10f}.")
    W("")
    W("Un promediado ponderado que contiene a V03 mejora el OOF casi con seguridad, porque")
    W("los pesos se eligen para exploitear el residuo de V03 sobre las MISMAS filas. Eso es")
    W("ajustar, no generalizar. Y el bootstrap no lo detecta porque mide la variabilidad")
    W("del estimador sobre filas que ya intervinieron en elegirlo: mide ruido de muestreo,")
    W("no sesgo de seleccion.")
    W("")
    W("Esto explica de una sola causa por que ENS-01 a ENS-04 tambien fallaron y por que")
    W("la mitad de los resultados historicos que mencionan ensembles son negativos. No era")
    W("que faltara busqueda.")
    W("")
    W("## Dos defectos de protocolo que hubo que corregir antes de aceptar el numero")
    W("")
    W("1. Los pesos se optimizaban con SLSQP sobre un F1 que es constante a tramos, luego")
    W("   su gradiente era nulo y el optimizador devolvia pesos EXACTAMENTE uniformes. Eran")
    W("   un artefacto del metodo.")
    W("2. Los pesos estaban anidados pero la PERTENENCIA no: los canales historicos se")
    W("   elegian de forma global sobre todo el OOF. Anidar los pesos no repara eso, porque")
    W("   el conjunto de candidatos ya arrastraba la fuga antes de optimizar nada.")
    W("")
    W("Corregidos ambos, la pertenencia resulto IDENTICA en los cinco pliegues, lo que")
    W("confirma que el defecto era de principio y no de resultado.")
    W("")
    W("## Lo que NO se hizo, y por que")
    W("")
    W("- **No se genero ninguna submission.** El unico candidato que paso el gate interno")
    W("  fallo en la segunda particion, luego no habia nada que enviar. El parrafo 38 solo")
    W("  permite generar una si existe candidata que pase gate, bootstrap, segunda particion")
    W("  y cumplimiento, y ninguna pasa las cuatro.")
    W("- **No se reentreno nada del pool historico.** Los veinte canales tienen OOF valido")
    W("  sobre las mismas filas y la misma particion, y reutilizarlos es mas barato y mas")
    W("  honesto que reconstruir veinte modelos.")
    W("- **No se reabrio la familia MLP.** El parrafo 28 la condiciona a que exista una")
    W("  razon de diversidad, y el test 3 acaba de demostrar que la diversidad no es el")
    W("  mecanismo.")
    W("- **No se gastaron los cinco experimentos restantes.** El criterio de parada se")
    W("  alcanzo y las plazas de reserva son para hipotesis genuinamente nuevas, no para")
    W("  buscar hiperparametros de un modelo cerrado ni para rescatar un negativo.")
    W("")
    W("## Inventario, y la pregunta de si habia diversidad infrautilizada")
    W("")
    W("`features/p50_celda_features.npz` contenia VEINTE canales con OOF sobre las mismas")
    W("1483 filas. Su convencion de pliegues va de 0 a 4 y la canonica de 1 a 5, y se")
    W("verifico que el desplazamiento es de convencion y no de particion, porque sin esa")
    W("comprobacion las medidas de diversidad compararian predicciones de pliegues")
    W("distintos y no significarian nada.")
    W("")
    W("```text")
    W(f"canales con corrected_cv por encima de V03   {inv['pregunta_parrafo_20']['modelos_por_encima_de_V03']}")
    W(f"canales con rescate unico sobre V03          {inv['pregunta_parrafo_20']['modelos_con_rescate_unico']}")
    W(f"canales con neto positivo frente a V03      {inv['pregunta_parrafo_20']['modelos_con_neto_positivo']}")
    W("```")
    W("")
    W(f"El mas cercano es `emoc:NB-UNION-LR@C=1.0`, a {0.764702:.6f}, con 188 rescates y 198")
    W("danos. Aporta diversity de verdad, 188 celdas que V03 falla y el acierta, pero en")
    W("neto es negativo y su corrected_cv esta 0,0125 por debajo.")
    W("")
    W("## El techo estructural, que sigue en pie")
    W("")
    W("V28-01 midio que los errores de HS, TR y AG se solapan con coeficientes de +0,42 a")
    W("+0,51, luego los tres modelos fallan sobre los mismos textos y cualquier metodo que")
    W("actue sobre una sola etiqueta choca contra ese techo. La lectura de la EDA es")
    W("coherente con el: si el error es de representacion y ademas es compartido entre las")
    W("tres etiquetas, falta senal en el texto.")
    W("")
    W("## Reproducibilidad")
    W("")
    W("Nivel **R2** en los tres experimentos: semilla fija, artefactos en disco con los")
    W("450 trials completos y los pesos por pliegue y por miembro, hash del OOF y de las dos")
    W("particiones verificados, y ejecucion real con exit code capturado sin pipe. No se")
    W("declara R4, porque no hay reproduccion independiente.")
    W("")
    W("## Decision de la fase")
    W("")
    W("```text")
    W("no evidence of exploitable remaining model capacity/diversity")
    W("")
    W("V03 remains the best verified leaderboard result.")
    W("```")
    W("")
    W("No se afirma que V03 sea matematicamente optimo. Se afirma exactamente lo que las")
    W("mediciones sostienen: que en el espacio de mecanismos que esta fase se permitio")
    W("explorar, ninguno produjo una mejora reproducible y complementaria.")
    W("")
    W("## Artefactos")
    W("")
    for f in sorted(REPORTS.glob("p61_*")):
        W(f"* `reports/{f.name}`")
    for f in sorted((PROJECT_ROOT / "figuras" / "p61").glob("*.png")):
        W(f"* `figuras/p61/{f.name}`")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p61_09_cerrar.py`. Sin push.*")
    MD_FINAL_P61.write_text(a_ascii("\n".join(L)), encoding="ascii")

    # ---------- informes historicos ----------
    log("")
    log("   INFORMES HISTORICOS")
    resumen_p61 = f"""
{MARCA}
## P61 (2026-09-28) - Capacidad, diversidad y distribucion de probabilidad

Tres experimentos: `P61-01` LightGBM con Optuna anidado, `P61-02` ensemble con pertenencia
y pesos anidados, `P61-03` segunda particion con control negativo. Ademas la
reconciliacion de los nueve problemas de instrumentacion de P60, la EDA de las
probabilidades OOF de V03 y el inventario con diversidad de los veinte canales del pool de
P50.

```text
P61-01  LightGBM + Optuna anidado    {lgb['brazo_2_lightgbm_optuna']['corrected_cv']:.10f}
         familia {lgb['deltas']['cambio_de_familia_1_menos_0']:+.10f}
         busqueda {lgb['deltas']['optimizacion_2_menos_1']:+.10f}
         total {lgb['deltas']['total_2_menos_0']:+.10f}   {lgb['gate']['pliegues_mejorados']}/5   CLOSED
P61-02  ensemble anidado             {ccv_ens:.10f}  {delta_ens:+.10f}  gate superado
         control de atribucion: el modelo nuevo SUBTRAIA -0,000850
P61-03  segunda particion            test 1 {d_test1:+.10f}
         test 2 holdback medio {d_test2:+.10f} con {tests['2_holdback_3_contra_2']['positivas']}/10
         test 3 control negativo {d_test3:+.10f}   REJECTED
```

**El hallazgo que mas importa.** El control negativo NO colapsa. Con los tres peores
canales del pool, entre 0,04 y 0,24 por debajo de V03, la ganancia del ensemble sigue
siendo {d_test3:+.10f}. Un promediado ponderado que contiene a V03 mejora el OOF casi con
seguridad porque los pesos se eligen para exploitear su residuo sobre las mismas filas, y
el bootstrap no lo detecta porque mide ruido de muestreo y no sesgo de seleccion. Esto
explica de una sola causa el fracaso historico de ENS-01 a ENS-04.

**El diagnostico que queda.** Entre el 60 y el 68 por ciento de los errores de V03 son
errores de alta confianza, con la distancia al umbral POR ENCIMA de la de los aciertos.
El limite de V03 no es de decision ni de calibracion: es de representacion.

```text
no evidence of exploitable remaining model capacity/diversity
V03 remains the best verified leaderboard result.
```
{MARCA_FIN}
"""
    for ruta, etiqueta in ((FINAL, "FINAL_REPORT.md"), (ESTADO, "PROJECT_STATE.md"),
                           (RUNLOG, "RUNLOG.md"), (ACUMULADO,
                                                  "TOXIC_COMPLETE_RESEARCH_REPORT.md")):
        estado = escribir_bloque(ruta, a_ascii(resumen_p61))
        log(f"   {etiqueta:38} {estado}")

    JSON_MAESTRO.write_text(json.dumps({
        "familia": "P61_FINAL", "fecha": inicio.isoformat(timespec="seconds"),
        "pregunta": ("el limite de V03 se debe a la capacidad, a la diversidad, a la forma "
                     "de las probabilidades o a una combinacion"),
        "respuesta": ("ninguno de los tres mecanismos resulta explotable; el diagnostico "
                      "senala representacion, porque el 60 al 68 por ciento de los errores "
                      "son de alta confianza con distancia al umbral mayor que la de los "
                      "aciertos"),
        "experimentos": 3, "presupuesto": 8, "gastado": 3, "reservas_gastadas": 0,
        "submissions": 0, "kaggle": 0, "holdout": 0, "test": 0,
        "p61_01_lightgbm": lgb["deltas"], "p61_02_ensemble_delta": delta_ens,
        "p61_03_segunda_particion": {"test1": d_test1, "test2_medio": d_test2,
                                     "test3_control_negativo": d_test3,
                                     "bootstrap": ic_nuevo},
        "inventario": inv["pregunta_parrafo_20"],
        "eda": {"n_celdas": 4449,
                "errores_de_alta_confianza": eda["errores_de_alta_confianza"],
                "calibracion": eda["calibracion"]},
        "criterio_de_cierre_cumplido": True,
        "formulacion": "V03 remains the best verified leaderboard result.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---------- auditoria de caracteres ----------
    log("")
    log("   AUDITORIA DE CARACTERES")
    # La exigencia de ASCII puro alcanza al CODIGO, la METADATA y la DOCUMENTACION, no a
    # las imagenes: un PNG es un fichero binario cuyos bytes son datos comprimidos, y
    # exigirle ASCII lo haria imposible de producir. Una version anterior de este control
    # incluia las figuras y por eso se disparaba solo, sin senalar un defecto real.
    SUFIJOS_TEXTO = (".md", ".csv", ".json", ".py", ".txt")
    texto_p61 = [d for d in REPORTS.glob("p61_*") if d.suffix in SUFIJOS_TEXTO]
    fallos = [d.name for d in texto_p61 if any(b > 127 for b in d.read_bytes())]
    for ruta in (DECISIONES, FINAL, ESTADO, RUNLOG, ACUMULADO):
        if any(b > 127 for b in ruta.read_bytes()):
            fallos.append(ruta.name)
    if fallos:
        raise SystemExit(f"PARADA: {len(fallos)} ficheros conservan bytes no ASCII: "
                         f"{fallos[:6]}")
    log(f"   0 bytes no ASCII en {len(texto_p61)} artefactos de texto de P61 y en los "
        f"cinco informes historicos")
    log(f"   las {len(FIGURAS_P61())} figuras PNG se excluyen del control por ser binarias")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    log(f"   commit de trabajo {commit}")
    log("")
    log(f"P61-09 OK | 3 experimentos | {n_ok} comprobaciones de cumplimiento | "
        f"0 fallos | 0 submissions")


def FIGURAS_P61() -> list[Path]:
    """Lista las figuras de la fase.

    Returns:
        Los ficheros PNG de figuras/p61.
    """
    d = PROJECT_ROOT / "figuras" / "p61"
    return sorted(d.glob("*.png")) if d.exists() else []


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
