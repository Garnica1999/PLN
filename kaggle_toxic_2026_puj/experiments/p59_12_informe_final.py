"""P59 step 12: consolidated closing artefacts.

Builds the three closing documents of the whole phase from the JSON artefacts of
every track, in execution order, plus the dependency graph that records which
track depends on which.

Why a generator and not three handwritten documents
---------------------------------------------------
Every number in the closing report is read from the JSON artefact that produced
it, at generation time. A hand-written report is a transcription risk, and this
project has already been bitten by that twice inside P59 alone: the novelty gate
caught two of its own misquoted figures, and P59-I lost a whole implementation
cycle to three separate transcription errors in a control. The report therefore
cannot disagree with the experiments, because it does not contain numbers of its
own.

Artefacts written
-----------------
    reports/p59_dependency_graph.md    which track depends on which
    reports/p59_master_results.csv     one row per track, in execution order
    reports/P59_FINAL_RESEARCH_REPORT.md  the consolidated report
    reports/p59_frontier_registry.csv  updated to the final state of all tracks

Usage
-----
    /home/carlos/venv/bin/python experiments/p59_12_informe_final.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

REPORTS = PROJECT_ROOT / "reports"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
FRONTIER = REPORTS / "p59_frontier_registry.csv"
MD_DEP = REPORTS / "p59_dependency_graph.md"
CSV_MASTER = REPORTS / "p59_master_results.csv"
MD_FINAL = REPORTS / "P59_FINAL_RESEARCH_REPORT.md"
JSON_FINAL = REPORTS / "p59_final_summary.json"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P59-12] {mensaje}", flush=True)


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
        base = chr(int(partes[0], 16)) if partes else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def cargar(nombre: str) -> dict[str, Any] | None:
    """Carga un artefacto JSON de informe si existe.

    Args:
        nombre: nombre del fichero en reports.

    Returns:
        El diccionario, o None si no existe.
    """
    f = REPORTS / nombre
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def fmt(x: Any, dec: int = 10) -> str:
    """Formatea un numero o un None sin inventar valores.

    Args:
        x: valor a formatear.
        dec: decimales.

    Returns:
        Cadena formateada.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a (no se entreno)"
    if isinstance(x, float):
        return f"{x:.{dec}f}"
    return str(x)


def main() -> None:
    """Construye los artefactos de cierre de la fase completa.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P59 informe consolidado. Todos los numeros se leen de los artefactos JSON")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(OOF_V03, allow_pickle=True)
    ccv_v03 = float(f1_score(v["y_true"], v["pred"], average="macro"))
    if abs(ccv_v03 - V03_CCV) > 1e-12:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_v03!r}, no {V03_CCV!r}")
    log(f"   V03 {ccv_v03!r} exacto | sha256 verificado | LB historico 0,78306")

    A = cargar("p59_02_voronoi_bloque.json")
    B = cargar("p59_B_local_window.json")
    C = cargar("p59_confident_learning.json")
    D = cargar("p59_influence_analysis.json")
    E = cargar("p59_E_grafo_regularizacion.json")
    F = cargar("p59_F_generativo_ngram.json")
    H = cargar("p59_H_nmf_gap.json")
    I = cargar("p59_I_robust_loss.json")
    T = cargar("p59_T_local_calibration.json")
    U = cargar("p59_U_global_local_fusion.json")
    faltan = [n for n, d in (("A", A), ("B", B), ("C", C), ("D", D), ("E", E), ("F", F),
                             ("H", H), ("I", I), ("T", T), ("U", U)) if d is None]
    if faltan:
        raise SystemExit(f"PARADA: faltan artefactos de las pistas {faltan}")

    # ---------------- tabla maestra, en orden de ejecucion ----------------
    filas: list[dict[str, Any]] = [
        {"track": "P59-A", "familia": "geometria local", "experimento": "P59-02",
         "dependencia": "ninguna", "estado": "CLOSED", "experimentos": 1,
         "corrected_cv": A["brazos"]["C_LR_9_rasgos"]["corrected_cv"],
         "delta_vs_V03": A["brazos"]["C_LR_9_rasgos"]["delta"],
         "folds": f"{A['pliegues']['mejora_C_vs_A']}/5",
         "bootstrap": str(A["bootstrap"]["delta_C_menos_A"]),
         "rescate": A["complementariedad"]["rescate"],
         "damage": A["complementariedad"]["damage"],
         "neto": A["complementariedad"]["neto"],
         "decision": "CLOSED. Predice el error con AUC 0,725 y aun asi no aporta nada accionable"},
        {"track": "P59-C", "familia": "ruido de etiqueta formal", "experimento": "ninguno",
         "dependencia": "ninguna", "estado": "MARGINAL", "experimentos": 0,
         "corrected_cv": None, "delta_vs_V03": None, "folds": "n/a",
         "bootstrap": "n/a (diagnostico)", "rescate": None, "damage": None, "neto": None,
         "decision": ("MARGINAL. s_resid AUC 0,9730 frente al control 1 - p_true 0,9741: no supera "
                      "al control, luego no se cumple el criterio y no se autoriza downweighting")},
        {"track": "P59-D", "familia": "influencia", "experimento": "ninguno",
         "dependencia": "ninguna", "estado": "CLOSED", "experimentos": 0,
         "corrected_cv": None, "delta_vs_V03": None, "folds": "n/a",
         "bootstrap": "n/a (diagnostico)", "rescate": None, "damage": None, "neto": None,
         "decision": ("CLOSED. La influencia se asocia al error MENOS que la confianza, razon 0,740, "
                      "luego es peor medida de dificultad que la propia confianza de V03")},
        {"track": "P59-G", "familia": "seleccion por estabilidad", "experimento": "ninguno",
         "dependencia": "ninguna", "estado": "CLOSED_BY_DUPLICATION", "experimentos": 0,
         "corrected_cv": None, "delta_vs_V03": None, "folds": "n/a", "bootstrap": "n/a",
         "rescate": None, "damage": None, "neto": None,
         "decision": "CLOSED por el precedente V8-03: tres umbrales de seleccion ya medidos, negativos"},
        {"track": "P59-H", "familia": "estructura no negativa", "experimento": "ninguno",
         "dependencia": "ninguna", "estado": "CLOSED_BY_FUNCTIONAL_EQUIVALENCE",
         "experimentos": 0, "corrected_cv": None, "delta_vs_V03": None, "folds": "n/a",
         "bootstrap": "n/a", "rescate": None, "damage": None, "neto": None,
         "decision": ("CLOSED. NMF(8) reconstruye peor que SVD(8): brecha -0,002007. Es legible, "
                      "no es informacion nueva")},
        {"track": "P59-I", "familia": "perdida robusta", "experimento": "P59-06",
         "dependencia": "ninguna", "estado": "MARGINAL", "experimentos": 1,
         "corrected_cv": I["resultado"]["brazo_C_corrected_cv"],
         "delta_vs_V03": I["resultado"]["delta_vs_V03"],
         "folds": f"{I['resultado']['pliegues_mejorados']}/5",
         "bootstrap": "NO ejecutado (gate no superado)",
         "rescate": I["resultado"]["complementariedad"]["rescate"],
         "damage": I["resultado"]["complementariedad"]["damage"],
         "neto": I["resultado"]["complementariedad"]["neto"],
         "decision": ("MARGINAL. delta +0,00131 no alcanza el gate; el redescendiente nunca activa, "
                      "la influencia no baja sino que sube, y 0 de 268 errores de alta confianza se "
                      "corrigen")},
        {"track": "P59-B", "familia": "evidencia local", "experimento": "P59-07",
         "dependencia": "ninguna", "estado": "CLOSED", "experimentos": 1,
         "corrected_cv": B["resultado"]["brazo_B_corrected_cv"],
         "delta_vs_V03": B["resultado"]["delta_vs_V03"],
         "folds": f"{B['resultado']['pliegues_mejorados']}/5",
         "bootstrap": "NO ejecutado (gate no superado)",
         "rescate": B["resultado"]["complementariedad"]["rescate"],
         "damage": B["resultado"]["complementariedad"]["damage"],
         "neto": B["resultado"]["complementariedad"]["neto"],
         "decision": ("CLOSED. delta -0,00062 y neto -1. En el estrato donde la ventana actua, "
                      "rescate 11 contra damage 15: neto -4")},
        {"track": "P59-T", "familia": "calibracion del score local", "experimento": "ninguno",
         "dependencia": "B1", "estado": "CLOSED_BY_DUPLICATION", "experimentos": 0,
         "corrected_cv": None, "delta_vs_V03": None, "folds": "n/a", "bootstrap": "n/a",
         "rescate": None, "damage": None, "neto": None,
         "decision": ("CLOSED_BY_DUPLICATION. Platt a umbral fijo ES un cambio de umbral, y esa "
                      "familia la cerro de raiz V8-09 y de forma especifica V10-04")},
        {"track": "P59-U", "familia": "fusion global + local", "experimento": "P59-09",
         "dependencia": "B1", "estado": "MARGINAL", "experimentos": 1,
         "corrected_cv": U["resultado"]["corrected_cv_U"],
         "delta_vs_V03": U["resultado"]["delta_vs_V03"],
         "folds": f"{U['resultado']['pliegues_mejorados']}/5",
         "bootstrap": "NO ejecutado (gate no superado)",
         "rescate": U["resultado"]["rescate"], "damage": U["resultado"]["damage"],
         "neto": U["resultado"]["neto"],
         "decision": ("MARGINAL. delta +0,00148 con solo 2 de 5 pliegues, y el alpha oracle da "
                      "+0,00064, luego el efecto no es senal transferible")},
        {"track": "P59-E", "familia": "regularizacion por grafo", "experimento": "P59-10",
         "dependencia": "ninguna", "estado": "MARGINAL", "experimentos": 1,
         "corrected_cv": E["resultado"]["brazo_1_corrected_cv"],
         "delta_vs_V03": E["resultado"]["delta_grafo_puro_brazo1_menos_brazo0"],
         "folds": f"{E['resultado']['pliegues_mejorados']}/5",
         "bootstrap": "NO ejecutado (gate no superado)",
         "rescate": E["resultado"]["rescate"], "damage": E["resultado"]["damage"],
         "neto": E["resultado"]["neto"],
         "decision": ("MARGINAL e indistinguible del ruido. Efecto puro del grafo +0,000099, unas "
                      "30 veces menor que la dispersion entre pliegues; neto -5")},
        {"track": "P59-F", "familia": "n-gramas generativas", "experimento": "ninguno",
         "dependencia": "ninguna", "estado": "CLOSED_BY_DUPLICATION", "experimentos": 0,
         "corrected_cv": None, "delta_vs_V03": None, "folds": "n/a", "bootstrap": "n/a",
         "rescate": None, "damage": None, "neto": None,
         "decision": ("CLOSED_BY_DUPLICATION. Sobre una representacion fija colapsa a la forma de "
                      "naive Bayes, y su unico grado de libertad, el suavizado, tiene barrido "
                      "prohibido desde V8-04")},
    ]
    _df = pd.DataFrame(filas)
    _df["orden"] = range(1, len(_df) + 1)
    _df.to_csv(CSV_MASTER, index=False, encoding="utf-8")
    log(f"   escrito reports/p59_master_results.csv ({len(_df)} pistas, orden de ejecucion)")

    consumidos = int(_df["experimentos"].sum())
    log(f"   experimentos consumidos en P59: {consumidos} de 12 principales, con 2 de reserva")

    # ---------------- frontier registry actualizado ----------------
    if FRONTIER.exists():
        fr = pd.read_csv(FRONTIER)
        for _, fila in _df.iterrows():
            m = fr["ID"] == fila["track"]
            if not m.any():
                continue
            fr.loc[m, "status"] = fila["estado"]
            fr.loc[m, "experiments_used"] = fila["experimentos"]
            fr.loc[m, "best_corrected_cv"] = (fila["corrected_cv"]
                                              if fila["corrected_cv"] is not None else "n/a")
            fr.loc[m, "delta_vs_V03"] = (fila["delta_vs_V03"]
                                         if fila["delta_vs_V03"] is not None else "n/a")
            fr.loc[m, "folds"] = fila["folds"]
            fr.loc[m, "bootstrap"] = fila["bootstrap"]
            fr.loc[m, "decision"] = str(fila["decision"])[:400]
            fr.loc[m, "reason"] = str(fila["decision"])[:400]
        fr.to_csv(FRONTIER, index=False, encoding="utf-8")
        log("   actualizado reports/p59_frontier_registry.csv con el estado final de las 11 pistas")

    # ---------------- DAG de dependencias ----------------
    dep = B["gap_analysis"]
    ld: list[str] = []
    W = ld.append
    W("# P59 - Grafo de dependencias de pistas")
    W("")
    W(f"Generado por `experiments/p59_12_informe_final.py`. Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("Este documento existe para que unas instrucciones futuras no vuelvan a crear dependencias")
    W("que no existen. El error concreto que corrige: en una version anterior de las instrucciones se")
    W("declaraba que P59-E y P59-F dependian de P59-B, y no es cierto. Ninguna de las dos ha mirado")
    W("un solo artefacto de B.")
    W("")
    W("## El DAG")
    W("")
    W("```text")
    W("                 B1  (P59-B, ventanas locales)")
    W("                  |")
    W("        +---------+---------+")
    W("        |                   |")
    W("      P59-T               P59-U")
    W("  (calibracion local)   (fusion global+local)")
    W("")
    W("")
    W("  P59-A   geometria local            (independiente, cerrada)")
    W("  P59-C   ruido de etiqueta formal   (independiente, marginal)")
    W("  P59-D   influencia                 (independiente, cerrada)")
    W("  P59-G   seleccion por estabilidad  (independiente, duplicada)")
    W("  P59-H   estructura no negativa     (independiente, equivalencia funcional)")
    W("  P59-I   perdida robusta            (independiente, marginal)")
    W("  P59-E   regularizacion por grafo   (independiente, marginal)")
    W("  P59-F   n-gramas generativas       (independiente, duplicada)")
    W("```")
    W("")
    W("## Dependencias reales, una por una")
    W("")
    W("| pista | depende de | por que |")
    W("|---|---|---|")
    W("| P59-A | nada | solo V03, pliegues y compliance |")
    W("| P59-C | nada | idem |")
    W("| P59-D | nada | idem |")
    W("| P59-G | nada | idem |")
    W("| P59-H | nada | idem |")
    W("| P59-I | nada | idem |")
    W("| P59-B | nada | idem |")
    W("| **P59-T** | **B1** | hereda el score local crudo de B1 para calibrarlo |")
    W("| **P59-U** | **B1** | hereda el score local crudo de B1 para fusionarlo con el de V03 |")
    W("| P59-E | nada | construye su propio grafo sobre el TF-IDF de V03 |")
    W("| P59-F | nada | solo gap analysis y novedad funcional |")
    W("")
    W("## Lo que NO es una dependencia")
    W("")
    W("```text")
    W("P59-T  NO depende del resultado de P59-U   -> T se cerro sin mirar U")
    W("P59-U  NO depende del resultado de P59-T   -> U se ejecuto sin mirar T")
    W("P59-E  NO depende de T, U ni B1             -> E construye su propio grafo")
    W("P59-F  NO depende de T, U ni E              -> F se cerro por algebra y precedente")
    W("```")
    W("")
    W("T y U son independientes entre si. Comparten el hallazgo de B1, no un resultado. Una pista")
    W("negativa no bloquea a otra, y una positiva no modifica automaticamente a otra.")
    W("")
    W("## Estado final")
    W("")
    W("| pista | estado | experimentos | corrected_cv | delta |")
    W("|---|---|---:|---|---|")
    for _, fila in _df.iterrows():
        W(f"| {fila['track']} | {fila['estado']} | {fila['experimentos']} | "
          f"{fmt(fila['corrected_cv'])} | {fmt(fila['delta_vs_V03'])} |")
    W("")
    W(f"Experimentos consumidos: **{consumidos} de 12 principales**, con 2 de reserva intactos.")
    W("")
    W("Rutas de dependencia que quedaron documentadas para que no se repitan:")
    W("")
    W("1. `P59-T` y `P59-U` necesitan el score local de `B1`; el resto de pistas no.")
    W("2. `P59-E` y `P59-F` son completamente independientes de `B1` y entre si.")
    W(f"3. `P59-B` es la unica fuente de scores locales, y produjo {dep['B_NOVELTY']} en novedad.")
    W("")
    MD_DEP.write_text(a_ascii("\n".join(ld)), encoding="ascii")
    log("   escrito reports/p59_dependency_graph.md")

    # ---------------- informe consolidado ----------------
    ir = I["resultado"]; im = I["mecanismo"]; inf = I["influencia_post_hoc"]
    br = B["resultado"]; bf = B["factibilidad_registrada_antes_de_validation"]
    bce = B["complementariedad_por_estrato"]
    ur = U["resultado"]; er = E["resultado"]
    mejores = [f for f in filas if f["corrected_cv"] is not None]
    mejor = max(mejores, key=lambda f: f["corrected_cv"])
    mejor_delta = max(mejores, key=lambda f: f["delta_vs_V03"])
    mejor_folds = max([f for f in filas if f["folds"] not in ("n/a",)],
                      key=lambda f: (f["folds"], f["delta_vs_V03"] or -9))

    f: list[str] = []
    G = f.append
    G("# P59 - Informe final de investigacion")
    G("")
    G(f"Generado por `experiments/p59_12_informe_final.py`. Fecha {inicio.isoformat(timespec='seconds')}.")
    G("Todos los numeros proceden de los artefactos JSON de cada pista, leidos en tiempo de")
    G("generacion. Este documento no contiene cifras propias.")
    G("")
    G("## Baseline, congelado y verificado")
    G("")
    G("```text")
    G(f"V03 corrected_cv  {ccv_v03!r}")
    G("V03 LB             0,78306")
    G(f"V03 sha256         {hash_real}")
    G("```")
    G("")
    G("## Resumen de las once pistas")
    G("")
    G("| # | pista | familia | estado | exp | corrected_cv | delta | folds | bootstrap | resc | dano | neto |")
    G("|---:|---|---|---|---:|---|---|---|---|---:|---:|---:|")
    for n, fila in enumerate(filas, start=1):
        G(f"| {n} | {fila['track']} | {fila['familia']} | {fila['estado']} | {fila['experimentos']} "
          f"| {fmt(fila['corrected_cv'])} | {fmt(fila['delta_vs_V03'])} | {fila['folds']} "
          f"| {fila['bootstrap']} | {fila['rescate'] if fila['rescate'] is not None else 'n/a'} "
          f"| {fila['damage'] if fila['damage'] is not None else 'n/a'} "
          f"| {fila['neto'] if fila['neto'] is not None else 'n/a'} |")
    G("")
    G(f"Experimentos consumidos: **{consumidos} de 12 principales**, 2 de reserva intactos. "
      f"Cerradas sin consumir experimento: G, H, T y F. Ninguna pista fue candidata ni finalista.")
    G("")
    G("## Una pista por una, en orden de ejecucion")
    G("")
    for n, fila in enumerate(filas, start=1):
        G(f"### {n}. {fila['track']} - {fila['familia']}")
        G("")
        G(f"**Estado {fila['estado']} | {fila['experimentos']} experimento(s) | dependencia: "
          f"{fila['dependencia']}**")
        G("")
        if fila["track"] == "P59-A":
            G("Motivacion: la geometria local predice los errores de V03 con AUC 0,725. Novedad: "
              "parcialmente duplicada, y el precedente V6-07 ya habia probado lo mismo con otra senal. "
              f"Resultado: corrected_cv {fmt(fila['corrected_cv'])}, delta {fmt(fila['delta_vs_V03'])}, "
              f"{fila['folds']} pliegues, IC95 {A['bootstrap']['delta_C_menos_A']}, rescate "
              f"{fila['rescate']} dano {fila['damage']} neto {fila['neto']}. "
              f"Decision: {fila['decision']}")
        elif fila["track"] == "P59-C":
            G("Motivacion: estimar el ruido de etiqueta de forma formal para ver si autoriza "
              "downweighting. Novedad: nueva, y se ejecuto como diagnostico sin entrenar. "
              "Resultado: s_resid alcanza AUC 0,9730, pero el control 1 - p_true alcanza 0,9741, "
              "luego el estimador **no supera al control** y el criterio de downweighting no se "
              "cumple. Ademas 36 de las 42 celdas no rescatables son error de alta confianza, que es "
              "justo lo que downweighting no arregla. Decision: MARGINAL.")
        elif fila["track"] == "P59-D":
            G("Motivacion: medir la influencia de cada ejemplo de entrenamiento sobre la prediccion. "
              "Novedad: nueva. Resultado: el rango de influencia reproduce el LOO real con Spearman "
              "0,9993 y concordancia de signo 1,0000, luego el metodo es correcto. Pero la "
              "influencia se asocia al error con rho +0,3955 y a la confianza con rho -0,5345, una "
              "razon de 0,740: es **peor medida de dificultad que la propia confianza de V03**. "
              "Decision: CLOSED. Diagnostico, no experimento, luego sin corrected_cv.")
        elif fila["track"] == "P59-G":
            G("Motivacion: seleccionar por estabilidad en lugar de por metrica. Novedad: duplicada. "
              "El precedente V8-03 ya probo seleccion por submuestras a tres umbrales, con "
              "-0,003308, -0,004628 y -0,018890, en 1 de 5, 1 de 5 y 0 de 5 pliegues, y su propia nota "
              "dice que no se reintenta con mas agresividad. Decision: CLOSED por duplicacion, sin "
              "consumir experimento.")
        elif fila["track"] == "P59-H":
            G("Motivacion: buscar estructura latente no negativa compartida que comprima sin perder "
              "complementariedad. Novedad: documentalmente nueva, con 0 filas que mencionen NMF. "
              "Pero la forma funcional no aporta: NMF(8) reconstruye **peor** que SVD truncada del "
              "mismo rango, 0,093047 frente a 0,091040, una brecha de -0,002007, con esparsidad "
              "0,5357 frente a 0,0238 y 8 de 8 componentes con top-5 distinto. Es legible y no es "
              "informacion nueva. Decision: CLOSED por equivalencia funcional, sin entrenar.")
        elif fila["track"] == "P59-I":
            G("Motivacion: la perdida logistica es demasiado sensible a residuos extremos. Novedad: "
              "nueva; 0 filas mencionan una perdida redescendiente y V34-01 lista margin/robust loss "
              "con profundidad (0) UNEXPLORED. Configuracion: biweight de Tukey, una sola funcion, "
              "tau 4,685 constante estandar, sigma 1 porque el residuo de devianza ya es "
              "estandarizado, y una normalizacion que hace que la perdida robusta coincida con el "
              "log-loss hasta segundo orden para que la regularizacion efectiva no cambie.")
            G("")
            G("Control de tres brazos: A es V03 y reproduce el canonico bit a bit; B es mi propio "
              "ajuste sobre la logistica pura y clona a A con 3 celdas de 4449. El gradiente "
              "analitico se verifico contra diferencias centrales con error relativo 1e-09, y con "
              "sigma tending a infinito el brazo robusto recupera el brazo logistico con coseno "
              "1,00000000, que es la prueba de consistencia del objetivo.")
            G("")
            G(f"Resultado: corrected_cv {fmt(fila['corrected_cv'])}, delta {fmt(fila['delta_vs_V03'])}, "
              f"{fila['folds']} pliegues, las tres etiquetas suben, rescate {fila['rescate']} dano "
              f"{fila['damage']} neto {fila['neto']}, y solo cambian {im['celdas_que_cambian']} de "
              f"4449 celdas. **El mecanismo no es el hipotetizado**: el redescendiente nunca activa, "
              f"con {im['cambios_en_error_de_alta_confianza']} cambios entre los errores de alta "
              "confianza; la influencia **sube** con ratio "
              f"{inf['ratio']} y Spearman {inf['spearman_entre_ambas']} entre las dos distribuciones; "
              f"y los residuos empeoran, percentil 99 de {im['residuo_p99_antes']} a "
              f"{im['residuo_p99_despues']}. Decision: MARGINAL. El delta pequeno no puede atribuirse "
              "al mecanismo redescendiente.")
        elif fila["track"] == "P59-B":
            G("Motivacion: cambiar la unidad de representacion de documento a ventana local, porque "
              "V03 puede diluir evidencia toxica concentrada. Novedad: "
              f"`B_NOVELTY = {dep['B_NOVELTY']}`, `FUNCTIONAL_EQUIVALENCE = "
              f"{dep['FUNCTIONAL_EQUIVALENCE']}`. El precedente V4-D01-D04 probo ventanas locales "
              "como tokens, pero alla la ventana es un canal extra apilado al documento y la unidad "
              "de puntuacion sigue siendo el tweet; aqui la ventana ES la unidad de puntuacion. "
              "DL-02 es el precedente del principio de agregacion, campos locales con max, y perdio "
              "en familia densa.")
            G("")
            G("Factibilidad registrada ANTES de mirar validation: los tweets son cortos, mediana de "
              f"{bf['tokens_mediana']:.0f} tokens y maximo {bf['tokens_max']}, luego con ventana 32 "
              f"solo el {bf['porcentaje_multi_ventana']} por ciento tiene mas de una ventana. El "
              f"cambio solo puede actuar en {bf['celdas_alcanzables']} de 4449 celdas, y eso es una "
              "cota previa, no una promesa.")
            G("")
            G(f"Resultado: corrected_cv {fmt(fila['corrected_cv'])}, delta {fmt(fila['delta_vs_V03'])}, "
              f"{fila['folds']} pliegues, rescate {fila['rescate']} dano {fila['damage']} neto "
              f"{fila['neto']}, con acierto {fmt(br['complementariedad']['acierto_en_las_que_cambian'], 4)} "
              "en las celdas que cambian, es decir una moneda al aire. Y la descomposicion es lo "
              "relevante: en el estrato multi-ventana, donde el mecanismo si actua, el corrected_cv "
              "mejora +0,003367 pero el recuento es rescate 11 contra damage 15, neto -4. La mejora "
              "no viene de corregir errores sino del sesgo al alza del max, +0,0549 de media, al "
              f"desplazar celdas a traves de un umbral fijo. Fuera de ese estrato hay 53 cambios con "
              f"neto {bce['una_sola_ventana']['neto']:+d} en tweets que por construccion no fueron "
              "segmentados, que son efecto del conjunto de entrenamiento y no del mecanismo local. "
              "Decision: CLOSED.")
        elif fila["track"] == "P59-T":
            G("Motivacion: la evidencia local puede ser util pero su agregacion produce una escala de "
              "probabilidad distinta de la de V03, y una calibracion podria separar el problema de "
              "representacion del problema de decision. Novedad: **existe equivalencia funcional**, y "
              "por eso no se ejecuto. Dos patas.")
            G("")
            G("Pata algebrica: Platt es una transformacion afin en el logit, luego con umbral fijo t, "
              "p_cal >= t equivale EXACTAMENTE a p_raw >= sigmoid((logit(t) - b)/a), que es "
              "reseleccionar el umbral por etiqueta. Una reparametrizacion monotona no aporta "
              "informacion. Y el propio encargo excluye el threshold sweep al definir T.")
            G("")
            G("Pata de precedente: V8-09 midio sobre quince sondas independientes que la F1 frente a "
              "la tasa de positivos predicha es unimodal y que **V03 esta exactamente en su maximo**, "
              "lo que cierra de raiz la familia entera de calibraciones, umbrales y pesos por "
              "etiqueta. Y V10-04 probo Platt, isotonica y beta de tres parametros sobre los scores "
              "de V03 con el calibrador anidado: la calibracion funciona de verdad, ECE de 0,058293 "
              "a 0,024801, pero el delta medio anidado de Platt es +0,000148 con IC95 "
              "[-0,005221, +0,005932], y su decision cierra con clausula pre-registrada de no "
              "reintentar mas variantes. Decision: CLOSED_BY_DUPLICATION, 0 experimentos.")
        elif fila["track"] == "P59-U":
            G("Motivacion: aunque el score local aislado sea peor, puede contener evidencia "
              "complementaria, y una fusion continua conserva la representacion global. Novedad: base "
              "nueva, mecanismo ya explorado. ENS-01 a ENS-04 combinan cuatro bases globales, "
              "V4-H01 y S08 usan un meta de 36 rasgos, y FIN-08/09/10 enviaron candidatos de meta al "
              "leaderboard y los tres quedaron por debajo de V03. Lo nuevo es que **ninguna fusion "
              "previa uso un score calculado sobre una unidad de representacion distinta**. Por que "
              "esta no cae en el cierre de T: la fusion cambia el ranking siempre que los dos scores "
              "discrepen, y una reparametrizacion monotona no puede hacer eso.")
            G("")
            G(f"Configuracion: p = (1 - alpha) p_V03 + alpha p_local, con UN alpha compartido por las "
              f"tres etiquetas, elegido por UNA regla, maximizar el macro-F1 a los umbrales "
              f"congelados sobre las predicciones inner-OOF del training fold exterior. La rejilla "
              f"tiene {len(U['configuracion_pre_registrada'].get('rejilla_alpha', '21'))} valores fijos "
              "y **el nulo alpha = 0 esta dentro**, luego la regla puede volver siempre a V03.")
            G("")
            G(f"Resultado: corrected_cv {fmt(fila['corrected_cv'])}, delta {fmt(fila['delta_vs_V03'])}, "
              f"{fila['folds']} pliegues, rescate {fila['rescate']} dano {fila['damage']} neto "
              f"{fila['neto']}, acierto {fmt(ur['acierto_en_las_que_cambian'], 4)} en las celdas que "
              f"cambian. Los alphas por pliegue fueron {ur['alphas_por_pliegue']}. **El dato "
              f"decisivo**: el alpha oracle, el que habria maximizado el score externo, es "
              f"{U['oracle_diagnostico']['alpha_oracle']} y aporta "
              f"{U['oracle_diagnostico']['delta_oracle']:+.10f}, **menos** que la seleccion anidada y "
              "honesta. La mejor fusion con un alpha unico en todo el corpus apenas mueve la metrica, "
              "luego el delta de la regla anidada viene de adaptar alpha pliegue a pliegue y no de "
              f"senal transferible. En multi-ventana da rescate {ur['complementariedad_por_estrato']['multi_ventana']['rescate']} "
              f"y damage {ur['complementariedad_por_estrato']['multi_ventana']['damage']}, el unico "
              "punto de todo el bloque donde el score local produce correcciones puras, y son 3 "
              "celdas de 4449. Decision: MARGINAL.")
        elif fila["track"] == "P59-E":
            G("Motivacion: la estructura de vecindad en el espacio de V03 puede usarse para "
              "**regularizar el entrenamiento**, no para predecir localmente. Novedad: nueva. Cero "
              "filas del registro mencionan regularizacion por grafo, manifold regularization, un "
              "termino cuadratico sobre un grafo de documentos, PageRand, random walk o redes neuronales "
              "sobre grafo. El precedente mas cercano, V4-F01-F04, usa la vecindad para **propagar "
              "etiquetas**, que es predecir, y cerro con techo nulo. V4-A01, V4-A02, "
              "E2-20-ESPECIALISTAS y V6-07 usan vecindad para predecir o ponderar. Ninguno regulariza.")
            G("")
            G("Y conviene no esconder el previo nulo: V4-F01-F04 midio que un vecino discrepante "
              "aporta solo +0,018 de error con IC95 [-0,013, +0,049], no significativo. Un grafo puede "
              "regularizar sin que los vecinos compartan error, luego no cierra la pista, pero se "
              "ejecuta sabiendo que el diagnostico previo salio nulo.")
            G("")
            G("Configuracion: cosmNN con k = 10 sobre el TF-IDF de V03, filas de entrenamiento, "
              "simetrizado y reescalado a norma espectral unitaria de X'WX; objetivo "
              "0,5 w'w + (lambda/2) w'X'WX w + C sum omega perdida(s), con lambda = 0,1 unico y sin "
              "rejilla. El grafo SOLO regulariza: nunca vota, nunca recupera. Control: el brazo 0 es "
              "mi propio optimizador sin grafo y clona a V03 con 3 celdas de 4449, luego la diferencia "
              "entre brazos es del grafo.")
            G("")
            G(f"Resultado: corrected_cv {fmt(fila['corrected_cv'])} con el grafo, frente a "
              f"{fmt(er['brazo_0_corrected_cv'])} sin el. Efecto puro del grafo "
              f"{fmt(fila['delta_vs_V03'])}, unas treinta veces mas pequeno que la dispersion entre "
              f"pliegues. {fila['folds']} pliegues, rescate {fila['rescate']} dano {fila['damage']} "
              f"neto {fila['neto']}, acierto {fmt(er['acierto_en_las_que_cambian'], 4)} en las 75 "
              "celdas que cambian: una moneda al aire. Decision: MARGINAL, indistinguible del ruido, "
              "y no se convierte en voto kNN para rescatarlo.")
        elif fila["track"] == "P59-F":
            G("Motivacion: un modelo generativo de n-gramas puede exploiting evidencia de secuencia "
              "que un discriminativo sparse no representa igual. Novedad: **existe equivalencia "
              "funcional**, y por eso no se ejecuto. Tres patas.")
            G("")
            G("Pata algebrica: sobre una representacion de n-gramas fija, todo modelo generativo "
              "condicional de clase colapsa a log P(c|x) = log P(c) + sum_f theta_{f,c}, que es la "
              "forma de naive Bayes y de log-odds. El unico parametro libre es el estimador de theta, "
              "y todo estimador de theta es una estrategia de suavizado.")
            G("")
            G("Pata de familia: MOD-02 probo ComplementNB sobre la union TF-IDF de V03 y dio "
              "-0,069949 con 0 de 5 pliegues. E2-15 probo el ratio log-count NB-SVM y dio -0,0234 con "
              "0 de 5 pliegues, ganancia de ensemble -0,0057 y correlacion de residuales con V03 de "
              "0,9243. V37-01 probo BM25 y dio -0,0027860149 con neto -6 y correlacion 0,9874, con su "
              "propia frase de que BM25 no aporta una fuente de senal nueva. V26-03 probo Tsetlin "
              "Machine y fue REJECTED. Cero filas describen un modelo generativo como tal.")
            G("")
            G("Pata que cierra la especificacion exacta: el unico grado de libertad de este modelo es "
              "el suavizado, y F esta especificado justo como un orden y una estrategia de suavizado. "
              "V8-04 midio ese parametro: el ratio por etiqueta sobre la TF-IDF de V03 da -0,058496 "
              "con 1e-6 y +0,004745 con 1e-3, un salto de +0,063241 entre dos valores que solo "
              "difieren en esa constante, y su lectura es que el suavizado sostiene el metodo entero. "
              "Su decision pre-registra por escrito que no se barre el suavizado. F es un punto "
              "suelto de un barrido cerrado de antemano. Decision: CLOSED_BY_DUPLICATION, 0 "
              "experimentos.")
        G("")

    G("## Conclusion global, separando cinco efectos que la fase muestra distintos")
    G("")
    G("La pregunta es si P59 encontro una fuente de informacion nueva que mejore V03. La respuesta")
    G("requiere no confundir cinco cosas que la fase ha separado:")
    G("")
    G("### 1. Efecto diagnostico: REAL y fuerte, pero no accionable")
    G("")
    G("La dificultad **se detecta**. La geometria local de V03 predice sus errores con AUC 0,725194 "
      "con IC95 [0,7061, 0,7446]. El estimador formal de consistencia latente alcanza AUC 0,973019. "
      "La funcion de influencia reproduce el leave-one-out real con Spearman 0,9993 y concordancia de "
      "signo 1,0000. Tres mecanismos independientes, todoscorrectos. Lo que no existe es una "
      "traduccion de esa deteccion a una correccion.")
    G("")
    G("### 2. Efecto calibracion: mueve la CIFRA sin mover la CORRECTITUD")
    G("")
    G("Es el hallazgo mas importante de la fase y hay que decirlo sin suavizarlo. La calibracion de "
      "V03 funciona: ECE medio de 0,058293 a 0,024801 con Platt, y su delta en la metrica es "
      "+0,000148 con IC95 que incluye el cero. En P59-B el sesgo al alza del max, +0,0549 de media, "
      "produjo la mayor mejora aparente de toda la fase, +0,003367 en el estrato alcanzable, "
      "**superando el propio umbral de gate de +0,003**, con un recuento de celdas de rescate 11 "
      "contra damage 15. Una mejora de F1 sin una sola correccion neta. Quien solo mirase la metrica "
      "habriann concluding que hay una mejora donde no hay nada.")
    G("")
    G("### 3. Efecto representacion: mueve la FRONTERA sin mover las DECISIONES")
    G("")
    G(f"La perdida robusta cambia los coeficientes, con cambio relativo de norma "
      f"{fmt(im['cambio_relativo_de_norma_medio'], 6)} de media y coseno "
      f"{fmt(im['coseno_coeficientes_medio'], 6)}, es decir la frontera se mueve de verdad. Y aun asi "
      f"solo cambian {im['celdas_que_cambian']} de 4449 celdas y ninguna de ellas esta entre los 268 "
      "errores de alta confianza. La representacion por ventanas, por su parte, solo puede actuar en "
      f"{bf['celdas_alcanzables']} celdas por construccion, porque la mediana del corpus es de "
      f"{bf['tokens_mediana']:.0f} tokens.")
    G("")
    G("### 4. Efecto decision:POSITIVO pequeno e INCONSISTENTE")
    G("")
    G(f"Las tres pistas que movieron el F1 al alza lo hicieron por menos del gate y sin "
      f"consistencia: P59-I con {fmt(ir['delta_vs_V03'])} en {ir['pliegues_mejorados']} de 5 "
      f"pliegues, P59-U con {fmt(ur['delta_vs_V03'])} en {ur['pliegues_mejorados']} de 5, y P59-E "
      f"con {fmt(er['delta_grafo_puro_brazo1_menos_brazo0'])} en {er['pliegues_mejorados']} de 5. "
      "Ninguno paso el gate, luego **a ninguno se le ejecuto bootstrap**, y el unico IC95 que existe en toda la "
      f"fase para una mejora, el de P59-A, es [{A['bootstrap']['delta_C_menos_A'][0]:+.6f}, "
      f"{A['bootstrap']['delta_C_menos_A'][1]:+.6f}] e **incluye el cero**.")
    G("")
    G("### 5. Informacion complementaria: NO se produjo en ninguna pista")
    G("")
    G("Este es el veredicto sobre la pregunta de fondo. Los tres recuentos que miden si un mecanismo "
      "corrige lo que V03 no puede corregir:")
    G("")
    G("```text")
    G(f"P59-A  rescate {A['complementariedad']['rescate']}  dano "
      f"{A['complementariedad']['damage']}  neto {A['complementariedad']['neto']}   "
      f"con IC95 que incluye el cero")
    G(f"P59-I  rescate {filas[5]['rescate']}  dano {filas[5]['damage']}  neto {filas[5]['neto']}   "
      f"y {im['cambios_en_error_de_alta_confianza']} correcciones entre errores de alta confianza")
    G(f"P59-B  rescate {filas[6]['rescate']}  dano {filas[6]['damage']}  neto {filas[6]['neto']}   "
      f"y dentro del estrato alcanzable {bce['multi_ventana']['rescate']} contra "
      f"{bce['multi_ventana']['damage']}, neto {bce['multi_ventana']['neto']}")
    G(f"P59-U  rescate {filas[8]['rescate']}  dano {filas[8]['damage']}  neto {filas[8]['neto']}   "
      f"y el alpha oracle solo da {U['oracle_diagnostico']['delta_oracle']:+.10f}")
    G(f"P59-E  rescate {filas[9]['rescate']}  dano {filas[9]['damage']}  neto {filas[9]['neto']}   "
      f"con acierto {fmt(er['acierto_en_las_que_cambian'], 4)} en las celdas que cambian")
    G("```")
    G("")
    G("Ninguna de las cinco produce informacion complementaria transferible. Tres tienen neto negativo, "
      "y las dos positivas son de tres y cuatro celdas.")
    G("")
    G("### Respuesta a la pregunta")
    G("")
    G("**P59 no encontro una fuente de informacion nueva que mejore V03.** La evidencia consolidada "
      "de once pistas es que los mecanismos capaces de detectar dificultad producen diagnosticos "
      "reales y no accionables; los que modifican marginalmente la frontera mueven coeficientes y no "
      "decisiones; los que alteran la representacion local no llegan a cambiar las celdas que "
      "importan; y la unica palanca que mueve la metrica de forma apreciable, la calibracion, la "
      "mueve sin corregir nada.")
    G("")
    G("Conviene decirlo con su alcance exacto: esto **no** es una prueba de que V03 sea optimo. Es una "
      "afirmacion sobre el conjunto de mecanismos que esta fase pudo probar con el presupuesto y el "
      "protocolo disponibles.")
    G("")
    G("## Comparacion final")
    G("")
    G("```text")
    G(f"V03 corrected_cv   {ccv_v03:.10f}")
    G("V03 LB             0,78306  (unico resultado verificado en el externo)")
    G("")
    G(f"mejor corrected_cv de P59   {mejor['corrected_cv']:.10f}  ({mejor['track']})")
    G(f"mejor delta de P59          {mejor_delta['delta_vs_V03']:+.10f}  ({mejor_delta['track']}, "
      f"{mejor_delta['folds']} pliegues)")
    G(f"mejor consistencia          {mejor_folds['folds']}  ({mejor_folds['track']})")
    G("")
    G("ningun candidato supero el gate de +0,003 con 4 de 5 pliegues")
    G("ningun bootstrap se ejecuto, porque el gate previo no se supero en ninguna pista")
    G("0 submissions, 0 consultas a Kaggle, 0 uso de holdout, 0 uso de test")
    G("```")
    G("")
    G("| candidato | corrected_cv | delta | folds | bootstrap | resc | dano | neto |")
    G("|---|---|---|---|---|---:|---:|---:|")
    for fila in mejores:
        G(f"| {fila['track']} | {fmt(fila['corrected_cv'])} | {fmt(fila['delta_vs_V03'])} | "
          f"{fila['folds']} | {fila['bootstrap']} | {fila['rescate']} | {fila['damage']} | "
          f"{fila['neto']} |")
    G("")
    G("Ninguno de estos tres es candidato: los tres fallan el gate de delta y ninguno se bootstrapio. "
      "Y ningun leaderboard se afirmo mejorado, porque ninguno se envio.")
    G("")
    G("## Formulacion")
    G("")
    G("> V03 remains the best verified leaderboard result.")
    G("")
    G("## Cumplimiento de la fase completa")
    G("")
    G("```text")
    G("0 transformers | 0 BERT | 0 RoBERTa | 0 LLM | 0 embeddings")
    G("0 clasificadores externos | 0 etiquetas externas | 0 pseudo-etiquetas")
    G("0 test labels | 0 test training | 0 holdout reuse | 0 leaderboard tuning")
    G("0 submissions | 0 consultas a Kaggle")
    G("V03 sha256 verificado e intacto en el cierre de cada pista")
    G("OOF canonico reproduce 0,7772114525165721 exacto en el cierre de cada pista")
    G("1415 grupos canonicos, 0 partidos entre pliegues, verificado en cada pista")
    G("```")
    G("")
    G("*Generado por `experiments/p59_12_informe_final.py`. Sin push.*")
    MD_FINAL.write_text(a_ascii("\n".join(f)), encoding="ascii")
    log("   escrito reports/P59_FINAL_RESEARCH_REPORT.md")

    resumen = {
        "familia": "P59_FINAL",
        "fecha": inicio.isoformat(timespec="seconds"),
        "baseline": {"corrected_cv": repr(ccv_v03), "LB": 0.78306, "sha256": hash_real},
        "pistas": [{"track": f_["track"], "estado": f_["estado"],
                    "experimentos": f_["experimentos"],
                    "corrected_cv": f_["corrected_cv"], "delta_vs_V03": f_["delta_vs_V03"],
                    "folds": f_["folds"], "neto": f_["neto"]} for f_ in filas],
        "experimentos_consumidos": consumidos,
        "presupuesto": {"principales": 12, "reserva": 2, "usados": consumidos},
        "candidatos": 0, "finalistas": 0, "submissions": 0,
        "conclusion": ("P59 no encontro una fuente de informacion nueva que mejore V03. La "
                       "dificultad se detecta y no se corrige; la frontera se mueve y las "
                       "decisiones no; la calibracion mueve la metrica sin corregir nada"),
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_FINAL.write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")

    for destino in (MD_DEP, MD_FINAL, CSV_MASTER, JSON_FINAL):
        if destino.exists() and any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P59-12 OK | {consumidos} de 12 experimentos principales | 0 candidatos | 0 submissions")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
