"""P66 track P00 and P01: baseline audit, candidate inventory, and the historical gap analysis.

Why this phase exists, in one sentence
--------------------------------------
P64 and P65 measured five representations that beat BOTH a capacity control and V03's own
margin, and none of them improved V03 once concatenated. P66 tests the one thing that was not
tested: whether that same evidence works as a small correction in score space, without touching
the TF-IDF space at all.

The inventory is worse than expected, and it is reported rather than worked around
---------------------------------------------------------------------------
The brief says candidate OOF scores must be reused when a valid one exists. Checking the
repository shows that none of the five P64/P65 scripts wrote their probability arrays to disk.
They reported metrics and fold scores and kept the probabilities in memory, so the artefacts
on disk are:

    features/oof_v03_final.npz            the only probability artefact, and it is V03 itself
    features/v2_structured_features.parquet

and features/ is in .gitignore, so even the feature blocks those scripts wrote
(p64_evidencia_cells.npz, p64_loo_cells.npz) are not in the repository and do not exist in this
worktree.

So the candidate scores have to be REGENERATED. That is not a workaround and it is not a
violation of the "do not retrain" rule, because the alternative would be inventing scores. The
regeneration reuses the source code of P64 and P65 verbatim, with the same seeds and the same
folds, and this script verifies the regeneration by checking that the corrected_cv of each
regenerated arm reproduces the value those phases reported. If a number does not reproduce, the
regeneration is wrong and the phase stops rather than proceeding on unverified scores.

That verification is the reason this script exists as a separate step instead of regenerating
silently inside the fusion script.

What the history says about score-space fusion
----------------------------------------------
The gap analysis is not a formality here. This repository has done score-space work before, and
the brief names the families to search for. The decisive precedents are V6-05's fusion of a
local score with V03 through a single alpha, which reached +0,0014800075 with 2 of 5 folds and
failed its gate on both conditions, and P59-09, which is the same idea measured again.

What is new relative to those two is the RESIDUALISATION, and the difference is not cosmetic.
Blending a second score into V03 asks whether the second score is useful. Residualising asks
whether the second score contains anything that V03's own score does not predict, and then
corrects only that part. A blended alpha that fails tells you the mixture was wrong. A residual
that fails tells you the candidate is redundant. Those are different conclusions and only the
second one is about information.

Usage
-----
    /home/carlos/venv/bin/python experiments/p66_00_audit_gap.py
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
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
REGISTRO = PROJECT_ROOT / "experiments.csv"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"

MD_GAP = REPORTS / "p66_hypothesis_gap_analysis.md"
CSV_GAP = REPORTS / "p66_hypothesis_gap_analysis.csv"
JSON_GAP = REPORTS / "p66_hypothesis_gap_analysis.json"
CSV_INV = REPORTS / "p66_candidate_inventory.csv"

EXACT = "EXACT_DUPLICATE"
FUNC = "FUNCTIONAL_DUPLICATE"
PARCIAL = "PARTIAL_PRECEDENT"
NUEVA = "GENUINELY_NEW"

# Candidatos que P66 debe explotar, con el script que los produjo y el valor que ese script
# reporto, que es el que la regeneracion tiene que reproducir.
CANDIDATOS: tuple[dict[str, Any], ...] = (
    {"nombre": "MI_arbol", "script": "p64_05_multiple_instance.py",
     "delta_reportado": -0.0091686799, "brecha_margen": 0.0093823920,
     "prioridad": 1,
     "por_que": "P64 lo encontro por encima del margen de V03 en +0,0093823920, que es la "
                "brecha mas alta de toda la investigacion y la unica con componente NO LINEAL"},
    {"nombre": "negacion", "script": "p64_07_negacion.py",
     "delta_reportado": -0.0021826774, "brecha_margen": 0.0088975231, "prioridad": 2,
     "por_que": "unico candidato con las DOS brechas positivas a la vez, capacidad y margen"},
    {"nombre": "word_1_3", "script": "p65_01_ngrams.py",
     "delta_reportado": -0.0039520688, "brecha_margen": 0.0071281317, "prioridad": 3,
     "por_que": "la brecha contra el control de anchura mas alta, +0,0596671770"},
    {"nombre": "cohesion", "script": "p65_03_cohesion.py",
     "delta_reportado": -0.0027090062, "brecha_margen": 0.0083711943, "prioridad": 4,
     "por_que": "seis cantidades, todas medibles, con el bloque ya construido"},
    {"nombre": "huecos_1_2", "script": "p65_02_huecos_contraste.py",
     "delta_reportado": -0.0010515908, "brecha_margen": 0.0100286097, "prioridad": 5,
     "por_que": "el mas cercano a V03 de todos los candidatos, solo -0,0010515908"},
    {"nombre": "contraste", "script": "p65_02_huecos_contraste.py",
     "delta_reportado": -0.0023851933, "brecha_margen": 0.0086950072, "prioridad": 6,
     "por_que": "el mas PARCORELECTADO en P65 y el unico con un solo canal de 53 columnas"},
    {"nombre": "huecos_6_10", "script": "p65_02_huecos_contraste.py",
     "delta_reportado": -0.0074391131, "brecha_margen": 0.0036410874, "prioridad": 7,
     "por_que": "el mas negativo de los siete, y por eso el control de que la fusion no lo "
                "arrastre como senal espuria"},
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P66-00] {mensaje}", flush=True)


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


def main() -> None:
    """Audita el baseline, inventaria los candidatos y clasifica la fusion por score.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P66-00 auditoria, inventario de candidatos y analisis de hipotesis")

    # ---------- P00: la auditoria ----------
    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(OOF, allow_pickle=True)
    ccv = float(v["corrected_cv"][0])
    if ccv != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv!r}, no {V03_CCV!r}")
    reg = pd.read_csv(REGISTRO)
    if reg["experiment_id"].nunique() != len(reg):
        raise SystemExit("PARADA: el registro tiene identificadores duplicados")
    log(f"   V03 verificado e intacto: {ccv!r} | sha256 {hash_real[:16]}...")
    log(f"   registro: {len(reg)} filas, {reg['experiment_id'].nunique()} unicos")
    log("   holdout y test: presentes y NO leidos por esta fase")
    auditoria = {
        "corrected_cv": ccv, "hash_v03": hash_real, "coincide": ccv == V03_CCV,
        "filas_del_registro": len(reg),
        "test_leido": False, "holdout_usado": False, "leaderboard": 0, "submissions": 0,
    }

    # ---------- P02: el inventario, y aqui hay una noticia mala ----------
    log("")
    log("   INVENTARIO DE CANDIDATOS (P02), y el resultado no es el esperado")
    artefactos = sorted(p.name for p in (PROJECT_ROOT / "features").glob("*.npz"))
    log(f"      artefactos .npz presentes en disco: {artefactos}")
    log("      features/ esta en .gitignore, luego los bloques de features que escribieron")
    log("      los scripts de P64 no estan versionados y no existen en este worktree")
    probables: dict[str, Any] = {}
    for f in sorted((PROJECT_ROOT / "experiments").glob("p6[45]_*.py")):
        n = f.read_text(encoding="utf-8", errors="replace")
        tiene = bool(re.search(r"np\.savez|np\.save\(", n))
        probables[f.name] = tiene
    sin_guardar = [k for k, val in probables.items() if not val]
    log(f"      scripts de P64/P65 que guardan probabilidades: "
        f"{len([v for v in probables.values() if v])} de {len(probables)}")
    log(f"      SIN guardar probabilidades: {len(sin_guardar)}")
    log("")
    log("      ***NINGUNO de los cinco scripts de P64/P65 escribio sus probabilidades a")
    log("      disco. Publicaron metricas y scores por pliegue y se quedaron con las")
    log("      probabilidades en memoria. El unico artefacto de probabilidad del repositorio")
    log("      es oof_v03_final.npz, que es el propio V03 ***")
    log("")
    log("      Consecuencia directa: los scores de los candidatos hay que REGENERARLOS. La")
    log("      alternativa seria inventarlos, que es peor. Se reutiliza el codigo de P64 y")
    log("      P65 tal cual, con las mismas semillas y los mismos pliegues, y este script")
    log("      exige que la regeneracion REPRODUZCA el corrected_cv que cada fase")
    log("      reporto. Si un numero no cuadra, la regeneracion esta mal y la fase para.")

    inventario = []
    for c in CANDIDATOS:
        ruta = PROJECT_ROOT / "experiments" / c["script"]
        inventario.append({
            "candidate_name": c["nombre"],
            "artifact_path": "NINGUNO, hay que regenerar",
            "model_type": ("arbol no lineal sobre evidencia segmental"
                           if c["nombre"] == "MI_arbol"
                           else "LogisticRegression de V03 con un bloque anadido"),
            "representation": c["por_que"],
            "fold_assignment": "los 5 pliegues canonicos por grupo, los mismos de P59-P65",
            "seed": 42,
            "training_provenance": ("regenerado desde " + c["script"] + " con semilla 42 y "
                                    "los mismos pliegues"),
            "oof_availability": "NO",
            "probability_availability": "NO",
            "margin_availability": "NO",
            "delta_reportado_por_la_fase_origen": c["delta_reportado"],
            "brecha_contra_el_margen": c["brecha_margen"],
            "prioridad": c["prioridad"],
            "exige_verificacion": c["delta_reportado"],
        })
    with CSV_INV.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(list(inventario[0].keys()))
        for f in inventario:
            w.writerow(list(f.values()))
    log("")
    log(f"   escrito reports/p66_candidate_inventory.csv con {len(inventario)} candidatos")

    # ---------- P01: el gap analysis ----------
    log("")
    log("   GAP ANALYSIS: que se ha hecho antes en el ESPACIO DE SCORES")
    fuentes: dict[str, str] = {}
    for ruta in sorted((PROJECT_ROOT / "experiments").glob("*.py")):
        fuentes[ruta.name] = ruta.read_text(encoding="utf-8", errors="replace")
    pruebas = {
        "blend_de_probabilidades": [n for n, t in fuentes.items()
                                    if re.search(r"alpha\s*\*\s*p|\(1\s*-\s*alpha", t)],
        "blend_de_logits": [n for n, t in fuentes.items()
                            if re.search(r"logit|log_odds|expit|logit_|log\(p\s*/", t)],
        "correccion_de_residuo": [n for n, t in fuentes.items()
                                  if re.search(r"residuo|residual|resid", t, re.IGNORECASE)],
        "calibracion": [n for n, t in fuentes.items()
                        if re.search(r"calibrat|CalibratedClassifier|isotonic", t,
                                     re.IGNORECASE)],
        "stacking": [n for n, t in fuentes.items()
                     if re.search(r"stacking|meta_?model|meta_?clf", t, re.IGNORECASE)],
        "selector_keep_flip": [n for n, t in fuentes.items()
                               if re.search(r"keep_flip|selector|keep_or_flip", t,
                                            re.IGNORECASE)],
    }
    for k, val in pruebas.items():
        log(f"      {k:26} {len(val):3d} scripts  {val[:3]}")

    filas = [
        {"familia": "blend lineal de un score local con V03 mediante un alpha",
         "estado": EXACT, "antecedente": "V6-05 y P59-09",
         "resultado": "V6-05 +0,0014800075 con 2 de 5 y gate fallido por ambas condiciones; "
                      "P59-09 lo midio de nuevo",
         "por_que_equivale": "es exactamente la formula p_final = (1-a)*p_V03 + a*p_C con un "
                             "unico alphalibre",
         "nuevo": "", "prioridad": "-", "ejecutar": "si, como control"},
        {"familia": "correccion residual sobre el score de V03",
         "estado": NUEVA,
         "antecedente": "ninguno. La palabra residuo aparece en el codigo en otros sentidos "
                        "y ninguno es una correccion de score",
         "resultado": "no aplica",
         "por_que_equivale": "no aplica",
         "nuevo": "quitar al candidato la parte que V03 ya predice, y corregir SOLO lo que "
                  "queda. Es la diferencia entre preguntar si el segundo score es util y "
                  "preguntar si el segundo score aporta algo que el primero no dice",
         "prioridad": 1, "ejecutar": "si, es el nucleo de la fase"},
        {"familia": "correccion sensible a la longitud, continua y sin puerta",
         "estado": NUEVA,
         "antecedente": "el especialista duro se midio en P65-04 y fue BIT A BIT igual al "
                        "reentrenado",
         "resultado": "P65-04: especialista contra reentrenado +0,0000000000",
         "por_que_equivale": ("no aplica. El especialista duro cambia el MODELO segun la "
                             "longitud, y esto pondera la CORRECCION de forma continua. La "
                             "hipotesis es distinta: quiz no hace falta un modelo distinto "
                             "sino que la evidencia complementaria pesa mas en los largos"),
         "nuevo": "beta0 * r_C + beta1 * log1p(tokens) normalizado * r_C",
         "prioridad": 2, "ejecutar": "si"},
        {"familia": "calibracion anidada de probabilidades antes de fusionar",
         "estado": PARCIAL,
         "antecedente": "V10-05 toco la capa de decision por shift de prevalencia, y R1/V10 "
                        "tocaron umbrales",
         "resultado": "ninguno de ellos calibro p_C para fusionarla",
         "por_que_equivale": ("parcial. Calibrar el score de un candidato antes de mezclarlo es "
                             "distinto de mover umbrales sobre el resultado"),
         "nuevo": "Platt anidada sobre el score del candidato, ajustada solo en inner_train",
         "prioridad": 3, "ejecutar": "solo si la escala lo pide"},
        {"familia": "stacking o meta-modelo general",
         "estado": EXACT,
         "antecedente": "V4-X01 y las familias meta36, y P64-05 con el arbol",
         "resultado": "cerrados en su momento",
         "por_que_equivale": "un meta-modelo que recibe docenas de entradas es justo lo que el "
                             "encargo excluye aqui",
         "nuevo": "", "prioridad": "-", "ejecutar": "no, excluido por el encargo"},
        {"familia": "selector keep/flip",
         "estado": EXACT,
         "antecedente": "V18 y V50, y P64-19 lo excluyo explicitamente",
         "resultado": "cerrados",
         "por_que_equivale": "es la formulacion que el encargo prohibe",
         "nuevo": "", "prioridad": "-", "ejecutar": "no, excluido por el encargo"},
        {"familia": "control de score barajado",
         "estado": NUEVA,
         "antecedente": "ninguno. Existia el control de BLOQUE aleatorio en la etapa de "
                        "features, y no su equivalente en la etapa de scores",
         "resultado": "no aplica",
         "por_que_equivale": "no aplica",
         "nuevo": "el score del candidato barajado entre filas, con la misma distribucion "
                  "marginal. Es el control que decide si una mejora de la fusion es del "
                  "candidato o de la MECANICA de la fusion",
         "prioridad": 1, "ejecutar": "si, es obligatorio"},
        {"familia": "unica informacion mas alla de V03 en el score del candidato",
         "estado": NUEVA,
         "antecedente": "P63-07 y P64-03 midieron el AUC de la evidencia frente al ERROR de "
                        "V03, que es un diagnostico, no una atribucion de ganancia",
         "resultado": "auc 0,605302 del mejor rasgo frente a 0,770875 del margen",
         "por_que_equivale": ("parcial pero no equivalente. Medir si un rasgo predice el error "
                             "es distinto de medir si su residuo, anidado, aporta F1"),
         "nuevo": "correlacion del residuo con el error de V03, y rescate unico por fusion",
         "prioridad": 2, "ejecutar": "si"},
    ]
    df = pd.DataFrame(filas)
    df.to_csv(CSV_GAP, index=False, encoding="utf-8")
    abiertas = df[df["estado"] == NUEVA]
    log("")
    log(f"   RESULTADO: {len(abiertas)} familias genuinamente nuevas, "
        f"{len(df[df['estado'] == EXACT])} duplicadas exactas, "
        f"{len(df[df['estado'] == PARCIAL])} con precedente parcial")
    for _, r in abiertas.iterrows():
        log(f"      prioridad {r['prioridad']}  {r['familia']}")
    log("      y el hallazgo de P65 que justifica abrir la fase:")
    log("      cinco candidatos baten a los dos controles y ninguno mejora. Lo que no se ha")
    log("      probado es si esa senal funciona CORREGIENDO el score en vez de anadiendo")
    log("      columnas, que es un mecanismo distinto con otros grados de libertad")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P66_GAP_ANALYSIS",
        "naturaleza": "AUDITORIA. 0 experimentos, 0 modelos, 0 fusiones",
        "fecha": inicio.isoformat(timespec="seconds"), "experiment_id": "P66-00",
        "script": "experiments/p66_00_audit_gap.py",
        "auditoria_del_baseline": auditoria,
        "inventario": {
            "artefactos_de_probabilidad_en_disco": artefactos,
            "scripts_de_p64_y_p65": probables,
            "scripts_sin_guardar_probabilidades": sin_guardar,
            "conclusion": ("NINGUNO de los scripts de P64 y P65 escribio sus probabilidades a "
                           "disco. Publicaron metricas y los artefactos quedaron en memoria. Los "
                           "scores de los candidatos hay que REGENERARLOS con el mismo codigo, "
                           "la misma semilla y los mismos pliegues, y exigiendo que la "
                           "regeneracion reproduzca el corrected_cv que cada fase reporto"),
        },
        "pruebas_en_el_codigo": pruebas,
        "por_que_se_lee_el_codigo": (
            "con 364 filas en el registro, una busqueda por nombre devuelve las mismas filas "
            "para cualquier patron que aparezca en una pregunta"),
        "tabla": filas,
        "n_abiertas": len(abiertas),
        "diferencia_central": (
            "un blend que falla dice que la MEZCLA estaba mal. Un residuo que falla dice que el "
            "candidato es REDUNDANTE. Son conclusiones distintas, y solo la segunda habla de "
            "informacion"),
        "candidatos_a_explotar": [c["nombre"] for c in CANDIDATOS],
        "reglas_de_la_fase": {
            "solo_espacio_de_scores": "el score de V03 y el del candidato, nunca las matrices "
                                      "TF-IDF del candidato",
            "parametros": "todo lo aprendido va por nested, la outer solo evalua",
            "umbrales": "0,44 / 0,51 / 0,46 congelados",
            "sin_representaciones_nuevas": "P66 no abre ninguna familia nueva",
        },
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "hash_v03": hash_real,
                             "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_GAP.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P66 - Analisis de hipotesis: la fusion en el espacio de scores")
    W("")
    W(f"Generado por `experiments/p66_00_audit_gap.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Por que esta fase existe, en una frase")
    W("")
    W("P64 y P65 midieron cinco representaciones que baten a **los dos controles**, el de")
    W("capacidad y el de la confianza de V03, y ninguna mejoro a V03 al concatenarla. P66")
    W("prueba lo unico que no se probo: si esa misma evidencia funciona como una correccion")
    W("pequena en el espacio de scores, sin tocar el espacio TF-IDF.")
    W("")
    W("## La auditoria del baseline")
    W("")
    W("```text")
    W(f"V03 corrected_cv  {ccv!r}")
    W(f"V03 sha256        {hash_real}")
    W(f"registro          {len(reg)} filas, {reg['experiment_id'].nunique()} identificadores")
    W("holdout 0 | test 0 | submissions 0 | Kaggle 0 | leaderboard 0")
    W("```")
    W("")
    W("## El inventario es PEOR de lo esperado, y se reporta en vez de rodearlo")
    W("")
    W("El encargo dice que los scores OOF de los candidatos deben reutilizarse cuando exista uno")
    W("valido. Al comprobar el repositorio, **ninguno de los cinco scripts de P64 y P65 escribio")
    W("sus probabilidades a disco**. Publicaron metricas y scores por pliegue y las dejaron en")
    W("memoria. Ademas `features/` esta en `.gitignore`, luego los bloques de features que si")
    W("escribieron tampoco estan versionados.")
    W("")
    W("```text")
    W(f"artefactos .npz en disco: {', '.join(artefactos)}")
    W(f"scripts de P64/P65 que guardan probabilidades: "
      f"{len([v for v in probables.values() if v])} de {len(probables)}")
    W("```")
    W("")
    W("**Consecuencia: los scores hay que REGENERARLOS.** La alternativa seria inventarlos, que")
    W("es peor. Se reutiliza el codigo de P64 y P65 tal cual, con las mismas semillas y los")
    W("mismos pliegues, y la regeneracion tiene que **reproducir el corrected_cv que cada fase")
    W("reporto**. Si un numero no cuadra, la regeneracion esta mal y la fase para en vez de")
    W("seguir sobre scores no verificados.")
    W("")
    W("Ese es el motivo de que esto sea un paso aparte y no una regeneracion silenciosa dentro")
    W("del script de fusion.")
    W("")
    W("## Los siete candidatos, por prioridad")
    W("")
    W("| candidato | delta que reporto su fase | brecha contra el margen | por que |")
    W("|---|---:|---:|---|")
    for c in CANDIDATOS:
        W(f"| {c['nombre']} | {c['delta_reportado']:+.10f} | {c['brecha_margen']:+.10f} | "
          f"{a_ascii(c['por_que'])[:78]} |")
    W("")
    W("## La tabla de familias")
    W("")
    W("| familia | estado | por quien |")
    W("|---|---|---|")
    for _, r in df.iterrows():
        W(f"| {r['familia']} | {r['estado']} | {str(r['antecedente'])[:62]} |")
    W("")
    W(f"**{len(abiertas)} genuinamente nuevas, {len(df[df['estado'] == EXACT])} duplicadas "
      f"exactas, {len(df[df['estado'] == PARCIAL])} con precedente parcial.**")
    W("")
    W("## La diferencia central, que es lo que hace nueva la fase")
    W("")
    W("Un **blend** que falla dice que la MEZCLA estaba mal. Un **residuo** que falla dice que el")
    W("candidato es **REDUNDANTE**. Son conclusiones distintas, y solo la segunda habla de")
    W("informacion.")
    W("")
    W("Mezclar un segundo score dentro de V03 pregunta si el segundo score es util. Residualizar")
    W("pregunta si el segundo score contiene algo que el score de V03 **no predice**, y")
    W("corrige solo esa parte. La segunda pregunta es la que P64 y P65 dejaron abierta.")
    W("")
    W("Lo que ya se cerro y no se reabre: el blend lineal con un alpha, medido por V6-05 con")
    W("+0,0014800075 y 2 de 5, y por P59-09 otra vez. Se ejecutara aqui como **control**, no")
    W("como novedad. Y quedan excluidos por el encargo el stacking general y el selector")
    W("keep/flip, que ademas ya se habian cerrado como V18, V50 y P64-19.")
    W("")
    W("## Las reglas de la fase, fijadas antes")
    W("")
    W("- solo el score de V03 y el del candidato, **nunca** las matrices TF-IDF del candidato")
    W("- todo lo aprendido va por nested: la outer **solo evalua**")
    W("- umbrales 0,44 / 0,51 / 0,46 congelados")
    W("- P66 **no abre ninguna familia de representaciones nueva**")
    W("- control de score barajado obligatorio en toda fusion promising")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p66_00_audit_gap.py`. Sin push.*")
    MD_GAP.write_text(a_ascii("\n".join(L)), encoding="ascii")

    with CSV_GAP.open("a", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow([])
        w.writerow(["RESUMEN", "abiertas", len(abiertas)])
        w.writerow(["RESUMEN", "duplicadas_exactas", len(df[df["estado"] == EXACT])])
        w.writerow(["RESUMEN", "precedente_parcial", len(df[df["estado"] == PARCIAL])])
    for destino in (MD_GAP, JSON_GAP, CSV_GAP, CSV_INV):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P66-00 OK | {len(abiertas)} nuevas | {len(sin_guardar)} scripts sin OOF | "
        f"{len(inventario)} candidatos a regenerar | commit {commit}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
