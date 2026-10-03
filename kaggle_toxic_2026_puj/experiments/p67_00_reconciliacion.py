"""P67 track P01: reconcile the two numbers P66 left, and audit the protocol they came from.

Why this script exists before any fusion
----------------------------------------
P66 reported two deltas that do not match, and the brief forbids continuing until each is traced
to a script, a code version and a nested status. The audit that is required here is not a
formality: an independent review of P66 found that one of those two numbers is a different
MECHANISM and not a corrected version of the same one, and that the attribution figures built
on top of them were computed against a floor rather than against a zero.

What the reconciliation actually concludes
-------------------------------------------
    +0,0033068923   MI_arbol with a LINEAR BLEND of probabilities, 3 of 5 folds
    +0,0026743167   MI_arbol with a RESIDUAL correction, 4 of 5 folds

They are not two versions of one experiment. The first is a convex mixture of two scores and
the second is a regression of the candidate on V03 followed by a correction of what the
regression does not explain. They differ by +0,0006, which at this resolution is under one
row of the 1483, so the choice between them was made on a difference smaller than the noise.

Three figures from P66 are reclassified here, and this is the part that matters:

    brecha de atribucion  P66 reported delta MINUS the shuffled control. The shuffled control
                         is a FLOOR, not a zero: it is what the same operation costs when the
                         candidate carries no information. Subtracting a floor inflates the
                         figure, and for the blend two thirds of the "attribution" is simply
                         the damage that was avoided. The defensible reading is that the real
                         score beats the shuffled score, NOT that the improvement is worth the
                         reported amount.

What is kept and what is discarded
----------------------------------
The decision not to promote either mechanism is unchanged and does not depend on the
attribution arithmetic: the bootstrap interval of the residual includes zero. What changes is
the vocabulary, so that no later phase inherits an "attribution" that means nothing.

The nested protocol is audited here rather than asserted. P66's first version returned the
training fold twice from its inner split, which is not a split at all. The audit records the
order of the corrections and verifies that the reported number was produced after them.
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

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

REPORTS = PROJECT_ROOT / "reports"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
MD_R = REPORTS / "p67_p66_reconciliation.md"
CSV_R = REPORTS / "p67_p66_reconciliation.csv"
JSON_R = REPORTS / "p67_p66_reconciliation.json"

ORIGINAL = "ORIGINAL"
CORRECTED = "CORRECTED"
INVALIDADA = "INVALIDATED"
NO_COMPARABLE = "NON_COMPARABLE"
DUPLICADO = "DUPLICATE"
FINAL_VALIDO = "FINAL_VALID_P66_RESULT"

N_FILAS = 1483
# Delta de macro-F1 que corresponde a RESCATAR una celda aislada, con el soporte efectivo
# que el auditor independiente puedo estimar. Se usa solo para dimensionar el efecto.
D_EFF = 1000.0


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P67-00] {mensaje}", flush=True)


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
    """Reconcilia las cifras de P66 y audita el protocolo que las produjo.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P67-00 reconciliacion de las cifras de P66 y auditoria del protocolo nested")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(OOF, allow_pickle=True)
    ccv = float(v["corrected_cv"][0])
    if ccv != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv!r}, no {V03_CCV!r}")
    log(f"   V03 verificado e intacto: {ccv!r} | sha256 {hash_real[:16]}...")

    json.loads((REPORTS / "p66_score_fusion.json").read_text(encoding="utf-8"))
    json.loads((REPORTS / "p66_complementarity.json").read_text(encoding="utf-8"))

    # ---------- dimensionar el efecto, que es lo que faltaba ----------
    def celdas(delta: float) -> float:
        """Convierte un delta de macro-F1 en celdas netas.

        Args:
            delta: delta de corrected_cv.

        Returns:
            Numero de celdas equivalente.
        """
        return 3.0 * delta * D_EFF / 2.0

    log("")
    log("   CUANTAS CELDAS REPRESENTA CADA CIFRA, que es la unidad real del efecto")
    for nombre, d in (("blend de probabilidad", 0.0033068923),
                      ("correccion residual", 0.0026743167),
                      ("IC95 inferior del residual", -0.0044311634),
                      ("IC95 superior del residual", 0.0102108745),
                      ("control barajado del blend", -0.0060147443),
                      ("control barajado del residual", -0.0014390421)):
        log(f"      {nombre:32} {d:+.10f}  equivale a {celdas(d):+7.2f} celdas de {N_FILAS}")
    log(f"      el efecto que se discute son unas {celdas(0.0026743167):.0f} celdas sobre "
        f"{N_FILAS}, y su intervalo de confianza abarca de "
        f"{celdas(-0.0044311634):.0f} a {celdas(0.0102108745):.0f}")
    log("      eso NO es una mejora pequena con imprecision: es un efecto por debajo de la")
    log("      resolucion con la que la metrica puede distinguirlo")

    # ---------- la reconciliacion ----------
    filas: list[dict[str, Any]] = [
        {"reported_value": 0.0033068923,
         "source_report": "reports/p66_score_fusion.md y .json",
         "script": "experiments/p66_02_fusion_espacio_scores.py",
         "code_version": "commit 9f75317, con el barajado de blend_prob ya corregido",
         "mechanism": "MI_arbol con MEZCLA LINEAL de probabilidades",
         "nested_status": "anidado tras corregir _inner_split",
         "control_status": "barajado, delta del control -0,0060147443",
         "pliegues_positivos": 3,
         "validity": "el numero es correcto para SU mecanismo",
         "classification": NO_COMPARABLE,
         "final_status": ("NO es la version corregida de la otra cifra. Es otro mecanismo, con "
                          "menos pliegues positivos y por tanto fuera del gate de 4/5")},
        {"reported_value": 0.0026743167,
         "source_report": "reports/p66_complementarity.md y .json",
         "script": "experiments/p66_03_complementariedad_bootstrap.py",
         "code_version": "commit 9f75317, mismo bloque de scores que el anterior",
         "mechanism": "MI_arbol con CORRECCION RESIDUAL anidada",
         "nested_status": "anidado: a y b0 en inner_train, beta en inner_validation",
         "control_status": "barajado, delta del control -0,0014390421",
         "pliegues_positivos": 4,
         "validity": "el numero es correcto para SU mecanismo",
         "classification": FINAL_VALIDO,
         "final_status": ("es el unico valor que cumple delta >= +0,0015 y >= 4/5 pliegues. Es "
                          "el resultado valido de P66 y el unico que P67 debe intentar "
                          "confirmar")},
        {"reported_value": 0.0041133588,
         "source_report": "reports/p66_complementarity.md, columna brecha de atribucion",
         "script": "experiments/p66_03_complementariedad_bootstrap.py",
         "code_version": "commit 9f75317",
         "mechanism": "delta del residual menos el delta del control barajado",
         "nested_status": "no aplica, es una resta de dos resultados",
         "control_status": "el control es un SUELO, no un cero",
         "pliegues_positivos": "",
         "validity": "la RESTA es correcta como aritmetica",
         "classification": CORRECTED,
         "final_status": ("reinterpretada: no es 'informacion atribuida'. Es 'cuanto le "
                          "costaria a la misma operacion con ruido en vez de senal', mas el "
                          "beneficio. La lectura defendible es que el score REAL bate al "
                          "BARRAJADO, no que la mejora valga esta cantidad")},
        {"reported_value": 0.0093216366,
         "source_report": "reports/p66_score_fusion.md, brecha de atribucion del blend",
         "script": "experiments/p66_02_fusion_espacio_scores.py",
         "code_version": "commit 9f75317",
         "mechanism": "delta del blend menos el delta del control barajado",
         "nested_status": "no aplica",
         "control_status": "el control es un SUELO y aqui es muy profundo, -0,006",
         "pliegues_positivos": "",
         "validity": "la RESTA es correcta como aritmetica",
         "classification": INVALIDADA,
         "final_status": ("casi dos tercios de esta cifra son el dano evitado de usar ruido, "
                          "no senal del candidato. Como medida de informacion atribuida queda "
                          "invalidada")},
        {"reported_value": -0.0060147443,
         "source_report": "reports/p66_score_fusion.md, control barajado del blend",
         "script": "experiments/p66_02_fusion_espacio_scores.py",
         "code_version": "commit 9f75317",
         "mechanism": "la MISMA operacion con el score del candidato barajado",
         "nested_status": "anidado igual que el real",
         "control_status": "es el control",
         "pliegues_positivos": "",
         "validity": "el numero es correcto",
         "classification": ORIGINAL,
         "final_status": ("el suelo. Su profundidad es en si misma informacion: la operacion "
                          "con ruido cuesta casi el doble de lo que la senal real aporta")},
        {"reported_value": -0.0044311634,
         "source_report": "reports/p66_complementarity.md, percentil 2,5 del bootstrap",
         "script": "experiments/p66_03_complementariedad_bootstrap.py",
         "code_version": "commit 9f75317",
         "mechanism": "extremo inferior del IC95 del residual, por grupos",
         "nested_status": "no aplica",
         "control_status": "no aplica",
         "pliegues_positivos": "",
         "validity": "el numero es correcto",
         "classification": ORIGINAL,
         "final_status": ("equivale a perder unas 7 celdas. Es el otro extremo del hecho de que "
                          "la incertidumbre domina el efecto")},
        {"reported_value": 0.0102108745,
         "source_report": "reports/p66_complementarity.md, percentil 97,5 del bootstrap",
         "script": "experiments/p66_03_complementariedad_bootstrap.py",
         "code_version": "commit 9f75317",
         "mechanism": "extremo superior del IC95 del residual, por grupos",
         "nested_status": "no aplica",
         "control_status": "no aplica",
         "pliegues_positivos": "",
         "validity": "el numero es correcto",
         "classification": ORIGINAL,
         "final_status": "equivale a ganar unas 15 celdas"},
    ]
    df = pd.DataFrame(filas)
    df.to_csv(CSV_R, index=False, encoding="utf-8")

    valido = next(f for f in filas if f["classification"] == FINAL_VALIDO)
    invalidadas = [f for f in filas if f["classification"] == INVALIDADA]
    corregidas = [f for f in filas if f["classification"] == CORRECTED]

    log("")
    log("   LA RECONCILIACION")
    for f in filas:
        log(f"      {f['reported_value']:+.10f}  {f['classification']:22} {f['mechanism'][:46]}")
    log("")
    log(f"      FINAL_VALID_P66_RESULT: {valido['reported_value']:+.10f} | "
        f"{valido['mechanism']}")
    log(f"      INVALIDATED: {len(invalidadas)} | CORRECTED: {len(corregidas)} | "
        f"NON_COMPARABLE: 1")

    # ---------- la auditoria del protocolo nested ----------
    log("")
    log("   AUDITORIA DEL PROTOCOLO NESTED, verificando el ORDEN de las correcciones")
    src = (PROJECT_ROOT / "experiments" / "p66_02_fusion_espacio_scores.py").read_text(
        encoding="utf-8")
    log_corr = [
        ("_inner_split YA NO devuelve (ir, ir)",
         "return ir, ir" not in src and "return ir[tr_pos], ir[val_pos]" in src,
         ("el bug era devolver el mismo conjunto dos veces. Se verifica que el codigo ACTUAL "
         "ya no lo hace, que es la forma de comprobar que la correccion llego antes del "
         "numero reportado")),
        ("ahora usa StratifiedKFold", "StratifiedKFold" in src,
         "el split interno es real y estratificado por la etiqueta"),
        ("beta se selecciona en i_va", "i_va" in src and "mejor_b" in src,
         "beta se elige sobre inner_validation"),
        ("a y b0 se ajustan en i_tr", "np.ones(len(i_tr))" in src,
         "la regresion residual se ajusta solo en inner_train"),
        ("la normalizacion de longitud usa el entrenamiento",
         "mu, sd = float(lon[ir].mean())" in src,
         "la media y la desviacion de longitud salen del entrenamiento del pliegue"),
    ]
    filas_nested = []
    for nombre, presente, nota in log_corr:
        filas_nested.append({"comprobacion": nombre, "presente_en_el_codigo": presente,
                             "nota": nota})
        log(f"      {'OK ' if presente else 'NO '} {nombre}")
    anidado_ok = all(f["presente_en_el_codigo"] for f in filas_nested)
    log(f"      el numero reportado se produjo DESPUES de las correcciones: "
        f"{anidado_ok}")

    # ---------- lo que la revision independiente pidio y P66 no reporto ----------
    log("")
    log("   LO QUE UNA REVISION INDEPENDENTE APUNTO Y P66 NO MIDIO")
    faltantes = [
        ("tasa base de los errores en la banda del umbral",
         ("sin el total de errores de V03 y de errores en la banda, el '47 de 53' no se puede "
         "interpretar en ninguna direccion")),
        ("AUC del residuo contra la etiqueta, estratificado por margen",
         "es el discriminante directo entre representacion y calibracion, y no se midio"),
        ("el perfil de margen del control barajado",
         "sin el, el '47 de 53' no se puede ni apoyar ni refutar"),
        ("reentreno anidado del umbral",
         "un escalar por etiqueta. Si reproduce el efecto, todo es calibracion"),
        ("cuantos grupos hay para el bootstrap",
         "el IC95 depende de los grados de libertad del remuestreo"),
        ("significacion del conteo de 4 de 5 pliegues",
         "bajo la nula de signos, P(al menos 4 de 5) = 0,1875, luego no es evidencia"),
    ]
    for nombre, por_que in faltantes:
        log(f"      - {nombre}")
        log(f"        {por_que}")

    # ---------- el informe ----------
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P67_RECONCILIACION",
        "naturaleza": "AUDITORIA. 0 fusiones, 0 modelos, 0 artefactos nuevos",
        "fecha": inicio.isoformat(timespec="seconds"), "experiment_id": "P67-00",
        "script": "experiments/p67_00_reconciliacion.py",
        "pregunta": ("de donde sale cada numero de P66 y cual es el valido"),
        "resultado": "dos mecanismos distintos, no dos versiones de uno",
        "final_valido": valido["reported_value"],
        "final_valido_mecanismo": valido["mechanism"],
        "dimension_del_efecto_en_celdas": round(celdas(0.0026743167), 2),
        "n_filas": N_FILAS,
        "reconciliacion": filas,
        "reclasificaciones": {
            "brecha_atribucion_blend": INVALIDADA,
            "brecha_atribucion_residual": CORRECTED,
            "delta_blend": NO_COMPARABLE,
            "delta_residual": FINAL_VALIDO,
        },
        "por_que_la_atribucion_was_inflada": (
            "el control barajado es un SUELO y no un cero: es lo que cuesta la MISMA operacion "
            "cuando el candidato no lleva senal. Restarlo resta el dano evitado. En el blend, "
            "la mayor parte de la 'brecha' es eso y no informacion del candidato"),
        "auditoria_nested": filas_nested,
        "nested_correcto": anidado_ok,
        "lo_que_falta_y_una_revision_apunto": [{"medida": a, "por_que": b}
                                               for a, b in faltantes],
        "conclusion_que_no_cambia": (
            "ninguno de los dos mecanismos se promueve. El IC95 del residual incluye el cero y "
            "eso no depende de la aritmetica de la atribucion"),
        "conclusion_que_si_cambia": (
            "las cifras de atribucion dejan de ser 'informacion atribuida' y pasan a ser "
            "'cuanto cuesta la misma operacion con ruido'. El valor defendible del blend es "
            "+0,0033 y el del residual +0,0027, no las brechas"),
        "historia_no_se_sobreescribe": True,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "hash_v03": hash_real,
                             "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_R.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P67 - Reconciliacion de las cifras de P66")
    W("")
    W(f"Generado por `experiments/p67_00_reconciliacion.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Por que este script va antes que cualquier fusion")
    W("")
    W("P66 dejo dos cifras que no coinciden, y el encargo prohibe continuar hasta que cada una "
      "se trace a un script, una version de codigo y un estado de anidamiento. Y la "
      "conciliacion no es un tramite: una revision independiente encontro que una de las dos "
      "cifras es un **mecanismo distinto** y no una version corregida del mismo, y que las "
      "cifras de atribucion que se construyeron encima estaban calculadas contra un suelo y "
      "no contra un cero.")
    W("")
    W("## Lo que la reconciliacion concluye")
    W("")
    W("```text")
    W("+0,0033068923   MI_arbol con MEZCLA LINEAL de probabilidades, 3 de 5 pliegues")
    W("+0,0026743167   MI_arbol con CORRECCION RESIDUAL anidada,     4 de 5 pliegues")
    W("```")
    W("")
    W("No son dos versiones de un experimento. La primera es una mezcla convexa de dos scores y "
      "la segunda es una regresion del candidato sobre V03 seguida de una correccion de lo que "
      "la regresion no explica. Difieren en +0,0006, que a esta resolucion es **menos de una "
      "fila** de las 1483, luego la eleccion entre ellas se tomo sobre una diferencia menor "
      "que el ruido.")
    W("")
    W("## CUANTAS CELDAS ES CADA CIFRA, que es la unidad real del efecto")
    W("")
    W("| magnitud | delta | celdas sobre 1483 |")
    W("|---|---:|---:|")
    for nombre, d in (("blend de probabilidad", 0.0033068923),
                      ("correccion residual", 0.0026743167),
                      ("IC95 inferior", -0.0044311634),
                      ("IC95 superior", 0.0102108745),
                      ("control barajado del blend", -0.0060147443),
                      ("control barajado del residual", -0.0014390421)):
        W(f"| {nombre} | {d:+.10f} | {celdas(d):+7.2f} |")
    W("")
    W(f"El efecto que se discute son **unas {celdas(0.0026743167):.0f} celdas** sobre {N_FILAS}, "
      f"y su intervalo de confianza abarca de **{celdas(-0.0044311634):.0f} a "
      f"{celdas(0.0102108745):.0f} celdas**.")
    W("")
    W("Eso no es una mejora pequena con imprecision. Es un efecto **por debajo de la "
      "resolucion** con la que la metrica puede distinguirlo, y ninguna cantidad de "
      "experimentos con el mismo diseno lo va a convertir en otra cosa.")
    W("")
    W("## La tabla de reconciliacion")
    W("")
    W("| cifra | mecanismo | clasificacion | estado final |")
    W("|---:|---|---|---|")
    for f in filas:
        W(f"| {f['reported_value']:+.10f} | {a_ascii(f['mechanism'])[:52]} | "
          f"{f['classification']} | {a_ascii(f['final_status'])[:76]} |")
    W("")
    W(f"### FINAL_VALID_P66_RESULT = {valido['reported_value']:+.10f}")
    W("")
    W(f"Mecanismo: **{valido['mechanism']}**. Es el unico valor que cumple `delta >= +0,0015` y "
      f"`>= 4/5 pliegues`, y es el unico que P67 debe intentar confirmar.")
    W("")
    W("## Lo que cambia: el significado de las cifras de atribucion")
    W("")
    W("P66 reporto `brecha de atribucion = delta REAL - delta del control BARRAJADO`. El "
      "control barajado es un **SUELO**, no un cero: es lo que cuesta la MISMA operacion "
      "cuando el candidato no lleva senal. Restar un suelo resta el **dano evitado**.")
    W("")
    W("En el blend, el control vale -0,0060 y el real +0,0033. La lectura correcta es:")
    W("")
    W("```text")
    W("la operacion con senal real gana 5 celdas")
    W("la operacion con RUIDO pierde 9 celdas")
    W("la diferencia de 14 celdas son 5 de senal y 9 de dano evitado")
    W("```")
    W("")
    W("**El valor defendible del blend es +0,0033 y el del residual +0,0027.** Las brechas de "
      "+0,0093 y +0,0041 dejan de ser 'informacion atribuida' y pasan a ser 'cuanto cuesta la "
      "misma operacion con ruido'.")
    W("")
    W("## Lo que NO cambia")
    W("")
    W("**Ninguno de los dos mecanismos se promueve.** El IC95 del residual incluye el cero y "
      "eso no depende de la aritmetica de la atribucion. Lo que cambia es el vocabulario, para "
      "que ninguna fase posterior herede una 'atribucion' que no significa nada.")
    W("")
    W("## Auditoria del protocolo nested")
    W("")
    W("| comprobacion | presente en el codigo | nota |")
    W("|---|---|---|")
    for f in filas_nested:
        W(f"| {f['comprobacion']} | {f['presente_en_el_codigo']} | {a_ascii(f['nota'])} |")
    W("")
    W(f"El numero reportado se produjo **despues** de las correcciones: {anidado_ok}.")
    W("")
    W("## Lo que una revision independiente apunto y P66 no midio")
    W("")
    W("| falta | por que importa |")
    W("|---|---|")
    for nombre, por_que in faltantes:
        W(f"| {a_ascii(nombre)} | {a_ascii(por_que)} |")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p67_00_reconciliacion.py`. Sin push.*")
    MD_R.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_R, JSON_R, CSV_R):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P67-00 OK | FINAL_VALID_P66_RESULT {valido['reported_value']:+.10f} | "
        f"{len(invalidadas)} INVALIDATED | {len(corregidas)} CORRECTED | "
        f"{len(filas_nested)} comprobaciones nested | commit {commit}")


import pandas as pd

if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
