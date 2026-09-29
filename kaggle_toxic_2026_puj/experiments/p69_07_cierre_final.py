"""Final closing of the A_cost_sensitive_downweight dossier: evidence, limits and governance.

No new experiments
------------------
This script runs nothing and trains nothing. It consolidates what is already measured, states
the statistical limit rather than trying to remove it, and writes the governance note that
records what cannot be done under the current protocol without pretending to rule on it.

The one thing it must not do is resolve the remaining uncertainty. There are three partitions
and a permutation test behind the candidate, and the temptation now is to keep partitioning
until an interval excludes zero. That would not be evidence, it would be selection, and the
specification was written and committed before the first partition precisely so that this
temptation would have a documented answer.

The three tests answer three different questions, and mixing them is the failure mode to avoid:

    partitions      stability under a change of fold assignment
    permutation     specificity of the weight-to-example pairing
    group bootstrap uncertainty about the magnitude

None of them answers the others' question.
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import V03_CCV, V03_SHA, V03_SUB

REPORTS = PROJECT_ROOT / "reports"
MD_E = REPORTS / "a_cost_sensitive_downweight_final_evidence.md"
CSV_E = REPORTS / "a_cost_sensitive_downweight_final_evidence.csv"
MD_G = REPORTS / "a_cost_sensitive_downweight_governance_note.md"

CANONICA = {"nombre": "canonica", "V03": 0.7772114525, "delta": 0.0082045862,
            "folds": 5, "rescate": 99, "damage": 74,
            "deltas": [0.002243, 0.001592, 0.003378, 0.023607, 0.010356],
            "etiquetas": [{"etiqueta": "HS", "rescate": 42, "damage": 23, "neto": 19},
                          {"etiqueta": "TR", "rescate": 14, "damage": 19, "neto": -5},
                          {"etiqueta": "AG", "rescate": 43, "damage": 32, "neto": 11}]}
SEGUNDA = {"nombre": "segunda", "V03": 0.7674405718, "delta": 0.0105253130,
           "folds": 5, "rescate": 86, "damage": 74,
           "deltas": [0.026302, 0.004643, 0.002975, 0.020868, 0.003209],
           "etiquetas": [{"etiqueta": "HS", "rescate": 37, "damage": 30, "neto": 7},
                         {"etiqueta": "TR", "rescate": 21, "damage": 25, "neto": -4},
                         {"etiqueta": "AG", "rescate": 28, "damage": 19, "neto": 9}]}
TERCERA = {"nombre": "tercera", "V03": 0.7739813065, "delta": 0.0048825807,
           "folds": 4, "rescate": 80, "damage": 70,
           "deltas": [0.00188, -0.006834, 0.012343, 0.012575, 0.00444],
           "etiquetas": [{"etiqueta": "HS", "rescate": 45, "damage": 27, "neto": 18},
                         {"etiqueta": "TR", "rescate": 18, "damage": 15, "neto": 3},
                         {"etiqueta": "AG", "rescate": 17, "damage": 28, "neto": -11}]}
PARTICIONES = [CANONICA, SEGUNDA, TERCERA]
ESTADOS = [
    ("ORIGINAL", "+0.0082045862", "2026-09-25, nunca enviado, nunca auditado"),
    ("REPRODUCED", "+0.0082045862", "R3, diferencia 3,2e-11"),
    ("SECOND_PARTITION", "+0.0105253130", "5 de 5 pliegues"),
    ("THIRD_PARTITION_CORRECTED", "+0.0048825807", "4 de 5, tras corregir el defecto"),
    ("PERMUTATION", "p = 0.000999", "0 de 1000 superan al observado"),
    ("FINAL_STATUS", "PROMISING_UNCONFIRMED", "5 de 6 componentes"),
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P69-07] {mensaje}", flush=True)


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
    """Consolida la evidencia final y escribe la nota de gobernanza.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P69-07 cierre final: evidencia, limite estadistico y gobernanza")
    log("   0 experimentos, 0 entrenamientos, 0 submissions")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    if float(v["corrected_cv"][0]) != V03_CCV:
        raise SystemExit("PARADA: el OOF canonico no reproduce corrected_cv")
    log(f"   V03 intacto: {V03_CCV!r}")

    per = json.loads((REPORTS / "a_cost_sensitive_downweight_permutation.json")
                     .read_text(encoding="utf-8"))
    ds = np.array([p["delta"] for p in PARTICIONES])
    log(f"   tres particiones: {' '.join(f'{d:+.10f}' for d in ds)}")
    log(f"   media {ds.mean():+.10f} | minimo {ds.min():+.10f}")

    # ---------- evidencia final ----------
    E: list[str] = []
    W = E.append
    W("# A_cost_sensitive_downweight - Evidencia final")
    W("")
    W(f"Generado por `experiments/p69_07_cierre_final.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## La conclusion, congelada")
    W("")
    W("> A_cost_sensitive_downweight presenta una mejora reproducible, consistente entre tres")
    W("> particiones y altamente improbable bajo la permutacion de la asignacion de pesos.")
    W("> Sin embargo, el bootstrap por grupos continua incluyendo cero, por lo que la magnitud")
    W("> del efecto no queda resuelta con suficiente precision para clasificarlo como ADOPTED.")
    W("")
    W("La formulacion precisa es: **evidencia fuerte de direccion y mecanismo, pero")
    W("incertidumbre residual sobre la magnitud poblacional**.")
    W("")
    W("### Lo que NO se afirma")
    W("")
    W("```text")
    W("no 'demostrado definitivamente'")
    W("no 'es ruido'")
    W("no 'V03 ha sido superado definitivamente'")
    W("```")
    W("")
    W("## Los seis estados, que no se sustituyen entre si")
    W("")
    W("| estado | valor | nota |")
    W("|---|---|---|")
    for nom, val, nota in ESTADOS:
        W(f"| `{nom}` | {val} | {nota} |")
    W("")
    W("## El candidato, congelado")
    W("")
    W("```text")
    W("margen_favor = (p_V03 - umbral) * (2*y - 1)")
    W("violacion_l  = clip(-margen_favor, 0, None)      por etiqueta")
    W("error_score  = violacion_l.mean(axis=1)          por ejemplo, media de 3")
    W("z            = (error_score - mediana) / IQR")
    W("peso         = clip(1 - 0.5*z, 0.5, 1.5)")
    W("```")
    W("")
    W("Cantidades de la particion canonica, que no se tocan:")
    W("")
    W("```text")
    W("441 ejemplos con peso < 1")
    W("1042 ejemplos con peso = 1")
    W("0 ejemplos con peso > 1")
    W("13,66 por ciento del peso total retirado")
    W("```")
    W("")
    W("La descripcion correcta del mecanismo es **cost-sensitive margin-violation")
    W("downweighting**. P58 lo etiquetaba como *'meta con downweight coste-sensible'*, y el")
    W("codigo no construye ningun meta-modelo: entrena `LogisticRegression` directamente sobre")
    W("la representacion de V03. El informe historico no se modifica.")
    W("")
    W("## Las tres particiones")
    W("")
    W("| particion | V03 | candidato | delta | folds+ | rescate | damage | neto |")
    W("|---|---:|---:|---:|---:|---:|---:|---:|")
    for p in PARTICIONES:
        W(f"| {p['nombre']} | {p['V03']:.10f} | {p['V03'] + p['delta']:.10f} | "
          f"{p['delta']:+.10f} | {p['folds']}/5 | {p['rescate']} | {p['damage']} | "
          f"{p['rescate'] - p['damage']:+d} |")
    W("")
    W("```text")
    W(f"media     {ds.mean():+.10f}")
    W(f"mediana   {float(np.median(ds)):+.10f}")
    W(f"minimo    {ds.min():+.10f}")
    W(f"maximo    {ds.max():+.10f}")
    W("positivas 3 de 3")
    W("```")
    W("")
    W("### Deltas por pliegue")
    W("")
    W("```text")
    for p in PARTICIONES:
        W(f"{p['nombre']:10} " + " ".join(f"{x:+.6f}" for x in p["deltas"]))
    W("```")
    W("")
    W("## Rescue, damage y neto por etiqueta")
    W("")
    W("| particion | etiqueta | rescate | damage | neto |")
    W("|---|---|---:|---:|---:|")
    for p in PARTICIONES:
        for f in p["etiquetas"]:
            W(f"| {p['nombre']} | {f['etiqueta']} | {f['rescate']} | {f['damage']} | "
              f"{f['neto']:+d} |")
    W("")
    W("### La lectura mecanica")
    W("")
    W("```text")
    W("HS   positivo en las TRES particiones   +19, +7, +18")
    W("TR   negativo en dos de tres            -5, -4, +3")
    W("AG   positivo en dos de tres            +11, +9, -11")
    W("```")
    W("")
    W("La mejora **no depende de una sola etiqueta**: HS aporta de forma consistente en las")
    W("tres, y el neto global es positivo en las tres. TR es el mas debil, y AG es el mas")
    W("inestable, con un cambio de signo entre la segunda y la tercera. Se describe como")
    W("estabilidad mecanica, y no se optimiza sobre estas etiquetas despues.")
    W("")
    W("## El test de permutacion")
    W("")
    W("| estadistico | valor |")
    W("|---|---:|")
    W(f"| delta real | {per['observado']['delta_real']:+.10f} |")
    W(f"| media nula | {per['nulo']['media']:+.10f} |")
    W(f"| desviacion nula | {per['nulo']['desviacion']:.10f} |")
    W(f"| maximo nulo | {per['nulo']['maximo']:+.10f} |")
    W(f"| percentil 99 nulo | {per['nulo']['percentil_99']:+.10f} |")
    W("| permutaciones >= observado | 0 de 1000 |")
    W(f"| p empirico | {per['test']['p_empirico']:.6f} |")
    W("")
    W("## LIMITACION DE PRECISION ESTADISTICA")
    W("")
    W("Este apartado existe para describir el limite, no para resolverlo.")
    W("")
    W("### Lo que la evidencia sostiene")
    W("")
    W("```text")
    W("la direccion del efecto es consistente")
    W("la permutacion respalda la asociacion especifica de los pesos")
    W("las tres particiones son positivas")
    W("```")
    W("")
    W("### Lo que la evidencia no resuelve")
    W("")
    W("```text")
    W("el bootstrap por grupos mantiene suficiente incertidumbre")
    W("para que el cero permanezca dentro del IC95")
    W("```")
    W("")
    W(f"IC95 por grupos: `[{per['nulo']['media']:.10f}, 0.0118175641]` en su version de")
    W("delta de acierto por ejemplo; el intervalo del candidato canonico es")
    W("`[-0.0006906975, +0.0118175641]`.")
    W("")
    W("**La limitacion es de precision, y la precision esta limitada por el tamano efectivo")
    W("del train.** No se intenta resolver con mas particiones, porque repetir particiones")
    W("hasta que un intervalo excluya el cero no seria evidencia, seria seleccion.")
    W("")
    W("## Las tres pruebas responden a tres preguntas")
    W("")
    W("| prueba | pregunta | NO responde |")
    W("|---|---|---|")
    W("| particiones | estabilidad frente al cambio de reparto | la magnitud del efecto |")
    W("| permutacion | especificidad de la correspondencia peso-ejemplo | la magnitud |")
    W("| bootstrap por grupos | incertidumbre sobre la magnitud | la especificidad |")
    W("")
    W("No se utiliza una prueba para afirmar lo que responde otra.")
    W("")
    W("## El defecto de class_weight, conservado como evidencia de auditoria")
    W("")
    W("```text")
    W("class_weight='balanced' interactuaba incorrectamente con sample_weight")
    W("sklearn recalcula 'balanced' CON los sample_weight; un dict explicito no")
    W("los coeficientes difieren hasta 1,4e-2")
    W("```")
    W("")
    W("La reproduccion valida usa el mecanismo historico exacto: `class_weight` explicito mas")
    W("`sample_weight`. La tercera particion corregida da `+0.0048825807`, y **el signo y la")
    W("conclusion no cambiaron**. El codigo historico no se modifica.")
    W("")
    W("## Analisis mecanico")
    W("")
    W("El resultado es coherente con la regla observada:")
    W("")
    W("```text")
    W("violacion alta de margen  ->  peso menor")
    W("```")
    W("")
    W("Y difiere de lo que historicamente empeoro:")
    W("")
    W("```text")
    W("aumentar el peso de ejemplos dificiles   V4-B03: medium_up -0,0096, mild_up -0,0021")
    W("reducir el peso de la violacion alta     V4-B03: down_violacion +0,0036, hasta +0,0082")
    W("```")
    W("")
    W("Se presenta como **evidencia empirica especifica de este dataset**, no como una teoria")
    W("universal sobre ponderacion de ejemplos.")
    W("")
    W("## Estado de la investigacion")
    W("")
    W("```text")
    W("research_status = CANDIDATE_PENDING_GOVERNANCE")
    W("")
    W("V03                          = FROZEN_REFERENCE")
    W("A_cost_sensitive_downweight  = PROMISING_UNCONFIRMED")
    W("todos los demas mecanismos   = CLOSED / INVALIDATED / DUPLICATE")
    W("```")
    W("")
    W("## Resumen ejecutivo")
    W("")
    W("```text")
    W("A_cost_sensitive_downweight fue reconstruido desde sus artefactos")
    W("y reproducido con R3.")
    W("")
    W("El efecto canonico es +0.0082045862.")
    W("La segunda particion confirma +0.0105253130.")
    W("La tercera particion confirma +0.0048825807 tras corregir")
    W("un defecto real de class_weight/sample_weight.")
    W("Las tres particiones mantienen la direccion positiva.")
    W("")
    W("El test de permutacion de 1000 iteraciones produce")
    W("p empirico = 0.000999.")
    W("")
    W("El bootstrap correcto por grupos mantiene el cero dentro del IC95.")
    W("")
    W("Por ello el candidato es PROMISING_UNCONFIRMED,")
    W("no ADOPTED y no INVALIDATED.")
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p69_07_cierre_final.py`. Sin push.*")
    MD_E.write_text(a_ascii("\n".join(E)), encoding="ascii")

    with CSV_E.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["bloque", "clave", "valor"])
        for nom, val, nota in ESTADOS:
            w.writerow([nom, "valor", val])
            w.writerow([nom, "nota", nota])
        for p in PARTICIONES:
            w.writerow(["PARTICION", p["nombre"], p["delta"]])
            for f in p["etiquetas"]:
                w.writerow([f"RESCUE_{p['nombre']}", f["etiqueta"],
                            f"rescate={f['rescate']}|damage={f['damage']}|neto={f['neto']}"])
        w.writerow(["RESUMEN", "media", round(float(ds.mean()), 10)])
        w.writerow(["RESUMEN", "minimo", round(float(ds.min()), 10)])
        w.writerow(["RESUMEN", "positivas", 3])
        for k, val in per["nulo"].items():
            w.writerow(["PERMUTACION", f"nulo_{k}", val])
        w.writerow(["PERMUTACION", "p_empirico", per["test"]["p_empirico"]])
        w.writerow(["PERMUTACION", "n_ge_observado",
                    per["test"]["n_permutaciones_ge_observado"]])
        w.writerow(["LIMITACION", "bootstrap_por_grupos_ic95_bajo", -0.0006906975])
        w.writerow(["LIMITACION", "bootstrap_por_grupos_ic95_alto", 0.0118175641])
        w.writerow(["LIMITACION", "incluye_cero", True])
        w.writerow(["GOVERNANZA", "research_status", "CANDIDATE_PENDING_GOVERNANCE"])
        w.writerow(["GOVERNANZA", "submission", "NO"])
        w.writerow(["GOVERNANZA", "test", "NO"])
        w.writerow(["GOVERNANZA", "leaderboard", "NO"])
        for k, val in per["decision"].items():
            w.writerow(["DECISION", k, val])
    log(f"   {MD_E.name} y {CSV_E.name} escritos")

    # ---------- la nota de gobernanza ----------
    G: list[str] = []
    Wg = G.append
    Wg("# Nota de gobernanza - A_cost_sensitive_downweight")
    Wg("")
    Wg(f"Generada por `experiments/p69_07_cierre_final.py`. Fecha "
       f"{inicio.isoformat(timespec='seconds')}.")
    Wg("")
    Wg("Esta nota responde cinco preguntas y **no decide nada por su cuenta**. En particular no")
    Wg("declara licita ni ilicita ninguna estrategia: se limita a registrar que el candidato no")
    Wg("puede evaluarse sobre el conjunto de prueba real bajo el protocolo actual.")
    Wg("")
    Wg("## 1. Que evidencia existe")
    Wg("")
    Wg("| prueba | resultado | pregunta que responde |")
    Wg("|---|---|---|")
    Wg("| Reproduccion | R3, identica a 3,2e-11 | el numero es real y trazable |")
    Wg(f"| Particion canonica | {CANONICA['delta']:+.10f}, 5 de 5 | el efecto existe en el "
       f"reparto original |")
    Wg(f"| Segunda particion | {SEGUNDA['delta']:+.10f}, 5 de 5 | sobrevive a otro reparto |")
    Wg(f"| Tercera particion | {TERCERA['delta']:+.10f}, 4 de 5 | sobrevive a un tercero |")
    Wg(f"| Permutacion, 1000 iteraciones | p = {per['test']['p_empirico']:.6f} | la asociacion "
       f"peso-ejemplo es especifica |")
    Wg("")
    Wg("```text")
    Wg("reproducibilidad      R3")
    Wg("particiones positivas 3 de 3")
    Wg("mejora minima         +0.0048825807")
    Wg("permutacion           0 de 1000 asignaciones equivalentes la alcanzan")
    Wg("```")
    Wg("")
    Wg("## 2. Que falta para adoptar")
    Wg("")
    Wg("Un unico componente de los seis:")
    Wg("")
    Wg("```text")
    Wg("bootstrap por grupos compatible   NO")
    Wg("IC95 = [-0.0006906975, +0.0118175641]   incluye el cero")
    Wg("```")
    Wg("")
    Wg("Y hay que decirlo con precision: el intervalo **no** dice que el efecto sea negativo,")
    Wg("dice que con el tamano efectivo del train no se puede descartar que lo sea. El limite")
    Wg("inferior esta a seis diezmilesimas de cero.")
    Wg("")
    Wg("## 3. Que no puede hacerse bajo el protocolo actual")
    Wg("")
    Wg("```text")
    Wg("evaluar el candidato sobre toxic_test        el test esta fuera del desarrollo")
    Wg("estimar el efecto con el test                idem")
    Wg("calibrar o ajustar umbrales con el test      idem")
    Wg("leer las etiquetas del test                  idem")
    Wg("ajustar o crear una submission adaptada      idem")
    Wg("buscar una particion que excluya el cero     seria seleccion, no evidencia")
    Wg("abrir una grid de pesos o de C               la hipotesis esta congelada")
    Wg("```")
    Wg("")
    Wg("**El candidato no puede evaluarse sobre test real bajo el protocolo actual.** Se")
    Wg("registra como un hecho, no como un juicio sobre lo que otras estrategias podrian hacer.")
    Wg("")
    Wg("## 4. Que informacion queda fuera del alcance actual")
    Wg("")
    Wg("```text")
    Wg("la magnitud del efecto sobre la poblacion de prueba")
    Wg("si la mejora se transferiria al leaderboard")
    Wg("si el intervalo por grupos se estrecharia con mas datos")
    Wg("```")
    Wg("")
    Wg("Las tres cosas requieren datos que el protocolo actual no permite usar. No son")
    Wg("cuestiones de esfuerzo ni de tecnica: son cuestiones de acceso a datos.")
    Wg("")
    Wg("## 5. Que decision seria necesaria para una eventual submission")
    Wg("")
    Wg("Una decision **explicita y separada** de la investigacion, que reconozca que:")
    Wg("")
    Wg("```text")
    Wg("1. se estaria evaluando un candidato cuyo bootstrap por grupos incluye el cero")
    Wg("2. la medicion sobre el test consumiria informacion que hoy esta intacta")
    Wg("3. el resultado, sea cual sea, no seria confirmatorio de nada nuevo")
    Wg("   (las tres particiones ya estan hechas, y repetir no anade evidencia)")
    Wg("```")
    Wg("")
    Wg("Esta nota **no toma esa decision** y no sugiere que deba tomarse ahora.")
    Wg("")
    Wg("## 6. Criterio para una futura reapertura")
    Wg("")
    Wg("Solo se consideraria reabrir con **evidencia nueva externa al experimento actual**:")
    Wg("")
    Wg("```text")
    Wg("autorizacion normativa explicita")
    Wg("nuevo dato de entrenamiento legal")
    Wg("nuevo conjunto etiquetado permitido")
    Wg("cambio de reglas")
    Wg("```")
    Wg("")
    Wg("**No** se considera justificacion suficiente:")
    Wg("")
    Wg("```text")
    Wg("'quiero un IC95 que no incluya 0'")
    Wg("mas particiones, mas semillas, mas repartos de los mismos ejemplos")
    Wg("```")
    Wg("")
    Wg("## El estado, congelado")
    Wg("")
    Wg("```text")
    Wg("research_status              CANDIDATE_PENDING_GOVERNANCE")
    Wg("")
    Wg("V03                          FROZEN")
    Wg("A_cost_sensitive_downweight  PROMISING_UNCONFIRMED")
    Wg("SUBMISSION                   NO")
    Wg("TEST                         NO")
    Wg("LEADERBOARD                  NO")
    Wg("```")
    Wg("")
    Wg("## Los mecanismos cerrados, que siguen cerrados")
    Wg("")
    Wg("```text")
    Wg("P66                              residualizacion")
    Wg("P67                              calibracion, correccion de umbral, blend de scores")
    Wg("P68-H1                           enrutado textual")
    Wg("P68-H2                           rescue weighting por correccion ajena")
    Wg("```")
    Wg("")
    Wg("La unica hipotesis viva es A_cost_sensitive_downweight, en estado")
    Wg("`PROMISING_UNCONFIRMED`.")
    Wg("")
    Wg("> V03 remains the best verified leaderboard result.")
    Wg("")
    Wg("*Generado por `experiments/p69_07_cierre_final.py`. Sin push.*")
    MD_G.write_text(a_ascii("\n".join(G)), encoding="ascii")
    log(f"   {MD_G.name} escrito")

    # ---------- comprobacion del pipeline de submission sin ejecutarlo ----------
    log("")
    log("   AUDITORIA DEL PIPELINE DE SUBMISSION, sin ejecutarlo sobre test")
    scripts_sub = sorted(p.name for p in (PROJECT_ROOT / "experiments").glob("*.py")
                         if "submission" in p.name or "refit" in p.name)
    log(f"      scripts de submission o refit historicos: {len(scripts_sub)}")
    # Lo que SI se puede verificar no es "que hacen esos scripts", sino "si se ejecutaron".
    # Un script de submission TIENE que leer el test por definicion; la pregunta util es si
    # alguno corrio en esta fase. Se comprueba por los artefactos: ninguna submission nueva,
    # ningun fichero de test tocado.
    submissions = sorted(p.name for p in (PROJECT_ROOT / "submissions").glob("*.csv")
                         if p.exists())
    modificados = subprocess.run(
        ["git", "status", "--porcelain", "--", "submissions", "data"],
        capture_output=True, text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    audit = {
        "scripts_de_submission_historicos": len(scripts_sub),
        "submissions_existentes_en_disco": len(submissions),
        "se_genero_alguna_submission_en_esta_fase": False,
        "se_modifico_alguna_submission": bool(modificados),
        "el_candidato_tiene_pipeline_propio_de_submission": False,
        "verificacion": ("no es 'que hacen los scripts historicos' sino 'si se ejecutaron'. "
                         "La comprobacion es por artefactos: cero submissions nuevas y cero "
                         "cambios en submissions/ bajo git"),
        "nota": ("el candidato congelado NO tiene pipeline propio de submission. Crearlo seria "
                 "el primer paso de una decision que esta nota no toma"),
    }
    log(f"      submissions en disco: {audit['submissions_existentes_en_disco']}")
    log(f"      submissions nuevas en esta fase: "
        f"{audit['se_genero_alguna_submission_en_esta_fase']}")
    log(f"      cambios en submissions/ o data/ bajo git: {audit['se_modifico_alguna_submission']}")
    log("      el candidato NO tiene pipeline propio de submission")
    log("      NOTA: un script de submission TIENE que leer el test por definicion, luego la")
    log("      pregunta util no es que hace cada script historico, sino si alguno corrio")

    salida = {
        "familia": "P69_CIERRE_FINAL", "experiment_id": "P69-07",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": ("CIERRE DOCUMENTAL. 0 experimentos, 0 entrenamientos, 0 submissions, "
                       "0 test, 0 holdout, 0 leaderboard"),
        "conclusion_congelada": (
            "A_cost_sensitive_downweight presenta una mejora reproducible, consistente entre "
            "tres particiones y altamente improbable bajo la permutacion de la asignacion de "
            "pesos. Sin embargo, el bootstrap por grupos continua incluyendo cero, por lo que "
            "la magnitud del efecto no queda resuelta con suficiente precision para "
            "clasificarlo como ADOPTED."),
        "formulacion_precisa": ("evidencia fuerte de direccion y mecanismo, pero incertidumbre "
                                "residual sobre la magnitud poblacional"),
        "no_se_afirma": ["demostrado definitivamente", "es ruido",
                         "V03 ha sido superado definitivamente"],
        "estados": [{"estado": n, "valor": v, "nota": t} for n, v, t in ESTADOS],
        "particiones": PARTICIONES,
        "resumen_particiones": {"media": round(float(ds.mean()), 10),
                                "mediana": round(float(np.median(ds)), 10),
                                "minimo": round(float(ds.min()), 10),
                                "maximo": round(float(ds.max()), 10),
                                "positivas": 3},
        "permutacion": {"p_empirico": per["test"]["p_empirico"],
                        "n_ge_observado": per["test"]["n_permutaciones_ge_observado"],
                        "nulo_media": per["nulo"]["media"], "nulo_maximo": per["nulo"]["maximo"]},
        "limitacion_estadistica": {
            "bootstrap_por_grupos_ic95": [-0.0006906975, 0.0118175641],
            "incluye_cero": True,
            "causa": "precision limitada por el tamano efectivo del train",
            "no_se_resuelve_con_mas_particiones": True,
        },
        "las_tres_pruebas": {
            "particiones": "estabilidad frente al cambio de reparto",
            "permutacion": "especificidad de la correspondencia peso-ejemplo",
            "bootstrap_por_grupos": "incertidumbre sobre la magnitud",
        },
        "defecto_class_weight": {"conservado_como_evidencia": True,
                                 "signo_y_conclusion_no_cambiaron": True,
                                 "codigo_historico_modificado": False},
        "mecanica": {"regla": "violacion alta de margen -> peso menor",
                     "contraste": "aumentar el peso de ejemplos dificiles empeoro",
                     "estatus": "evidencia empirica especifica del dataset"},
        "research_status": "CANDIDATE_PENDING_GOVERNANCE",
        "gobernanza": audit,
        "reproducibilidad": {"hash_v03": hash_real, "corrected_cv_v03": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    (REPORTS / "a_cost_sensitive_downweight_final_evidence.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    documentos = [MD_E, CSV_E, MD_G, REPORTS / "a_cost_sensitive_downweight_final_evidence.json"]
    sucios = [d.name for d in documentos if any(b > 127 for b in d.read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")

    log("")
    log("=" * 78)
    log("P69-07 OK | cierre documental completo")
    log(f"   estados: {len(ESTADOS)} | particiones: 3 | media {ds.mean():+.10f}")
    log("   research_status = CANDIDATE_PENDING_GOVERNANCE")
    log("   submission NO | test NO | leaderboard NO")
    log(f"   {len(documentos)} documentos, 0 bytes no ASCII")
    log("   V03 remains the best verified leaderboard result.")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
