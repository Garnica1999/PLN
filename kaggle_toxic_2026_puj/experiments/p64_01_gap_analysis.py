"""P64 track P1: what is genuinely new about segmentation, and what P59 already closed.

The precedent that has to be read first
---------------------------------------
P59-07 is the closest antecedent in the whole repository and the brief names it. It changed the
UNIT OF REPRESENTATION from the document to a local window and got

    delta -0,0006246152, 3 of 5 folds, net -1, CLOSED

with fixed windows of 32 tokens and stride 16, aggregating by the maximum window probability.
P59-09 then fused that local score with V03 through a single alpha and got +0,0014800075,
MARGINAL, and P59-02 used the local geometry as a feature block and got CLOSED.

The family is NOT untouched, and pretending otherwise would be the easiest way to waste this
phase. The question is narrower: is there a mechanism here that P59 could not reach, and is
there a stratum where it can act?

The measurement that decides it
-------------------------------
P59-07 recorded why it failed, and the reason was coverage rather than concept. The tweets are
short, median 15 tokens, so a 32-token window leaves more than one window in only a small
fraction of the cases and the change of representation can act in very few cells. That is a
statement about the SIZE of the window, not about the idea of local evidence.

A segmentation driven by punctuation is a different mechanism from a fixed numerical window,
and on this corpus it reaches a very different stratum. On this development partition:

    fixed window 32 tokens, stride 16   142 of 1483 rows with more than one unit,  9,6 per cent
    punctuation segmentation S1          751 of 1483 rows with more than one segment, 50,6 per cent

A 5,3-fold difference in reachable cells is not a parameter variation. It is the difference
between a mechanism that can act on one row in ten and one that can act on one row in two. The
brief anticipates exactly this, requiring that any window work be justified by a genuinely
different mechanism rather than a different number.

What is genuinely new, and what is not
--------------------------------------
NEW, because no historical representation computes it:

    evidence DECOMPOSITION   each segment scored with the V03 coefficients learned on the
                             training fold, which decomposes an existing model instead of
                             training a new classifier
    evidence DISTRIBUTION    concentration, conflict, disagreement and entropy of the
                             per-segment scores, a property of the SPREAD and not of the max
    leave-one-segment-out    the contribution of each segment to the full score, which P59
                             never computed at all
    STRUCTURAL stratum       the residual margin sits in LARGA and MUY_LARGA, and that is
                             where the multi-segment rows live

NOT new, and therefore forbidden here without new evidence:

    max-pooling of local scores  P59-07 aggregated exactly that and it did not work
    alpha fusion of two scores   P59-09 did it and the gate failed on both conditions
    fixed numerical windows      P59-07, plus E2-09 and V4-D01-D04
    thresholds per stratum       E2-16 and E2-20, both closed
    generic local geometry       P59-02, closed, and the same result as V6-07 before it

The association that makes the phase worth running
--------------------------------------------------
The error rate of V03 rises with the number of segments, and the stratum where the residual
margin lives is also the stratum with the most internal structure. An association between
internal structure and error is exactly what a bag of n-grams cannot express, and exactly what
would be expected if the hypothesis of this phase were true.
"""

from __future__ import annotations

import ast
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
MD_GAP = REPORTS / "p64_hypothesis_gap_analysis.md"
CSV_GAP = REPORTS / "p64_hypothesis_gap_analysis.csv"
JSON_GAP = REPORTS / "p64_hypothesis_gap_analysis.json"
CSV_REG = REPORTS / "p64_hypothesis_registry.csv"

# Cortes de longitud CONGELADOS, heredados de P63 para que las fases comparen las mismas
# regiones. No se ajustan en esta fase.
BINS: tuple[tuple[str, int, int], ...] = (
    ("CORTA", 0, 6),
    ("MEDIA", 7, 15),
    ("LARGA", 16, 30),
    ("MUY_LARGA", 31, 10**9),
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P64-01] {mensaje}", flush=True)


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
    """Clasifica las hipotesis segmentales de P64 contra el registro historico.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P64-01 que es genuinamente nuevo en segmentar, y que ya cerro P59")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    if float(v["corrected_cv"][0]) != V03_CCV:
        raise SystemExit("PARADA: el OOF canonico no reproduce el baseline")
    log(f"   V03 verificado e intacto: {V03_CCV!r} | sha256 {hash_real[:16]}...")

    ids, y, pred = v["ids"], v["y_true"].astype(int), v["pred"].astype(int)
    tr = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "toxic_train.csv", sep=";",
                     encoding="latin-1", dtype=str)
    tr["idn"] = pd.to_numeric(tr["id"].str.replace('"', "", regex=False),
                              errors="coerce").astype("int64")
    textos = tr.set_index("idn").loc[ids, "text"].astype(str).tolist()
    n_tok = np.array([len(t.split()) for t in textos])
    error = pred != y

    # ---------- la cobertura, que es lo que decidio el fracaso de P59 ----------
    s1 = np.array([len([s for s in re.split(r"[.!?]+", t) if s.strip()]) for t in textos])
    v32 = np.maximum(1, np.ceil(np.maximum(n_tok - 32, 0) / 16) + 1)
    s_cov = {
        "n_filas": len(textos),
        "tokens_mediana": float(np.median(n_tok)),
        "tokens_maximo": int(n_tok.max()),
        "tokens_percentil_90": float(np.percentile(n_tok, 90)),
        "ventana_fija_32_16": {
            "filas_con_mas_de_una_unidad": int((v32 >= 2).sum()),
            "porcentaje": round(float((v32 >= 2).mean() * 100), 4),
        },
        "segmentacion_por_puntuacion": {
            "filas_con_mas_de_uno": int((s1 >= 2).sum()),
            "porcentaje": round(float((s1 >= 2).mean() * 100), 4),
            "segmentos_medianos": float(np.median(s1)),
            "segmentos_maximos": int(s1.max()),
        },
        "factor_de_cobertura": round(float((s1 >= 2).mean() / max((v32 >= 2).mean(), 1e-9)), 4),
    }
    log("")
    log("   LA MEDICION QUE DECIDE SI ESTA FASE EXISTE")
    log(f"      tokens: mediana {s_cov['tokens_mediana']:.0f}, maximo "
        f"{s_cov['tokens_maximo']}, percentil 90 {s_cov['tokens_percentil_90']:.0f}")
    log(f"      ventana fija 32/16 de P59: "
        f"{s_cov['ventana_fija_32_16']['filas_con_mas_de_una_unidad']} filas con mas de una "
        f"unidad, {s_cov['ventana_fija_32_16']['porcentaje']:.1f} por ciento")
    log(f"      segmentacion por puntuacion: "
        f"{s_cov['segmentacion_por_puntuacion']['filas_con_mas_de_uno']} filas, "
        f"{s_cov['segmentacion_por_puntuacion']['porcentaje']:.1f} por ciento")
    log(f"      FACTOR DE COBERTURA: {s_cov['factor_de_cobertura']:.2f}x")
    log("      un factor de esa magnitud en las celdas alcanzables no es una variacion de")
    log("      parametro: es la diferencia entre un mecanismo que puede actuar en una de cada")
    log("      diez filas y otro que puede actuar en una de cada dos")

    # ---------- la asociacion con el error ----------
    por_segmentos = []
    for etiqueta, mascara in (("1", s1 == 1), ("2", s1 == 2), ("3", s1 == 3), ("4", s1 == 4),
                             ("5 o mas", s1 >= 5)):
        if mascara.sum() < 5:
            continue
        por_segmentos.append({"n_segmentos": etiqueta, "n_filas": int(mascara.sum()),
                              "tasa_de_error": round(float(error[mascara].mean()), 6),
                              "tokens_medianos": float(np.median(n_tok[mascara]))})
    s_assoc = {
        "por_numero_de_segmentos": por_segmentos,
        "tasa_1_segmento": por_segmentos[0]["tasa_de_error"],
        "tasa_5_o_mas": por_segmentos[-1]["tasa_de_error"],
        "diferencia": round(por_segmentos[-1]["tasa_de_error"]
                            - por_segmentos[0]["tasa_de_error"], 6),
        "crece_con_los_segmentos": bool(
            por_segmentos[-1]["tasa_de_error"] > por_segmentos[0]["tasa_de_error"]),
        "aviso": ("es una ASOCIACION y no un mecanismo, y la fase existe justamente para "
                  "saber si lo es"),
    }
    log("")
    log("   LA ASOCIACION QUE HACE QUE LA FASE MEREZCA LA PENA")
    log(f"      {'segmentos':>10}{'filas':>8}{'error':>10}{'tokens med.':>14}")
    for r in por_segmentos:
        log(f"      {r['n_segmentos']:>10}{r['n_filas']:>8}{r['tasa_de_error'] * 100:>9.2f}%"
            f"{r['tokens_medianos']:>14.0f}")
    log(f"      diferencia entre 1 segmento y 5 o mas: {s_assoc['diferencia'] * 100:+.2f} puntos")
    log("      una asociacion entre estructura interna y error es justo lo que una bolsa de")
    log("      n-gramas no puede expresar, y justo lo que cabria esperar si la hipotesis de la")
    log("      fase fuera cierta")

    por_rebanada = []
    for nombre, lo, hi in BINS:
        m = (n_tok >= lo) & (n_tok <= hi)
        por_rebanada.append({
            "bin": nombre, "n": int(m.sum()),
            "tokens_medios": round(float(n_tok[m].mean()), 2),
            "segmentos_medios": round(float(s1[m].mean()), 3),
            "con_dos_o_mas": int((s1[m] >= 2).sum()),
            "porcentaje_con_dos_o_mas": round(float((s1[m] >= 2).mean() * 100), 2),
            "tasa_de_error": round(float(error[m].mean()), 6),
        })
    s_rebanada = {"bins": por_rebanada}
    log("")
    log("   DONDE CAE LA ESTRUCTURA INTERNA, con los cortes de longitud heredados de P63")
    log(f"      {'bin':>10}{'n':>6}{'tokens':>9}{'segmentos':>11}{'2+ seg':>9}{'error':>9}")
    for r in por_rebanada:
        log(f"      {r['bin']:>10}{r['n']:>6}{r['tokens_medios']:>9.1f}"
            f"{r['segmentos_medios']:>11.2f}{r['con_dos_o_mas']:>9}"
            f"{r['tasa_de_error'] * 100:>8.2f}%")
    log("      la estructura interna aparece justo en LARGA y MUY_LARGA, que es donde P63")
    log("      midio que esta el margen residual")

    # ---------- la tabla de hipotesis ----------
    CERRADA = "si"
    filas: list[dict[str, Any]] = [
        {"family": "ventana numerica fija",
         "hypothesis": "cambiar la unidad de documento a ventana de 32 tokens",
         "historical_status": "exact_duplicate", "experiments": "P59-07",
         "historical_result": "-0,0006246152, 3/5, neto -1, CLOSED",
         "closed": CERRADA, "reason_closed": "medido y negativo, con su propio diagnostico de cobertura",
         "new_formulation": "", "novelty": "ninguna",
         "expected_mechanism": "", "priority": "-"},
        {"family": "fusion de score local y global",
         "hypothesis": "mezclar el score local con el de V03 mediante un alpha",
         "historical_status": "exact_duplicate", "experiments": "P59-09",
         "historical_result": "+0,0014800075, 2/5, MARGINAL, gate fallido por ambas condiciones",
         "closed": CERRADA, "reason_closed": "el alpha optimo externo daba menos que la regla anidada",
         "new_formulation": "", "novelty": "ninguna",
         "expected_mechanism": "", "priority": "-"},
        {"family": "geometria local como bloque",
         "hypothesis": "rasgos de geometria Voronoi anadidos a V03",
         "historical_status": "functional_duplicate", "experiments": "P59-02, V6-07",
         "historical_result": "AUC 0,725 para predecir errores y nada accionable, CLOSED",
         "closed": CERRADA, "reason_closed": "predecir la dificultad no es mejorar la decision",
         "new_formulation": "", "novelty": "ninguna",
         "expected_mechanism": "", "priority": "-"},
        {"family": "max-pooling de evidencia local",
         "hypothesis": "agregar la evidencia segmental tomando el maximo",
         "historical_status": "exact_duplicate", "experiments": "P59-07",
         "historical_result": "neto -4 en su propio estrato alcanzable",
         "closed": CERRADA, "reason_closed": "el max sesga al alza y desplaza celdas sin corregir errores",
         "new_formulation": "", "novelty": "ninguna",
         "expected_mechanism": "", "priority": "-"},
        {"family": "ventana como canal sparse extra",
         "hypothesis": "anadir la ventana como un canal mas al documento",
         "historical_status": "exact_duplicate", "experiments": "V4-D01-D04, E2-09",
         "historical_result": "CERRADA sin senal",
         "closed": CERRADA, "reason_closed": "la unidad de puntuacion seguia siendo el tweet",
         "new_formulation": "", "novelty": "ninguna",
         "expected_mechanism": "", "priority": "-"},
        {"family": "canal TF-IDF segmental",
         "hypothesis": "un canal de n-gramas por segmento aporta senal directa",
         "historical_status": "functional_duplicate", "experiments": "P59-07, V4-D01-D04",
         "historical_result": "la ventana como unidad dio -0,0006246152",
         "closed": CERRADA,
         "reason_closed": "misma mecanica, y P59-07 demuestra ademas que el max no es la via",
         "new_formulation": "", "novelty": "ninguna",
         "expected_mechanism": "", "priority": "-"},
        {"family": "umbral por estrato de longitud o de segmentos",
         "hypothesis": "el umbral debe depender de la region",
         "historical_status": "exact_duplicate", "experiments": "E2-16, E2-20, V8-09",
         "historical_result": "CERRADAS sin mejora",
         "closed": CERRADA, "reason_closed": "medido tres veces con variantes distintas",
         "new_formulation": "", "novelty": "ninguna",
         "expected_mechanism": "", "priority": "-"},
        {"family": "segmentacion determinista por puntuacion",
         "hypothesis": "la distribucion interna de la evidencia es lo que V03 aplana",
         "historical_status": "genuinely_new", "experiments": "0",
         "historical_result": "nunca medida; P59 uso ventanas numericas, no estructura",
         "closed": "no", "reason_closed": "",
         "new_formulation": ("segmentos por puntuacion fuerte, con la evidencia de cada uno "
                            "medida con los coeficientes de V03 aprendidos en el pliegue"),
         "novelty": "ALTA",
         "expected_mechanism": "el promedio implicito de TF-IDF destruye la distribucion interna",
         "priority": "1"},
        {"family": "descomposicion de la evidencia por segmento",
         "hypothesis": "cada segmento aporta una contribucion identificable al score",
         "historical_status": "genuinely_new", "experiments": "0",
         "historical_result": "P59 no computo nunca contribuciones por unidad",
         "closed": "no", "reason_closed": "",
         "new_formulation": ("leave-one-segment-out: la diferencia entre el score completo y el "
                            "score sin ese segmento, por etiqueta"),
         "novelty": "ALTA",
         "expected_mechanism": "un segmento domina, o hay varios que compiten",
         "priority": "2"},
        {"family": "conflicto de evidencia interna",
         "hypothesis": "los tweets largos pueden tener evidencia a favor y en contra a la vez",
         "historical_status": "genuinely_new", "experiments": "0",
         "historical_result": "nunca medida",
         "closed": "no", "reason_closed": "",
         "new_formulation": "positiva contra negativa dentro del mismo tweet, por etiqueta",
         "novelty": "ALTA", "expected_mechanism": "coexistencia de evidencia contradictoria",
         "priority": "3"},
        {"family": "perfil de longitud y dispersion de segmentos",
         "hypothesis": "la estructura segmental en si es un rasgo con senal propia",
         "historical_status": "genuinely_new", "experiments": "0",
         "historical_result": "nunca medida; V30-01 y V31 trabajan con estructura global",
         "closed": "no", "reason_closed": "",
         "new_formulation": "perfil de longitudes de segmento, dispersion y entropia",
         "novelty": "MEDIA", "expected_mechanism": "forma de la distribucion interna",
         "priority": "4"},
    ]
    df = pd.DataFrame(filas)
    df.to_csv(CSV_GAP, index=False, encoding="utf-8")
    abiertas = df[df["closed"] == "no"]
    cerradas = df[df["closed"] == CERRADA]
    log("")
    log(f"   RESULTADO: {len(abiertas)} familias abiertas, {len(cerradas)} cerradas sin gastar")
    for _, r in abiertas.iterrows():
        log(f"      prioridad {r['priority']}  {r['family']}")

    registry = [
        {"ID": "P64-00", "Track": "gap analysis",
         "Hypothesis": "que es genuinamente nuevo en segmentar",
         "Historical evidence": "P59-07, P59-09, P59-02, V4-D, E2-09, E2-16, E2-20",
         "Novelty": "n/a", "Expected mechanism": "n/a",
         "Method": "lectura del registro y medicion de cobertura", "Status": "cerrado",
         "Result": f"cobertura {s_cov['factor_de_cobertura']:.2f}x mayor por puntuacion",
         "Decision": f"{len(abiertas)} abiertas, {len(cerradas)} cerradas"},
    ]
    for _, r in abiertas.iterrows():
        registry.append({
            "ID": f"P64-H{r['priority']}", "Track": r["family"],
            "Hypothesis": r["hypothesis"], "Historical evidence": r["historical_result"],
            "Novelty": r["novelty"], "Expected mechanism": r["expected_mechanism"],
            "Method": r["new_formulation"], "Status": "abierta", "Result": "",
            "Decision": ""})
    pd.DataFrame(registry).to_csv(CSV_REG, index=False, encoding="utf-8")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P64_GAP_ANALYSIS",
        "naturaleza": "AUDITORIA. 0 experimentos, 0 modelos, 0 features",
        "fecha": inicio.isoformat(timespec="seconds"), "experiment_id": "P64-00",
        "script": "experiments/p64_01_gap_analysis.py",
        "pregunta": ("que es genuinamente nuevo en segmentar, dado que P59-07 ya probo "
                     "ventanas locales"),
        "precedente_mas_cercano": {
            "P59-07": ("ventana fija de 32 tokens con avance 16, agregacion por maximo, "
                       "delta -0,0006246152, 3/5, neto -1, CLOSED"),
            "P59-09": "fusion con un alpha, +0,0014800075, 2/5, MARGINAL",
            "P59-02": "geometria local como bloque, AUC 0,725 y nada accionable, CLOSED",
        },
        "cobertura": s_cov,
        "asociacion_con_el_error": s_assoc,
        "por_rebanada": s_rebanada,
        "tabla": filas,
        "n_abiertas": len(abiertas), "n_cerradas": len(cerradas),
        "cortes_de_longitud_congelados": [{"nombre": a, "desde": b, "hasta": c}
                                          for a, b, c in BINS],
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "hash_v03": hash_real,
                             "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_GAP.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P64 - Analisis de hypotheses: que es genuinamente nuevo en segmentar")
    W("")
    W(f"Generado por `experiments/p64_01_gap_analysis.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## El precedente que hay que leer antes que nada")
    W("")
    W("**P59-07 es el antecedente mas cercano de todo el repositorio**, y el encargo lo nombra")
    W("explicitamente. Cambio la UNIDAD DE REPRESENTACION del documento a una ventana local y")
    W("consiguio:")
    W("")
    W("```text")
    W("delta -0,0006246152, 3 de 5 pliegues, neto -1, CLOSED")
    W("ventana fija de 32 tokens, avance 16, agregacion por maximo")
    W("```")
    W("")
    W("Despues P59-09 fusiono ese score local con V03 mediante un alpha y dio +0,0014800075 con")
    W("2 de 5, MARGINAL, y P59-02 uso la geometria local como bloque y quedo CLOSED.")
    W("")
    W("Asi que la familia **NO esta intacta**, y fingir lo contrario seria la forma mas facil de")
    W("tirar esta fase. La pregunta es mas estrecha: hay un mecanismo que P59 no podia alcanzar,")
    W("y hay un estrato donde puede actuar?")
    W("")
    W("## La medicion que decide si la fase existe")
    W("")
    W("P59-07 registro por que fallo, y la razon fue **COBERTURA y no concepto**. Los tweets son")
    W(f"cortos, mediana de {s_cov['tokens_mediana']:.0f} tokens, luego una ventana de 32 tokens")
    W("deja mas de una ventana en una fraccion pequena de los casos. Eso es una afirmacion sobre")
    W("el TAMANO de la ventana, no sobre la idea de evidencia local.")
    W("")
    W("Una segmentacion guiada por puntuacion es **otro mecanismo**, y en este corpus alcanza")
    W("otro estrato:")
    W("")
    W("| criterio | filas con mas de una unidad | porcentaje |")
    W("|---|---:|---:|")
    W(f"| ventana fija 32 tokens, avance 16 | "
      f"{s_cov['ventana_fija_32_16']['filas_con_mas_de_una_unidad']} | "
      f"{s_cov['ventana_fija_32_16']['porcentaje']:.1f} por ciento |")
    W(f"| segmentacion por puntuacion fuerte | "
      f"{s_cov['segmentacion_por_puntuacion']['filas_con_mas_de_uno']} | "
      f"{s_cov['segmentacion_por_puntuacion']['porcentaje']:.1f} por ciento |")
    W("")
    W(f"**Factor de cobertura: {s_cov['factor_de_cobertura']:.2f}x.** Un factor de esa magnitud en")
    W("las celdas alcanzables no es una variacion de parametro: es la diferencia entre un")
    W("mecanismo que puede actuar en una de cada diez filas y otro que puede actuar en una de")
    W("cada dos. El encargo lo anticipa: exige que cualquier trabajo con ventanas se justifique")
    W("por un mecanismo genuinamente diferente y no por un numero distinto.")
    W("")
    W("## Lo que es genuinamente nuevo")
    W("")
    W("Ninguna representacion historica calcula esto:")
    W("")
    W("- **descomposicion de la evidencia**: cada segmento puntuado con los coeficientes de")
    W("  V03 aprendidos en el pliegue de entrenamiento, que es descomponer un modelo que ya")
    W("  existe y no entrenar un clasificador nuevo.")
    W("- **distribucion de la evidencia**: concentracion, conflicto, desacuerdo y entropia de")
    W("  las puntuaciones por segmento, que es una propiedad de la DISPERSION y no del maximo.")
    W("- **leave-one-segment-out**: la contribucion de cada segmento al score completo, que")
    W("  P59 no computo nunca.")
    W("- **foco en el estrato estructural**: el margen residual esta en LARGA y MUY_LARGA, y")
    W("  ahi es donde viven las filas multi-segmento.")
    W("")
    W("## Lo que NO es nuevo, y queda prohibido sin evidencia nueva")
    W("")
    W("| familia | quien la cerro | con que numero |")
    W("|---|---|---|")
    for _, r in cerradas.iterrows():
        W(f"| {r['family']} | {r['experiments']} | {r['historical_result']} |")
    W("")
    W("El punto delicate es el segundo: **max-pooling de evidencia local** lo hizo P59-07 y dio")
    W("neto -4 en su propio estrato alcanzable. La agregacion por maximo no es la via, y por eso")
    W("esta fase se ocupa de la DISTRIBUCION y no del maximo.")
    W("")
    W("## La asociacion que hace que la fase merezca la pena")
    W("")
    W("| segmentos | filas | error de V03 | tokens medianos |")
    W("|---:|---:|---:|---:|")
    for r in por_segmentos:
        W(f"| {r['n_segmentos']} | {r['n_filas']} | {r['tasa_de_error'] * 100:.2f}% | "
          f"{r['tokens_medianos']:.0f} |")
    W("")
    W(f"Diferencia entre un segmento y cinco o mas: **{s_assoc['diferencia'] * 100:+.2f} "
      f"puntos**.")
    W("")
    W("Es una **asociacion y no un mecanismo**, y la fase existe justamente para saber si lo es.")
    W("Una asociacion entre estructura interna y error es exactamente lo que una bolsa de")
    W("n-gramas no puede expresar, y exactamente lo que cabria esperar si la hipotesis de la")
    W("fase fuera cierta.")
    W("")
    W("## Donde cae la estructura interna")
    W("")
    W("| bin | n | tokens medios | segmentos medios | con 2 o mas | error |")
    W("|---|---:|---:|---:|---:|---:|")
    for r in por_rebanada:
        W(f"| {r['bin']} | {r['n']} | {r['tokens_medios']:.1f} | {r['segmentos_medios']:.2f} | "
          f"{r['con_dos_o_mas']} ({r['porcentaje_con_dos_o_mas']:.0f}%) | "
          f"{r['tasa_de_error'] * 100:.2f}% |")
    W("")
    W("La estructura interna aparece justo en LARGA y MUY_LARGA, que es donde P63 midio que")
    W("esta el margen residual. Los cortes de longitud son los de P63, heredados y congelados,")
    W("y no se ajustan en esta fase.")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p64_01_gap_analysis.py`. Sin push.*")
    MD_GAP.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_GAP, JSON_GAP, CSV_GAP, CSV_REG):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P64-01 OK | {len(abiertas)} abiertas | {len(cerradas)} cerradas | cobertura "
        f"{s_cov['factor_de_cobertura']:.2f}x | asociacion {s_assoc['diferencia'] * 100:+.2f} "
        f"puntos | commit {commit}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
