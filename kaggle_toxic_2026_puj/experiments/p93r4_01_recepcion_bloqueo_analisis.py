#!/usr/bin/env python
"""
P93-R4 - Recepcion, bloqueo y analisis de la segunda ronda (C1/C2).

Entradas (solo lectura):
  artifacts/p93_adjudication_v2/responses/response_C1_round2.csv
  artifacts/p93_adjudication_v2/responses/response_C2_round2_correcciones.csv
  artifacts/p93_adjudication_v1/responses/response_C1.csv (ronda 1 archivada)
  artifacts/p93_adjudication_v1/responses/response_C2.csv (ronda 1 archivada)
  reports/p93r3_round2_id_assignment.csv (celdas <-> ids P93S1/P93S2)
  artifacts/p93_adjudication_v2/annotator_C{1,2}/adjudication_round2_C{1,2}.csv (paquetes)

Salidas:
  artifacts/p93_adjudication_v2/responses/PROVENANCE.md
  reports/p93r4_final_decisions.csv
  reports/p93r4_final_human_agreement.md
  reports/p93r4_final_human_agreement.csv
  reports/p93r4_pending_cases.csv
  reports/p93r4_manifest.json

Reglas: sin gold (no se lee original_HS/TR/AG en esta fase), sin V03/OOF, sin
test/holdout/leaderboard, sin entrenamiento, sin cambios de etiquetas.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_RESP_V2 = RAIZ / "artifacts" / "p93_adjudication_v2" / "responses"
DIR_RESP_V1 = RAIZ / "artifacts" / "p93_adjudication_v1" / "responses"
V2 = RAIZ / "artifacts" / "p93_adjudication_v2"
RUTA_C1_R2 = DIR_RESP_V2 / "response_C1_round2.csv"
RUTA_C2_R2 = DIR_RESP_V2 / "response_C2_round2_correcciones.csv"
RUTA_C1_R1 = DIR_RESP_V1 / "response_C1.csv"
RUTA_C2_R1 = DIR_RESP_V1 / "response_C2.csv"
RUTA_ASSIGN = RAIZ / "reports" / "p93r3_round2_id_assignment.csv"
RUTA_CLAVE = RAIZ / "reports" / "p92w_internal_sampling_key.csv"
SAL_PROV = DIR_RESP_V2 / "PROVENANCE.md"
SAL_DEC = RAIZ / "reports" / "p93r4_final_decisions.csv"
SAL_MD = RAIZ / "reports" / "p93r4_final_human_agreement.md"
SAL_AGR = RAIZ / "reports" / "p93r4_final_human_agreement.csv"
SAL_PEND = RAIZ / "reports" / "p93r4_pending_cases.csv"
SAL_MANIFEST = RAIZ / "reports" / "p93r4_manifest.json"

TOKENS = ("POSITIVE", "NEGATIVE", "UNABLE_TO_DETERMINE")
UTD = "UNABLE_TO_DETERMINE"
ETIQUETAS = ("HS", "TR", "AG")
COL = {"HS": "HS_annotation", "TR": "TR_annotation", "AG": "AG_annotation"}


def sha256_archivo(ruta: Path) -> str:
    """sha256 completo de un archivo (lectura en bloques)."""
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def leer_csv(ruta: Path) -> list[dict]:
    """Lee un CSV con comillas estandar."""
    with open(ruta, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def celdas_3x3(va: dict, vb: dict, ids: list) -> dict:
    """Matriz 3x3 aplanada (X=revisor1 filas, Y=revisor2 columnas)."""
    corto = {"POSITIVE": "P", "NEGATIVE": "N", UTD: "U"}
    c = defaultdict(int)
    for r in ids:
        c[corto[va[r]] + corto[vb[r]]] += 1
    return {k: c.get(k, 0) for k in ("NN", "NP", "NU", "PN", "PP", "PU", "UN", "UP", "UU")}


def kappa_nominal(matriz: dict, categorias: tuple) -> float | None:
    """Kappa de Cohen nominal a partir de una matriz aplanada."""
    total = sum(matriz.values())
    if total == 0:
        return None
    po = sum(matriz.get(x + x, 0) for x in categorias) / total
    pe = 0.0
    for x in categorias:
        nx = sum(matriz.get(x + y, 0) for y in categorias)
        ny = sum(matriz.get(y + x, 0) for y in categorias)
        pe += (nx / total) * (ny / total)
    if pe >= 1.0:
        return None
    return (po - pe) / (1.0 - pe)


def krippendorff_alpha(unidades: list[list[str | None]]) -> float | None:
    """Alpha de Krippendorff nominal via matriz de coincidencias."""
    coincidencias: dict[tuple[str, str], float] = defaultdict(float)
    valores: set[str] = set()
    for unidad in unidades:
        presentes = [v for v in unidad if v is not None]
        m = len(presentes)
        if m < 2:
            continue
        for i, vi in enumerate(presentes):
            for j, vj in enumerate(presentes):
                if i != j:
                    coincidencias[(vi, vj)] += 1.0 / (m - 1)
        valores.update(presentes)
    if not valores:
        return None
    n_c = {c: sum(coincidencias[(c, k)] for k in valores) for c in valores}
    n = sum(n_c.values())
    if n <= 1:
        return None
    d_o = sum(v for (c, k), v in coincidencias.items() if c != k) / n
    d_e = (n * n - sum(v * v for v in n_c.values())) / (n * (n - 1))
    if d_e <= 0:
        return None
    return 1.0 - d_o / d_e


def main() -> None:
    errores: list[str] = []
    avisos: list[str] = []
    chequeos: dict[str, bool] = {}

    # ---------------------------------------------------------------- cargas
    c1r2 = leer_csv(RUTA_C1_R2)
    c2r2 = leer_csv(RUTA_C2_R2)
    c1r1 = {f["adjudication_review_id"]: f for f in leer_csv(RUTA_C1_R1)}
    c2r1 = {f["adjudication_review_id"]: f for f in leer_csv(RUTA_C2_R1)}
    assign = leer_csv(RUTA_ASSIGN)
    pkg1 = {f["adjudication_review_id"]: f for f in
            leer_csv(V2 / "annotator_C1" / "adjudication_round2_C1.csv")}
    pkg2 = {f["adjudication_review_id"]: f for f in
            leer_csv(V2 / "annotator_C2" / "adjudication_round2_C2.csv")}

    celda_de_s1 = {f["P93S1_id"]: (f["review_id"], f["label"]) for f in assign}
    celda_de_s2 = {f["P93S2_id"]: (f["review_id"], f["label"]) for f in assign}
    s1_de_celda = {v: k for k, v in celda_de_s1.items()}
    s2_de_celda = {v: k for k, v in celda_de_s2.items()}
    celdas = sorted(celda_de_s1.values())
    # cadena de ids: ronda 1 uso P93R-####; la clave interna la une a review_id
    clave_r1 = leer_csv(RAIZ / "reports" / "p93r_internal_adjudication_key.csv")
    p93r_de_review = {f["review_id"]: f["adjudication_review_id"] for f in clave_r1}
    chequeos_previos = {
        "cadena_ids_ronda1": set(p93r_de_review) == {c[0] for c in celdas}
        and set(c1r1) == set(p93r_de_review.values())
        and set(c2r1) == set(p93r_de_review.values()),
    }
    chequeos.update(chequeos_previos)

    # ------------------------------------------------------ validacion C1
    chequeos["C1_176_filas"] = len(c1r2) == 176
    ids1 = [f["adjudication_review_id"] for f in c1r2]
    chequeos["C1_ids_unicos_esperados"] = len(set(ids1)) == 176 and set(ids1) == set(celda_de_s1)
    c1p = {f["adjudication_review_id"]: f for f in c1r2}
    chequeos["C1_etiquetas_y_AB_coinciden_paquete"] = all(
        c1p[i]["target_label"] == pkg1[i]["target_label"]
        and c1p[i]["annotator_A_response"] == pkg1[i]["annotator_A_response"]
        and c1p[i]["annotator_B_response"] == pkg1[i]["annotator_B_response"]
        for i in ids1)
    chequeos["C1_inicial_coincide_ronda1"] = all(
        c1p[i]["reviewer_initial_response"]
        == c1r1[p93r_de_review[celda_de_s1[i][0]]][COL[celda_de_s1[i][1]]]
        for i in ids1)
    chequeos["C1_final_tokens"] = all(f["final_response"] in TOKENS for f in c1r2)
    chequeos["C1_version_v21"] = all(f["guideline_version"] == "v2.1" for f in c1r2)
    cambios_c1 = [i for i in ids1
                  if c1p[i]["final_response"] != c1p[i]["reviewer_initial_response"]]
    chequeos["C1_cambios_con_justificacion"] = all(c1p[i]["final_rationale"] for i in cambios_c1)
    chequeos["C1_sin_utd_final"] = not any(f["final_response"] == UTD for f in c1r2)
    if cambios_c1:
        avisos.append(f"C1: {len(cambios_c1)} cambios de decision ({sorted(cambios_c1)})")

    # ------------------------------------------------------ validacion C2
    chequeos["C2_16_filas"] = len(c2r2) == 16
    ids2 = [f["adjudication_review_id"] for f in c2r2]
    chequeos["C2_ids_esperados"] = len(set(ids2)) == 16 and set(ids2) <= set(celda_de_s2)
    c2p = {f["adjudication_review_id"]: f for f in c2r2}
    chequeos["C2_etiquetas_coinciden"] = all(
        c2p[i]["target_label"] == celda_de_s2[i][1] for i in ids2)
    chequeos["C2_inicial_coincide_ronda1"] = all(
        c2p[i]["reviewer_initial_response"]
        == c2r1[p93r_de_review[celda_de_s2[i][0]]][COL[celda_de_s2[i][1]]]
        for i in ids2)
    chequeos["C2_final_tokens"] = all(f["final_response"] in TOKENS for f in c2r2)
    chequeos["C2_version_v21"] = all(f["guideline_version"] == "v2.1" for f in c2r2)
    chequeos["C2_filas_son_cambios_o_utd"] = all(
        c2p[i]["final_response"] != c2p[i]["reviewer_initial_response"]
        or c2p[i]["final_response"] == UTD for i in ids2)
    chequeos["C2_justificacion_en_todas"] = all(f["final_rationale"] for f in c2r2)
    cambios_c2 = [i for i in ids2
                  if c2p[i]["final_response"] != c2p[i]["reviewer_initial_response"]]
    avisos.append(f"C2: archivo de CORRECCIONES ({len(c2r2)} filas: {len(cambios_c2)} cambios + "
                  f"{len(c2r2) - len(cambios_c2)} casos que permanecen en UTD); las 160 celdas "
                  "restantes conservan su respuesta inicial (reconstruccion documentada)")

    # --------------------------------------------------- bloqueo de decisiones
    c1_final = {i: c1p[i]["final_response"] for i in ids1}
    c1_ini = {i: c1p[i]["reviewer_initial_response"] for i in ids1}
    c2_final = {}
    c2_ini = {}
    for celda in celdas:
        s2 = s2_de_celda[celda]
        if s2 in c2p:
            c2_final[s2] = c2p[s2]["final_response"]
        else:
            c2_final[s2] = c2r1[p93r_de_review[celda[0]]][COL[celda[1]]]
        c2_ini[s2] = c2r1[p93r_de_review[celda[0]]][COL[celda[1]]]
    chequeos["decisiones_finales_completas"] = all(
        c1_final[s1_de_celda[c]] in TOKENS and c2_final[s2_de_celda[c]] in TOKENS
        for c in celdas)

    with open(SAL_DEC, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["example_id", "label", "C1_initial_response", "C1_final_response",
                    "C1_final_rationale", "C1_remaining_uncertainty",
                    "C2_initial_response", "C2_final_response", "C2_final_rationale",
                    "C2_remaining_uncertainty", "guideline_version"])
        for celda in celdas:
            s1, s2 = s1_de_celda[celda], s2_de_celda[celda]
            w.writerow([celda[0], celda[1], c1_ini[s1], c1_final[s1],
                        c1p[s1]["final_rationale"], c1p[s1]["remaining_uncertainty"],
                        c2_ini[s2], c2_final[s2],
                        c2p.get(s2, {}).get("final_rationale", ""),
                        c2p.get(s2, {}).get("remaining_uncertainty", ""),
                        "v2.1"])

    # --------------------------------------------------------- acuerdo final
    metricas = {}
    for et in ETIQUETAS:
        ids_et = [s1_de_celda[c] for c in celdas if c[1] == et]
        va = {i: c1_final[i] for i in ids_et}
        vb = {i: c2_final[s2_de_celda[celda_de_s1[i]]] for i in ids_et}
        m3 = celdas_3x3(va, vb, ids_et)
        total = len(ids_et)
        raw = m3["NN"] + m3["PP"] + m3["UU"]
        bin_cells = m3["NN"] + m3["NP"] + m3["PN"] + m3["PP"]
        pos_a = m3["PP"] + m3["PN"]
        pos_b = m3["PP"] + m3["NP"]
        neg_a = m3["NN"] + m3["NP"]
        neg_b = m3["NN"] + m3["PN"]
        metricas[et] = {
            "total": total, "m3": m3, "raw": raw, "bin_cells": bin_cells,
            "utd_a": sum(1 for v in va.values() if v == UTD),
            "utd_b": sum(1 for v in vb.values() if v == UTD),
            "pos_dice": (2 * m3["PP"], pos_a + pos_b),
            "neg_dice": (2 * m3["NN"], neg_a + neg_b),
            "kappa3": kappa_nominal(m3, ("N", "P", "U")),
            "kappa_bin": kappa_nominal({k: m3[k] for k in ("NN", "NP", "PN", "PP")},
                                       ("N", "P")),
            "alpha_cat": krippendorff_alpha([[va[i], vb[i]] for i in ids_et]),
            "alpha_miss": krippendorff_alpha(
                [[None if v == UTD else v for v in (va[i], vb[i])] for i in ids_et]),
        }
    # conjunto: celdas (176) y tweets (127)
    exacto_celdas = sum(1 for c in celdas
                        if c1_final[s1_de_celda[c]] == c2_final[s2_de_celda[c]])
    por_tweet = defaultdict(list)
    for c in celdas:
        por_tweet[c[0]].append(c1_final[s1_de_celda[c]] == c2_final[s2_de_celda[c]])
    exacto_tweets = sum(1 for v in por_tweet.values() if all(v))
    # cambios desde ronda inicial
    cambios_por_et = Counter()
    for c in celdas:
        s1, s2 = s1_de_celda[c], s2_de_celda[c]
        if c1_final[s1] != c1_ini[s1]:
            cambios_por_et[f"C1_{c[1]}"] += 1
        if c2_final[s2] != c2_ini[s2]:
            cambios_por_et[f"C2_{c[1]}"] += 1

    # -------------------------------------------------------- clasificacion
    clasificacion = {}
    for c in celdas:
        s1, s2 = s1_de_celda[c], s2_de_celda[c]
        f1, f2 = c1_final[s1], c2_final[s2]
        if f1 not in TOKENS or f2 not in TOKENS:
            clase = "MISSING_OR_INVALID"
        elif f1 == UTD and f2 == UTD:
            texto = ((c1p[s1]["final_rationale"] or "") + " " +
                     (c2p.get(s2, {}).get("final_rationale") or "") + " " +
                     (c2p.get(s2, {}).get("remaining_uncertainty") or "")).lower()
            politica = any(k in texto for k in
                           ("frontera", "umbral", "guía no", "guia no", "no fija",
                            "no resuelve", "ambigüedad normativa", "ambiguedad normativa"))
            clase = "POLICY_AMBIGUITY" if politica else "FINAL_AGREEMENT_UTD"
        elif f1 == f2:
            clase = "FINAL_AGREEMENT_BINARY"
        else:
            clase = "FINAL_DISAGREEMENT"
        clasificacion[c] = clase
    conteo_clase = Counter(clasificacion.values())
    pendientes = [c for c in celdas
                  if clasificacion[c] in ("FINAL_DISAGREEMENT", "POLICY_AMBIGUITY")]

    with open(SAL_PEND, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["review_id", "label", "C1_final", "C2_final", "C1_rationale",
                    "C2_rationale", "category", "third_review_proposed"])
        for c in pendientes:
            s1, s2 = s1_de_celda[c], s2_de_celda[c]
            w.writerow([c[0], c[1], c1_final[s1], c2_final[s2],
                        c1p[s1]["final_rationale"],
                        c2p.get(s2, {}).get("final_rationale", ""),
                        clasificacion[c], "PENDING_THIRD_REVIEW"])

    for clave_chk, valor in chequeos.items():
        if not valor:
            errores.append(f"chequeo fallido: {clave_chk}")

    # ------------------------------------------------------------ salidas
    with open(SAL_AGR, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["label", "metrica", "numerador", "denominador", "valor", "nota"])
        for et in ETIQUETAS:
            m = metricas[et]
            w.writerow([et, "celdas", m["total"], m["total"], "", "unidad: celdas (tweet,etiqueta)"])
            w.writerow([et, "acuerdo_bruto_3cat", m["raw"], m["total"],
                        f"{m['raw'] / m['total']:.4f}", "UTD como tercera categoria"])
            w.writerow([et, "acuerdo_binario_condicional",
                        m["m3"]["NN"] + m["m3"]["PP"], m["bin_cells"],
                        f"{(m['m3']['NN'] + m['m3']['PP']) / m['bin_cells']:.4f}"
                        if m["bin_cells"] else "", "solo celdas binarias de ambos"])
            w.writerow([et, "dice_positivo", m["pos_dice"][0], m["pos_dice"][1],
                        f"{m['pos_dice'][0] / m['pos_dice'][1]:.4f}" if m["pos_dice"][1] else "",
                        "celdas binarias"])
            w.writerow([et, "dice_negativo", m["neg_dice"][0], m["neg_dice"][1],
                        f"{m['neg_dice'][0] / m['neg_dice'][1]:.4f}" if m["neg_dice"][1] else "",
                        "celdas binarias"])
            w.writerow([et, "kappa_cohen_3cat", "", m["total"],
                        f"{m['kappa3']:.4f}" if m["kappa3"] is not None else "n/d",
                        "nominal, UTD como categoria"])
            w.writerow([et, "kappa_cohen_bin", "", m["bin_cells"],
                        f"{m['kappa_bin']:.4f}" if m["kappa_bin"] is not None else "n/d",
                        "solo celdas binarias"])
            w.writerow([et, "alpha_utd_categoria", "", m["total"],
                        f"{m['alpha_cat']:.4f}" if m["alpha_cat"] is not None else "n/d", ""])
            w.writerow([et, "alpha_utd_faltante", "", m["total"],
                        f"{m['alpha_miss']:.4f}" if m["alpha_miss"] is not None else "n/d", ""])
            w.writerow([et, "utd_final_C1", m["utd_a"], m["total"], "", ""])
            w.writerow([et, "utd_final_C2", m["utd_b"], m["total"], "", ""])
        w.writerow(["JOINT", "exacto_celdas", exacto_celdas, 176,
                    f"{exacto_celdas / 176:.4f}", "celdas (tweet,etiqueta)"])
        w.writerow(["JOINT", "exacto_tweets", exacto_tweets, 127,
                    f"{exacto_tweets / 127:.4f}",
                    "tweets cuyas celdas candidatas coinciden todas"])
        for k, v in sorted(cambios_por_et.items()):
            w.writerow(["JOINT", f"cambios_{k}", v, "", "", "final != inicial"])

    lineas = []
    lineas.append("# P93-R4 - Acuerdo humano final (decisiones bloqueadas de C1 y C2)")
    lineas.append("")
    lineas.append("Fecha: 2026-10-13. Unidad: celda (tweet, etiqueta); 176 celdas")
    lineas.append("candidatas (HS 23, TR 87, AG 66); 127 tweets unicos. Las decisiones")
    lineas.append("finales quedan BLOQUEADAS; el gold NO se consulta en esta fase.")
    lineas.append("")
    lineas.append("## 1. Recepcion y bloqueo")
    lineas.append("")
    lineas.append("- C1: archivo completo (176 filas, 9 columnas sin text/nota inicial;")
    lineas.append(f"  {len(cambios_c1)} cambios de decision: {', '.join(sorted(cambios_c1))}).")
    lineas.append("- C2: archivo de CORRECCIONES (16 filas: 12 cambios + 4 casos que")
    lineas.append("  permanecen en UTD); las 160 celdas restantes conservan su respuesta")
    lineas.append("  inicial (reconstruccion documentada en PROVENANCE).")
    lineas.append("- Decisiones iniciales de la ronda 1 preservadas (verificadas contra los")
    lineas.append("  archivos archivados).")
    lineas.append("")
    lineas.append("## 2. Distribuciones finales y UTD")
    lineas.append("")
    lineas.append("| etiqueta | celdas | UTD C1 | UTD C2 |")
    lineas.append("|---|---|---|---|")
    for et in ETIQUETAS:
        m = metricas[et]
        lineas.append(f"| {et} | {m['total']} | {m['utd_a']} | {m['utd_b']} |")
    lineas.append("")
    lineas.append("## 3. Acuerdo final por etiqueta (numeradores/denominadores)")
    lineas.append("")
    lineas.append("| etiqueta | bruto 3cat | binario cond. | dice pos | dice neg | kappa 3cat | kappa bin | alpha cat | alpha falt |")
    lineas.append("|---|---|---|---|---|---|---|---|---|")
    for et in ETIQUETAS:
        m = metricas[et]
        k3 = "n/d" if m["kappa3"] is None else f"{m['kappa3']:.3f}"
        kb = "n/d" if m["kappa_bin"] is None else f"{m['kappa_bin']:.3f}"
        ac = "n/d" if m["alpha_cat"] is None else f"{m['alpha_cat']:.3f}"
        am = "n/d" if m["alpha_miss"] is None else f"{m['alpha_miss']:.3f}"
        pd = f"{m['pos_dice'][0]}/{m['pos_dice'][1]}" if m["pos_dice"][1] else "n/d"
        nd = f"{m['neg_dice'][0]}/{m['neg_dice'][1]}" if m["neg_dice"][1] else "n/d"
        lineas.append(f"| {et} | {m['raw']}/{m['total']} | "
                      f"{m['m3']['NN'] + m['m3']['PP']}/{m['bin_cells']} | {pd} | {nd} | "
                      f"{k3} | {kb} | {ac} | {am} |")
    lineas.append("")
    lineas.append("Matrices 3x3 finales (filas C1, columnas C2):")
    lineas.append("")
    for et in ETIQUETAS:
        m3 = metricas[et]["m3"]
        lineas.append(f"- {et}: NN={m3['NN']} NP={m3['NP']} NU={m3['NU']} | PN={m3['PN']} "
                      f"PP={m3['PP']} PU={m3['PU']} | UN={m3['UN']} UP={m3['UP']} UU={m3['UU']}")
    lineas.append("")
    lineas.append("## 4. Acuerdo conjunto")
    lineas.append("")
    lineas.append(f"- Celdas con acuerdo exacto: {exacto_celdas}/176.")
    lineas.append(f"- Tweets cuyas celdas candidatas coinciden todas: {exacto_tweets}/127.")
    lineas.append(f"- Cambios de decision desde la ronda inicial: "
                  + ", ".join(f"{k}={v}" for k, v in sorted(cambios_por_et.items())))
    lineas.append("")
    lineas.append("## 5. Clasificacion de resultados humanos")
    lineas.append("")
    for k, v in sorted(conteo_clase.items()):
        lineas.append(f"- {k}: {v} celdas.")
    lineas.append(f"- Cola de pendientes (FINAL_DISAGREEMENT + POLICY_AMBIGUITY): "
                  f"{len(pendientes)} celdas -> p93r4_pending_cases.csv (tercera revision")
    lineas.append("  propuesta; sin desempate automatico).")
    lineas.append("")
    lineas.append("## 6. Cegamiento y procedencia (limitaciones)")
    lineas.append("")
    lineas.append("- El cegamiento de la RONDA 1 sigue sin poder verificarse completamente")
    lineas.append("  (P93-R3: sin registros de acceso; manifest con recuentos agregados;")
    lineas.append("  carpeta compartida).")
    lineas.append("- La ronda 2 incluyo DELIBERADAMENTE las respuestas de A/B: las decisiones")
    lineas.append("  finales NO son independientes de esas respuestas; no se declara")
    lineas.append("  independencia estadistica entre rondas.")
    lineas.append("- El gold permanece oculto: no se afirma ninguna correccion de etiquetas")
    lineas.append("  ni error confirmado. La coincidencia C1-C2 no prueba que el gold sea")
    lineas.append("  incorrecto.")
    lineas.append("- La reconstruccion de C2 (160 celdas sin cambio) asume que el archivo de")
    lineas.append("  correcciones contiene TODOS los cambios; no hay evidencia de otros.")
    SAL_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # ------------------------------------------------------------ procedencia
    lineas = []
    lineas.append("# Procedencia de las respuestas de la segunda ronda (recepcion 2026-10-13)")
    lineas.append("")
    lineas.append("Dato humano; se conserva SIN CAMBIOS (los textos pueden contener caracteres")
    lineas.append("no ASCII del original de las personas revisoras; no se convierten).")
    lineas.append("")
    lineas.append("- response_C1_round2.csv: transcripcion verbatim del archivo entregado por")
    lineas.append("  C1 (via mensaje del propietario). 176 filas; 9 columnas (sin text ni nota")
    lineas.append("  inicial; con A/B).")
    lineas.append("- response_C2_round2_correcciones.csv: transcripcion verbatim del archivo")
    lineas.append("  entregado por C2: SOLO registros corregidos (12 cambios) y casos que")
    lineas.append("  permanecen en UTD (4). Las 160 celdas restantes conservan su respuesta")
    lineas.append("  inicial de la ronda 1 (reconstruccion documentada).")
    lineas.append("- Las decisiones iniciales se verificaron contra los archivos archivados")
    lineas.append("  de la ronda 1 (sin cambios).")
    lineas.append("- El bloqueo previo (p93r4_reception_blocked.md) queda RESUELTO con esta")
    lineas.append("  recepcion por mensaje.")
    lineas.append("")
    lineas.append("Hashes (sha256 completo):")
    lineas.append("")
    for ruta in (RUTA_C1_R2, RUTA_C2_R2, RUTA_C1_R1, RUTA_C2_R1, SAL_DEC, SAL_MD,
                 SAL_AGR, SAL_PEND):
        lineas.append(f"- {ruta.name}: {sha256_archivo(ruta)}")
    SAL_PROV.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # ------------------------------------------------------------ manifiesto
    hashes = {}
    for ruta in (RUTA_C1_R2, RUTA_C2_R2, RUTA_C1_R1, RUTA_C2_R1, RUTA_ASSIGN,
                 SAL_PROV, SAL_DEC, SAL_MD, SAL_AGR, SAL_PEND):
        hashes[str(ruta.relative_to(RAIZ))] = sha256_archivo(ruta)
    verif = subprocess.run([sys.executable, str(RAIZ / "tools" / "verificar_caracteres.py")],
                           capture_output=True, text=True)
    chequeos["verificador_ascii_oficial"] = verif.returncode == 0
    resumen_verif = (verif.stdout or verif.stderr).strip().splitlines()[-1] if \
        (verif.stdout or verif.stderr).strip() else "sin salida"
    for clave_chk, valor in chequeos.items():
        if not valor:
            errores.append(f"chequeo fallido: {clave_chk}")
    estado = ("SECOND_ROUND_RESPONSES_VALIDATED" if not errores
              else "RECEPTION_WITH_ISSUES")
    manifiesto = {
        "phase": "P93-R4 second round reception, lock and analysis",
        "date": "2026-10-13",
        "status": estado,
        "decisions": "FINAL_HUMAN_DECISIONS_LOCKED" if not errores else "NOT_LOCKED",
        "gold_comparison": "NOT_YET_AUTHORIZED",
        "units": {"cells": 176, "unique_tweets": 127,
                  "HS": 23, "TR": 87, "AG": 66},
        "changes_from_round1": dict(cambios_por_et),
        "classification": dict(conteo_clase),
        "pending_cases": len(pendientes),
        "agreement": {et: {
            "raw_3cat": f"{metricas[et]['raw']}/{metricas[et]['total']}",
            "bin_cond": f"{metricas[et]['m3']['NN'] + metricas[et]['m3']['PP']}/{metricas[et]['bin_cells']}",
            "kappa3": None if metricas[et]["kappa3"] is None else round(metricas[et]["kappa3"], 4),
            "alpha_cat": None if metricas[et]["alpha_cat"] is None else round(metricas[et]["alpha_cat"], 4),
        } for et in ETIQUETAS},
        "hashes": hashes,
        "checks": chequeos,
        "avisos": avisos,
        "limitations": [
            "cegamiento ronda 1 no verificable completamente (P93-R3)",
            "ronda 2 con A/B visibles: decisiones no independientes de esas respuestas",
            "C2 entrego solo correcciones; reconstruccion documentada",
            "gold oculto; sin correcciones de etiquetas ni errores confirmados",
        ],
    }
    SAL_MANIFEST.write_text(json.dumps(manifiesto, ensure_ascii=True, indent=2) + "\n",
                            encoding="utf-8")

    print("P93-R4 recepcion/bloqueo/analisis: ejecutado")
    print(f"  C1: 176 filas, {len(cambios_c1)} cambios | C2: 16 correcciones "
          f"({len(cambios_c2)} cambios + {len(c2r2) - len(cambios_c2)} UTD retenidos)")
    for et in ETIQUETAS:
        m = metricas[et]
        k3 = "n/d" if m["kappa3"] is None else f"{m['kappa3']:.3f}"
        print(f"  {et}: bruto {m['raw']}/{m['total']} | bin {m['m3']['NN'] + m['m3']['PP']}/"
              f"{m['bin_cells']} | kappa3 {k3} | UTD C1={m['utd_a']} C2={m['utd_b']}")
    print(f"  exacto celdas {exacto_celdas}/176 | tweets {exacto_tweets}/127")
    print(f"  clasificacion: {dict(conteo_clase)} | pendientes: {len(pendientes)}")
    print(f"  verificador: {resumen_verif}")
    print(f"  chequeos: {sum(chequeos.values())}/{len(chequeos)} | errores: {errores if errores else 'ninguno'}")
    print(f"  estado: {estado}")
    if errores:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
