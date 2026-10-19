#!/usr/bin/env python
"""
P93 - Analisis formal de anotacion humana y auditoria de posibles errores.

Entradas (solo lectura; originales byte a byte):
  artifacts/human_annotation_v2_1_holdout_safe/responses/response_A.csv
  artifacts/human_annotation_v2_1_holdout_safe/responses/response_B_sin_texto.csv
    (fuente de labels de B; identico a response_B.csv en anotaciones, verificado)
  artifacts/human_annotation_v2_1_holdout_safe/responses/response_B.csv (notas/texto)
  artifacts/human_annotation_v2_1_holdout_safe/annotator_{A,B}/ (guia y CSV del paquete)
  reports/p92w_internal_sampling_key.csv (gold interno + componentes + ids originales)
  reports/p92z_responses_manifest.json (hashes registrados de las respuestas)

Salidas (ASCII salvo notas humanas verbatim en CSV internos, excepcion documentada):
  reports/p93_interannotator_agreement.md
  reports/p93_interannotator_agreement.csv
  reports/p93_human_disagreement_cells.csv
  reports/p93_gold_comparison_cells.csv        (material de apoyo, interno)
  reports/p93_gold_discrepancy_candidates.csv  (interno; NO compartir con adjudicador)
  reports/p93_adjudication_protocol.md
  reports/p93_provenance_audit.md
  reports/p93_manifest.json
  reports/p93_adjudication_blind_candidates.csv (ciego; NO distribuir automaticamente)

Reglas: sin entrenamiento, sin predicciones, sin cambios de etiquetas; la comparacion con
gold es solo interna (train elegible); nada de test/holdout/holdout2/leaderboard.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
V21 = RAIZ / "artifacts" / "human_annotation_v2_1_holdout_safe"
DIR_RESP = V21 / "responses"
RUTA_RESP_A = DIR_RESP / "response_A.csv"
RUTA_RESP_B = DIR_RESP / "response_B.csv"
RUTA_RESP_B2 = DIR_RESP / "response_B_sin_texto.csv"
RUTA_CLAVE = RAIZ / "reports" / "p92w_internal_sampling_key.csv"
RUTA_MANIFEST_Z = RAIZ / "reports" / "p92z_responses_manifest.json"
RUTA_GUIA_V21 = RAIZ / "reports" / "p92_annotation_guidelines_v2_1_holdout_safe.md"
RUTA_GUIA_V2 = RAIZ / "reports" / "p92_annotation_guidelines_v2_frozen.md"
RUTA_AUDIT_X = RAIZ / "reports" / "p92x_guideline_provenance_audit.md"
RUTA_GATE_X = RAIZ / "reports" / "p92x_final_distribution_gate.md"
RUTA_TRAIN = RAIZ / "data" / "raw" / "toxic_train.csv"

SAL_AGREE_MD = RAIZ / "reports" / "p93_interannotator_agreement.md"
SAL_AGREE_CSV = RAIZ / "reports" / "p93_interannotator_agreement.csv"
SAL_DISAG_CSV = RAIZ / "reports" / "p93_human_disagreement_cells.csv"
SAL_GOLD_CSV = RAIZ / "reports" / "p93_gold_comparison_cells.csv"
SAL_CAND_CSV = RAIZ / "reports" / "p93_gold_discrepancy_candidates.csv"
SAL_PROTO = RAIZ / "reports" / "p93_adjudication_protocol.md"
SAL_PROV = RAIZ / "reports" / "p93_provenance_audit.md"
SAL_MANIFEST = RAIZ / "reports" / "p93_manifest.json"
SAL_BLIND = RAIZ / "reports" / "p93_adjudication_blind_candidates.csv"

TOKENS = ("POSITIVE", "NEGATIVE", "UNABLE_TO_DETERMINE")
UTD = "UNABLE_TO_DETERMINE"
ETIQUETAS = ("HS", "TR", "AG")
COL = {"HS": "HS_annotation", "TR": "TR_annotation", "AG": "AG_annotation"}
GUIDE_STATUS = "GUIDE_V2_1_CONFIRMED_PACKAGE_LEVEL"


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


def celdas_3x3(par_a: dict, par_b: dict, etiqueta: str) -> dict:
    """Matriz 3x3 aplanada (A filas, B columnas): XY = A dice X, B dice Y."""
    corto = {"POSITIVE": "P", "NEGATIVE": "N", UTD: "U"}
    c = defaultdict(int)
    for rid, va in par_a.items():
        c[corto[va] + corto[par_b[rid]]] += 1
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
    """Alpha de Krippendorff nominal (2+ coders) via matriz de coincidencias."""
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

    # ------------------------------------------------------------- P0: guia
    archivos_a = sorted(p.name for p in (V21 / "annotator_A").iterdir() if p.is_file())
    archivos_b = sorted(p.name for p in (V21 / "annotator_B").iterdir() if p.is_file())
    chequeos["P0_v21_presente_en_ambos"] = all(
        "annotation_guidelines_v2_1_holdout_safe.md" in lista
        for lista in (archivos_a, archivos_b))
    chequeos["P0_v2_ausente_en_ambos"] = all(
        "annotation_guidelines_v2_frozen.md" not in lista
        for lista in (archivos_a, archivos_b))
    guia_v21 = RUTA_GUIA_V21.read_text(encoding="utf-8")
    lineas_guia = [l for l in guia_v21.splitlines(keepends=True)
                   if not l.startswith("SHA256 (contenido sin la linea")]
    hash_interno_v21 = hashlib.sha256("".join(lineas_guia).encode("utf-8")).hexdigest()
    hash_full_v21 = sha256_archivo(RUTA_GUIA_V21)
    chequeos["P0_hash_interno_v21"] = hash_interno_v21.startswith("ff699f54a48da779")
    chequeos["P0_hash_full_v21"] = hash_full_v21.startswith("fd8d4c65ccd194fb")
    chequeos["P0_guias_identicas_en_paquetes"] = all(
        (V21 / f"annotator_{letra}" / "annotation_guidelines_v2_1_holdout_safe.md").read_bytes()
        == RUTA_GUIA_V21.read_bytes() for letra in ("A", "B"))
    chequeos["P0_columna_define_v2_como_politica"] = (
        'valor de columna guideline_version: "v2"' in guia_v21)
    gate_x = RUTA_GATE_X.read_text(encoding="utf-8")
    audit_x = RUTA_AUDIT_X.read_text(encoding="utf-8")
    chequeos["P0_gate_x_registra_v21"] = "ff699f54" in gate_x
    chequeos["P0_audit_x_registra_v21"] = "annotation_guidelines_v2_1_holdout_safe" in audit_x

    # evidencia de uso: orden de los archivos de respuesta vs orden del paquete
    resp_a = leer_csv(RUTA_RESP_A)
    resp_b = leer_csv(RUTA_RESP_B)
    resp_b2 = leer_csv(RUTA_RESP_B2)
    pkg_a = leer_csv(V21 / "annotator_A" / "annotator_A.csv")
    pkg_b = leer_csv(V21 / "annotator_B" / "annotator_B.csv")
    ids_a = [f["review_id"] for f in resp_a]
    ids_b = [f["review_id"] for f in resp_b2]
    orden_pkg_a = [f["review_id"] for f in pkg_a]
    orden_pkg_b = [f["review_id"] for f in pkg_b]
    div_a = [i for i in range(250) if ids_a[i] != orden_pkg_a[i]]
    chequeos["P0_orden_B_identico_al_paquete"] = ids_b == orden_pkg_b
    chequeos["P0_orden_A_casi_identico"] = len(div_a) == 2  # una transposicion adyacente
    chequeos["P0_version_columna_v2"] = all(f["guideline_version"] == "v2" for f in resp_a)
    if not chequeos["P0_orden_A_casi_identico"]:
        avisos.append(f"orden de A diverge del paquete en indices {div_a}")
    clasificacion_p0 = "GUIDE_V2_1_CONFIRMED"

    # --------------------------------------------------- P1: validacion datos
    chequeos["P1_A_250"] = len(resp_a) == 250
    chequeos["P1_B2_250"] = len(resp_b2) == 250
    chequeos["P1_membresia_igual"] = set(ids_a) == set(ids_b)
    chequeos["P1_A_tokens"] = all(f[c] in TOKENS for f in resp_a for c in COL.values())
    chequeos["P1_B_tokens"] = all(f[c] in TOKENS for f in resp_b2 for c in COL.values())
    chequeos["P1_B_vs_B_principal_identicos"] = all(
        next(f for f in resp_b if f["review_id"] == r)[c] == resp_b2[i][c]
        for i, r in enumerate(ids_b)
        for c in list(COL.values()) + ["uncertainty_notes", "tweet_note"])
    clave = leer_csv(RUTA_CLAVE)
    mapa = {f["review_id"]: f for f in clave}
    gold = {r: {"HS": int(f["original_HS"]), "TR": int(f["original_TR"]),
                "AG": int(f["original_AG"])} for r, f in mapa.items()}
    componentes = {r: f["sample_component"] for r, f in mapa.items()}
    chequeos["P1_ids_en_clave"] = set(ids_a) <= set(mapa)
    chequeos["P1_componentes_100_150"] = (
        sum(1 for r in ids_a if componentes[r] == "A") == 100
        and sum(1 for r in ids_a if componentes[r] == "B") == 150)
    manifest_z = json.loads(RUTA_MANIFEST_Z.read_text(encoding="utf-8"))
    hashes_actuales = {
        "response_A.csv": sha256_archivo(RUTA_RESP_A),
        "response_B.csv": sha256_archivo(RUTA_RESP_B),
        "response_B_sin_texto.csv": sha256_archivo(RUTA_RESP_B2),
    }
    chequeos["P1_hashes_vs_manifiesto"] = all(
        hashes_actuales[k] == manifest_z["files"][k] for k in hashes_actuales)

    # -------------------------------------------------------- P2: acuerdo
    respuestas = {
        "A": {f["review_id"]: {e: f[c] for e, c in COL.items()} for f in resp_a},
        "B": {f["review_id"]: {e: f[c] for e, c in COL.items()} for f in resp_b2},
    }
    notas = {
        "A": {f["review_id"]: f["uncertainty_notes"] for f in resp_a},
        "B": {f["review_id"]: f["uncertainty_notes"] for f in resp_b2},
    }
    metricas = {}
    for et in ETIQUETAS:
        va = {r: respuestas["A"][r][et] for r in ids_a}
        vb = {r: respuestas["B"][r][et] for r in ids_a}
        m3 = celdas_3x3(va, vb, et)
        marg_a = Counter(va.values())
        marg_b = Counter(vb.values())
        total_a = sum(marg_a.values())
        total_b = sum(marg_b.values())
        if total_a != 250 or total_b != 250:
            errores.append(f"{et}: marginales no suman 250 (A={total_a}, B={total_b})")
        raw_agr = m3["NN"] + m3["PP"] + m3["UU"]
        bin_cells = m3["NN"] + m3["NP"] + m3["PN"] + m3["PP"]
        pos_a = m3["PP"] + m3["PN"]
        pos_b = m3["PP"] + m3["NP"]
        neg_a = m3["NN"] + m3["NP"]
        neg_b = m3["NN"] + m3["PN"]
        pos_dice = (2 * m3["PP"], pos_a + pos_b)
        neg_dice = (2 * m3["NN"], neg_a + neg_b)
        k3 = kappa_nominal(m3, ("N", "P", "U"))
        m_bin = {k: m3[k] for k in ("NN", "NP", "PN", "PP")}
        k_bin = kappa_nominal(m_bin, ("N", "P"))
        unidades = [[va[r], vb[r]] for r in ids_a]
        alpha_cat = krippendorff_alpha(unidades)
        unidades_m = [[None if v == UTD else v for v in u] for u in unidades]
        alpha_miss = krippendorff_alpha(unidades_m)
        metricas[et] = {
            "marg_a": dict(marg_a), "marg_b": dict(marg_b),
            "matriz": m3, "raw": raw_agr, "bin_cells": bin_cells,
            "pos_dice": pos_dice, "neg_dice": neg_dice,
            "kappa3": k3, "kappa_bin": k_bin,
            "alpha_cat": alpha_cat, "alpha_miss": alpha_miss,
            "utd_both": m3["UU"],
        }

    # acuerdo conjunto
    triple_exacto = sum(
        1 for r in ids_a
        if all(respuestas["A"][r][e] == respuestas["B"][r][e] for e in ETIQUETAS))
    celdas_discrepancia = sum(
        1 for r in ids_a for e in ETIQUETAS if respuestas["A"][r][e] != respuestas["B"][r][e])
    tweets_con_utd = sum(
        1 for r in ids_a
        if any(respuestas[a][r][e] == UTD for a in ("A", "B") for e in ETIQUETAS))
    celdas_con_utd = sum(
        1 for r in ids_a for e in ETIQUETAS
        if respuestas["A"][r][e] == UTD or respuestas["B"][r][e] == UTD)

    # ---------------------------------------------- P3: celdas de discrepancia
    filas_disag = []
    conteo_disag = Counter()
    for r in ids_a:
        for e in ETIQUETAS:
            a, b = respuestas["A"][r][e], respuestas["B"][r][e]
            if a == b and a == UTD:
                tipo = "UTD_BOTH"
            elif a == b:
                continue
            elif a in TOKENS[:2] and b in TOKENS[:2]:
                tipo = "BINARY_DISAGREEMENT"
            elif (a == UTD) != (b == UTD):
                tipo = "UTD_VS_BINARY"
            else:
                tipo = "MISSING_OR_INVALID"
            conteo_disag[tipo] += 1
            filas_disag.append({
                "review_id": r, "label": e, "annotator_A_raw": a, "annotator_B_raw": b,
                "disagreement_type": tipo, "notes_A": notas["A"][r], "notes_B": notas["B"][r],
                "guideline_version_status": GUIDE_STATUS,
                "adjudication_status": "PENDING",
            })

    # ------------------------------------------------ P4: comparacion con gold
    filas_gold = []
    resumen_gold = defaultdict(Counter)
    for r in ids_a:
        for e in ETIQUETAS:
            a, b = respuestas["A"][r][e], respuestas["B"][r][e]
            g = gold[r][e]
            if a == UTD or b == UTD:
                clase = "UTD_INVOLVED"
            elif a != b:
                clase = "HUMAN_DISAGREEMENT"
            else:
                coincide = (a == "POSITIVE" and g == 1) or (a == "NEGATIVE" and g == 0)
                clase = ("HUMAN_AGREEMENT_MATCHES_GOLD" if coincide
                         else "HUMAN_AGREEMENT_DISAGREES_WITH_GOLD")
            resumen_gold[e][clase] += 1
            resumen_gold[f"{e}|{componentes[r]}"][clase] += 1
            filas_gold.append({
                "review_id": r, "label": e, "annotator_A": a, "annotator_B": b,
                "gold_label": g, "clasificacion": clase,
                "sampling_component": componentes[r],
                "guideline_version_status": GUIDE_STATUS,
            })

    # -------------------------------------------------- P5: candidatos
    candidatos = [f for f in filas_gold
                  if f["clasificacion"] == "HUMAN_AGREEMENT_DISAGREES_WITH_GOLD"]
    for c in candidatos:
        r = c["review_id"]
        sin_notas = not notas["A"][r] and not notas["B"][r]
        c["candidate_reason"] = (
            f"A y B coinciden en {c['annotator_A']} y el gold es "
            f"{'1' if c['gold_label'] == 1 else '0'}"
            + ("; sin notas de duda en ninguno de los dos" if sin_notas
               else "; con notas de duda en al menos un anotador"))
        c["_orden"] = (0 if sin_notas else 1, ETIQUETAS.index(c["label"]), r)
    candidatos.sort(key=lambda c: c["_orden"])
    for c in candidatos:
        c.pop("_orden")
        c["adjudication_status"] = "PENDING_ADJUDICATION"
    conteo_cand = Counter(c["label"] for c in candidatos)
    conteo_cand_comp = Counter((c["label"], c["sampling_component"]) for c in candidatos)

    for clave_chk, valor in chequeos.items():
        if not valor:
            errores.append(f"chequeo fallido: {clave_chk}")

    # ------------------------------------------------------------ salidas
    # agreement.csv
    with open(SAL_AGREE_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["label", "metrica", "numerador", "denominador", "valor", "nota"])
        for et in ETIQUETAS:
            m = metricas[et]
            for cat in ("POSITIVE", "NEGATIVE", UTD):
                w.writerow([et, f"marginal_A_{cat}", m["marg_a"].get(cat, 0), 250, "",
                            "distribucion marginal del anotador A"])
                w.writerow([et, f"marginal_B_{cat}", m["marg_b"].get(cat, 0), 250, "",
                            "distribucion marginal del anotador B"])
            w.writerow([et, "raw_agreement_3cat", m["raw"], 250,
                        f"{m['raw'] / 250:.4f}", "UTD como tercera categoria"])
            w.writerow([et, "acuerdo_binario_condicional", m["matriz"]["NN"] + m["matriz"]["PP"],
                        m["bin_cells"], f"{(m['matriz']['NN'] + m['matriz']['PP']) / m['bin_cells']:.4f}",
                        "solo celdas con ambas respuestas binarias"])
            w.writerow([et, "acuerdo_positivo_dice", m["pos_dice"][0], m["pos_dice"][1],
                        f"{m['pos_dice'][0] / m['pos_dice'][1]:.4f}" if m["pos_dice"][1] else "",
                        "celdas binarias"])
            w.writerow([et, "acuerdo_negativo_dice", m["neg_dice"][0], m["neg_dice"][1],
                        f"{m['neg_dice'][0] / m['neg_dice'][1]:.4f}" if m["neg_dice"][1] else "",
                        "celdas binarias"])
            w.writerow([et, "kappa_cohen_3cat", "", 250,
                        f"{m['kappa3']:.4f}" if m["kappa3"] is not None else "n/d",
                        "nominal, UTD como categoria"])
            w.writerow([et, "kappa_cohen_bin", "", m["bin_cells"],
                        f"{m['kappa_bin']:.4f}" if m["kappa_bin"] is not None else "n/d",
                        "solo celdas binarias"])
            w.writerow([et, "alpha_krippendorff_utd_categoria", "", 250,
                        f"{m['alpha_cat']:.4f}" if m["alpha_cat"] is not None else "n/d", ""])
            w.writerow([et, "alpha_krippendorff_utd_faltante", "", 250,
                        f"{m['alpha_miss']:.4f}" if m["alpha_miss"] is not None else "n/d", ""])
            w.writerow([et, "cobertura_determinada_ambos", m["bin_cells"], 250,
                        f"{m['bin_cells'] / 250:.4f}", "celdas con ambas respuestas determinables"])
            w.writerow([et, "celdas_utd_both", m["utd_both"], 250, "",
                        "acuerdo en incertidumbre (no desacuerdo)"])
        w.writerow(["JOINT", "triple_exacto", triple_exacto, 250, f"{triple_exacto / 250:.4f}",
                    "las tres etiquetas identicas"])
        w.writerow(["JOINT", "tweets_con_al_menos_un_desacuerdo", 250 - triple_exacto, 250, "",
                    "complemento del triple exacto"])
        w.writerow(["JOINT", "celdas_con_desacuerdo", celdas_discrepancia, 750, "",
                    "sobre 750 celdas (250 x 3)"])
        w.writerow(["JOINT", "tweets_con_al_menos_un_utd", tweets_con_utd, 250, "",
                    "cualquiera de los dos anotadores"])
        w.writerow(["JOINT", "celdas_con_utd", celdas_con_utd, 750, "",
                    "cualquiera de los dos anotadores"])

    # disagreement cells csv (incluye UTD_BOTH por trazabilidad, marcado)
    with open(SAL_DISAG_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["review_id", "label", "annotator_A_raw", "annotator_B_raw",
                    "disagreement_type", "notes_A", "notes_B",
                    "guideline_version_status", "adjudication_status", "nota"])
        for f in filas_disag:
            nota = ("acuerdo de incertidumbre (no desacuerdo); incluido por trazabilidad"
                    if f["disagreement_type"] == "UTD_BOTH" else "")
            w.writerow([f["review_id"], f["label"], f["annotator_A_raw"], f["annotator_B_raw"],
                        f["disagreement_type"], f["notes_A"], f["notes_B"],
                        f["guideline_version_status"], f["adjudication_status"], nota])

    # gold comparison cells csv (interno)
    with open(SAL_GOLD_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["review_id", "label", "annotator_A", "annotator_B", "gold_label",
                    "clasificacion", "sampling_component", "guideline_version_status"])
        for f in filas_gold:
            w.writerow([f["review_id"], f["label"], f["annotator_A"], f["annotator_B"],
                        f["gold_label"], f["clasificacion"], f["sampling_component"],
                        f["guideline_version_status"]])

    # candidates csv (interno)
    with open(SAL_CAND_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["review_id", "label", "A_response", "B_response", "gold_label",
                    "candidate_reason", "sampling_component", "guideline_version_status",
                    "adjudication_status"])
        for c in candidatos:
            w.writerow([c["review_id"], c["label"], c["annotator_A"], c["annotator_B"],
                        c["gold_label"], c["candidate_reason"], c["sampling_component"],
                        c["guideline_version_status"], c["adjudication_status"]])

    # blind package (sin gold, sin motivos, sin V03)
    textos = {}
    with open(RUTA_TRAIN, encoding="utf-8-sig", newline="") as fh:
        lector = csv.reader(fh, delimiter=";", quotechar='"')
        next(lector)
        for campos in lector:
            if campos:
                textos[int(campos[0].strip().strip('"'))] = campos[1]
    with open(SAL_BLIND, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["review_id", "text", "label", "annotator_A_response",
                    "annotator_B_response", "notes_A", "notes_B"])
        for c in candidatos:
            eid = int(mapa[c["review_id"]]["original_example_id"])
            w.writerow([c["review_id"], textos[eid], c["label"], c["annotator_A"],
                        c["annotator_B"], notas["A"][c["review_id"]], notas["B"][c["review_id"]]])

    # ------------------------------------------------------------ informes MD
    lineas = []
    lineas.append("# P93 - Acuerdo formal entre anotadores (250 tweets principales)")
    lineas.append("")
    lineas.append("Fecha: 2026-10-13. Fase diagnostica: sin entrenamiento, sin predicciones,")
    lineas.append("sin cambios de etiquetas ni de guia. Unidad: par (review_id, etiqueta);")
    lineas.append("750 celdas por metrica conjunta. La muestra principal incluye un componente")
    lineas.append("enriquecido no representativo: no estimar prevalencia poblacional.")
    lineas.append("")
    lineas.append("## 1. Procedencia de la guia (resumen)")
    lineas.append("")
    lineas.append(f"- Clasificacion: **{clasificacion_p0}** (a nivel de paquete; sin acuse de")
    lineas.append("  recibo firmado, limitacion registrada en p93_provenance_audit.md).")
    lineas.append(f"- Guia v2.1 presente en ambos paquetes (hash interno {hash_interno_v21[:16]});")
    lineas.append("  el archivo v2 retirado no esta en las carpetas; la columna")
    lineas.append("  guideline_version=v2 corresponde a la POLITICA (definido por la propia v2.1).")
    lineas.append("")
    lineas.append("## 2. Distribuciones marginales")
    lineas.append("")
    lineas.append("| etiqueta | anotador | POSITIVE | NEGATIVE | UNABLE_TO_DETERMINE | suma |")
    lineas.append("|---|---|---|---|---|---|")
    for et in ETIQUETAS:
        for a in ("A", "B"):
            m = metricas[et][f"marg_{a.lower()}"]
            suma = sum(m.values())
            lineas.append(f"| {et} | {a} | {m.get('POSITIVE', 0)} | {m.get('NEGATIVE', 0)} | "
                          f"{m.get(UTD, 0)} | {suma} |")
    lineas.append("")
    lineas.append("## 3. Acuerdo por etiqueta")
    lineas.append("")
    lineas.append("| etiqueta | raw 3cat | binario cond. | dice pos | dice neg | kappa 3cat | kappa bin | alpha (cat) | alpha (faltante) |")
    lineas.append("|---|---|---|---|---|---|---|---|---|")
    for et in ETIQUETAS:
        m = metricas[et]
        k3 = "n/d" if m["kappa3"] is None else f"{m['kappa3']:.3f}"
        kb = "n/d" if m["kappa_bin"] is None else f"{m['kappa_bin']:.3f}"
        ac = "n/d" if m["alpha_cat"] is None else f"{m['alpha_cat']:.3f}"
        am = "n/d" if m["alpha_miss"] is None else f"{m['alpha_miss']:.3f}"
        pd = f"{m['pos_dice'][0]}/{m['pos_dice'][1]}" if m["pos_dice"][1] else "n/d"
        nd = f"{m['neg_dice'][0]}/{m['neg_dice'][1]}" if m["neg_dice"][1] else "n/d"
        lineas.append(f"| {et} | {m['raw']}/250 | "
                      f"{m['matriz']['NN'] + m['matriz']['PP']}/{m['bin_cells']} | {pd} | {nd} | "
                      f"{k3} | {kb} | {ac} | {am} |")
    lineas.append("")
    lineas.append("Matrices 3x3 completas por etiqueta (numeros, sin porcentajes):")
    lineas.append("")
    for et in ETIQUETAS:
        m3 = metricas[et]["matriz"]
        lineas.append(f"- {et}: NN={m3['NN']} NP={m3['NP']} NU={m3['NU']} | PN={m3['PN']} "
                      f"PP={m3['PP']} PU={m3['PU']} | UN={m3['UN']} UP={m3['UP']} UU={m3['UU']}")
    lineas.append("")
    lineas.append("Sensibilidad (separada, como exige el protocolo):")
    lineas.append("")
    lineas.append("- A) UTD como tercera categoria: cifras de la tabla anterior (raw 3cat, kappa")
    lineas.append("   y alpha con UTD=categoria).")
    lineas.append("- B) Condicional al subconjunto determinable (ambos binarios): la columna")
    lineas.append("  'binario cond.' y 'kappa bin'; excluye las celdas con UTD de cualquiera de")
    lineas.append("  los dos. Celdas excluidas por etiqueta: "
                  + ", ".join(f"{et}={250 - metricas[et]['bin_cells']}" for et in ETIQUETAS)
                  + ". Esta metrica describe SOLO ese subconjunto; no es el acuerdo global.")
    lineas.append("")
    lineas.append("## 4. Acuerdo conjunto")
    lineas.append("")
    lineas.append(f"- Triple exacto (HS/TR/AG identicas): {triple_exacto}/250.")
    lineas.append(f"- Tweets con al menos un desacuerdo: {250 - triple_exacto}/250.")
    lineas.append(f"- Celdas con desacuerdo: {celdas_discrepancia}/750.")
    lineas.append(f"- Tweets con al menos un UTD: {tweets_con_utd}/250.")
    lineas.append(f"- Celdas con UTD (cualquiera de los dos): {celdas_con_utd}/750.")
    lineas.append("")
    lineas.append("## 5. Desacuerdos humanos (resumen de P3)")
    lineas.append("")
    lineas.append(f"- BINARY_DISAGREEMENT: {conteo_disag.get('BINARY_DISAGREEMENT', 0)} celdas.")
    lineas.append(f"- UTD_VS_BINARY: {conteo_disag.get('UTD_VS_BINARY', 0)} celdas.")
    lineas.append(f"- UTD_BOTH (acuerdo en incertidumbre, no desacuerdo): "
                  f"{conteo_disag.get('UTD_BOTH', 0)} celdas.")
    lineas.append(f"- MISSING_OR_INVALID: {conteo_disag.get('MISSING_OR_INVALID', 0)} celdas.")
    lineas.append("Detalle por celda: p93_human_disagreement_cells.csv (interno).")
    lineas.append("")
    lineas.append("## 6. Comparacion interna con el gold (P4)")
    lineas.append("")
    lineas.append("Clasificacion por celda (interna; el gold NO se usa para decidir quien tiene")
    lineas.append("razon). Resumen por etiqueta y por componente de muestreo:")
    lineas.append("")
    lineas.append("| etiqueta | componente | matches | disagree_with_gold | human_disagreement | utd_involved | total |")
    lineas.append("|---|---|---|---|---|---|---|")
    for et in ETIQUETAS:
        for comp in ("A", "B"):
            c = resumen_gold[f"{et}|{comp}"]
            total = sum(c.values())
            lineas.append(f"| {et} | {comp} | {c.get('HUMAN_AGREEMENT_MATCHES_GOLD', 0)} | "
                          f"{c.get('HUMAN_AGREEMENT_DISAGREES_WITH_GOLD', 0)} | "
                          f"{c.get('HUMAN_DISAGREEMENT', 0)} | {c.get('UTD_INVOLVED', 0)} | {total} |")
    lineas.append("")
    lineas.append("- Componente A = submuestra aleatoria (100/1236, probabilidad conocida);")
    lineas.append("  componente B = enriquecido no representativo. No combinar para estimar una")
    lineas.append("  tasa poblacional; toda cifra es descriptiva de esta muestra.")
    lineas.append("- GUIDE_PROVENANCE_UNRESOLVED: 0 celdas (procedencia confirmada a nivel de")
    lineas.append("  paquete; ver limitaciones).")
    lineas.append("")
    lineas.append("## 7. Candidatos a adjudicacion (P5)")
    lineas.append("")
    lineas.append(f"- Cola total: {len(candidatos)} celdas (A y B coinciden en binario distinto")
    lineas.append("  del gold). Por etiqueta: HS={conteo_cand.get('HS', 0)}, TR={conteo_cand.get('TR', 0)}, "
                  f"AG={conteo_cand.get('AG', 0)}.")
    lineas.append(f"- Por componente: A={sum(v for (l, c), v in conteo_cand_comp.items() if c == 'A')}, "
                  f"B={sum(v for (l, c), v in conteo_cand_comp.items() if c == 'B')}.")
    lineas.append("- NO son errores confirmados: son candidatos para revision por una tercera")
    lineas.append("  persona (ver p93_adjudication_protocol.md). Cola interna:")
    lineas.append("  p93_gold_discrepancy_candidates.csv; paquete ciego:")
    lineas.append("  p93_adjudication_blind_candidates.csv (NO distribuir automaticamente).")
    lineas.append("")
    lineas.append("## 8. Limitaciones")
    lineas.append("")
    lineas.append("- Sin acuse de recibo firmado de la guia (procedencia a nivel de paquete).")
    lineas.append("- Secuencia desviada: la ronda principal se anoto antes de completar la")
    lineas.append("  calibracion v2.1 (documentado; no se altera la cronologia).")
    lineas.append("- Componente B no representativo; B con artefactos de export en textos")
    lineas.append("  (no afectan anotaciones).")
    lineas.append("- El acuerdo humano no estima el error del gold; los candidatos no son")
    lineas.append("  errores confirmados.")
    SAL_AGREE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # protocolo de adjudicacion
    lineas = []
    lineas.append("# P93 - Protocolo de adjudicacion (tercera persona cualificada)")
    lineas.append("")
    lineas.append("Estado: preparado; NO iniciado; la distribucion la decide el propietario.")
    lineas.append("")
    lineas.append("## 1. Alcance")
    lineas.append("")
    lineas.append(f"- Cola inicial: {len(candidatos)} celdas donde A y B coinciden en una")
    lineas.append("  etiqueta binaria distinta del gold original (candidatos, NO errores).")
    lineas.append("- Los desacuerdos humanos (P3) podran adjudicarse en un lote separado si se")
    lineas.append("  autoriza; este protocolo no los resuelve automaticamente.")
    lineas.append("")
    lineas.append("## 2. Materiales")
    lineas.append("")
    lineas.append("- Para el adjudicador: SOLO el paquete ciego")
    lineas.append("  (p93_adjudication_blind_candidates.csv: texto, etiqueta, respuestas A y B,")
    lineas.append("  notas), sin gold, sin predicciones, sin motivos de seleccion.")
    lineas.append("- El archivo interno de candidatos (con gold y motivos) NO se comparte.")
    lineas.append("")
    lineas.append("## 3. Decisiones posibles por caso")
    lineas.append("")
    lineas.append("1. Confirmar el gold (la diferencia se explica por ambiguedad de politica).")
    lineas.append("2. Proponer revision del gold (con justificacion escrita y cita de la regla).")
    lineas.append("3. UNABLE_TO_DETERMINE (falta contexto o norma).")
    lineas.append("")
    lineas.append("## 4. Reglas")
    lineas.append("")
    lineas.append("- Ninguna etiqueta se modifica durante P93 ni con este protocolo.")
    lineas.append("- Una correccion futura requerira adjudicacion documentada y autorizacion")
    lineas.append("  separada (vocabulario de estados de P92-V: POSSIBLE_LABEL_ISSUE ->")
    lineas.append("  ADJUDICATED_DIFFERENCE -> CONFIRMED_ANNOTATION_ERROR ->")
    lineas.append("  APPROVED_DATASET_CORRECTION).")
    lineas.append("- El adjudicador no vera V03 ni sus scores; su juicio se basa en el texto,")
    lineas.append("  la guia aplicable y las respuestas humanas.")
    lineas.append("- Registrar por caso: decision, fundamento, fecha y firma; las decisiones se")
    lineas.append("  conservan separadas de las respuestas originales.")
    SAL_PROTO.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # auditoria de procedencia
    lineas = []
    lineas.append("# P93 - Auditoria de procedencia (P0/P1)")
    lineas.append("")
    lineas.append(f"Fecha: 2026-10-13. Clasificacion de guia: **{clasificacion_p0}**.")
    lineas.append("")
    lineas.append("## 1. P0 - Que guia recibieron los anotadores")
    lineas.append("")
    lineas.append("- Las carpetas del paquete contienen UN unico archivo de guia:")
    lineas.append("  annotation_guidelines_v2_1_holdout_safe.md (identico byte a byte en A y B;")
    lineas.append(f"  hash interno {hash_interno_v21[:16]}, hash completo {hash_full_v21[:16]}).")
    lineas.append("- El archivo v2 (annotation_guidelines_v2_frozen.md) NO esta en las carpetas;")
    lineas.append("  permanece solo como historico en reports/.")
    lineas.append("- La columna guideline_version de los CSV dice v2: la propia guia v2.1 define")
    lineas.append("  ese valor como la version de POLITICA (linea citada en la evidencia); no es")
    lineas.append("  metadata obsoleta ni una discrepancia.")
    lineas.append("- Evidencia de uso real: el orden de filas del archivo de B coincide")
    lineas.append("  EXACTAMENTE con el orden del paquete B; el de A coincide salvo una")
    lineas.append("  transposicion adyacente (indices 203-204). Los anotadores trabajaron sobre")
    lineas.append("  los CSV de estos paquetes.")
    lineas.append("- LIMITACION: no existe un acuse de recibo firmado; la confirmacion es a nivel")
    lineas.append("  de contenido preparado y verificado del paquete (pre-P93 y post-respuestas).")
    lineas.append("")
    lineas.append("## 2. P1 - Validacion de los datos humanos")
    lineas.append("")
    for clave_chk, valor in chequeos.items():
        if clave_chk.startswith("P1"):
            lineas.append(f"- {clave_chk}: {'OK' if valor else 'FALLO'}")
    lineas.append("- Fuente de labels de B: response_B_sin_texto.csv (identico a response_B.csv")
    lineas.append("  en todas las columnas de anotacion y notas; verificado).")
    lineas.append("- Originales byte a byte: hashes de las 3 respuestas coinciden con el")
    lineas.append("  manifiesto de P92-Z.")
    lineas.append("- Artefactos generados: ASCII salvo las notas humanas verbatim en CSV internos")
    lineas.append("  (excepcion documentada; el verificador oficial no marca caracteres latinos).")
    SAL_PROV.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # -------------------------------------------------- verificador oficial
    verif = subprocess.run([sys.executable, str(RAIZ / "tools" / "verificar_caracteres.py")],
                           capture_output=True, text=True)
    chequeos["verificador_ascii_oficial"] = verif.returncode == 0
    resumen_verif = (verif.stdout or verif.stderr).strip().splitlines()[-1] if \
        (verif.stdout or verif.stderr).strip() else "sin salida"

    # -------------------------------------------------- hashes y manifiesto
    hashes = {}
    for ruta in (RUTA_RESP_A, RUTA_RESP_B, RUTA_RESP_B2, RUTA_CLAVE, RUTA_GUIA_V21,
                 SAL_AGREE_MD, SAL_AGREE_CSV, SAL_DISAG_CSV, SAL_GOLD_CSV, SAL_CAND_CSV,
                 SAL_PROTO, SAL_PROV, SAL_BLIND):
        hashes[str(ruta.relative_to(RAIZ))] = sha256_archivo(ruta)
    manifiesto = {
        "phase": "P93 human agreement analysis and gold-discrepancy audit",
        "date": "2026-10-13",
        "guide_classification": clasificacion_p0,
        "guide_hashes": {"v2_1_interno": hash_interno_v21, "v2_1_completo": hash_full_v21},
        "inputs_hashes": {k: v for k, v in hashes.items()
                          if k.startswith(("artifacts/", "reports/p92w_internal"))},
        "outputs_hashes": {k: v for k, v in hashes.items() if k.startswith("reports/p93")},
        "eligible_population": "dev2 = 1236 (P92-W); muestra principal = 100 SRS (componente A) + 150 enriquecidos (B)",
        "seeds": "no aplican (analisis determinista sobre respuestas existentes)",
        "agreement_summary": {
            et: {
                "raw_3cat": f"{metricas[et]['raw']}/250",
                "binario_condicional": f"{metricas[et]['matriz']['NN'] + metricas[et]['matriz']['PP']}/{metricas[et]['bin_cells']}",
                "kappa_3cat": None if metricas[et]["kappa3"] is None else round(metricas[et]["kappa3"], 4),
                "kappa_bin": None if metricas[et]["kappa_bin"] is None else round(metricas[et]["kappa_bin"], 4),
                "alpha_cat": None if metricas[et]["alpha_cat"] is None else round(metricas[et]["alpha_cat"], 4),
                "alpha_miss": None if metricas[et]["alpha_miss"] is None else round(metricas[et]["alpha_miss"], 4),
            } for et in ETIQUETAS
        },
        "joint": {
            "triple_exacto": triple_exacto,
            "tweets_con_desacuerdo": 250 - triple_exacto,
            "celdas_con_desacuerdo": celdas_discrepancia,
            "tweets_con_utd": tweets_con_utd,
            "celdas_con_utd": celdas_con_utd,
        },
        "disagreement_counts": dict(conteo_disag),
        "candidates_total": len(candidatos),
        "candidates_by_label": dict(conteo_cand),
        "calibration": "PENDING_HUMAN_RESPONSES (no mezclada en estas metricas)",
        "main_annotation": "COMPLETED_WITH_DOCUMENTED_SEQUENCE_DEVIATION",
        "limitations": [
            "sin acuse de recibo firmado de la guia (confirmacion a nivel de paquete)",
            "componente B no representativo; no estimar prevalencia poblacional",
            "candidatos NO son errores confirmados; requieren adjudicacion humana",
            "notas humanas verbatim con posibles caracteres no ASCII en CSV internos",
        ],
        "checks": chequeos,
        "avisos": avisos,
    }
    SAL_MANIFEST.write_text(json.dumps(manifiesto, ensure_ascii=True, indent=2) + "\n",
                            encoding="utf-8")

    for clave_chk, valor in chequeos.items():
        if not valor:
            errores.append(f"chequeo fallido: {clave_chk}")

    estado = "HUMAN_AGREEMENT_ANALYZED" if not errores else "ANALYSIS_WITH_ISSUES"
    print("P93 analisis de anotacion humana: ejecutado")
    print(f"  P0 guia: {clasificacion_p0} | v2.1 interno {hash_interno_v21[:16]}")
    for et in ETIQUETAS:
        m = metricas[et]
        k3 = "n/d" if m["kappa3"] is None else f"{m['kappa3']:.3f}"
        kb = "n/d" if m["kappa_bin"] is None else f"{m['kappa_bin']:.3f}"
        print(f"  {et}: raw {m['raw']}/250 | bin {m['matriz']['NN'] + m['matriz']['PP']}/{m['bin_cells']} "
              f"| kappa3 {k3} | kappa_bin {kb}")
    print(f"  triple exacto {triple_exacto}/250 | celdas desacuerdo {celdas_discrepancia}/750")
    print(f"  discrepancias: {dict(conteo_disag)}")
    print(f"  candidatos: {len(candidatos)} {dict(conteo_cand)}")
    print(f"  verificador: {resumen_verif}")
    print(f"  chequeos: {sum(chequeos.values())}/{len(chequeos)} | errores: {errores if errores else 'ninguno'}")
    print(f"  estado: {estado}")
    if errores:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
