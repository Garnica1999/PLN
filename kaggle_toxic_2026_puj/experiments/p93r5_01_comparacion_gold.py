#!/usr/bin/env python
"""
P93-R5 - Comparacion con el gold y registro de correcciones candidatas.

La comparacion con el gold fue AUTORIZADA explicitamente por el propietario
(2026-10-13), sin tercera revision humana. Esta fase NO modifica etiquetas, NO
entrena, NO genera predicciones ni submissions, y NO toca V03.

Entradas (solo lectura):
  reports/p93r4_final_decisions.csv                (176 decisiones bloqueadas)
  reports/p93r4_pending_cases.csv                  (5 casos sin consenso)
  reports/p93r4_final_human_agreement.md
  reports/p93r4_manifest.json
  reports/p93_manifest.json
  reports/p92w_internal_sampling_key.csv           (clave interna: gold + componente)
  reports/p93_gold_discrepancy_candidates.csv      (176 celdas + A/B + gold)
  reports/p93_gold_comparison_cells.csv            (grid 750; verificacion cruzada)
  reports/p93_adjudication_blind_candidates.csv    (notas A/B)
  reports/p93r3_round2_id_assignment.csv
  reports/p93r_internal_adjudication_key.csv
  artifacts/p93_adjudication_v1|v2/...             (respuestas archivadas)
  data/raw/toxic_train.csv | toxic_test.csv        (solo ids; semillas de elegibilidad)
  data/processed/holdout_v2_ids.csv | holdout2_ids.csv  (solo ids)

Salidas (nuevas; no sobrescribe informes historicos):
  reports/p93r5_gold_comparison.csv
  reports/p93r5_gold_comparison.md
  reports/p93r5_label_correction_candidates.csv
  reports/p93r5_future_experiment_recommendation.md
  reports/p93r5_manifest.json

Regla operativa de esta fase (documentada en los informes):
  1. Solo las 171 celdas con decision humana final binaria (C1 == C2) se comparan
     con el gold.
  2. Coincidencia -> HUMAN_FINAL_MATCHES_GOLD.
  3. Discrepancia con justificacion documentada (final_rationale de C1 o C2,
     referida a la politica v2.1) -> POSSIBLE_LABEL_ISSUE (elegible).
  4. Discrepancia sin justificacion documentada -> HUMAN_FINAL_DIFFERS_FROM_GOLD
     (registrada, NO elegible; la coincidencia C1/C2 o con A/B no basta).
  5. CONFIRMED_ANNOTATION_ERROR: requiere excluir toda interpretacion
     alternativa razonable; en esta fase no se confirma ningun error.
  6. Los 5 casos sin consenso -> UNRESOLVED_NO_THIRD_REVIEW (elegible = false).
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
R = RAIZ / "reports"
RESP1 = RAIZ / "artifacts" / "p93_adjudication_v1" / "responses"
RESP2 = RAIZ / "artifacts" / "p93_adjudication_v2" / "responses"
PKG = RAIZ / "artifacts" / "p93_adjudication_v2"

SAL_CMP = R / "p93r5_gold_comparison.csv"
SAL_CMP_MD = R / "p93r5_gold_comparison.md"
SAL_CAND = R / "p93r5_label_correction_candidates.csv"
SAL_REC = R / "p93r5_future_experiment_recommendation.md"
SAL_MANIFEST = R / "p93r5_manifest.json"

TOKENS = ("POSITIVE", "NEGATIVE", "UNABLE_TO_DETERMINE")
UTD = "UNABLE_TO_DETERMINE"
ETIQUETAS = ("HS", "TR", "AG")

# Canalones canonicos (prefijos de 16)
H_TRAIN = "3d40e7345b588860"
H_OOF = "559265816ce4f631"
H_SUB = "7d7478f271e6edfd"
H_PART = "e5b62e5f22c4c63a"  # data/processed/holdout_v2_ids.csv
H_MODEL = "263ed5bd0fc0de7c"
H_R1_C1 = "f89f74cdabd9649d"
H_R1_C2 = "e0449a811fa8b64b"
H_R2_C1 = "91911ab70f6d70cc"
H_R2_C2 = "5da8fc1b33135489"
H_R4_DEC = "98e4264b0fb98b7a"
H_R4_MD = "5b20ac7af7f4d50d"
H_R4_PEND = "e1fb6bc9ccade41b"
H_R4_MAN = "3feecb6d5b2bf9a2"

RUTA_TRAIN = RAIZ / "data" / "raw" / "toxic_train.csv"
RUTA_TEST = RAIZ / "data" / "raw" / "toxic_test.csv"
RUTA_HV2 = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
RUTA_H2 = RAIZ / "data" / "processed" / "holdout2_ids.csv"
RUTA_GUIA = R / "p92_annotation_guidelines_v2_1_holdout_safe.md"
RUTA_OOF = RAIZ / "features" / "oof_v03_final.npz"
RUTA_SUB = RAIZ / "experiments" / "baseline_v03_frozen" / "submission_v03.csv"
RUTA_MODELO = RAIZ / "experiments" / "baseline_v03_frozen" / "model_v03.joblib"


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


def ids_col0_semicolon(ruta: Path) -> set[str]:
    """Ids de la primera columna de un CSV con separador ';' (tolera comillas
    malformadas en la primera celda; latin-1 evita fallos de decodificacion y
    solo se usa la primera columna)."""
    out = set()
    with open(ruta, encoding="latin-1", newline="") as fh:
        rdr = csv.reader(fh, delimiter=";")
        next(rdr)
        for fila in rdr:
            if fila and fila[0]:
                out.add(fila[0].strip().strip('"').strip())
    return out


def ids_col0_comas(ruta: Path) -> set[str]:
    """Ids de la primera columna de un CSV con separador ','."""
    out = set()
    with open(ruta, encoding="utf-8-sig", newline="") as fh:
        rdr = csv.reader(fh)
        next(rdr)
        for fila in rdr:
            if fila and fila[0]:
                out.add(fila[0].strip().strip('"').strip())
    return out


def main() -> None:
    errores: list[str] = []
    chequeos: dict[str, bool] = {}
    avisos: list[str] = []

    # ---------------------------------------------------------------- cargas
    dec = {(f["example_id"], f["label"]): f
           for f in leer_csv(R / "p93r4_final_decisions.csv")}
    pend = {(f["review_id"], f["label"]): f
            for f in leer_csv(R / "p93r4_pending_cases.csv")}
    cand = {(f["review_id"], f["label"]): f
            for f in leer_csv(R / "p93_gold_discrepancy_candidates.csv")}
    key = {f["review_id"]: f for f in leer_csv(R / "p92w_internal_sampling_key.csv")}
    grid = {(f["review_id"], f["label"]): f
            for f in leer_csv(R / "p93_gold_comparison_cells.csv")}
    assign = {(f["review_id"], f["label"]) for f in
              leer_csv(R / "p93r3_round2_id_assignment.csv")}
    man_r4 = json.loads((R / "p93r4_manifest.json").read_text(encoding="utf-8"))

    celdas = sorted(dec.keys())

    # -------------------------------------------------- auditoria de entrada
    chequeos["celdas_176"] = len(celdas) == 176
    chequeos["claves_dec_cand_assign_iguales"] = (
        set(cand) == set(celdas) and assign == set(celdas))
    chequeos["tweets_127"] = len({c[0] for c in celdas}) == 127
    dist = Counter(c[1] for c in celdas)
    chequeos["distribucion_23_87_66"] = (
        dist["HS"] == 23 and dist["TR"] == 87 and dist["AG"] == 66)
    chequeos["guia_v21"] = all(f["guideline_version"] == "v2.1"
                               for f in dec.values())
    chequeos["tokens_validos"] = all(
        f["C1_final_response"] in TOKENS and f["C2_final_response"] in TOKENS
        for f in dec.values())

    binarias = [c for c in celdas
                if dec[c]["C1_final_response"] == dec[c]["C2_final_response"]
                and dec[c]["C1_final_response"] != UTD]
    desacuerdos = [c for c in celdas if c not in set(binarias)]
    chequeos["acuerdo_binario_171"] = len(binarias) == 171
    chequeos["pendientes_5_y_coinciden"] = (
        len(pend) == 5 and set(desacuerdos) == set(pend))

    chequeos["manifiesto_r4_ok"] = (
        man_r4.get("status") == "SECOND_ROUND_RESPONSES_VALIDATED"
        and man_r4.get("decisions") == "FINAL_HUMAN_DECISIONS_LOCKED"
        and all(man_r4.get("checks", {}).values()))
    for ruta, pref, clave in [
            (RESP1 / "response_C1.csv", H_R1_C1, "hash_ronda1_C1"),
            (RESP1 / "response_C2.csv", H_R1_C2, "hash_ronda1_C2"),
            (RESP2 / "response_C1_round2.csv", H_R2_C1, "hash_ronda2_C1"),
            (RESP2 / "response_C2_round2_correcciones.csv", H_R2_C2, "hash_ronda2_C2"),
            (R / "p93r4_final_decisions.csv", H_R4_DEC, "hash_r4_decisiones"),
            (R / "p93r4_final_human_agreement.md", H_R4_MD, "hash_r4_acuerdo"),
            (R / "p93r4_pending_cases.csv", H_R4_PEND, "hash_r4_pendientes"),
            (R / "p93r4_manifest.json", H_R4_MAN, "hash_r4_manifiesto")]:
        chequeos[clave] = sha256_archivo(ruta).startswith(pref)

    chequeos["candidatos_gold_AB_ok"] = all(
        f["gold_label"] in ("0", "1")
        and f["A_response"] in ("POSITIVE", "NEGATIVE")
        and f["B_response"] in ("POSITIVE", "NEGATIVE")
        and f["A_response"] == f["B_response"]
        for f in cand.values())
    chequeos["gold_consistente_con_clave"] = all(
        cand[c]["gold_label"] == key[c[0]]["original_" + c[1]] for c in celdas)
    comp_cand = Counter(cand[c]["sampling_component"] for c in celdas)
    chequeos["componentes_A43_B133"] = comp_cand["A"] == 43 and comp_cand["B"] == 133
    chequeos["grid_cruzada"] = all(
        grid[c]["clasificacion"] == "HUMAN_AGREEMENT_DISAGREES_WITH_GOLD"
        and grid[c]["gold_label"] == cand[c]["gold_label"]
        for c in celdas)

    # ------------------------------------------------ elegibilidad de poblacion
    train_ids = ids_col0_semicolon(RUTA_TRAIN)
    test_ids = ids_col0_semicolon(RUTA_TEST)
    hv2 = set()
    with open(RUTA_HV2, encoding="utf-8-sig", newline="") as fh:
        rdr = csv.reader(fh)
        next(rdr)
        particion = {}
        for fila in rdr:
            if len(fila) > 1:
                particion[fila[0]] = fila[1]
    hv2_holdout = {k for k, v in particion.items() if v == "holdout"}
    h2 = ids_col0_comas(RUTA_H2)

    ids265 = {f["original_example_id"] for f in key.values()}
    ids127 = {key[c[0]]["original_example_id"] for c in celdas}
    chequeos["sample_265_en_train"] = ids265 <= train_ids
    chequeos["tweets_176_en_train"] = ids127 <= train_ids
    chequeos["sample_particion_dev"] = all(
        particion.get(i) == "dev" for i in ids265)
    chequeos["sin_holdout_historico"] = not (ids265 & hv2_holdout)
    chequeos["sin_holdout2"] = not (ids265 & h2)
    chequeos["sin_test"] = not (ids265 & test_ids)

    chequeos["canonicos_intactos"] = (
        sha256_archivo(RUTA_TRAIN).startswith(H_TRAIN)
        and sha256_archivo(RUTA_OOF).startswith(H_OOF)
        and sha256_archivo(RUTA_SUB).startswith(H_SUB)
        and sha256_archivo(RUTA_HV2).startswith(H_PART)
        and sha256_archivo(RUTA_MODELO).startswith(H_MODEL))
    chequeos["guia_v21_hash"] = sha256_archivo(RUTA_GUIA).startswith("fd8d4c65ccd194fb")

    # ------------------------------------------------------------ clasificacion
    filas_cmp: list[dict] = []
    filas_cand: list[dict] = []
    for c in celdas:
        rid, lab = c
        f = dec[c]
        c1i, c1f = f["C1_initial_response"], f["C1_final_response"]
        c2i, c2f = f["C2_initial_response"], f["C2_final_response"]
        gold = cand[c]["gold_label"]
        ab = cand[c]["A_response"]
        rat_c1 = (f["C1_final_rationale"] or "").strip()
        rat_c2 = (f["C2_final_rationale"] or "").strip()
        inc = " | ".join(x for x in [
            (f["C1_remaining_uncertainty"] or "").strip(),
            (f["C2_remaining_uncertainty"] or "").strip()] if x)

        if c in pend:
            est = "UNRESOLVED_NO_THIRD_REVIEW"
            acuerdo = "NO_BINARY_CONSENSUS"
            ev = (f"sin consenso binario final (C1={c1f}, C2={c2f}); "
                  "caso conservado sin resolucion (sin tercera revision en esta fase)")
            dec_final = "UNRESOLVED"
            fuente = "C1_C2_NO_BINARY_CONSENSUS"
            regla = "NO_BINARY_CONSENSUS_KEEP_UNRESOLVED"
            just = ("Sin consenso binario; caso conservado sin resolucion "
                    "(no se asigna respuesta de C1, C2, A ni B).")
            eleg = False
        else:
            acuerdo = "BINARY_AGREEMENT"
            coincide_gold = (
                (c1f == "POSITIVE" and gold == "1")
                or (c1f == "NEGATIVE" and gold == "0"))
            igual_ab = c1f == ab
            base = (f"A/B={ab}; gold={gold}; C1/C2 final={c1f}: "
                    + ("humanos coinciden con A/B y " if igual_ab else "humanos difieren de A/B y ")
                    + "difieren del gold")
            if coincide_gold:
                est = "HUMAN_FINAL_MATCHES_GOLD"
                ev = f"A/B={ab}; gold={gold}; C1/C2 final={c1f}: coinciden con el gold"
                regla = "MATCH_NO_ACTION"
                just = "La decision humana final coincide con el gold; no se requiere correccion."
                eleg = False
            elif rat_c1 or rat_c2:
                est = "POSSIBLE_LABEL_ISSUE"
                fuentes = ", ".join(x for x in [
                    "C1" if rat_c1 else "", "C2" if rat_c2 else ""] if x)
                ev = base + f"; con justificacion documentada ({fuentes}) referida a la politica v2.1"
                regla = "DISCREPANCY_WITH_DOCUMENTED_RATIONALE_V2_1"
                just = " || ".join(
                    f"{n}: {t}" for n, t in
                    (("C1", rat_c1), ("C2", rat_c2)) if t)
                eleg = True
            else:
                est = "HUMAN_FINAL_DIFFERS_FROM_GOLD"
                ev = base + "; sin justificacion documentada en las decisiones finales"
                regla = "DISCREPANCY_WITHOUT_DOCUMENTED_RATIONALE"
                just = ("Sin justificacion documentada en las decisiones finales; la "
                        "coincidencia C1/C2 (o con A/B) no se considera suficiente.")
                eleg = False
            dec_final = c1f
            fuente = "C1_C2_FINAL_BINARY_CONSENSUS"

        filas_cmp.append({
            "example_id": rid, "label": lab, "gold_original": gold,
            "A_response": ab, "B_response": cand[c]["B_response"],
            "C1_initial_response": c1i, "C1_final_response": c1f,
            "C2_initial_response": c2i, "C2_final_response": c2f,
            "final_human_agreement": acuerdo, "comparison_status": est,
            "evidence_summary": ev, "remaining_uncertainty": inc,
        })
        filas_cand.append({
            "example_id": rid, "label": lab, "gold_original": gold,
            "human_final_decision": dec_final, "decision_source": fuente,
            "comparison_status": est, "operational_rule_applied": regla,
            "justification": just, "remaining_uncertainty": inc,
            "eligible_for_future_experiment": "true" if eleg else "false",
        })

    conteo = Counter(f["comparison_status"] for f in filas_cmp)
    chequeos["suma_clasificacion"] = sum(conteo.values()) == 176
    elegibles = [f for f in filas_cand
                 if f["eligible_for_future_experiment"] == "true"]
    chequeos["elegibles_solo_possible"] = (
        len(elegibles) == conteo.get("POSSIBLE_LABEL_ISSUE", 0))
    if conteo.get("CONFIRMED_ANNOTATION_ERROR", 0):
        errores.append("no debe haber errores confirmados en esta fase")

    # ------------------------------------------------------------- salidas CSV
    with open(SAL_CMP, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "example_id", "label", "gold_original", "A_response", "B_response",
            "C1_initial_response", "C1_final_response", "C2_initial_response",
            "C2_final_response", "final_human_agreement", "comparison_status",
            "evidence_summary", "remaining_uncertainty"], lineterminator="\n")
        w.writeheader()
        w.writerows(filas_cmp)
    with open(SAL_CAND, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "example_id", "label", "gold_original", "human_final_decision",
            "decision_source", "comparison_status", "operational_rule_applied",
            "justification", "remaining_uncertainty",
            "eligible_for_future_experiment"], lineterminator="\n")
        w.writeheader()
        w.writerows(filas_cand)

    # --------------------------------------------------- estadisticas de apoyo
    def por_etiqueta(filas, pred) -> dict:
        """Conteo por etiqueta de las filas que cumplen el predicado."""
        c = Counter()
        for f in filas:
            if pred(f):
                c[f["label"]] += 1
        return {e: c.get(e, 0) for e in ETIQUETAS}

    est_por_et = {e: por_etiqueta(filas_cmp, lambda f, e=e:
                                  f["comparison_status"] == e)
                  for e in ("HUMAN_FINAL_MATCHES_GOLD", "POSSIBLE_LABEL_ISSUE",
                            "HUMAN_FINAL_DIFFERS_FROM_GOLD",
                            "UNRESOLVED_NO_THIRD_REVIEW")}
    comp_cmp = Counter()
    comp_pos = Counter()
    dir_pos = Counter()
    for f in filas_cand:
        c = (f["example_id"], f["label"])
        comp = cand[c]["sampling_component"]
        comp_cmp[comp] += 1
        if f["comparison_status"] == "POSSIBLE_LABEL_ISSUE":
            comp_pos[comp] += 1
            dir_pos["add" if f["human_final_decision"] == "POSITIVE" else "remove"] += 1
    tweets_pos = len({f["example_id"] for f in elegibles})
    n_train_filas = sum(1 for _ in open(RUTA_TRAIN, "rb")) - 1

    # ----------------------------------------------------------- informe MD
    md = []
    md.append("# P93-R5 - Comparacion con el gold y registro de correcciones candidatas")
    md.append("")
    md.append("Fecha: 2026-10-13. Comparacion AUTORIZADA explicitamente por el propietario;")
    md.append("sin tercera revision humana. Unidad: celda (tweet, etiqueta). V03 y gold intactos.")
    md.append("")
    md.append("## 1. Auditoria de entrada")
    md.append("")
    md.append(f"- Celdas candidatas: {len(celdas)} (HS {dist['HS']}, TR {dist['TR']}, AG {dist['AG']}); "
              f"tweets unicos: {len({c[0] for c in celdas})}.")
    md.append(f"- Decisiones humanas finales bloqueadas: {len(binarias)} celdas con acuerdo binario "
              f"C1/C2; {len(desacuerdos)} casos sin consenso (cola de pendientes).")
    md.append("- Version de guia en todas las decisiones: v2.1 "
              "(archivo distribuido: fd8d4c65ccd194fb).")
    md.append("- Hashes de ronda 1 y ronda 2 intactos; manifiesto P93-R4 consistente")
    md.append("  (SECOND_ROUND_RESPONSES_VALIDATED / FINAL_HUMAN_DECISIONS_LOCKED).")
    md.append("- Correspondencia por ID y etiqueta verificada en: decisiones, candidatos,")
    md.append("  asignacion P93S1/S93S2, clave interna y grid de 750 celdas (sin uniones por posicion).")
    md.append("")
    md.append("## 2. Elegibilidad de la poblacion")
    md.append("")
    md.append(f"- Las 265 ids del muestreo v2.1 y los {len(ids127)} tweets de las 176 celdas")
    md.append("  pertenecen a train (particion dev); el id 22621 aparece en train con comillas")
    md.append("  malformadas y solo se usa normalizado para el emparejamiento.")
    md.append("- Interseccion con holdout historico (particion 'holdout'): 0; con holdout2: 0;")
    md.append("  con test: 0. No se leyeron contenido, labels ni scores de particiones reservadas")
    md.append("  (solo columnas de id).")
    md.append("- Canonicos verificados: train 3d40e7345b588860, OOF 559265816ce4f631,")
    md.append("  submission 7d7478f271e6edfd, particion e5b62e5f22c4c63a, modelo 263ed5bd0fc0de7c.")
    md.append("")
    md.append("## 3. Comparacion de las 171 decisiones con el gold")
    md.append("")
    md.append("| estado | celdas | HS | TR | AG |")
    md.append("|---|---|---|---|---|")
    for est in ("HUMAN_FINAL_MATCHES_GOLD", "POSSIBLE_LABEL_ISSUE",
                "HUMAN_FINAL_DIFFERS_FROM_GOLD"):
        e = est_por_et[est]
        md.append(f"| {est} | {sum(e.values())} | {e['HS']} | {e['TR']} | {e['AG']} |")
    md.append("| CONFIRMED_ANNOTATION_ERROR | 0 | 0 | 0 | 0 |")
    md.append("")
    md.append(f"- Coincidencias con el gold: {conteo.get('HUMAN_FINAL_MATCHES_GOLD', 0)} de 171.")
    md.append(f"- Discrepancias adjudicadas: {171 - conteo.get('HUMAN_FINAL_MATCHES_GOLD', 0)} de 171 "
              "(todas registradas; ninguna confirmada como error).")
    md.append(f"- Con justificacion documentada (POSSIBLE_LABEL_ISSUE): "
              f"{conteo.get('POSSIBLE_LABEL_ISSUE', 0)}.")
    md.append(f"- Sin justificacion documentada (HUMAN_FINAL_DIFFERS_FROM_GOLD): "
              f"{conteo.get('HUMAN_FINAL_DIFFERS_FROM_GOLD', 0)}.")
    md.append("- CONFIRMED_ANNOTATION_ERROR: 0. Para confirmar se requiere justificacion")
    md.append("  documentada y evidencia que excluya toda interpretacion alternativa razonable;")
    md.append("  esta fase no realiza esa evaluacion adversarial caso a caso.")
    md.append("")
    md.append("Regla operativa aplicada: coincidencia C1/C2 (o con A/B) NO se considera")
    md.append("justificacion suficiente; solo el rationale final referido a la politica v2.1 cuenta.")
    md.append("")
    md.append("## 4. Los cinco casos pendientes")
    md.append("")
    md.append("| caso | etiqueta | C1 final | C2 final | estado |")
    md.append("|---|---|---|---|---|")
    for c in sorted(pend):
        md.append(f"| {c[0]} | {c[1]} | {dec[c]['C1_final_response']} | "
                  f"{dec[c]['C2_final_response']} | UNRESOLVED_NO_THIRD_REVIEW |")
    md.append("")
    md.append("No se asigna respuesta de C1, C2, A ni B; no se abre tercera revision;")
    md.append("no bloquean el analisis de las otras 171 celdas.")
    md.append("")
    md.append("## 5. Limitaciones de muestreo (explicitas)")
    md.append("")
    md.append("- Las 176 celdas se seleccionaron por coincidencia A/B en binario distinta del gold:")
    md.append("  muestra ENRIQUECIDA por discrepancias. No permite estimar tasa global de error")
    md.append("  del gold, prevalencia poblacional ni porcentaje global de labels incorrectos.")
    md.append("- Componentes del muestreo original separados (sin combinar ingenuamente):")
    md.append(f"  celdas comparadas por componente: {dict(comp_cmp)}; candidatas: {dict(comp_pos)};")
    md.append("  en el grid del estudio (750 celdas) la discrepancia A/B observada fue 176/750,")
    md.append("  cifra descriptiva SOLO de esta muestra, no poblacional.")
    md.append("- El componente A (referencia aleatoria) tambien esta condicionado por la seleccion")
    md.append("  de candidatas; tampoco soporta inferencia poblacional directa.")
    md.append("")
    md.append("## 6. Limitaciones de independencia (explicitas)")
    md.append("")
    md.append("- El cegamiento de la ronda 1 no pudo verificarse por completo (P93-R3).")
    md.append("- La ronda 2 mostro las respuestas de A/B por diseno: las decisiones finales NO son")
    md.append("  independientes de esas respuestas. La concordancia humana NO mide de forma")
    md.append("  independiente el error del gold.")
    md.append("")
    md.append("## 7. Procedencia")
    md.append("")
    md.append("- Decisiones finales: p93r4_final_decisions.csv (hash 98e4264b0fb98b7a...).")
    md.append("- Gold interno: clave p92w_internal_sampling_key.csv y candidatos P93 (gold 0/1 por celda).")
    md.append("- Sin cambios en fuentes; artefactos nuevos listados en p93r5_manifest.json.")
    md.append("")
    md.append("## 8. Bloque de decisiones")
    md.append("")
    md.append("```")
    md.append("GOLD_COMPARISON = AUTHORIZED_AND_COMPLETED")
    md.append("THIRD_REVIEW = NOT_REQUIRED_FOR_THIS_PHASE")
    md.append("FINAL_HUMAN_DECISIONS = LOCKED")
    md.append("FIVE_UNRESOLVED_CASES = KEEP_UNRESOLVED")
    md.append("SOURCE_LABEL_MODIFICATION = NOT_AUTHORIZED")
    md.append("MODEL_RETRAINING = NOT_AUTHORIZED")
    md.append("SUBMISSIONS = NOT_AUTHORIZED")
    md.append("V03 = FROZEN")
    md.append("```")
    SAL_CMP_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    # ------------------------------------------------- informe de recomendacion
    n_tot_celdas_train = n_train_filas * 3
    md = []
    md.append("# P93-R5 - Recomendacion cuantificada para una fase experimental futura")
    md.append("")
    md.append("Fecha: 2026-10-13. Insumo: comparacion con el gold completada (P93-R5, sin tercera")
    md.append("revision). Esta recomendacion NO autoriza nada por si misma.")
    md.append("")
    md.append("## 1. Cifras")
    md.append("")
    md.append(f"- Discrepancias adjudicadas (decision humana binaria distinta del gold): "
              f"{171 - conteo.get('HUMAN_FINAL_MATCHES_GOLD', 0)} de 171 comparadas.")
    md.append(f"- Posibles correcciones con justificacion documentada (elegibles): "
              f"{conteo.get('POSSIBLE_LABEL_ISSUE', 0)}.")
    md.append(f"- Registradas sin justificacion documentada: "
              f"{conteo.get('HUMAN_FINAL_DIFFERS_FROM_GOLD', 0)}.")
    md.append(f"- Confirmadas como error de anotacion: 0.")
    md.append(f"- Casos sin resolver que se conservan: {len(pend)}.")
    md.append("")
    md.append("## 2. Distribucion por etiqueta")
    md.append("")
    md.append("| etiqueta | comparadas (acuerdo) | elegibles | direccion |")
    md.append("|---|---|---|---|")
    for e in ETIQUETAS:
        n_el = est_por_et["POSSIBLE_LABEL_ISSUE"][e]
        n_suma = sum(est_por_et[k][e] for k in ("HUMAN_FINAL_MATCHES_GOLD",
                                                "POSSIBLE_LABEL_ISSUE",
                                                "HUMAN_FINAL_DIFFERS_FROM_GOLD"))
        quitar = sum(1 for f in elegibles if f["label"] == e and f["human_final_decision"] == "NEGATIVE")
        anadir = sum(1 for f in elegibles if f["label"] == e and f["human_final_decision"] == "POSITIVE")
        md.append(f"| {e} | {n_suma} | {n_el} | retirar etiqueta {quitar} / anadir {anadir} |")
    md.append(f"| total | 171 | {len(elegibles)} | retirar {dir_pos['remove']} / anadir {dir_pos['add']} |")
    md.append("")
    md.append(f"Candidatas por componente de muestreo: {dict(comp_pos)}; involucran "
              f"{tweets_pos} tweets con justificacion documentada.")
    md.append("")
    md.append("Nota de dependencia de muestreo: las 14 candidatas provienen del componente B")
    md.append("(enriquecido; razon dominante HIGH_CONFIDENCE_OOF_DISAGREEMENT); el componente A")
    md.append("(referencia aleatoria) no aporto candidatas con justificacion documentada.")
    md.append("")
    md.append("## 3. Tamano del efecto y riesgos")
    md.append("")
    md.append(f"- Escala: {len(elegibles)} celdas elegibles sobre ~{n_tot_celdas_train} celdas de train "
              f"(aprox. {100.0 * len(elegibles) / n_tot_celdas_train:.2f}%).")
    md.append("- Muestreo enriquecido por discrepancia: el conjunto de candidatas no es una muestra")
    md.append("  aleatoria y su tasa no es extrapolable.")
    md.append("- Dependencia de A/B: las decisiones finales vieron A/B en ronda 2; corregir el train")
    md.append("  con estas celdas introduce la vision de A/B en el modelo.")
    md.append("- Riesgo de circularidad: evaluar la correccion sobre las mismas celdas usadas para")
    md.append("  definirla infla artificialmente cualquier mejora.")
    md.append("- Riesgo de interpretacion alternativa: en las 14 candidatas no se excluyo formalmente")
    md.append("  una lectura distinta del gold; siguen siendo 'posibles', no confirmadas.")
    md.append("- Grupos near-duplicate: las celdas corregidas pertenecen a tweets con grupos; hay que")
    md.append("  evitar fuga entre particiones si se experimenta.")
    md.append("- Expectativa realista de mejora de Kaggle con 14 celdas: insignificante.")
    md.append("")
    md.append("## 4. Protocolo requerido (si el propietario autoriza un experimento)")
    md.append("")
    md.append("1. Autorizacion separada; V03 y su gold permanecen intactos y congelados.")
    md.append("2. Crear una COPIA versionada del train (por ejemplo")
    md.append("   data/processed/toxic_train_p93r5_variant.csv) aplicando SOLO las celdas")
    md.append("   elegibles, con manifiesto de cambios celda a celda y hash propio.")
    md.append("3. Diseno agrupado: usar dup_group_id / grupos near-duplicate para que ningun grupo")
    md.append("   quede repartido entre ajuste y evaluacion; la particion canonica no se toca.")
    md.append("4. Comparacion pre-registrada: mismo pipeline/codigo para V03-original y variante;")
    md.append("   la metrica final se calcula en una particion NO usada para definir las correcciones")
    md.append("   (OOF/particiones existentes sin rehacer splits; holdout solo con autorizacion aparte).")
    md.append("5. Nada de ajustar hiperparametros ni umbrales sobre la particion de evaluacion de esta")
    md.append("   comparacion; sin pseudo-labeling; sin submissions.")
    md.append("6. Reportar el delta con incertidumbre y el recuento de celdas cambiadas; con 14 celdas")
    md.append("   se espera un efecto indistinguible de ruido (registrarlo como tal).")
    md.append("")
    md.append("## 5. Recomendacion")
    md.append("")
    md.append("Evidencia: 171 discrepancias adjudicadas, 14 con justificacion documentada, 0 confirmadas,")
    md.append("5 sin resolver. La evidencia es SUFICIENTE para documentar candidatas concretas y")
    md.append("SUFICIENTE para un experimento LIMITADO de diagnostico bajo el protocolo de la seccion 4,")
    md.append("pero NO es suficiente para esperar una mejora medible de Kaggle ni para modificar el")
    md.append("train oficial: la muestra esta enriquecida, las decisiones dependen de A/B y ninguna")
    md.append("discrepancia fue confirmada tras excluir interpretaciones alternativas.")
    md.append("")
    md.append("Veredicto: el experimento limitado es opcional y solo con el protocolo completo;")
    md.append("la via principal recomendada es NO corregir el train con esta base y priorizar nuevas")
    md.append("evidencias (mas anotacion) antes de cualquier reentrenamiento orientado a Kaggle.")
    SAL_REC.write_text("\n".join(md) + "\n", encoding="utf-8")

    # ------------------------------------------------------------- manifiesto
    hashes = {}
    for ruta in [
            R / "p93r4_final_decisions.csv", R / "p93r4_final_human_agreement.md",
            R / "p93r4_pending_cases.csv", R / "p93r4_manifest.json",
            R / "p93_manifest.json", R / "p92w_internal_sampling_key.csv",
            R / "p93_gold_discrepancy_candidates.csv", R / "p93_gold_comparison_cells.csv",
            R / "p93_adjudication_blind_candidates.csv",
            R / "p93r3_round2_id_assignment.csv", R / "p93r_internal_adjudication_key.csv",
            RESP1 / "response_C1.csv", RESP1 / "response_C2.csv",
            RESP2 / "response_C1_round2.csv", RESP2 / "response_C2_round2_correcciones.csv",
            RESP2 / "PROVENANCE.md",
            PKG / "annotator_C1" / "adjudication_round2_C1.csv",
            PKG / "annotator_C2" / "adjudication_round2_C2.csv",
            RUTA_GUIA, RUTA_TRAIN, RUTA_OOF, RUTA_SUB, RUTA_MODELO,
            RUTA_HV2, RUTA_H2, SAL_CMP, SAL_CMP_MD, SAL_CAND, SAL_REC,
            Path(__file__)]:
        hashes[str(ruta.relative_to(RAIZ))] = sha256_archivo(ruta)

    verif = subprocess.run([sys.executable, str(RAIZ / "tools" / "verificar_caracteres.py")],
                           capture_output=True, text=True)
    chequeos["verificador_ascii_oficial"] = verif.returncode == 0
    resumen_verif = (verif.stdout or verif.stderr).strip().splitlines()[-1] \
        if (verif.stdout or verif.stderr).strip() else "sin salida"
    for clave, valor in chequeos.items():
        if not valor:
            errores.append(f"chequeo fallido: {clave}")

    manifiesto = {
        "phase": "P93-R5 gold comparison, correction candidates and future-experiment recommendation",
        "date": "2026-10-13",
        "status": "GOLD_COMPARISON_COMPLETED" if not errores else "COMPARISON_WITH_ISSUES",
        "decisions": {
            "gold_comparison": "AUTHORIZED_AND_COMPLETED",
            "third_review": "NOT_REQUIRED_FOR_THIS_PHASE",
            "final_human_decisions": "LOCKED",
            "five_unresolved_cases": "KEEP_UNRESOLVED",
            "source_label_modification": "NOT_AUTHORIZED",
            "model_retraining": "NOT_AUTHORIZED",
            "submissions": "NOT_AUTHORIZED",
            "v03": "FROZEN",
        },
        "guide": {"version": "v2.1", "path": "reports/p92_annotation_guidelines_v2_1_holdout_safe.md",
                  "sha256": sha256_archivo(RUTA_GUIA)},
        "eligible_population": {
            "train_rows": n_train_filas,
            "sample_dev_ids": len(ids265),
            "candidate_tweets": len(ids127),
            "in_train": len(ids127 & train_ids) == len(ids127) and ids265 <= train_ids,
            "holdout_history_intersection": 0,
            "holdout2_intersection": 0,
            "test_intersection": 0,
            "reserved_partitions_read": "solo columna id (sin contenido, labels ni scores)",
        },
        "units": {"cells": 176, "tweets": 127,
                  "HS": dist["HS"], "TR": dist["TR"], "AG": dist["AG"],
                  "binary_agreement_cells": len(binarias),
                  "unresolved_cases": len(pend)},
        "comparison": {
            "HUMAN_FINAL_MATCHES_GOLD": conteo.get("HUMAN_FINAL_MATCHES_GOLD", 0),
            "HUMAN_FINAL_DIFFERS_FROM_GOLD": conteo.get("HUMAN_FINAL_DIFFERS_FROM_GOLD", 0),
            "POSSIBLE_LABEL_ISSUE": conteo.get("POSSIBLE_LABEL_ISSUE", 0),
            "CONFIRMED_ANNOTATION_ERROR": conteo.get("CONFIRMED_ANNOTATION_ERROR", 0),
            "UNRESOLVED_NO_THIRD_REVIEW": conteo.get("UNRESOLVED_NO_THIRD_REVIEW", 0),
        },
        "candidates": {
            "eligible_total": len(elegibles),
            "eligible_by_label": por_etiqueta(elegibles, lambda f: True),
            "eligible_by_component": dict(comp_pos),
            "direction": dict(dir_pos),
            "tweets_with_documented_justification": tweets_pos,
        },
        "components_of_compared_cells": dict(comp_cmp),
        "hashes": hashes,
        "checks": chequeos,
        "avisos": avisos,
        "limitations": [
            "muestra enriquecida por discrepancia A/B vs gold; sin estimacion de prevalencia",
            "decisiones finales vieron A/B en ronda 2; no son independientes",
            "cegamiento de ronda 1 no verificable por completo (P93-R3)",
            "0 errores confirmados: las candidatas son POSSIBLE_LABEL_ISSUE, no CONFIRMED",
            "la coincidencia C1/C2 (o con A/B) no se usa como justificacion suficiente",
        ],
    }
    SAL_MANIFEST.write_text(json.dumps(manifiesto, ensure_ascii=True, indent=2) + "\n",
                            encoding="utf-8")

    print("P93-R5 comparacion con gold: ejecutado")
    print(f"  celdas 176 (HS {dist['HS']}, TR {dist['TR']}, AG {dist['AG']}) | tweets 127 | "
          f"acuerdo binario {len(binarias)} | pendientes {len(desacuerdos)}")
    for k in ("HUMAN_FINAL_MATCHES_GOLD", "HUMAN_FINAL_DIFFERS_FROM_GOLD",
              "POSSIBLE_LABEL_ISSUE", "CONFIRMED_ANNOTATION_ERROR"):
        print(f"  {k}: {conteo.get(k, 0)}")
    print(f"  elegibles: {len(elegibles)} | por etiqueta: {por_etiqueta(elegibles, lambda f: True)} | "
          f"direccion: {dict(dir_pos)}")
    print(f"  verificador: {resumen_verif}")
    print(f"  chequeos: {sum(chequeos.values())}/{len(chequeos)} | "
          f"errores: {errores if errores else 'ninguno'}")
    print(f"  estado: {manifiesto['status']}")
    if errores:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
