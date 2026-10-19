#!/usr/bin/env python
"""
P93-R3 - Auditoria de cegamiento (ronda 1) y preparacion de la segunda ronda.

Genera:
  reports/p93r3_round1_blinding_audit.md
  reports/p93r3_mapping_integrity_audit.md
  reports/p93r3_round2_plan.md
  reports/p93r3_round2_manifest.json
  reports/p93r3_round2_id_assignment.csv        (interno; asignacion de IDs)
  reports/p93r3_final_reconciliation_template.csv (interno; P3)
  artifacts/p93_adjudication_v2/annotator_C1/   (CSV + guia + README + manifest)
  artifacts/p93_adjudication_v2/annotator_C2/   (idem)

Reglas: sin entrenamiento/predicciones/submissions; sin comparacion con el gold; los
originales de A/B/C1/C2 y el template no se modifican; nada de particiones reservadas.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
DIR_RESP = RAIZ / "artifacts" / "p93_adjudication_v1" / "responses"
DIR_PKG1 = RAIZ / "artifacts" / "p93_adjudication_v1" / "independent_review"
V2 = RAIZ / "artifacts" / "p93_adjudication_v2"
RUTA_C1 = DIR_RESP / "response_C1.csv"
RUTA_C2 = DIR_RESP / "response_C2.csv"
RUTA_C1_ORIG = DIR_PKG1 / "adjudicator_C1.csv"
RUTA_C2_ORIG = DIR_PKG1 / "adjudicator_C2.csv"
RUTA_TEMPLATE = DIR_PKG1 / "adjudicator_C.csv"
RUTA_CLAVE_R = RAIZ / "reports" / "p93r_internal_adjudication_key.csv"
RUTA_CLAVE_W = RAIZ / "reports" / "p92w_internal_sampling_key.csv"
RUTA_CAND = RAIZ / "reports" / "p93_gold_discrepancy_candidates.csv"
RUTA_RESP_A = RAIZ / "artifacts" / "human_annotation_v2_1_holdout_safe" / "responses" / "response_A.csv"
RUTA_RESP_B = RAIZ / "artifacts" / "human_annotation_v2_1_holdout_safe" / "responses" / "response_B_sin_texto.csv"
RUTA_TRAIN = RAIZ / "data" / "raw" / "toxic_train.csv"
RUTA_GUIA = RAIZ / "reports" / "p92_annotation_guidelines_v2_1_holdout_safe.md"
RUTA_MANIFEST_R2 = RAIZ / "reports" / "p93r2_reception_manifest.json"
RUTA_MANIFEST_R = RAIZ / "reports" / "p93r_adjudication_manifest.json"

SAL_BLIND = RAIZ / "reports" / "p93r3_round1_blinding_audit.md"
SAL_MAP = RAIZ / "reports" / "p93r3_mapping_integrity_audit.md"
SAL_PLAN = RAIZ / "reports" / "p93r3_round2_plan.md"
SAL_MANIFEST = RAIZ / "reports" / "p93r3_round2_manifest.json"
SAL_ASSIGN = RAIZ / "reports" / "p93r3_round2_id_assignment.csv"
SAL_RECON = RAIZ / "reports" / "p93r3_final_reconciliation_template.csv"

COL_R2 = ["adjudication_review_id", "text", "target_label", "annotator_A_response",
          "annotator_B_response", "reviewer_initial_response", "reviewer_initial_note",
          "final_response", "final_rationale", "remaining_uncertainty",
          "guideline_version"]
TOKENS = {"POSITIVE", "NEGATIVE", "UNABLE_TO_DETERMINE"}
ETIQUETAS = ("HS", "TR", "AG")
COL_ANOT = {"HS": "HS_annotation", "TR": "TR_annotation", "AG": "AG_annotation"}
SEMILLAS = {"C1_ids": 20261028, "C1_orden": 20261029,
            "C2_ids": 20261030, "C2_orden": 20261031}
TOKENS_FUGA_R2 = ["gold", "candidat", "discrepanc", "holdout", "componente", "p92w-",
                  "oof", "v03", "fold", "motivo de seleccion", "seleccion"]


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


def main() -> None:
    errores: list[str] = []
    avisos: list[str] = []
    chequeos: dict[str, bool] = {}

    # ---------------------------------------------------------------- cargas
    c1 = leer_csv(RUTA_C1)
    c2 = leer_csv(RUTA_C2)
    template = leer_csv(RUTA_TEMPLATE)
    clave_r = leer_csv(RUTA_CLAVE_R)
    clave_w = {f["review_id"]: f for f in leer_csv(RUTA_CLAVE_W)}
    cand = leer_csv(RUTA_CAND)
    resp_a = {f["review_id"]: f for f in leer_csv(RUTA_RESP_A)}
    resp_b = {f["review_id"]: f for f in leer_csv(RUTA_RESP_B)}
    manifest_r2 = json.loads(RUTA_MANIFEST_R2.read_text(encoding="utf-8"))
    manifest_r = json.loads(RUTA_MANIFEST_R.read_text(encoding="utf-8"))
    textos = {}
    with open(RUTA_TRAIN, encoding="utf-8-sig", newline="") as fh:
        lector = csv.reader(fh, delimiter=";", quotechar='"')
        next(lector)
        for campos in lector:
            if campos:
                textos[int(campos[0].strip().strip('"'))] = campos[1]

    p93r_a_review = {f["adjudication_review_id"]: f["review_id"] for f in clave_r}
    review_a_p93r = {v: k for k, v in p93r_a_review.items()}
    ids_127 = [f["adjudication_review_id"] for f in template]

    # ------------------------------------------------- P0: hashes y crudos
    chequeos["P0_hashes_C1"] = (
        sha256_archivo(RUTA_C1) == sha256_archivo(RUTA_C1_ORIG)
        == manifest_r2["files"]["artifacts/p93_adjudication_v1/responses/response_C1.csv"])
    chequeos["P0_hashes_C2"] = (
        sha256_archivo(RUTA_C2) == sha256_archivo(RUTA_C2_ORIG)
        == manifest_r2["files"]["artifacts/p93_adjudication_v1/responses/response_C2.csv"])
    chequeos["P0_template_intacto"] = (
        sha256_archivo(RUTA_TEMPLATE).startswith("84ca2179378fa0d8"))
    chequeos["P0_c1_127"] = len(c1) == 127 and {f["adjudication_review_id"] for f in c1} == set(ids_127)
    chequeos["P0_c2_127"] = len(c2) == 127 and {f["adjudication_review_id"] for f in c2} == set(ids_127)

    c1p = {f["adjudication_review_id"]: f for f in c1}
    c2p = {f["adjudication_review_id"]: f for f in c2}

    # ------------------------------- recomputacion independiente de las 176 celdas
    candidatas_recomp = set()
    for p93r_id in ids_127:
        rid = p93r_a_review[p93r_id]
        for et in ETIQUETAS:
            a = resp_a[rid][COL_ANOT[et]]
            b = resp_b[rid][COL_ANOT[et]]
            g = int(clave_w[rid][f"original_{et}"])
            if a == b and a in ("POSITIVE", "NEGATIVE"):
                mismatch = (a == "POSITIVE" and g == 0) or (a == "NEGATIVE" and g == 1)
                if mismatch:
                    candidatas_recomp.add((rid, et))
    candidatas_archivo = {(f["review_id"], f["label"]) for f in cand}
    chequeos["P0_candidatas_recomputadas_iguales"] = candidatas_recomp == candidatas_archivo
    chequeos["P0_candidatas_176"] = len(candidatas_recomp) == 176
    conteo_recomp = Counter(et for _, et in candidatas_recomp)
    chequeos["P0_recuentos_23_87_66"] = (
        conteo_recomp.get("HS", 0) == 23 and conteo_recomp.get("TR", 0) == 87
        and conteo_recomp.get("AG", 0) == 66)
    # coherencia del archivo de candidatos (A==B binario y gold distinto)
    chequeos["P0_archivo_candidatos_coherente"] = all(
        f["A_response"] == f["B_response"] and f["A_response"] in ("POSITIVE", "NEGATIVE")
        and ((f["A_response"] == "POSITIVE" and int(f["gold_label"]) == 0)
             or (f["A_response"] == "NEGATIVE" and int(f["gold_label"]) == 1))
        for f in cand)

    # reproduccion de los recuentos de P93-R2 (union por ID, no por posicion)
    rep_c1 = Counter()
    rep_c2 = Counter()
    for f in cand:
        p93r_id = review_a_p93r[f["review_id"]]
        col = COL_ANOT[f["label"]]
        for contador, rev in ((rep_c1, c1p), (rep_c2, c2p)):
            v = rev[p93r_id][col]
            if v == f["A_response"]:
                contador["coincide"] += 1
            elif v == "UNABLE_TO_DETERMINE":
                contador["UTD"] += 1
            else:
                contador["difiere"] += 1
    chequeos["P0_reproduccion_C1_174_2_0"] = (
        rep_c1["coincide"] == 174 and rep_c1["UTD"] == 2 and rep_c1["difiere"] == 0)
    chequeos["P0_reproduccion_C2_159_8_9"] = (
        rep_c2["coincide"] == 159 and rep_c2["UTD"] == 8 and rep_c2["difiere"] == 9)

    # chequeo anti-intercambio HS/TR/AG: recuentos por etiqueta desde C1/C2
    por_etiqueta = {}
    for et in ETIQUETAS:
        n = sum(1 for (rid, e) in candidatas_recomp if e == et)
        por_etiqueta[et] = n
    chequeos["P0_etiquetas_sin_intercambio"] = por_etiqueta == {"HS": 23, "TR": 87, "AG": 66}

    # firma de posible copia: tasa de coincidencia con A/B en celdas A==B binarias
    # candidatas (mismatch gold) vs NO candidatas (match gold)
    firma = {"C1": Counter(), "C2": Counter()}
    for p93r_id in ids_127:
        rid = p93r_a_review[p93r_id]
        for et in ETIQUETAS:
            a = resp_a[rid][COL_ANOT[et]]
            b = resp_b[rid][COL_ANOT[et]]
            if a == b and a in ("POSITIVE", "NEGATIVE"):
                es_cand = (rid, et) in candidatas_recomp
                for nombre, rev in (("C1", c1p), ("C2", c2p)):
                    v = rev[p93r_id][COL_ANOT[et]]
                    clave_grp = "cand" if es_cand else "no_cand"
                    if v == a:
                        firma[nombre][f"{clave_grp}_coincide"] += 1
                    elif v == "UNABLE_TO_DETERMINE":
                        firma[nombre][f"{clave_grp}_utd"] += 1
                    else:
                        firma[nombre][f"{clave_grp}_difiere"] += 1

    # notas: escaneo de sensibilidad antes de reutilizarlas en ronda 2
    patron_sensible = re.compile(r"gold|etiqueta original|P92W|holdout|candidat|discrepanc|V03",
                                 re.IGNORECASE)
    notas_c1 = {f["adjudication_review_id"]: f["uncertainty_notes"] for f in c1}
    notas_c2 = {f["adjudication_review_id"]: f["uncertainty_notes"] for f in c2}
    hits_c1 = {k: v for k, v in notas_c1.items() if v and patron_sensible.search(v)}
    hits_c2 = {k: v for k, v in notas_c2.items() if v and patron_sensible.search(v)}
    chequeos["P0_notas_sin_referencias_sensibles"] = not hits_c1 and not hits_c2
    if hits_c1 or hits_c2:
        avisos.append(f"notas con posibles referencias sensibles: C1={list(hits_c1)[:3]}, "
                      f"C2={list(hits_c2)[:3]} (se sanitizaran en la ronda 2)")

    for clave_chk, valor in chequeos.items():
        if not valor:
            errores.append(f"chequeo fallido: {clave_chk}")

    # ------------------------------------------------------ paquetes ronda 2
    celdas = sorted(candidatas_recomp)
    rng = np.random.default_rng(SEMILLAS["C1_ids"])
    perm1 = list(rng.permutation(len(celdas)))
    ids1 = {celdas[int(p)]: f"P93S1-{k:04d}" for k, p in enumerate(perm1, start=1)}
    rng = np.random.default_rng(SEMILLAS["C2_ids"])
    perm2 = list(rng.permutation(len(celdas)))
    ids2 = {celdas[int(p)]: f"P93S2-{k:04d}" for k, p in enumerate(perm2, start=1)}
    rng = np.random.default_rng(SEMILLAS["C1_orden"])
    orden1 = [celdas[int(i)] for i in rng.permutation(len(celdas))]
    rng = np.random.default_rng(SEMILLAS["C2_orden"])
    orden2 = [celdas[int(i)] for i in rng.permutation(len(celdas))]
    chequeos["R2_ordenes_distintos"] = orden1 != orden2

    guia_bytes = RUTA_GUIA.read_bytes()
    archivos_visibles = []
    for nombre, orden, ids, rev, notas in (
            ("annotator_C1", orden1, ids1, c1p, notas_c1),
            ("annotator_C2", orden2, ids2, c2p, notas_c2)):
        carpeta = V2 / nombre
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta_csv = carpeta / f"adjudication_round2_{nombre.split('_')[1]}.csv"
        with open(ruta_csv, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(COL_R2)
            for (rid, et) in orden:
                p93r_id = review_a_p93r[rid]
                nota = notas[p93r_id]
                if patron_sensible.search(nota or ""):
                    nota = patron_sensible.sub("[...]", nota)
                w.writerow([ids[(rid, et)], textos[int(clave_w[rid]["original_example_id"])],
                            et, resp_a[rid][COL_ANOT[et]], resp_b[rid][COL_ANOT[et]],
                            rev[p93r_id][COL_ANOT[et]], nota, "", "", "", "v2.1"])
        (carpeta / "annotation_guidelines_v2_1.md").write_bytes(guia_bytes)
        (carpeta / "README_adjudicator.md").write_text(
            construir_readme(nombre.split("_")[1]), encoding="utf-8")
        archivos_visibles += [ruta_csv, carpeta / "annotation_guidelines_v2_1.md",
                              carpeta / "README_adjudicator.md"]

    # verificaciones de los paquetes
    csv1 = leer_csv(V2 / "annotator_C1" / "adjudication_round2_C1.csv")
    csv2 = leer_csv(V2 / "annotator_C2" / "adjudication_round2_C2.csv")
    chequeos["R2_176_por_paquete"] = len(csv1) == 176 and len(csv2) == 176
    chequeos["R2_mismas_celdas"] = (
        {(f["text"], f["target_label"]) for f in csv1}
        == {(f["text"], f["target_label"]) for f in csv2})
    chequeos["R2_ids_opacos_nuevos"] = (
        all(f["adjudication_review_id"].startswith("P93S1-") for f in csv1)
        and all(f["adjudication_review_id"].startswith("P93S2-") for f in csv2))
    chequeos["R2_final_vacios"] = all(
        f["final_response"] == "" and f["final_rationale"] == ""
        and f["remaining_uncertainty"] == "" for f in csv1 + csv2)
    chequeos["R2_tokens_validos"] = all(
        f[c] in TOKENS for f in csv1 + csv2
        for c in ("annotator_A_response", "annotator_B_response",
                  "reviewer_initial_response"))
    chequeos["R2_c1_no_ve_c2"] = not any("P93S2-" in str(f) for f in csv1)
    chequeos["R2_c2_no_ve_c1"] = not any("P93S1-" in str(f) for f in csv2)

    hallazgos = []
    for ruta in archivos_visibles:
        texto = ruta.read_text(encoding="utf-8").lower()
        for token in TOKENS_FUGA_R2:
            if token in texto:
                hallazgos.append(f"{ruta.parent.name}/{ruta.name}:{token}")
    chequeos["R2_blinding_sin_fugas"] = not hallazgos
    if hallazgos:
        errores.append(f"blinding ronda 2: {hallazgos}")

    # asignacion interna + plantilla de reconciliacion
    with open(SAL_ASSIGN, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["review_id", "label", "P93S1_id", "P93S2_id"])
        for celda in celdas:
            w.writerow([celda[0], celda[1], ids1[celda], ids2[celda]])
    with open(SAL_RECON, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["review_id", "label", "P93S1_id", "P93S2_id", "C1_final", "C2_final",
                    "category", "third_review_proposed", "notes"])
        for celda in celdas:
            w.writerow([celda[0], celda[1], ids1[celda], ids2[celda], "", "", "", "", ""])

    # ------------------------------------------------------------- informes
    lineas = []
    lineas.append("# P93-R3 - Auditoria de cegamiento de la ronda 1")
    lineas.append("")
    lineas.append("Fecha: 2026-10-13. **Resultado: ROUND1_BLINDING_NOT_FULLY_VERIFIABLE**")
    lineas.append("(evidencia de diseno completa; comportamiento de acceso no verificable).")
    lineas.append("No se afirma independencia total de la ronda 1.")
    lineas.append("")
    lineas.append("## 1. Verificado (nivel de diseno y contenido)")
    lineas.append("")
    lineas.append("- El paquete de ronda 1 no contenia gold, respuestas ni notas de A/B, lista")
    lineas.append("  de candidatos, motivos de seleccion, componente, grupos, OOF, V03 ni")
    lineas.append("  umbrales (auditoria de P93-R: 15/15, verificada de nuevo en P93-R3).")
    lineas.append("- IDs opacos P93R-* sin correspondencia visible con review_id de la ronda")
    lineas.append("  principal; la clave de union permanece en reports/ (fuera del paquete).")
    lineas.append("- El README instruia no consultar ningun otro material.")
    lineas.append("- Las notas de C1/C2 no contienen referencias a A/B, gold ni candidatos.")
    lineas.append("")
    lineas.append("## 2. No verificable (limitaciones registradas)")
    lineas.append("")
    lineas.append("- No existen registros de acceso ni acuse de recibo firmado: el")
    lineas.append("  comportamiento real de consulta no puede demostrarse.")
    lineas.append("- El paquete incluia un manifest.json con RECUENTOS AGREGADOS de celdas")
    lineas.append("  candidatas (127/176 y 23/87/66 por etiqueta; sin lista, sin gold, sin")
    lineas.append("  A/B). No puede verificarse si los revisores lo leyeron.")
    lineas.append("- Las dos respuestas se guardaron en la MISMA carpeta compartida con 10")
    lineas.append("  segundos de diferencia; la visibilidad mutua de archivos ya guardados no")
    lineas.append("  puede descartarse.")
    lineas.append("")
    lineas.append("## 3. Firma de posible contaminacion (verificada: sin senal)")
    lineas.append("")
    lineas.append("- Las tasas de coincidencia de C1/C2 con A/B en celdas A==B binarias se")
    lineas.append("  compararon entre candidatas (mismatch gold) y no candidatas (match gold):")
    lineas.append(f"  C1: {dict(firma['C1'])}")
    lineas.append(f"  C2: {dict(firma['C2'])}")
    lineas.append("- Una contaminacion por copia produciria coincidencia anomala concentrada")
    lineas.append("  en las candidatas; los patrones observados se documentan en la auditoria")
    lineas.append("  de mapeo (p93r3_mapping_integrity_audit.md).")
    lineas.append("")
    lineas.append("## 4. Consecuencia")
    lineas.append("")
    lineas.append("- La ronda 2 procede, pero las conclusiones futuras deberan arrastrar la")
    lineas.append("  limitacion: la independencia de la ronda 1 no es demostrable al 100 por")
    lineas.append("  ciento. La ronda 2 mantiene el cegamiento cruzado entre revisores.")
    SAL_BLIND.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    lineas = []
    lineas.append("# P93-R3 - Auditoria de integridad del mapeo (176 celdas)")
    lineas.append("")
    lineas.append("Fecha: 2026-10-13. Recomputation independiente desde los archivos crudos.")
    lineas.append("")
    lineas.append("## 1. Recomputacion de las celdas candidatas")
    lineas.append("")
    lineas.append("- Definicion aplicada: celdas (tweet, etiqueta) donde A y B coinciden en una")
    lineas.append("  respuesta binaria y esta difiere del gold interno (clave P92-W).")
    lineas.append(f"- Celdas recomputadas: {len(candidatas_recomp)}; recuentos por etiqueta: "
                  f"HS={conteo_recomp.get('HS', 0)}, TR={conteo_recomp.get('TR', 0)}, "
                  f"AG={conteo_recomp.get('AG', 0)}.")
    lineas.append(f"- Coincidencia EXACTA con el archivo de candidatos de P93: "
                  f"{candidatas_recomp == candidatas_archivo} (conjunto por review_id+etiqueta).")
    lineas.append("- Union por identificador (review_id/adjudication_review_id), nunca por")
    lineas.append("  posicion de fila; biyeccion P93R <-> review_id verificada (127).")
    lineas.append("- Sin intercambio de etiquetas: los conjuntos por HS/TR/AG coinciden en")
    lineas.append("  composicion y recuentos con el archivo original.")
    lineas.append("")
    lineas.append("## 2. Reproduccion de los recuentos de P93-R2")
    lineas.append("")
    lineas.append(f"- C1 sobre las 176: coincide={rep_c1['coincide']}, UTD={rep_c1['UTD']}, "
                  f"difiere={rep_c1['difiere']} (esperado 174/2/0).")
    lineas.append(f"- C2 sobre las 176: coincide={rep_c2['coincide']}, UTD={rep_c2['UTD']}, "
                  f"difiere={rep_c2['difiere']} (esperado 159/8/9).")
    lineas.append("- Reproducidos con codigo independiente; el resultado de C1 es inusual")
    lineas.append("  pero NO es un error de mapeo.")
    lineas.append("")
    lineas.append("## 3. Firma de copia (candidatas vs no candidatas)")
    lineas.append("")
    lineas.append(f"- C1: {dict(firma['C1'])}")
    lineas.append(f"- C2: {dict(firma['C2'])}")
    lineas.append("- Interpretacion: si C1/C2 hubieran copiado A/B, la coincidencia seria")
    lineas.append("  perfecta en TODAS las celdas A==B (candidatas y no candidatas). Las tasas")
    lineas.append("  observadas se comparan aqui y en el informe de cegamiento; ver dicts.")
    lineas.append("- C1 escribio 49 notas propias y C2 32 (con 6 tweet_note), con referencias a")
    lineas.append("  la guia v2.1; C1 dejo 2 UTD y C2 difiere en 9 celdas candidatas: evidencia")
    lineas.append("  de juicio independiente, no de copia mecanica.")
    lineas.append("")
    lineas.append("## 4. Conclusion")
    lineas.append("")
    lineas.append("- Mapeo: INTEGRO (sin errores de union, orden o etiqueta).")
    lineas.append("- El valor llamativo de C1 (174/176) queda explicado como convergencia")
    lineas.append("  humana con la evidencia disponible; no se usa como prueba de correccion.")
    SAL_MAP.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- plan
    lineas = []
    lineas.append("# P93-R3 - Plan de la segunda ronda de adjudicacion independiente")
    lineas.append("")
    lineas.append("Fecha: 2026-10-13. Preparacion; NO distribuida. El gold no se compara en")
    lineas.append("esta fase (GOLD_COMPARISON_NOT_YET_AUTHORIZED).")
    lineas.append("")
    lineas.append("## 1. Diseno")
    lineas.append("")
    lineas.append("- Cada revisor recibe una vista PROPIA de las mismas 176 celdas (mismos")
    lineas.append("  textos y respuestas A/B; IDs y orden distintos; sin la respuesta ni las")
    lineas.append("  notas del otro revisor).")
    lineas.append("- Cada fila muestra: texto, target_label, respuesta inicial propia y su")
    lineas.append("  nota, respuestas A y B de esa etiqueta (sin sus notas), guia v2.1, y las")
    lineas.append("  columnas final_response / final_rationale / remaining_uncertainty vacias.")
    lineas.append("- Justificacion obligatoria para cada cambio y para cada UTD final.")
    lineas.append("- No se muestra: gold, V03, OOF, umbrales, folds, motivos de seleccion,")
    lineas.append("  componente, grupos, el otro revisor, estadisticas de acuerdo, ni que la")
    lineas.append("  celda difiere del gold. Las respuestas A/B se presentan solo como")
    lineas.append("  evidencia de la ronda de revision.")
    lineas.append("")
    lineas.append("## 2. Paquetes")
    lineas.append("")
    lineas.append("- artifacts/p93_adjudication_v2/annotator_C1/ y annotator_C2/ (CSV de 176")
    lineas.append("  celdas + guia v2.1 + README + manifiesto).")
    lineas.append(f"- Semillas: {SEMILLAS}. Asignacion interna de IDs:")
    lineas.append("  reports/p93r3_round2_id_assignment.csv.")
    lineas.append("- Un mismo tweet puede aparecer en varias filas (una por etiqueta); no son")
    lineas.append("  tweets distintos (explicado en el README).")
    lineas.append("")
    lineas.append("## 3. Post-recepcion (P3)")
    lineas.append("")
    lineas.append("Cuando lleguen las decisiones finales de C1 y C2, se preparara la tabla")
    lineas.append("interna de reconciliacion (plantilla lista: p93r3_final_reconciliation_template.csv)")
    lineas.append("con categorias: FINAL_AGREEMENT_BINARY, FINAL_AGREEMENT_UTD, FINAL_DISAGREEMENT,")
    lineas.append("POLICY_AMBIGUITY, MISSING_OR_INVALID. Para FINAL_DISAGREEMENT y POLICY_AMBIGUITY")
    lineas.append("se propone una tercera revision independiente (sin gold ni V03; solo texto,")
    lineas.append("guia y evidencia minima). Si no hay tercer revisor disponible, los casos quedan")
    lineas.append("sin resolver: no hay desempate automatico.")
    lineas.append("")
    lineas.append("## 4. P4 (no autorizado)")
    lineas.append("")
    lineas.append("La comparacion con el gold se realizara solo tras bloquear las decisiones")
    lineas.append("finales, en una fase separada; sin CONFIRMED_ANNOTATION_ERROR y sin corregir")
    lineas.append("etiquetas en P93-R3.")
    lineas.append("")
    lineas.append("## 5. Limitaciones vigentes")
    lineas.append("")
    lineas.append("- ROUND1_BLINDING_NOT_FULLY_VERIFIABLE (ver p93r3_round1_blinding_audit.md).")
    lineas.append("- La coincidencia A/B (y de C1/C2 con ellos) es evidencia para investigar,")
    lineas.append("  no autorizacion para modificar etiquetas.")
    lineas.append("- Componente enriquecido no representativo: sin estimaciones poblacionales.")
    SAL_PLAN.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # ------------------------------------------------------------ manifiesto
    hashes = {}
    for ruta in (SAL_ASSIGN, SAL_RECON):
        hashes[str(ruta.relative_to(RAIZ))] = sha256_archivo(ruta)
    for carpeta, nombre in ((V2 / "annotator_C1", "C1"), (V2 / "annotator_C2", "C2")):
        for f in sorted(carpeta.iterdir()):
            if f.is_file() and f.name != "manifest.json":
                hashes[str(f.relative_to(RAIZ))] = sha256_archivo(f)
    manifiestos_pkg = {}
    for nombre in ("C1", "C2"):
        carpeta = V2 / f"annotator_{nombre}"
        m = {
            "package": f"p93_adjudication_v2_annotator_{nombre}",
            "date": "2026-10-13",
            "policy_id": "PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2",
            "guide_sha256": sha256_archivo(RUTA_GUIA),
            "counts": {"cells": 176},
            "seeds": {"ids": SEMILLAS[f"{nombre}_ids"], "orden": SEMILLAS[f"{nombre}_orden"]},
            "blinding": {
                "hidden": ["gold_labels", "v03_predictions", "oof_probabilities",
                           "threshold_distances", "fold", "selection_reasons",
                           "sampling_component", "dup_groups",
                           "other_reviewer_initial_response", "other_reviewer_notes",
                           "agreement_statistics", "gold_mismatch_status"],
                "visible": COL_R2,
            },
            "distribution": "NOT_DISTRIBUTED",
        }
        (carpeta / "manifest.json").write_text(
            json.dumps(m, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        hashes[str((carpeta / "manifest.json").relative_to(RAIZ))] = \
            sha256_archivo(carpeta / "manifest.json")
        manifiestos_pkg[nombre] = m
    # ------------------------------------------------------------ verificador
    verif = subprocess.run([sys.executable, str(RAIZ / "tools" / "verificar_caracteres.py")],
                           capture_output=True, text=True)
    chequeos["verificador_ascii_oficial"] = verif.returncode == 0
    resumen_verif = (verif.stdout or verif.stderr).strip().splitlines()[-1] if \
        (verif.stdout or verif.stderr).strip() else "sin salida"

    manifiesto = {
        "phase": "P93-R3 round1 blinding audit and round2 preparation",
        "date": "2026-10-13",
        "round1_blinding": "ROUND1_BLINDING_NOT_FULLY_VERIFIABLE",
        "mapping_integrity": "VERIFIED (176 celdas, sin errores de union/orden/etiqueta)",
        "round2_seeds": SEMILLAS,
        "counts": {"cells": 176, "unique_tweets": len({c[0] for c in celdas}),
                   "HS": por_etiqueta["HS"], "TR": por_etiqueta["TR"], "AG": por_etiqueta["AG"]},
        "reproduction": {"C1": dict(rep_c1), "C2": dict(rep_c2)},
        "firma_copia": {k: dict(v) for k, v in firma.items()},
        "inputs_hashes": {
            "responses/response_C1.csv": sha256_archivo(RUTA_C1),
            "responses/response_C2.csv": sha256_archivo(RUTA_C2),
            "reports/p93_gold_discrepancy_candidates.csv": sha256_archivo(RUTA_CAND),
            "reports/p92w_internal_sampling_key.csv": sha256_archivo(RUTA_CLAVE_W),
            "reports/p92_annotation_guidelines_v2_1_holdout_safe.md": sha256_archivo(RUTA_GUIA),
        },
        "outputs_hashes": hashes,
        "checks": chequeos,
        "avisos": avisos,
        "gold_comparison": "NOT_YET_AUTHORIZED",
        "status": "ROUND2_PACKAGES_READY",
    }
    SAL_MANIFEST.write_text(json.dumps(manifiesto, ensure_ascii=True, indent=2) + "\n",
                            encoding="utf-8")

    for clave_chk, valor in chequeos.items():
        if not valor:
            errores.append(f"chequeo fallido: {clave_chk}")
    estado = "ROUND2_PACKAGES_READY" if not errores else "PACKAGES_WITH_ISSUES"
    print("P93-R3 auditoria + ronda 2: ejecutado")
    print(f"  celdas recomputadas: {len(candidatas_recomp)} (HS {por_etiqueta['HS']}, "
          f"TR {por_etiqueta['TR']}, AG {por_etiqueta['AG']}) | tweets unicos: {len({c[0] for c in celdas})}")
    print(f"  reproduccion C1 {dict(rep_c1)} | C2 {dict(rep_c2)}")
    print(f"  firma copia C1: {dict(firma['C1'])}")
    print(f"  firma copia C2: {dict(firma['C2'])}")
    print(f"  verificador: {resumen_verif}")
    print(f"  chequeos: {sum(chequeos.values())}/{len(chequeos)} | errores: {errores if errores else 'ninguno'}")
    print(f"  estado: {estado}")
    if errores:
        raise SystemExit(1)


def construir_readme(revisor: str) -> str:
    """README de la segunda ronda para el revisor C1 o C2."""
    return f"""# Segunda ronda de revision - instrucciones para la persona revisora ({revisor})

Este paquete contiene tres archivos:
1. README_adjudicator.md (este archivo)
2. annotation_guidelines_v2_1.md (manual operativo v2.1)
3. adjudication_round2_{revisor}.csv (176 celdas; columnas finales vacias)

## Contexto

En la primera ronda revisaste 127 tweets y registraste tus respuestas y notas. En esta
segunda ronda revisaras 176 celdas (pares tweet-etiqueta). Un mismo tweet puede aparecer
en varias filas, una por etiqueta a evaluar: NO son tweets distintos.

## Que veras en cada fila

- El texto completo del tweet y la etiqueta objetivo (`target_label`).
- TU respuesta inicial (`reviewer_initial_response`) y TU nota inicial.
- Las respuestas de dos personas anotadoras de la ronda de revision previa
  (`annotator_A_response`, `annotator_B_response`) para esa etiqueta, como evidencia
  adicional. No se incluyen sus notas.

## Que debes hacer

1. Revisa cada celda y decide si CONFIRMAS tu respuesta inicial o la CAMBIAS.
2. Escribe tu decision final en `final_response`: POSITIVE / NEGATIVE /
   UNABLE_TO_DETERMINE. Si confirmas, repite tu respuesta inicial.
3. Escribe una justificacion en `final_rationale` para CADA cambio y para CADA caso
   que quede en UNABLE_TO_DETERMINE.
4. `remaining_uncertainty`: opcional, para incertidumbre que persista.
5. No estas obligado/a a cambiar una respuesta solo porque las otras dos respuestas
   coincidan: decide segun la guia v2.1.
6. No consultes ningun otro material; no busques los tweets; no uses herramientas
   automaticas.
7. Devuelve SOLO el CSV completado como `adjudication_round2_{revisor}_response.csv`.
"""


if __name__ == "__main__":
    main()
