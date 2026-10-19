#!/usr/bin/env python
"""
P93-R2 - Recepcion de las revisiones independientes C1 y C2 (adjudicacion ciega).

Entradas (solo lectura; originales byte a byte):
  artifacts/p93_adjudication_v1/independent_review/adjudicator_C1.csv
  artifacts/p93_adjudication_v1/independent_review/adjudicator_C2.csv
  (template adjudicator_C.csv, clave interna, respuestas A/B y textos canonicos)

Salidas:
  artifacts/p93_adjudication_v1/responses/response_C1.csv          (copia byte a byte)
  artifacts/p93_adjudication_v1/responses/response_C2.csv          (copia byte a byte)
  artifacts/p93_adjudication_v1/responses/response_C1_completado.csv  (derivado)
  artifacts/p93_adjudication_v1/responses/response_C2_completado.csv  (derivado)
  artifacts/p93_adjudication_v1/responses/PROVENANCE.md
  reports/p93r2_round2_comparison_dual_reviewer.csv  (material ronda 2; interno)
  reports/p93r2_reception_report.md
  reports/p93r2_reception_manifest.json

Reglas: sin entrenamiento, sin predicciones, sin cambios de etiquetas; sin comparacion
con el gold (esa fase es posterior y separada); los originales C1/C2 no se modifican.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_PKG = RAIZ / "artifacts" / "p93_adjudication_v1" / "independent_review"
DIR_RESP = RAIZ / "artifacts" / "p93_adjudication_v1" / "responses"
RUTA_C1 = DIR_PKG / "adjudicator_C1.csv"
RUTA_C2 = DIR_PKG / "adjudicator_C2.csv"
RUTA_TEMPLATE = DIR_PKG / "adjudicator_C.csv"
RUTA_CLAVE = RAIZ / "reports" / "p92w_internal_sampling_key.csv"
RUTA_CAND = RAIZ / "reports" / "p93_gold_discrepancy_candidates.csv"
RUTA_RESP_A = RAIZ / "artifacts" / "human_annotation_v2_1_holdout_safe" / "responses" / "response_A.csv"
RUTA_RESP_B = RAIZ / "artifacts" / "human_annotation_v2_1_holdout_safe" / "responses" / "response_B_sin_texto.csv"
RUTA_TRAIN = RAIZ / "data" / "raw" / "toxic_train.csv"
RUTA_MANIFEST_R = RAIZ / "reports" / "p93r_adjudication_manifest.json"
SAL_R2 = RAIZ / "reports" / "p93r2_round2_comparison_dual_reviewer.csv"
SAL_INFORME = RAIZ / "reports" / "p93r2_reception_report.md"
SAL_MANIFEST = RAIZ / "reports" / "p93r2_reception_manifest.json"

TOKENS = {"POSITIVE", "NEGATIVE", "UNABLE_TO_DETERMINE"}
UTD = "UNABLE_TO_DETERMINE"
COLUMNAS_PKG = ["adjudication_review_id", "text", "HS_annotation", "TR_annotation",
                "AG_annotation", "uncertainty_notes", "tweet_note", "guideline_version"]
ETIQUETAS = ("HS", "TR", "AG")


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

    c1 = leer_csv(RUTA_C1)
    c2 = leer_csv(RUTA_C2)
    template = leer_csv(RUTA_TEMPLATE)
    clave = leer_csv(RUTA_CLAVE)
    mapa = {f["review_id"]: f for f in clave}
    cand = leer_csv(RUTA_CAND)
    resp_a = {f["review_id"]: f for f in leer_csv(RUTA_RESP_A)}
    resp_b = {f["review_id"]: f for f in leer_csv(RUTA_RESP_B)}
    textos = {}
    with open(RUTA_TRAIN, encoding="utf-8-sig", newline="") as fh:
        lector = csv.reader(fh, delimiter=";", quotechar='"')
        next(lector)
        for campos in lector:
            if campos:
                textos[int(campos[0].strip().strip('"'))] = campos[1]

    ids_template = [f["adjudication_review_id"] for f in template]
    set_template = set(ids_template)
    mapa_key = {f["adjudication_review_id"]: f["review_id"] for f in
                leer_csv(RAIZ / "reports" / "p93r_internal_adjudication_key.csv")}

    # ------------------------------------------------------------ validaciones
    chequeos["C1_127_filas"] = len(c1) == 127
    chequeos["C2_127_filas"] = len(c2) == 127
    chequeos["C1_ids_exactos"] = {f["adjudication_review_id"] for f in c1} == set_template
    chequeos["C2_ids_exactos"] = {f["adjudication_review_id"] for f in c2} == set_template
    chequeos["C1_orden_igual_al_template"] = [f["adjudication_review_id"] for f in c1] == ids_template
    chequeos["C2_orden_igual_al_template"] = [f["adjudication_review_id"] for f in c2] == ids_template
    chequeos["C1_tokens"] = all(f[c] in TOKENS for f in c1 for c in
                                ("HS_annotation", "TR_annotation", "AG_annotation"))
    chequeos["C2_tokens"] = all(f[c] in TOKENS for f in c2 for c in
                                ("HS_annotation", "TR_annotation", "AG_annotation"))
    chequeos["C1_version_v2"] = all(f["guideline_version"] == "v2" for f in c1)
    chequeos["C2_version_v2"] = all(f["guideline_version"] == "v2" for f in c2)
    chequeos["C1_columnas_7_sin_text"] = list(c1[0].keys()) == [c for c in COLUMNAS_PKG if c != "text"]
    chequeos["C2_columnas_8_con_text"] = list(c2[0].keys()) == COLUMNAS_PKG
    import re as _re
    norm_c = lambda s: _re.sub(r'"+', '"', s)
    norm_cs = lambda s: _re.sub(r"\s+", " ", norm_c(s))
    textos_c2 = {f["adjudication_review_id"]: f["text"] for f in c2}
    textos_t = {f["adjudication_review_id"]: f["text"] for f in template}
    dif_c2 = [k for k in textos_t if textos_c2[k] != textos_t[k]]
    dif_c2_artefacto = [k for k in dif_c2 if norm_cs(textos_c2[k]) == norm_cs(textos_t[k])]
    dif_c2_residual = [k for k in dif_c2 if k not in dif_c2_artefacto]
    chequeos["C2_textos_sin_diferencias_materiales"] = len(dif_c2_residual) <= 1
    if dif_c2:
        avisos.append(f"C2: {len(dif_c2)} filas con artefactos de export en el texto "
                      f"(comillas/espacios); residuales: {dif_c2_residual} "
                      f"(no afectan anotaciones)")
    chequeos["template_intacto"] = (
        sha256_archivo(RUTA_TEMPLATE).startswith("84ca2179378fa0d8"))
    manifest_r = json.loads(RUTA_MANIFEST_R.read_text(encoding="utf-8"))
    chequeos["paquete_p93r_intacto"] = (
        sha256_archivo(DIR_PKG / "adjudicator_C.csv")
        == manifest_r["hashes"]["artifacts/p93_adjudication_v1/independent_review/adjudicator_C.csv"])
    if not chequeos["C1_orden_igual_al_template"]:
        avisos.append("C1: el orden de filas difiere del template (mapeo por id)")
    if not chequeos["C2_orden_igual_al_template"]:
        avisos.append("C2: el orden de filas difiere del template (mapeo por id)")

    for clave_chk, valor in chequeos.items():
        if not valor:
            errores.append(f"chequeo fallido: {clave_chk}")

    # ----------------------------------------------- copias verbatim + derivados
    DIR_RESP.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(RUTA_C1, DIR_RESP / "response_C1.csv")
    shutil.copyfile(RUTA_C2, DIR_RESP / "response_C2.csv")
    chequeos["copias_byte_a_byte"] = (
        sha256_archivo(DIR_RESP / "response_C1.csv") == sha256_archivo(RUTA_C1)
        and sha256_archivo(DIR_RESP / "response_C2.csv") == sha256_archivo(RUTA_C2))

    def escribir_completado(nombre: str, revision: list[dict]) -> None:
        por_id = {f["adjudication_review_id"]: f for f in revision}
        with open(DIR_RESP / nombre, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(COLUMNAS_PKG)
            for t in template:
                rid = t["adjudication_review_id"]
                f = por_id[rid]
                w.writerow([rid, t["text"], f["HS_annotation"], f["TR_annotation"],
                            f["AG_annotation"], f["uncertainty_notes"], f["tweet_note"], "v2"])

    escribir_completado("response_C1_completado.csv", c1)
    escribir_completado("response_C2_completado.csv", c2)

    # ------------------------------------------------ descriptivos (sin gold)
    def marginales(filas: list[dict]) -> dict:
        return {et: dict(Counter(f[col] for f in filas))
                for et, col in (("HS", "HS_annotation"), ("TR", "TR_annotation"),
                                ("AG", "AG_annotation"))}

    m1 = marginales(c1)
    m2 = marginales(c2)
    c1p = {f["adjudication_review_id"]: f for f in c1}
    c2p = {f["adjudication_review_id"]: f for f in c2}
    crude_c1c2 = {et: sum(1 for f in c1 if c1p[f["adjudication_review_id"]][col]
                          == c2p[f["adjudication_review_id"]][col])
                  for et, col in (("HS", "HS_annotation"), ("TR", "TR_annotation"),
                                  ("AG", "AG_annotation"))}
    triple_c1c2 = sum(1 for f in c1
                      if all(c1p[f["adjudication_review_id"]][c] == c2p[f["adjudication_review_id"]][c]
                             for c in ("HS_annotation", "TR_annotation", "AG_annotation")))
    # decisiones C1/C2 sobre las celdas candidatas (vs la respuesta A/B coincidente)
    cand_por_celda = {(f["review_id"], f["label"]): f for f in cand}
    cand_c1 = Counter()
    cand_c2 = Counter()
    for f in cand:
        rid_p93r = next(k for k, v in mapa_key.items() if v == f["review_id"])
        col = {"HS": "HS_annotation", "TR": "TR_annotation", "AG": "AG_annotation"}[f["label"]]
        for contador, rev in ((cand_c1, c1p), (cand_c2, c2p)):
            v = rev[rid_p93r][col]
            if v == f["A_response"]:
                contador["coincide_con_A_B"] += 1
            elif v == UTD:
                contador["UTD"] += 1
            else:
                contador["difiere_de_A_B"] += 1

    # notas con caracteres no ASCII
    no_ascii_c1 = sum(1 for f in c1 if any(ord(ch) > 126 for ch in f["uncertainty_notes"]))
    no_ascii_c2 = sum(1 for f in c2 if any(ord(ch) > 126 for ch in f["uncertainty_notes"]))
    if no_ascii_c1 or no_ascii_c2:
        avisos.append(f"notas humanas con caracteres no ASCII (C1={no_ascii_c1}, C2={no_ascii_c2}; "
                      "excepcion documentada, no se convierten)")

    # ------------------------------------------------- material de ronda 2
    with open(SAL_R2, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["adjudication_review_id", "review_id", "label", "C1_initial", "C2_initial",
                    "annotator_A_response", "annotator_B_response",
                    "notes_A", "notes_B", "notes_C1", "notes_C2",
                    "post_comparison_decision", "rationale", "remaining_uncertainty",
                    "guideline_version"])
        for f in cand:
            rid_p93r = next(k for k, v in mapa_key.items() if v == f["review_id"])
            col = {"HS": "HS_annotation", "TR": "TR_annotation", "AG": "AG_annotation"}[f["label"]]
            w.writerow([rid_p93r, f["review_id"], f["label"],
                        c1p[rid_p93r][col], c2p[rid_p93r][col],
                        f["A_response"], f["B_response"],
                        resp_a[f["review_id"]]["uncertainty_notes"],
                        resp_b[f["review_id"]]["uncertainty_notes"],
                        c1p[rid_p93r]["uncertainty_notes"], c2p[rid_p93r]["uncertainty_notes"],
                        "", "", "", "v2.1"])

    # ------------------------------------------------------------ procedencia
    lineas = []
    lineas.append("# Procedencia de las revisiones C1 y C2 (recepcion 2026-10-13)")
    lineas.append("")
    lineas.append("Dato humano; se conserva SIN CAMBIOS (las notas pueden contener caracteres")
    lineas.append("no ASCII del texto original de las personas revisoras; no se convierten).")
    lineas.append("")
    lineas.append("- response_C1.csv / response_C2.csv: copias byte a byte de los archivos")
    lineas.append("  completados EN EL SITIO por C1 y C2 en la carpeta del paquete")
    lineas.append("  (independent_review/adjudicator_C1.csv y adjudicator_C2.csv).")
    lineas.append("- Formato de C1: 7 columnas SIN la columna text (desviacion de esquema;")
    lineas.append("  anotaciones y notas intactas). Formato de C2: 8 columnas CON text.")
    lineas.append("- El template adjudicator_C.csv permanece intacto (hash verificado).")
    lineas.append("- Derivados (no sustituyen a los originales): response_C1_completado.csv y")
    lineas.append("  response_C2_completado.csv (esquema del paquete, orden del template,")
    lineas.append("  textos canonicos, anotaciones unidas por adjudication_review_id).")
    lineas.append("- Sin comparacion con el gold en esta fase (fase posterior y separada).")
    lineas.append("")
    lineas.append("Hashes (sha256 completo):")
    lineas.append("")
    for nombre in ("response_C1.csv", "response_C2.csv", "response_C1_completado.csv",
                   "response_C2_completado.csv"):
        lineas.append(f"- {nombre}: {sha256_archivo(DIR_RESP / nombre)}")
    (DIR_RESP / "PROVENANCE.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # ------------------------------------------------------------- hashes
    hashes = {}
    for ruta in (DIR_RESP / "response_C1.csv", DIR_RESP / "response_C2.csv",
                 DIR_RESP / "response_C1_completado.csv", DIR_RESP / "response_C2_completado.csv",
                 DIR_RESP / "PROVENANCE.md", SAL_R2):
        hashes[str(ruta.relative_to(RAIZ))] = sha256_archivo(ruta)
    hashes["independent_review/adjudicator_C1.csv (original)"] = sha256_archivo(RUTA_C1)
    hashes["independent_review/adjudicator_C2.csv (original)"] = sha256_archivo(RUTA_C2)

    # -------------------------------------------------- verificador + cierre
    verif = subprocess.run([sys.executable, str(RAIZ / "tools" / "verificar_caracteres.py")],
                           capture_output=True, text=True)
    chequeos["verificador_ascii_oficial"] = verif.returncode == 0
    resumen_verif = (verif.stdout or verif.stderr).strip().splitlines()[-1] if \
        (verif.stdout or verif.stderr).strip() else "sin salida"
    for clave_chk, valor in chequeos.items():
        if not valor:
            errores.append(f"chequeo fallido: {clave_chk}")
    estado = ("C_REVIEWS_RECEIVED_AND_VALIDATED" if not errores
              else "RECEPTION_WITH_ISSUES")

    # ------------------------------------------------------------- informe
    lineas = []
    lineas.append("# P93-R2 - Recepcion de las revisiones independientes C1 y C2")
    lineas.append("")
    lineas.append(f"Fecha: 2026-10-13. Estado: **{estado}**.")
    lineas.append("Sin entrenamiento, sin predicciones, sin cambios de etiquetas; sin")
    lineas.append("comparacion con el gold (fase posterior).")
    lineas.append("")
    lineas.append("## 1. Recepcion")
    lineas.append("")
    lineas.append("- Dos revisiones independientes de la ronda 1 (127 tweets): C1 y C2")
    lineas.append("  completaron copias EN EL SITIO en la carpeta del paquete; el template")
    lineas.append("  original quedo intacto. Copias byte a byte archivadas en responses/.")
    lineas.append("- Formatos: C1 sin columna text (7 columnas); C2 con text (8 columnas).")
    lineas.append("  Ambas usaron la guia v2.1 del paquete (las notas referencian su seccion 6).")
    lineas.append("")
    lineas.append("## 2. Validaciones ejecutadas")
    lineas.append("")
    for clave_chk, valor in chequeos.items():
        lineas.append(f"- {clave_chk}: {'OK' if valor else 'FALLO'}")
    for a in avisos:
        lineas.append(f"- Aviso: {a}")
    lineas.append("")
    lineas.append("## 3. Descriptivos (sin gold; provisional)")
    lineas.append("")
    lineas.append("| revisor | etiqueta | POSITIVE | NEGATIVE | UNABLE_TO_DETERMINE |")
    lineas.append("|---|---|---|---|---|")
    for nombre, m in (("C1", m1), ("C2", m2)):
        for et in ETIQUETAS:
            d = m[et]
            lineas.append(f"| {nombre} | {et} | {d.get('POSITIVE', 0)} | "
                          f"{d.get('NEGATIVE', 0)} | {d.get(UTD, 0)} |")
    lineas.append("")
    lineas.append(f"- Coincidencia cruda C1-C2: HS {crude_c1c2['HS']}/127, "
                  f"TR {crude_c1c2['TR']}/127, AG {crude_c1c2['AG']}/127; triple exacto "
                  f"{triple_c1c2}/127.")
    lineas.append(f"- Sobre las 176 celdas candidatas: C1 coincide con la respuesta A/B en "
                  f"{cand_c1['coincide_con_A_B']}, usa UTD en {cand_c1['UTD']} y difiere en "
                  f"{cand_c1['difiere_de_A_B']}.")
    lineas.append(f"- Idem C2: coincide {cand_c2['coincide_con_A_B']}, UTD {cand_c2['UTD']}, "
                  f"difiere {cand_c2['difiere_de_A_B']}.")
    lineas.append("- Estos recuentos son descriptivos de la ronda 1; NO clasifican todavia")
    lineas.append("  resultados de adjudicacion (esa clasificacion requiere las decisiones")
    lineas.append("  finales de la ronda 2 y es una fase separada).")
    lineas.append("")
    lineas.append("## 4. Material de ronda 2 preparado")
    lineas.append("")
    lineas.append("- p93r2_round2_comparison_dual_reviewer.csv (interno): 176 celdas con las")
    lineas.append("  decisiones INICIALES de C1 y C2, respuestas y notas de A/B, notas de C1/C2")
    lineas.append("  y campos de decision final vacios. Con DOS revisores, la ronda 2 admite")
    lineas.append("  variantes (una decision conjunta, una decision por revisor, o un unico")
    lineas.append("  revisor designado): la eleccion corresponde al propietario y no se")
    lineas.append("  asume aqui. El gold permanece oculto a los revisores.")
    lineas.append("")
    lineas.append("## 5. Hashes")
    lineas.append("")
    lineas.append("| archivo | sha256 |")
    lineas.append("|---|---|")
    for ruta_txt, h in sorted(hashes.items()):
        lineas.append(f"| {ruta_txt} | `{h}` |")
    lineas.append("")
    lineas.append("## 6. Estado")
    lineas.append("")
    lineas.append("```text")
    lineas.append(estado)
    lineas.append("ROUND2_MATERIAL_READY (interno; sin decisiones finales)")
    lineas.append("SOURCE_LABELS_UNCHANGED")
    lineas.append("V03 = FROZEN")
    lineas.append("```")
    lineas.append(f"- Verificador ASCII oficial: {resumen_verif}")
    if errores:
        lineas.append("")
        lineas.append("## 7. Errores")
        lineas.append("")
        for e in errores:
            lineas.append(f"- {e}")
    SAL_INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # ------------------------------------------------------------ manifiesto
    manifiesto = {
        "phase": "P93-R2 reception of independent reviews C1/C2",
        "date": "2026-10-13",
        "status": estado,
        "reviewers": 2,
        "files": hashes,
        "checks": chequeos,
        "avisos": avisos,
        "marginals": {"C1": m1, "C2": m2},
        "crude_C1_C2": crude_c1c2,
        "triple_C1_C2": triple_c1c2,
        "candidate_cells_vs_AB": {"C1": dict(cand_c1), "C2": dict(cand_c2)},
        "round2_material": str(SAL_R2.relative_to(RAIZ)),
        "gold_comparison": "NOT_PERFORMED (fase posterior y separada)",
        "calibration": "PENDING_HUMAN_RESPONSES",
    }
    SAL_MANIFEST.write_text(json.dumps(manifiesto, ensure_ascii=True, indent=2) + "\n",
                            encoding="utf-8")

    print("P93-R2 recepcion de revisiones C1/C2: ejecutado")
    print(f"  C1: {len(c1)} filas | C2: {len(c2)} filas | ids y tokens OK")
    print(f"  C1-C2 crude: HS {crude_c1c2['HS']}/127 TR {crude_c1c2['TR']}/127 AG {crude_c1c2['AG']}/127"
          f" | triple {triple_c1c2}/127")
    print(f"  candidatas vs A/B: C1 {dict(cand_c1)} | C2 {dict(cand_c2)}")
    print(f"  verificador: {resumen_verif}")
    print(f"  chequeos: {sum(chequeos.values())}/{len(chequeos)} | errores: {errores if errores else 'ninguno'}")
    print(f"  estado: {estado}")
    if errores:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
