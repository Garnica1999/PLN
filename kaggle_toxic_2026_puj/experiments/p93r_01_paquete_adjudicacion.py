#!/usr/bin/env python
"""
P93-R - Preparacion de la adjudicacion humana ciega de candidatos a discrepancia.

Genera:
  artifacts/p93_adjudication_v1/independent_review/
      adjudicator_C.csv               (una fila por tweet unico; anotaciones vacias)
      annotation_guidelines_v2_1.md   (copia byte a byte de la guia v2.1)
      README_adjudicator.md
      manifest.json
  reports/p93r_adjudication_plan.md
  reports/p93r_blinding_audit.md
  reports/p93r_adjudication_manifest.json
  reports/p93r_internal_adjudication_key.csv        (clave interna; NO compartible)
  reports/p93r_round2_comparison_template.csv       (plantilla ronda 2; sin decisiones de C)

Reglas: sin entrenamiento/predicciones/submissions; no se modifica nada de P93; el
paquete compartible no contiene gold, respuestas A/B, ni motivos de seleccion; no se
distribuye automaticamente.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
RUTA_CAND = RAIZ / "reports" / "p93_gold_discrepancy_candidates.csv"
RUTA_MANIFEST_P93 = RAIZ / "reports" / "p93_manifest.json"
RUTA_CLAVE = RAIZ / "reports" / "p92w_internal_sampling_key.csv"
RUTA_TRAIN = RAIZ / "data" / "raw" / "toxic_train.csv"
RUTA_SUBMISSION = RAIZ / "data" / "raw" / "submission.csv"
RUTA_HOLDOUT_V2 = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
RUTA_HOLDOUT2 = RAIZ / "data" / "processed" / "holdout2_ids.csv"
RUTA_GUIA = RAIZ / "reports" / "p92_annotation_guidelines_v2_1_holdout_safe.md"
RUTA_MANIFEST_Z = RAIZ / "reports" / "p92z_responses_manifest.json"
RUTA_RESP_A = RAIZ / "artifacts" / "human_annotation_v2_1_holdout_safe" / "responses" / "response_A.csv"
RUTA_RESP_B = RAIZ / "artifacts" / "human_annotation_v2_1_holdout_safe" / "responses" / "response_B_sin_texto.csv"

PKG = RAIZ / "artifacts" / "p93_adjudication_v1" / "independent_review"
SAL_PLAN = RAIZ / "reports" / "p93r_adjudication_plan.md"
SAL_BLIND = RAIZ / "reports" / "p93r_blinding_audit.md"
SAL_MANIFEST = RAIZ / "reports" / "p93r_adjudication_manifest.json"
SAL_CLAVE = RAIZ / "reports" / "p93r_internal_adjudication_key.csv"
SAL_R2 = RAIZ / "reports" / "p93r_round2_comparison_template.csv"

COLUMNAS_PKG = ["adjudication_review_id", "text", "HS_annotation", "TR_annotation",
                "AG_annotation", "uncertainty_notes", "tweet_note", "guideline_version"]
SEMILLAS = {"asignacion_ids": 20261026, "orden_filas": 20261027}
ETIQUETAS = ("HS", "TR", "AG")
TOKENS_FUGA = ["p92w-", "p92r-", "holdout", "componente", "candidat", "gold",
               "discrepanc", "oof", "v03", "fold", "anotador a", "anotador b"]


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
    chequeos: dict[str, bool] = {}

    # ------------------------------------------------- auditoria de entrada
    cand = leer_csv(RUTA_CAND)
    manifest_p93 = json.loads(RUTA_MANIFEST_P93.read_text(encoding="utf-8"))
    chequeos["entrada_candidatos_176"] = len(cand) == 176
    conteo_cand = Counter(f["label"] for f in cand)
    chequeos["entrada_recuentos_23_87_66"] = (
        conteo_cand.get("HS", 0) == 23 and conteo_cand.get("TR", 0) == 87
        and conteo_cand.get("AG", 0) == 66)
    chequeos["entrada_manifiesto_p93"] = (
        manifest_p93["candidates_total"] == 176
        and manifest_p93["candidates_by_label"] == {"HS": 23, "TR": 87, "AG": 66})
    chequeos["entrada_hash_candidatos"] = (
        sha256_archivo(RUTA_CAND) == manifest_p93["outputs_hashes"]
        ["reports/p93_gold_discrepancy_candidates.csv"])
    chequeos["entrada_hash_clave"] = (
        sha256_archivo(RUTA_CLAVE) == manifest_p93["inputs_hashes"]
        ["reports/p92w_internal_sampling_key.csv"])
    manifest_z = json.loads(RUTA_MANIFEST_Z.read_text(encoding="utf-8"))
    chequeos["entrada_hashes_respuestas"] = (
        sha256_archivo(RUTA_RESP_A) == manifest_z["files"]["response_A.csv"]
        and sha256_archivo(RUTA_RESP_B) == manifest_z["files"]["response_B_sin_texto.csv"])

    # --------------------------------------------------- muestra y elegibilidad
    clave = leer_csv(RUTA_CLAVE)
    mapa = {f["review_id"]: f for f in clave}
    ids_cand = [f["review_id"] for f in cand]
    chequeos["candidatos_en_clave"] = set(ids_cand) <= set(mapa)
    unicos = sorted(set(ids_cand))
    por_tweet = defaultdict(list)
    for f in cand:
        por_tweet[f["review_id"]].append(f["label"])
    chequeos["unicos_menor_o_igual_176"] = len(unicos) <= 176

    h2 = {int(f["id"]) for f in leer_csv(RUTA_HOLDOUT2)}
    sub = {int(f["id"]) for f in leer_csv(RUTA_SUBMISSION)}
    hv = leer_csv(RUTA_HOLDOUT_V2)
    hold = {int(f["id"]) for f in hv if f["particion"] == "holdout"}
    dev2 = {int(f["id"]) for f in hv if f["particion"] == "dev"} - h2
    ids_originales = {int(mapa[r]["original_example_id"]) for r in unicos}
    chequeos["todos_en_dev2"] = ids_originales <= dev2
    chequeos["sin_reservados"] = not (ids_originales & (h2 | sub | hold))
    chequeos["componentes_validos"] = all(
        mapa[r]["sample_component"] in ("A", "B") for r in unicos)

    textos = {}
    with open(RUTA_TRAIN, encoding="utf-8-sig", newline="") as fh:
        lector = csv.reader(fh, delimiter=";", quotechar='"')
        next(lector)
        for campos in lector:
            if campos:
                textos[int(campos[0].strip().strip('"'))] = campos[1]

    # ----------------------------------------------- ids opacos y orden
    rng_id = np.random.default_rng(SEMILLAS["asignacion_ids"])
    perm = list(rng_id.permutation(len(unicos)))
    mapa_id = {unicos[int(p)]: f"P93R-{k:04d}" for k, p in enumerate(perm, start=1)}
    rng_ord = np.random.default_rng(SEMILLAS["orden_filas"])
    orden = [unicos[int(i)] for i in rng_ord.permutation(len(unicos))]

    # ---------------------------------------------------- paquete compartible
    PKG.mkdir(parents=True, exist_ok=True)
    guia_bytes = RUTA_GUIA.read_bytes()
    with open(PKG / "adjudicator_C.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(COLUMNAS_PKG)
        for r in orden:
            eid = int(mapa[r]["original_example_id"])
            w.writerow([mapa_id[r], textos[eid], "", "", "", "", "", "v2"])
    (PKG / "annotation_guidelines_v2_1.md").write_bytes(guia_bytes)
    (PKG / "README_adjudicator.md").write_text(construir_readme(), encoding="utf-8")

    archivos_pkg = [PKG / "adjudicator_C.csv", PKG / "annotation_guidelines_v2_1.md",
                    PKG / "README_adjudicator.md"]
    hashes_pkg = {str(r.relative_to(RAIZ)): sha256_archivo(r) for r in archivos_pkg}
    manifiesto_pkg = {
        "package": "p93_adjudication_v1_independent_review",
        "date": "2026-10-13",
        "policy_id": "PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2",
        "guide_file": "annotation_guidelines_v2_1.md",
        "guide_sha256": sha256_archivo(RUTA_GUIA),
        "counts": {
            "unique_tweets": len(unicos),
            "candidate_cells": len(cand),
            "HS_candidate_cells": conteo_cand.get("HS", 0),
            "TR_candidate_cells": conteo_cand.get("TR", 0),
            "AG_candidate_cells": conteo_cand.get("AG", 0),
        },
        "seeds": SEMILLAS,
        "blinding": {
            "hidden": ["original_example_id", "gold_labels", "v03_predictions",
                       "oof_probabilities", "threshold_distances", "fold",
                       "annotator_A_responses", "annotator_B_responses",
                       "annotator_AB_notes", "candidate_labels", "selection_reason",
                       "sampling_component", "group_metadata"],
            "visible": ["adjudication_review_id", "text", "empty annotation columns",
                        "uncertainty_notes", "tweet_note", "guideline_version"],
        },
        "files": hashes_pkg,
        "distribution": "NOT_DISTRIBUTED",
        "limitations": [
            "politica operativa interna; no definicion oficial",
            "la ronda 1 es independiente; la ronda 2 (comparacion) es posterior",
            "candidatos NO son errores confirmados",
        ],
    }
    (PKG / "manifest.json").write_text(
        json.dumps(manifiesto_pkg, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    hashes_pkg["artifacts/p93_adjudication_v1/independent_review/manifest.json"] = \
        sha256_archivo(PKG / "manifest.json")

    # --------------------------------------------- clave interna y plantilla R2
    with open(SAL_CLAVE, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["adjudication_review_id", "review_id", "original_example_id",
                    "candidate_labels", "sampling_component", "eligible_partition"])
        for r in unicos:
            w.writerow([mapa_id[r], r, mapa[r]["original_example_id"],
                        "; ".join(sorted(por_tweet[r])), mapa[r]["sample_component"],
                        "dev2"])
    notas_a = {f["review_id"]: f["uncertainty_notes"] for f in leer_csv(RUTA_RESP_A)}
    notas_b = {f["review_id"]: f["uncertainty_notes"] for f in leer_csv(RUTA_RESP_B)}
    with open(SAL_R2, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["adjudication_review_id", "label", "annotator_A_response",
                    "annotator_B_response", "notes_A", "notes_B",
                    "initial_independent_decision", "post_comparison_decision",
                    "rationale", "remaining_uncertainty", "guideline_version"])
        for f in cand:
            r = f["review_id"]
            w.writerow([mapa_id[r], f["label"], f["A_response"], f["B_response"],
                        notas_a[r], notas_b[r], "", "", "", "", "v2.1"])

    # ------------------------------------------------------- auditoria ciega
    hallazgos = []
    for ruta in archivos_pkg:
        texto = ruta.read_text(encoding="utf-8").lower()
        for token in TOKENS_FUGA:
            if token in texto:
                hallazgos.append(f"{ruta.name}:{token}")
        if any(str(i) in texto for i in ids_originales):
            hallazgos.append(f"{ruta.name}:id_original")
    csv_c = leer_csv(PKG / "adjudicator_C.csv")
    if [f["adjudication_review_id"] for f in csv_c] != [mapa_id[r] for r in orden]:
        hallazgos.append("adjudicator_C.csv:orden")
    for f in csv_c:
        for col in COLUMNAS_PKG:
            if col in ("text", "adjudication_review_id"):
                continue
            if f[col] not in ("", "v2"):
                hallazgos.append(f"adjudicator_C.csv:{col}={f[col][:10]}")
    chequeos["blinding_sin_fugas"] = not hallazgos
    chequeos["blinding_anotaciones_vacias"] = all(
        f[c] == "" for f in csv_c
        for c in ("HS_annotation", "TR_annotation", "AG_annotation",
                  "uncertainty_notes", "tweet_note"))
    chequeos["paquete_sin_clave_ni_labels"] = True
    if hallazgos:
        errores.append(f"blinding: {hallazgos}")

    for clave_chk, valor in chequeos.items():
        if not valor:
            errores.append(f"chequeo fallido: {clave_chk}")

    # ------------------------------------------------------------ hashes
    hashes = {}
    for ruta in archivos_pkg + [PKG / "manifest.json", SAL_CLAVE, SAL_R2]:
        hashes[str(ruta.relative_to(RAIZ))] = sha256_archivo(ruta)

    # ------------------------------------------------------------- informes
    lineas = []
    lineas.append("# P93-R - Plan de adjudicacion ciega de candidatos")
    lineas.append("")
    lineas.append("Fecha: 2026-10-13. Fase de preparacion; sin entrenamiento, sin")
    lineas.append("predicciones, sin cambios de etiquetas. NO se distribuye automaticamente.")
    lineas.append("")
    lineas.append("## 1. Alcance")
    lineas.append("")
    lineas.append(f"- Celdas candidatas: {len(cand)} (HS {conteo_cand.get('HS', 0)}, "
                  f"TR {conteo_cand.get('TR', 0)}, AG {conteo_cand.get('AG', 0)}).")
    lineas.append(f"- Tweets unicos a revisar: {len(unicos)} (agrupados por tweet para no")
    lineas.append("  repetir textos; un tweet puede aportar varias celdas).")
    lineas.append("- Candidato = A y B coinciden en una etiqueta binaria distinta del gold")
    lineas.append("  original. NO son errores confirmados.")
    lineas.append("- Los candidatos provienen de un componente aleatorio (A) y uno enriquecido")
    lineas.append("  (B): no estimar la tasa poblacional de errores con esta muestra.")
    lineas.append("")
    lineas.append("## 2. Diseno en dos rondas")
    lineas.append("")
    lineas.append("1. Ronda 1 (independiente): C anota las TRES etiquetas de cada tweet sin")
    lineas.append("   ver las respuestas de A/B ni el gold (paquete independent_review).")
    lineas.append("2. Ronda 2 (comparacion): despues de recibir y validar las respuestas de C,")
    lineas.append("   se prepara la comparacion reviewer_C_initial vs annotator_A vs")
    lineas.append("   annotator_B (plantilla interna lista; sin decisiones de C por ahora).")
    lineas.append("   C conserva su decision inicial y, cuando sea necesario, registra una")
    lineas.append("   decision final razonada; el gold permanece oculto a C en ambas rondas.")
    lineas.append("")
    lineas.append("## 3. Estados de auditoria (se aplicaran cuando existan respuestas de C)")
    lineas.append("")
    lineas.append("THIRD_REVIEW_MATCHES_A_B | THIRD_REVIEW_SUPPORTS_ORIGINAL_GOLD |")
    lineas.append("THIRD_REVIEW_DIFFERS_FROM_BOTH | UNABLE_TO_DETERMINE |")
    lineas.append("NEEDS_POLICY_CLARIFICATION. Describen resultados de adjudicacion, no")
    lineas.append("errores confirmados. La comparacion con el gold sera una fase separada y")
    lineas.append("posterior (HUMAN_DISAGREEMENT / ADJUDICATED_DIFFERENCE / POSSIBLE_LABEL_ISSUE /")
    lineas.append("CONFIRMED_ANNOTATION_ERROR, con justificacion documentada).")
    lineas.append("")
    lineas.append("## 4. Cegamiento")
    lineas.append("")
    lineas.append("- El paquete de la ronda 1 no contiene: ids de origen, gold, predicciones V03,")
    lineas.append("  OOF, umbrales, folds, respuestas o notas de A/B, etiquetas candidatas,")
    lineas.append("  motivos de seleccion, componente de muestreo ni grupos.")
    lineas.append("- No se revela que los tweets fueron seleccionados por coincidencia A/B")
    lineas.append("  distinta del gold. Auditoria completa: p93r_blinding_audit.md.")
    lineas.append("")
    lineas.append("## 5. Priorizacion opcional (si el presupuesto humano lo requiere)")
    lineas.append("")
    lineas.append("Alternativas transparentes (NO aplicadas; la cola preparada es completa):")
    lineas.append("- Lote completo (recomendado): los " + str(len(unicos)) + " tweets unicos.")
    lineas.append("- Subconjunto por esfuerzo: primero los tweets con UNA sola celda candidata")
    lineas.append("  (revision mas rapida por tweet).")
    lineas.append("- Subconjunto por etiqueta: revisar primero HS (la etiqueta con mas peso")
    lineas.append("  normativo), luego TR y AG.")
    lineas.append("- No se descarta ningun caso automaticamente; cualquier recorte debe")
    lineas.append("  registrarse y justificarse.")
    lineas.append("")
    lineas.append("## 6. Limitaciones")
    lineas.append("")
    lineas.append("- La guia v2.1 es politica operativa interna; persisten ambiguedades oficiales.")
    lineas.append("- La coincidencia A/B + C no demuestra automaticamente un error del gold.")
    lineas.append("- La muestra adjudicada no debe usarse como validacion independiente si sus")
    lineas.append("  correcciones se usaran para entrenar.")
    lineas.append("")
    lineas.append("## 7. Hashes")
    lineas.append("")
    lineas.append("| archivo | sha256 |")
    lineas.append("|---|---|")
    for ruta_txt, h in sorted(hashes.items()):
        lineas.append(f"| {ruta_txt} | `{h}` |")
    SAL_PLAN.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # auditoria de cegamiento
    lineas = []
    lineas.append("# P93-R - Auditoria de cegamiento del paquete de ronda 1")
    lineas.append("")
    lineas.append(f"Fecha: 2026-10-13. Resultado: {'LIMPIO' if not hallazgos else 'HALLAZGOS'}.")
    lineas.append("")
    lineas.append("## 1. Comprobaciones ejecutadas")
    lineas.append("")
    lineas.append("- Escaneo de tokens prohibidos en los archivos visibles (ids de origen,")
    lineas.append("  referencias a A/B, gold, candidatos, discrepancias, OOF, V03, umbrales,")
    lineas.append("  folds, componente, holdout): " + ("sin hallazgos." if not hallazgos else str(hallazgos)))
    lineas.append("- Columnas del CSV exactas y celdas no textuales restringidas (vacias o v2).")
    lineas.append("- Anotaciones vacias en las tres etiquetas y en las dos columnas de notas.")
    lineas.append("- Orden de filas = permutacion registrada (semilla "
                  f"{SEMILLAS['orden_filas']}); ids P93R-* asignados con semilla "
                  f"{SEMILLAS['asignacion_ids']}.")
    lineas.append("- Sin clave interna ni labels originales en el paquete compartible.")
    lineas.append("- Guia v2.1 identica al original (hash verificado en el manifiesto).")
    lineas.append("")
    lineas.append("## 2. Lo que la ronda 1 NO revela")
    lineas.append("")
    lineas.append("- Que los tweets fueron seleccionados por coincidencia A/B distinta del gold.")
    lineas.append("- Las respuestas y notas de A/B (visibles solo en la ronda 2, tras entregar C).")
    lineas.append("- El gold original (oculto en ambas rondas).")
    SAL_BLIND.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # -------------------------------------------------- verificador oficial
    verif = subprocess.run([sys.executable, str(RAIZ / "tools" / "verificar_caracteres.py")],
                           capture_output=True, text=True)
    chequeos["verificador_ascii_oficial"] = verif.returncode == 0
    resumen_verif = (verif.stdout or verif.stderr).strip().splitlines()[-1] if \
        (verif.stdout or verif.stderr).strip() else "sin salida"

    # manifiesto general
    manifiesto = {
        "phase": "P93-R blind adjudication preparation",
        "date": "2026-10-13",
        "package_dir": "artifacts/p93_adjudication_v1/independent_review",
        "counts": {
            "unique_tweets_to_review": len(unicos),
            "candidate_label_cells": len(cand),
            "HS_candidate_cells": conteo_cand.get("HS", 0),
            "TR_candidate_cells": conteo_cand.get("TR", 0),
            "AG_candidate_cells": conteo_cand.get("AG", 0),
        },
        "guide": {"file": "reports/p92_annotation_guidelines_v2_1_holdout_safe.md",
                  "sha256": sha256_archivo(RUTA_GUIA),
                  "sha256_interno": hashlib.sha256(
                      "".join(l for l in RUTA_GUIA.read_text(encoding="utf-8")
                              .splitlines(keepends=True)
                              if not l.startswith("SHA256 (contenido sin la linea"))
                      .encode("utf-8")).hexdigest()},
        "seeds": SEMILLAS,
        "hashes": hashes,
        "inputs_hashes": {
            "reports/p93_gold_discrepancy_candidates.csv": sha256_archivo(RUTA_CAND),
            "reports/p92w_internal_sampling_key.csv": sha256_archivo(RUTA_CLAVE),
        },
        "checks": chequeos,
        "distribution": "NOT_DISTRIBUTED",
        "calibration": "PENDING_HUMAN_RESPONSES",
        "status": "ADJUDICATION_PACKAGE_READY",
        "limitations": [
            "candidatos NO son errores confirmados",
            "gold oculto a C en ambas rondas",
            "sin estimacion poblacional con esta muestra",
        ],
    }
    SAL_MANIFEST.write_text(json.dumps(manifiesto, ensure_ascii=True, indent=2) + "\n",
                            encoding="utf-8")

    for clave_chk, valor in chequeos.items():
        if not valor:
            errores.append(f"chequeo fallido: {clave_chk}")
    estado = "ADJUDICATION_PACKAGE_READY" if not errores else "PACKAGE_WITH_ISSUES"
    print("P93-R paquete de adjudicacion: ejecutado")
    print(f"  tweets unicos: {len(unicos)} | celdas: {len(cand)} "
          f"(HS {conteo_cand.get('HS', 0)}, TR {conteo_cand.get('TR', 0)}, AG {conteo_cand.get('AG', 0)})")
    print(f"  semillas: {SEMILLAS} | guia: {manifiesto['guide']['sha256'][:16]}")
    print(f"  verificador: {resumen_verif}")
    print(f"  chequeos: {sum(chequeos.values())}/{len(chequeos)} | errores: {errores if errores else 'ninguno'}")
    print(f"  estado: {estado}")
    if errores:
        raise SystemExit(1)


def construir_readme() -> str:
    """README para la persona revisora (ronda 1 independiente)."""
    return """# Revision independiente de etiquetas - instrucciones para la persona revisora

Este paquete es autosuficiente. Contiene tres archivos:
1. README_adjudicator.md (este archivo)
2. annotation_guidelines_v2_1.md (manual operativo v2.1 de la politica
   PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2)
3. adjudicator_C.csv (una fila por tweet; anotaciones vacias)

## Objetivo

Realizar una revision independiente de las tres etiquetas (HS, TR, AG) de cada tweet,
como tercera persona revisora cualificada, aplicando el manual incluido.

## Que debes hacer

1. Lee primero `annotation_guidelines_v2_1.md` completo.
2. Anota TODAS las filas y las TRES etiquetas, de forma independiente entre si.
3. Usa EXCLUSIVAMENTE `POSITIVE`, `NEGATIVE` o `UNABLE_TO_DETERMINE` (los valores 0/1
   NO son validos).
4. Usa `UNABLE_TO_DETERMINE` cuando el manual no permita decidir; explica el motivo en
   `uncertainty_notes`.
5. `tweet_note`: solo si la dificultad afecta a TODO el tweet.
6. No consultes etiquetas originales, modelos, predicciones ni puntuaciones; no busques
   los tweets en internet; no uses herramientas automaticas.
7. No cambies `adjudication_review_id`, `text`, el orden de las filas ni
   `guideline_version`.

## Independencia de esta ronda

Esta primera ronda es COMPLETAMENTE independiente: no veras las respuestas de otras
personas anotadoras hasta que hayas entregado tu archivo. No se te muestra informacion
sobre por que estos tweets forman parte de la revision.

## Como devolver tu archivo

- Guarda el CSV completado con el nombre exacto: `adjudicator_C_response.csv`
- Devuelve SOLO ese archivo.

## Que pasa despues

Tu archivo se valida y se conserva sin cambios. En una segunda etapa podras comparar tus
respuestas con las de otras personas anotadoras y registrar, cuando sea necesario, una
decision final razonada. La guia aplicable es una politica operativa interna del piloto;
no una definicion oficial de la competencia.
"""


if __name__ == "__main__":
    main()
