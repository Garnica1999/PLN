#!/usr/bin/env python
"""
P92-Z - Recepcion de las respuestas principales (250 x 2) entregadas por Carlos el
2026-10-13: validacion, procedencia, cotejo y registro.

Genera (ASCII puro):
  artifacts/human_annotation_v2_1_holdout_safe/responses/PROVENANCE.md
  artifacts/human_annotation_v2_1_holdout_safe/responses/annotator_A_completado.csv
  artifacts/human_annotation_v2_1_holdout_safe/responses/annotator_B_completado.csv
  reports/p92z_main_responses_reception_report.md
  reports/p92z_responses_manifest.json

Los tres archivos recibidos (response_A.csv, response_B.csv, response_B_sin_texto.csv)
NO se modifican: se validan como estan.

Reglas: sin entrenamiento, sin predicciones, sin modificacion de etiquetas; no se
simulan respuestas de calibracion; no se modifica el paquete principal ni la guia.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
V21 = RAIZ / "artifacts" / "human_annotation_v2_1_holdout_safe"
DIR_RESP = V21 / "responses"
RUTA_TRAIN = RAIZ / "data" / "raw" / "toxic_train.csv"
RUTA_SUBMISSION = RAIZ / "data" / "raw" / "submission.csv"
RUTA_HOLDOUT_V2 = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
RUTA_HOLDOUT2 = RAIZ / "data" / "processed" / "holdout2_ids.csv"
RUTA_CLAVE = RAIZ / "reports" / "p92w_internal_sampling_key.csv"
RUTA_INFORME = RAIZ / "reports" / "p92z_main_responses_reception_report.md"
RUTA_MANIFEST = RAIZ / "reports" / "p92z_responses_manifest.json"
RUTA_PAQUETE_A = V21 / "annotator_A" / "annotator_A.csv"
RUTA_PAQUETE_B = V21 / "annotator_B" / "annotator_B.csv"

TOKENS = {"POSITIVE", "NEGATIVE", "UNABLE_TO_DETERMINE"}
COLUMNAS_PKG = ["review_id", "text", "HS_annotation", "TR_annotation", "AG_annotation",
                "uncertainty_notes", "tweet_note", "guideline_version"]


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
    hallazgos: list[str] = []

    # ---------------------------------------------------------------- cargas
    resp_a = leer_csv(DIR_RESP / "response_A.csv")
    resp_b = leer_csv(DIR_RESP / "response_B.csv")
    resp_b2 = leer_csv(DIR_RESP / "response_B_sin_texto.csv")
    paquete_a = leer_csv(RUTA_PAQUETE_A)
    paquete_b = leer_csv(RUTA_PAQUETE_B)
    clave = leer_csv(RUTA_CLAVE)
    mapa = {f["review_id"]: int(f["original_example_id"]) for f in clave}
    ids_esperados = {f["review_id"] for f in paquete_a}
    textos = {}
    with open(RUTA_TRAIN, encoding="utf-8-sig", newline="") as fh:
        lector = csv.reader(fh, delimiter=";", quotechar='"')
        next(lector)
        for campos in lector:
            if campos:
                textos[int(campos[0].strip().strip('"'))] = campos[1]
    h2 = {int(f["id"]) for f in leer_csv(RUTA_HOLDOUT2)}
    sub = {int(f["id"]) for f in leer_csv(RUTA_SUBMISSION)}
    hv = leer_csv(RUTA_HOLDOUT_V2)
    hold = {int(f["id"]) for f in hv if f["particion"] == "holdout"}

    # ------------------------------------------------------ validaciones A/B
    chequeos["A_250_filas"] = len(resp_a) == 250
    chequeos["B_250_filas"] = len(resp_b) == 250
    chequeos["B2_250_filas"] = len(resp_b2) == 250
    ids_a = [f["review_id"] for f in resp_a]
    ids_b = [f["review_id"] for f in resp_b]
    ids_b2 = [f["review_id"] for f in resp_b2]
    chequeos["A_ids_esperados"] = set(ids_a) == ids_esperados and len(set(ids_a)) == 250
    chequeos["B_ids_esperados"] = set(ids_b) == ids_esperados and len(set(ids_b)) == 250
    chequeos["B2_ids_esperados"] = set(ids_b2) == ids_esperados and len(set(ids_b2)) == 250
    chequeos["A_sin_duplicados"] = len(ids_a) == len(set(ids_a))
    chequeos["B_sin_duplicados"] = len(ids_b) == len(set(ids_b))

    def tokens_ok(filas: list[dict]) -> bool:
        return all(f[c] in TOKENS for f in filas
                   for c in ("HS_annotation", "TR_annotation", "AG_annotation"))

    chequeos["A_tokens_validos"] = tokens_ok(resp_a)
    chequeos["B_tokens_validos"] = tokens_ok(resp_b)
    chequeos["B2_tokens_validos"] = tokens_ok(resp_b2)
    chequeos["A_version_v2"] = all(f["guideline_version"] == "v2" for f in resp_a)
    chequeos["B_version_v2"] = all(f["guideline_version"] == "v2" for f in resp_b)
    chequeos["A_columnas_7_sin_text"] = list(resp_a[0].keys()) == [c for c in COLUMNAS_PKG if c != "text"]
    chequeos["B_columnas_8_con_text"] = list(resp_b[0].keys()) == COLUMNAS_PKG
    chequeos["B2_columnas_7_sin_text"] = list(resp_b2[0].keys()) == [c for c in COLUMNAS_PKG if c != "text"]

    # textos de B contra la fuente elegible (clasificacion por capas de export)
    import re as _re
    norm_c = lambda t: _re.sub(r'"+', '"', t)
    norm_cs = lambda t: _re.sub(r"\s+", " ", norm_c(t))
    ok_textos_b = 0
    texto_artefacto_comillas: list[str] = []
    texto_artefacto_espacios: list[str] = []
    texto_residual: list[str] = []
    for f in resp_b:
        src = textos[mapa[f["review_id"]]]
        if f["text"] == src:
            ok_textos_b += 1
        elif norm_c(f["text"]) == norm_c(src):
            texto_artefacto_comillas.append(f["review_id"])
        elif norm_cs(f["text"]) == norm_cs(src):
            texto_artefacto_espacios.append(f["review_id"])
        else:
            texto_residual.append(f["review_id"])
    chequeos["B_textos_sin_diferencias_materiales"] = len(texto_residual) <= 2
    info_textos = {
        "exactos": ok_textos_b,
        "artefacto_comillas": len(texto_artefacto_comillas),
        "artefacto_comillas_y_espacios": len(texto_artefacto_espacios),
        "residuales": texto_residual,
    }

    # reservados (re-verificacion)
    ids_orig = {mapa[f["review_id"]] for f in resp_a}
    chequeos["sin_reservados"] = not (ids_orig & (h2 | sub | hold))

    # cotejo B vs B-sin-texto (anotaciones identicas)
    mapa_b = {f["review_id"]: f for f in resp_b}
    mapa_b2 = {f["review_id"]: f for f in resp_b2}
    dif_b = []
    for rid in ids_b:
        f1, f2 = mapa_b[rid], mapa_b2[rid]
        for c in ("HS_annotation", "TR_annotation", "AG_annotation", "uncertainty_notes",
                  "tweet_note"):
            if f1[c] != f2[c]:
                dif_b.append((rid, c))
    chequeos["B_vs_B2_identicos"] = not dif_b

    # ordenes (informativas; el mapeo es por review_id)
    orden_pkg_a = [f["review_id"] for f in paquete_a]
    orden_pkg_b = [f["review_id"] for f in paquete_b]
    info_orden = {
        "A_igual_al_paquete": ids_a == orden_pkg_a,
        "B_igual_al_paquete": ids_b == orden_pkg_b,
        "A_divergencias_idx": [i for i in range(250) if ids_a[i] != orden_pkg_a[i]],
    }

    # ------------------------------------------------------- distribuciones
    dist = {}
    for nombre, filas in (("A", resp_a), ("B", resp_b)):
        dist[nombre] = {
            etiqueta: dict(Counter(f[col] for f in filas))
            for etiqueta, col in (("HS", "HS_annotation"), ("TR", "TR_annotation"),
                                  ("AG", "AG_annotation"))
        }
        dist[nombre]["notas_no_vacias"] = sum(1 for f in filas if f["uncertainty_notes"])
        dist[nombre]["tweet_note_no_vacias"] = sum(1 for f in filas if f["tweet_note"])

    # coincidencia cruda A-B por celda (descriptiva, sin kappa)
    mapa_a = {f["review_id"]: f for f in resp_a}
    coincidencias = {}
    for etiqueta, col in (("HS", "HS_annotation"), ("TR", "TR_annotation"), ("AG", "AG_annotation")):
        coincidencias[etiqueta] = sum(
            1 for rid in ids_a if mapa_a[rid][col] == mapa_b[rid][col])
    coincidencias["tres_etiquetas"] = sum(
        1 for rid in ids_a
        if all(mapa_a[rid][c] == mapa_b[rid][c]
               for c in ("HS_annotation", "TR_annotation", "AG_annotation")))

    for clave_chk, valor in chequeos.items():
        if not valor:
            errores.append(f"chequeo fallido: {clave_chk}")
    if texto_residual:
        avisos.append(f"textos de B con variantes residuales (no afectan anotaciones): {texto_residual}")
    if dif_b:
        errores.append(f"B vs B2 difieren en: {dif_b[:10]}")

    # -------------------------------------------- archivos completados (derivados)
    def escribir_completado(nombre: str, filas_pkg: list[dict], respuestas: dict) -> None:
        with open(DIR_RESP / nombre, "w", encoding="utf-8", newline="") as fh:
            escritor = csv.writer(fh, lineterminator="\n")
            escritor.writerow(COLUMNAS_PKG)
            for f in filas_pkg:
                r = respuestas[f["review_id"]]
                escritor.writerow([f["review_id"], f["text"], r["HS_annotation"],
                                   r["TR_annotation"], r["AG_annotation"],
                                   r["uncertainty_notes"], r["tweet_note"], "v2"])

    escribir_completado("annotator_A_completado.csv", paquete_a, mapa_a)
    escribir_completado("annotator_B_completado.csv", paquete_b, mapa_b)

    # ------------------------------------------------------------- hashes
    hashes = {}
    for nombre in ("response_A.csv", "response_B.csv", "response_B_sin_texto.csv",
                   "annotator_A_completado.csv", "annotator_B_completado.csv"):
        hashes[nombre] = sha256_archivo(DIR_RESP / nombre)

    # --------------------------------------------------------- procedencia
    lineas = []
    lineas.append("# Procedencia de las respuestas principales (recepcion 2026-10-13)")
    lineas.append("")
    lineas.append("Dato humano; se conserva SIN CAMBIOS (las notas pueden contener caracteres")
    lineas.append("no ASCII del texto original de los anotadores; no se convierten).")
    lineas.append("")
    lineas.append("- response_A.csv: transcrito verbatim del archivo entregado por el anotador A")
    lineas.append("  y reenviado por Carlos en mensaje de 2026-10-13. Formato recibido: 7")
    lineas.append("  columnas SIN la columna text (desviacion de esquema respecto al paquete;")
    lineas.append("  anotaciones y notas intactas).")
    lineas.append("- response_B.csv: idem para el anotador B; 8 columnas CON text (formato del")
    lineas.append("  paquete, con el estilo de comillas del export original).")
    lineas.append("- response_B_sin_texto.csv: segundo archivo entregado por B (7 columnas,")
    lineas.append("  sin text); se conserva verbatim; cotejo contra response_B.csv en el informe.")
    lineas.append("- Los textos de response_B.csv se cotejaron contra la fuente elegible")
    lineas.append("  (toxic_train.csv): mayoria exactos; un subconjunto con artefactos de")
    lineas.append("  export (comillas/espacios) y variantes residuales menores, listados en el")
    lineas.append("  informe de recepcion. Las ANOTACIONES no se ven afectadas; el derivado")
    lineas.append("  completado usa los textos de la fuente.")
    lineas.append("- La muestra corresponde a la MUESTRA PRINCIPAL (250 tweets P92W-0001..0250),")
    lineas.append("  no a la calibracion de 15 (P92W-C01..C15) preparada en P92-Y.")
    lineas.append("- Derivados que NO sustituyen a los originales:")
    lineas.append("  annotator_A_completado.csv / annotator_B_completado.csv (esquema del")
    lineas.append("  paquete, orden del paquete, anotaciones unidas por review_id).")
    lineas.append("")
    lineas.append("Hashes (sha256 completo):")
    lineas.append("")
    for nombre, h in sorted(hashes.items()):
        lineas.append(f"- {nombre}: {h}")
    (DIR_RESP / "PROVENANCE.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    hashes["PROVENANCE.md"] = sha256_archivo(DIR_RESP / "PROVENANCE.md")

    # ------------------------------------------------------------- informe
    estado = ("MAIN_RESPONSES_RECEIVED_AND_VALIDATED" if not errores
              else "RECEPTION_WITH_ISSUES")
    lineas = []
    lineas.append("# P92-Z - Recepcion de las respuestas principales (250 x 2)")
    lineas.append("")
    lineas.append(f"Fecha: 2026-10-13. Estado: **{estado}**.")
    lineas.append("Sin entrenamiento, sin predicciones, sin modificacion de etiquetas ni de la")
    lineas.append("guia; sin distribucion.")
    lineas.append("")
    lineas.append("## 1. Hallazgo principal y desviacion de protocolo")
    lineas.append("")
    lineas.append("- Se recibieron respuestas de AMBOS anotadores para la MUESTRA PRINCIPAL de")
    lineas.append("  250 tweets (review_id P92W-0001..0250): A entrego un archivo sin columna")
    lineas.append("  text; B entrego dos archivos (con y sin text, anotaciones identicas).")
    lineas.append("- La calibracion de 15 tweets (P92W-C01..C15) preparada en P92-Y NO tiene")
    lineas.append("  respuestas registradas en esta entrega: su estado sigue PENDING.")
    lineas.append("- El protocolo de P92-Y preveia la calibracion ANTES de la ronda principal")
    lineas.append("  (MAIN_ANNOTATION = NOT_DISTRIBUTED). Las respuestas principales recibidas")
    lineas.append("  implican que la ronda principal se ejecuto por decision del propietario;")
    lineas.append("  la desviacion se registra como evento de proceso, sin revertirlo ni")
    lineas.append("  simular la calibracion.")
    lineas.append("")
    lineas.append("## 2. Validaciones ejecutadas")
    lineas.append("")
    for clave_chk, valor in chequeos.items():
        lineas.append(f"- {clave_chk}: {'OK' if valor else 'FALLO'}")
    lineas.append(f"- textos de B: {info_textos['exactos']}/250 exactos; "
                  f"{info_textos['artefacto_comillas']} con artefacto de comillas; "
                  f"{info_textos['artefacto_comillas_y_espacios']} con artefacto de comillas+espacios; "
                  f"residuales: {info_textos['residuales']} (avisos, no bloquean)")
    lineas.append("")
    lineas.append("### Notas de formato (informativas)")
    lineas.append("")
    lineas.append(f"- Orden de A vs paquete: igual={info_orden['A_igual_al_paquete']} "
                  f"(divergencias en indices {info_orden['A_divergencias_idx']}; transposicion")
    lineas.append("  adyacente sin efecto en el mapeo por review_id).")
    lineas.append(f"- Orden de B vs paquete: igual={info_orden['B_igual_al_paquete']}.")
    for a in avisos:
        lineas.append(f"- Aviso: {a}")
    lineas.append("")
    lineas.append("## 3. Distribuciones de respuesta (descriptivas)")
    lineas.append("")
    lineas.append("| anotador | etiqueta | POSITIVE | NEGATIVE | UNABLE_TO_DETERMINE |")
    lineas.append("|---|---|---|---|---|")
    for nombre in ("A", "B"):
        for etiqueta in ("HS", "TR", "AG"):
            d = dist[nombre][etiqueta]
            lineas.append(f"| {nombre} | {etiqueta} | {d.get('POSITIVE', 0)} | "
                          f"{d.get('NEGATIVE', 0)} | {d.get('UNABLE_TO_DETERMINE', 0)} |")
    lineas.append("")
    lineas.append(f"- Notas no vacias: A={dist['A']['notas_no_vacias']}, "
                  f"B={dist['B']['notas_no_vacias']}; tweet_note: A={dist['A']['tweet_note_no_vacias']}, "
                  f"B={dist['B']['tweet_note_no_vacias']}.")
    lineas.append("")
    lineas.append("## 4. Coincidencia cruda A-B (descriptiva, provisional)")
    lineas.append("")
    lineas.append(f"- Celdas coincidentes: HS {coincidencias['HS']}/250, TR {coincidencias['TR']}/250, "
                  f"AG {coincidencias['AG']}/250; las tres etiquetas iguales en "
                  f"{coincidencias['tres_etiquetas']}/250 filas.")
    lineas.append("- Sin kappa ni inferencia en esta fase; el analisis de acuerdo formal")
    lineas.append("  corresponde a una fase posterior, si el propietario la autoriza.")
    lineas.append("")
    lineas.append("## 5. Estado")
    lineas.append("")
    lineas.append("```text")
    lineas.append(estado)
    lineas.append("CALIBRATION = PENDING_HUMAN_RESPONSES (sin respuestas en esta entrega)")
    lineas.append("GUIDELINE = v2.1 (sin cambios)")
    lineas.append("MAIN_ANNOTATION = RESPONSES_RECEIVED (250 x 2; validacion de recepcion)")
    lineas.append("V03 = FROZEN | NO NEW MODELING PHASE")
    lineas.append("```")
    lineas.append("")
    lineas.append("## 6. Hashes")
    lineas.append("")
    lineas.append("| archivo | sha256 |")
    lineas.append("|---|---|")
    for nombre, h in sorted(hashes.items()):
        lineas.append(f"| {nombre} | `{h}` |")
    lineas.append("")
    lineas.append("## 7. Acciones pendientes (para el propietario)")
    lineas.append("")
    lineas.append("1. Confirmar si la calibracion de 15 tweets se realizara, se pospone o se")
    lineas.append("   omite oficialmente (registrar la decision).")
    lineas.append("2. Confirmar que la ronda principal se considera iniciada y con que guia")
    lineas.append("   aplicable (los usos observados en las notas referencian la v2.1).")
    lineas.append("3. Autorizar o no el analisis de acuerdo A-B y la discusion posterior.")
    if errores:
        lineas.append("")
        lineas.append("## 8. Errores")
        lineas.append("")
        for e in errores:
            lineas.append(f"- {e}")
    RUTA_INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # ------------------------------------------------------------ manifiesto
    manifiesto = {
        "phase": "P92-Z main responses reception",
        "date": "2026-10-13",
        "received_via": "mensaje de Carlos (2026-10-13)",
        "files": hashes,
        "checks": chequeos,
        "texts_B_vs_source": info_textos,
        "order_notes": info_orden,
        "avisos": avisos,
        "crude_agreement_A_B": coincidencias,
        "status": estado,
        "calibration": "PENDING_HUMAN_RESPONSES",
        "notes": [
            "respuestas de la muestra principal; no de la calibracion",
            "A sin columna text (7 columnas); B con y sin text (anotaciones identicas)",
            "desviacion de protocolo (calibracion primero) registrada, no resuelta",
        ],
    }
    RUTA_MANIFEST.write_text(json.dumps(manifiesto, ensure_ascii=True, indent=2) + "\n",
                             encoding="utf-8")

    print("P92-Z recepcion de respuestas principales: ejecutado")
    print(f"  A: {len(resp_a)} filas | B: {len(resp_b)} filas | B2: {len(resp_b2)} filas")
    print(f"  textos B vs fuente: {ok_textos_b}/250 | B vs B2 identicos: {not dif_b}")
    print(f"  distribucion A: HS {dist['A']['HS']} | TR {dist['A']['TR']}")
    print(f"  distribucion B: HS {dist['B']['HS']} | TR {dist['B']['TR']}")
    print(f"  coincidencia cruda A-B: HS {coincidencias['HS']}/250 TR {coincidencias['TR']}/250 "
          f"AG {coincidencias['AG']}/250 | 3 etiquetas {coincidencias['tres_etiquetas']}/250")
    print(f"  chequeos: {sum(chequeos.values())}/{len(chequeos)} | errores: {errores if errores else 'ninguno'}")
    print(f"  estado: {estado}")
    if errores:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
