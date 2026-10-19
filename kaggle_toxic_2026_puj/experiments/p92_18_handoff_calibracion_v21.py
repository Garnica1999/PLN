#!/usr/bin/env python
"""
P92-Y - Entregas de calibracion holdout-safe con guia v2.1 (sin distribuir).

Genera (ASCII puro):
  artifacts/human_annotation_v2_1_holdout_safe/calibration_handoff/annotator_A/
      (calibration_sample.csv reordenado + README_annotator.md + annotation_guidelines_v2_1.md)
  artifacts/human_annotation_v2_1_holdout_safe/calibration_handoff/annotator_B/ (idem)
  artifacts/human_annotation_v2_1_holdout_safe/calibration_handoff/README.md (interno)
  reports/p92y_calibration_handoff_manifest.json
  reports/p92y_calibration_handoff_report.md

Reglas: no se re-seleccionan tweets; se reutiliza la calibracion existente; no se
modifica el paquete principal ni su manifiesto; sin entrenamiento/predicciones/
submissions/leaderboard; sin distribucion automatica.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
V21 = RAIZ / "artifacts" / "human_annotation_v2_1_holdout_safe"
CAL = V21 / "calibration"
HANDOFF = V21 / "calibration_handoff"
RUTA_CAL_COMUN = CAL / "calibration_sample.csv"
RUTA_CLAVE = RAIZ / "reports" / "p92w_internal_sampling_key.csv"
RUTA_TRAIN = RAIZ / "data" / "raw" / "toxic_train.csv"
RUTA_SUBMISSION = RAIZ / "data" / "raw" / "submission.csv"
RUTA_HOLDOUT_V2 = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
RUTA_HOLDOUT2 = RAIZ / "data" / "processed" / "holdout2_ids.csv"
RUTA_GUIA = RAIZ / "reports" / "p92_annotation_guidelines_v2_1_holdout_safe.md"
RUTA_GATE_X = RAIZ / "reports" / "p92x_final_distribution_gate.md"
RUTA_MANIFEST = RAIZ / "reports" / "p92y_calibration_handoff_manifest.json"
RUTA_INFORME = RAIZ / "reports" / "p92y_calibration_handoff_report.md"

FECHA = "2026-10-13"
POLICY_ID = "PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2"
NOMBRE_GUIA = "annotation_guidelines_v2_1.md"
COLUMNAS = ["review_id", "text", "HS_annotation", "TR_annotation", "AG_annotation",
            "uncertainty_notes", "tweet_note", "guideline_version"]
SEMILLAS_ORDEN = {"A": 20261024, "B": 20261025}
HASH_GUIA_INTERNO_ESPERADO = "ff699f54a48da779"
HASH_GUIA_COMPLETO_ESPERADO = "fd8d4c65ccd194fb"
HASH_MANIFIESTO_PKG_ESPERADO = "00998e6006035c9f"
TOKENS_FUGA = ["p92r-", "p92c-", "holdout", "original_example_id", "sample_component",
               "selection_reason", "dup_group", "source_partition", "oof", "v03"]


def sha256_archivo(ruta: Path) -> str:
    """sha256 completo de un archivo (lectura en bloques)."""
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def leer_csv(ruta: Path, delimiter: str | None = None) -> list[dict]:
    """Lee un CSV detectando delimitador si no se especifica."""
    with open(ruta, encoding="utf-8-sig", newline="") as fh:
        if delimiter is None:
            primera = fh.readline()
            delimiter = ";" if primera.count(";") > primera.count(",") else ","
            fh.seek(0)
        return list(csv.DictReader(fh, delimiter=delimiter))


def construir_readme(letra: str) -> str:
    """README de la ronda de calibracion para el anotador/a A o B."""
    return f"""# Calibracion de anotacion - ronda de familiarizacion (anotador/a {letra})

Este paquete es autosuficiente. Contiene tres archivos:
1. README_annotator.md (este archivo)
2. {NOMBRE_GUIA} (manual operativo v2.1 de la politica {POLICY_ID})
3. calibration_sample.csv (15 textos; anotaciones vacias)

## Objetivo

Familiarizarte con el manual antes de la tarea principal y ayudar a evaluar la claridad
de las instrucciones. La ronda es independiente: no comentes los casos con la otra
persona anotadora hasta que AMBAS hayan entregado sus archivos.

## Que debes hacer

1. Lee primero `{NOMBRE_GUIA}` completo.
2. Anota los 15 textos de forma independiente y completa.
3. Decide HS, TR y AG por separado.
4. Usa EXCLUSIVAMENTE `POSITIVE`, `NEGATIVE` o `UNABLE_TO_DETERMINE` (los valores 0/1
   NO son validos).
5. Usa `UNABLE_TO_DETERMINE` cuando las reglas no permitan una respuesta responsable y
   explica la incertidumbre en `uncertainty_notes` (motivo obligatorio).
6. `tweet_note`: solo si la dificultad afecta a TODO el tweet; no sustituye las
   respuestas por etiqueta.
7. No consultes etiquetas originales, modelos, predicciones ni puntuaciones; no busques
   los tweets en internet; no uses herramientas automaticas.
8. No cambies `review_id`, `text`, el orden de las filas ni `guideline_version`.

## Como devolver tu archivo

- Guarda el CSV completado con el nombre exacto: `calibration_response_{letra}.csv`
- Devuelve SOLO tu archivo, por separado.
- No acuerdes respuestas con la otra persona antes de devolver tu archivo.

## Que pasa despues (protocolo de recepcion)

Tu archivo se validara (esquema, 15 filas, review_id, texto sin cambios, respuestas
permitidas, guideline_version); tus respuestas originales se almacenan sin cambios. La
ronda sirve para evaluar la claridad de las instrucciones: no es una estimacion
poblacional de errores de anotacion ni forma parte de una metrica del dataset. Los
desacuerdos y dudas se documentan en la etapa posterior.

## Version

Manual aplicable: revision v2.1 (revision documental de procedencia; las reglas no
cambian respecto a la politica v2). Es una politica operativa interna del piloto; no
una definicion oficial del organizador ni una resolucion normativa definitiva.
"""


def main() -> None:
    errores: list[str] = []
    chequeos: dict[str, bool] = {}

    # ---------------------------------------------------------------- cargas
    cal_comun = leer_csv(RUTA_CAL_COMUN)
    clave = leer_csv(RUTA_CLAVE)
    filas_cal = [f for f in clave if f["sample_component"] == "CALIBRACION2"]
    mapa_cal = {f["review_id"]: int(f["original_example_id"]) for f in filas_cal}
    comp_a = {int(f["original_example_id"]) for f in clave if f["sample_component"] == "A"}
    comp_b = {int(f["original_example_id"]) for f in clave if f["sample_component"] == "B"}
    h2 = {int(f["id"]) for f in leer_csv(RUTA_HOLDOUT2)}
    sub = {int(f["id"]) for f in leer_csv(RUTA_SUBMISSION)}
    hv = leer_csv(RUTA_HOLDOUT_V2)
    dev = {int(f["id"]) for f in hv if f["particion"] == "dev"}
    hold = {int(f["id"]) for f in hv if f["particion"] == "holdout"}
    reservados = h2 | sub | hold
    dev2 = dev - h2
    textos: dict[int, str] = {}
    with open(RUTA_TRAIN, encoding="utf-8-sig", newline="") as fh:
        lector = csv.reader(fh, delimiter=";", quotechar='"')
        next(lector)
        for campos in lector:
            if campos:
                textos[int(campos[0].strip().strip('"'))] = campos[1]

    guia_bytes = RUTA_GUIA.read_bytes()
    lineas_guia = [l for l in guia_bytes.decode("utf-8").splitlines(keepends=True)
                   if not l.startswith("SHA256 (contenido sin la linea")]
    hash_guia_interno = hashlib.sha256("".join(lineas_guia).encode("utf-8")).hexdigest()

    # ------------------------------------- verificacion de la calibracion
    chequeos["cal_15_filas"] = len(cal_comun) == 15
    ids_cal = {f["review_id"] for f in cal_comun}
    chequeos["cal_ids_coinciden_con_clave"] = ids_cal == set(mapa_cal)
    chequeos["cal_anotaciones_vacias"] = all(
        f[c] == "" for f in cal_comun
        for c in ("HS_annotation", "TR_annotation", "AG_annotation",
                  "uncertainty_notes", "tweet_note"))
    chequeos["cal_version_v2"] = all(f["guideline_version"] == "v2" for f in cal_comun)
    ids_originales = {mapa_cal[f["review_id"]] for f in cal_comun}
    chequeos["cal_sin_reservados"] = not (ids_originales & reservados)
    chequeos["cal_en_dev2"] = ids_originales <= dev2
    chequeos["cal_sin_solape_principal"] = not (ids_originales & (comp_a | comp_b))
    ok_textos = sum(1 for f in cal_comun
                    if textos[mapa_cal[f["review_id"]]] == f["text"])
    chequeos["cal_textos_identicos_fuente"] = ok_textos == 15
    chequeos["guia_hash_interno"] = hash_guia_interno.startswith(HASH_GUIA_INTERNO_ESPERADO)
    chequeos["guia_hash_completo"] = sha256_archivo(RUTA_GUIA).startswith(HASH_GUIA_COMPLETO_ESPERADO)
    chequeos["manifiesto_paquete_intacto"] = sha256_archivo(
        V21 / "manifest.json").startswith(HASH_MANIFIESTO_PKG_ESPERADO)
    gate_x = RUTA_GATE_X.read_text(encoding="utf-8")
    chequeos["gate_x_registra_manifiesto"] = HASH_MANIFIESTO_PKG_ESPERADO in gate_x

    # ------------------------------------------------- ordenes nuevas (A/B)
    rng_a = np.random.default_rng(SEMILLAS_ORDEN["A"])
    rng_b = np.random.default_rng(SEMILLAS_ORDEN["B"])
    rids = sorted(ids_cal)
    orden_a = [rids[int(i)] for i in rng_a.permutation(len(rids))]
    orden_b = [rids[int(i)] for i in rng_b.permutation(len(rids))]
    chequeos["ordenes_distintos"] = orden_a != orden_b

    # -------------------------------------------------- escritura del handoff
    HANDOFF.mkdir(parents=True, exist_ok=True)
    por_id = {f["review_id"]: f for f in cal_comun}
    archivos_visibles = []
    for letra, orden in (("A", orden_a), ("B", orden_b)):
        carpeta = HANDOFF / f"annotator_{letra}"
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta_csv = carpeta / "calibration_sample.csv"
        with open(ruta_csv, "w", encoding="utf-8", newline="") as fh:
            escritor = csv.writer(fh, lineterminator="\n")
            escritor.writerow(COLUMNAS)
            for rid in orden:
                f = por_id[rid]
                escritor.writerow([f["review_id"], f["text"], "", "", "", "", "", "v2"])
        (carpeta / NOMBRE_GUIA).write_bytes(guia_bytes)
        (carpeta / "README_annotator.md").write_text(construir_readme(letra), encoding="utf-8")
        archivos_visibles += [ruta_csv, carpeta / NOMBRE_GUIA, carpeta / "README_annotator.md"]

    (HANDOFF / "README.md").write_text(
        """# Entregas de calibracion v2.1 (interno)

Estado: preparadas, NO distribuidas (2026-10-13, P92-Y).

Contenido: `annotator_A/` y `annotator_B/` (los mismos 15 tweets, orden distinto,
respuestas vacias) + este README interno.

- No entregar esta carpeta completa: entregar cada subcarpeta por separado.
- Hashes y verificaciones: `reports/p92y_calibration_handoff_manifest.json` y
  `reports/p92y_calibration_handoff_report.md`.
- El manifiesto del paquete (P92-X) no se modifica; este handoff se documenta aparte.
- La clave interna permanece en `reports/p92w_internal_sampling_key.csv` (fuera de las
  carpetas compartibles).
""", encoding="utf-8")

    # ------------------------------------------------ verificaciones handoff
    csv_a = leer_csv(HANDOFF / "annotator_A" / "calibration_sample.csv")
    csv_b = leer_csv(HANDOFF / "annotator_B" / "calibration_sample.csv")
    chequeos["handoff_15_y_15"] = len(csv_a) == 15 and len(csv_b) == 15
    chequeos["handoff_misma_membresia"] = {f["review_id"] for f in csv_a} == {f["review_id"] for f in csv_b}
    chequeos["handoff_orden_distinto"] = ([f["review_id"] for f in csv_a]
                                          != [f["review_id"] for f in csv_b])
    chequeos["handoff_anotaciones_vacias"] = all(
        f[c] == "" for f in csv_a + csv_b
        for c in ("HS_annotation", "TR_annotation", "AG_annotation",
                  "uncertainty_notes", "tweet_note"))
    ok_textos_handoff = sum(1 for f in csv_a + csv_b
                            if textos[mapa_cal[f["review_id"]]] == f["text"])
    chequeos["handoff_textos_30"] = ok_textos_handoff == 30
    chequeos["handoff_guia_v21_identica"] = all(
        (HANDOFF / f"annotator_{letra}" / NOMBRE_GUIA).read_bytes() == guia_bytes
        for letra in ("A", "B"))

    hallazgos = []
    for ruta in archivos_visibles:
        if ruta.suffix != ".md":
            continue
        texto = ruta.read_text(encoding="utf-8").lower()
        for token in TOKENS_FUGA:
            if token in texto:
                hallazgos.append(f"{ruta.parent.name}/{ruta.name}:{token}")
        if any(str(i) in texto for i in ids_originales):
            hallazgos.append(f"{ruta.parent.name}/{ruta.name}:id_original")
    for ruta in (HANDOFF / "annotator_A" / "calibration_sample.csv",
                 HANDOFF / "annotator_B" / "calibration_sample.csv"):
        for f in leer_csv(ruta):
            if f["review_id"] not in mapa_cal:
                hallazgos.append(f"{ruta.parent.name}/{ruta.name}:review_id")
            for col in COLUMNAS:
                if col in ("text", "review_id"):
                    continue
                if f[col] not in ("", "v2"):
                    hallazgos.append(f"{ruta.parent.name}/{ruta.name}:{col}={f[col][:12]}")
                if any(str(i) in f[col] for i in ids_originales):
                    hallazgos.append(f"{ruta.parent.name}/{ruta.name}:{col}:id_original")
    chequeos["handoff_blinding_sin_fugas"] = not hallazgos
    if hallazgos:
        errores.append(f"blinding: {hallazgos}")
    chequeos["clave_interna_fuera"] = not str(RUTA_CLAVE).startswith(str(HANDOFF))

    # ------------------------------------------------------------- hashes
    hashes_handoff = {}
    for ruta in archivos_visibles + [HANDOFF / "README.md"]:
        hashes_handoff[str(ruta.relative_to(RAIZ))] = sha256_archivo(ruta)

    # -------------------------------------------------- verificador ASCII
    verif = subprocess.run([sys.executable, str(RAIZ / "tools" / "verificar_caracteres.py")],
                           capture_output=True, text=True)
    chequeos["verificador_ascii"] = verif.returncode == 0
    resumen_verif = (verif.stdout or verif.stderr).strip().splitlines()[-1] if \
        (verif.stdout or verif.stderr).strip() else "sin salida"

    # --------------------------------------------------------- manifiesto
    manifiesto = {
        "pilot": "P92-Y calibration handoff (holdout-safe, guide v2.1)",
        "date": FECHA,
        "policy_id": POLICY_ID,
        "guide": {
            "source": str(RUTA_GUIA.relative_to(RAIZ)),
            "sha256_completo": sha256_archivo(RUTA_GUIA),
            "sha256_interno": hash_guia_interno,
            "copias_en_handoff": NOMBRE_GUIA,
        },
        "calibration_sample": {
            "source": str(RUTA_CAL_COMUN.relative_to(RAIZ)),
            "reuse": "reutilizada sin regenerar; los handoffs son copias reordenadas",
            "tweets": 15,
            "review_ids": sorted(ids_cal),
            "orders_seeds": SEMILLAS_ORDEN,
        },
        "files": hashes_handoff,
        "checks": chequeos,
        "distribution": "NOT_DISTRIBUTED",
        "package_manifest": {
            "path": "artifacts/human_annotation_v2_1_holdout_safe/manifest.json",
            "sha256_prefix": HASH_MANIFIESTO_PKG_ESPERADO,
            "status": "sin cambios (por instruccion de la fase)",
        },
        "limitations": [
            "politica operativa interna; no definicion oficial",
            "la calibracion evalua claridad de la guia; no estima prevalencia",
        ],
    }
    RUTA_MANIFEST.write_text(json.dumps(manifiesto, ensure_ascii=True, indent=2) + "\n",
                             encoding="utf-8")

    # ------------------------------------------------------------- informe
    for clave_chk, valor in chequeos.items():
        if not valor:
            errores.append(f"chequeo fallido: {clave_chk}")
    clasificacion = ("CALIBRATION_V2_1_PACKAGE_READY" if not errores
                     else "ANNOTATION_PACKAGE_BLOCKED")
    lineas = []
    lineas.append("# P92-Y - Informe de entrega: calibracion holdout-safe con guia v2.1")
    lineas.append("")
    lineas.append(f"Fecha: {FECHA}. Estado: **{clasificacion}** + HUMAN_CALIBRATION_PENDING.")
    lineas.append("Sin entrenamiento, sin predicciones, sin submissions, sin leaderboard;")
    lineas.append("sin distribucion automatica; el paquete principal no se modifica.")
    lineas.append("")
    lineas.append("## 1. Preparacion")
    lineas.append("")
    lineas.append("- La calibracion comun ya existia (`calibration/calibration_sample.csv`,")
    lineas.append("  15 tweets, vacia) y se reutilizo sin regenerar; no se re-selecciono nada.")
    lineas.append("- Se crearon las dos copias ciegas en `calibration_handoff/annotator_A/` y")
    lineas.append("  `annotator_B/` con los MISMOS 15 review_id en ORDEN DISTINTO")
    lineas.append(f"  (semillas de orden: A={SEMILLAS_ORDEN['A']}, B={SEMILLAS_ORDEN['B']}).")
    lineas.append("- Cada carpeta incluye: calibration_sample.csv, README_annotator.md y")
    lineas.append(f"  {NOMBRE_GUIA} (copia byte a byte de la guia v2.1).")
    lineas.append("")
    lineas.append("## 2. Controles ejecutados")
    lineas.append("")
    for clave_chk, valor in chequeos.items():
        lineas.append(f"- {clave_chk}: {'OK' if valor else 'FALLO'}")
    lineas.append(f"- textos de la calibracion contra la fuente: {ok_textos}/15")
    lineas.append(f"- textos en los handoffs: {ok_textos_handoff}/30")
    lineas.append("")
    lineas.append("## 3. Elegibilidad y cegamiento")
    lineas.append("")
    lineas.append("- 15/15 ids en dev2 (poblacion elegible); 0 solapes con test, holdout")
    lineas.append("  historico y holdout2 (verificado con metadatos de particion).")
    lineas.append("- 0 solapes con los componentes principales A/B.")
    lineas.append("- Archivos visibles sin etiquetas originales, ids de origen, OOF,")
    lineas.append("  predicciones, thresholds, folds, grupos, motivos ni clave interna.")
    lineas.append("")
    lineas.append("## 4. Guia y hashes")
    lineas.append("")
    lineas.append(f"- Guia v2.1 en ambos handoffs: sha256 completo `{sha256_archivo(RUTA_GUIA)}`;")
    lineas.append(f"  sha256 interno (sin linea de hash) `{hash_guia_interno}`.")
    lineas.append("")
    lineas.append("| archivo | sha256 |")
    lineas.append("|---|---|")
    for ruta_txt, h in sorted(hashes_handoff.items()):
        lineas.append(f"| {ruta_txt} | `{h}` |")
    lineas.append("")
    lineas.append("## 5. Instrucciones exactas para la entrega")
    lineas.append("")
    lineas.append("1. Entregar a la persona A la carpeta COMPLETA")
    lineas.append("   `artifacts/human_annotation_v2_1_holdout_safe/calibration_handoff/annotator_A/`")
    lineas.append("   (3 archivos) y a la persona B la carpeta `annotator_B/` (3 archivos).")
    lineas.append("2. Por separado, sin comunicacion entre ambas hasta que las DOS hayan")
    lineas.append("   entregado sus archivos.")
    lineas.append("3. Cada persona completa los 15 tweets y devuelve SOLO su CSV renombrado")
    lineas.append("   como `calibration_response_A.csv` / `calibration_response_B.csv`.")
    lineas.append("4. No completar ni simular respuestas; las columnas de anotacion se llenan")
    lineas.append("   unicamente por las personas.")
    lineas.append("5. Al recibir ambas respuestas: validar esquema (15 filas, tokens")
    lineas.append("   permitidos), almacenar verbatim en `calibration_responses/` de esta")
    lineas.append("   ronda y solo entonces abrir la discusion.")
    lineas.append("6. La guia v2.1 es politica operativa interna: no presentarla como")
    lineas.append("   resolucion normativa oficial; la ronda evalua claridad, no prevalencia.")
    lineas.append("")
    lineas.append("## 6. Estado")
    lineas.append("")
    lineas.append("```text")
    lineas.append(clasificacion)
    lineas.append("HUMAN_CALIBRATION_PENDING")
    lineas.append("MAIN_ANNOTATION_NOT_DISTRIBUTED")
    lineas.append("V03 = FROZEN")
    lineas.append("NO NEW MODELING PHASE")
    lineas.append("```")
    lineas.append("")
    lineas.append(f"- Verificador ASCII oficial: {resumen_verif}")
    lineas.append("- La muestra principal de 250 tweets seguira sin distribuirse hasta recibir")
    lineas.append("  ambas calibraciones y decidir si la guia puede mantenerse congelada.")
    if errores:
        lineas.append("")
        lineas.append("## 7. Errores")
        lineas.append("")
        for e in errores:
            lineas.append(f"- {e}")
    RUTA_INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- resumen
    print("P92-Y handoff de calibracion v2.1: ejecutado")
    print(f"  handoffs: A y B con 15 tweets (ordenes distintos, semillas {SEMILLAS_ORDEN})")
    print(f"  textos: calibracion {ok_textos}/15 | handoffs {ok_textos_handoff}/30")
    print(f"  verificador ASCII: {resumen_verif}")
    print(f"  chequeos: {sum(chequeos.values())}/{len(chequeos)} | errores: {errores if errores else 'ninguno'}")
    print(f"  clasificacion: {clasificacion}")
    if errores:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
