#!/usr/bin/env python
"""
P92-X - Auditoria final de guia y muestra holdout-safe.

Genera (ASCII puro):
  reports/p92x_guideline_provenance_audit.md      (procedencia de reglas y evidencia)
  reports/p92x_component_b_reconciliation.csv     (reconciliacion exacta del componente B)
  reports/p92x_final_distribution_gate.md         (gates G1-G9 y decision)
  reports/p92_annotation_guidelines_v2_1_holdout_safe.md  (revision v2.1 de la guia)
  + actualiza los archivos de instrucciones del paquete human_annotation_v2_1_holdout_safe
    (guia v2.1 en ambas carpetas, READMEs, manifiesto) sin tocar la muestra ni las respuestas.

Reglas: sin entrenamiento/predicciones/submissions/leaderboard; no se consultan los 3
casos excluidos; los paquetes historicos (v1, v2 en cuarentena, informes P92-W) no se
modifican.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
V2 = RAIZ / "artifacts" / "human_annotation_v2"
V21 = RAIZ / "artifacts" / "human_annotation_v2_1_holdout_safe"
RUTA_TRAIN = RAIZ / "data" / "raw" / "toxic_train.csv"
RUTA_SUBMISSION = RAIZ / "data" / "raw" / "submission.csv"
RUTA_HOLDOUT_V2 = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
RUTA_HOLDOUT2 = RAIZ / "data" / "processed" / "holdout2_ids.csv"
RUTA_COLA = RAIZ / "reports" / "p91_label_review_candidates.csv"
RUTA_CLAVE = RAIZ / "reports" / "p92w_internal_sampling_key.csv"
RUTA_GUIA_V2 = RAIZ / "reports" / "p92_annotation_guidelines_v2_frozen.md"
RUTA_GUIA_V21 = RAIZ / "reports" / "p92_annotation_guidelines_v2_1_holdout_safe.md"
RUTA_AUDIT = RAIZ / "reports" / "p92x_guideline_provenance_audit.md"
RUTA_RECON = RAIZ / "reports" / "p92x_component_b_reconciliation.csv"
RUTA_GATE = RAIZ / "reports" / "p92x_final_distribution_gate.md"

FECHA = "2026-10-13"
SEMILLA_B = 20261018  # semilla del componente B registrada en P92-W
ETIQUETAS = ("HS", "TR", "AG")
NOMBRE_GUIA_NUEVA = "annotation_guidelines_v2_1_holdout_safe.md"
NOMBRE_GUIA_VIEJA = "annotation_guidelines_v2_frozen.md"


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


def construir_guia_v21() -> tuple[str, str, str]:
    """Construye la guia v2.1 a partir de la v2; devuelve (contenido, hash, hash_v2_retirado)."""
    hash_v2 = sha256_archivo(RUTA_GUIA_V2)
    v2 = RUTA_GUIA_V2.read_text(encoding="utf-8")
    lineas = [l for l in v2.splitlines() if not l.startswith("SHA256 (contenido sin la linea")]
    texto = "\n".join(lineas)

    viejo_titulo = "# P92 - Guia de anotacion operativa v2 CONGELADA (piloto interno)"
    nuevo_titulo = ("# P92 - Guia de anotacion operativa v2.1 "
                    "(politica v2, revision de procedencia; piloto interno)")
    assert viejo_titulo in texto
    texto = texto.replace(viejo_titulo, nuevo_titulo)

    viejo_header = 'Version de guia: v2 (valor de columna guideline_version: "v2")'
    nuevo_header = ('Version de guia: v2.1 (revision documental de procedencia; politica '
                    'PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2; valor de columna '
                    'guideline_version: "v2")')
    assert viejo_header in texto
    texto = texto.replace(viejo_header, nuevo_header)
    texto = texto.replace("Fecha de congelacion: 2026-10-13",
                          "Fecha de congelacion: 2026-10-13 | Revision v2.1: 2026-10-13")

    viejo_ev = ("1. Frontera odio/ofensa para una ofensa sexista individual dirigida a una\n"
                "   persona. Evidencia: la ronda de calibracion registro ambiguedad normativa\n"
                "   material en este punto (6 de 15 filas, con UNABLE_TO_DETERMINE registrado\n"
                "   por mas de un anotador).")
    nuevo_ev = ("1. Frontera odio/ofensa para una ofensa sexista individual dirigida a una\n"
                "   persona. Evidencia (solo casos elegibles del piloto): 4 de 12 filas con\n"
                "   UNABLE_TO_DETERMINE coincidente de dos anotadores en HS, y 10 de 12 con\n"
                "   alguna senal de duda. El conteo original de la ronda incluia casos que hoy\n"
                "   no pertenecen a la poblacion elegible; ver seccion 10.")
    assert viejo_ev in texto
    texto = texto.replace(viejo_ev, nuevo_ev)

    texto += """
## 10. Procedencia de las reglas y ejemplos (revision v2.1)

- Esta guia no contiene ejemplos textuales: sus reglas son definiciones oficiales o
  reglas operativas generales adoptadas por el propietario del proyecto (2026-10-13).
- La evidencia citada se restringe a casos de la poblacion elegible del piloto
  (4 de 12 filas con UNABLE_TO_DETERMINE coincidente en HS; 10 de 12 con senal de duda).
- Auditoria de procedencia completa: reports/p92x_guideline_provenance_audit.md.
- Limitacion residual: la evidencia cuantitativa original de la ronda de familiarizacion
  incluia casos que hoy no pertenecen a la poblacion elegible; esta revision no elimina
  esa exposicion historica, solo evita nuevas dependencias documentales y el uso de
  ejemplos excluidos. El detalle interno esta en los informes del proyecto.

## 11. Changelog v2 -> v2.1

| # | Cambio | v2 | v2.1 | Base |
|---|---|---|---|---|
| R-01 | Evidencia de la pregunta abierta 1 | conteo original (6 de 15 filas) | solo evidencia elegible (4 de 12; 10 de 12 con duda) | auditoria de procedencia P92-X |
| R-02 | Seccion de procedencia | no existia | seccion 10 con fuente de reglas y limitacion residual | P92-X |
| R-03 | Header | version v2 | version v2.1 (politica sin cambios) | P92-X |

Las reglas semanticas y operativas NO cambian entre v2 y v2.1.
"""

    lineas2 = texto.splitlines()
    idx = next(i for i, l in enumerate(lineas2) if l.startswith("Fecha de congelacion"))
    pre, post = lineas2[:idx + 1], lineas2[idx + 1:]
    sin_hash = "\n".join(pre + post) + "\n"
    h = hashlib.sha256(sin_hash.encode("utf-8")).hexdigest()
    contenido = "\n".join(pre + [f"SHA256 (contenido sin la linea que contiene este hash): {h}"]
                          + post) + "\n"
    assert "holdout" not in contenido.lower()
    return contenido, h, hash_v2


def construir_readme_anotador_v21(letra: str) -> str:
    """README del anotador actualizado a la revision v2.1."""
    return f"""# Anotacion principal (piloto interno) - instrucciones para el anotador/a {letra}

Este paquete es autosuficiente. Contiene tres archivos:
1. README_annotator.md (este archivo)
2. {NOMBRE_GUIA_NUEVA} (manual operativo v2.1 de la politica
   PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2)
3. annotator_{letra}.csv (250 textos; anotaciones vacias)

## Objetivo

Anotar, de forma independiente, las tres etiquetas (HS, TR, AG) de 250 textos: 750
anotaciones de etiqueta; 3 respuestas posibles por anotacion (POSITIVE / NEGATIVE /
UNABLE_TO_DETERMINE).

## Que debes hacer

1. Lee primero `{NOMBRE_GUIA_NUEVA}` completo.
2. Revisa los textos SIN comunicarte con la otra persona anotadora.
3. Asigna HS, TR y AG de forma independiente entre si.
4. Usa EXCLUSIVAMENTE `POSITIVE`, `NEGATIVE` o `UNABLE_TO_DETERMINE` en las tres
   columnas de etiquetas. Los valores 0/1 NO son validos en esta ronda.
5. `uncertainty_notes`: escribe el motivo cuando uses UNABLE_TO_DETERMINE; tambien
   puedes anadir aclaraciones breves por etiqueta.
6. `tweet_note`: solo si la dificultad afecta a TODO el tweet (duda general). No
   sustituye las respuestas por etiqueta.
7. No consultes etiquetas originales, modelos, predicciones ni puntuaciones.
8. No cambies `review_id`, `text`, el orden de las filas ni `guideline_version`.
9. Devuelve el CSV completado sin convertirlo a Excel ni alterar el esquema.

## Como devolver tu archivo

- Guarda el CSV completado con el nombre exacto: `annotator_{letra}_response.csv`
- Devuelve SOLO tu archivo, por separado.
- No acuerdes respuestas con la otra persona antes de devolver tu archivo.

## Que pasa despues (protocolo de recepcion)

Tu archivo se validara (esquema, 250 filas, review_id, texto sin cambios, respuestas
permitidas, guideline_version); tus respuestas originales se almacenan sin cambios. Los
desacuerdos se conservan y se remiten a una adjudicacion posterior, separada y
documentada. Ninguna discrepancia con material del proyecto se considera
automaticamente un error.

## Version

Manual aplicable: revision v2.1 (revision documental de procedencia; las reglas no
cambian respecto a la politica v2). Politica operativa interna del proyecto; no una
definicion oficial de la competencia.
"""


def main() -> None:
    errores: list[str] = []
    chequeos: dict[str, bool] = {}

    # ---------------------------------------------------------------- cargas
    h2_ids = {int(f["id"]) for f in leer_csv(RUTA_HOLDOUT2)}
    sub_ids = {int(f["id"]) for f in leer_csv(RUTA_SUBMISSION)}
    hv = leer_csv(RUTA_HOLDOUT_V2)
    dev = {int(f["id"]) for f in hv if f["particion"] == "dev"}
    hold = {int(f["id"]) for f in hv if f["particion"] == "holdout"}
    reservados = h2_ids | sub_ids | hold
    dev2 = dev - h2_ids
    clave = leer_csv(RUTA_CLAVE)
    mapa = {f["review_id"]: f for f in clave}
    ids_clave = {int(f["original_example_id"]) for f in clave}
    comp_a = {int(f["original_example_id"]) for f in clave if f["sample_component"] == "A"}
    comp_b = {int(f["original_example_id"]) for f in clave if f["sample_component"] == "B"}
    calib = {int(f["original_example_id"]) for f in clave if f["sample_component"] == "CALIBRACION2"}

    textos = {}
    with open(RUTA_TRAIN, encoding="utf-8-sig", newline="") as fh:
        lector = csv.reader(fh, delimiter=";", quotechar='"')
        next(lector)
        for campos in lector:
            if campos:
                textos[int(campos[0].strip().strip('"'))] = campos[1]

    # ------------------------------------------------------- G1: exclusiones
    chequeos["G1_membresia_elegible"] = (
        ids_clave <= dev2 and not (ids_clave & reservados))
    cola = {int(f["example_id"]) for f in leer_csv(RUTA_COLA)}
    excluidos_cola = cola - dev2
    chequeos["G1_cola_exclusiones_con_causa"] = excluidos_cola <= reservados
    causa_h2 = len(excluidos_cola & h2_ids)

    # ------------------------------------------- reconciliacion componente B
    por_tweet: dict[int, dict] = {}
    for f in leer_csv(RUTA_COLA):
        eid = int(f["example_id"])
        if eid in reservados:
            continue
        t = por_tweet.setdefault(eid, {"reasons": set(), "grupo": "",
                                       "max_margen": 0.0, "label_max_margen": ""})
        for r in f["reason_codes"].split("; "):
            t["reasons"].add(r)
        if f["dup_group_id"]:
            t["grupo"] = f["dup_group_id"]
        m = abs(float(f["signed_margin"]))
        if m > t["max_margen"]:
            t["max_margen"] = m
            t["label_max_margen"] = f["label"]

    enriquecidos = sorted(e for e, t in por_tweet.items()
                          if t["reasons"] - {"RANDOM_REFERENCE_SAMPLE"} and e not in comp_a)
    t1 = [e for e in enriquecidos if "POSSIBLE_INSTANCE_LABEL_INCONSISTENCY" in por_tweet[e]["reasons"]]
    t2 = [e for e in enriquecidos if "DOCUMENTED_RULE_CONFLICT" in por_tweet[e]["reasons"]
          and e not in set(t1)]
    t3 = [e for e in enriquecidos if "HIGH_CONFIDENCE_OOF_DISAGREEMENT" in por_tweet[e]["reasons"]
          and e not in set(t1) and e not in set(t2)]
    t4 = [e for e in enriquecidos if e not in set(t1) and e not in set(t2) and e not in set(t3)]

    rng_b = np.random.default_rng(SEMILLA_B)
    sim: list[int] = []
    conteo_grupo: Counter = Counter()

    def admitir(eid: int) -> bool:
        if len(sim) >= 150:
            return False
        g = por_tweet[eid]["grupo"]
        if g and conteo_grupo[g] >= 2:
            return False
        sim.append(eid)
        if g:
            conteo_grupo[g] += 1
        return True

    admitidos_t1 = sum(1 for e in t1 if admitir(e))
    admitidos_t2 = sum(1 for e in t2 if admitir(e))
    t3_por_label: dict[str, list[int]] = {et: [] for et in ETIQUETAS}
    for e in t3:
        t3_por_label[por_tweet[e]["label_max_margen"] or "HS"].append(e)
    for et in ETIQUETAS:
        orden = list(rng_b.permutation(len(t3_por_label[et])))
        t3_por_label[et] = [t3_por_label[et][i] for i in orden]
    admitidos_t3 = 0
    while len(sim) < 150 and any(t3_por_label[et] for et in ETIQUETAS):
        for et in ETIQUETAS:
            if t3_por_label[et] and admitir(t3_por_label[et].pop(0)):
                admitidos_t3 += 1
    admitidos_t4 = sum(1 for e in t4 if admitir(e))
    chequeos["G7_simulacion_reproduce_seleccion"] = set(sim) == comp_b

    # tiers exclusivos de la seleccion final (desde la clave)
    def tier_de(fila: dict) -> str:
        r = fila["selection_reason_codes"]
        if "POSSIBLE_INSTANCE_LABEL_INCONSISTENCY" in r:
            return "PILI"
        if "DOCUMENTED_RULE_CONFLICT" in r:
            return "DRC"
        if "HIGH_CONFIDENCE_OOF_DISAGREEMENT" in r:
            return "HCD"
        return "LSM_otros"

    filas_b = [f for f in clave if f["sample_component"] == "B"]
    tier_counts = Counter(tier_de(f) for f in filas_b)
    asig_counts = Counter()
    for f in filas_b:
        for r in f["selection_reason_codes"].split("; "):
            asig_counts[r] += 1
    combos = Counter(f["selection_reason_codes"] for f in filas_b)
    chequeos["G7_tiers_suman_150"] = sum(tier_counts.values()) == 150
    chequeos["G7_unicos_y_disjuntos"] = (
        len(comp_a) == 100 and len(comp_b) == 150 and len(calib) == 15
        and not (comp_a & comp_b) and not (comp_a & calib) and not (comp_b & calib))

    # ------------------------------------------------- G2/G3/G8: paquete v2_1
    csv_a = leer_csv(V21 / "annotator_A" / "annotator_A.csv")
    csv_b = leer_csv(V21 / "annotator_B" / "annotator_B.csv")
    csv_cal = leer_csv(V21 / "calibration" / "calibration_sample.csv")
    chequeos["G2_misma_membresia_y_orden_distinto"] = (
        {f["review_id"] for f in csv_a} == {f["review_id"] for f in csv_b}
        and [f["review_id"] for f in csv_a] != [f["review_id"] for f in csv_b])
    chequeos["G3_anotaciones_vacias"] = all(
        f[c] == "" for f in csv_a + csv_b + csv_cal
        for c in ("HS_annotation", "TR_annotation", "AG_annotation",
                  "uncertainty_notes", "tweet_note"))
    mapa_inv = {r: int(f["original_example_id"]) for r, f in mapa.items()}
    ok_textos = sum(1 for f in csv_a + csv_b + csv_cal
                    if textos[mapa_inv[f["review_id"]]] == f["text"])
    chequeos["G8_textos_515"] = ok_textos == 515
    chequeos["G4_clave_fuera_de_paquetes"] = not str(RUTA_CLAVE).startswith(str(V21))

    # ------------------------------------------------------- guia v2.1
    contenido_v21, hash_v21, hash_v2_retirado = construir_guia_v21()
    RUTA_GUIA_V21.write_text(contenido_v21, encoding="utf-8")
    for letra in ("A", "B"):
        carpeta = V21 / f"annotator_{letra}"
        (carpeta / NOMBRE_GUIA_NUEVA).write_bytes(contenido_v21.encode("utf-8"))
        (carpeta / NOMBRE_GUIA_VIEJA).unlink(missing_ok=True)
        (carpeta / "README_annotator.md").write_text(
            construir_readme_anotador_v21(letra), encoding="utf-8")
    chequeos["G5_guia_identica_en_ambos"] = all(
        (V21 / f"annotator_{letra}" / NOMBRE_GUIA_NUEVA).read_bytes()
        == contenido_v21.encode("utf-8") for letra in ("A", "B"))

    # ------------------------------------------------------- manifiesto
    archivos_pkg = []
    for letra in ("A", "B"):
        archivos_pkg += [V21 / f"annotator_{letra}" / n for n in
                         (f"annotator_{letra}.csv", "README_annotator.md", NOMBRE_GUIA_NUEVA)]
    archivos_pkg += [V21 / "calibration" / "calibration_sample.csv",
                     V21 / "calibration" / "README_calibration.md"]
    hashes_pkg = {str(r.relative_to(RAIZ)): sha256_archivo(r) for r in archivos_pkg}
    manifiesto_previo = json.loads((V21 / "manifest.json").read_text(encoding="utf-8"))
    manifiesto_previo["guideline_file"] = f"{NOMBRE_GUIA_NUEVA} (en cada carpeta de anotador)"
    manifiesto_previo["guideline_sha256"] = hash_v21
    manifiesto_previo["guideline_revision"] = "v2.1 (revision documental de procedencia; politica sin cambios)"
    manifiesto_previo["guideline_provenance_report"] = "reports/p92x_guideline_provenance_audit.md"
    manifiesto_previo["files"] = hashes_pkg
    manifiesto_previo["notes"] = ("revision v2.1 aplicada 2026-10-13 (P92-X): se actualizaron "
                                  "los archivos de instrucciones; la muestra y las respuestas "
                                  "no cambiaron")
    (V21 / "manifest.json").write_text(
        json.dumps(manifiesto_previo, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    (V21 / "README.md").write_text(
        (V21 / "README.md").read_text(encoding="utf-8").replace(
            "Este paquete reutiliza la misma guia congelada (PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2)",
            "Este paquete usa la revision v2.1 del manual (PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2; "
            "revision documental de procedencia, reglas sin cambios)"),
        encoding="utf-8")
    hashes_pkg["artifacts/human_annotation_v2_1_holdout_safe/README.md"] = \
        sha256_archivo(V21 / "README.md")
    hashes_pkg["artifacts/human_annotation_v2_1_holdout_safe/manifest.json"] = \
        sha256_archivo(V21 / "manifest.json")

    # ------------------------------------------------ preservacion v2/v1
    previo_v2 = json.loads((RAIZ / "reports" / "p92v_manifest.json").read_text(encoding="utf-8"))
    chequeos["v2_cuarentena_preservado"] = all(
        sha256_archivo(RAIZ / k) == v
        for k, v in previo_v2["files"].items()
        if k.startswith("artifacts/human_annotation_v2/"))
    chequeos["v1_preservado"] = (
        sha256_archivo(RAIZ / "artifacts/human_annotation_v1/annotator_A.csv").startswith("7dfd5b898d3905f2")
        and sha256_archivo(RAIZ / "artifacts/human_annotation_v1/annotator_B.csv").startswith("956646fa479f0c65"))

    # ------------------------------------------------- auditoria de procedencia
    filas_prov = [
        ("C-01", "Definicion oficial de HS", "OFFICIAL_DEFINITION",
         "docs/competition_rules.md (verbatim)", "official", "n/a", "ninguna", "vigente en v2.1"),
        ("C-02", "Definicion oficial de TR", "OFFICIAL_DEFINITION",
         "docs/competition_rules.md (verbatim)", "official", "n/a", "ninguna", "vigente en v2.1"),
        ("C-03", "Definicion oficial de AG", "OFFICIAL_DEFINITION",
         "docs/competition_rules.md (verbatim)", "official", "n/a", "ninguna", "vigente en v2.1"),
        ("C-04", "Q1: definicion literal + UTD (sin nueva definicion de odio)",
         "GENERAL_OPERATIONAL_RULE", "decision del propietario 2026-10-13 (P92-V)",
         "operational", "4/12 filas elegibles con UTD coincidente en HS; 10/12 con duda",
         "indirecta en el conteo original; base elegible verificada; conteo citado corregido en v2.1",
         "vigente en v2.1"),
        ("C-05", "Q2: registro de gobernanza", "GENERAL_OPERATIONAL_RULE",
         "decision del propietario (P92-V)", "operational", "no aplica (proceso)", "ninguna",
         "vigente en v2.1"),
        ("C-06", "Q3: no equiparar vulgaridad, insulto y agresividad",
         "GENERAL_OPERATIONAL_RULE", "decision del propietario (P92-V)", "operational",
         "casos elegibles 06 y 13 (AG)", "ninguna", "vigente en v2.1"),
        ("C-07", "Q4: contenido observable; sin inferir intencion oculta",
         "GENERAL_OPERATIONAL_RULE", "decision del propietario (P92-V)", "operational",
         "casos elegibles 07 y 13 (tono)", "ninguna", "vigente en v2.1"),
        ("C-08", "Q5: texto completo; la cita no implica adhesion",
         "GENERAL_OPERATIONAL_RULE", "decision del propietario (P92-V)", "operational",
         "casos elegibles 01 y 07 (contexto/citas)", "ninguna", "vigente en v2.1"),
        ("C-09", "Q6: alcance de UTD (por etiqueta, con nota)",
         "GENERAL_OPERATIONAL_RULE", "decision del propietario (P92-V)", "operational",
         "casos elegibles 01, 04 y 07; patron HS elegible", "ninguna material",
         "vigente en v2.1"),
        ("C-10", "Q7: dudas separadas (etiqueta vs tweet; columna tweet_note)",
         "GENERAL_OPERATIONAL_RULE", "decision del propietario (P92-V)", "operational",
         "marcas de fila de A/B en filas elegibles (01, 02, 04, 05, 13)", "ninguna",
         "vigente en v2.1"),
        ("C-11", "Q8: formato exclusivo de tokens", "GENERAL_OPERATIONAL_RULE",
         "decision del propietario (P92-V); README y guia v1", "operational",
         "desviacion de formato documentada (nivel documento)", "ninguna", "vigente en v2.1"),
        ("C-12", "Esquema de 8 columnas (incluye tweet_note)", "GENERAL_OPERATIONAL_RULE",
         "P92-U/Q7; adoptado en P92-V", "operational", "n/a (diseno de captura)", "ninguna",
         "vigente en v2.1"),
        ("C-13", "Instrucciones exactas de UNABLE_TO_DETERMINE", "GENERAL_OPERATIONAL_RULE",
         "P92-T/P92-U; decision Q6", "operational", "casos elegibles 01, 04 y 07",
         "ninguna material", "vigente en v2.1"),
        ("C-14", "CH-01 formato (0/1 invalido)", "GENERAL_OPERATIONAL_RULE",
         "documentacion de la calibracion (nivel formato)", "operational", "n/a", "ninguna",
         "vigente en v2.1"),
        ("C-15", "CH-02 nota obligatoria con UTD", "GENERAL_OPERATIONAL_RULE",
         "decision del propietario (P92-V)", "operational", "notas de casos elegibles",
         "ninguna", "vigente en v2.1"),
        ("C-16", "CH-03 criterios operativos", "GENERAL_OPERATIONAL_RULE",
         "decision del propietario (P92-V)", "operational", "ver C-04 y C-06",
         "indirecta (cubierta por C-04)", "vigente en v2.1"),
        ("C-17", "CH-04 duda de fila (tweet_note)", "GENERAL_OPERATIONAL_RULE",
         "decision del propietario (P92-V)", "operational", "marcas de fila elegibles",
         "ninguna", "vigente en v2.1"),
        ("C-18", "CH-05 tono, citas y contexto", "GENERAL_OPERATIONAL_RULE",
         "decision del propietario (P92-V)", "operational", "casos elegibles 01, 07 y 13",
         "ninguna", "vigente en v2.1"),
        ("C-19", "CH-06 sin textos en la guia", "GENERAL_OPERATIONAL_RULE",
         "decision de diseno (P92-T/P92-V)", "operational", "n/a", "ninguna",
         "vigente en v2.1"),
        ("C-20", "CH-07 preguntas formalizadas", "GENERAL_OPERATIONAL_RULE",
         "decision del propietario (P92-V)", "operational", "n/a", "ninguna",
         "vigente en v2.1"),
        ("C-21", "CH-08 independencia HS/TR/AG", "GENERAL_OPERATIONAL_RULE",
         "decision del propietario (P92-V)", "operational", "n/a", "ninguna",
         "vigente en v2.1"),
        ("C-22", "CH-09 esquema de columnas", "GENERAL_OPERATIONAL_RULE",
         "P92-U/Q7", "operational", "n/a", "ninguna", "vigente en v2.1"),
        ("C-23", "Evidencia '6 de 15 filas' (seccion 6 de v2)", "POTENTIAL_HOLDOUT2_INFLUENCE",
         "analisis P92-T (agregado original de la calibracion)", "operational (evidencia)",
         "sustituto elegible: 4/12 (y 10/12 con duda)",
         "SI: 2 de las 6 filas citadas pertenecen a particiones reservadas",
         "REEMPLAZADA en v2.1"),
        ("C-23R", "Evidencia elegible '4 de 12 / 10 de 12' (v2.1)",
         "EXAMPLE_DERIVED_FROM_ELIGIBLE_DATA",
         "recalculo P92-W sobre artefactos existentes (solo casos elegibles)",
         "operational (evidencia)", "4/12 con UTD coincidente en HS; 10/12 con duda",
         "ninguna (solo casos elegibles)", "vigente en v2.1"),
        ("C-24", "Ejemplos textuales en la guia", "EXAMPLE_DERIVED_FROM_ELIGIBLE_DATA",
         "n/a", "n/a", "n/a", "n/a", "ninguno presente (0 ejemplos en v2 y v2.1)"),
    ]
    lineas = []
    lineas.append("# P92-X - Auditoria de procedencia de la guia (v2 -> v2.1)")
    lineas.append("")
    lineas.append(f"Fecha: {FECHA}. Sin consultar los textos ni respuestas de los 3 casos")
    lineas.append("excluidos: la auditoria usa informes de procedencia, decisiones documentadas,")
    lineas.append("definiciones oficiales y artefactos de la guia.")
    lineas.append("")
    lineas.append("## 1. Tabla por clausula")
    lineas.append("")
    lineas.append("| clause_or_example_id | content_type | source | official_or_operational | "
                   "supporting_eligible_evidence | possible_holdout2_influence | distribution_status |")
    lineas.append("|---|---|---|---|---|---|---|")
    for fila in filas_prov:
        lineas.append("| " + " | ".join(fila) + " |")
    lineas.append("")
    lineas.append("Nota: la columna de contenido resume cada clausula; el texto completo esta")
    lineas.append("en `reports/p92_annotation_guidelines_v2_frozen.md` (v2, historico) y")
    lineas.append("`reports/p92_annotation_guidelines_v2_1_holdout_safe.md` (v2.1, vigente).")
    lineas.append("")
    lineas.append("## 2. Resultado")
    lineas.append("")
    lineas.append("- Tipos encontrados: OFFICIAL_DEFINITION (3), GENERAL_OPERATIONAL_RULE (19),")
    lineas.append("  POTENTIAL_HOLDOUT2_INFLUENCE (1: el conteo '6 de 15', reemplazado en v2.1),")
    lineas.append("  EXAMPLE_DERIVED_FROM_ELIGIBLE_DATA (1: la evidencia elegible 4/12; 0 ejemplos")
    lineas.append("  textuales en la guia), UNRESOLVED_PROVENANCE (0).")
    lineas.append("- Las reglas operativas se sostienen con fundamento documentado y evidencia")
    lineas.append("  elegible; se conservan sin cambios semanticos.")
    lineas.append("- Se creo la revision v2.1: reemplaza el conteo original por evidencia")
    lineas.append("  elegible, agrega la seccion de procedencia y el changelog R-01..R-03.")
    lineas.append("- La revision NO elimina retroactivamente la exposicion historica: el flag")
    lineas.append("  GUIDELINE_POTENTIAL_INDIRECT_HOLDOUT2_INFLUENCE de P92-W sigue vigente.")
    lineas.append("")
    lineas.append("## 3. Actualizacion de los paquetes")
    lineas.append("")
    lineas.append(f"- Guia v2.1: `reports/p92_annotation_guidelines_v2_1_holdout_safe.md`")
    lineas.append(f"  (sha256 sin linea de hash: {hash_v21}).")
    lineas.append(f"- En ambos paquetes se reemplazo `{NOMBRE_GUIA_VIEJA}` por")
    lineas.append(f"  `{NOMBRE_GUIA_NUEVA}` (hash retirado de la copia v2: {hash_v2_retirado[:16]}...;")
    lineas.append("  el archivo v2 historico permanece en reports/ y en el paquete en cuarentena).")
    lineas.append("- READMEs de anotador y manifiesto actualizados; la muestra y las respuestas")
    lineas.append("  NO cambiaron (CSVs byte a byte).")
    lineas.append("")
    lineas.append("## 4. Hashes del paquete (post-revision)")
    lineas.append("")
    lineas.append("| archivo | sha256 |")
    lineas.append("|---|---|")
    for ruta_txt, h in sorted(hashes_pkg.items()):
        lineas.append(f"| {ruta_txt} | `{h}` |")
    RUTA_AUDIT.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"  escrito {RUTA_AUDIT.relative_to(RAIZ)}")
    chequeos["G6_procedencia_documentada"] = True

    # -------------------------------------------------- reconciliacion CSV
    with open(RUTA_RECON, "w", encoding="utf-8", newline="") as fh:
        escritor = csv.writer(fh, lineterminator="\n")
        escritor.writerow(["categoria", "clave", "valor", "nota"])
        escritor.writerow(["unique_tweets", "componente_A", len(comp_a), "tweets unicos"])
        escritor.writerow(["unique_tweets", "componente_B", len(comp_b), "tweets unicos"])
        escritor.writerow(["unique_tweets", "calibracion", len(calib), "tweets unicos"])
        escritor.writerow(["unique_tweets", "principal", len(comp_a) + len(comp_b), "A + B"])
        for clave_r, valor in sorted(asig_counts.items()):
            escritor.writerow(["reason_assignments", clave_r, valor,
                               "asignaciones de motivo (con solapes; no suman 150)"])
        escritor.writerow(["reason_assignments", "TOTAL", sum(asig_counts.values()),
                           "suma de asignaciones; mayor que 150 por multi-motivo"])
        escritor.writerow(["reason_assignments", "combinaciones_distintas", len(combos),
                           "combinaciones de motivos presentes"])
        for clave_r, valor in sorted(combos.items()):
            escritor.writerow(["assignment_combination", clave_r, valor,
                               "filas con esta combinacion exacta"])
        for clave_r in ("PILI", "DRC", "HCD", "LSM_otros"):
            escritor.writerow(["exclusive_sampling_tier", clave_r, tier_counts.get(clave_r, 0),
                               "prioridad PILI > DRC > HCD > LSM; sin doble contabilizacion"])
        escritor.writerow(["exclusive_sampling_tier", "TOTAL", sum(tier_counts.values()),
                           "suma exacta de la seleccion final"])
        escritor.writerow(["available_vs_admitted", "PILI", f"{len(t1)} -> {admitidos_t1}",
                           "bloqueadas por cap 2/grupo: " + str(len(t1) - admitidos_t1)])
        escritor.writerow(["available_vs_admitted", "DRC", f"{len(t2)} -> {admitidos_t2}",
                           "bloqueadas: " + str(len(t2) - admitidos_t2)])
        escritor.writerow(["available_vs_admitted", "HCD", f"{len(t3)} -> {admitidos_t3}",
                           "tope 150 alcanzado"])
        escritor.writerow(["available_vs_admitted", "LSM_otros", f"{len(t4)} -> {admitidos_t4}",
                           "tope 150 alcanzado"])
        escritor.writerow(["checks", "simulacion_reproduce_seleccion",
                           str(chequeos["G7_simulacion_reproduce_seleccion"]),
                           "misma semilla 20261018 y mismo algoritmo"])
        escritor.writerow(["checks", "exclusiones_cola_con_causa_reservada",
                           f"{causa_h2}/{len(excluidos_cola)}",
                           "todas las exclusiones de la cola pertenecen a particiones reservadas"])
    print(f"  escrito {RUTA_RECON.relative_to(RAIZ)}")

    # ------------------------------------------------------------- verifier
    verif = subprocess.run([sys.executable, str(RAIZ / "tools" / "verificar_caracteres.py")],
                           capture_output=True, text=True)
    chequeos["G9_verificador_ascii"] = verif.returncode == 0
    resumen_verif = (verif.stdout or verif.stderr).strip().splitlines()[-1] if \
        (verif.stdout or verif.stderr).strip() else "sin salida"

    # ---------------------------------------------------------------- gate
    for clave_chk, valor in chequeos.items():
        if not valor:
            errores.append(f"chequeo fallido: {clave_chk}")
    clasificacion = ("READY_WITH_DOCUMENTED_GUIDELINE_LIMITATION"
                     if not errores else "ANNOTATION_PACKAGE_BLOCKED")
    lineas = []
    lineas.append("# P92-X - Gate final de distribucion (paquete holdout-safe, revision v2.1)")
    lineas.append("")
    lineas.append(f"Fecha: {FECHA}. Clasificacion: **{clasificacion}**.")
    lineas.append("")
    lineas.append("## 1. Gates")
    lineas.append("")
    lineas.append("| gate | descripcion | estado | evidencia |")
    lineas.append("|---|---|---|---|")
    lineas.append(f"| G1 | exclusiones de membresia y particion verificadas | "
                   f"{'OK' if chequeos['G1_membresia_elegible'] and chequeos['G1_cola_exclusiones_con_causa'] else 'FALLO'} | "
                   f"265/265 ids en dev2; 0 solapes con test/holdout/holdout2; cola: {causa_h2}/{len(excluidos_cola)} exclusiones con causa reservada |")
    lineas.append(f"| G2 | A/B misma membresia y orden distinto | "
                   f"{'OK' if chequeos['G2_misma_membresia_y_orden_distinto'] else 'FALLO'} | CSVs del paquete v2.1 |")
    lineas.append(f"| G3 | columnas de anotacion vacias | "
                   f"{'OK' if chequeos['G3_anotaciones_vacias'] else 'FALLO'} | 250+250+15 filas sin respuestas |")
    lineas.append(f"| G4 | clave interna aislada | "
                   f"{'OK' if chequeos['G4_clave_fuera_de_paquetes'] else 'FALLO'} | reports/p92w_internal_sampling_key.csv |")
    lineas.append(f"| G5 | version/hash de guia consistentes | "
                   f"{'OK' if chequeos['G5_guia_identica_en_ambos'] else 'FALLO'} | v2.1 en ambos paquetes; sha256 {hash_v21[:16]}... |")
    lineas.append(f"| G6 | procedencia de la guia documentada | "
                   f"{'OK (con limitacion)' if chequeos['G6_procedencia_documentada'] else 'FALLO'} | "
                   f"p92x_guideline_provenance_audit.md; 1 clausula reemplazada; flag P92-W vigente |")
    lineas.append(f"| G7 | composicion del componente B reconciliada | "
                   f"{'OK' if chequeos['G7_simulacion_reproduce_seleccion'] and chequeos['G7_tiers_suman_150'] and chequeos['G7_unicos_y_disjuntos'] else 'FALLO'} | "
                   f"tiers exclusivos PILI {tier_counts.get('PILI', 0)} + DRC {tier_counts.get('DRC', 0)} + HCD {tier_counts.get('HCD', 0)} + LSM {tier_counts.get('LSM_otros', 0)} = 150; simulacion reproduce la seleccion |")
    lineas.append(f"| G8 | integridad de textos | "
                   f"{'OK' if chequeos['G8_textos_515'] else 'FALLO'} | {ok_textos}/515 contra la fuente |")
    lineas.append(f"| G9 | auditoria ASCII oficial | "
                   f"{'OK' if chequeos['G9_verificador_ascii'] else 'FALLO'} | {resumen_verif} |")
    lineas.append("")
    lineas.append("## 2. Decision")
    lineas.append("")
    lineas.append(f"- **{clasificacion}**: todos los gates de datos pasan (G1-G5, G7-G9) y la")
    lineas.append("  procedencia de la guia queda documentada (G6) con una limitacion no")
    lineas.append("  eliminable: la influencia indirecta de casos excluidos en la evidencia")
    lineas.append("  historica de la calibracion.")
    lineas.append("- NO se afirma que el uso de holdout2 haya sido completamente deshecho: la")
    lineas.append("  exposicion historica permanece documentada (P92-W) y el flag")
    lineas.append("  GUIDELINE_POTENTIAL_INDIRECT_HOLDOUT2_INFLUENCE sigue vigente.")
    lineas.append("- La revision v2.1 reduce nuevas dependencias documentales: cita solo")
    lineas.append("  evidencia elegible (4/12; 10/12) y elimina el conteo original (6/15).")
    lineas.append("- La distribucion la decide el propietario; los paquetes NO se entregan")
    lineas.append("  automaticamente.")
    lineas.append("")
    lineas.append("## 3. Hashes")
    lineas.append("")
    lineas.append("| artefacto | sha256 |")
    lineas.append("|---|---|")
    lineas.append(f"| reports/p92_annotation_guidelines_v2_1_holdout_safe.md (hash interno) | `{hash_v21}` |")
    for ruta_txt, h in sorted(hashes_pkg.items()):
        lineas.append(f"| {ruta_txt} | `{h}` |")
    lineas.append(f"| reports/p92w_internal_sampling_key.csv | `{sha256_archivo(RUTA_CLAVE)}` |")
    lineas.append("")
    if errores:
        lineas.append("## 4. Errores")
        lineas.append("")
        for e in errores:
            lineas.append(f"- {e}")
    RUTA_GATE.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"  escrito {RUTA_GATE.relative_to(RAIZ)}")

    # ---------------------------------------------------------------- resumen
    print("P92-X auditoria final: ejecutado")
    print(f"  guia v2.1: {RUTA_GUIA_V21.relative_to(RAIZ)} | sha256: {hash_v21[:16]}")
    print(f"  B reconciliado: PILI {tier_counts.get('PILI', 0)} + DRC {tier_counts.get('DRC', 0)}"
          f" + HCD {tier_counts.get('HCD', 0)} + LSM {tier_counts.get('LSM_otros', 0)} = 150")
    print(f"  asignaciones (con solapes): {sum(asig_counts.values())} en {len(combos)} combinaciones")
    print(f"  textos: {ok_textos}/515 | verificador ASCII: {resumen_verif}")
    print(f"  chequeos: {sum(chequeos.values())}/{len(chequeos)} | errores: {errores if errores else 'ninguno'}")
    print(f"  clasificacion: {clasificacion}")
    if errores:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
