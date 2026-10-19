#!/usr/bin/env python
"""
P92-W - Auditoria de exposicion de holdout2 y reconstruccion segura del paquete.

Genera (ASCII puro):
  reports/p92w_holdout2_exposure_audit.md    (auditoria de exposicion)
  reports/p92w_eligible_population.csv       (poblacion elegible dev2 = 1236)
  reports/p92w_sampling_manifest.json        (muestreo nuevo: semillas, margenes, composicion)
  reports/p92w_package_audit.md              (auditoria del paquete nuevo)
  reports/p92w_internal_sampling_key.csv     (clave interna; fuera de los paquetes)
  artifacts/human_annotation_v2_1_holdout_safe/  (paquete nuevo)
  artifacts/human_annotation_v2/QUARANTINE_NOTICE.md (marca de cuarentena del v2)

Reglas: sin entrenamiento, sin predicciones, sin submissions, sin leaderboard; no se
releen labels/scores/metricas de holdout2; la cola P91 se filtra por id reservado ANTES
de usar cualquier otro campo; no se modifican artefactos historicos.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
V1 = RAIZ / "artifacts" / "human_annotation_v1"
V2 = RAIZ / "artifacts" / "human_annotation_v2"
V21 = RAIZ / "artifacts" / "human_annotation_v2_1_holdout_safe"
RUTA_TRAIN = RAIZ / "data" / "raw" / "toxic_train.csv"
RUTA_SUBMISSION = RAIZ / "data" / "raw" / "submission.csv"
RUTA_HOLDOUT_V2 = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
RUTA_HOLDOUT2 = RAIZ / "data" / "processed" / "holdout2_ids.csv"
RUTA_GRUPOS = RAIZ / "data" / "processed" / "grupos_near_duplicados.csv"
RUTA_COLA = RAIZ / "reports" / "p91_label_review_candidates.csv"
RUTA_CLAVE_VIEJA = RAIZ / "reports" / "p92_internal_sampling_key.csv"
RUTA_GUIA = RAIZ / "reports" / "p92_annotation_guidelines_v2_frozen.md"
RUTA_P92T = RAIZ / "reports" / "p92t_audit_results.json"
RUTA_AUDIT_EXPO = RAIZ / "reports" / "p92w_holdout2_exposure_audit.md"
RUTA_POBLACION = RAIZ / "reports" / "p92w_eligible_population.csv"
RUTA_MANIFEST = RAIZ / "reports" / "p92w_sampling_manifest.json"
RUTA_AUDIT_PKG = RAIZ / "reports" / "p92w_package_audit.md"
RUTA_CLAVE_NUEVA = RAIZ / "reports" / "p92w_internal_sampling_key.csv"

FECHA = "2026-10-13"
POLICY_ID = "PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2"
COLUMNAS_V2 = ["review_id", "text", "HS_annotation", "TR_annotation", "AG_annotation",
               "uncertainty_notes", "tweet_note", "guideline_version"]
ETIQUETAS = ("HS", "TR", "AG")
CAP_B = 150
CAP_GRUPO = 2

# Semillas nuevas, registradas antes de seleccionar (distintas de las de P92)
SEMILLAS = {
    "A": 20261017,
    "B": 20261018,
    "calibracion": 20261019,
    "review_id": 20261020,
    "orden_A": 20261021,
    "orden_B": 20261022,
    "orden_calibracion": 20261023,
}

HASHES_ESPERADOS = {
    "data/raw/toxic_train.csv": "3d40e7345b588860",
    "features/oof_v03_final.npz": "559265816ce4f631",
    "data/processed/holdout_v2_ids.csv": "e5b62e5f22c4c63a",
    "submissions/toxic_submission_v03_lr_union_thr.csv": "7d7478f271e6edfd",
    "artifacts/human_annotation_v1/annotator_A.csv": "7dfd5b898d3905f2",
    "artifacts/human_annotation_v1/annotator_B.csv": "956646fa479f0c65",
}
HASHES_V2_PRESERVAR = {
    "annotator_A/annotator_A.csv": None,
    "annotator_A/README_annotator.md": None,
    "annotator_A/annotation_guidelines_v2_frozen.md": None,
    "annotator_B/annotator_B.csv": None,
    "annotator_B/README_annotator.md": None,
    "annotator_B/annotation_guidelines_v2_frozen.md": None,
    "manifest.json": None,
}

TOKENS_FUGA = ["original_example_id", "sample_component", "selection_reason", "dup_group",
               "sampling_stratum", "source_partition", "original_hs", "original_tr",
               "original_ag", "v03", "holdout2", "holdout_v2"]


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


def leer_train() -> dict[int, str]:
    """id -> texto del corpus etiquetado (CSV estandar con quotechar)."""
    id_texto: dict[int, str] = {}
    with open(RUTA_TRAIN, encoding="utf-8-sig", newline="") as fh:
        lector = csv.reader(fh, delimiter=";", quotechar='"')
        next(lector)
        for campos in lector:
            if not campos:
                continue
            id_texto[int(campos[0].strip().strip('"'))] = campos[1]
    return id_texto


def margen(p: float, n: int, N: int, deff: float = 1.2, z: float = 1.96) -> float:
    """Margen de error (95%) con FPC y DEFF declarado, en puntos porcentuales."""
    fpc = math.sqrt((N - n) / (N - 1))
    return z * math.sqrt(p * (1 - p) / n) * math.sqrt(deff) * fpc * 100.0


def main() -> None:
    errores: list[str] = []
    avisos: list[str] = []

    # ------------------------------------------------------------------ cargas
    h2_ids = {int(f["id"]) for f in leer_csv(RUTA_HOLDOUT2)}
    sub_ids = {int(f["id"]) for f in leer_csv(RUTA_SUBMISSION)}
    hv = leer_csv(RUTA_HOLDOUT_V2)
    dev_ids = {int(f["id"]) for f in hv if f["particion"] == "dev"}
    hold_ids = {int(f["id"]) for f in hv if f["particion"] == "holdout"}
    reservados = h2_ids | sub_ids | hold_ids
    textos = leer_train()
    train_ids = set(textos)
    grupos = {int(f["id"]): int(f["grupo"]) for f in leer_csv(RUTA_GRUPOS)
              if f["split"] == "train"}

    # dev2 = dev - holdout2 (la particion cubre el train completo)
    dev2 = sorted(dev_ids - h2_ids)
    if train_ids != (dev_ids | hold_ids | h2_ids):
        errores.append("la particion no cubre exactamente el train")
    if len(dev2) != 1236:
        errores.append(f"dev2 inesperado: {len(dev2)}")

    # cola P91: filtrar reservados ANTES de usar cualquier otro campo
    cola_cruda = leer_csv(RUTA_COLA)
    por_tweet: dict[int, dict] = {}
    for f in cola_cruda:
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

    # --------------------------------------------------- exposicion (metadatos)
    clave_vieja = leer_csv(RUTA_CLAVE_VIEJA)
    expuestos_main = sorted(
        f["review_id"] for f in clave_vieja
        if "CALIB" not in f["sample_component"].upper()
        and int(f["original_example_id"]) in h2_ids)
    expuestos_cal = sorted(
        f["review_id"] for f in clave_vieja
        if "CALIB" in f["sample_component"].upper()
        and int(f["original_example_id"]) in h2_ids)
    cal_elegibles = sorted(
        f["review_id"] for f in clave_vieja
        if "CALIB" in f["sample_component"].upper()
        and int(f["original_example_id"]) not in h2_ids)
    citados_seis = {"P92C-05", "P92C-09", "P92C-11", "P92C-13", "P92C-14", "P92C-15"}
    expuestos_citados = [r for r in expuestos_cal if r in citados_seis]

    # evaluacion limitada con los 12 elegibles (sin procesar los 3 excluidos)
    with open(RUTA_P92T, encoding="utf-8") as fh:
        d_p92t = json.load(fh)
    casos = d_p92t["casos"]["respuestas_crudas_por_fila"]
    UTD = "UNABLE_TO_DETERMINE"
    doce_coinc_utd_hs, doce_con_duda = [], []
    for rid in cal_elegibles:
        fila = casos[rid]
        utd_hs = (fila["C"]["HS"] == UTD and fila["D"]["HS"] == UTD)
        alguna = (utd_hs or any(fila[a][e] == UTD for a in ("C", "D") for e in ETIQUETAS)
                  or fila["A"]["notas"] == "1" or fila["B"]["notas"] == "1")
        if utd_hs:
            doce_coinc_utd_hs.append(rid)
        if alguna:
            doce_con_duda.append(rid)

    # ------------------------------------------------------- muestreo nuevo
    rng_a = np.random.default_rng(SEMILLAS["A"])
    idx_a = sorted(rng_a.choice(len(dev2), size=100, replace=False).tolist())
    comp_a = [dev2[i] for i in idx_a]

    enriquecidos = sorted(e for e, t in por_tweet.items()
                          if t["reasons"] - {"RANDOM_REFERENCE_SAMPLE"} and e not in set(comp_a))
    t1 = [e for e in enriquecidos if "POSSIBLE_INSTANCE_LABEL_INCONSISTENCY" in por_tweet[e]["reasons"]]
    t2 = [e for e in enriquecidos if "DOCUMENTED_RULE_CONFLICT" in por_tweet[e]["reasons"]
          and e not in set(t1)]
    t3 = [e for e in enriquecidos if "HIGH_CONFIDENCE_OOF_DISAGREEMENT" in por_tweet[e]["reasons"]
          and e not in set(t1) and e not in set(t2)]
    t4 = [e for e in enriquecidos if e not in set(t1) and e not in set(t2) and e not in set(t3)]

    rng_b = np.random.default_rng(SEMILLAS["B"])
    seleccion_b: list[int] = []
    conteo_grupo: Counter = Counter()

    def admitir(eid: int) -> bool:
        if len(seleccion_b) >= CAP_B:
            return False
        g = por_tweet[eid]["grupo"]
        if g and conteo_grupo[g] >= CAP_GRUPO:
            return False
        seleccion_b.append(eid)
        if g:
            conteo_grupo[g] += 1
        return True

    for e in t1:
        admitir(e)
    for e in t2:
        admitir(e)
    t3_por_label: dict[str, list[int]] = {et: [] for et in ETIQUETAS}
    for e in t3:
        t3_por_label[por_tweet[e]["label_max_margen"] or "HS"].append(e)
    for et in ETIQUETAS:
        orden = list(rng_b.permutation(len(t3_por_label[et])))
        t3_por_label[et] = [t3_por_label[et][i] for i in orden]
    while len(seleccion_b) < CAP_B and any(t3_por_label[et] for et in ETIQUETAS):
        for et in ETIQUETAS:
            if t3_por_label[et]:
                admitir(t3_por_label[et].pop(0))
    for e in t4:
        admitir(e)
    comp_b = seleccion_b

    if not comp_b or set(comp_b) & set(comp_a) or len(set(comp_b)) != len(comp_b):
        errores.append("componente B invalido")

    resto = sorted(set(dev2) - set(comp_a) - set(comp_b))
    rng_c = np.random.default_rng(SEMILLAS["calibracion"])
    idx_cal = sorted(rng_c.choice(len(resto), size=15, replace=False).tolist())
    calib = [resto[i] for i in idx_cal]

    principales = comp_a + comp_b
    if set(principales) & set(calib) or len(set(principales)) != 250:
        errores.append("muestra principal/calibracion invalida")
    for e in principales + calib:
        if e not in dev2:
            errores.append(f"id {e} fuera de dev2")

    # review_id opacos, ordenes y clave
    rng_id = np.random.default_rng(SEMILLAS["review_id"])
    perm_id = list(rng_id.permutation(len(principales)))
    mapa_id = {principales[int(p)]: f"P92W-{k:04d}" for k, p in enumerate(perm_id, start=1)}
    rng_oa = np.random.default_rng(SEMILLAS["orden_A"])
    orden_a = [principales[int(i)] for i in rng_oa.permutation(len(principales))]
    rng_ob = np.random.default_rng(SEMILLAS["orden_B"])
    orden_b = [principales[int(i)] for i in rng_ob.permutation(len(principales))]
    rng_oc = np.random.default_rng(SEMILLAS["orden_calibracion"])
    orden_cal = [calib[int(i)] for i in rng_oc.permutation(len(calib))]
    mapa_cal = {eid: f"P92W-C{k:02d}" for k, eid in enumerate(orden_cal, start=1)}
    if orden_a == orden_b:
        errores.append("los ordenes de A y B coinciden")

    # ------------------------------------------------------------- escritura
    guia_bytes = RUTA_GUIA.read_bytes()
    V21.mkdir(parents=True, exist_ok=True)
    for letra, orden in (("A", orden_a), ("B", orden_b)):
        carpeta = V21 / f"annotator_{letra}"
        carpeta.mkdir(parents=True, exist_ok=True)
        with open(carpeta / f"annotator_{letra}.csv", "w", encoding="utf-8", newline="") as fh:
            escritor = csv.writer(fh, lineterminator="\n")
            escritor.writerow(COLUMNAS_V2)
            for e in orden:
                escritor.writerow([mapa_id[e], textos[e], "", "", "", "", "", "v2"])
        (carpeta / "annotation_guidelines_v2_frozen.md").write_bytes(guia_bytes)
        (carpeta / "README_annotator.md").write_text(
            construir_readme_anotador(letra), encoding="utf-8")
    carpeta_cal = V21 / "calibration"
    carpeta_cal.mkdir(parents=True, exist_ok=True)
    with open(carpeta_cal / "calibration_sample.csv", "w", encoding="utf-8", newline="") as fh:
        escritor = csv.writer(fh, lineterminator="\n")
        escritor.writerow(COLUMNAS_V2)
        for e in orden_cal:
            escritor.writerow([mapa_cal[e], textos[e], "", "", "", "", "", "v2"])
    (carpeta_cal / "README_calibration.md").write_text(
        construir_readme_calibracion(), encoding="utf-8")

    # clave interna nueva (fuera de los paquetes); los labels del corpus se incorporan
    # SOLO para ids elegibles y no salen del registro interno
    # (los labels del corpus se incorporan a la clave interna; el paquete no los lleva)
    etiquetas_train: dict[int, tuple[int, int, int]] = {}
    with open(RUTA_TRAIN, encoding="utf-8-sig", newline="") as fh:
        lector = csv.reader(fh, delimiter=";", quotechar='"')
        next(lector)
        for campos in lector:
            if not campos:
                continue
            ident = int(campos[0].strip().strip('"'))
            if ident in set(principales) | set(calib):
                etiquetas_train[ident] = (int(campos[2]), int(campos[3]), int(campos[4]))
    columnas = ["review_id", "original_example_id", "sample_component",
                "selection_reason_codes", "sampling_stratum", "original_HS",
                "original_TR", "original_AG", "dup_group_id", "source_partition"]
    with open(RUTA_CLAVE_NUEVA, "w", encoding="utf-8", newline="") as fh:
        escritor = csv.writer(fh, lineterminator="\n")
        escritor.writerow(columnas)
        for e in principales:
            t = por_tweet.get(e, {"reasons": set(), "grupo": ""})
            razones = "; ".join(sorted(t["reasons"])) if t["reasons"] else "RANDOM_REFERENCE_SAMPLE"
            hs, tr, ag = etiquetas_train[e]
            escritor.writerow([mapa_id[e], e, "A" if e in set(comp_a) else "B", razones,
                               "referencia_aleatoria" if e in set(comp_a) else "enriquecido",
                               hs, tr, ag, t["grupo"], "dev"])
        for e in orden_cal:
            hs, tr, ag = etiquetas_train[e]
            escritor.writerow([mapa_cal[e], e, "CALIBRACION2", "SRS_SOBRE_DEV2_FUERA_DE_A_B",
                               "calibracion", hs, tr, ag, "", "dev"])

    # cuarentena del v2 (marca documental aditiva)
    (V2 / "QUARANTINE_NOTICE.md").write_text(aviso_cuarentena(), encoding="utf-8")

    # --------------------------------------------- verificaciones del paquete
    chequeos: dict[str, bool] = {}
    csv_a = leer_csv(V21 / "annotator_A" / "annotator_A.csv")
    csv_b = leer_csv(V21 / "annotator_B" / "annotator_B.csv")
    csv_cal = leer_csv(V21 / "calibration" / "calibration_sample.csv")
    chequeos["filas_250_250_15"] = len(csv_a) == 250 and len(csv_b) == 250 and len(csv_cal) == 15
    chequeos["mismos_casos_A_B"] = {f["review_id"] for f in csv_a} == {f["review_id"] for f in csv_b}
    chequeos["orden_distinto_A_B"] = [f["review_id"] for f in csv_a] != [f["review_id"] for f in csv_b]
    chequeos["sin_duplicados"] = (len({f["review_id"] for f in csv_a}) == 250
                                  and len({f["review_id"] for f in csv_b}) == 250)
    chequeos["anotaciones_vacias"] = all(
        f[c] == "" for f in csv_a + csv_b + csv_cal
        for c in ("HS_annotation", "TR_annotation", "AG_annotation",
                  "uncertainty_notes", "tweet_note"))
    chequeos["version_v2"] = all(f["guideline_version"] == "v2" for f in csv_a + csv_b + csv_cal)
    # textos contra la fuente
    mapa_inv = {r: e for e, r in mapa_id.items()}
    mapa_inv.update({r: e for e, r in mapa_cal.items()})
    ok_textos = sum(1 for f in csv_a + csv_b + csv_cal
                    if textos[mapa_inv[f["review_id"]]] == f["text"])
    chequeos["textos_identicos_fuente"] = ok_textos == 500 + 15
    # elegibilidad de todos los ids
    elegibles_ok = all(mapa_inv[f["review_id"]] in set(dev2)
                       for f in csv_a + csv_b + csv_cal)
    chequeos["todos_elegibles"] = elegibles_ok
    chequeos["sin_reservados"] = not (set(mapa_inv.values()) & reservados)
    # componentes disjuntos
    chequeos["componentes_disjuntos"] = not (set(comp_a) & set(comp_b)) \
        and not (set(principales) & set(calib))
    # guia identica
    chequeos["guia_identica"] = all(
        (V21 / f"annotator_{letra}" / "annotation_guidelines_v2_frozen.md").read_bytes()
        == guia_bytes for letra in ("A", "B"))
    # cegamiento: tokens de fuga en archivos entregables
    hallazgos = []
    for ruta in [V21 / f"annotator_{letra}" / n for letra in ("A", "B")
                 for n in (f"annotator_{letra}.csv", "README_annotator.md",
                           "annotation_guidelines_v2_frozen.md")] \
            + [V21 / "calibration" / "calibration_sample.csv",
               V21 / "calibration" / "README_calibration.md"]:
        texto = ruta.read_text(encoding="utf-8").lower()
        for token in TOKENS_FUGA + ["p92r-", "p92c-"]:
            if token in texto:
                hallazgos.append(f"{ruta.name}:{token}")
        if ruta.suffix == ".md":
            if any(str(e) in texto for e in set(principales) | set(calib)):
                hallazgos.append(f"{ruta.name}:id_original")
    chequeos["blinding_sin_tokens_fuga"] = not hallazgos
    if hallazgos:
        errores.append(f"blinding: {hallazgos}")
    # clave interna fuera de los paquetes
    chequeos["clave_interna_fuera"] = not str(RUTA_CLAVE_NUEVA).startswith(str(V21))
    # hashes historicos preservados
    v2_hashes = {n: sha256_archivo(V2 / n) for n in HASHES_V2_PRESERVAR}
    manifiesto_v2_previo = json.loads(
        (RAIZ / "reports" / "p92v_manifest.json").read_text(encoding="utf-8"))
    v2_previo = {k: v for k, v in manifiesto_v2_previo["files"].items()
                 if k.startswith("artifacts/human_annotation_v2/")}
    chequeos["v2_preservado"] = all(
        sha256_archivo(RAIZ / k) == v for k, v in v2_previo.items())
    canonicos_ok = all(
        sha256_archivo(RAIZ / r).startswith(p) for r, p in HASHES_ESPERADOS.items()
        if (RAIZ / r).exists())
    chequeos["hashes_canonicos_ok"] = canonicos_ok
    if not canonicos_ok:
        errores.append("hashes canonicos alterados")
    # ASCII
    generados = ([V21 / f"annotator_{letra}" / n for letra in ("A", "B")
                  for n in (f"annotator_{letra}.csv", "README_annotator.md",
                            "annotation_guidelines_v2_frozen.md")]
                 + [V21 / "calibration" / "calibration_sample.csv",
                    V21 / "calibration" / "README_calibration.md",
                    V21 / "manifest.json", V21 / "README.md",
                    RUTA_AUDIT_EXPO, RUTA_POBLACION, RUTA_MANIFEST, RUTA_AUDIT_PKG,
                    RUTA_CLAVE_NUEVA, V2 / "QUARANTINE_NOTICE.md"])
    # ------------------------------------------------ manifiesto del paquete
    (V21 / "README.md").write_text(construir_readme_paquete(), encoding="utf-8")
    hashes_pkg = {}
    for ruta in generados:
        if ruta.exists() and str(ruta).startswith(str(V21)) and ruta.name != "manifest.json":
            hashes_pkg[str(ruta.relative_to(RAIZ))] = sha256_archivo(ruta)
    manifiesto_pkg = {
        "package": "human_annotation_v2_1_holdout_safe",
        "policy_id": POLICY_ID,
        "date": FECHA,
        "guideline_sha256": sha256_archivo(RUTA_GUIA),
        "counts": {"annotator_rows": 250, "principal_tweets": 250,
                   "cells_per_annotator": 750, "calibration_tweets": 15},
        "population": "dev2 = dev (1483) menos holdout2 (247); ver reports/p92w_sampling_manifest.json",
        "blinding": {
            "hidden": ["original_example_id", "original_labels", "oof_probabilities",
                       "v03_predictions", "threshold_distances", "fold", "sample_component",
                       "selection_reasons", "dup_groups", "internal_key",
                       "other_annotator_responses", "partition_membership"],
            "visible": ["review_id", "text", "empty annotation columns",
                        "uncertainty_notes", "tweet_note", "guideline_version"],
        },
        "files": hashes_pkg,
        "distribution": "NOT_DISTRIBUTED",
        "supersedes": "human_annotation_v2 (QUARANTINED_NOT_FOR_DISTRIBUTION_PENDING_HOLDOUT2_AUDIT)",
        "limitations": [
            "politica operativa interna; no definicion oficial",
            "componente B no representativa (sin ponderacion no estima prevalencia)",
            "guia v2 con influencia indirecta documentada de 2 casos de calibracion expuestos",
            "sin adjudicacion aun; ningun gold standard",
        ],
    }
    (V21 / "manifest.json").write_text(
        json.dumps(manifiesto_pkg, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------ manifiesto de muestreo
    composicion_b_motivos = Counter(
        "; ".join(sorted(por_tweet[e]["reasons"])) if por_tweet.get(e, {}).get("reasons")
        else "RANDOM_REFERENCE_SAMPLE" for e in comp_b)
    combos = Counter()
    for e in comp_b:
        hs, tr, ag = etiquetas_train[e]
        combos[f"{hs}{tr}{ag}"] += 1
    grupos_b = Counter(por_tweet[e]["grupo"] for e in comp_b if por_tweet[e]["grupo"])
    manifiesto = {
        "pilot": "P92-W rebuilt main annotation package (holdout-safe)",
        "date": FECHA,
        "population_definition": "tweets etiquetados del corpus (toxic_train.csv, 1731) que pertenecen a dev (1483) y NO a holdout2 (247): dev2 = 1236",
        "exclusion_rules": {
            "test": "ids de la plantilla submission.csv (1154); no accedido el archivo de test",
            "historical_holdout": "particion holdout de holdout_v2_ids.csv (248)",
            "holdout2": "data/processed/holdout2_ids.csv (247)",
            "no_verificables": "ninguna fila quedo sin particion demostrable (1731 = 1236 + 248 + 247)",
        },
        "sampling_frame_size": len(dev2),
        "sample_size": {"A_referencia": len(comp_a), "B_enriquecido": len(comp_b),
                        "calibracion": len(calib), "principal": len(principales)},
        "random_seeds": SEMILLAS,
        "inclusion_probability": {
            "A": f"{len(comp_a)}/{len(dev2)} = SRS sin reemplazo",
            "B": "no probabilistica (seleccion por evidencia; no ponderable sin diseno)",
            "calibracion": f"SRS de {len(calib)} sobre dev2 fuera de A y B",
        },
        "finite_population_correction": math.sqrt((len(dev2) - 100) / (len(dev2) - 1)),
        "expected_sampling_precision": {
            "supuestos": "95%, DEFF 1.2 (heredado de P91-R), FPC aplicada, p por etiqueta",
            "por_etiqueta_p005": f"+-{margen(0.05, 100, len(dev2)):.2f} pp (N={len(dev2)}, n=100)",
            "por_etiqueta_p010": f"+-{margen(0.10, 100, len(dev2)):.2f} pp",
            "por_etiqueta_p050": f"+-{margen(0.50, 100, len(dev2)):.2f} pp (peor caso)",
            "agregado_celdas_p005": f"+-{margen(0.05, 300, 3 * len(dev2)):.2f} pp (N={3 * len(dev2)}, n=300)",
            "nota": "recalculado para N=1236; no se reutilizan los margenes de P91/P92",
        },
        "stratification": {
            "A": "ninguna (SRS sobre dev2 ordenado)",
            "B": "tiers PILI/DRC/HCD/LSM con round-robin por etiqueta en HCD; cap 2 por grupo",
            "tiers_disponibles_en_dev2": {"PILI": len(t1), "DRC": len(t2),
                                          "HCD": len(t3), "LSM_otros": len(t4)},
        },
        "composition_B": {
            "motivos": dict(composicion_b_motivos),
            "combinaciones_etiqueta": dict(sorted(combos.items())),
            "grupos_usados": len(grupos_b),
            "max_por_grupo": max(grupos_b.values()) if grupos_b else 0,
        },
        "hashes_origen": {
            "train": sha256_archivo(RUTA_TRAIN),
            "particion": sha256_archivo(RUTA_HOLDOUT_V2),
            "holdout2_ids": sha256_archivo(RUTA_HOLDOUT2),
            "submission_plantilla": sha256_archivo(RUTA_SUBMISSION),
            "cola_p91": sha256_archivo(RUTA_COLA),
            "guia_congelada": sha256_archivo(RUTA_GUIA),
        },
        "limitations": [
            "B no representativa de la poblacion",
            "dependencia residual por grupos near-dup (cap 2)",
            "politica operativa no oficial",
            "sin adjudicacion; no es gold standard",
        ],
    }
    RUTA_MANIFEST.write_text(json.dumps(manifiesto, ensure_ascii=True, indent=2) + "\n",
                             encoding="utf-8")

    # ---------------------------------------------------- poblacion elegible
    with open(RUTA_POBLACION, "w", encoding="utf-8", newline="") as fh:
        escritor = csv.writer(fh, lineterminator="\n")
        escritor.writerow(["original_example_id", "eligible", "source_partition",
                           "dup_group_id", "exclusion_reason"])
        for e in dev2:
            escritor.writerow([e, "True", "dev", grupos.get(e, ""), ""])

    # ------------------------------------------ ASCII puro y cierre de chequeos
    no_ascii = [str(r.relative_to(RAIZ)) for r in generados if r.exists()
                and any(b > 127 for b in r.read_bytes())]
    chequeos["ascii_puro"] = not no_ascii
    if no_ascii:
        errores.append(f"bytes > 127 en {no_ascii}")
    for clave_chk, valor in chequeos.items():
        if not valor:
            errores.append(f"chequeo fallido: {clave_chk}")

    # ------------------------------------------------------------ informes
    hashes_pkg["artifacts/human_annotation_v2_1_holdout_safe/manifest.json"] = \
        sha256_archivo(V21 / "manifest.json")
    escribir_auditoria_exposicion(expuestos_main, expuestos_cal, expuestos_citados,
                                  cal_elegibles, doce_coinc_utd_hs, doce_con_duda, v2_hashes)
    escribir_auditoria_paquete(chequeos, hashes_pkg, manifiesto, errores, avisos,
                               ok_textos, len(dev2))

    estado = ("NEW_HUMAN_ANNOTATION_PACKAGE = READY_WITH_VERIFIED_EXCLUSIONS"
              if not errores else "ANNOTATION_PACKAGE_BLOCKED")
    print("P92-W reconstruccion holdout-safe: ejecutado")
    print(f"  exposicion: {len(expuestos_main)} principales + {len(expuestos_cal)} calibracion")
    print(f"  poblacion dev2: {len(dev2)} | A={len(comp_a)} B={len(comp_b)} calibracion={len(calib)}")
    print(f"  textos verificados: {ok_textos}/515 | chequeos: {sum(chequeos.values())}/{len(chequeos)}")
    print(f"  errores: {errores if errores else 'ninguno'}")
    print(f"  estado: {estado}")
    if errores:
        raise SystemExit(1)


def construir_readme_anotador(letra: str) -> str:
    """README del anotador para el paquete holdout-safe."""
    return f"""# Anotacion principal (piloto interno) - instrucciones para el anotador/a {letra}

Este paquete es autosuficiente. Contiene tres archivos:
1. README_annotator.md (este archivo)
2. annotation_guidelines_v2_frozen.md (guia operativa v2 congelada:
   PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2)
3. annotator_{letra}.csv (250 textos; anotaciones vacias)

## Objetivo

Anotar, de forma independiente, las tres etiquetas (HS, TR, AG) de 250 textos: 750
anotaciones de etiqueta; 3 respuestas posibles por anotacion (POSITIVE / NEGATIVE /
UNABLE_TO_DETERMINE).

## Que debes hacer

1. Lee primero `annotation_guidelines_v2_frozen.md` completo.
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

Guia aplicable: PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2 (politica operativa interna
del proyecto; no una definicion oficial de la competencia).
"""


def construir_readme_calibracion() -> str:
    """README de la calibracion del paquete holdout-safe."""
    return """# Calibracion - ronda de familiarizacion (paquete holdout-safe)

Este paquete contiene dos archivos:
1. README_calibration.md (este archivo)
2. calibration_sample.csv (15 textos; anotaciones vacias)

## Objetivo

Familiarizarse con la guia congelada antes de la tarea principal. La ronda es
independiente: no comentes los casos con la otra persona anotadora hasta terminar.

## Que debes hacer

1. Lee primero la guia congelada de tu paquete principal.
2. Anota los 15 textos de forma independiente y completa.
3. Usa EXCLUSIVAMENTE POSITIVE / NEGATIVE / UNABLE_TO_DETERMINE (los 0/1 no son
   validos); motivos en `uncertainty_notes` cuando uses UNABLE_TO_DETERMINE.
4. No consultes etiquetas originales, modelos, predicciones ni puntuaciones.
5. Devuelve el CSV completado con el nombre exacto: `calibration_response.csv`

## Que pasa despues

Se conservan tus respuestas originales sin cambios; la version aplicable a la tarea
principal ya esta congelada y no cambia con esta ronda. Los desacuerdos y dudas se
documentan en la adjudicacion posterior.
"""


def construir_readme_paquete() -> str:
    """README interno del paquete (no se entrega a los anotadores)."""
    return f"""# Paquete human_annotation_v2_1_holdout_safe (interno)

Fecha: {FECHA}. Estado: preparado, NO distribuido.
Relacion con versiones previas:
- `human_annotation_v2` queda QUARANTINED_NOT_FOR_DISTRIBUTION_PENDING_HOLDOUT2_AUDIT
  (35 filas principales y 3 de calibracion pertenecian a holdout2; ver
  reports/p92w_holdout2_exposure_audit.md).
- `human_annotation_v1` es historico y tampoco debe distribuirse (contiene las mismas
  filas expuestas, sin anotaciones).
- Este paquete reutiliza la misma guia congelada (PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2)
  y una poblacion reconstruida (dev2 = dev sin holdout2) con semillas nuevas.

Estructura: annotator_A/, annotator_B/ (CSV + guia + README), calibration/ (muestra
nueva + README), manifest.json. La clave interna esta en
reports/p92w_internal_sampling_key.csv (fuera de este directorio).

No distribuir automaticamente: la entrega la decide el propietario del proyecto.
"""


def aviso_cuarentena() -> str:
    """Marca documental de cuarentena del paquete v2."""
    return f"""# AVISO DE CUARENTENA - human_annotation_v2

Estado: QUARANTINED_NOT_FOR_DISTRIBUTION_PENDING_HOLDOUT2_AUDIT ({FECHA}).

Motivo: 35 de las 250 filas principales y 3 de las 15 de calibracion pertenecen al
pliegue reservado holdout2. Detalle completo en:
reports/p92w_holdout2_exposure_audit.md.

- NO distribuir estos paquetes a ningun anotador.
- No eliminar: se conservan como version historica y como evidencia de la auditoria.
- El reemplazo seguro esta en artifacts/human_annotation_v2_1_holdout_safe/.
- Este aviso es aditivo: no modifica ningun archivo del paquete.
"""


def escribir_auditoria_exposicion(expuestos_main, expuestos_cal, expuestos_citados,
                                  cal_elegibles, doce_coinc, doce_duda, v2_hashes) -> None:
    """Auditoria de exposicion de holdout2."""
    lineas = []
    lineas.append("# P92-W - Auditoria de exposicion de holdout2 y reconstruccion segura")
    lineas.append("")
    lineas.append(f"Fecha: {FECHA}. Fase diagnostica: sin entrenamiento, sin predicciones, sin")
    lineas.append("submissions, sin leaderboard. No se releyeron labels, scores ni metricas de")
    lineas.append("holdout2: la auditoria usa exclusivamente metadatos de pertenencia ya")
    lineas.append("registrados y documentacion existente.")
    lineas.append("")
    lineas.append("## 1. Objeto")
    lineas.append("")
    lineas.append("La instruccion global del proyecto establece HOLDOUT2 = FROZEN y su no uso")
    lineas.append("durante la exploracion. El muestreo de P92 declaro excluir holdout2 en su")
    lineas.append("manifiesto, pero el marco usado (dev = 1483) lo contenia. Esta auditoria")
    lineas.append("documenta la exposicion real y reconstruye el paquete con exclusiones")
    lineas.append("verificadas.")
    lineas.append("")
    lineas.append("## 2. Cronologia (reconstruida de artefactos existentes)")
    lineas.append("")
    lineas.append("- 2026-09-26: se crea holdout2 (experiments/v10_01_holdout2.py) como pliegue de")
    lineas.append("  dev para deteccion de sobreajuste; el propio script advierte que NO es un")
    lineas.append("  conjunto virginal (sus filas participaron en decisiones de fases v1-v9).")
    lineas.append("- 2026-10-09: P92 construye el muestreo. El manifiesto declara excluidos test,")
    lineas.append("  holdout y holdout2; sin embargo el marco efectivo fue dev (1483), que incluye")
    lineas.append("  holdout2 (247). La auditoria de preparacion valido 14/14 contra la particion")
    lineas.append("  holdout de holdout_v2, sin comprobar holdout2.")
    lineas.append("- 2026-10-12: calibracion (15 tweets, 4 anotadores) y analisis P92-T; el")
    lineas.append("  analisis no consulto particiones reservadas.")
    lineas.append("- 2026-10-13: la auditoria de P92-V detecta 35 filas principales y 3 de")
    lineas.append("  calibracion con pertenencia a holdout2 (por interseccion de ids) y lo")
    lineas.append("  registra como aviso; P92-W escala a auditoria formal y reconstruccion.")
    lineas.append("")
    lineas.append("## 3. Inventario de acceso")
    lineas.append("")
    lineas.append("- Consultado (solo metadatos): data/processed/holdout2_ids.csv (lista de 247")
    lineas.append("  ids; el archivo no contiene labels ni scores); cabecera de v10_01_holdout2.py")
    lineas.append("  (procedencia); declaraciones de auditorias P90 ('congelado, no tocado').")
    lineas.append("- NO consultado: textos, labels, scores, predicciones o metricas de holdout2")
    lineas.append("  como conjunto; no existe ningun artefacto especifico de contenido de holdout2.")
    lineas.append("- En P92-W la cola P91 (candidatos) se filtro por id reservado ANTES de usar")
    lineas.append("  cualquier otro campo; el OOF canonico no se cargo en esta fase.")
    lineas.append("")
    lineas.append("## 4. Casos expuestos (metadatos de pertenencia)")
    lineas.append("")
    lineas.append(f"- Filas principales afectadas ({len(expuestos_main)}/250): "
                   f"{', '.join(expuestos_main)}.")
    lineas.append(f"- Calibracion afectada ({len(expuestos_cal)}/15): "
                   f"{', '.join(expuestos_cal)}.")
    lineas.append("- Deteccion: interseccion de ids en la clave interna contra la lista de")
    lineas.append("  holdout2; no requirio leer ningun contenido de holdout2.")
    lineas.append("")
    lineas.append("## 5. Clasificacion de la exposicion")
    lineas.append("")
    lineas.append("```text")
    lineas.append("HOLDOUT2_TEXT_EXPOSURE:              DEMOSTRADO (38 filas: textos dentro de")
    lineas.append("                                    paquetes de anotacion; 3 revisados por humanos)")
    lineas.append("HOLDOUT2_HUMAN_ANNOTATION_EXPOSURE: DEMOSTRADO para 3 filas de calibracion")
    lineas.append("                                    (4 respuestas humanas + notas + analisis);")
    lineas.append("                                    NO OCURRIO para las 35 principales (sin anotaciones)")
    lineas.append("HOLDOUT2_LABEL_ACCESS:              En P92: no se leyeron/analizaron labels por fila")
    lineas.append("                                    (diseno ciego; analisis de calibracion sin labels).")
    lineas.append("                                    Historico: las filas eran dev antes de V10; su")
    lineas.append("                                    participacion en analisis agregados de dev no es")
    lineas.append("                                    verificable por fila -> NO VERIFICABLE por fila")
    lineas.append("HOLDOUT2_SCORE_ACCESS:              El OOF canonico cubre dev (incluye esas filas);")
    lineas.append("                                    no se uso para seleccion ni analisis en P92/P92-W;")
    lineas.append("                                    uso historico agregado -> NO VERIFICABLE por fila")
    lineas.append("HOLDOUT2_METRIC_ACCESS:             NO OCURRIO en P92/P92-V/P92-W (ninguna metrica")
    lineas.append("                                    se computo sobre holdout2 como conjunto)")
    lineas.append("```")
    lineas.append("")
    lineas.append("No se afirma que holdout2 haya permanecido completamente intacto: la exposicion")
    lineas.append("real queda documentada arriba, sin inferir acceso a labels o scores a partir de")
    lineas.append("la revision de textos.")
    lineas.append("")
    lineas.append("## 6. Influencia en la guia v2 (evaluacion)")
    lineas.append("")
    lineas.append(f"- De los 3 casos expuestos de calibracion, {len(expuestos_citados)} "
                   f"({', '.join(expuestos_citados)}) figuran entre las seis filas HS citadas")
    lineas.append("  como evidencia central de la guia v2 (patron odio vs. ofensa); P92C-10 se")
    lineas.append("  describio como caso sin duda. Por tanto NO puede demostrarse que las")
    lineas.append("  decisiones operativas de v2 sean independientes de la evidencia expuesta.")
    lineas.append("- Se registra: GUIDELINE_POTENTIAL_INDIRECT_HOLDOUT2_INFLUENCE.")
    lineas.append(f"- Evaluacion limitada con los {len(cal_elegibles)} casos elegibles de la")
    lineas.append("  calibracion (sin reabrir los 3 excluidos; usando el analisis ya existente):")
    lineas.append(f"  UTD coincidente C/D en HS: {len(doce_coinc)}/{len(cal_elegibles)} "
                   f"({', '.join(doce_coinc)}); con alguna senal de duda: "
                   f"{len(doce_duda)}/{len(cal_elegibles)}.")
    lineas.append("  El patron de ambiguedad normativa que fundamenta Q1 se conserva en")
    lineas.append("  evidencia elegible; esto sostiene la direccion operativa, pero no")
    lineas.append("  sustituye la independencia plena (los conteos originales 6/15 incluian 2")
    lineas.append("  filas expuestas).")
    lineas.append("- v2 se mantiene como POLITICA OPERATIVA interna (nunca fue definicion")
    lineas.append("  oficial). Se registra la necesidad de una revision humana de la guia")
    lineas.append("  apoyada solo en evidencia elegible si se desea independencia plena; la")
    lineas.append("  nueva calibracion del paquete holdout-safe sirve a ese fin.")
    lineas.append("")
    lineas.append("## 7. Artefactos derivados que contienen informacion de los casos expuestos")
    lineas.append("")
    lineas.append("- artifacts/human_annotation_v1/ (38 textos; sin anotaciones) [historico]")
    lineas.append("- artifacts/human_annotation_v2/ (38 textos) "
                   "[QUARANTINED_NOT_FOR_DISTRIBUTION_PENDING_HOLDOUT2_AUDIT]")
    lineas.append("- calibration_response_A/B/C/D.csv (3 filas con respuestas humanas)")
    lineas.append("- reports/p92_calibration_analysis.md, p92t_audit_results.json, p92s2_*, p92u_*,")
    lineas.append("  p92_guideline_decision_record.md, p92v_* y notebook (secciones 135-143)")
    lineas.append("- reports/p92_internal_sampling_key.csv (pertenencia y labels de los 265)")
    lineas.append("- Esta auditoria y el manifiesto P92-W (metadatos de pertenencia)")
    lineas.append("")
    lineas.append("## 8. Tratamiento")
    lineas.append("")
    lineas.append("- Las respuestas humanas y los analisis derivados de los 3 casos de")
    lineas.append("  calibracion quedan EXCLUIDOS de futuras evaluaciones que requieran holdout2")
    lineas.append("  sin exposicion; no se les pedira a los anotadores volver a etiquetarlos.")
    lineas.append("- Las 35 filas principales expuestas quedan excluidas de la muestra nueva;")
    lineas.append("  la reconstruccion usa la poblacion dev2 (1236) con semillas nuevas.")
    lineas.append("- No se modifico ningun label original; ninguna discrepancia futura sera")
    lineas.append("  correccion automatica (vocabulario de estados de P92-V vigente).")
    lineas.append("")
    lineas.append("## 9. Hashes relevantes")
    lineas.append("")
    lineas.append("```text")
    for ruta_rel, prefijo in HASHES_ESPERADOS.items():
        lineas.append(f"{ruta_rel}: {prefijo}...")
    lineas.append("```")
    lineas.append("")
    lineas.append("Hashes completos de los paquetes en cuarentena (v2, sin cambios):")
    lineas.append("")
    for nombre, h in sorted(v2_hashes.items()):
        lineas.append(f"- {nombre}: {h}")
    RUTA_AUDIT_EXPO.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"  escrito {RUTA_AUDIT_EXPO.relative_to(RAIZ)}")


def escribir_auditoria_paquete(chequeos, hashes_pkg, manifiesto, errores, avisos,
                               ok_textos, tam_poblacion) -> None:
    """Auditoria del paquete holdout-safe."""
    estado = ("NEW_HUMAN_ANNOTATION_PACKAGE = READY_WITH_VERIFIED_EXCLUSIONS"
              if not errores else "ANNOTATION_PACKAGE_BLOCKED")
    lineas = []
    lineas.append("# P92-W - Auditoria del paquete holdout-safe (human_annotation_v2_1)")
    lineas.append("")
    lineas.append(f"Fecha: {FECHA}. Estado: **{estado}**.")
    lineas.append("")
    lineas.append("## 1. Poblacion y exclusiones")
    lineas.append("")
    lineas.append(f"- Poblacion elegible dev2 = {tam_poblacion} ids (dev 1483 menos holdout2 247).")
    lineas.append("- Reglas: elegible si pertenece a dev en la particion congelada y no a")
    lineas.append("  holdout2, holdout historico ni test; ninguna fila quedo sin particion")
    lineas.append("  demostrable (1731 = 1236 + 248 + 247).")
    lineas.append("- Test: ids de la plantilla submission.csv (solo columna id); el archivo")
    lineas.append("  toxic_test.csv no se accedio.")
    lineas.append("")
    lineas.append("## 2. Muestreo")
    lineas.append("")
    lineas.append(f"- A: SRS de 100 sobre {tam_poblacion} (semilla {SEMILLAS['A']}); B: hasta")
    lineas.append(f"  150 por tiers con cap 2 por grupo (semilla {SEMILLAS['B']}); calibracion:")
    lineas.append(f"  SRS de 15 fuera de A y B (semilla {SEMILLAS['calibracion']}).")
    lineas.append(f"- Probabilidad de inclusion A: 100/{tam_poblacion}; margenes recalculados")
    prec = manifiesto["expected_sampling_precision"]
    lineas.append(f"  ({prec['por_etiqueta_p005']}; {prec['por_etiqueta_p050']};")
    lineas.append("  DEFF 1.2, FPC; margenes recalculados, no reutilizados; ver manifiesto).")
    lineas.append("")
    lineas.append("## 3. Verificaciones")
    lineas.append("")
    for clave_chk, valor in chequeos.items():
        lineas.append(f"- {clave_chk}: {'OK' if valor else 'FALLO'}")
    lineas.append(f"- textos contra la fuente: {ok_textos}/515")
    lineas.append("")
    lineas.append("## 4. Hashes de los archivos nuevos")
    lineas.append("")
    lineas.append("| archivo | sha256 |")
    lineas.append("|---|---|")
    for ruta_txt, h in sorted(hashes_pkg.items()):
        lineas.append(f"| {ruta_txt} | `{h}` |")
    lineas.append("")
    lineas.append("## 5. Relacion con paquetes anteriores")
    lineas.append("")
    lineas.append("- human_annotation_v1: historico, nunca distribuido, contiene las 38 filas")
    lineas.append("  expuestas; no usar para el piloto.")
    lineas.append("- human_annotation_v2: QUARANTINED_NOT_FOR_DISTRIBUTION_PENDING_HOLDOUT2_AUDIT")
    lineas.append("  (aviso aditivo en artifacts/human_annotation_v2/QUARANTINE_NOTICE.md).")
    lineas.append("- human_annotation_v2_1_holdout_safe: este paquete (NO distribuido).")
    lineas.append("")
    lineas.append("## 6. Resultado")
    lineas.append("")
    if errores:
        lineas.append("Errores:")
        for e in errores:
            lineas.append(f"- {e}")
    else:
        lineas.append("- Sin errores: paquete listo con exclusiones verificadas (NO distribuido).")
    if avisos:
        lineas.append("")
        lineas.append("Avisos:")
        for a in avisos:
            lineas.append(f"- {a}")
    RUTA_AUDIT_PKG.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"  escrito {RUTA_AUDIT_PKG.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
