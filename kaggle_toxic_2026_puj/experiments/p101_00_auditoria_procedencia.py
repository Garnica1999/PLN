#!/usr/bin/env python
"""
P101-00: auditoria de procedencia de los materiales de anotacion (P90-P93).

Solo lectura sobre los materiales historicos; escribe:
  reports/p101_materials_inventory.csv
  reports/p101_provenance_table.csv
  reports/p101_provenance_audit.md

Reglas: las particiones reservadas (holdout, holdout2, test) se consultan
SOLO por su columna de id (nunca texto, labels ni scores); no se modifica
ningun artefacto historico; sin entrenamiento.

Uso:
    /home/carlos/venv/bin/python experiments/p101_00_auditoria_procedencia.py
"""

from __future__ import annotations

import csv
import hashlib
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
SAL_INV = RAIZ / "reports" / "p101_materials_inventory.csv"
SAL_TAB = RAIZ / "reports" / "p101_provenance_table.csv"
SAL_AUD = RAIZ / "reports" / "p101_provenance_audit.md"

CANONICOS = {
    "data/raw/toxic_train.csv": "3d40e7345b588860",
    "features/oof_v03_final.npz": "559265816ce4f631",
    "experiments/baseline_v03_frozen/submission_v03.csv": "7d7478f271e6edfd",
    "experiments/baseline_v03_frozen/model_v03.joblib": "263ed5bd0fc0de7c",
    "data/processed/holdout_v2_ids.csv": "e5b62e5f22c4c63a",
    "data/processed/holdout2_ids.csv": "f6c6ed958abf728b",
}


def sha256_16(ruta: Path) -> str:
    """sha256 truncado a 16 hexdigitos."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()[:16]


def normaliza_id(texto: str) -> int:
    """Id normalizado (sin comillas ni espacios; entero)."""
    return int(texto.strip().strip('"').strip())


def ids_columna_cero(ruta: Path) -> set:
    """Ids de un CSV con separador ';' usando SOLO la columna 0."""
    ids = set()
    with open(ruta, encoding="utf-8", errors="replace") as fh:
        for linea in fh:
            token = linea.split(";", 1)[0]
            if token == "id":
                continue
            try:
                ids.add(normaliza_id(token))
            except ValueError:
                continue
    return ids


def leer_csv(ruta: Path) -> list:
    """Filas de un CSV (separador coma)."""
    with open(ruta, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    """Construye inventario, tabla de procedencia e informe."""
    for ruta in (SAL_INV, SAL_TAB, SAL_AUD):
        if ruta.exists():
            raise SystemExit(f"Ya existe {ruta}; no se sobrescribe.")
    inicio = time.perf_counter()

    for rel, pref in CANONICOS.items():
        real = sha256_16(RAIZ / rel)
        if not real.startswith(pref):
            raise SystemExit(f"canonico alterado: {rel} -> {real}")

    z = np.load(RAIZ / "features" / "oof_v03_final.npz", allow_pickle=True)
    dev_ids = {int(v) for v in z["ids"]}
    particion = {}
    for r in leer_csv(RAIZ / "data" / "processed" / "holdout_v2_ids.csv"):
        particion[int(r["id"])] = r["particion"]
    holdout_ids = {k for k, v in particion.items() if v == "holdout"}
    holdout2_ids = {int(r["id"])
                    for r in leer_csv(RAIZ / "data" / "processed"
                                      / "holdout2_ids.csv")}
    test_ids = ids_columna_cero(RAIZ / "data" / "raw" / "toxic_test.csv")
    train_ids = ids_columna_cero(RAIZ / "data" / "raw" / "toxic_train.csv")
    dev2 = {k for k, v in particion.items() if v == "dev"} - holdout2_ids

    v21 = leer_csv(RAIZ / "reports" / "p92w_internal_sampling_key.csv")
    v21_por_id = {int(r["original_example_id"]): r for r in v21}
    v21_por_rid = {r["review_id"]: r for r in v21}
    v21_main = {i for i, r in v21_por_id.items()
                if r["review_id"].startswith("P92W-0")}
    v21_cal = {i for i, r in v21_por_id.items()
               if r["review_id"].startswith("P92W-C")}
    v2orig = {int(r["original_example_id"])
              for r in leer_csv(RAIZ / "reports"
                                / "p92_internal_sampling_key.csv")}
    p90 = {int(r["tweet_id"])
           for r in leer_csv(RAIZ / "reports" / "v03_error_review_sample.csv")}
    p91 = {int(r["example_id"])
           for r in leer_csv(RAIZ / "reports"
                             / "p91_label_review_candidates.csv")}
    r2_celdas = {}
    for r in leer_csv(RAIZ / "reports" / "p93r3_round2_id_assignment.csv"):
        rid = r["review_id"]
        if rid not in v21_por_rid:
            continue
        oid = int(v21_por_rid[rid]["original_example_id"])
        r2_celdas[oid] = r2_celdas.get(oid, 0) + 1

    def fuente(i: int) -> str:
        if i in test_ids:
            return "test"
        if i in holdout2_ids:
            return "holdout2"
        if i in holdout_ids:
            return "holdout"
        if i in particion and particion[i] == "dev":
            return "dev"
        if i in train_ids:
            return "train_otro"
        return "desconocido"

    universo = dev2 | set(v21_por_id) | v2orig | p90 | p91
    filas = []
    for i in sorted(universo):
        f = fuente(i)
        es_main = i in v21_main
        es_cal = i in v21_cal
        es_v2o = i in v2orig
        es_p90 = i in p90
        es_p91 = i in p91
        celdas_r2 = r2_celdas.get(i, 0)
        elegible = (f == "dev" and not es_main and not es_cal)
        notas = []
        if es_p90:
            notas.append("identificado en muestra interna P90 (sin "
                         "revision humana)")
        if es_p91:
            notas.append("candidato P91 no distribuido (gobernanza "
                         "pendiente)")
        if es_v2o and not es_main:
            notas.append("presente en paquete v2 original (en cuarentena, "
                         "no distribuido)")
        if es_cal:
            notas.append("calibracion v2.1 preparada; sin respuestas; "
                         "exposicion incierta -> excluida por prudencia")
        if es_main:
            notas.append("muestra principal v2.1 anotada por A/B (ronda 1)")
        if celdas_r2:
            notas.append(f"ronda 2 C1/C2: {celdas_r2} celdas")
        if elegible and not (es_p90 or es_p91 or es_v2o):
            notas.append("elegible; sin materiales historicos")
        if not elegible and not notas:
            notas.append("no elegible")
        filas.append({
            "example_id": i,
            "source_partition": f,
            "in_dev_frame": f == "dev",
            "p90_casebook_sample": es_p90,
            "p91_candidate_pending": es_p91,
            "v2_original_quarantined": es_v2o,
            "v2_1_main_sample": es_main,
            "v2_1_calibration_no_response": es_cal,
            "review_id": v21_por_id[i]["review_id"] if i in v21_por_id else "",
            "sample_component": (v21_por_id[i]["sample_component"]
                                 if i in v21_por_id else ""),
            "r1_ab_cells": 3 if es_main else 0,
            "r2_c12_cells": celdas_r2,
            "human_decisions_available": (
                ("r1_A/B" + (f" + r2_C1/C2 ({celdas_r2} celdas)"
                             if celdas_r2 else "")) if es_main else ""),
            "gold_exposure": ("ninguna antes del bloqueo (R4); comparacion "
                              "R5 interna" if es_main else "n/a"),
            "guide_version": "v2" if es_main else "n/a",
            "independence_status": ("ronda 1 ciega"
                                    + ("; ronda 2 depende de A/B"
                                       if celdas_r2 else "")
                                    if es_main else "n/a"),
            "eligibility_new_audit": elegible,
            "notes": "; ".join(notas),
        })

    with open(SAL_TAB, "w", encoding="ascii", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()),
                           lineterminator="\n")
        w.writeheader()
        for fila in filas:
            w.writerow(fila)

    materiales = []
    patrones = [
        (RAIZ / "reports", "p9[0-3]*"), (RAIZ / "reports", "v03_error_*"),
        (RAIZ / "reports", "toxic_tr_annotation_governance*"),
        (RAIZ / "reports", "p94_*"), (RAIZ / "reports", "p95_*"),
        (RAIZ / "reports", "p96_*"),
    ]
    for base, patron in patrones:
        for p in sorted(base.glob(patron)):
            if p.is_file():
                materiales.append(p)
    for sub in ("human_annotation_v1", "human_annotation_v2",
                "human_annotation_v2_1_holdout_safe", "p93_adjudication_v1",
                "p93_adjudication_v2"):
        for p in sorted((RAIZ / "artifacts" / sub).rglob("*")):
            if p.is_file():
                materiales.append(p)
    for rel in ("docs/competition_rules.md",
                "docs/p73_provenance_autorizacion_corpus.md",
                "docs/p98_provenance_autorizacion_bilstm.md"):
        materiales.append(RAIZ / rel)
    materiales = sorted(set(materiales))

    with open(SAL_INV, "w", encoding="ascii", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["ruta", "sha256_16", "bytes"])
        for p in materiales:
            w.writerow([str(p.relative_to(RAIZ)), sha256_16(p),
                        p.stat().st_size])

    revisados_main = len(v21_main)
    revisados_cal = len(v21_cal)
    revisados_total = revisados_main + revisados_cal
    interseccion_reservados = v21_main & (holdout_ids | holdout2_ids
                                          | test_ids)
    cal_reservados = v21_cal & (holdout_ids | holdout2_ids | test_ids)
    contam_v2o = len(v2orig & holdout2_ids)
    p90_h2 = len(p90 & holdout2_ids)
    p91_h2 = len(p91 & holdout2_ids)
    elegibles = sum(1 for f in filas if f["eligibility_new_audit"])
    esperado_elegibles = len(dev2) - revisados_total
    if elegibles != esperado_elegibles:
        raise SystemExit(f"elegibles {elegibles} != esperado "
                         f"{esperado_elegibles}")
    duracion = round(time.perf_counter() - inicio, 1)

    md = [
        ("# P101-00 - Auditoria de procedencia de los materiales de "
         "anotacion (P90-P93)"),
        "",
        ("Fecha: 2026-10-14. Solo lectura sobre materiales historicos; sin "
         "entrenamiento; sin modificacion de labels; las particiones "
         "reservadas se consultaron SOLO por su columna de id (nunca "
         "texto, labels ni scores)."),
        "",
        "## 1. Controles canonicos",
        "",
        ("Verificados (sha256, prefijos): "
         + ", ".join(f"{k} {v}" for k, v in CANONICOS.items()) + "."),
        "",
        "## 2. Universo de ejemplos en materiales historicos",
        "",
        (f"- Muestra principal v2.1 (anotada por A/B en ronda 1): "
         f"{revisados_main} tweets (review_id P92W-0001..0250)."),
        (f"- Calibracion v2.1 (handoff preparado, sin respuestas): "
         f"{revisados_cal} tweets (P92W-C01..C15)."),
        (f"- Paquete v2 original: {len(v2orig)} ids, EN CUARENTENA y no "
         f"distribuido (contaminacion holdout2: {contam_v2o} ids; 35 "
         f"principales + 3 calibracion segun el aviso de cuarentena)."),
        (f"- Muestra interna del casebook P90 (sin revision humana): "
         f"{len(p90)} tweets."),
        (f"- Candidatos P91 (nunca distribuidos; gobernanza pendiente): "
         f"{len(p91)} ids."),
        (f"- Ronda 2 C1/C2 (P93): {sum(r2_celdas.values())} celdas sobre "
         f"{len(r2_celdas)} tweets."),
        (f"- Filas de la tabla de procedencia (dev2 completo + universo "
         f"historico): {len(filas)}."),
        "",
        "## 3. Particiones y elegibilidad",
        "",
        (f"- dev (particion de oof_v03_final): {len(dev_ids)} | dev2 "
         f"(dev menos holdout2): {len(dev2)} | holdout historico: "
         f"{len(holdout_ids)} | holdout2: {len(holdout2_ids)} | ids de "
         f"test (solo columna id): {len(test_ids)}."),
        (f"- Interseccion de la muestra principal v2.1 con particiones "
         f"reservadas (holdout/holdout2/test): "
         f"{len(interseccion_reservados)} (esperado 0). Calibracion: "
         f"{len(cal_reservados)} (esperado 0)."),
        (f"- Poblacion elegible para una auditoria humana NUEVA: "
         f"{elegibles} = dev2 ({len(dev2)}) menos {revisados_main} "
         f"revisados y menos {revisados_cal} de calibracion excluidos "
         f"por prudencia. La columna eligibility_new_audit de la tabla "
         f"lo refleja por ejemplo."),
        "",
        "## 4. Exposicion de revisores y del gold",
        "",
        ("- Ronda 1 (A/B): ciega; sin labels, predicciones ni respuestas "
         "cruzadas; guia v2; sin feedback del gold."),
        ("- Ronda 2 (C1/C2): decisiones bloqueadas ANTES de revelar el "
         "gold (P93-R4); los adjudicadores vieron las respuestas A/B de "
         "la ronda 1 (dependencia registrada)."),
        ("- Gold: revelado solo en la comparacion P93-R5 (interna del "
         "proyecto, autorizada); sin evidencia de feedback a los "
         "anotadores."),
        "",
        "## 5. Hueco de trazabilidad (P91-P93)",
        "",
        ("Las fases P91-P93 no constan en DECISIONS.md (salta D-295 a "
         "D-296), RUNLOG.md (P90-00 a P98), PROJECT_STATE.md ni "
         "experiments.csv; su registro vive en reports/p9x_* y "
         "artifacts/. Se completa con un addendum retrospectivo marcado "
         "(RUNLOG y DECISIONS) y se documentan las limitaciones en "
         "reports/p101_compliance.md."),
        "",
        "## 6. Limitaciones",
        "",
        ("- Identidades de revisores: solo IDs opacos (A/B/C1/C2); sin "
         "registro nominal."),
        ("- Exposicion de la calibracion v2.1 incierta (handoff "
         "preparado sin respuestas) -> excluida de elegibilidad por "
         "prudencia."),
        ("- Los candidatos P91 y la muestra P90 nunca pasaron por "
         "revision humana; su elegibilidad se conserva pero se registra "
         "la seleccion previa (enriquecida/interna)."),
        (f"- Las muestras internas P90 ({p90_h2} ids) y P91 ({p91_h2} "
         f"ids) incluyeron filas de holdout2 (seleccion interna sobre "
         f"dev completo; nunca distribuidas; sin revision humana)."),
        ("- La tabla cubre dev2 completo y el universo historico; no es "
         "una estimacion de la tasa de error del train."),
        "",
        (f"Duracion: {duracion} s. Inventario: "
         f"{SAL_INV.relative_to(RAIZ)} ({len(materiales)} archivos)."),
        "",
    ]
    SAL_AUD.write_text("\n".join(md), encoding="ascii")

    print("P101-00 auditoria de procedencia completada:")
    print(f"  tabla: {SAL_TAB.relative_to(RAIZ)} ({len(filas)} filas)")
    print(f"  inventario: {SAL_INV.relative_to(RAIZ)} "
          f"({len(materiales)} archivos)")
    print(f"  informe: {SAL_AUD.relative_to(RAIZ)}")
    print(f"  revisados (main+cal): {revisados_total} | reservados en "
          f"v2.1: {len(interseccion_reservados)} | elegibles: "
          f"{elegibles}/{len(dev2)}")
    print(f"  duracion: {duracion} s")


if __name__ == "__main__":
    main()
