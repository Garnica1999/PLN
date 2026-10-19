"""P92-01: preparacion del paquete de revision humana ciega (sin ejecutar la anotacion).

Fase P92 (human label review). NO entrena, NO modifica etiquetas, NO usa test/holdout,
NO genera predicciones. Construye:
- artifacts/human_annotation_v1/annotator_A.csv, annotator_B.csv, calibration_sample.csv,
  adjudication_template.csv (paquetes ciegos; sin labels ni scores);
- reports/p92_internal_sampling_key.csv (clave interna, FUERA del paquete);
- reports/p92_audit_results.json (seleccion, verificaciones y provenance para el informe).

Uso:
    /home/carlos/venv/bin/python experiments/p92_01_paquete_anotacion.py
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

RAIZ: Path = Path(__file__).resolve().parents[1]

RUTA_OOF = RAIZ / "features" / "oof_v03_final.npz"
RUTA_TRAIN = RAIZ / "data" / "raw" / "toxic_train.csv"
RUTA_PARTICION = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
RUTA_CANDIDATOS = RAIZ / "reports" / "p91_label_review_candidates.csv"

DIR_PAQUETE = RAIZ / "artifacts" / "human_annotation_v1"
RUTA_ANOT_A = DIR_PAQUETE / "annotator_A.csv"
RUTA_ANOT_B = DIR_PAQUETE / "annotator_B.csv"
RUTA_CALIB = DIR_PAQUETE / "calibration_sample.csv"
RUTA_ADJUD = DIR_PAQUETE / "adjudication_template.csv"
RUTA_CLAVE = RAIZ / "reports" / "p92_internal_sampling_key.csv"
RUTA_RESULTS = RAIZ / "reports" / "p92_audit_results.json"

HASH_OOF = "559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab"
HASH_TRAIN = "3d40e7345b588860cf4744914be88ddc094e4f03591516c8176ab40b7f76d42f"
HASH_PARTICION = "e5b62e5f22c4c63a8c43df4019969504d44c0baa5b503c19ba3560032f5ee6cd"
ETIQUETAS = ("HS", "TR", "AG")
UMBRALES = (0.44, 0.51, 0.46)
GUIA_VERSION = "v1"

SEMILLA_A = 20261010       # recuperada de P91 (verificada contra el CSV)
SEMILLA_B = 20261011       # seleccion del componente enriquecido
SEMILLA_CAL = 20261012     # seleccion de calibracion
SEMILLA_ID = 20261013      # asignacion de review_id opacos
SEMILLA_ORDEN_A = 20261015
SEMILLA_ORDEN_B = 20261016
CAP_B = 150
CAP_GRUPO = 2
N_CALIB = 15


def log(mensaje: str) -> None:
    """Imprime una linea de progreso."""
    print(f"[P92] {mensaje}", flush=True)


def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo."""
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def leer_csv(ruta: Path, encoding: str = "utf-8") -> list[dict[str, str]]:
    """Lee un CSV respetando saltos de linea embebidos."""
    with ruta.open(encoding=encoding, newline="") as fh:
        return list(csv.DictReader(fh))


def escribir_csv(ruta: Path, filas: list[dict[str, Any]]) -> None:
    """Escribe un CSV con terminador LF."""
    if not filas:
        raise SystemExit(f"STOP: {ruta.name} quedaria vacio")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(filas[0].keys()), lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)


def main() -> int:
    """Construye el paquete, la clave interna y las verificaciones."""
    for ruta in (RUTA_ANOT_A, RUTA_ANOT_B, RUTA_CALIB, RUTA_ADJUD, RUTA_CLAVE, RUTA_RESULTS):
        if ruta.exists():
            raise SystemExit(f"STOP: ya existe {ruta}")

    if sha256_de(RUTA_OOF) != HASH_OOF or sha256_de(RUTA_TRAIN) != HASH_TRAIN \
            or sha256_de(RUTA_PARTICION) != HASH_PARTICION:
        raise SystemExit("STOP: hashes canonicos inesperados")

    oof = np.load(RUTA_OOF, allow_pickle=True)
    ids = [int(i) for i in oof["ids"]]
    y = oof["y_true"].astype(np.int64)
    proba = oof["proba"].astype(np.float64)
    umbrales = np.asarray(oof["umbrales"], dtype=np.float64)
    dev = set(ids)
    idx = {i: k for k, i in enumerate(ids)}

    # ---------- P1: procedencia y elegibilidad ----------
    particion = {}
    with RUTA_PARTICION.open(encoding="utf-8-sig", newline="") as fh:
        for f in csv.DictReader(fh):
            particion[int(str(f["id"]).strip('"'))] = str(f["particion"]).strip()
    dev_part = {i for i, p in particion.items() if p == "dev"}
    holdout_part = {i for i, p in particion.items() if p == "holdout"}
    if dev_part != dev:
        raise SystemExit("STOP: la particion dev no coincide con el OOF canonico")

    textos: dict[int, str] = {}
    with RUTA_TRAIN.open(encoding="utf-8") as fh:
        for f in csv.DictReader(fh, delimiter=";"):
            ident = int(str(f["id"]).strip('"'))
            if ident in dev:
                textos[ident] = f["text"]
    if len(textos) != 1483:
        raise SystemExit("STOP: faltan textos de desarrollo")

    provenance = [
        {"source_path": "data/raw/toxic_train.csv", "partition": "train (dev + holdout)",
         "row_count": 1731, "unique_ids": 1731, "label_availability": "HS/TR/AG",
         "oof_available": "solo dev (1483)", "holdout_status": "248 filas NO elegibles (no usadas)",
         "test_status": "n/a", "eligible_for_review": 1483,
         "reason": "elegibilidad demostrada por id contra la particion congelada"},
        {"source_path": "data/processed/holdout_v2_ids.csv", "partition": "dev=1483 / holdout=248",
         "row_count": 1731, "unique_ids": 1731, "label_availability": "n/a (metadatos)",
         "oof_available": "n/a", "holdout_status": "holdout excluido por particion",
         "test_status": "n/a", "eligible_for_review": 1483,
         "reason": "manifiesto de particion congelado (hash verificado)"},
        {"source_path": "features/oof_v03_final.npz", "partition": "dev",
         "row_count": 1483, "unique_ids": 1483, "label_availability": "HS/TR/AG",
         "oof_available": "si (proba/pred)", "holdout_status": "no contiene holdout",
         "test_status": "n/a", "eligible_for_review": 1483,
         "reason": "interseccion exacta con la particion dev"},
        {"source_path": "data/raw/toxic_test.csv", "partition": "test",
         "row_count": "no inspeccionado", "unique_ids": "no inspeccionado",
         "label_availability": "ocultas", "oof_available": "no",
         "holdout_status": "n/a", "test_status": "PROHIBIDO: no accedido",
         "eligible_for_review": 0, "reason": "excluido por reglas"},
    ]

    # ---------- Componente A: recuperado de P91 y verificado contra el RNG ----------
    cand = leer_csv(RUTA_CANDIDATOS)
    ref_csv = sorted({int(f["example_id"]) for f in cand if f["in_reference_sample"] == "1"})
    rng_a = np.random.default_rng(SEMILLA_A)
    ref_rows = rng_a.choice(len(ids), size=100, replace=False)
    ref_recomputado = sorted(ids[int(i)] for i in ref_rows)
    if ref_csv != ref_recomputado or len(ref_csv) != 100:
        raise SystemExit("STOP: el componente A no se recupera de forma verificable")
    comp_a = ref_csv

    # ---------- Componente B: enriquecido desde la cola P91 ----------
    por_tweet: dict[int, dict[str, Any]] = {}
    for f in cand:
        eid = int(f["example_id"])
        t = por_tweet.setdefault(eid, {"reasons": set(), "labels": set(), "grupo": "",
                                       "max_margen": 0.0, "label_max_margen": "",
                                       "hcd_labels": set()})
        for r in f["reason_codes"].split("; "):
            t["reasons"].add(r)
        t["labels"].add(f["label"])
        if f["dup_group_id"]:
            t["grupo"] = f["dup_group_id"]
        m = abs(float(f["signed_margin"]))
        if m > t["max_margen"]:
            t["max_margen"] = m
            t["label_max_margen"] = f["label"]
        if "HIGH_CONFIDENCE_OOF_DISAGREEMENT" in f["reason_codes"]:
            t["hcd_labels"].add(f["label"])

    enriquecidos = sorted(eid for eid, t in por_tweet.items()
                          if t["reasons"] - {"RANDOM_REFERENCE_SAMPLE"} and eid not in set(comp_a))
    t1 = [e for e in enriquecidos if "POSSIBLE_INSTANCE_LABEL_INCONSISTENCY" in por_tweet[e]["reasons"]]
    t2 = [e for e in enriquecidos if "DOCUMENTED_RULE_CONFLICT" in por_tweet[e]["reasons"]
          and e not in set(t1)]
    t3 = [e for e in enriquecidos if "HIGH_CONFIDENCE_OOF_DISAGREEMENT" in por_tweet[e]["reasons"]
          and e not in set(t1) and e not in set(t2)]
    t4 = [e for e in enriquecidos if e not in set(t1) and e not in set(t2) and e not in set(t3)]

    rng_b = np.random.default_rng(SEMILLA_B)
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
    # T3 estratificado por etiqueta del margen maximo (round-robin determinista con RNG)
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
        raise SystemExit("STOP: componente B invalido")

    # ---------- Calibracion: 15 tweets fuera de A y B ----------
    resto = sorted(dev - set(comp_a) - set(comp_b))
    rng_c = np.random.default_rng(SEMILLA_CAL)
    cal_rows = rng_c.choice(len(resto), size=N_CALIB, replace=False)
    calib = [resto[int(i)] for i in cal_rows]

    # ---------- review_id opacos y ordenes independientes ----------
    principales = comp_a + comp_b
    rng_id = np.random.default_rng(SEMILLA_ID)
    perm_id = list(rng_id.permutation(len(principales)))
    mapa_id = {principales[int(p)]: f"P92R-{k:04d}" for k, p in enumerate(perm_id, start=1)}
    rng_oa = np.random.default_rng(SEMILLA_ORDEN_A)
    orden_a = [principales[int(i)] for i in rng_oa.permutation(len(principales))]
    rng_ob = np.random.default_rng(SEMILLA_ORDEN_B)
    orden_b = [principales[int(i)] for i in rng_ob.permutation(len(principales))]
    if orden_a == orden_b:
        raise SystemExit("STOP: los ordenes de los anotadores coinciden")
    rng_oc = np.random.default_rng(20261014)
    orden_cal = [calib[int(i)] for i in rng_oc.permutation(len(calib))]

    def fila_anotador(eid: int) -> dict[str, str]:
        return {"review_id": mapa_id[eid], "text": textos[eid],
                "HS_annotation": "", "TR_annotation": "", "AG_annotation": "",
                "uncertainty_notes": "", "guideline_version": GUIA_VERSION}

    escribir_csv(RUTA_ANOT_A, [fila_anotador(e) for e in orden_a])
    escribir_csv(RUTA_ANOT_B, [fila_anotador(e) for e in orden_b])
    mapa_cal = {eid: f"P92C-{k:02d}" for k, eid in enumerate(orden_cal, start=1)}
    escribir_csv(RUTA_CALIB, [{"review_id": mapa_cal[e], "text": textos[e],
                               "HS_annotation": "", "TR_annotation": "", "AG_annotation": "",
                               "uncertainty_notes": "", "guideline_version": GUIA_VERSION}
                              for e in orden_cal])
    filas_adj = []
    for e in principales:
        for et in ETIQUETAS:
            filas_adj.append({"review_id": mapa_id[e], "label": et, "annotator_A": "",
                              "annotator_B": "", "disagreement_type": "",
                              "adjudicator_decision": "", "adjudicator_rationale": "",
                              "adjudication_status": "PENDING_ANNOTATIONS",
                              "guideline_version": GUIA_VERSION})
    escribir_csv(RUTA_ADJUD, filas_adj)

    # ---------- clave interna ----------
    filas_clave = []
    for e in principales:
        t = por_tweet.get(e, {"reasons": set(), "grupo": ""})
        razones = "; ".join(sorted(t["reasons"])) if t["reasons"] else "RANDOM_REFERENCE_SAMPLE"
        filas_clave.append({
            "review_id": mapa_id[e], "original_example_id": e,
            "sample_component": "A" if e in set(comp_a) else "B",
            "selection_reason_codes": razones,
            "sampling_stratum": "referencia_aleatoria" if e in set(comp_a) else "enriquecido",
            "original_HS": int(y[idx[e], 0]), "original_TR": int(y[idx[e], 1]),
            "original_AG": int(y[idx[e], 2]),
            "dup_group_id": t["grupo"], "source_partition": "dev",
        })
    for e in orden_cal:
        filas_clave.append({
            "review_id": mapa_cal[e], "original_example_id": e, "sample_component": "CALIBRACION",
            "selection_reason_codes": "CALIBRATION_ROUND", "sampling_stratum": "calibracion",
            "original_HS": int(y[idx[e], 0]), "original_TR": int(y[idx[e], 1]),
            "original_AG": int(y[idx[e], 2]), "dup_group_id": "", "source_partition": "dev",
        })
    escribir_csv(RUTA_CLAVE, filas_clave)

    # ---------- verificaciones de integridad ----------
    a = leer_csv(RUTA_ANOT_A)
    b = leer_csv(RUTA_ANOT_B)
    checks = {
        "mismo_conjunto_A_B": sorted(f["review_id"] for f in a) == sorted(f["review_id"] for f in b),
        "orden_diferente": [f["review_id"] for f in a] != [f["review_id"] for f in b],
        "sin_repetidos_en_paquete": len({f["review_id"] for f in a}) == len(a) == 250,
        "anotaciones_vacias": all(
            f[c] == "" for f in a + b for c in ("HS_annotation", "TR_annotation", "AG_annotation", "uncertainty_notes")),
        "columnas_visibles_exactas": list(a[0].keys()) == ["review_id", "text", "HS_annotation",
                                                           "TR_annotation", "AG_annotation",
                                                           "uncertainty_notes", "guideline_version"],
        "sin_columnas_sensibles": not any(
            c in f for f in a for c in ("original_example_id", "original_labels", "proba", "fold",
                                        "reason_codes", "sample_component", "dup_group_id")),
        "textos_identicos_a_la_fuente": all(
            f["text"] == textos[int(k["original_example_id"])]
            for f, k in zip(sorted(a, key=lambda x: x["review_id"]),
                            sorted((k for k in filas_clave if k["sample_component"] != "CALIBRACION"),
                                   key=lambda x: x["review_id"]))),
        "review_ids_opacos": all(re.fullmatch(r"P92R-\d{4}", f["review_id"]) for f in a),
        "todos_en_dev": all(int(k["original_example_id"]) in dev for k in filas_clave),
        "sin_holdout_en_clave": not any(int(k["original_example_id"]) in holdout_part for k in filas_clave),
        "calibracion_disjunta": not (set(calib) & (set(comp_a) | set(comp_b))),
        "tamano_principal": len(principales) == 250,
        "plantilla_adjudicacion_750": len(filas_adj) == 750,
        "plantilla_sin_decisiones": all(f["adjudicator_decision"] == "" for f in filas_adj),
    }
    if not all(checks.values()):
        raise SystemExit(f"STOP: verificaciones fallidas: {[k for k, v in checks.items() if not v]}")

    # ---------- resultados ----------
    def resumen_etiquetas(conj: list[int]) -> dict[str, int]:
        c = Counter()
        for e in conj:
            for j, et in enumerate(ETIQUETAS):
                c[f"{et}_{int(y[idx[e], j])}"] += 1
        return dict(sorted(c.items()))

    resultados = {
        "audit_id": "P92-01",
        "guia_version": GUIA_VERSION,
        "hashes": {"oof": HASH_OOF, "train": HASH_TRAIN, "particion": HASH_PARTICION,
                   "candidatos_p91": sha256_de(RUTA_CANDIDATOS)},
        "provenance": provenance,
        "componente_A": {"tweets": len(comp_a), "celdas": len(comp_a) * 3,
                         "recuperado_verificado_rng": True, "semilla": SEMILLA_A},
        "componente_B": {"tweets": len(comp_b), "cap": CAP_B,
                         "disponibles_enriquecidos": len(enriquecidos),
                         "t1_pili": len(t1), "t2_drc": len(t2),
                         "t3_hcd_disponibles": len(t3), "t4_lsm_disponibles": len(t4),
                         "cap_grupo": CAP_GRUPO,
                         "seleccionados_por_tier": {
                             "pili": sum(1 for e in comp_b if "POSSIBLE_INSTANCE_LABEL_INCONSISTENCY" in por_tweet[e]["reasons"]),
                             "drc": sum(1 for e in comp_b if "DOCUMENTED_RULE_CONFLICT" in por_tweet[e]["reasons"]),
                             "hcd": sum(1 for e in comp_b if "HIGH_CONFIDENCE_OOF_DISAGREEMENT" in por_tweet[e]["reasons"]),
                             "lsm": sum(1 for e in comp_b if "LABEL_SPECIFIC_MARGIN" in por_tweet[e]["reasons"]),
                         },
                         "distribucion_etiquetas": resumen_etiquetas(comp_b)},
        "calibracion": {"tweets": len(calib), "semilla": SEMILLA_CAL,
                        "distribucion_etiquetas": resumen_etiquetas(calib)},
        "principal": {"tweets": len(principales), "celdas": len(principales) * 3,
                      "distribucion_etiquetas": resumen_etiquetas(principales),
                      "componente_A": len(comp_a), "componente_B": len(comp_b)},
        "semillas": {"A": SEMILLA_A, "B": SEMILLA_B, "calibracion": SEMILLA_CAL,
                     "review_id": SEMILLA_ID, "orden_A": SEMILLA_ORDEN_A, "orden_B": SEMILLA_ORDEN_B},
        "checks": checks,
    }
    RUTA_RESULTS.write_text(json.dumps(resultados, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    generados = [RUTA_ANOT_A, RUTA_ANOT_B, RUTA_CALIB, RUTA_ADJUD, RUTA_CLAVE, RUTA_RESULTS, Path(__file__)]
    sucios = [p.name for p in generados if any(b > 127 for b in p.read_bytes())]
    if sucios:
        raise SystemExit(f"STOP: bytes > 127 en {sucios}")

    arbol = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    puro = re.compile(r"^[\w./\\-]+\.\w{2,5}$|^[\w./\\-]+/$")
    docstrings = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            cuerpo = getattr(nodo, "body", [])
            if cuerpo and isinstance(cuerpo[0], ast.Expr) and isinstance(cuerpo[0].value, ast.Constant):
                if isinstance(cuerpo[0].value.value, str):
                    docstrings.add(id(cuerpo[0].value))
    permitidos = {"holdout_v2_ids.csv", "data/processed/holdout_v2_ids.csv",
                  "p91_label_review_candidates.csv", "p92_01_paquete_anotacion.py",
                  "toxic_test.csv", "data/raw/toxic_test.csv", "test.csv",
                  "leaderboard", "holdout", "submission", "submissions"}
    raros = [nodo.value for nodo in ast.walk(arbol)
             if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str)
             and id(nodo) not in docstrings and puro.match(nodo.value)
             and any(p in nodo.value.lower() for p in ("toxic_test", "test.csv", "leaderboard", "holdout", "submission"))
             and nodo.value not in permitidos]
    if raros:
        raise SystemExit(f"STOP: tokens de ruta inesperados: {sorted(set(raros))}")

    print(json.dumps({"principal": len(principales), "A": len(comp_a), "B": len(comp_b),
                      "calibracion": len(calib), "checks_ok": all(checks.values()),
                      "entrenamientos": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
