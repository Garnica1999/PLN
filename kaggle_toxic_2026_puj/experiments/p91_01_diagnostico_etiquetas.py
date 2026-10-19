"""P91-01: diagnostico computacional de calidad de etiquetas y cola de candidatos.

Fase P91 (label quality research). NO entrena modelos, NO genera predicciones nuevas,
NO modifica etiquetas. Usa exclusivamente artefactos OOF historicos y datos de desarrollo
elegibles (1483 filas dev; el holdout no se toca).

Produce:
- reports/p91_label_quality_diagnostics.csv (estadisticas por etiqueta y diagnostico);
- reports/p91_label_review_candidates.csv (cola trazable de candidatos a revision);
- reports/p91_audit_results.json (resultados para el informe).

Uso:
    /home/carlos/venv/bin/python experiments/p91_01_diagnostico_etiquetas.py
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
RUTA_V2BASE = RAIZ / "features" / "v2_oof_base_corrected.npz"
RUTA_CONSIST = RAIZ / "features" / "label_consistency_v4.npz"
RUTA_GRUPOS = RAIZ / "data" / "processed" / "grupos_near_duplicados.csv"
RUTA_PARTICION = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"

RUTA_DIAG = RAIZ / "reports" / "p91_label_quality_diagnostics.csv"
RUTA_CAND = RAIZ / "reports" / "p91_label_review_candidates.csv"
RUTA_RESULTS = RAIZ / "reports" / "p91_audit_results.json"

HASH_OOF = "559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab"
HASH_V2BASE = "27dfa83736b979518a94bc93e32bb9cae07bc82eaff92234fe9e15d1f9822774"
ETIQUETAS = ("HS", "TR", "AG")
UMBRALES = (0.44, 0.51, 0.46)
SEMILLA_MUESTRA = 20261010
N_MUESTRA_REFERENCIA = 100  # tweets; justificado en el protocolo (FPC + 3 etiquetas por tweet)
CAP_ENRIQUECIDO = 150       # tope por componente enriquecido (documentado)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso."""
    print(f"[P91] {mensaje}", flush=True)


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
    with ruta.open("w", encoding="utf-8", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(filas[0].keys()), lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)


def main() -> int:
    """Ejecuta el diagnostico y construye la cola de candidatos."""
    if sha256_de(RUTA_OOF) != HASH_OOF or sha256_de(RUTA_V2BASE) != HASH_V2BASE:
        raise SystemExit("STOP: hashes canonicos inesperados")

    oof = np.load(RUTA_OOF, allow_pickle=True)
    ids = [int(i) for i in oof["ids"]]
    y = oof["y_true"].astype(np.int64)
    proba = oof["proba"].astype(np.float64)
    pred = oof["pred"].astype(np.int64)
    pliegues = oof["pliegues"].astype(np.int64)
    umbrales = np.asarray(oof["umbrales"], dtype=np.float64)
    if len(ids) != 1483 or tuple(umbrales.tolist()) != UMBRALES:
        raise SystemExit("STOP: estructura del OOF inesperada")

    v2 = np.load(RUTA_V2BASE, allow_pickle=True)
    modelos_v2 = [str(m) for m in v2["modelos"]]
    orden = {int(i): k for k, i in enumerate(v2["ids"])}
    proba4 = v2["proba"][[orden[i] for i in ids], :, :].astype(np.float64)

    margen = proba - umbrales[None, :]
    distancia = np.abs(margen)
    error = pred != y
    n = len(ids)

    # ---------- A. Consistencia estructural ----------
    combos = Counter(tuple(int(v) for v in y[i]) for i in range(n))
    tr_sin_hs = int(((y[:, 1] == 1) & (y[:, 0] == 0)).sum())
    ag_sin_hs = int(((y[:, 2] == 1) & (y[:, 0] == 0)).sum())
    marginales = {et: int(y[:, j].sum()) for j, et in enumerate(ETIQUETAS)}

    # Duplicados: near-dup (solo) y fusionados, divergencia de labels en dev.
    dev = set(ids)
    g_nd: dict[int, int] = {}
    with RUTA_GRUPOS.open(encoding="utf-8-sig", newline="") as fh:
        for f in csv.DictReader(fh):
            ident = int(str(f["id"]).strip('"'))
            if ident in dev:
                g_nd[ident] = int(f["grupo"])
    g_fu: dict[int, int] = {}
    with RUTA_PARTICION.open(encoding="utf-8-sig", newline="") as fh:
        for f in csv.DictReader(fh):
            if str(f["particion"]).strip() == "dev":
                g_fu[int(str(f["id"]).strip('"'))] = int(f["grupo_near_dup_autor"])
    idx = {ident: k for k, ident in enumerate(ids)}

    def divergentes(mapa: dict[int, int]) -> tuple[list[int], int, int]:
        por_g: dict[int, list[int]] = {}
        for ident, g in mapa.items():
            por_g.setdefault(g, []).append(ident)
        grupos, filas = [], 0
        for g, miembros in sorted(por_g.items()):
            if len(miembros) < 2:
                continue
            etiquetas = [tuple(int(v) for v in y[idx[i]]) for i in miembros]
            if any(len({e[j] for e in etiquetas}) > 1 for j in range(3)):
                grupos.append(g)
                filas += len(miembros)
        return grupos, filas, len([g for g, m in por_g.items() if len(m) > 1])

    gdiv_nd, fdiv_nd, tot_nd = divergentes(g_nd)
    gdiv_fu, fdiv_fu, tot_fu = divergentes(g_fu)

    # Consistencia v4 (vecindad): discrepancia_etiqueta y en_cluster, filtrado a dev.
    cons = np.load(RUTA_CONSIST, allow_pickle=True)
    ids_c = [int(i) for i in cons["ids"]]
    disp_c = cons["discrepancia_etiqueta"].astype(np.int64)
    en_cluster = cons["en_cluster"].astype(np.int64)
    pos_dev = {i: k for k, i in enumerate(ids_c)}
    filas_dev = [pos_dev[i] for i in ids]
    disp_dev = disp_c[filas_dev, :]
    en_cluster_dev = int(en_cluster[filas_dev].sum())
    disp_por_etiqueta = {et: int(disp_dev[:, j].sum()) for j, et in enumerate(ETIQUETAS)}

    # ---------- B. Diagnostico por instancia-etiqueta ----------
    conteos: dict[str, dict[str, int]] = {}
    for j, et in enumerate(ETIQUETAS):
        pos = y[:, j] == 1
        neg = ~pos
        conteos[et] = {
            "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
            "fn_p_le_0.05": int((pos & (proba[:, j] <= 0.05)).sum()),
            "fn_p_le_0.10": int((pos & (proba[:, j] <= 0.10)).sum()),
            "fp_p_ge_0.95": int((neg & (proba[:, j] >= 0.95)).sum()),
            "fp_p_ge_0.90": int((neg & (proba[:, j] >= 0.90)).sum()),
            "cerca_threshold": int((distancia[:, j] < 0.025).sum()),
            "errores": int(error[:, j].sum()),
        }

    # Desacuerdo multi-modelo (M1/M3/M4 a 0.5; M2 con umbrales canonicos, como el historial).
    preds4 = np.zeros((n, 4, 3), dtype=np.int64)
    for j in range(3):
        preds4[:, 0, j] = (proba4[:, 0, j] >= 0.5).astype(np.int64)   # M1 word
        preds4[:, 1, j] = pred[:, j]                                  # M2 union (canonica)
        preds4[:, 2, j] = (proba4[:, 2, j] >= 0.5).astype(np.int64)   # M3 svc
        preds4[:, 3, j] = (proba4[:, 3, j] >= 0.5).astype(np.int64)   # M4 char
    desacuerdo = (preds4 != y[:, None, :]).sum(axis=1)  # (n,3): cuantos de 4 discrepan
    unanimidad = {et: int((desacuerdo[:, j] == 4).sum()) for j, et in enumerate(ETIQUETAS)}

    # ---------- C. Diagnosticos CSV ----------
    filas_diag: list[dict[str, Any]] = []

    def diag(et: str, nombre: str, valor: int, num: int, den: int, unidad: str,
             fuente: str, limitacion: str) -> None:
        filas_diag.append({"label": et, "diagnostic": nombre, "value": valor,
                           "numerator": num, "denominator": den, "unit": unidad,
                           "source_artifact": fuente, "limitations": limitacion})

    for j, et in enumerate(ETIQUETAS):
        c = conteos[et]
        diag(et, "marginal_positive", c["n_pos"], c["n_pos"], n, "tweets", "v03_error_casebook",
             "gold del desarrollo; sin juicio de calidad")
        diag(et, "oof_error_cells", c["errores"], c["errores"], c["n_pos"] + c["n_neg"], "celdas",
             "p90_02_error_diagnosis", "error del modelo, no de la etiqueta")
        diag(et, "positive_with_strong_negative_oof_evidence_p_le_0.05", c["fn_p_le_0.05"],
             c["fn_p_le_0.05"], c["n_pos"], "celdas", "oof_v03_final",
             "candidato a revision; la evidencia OOF no es prueba de error de anotacion")
        diag(et, "positive_with_negative_oof_evidence_p_le_0.10", c["fn_p_le_0.10"],
             c["fn_p_le_0.10"], c["n_pos"], "celdas", "oof_v03_final", "idem")
        diag(et, "negative_with_strong_positive_oof_evidence_p_ge_0.95", c["fp_p_ge_0.95"],
             c["fp_p_ge_0.95"], c["n_neg"], "celdas", "oof_v03_final", "idem")
        diag(et, "negative_with_positive_oof_evidence_p_ge_0.90", c["fp_p_ge_0.90"],
             c["fp_p_ge_0.90"], c["n_neg"], "celdas", "oof_v03_final", "idem")
        diag(et, "cells_near_threshold_abs_margin_lt_0.025", c["cerca_threshold"],
             c["cerca_threshold"], n, "celdas", "oof_v03_final",
             "proximidad a la decision; no es probabilidad de error de anotacion")
        diag(et, "cells_with_unanimous_4model_disagreement", unanimidad[et],
             unanimidad[et], n, "celdas", "v2_oof_base_corrected",
             "M1/M3/M4 a umbral 0.5 y M2 canonico (convencion historica); desacuerdo != error")
        diag(et, "cells_with_neighbor_label_discrepancy", disp_por_etiqueta[et],
             disp_por_etiqueta[et], n, "celdas", "label_consistency_v4",
             "cluster por similitud (4 criterios); similitud textual != identidad semantica")

    diag("ALL", "tr_positive_with_hs_0", tr_sin_hs, tr_sin_hs, n, "tweets", "toxic_train",
         "identidad empirica del desarrollo; la regla formal no esta documentada")
    diag("ALL", "ag_positive_with_hs_0", ag_sin_hs, ag_sin_hs, n, "tweets", "toxic_train",
         "excepciones a la lectura condicionada de AG; estado formal sin documentar")
    diag("ALL", "cells_in_consistency_clusters", en_cluster_dev, en_cluster_dev, n, "tweets",
         "label_consistency_v4", "4.853 por ciento del train (P87); dev aqui")
    diag("ALL", "near_dup_groups_divergent", len(gdiv_nd), len(gdiv_nd), tot_nd, "grupos",
         "grupos_near_duplicados", "umbral 0.85; solo texto casi duplicado")
    diag("ALL", "near_dup_rows_divergent", fdiv_nd, fdiv_nd, n, "filas",
         "grupos_near_duplicados", "idem")
    diag("ALL", "fused_groups_divergent", len(gdiv_fu), len(gdiv_fu), tot_fu, "grupos",
         "holdout_v2_ids", "grupos fusionados (near-dup + pseudo-autor)")
    diag("ALL", "fused_rows_divergent", fdiv_fu, fdiv_fu, n, "filas", "holdout_v2_ids", "idem")
    for combo, cuenta in sorted(combos.items()):
        diag("ALL", f"combo_HS{combo[0]}_TR{combo[1]}_AG{combo[2]}", cuenta, cuenta, n,
             "tweets", "toxic_train", "estructura observada")
    escribir_csv(RUTA_DIAG, filas_diag)
    log(f"diagnosticos: {len(filas_diag)} filas")

    # ---------- D. Cola de candidatos ----------
    rng = np.random.default_rng(SEMILLA_MUESTRA)
    muestra_ref = set(int(v) for v in rng.choice(n, size=N_MUESTRA_REFERENCIA, replace=False))

    candidatos: dict[tuple[int, str], dict[str, Any]] = {}

    def anotar(i: int, j: int, razon: str, prioridad: int, nota: str = "") -> None:
        clave = (int(ids[i]), ETIQUETAS[j])
        fila = candidatos.setdefault(clave, {
            "review_id": "", "example_id": int(ids[i]), "label": ETIQUETAS[j],
            "gold_label": int(y[i, j]), "oof_probability": f"{float(proba[i, j]):.6f}",
            "threshold": f"{umbrales[j]:.2f}", "signed_margin": f"{float(margen[i, j]):.6f}",
            "abs_margin": f"{float(distancia[i, j]):.6f}",
            "oof_error_type": ("FP" if y[i, j] == 0 else "FN") if error[i, j] else "",
            "fold": int(pliegues[i]), "reason_codes": [], "n_models_disagreeing": int(desacuerdo[i, j]),
            "dup_group_id": "", "in_reference_sample": int(i in muestra_ref),
            "in_enriched": 0, "priority": prioridad, "evidence_notes": [],
            "annotation_status": "pending_authorization",
            "package_status": "not_generated_governance_pending",
        })
        if razon not in fila["reason_codes"]:
            fila["reason_codes"].append(razon)
        fila["priority"] = min(fila["priority"], prioridad)
        if nota and nota not in fila["evidence_notes"]:
            fila["evidence_notes"].append(nota)

    # E1: desacuerdo unanime de los 4 modelos (prioridad 1).
    for j, et in enumerate(ETIQUETAS):
        sel = [i for i in range(n) if desacuerdo[i, j] == 4]
        sel.sort(key=lambda i: -distancia[i, j])
        for i in sel[:CAP_ENRIQUECIDO]:
            anotar(i, j, "HIGH_CONFIDENCE_OOF_DISAGREEMENT", 1,
                   "los 4 modelos historicos discrepan del gold")

    # E2: evidencia extrema de un solo modelo (prioridad 2).
    for j, et in enumerate(ETIQUETAS):
        sel = [i for i in range(n)
               if (y[i, j] == 1 and proba[i, j] <= 0.05) or (y[i, j] == 0 and proba[i, j] >= 0.95)]
        sel.sort(key=lambda i: -distancia[i, j])
        for i in sel[:CAP_ENRIQUECIDO]:
            anotar(i, j, "HIGH_CONFIDENCE_OOF_DISAGREEMENT", 2,
                   "probabilidad OOF extrema contra el gold (un modelo)")

    # E3: filas en grupos divergentes (prioridad 1 near-dup; 2 fusionado).
    for g in gdiv_nd:
        for ident, gg in g_nd.items():
            if gg == g:
                i = idx[ident]
                for j in range(3):
                    anotar(i, j, "POSSIBLE_INSTANCE_LABEL_INCONSISTENCY", 1,
                           f"grupo near-dup {g} con labels divergentes")
                    candidatos[(ident, ETIQUETAS[j])]["dup_group_id"] = f"near_dup:{g}"
    for g in gdiv_fu:
        for ident, gg in g_fu.items():
            if gg == g:
                i = idx[ident]
                for j in range(3):
                    anotar(i, j, "POSSIBLE_INSTANCE_LABEL_INCONSISTENCY", 2,
                           f"grupo fusionado {g} con labels divergentes")
                    if not candidatos[(ident, ETIQUETAS[j])]["dup_group_id"]:
                        candidatos[(ident, ETIQUETAS[j])]["dup_group_id"] = f"fused:{g}"

    # E4: AG=1 con HS=0 (prioridad 3; lectura condicionada no documentada formalmente).
    for i in range(n):
        if y[i, 2] == 1 and y[i, 0] == 0:
            anotar(i, 2, "DOCUMENTED_RULE_CONFLICT", 3,
                   "AG=1 con HS=0: excepcion a la lectura condicionada (regla formal sin documentar)")

    # E5: errores cerca del threshold (prioridad 3).
    for j, et in enumerate(ETIQUETAS):
        sel = [i for i in range(n) if error[i, j] and distancia[i, j] < 0.025]
        for i in sel:
            anotar(i, j, "LABEL_SPECIFIC_MARGIN", 3, "error de V03 en la frontera del umbral")

    # Muestra de referencia: 100 tweets, las 3 etiquetas (prioridad 3).
    for i in sorted(muestra_ref):
        for j in range(3):
            anotar(i, j, "RANDOM_REFERENCE_SAMPLE", 3,
                   "muestra aleatoria de referencia (semilla fija)")

    filas_cand = []
    for k, (clave, fila) in enumerate(sorted(candidatos.items(), key=lambda kv: (kv[1]["priority"], kv[0])), start=1):
        fila["reason_codes"] = "; ".join(fila["reason_codes"])
        fila["evidence_notes"] = "; ".join(fila["evidence_notes"])
        fila["in_enriched"] = int(not (fila["reason_codes"] == "RANDOM_REFERENCE_SAMPLE"))
        fila["review_id"] = f"P91R-{k:04d}"
        filas_cand.append(fila)
    escribir_csv(RUTA_CAND, filas_cand)
    razones = Counter()
    for f in filas_cand:
        for r in f["reason_codes"].split("; "):
            razones[r] += 1
    log(f"cola: {len(filas_cand)} celdas | razones: {dict(razones)}")

    resultados = {
        "audit_id": "P91-01",
        "hashes": {"oof": HASH_OOF, "v2base": HASH_V2BASE,
                   "consistencia": sha256_de(RUTA_CONSIST), "grupos": sha256_de(RUTA_GRUPOS),
                   "particion": sha256_de(RUTA_PARTICION)},
        "marginales": marginales,
        "combos": {f"{k}": v for k, v in sorted(combos.items())},
        "tr_sin_hs": tr_sin_hs, "ag_sin_hs": ag_sin_hs,
        "conteos_por_etiqueta": conteos,
        "unanimidad_4modelos": unanimidad,
        "desacuerdo_histograma": {f"{k}": int((desacuerdo == k).sum()) for k in range(5)},
        "vecindad": {"en_cluster_dev": en_cluster_dev, "discrepancia_por_etiqueta": disp_por_etiqueta},
        "duplicados": {"near_dup_grupos_div": len(gdiv_nd), "near_dup_filas_div": fdiv_nd,
                       "near_dup_grupos_no_unitarios": tot_nd,
                       "fused_grupos_div": len(gdiv_fu), "fused_filas_div": fdiv_fu,
                       "fused_grupos_no_unitarios": tot_fu},
        "cola": {"celdas": len(filas_cand), "razones": dict(razones),
                 "muestra_referencia_tweets": len(muestra_ref), "semilla": SEMILLA_MUESTRA,
                 "cap_enriquecido_por_componente": CAP_ENRIQUECIDO},
    }
    RUTA_RESULTS.write_text(json.dumps(resultados, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    generados = [RUTA_DIAG, RUTA_CAND, RUTA_RESULTS, Path(__file__)]
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
    raros = [nodo.value for nodo in ast.walk(arbol)
             if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str)
             and id(nodo) not in docstrings and puro.match(nodo.value)
             and any(p in nodo.value.lower() for p in ("toxic_test", "test.csv", "leaderboard", "holdout", "submission"))
             and nodo.value not in {"holdout_v2_ids.csv", "data/processed/holdout_v2_ids.csv",
                                    "p91_01_diagnostico_etiquetas.py",
                                    "toxic_test", "test.csv", "leaderboard", "holdout", "submission"}]
    if raros:
        raise SystemExit(f"STOP: tokens de ruta inesperados: {sorted(set(raros))}")

    print(json.dumps({"diagnosticos": len(filas_diag), "candidatos": len(filas_cand),
                      "razones": dict(razones), "entrenamientos": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
