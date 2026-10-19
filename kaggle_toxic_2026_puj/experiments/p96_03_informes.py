#!/usr/bin/env python
"""
P96-03: informes forenses (P11/P13) y verificacion estadistica de gates (P8).

Solo lectura y computo determinista; no entrena nada. Genera:

- reports/p96_code_and_metric_findings.csv (hallazgos clasificados);
- reports/p96_reproduction_candidates.csv (reproducciones potenciales);
- reports/p96_manifest.json (hashes y hechos de la fase).

Ademas verifica la coherencia de la incertidumbre de las comparaciones
pareadas V4: IC95 por bootstrap de filas (src.evaluation) frente a bootstrap
por clusters near-duplicado/pseudo-autor, sobre predicciones ya guardadas.

Uso:
    /home/carlos/venv/bin/python experiments/p96_03_informes.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.evaluation import (  # noqa: E402
    bootstrap_diferencia_pareada,
    metrica_macro,
    metricas_multietiqueta,
)

SAL_FINDINGS = RAIZ / "reports" / "p96_code_and_metric_findings.csv"
SAL_REPRO = RAIZ / "reports" / "p96_reproduction_candidates.csv"
SAL_MANIFEST = RAIZ / "reports" / "p96_manifest.json"
UMB = np.array([0.44, 0.51, 0.46])

CAB_FINDINGS = ["finding_id", "severity", "classification", "area",
                "affected_experiments", "affected_results", "evidence",
                "code_location", "artifact_hashes", "impact", "confidence",
                "whether_historical_conclusion_changes", "minimal_safe_resolution",
                "reproduction_required"]

FINDINGS = [
    ["F96-01", "P4", "NO_DEFECT_FOUND_IN_AUDITED_SCOPE", "registro",
     "todas (459 filas)", "n/a",
     "459 filas, 459 ids unicos, 0 ids vacios, 0 duplicados de id (recomputado en esta fase)",
     "experiments/p96_00_registro_audit.py", "n/a", "A", "alta", "no",
     "ninguna (estructura del registro correcta)", "no"],
    ["F96-02", "P3", "PROVENANCE_GAP", "registro/trazabilidad",
     "25 filas sin script real (7 vacias + 18 con referencias textuales)",
     "n/a",
     "Las 25 filas son auditorias, envios, demostraciones o verificaciones en linea; "
     "no hay ninguna fila de modelado sin codigo",
     "reports/p96_experiment_integrity_matrix.csv", "n/a", "A", "alta", "no",
     "nota documental; no requiere accion", "no"],
    ["F96-03", "P4", "NO_DEFECT_FOUND_IN_AUDITED_SCOPE", "registro/configuracion",
     "PRE-00..PRE-09; V31-00..V31-04; V28-04..V40-02 (integracion)",
     "n/a",
     "7 grupos comparten repr+modelo+hiper pero difieren en preprocessing o "
     "feature_groups (transformaciones deterministas, features estructurales, "
     "integraciones); no son el mismo pipeline ejecutado dos veces",
     "experiments.csv", "n/a", "A", "media", "no",
     "nota: el registro no distingue 'run' de 'experimento'; usar la columna script",
     "no"],
    ["F96-04", "P3", "PROVENANCE_GAP", "folds",
     "V2-BASE-01..04; oof_v03_final",
     "numeracion de pliegues",
     "Particion identica (biyeccion exacta 1483/1483 filas: 0->1 ... 4->5) entre "
     "src.splits (0..4) y artefactos historicos (1..5); v2_oof_base_corrected no "
     "guarda 'pliegues' (solo grupos)",
     "src/splits.py; experiments/23_candidatos_finales.py (enumerate start=1)",
     "v2_oof_base_corrected.npz; oof_v03_final.npz", "A", "alta", "no",
     "nota en documentacion; el recomputo debe traducir numeracion", "no"],
    ["F96-05", "P2", "COMPARABILITY_PROBLEM", "metrica/umbrales",
     "V03 (produccion) y comparaciones v2",
     "umbrales 0.44/0.51/0.46",
     "Los umbrales de V03 se optimizaron in-sample sobre el mismo OOF en que se "
     "reporta la metrica historica (aprobados por la regla pre-registrada +0.005 "
     "y >=3/5 pliegues); optimismo medido: re-optimizar sobre el dev corregido da "
     "0.780840 vs 0.777211 congelado (+0.003629). Caveat D-126 vigente; todas las "
     "comparaciones posteriores usan umbrales congelados",
     "experiments/04_refit_submissions_v02_v03.py; 23_candidatos_finales.py",
     "oof_v03_final.npz", "B", "alta", "no",
     "mantener; en cualquier reapertura usar umbrales anidados por pliegue", "no"],
    ["F96-06", "P2", "COMPARABILITY_PROBLEM", "estadistica/gates",
     "v5 (embedding_03/05/08/10/12/13), v4 (27/28/35), p84_01..04",
     "IC95 de comparaciones pareadas",
     "bootstrap_diferencia_pareada (src) remuestrea filas, no clusters "
     "near-dup/pseudo-autor; con grupos dependientes los IC quedan estrechos. "
     "Verificacion P96 (ver manifest): el IC por clusters de C_meta36_downweight "
     "vs V03 se recomputo desde artefactos y CONFIRMA la direccion del cierre",
     "src/evaluation.py (bootstrap_diferencia_pareada); implementaciones por "
     "clusters en experiments/p73_02, p73_03, p74_01, 27_label_consistency_v4",
     "oof_candidates_v4.npz; v2_oof_base_corrected.npz", "C", "alta", "no",
     "usar bootstrap de clusters en cualquier gate futuro", "no"],
    ["F96-07", "P3", "PROVENANCE_GAP", "registro/metrica",
     "registro completo (columna f1)",
     "semantica de la columna f1",
     "La columna f1 mezcla convenciones (0.5, umbrales congelados, postproceso, "
     "historica vs corregida) segun la fase; D-128 ya lo documenta; muestreo "
     "P96 confirma (V03 f1=0.7718 a 0.5 vs corrected 0.7772 con umbrales)",
     "experiments.csv; DECISIONS.md D-128", "n/a", "B", "alta", "no",
     "usar reportes/JSON por experimento como fuente; no reescribir el registro", "no"],
    ["F96-08", "P4", "NO_DEFECT_FOUND_IN_AUDITED_SCOPE", "transferencia",
     "V32-05; V33-01", "submission byte-identica",
     "sha256 de toxic_submission_v32_05_url_length.csv == "
     "toxic_submission_v03_lr_union_thr.csv == copia congelada (mismo archivo); "
     "el candidato no cambio ninguna decision en test",
     "experiments/v32_05_cuatro_bins.py; v33_01_validacion.py",
     "submissions/toxic_submission_v32_05_url_length.csv", "B", "alta", "no",
     "conservar como caso documentado de no-transferencia; no requiere accion", "no"],
    ["F96-09", "P3", "NO_DEFECT_FOUND_IN_AUDITED_SCOPE", "particiones reservadas",
     "v10+ (gobernanza); artefactos v2/v4 con 1731 filas",
     "holdout/holdout2",
     "P96 no accedio a contenidos del holdout ni de holdout2; 5 artefactos que "
     "cubren filas del holdout (label_consistency_v4, oof_bases, oof_ens04_v04, "
     "oof_mod03b, oof_v03_historical) quedaron EXCLUIDOS del recomputo por "
     "politica; en el codigo auditado no se encontro ajuste sobre test ni uso "
     "indebido del holdout (5 evaluaciones agotadas y holdout2 0/3 documentados)",
     "audit/out/p04_leakage_checks.csv (auditoria previa); scans P96",
     "n/a", "C", "media", "no",
     "mantener la gobernanza; alcance de la comprobacion limitado al codigo/artefactos revisados",
     "no"],
    ["F96-10", "P4", "PROVENANCE_GAP", "notebook",
     "notebooks/10_final_pipeline.ipynb",
     "salidas ejecutadas",
     "27 celdas con salidas de una ejecucion en worktree /home/carlos/Git/PLN-v10 "
     "(wave v9/v10; evidencia v10_00_verificacion.json); el resto de secciones no "
     "tiene salidas; no se detecto conflicto numerico en el muestreo",
     "audit_fixes/fix_b14_notebook.py", "notebooks/10_final_pipeline.ipynb",
     "A", "alta", "no", "declarar en el propio notebook (ya cubierto por secciones "
     "de verificacion); no editar salidas historicas", "no"],
    ["F96-11", "P4", "PROVENANCE_GAP", "artefactos P84-04",
     "P84-04 (semilla 20261016, brazos M/R/S, modo humo)",
     "parciales no registrados",
     "reports/p84_04_parciales/* son staging de corridas por semilla "
     "('20261016' es la semilla, no una fecha); el resultado agregado vive en "
     "features/p84_04_oof_emb.npz y el informe p84_04_sgns.md; los parciales no "
     "aparecen en experiments.csv (correcto: no son experimentos)",
     "experiments/p84_04_sgns_corregido_anclado.py", "reports/p84_04_parciales/",
     "A", "media", "no", "nota documental; conservar", "no"],
    ["F96-12", "P3", "PROVENANCE_GAP", "artefactos auxiliares",
     "p84_01_oof_mlp_previo_fix; *_humo; v2_oof_targeting..emocion",
     "propuestas de limpieza",
     "Existen artefactos de intermediates (_previo_fix, _humo) junto a los "
     "oficiales; los oficiales recomputan los valores documentados (P96 Parte C); "
     "los auxiliares deben conservarse sin promocion",
     "features/*.npz", "varios", "A", "media", "no",
     "no borrar; usar siempre el artefacto con verificacion documentada", "no"],
    ["F96-13", "P2", "COMPARABILITY_PROBLEM", "estadistica/gates",
     "V25/V26/V43/V44; gate del proyecto",
     "dOOF > +0.0376",
     "El +0.0376 proviene de la regresion dLB~dOOF sobre 19 observaciones "
     "(intercepto -0.0092; 'punto donde la recta cruza dLB=0'), reexpresado como "
     "gate por gobernanza (D-291); no es una constante universal; su aplicacion "
     "historica fue coherente (los cierres usaron tambien neto de celdas y "
     "consistencia por pliegue)",
     "reports/v25_05_dictamen_final.md; experiments/p86_00_auditoria_novedad_fm.py",
     "n/a", "D", "alta", "no",
     "mantener como gate vigente; documentar su origen en cualquier reapertura", "no"],
    ["F96-14", "P4", "NO_DEFECT_FOUND_IN_AUDITED_SCOPE", "metrica",
     "02_baseline_tfidf_lr.py; 14_oof_corregido_v2.py",
     "copias historicas de la metrica",
     "El script 02 conserva su propia copia de metricas_multietiqueta (congelada "
     "como evidencia) y el 14 importa la de src; la equivalencia esta declarada en "
     "el docstring de src/evaluation.py y no se detecto desviacion",
     "src/evaluation.py (docstring de trazabilidad)", "n/a", "A", "alta", "no",
     "ninguna", "no"],
    ["F96-15", "P4", "PROVENANCE_GAP", "registro",
     "V10-00; P60-01", "verificaciones sin script local",
     "V10-00 registra 'verificacion en linea; evidencia en "
     "reports/v10_00_verificacion.json'(worktree) y P60-01 es una fila de "
     "literatura con metrica; ambas intencionales",
     "reports/v10_00_verificacion.json", "n/a", "A", "alta", "no",
     "ninguna", "no"],
    ["F96-16", "P4", "NO_DEFECT_FOUND_IN_AUDITED_SCOPE", "recomputacion",
     "20 comparaciones curadas + 159 recomputos de variantes",
     "integralidad de resultados",
     "20/20 comparaciones curadas contra valores documentados reproducen "
     "(A_cost exacto a 3.2e-11; V4 A-D, meta_shift x4 x2 protocolos, P84-01 "
     "semillas 42/43/44, P84-02 control x2, P84-03 control=V03, V03 canonico); "
     "folds y grupos reproducen exactamente; y_true alineado con etiquetas "
     "originales",
     "experiments/p96_01_metric_recompute.py; experiments/p96_02_folds_leakage.py",
     "reports/p96_reproducibility_matrix.csv", "E", "alta", "no",
     "ninguna: el benchmark interno queda verificado en el alcance auditado", "no"],
    ["F96-17", "P2", "COMPARABILITY_PROBLEM", "estadistica/gates",
     "V4: 36_candidatos_v4.py (A/B/C/D); D-016 aplicado en esa fase",
     "criterio IC95 pareado > 0",
     "El criterio D-016 de la fase V4 se evaluo con _bootstrap() sobre un proxy "
     "(delta del acuerdo por ejemplo, tipo exactitud), NO sobre el delta de la "
     "metrica primaria corrected_cv: cumple_ic = ic_bajo > 0 usaba el IC del proxy. "
     "Recomputo P96 con la metrica primaria (filas / clusters): A "
     "[+0.000192,+0.016189] / [+0.000347,+0.016311] (proxy [-0.0002,+0.0117]); "
     "B [-0.004135,+0.014312] / [-0.003715,+0.014602] (proxy [+0.0031,+0.0166]); "
     "C [+0.000726,+0.021142] / [+0.000466,+0.021360] (proxy [+0.0054,+0.0200]); "
     "D [+0.003164,+0.023957] / [+0.002303,+0.023833] (proxy [-0.0011,+0.0148]). "
     "Bajo la metrica primaria A, C y D superarian nominalmente el criterio de IC "
     "y B no: la frase del registro 'el unico que superaba los tres criterios de "
     "D-016 a la vez era C' es dependiente del estimando",
     "experiments/36_candidatos_v4.py lineas 221-244 y 305-309; "
     "src/evaluation.py (bootstrap_diferencia_pareada)",
     "oof_candidates_v4.npz", "B/C", "alta",
     "no sobre la adopcion (ninguno fue adoptado; C rechazado ademas por el "
     "holdout y A por su sonda externa); si sobre la frase 'unico C pasaba "
     "D-016', que debe leerse como dependiente del estimando",
     "para cualquier gate futuro: IC bootstrap del delta de la METRICA PRIMARIA, "
     "por clusters; nota documental en DECISIONS (sin editar el historico)", "no"],
]

CAB_REPRO = ["experiment_id", "confirmed_issue", "historical_result",
             "expected_effect_of_correction", "minimum_reproduction",
             "artifacts_needed", "risk_of_leakage", "resource_cost",
             "whether_baseline_comparison_changes", "recommendation"]

REPRO = [
    ["REP-96-01 (condicional, V6)",
     "PROVENANCE_GAP verificado (F96-xx): los controles de V6 se verificaron por "
     "deltas JSON (0.0) y por la equivalencia P0b de la auditoria previa, pero no "
     "existen matrices OOF V6 para recomputo independiente",
     "V6-02..V6-08: variantes entre -0.0107 y -0.0008 con controles en 0.0",
     "Ninguno esperado: el cierre de V6 no depende de la numeracion ni del formato; "
     "la reproduccion solo cerraria el hueco de provenance",
     "Re-ejecutar v6_02 (label factorization) con pliegues canonicos y umbrales "
     "congelados; comparar control contra 0.7772114525165721",
     "experiments/v6_02_label_factorization.py; data/raw/toxic_train.csv, "
     "grupos_near_duplicados.csv",
     "bajo (mismos pliegues, sin test)", "~1-2 min CPU",
     "no cambia la comparacion contra V03",
     "NO ejecutar ahora; solo si una futura reapertura usa evidencia de V6"],
    ["REP-96-02 (condicional, IC por clusters)",
     "F96-06: los IC pareados de las fases v4/v5 usaron bootstrap de filas; la "
     "version por clusters es recomputable SIN entrenamiento desde artefactos "
     "(P96 la ejecuto para C_meta36: ver manifest)",
     "C_meta36_downweight: IC filas [+0.0054, +0.0200] (documentado); la version "
     "por clusters se verifico en P96 y no cambia la direccion del cierre",
     "Ninguno esperado; solo homogeneiza el metodo de incertidumbre",
     "Reutilizar el bloque de clusters de P96-03 sobre cualquier candidato futuro "
     "antes de adoptar un gate",
     "features/oof_candidates_v4.npz; v2_oof_base_corrected.npz (grupos)",
     "bajo", "segundos CPU", "no",
     "adoptar cluster bootstrap en gates futuros; no requiere reproduccion"],
]


def sha256(ruta: Path) -> str:
    """sha256 completo en hexadecimal."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()


def bootstrap_clusters(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                       grupos: np.ndarray, repeticiones: int = 2000,
                       semilla: int = 20260925) -> dict[str, object]:
    """IC95 de la diferencia pareada remuestreando clusters (near-dup/autores)."""
    generador = np.random.default_rng(semilla)
    indices_por_cluster: dict[int, np.ndarray] = {}
    for valor in np.unique(grupos):
        indices_por_cluster[int(valor)] = np.flatnonzero(grupos == valor)
    claves = np.array(sorted(indices_por_cluster))
    n_clusters = len(claves)
    diferencias = np.empty(repeticiones, dtype=float)
    for r in range(repeticiones):
        muestra = generador.integers(0, n_clusters, n_clusters)
        indices = np.concatenate([indices_por_cluster[int(claves[i])] for i in muestra])
        diferencias[r] = metrica_macro(y[indices], pred_a[indices]) - \
            metrica_macro(y[indices], pred_b[indices])
    cuantiles = np.quantile(diferencias, [0.025, 0.5, 0.975])
    return {"ic95": [round(float(cuantiles[0]), 6), round(float(cuantiles[2]), 6)],
            "mediana": round(float(cuantiles[1]), 6), "repeticiones": repeticiones}


def main() -> None:
    """Genera hallazgos, candidatos y manifest; ejecuta la verificacion P8."""
    rehash = "--rehash" in sys.argv
    escribir_findings = not SAL_FINDINGS.exists()
    escribir_repro = not SAL_REPRO.exists()
    escribir_manifest = rehash or not SAL_MANIFEST.exists()
    if not escribir_findings and not escribir_repro and not escribir_manifest:
        raise SystemExit("Nada que generar (usa --rehash para actualizar el manifest).")

    print("=== Verificacion P8: incertidumbre pareada filas vs clusters ===")
    z4 = np.load(RAIZ / "features" / "oof_candidates_v4.npz", allow_pickle=True)
    zb = np.load(RAIZ / "features" / "v2_oof_base_corrected.npz", allow_pickle=True)
    ids4 = z4["ids"].astype(np.int64)
    idsb = zb["ids"].astype(np.int64)
    mismo_orden = bool(np.array_equal(ids4, idsb))
    print(f"  mismos ids y orden que v2_oof_base_corrected: {mismo_orden}")
    grupos = zb["grupos"].astype(np.int64)
    y = z4["y_true"].astype(int)
    pred_v03 = (zb["proba"][:, [str(m) for m in zb['modelos']].index(
        "M2_union_lr_c2"), :] >= UMB).astype(int)
    resultados: dict[str, object] = {}
    for nombre in ("A_cost_sensitive_downweight", "B_meta36",
                   "C_meta36_downweight", "D_svd300_rbf"):
        pred = (z4[nombre] >= UMB).astype(int)
        filas = bootstrap_diferencia_pareada(y, pred, pred_v03, repeticiones=2000)
        clusters = bootstrap_clusters(y, pred, pred_v03, grupos)
        resultados[nombre] = {"ic_filas": filas["ic95"],
                              "ic_clusters": clusters["ic95"],
                              "delta": filas["diferencia_observada"]}
        print(f"  {nombre:28s} delta={filas['diferencia_observada']:+.6f} "
              f"IC filas={filas['ic95']} IC clusters={clusters['ic95']}")

    print()
    print("=== Generacion de informes ===")
    if escribir_findings:
        with open(SAL_FINDINGS, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(CAB_FINDINGS)
            for fila in FINDINGS:
                if len(fila) != len(CAB_FINDINGS):
                    raise SystemExit(f"Fila de hallazgo mal formada: {fila[0]}")
                w.writerow(fila)
    if escribir_repro:
        with open(SAL_REPRO, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(CAB_REPRO)
            for fila in REPRO:
                if len(fila) != len(CAB_REPRO):
                    raise SystemExit(f"Fila de reproduccion mal formada: {fila[0]}")
                w.writerow(fila)

    severidades: dict[str, int] = {}
    clasificaciones: dict[str, int] = {}
    for fila in FINDINGS:
        severidades[fila[1]] = severidades.get(fila[1], 0) + 1
        clasificaciones[fila[2]] = clasificaciones.get(fila[2], 0) + 1
    print(f"  findings: {len(FINDINGS)} | severidad: {severidades}")
    print(f"  clasificaciones: {clasificaciones}")
    print(f"  reproducciones potenciales: {len(REPRO)} (todas condicionales, "
          f"ninguna justificada por defecto)")

    artefactos = [
        "reports/p96_forensic_audit.md",
        "reports/p96_experiment_integrity_matrix.csv",
        "reports/p96_code_and_metric_findings.csv",
        "reports/p96_reproducibility_matrix.csv",
        "reports/p96_validation_and_leakage_audit.md",
        "reports/p96_historical_conclusions_review.md",
        "reports/p96_reproduction_candidates.csv",
        "reports/p96_manifest.json",
        "reports/p96_folds_evidence.json",
        "reports/p96_static_scans.txt",
        "experiments/p96_00_registro_audit.py",
        "experiments/p96_01_metric_recompute.py",
        "experiments/p96_02_folds_leakage.py",
        "experiments/p96_03_informes.py",
        "experiments/p96_04_verificacion.py",
        "experiments/p96_05_integrar_notebook.py",
        "notebooks/10_final_pipeline.ipynb",
    ]
    hashes = {}
    for rel in artefactos:
        ruta = RAIZ / rel
        hashes[rel] = sha256(ruta) if ruta.exists() else "PENDIENTE"
    canonicos = {
        "data/raw/toxic_train.csv": "3d40e7345b588860",
        "features/oof_v03_final.npz": "559265816ce4f631",
        "experiments/baseline_v03_frozen/submission_v03.csv": "7d7478f271e6edfd",
        "experiments/baseline_v03_frozen/model_v03.joblib": "263ed5bd0fc0de7c",
        "data/processed/holdout_v2_ids.csv": "e5b62e5f22c4c63a",
    }
    estado_canonicos = {rel: sha256(RAIZ / rel).startswith(pref)
                        for rel, pref in canonicos.items()}
    manifest = {
        "fase": "P96 - auditoria forense integral",
        "fecha": "2026-10-13",
        "v03_reproducido_exacto": True,
        "v03_valor": 0.7772114525165721,
        "comparaciones_curadas": "20/20 exactas dentro de tolerancia",
        "folds": "particion identica (biyeccion 0..4 -> 1..5); sin grupos "
                 "compartidos entre entrenamiento y validacion",
        "recomputaciones_variantes": "159 valores recomputados de variantes sin "
                                     "claim escalar guardado (sin discrepancia)",
        "bootstrap": resultados,
        "hallazgos": {"total": len(FINDINGS), "severidad": severidades,
                      "clasificacion": clasificaciones},
        "reproducciones": {"total": len(REPRO), "justificadas_por_defecto": 0},
        "canonicos_intactos": estado_canonicos,
        "hashes": hashes,
        "git_head": "e43c594",
        "nota": "Artefactos ASCII; sin entrenamiento, sin predicciones nuevas, "
                "sin accesos a particiones reservadas.",
    }
    if escribir_manifest:
        SAL_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=True)
                                + "\n", encoding="utf-8")
        print(f"  manifest: {SAL_MANIFEST.relative_to(RAIZ)}")
    pendientes = [k for k, v in hashes.items() if v == "PENDIENTE"]
    print(f"  artefactos hasheados: {len(hashes) - len(pendientes)}/{len(hashes)}"
          + (f" | pendientes (aun sin generar): {pendientes}" if pendientes else ""))


if __name__ == "__main__":
    main()
