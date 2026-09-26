"""P6e: interpreta el resumen de re-ejecucion y produce la matriz de reproducibilidad.

Lee audit/out_p06/p06_rerun_summary.json del worktree de re-ejecucion y los
hallazgos de la auditoria; escribe reports/audit_reproducibility_matrix.csv.
Se ejecuta DESPUES de que termine el driver P6.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

PROJ = Path("/home/carlos/Git/PLN-toxic-audit/kaggle_toxic_2026_puj")
RERUN = Path("/home/carlos/Git/PLN-toxic-rerun/kaggle_toxic_2026_puj")
SUMMARY = RERUN / "audit" / "out_p06" / "p06_rerun_summary.json"
REF_COMMIT = "b0edf7a (v8)"

# claves JSON "ruidosas" que no cuentan como discrepancia cientifica
RUIDO = ("fecha", "duracion", "segundos", "timestamp", "hora")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(script: str) -> str:
    path = PROJ / script
    return sha256_of(path)[:16] if path.exists() else ""


ROWS_BASE = [
    # (id, track, hipotesis, script, seed, artefacto, ref_reporte, resultados esperados)
    ("V03_corrected", "v2 protocolo", "OOF corregido reproduce 0.777211",
     "experiments/14_oof_corregido_v2.py", "42",
     "features/oof_v03_final.npz; reports/baseline_v03_dev_metrics.json",
     "reports/baseline_v03_dev_metrics.json", "corrected_cv=0.777211"),
    ("PA-01", "v2 control", "control pseudo-autor 0.783825 vs 0.792659",
     "experiments/12_control_pseudo_autores.py", "42",
     "reports/control_pseudo_autores.json", "reports/control_pseudo_autores.json",
     "f1 fusionados=0.783825, solo near-dup=0.792659"),
    ("V6-02..08 controles", "v6", "7 controles exactos 0.777211",
     "experiments/v6_02..v6_08", "42", "reports/v6_*.json",
     "reports/v6_reproducibility.md", "controles = 0.777211 exactos"),
    ("V7-01", "v7", "perdida F1 cierra negativo con control 0.777165",
     "experiments/v7_01_perdida_f1.py", "42", "reports/v7_perdida_f1.json",
     "reports/v7_perdida_f1.md", "control 0.777165; FOCAL -0.0024; F1-SUAVE -0.0033"),
    ("V7-02", "v7", "prior de grupo no aporta",
     "experiments/v7_02_prior_grupo.py", "42", "reports/v7_prior_grupo.json",
     "reports/v7_prior_grupo.md", "deltas -0.0011/-0.0017"),
    ("V7-03", "v7", "umbral anidado estricto; 0.767228",
     "experiments/v7_03_umbral_anidado.py", "42", "reports/v7_umbral_anidado.json",
     "reports/v7_umbral_anidado.md", "P0-0=0.777211; anidado=0.767228"),
    ("V4-29 reweighting", "v4", "hard example mining clasico empeora",
     "experiments/29_reweighting_v4.py", "42", "reports/reweighting_v4.json",
     "reports/reweighting_v4.json", "mejor variante ~0.7854 dev"),
    ("V4-32 meta shift", "v4", "shift de calibracion refutado",
     "experiments/32_meta_shift_v4.py", "42", "reports/meta_shift_v4.json",
     "reports/meta_shift_v4.json", "dano por shift no explica el error"),
    ("V4-36 candidatos", "v4", "P8-C pasa gates y el holdout lo rechaza",
     "experiments/36_candidatos_v4.py", "42", "reports/v4_candidates.json",
     "reports/v4_candidates.json", "mejor por corrected_cv ~0.7854"),
    ("V8-01 ensamble folds", "v8", "promedio de particiones mejora local",
     "experiments/v8_01_ensamble_folds.py", "42", "reports/v8_ensamble_folds.json",
     "reports/v8_ensamble_folds.md", "+0.0051 OOF aprox"),
    ("V8-03 estabilidad features", "v8", "estabilidad de features negativa",
     "experiments/v8_03_estabilidad_features.py", "42", "reports/v8_estabilidad_features.json",
     "reports/v8_estabilidad_features.md", "deltas negativos"),
    ("V8-05 reg conjunta", "v8", "regularizacion conjunta funciona y empeora",
     "experiments/v8_05_regularizacion_conjunta.py", "42", "reports/v8_regularizacion_conjunta.json",
     "reports/v8_regularizacion_conjunta.md", "deltas negativos"),
    ("DL-01/02/03", "v2 DL", "tres redes desde cero; DL-01 mejor (0.7949 OOF hist.)",
     "experiments/11_dl_desde_cero.py", "42-44", "reports/dl_desde_cero_resultados.json",
     "reports/dl_desde_cero_resultados.json", "0.7949 / 0.7410 / 0.7364"),
    ("EBD FASTTEXT-BASELINE", "v5", "FastText desde cero reproducible",
     "experiments/embedding_02_fasttext_baseline_v5.py", "42",
     "data/external/embeddings/fasttext_baseline.vec",
     "reports/embedding_intrinsic_eval.md", "sha256 e54a363d..."),
    ("EBD v5.1 ejes", "v5.1", "5 ejes cerrados por inercia", "experiments/embedding_12..16_v5_1.py",
     "42", "reports/embedding_*_v5_1.json", "reports/embedding_arquitectura_v5_1.md",
     "sin mejora > gate"),
    ("V2 candidatos (15-25)", "v2", "canales y metas restringidos",
     "experiments/15..25_*_v2.py", "42", "reports/stacking_v2.json; final_candidates.csv",
     "reports/final_candidate_screening.md", "mejor LB 0.78163"),
    ("Notebook (EBD-26)", "entregable", "notebook ejecuta sin errores",
     "notebooks/10_final_pipeline.ipynb", "n/a", "reports/embedding_notebook_validation_v5_1.json",
     "reports/embedding_notebook_validation.md", "63 celdas, 0 errores, 4.6 s"),
]

NO_RERUN = {
    "EBD v5.1 ejes": ("R1", "corpus y binarios locales presentes; no re-ejecutado por presupuesto"),
    "V2 candidatos (15-25)": ("R2", "valores recomputados desde artefactos (P3); no re-ejecutado"),
    "Notebook (EBD-26)": ("R2", "evidencia de ejecucion registrada (4.6 s, 0 errores); re-verificado por lectura"),
}


def main() -> None:
    resumen = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else []
    por_nombre = {r["name"]: r for r in resumen}
    mapa = {
        "V03_corrected": "v03_oof", "PA-01": "pa01",
        "V6-02..08 controles": ["v6_02", "v6_03", "v6_04", "v6_05", "v6_06", "v6_07", "v6_08"],
        "V7-01": "v7_01", "V7-02": "v7_02", "V7-03": "v7_03",
        "V4-29 reweighting": "v4_reweighting", "V4-32 meta shift": "v4_meta_shift",
        "V4-36 candidatos": "v4_candidatos",
        "V8-01 ensamble folds": "v8_01", "V8-03 estabilidad features": "v8_03",
        "V8-05 reg conjunta": "v8_05", "DL-01/02/03": "dl11",
    }
    filas = []
    for (eid, track, hip, script, seed, artefacto, ref, esperado) in ROWS_BASE:
        reproduced = ""
        diferencia = ""
        status = "UNKNOWN"
        if eid == "EBD FASTTEXT-BASELINE":
            ft = RERUN / "audit" / "out_p06" / "fasttext" / "p06b_fasttext.json"
            if ft.exists():
                datos = json.loads(ft.read_text())
                reproduced = (f"determinismo={datos.get('determinismo_intra_config')}; "
                              f"igual al almacenado={datos.get('igual_al_almacenado')}; "
                              f"solapamiento={datos.get('comparacion_semantica_run1_vs_almacenado', {}).get('solapamiento_medio_top10')}")
                status = "R3" if datos.get("igual_al_almacenado") else "R2"
                diferencia = "hash exacto" if datos.get("igual_al_almacenado") else "hash distinto; medido semanticamente"
            else:
                status, reproduced = "R1", "pendiente de p06b"
        elif eid in mapa:
            nombres = mapa[eid] if isinstance(mapa[eid], list) else [mapa[eid]]
            exits = [por_nombre.get(n, {}).get("exit") for n in nombres]
            diffs = []
            for n in nombres:
                r = por_nombre.get(n, {})
                diffs += [c for c in r.get("comparaciones", [])
                          if isinstance(c.get("n_diffs"), int) and c["n_diffs"] > 0]
            if exits and all(e == 0 for e in exits):
                status = "R2" if diffs else "R3"
                reproduced = f"exit=0 en {len(nombres)} script(s); json_con_diffs={len(diffs)}"
                diferencia = "0" if not diffs else f"{len(diffs)} json con diffs (revisar ruido)"
            else:
                status = "R0/R1"
                reproduced = f"exits={exits}"
        elif eid in NO_RERUN:
            status, reproduced = NO_RERUN[eid]
            diferencia = "no aplica"
        filas.append({
            "Experiment_ID": eid, "Track": track, "Hypothesis": hip,
            "Code_Path": script, "Git_Commit": REF_COMMIT, "Data_Hash":
                "train 3d40e734 / test e91ff825 (sha256 corto)",
            "Seed": seed, "Config_Hash": config_hash(script.split("..")[0].split("_v5_1")[0]) if script.endswith(".py") else "",
            "Artifact_Path": artefacto, "Notebook_Cell": "ver P7",
            "Report_Reference": ref, "Expected_Result": esperado,
            "Reproduced_Result": reproduced, "Difference": diferencia, "Status": status,
        })
    with (PROJ / "reports" / "audit_reproducibility_matrix.csv").open(
            "w", newline="", encoding="ascii") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    conteo = {}
    for f in filas:
        conteo[f["Status"]] = conteo.get(f["Status"], 0) + 1
    print("filas:", len(filas), conteo)
    for f in filas:
        print(f"  {f['Status']:4s} {f['Experiment_ID']:26s} {f['Difference'][:60]}")


if __name__ == "__main__":
    main()
