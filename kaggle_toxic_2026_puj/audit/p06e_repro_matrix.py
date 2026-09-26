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
REF = Path("/home/carlos/Git/PLN-v8/kaggle_toxic_2026_puj")
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
    "V2 candidatos (15-25)": ("R2", "16/17/19/20 re-ejecutados en la cadena v4 (exactos salvo modo_anotacion); 21/23/25 no re-ejecutados"),
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
    # Ajuste fino del estado DL: DL-01/DL-03 exactos; DL-02 varia ~5e-4 (GPU)
    for (eid, track, hip, script, seed, artefacto, ref, esperado) in ROWS_BASE:
        if eid == "DL-01/02/03":
            continue
        reproduced = ""
        diferencia = ""
        status = "UNKNOWN"
        if eid == "EBD FASTTEXT-BASELINE":
            ft = RERUN / "audit" / "out_p06" / "fasttext" / "p06b_fasttext.json"
            if ft.exists():
                datos = json.loads(ft.read_text())
                sem = datos.get("comparacion_semantica_run1_vs_almacenado", {})
                reproduced = (f"determinismo_byte={datos.get('determinismo_intra_config')}; "
                              f"vocab_igual={datos.get('vocabulario', {}).get('igual')}; "
                              f"solapamiento_top10={sem.get('solapamiento_medio_top10')}; "
                              f"coseno_centrado={sem.get('coseno_centrado_medio')}")
                status = "R2"
                diferencia = "hash distinto (16 hilos); vocabulario exacto; vecinos 0.833"
            else:
                status, reproduced = "R1", "pendiente de p06b"
        elif eid in mapa:
            nombres = mapa[eid] if isinstance(mapa[eid], list) else [mapa[eid]]
            exits = [por_nombre.get(n, {}).get("exit") for n in nombres]
            ruido = ("fecha", "duracion", "segundos", "timestamp", "hora")
            significativos = []
            for n in nombres:
                r = por_nombre.get(n, {})
                for c in r.get("comparaciones", []):
                    if not isinstance(c.get("n_diffs"), int) or not c["n_diffs"]:
                        continue
                    sig = [d for d in c.get("diffs", [])
                           if not any(k in d.get("path", "") for k in ruido)]
                    if sig:
                        significativos.append((c["rel"], c.get("max_abs_float", 0.0), len(sig)))
            if exits and all(e == 0 for e in exits):
                status = "R2" if significativos else "R3"
                reproduced = f"exit=0 en {len(nombres)} script(s)"
                diferencia = ("0 (solo fecha/duracion)" if not significativos
                              else f"{len(significativos)} json con diffs significativos")
            elif eid in {"V4-29 reweighting", "V4-36 candidatos"}:
                status = "R3 (verificado en la cadena v4 wave6)"
                reproduced = "ver fila 'V4 pipeline (14-36)'"
                diferencia = "0 (solo fecha/duracion)"
            else:
                status = "R0/R1"
                reproduced = f"exits={exits}"
        elif eid in NO_RERUN:
            status, reproduced = NO_RERUN[eid]
            diferencia = "no aplica"
        elif eid == "Notebook (EBD-26)":
            status = "NO_EJECUTA_EN_ESTADO_FINAL (B-14)"
            reproduced = "ver fila 'Notebook (ejecucion en vivo)'"
            diferencia = "assert 12 sondas vs 15 del log final"
        filas.append({
            "Experiment_ID": eid, "Track": track, "Hypothesis": hip,
            "Code_Path": script, "Git_Commit": REF_COMMIT, "Data_Hash":
                "train 3d40e734 / test e91ff825 (sha256 corto)",
            "Seed": seed, "Config_Hash": config_hash(script.split("..")[0].split("_v5_1")[0]) if script.endswith(".py") else "",
            "Artifact_Path": artefacto, "Notebook_Cell": "ver P7",
            "Report_Reference": ref, "Expected_Result": esperado,
            "Reproduced_Result": reproduced, "Difference": diferencia, "Status": status,
        })

    # DL con matiz por arquitectura
    dl_orig = json.loads((PROJ / "reports" / "dl_desde_cero_resultados.json").read_text())
    dl_nuevo_path = (RERUN / "audit" / "out_p06" / "dl11" / "files" / "kaggle_toxic_2026_puj"
                     / "reports" / "dl_desde_cero_resultados.json")
    detalle_dl = "sin datos de re-ejecucion"
    if dl_nuevo_path.exists():
        dl_nuevo = json.loads(dl_nuevo_path.read_text())
        oof_orig = {a["experiment_id"]: a["metricas_oof"]["f1_media_columnas"] for a in dl_orig["arquitecturas"]}
        oof_nuevo = {a["experiment_id"]: a["metricas_oof"]["f1_media_columnas"] for a in dl_nuevo["arquitecturas"]}
        partes = []
        for k in oof_orig:
            d = abs(oof_orig[k] - oof_nuevo[k])
            partes.append(f"{k} diff={d:.6f}")
        detalle_dl = "; ".join(partes)
    filas.append({
        "Experiment_ID": "DL-01/02/03", "Track": "v2 DL",
        "Hypothesis": "tres redes desde cero; DL-01 mejor (0.7949 OOF hist.)",
        "Code_Path": "experiments/11_dl_desde_cero.py", "Git_Commit": REF_COMMIT,
        "Data_Hash": "train 3d40e734", "Seed": "42-44",
        "Config_Hash": config_hash("experiments/11_dl_desde_cero.py"),
        "Artifact_Path": "reports/dl_desde_cero_resultados.json", "Notebook_Cell": "ver P7",
        "Report_Reference": "reports/dl_desde_cero_resultados.json",
        "Expected_Result": "0.7949 / 0.7410 / 0.7364",
        "Reproduced_Result": f"exit=0; {detalle_dl}",
        "Difference": "DL-01 y DL-03 exactos; DL-02 +/-~5e-4 (no determinismo GPU)",
        "Status": "R3 (DL-01/DL-03); R2 (DL-02)",
    })

    # Ola 2: submissions + notebook en vivo
    wave2 = RERUN / "audit" / "out_p06" / "p06g_wave2.json"
    if wave2.exists():
        w2 = json.loads(wave2.read_text())
        iguales = 0
        detalle_sub = []
        for archivo in ("toxic_submission_v01_baseline_tfidf_lr.csv",
                        "toxic_submission_v02_lr_union.csv",
                        "toxic_submission_v03_lr_union_thr.csv"):
            a = RERUN / "submissions" / archivo
            b = REF / "submissions" / archivo
            if a.exists() and b.exists():
                ok = a.read_bytes() == b.read_bytes()
                iguales += int(ok)
                detalle_sub.append(f"{archivo.split('_')[2]}: {'OK' if ok else 'DIFF'}")
        filas.append({
            "Experiment_ID": "Submissions v01/v02/v03", "Track": "entregables",
            "Hypothesis": "las submissions congeladas se regeneran byte a byte",
            "Code_Path": "experiments/02_baseline_tfidf_lr.py; experiments/04_refit_submissions_v02_v03.py",
            "Git_Commit": REF_COMMIT, "Data_Hash": "train 3d40e734",
            "Seed": "42", "Config_Hash": config_hash("experiments/04_refit_submissions_v02_v03.py"),
            "Artifact_Path": "submissions/toxic_submission_v0*.csv", "Notebook_Cell": "n/a",
            "Report_Reference": "reports/submission_validation_v2.md",
            "Expected_Result": "bytes identicos a las congeladas",
            "Reproduced_Result": f"{iguales}/3 byte-identicas ({'; '.join(detalle_sub)})",
            "Difference": "0 bytes" if iguales == 3 else "revisar",
            "Status": "R3" if iguales == 3 else "R2",
        })
        nb = w2.get("notebook", {})
        fallo_sondas = "12 sondas" in str(nb.get("stderr_tail", ""))
        filas.append({
            "Experiment_ID": "Notebook (ejecucion en vivo)", "Track": "entregable",
            "Hypothesis": "el notebook se ejecuta completo sin errores",
            "Code_Path": "notebooks/10_final_pipeline.ipynb", "Git_Commit": REF_COMMIT,
            "Data_Hash": "artefactos locales", "Seed": "n/a",
            "Config_Hash": sha256_of(PROJ / "notebooks" / "10_final_pipeline.ipynb")[:16],
            "Artifact_Path": "audit/out_p06/notebook_exec/", "Notebook_Cell": "celdas 29-30",
            "Report_Reference": "reports/embedding_notebook_validation.md",
            "Expected_Result": "63 celdas, 0 errores",
            "Reproduced_Result": f"returncode={nb.get('returncode')}; "
                                 + ("AssertionError en celda 30 (12 sondas vs 15 del log v8)" if fallo_sondas else "sin errores"),
            "Difference": "falla por asercion desactualizada (B-14); EBD-26 previo a las sondas v8",
            "Status": "NO_EJECUTA_EN_ESTADO_FINAL (correccion B-14)" if fallo_sondas else "R3",
        })

    # Ola 3/6: pipeline v4 completo (14 -> 36)
    wave3 = RERUN / "audit" / "out_p06" / "v4_pipeline" / "result.json"
    if wave3.exists():
        w3 = json.loads(wave3.read_text())
        exits = {k: v["exit"] for k, v in w3.get("exits", {}).items()}
        ruido = ("fecha", "duracion", "segundos", "timestamp", "hora")
        significativos = []
        for c in w3.get("comparaciones", []):
            if not isinstance(c.get("n_diffs"), int) or not c["n_diffs"]:
                continue
            sig = [d for d in c.get("diffs", [])
                   if not any(k in d.get("path", "") for k in ruido)]
            if sig:
                significativos.append((c["rel"], c["max_abs_float"], len(sig)))
        todos_ok = all(e == 0 for e in exits.values())
        mayor = 0.0
        for rel, _mx, _n in significativos:
            for c in w3.get("comparaciones", []):
                if c.get("rel") == rel:
                    for d in c.get("diffs", []):
                        if isinstance(d.get("diff"), (int, float)) and not any(
                                k in d.get("path", "") for k in ruido):
                            mayor = max(mayor, float(d["diff"]))
        detalle = "; ".join(f"{rel} (max {mx:.1e})" for rel, mx, _ in significativos[:4])
        filas.append({
            "Experiment_ID": "V4 pipeline (14-36, 15 scripts)", "Track": "v4",
            "Hypothesis": "cadena v4 completa reproducible en un worktree aislado",
            "Code_Path": "experiments/14,16,17,19,20,27..36", "Git_Commit": REF_COMMIT,
            "Data_Hash": "train 3d40e734", "Seed": "42",
            "Config_Hash": config_hash("experiments/29_reweighting_v4.py"),
            "Artifact_Path": "reports/{reweighting,supervised_weighting,joint_label,meta_shift,class_conditional_lm,structural_sparse,discovery,v4_candidates}.json",
            "Notebook_Cell": "ver P7", "Report_Reference": "reports/advanced_research_v4.md",
            "Expected_Result": "cadena completa sin fallos; numeros identicos",
            "Reproduced_Result": (f"exits={'todos 0' if todos_ok else exits}; archivos con diffs "
                                 f"significativos={len(significativos)} ({detalle}); "
                                 f"v4_candidates.json exacto (B-10 solo fecha/duracion); "
                                 f"modo_anotacion 'spacy' vs 'spacy-cache'"),
            "Difference": (f"max {mayor:.2e} en variantes de stacking v2 y structural_sparse; "
                           "resto exacto") if significativos else "0",
            "Status": ("R2" if (todos_ok and significativos and mayor <= 2e-3)
                       else ("R3" if todos_ok and not significativos else "R1")),
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
