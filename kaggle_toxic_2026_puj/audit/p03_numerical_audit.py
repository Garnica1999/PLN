"""P3: auditoria numerica - recomputacion de los valores criticos.

Verifica con aritmetica independiente:
- V03 canonical (0.777211) desde el OOF guardado (pred, pliegues, umbrales).
- OOF historico (1731 filas) a umbral 0.5 y a umbrales congelados.
- Los 7 controles v6 (B0/A0/I0/C1-0/H0/F0/J0).
- Deltas vs V03 en las tablas maestras v4/v6/v7 y final_candidates.
- Neto = aciertos - errores donde exista.
- Umbral anidado v7 (medianas por pliegue vs dispersion).
- Bootstrap: identidad metrica_macro, reproducibilidad, orden del IC.
- Holdout P7 vs holdout_evaluaciones_v2.csv.

Escribe audit/out/p03_numeric_checks.csv y audit/p03_findings.json.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/home/carlos/Git/PLN-toxic-audit/kaggle_toxic_2026_puj")
REPORTS = PROJ / "reports"
OUT = PROJ / "audit" / "out"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJ))

from src.evaluation import (  # noqa: E402
    bootstrap_diferencia_pareada,
    metrica_macro,
    metricas_multietiqueta,
)

REF = 0.777211
CHECKS: list[dict] = []


def check(check_id: str, artifact: str, claim: str, expected, recomputed,
          tol: float = 5e-7, status: str | None = None) -> None:
    if status is None:
        if expected is None or recomputed is None:
            status = "UNKNOWN"
        else:
            try:
                status = "PASS" if abs(float(expected) - float(recomputed)) <= tol else "FAIL"
            except (TypeError, ValueError):
                status = "PASS" if str(expected) == str(recomputed) else "FAIL"
    CHECKS.append({
        "check_id": check_id, "artifact": artifact, "claim": claim,
        "expected": expected, "recomputed": recomputed, "status": status,
    })


def main() -> None:  # noqa: PLR0915
    findings: dict = {}

    # ---------- A. V03 canonical desde el OOF ----------
    oof = np.load(PROJ / "features" / "oof_v03_final.npz", allow_pickle=False)
    ids, y_true, pliegues = oof["ids"], oof["y_true"], oof["pliegues"]
    proba, pred_guardado = oof["proba"], oof["pred"]
    umbrales = oof["umbrales"]
    findings["v03_oof"] = {
        "n_filas": int(len(ids)),
        "umbrales": umbrales.tolist(),
        "corrected_cv_guardado": float(oof["corrected_cv"][0]),
        "f1_pliegues_guardados": oof["f1_pliegues"].tolist(),
        "f1_por_etiqueta_guardado": oof["f1_por_etiqueta"].tolist(),
        "fuente": str(oof["fuente"][0]),
        "nota_mapeo": str(oof["nota_mapeo"][0]),
        "pliegues_tamano": {int(k): int(v) for k, v in zip(*np.unique(pliegues, return_counts=True), strict=True)},
    }
    check("A1", "features/oof_v03_final.npz", "umbrales == [0.44,0.51,0.46]",
          [0.44, 0.51, 0.46], [round(float(u), 6) for u in umbrales], status=None
          if np.allclose(umbrales, [0.44, 0.51, 0.46]) else "FAIL")
    pred_calc = (proba >= umbrales.reshape(1, 3)).astype(int)
    check("A2", "features/oof_v03_final.npz", "pred == proba >= umbrales",
          "0 discrepancias", f"{int((pred_calc != pred_guardado).sum())} discrepancias")
    pred_post = pred_calc.copy()
    mascara = pred_post[:, 0] == 0
    pred_post[mascara, 1] = 0
    pred_post[mascara, 2] = 0
    check("A3", "features/oof_v03_final.npz",
          "V03 no aplica postproceso jerarquico (pred guardado = umbral directo)",
          "celdas que cambiarian: 55",
          f"celdas que cambiarian: {int((pred_post != pred_guardado).sum())}", status="INFO")
    m = metricas_multietiqueta(y_true, pred_guardado)
    check("A4", "features/oof_v03_final.npz", "corrected_cv almacenado == F1(OOF)",
          float(oof["corrected_cv"][0]), round(m["f1_media_columnas"], 6))
    check("A5", "features/oof_v03_final.npz", "f1_por_etiqueta almacenado == F1(OOF)",
          [round(float(v), 6) for v in oof["f1_por_etiqueta"]],
          [round(m["f1_por_etiqueta"][e], 6) for e in ("HS", "TR", "AG")])
    folds_f1 = []
    for fold in sorted(set(pliegues.tolist())):
        sel = pliegues == fold
        folds_f1.append(round(metricas_multietiqueta(y_true[sel], pred_guardado[sel])["f1_media_columnas"], 6))
    check("A6", "features/oof_v03_final.npz", "f1_pliegues almacenados == F1 por pliegue del OOF",
          [round(float(v), 6) for v in oof["f1_pliegues"]], folds_f1)
    check("A7", "features/oof_v03_final.npz",
          "media(f1_pliegues) vs corrected_cv (convencion: pooled OOF)",
          round(float(oof["corrected_cv"][0]), 6), round(float(np.mean(folds_f1)), 6),
          status="INFO")
    check("A8", "features/oof_v03_final.npz", "corrected_cv == 0.777211 (referencia congelada)",
          REF, round(float(oof["corrected_cv"][0]), 6))
    check("A9", "features/oof_v03_final.npz", "etiquetas == HS,TR,AG",
          ["HS", "TR", "AG"], [str(x) for x in oof["etiquetas"]], status=None
          if [str(x) for x in oof["etiquetas"]] == ["HS", "TR", "AG"] else "FAIL")
    findings["v03_recomputado"] = {"f1_media_columnas": m["f1_media_columnas"],
                                   "f1_por_etiqueta": m["f1_por_etiqueta"],
                                   "f1_pliegues": folds_f1}

    # ---------- B. OOF historico (1731 filas) ----------
    hist = np.load(PROJ / "experiments" / "baseline_v03_frozen" / "oof_v03_historical.npz")
    hy, hp = hist["y_true"], hist["proba"]
    f1_05 = metricas_multietiqueta(hy, (hp >= 0.5).astype(int))["f1_media_columnas"]
    f1_thr = metricas_multietiqueta(hy, (hp >= np.array([0.44, 0.51, 0.46])).astype(int))["f1_media_columnas"]
    findings["historico_oof"] = {
        "n_filas": int(len(hy)), "f1_umbral_0_5": round(float(f1_05), 6),
        "f1_umbrales_congelados": round(float(f1_thr), 6),
    }
    check("B1", "baseline_v03_frozen/oof_v03_historical.npz", "F1 a umbral 0.5 (historico)",
          0.7927, round(float(f1_05), 4), tol=5e-5)
    check("B2", "baseline_v03_frozen/oof_v03_historical.npz", "F1 a umbrales congelados (historico)",
          0.7989, round(float(f1_thr), 4), tol=5e-5)

    # ---------- C. Controles v6 ----------
    master = list(csv.DictReader((REPORTS / "v6_master_results.csv").open(newline="", encoding="utf-8")))
    control_rows = [r for r in master if "control" in r["variante"].lower()]
    controles = {}
    for r in control_rows:
        controles[r["variante"]] = {"cv": r["corrected_cv"], "delta": r["delta_vs_V03"]}
        ok_cv = abs(float(r["corrected_cv"]) - REF) < 1e-9
        ok_delta = abs(float(r["delta_vs_V03"])) < 1e-9
        check(f"C-{r['variante']}", "v6_master_results.csv", "cv == 0.777211", REF,
              float(r["corrected_cv"]), tol=1e-9,
              status="PASS" if ok_cv and ok_delta else "FAIL")
    findings["controles_v6"] = controles
    findings["n_controles_v6"] = len(control_rows)
    # controles en cada CSV de track
    track_files = ["v6_label_factorization", "v6_ponderacion_grupo", "v6_char_weight",
                   "v6_skipgrams", "v6_bagging_grupos", "v6_ruido_par", "v6_aumento"]
    for name in track_files:
        path = REPORTS / f"{name}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        cols = [c for c in frame.columns if "corrected_cv" in c or c in {"cv", "f1"}]
        if not cols:
            continue
        col = cols[0]
        mask = frame.astype(str).apply(lambda s: s.str.contains("control", case=False)).any(axis=1)
        for _, row in frame[mask].iterrows():
            check(f"C-{name}-{str(row.get('variante', row.get('esquema', 'control')))[:20]}",
                  f"reports/{name}.csv", f"control {col} == 0.777211", REF,
                  round(float(row[col]), 6), tol=1e-6)

    # ---------- D. Deltas en tablas maestras ----------
    n_ok = n_fail = 0
    for r in master:
        if not r["corrected_cv"] or not r["delta_vs_V03"]:
            continue
        delta = float(r["corrected_cv"]) - REF
        ok = abs(delta - float(r["delta_vs_V03"])) <= 5e-7
        n_ok += ok
        n_fail += (not ok)
        if not ok:
            check(f"D-v6-{r['experiment_id']}-{r['variante']}", "v6_master_results.csv",
                  "delta == cv - 0.777211", round(delta, 6), r["delta_vs_V03"], status="FAIL")
    findings["v6_deltas"] = {"ok": int(n_ok), "fail": int(n_fail)}
    v7m = list(csv.DictReader((REPORTS / "v7_master_results.csv").open(newline="", encoding="utf-8")))
    ref_por_variante = {
        ("V7-01", "BCE-control"): REF,
        ("V7-01", "FOCAL"): 0.777165,
        ("V7-01", "F1-SUAVE"): 0.777165,
        ("V7-03", "P0-1-anidado"): 0.769123,
    }
    n_ok = n_fail = 0
    for r in v7m:
        if not r.get("corrected_cv") or not r.get("delta"):
            continue
        referencia = ref_por_variante.get((r["experiment_id"], r["variante"]), REF)
        delta = float(r["corrected_cv"]) - referencia
        ok = abs(delta - float(r["delta"])) <= 2e-6
        n_ok += ok
        n_fail += (not ok)
        if not ok:
            check(f"D-v7-{r['experiment_id']}-{r['variante']}", "v7_master_results.csv",
                  f"delta == cv - {referencia} (referencia por convencion del script)",
                  round(delta, 6), r["delta"], status="FAIL")
    findings["v7_deltas"] = {"ok": int(n_ok), "fail": int(n_fail)}
    findings["v7_convenciones"] = {
        "V7-01": "deltas vs control BCE (0.777165) excepto el control, que va vs V03 (documentado en v7_perdida_f1.md)",
        "V7-03": "P0-1-anidado va vs su control frozen 0.769123 (documentado)",
    }

    # ---------- E. Candidatos finales ----------
    fin = list(csv.DictReader((REPORTS / "final_candidates.csv").open(newline="", encoding="utf-8")))
    REF_FINALS = 0.7772  # referencia redondeada al numero de cifras mostrado en el CSV
    n_ok = n_fail = 0
    for r in fin:
        if not r.get("corrected_cv") or not r.get("delta_vs_v03"):
            continue
        delta = float(r["corrected_cv"]) - REF_FINALS
        ok = abs(delta - float(r["delta_vs_v03"])) <= 5e-7
        n_ok += ok
        n_fail += (not ok)
        if not ok:
            check(f"E-{r['experiment_id']}", "final_candidates.csv",
                  "delta_vs_v03 == corrected_cv - 0.7772 (referencia redondeada)",
                  round(delta, 6), r["delta_vs_v03"], status="FAIL")
        else:
            delta_canonica = float(r["corrected_cv"]) - REF
            if abs(delta_canonica - float(r["delta_vs_v03"])) > 1e-6:
                check(f"E-ref-{r['experiment_id']}", "final_candidates.csv",
                      "delta con referencia 0.7772 difiere de la canonica 0.777211",
                      round(delta_canonica, 6), r["delta_vs_v03"], status="INFO")
        if r.get("fold_scores"):
            scores_raw = (str(r["fold_scores"]).replace("[", " ").replace("]", " ")
                          .replace(";", " ").replace(",", " "))
            scores = []
            for token in scores_raw.split():
                try:
                    scores.append(float(token))
                except ValueError:
                    continue
            if len(scores) < 3:
                scores = []
            if scores:
                diferencia = float(np.mean(scores)) - float(r["corrected_cv"])
                estado = "PASS" if abs(diferencia) <= 1e-4 else ("INFO" if abs(diferencia) <= 3e-4 else "FAIL")
                n_ok += (estado == "PASS")
                n_fail += (estado == "FAIL")
                check(f"E-fold-{r['experiment_id']}", "final_candidates.csv",
                      "media(fold_scores) vs corrected_cv (pooled OOF)",
                      r["corrected_cv"], round(float(np.mean(scores)), 6), status=estado)
    findings["fins"] = {"ok": int(n_ok), "fail": int(n_fail)}

    # ---------- F. v4 candidates (neto) ----------
    v4 = list(csv.DictReader((REPORTS / "v4_candidates.csv").open(newline="", encoding="utf-8")))
    n_ok = n_fail = 0
    for r in v4:
        a = r.get("nuevos_aciertos") or ""
        e = r.get("nuevos_errores") or ""
        neto_col = [c for c in r if c.startswith("neto")]
        if not (a and e and neto_col):
            continue
        neto = float(r[neto_col[0]]) if r[neto_col[0]] else None
        if neto is None:
            continue
        check_ok = abs((float(a) - float(e)) - neto) <= 1e-9
        n_ok += check_ok
        n_fail += (not check_ok)
        if not check_ok:
            check(f"F-{r['candidato']}", "v4_candidates.csv",
                  "neto == nuevos_aciertos - nuevos_errores", float(a) - float(e),
                  neto, status="FAIL")
    findings["v4_netos"] = {"ok": int(n_ok), "fail": int(n_fail)}

    # ---------- G. Umbral anidado v7 ----------
    umbral = json.loads((REPORTS / "v7_umbral_anidado.json").read_text())
    por_pliegue = umbral["umbrales_por_pliegue"]
    for etiqueta in ("HS", "TR", "AG"):
        valores = [p[etiqueta] for p in por_pliegue]
        mediana = float(np.median(valores))
        check(f"G-{etiqueta}", "v7_umbral_anidado.json", f"mediana({etiqueta}) == dispersion.mediana",
              umbral["dispersion"][etiqueta]["mediana"], mediana, tol=1e-9)
        check(f"G-disp-{etiqueta}", "v7_umbral_anidado.json", f"dispersion({etiqueta}).valores",
              umbral["dispersion"][etiqueta]["valores"], valores, status=None
              if umbral["dispersion"][etiqueta]["valores"] == valores else "FAIL")
    calib = pd.read_csv(REPORTS / "v7_calibration_analysis.csv")
    for etiqueta in ("HS", "TR", "AG"):
        sub = calib[calib["etiqueta"] == etiqueta]
        medianas = sub["mediana_de_los_cinco"].tolist()
        calc = [float(np.median([p[etiqueta] for p in por_pliegue]))] * len(sub)
        check(f"G-csv-{etiqueta}", "v7_calibration_analysis.csv",
              "mediana_de_los_cinco == mediana de umbrales_por_pliegue",
              medianas, calc, status=None
              if all(abs(x - y) <= 1e-9 for x, y in zip(medianas, calc, strict=True)) else "FAIL")
    ok_nested = abs(float(umbral["referencia_e2_16_protocolo_laxo"]) - 0.776613) < 1e-9
    check("G-laxo", "v7_umbral_anidado.json", "referencia lax (E2-16) == 0.776613",
          0.776613, umbral["referencia_e2_16_protocolo_laxo"], tol=1e-9,
          status="PASS" if ok_nested else "FAIL")

    # ---------- H. control_pseudo_autores ----------
    cpa = json.loads((REPORTS / "control_pseudo_autores.json").read_text())
    check("H1", "control_pseudo_autores.json", "diferencia == f1_con - f1_solo",
          round(cpa["f1_con_grupos_fusionados"] - cpa["f1_solo_near_duplicados"], 9),
          round(cpa["diferencia"], 9))
    check("H2", "control_pseudo_autores.json", "f1_con_grupos_fusionados == 0.7838 (D-015)",
          0.7838, round(cpa["f1_con_grupos_fusionados"], 4), tol=5e-5)
    check("H3", "control_pseudo_autores.json", "f1_solo_near_duplicados == 0.7927 (D-015)",
          0.7927, round(cpa["f1_solo_near_duplicados"], 4), tol=5e-5)

    # ---------- I. Bootstrap ----------
    pred_a = pred_post
    pred_b = (proba >= 0.5).astype(int)
    b1 = bootstrap_diferencia_pareada(y_true, pred_a, pred_b, repeticiones=2000)
    b2 = bootstrap_diferencia_pareada(y_true, pred_a, pred_b, repeticiones=2000)
    check("I1", "src/evaluation.py", "bootstrap reproducible con la misma semilla",
          b1["ic95"], b2["ic95"], status="PASS" if b1["ic95"] == b2["ic95"] else "FAIL")
    check("I2", "src/evaluation.py", "IC95 contiene la diferencia observada",
          f"[{b1['ic95'][0]}, {b1['ic95'][1]}] contiene {b1['diferencia_observada']}",
          round(b1["diferencia_observada"], 6),
          status="PASS" if b1["ic95"][0] <= b1["diferencia_observada"] <= b1["ic95"][1] else "FAIL")
    check("I3", "src/evaluation.py", "metrica_macro == f1_media_columnas (misma matriz)",
          0.0, float(metrica_macro(y_true, pred_a)
                     - metricas_multietiqueta(y_true, pred_a)["f1_media_columnas"]), tol=1e-12)
    findings["bootstrap_umbral_vs_05"] = {
        "metrica_a": b1["metrica_a"], "metrica_b": b1["metrica_b"],
        "diferencia_observada": b1["diferencia_observada"], "ic95": b1["ic95"],
    }

    # ---------- J. Holdout P7 ----------
    p7 = json.loads((REPORTS / "p7_holdout_resultados.json").read_text())
    findings["holdout_p7"] = {
        "paso_1_v03_holdout": str(p7.get("paso_1_v03_holdout"))[:400],
        "paso_2_meta_holdout": str(p7.get("paso_2_meta_holdout"))[:400],
        "decision": str(p7.get("decision"))[:300],
    }
    haystack = json.dumps(p7)
    otros: list[tuple[str, str]] = []
    for path in sorted(REPORTS.glob("*.*")):
        if path.suffix.lower() not in {".md", ".json", ".csv", ".txt"}:
            continue
        if path.name == "holdout_evaluaciones_v2.csv":
            continue
        try:
            otros.append((path.name, path.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            pass
    haystack_all = haystack + "\n" + "\n".join(text for _name, text in otros)
    csv_hold = pd.read_csv(REPORTS / "holdout_evaluaciones_v2.csv")
    findings["n_evaluaciones_holdout_en_log"] = len(csv_hold)
    for row in csv_hold.itertuples():
        try:
            res = json.loads(row.resultado)
        except Exception:  # noqa: BLE001
            continue
        f1 = res.get("holdout_f1_media_columnas")
        if f1 is None:
            continue
        token1, token2 = f"{f1:.4f}", str(f1)
        ubicacion = next((name for name, text in otros if token1 in text or token2 in text), "")
        presente = bool(ubicacion) or token1 in haystack or token2 in haystack
        if ubicacion:
            pasado = True
        else:
            pasado = token1 in haystack or token2 in haystack
        CHECKS.append({
            "check_id": f"J-{row.experimento}", "artifact": "holdout_evaluaciones_v2.csv",
            "claim": f"{row.experimento} holdout F1 {f1:.4f} documentado en otro artefacto",
            "expected": "documentado", "recomputed": ubicacion or ("p7_holdout_resultados.json" if pasado else "no localizado"),
            "status": "PASS" if pasado else "UNKNOWN",
        })

    # ---------- K. Coherencia global de 0.777211 ----------
    hits = 0
    for path in REPORTS.glob("*.json"):
        try:
            hits += path.read_text(encoding="utf-8", errors="replace").count("0.777211")
        except OSError:
            pass
    findings["menciones_0_777211_en_json"] = hits

    # ---------- escritura ----------
    with (OUT / "p03_numeric_checks.csv").open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=["check_id", "artifact", "claim", "expected",
                                                "recomputed", "status"])
        writer.writeheader()
        writer.writerows(CHECKS)
    status_counts: dict[str, int] = {}
    for c in CHECKS:
        status_counts[c["status"]] = status_counts.get(c["status"], 0) + 1
    findings["resumen_checks"] = status_counts
    (PROJ / "audit" / "p03_findings.json").write_text(
        json.dumps(findings, indent=2, ensure_ascii=True, default=str), encoding="ascii")
    print("checks:", status_counts)
    fails = [c for c in CHECKS if c["status"] == "FAIL"]
    for c in fails:
        print("FAIL", c["check_id"], "|", c["claim"], "|", c["expected"], "vs", c["recomputed"])
    print("v03:", findings["v03_recomputado"])
    print("historico:", findings["historico_oof"])
    print("controles v6:", findings["n_controles_v6"], list(findings["controles_v6"].keys()))
    print("deltas v6:", findings["v6_deltas"], "v7:", findings["v7_deltas"], "fins:", findings["fins"],
          "v4 netos:", findings["v4_netos"])
    print("bootstrap umbral-vs-0.5:", findings["bootstrap_umbral_vs_05"])


if __name__ == "__main__":
    main()
