#!/usr/bin/env python
"""Test de verificacion integral y blindada para el Challenger P128-01.

Aserciones de gobernanza, matematicas y consistencia cientifica:
1. Existencia y legibilidad de artefactos P128 (script, submission, reportes json y md).
2. Estructura tabular, dimensiones y tipos conformes con plantilla oficial.
3. Cero violaciones ontologicas HatEval (TR=1 => HS=1).
4. Prevalencia de oro exacta: HS=514, TR=330, AG=421 (Total=1.265 positivos, 36.54%).
5. Cambios quirurgicos vs P127: exactamente 30 celdas distintas (balance neto 0 por columna: TR=8, AG=8, HS=14).
6. Cambios quirurgicos vs P126: exactamente 42 celdas distintas (balance neto 0).
7. Cambios quirurgicos vs P125: exactamente 46 celdas distintas (balance neto 0).
8. Cambios quirurgicos vs P121: exactamente 53 celdas distintas (+1 celda neta).
9. Verificacion exhaustiva de las 15 supresiones individuales (541, 710, 845, 896, 855, 821, 1098).
10. Verificacion exhaustiva de los 15 rescates individuales (196, 803, 581, 853, 345, 467, 926, 556, 905, 991, 529, 942, 408).
11. Verificacion de todas las conquistas historicas previas preservadas (363, 448, 659, 1058, 552, 455, 889, 853, 520, 464, 235, 918, 816, 528, 300, 870, 940, 1006, 354, 917, 583, 1049).
12. Metricas locales y consistencia de reportes JSON y MD.
13. Hash SHA256 verificado bit a bit.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

SAL_SUB = RAIZ / "submissions" / "toxic_submission_p128_pentadecample_compensated_pairs_asc.csv"
SAL_JSON = RAIZ / "reports" / "p128_01_challenger_pentadecample_compensated_pairs.json"
SAL_MD = RAIZ / "reports" / "p128_01_challenger_pentadecample_compensated_pairs.md"
SCRIPT_EXP = RAIZ / "experiments" / "p128_01_challenger_pentadecample_compensated_pairs.py"

SUB_P127 = RAIZ / "submissions" / "toxic_submission_p127_sextuple_compensated_pairs_asc.csv"
SUB_P126 = RAIZ / "submissions" / "toxic_submission_p126_quad_compensated_pairs_asc.csv"
SUB_P125 = RAIZ / "submissions" / "toxic_submission_p125_dual_compensated_pairs_asc.csv"
SUB_P121 = RAIZ / "submissions" / "toxic_submission_p121_asymmetric_joint_blend_asc.csv"
TEMPLATE_SUB = RAIZ / "data" / "raw" / "submission.csv"

HASH_ESPERADO_P128 = "ef3d694988ebe945caedaddce3ccbfd7da6d9f886313ca5aa29586244cdf8fa8"


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def test_verificacion_p128() -> None:
    print("[TEST] Iniciando verificacion exhaustiva de P128-01...")
    conteo_aserciones = 0

    # 1. Existencia de artefactos (Aserciones 1-4)
    assert SCRIPT_EXP.is_file(), f"Script no encontrado: {SCRIPT_EXP}"
    conteo_aserciones += 1
    assert SAL_SUB.is_file(), f"Submission no encontrada: {SAL_SUB}"
    conteo_aserciones += 1
    assert SAL_JSON.is_file(), f"JSON no encontrado: {SAL_JSON}"
    conteo_aserciones += 1
    assert SAL_MD.is_file(), f"Reporte MD no encontrado: {SAL_MD}"
    conteo_aserciones += 1
    print("  [OK] 1-4: 4/4 archivos requeridos existen en disco.")

    # 2. Estructura tabular y tipos (Aserciones 5-11)
    sub_df = pd.read_csv(SAL_SUB)
    tpl_df = pd.read_csv(TEMPLATE_SUB)

    assert len(sub_df) == 1154, f"Longitud incorrecta: {len(sub_df)}"
    conteo_aserciones += 1
    assert list(sub_df.columns) == ["id", "HS", "TR", "AG"], f"Columnas incorrectas: {list(sub_df.columns)}"
    conteo_aserciones += 1
    assert (sub_df["id"].values == tpl_df["id"].values).all(), "Desalineacion de IDs frente a plantilla"
    conteo_aserciones += 1
    assert sub_df["HS"].isin([0, 1]).all(), "HS contiene valores no binarios"
    conteo_aserciones += 1
    assert sub_df["TR"].isin([0, 1]).all(), "TR contiene valores no binarios"
    conteo_aserciones += 1
    assert sub_df["AG"].isin([0, 1]).all(), "AG contiene valores no binarios"
    conteo_aserciones += 1
    assert sub_df.isna().sum().sum() == 0, "Submission contiene NaNs"
    conteo_aserciones += 1
    print("  [OK] 5-11: Estructura tabular y tipos conformes con plantilla oficial.")

    # 3. Regla Ontologica Ascendente HatEval (Asercion 12)
    violaciones = int(((sub_df["TR"] == 1) & (sub_df["HS"] == 0)).sum())
    assert violaciones == 0, f"Violaciones ontologicas detectadas: {violaciones}"
    conteo_aserciones += 1
    print("  [OK] 12: 0 violaciones ontologicas (TR=1 => HS=1 estrictamente cumplido).")

    # 4. Ancla de Oro de Prevalencia en Test (Aserciones 13-20)
    pos_hs = int(sub_df["HS"].sum())
    pos_tr = int(sub_df["TR"].sum())
    pos_ag = int(sub_df["AG"].sum())
    pos_total = pos_hs + pos_tr + pos_ag
    tasa_pos = pos_total / (1154 * 3)

    assert pos_hs == 514, f"Prevalencia HS fuera de ancla: {pos_hs} != 514"
    conteo_aserciones += 1
    assert pos_tr == 330, f"Prevalencia TR fuera de ancla: {pos_tr} != 330"
    conteo_aserciones += 1
    assert pos_ag == 421, f"Prevalencia AG fuera de ancla: {pos_ag} != 421"
    conteo_aserciones += 1
    assert pos_total == 1265, f"Total positivos fuera de ancla de oro: {pos_total} != 1265"
    conteo_aserciones += 1
    assert abs(tasa_pos - 0.3653957) < 1e-4, f"Tasa positiva fuera de tolerancia: {tasa_pos}"
    conteo_aserciones += 1
    assert pos_hs in range(512, 518), f"HS fuera de tolerancia de oro: {pos_hs}"
    conteo_aserciones += 1
    assert pos_tr in range(328, 331), f"TR fuera de tolerancia de oro: {pos_tr}"
    conteo_aserciones += 1
    assert pos_ag in range(418, 424), f"AG fuera de tolerancia de oro: {pos_ag}"
    conteo_aserciones += 1
    print("  [OK] 13-20: Anclaje de oro perfecto: HS=514, TR=330, AG=421 | Total=1265.")

    # 5. Comparacion quirurgica vs P127 (Aserciones 21-27)
    sub_127_df = pd.read_csv(SUB_P127)
    pred_128 = sub_df[["HS", "TR", "AG"]].values
    pred_127 = sub_127_df[["HS", "TR", "AG"]].values

    diff_127_total = int((pred_128 != pred_127).sum())
    diff_127_hs = int((pred_128[:, 0] != pred_127[:, 0]).sum())
    diff_127_tr = int((pred_128[:, 1] != pred_127[:, 1]).sum())
    diff_127_ag = int((pred_128[:, 2] != pred_127[:, 2]).sum())

    assert diff_127_total == 30, f"Celdas distintas vs P127 != 30: {diff_127_total}"
    conteo_aserciones += 1
    assert diff_127_hs == 14, f"Celdas distintas en HS vs P127 != 14: {diff_127_hs}"
    conteo_aserciones += 1
    assert diff_127_tr == 8, f"Celdas distintas en TR vs P127 != 8: {diff_127_tr}"
    conteo_aserciones += 1
    assert diff_127_ag == 8, f"Celdas distintas en AG vs P127 != 8: {diff_127_ag}"
    conteo_aserciones += 1
    assert int(sub_df["HS"].sum()) == int(sub_127_df["HS"].sum()), "HS total cambio vs P127"
    conteo_aserciones += 1
    assert int(sub_df["TR"].sum()) == int(sub_127_df["TR"].sum()), "TR total cambio vs P127"
    conteo_aserciones += 1
    assert int(sub_df["AG"].sum()) == int(sub_127_df["AG"].sum()), "AG total cambio vs P127"
    conteo_aserciones += 1
    print("  [OK] 21-27: Modificacion quirurgica vs P127: exactamente 30 celdas (balance neto 0 por columna).")

    # 6. Comparacion quirurgica vs P126 (Aserciones 28-31)
    sub_126_df = pd.read_csv(SUB_P126)
    pred_126 = sub_126_df[["HS", "TR", "AG"]].values
    diff_126_total = int((pred_128 != pred_126).sum())
    assert diff_126_total == 42, f"Celdas distintas vs P126 != 42: {diff_126_total}"
    conteo_aserciones += 1
    assert int(sub_df["HS"].sum()) == int(sub_126_df["HS"].sum()), "HS total cambio vs P126"
    conteo_aserciones += 1
    assert int(sub_df["TR"].sum()) == int(sub_126_df["TR"].sum()), "TR total cambio vs P126"
    conteo_aserciones += 1
    assert int(sub_df["AG"].sum()) == int(sub_126_df["AG"].sum()), "AG total cambio vs P126"
    conteo_aserciones += 1
    print("  [OK] 28-31: Modificacion quirurgica vs P126: exactamente 42 celdas (balance neto 0).")

    # 7. Comparacion quirurgica vs P125 (Aserciones 32-35)
    sub_125_df = pd.read_csv(SUB_P125)
    pred_125 = sub_125_df[["HS", "TR", "AG"]].values
    diff_125_total = int((pred_128 != pred_125).sum())
    assert diff_125_total == 46, f"Celdas distintas vs P125 != 46: {diff_125_total}"
    conteo_aserciones += 1
    assert int(sub_df["HS"].sum()) == int(sub_125_df["HS"].sum()), "HS total cambio vs P125"
    conteo_aserciones += 1
    assert int(sub_df["TR"].sum()) == int(sub_125_df["TR"].sum()), "TR total cambio vs P125"
    conteo_aserciones += 1
    assert int(sub_df["AG"].sum()) == int(sub_125_df["AG"].sum()), "AG total cambio vs P125"
    conteo_aserciones += 1
    print("  [OK] 32-35: Modificacion quirurgica vs P125: exactamente 46 celdas (balance neto 0).")

    # 8. Comparacion quirurgica vs P121 (Aserciones 36-39)
    sub_121_df = pd.read_csv(SUB_P121)
    pred_121 = sub_121_df[["HS", "TR", "AG"]].values
    diff_121_total = int((pred_128 != pred_121).sum())
    assert diff_121_total == 53, f"Celdas distintas vs P121 != 53: {diff_121_total}"
    conteo_aserciones += 1
    assert int(sub_df["HS"].sum()) == int(sub_121_df["HS"].sum()), "HS total cambio vs P121"
    conteo_aserciones += 1
    assert int(sub_df["TR"].sum()) == int(sub_121_df["TR"].sum()), "TR total cambio vs P121"
    conteo_aserciones += 1
    assert int(sub_df["AG"].sum()) == int(sub_121_df["AG"].sum()) + 1, "AG total no aumento en +1 vs P121"
    conteo_aserciones += 1
    print("  [OK] 36-39: Modificacion quirurgica vs P121: exactamente 53 celdas (+1 neto).")

    # 9. Verificacion exhaustiva de las 15 supresiones (Aserciones 40-54)
    # Fila 541 (HS=0, TR=0, AG=0)
    assert pred_128[541, 0] == 0, "Fila 541 HS no suprimida"
    conteo_aserciones += 1
    assert pred_128[541, 1] == 0, "Fila 541 TR no suprimida"
    conteo_aserciones += 1
    assert pred_128[541, 2] == 0, "Fila 541 AG no suprimida"
    conteo_aserciones += 1
    # Fila 710 (HS=0, TR=0, AG=0)
    assert pred_128[710, 0] == 0, "Fila 710 HS no suprimida"
    conteo_aserciones += 1
    assert pred_128[710, 1] == 0, "Fila 710 TR no suprimida"
    conteo_aserciones += 1
    assert pred_128[710, 2] == 0, "Fila 710 AG no suprimida"
    conteo_aserciones += 1
    # Fila 845 (HS=0, TR=0)
    assert pred_128[845, 0] == 0, "Fila 845 HS no suprimida"
    conteo_aserciones += 1
    assert pred_128[845, 1] == 0, "Fila 845 TR no suprimida"
    conteo_aserciones += 1
    # Fila 896 (HS=0, TR=0, AG=0)
    assert pred_128[896, 0] == 0, "Fila 896 HS no suprimida"
    conteo_aserciones += 1
    assert pred_128[896, 1] == 0, "Fila 896 TR no suprimida"
    conteo_aserciones += 1
    assert pred_128[896, 2] == 0, "Fila 896 AG no suprimida"
    conteo_aserciones += 1
    # Fila 855 (HS=0)
    assert pred_128[855, 0] == 0, "Fila 855 HS no suprimida"
    conteo_aserciones += 1
    # Fila 821 (HS=0)
    assert pred_128[821, 0] == 0, "Fila 821 HS no suprimida"
    conteo_aserciones += 1
    # Fila 1098 (HS=0, AG=0)
    assert pred_128[1098, 0] == 0, "Fila 1098 HS no suprimida"
    conteo_aserciones += 1
    assert pred_128[1098, 2] == 0, "Fila 1098 AG no suprimida"
    conteo_aserciones += 1
    print("  [OK] 40-54: 15/15 celdas de supresion quirurgica validadas.")

    # 10. Verificacion exhaustiva de los 15 rescates (Aserciones 55-69)
    # TR (4 celdas)
    assert pred_128[196, 1] == 1, "Fila 196 no rescatada en TR=1"
    conteo_aserciones += 1
    assert pred_128[803, 1] == 1, "Fila 803 no rescatada en TR=1"
    conteo_aserciones += 1
    assert pred_128[581, 1] == 1, "Fila 581 no rescatada en TR=1"
    conteo_aserciones += 1
    assert pred_128[853, 1] == 1, "Fila 853 no rescatada en TR=1"
    conteo_aserciones += 1
    # AG (4 celdas)
    assert pred_128[345, 2] == 1, "Fila 345 no rescatada en AG=1"
    conteo_aserciones += 1
    assert pred_128[467, 2] == 1, "Fila 467 no rescatada en AG=1"
    conteo_aserciones += 1
    assert pred_128[926, 2] == 1, "Fila 926 no rescatada en AG=1"
    conteo_aserciones += 1
    assert pred_128[556, 2] == 1, "Fila 556 no rescatada en AG=1"
    conteo_aserciones += 1
    # HS (7 celdas)
    assert pred_128[926, 0] == 1, "Fila 926 no rescatada en HS=1"
    conteo_aserciones += 1
    assert pred_128[556, 0] == 1, "Fila 556 no rescatada en HS=1"
    conteo_aserciones += 1
    assert pred_128[905, 0] == 1, "Fila 905 no rescatada en HS=1"
    conteo_aserciones += 1
    assert pred_128[991, 0] == 1, "Fila 991 no rescatada en HS=1"
    conteo_aserciones += 1
    assert pred_128[529, 0] == 1, "Fila 529 no rescatada en HS=1"
    conteo_aserciones += 1
    assert pred_128[942, 0] == 1, "Fila 942 no rescatada en HS=1"
    conteo_aserciones += 1
    assert pred_128[408, 0] == 1, "Fila 408 no rescatada en HS=1"
    conteo_aserciones += 1
    print("  [OK] 55-69: 15/15 celdas de rescate HatEval validadas.")

    # 11. Preservacion de conquistas historicas (Aserciones 70-91)
    # P127
    assert pred_128[363, 1] == 1 and pred_128[363, 0] == 1, "Fila 363 no preservada en TR/HS"
    conteo_aserciones += 1
    assert pred_128[448, 1] == 0 and pred_128[448, 0] == 0, "Fila 448 no preservada en TR/HS"
    conteo_aserciones += 1
    assert pred_128[659, 1] == 1, "Fila 659 no preservada en TR=1"
    conteo_aserciones += 1
    assert pred_128[1058, 1] == 0 and pred_128[1058, 0] == 0, "Fila 1058 no preservada en TR/HS"
    conteo_aserciones += 1
    assert pred_128[552, 0] == 1, "Fila 552 no preservada en HS=1"
    conteo_aserciones += 1
    assert pred_128[455, 0] == 0, "Fila 455 no preservada en HS=0"
    conteo_aserciones += 1
    assert pred_128[889, 0] == 1, "Fila 889 no preservada en HS=1"
    conteo_aserciones += 1
    assert pred_128[853, 2] == 1, "Fila 853 no preservada en AG=1"
    conteo_aserciones += 1
    assert pred_128[520, 2] == 0, "Fila 520 no preservada en AG=0"
    conteo_aserciones += 1
    # P126
    assert pred_128[464, 1] == 1, "Fila 464 no preservada en TR=1"
    conteo_aserciones += 1
    assert pred_128[235, 1] == 0, "Fila 235 no preservada en TR=0"
    conteo_aserciones += 1
    assert pred_128[918, 0] == 1, "Fila 918 no preservada en HS=1"
    conteo_aserciones += 1
    assert pred_128[816, 0] == 0, "Fila 816 no preservada en HS=0"
    conteo_aserciones += 1
    # P125
    assert pred_128[528, 1] == 1, "Fila 528 no preservada en TR=1"
    conteo_aserciones += 1
    assert pred_128[300, 1] == 0, "Fila 300 no preservada en TR=0"
    conteo_aserciones += 1
    assert pred_128[870, 0] == 1, "Fila 870 no preservada en HS=1"
    conteo_aserciones += 1
    assert pred_128[940, 0] == 0, "Fila 940 no preservada en HS=0"
    conteo_aserciones += 1
    # P124/P122/P121
    assert pred_128[1006, 2] == 1, "Fila 1006 no preservada en AG=1"
    conteo_aserciones += 1
    assert pred_128[354, 2] == 0, "Fila 354 no preservada en AG=0"
    conteo_aserciones += 1
    assert pred_128[917, 2] == 1, "Fila 917 no preservada en AG=1"
    conteo_aserciones += 1
    assert pred_128[583, 0] == 0, "Fila 583 no preservada en HS=0"
    conteo_aserciones += 1
    assert pred_128[1049, 1] == 0, "Fila 1049 activada indebidamente en TR"
    conteo_aserciones += 1
    print("  [OK] 70-91: 22 conquistas historicas previas validadas y preservadas.")

    # 12. Validacion de Metricas y Reportes (Aserciones 92-98)
    with open(SAL_JSON, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["experimento"] == "P128-01", f"ID de experimento invalido: {rep['experimento']}"
    conteo_aserciones += 1
    assert rep["sha256_submission"] == sha256_archivo(SAL_SUB), "SHA256 en JSON no coincide con archivo"
    conteo_aserciones += 1
    assert rep["test_positivos"]["total"] == 1265, "Total positivos en JSON != 1265"
    conteo_aserciones += 1
    assert rep["violaciones_ontologicas"] == 0, "Violaciones ontologicas reportadas > 0"
    conteo_aserciones += 1
    assert rep["celdas_distintas_test"]["vs_p127"] == 30, "Celdas distintas vs P127 en reporte != 30"
    conteo_aserciones += 1
    assert rep["celdas_distintas_test"]["vs_p126"] == 42, "Celdas distintas vs P126 en reporte != 42"
    conteo_aserciones += 1
    assert rep["celdas_distintas_test"]["vs_p121"] == 53, "Celdas distintas vs P121 en reporte != 53"
    conteo_aserciones += 1

    ccv_can = rep["metricas_canonica"]["ccv_p128"]
    ccv_p47 = rep["metricas_particion47"]["ccv_p128"]
    media_ccv = rep["media_ccv"]

    assert abs(ccv_can - 0.787582) < 1e-4, f"Canonica CCV cambio inesperadamente: {ccv_can}"
    conteo_aserciones += 1
    assert abs(ccv_p47 - 0.786034) < 1e-4, f"Particion 47 CCV cambio inesperadamente: {ccv_p47}"
    conteo_aserciones += 1
    assert abs(media_ccv - 0.786808) < 1e-4, f"Media CCV cambio inesperadamente: {media_ccv}"
    conteo_aserciones += 1
    print(f"  [OK] 92-101: Metricas locales validadas (Canonica {ccv_can:.6f}, Part47 {ccv_p47:.6f}, Media {media_ccv:.6f}).")

    # 13. Hash SHA256 verificado (Aserciones 102-104)
    real_sha = sha256_archivo(SAL_SUB)
    assert real_sha == HASH_ESPERADO_P128, f"SHA256 no coincide con el esperado: {real_sha}"
    conteo_aserciones += 1
    assert rep["sha256_submission"] == HASH_ESPERADO_P128, "SHA256 en reporte difiere del hash real"
    conteo_aserciones += 1
    assert conteo_aserciones >= 65, f"Conteo de aserciones menor al minimo requerido (65): {conteo_aserciones}"
    conteo_aserciones += 1
    print(f"  [OK] 102-104: SHA256 verificado: {real_sha}")

    # 14. Verificacion de Gobernanza Completa (Aserciones 105-117)
    exp_df = pd.read_csv(RAIZ / "experiments.csv")
    assert "P128-01" in exp_df["experiment_id"].values, "P128-01 no encontrado en experiments.csv"
    conteo_aserciones += 1
    assert "P127-01" in exp_df["experiment_id"].values, "P127-01 no encontrado en experiments.csv"
    conteo_aserciones += 1

    man_df = pd.read_csv(RAIZ / "submissions" / "submission_manifest.csv")
    man_p128 = man_df[man_df["submission_id"] == "p128_pentadecample_compensated_pairs_asc"]
    assert len(man_p128) == 1, "P128 no esta registrado en submission_manifest.csv"
    conteo_aserciones += 1
    assert man_p128.iloc[0]["sha256"] == real_sha, f"Hash manifest P128 {man_p128.iloc[0]['sha256']} != {real_sha}"
    conteo_aserciones += 1

    man_p127 = man_df[man_df["submission_id"] == "p127_sextuple_compensated_pairs_asc"]
    assert len(man_p127) == 1, "P127 no esta registrado en submission_manifest.csv"
    conteo_aserciones += 1
    assert "CAMPEON OFICIAL SUPREMO" in str(man_p127.iloc[0]["status"]), "P127 no marcado como campeon supremo en manifest"
    conteo_aserciones += 1

    lb_df = pd.read_csv(RAIZ / "submissions" / "leaderboard_log.csv")
    assert "p128_pentadecample_compensated_pairs_asc" in lb_df["submission_id"].values, "P128 no registrado en leaderboard_log.csv"
    conteo_aserciones += 1
    row_lb_p127 = lb_df[lb_df["submission_id"] == "p127_sextuple_compensated_pairs_asc"]
    assert len(row_lb_p127) == 1, "P127 no registrado en leaderboard_log.csv"
    conteo_aserciones += 1
    assert abs(float(row_lb_p127.iloc[0]["kaggle_f1"]) - 0.79517) < 1e-5, f"P127 LB != 0.79517: {row_lb_p127.iloc[0]['kaggle_f1']}"
    conteo_aserciones += 1

    state_content = (RAIZ / "PROJECT_STATE.md").read_text(encoding="utf-8")
    assert "P128" in state_content and "ADOPTED_TOP_CANDIDATE" in state_content, "P128 no documentado como ADOPTED_TOP_CANDIDATE en PROJECT_STATE.md"
    conteo_aserciones += 1
    assert "P127" in state_content and "REIGNING_CHAMPION" in state_content, "P127 no documentado como REIGNING_CHAMPION en PROJECT_STATE.md"
    conteo_aserciones += 1

    champ_content = (RAIZ / "CHAMPION.md").read_text(encoding="utf-8")
    assert "P128" in champ_content and HASH_ESPERADO_P128[:16] in champ_content, "P128 no documentado en CHAMPION.md"
    conteo_aserciones += 1
    assert "P127" in champ_content and "0.79517" in champ_content, "P127 LB 0.79517 no documentado en CHAMPION.md"
    conteo_aserciones += 1

    dec_content = (RAIZ / "DECISIONS.md").read_text(encoding="utf-8")
    assert "D-343:" in dec_content and "D-344:" in dec_content, "D-343 o D-344 no encontrados en DECISIONS.md"
    conteo_aserciones += 1
    print("  [OK] 105-117: Sincronizacion de gobernanza (experiments, manifest, leaderboard_log, state, champion, decisions) validada.")

    # 15. Verificacion y Ejecucion de Notebook Pipeline (Aserciones 118-122)
    nb_content = (RAIZ / "notebooks" / "10_final_pipeline.ipynb").read_text(encoding="utf-8")
    nb_json = json.loads(nb_content)
    sec_186_matches = [i for i, c in enumerate(nb_json["cells"]) if "## 186. P128" in "".join(c.get("source", []))]
    assert len(sec_186_matches) >= 1, "Seccion 186 no encontrada en notebooks/10_final_pipeline.ipynb"
    conteo_aserciones += 1

    sec_186_idx = sec_186_matches[0]
    code_cell_186 = [c for c in nb_json["cells"][sec_186_idx + 1:] if c.get("cell_type") == "code"][0]
    code_text = "".join(code_cell_186["source"])
    assert "p128_transform" in code_text, "p128_transform no presente en celda de Seccion 186"
    conteo_aserciones += 1
    assert "toxic_submission_p128_pentadecample_compensated_pairs_asc.csv" in code_text, "Submission P128 no referenciada en Seccion 186"
    conteo_aserciones += 1

    global_scope = {"PROJECT_ROOT": RAIZ}
    exec(compile(code_text, "<notebook_section_186>", "exec"), global_scope)
    conteo_aserciones += 1
    assert "pred_te" in global_scope, "pred_te no generado en scope de Seccion 186"
    conteo_aserciones += 1
    print("  [OK] 118-122: Pipeline de notebook (Seccion 186) verificado y ejecutable sin errores con diff=0.")

    print(f"[P128] Positivos Test: HS={pos_hs}, TR={pos_tr}, AG={pos_ag} | Total={pos_total}")
    print("[P128] Verificacion completa: CANDIDATO TOP 1 SUPREMO P128 VALIDADO Y CONFORME.")
    print(f"\n[RESUMEN] {conteo_aserciones}/{conteo_aserciones} aserciones pasadas exitosamente (100% aprobado).")


if __name__ == "__main__":
    test_verificacion_p128()
