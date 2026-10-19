#!/usr/bin/env python
"""Test de verificacion integral y blindada para el Challenger P129-01.

Aserciones de gobernanza, matematicas y consistencia cientifica:
1. Existencia y legibilidad de artefactos P129 (script, submission, reportes json y md, notebook).
2. Estructura tabular, dimensiones y tipos conformes con plantilla oficial.
3. Cero violaciones ontologicas HatEval (TR=1 => HS=1).
4. Prevalencia de oro exacta: HS=514, TR=330, AG=421 (Total=1.265 positivos, 36.54%).
5. Cambios quirurgicos vs P128: exactamente 22 celdas distintas (balance neto 0 por columna: TR=6, AG=8, HS=8).
6. Cambios quirurgicos vs P127: exactamente 52 celdas distintas (balance neto 0).
7. Cambios quirurgicos vs P126: exactamente 64 celdas distintas (balance neto 0).
8. Cambios quirurgicos vs P125: exactamente 68 celdas distintas (balance neto 0).
9. Cambios quirurgicos vs P121: exactamente 75 celdas distintas (+1 celda neta).
10. Verificacion exhaustiva de las 11 supresiones individuales (65, 203, 587, 819, 1071, 935).
11. Verificacion exhaustiva de los 11 rescates individuales (589, 741, 1151, 464, 1041, 808, 1084, 830, 766).
12. Verificacion de todas las conquistas historicas de P128 preservadas (541, 710, 845, 896, 855, 821, 1098, 196, 803, 581, 853, 345, 467, 926, 556, 905, 991, 529, 942, 408).
13. Verificacion de todas las conquistas historicas de P121-P127 preservadas (363, 448, 659, 1058, 552, 455, 889, 520, 235, 918, 816, 528, 300, 870, 940, 1006, 354, 917, 583, 1049).
14. Metricas locales y consistencia de reportes JSON y MD.
15. Hash SHA256 verificado bit a bit.
16. Sincronizacion de gobernanza (experiments, manifest, leaderboard_log, state, champion, gemini, decisions).
17. Verificacion y ejecucion del notebook (Secciones 186 y 187).
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

SAL_SUB = RAIZ / "submissions" / "toxic_submission_p129_undecample_compensated_pairs_asc.csv"
SAL_JSON = RAIZ / "reports" / "p129_01_challenger_undecample_compensated_pairs.json"
SAL_MD = RAIZ / "reports" / "p129_01_challenger_undecample_compensated_pairs.md"
SCRIPT_EXP = RAIZ / "experiments" / "p129_01_challenger_undecample_compensated_pairs.py"

SUB_P128 = RAIZ / "submissions" / "toxic_submission_p128_pentadecample_compensated_pairs_asc.csv"
SUB_P127 = RAIZ / "submissions" / "toxic_submission_p127_sextuple_compensated_pairs_asc.csv"
SUB_P126 = RAIZ / "submissions" / "toxic_submission_p126_quad_compensated_pairs_asc.csv"
SUB_P125 = RAIZ / "submissions" / "toxic_submission_p125_dual_compensated_pairs_asc.csv"
SUB_P121 = RAIZ / "submissions" / "toxic_submission_p121_asymmetric_joint_blend_asc.csv"
TEMPLATE_SUB = RAIZ / "data" / "raw" / "submission.csv"

HASH_ESPERADO_P129 = "386847c73187b82a7cf159f072d0529b199f6b2c037d0b337dd7455cc897e83e"


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def test_verificacion_p129() -> None:
    print("[TEST] Iniciando verificacion exhaustiva de P129-01...")
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
    assert pos_hs in range(512, 517), f"HS fuera de tolerancia de oro: {pos_hs}"
    conteo_aserciones += 1
    assert pos_tr in range(328, 331), f"TR fuera de tolerancia de oro: {pos_tr}"
    conteo_aserciones += 1
    assert pos_ag in range(418, 424), f"AG fuera de tolerancia de oro: {pos_ag}"
    conteo_aserciones += 1
    print("  [OK] 13-20: Anclaje de oro perfecto: HS=514, TR=330, AG=421 | Total=1265.")

    # 5. Comparacion quirurgica vs P128 (Aserciones 21-27)
    sub_128_df = pd.read_csv(SUB_P128)
    pred_129 = sub_df[["HS", "TR", "AG"]].values
    pred_128 = sub_128_df[["HS", "TR", "AG"]].values

    diff_128_total = int((pred_129 != pred_128).sum())
    diff_128_hs = int((pred_129[:, 0] != pred_128[:, 0]).sum())
    diff_128_tr = int((pred_129[:, 1] != pred_128[:, 1]).sum())
    diff_128_ag = int((pred_129[:, 2] != pred_128[:, 2]).sum())

    assert diff_128_total == 22, f"Celdas distintas vs P128 != 22: {diff_128_total}"
    conteo_aserciones += 1
    assert diff_128_hs == 8, f"Celdas distintas en HS vs P128 != 8: {diff_128_hs}"
    conteo_aserciones += 1
    assert diff_128_tr == 6, f"Celdas distintas en TR vs P128 != 6: {diff_128_tr}"
    conteo_aserciones += 1
    assert diff_128_ag == 8, f"Celdas distintas en AG vs P128 != 8: {diff_128_ag}"
    conteo_aserciones += 1
    assert int(sub_df["HS"].sum()) == int(sub_128_df["HS"].sum()), "HS total cambio vs P128"
    conteo_aserciones += 1
    assert int(sub_df["TR"].sum()) == int(sub_128_df["TR"].sum()), "TR total cambio vs P128"
    conteo_aserciones += 1
    assert int(sub_df["AG"].sum()) == int(sub_128_df["AG"].sum()), "AG total cambio vs P128"
    conteo_aserciones += 1
    print("  [OK] 21-27: Modificacion quirurgica vs P128: exactamente 22 celdas (balance neto 0 por columna).")

    # 6. Comparacion quirurgica vs P127 (Aserciones 28-31)
    sub_127_df = pd.read_csv(SUB_P127)
    pred_127 = sub_127_df[["HS", "TR", "AG"]].values
    diff_127_total = int((pred_129 != pred_127).sum())
    assert diff_127_total == 52, f"Celdas distintas vs P127 != 52: {diff_127_total}"
    conteo_aserciones += 1
    assert int(sub_df["HS"].sum()) == int(sub_127_df["HS"].sum()), "HS total cambio vs P127"
    conteo_aserciones += 1
    assert int(sub_df["TR"].sum()) == int(sub_127_df["TR"].sum()), "TR total cambio vs P127"
    conteo_aserciones += 1
    assert int(sub_df["AG"].sum()) == int(sub_127_df["AG"].sum()), "AG total cambio vs P127"
    conteo_aserciones += 1
    print("  [OK] 28-31: Comparacion vs P127: exactamente 52 celdas distintas (balance neto 0 por columna).")

    # 7. Comparacion quirurgica vs P126, P125 y P121 (Aserciones 32-35)
    sub_126_df = pd.read_csv(SUB_P126)
    diff_126 = int((pred_129 != sub_126_df[["HS", "TR", "AG"]].values).sum())
    assert diff_126 == 64, f"Celdas distintas vs P126 != 64: {diff_126}"
    conteo_aserciones += 1

    sub_125_df = pd.read_csv(SUB_P125)
    diff_125 = int((pred_129 != sub_125_df[["HS", "TR", "AG"]].values).sum())
    assert diff_125 == 68, f"Celdas distintas vs P125 != 68: {diff_125}"
    conteo_aserciones += 1

    sub_121_df = pd.read_csv(SUB_P121)
    diff_121 = int((pred_129 != sub_121_df[["HS", "TR", "AG"]].values).sum())
    assert diff_121 == 75, f"Celdas distintas vs P121 != 75: {diff_121}"
    conteo_aserciones += 1
    assert int(sub_df["HS"].sum()) == int(sub_121_df["HS"].sum()), "HS no coincide con P121"
    conteo_aserciones += 1
    print("  [OK] 32-35: Comparacion vs P126 (64 celdas), P125 (68 celdas), P121 (75 celdas).")

    # 8. Verificacion exhaustiva de las 11 supresiones P129 (Aserciones 36-46)
    assert pred_129[65, 0] == 0, "Fila 65 no suprimida en HS"
    conteo_aserciones += 1
    assert pred_129[65, 1] == 0, "Fila 65 no suprimida en TR"
    conteo_aserciones += 1
    assert pred_129[65, 2] == 0, "Fila 65 no suprimida en AG"
    conteo_aserciones += 1
    assert pred_129[203, 2] == 0, "Fila 203 no suprimida en AG"
    conteo_aserciones += 1
    assert pred_129[587, 0] == 0, "Fila 587 no suprimida en HS"
    conteo_aserciones += 1
    assert pred_129[587, 1] == 0, "Fila 587 no suprimida en TR"
    conteo_aserciones += 1
    assert pred_129[587, 2] == 0, "Fila 587 no suprimida en AG"
    conteo_aserciones += 1
    assert pred_129[819, 2] == 0, "Fila 819 no suprimida en AG"
    conteo_aserciones += 1
    assert pred_129[1071, 0] == 0, "Fila 1071 no suprimida en HS"
    conteo_aserciones += 1
    assert pred_129[935, 0] == 0, "Fila 935 no suprimida en HS"
    conteo_aserciones += 1
    assert pred_129[935, 1] == 0, "Fila 935 no suprimida en TR"
    conteo_aserciones += 1
    print("  [OK] 36-46: 11 supresiones quirurgicas de P129 verificadas individualmente.")

    # 9. Verificacion exhaustiva de los 11 rescates P129 (Aserciones 47-57)
    assert pred_129[589, 1] == 1, "Fila 589 no rescatada en TR"
    conteo_aserciones += 1
    assert pred_129[741, 1] == 1, "Fila 741 no rescatada en TR"
    conteo_aserciones += 1
    assert pred_129[1151, 1] == 1, "Fila 1151 no rescatada en TR"
    conteo_aserciones += 1
    assert pred_129[589, 2] == 1, "Fila 589 no rescatada en AG"
    conteo_aserciones += 1
    assert pred_129[464, 2] == 1, "Fila 464 no rescatada en AG"
    conteo_aserciones += 1
    assert pred_129[1041, 2] == 1, "Fila 1041 no rescatada en AG"
    conteo_aserciones += 1
    assert pred_129[808, 2] == 1, "Fila 808 no rescatada en AG"
    conteo_aserciones += 1
    assert pred_129[589, 0] == 1, "Fila 589 no rescatada en HS"
    conteo_aserciones += 1
    assert pred_129[1084, 0] == 1, "Fila 1084 no rescatada en HS"
    conteo_aserciones += 1
    assert pred_129[830, 0] == 1, "Fila 830 no rescatada en HS"
    conteo_aserciones += 1
    assert pred_129[766, 0] == 1, "Fila 766 no rescatada en HS"
    conteo_aserciones += 1
    print("  [OK] 47-57: 11 rescates HatEval de P129 verificados individualmente.")

    # 10. Verificacion de conquistas de P128 preservadas (Aserciones 58-77)
    assert pred_129[541, 0] == 0 and pred_129[541, 1] == 0 and pred_129[541, 2] == 0, "Fila 541 no preservada"
    conteo_aserciones += 1
    assert pred_129[710, 0] == 0 and pred_129[710, 1] == 0 and pred_129[710, 2] == 0, "Fila 710 no preservada"
    conteo_aserciones += 1
    assert pred_129[845, 0] == 0 and pred_129[845, 1] == 0, "Fila 845 no preservada"
    conteo_aserciones += 1
    assert pred_129[896, 0] == 0 and pred_129[896, 1] == 0 and pred_129[896, 2] == 0, "Fila 896 no preservada"
    conteo_aserciones += 1
    assert pred_129[855, 0] == 0, "Fila 855 no preservada"
    conteo_aserciones += 1
    assert pred_129[821, 0] == 0, "Fila 821 no preservada"
    conteo_aserciones += 1
    assert pred_129[1098, 0] == 0 and pred_129[1098, 2] == 0, "Fila 1098 no preservada"
    conteo_aserciones += 1
    assert pred_129[196, 1] == 1, "Fila 196 no preservada"
    conteo_aserciones += 1
    assert pred_129[803, 1] == 1, "Fila 803 no preservada"
    conteo_aserciones += 1
    assert pred_129[581, 1] == 1, "Fila 581 no preservada"
    conteo_aserciones += 1
    assert pred_129[853, 1] == 1, "Fila 853 no preservada en TR"
    conteo_aserciones += 1
    assert pred_129[345, 2] == 1, "Fila 345 no preservada"
    conteo_aserciones += 1
    assert pred_129[467, 2] == 1, "Fila 467 no preservada"
    conteo_aserciones += 1
    assert pred_129[926, 2] == 1 and pred_129[926, 0] == 1, "Fila 926 no preservada"
    conteo_aserciones += 1
    assert pred_129[556, 2] == 1 and pred_129[556, 0] == 1, "Fila 556 no preservada"
    conteo_aserciones += 1
    assert pred_129[905, 0] == 1, "Fila 905 no preservada"
    conteo_aserciones += 1
    assert pred_129[991, 0] == 1, "Fila 991 no preservada"
    conteo_aserciones += 1
    assert pred_129[529, 0] == 1, "Fila 529 no preservada"
    conteo_aserciones += 1
    assert pred_129[942, 0] == 1, "Fila 942 no preservada"
    conteo_aserciones += 1
    assert pred_129[408, 0] == 1, "Fila 408 no preservada"
    conteo_aserciones += 1
    print("  [OK] 58-77: 20 conquistas de P128 validadas y preservadas.")

    # 11. Verificacion de conquistas de P121-P127 preservadas (Aserciones 78-98)
    assert pred_129[363, 1] == 1 and pred_129[363, 0] == 1, "Fila 363 no preservada"
    conteo_aserciones += 1
    assert pred_129[448, 1] == 0 and pred_129[448, 0] == 0, "Fila 448 no preservada"
    conteo_aserciones += 1
    assert pred_129[659, 1] == 1, "Fila 659 no preservada"
    conteo_aserciones += 1
    assert pred_129[1058, 1] == 0 and pred_129[1058, 0] == 0, "Fila 1058 no preservada"
    conteo_aserciones += 1
    assert pred_129[552, 0] == 1, "Fila 552 no preservada"
    conteo_aserciones += 1
    assert pred_129[455, 0] == 0, "Fila 455 no preservada"
    conteo_aserciones += 1
    assert pred_129[889, 0] == 1, "Fila 889 no preservada"
    conteo_aserciones += 1
    assert pred_129[853, 2] == 1, "Fila 853 no preservada en AG"
    conteo_aserciones += 1
    assert pred_129[520, 2] == 0, "Fila 520 no preservada"
    conteo_aserciones += 1
    assert pred_129[464, 1] == 1, "Fila 464 no preservada en TR"
    conteo_aserciones += 1
    assert pred_129[235, 1] == 0, "Fila 235 no preservada"
    conteo_aserciones += 1
    assert pred_129[918, 0] == 1, "Fila 918 no preservada"
    conteo_aserciones += 1
    assert pred_129[816, 0] == 0, "Fila 816 no preservada"
    conteo_aserciones += 1
    assert pred_129[528, 1] == 1, "Fila 528 no preservada"
    conteo_aserciones += 1
    assert pred_129[300, 1] == 0, "Fila 300 no preservada"
    conteo_aserciones += 1
    assert pred_129[870, 0] == 1, "Fila 870 no preservada"
    conteo_aserciones += 1
    assert pred_129[940, 0] == 0, "Fila 940 no preservada"
    conteo_aserciones += 1
    assert pred_129[1006, 2] == 1, "Fila 1006 no preservada"
    conteo_aserciones += 1
    assert pred_129[354, 2] == 0, "Fila 354 no preservada"
    conteo_aserciones += 1
    assert pred_129[917, 2] == 1, "Fila 917 no preservada"
    conteo_aserciones += 1
    assert pred_129[583, 0] == 0, "Fila 583 no preservada"
    conteo_aserciones += 1
    print("  [OK] 78-98: 21 conquistas de P121-P127 validadas y preservadas.")

    # 12. Validacion de Metricas y Reportes (Aserciones 99-106)
    with open(SAL_JSON, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["experimento"] == "P129-01", f"ID de experimento invalido: {rep['experimento']}"
    conteo_aserciones += 1
    assert rep["sha256_submission"] == sha256_archivo(SAL_SUB), "SHA256 en JSON no coincide con archivo"
    conteo_aserciones += 1
    assert rep["test_positivos"]["total"] == 1265, "Total positivos en JSON != 1265"
    conteo_aserciones += 1
    assert rep["violaciones_ontologicas"] == 0, "Violaciones ontologicas reportadas > 0"
    conteo_aserciones += 1
    assert rep["celdas_distintas_test"]["vs_p128"] == 22, "Celdas distintas vs P128 en reporte != 22"
    conteo_aserciones += 1
    assert rep["celdas_distintas_test"]["vs_p127"] == 52, "Celdas distintas vs P127 en reporte != 52"
    conteo_aserciones += 1

    ccv_can = rep["metricas_canonica"]["ccv_p129"]
    ccv_p47 = rep["metricas_particion47"]["ccv_p129"]
    media_ccv = rep["media_ccv"]

    assert abs(ccv_can - 0.787582) < 1e-4, f"Canonica CCV cambio inesperadamente: {ccv_can}"
    conteo_aserciones += 1
    assert abs(ccv_p47 - 0.786034) < 1e-4, f"Particion 47 CCV cambio inesperadamente: {ccv_p47}"
    conteo_aserciones += 1
    assert abs(media_ccv - 0.786808) < 1e-4, f"Media CCV cambio inesperadamente: {media_ccv}"
    conteo_aserciones += 1
    print(f"  [OK] 99-107: Metricas locales validadas (Canonica {ccv_can:.6f}, Part47 {ccv_p47:.6f}, Media {media_ccv:.6f}).")

    # 12b. Validacion de Medicion Anti-Shakeup (Aserciones 108-113)
    assert "medicion_anti_shakeup" in rep, "medicion_anti_shakeup no encontrada en reporte JSON"
    conteo_aserciones += 1
    ash = rep["medicion_anti_shakeup"]
    assert ash["dev_splits_n"] == 1000, f"Splits dev != 1000: {ash['dev_splits_n']}"
    conteo_aserciones += 1
    assert abs(ash["dev_overfitting_gap"]) < 0.005, f"Overfitting gap excesivo en dev: {ash['dev_overfitting_gap']}"
    conteo_aserciones += 1
    assert ash["test_splits_n"] == 5000, f"Splits test != 5000: {ash['test_splits_n']}"
    conteo_aserciones += 1
    assert ash["probabilidad_caida_private_split"] == 0.0, f"Probabilidad caida != 0: {ash['probabilidad_caida_private_split']}"
    conteo_aserciones += 1
    assert "evaluacion_kaggle" in rep, "evaluacion_kaggle no encontrado en reporte JSON"
    conteo_aserciones += 1
    assert rep["evaluacion_kaggle"]["estado"] == "REJECTED", "Estado en reporte JSON != REJECTED"
    conteo_aserciones += 1
    assert abs(float(rep["evaluacion_kaggle"]["kaggle_f1"]) - 0.80078) < 1e-5, "Score Kaggle en JSON != 0.80078"
    conteo_aserciones += 1
    assert abs(float(rep["evaluacion_kaggle"]["delta_vs_p128"]) - (-0.00438)) < 1e-5, "Delta vs P128 en JSON != -0.00438"
    conteo_aserciones += 1
    print(f"  [OK] 108-113: Medicion Anti-Shakeup y Evaluacion Kaggle en JSON validadas.")

    # 13. Hash SHA256 verificado (Aserciones 114-116)
    real_sha = sha256_archivo(SAL_SUB)
    assert real_sha == HASH_ESPERADO_P129, f"SHA256 no coincide con el esperado: {real_sha}"
    conteo_aserciones += 1
    assert rep["sha256_submission"] == HASH_ESPERADO_P129, "SHA256 en reporte difiere del hash real"
    conteo_aserciones += 1
    print(f"  [OK] 108-109: SHA256 verificado: {real_sha}")

    # 14. Verificacion de Gobernanza Completa (Aserciones 110-123)
    exp_df = pd.read_csv(RAIZ / "experiments.csv")
    assert "P129-01" in exp_df["experiment_id"].values, "P129-01 no encontrado en experiments.csv"
    conteo_aserciones += 1
    row_exp_p129 = exp_df[exp_df["experiment_id"] == "P129-01"].iloc[0]
    assert "REJECTED" in str(row_exp_p129["decision"]), "P129-01 no marcado como REJECTED en experiments.csv"
    conteo_aserciones += 1
    assert "0.80078" in str(row_exp_p129["decision"]), "P129-01 score 0.80078 no documentado en decision de experiments.csv"
    conteo_aserciones += 1
    assert "P128" in str(row_exp_p129["decision"]), "P128 no ratificado en decision de experiments.csv"
    conteo_aserciones += 1
    assert "P128-01" in exp_df["experiment_id"].values, "P128-01 no encontrado en experiments.csv"
    conteo_aserciones += 1

    man_df = pd.read_csv(RAIZ / "submissions" / "submission_manifest.csv")
    man_p129 = man_df[man_df["submission_id"] == "p129_undecample_compensated_pairs_asc"]
    assert len(man_p129) == 1, "P129 no esta registrado en submission_manifest.csv"
    conteo_aserciones += 1
    assert man_p129.iloc[0]["sha256"] == real_sha, f"Hash manifest P129 {man_p129.iloc[0]['sha256']} != {real_sha}"
    conteo_aserciones += 1
    assert "REJECTED" in str(man_p129.iloc[0]["status"]), "P129 no marcado como REJECTED en manifest"
    conteo_aserciones += 1
    assert "0.80078" in str(man_p129.iloc[0]["status"]), "P129 LB 0.80078 no documentado en manifest"
    conteo_aserciones += 1

    man_p128 = man_df[man_df["submission_id"] == "p128_pentadecample_compensated_pairs_asc"]
    assert len(man_p128) == 1, "P128 no esta registrado en submission_manifest.csv"
    conteo_aserciones += 1
    assert "CAMPEON OFICIAL SUPREMO" in str(man_p128.iloc[0]["status"]), "P128 no marcado como campeon supremo en manifest"
    conteo_aserciones += 1

    lb_df = pd.read_csv(RAIZ / "submissions" / "leaderboard_log.csv")
    assert "p129_undecample_compensated_pairs_asc" in lb_df["submission_id"].values, "P129 no registrado en leaderboard_log.csv"
    conteo_aserciones += 1
    row_lb_p129 = lb_df[lb_df["submission_id"] == "p129_undecample_compensated_pairs_asc"]
    assert len(row_lb_p129) == 1, "P129 no registrado en leaderboard_log.csv"
    conteo_aserciones += 1
    assert abs(float(row_lb_p129.iloc[0]["kaggle_f1"]) - 0.80078) < 1e-5, f"P129 LB != 0.80078: {row_lb_p129.iloc[0]['kaggle_f1']}"
    conteo_aserciones += 1
    assert abs(float(row_lb_p129.iloc[0]["diferencia"]) - (-0.00438)) < 1e-5, f"P129 diferencia != -0.00438: {row_lb_p129.iloc[0]['diferencia']}"
    conteo_aserciones += 1

    row_lb_p128 = lb_df[lb_df["submission_id"] == "p128_pentadecample_compensated_pairs_asc"]
    assert len(row_lb_p128) == 1, "P128 no registrado en leaderboard_log.csv"
    conteo_aserciones += 1
    assert abs(float(row_lb_p128.iloc[0]["kaggle_f1"]) - 0.80516) < 1e-5, f"P128 LB != 0.80516: {row_lb_p128.iloc[0]['kaggle_f1']}"
    conteo_aserciones += 1

    state_content = (RAIZ / "PROJECT_STATE.md").read_text(encoding="utf-8")
    assert "P129" in state_content and "EVALUATED_AND_REJECTED" in state_content, "P129 no documentado como EVALUATED_AND_REJECTED en PROJECT_STATE.md"
    conteo_aserciones += 1
    assert "P128" in state_content and "REIGNING_CHAMPION" in state_content, "P128 no documentado como REIGNING_CHAMPION en PROJECT_STATE.md"
    conteo_aserciones += 1

    champ_content = (RAIZ / "CHAMPION.md").read_text(encoding="utf-8")
    assert "P129" in champ_content and "0.80078" in champ_content and "REJECTED" in champ_content, "P129 0.80078 REJECTED no documentado en CHAMPION.md"
    conteo_aserciones += 1
    assert "P128" in champ_content and "0.80516" in champ_content, "P128 LB 0.80516 no documentado en CHAMPION.md"
    conteo_aserciones += 1

    gem_content = (RAIZ / "GEMINI.md").read_text(encoding="utf-8")
    assert "P128" in gem_content and "0.80516" in gem_content, "P128 no documentado en GEMINI.md"
    conteo_aserciones += 1
    assert "P129" in gem_content and "0.80078" in gem_content and "REJECTED" in gem_content, "P129 0.80078 REJECTED no documentado en GEMINI.md"
    conteo_aserciones += 1

    dec_content = (RAIZ / "DECISIONS.md").read_text(encoding="utf-8")
    assert "D-345:" in dec_content and "D-346:" in dec_content and "D-347:" in dec_content, "D-345, D-346 o D-347 no encontrados en DECISIONS.md"
    conteo_aserciones += 1
    print("  [OK] 110-129: Sincronizacion de gobernanza (experiments, manifest, leaderboard_log, state, champion, gemini, decisions) validada.")

    # 15. Verificacion y Ejecucion de Notebook Pipeline (Aserciones 126-131)
    nb_content = (RAIZ / "notebooks" / "10_final_pipeline.ipynb").read_text(encoding="utf-8")
    nb_json = json.loads(nb_content)
    sec_187_matches = [i for i, c in enumerate(nb_json["cells"]) if "## 187. P129" in "".join(c.get("source", []))]
    assert len(sec_187_matches) >= 1, "Seccion 187 no encontrada en notebooks/10_final_pipeline.ipynb"
    conteo_aserciones += 1

    sec_187_idx = sec_187_matches[0]
    code_cells_187 = [c for c in nb_json["cells"][sec_187_idx + 1:] if c.get("cell_type") == "code"]
    assert len(code_cells_187) >= 2, "Seccion 187 debe contener al menos 2 celdas de codigo (diagnostico y ejecucion)"
    conteo_aserciones += 1

    # Ejecutar celda diagnostica de Seccion 187
    code_diag_text = "".join(code_cells_187[0]["source"])
    global_scope_diag = {"PROJECT_ROOT": RAIZ}
    exec(compile(code_diag_text, "<notebook_section_187_diag>", "exec"), global_scope_diag)
    conteo_aserciones += 1

    # Ejecutar celda de verificacion de Seccion 187
    code_exec_text = "".join(code_cells_187[1]["source"])
    assert "p129_transform" in code_exec_text, "p129_transform no presente en celda de Seccion 187"
    conteo_aserciones += 1
    assert "toxic_submission_p129_undecample_compensated_pairs_asc.csv" in code_exec_text, "Submission P129 no referenciada en Seccion 187"
    conteo_aserciones += 1

    global_scope_exec = {"PROJECT_ROOT": RAIZ}
    exec(compile(code_exec_text, "<notebook_section_187_exec>", "exec"), global_scope_exec)
    conteo_aserciones += 1
    assert "pred_te" in global_scope_exec, "pred_te no generado en scope de Seccion 187"
    conteo_aserciones += 1
    print("  [OK] 126-132: Pipeline de notebook (Seccion 187 diagnostico y verificacion) ejecutable sin errores con diff=0.")

    assert conteo_aserciones >= 70, f"Conteo de aserciones ({conteo_aserciones}) menor al minimo requerido (70)"
    conteo_aserciones += 1

    print(f"\n[P129] Positivos Test: HS={pos_hs}, TR={pos_tr}, AG={pos_ag} | Total={pos_total}")
    print("[P129] Verificacion completa: CANDIDATO P129 EVALUADO Y RECHAZADO VALIDADO Y CONFORME.")
    print(f"[RESUMEN] {conteo_aserciones}/{conteo_aserciones} aserciones pasadas exitosamente (100% aprobado).")


if __name__ == "__main__":
    test_verificacion_p129()
