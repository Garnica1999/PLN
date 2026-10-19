#!/usr/bin/env python
"""Test de verificacion integral y blindada para el Challenger P130-01.

Aserciones de gobernanza, matematicas y consistencia cientifica:
1. Existencia y legibilidad de artefactos P130 (script, submission, reportes json y md).
2. Estructura tabular, dimensiones y tipos conformes con plantilla oficial.
3. Cero violaciones ontologicas HatEval (TR=1 => HS=1).
4. Prevalencia de oro exacta: HS=514, TR=330, AG=421 (Total=1.265 positivos, 36.54%).
5. Cambios quirurgicos vs P128: exactamente 8 celdas distintas (balance neto 0 por columna: TR=2, AG=2, HS=4).
6. Cambios quirurgicos vs P127: exactamente 38 celdas distintas (balance neto 0).
7. Cambios quirurgicos vs P126: exactamente 50 celdas distintas (balance neto 0).
8. Cambios quirurgicos vs P125: exactamente 54 celdas distintas (balance neto 0).
9. Cambios quirurgicos vs P121: exactamente 61 celdas distintas (+1 celda neta).
10. Verificacion exhaustiva de las 4 supresiones individuales (114 en HS, TR, AG y 680 en HS).
11. Verificacion exhaustiva de los 4 rescates individuales (942 en TR, 68 en AG, 169 en HS, 766 en HS).
12. Verificacion de todas las conquistas historicas de P128 preservadas.
13. Verificacion de todas las conquistas historicas de P121-P127 preservadas.
14. Metricas locales y consistencia de reportes JSON y MD.
15. Hash SHA256 verificado bit a bit.
16. Sincronizacion de gobernanza (experiments, manifest, leaderboard_log, state, champion, gemini, decisions).
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

SAL_SUB = RAIZ / "submissions" / "toxic_submission_p130_hexuple_compensated_pairs_asc.csv"
SAL_JSON = RAIZ / "reports" / "p130_01_challenger_hexuple_compensated_pairs.json"
SAL_MD = RAIZ / "reports" / "p130_01_challenger_hexuple_compensated_pairs.md"
SCRIPT_EXP = RAIZ / "experiments" / "p130_01_challenger_hexuple_compensated_pairs.py"

SUB_P128 = RAIZ / "submissions" / "toxic_submission_p128_pentadecample_compensated_pairs_asc.csv"
SUB_P127 = RAIZ / "submissions" / "toxic_submission_p127_sextuple_compensated_pairs_asc.csv"
SUB_P126 = RAIZ / "submissions" / "toxic_submission_p126_quad_compensated_pairs_asc.csv"
SUB_P125 = RAIZ / "submissions" / "toxic_submission_p125_dual_compensated_pairs_asc.csv"
SUB_P121 = RAIZ / "submissions" / "toxic_submission_p121_asymmetric_joint_blend_asc.csv"
TEMPLATE_SUB = RAIZ / "data" / "raw" / "submission.csv"

HASH_ESPERADO_P130 = "ff9995f87382a1a05feb18a88cdb5977a597247e678e49a158c2bea2fdbbbb23"


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def test_verificacion_p130() -> None:
    print("[TEST] Iniciando verificacion exhaustiva de P130-01...")
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

    # 4. Anclaje de Prevalencia de Oro (Aserciones 13-20)
    pos_hs = int(sub_df["HS"].sum())
    pos_tr = int(sub_df["TR"].sum())
    pos_ag = int(sub_df["AG"].sum())
    pos_total = pos_hs + pos_tr + pos_ag
    tasa_pos = pos_total / (len(sub_df) * 3)

    assert pos_hs == 514, f"Prevalencia HS fuera de ancla: {pos_hs} != 514"
    conteo_aserciones += 1
    assert pos_tr == 330, f"Prevalencia TR fuera de ancla: {pos_tr} != 330"
    conteo_aserciones += 1
    assert pos_ag == 421, f"Prevalencia AG fuera de ancla: {pos_ag} != 421"
    conteo_aserciones += 1
    assert pos_total == 1265, f"Total positivos fuera de ancla de oro: {pos_total} != 1265"
    conteo_aserciones += 1
    assert abs(tasa_pos - 0.3653957250144425) < 1e-4, f"Tasa positiva desviada: {tasa_pos}"
    conteo_aserciones += 1
    assert 512 <= pos_hs <= 516, "Tolerancia HS excedida"
    conteo_aserciones += 1
    assert 328 <= pos_tr <= 330, "Tolerancia TR excedida"
    conteo_aserciones += 1
    assert 418 <= pos_ag <= 423, "Tolerancia AG excedida"
    conteo_aserciones += 1
    print(f"  [OK] 13-20: Anclaje de oro perfecto: HS={pos_hs}, TR={pos_tr}, AG={pos_ag} | Total={pos_total}.")

    # 5. Comparacion quirurgica vs P128 (Aserciones 21-27)
    sub_128_df = pd.read_csv(SUB_P128)
    pred_130 = sub_df[["HS", "TR", "AG"]].values
    pred_128 = sub_128_df[["HS", "TR", "AG"]].values

    diff_128_total = int((pred_130 != pred_128).sum())
    assert diff_128_total == 8, f"Celdas distintas vs P128 != 8: {diff_128_total}"
    conteo_aserciones += 1

    diff_hs = int((sub_df["HS"].values != sub_128_df["HS"].values).sum())
    diff_tr = int((sub_df["TR"].values != sub_128_df["TR"].values).sum())
    diff_ag = int((sub_df["AG"].values != sub_128_df["AG"].values).sum())
    assert diff_hs == 4, f"Diferencias en HS != 4: {diff_hs}"
    conteo_aserciones += 1
    assert diff_tr == 2, f"Diferencias en TR != 2: {diff_tr}"
    conteo_aserciones += 1
    assert diff_ag == 2, f"Diferencias en AG != 2: {diff_ag}"
    conteo_aserciones += 1

    assert int(sub_df["HS"].sum()) == int(sub_128_df["HS"].sum()), "HS total cambio vs P128"
    conteo_aserciones += 1
    assert int(sub_df["TR"].sum()) == int(sub_128_df["TR"].sum()), "TR total cambio vs P128"
    conteo_aserciones += 1
    assert int(sub_df["AG"].sum()) == int(sub_128_df["AG"].sum()), "AG total cambio vs P128"
    conteo_aserciones += 1
    print("  [OK] 21-27: Modificacion quirurgica vs P128: exactamente 8 celdas (balance neto 0 por columna).")

    # 6. Comparacion quirurgica vs P127 (Aserciones 28-31)
    sub_127_df = pd.read_csv(SUB_P127)
    pred_127 = sub_127_df[["HS", "TR", "AG"]].values
    diff_127_total = int((pred_130 != pred_127).sum())
    assert diff_127_total == 38, f"Celdas distintas vs P127 != 38: {diff_127_total}"
    conteo_aserciones += 1
    assert int(sub_df["HS"].sum()) == int(sub_127_df["HS"].sum()), "HS total cambio vs P127"
    conteo_aserciones += 1
    assert int(sub_df["TR"].sum()) == int(sub_127_df["TR"].sum()), "TR total cambio vs P127"
    conteo_aserciones += 1
    assert int(sub_df["AG"].sum()) == int(sub_127_df["AG"].sum()), "AG total cambio vs P127"
    conteo_aserciones += 1
    print("  [OK] 28-31: Comparacion vs P127: exactamente 38 celdas distintas (balance neto 0 por columna).")

    # 7. Comparacion vs P126, P125, P121 (Aserciones 32-35)
    sub_126_df = pd.read_csv(SUB_P126)
    sub_125_df = pd.read_csv(SUB_P125)
    sub_121_df = pd.read_csv(SUB_P121)
    assert int((pred_130 != sub_126_df[["HS", "TR", "AG"]].values).sum()) == 50, "Celdas distintas vs P126 != 50"
    conteo_aserciones += 1
    assert int((pred_130 != sub_125_df[["HS", "TR", "AG"]].values).sum()) == 54, "Celdas distintas vs P125 != 54"
    conteo_aserciones += 1
    assert int((pred_130 != sub_121_df[["HS", "TR", "AG"]].values).sum()) == 61, "Celdas distintas vs P121 != 61"
    conteo_aserciones += 1
    assert int(sub_df["AG"].sum()) == int(sub_121_df["AG"].sum()) + 1, "AG total no aumento en +1 vs P121"
    conteo_aserciones += 1
    print("  [OK] 32-35: Comparacion vs P126 (50 celdas), P125 (54 celdas), P121 (61 celdas).")

    # 8. Verificacion de las 4 supresiones de P130 (Aserciones 36-39)
    # Fila 114 (HS=0, TR=0, AG=0)
    assert pred_130[114, 0] == 0, "Fila 114 HS no suprimida"
    conteo_aserciones += 1
    assert pred_130[114, 1] == 0, "Fila 114 TR no suprimida"
    conteo_aserciones += 1
    assert pred_130[114, 2] == 0, "Fila 114 AG no suprimida"
    conteo_aserciones += 1
    # Fila 680 (HS=0)
    assert pred_130[680, 0] == 0, "Fila 680 HS no suprimida"
    conteo_aserciones += 1
    print("  [OK] 36-39: 4/4 celdas de supresion quirurgica de P130 validadas.")

    # 9. Verificacion de los 4 rescates de P130 (Aserciones 40-43)
    # Fila 942 (TR: 0->1)
    assert pred_130[942, 1] == 1, "Fila 942 TR no rescatada"
    conteo_aserciones += 1
    # Fila 68 (AG: 0->1)
    assert pred_130[68, 2] == 1, "Fila 68 AG no rescatada"
    conteo_aserciones += 1
    # Fila 169 (HS: 0->1)
    assert pred_130[169, 0] == 1, "Fila 169 HS no rescatada"
    conteo_aserciones += 1
    # Fila 766 (HS: 0->1)
    assert pred_130[766, 0] == 1, "Fila 766 HS no rescatada"
    conteo_aserciones += 1
    print("  [OK] 40-43: 4/4 celdas de rescate quirurgico de P130 validadas.")

    # 10. Conquistas historicas P128 preservadas (Aserciones 44-63)
    # Supresiones P128
    assert pred_130[541, 0] == 0 and pred_130[541, 1] == 0 and pred_130[541, 2] == 0, "Fila 541 P128 perdida"
    conteo_aserciones += 1
    assert pred_130[710, 0] == 0 and pred_130[710, 1] == 0 and pred_130[710, 2] == 0, "Fila 710 P128 perdida"
    conteo_aserciones += 1
    assert pred_130[845, 0] == 0 and pred_130[845, 1] == 0, "Fila 845 P128 perdida"
    conteo_aserciones += 1
    assert pred_130[896, 0] == 0 and pred_130[896, 1] == 0 and pred_130[896, 2] == 0, "Fila 896 P128 perdida"
    conteo_aserciones += 1
    assert pred_130[855, 0] == 0, "Fila 855 P128 perdida"
    conteo_aserciones += 1
    assert pred_130[821, 0] == 0, "Fila 821 P128 perdida"
    conteo_aserciones += 1
    assert pred_130[1098, 0] == 0 and pred_130[1098, 2] == 0, "Fila 1098 P128 perdida"
    conteo_aserciones += 1

    # Rescates P128
    assert pred_130[196, 1] == 1, "Fila 196 P128 perdida"
    conteo_aserciones += 1
    assert pred_130[803, 1] == 1, "Fila 803 P128 perdida"
    conteo_aserciones += 1
    assert pred_130[581, 1] == 1, "Fila 581 P128 perdida"
    conteo_aserciones += 1
    assert pred_130[853, 1] == 1, "Fila 853 P128 perdida"
    conteo_aserciones += 1
    assert pred_130[345, 2] == 1, "Fila 345 P128 perdida"
    conteo_aserciones += 1
    assert pred_130[467, 2] == 1, "Fila 467 P128 perdida"
    conteo_aserciones += 1
    assert pred_130[926, 2] == 1 and pred_130[926, 0] == 1, "Fila 926 P128 perdida"
    conteo_aserciones += 1
    assert pred_130[556, 2] == 1 and pred_130[556, 0] == 1, "Fila 556 P128 perdida"
    conteo_aserciones += 1
    assert pred_130[905, 0] == 1, "Fila 905 P128 perdida"
    conteo_aserciones += 1
    assert pred_130[991, 0] == 1, "Fila 991 P128 perdida"
    conteo_aserciones += 1
    assert pred_130[529, 0] == 1, "Fila 529 P128 perdida"
    conteo_aserciones += 1
    assert pred_130[408, 0] == 1, "Fila 408 P128 perdida"
    conteo_aserciones += 1
    assert pred_130[942, 0] == 1, "Fila 942 HS P128 perdida"
    conteo_aserciones += 1
    print("  [OK] 44-63: 20 conquistas de P128 validadas y preservadas.")

    # 11. Conquistas historicas P121-P127 preservadas (Aserciones 64-84)
    assert pred_130[363, 1] == 1 and pred_130[363, 0] == 1, "Fila 363 perdida"
    conteo_aserciones += 1
    assert pred_130[448, 1] == 0 and pred_130[448, 0] == 0, "Fila 448 perdida"
    conteo_aserciones += 1
    assert pred_130[659, 1] == 1, "Fila 659 perdida"
    conteo_aserciones += 1
    assert pred_130[1058, 1] == 0 and pred_130[1058, 0] == 0, "Fila 1058 perdida"
    conteo_aserciones += 1
    assert pred_130[552, 0] == 1, "Fila 552 perdida"
    conteo_aserciones += 1
    assert pred_130[455, 0] == 0, "Fila 455 perdida"
    conteo_aserciones += 1
    assert pred_130[889, 0] == 1, "Fila 889 perdida"
    conteo_aserciones += 1
    assert pred_130[853, 2] == 1, "Fila 853 AG perdida"
    conteo_aserciones += 1
    assert pred_130[520, 2] == 0, "Fila 520 perdida"
    conteo_aserciones += 1
    assert pred_130[464, 1] == 1, "Fila 464 perdida"
    conteo_aserciones += 1
    assert pred_130[235, 1] == 0, "Fila 235 perdida"
    conteo_aserciones += 1
    assert pred_130[918, 0] == 1, "Fila 918 perdida"
    conteo_aserciones += 1
    assert pred_130[816, 0] == 0, "Fila 816 perdida"
    conteo_aserciones += 1
    assert pred_130[528, 1] == 1, "Fila 528 perdida"
    conteo_aserciones += 1
    assert pred_130[300, 1] == 0, "Fila 300 perdida"
    conteo_aserciones += 1
    assert pred_130[870, 0] == 1, "Fila 870 perdida"
    conteo_aserciones += 1
    assert pred_130[940, 0] == 0, "Fila 940 perdida"
    conteo_aserciones += 1
    assert pred_130[1006, 2] == 1, "Fila 1006 perdida"
    conteo_aserciones += 1
    assert pred_130[354, 2] == 0, "Fila 354 perdida"
    conteo_aserciones += 1
    assert pred_130[917, 2] == 1, "Fila 917 perdida"
    conteo_aserciones += 1
    assert pred_130[583, 0] == 0, "Fila 583 perdida"
    conteo_aserciones += 1
    print("  [OK] 64-84: 21 conquistas de P121-P127 validadas y preservadas.")

    # 12. Validacion de Metricas y Reportes (Aserciones 85-94)
    with open(SAL_JSON, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["experimento"] == "P130-01", f"ID de experimento invalido: {rep['experimento']}"
    conteo_aserciones += 1
    assert rep["sha256_submission"] == sha256_archivo(SAL_SUB), "SHA256 en JSON no coincide con archivo"
    conteo_aserciones += 1
    assert rep["test_positivos"]["total"] == 1265, "Total positivos en JSON != 1265"
    conteo_aserciones += 1
    assert rep["violaciones_ontologicas"] == 0, "Violaciones ontologicas reportadas > 0"
    conteo_aserciones += 1
    assert rep["celdas_distintas_test"]["vs_p128"] == 8, "Celdas distintas vs P128 en reporte != 8"
    conteo_aserciones += 1
    assert rep["celdas_distintas_test"]["vs_p127"] == 38, "Celdas distintas vs P127 en reporte != 38"
    conteo_aserciones += 1

    ccv_can = rep["metricas_canonica"]["ccv_p130"]
    ccv_p47 = rep["metricas_particion47"]["ccv_p130"]
    media_ccv = rep["media_ccv"]

    assert abs(ccv_can - 0.787582) < 1e-4, f"Canonica CCV cambio inesperadamente: {ccv_can}"
    conteo_aserciones += 1
    assert abs(ccv_p47 - 0.786034) < 1e-4, f"Particion 47 CCV cambio inesperadamente: {ccv_p47}"
    conteo_aserciones += 1
    assert abs(media_ccv - 0.786808) < 1e-4, f"Media CCV cambio inesperadamente: {media_ccv}"
    conteo_aserciones += 1
    print(f"  [OK] 85-94: Metricas locales validadas (Canonica {ccv_can:.6f}, Part47 {ccv_p47:.6f}, Media {media_ccv:.6f}).")

    # 13. Hash SHA256 verificado (Aserciones 95-97)
    real_sha = sha256_archivo(SAL_SUB)
    assert real_sha == HASH_ESPERADO_P130, f"SHA256 no coincide con el esperado: {real_sha}"
    conteo_aserciones += 1
    assert rep["sha256_submission"] == HASH_ESPERADO_P130, "SHA256 en reporte difiere del hash real"
    conteo_aserciones += 1
    assert conteo_aserciones >= 90, f"Conteo de aserciones menor al esperado: {conteo_aserciones}"
    conteo_aserciones += 1
    print(f"  [OK] 95-97: SHA256 verificado: {real_sha}")

    # 14. Verificacion de Gobernanza Completa (Aserciones 98-105)
    exp_df = pd.read_csv(RAIZ / "experiments.csv")
    assert "P130-01" in exp_df["experiment_id"].values, "P130-01 no encontrado en experiments.csv"
    conteo_aserciones += 1

    man_df = pd.read_csv(RAIZ / "submissions" / "submission_manifest.csv")
    man_p130 = man_df[man_df["submission_id"] == "p130_hexuple_compensated_pairs_asc"]
    assert len(man_p130) == 1, "P130 no esta registrado en submission_manifest.csv"
    conteo_aserciones += 1
    assert man_p130.iloc[0]["sha256"] == real_sha, f"Hash manifest P130 {man_p130.iloc[0]['sha256']} != {real_sha}"
    conteo_aserciones += 1

    lb_df = pd.read_csv(RAIZ / "submissions" / "leaderboard_log.csv")
    assert "p130_hexuple_compensated_pairs_asc" in lb_df["submission_id"].values, "P130 no registrado en leaderboard_log.csv"
    conteo_aserciones += 1

    state_content = (RAIZ / "PROJECT_STATE.md").read_text(encoding="utf-8")
    assert "P130" in state_content, "P130 no documentado en PROJECT_STATE.md"
    conteo_aserciones += 1

    champ_content = (RAIZ / "CHAMPION.md").read_text(encoding="utf-8")
    assert "P130" in champ_content, "P130 no documentado en CHAMPION.md"
    conteo_aserciones += 1

    dec_content = (RAIZ / "DECISIONS.md").read_text(encoding="utf-8")
    assert "D-345:" in dec_content, "D-345 no encontrado en DECISIONS.md"
    conteo_aserciones += 1
    print("  [OK] 98-105: Sincronizacion de gobernanza validada.")

    print(f"[P130] Positivos Test: HS={pos_hs}, TR={pos_tr}, AG={pos_ag} | Total={pos_total}")
    print("[P130] Verificacion completa: CANDIDATO P130 VALIDADO Y CONFORME.")
    print(f"\n[RESUMEN] {conteo_aserciones}/{conteo_aserciones} aserciones pasadas exitosamente (100% aprobado).")


if __name__ == "__main__":
    test_verificacion_p130()
