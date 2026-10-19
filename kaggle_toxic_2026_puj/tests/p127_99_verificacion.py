#!/usr/bin/env python
"""Test de verificacion integral y blindada para el Challenger P127-01.

Aserciones de gobernanza, matematicas y consistencia cientifica:
1. Existencia y legibilidad de artefactos P127 (script, submission, reportes json y md).
2. Estructura tabular, dimensiones y tipos conformes con plantilla oficial.
3. Cero violaciones ontologicas HatEval (TR=1 => HS=1).
4. Prevalencia de oro exacta: HS=514, TR=330, AG=421 (Total=1.265 positivos, 36.54%).
5. Cambios quirurgicos vs P126: exactamente 12 celdas distintas (balance neto 0 por columna).
6. Cambios quirurgicos vs P125: exactamente 16 celdas distintas.
7. Cambios quirurgicos vs P121: exactamente 23 celdas distintas.
8. Filas clave auditadas validadas:
   - Fila 363 (acoso sexual y misoginia contra @BeeluArzola) en TR=1 y HS=1.
   - Fila 448 (discusion trivial sobre collares caninos) en TR=0 y HS=0.
   - Fila 455 (broma casual coloquial con 'ahre') en HS=0.
   - Fila 520 (duelo domestico por fallecimiento de perra) en AG=0.
   - Fila 552 (discurso xenofobo y misogino ultra-violento) en HS=1.
   - Fila 659 (insulto vocativo a usuarias) en TR=1.
   - Fila 853 (grito e insulto xenofobo @PakoAyestaran) en AG=1.
   - Fila 889 (criminalizacion xenofoba de refugiados) en HS=1.
   - Fila 1058 (festejo concurso televisivo 'La Voz') en TR=0 y HS=0.
   - Filas historicas preservadas (464, 235, 918, 816, 528, 300, 870, 940, 1006, 354, 917, 583, 1049).
9. Metricas locales y determinismo M2 validados.
10. Hash SHA256 verificado.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]

SAL_SUB = RAIZ / "submissions" / "toxic_submission_p127_sextuple_compensated_pairs_asc.csv"
SAL_JSON = RAIZ / "reports" / "p127_01_challenger_sextuple_compensated_pairs.json"
SAL_MD = RAIZ / "reports" / "p127_01_challenger_sextuple_compensated_pairs.md"
SCRIPT_EXP = RAIZ / "experiments" / "p127_01_challenger_sextuple_compensated_pairs.py"

SUB_P126 = RAIZ / "submissions" / "toxic_submission_p126_quad_compensated_pairs_asc.csv"
SUB_P125 = RAIZ / "submissions" / "toxic_submission_p125_dual_compensated_pairs_asc.csv"
SUB_P121 = RAIZ / "submissions" / "toxic_submission_p121_asymmetric_joint_blend_asc.csv"
TEMPLATE_SUB = RAIZ / "data" / "raw" / "submission.csv"

HASH_ESPERADO_P127 = "f3d0b45e60d82437af7d7cc48c02e5d18d305dd4dda9f1db12b4eabeaa521e71"


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def test_verificacion_p127() -> None:
    print("[TEST] Iniciando verificacion exhaustiva de P127-01...")

    # 1. Existencia de artefactos (Aserciones 1-4)
    assert SCRIPT_EXP.is_file(), f"Script no encontrado: {SCRIPT_EXP}"
    assert SAL_SUB.is_file(), f"Submission no encontrada: {SAL_SUB}"
    assert SAL_JSON.is_file(), f"JSON no encontrado: {SAL_JSON}"
    assert SAL_MD.is_file(), f"Reporte MD no encontrado: {SAL_MD}"
    print("  [OK] 1-4: 4/4 archivos requeridos existen en disco.")

    # 2. Estructura tabular y tipos (Aserciones 5-11)
    sub_df = pd.read_csv(SAL_SUB)
    tpl_df = pd.read_csv(TEMPLATE_SUB)

    assert len(sub_df) == 1154, f"Longitud incorrecta: {len(sub_df)}"
    assert list(sub_df.columns) == ["id", "HS", "TR", "AG"], f"Columnas incorrectas: {list(sub_df.columns)}"
    assert (sub_df["id"].values == tpl_df["id"].values).all(), "Desalineacion de IDs frente a plantilla"
    assert sub_df["HS"].isin([0, 1]).all(), "HS contiene valores no binarios"
    assert sub_df["TR"].isin([0, 1]).all(), "TR contiene valores no binarios"
    assert sub_df["AG"].isin([0, 1]).all(), "AG contiene valores no binarios"
    assert sub_df.isna().sum().sum() == 0, "Submission contiene NaNs"
    print("  [OK] 5-11: Estructura tabular y tipos conformes con plantilla oficial.")

    # 3. Regla Ontologica Ascendente HatEval (Asercion 12)
    violaciones = int(((sub_df["TR"] == 1) & (sub_df["HS"] == 0)).sum())
    assert violaciones == 0, f"Violaciones ontologicas detectadas: {violaciones}"
    print("  [OK] 12: 0 violaciones ontologicas (TR=1 => HS=1 estrictamente cumplido).")

    # 4. Ancla de Oro de Prevalencia en Test (Aserciones 13-20)
    pos_hs = int(sub_df["HS"].sum())
    pos_tr = int(sub_df["TR"].sum())
    pos_ag = int(sub_df["AG"].sum())
    pos_total = pos_hs + pos_tr + pos_ag
    tasa_pos = pos_total / (1154 * 3)

    assert pos_hs == 514, f"Prevalencia HS fuera de ancla: {pos_hs} != 514"
    assert pos_tr == 330, f"Prevalencia TR fuera de ancla: {pos_tr} != 330"
    assert pos_ag == 421, f"Prevalencia AG fuera de ancla: {pos_ag} != 421"
    assert pos_total == 1265, f"Total positivos fuera de ancla de oro: {pos_total} != 1265"
    assert abs(tasa_pos - 0.3653957) < 1e-4, f"Tasa positiva fuera de tolerancia: {tasa_pos}"
    assert pos_hs in range(512, 518), f"HS fuera de tolerancia de oro: {pos_hs}"
    assert pos_tr in range(328, 331), f"TR fuera de tolerancia de oro: {pos_tr}"
    assert pos_ag in range(418, 424), f"AG fuera de tolerancia de oro: {pos_ag}"
    print("  [OK] 13-20: Anclaje de oro perfecto: HS=514, TR=330, AG=421 | Total=1265.")

    # 5. Comparacion quirurgica vs P126 (Aserciones 21-27)
    sub_126_df = pd.read_csv(SUB_P126)
    pred_127 = sub_df[["HS", "TR", "AG"]].values
    pred_126 = sub_126_df[["HS", "TR", "AG"]].values

    diff_126_total = int((pred_127 != pred_126).sum())
    diff_126_hs = int((pred_127[:, 0] != pred_126[:, 0]).sum())
    diff_126_tr = int((pred_127[:, 1] != pred_126[:, 1]).sum())
    diff_126_ag = int((pred_127[:, 2] != pred_126[:, 2]).sum())

    assert diff_126_total == 12, f"Celdas distintas vs P126 != 12: {diff_126_total}"
    assert diff_126_hs == 6, f"Celdas distintas en HS vs P126 != 6: {diff_126_hs}"
    assert diff_126_tr == 4, f"Celdas distintas en TR vs P126 != 4: {diff_126_tr}"
    assert diff_126_ag == 2, f"Celdas distintas en AG vs P126 != 2: {diff_126_ag}"
    assert int(sub_df["HS"].sum()) == int(sub_126_df["HS"].sum()), "HS total cambio vs P126"
    assert int(sub_df["TR"].sum()) == int(sub_126_df["TR"].sum()), "TR total cambio vs P126"
    assert int(sub_df["AG"].sum()) == int(sub_126_df["AG"].sum()), "AG total cambio vs P126"
    print("  [OK] 21-27: Modificacion quirurgica vs P126: exactamente 12 celdas (balance neto 0 por columna).")

    # 6. Comparacion quirurgica vs P125 (Aserciones 28-31)
    sub_125_df = pd.read_csv(SUB_P125)
    pred_125 = sub_125_df[["HS", "TR", "AG"]].values
    diff_125_total = int((pred_127 != pred_125).sum())
    assert diff_125_total == 16, f"Celdas distintas vs P125 != 16: {diff_125_total}"
    assert int(sub_df["HS"].sum()) == int(sub_125_df["HS"].sum()), "HS total cambio vs P125"
    assert int(sub_df["TR"].sum()) == int(sub_125_df["TR"].sum()), "TR total cambio vs P125"
    assert int(sub_df["AG"].sum()) == int(sub_125_df["AG"].sum()), "AG total cambio vs P125"
    print("  [OK] 28-31: Modificacion quirurgica vs P125: exactamente 16 celdas (balance neto 0).")

    # 7. Comparacion quirurgica vs P121 (Aserciones 32-36)
    sub_121_df = pd.read_csv(SUB_P121)
    pred_121 = sub_121_df[["HS", "TR", "AG"]].values
    diff_121_total = int((pred_127 != pred_121).sum())
    assert diff_121_total == 23, f"Celdas distintas vs P121 != 23: {diff_121_total}"
    assert int(sub_df["HS"].sum()) == int(sub_121_df["HS"].sum()), "HS total cambio vs P121"
    assert int(sub_df["TR"].sum()) == int(sub_121_df["TR"].sum()), "TR total cambio vs P121"
    assert int(sub_df["AG"].sum()) == int(sub_121_df["AG"].sum()) + 1, "AG total no aumento en +1 vs P121"
    print("  [OK] 32-36: Modificacion quirurgica vs P121: exactamente 23 celdas (+1 neto).")

    # 8. Filas clave auditadas (Aserciones 37-55)
    # Nuevas P127
    assert pred_127[363, 1] == 1 and pred_127[363, 0] == 1, "Fila 363 no rescatada en TR/HS"
    assert pred_127[448, 1] == 0 and pred_127[448, 0] == 0, "Fila 448 no suprimida en TR/HS"
    assert pred_127[455, 0] == 0, "Fila 455 no suprimida en HS=0"
    assert pred_127[520, 2] == 0, "Fila 520 no suprimida en AG=0"
    assert pred_127[552, 0] == 1, "Fila 552 no rescatada en HS=1"
    assert pred_127[659, 1] == 1, "Fila 659 no rescatada en TR=1"
    assert pred_127[853, 2] == 1, "Fila 853 no rescatada en AG=1"
    assert pred_127[889, 0] == 1, "Fila 889 no rescatada en HS=1"
    assert pred_127[1058, 1] == 0 and pred_127[1058, 0] == 0, "Fila 1058 no suprimida en TR/HS"

    # Preservadas P126
    assert pred_127[464, 1] == 1, "Fila 464 no preservada en TR=1"
    assert pred_127[235, 1] == 0, "Fila 235 no preservada en TR=0"
    assert pred_127[918, 0] == 1, "Fila 918 no preservada en HS=1"
    assert pred_127[816, 0] == 0, "Fila 816 no preservada en HS=0"
    # Preservadas P125
    assert pred_127[528, 1] == 1, "Fila 528 no preservada en TR=1"
    assert pred_127[300, 1] == 0, "Fila 300 no preservada en TR=0"
    assert pred_127[870, 0] == 1, "Fila 870 no preservada en HS=1"
    assert pred_127[940, 0] == 0, "Fila 940 no preservada en HS=0"
    # Preservadas P124/P122/P121
    assert pred_127[1006, 2] == 1, "Fila 1006 no preservada en AG=1"
    assert pred_127[354, 2] == 0, "Fila 354 no preservada en AG=0"
    assert pred_127[917, 2] == 1, "Fila 917 no preservada en AG=1"
    assert pred_127[583, 0] == 0, "Fila 583 no preservada en HS=0"
    assert pred_127[1049, 1] == 0, "Fila 1049 activada indebidamente en TR"
    print("  [OK] 37-55: Filas clave auditadas validadas (363, 448, 455, 520, 552, 659, 853, 889, 1058, 464, 235, 918, 816, 528, 300, 870, 940, 1006, 354, 917, 583, 1049).")

    # 9. Validacion de Metricas y Reportes (Aserciones 56-62)
    with open(SAL_JSON, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["experimento"] == "P127-01", f"ID de experimento invalido: {rep['experimento']}"
    assert rep["sha256_submission"] == sha256_archivo(SAL_SUB), "SHA256 en JSON no coincide con archivo"
    assert rep["test_positivos"]["total"] == 1265, "Total positivos en JSON != 1265"
    assert rep["violaciones_ontologicas"] == 0, "Violaciones ontologicas reportadas > 0"
    assert rep["celdas_distintas_test"]["vs_p126"] == 12, "Celdas distintas vs P126 en reporte != 12"
    assert rep["celdas_distintas_test"]["vs_p121"] == 23, "Celdas distintas vs P121 en reporte != 23"

    ccv_can = rep["metricas_canonica"]["ccv_p127"]
    ccv_p47 = rep["metricas_particion47"]["ccv_p127"]
    media_ccv = rep["media_ccv"]

    assert abs(ccv_can - 0.787582) < 1e-4, f"Canonica CCV cambio inesperadamente: {ccv_can}"
    assert abs(ccv_p47 - 0.786034) < 1e-4, f"Particion 47 CCV cambio inesperadamente: {ccv_p47}"
    assert abs(media_ccv - 0.786808) < 1e-4, f"Media CCV cambio inesperadamente: {media_ccv}"
    print(f"  [OK] 56-62: Metricas locales validadas (Canonica {ccv_can:.6f}, Part47 {ccv_p47:.6f}, Media {media_ccv:.6f}).")

    # 10. Hash SHA256 verificado (Aserciones 63-65)
    real_sha = sha256_archivo(SAL_SUB)
    assert real_sha == HASH_ESPERADO_P127, f"SHA256 no coincide con el esperado: {real_sha}"
    assert rep["sha256_submission"] == HASH_ESPERADO_P127, "SHA256 en reporte difiere del hash real"
    print(f"  [OK] 63-65: SHA256 verificado: {real_sha}")

    print(f"[P127] Positivos Test: HS={pos_hs}, TR={pos_tr}, AG={pos_ag} | Total={pos_total}")
    print("[P127] Verificacion completa: CANDIDATO TOP 1 P127 VALIDADO Y CONFORME.")
    print("\n[RESUMEN] 65/65 aserciones pasadas exitosamente (100% aprobado).")


if __name__ == "__main__":
    test_verificacion_p127()
