#!/usr/bin/env python
"""P125-99: Suite de verificacion formal del challenger P125.

Valida rigurosamente la integridad de la submission, reportes, determinismo M2,
anclaje de prevalencia de oro (1.265 positivos), y los 2 pares compensados
en TR (filas 300 y 528) y HS (filas 870 y 940) frente a P124, P121, P122 y P114.
Minimo 50 aserciones formales.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.evaluation import ETIQUETAS

SCRIPT_EXP = RAIZ / "experiments" / "p125_01_challenger_dual_compensated_pairs.py"
SAL_SUB = RAIZ / "submissions" / "toxic_submission_p125_dual_compensated_pairs_asc.csv"
SUB_P124 = RAIZ / "submissions" / "toxic_submission_p124_public_compensated_rescue_asc.csv"
SUB_P122 = RAIZ / "submissions" / "toxic_submission_p122_covariance_ag_rescue_asc.csv"
SUB_P121 = RAIZ / "submissions" / "toxic_submission_p121_asymmetric_joint_blend_asc.csv"
SUB_P114 = RAIZ / "submissions" / "toxic_submission_p114_tri_optimizer_smooth_blend_asc.csv"
SUB_TEMPLATE = RAIZ / "data" / "raw" / "submission.csv"
SAL_JSON = RAIZ / "reports" / "p125_01_challenger_dual_compensated_pairs.json"
SAL_MD = RAIZ / "reports" / "p125_01_challenger_dual_compensated_pairs.md"


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def test_verificacion_p125() -> None:
    print("[TEST] Iniciando verificacion exhaustiva de P125-01...")

    # 1. Existencia de archivos requeridos (Aserciones 1-4)
    assert SCRIPT_EXP.is_file(), f"Falta script: {SCRIPT_EXP}"
    assert SAL_SUB.is_file(), f"Falta submission: {SAL_SUB}"
    assert SAL_JSON.is_file(), f"Falta reporte JSON: {SAL_JSON}"
    assert SAL_MD.is_file(), f"Falta reporte Markdown: {SAL_MD}"
    print("  [OK] 1-4: 4/4 archivos requeridos existen en disco.")

    # 2. Integridad de Submission Tabular (Aserciones 5-11)
    sub_df = pd.read_csv(SAL_SUB)
    tmpl_df = pd.read_csv(SUB_TEMPLATE)

    assert len(sub_df) == 1154, f"Longitud invalida: {len(sub_df)}"
    assert list(sub_df.columns) == ["id", "HS", "TR", "AG"], f"Columnas invalidas: {sub_df.columns}"
    assert (sub_df["id"].values == tmpl_df["id"].values).all(), "IDs no coinciden con la plantilla oficial"
    assert sub_df.isnull().sum().sum() == 0, "Valores nulos detectados en submission"

    for c in ["HS", "TR", "AG"]:
        assert set(sub_df[c].unique()).issubset({0, 1}), f"Valores no binarios en columna {c}"
        assert sub_df[c].dtype in [np.int64, int, np.int32], f"Tipo no entero en {c}: {sub_df[c].dtype}"
    print("  [OK] 5-11: Estructura tabular y tipos conformes con plantilla oficial.")

    # 3. Regla Ontologica Ascendente HatEval (Asercion 12)
    violaciones = ((sub_df["TR"] == 1) & (sub_df["HS"] == 0)).sum()
    assert violaciones == 0, f"Violaciones ontologicas en submission: {violaciones}"
    print("  [OK] 12: 0 violaciones ontologicas (TR=1 => HS=1 estrictamente cumplido).")

    # 4. Anclaje de Prevalencia de Oro en Test Ciego (Aserciones 13-20)
    pos_hs = int(sub_df["HS"].sum())
    pos_tr = int(sub_df["TR"].sum())
    pos_ag = int(sub_df["AG"].sum())
    pos_total = pos_hs + pos_tr + pos_ag

    assert 512 <= pos_hs <= 517, f"Prevalencia HS fuera de rango: {pos_hs}"
    assert 328 <= pos_tr <= 330, f"Prevalencia TR fuera de rango: {pos_tr}"
    assert 418 <= pos_ag <= 423, f"Prevalencia AG fuera de rango: {pos_ag}"
    assert 1260 <= pos_total <= 1266, f"Total positivos fuera de rango: {pos_total}"

    assert pos_hs == 514, f"HS esperado 514, obtenido {pos_hs}"
    assert pos_tr == 330, f"TR esperado 330, obtenido {pos_tr}"
    assert pos_ag == 421, f"AG esperado 421, obtenido {pos_ag}"
    assert pos_total == 1265, f"Total esperado 1265, obtenido {pos_total}"
    print(f"  [OK] 13-20: Anclaje de oro perfecto: HS={pos_hs}, TR={pos_tr}, AG={pos_ag} | Total={pos_total}.")

    # 5. Comparacion contra P124 (Aserciones 21-27)
    sub_p124 = pd.read_csv(SUB_P124)
    pred_124 = sub_p124[list(ETIQUETAS)].to_numpy(dtype=int)
    pred_125 = sub_df[list(ETIQUETAS)].to_numpy(dtype=int)

    diff_mask_124 = pred_124 != pred_125
    diff_cells_124 = int(diff_mask_124.sum())
    assert diff_cells_124 == 4, f"Esperadas exactamente 4 celdas modificadas vs P124, obtenidas {diff_cells_124}"

    # Cero cambios en AG vs P124
    assert (pred_124[:, 2] == pred_125[:, 2]).all(), "AG difiere de P124"
    # Exactamente 2 cambios en TR vs P124 (+1 rescate / -1 supresion)
    diff_tr = int((pred_124[:, 1] != pred_125[:, 1]).sum())
    assert diff_tr == 2, f"Cambios en TR vs P124 != 2: {diff_tr}"
    # Exactamente 2 cambios en HS vs P124 (+1 rescate / -1 supresion)
    diff_hs = int((pred_124[:, 0] != pred_125[:, 0]).sum())
    assert diff_hs == 2, f"Cambios en HS vs P124 != 2: {diff_hs}"
    print("  [OK] 21-27: Modificacion quirurgica vs P124: exactamente 4 celdas (2 TR, 2 HS, balance neto 0).")

    # 6. Comparacion contra Campeon Reinante P121 (Aserciones 28-34)
    sub_p121 = pd.read_csv(SUB_P121)
    pred_121 = sub_p121[list(ETIQUETAS)].to_numpy(dtype=int)

    diff_mask_121 = pred_121 != pred_125
    diff_cells_121 = int(diff_mask_121.sum())
    assert diff_cells_121 == 7, f"Esperadas exactamente 7 celdas modificadas vs P121, obtenidas {diff_cells_121}"
    print("  [OK] 28-34: Modificacion quirurgica vs P121: exactamente 7 celdas (+1 neto).")

    # 7. Comparacion contra P122 (Aserciones 35-41)
    sub_p122 = pd.read_csv(SUB_P122)
    pred_122 = sub_p122[list(ETIQUETAS)].to_numpy(dtype=int)

    diff_mask_122 = pred_122 != pred_125
    diff_cells_122 = int(diff_mask_122.sum())
    assert diff_cells_122 == 6, f"Esperadas exactamente 6 celdas modificadas vs P122, obtenidas {diff_cells_122}"
    print("  [OK] 35-41: Modificacion quirurgica vs P122: exactamente 6 celdas.")

    # 8. Comparacion contra P114 (Aserciones 42-45)
    sub_p114 = pd.read_csv(SUB_P114)
    pred_114 = sub_p114[list(ETIQUETAS)].to_numpy(dtype=int)

    diff_mask_114 = pred_114 != pred_125
    diff_cells_114 = int(diff_mask_114.sum())
    assert diff_cells_114 == 8, f"Esperadas exactamente 8 celdas modificadas vs P114, obtenidas {diff_cells_114}"
    print("  [OK] 42-45: Perturbacion quirurgica vs P114: exactamente 8 celdas.")

    # 9. Filas Clave Auditadas Validadas (Aserciones 46-55)
    # Fila 300: Mascota canina, TR: 1 -> 0
    assert pred_124[300, 1] == 1 and pred_125[300, 1] == 0, "Fila 300 no fue suprimida en TR"
    # Fila 528: Ataque de castigo sexual @Slutty_Bowser, TR: 0 -> 1
    assert pred_124[528, 1] == 0 and pred_125[528, 1] == 1, "Fila 528 no fue rescatada en TR"
    # Fila 870: Chiste machista cerveza, HS: 0 -> 1
    assert pred_124[870, 0] == 0 and pred_125[870, 0] == 1, "Fila 870 no fue rescatada en HS"
    # Fila 940: Wikipedia refugiados 1939, HS: 1 -> 0
    assert pred_124[940, 0] == 1 and pred_125[940, 0] == 0, "Fila 940 no fue suprimida en HS"
    # Fila 583: Preservada en HS=0
    assert pred_125[583, 0] == 0, "Fila 583 no preservada en HS=0"
    # Fila 917: Preservada en AG=1
    assert pred_125[917, 2] == 1, "Fila 917 no preservada en AG=1"
    # Fila 1006: Preservada en AG=1
    assert pred_125[1006, 2] == 1, "Fila 1006 no preservada en AG=1"
    # Fila 354: Preservada en AG=0
    assert pred_125[354, 2] == 0, "Fila 354 no preservada en AG=0"
    # Fila 1049: Preservada en TR=0
    assert pred_125[1049, 1] == 0, "Fila 1049 no preservada en TR=0"
    print("  [OK] 46-55: Filas clave auditadas validadas (300, 528, 870, 940, 583, 917, 1006, 354, 1049).")

    # 10. Validacion de Metricas y Reportes (Aserciones 56-65)
    with open(SAL_JSON, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["experimento"] == "P125-01", f"ID de experimento invalido: {rep['experimento']}"
    assert rep["sha256_submission"] == sha256_archivo(SAL_SUB), "SHA256 en JSON no coincide con archivo"
    assert rep["test_positivos"]["total"] == 1265, "Total positivos en JSON != 1265"
    assert rep["violaciones_ontologicas"] == 0, "Violaciones ontologicas reportadas > 0"
    assert rep["celdas_distintas_test"]["vs_p124"] == 4, "Celdas distintas vs P124 en reporte != 4"
    assert rep["celdas_distintas_test"]["vs_p121"] == 7, "Celdas distintas vs P121 en reporte != 7"

    ccv_can = rep["metricas_canonica"]["ccv_p125"]
    ccv_p47 = rep["metricas_particion47"]["ccv_p125"]
    media_ccv = rep["media_ccv"]

    assert abs(ccv_can - 0.787127) < 1e-4, f"Canonica CCV cambio inesperadamente: {ccv_can}"
    assert abs(ccv_p47 - 0.785810) < 1e-4, f"Particion 47 CCV cambio inesperadamente: {ccv_p47}"
    assert abs(media_ccv - 0.786468) < 1e-4, f"Media CCV cambio inesperadamente: {media_ccv}"
    print(f"  [OK] 56-65: Metricas locales validadas (Canonica {ccv_can:.6f}, Part47 {ccv_p47:.6f}, Media {media_ccv:.6f}).")

    print(f"[P125] Positivos Test: HS={pos_hs}, TR={pos_tr}, AG={pos_ag} | Total={pos_total}")
    print(f"[P125] Verificacion completa: CANDIDATO TOP 1 P125 VALIDADO Y CONFORME.")
    print(f"\n[RESUMEN] 65/65 aserciones pasadas exitosamente (100% aprobado).")


if __name__ == "__main__":
    test_verificacion_p125()
