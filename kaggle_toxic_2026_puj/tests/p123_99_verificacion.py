#!/usr/bin/env python
"""P123-99: Suite de verificacion formal del challenger P123.

Valida rigurosamente la integridad de la submission, reportes, determinismo M2,
anclaje de prevalencia de oro (1.265 positivos), atenuacion ontologica de TR en baja
confianza conjunta, proteccion de la guardia de amenaza, y consistencia con P121, P122 y P114.
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

SCRIPT_EXP = RAIZ / "experiments" / "p123_01_challenger_probabilistic_joint_blend.py"
SAL_SUB = RAIZ / "submissions" / "toxic_submission_p123_probabilistic_joint_blend_asc.csv"
SUB_P122 = RAIZ / "submissions" / "toxic_submission_p122_covariance_ag_rescue_asc.csv"
SUB_P121 = RAIZ / "submissions" / "toxic_submission_p121_asymmetric_joint_blend_asc.csv"
SUB_P114 = RAIZ / "submissions" / "toxic_submission_p114_tri_optimizer_smooth_blend_asc.csv"
SUB_TEMPLATE = RAIZ / "data" / "raw" / "submission.csv"
SAL_JSON = RAIZ / "reports" / "p123_01_challenger_probabilistic_joint_blend.json"
SAL_MD = RAIZ / "reports" / "p123_01_challenger_probabilistic_joint_blend.md"


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def test_verificacion_p123() -> None:
    print("[TEST] Iniciando verificacion exhaustiva de P123-01...")

    # 1. Existencia de archivos
    assert SCRIPT_EXP.is_file(), f"Falta script: {SCRIPT_EXP}"
    assert SAL_SUB.is_file(), f"Falta submission: {SAL_SUB}"
    assert SAL_JSON.is_file(), f"Falta reporte JSON: {SAL_JSON}"
    assert SAL_MD.is_file(), f"Falta reporte Markdown: {SAL_MD}"
    print("  [OK] 1-4: 4/4 archivos requeridos existen en disco.")

    # 2. Integridad de Submission
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

    # 3. Regla Ontologica Ascendente HatEval
    violaciones = ((sub_df["TR"] == 1) & (sub_df["HS"] == 0)).sum()
    assert violaciones == 0, f"Violaciones ontologicas en submission: {violaciones}"
    print("  [OK] 12: 0 violaciones ontologicas (TR=1 => HS=1 estrictamente cumplido).")

    # 4. Anclaje de Prevalencia de Oro en Test Ciego
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

    # 5. Comparacion contra Campeon Reinante P121
    sub_p121 = pd.read_csv(SUB_P121)
    pred_121 = sub_p121[list(ETIQUETAS)].to_numpy(dtype=int)
    pred_123 = sub_df[list(ETIQUETAS)].to_numpy(dtype=int)

    diff_mask_121 = pred_121 != pred_123
    diff_cells_121 = int(diff_mask_121.sum())
    assert diff_cells_121 == 1, f"Esperada exactamente 1 celda modificada vs P121, obtenidas {diff_cells_121}"

    # Exactamente 0 cambios en HS y TR vs P121
    assert (pred_121[:, 0] == pred_123[:, 0]).all(), "HS difiere de P121"
    assert (pred_121[:, 1] == pred_123[:, 1]).all(), "TR difiere de P121"

    cambios_01_121 = int(((pred_121 == 0) & (pred_123 == 1)).sum())
    cambios_10_121 = int(((pred_121 == 1) & (pred_123 == 0)).sum())
    assert cambios_01_121 == 1, f"Adiciones esperadas vs P121: 1, obtenidas: {cambios_01_121}"
    assert cambios_10_121 == 0, f"Supresiones esperadas vs P121: 0, obtenidas: {cambios_10_121}"
    assert cambios_01_121 - cambios_10_121 == 1, "Balance neto vs P121 debe ser +1"
    print("  [OK] 21-26: Modificacion quirurgica vs P121: exactamente 1 celda modificada (+1 en AG fila 917).")

    # 6. Comparacion contra P122
    sub_p122 = pd.read_csv(SUB_P122)
    pred_122 = sub_p122[list(ETIQUETAS)].to_numpy(dtype=int)

    diff_mask_122 = pred_122 != pred_123
    diff_cells_122 = int(diff_mask_122.sum())
    assert diff_cells_122 == 0, f"Esperadas exactamente 0 celdas modificadas vs P122 en test, obtenidas {diff_cells_122}"
    assert (pred_122[:, 0] == pred_123[:, 0]).all(), "HS difiere de P122"
    assert (pred_122[:, 1] == pred_123[:, 1]).all(), "TR difiere de P122"
    assert (pred_122[:, 2] == pred_123[:, 2]).all(), "AG difiere de P122"
    print("  [OK] 27-31: Identidad bit a bit con P122 en test (proteccion de guardia lexica sobre fila 345).")

    # 7. Comparacion contra Ancla P114
    sub_p114 = pd.read_csv(SUB_P114)
    pred_114 = sub_p114[list(ETIQUETAS)].to_numpy(dtype=int)

    diff_mask_114 = pred_114 != pred_123
    diff_cells_114 = int(diff_mask_114.sum())
    assert diff_cells_114 == 2, f"Esperadas exactamente 2 celdas modificadas vs P114, obtenidas {diff_cells_114}"

    cambios_01_114 = int(((pred_114 == 0) & (pred_123 == 1)).sum())
    cambios_10_114 = int(((pred_114 == 1) & (pred_123 == 0)).sum())
    assert cambios_01_114 == 1, f"Adiciones esperadas vs P114: 1, obtenidas: {cambios_01_114}"
    assert cambios_10_114 == 1, f"Supresiones esperadas vs P114: 1, obtenidas: {cambios_10_114}"
    assert cambios_01_114 - cambios_10_114 == 0, "Balance neto vs P114 debe ser 0"
    print("  [OK] 32-36: Perturbacion quirurgica vs P114: exactamente 2 celdas (fila 583 HS: 1->0, fila 917 AG: 0->1).")

    # 8. Filas Clave Auditadas en Test
    # Fila 583 (id 24341): P114=[1,0,0], P121=[0,0,0], P122=[0,0,0], P123=[0,0,0]
    assert (pred_114[583] == np.array([1, 0, 0])).all(), f"Fila 583 en P114 no es [1,0,0]: {pred_114[583]}"
    assert (pred_123[583] == np.array([0, 0, 0])).all(), f"Fila 583 en P123 no es [0,0,0]: {pred_123[583]}"

    # Fila 917 (id 24539): P121=[1,1,0], P122=[1,1,1], P123=[1,1,1]
    assert (pred_121[917] == np.array([1, 1, 0])).all(), f"Fila 917 en P121 no es [1,1,0]: {pred_121[917]}"
    assert (pred_123[917] == np.array([1, 1, 1])).all(), f"Fila 917 en P123 no es [1,1,1]: {pred_123[917]}"

    # Fila 300 (id 20051): Mascota P123=[1,1,0] (AG permanece en 0, no rescatada por P(TR)=0.5378 < 0.58)
    assert (pred_123[300] == np.array([1, 1, 0])).all(), f"Fila 300 en P123 no es [1,1,0]: {pred_123[300]}"
    assert pred_123[300, 2] == 0, f"Fila 300 AG no debe ser 1: {pred_123[300, 2]}"

    # Fila 1049 (id 24874): Gaming TR=0
    assert (pred_123[1049] == np.array([1, 0, 1])).all(), f"Fila 1049 en P123 no es [1,0,1]: {pred_123[1049]}"

    # Fila 345 (id 23782): Amenaza de violencia explicita ("Tiro"), TR protegido en 1 por la guardia, HS=1 por regla ontologica
    assert (pred_123[345] == np.array([1, 1, 0])).all(), f"Fila 345 en P123 no es [1,1,0]: {pred_123[345]}"
    assert pred_123[345, 1] == 1, f"Fila 345 TR debe estar protegido en 1: {pred_123[345, 1]}"
    assert pred_123[345, 0] == 1, f"Fila 345 HS debe ser 1 por regla ontologica: {pred_123[345, 0]}"
    print("  [OK] 37-44: Filas clave auditadas validadas (583, 917, 300, 1049, 345).")

    # 9. Validacion del Reporte JSON y Determinismo M2
    with open(SAL_JSON, encoding="utf-8") as fh:
        rep = json.load(fh)

    assert rep["experimento"] == "P123-01"
    assert rep["features_total"] == 86275, f"Features esperadas 86275, reportadas {rep['features_total']}"

    # Controles M2
    diff_v03 = abs(rep["canonica"]["ccv_v03"] - 0.7772114525165721)
    assert diff_v03 < 1e-12, f"Control V03 alterado: {diff_v03}"

    diff_p114_can = abs(rep["canonica"]["ccv_p114"] - 0.787217983050478)
    assert diff_p114_can < 1e-6, f"Control P114 Canonica alterado: {diff_p114_can}"

    diff_p114_p47 = abs(rep["particion_47"]["ccv_p114"] - 0.7856734188734289)
    assert diff_p114_p47 < 1e-6, f"Control P114 Part47 alterado: {diff_p114_p47}"

    diff_p121_can = abs(rep["canonica"]["ccv_p121"] - 0.7875816993756768)
    assert diff_p121_can < 1e-6, f"Control P121 Canonica alterado: {diff_p121_can}"

    diff_p121_p47 = abs(rep["particion_47"]["ccv_p121"] - 0.7856733330466952)
    assert diff_p121_p47 < 1e-6, f"Control P121 Part47 alterado: {diff_p121_p47}"

    diff_p122_p47 = abs(rep["particion_47"]["ccv_p122"] - 0.7860341774026366)
    assert diff_p122_p47 < 1e-6, f"Control P122 Part47 alterado: {diff_p122_p47}"

    # Validacion de cotas exactas de P123
    can_ccv = rep["canonica"]["ccv_p123"]
    assert abs(can_ccv - 0.788584) < 1e-5, f"Canonica CCV difiere: {can_ccv}"

    p47_ccv = rep["particion_47"]["ccv_p123"]
    assert abs(p47_ccv - 0.786034) < 1e-5, f"Particion 47 CCV difiere: {p47_ccv}"

    med_ccv = rep["media_ccv"]["p123"]
    assert abs(med_ccv - 0.787309) < 1e-5, f"Media CCV difiere: {med_ccv}"

    delta_can_121 = rep["canonica"]["delta_vs_p121"]
    assert delta_can_121 > 0, f"Canonica debe superar a P121: {delta_can_121}"
    assert abs(delta_can_121 - 0.001002) < 1e-5, f"Delta Canonica vs P121 inesperado: {delta_can_121}"

    delta_p47_121 = rep["particion_47"]["delta_vs_p121"]
    assert delta_p47_121 > 0, f"Particion 47 debe superar a P121: {delta_p47_121}"
    assert abs(delta_p47_121 - 0.000361) < 1e-5, f"Delta Part47 vs P121 inesperado: {delta_p47_121}"

    delta_med_121 = rep["media_ccv"]["delta_vs_p121"]
    assert delta_med_121 > 0, f"Media CCV debe superar a P121: {delta_med_121}"
    assert abs(delta_med_121 - 0.000682) < 1e-5, f"Delta Media vs P121 inesperado: {delta_med_121}"

    delta_med_122 = rep["media_ccv"]["delta_vs_p122"]
    assert delta_med_122 > 0, f"Media CCV debe superar a P122: {delta_med_122}"
    assert abs(delta_med_122 - 0.000501) < 1e-5, f"Delta Media vs P122 inesperado: {delta_med_122}"

    delta_med_114 = rep["media_ccv"]["delta_vs_p114"]
    assert delta_med_114 > 0, f"Media CCV debe superar a P114: {delta_med_114}"
    assert abs(delta_med_114 - 0.000864) < 1e-5, f"Delta Media vs P114 inesperado: {delta_med_114}"
    print(f"  [OK] 45-60: Gating y determinismo M2 superados (Canonica {can_ccv:.6f}, Part47 {p47_ccv:.6f}, Media {med_ccv:.6f}).")

    # 10. Hash SHA256 verificado
    hash_sub = sha256_archivo(SAL_SUB)
    assert hash_sub == rep["test_eval"]["sha256"], "Hash de submission no coincide con el reporte"
    assert hash_sub == "daa97f5d9b68b26003083ba5588aefa99e433e77e30fac85dc26ab3c07916048", f"Hash inesperado: {hash_sub}"
    print(f"  [OK] 61-62: SHA256 verificado: {hash_sub}")

    print("\n[RESUMEN] 62/62 aserciones pasadas exitosamente. P123-01 validado al 100%.")


if __name__ == "__main__":
    test_verificacion_p123()
