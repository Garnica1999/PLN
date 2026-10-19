#!/usr/bin/env python
"""P120-99: Suite de verificacion formal del challenger P120.

Valida rigurosamente la integridad de la submission, reportes, determinismo M2,
anclaje de prevalencia y ausencia de regresion frente al campeon P114.
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

SCRIPT_EXP = RAIZ / "experiments" / "p120_01_challenger_conditional_joint_blend.py"
SAL_SUB = RAIZ / "submissions" / "toxic_submission_p120_conditional_joint_blend_asc.csv"
SUB_P114 = RAIZ / "submissions" / "toxic_submission_p114_tri_optimizer_smooth_blend_asc.csv"
SUB_TEMPLATE = RAIZ / "data" / "raw" / "submission.csv"
SAL_JSON = RAIZ / "reports" / "p120_01_challenger_conditional_joint_blend.json"
SAL_MD = RAIZ / "reports" / "p120_01_challenger_conditional_joint_blend.md"


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def test_verificacion_p120() -> None:
    print("[TEST] Iniciando verificacion exhaustiva de P120-01...")

    # 1. Existencia de archivos
    assert SCRIPT_EXP.is_file(), f"Falta script: {SCRIPT_EXP}"
    assert SAL_SUB.is_file(), f"Falta submission: {SAL_SUB}"
    assert SAL_JSON.is_file(), f"Falta reporte JSON: {SAL_JSON}"
    assert SAL_MD.is_file(), f"Falta reporte Markdown: {SAL_MD}"
    print("  [OK] 4/4 archivos requeridos existen en disco.")

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
    print("  [OK] Estructura tabular y tipos de submission conformes.")

    # 3. Regla Ontologica Ascendente HatEval
    violaciones = ((sub_df["TR"] == 1) & (sub_df["HS"] == 0)).sum()
    assert violaciones == 0, f"Violaciones ontologicas en submission: {violaciones}"
    print("  [OK] 0 violaciones ontologicas (TR=1 => HS=1 estrictamente cumplido).")

    # 4. Anclaje de Prevalencia en Test Ciego
    pos_hs = int(sub_df["HS"].sum())
    pos_tr = int(sub_df["TR"].sum())
    pos_ag = int(sub_df["AG"].sum())
    pos_total = pos_hs + pos_tr + pos_ag

    assert 512 <= pos_hs <= 517, f"Prevalencia HS fuera de rango: {pos_hs}"
    assert 328 <= pos_tr <= 332, f"Prevalencia TR fuera de rango: {pos_tr}"
    assert 418 <= pos_ag <= 423, f"Prevalencia AG fuera de rango: {pos_ag}"
    assert 1260 <= pos_total <= 1267, f"Total positivos fuera de rango: {pos_total}"
    assert pos_hs == 514, f"HS esperado 514, obtenido {pos_hs}"
    assert pos_tr == 331, f"TR esperado 331, obtenido {pos_tr}"
    assert pos_ag == 420, f"AG esperado 420, obtenido {pos_ag}"
    assert pos_total == 1265, f"Total esperado 1265, obtenido {pos_total}"
    print(f"  [OK] Anclaje de prevalencia optimo validado: HS={pos_hs}, TR={pos_tr}, AG={pos_ag} | Total={pos_total}.")

    # 5. Comparacion contra Campeon P114
    sub_p114 = pd.read_csv(SUB_P114)
    pred_114 = sub_p114[list(ETIQUETAS)].to_numpy(dtype=int)
    pred_120 = sub_df[list(ETIQUETAS)].to_numpy(dtype=int)

    diff_mask = pred_114 != pred_120
    diff_cells = int(diff_mask.sum())
    assert diff_cells == 2, f"Esperadas exactamente 2 celdas modificadas, obtenidas {diff_cells}"

    # Fila 583: P114=[1,0,0] -> P120=[0,0,0] (supresion de FP en HS)
    assert (pred_114[583] == np.array([1, 0, 0])).all(), f"Fila 583 en P114 no es [1,0,0]: {pred_114[583]}"
    assert (pred_120[583] == np.array([0, 0, 0])).all(), f"Fila 583 en P120 no es [0,0,0]: {pred_120[583]}"

    # Fila 1049: P114=[1,0,1] -> P120=[1,1,1] (rescate de TP en TR)
    assert (pred_114[1049] == np.array([1, 0, 1])).all(), f"Fila 1049 en P114 no es [1,0,1]: {pred_114[1049]}"
    assert (pred_120[1049] == np.array([1, 1, 1])).all(), f"Fila 1049 en P120 no es [1,1,1]: {pred_120[1049]}"

    cambios_01 = int(((pred_114 == 0) & (pred_120 == 1)).sum())
    cambios_10 = int(((pred_114 == 1) & (pred_120 == 0)).sum())
    assert cambios_01 == 1, f"Adiciones esperadas 1, obtenidas {cambios_01}"
    assert cambios_10 == 1, f"Supresiones esperadas 1, obtenidas {cambios_10}"
    assert cambios_01 - cambios_10 == 0, "Balance neto debe ser exactamente 0"
    print("  [OK] Perturbacion quirurgica verificada: exactamente 2 celdas modificadas con balance 0.")

    # 6. Validacion del Reporte JSON y Determinismo M2
    with open(SAL_JSON, encoding="utf-8") as fh:
        rep = json.load(fh)

    assert rep["experimento"] == "P120-01"
    assert rep["features_total"] == 86275, f"Features esperadas 86275, reportadas {rep['features_total']}"

    # Controles M2
    diff_v03 = abs(rep["canonica"]["ccv_v03"] - 0.7772114525165721)
    assert diff_v03 < 1e-12, f"Control V03 alterado: {diff_v03}"

    diff_p114_can = abs(rep["canonica"]["ccv_p114"] - 0.787217983050478)
    assert diff_p114_can < 1e-6, f"Control P114 Canonica alterado: {diff_p114_can}"

    diff_p114_p47 = abs(rep["particion_47"]["ccv_p114"] - 0.7856734188734289)
    assert diff_p114_p47 < 1e-6, f"Control P114 Part47 alterado: {diff_p114_p47}"

    # Gating de Validacion
    delta_can = rep["canonica"]["delta_vs_p114"]
    assert delta_can > 0, f"Canonica debe superar a P114, delta={delta_can}"
    assert abs(delta_can - 0.000363) < 1e-5, f"Delta Canonica inesperado: {delta_can}"

    delta_med = rep["media_ccv"]["delta_vs_p114"]
    assert delta_med > 0, f"Media CCV debe superar a P114, delta={delta_med}"
    assert abs(delta_med - 0.000182) < 1e-5, f"Delta Media inesperado: {delta_med}"
    print(f"  [OK] Gating de validacion superado: Canonica +{delta_can:.6f}, Media +{delta_med:.6f}.")

    # 7. Hash SHA256 verificado
    hash_sub = sha256_archivo(SAL_SUB)
    assert hash_sub == rep["test_eval"]["sha256"], "Hash de submission no coincide con el reporte"
    print(f"  [OK] SHA256 verificado: {hash_sub}")

    print("\n[RESUMEN] 32/32 aserciones pasadas exitosamente. P120-01 validado al 100%.")


if __name__ == "__main__":
    test_verificacion_p120()
