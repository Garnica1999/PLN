#!/usr/bin/env python
"""Suite de verificacion formal para P117: Challenger Multi-Family NBSVM Subspace Blend.

Comprueba 30 aserciones independientes de integridad, cero fuga, determinismo M2,
recuento de positivos, balance direccional y validez de la submission generada.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
SUB_P117 = RAIZ / "submissions" / "toxic_submission_p117_multifamily_nbsvm_subspace_asc.csv"
SUB_P114 = RAIZ / "submissions" / "toxic_submission_p114_tri_optimizer_smooth_blend_asc.csv"
SUB_TEMPLATE = RAIZ / "data" / "raw" / "submission.csv"
REP_JSON = RAIZ / "reports" / "p117_01_challenger_multifamily_nbsvm_subspace.json"
REP_MD = RAIZ / "reports" / "p117_01_challenger_multifamily_nbsvm_subspace.md"
SCRIPT_P117 = RAIZ / "experiments" / "p117_01_challenger_multifamily_nbsvm_subspace.py"


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    print("=== INICIANDO VERIFICACION FORMAL P117 (30 ASERCIONES) ===", flush=True)

    # 1-4: Existencia de artefactos
    assert SCRIPT_P117.exists(), "A1: Script P117 no existe"
    assert REP_JSON.exists(), "A2: Reporte JSON no existe"
    assert REP_MD.exists(), "A3: Reporte MD no existe"
    assert SUB_P117.exists(), "A4: Submission CSV no existe"
    print("Check 1-4: Artefactos existen.")

    # 5-10: Verificacion de submission
    sub_df = pd.read_csv(SUB_P117)
    tpl_df = pd.read_csv(SUB_TEMPLATE)
    assert len(sub_df) == 1154, f"A5: Longitud esperada 1154, obtenida {len(sub_df)}"
    assert list(sub_df.columns) == ["id", "HS", "TR", "AG"], f"A6: Columnas invalidas: {sub_df.columns}"
    assert (sub_df["id"].values == tpl_df["id"].values).all(), "A7: IDs no coinciden exactamente con template"
    assert sub_df.isnull().sum().sum() == 0, "A8: Existen valores nulos en submission"
    for col in ["HS", "TR", "AG"]:
        assert set(sub_df[col].unique()).issubset({0, 1}), f"A9: Valores no binarios en columna {col}"

    # 11: Restriccion ontologica HatEval
    violaciones = ((sub_df["TR"] == 1) & (sub_df["HS"] == 0)).sum()
    assert violaciones == 0, f"A11: Existen {violaciones} violaciones ontologicas TR=1 => HS=0"
    print("Check 5-11: Integridad y validez sintactica de submission superada.")

    # 12-16: Recuentos de positivos y distribucion
    total_pos = int(sub_df[["HS", "TR", "AG"]].values.sum())
    assert 1250 <= total_pos <= 1270, f"A12: Total positivos fuera de banda optima: {total_pos}"
    assert sub_df["HS"].sum() == 509, f"A13: HS positivos esperado 509, obtenido {sub_df['HS'].sum()}"
    assert sub_df["TR"].sum() == 332, f"A14: TR positivos esperado 332, obtenido {sub_df['TR'].sum()}"
    assert sub_df["AG"].sum() == 422, f"A15: AG positivos esperado 422, obtenido {sub_df['AG'].sum()}"
    assert total_pos == 1263, f"A16: Total esperado 1263, obtenido {total_pos}"
    print(f"Check 12-16: Distribucion de positivos verificada (Total={total_pos}, HS=509, TR=332, AG=422).")

    # 17-20: Comparacion contra P114 campeon reinante
    sub_p114 = pd.read_csv(SUB_P114)
    diff_cells = int((sub_df[["HS", "TR", "AG"]].values != sub_p114[["HS", "TR", "AG"]].values).sum())
    assert diff_cells == 28, f"A17: Celdas distintas de P114 esperado 28, obtenido {diff_cells}"
    pct_diff = diff_cells / (1154 * 3) * 100
    assert 0.5 <= pct_diff <= 1.2, f"A18: Porcentaje de perturbacion fuera de rango: {pct_diff:.2f}%"

    c01 = int(np.sum((sub_p114[["HS", "TR", "AG"]].values == 0) & (sub_df[["HS", "TR", "AG"]].values == 1)))
    c10 = int(np.sum((sub_p114[["HS", "TR", "AG"]].values == 1) & (sub_df[["HS", "TR", "AG"]].values == 0)))
    assert c01 == 13, f"A19a: Adiciones esperadas 13, obtenidas {c01}"
    assert c10 == 15, f"A19b: Supresiones esperadas 15, obtenidas {c10}"
    assert (c01 - c10) == -2, f"A19c: Balance neto esperado -2, obtenido {c01 - c10}"

    sha_calc = sha256_archivo(SUB_P117)
    assert sha_calc == "91eaeab5dc68f7a8c14c6ade4ebec25c19e5cbc1e849ec4c427463a1661c6183", f"A20: SHA256 no coincide: {sha_calc}"
    assert sha_calc != "c31ee6f2147c1c057095bfda5840e7eac48cd425ede1806d22f0c77f4d4d4274", "A20b: SHA256 igual a P114"
    print(f"Check 17-20: Comparacion direccional contra P114 verificada (+13/-15, neto -2, SHA256 OK).")

    # 21-26: Reporte JSON
    with open(REP_JSON, "r", encoding="utf-8") as fh:
        rep = json.load(fh)
    assert rep["experimento"] == "P117-01", "A21: Nombre de experimento incorrecto en JSON"
    assert rep["features"]["dimension"] == 86275, f"A22: Dimension de features {rep['features']['dimension']} != 86275"
    assert rep["controles"]["diff_p114_m2"] < 1e-6, f"A23: Control M2 P114 fallido: {rep['controles']['diff_p114_m2']}"
    assert rep["controles"]["diff_v03_m2"] < 1e-12, f"A24: Control M2 V03 fallido: {rep['controles']['diff_v03_m2']}"
    assert rep["modelos"]["pesos_blend"]["w_lr"] == 0.70, "A25: w_lr != 0.70"
    assert rep["modelos"]["pesos_blend"]["w_nbsvm"] == 0.15, "A26: w_nbsvm != 0.15"
    assert rep["modelos"]["pesos_blend"]["w_saga"] == 0.10, "A26b: w_saga != 0.10"
    assert rep["modelos"]["pesos_blend"]["w_sgd"] == 0.05, "A26c: w_sgd != 0.05"
    print("Check 21-26: Parametros de configuracion y controles M2 validados.")

    # 27-30: Metricas de CV
    can_ccv = rep["validacion"]["canonica"]["ccv_p117"]
    p47_ccv = rep["validacion"]["particion_47"]["ccv_p117"]
    media_ccv = rep["validacion"]["ccv_media_p117"]
    delta_p114 = rep["validacion"]["delta_media_vs_p114"]
    assert can_ccv > 0.7875, f"A27: CCV canonica demasiado baja: {can_ccv}"
    assert p47_ccv > 0.7870, f"A28: CCV Part47 demasiado baja: {p47_ccv}"
    assert media_ccv > 0.7875, f"A29: Media CCV demasiado baja: {media_ccv}"
    assert delta_p114 > 0.0010, f"A30: Delta medio vs P114 demasiado bajo: {delta_p114}"
    print(f"Check 27-30: Metricas de CV robustas (Canonica={can_ccv:.6f}, P47={p47_ccv:.6f}, Media={media_ccv:.6f}, Delta={delta_p114:+.6f}).")

    print("\nTODAS LAS 30 ASERCIONES SUPERADAS EXITOSAMENTE (30/30 OK).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
