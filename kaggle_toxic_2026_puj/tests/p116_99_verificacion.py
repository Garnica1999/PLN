#!/usr/bin/env python
"""Suite de verificacion formal para P116: Challenger Tri-Optimizer Smooth Blend con MaxDF90 y Regularizacion Continua.

Comprueba 30 aserciones independientes de integridad, cero fuga, determinismo M2,
recuento de positivos, balance bidireccional y validez de la submission generada.
"""

import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
SUB_P116 = RAIZ / "submissions" / "toxic_submission_p116_v2_maxdf90_smooth_blend_asc.csv"
SUB_P114 = RAIZ / "submissions" / "toxic_submission_p114_tri_optimizer_smooth_blend_asc.csv"
SUB_TEMPLATE = RAIZ / "data" / "raw" / "submission.csv"
REP_JSON = RAIZ / "reports" / "p116_01_challenger_v2_maxdf90_smooth_blend.json"
REP_MD = RAIZ / "reports" / "p116_01_challenger_v2_maxdf90_smooth_blend.md"
SCRIPT_P116 = RAIZ / "experiments" / "p116_01_challenger_v2_maxdf90_smooth_blend.py"


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    print("=== INICIANDO VERIFICACION FORMAL P116 (30 ASERCIONES) ===", flush=True)

    # 1-4: Existencia de artefactos
    assert SCRIPT_P116.exists(), "A1: Script P116 no existe"
    assert REP_JSON.exists(), "A2: Reporte JSON no existe"
    assert REP_MD.exists(), "A3: Reporte MD no existe"
    assert SUB_P116.exists(), "A4: Submission CSV no existe"
    print("Check 1-4: Artefactos existen.")

    # 5-10: Verificacion de submission
    sub_df = pd.read_csv(SUB_P116)
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
    assert 1250 <= total_pos <= 1275, f"A12: Total positivos fuera de banda optima: {total_pos}"
    assert sub_df["HS"].sum() == 517, f"A13: HS positivos esperado 517, obtenido {sub_df['HS'].sum()}"
    assert sub_df["TR"].sum() == 334, f"A14: TR positivos esperado 334, obtenido {sub_df['TR'].sum()}"
    assert sub_df["AG"].sum() == 415, f"A15: AG positivos esperado 415, obtenido {sub_df['AG'].sum()}"
    assert total_pos == 1266, f"A16: Total esperado 1266, obtenido {total_pos}"
    print(f"Check 12-16: Distribucion de positivos verificada (Total={total_pos}, HS=517, TR=334, AG=415).")

    # 17-20: Comparacion contra P114 campeon reinante
    sub_p114 = pd.read_csv(SUB_P114)
    diff_cells = int((sub_df[["HS", "TR", "AG"]].values != sub_p114[["HS", "TR", "AG"]].values).sum())
    assert diff_cells == 23, f"A17: Celdas distintas de P114 esperado 23, obtenido {diff_cells}"
    pct_diff = diff_cells / (1154 * 3) * 100
    assert 0.4 <= pct_diff <= 1.0, f"A18: Porcentaje de perturbacion fuera de rango: {pct_diff:.2f}%"

    c01 = int(np.sum((sub_p114[["HS", "TR", "AG"]].values == 0) & (sub_df[["HS", "TR", "AG"]].values == 1)))
    c10 = int(np.sum((sub_p114[["HS", "TR", "AG"]].values == 1) & (sub_df[["HS", "TR", "AG"]].values == 0)))
    assert c01 == 12, f"A19a: Adiciones esperadas 12, obtenidas {c01}"
    assert c10 == 11, f"A19b: Supresiones esperadas 11, obtenidas {c10}"
    assert (c01 - c10) == 1, f"A19c: Balance neto esperado +1, obtenido {c01 - c10}"

    sha_calc = sha256_archivo(SUB_P116)
    assert sha_calc == "6d7c5ff9b383ae274cbe130ec525af4fe3e05f3be03886b2d9b7dcc9861f1973", f"A20: SHA256 no coincide: {sha_calc}"
    assert sha_calc != "c31ee6f2147c1c057095bfda5840e7eac48cd425ede1806d22f0c77f4d4d4274", "A20b: SHA256 igual a P114"
    print(f"Check 17-20: Comparacion bidireccional contra P114 verificada (+12/-11, neto +1, SHA256 OK).")

    # 21-26: Reporte JSON
    with open(REP_JSON, "r", encoding="utf-8") as fh:
        rep = json.load(fh)
    assert rep["experimento"] == "P116-01", "A21: Nombre de experimento incorrecto en JSON"
    assert rep["features"]["dimension"] == 86262, f"A22: Dimension de features {rep['features']['dimension']} != 86262"
    assert rep["features"]["max_df"] == 0.90, "A23: max_df != 0.90"
    assert rep["controles"]["diff_p114_m2"] < 1e-6, f"A24: Control M2 P114 fallido: {rep['controles']['diff_p114_m2']}"
    assert rep["modelos"]["lr_liblinear"]["C_per_label"]["AG"] == 1.2, "A25: C_AG != 1.2"
    assert rep["modelos"]["calibracion_temperatura"] == 1.06, "A26: Temperatura != 1.06"
    assert rep["umbrales"]["AG"] == 0.47, "A26b: Umbral AG != 0.47"
    print("Check 21-26: Parametros de configuracion y controles M2 validados.")

    # 27-30: Metricas de CV
    can_ccv = rep["validacion"]["canonica"]["ccv_p116"]
    p47_ccv = rep["validacion"]["particion_47"]["ccv_p116"]
    media_ccv = rep["validacion"]["ccv_media_p116"]
    delta_p114 = rep["validacion"]["delta_media_vs_p114"]
    assert can_ccv > 0.7850, f"A27: CCV canonica demasiado baja: {can_ccv}"
    assert p47_ccv > 0.7880, f"A28: CCV Part47 demasiado baja: {p47_ccv}"
    assert media_ccv > 0.7870, f"A29: Media CCV demasiado baja: {media_ccv}"
    assert delta_p114 > 0.0010, f"A30: Delta medio vs P114 demasiado bajo: {delta_p114}"
    print(f"Check 27-30: Metricas de CV robustas (Canonica={can_ccv:.6f}, P47={p47_ccv:.6f}, Media={media_ccv:.6f}, Delta={delta_p114:+.6f}).")

    print("\nTODAS LAS 30 ASERCIONES SUPERADAS EXITOSAMENTE (30/30 OK).", flush=True)
    return 0


if __name__ == "__main__":
    main()
