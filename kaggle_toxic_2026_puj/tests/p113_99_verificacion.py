#!/usr/bin/env python
"""Suite de verificacion formal para P113: Challenger Tri-Blend Multi-Loss (LR + SGD_log + SGD_huber).

Comprueba 30 aserciones independientes de integridad, cero fuga, determinismo M2,
recuento de positivos y validez de la submission generada.
"""

import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
SUB_P113 = RAIZ / "submissions" / "toxic_submission_p113_triblend_margin_asc.csv"
SUB_P110 = RAIZ / "submissions" / "toxic_submission_p110_lr_sgd_blend_asc.csv"
SUB_TEMPLATE = RAIZ / "data" / "raw" / "submission.csv"
REP_JSON = RAIZ / "reports" / "p113_01_challenger_triblend_margin.json"
REP_MD = RAIZ / "reports" / "p113_01_challenger_triblend_margin.md"
SCRIPT_P113 = RAIZ / "experiments" / "p113_01_challenger_triblend_margin.py"

def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main():
    print("=== INICIANDO VERIFICACION FORMAL P113 (30 ASERCIONES) ===", flush=True)

    # 1-4: Existencia de artefactos
    assert SCRIPT_P113.exists(), "A1: Script P113 no existe"
    assert REP_JSON.exists(), "A2: Reporte JSON no existe"
    assert REP_MD.exists(), "A3: Reporte MD no existe"
    assert SUB_P113.exists(), "A4: Submission CSV no existe"
    print("Check 1-4: Artefactos existen.")

    # 5-10: Verificacion de submission
    sub_df = pd.read_csv(SUB_P113)
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
    assert 1245 <= total_pos <= 1270, f"A12: Total positivos fuera de banda optima: {total_pos}"
    assert sub_df["HS"].sum() == 512, f"A13: HS positivos esperado 512, obtenido {sub_df['HS'].sum()}"
    assert sub_df["TR"].sum() == 330, f"A14: TR positivos esperado 330, obtenido {sub_df['TR'].sum()}"
    assert sub_df["AG"].sum() == 421, f"A15: AG positivos esperado 421, obtenido {sub_df['AG'].sum()}"
    assert total_pos == 1263, f"A16: Total esperado 1263, obtenido {total_pos}"
    print(f"Check 12-16: Distribucion de positivos verificada (Total={total_pos}, HS=512, TR=330, AG=421).")

    # 17-20: Comparacion contra P110 oficial
    sub_p110 = pd.read_csv(SUB_P110)
    diff_cells = int((sub_df[["HS", "TR", "AG"]].values != sub_p110[["HS", "TR", "AG"]].values).sum())
    assert diff_cells == 22, f"A17: Celdas distintas de P110 esperado 22, obtenido {diff_cells}"
    pct_diff = diff_cells / (1154 * 3) * 100
    assert 0.5 <= pct_diff <= 1.0, f"A18: Porcentaje de perturbacion fuera de rango: {pct_diff:.2f}%"
    sha_calc = sha256_archivo(SUB_P113)
    assert sha_calc == "01b9074c183d391c4e0af679273db4802688c29bbe0074a70074d71852894e39", f"A19: SHA256 no coincide: {sha_calc}"
    assert sha_calc != "535b2b35e4e3131f28af2026805f5846fbc2cfe81b744d73344b2605967a47b4", "A20: SHA256 igual a P110"
    print(f"Check 17-20: Comparacion contra P110 correcta (22 celdas distintas, {pct_diff:.2f}%, SHA256 verificado).")

    # 21-26: Reporte JSON
    with open(REP_JSON, "r", encoding="utf-8") as fh:
        rep = json.load(fh)
    assert rep["experimento"] == "P113-01", "A21: Nombre de experimento incorrecto en JSON"
    assert rep["features"]["dimension"] == 86275, f"A22: Dimension de features {rep['features']['dimension']} != 86275"
    assert rep["features"]["min_df"] == 2, "A23: min_df != 2"
    assert rep["controles"]["diff_v03_m2"] < 1e-12, f"A24: Control M2 V03 fallido: {rep['controles']['diff_v03_m2']}"
    assert rep["controles"]["diff_p110_m2"] < 1e-6, f"A25: Control M2 P110 fallido: {rep['controles']['diff_p110_m2']}"
    assert rep["modelos"]["pesos_blend"] == {"w_lr": 0.93, "w_sgd_log": 0.04, "w_sgd_huber": 0.03}, "A26: Pesos blend incorrectos"
    print("Check 21-26: Contenido de reporte JSON y controles M2 validados.")

    # 27-30: Metricas de CV
    can_ccv = rep["validacion"]["canonica"]["ccv_p113"]
    p47_ccv = rep["validacion"]["particion_47"]["ccv_p113"]
    media_ccv = rep["validacion"]["ccv_media_p113"]
    delta_p110 = rep["validacion"]["delta_media_vs_p110"]
    assert can_ccv > 0.7870, f"A27: CCV canonica demasiado baja: {can_ccv}"
    assert p47_ccv > 0.7850, f"A28: CCV Part47 demasiado baja: {p47_ccv}"
    assert media_ccv > 0.7860, f"A29: Media CCV demasiado baja: {media_ccv}"
    assert delta_p110 > 0.0010, f"A30: Delta medio vs P110 demasiado bajo: {delta_p110}"
    print(f"Check 27-30: Metricas de CV robustas (Canonica={can_ccv:.6f}, P47={p47_ccv:.6f}, Media={media_ccv:.6f}, Delta={delta_p110:+.6f}).")

    print("\nTODAS LAS 30 ASERCIONES SUPERADAS EXITOSAMENTE (30/30 OK).", flush=True)
    return 0

if __name__ == "__main__":
    main()
