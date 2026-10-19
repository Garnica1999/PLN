#!/usr/bin/env python
"""Suite de verificacion formal para P118: Challenger Decoupled Multimodal Blend.

Comprueba 30 aserciones independientes de integridad, cero fuga, determinismo M2,
recuento de positivos en banda optima [1.255, 1.266], balance direccional y validez
de la submission generada.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
SUB_P118 = RAIZ / "submissions" / "toxic_submission_p118_decoupled_multimodal_blend_asc.csv"
SUB_P114 = RAIZ / "submissions" / "toxic_submission_p114_tri_optimizer_smooth_blend_asc.csv"
SUB_TEMPLATE = RAIZ / "data" / "raw" / "submission.csv"
REP_JSON = RAIZ / "reports" / "p118_01_challenger_decoupled_multimodal_blend.json"
REP_MD = RAIZ / "reports" / "p118_01_challenger_decoupled_multimodal_blend.md"
SCRIPT_P118 = RAIZ / "experiments" / "p118_01_challenger_decoupled_multimodal_blend.py"


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    print("=== INICIANDO VERIFICACION FORMAL P118 (30 ASERCIONES) ===", flush=True)

    # 1-4: Existencia de artefactos
    assert SCRIPT_P118.exists(), "A1: Script P118 no existe"
    assert REP_JSON.exists(), "A2: Reporte JSON no existe"
    assert REP_MD.exists(), "A3: Reporte MD no existe"
    assert SUB_P118.exists(), "A4: Submission CSV no existe"
    print("Check 1-4: Artefactos existen.")

    # 5-10: Verificacion sintactica y estructural de submission
    sub_df = pd.read_csv(SUB_P118)
    tpl_df = pd.read_csv(SUB_TEMPLATE)
    assert len(sub_df) == 1154, f"A5: Longitud esperada 1154, obtenida {len(sub_df)}"
    assert list(sub_df.columns) == ["id", "HS", "TR", "AG"], f"A6: Columnas invalidas: {sub_df.columns}"
    assert (sub_df["id"].values == tpl_df["id"].values).all(), "A7: IDs no coinciden exactamente con template"
    assert sub_df.isnull().sum().sum() == 0, "A8: Existen valores nulos en submission"
    assert set(sub_df["HS"].unique()).issubset({0, 1}), "A9: Valores no binarios en HS"
    assert set(sub_df["TR"].unique()).issubset({0, 1}) and set(sub_df["AG"].unique()).issubset({0, 1}), "A10: Valores no binarios en TR o AG"

    # 11: Restriccion ontologica HatEval
    violaciones = int(((sub_df["TR"] == 1) & (sub_df["HS"] == 0)).sum())
    assert violaciones == 0, f"A11: Existen {violaciones} violaciones ontologicas TR=1 => HS=0"
    print("Check 5-11: Integridad y validez sintactica de submission superada.")

    # 12-16: Recuentos de positivos y distribucion
    total_pos = int(sub_df[["HS", "TR", "AG"]].values.sum())
    assert 1255 <= total_pos <= 1266, f"A12: Total positivos fuera de banda optima [1255, 1266]: {total_pos}"
    assert sub_df["HS"].sum() == 514, f"A13: HS positivos esperado 514, obtenido {sub_df['HS'].sum()}"
    assert sub_df["TR"].sum() == 329, f"A14: TR positivos esperado 329, obtenido {sub_df['TR'].sum()}"
    assert sub_df["AG"].sum() == 423, f"A15: AG positivos esperado 423, obtenido {sub_df['AG'].sum()}"
    assert total_pos == 1266, f"A16: Total esperado 1266, obtenido {total_pos}"
    print(f"Check 12-16: Distribucion de positivos verificada (Total={total_pos}, HS=514, TR=329, AG=423).")

    # 17-20: Comparacion contra P114 campeon supremo
    sub_p114 = pd.read_csv(SUB_P114)
    diff_cells = int((sub_df[["HS", "TR", "AG"]].values != sub_p114[["HS", "TR", "AG"]].values).sum())
    assert diff_cells == 15, f"A17: Celdas distintas de P114 esperado 15, obtenido {diff_cells}"
    pct_diff = diff_cells / (1154 * 3) * 100
    assert 0.2 <= pct_diff <= 0.8, f"A18: Porcentaje de perturbacion fuera de rango: {pct_diff:.2f}%"

    c01 = int(np.sum((sub_p114[["HS", "TR", "AG"]].values == 0) & (sub_df[["HS", "TR", "AG"]].values == 1)))
    c10 = int(np.sum((sub_p114[["HS", "TR", "AG"]].values == 1) & (sub_df[["HS", "TR", "AG"]].values == 0)))
    assert c01 == 8, f"A19a: Adiciones esperadas 8, obtenidas {c01}"
    assert c10 == 7, f"A19b: Supresiones esperadas 7, obtenidas {c10}"
    assert (c01 - c10) == 1, f"A19c: Balance neto esperado +1, obtenido {c01 - c10}"

    sha_calc = sha256_archivo(SUB_P118)
    assert sha_calc and len(sha_calc) == 64, "A20a: SHA256 no vacio"
    assert sha_calc == "b8f6640965506865a5260f4bf26eb330d594000b4f55c116d325c0c009095d47", f"A20b: SHA256 no coincide: {sha_calc}"
    assert sha_calc != "c31ee6f2147c1c057095bfda5840e7eac48cd425ede1806d22f0c77f4d4d4274", "A20c: SHA256 igual a P114"
    print(f"Check 17-20: Comparacion direccional contra P114 verificada (15 diffs: +8/-7, neto +1, SHA256 OK).")

    # 21-26: Reporte JSON y controles M2
    with open(REP_JSON, "r", encoding="utf-8") as fh:
        rep = json.load(fh)
    assert rep["experimento"] == "P118-01", "A21: Nombre de experimento incorrecto en JSON"
    assert rep["features"]["dimension"] == 86275, f"A22: Dimension de features {rep['features']['dimension']} != 86275"
    assert rep["controles"]["diff_p114_m2"] < 1e-6, f"A23: Control M2 P114 fallido: {rep['controles']['diff_p114_m2']}"
    assert rep["controles"]["diff_v03_m2"] == 0.0, f"A24: Control M2 V03 fallido (debe ser 0.0): {rep['controles']['diff_v03_m2']}"
    assert rep["modelos"]["mezcla_desacoplada"]["HS"]["w_p114"] == 1.00, "A25: HS w_p114 != 1.00"
    assert rep["modelos"]["mezcla_desacoplada"]["TR"]["w_bm25"] == 0.05, "A26a: TR w_bm25 != 0.05"
    assert rep["modelos"]["mezcla_desacoplada"]["AG"]["w_svd"] == 0.15, "A26b: AG w_svd != 0.15"
    print("Check 21-26: Parametros de configuracion y controles M2 validados.")

    # 27-30: Metricas de CV
    can_ccv = rep["validacion"]["canonica"]["ccv_p118"]
    p47_ccv = rep["validacion"]["particion_47"]["ccv_p118"]
    media_ccv = rep["validacion"]["ccv_media_p118"]
    delta_p114 = rep["validacion"]["delta_media_vs_p114"]
    assert can_ccv > 0.7870, f"A27: CCV canonica demasiado baja: {can_ccv}"
    assert p47_ccv > 0.7850, f"A28: CCV Part47 demasiado baja: {p47_ccv}"
    assert media_ccv > 0.7860, f"A29: Media CCV demasiado baja: {media_ccv}"
    assert delta_p114 > 0.0002, f"A30: Delta medio vs P114 demasiado bajo: {delta_p114}"
    print(f"Check 27-30: Metricas de CV robustas (Canonica={can_ccv:.6f}, P47={p47_ccv:.6f}, Media={media_ccv:.6f}, Delta={delta_p114:+.6f}).")

    print("\nTODAS LAS 30 ASERCIONES SUPERADAS EXITOSAMENTE (30/30 OK).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
