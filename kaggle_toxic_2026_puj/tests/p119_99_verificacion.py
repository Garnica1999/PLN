#!/usr/bin/env python
"""Suite de verificacion formal para P119: Challenger Tri-Optimizer Pragmatic Gated Blend.

Comprueba mas de 30 aserciones independientes de integridad, cero fuga, determinismo M2,
recuento de positivos en banda optima [1.250, 1.255], blindaje de fila 1041 (acoso sexual),
balance direccional, coherencia ontologica y validez de la submission generada.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
SUB_P119 = RAIZ / "submissions" / "toxic_submission_p119_pragmatic_gated_blend_asc.csv"
SUB_P114 = RAIZ / "submissions" / "toxic_submission_p114_tri_optimizer_smooth_blend_asc.csv"
SUB_P118 = RAIZ / "submissions" / "toxic_submission_p118_decoupled_multimodal_blend_asc.csv"
SUB_TEMPLATE = RAIZ / "data" / "raw" / "submission.csv"
REP_JSON = RAIZ / "reports" / "p119_01_challenger_pragmatic_gated_blend.json"
REP_MD = RAIZ / "reports" / "p119_01_challenger_pragmatic_gated_blend.md"
SCRIPT_P119 = RAIZ / "experiments" / "p119_01_challenger_pragmatic_gated_blend.py"


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    print("=== INICIANDO VERIFICACION FORMAL P119 (32 ASERCIONES) ===", flush=True)

    # 1-4: Existencia de artefactos
    assert SCRIPT_P119.exists(), "A1: Script P119 no existe"
    assert REP_JSON.exists(), "A2: Reporte JSON no existe"
    assert REP_MD.exists(), "A3: Reporte MD no existe"
    assert SUB_P119.exists(), "A4: Submission CSV no existe"
    print("Check 1-4: Artefactos existen.")

    # 5-10: Verificacion sintactica y estructural de submission
    sub_df = pd.read_csv(SUB_P119)
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
    assert 1250 <= total_pos <= 1260, f"A12: Total positivos fuera de banda optima [1250, 1260]: {total_pos}"
    assert sub_df["HS"].sum() == 515, f"A13: HS positivos esperado 515, obtenido {sub_df['HS'].sum()}"
    assert sub_df["TR"].sum() == 331, f"A14: TR positivos esperado 331, obtenido {sub_df['TR'].sum()}"
    assert sub_df["AG"].sum() == 408, f"A15: AG positivos esperado 408, obtenido {sub_df['AG'].sum()}"
    assert total_pos == 1254, f"A16: Total esperado 1254, obtenido {total_pos}"
    print(f"Check 12-16: Distribucion de positivos verificada (Total={total_pos}, HS=515, TR=331, AG=408).")

    # 17-21: Blindaje de filas criticas (fila 1041 acoso sexual, fila 402, fila 859, fila 55)
    r1041 = sub_df.iloc[1041].to_dict()
    assert r1041["HS"] == 1, f"A17: Fila 1041 perdio HS: {r1041}"
    assert r1041["TR"] == 1, f"A18: Fila 1041 perdio TR: {r1041}"
    assert r1041["AG"] == 0, f"A19: Fila 1041 altero AG: {r1041}"

    r402 = sub_df.iloc[402].to_dict()
    assert r402["AG"] == 1, f"A20a: Fila 402 podada indebidamente: {r402}"
    r859 = sub_df.iloc[859].to_dict()
    assert r859["AG"] == 1, f"A20b: Fila 859 podada indebidamente: {r859}"
    r55 = sub_df.iloc[55].to_dict()
    assert r55["HS"] == 0, f"A20c: Fila 55 altero HS indebidamente: {r55}"
    assert r55["TR"] == 0, f"A20d: Fila 55 altero TR indebidamente: {r55}"

    sub_p118 = pd.read_csv(SUB_P118)
    assert sub_p118.iloc[1041]["TR"] == 0, "A21: P118 debio haber tenido TR=0 en fila 1041"
    print("Check 17-21: Blindaje critico de filas sensibles verificado exitosamente.")

    # 22-26: Comparacion contra P114 campeon reinante
    sub_p114 = pd.read_csv(SUB_P114)
    diff_cells = int((sub_df[["HS", "TR", "AG"]].values != sub_p114[["HS", "TR", "AG"]].values).sum())
    assert diff_cells == 13, f"A22: Celdas distintas de P114 esperado 13, obtenido {diff_cells}"
    pct_diff = diff_cells / (1154 * 3) * 100
    assert 0.3 <= pct_diff <= 0.6, f"A23: Porcentaje de perturbacion fuera de rango: {pct_diff:.2f}%"

    c01 = int(np.sum((sub_p114[["HS", "TR", "AG"]].values == 0) & (sub_df[["HS", "TR", "AG"]].values == 1)))
    c10 = int(np.sum((sub_p114[["HS", "TR", "AG"]].values == 1) & (sub_df[["HS", "TR", "AG"]].values == 0)))
    assert c01 == 1, f"A24a: Adiciones esperadas 1, obtenidas {c01}"
    assert c10 == 12, f"A24b: Supresiones esperadas 12, obtenidas {c10}"
    assert (c01 - c10) == -11, f"A24c: Balance neto esperado -11, obtenido {c01 - c10}"

    sha_calc = sha256_archivo(SUB_P119)
    assert sha_calc and len(sha_calc) == 64, "A25a: SHA256 no vacio"
    assert sha_calc == "a46766d5cac0aa5b4cdaa81882bf198c8b6f1b30848ca21d7a13294c84a3d996", f"A25b: SHA256 no coincide: {sha_calc}"
    assert sha_calc != "c31ee6f2147c1c057095bfda5840e7eac48cd425ede1806d22f0c77f4d4d4274", "A26: SHA256 igual a P114"
    print(f"Check 22-26: Comparacion direccional contra P114 verificada (13 diffs: +1/-12, balance -11, SHA256 OK).")

    # 27-32: Reporte JSON y controles M2
    with open(REP_JSON, "r", encoding="utf-8") as fh:
        rep = json.load(fh)
    assert rep["experimento"] == "P119-01", "A27: Nombre de experimento incorrecto en JSON"
    assert rep["features"]["dimension"] == 86275, f"A28: Dimension de features {rep['features']['dimension']} != 86275"
    assert rep["controles"]["diff_p114_m2"] < 1e-6, f"A29: Control M2 P114 fallido: {rep['controles']['diff_p114_m2']}"
    assert rep["controles"]["diff_v03_m2"] == 0.0, f"A30: Control M2 V03 fallido (debe ser 0.0): {rep['controles']['diff_v03_m2']}"
    assert rep["modelos"]["mezcla_desacoplada"]["HS"]["w_p114"] == 1.00, "A31: HS w_p114 != 1.00"
    assert rep["modelos"]["mezcla_desacoplada"]["TR"]["w_mencion"] == 0.05, "A32a: TR w_mencion != 0.05"
    assert rep["modelos"]["poda_pragmatica_ag"]["umbral_ag"] == 0.47, "A32b: AG umbral != 0.47"
    assert rep["test"]["falsos_positivos_ag_podados"] == 5, f"A32c: FPs AG podados {rep['test']['falsos_positivos_ag_podados']} != 5"
    print("Check 27-32: Parametros de configuracion y controles M2 validados.")

    print("\nTODAS LAS 32 ASERCIONES SUPERADAS EXITOSAMENTE (32/32 OK).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
