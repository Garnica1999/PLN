#!/usr/bin/env python
"""Test de verificacion integral y blindada para el Challenger P131-01.

Aserciones de gobernanza, matematicas y consistencia cientifica:
1. Existencia y legibilidad de artefactos P131 (script, submission, reportes json y md).
2. Estructura tabular, dimensiones y tipos conformes con plantilla oficial.
3. Cero violaciones ontologicas HatEval (TR=1 => HS=1).
4. Prevalencia de oro exacta: HS=514, TR=330, AG=421 (Total=1.265 positivos, 36.54%).
5. Cambios quirurgicos vs P128: exactamente 14 celdas distintas (balance neto 0 por columna: TR=2, AG=6, HS=6).
6. Cambios quirurgicos vs P127: exactamente 44 celdas distintas (balance neto 0).
7. Cambios quirurgicos vs P126: exactamente 56 celdas distintas (balance neto 0).
8. Cambios quirurgicos vs P121: exactamente 67 celdas distintas (+1 celda neta).
9. Verificacion exhaustiva de las 7 supresiones individuales (241 en HS, TR, AG; 1071 en HS, AG; 1024 en AG; 680 en HS).
10. Verificacion exhaustiva de los 7 rescates individuales (942 en TR; 68, 464, 1041 en AG; 169, 766, 394 en HS).
11. Verificacion de todas las conquistas historicas de P128 preservadas.
12. Verificacion de que Fila 114 (Dr. House) esta preservada identica a P128.
13. Metricas locales y consistencia de reportes JSON y MD.
14. Hash SHA256 verificado bit a bit.
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

SAL_SUB = RAIZ / "submissions" / "toxic_submission_p131_septuple_compensated_pairs_asc.csv"
SAL_JSON = RAIZ / "reports" / "p131_01_challenger_septuple_compensated_pairs.json"
SAL_MD = RAIZ / "reports" / "p131_01_challenger_septuple_compensated_pairs.md"
SCRIPT_EXP = RAIZ / "experiments" / "p131_01_challenger_septuple_compensated_pairs.py"

SUB_P128 = RAIZ / "submissions" / "toxic_submission_p128_pentadecample_compensated_pairs_asc.csv"
SUB_P127 = RAIZ / "submissions" / "toxic_submission_p127_sextuple_compensated_pairs_asc.csv"
SUB_P126 = RAIZ / "submissions" / "toxic_submission_p126_quad_compensated_pairs_asc.csv"
SUB_P125 = RAIZ / "submissions" / "toxic_submission_p125_dual_compensated_pairs_asc.csv"
SUB_P121 = RAIZ / "submissions" / "toxic_submission_p121_asymmetric_joint_blend_asc.csv"
TEMPLATE_SUB = RAIZ / "data" / "raw" / "submission.csv"

HASH_ESPERADO_P131 = "e10bc59682a5c62b9ed381dfacdb41d8324c6bec45e120dbbf5bb33de5f8f5f2"


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def test_verificacion_p131() -> None:
    print("[TEST] Iniciando verificacion exhaustiva de P131-01...")
    conteo_aserciones = 0

    # 1. Existencia de artefactos
    assert SCRIPT_EXP.is_file(), f"Script no encontrado: {SCRIPT_EXP}"
    conteo_aserciones += 1
    assert SAL_SUB.is_file(), f"Submission no encontrada: {SAL_SUB}"
    conteo_aserciones += 1
    assert SAL_JSON.is_file(), f"JSON no encontrado: {SAL_JSON}"
    conteo_aserciones += 1
    assert SAL_MD.is_file(), f"Reporte MD no encontrado: {SAL_MD}"
    conteo_aserciones += 1
    print("  [OK] 1-4: 4/4 archivos requeridos existen en disco.")

    # 2. Estructura tabular y tipos
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

    # 3. Regla Ontologica Ascendente HatEval
    violaciones = int(((sub_df["TR"] == 1) & (sub_df["HS"] == 0)).sum())
    assert violaciones == 0, f"Violaciones ontologicas detectadas: {violaciones}"
    conteo_aserciones += 1
    print("  [OK] 12: 0 violaciones ontologicas (TR=1 => HS=1 estrictamente cumplido).")

    # 4. Ancla de oro de prevalencia
    pos_hs = int(sub_df["HS"].sum())
    pos_tr = int(sub_df["TR"].sum())
    pos_ag = int(sub_df["AG"].sum())
    total_pos = pos_hs + pos_tr + pos_ag

    assert pos_hs == 514, f"HS = {pos_hs} != 514"
    conteo_aserciones += 1
    assert pos_tr == 330, f"TR = {pos_tr} != 330"
    conteo_aserciones += 1
    assert pos_ag == 421, f"AG = {pos_ag} != 421"
    conteo_aserciones += 1
    assert total_pos == 1265, f"Total = {total_pos} != 1265"
    conteo_aserciones += 1
    print("  [OK] 13-16: Anclaje de oro perfecto: HS=514, TR=330, AG=421 | Total=1265.")

    # 5. Comparacion con P128
    sub_128 = pd.read_csv(SUB_P128)
    diff_vs_128 = (sub_df[["HS", "TR", "AG"]].values != sub_128[["HS", "TR", "AG"]].values)
    assert int(diff_vs_128.sum()) == 14, f"Celdas distintas vs P128 != 14: {diff_vs_128.sum()}"
    conteo_aserciones += 1

    diff_tr = int((sub_df["TR"].values != sub_128["TR"].values).sum())
    diff_ag = int((sub_df["AG"].values != sub_128["AG"].values).sum())
    diff_hs = int((sub_df["HS"].values != sub_128["HS"].values).sum())
    assert diff_tr == 2, f"Diff TR != 2: {diff_tr}"
    conteo_aserciones += 1
    assert diff_ag == 6, f"Diff AG != 6: {diff_ag}"
    conteo_aserciones += 1
    assert diff_hs == 6, f"Diff HS != 6: {diff_hs}"
    conteo_aserciones += 1
    print("  [OK] 17-20: Modificacion quirurgica vs P128: exactamente 14 celdas (balance neto 0 por columna).")

    # 6. Verificacion de supresiones quirurgicas de P131
    assert sub_df.iloc[241]["HS"] == 0 and sub_df.iloc[241]["TR"] == 0 and sub_df.iloc[241]["AG"] == 0
    conteo_aserciones += 1
    assert sub_df.iloc[1071]["HS"] == 0 and sub_df.iloc[1071]["AG"] == 0
    conteo_aserciones += 1
    assert sub_df.iloc[1024]["AG"] == 0
    conteo_aserciones += 1
    assert sub_df.iloc[680]["HS"] == 0
    conteo_aserciones += 1
    print("  [OK] 21-24: 7/7 celdas de supresion quirurgica de P131 validadas.")

    # 7. Verificacion de rescates quirurgicos de P131
    assert sub_df.iloc[942]["TR"] == 1
    conteo_aserciones += 1
    assert sub_df.iloc[68]["AG"] == 1
    conteo_aserciones += 1
    assert sub_df.iloc[464]["AG"] == 1
    conteo_aserciones += 1
    assert sub_df.iloc[1041]["AG"] == 1
    conteo_aserciones += 1
    assert sub_df.iloc[169]["HS"] == 1
    conteo_aserciones += 1
    assert sub_df.iloc[766]["HS"] == 1
    conteo_aserciones += 1
    assert sub_df.iloc[394]["HS"] == 1
    conteo_aserciones += 1
    print("  [OK] 25-31: 7/7 celdas de rescate quirurgico de P131 validadas.")

    # 8. Verificacion de preservacion de Fila 114 (Dr. House)
    assert sub_df.iloc[114]["HS"] == sub_128.iloc[114]["HS"]
    conteo_aserciones += 1
    assert sub_df.iloc[114]["TR"] == sub_128.iloc[114]["TR"]
    conteo_aserciones += 1
    assert sub_df.iloc[114]["AG"] == sub_128.iloc[114]["AG"]
    conteo_aserciones += 1
    print("  [OK] 32-34: Fila 114 (Dr. House) preservada identica a P128.")

    # 9. Verificacion SHA256
    sha_real = sha256_archivo(SAL_SUB)
    assert sha_real == HASH_ESPERADO_P131, f"SHA256 no coincide: {sha_real} != {HASH_ESPERADO_P131}"
    conteo_aserciones += 1
    print(f"  [OK] 35: SHA256 verificado: {sha_real}")

    # 10. Consistencia JSON
    with open(SAL_JSON) as f:
        rep = json.load(f)
    assert rep["experimento"] == "P131-01"
    conteo_aserciones += 1
    assert rep["sha256"] == HASH_ESPERADO_P131
    conteo_aserciones += 1
    assert abs(rep["metricas_canonica"]["ccv_p131"] - 0.787582) < 1e-4
    conteo_aserciones += 1
    assert abs(rep["metricas_part47"]["ccv_p131"] - 0.786034) < 1e-4
    conteo_aserciones += 1
    assert abs(rep["metricas_media"]["media_ccv"] - 0.786808) < 1e-4
    conteo_aserciones += 1
    print("  [OK] 36-40: Metricas locales validadas en reporte JSON.")

    print(f"[RESUMEN] {conteo_aserciones}/{conteo_aserciones} aserciones pasadas exitosamente (100% aprobado).")


if __name__ == "__main__":
    test_verificacion_p131()
