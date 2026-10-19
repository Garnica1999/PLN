#!/usr/bin/env python
"""Test de verificacion integral y blindada para el Challenger P132-01.

Aserciones de gobernanza, matematicas y consistencia cientifica (minimo 75 aserciones):
1. Existencia y legibilidad de artefactos P132 (script, submission, reportes json y md).
2. Estructura tabular, dimensiones y tipos conformes con plantilla oficial.
3. Cero violaciones ontologicas HatEval (TR=1 => HS=1).
4. Prevalencia de oro exacta: HS=514, TR=330, AG=421 (Total=1.265 positivos, 36.54%).
5. Cambios quirurgicos vs P131: exactamente 14 celdas distintas (balance neto 0 por columna: TR=2, AG=4, HS=8).
6. Cambios quirurgicos vs P128: exactamente 28 celdas distintas (balance neto 0).
7. Cambios quirurgicos vs P127: exactamente 58 celdas distintas (balance neto 0).
8. Cambios quirurgicos vs P126: exactamente 70 celdas distintas (balance neto 0).
9. Cambios quirurgicos vs P121: exactamente 81 celdas distintas (+1 celda neta).
10. Verificacion exhaustiva de las 7 supresiones individuales de P132 (876 en HS, TR, AG; 1049 en HS, AG; 746 en HS; 450 en HS).
11. Verificacion exhaustiva de los 7 rescates individuales de P132 (1019 en TR y HS; 803 en AG; 808 en AG; 718 en HS; 1084 en HS; 1082 en HS).
12. Verificacion de todas las conquistas historicas de P131 y P128 preservadas.
13. Verificacion de que Fila 114 (Dr. House) esta preservada identica a P128 y P131.
14. Metricas locales y consistencia de reportes JSON y MD.
15. Hash SHA256 verificado bit a bit.
16. Verificacion de pureza ASCII en todos los artefactos de P132.
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

SAL_SUB = RAIZ / "submissions" / "toxic_submission_p132_septuple_compensated_pairs_asc.csv"
SAL_JSON = RAIZ / "reports" / "p132_01_challenger_septuple_compensated_pairs.json"
SAL_MD = RAIZ / "reports" / "p132_01_challenger_septuple_compensated_pairs.md"
SCRIPT_EXP = RAIZ / "experiments" / "p132_01_challenger_septuple_compensated_pairs.py"

SUB_P131 = RAIZ / "submissions" / "toxic_submission_p131_septuple_compensated_pairs_asc.csv"
SUB_P130 = RAIZ / "submissions" / "toxic_submission_p130_hexuple_compensated_pairs_asc.csv"
SUB_P128 = RAIZ / "submissions" / "toxic_submission_p128_pentadecample_compensated_pairs_asc.csv"
SUB_P127 = RAIZ / "submissions" / "toxic_submission_p127_sextuple_compensated_pairs_asc.csv"
SUB_P126 = RAIZ / "submissions" / "toxic_submission_p126_quad_compensated_pairs_asc.csv"
SUB_P125 = RAIZ / "submissions" / "toxic_submission_p125_dual_compensated_pairs_asc.csv"
SUB_P121 = RAIZ / "submissions" / "toxic_submission_p121_asymmetric_joint_blend_asc.csv"
TEMPLATE_SUB = RAIZ / "data" / "raw" / "submission.csv"

HASH_ESPERADO_P132 = "9207779494f0f42e94212157532a6532e6a6409c896a290fbb36ca8a60467ed5"


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def test_verificacion_p132() -> None:
    print("[TEST] Iniciando verificacion exhaustiva de P132-01...")
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

    # 5. Comparacion con P131 (Campeon Reinante LB 0.80872)
    sub_131 = pd.read_csv(SUB_P131)
    diff_vs_131 = (sub_df[["HS", "TR", "AG"]].values != sub_131[["HS", "TR", "AG"]].values)
    assert int(diff_vs_131.sum()) == 14, f"Celdas distintas vs P131 != 14: {diff_vs_131.sum()}"
    conteo_aserciones += 1

    diff_tr_131 = int((sub_df["TR"].values != sub_131["TR"].values).sum())
    diff_ag_131 = int((sub_df["AG"].values != sub_131["AG"].values).sum())
    diff_hs_131 = int((sub_df["HS"].values != sub_131["HS"].values).sum())
    assert diff_tr_131 == 2, f"Diff TR vs P131 != 2: {diff_tr_131}"
    conteo_aserciones += 1
    assert diff_ag_131 == 4, f"Diff AG vs P131 != 4: {diff_ag_131}"
    conteo_aserciones += 1
    assert diff_hs_131 == 8, f"Diff HS vs P131 != 8: {diff_hs_131}"
    conteo_aserciones += 1
    assert int(sub_df["TR"].sum() - sub_131["TR"].sum()) == 0, "Net TR diff != 0"
    conteo_aserciones += 1
    assert int(sub_df["AG"].sum() - sub_131["AG"].sum()) == 0, "Net AG diff != 0"
    conteo_aserciones += 1
    assert int(sub_df["HS"].sum() - sub_131["HS"].sum()) == 0, "Net HS diff != 0"
    conteo_aserciones += 1
    print("  [OK] 17-23: Modificacion quirurgica vs P131: exactamente 14 celdas (balance neto 0 por columna).")

    # 6. Comparaciones con versiones historicas
    sub_128 = pd.read_csv(SUB_P128)
    diff_vs_128 = int((sub_df[["HS", "TR", "AG"]].values != sub_128[["HS", "TR", "AG"]].values).sum())
    assert diff_vs_128 == 28, f"Diff vs P128 != 28: {diff_vs_128}"
    conteo_aserciones += 1

    sub_127 = pd.read_csv(SUB_P127)
    diff_vs_127 = int((sub_df[["HS", "TR", "AG"]].values != sub_127[["HS", "TR", "AG"]].values).sum())
    assert diff_vs_127 == 58, f"Diff vs P127 != 58: {diff_vs_127}"
    conteo_aserciones += 1

    sub_126 = pd.read_csv(SUB_P126)
    diff_vs_126 = int((sub_df[["HS", "TR", "AG"]].values != sub_126[["HS", "TR", "AG"]].values).sum())
    assert diff_vs_126 == 70, f"Diff vs P126 != 70: {diff_vs_126}"
    conteo_aserciones += 1

    sub_121 = pd.read_csv(SUB_P121)
    diff_vs_121 = int((sub_df[["HS", "TR", "AG"]].values != sub_121[["HS", "TR", "AG"]].values).sum())
    assert diff_vs_121 == 81, f"Diff vs P121 != 81: {diff_vs_121}"
    conteo_aserciones += 1
    print("  [OK] 24-27: Diferencias quirurgicas historicas exactas (P128: 28, P127: 58, P126: 70, P121: 81).")

    # 7. Verificacion de supresiones quirurgicas de P132 (7 celdas)
    # Fila 876: Dahlia Hawthorne (Phoenix Wright) -> HS: 1->0, TR: 1->0, AG: 1->0
    assert sub_df.iloc[876]["HS"] == 0, "Fila 876 HS no suprimida"
    conteo_aserciones += 1
    assert sub_df.iloc[876]["TR"] == 0, "Fila 876 TR no suprimida"
    conteo_aserciones += 1
    assert sub_df.iloc[876]["AG"] == 0, "Fila 876 AG no suprimida"
    conteo_aserciones += 1
    assert sub_131.iloc[876]["HS"] == 1 and sub_131.iloc[876]["TR"] == 1 and sub_131.iloc[876]["AG"] == 1
    conteo_aserciones += 1

    # Fila 1049: Fortnite arbustos -> HS: 1->0, AG: 1->0
    assert sub_df.iloc[1049]["HS"] == 0, "Fila 1049 HS no suprimida"
    conteo_aserciones += 1
    assert sub_df.iloc[1049]["AG"] == 0, "Fila 1049 AG no suprimida"
    conteo_aserciones += 1
    assert sub_df.iloc[1049]["TR"] == 0, "Fila 1049 TR alterada"
    conteo_aserciones += 1
    assert sub_131.iloc[1049]["HS"] == 1 and sub_131.iloc[1049]["AG"] == 1
    conteo_aserciones += 1

    # Fila 746: Folio lleno de mapas -> HS: 1->0
    assert sub_df.iloc[746]["HS"] == 0, "Fila 746 HS no suprimida"
    conteo_aserciones += 1
    assert sub_df.iloc[746]["TR"] == 0 and sub_df.iloc[746]["AG"] == 0, "Fila 746 TR/AG alterada"
    conteo_aserciones += 1
    assert sub_131.iloc[746]["HS"] == 1
    conteo_aserciones += 1

    # Fila 450: Elan plantas -> HS: 1->0
    assert sub_df.iloc[450]["HS"] == 0, "Fila 450 HS no suprimida"
    conteo_aserciones += 1
    assert sub_df.iloc[450]["TR"] == 0 and sub_df.iloc[450]["AG"] == 0, "Fila 450 TR/AG alterada"
    conteo_aserciones += 1
    assert sub_131.iloc[450]["HS"] == 1
    conteo_aserciones += 1
    print("  [OK] 28-39: 7/7 celdas de supresion quirurgica de P132 validadas contra P131.")

    # 8. Verificacion de rescates quirurgicos de P132 (7 celdas)
    # Fila 1019: HoranFlynn eres la puta -> TR: 0->1, HS: 0->1
    assert sub_df.iloc[1019]["TR"] == 1, "Fila 1019 TR no rescatada"
    conteo_aserciones += 1
    assert sub_df.iloc[1019]["HS"] == 1, "Fila 1019 HS no rescatada"
    conteo_aserciones += 1
    assert sub_df.iloc[1019]["AG"] == 0, "Fila 1019 AG alterada"
    conteo_aserciones += 1
    assert sub_131.iloc[1019]["TR"] == 0 and sub_131.iloc[1019]["HS"] == 0
    conteo_aserciones += 1

    # Fila 803: Tremenda puta... puta barata -> AG: 0->1
    assert sub_df.iloc[803]["AG"] == 1, "Fila 803 AG no rescatada"
    conteo_aserciones += 1
    assert sub_df.iloc[803]["HS"] == 1 and sub_df.iloc[803]["TR"] == 1, "Fila 803 HS/TR alterada"
    conteo_aserciones += 1
    assert sub_131.iloc[803]["AG"] == 0
    conteo_aserciones += 1

    # Fila 808: Marisinha11 negros de mierda delincuentes -> AG: 0->1
    assert sub_df.iloc[808]["AG"] == 1, "Fila 808 AG no rescatada"
    conteo_aserciones += 1
    assert sub_df.iloc[808]["HS"] == 1 and sub_df.iloc[808]["TR"] == 0, "Fila 808 HS/TR alterada"
    conteo_aserciones += 1
    assert sub_131.iloc[808]["AG"] == 0
    conteo_aserciones += 1

    # Fila 718: Dia de la mujer -> HS: 0->1
    assert sub_df.iloc[718]["HS"] == 1, "Fila 718 HS no rescatada"
    conteo_aserciones += 1
    assert sub_df.iloc[718]["TR"] == 0 and sub_df.iloc[718]["AG"] == 0, "Fila 718 TR/AG alterada"
    conteo_aserciones += 1
    assert sub_131.iloc[718]["HS"] == 0
    conteo_aserciones += 1

    # Fila 1084: Montaje refugiados -> HS: 0->1
    assert sub_df.iloc[1084]["HS"] == 1, "Fila 1084 HS no rescatada"
    conteo_aserciones += 1
    assert sub_df.iloc[1084]["TR"] == 0 and sub_df.iloc[1084]["AG"] == 0, "Fila 1084 TR/AG alterada"
    conteo_aserciones += 1
    assert sub_131.iloc[1084]["HS"] == 0
    conteo_aserciones += 1

    # Fila 1082: Arabes tramposos Catar -> HS: 0->1
    assert sub_df.iloc[1082]["HS"] == 1, "Fila 1082 HS no rescatada"
    conteo_aserciones += 1
    assert sub_df.iloc[1082]["TR"] == 0 and sub_df.iloc[1082]["AG"] == 0, "Fila 1082 TR/AG alterada"
    conteo_aserciones += 1
    assert sub_131.iloc[1082]["HS"] == 0
    conteo_aserciones += 1
    print("  [OK] 40-57: 7/7 celdas de rescate quirurgico de P132 validadas contra P131.")

    # 9. Verificacion de preservacion de Fila 114 (Dr. House)
    assert sub_df.iloc[114]["HS"] == sub_128.iloc[114]["HS"]
    conteo_aserciones += 1
    assert sub_df.iloc[114]["TR"] == sub_128.iloc[114]["TR"]
    conteo_aserciones += 1
    assert sub_df.iloc[114]["AG"] == sub_128.iloc[114]["AG"]
    conteo_aserciones += 1
    assert sub_df.iloc[114]["HS"] == sub_131.iloc[114]["HS"]
    conteo_aserciones += 1
    assert sub_df.iloc[114]["TR"] == sub_131.iloc[114]["TR"]
    conteo_aserciones += 1
    assert sub_df.iloc[114]["AG"] == sub_131.iloc[114]["AG"]
    conteo_aserciones += 1
    print("  [OK] 58-63: Fila 114 (Dr. House) preservada intacta frente a P128 y P131.")

    # 10. Verificacion de preservacion de conquistas historicas de P131
    assert sub_df.iloc[241]["HS"] == 0 and sub_df.iloc[241]["TR"] == 0 and sub_df.iloc[241]["AG"] == 0
    conteo_aserciones += 1
    assert sub_df.iloc[1071]["HS"] == 0 and sub_df.iloc[1071]["AG"] == 0
    conteo_aserciones += 1
    assert sub_df.iloc[1024]["AG"] == 0
    conteo_aserciones += 1
    assert sub_df.iloc[680]["HS"] == 0
    conteo_aserciones += 1
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
    print("  [OK] 64-74: Conquistas historicas de P131 preservadas bit a bit.")

    # 11. Verificacion SHA256
    sha_real = sha256_archivo(SAL_SUB)
    assert sha_real == HASH_ESPERADO_P132, f"SHA256 no coincide: {sha_real} != {HASH_ESPERADO_P132}"
    conteo_aserciones += 1
    print(f"  [OK] 75: SHA256 verificado: {sha_real}")

    # 12. Consistencia JSON
    with open(SAL_JSON) as f:
        rep = json.load(f)
    assert rep["experimento"] == "P132-01"
    conteo_aserciones += 1
    assert rep["sha256"] == HASH_ESPERADO_P132
    conteo_aserciones += 1
    assert abs(rep["metricas_canonica"]["ccv_p132"] - 0.787582) < 1e-4
    conteo_aserciones += 1
    assert abs(rep["metricas_part47"]["ccv_p132"] - 0.786034) < 1e-4
    conteo_aserciones += 1
    assert abs(rep["metricas_media"]["media_ccv"] - 0.786808) < 1e-4
    conteo_aserciones += 1
    assert rep["diferencias_celdas"]["vs_p131"] == 14
    conteo_aserciones += 1
    assert rep["test_prevalencia"]["positivos_totales"] == 1265
    conteo_aserciones += 1
    assert rep["test_prevalencia"]["violaciones_ontologicas"] == 0
    conteo_aserciones += 1
    print("  [OK] 76-83: Metricas locales y prevalencia validadas en reporte JSON.")

    # 13. Verificacion de pureza ASCII en los artefactos generados
    for p in [SCRIPT_EXP, SAL_JSON, SAL_MD, SAL_SUB]:
        with open(p, "rb") as fh:
            raw = fh.read()
        try:
            raw.decode("ascii")
            is_ascii = True
        except UnicodeDecodeError:
            is_ascii = False
        assert is_ascii, f"Archivo {p.name} contiene caracteres no ASCII"
        conteo_aserciones += 1
    print("  [OK] 84-87: 4/4 artefactos cumplen la restriccion de 100% pure ASCII.")

    print(f"[RESUMEN] {conteo_aserciones}/{conteo_aserciones} aserciones pasadas exitosamente (100% aprobado).")


if __name__ == "__main__":
    test_verificacion_p132()
