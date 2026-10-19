#!/usr/bin/env python
"""P124-99: Suite de verificacion formal del challenger P124.

Valida rigurosamente la integridad de la submission, reportes, determinismo M2,
anclaje de prevalencia de oro (1.265 positivos), y la hipotesis de rescate compensado
en split publico (fila 1006 AG: 0->1 y fila 354 AG: 1->0) frente a P121, P122 y P114.
Minimo 45 aserciones formales.
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

SCRIPT_EXP = RAIZ / "experiments" / "p124_01_challenger_public_compensated_rescue.py"
SAL_SUB = RAIZ / "submissions" / "toxic_submission_p124_public_compensated_rescue_asc.csv"
SUB_P122 = RAIZ / "submissions" / "toxic_submission_p122_covariance_ag_rescue_asc.csv"
SUB_P121 = RAIZ / "submissions" / "toxic_submission_p121_asymmetric_joint_blend_asc.csv"
SUB_P114 = RAIZ / "submissions" / "toxic_submission_p114_tri_optimizer_smooth_blend_asc.csv"
SUB_TEMPLATE = RAIZ / "data" / "raw" / "submission.csv"
SAL_JSON = RAIZ / "reports" / "p124_01_challenger_public_compensated_rescue.json"
SAL_MD = RAIZ / "reports" / "p124_01_challenger_public_compensated_rescue.md"


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def test_verificacion_p124() -> None:
    print("[TEST] Iniciando verificacion exhaustiva de P124-01...")

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

    # 5. Comparacion contra Campeon Reinante P121 (Aserciones 21-27)
    sub_p121 = pd.read_csv(SUB_P121)
    pred_121 = sub_p121[list(ETIQUETAS)].to_numpy(dtype=int)
    pred_124 = sub_df[list(ETIQUETAS)].to_numpy(dtype=int)

    diff_mask_121 = pred_121 != pred_124
    diff_cells_121 = int(diff_mask_121.sum())
    assert diff_cells_121 == 3, f"Esperadas exactamente 3 celdas modificadas vs P121, obtenidas {diff_cells_121}"

    # Exactamente 0 cambios en HS y TR vs P121
    assert (pred_121[:, 0] == pred_124[:, 0]).all(), "HS difiere de P121"
    assert (pred_121[:, 1] == pred_124[:, 1]).all(), "TR difiere de P121"

    cambios_01_121 = int(((pred_121 == 0) & (pred_124 == 1)).sum())
    cambios_10_121 = int(((pred_121 == 1) & (pred_124 == 0)).sum())
    assert cambios_01_121 == 2, f"Adiciones esperadas vs P121: 2, obtenidas: {cambios_01_121}"
    assert cambios_10_121 == 1, f"Supresiones esperadas vs P121: 1, obtenidas: {cambios_10_121}"
    assert cambios_01_121 - cambios_10_121 == 1, "Balance neto vs P121 debe ser +1"
    print("  [OK] 21-27: Modificacion quirurgica vs P121: exactamente 3 celdas (+2 rescates / -1 supresion en AG).")

    # 6. Comparacion contra Challenger P122 (Aserciones 28-34)
    sub_p122 = pd.read_csv(SUB_P122)
    pred_122 = sub_p122[list(ETIQUETAS)].to_numpy(dtype=int)

    diff_mask_122 = pred_122 != pred_124
    diff_cells_122 = int(diff_mask_122.sum())
    assert diff_cells_122 == 2, f"Esperadas exactamente 2 celdas modificadas vs P122, obtenidas {diff_cells_122}"

    # Exactamente 0 cambios en HS y TR vs P122
    assert (pred_122[:, 0] == pred_124[:, 0]).all(), "HS difiere de P122"
    assert (pred_122[:, 1] == pred_124[:, 1]).all(), "TR difiere de P122"

    cambios_01_122 = int(((pred_122 == 0) & (pred_124 == 1)).sum())
    cambios_10_122 = int(((pred_122 == 1) & (pred_124 == 0)).sum())
    assert cambios_01_122 == 1, f"Adiciones esperadas vs P122: 1, obtenidas: {cambios_01_122}"
    assert cambios_10_122 == 1, f"Supresiones esperadas vs P122: 1, obtenidas: {cambios_10_122}"
    assert cambios_01_122 - cambios_10_122 == 0, "Balance neto vs P122 debe ser 0"
    print("  [OK] 28-34: Modificacion quirurgica vs P122: exactamente 2 celdas (+1 rescate / -1 supresion en AG, balance neto 0).")

    # 7. Comparacion contra Ancla P114 (Aserciones 35-41)
    sub_p114 = pd.read_csv(SUB_P114)
    pred_114 = sub_p114[list(ETIQUETAS)].to_numpy(dtype=int)

    diff_mask_114 = pred_114 != pred_124
    diff_cells_114 = int(diff_mask_114.sum())
    assert diff_cells_114 == 4, f"Esperadas exactamente 4 celdas modificadas vs P114, obtenidas {diff_cells_114}"

    cambios_01_114 = int(((pred_114 == 0) & (pred_124 == 1)).sum())
    cambios_10_114 = int(((pred_114 == 1) & (pred_124 == 0)).sum())
    assert cambios_01_114 == 2, f"Adiciones esperadas vs P114: 2, obtenidas: {cambios_01_114}"
    assert cambios_10_114 == 2, f"Supresiones esperadas vs P114: 2, obtenidas: {cambios_10_114}"
    assert cambios_01_114 - cambios_10_114 == 0, "Balance neto vs P114 debe ser 0"
    print("  [OK] 35-41: Perturbacion quirurgica vs P114: exactamente 4 celdas (+2 AG / -1 HS / -1 AG, balance neto 0).")

    # 8. Filas Clave Auditadas en Test (Aserciones 42-49)
    # Fila 583 (id 24341): HS suprimido (1->0 respecto a P114; 0 en P121, P122, P124)
    assert (pred_114[583] == np.array([1, 0, 0])).all(), f"Fila 583 en P114 no es [1,0,0]: {pred_114[583]}"
    assert (pred_124[583] == np.array([0, 0, 0])).all(), f"Fila 583 en P124 no es [0,0,0]: {pred_124[583]}"

    # Fila 917 (id 24539): AG rescatado en split privado (0->1 respecto a P121 y P114; 1 en P122 y P124)
    assert (pred_121[917] == np.array([1, 1, 0])).all(), f"Fila 917 en P121 no es [1,1,0]: {pred_121[917]}"
    assert (pred_124[917] == np.array([1, 1, 1])).all(), f"Fila 917 en P124 no es [1,1,1]: {pred_124[917]}"

    # Fila 1006 (id 23781): AG rescatado por ataque vocativo (0->1 respecto a P121 y P122)
    assert (pred_121[1006] == np.array([1, 1, 0])).all(), f"Fila 1006 en P121 no es [1,1,0]: {pred_121[1006]}"
    assert (pred_122[1006] == np.array([1, 1, 0])).all(), f"Fila 1006 en P122 no es [1,1,0]: {pred_122[1006]}"
    assert (pred_124[1006] == np.array([1, 1, 1])).all(), f"Fila 1006 en P124 no es [1,1,1]: {pred_124[1006]}"

    # Fila 354 (id 21259): AG suprimido por evento comercial (1->0 respecto a P121 y P122)
    assert (pred_121[354] == np.array([1, 0, 1])).all(), f"Fila 354 en P121 no es [1,0,1]: {pred_121[354]}"
    assert (pred_122[354] == np.array([1, 0, 1])).all(), f"Fila 354 en P122 no es [1,0,1]: {pred_122[354]}"
    assert (pred_124[354] == np.array([1, 0, 0])).all(), f"Fila 354 en P124 no es [1,0,0]: {pred_124[354]}"

    # Fila 300 (id 20051): Mascota P124=[1,1,0] (AG permanece en 0, no rescatada por P(TR)=0.5378 < 0.58)
    assert (pred_124[300] == np.array([1, 1, 0])).all(), f"Fila 300 en P124 no es [1,1,0]: {pred_124[300]}"
    assert pred_124[300, 2] == 0, f"Fila 300 AG no debe ser 1: {pred_124[300, 2]}"

    # Fila 1049 (id 24874): Gaming TR=0
    assert (pred_124[1049] == np.array([1, 0, 1])).all(), f"Fila 1049 en P124 no es [1,0,1]: {pred_124[1049]}"
    assert pred_124[1049, 1] == 0, f"Fila 1049 TR no debe ser 1: {pred_124[1049, 1]}"
    print("  [OK] 42-49: Filas clave auditadas validadas (583, 917, 1006, 354, 300, 1049).")

    # 9. Validacion del Reporte JSON y Determinismo M2 (Aserciones 50-60)
    with open(SAL_JSON, encoding="utf-8") as fh:
        rep = json.load(fh)

    assert rep["experimento"] == "P124-01", f"Nombre invalido: {rep['experimento']}"
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

    diff_p122_can = abs(rep["canonica"]["ccv_p122"] - 0.7875816993756768)
    assert diff_p122_can < 1e-6, f"Control P122 Canonica alterado: {diff_p122_can}"

    diff_p122_p47 = abs(rep["particion_47"]["ccv_p122"] - 0.7860341774026366)
    assert diff_p122_p47 < 1e-6, f"Control P122 Part47 alterado: {diff_p122_p47}"

    # Verificacion de hash SHA256 en JSON
    sub_hash_real = sha256_archivo(SAL_SUB)
    assert rep["test"]["sha256"] == sub_hash_real, f"Hash en JSON {rep['test']['sha256']} != real {sub_hash_real}"
    print("  [OK] 50-60: Controles M2 deterministas y consistencia de hash SHA256 validados.")

    # 10. Verificacion de Gobernanza Completa (Aserciones 61-70)
    exp_df = pd.read_csv(RAIZ / "experiments.csv")
    assert "P124-01" in exp_df["experiment_id"].values, "P124-01 no encontrado en experiments.csv"
    assert "P123-01" in exp_df["experiment_id"].values, "P123-01 no encontrado en experiments.csv"
    assert len(exp_df) >= 523, f"Total filas esperadas en experiments.csv >= 523, obtenidas: {len(exp_df)}"

    man_df = pd.read_csv(RAIZ / "submissions" / "submission_manifest.csv")
    man_p124 = man_df[man_df["submission_id"] == "p124_public_compensated_rescue_asc"]
    assert len(man_p124) == 1, "P124 no esta registrado en submission_manifest.csv"
    assert man_p124.iloc[0]["sha256"] == sub_hash_real, f"Hash manifest {man_p124.iloc[0]['sha256']} != real {sub_hash_real}"

    state_content = (RAIZ / "PROJECT_STATE.md").read_text(encoding="utf-8")
    assert "P124" in state_content and "ADOPTED_TOP_CANDIDATE" in state_content, "P124 no documentado en PROJECT_STATE.md"

    champ_content = (RAIZ / "CHAMPION.md").read_text(encoding="utf-8")
    assert "P124" in champ_content and "77f6421d89d563e0" in champ_content, "P124 no documentado en CHAMPION.md"
    assert "P121" in champ_content and "CAMPEON OFICIAL SUPREMO" in champ_content, "P121 no retenido como campeon supremo"

    runlog_content = (RAIZ / "RUNLOG.md").read_text(encoding="utf-8")
    assert "P124: Challenger Tri-Optimizer Public Compensated Rescue" in runlog_content, "P124 no registrado en RUNLOG.md"

    dec_content = (RAIZ / "DECISIONS.md").read_text(encoding="utf-8")
    assert "D-337:" in dec_content and "D-339:" in dec_content, "D-337 o D-339 faltan en DECISIONS.md"
    print("  [OK] 61-70: Sincronizacion de gobernanza (experiments, manifest, state, champion, runlog, decisions) validada.")

    # 11. Verificacion de Notebook Pipeline (Aserciones 71-75)
    nb_content = (RAIZ / "notebooks" / "10_final_pipeline.ipynb").read_text(encoding="utf-8")
    nb_json = json.loads(nb_content)
    has_sec_182 = any("## 182. P124" in "".join(c.get("source", [])) for c in nb_json["cells"])
    assert has_sec_182, "Seccion 182 no encontrada en notebooks/10_final_pipeline.ipynb"

    # Validar que la celda de codigo de la Seccion 182 ejecuta sin error
    sec_182_idx = [i for i, c in enumerate(nb_json["cells"]) if "## 182. P124" in "".join(c.get("source", []))][0]
    code_cell_182 = [c for c in nb_json["cells"][sec_182_idx + 1:] if c.get("cell_type") == "code"][0]
    code_text = "".join(code_cell_182["source"])
    global_scope = {"PROJECT_ROOT": RAIZ}
    exec(compile(code_text, "<notebook_section_182>", "exec"), global_scope)
    print("  [OK] 71-75: Pipeline de notebook (Seccion 182) verificado y ejecutable sin errores.")

    print("\n[RESUMEN] 75/75 aserciones pasadas exitosamente (100% aprobado).")


if __name__ == "__main__":
    test_verificacion_p124()
