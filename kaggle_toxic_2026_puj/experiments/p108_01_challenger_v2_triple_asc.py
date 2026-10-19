#!/usr/bin/env python
"""P108-01: Generacion y validacion formal del challenger V2_triple + Regla Ascendente.

Pre-registro P108:
Pregunta: Confirma la combinacion V2_triple (word(1,2) + char_wb(3,5) + char(1,6))
con la regla ontologica ascendente (TR=1 => HS=1) un beneficio robusto sobre V03
en ambas particiones de validacion cruzada y produce una submission valida para Kaggle?

Componentes:
  - Representacion V2_triple:
      * word(1,2), min_df=2, sublinear_tf=True, lowercase=True
      * char_wb(3,5), min_df=2, sublinear_tf=True, lowercase=True
      * char(1,6), min_df=2, sublinear_tf=True, lowercase=True
  - Modelo base: LogisticRegression C=2.0, class_weight='balanced', solver='liblinear'
  - Umbrales congelados de V03: HS=0.44, TR=0.51, AG=0.46
  - Regla ontologica: Si pred_TR == 1 => pred_HS = 1
  - Fit completo: Entrenado sobre las 1.731 filas de toxic_train.csv
  - Submission: toxic_submission_p108_v2_triple_asc.csv
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta
from src.splits import pliegues_corregidos

OOF_FINAL = RAIZ / "features" / "oof_v03_final.npz"
OOF_BASE = RAIZ / "features" / "v2_oof_base_corrected.npz"
IDS_PATH = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
SUB_OFICIAL = RAIZ / "submissions" / "toxic_submission_v03_lr_union_thr.csv"
SUBMISSION_TEMPLATE = RAIZ / "data" / "raw" / "submission.csv"

SAL_SUB = RAIZ / "submissions" / "toxic_submission_p108_v2_triple_asc.csv"
SAL_JSON = RAIZ / "reports" / "p108_01_challenger_v2_triple_asc.json"
SAL_MD = RAIZ / "reports" / "p108_01_challenger_v2_triple_asc.md"

V03_CCV = 0.7772114525165721
UMBRALES_V03 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
SEMILLA_BOOT = 20260925

LR_PARAMS = {
    "C": 2.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "liblinear",
    "random_state": 42,
}

BLOQUES_V03 = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
]

BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]


def log(mensaje: str) -> None:
    print(f"[P108-01] {mensaje}", flush=True)


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def construir_matrices(textos_tr, textos_va, bloques):
    mats_tr, mats_va = [], []
    for analyzer, ngram, lower in bloques:
        v = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram,
            min_df=2,
            sublinear_tf=True,
            lowercase=lower,
        )
        mats_tr.append(v.fit_transform(textos_tr))
        mats_va.append(v.transform(textos_va))
    return sparse.hstack(mats_tr).tocsr(), sparse.hstack(mats_va).tocsr()


def oof_proba_pliegues(textos, y_mat, pliegues_arr, bloques):
    proba = np.zeros((len(textos), len(ETIQUETAS)), dtype=float)
    for fold in sorted(set(pliegues_arr)):
        idx_tr = np.flatnonzero(pliegues_arr != fold)
        idx_va = np.flatnonzero(pliegues_arr == fold)
        m_tr, m_va = construir_matrices(textos.iloc[idx_tr], textos.iloc[idx_va], bloques)
        for j, etiqueta in enumerate(ETIQUETAS):
            lr = LogisticRegression(**LR_PARAMS)
            lr.fit(m_tr, y_mat[idx_tr, j])
            proba[idx_va, j] = lr.predict_proba(m_va)[:, 1]
    return proba


def pred_umbrales(proba: np.ndarray, umbrales: dict = UMBRALES_V03) -> np.ndarray:
    return np.column_stack([
        (proba[:, j] >= umbrales[etiqueta]).astype(int)
        for j, etiqueta in enumerate(ETIQUETAS)
    ])


def aplicar_regla_ascendente(pred: np.ndarray) -> tuple[np.ndarray, int]:
    p = pred.copy()
    mask = (p[:, 1] == 1) & (p[:, 0] == 0)
    p[mask, 0] = 1
    return p, int(mask.sum())


def bootstrap_clusters_delta(y, pred_a, pred_b, grupos, repeticiones=2000):
    generador = np.random.default_rng(SEMILLA_BOOT)
    indices = {int(g): np.flatnonzero(grupos == g) for g in np.unique(grupos)}
    claves = np.array(sorted(indices))
    dif = np.empty(repeticiones)
    for r in range(repeticiones):
        muestra = generador.integers(0, len(claves), len(claves))
        idx = np.concatenate([indices[int(claves[i])] for i in muestra])
        f1_a = metricas_multietiqueta(y[idx], pred_a[idx])["f1_media_columnas"]
        f1_b = metricas_multietiqueta(y[idx], pred_b[idx])["f1_media_columnas"]
        dif[r] = f1_a - f1_b
    c = np.quantile(dif, [0.025, 0.975])
    return {
        "ic95": [round(float(c[0]), 6), round(float(c[1]), 6)],
        "pct_mejora": round(float((dif > 0).mean()), 6),
    }


def main() -> None:
    t_inicio = time.perf_counter()
    log("Iniciando P108-01: validacion y generacion del challenger V2_triple_asc")

    # 1. Cargar datos de desarrollo (1.483 filas)
    zf = np.load(OOF_FINAL, allow_pickle=True)
    zb = np.load(OOF_BASE, allow_pickle=True)
    y_dev = zf["y_true"].astype(int)
    pred_v03_dev = zf["pred"].astype(int)
    pliegues_can = zf["pliegues"].astype(int)
    grupos = zb["grupos"].astype(np.int64)

    train = data_module.read_train()
    particion = pd.read_csv(IDS_PATH)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    train_dev = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    textos_dev = train_dev["text"]

    # 2. Control de reproduccion M2 en dev
    log("Verificando control de reproduccion M2 (V03) en particion canonica...")
    proba_v03_can = oof_proba_pliegues(textos_dev, y_dev, pliegues_can, BLOQUES_V03)
    pred_v03_can = pred_umbrales(proba_v03_can)
    m_v03_can = metricas_multietiqueta(y_dev, pred_v03_can)
    diff_v03_can = float(np.max(np.abs(pred_v03_can - pred_v03_dev)))
    log(f"Control canonico CCV: {m_v03_can['f1_media_columnas']:.6f} (diff max: {diff_v03_can})")
    if diff_v03_can != 0.0 or abs(m_v03_can["f1_media_columnas"] - V03_CCV) > 1e-9:
        raise SystemExit("Fallo control canonico M2 V03")

    # 3. Evaluacion V2_triple + Regla Ascendente en Canonica
    log("Ajustando V2_triple en particion canonica (5 pliegues)...")
    proba_v2_can = oof_proba_pliegues(textos_dev, y_dev, pliegues_can, BLOQUES_V2_TRIPLE)
    pred_v2_can = pred_umbrales(proba_v2_can)
    pred_v2_asc_can, n_asc_can = aplicar_regla_ascendente(pred_v2_can)

    m_v2_asc_can = metricas_multietiqueta(y_dev, pred_v2_asc_can)
    delta_can = m_v2_asc_can["f1_media_columnas"] - m_v03_can["f1_media_columnas"]
    celdas_can = int(np.sum(pred_v2_asc_can != pred_v03_dev))

    folds_can = sorted(set(pliegues_can))
    f1_f_v03 = [metricas_multietiqueta(y_dev[pliegues_can == f], pred_v03_dev[pliegues_can == f])["f1_media_columnas"] for f in folds_can]
    f1_f_v2 = [metricas_multietiqueta(y_dev[pliegues_can == f], pred_v2_asc_can[pliegues_can == f])["f1_media_columnas"] for f in folds_can]
    folds_ganados_can = sum(1 for a, b in zip(f1_f_v2, f1_f_v03) if a > b)
    boot_can = bootstrap_clusters_delta(y_dev, pred_v2_asc_can, pred_v03_dev, grupos)

    log(f"Canonica V2_triple+Asc: F1={m_v2_asc_can['f1_media_columnas']:.6f} | Delta={delta_can:+.6f} | Ganados={folds_ganados_can}/5 | Celdas={celdas_can} | PctBoot={boot_can['pct_mejora']*100:.1f}%")

    # 4. Evaluacion en Particion 47
    log("Evaluando en Particion 47 (semilla 43, 5 pliegues)...")
    combinacion = np.array([f"{a}{b}{c}" for a, b, c in y_dev])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
    pliegues_47 = np.full(len(y_dev), -1, dtype=int)
    for num, (_, val) in enumerate(sgkf.split(np.arange(len(y_dev)).reshape(-1, 1), combinacion, groups=grupos)):
        pliegues_47[val] = num + 1

    proba_v03_47 = oof_proba_pliegues(textos_dev, y_dev, pliegues_47, BLOQUES_V03)
    pred_v03_47 = pred_umbrales(proba_v03_47)
    m_v03_47 = metricas_multietiqueta(y_dev, pred_v03_47)

    proba_v2_47 = oof_proba_pliegues(textos_dev, y_dev, pliegues_47, BLOQUES_V2_TRIPLE)
    pred_v2_47 = pred_umbrales(proba_v2_47)
    pred_v2_asc_47, n_asc_47 = aplicar_regla_ascendente(pred_v2_47)
    m_v2_asc_47 = metricas_multietiqueta(y_dev, pred_v2_asc_47)
    delta_47 = m_v2_asc_47["f1_media_columnas"] - m_v03_47["f1_media_columnas"]
    celdas_47 = int(np.sum(pred_v2_asc_47 != pred_v03_47))

    f1_f_v03_47 = [metricas_multietiqueta(y_dev[pliegues_47 == f], pred_v03_47[pliegues_47 == f])["f1_media_columnas"] for f in sorted(set(pliegues_47))]
    f1_f_v2_47 = [metricas_multietiqueta(y_dev[pliegues_47 == f], pred_v2_asc_47[pliegues_47 == f])["f1_media_columnas"] for f in sorted(set(pliegues_47))]
    folds_ganados_47 = sum(1 for a, b in zip(f1_f_v2_47, f1_f_v03_47) if a > b)
    boot_47 = bootstrap_clusters_delta(y_dev, pred_v2_asc_47, pred_v03_47, grupos)

    log(f"Particion 47 V2_triple+Asc: F1={m_v2_asc_47['f1_media_columnas']:.6f} | Delta={delta_47:+.6f} | Ganados={folds_ganados_47}/5 | Celdas={celdas_47} | PctBoot={boot_47['pct_mejora']*100:.1f}%")

    # 5. Entrenamiento sobre todo el conjunto de entrenamiento (1.731 filas) y prediccion sobre test (1.154 filas)
    log("Entrenando modelo completo V2_triple sobre todo train (1.731 filas)...")
    test = data_module.read_test()

    mats_tr, mats_te = [], []
    for analyzer, ngram, lower in BLOQUES_V2_TRIPLE:
        v = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram,
            min_df=2,
            sublinear_tf=True,
            lowercase=lower,
        )
        mats_tr.append(v.fit_transform(train["text"]))
        mats_te.append(v.transform(test["text"]))

    X_train_full = sparse.hstack(mats_tr).tocsr()
    X_test_full = sparse.hstack(mats_te).tocsr()
    log(f"Matrices completas: X_train={X_train_full.shape}, X_test={X_test_full.shape}")

    proba_test = np.zeros((len(test), len(ETIQUETAS)), dtype=float)
    for j, etiqueta in enumerate(ETIQUETAS):
        lr = LogisticRegression(**LR_PARAMS)
        lr.fit(X_train_full, train[etiqueta])
        proba_test[:, j] = lr.predict_proba(X_test_full)[:, 1]

    pred_test_raw = pred_umbrales(proba_test)
    pred_test_asc, violaciones_corregidas = aplicar_regla_ascendente(pred_test_raw)
    log(f"Predicciones en test generadas. Violaciones TR=1 & HS=0 corregidas: {violaciones_corregidas}")

    # 6. Validar y generar archivo de submission
    sub_df = pd.DataFrame({
        "id": test["id"].astype(int),
        "HS": pred_test_asc[:, 0],
        "TR": pred_test_asc[:, 1],
        "AG": pred_test_asc[:, 2],
    })

    # Verificacion estricta de formato contra plantilla
    template = pd.read_csv(SUBMISSION_TEMPLATE)
    if list(sub_df.columns) != list(template.columns):
        raise SystemExit(f"Columnas no coinciden: {sub_df.columns} vs {template.columns}")
    if len(sub_df) != len(template):
        raise SystemExit(f"Filas no coinciden: {len(sub_df)} vs {len(template)}")
    if not (sub_df["id"].to_numpy() == template["id"].to_numpy()).all():
        raise SystemExit("IDs no coinciden con la plantilla oficial")

    sub_df.to_csv(SAL_SUB, index=False)
    sha_sub = sha256_archivo(SAL_SUB)
    log(f"Submission guardada exitosamente en: {SAL_SUB} (sha256: {sha_sub})")

    # Comparacion contra V03 en test
    sub_v03 = pd.read_csv(SUB_OFICIAL)
    v03_test_arr = sub_v03[list(ETIQUETAS)].to_numpy()
    celdas_dist_test = int(np.sum(pred_test_asc != v03_test_arr))
    diferencias_col = {e: int(np.sum(pred_test_asc[:, j] != v03_test_arr[:, j])) for j, e in enumerate(ETIQUETAS)}
    pos_v03 = {e: int(v03_test_arr[:, j].sum()) for j, e in enumerate(ETIQUETAS)}
    pos_v2 = {e: int(pred_test_asc[:, j].sum()) for j, e in enumerate(ETIQUETAS)}

    log(f"Diferencias vs V03 en test: {celdas_dist_test} celdas ({celdas_dist_test/v03_test_arr.size*100:.2f}%) | Por col: {diferencias_col}")
    log(f"Positivos V03: {pos_v03} | Positivos V2_triple_asc: {pos_v2}")

    t_total = time.perf_counter() - t_inicio

    # 7. Guardar reporte JSON y MD
    reporte = {
        "fase": "P108",
        "experimento": "P108-01",
        "descripcion": "Challenger V2_triple (word(1,2) + char_wb(3,5) + char(1,6)) con regla ascendente",
        "duracion_segundos": round(t_total, 2),
        "submission_archivo": str(SAL_SUB),
        "submission_sha256": sha_sub,
        "control_canonica": {
            "ccv": round(m_v03_can["f1_media_columnas"], 6),
            "diff_max": diff_v03_can,
            "ok": True,
        },
        "metricas_dev_canonica": {
            "v03_ccv": round(m_v03_can["f1_media_columnas"], 6),
            "v2_triple_asc_ccv": round(m_v2_asc_can["f1_media_columnas"], 6),
            "delta": round(delta_can, 6),
            "pliegues_ganados": folds_ganados_can,
            "pliegues_f1_v03": [round(x, 6) for x in f1_f_v03],
            "pliegues_f1_v2": [round(x, 6) for x in f1_f_v2],
            "celdas_cambiadas": celdas_can,
            "bootstrap_ic95": boot_can["ic95"],
            "bootstrap_pct_mejora": boot_can["pct_mejora"],
        },
        "metricas_dev_part47": {
            "v03_ccv": round(m_v03_47["f1_media_columnas"], 6),
            "v2_triple_asc_ccv": round(m_v2_asc_47["f1_media_columnas"], 6),
            "delta": round(delta_47, 6),
            "pliegues_ganados": folds_ganados_47,
            "pliegues_f1_v03": [round(x, 6) for x in f1_f_v03_47],
            "pliegues_f1_v2": [round(x, 6) for x in f1_f_v2_47],
            "celdas_cambiadas": celdas_47,
            "bootstrap_ic95": boot_47["ic95"],
            "bootstrap_pct_mejora": boot_47["pct_mejora"],
        },
        "comparacion_test_vs_v03": {
            "total_celdas_distintas": celdas_dist_test,
            "pct_celdas_distintas": round(celdas_dist_test / v03_test_arr.size * 100, 2),
            "distribucion_diferencias": diferencias_col,
            "violaciones_corregidas_test": violaciones_corregidas,
            "positivos_v03": pos_v03,
            "positivos_v2_asc": pos_v2,
        },
    }

    SAL_JSON.write_text(json.dumps(reporte, indent=2, ensure_ascii=True), encoding="utf-8")

    md_content = f"""# P108-01: Challenger V2_triple + Regla Ascendente

**Fecha:** 2026-10-14  
**Archivo Submission:** `{SAL_SUB.name}`  
**SHA256:** `{sha_sub}`  
**Duracion ejecucion:** {t_total:.1f} s  

## 1. Resumen de Desempeno en Validacion Cruzada (10 Pliegues)

| Particion | V03 CCV | Challenger CCV | Delta | Pliegues Ganados | Bootstrap IC95 | Confianza Mejora |
|---|---|---|---|---|---|---|
| **Canonica (1.483 filas)** | {m_v03_can['f1_media_columnas']:.6f} | **{m_v2_asc_can['f1_media_columnas']:.6f}** | **{delta_can:+.6f}** | **{folds_ganados_can}/5** | {boot_can['ic95']} | **{boot_can['pct_mejora']*100:.1f}%** |
| **Particion 47 (1.483 filas)** | {m_v03_47['f1_media_columnas']:.6f} | **{m_v2_asc_47['f1_media_columnas']:.6f}** | **{delta_47:+.6f}** | **{folds_ganados_47}/5** | {boot_47['ic95']} | **{boot_47['pct_mejora']*100:.1f}%** |

## 2. Comparacion en el Conjunto de Test (1.154 filas)

- **Total celdas distintas vs V03:** {celdas_dist_test} de 3.462 ({celdas_dist_test/v03_test_arr.size*100:.2f}%)
- **Diferencias por etiqueta:** HS={diferencias_col['HS']}, TR={diferencias_col['TR']}, AG={diferencias_col['AG']}
- **Violaciones ontologicas TR=1 & HS=0 corregidas:** {violaciones_corregidas}
- **Recuento de positivos predichos:**
  * V03: HS={pos_v03['HS']}, TR={pos_v03['TR']}, AG={pos_v03['AG']}
  * Challenger: HS={pos_v2['HS']}, TR={pos_v2['TR']}, AG={pos_v2['AG']}

## 3. Conformidad con Reglas
- Cero transformers / LLMs.
- Cero embeddings preentrenados.
- Entrenamiento exclusivo desde cero con las 1.731 filas de `toxic_train.csv`.
- Cero fuga hacia el conjunto de test.
- Verificacion de formato de submission 100% identica a `submission.csv`.
"""
    SAL_MD.write_text(md_content, encoding="utf-8")
    log("Reportes guardados exitosamente.")
    log(f"P108-01 completado en {t_total:.1f} s.")


if __name__ == "__main__":
    main()
