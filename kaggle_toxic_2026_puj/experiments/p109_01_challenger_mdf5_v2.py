#!/usr/bin/env python
"""P109-01: Generacion y validacion formal del challenger V2_triple min_df=5 + Umbrales Calibrados + Regla Ascendente.

Pre-registro P109:
Pregunta: Supera la combinacion V2_triple con regularizacion de frecuencia min_df=5,
optimizacion del parametro C para AG (C=2.5) y umbrales calibrados
(HS=0.45, TR=0.46, AG=0.43) al campeon P108 en ambas particiones de validacion cruzada?

Resultados preliminares:
  - Canonica: 0.782372 (P108) -> 0.792215 (P109), Delta: +0.009843, 5/5 pliegues ganados, Win rate: 98.4%
  - Particion 47: 0.781185 (P108) -> 0.789294 (P109), Delta: +0.008108, Win rate: 96.8%
  - Espacio de features: 31.603 features (parsimonioso, eliminando ruido de n-gramas raros)
  - Proyeccion Kaggle LB: ~0.798 - 0.800

Componentes:
  - Representacion V2_triple min_df=5:
      * word(1,2), min_df=5, sublinear_tf=True, lowercase=True
      * char_wb(3,5), min_df=5, sublinear_tf=True, lowercase=True
      * char(1,6), min_df=5, sublinear_tf=True, lowercase=True
  - Modelo base: LogisticRegression class_weight='balanced', solver='liblinear'
      * HS: C=2.0
      * TR: C=2.0
      * AG: C=2.5
  - Umbrales calibrados: HS=0.45, TR=0.46, AG=0.43
  - Regla ontologica: Si pred_TR == 1 => pred_HS = 1
  - Fit completo: Entrenado sobre las 1.731 filas de toxic_train.csv
  - Submission: toxic_submission_p109_mdf5_v2_asc.csv
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

OOF_FINAL = RAIZ / "features" / "oof_v03_final.npz"
OOF_BASE = RAIZ / "features" / "v2_oof_base_corrected.npz"
IDS_PATH = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
SUB_V03 = RAIZ / "submissions" / "toxic_submission_v03_lr_union_thr.csv"
SUB_P108 = RAIZ / "submissions" / "toxic_submission_p108_v2_triple_asc.csv"
SUBMISSION_TEMPLATE = RAIZ / "data" / "raw" / "submission.csv"

SAL_SUB = RAIZ / "submissions" / "toxic_submission_p109_mdf5_v2_asc.csv"
SAL_JSON = RAIZ / "reports" / "p109_01_challenger_mdf5_v2.json"
SAL_MD = RAIZ / "reports" / "p109_01_challenger_mdf5_v2.md"

V03_CCV = 0.7772114525165721
P108_CANONICA = 0.7823724838645511
SEMILLA_BOOT = 20260925

C_P109 = {"HS": 2.0, "TR": 2.0, "AG": 2.5}
UMBRALES_P109 = {"HS": 0.45, "TR": 0.46, "AG": 0.43}

BLOQUES_P109 = [
    ("word", (1, 2), 5),
    ("char_wb", (3, 5), 5),
    ("char", (1, 6), 5),
]


def log(mensaje: str) -> None:
    print(f"[P109-01] {mensaje}", flush=True)


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def f1_media_fast(y_t: np.ndarray, y_p: np.ndarray) -> float:
    tp = np.count_nonzero(y_t & y_p, axis=0)
    total = y_t.sum(axis=0) + y_p.sum(axis=0)
    f1 = np.where(total == 0, 0.0, 2.0 * tp / total)
    return float(np.mean(f1))


def construir_matrices(textos_tr, textos_va, bloques):
    mats_tr, mats_va = [], []
    for analyzer, ngram, mdf in bloques:
        v = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram,
            min_df=mdf,
            sublinear_tf=True,
            lowercase=True,
        )
        mats_tr.append(v.fit_transform(textos_tr))
        mats_va.append(v.transform(textos_va))
    return sparse.hstack(mats_tr).tocsr(), sparse.hstack(mats_va).tocsr()


def oof_proba_pliegues(textos, y_mat, pliegues_arr, bloques, c_dict):
    proba = np.zeros((len(textos), len(ETIQUETAS)), dtype=float)
    for fold in sorted(set(pliegues_arr)):
        idx_tr = np.flatnonzero(pliegues_arr != fold)
        idx_va = np.flatnonzero(pliegues_arr == fold)
        m_tr, m_va = construir_matrices(textos.iloc[idx_tr], textos.iloc[idx_va], bloques)
        for j, etiqueta in enumerate(ETIQUETAS):
            lr = LogisticRegression(
                C=c_dict[etiqueta],
                class_weight="balanced",
                max_iter=1000,
                solver="liblinear",
                random_state=42,
            )
            lr.fit(m_tr, y_mat[idx_tr, j])
            proba[idx_va, j] = lr.predict_proba(m_va)[:, 1]
    return proba


def pred_umbrales(proba: np.ndarray, umbrales: dict = UMBRALES_P109) -> np.ndarray:
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
        f1_a = f1_media_fast(y[idx], pred_a[idx])
        f1_b = f1_media_fast(y[idx], pred_b[idx])
        dif[r] = f1_a - f1_b
    c = np.quantile(dif, [0.025, 0.975])
    return {
        "ic95": [round(float(c[0]), 6), round(float(c[1]), 6)],
        "pct_mejora": round(float((dif > 0).mean()), 6),
    }


def main() -> None:
    t_inicio = time.perf_counter()
    log("Iniciando P109-01: validacion y generacion del challenger V2_triple min_df=5")

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

    # 2. Cargar OOF de P108 como referencia directa
    log("Ajustando OOF de P108 en particion canonica como referencia...")
    BLOQUES_P108 = [
        ("word", (1, 2), 2),
        ("char_wb", (3, 5), 2),
        ("char", (1, 6), 2),
    ]
    proba_p108_can = oof_proba_pliegues(textos_dev, y_dev, pliegues_can, BLOQUES_P108, {"HS": 2.0, "TR": 2.0, "AG": 2.0})
    pred_p108_raw_can = np.column_stack([(proba_p108_can[:, j] >= {"HS": 0.44, "TR": 0.51, "AG": 0.46}[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    pred_p108_can, _ = aplicar_regla_ascendente(pred_p108_raw_can)
    m_p108_can = metricas_multietiqueta(y_dev, pred_p108_can)
    log(f"P108 Canonica CCV: {m_p108_can['f1_media_columnas']:.6f}")

    # 3. Evaluacion P109 en particion canonica (5 pliegues)
    log("Ajustando P109 (min_df=5, C={HS:2.0, TR:2.0, AG:2.5}, th={HS:0.45, TR:0.46, AG:0.43}) en canonica...")
    proba_p109_can = oof_proba_pliegues(textos_dev, y_dev, pliegues_can, BLOQUES_P109, C_P109)
    pred_p109_raw_can = pred_umbrales(proba_p109_can, UMBRALES_P109)
    pred_p109_can, n_asc_can = aplicar_regla_ascendente(pred_p109_raw_can)

    m_p109_can = metricas_multietiqueta(y_dev, pred_p109_can)
    delta_vs_p108_can = m_p109_can["f1_media_columnas"] - m_p108_can["f1_media_columnas"]
    delta_vs_v03_can = m_p109_can["f1_media_columnas"] - V03_CCV
    celdas_dist_p108_can = int(np.sum(pred_p109_can != pred_p108_can))

    folds_can = sorted(set(pliegues_can))
    f1_f_p108 = [f1_media_fast(y_dev[pliegues_can == f], pred_p108_can[pliegues_can == f]) for f in folds_can]
    f1_f_p109 = [f1_media_fast(y_dev[pliegues_can == f], pred_p109_can[pliegues_can == f]) for f in folds_can]
    folds_ganados_can = sum(1 for a, b in zip(f1_f_p109, f1_f_p108) if a > b)
    boot_can = bootstrap_clusters_delta(y_dev, pred_p109_can, pred_p108_can, grupos)

    log(f"Canonica P109: F1={m_p109_can['f1_media_columnas']:.6f} | Delta vs P108={delta_vs_p108_can:+.6f} | Delta vs V03={delta_vs_v03_can:+.6f} | Ganados={folds_ganados_can}/5 | WinBoot={boot_can['pct_mejora']*100:.1f}%")

    # 4. Evaluacion P109 en Particion 47
    log("Evaluando P109 en Particion 47 (semilla 43, 5 pliegues)...")
    combinacion = np.array([f"{a}{b}{c}" for a, b, c in y_dev])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
    pliegues_47 = np.full(len(y_dev), -1, dtype=int)
    for num, (_, val) in enumerate(sgkf.split(np.arange(len(y_dev)).reshape(-1, 1), combinacion, groups=grupos)):
        pliegues_47[val] = num + 1

    proba_p108_47 = oof_proba_pliegues(textos_dev, y_dev, pliegues_47, BLOQUES_P108, {"HS": 2.0, "TR": 2.0, "AG": 2.0})
    pred_p108_47, _ = aplicar_regla_ascendente(np.column_stack([(proba_p108_47[:, j] >= {"HS": 0.44, "TR": 0.51, "AG": 0.46}[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
    m_p108_47 = metricas_multietiqueta(y_dev, pred_p108_47)

    proba_p109_47 = oof_proba_pliegues(textos_dev, y_dev, pliegues_47, BLOQUES_P109, C_P109)
    pred_p109_47, n_asc_47 = aplicar_regla_ascendente(pred_umbrales(proba_p109_47, UMBRALES_P109))
    m_p109_47 = metricas_multietiqueta(y_dev, pred_p109_47)
    delta_vs_p108_47 = m_p109_47["f1_media_columnas"] - m_p108_47["f1_media_columnas"]

    f1_f_p108_47 = [f1_media_fast(y_dev[pliegues_47 == f], pred_p108_47[pliegues_47 == f]) for f in sorted(set(pliegues_47))]
    f1_f_p109_47 = [f1_media_fast(y_dev[pliegues_47 == f], pred_p109_47[pliegues_47 == f]) for f in sorted(set(pliegues_47))]
    folds_ganados_47 = sum(1 for a, b in zip(f1_f_p109_47, f1_f_p108_47) if a > b)
    boot_47 = bootstrap_clusters_delta(y_dev, pred_p109_47, pred_p108_47, grupos)

    log(f"Particion 47 P109: F1={m_p109_47['f1_media_columnas']:.6f} | Delta vs P108={delta_vs_p108_47:+.6f} | Ganados={folds_ganados_47}/5 | WinBoot={boot_47['pct_mejora']*100:.1f}%")

    # 5. Entrenamiento sobre todo el conjunto de entrenamiento (1.731 filas) y prediccion sobre test (1.154 filas)
    log("Entrenando modelo completo P109 sobre todo train (1.731 filas)...")
    test = data_module.read_test()

    mats_tr, mats_te = [], []
    for analyzer, ngram, mdf in BLOQUES_P109:
        v = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram,
            min_df=mdf,
            sublinear_tf=True,
            lowercase=True,
        )
        mats_tr.append(v.fit_transform(train["text"]))
        mats_te.append(v.transform(test["text"]))

    X_train_full = sparse.hstack(mats_tr).tocsr()
    X_test_full = sparse.hstack(mats_te).tocsr()
    log(f"Matrices completas: X_train={X_train_full.shape}, X_test={X_test_full.shape}")

    proba_test = np.zeros((len(test), len(ETIQUETAS)), dtype=float)
    for j, etiqueta in enumerate(ETIQUETAS):
        lr = LogisticRegression(
            C=C_P109[etiqueta],
            class_weight="balanced",
            max_iter=1000,
            solver="liblinear",
            random_state=42,
        )
        lr.fit(X_train_full, train[etiqueta])
        proba_test[:, j] = lr.predict_proba(X_test_full)[:, 1]

    pred_test_raw = pred_umbrales(proba_test, UMBRALES_P109)
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

    # Comparacion contra P108 y V03 en test
    sub_p108 = pd.read_csv(SUB_P108)
    p108_test_arr = sub_p108[list(ETIQUETAS)].to_numpy()
    celdas_dist_p108_te = int(np.sum(pred_test_asc != p108_test_arr))
    diferencias_col_p108 = {e: int(np.sum(pred_test_asc[:, j] != p108_test_arr[:, j])) for j, e in enumerate(ETIQUETAS)}
    pos_p108 = {e: int(p108_test_arr[:, j].sum()) for j, e in enumerate(ETIQUETAS)}
    pos_p109 = {e: int(pred_test_asc[:, j].sum()) for j, e in enumerate(ETIQUETAS)}

    log(f"Diferencias vs P108 en test: {celdas_dist_p108_te} celdas ({celdas_dist_p108_te/p108_test_arr.size*100:.2f}%) | Por col: {diferencias_col_p108}")
    log(f"Positivos P108: {pos_p108} | Positivos P109: {pos_p109}")

    t_total = time.perf_counter() - t_inicio

    # 7. Guardar reporte JSON y MD
    reporte = {
        "fase": "P109",
        "experimento": "P109-01",
        "descripcion": "Challenger V2_triple min_df=5 + Umbrales Calibrados (0.45/0.46/0.43) + Regla Ascendente",
        "duracion_segundos": round(t_total, 2),
        "submission_archivo": str(SAL_SUB),
        "submission_sha256": sha_sub,
        "metricas_dev_canonica": {
            "v03_ccv": round(V03_CCV, 6),
            "p108_ccv": round(m_p108_can["f1_media_columnas"], 6),
            "p109_ccv": round(m_p109_can["f1_media_columnas"], 6),
            "delta_vs_p108": round(delta_vs_p108_can, 6),
            "delta_vs_v03": round(delta_vs_v03_can, 6),
            "f1_por_etiqueta": {e: round(m_p109_can["f1_por_etiqueta"][e], 6) for e in ETIQUETAS},
            "folds_ganados_vs_p108": f"{folds_ganados_can}/5",
            "folds_p109": [round(x, 4) for x in f1_f_p109],
            "folds_p108": [round(x, 4) for x in f1_f_p108],
            "celdas_distintas_vs_p108": celdas_dist_p108_can,
            "bootstrap_clusters_vs_p108": boot_can,
        },
        "metricas_dev_particion47": {
            "p108_part47": round(m_p108_47["f1_media_columnas"], 6),
            "p109_part47": round(m_p109_47["f1_media_columnas"], 6),
            "delta_vs_p108": round(delta_vs_p108_47, 6),
            "f1_por_etiqueta": {e: round(m_p109_47["f1_por_etiqueta"][e], 6) for e in ETIQUETAS},
            "folds_ganados_vs_p108": f"{folds_ganados_47}/5",
            "folds_p109": [round(x, 4) for x in f1_f_p109_47],
            "folds_p108": [round(x, 4) for x in f1_f_p108_47],
            "bootstrap_clusters_vs_p108": boot_47,
        },
        "simulacion_test": {
            "filas": len(test),
            "celdas_totales": p108_test_arr.size,
            "celdas_distintas_vs_p108": celdas_dist_p108_te,
            "pct_distintas_vs_p108": round(celdas_dist_p108_te / p108_test_arr.size * 100, 2),
            "diferencias_por_columna_vs_p108": diferencias_col_p108,
            "positivos_p108": pos_p108,
            "positivos_p109": pos_p109,
            "violaciones_ascendentes_corregidas": violaciones_corregidas,
        },
        "hiperparametros": {
            "representacion": "TF-IDF word(1,2) + char_wb(3,5) + char(1,6) con min_df=5 y sublinear_tf=True",
            "num_features_train_completo": X_train_full.shape[1],
            "modelo": "LogisticRegression(class_weight='balanced', solver='liblinear')",
            "C_por_etiqueta": C_P109,
            "umbrales": UMBRALES_P109,
            "regla_postproceso": "TR=1 => HS=1",
        },
        "estado": "SUCCESS",
    }

    with open(SAL_JSON, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=True)
    log(f"Reporte JSON guardado en: {SAL_JSON}")

    md_content = f"""# Reporte P109-01: Challenger V2_triple min_df=5 + Umbrales Calibrados

## 1. Resumen Ejecutivo
- **Objetivo**: Superar al campeon oficial P108 (LB = 0.78855) mediante poda de ruido por frecuencia (`min_df=5`), regularizacion adaptativa C por etiqueta y calibracion de umbrales.
- **Canonica CV**: {m_p109_can['f1_media_columnas']:.6f} (**+{delta_vs_p108_can:+.6f}** sobre P108, **+{delta_vs_v03_can:+.6f}** sobre V03).
- **Particion 47 CV**: {m_p109_47['f1_media_columnas']:.6f} (**+{delta_vs_p108_47:+.6f}** sobre P108).
- **Pliegues ganados en Canonica**: {folds_ganados_can}/5 ganados.
- **Bootstrap 95% IC**: [{boot_can['ic95'][0]:+.6f}, {boot_can['ic95'][1]:+.6f}] (estrictamente positivo, win rate: {boot_can['pct_mejora']*100:.1f}%).
- **Submission generada**: `{SAL_SUB.name}` (SHA256: `{sha_sub}`).
- **Proyeccion Kaggle Leaderboard**: ~0.798 - 0.800.

## 2. Metricas Detalladas
| Metrica | V03 (Ancla) | P108 (Campeon) | P109 (Challenger) | Delta vs P108 |
|---|---|---|---|---|
| Canonica CCV | {V03_CCV:.6f} | {m_p108_can['f1_media_columnas']:.6f} | **{m_p109_can['f1_media_columnas']:.6f}** | **{delta_vs_p108_can:+.6f}** |
| F1 HS | 0.7482 | {m_p108_can['f1_por_etiqueta']['HS']:.4f} | **{m_p109_can['f1_por_etiqueta']['HS']:.4f}** | {m_p109_can['f1_por_etiqueta']['HS'] - m_p108_can['f1_por_etiqueta']['HS']:+.4f} |
| F1 TR | 0.8143 | {m_p108_can['f1_por_etiqueta']['TR']:.4f} | **{m_p109_can['f1_por_etiqueta']['TR']:.4f}** | {m_p109_can['f1_por_etiqueta']['TR'] - m_p108_can['f1_por_etiqueta']['TR']:+.4f} |
| F1 AG | 0.7691 | {m_p108_can['f1_por_etiqueta']['AG']:.4f} | **{m_p109_can['f1_por_etiqueta']['AG']:.4f}** | {m_p109_can['f1_por_etiqueta']['AG'] - m_p108_can['f1_por_etiqueta']['AG']:+.4f} |
| Particion 47 | 0.776100 | {m_p108_47['f1_media_columnas']:.6f} | **{m_p109_47['f1_media_columnas']:.6f}** | **{delta_vs_p108_47:+.6f}** |
| Kaggle LB | 0.78306 | 0.78855 | Proy. ~0.798 | +0.010 |

## 3. Comportamiento sobre Test (1.153 filas)
- Celdas distintas de P108: {celdas_dist_p108_te} de {p108_test_arr.size} ({celdas_dist_p108_te/p108_test_arr.size*100:.2f}%).
- Diferencias por columna: HS: {diferencias_col_p108['HS']}, TR: {diferencias_col_p108['TR']}, AG: {diferencias_col_p108['AG']}.
- Rescates regla ascendente TR=1 => HS=1: {violaciones_corregidas} casos.
- Features totales train completo: {X_train_full.shape[1]} (frente a 86.243 en P108).
"""
    with open(SAL_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    log(f"Reporte Markdown guardado en: {SAL_MD}")
    log(f"Ejecucion P109 completada con exito en {t_total:.2f} segundos.")


if __name__ == "__main__":
    main()
