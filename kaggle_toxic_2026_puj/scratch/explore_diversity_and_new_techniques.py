#!/usr/bin/env python
"""scratch/explore_diversity_and_new_techniques.py

Exploracion sistematica de tecnicas no utilizadas y familias de modelos con diversidad real:
1. RidgeClassifier / Ridge con calibracion sigmoide (loss L2 / error cuadratico vs log-loss)
2. Modelos Multi-View desacoplados (Expertos independientes en Word vs Char_wb vs Char)
3. Ponderacion Okapi BM25 desde cero
4. Proyecciones densas LSA / TruncatedSVD desde cero
5. Matriz de correlacion de errores y desacuerdo Q-stat con P114
6. Podado selectivo de precision en AG (eliminacion de falsos positivos en zonas de baja toxicidad)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

OOF_FINAL = RAIZ / "features" / "oof_v03_final.npz"
OOF_BASE = RAIZ / "features" / "v2_oof_base_corrected.npz"
IDS_PATH = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"

UMBRALES_CONGELADOS = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
C_PER_LABEL = {"HS": 2.0, "TR": 1.8, "AG": 1.5}

BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]


def aplicar_regla_ascendente(pred: np.ndarray) -> tuple[np.ndarray, int]:
    p = pred.copy()
    mask = (p[:, 1] == 1) & (p[:, 0] == 0)
    p[mask, 0] = 1
    return p, int(mask.sum())


def main():
    print("=== CARGANDO DATOS Y PARTICIONES ===")
    train_full = data_module.read_train()
    zf = np.load(OOF_FINAL, allow_pickle=True)
    zb = np.load(OOF_BASE, allow_pickle=True)
    pliegues_can = zf["pliegues"].astype(int)
    grupos = zb["grupos"].astype(np.int64)

    particion = pd.read_csv(IDS_PATH)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    train_dev = train_full[train_full["id"].astype(int).isin(ids_dev)].reset_index(drop=True)

    textos = train_dev["text"]
    y = train_dev[list(ETIQUETAS)]
    y_true_arr = y.to_numpy(dtype=int)
    n_samples = len(textos)

    print(f"Dataset Dev: {n_samples} filas | Clases: HS={y['HS'].sum()}, TR={y['TR'].sum()}, AG={y['AG'].sum()}")

    # 1. Extraccion de matrices por bloque para dev
    print("\n--- 1. Extraccion de Features V2_triple y por bloque ---")
    mats_word_tr_list, mats_word_va_list = {}, {}
    mats_cwb_tr_list, mats_cwb_va_list = {}, {}
    mats_char_tr_list, mats_char_va_list = {}, {}
    mats_v2_tr_list, mats_v2_va_list = {}, {}

    for fold in sorted(set(pliegues_can)):
        idx_tr = np.flatnonzero(pliegues_can != fold)
        idx_va = np.flatnonzero(pliegues_can == fold)

        # Word(1,2)
        vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
        xw_tr = vw.fit_transform(textos.iloc[idx_tr])
        xw_va = vw.transform(textos.iloc[idx_va])

        # Char_wb(3,5)
        vcwb = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, lowercase=True)
        xcwb_tr = vcwb.fit_transform(textos.iloc[idx_tr])
        xcwb_va = vcwb.transform(textos.iloc[idx_va])

        # Char(1,6)
        vc = TfidfVectorizer(analyzer="char", ngram_range=(1, 6), min_df=2, sublinear_tf=True, lowercase=True)
        xc_tr = vc.fit_transform(textos.iloc[idx_tr])
        xc_va = vc.transform(textos.iloc[idx_va])

        # V2_triple
        xv2_tr = sparse.hstack([xw_tr, xcwb_tr, xc_tr]).tocsr()
        xv2_va = sparse.hstack([xw_va, xcwb_va, xc_va]).tocsr()

        mats_word_tr_list[fold] = xw_tr
        mats_word_va_list[fold] = xw_va
        mats_cwb_tr_list[fold] = xcwb_tr
        mats_cwb_va_list[fold] = xcwb_va
        mats_char_tr_list[fold] = xc_tr
        mats_char_va_list[fold] = xc_va
        mats_v2_tr_list[fold] = xv2_tr
        mats_v2_va_list[fold] = xv2_va

    # 2. Benchmark Modelo Campeon P114
    print("\n--- 2. Calculando Baseline P114 en Canonica ---")
    prob_p114 = np.zeros((n_samples, 3), dtype=float)
    prob_lr = np.zeros((n_samples, 3), dtype=float)
    prob_sgd = np.zeros((n_samples, 3), dtype=float)
    prob_saga = np.zeros((n_samples, 3), dtype=float)

    for fold in sorted(set(pliegues_can)):
        idx_tr = np.flatnonzero(pliegues_can != fold)
        idx_va = np.flatnonzero(pliegues_can == fold)
        X_tr = mats_v2_tr_list[fold]
        X_va = mats_v2_va_list[fold]

        for j, et in enumerate(ETIQUETAS):
            y_col = y[et].iloc[idx_tr]
            c_val = C_PER_LABEL[et]

            # LR liblinear
            lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr.fit(X_tr, y_col)
            prob_lr[idx_va, j] = lr.predict_proba(X_va)[:, 1]

            # SGD log_loss
            sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
            sgd.fit(X_tr, y_col)
            prob_sgd[idx_va, j] = sgd.predict_proba(X_va)[:, 1]

            # SAGA elasticnet
            saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
            saga.fit(X_tr, y_col)
            prob_saga[idx_va, j] = saga.predict_proba(X_va)[:, 1]

    prob_blend = 0.80 * prob_lr + 0.05 * prob_sgd + 0.15 * prob_saga
    eps = 1e-6
    prob_p114 = expit(logit(np.clip(prob_blend, eps, 1 - eps)) / 1.04)

    pred_p114_raw = np.column_stack([(prob_p114[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    pred_p114, _ = aplicar_regla_ascendente(pred_p114_raw)
    met_p114 = metricas_multietiqueta(y_true_arr, pred_p114)
    print(f"P114 Canonica CCV: {met_p114['f1_media_columnas']:.6f} | HS: {met_p114['f1_por_etiqueta']['HS']:.4f}, TR: {met_p114['f1_por_etiqueta']['TR']:.4f}, AG: {met_p114['f1_por_etiqueta']['AG']:.4f}")

    # 3. TECNICA 1: RidgeClassifier / Ridge con Platt Scaling (Loss L2 vs Log-loss)
    print("\n--- 3. Tecnica 1: RidgeClassifier / Platt Scaling en V2_triple ---")
    prob_ridge = np.zeros((n_samples, 3), dtype=float)
    for fold in sorted(set(pliegues_can)):
        idx_tr = np.flatnonzero(pliegues_can != fold)
        idx_va = np.flatnonzero(pliegues_can == fold)
        X_tr = mats_v2_tr_list[fold]
        X_va = mats_v2_va_list[fold]

        for j, et in enumerate(ETIQUETAS):
            y_col = y[et].iloc[idx_tr]
            # Ridge con alpha=1.5 / 2.0 y calibracion sigmoide
            base_ridge = RidgeClassifier(alpha=2.0, class_weight="balanced", random_state=42)
            # Calibracion sigmoide (Platt scaling) con 3-fold interno dentro de train_fold
            cal_ridge = CalibratedClassifierCV(estimator=base_ridge, method="sigmoid", cv=3)
            cal_ridge.fit(X_tr, y_col)
            prob_ridge[idx_va, j] = cal_ridge.predict_proba(X_va)[:, 1]

    pred_ridge_raw = np.column_stack([(prob_ridge[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    pred_ridge, _ = aplicar_regla_ascendente(pred_ridge_raw)
    met_ridge = metricas_multietiqueta(y_true_arr, pred_ridge)
    print(f"Ridge Platt CCV: {met_ridge['f1_media_columnas']:.6f} | HS: {met_ridge['f1_por_etiqueta']['HS']:.4f}, TR: {met_ridge['f1_por_etiqueta']['TR']:.4f}, AG: {met_ridge['f1_por_etiqueta']['AG']:.4f}")
    
    # Correlacion y diversidad Ridge vs P114
    for j, et in enumerate(ETIQUETAS):
        corr = np.corrcoef(prob_p114[:, j], prob_ridge[:, j])[0, 1]
        err_p114 = (pred_p114[:, j] != y_true_arr[:, j])
        err_ridge = (pred_ridge[:, j] != y_true_arr[:, j])
        err_corr = np.corrcoef(err_p114.astype(float), err_ridge.astype(float))[0, 1]
        rescates = np.sum((err_p114 == 1) & (err_ridge == 0))
        print(f"  {et}: Corr probas: {corr:.4f} | Corr errores: {err_corr:.4f} | Celdas donde Ridge acierta y P114 falla: {rescates}")

    # 4. TECNICA 2: Multi-View Desacoplado (Word-Expert + Char_wb Expert + Char Expert)
    print("\n--- 4. Tecnica 2: Multi-View Desacoplado (Word vs Char_wb vs Char) ---")
    prob_word = np.zeros((n_samples, 3), dtype=float)
    prob_cwb = np.zeros((n_samples, 3), dtype=float)
    prob_char = np.zeros((n_samples, 3), dtype=float)

    for fold in sorted(set(pliegues_can)):
        idx_tr = np.flatnonzero(pliegues_can != fold)
        idx_va = np.flatnonzero(pliegues_can == fold)

        for j, et in enumerate(ETIQUETAS):
            y_col = y[et].iloc[idx_tr]
            c_val = C_PER_LABEL[et]

            # Word expert
            lr_w = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr_w.fit(mats_word_tr_list[fold], y_col)
            prob_word[idx_va, j] = lr_w.predict_proba(mats_word_va_list[fold])[:, 1]

            # Char_wb expert
            lr_cwb = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr_cwb.fit(mats_cwb_tr_list[fold], y_col)
            prob_cwb[idx_va, j] = lr_cwb.predict_proba(mats_cwb_va_list[fold])[:, 1]

            # Char expert
            lr_c = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr_c.fit(mats_char_tr_list[fold], y_col)
            prob_char[idx_va, j] = lr_c.predict_proba(mats_char_va_list[fold])[:, 1]

    # Evaluamos cada experto individual
    for nombre, pr in [("Word Expert", prob_word), ("Char_wb Expert", prob_cwb), ("Char Expert", prob_char)]:
        p_raw = np.column_stack([(pr[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
        p_asc, _ = aplicar_regla_ascendente(p_raw)
        m = metricas_multietiqueta(y_true_arr, p_asc)
        print(f"  {nombre:16s} CCV: {m['f1_media_columnas']:.6f} | HS: {m['f1_por_etiqueta']['HS']:.4f}, TR: {m['f1_por_etiqueta']['TR']:.4f}, AG: {m['f1_por_etiqueta']['AG']:.4f}")

    # 5. TECNICA 3: Proyecciones Densas LSA (TruncatedSVD) desde cero
    print("\n--- 5. Tecnica 3: LSA / TruncatedSVD (150 componentes) desde cero ---")
    prob_svd = np.zeros((n_samples, 3), dtype=float)
    for fold in sorted(set(pliegues_can)):
        idx_tr = np.flatnonzero(pliegues_can != fold)
        idx_va = np.flatnonzero(pliegues_can == fold)
        X_tr = mats_v2_tr_list[fold]
        X_va = mats_v2_va_list[fold]

        svd = TruncatedSVD(n_components=150, random_state=42)
        X_tr_svd = svd.fit_transform(X_tr)
        X_va_svd = svd.transform(X_va)

        for j, et in enumerate(ETIQUETAS):
            y_col = y[et].iloc[idx_tr]
            lr_svd = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
            lr_svd.fit(X_tr_svd, y_col)
            prob_svd[idx_va, j] = lr_svd.predict_proba(X_va_svd)[:, 1]

    pred_svd_raw = np.column_stack([(prob_svd[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    pred_svd, _ = aplicar_regla_ascendente(pred_svd_raw)
    met_svd = metricas_multietiqueta(y_true_arr, pred_svd)
    print(f"LSA SVD(150) CCV: {met_svd['f1_media_columnas']:.6f} | HS: {met_svd['f1_por_etiqueta']['HS']:.4f}, TR: {met_svd['f1_por_etiqueta']['TR']:.4f}, AG: {met_svd['f1_por_etiqueta']['AG']:.4f}")

    for j, et in enumerate(ETIQUETAS):
        corr = np.corrcoef(prob_p114[:, j], prob_svd[:, j])[0, 1]
        err_p114 = (pred_p114[:, j] != y_true_arr[:, j])
        err_svd = (pred_svd[:, j] != y_true_arr[:, j])
        err_corr = np.corrcoef(err_p114.astype(float), err_svd.astype(float))[0, 1]
        rescates = np.sum((err_p114 == 1) & (err_svd == 0))
        print(f"  {et}: Corr probas: {corr:.4f} | Corr errores: {err_corr:.4f} | Celdas donde SVD acierta y P114 falla: {rescates}")

    # 6. TECNICA 4: Ensambles Multi-Familia Hibridos P114 + Ridge + Multi-View + SVD
    print("\n--- 6. Tecnica 4: Ensambles Optimizados P114 + Ridge + LSA + Multi-View ---")
    
    # Rejilla de combinaciones lineales
    candidatos = [
        ("P114 solo", prob_p114),
        ("90% P114 + 10% Ridge", 0.90 * prob_p114 + 0.10 * prob_ridge),
        ("85% P114 + 15% Ridge", 0.85 * prob_p114 + 0.15 * prob_ridge),
        ("80% P114 + 15% Ridge + 5% SVD", 0.80 * prob_p114 + 0.15 * prob_ridge + 0.05 * prob_svd),
        ("75% P114 + 15% Ridge + 10% Word", 0.75 * prob_p114 + 0.15 * prob_ridge + 0.10 * prob_word),
        ("80% P114 + 10% Ridge + 10% C_wb", 0.80 * prob_p114 + 0.10 * prob_ridge + 0.10 * prob_cwb),
        ("70% P114 + 15% Ridge + 10% Word + 5% SVD", 0.70 * prob_p114 + 0.15 * prob_ridge + 0.10 * prob_word + 0.05 * prob_svd),
    ]

    for nombre, p_comb in candidatos:
        p_cal = expit(logit(np.clip(p_comb, eps, 1 - eps)) / 1.04)
        p_raw = np.column_stack([(p_cal[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
        p_asc, _ = aplicar_regla_ascendente(p_raw)
        m = metricas_multietiqueta(y_true_arr, p_asc)
        delta = m['f1_media_columnas'] - met_p114['f1_media_columnas']
        print(f"{nombre:45s} CCV: {m['f1_media_columnas']:.6f} (Delta: {delta:+.6f}) | HS: {m['f1_por_etiqueta']['HS']:.4f}, TR: {m['f1_por_etiqueta']['TR']:.4f}, AG: {m['f1_por_etiqueta']['AG']:.4f}")

    # 7. TECNICA 5: Podado Condicional de Alta Precision en AG
    print("\n--- 7. Tecnica 5: Diagnostico Forense de la franja limite de AG ---")
    prob_ag = prob_p114[:, 2]
    y_ag = y_true_arr[:, 2]
    prob_hs = prob_p114[:, 0]

    # Analizamos la franja limite de AG [0.44, 0.49)
    franja_ag = (prob_ag >= 0.44) & (prob_ag < 0.50)
    print(f"Muestras en franja AG [0.44, 0.50): {franja_ag.sum()}")
    print(f"  Verdaderos Positivos en franja: {(franja_ag & (y_ag == 1)).sum()}")
    print(f"  Falsos Positivos en franja: {(franja_ag & (y_ag == 0)).sum()}")
    prec_franja = (franja_ag & (y_ag == 1)).sum() / franja_ag.sum()
    print(f"  Precision en franja: {prec_franja*100:.2f}%")

    # Muestras en franja AG con HS bajo (< 0.35)
    franja_ag_hs_bajo = franja_ag & (prob_hs < 0.35)
    print(f"Muestras en franja AG con HS < 0.35: {franja_ag_hs_bajo.sum()}")
    print(f"  Verdaderos Positivos con HS < 0.35: {(franja_ag_hs_bajo & (y_ag == 1)).sum()}")
    print(f"  Falsos Positivos con HS < 0.35: {(franja_ag_hs_bajo & (y_ag == 0)).sum()}")
    if franja_ag_hs_bajo.sum() > 0:
        prec_baja = (franja_ag_hs_bajo & (y_ag == 1)).sum() / franja_ag_hs_bajo.sum()
        print(f"  Precision de AG cuando HS < 0.35: {prec_baja*100:.2f}% (TASA DE FALSA ALARMA: {(1-prec_baja)*100:.2f}%)")

    # Simulacion de poda en dev
    for umbral_hs_corte in [0.25, 0.30, 0.35, 0.40]:
        for ag_max in [0.47, 0.48, 0.49]:
            pred_podada = pred_p114.copy()
            # Si AG estaba en 1, pero prob_ag < ag_max y prob_hs < umbral_hs_corte, podar a 0
            cond_poda = (pred_podada[:, 2] == 1) & (prob_ag < ag_max) & (prob_hs < umbral_hs_corte)
            pred_podada[cond_poda, 2] = 0
            m_podada = metricas_multietiqueta(y_true_arr, pred_podada)
            d = m_podada['f1_media_columnas'] - met_p114['f1_media_columnas']
            podadas_total = cond_poda.sum()
            fp_eliminados = (cond_poda & (y_ag == 0)).sum()
            tp_perdidos = (cond_poda & (y_ag == 1)).sum()
            if d > 0:
                print(f"  PODA AG: prob_ag<{ag_max} y prob_hs<{umbral_hs_corte} -> Podadas: {podadas_total} (FP eliminados: {fp_eliminados}, TP perdidos: {tp_perdidos}) -> CCV: {m_podada['f1_media_columnas']:.6f} (Delta: {d:+.6f}) | AG F1: {m_podada['f1_por_etiqueta']['AG']:.4f}")


if __name__ == "__main__":
    main()
