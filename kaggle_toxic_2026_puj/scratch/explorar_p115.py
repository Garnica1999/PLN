#!/usr/bin/env python
"""Exploracion rigurosa para P115 sobre Canonica y Part47."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

OOF_FINAL = RAIZ / "features" / "oof_v03_final.npz"
OOF_BASE = RAIZ / "features" / "v2_oof_base_corrected.npz"
IDS_PATH = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
SUB_P114 = RAIZ / "submissions" / "toxic_submission_p114_tri_optimizer_smooth_blend_asc.csv"

UMBRALES_CONGELADOS = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
C_PER_LABEL = {"HS": 2.0, "TR": 1.8, "AG": 1.5}
TEMPERATURE = 1.04

BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]

def aplicar_regla_ascendente(pred: np.ndarray) -> np.ndarray:
    p = pred.copy()
    mask = (p[:, 1] == 1) & (p[:, 0] == 0)
    p[mask, 0] = 1
    return p

def main():
    print("=== EXPLORACION P115 ===")
    train_full = data_module.read_train()
    test_df = data_module.read_test()

    zf = np.load(OOF_FINAL, allow_pickle=True)
    zb = np.load(OOF_BASE, allow_pickle=True)
    pliegues_can = zf["pliegues"].astype(int)
    grupos = zb["grupos"].astype(np.int64)

    particion = pd.read_csv(IDS_PATH)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    train_dev = train_full[train_full["id"].astype(int).isin(ids_dev)].reset_index(drop=True)

    y_true_dev = train_dev[list(ETIQUETAS)].to_numpy(dtype=int)
    combinacion = np.array([f"{a}{b}{c}" for a, b, c in y_true_dev])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
    pliegues_47 = np.full(len(train_dev), -1, dtype=int)
    for num, (_, val) in enumerate(sgkf.split(np.arange(len(train_dev)).reshape(-1, 1), combinacion, groups=grupos)):
        pliegues_47[val] = num + 1

    # 1. Obtener probabilidades OOF para Canonica y Part47 con P114
    for nombre_part, pliegues in [("Canonica", pliegues_can), ("Part47", pliegues_47)]:
        print(f"\n--- Evaluando {nombre_part} ---")
        n_samples = len(train_dev)
        prob_lr = np.zeros((n_samples, 3), dtype=float)
        prob_sgd = np.zeros((n_samples, 3), dtype=float)
        prob_saga = np.zeros((n_samples, 3), dtype=float)
        prob_lbfgs = np.zeros((n_samples, 3), dtype=float)

        for fold in sorted(set(pliegues)):
            idx_tr = np.flatnonzero(pliegues != fold)
            idx_va = np.flatnonzero(pliegues == fold)

            mats_tr, mats_va = [], []
            for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
                v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
                mats_tr.append(v.fit_transform(train_dev["text"].iloc[idx_tr]))
                mats_va.append(v.transform(train_dev["text"].iloc[idx_va]))
            X_tr = sparse.hstack(mats_tr).tocsr()
            X_va = sparse.hstack(mats_va).tocsr()

            for j, et in enumerate(ETIQUETAS):
                y_col = train_dev[et].iloc[idx_tr]
                c_val = C_PER_LABEL[et]

                # Liblinear
                lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
                lr.fit(X_tr, y_col)
                prob_lr[idx_va, j] = lr.predict_proba(X_va)[:, 1]

                # SGD log_loss
                sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
                sgd.fit(X_tr, y_col)
                prob_sgd[idx_va, j] = sgd.predict_proba(X_va)[:, 1]

                # SAGA ElasticNet
                saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
                saga.fit(X_tr, y_col)
                prob_saga[idx_va, j] = saga.predict_proba(X_va)[:, 1]

                # L-BFGS
                lbfgs = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42)
                lbfgs.fit(X_tr, y_col)
                prob_lbfgs[idx_va, j] = lbfgs.predict_proba(X_va)[:, 1]

        # P114 base
        p114_blend = 0.80 * prob_lr + 0.05 * prob_sgd + 0.15 * prob_saga
        eps = 1e-6
        p114_cal = expit(logit(np.clip(p114_blend, eps, 1 - eps)) / TEMPERATURE)
        p114_pred_raw = np.column_stack([(p114_cal[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
        p114_pred = aplicar_regla_ascendente(p114_pred_raw)
        m_p114 = metricas_multietiqueta(y_true_dev, p114_pred)
        print(f"P114 Base F1: {m_p114['f1_media_columnas']:.6f} | HS: {m_p114['f1_por_etiqueta']['HS']:.4f}, TR: {m_p114['f1_por_etiqueta']['TR']:.4f}, AG: {m_p114['f1_por_etiqueta']['AG']:.4f}")

        # Test Quad-Optimizer Blend (e.g. 60% LR + 20% L-BFGS + 15% SAGA + 5% SGD)
        for w_lr, w_lbfgs, w_saga, w_sgd in [
            (0.70, 0.10, 0.15, 0.05),
            (0.60, 0.20, 0.15, 0.05),
            (0.50, 0.30, 0.15, 0.05),
            (0.40, 0.40, 0.15, 0.05),
        ]:
            quad_blend = w_lr * prob_lr + w_lbfgs * prob_lbfgs + w_saga * prob_saga + w_sgd * prob_sgd
            quad_cal = expit(logit(np.clip(quad_blend, eps, 1 - eps)) / TEMPERATURE)
            quad_pred_raw = np.column_stack([(quad_cal[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
            quad_pred = aplicar_regla_ascendente(quad_pred_raw)
            m_quad = metricas_multietiqueta(y_true_dev, quad_pred)
            delta = m_quad['f1_media_columnas'] - m_p114['f1_media_columnas']
            print(f"Quad ({w_lr:.2f}/{w_lbfgs:.2f}/{w_saga:.2f}/{w_sgd:.2f}) F1: {m_quad['f1_media_columnas']:.6f} (Delta: {delta:+.6f})")

        # Test Ontological Gating on P114 probabilities
        print("\nPruebas de Gating Ontologico sobre P114:")
        # Error analysis on P114 OOF:
        # Check rows where p114_pred says HS=0, AG=1
        hs_0_ag_1 = (p114_pred[:, 0] == 0) & (p114_pred[:, 2] == 1)
        print(f"Casos predichos con HS=0, AG=1: {hs_0_ag_1.sum()}")
        if hs_0_ag_1.sum() > 0:
            true_hs = y_true_dev[hs_0_ag_1, 0]
            true_ag = y_true_dev[hs_0_ag_1, 2]
            print(f"  En esos casos, Verdadero HS=1: {true_hs.sum()}/{len(true_hs)} ({true_hs.mean()*100:.1f}%)")
            print(f"  En esos casos, Verdadero AG=1: {true_ag.sum()}/{len(true_ag)} ({true_ag.mean()*100:.1f}%)")
            print(f"  En esos casos, Verdadero (HS=0, AG=0): {((true_hs==0) & (true_ag==0)).sum()}")
            print(f"  En esos casos, Verdadero (HS=1, AG=1): {((true_hs==1) & (true_ag==1)).sum()}")
            print(f"  En esos casos, Verdadero (HS=1, AG=0): {((true_hs==1) & (true_ag==0)).sum()}")
            print(f"  En esos casos, Verdadero (HS=0, AG=1): {((true_hs==0) & (true_ag==1)).sum()}")

        # Grid search over gating rules
        best_f1 = m_p114['f1_media_columnas']
        best_rule = None
        for hs_rescue_min in [0.38, 0.39, 0.40, 0.41, 0.42]:
            for ag_strict_min in [0.46, 0.48, 0.50, 0.52, 0.54, 0.56]:
                p_gated = p114_pred.copy()
                # 1. Rescue HS if AG is predicted and HS is borderline
                mask_rescue_hs = (p114_pred[:, 0] == 0) & (p114_pred[:, 2] == 1) & (p114_cal[:, 0] >= hs_rescue_min)
                p_gated[mask_rescue_hs, 0] = 1

                # 2. Suppress AG if HS is definitely 0 and AG is not super strong
                mask_suppress_ag = (p_gated[:, 0] == 0) & (p114_cal[:, 2] < ag_strict_min)
                p_gated[mask_suppress_ag, 2] = 0

                p_gated = aplicar_regla_ascendente(p_gated)
                m_gated = metricas_multietiqueta(y_true_dev, p_gated)
                diff = m_gated['f1_media_columnas'] - m_p114['f1_media_columnas']
                if m_gated['f1_media_columnas'] > best_f1:
                    best_f1 = m_gated['f1_media_columnas']
                    best_rule = (hs_rescue_min, ag_strict_min, diff, m_gated)

        if best_rule:
            print(f"Mejor regla gating: rescue_hs>={best_rule[0]}, ag_strict>={best_rule[1]} -> F1: {best_rule[3]['f1_media_columnas']:.6f} (Delta: {best_rule[2]:+.6f})")
            print(f"  HS: {best_rule[3]['f1_por_etiqueta']['HS']:.4f}, TR: {best_rule[3]['f1_por_etiqueta']['TR']:.4f}, AG: {best_rule[3]['f1_por_etiqueta']['AG']:.4f}")
        else:
            print("Ninguna regla supero a P114 Base.")

if __name__ == "__main__":
    main()
