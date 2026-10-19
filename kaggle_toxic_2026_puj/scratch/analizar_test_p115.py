#!/usr/bin/env python
"""Analisis detallado de predicciones de test para P115 (Tri-Optimizer + Dual Ontological Gating)."""

from __future__ import annotations
import hashlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

SUB_P114 = RAIZ / "submissions" / "toxic_submission_p114_tri_optimizer_smooth_blend_asc.csv"
SUB_P110 = RAIZ / "submissions" / "toxic_submission_p110_lr_sgd_blend_asc.csv"
SUB_P108 = RAIZ / "submissions" / "toxic_submission_p108_v2_triple_asc.csv"

UMBRALES_CONGELADOS = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
C_PER_LABEL = {"HS": 2.0, "TR": 1.8, "AG": 1.5}
TEMPERATURE = 1.04

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

def aplicar_gating_ontologico(
    pred_raw: np.ndarray,
    probs_cal: np.ndarray,
    hs_rescue: float = 0.39,
    hs_for_tr: float = 0.60,
    tr_rescue: float = 0.48,
) -> tuple[np.ndarray, dict]:
    p = pred_raw.copy()
    
    # 1. Rescate de HS cuando AG=1 y HS esta en la frontera
    mask_rescue_hs = (p[:, 0] == 0) & (p[:, 2] == 1) & (probs_cal[:, 0] >= hs_rescue)
    n_rescue_hs = int(mask_rescue_hs.sum())
    p[mask_rescue_hs, 0] = 1
    
    # 2. Rescate de TR cuando HS es de alta confianza y TR esta en la frontera
    mask_rescue_tr = (p[:, 0] == 1) & (probs_cal[:, 0] >= hs_for_tr) & (p[:, 1] == 0) & (probs_cal[:, 1] >= tr_rescue)
    n_rescue_tr = int(mask_rescue_tr.sum())
    p[mask_rescue_tr, 1] = 1
    
    # 3. Regla ascendente estricta
    p, n_asc = aplicar_regla_ascendente(p)
    
    stats = {
        "n_rescue_hs": n_rescue_hs,
        "n_rescue_tr": n_rescue_tr,
        "n_asc": n_asc,
    }
    return p, stats

def main():
    print("=== ANALISIS DE TEST PARA P115 ===")
    train_full = data_module.read_train()
    test_df = data_module.read_test()
    
    # Fit en 100% train (1.731 filas)
    print("Entrenando modelos completos sobre train (1.731 filas)...")
    mats_tr, mats_te = [], []
    for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
        v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
        mats_tr.append(v.fit_transform(train_full["text"]))
        mats_te.append(v.transform(test_df["text"]))
    X_tr = sparse.hstack(mats_tr).tocsr()
    X_te = sparse.hstack(mats_te).tocsr()
    
    prob_lr = np.zeros((len(test_df), 3), dtype=float)
    prob_sgd = np.zeros((len(test_df), 3), dtype=float)
    prob_saga = np.zeros((len(test_df), 3), dtype=float)
    
    for j, et in enumerate(ETIQUETAS):
        y_col = train_full[et]
        c_val = C_PER_LABEL[et]
        
        # LR LibLinear
        lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        lr.fit(X_tr, y_col)
        prob_lr[:, j] = lr.predict_proba(X_te)[:, 1]
        
        # SGD log_loss
        sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
        sgd.fit(X_tr, y_col)
        prob_sgd[:, j] = sgd.predict_proba(X_te)[:, 1]
        
        # SAGA ElasticNet
        saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
        saga.fit(X_tr, y_col)
        prob_saga[:, j] = saga.predict_proba(X_te)[:, 1]
        
    eps = 1e-6
    prob_blend = 0.80 * prob_lr + 0.05 * prob_sgd + 0.15 * prob_saga
    prob_cal = expit(logit(np.clip(prob_blend, eps, 1 - eps)) / TEMPERATURE)
    
    pred_raw = np.column_stack([(prob_cal[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    
    # Cargar P114 para verificacion
    sub_p114 = pd.read_csv(SUB_P114)
    pred_p114 = sub_p114[list(ETIQUETAS)].to_numpy(dtype=int)
    
    # Probar diferentes configuraciones de gating en test
    for hs_rescue in [0.38, 0.39, 0.40]:
        for tr_rescue in [0.47, 0.48, 0.49]:
            pred_p115, stats = aplicar_gating_ontologico(
                pred_raw, prob_cal, hs_rescue=hs_rescue, hs_for_tr=0.60, tr_rescue=tr_rescue
            )
            
            diff_p114 = int((pred_p114 != pred_p115).sum())
            pos = {et: int(pred_p115[:, j].sum()) for j, et in enumerate(ETIQUETAS)}
            total_pos = sum(pos.values())
            pct_pos = total_pos / (len(test_df) * 3) * 100
            
            print(f"hs_rescue={hs_rescue}, tr_rescue={tr_rescue}:")
            print(f"  Rescates: HS={stats['n_rescue_hs']}, TR={stats['n_rescue_tr']}, Violaciones ASC={stats['n_asc']}")
            print(f"  Positivos: HS={pos['HS']}, TR={pos['TR']}, AG={pos['AG']} | Total={total_pos} ({pct_pos:.2f}%)")
            print(f"  Celdas distintas vs P114: {diff_p114} ({diff_p114/3462*100:.2f}%)")
            
            # Ver casos especificos modificados
            idx_diff = np.where(pred_p114 != pred_p115)[0]
            if hs_rescue == 0.39 and tr_rescue == 0.48:
                print(f"  Filas modificadas ({len(np.unique(idx_diff))} filas):")
                for i in np.unique(idx_diff)[:10]:
                    t = test_df.iloc[i]
                    p114_labels = pred_p114[i]
                    p115_labels = pred_p115[i]
                    print(f"    [Fila {i}, ID {t['id']}] P114: {p114_labels} -> P115: {p115_labels}")
                    print(f"      Probs: HS={prob_cal[i,0]:.3f}, TR={prob_cal[i,1]:.3f}, AG={prob_cal[i,2]:.3f}")
                    print(f"      Texto: {t['text'][:80]}...")

if __name__ == "__main__":
    main()
