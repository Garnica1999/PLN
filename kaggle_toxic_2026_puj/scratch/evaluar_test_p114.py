#!/usr/bin/env python
"""Evaluate P114 candidate on test set and run full validation metrics."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier

RAIZ = Path("/home/carlos/Git/PLN/kaggle_toxic_2026_puj")
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

SUB_P110 = RAIZ / "submissions" / "toxic_submission_p110_lr_sgd_blend_asc.csv"
SUB_P108 = RAIZ / "submissions" / "toxic_submission_p108_v2_triple_asc.csv"
UMBRALES_CONGELADOS = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]

C_PER_LABEL = {"HS": 2.0, "TR": 1.8, "AG": 1.5}

def aplicar_regla_ascendente(pred: np.ndarray) -> tuple[np.ndarray, int]:
    p = pred.copy()
    mask = (p[:, 1] == 1) & (p[:, 0] == 0)
    p[mask, 0] = 1
    return p, int(mask.sum())

def main():
    train_full = data_module.read_train()
    test_df = data_module.read_test()
    sub_p110_df = pd.read_csv(SUB_P110)
    pred_p110_arr = sub_p110_df[list(ETIQUETAS)].to_numpy(dtype=int)

    # 1. Fit on 100% of train (1,731 rows)
    mats_tr_full, mats_te = [], []
    for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
        v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
        mats_tr_full.append(v.fit_transform(train_full["text"]))
        mats_te.append(v.transform(test_df["text"]))
    X_tr_full = sparse.hstack(mats_tr_full).tocsr()
    X_te = sparse.hstack(mats_te).tocsr()

    print(f"Features: {X_tr_full.shape[1]}")

    prob_lr = np.zeros((len(test_df), len(ETIQUETAS)), dtype=float)
    prob_sgd = np.zeros((len(test_df), len(ETIQUETAS)), dtype=float)
    prob_saga = np.zeros((len(test_df), len(ETIQUETAS)), dtype=float)

    for j, et in enumerate(ETIQUETAS):
        y_col = train_full[et]
        c_val = C_PER_LABEL[et]

        # LR
        clf_lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        clf_lr.fit(X_tr_full, y_col)
        prob_lr[:, j] = clf_lr.predict_proba(X_te)[:, 1]

        # SGD
        clf_sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
        clf_sgd.fit(X_tr_full, y_col)
        prob_sgd[:, j] = clf_sgd.predict_proba(X_te)[:, 1]

        # SAGA ElasticNet (l1=0.05)
        clf_saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
        clf_saga.fit(X_tr_full, y_col)
        prob_saga[:, j] = clf_saga.predict_proba(X_te)[:, 1]

    # Candidate 1: 0.80*LR + 0.05*SGD + 0.15*SAGA with T=1.04
    blend_raw = 0.80 * prob_lr + 0.05 * prob_sgd + 0.15 * prob_saga
    eps = 1e-6
    blend_t104 = expit(logit(np.clip(blend_raw, eps, 1 - eps)) / 1.04)

    # Predictions
    pred_raw = np.column_stack([(blend_t104[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    pred_final, n_asc = aplicar_regla_ascendente(pred_raw)

    diff_cells = int((pred_final != pred_p110_arr).sum())
    pos_hs = int(pred_final[:, 0].sum())
    pos_tr = int(pred_final[:, 1].sum())
    pos_ag = int(pred_final[:, 2].sum())
    total_pos = int(pred_final.sum())

    print("\n=== P114 CANDIDATE (80% LR_c + 5% SGD + 15% SAGA_c, T=1.04) ===")
    print(f"Celdas distintas de P110: {diff_cells} ({diff_cells / 3462 * 100:.2f}%)")
    print(f"Positivos predichos: HS={pos_hs}, TR={pos_tr}, AG={pos_ag} | Total={total_pos} ({total_pos / 3462 * 100:.2f}%)")
    print(f"P110 Positivos: HS=513, TR=327, AG=419 | Total=1259 (36.37%)")
    print(f"Violaciones ontologicas TR=1 => HS=1 corregidas: {n_asc}")

    # Inspect cell changes
    changed_idx = np.where((pred_final != pred_p110_arr).any(axis=1))[0]
    print(f"\nFilas modificadas: {len(changed_idx)}")
    for idx in changed_idx[:10]:
        print(f"Row {idx} (id={test_df['id'].iloc[idx]}):")
        print(f"  P110: {pred_p110_arr[idx]} | P114: {pred_final[idx]}")
        print(f"  Tweet: {test_df['text'].iloc[idx][:100]}")

if __name__ == "__main__":
    main()
