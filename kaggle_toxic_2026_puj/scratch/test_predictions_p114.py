#!/usr/bin/env python
"""Inspect test set predictions of SAGA ElasticNet blend for P114."""

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

    # 1. Feature matrices on full data (1,731 rows)
    mats_tr_full, mats_te = [], []
    for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
        v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
        mats_tr_full.append(v.fit_transform(train_full["text"]))
        mats_te.append(v.transform(test_df["text"]))
    X_tr_full = sparse.hstack(mats_tr_full).tocsr()
    X_te = sparse.hstack(mats_te).tocsr()

    print(f"Dimension total features: {X_tr_full.shape[1]}")

    prob_lr_test = np.zeros((len(test_df), len(ETIQUETAS)), dtype=float)
    prob_sgd_test = np.zeros((len(test_df), len(ETIQUETAS)), dtype=float)
    prob_saga02_test = np.zeros((len(test_df), len(ETIQUETAS)), dtype=float)
    prob_saga05_test = np.zeros((len(test_df), len(ETIQUETAS)), dtype=float)

    for j, et in enumerate(ETIQUETAS):
        y_col = train_full[et]
        # LR liblinear
        clf_lr = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        clf_lr.fit(X_tr_full, y_col)
        prob_lr_test[:, j] = clf_lr.predict_proba(X_te)[:, 1]

        # SGD log_loss
        clf_sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
        clf_sgd.fit(X_tr_full, y_col)
        prob_sgd_test[:, j] = clf_sgd.predict_proba(X_te)[:, 1]

        # SAGA l1=0.02
        clf_saga02 = LogisticRegression(C=2.0, penalty="elasticnet", l1_ratio=0.02, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
        clf_saga02.fit(X_tr_full, y_col)
        prob_saga02_test[:, j] = clf_saga02.predict_proba(X_te)[:, 1]

        # SAGA l1=0.05
        clf_saga05 = LogisticRegression(C=2.0, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
        clf_saga05.fit(X_tr_full, y_col)
        prob_saga05_test[:, j] = clf_saga05.predict_proba(X_te)[:, 1]

    # Test candidate blends
    candidates = [
        ("P110 Baseline (95% LR + 5% SGD)", 0.95 * prob_lr_test + 0.05 * prob_sgd_test, 1.0),
        ("C1: 80% LR + 5% SGD + 15% SAGA(l1=0.05), T=1.00", 0.80 * prob_lr_test + 0.05 * prob_sgd_test + 0.15 * prob_saga05_test, 1.0),
        ("C2: 80% LR + 5% SGD + 15% SAGA(l1=0.05), T=1.02", 0.80 * prob_lr_test + 0.05 * prob_sgd_test + 0.15 * prob_saga05_test, 1.02),
        ("C3: 80% LR + 5% SGD + 15% SAGA(l1=0.05), T=1.04", 0.80 * prob_lr_test + 0.05 * prob_sgd_test + 0.15 * prob_saga05_test, 1.04),
        ("C4: 85% LR + 5% SGD + 10% SAGA(l1=0.05), T=1.00", 0.85 * prob_lr_test + 0.05 * prob_sgd_test + 0.10 * prob_saga05_test, 1.0),
        ("C5: 85% LR + 5% SGD + 10% SAGA(l1=0.05), T=1.02", 0.85 * prob_lr_test + 0.05 * prob_sgd_test + 0.10 * prob_saga05_test, 1.02),
        ("C6: 75% LR + 5% SGD + 20% SAGA(l1=0.02), T=1.04", 0.75 * prob_lr_test + 0.05 * prob_sgd_test + 0.20 * prob_saga02_test, 1.04),
        ("C7: 80% LR + 5% SGD + 15% SAGA(l1=0.02), T=1.00", 0.80 * prob_lr_test + 0.05 * prob_sgd_test + 0.15 * prob_saga02_test, 1.0),
    ]

    eps = 1e-6
    for name, p_mat, T in candidates:
        if T != 1.0:
            p_adj = expit(logit(np.clip(p_mat, eps, 1 - eps)) / T)
        else:
            p_adj = p_mat

        pred_raw = np.column_stack([(p_adj[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
        pred, n_asc = aplicar_regla_ascendente(pred_raw)

        diff_p110 = int((pred != pred_p110_arr).sum())
        pos_hs = int(pred[:, 0].sum())
        pos_tr = int(pred[:, 1].sum())
        pos_ag = int(pred[:, 2].sum())
        total_pos = int(pred.sum())
        pct_pos = total_pos / (len(test_df) * 3) * 100

        print(f"\n{name}:")
        print(f"  Celdas distintas de P110: {diff_p110} ({diff_p110 / 3462 * 100:.2f}%)")
        print(f"  Positivos: HS={pos_hs}, TR={pos_tr}, AG={pos_ag} | Total={total_pos} ({pct_pos:.2f}%)")
        print(f"  Violaciones TR=1 => HS=1 corregidas: {n_asc}")

if __name__ == "__main__":
    main()
