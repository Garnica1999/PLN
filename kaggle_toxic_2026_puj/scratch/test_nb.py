#!/usr/bin/env python
"""Test Naive Bayes (ComplementNB, MultinomialNB) on V2_triple and blends with LR."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB

RAIZ = Path("/home/carlos/Git/PLN/kaggle_toxic_2026_puj")
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

OOF_FINAL = RAIZ / "features" / "oof_v03_final.npz"
OOF_BASE = RAIZ / "features" / "v2_oof_base_corrected.npz"
IDS_PATH = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
UMBRALES_CONGELADOS = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

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
    train_full = data_module.read_train()
    zf = np.load(OOF_FINAL, allow_pickle=True)
    zb = np.load(OOF_BASE, allow_pickle=True)
    pliegues_can = zf["pliegues"].astype(int)

    particion = pd.read_csv(IDS_PATH)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    train_dev = train_full[train_full["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    y_true = train_dev[list(ETIQUETAS)].to_numpy(dtype=int)

    n_samples = len(train_dev)
    prob_lr = np.zeros((n_samples, 3), dtype=float)
    prob_sgd = np.zeros((n_samples, 3), dtype=float)
    prob_cnb = np.zeros((n_samples, 3), dtype=float)
    prob_mnb = np.zeros((n_samples, 3), dtype=float)

    for fold in sorted(set(pliegues_can)):
        idx_tr = np.flatnonzero(pliegues_can != fold)
        idx_va = np.flatnonzero(pliegues_can == fold)

        mats_tr, mats_va = [], []
        for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
            v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
            mats_tr.append(v.fit_transform(train_dev["text"].iloc[idx_tr]))
            mats_va.append(v.transform(train_dev["text"].iloc[idx_va]))
        X_tr = sparse.hstack(mats_tr).tocsr()
        X_va = sparse.hstack(mats_va).tocsr()

        for j, et in enumerate(ETIQUETAS):
            y_col = train_dev[et].iloc[idx_tr]

            lr = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr.fit(X_tr, y_col)
            prob_lr[idx_va, j] = lr.predict_proba(X_va)[:, 1]

            sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
            sgd.fit(X_tr, y_col)
            prob_sgd[idx_va, j] = sgd.predict_proba(X_va)[:, 1]

            cnb = ComplementNB(alpha=1.0)
            cnb.fit(X_tr, y_col)
            prob_cnb[idx_va, j] = cnb.predict_proba(X_va)[:, 1]

            mnb = MultinomialNB(alpha=1.0)
            mnb.fit(X_tr, y_col)
            prob_mnb[idx_va, j] = mnb.predict_proba(X_va)[:, 1]

    # Baseline P110 F1
    p110 = 0.95 * prob_lr + 0.05 * prob_sgd
    pred_base = aplicar_regla_ascendente(np.column_stack([(p110[:, j] >= [0.44, 0.51, 0.46][j]).astype(int) for j in range(3)]))
    met_base = metricas_multietiqueta(y_true, pred_base)
    print(f"P110 F1 macro (Canonica): {met_base['f1_media_columnas']:.6f}")

    # Standalone CNB and MNB
    pred_cnb = aplicar_regla_ascendente(np.column_stack([(prob_cnb[:, j] >= 0.5).astype(int) for j in range(3)]))
    print(f"Standalone ComplementNB (thr=0.5): {metricas_multietiqueta(y_true, pred_cnb)['f1_media_columnas']:.6f}")
    pred_mnb = aplicar_regla_ascendente(np.column_stack([(prob_mnb[:, j] >= 0.5).astype(int) for j in range(3)]))
    print(f"Standalone MultinomialNB (thr=0.5): {metricas_multietiqueta(y_true, pred_mnb)['f1_media_columnas']:.6f}")

    # Test blending CNB with P110
    print("\nTesting Blends with ComplementNB:")
    for w_cnb in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
        blend = (1 - w_cnb) * p110 + w_cnb * prob_cnb
        pred = aplicar_regla_ascendente(np.column_stack([(blend[:, j] >= [0.44, 0.51, 0.46][j]).astype(int) for j in range(3)]))
        sc = metricas_multietiqueta(y_true, pred)["f1_media_columnas"]
        print(f"  w_cnb={w_cnb:.2f}: F1={sc:.6f} (Delta={sc - met_base['f1_media_columnas']:+.6f})")

if __name__ == "__main__":
    main()
