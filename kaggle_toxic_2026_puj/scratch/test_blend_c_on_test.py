#!/usr/bin/env python
"""scratch/test_blend_c_on_test.py

Verificacion de Blend C en el test set ciego (1.154 filas):
- HS: 0.85 P114 + 0.05 BM25 + 0.10 SVD, thr=0.44
- TR: 0.85 P114 + 0.05 BM25 + 0.10 SVD, thr=0.51
- AG: 0.80 P114 + 0.00 BM25 + 0.20 SVD, thr=0.47
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.decomposition import TruncatedSVD

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS

C_PER_LABEL = {"HS": 2.0, "TR": 1.8, "AG": 1.5}


class BM25Transformer:
    def __init__(self, k1=1.2, b=0.6):
        self.k1 = k1
        self.b = b
        self.idf_ = None
        self.avgdl_ = None

    def fit(self, X):
        n_samples, n_features = X.shape
        df = np.bincount(X.indices, minlength=n_features)
        self.idf_ = np.log((n_samples - df + 0.5) / (df + 0.5) + 1.0)
        dl = X.sum(axis=1).A1
        self.avgdl_ = max(float(dl.mean()), 1.0)
        return self

    def transform(self, X):
        dl = X.sum(axis=1).A1
        len_norm = 1.0 - self.b + self.b * (dl / self.avgdl_)
        X_coo = X.tocoo()
        data = X_coo.data
        row = X_coo.row
        col = X_coo.col
        num = data * (self.k1 + 1.0)
        den = data + self.k1 * len_norm[row]
        new_data = (num / den) * self.idf_[col]
        return sparse.csr_matrix((new_data, (row, col)), shape=X.shape)


def aplicar_regla_ascendente(pred: np.ndarray) -> tuple[np.ndarray, int]:
    p = pred.copy()
    mask = (p[:, 1] == 1) & (p[:, 0] == 0)
    p[mask, 0] = 1
    return p, int(mask.sum())


def main():
    train_full = data_module.read_train()
    test_df = data_module.read_test()

    text_tr = train_full["text"]
    text_te = test_df["text"]
    y_tr = train_full[list(ETIQUETAS)]

    bloques = [
        ("word", (1, 2), True),
        ("char_wb", (3, 5), True),
        ("char", (1, 6), True),
    ]

    mats_tfidf_tr, mats_tfidf_te = [], []
    mats_cnt_tr, mats_cnt_te = [], []

    for analyzer, ngram, sublinear in bloques:
        v_tfidf = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
        mats_tfidf_tr.append(v_tfidf.fit_transform(text_tr))
        mats_tfidf_te.append(v_tfidf.transform(text_te))

        v_cnt = CountVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, lowercase=True)
        mats_cnt_tr.append(v_cnt.fit_transform(text_tr))
        mats_cnt_te.append(v_cnt.transform(text_te))

    X_tfidf_tr = sparse.hstack(mats_tfidf_tr).tocsr()
    X_tfidf_te = sparse.hstack(mats_tfidf_te).tocsr()

    X_cnt_tr = sparse.hstack(mats_cnt_tr).tocsr()
    X_cnt_te = sparse.hstack(mats_cnt_te).tocsr()

    bm25 = BM25Transformer(k1=1.2, b=0.6)
    bm25.fit(X_cnt_tr)
    X_bm25_tr = bm25.transform(X_cnt_tr)
    X_bm25_te = bm25.transform(X_cnt_te)

    svd = TruncatedSVD(n_components=150, random_state=42)
    X_svd_tr = svd.fit_transform(X_tfidf_tr)
    X_svd_te = svd.transform(X_tfidf_te)

    prob_p114_te = np.zeros((len(test_df), 3), dtype=float)
    prob_bm25_te = np.zeros((len(test_df), 3), dtype=float)
    prob_svd_te = np.zeros((len(test_df), 3), dtype=float)

    for j, et in enumerate(ETIQUETAS):
        y_col = y_tr[et]
        c_val = C_PER_LABEL[et]

        # P114 components: LR + SGD + SAGA
        lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        lr.fit(X_tfidf_tr, y_col)
        p_lr = lr.predict_proba(X_tfidf_te)[:, 1]

        sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
        sgd.fit(X_tfidf_tr, y_col)
        p_sgd = sgd.predict_proba(X_tfidf_te)[:, 1]

        saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
        saga.fit(X_tfidf_tr, y_col)
        p_saga = saga.predict_proba(X_tfidf_te)[:, 1]

        blend_p114 = 0.80 * p_lr + 0.05 * p_sgd + 0.15 * p_saga
        eps = 1e-6
        prob_p114_te[:, j] = expit(logit(np.clip(blend_p114, eps, 1 - eps)) / 1.04)

        # BM25
        lr_bm25 = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        lr_bm25.fit(X_bm25_tr, y_col)
        prob_bm25_te[:, j] = lr_bm25.predict_proba(X_bm25_te)[:, 1]

        # SVD
        lr_svd = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
        lr_svd.fit(X_svd_tr, y_col)
        prob_svd_te[:, j] = lr_svd.predict_proba(X_svd_te)[:, 1]

    # Evaluamos varias variantes de threshold para encontrar el balance perfecto
    sub_p114 = pd.read_csv(RAIZ / "submissions" / "toxic_submission_p114_tri_optimizer_smooth_blend_asc.csv")
    pred_p114 = sub_p114[list(ETIQUETAS)].to_numpy(dtype=int)

    print("\n=== COMPARATIVA DE CONFIGURACIONES EN TEST SET (1.154 filas) ===")
    print(f"P114 (Campeon LB 0.79057): HS={pred_p114[:, 0].sum()}, TR={pred_p114[:, 1].sum()}, AG={pred_p114[:, 2].sum()} | Total: {pred_p114.sum()} (36.54%)")

    test_configs = [
        ("Blend C1: 0.85/0.05/0.10 HS, 0.85/0.05/0.10 TR, 0.80/0.00/0.20 AG | thr 0.44/0.51/0.47",
         (0.85, 0.05, 0.10), (0.85, 0.05, 0.10), (0.80, 0.00, 0.20), 0.44, 0.51, 0.47),
        ("Blend C2: 0.88/0.05/0.07 HS, 0.85/0.05/0.10 TR, 0.80/0.00/0.20 AG | thr 0.44/0.51/0.47",
         (0.88, 0.05, 0.07), (0.85, 0.05, 0.10), (0.80, 0.00, 0.20), 0.44, 0.51, 0.47),
        ("Blend C3: 0.88/0.05/0.07 HS, 0.85/0.05/0.10 TR, 0.85/0.00/0.15 AG | thr 0.44/0.51/0.465",
         (0.88, 0.05, 0.07), (0.85, 0.05, 0.10), (0.85, 0.00, 0.15), 0.44, 0.51, 0.465),
        ("Blend C4: 0.90/0.05/0.05 HS, 0.85/0.05/0.10 TR, 0.85/0.00/0.15 AG | thr 0.44/0.51/0.468",
         (0.90, 0.05, 0.05), (0.85, 0.05, 0.10), (0.85, 0.00, 0.15), 0.44, 0.51, 0.468),
        ("Blend C5: 0.85/0.05/0.10 HS, 0.85/0.05/0.10 TR, 0.80/0.00/0.20 AG | thr 0.435/0.51/0.47",
         (0.85, 0.05, 0.10), (0.85, 0.05, 0.10), (0.80, 0.00, 0.20), 0.435, 0.51, 0.47),
    ]

    for nom, w_hs, w_tr, w_ag, th_hs, th_tr, th_ag in test_configs:
        p_hs = w_hs[0] * prob_p114_te[:, 0] + w_hs[1] * prob_bm25_te[:, 0] + w_hs[2] * prob_svd_te[:, 0]
        p_tr = w_tr[0] * prob_p114_te[:, 1] + w_tr[1] * prob_bm25_te[:, 1] + w_tr[2] * prob_svd_te[:, 1]
        p_ag = w_ag[0] * prob_p114_te[:, 2] + w_ag[1] * prob_bm25_te[:, 2] + w_ag[2] * prob_svd_te[:, 2]
        
        raw = np.column_stack([
            (p_hs >= th_hs).astype(int),
            (p_tr >= th_tr).astype(int),
            (p_ag >= th_ag).astype(int),
        ])
        asc, n_asc = aplicar_regla_ascendente(raw)
        
        hs_c = asc[:, 0].sum()
        tr_c = asc[:, 1].sum()
        ag_c = asc[:, 2].sum()
        tot = hs_c + tr_c + ag_c
        diff = (asc != pred_p114).sum()
        
        print(f"\n{nom}:")
        print(f"  Positivos: HS={hs_c:3d}, TR={tr_c:3d}, AG={ag_c:3d} | Total: {tot} ({tot/(len(test_df)*3)*100:.2f}%)")
        print(f"  Diferencias vs P114: {diff} celdas ({diff/(len(test_df)*3)*100:.2f}%)")
        for j, et in enumerate(ETIQUETAS):
            add = ((asc[:, j] == 1) & (pred_p114[:, j] == 0)).sum()
            rem = ((asc[:, j] == 0) & (pred_p114[:, j] == 1)).sum()
            print(f"    {et}: {add+rem} diffs (+{add} / -{rem} | neto: {add-rem:+d})")


if __name__ == "__main__":
    main()
