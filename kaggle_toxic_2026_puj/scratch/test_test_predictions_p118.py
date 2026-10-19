#!/usr/bin/env python
"""scratch/test_test_predictions_p118.py

Generacion y auditoria forense de las predicciones en Test Set (1.154 filas) del
Ensamble Heterogeneo Desacoplado Multimodal (P118):
1. Fit completo sobre 1.731 filas de train
2. Predict sobre 1.154 filas de test
3. Comparacion celda a celda contra P114 (0.79057) y P117 (0.78885)
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

UMBRALES_CONGELADOS = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
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
    print("=== AJUSTE COMPLETO Y PREDICCION EN TEST SET ===")
    train_full = data_module.read_train()
    test_df = data_module.read_test()

    text_tr = train_full["text"]
    text_te = test_df["text"]
    y_tr = train_full[list(ETIQUETAS)]

    print(f"Train samples: {len(train_full)} | Test samples: {len(test_df)}")

    # 1. Extraccion de features
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

    print(f"Dimensiones TF-IDF: {X_tfidf_tr.shape[1]}")

    # BM25
    bm25 = BM25Transformer(k1=1.2, b=0.6)
    bm25.fit(X_cnt_tr)
    X_bm25_tr = bm25.transform(X_cnt_tr)
    X_bm25_te = bm25.transform(X_cnt_te)

    # SVD
    svd = TruncatedSVD(n_components=150, random_state=42)
    X_svd_tr = svd.fit_transform(X_tfidf_tr)
    X_svd_te = svd.transform(X_tfidf_te)

    # Modelos sobre 100% train
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

    # Ensamble Desacoplado P118:
    prob_p118_te = np.zeros_like(prob_p114_te)
    prob_p118_te[:, 0] = 0.85 * prob_p114_te[:, 0] + 0.05 * prob_bm25_te[:, 0] + 0.10 * prob_svd_te[:, 0]
    prob_p118_te[:, 1] = 0.80 * prob_p114_te[:, 1] + 0.05 * prob_bm25_te[:, 1] + 0.15 * prob_svd_te[:, 1]
    prob_p118_te[:, 2] = 0.70 * prob_p114_te[:, 2] + 0.00 * prob_bm25_te[:, 2] + 0.30 * prob_svd_te[:, 2]

    # Aplicar umbrales
    raw_p118 = np.column_stack([(prob_p118_te[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    pred_p118, n_asc = aplicar_regla_ascendente(raw_p118)

    # Cargar P114 y P117 submissions
    sub_p114 = pd.read_csv(RAIZ / "submissions" / "toxic_submission_p114_tri_optimizer_smooth_blend_asc.csv")
    sub_p117 = pd.read_csv(RAIZ / "submissions" / "toxic_submission_p117_multifamily_nbsvm_subspace_asc.csv")
    pred_p114 = sub_p114[list(ETIQUETAS)].to_numpy(dtype=int)
    pred_p117 = sub_p117[list(ETIQUETAS)].to_numpy(dtype=int)

    # Auditoria de Positivos
    hs_cnt = pred_p118[:, 0].sum()
    tr_cnt = pred_p118[:, 1].sum()
    ag_cnt = pred_p118[:, 2].sum()
    tot_cnt = hs_cnt + tr_cnt + ag_cnt

    print("\n--- Distribucion de Positivos Predichos en Test (1.154 filas) ---")
    print(f"P114 (LB 0.79057): HS={pred_p114[:, 0].sum()}, TR={pred_p114[:, 1].sum()}, AG={pred_p114[:, 2].sum()} | Total: {pred_p114.sum()} (36.54%)")
    print(f"P117 (LB 0.78885): HS={pred_p117[:, 0].sum()}, TR={pred_p117[:, 1].sum()}, AG={pred_p117[:, 2].sum()} | Total: {pred_p117.sum()} (36.48%)")
    print(f"P118 (Candidato) : HS={hs_cnt}, TR={tr_cnt}, AG={ag_cnt} | Total: {tot_cnt} ({tot_cnt/(len(test_df)*3)*100:.2f}%)")
    print(f"Reglas ontologicas aplicadas: {n_asc}")

    # Comparacion celda a celda vs P114
    diff_mask = (pred_p118 != pred_p114)
    celdas_distintas = diff_mask.sum()
    print(f"\nCeldas distintas respecto a P114: {celdas_distintas} ({celdas_distintas/(len(test_df)*3)*100:.2f}%)")

    for j, et in enumerate(ETIQUETAS):
        col_diff = (pred_p118[:, j] != pred_p114[:, j])
        add = ((pred_p118[:, j] == 1) & (pred_p114[:, j] == 0)).sum()
        rem = ((pred_p118[:, j] == 0) & (pred_p114[:, j] == 1)).sum()
        print(f"  {et}: {col_diff.sum()} diffs (+{add} adiciones 0->1, -{rem} supresiones 1->0 | neto: {add - rem:+d})")

    # Muestras cualitativas de cambios
    print("\n--- Muestras Cualitativas de Celdas Modificadas ---")
    filas_mod = np.where(diff_mask.any(axis=1))[0]
    for idx in filas_mod[:8]:
        t_id = test_df["id"].iloc[idx]
        t_txt = test_df["text"].iloc[idx][:70].replace("\n", " ")
        p114_str = f"HS={pred_p114[idx, 0]}, TR={pred_p114[idx, 1]}, AG={pred_p114[idx, 2]}"
        p118_str = f"HS={pred_p118[idx, 0]}, TR={pred_p118[idx, 1]}, AG={pred_p118[idx, 2]}"
        print(f"  Fila {idx:4d} (id {t_id}): '{t_txt}...'")
        print(f"    P114: [{p114_str}] -> P118: [{p118_str}]")


if __name__ == "__main__":
    main()
