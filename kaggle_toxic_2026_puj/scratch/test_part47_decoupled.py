#!/usr/bin/env python
"""scratch/test_part47_decoupled.py

Evaluacion de la generalizacion en Particion 47 del Ensamble Heterogeneo Desacoplado:
1. Extraer features V2_triple, BM25 y SVD(150) en Particion 47 (1.483 filas dev_p47)
2. Evaluar P114 baseline en Particion 47
3. Evaluar Ensamble Heterogeneo Desacoplado con los mismos pesos:
   - HS: 0.85 P114 + 0.05 BM25 + 0.10 SVD
   - TR: 0.80 P114 + 0.05 BM25 + 0.15 SVD
   - AG: 0.70 P114 + 0.00 BM25 + 0.30 SVD
4. Aplicar regla ontologica HatEval (TR=1 => HS=1)
5. Medir CCV en Particion 47, deltas por etiqueta y pliegues ganados
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
from src.evaluation import ETIQUETAS, metricas_multietiqueta

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
    print("=== EVALUACION EN PARTICION 47 ===")
    from sklearn.model_selection import StratifiedGroupKFold

    train_full = data_module.read_train()
    textos = train_full["text"]
    y = train_full[list(ETIQUETAS)]
    y_true = y.to_numpy(dtype=int)
    n_samples = len(textos)

    # Reconstruccion de Particion 47 de 5 pliegues sobre 1.483 filas dev
    zb = np.load(RAIZ / "features" / "v2_oof_base_corrected.npz", allow_pickle=True)
    grupos_full = zb["grupos"].astype(np.int64)

    particion = pd.read_csv(RAIZ / "data" / "processed" / "holdout_v2_ids.csv")
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    train_dev = train_full[train_full["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    textos_dev = train_dev["text"]
    y_dev = train_dev[list(ETIQUETAS)]
    y_dev_arr = y_dev.to_numpy(dtype=int)
    grupos_dev = grupos_full

    # Particion 47 oficial (random_state=43 sobre dev con grupos)
    combinacion = np.array([f"{a}{b}{c}" for a, b, c in y_dev_arr])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
    pliegues_p47 = np.full(len(train_dev), -1, dtype=int)
    for num, (_, val) in enumerate(sgkf.split(np.arange(len(train_dev)).reshape(-1, 1), combinacion, groups=grupos_dev)):
        pliegues_p47[val] = num + 1

    print(f"Dev Particion 47: {len(train_dev)} filas | 5 pliegues creados")

    n_dev = len(train_dev)
    prob_p114 = np.zeros((n_dev, 3), dtype=float)
    prob_bm25 = np.zeros((n_dev, 3), dtype=float)
    prob_svd = np.zeros((n_dev, 3), dtype=float)

    bloques = [
        ("word", (1, 2), True),
        ("char_wb", (3, 5), True),
        ("char", (1, 6), True),
    ]

    for fold in sorted(set(pliegues_p47)):
        idx_tr = np.flatnonzero(pliegues_p47 != fold)
        idx_va = np.flatnonzero(pliegues_p47 == fold)

        mats_tfidf_tr, mats_tfidf_va = [], []
        mats_cnt_tr, mats_cnt_va = [], []

        for analyzer, ngram, sublinear in bloques:
            v_tfidf = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
            mats_tfidf_tr.append(v_tfidf.fit_transform(textos_dev.iloc[idx_tr]))
            mats_tfidf_va.append(v_tfidf.transform(textos_dev.iloc[idx_va]))

            v_cnt = CountVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, lowercase=True)
            mats_cnt_tr.append(v_cnt.fit_transform(textos_dev.iloc[idx_tr]))
            mats_cnt_va.append(v_cnt.transform(textos_dev.iloc[idx_va]))

        X_tfidf_tr = sparse.hstack(mats_tfidf_tr).tocsr()
        X_tfidf_va = sparse.hstack(mats_tfidf_va).tocsr()

        X_cnt_tr = sparse.hstack(mats_cnt_tr).tocsr()
        X_cnt_va = sparse.hstack(mats_cnt_va).tocsr()

        # BM25 transform
        bm25 = BM25Transformer(k1=1.2, b=0.6)
        bm25.fit(X_cnt_tr)
        X_bm25_tr = bm25.transform(X_cnt_tr)
        X_bm25_va = bm25.transform(X_cnt_va)

        # SVD transform
        svd = TruncatedSVD(n_components=150, random_state=42)
        X_svd_tr = svd.fit_transform(X_tfidf_tr)
        X_svd_va = svd.transform(X_tfidf_va)

        prob_lr_f = np.zeros((len(idx_va), 3), dtype=float)
        prob_sgd_f = np.zeros((len(idx_va), 3), dtype=float)
        prob_saga_f = np.zeros((len(idx_va), 3), dtype=float)

        for j, et in enumerate(ETIQUETAS):
            y_col = y_dev[et].iloc[idx_tr]
            c_val = C_PER_LABEL[et]

            # LR Liblinear
            lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr.fit(X_tfidf_tr, y_col)
            prob_lr_f[:, j] = lr.predict_proba(X_tfidf_va)[:, 1]

            # SGD log_loss
            sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
            sgd.fit(X_tfidf_tr, y_col)
            prob_sgd_f[:, j] = sgd.predict_proba(X_tfidf_va)[:, 1]

            # SAGA ElasticNet
            saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
            saga.fit(X_tfidf_tr, y_col)
            prob_saga_f[:, j] = saga.predict_proba(X_tfidf_va)[:, 1]

            # BM25
            lr_bm25 = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr_bm25.fit(X_bm25_tr, y_col)
            prob_bm25[idx_va, j] = lr_bm25.predict_proba(X_bm25_va)[:, 1]

            # SVD
            lr_svd = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
            lr_svd.fit(X_svd_tr, y_col)
            prob_svd[idx_va, j] = lr_svd.predict_proba(X_svd_va)[:, 1]

        # P114 blend
        blend_f = 0.80 * prob_lr_f + 0.05 * prob_sgd_f + 0.15 * prob_saga_f
        eps = 1e-6
        prob_p114[idx_va] = expit(logit(np.clip(blend_f, eps, 1 - eps)) / 1.04)

    # 1. Metricas P114 en Particion 47
    p114_raw = np.column_stack([(prob_p114[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    p114_asc, _ = aplicar_regla_ascendente(p114_raw)
    met_p114 = metricas_multietiqueta(y_dev_arr, p114_asc)
    print(f"\n--- Particion 47 Baseline P114 ---")
    print(f"CCV P114: {met_p114['f1_media_columnas']:.6f} | HS: {met_p114['f1_por_etiqueta']['HS']:.4f}, TR: {met_p114['f1_por_etiqueta']['TR']:.4f}, AG: {met_p114['f1_por_etiqueta']['AG']:.4f}")

    # 2. Metricas Ensamble Heterogeneo Desacoplado en Particion 47
    # Pesos fijados en Canonica:
    # HS: 0.85 P114 + 0.05 BM25 + 0.10 SVD
    # TR: 0.80 P114 + 0.05 BM25 + 0.15 SVD
    # AG: 0.70 P114 + 0.00 BM25 + 0.30 SVD
    prob_decoupled = np.zeros_like(prob_p114)
    prob_decoupled[:, 0] = 0.85 * prob_p114[:, 0] + 0.05 * prob_bm25[:, 0] + 0.10 * prob_svd[:, 0]
    prob_decoupled[:, 1] = 0.80 * prob_p114[:, 1] + 0.05 * prob_bm25[:, 1] + 0.15 * prob_svd[:, 1]
    prob_decoupled[:, 2] = 0.70 * prob_p114[:, 2] + 0.00 * prob_bm25[:, 2] + 0.30 * prob_svd[:, 2]

    dec_raw = np.column_stack([(prob_decoupled[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    dec_asc, n_asc = aplicar_regla_ascendente(dec_raw)
    met_dec = metricas_multietiqueta(y_dev_arr, dec_asc)

    delta_ccv = met_dec['f1_media_columnas'] - met_p114['f1_media_columnas']
    print(f"\n--- Particion 47 Ensamble Heterogeneo Desacoplado ---")
    print(f"CCV Desacoplado: {met_dec['f1_media_columnas']:.6f} (Delta: {delta_ccv:+.6f})")
    print(f"  HS: {met_dec['f1_por_etiqueta']['HS']:.4f} (Delta: {met_dec['f1_por_etiqueta']['HS'] - met_p114['f1_por_etiqueta']['HS']:+.4f})")
    print(f"  TR: {met_dec['f1_por_etiqueta']['TR']:.4f} (Delta: {met_dec['f1_por_etiqueta']['TR'] - met_p114['f1_por_etiqueta']['TR']:+.4f})")
    print(f"  AG: {met_dec['f1_por_etiqueta']['AG']:.4f} (Delta: {met_dec['f1_por_etiqueta']['AG'] - met_p114['f1_por_etiqueta']['AG']:+.4f})")

    # Pliegues ganados
    ganados = 0
    for f in range(5):
        idx = np.flatnonzero(pliegues_p47 == f)
        f1_p = metricas_multietiqueta(y_dev_arr[idx], p114_asc[idx])["f1_media_columnas"]
        f1_d = metricas_multietiqueta(y_dev_arr[idx], dec_asc[idx])["f1_media_columnas"]
        if f1_d > f1_p:
            ganados += 1
        print(f"  Pliegue {f}: P114={f1_p:.6f} vs Desacoplado={f1_d:.6f} (Delta: {f1_d - f1_p:+.6f})")
    print(f"Pliegues ganados: {ganados}/5")


if __name__ == "__main__":
    main()
