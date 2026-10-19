#!/usr/bin/env python
"""scratch/test_bm25_and_label_specialization.py

Prueba rigurosa de:
1. BM25 vs TF-IDF sobre V2_triple
2. Ensamble Heterogeneo Especializado por Etiqueta (Label-Specific Decoupled Ensemble):
   - Modelos base:
     * LR LibLinear (Coordinate Descent)
     * SGD log_loss
     * SAGA ElasticNet (l1_ratio=0.05)
     * RidgeClassifier (Platt Scaled)
     * SVD(150) Dense Classifier
     * BM25 Classifier
   - Optimizacion de pesos por etiqueta independiente
3. Regla ontologica HatEval (TR=1 => HS=1)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from scipy.optimize import minimize

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

OOF_FINAL = RAIZ / "features" / "oof_v03_final.npz"
OOF_BASE = RAIZ / "features" / "v2_oof_base_corrected.npz"
IDS_PATH = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"

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
    print("=== CARGANDO DATOS Y PARTICIONES ===")
    train_full = data_module.read_train()
    zf = np.load(OOF_FINAL, allow_pickle=True)
    zb = np.load(OOF_BASE, allow_pickle=True)
    pliegues_can = zf["pliegues"].astype(int)

    particion = pd.read_csv(IDS_PATH)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    train_dev = train_full[train_full["id"].astype(int).isin(ids_dev)].reset_index(drop=True)

    textos = train_dev["text"]
    y = train_dev[list(ETIQUETAS)]
    y_true = y.to_numpy(dtype=int)
    n_samples = len(textos)

    print(f"Dev samples: {n_samples}")

    # Matrices OOF para almacenar predicciones de cada modelo
    # Modelos:
    # 0: LR LibLinear (Coordinate Descent L2)
    # 1: SGD log_loss
    # 2: SAGA ElasticNet (l1_ratio=0.05)
    # 3: Ridge Classifier Platt-Scaled (L2 loss)
    # 4: SVD(150) Dense Classifier
    # 5: BM25 Classifier
    n_modelos = 6
    n_labels = 3
    cache_path = RAIZ / "scratch" / "oof_6models.npz"

    if cache_path.exists():
        print(f"Cargando predicciones OOF cacheadas desde {cache_path}...")
        c_data = np.load(cache_path)
        oof_probs = c_data["oof_probs"]
    else:
        print("Calculando predicciones OOF de las 6 familias...")
        oof_probs = np.zeros((n_modelos, n_samples, n_labels), dtype=float)

        for fold in sorted(set(pliegues_can)):
            idx_tr = np.flatnonzero(pliegues_can != fold)
            idx_va = np.flatnonzero(pliegues_can == fold)

            # 1. TF-IDF V2_triple
            bloques = [
                ("word", (1, 2), True),
                ("char_wb", (3, 5), True),
                ("char", (1, 6), True),
            ]
            mats_tfidf_tr, mats_tfidf_va = [], []
            mats_cnt_tr, mats_cnt_va = [], []

            for analyzer, ngram, sublinear in bloques:
                v_tfidf = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
                mats_tfidf_tr.append(v_tfidf.fit_transform(textos.iloc[idx_tr]))
                mats_tfidf_va.append(v_tfidf.transform(textos.iloc[idx_va]))

                v_cnt = CountVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, lowercase=True)
                mats_cnt_tr.append(v_cnt.fit_transform(textos.iloc[idx_tr]))
                mats_cnt_va.append(v_cnt.transform(textos.iloc[idx_va]))

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

            for j, et in enumerate(ETIQUETAS):
                y_col = y[et].iloc[idx_tr]
                c_val = C_PER_LABEL[et]

                # 0: LR LibLinear
                lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
                lr.fit(X_tfidf_tr, y_col)
                oof_probs[0, idx_va, j] = lr.predict_proba(X_tfidf_va)[:, 1]

                # 1: SGD log_loss
                sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
                sgd.fit(X_tfidf_tr, y_col)
                oof_probs[1, idx_va, j] = sgd.predict_proba(X_tfidf_va)[:, 1]

                # 2: SAGA ElasticNet
                saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
                saga.fit(X_tfidf_tr, y_col)
                oof_probs[2, idx_va, j] = saga.predict_proba(X_tfidf_va)[:, 1]

                # 3: Ridge Platt Scaled
                ridge = RidgeClassifier(alpha=2.0, class_weight="balanced", random_state=42)
                cal_ridge = CalibratedClassifierCV(estimator=ridge, method="sigmoid", cv=3)
                cal_ridge.fit(X_tfidf_tr, y_col)
                oof_probs[3, idx_va, j] = cal_ridge.predict_proba(X_tfidf_va)[:, 1]

                # 4: SVD Dense LR
                lr_svd = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
                lr_svd.fit(X_svd_tr, y_col)
                oof_probs[4, idx_va, j] = lr_svd.predict_proba(X_svd_va)[:, 1]

                # 5: BM25 LR
                lr_bm25 = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
                lr_bm25.fit(X_bm25_tr, y_col)
                oof_probs[5, idx_va, j] = lr_bm25.predict_proba(X_bm25_va)[:, 1]

        np.savez(cache_path, oof_probs=oof_probs)
        print(f"Predicciones OOF guardadas en {cache_path}")

    # Evaluacion individual de cada familia
    nombres_modelos = ["LR_Liblinear", "SGD_log_loss", "SAGA_ElasticNet", "Ridge_Platt", "SVD_150_Dense", "BM25_Liblinear"]
    print("\n--- Evaluacion Individual por Familia (Umbrales Congelados + Regla Ascendente) ---")
    for m_idx, nom in enumerate(nombres_modelos):
        pr = oof_probs[m_idx]
        p_raw = np.column_stack([(pr[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
        p_asc, _ = aplicar_regla_ascendente(p_raw)
        met = metricas_multietiqueta(y_true, p_asc)
        print(f"  {nom:20s} CCV: {met['f1_media_columnas']:.6f} | HS: {met['f1_por_etiqueta']['HS']:.4f}, TR: {met['f1_por_etiqueta']['TR']:.4f}, AG: {met['f1_por_etiqueta']['AG']:.4f}")

    # Baseline P114 (80% LR + 5% SGD + 15% SAGA, T=1.04)
    p_p114_blend = 0.80 * oof_probs[0] + 0.05 * oof_probs[1] + 0.15 * oof_probs[2]
    eps = 1e-6
    p_p114_cal = expit(logit(np.clip(p_p114_blend, eps, 1 - eps)) / 1.04)
    p_p114_raw = np.column_stack([(p_p114_cal[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    p_p114_asc, _ = aplicar_regla_ascendente(p_p114_raw)
    met_p114 = metricas_multietiqueta(y_true, p_p114_asc)
    print(f"\n  P114 Baseline CCV: {met_p114['f1_media_columnas']:.6f} | HS: {met_p114['f1_por_etiqueta']['HS']:.4f}, TR: {met_p114['f1_por_etiqueta']['TR']:.4f}, AG: {met_p114['f1_por_etiqueta']['AG']:.4f}")

    # Exploracion de Blends Heterogeneos con BM25 y SVD
    print("\n--- Exploracion de Blends Heterogeneos Avanzados ---")
    blends = [
        # (Nombre, w_LR, w_SGD, w_SAGA, w_Ridge, w_SVD, w_BM25, Temp)
        ("P114 + BM25 (75/5/15/0/0/5, T=1.04)", [0.75, 0.05, 0.15, 0.0, 0.0, 0.05], 1.04),
        ("P114 + BM25 (70/5/15/0/0/10, T=1.04)", [0.70, 0.05, 0.15, 0.0, 0.0, 0.10], 1.04),
        ("P114 + BM25 (60/5/15/0/0/20, T=1.04)", [0.60, 0.05, 0.15, 0.0, 0.0, 0.20], 1.04),
        ("P114 + SVD (75/5/15/0/5/0, T=1.04)", [0.75, 0.05, 0.15, 0.0, 0.05, 0.0], 1.04),
        ("P114 + BM25 + SVD (70/5/15/0/5/5, T=1.04)", [0.70, 0.05, 0.15, 0.0, 0.05, 0.05], 1.04),
        ("P114 + BM25 + SVD (65/5/15/0/5/10, T=1.04)", [0.65, 0.05, 0.15, 0.0, 0.05, 0.10], 1.04),
        ("P114 + BM25 + SVD (60/5/15/0/10/10, T=1.04)", [0.60, 0.05, 0.15, 0.0, 0.10, 0.10], 1.04),
        ("P114 + Ridge + BM25 (65/5/15/5/0/10, T=1.04)", [0.65, 0.05, 0.15, 0.05, 0.0, 0.10], 1.04),
    ]

    for nom, w, temp in blends:
        w_arr = np.array(w)
        p_blend = np.tensordot(w_arr, oof_probs, axes=(0, 0))
        p_cal = expit(logit(np.clip(p_blend, eps, 1 - eps)) / temp)
        p_raw = np.column_stack([(p_cal[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
        p_asc, _ = aplicar_regla_ascendente(p_raw)
        m = metricas_multietiqueta(y_true, p_asc)
        delta = m['f1_media_columnas'] - met_p114['f1_media_columnas']
        print(f"  {nom:45s} CCV: {m['f1_media_columnas']:.6f} (Delta: {delta:+.6f}) | HS: {m['f1_por_etiqueta']['HS']:.4f}, TR: {m['f1_por_etiqueta']['TR']:.4f}, AG: {m['f1_por_etiqueta']['AG']:.4f}")

    # Ensamble con Pesos Desacoplados por Etiqueta (Label-Specific Tuning)
    print("\n--- Optimizacion de Pesos Desacoplados por Etiqueta (Grid Simplex) ---")
    # Buscamos los mejores pesos para cada etiqueta independientemente
    best_weights_per_label = {}
    best_probs_per_label = np.zeros((n_samples, n_labels), dtype=float)

    # Consideramos 3 componentes principales para evitar sobreajuste:
    # Comp A: P114 base (LR + SGD + SAGA)
    # Comp B: BM25 (LR)
    # Comp C: SVD (Dense LR)
    p114_comp = p_p114_cal
    bm25_comp = oof_probs[5]
    svd_comp = oof_probs[4]

    for j, et in enumerate(ETIQUETAS):
        y_col = y_true[:, j]
        thr = UMBRALES_CONGELADOS[et]
        best_f1 = -1.0
        best_w = None

        # Busqueda en rejilla de pesos: w_p114, w_bm25, w_svd sumando 1
        for w1 in np.linspace(0.60, 1.0, 9):
            for w2 in np.linspace(0.0, 1.0 - w1, int(round((1.0 - w1) * 20)) + 1):
                w3 = max(0.0, 1.0 - w1 - w2)
                p_cand = w1 * p114_comp[:, j] + w2 * bm25_comp[:, j] + w3 * svd_comp[:, j]
                pred_col = (p_cand >= thr).astype(int)
                f1_val = f1_score(y_col, pred_col, pos_label=1, zero_division=0)
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_w = (w1, w2, w3)

        best_weights_per_label[et] = best_w
        w1, w2, w3 = best_w
        best_probs_per_label[:, j] = w1 * p114_comp[:, j] + w2 * bm25_comp[:, j] + w3 * svd_comp[:, j]
        f1_base = f1_score(y_col, (p114_comp[:, j] >= thr).astype(int), pos_label=1)
        print(f"  {et}: Mejor peso (P114={w1:.2f}, BM25={w2:.2f}, SVD={w3:.2f}) -> F1: {best_f1:.6f} (Base P114: {f1_base:.6f}, Delta: {best_f1 - f1_base:+.6f})")

    # Evaluacion global del ensamble desacoplado con regla ascendente
    p_opt_raw = np.column_stack([(best_probs_per_label[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    p_opt_asc, n_asc = aplicar_regla_ascendente(p_opt_raw)
    met_opt = metricas_multietiqueta(y_true, p_opt_asc)
    delta_global = met_opt['f1_media_columnas'] - met_p114['f1_media_columnas']
    print(f"\n>>> Ensamble Heterogeneo Desacoplado CCV: {met_opt['f1_media_columnas']:.6f} (Delta vs P114: {delta_global:+.6f})")
    print(f"    Por etiqueta: HS={met_opt['f1_por_etiqueta']['HS']:.4f}, TR={met_opt['f1_por_etiqueta']['TR']:.4f}, AG={met_opt['f1_por_etiqueta']['AG']:.4f}")
    print(f"    Regla ascendente aplicada: {n_asc} veces")


if __name__ == "__main__":
    main()
