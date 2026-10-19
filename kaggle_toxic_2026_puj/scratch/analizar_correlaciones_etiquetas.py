#!/usr/bin/env python
"""Analyze multi-label dependency structure to see if 2nd-stage joint smoothing works."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier

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

    prob_p110 = np.zeros((len(train_dev), 3), dtype=float)
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
            sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
            sgd.fit(X_tr, y_col)
            prob_p110[idx_va, j] = 0.95 * lr.predict_proba(X_va)[:, 1] + 0.05 * sgd.predict_proba(X_va)[:, 1]

    # Baseline P110 F1
    pred_raw = np.column_stack([(prob_p110[:, j] >= [0.44, 0.51, 0.46][j]).astype(int) for j in range(3)])
    pred_p110 = aplicar_regla_ascendente(pred_raw)
    met_base = metricas_multietiqueta(y_true, pred_p110)
    print(f"Base P110 F1 macro: {met_base['f1_media_columnas']:.6f}")
    print(f"  HS: {met_base['f1_por_etiqueta']['HS']:.4f}, TR: {met_base['f1_por_etiqueta']['TR']:.4f}, AG: {met_base['f1_por_etiqueta']['AG']:.4f}")

    # Test soft label coupling:
    # 1. Soft Ascending Coupling:
    # In P110, we do hard rule: if pred_TR == 1 => pred_HS = 1.
    # What if in probability space: prob_HS_new = max(prob_HS, prob_TR)?
    prob_coupled = prob_p110.copy()
    prob_coupled[:, 0] = np.maximum(prob_coupled[:, 0], prob_coupled[:, 1])
    pred_c_raw = np.column_stack([(prob_coupled[:, j] >= [0.44, 0.51, 0.46][j]).astype(int) for j in range(3)])
    pred_c = aplicar_regla_ascendente(pred_c_raw)
    met_c = metricas_multietiqueta(y_true, pred_c)
    print(f"Soft ascending prob_HS = max(prob_HS, prob_TR): {met_c['f1_media_columnas']:.6f} (Delta={met_c['f1_media_columnas'] - met_base['f1_media_columnas']:+.6f})")

    # 2. Mutual Information Coupling:
    # What if prob_TR helps prob_HS when prob_TR is high:
    for w in [0.05, 0.10, 0.15, 0.20]:
        prob_blend_hs = prob_p110.copy()
        prob_blend_hs[:, 0] = (1 - w) * prob_blend_hs[:, 0] + w * prob_blend_hs[:, 1]
        pred_w_raw = np.column_stack([(prob_blend_hs[:, j] >= [0.44, 0.51, 0.46][j]).astype(int) for j in range(3)])
        pred_w = aplicar_regla_ascendente(pred_w_raw)
        met_w = metricas_multietiqueta(y_true, pred_w)
        print(f"  w={w:.2f} blend HS with TR: {met_w['f1_media_columnas']:.6f} (Delta={met_w['f1_media_columnas'] - met_base['f1_media_columnas']:+.6f})")

    # 3. What if AG probability helps HS?
    for w in [0.02, 0.05, 0.08]:
        prob_blend_ag = prob_p110.copy()
        prob_blend_ag[:, 0] = (1 - w) * prob_blend_ag[:, 0] + w * prob_blend_ag[:, 2]
        pred_w_raw = np.column_stack([(prob_blend_ag[:, j] >= [0.44, 0.51, 0.46][j]).astype(int) for j in range(3)])
        pred_w = aplicar_regla_ascendente(pred_w_raw)
        met_w = metricas_multietiqueta(y_true, pred_w)
        print(f"  w={w:.2f} blend HS with AG: {met_w['f1_media_columnas']:.6f} (Delta={met_w['f1_media_columnas'] - met_base['f1_media_columnas']:+.6f})")

if __name__ == "__main__":
    main()
