#!/usr/bin/env python
"""Detailed exploration of SAGA ElasticNet + Liblinear + SGD on V2_triple for P114."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path("/home/carlos/Git/PLN/kaggle_toxic_2026_puj")
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

OOF_FINAL = RAIZ / "features" / "oof_v03_final.npz"
OOF_BASE = RAIZ / "features" / "v2_oof_base_corrected.npz"
IDS_PATH = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
SUB_P110 = RAIZ / "submissions" / "toxic_submission_p110_lr_sgd_blend_asc.csv"
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

def get_data():
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

    return train_full, train_dev, test_df, pliegues_can, pliegues_47, grupos

def evaluate_models(train_dev, pliegues, l1_ratio=0.02):
    textos = train_dev["text"]
    y_df = train_dev[list(ETIQUETAS)]
    n_samples = len(textos)
    n_labels = len(ETIQUETAS)
    
    prob_lr = np.zeros((n_samples, n_labels), dtype=float)
    prob_sgd = np.zeros((n_samples, n_labels), dtype=float)
    prob_saga = np.zeros((n_samples, n_labels), dtype=float)

    for fold in sorted(set(pliegues)):
        idx_tr = np.flatnonzero(pliegues != fold)
        idx_va = np.flatnonzero(pliegues == fold)

        mats_tr, mats_va = [], []
        for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
            v = TfidfVectorizer(
                analyzer=analyzer,
                ngram_range=ngram,
                min_df=2,
                sublinear_tf=sublinear,
                lowercase=True,
            )
            mats_tr.append(v.fit_transform(textos.iloc[idx_tr]))
            mats_va.append(v.transform(textos.iloc[idx_va]))
        X_tr = sparse.hstack(mats_tr).tocsr()
        X_va = sparse.hstack(mats_va).tocsr()

        for j, et in enumerate(ETIQUETAS):
            y_tr = y_df[et].iloc[idx_tr]
            
            # LR Liblinear
            clf_lr = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            clf_lr.fit(X_tr, y_tr)
            prob_lr[idx_va, j] = clf_lr.predict_proba(X_va)[:, 1]

            # SGD log_loss
            clf_sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
            clf_sgd.fit(X_tr, y_tr)
            prob_sgd[idx_va, j] = clf_sgd.predict_proba(X_va)[:, 1]

            # LR SAGA ElasticNet
            clf_saga = LogisticRegression(C=2.0, penalty="elasticnet", l1_ratio=l1_ratio, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
            clf_saga.fit(X_tr, y_tr)
            prob_saga[idx_va, j] = clf_saga.predict_proba(X_va)[:, 1]

    return prob_lr, prob_sgd, prob_saga

def compute_score(prob_mat, y_true):
    pred = np.column_stack([(prob_mat[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    pred_asc = aplicar_regla_ascendente(pred)
    met = metricas_multietiqueta(y_true, pred_asc)
    return met["f1_media_columnas"], met["f1_por_etiqueta"]

def main():
    train_full, train_dev, test_df, pliegues_can, pliegues_47, grupos = get_data()
    y_true = train_dev[list(ETIQUETAS)].to_numpy(dtype=int)

    base_can = 0.784998
    base_47 = 0.784318
    base_avg = (base_can + base_47) / 2
    print(f"P110 Baseline: Can={base_can:.6f}, 47={base_47:.6f}, Avg={base_avg:.6f}\n")

    for l1 in [0.01, 0.02, 0.05]:
        print(f"\n--- Testing SAGA with l1_ratio = {l1} ---")
        lr_can, sgd_can, saga_can = evaluate_models(train_dev, pliegues_can, l1_ratio=l1)
        lr_47, sgd_47, saga_47 = evaluate_models(train_dev, pliegues_47, l1_ratio=l1)

        for w_saga in [0.05, 0.10, 0.15, 0.20, 0.25]:
            w_sgd = 0.05
            w_lr = 1.0 - w_sgd - w_saga
            b_can = w_lr * lr_can + w_sgd * sgd_can + w_saga * saga_can
            b_47 = w_lr * lr_47 + w_sgd * sgd_47 + w_saga * saga_47

            sc_can, per_can = compute_score(b_can, y_true)
            sc_47, per_47 = compute_score(b_47, y_true)
            sc_avg = (sc_can + sc_47) / 2
            delta = sc_avg - base_avg
            print(f"  Blend: {w_lr:.2f}*LR + {w_sgd:.2f}*SGD + {w_saga:.2f}*SAGA(l1={l1}) | Can={sc_can:.6f} | 47={sc_47:.6f} | Avg={sc_avg:.6f} | Delta={delta:+.6f}")

            # Also check with T=1.03 and T=1.05
            eps = 1e-6
            for T in [1.02, 1.04]:
                z_can = logit(np.clip(b_can, eps, 1 - eps)) / T
                z_47 = logit(np.clip(b_47, eps, 1 - eps)) / T
                sc_t_can, _ = compute_score(expit(z_can), y_true)
                sc_t_47, _ = compute_score(expit(z_47), y_true)
                sc_t_avg = (sc_t_can + sc_t_47) / 2
                delta_t = sc_t_avg - base_avg
                print(f"     + Temp T={T:.2f}: Can={sc_t_can:.6f} | 47={sc_t_47:.6f} | Avg={sc_t_avg:.6f} | Delta={delta_t:+.6f}")

if __name__ == "__main__":
    main()
