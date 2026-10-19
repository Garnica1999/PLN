#!/usr/bin/env python
"""Scratch script to systematically explore challenger hypotheses for P114."""

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

def aplicar_regla_ascendente(pred: np.ndarray) -> np.ndarray:
    p = pred.copy()
    mask = (p[:, 1] == 1) & (p[:, 0] == 0)
    p[mask, 0] = 1
    return p

def get_data_and_folds():
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

def evaluate_models_on_folds(textos, y_df, pliegues, bloques, strip_accents=None):
    n_samples = len(textos)
    n_labels = len(ETIQUETAS)
    
    prob_lr = np.zeros((n_samples, n_labels), dtype=float)
    prob_sgd = np.zeros((n_samples, n_labels), dtype=float)
    prob_lbfgs = np.zeros((n_samples, n_labels), dtype=float)
    prob_saga = np.zeros((n_samples, n_labels), dtype=float)

    for fold in sorted(set(pliegues)):
        idx_tr = np.flatnonzero(pliegues != fold)
        idx_va = np.flatnonzero(pliegues == fold)

        mats_tr, mats_va = [], []
        for analyzer, ngram, sublinear in bloques:
            v = TfidfVectorizer(
                analyzer=analyzer,
                ngram_range=ngram,
                min_df=2,
                sublinear_tf=sublinear,
                lowercase=True,
                strip_accents=strip_accents,
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

            # LR L-BFGS
            clf_lbfgs = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42)
            clf_lbfgs.fit(X_tr, y_tr)
            prob_lbfgs[idx_va, j] = clf_lbfgs.predict_proba(X_va)[:, 1]

            # LR SAGA ElasticNet (mild L1: l1_ratio=0.02)
            clf_saga = LogisticRegression(C=2.0, penalty="elasticnet", l1_ratio=0.02, solver="saga", max_iter=500, random_state=42, class_weight="balanced")
            clf_saga.fit(X_tr, y_tr)
            prob_saga[idx_va, j] = clf_saga.predict_proba(X_va)[:, 1]

    return prob_lr, prob_sgd, prob_lbfgs, prob_saga

def compute_score(prob_mat, y_true):
    pred = np.column_stack([(prob_mat[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    pred_asc = aplicar_regla_ascendente(pred)
    met = metricas_multietiqueta(y_true, pred_asc)
    return met["f1_media_columnas"], met["f1_por_etiqueta"]

def run_tests():
    train_full, train_dev, test_df, pliegues_can, pliegues_47, grupos = get_data_and_folds()
    y_true = train_dev[list(ETIQUETAS)].to_numpy(dtype=int)

    bloques_baseline = [
        ("word", (1, 2), True),
        ("char_wb", (3, 5), True),
        ("char", (1, 6), True),
    ]

    print("=== TEST 1: REPRODUCE BASELINE P110 ON CANONICA & PART47 ===")
    prob_lr_can, prob_sgd_can, prob_lbfgs_can, prob_saga_can = evaluate_models_on_folds(
        train_dev["text"], train_dev[list(ETIQUETAS)], pliegues_can, bloques_baseline
    )
    prob_lr_47, prob_sgd_47, prob_lbfgs_47, prob_saga_47 = evaluate_models_on_folds(
        train_dev["text"], train_dev[list(ETIQUETAS)], pliegues_47, bloques_baseline
    )

    # Baseline P110
    p110_can = 0.95 * prob_lr_can + 0.05 * prob_sgd_can
    p110_47 = 0.95 * prob_lr_47 + 0.05 * prob_sgd_47
    score_p110_can, f1_can = compute_score(p110_can, y_true)
    score_p110_47, f1_47 = compute_score(p110_47, y_true)
    score_p110_avg = (score_p110_can + score_p110_47) / 2
    print(f"P110 Baseline -> Canonica: {score_p110_can:.6f}, Part47: {score_p110_47:.6f}, Media: {score_p110_avg:.6f}")

    # Check individual models
    print("\nIndividual models on Canonica / Part47 / Media:")
    for name, p_can, p_47 in [
        ("LR liblinear", prob_lr_can, prob_lr_47),
        ("SGD log_loss", prob_sgd_can, prob_sgd_47),
        ("LR lbfgs", prob_lbfgs_can, prob_lbfgs_47),
        ("LR saga elasticnet", prob_saga_can, prob_saga_47),
    ]:
        sc_can, _ = compute_score(p_can, y_true)
        sc_47, _ = compute_score(p_47, y_true)
        sc_avg = (sc_can + sc_47) / 2
        print(f"  {name:20s}: Can={sc_can:.6f}, 47={sc_47:.6f}, Avg={sc_avg:.6f} (Delta vs P110: {sc_avg - score_p110_avg:+.6f})")

    # Test Smooth Blends
    print("\nTesting Smooth Multi-Solver Blends on V2_triple:")
    blends = [
        ("P110: 95% LR_lib + 5% SGD", 0.95, 0.05, 0.0, 0.0),
        ("90% LR_lib + 5% SGD + 5% L-BFGS", 0.90, 0.05, 0.05, 0.0),
        ("85% LR_lib + 5% SGD + 10% L-BFGS", 0.85, 0.05, 0.10, 0.0),
        ("80% LR_lib + 5% SGD + 15% L-BFGS", 0.80, 0.05, 0.15, 0.0),
        ("70% LR_lib + 5% SGD + 25% L-BFGS", 0.70, 0.05, 0.25, 0.0),
        ("50% LR_lib + 5% SGD + 45% L-BFGS", 0.50, 0.05, 0.45, 0.0),
        ("90% LR_lib + 5% SGD + 5% SAGA", 0.90, 0.05, 0.0, 0.05),
        ("80% LR_lib + 5% SGD + 15% SAGA", 0.80, 0.05, 0.0, 0.15),
        ("75% LR_lib + 5% SGD + 10% LBFGS + 10% SAGA", 0.75, 0.05, 0.10, 0.10),
    ]

    for label, w_lr, w_sgd, w_lbfgs, w_saga in blends:
        b_can = w_lr * prob_lr_can + w_sgd * prob_sgd_can + w_lbfgs * prob_lbfgs_can + w_saga * prob_saga_can
        b_47 = w_lr * prob_lr_47 + w_sgd * prob_sgd_47 + w_lbfgs * prob_lbfgs_47 + w_saga * prob_saga_47
        sc_can, _ = compute_score(b_can, y_true)
        sc_47, _ = compute_score(b_47, y_true)
        sc_avg = (sc_can + sc_47) / 2
        delta = sc_avg - score_p110_avg
        print(f"  {label:45s}: Can={sc_can:.6f}, 47={sc_47:.6f}, Avg={sc_avg:.6f} (Delta={delta:+.6f})")

    # Test Logit-Space Blending
    print("\nTesting Logit-space Blending vs Probability Blending:")
    eps = 1e-6
    z_lr_can = logit(np.clip(prob_lr_can, eps, 1 - eps))
    z_sgd_can = logit(np.clip(prob_sgd_can, eps, 1 - eps))
    z_lbfgs_can = logit(np.clip(prob_lbfgs_can, eps, 1 - eps))
    
    z_lr_47 = logit(np.clip(prob_lr_47, eps, 1 - eps))
    z_sgd_47 = logit(np.clip(prob_sgd_47, eps, 1 - eps))
    z_lbfgs_47 = logit(np.clip(prob_lbfgs_47, eps, 1 - eps))

    for w_sgd in [0.02, 0.04, 0.05, 0.06, 0.08]:
        z_blend_can = (1 - w_sgd) * z_lr_can + w_sgd * z_sgd_can
        z_blend_47 = (1 - w_sgd) * z_lr_47 + w_sgd * z_sgd_47
        sc_can, _ = compute_score(expit(z_blend_can), y_true)
        sc_47, _ = compute_score(expit(z_blend_47), y_true)
        sc_avg = (sc_can + sc_47) / 2
        print(f"  Logit blend (1-{w_sgd:.2f})*z_LR + {w_sgd:.2f}*z_SGD: Can={sc_can:.6f}, 47={sc_47:.6f}, Avg={sc_avg:.6f} (Delta={sc_avg - score_p110_avg:+.6f})")

    # Test Temperature Scaling
    print("\nTesting Temperature Scaling on Logits:")
    z_p110_can = logit(np.clip(p110_can, eps, 1 - eps))
    z_p110_47 = logit(np.clip(p110_47, eps, 1 - eps))
    for T in [0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10]:
        sc_can, _ = compute_score(expit(z_p110_can / T), y_true)
        sc_47, _ = compute_score(expit(z_p110_47 / T), y_true)
        sc_avg = (sc_can + sc_47) / 2
        print(f"  Temp T={T:.2f}: Can={sc_can:.6f}, 47={sc_47:.6f}, Avg={sc_avg:.6f} (Delta={sc_avg - score_p110_avg:+.6f})")

if __name__ == "__main__":
    run_tests()
