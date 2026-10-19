#!/usr/bin/env python
"""Analyze OOF errors of P110 on Canonica dev set (1,483 rows)."""

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

    n_samples = len(train_dev)
    prob_p110 = np.zeros((n_samples, len(ETIQUETAS)), dtype=float)

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
            p_lr = lr.predict_proba(X_va)[:, 1]

            sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
            sgd.fit(X_tr, y_col)
            p_sgd = sgd.predict_proba(X_va)[:, 1]

            prob_p110[idx_va, j] = 0.95 * p_lr + 0.05 * p_sgd

    pred_raw = np.column_stack([(prob_p110[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    pred_p110 = aplicar_regla_ascendente(pred_raw)

    print("=== P110 ERROR ANALYSIS ON CANONICAL DEV (1,483 rows) ===")
    for j, et in enumerate(ETIQUETAS):
        thr = UMBRALES_CONGELADOS[et]
        y_t = y_true[:, j]
        p_c = pred_p110[:, j]
        probs = prob_p110[:, j]

        tp = int(((p_c == 1) & (y_t == 1)).sum())
        fp = int(((p_c == 1) & (y_t == 0)).sum())
        fn = int(((p_c == 0) & (y_t == 1)).sum())
        tn = int(((p_c == 0) & (y_t == 0)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn)

        print(f"\nLabel {et} (thr={thr}): F1={f1:.4f} | TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        # Margin errors: borderline cases
        border_fp = int(((p_c == 1) & (y_t == 0) & (probs >= thr) & (probs < thr + 0.08)).sum())
        border_fn = int(((p_c == 0) & (y_t == 1) & (probs < thr) & (probs >= thr - 0.08)).sum())
        print(f"  Borderline FPs (prob in [{thr:.2f}, {thr+0.08:.2f}]): {border_fp} of {fp} ({border_fp/fp*100:.1f}%)")
        print(f"  Borderline FNs (prob in [{thr-0.08:.2f}, {thr:.2f}]): {border_fn} of {fn} ({border_fn/fn*100:.1f}%)")

    # Let's inspect correlation between HS, TR, AG probabilities and errors
    # In HatEval ontology:
    # TR implies HS (TR=1 => HS=1)
    # AG is aggressiveness. Does AG=1 correlate with HS or TR?
    # What about AG=1 and TR=1?
    print("\n--- Interaction patterns ---")
    tr_and_ag_both_1 = int(((pred_p110[:, 1] == 1) & (pred_p110[:, 2] == 1)).sum())
    print(f"Pred TR=1 and AG=1: {tr_and_ag_both_1}")
    
    # What if we examine conditional logic:
    # If TR=1 and AG=1 => HS is virtually 100% positive!
    # Does P110 ever predict TR=1, AG=1 and HS=0? The ascending rule already forces HS=1 when TR=1!
    
    # What about the reverse or mutual support?
    # What if HS=1 and AG=1: what is TR?
    # In train set:
    hs1_ag1 = (train_dev["HS"] == 1) & (train_dev["AG"] == 1)
    print(f"Train rows with HS=1 and AG=1: {hs1_ag1.sum()}")
    print(f"Among HS=1 & AG=1, TR=1 rate: {train_dev.loc[hs1_ag1, 'TR'].mean():.4f}")
    
    # What about when HS=0?
    hs0 = (train_dev["HS"] == 0)
    print(f"Train rows with HS=0: {hs0.sum()}")
    print(f"Among HS=0, TR=1 rate: {train_dev.loc[hs0, 'TR'].mean():.4f} (must be 0 by definition!)")
    print(f"Among HS=0, AG=1 rate: {train_dev.loc[hs0, 'AG'].mean():.4f}")

if __name__ == "__main__":
    main()
