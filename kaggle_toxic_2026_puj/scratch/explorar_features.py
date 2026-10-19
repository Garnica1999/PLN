#!/usr/bin/env python
"""Scratch script to explore feature representations for P114."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
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
UMBRALES_CONGELADOS = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

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

def evaluate_feature_set(train_dev, pliegues, bloques, strip_accents=None):
    textos = train_dev["text"]
    y_df = train_dev[list(ETIQUETAS)]
    n_samples = len(textos)
    n_labels = len(ETIQUETAS)
    
    prob_lr = np.zeros((n_samples, n_labels), dtype=float)
    prob_sgd = np.zeros((n_samples, n_labels), dtype=float)

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
            
            clf_lr = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            clf_lr.fit(X_tr, y_tr)
            prob_lr[idx_va, j] = clf_lr.predict_proba(X_va)[:, 1]

            clf_sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
            clf_sgd.fit(X_tr, y_tr)
            prob_sgd[idx_va, j] = clf_sgd.predict_proba(X_va)[:, 1]

    # Return P110-style blend (95% LR + 5% SGD)
    prob_blend = 0.95 * prob_lr + 0.05 * prob_sgd
    pred = np.column_stack([(prob_blend[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    pred_asc = aplicar_regla_ascendente(pred)
    met = metricas_multietiqueta(y_df.to_numpy(dtype=int), pred_asc)
    return met["f1_media_columnas"], met["f1_por_etiqueta"]

def main():
    train_full, train_dev, test_df, pliegues_can, pliegues_47, grupos = get_data()

    feature_sets = [
        ("Baseline V2_triple (word 1-2, char_wb 3-5, char 1-6)", [
            ("word", (1, 2), True),
            ("char_wb", (3, 5), True),
            ("char", (1, 6), True),
        ], None),
        ("F1: char_wb(2, 5) instead of (3, 5)", [
            ("word", (1, 2), True),
            ("char_wb", (2, 5), True),
            ("char", (1, 6), True),
        ], None),
        ("F2: char_wb(3, 6) instead of (3, 5)", [
            ("word", (1, 2), True),
            ("char_wb", (3, 6), True),
            ("char", (1, 6), True),
        ], None),
        ("F3: char_wb(2, 6) instead of (3, 5)", [
            ("word", (1, 2), True),
            ("char_wb", (2, 6), True),
            ("char", (1, 6), True),
        ], None),
        ("F4: word(1, 3) + char_wb(3, 5) + char(1, 6)", [
            ("word", (1, 3), True),
            ("char_wb", (3, 5), True),
            ("char", (1, 6), True),
        ], None),
        ("F5: char(2, 6) instead of (1, 6)", [
            ("word", (1, 2), True),
            ("char_wb", (3, 5), True),
            ("char", (2, 6), True),
        ], None),
        ("F6: char(1, 7) instead of (1, 6)", [
            ("word", (1, 2), True),
            ("char_wb", (3, 5), True),
            ("char", (1, 7), True),
        ], None),
        ("F7: strip_accents='unicode' on Baseline", [
            ("word", (1, 2), True),
            ("char_wb", (3, 5), True),
            ("char", (1, 6), True),
        ], "unicode"),
    ]

    base_can = 0.784998
    base_47 = 0.784318
    base_avg = (base_can + base_47) / 2

    print(f"P110 Baseline: Can={base_can:.6f}, 47={base_47:.6f}, Avg={base_avg:.6f}\n")
    for name, bloques, strip_accents in feature_sets:
        f1_can, per_can = evaluate_feature_set(train_dev, pliegues_can, bloques, strip_accents)
        f1_47, per_47 = evaluate_feature_set(train_dev, pliegues_47, bloques, strip_accents)
        f1_avg = (f1_can + f1_47) / 2
        delta = f1_avg - base_avg
        print(f"{name:55s} | Can={f1_can:.6f} | 47={f1_47:.6f} | Avg={f1_avg:.6f} | Delta={delta:+.6f}")
        print(f"      HS: Can={per_can['HS']:.4f}, 47={per_47['HS']:.4f} | TR: Can={per_can['TR']:.4f}, 47={per_47['TR']:.4f} | AG: Can={per_can['AG']:.4f}, 47={per_47['AG']:.4f}")

if __name__ == "__main__":
    main()
