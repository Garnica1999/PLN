#!/usr/bin/env python
"""Exploracion P116: Optimizacion de Ensamble Continuo, MaxDF y Calibracion."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedGroupKFold
from scipy.optimize import minimize

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

OOF_FINAL = RAIZ / "features" / "oof_v03_final.npz"
OOF_BASE = RAIZ / "features" / "v2_oof_base_corrected.npz"
IDS_PATH = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
SUB_P114 = RAIZ / "submissions" / "toxic_submission_p114_tri_optimizer_smooth_blend_asc.csv"

UMBRALES_BASE = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

def aplicar_regla_ascendente(pred: np.ndarray) -> np.ndarray:
    p = pred.copy()
    mask = (p[:, 1] == 1) & (p[:, 0] == 0)
    p[mask, 0] = 1
    return p

def main():
    print("=== EXPLORACION P116: CONTINUO, MAX_DF, SIMPLEX Y CALIBRACION ===")
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

    particiones = [("Canonica", pliegues_can), ("Part47", pliegues_47)]

    # Cargar P114
    sub_p114 = pd.read_csv(SUB_P114)
    pred_p114_test = sub_p114[list(ETIQUETAS)].to_numpy(dtype=int)

    # Bloques base
    bloques_base = [
        ("word", (1, 2), True, None),
        ("char_wb", (3, 5), True, None),
        ("char", (1, 6), True, None),
    ]

    bloques_maxdf90 = [
        ("word", (1, 2), True, 0.90),
        ("char_wb", (3, 5), True, 0.90),
        ("char", (1, 6), True, 0.90),
    ]

    for label_espacio, bloques in [("V2_triple (Base P114)", bloques_base), ("V2_maxdf90", bloques_maxdf90)]:
        print(f"\n==========================================")
        print(f"Evaluando modelos base en: {label_espacio}")
        print(f"==========================================")

        # Precomputar probabilidades OOF
        # Modelos a extraer:
        # 0: LR LibLinear (C={HS:2.0, TR:1.8, AG:1.5})
        # 1: SGD log_loss (alpha=5e-6)
        # 2: SAGA ElasticNet (l1=0.05)
        # 3: SAGA ElasticNet (l1=0.10)
        # 4: ComplementNB
        # 5: LR LibLinear con C_AG=1.0

        n_samples = len(train_dev)
        oof_dict = {}

        for nombre_part, pliegues in particiones:
            prob_lr = np.zeros((n_samples, 3), dtype=float)
            prob_sgd = np.zeros((n_samples, 3), dtype=float)
            prob_saga05 = np.zeros((n_samples, 3), dtype=float)
            prob_saga10 = np.zeros((n_samples, 3), dtype=float)
            prob_cnb = np.zeros((n_samples, 3), dtype=float)
            prob_lr_cag1 = np.zeros((n_samples, 3), dtype=float)

            for fold in sorted(set(pliegues)):
                idx_tr = np.flatnonzero(pliegues != fold)
                idx_va = np.flatnonzero(pliegues == fold)

                mats_tr, mats_va = [], []
                for analyzer, ngram, sublinear, max_df in bloques:
                    kw = {"analyzer": analyzer, "ngram_range": ngram, "min_df": 2, "sublinear_tf": sublinear, "lowercase": True}
                    if max_df is not None:
                        kw["max_df"] = max_df
                    v = TfidfVectorizer(**kw)
                    mats_tr.append(v.fit_transform(train_dev["text"].iloc[idx_tr]))
                    mats_va.append(v.transform(train_dev["text"].iloc[idx_va]))
                X_tr = sparse.hstack(mats_tr).tocsr()
                X_va = sparse.hstack(mats_va).tocsr()

                for j, et in enumerate(ETIQUETAS):
                    y_tr = train_dev[et].iloc[idx_tr]
                    c_val = {"HS": 2.0, "TR": 1.8, "AG": 1.5}[et]

                    # LR base
                    lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
                    lr.fit(X_tr, y_tr)
                    prob_lr[idx_va, j] = lr.predict_proba(X_va)[:, 1]

                    # SGD log_loss
                    sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
                    sgd.fit(X_tr, y_tr)
                    prob_sgd[idx_va, j] = sgd.predict_proba(X_va)[:, 1]

                    # SAGA l1=0.05
                    saga05 = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
                    saga05.fit(X_tr, y_tr)
                    prob_saga05[idx_va, j] = saga05.predict_proba(X_va)[:, 1]

                    # SAGA l1=0.10
                    saga10 = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.10, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
                    saga10.fit(X_tr, y_tr)
                    prob_saga10[idx_va, j] = saga10.predict_proba(X_va)[:, 1]

                    # ComplementNB
                    cnb = ComplementNB(alpha=1.0)
                    cnb.fit(X_tr, y_tr)
                    prob_cnb[idx_va, j] = cnb.predict_proba(X_va)[:, 1]

                    # LR C_AG=1.0
                    c_val_cag1 = {"HS": 2.0, "TR": 1.8, "AG": 1.0}[et]
                    lr_cag1 = LogisticRegression(C=c_val_cag1, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
                    lr_cag1.fit(X_tr, y_tr)
                    prob_lr_cag1[idx_va, j] = lr_cag1.predict_proba(X_va)[:, 1]

            oof_dict[nombre_part] = {
                "lr": prob_lr,
                "sgd": prob_sgd,
                "saga05": prob_saga05,
                "saga10": prob_saga10,
                "cnb": prob_cnb,
                "lr_cag1": prob_lr_cag1,
            }

        eps = 1e-6
        # Evaluar variantes de ensamble
        configuraciones = [
            ("P114 Base (80% LR + 5% SGD + 15% SAGA05, T=1.04)",
             lambda d: 0.80 * d["lr"] + 0.05 * d["sgd"] + 0.15 * d["saga05"], 1.04),
            ("P116-c1 (75% LR + 5% SGD + 20% SAGA05, T=1.04)",
             lambda d: 0.75 * d["lr"] + 0.05 * d["sgd"] + 0.20 * d["saga05"], 1.04),
            ("P116-c2 (70% LR + 5% SGD + 25% SAGA05, T=1.04)",
             lambda d: 0.70 * d["lr"] + 0.05 * d["sgd"] + 0.25 * d["saga05"], 1.04),
            ("P116-c3 (80% LR + 5% SGD + 15% SAGA10, T=1.04)",
             lambda d: 0.80 * d["lr"] + 0.05 * d["sgd"] + 0.15 * d["saga10"], 1.04),
            ("P116-c4 (75% LR + 5% SGD + 10% SAGA05 + 10% SAGA10, T=1.04)",
             lambda d: 0.75 * d["lr"] + 0.05 * d["sgd"] + 0.10 * d["saga05"] + 0.10 * d["saga10"], 1.04),
            ("P116-c5 (78% LR + 5% SGD + 15% SAGA05 + 2% CNB, T=1.04)",
             lambda d: 0.78 * d["lr"] + 0.05 * d["sgd"] + 0.15 * d["saga05"] + 0.02 * d["cnb"], 1.04),
            ("P116-c6 (80% LR_cag1 + 5% SGD + 15% SAGA05, T=1.04)",
             lambda d: 0.80 * d["lr_cag1"] + 0.05 * d["sgd"] + 0.15 * d["saga05"], 1.04),
            ("P116-c7 (80% LR + 5% SGD + 15% SAGA05, T=1.02)",
             lambda d: 0.80 * d["lr"] + 0.05 * d["sgd"] + 0.15 * d["saga05"], 1.02),
            ("P116-c8 (80% LR + 5% SGD + 15% SAGA05, T=1.06)",
             lambda d: 0.80 * d["lr"] + 0.05 * d["sgd"] + 0.15 * d["saga05"], 1.06),
        ]

        print(f"\nResultados para {label_espacio}:")
        for desc, blend_fn, temp in configuraciones:
            f1s = []
            for nombre_part, pliegues in particiones:
                d = oof_dict[nombre_part]
                p_b = blend_fn(d)
                p_cal = expit(logit(np.clip(p_b, eps, 1 - eps)) / temp)
                pred_raw = np.column_stack([(p_cal[:, j] >= UMBRALES_BASE[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
                pred = aplicar_regla_ascendente(pred_raw)
                m = metricas_multietiqueta(y_true_dev, pred)
                f1s.append(m["f1_media_columnas"])
            media = np.mean(f1s)
            delta = media - 0.786446
            print(f"  {desc} -> Can: {f1s[0]:.6f}, P47: {f1s[1]:.6f}, Media: {media:.6f} (Delta vs P114: {delta:+.6f})")

if __name__ == "__main__":
    main()
