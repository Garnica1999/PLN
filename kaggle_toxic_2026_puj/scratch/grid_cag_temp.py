#!/usr/bin/env python
"""Analisis rapido de grilla C_AG x T x max_df y proyeccion a Test."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedGroupKFold

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

    sub_p114 = pd.read_csv(SUB_P114)
    pred_p114_test = sub_p114[list(ETIQUETAS)].to_numpy(dtype=int)

    # Entrenar vectorizadores para V2_maxdf90 y V2_triple
    for label, max_df in [("V2_maxdf90", 0.90), ("V2_triple_base", None)]:
        print(f"\n=================== {label} ===================")
        bloques = [
            ("word", (1, 2), True, max_df),
            ("char_wb", (3, 5), True, max_df),
            ("char", (1, 6), True, max_df),
        ]

        # Precomputar OOF para diferentes C_AG
        # En train_dev:
        c_ag_vals = [0.8, 1.0, 1.2, 1.5]
        oof_por_cag = {c: {} for c in c_ag_vals}

        for nombre_part, pliegues in particiones:
            n_samples = len(train_dev)
            for c_ag in c_ag_vals:
                oof_por_cag[c_ag][nombre_part] = {
                    "lr": np.zeros((n_samples, 3), dtype=float),
                    "sgd": np.zeros((n_samples, 3), dtype=float),
                    "saga": np.zeros((n_samples, 3), dtype=float),
                }

            for fold in sorted(set(pliegues)):
                idx_tr = np.flatnonzero(pliegues != fold)
                idx_va = np.flatnonzero(pliegues == fold)

                mats_tr, mats_va = [], []
                for analyzer, ngram, sublinear, mdf in bloques:
                    kw = {"analyzer": analyzer, "ngram_range": ngram, "min_df": 2, "sublinear_tf": sublinear, "lowercase": True}
                    if mdf is not None:
                        kw["max_df"] = mdf
                    v = TfidfVectorizer(**kw)
                    mats_tr.append(v.fit_transform(train_dev["text"].iloc[idx_tr]))
                    mats_va.append(v.transform(train_dev["text"].iloc[idx_va]))
                X_tr = sparse.hstack(mats_tr).tocsr()
                X_va = sparse.hstack(mats_va).tocsr()

                for j, et in enumerate(ETIQUETAS):
                    y_tr = train_dev[et].iloc[idx_tr]

                    # SGD base
                    sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
                    sgd.fit(X_tr, y_tr)
                    p_sgd_va = sgd.predict_proba(X_va)[:, 1]

                    for c_ag in c_ag_vals:
                        c_val = {"HS": 2.0, "TR": 1.8, "AG": c_ag}[et]
                        lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
                        lr.fit(X_tr, y_tr)
                        p_lr_va = lr.predict_proba(X_va)[:, 1]

                        saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
                        saga.fit(X_tr, y_tr)
                        p_saga_va = saga.predict_proba(X_va)[:, 1]

                        d = oof_por_cag[c_ag][nombre_part]
                        d["lr"][idx_va, j] = p_lr_va
                        d["sgd"][idx_va, j] = p_sgd_va
                        d["saga"][idx_va, j] = p_saga_va

        # Evaluar grilla de C_AG y T
        eps = 1e-6
        mejores = []
        for c_ag in c_ag_vals:
            for temp in [1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08]:
                f1s = []
                for nombre_part, pliegues in particiones:
                    d = oof_por_cag[c_ag][nombre_part]
                    p_b = 0.80 * d["lr"] + 0.05 * d["sgd"] + 0.15 * d["saga"]
                    p_cal = expit(logit(np.clip(p_b, eps, 1 - eps)) / temp)
                    pred_raw = np.column_stack([(p_cal[:, j] >= UMBRALES_BASE[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
                    pred = aplicar_regla_ascendente(pred_raw)
                    m = metricas_multietiqueta(y_true_dev, pred)
                    f1s.append(m["f1_media_columnas"])
                media = np.mean(f1s)
                mejores.append((media, f1s[0], f1s[1], c_ag, temp))

        mejores.sort(key=lambda x: x[0], reverse=True)
        print("Top 5 combinaciones C_AG x T:")
        for media, can, p47, c_ag, temp in mejores[:5]:
            delta = media - 0.786446
            print(f"  C_AG={c_ag:.1f}, T={temp:.2f} -> Media: {media:.6f} (Can: {can:.6f}, P47: {p47:.6f}) Delta: {delta:+.6f}")

        # Proyectar el mejor en Test Set entrenado con 100% train (1.731 filas)
        mejor_media, mejor_can, mejor_p47, mejor_cag, mejor_temp = mejores[0]
        print(f"\nEntrenando mejor configuracion sobre 100% train ({label}): C_AG={mejor_cag}, T={mejor_temp}")
        mats_tr_full, mats_te_full = [], []
        for analyzer, ngram, sublinear, mdf in bloques:
            kw = {"analyzer": analyzer, "ngram_range": ngram, "min_df": 2, "sublinear_tf": sublinear, "lowercase": True}
            if mdf is not None:
                kw["max_df"] = mdf
            v = TfidfVectorizer(**kw)
            mats_tr_full.append(v.fit_transform(train_full["text"]))
            mats_te_full.append(v.transform(test_df["text"]))
        X_tr_full = sparse.hstack(mats_tr_full).tocsr()
        X_te_full = sparse.hstack(mats_te_full).tocsr()

        prob_test_lr = np.zeros((len(test_df), 3), dtype=float)
        prob_test_sgd = np.zeros((len(test_df), 3), dtype=float)
        prob_test_saga = np.zeros((len(test_df), 3), dtype=float)

        for j, et in enumerate(ETIQUETAS):
            y_tr_col = train_full[et]
            c_val = {"HS": 2.0, "TR": 1.8, "AG": mejor_cag}[et]

            lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr.fit(X_tr_full, y_tr_col)
            prob_test_lr[:, j] = lr.predict_proba(X_te_full)[:, 1]

            sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
            sgd.fit(X_tr_full, y_tr_col)
            prob_test_sgd[:, j] = sgd.predict_proba(X_te_full)[:, 1]

            saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
            saga.fit(X_tr_full, y_tr_col)
            prob_test_saga[:, j] = saga.predict_proba(X_te_full)[:, 1]

        p_b_test = 0.80 * prob_test_lr + 0.05 * prob_test_sgd + 0.15 * prob_test_saga
        p_cal_test = expit(logit(np.clip(p_b_test, eps, 1 - eps)) / mejor_temp)
        pred_raw_test = np.column_stack([(p_cal_test[:, j] >= UMBRALES_BASE[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
        pred_test = aplicar_regla_ascendente(pred_raw_test)

        diffs = np.sum(pred_test != pred_p114_test)
        print(f"Test Set: Celdas modificadas vs P114: {diffs} ({diffs/34.62:.2f}%)")
        print(f"Test Set Positivos: HS={pred_test[:,0].sum()}, TR={pred_test[:,1].sum()}, AG={pred_test[:,2].sum()}, Total={pred_test.sum()}")
        # Desglose de cambios (0->1 vs 1->0)
        c01 = np.sum((pred_p114_test == 0) & (pred_test == 1))
        c10 = np.sum((pred_p114_test == 1) & (pred_test == 0))
        print(f"Desglose de cambios vs P114: 0->1 (positivos anadidos): {c01}, 1->0 (positivos suprimidos): {c10}, Balance neto: {c01 - c10:+d}")

if __name__ == "__main__":
    main()
