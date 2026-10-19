#!/usr/bin/env python
"""Evaluacion detallada de proyecciones en Test y CV para V2_maxdf90 con C_AG y T."""

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

    bloques = [
        ("word", (1, 2), True, 0.90),
        ("char_wb", (3, 5), True, 0.90),
        ("char", (1, 6), True, 0.90),
    ]

    # Precomputar matrices train_dev
    mats_dev_tr_dict = {}
    n_samples = len(train_dev)

    c_ag_list = [1.0, 1.2, 1.5]
    oof_por_cag = {c: {} for c in c_ag_list}

    for nombre_part, pliegues in particiones:
        for c_ag in c_ag_list:
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
                v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, max_df=mdf, lowercase=True)
                mats_tr.append(v.fit_transform(train_dev["text"].iloc[idx_tr]))
                mats_va.append(v.transform(train_dev["text"].iloc[idx_va]))
            X_tr = sparse.hstack(mats_tr).tocsr()
            X_va = sparse.hstack(mats_va).tocsr()

            for j, et in enumerate(ETIQUETAS):
                y_tr = train_dev[et].iloc[idx_tr]

                sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
                sgd.fit(X_tr, y_tr)
                p_sgd_va = sgd.predict_proba(X_va)[:, 1]

                for c_ag in c_ag_list:
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

    # Precomputar Test completo 1.731 filas
    mats_tr_full, mats_te_full = [], []
    for analyzer, ngram, sublinear, mdf in bloques:
        v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, max_df=mdf, lowercase=True)
        mats_tr_full.append(v.fit_transform(train_full["text"]))
        mats_te_full.append(v.transform(test_df["text"]))
    X_tr_full = sparse.hstack(mats_tr_full).tocsr()
    X_te_full = sparse.hstack(mats_te_full).tocsr()

    test_por_cag = {}
    for c_ag in c_ag_list:
        prob_test_lr = np.zeros((len(test_df), 3), dtype=float)
        prob_test_sgd = np.zeros((len(test_df), 3), dtype=float)
        prob_test_saga = np.zeros((len(test_df), 3), dtype=float)

        for j, et in enumerate(ETIQUETAS):
            y_tr_col = train_full[et]
            c_val = {"HS": 2.0, "TR": 1.8, "AG": c_ag}[et]

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
        test_por_cag[c_ag] = p_b_test

    eps = 1e-6
    print("\n================ RESULTADOS FINALES V2_MAXDF90 ================")
    print("Base P114: CCV Media = 0.786446 (Can: 0.787218, P47: 0.785673) | Test Pos: 1265 (HS:515, TR:330, AG:420)")
    print("----------------------------------------------------------------")

    for c_ag in c_ag_list:
        for temp in [1.04, 1.05, 1.06, 1.07]:
            for thr_ag in [0.46, 0.47, 0.48]:
                umbrales = {"HS": 0.44, "TR": 0.51, "AG": thr_ag}
                f1s = []
                for nombre_part, pliegues in particiones:
                    d = oof_por_cag[c_ag][nombre_part]
                    p_b = 0.80 * d["lr"] + 0.05 * d["sgd"] + 0.15 * d["saga"]
                    p_cal = expit(logit(np.clip(p_b, eps, 1 - eps)) / temp)
                    pred_raw = np.column_stack([(p_cal[:, j] >= umbrales[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
                    pred = aplicar_regla_ascendente(pred_raw)
                    m = metricas_multietiqueta(y_true_dev, pred)
                    f1s.append(m["f1_media_columnas"])
                media = np.mean(f1s)

                p_b_te = test_por_cag[c_ag]
                p_cal_te = expit(logit(np.clip(p_b_te, eps, 1 - eps)) / temp)
                pred_raw_te = np.column_stack([(p_cal_te[:, j] >= umbrales[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
                pred_te = aplicar_regla_ascendente(pred_raw_te)

                tot_pos = pred_te.sum()
                hs_pos = pred_te[:, 0].sum()
                tr_pos = pred_te[:, 1].sum()
                ag_pos = pred_te[:, 2].sum()
                c01 = np.sum((pred_p114_test == 0) & (pred_te == 1))
                c10 = np.sum((pred_p114_test == 1) & (pred_te == 0))
                delta = media - 0.786446

                if delta > 0.0003:
                    print(f"C_AG={c_ag:.1f}, T={temp:.2f}, thr_AG={thr_ag:.2f} | Media: {media:.6f} (Can: {f1s[0]:.6f}, P47: {f1s[1]:.6f}) Delta: {delta:+.6f} | Test: {tot_pos} (HS:{hs_pos}, TR:{tr_pos}, AG:{ag_pos}) | +{c01}/-{c10} (Net: {c01-c10:+d})")

if __name__ == "__main__":
    main()
