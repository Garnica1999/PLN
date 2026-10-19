#!/usr/bin/env python
"""Exploracion rigurosa para P116: Supresion de Falsos Positivos en AG y Calibracion de Prevalencia."""

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
C_PER_LABEL = {"HS": 2.0, "TR": 1.8, "AG": 1.5}
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
    print("=== EXPLORACION P116: SUPRESION DE FALSOS POSITIVOS Y CONTROL DE PREVALENCIA ===")
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
    probs_dev = {}

    for nombre_part, pliegues in particiones:
        n_samples = len(train_dev)
        prob_lr = np.zeros((n_samples, 3), dtype=float)
        prob_sgd = np.zeros((n_samples, 3), dtype=float)
        prob_saga = np.zeros((n_samples, 3), dtype=float)

        for fold in sorted(set(pliegues)):
            idx_tr = np.flatnonzero(pliegues != fold)
            idx_va = np.flatnonzero(pliegues == fold)

            mats_tr, mats_va = [], []
            for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
                v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
                mats_tr.append(v.fit_transform(train_dev["text"].iloc[idx_tr]))
                mats_va.append(v.transform(train_dev["text"].iloc[idx_va]))
            X_tr = sparse.hstack(mats_tr).tocsr()
            X_va = sparse.hstack(mats_va).tocsr()

            for j, et in enumerate(ETIQUETAS):
                y_col = train_dev[et].iloc[idx_tr]
                c_val = C_PER_LABEL[et]

                lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
                lr.fit(X_tr, y_col)
                prob_lr[idx_va, j] = lr.predict_proba(X_va)[:, 1]

                sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
                sgd.fit(X_tr, y_col)
                prob_sgd[idx_va, j] = sgd.predict_proba(X_va)[:, 1]

                saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
                saga.fit(X_tr, y_col)
                prob_saga[idx_va, j] = saga.predict_proba(X_va)[:, 1]

        probs_dev[nombre_part] = (prob_lr, prob_sgd, prob_saga)

    # 1. Ajuste completo sobre train para test
    print("Ajustando modelos completos sobre train (1.731 filas) para test...")
    mats_tr_full, mats_te = [], []
    for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
        v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
        mats_tr_full.append(v.fit_transform(train_full["text"]))
        mats_te.append(v.transform(test_df["text"]))
    X_tr_full = sparse.hstack(mats_tr_full).tocsr()
    X_te = sparse.hstack(mats_te).tocsr()

    prob_lr_test = np.zeros((len(test_df), 3), dtype=float)
    prob_sgd_test = np.zeros((len(test_df), 3), dtype=float)
    prob_saga_test = np.zeros((len(test_df), 3), dtype=float)

    for j, et in enumerate(ETIQUETAS):
        y_col = train_full[et]
        c_val = C_PER_LABEL[et]

        clf_lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        clf_lr.fit(X_tr_full, y_col)
        prob_lr_test[:, j] = clf_lr.predict_proba(X_te)[:, 1]

        clf_sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
        clf_sgd.fit(X_tr_full, y_col)
        prob_sgd_test[:, j] = clf_sgd.predict_proba(X_te)[:, 1]

        clf_saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
        clf_saga.fit(X_tr_full, y_col)
        prob_saga_test[:, j] = clf_saga.predict_proba(X_te)[:, 1]

    # Cargar P114
    sub_p114 = pd.read_csv(SUB_P114)
    pred_p114_test = sub_p114[list(ETIQUETAS)].to_numpy(dtype=int)

    # 2. Experimentos con control de prevalencia
    eps = 1e-6
    print("\n--- EXPERIMENTOS SOBRE DEV (CCV) Y PROYECCION EN TEST ---")
    # P114 Base
    prob_blend_te = 0.80 * prob_lr_test + 0.05 * prob_sgd_test + 0.15 * prob_saga_test
    prob_cal_te_104 = expit(logit(np.clip(prob_blend_te, eps, 1 - eps)) / 1.04)

    prob_lr_c, prob_sgd_c, prob_saga_c = probs_dev["Canonica"]
    prob_lr_47, prob_sgd_47, prob_saga_47 = probs_dev["Part47"]
    prob_blend_c = 0.80 * prob_lr_c + 0.05 * prob_sgd_c + 0.15 * prob_saga_c
    prob_blend_47 = 0.80 * prob_lr_47 + 0.05 * prob_sgd_47 + 0.15 * prob_saga_47

    # P114 Base performance
    p114_c = aplicar_regla_ascendente(np.column_stack([(expit(logit(np.clip(prob_blend_c, eps, 1 - eps)) / 1.04)[:, j] >= UMBRALES_BASE[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
    p114_47 = aplicar_regla_ascendente(np.column_stack([(expit(logit(np.clip(prob_blend_47, eps, 1 - eps)) / 1.04)[:, j] >= UMBRALES_BASE[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
    f1_p114_c = metricas_multietiqueta(y_true_dev, p114_c)["f1_media_columnas"]
    f1_p114_47 = metricas_multietiqueta(y_true_dev, p114_47)["f1_media_columnas"]
    f1_p114_media = (f1_p114_c + f1_p114_47) / 2
    print(f"P114 Base Media CCV: {f1_p114_media:.6f} (Canonica: {f1_p114_c:.6f}, Part47: {f1_p114_47:.6f})")

    # Exploracion A: Variar Umbral de AG directamente (0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52)
    print("\nExploracion A: Umbral de AG directo")
    for ag_th in [0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52]:
        th_dict = {"HS": 0.44, "TR": 0.51, "AG": ag_th}
        pred_c = aplicar_regla_ascendente(np.column_stack([(expit(logit(np.clip(prob_blend_c, eps, 1 - eps)) / 1.04)[:, j] >= th_dict[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
        pred_47 = aplicar_regla_ascendente(np.column_stack([(expit(logit(np.clip(prob_blend_47, eps, 1 - eps)) / 1.04)[:, j] >= th_dict[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
        m_c = metricas_multietiqueta(y_true_dev, pred_c)
        m_47 = metricas_multietiqueta(y_true_dev, pred_47)
        media_ccv = (m_c["f1_media_columnas"] + m_47["f1_media_columnas"]) / 2

        # Test
        pred_te = aplicar_regla_ascendente(np.column_stack([(prob_cal_te_104[:, j] >= th_dict[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
        diff_p114 = int((pred_p114_test != pred_te).sum())
        pos_hs = int(pred_te[:, 0].sum())
        pos_tr = int(pred_te[:, 1].sum())
        pos_ag = int(pred_te[:, 2].sum())
        total_pos = int(pred_te.sum())
        pct_pos = total_pos / 3462 * 100

        print(f"AG th={ag_th:.2f} -> Media CCV: {media_ccv:.6f} (Delta: {media_ccv - f1_p114_media:+.6f}) | Test Pos: HS={pos_hs}, TR={pos_tr}, AG={pos_ag} | Total={total_pos} ({pct_pos:.2f}%) | Diff P114: {diff_p114}")

    # Exploracion B: Supresion Ontologica de AG cuando HS=0
    # Regla: Si pred_HS == 0 y P_cal(AG) < ag_strict_hs0 => pred_AG = 0
    print("\nExploracion B: Supresion Ontologica de AG en HS=0")
    for ag_strict in [0.47, 0.48, 0.49, 0.50, 0.52, 0.54, 0.56, 0.60]:
        # En Canonica
        pred_c = p114_c.copy()
        cal_c = expit(logit(np.clip(prob_blend_c, eps, 1 - eps)) / 1.04)
        mask_c = (pred_c[:, 0] == 0) & (pred_c[:, 2] == 1) & (cal_c[:, 2] < ag_strict)
        pred_c[mask_c, 2] = 0
        pred_c = aplicar_regla_ascendente(pred_c)
        m_c = metricas_multietiqueta(y_true_dev, pred_c)["f1_media_columnas"]

        # En Part47
        pred_47 = p114_47.copy()
        cal_47 = expit(logit(np.clip(prob_blend_47, eps, 1 - eps)) / 1.04)
        mask_47 = (pred_47[:, 0] == 0) & (pred_47[:, 2] == 1) & (cal_47[:, 2] < ag_strict)
        pred_47[mask_47, 2] = 0
        pred_47 = aplicar_regla_ascendente(pred_47)
        m_47 = metricas_multietiqueta(y_true_dev, pred_47)["f1_media_columnas"]

        media_ccv = (m_c + m_47) / 2

        # En Test
        pred_te = pred_p114_test.copy()
        mask_te = (pred_te[:, 0] == 0) & (pred_te[:, 2] == 1) & (prob_cal_te_104[:, 2] < ag_strict)
        pred_te[mask_te, 2] = 0
        pred_te = aplicar_regla_ascendente(pred_te)
        diff_p114 = int((pred_p114_test != pred_te).sum())
        total_pos = int(pred_te.sum())
        pos_ag = int(pred_te[:, 2].sum())

        print(f"AG supresion en HS=0 (ag_strict={ag_strict:.2f}) -> Media CCV: {media_ccv:.6f} (Delta: {media_ccv - f1_p114_media:+.6f}) | Test AG: {pos_ag} | Total Pos: {total_pos} | Diff P114: {diff_p114} (supresiones: {mask_te.sum()})")

    # Exploracion C: Temperatura T superior (1.04 -> 1.06, 1.08, 1.10)
    print("\nExploracion C: Temperatura mas fria T en log-odds (reduce prevalencia hacia k=0.935)")
    for T in [1.02, 1.04, 1.06, 1.08, 1.10]:
        cal_c = expit(logit(np.clip(prob_blend_c, eps, 1 - eps)) / T)
        cal_47 = expit(logit(np.clip(prob_blend_47, eps, 1 - eps)) / T)
        pred_c = aplicar_regla_ascendente(np.column_stack([(cal_c[:, j] >= UMBRALES_BASE[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
        pred_47 = aplicar_regla_ascendente(np.column_stack([(cal_47[:, j] >= UMBRALES_BASE[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
        m_c = metricas_multietiqueta(y_true_dev, pred_c)["f1_media_columnas"]
        m_47 = metricas_multietiqueta(y_true_dev, pred_47)["f1_media_columnas"]
        media_ccv = (m_c + m_47) / 2

        # Test
        cal_te = expit(logit(np.clip(prob_blend_te, eps, 1 - eps)) / T)
        pred_te = aplicar_regla_ascendente(np.column_stack([(cal_te[:, j] >= UMBRALES_BASE[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
        diff_p114 = int((pred_p114_test != pred_te).sum())
        total_pos = int(pred_te.sum())
        print(f"T={T:.2f} -> Media CCV: {media_ccv:.6f} (Delta: {media_ccv - f1_p114_media:+.6f}) | Test Pos: {total_pos} ({total_pos/3462*100:.2f}%) | Diff P114: {diff_p114}")

if __name__ == "__main__":
    main()
