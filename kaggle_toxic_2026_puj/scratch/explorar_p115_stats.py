#!/usr/bin/env python
"""Estadisticas completas de validacion para P115."""

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

UMBRALES_CONGELADOS = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
C_PER_LABEL = {"HS": 2.0, "TR": 1.8, "AG": 1.5}
TEMPERATURE = 1.04
SEMILLA_BOOT = 20260925

BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]

def aplicar_gating_ontologico(
    pred_raw: np.ndarray,
    probs_cal: np.ndarray,
    hs_rescue: float = 0.39,
    hs_for_tr: float = 0.60,
    tr_rescue: float = 0.48,
) -> tuple[np.ndarray, dict]:
    p = pred_raw.copy()
    mask_rescue_hs = (p[:, 0] == 0) & (p[:, 2] == 1) & (probs_cal[:, 0] >= hs_rescue)
    n_rescue_hs = int(mask_rescue_hs.sum())
    p[mask_rescue_hs, 0] = 1
    mask_rescue_tr = (p[:, 0] == 1) & (probs_cal[:, 0] >= hs_for_tr) & (p[:, 1] == 0) & (probs_cal[:, 1] >= tr_rescue)
    n_rescue_tr = int(mask_rescue_tr.sum())
    p[mask_rescue_tr, 1] = 1
    # Regla ascendente
    mask_asc = (p[:, 1] == 1) & (p[:, 0] == 0)
    p[mask_asc, 0] = 1
    return p, {"n_rescue_hs": n_rescue_hs, "n_rescue_tr": n_rescue_tr, "n_asc": int(mask_asc.sum())}

def bootstrap_ci(y_true, pred_base, pred_challenger, grupos, n_boot=2000, seed=SEMILLA_BOOT):
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(grupos)
    n_groups = len(unique_groups)
    group_to_idx = {g: np.flatnonzero(grupos == g) for g in unique_groups}

    diffs = []
    for _ in range(n_boot):
        sample_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sample_groups])
        f1_b = metricas_multietiqueta(y_true[idx], pred_base[idx])["f1_media_columnas"]
        f1_c = metricas_multietiqueta(y_true[idx], pred_challenger[idx])["f1_media_columnas"]
        diffs.append(f1_c - f1_b)

    diffs = np.array(diffs)
    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    pct_mejora = float(np.mean(diffs > 0))
    return ci_low, ci_high, pct_mejora

def main():
    train_full = data_module.read_train()
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

    for nombre_part, pliegues in particiones:
        print(f"\n==========================================")
        print(f"Estadisticas Detalladas para: {nombre_part}")
        print(f"==========================================")
        n_samples = len(train_dev)
        prob_lr = np.zeros((n_samples, 3), dtype=float)
        prob_sgd = np.zeros((n_samples, 3), dtype=float)
        prob_saga = np.zeros((n_samples, 3), dtype=float)
        prob_v03 = np.zeros((n_samples, 3), dtype=float)

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

            # Control V03
            mats_tr_v03, mats_va_v03 = [], []
            for analyzer, ngram, sublinear in [("word", (1, 2), True), ("char_wb", (3, 5), True)]:
                v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
                mats_tr_v03.append(v.fit_transform(train_dev["text"].iloc[idx_tr]))
                mats_va_v03.append(v.transform(train_dev["text"].iloc[idx_va]))
            X_tr_v03 = sparse.hstack(mats_tr_v03).tocsr()
            X_va_v03 = sparse.hstack(mats_va_v03).tocsr()
            for j, et in enumerate(ETIQUETAS):
                clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
                clf.fit(X_tr_v03, train_dev[et].iloc[idx_tr])
                prob_v03[idx_va, j] = clf.predict_proba(X_va_v03)[:, 1]

        eps = 1e-6
        # P114 Base
        prob_blend = 0.80 * prob_lr + 0.05 * prob_sgd + 0.15 * prob_saga
        prob_cal = expit(logit(np.clip(prob_blend, eps, 1 - eps)) / TEMPERATURE)
        pred_p114_raw = np.column_stack([(prob_cal[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
        mask_asc_114 = (pred_p114_raw[:, 1] == 1) & (pred_p114_raw[:, 0] == 0)
        pred_p114 = pred_p114_raw.copy()
        pred_p114[mask_asc_114, 0] = 1
        m_p114 = metricas_multietiqueta(y_true_dev, pred_p114)

        # Challenger P115 (P114 + Dual Ontological Gating)
        pred_p115, stats_p115 = aplicar_gating_ontologico(pred_p114_raw, prob_cal, hs_rescue=0.39, hs_for_tr=0.60, tr_rescue=0.48)
        m_p115 = metricas_multietiqueta(y_true_dev, pred_p115)

        # Control V03
        pred_v03_raw = np.column_stack([(prob_v03[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
        m_v03 = metricas_multietiqueta(y_true_dev, pred_v03_raw)

        delta_vs_p114 = m_p115["f1_media_columnas"] - m_p114["f1_media_columnas"]
        delta_vs_v03 = m_p115["f1_media_columnas"] - m_v03["f1_media_columnas"]

        # Folds
        folds_p114 = []
        folds_p115 = []
        for f in sorted(set(pliegues)):
            idx = np.flatnonzero(pliegues == f)
            folds_p114.append(metricas_multietiqueta(y_true_dev[idx], pred_p114[idx])["f1_media_columnas"])
            folds_p115.append(metricas_multietiqueta(y_true_dev[idx], pred_p115[idx])["f1_media_columnas"])
        deltas_folds = np.array(folds_p115) - np.array(folds_p114)
        pliegues_ganados = int(np.sum(deltas_folds > 0))
        pliegues_empatados = int(np.sum(deltas_folds == 0))

        ci_low, ci_high, win_rate = bootstrap_ci(y_true_dev, pred_p114, pred_p115, grupos)
        diff_cells = int((pred_p114 != pred_p115).sum())

        print(f"P115 F1: {m_p115['f1_media_columnas']:.6f} (Delta vs P114: {delta_vs_p114:+.6f}, vs V03: {delta_vs_v03:+.6f})")
        print(f"Por etiqueta P115: HS={m_p115['f1_por_etiqueta']['HS']:.4f}, TR={m_p115['f1_por_etiqueta']['TR']:.4f}, AG={m_p115['f1_por_etiqueta']['AG']:.4f}")
        print(f"Por etiqueta P114: HS={m_p114['f1_por_etiqueta']['HS']:.4f}, TR={m_p114['f1_por_etiqueta']['TR']:.4f}, AG={m_p114['f1_por_etiqueta']['AG']:.4f}")
        print(f"Pliegues: ganados {pliegues_ganados}/5, empatados {pliegues_empatados}/5 | Deltas: {[round(d, 6) for d in deltas_folds]}")
        print(f"Bootstrap Win Rate: {win_rate*100:.1f}% | IC95: [{ci_low:+.6f}, {ci_high:+.6f}]")
        print(f"Celdas distintas en dev: {diff_cells}")
        print(f"Rescates: HS={stats_p115['n_rescue_hs']}, TR={stats_p115['n_rescue_tr']}, Violaciones ASC={stats_p115['n_asc']}")

if __name__ == "__main__":
    main()
