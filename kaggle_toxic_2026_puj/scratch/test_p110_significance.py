import sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path

RAIZ = Path("/home/carlos/Git/PLN/kaggle_toxic_2026_puj")
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS

OOF_FINAL = RAIZ / "features" / "oof_v03_final.npz"
OOF_BASE = RAIZ / "features" / "v2_oof_base_corrected.npz"
IDS_PATH = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"

zf = np.load(OOF_FINAL, allow_pickle=True)
zb = np.load(OOF_BASE, allow_pickle=True)
pliegues_can = zf["pliegues"].astype(int)
y_true = zf["y_true"].astype(int)
grupos = zb["grupos"].astype(np.int64)

train = data_module.read_train()
test = data_module.read_test()
particion = pd.read_csv(IDS_PATH)
ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
train_dev = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)

# Particion 47
combinacion = np.array([f"{a}{b}{c}" for a, b, c in y_true])
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
pliegues_47 = np.full(len(y_true), -1, dtype=int)
for num, (_, val) in enumerate(sgkf.split(np.arange(len(y_true)).reshape(-1, 1), combinacion, groups=grupos)):
    pliegues_47[val] = num + 1

BLOQUES_P108 = [
    ("word", (1, 2), 2),
    ("char_wb", (3, 5), 2),
    ("char", (1, 6), 2),
]
th_p108 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

def aplicar_regla_ascendente(pred):
    p = pred.copy()
    mask = (p[:, 1] == 1) & (p[:, 0] == 0)
    p[mask, 0] = 1
    return p, int(mask.sum())

def f1_media_fast(y_t, y_p):
    tp = np.count_nonzero(y_t & y_p, axis=0)
    total = y_t.sum(axis=0) + y_p.sum(axis=0)
    f1 = np.where(total == 0, 0.0, 2.0 * tp / total)
    return float(np.mean(f1))

def construir_matrices_folds(pliegues_arr, bloques):
    folds_data = []
    for fold in sorted(set(pliegues_arr)):
        idx_tr = np.flatnonzero(pliegues_arr != fold)
        idx_va = np.flatnonzero(pliegues_arr == fold)
        mats_tr, mats_va = [], []
        for analyzer, ngram, mdf in bloques:
            v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=mdf, sublinear_tf=True, lowercase=True)
            mats_tr.append(v.fit_transform(train_dev["text"].iloc[idx_tr]))
            mats_va.append(v.transform(train_dev["text"].iloc[idx_va]))
        folds_data.append((idx_tr, idx_va, sparse.hstack(mats_tr).tocsr(), sparse.hstack(mats_va).tocsr()))
    return folds_data

folds_can = construir_matrices_folds(pliegues_can, BLOQUES_P108)
folds_47 = construir_matrices_folds(pliegues_47, BLOQUES_P108)

def compute_oof(folds_data, alpha_sgd=1e-5, w_sgd=0.05):
    prob = np.zeros((len(train_dev), len(ETIQUETAS)), dtype=float)
    for idx_tr, idx_va, m_tr, m_va in folds_data:
        for j, et in enumerate(ETIQUETAS):
            lr = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr.fit(m_tr, train_dev[et].iloc[idx_tr])
            p_lr = lr.predict_proba(m_va)[:, 1]
            if w_sgd > 0:
                sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=alpha_sgd, max_iter=2000, random_state=42, class_weight="balanced")
                sgd.fit(m_tr, train_dev[et].iloc[idx_tr])
                p_sgd = sgd.predict_proba(m_va)[:, 1]
                prob[idx_va, j] = (1.0 - w_sgd) * p_lr + w_sgd * p_sgd
            else:
                prob[idx_va, j] = p_lr
    pred, n_asc = aplicar_regla_ascendente(np.column_stack([(prob[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
    return prob, pred, n_asc

print("Calculando OOFs para P108, Blend 1 y Blend 2...")
# P108 Base
prob_p108_can, pred_p108_can, _ = compute_oof(folds_can, w_sgd=0.0)
prob_p108_47, pred_p108_47, _ = compute_oof(folds_47, w_sgd=0.0)

# Candidate: Blend LR + 5% SGD(5e-6)
prob_b1_can, pred_b1_can, n_asc_b1_can = compute_oof(folds_can, alpha_sgd=5e-6, w_sgd=0.05)
prob_b1_47, pred_b1_47, n_asc_b1_47 = compute_oof(folds_47, alpha_sgd=5e-6, w_sgd=0.05)

# Candidate: Blend LR + 5% SGD(1e-5)
prob_b2_can, pred_b2_can, n_asc_b2_can = compute_oof(folds_can, alpha_sgd=1e-5, w_sgd=0.05)
prob_b2_47, pred_b2_47, n_asc_b2_47 = compute_oof(folds_47, alpha_sgd=1e-5, w_sgd=0.05)

def per_fold_eval(pliegues_arr, y_t, y_p1, y_p2):
    f1_1, f1_2 = [], []
    for f in sorted(set(pliegues_arr)):
        idx = np.flatnonzero(pliegues_arr == f)
        f1_1.append(f1_media_fast(y_t[idx], y_p1[idx]))
        f1_2.append(f1_media_fast(y_t[idx], y_p2[idx]))
    f1_1 = np.array(f1_1)
    f1_2 = np.array(f1_2)
    deltas = f1_2 - f1_1
    return f1_1, f1_2, deltas

def bootstrap_ci_fast(y_t, y_p1, y_p2, grupos_arr, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(grupos_arr)
    n_groups = len(unique_groups)
    group_to_idx = {g: np.flatnonzero(grupos_arr == g) for g in unique_groups}
    diffs = []
    for _ in range(n_boot):
        sample_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sample_groups])
        f1_a = f1_media_fast(y_t[idx], y_p1[idx])
        f1_b = f1_media_fast(y_t[idx], y_p2[idx])
        diffs.append(f1_b - f1_a)
    diffs = np.array(diffs)
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    win_rate = np.mean(diffs > 0)
    return ci_low, ci_high, win_rate

f1_p108_can_total = f1_media_fast(y_true, pred_p108_can)
f1_p108_47_total = f1_media_fast(y_true, pred_p108_47)

print("\n=== EVALUACION BLEND 1: 95% LR + 5% SGD(5e-6) ===")
f1_base_can, f1_b1_can, deltas_b1_can = per_fold_eval(pliegues_can, y_true, pred_p108_can, pred_b1_can)
f1_base_47, f1_b1_47, deltas_b1_47 = per_fold_eval(pliegues_47, y_true, pred_p108_47, pred_b1_47)
ci_l_b1_can, ci_h_b1_can, win_b1_can = bootstrap_ci_fast(y_true, pred_p108_can, pred_b1_can, grupos)
ci_l_b1_47, ci_h_b1_47, win_b1_47 = bootstrap_ci_fast(y_true, pred_p108_47, pred_b1_47, grupos)
f1_b1_can_total = f1_media_fast(y_true, pred_b1_can)
f1_b1_47_total = f1_media_fast(y_true, pred_b1_47)

print(f"Canonica CCV: P108={f1_p108_can_total:.6f} -> Blend1={f1_b1_can_total:.6f} (Delta={f1_b1_can_total-f1_p108_can_total:+.6f})")
print(f"  Pliegues ganados: {np.sum(deltas_b1_can > 0)}/5 (deltas={np.round(deltas_b1_can, 6)})")
print(f"  IC95 Bootstrap: [{ci_l_b1_can:+.6f}, {ci_h_b1_can:+.6f}] | Win Rate: {win_b1_can*100:.1f}%")
print(f"Part47 CCV: P108={f1_p108_47_total:.6f} -> Blend1={f1_b1_47_total:.6f} (Delta={f1_b1_47_total-f1_p108_47_total:+.6f})")
print(f"  Pliegues ganados: {np.sum(deltas_b1_47 > 0)}/5 (deltas={np.round(deltas_b1_47, 6)})")
print(f"  IC95 Bootstrap: [{ci_l_b1_47:+.6f}, {ci_h_b1_47:+.6f}] | Win Rate: {win_b1_47*100:.1f}%")

print("\n=== EVALUACION BLEND 2: 95% LR + 5% SGD(1e-5) ===")
f1_base_can, f1_b2_can, deltas_b2_can = per_fold_eval(pliegues_can, y_true, pred_p108_can, pred_b2_can)
f1_base_47, f1_b2_47, deltas_b2_47 = per_fold_eval(pliegues_47, y_true, pred_p108_47, pred_b2_47)
ci_l_b2_can, ci_h_b2_can, win_b2_can = bootstrap_ci_fast(y_true, pred_p108_can, pred_b2_can, grupos)
ci_l_b2_47, ci_h_b2_47, win_b2_47 = bootstrap_ci_fast(y_true, pred_p108_47, pred_b2_47, grupos)
f1_b2_can_total = f1_media_fast(y_true, pred_b2_can)
f1_b2_47_total = f1_media_fast(y_true, pred_b2_47)

print(f"Canonica CCV: P108={f1_p108_can_total:.6f} -> Blend2={f1_b2_can_total:.6f} (Delta={f1_b2_can_total-f1_p108_can_total:+.6f})")
print(f"  Pliegues ganados: {np.sum(deltas_b2_can > 0)}/5 (deltas={np.round(deltas_b2_can, 6)})")
print(f"  IC95 Bootstrap: [{ci_l_b2_can:+.6f}, {ci_h_b2_can:+.6f}] | Win Rate: {win_b2_can*100:.1f}%")
print(f"Part47 CCV: P108={f1_p108_47_total:.6f} -> Blend2={f1_b2_47_total:.6f} (Delta={f1_b2_47_total-f1_p108_47_total:+.6f})")
print(f"  Pliegues ganados: {np.sum(deltas_b2_47 > 0)}/5 (deltas={np.round(deltas_b2_47, 6)})")
print(f"  IC95 Bootstrap: [{ci_l_b2_47:+.6f}, {ci_h_b2_47:+.6f}] | Win Rate: {win_b2_47*100:.1f}%")
