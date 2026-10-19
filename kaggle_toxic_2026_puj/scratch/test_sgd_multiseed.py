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

# LR Base P108
prob_lr_can = np.zeros((len(train_dev), 3), dtype=float)
prob_lr_47 = np.zeros((len(train_dev), 3), dtype=float)
for idx_tr, idx_va, m_tr, m_va in folds_can:
    for j, et in enumerate(ETIQUETAS):
        lr = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        lr.fit(m_tr, train_dev[et].iloc[idx_tr])
        prob_lr_can[idx_va, j] = lr.predict_proba(m_va)[:, 1]

for idx_tr, idx_va, m_tr, m_va in folds_47:
    for j, et in enumerate(ETIQUETAS):
        lr = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        lr.fit(m_tr, train_dev[et].iloc[idx_tr])
        prob_lr_47[idx_va, j] = lr.predict_proba(m_va)[:, 1]

p_base_c, _ = aplicar_regla_ascendente(np.column_stack([(prob_lr_can[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
p_base_47, _ = aplicar_regla_ascendente(np.column_stack([(prob_lr_47[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
f_c_base = f1_media_fast(y_true, p_base_c)
f_47_base = f1_media_fast(y_true, p_base_47)
print(f"P108 Base: Canonica={f_c_base:.6f} | Part47={f_47_base:.6f} | Media={(f_c_base+f_47_base)/2:.6f}")

# Multi-seed SGD (5 seeds: 42, 43, 44, 45, 46)
seeds = [42, 43, 44, 45, 46]
prob_sgd_can_seeds = np.zeros((len(train_dev), 3), dtype=float)
prob_sgd_47_seeds = np.zeros((len(train_dev), 3), dtype=float)

for s in seeds:
    for idx_tr, idx_va, m_tr, m_va in folds_can:
        for j, et in enumerate(ETIQUETAS):
            sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=s, class_weight="balanced")
            sgd.fit(m_tr, train_dev[et].iloc[idx_tr])
            prob_sgd_can_seeds[idx_va, j] += sgd.predict_proba(m_va)[:, 1] / len(seeds)

    for idx_tr, idx_va, m_tr, m_va in folds_47:
        for j, et in enumerate(ETIQUETAS):
            sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=s, class_weight="balanced")
            sgd.fit(m_tr, train_dev[et].iloc[idx_tr])
            prob_sgd_47_seeds[idx_va, j] += sgd.predict_proba(m_va)[:, 1] / len(seeds)

print("\n--- EVALUANDO BLEND CON MULTI-SEED SGD (alpha=5e-6) ---")
for w in [0.03, 0.04, 0.05, 0.06, 0.07]:
    b_c = (1 - w) * prob_lr_can + w * prob_sgd_can_seeds
    b_47 = (1 - w) * prob_lr_47 + w * prob_sgd_47_seeds
    pr_c, _ = aplicar_regla_ascendente(np.column_stack([(b_c[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
    pr_47, _ = aplicar_regla_ascendente(np.column_stack([(b_47[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
    fc = f1_media_fast(y_true, pr_c)
    f47 = f1_media_fast(y_true, pr_47)
    m = (fc + f47) / 2
    print(f"Blend LR + {w*100:.1f}% Multi-Seed SGD: Can={fc:.6f} ({fc-f_c_base:+.5f}) | Part47={f47:.6f} ({f47-f_47_base:+.5f}) | Media={m:.6f} ({m-(f_c_base+f_47_base)/2:+.5f})")

# Simulacion en test con Multi-seed SGD
mats_tr_full, mats_te = [], []
for analyzer, ngram, mdf in BLOQUES_P108:
    v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=mdf, sublinear_tf=True, lowercase=True)
    mats_tr_full.append(v.fit_transform(train["text"]))
    mats_te.append(v.transform(test["text"]))
X_tr_full = sparse.hstack(mats_tr_full).tocsr()
X_te = sparse.hstack(mats_te).tocsr()

p_lr_test = np.zeros((len(test), 3), dtype=float)
for j, et in enumerate(ETIQUETAS):
    lr = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
    lr.fit(X_tr_full, train[et])
    p_lr_test[:, j] = lr.predict_proba(X_te)[:, 1]

p_sgd_test = np.zeros((len(test), 3), dtype=float)
for s in seeds:
    for j, et in enumerate(ETIQUETAS):
        sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=s, class_weight="balanced")
        sgd.fit(X_tr_full, train[et])
        p_sgd_test[:, j] += sgd.predict_proba(X_te)[:, 1] / len(seeds)

for w in [0.04, 0.05, 0.06]:
    p_blend = (1 - w) * p_lr_test + w * p_sgd_test
    pred_b, _ = aplicar_regla_ascendente(np.column_stack([(p_blend[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
    pred_p108, _ = aplicar_regla_ascendente(np.column_stack([(p_lr_test[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
    diff = int((pred_p108 != pred_b).sum())
    print(f"\nTest Multi-Seed w={w*100:.1f}%:")
    print(f"  Positivos: HS={pred_b[:,0].sum()}, TR={pred_b[:,1].sum()}, AG={pred_b[:,2].sum()} | Total={pred_b.sum()} ({pred_b.sum()/(len(test)*3)*100:.2f}%)")
    print(f"  Diff cells vs P108: {diff} ({diff/(len(test)*3)*100:.2f}%)")
