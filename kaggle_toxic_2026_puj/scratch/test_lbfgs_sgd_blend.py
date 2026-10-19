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

# LR lbfgs C=2.0
prob_lbfgs_can = np.zeros((len(train_dev), 3), dtype=float)
prob_lbfgs_47 = np.zeros((len(train_dev), 3), dtype=float)
for idx_tr, idx_va, m_tr, m_va in folds_can:
    for j, et in enumerate(ETIQUETAS):
        clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42)
        clf.fit(m_tr, train_dev[et].iloc[idx_tr])
        prob_lbfgs_can[idx_va, j] = clf.predict_proba(m_va)[:, 1]

for idx_tr, idx_va, m_tr, m_va in folds_47:
    for j, et in enumerate(ETIQUETAS):
        clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42)
        clf.fit(m_tr, train_dev[et].iloc[idx_tr])
        prob_lbfgs_47[idx_va, j] = clf.predict_proba(m_va)[:, 1]

# SGD alpha=5e-6
prob_sgd_can = np.zeros((len(train_dev), 3), dtype=float)
prob_sgd_47 = np.zeros((len(train_dev), 3), dtype=float)
for idx_tr, idx_va, m_tr, m_va in folds_can:
    for j, et in enumerate(ETIQUETAS):
        clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
        clf.fit(m_tr, train_dev[et].iloc[idx_tr])
        prob_sgd_can[idx_va, j] = clf.predict_proba(m_va)[:, 1]

for idx_tr, idx_va, m_tr, m_va in folds_47:
    for j, et in enumerate(ETIQUETAS):
        clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
        clf.fit(m_tr, train_dev[et].iloc[idx_tr])
        prob_sgd_47[idx_va, j] = clf.predict_proba(m_va)[:, 1]

# Evaluar Blends
print("=== COMPARACION LR L-BFGS + SGD(5e-6) ===")
for w in [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]:
    b_c = (1 - w) * prob_lbfgs_can + w * prob_sgd_can
    b_47 = (1 - w) * prob_lbfgs_47 + w * prob_sgd_47
    pr_c, _ = aplicar_regla_ascendente(np.column_stack([(b_c[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
    pr_47, _ = aplicar_regla_ascendente(np.column_stack([(b_47[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
    fc = f1_media_fast(y_true, pr_c)
    f47 = f1_media_fast(y_true, pr_47)
    m = (fc + f47) / 2
    print(f"L-BFGS + {w*100:.1f}% SGD: Canonica={fc:.6f} | Part47={f47:.6f} | Media={m:.6f}")

# Simular test
mats_tr_full, mats_te = [], []
for analyzer, ngram, mdf in BLOQUES_P108:
    v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=mdf, sublinear_tf=True, lowercase=True)
    mats_tr_full.append(v.fit_transform(train["text"]))
    mats_te.append(v.transform(test["text"]))
X_tr_full = sparse.hstack(mats_tr_full).tocsr()
X_te = sparse.hstack(mats_te).tocsr()

p_lbfgs_test = np.zeros((len(test), 3), dtype=float)
p_sgd_test = np.zeros((len(test), 3), dtype=float)
for j, et in enumerate(ETIQUETAS):
    clf1 = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42)
    clf1.fit(X_tr_full, train[et])
    p_lbfgs_test[:, j] = clf1.predict_proba(X_te)[:, 1]

    clf2 = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
    clf2.fit(X_tr_full, train[et])
    p_sgd_test[:, j] = clf2.predict_proba(X_te)[:, 1]

p_opt_test = 0.95 * p_lbfgs_test + 0.05 * p_sgd_test
pred_opt, _ = aplicar_regla_ascendente(np.column_stack([(p_opt_test[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))

# P108 liblinear oficial
p_liblinear_test = np.zeros((len(test), 3), dtype=float)
for j, et in enumerate(ETIQUETAS):
    clf0 = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
    clf0.fit(X_tr_full, train[et])
    p_liblinear_test[:, j] = clf0.predict_proba(X_te)[:, 1]
pred_p108, _ = aplicar_regla_ascendente(np.column_stack([(p_liblinear_test[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))

diff_cells = int((pred_p108 != pred_opt).sum())
print("\n--- TEST COMPARISON ---")
print(f"P108 Oficial (LB 0.78855)  : Positivos HS={pred_p108[:,0].sum()}, TR={pred_p108[:,1].sum()}, AG={pred_p108[:,2].sum()} | Total={pred_p108.sum()} ({pred_p108.sum()/(len(test)*3)*100:.2f}%)")
print(f"L-BFGS + 5% SGD(5e-6) Test : Positivos HS={pred_opt[:,0].sum()}, TR={pred_opt[:,1].sum()}, AG={pred_opt[:,2].sum()} | Total={pred_opt.sum()} ({pred_opt.sum()/(len(test)*3)*100:.2f}%)")
print(f"Celdas distintas de P108   : {diff_cells} ({diff_cells/(len(test)*3)*100:.2f}%)")
