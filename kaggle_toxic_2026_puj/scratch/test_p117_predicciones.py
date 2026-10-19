#!/usr/bin/env python
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier

RAIZ = Path("/home/carlos/Git/PLN/kaggle_toxic_2026_puj")
sys.path.insert(0, str(RAIZ))
from src import data as data_module
from src.evaluation import ETIQUETAS

train_full = data_module.read_train()
test_df = data_module.read_test()

BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]

mats_tr, mats_te = [], []
for a, n, sub in BLOQUES_V2_TRIPLE:
    v = TfidfVectorizer(analyzer=a, ngram_range=n, min_df=2, sublinear_tf=sub, lowercase=True)
    mats_tr.append(v.fit_transform(train_full["text"]))
    mats_te.append(v.transform(test_df["text"]))

X_tr = sparse.hstack(mats_tr).tocsr()
X_te = sparse.hstack(mats_te).tocsr()

C_PER_LABEL = {"HS": 2.0, "TR": 1.8, "AG": 1.5}

prob_lr = np.zeros((len(test_df), 3))
prob_sgd = np.zeros((len(test_df), 3))
prob_saga = np.zeros((len(test_df), 3))
prob_nbsvm = np.zeros((len(test_df), 3))

def nbsvm_r(X_tr, y_tr, X_te, alpha=1.0):
    y_arr = y_tr.to_numpy() if hasattr(y_tr, "to_numpy") else np.asarray(y_tr)
    p = np.array(X_tr[y_arr == 1].sum(axis=0)) + alpha
    q = np.array(X_tr[y_arr == 0].sum(axis=0)) + alpha
    p = p / np.sum(p)
    q = q / np.sum(q)
    r = np.log(p / q)
    r_diag = sparse.diags(r.flatten())
    return X_tr.dot(r_diag), X_te.dot(r_diag)

for j, et in enumerate(ETIQUETAS):
    y_col = train_full[et]
    c_val = C_PER_LABEL[et]
    
    lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
    lr.fit(X_tr, y_col)
    prob_lr[:, j] = lr.predict_proba(X_te)[:, 1]
    
    sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
    sgd.fit(X_tr, y_col)
    prob_sgd[:, j] = sgd.predict_proba(X_te)[:, 1]
    
    saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
    saga.fit(X_tr, y_col)
    prob_saga[:, j] = saga.predict_proba(X_te)[:, 1]
    
    Xtr_nb, Xte_nb = nbsvm_r(X_tr, y_col, X_te, alpha=1.0)
    nb = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
    nb.fit(Xtr_nb, y_col)
    prob_nbsvm[:, j] = nb.predict_proba(Xte_nb)[:, 1]

eps = 1e-6
# P114
b_114 = 0.80 * prob_lr + 0.05 * prob_sgd + 0.15 * prob_saga
p_114 = expit(logit(np.clip(b_114, eps, 1 - eps)) / 1.04)
pred_114 = np.column_stack([(p_114[:, j] >= {"HS": 0.44, "TR": 0.51, "AG": 0.46}[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
# Ascendente
m = (pred_114[:, 1] == 1) & (pred_114[:, 0] == 0)
pred_114[m, 0] = 1

# P117 Base (70 LR + 15 NBSVM + 10 SAGA + 5 SGD, T=1.04)
b_117 = 0.70 * prob_lr + 0.15 * prob_nbsvm + 0.10 * prob_saga + 0.05 * prob_sgd
p_117 = expit(logit(np.clip(b_117, eps, 1 - eps)) / 1.04)
pred_117_base = np.column_stack([(p_117[:, j] >= {"HS": 0.44, "TR": 0.51, "AG": 0.46}[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
m = (pred_117_base[:, 1] == 1) & (pred_117_base[:, 0] == 0)
pred_117_base[m, 0] = 1

# P117 con Rescue TR (HS>=0.70, TR in [0.485, 0.510))
pred_117_rescue = np.column_stack([(p_117[:, j] >= {"HS": 0.44, "TR": 0.51, "AG": 0.46}[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
mask_rescue = (p_117[:, 0] >= 0.70) & (p_117[:, 1] >= 0.485) & (p_117[:, 1] < 0.510)
print(f"Filas rescatadas en test por regla de odio extremo: {mask_rescue.sum()}")
for idx in np.flatnonzero(mask_rescue):
    t_id = test_df['id'].iloc[idx]
    txt = test_df['text'].iloc[idx][:80]
    print(f"  Fila {idx} (id {t_id}): P(HS)={p_117[idx, 0]:.4f}, P(TR)={p_117[idx, 1]:.4f}, P(AG)={p_117[idx, 2]:.4f}")
    print(f"    Texto: {txt}")

pred_117_rescue[mask_rescue, 1] = 1
m = (pred_117_rescue[:, 1] == 1) & (pred_117_rescue[:, 0] == 0)
pred_117_rescue[m, 0] = 1

print("\n--- POSITIVOS EN TEST ---")
print(f"P114        : HS={pred_114[:,0].sum()}, TR={pred_114[:,1].sum()}, AG={pred_114[:,2].sum()} | Total={pred_114.sum()}")
print(f"P117 Base   : HS={pred_117_base[:,0].sum()}, TR={pred_117_base[:,1].sum()}, AG={pred_117_base[:,2].sum()} | Total={pred_117_base.sum()}")
print(f"P117 Rescue : HS={pred_117_rescue[:,0].sum()}, TR={pred_117_rescue[:,1].sum()}, AG={pred_117_rescue[:,2].sum()} | Total={pred_117_rescue.sum()}")

diff_cells_base = (pred_114 != pred_117_base).sum()
diff_cells_rescue = (pred_114 != pred_117_rescue).sum()
print(f"Celdas distintas P117 Base vs P114: {diff_cells_base} ({diff_cells_base/3462*100:.2f}%)")
print(f"Celdas distintas P117 Rescue vs P114: {diff_cells_rescue} ({diff_cells_rescue/3462*100:.2f}%)")

# Que celdas cambian entre P114 y P117 Base?
cambios_0_a_1 = ((pred_114 == 0) & (pred_117_base == 1)).sum()
cambios_1_a_0 = ((pred_114 == 1) & (pred_117_base == 0)).sum()
print(f"P117 Base cambios: 0->1 = {cambios_0_a_1}, 1->0 = {cambios_1_a_0}, neto = {cambios_0_a_1 - cambios_1_a_0}")
