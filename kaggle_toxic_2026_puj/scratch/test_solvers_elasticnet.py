import sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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

def eval_lr_config(name, get_clf):
    p_c = np.zeros((len(train_dev), 3), dtype=float)
    p_47 = np.zeros((len(train_dev), 3), dtype=float)
    for idx_tr, idx_va, m_tr, m_va in folds_can:
        for j, et in enumerate(ETIQUETAS):
            clf = get_clf(et)
            clf.fit(m_tr, train_dev[et].iloc[idx_tr])
            p_c[idx_va, j] = clf.predict_proba(m_va)[:, 1]
    for idx_tr, idx_va, m_tr, m_va in folds_47:
        for j, et in enumerate(ETIQUETAS):
            clf = get_clf(et)
            clf.fit(m_tr, train_dev[et].iloc[idx_tr])
            p_47[idx_va, j] = clf.predict_proba(m_va)[:, 1]

    pr_c, _ = aplicar_regla_ascendente(np.column_stack([(p_c[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
    pr_47, _ = aplicar_regla_ascendente(np.column_stack([(p_47[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
    fc = f1_media_fast(y_true, pr_c)
    f47 = f1_media_fast(y_true, pr_47)
    m = (fc + f47) / 2
    print(f"{name:35s}: Can={fc:.6f} | Part47={f47:.6f} | Media={m:.6f}")
    return fc, f47, m

# Base P108
fc_b, f47_b, m_b = eval_lr_config("Base P108 (liblinear C=2.0)",
    lambda e: LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42))

# L-BFGS
eval_lr_config("LR lbfgs C=2.0",
    lambda e: LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42))

# SAGA l2
eval_lr_config("LR saga l2 C=2.0",
    lambda e: LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="saga", random_state=42))

# SAGA ElasticNet (l1_ratio=0.05, 0.1)
for l1_r in [0.02, 0.05, 0.10]:
    eval_lr_config(f"LR saga ElasticNet (l1_r={l1_r})",
        lambda e: LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="saga", penalty="elasticnet", l1_ratio=l1_r, random_state=42))
