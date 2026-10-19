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

def compute_blend_prob(folds_data):
    prob = np.zeros((len(train_dev), 3), dtype=float)
    for idx_tr, idx_va, m_tr, m_va in folds_data:
        for j, et in enumerate(ETIQUETAS):
            lr = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr.fit(m_tr, train_dev[et].iloc[idx_tr])
            p_lr = lr.predict_proba(m_va)[:, 1]

            sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
            sgd.fit(m_tr, train_dev[et].iloc[idx_tr])
            p_sgd = sgd.predict_proba(m_va)[:, 1]

            prob[idx_va, j] = 0.95 * p_lr + 0.05 * p_sgd
    return prob

prob_can = compute_blend_prob(folds_can)
prob_47 = compute_blend_prob(folds_47)

# Grid fino de umbrales
print("=== GRID DE UMBRALES SOBRE BLEND (0.95 LR + 0.05 SGD) ===")
base_th = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
pr_c_base, _ = aplicar_regla_ascendente(np.column_stack([(prob_can[:, j] >= base_th[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
pr_47_base, _ = aplicar_regla_ascendente(np.column_stack([(prob_47[:, j] >= base_th[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
f_c_base = f1_media_fast(y_true, pr_c_base)
f_47_base = f1_media_fast(y_true, pr_47_base)
print(f"Base (0.44, 0.51, 0.46): Can={f_c_base:.6f} | Part47={f_47_base:.6f} | Media={(f_c_base+f_47_base)/2:.6f}")

mejores_th = []
for d_hs in [-0.02, -0.01, 0.0, 0.01, 0.02]:
    for d_tr in [-0.02, -0.01, 0.0, 0.01, 0.02]:
        for d_ag in [-0.02, -0.01, 0.0, 0.01, 0.02]:
            th = {
                "HS": round(base_th["HS"] + d_hs, 2),
                "TR": round(base_th["TR"] + d_tr, 2),
                "AG": round(base_th["AG"] + d_ag, 2),
            }
            pr_c, _ = aplicar_regla_ascendente(np.column_stack([(prob_can[:, j] >= th[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
            pr_47, _ = aplicar_regla_ascendente(np.column_stack([(prob_47[:, j] >= th[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
            fc = f1_media_fast(y_true, pr_c)
            f47 = f1_media_fast(y_true, pr_47)
            m = (fc + f47) / 2
            if fc >= f_c_base and f47 >= f_47_base and m > (f_c_base + f_47_base) / 2:
                mejores_th.append((m, fc, f47, th))

mejores_th.sort(key=lambda x: x[0], reverse=True)
for m, fc, f47, th in mejores_th[:8]:
    print(f"Umbrales={th}: Can={fc:.6f} ({fc-f_c_base:+.5f}) | Part47={f47:.6f} ({f47-f_47_base:+.5f}) | Media={m:.6f}")
