import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
import sys

RAIZ = Path(".")
sys.path.insert(0, str(RAIZ))
from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

OOF_FINAL = RAIZ / "features" / "oof_v03_final.npz"
OOF_BASE = RAIZ / "features" / "v2_oof_base_corrected.npz"
IDS_PATH = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"

train_full = data_module.read_train()
zf = np.load(OOF_FINAL, allow_pickle=True)
zb = np.load(OOF_BASE, allow_pickle=True)
pliegues_can = zf["pliegues"].astype(int)

particion = pd.read_csv(IDS_PATH)
ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
train_dev = train_full[train_full["id"].astype(int).isin(ids_dev)].reset_index(drop=True)

y = train_dev[list(ETIQUETAS)].to_numpy(dtype=int)

BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]
C_PER_LABEL = {"HS": 2.0, "TR": 1.8, "AG": 1.5}

textos = train_dev["text"]
probs_p114 = np.zeros((len(train_dev), 3))

for fold in sorted(set(pliegues_can)):
    idx_tr = np.flatnonzero(pliegues_can != fold)
    idx_va = np.flatnonzero(pliegues_can == fold)
    mats_tr, mats_va = [], []
    for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
        v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
        mats_tr.append(v.fit_transform(textos.iloc[idx_tr]))
        mats_va.append(v.transform(textos.iloc[idx_va]))
    X_tr = sparse.hstack(mats_tr).tocsr()
    X_va = sparse.hstack(mats_va).tocsr()
    
    for j, et in enumerate(ETIQUETAS):
        c_val = C_PER_LABEL[et]
        y_col = train_dev[et].iloc[idx_tr]
        lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        lr.fit(X_tr, y_col)
        p_lr = lr.predict_proba(X_va)[:, 1]
        
        sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
        sgd.fit(X_tr, y_col)
        p_sgd = sgd.predict_proba(X_va)[:, 1]
        
        saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
        saga.fit(X_tr, y_col)
        p_saga = saga.predict_proba(X_va)[:, 1]
        
        blend = 0.80 * p_lr + 0.05 * p_sgd + 0.15 * p_saga
        z = logit(np.clip(blend, 1e-7, 1 - 1e-7))
        probs_p114[idx_va, j] = expit(z / 1.04)

# Current score with 0.44, 0.51, 0.46
bin_curr = (probs_p114 >= [0.44, 0.51, 0.46]).astype(int)
bin_curr[bin_curr[:, 1] == 1, 0] = 1
m_curr = metricas_multietiqueta(y, bin_curr)
f1_base = m_curr["f1_media_columnas"]
print(f"Current Canonica CCV: {f1_base:.6f} | HS={m_curr['f1_por_etiqueta']['HS']:.4f}, TR={m_curr['f1_por_etiqueta']['TR']:.4f}, AG={m_curr['f1_por_etiqueta']['AG']:.4f}")

# Grid search optimal thresholds
best_f1 = 0
best_thrs = None

for t_hs in np.arange(0.40, 0.52, 0.01):
    for t_tr in np.arange(0.46, 0.56, 0.01):
        for t_ag in np.arange(0.42, 0.54, 0.01):
            b = (probs_p114 >= [t_hs, t_tr, t_ag]).astype(int)
            b[b[:, 1] == 1, 0] = 1
            f1_m = (f1_score(y[:, 0], b[:, 0], pos_label=1, zero_division=0) +
                    f1_score(y[:, 1], b[:, 1], pos_label=1, zero_division=0) +
                    f1_score(y[:, 2], b[:, 2], pos_label=1, zero_division=0)) / 3.0
            if f1_m > best_f1:
                best_f1 = f1_m
                best_thrs = (round(t_hs, 2), round(t_tr, 2), round(t_ag, 2))

print(f"Optimal thresholds on Canonica: {best_thrs} -> CCV={best_f1:.6f} (delta={best_f1 - f1_base:+.6f})")

