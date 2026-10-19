import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier

RAIZ = Path(".").resolve()
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

train_full = data_module.read_train()
particion = pd.read_csv(RAIZ / "data" / "processed" / "holdout_v2_ids.csv")
ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
train_dev = train_full[train_full["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
y_dev = train_dev[list(ETIQUETAS)].to_numpy(dtype=int)

zf = np.load(RAIZ / "features" / "oof_v03_final.npz", allow_pickle=True)
zb = np.load(RAIZ / "features" / "v2_oof_base_corrected.npz", allow_pickle=True)
pliegues_can = zf["pliegues"].astype(int)
grupos = zb["grupos"].astype(np.int64)

from sklearn.model_selection import StratifiedGroupKFold
combinacion = np.array([f"{a}{b}{c}" for a, b, c in y_dev])
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
pliegues_47 = np.full(len(train_dev), -1, dtype=int)
for num, (_, val) in enumerate(sgkf.split(np.arange(len(train_dev)).reshape(-1, 1), combinacion, groups=grupos)):
    pliegues_47[val] = num + 1

BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]
C_PER_LABEL = {"HS": 2.0, "TR": 1.8, "AG": 1.5}
UMBRALES = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

def get_oof_individual_models(pliegues):
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
            c_val = C_PER_LABEL[et]
            y_col = train_dev[et].iloc[idx_tr]
            
            lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr.fit(X_tr, y_col)
            prob_lr[idx_va, j] = lr.predict_proba(X_va)[:, 1]
            
            sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
            sgd.fit(X_tr, y_col)
            prob_sgd[idx_va, j] = sgd.predict_proba(X_va)[:, 1]
            
            saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
            saga.fit(X_tr, y_col)
            prob_saga[idx_va, j] = saga.predict_proba(X_va)[:, 1]
            
    return prob_lr, prob_sgd, prob_saga

print("Computing individual OOF models on Canonica...")
lr_can, sgd_can, saga_can = get_oof_individual_models(pliegues_can)
print("Computing individual OOF models on Part47...")
lr_p47, sgd_p47, saga_p47 = get_oof_individual_models(pliegues_47)

# Save individual models
np.savez_compressed("scratch/individual_models_oof.npz",
    lr_can=lr_can, sgd_can=sgd_can, saga_can=saga_can,
    lr_p47=lr_p47, sgd_p47=sgd_p47, saga_p47=saga_p47,
    y=y_dev
)
print("Saved individual models OOF.")
