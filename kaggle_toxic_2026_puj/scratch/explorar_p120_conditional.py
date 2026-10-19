import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(".").resolve()
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

# Load dev and fold data
train_full = data_module.read_train()
zf = np.load(RAIZ / "features" / "oof_v03_final.npz", allow_pickle=True)
zb = np.load(RAIZ / "features" / "v2_oof_base_corrected.npz", allow_pickle=True)
pliegues_can = zf["pliegues"].astype(int)
grupos = zb["grupos"].astype(np.int64)

particion = pd.read_csv(RAIZ / "data" / "processed" / "holdout_v2_ids.csv")
ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
train_dev = train_full[train_full["id"].astype(int).isin(ids_dev)].reset_index(drop=True)

y_dev = train_dev[list(ETIQUETAS)].to_numpy(dtype=int)

# Particion 47
combinacion = np.array([f"{a}{b}{c}" for a, b, c in y_dev])
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
pliegues_47 = np.full(len(train_dev), -1, dtype=int)
for num, (_, val) in enumerate(sgkf.split(np.arange(len(train_dev)).reshape(-1, 1), combinacion, groups=grupos)):
    pliegues_47[val] = num + 1

print("Loaded train_dev:", train_dev.shape)

BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]
C_PER_LABEL = {"HS": 2.0, "TR": 1.8, "AG": 1.5}
UMBRALES_CONGELADOS = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

def run_p114_oof(pliegues):
    n_samples = len(train_dev)
    prob_p114 = np.zeros((n_samples, 3), dtype=float)
    
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
        
        prob_lr_va = np.zeros((len(idx_va), 3))
        prob_sgd_va = np.zeros((len(idx_va), 3))
        prob_saga_va = np.zeros((len(idx_va), 3))
        
        for j, et in enumerate(ETIQUETAS):
            c_val = C_PER_LABEL[et]
            y_col = train_dev[et].iloc[idx_tr]
            
            lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr.fit(X_tr, y_col)
            prob_lr_va[:, j] = lr.predict_proba(X_va)[:, 1]
            
            sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
            sgd.fit(X_tr, y_col)
            prob_sgd_va[:, j] = sgd.predict_proba(X_va)[:, 1]
            
            saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
            saga.fit(X_tr, y_col)
            prob_saga_va[:, j] = saga.predict_proba(X_va)[:, 1]
            
        prob_blend = 0.80 * prob_lr_va + 0.05 * prob_sgd_va + 0.15 * prob_saga_va
        prob_p114[idx_va] = expit(logit(np.clip(prob_blend, 1e-6, 1 - 1e-6)) / 1.04)
        
    return prob_p114

prob_p114_can = run_p114_oof(pliegues_can)
prob_p114_p47 = run_p114_oof(pliegues_47)

def eval_preds(probs, ths=UMBRALES_CONGELADOS):
    pred = np.column_stack([(probs[:, j] >= ths[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    # Apply ascending rule
    mask = (pred[:, 1] == 1) & (pred[:, 0] == 0)
    pred[mask, 0] = 1
    return pred

pred_p114_can = eval_preds(prob_p114_can)
pred_p114_p47 = eval_preds(prob_p114_p47)

f1_can_p114 = metricas_multietiqueta(y_dev, pred_p114_can)["f1_media_columnas"]
f1_p47_p114 = metricas_multietiqueta(y_dev, pred_p114_p47)["f1_media_columnas"]
print(f"P114 Canonica: {f1_can_p114:.6f} | Part47: {f1_p47_p114:.6f} | Media: {(f1_can_p114+f1_p47_p114)/2:.6f}")

# Save baseline oofs
np.savez_compressed("scratch/p114_oofs.npz", can=prob_p114_can, p47=prob_p114_p47, y=y_dev)
print("Saved baseline P114 OOFs.")
