import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

RAIZ = Path(".")
import sys
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

BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]
C_PER_LABEL = {"HS": 2.0, "TR": 1.8, "AG": 1.5}
UMBRALES_CONGELADOS = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

print(f"Dataset dev loaded: {len(train_dev)} rows.")

def run_experiment(name, eval_func):
    for p_name, p_splits in [("Canonica", pliegues_can), ("Particion 47", pliegues_47)]:
        scores = eval_func(p_splits)
        print(f"[{name}] {p_name}: CCV={scores['ccv']:.6f} | HS={scores['f1_hs']:.4f}, TR={scores['f1_tr']:.4f}, AG={scores['f1_ag']:.4f}")

# Precompute V2_triple fold matrices to be fast
print("Precomputing matrices for Canonica and Part47...")
def get_fold_mats(pliegues):
    fold_data = []
    textos = train_dev["text"]
    for fold in sorted(set(pliegues)):
        idx_tr = np.flatnonzero(pliegues != fold)
        idx_va = np.flatnonzero(pliegues == fold)
        mats_tr, mats_va = [], []
        for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
            v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
            mats_tr.append(v.fit_transform(textos.iloc[idx_tr]))
            mats_va.append(v.transform(textos.iloc[idx_va]))
        X_tr = sparse.hstack(mats_tr).tocsr()
        X_va = sparse.hstack(mats_va).tocsr()
        fold_data.append((fold, idx_tr, idx_va, X_tr, X_va))
    return fold_data

folds_can = get_fold_mats(pliegues_can)
folds_p47 = get_fold_mats(pliegues_47)

def eval_with_folds(fold_data, get_probs_fn):
    n_samples = len(train_dev)
    probs = np.zeros((n_samples, len(ETIQUETAS)), dtype=float)
    y = train_dev[list(ETIQUETAS)]
    
    for fold, idx_tr, idx_va, X_tr, X_va in fold_data:
        p_fold = get_probs_fn(X_tr, X_va, y.iloc[idx_tr])
        probs[idx_va] = p_fold
        
    thrs = np.array([UMBRALES_CONGELADOS[e] for e in ETIQUETAS])
    bin_pred = (probs >= thrs).astype(int)
    # Regla ontologica TR=1 => HS=1
    bin_pred[bin_pred[:, 1] == 1, 0] = 1
    
    y_true = y.to_numpy(dtype=int)
    res = metricas_multietiqueta(y_true, bin_pred)
    return {
        "ccv": res["f1_media_columnas"],
        "f1_hs": res["f1_por_etiqueta"]["HS"],
        "f1_tr": res["f1_por_etiqueta"]["TR"],
        "f1_ag": res["f1_por_etiqueta"]["AG"],
        "probs": probs
    }

# 1. P114 Baseline
def p114_probs(X_tr, X_va, y_tr):
    probs = np.zeros((X_va.shape[0], 3))
    for j, et in enumerate(ETIQUETAS):
        c_val = C_PER_LABEL[et]
        y_col = y_tr[et]
        lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        lr.fit(X_tr, y_col)
        p_lr = lr.predict_proba(X_va)[:, 1]
        
        sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
        sgd.fit(X_tr, y_col)
        p_sgd = sgd.predict_proba(X_va)[:, 1]
        
        saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
        saga.fit(X_tr, y_col)
        p_saga = saga.predict_proba(X_va)[:, 1]
        
        p_blend = 0.80 * p_lr + 0.05 * p_sgd + 0.15 * p_saga
        z = logit(np.clip(p_blend, 1e-7, 1 - 1e-7))
        probs[:, j] = expit(z / 1.04)
    return probs

res_can_p114 = eval_with_folds(folds_can, p114_probs)
res_p47_p114 = eval_with_folds(folds_p47, p114_probs)
print(f"[P114] Canonica: {res_can_p114['ccv']:.6f} | Part47: {res_p47_p114['ccv']:.6f} | Media: {(res_can_p114['ccv']+res_p47_p114['ccv'])/2:.6f}")

# 2. LinearSVC Blend with P114
for w_svc in [0.05, 0.10, 0.15]:
    def svc_blend_probs(X_tr, X_va, y_tr, w=w_svc):
        probs = np.zeros((X_va.shape[0], 3))
        for j, et in enumerate(ETIQUETAS):
            c_val = C_PER_LABEL[et]
            y_col = y_tr[et]
            lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr.fit(X_tr, y_col)
            p_lr = lr.predict_proba(X_va)[:, 1]
            
            sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
            sgd.fit(X_tr, y_col)
            p_sgd = sgd.predict_proba(X_va)[:, 1]
            
            saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
            saga.fit(X_tr, y_col)
            p_saga = saga.predict_proba(X_va)[:, 1]
            
            svc = LinearSVC(C=c_val, class_weight="balanced", max_iter=2000, random_state=42)
            svc.fit(X_tr, y_col)
            df = svc.decision_function(X_va)
            p_svc = expit(df)
            
            p_blend = (1.0 - w) * (0.80 * p_lr + 0.05 * p_sgd + 0.15 * p_saga) + w * p_svc
            z = logit(np.clip(p_blend, 1e-7, 1 - 1e-7))
            probs[:, j] = expit(z / 1.04)
        return probs

    res_can_svc = eval_with_folds(folds_can, svc_blend_probs)
    res_p47_svc = eval_with_folds(folds_p47, svc_blend_probs)
    media = (res_can_svc['ccv'] + res_p47_svc['ccv']) / 2
    print(f"[P114 + {int(w_svc*100)}% LinearSVC] Canonica: {res_can_svc['ccv']:.6f} | Part47: {res_p47_svc['ccv']:.6f} | Media: {media:.6f} (d={media - (res_can_p114['ccv']+res_p47_p114['ccv'])/2:+.6f})")

# 3. Classifier Chain with P114
def chain_probs(X_tr, X_va, y_tr):
    # HS -> TR -> AG
    probs = np.zeros((X_va.shape[0], 3))
    
    # 1. HS is standard P114
    c_hs = C_PER_LABEL["HS"]
    lr_hs = LogisticRegression(C=c_hs, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
    lr_hs.fit(X_tr, y_tr["HS"])
    p_hs = lr_hs.predict_proba(X_va)[:, 1]
    probs[:, 0] = p_hs
    
    # Augmented X for TR: add HS prediction feature
    p_tr_train_hs = lr_hs.predict_proba(X_tr)[:, 1].reshape(-1, 1)
    p_tr_val_hs = p_hs.reshape(-1, 1)
    X_tr_aug = sparse.hstack([X_tr, sparse.csr_matrix(p_tr_train_hs)]).tocsr()
    X_va_aug = sparse.hstack([X_va, sparse.csr_matrix(p_tr_val_hs)]).tocsr()
    
    c_tr = C_PER_LABEL["TR"]
    lr_tr = LogisticRegression(C=c_tr, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
    lr_tr.fit(X_tr_aug, y_tr["TR"])
    p_tr = lr_tr.predict_proba(X_va_aug)[:, 1]
    probs[:, 1] = p_tr
    
    # Augmented X for AG: add HS and TR prediction features
    p_tr_train_tr = lr_tr.predict_proba(X_tr_aug)[:, 1].reshape(-1, 1)
    p_tr_val_tr = p_tr.reshape(-1, 1)
    X_tr_aug2 = sparse.hstack([X_tr, sparse.csr_matrix(p_tr_train_hs), sparse.csr_matrix(p_tr_train_tr)]).tocsr()
    X_va_aug2 = sparse.hstack([X_va, sparse.csr_matrix(p_tr_val_hs), sparse.csr_matrix(p_tr_val_tr)]).tocsr()
    
    c_ag = C_PER_LABEL["AG"]
    lr_ag = LogisticRegression(C=c_ag, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
    lr_ag.fit(X_tr_aug2, y_tr["AG"])
    p_ag = lr_ag.predict_proba(X_va_aug2)[:, 1]
    probs[:, 2] = p_ag
    
    return probs

res_can_chain = eval_with_folds(folds_can, chain_probs)
res_p47_chain = eval_with_folds(folds_p47, chain_probs)
media_chain = (res_can_chain['ccv'] + res_p47_chain['ccv']) / 2
print(f"[Classifier Chain] Canonica: {res_can_chain['ccv']:.6f} | Part47: {res_p47_chain['ccv']:.6f} | Media: {media_chain:.6f} (d={media_chain - (res_can_p114['ccv']+res_p47_p114['ccv'])/2:+.6f})")

