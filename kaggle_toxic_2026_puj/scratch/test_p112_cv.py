import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedGroupKFold

PROY = Path("/home/carlos/Git/PLN/kaggle_toxic_2026_puj")
sys.path.insert(0, str(PROY))
from src import data as data_module
from src.splits import grupos_fusionados, combinaciones
from src.evaluation import ETIQUETAS, metricas_multietiqueta

train_full = data_module.read_train()
particion = pd.read_csv(PROY / "data" / "processed" / "holdout_v2_ids.csv")
ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
train_dev = train_full[train_full["id"].astype(int).isin(ids_dev)].reset_index(drop=True)

zf = np.load(PROY / "features" / "oof_v03_final.npz", allow_pickle=True)
zb = np.load(PROY / "features" / "oof_v2_base_corrected.npz" if (PROY / "features" / "oof_v2_base_corrected.npz").exists() else PROY / "features" / "v2_oof_base_corrected.npz", allow_pickle=True)
pliegues_can = zf["pliegues"].astype(int)
grupos_dev = zb["grupos"].astype(np.int64)

y_dev = train_dev[list(ETIQUETAS)]
y_true_arr = y_dev.to_numpy(dtype=int)
textos_dev = train_dev["text"]

BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]
C_MAP = {"HS": 2.0, "TR": 1.8, "AG": 1.5}
UMBRALES_CONGELADOS = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

def aplicar_regla_ascendente(pred: np.ndarray) -> tuple[np.ndarray, int]:
    p = pred.copy()
    mask = (p[:, 1] == 1) & (p[:, 0] == 0)
    p[mask, 0] = 1
    return p, int(mask.sum())

print("Testing Option A: Inner 10-fold bagging on train_fold (fold-local TFIDF)")
prob_p112_optA = np.zeros((len(train_dev), len(ETIQUETAS)), dtype=float)
prob_p111 = np.zeros((len(train_dev), len(ETIQUETAS)), dtype=float)
prob_p110 = np.zeros((len(train_dev), len(ETIQUETAS)), dtype=float)

for fold in sorted(set(pliegues_can)):
    idx_tr = np.flatnonzero(pliegues_can != fold)
    idx_va = np.flatnonzero(pliegues_can == fold)
    
    # Baseline P111 / P110: single model on idx_tr
    mats_tr, mats_va = [], []
    for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
        v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
        mats_tr.append(v.fit_transform(textos_dev.iloc[idx_tr]))
        mats_va.append(v.transform(textos_dev.iloc[idx_va]))
    X_tr_triple = sparse.hstack(mats_tr).tocsr()
    X_va_triple = sparse.hstack(mats_va).tocsr()
    
    prob_lr_p110 = np.zeros((len(idx_va), len(ETIQUETAS)), dtype=float)
    prob_sgd_p110 = np.zeros((len(idx_va), len(ETIQUETAS)), dtype=float)
    prob_lr_p111 = np.zeros((len(idx_va), len(ETIQUETAS)), dtype=float)
    
    for j, et in enumerate(ETIQUETAS):
        lr_110 = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        lr_110.fit(X_tr_triple, y_dev[et].iloc[idx_tr])
        prob_lr_p110[:, j] = lr_110.predict_proba(X_va_triple)[:, 1]
        
        sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
        sgd.fit(X_tr_triple, y_dev[et].iloc[idx_tr])
        prob_sgd_p110[:, j] = sgd.predict_proba(X_va_triple)[:, 1]
        
        lr_111 = LogisticRegression(C=C_MAP[et], class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        lr_111.fit(X_tr_triple, y_dev[et].iloc[idx_tr])
        prob_lr_p111[:, j] = lr_111.predict_proba(X_va_triple)[:, 1]
        
    prob_p110[idx_va] = 0.95 * prob_lr_p110 + 0.05 * prob_sgd_p110
    prob_p111[idx_va] = 0.95 * prob_lr_p111 + 0.05 * prob_sgd_p110
    
    # Option A: Inner 10-fold StratifiedGroupKFold on idx_tr
    df_tr = train_dev.iloc[idx_tr].reset_index(drop=True)
    grupos_inner = grupos_dev[idx_tr]
    combo_inner = combinaciones(df_tr)
    
    sgkf_inner = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    prob_inner_folds = np.zeros((len(idx_va), len(ETIQUETAS)), dtype=float)
    
    for inner_tr_idx, inner_va_idx in sgkf_inner.split(df_tr, combo_inner, groups=grupos_inner):
        mats_in_tr, mats_in_va = [], []
        for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
            v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
            mats_in_tr.append(v.fit_transform(df_tr["text"].iloc[inner_tr_idx]))
            mats_in_va.append(v.transform(textos_dev.iloc[idx_va]))
        X_in_tr = sparse.hstack(mats_in_tr).tocsr()
        X_in_va = sparse.hstack(mats_in_va).tocsr()
        
        prob_lr_in = np.zeros((len(idx_va), len(ETIQUETAS)), dtype=float)
        prob_sgd_in = np.zeros((len(idx_va), len(ETIQUETAS)), dtype=float)
        for j, et in enumerate(ETIQUETAS):
            lr = LogisticRegression(C=C_MAP[et], class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr.fit(X_in_tr, df_tr[et].iloc[inner_tr_idx])
            prob_lr_in[:, j] = lr.predict_proba(X_in_va)[:, 1]
            
            sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
            sgd.fit(X_in_tr, df_tr[et].iloc[inner_tr_idx])
            prob_sgd_in[:, j] = sgd.predict_proba(X_in_va)[:, 1]
        prob_inner_folds += (0.95 * prob_lr_in + 0.05 * prob_sgd_in)
    
    prob_p112_optA[idx_va] = prob_inner_folds / 10.0

# Metrics
def compute_metrics(prob):
    pred_raw = np.column_stack([(prob[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    pred, n_asc = aplicar_regla_ascendente(pred_raw)
    met = metricas_multietiqueta(y_true_arr, pred)
    return met["f1_media_columnas"], met["f1_por_etiqueta"], n_asc, pred

f1_110, f1_et_110, _, pred_110 = compute_metrics(prob_p110)
f1_111, f1_et_111, _, pred_111 = compute_metrics(prob_p111)
f1_112_A, f1_et_112_A, _, pred_112_A = compute_metrics(prob_p112_optA)

print(f"P110 F1: {f1_110:.6f} {f1_et_110}")
print(f"P111 F1: {f1_111:.6f} {f1_et_111}")
print(f"P112 Opt A F1: {f1_112_A:.6f} {f1_et_112_A}")
print(f"Delta P112 vs P111: {f1_112_A - f1_111:+.6f}")
print(f"Delta P112 vs P110: {f1_112_A - f1_110:+.6f}")
