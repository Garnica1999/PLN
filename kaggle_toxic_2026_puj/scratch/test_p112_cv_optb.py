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
zb = np.load(PROY / "features" / "v2_oof_base_corrected.npz", allow_pickle=True)
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

print("Testing Option B: TFIDF fit on idx_tr, inner 10-fold models on X_tr")
prob_p112_optB = np.zeros((len(train_dev), len(ETIQUETAS)), dtype=float)

for fold in sorted(set(pliegues_can)):
    idx_tr = np.flatnonzero(pliegues_can != fold)
    idx_va = np.flatnonzero(pliegues_can == fold)
    
    mats_tr, mats_va = [], []
    for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
        v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
        mats_tr.append(v.fit_transform(textos_dev.iloc[idx_tr]))
        mats_va.append(v.transform(textos_dev.iloc[idx_va]))
    X_tr_triple = sparse.hstack(mats_tr).tocsr()
    X_va_triple = sparse.hstack(mats_va).tocsr()
    
    df_tr = train_dev.iloc[idx_tr].reset_index(drop=True)
    grupos_inner = grupos_dev[idx_tr]
    combo_inner = combinaciones(df_tr)
    
    sgkf_inner = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    prob_inner_folds = np.zeros((len(idx_va), len(ETIQUETAS)), dtype=float)
    
    for inner_tr_idx, inner_va_idx in sgkf_inner.split(df_tr, combo_inner, groups=grupos_inner):
        X_in_tr = X_tr_triple[inner_tr_idx]
        prob_lr_in = np.zeros((len(idx_va), len(ETIQUETAS)), dtype=float)
        prob_sgd_in = np.zeros((len(idx_va), len(ETIQUETAS)), dtype=float)
        for j, et in enumerate(ETIQUETAS):
            lr = LogisticRegression(C=C_MAP[et], class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr.fit(X_in_tr, df_tr[et].iloc[inner_tr_idx])
            prob_lr_in[:, j] = lr.predict_proba(X_va_triple)[:, 1]
            
            sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
            sgd.fit(X_in_tr, df_tr[et].iloc[inner_tr_idx])
            prob_sgd_in[:, j] = sgd.predict_proba(X_va_triple)[:, 1]
        prob_inner_folds += (0.95 * prob_lr_in + 0.05 * prob_sgd_in)
    
    prob_p112_optB[idx_va] = prob_inner_folds / 10.0

pred_raw = np.column_stack([(prob_p112_optB[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
pred, n_asc = aplicar_regla_ascendente(pred_raw)
met = metricas_multietiqueta(y_true_arr, pred)
print(f"P112 Opt B F1: {met['f1_media_columnas']:.6f} {met['f1_por_etiqueta']}")
