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

zb = np.load(PROY / "features" / "v2_oof_base_corrected.npz", allow_pickle=True)
grupos_dev = zb["grupos"].astype(np.int64)

y_dev = train_dev[list(ETIQUETAS)]
y_true_arr = y_dev.to_numpy(dtype=int)
textos_dev = train_dev["text"]
combo_dev = combinaciones(train_dev)

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

print("Testing Option C: 10-fold CV on dev")
sgkf10 = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
prob_oof_10 = np.zeros((len(train_dev), len(ETIQUETAS)), dtype=float)
prob_oof_110_10 = np.zeros((len(train_dev), len(ETIQUETAS)), dtype=float)

for train_idx, val_idx in sgkf10.split(train_dev, combo_dev, groups=grupos_dev):
    mats_tr, mats_va = [], []
    for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
        v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
        mats_tr.append(v.fit_transform(textos_dev.iloc[train_idx]))
        mats_va.append(v.transform(textos_dev.iloc[val_idx]))
    X_tr = sparse.hstack(mats_tr).tocsr()
    X_va = sparse.hstack(mats_va).tocsr()
    
    for j, et in enumerate(ETIQUETAS):
        lr111 = LogisticRegression(C=C_MAP[et], class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        lr111.fit(X_tr, y_dev[et].iloc[train_idx])
        p_lr111 = lr111.predict_proba(X_va)[:, 1]
        
        lr110 = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        lr110.fit(X_tr, y_dev[et].iloc[train_idx])
        p_lr110 = lr110.predict_proba(X_va)[:, 1]
        
        sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
        sgd.fit(X_tr, y_dev[et].iloc[train_idx])
        p_sgd = sgd.predict_proba(X_va)[:, 1]
        
        prob_oof_10[val_idx, j] = 0.95 * p_lr111 + 0.05 * p_sgd
        prob_oof_110_10[val_idx, j] = 0.95 * p_lr110 + 0.05 * p_sgd

pred_raw10 = np.column_stack([(prob_oof_10[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
pred10, _ = aplicar_regla_ascendente(pred_raw10)
met10 = metricas_multietiqueta(y_true_arr, pred10)
print(f"10-fold CV on dev (P111 model): F1 = {met10['f1_media_columnas']:.6f} {met10['f1_por_etiqueta']}")

pred_raw110 = np.column_stack([(prob_oof_110_10[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
pred110, _ = aplicar_regla_ascendente(pred_raw110)
met110 = metricas_multietiqueta(y_true_arr, pred110)
print(f"10-fold CV on dev (P110 model): F1 = {met110['f1_media_columnas']:.6f} {met110['f1_por_etiqueta']}")
