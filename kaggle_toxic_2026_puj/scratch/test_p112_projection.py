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
from src.evaluation import ETIQUETAS

train_full = data_module.read_train()
test_df = data_module.read_test()

print(f"train_full: {len(train_full)}, test_df: {len(test_df)}")

# grupos de train_full
grupos = grupos_fusionados(train_full)
print(f"Unique groups: {len(np.unique(grupos))}")

# combo
combo = combinaciones(train_full)

BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]

mats_tr_full, mats_te = [], []
for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
    v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
    mats_tr_full.append(v.fit_transform(train_full["text"]))
    mats_te.append(v.transform(test_df["text"]))
X_tr_full = sparse.hstack(mats_tr_full).tocsr()
X_te = sparse.hstack(mats_te).tocsr()
print(f"Features: {X_tr_full.shape[1]}")

sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)

prob_test_folds = np.zeros((len(test_df), len(ETIQUETAS)), dtype=float)

C_MAP = {"HS": 2.0, "TR": 1.8, "AG": 1.5}
UMBRALES_CONGELADOS = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

fold_count = 0
for train_idx, val_idx in sgkf.split(train_full, combo, groups=grupos):
    fold_count += 1
    # En cada uno de los 10 pliegues (entrenado en ~1.558 filas):
    # - 95% LogisticRegression(C_per_label={HS: 2.0, TR: 1.8, AG: 1.5}, class_weight='balanced', solver='liblinear', random_state=42)
    # - 5% SGDClassifier(loss='log_loss', penalty='l2', alpha=5e-6, class_weight='balanced', max_iter=2000, random_state=42)
    X_train_f = X_tr_full[train_idx]
    
    prob_lr_f = np.zeros((len(test_df), len(ETIQUETAS)), dtype=float)
    prob_sgd_f = np.zeros((len(test_df), len(ETIQUETAS)), dtype=float)
    
    for j, et in enumerate(ETIQUETAS):
        clf_lr = LogisticRegression(C=C_MAP[et], class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        clf_lr.fit(X_train_f, train_full[et].iloc[train_idx])
        prob_lr_f[:, j] = clf_lr.predict_proba(X_te)[:, 1]
        
        clf_sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
        clf_sgd.fit(X_train_f, train_full[et].iloc[train_idx])
        prob_sgd_f[:, j] = clf_sgd.predict_proba(X_te)[:, 1]
        
    prob_model_k = 0.95 * prob_lr_f + 0.05 * prob_sgd_f
    prob_test_folds += prob_model_k

P_test = prob_test_folds / 10.0

pred_test_raw = np.column_stack([(P_test[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
# Regla ontologica HatEval: Si pred_TR == 1 => pred_HS = 1
pred_test = pred_test_raw.copy()
mask = (pred_test[:, 1] == 1) & (pred_test[:, 0] == 0)
pred_test[mask, 0] = 1

pos = {et: int(pred_test[:, j].sum()) for j, et in enumerate(ETIQUETAS)}
total_pos = int(pred_test.sum())
pct_pos = total_pos / (len(test_df) * 3)

print(f"Positivos P112 Test: HS={pos['HS']}, TR={pos['TR']}, AG={pos['AG']} | Total={total_pos} ({pct_pos*100:.2f}%)")

sub_p110 = pd.read_csv(PROY / "submissions" / "toxic_submission_p110_lr_sgd_blend_asc.csv")
pred_p110 = sub_p110[list(ETIQUETAS)].to_numpy(dtype=int)
diff_p110 = int((pred_p110 != pred_test).sum())
pct_diff_p110 = diff_p110 / (len(test_df) * 3) * 100
print(f"Celdas distintas vs P110: {diff_p110} ({pct_diff_p110:.2f}%)")

sub_p111 = pd.read_csv(PROY / "submissions" / "toxic_submission_p111_per_label_c_asc.csv")
pred_p111 = sub_p111[list(ETIQUETAS)].to_numpy(dtype=int)
diff_p111 = int((pred_p111 != pred_test).sum())
pct_diff_p111 = diff_p111 / (len(test_df) * 3) * 100
print(f"Celdas distintas vs P111: {diff_p111} ({pct_diff_p111:.2f}%)")
