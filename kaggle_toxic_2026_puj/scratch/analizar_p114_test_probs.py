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
test_df = data_module.read_test()

BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]
C_PER_LABEL = {"HS": 2.0, "TR": 1.8, "AG": 1.5}
TEMPERATURE = 1.04

mats_tr, mats_te = [], []
for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
    v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
    mats_tr.append(v.fit_transform(train_full["text"]))
    mats_te.append(v.transform(test_df["text"]))
X_tr = sparse.hstack(mats_tr).tocsr()
X_te = sparse.hstack(mats_te).tocsr()

prob_lr = np.zeros((len(test_df), 3))
prob_sgd = np.zeros((len(test_df), 3))
prob_saga = np.zeros((len(test_df), 3))

for j, et in enumerate(ETIQUETAS):
    c_val = C_PER_LABEL[et]
    y_col = train_full[et]
    
    lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
    lr.fit(X_tr, y_col)
    prob_lr[:, j] = lr.predict_proba(X_te)[:, 1]
    
    sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
    sgd.fit(X_tr, y_col)
    prob_sgd[:, j] = sgd.predict_proba(X_te)[:, 1]
    
    saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
    saga.fit(X_tr, y_col)
    prob_saga[:, j] = saga.predict_proba(X_te)[:, 1]

prob_blend = 0.80 * prob_lr + 0.05 * prob_sgd + 0.15 * prob_saga
prob_p114_test = expit(logit(np.clip(prob_blend, 1e-6, 1 - 1e-6)) / TEMPERATURE)

# Verify against official submission
sub_p114 = pd.read_csv("submissions/toxic_submission_p114_tri_optimizer_smooth_blend_asc.csv")
pred_raw = np.column_stack([(prob_p114_test[:, j] >= [0.44, 0.51, 0.46][j]).astype(int) for j in range(3)])
mask = (pred_raw[:, 1] == 1) & (pred_raw[:, 0] == 0)
pred_raw[mask, 0] = 1

diff = (pred_raw != sub_p114[list(ETIQUETAS)].to_numpy()).sum()
print("Diff vs official P114 submission:", diff, "(must be 0)")
assert diff == 0

np.savez_compressed("scratch/p114_test_probs.npz", prob=prob_p114_test)
print("Saved exact test probabilities.")
