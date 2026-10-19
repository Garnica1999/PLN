#!/usr/bin/env python
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta
SUB_P121 = RAIZ / "submissions" / "toxic_submission_p121_asymmetric_joint_blend_asc.csv"
SUB_P122 = RAIZ / "submissions" / "toxic_submission_p122_covariance_ag_rescue_asc.csv"

train_full = data_module.read_train()
test_df = data_module.read_test()

BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]
C_PER_LABEL = {"HS": 2.0, "TR": 1.8, "AG": 1.5}
TEMPERATURE = 1.04

print("Vectorizando features V2_triple...", flush=True)
mats_tr_full, mats_te = [], []
for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
    v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
    mats_tr_full.append(v.fit_transform(train_full["text"]))
    mats_te.append(v.transform(test_df["text"]))
X_tr_full = sparse.hstack(mats_tr_full).tocsr()
X_te = sparse.hstack(mats_te).tocsr()

prob_lr_test = np.zeros((len(test_df), len(ETIQUETAS)), dtype=float)
prob_sgd_test = np.zeros((len(test_df), len(ETIQUETAS)), dtype=float)
prob_saga_test = np.zeros((len(test_df), len(ETIQUETAS)), dtype=float)

print("Entrenando clasificadores...", flush=True)
for j, et in enumerate(ETIQUETAS):
    y_col = train_full[et]
    c_val = C_PER_LABEL[et]
    clf_lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
    clf_lr.fit(X_tr_full, y_col)
    prob_lr_test[:, j] = clf_lr.predict_proba(X_te)[:, 1]

    clf_sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
    clf_sgd.fit(X_tr_full, y_col)
    prob_sgd_test[:, j] = clf_sgd.predict_proba(X_te)[:, 1]

    clf_saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
    clf_saga.fit(X_tr_full, y_col)
    prob_saga_test[:, j] = clf_saga.predict_proba(X_te)[:, 1]

prob_blend_test = 0.80 * prob_lr_test + 0.05 * prob_sgd_test + 0.15 * prob_saga_test
eps = 1e-6
prob_cal_test = expit(logit(np.clip(prob_blend_test, eps, 1.0 - eps)) / TEMPERATURE)

print("\nProbabilidades base para filas de interes:")
for idx in [300, 354, 583, 917, 1006, 1049]:
    p = prob_cal_test[idx]
    print(f"Index {idx} | id {test_df.iloc[idx].id} | pHS={p[0]:.4f}, pTR={p[1]:.4f}, pAG={p[2]:.4f}")

# Examinar candidatos a rescate AG: P(TR) >= 0.58 y P(AG) >= 0.44
cond_rescue_all = (prob_cal_test[:, 1] >= 0.58) & (prob_cal_test[:, 2] >= 0.44) & (prob_cal_test[:, 2] < 0.46)
print(f"\nFilas en [0.44, 0.46) con P(TR) >= 0.58: {np.flatnonzero(cond_rescue_all)}")
for i in np.flatnonzero(cond_rescue_all):
    p = prob_cal_test[i]
    print(f"  Idx {i} (id {test_df.iloc[i].id}): pHS={p[0]:.4f}, pTR={p[1]:.4f}, pAG={p[2]:.4f} | {test_df.iloc[i].text}")

# Examinar candidatos a supresion AG comercial
lexico_comercial = ["sorteo", "entradas", "premio", "concurso", "fiesta exclusiva", "ganadoras del primer"]
print("\nFilas con lexico comercial en test:")
for i, t in enumerate(test_df["text"]):
    t_low = t.lower()
    if any(k in t_low for k in lexico_comercial):
        p = prob_cal_test[i]
        cumple_prob = (p[1] < 0.20) and (p[2] <= 0.470) and (p[2] >= 0.460)
        print(f"  Idx {i} (id {test_df.iloc[i].id}): pTR={p[1]:.4f}, pAG={p[2]:.4f}, cumple_corte={cumple_prob} | {t[:80]}")
