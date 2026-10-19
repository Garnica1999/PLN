#!/usr/bin/env python
"""Investigacion avanzada P117:
1. Subspace vs Full space NBSVM (linear vs sqrt log-ratio)
2. Modelos alternativos diversos (Modified Huber, Calibrated Ridge, CNB)
3. Asymmetric AG tuning + Extreme-Hate Guided Targeting Rescue
4. Impacto en ambas particiones (Canonica y Part47) y proyeccion a Test
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path("/home/carlos/Git/PLN/kaggle_toxic_2026_puj")
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

# Cargar particiones
zf = np.load(RAIZ / "features/oof_v03_final.npz", allow_pickle=True)
zb = np.load(RAIZ / "features/v2_oof_base_corrected.npz", allow_pickle=True)
pliegues_can = zf["pliegues"].astype(int)
grupos = zb["grupos"].astype(np.int64)

train_full = data_module.read_train()
test_df = data_module.read_test()
particion = pd.read_csv(RAIZ / "data/processed/holdout_v2_ids.csv")
ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
train_dev = train_full[train_full["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
y_dev = train_dev[list(ETIQUETAS)].to_numpy(dtype=int)

# Particion 47
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

UMBRALES_BASE = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
C_PER_LABEL = {"HS": 2.0, "TR": 1.8, "AG": 1.5}

def aplicar_regla_ascendente(pred: np.ndarray) -> np.ndarray:
    p = pred.copy()
    mask = (p[:, 1] == 1) & (p[:, 0] == 0)
    p[mask, 0] = 1
    return p

print("Construyendo matrices TF-IDF completas y subespacios...", flush=True)
t0 = time.time()

def precompute_subspaces(pliegues):
    data = []
    for fold in sorted(set(pliegues)):
        idx_tr = np.flatnonzero(pliegues != fold)
        idx_va = np.flatnonzero(pliegues == fold)
        
        # Word subspace (1, 3)
        vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, sublinear_tf=True, lowercase=True)
        Xw_tr = vw.fit_transform(train_dev["text"].iloc[idx_tr])
        Xw_va = vw.transform(train_dev["text"].iloc[idx_va])
        
        # Char subspace (char_wb 3,5 + char 1,6)
        vc1 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, lowercase=True)
        vc2 = TfidfVectorizer(analyzer="char", ngram_range=(1, 6), min_df=2, sublinear_tf=True, lowercase=True)
        Xc_tr = sparse.hstack([vc1.fit_transform(train_dev["text"].iloc[idx_tr]), vc2.fit_transform(train_dev["text"].iloc[idx_tr])]).tocsr()
        Xc_va = sparse.hstack([vc1.transform(train_dev["text"].iloc[idx_va]), vc2.transform(train_dev["text"].iloc[idx_va])]).tocsr()
        
        # Full V2_triple
        v_w12 = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
        Xfull_tr = sparse.hstack([v_w12.fit_transform(train_dev["text"].iloc[idx_tr]), Xc_tr]).tocsr()
        Xfull_va = sparse.hstack([v_w12.transform(train_dev["text"].iloc[idx_va]), Xc_va]).tocsr()
        
        data.append({
            "idx_tr": idx_tr, "idx_va": idx_va,
            "Xw_tr": Xw_tr, "Xw_va": Xw_va,
            "Xc_tr": Xc_tr, "Xc_va": Xc_va,
            "Xfull_tr": Xfull_tr, "Xfull_va": Xfull_va,
        })
    return data

subs_can = precompute_subspaces(pliegues_can)
subs_p47 = precompute_subspaces(pliegues_47)
print(f"Subespacios listos en {time.time()-t0:.1f}s.", flush=True)

# Helper para calcular log-ratios NBSVM
def nbsvm_transform(X_tr, y_tr, X_va, alpha=1.0, sublinear_r=False):
    p = np.array(X_tr[y_tr == 1].sum(axis=0)) + alpha
    q = np.array(X_tr[y_tr == 0].sum(axis=0)) + alpha
    p = p / np.sum(p)
    q = q / np.sum(q)
    r = np.log(p / q)
    if sublinear_r:
        r = np.sign(r) * np.sqrt(np.abs(r))
    r_diag = sparse.diags(r.flatten())
    return X_tr.dot(r_diag), X_va.dot(r_diag)

def train_partition_models(subs):
    n = len(train_dev)
    prob_lr = np.zeros((n, 3))
    prob_sgd = np.zeros((n, 3))
    prob_saga = np.zeros((n, 3))
    prob_nbsvm_full = np.zeros((n, 3))
    prob_nbsvm_full_sqrt = np.zeros((n, 3))
    prob_nbsvm_word = np.zeros((n, 3))
    prob_nbsvm_word_sqrt = np.zeros((n, 3))
    prob_sgd_huber = np.zeros((n, 3))
    prob_ridge_cal = np.zeros((n, 3))

    for item in subs:
        idx_tr, idx_va = item["idx_tr"], item["idx_va"]
        Xfull_tr, Xfull_va = item["Xfull_tr"], item["Xfull_va"]
        Xw_tr, Xw_va = item["Xw_tr"], item["Xw_va"]

        for j, et in enumerate(ETIQUETAS):
            y_col = y_dev[idx_tr, j]
            c_val = C_PER_LABEL[et]

            # 1. LR LibLinear en Xfull
            lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr.fit(Xfull_tr, y_col)
            prob_lr[idx_va, j] = lr.predict_proba(Xfull_va)[:, 1]

            # 2. SGD log_loss en Xfull
            sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
            sgd.fit(Xfull_tr, y_col)
            prob_sgd[idx_va, j] = sgd.predict_proba(Xfull_va)[:, 1]

            # 3. SAGA ElasticNet en Xfull
            saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
            saga.fit(Xfull_tr, y_col)
            prob_saga[idx_va, j] = saga.predict_proba(Xfull_va)[:, 1]

            # 4. NBSVM Full linear
            Xtr_nb, Xva_nb = nbsvm_transform(Xfull_tr, y_col, Xfull_va, alpha=1.0, sublinear_r=False)
            nb_clf = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            nb_clf.fit(Xtr_nb, y_col)
            prob_nbsvm_full[idx_va, j] = nb_clf.predict_proba(Xva_nb)[:, 1]

            # 5. NBSVM Full sqrt
            Xtr_nb_sq, Xva_nb_sq = nbsvm_transform(Xfull_tr, y_col, Xfull_va, alpha=1.0, sublinear_r=True)
            nb_clf_sq = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            nb_clf_sq.fit(Xtr_nb_sq, y_col)
            prob_nbsvm_full_sqrt[idx_va, j] = nb_clf_sq.predict_proba(Xva_nb_sq)[:, 1]

            # 6. NBSVM Word subspace linear
            Xtr_w_nb, Xva_w_nb = nbsvm_transform(Xw_tr, y_col, Xw_va, alpha=1.0, sublinear_r=False)
            nb_w = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            nb_w.fit(Xtr_w_nb, y_col)
            prob_nbsvm_word[idx_va, j] = nb_w.predict_proba(Xva_w_nb)[:, 1]

            # 7. NBSVM Word subspace sqrt
            Xtr_w_sq, Xva_w_sq = nbsvm_transform(Xw_tr, y_col, Xw_va, alpha=1.0, sublinear_r=True)
            nb_w_sq = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            nb_w_sq.fit(Xtr_w_sq, y_col)
            prob_nbsvm_word_sqrt[idx_va, j] = nb_w_sq.predict_proba(Xva_w_sq)[:, 1]

            # 8. SGD Modified Huber
            sgd_h = SGDClassifier(loss="modified_huber", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
            sgd_h.fit(Xfull_tr, y_col)
            prob_sgd_huber[idx_va, j] = sgd_h.predict_proba(Xfull_va)[:, 1]

            # 9. Calibrated RidgeClassifier
            rc = RidgeClassifier(alpha=2.0, class_weight="balanced", random_state=42)
            cal_rc = CalibratedClassifierCV(estimator=rc, method="sigmoid", cv=3)
            cal_rc.fit(Xfull_tr, y_col)
            prob_ridge_cal[idx_va, j] = cal_rc.predict_proba(Xfull_va)[:, 1]

    return {
        "lr": prob_lr,
        "sgd": prob_sgd,
        "saga": prob_saga,
        "nbsvm_full": prob_nbsvm_full,
        "nbsvm_full_sqrt": prob_nbsvm_full_sqrt,
        "nbsvm_word": prob_nbsvm_word,
        "nbsvm_word_sqrt": prob_nbsvm_word_sqrt,
        "sgd_huber": prob_sgd_huber,
        "ridge_cal": prob_ridge_cal,
    }

print("Entrenando modelos en Canonica...", flush=True)
mods_can = train_partition_models(subs_can)
print("Entrenando modelos en Part47...", flush=True)
mods_p47 = train_partition_models(subs_p47)

def score_pipeline(prob_c, prob_47, th=UMBRALES_BASE, rescue_tr_from_hs=False, hs_thresh=0.75, tr_window=(0.485, 0.510)):
    eps = 1e-6
    # Post proceso
    def get_pred(prob):
        pred = np.column_stack([(prob[:, j] >= th[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
        if rescue_tr_from_hs:
            # Si HS >= hs_thresh y TR en tr_window => rescatar TR = 1
            mask_rescue = (prob[:, 0] >= hs_thresh) & (prob[:, 1] >= tr_window[0]) & (prob[:, 1] < tr_window[1])
            pred[mask_rescue, 1] = 1
        # Regla ascendente TR=1 => HS=1
        pred = aplicar_regla_ascendente(pred)
        return pred

    pred_c = get_pred(prob_c)
    pred_47 = get_pred(prob_47)
    f1_c = metricas_multietiqueta(y_dev, pred_c)["f1_media_columnas"]
    f1_47 = metricas_multietiqueta(y_dev, pred_47)["f1_media_columnas"]
    return f1_c, f1_47, (f1_c + f1_47) / 2.0

# P114 Baseline
p114_c = 0.80 * mods_can["lr"] + 0.05 * mods_can["sgd"] + 0.15 * mods_can["saga"]
p114_47 = 0.80 * mods_p47["lr"] + 0.05 * mods_p47["sgd"] + 0.15 * mods_p47["saga"]
eps = 1e-6
p114_c = expit(logit(np.clip(p114_c, eps, 1 - eps)) / 1.04)
p114_47 = expit(logit(np.clip(p114_47, eps, 1 - eps)) / 1.04)

f1_base_c, f1_base_47, f1_base_med = score_pipeline(p114_c, p114_47)
print(f"\n=======================================================", flush=True)
print(f"P114 Baseline: Canonica={f1_base_c:.6f}, Part47={f1_base_47:.6f}, Media={f1_base_med:.6f}", flush=True)
print(f"=======================================================\n", flush=True)

# Evaluacion de variantes NBSVM en ensamble
print("=== 1. COMPARATIVA DE VARIANTES NBSVM EN EL ENSAMBLE ===")
for nb_key in ["nbsvm_full", "nbsvm_full_sqrt", "nbsvm_word", "nbsvm_word_sqrt"]:
    for w_nb in [0.08, 0.10, 0.12, 0.15]:
        w_lr = 0.85 - w_nb
        w_saga = 0.10
        w_sgd = 0.05
        b_c = w_lr * mods_can["lr"] + w_saga * mods_can["saga"] + w_sgd * mods_can["sgd"] + w_nb * mods_can[nb_key]
        b_47 = w_lr * mods_p47["lr"] + w_saga * mods_p47["saga"] + w_sgd * mods_p47["sgd"] + w_nb * mods_p47[nb_key]
        for t in [1.02, 1.04]:
            b_c_cal = expit(logit(np.clip(b_c, eps, 1 - eps)) / t)
            b_47_cal = expit(logit(np.clip(b_47, eps, 1 - eps)) / t)
            c, p, m = score_pipeline(b_c_cal, b_47_cal)
            if m > f1_base_med:
                print(f"  {nb_key} w={w_nb:.2f} T={t}: Can={c:.6f} (+{c-f1_base_c:+.6f}), P47={p:.6f} (+{p-f1_base_47:+.6f}), Med={m:.6f} (+{m-f1_base_med:+.6f})")

# Evaluacion de modelos adicionales (Huber, Calibrated Ridge)
print("\n=== 2. ENSAMBLE MULTI-FAMILIA CON MODIFIED HUBER Y RIDGE ===")
for w_huber in [0.02, 0.04]:
    for w_ridge in [0.0, 0.03]:
        w_nb = 0.12
        w_saga = 0.10
        w_sgd = 0.05 - w_huber
        w_lr = 1.0 - (w_nb + w_saga + w_sgd + w_huber + w_ridge)
        b_c = (w_lr * mods_can["lr"] + w_saga * mods_can["saga"] + w_sgd * mods_can["sgd"] + 
               w_nb * mods_can["nbsvm_full_sqrt"] + w_huber * mods_can["sgd_huber"] + w_ridge * mods_can["ridge_cal"])
        b_47 = (w_lr * mods_p47["lr"] + w_saga * mods_p47["saga"] + w_sgd * mods_p47["sgd"] + 
                w_nb * mods_p47["nbsvm_full_sqrt"] + w_huber * mods_p47["sgd_huber"] + w_ridge * mods_p47["ridge_cal"])
        b_c_cal = expit(logit(np.clip(b_c, eps, 1 - eps)) / 1.04)
        b_47_cal = expit(logit(np.clip(b_47, eps, 1 - eps)) / 1.04)
        c, p, m = score_pipeline(b_c_cal, b_47_cal)
        print(f"  Huber={w_huber} Ridge={w_ridge} (w_lr={w_lr:.2f}): Can={c:.6f}, P47={p:.6f}, Med={m:.6f} (Delta={m-f1_base_med:+.6f})")

# Evaluacion de afinamiento de umbrales y targeting rescue
print("\n=== 3. AFINAMIENTO DE UMBRAL AG Y RESCATE DE TARGETING (TR) ===")
# Mejor ensamble base: 70% LR + 15% NBSVM_full_sqrt + 10% SAGA + 5% SGD, T=1.04
best_b_c = 0.70 * mods_can["lr"] + 0.15 * mods_can["nbsvm_full_sqrt"] + 0.10 * mods_can["saga"] + 0.05 * mods_can["sgd"]
best_b_47 = 0.70 * mods_p47["lr"] + 0.15 * mods_p47["nbsvm_full_sqrt"] + 0.10 * mods_p47["saga"] + 0.05 * mods_p47["sgd"]
best_b_c_cal = expit(logit(np.clip(best_b_c, eps, 1 - eps)) / 1.04)
best_b_47_cal = expit(logit(np.clip(best_b_47, eps, 1 - eps)) / 1.04)

for th_ag in [0.460, 0.463, 0.465, 0.468, 0.470]:
    for rescue in [False, True]:
        for hs_th in [0.70, 0.75, 0.80]:
            if not rescue and hs_th != 0.75:
                continue
            th_curr = {"HS": 0.44, "TR": 0.51, "AG": th_ag}
            c, p, m = score_pipeline(best_b_c_cal, best_b_47_cal, th=th_curr, rescue_tr_from_hs=rescue, hs_thresh=hs_th)
            print(f"  AG_th={th_ag} Rescue={rescue} (HS_th={hs_th}): Can={c:.6f} (+{c-f1_base_c:+.6f}), P47={p:.6f} (+{p-f1_base_47:+.6f}), Med={m:.6f} (+{m-f1_base_med:+.6f})")
