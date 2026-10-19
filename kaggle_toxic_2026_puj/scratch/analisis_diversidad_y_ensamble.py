import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path("/home/carlos/Git/PLN/kaggle_toxic_2026_puj")
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

zf = np.load(RAIZ / "features/oof_v03_final.npz", allow_pickle=True)
zb = np.load(RAIZ / "features/v2_oof_base_corrected.npz", allow_pickle=True)
pliegues_can = zf["pliegues"].astype(int)
grupos = zb["grupos"].astype(np.int64)

train_full = data_module.read_train()
particion = pd.read_csv(RAIZ / "data/processed/holdout_v2_ids.csv")
ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
train_dev = train_full[train_full["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
y_dev = train_dev[list(ETIQUETAS)].to_numpy(dtype=int)

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

UMBRALES_P114 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

def aplicar_regla_ascendente(pred: np.ndarray) -> np.ndarray:
    p = pred.copy()
    mask = (p[:, 1] == 1) & (p[:, 0] == 0)
    p[mask, 0] = 1
    return p

def precompute_matrices(pliegues):
    matrices = []
    for fold in sorted(set(pliegues)):
        idx_tr = np.flatnonzero(pliegues != fold)
        idx_va = np.flatnonzero(pliegues == fold)
        mats_tr, mats_va = [], []
        for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
            v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
            mats_tr.append(v.fit_transform(train_dev["text"].iloc[idx_tr]))
            mats_va.append(v.transform(train_dev["text"].iloc[idx_va]))
        matrices.append((idx_tr, idx_va, sparse.hstack(mats_tr).tocsr(), sparse.hstack(mats_va).tocsr()))
    return matrices

print("Precomputando matrices TF-IDF...", flush=True)
mats_can = precompute_matrices(pliegues_can)
mats_p47 = precompute_matrices(pliegues_47)

def compute_nbsvm_features(X_tr, y_tr, X_va, alpha=1.0):
    p = np.array(X_tr[y_tr == 1].sum(axis=0)) + alpha
    q = np.array(X_tr[y_tr == 0].sum(axis=0)) + alpha
    p = p / np.sum(p)
    q = q / np.sum(q)
    r = np.log(p / q)
    r_diag = sparse.diags(r.flatten())
    return X_tr.dot(r_diag), X_va.dot(r_diag)

def train_all_models_for_partition(mats):
    n = len(train_dev)
    prob_lr = np.zeros((n, 3))
    prob_sgd = np.zeros((n, 3))
    prob_saga = np.zeros((n, 3))
    prob_nbsvm = np.zeros((n, 3))
    prob_calsvc = np.zeros((n, 3))
    prob_mnb = np.zeros((n, 3))

    for idx_tr, idx_va, X_tr, X_va in mats:
        for j, et in enumerate(ETIQUETAS):
            y_col = y_dev[idx_tr, j]
            c_val = {"HS": 2.0, "TR": 1.8, "AG": 1.5}[et]

            # 1. LR LibLinear
            lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr.fit(X_tr, y_col)
            prob_lr[idx_va, j] = lr.predict_proba(X_va)[:, 1]

            # 2. SGD log_loss
            sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
            sgd.fit(X_tr, y_col)
            prob_sgd[idx_va, j] = sgd.predict_proba(X_va)[:, 1]

            # 3. SAGA ElasticNet
            saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
            saga.fit(X_tr, y_col)
            prob_saga[idx_va, j] = saga.predict_proba(X_va)[:, 1]

            # 4. NBSVM
            X_tr_nb, X_va_nb = compute_nbsvm_features(X_tr, y_col, X_va, alpha=1.0)
            nbsvm = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            nbsvm.fit(X_tr_nb, y_col)
            prob_nbsvm[idx_va, j] = nbsvm.predict_proba(X_va_nb)[:, 1]

            # 5. Calibrated LinearSVC
            base_svc = LinearSVC(C=0.5, class_weight="balanced", random_state=42, max_iter=2000)
            calsvc = CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=3)
            calsvc.fit(X_tr, y_col)
            prob_calsvc[idx_va, j] = calsvc.predict_proba(X_va)[:, 1]

            # 6. MNB
            mnb = MultinomialNB(alpha=0.2)
            mnb.fit(X_tr, y_col)
            prob_mnb[idx_va, j] = mnb.predict_proba(X_va)[:, 1]

    return {
        "lr": prob_lr,
        "sgd": prob_sgd,
        "saga": prob_saga,
        "nbsvm": prob_nbsvm,
        "calsvc": prob_calsvc,
        "mnb": prob_mnb,
    }

print("Entrenando modelos en Canonica...", flush=True)
preds_can = train_all_models_for_partition(mats_can)
print("Entrenando modelos en Part47...", flush=True)
preds_p47 = train_all_models_for_partition(mats_p47)

# Compute P114 for both
def get_p114_prob(d):
    eps = 1e-6
    blend = 0.80 * d["lr"] + 0.05 * d["sgd"] + 0.15 * d["saga"]
    return expit(logit(np.clip(blend, eps, 1 - eps)) / 1.04)

p114_prob_can = get_p114_prob(preds_can)
p114_prob_p47 = get_p114_prob(preds_p47)

# Helper to score probabilities
def score_proba(prob_can, prob_p47, th=UMBRALES_P114):
    p_can = aplicar_regla_ascendente(np.column_stack([(prob_can[:, j] >= th[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
    p_p47 = aplicar_regla_ascendente(np.column_stack([(prob_p47[:, j] >= th[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
    f1_c = metricas_multietiqueta(y_dev, p_can)["f1_media_columnas"]
    f1_47 = metricas_multietiqueta(y_dev, p_p47)["f1_media_columnas"]
    return f1_c, f1_47, (f1_c + f1_47) / 2.0

f1_c, f1_47, med_p114 = score_proba(p114_prob_can, p114_prob_p47)
print(f"\nP114 Champion Baseline: Can={f1_c:.6f}, P47={f1_47:.6f}, Media={med_p114:.6f}")

# Diversity and Error Correlation Analysis on Canonica
print("\n=== ANALISIS DE CORRELACION DE ERRORES Y DIVERSIDAD (vs P114) ===")
models = ["lr", "sgd", "saga", "nbsvm", "calsvc", "mnb"]
y_flat = y_dev.flatten()
p114_pred_flat = aplicar_regla_ascendente(np.column_stack([(p114_prob_can[:, j] >= UMBRALES_P114[e]).astype(int) for j, e in enumerate(ETIQUETAS)])).flatten()
p114_error_flat = y_flat - p114_pred_flat

for m_name in models:
    prob = preds_can[m_name]
    pred = aplicar_regla_ascendente(np.column_stack([(prob[:, j] >= UMBRALES_P114[e]).astype(int) for j, e in enumerate(ETIQUETAS)])).flatten()
    err = y_flat - pred
    
    # Pearson corr of errors
    corr_err = np.corrcoef(p114_error_flat, err)[0, 1]
    # Disagreement rate
    disagree = np.mean(p114_pred_flat != pred) * 100
    # Q-statistic
    n11 = np.sum((p114_pred_flat == y_flat) & (pred == y_flat))
    n00 = np.sum((p114_pred_flat != y_flat) & (pred != y_flat))
    n10 = np.sum((p114_pred_flat == y_flat) & (pred != y_flat))
    n01 = np.sum((p114_pred_flat != y_flat) & (pred == y_flat))
    q_stat = (n11 * n00 - n10 * n01) / (n11 * n00 + n10 * n01 + 1e-12)
    
    print(f"Modelo {m_name.upper():<8}: Corr Errores={corr_err:.4f} | Desacuerdo={disagree:.2f}% | Q-stat={q_stat:.4f} | Solo-P114 acierta={n10} | Solo-{m_name.upper()} acierta={n01}")

print("\n=== BLENDS DE P114 CON MODELOS ORTOGONALES ===")
# 1. P114 + NBSVM
print("\n--- 1. P114 + NBSVM ---")
for w in [0.02, 0.05, 0.10, 0.15, 0.20, 0.25]:
    b_can = (1 - w) * p114_prob_can + w * preds_can["nbsvm"]
    b_p47 = (1 - w) * p114_prob_p47 + w * preds_p47["nbsvm"]
    c, p, m = score_proba(b_can, b_p47)
    delta = m - med_p114
    print(f"  P114 + {int(w*100):2d}% NBSVM: Can={c:.6f}, P47={p:.6f}, Media={m:.6f} (Delta={delta:+.6f})")

# 2. P114 + Calibrated LinearSVC
print("\n--- 2. P114 + Calibrated LinearSVC ---")
for w in [0.02, 0.05, 0.10, 0.15, 0.20]:
    b_can = (1 - w) * p114_prob_can + w * preds_can["calsvc"]
    b_p47 = (1 - w) * p114_prob_p47 + w * preds_p47["calsvc"]
    c, p, m = score_proba(b_can, b_p47)
    delta = m - med_p114
    print(f"  P114 + {int(w*100):2d}% CalSVC: Can={c:.6f}, P47={p:.6f}, Media={m:.6f} (Delta={delta:+.6f})")

# 3. P114 + MNB
print("\n--- 3. P114 + MultinomialNB ---")
for w in [0.01, 0.02, 0.03, 0.05]:
    b_can = (1 - w) * p114_prob_can + w * preds_can["mnb"]
    b_p47 = (1 - w) * p114_prob_p47 + w * preds_p47["mnb"]
    c, p, m = score_proba(b_can, b_p47)
    delta = m - med_p114
    print(f"  P114 + {int(w*100):2d}% MNB: Can={c:.6f}, P47={p:.6f}, Media={m:.6f} (Delta={delta:+.6f})")

# 4. Multi-family quad blend: LR (70%) + SAGA (10%) + SGD (5%) + NBSVM (10%) + CalSVC (5%)
print("\n--- 4. Multi-family Blends ---")
configs = [
    ("LR75 + SAGA10 + SGD5 + NBSVM10", {"lr": 0.75, "saga": 0.10, "sgd": 0.05, "nbsvm": 0.10, "calsvc": 0.0}),
    ("LR70 + SAGA10 + SGD5 + NBSVM15", {"lr": 0.70, "saga": 0.10, "sgd": 0.05, "nbsvm": 0.15, "calsvc": 0.0}),
    ("LR70 + SAGA10 + SGD5 + NBSVM10 + CalSVC5", {"lr": 0.70, "saga": 0.10, "sgd": 0.05, "nbsvm": 0.10, "calsvc": 0.05}),
    ("LR65 + SAGA15 + SGD5 + NBSVM10 + CalSVC5", {"lr": 0.65, "saga": 0.15, "sgd": 0.05, "nbsvm": 0.10, "calsvc": 0.05}),
    ("LR70 + SAGA15 + SGD5 + NBSVM10", {"lr": 0.70, "saga": 0.15, "sgd": 0.05, "nbsvm": 0.10, "calsvc": 0.0}),
]

for desc, weights in configs:
    eps = 1e-6
    b_can = sum(weights[k] * preds_can[k] for k in weights)
    b_p47 = sum(weights[k] * preds_p47[k] for k in weights)
    # Apply temperature
    for temp in [1.02, 1.04, 1.06]:
        b_can_cal = expit(logit(np.clip(b_can, eps, 1 - eps)) / temp)
        b_p47_cal = expit(logit(np.clip(b_p47, eps, 1 - eps)) / temp)
        c, p, m = score_proba(b_can_cal, b_p47_cal)
        delta = m - med_p114
        print(f"  {desc} (T={temp}): Can={c:.6f}, P47={p:.6f}, Media={m:.6f} (Delta={delta:+.6f})")
