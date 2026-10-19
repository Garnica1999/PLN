import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path("/home/carlos/Git/PLN/kaggle_toxic_2026_puj")
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

# Load splits
zf = np.load(RAIZ / "features/oof_v03_final.npz", allow_pickle=True)
zb = np.load(RAIZ / "features/v2_oof_base_corrected.npz", allow_pickle=True)
pliegues_can = zf["pliegues"].astype(int)
grupos = zb["grupos"].astype(np.int64)

train_full = data_module.read_train()
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

particiones = [
    ("Canonica", pliegues_can),
    ("Part47", pliegues_47),
]

BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]

UMBRALES = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

def aplicar_regla_ascendente(pred: np.ndarray) -> np.ndarray:
    p = pred.copy()
    mask = (p[:, 1] == 1) & (p[:, 0] == 0)
    p[mask, 0] = 1
    return p

print("Construyendo matrices TF-IDF para Canonica y Part47...", flush=True)
t0 = time.time()

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

mats_can = precompute_matrices(pliegues_can)
mats_p47 = precompute_matrices(pliegues_47)
print(f"Matrices listas en {time.time()-t0:.1f}s. Dimensiones: {mats_can[0][2].shape}", flush=True)

# Helper to compute NBSVM log-ratio feature transformation
def compute_nbsvm_features(X_tr, y_tr, X_va, alpha=1.0):
    # p = alpha + sum(X[y==1])
    # q = alpha + sum(X[y==0])
    p = np.array(X_tr[y_tr == 1].sum(axis=0)) + alpha
    q = np.array(X_tr[y_tr == 0].sum(axis=0)) + alpha
    p = p / np.sum(p)
    q = q / np.sum(q)
    r = np.log(p / q) # shape (1, D)
    r_diag = sparse.diags(r.flatten())
    return X_tr.dot(r_diag), X_va.dot(r_diag)

# Evaluation helper
def eval_cv(model_name, fit_fn):
    results = {}
    for part_name, mats in [("Canonica", mats_can), ("Part47", mats_p47)]:
        n_samples = len(train_dev)
        prob = np.zeros((n_samples, len(ETIQUETAS)), dtype=float)
        for idx_tr, idx_va, X_tr, X_va in mats:
            for j, et in enumerate(ETIQUETAS):
                y_tr_j = y_dev[idx_tr, j]
                prob[idx_va, j] = fit_fn(X_tr, y_tr_j, X_va, et, j)
        pred_raw = np.column_stack([(prob[:, j] >= UMBRALES[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
        pred_asc = aplicar_regla_ascendente(pred_raw)
        m = metricas_multietiqueta(y_dev, pred_asc)
        results[part_name] = {"f1": m["f1_media_columnas"], "prob": prob, "pred": pred_asc}
    media = (results["Canonica"]["f1"] + results["Part47"]["f1"]) / 2.0
    return media, results

# 1. P114 Components
print("\n--- Evaluando Componentes P114 Baseline ---", flush=True)
def fit_lr(X_tr, y_tr, X_va, et, j):
    c_val = {"HS": 2.0, "TR": 1.8, "AG": 1.5}[et]
    clf = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_va)[:, 1]

media_lr, res_lr = eval_cv("LR_LibLinear", fit_lr)
print(f"LR LibLinear: Can={res_lr['Canonica']['f1']:.6f}, P47={res_lr['Part47']['f1']:.6f}, Med={media_lr:.6f}")

def fit_sgd(X_tr, y_tr, X_va, et, j):
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_va)[:, 1]

media_sgd, res_sgd = eval_cv("SGD_log", fit_sgd)
print(f"SGD log_loss: Can={res_sgd['Canonica']['f1']:.6f}, P47={res_sgd['Part47']['f1']:.6f}, Med={media_sgd:.6f}")

def fit_saga(X_tr, y_tr, X_va, et, j):
    c_val = {"HS": 2.0, "TR": 1.8, "AG": 1.5}[et]
    clf = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_va)[:, 1]

media_saga, res_saga = eval_cv("SAGA_ElasticNet", fit_saga)
print(f"SAGA ElasticNet: Can={res_saga['Canonica']['f1']:.6f}, P47={res_saga['Part47']['f1']:.6f}, Med={media_saga:.6f}")

# Reconstruct P114
prob_p114_can = 0.80 * res_lr["Canonica"]["prob"] + 0.05 * res_sgd["Canonica"]["prob"] + 0.15 * res_saga["Canonica"]["prob"]
prob_p114_p47 = 0.80 * res_lr["Part47"]["prob"] + 0.05 * res_sgd["Part47"]["prob"] + 0.15 * res_saga["Part47"]["prob"]
eps = 1e-6
prob_p114_can = expit(logit(np.clip(prob_p114_can, eps, 1 - eps)) / 1.04)
prob_p114_p47 = expit(logit(np.clip(prob_p114_p47, eps, 1 - eps)) / 1.04)
pred_p114_can = aplicar_regla_ascendente(np.column_stack([(prob_p114_can[:, j] >= UMBRALES[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
pred_p114_p47 = aplicar_regla_ascendente(np.column_stack([(prob_p114_p47[:, j] >= UMBRALES[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
f1_p114_can = metricas_multietiqueta(y_dev, pred_p114_can)["f1_media_columnas"]
f1_p114_p47 = metricas_multietiqueta(y_dev, pred_p114_p47)["f1_media_columnas"]
print(f"P114 Reconstructed: Can={f1_p114_can:.6f}, P47={f1_p114_p47:.6f}, Med={(f1_p114_can+f1_p114_p47)/2:.6f}")

# 2. Complement Naive Bayes
print("\n--- Evaluando Complement Naive Bayes ---", flush=True)
for alpha in [0.2, 0.5, 1.0, 2.0]:
    def fit_cnb(X_tr, y_tr, X_va, et, j):
        clf = ComplementNB(alpha=alpha, norm=True)
        clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_va)[:, 1]
    med, res = eval_cv(f"CNB_{alpha}", fit_cnb)
    print(f"CNB(alpha={alpha}): Can={res['Canonica']['f1']:.6f}, P47={res['Part47']['f1']:.6f}, Med={med:.6f}")

# 3. Multinomial Naive Bayes
print("\n--- Evaluando Multinomial Naive Bayes ---", flush=True)
for alpha in [0.2, 0.5, 1.0]:
    def fit_mnb(X_tr, y_tr, X_va, et, j):
        clf = MultinomialNB(alpha=alpha)
        clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_va)[:, 1]
    med, res = eval_cv(f"MNB_{alpha}", fit_mnb)
    print(f"MNB(alpha={alpha}): Can={res['Canonica']['f1']:.6f}, P47={res['Part47']['f1']:.6f}, Med={med:.6f}")

# 4. NBSVM (Wang & Manning Log-Ratio + Logistic Regression)
print("\n--- Evaluando NBSVM (NB-weighted TF-IDF + LR) ---", flush=True)
for c_val in [0.5, 1.0, 2.0]:
    def fit_nbsvm(X_tr, y_tr, X_va, et, j):
        X_tr_nb, X_va_nb = compute_nbsvm_features(X_tr, y_tr, X_va, alpha=1.0)
        clf = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        clf.fit(X_tr_nb, y_tr)
        return clf.predict_proba(X_va_nb)[:, 1]
    med, res = eval_cv(f"NBSVM_C{c_val}", fit_nbsvm)
    print(f"NBSVM(C={c_val}): Can={res['Canonica']['f1']:.6f}, P47={res['Part47']['f1']:.6f}, Med={med:.6f}")

# 5. LinearSVC with CalibratedClassifierCV (Sigmoid / Platt scaling)
print("\n--- Evaluando Calibrated LinearSVC ---", flush=True)
for c_val in [0.2, 0.5, 1.0]:
    def fit_cal_svc(X_tr, y_tr, X_va, et, j):
        base_svc = LinearSVC(C=c_val, class_weight="balanced", random_state=42, max_iter=2000)
        clf = CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=3)
        clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_va)[:, 1]
    med, res = eval_cv(f"CalSVC_C{c_val}", fit_cal_svc)
    print(f"Calibrated LinearSVC(C={c_val}): Can={res['Canonica']['f1']:.6f}, P47={res['Part47']['f1']:.6f}, Med={med:.6f}")

# 6. MLPClassifier (Neural Network)
print("\n--- Evaluando MLPClassifier (Neural Network from scratch) ---", flush=True)
for hidden in [(32,), (64,)]:
    def fit_mlp(X_tr, y_tr, X_va, et, j):
        clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", alpha=1e-3, max_iter=200, random_state=42, early_stopping=True)
        clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_va)[:, 1]
    med, res = eval_cv(f"MLP_{hidden}", fit_mlp)
    print(f"MLP{hidden}: Can={res['Canonica']['f1']:.6f}, P47={res['Part47']['f1']:.6f}, Med={med:.6f}")
