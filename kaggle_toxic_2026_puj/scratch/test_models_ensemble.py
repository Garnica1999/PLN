import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path("/home/carlos/Git/PLN/kaggle_toxic_2026_puj")
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

OOF_FINAL = RAIZ / "features" / "oof_v03_final.npz"
OOF_BASE = RAIZ / "features" / "v2_oof_base_corrected.npz"
IDS_PATH = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"

zf = np.load(OOF_FINAL, allow_pickle=True)
zb = np.load(OOF_BASE, allow_pickle=True)
pliegues_can = zf["pliegues"].astype(int)
y_true = zf["y_true"].astype(int)
grupos = zb["grupos"].astype(np.int64)

train = data_module.read_train()
particion = pd.read_csv(IDS_PATH)
ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
train_dev = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)

# Particion 47
combinacion = np.array([f"{a}{b}{c}" for a, b, c in y_true])
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
pliegues_47 = np.full(len(y_true), -1, dtype=int)
for num, (_, val) in enumerate(sgkf.split(np.arange(len(y_true)).reshape(-1, 1), combinacion, groups=grupos)):
    pliegues_47[val] = num + 1

def aplicar_regla_ascendente(pred):
    p = pred.copy()
    mask = (p[:, 1] == 1) & (p[:, 0] == 0)
    p[mask, 0] = 1
    return p, int(mask.sum())

def f1_media_fast(y_t, y_p):
    tp = np.count_nonzero(y_t & y_p, axis=0)
    total = y_t.sum(axis=0) + y_p.sum(axis=0)
    f1 = np.where(total == 0, 0.0, 2.0 * tp / total)
    return float(np.mean(f1))

BLOQUES_P108 = [
    ("word", (1, 2), 2),
    ("char_wb", (3, 5), 2),
    ("char", (1, 6), 2),
]

def construir_matrices_folds(pliegues_arr, bloques):
    folds_data = []
    for fold in sorted(set(pliegues_arr)):
        idx_tr = np.flatnonzero(pliegues_arr != fold)
        idx_va = np.flatnonzero(pliegues_arr == fold)
        mats_tr, mats_va = [], []
        for analyzer, ngram, mdf in bloques:
            v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=mdf, sublinear_tf=True, lowercase=True)
            mats_tr.append(v.fit_transform(train_dev["text"].iloc[idx_tr]))
            mats_va.append(v.transform(train_dev["text"].iloc[idx_va]))
        folds_data.append((idx_tr, idx_va, sparse.hstack(mats_tr).tocsr(), sparse.hstack(mats_va).tocsr()))
    return folds_data

print("Precomputando matrices P108 (min_df=2, 86k features)...")
sys.stdout.flush()
folds_can = construir_matrices_folds(pliegues_can, BLOQUES_P108)
folds_47 = construir_matrices_folds(pliegues_47, BLOQUES_P108)
print("Matrices precomputadas.")
sys.stdout.flush()

# Umbrales P108 (congelados de V03)
th_p108 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

def eval_model(folds_data, model_fn):
    proba = np.zeros((len(train_dev), len(ETIQUETAS)), dtype=float)
    for idx_tr, idx_va, m_tr, m_va in folds_data:
        for j, etiqueta in enumerate(ETIQUETAS):
            clf = model_fn(etiqueta)
            clf.fit(m_tr, train_dev[etiqueta].iloc[idx_tr])
            if hasattr(clf, "predict_proba"):
                proba[idx_va, j] = clf.predict_proba(m_va)[:, 1]
            else:
                # decision function sigmoidal
                df = clf.decision_function(m_va)
                proba[idx_va, j] = 1.0 / (1.0 + np.exp(-df))
    return proba

# 1. P108 LogisticRegression Base
print("\n--- 1. EVALUANDO P108 LR BASE ---")
prob_lr_can = eval_model(folds_can, lambda e: LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42))
prob_lr_47 = eval_model(folds_47, lambda e: LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42))

pred_lr_can, _ = aplicar_regla_ascendente(np.column_stack([(prob_lr_can[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
pred_lr_47, _ = aplicar_regla_ascendente(np.column_stack([(prob_lr_47[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))

f1_lr_can = f1_media_fast(y_true, pred_lr_can)
f1_lr_47 = f1_media_fast(y_true, pred_lr_47)
print(f"P108 LR Base: Canonica={f1_lr_can:.6f} | Part47={f1_lr_47:.6f} | Media={(f1_lr_can+f1_lr_47)/2:.6f}")
sys.stdout.flush()

# 2. Complement Naive Bayes
print("\n--- 2. EVALUANDO COMPLEMENT NAIVE BAYES ---")
for alpha in [0.1, 0.5, 1.0, 2.0]:
    prob_cnb_can = eval_model(folds_can, lambda e: ComplementNB(alpha=alpha, norm=True))
    prob_cnb_47 = eval_model(folds_47, lambda e: ComplementNB(alpha=alpha, norm=True))
    # Evaluar umbral 0.5 o grid
    p_c, _ = aplicar_regla_ascendente((prob_cnb_can >= 0.5).astype(int))
    p_47, _ = aplicar_regla_ascendente((prob_cnb_47 >= 0.5).astype(int))
    print(f"CNB alpha={alpha}: Canonica={f1_media_fast(y_true, p_c):.6f} | Part47={f1_media_fast(y_true, p_47):.6f}")
    
    # Probar blend con P108 LR:
    for w in [0.05, 0.10, 0.15, 0.20]:
        p_blend_can = (1 - w) * prob_lr_can + w * prob_cnb_can
        p_blend_47 = (1 - w) * prob_lr_47 + w * prob_cnb_47
        pb_c, _ = aplicar_regla_ascendente(np.column_stack([(p_blend_can[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
        pb_47, _ = aplicar_regla_ascendente(np.column_stack([(p_blend_47[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
        print(f"  Blend LR + {int(w*100)}% CNB(a={alpha}): Canonica={f1_media_fast(y_true, pb_c):.6f} | Part47={f1_media_fast(y_true, pb_47):.6f} | Media={(f1_media_fast(y_true, pb_c)+f1_media_fast(y_true, pb_47))/2:.6f}")
    sys.stdout.flush()

# 3. SGDClassifier log_loss (equivalente a LR con regularizacion estocastica)
print("\n--- 3. EVALUANDO SGDClassifier log_loss ---")
for a in [1e-5, 5e-5, 1e-4, 5e-4]:
    prob_sgd_can = eval_model(folds_can, lambda e: SGDClassifier(loss="log_loss", penalty="l2", alpha=a, max_iter=2000, random_state=42, class_weight="balanced"))
    prob_sgd_47 = eval_model(folds_47, lambda e: SGDClassifier(loss="log_loss", penalty="l2", alpha=a, max_iter=2000, random_state=42, class_weight="balanced"))
    for w in [0.05, 0.10, 0.20]:
        p_blend_can = (1 - w) * prob_lr_can + w * prob_sgd_can
        p_blend_47 = (1 - w) * prob_lr_47 + w * prob_sgd_47
        pb_c, _ = aplicar_regla_ascendente(np.column_stack([(p_blend_can[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
        pb_47, _ = aplicar_regla_ascendente(np.column_stack([(p_blend_47[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
        print(f"  Blend LR + {int(w*100)}% SGD(alpha={a}): Canonica={f1_media_fast(y_true, pb_c):.6f} | Part47={f1_media_fast(y_true, pb_47):.6f} | Media={(f1_media_fast(y_true, pb_c)+f1_media_fast(y_true, pb_47))/2:.6f}")
    sys.stdout.flush()
