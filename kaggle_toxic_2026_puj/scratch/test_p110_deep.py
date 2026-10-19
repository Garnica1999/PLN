import sys
import time
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path("/home/carlos/Git/PLN/kaggle_toxic_2026_puj")
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS

OOF_FINAL = RAIZ / "features" / "oof_v03_final.npz"
OOF_BASE = RAIZ / "features" / "v2_oof_base_corrected.npz"
IDS_PATH = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"

zf = np.load(OOF_FINAL, allow_pickle=True)
zb = np.load(OOF_BASE, allow_pickle=True)
pliegues_can = zf["pliegues"].astype(int)
y_true = zf["y_true"].astype(int)
grupos = zb["grupos"].astype(np.int64)

train = data_module.read_train()
test = data_module.read_test()
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

th_p108 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

# 1. Extraer features de superficie de texto
def extraer_features_superficie(df_textos):
    feats = []
    for text in df_textos:
        l_char = len(text)
        words = text.split()
        l_words = len(words)
        n_upper = sum(1 for c in text if c.isupper())
        r_upper = n_upper / max(l_char, 1)
        n_excl = text.count("!") + text.count(chr(161)) if chr(161) in text else text.count("!")
        n_quest = text.count("?") + text.count(chr(191)) if chr(191) in text else text.count("?")
        n_digits = sum(1 for c in text if c.isdigit())
        n_mentions = text.count("@")
        n_urls = text.lower().count("http")
        feats.append([l_char, l_words, r_upper, n_excl, n_quest, n_digits, n_mentions, n_urls])
    return np.array(feats, dtype=float)

feats_sup_dev = extraer_features_superficie(train_dev["text"])
feats_sup_full = extraer_features_superficie(train["text"])
feats_sup_test = extraer_features_superficie(test["text"])

def construir_matrices_folds(pliegues_arr, bloques, incluir_sup=False):
    folds_data = []
    for fold in sorted(set(pliegues_arr)):
        idx_tr = np.flatnonzero(pliegues_arr != fold)
        idx_va = np.flatnonzero(pliegues_arr == fold)
        mats_tr, mats_va = [], []
        for analyzer, ngram, mdf in bloques:
            v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=mdf, sublinear_tf=True, lowercase=True)
            mats_tr.append(v.fit_transform(train_dev["text"].iloc[idx_tr]))
            mats_va.append(v.transform(train_dev["text"].iloc[idx_va]))
        if incluir_sup:
            scaler = StandardScaler()
            s_tr = sparse.csr_matrix(scaler.fit_transform(feats_sup_dev[idx_tr]))
            s_va = sparse.csr_matrix(scaler.transform(feats_sup_dev[idx_va]))
            mats_tr.append(s_tr)
            mats_va.append(s_va)
        folds_data.append((idx_tr, idx_va, sparse.hstack(mats_tr).tocsr(), sparse.hstack(mats_va).tocsr()))
    return folds_data

print("Construyendo matrices P108 estandar y con superficie...")
sys.stdout.flush()
folds_can_std = construir_matrices_folds(pliegues_can, BLOQUES_P108, incluir_sup=False)
folds_47_std = construir_matrices_folds(pliegues_47, BLOQUES_P108, incluir_sup=False)

folds_can_sup = construir_matrices_folds(pliegues_can, BLOQUES_P108, incluir_sup=True)
folds_47_sup = construir_matrices_folds(pliegues_47, BLOQUES_P108, incluir_sup=True)
print("Matrices construidas.")
sys.stdout.flush()

# Funcion para evaluar combinacion
def evaluar_config(folds_data_c, folds_data_47, clf_builder_lr, clf_builder_sgd=None, w_sgd=0.0):
    prob_c = np.zeros((len(train_dev), len(ETIQUETAS)), dtype=float)
    prob_47 = np.zeros((len(train_dev), len(ETIQUETAS)), dtype=float)

    for idx_tr, idx_va, m_tr, m_va in folds_data_c:
        for j, et in enumerate(ETIQUETAS):
            lr = clf_builder_lr(et)
            lr.fit(m_tr, train_dev[et].iloc[idx_tr])
            p_lr = lr.predict_proba(m_va)[:, 1]
            if clf_builder_sgd is not None and w_sgd > 0:
                sgd = clf_builder_sgd(et)
                sgd.fit(m_tr, train_dev[et].iloc[idx_tr])
                p_sgd = sgd.predict_proba(m_va)[:, 1]
                prob_c[idx_va, j] = (1.0 - w_sgd) * p_lr + w_sgd * p_sgd
            else:
                prob_c[idx_va, j] = p_lr

    for idx_tr, idx_va, m_tr, m_va in folds_data_47:
        for j, et in enumerate(ETIQUETAS):
            lr = clf_builder_lr(et)
            lr.fit(m_tr, train_dev[et].iloc[idx_tr])
            p_lr = lr.predict_proba(m_va)[:, 1]
            if clf_builder_sgd is not None and w_sgd > 0:
                sgd = clf_builder_sgd(et)
                sgd.fit(m_tr, train_dev[et].iloc[idx_tr])
                p_sgd = sgd.predict_proba(m_va)[:, 1]
                prob_47[idx_va, j] = (1.0 - w_sgd) * p_lr + w_sgd * p_sgd
            else:
                prob_47[idx_va, j] = p_lr

    pr_c, _ = aplicar_regla_ascendente(np.column_stack([(prob_c[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
    pr_47, _ = aplicar_regla_ascendente(np.column_stack([(prob_47[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
    f_c = f1_media_fast(y_true, pr_c)
    f_47 = f1_media_fast(y_true, pr_47)
    return f_c, f_47, (f_c + f_47) / 2

# Baseline P108
print("\n--- BASELINE P108 ---")
fc_b, f47_b, m_b = evaluar_config(folds_can_std, folds_47_std,
    lambda e: LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42))
print(f"P108 Base: Can={fc_b:.6f} | Part47={f47_b:.6f} | Media={m_b:.6f}")
sys.stdout.flush()

# Prueba A: Features de superficie (longitud, mayusculas, exclamaciones, etc.)
print("\n--- PRUEBA A: IMPACTO DE FEATURES DE SUPERFICIE ---")
fc_s, f47_s, m_s = evaluar_config(folds_can_sup, folds_47_sup,
    lambda e: LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42))
print(f"P108 + Superficie: Can={fc_s:.6f} ({fc_s-fc_b:+.5f}) | Part47={f47_s:.6f} ({f47_s-f47_b:+.5f}) | Media={m_s:.6f}")
sys.stdout.flush()

# Prueba B: SGD Blends finos sobre P108 estandar
print("\n--- PRUEBA B: BLENDS LR + SGD EN P108 ---")
configs_sgd = [
    (1e-5, 0.03),
    (1e-5, 0.05),
    (1e-5, 0.06),
    (5e-6, 0.04),
    (5e-6, 0.05),
    (5e-6, 0.06),
    (2e-5, 0.05),
]
for a, w in configs_sgd:
    fc, f47, m = evaluar_config(folds_can_std, folds_47_std,
        lambda e: LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42),
        lambda e: SGDClassifier(loss="log_loss", penalty="l2", alpha=a, max_iter=2000, random_state=42, class_weight="balanced"),
        w_sgd=w
    )
    print(f"Blend LR + {w*100:.1f}% SGD(a={a}): Can={fc:.6f} ({fc-fc_b:+.5f}) | Part47={f47:.6f} ({f47-f47_b:+.5f}) | Media={m:.6f} ({m-m_b:+.5f})")
sys.stdout.flush()

# Prueba C: C optimizado por etiqueta + SGD blend
print("\n--- PRUEBA C: C OPTIMIZADO + SGD BLEND ---")
c_opt_map = {"HS": 2.2, "TR": 3.0, "AG": 2.0}
fc_c, f47_c, m_c = evaluar_config(folds_can_std, folds_47_std,
    lambda e: LogisticRegression(C=c_opt_map[e], class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42))
print(f"LR con C={{HS:2.2, TR:3.0, AG:2.0}}: Can={fc_c:.6f} ({fc_c-fc_b:+.5f}) | Part47={f47_c:.6f} ({f47_c-f47_b:+.5f}) | Media={m_c:.6f}")

for a, w in [(5e-6, 0.05), (1e-5, 0.05)]:
    fc, f47, m = evaluar_config(folds_can_std, folds_47_std,
        lambda e: LogisticRegression(C=c_opt_map[e], class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42),
        lambda e: SGDClassifier(loss="log_loss", penalty="l2", alpha=a, max_iter=2000, random_state=42, class_weight="balanced"),
        w_sgd=w
    )
    print(f"LR(C_opt) + {w*100:.1f}% SGD(a={a}): Can={fc:.6f} ({fc-fc_b:+.5f}) | Part47={f47:.6f} ({f47-f47_b:+.5f}) | Media={m:.6f} ({m-m_b:+.5f})")
sys.stdout.flush()

# Evaluacion detallada de test para las dos mejores variantes
print("\n--- EVALUACION DE TEST Y PERFIL DE PREDICCIONES ---")
mats_tr_full, mats_te = [], []
for analyzer, ngram, mdf in BLOQUES_P108:
    v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=mdf, sublinear_tf=True, lowercase=True)
    mats_tr_full.append(v.fit_transform(train["text"]))
    mats_te.append(v.transform(test["text"]))
X_tr_full = sparse.hstack(mats_tr_full).tocsr()
X_te = sparse.hstack(mats_te).tocsr()

# LR P108
p_lr_p108 = np.zeros((len(test), len(ETIQUETAS)), dtype=float)
for j, et in enumerate(ETIQUETAS):
    clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
    clf.fit(X_tr_full, train[et])
    p_lr_p108[:, j] = clf.predict_proba(X_te)[:, 1]
pred_p108, _ = aplicar_regla_ascendente(np.column_stack([(p_lr_p108[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))

# SGD 5e-6
p_sgd_5e6 = np.zeros((len(test), len(ETIQUETAS)), dtype=float)
for j, et in enumerate(ETIQUETAS):
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
    clf.fit(X_tr_full, train[et])
    p_sgd_5e6[:, j] = clf.predict_proba(X_te)[:, 1]

# SGD 1e-5
p_sgd_1e5 = np.zeros((len(test), len(ETIQUETAS)), dtype=float)
for j, et in enumerate(ETIQUETAS):
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5, max_iter=2000, random_state=42, class_weight="balanced")
    clf.fit(X_tr_full, train[et])
    p_sgd_1e5[:, j] = clf.predict_proba(X_te)[:, 1]

# Blend 1: 95% LR + 5% SGD(5e-6)
p_b1 = 0.95 * p_lr_p108 + 0.05 * p_sgd_5e6
pred_b1, _ = aplicar_regla_ascendente(np.column_stack([(p_b1[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
diff_b1 = int((pred_p108 != pred_b1).sum())

# Blend 2: 95% LR + 5% SGD(1e-5)
p_b2 = 0.95 * p_lr_p108 + 0.05 * p_sgd_1e5
pred_b2, _ = aplicar_regla_ascendente(np.column_stack([(p_b2[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
diff_b2 = int((pred_p108 != pred_b2).sum())

print(f"P108 Oficial (LB 0.78855): Positivos HS={pred_p108[:,0].sum()}, TR={pred_p108[:,1].sum()}, AG={pred_p108[:,2].sum()} | Total={pred_p108.sum()} ({pred_p108.sum()/(len(test)*3)*100:.2f}%)")
print(f"Blend 1 (95% LR + 5% SGD 5e-6): Positivos HS={pred_b1[:,0].sum()}, TR={pred_b1[:,1].sum()}, AG={pred_b1[:,2].sum()} | Total={pred_b1.sum()} ({pred_b1.sum()/(len(test)*3)*100:.2f}%) | Diff={diff_b1} ({diff_b1/(len(test)*3)*100:.2f}%)")
print(f"Blend 2 (95% LR + 5% SGD 1e-5): Positivos HS={pred_b2[:,0].sum()}, TR={pred_b2[:,1].sum()}, AG={pred_b2[:,2].sum()} | Total={pred_b2.sum()} ({pred_b2.sum()/(len(test)*3)*100:.2f}%) | Diff={diff_b2} ({diff_b2/(len(test)*3)*100:.2f}%)")
