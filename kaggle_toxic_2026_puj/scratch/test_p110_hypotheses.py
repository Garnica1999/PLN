import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
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

t0 = time.time()
print("Precomputando matrices P108 (min_df=2, ~86k features)...")
sys.stdout.flush()
folds_can = construir_matrices_folds(pliegues_can, BLOQUES_P108)
folds_47 = construir_matrices_folds(pliegues_47, BLOQUES_P108)
print(f"Matrices listas en {time.time() - t0:.1f}s.")
sys.stdout.flush()

th_p108 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

# Precomputar predicciones OOF de Logistic Regression para distintos valores de C por etiqueta
c_candidates = [1.5, 1.8, 2.0, 2.2, 2.5, 3.0]
prob_lr_can = {et: {} for et in ETIQUETAS}
prob_lr_47 = {et: {} for et in ETIQUETAS}

print("Precomputando modelos LR individuales...")
sys.stdout.flush()
for et_idx, et in enumerate(ETIQUETAS):
    for c_val in c_candidates:
        # Canonica
        p_can = np.zeros(len(train_dev), dtype=float)
        for idx_tr, idx_va, m_tr, m_va in folds_can:
            clf = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            clf.fit(m_tr, train_dev[et].iloc[idx_tr])
            p_can[idx_va] = clf.predict_proba(m_va)[:, 1]
        prob_lr_can[et][c_val] = p_can

        # Particion 47
        p_47 = np.zeros(len(train_dev), dtype=float)
        for idx_tr, idx_va, m_tr, m_va in folds_47:
            clf = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            clf.fit(m_tr, train_dev[et].iloc[idx_tr])
            p_47[idx_va] = clf.predict_proba(m_va)[:, 1]
        prob_lr_47[et][c_val] = p_47

# Baseline P108 LR (C=2.0 en todos)
p_base_can = np.column_stack([prob_lr_can[e][2.0] for e in ETIQUETAS])
p_base_47 = np.column_stack([prob_lr_47[e][2.0] for e in ETIQUETAS])

pred_b_can, _ = aplicar_regla_ascendente(np.column_stack([(p_base_can[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
pred_b_47, _ = aplicar_regla_ascendente(np.column_stack([(p_base_47[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))

f1_can_base = f1_media_fast(y_true, pred_b_can)
f1_47_base = f1_media_fast(y_true, pred_b_47)
media_base = (f1_can_base + f1_47_base) / 2
print(f"\n=== BASELINE P108 LR C=2.0 ===")
print(f"Canonica: {f1_can_base:.6f} | Part47: {f1_47_base:.6f} | Media: {media_base:.6f}")
sys.stdout.flush()

# Grid search rapido de C por etiqueta (HS, TR, AG)
print("\n--- 1. GRID SEARCH C POR ETIQUETA ---")
mejores_c = []
for c_hs in c_candidates:
    for c_tr in c_candidates:
        for c_ag in c_candidates:
            p_c = np.column_stack([prob_lr_can["HS"][c_hs], prob_lr_can["TR"][c_tr], prob_lr_can["AG"][c_ag]])
            p_47 = np.column_stack([prob_lr_47["HS"][c_hs], prob_lr_47["TR"][c_tr], prob_lr_47["AG"][c_ag]])
            pr_c, _ = aplicar_regla_ascendente(np.column_stack([(p_c[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
            pr_47, _ = aplicar_regla_ascendente(np.column_stack([(p_47[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
            f_c = f1_media_fast(y_true, pr_c)
            f_47 = f1_media_fast(y_true, pr_47)
            m = (f_c + f_47) / 2
            if f_c >= f1_can_base and f_47 >= f1_47_base and m > media_base:
                mejores_c.append((m, f_c, f_47, c_hs, c_tr, c_ag))

mejores_c.sort(key=lambda x: x[0], reverse=True)
for m, f_c, f_47, c_hs, c_tr, c_ag in mejores_c[:5]:
    print(f"C={{HS:{c_hs}, TR:{c_tr}, AG:{c_ag}}}: Can={f_c:.6f} (+{f_c-f1_can_base:+.5f}) | Part47={f_47:.6f} (+{f_47-f1_47_base:+.5f}) | Media={m:.6f}")
sys.stdout.flush()

# 2. Precomputar SGDClassifier
print("\n--- 2. SGDClassifier log_loss ---")
prob_sgd_can = {et: {} for et in ETIQUETAS}
prob_sgd_47 = {et: {} for et in ETIQUETAS}
for et in ETIQUETAS:
    for a in [5e-6, 1e-5, 2e-5, 5e-5]:
        p_c = np.zeros(len(train_dev), dtype=float)
        for idx_tr, idx_va, m_tr, m_va in folds_can:
            clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=a, max_iter=2000, random_state=42, class_weight="balanced")
            clf.fit(m_tr, train_dev[et].iloc[idx_tr])
            p_c[idx_va] = clf.predict_proba(m_va)[:, 1]
        prob_sgd_can[et][a] = p_c

        p_47 = np.zeros(len(train_dev), dtype=float)
        for idx_tr, idx_va, m_tr, m_va in folds_47:
            clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=a, max_iter=2000, random_state=42, class_weight="balanced")
            clf.fit(m_tr, train_dev[et].iloc[idx_tr])
            p_47[idx_va] = clf.predict_proba(m_va)[:, 1]
        prob_sgd_47[et][a] = p_47

# Probar blending LR (C=2.0) + SGD (alpha=1e-5 o 5e-6)
mejores_sgd_blend = []
for a in [5e-6, 1e-5, 2e-5]:
    p_sgd_c = np.column_stack([prob_sgd_can[e][a] for e in ETIQUETAS])
    p_sgd_47 = np.column_stack([prob_sgd_47[e][a] for e in ETIQUETAS])
    for w in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
        b_c = (1 - w) * p_base_can + w * p_sgd_c
        b_47 = (1 - w) * p_base_47 + w * p_sgd_47
        pr_c, _ = aplicar_regla_ascendente(np.column_stack([(b_c[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
        pr_47, _ = aplicar_regla_ascendente(np.column_stack([(b_47[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
        f_c = f1_media_fast(y_true, pr_c)
        f_47 = f1_media_fast(y_true, pr_47)
        m = (f_c + f_47) / 2
        if f_c >= f1_can_base and f_47 >= f1_47_base:
            mejores_sgd_blend.append((m, f_c, f_47, a, w))

mejores_sgd_blend.sort(key=lambda x: x[0], reverse=True)
for m, f_c, f_47, a, w in mejores_sgd_blend[:5]:
    print(f"Blend LR + {w*100:.1f}% SGD(a={a}): Can={f_c:.6f} (+{f_c-f1_can_base:+.5f}) | Part47={f_47:.6f} (+{f_47-f1_47_base:+.5f}) | Media={m:.6f}")
sys.stdout.flush()

# 3. Precomputar LinearSVC Sigmoid
print("\n--- 3. LinearSVC Sigmoid ---")
prob_svc_can = {et: {} for et in ETIQUETAS}
prob_svc_47 = {et: {} for et in ETIQUETAS}
for et in ETIQUETAS:
    for c_svc in [0.5, 1.0, 1.5]:
        p_c = np.zeros(len(train_dev), dtype=float)
        for idx_tr, idx_va, m_tr, m_va in folds_can:
            clf = LinearSVC(C=c_svc, class_weight="balanced", max_iter=5000, random_state=42)
            clf.fit(m_tr, train_dev[et].iloc[idx_tr])
            df = clf.decision_function(m_va)
            p_c[idx_va] = 1.0 / (1.0 + np.exp(-df))
        prob_svc_can[et][c_svc] = p_c

        p_47 = np.zeros(len(train_dev), dtype=float)
        for idx_tr, idx_va, m_tr, m_va in folds_47:
            clf = LinearSVC(C=c_svc, class_weight="balanced", max_iter=5000, random_state=42)
            clf.fit(m_tr, train_dev[et].iloc[idx_tr])
            df = clf.decision_function(m_va)
            p_47[idx_va] = 1.0 / (1.0 + np.exp(-df))
        prob_svc_47[et][c_svc] = p_47

mejores_svc_blend = []
for c_svc in [0.5, 1.0, 1.5]:
    p_svc_c = np.column_stack([prob_svc_can[e][c_svc] for e in ETIQUETAS])
    p_svc_47 = np.column_stack([prob_svc_47[e][c_svc] for e in ETIQUETAS])
    for w in [0.02, 0.03, 0.05, 0.08, 0.10]:
        b_c = (1 - w) * p_base_can + w * p_svc_c
        b_47 = (1 - w) * p_base_47 + w * p_svc_47
        pr_c, _ = aplicar_regla_ascendente(np.column_stack([(b_c[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
        pr_47, _ = aplicar_regla_ascendente(np.column_stack([(b_47[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
        f_c = f1_media_fast(y_true, pr_c)
        f_47 = f1_media_fast(y_true, pr_47)
        m = (f_c + f_47) / 2
        if f_c >= f1_can_base and f_47 >= f1_47_base:
            mejores_svc_blend.append((m, f_c, f_47, c_svc, w))

mejores_svc_blend.sort(key=lambda x: x[0], reverse=True)
for m, f_c, f_47, c_svc, w in mejores_svc_blend[:5]:
    print(f"Blend LR + {w*100:.1f}% SVC(C={c_svc}): Can={f_c:.6f} (+{f_c-f1_can_base:+.5f}) | Part47={f_47:.6f} (+{f_47-f1_47_base:+.5f}) | Media={m:.6f}")
sys.stdout.flush()

# 4. Probar Triple Blend: LR + SGD + SVC
print("\n--- 4. TRIPLE BLEND: LR + SGD + SVC ---")
p_sgd_opt_c = np.column_stack([prob_sgd_can[e][1e-5] for e in ETIQUETAS])
p_sgd_opt_47 = np.column_stack([prob_sgd_47[e][1e-5] for e in ETIQUETAS])
p_svc_opt_c = np.column_stack([prob_svc_can[e][1.0] for e in ETIQUETAS])
p_svc_opt_47 = np.column_stack([prob_svc_47[e][1.0] for e in ETIQUETAS])

for w_sgd in [0.03, 0.05]:
    for w_svc in [0.02, 0.04]:
        w_lr = 1.0 - w_sgd - w_svc
        b_c = w_lr * p_base_can + w_sgd * p_sgd_opt_c + w_svc * p_svc_opt_c
        b_47 = w_lr * p_base_47 + w_sgd * p_sgd_opt_47 + w_svc * p_svc_opt_47
        pr_c, _ = aplicar_regla_ascendente(np.column_stack([(b_c[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
        pr_47, _ = aplicar_regla_ascendente(np.column_stack([(b_47[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
        f_c = f1_media_fast(y_true, pr_c)
        f_47 = f1_media_fast(y_true, pr_47)
        m = (f_c + f_47) / 2
        print(f"Triple ({w_lr:.2f} LR + {w_sgd:.2f} SGD + {w_svc:.2f} SVC): Can={f_c:.6f} (+{f_c-f1_can_base:+.5f}) | Part47={f_47:.6f} (+{f_47-f1_47_base:+.5f}) | Media={m:.6f}")
sys.stdout.flush()

# 5. Simulacion de Test y Recuento de Positivos
print("\n--- 5. SIMULACION EN TEST PARA MEJOR CONFIGURACION ---")
# Entrenar en train completo (1.731 filas)
mats_tr_full, mats_te = [], []
for analyzer, ngram, mdf in BLOQUES_P108:
    v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=mdf, sublinear_tf=True, lowercase=True)
    mats_tr_full.append(v.fit_transform(train["text"]))
    mats_te.append(v.transform(test["text"]))
X_tr_full = sparse.hstack(mats_tr_full).tocsr()
X_te = sparse.hstack(mats_te).tocsr()

# Predecir LR full
p_lr_test = np.zeros((len(test), len(ETIQUETAS)), dtype=float)
for j, et in enumerate(ETIQUETAS):
    clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
    clf.fit(X_tr_full, train[et])
    p_lr_test[:, j] = clf.predict_proba(X_te)[:, 1]

# Prediccion P108 oficial
pred_p108_test, _ = aplicar_regla_ascendente(np.column_stack([(p_lr_test[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
print(f"P108 Oficial Test Positivos: HS={pred_p108_test[:, 0].sum()}, TR={pred_p108_test[:, 1].sum()}, AG={pred_p108_test[:, 2].sum()} | Total={pred_p108_test.sum()}")

# SGD full
p_sgd_test = np.zeros((len(test), len(ETIQUETAS)), dtype=float)
for j, et in enumerate(ETIQUETAS):
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5, max_iter=2000, random_state=42, class_weight="balanced")
    clf.fit(X_tr_full, train[et])
    p_sgd_test[:, j] = clf.predict_proba(X_te)[:, 1]

# Blend 95% LR + 5% SGD
p_blend_test = 0.95 * p_lr_test + 0.05 * p_sgd_test
pred_blend_test, _ = aplicar_regla_ascendente(np.column_stack([(p_blend_test[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
diff_cells = int((pred_p108_test != pred_blend_test).sum())
print(f"Blend 95% LR + 5% SGD Test Positivos: HS={pred_blend_test[:, 0].sum()}, TR={pred_blend_test[:, 1].sum()}, AG={pred_blend_test[:, 2].sum()} | Total={pred_blend_test.sum()}")
print(f"Celdas distintas de P108: {diff_cells} ({diff_cells / (len(test)*3) * 100:.2f}%)")
print("Listo.")
