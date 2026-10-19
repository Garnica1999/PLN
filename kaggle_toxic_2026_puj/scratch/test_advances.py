import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path("/home/carlos/Git/PLN/kaggle_toxic_2026_puj")
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import metricas_multietiqueta, ETIQUETAS

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

BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]

def construir_matrices_folds(pliegues_arr, bloques):
    folds_data = []
    for fold in sorted(set(pliegues_arr)):
        idx_tr = np.flatnonzero(pliegues_arr != fold)
        idx_va = np.flatnonzero(pliegues_arr == fold)
        mats_tr, mats_va = [], []
        for analyzer, ngram, lower in bloques:
            v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=True, lowercase=lower)
            mats_tr.append(v.fit_transform(train_dev["text"].iloc[idx_tr]))
            mats_va.append(v.transform(train_dev["text"].iloc[idx_va]))
        folds_data.append((idx_tr, idx_va, sparse.hstack(mats_tr).tocsr(), sparse.hstack(mats_va).tocsr()))
    return folds_data

print("Precomputando matrices TF-IDF para Canonica y Particion 47...")
sys.stdout.flush()
folds_can = construir_matrices_folds(pliegues_can, BLOQUES_V2_TRIPLE)
folds_47 = construir_matrices_folds(pliegues_47, BLOQUES_V2_TRIPLE)
print("Matrices precomputadas con exito.")
sys.stdout.flush()

def compute_oof_probs(folds_data, c_dict, solver="liblinear", penalty="l2"):
    proba = np.zeros((len(train_dev), len(ETIQUETAS)), dtype=float)
    for idx_tr, idx_va, m_tr, m_va in folds_data:
        for j, etiqueta in enumerate(ETIQUETAS):
            lr = LogisticRegression(C=c_dict[etiqueta], class_weight="balanced", max_iter=1000, solver=solver, penalty=penalty, random_state=42)
            lr.fit(m_tr, train_dev[etiqueta].iloc[idx_tr])
            proba[idx_va, j] = lr.predict_proba(m_va)[:, 1]
    return proba

def eval_thresholds(proba, th_dict):
    pred = np.column_stack([(proba[:, j] >= th_dict[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    pred_asc, _ = aplicar_regla_ascendente(pred)
    return metricas_multietiqueta(y_true, pred_asc)["f1_media_columnas"]

# 1. Base C=2.0
base_c = {"HS": 2.0, "TR": 2.0, "AG": 2.0}
th_p108 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
th_robusto = {"HS": 0.46, "TR": 0.46, "AG": 0.42}

prob_can_base = compute_oof_probs(folds_can, base_c)
prob_47_base = compute_oof_probs(folds_47, base_c)

f1_can_p108 = eval_thresholds(prob_can_base, th_p108)
f1_47_p108 = eval_thresholds(prob_47_base, th_p108)

f1_can_rob = eval_thresholds(prob_can_base, th_robusto)
f1_47_rob = eval_thresholds(prob_47_base, th_robusto)

print(f"\n--- BASE C=2.0 ---")
print(f"P108 (HS 0.44/TR 0.51/AG 0.46): Canonica={f1_can_p108:.6f} | Part47={f1_47_p108:.6f} | Min={min(f1_can_p108, f1_47_p108):.6f}")
print(f"Robusto (HS 0.46/TR 0.46/AG 0.42): Canonica={f1_can_rob:.6f} | Part47={f1_47_rob:.6f} | Min={min(f1_can_rob, f1_47_rob):.6f} (Delta={min(f1_can_rob, f1_47_rob) - min(f1_can_p108, f1_47_p108):+.6f})")

# 2. Barrido fino de C por etiqueta
print(f"\n--- BARRIDO DE C POR ETIQUETA ---")
c_values = [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
for etiqueta in ETIQUETAS:
    print(f"\nVariando C para {etiqueta} (manteniendo C=2.0 en las otras):")
    for c in c_values:
        cur_c = dict(base_c)
        cur_c[etiqueta] = c
        # computar solo para esa etiqueta
        # Para optimizar, solo reentrenamos la columna de esa etiqueta
        j = ETIQUETAS.index(etiqueta)
        p_can = prob_can_base.copy()
        p_47 = prob_47_base.copy()
        for idx_tr, idx_va, m_tr, m_va in folds_can:
            lr = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr.fit(m_tr, train_dev[etiqueta].iloc[idx_tr])
            p_can[idx_va, j] = lr.predict_proba(m_va)[:, 1]
        for idx_tr, idx_va, m_tr, m_va in folds_47:
            lr = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr.fit(m_tr, train_dev[etiqueta].iloc[idx_tr])
            p_47[idx_va, j] = lr.predict_proba(m_va)[:, 1]
        sc_can = eval_thresholds(p_can, th_robusto)
        sc_47 = eval_thresholds(p_47, th_robusto)
        sc_min = min(sc_can, sc_47)
        delta = sc_min - min(f1_can_rob, f1_47_rob)
        print(f"  {etiqueta} C={c:.1f}: Canonica={sc_can:.6f} | Part47={sc_47:.6f} | Min={sc_min:.6f} ({delta:+.6f})")
    sys.stdout.flush()

# 3. Grid search conjunto de umbrales optimos
print(f"\n--- GRID SEARCH FINO DE UMBRALES (resolucion 0.01) ---")
best_th = None
best_min = min(f1_can_rob, f1_47_rob)
for hs_th in np.arange(0.42, 0.49, 0.01):
    for tr_th in np.arange(0.42, 0.49, 0.01):
        for ag_th in np.arange(0.38, 0.46, 0.01):
            th_c = {"HS": round(hs_th, 2), "TR": round(tr_th, 2), "AG": round(ag_th, 2)}
            s_can = eval_thresholds(prob_can_base, th_c)
            s_47 = eval_thresholds(prob_47_base, th_c)
            s_min = min(s_can, s_47)
            if s_min > best_min:
                best_min = s_min
                best_th = (th_c, s_can, s_47, s_min)

if best_th:
    print(f"Mejor configuracion de umbrales: {best_th[0]}")
    print(f"Canonica: {best_th[1]:.6f} | Part47: {best_th[2]:.6f} | Min: {best_th[3]:.6f} | Delta vs P108: {best_th[3] - min(f1_can_p108, f1_47_p108):+.6f}")
else:
    print(f"El umbral robusto {th_robusto} ya es optimo en la malla!")

