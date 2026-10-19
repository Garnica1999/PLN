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

folds_can = construir_matrices_folds(pliegues_can, BLOQUES_V2_TRIPLE)
folds_47 = construir_matrices_folds(pliegues_47, BLOQUES_V2_TRIPLE)

def compute_oof_probs(folds_data, c_dict):
    proba = np.zeros((len(train_dev), len(ETIQUETAS)), dtype=float)
    for idx_tr, idx_va, m_tr, m_va in folds_data:
        for j, etiqueta in enumerate(ETIQUETAS):
            lr = LogisticRegression(C=c_dict[etiqueta], class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr.fit(m_tr, train_dev[etiqueta].iloc[idx_tr])
            proba[idx_va, j] = lr.predict_proba(m_va)[:, 1]
    return proba

def f1_media_fast(y_t, y_p):
    tp = np.count_nonzero(y_t & y_p, axis=0)
    total = y_t.sum(axis=0) + y_p.sum(axis=0)
    f1 = np.where(total == 0, 0.0, 2.0 * tp / total)
    return float(np.mean(f1))

def bootstrap_clusters_fast(y, pred_a, pred_b, grupos, repeticiones=2000, semilla=20260925):
    generador = np.random.default_rng(semilla)
    indices = {int(g): np.flatnonzero(grupos == g) for g in np.unique(grupos)}
    claves = np.array(sorted(indices))
    dif = np.empty(repeticiones)
    for r in range(repeticiones):
        muestra = generador.integers(0, len(claves), len(claves))
        idx = np.concatenate([indices[int(claves[i])] for i in muestra])
        f1_a = f1_media_fast(y[idx], pred_a[idx])
        f1_b = f1_media_fast(y[idx], pred_b[idx])
        dif[r] = f1_a - f1_b
    c = np.quantile(dif, [0.025, 0.975])
    return {
        "ic95": [round(float(c[0]), 6), round(float(c[1]), 6)],
        "pct_mejora": round(float((dif > 0).mean()), 6),
    }

th_opt = {"HS": 0.45, "TR": 0.46, "AG": 0.43}
th_rob = {"HS": 0.46, "TR": 0.46, "AG": 0.42}

configs_c = [
    ("C_todos_2.0", {"HS": 2.0, "TR": 2.0, "AG": 2.0}),
    ("C_AG_2.5",    {"HS": 2.0, "TR": 2.0, "AG": 2.5}),
    ("C_TR3_AG2.5", {"HS": 2.0, "TR": 3.0, "AG": 2.5}),
    ("C_HS1_AG2.5", {"HS": 1.0, "TR": 2.0, "AG": 2.5}),
]

p108_probs_can = compute_oof_probs(folds_can, {"HS": 2.0, "TR": 2.0, "AG": 2.0})
p108_pred_can, _ = aplicar_regla_ascendente(np.column_stack([(p108_probs_can[:, j] >= {"HS": 0.44, "TR": 0.51, "AG": 0.46}[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
f1_p108_can = f1_media_fast(y_true, p108_pred_can)

p108_probs_47 = compute_oof_probs(folds_47, {"HS": 2.0, "TR": 2.0, "AG": 2.0})
p108_pred_47, _ = aplicar_regla_ascendente(np.column_stack([(p108_probs_47[:, j] >= {"HS": 0.44, "TR": 0.51, "AG": 0.46}[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
f1_p108_47 = f1_media_fast(y_true, p108_pred_47)

print(f"P108 Canonica: {f1_p108_can:.6f} | Part47: {f1_p108_47:.6f} | Min: {min(f1_p108_can, f1_p108_47):.6f}")
sys.stdout.flush()

for nombre, c_dict in configs_c:
    p_can = compute_oof_probs(folds_can, c_dict)
    p_47 = compute_oof_probs(folds_47, c_dict)
    
    for nombre_th, th in [("th_opt (0.45/0.46/0.43)", th_opt), ("th_rob (0.46/0.46/0.42)", th_rob)]:
        pred_can, _ = aplicar_regla_ascendente(np.column_stack([(p_can[:, j] >= th[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
        pred_47, _ = aplicar_regla_ascendente(np.column_stack([(p_47[:, j] >= th[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
        
        sc_can = f1_media_fast(y_true, pred_can)
        sc_47 = f1_media_fast(y_true, pred_47)
        boot_can = bootstrap_clusters_fast(y_true, pred_can, p108_pred_can, grupos)
        boot_47 = bootstrap_clusters_fast(y_true, pred_47, p108_pred_47, grupos)
        
        print(f"\n[{nombre} + {nombre_th}]")
        print(f"  Canonica: {sc_can:.6f} (Delta={sc_can - f1_p108_can:+.6f}) | IC95={boot_can['ic95']} | Win={boot_can['pct_mejora']*100:.1f}%")
        print(f"  Part47:   {sc_47:.6f} (Delta={sc_47 - f1_p108_47:+.6f}) | IC95={boot_47['ic95']} | Win={boot_47['pct_mejora']*100:.1f}%")
        print(f"  Minimo:   {min(sc_can, sc_47):.6f} (Delta Min={min(sc_can, sc_47) - min(f1_p108_can, f1_p108_47):+.6f})")
        sys.stdout.flush()
