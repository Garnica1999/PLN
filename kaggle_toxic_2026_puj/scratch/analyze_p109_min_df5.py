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

# P108 baseline OOF
BLOQUES_P108 = [
    ("word", (1, 2), 2),
    ("char_wb", (3, 5), 2),
    ("char", (1, 6), 2),
]

def run_oof(bloques, c_dict, pliegues_arr, th_dict):
    proba = np.zeros((len(train_dev), len(ETIQUETAS)), dtype=float)
    for fold in sorted(set(pliegues_arr)):
        idx_tr = np.flatnonzero(pliegues_arr != fold)
        idx_va = np.flatnonzero(pliegues_arr == fold)
        mats_tr, mats_va = [], []
        for analyzer, ngram, mdf in bloques:
            v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=mdf, sublinear_tf=True, lowercase=True)
            mats_tr.append(v.fit_transform(train_dev["text"].iloc[idx_tr]))
            mats_va.append(v.transform(train_dev["text"].iloc[idx_va]))
        m_tr = sparse.hstack(mats_tr).tocsr()
        m_va = sparse.hstack(mats_va).tocsr()
        for j, etiqueta in enumerate(ETIQUETAS):
            lr = LogisticRegression(C=c_dict[etiqueta], class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr.fit(m_tr, train_dev[etiqueta].iloc[idx_tr])
            proba[idx_va, j] = lr.predict_proba(m_va)[:, 1]
    pred = np.column_stack([(proba[:, j] >= th_dict[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    pred_asc, n_asc = aplicar_regla_ascendente(pred)
    f1 = metricas_multietiqueta(y_true, pred_asc)
    return f1, proba, pred_asc, n_asc

th_p108 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
c_p108 = {"HS": 2.0, "TR": 2.0, "AG": 2.0}

print("Calculando OOF P108...")
f1_p108_can, _, pred_p108_can, _ = run_oof(BLOQUES_P108, c_p108, pliegues_can, th_p108)
f1_p108_47, _, pred_p108_47, _ = run_oof(BLOQUES_P108, c_p108, pliegues_47, th_p108)

print(f"P108 Canonica: {f1_p108_can['f1_media_columnas']:.6f} (HS={f1_p108_can['f1_por_etiqueta']['HS']:.4f}, TR={f1_p108_can['f1_por_etiqueta']['TR']:.4f}, AG={f1_p108_can['f1_por_etiqueta']['AG']:.4f})")
print(f"P108 Part47:   {f1_p108_47['f1_media_columnas']:.6f}")

# Candidato P109: V2_triple con min_df=5, C={HS:2.0, TR:2.0, AG:2.5}, th={HS:0.45, TR:0.46, AG:0.43}
BLOQUES_P109_MDF5 = [
    ("word", (1, 2), 5),
    ("char_wb", (3, 5), 5),
    ("char", (1, 6), 5),
]
th_p109 = {"HS": 0.45, "TR": 0.46, "AG": 0.43}
c_p109 = {"HS": 2.0, "TR": 2.0, "AG": 2.5}

print("\nCalculando OOF P109 (min_df=5)...")
f1_p109_can, _, pred_p109_can, n_asc_can = run_oof(BLOQUES_P109_MDF5, c_p109, pliegues_can, th_p109)
f1_p109_47, _, pred_p109_47, n_asc_47 = run_oof(BLOQUES_P109_MDF5, c_p109, pliegues_47, th_p109)

boot_can = bootstrap_clusters_fast(y_true, pred_p109_can, pred_p108_can, grupos)
boot_47 = bootstrap_clusters_fast(y_true, pred_p109_47, pred_p108_47, grupos)

print(f"P109 Canonica: {f1_p109_can['f1_media_columnas']:.6f} | Delta: {f1_p109_can['f1_media_columnas'] - f1_p108_can['f1_media_columnas']:+.6f} | IC95: {boot_can['ic95']} | Win: {boot_can['pct_mejora']*100:.1f}%")
print(f"  Detalle por columna: HS={f1_p109_can['f1_por_etiqueta']['HS']:.4f}, TR={f1_p109_can['f1_por_etiqueta']['TR']:.4f}, AG={f1_p109_can['f1_por_etiqueta']['AG']:.4f}")
print(f"P109 Part47:   {f1_p109_47['f1_media_columnas']:.6f} | Delta: {f1_p109_47['f1_media_columnas'] - f1_p108_47['f1_media_columnas']:+.6f} | IC95: {boot_47['ic95']} | Win: {boot_47['pct_mejora']*100:.1f}%")
print(f"  Detalle por columna: HS={f1_p109_47['f1_por_etiqueta']['HS']:.4f}, TR={f1_p109_47['f1_por_etiqueta']['TR']:.4f}, AG={f1_p109_47['f1_por_etiqueta']['AG']:.4f}")

# Evaluacion por fold
folds_can = sorted(set(pliegues_can))
fc_108 = [f1_media_fast(y_true[pliegues_can==f], pred_p108_can[pliegues_can==f]) for f in folds_can]
fc_109 = [f1_media_fast(y_true[pliegues_can==f], pred_p109_can[pliegues_can==f]) for f in folds_can]
print(f"Folds Canonica P108: {[round(x, 4) for x in fc_108]}")
print(f"Folds Canonica P109: {[round(x, 4) for x in fc_109]}")
print(f"Folds ganados Canonica: {sum(1 for a,b in zip(fc_109, fc_108) if a > b)}/5")

# Test set simulation
test = data_module.read_test()
mats_tr, mats_te = [], []
for analyzer, ngram, mdf in BLOQUES_P109_MDF5:
    v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=mdf, sublinear_tf=True, lowercase=True)
    mats_tr.append(v.fit_transform(train["text"]))
    mats_te.append(v.transform(test["text"]))

X_tr = sparse.hstack(mats_tr).tocsr()
X_te = sparse.hstack(mats_te).tocsr()
print(f"\nMatriz de features sobre train completo: {X_tr.shape[1]} features (V03 tenia ~27k, P108 tenia ~86k)")

proba_te = np.zeros((len(test), len(ETIQUETAS)), dtype=float)
for j, e in enumerate(ETIQUETAS):
    lr = LogisticRegression(C=c_p109[e], class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
    lr.fit(X_tr, train[e])
    proba_te[:, j] = lr.predict_proba(X_te)[:, 1]

raw_pred_te = np.column_stack([(proba_te[:, j] >= th_p109[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
asc_pred_te, n_asc_te = aplicar_regla_ascendente(raw_pred_te)

sub_p108 = pd.read_csv(RAIZ / "submissions" / "toxic_submission_p108_v2_triple_asc.csv")
p108_arr = sub_p108[list(ETIQUETAS)].to_numpy()

diff_te = np.sum(asc_pred_te != p108_arr)
diff_cols = {e: int(np.sum(asc_pred_te[:, j] != p108_arr[:, j])) for j, e in enumerate(ETIQUETAS)}
pos_109 = {e: int(asc_pred_te[:, j].sum()) for j, e in enumerate(ETIQUETAS)}
pos_108 = {e: int(p108_arr[:, j].sum()) for j, e in enumerate(ETIQUETAS)}

print(f"\nSimulacion sobre Test (1153 filas):")
print(f"  Celdas distintas de P108: {diff_te} de {p108_arr.size} ({diff_te/p108_arr.size*100:.2f}%)")
print(f"  Por columna: {diff_cols}")
print(f"  Positivos P108: {pos_108}")
print(f"  Positivos P109: {pos_109}")
print(f"  Regla ascendente rescato: {n_asc_te} celdas en test")
