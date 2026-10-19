import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path
import sys

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

def construir_matrices(textos_tr, textos_va, bloques):
    mats_tr, mats_va = [], []
    for analyzer, ngram, lower in bloques:
        v = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram,
            min_df=2,
            sublinear_tf=True,
            lowercase=lower,
        )
        mats_tr.append(v.fit_transform(textos_tr))
        mats_va.append(v.transform(textos_va))
    return sparse.hstack(mats_tr).tocsr(), sparse.hstack(mats_va).tocsr()

def bootstrap_clusters_delta(y, pred_a, pred_b, grupos, repeticiones=2000, semilla=20260925):
    generador = np.random.default_rng(semilla)
    indices = {int(g): np.flatnonzero(grupos == g) for g in np.unique(grupos)}
    claves = np.array(sorted(indices))
    dif = np.empty(repeticiones)
    for r in range(repeticiones):
        muestra = generador.integers(0, len(claves), len(claves))
        idx = np.concatenate([indices[int(claves[i])] for i in muestra])
        f1_a = metricas_multietiqueta(y[idx], pred_a[idx])["f1_media_columnas"]
        f1_b = metricas_multietiqueta(y[idx], pred_b[idx])["f1_media_columnas"]
        dif[r] = f1_a - f1_b
    c = np.quantile(dif, [0.025, 0.975])
    return {
        "ic95": [round(float(c[0]), 6), round(float(c[1]), 6)],
        "pct_mejora": round(float((dif > 0).mean()), 6),
    }

print("=== 1. ANALISIS DE UMBRALES ROBUSTOS P108-B ===")
# Bloques V2_triple
BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]

def run_oof_lr(bloques, pliegues_arr, c_dict={"HS": 2.0, "TR": 2.0, "AG": 2.0}):
    proba = np.zeros((len(train_dev), len(ETIQUETAS)), dtype=float)
    for fold in sorted(set(pliegues_arr)):
        idx_tr = np.flatnonzero(pliegues_arr != fold)
        idx_va = np.flatnonzero(pliegues_arr == fold)
        m_tr, m_va = construir_matrices(train_dev["text"].iloc[idx_tr], train_dev["text"].iloc[idx_va], bloques)
        for j, etiqueta in enumerate(ETIQUETAS):
            lr = LogisticRegression(C=c_dict[etiqueta], class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr.fit(m_tr, train_dev[etiqueta].iloc[idx_tr])
            proba[idx_va, j] = lr.predict_proba(m_va)[:, 1]
    return proba

proba_can = run_oof_lr(BLOQUES_V2_TRIPLE, pliegues_can)
proba_47 = run_oof_lr(BLOQUES_V2_TRIPLE, pliegues_47)

th_p108 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
th_robusto = {"HS": 0.46, "TR": 0.46, "AG": 0.42}

# Predicciones P108 base
pred_p108_can, _ = aplicar_regla_ascendente(np.column_stack([(proba_can[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
pred_p108_47, _ = aplicar_regla_ascendente(np.column_stack([(proba_47[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))

f1_p108_can = metricas_multietiqueta(y_true, pred_p108_can)["f1_media_columnas"]
f1_p108_47 = metricas_multietiqueta(y_true, pred_p108_47)["f1_media_columnas"]

# Predicciones con umbrales robustos
pred_rob_can, n_rob_can = aplicar_regla_ascendente(np.column_stack([(proba_can[:, j] >= th_robusto[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
pred_rob_47, n_rob_47 = aplicar_regla_ascendente(np.column_stack([(proba_47[:, j] >= th_robusto[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))

f1_rob_can = metricas_multietiqueta(y_true, pred_rob_can)["f1_media_columnas"]
f1_rob_47 = metricas_multietiqueta(y_true, pred_rob_47)["f1_media_columnas"]

# Folds breakdown
folds_can = sorted(set(pliegues_can))
f_can_p108 = [metricas_multietiqueta(y_true[pliegues_can == f], pred_p108_can[pliegues_can == f])["f1_media_columnas"] for f in folds_can]
f_can_rob = [metricas_multietiqueta(y_true[pliegues_can == f], pred_rob_can[pliegues_can == f])["f1_media_columnas"] for f in folds_can]
ganados_can = sum(1 for a, b in zip(f_can_rob, f_can_p108) if a > b)

folds_47 = sorted(set(pliegues_47))
f_47_p108 = [metricas_multietiqueta(y_true[pliegues_47 == f], pred_p108_47[pliegues_47 == f])["f1_media_columnas"] for f in folds_47]
f_47_rob = [metricas_multietiqueta(y_true[pliegues_47 == f], pred_rob_47[pliegues_47 == f])["f1_media_columnas"] for f in folds_47]
ganados_47 = sum(1 for a, b in zip(f_47_rob, f_47_p108) if a > b)

boot_rob_can = bootstrap_clusters_delta(y_true, pred_rob_can, pred_p108_can, grupos)
boot_rob_47 = bootstrap_clusters_delta(y_true, pred_rob_47, pred_p108_47, grupos)

print(f"P108 Canonica: {f1_p108_can:.6f} | Umbrales Robustos: {f1_rob_can:.6f} | Delta: {f1_rob_can - f1_p108_can:+.6f} | Ganados: {ganados_can}/5 | IC95: {boot_rob_can['ic95']} | Pct: {boot_rob_can['pct_mejora']*100:.1f}%")
print(f"P108 Part47:   {f1_p108_47:.6f} | Umbrales Robustos: {f1_rob_47:.6f} | Delta: {f1_rob_47 - f1_p108_47:+.6f} | Ganados: {ganados_47}/5 | IC95: {boot_rob_47['ic95']} | Pct: {boot_rob_47['pct_mejora']*100:.1f}%")

# Celdas cambiadas vs P108
print(f"Celdas cambiadas vs P108 en dev: {np.sum(pred_rob_can != pred_p108_can)} de {pred_p108_can.size}")

# Test set simulation
test = data_module.read_test()
mats_tr, mats_te = [], []
for analyzer, ngram, lower in BLOQUES_V2_TRIPLE:
    v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=True, lowercase=lower)
    mats_tr.append(v.fit_transform(train["text"]))
    mats_te.append(v.transform(test["text"]))

X_tr = sparse.hstack(mats_tr).tocsr()
X_te = sparse.hstack(mats_te).tocsr()

proba_te = np.zeros((len(test), len(ETIQUETAS)), dtype=float)
for j, e in enumerate(ETIQUETAS):
    lr = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
    lr.fit(X_tr, train[e])
    proba_te[:, j] = lr.predict_proba(X_te)[:, 1]

# Prediccion con umbrales robustos
pred_te_rob = np.column_stack([(proba_te[:, j] >= th_robusto[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
pred_te_rob_asc, n_asc_te = aplicar_regla_ascendente(pred_te_rob)

sub_p108 = pd.read_csv(RAIZ / "submissions" / "toxic_submission_p108_v2_triple_asc.csv")
p108_arr = sub_p108[list(ETIQUETAS)].to_numpy()

diff_test = np.sum(pred_te_rob_asc != p108_arr)
diff_cols = {e: int(np.sum(pred_te_rob_asc[:, j] != p108_arr[:, j])) for j, e in enumerate(ETIQUETAS)}
pos_rob = {e: int(pred_te_rob_asc[:, j].sum()) for j, e in enumerate(ETIQUETAS)}
pos_p108 = {e: int(p108_arr[:, j].sum()) for j, e in enumerate(ETIQUETAS)}

print(f"\nSimulacion en Test:")
print(f"  Celdas distintas de P108: {diff_test} ({diff_test/p108_arr.size*100:.2f}%) | Por col: {diff_cols}")
print(f"  Positivos P108: {pos_p108}")
print(f"  Positivos Robustos: {pos_rob}")
print(f"  Violaciones TR=1 & HS=0 corregidas: {n_asc_te}")

# 2. Probar Word Trigramas word(1,3)
print("\n=== 2. PROBAR WORD(1,3) + CHAR_WB(3,5) + CHAR(1,6) ===")
BLOQUES_TRIGRAMAS = [
    ("word", (1, 3), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]
proba_tri_can = run_oof_lr(BLOQUES_TRIGRAMAS, pliegues_can)
pred_tri_can, _ = aplicar_regla_ascendente(np.column_stack([(proba_tri_can[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
f1_tri_can = metricas_multietiqueta(y_true, pred_tri_can)["f1_media_columnas"]
print(f"Word(1,3) Canonica F1 (umbrales p108): {f1_tri_can:.6f} (Delta vs P108: {f1_tri_can - f1_p108_can:+.6f})")

# 3. Probar Blend LogisticRegression + LinearSVC calibrado en V2_triple
print("\n=== 3. PROBAR BLEND LR + LINEARSVC EN V2_TRIPLE ===")
def run_oof_svc(bloques, pliegues_arr):
    proba = np.zeros((len(train_dev), len(ETIQUETAS)), dtype=float)
    for fold in sorted(set(pliegues_arr)):
        idx_tr = np.flatnonzero(pliegues_arr != fold)
        idx_va = np.flatnonzero(pliegues_arr == fold)
        m_tr, m_va = construir_matrices(train_dev["text"].iloc[idx_tr], train_dev["text"].iloc[idx_va], bloques)
        for j, etiqueta in enumerate(ETIQUETAS):
            svc = LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=2000)
            cal = CalibratedClassifierCV(estimator=svc, method="sigmoid", cv=3)
            cal.fit(m_tr, train_dev[etiqueta].iloc[idx_tr])
            proba[idx_va, j] = cal.predict_proba(m_va)[:, 1]
    return proba

proba_svc_can = run_oof_svc(BLOQUES_V2_TRIPLE, pliegues_can)
for w in [0.1, 0.2, 0.3]:
    p_blend = (1 - w) * proba_can + w * proba_svc_can
    p_b_pred, _ = aplicar_regla_ascendente(np.column_stack([(p_blend[:, j] >= th_p108[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
    f1_b = metricas_multietiqueta(y_true, p_b_pred)["f1_media_columnas"]
    print(f"Blend LR + {int(w*100)}% SVC Canonica F1: {f1_b:.6f} (Delta vs P108: {f1_b - f1_p108_can:+.6f})")
