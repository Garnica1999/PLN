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
pred_v03 = zf["pred"].astype(int)
grupos = zb["grupos"].astype(np.int64)

train = data_module.read_train()
particion = pd.read_csv(IDS_PATH)
ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
train_dev = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)

# 1. Setup Partition 47
combinacion = np.array([f"{a}{b}{c}" for a, b, c in y_true])
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
pliegues_47 = np.full(len(y_true), -1, dtype=int)
for num, (_, val) in enumerate(sgkf.split(np.arange(len(y_true)).reshape(-1, 1), combinacion, groups=grupos)):
    pliegues_47[val] = num + 1

BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]

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

def aplicar_regla_ascendente(pred):
    p = pred.copy()
    mask = (p[:, 1] == 1) & (p[:, 0] == 0)
    p[mask, 0] = 1
    return p, int(mask.sum())

def get_oof_proba(bloques, pliegues_arr, c_dict={"HS": 2.0, "TR": 2.0, "AG": 2.0}):
    proba = np.zeros((len(train_dev), len(ETIQUETAS)), dtype=float)
    for fold in sorted(set(pliegues_arr)):
        idx_tr = np.flatnonzero(pliegues_arr != fold)
        idx_va = np.flatnonzero(pliegues_arr == fold)
        m_tr, m_va = construir_matrices(
            train_dev["text"].iloc[idx_tr],
            train_dev["text"].iloc[idx_va],
            bloques,
        )
        for j, etiqueta in enumerate(ETIQUETAS):
            lr = LogisticRegression(C=c_dict[etiqueta], class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr.fit(m_tr, train_dev[etiqueta].iloc[idx_tr])
            proba[idx_va, j] = lr.predict_proba(m_va)[:, 1]
    return proba

print("1. Calculando OOF de V2_triple en Canonica y Part47...")
proba_can = get_oof_proba(BLOQUES_V2_TRIPLE, pliegues_can)
proba_47 = get_oof_proba(BLOQUES_V2_TRIPLE, pliegues_47)

# Evaluar P108 baseline en ambas
def eval_pred(proba, th_dict):
    p = np.column_stack([(proba[:, j] >= th_dict[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    p_asc, _ = aplicar_regla_ascendente(p)
    return metricas_multietiqueta(y_true, p_asc)["f1_media_columnas"]

th_v03 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
f1_can_p108 = eval_pred(proba_can, th_v03)
f1_47_p108 = eval_pred(proba_47, th_v03)

print(f"P108 Canonica: {f1_can_p108:.6f} | Part47: {f1_47_p108:.6f}")

# Probar los umbrales encontrados {'HS': 0.42, 'TR': 0.46, 'AG': 0.46}
th_cand1 = {"HS": 0.42, "TR": 0.46, "AG": 0.46}
f1_can_cand1 = eval_pred(proba_can, th_cand1)
f1_47_cand1 = eval_pred(proba_47, th_cand1)
print(f"Candidato 1 {th_cand1}:")
print(f"  Canonica: {f1_can_cand1:.6f} (Delta: {f1_can_cand1 - f1_can_p108:+.6f})")
print(f"  Part47:   {f1_47_cand1:.6f} (Delta: {f1_47_cand1 - f1_47_p108:+.6f})")

# Barrido conjunto en Particion 47 para ver si los umbrales optimos coinciden
print("\nBúsqueda de umbrales óptimos independientes en Particion 47...")
mejor_f1_47 = f1_47_p108
mejores_th_47 = th_v03.copy()
for th_hs in np.arange(0.38, 0.52, 0.02):
    for th_tr in np.arange(0.44, 0.58, 0.02):
        for th_ag in np.arange(0.40, 0.54, 0.02):
            th = {"HS": round(th_hs, 2), "TR": round(th_tr, 2), "AG": round(th_ag, 2)}
            f1 = eval_pred(proba_47, th)
            if f1 > mejor_f1_47:
                mejor_f1_47 = f1
                mejores_th_47 = th

print(f"Mejores umbrales OOF Part47: {mejores_th_47} -> F1: {mejor_f1_47:.6f} (Delta: {mejor_f1_47 - f1_47_p108:+.6f})")

# Que pasa con umbrales intermedios / robustos que mejoren AMBAS particiones?
print("\nBuscando umbrales robustos que maximicen min(F1_can, F1_47):")
mejor_min_f1 = min(f1_can_p108, f1_47_p108)
mejor_th_robusto = th_v03.copy()
for th_hs in np.arange(0.40, 0.48, 0.02):
    for th_tr in np.arange(0.46, 0.54, 0.02):
        for th_ag in np.arange(0.42, 0.50, 0.02):
            th = {"HS": round(th_hs, 2), "TR": round(th_tr, 2), "AG": round(th_ag, 2)}
            f1_c = eval_pred(proba_can, th)
            f1_4 = eval_pred(proba_47, th)
            m = min(f1_c, f1_4)
            if m > mejor_min_f1:
                mejor_min_f1 = m
                mejor_th_robusto = (th, round(f1_c, 6), round(f1_4, 6))

print(f"Mejor umbral robusto min(Can, 47): {mejor_th_robusto}")
