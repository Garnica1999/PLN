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

UMBRALES_V03 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

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

print("Calculando OOF probabilidades para V2_triple...")
proba_v2 = np.zeros((len(train_dev), len(ETIQUETAS)), dtype=float)
for fold in sorted(set(pliegues_can)):
    idx_tr = np.flatnonzero(pliegues_can != fold)
    idx_va = np.flatnonzero(pliegues_can == fold)
    m_tr, m_va = construir_matrices(
        train_dev["text"].iloc[idx_tr],
        train_dev["text"].iloc[idx_va],
        BLOQUES_V2_TRIPLE,
    )
    for j, etiqueta in enumerate(ETIQUETAS):
        lr = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        lr.fit(m_tr, train_dev[etiqueta].iloc[idx_tr])
        proba_v2[idx_va, j] = lr.predict_proba(m_va)[:, 1]

# Baseline P108
pred_p108 = np.column_stack([(proba_v2[:, j] >= UMBRALES_V03[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
pred_p108_asc, _ = aplicar_regla_ascendente(pred_p108)
f1_p108 = metricas_multietiqueta(y_true, pred_p108_asc)["f1_media_columnas"]
print(f"P108 Canonica F1: {f1_p108:.6f}")

# 1. Explorar optimizacion de umbrales para V2_triple
print("\n--- 1. Búsqueda de umbrales óptimos para V2_triple ---")
mejor_f1 = f1_p108
mejores_umbrales = UMBRALES_V03.copy()

hs_range = np.arange(0.38, 0.52, 0.02)
tr_range = np.arange(0.44, 0.58, 0.02)
ag_range = np.arange(0.40, 0.54, 0.02)

for th_hs in hs_range:
    for th_tr in tr_range:
        for th_ag in ag_range:
            p = np.column_stack([
                (proba_v2[:, 0] >= th_hs).astype(int),
                (proba_v2[:, 1] >= th_tr).astype(int),
                (proba_v2[:, 2] >= th_ag).astype(int),
            ])
            p_asc, _ = aplicar_regla_ascendente(p)
            f1 = metricas_multietiqueta(y_true, p_asc)["f1_media_columnas"]
            if f1 > mejor_f1:
                mejor_f1 = f1
                mejores_umbrales = {"HS": round(th_hs, 2), "TR": round(th_tr, 2), "AG": round(th_ag, 2)}

print(f"Mejores umbrales OOF: {mejores_umbrales} -> F1: {mejor_f1:.6f} (Delta: {mejor_f1 - f1_p108:+.6f})")

# 2. Explorar C por etiqueta en V2_triple
print("\n--- 2. Explorar C por etiqueta ---")
for c_hs in [1.5, 2.0, 2.5, 3.0]:
    for c_tr in [1.0, 1.5, 2.0, 2.5]:
        for c_ag in [1.5, 2.0, 2.5, 3.0]:
            if (c_hs, c_tr, c_ag) == (2.0, 2.0, 2.0):
                continue
            # Probamos en pliegues solo si c_hs o c_tr o c_ag cambian
            pass # Veremos en analisis detallado

# 3. Analizar relacion AG -> HS
print("\n--- 3. Analizar AG -> HS ---")
# Si AG=1 y HS=0 en prediccion, que pasa si aplicamos regla suave o dura?
pred_test_ag = pred_p108_asc.copy()
# Casos donde pred AG=1 pero pred HS=0
viol_ag = (pred_test_ag[:, 2] == 1) & (pred_test_ag[:, 0] == 0)
print(f"Casos en dev donde pred AG=1 y pred HS=0: {viol_ag.sum()}")
if viol_ag.sum() > 0:
    tp_rescatados = (y_true[viol_ag, 0] == 1).sum()
    fp_generados = (y_true[viol_ag, 0] == 0).sum()
    print(f"De esos {viol_ag.sum()}: Reales HS=1 (TP potenciales): {tp_rescatados} | Reales HS=0 (FP potenciales): {fp_generados}")
    
    # Que pasa si seteamos HS=1 cuando AG=1?
    pred_test_ag[viol_ag, 0] = 1
    f1_ag_hard = metricas_multietiqueta(y_true, pred_test_ag)["f1_media_columnas"]
    print(f"F1 si forzamos AG=1 => HS=1: {f1_ag_hard:.6f} (Delta: {f1_ag_hard - f1_p108:+.6f})")

    # Que pasa si seteamos HS=1 cuando AG=1 Y p(HS) >= 0.35 (soft)?
    for th_boost in [0.30, 0.35, 0.38, 0.40]:
        p_soft = pred_p108_asc.copy()
        mask_soft = (p_soft[:, 2] == 1) & (p_soft[:, 0] == 0) & (proba_v2[:, 0] >= th_boost)
        p_soft[mask_soft, 0] = 1
        f1_soft = metricas_multietiqueta(y_true, p_soft)["f1_media_columnas"]
        print(f"F1 soft AG=1 & p(HS)>={th_boost} => HS=1 (modifica {mask_soft.sum()} celdas): {f1_soft:.6f} (Delta: {f1_soft - f1_p108:+.6f})")
