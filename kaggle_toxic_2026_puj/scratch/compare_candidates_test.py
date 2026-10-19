import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

RAIZ = Path("/home/carlos/Git/PLN/kaggle_toxic_2026_puj")
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS

train = data_module.read_train()
test = data_module.read_test()

sub_p108 = pd.read_csv(RAIZ / "submissions" / "toxic_submission_p108_v2_triple_asc.csv")
p108_arr = sub_p108[list(ETIQUETAS)].to_numpy()

BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]

mats_tr, mats_te = [], []
for analyzer, ngram, lower in BLOQUES_V2_TRIPLE:
    v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=True, lowercase=lower)
    mats_tr.append(v.fit_transform(train["text"]))
    mats_te.append(v.transform(test["text"]))

X_tr = sparse.hstack(mats_tr).tocsr()
X_te = sparse.hstack(mats_te).tocsr()

def aplicar_regla_ascendente(pred):
    p = pred.copy()
    mask = (p[:, 1] == 1) & (p[:, 0] == 0)
    p[mask, 0] = 1
    return p, int(mask.sum())

def generate_pred(c_dict, th_dict):
    proba_te = np.zeros((len(test), len(ETIQUETAS)), dtype=float)
    for j, e in enumerate(ETIQUETAS):
        lr = LogisticRegression(C=c_dict[e], class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        lr.fit(X_tr, train[e])
        proba_te[:, j] = lr.predict_proba(X_te)[:, 1]
    raw_pred = np.column_stack([(proba_te[:, j] >= th_dict[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    asc_pred, n_asc = aplicar_regla_ascendente(raw_pred)
    return asc_pred, n_asc, proba_te

candidates = [
    ("Cand_A_C2_th_opt", {"HS": 2.0, "TR": 2.0, "AG": 2.0}, {"HS": 0.45, "TR": 0.46, "AG": 0.43}),
    ("Cand_B_CHS1_AG2.5_th_rob", {"HS": 1.0, "TR": 2.0, "AG": 2.5}, {"HS": 0.46, "TR": 0.46, "AG": 0.42}),
    ("Cand_C_CTR3_AG2.5_th_rob", {"HS": 2.0, "TR": 3.0, "AG": 2.5}, {"HS": 0.46, "TR": 0.46, "AG": 0.42}),
    ("Cand_D_C2_th_rob", {"HS": 2.0, "TR": 2.0, "AG": 2.0}, {"HS": 0.46, "TR": 0.46, "AG": 0.42}),
]

print("=== COMPARACION DE CANDIDATOS SOBRE TEST (1153 filas, 3459 celdas) ===")
for name, c_dict, th_dict in candidates:
    pred, n_asc, prob = generate_pred(c_dict, th_dict)
    diff = np.sum(pred != p108_arr)
    diff_cols = {e: int(np.sum(pred[:, j] != p108_arr[:, j])) for j, e in enumerate(ETIQUETAS)}
    pos = {e: int(pred[:, j].sum()) for j, e in enumerate(ETIQUETAS)}
    print(f"\n{name}:")
    print(f"  Celdas distintas de P108: {diff} ({diff / p108_arr.size * 100:.2f}%)")
    print(f"  Por columna: {diff_cols}")
    print(f"  Positivos: {pos} (P108 tenia: HS={p108_arr[:,0].sum()}, TR={p108_arr[:,1].sum()}, AG={p108_arr[:,2].sum()})")
    print(f"  Regla ascendente TR=1 => HS=1 rescato: {n_asc} celdas")
