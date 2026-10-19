import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(".").resolve()
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

train_full = data_module.read_train()
particion = pd.read_csv(RAIZ / "data" / "processed" / "holdout_v2_ids.csv")
ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
train_dev = train_full[train_full["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
y_dev = train_dev[list(ETIQUETAS)].to_numpy(dtype=int)

zf = np.load(RAIZ / "features" / "oof_v03_final.npz", allow_pickle=True)
zb = np.load(RAIZ / "features" / "v2_oof_base_corrected.npz", allow_pickle=True)
pliegues_can = zf["pliegues"].astype(int)
grupos = zb["grupos"].astype(np.int64)

# Particion 47
combinacion = np.array([f"{a}{b}{c}" for a, b, c in y_dev])
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
pliegues_47 = np.full(len(train_dev), -1, dtype=int)
for num, (_, val) in enumerate(sgkf.split(np.arange(len(train_dev)).reshape(-1, 1), combinacion, groups=grupos)):
    pliegues_47[val] = num + 1

BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]
UMBRALES_CONGELADOS = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

# Load baseline P114 oofs
oofs = np.load("scratch/p114_oofs.npz")
prob_p114_can = oofs["can"]
prob_p114_p47 = oofs["p47"]

def eval_f1(probs, name=""):
    pred = np.column_stack([(probs[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    mask = (pred[:, 1] == 1) & (pred[:, 0] == 0)
    pred[mask, 0] = 1
    m = metricas_multietiqueta(y_dev, pred)
    return m["f1_media_columnas"]

f1_can_p114 = eval_f1(prob_p114_can)
f1_p47_p114 = eval_f1(prob_p114_p47)
print(f"P114 Baseline: Canonica={f1_can_p114:.6f}, Part47={f1_p47_p114:.6f}, Media={(f1_can_p114+f1_p47_p114)/2:.6f}")

# Let's test:
# 1. Logistic Regression with solver='lbfgs' (quasi-Newton) vs liblinear vs saga
# 2. Logistic Regression with different C per label:
# In P114: C_HS=2.0, C_TR=1.8, C_AG=1.5
# What if C_HS=1.8 or 2.2? What if C_TR=1.6 or 2.0? What if C_AG=1.4 or 1.6?
# 3. Blending L-BFGS or Newton-CG into the tri-optimizer blend (creating Quad-Optimizer):
# Let's run a test on Canonica and Part47
def evaluate_solver(solver, penalty="l2", l1_ratio=None, C_dict=None):
    if C_dict is None:
        C_dict = {"HS": 2.0, "TR": 1.8, "AG": 1.5}
    
    results = {}
    for part_name, pliegues in [("Canonica", pliegues_can), ("Part47", pliegues_47)]:
        n_samples = len(train_dev)
        prob = np.zeros((n_samples, 3), dtype=float)
        for fold in sorted(set(pliegues)):
            idx_tr = np.flatnonzero(pliegues != fold)
            idx_va = np.flatnonzero(pliegues == fold)
            
            mats_tr, mats_va = [], []
            for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
                v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
                mats_tr.append(v.fit_transform(train_dev["text"].iloc[idx_tr]))
                mats_va.append(v.transform(train_dev["text"].iloc[idx_va]))
            X_tr = sparse.hstack(mats_tr).tocsr()
            X_va = sparse.hstack(mats_va).tocsr()
            
            for j, et in enumerate(ETIQUETAS):
                kwargs = {
                    "C": C_dict[et],
                    "class_weight": "balanced",
                    "max_iter": 1000,
                    "solver": solver,
                    "random_state": 42
                }
                if l1_ratio is not None:
                    kwargs["l1_ratio"] = l1_ratio
                if penalty is not None:
                    kwargs["penalty"] = penalty
                clf = LogisticRegression(**kwargs)
                clf.fit(X_tr, train_dev[et].iloc[idx_tr])
                prob[idx_va, j] = clf.predict_proba(X_va)[:, 1]
        results[part_name] = prob
    return results

print("Testing L-BFGS solver...")
res_lbfgs = evaluate_solver("lbfgs")
f1_can_lbfgs = eval_f1(res_lbfgs["Canonica"])
f1_p47_lbfgs = eval_f1(res_lbfgs["Part47"])
print(f"L-BFGS Standalone: Can={f1_can_lbfgs:.6f}, P47={f1_p47_lbfgs:.6f}, Med={(f1_can_lbfgs+f1_p47_lbfgs)/2:.6f}")

# What if we blend L-BFGS into P114?
for w_lbfgs in [0.05, 0.10, 0.15, 0.20]:
    p_blend_can = (1 - w_lbfgs) * prob_p114_can + w_lbfgs * res_lbfgs["Canonica"]
    p_blend_p47 = (1 - w_lbfgs) * prob_p114_p47 + w_lbfgs * res_lbfgs["Part47"]
    f1_can = eval_f1(p_blend_can)
    f1_p47 = eval_f1(p_blend_p47)
    med = (f1_can + f1_p47) / 2
    d_can = f1_can - f1_can_p114
    d_p47 = f1_p47 - f1_p47_p114
    print(f"P114 + {int(w_lbfgs*100)}% L-BFGS: Can={f1_can:.6f} ({d_can:+.6f}) | P47={f1_p47:.6f} ({d_p47:+.6f}) | Med={med:.6f}")

