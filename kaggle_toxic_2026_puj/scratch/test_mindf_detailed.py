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

def eval_mdf(mdf, bloques_base, c_dict, th_dict):
    def get_oof(pliegues_arr):
        proba = np.zeros((len(train_dev), len(ETIQUETAS)), dtype=float)
        for fold in sorted(set(pliegues_arr)):
            idx_tr = np.flatnonzero(pliegues_arr != fold)
            idx_va = np.flatnonzero(pliegues_arr == fold)
            mats_tr, mats_va = [], []
            for analyzer, ngram in bloques_base:
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
        pred_asc, _ = aplicar_regla_ascendente(pred)
        return f1_media_fast(y_true, pred_asc), proba, m_tr.shape[1]

    f1_can, prob_can, n_feat = get_oof(pliegues_can)
    f1_47, prob_47, _ = get_oof(pliegues_47)
    return f1_can, f1_47, n_feat

bloques_v2 = [
    ("word", (1, 2)),
    ("char_wb", (3, 5)),
    ("char", (1, 6)),
]

th_opt = {"HS": 0.45, "TR": 0.46, "AG": 0.43}
c_dict = {"HS": 2.0, "TR": 2.0, "AG": 2.5}

print("=== BARRIDO DETALLADO DE MIN_DF EN V2_TRIPLE (4 a 8) ===")
for mdf in [4, 5, 6, 7, 8]:
    c_can, c_47, n_f = eval_mdf(mdf, bloques_v2, c_dict, th_opt)
    print(f"min_df={mdf} ({n_f} features): Canonica={c_can:.6f} | Part47={c_47:.6f} | Min={min(c_can, c_47):.6f} | Media={(c_can+c_47)/2:.6f}")
    sys.stdout.flush()
