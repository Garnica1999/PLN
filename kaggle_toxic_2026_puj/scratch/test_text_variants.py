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

def eval_variant(nombre, config_bloques, c_dict, th_dict):
    def get_oof(pliegues_arr):
        proba = np.zeros((len(train_dev), len(ETIQUETAS)), dtype=float)
        for fold in sorted(set(pliegues_arr)):
            idx_tr = np.flatnonzero(pliegues_arr != fold)
            idx_va = np.flatnonzero(pliegues_arr == fold)
            mats_tr, mats_va = [], []
            for analyzer, ngram, lower, sublinear, min_df, strip in config_bloques:
                v = TfidfVectorizer(
                    analyzer=analyzer,
                    ngram_range=ngram,
                    min_df=min_df,
                    sublinear_tf=sublinear,
                    lowercase=lower,
                    strip_accents=strip,
                )
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
        return metricas_multietiqueta(y_true, pred_asc)["f1_media_columnas"], proba

    f1_can, prob_can = get_oof(pliegues_can)
    f1_47, prob_47 = get_oof(pliegues_47)
    min_sc = min(f1_can, f1_47)
    print(f"[{nombre}] Canonica={f1_can:.6f} | Part47={f1_47:.6f} | Min={min_sc:.6f}")
    sys.stdout.flush()
    return f1_can, f1_47

th_opt = {"HS": 0.45, "TR": 0.46, "AG": 0.43}
c_base = {"HS": 2.0, "TR": 2.0, "AG": 2.5}

print("=== EVALUACION DE VARIANTES DE REPRESENTACION TEXTUAL ===")
sys.stdout.flush()

# 1. Base V2_triple
v2_base = [
    ("word", (1, 2), True, True, 2, None),
    ("char_wb", (3, 5), True, True, 2, None),
    ("char", (1, 6), True, True, 2, None),
]
eval_variant("V2_triple_base", v2_base, c_base, th_opt)

# 2. Cased char(1,6): uppercase preserved in raw char ngrams
v2_cased_char = [
    ("word", (1, 2), True, True, 2, None),
    ("char_wb", (3, 5), True, True, 2, None),
    ("char", (1, 6), False, True, 2, None),
]
eval_variant("V2_cased_char", v2_cased_char, c_base, th_opt)

# 3. Strip accents unicode
v2_strip_accents = [
    ("word", (1, 2), True, True, 2, "unicode"),
    ("char_wb", (3, 5), True, True, 2, "unicode"),
    ("char", (1, 6), True, True, 2, "unicode"),
]
eval_variant("V2_strip_accents", v2_strip_accents, c_base, th_opt)

# 4. min_df=3 to reduce sparse noise
v2_mindf3 = [
    ("word", (1, 2), True, True, 3, None),
    ("char_wb", (3, 5), True, True, 3, None),
    ("char", (1, 6), True, True, 3, None),
]
eval_variant("V2_mindf3", v2_mindf3, c_base, th_opt)
