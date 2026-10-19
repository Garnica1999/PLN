import sys
from pathlib import Path
import numpy as np
import pandas as pd
import re

RAIZ = Path(".").resolve()
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

train_full = data_module.read_train()
particion = pd.read_csv(RAIZ / "data" / "processed" / "holdout_v2_ids.csv")
ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
train_dev = train_full[train_full["id"].astype(int).isin(ids_dev)].reset_index(drop=True)

oofs = np.load("scratch/p114_oofs.npz")
prob_can = oofs["can"]
y_dev = oofs["y"]

UMBRALES = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
pred_raw = np.column_stack([(prob_can[:, j] >= UMBRALES[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
pred_can = pred_raw.copy()
mask_asc = (pred_can[:, 1] == 1) & (pred_can[:, 0] == 0)
pred_can[mask_asc, 0] = 1

base_f1 = metricas_multietiqueta(y_dev, pred_can)["f1_media_columnas"]
print(f"Base P114 Canonica F1: {base_f1:.6f}")

# Look at tweets with HS_pred == 1, but TR_pred == 0 and AG_pred == 0 (pattern 1,0,0)
idx_100 = np.where((pred_can[:, 0] == 1) & (pred_can[:, 1] == 0) & (pred_can[:, 2] == 0))[0]
print(f"Total predicted pattern (1,0,0): {len(idx_100)}")
y_100 = y_dev[idx_100]
tp_hs = (y_100[:, 0] == 1).sum()
fp_hs = (y_100[:, 0] == 0).sum()
print(f"Of these {len(idx_100)} predictions: TP={tp_hs}, FP={fp_hs} (Precision = {tp_hs/len(idx_100):.3f})")

# Check if there are identifiable patterns in training for these FPs:
# e.g., Presence of URL (news links), lack of direct insults, positive sentiment cues
for i in idx_100:
    row = train_dev.iloc[i]
    p0, p1, p2 = prob_can[i]
    is_fp = (y_dev[i, 0] == 0)
    has_url = "http" in row["text"]
    has_rt = "rt" in row["text"].lower().split()
    # print if informative
    if is_fp and has_url and p0 < 0.58:
        # print candidate
        pass

# Let's test a simple hypothesis:
# In pattern (1,0,0), if p_HS is near the threshold [0.44, 0.52] and p_TR is very low (< 0.15) and p_AG < 0.35:
# Is it predominantly FP?
for max_p0 in [0.48, 0.50, 0.52, 0.55]:
    for max_p1 in [0.10, 0.15, 0.20]:
        for max_p2 in [0.30, 0.35, 0.40]:
            cond = (pred_can[:, 0] == 1) & (pred_can[:, 1] == 0) & (pred_can[:, 2] == 0) & (prob_can[:, 0] <= max_p0) & (prob_can[:, 1] <= max_p1) & (prob_can[:, 2] <= max_p2)
            n_match = cond.sum()
            if n_match >= 5:
                n_fp = (y_dev[cond, 0] == 0).sum()
                n_tp = (y_dev[cond, 0] == 1).sum()
                prec = n_tp / n_match
                if prec < 0.25: # mostly FP!
                    print(f"p0<={max_p0}, p1<={max_p1}, p2<={max_p2}: match={n_match}, FP={n_fp}, TP={n_tp} (FP rate: {n_fp/n_match:.2f})")
