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
prob_p47 = oofs["p47"]
y_dev = oofs["y"]

UMBRALES = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

def eval_partition(probs):
    pred = np.column_stack([(probs[:, j] >= UMBRALES[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    mask = (pred[:, 1] == 1) & (pred[:, 0] == 0)
    pred[mask, 0] = 1
    return pred

pred_can = eval_partition(prob_can)
pred_p47 = eval_partition(prob_p47)

# Analyze correlation between text features in train_dev and the error (HS_true == 0 & HS_pred == 1)
# Specifically in pattern (1,0,0)
mask_100_can = (pred_can[:, 0] == 1) & (pred_can[:, 1] == 0) & (pred_can[:, 2] == 0)
df_100 = train_dev[mask_100_can].copy()
df_100["y_HS"] = y_dev[mask_100_can, 0]
df_100["p_HS"] = prob_can[mask_100_can, 0]
df_100["p_TR"] = prob_can[mask_100_can, 1]
df_100["p_AG"] = prob_can[mask_100_can, 2]

print("Distribution of p_HS for true vs false positives in pattern (1,0,0):")
print("TP p_HS mean/std/min/max:", df_100[df_100["y_HS"] == 1]["p_HS"].describe())
print("FP p_HS mean/std/min/max:", df_100[df_100["y_HS"] == 0]["p_HS"].describe())

# Check threshold tuning on pattern (1,0,0) alone:
# What if isolated HS (when TR=0 and AG=0) requires a slightly higher threshold, e.g. 0.46, 0.47, 0.48?
for th_isolated in [0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50]:
    for name, p_arr, p_base in [("Canonica", prob_can, pred_can), ("Part47", prob_p47, pred_p47)]:
        p_mod = p_base.copy()
        # if raw pred was (1,0,0) and p_HS < th_isolated, change HS to 0
        cond = (p_arr[:, 0] >= 0.44) & (p_arr[:, 0] < th_isolated) & (p_arr[:, 1] < 0.51) & (p_arr[:, 2] < 0.46)
        p_mod[cond, 0] = 0
        # re-apply ascending rule
        mask = (p_mod[:, 1] == 1) & (p_mod[:, 0] == 0)
        p_mod[mask, 0] = 1
        
        f1 = metricas_multietiqueta(y_dev, p_mod)["f1_media_columnas"]
        f1_base = metricas_multietiqueta(y_dev, p_base)["f1_media_columnas"]
        delta = f1 - f1_base
        if name == "Canonica":
            print(f"th_isolated={th_isolated:.2f} | Canonica: {f1:.6f} ({delta:+.6f})", end=" | ")
        else:
            print(f"Part47: {f1:.6f} ({delta:+.6f})")

