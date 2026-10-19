import sys
from pathlib import Path
import numpy as np
import pandas as pd

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

# Analyze on Canonica
UMBRALES = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
pred_raw = np.column_stack([(prob_can[:, j] >= UMBRALES[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
pred_can = pred_raw.copy()
mask_asc = (pred_can[:, 1] == 1) & (pred_can[:, 0] == 0)
pred_can[mask_asc, 0] = 1

# Check patterns
patterns_true = [f"({r[0]},{r[1]},{r[2]})" for r in y_dev]
patterns_pred = [f"({r[0]},{r[1]},{r[2]})" for r in pred_can]

df_analysis = pd.DataFrame({
    "id": train_dev["id"],
    "text": train_dev["text"],
    "true_pattern": patterns_true,
    "pred_pattern": patterns_pred,
    "p_HS": prob_can[:, 0],
    "p_TR": prob_can[:, 1],
    "p_AG": prob_can[:, 2],
    "HS_true": y_dev[:, 0],
    "TR_true": y_dev[:, 1],
    "AG_true": y_dev[:, 2],
    "HS_pred": pred_can[:, 0],
    "TR_pred": pred_can[:, 1],
    "AG_pred": pred_can[:, 2],
})

# 1. Look at pattern (1,0,0) excess: true is (0,0,0), pred is (1,0,0)
fp_hs_only = df_analysis[(df_analysis["true_pattern"] == "(0,0,0)") & (df_analysis["pred_pattern"] == "(1,0,0)")]
print(f"Excess (1,0,0) FP count: {len(fp_hs_only)}")
print("Sample excess (1,0,0) false positives:")
for _, row in fp_hs_only.head(10).iterrows():
    print(f"ID {row['id']}: p_HS={row['p_HS']:.3f}, p_TR={row['p_TR']:.3f}, p_AG={row['p_AG']:.3f} | {row['text'][:80]}")

# 2. Look at TR-AG conditional relationship:
# In ground truth, when TR=1, AG=1 in 94.6% of cases.
# Where does P114 predict TR=1, but AG=0?
tr1_ag0 = df_analysis[(df_analysis["TR_pred"] == 1) & (df_analysis["AG_pred"] == 0)]
print(f"\nP114 predicts TR=1 and AG=0 in {len(tr1_ag0)} cases.")
print("Of these, what is true AG?")
print(tr1_ag0["AG_true"].value_counts())
print("Sample TR_pred=1, AG_pred=0:")
for _, row in tr1_ag0.iterrows():
    print(f"ID {row['id']}: True={row['true_pattern']} | p_TR={row['p_TR']:.3f}, p_AG={row['p_AG']:.3f} | {row['text'][:80]}")

