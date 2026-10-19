import numpy as np
import pandas as pd
import sys
from pathlib import Path

RAIZ = Path(".").resolve()
sys.path.insert(0, str(RAIZ))
from src import data as data_module

train_full = data_module.read_train()
test_df = data_module.read_test()

particion = pd.read_csv(RAIZ / "data" / "processed" / "holdout_v2_ids.csv")
ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
train_dev = train_full[train_full["id"].astype(int).isin(ids_dev)].reset_index(drop=True)

oofs = np.load("scratch/p114_oofs.npz")
prob_can = oofs["can"]
y_dev = oofs["y"]

test_data = np.load("scratch/p114_test_probs.npz")
prob_test = test_data["prob"]

# In dev:
cond_dev = (prob_can[:, 1] >= 0.51) & (prob_can[:, 2] >= 0.40) & (prob_can[:, 2] < 0.46)
idx_dev = np.where(cond_dev)[0]
print(f"Dev rows with P(TR)>=0.51 and P(AG) in [0.40, 0.46): {len(idx_dev)}")
for i in idx_dev:
    print(f"ID {train_dev.iloc[i]['id']}: True y={y_dev[i]} | p_TR={prob_can[i, 1]:.3f}, p_AG={prob_can[i, 2]:.3f} | {train_dev.iloc[i]['text'][:80]}")

# In test:
cond_te = (prob_test[:, 1] >= 0.51) & (prob_test[:, 2] >= 0.40) & (prob_test[:, 2] < 0.46)
idx_te = np.where(cond_te)[0]
print(f"\nTest rows with P(TR)>=0.51 and P(AG) in [0.40, 0.46): {len(idx_te)}")
for i in idx_te:
    print(f"Test Row {i} (id {test_df.iloc[i]['id']}): p_TR={prob_test[i, 1]:.3f}, p_AG={prob_test[i, 2]:.3f} | {test_df.iloc[i]['text'][:80]}")

