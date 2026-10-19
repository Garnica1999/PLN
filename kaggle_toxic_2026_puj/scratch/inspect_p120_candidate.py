import numpy as np
import pandas as pd
from scipy.special import expit, logit
import sys
from pathlib import Path

RAIZ = Path(".").resolve()
sys.path.insert(0, str(RAIZ))
from src.evaluation import ETIQUETAS, metricas_multietiqueta
from src import data as data_module

test_df = data_module.read_test()
sub_p114 = pd.read_csv("submissions/toxic_submission_p114_tri_optimizer_smooth_blend_asc.csv")
pred_p114 = sub_p114[list(ETIQUETAS)].to_numpy(dtype=int)

test_data = np.load("scratch/p114_test_probs.npz")
prob_test = test_data["prob"]

def transform(p_in, beta_ag=0.10, tr_th=0.48, beta_tr=0.02, hs_th=0.44, gamma_hs=0.02):
    p = p_in.copy()
    if gamma_hs > 0:
        cond_iso = (p[:, 1] < 0.15) & (p[:, 2] < 0.35)
        z_hs = logit(np.clip(p[:, 0], 1e-6, 1 - 1e-6))
        z_hs[cond_iso] -= gamma_hs * (0.35 - p[cond_iso, 2])
        p[:, 0] = expit(z_hs)
    
    if beta_tr > 0:
        cond_hs = p[:, 0] >= hs_th
        z_tr = logit(np.clip(p[:, 1], 1e-6, 1 - 1e-6))
        z_tr[cond_hs] += beta_tr * (p[cond_hs, 0] - hs_th)
        p[:, 1] = expit(z_tr)
        
    if beta_ag > 0:
        cond_tr = p[:, 1] >= tr_th
        z_ag = logit(np.clip(p[:, 2], 1e-6, 1 - 1e-6))
        z_ag[cond_tr] += beta_ag * (p[cond_tr, 1] - tr_th)
        p[:, 2] = expit(z_ag)
        
    return p

p_test_mod = transform(prob_test)

pred_mod_raw = np.column_stack([(p_test_mod[:, j] >= [0.44, 0.51, 0.46][j]).astype(int) for j in range(3)])
pred_mod = pred_mod_raw.copy()
mask_asc = (pred_mod[:, 1] == 1) & (pred_mod[:, 0] == 0)
pred_mod[mask_asc, 0] = 1

diff_mask = pred_p114 != pred_mod
diff_cells = int(diff_mask.sum())
print(f"Total cell differences vs P114 on test: {diff_cells}")
print(f"P114 Test Positives: HS={pred_p114[:,0].sum()}, TR={pred_p114[:,1].sum()}, AG={pred_p114[:,2].sum()} | Total={pred_p114.sum()}")
print(f"P120 Test Positives: HS={pred_mod[:,0].sum()}, TR={pred_mod[:,1].sum()}, AG={pred_mod[:,2].sum()} | Total={pred_mod.sum()}")

rows_changed = np.where(diff_mask.any(axis=1))[0]
print(f"Total rows modified: {len(rows_changed)}")
for r in rows_changed:
    print(f"Row {r} (id {test_df.iloc[r]['id']}): P114={pred_p114[r]} -> P120={pred_mod[r]} | Text: {test_df.iloc[r]['text'][:80]}")

