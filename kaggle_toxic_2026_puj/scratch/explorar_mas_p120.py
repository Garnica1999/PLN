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

oofs = np.load("scratch/p114_oofs.npz")
prob_can = oofs["can"]
prob_p47 = oofs["p47"]
y_dev = oofs["y"]

test_data = np.load("scratch/p114_test_probs.npz")
prob_test = test_data["prob"]

f1_can_base = metricas_multietiqueta(y_dev, pred_p114[:len(y_dev)]) # wait, baseline pred
def eval_p(p):
    pred = np.column_stack([(p[:, j] >= [0.44, 0.51, 0.46][j]).astype(int) for j in range(3)])
    mask = (pred[:, 1] == 1) & (pred[:, 0] == 0)
    pred[mask, 0] = 1
    return metricas_multietiqueta(y_dev, pred)["f1_media_columnas"]

f1_can_base = eval_p(prob_can)
f1_p47_base = eval_p(prob_p47)

print(f"P114 Base: Can={f1_can_base:.6f}, P47={f1_p47_base:.6f}")

# Test variations
results = []
for g_hs in [0.01, 0.02, 0.03, 0.04]:
    for b_tr in [0.01, 0.02, 0.03, 0.04, 0.05]:
        for b_ag in [0.05, 0.08, 0.10, 0.12, 0.15]:
            def transform(p_in):
                p = p_in.copy()
                cond_iso = (p[:, 1] < 0.15) & (p[:, 2] < 0.35)
                z_hs = logit(np.clip(p[:, 0], 1e-6, 1 - 1e-6))
                z_hs[cond_iso] -= g_hs * (0.35 - p[cond_iso, 2])
                p[:, 0] = expit(z_hs)
                
                cond_hs = p[:, 0] >= 0.44
                z_tr = logit(np.clip(p[:, 1], 1e-6, 1 - 1e-6))
                z_tr[cond_hs] += b_tr * (p[cond_hs, 0] - 0.44)
                p[:, 1] = expit(z_tr)
                
                cond_tr = p[:, 1] >= 0.48
                z_ag = logit(np.clip(p[:, 2], 1e-6, 1 - 1e-6))
                z_ag[cond_tr] += b_ag * (p[cond_tr, 1] - 0.48)
                p[:, 2] = expit(z_ag)
                return p

            f_c = eval_p(transform(prob_can))
            f_47 = eval_p(transform(prob_p47))
            
            p_te = transform(prob_test)
            pred_te = np.column_stack([(p_te[:, j] >= [0.44, 0.51, 0.46][j]).astype(int) for j in range(3)])
            mask = (pred_te[:, 1] == 1) & (pred_te[:, 0] == 0)
            pred_te[mask, 0] = 1
            
            diff = (pred_p114 != pred_te).sum()
            hs_c, tr_c, ag_c = pred_te[:,0].sum(), pred_te[:,1].sum(), pred_te[:,2].sum()
            tot = pred_te.sum()
            
            if (512 <= hs_c <= 517) and (328 <= tr_c <= 332) and (418 <= ag_c <= 423) and (1260 <= tot <= 1267):
                if f_c >= f1_can_base and f_47 >= f1_p47_base and (f_c > f1_can_base or f_47 > f1_p47_base):
                    results.append((f_c, f_47, (f_c+f_47)/2, g_hs, b_tr, b_ag, diff, hs_c, tr_c, ag_c, tot))

results.sort(key=lambda x: (x[2], x[0]), reverse=True)
print(f"Top 5 fine-tuned results (total valid: {len(results)}):")
for r in results[:5]:
    f_c, f_47, med, g_hs, b_tr, b_ag, diff, hs_c, tr_c, ag_c, tot = r
    print(f"g_hs={g_hs:.2f}, b_tr={b_tr:.2f}, b_ag={b_ag:.2f} | Can={f_c:.6f} (+{f_c-f1_can_base:.6f}) | P47={f_47:.6f} (+{f_47-f1_p47_base:.6f}) | Med={med:.6f} | diff={diff} cells | Test: HS={hs_c}, TR={tr_c}, AG={ag_c}, Tot={tot}")

