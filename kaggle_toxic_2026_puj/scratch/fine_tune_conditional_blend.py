import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit

RAIZ = Path(".").resolve()
sys.path.insert(0, str(RAIZ))

from src.evaluation import ETIQUETAS, metricas_multietiqueta

oofs = np.load("scratch/p114_oofs.npz")
prob_can = oofs["can"]
prob_p47 = oofs["p47"]
y_dev = oofs["y"]

test_data = np.load("scratch/p114_test_probs.npz")
prob_test = test_data["prob"]

UMBRALES_BASE = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

def eval_preds(probs, ths=UMBRALES_BASE):
    pred = np.column_stack([(probs[:, j] >= ths[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    mask = (pred[:, 1] == 1) & (pred[:, 0] == 0)
    pred[mask, 0] = 1
    return pred

f1_can_base = metricas_multietiqueta(y_dev, eval_preds(prob_can))["f1_media_columnas"]
f1_p47_base = metricas_multietiqueta(y_dev, eval_preds(prob_p47))["f1_media_columnas"]
med_base = (f1_can_base + f1_p47_base) / 2

print(f"P114 Baseline: Can={f1_can_base:.6f}, P47={f1_p47_base:.6f}, Med={med_base:.6f}")

# Explore joint conditioning:
# 1. TR -> AG covariance coupling: z_ag += beta_ag * (P(TR) - tr_th)
# 2. HS -> TR hierarchical gating: z_tr += beta_tr * (P(HS) - hs_th)
# 3. HS FP suppression: if P(TR) < 0.15 and P(AG) < 0.35, z_hs -= gamma_hs * (0.35 - P(AG))

best_configs = []

for beta_ag in [0.0, 0.05, 0.10, 0.12, 0.15, 0.20]:
    for tr_th in [0.48, 0.50, 0.51, 0.55]:
        for beta_tr in [0.0, 0.02, 0.05, 0.08]:
            for hs_th in [0.44, 0.50]:
                for gamma_hs in [0.0, 0.02, 0.05]:
                    def transform(p_in):
                        p = p_in.copy()
                        # HS FP suppression in isolated region
                        if gamma_hs > 0:
                            cond_iso = (p[:, 1] < 0.15) & (p[:, 2] < 0.35)
                            z_hs = logit(np.clip(p[:, 0], 1e-6, 1 - 1e-6))
                            z_hs[cond_iso] -= gamma_hs * (0.35 - p[cond_iso, 2])
                            p[:, 0] = expit(z_hs)
                        
                        # HS -> TR hierarchical gating
                        if beta_tr > 0:
                            cond_hs = p[:, 0] >= hs_th
                            z_tr = logit(np.clip(p[:, 1], 1e-6, 1 - 1e-6))
                            z_tr[cond_hs] += beta_tr * (p[cond_hs, 0] - hs_th)
                            p[:, 1] = expit(z_tr)
                            
                        # TR -> AG covariance coupling
                        if beta_ag > 0:
                            cond_tr = p[:, 1] >= tr_th
                            z_ag = logit(np.clip(p[:, 2], 1e-6, 1 - 1e-6))
                            z_ag[cond_tr] += beta_ag * (p[cond_tr, 1] - tr_th)
                            p[:, 2] = expit(z_ag)
                            
                        return p

                    p_c = transform(prob_can)
                    p_47 = transform(prob_p47)
                    p_t = transform(prob_test)
                    
                    f_c = metricas_multietiqueta(y_dev, eval_preds(p_c))["f1_media_columnas"]
                    f_47 = metricas_multietiqueta(y_dev, eval_preds(p_47))["f1_media_columnas"]
                    med = (f_c + f_47) / 2
                    
                    pred_t = eval_preds(p_t)
                    n_hs = pred_t[:, 0].sum()
                    n_tr = pred_t[:, 1].sum()
                    n_ag = pred_t[:, 2].sum()
                    tot = pred_t.sum()
                    
                    # Check prevalence anchor:
                    # HS: 512-517, TR: 328-332, AG: 418-423, Total: 1260-1266
                    if (512 <= n_hs <= 517) and (328 <= n_tr <= 332) and (418 <= n_ag <= 423) and (1260 <= tot <= 1267):
                        if f_c >= f1_can_base and f_47 >= f1_p47_base and (f_c > f1_can_base or f_47 > f1_p47_base):
                            best_configs.append((f_c, f_47, med, beta_ag, tr_th, beta_tr, hs_th, gamma_hs, n_hs, n_tr, n_ag, tot))

best_configs.sort(key=lambda x: (x[2], x[0]), reverse=True)
print(f"Total valid improving configurations satisfying prevalence invariant: {len(best_configs)}")
for c in best_configs[:15]:
    f_c, f_47, med, beta_ag, tr_th, beta_tr, hs_th, gamma_hs, n_hs, n_tr, n_ag, tot = c
    d_c = f_c - f1_can_base
    d_47 = f_47 - f1_p47_base
    d_med = med - med_base
    print(f"b_ag={beta_ag:.2f}(th={tr_th:.2f}), b_tr={beta_tr:.2f}(th={hs_th:.2f}), g_hs={gamma_hs:.2f} | Can={f_c:.6f} ({d_c:+.6f}) | P47={f_47:.6f} ({d_47:+.6f}) | Med={med:.6f} ({d_med:+.6f}) | Test: {n_hs},{n_tr},{n_ag} (Tot={tot})")

