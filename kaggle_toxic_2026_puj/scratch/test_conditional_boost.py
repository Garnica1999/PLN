import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit

RAIZ = Path(".").resolve()
sys.path.insert(0, str(RAIZ))

from src.evaluation import ETIQUETAS, metricas_multietiqueta

# Load OOFs and test probs
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

pred_can_base = eval_preds(prob_can)
pred_p47_base = eval_preds(prob_p47)
pred_test_base = eval_preds(prob_test)

f1_can_base = metricas_multietiqueta(y_dev, pred_can_base)["f1_media_columnas"]
f1_p47_base = metricas_multietiqueta(y_dev, pred_p47_base)["f1_media_columnas"]
med_base = (f1_can_base + f1_p47_base) / 2
print(f"P114 Baseline: Can={f1_can_base:.6f}, P47={f1_p47_base:.6f}, Med={med_base:.6f}")
print(f"P114 Test Positives: HS={pred_test_base[:,0].sum()}, TR={pred_test_base[:,1].sum()}, AG={pred_test_base[:,2].sum()} | Total={pred_test_base.sum()}")

# Let's explore Conditional Boost from TR to AG:
# If P(TR) is high, P(AG | TR) is 0.946 in ground truth.
# How does boosting AG log-odds when P(TR) is high affect Canonica and Part47?
print("\n--- Exploring TR -> AG Conditional Boost ---")
for beta_tr in [0.02, 0.05, 0.08, 0.10, 0.15]:
    for tr_threshold in [0.50, 0.55, 0.60, 0.70]:
        # Boost AG where P(TR) >= tr_threshold
        # P_new(AG) = sigma(logit(P(AG)) + beta_tr * (P(TR) - tr_threshold))
        def apply_boost(prob):
            p = prob.copy()
            cond = p[:, 1] >= tr_threshold
            z_ag = logit(np.clip(p[:, 2], 1e-6, 1 - 1e-6))
            z_ag[cond] += beta_tr * (p[cond, 1] - tr_threshold)
            p[:, 2] = expit(z_ag)
            return p
        
        p_can = apply_boost(prob_can)
        p_p47 = apply_boost(prob_p47)
        p_te = apply_boost(prob_test)
        
        f_can = metricas_multietiqueta(y_dev, eval_preds(p_can))["f1_media_columnas"]
        f_p47 = metricas_multietiqueta(y_dev, eval_preds(p_p47))["f1_media_columnas"]
        med = (f_can + f_p47) / 2
        
        pred_te = eval_preds(p_te)
        n_hs, n_tr, n_ag = pred_te[:,0].sum(), pred_te[:,1].sum(), pred_te[:,2].sum()
        tot = pred_te.sum()
        
        if med >= med_base and f_can >= f1_can_base:
            d_can = f_can - f1_can_base
            d_p47 = f_p47 - f1_p47_base
            print(f"beta={beta_tr:.2f}, tr_th={tr_threshold:.2f} | Can: {f_can:.6f} ({d_can:+.6f}) | P47: {f_p47:.6f} ({d_p47:+.6f}) | Med: {med:.6f} | Test: HS={n_hs}, TR={n_tr}, AG={n_ag}, Tot={tot}")

