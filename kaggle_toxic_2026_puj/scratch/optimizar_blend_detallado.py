import numpy as np
from scipy.special import expit, logit
import sys
from pathlib import Path

RAIZ = Path(".").resolve()
sys.path.insert(0, str(RAIZ))
from src.evaluation import ETIQUETAS, metricas_multietiqueta

data = np.load("scratch/individual_models_oof.npz")
lr_can = data["lr_can"]
sgd_can = data["sgd_can"]
saga_can = data["saga_can"]

lr_p47 = data["lr_p47"]
sgd_p47 = data["sgd_p47"]
saga_p47 = data["saga_p47"]
y_dev = data["y"]

UMBRALES_CONGELADOS = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

# Baseline P114 calculation
def calc_p114(lr, sgd, saga, w_lr=0.80, w_sgd=0.05, w_saga=0.15, T=1.04):
    p_blend = w_lr * lr + w_sgd * sgd + w_saga * saga
    p_cal = expit(logit(np.clip(p_blend, 1e-6, 1 - 1e-6)) / T)
    return p_cal

def eval_metric(prob):
    pred = np.column_stack([(prob[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    mask = (pred[:, 1] == 1) & (pred[:, 0] == 0)
    pred[mask, 0] = 1
    m = metricas_multietiqueta(y_dev, pred)
    return m["f1_media_columnas"], m["f1_por_etiqueta"]

# Verify P114 baseline exactly
p_can_base = calc_p114(lr_can, sgd_can, saga_can)
p_p47_base = calc_p114(lr_p47, sgd_p47, saga_p47)
f1_can_base, f1_can_labels = eval_metric(p_can_base)
f1_p47_base, f1_p47_labels = eval_metric(p_p47_base)
print(f"P114 Baseline: Canonica={f1_can_base:.6f}, Part47={f1_p47_base:.6f}, Media={(f1_can_base+f1_p47_base)/2:.6f}")
print("Canonica per label:", {e: round(f1_can_labels[e], 6) for e in ETIQUETAS})

# Grid search over weights: w_lr, w_sgd, w_saga where sum = 1.0
best_media = (f1_can_base + f1_p47_base) / 2
best_can = f1_can_base
best_config = None

configs = []
for w_sgd in np.linspace(0.01, 0.12, 12):
    for w_saga in np.linspace(0.05, 0.35, 13):
        w_lr = 1.0 - w_sgd - w_saga
        if w_lr < 0.50:
            continue
        for T in [1.00, 1.02, 1.04, 1.06, 1.08]:
            p_can = calc_p114(lr_can, sgd_can, saga_can, w_lr, w_sgd, w_saga, T)
            p_p47 = calc_p114(lr_p47, sgd_p47, saga_p47, w_lr, w_sgd, w_saga, T)
            f_can, _ = eval_metric(p_can)
            f_p47, _ = eval_metric(p_p47)
            med = (f_can + f_p47) / 2
            configs.append((f_can, f_p47, med, w_lr, w_sgd, w_saga, T))

configs.sort(key=lambda x: x[2], reverse=True)
print("\nTop 10 configurations by combined Media CV:")
for f_can, f_p47, med, w_lr, w_sgd, w_saga, T in configs[:10]:
    d_can = f_can - f1_can_base
    d_p47 = f_p47 - f1_p47_base
    d_med = med - ((f1_can_base + f1_p47_base) / 2)
    print(f"w_lr={w_lr:.2f}, w_sgd={w_sgd:.2f}, w_saga={w_saga:.2f}, T={T:.2f} | Can={f_can:.6f} ({d_can:+.6f}) | P47={f_p47:.6f} ({d_p47:+.6f}) | Med={med:.6f} ({d_med:+.6f})")

# Look for configs that strictly improve BOTH Canonica AND Part47:
both_better = [c for c in configs if c[0] >= f1_can_base and c[1] >= f1_p47_base and (c[0] > f1_can_base or c[1] > f1_p47_base)]
print(f"\nConfigurations strictly improving BOTH Canonica and Part47: {len(both_better)}")
for f_can, f_p47, med, w_lr, w_sgd, w_saga, T in both_better[:10]:
    d_can = f_can - f1_can_base
    d_p47 = f_p47 - f1_p47_base
    d_med = med - ((f1_can_base + f1_p47_base) / 2)
    print(f"w_lr={w_lr:.2f}, w_sgd={w_sgd:.2f}, w_saga={w_saga:.2f}, T={T:.2f} | Can={f_can:.6f} ({d_can:+.6f}) | P47={f_p47:.6f} ({d_p47:+.6f}) | Med={med:.6f} ({d_med:+.6f})")

