#!/usr/bin/env python
"""scratch/test_prevalence_calibration.py

Simulacion de calibracion de umbrales y pesos para:
1. Blindar HS >= 515 (no suprimir TPs de odio)
2. Podar AG hacia [400, 415] (eliminar falsos positivos en tweets coloquiales)
3. Estabilizar TR en [328, 334]
4. Medir CCV en Canonica y Particion 47
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

# Cargamos OOF de Canonica cacheado
cache_path = RAIZ / "scratch" / "oof_6models.npz"
data = np.load(cache_path)
oof_probs = data["oof_probs"]

# Modelos:
# 0: LR Liblinear
# 1: SGD log_loss
# 2: SAGA ElasticNet
# 3: Ridge Platt
# 4: SVD Dense LR
# 5: BM25 LR

train_full = data_module.read_train()
zf = np.load(RAIZ / "features" / "oof_v03_final.npz", allow_pickle=True)
pliegues_can = zf["pliegues"].astype(int)

particion = pd.read_csv(RAIZ / "data" / "processed" / "holdout_v2_ids.csv")
ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
train_dev = train_full[train_full["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
y_dev = train_dev[list(ETIQUETAS)].to_numpy(dtype=int)

# P114 base en dev
p114_blend = 0.80 * oof_probs[0] + 0.05 * oof_probs[1] + 0.15 * oof_probs[2]
eps = 1e-6
p114_cal = expit(logit(np.clip(p114_blend, eps, 1 - eps)) / 1.04)

bm25_cal = oof_probs[5]
svd_cal = oof_probs[4]

def aplicar_regla_ascendente(pred: np.ndarray) -> tuple[np.ndarray, int]:
    p = pred.copy()
    mask = (p[:, 1] == 1) & (p[:, 0] == 0)
    p[mask, 0] = 1
    return p, int(mask.sum())

print("=== CALIBRACION DE SENSIBILIDAD EN CANONICA ===")
# Base P114
p114_raw = np.column_stack([(p114_cal[:, j] >= [0.44, 0.51, 0.46][j]).astype(int) for j in range(3)])
p114_asc, _ = aplicar_regla_ascendente(p114_raw)
met_p114 = metricas_multietiqueta(y_dev, p114_asc)
print(f"P114 Canonica: {met_p114['f1_media_columnas']:.6f} | HS={met_p114['f1_por_etiqueta']['HS']:.4f}, TR={met_p114['f1_por_etiqueta']['TR']:.4f}, AG={met_p114['f1_por_etiqueta']['AG']:.4f}")

# Exploramos configuraciones donde:
# - HS: peso P114 alto (0.85 - 0.90) para blindar HS >= 515
# - AG: threshold AG ligeramente mas exigente (0.465 o 0.47) O menor peso SVD (0.10 - 0.15) para evitar inflar falsos positivos
configs = [
    # (nombre, w_hs, w_tr, w_ag, thr_hs, thr_tr, thr_ag)
    ("Blend A: SVD moderado (0.10/0.10/0.15) thr 0.44/0.51/0.46", 
     (0.85, 0.05, 0.10), (0.85, 0.05, 0.10), (0.80, 0.05, 0.15), 0.44, 0.51, 0.46),
    ("Blend B: SVD conservador (0.05/0.08/0.10) thr 0.44/0.51/0.46", 
     (0.90, 0.05, 0.05), (0.87, 0.05, 0.08), (0.85, 0.05, 0.10), 0.44, 0.51, 0.46),
    ("Blend C: SVD moderado + AG thr 0.47", 
     (0.85, 0.05, 0.10), (0.85, 0.05, 0.10), (0.80, 0.00, 0.20), 0.44, 0.51, 0.47),
    ("Blend D: SVD conservador + AG thr 0.465", 
     (0.90, 0.05, 0.05), (0.85, 0.05, 0.10), (0.85, 0.00, 0.15), 0.44, 0.51, 0.465),
    ("Blend E: BM25 dominante en TR (0.85/0.10/0.05 en TR)", 
     (0.90, 0.05, 0.05), (0.80, 0.15, 0.05), (0.85, 0.00, 0.15), 0.44, 0.51, 0.465),
    ("Blend F: Desacoplado 0.88/0.05/0.07 HS, 0.82/0.08/0.10 TR, 0.80/0.05/0.15 AG, thr 0.44/0.51/0.465",
     (0.88, 0.05, 0.07), (0.82, 0.08, 0.10), (0.80, 0.05, 0.15), 0.44, 0.51, 0.465),
]

for nom, w_hs, w_tr, w_ag, th_hs, th_tr, th_ag in configs:
    p_hs = w_hs[0] * p114_cal[:, 0] + w_hs[1] * bm25_cal[:, 0] + w_hs[2] * svd_cal[:, 0]
    p_tr = w_tr[0] * p114_cal[:, 1] + w_tr[1] * bm25_cal[:, 1] + w_tr[2] * svd_cal[:, 1]
    p_ag = w_ag[0] * p114_cal[:, 2] + w_ag[1] * bm25_cal[:, 2] + w_ag[2] * svd_cal[:, 2]
    
    p_comb = np.column_stack([p_hs, p_tr, p_ag])
    thrs = [th_hs, th_tr, th_ag]
    raw = np.column_stack([(p_comb[:, j] >= thrs[j]).astype(int) for j in range(3)])
    asc, n_asc = aplicar_regla_ascendente(raw)
    m = metricas_multietiqueta(y_dev, asc)
    d = m['f1_media_columnas'] - met_p114['f1_media_columnas']
    print(f"\n{nom}:")
    print(f"  CCV: {m['f1_media_columnas']:.6f} (Delta: {d:+.6f}) | HS={m['f1_por_etiqueta']['HS']:.4f}, TR={m['f1_por_etiqueta']['TR']:.4f}, AG={m['f1_por_etiqueta']['AG']:.4f}")
    print(f"  Positivos dev: HS={asc[:, 0].sum()}, TR={asc[:, 1].sum()}, AG={asc[:, 2].sum()} | Total: {asc.sum()}")
