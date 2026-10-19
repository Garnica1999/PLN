#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

OOF_FINAL = RAIZ / "features" / "oof_v03_final.npz"
OOF_BASE = RAIZ / "features" / "v2_oof_base_corrected.npz"
IDS_PATH = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
SUB_P121 = RAIZ / "submissions" / "toxic_submission_p121_asymmetric_joint_blend_asc.csv"
SUB_P122 = RAIZ / "submissions" / "toxic_submission_p122_covariance_ag_rescue_asc.csv"

train_full = data_module.read_train()
test_df = data_module.read_test()

particion = pd.read_csv(IDS_PATH)
ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
train_dev = train_full[train_full["id"].astype(int).isin(ids_dev)].reset_index(drop=True)

zf = np.load(OOF_FINAL, allow_pickle=True)
zb = np.load(OOF_BASE, allow_pickle=True)
pliegues_can = zf["pliegues"].astype(int)
grupos = zb["grupos"].astype(np.int64)

# Particion 47
y_true_dev = train_dev[list(ETIQUETAS)].to_numpy(dtype=int)
combinacion = np.array([f"{a}{b}{c}" for a, b, c in y_true_dev])
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
pliegues_47 = np.full(len(train_dev), -1, dtype=int)
for num, (_, val) in enumerate(sgkf.split(np.arange(len(train_dev)).reshape(-1, 1), combinacion, groups=grupos)):
    pliegues_47[val] = num + 1

# Check vocativo regex in train_dev
RE_VOCATIVO = re.compile(r"(?i)(@\w+.*(c[al]{1,2}ate|callate|cllate)|c[al]{1,2}ate|callate|cllate)")
print("Rows matching RE_VOCATIVO in train_dev:")
for i, t in enumerate(train_dev["text"]):
    if RE_VOCATIVO.search(t):
        row = train_dev.iloc[i]
        print(f"  Dev idx {i} (id {row.id}): HS={row.HS}, TR={row.TR}, AG={row.AG} | {t[:80]}")

RE_COMERCIAL = re.compile(r"(?i)\b(sorteo|entradas|premio|concurso|fiesta exclusiva|ganadoras del primer)\b")
print("\nRows matching RE_COMERCIAL in train_dev:")
for i, t in enumerate(train_dev["text"]):
    if RE_COMERCIAL.search(t):
        row = train_dev.iloc[i]
        print(f"  Dev idx {i} (id {row.id}): HS={row.HS}, TR={row.TR}, AG={row.AG} | {t[:80]}")
