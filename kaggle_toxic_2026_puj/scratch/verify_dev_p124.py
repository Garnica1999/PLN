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

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

OOF_FINAL = RAIZ / "features" / "oof_v03_final.npz"
OOF_BASE = RAIZ / "features" / "v2_oof_base_corrected.npz"
IDS_PATH = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"

train_full = data_module.read_train()
particion = pd.read_csv(IDS_PATH)
ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
train_dev = train_full[train_full["id"].astype(int).isin(ids_dev)].reset_index(drop=True)

zf = np.load(OOF_FINAL, allow_pickle=True)
pliegues_can = zf["pliegues"].astype(int)

# Let us run fold 1 or quick check on Canonica for rows 1133, 1298
RE_COMERCIAL = re.compile(r"(?i)\b(sorteo|entradas|premio|concurso|fiesta exclusiva|ganadoras del primer)\b")
RE_VOCATIVO_ATAQUE = re.compile(r"(?i)@\w+.*(c[al]{1,2}ate|callate|cllate)")

print("Script template ready.")
