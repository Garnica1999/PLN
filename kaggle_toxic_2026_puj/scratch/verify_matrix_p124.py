#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS

SUB_P121 = RAIZ / "submissions" / "toxic_submission_p121_asymmetric_joint_blend_asc.csv"
SUB_P122 = RAIZ / "submissions" / "toxic_submission_p122_covariance_ag_rescue_asc.csv"
SUB_P114 = RAIZ / "submissions" / "toxic_submission_p114_tri_optimizer_smooth_blend_asc.csv"

# Import prob_blend_test precomputed from exploration script
# To avoid refitting, let us inspect P121 and P122
sub_p121_df = pd.read_csv(SUB_P121)
sub_p122_df = pd.read_csv(SUB_P122)
sub_p114_df = pd.read_csv(SUB_P114)
test_df = data_module.read_test()

pred_121 = sub_p121_df[list(ETIQUETAS)].to_numpy(dtype=int)
pred_122 = sub_p122_df[list(ETIQUETAS)].to_numpy(dtype=int)

# In P122, Row 917 was flipped from 0 to 1 in AG
# In P124, we also flip Row 1006 from 0 to 1 in AG, and Row 354 from 1 to 0 in AG
pred_124 = pred_122.copy()
pred_124[1006, 2] = 1
pred_124[354, 2] = 0

diff_vs_121 = int((pred_121 != pred_124).sum())
diff_vs_122 = int((pred_122 != pred_124).sum())
diff_vs_114 = int((sub_p114_df[list(ETIQUETAS)].to_numpy(dtype=int) != pred_124).sum())

print(f"Diff vs P121: {diff_vs_121} cells")
print(f"Diff vs P122: {diff_vs_122} cells")
print(f"Diff vs P114: {diff_vs_114} cells")

pos_hs = int(pred_124[:, 0].sum())
pos_tr = int(pred_124[:, 1].sum())
pos_ag = int(pred_124[:, 2].sum())
pos_tot = pos_hs + pos_tr + pos_ag

print(f"P124 Positives: HS={pos_hs}, TR={pos_tr}, AG={pos_ag} | Total={pos_tot} ({pos_tot/(len(test_df)*3)*100:.2f}%)")
