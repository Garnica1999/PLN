#!/usr/bin/env python
"""Exploracion rapida para P113: Comparacion de familias de optimizadores y modelos sobre V2_triple.

Prueba sobre particion canonica (1.483 filas dev) y particion 47:
  - Base P110: 95% LR C=2.0 + 5% SGD log_loss alpha=5e-6
  - Candidato 1: SGD modified_huber (smooth SVM margin)
  - Candidato 2: ComplementNB (clasico HatEval ganador)
  - Candidato 3: Blend LR + SGD_log + SGD_huber (multi-loss ensemble: logistic + smooth hinge)
  - Candidato 4: LR ElasticNet (saga l1_ratio=0.05, C=2.0)
  - Candidato 5: Block-weighting (reponderacion morfologica char_wb 1.2x)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

OOF_FINAL = RAIZ / "features" / "oof_v03_final.npz"
OOF_BASE = RAIZ / "features" / "v2_oof_base_corrected.npz"
IDS_PATH = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
SUB_P110 = RAIZ / "submissions" / "toxic_submission_p110_lr_sgd_blend_asc.csv"

UMBRALES_CONGELADOS = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]

def aplicar_regla_ascendente(pred: np.ndarray) -> np.ndarray:
    p = pred.copy()
    mask = (p[:, 1] == 1) & (p[:, 0] == 0)
    p[mask, 0] = 1
    return p

def construir_matrices(textos_tr, textos_va, bloques, pesos=None):
    mats_tr, mats_va = [], []
    for i, (analyzer, ngram, sublinear) in enumerate(bloques):
        v = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram,
            min_df=2,
            sublinear_tf=sublinear,
            lowercase=True,
        )
        tr = v.fit_transform(textos_tr)
        va = v.transform(textos_va)
        if pesos is not None:
            tr = tr * pesos[i]
            va = va * pesos[i]
        mats_tr.append(tr)
        mats_va.append(va)
    return sparse.hstack(mats_tr).tocsr(), sparse.hstack(mats_va).tocsr()

def main():
    print("Cargando datos...", flush=True)
    train_full = data_module.read_train()
    test_df = data_module.read_test()
    sub_p110_df = pd.read_csv(SUB_P110)
    pred_p110_test = sub_p110_df[list(ETIQUETAS)].to_numpy(dtype=int)

    zf = np.load(OOF_FINAL, allow_pickle=True)
    zb = np.load(OOF_BASE, allow_pickle=True)
    pliegues_can = zf["pliegues"].astype(int)
    grupos = zb["grupos"].astype(np.int64)

    particion = pd.read_csv(IDS_PATH)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    train_dev = train_full[train_full["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    y_dev = train_dev[list(ETIQUETAS)].to_numpy(dtype=int)
    textos_dev = train_dev["text"]

    # Particion 47
    combinacion = np.array([f"{a}{b}{c}" for a, b, c in y_dev])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
    pliegues_47 = np.full(len(train_dev), -1, dtype=int)
    for num, (_, val) in enumerate(sgkf.split(np.arange(len(train_dev)).reshape(-1, 1), combinacion, groups=grupos)):
        pliegues_47[val] = num + 1

    particiones = [
        ("Canonica", pliegues_can),
        ("Particion 47", pliegues_47),
    ]

    # Evaluaremos varias configuraciones
    # 0: P110 (referencia)
    # 1: P110 + SGD modified_huber (90% LR, 5% SGD_log, 5% SGD_huber)
    # 2: 92% LR + 4% SGD_log + 4% SGD_huber
    # 3: 90% LR + 5% SGD_log + 5% ComplementNB
    # 4: 90% LR(C=2) + 10% SGD_huber(alpha=5e-6)
    # 5: P110 con block weighting [1.0, 1.2, 1.0]
    # 6: 92% LR + 4% SGD_log + 4% SGD_huber con block weighting [1.0, 1.2, 1.0]

    configuraciones = [
        ("P110_ref", {"pesos_bloques": None, "blend": "p110"}),
        ("LR_SGDlog_SGDhuber_90_5_5", {"pesos_bloques": None, "blend": "lr_sgd_huber_90_5_5"}),
        ("LR_SGDlog_SGDhuber_92_4_4", {"pesos_bloques": None, "blend": "lr_sgd_huber_92_4_4"}),
        ("LR_SGDhuber_90_10", {"pesos_bloques": None, "blend": "lr_sgd_huber_90_10"}),
        ("LR_SGDlog_CNB_90_5_5", {"pesos_bloques": None, "blend": "lr_sgd_cnb_90_5_5"}),
        ("P110_block_1_1.2_1", {"pesos_bloques": [1.0, 1.2, 1.0], "blend": "p110"}),
        ("Blend_TriOpt_block_1_1.2_1", {"pesos_bloques": [1.0, 1.2, 1.0], "blend": "lr_sgd_huber_92_4_4"}),
        ("Blend_TriOpt_block_0.8_1.2_1.0", {"pesos_bloques": [0.8, 1.2, 1.0], "blend": "lr_sgd_huber_92_4_4"}),
    ]

    for nombre_cfg, cfg in configuraciones:
        print(f"\n--- Probando configuracion: {nombre_cfg} ---", flush=True)
        scores = {}
        for nombre_part, pliegues in particiones:
            n_samples = len(train_dev)
            prob_final = np.zeros((n_samples, len(ETIQUETAS)), dtype=float)

            for fold in sorted(set(pliegues)):
                idx_tr = np.flatnonzero(pliegues != fold)
                idx_va = np.flatnonzero(pliegues == fold)

                X_tr, X_va = construir_matrices(
                    textos_dev.iloc[idx_tr], textos_dev.iloc[idx_va], BLOQUES_V2_TRIPLE, pesos=cfg["pesos_bloques"]
                )

                for j, et in enumerate(ETIQUETAS):
                    y_tr_j = y_dev[idx_tr, j]

                    # Modelos
                    lr = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
                    lr.fit(X_tr, y_tr_j)
                    p_lr = lr.predict_proba(X_va)[:, 1]

                    sgd_log = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
                    sgd_log.fit(X_tr, y_tr_j)
                    p_sgd_log = sgd_log.predict_proba(X_va)[:, 1]

                    blend_type = cfg["blend"]
                    if blend_type == "p110":
                        p_j = 0.95 * p_lr + 0.05 * p_sgd_log
                    elif blend_type == "lr_sgd_huber_90_5_5":
                        sgd_huber = SGDClassifier(loss="modified_huber", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
                        sgd_huber.fit(X_tr, y_tr_j)
                        p_sgd_huber = sgd_huber.predict_proba(X_va)[:, 1]
                        p_j = 0.90 * p_lr + 0.05 * p_sgd_log + 0.05 * p_sgd_huber
                    elif blend_type == "lr_sgd_huber_92_4_4":
                        sgd_huber = SGDClassifier(loss="modified_huber", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
                        sgd_huber.fit(X_tr, y_tr_j)
                        p_sgd_huber = sgd_huber.predict_proba(X_va)[:, 1]
                        p_j = 0.92 * p_lr + 0.04 * p_sgd_log + 0.04 * p_sgd_huber
                    elif blend_type == "lr_sgd_huber_90_10":
                        sgd_huber = SGDClassifier(loss="modified_huber", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
                        sgd_huber.fit(X_tr, y_tr_j)
                        p_sgd_huber = sgd_huber.predict_proba(X_va)[:, 1]
                        p_j = 0.90 * p_lr + 0.10 * p_sgd_huber
                    elif blend_type == "lr_sgd_cnb_90_5_5":
                        cnb = ComplementNB(alpha=1.0, norm=True)
                        cnb.fit(X_tr, y_tr_j)
                        p_cnb = cnb.predict_proba(X_va)[:, 1]
                        p_j = 0.90 * p_lr + 0.05 * p_sgd_log + 0.05 * p_cnb

                    prob_final[idx_va, j] = p_j

            # Predicciones binarias y F1
            pred_raw = np.column_stack([(prob_final[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
            pred_asc = aplicar_regla_ascendente(pred_raw)
            met = metricas_multietiqueta(y_dev, pred_asc)
            scores[nombre_part] = met["f1_media_columnas"]

        ccv_can = scores["Canonica"]
        ccv_p47 = scores["Particion 47"]
        ccv_media = (ccv_can + ccv_p47) / 2.0
        delta_p110 = ccv_media - 0.784658
        print(f"  Resultado {nombre_cfg}: Canonica={ccv_can:.6f} | Part47={ccv_p47:.6f} | Media={ccv_media:.6f} | Delta vs P110={delta_p110:+.6f}", flush=True)

if __name__ == "__main__":
    main()
