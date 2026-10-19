#!/usr/bin/env python
"""scratch/test_targeted_ag_upgrade.py

Validacion rigurosa de Ensamble Desacoplado Especializado (Candidate P118):
- HS: Nucleo Campeon P114 (80% LR + 5% SGD + 15% SAGA, T=1.04, thr=0.44) -> Blindaje total de los 515 positivos de HS
- TR: Nucleo Campeon P114 (thr=0.51) -> Blindaje de los 330 positivos de TR
- AG: Ensamble Hibrido Semantico Denso (80% P114 + 20% TruncatedSVD(150), thr=0.47) -> Reduccion de falsos positivos en tweets coloquiales y rescate de agresividad semantica
- Regla ontologica HatEval: TR=1 => HS=1
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

UMBRALES_BASE = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
C_PER_LABEL = {"HS": 2.0, "TR": 1.8, "AG": 1.5}


def aplicar_regla_ascendente(pred: np.ndarray) -> tuple[np.ndarray, int]:
    p = pred.copy()
    mask = (p[:, 1] == 1) & (p[:, 0] == 0)
    p[mask, 0] = 1
    return p, int(mask.sum())


def evaluar_particion(textos: pd.Series, y: pd.DataFrame, pliegues: np.ndarray, nombre: str):
    n_samples = len(textos)
    prob_p114 = np.zeros((n_samples, 3), dtype=float)
    prob_svd = np.zeros((n_samples, 3), dtype=float)

    bloques = [
        ("word", (1, 2), True),
        ("char_wb", (3, 5), True),
        ("char", (1, 6), True),
    ]

    for fold in sorted(set(pliegues)):
        idx_tr = np.flatnonzero(pliegues != fold)
        idx_va = np.flatnonzero(pliegues == fold)

        mats_tfidf_tr, mats_tfidf_va = [], []
        for analyzer, ngram, sublinear in bloques:
            v_tfidf = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
            mats_tfidf_tr.append(v_tfidf.fit_transform(textos.iloc[idx_tr]))
            mats_tfidf_va.append(v_tfidf.transform(textos.iloc[idx_va]))

        X_tr = sparse.hstack(mats_tfidf_tr).tocsr()
        X_va = sparse.hstack(mats_tfidf_va).tocsr()

        svd = TruncatedSVD(n_components=150, random_state=42)
        X_svd_tr = svd.fit_transform(X_tr)
        X_svd_va = svd.transform(X_va)

        for j, et in enumerate(ETIQUETAS):
            y_col = y[et].iloc[idx_tr]
            c_val = C_PER_LABEL[et]

            # LR Liblinear
            lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr.fit(X_tr, y_col)
            p_lr = lr.predict_proba(X_va)[:, 1]

            # SGD log_loss
            sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
            sgd.fit(X_tr, y_col)
            p_sgd = sgd.predict_proba(X_va)[:, 1]

            # SAGA ElasticNet
            saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
            saga.fit(X_tr, y_col)
            p_saga = saga.predict_proba(X_va)[:, 1]

            blend_f = 0.80 * p_lr + 0.05 * p_sgd + 0.15 * p_saga
            eps = 1e-6
            prob_p114[idx_va, j] = expit(logit(np.clip(blend_f, eps, 1 - eps)) / 1.04)

            # SVD Dense LR
            lr_svd = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
            lr_svd.fit(X_svd_tr, y_col)
            prob_svd[idx_va, j] = lr_svd.predict_proba(X_svd_va)[:, 1]

    y_arr = y.to_numpy(dtype=int)

    # Baseline P114
    p114_raw = np.column_stack([(prob_p114[:, j] >= UMBRALES_BASE[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    p114_asc, _ = aplicar_regla_ascendente(p114_raw)
    met_p114 = metricas_multietiqueta(y_arr, p114_asc)

    # Candidate P118 (Targeted AG Upgrade: 0.80 P114 + 0.20 SVD con thr=0.47)
    prob_p118 = prob_p114.copy()
    prob_p118[:, 2] = 0.80 * prob_p114[:, 2] + 0.20 * prob_svd[:, 2]

    p118_raw = np.column_stack([
        (prob_p118[:, 0] >= 0.44).astype(int),
        (prob_p118[:, 1] >= 0.51).astype(int),
        (prob_p118[:, 2] >= 0.47).astype(int),
    ])
    p118_asc, n_asc = aplicar_regla_ascendente(p118_raw)
    met_p118 = metricas_multietiqueta(y_arr, p118_asc)

    delta = met_p118["f1_media_columnas"] - met_p114["f1_media_columnas"]
    print(f"\n[{nombre}]")
    print(f"  P114 Base : {met_p114['f1_media_columnas']:.6f} | HS: {met_p114['f1_por_etiqueta']['HS']:.4f}, TR: {met_p114['f1_por_etiqueta']['TR']:.4f}, AG: {met_p114['f1_por_etiqueta']['AG']:.4f}")
    print(f"  P118 Cand : {met_p118['f1_media_columnas']:.6f} (Delta: {delta:+.6f}) | HS: {met_p118['f1_por_etiqueta']['HS']:.4f}, TR: {met_p118['f1_por_etiqueta']['TR']:.4f}, AG: {met_p118['f1_por_etiqueta']['AG']:.4f}")

    # Pliegues ganados
    ganados = 0
    pliegues_uniques = sorted(set(pliegues))
    for f in pliegues_uniques:
        idx = np.flatnonzero(pliegues == f)
        f1_p = metricas_multietiqueta(y_arr[idx], p114_asc[idx])["f1_media_columnas"]
        f1_c = metricas_multietiqueta(y_arr[idx], p118_asc[idx])["f1_media_columnas"]
        if f1_c > f1_p:
            ganados += 1
        print(f"    Pliegue {f}: P114={f1_p:.6f} vs P118={f1_c:.6f} (Delta: {f1_c - f1_p:+.6f})")
    print(f"  Pliegues ganados: {ganados}/{len(pliegues_uniques)}")

    return met_p114, met_p118


def main():
    print("=== EVALUACION COMPARATIVA COMPLETA P114 vs P118 ===")
    train_full = data_module.read_train()
    zf = np.load(RAIZ / "features" / "oof_v03_final.npz", allow_pickle=True)
    zb = np.load(RAIZ / "features" / "v2_oof_base_corrected.npz", allow_pickle=True)
    pliegues_can = zf["pliegues"].astype(int)
    grupos = zb["grupos"].astype(np.int64)

    particion = pd.read_csv(RAIZ / "data" / "processed" / "holdout_v2_ids.csv")
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    train_dev = train_full[train_full["id"].astype(int).isin(ids_dev)].reset_index(drop=True)

    # 1. Particion Canonica
    m114_can, m118_can = evaluar_particion(
        train_dev["text"], train_dev[list(ETIQUETAS)], pliegues_can, "Particion Canonica (1.483 filas)"
    )

    # 2. Particion 47
    combinacion = np.array([f"{a}{b}{c}" for a, b, c in train_dev[list(ETIQUETAS)].to_numpy(dtype=int)])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
    pliegues_p47 = np.full(len(train_dev), -1, dtype=int)
    for num, (_, val) in enumerate(sgkf.split(np.arange(len(train_dev)).reshape(-1, 1), combinacion, groups=grupos)):
        pliegues_p47[val] = num + 1

    m114_p47, m118_p47 = evaluar_particion(
        train_dev["text"], train_dev[list(ETIQUETAS)], pliegues_p47, "Particion 47 (1.483 filas)"
    )

    # Medias combinadas
    media_p114 = (m114_can["f1_media_columnas"] + m114_p47["f1_media_columnas"]) / 2
    media_p118 = (m118_can["f1_media_columnas"] + m118_p47["f1_media_columnas"]) / 2
    delta_media = media_p118 - media_p114

    print(f"\n==========================================")
    print(f"MEDIA COMBINADA CANONICA + PART47:")
    print(f"  P114 Reigning Champion: {media_p114:.6f}")
    print(f"  P118 Decoupled SVD-AG : {media_p118:.6f} (Delta medio: {delta_media:+.6f})")
    print(f"==========================================")


if __name__ == "__main__":
    main()
