#!/usr/bin/env python
"""Exploracion profunda P116: Optimizacion de Representacion y Regularizacion Continua."""

from __future__ import annotations
import sys
from pathlib import Path
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
SUB_P114 = RAIZ / "submissions" / "toxic_submission_p114_tri_optimizer_smooth_blend_asc.csv"

UMBRALES_BASE = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

def aplicar_regla_ascendente(pred: np.ndarray) -> np.ndarray:
    p = pred.copy()
    mask = (p[:, 1] == 1) & (p[:, 0] == 0)
    p[mask, 0] = 1
    return p

def main():
    print("=== EXPLORACION P116: REPRESENTACION Y REGULARIZACION CONTINUA ===")
    train_full = data_module.read_train()
    test_df = data_module.read_test()

    zf = np.load(OOF_FINAL, allow_pickle=True)
    zb = np.load(OOF_BASE, allow_pickle=True)
    pliegues_can = zf["pliegues"].astype(int)
    grupos = zb["grupos"].astype(np.int64)

    particion = pd.read_csv(IDS_PATH)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    train_dev = train_full[train_full["id"].astype(int).isin(ids_dev)].reset_index(drop=True)

    y_true_dev = train_dev[list(ETIQUETAS)].to_numpy(dtype=int)
    combinacion = np.array([f"{a}{b}{c}" for a, b, c in y_true_dev])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
    pliegues_47 = np.full(len(train_dev), -1, dtype=int)
    for num, (_, val) in enumerate(sgkf.split(np.arange(len(train_dev)).reshape(-1, 1), combinacion, groups=grupos)):
        pliegues_47[val] = num + 1

    particiones = [("Canonica", pliegues_can), ("Part47", pliegues_47)]

    # Cargar P114 para comparar
    sub_p114 = pd.read_csv(SUB_P114)
    pred_p114_test = sub_p114[list(ETIQUETAS)].to_numpy(dtype=int)

    # Configuraciones de Representacion a evaluar
    espacios_features = [
        ("V2_triple (Base P114)", [
            ("word", (1, 2), True, None),
            ("char_wb", (3, 5), True, None),
            ("char", (1, 6), True, None),
        ]),
        ("V2_quad_word3 (con trigramas de palabras)", [
            ("word", (1, 3), True, None),
            ("char_wb", (3, 5), True, None),
            ("char", (1, 6), True, None),
        ]),
        ("V2_charwb6 (char_wb 3-6)", [
            ("word", (1, 2), True, None),
            ("char_wb", (3, 6), True, None),
            ("char", (1, 6), True, None),
        ]),
        ("V2_maxdf90 (filtrando stopwords de frecuencia > 90%)", [
            ("word", (1, 2), True, 0.90),
            ("char_wb", (3, 5), True, 0.90),
            ("char", (1, 6), True, 0.90),
        ]),
    ]

    for nombre_espacio, bloques in espacios_features:
        print(f"\n=======================================================")
        print(f"Evaluando Espacio: {nombre_espacio}")
        print(f"=======================================================")

        # Precalcular matrices y probabilidades para ambas particiones
        probs_por_particion = {}
        for nombre_part, pliegues in particiones:
            n_samples = len(train_dev)
            prob_lr = np.zeros((n_samples, 3), dtype=float)
            prob_sgd = np.zeros((n_samples, 3), dtype=float)
            prob_saga = np.zeros((n_samples, 3), dtype=float)

            for fold in sorted(set(pliegues)):
                idx_tr = np.flatnonzero(pliegues != fold)
                idx_va = np.flatnonzero(pliegues == fold)

                mats_tr, mats_va = [], []
                for analyzer, ngram, sublinear, max_df in bloques:
                    kwargs = {"analyzer": analyzer, "ngram_range": ngram, "min_df": 2, "sublinear_tf": sublinear, "lowercase": True}
                    if max_df is not None:
                        kwargs["max_df"] = max_df
                    v = TfidfVectorizer(**kwargs)
                    mats_tr.append(v.fit_transform(train_dev["text"].iloc[idx_tr]))
                    mats_va.append(v.transform(train_dev["text"].iloc[idx_va]))
                X_tr = sparse.hstack(mats_tr).tocsr()
                X_va = sparse.hstack(mats_va).tocsr()

                for j, et in enumerate(ETIQUETAS):
                    y_col = train_dev[et].iloc[idx_tr]
                    c_val = {"HS": 2.0, "TR": 1.8, "AG": 1.5}[et]

                    lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
                    lr.fit(X_tr, y_col)
                    prob_lr[idx_va, j] = lr.predict_proba(X_va)[:, 1]

                    sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
                    sgd.fit(X_tr, y_col)
                    prob_sgd[idx_va, j] = sgd.predict_proba(X_va)[:, 1]

                    saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
                    saga.fit(X_tr, y_col)
                    prob_saga[idx_va, j] = saga.predict_proba(X_va)[:, 1]

            probs_por_particion[nombre_part] = (prob_lr, prob_sgd, prob_saga)

        # Evaluar configuracion P114 en este espacio
        eps = 1e-6
        f1_list = []
        for nombre_part, pliegues in particiones:
            prob_lr, prob_sgd, prob_saga = probs_por_particion[nombre_part]
            p_blend = 0.80 * prob_lr + 0.05 * prob_sgd + 0.15 * prob_saga
            p_cal = expit(logit(np.clip(p_blend, eps, 1 - eps)) / 1.04)
            pred_raw = np.column_stack([(p_cal[:, j] >= UMBRALES_BASE[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
            pred = aplicar_regla_ascendente(pred_raw)
            m = metricas_multietiqueta(y_true_dev, pred)
            f1_list.append(m["f1_media_columnas"])
        media_ccv = np.mean(f1_list)
        print(f"P114 en {nombre_espacio} -> Canonica: {f1_list[0]:.6f}, Part47: {f1_list[1]:.6f}, Media: {media_ccv:.6f} (Delta vs P114 Base: {media_ccv - 0.786446:+.6f})")

    # Exploracion de C_AG y pesos de ensamble
    print("\n=======================================================")
    print("Pruebas de Regularizacion C_AG y Simplex de Pesos")
    print("=======================================================")
    # Para el espacio V2_triple base, explorar variar C_AG directamente
    for c_ag in [0.8, 1.0, 1.2, 1.5, 1.8, 2.0]:
        c_dict = {"HS": 2.0, "TR": 1.8, "AG": c_ag}
        f1_list = []
        for nombre_part, pliegues in particiones:
            n_samples = len(train_dev)
            prob_lr = np.zeros((n_samples, 3), dtype=float)
            prob_sgd = np.zeros((n_samples, 3), dtype=float)
            prob_saga = np.zeros((n_samples, 3), dtype=float)

            for fold in sorted(set(pliegues)):
                idx_tr = np.flatnonzero(pliegues != fold)
                idx_va = np.flatnonzero(pliegues == fold)

                mats_tr, mats_va = [], []
                for analyzer, ngram, sublinear, _ in espacios_features[0][1]:
                    v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
                    mats_tr.append(v.fit_transform(train_dev["text"].iloc[idx_tr]))
                    mats_va.append(v.transform(train_dev["text"].iloc[idx_va]))
                X_tr = sparse.hstack(mats_tr).tocsr()
                X_va = sparse.hstack(mats_va).tocsr()

                for j, et in enumerate(ETIQUETAS):
                    y_col = train_dev[et].iloc[idx_tr]
                    c_val = c_dict[et]

                    lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
                    lr.fit(X_tr, y_col)
                    prob_lr[idx_va, j] = lr.predict_proba(X_va)[:, 1]

                    sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
                    sgd.fit(X_tr, y_col)
                    prob_sgd[idx_va, j] = sgd.predict_proba(X_va)[:, 1]

                    saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
                    saga.fit(X_tr, y_col)
                    prob_saga[idx_va, j] = saga.predict_proba(X_va)[:, 1]

            p_blend = 0.80 * prob_lr + 0.05 * prob_sgd + 0.15 * prob_saga
            p_cal = expit(logit(np.clip(p_blend, eps, 1 - eps)) / 1.04)
            pred_raw = np.column_stack([(p_cal[:, j] >= UMBRALES_BASE[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
            pred = aplicar_regla_ascendente(pred_raw)
            m = metricas_multietiqueta(y_true_dev, pred)
            f1_list.append(m["f1_media_columnas"])

        mean_f1 = np.mean(f1_list)
        print(f"C_AG={c_ag:.1f} -> Canonica: {f1_list[0]:.6f}, Part47: {f1_list[1]:.6f}, Media: {mean_f1:.6f} (Delta: {mean_f1 - 0.786446:+.6f})")

if __name__ == "__main__":
    main()
