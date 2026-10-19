#!/usr/bin/env python
"""Exploracion profunda Fase 2 para P115: Gating Ontologico Fino, SAGA tuning, y TR-HS coupling."""

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

UMBRALES_CONGELADOS = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
C_PER_LABEL = {"HS": 2.0, "TR": 1.8, "AG": 1.5}

BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]

BLOQUES_CHAR26 = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (2, 6), True),
]

def aplicar_regla_ascendente(pred: np.ndarray) -> np.ndarray:
    p = pred.copy()
    mask = (p[:, 1] == 1) & (p[:, 0] == 0)
    p[mask, 0] = 1
    return p

def main():
    print("=== EXPLORACION FASE 2: GATING ONTO Y ARQUITECTURA P115 ===")
    train_full = data_module.read_train()

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

    # Guardar probabilidades precomputadas para no recalcular
    # 1. Probar BLOQUES_V2_TRIPLE vs BLOQUES_CHAR26
    for bloque_nombre, bloques in [("V2_triple (char 1-6)", BLOQUES_V2_TRIPLE), ("V2_char26 (char 2-6)", BLOQUES_CHAR26)]:
        print(f"\n==========================================")
        print(f"Probando Representacion: {bloque_nombre}")
        print(f"==========================================")

        resultados_por_particion = {}
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
                for analyzer, ngram, sublinear in bloques:
                    v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
                    mats_tr.append(v.fit_transform(train_dev["text"].iloc[idx_tr]))
                    mats_va.append(v.transform(train_dev["text"].iloc[idx_va]))
                X_tr = sparse.hstack(mats_tr).tocsr()
                X_va = sparse.hstack(mats_va).tocsr()

                for j, et in enumerate(ETIQUETAS):
                    y_col = train_dev[et].iloc[idx_tr]
                    c_val = C_PER_LABEL[et]

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

        # Ahora evaluamos configuraciones sobre ambas particiones
        eps = 1e-6
        for w_saga in [0.10, 0.15, 0.20]:
            w_lr = 0.95 - w_saga
            w_sgd = 0.05
            for T in [1.02, 1.04, 1.06]:
                f1_part_list = []
                for nombre_part, pliegues in particiones:
                    prob_lr, prob_sgd, prob_saga = probs_por_particion[nombre_part]
                    p_blend = w_lr * prob_lr + w_sgd * prob_sgd + w_saga * prob_saga
                    p_cal = expit(logit(np.clip(p_blend, eps, 1 - eps)) / T)

                    p_pred_raw = np.column_stack([(p_cal[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
                    p_pred = aplicar_regla_ascendente(p_pred_raw)
                    m = metricas_multietiqueta(y_true_dev, p_pred)
                    f1_part_list.append(m["f1_media_columnas"])
                mean_f1 = np.mean(f1_part_list)
                print(f"Config w_saga={w_saga:.2f}, T={T:.2f} -> Canonica: {f1_part_list[0]:.6f}, Part47: {f1_part_list[1]:.6f}, Media: {mean_f1:.6f}")

        # Grid search sobre GATING ONTO en ambas particiones conjuntamente
        print("\n--- Optimizacion Conjunta de Reglas de Gating Ontologico ---")
        prob_lr_c, prob_sgd_c, prob_saga_c = probs_por_particion["Canonica"]
        prob_lr_47, prob_sgd_47, prob_saga_47 = probs_por_particion["Part47"]

        p_cal_c = expit(logit(np.clip(0.80 * prob_lr_c + 0.05 * prob_sgd_c + 0.15 * prob_saga_c, eps, 1 - eps)) / 1.04)
        p_cal_47 = expit(logit(np.clip(0.80 * prob_lr_47 + 0.05 * prob_sgd_47 + 0.15 * prob_saga_47, eps, 1 - eps)) / 1.04)

        base_raw_c = np.column_stack([(p_cal_c[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
        base_pred_c = aplicar_regla_ascendente(base_raw_c)
        f1_base_c = metricas_multietiqueta(y_true_dev, base_pred_c)["f1_media_columnas"]

        base_raw_47 = np.column_stack([(p_cal_47[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
        base_pred_47 = aplicar_regla_ascendente(base_raw_47)
        f1_base_47 = metricas_multietiqueta(y_true_dev, base_pred_47)["f1_media_columnas"]
        f1_base_media = (f1_base_c + f1_base_47) / 2

        print(f"P114 Base Media: {f1_base_media:.6f} (Canonica: {f1_base_c:.6f}, Part47: {f1_base_47:.6f})")

        # Explorar:
        # Regla 1: AG Ontological Gating (Si AG=1 predicho y HS_cal >= th_hs_rescue -> HS=1)
        # Regla 2: TR Ontological Gating (Si HS=1 predicho con HS_cal >= th_hs_high y TR_cal >= th_tr_rescue -> TR=1)
        # Regla 3: AG Suppression (Si HS=0 predicho y AG_cal < th_ag_strict -> AG=0)
        mejores_reglas = []
        for th_hs_rescue in [0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44]:
            for th_ag_strict in [0.46, 0.48, 0.50, 0.52]:
                for th_tr_rescue in [0.47, 0.48, 0.49, 0.50, 0.51]:
                    for th_hs_for_tr in [0.60, 0.65, 0.70, 1.0]: # 1.0 means disabled
                        # Evaluar en Canonica
                        p_c = base_pred_c.copy()
                        # Rescate HS por AG
                        if th_hs_rescue < 0.44:
                            mask_rescue = (p_c[:, 0] == 0) & (p_c[:, 2] == 1) & (p_cal_c[:, 0] >= th_hs_rescue)
                            p_c[mask_rescue, 0] = 1
                        # Supresion AG si HS=0
                        if th_ag_strict > 0.46:
                            mask_suppress = (p_c[:, 0] == 0) & (p_cal_c[:, 2] < th_ag_strict)
                            p_c[mask_suppress, 2] = 0
                        # Rescate TR si HS es alto
                        if th_hs_for_tr < 1.0 and th_tr_rescue < 0.51:
                            mask_tr = (p_c[:, 0] == 1) & (p_cal_c[:, 0] >= th_hs_for_tr) & (p_c[:, 1] == 0) & (p_cal_c[:, 1] >= th_tr_rescue)
                            p_c[mask_tr, 1] = 1
                        p_c = aplicar_regla_ascendente(p_c)
                        m_c = metricas_multietiqueta(y_true_dev, p_c)["f1_media_columnas"]

                        # Evaluar en Part47
                        p_47 = base_pred_47.copy()
                        if th_hs_rescue < 0.44:
                            mask_rescue = (p_47[:, 0] == 0) & (p_47[:, 2] == 1) & (p_cal_47[:, 0] >= th_hs_rescue)
                            p_47[mask_rescue, 0] = 1
                        if th_ag_strict > 0.46:
                            mask_suppress = (p_47[:, 0] == 0) & (p_cal_47[:, 2] < th_ag_strict)
                            p_47[mask_suppress, 2] = 0
                        if th_hs_for_tr < 1.0 and th_tr_rescue < 0.51:
                            mask_tr = (p_47[:, 0] == 1) & (p_cal_47[:, 0] >= th_hs_for_tr) & (p_47[:, 1] == 0) & (p_cal_47[:, 1] >= th_tr_rescue)
                            p_47[mask_tr, 1] = 1
                        p_47 = aplicar_regla_ascendente(p_47)
                        m_47 = metricas_multietiqueta(y_true_dev, p_47)["f1_media_columnas"]

                        media_f1 = (m_c + m_47) / 2
                        delta_media = media_f1 - f1_base_media
                        if m_c > f1_base_c and m_47 > f1_base_47: # DEBE GANAR EN AMBAS PARTICIONES
                            mejores_reglas.append((media_f1, delta_media, m_c, m_47, th_hs_rescue, th_ag_strict, th_tr_rescue, th_hs_for_tr))

        mejores_reglas.sort(key=lambda x: x[0], reverse=True)
        print(f"Top 5 reglas que ganan en AMBAS particiones:")
        for r in mejores_reglas[:5]:
            print(f"  Media: {r[0]:.6f} (Delta: {r[1]:+.6f}) | Canonica: {r[2]:.6f}, Part47: {r[3]:.6f}")
            print(f"    Params: hs_rescue={r[4]}, ag_strict={r[5]}, tr_rescue={r[6]}, hs_for_tr={r[7]}")

if __name__ == "__main__":
    main()
