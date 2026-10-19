#!/usr/bin/env python
"""Exploracion 2 para P113: Optimizacion fina del Tri-Blend (LR + SGD log_loss + SGD modified_huber).

Prueba combinaciones de:
  - C uniforme (2.0) vs C adaptado (HS=2.0, TR=1.8, AG=1.5)
  - Proporciones de mezcla entre LR, SGD_log y SGD_huber
  - Verificacion de distribucion de test (celdas distintas vs P110 y recuento de positivos)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
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

def construir_matrices(textos_tr, textos_va, bloques):
    mats_tr, mats_va = [], []
    for analyzer, ngram, sublinear in bloques:
        v = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram,
            min_df=2,
            sublinear_tf=sublinear,
            lowercase=True,
        )
        mats_tr.append(v.fit_transform(textos_tr))
        mats_va.append(v.transform(textos_va))
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

    # Pre-calcular probabilidades OOF para cada componente
    # M1: LR C=2.0
    # M2: LR C_adaptada (HS=2.0, TR=1.8, AG=1.5)
    # M3: SGD log_loss alpha=5e-6
    # M4: SGD modified_huber alpha=5e-6
    # M5: SGD modified_huber alpha=8e-6

    print("Pre-calculando OOF por fold...", flush=True)
    oof_data = {}
    for nombre_part, pliegues in particiones:
        n_samples = len(train_dev)
        p_lr_c2 = np.zeros((n_samples, len(ETIQUETAS)), dtype=float)
        p_lr_cadapt = np.zeros((n_samples, len(ETIQUETAS)), dtype=float)
        p_sgd_log = np.zeros((n_samples, len(ETIQUETAS)), dtype=float)
        p_sgd_huber5 = np.zeros((n_samples, len(ETIQUETAS)), dtype=float)
        p_sgd_huber8 = np.zeros((n_samples, len(ETIQUETAS)), dtype=float)

        c_adapt = {"HS": 2.0, "TR": 1.8, "AG": 1.5}

        for fold in sorted(set(pliegues)):
            idx_tr = np.flatnonzero(pliegues != fold)
            idx_va = np.flatnonzero(pliegues == fold)

            X_tr, X_va = construir_matrices(
                textos_dev.iloc[idx_tr], textos_dev.iloc[idx_va], BLOQUES_V2_TRIPLE
            )

            for j, et in enumerate(ETIQUETAS):
                y_tr_j = y_dev[idx_tr, j]

                # LR C=2.0
                lr2 = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
                lr2.fit(X_tr, y_tr_j)
                p_lr_c2[idx_va, j] = lr2.predict_proba(X_va)[:, 1]

                # LR C adapt
                lra = LogisticRegression(C=c_adapt[et], class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
                lra.fit(X_tr, y_tr_j)
                p_lr_cadapt[idx_va, j] = lra.predict_proba(X_va)[:, 1]

                # SGD log
                sgd_l = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
                sgd_l.fit(X_tr, y_tr_j)
                p_sgd_log[idx_va, j] = sgd_l.predict_proba(X_va)[:, 1]

                # SGD huber 5e-6
                sgd_h5 = SGDClassifier(loss="modified_huber", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
                sgd_h5.fit(X_tr, y_tr_j)
                p_sgd_huber5[idx_va, j] = sgd_h5.predict_proba(X_va)[:, 1]

                # SGD huber 8e-6
                sgd_h8 = SGDClassifier(loss="modified_huber", penalty="l2", alpha=8e-6, max_iter=2000, random_state=42, class_weight="balanced")
                sgd_h8.fit(X_tr, y_tr_j)
                p_sgd_huber8[idx_va, j] = sgd_h8.predict_proba(X_va)[:, 1]

        oof_data[nombre_part] = {
            "lr_c2": p_lr_c2,
            "lr_cadapt": p_lr_cadapt,
            "sgd_log": p_sgd_log,
            "sgd_huber5": p_sgd_huber5,
            "sgd_huber8": p_sgd_huber8,
        }

    # Ahora probamos rejilla de configuraciones instantaneamente
    experimentos = [
        # (nombre, lr_key, w_lr, w_log, w_huber, huber_key)
        ("P110_Base", "lr_c2", 0.95, 0.05, 0.0, "sgd_huber5"),
        ("P111_Base", "lr_cadapt", 0.95, 0.05, 0.0, "sgd_huber5"),
        ("TriBlend_C2_90_5_5", "lr_c2", 0.90, 0.05, 0.05, "sgd_huber5"),
        ("TriBlend_C2_92_4_4", "lr_c2", 0.92, 0.04, 0.04, "sgd_huber5"),
        ("TriBlend_C2_88_6_6", "lr_c2", 0.88, 0.06, 0.06, "sgd_huber5"),
        ("TriBlend_C2_90_6_4", "lr_c2", 0.90, 0.06, 0.04, "sgd_huber5"),
        ("TriBlend_C2_90_4_6", "lr_c2", 0.90, 0.04, 0.06, "sgd_huber5"),
        ("TriBlend_C2_90_5_5_h8", "lr_c2", 0.90, 0.05, 0.05, "sgd_huber8"),
        ("TriBlend_CAdapt_90_5_5", "lr_cadapt", 0.90, 0.05, 0.05, "sgd_huber5"),
        ("TriBlend_CAdapt_92_4_4", "lr_cadapt", 0.92, 0.04, 0.04, "sgd_huber5"),
        ("TriBlend_CAdapt_88_6_6", "lr_cadapt", 0.88, 0.06, 0.06, "sgd_huber5"),
        ("TriBlend_CAdapt_90_6_4", "lr_cadapt", 0.90, 0.06, 0.04, "sgd_huber5"),
        ("TriBlend_CAdapt_90_4_6", "lr_cadapt", 0.90, 0.04, 0.06, "sgd_huber5"),
        ("TriBlend_CAdapt_90_5_5_h8", "lr_cadapt", 0.90, 0.05, 0.05, "sgd_huber8"),
        ("TriBlend_CAdapt_91_5_4", "lr_cadapt", 0.91, 0.05, 0.04, "sgd_huber5"),
        ("TriBlend_CAdapt_93_4_3", "lr_cadapt", 0.93, 0.04, 0.03, "sgd_huber5"),
    ]

    print("\nResultados de evaluacion multietiqueta:", flush=True)
    tabla = []
    for exp_nom, lr_k, w_lr, w_log, w_hub, hub_k in experimentos:
        scores = {}
        for nombre_part, _ in particiones:
            d = oof_data[nombre_part]
            p_ens = w_lr * d[lr_k] + w_log * d["sgd_log"] + w_hub * d[hub_k]
            pred_raw = np.column_stack([(p_ens[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
            pred_asc = aplicar_regla_ascendente(pred_raw)
            met = metricas_multietiqueta(y_dev, pred_asc)
            scores[nombre_part] = met["f1_media_columnas"]

        can = scores["Canonica"]
        p47 = scores["Particion 47"]
        media = (can + p47) / 2.0
        delta_p110 = media - 0.784658
        delta_p111 = media - 0.786390
        tabla.append((exp_nom, can, p47, media, delta_p110, delta_p111))
        print(f"{exp_nom:28s} | Can: {can:.6f} | P47: {p47:.6f} | Media: {media:.6f} | d(P110): {delta_p110:+.6f} | d(P111): {delta_p111:+.6f}", flush=True)

    # Ordenar por media
    tabla.sort(key=lambda x: x[3], reverse=True)
    print("\n=== TOP 5 MODELOS POR CV MEDIA ===")
    for row in tabla[:5]:
        print(f"1. {row[0]}: Media={row[3]:.6f} (Can={row[1]:.6f}, P47={row[2]:.6f}, Delta P110={row[4]:+.6f})")

    # Para el mejor modelo, calcular predicciones sobre test set (1.731 filas completas)
    best = tabla[0]
    best_name = best[0]
    print(f"\nEntrenando mejor modelo ({best_name}) sobre 100% de train (1.731 filas) para test...", flush=True)

    # Identificar params del mejor
    for exp_nom, lr_k, w_lr, w_log, w_hub, hub_k in experimentos:
        if exp_nom == best_name:
            best_params = (lr_k, w_lr, w_log, w_hub, hub_k)
            break

    lr_k, w_lr, w_log, w_hub, hub_k = best_params
    alpha_hub = 8e-6 if hub_k == "sgd_huber8" else 5e-6
    c_dict = {"HS": 2.0, "TR": 1.8, "AG": 1.5} if lr_k == "lr_cadapt" else {"HS": 2.0, "TR": 2.0, "AG": 2.0}

    # Ajustar sobre 1.731 filas
    mats_tr, mats_te = [], []
    for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
        v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
        mats_tr.append(v.fit_transform(train_full["text"]))
        mats_te.append(v.transform(test_df["text"]))
    X_tr_full = sparse.hstack(mats_tr).tocsr()
    X_te = sparse.hstack(mats_te).tocsr()

    p_lr_te = np.zeros((len(test_df), len(ETIQUETAS)), dtype=float)
    p_sgd_log_te = np.zeros((len(test_df), len(ETIQUETAS)), dtype=float)
    p_sgd_hub_te = np.zeros((len(test_df), len(ETIQUETAS)), dtype=float)

    for j, et in enumerate(ETIQUETAS):
        # LR
        lr_m = LogisticRegression(C=c_dict[et], class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        lr_m.fit(X_tr_full, train_full[et])
        p_lr_te[:, j] = lr_m.predict_proba(X_te)[:, 1]

        # SGD log
        sgd_l_m = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
        sgd_l_m.fit(X_tr_full, train_full[et])
        p_sgd_log_te[:, j] = sgd_l_m.predict_proba(X_te)[:, 1]

        # SGD huber
        sgd_h_m = SGDClassifier(loss="modified_huber", penalty="l2", alpha=alpha_hub, max_iter=2000, random_state=42, class_weight="balanced")
        sgd_h_m.fit(X_tr_full, train_full[et])
        p_sgd_hub_te[:, j] = sgd_h_m.predict_proba(X_te)[:, 1]

    p_best_te = w_lr * p_lr_te + w_log * p_sgd_log_te + w_hub * p_sgd_hub_te
    pred_best_raw = np.column_stack([(p_best_te[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    pred_best_asc = aplicar_regla_ascendente(pred_best_raw)

    diff_cells = int((pred_best_asc != pred_p110_test).sum())
    total_pos = int(pred_best_asc.sum())
    pos_por_et = {et: int(pred_best_asc[:, j].sum()) for j, et in enumerate(ETIQUETAS)}

    print(f"\n--- Comportamiento de {best_name} en Test Set (1.154 filas) ---")
    print(f"Celdas distintas vs P110: {diff_cells} de 3462 ({diff_cells/3462*100:.2f}%)")
    print(f"Total positivos predichos: {total_pos} ({total_pos/3462*100:.2f}%)")
    print(f"Positivos por etiqueta: HS={pos_por_et['HS']}, TR={pos_por_et['TR']}, AG={pos_por_et['AG']}")
    print(f"P110 referencia positivos: HS=510, TR=329, AG=418 | Total=1257")

if __name__ == "__main__":
    main()
