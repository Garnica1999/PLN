#!/usr/bin/env python
"""Investigacion profunda para P113:
1. Correlacion entre errores de P110 y tweets de frontera [0.40, 0.55].
2. Impacto de la perdida de margen (SGD modified_huber) en el Tri-Blend:
   - Rescates vs Danios
   - Celdas y tweets modificados
   - Casos limite cualitativos beneficiados y perjudicados
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
    print("=== CARGANDO DATOS ===", flush=True)
    train_full = data_module.read_train()
    test_df = data_module.read_test()

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

    # Pre-calcular OOF para:
    # 1. LR C=2.0
    # 2. SGD log_loss alpha=5e-6
    # 3. SGD modified_huber alpha=5e-6
    # 4. SGD modified_huber alpha=8e-6
    # 5. LR C adaptada (HS=2.0, TR=1.8, AG=1.5)
    print("\n=== CALCULANDO OOF PARA MODELOS BASE ===", flush=True)
    oof_preds = {}
    for nombre_part, pliegues in particiones:
        print(f"Calculando {nombre_part}...", flush=True)
        n_samples = len(train_dev)
        p_lr2 = np.zeros((n_samples, 3), dtype=float)
        p_lra = np.zeros((n_samples, 3), dtype=float)
        p_sgd_log = np.zeros((n_samples, 3), dtype=float)
        p_sgd_hub5 = np.zeros((n_samples, 3), dtype=float)
        p_sgd_hub8 = np.zeros((n_samples, 3), dtype=float)

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
                p_lr2[idx_va, j] = lr2.predict_proba(X_va)[:, 1]

                # LR C adapt
                lra = LogisticRegression(C=c_adapt[et], class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
                lra.fit(X_tr, y_tr_j)
                p_lra[idx_va, j] = lra.predict_proba(X_va)[:, 1]

                # SGD log
                sgd_l = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
                sgd_l.fit(X_tr, y_tr_j)
                p_sgd_log[idx_va, j] = sgd_l.predict_proba(X_va)[:, 1]

                # SGD huber 5e-6
                sgd_h5 = SGDClassifier(loss="modified_huber", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
                sgd_h5.fit(X_tr, y_tr_j)
                p_sgd_hub5[idx_va, j] = sgd_h5.predict_proba(X_va)[:, 1]

                # SGD huber 8e-6
                sgd_h8 = SGDClassifier(loss="modified_huber", penalty="l2", alpha=8e-6, max_iter=2000, random_state=42, class_weight="balanced")
                sgd_h8.fit(X_tr, y_tr_j)
                p_sgd_hub8[idx_va, j] = sgd_h8.predict_proba(X_va)[:, 1]

        oof_preds[nombre_part] = {
            "p_lr2": p_lr2,
            "p_lra": p_lra,
            "p_sgd_log": p_sgd_log,
            "p_sgd_hub5": p_sgd_hub5,
            "p_sgd_hub8": p_sgd_hub8,
        }

    # Ahora analizamos P110 en Particion Canonica y Particion 47
    print("\n=======================================================", flush=True)
    print("PARTE 1: ANALISIS DE PROBABILIDADES Y ERRORES DE P110", flush=True)
    print("=======================================================", flush=True)

    # P110 es 0.95 * LR(C=2.0) + 0.05 * SGD(log_loss, 5e-6)
    p_p110_can = 0.95 * oof_preds["Canonica"]["p_lr2"] + 0.05 * oof_preds["Canonica"]["p_sgd_log"]
    pred_p110_raw_can = np.column_stack([(p_p110_can[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    pred_p110_can = aplicar_regla_ascendente(pred_p110_raw_can)

    # Analisis por bins de probabilidad en P110 Canonica
    bins = [0.0, 0.20, 0.40, 0.55, 0.80, 1.0001]
    bin_labels = ["[0.00, 0.20)", "[0.20, 0.40)", "[0.40, 0.55]", "(0.55, 0.80]", "(0.80, 1.00]"]

    for j, et in enumerate(ETIQUETAS):
        print(f"\n--- Analisis de Distribucion y Error P110 para {et} (Umbral = {UMBRALES_CONGELADOS[et]}) ---")
        p_et = p_p110_can[:, j]
        y_et = y_dev[:, j]
        pred_et = pred_p110_can[:, j]
        es_error = (pred_et != y_et)
        es_fp = (pred_et == 1) & (y_et == 0)
        es_fn = (pred_et == 0) & (y_et == 1)

        total_muestras = len(y_et)
        total_errores = int(es_error.sum())
        total_positivos_reales = int(y_et.sum())

        print(f"Total muestras: {total_muestras} | Positivos reales: {total_positivos_reales} | Errores totales: {total_errores} (Tasa error global: {total_errores/total_muestras*100:.2f}%)")

        # Desglose por bin
        print(f"{'Intervalo':16s} | {'N':>5s} | {'% N':>6s} | {'Errores':>7s} | {'FP':>4s} | {'FN':>4s} | {'Tasa Error':>10s} | {'% del Error Total':>18s}")
        print("-" * 85)
        for i in range(len(bins)-1):
            low, high = bins[i], bins[i+1]
            if i == 2: # [0.40, 0.55]
                mask = (p_et >= low) & (p_et <= high)
            elif i > 2:
                mask = (p_et > low) & (p_et <= high)
            else:
                mask = (p_et >= low) & (p_et < high)

            n_bin = int(mask.sum())
            err_bin = int(es_error[mask].sum())
            fp_bin = int(es_fp[mask].sum())
            fn_bin = int(es_fn[mask].sum())
            tasa_err = err_bin / n_bin if n_bin > 0 else 0.0
            pct_del_total_err = err_bin / total_errores * 100 if total_errores > 0 else 0.0

            print(f"{bin_labels[i]:16s} | {n_bin:5d} | {n_bin/total_muestras*100:5.1f}% | {err_bin:7d} | {fp_bin:4d} | {fn_bin:4d} | {tasa_err*100:9.2f}% | {pct_del_total_err:17.2f}%")

    # Resumen agregado multietiqueta (3 x 1483 = 4449 decisiones)
    print("\n--- RESUMEN AGREGADO DE TODAS LAS CELDAS (3 x 1.483 = 4.449) ---")
    p_all = p_p110_can.ravel()
    y_all = y_dev.ravel()
    pred_all = pred_p110_can.ravel()
    err_all = (pred_all != y_all)
    fp_all = (pred_all == 1) & (y_all == 0)
    fn_all = (pred_all == 0) & (y_all == 1)

    print(f"Total decisiones: {len(p_all)} | Total errores P110: {err_all.sum()} (Tasa global: {err_all.sum()/len(p_all)*100:.2f}%)")
    print(f"{'Intervalo':16s} | {'N':>5s} | {'% N':>6s} | {'Errores':>7s} | {'FP':>4s} | {'FN':>4s} | {'Tasa Error':>10s} | {'% del Error Total':>18s}")
    print("-" * 85)
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        if i == 2:
            mask = (p_all >= low) & (p_all <= high)
        elif i > 2:
            mask = (p_all > low) & (p_all <= high)
        else:
            mask = (p_all >= low) & (p_all < high)

        n_bin = int(mask.sum())
        err_bin = int(err_all[mask].sum())
        fp_bin = int(fp_all[mask].sum())
        fn_bin = int(fn_all[mask].sum())
        tasa_err = err_bin / n_bin if n_bin > 0 else 0.0
        pct_del_total_err = err_bin / err_all.sum() * 100

        print(f"{bin_labels[i]:16s} | {n_bin:5d} | {n_bin/len(p_all)*100:5.1f}% | {err_bin:7d} | {fp_bin:4d} | {fn_bin:4d} | {tasa_err*100:9.2f}% | {pct_del_total_err:17.2f}%")

    print("\n=======================================================", flush=True)
    print("PARTE 2: IMPACTO DE SGD MODIFIED_HUBER (PERDIDA DE MARGEN)", flush=True)
    print("=======================================================", flush=True)

    # Comparamos P110 vs Tri-Blend:
    # Variante A: Tri-Blend C=2.0 (90% LR C=2, 5% SGD_log, 5% SGD_huber5)
    # Variante B: Tri-Blend C_adapt (90% LR Cadapt, 5% SGD_log, 5% SGD_huber5)
    # Variante C: Tri-Blend C_adapt (92% LR Cadapt, 4% SGD_log, 4% SGD_huber5)
    # Variante D: Tri-Blend C=2.0 (92% LR C=2, 4% SGD_log, 4% SGD_huber5)

    tri_variantes = [
        ("TriBlend_C2_90_5_5", "p_lr2", 0.90, 0.05, 0.05, "p_sgd_hub5"),
        ("TriBlend_C2_92_4_4", "p_lr2", 0.92, 0.04, 0.04, "p_sgd_hub5"),
        ("TriBlend_CAdapt_90_5_5", "p_lra", 0.90, 0.05, 0.05, "p_sgd_hub5"),
        ("TriBlend_CAdapt_92_4_4", "p_lra", 0.92, 0.04, 0.04, "p_sgd_hub5"),
        ("TriBlend_CAdapt_91_5_4", "p_lra", 0.91, 0.05, 0.04, "p_sgd_hub5"),
    ]

    for nom_tri, lr_k, w_lr, w_log, w_hub, hub_k in tri_variantes:
        print(f"\n--- Analisis Detallado: {nom_tri} vs P110 (Canonica) ---")
        p_tri_can = w_lr * oof_preds["Canonica"][lr_k] + w_log * oof_preds["Canonica"]["p_sgd_log"] + w_hub * oof_preds["Canonica"][hub_k]
        pred_tri_raw = np.column_stack([(p_tri_can[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
        pred_tri = aplicar_regla_ascendente(pred_tri_raw)

        met_p110 = metricas_multietiqueta(y_dev, pred_p110_can)
        met_tri = metricas_multietiqueta(y_dev, pred_tri)

        print(f"F1 P110: {met_p110['f1_media_columnas']:.6f} (HS={met_p110['f1_por_etiqueta']['HS']:.4f}, TR={met_p110['f1_por_etiqueta']['TR']:.4f}, AG={met_p110['f1_por_etiqueta']['AG']:.4f})")
        print(f"F1 Tri : {met_tri['f1_media_columnas']:.6f} (HS={met_tri['f1_por_etiqueta']['HS']:.4f}, TR={met_tri['f1_por_etiqueta']['TR']:.4f}, AG={met_tri['f1_por_etiqueta']['AG']:.4f}) | Delta={met_tri['f1_media_columnas']-met_p110['f1_media_columnas']:+.6f}")

        # Rescates y Danios celda a celda
        # Celda i, j:
        # Rescate: P110 estaba mal, Tri esta bien
        # Danio: P110 estaba bien, Tri esta mal
        diff_mask = (pred_p110_can != pred_tri)
        rescates_mask = (pred_p110_can != y_dev) & (pred_tri == y_dev)
        danios_mask = (pred_p110_can == y_dev) & (pred_tri != y_dev)

        n_diff = int(diff_mask.sum())
        n_rescates = int(rescates_mask.sum())
        n_danios = int(danios_mask.sum())
        neto = n_rescates - n_danios

        print(f"Celdas modificadas: {n_diff} | Rescates: {n_rescates} | Danios: {n_danios} | Neto: {neto:+d}")

        for j, et in enumerate(ETIQUETAS):
            r_et = int(rescates_mask[:, j].sum())
            d_et = int(danios_mask[:, j].sum())
            diff_et = int(diff_mask[:, j].sum())
            print(f"  {et}: Modificadas={diff_et} | Rescates={r_et} | Danios={d_et} | Neto={r_et - d_et:+d}")

        # Ver donde caen las probabilidades de P110 en las celdas modificadas
        p_p110_mod = p_p110_can[diff_mask]
        if len(p_p110_mod) > 0:
            print(f"  Probabilidades P110 en celdas modificadas: min={p_p110_mod.min():.4f}, median={np.median(p_p110_mod):.4f}, max={p_p110_mod.max():.4f}")
            en_frontera = ((p_p110_mod >= 0.40) & (p_p110_mod <= 0.55)).sum()
            print(f"  Modificadas dentro de [0.40, 0.55]: {en_frontera} de {len(p_p110_mod)} ({en_frontera/len(p_p110_mod)*100:.1f}%)")

    # Muestreo cualitativo profundo para TriBlend_CAdapt_90_5_5 y TriBlend_C2_90_5_5
    print("\n=======================================================", flush=True)
    print("PARTE 3: CASOS CUALITATIVOS (RESCATES Y DANIOS DE MARGEN)", flush=True)
    print("=======================================================", flush=True)

    p_tri_c2 = 0.90 * oof_preds["Canonica"]["p_lr2"] + 0.05 * oof_preds["Canonica"]["p_sgd_log"] + 0.05 * oof_preds["Canonica"]["p_sgd_hub5"]
    pred_tri_c2 = aplicar_regla_ascendente(np.column_stack([(p_tri_c2[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))

    rescates_c2 = (pred_p110_can != y_dev) & (pred_tri_c2 == y_dev)
    danios_c2 = (pred_p110_can == y_dev) & (pred_tri_c2 != y_dev)

    print("\n--- CASOS RESCATADOS POR LA PERDIDA DE MARGEN (TriBlend C2) ---")
    idx_rescates = np.where(rescates_c2)
    for row_idx, col_idx in zip(idx_rescates[0], idx_rescates[1]):
        et = ETIQUETAS[col_idx]
        tweet_id = train_dev.iloc[row_idx]["id"]
        txt = train_dev.iloc[row_idx]["text"]
        y_val = y_dev[row_idx, col_idx]
        p_p110_val = p_p110_can[row_idx, col_idx]
        p_tri_val = p_tri_c2[row_idx, col_idx]
        p_sgd_hub_val = oof_preds["Canonica"]["p_sgd_hub5"][row_idx, col_idx]
        p_sgd_log_val = oof_preds["Canonica"]["p_sgd_log"][row_idx, col_idx]
        p_lr_val = oof_preds["Canonica"]["p_lr2"][row_idx, col_idx]
        th = UMBRALES_CONGELADOS[et]
        print(f"ID={tweet_id} | Etiqueta={et} | True={y_val} | Umbral={th}")
        print(f"  P110={p_p110_val:.4f} (Pred={int(p_p110_val>=th)}) -> Tri={p_tri_val:.4f} (Pred={int(p_tri_val>=th)})")
        print(f"  Detalle: LR={p_lr_val:.4f}, SGD_log={p_sgd_log_val:.4f}, SGD_huber={p_sgd_hub_val:.4f}")
        print(f"  Texto: {txt}")
        print("-" * 60)

    print("\n--- CASOS DANADOS POR LA PERDIDA DE MARGEN (TriBlend C2) ---")
    idx_danios = np.where(danios_c2)
    for row_idx, col_idx in zip(idx_danios[0], idx_danios[1]):
        et = ETIQUETAS[col_idx]
        tweet_id = train_dev.iloc[row_idx]["id"]
        txt = train_dev.iloc[row_idx]["text"]
        y_val = y_dev[row_idx, col_idx]
        p_p110_val = p_p110_can[row_idx, col_idx]
        p_tri_val = p_tri_c2[row_idx, col_idx]
        p_sgd_hub_val = oof_preds["Canonica"]["p_sgd_hub5"][row_idx, col_idx]
        p_sgd_log_val = oof_preds["Canonica"]["p_sgd_log"][row_idx, col_idx]
        p_lr_val = oof_preds["Canonica"]["p_lr2"][row_idx, col_idx]
        th = UMBRALES_CONGELADOS[et]
        print(f"ID={tweet_id} | Etiqueta={et} | True={y_val} | Umbral={th}")
        print(f"  P110={p_p110_val:.4f} (Pred={int(p_p110_val>=th)}) -> Tri={p_tri_val:.4f} (Pred={int(p_tri_val>=th)})")
        print(f"  Detalle: LR={p_lr_val:.4f}, SGD_log={p_sgd_log_val:.4f}, SGD_huber={p_sgd_hub_val:.4f}")
        print(f"  Texto: {txt}")
        print("-" * 60)

    # Evaluacion en Test Set para TriBlend_C2_90_5_5 y TriBlend_CAdapt_90_5_5
    print("\n=======================================================", flush=True)
    print("PARTE 4: EVALUACION EN TEST SET (100% TRAIN FIT)", flush=True)
    print("=======================================================", flush=True)

    sub_p110_df = pd.read_csv(SUB_P110)
    pred_p110_te = sub_p110_df[list(ETIQUETAS)].to_numpy(dtype=int)

    # Entrenar componentes al 100% sobre train_full (1.731 filas)
    mats_tr, mats_te = [], []
    for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
        v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
        mats_tr.append(v.fit_transform(train_full["text"]))
        mats_te.append(v.transform(test_df["text"]))
    X_tr_full = sparse.hstack(mats_tr).tocsr()
    X_te = sparse.hstack(mats_te).tocsr()

    p_lr2_te = np.zeros((len(test_df), 3), dtype=float)
    p_lra_te = np.zeros((len(test_df), 3), dtype=float)
    p_sgd_log_te = np.zeros((len(test_df), 3), dtype=float)
    p_sgd_hub_te = np.zeros((len(test_df), 3), dtype=float)

    c_adapt = {"HS": 2.0, "TR": 1.8, "AG": 1.5}
    for j, et in enumerate(ETIQUETAS):
        # LR C=2
        lr2 = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        lr2.fit(X_tr_full, train_full[et])
        p_lr2_te[:, j] = lr2.predict_proba(X_te)[:, 1]

        # LR Cadapt
        lra = LogisticRegression(C=c_adapt[et], class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        lra.fit(X_tr_full, train_full[et])
        p_lra_te[:, j] = lra.predict_proba(X_te)[:, 1]

        # SGD log
        sgd_l = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
        sgd_l.fit(X_tr_full, train_full[et])
        p_sgd_log_te[:, j] = sgd_l.predict_proba(X_te)[:, 1]

        # SGD huber
        sgd_h = SGDClassifier(loss="modified_huber", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
        sgd_h.fit(X_tr_full, train_full[et])
        p_sgd_hub_te[:, j] = sgd_h.predict_proba(X_te)[:, 1]

    # Modelos en Test
    p_p110_full_te = 0.95 * p_lr2_te + 0.05 * p_sgd_log_te
    p_tri_c2_te = 0.90 * p_lr2_te + 0.05 * p_sgd_log_te + 0.05 * p_sgd_hub_te
    p_tri_cadapt_te = 0.90 * p_lra_te + 0.05 * p_sgd_log_te + 0.05 * p_sgd_hub_te

    pred_tri_c2_te = aplicar_regla_ascendente(np.column_stack([(p_tri_c2_te[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))
    pred_tri_cadapt_te = aplicar_regla_ascendente(np.column_stack([(p_tri_cadapt_te[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)]))

    diff_c2_te = int((pred_tri_c2_te != pred_p110_te).sum())
    diff_cadapt_te = int((pred_tri_cadapt_te != pred_p110_te).sum())

    print(f"P110 Oficial Test Positivos: HS={pred_p110_te[:, 0].sum()}, TR={pred_p110_te[:, 1].sum()}, AG={pred_p110_te[:, 2].sum()} | Total={pred_p110_te.sum()} (36.37%)")
    print(f"TriBlend C2 Test Positivos: HS={pred_tri_c2_te[:, 0].sum()}, TR={pred_tri_c2_te[:, 1].sum()}, AG={pred_tri_c2_te[:, 2].sum()} | Total={pred_tri_c2_te.sum()} ({pred_tri_c2_te.sum()/3462*100:.2f}%) | Diff={diff_c2_te}")
    print(f"TriBlend CAdapt Test Positivos: HS={pred_tri_cadapt_te[:, 0].sum()}, TR={pred_tri_cadapt_te[:, 1].sum()}, AG={pred_tri_cadapt_te[:, 2].sum()} | Total={pred_tri_cadapt_te.sum()} ({pred_tri_cadapt_te.sum()/3462*100:.2f}%) | Diff={diff_cadapt_te}")

    # Celdas que cambian en Test
    print("\n--- Tweets de Test que cambian entre P110 y TriBlend C2 ---")
    diff_mask_te = (pred_tri_c2_te != pred_p110_te)
    idx_diff_te = np.where(diff_mask_te)
    for row_idx, col_idx in zip(idx_diff_te[0], idx_diff_te[1]):
        et = ETIQUETAS[col_idx]
        t_id = test_df.iloc[row_idx]["id"]
        txt = test_df.iloc[row_idx]["text"]
        p110_pred = pred_p110_te[row_idx, col_idx]
        tri_pred = pred_tri_c2_te[row_idx, col_idx]
        p_110_val = p_p110_full_te[row_idx, col_idx]
        p_tri_val = p_tri_c2_te[row_idx, col_idx]
        p_hub_val = p_sgd_hub_te[row_idx, col_idx]
        th = UMBRALES_CONGELADOS[et]
        print(f"ID={t_id} | Et={et} | Umbral={th} | P110={p_110_val:.4f} (pred {p110_pred}) -> Tri={p_tri_val:.4f} (pred {tri_pred}) | Huber={p_hub_val:.4f}")
        print(f"Texto: {txt}")
        print("-" * 60)

if __name__ == "__main__":
    main()
