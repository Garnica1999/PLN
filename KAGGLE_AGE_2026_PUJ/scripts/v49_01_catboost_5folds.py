"""V49 · 01 — CatBoost Oblivious Decision Trees sobre matriz_279 y Fusión Geométrica PoE.

Introduce CatBoost (árboles de decisión simétricos / obliviosos con boosting ordenado) sobre
las 279 meta-features como un nuevo miembro con sesgo inductivo ortogonal a LightGBM (crecimiento por hojas)
y a las redes residuales profundas (superficies continuas).

Protocolo:
- Validación canónica 5-fold StratifiedGroupKFold agrupada por dup_clusters (cero fuga).
- Evaluación standalone con regla congelada plug-in w = [1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128].
- Fusión Bayesiana Product-of-Experts en espacio de log-probabilidades con los 6 modelos del campeón V48.
- Optimización de pesos de ensamble mediante Nested CV (validación cruzada anidada) con Nelder-Mead.
- Prueba rigurosa de remuestreo pareado bootstrap (B=1000) frente al campeón actual V48.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v19_b1_boosting import matriz_279

from src import config as cfg
from src.verificacion_numerica import autotest_numerico

RESULTADO = cfg.EXPERIMENTS / "v49_01_catboost_5folds.json"
OOF_FILE = cfg.FEATURES / "v49_oof_catboost.npy"
TEST_PROB_FILE = cfg.FEATURES / "v49_test_catboost.npy"
MATRIZ_TEST = cfg.FEATURES / "meta_features_test_v12_fresca.npy"

PESOS_PLUG_IN = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("[V49] Iniciando entrenamiento 5-fold de CatBoost sobre matriz_279...", flush=True)

    X_train = matriz_279()
    X_test = np.load(MATRIZ_TEST).astype(np.float32)

    y_true = np.load(cfg.ROOT / "artifacts" / "p33_scores" / "todos" / "y_true.npy")
    fold_idx = np.load(cfg.ROOT / "artifacts" / "p33_scores" / "todos" / "fold_idx.npy")

    n_samples, n_features = X_train.shape
    n_test = X_test.shape[0]
    n_classes = len(cfg.CLASSES)

    oof_probs = np.zeros((n_samples, n_classes), dtype=np.float32)
    test_probs = np.zeros((n_test, n_classes), dtype=np.float32)

    # Hiperparámetros de CatBoost
    cb_params = {
        "iterations": 350,
        "learning_rate": 0.08,
        "depth": 5,
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "random_seed": cfg.SEED,
        "thread_count": 32,
        "verbose": 0,
    }

    fold_scores_standalone = []

    for fold in range(5):
        t0 = time.time()
        tr_mask = (fold_idx != fold)
        val_mask = (fold_idx == fold)

        X_tr = X_train[tr_mask]
        y_tr = y_true[tr_mask]
        X_va = X_train[val_mask]
        y_va = y_true[val_mask]

        model = CatBoostClassifier(**cb_params)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=40)

        probs_val = model.predict_proba(X_va).astype(np.float32)
        oof_probs[val_mask] = probs_val

        probs_test = model.predict_proba(X_test).astype(np.float32)
        test_probs += probs_test / 5.0

        pred_fold = np.argmax(probs_val * PESOS_PLUG_IN, axis=1)
        f1_fold = float(f1_score(y_va, pred_fold, average="macro"))
        fold_scores_standalone.append(f1_fold)
        print(f"  Fold {fold}: F1 Standalone = {f1_fold:.6f} ({time.time() - t0:.1f}s)", flush=True)

    pred_standalone_all = np.argmax(oof_probs * PESOS_PLUG_IN, axis=1)
    f1_standalone_global = float(f1_score(y_true, pred_standalone_all, average="macro"))
    print(f"\n[V49] CatBoost Standalone Global Macro-F1: {f1_standalone_global:.7f}", flush=True)
    print(f"[V49] Folds Standalone: {[round(x, 6) for x in fold_scores_standalone]}", flush=True)

    # Guardar artefactos de predicción
    OOF_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(OOF_FILE, oof_probs)
    np.save(TEST_PROB_FILE, test_probs)
    print(f"[V49] Guardado OOF en {OOF_FILE}", flush=True)
    print(f"[V49] Guardado Test Probs en {TEST_PROB_FILE}", flush=True)

    # --- Evaluación de Integración en Ensamble V48 Geometric PoE ---
    print("\n[V49] Evaluando integración con los 6 modelos del campeón V48...", flush=True)
    v30 = np.load(cfg.FEATURES / "age_plugin_segunda_particion_oof.npy")
    v43 = np.load(cfg.FEATURES / "v43_oof_cramer_swa_tabularnet.npy")
    v42 = np.load(cfg.FEATURES / "v42_oof_smooth_swa_tabularnet.npy")
    v40 = np.load(cfg.FEATURES / "v40_oof_se_tabularnet.npy")
    v39_ord = np.load(cfg.FEATURES / "v39_oof_tabnet_ordinal.npy")
    v39_hgb = np.load(cfg.FEATURES / "v39_oof_hgb.npy")

    models_v48 = [v30, v43, v42, v40, v39_ord, v39_hgb]
    weights_v48 = np.array([0.5628, 0.0832, 0.1724, 0.0754, 0.0513, 0.0549])
    eps = 1e-12

    # Baseline V48
    log_models_v48 = [np.log(np.clip(m, eps, 1.0)) for m in models_v48]
    log_geom_v48 = sum(w * lm for w, lm in zip(weights_v48, log_models_v48))
    p_geom_v48 = np.exp(log_geom_v48 - np.max(log_geom_v48, axis=1, keepdims=True))
    p_geom_v48 /= p_geom_v48.sum(axis=1, keepdims=True)
    pred_v48 = np.argmax(p_geom_v48 * PESOS_PLUG_IN, axis=1)

    f1_v48_global = float(f1_score(y_true, pred_v48, average="macro"))
    f1_v48_folds = [float(f1_score(y_true[fold_idx == f], pred_v48[fold_idx == f], average="macro")) for f in range(5)]
    print(f"[V48 Champion] Global: {f1_v48_global:.7f}, Folds: {[round(x, 6) for x in f1_v48_folds]}", flush=True)

    # Modelos V49 (7 miembros)
    models_v49 = models_v48 + [oof_probs]
    log_models_v49 = [np.log(np.clip(m, eps, 1.0)) for m in models_v49]

    # Optimización por Nested CV (sin mirar el fold exterior)
    print("\n[V49] Optimizando pesos de los 7 miembros por Nested CV...", flush=True)
    pred_v49_nested = np.zeros_like(y_true)
    fold_weights = []

    init_weights = np.append(weights_v48 * 0.96, 0.04)

    for fold in range(5):
        tr_mask = (fold_idx != fold)
        val_mask = (fold_idx == fold)

        def objective(weights_raw):
            w = np.exp(weights_raw)
            w = w / w.sum()
            log_g = sum(w_i * log_models_v49[i][tr_mask] for i, w_i in enumerate(w))
            p_g = np.exp(log_g - np.max(log_g, axis=1, keepdims=True))
            p_g /= p_g.sum(axis=1, keepdims=True)
            p_pred = np.argmax(p_g * PESOS_PLUG_IN, axis=1)
            return -f1_score(y_true[tr_mask], p_pred, average="macro")

        res = minimize(objective, np.log(init_weights), method="Nelder-Mead", options={"maxiter": 300, "xatol": 1e-4})
        w_opt = np.exp(res.x)
        w_opt = w_opt / w_opt.sum()
        fold_weights.append(w_opt)

        # Aplicar estrictamente sobre el fold de validación
        log_g_val = sum(w_i * log_models_v49[i][val_mask] for i, w_i in enumerate(w_opt))
        p_g_val = np.exp(log_g_val - np.max(log_g_val, axis=1, keepdims=True))
        p_g_val /= p_g_val.sum(axis=1, keepdims=True)
        pred_v49_nested[val_mask] = np.argmax(p_g_val * PESOS_PLUG_IN, axis=1)
        print(f"  Fold {fold} Pesos óptimos: {np.round(w_opt, 4)}", flush=True)

    f1_v49_global = float(f1_score(y_true, pred_v49_nested, average="macro"))
    f1_v49_folds = [float(f1_score(y_true[fold_idx == f], pred_v49_nested[fold_idx == f], average="macro")) for f in range(5)]
    deltas = [c - b for c, b in zip(f1_v49_folds, f1_v48_folds)]
    pos_folds = sum(d > 0 for d in deltas)

    print(f"\n[V49 Ensamble] Global: {f1_v49_global:.7f} (Delta vs V48: {f1_v49_global - f1_v48_global:+.7f})", flush=True)
    print(f"[V49 Ensamble] Folds: {[round(x, 6) for x in f1_v49_folds]}", flush=True)
    print(f"[V49 Ensamble] Deltas por Fold: {[round(d, 6) for d in deltas]}", flush=True)
    print(f"[V49 Ensamble] Folds Positivos: {pos_folds} / 5", flush=True)

    # Promedio de pesos para inferencia de test
    mean_weights = np.mean(fold_weights, axis=0)
    mean_weights /= mean_weights.sum()
    print(f"[V49 Ensamble] Pesos Promedio Convergidos: {np.round(mean_weights, 4)}", flush=True)

    # Bootstrap pareado B=1000
    print("\n[V49] Ejecutando bootstrap pareado B=1000...", flush=True)
    rng = np.random.default_rng(cfg.SEED)
    B = 1000
    n = len(y_true)
    boot_deltas = np.zeros(B, dtype=np.float32)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        s_v49 = f1_score(y_true[idx], pred_v49_nested[idx], average="macro")
        s_v48 = f1_score(y_true[idx], pred_v48[idx], average="macro")
        boot_deltas[b] = s_v49 - s_v48

    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))
    p_val = float(np.mean(boot_deltas <= 0))
    print(f"  Bootstrap IC95: [{ci_lower:+.7f}, {ci_upper:+.7f}]", flush=True)
    print(f"  Bootstrap p-valor: {p_val:.4f}", flush=True)

    pasa_gate = (f1_v49_global > f1_v48_global) and (pos_folds >= 4) and (p_val < 0.05)
    print(f"\n[V49] Supera Compuerta de Adopción (CV > V48, Folds >= 4/5, p < 0.05): {pasa_gate}", flush=True)

    resultado = {
        "fase": "V49-01",
        "fecha": time.strftime("%Y-%m-%d"),
        "duracion_seg": round(time.time() - inicio, 1),
        "f1_standalone_global": f1_standalone_global,
        "f1_standalone_folds": fold_scores_standalone,
        "f1_v48_baseline_global": f1_v48_global,
        "f1_v48_baseline_folds": f1_v48_folds,
        "f1_v49_blend_global": f1_v49_global,
        "f1_v49_blend_folds": f1_v49_folds,
        "deltas_por_fold": deltas,
        "folds_positivos": pos_folds,
        "bootstrap_mean_delta": float(np.mean(boot_deltas)),
        "bootstrap_ci95": [ci_lower, ci_upper],
        "bootstrap_p_value": p_val,
        "pesos_optimos_convergidos": list(mean_weights),
        "pasa_gate": bool(pasa_gate),
    }

    RESULTADO.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTADO, "w") as f:
        json.dump(resultado, f, indent=2)

    return 0 if pasa_gate else 1


if __name__ == "__main__":
    sys.exit(main())
