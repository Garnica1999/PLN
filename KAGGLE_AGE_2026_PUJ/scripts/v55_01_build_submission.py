"""V55 · 01 — Construcción y validación estricta de la submission V55 Clean Octa-Model PoE Consensus.

Ensambla las probabilidades de test de 8 modelos inductivos de alta calidad, excluyendo explícitamente
las redes de baja calidad standalone (como DenseNet standalone 0.6111) y unificando:
1. V30 (LightGBM Optuna) — peso 0.52829
2. Cramer-SWA-SE-TabNet (V43, L2 CDF Wasserstein) — peso 0.07720
3. Weighted-Cramer-Clipped-TabNet (V47, L2 CDF Ponderada Asimétrica + Winsorización in-fold) — peso 0.02895
4. Smooth-SWA-SE-TabNet (V42, Label Smoothing 0.10) — peso 0.17931
5. SE-TabNet Ordinal (V40, Cumulative Logits) — peso 0.07890
6. TabNet Ordinal (V39, Deep Residual Ordinal) — peso 0.05062
7. HistGradientBoosting (V39, Regularized Tree Binning) — peso 0.04869
8. CatBoost (V49, Regularized Oblivious Trees) — peso 0.00805

Utiliza el consenso Bayesiano Product-of-Experts en el espacio de log-probabilidades:
    log P_geom(y = k | x) = sum_{m=1}^8 w_m log P_m(y = k | x) - log Z(x)

Aplica la regla congelada plug-in w = [1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128]
y valida exhaustivamente contra age_test.csv y age_submission.csv (8 checks formales).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg
from src import data as datos
from src import submission
from src.verificacion_numerica import autotest_numerico

RESULTADO = cfg.EXPERIMENTS / "v55_01_submission_construccion.json"
SUBMISSION = cfg.ROOT / "submissions" / "age_submission_v55_clean_octa_consensus.csv"

TEST_V30 = cfg.FEATURES / "age_plugin_test_prob.npy"
TEST_V43 = cfg.FEATURES / "v43_test_cramer_swa_tabularnet.npy"
TEST_V47 = cfg.FEATURES / "v47_test_weighted_cramer_clipped_tabularnet.npy"
TEST_V42 = cfg.FEATURES / "v42_test_smooth_swa_tabularnet.npy"
TEST_V40 = cfg.FEATURES / "v40_test_se_tabularnet.npy"
TEST_ORD = cfg.FEATURES / "v39_test_tabnet_ordinal.npy"
TEST_HGB = cfg.FEATURES / "v39_test_hgb.npy"
TEST_CB = cfg.FEATURES / "v49_test_catboost.npy"

PESOS_PLUG_IN = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])

WEIGHTS_GEOM = np.array([0.52829, 0.07720, 0.02895, 0.17931, 0.07890, 0.05062, 0.04869, 0.00805])
WEIGHTS_GEOM = WEIGHTS_GEOM / WEIGHTS_GEOM.sum()


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("[V55] Iniciando construcción de submission V55 Clean Octa-Model PoE Consensus...", flush=True)

    p_v30 = np.load(TEST_V30)
    p_v43 = np.load(TEST_V43)
    p_v47 = np.load(TEST_V47)
    p_v42 = np.load(TEST_V42)
    p_v40 = np.load(TEST_V40)
    p_ord = np.load(TEST_ORD)
    p_hgb = np.load(TEST_HGB)
    p_cb = np.load(TEST_CB)

    models = [p_v30, p_v43, p_v47, p_v42, p_v40, p_ord, p_hgb, p_cb]
    eps = 1e-12

    print(f"[V55] Ensamblando {len(models)} modelos con PoE en log-espacio...", flush=True)
    log_probs = np.zeros_like(p_v30, dtype=np.float64)
    for w_m, p_m in zip(WEIGHTS_GEOM, models):
        p_clipped = np.clip(p_m, eps, 1.0)
        log_probs += w_m * np.log(p_clipped)

    max_log = np.max(log_probs, axis=1, keepdims=True)
    p_geom = np.exp(log_probs - max_log)
    p_geom /= np.sum(p_geom, axis=1, keepdims=True)

    print("[V55] Aplicando regla plug-in por clase congelada...", flush=True)
    p_plugin = p_geom * PESOS_PLUG_IN
    pred_idx = np.argmax(p_plugin, axis=1)

    labels = [cfg.CLASSES[i] for i in pred_idx]
    df_test = datos.load_test()
    sub = pd.DataFrame({"id": df_test["id"].values, "age_range": labels})

    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUBMISSION, index=False)
    duracion = time.time() - inicio

    sha256 = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()
    print(f"[V55] Archivo generado: {SUBMISSION} ({len(sub)} filas, SHA256={sha256[:16]}...)", flush=True)

    print("[V55] Ejecutando validación estricta (8 checks formales)...", flush=True)
    df_sample = pd.read_csv(cfg.RAW / "age_submission.csv")

    checks = {
        "columns_ok": list(sub.columns) == ["id", "age_range"],
        "columns_match_sample": list(sub.columns) == list(df_sample.columns),
        "rows_ok": len(sub) == len(df_test) and len(sub) == 112892,
        "ids_equal_test": (sub["id"].values == df_test["id"].values).all(),
        "ids_unique": sub["id"].nunique() == len(sub),
        "ids_same_order": list(sub["id"]) == list(df_sample["id"]),
        "no_nan": sub.isna().sum().sum() == 0,
        "labels_valid": set(sub["age_range"].unique()).issubset(set(cfg.CLASSES)),
    }

    for k, v in checks.items():
        assert v, f"Fallo check: {k}"
        print(f"  [OK] {k}: {v}")

    dist = sub["age_range"].value_counts().to_dict()
    print(f"[V55] Distribución de clases en test: {dist}", flush=True)

    # Comparación con V48 Champion
    v48_sub = pd.read_csv(cfg.ROOT / "submissions" / "age_submission_v48_geometric_poe_blend.csv")
    cambios_vs_v48 = int((sub["age_range"] != v48_sub["age_range"]).sum())
    pct_cambios_v48 = float(cambios_vs_v48 / len(sub) * 100)
    print(f"[V55] Cambios respecto a V48 Champion: {cambios_vs_v48} filas ({pct_cambios_v48:.4f}%)", flush=True)

    resultado = {
        "fase": "V55-01",
        "fecha": "2026-10-18",
        "duracion_seg": round(duracion, 2),
        "archivo": str(SUBMISSION),
        "sha256": sha256,
        "filas": len(sub),
        "distribucion": dist,
        "pesos_geometricos": list(WEIGHTS_GEOM),
        "pesos_plug_in": list(PESOS_PLUG_IN),
        "checks": {k: bool(v) for k, v in checks.items()},
        "cambios_vs_v48": cambios_vs_v48,
        "pct_cambios_vs_v48": pct_cambios_v48,
    }

    with open(RESULTADO, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2)

    print(f"[V55] Resultado guardado en {RESULTADO}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
