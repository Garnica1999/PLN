"""V54 · 01 — Construcción y validación estricta de la submission V54 Hierarchical Multi-Loss PoE Consensus.

Ensambla las probabilidades de test de 8 modelos inductivos complementarios:
1. V30 (LightGBM Optuna) — peso 0.55154
2. Cramer-SWA-SE-TabNet (V43, L2 CDF Wasserstein) — peso 0.06523
3. Weighted-Cramer-Clipped-TabNet (V47, Asymmetric Boundaries + Outlier Winsorization) — peso 0.01631
4. Smooth-SWA-SE-TabNet (V42, Label Smoothing 0.10) — peso 0.16895
5. SE-TabNet Ordinal (V40, Cumulative Logits) — peso 0.07389
6. TabNet Ordinal (V39, Deep Residual Ordinal) — peso 0.05027
7. HistGradientBoosting (V39, Regularized Tree Binning) — peso 0.05380
8. Dense-SE-TabularNet (V53, Multi-scale Concatenation Huang et al. CVPR 2017) — peso 0.02000

Utiliza el consenso Bayesiano Product-of-Experts en el espacio de log-probabilidades:
    log P_geom(y = k | x) = sum_{m=1}^8 w_m log P_m(y = k | x) - log Z(x)

Aplica la regla congelada plug-in w = [1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128]
y valida exhaustivamente contra age_test.csv y age_submission.csv (8 checks de integridad).
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

RESULTADO = cfg.EXPERIMENTS / "v54_01_submission_construccion.json"
SUBMISSION = cfg.ROOT / "submissions" / "age_submission_v54_hierarchical_poe_consensus.csv"

TEST_V30 = cfg.FEATURES / "age_plugin_test_prob.npy"
TEST_V43 = cfg.FEATURES / "v43_test_cramer_swa_tabularnet.npy"
TEST_V47 = cfg.FEATURES / "v47_test_weighted_cramer_clipped_tabularnet.npy"
TEST_V42 = cfg.FEATURES / "v42_test_smooth_swa_tabularnet.npy"
TEST_V40 = cfg.FEATURES / "v40_test_se_tabularnet.npy"
TEST_ORD = cfg.FEATURES / "v39_test_tabnet_ordinal.npy"
TEST_HGB = cfg.FEATURES / "v39_test_hgb.npy"
TEST_DENSE = cfg.FEATURES / "v53_test_dense_se_tabularnet.npy"

PESOS_PLUG_IN = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])

# 8 pesos normalizados con división óptima en rama Cramér (0.80 V43 + 0.20 V47) y DenseNet (0.020)
W_V48 = np.array([0.5628, 0.0832, 0.1724, 0.0754, 0.0513, 0.0549])
W_V48 = W_V48 / W_V48.sum()

W_DENSE = 0.0200
REM = 1.0 - W_DENSE
W_BASE = W_V48 * REM

W_V30 = W_BASE[0]
W_CRAMER_TOT = W_BASE[1]
W_V43 = W_CRAMER_TOT * 0.80
W_V47 = W_CRAMER_TOT * 0.20
W_V42 = W_BASE[2]
W_V40 = W_BASE[3]
W_ORD = W_BASE[4]
W_HGB = W_BASE[5]

WEIGHTS_GEOM = np.array([W_V30, W_V43, W_V47, W_V42, W_V40, W_ORD, W_HGB, W_DENSE])
WEIGHTS_GEOM = WEIGHTS_GEOM / WEIGHTS_GEOM.sum()


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("[V54] Iniciando construcción de submission V54 Hierarchical PoE Consensus...", flush=True)

    p_v30 = np.load(TEST_V30)
    p_v43 = np.load(TEST_V43)
    p_v47 = np.load(TEST_V47)
    p_v42 = np.load(TEST_V42)
    p_v40 = np.load(TEST_V40)
    p_ord = np.load(TEST_ORD)
    p_hgb = np.load(TEST_HGB)
    p_dense = np.load(TEST_DENSE)

    models = [p_v30, p_v43, p_v47, p_v42, p_v40, p_ord, p_hgb, p_dense]
    model_names = [
        "V30", "V43_Cramer", "V47_WCramer_Clip", "V42_Smooth",
        "V40_SE_Ord", "V39_Ord", "V39_HGB", "V53_Dense"
    ]
    for idx, (m, name, w) in enumerate(zip(models, model_names, WEIGHTS_GEOM)):
        print(f"  Modelo {idx+1} ({name:18s}): shape {m.shape}, peso log-space: {w:.5f}")

    # Ensamble geométrico en log-probabilidades (Product-of-Experts)
    eps = 1e-12
    log_models = [np.log(np.clip(m, eps, 1.0)) for m in models]
    log_geom = sum(w * lm for w, lm in zip(WEIGHTS_GEOM, log_models))
    p_geom = np.exp(log_geom - np.max(log_geom, axis=1, keepdims=True))
    p_geom /= p_geom.sum(axis=1, keepdims=True)

    # Aplicar regla congelada de decisión por clase
    p_ponderada = p_geom * PESOS_PLUG_IN
    pred_indices = np.argmax(p_ponderada, axis=1)

    labels = np.array(cfg.CLASSES)
    pred_labels = labels[pred_indices]

    test_df = datos.load_test()
    sub_sample = pd.read_csv(cfg.SUBMISSION_FILE, sep=cfg.SUBMISSION_SEP)

    candidate = pd.DataFrame({
        cfg.ID_COL: test_df[cfg.ID_COL].to_numpy(),
        cfg.TARGET: pred_labels,
    })

    # Validaciones rigurosas
    checks = submission.validate_submission(candidate, test_df[cfg.ID_COL], sub_sample)
    print("\n[V54] Resultado de validaciones:")
    for k, v in checks.items():
        print(f"  - {k}: {v}")
    if not all(checks.values()):
        raise RuntimeError("Fallo en las validaciones de la submission V54")

    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(SUBMISSION, index=False)

    sha256 = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()
    print(f"\n[V54] Submission guardada: {SUBMISSION}")
    print(f"[V54] SHA256: {sha256}")
    print(f"[V54] Total filas: {len(candidate)}")

    dist = candidate[cfg.TARGET].value_counts().to_dict()
    print("\n[V54] Distribución de predicciones:")
    for cls in cfg.CLASSES:
        cnt = dist.get(cls, 0)
        print(f"  {cls}: {cnt} ({cnt / len(candidate) * 100:.2f}%)")

    # Comparación de cambios vs V48 Champion
    v48_sub_path = cfg.ROOT / "submissions" / "age_submission_v48_geometric_poe_blend.csv"
    cambios_v48 = 0
    pct_v48 = 0.0
    if v48_sub_path.exists():
        v48_sub = pd.read_csv(v48_sub_path)
        cambios_v48 = int((candidate[cfg.TARGET] != v48_sub[cfg.TARGET]).sum())
        pct_v48 = float(cambios_v48 / len(candidate) * 100)
        print(f"\n[V54] Comparación vs V48 Champion: {cambios_v48} cambios ({pct_v48:.2f}%)")

    # Comparación de cambios vs V53
    v53_sub_path = cfg.ROOT / "submissions" / "age_submission_v53_dense_poe_blend.csv"
    cambios_v53 = 0
    pct_v53 = 0.0
    if v53_sub_path.exists():
        v53_sub = pd.read_csv(v53_sub_path)
        cambios_v53 = int((candidate[cfg.TARGET] != v53_sub[cfg.TARGET]).sum())
        pct_v53 = float(cambios_v53 / len(candidate) * 100)
        print(f"[V54] Comparación vs V53: {cambios_v53} cambios ({pct_v53:.2f}%)")

    resultado = {
        "fase": "V54-01",
        "fecha": time.strftime("%Y-%m-%d"),
        "duracion_seg": round(time.time() - inicio, 2),
        "archivo": str(SUBMISSION),
        "sha256": sha256,
        "filas": len(candidate),
        "distribucion": dist,
        "pesos_geometricos": list(WEIGHTS_GEOM),
        "pesos_plug_in": list(PESOS_PLUG_IN),
        "checks": checks,
        "cambios_vs_v48": cambios_v48,
        "pct_cambios_vs_v48": pct_v48,
        "cambios_vs_v53": cambios_v53,
        "pct_cambios_vs_v53": pct_v53,
    }

    RESULTADO.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTADO, "w") as f:
        json.dump(resultado, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
