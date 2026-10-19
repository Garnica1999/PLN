"""V53 · 02 — Construcción y validación estricta de la submission V53 Dense-SE-TabularNet Product-of-Experts Blend.

Ensambla las probabilidades de test de 7 modelos complementarios utilizando el consenso bayesiano
Product-of-Experts en el espacio de log-probabilidades:
    log P_geom(y = k | x) = sum_{m=1}^7 w_m log P_m(y = k | x) - log Z(x)

con los 7 modelos:
1. V30 (0.5487) - GBDT LightGBM refinado por Optuna
2. Cramer-SWA-SE-TabNet (0.0811) - Red residual profunda con pérdida cuadrática de Cramér (CDF) y SWA
3. Smooth-SWA-SE-TabNet (0.1681) - Red residual con atención SE, suavizado de etiquetas 0.10 y SWA
4. SE-TabNet Ordinal (0.0735) - Red residual con atención SE y pérdida ordinal
5. TabNet Ordinal (0.0500) - Red residual profunda con pérdida ordinal
6. HistGradientBoosting (0.0535) - Árboles con binning regularizado
7. Dense-SE-TabularNet (0.0250) - Red densa profunda con concatenación multi-escala, atención SE y SWA

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

RESULTADO = cfg.EXPERIMENTS / "v53_02_submission_construccion.json"
SUBMISSION = cfg.ROOT / "submissions" / "age_submission_v53_dense_poe_blend.csv"

TEST_V30 = cfg.FEATURES / "age_plugin_test_prob.npy"
TEST_V43 = cfg.FEATURES / "v43_test_cramer_swa_tabularnet.npy"
TEST_V42 = cfg.FEATURES / "v42_test_smooth_swa_tabularnet.npy"
TEST_V40 = cfg.FEATURES / "v40_test_se_tabularnet.npy"
TEST_ORD = cfg.FEATURES / "v39_test_tabnet_ordinal.npy"
TEST_HGB = cfg.FEATURES / "v39_test_hgb.npy"
TEST_DENSE = cfg.FEATURES / "v53_test_dense_se_tabularnet.npy"

PESOS_PLUG_IN = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])

# 7 pesos normalizados
W_V48 = np.array([0.5628, 0.0832, 0.1724, 0.0754, 0.0513, 0.0549])
W_V48 = W_V48 / W_V48.sum()
W_DENSE = 0.0250
REM = 1.0 - W_DENSE
WEIGHTS_GEOM = np.array(list(W_V48 * REM) + [W_DENSE])
WEIGHTS_GEOM = WEIGHTS_GEOM / WEIGHTS_GEOM.sum()


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("[V53] Iniciando construcción de submission V53 Dense-SE-TabularNet PoE Blend...", flush=True)

    p_v30 = np.load(TEST_V30)
    p_v43 = np.load(TEST_V43)
    p_v42 = np.load(TEST_V42)
    p_v40 = np.load(TEST_V40)
    p_ord = np.load(TEST_ORD)
    p_hgb = np.load(TEST_HGB)
    p_dense = np.load(TEST_DENSE)

    models = [p_v30, p_v43, p_v42, p_v40, p_ord, p_hgb, p_dense]
    model_names = ["V30", "V43_Cramer", "V42_Smooth", "V40_SE_Ord", "V39_Ord", "V39_HGB", "V53_Dense"]
    for idx, (m, name, w) in enumerate(zip(models, model_names, WEIGHTS_GEOM)):
        print(f"  Modelo {idx+1} ({name:12s}): shape {m.shape}, peso log-space: {w:.5f}")

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
    print("\n[V53] Resultado de validaciones:")
    for k, v in checks.items():
        print(f"  - {k}: {v}")
    if not all(checks.values()):
        raise RuntimeError("Fallo en las validaciones de la submission V53")

    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(SUBMISSION, index=False)

    sha256 = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()
    print(f"\n[V53] Submission guardada: {SUBMISSION}")
    print(f"[V53] SHA256: {sha256}")
    print(f"[V53] Total filas: {len(candidate)}")

    dist = candidate[cfg.TARGET].value_counts().to_dict()
    print("\n[V53] Distribución de predicciones:")
    for cls, cnt in dist.items():
        print(f"  {cls}: {cnt} ({cnt / len(candidate) * 100:.2f}%)")

    # Comparación de cambios vs V48 Champion
    v48_sub_path = cfg.ROOT / "submissions" / "age_submission_v48_geometric_poe_blend.csv"
    cambios_v48 = 0
    pct_v48 = 0.0
    if v48_sub_path.exists():
        v48_sub = pd.read_csv(v48_sub_path)
        cambios_v48 = int((candidate[cfg.TARGET] != v48_sub[cfg.TARGET]).sum())
        pct_v48 = float(cambios_v48 / len(candidate) * 100)
        print(f"\n[V53] Comparación vs V48 Champion: {cambios_v48} cambios ({pct_v48:.2f}%)")

    resultado = {
        "fase": "V53-02",
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
    }

    RESULTADO.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTADO, "w") as f:
        json.dump(resultado, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
