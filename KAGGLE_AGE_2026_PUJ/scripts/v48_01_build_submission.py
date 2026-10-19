"""V48 · 01 — Construcción y validación estricta de la submission V48 Geometric Product-of-Experts Blend.

Ensambla las probabilidades de test de 6 modelos con diversidad inductiva y métricas ordinales complementarias
utilizando la media geométrica (consenso Bayesiano Product-of-Experts en espacio de log-probabilidades):
    log P_geom(y = k | x) = sum_{m=1}^6 w_m log P_m(y = k | x) - log Z(x)

con los pesos óptimos convergidos en validación cruzada anidada (Nested CV):
    w = [0.5628, 0.0832, 0.1724, 0.0754, 0.0513, 0.0549]
correspondientes a:
- V30 (0.5628) - GBDT LightGBM refinado por Optuna
- Cramer-SWA-SE-TabNet (0.0832) - Red residual profunda con pérdida cuadrática de Cramér (CDF) y SWA
- Smooth-SWA-SE-TabNet (0.1724) - Red residual con atención SE, suavizado de etiquetas 0.10 y SWA
- SE-TabNet Ordinal (0.0754) - Red residual con atención SE y pérdida ordinal
- TabNet Ordinal (0.0513) - Red residual profunda con pérdida ordinal
- HistGradientBoosting (0.0549) - Árboles con binning regularizado

Aplica la regla congelada plug-in w = [1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128]
y valida exhaustivamente contra age_test.csv y age_submission.csv.
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

RESULTADO = cfg.EXPERIMENTS / "v48_01_submission_construccion.json"
SUBMISSION = cfg.ROOT / "submissions" / "age_submission_v48_geometric_poe_blend.csv"

TEST_V30 = cfg.FEATURES / "age_plugin_test_prob.npy"
TEST_V43 = cfg.FEATURES / "v43_test_cramer_swa_tabularnet.npy"
TEST_V42 = cfg.FEATURES / "v42_test_smooth_swa_tabularnet.npy"
TEST_V40 = cfg.FEATURES / "v40_test_se_tabularnet.npy"
TEST_ORD = cfg.FEATURES / "v39_test_tabnet_ordinal.npy"
TEST_HGB = cfg.FEATURES / "v39_test_hgb.npy"

PESOS_PLUG_IN = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])

WEIGHTS_GEOM = np.array([0.5628, 0.0832, 0.1724, 0.0754, 0.0513, 0.0549])
WEIGHTS_GEOM = WEIGHTS_GEOM / WEIGHTS_GEOM.sum()


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("[V48] Iniciando construcción de submission V48 Geometric PoE Blend...", flush=True)

    p_v30 = np.load(TEST_V30)
    p_v43 = np.load(TEST_V43)
    p_v42 = np.load(TEST_V42)
    p_v40 = np.load(TEST_V40)
    p_ord = np.load(TEST_ORD)
    p_hgb = np.load(TEST_HGB)

    models = [p_v30, p_v43, p_v42, p_v40, p_ord, p_hgb]
    for idx, (m, w) in enumerate(zip(models, WEIGHTS_GEOM)):
        print(f"  Modelo {idx+1}: shape {m.shape}, peso log-space: {w:.4f}")

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
    print("\n[V48] Resultado de validaciones:")
    for k, v in checks.items():
        print(f"  - {k}: {v}")
    if not all(checks.values()):
        raise RuntimeError("Fallo en las validaciones de la submission V48")

    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(SUBMISSION, index=False)

    sha256 = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()
    print(f"\n[V48] Submission guardada: {SUBMISSION}")
    print(f"[V48] SHA256: {sha256}")
    print(f"[V48] Total filas: {len(candidate)}")

    dist = candidate[cfg.TARGET].value_counts().to_dict()
    print("\n[V48] Distribución de predicciones:")
    for cls, cnt in dist.items():
        print(f"  {cls}: {cnt} ({cnt / len(candidate) * 100:.2f}%)")

    # Comparación de cambios vs V43
    v43_sub_path = cfg.ROOT / "submissions" / "age_submission_v43_cramer_blend.csv"
    if v43_sub_path.exists():
        v43_sub = pd.read_csv(v43_sub_path)
        cambios_v43 = (candidate[cfg.TARGET] != v43_sub[cfg.TARGET]).sum()
        pct_v43 = cambios_v43 / len(candidate) * 100
        print(f"\n[V48] Comparación vs V43 Champion: {cambios_v43} cambios ({pct_v43:.2f}%)")

    resultado = {
        "fase": "V48-01",
        "fecha": time.strftime("%Y-%m-%d"),
        "duracion_seg": round(time.time() - inicio, 2),
        "archivo": str(SUBMISSION),
        "sha256": sha256,
        "filas": len(candidate),
        "distribucion": dist,
        "pesos_geometricos": list(WEIGHTS_GEOM),
        "pesos_plug_in": list(PESOS_PLUG_IN),
        "checks": checks,
    }

    RESULTADO.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTADO, "w") as f:
        json.dump(resultado, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
