"""V43 · 02 — Construcción y validación estricta de la submission V43 Cramer-Smooth Blend.

Ensambla las probabilidades de test de 6 modelos con diversidad inductiva y métricas ordinales complementarias:
- V30 (0.555) - GBDT LightGBM refinado por Optuna
- Cramer-SWA-SE-TabNet (0.085) - Red residual profunda con pérdida cuadrática de Cramér (CDF / Wasserstein-2) y SWA
- Smooth-SWA-SE-TabNet (0.175) - Red residual con atención SE, suavizado de etiquetas 0.10, pérdida ordinal y SWA
- SE-TabNet Ordinal (0.080) - Red residual con atención SE y pérdida ordinal
- TabNet Ordinal (0.050) - Red residual profunda con pérdida ordinal
- HistGradientBoosting (0.055) - Árboles con binning regularizado

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

RESULTADO = cfg.EXPERIMENTS / "v43_02_submission_construccion.json"
SUBMISSION = cfg.ROOT / "submissions" / "age_submission_v43_cramer_blend.csv"

TEST_V30 = cfg.FEATURES / "age_plugin_test_prob.npy"
TEST_V43 = cfg.FEATURES / "v43_test_cramer_swa_tabularnet.npy"
TEST_V42 = cfg.FEATURES / "v42_test_smooth_swa_tabularnet.npy"
TEST_V40 = cfg.FEATURES / "v40_test_se_tabularnet.npy"
TEST_ORD = cfg.FEATURES / "v39_test_tabnet_ordinal.npy"
TEST_HGB = cfg.FEATURES / "v39_test_hgb.npy"

PESOS_PLUG_IN = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])

W_TREE = 0.555
W_V43 = 0.085
W_V42 = 0.175
W_V40 = 0.080
W_ORD = 0.050
W_HGB = 0.055


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("[V43] Iniciando construcción de submission V43 Cramer-Smooth Blend...", flush=True)

    p_v30 = np.load(TEST_V30)
    p_v43 = np.load(TEST_V43)
    p_v42 = np.load(TEST_V42)
    p_v40 = np.load(TEST_V40)
    p_ord = np.load(TEST_ORD)
    p_hgb = np.load(TEST_HGB)

    print(f"  Shape test V30: {p_v30.shape}")
    print(f"  Shape test V43: {p_v43.shape}")
    print(f"  Shape test V42: {p_v42.shape}")
    print(f"  Shape test V40: {p_v40.shape}")
    print(f"  Shape test ORD: {p_ord.shape}")
    print(f"  Shape test HGB: {p_hgb.shape}")

    # Ensamble ponderado de probabilidades de test
    p_blend = (
        W_TREE * p_v30 +
        W_V43 * p_v43 +
        W_V42 * p_v42 +
        W_V40 * p_v40 +
        W_ORD * p_ord +
        W_HGB * p_hgb
    )

    # Aplicar regla congelada de decisión por clase
    p_ponderada = p_blend * PESOS_PLUG_IN
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
    print("\n[V43] Resultado de validaciones:")
    for k, v in checks.items():
        print(f"  - {k}: {v}")
    if not all(checks.values()):
        raise RuntimeError("Fallo en las validaciones de la submission V43")

    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(SUBMISSION, index=False)

    sha256 = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()
    print(f"\n[V43] Submission guardada: {SUBMISSION}")
    print(f"[V43] SHA256: {sha256}")
    print(f"[V43] Total filas: {len(candidate)}")

    dist = candidate[cfg.TARGET].value_counts().to_dict()
    print("\n[V43] Distribución de predicciones:")
    for cls, cnt in dist.items():
        print(f"  {cls}: {cnt} ({cnt / len(candidate) * 100:.2f}%)")

    # Comparación de cambios vs V42, V41 y V30
    sub_v42 = pd.read_csv(cfg.ROOT / "submissions" / "age_submission_v42_smooth_swa_blend.csv")
    sub_v41 = pd.read_csv(cfg.ROOT / "submissions" / "age_submission_v41_swa_se_blend.csv")
    sub_v30 = pd.read_csv(cfg.ROOT / "submissions" / "age_submission_plugin_candidate.csv")

    diff_v42 = int((candidate[cfg.TARGET] != sub_v42[cfg.TARGET]).sum())
    diff_v41 = int((candidate[cfg.TARGET] != sub_v41[cfg.TARGET]).sum())
    diff_v30 = int((candidate[cfg.TARGET] != sub_v30[cfg.TARGET]).sum())

    print(f"\n[V43] Cambios vs V42 Blend: {diff_v42} / {len(candidate)} ({diff_v42 / len(candidate) * 100:.2f}%)")
    print(f"[V43] Cambios vs V41 Blend: {diff_v41} / {len(candidate)} ({diff_v41 / len(candidate) * 100:.2f}%)")
    print(f"[V43] Cambios vs V30 baseline: {diff_v30} / {len(candidate)} ({diff_v30 / len(candidate) * 100:.2f}%)")

    duracion = round(time.time() - inicio, 2)
    meta = {
        "fase": "V43-02",
        "fecha": time.strftime("%Y-%m-%d"),
        "duracion_seg": duracion,
        "archivo": str(SUBMISSION),
        "sha256": sha256,
        "filas": len(candidate),
        "pesos_blend": {
            "v30": W_TREE,
            "cramer_swa_tabularnet": W_V43,
            "smooth_swa_tabularnet": W_V42,
            "se_tabularnet": W_V40,
            "tabnet_ordinal": W_ORD,
            "hgb": W_HGB,
        },
        "distribucion": dist,
        "cambios_vs_v42": diff_v42,
        "cambios_vs_v41": diff_v41,
        "cambios_vs_v30": diff_v30,
        "checks_ok": True,
    }

    RESULTADO.parent.mkdir(parents=True, exist_ok=True)
    RESULTADO.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[V43] Metadatos guardados en {RESULTADO}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
