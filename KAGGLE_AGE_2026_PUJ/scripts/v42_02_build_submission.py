"""V42 · 02 — Construcción y validación estricta de la submission V42 Smooth-SWA Blend.

Ensambla las probabilidades de test de 6 modelos con diversidad inductiva:
- V30 (0.575) - GBDT LightGBM refinado por Optuna
- Smooth-SWA-SE-TabNet (0.280) - Red residual con atención SE, suavizado de etiquetas 0.10, pérdida ordinal y SWA
- SWA-SE-TabNet (0.020) - Red residual con atención SE, pérdida ordinal y SWA estándar
- SE-TabNet Ordinal (0.060) - Red residual con atención SE y pérdida ordinal
- TabNet Ordinal (0.050) - Red residual profunda con pérdida ordinal
- HistGradientBoosting (0.015) - Árboles con binning regularizado

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

RESULTADO = cfg.EXPERIMENTS / "v42_02_submission_construccion.json"
SUBMISSION = cfg.ROOT / "submissions" / "age_submission_v42_smooth_swa_blend.csv"

TEST_V30 = cfg.FEATURES / "age_plugin_test_prob.npy"
TEST_V42 = cfg.FEATURES / "v42_test_smooth_swa_tabularnet.npy"
TEST_V41 = cfg.FEATURES / "v41_test_swa_tabularnet.npy"
TEST_V40 = cfg.FEATURES / "v40_test_se_tabularnet.npy"
TEST_ORD = cfg.FEATURES / "v39_test_tabnet_ordinal.npy"
TEST_HGB = cfg.FEATURES / "v39_test_hgb.npy"

PESOS_PLUG_IN = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])

W_TREE = 0.575
W_V42 = 0.280
W_V41 = 0.020
W_V40 = 0.060
W_ORD = 0.050
W_HGB = 0.015


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("[V42] Iniciando construcción de submission V42 Smooth-SWA Blend...", flush=True)

    p_v30 = np.load(TEST_V30)
    p_v42 = np.load(TEST_V42)
    p_v41 = np.load(TEST_V41)
    p_v40 = np.load(TEST_V40)
    p_ord = np.load(TEST_ORD)
    p_hgb = np.load(TEST_HGB)

    print(f"  Shape test V30: {p_v30.shape}")
    print(f"  Shape test V42: {p_v42.shape}")
    print(f"  Shape test V41: {p_v41.shape}")
    print(f"  Shape test V40: {p_v40.shape}")
    print(f"  Shape test ORD: {p_ord.shape}")
    print(f"  Shape test HGB: {p_hgb.shape}")

    # Ensamble ponderado de probabilidades de test
    p_blend = (
        W_TREE * p_v30 +
        W_V42 * p_v42 +
        W_V41 * p_v41 +
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
    print("\n[V42] Resultado de validaciones:")
    for k, v in checks.items():
        print(f"  - {k}: {v}")
    if not all(checks.values()):
        raise RuntimeError("Fallo en las validaciones de la submission V42")

    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(SUBMISSION, index=False)

    sha256 = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()
    print(f"\n[V42] Submission guardada: {SUBMISSION}")
    print(f"[V42] SHA256: {sha256}")
    print(f"[V42] Total filas: {len(candidate)}")

    dist = candidate[cfg.TARGET].value_counts().to_dict()
    print("\n[V42] Distribución de predicciones:")
    for cls, cnt in dist.items():
        print(f"  {cls}: {cnt} ({cnt / len(candidate) * 100:.2f}%)")

    # Comparación de cambios vs V41 y V30
    sub_v41 = pd.read_csv(cfg.ROOT / "submissions" / "age_submission_v41_swa_se_blend.csv")
    sub_v30 = pd.read_csv(cfg.ROOT / "submissions" / "age_submission_plugin_candidate.csv")
    diff_v41 = int((candidate[cfg.TARGET] != sub_v41[cfg.TARGET]).sum())
    diff_v30 = int((candidate[cfg.TARGET] != sub_v30[cfg.TARGET]).sum())
    print(f"\n[V42] Cambios vs V41 Blend: {diff_v41} / {len(candidate)} ({diff_v41 / len(candidate) * 100:.2f}%)")
    print(f"[V42] Cambios vs V30 baseline: {diff_v30} / {len(candidate)} ({diff_v30 / len(candidate) * 100:.2f}%)")

    duracion = round(time.time() - inicio, 2)
    meta = {
        "fase": "V42-02",
        "fecha": "2026-10-16",
        "duracion_seg": duracion,
        "archivo": str(SUBMISSION),
        "sha256": sha256,
        "filas": len(candidate),
        "pesos_blend": {
            "v30": W_TREE,
            "smooth_swa_tabularnet": W_V42,
            "swa_se_tabularnet": W_V41,
            "se_tabularnet": W_V40,
            "tabnet_ordinal": W_ORD,
            "hgb": W_HGB,
        },
        "distribucion": dist,
        "cambios_vs_v41": diff_v41,
        "cambios_vs_v30": diff_v30,
        "checks_ok": True,
    }

    RESULTADO.parent.mkdir(parents=True, exist_ok=True)
    RESULTADO.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[V42] Metadatos guardados en {RESULTADO}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
