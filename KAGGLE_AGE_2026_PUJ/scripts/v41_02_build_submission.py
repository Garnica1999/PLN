"""V41 · 02 — Construcción y validación estricta de la submission V41 SWA-SE-Tabular Blend.

Ensambla las probabilidades de test de 5 modelos diversos:
- V30 (0.63) - GBDT LightGBM refinado por Optuna
- SWA-SE-TabNet (0.26) - Red residual con atención SE, pérdida ordinal y SWA en GPU
- SE-TabNet Ordinal (0.04) - Red residual con atención SE y pérdida ordinal en GPU
- TabNet Ordinal (0.05) - Red residual con pérdida ordinal en GPU
- HistGradientBoosting (0.02) - Árboles con binning regularizado

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

RESULTADO = cfg.EXPERIMENTS / "v41_02_submission_construccion.json"
SUBMISSION = cfg.ROOT / "submissions" / "age_submission_v41_swa_se_blend.csv"

TEST_V30 = cfg.FEATURES / "age_plugin_test_prob.npy"
TEST_SWA = cfg.FEATURES / "v41_test_swa_tabularnet.npy"
TEST_SE = cfg.FEATURES / "v40_test_se_tabularnet.npy"
TEST_ORD = cfg.FEATURES / "v39_test_tabnet_ordinal.npy"
TEST_HGB = cfg.FEATURES / "v39_test_hgb.npy"

PESOS_PLUG_IN = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])
W_V30 = 0.63
W_SWA = 0.26
W_SE = 0.04
W_ORD = 0.05
W_HGB = 0.02


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("[V41] Iniciando construcción de submission V41 SWA-SE-Tabular Blend...", flush=True)

    p_v30 = np.load(TEST_V30)
    p_swa = np.load(TEST_SWA)
    p_se = np.load(TEST_SE)
    p_ord = np.load(TEST_ORD)
    p_hgb = np.load(TEST_HGB)

    test = datos.load_test()
    test_ids = test[cfg.ID_COL]
    sample = pd.read_csv(cfg.SUBMISSION_FILE, sep=cfg.SUBMISSION_SEP)

    # Ensamble de probabilidades
    p_blend = W_V30 * p_v30 + W_SWA * p_swa + W_SE * p_se + W_ORD * p_ord + W_HGB * p_hgb

    # Regla de decisión por clase
    scores = p_blend * PESOS_PLUG_IN
    pred_idx = np.argmax(scores, axis=1)
    predictions = np.array(cfg.CLASSES)[pred_idx]

    candidate = pd.DataFrame({
        cfg.ID_COL: test_ids.to_numpy(),
        cfg.TARGET: predictions
    })

    # Validación estricta con módulo canónico
    checks = submission.validate_submission(candidate, test_ids, sample, strict_order=True)
    print(f"  Validaciones superadas: {checks}", flush=True)

    candidate.to_csv(SUBMISSION, index=False)
    sha256 = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()
    filas = len(candidate)

    # Discrepancias vs V40 y vs V30+plugin
    sub_v40 = pd.read_csv(cfg.ROOT / "submissions" / "age_submission_v40_se_tabular_blend.csv")
    sub_v30 = pd.read_csv(cfg.ROOT / "submissions" / "age_submission_plugin_candidate.csv")

    cambios_vs_v40 = int(np.sum(candidate[cfg.TARGET].to_numpy() != sub_v40[cfg.TARGET].to_numpy()))
    cambios_vs_v30 = int(np.sum(candidate[cfg.TARGET].to_numpy() != sub_v30[cfg.TARGET].to_numpy()))

    distribucion = candidate[cfg.TARGET].value_counts().to_dict()

    print(f"\n[V41] Archivo guardado: {SUBMISSION.name}")
    print(f"  Filas                 : {filas:,}")
    print(f"  SHA256                : {sha256}")
    print(f"  Cambios vs V40        : {cambios_vs_v40:,} ({cambios_vs_v40/filas*100:.2f}%)")
    print(f"  Cambios vs V30        : {cambios_vs_v30:,} ({cambios_vs_v30/filas*100:.2f}%)")
    print(f"  Distribución en test  :\n{json.dumps(distribucion, indent=2)}")

    duracion = time.time() - inicio
    resultado = {
        "fase": "V41-02",
        "fecha": time.strftime("%Y-%m-%d"),
        "duracion_seg": round(duracion, 2),
        "archivo": str(SUBMISSION),
        "sha256": sha256,
        "filas": filas,
        "pesos_blend": {
            "v30": W_V30,
            "swa_se_tabularnet": W_SWA,
            "se_tabularnet": W_SE,
            "tabnet_ordinal": W_ORD,
            "hgb": W_HGB,
        },
        "distribucion": distribucion,
        "cambios_vs_v40": cambios_vs_v40,
        "cambios_vs_v30": cambios_vs_v30,
        "checks_ok": bool(checks.get("todo_correcto", True)),
    }

    with open(RESULTADO, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2)

    print(f"[V41] Metadatos guardados en {RESULTADO} ({duracion:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
