"""V39 · 02 — Construcción y validación estricta de la submission V39 Trio Blend.

Ensambla las probabilidades de test:
- V30 (0.84)
- TabularResNet Ordinal (0.12)
- HistGradientBoosting (0.04)
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

RESULTADO = cfg.EXPERIMENTS / "v39_02_submission_construccion.json"
SUBMISSION = cfg.ROOT / "submissions" / "age_submission_v39_trio_blend.csv"

TEST_V30 = cfg.FEATURES / "age_plugin_test_prob.npy"
TEST_TAB = cfg.FEATURES / "v39_test_tabnet_ordinal.npy"
TEST_HGB = cfg.FEATURES / "v39_test_hgb.npy"

PESOS_PLUG_IN = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])
W_V30 = 0.84
W_TAB = 0.12
W_HGB = 0.04


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("[V39] Iniciando construcción de submission V39 Trío Blend...", flush=True)

    p_v30 = np.load(TEST_V30)
    p_tab = np.load(TEST_TAB)
    p_hgb = np.load(TEST_HGB)

    test = datos.load_test()
    test_ids = test[cfg.ID_COL]
    sample = pd.read_csv(cfg.SUBMISSION_FILE, sep=cfg.SUBMISSION_SEP)

    # Ensamble de probabilidades
    p_blend = W_V30 * p_v30 + W_TAB * p_tab + W_HGB * p_hgb

    # Regla de decisión por clase
    scores = p_blend * PESOS_PLUG_IN
    pred_idx = np.argmax(scores, axis=1)
    predictions = np.array(cfg.CLASSES)[pred_idx]

    candidate = pd.DataFrame({
        cfg.ID_COL: test_ids.to_numpy(),
        cfg.TARGET: predictions
    })

    # Validación estricta
    checks = submission.validate_submission(candidate, test_ids, sample, strict_order=True)
    print(f"  Validaciones superadas: {checks}", flush=True)

    candidate.to_csv(SUBMISSION, index=False)
    sha256 = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()
    filas = len(candidate)

    # Discrepancias vs V38 y vs V30+plugin
    sub_v38 = pd.read_csv(cfg.ROOT / "submissions" / "age_submission_v38_tabular_blend.csv")
    sub_v30 = pd.read_csv(cfg.ROOT / "submissions" / "age_submission_plugin_candidate.csv")

    cambios_vs_v38 = int(np.sum(candidate[cfg.TARGET].to_numpy() != sub_v38[cfg.TARGET].to_numpy()))
    cambios_vs_v30 = int(np.sum(candidate[cfg.TARGET].to_numpy() != sub_v30[cfg.TARGET].to_numpy()))

    print(f"\n[V39] Archivo guardado: {SUBMISSION.name}")
    print(f"[V39] Filas: {filas:,} | SHA256: {sha256}")
    print(f"[V39] Cambios vs V38: {cambios_vs_v38:,} ({cambios_vs_v38 / filas * 100:.2f}%)")
    print(f"[V39] Cambios vs V30+Plugin: {cambios_vs_v30:,} ({cambios_vs_v30 / filas * 100:.2f}%)")

    dist = candidate[cfg.TARGET].value_counts(normalize=True).to_dict()
    meta = {
        "fase": "V39 02 Construccion Submission",
        "archivo": SUBMISSION.name,
        "sha256": sha256,
        "filas": filas,
        "cambios_vs_v38": cambios_vs_v38,
        "cambios_vs_v30": cambios_vs_v30,
        "checks": checks,
        "distribucion": dist,
        "pesos_blend": {"v30": W_V30, "tabnet_ord": W_TAB, "hgb": W_HGB},
        "pesos_plugin": PESOS_PLUG_IN.tolist(),
        "duracion_seg": round(time.time() - inicio, 2)
    }
    RESULTADO.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
