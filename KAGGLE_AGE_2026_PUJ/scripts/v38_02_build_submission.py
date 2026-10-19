"""V38 · 02 — Construcción y validación estricta de la submission V38 Tabular Blend.

Ensambla las probabilidades de test del campeón V30 (0.88) con las del modelo
TabularNet (0.12), aplica la regla de decisión congelada plug-in w_plugin,
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

RESULTADO = cfg.EXPERIMENTS / "v38_02_submission_construccion.json"
SUBMISSION = cfg.ROOT / "submissions" / "age_submission_v38_tabular_blend.csv"
TEST_V30 = cfg.FEATURES / "age_plugin_test_prob.npy"
TEST_TAB = cfg.FEATURES / "v38_test_prob_tabularnet.npy"
PESOS_PLUG_IN = np.array([1.0000, 1.1749, 1.2721, 1.1630, 1.0010, 0.8128])
ALPHA_TABNET = 0.12
ALPHA_V30 = 0.88


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("[V38] Iniciando construcción de la submission V38...", flush=True)

    if not TEST_V30.exists():
        print(f"[STOP] No existe {TEST_V30.name}", flush=True)
        return 1
    if not TEST_TAB.exists():
        print(f"[STOP] No existe {TEST_TAB.name}", flush=True)
        return 1

    p_v30 = np.load(TEST_V30)
    p_tab = np.load(TEST_TAB)

    print(f"  Probabilidades V30 cargadas: {p_v30.shape}", flush=True)
    print(f"  Probabilidades TabNet cargadas: {p_tab.shape}", flush=True)

    test = datos.load_test()
    test_ids = test[cfg.ID_COL]
    sample = pd.read_csv(cfg.SUBMISSION_FILE, sep=cfg.SUBMISSION_SEP)

    # 1. Ensamble de probabilidades
    p_blend = ALPHA_V30 * p_v30 + ALPHA_TABNET * p_tab

    # 2. Regla de decisión por clase
    scores = p_blend * PESOS_PLUG_IN
    pred_idx = np.argmax(scores, axis=1)
    predictions = np.array(cfg.CLASSES)[pred_idx]

    candidate = pd.DataFrame({
        cfg.ID_COL: test_ids.to_numpy(),
        cfg.TARGET: predictions
    })

    # 3. Validación estricta
    checks = submission.validate_submission(candidate, test_ids, sample, strict_order=True)
    print(f"  Validaciones de formato superadas: {checks}", flush=True)

    # 4. Guardar archivo
    candidate.to_csv(SUBMISSION, index=False)
    sha256 = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()
    filas = len(candidate)

    # 5. Comparar con el campeón
    champ_sub = pd.read_csv(cfg.ROOT / "submissions" / "age_submission_plugin_candidate.csv")
    cambios = int(np.sum(candidate[cfg.TARGET].to_numpy() != champ_sub[cfg.TARGET].to_numpy()))
    pct_cambio = round(100.0 * cambios / filas, 3)

    print(f"\n[V38] Archivo generado: {SUBMISSION.name}")
    print(f"[V38] Filas: {filas:,} | SHA256: {sha256}")
    print(f"[V38] Discrepancia con Champion: {cambios:,} filas ({pct_cambio}%)")

    dist = candidate[cfg.TARGET].value_counts(normalize=True).to_dict()
    print(f"[V38] Distribución de clases predicha:\n{dist}")

    meta = {
        "fase": "V38 02 Construccion Submission",
        "archivo": SUBMISSION.name,
        "sha256": sha256,
        "filas": filas,
        "cambios_vs_champion": cambios,
        "pct_cambio": pct_cambio,
        "checks": checks,
        "distribucion": dist,
        "pesos_blend": {"v30": ALPHA_V30, "tabnet": ALPHA_TABNET},
        "pesos_plugin": PESOS_PLUG_IN.tolist(),
        "duracion_seg": round(time.time() - inicio, 2)
    }
    RESULTADO.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
