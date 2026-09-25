"""FASE 27 — Artefacto v12 reproducible (cierre de D-034).

Motivo
    La submission v12 subida no se reconstruye al 100% desde el repositorio (93,88%), aunque las
    matrices (fresca vs caché = 100,0000%), el ajuste (dos ajustes idénticos = 100,0000%) y el
    protocolo del holdout (0,6077 exacto) sí se reproducen. Este script deja un artefacto **v12
    reproducible desde el estado actual del repositorio** y lo compara con el archivo subido, sin
    sobrescribirlo.

Protocolo
    1. Matriz de test fresca verificada (``features/meta_features_test_v12_fresca.npy``).
    2. Ajuste del par de combinadores (LogReg + MLP) sobre las 279 features de todo el train.
    3. Submission con validación formal y de distribución, guardada como
       ``submissions/age_submission_v12_reproducible.csv``.
    4. Comparación con ``age_submission_v12.csv`` (el archivo enviado) y con la distribución que la
       fase 8 registró en su JSON.

Salida: ``experiments/phase27_reproduccion_v12.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import data, submission  # noqa: E402
from src.config import set_seed  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from phase2_ponderacion import EncodedMLP, build_features  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "phase27_reproduccion_v12.json"
ARCHIVO = cfg.SUBMISSIONS / "age_submission_v12_reproducible.csv"


def main() -> int:
    """Genera el artefacto v12 reproducible y lo compara con el archivo subido."""
    set_seed()
    started = time.time()
    payload: dict = {"fase": "reproducción del v12", "fecha": "2026-09-25",
                     "nota": "artefacto reproducible desde el repositorio; cierra D-034"}

    train = data.load_train()
    test = data.load_test()
    muestra = data.load_sample_submission()
    y = train[cfg.TARGET].astype(str).to_numpy()
    features, _, _, _, _ = build_features()
    features = np.concatenate([features, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    test_matrix = np.load(cfg.FEATURES / "meta_features_test_v12_fresca.npy")
    assert features.shape[1] == test_matrix.shape[1] == 279

    logreg = Pipeline([("scaler", StandardScaler()),
                       ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                  random_state=cfg.SEED))])
    logreg.fit(features, y)
    mlp = EncodedMLP()
    mlp.fit(features, y)
    probabilidades = 0.5 * (logreg.predict_proba(test_matrix)
                            + mlp.model.predict_proba(mlp.scaler.transform(test_matrix)))
    predicciones = np.array(cfg.CLASSES)[np.argmax(probabilidades, axis=1)]

    candidato = pd.DataFrame({cfg.ID_COL: test[cfg.ID_COL].to_numpy(), cfg.TARGET: predicciones})
    checks = submission.validate_submission(candidato, test[cfg.ID_COL], muestra)
    candidato.to_csv(ARCHIVO, index=False)

    subida = pd.read_csv(cfg.SUBMISSIONS / "age_submission_v12.csv")
    acuerdo = float((predicciones == subida[cfg.TARGET].astype(str).to_numpy()).mean())
    referencia_fase8 = json.loads((cfg.EXPERIMENTS / "phase8_holdout_v12.json")
                                  .read_text(encoding="utf-8"))["candidato"]["distribucion_predicha"]
    conteos = candidato[cfg.TARGET].value_counts().reindex(cfg.CLASSES).fillna(0).astype(int)
    payload.update({
        "checks": checks, "filas": len(candidato),
        "distribucion": conteos.to_dict(),
        "distribucion_registrada_fase8": referencia_fase8,
        "distribucion_identica_a_fase8": bool(conteos.to_dict() == referencia_fase8),
        "acuerdo_con_v12_subida_pct": round(100 * acuerdo, 4),
        "predicciones_distintas_vs_subida": int(round((1 - acuerdo) * len(candidato))),
        "archivo": str(ARCHIVO),
    })
    payload["segundos"] = round(time.time() - started, 1)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
    print(f"[v12 reproducible] checks: {checks}")
    print(f"[v12 reproducible] acuerdo con el archivo subido: {100 * acuerdo:.4f}%")
    print(f"[v12 reproducible] distribución idéntica a la registrada por la fase 8: "
          f"{payload['distribucion_identica_a_fase8']}")
    print(f"archivo: {ARCHIVO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
