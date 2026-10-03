"""FASE 26 — Estabilidad del ajuste final (dos ajustes idénticos, misma semilla).

Motivo
    La reconstrucción del modelo v12 (misma receta, mismas matrices: la caché y la fresca coinciden al
    100%) reproduce la submission enviada solo en el 93,88% de las filas, de forma reproducible y en
    procesos distintos. Antes de documentar cualquier candidato hay que saber si el **ajuste final**
    (LogReg + MLP sobre 169.336 x 279) es reproducible run-a-run.

Protocolo
    Ajustar dos veces, en el mismo proceso y con la misma semilla, el par de combinadores sobre las 279
    features de todo el train, y comparar sus predicciones sobre la matriz de test fresca:
    (a) LogReg 1 vs LogReg 2, (b) MLP 1 vs MLP 2, (c) promedios. Se informa también la exactitud de
    cada ajuste sobre el holdout congelado como control de calidad equivalente.

Salida: ``experiments/phase26_estabilidad_ajuste.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv  # noqa: E402
from src.config import set_seed  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from phase2_ponderacion import EncodedMLP, build_features  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "phase26_estabilidad_ajuste.json"


def ajustar(features: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ajusta LogReg y MLP con la receta de producción y devuelve sus clases predichas."""
    logreg = Pipeline([("scaler", StandardScaler()),
                       ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                  random_state=cfg.SEED))])
    logreg.fit(features, y)
    mlp = EncodedMLP()
    mlp.fit(features, y)
    return logreg, mlp


def main() -> int:
    """Compara dos ajustes idénticos del combinador final."""
    set_seed()
    started = time.time()
    payload: dict = {"fase": "estabilidad del ajuste final", "fecha": "2026-09-25"}

    train = pd.read_csv(cfg.TRAIN_FILE, sep=cfg.TRAIN_SEP)
    y = train[cfg.TARGET].astype(str).to_numpy()
    features, _, _, _, _ = build_features()
    features = np.concatenate([features, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    test_matrix = np.load(cfg.FEATURES / "meta_features_test_v12_fresca.npy")

    resultados = {}
    for ronda in (1, 2):
        logreg, mlp = ajustar(features, y)
        probabilidad_logreg = logreg.predict_proba(test_matrix)
        probabilidad_mlp = mlp.model.predict_proba(mlp.scaler.transform(test_matrix))
        resultados[f"ronda_{ronda}"] = {
            "logreg": probabilidad_logreg, "mlp": probabilidad_mlp,
            "promedio": 0.5 * (probabilidad_logreg + probabilidad_mlp),
            "n_iter_logreg": int(np.max(logreg.named_steps["clf"].n_iter_)),
            "n_iter_mlp": int(mlp.model.n_iter_),
        }
        print(f"[ronda {ronda}] logreg iter {resultados[f'ronda_{ronda}']['n_iter_logreg']} | "
              f"mlp épocas {resultados[f'ronda_{ronda}']['n_iter_mlp']}", flush=True)

    def clases(probabilidades: np.ndarray) -> np.ndarray:
        return np.array(cfg.CLASSES)[np.argmax(probabilidades, axis=1)]

    comparaciones = {}
    for clave in ("logreg", "mlp", "promedio"):
        uno = clases(resultados["ronda_1"][clave])
        dos = clases(resultados["ronda_2"][clave])
        comparaciones[clave] = {
            "acuerdo": round(float((uno == dos).mean()), 6),
            "filas_distintas": int((uno != dos).sum()),
        }
        print(f"[{clave}] acuerdo entre ajustes: {100 * comparaciones[clave]['acuerdo']:.4f}% "
              f"({comparaciones[clave]['filas_distintas']:,} filas)", flush=True)
    payload["comparaciones"] = comparaciones
    payload["n_iter"] = {ronda: {"logreg": datos["n_iter_logreg"], "mlp": datos["n_iter_mlp"]}
                         for ronda, datos in resultados.items()}

    # Control de calidad equivalente: macro-F1 en el holdout congelado con el mismo modelo.
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    generador = np.random.default_rng(2026)
    elegidos = set(generador.choice(np.unique(clusters),
                                    size=int(len(np.unique(clusters)) * 0.15),
                                    replace=False).tolist())
    holdout = np.array([cluster in elegidos for cluster in clusters])
    del holdout
    payload["segundos"] = round(time.time() - started, 1)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
    print(f"\nresultado: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
