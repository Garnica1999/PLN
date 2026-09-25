"""FASE 11 — Ruido de etiqueta: ¿mejora el modelo si se excluyen los textos contradictorios?

Observación
    La auditoría de datos encontró **203 textos normalizados con más de una etiqueta** (643 filas,
    0,38% del train), además de 1.356 filas duplicadas exactas. Esos casos son ruido de anotación por
    construcción: el mismo texto no puede pertenecer a dos rangos de edad distintos. El análisis de
    errores mostró que los errores se concentran en las clases medias y en los duplicados, así que la
    hipótesis es que esas filas contaminan precisamente las fronteras difíciles.

Hipótesis
    H37  Excluir del **entrenamiento** del combinador las filas cuyo texto aparece con etiquetas
         contradictorias mejora macro-F1 (la validación se mantiene intacta, porque el test contendrá
         casos igualmente ruidosos).
    H38  Excluir además todas las filas con texto duplicado (no solo las contradictorias) aporta más,
         al eliminar el sesgo de sobre-representación de contenido repetido.

Protocolo
    LOFO sobre las mismas particiones agrupadas; se filtran filas **solo del lado de entrenamiento**
    de cada fold. Se evalúa el combinador promediado (MLP + LogReg) del modelo final v12.

Salida: ``experiments/phase11_ruido_etiqueta.json``.
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
from src import cv, data  # noqa: E402
from src.config import set_seed  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from phase2_ponderacion import EncodedMLP, build_features  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "phase11_ruido_etiqueta.json"


def _audit_normalize(texts: pd.Series) -> pd.Series:
    """Normalización del audit de datos (Fase 0): minúsculas y colapso de espacios.

    **No** elimina URLs, menciones ni puntuación: la normalización agresiva de `normalize_for_dedup`
    hace colapsar multitud de tweets cortos distintos al mismo texto vacío, lo que infla
    artificialmente el conteo de conflictos (12.989 filas en vez de 643).
    """
    return (texts.fillna("").astype(str).str.lower()
            .str.replace(r"\s+", " ", regex=True).str.strip())


def conflicting_mask(texts: pd.Series, y: np.ndarray) -> np.ndarray:
    """Filas cuyo texto normalizado aparece con más de una etiqueta."""
    normalized = _audit_normalize(texts)
    frame = pd.DataFrame({"texto": normalized, "y": y})
    labels_per_text = frame.groupby("texto")["y"].nunique()
    conflicting = set(labels_per_text[labels_per_text > 1].index)
    return normalized.isin(conflicting).to_numpy()


def duplicated_mask(texts: pd.Series) -> np.ndarray:
    """Filas cuyo texto normalizado aparece más de una vez (contradictorias incluidas)."""
    return _audit_normalize(texts).duplicated(keep=False).to_numpy()


def main() -> int:
    """Evalúa la exclusión de filas ruidosas del entrenamiento del combinador."""
    set_seed()
    started = time.time()
    train = data.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    texts = train[cfg.TEXT_COL]

    features, y2, _, _, _ = build_features()
    features = np.concatenate([features, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    assert np.array_equal(y, y2)

    conflict = conflicting_mask(texts, y)
    duplicate = duplicated_mask(texts)
    print(f"filas con etiqueta contradictoria: {conflict.sum():,} ({100 * conflict.mean():.2f}%)",
          flush=True)
    print(f"filas con texto duplicado        : {duplicate.sum():,} ({100 * duplicate.mean():.2f}%)",
          flush=True)

    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    folds = cv.make_folds(y, groups=clusters["cluster"].to_numpy()[: len(y)],
                          n_splits=5, n_repeats=1, scheme="group")

    def evaluate(exclude: np.ndarray | None, label: str) -> dict:
        """Evalúa el combinador promediado excluyendo del train las filas indicadas."""
        scores = []
        for fold in folds:
            train_rows = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
            valid_rows = fold["val_idx"]
            if exclude is not None:
                train_rows = train_rows[~exclude[train_rows]]
            logreg = Pipeline([("s", StandardScaler()),
                               ("c", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                        random_state=cfg.SEED))])
            logreg.fit(features[train_rows], y[train_rows])
            mlp = EncodedMLP()
            mlp.fit(features[train_rows], y[train_rows])
            probabilities = 0.5 * (logreg.predict_proba(features[valid_rows])
                                   + mlp.model.predict_proba(
                                       mlp.scaler.transform(features[valid_rows])))
            predictions = np.array(cfg.CLASSES)[np.argmax(probabilities, axis=1)]
            scores.append(float(f1_score(y[valid_rows], predictions, average="macro",
                                         labels=cfg.CLASSES)))
        detail = {"macro_f1": round(float(np.mean(scores)), 4),
                  "std": round(float(np.std(scores)), 4),
                  "per_fold": [round(value, 4) for value in scores]}
        print(f"  {label:34s}: {detail['macro_f1']:.4f} ± {detail['std']:.4f} "
              f"folds={detail['per_fold']}", flush=True)
        return detail

    baseline = evaluate(None, "baseline (v12, todo el train)")
    without_conflicts = evaluate(conflict, "sin filas contradictorias")
    without_duplicates = evaluate(duplicate, "sin textos duplicados (definición del audit)")
    without_both = without_duplicates

    improvement = np.array(without_conflicts["per_fold"]) - np.array(baseline["per_fold"])
    payload = {
        "filas_contradictorias": int(conflict.sum()),
        "filas_duplicadas": int(duplicate.sum()),
        "baseline": baseline,
        "sin_contradictorias": without_conflicts,
        "sin_duplicadas": without_duplicates,
        "sin_ambas": without_both,
        "delta_sin_contradictorias": round(without_conflicts["macro_f1"] - baseline["macro_f1"], 4),
        "folds_con_mejora": int((improvement > 0).sum()),
        "mejora_por_fold": improvement.round(4).tolist(),
        "segundos": round(time.time() - started, 1),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== FASE 11 — RUIDO DE ETIQUETA ===")
    print(f"baseline                      : {baseline['macro_f1']:.4f} ± {baseline['std']:.4f}")
    print(f"sin contradictorias (0,38%)   : {without_conflicts['macro_f1']:.4f} "
          f"({payload['delta_sin_contradictorias']:+.4f}, "
          f"{payload['folds_con_mejora']}/5 folds)")
    print(f"sin duplicadas                : {without_duplicates['macro_f1']:.4f} "
          f"({without_duplicates['macro_f1'] - baseline['macro_f1']:+.4f})")
    print(f"sin ambas                     : {without_both['macro_f1']:.4f} "
          f"({without_both['macro_f1'] - baseline['macro_f1']:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
