"""E-028 — Supervisión ordinal fina: pares adyacentes y jerarquía gruesa→fina.

Contexto (Fase 17 y D-016)
    La confusión del ensamble es casi toda entre rangos **contiguos** y las clases medias son las
    más débiles. El modelo ordinal acumulativo (E-021a) aportó poco porque K-1 umbrales siguen
    siendo modelos globales entrenados con todas las clases.

Hipótesis
    H23  **Clasificadores de pares adyacentes**: un SVM entrenado *solo* con las dos clases de cada
         frontera (p. ej. 18-24 vs 25-34) concentra capacidad en la ambigüedad real y produce
         márgenes más nítidos en la frontera, información nueva para el meta-modelo.
    H24  **Jerarquía gruesa→fina**: agrupar en jóvenes (13-24), medios (25-49) y mayores (50-xx)
         da un problema más separable (más datos por clase, fronteras más limpias); sus márgenes
         son una vista distinta de la del clasificador de 6 clases.

Protocolo
    Mismas 5 particiones agrupadas; la representación se construye una vez por fold.
    Se guardan las matrices OOF (5 márgenes de pares + 3 márgenes de grupos).

Salida
    ``experiments/e028_ordinal_fino.json`` + ``features/oof_E028a_pares.npy`` y
    ``features/oof_E028b_grupos.npy``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, data, preprocessing  # noqa: E402
from src.config import set_seed  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "e028_ordinal_fino.json"
CHAR_CONFIG = {"analyzer": "char_wb", "ngram_range": (3, 6), "min_df": 5,
               "sublinear_tf": True, "max_features": 300_000, "lowercase": True}
MENTION_CONFIG = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2,
                  "sublinear_tf": True, "lowercase": True}

#: Fronteras entre rangos contiguos (pares adyacentes).
ADJACENT_PAIRS = [("13-17", "18-24"), ("18-24", "25-34"), ("25-34", "35-49"),
                  ("35-49", "50-64"), ("50-64", "65-xx")]
#: Agrupación gruesa para la jerarquía.
COARSE_GROUPS = {"13-17": "jovenes", "18-24": "jovenes", "25-34": "medios", "35-49": "medios",
                 "50-64": "mayores", "65-xx": "mayores"}
COARSE_LABELS = ["jovenes", "medios", "mayores"]


def tokens_of(series: pd.Series, pattern: str) -> pd.Series:
    """Extrae tokens que cumplen un patrón."""
    return preprocessing.collapse_spaces(
        series.str.findall(pattern).apply(lambda values: " ".join(values).lower()))


def main() -> int:
    """Entrena los miembros de ordinal fino y guarda sus OOF."""
    set_seed()
    started = time.time()
    train = data.load_train()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]
    y = train[cfg.TARGET].astype(str).to_numpy()
    texts = train[cfg.TEXT_COL]
    folds = cv.make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}
    oof_pairs = np.zeros((len(train), len(ADJACENT_PAIRS)), dtype=np.float32)
    oof_groups = np.zeros((len(train), len(COARSE_LABELS)), dtype=np.float32)

    for fold in folds:
        train_idx, valid_idx = fold["train_idx"], fold["val_idx"]
        char = TfidfVectorizer(**CHAR_CONFIG)
        mentions = TfidfVectorizer(**MENTION_CONFIG)
        x_train = hstack([char.fit_transform(texts.iloc[train_idx]),
                          mentions.fit_transform(tokens_of(texts.iloc[train_idx], r"@\w+"))]).tocsr()
        x_valid = hstack([char.transform(texts.iloc[valid_idx]),
                          mentions.transform(tokens_of(texts.iloc[valid_idx], r"@\w+"))]).tocsr()
        y_train = y[train_idx]

        # --- H23: un SVM por frontera adyacente (margen hacia la clase mayor) ---
        pair_margins = np.zeros((len(valid_idx), len(ADJACENT_PAIRS)), dtype=np.float32)
        pair_accuracy = []
        for position, (lower, upper) in enumerate(ADJACENT_PAIRS):
            mask = (y_train == lower) | (y_train == upper)
            binary_target = (y_train[mask] == upper).astype(int)
            model = LinearSVC(C=0.2, max_iter=3000)
            model.fit(x_train[mask], binary_target)
            margin = model.decision_function(x_valid)
            pair_margins[:, position] = margin
            validation_mask = (y[valid_idx] == lower) | (y[valid_idx] == upper)
            if validation_mask.sum() > 0:
                predicted = np.where(margin[validation_mask] > 0, upper, lower)
                pair_accuracy.append(float((predicted == y[valid_idx][validation_mask]).mean()))
        oof_pairs[valid_idx] = pair_margins

        # --- H24: clasificador grueso de 3 grupos ---
        coarse_train = np.array([COARSE_GROUPS[label] for label in y_train])
        coarse_model = LinearSVC(C=0.2, max_iter=3000)
        coarse_model.fit(x_train, coarse_train)
        order = [list(coarse_model.classes_).index(label) for label in COARSE_LABELS]
        coarse_margin = coarse_model.decision_function(x_valid)[:, order]
        oof_groups[valid_idx] = coarse_margin
        coarse_target = np.array([COARSE_GROUPS[label] for label in y[valid_idx]])
        coarse_accuracy = float((np.array(COARSE_LABELS)[np.argmax(coarse_margin, axis=1)]
                                 == coarse_target).mean())

        results.setdefault("E-028a", {"per_fold": [], "descripcion":
                                      "5 SVM de pares adyacentes (márgenes)"})
        results["E-028a"]["per_fold"].append({
            "fold": fold["fold"],
            "accuracy_promedio_en_fronteras": float(np.mean(pair_accuracy)),
            "accuracy_por_frontera": dict(zip([f"{a}|{b}" for a, b in ADJACENT_PAIRS],
                                              [round(value, 4) for value in pair_accuracy],
                                              strict=True)),
        })
        results.setdefault("E-028b", {"per_fold": [], "descripcion":
                                      "clasificador grueso de 3 grupos (jóvenes/medios/mayores)"})
        results["E-028b"]["per_fold"].append({"fold": fold["fold"],
                                              "accuracy_grueso": coarse_accuracy})
        print(f"[fold {fold['fold']}] pares (acc. media en fronteras) "
              f"{np.mean(pair_accuracy):.4f} | grupos {coarse_accuracy:.4f} "
              f"({round(time.time() - started, 1)}s)", flush=True)
        RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    for key in ("E-028a", "E-028b"):
        frame = pd.DataFrame(results[key]["per_fold"])
        if key == "E-028a":
            results[key]["accuracy_fronteras_mean"] = float(
                frame["accuracy_promedio_en_fronteras"].mean())
        else:
            results[key]["accuracy_grueso_mean"] = float(frame["accuracy_grueso"].mean())
    results["segundos"] = round(time.time() - started, 1)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    np.save(cfg.FEATURES / "oof_E028a_pares.npy", oof_pairs)
    np.save(cfg.FEATURES / "oof_E028b_grupos.npy", oof_groups)

    print("\n=== E-028 — SUPERVISIÓN ORDINAL FINA ===")
    print(f"E-028a — precisión media en las 5 fronteras adyacentes: "
          f"{results['E-028a']['accuracy_fronteras_mean']:.4f}")
    for pair, value in results["E-028a"]["per_fold"][-1]["accuracy_por_frontera"].items():
        print(f"    {pair}: {value:.4f}")
    print(f"E-028b — clasificador grueso de 3 grupos: "
          f"{results['E-028b']['accuracy_grueso_mean']:.4f}")
    print("\nReferencias: clasificador de 6 clases 0,5658 · ensamble 0,5860")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
