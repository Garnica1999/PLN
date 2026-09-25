"""E-023 — Miembros especializados por longitud para el ensamble.

Contexto (Fase 17)
    El error se concentra en los tweets cortos: 52,5% en el quintil más corto frente a 37,2% en el
    más largo. El meta-modelo ya recibe la longitud, pero todos los miembros base se entrenaron
    sobre la mezcla completa.

Hipótesis
    H20  Un SVM entrenado **solo con tweets cortos** captura patrones distintos de uno entrenado
         con tweets largos (donde hay más texto y otras señales). Añadidos como miembros, el
         meta-modelo puede **elegir** en quién confiar según la longitud de cada tweet.

Protocolo
    Mismas 5 particiones agrupadas. Por fold se entrena:
      - ``corto``: SVM sobre el 50% de menor longitud (umbral calculado **solo con el train del fold**).
      - ``largo``: SVM sobre el 50% restante.
    Ambos predicen **todas** las filas de validación (el meta-modelo aprende cuándo confiar en cada
    uno). Se guardan las matrices OOF.

Salida
    ``experiments/e023_especialistas_longitud.json`` + ``features/oof_E023a_corto.npy``,
    ``features/oof_E023b_largo.npy``.
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, data, preprocessing  # noqa: E402
from src.config import set_seed  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "e023_especialistas_longitud.json"
CHAR_CONFIG = {"analyzer": "char_wb", "ngram_range": (3, 6), "min_df": 5,
               "sublinear_tf": True, "max_features": 300_000, "lowercase": True}
MENTION_CONFIG = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2,
                  "sublinear_tf": True, "lowercase": True}


def tokens_of(series: pd.Series, pattern: str) -> pd.Series:
    """Extrae tokens que cumplen un patrón."""
    return preprocessing.collapse_spaces(
        series.str.findall(pattern).apply(lambda values: " ".join(values).lower()))


def main() -> int:
    """Entrena los especialistas por longitud y guarda sus matrices OOF."""
    set_seed()
    started = time.time()
    train = data.load_train()
    dense = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")
    lengths = dense["n_chars"].to_numpy()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]
    y = train[cfg.TARGET].astype(str).to_numpy()
    texts = train[cfg.TEXT_COL]
    folds = cv.make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}
    oof_short = np.zeros((len(train), len(cfg.CLASSES)), dtype=np.float32)
    oof_long = np.zeros((len(train), len(cfg.CLASSES)), dtype=np.float32)

    for fold in folds:
        train_idx, valid_idx = fold["train_idx"], fold["val_idx"]
        char = TfidfVectorizer(**CHAR_CONFIG)
        mentions = TfidfVectorizer(**MENTION_CONFIG)
        x_train = hstack([char.fit_transform(texts.iloc[train_idx]),
                          mentions.fit_transform(tokens_of(texts.iloc[train_idx], r"@\w+"))]).tocsr()
        x_valid = hstack([char.transform(texts.iloc[valid_idx]),
                          mentions.transform(tokens_of(texts.iloc[valid_idx], r"@\w+"))]).tocsr()

        # Umbral de longitud calculado SOLO con el train del fold (mediana).
        threshold = float(np.median(lengths[train_idx]))
        short_mask = lengths[train_idx] < threshold
        long_mask = ~short_mask

        scores = {}
        for name, mask in (("E-023a", short_mask), ("E-023b", long_mask)):
            model = LinearSVC(C=0.2, max_iter=3000)
            model.fit(x_train[mask], y[train_idx][mask])
            order = [list(model.classes_).index(label) for label in cfg.CLASSES]
            decision = model.decision_function(x_valid)[:, order]
            scores[name] = decision
            predictions = np.array(cfg.CLASSES)[np.argmax(decision, axis=1)]
            macro_all = f1_score(y[valid_idx], predictions, average="macro")
            # Desempeño en su subconjunto natural (el que le corresponde por longitud).
            subset = lengths[valid_idx] < threshold if name == "E-023a" else lengths[valid_idx] >= threshold
            macro_subset = f1_score(y[valid_idx][subset], predictions[subset], average="macro")
            results.setdefault(name, {"per_fold": [],
                                      "descripcion": "SVM especialista en tweets "
                                                     + ("cortos" if name == "E-023a" else "largos")})
            results[name]["per_fold"].append({
                "fold": fold["fold"], "umbral_longitud": threshold,
                "macro_f1_global": macro_all, "macro_f1_en_su_subconjunto": macro_subset,
                "accuracy_en_su_subconjunto": accuracy_score(y[valid_idx][subset],
                                                            predictions[subset]),
            })
        oof_short[valid_idx] = scores["E-023a"]
        oof_long[valid_idx] = scores["E-023b"]
        print(f"[fold {fold['fold']}] umbral={threshold:.0f} | "
              f"corto (global {results['E-023a']['per_fold'][-1]['macro_f1_global']:.4f}, "
              f"subconjunto {results['E-023a']['per_fold'][-1]['macro_f1_en_su_subconjunto']:.4f}) | "
              f"largo (global {results['E-023b']['per_fold'][-1]['macro_f1_global']:.4f}, "
              f"subconjunto {results['E-023b']['per_fold'][-1]['macro_f1_en_su_subconjunto']:.4f}) "
              f"({round(time.time() - started, 1)}s)", flush=True)
        RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    for key in ("E-023a", "E-023b"):
        frame = pd.DataFrame(results[key]["per_fold"])
        results[key]["macro_f1_global_mean"] = float(frame["macro_f1_global"].mean())
        results[key]["macro_f1_subconjunto_mean"] = float(frame["macro_f1_en_su_subconjunto"].mean())
    results["segundos"] = round(time.time() - started, 1)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    np.save(cfg.FEATURES / "oof_E023a_corto.npy", oof_short)
    np.save(cfg.FEATURES / "oof_E023b_largo.npy", oof_long)

    print("\n=== E-023 — ESPECIALISTAS POR LONGITUD ===")
    for key, label in (("E-023a", "SVM especialista en tweets CORTOS"),
                       ("E-023b", "SVM especialista en tweets LARGOS")):
        print(f"{key} ({label}): macro-F1 en su subconjunto "
              f"{results[key]['macro_f1_subconjunto_mean']:.4f} | global "
              f"{results[key]['macro_f1_global_mean']:.4f}")
    print("\nReferencia: SVM generalista 0,5658 (global) · ensamble 0,5860")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
