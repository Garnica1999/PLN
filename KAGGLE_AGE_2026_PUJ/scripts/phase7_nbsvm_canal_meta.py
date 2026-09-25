"""FASE 7 / L7 + combinador — NB-SVM por canal y ensamble de meta-modelos.

Contexto
    Las ganancias de esta fase no vinieron de features nuevas sino del **combinador**: el
    meta-modelo MLP superó al LogReg en +0,0061 (5/5 folds) y confirmó +0,0043 en el holdout. Si el
    cuello de botella es la combinación, hay dos palancas directas:
      (a) darle al combinador **más vistas diversas** de los datos (NB-SVM por canal, línea L7);
      (b) **promediar dos combinadores diversos** (MLP no lineal y LogReg lineal), el mismo principio
          que hizo funcionar al ensamble de primer nivel.

Hipótesis
    H33  Un NB-SVM **solo de palabras** captura una vista distinta de la de caracteres y aporta al
         combinador actual.
    H34  Promediar las probabilidades del meta-modelo MLP y del LogReg mejora a cada uno por
         separado (diversidad lineal/no lineal en el segundo nivel).

Protocolo
    LOFO sobre las mismas particiones; el NB-SVM de palabras se calcula in-fold.

Salida: ``experiments/phase7_nbsvm_canal_meta.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, data  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.textometry import SPANISH_STOPWORDS  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from phase2_ponderacion import (EncodedMLP, build_features, evaluate_global,  # noqa: E402
                                make_model)

RESULTS_PATH = cfg.EXPERIMENTS / "phase7_nbsvm_canal_meta.json"
WORD_CONFIG = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 3, "sublinear_tf": True,
               "max_features": 300_000, "lowercase": True}


def nb_log_count_ratio(x_train: sp.csr_matrix, binary_target: np.ndarray) -> np.ndarray:
    """Log-count ratio de la clase positiva frente al resto (Wang & Manning 2012)."""
    positive = np.asarray(x_train[binary_target == 1].sum(axis=0)).ravel() + 1.0
    negative = np.asarray(x_train[binary_target == 0].sum(axis=0)).ravel() + 1.0
    return np.log((positive / positive.sum()) / (negative / negative.sum()))


def nb_svm_word_margins(texts: pd.Series, y: np.ndarray, train_idx: np.ndarray,
                        valid_idx: np.ndarray) -> np.ndarray:
    """NB-SVM uno-vs-rest sobre n-gramas de palabra (in-fold)."""
    vectorizer = TfidfVectorizer(**WORD_CONFIG)
    x_train = vectorizer.fit_transform(texts.iloc[train_idx])
    x_valid = vectorizer.transform(texts.iloc[valid_idx])
    y_train = y[train_idx]
    margins = np.zeros((len(valid_idx), len(cfg.CLASSES)), dtype=np.float32)
    for position, label in enumerate(cfg.CLASSES):
        binary_target = (y_train == label).astype(int)
        ratio = nb_log_count_ratio(x_train, binary_target)
        model = LinearSVC(C=0.2, max_iter=3000)
        model.fit(x_train.multiply(ratio).tocsr(), binary_target)
        margins[:, position] = model.decision_function(x_valid.multiply(ratio).tocsr())
    return margins


def lofo_probabilities(features: np.ndarray, y: np.ndarray, folds: list[dict],
                       kind: str) -> np.ndarray:
    """Probabilidades out-of-fold de un combinador."""
    probabilities = np.zeros((len(y), len(cfg.CLASSES)))
    for fold in folds:
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        if kind == "logreg":
            meta = Pipeline([("scaler", StandardScaler()),
                             ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                        random_state=cfg.SEED))])
            meta.fit(features[train_idx], y[train_idx])
            probabilities[valid_idx] = meta.predict_proba(features[valid_idx])
        else:
            encoder = EncodedMLP()
            encoder.fit(features[train_idx], y[train_idx])
            scaled = encoder.scaler.transform(features[valid_idx])
            probabilities[valid_idx] = encoder.model.predict_proba(scaled)
    return probabilities


def macro_from_probabilities(probabilities: np.ndarray, y: np.ndarray) -> tuple[float, list]:
    """macro-F1 global y por fold."""
    predictions = np.array(cfg.CLASSES)[np.argmax(probabilities, axis=1)]
    return float(f1_score(y, predictions, average="macro", labels=cfg.CLASSES))


def main() -> int:
    """Evalúa el canal NB-SVM de palabras y el promedio de combinadores."""
    set_seed()
    started = time.time()
    train = data.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    folds = cv.make_folds(y, groups=clusters["cluster"].to_numpy()[: len(y)],
                          n_splits=5, n_repeats=1, scheme="group")

    # --- L7: NB-SVM de palabras como miembro ---
    print("[1/2] NB-SVM de palabras (in-fold)...", flush=True)
    margins_word = np.zeros((len(y), len(cfg.CLASSES)), dtype=np.float32)
    per_fold_word = []
    for fold in folds:
        margins = nb_svm_word_margins(train[cfg.TEXT_COL], y, fold["train_idx"], fold["val_idx"])
        margins_word[fold["val_idx"]] = margins
        per_fold_word.append(float(f1_score(y[fold["val_idx"]],
                                            np.array(cfg.CLASSES)[np.argmax(margins, axis=1)],
                                            average="macro", labels=cfg.CLASSES)))
        print(f"    fold {fold['fold']}: {per_fold_word[-1]:.4f} "
              f"({round(time.time() - started, 1)}s)", flush=True)
    np.save(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy", margins_word)
    word_macro = float(np.mean(per_fold_word))
    print(f"    NB-SVM palabra CV: {word_macro:.4f} ± {np.std(per_fold_word):.4f}", flush=True)

    # --- Evaluación del aporte al combinador ---
    print("[2/2] aporte al meta-modelo MLP (LOFO)...", flush=True)
    features, y2, _, _, _ = build_features()
    assert np.array_equal(y, y2)
    base_mean, base_std, base_folds = evaluate_global(features, y, folds, "mlp")
    with_word = np.concatenate([features, margins_word], axis=1)
    with_mean, with_std, with_folds = evaluate_global(with_word, y, folds, "mlp")
    difference = np.array(with_folds) - np.array(base_folds)
    print(f"    MLP base {base_mean:.4f} | + NB palabra {with_mean:.4f} "
          f"({with_mean - base_mean:+.4f}, {int((difference > 0).sum())}/5 folds)", flush=True)

    # --- Combinadores promediados ---
    print("[3/3] promedio de combinadores (MLP + LogReg)...", flush=True)
    probabilities_mlp = lofo_probabilities(with_word, y, folds, "mlp")
    probabilities_logreg = lofo_probabilities(with_word, y, folds, "logreg")
    mlp_macro = macro_from_probabilities(probabilities_mlp, y)
    logreg_macro = macro_from_probabilities(probabilities_logreg, y)
    average_macro = macro_from_probabilities(0.5 * (probabilities_mlp + probabilities_logreg), y)
    print(f"    MLP {mlp_macro:.4f} | LogReg {logreg_macro:.4f} | promedio {average_macro:.4f}",
          flush=True)

    payload = {
        "nbsvm_palabra": {"macro_f1": round(word_macro, 4),
                          "per_fold": [round(v, 4) for v in per_fold_word],
                          "referencia_nbsvm_char_mencion": 0.5641, "referencia_svm_300k": 0.5658},
        "aporte_nbsvm_palabra_al_meta": {
            "mlp_base": round(base_mean, 4), "mlp_con_nbsvm_palabra": round(with_mean, 4),
            "delta": round(with_mean - base_mean, 4),
            "folds_con_mejora": int((difference > 0).sum()),
            "mejora_por_fold": difference.round(4).tolist(),
        },
        "combinadores": {"mlp": round(mlp_macro, 4), "logreg": round(logreg_macro, 4),
                         "promedio": round(average_macro, 4),
                         "delta_promedio_vs_mlp": round(average_macro - mlp_macro, 4)},
        "segundos": round(time.time() - started, 1),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== L7 + COMBINADOR ===")
    print(f"NB-SVM palabra como modelo : {word_macro:.4f} (svm completo: 0,5658)")
    print(f"aporte al combinador       : {payload['aporte_nbsvm_palabra_al_meta']['delta']:+.4f} "
          f"({payload['aporte_nbsvm_palabra_al_meta']['folds_con_mejora']}/5 folds)")
    print(f"combinadores               : MLP {mlp_macro:.4f} | LogReg {logreg_macro:.4f} | "
          f"promedio {average_macro:.4f} ({payload['combinadores']['delta_promedio_vs_mlp']:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
