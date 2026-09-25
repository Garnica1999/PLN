"""FASE 8 — Confirmación y candidato v12: promedio de combinadores + NB-SVM de palabras.

Hallazgo que se confirma (Fase 7)
    - NB-SVM **solo de palabras** como miembro: 0,5254 individual y +0,0009 al combinador (3/5
      folds, por debajo del umbral).
    - **Promedio de combinadores** (MLP no lineal + LogReg lineal): **0,6005** frente a 0,5978 del
      MLP solo (**+0,0027**). Es el mismo principio que sostiene el ensamble de primer nivel:
      diversidad de sesgos. El promedio es una ponderación fija (0,5/0,5), sin nada ajustado a mano.

Protocolo
    1. Holdout congelado (15% de clústeres, semilla 2026, el mismo de ADV-03 y Fase 3).
    2. Si Δ ≥ +0,003 sobre el candidato v11 equivalente, se generan las features del lado test y la
       submission v12.

Salida: ``experiments/phase8_holdout_v12.json`` y ``submissions/age_submission_v12.csv``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import data, submission  # noqa: E402
from src.config import set_seed  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from phase2_ponderacion import EncodedMLP, build_features  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "phase8_holdout_v12.json"
HOLDOUT_SEED, HOLDOUT_SHARE = 2026, 0.15


def main() -> int:
    """Confirma el promedio de combinadores y genera la submission v12."""
    set_seed()
    started = time.time()
    payload: dict = {}

    features, y, _, _, _ = build_features()
    margins_word = np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")
    features = np.concatenate([features, margins_word], axis=1)

    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(y)]
    rng = np.random.default_rng(HOLDOUT_SEED)
    holdout_clusters = set(rng.choice(np.unique(cluster_ids),
                                      size=int(len(np.unique(cluster_ids)) * HOLDOUT_SHARE),
                                      replace=False).tolist())
    is_holdout = np.array([cluster in holdout_clusters for cluster in cluster_ids])
    development_idx = np.where(~is_holdout)[0]
    holdout_idx = np.where(is_holdout)[0]

    def fit_pair(train_idx: np.ndarray):
        """Ajusta el par de combinadores (MLP + LogReg) sobre las filas indicadas."""
        logreg = Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                      random_state=cfg.SEED))])
        logreg.fit(features[train_idx], y[train_idx])
        mlp = EncodedMLP()
        mlp.fit(features[train_idx], y[train_idx])
        return logreg, mlp

    def probabilities(logreg, mlp, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Probabilidades de cada combinador en las filas indicadas."""
        return (logreg.predict_proba(features[rows]),
                mlp.model.predict_proba(mlp.scaler.transform(features[rows])))

    logreg, mlp = fit_pair(development_idx)
    probabilities_logreg, probabilities_mlp = probabilities(logreg, mlp, holdout_idx)
    results = {}
    for label, matrix in (("LogReg", probabilities_logreg), ("MLP", probabilities_mlp),
                          ("promedio 0,5/0,5", 0.5 * (probabilities_mlp + probabilities_logreg))):
        predictions = np.array(cfg.CLASSES)[np.argmax(matrix, axis=1)]
        results[label] = {
            "macro_f1": round(float(f1_score(y[holdout_idx], predictions, average="macro",
                                             labels=cfg.CLASSES)), 4),
            "accuracy": round(float((predictions == y[holdout_idx]).mean()), 4),
        }
        print(f"[holdout] {label:16s}: macro-F1 {results[label]['macro_f1']:.4f} "
              f"| accuracy {results[label]['accuracy']:.4f}", flush=True)

    delta = results["promedio 0,5/0,5"]["macro_f1"] - results["MLP"]["macro_f1"]
    confirmed = results["promedio 0,5/0,5"]["macro_f1"] >= 0.003 + 0.6024
    payload["holdout"] = {"resultados": results, "delta_promedio_vs_mlp": round(delta, 4),
                          "criterio": "promedio ≥ 0,6024 + 0,003 = 0,6054",
                          "confirmado": bool(confirmed)}
    print(f"[holdout] Δ promedio vs MLP = {delta:+.4f} | confirmado: {confirmed}", flush=True)

    if not confirmed:
        print("\nNo se genera submission: el holdout no alcanza el criterio.", flush=True)
        RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                                encoding="utf-8")
        return 0

    # ---------------------------------------------------------------- Candidato v12
    print("\n[candidato] features del lado test...", flush=True)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from src import advanced_features as af, pipeline
    from src.textometry import SPANISH_STOPWORDS  # noqa: F401

    train = data.load_train()
    test = data.load_test()
    sample = data.load_sample_submission()
    dense_train = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")
    dense_test = pd.read_parquet(cfg.FEATURES / "dense_test.parquet")
    from src import textometry
    dense_columns = [name for name in textometry.TEXTOMETRY_FEATURES + textometry.STYLE_FEATURES
                     if name not in {"n_emoji", "emoji_diversity", "has_emoji"}]
    structured_test = pd.read_parquet(cfg.FEATURES / "age_structured_features_test.parquet").to_numpy(float)
    supervised_test = pd.concat([
        af.supervised_entity_features(train[cfg.TEXT_COL], y, test[cfg.TEXT_COL], "mention"),
        af.supervised_entity_features(train[cfg.TEXT_COL], y, test[cfg.TEXT_COL], "hashtag"),
        af.supervised_entity_features(train[cfg.TEXT_COL], y, test[cfg.TEXT_COL], "rt_source"),
        af.lexical_rarity_features(train[cfg.TEXT_COL], test[cfg.TEXT_COL]),
        af.class_lexicon_scores(train[cfg.TEXT_COL], y, test[cfg.TEXT_COL]),
    ], axis=1).to_numpy(float)

    from scripts.advanced_06_submission_v10 import nb_log_count_ratio, tokens_of
    from scripts.advanced_06_submission_v10 import CHAR_CONFIG, MENTION_CONFIG
    from sklearn.svm import LinearSVC

    representations = pipeline.build_representations(train, test)
    test_scores, _ = pipeline.fit_base_models(representations, y, dense_train, dense_test)

    char = TfidfVectorizer(**CHAR_CONFIG)
    mentions = TfidfVectorizer(**MENTION_CONFIG)
    x_train_nb = sp.hstack([char.fit_transform(train[cfg.TEXT_COL]),
                            mentions.fit_transform(tokens_of(train[cfg.TEXT_COL], r"@\w+"))]).tocsr()
    x_test_nb = sp.hstack([char.transform(test[cfg.TEXT_COL]),
                           mentions.transform(tokens_of(test[cfg.TEXT_COL], r"@\w+"))]).tocsr()
    nb_test = np.zeros((len(test), len(cfg.CLASSES)), dtype=np.float32)
    for position, label in enumerate(cfg.CLASSES):
        binary_target = (y == label).astype(int)
        ratio = nb_log_count_ratio(x_train_nb, binary_target)
        model = LinearSVC(C=0.2, max_iter=3000)
        model.fit(x_train_nb.multiply(ratio).tocsr(), binary_target)
        nb_test[:, position] = model.decision_function(x_test_nb.multiply(ratio).tocsr())

    nb_word_test = np.zeros((len(test), len(cfg.CLASSES)), dtype=np.float32)
    word = TfidfVectorizer(**{
        "analyzer": "word", "ngram_range": (1, 2), "min_df": 3, "sublinear_tf": True,
        "max_features": 300_000, "lowercase": True})
    x_train_w = word.fit_transform(train[cfg.TEXT_COL])
    x_test_w = word.transform(test[cfg.TEXT_COL])
    for position, label in enumerate(cfg.CLASSES):
        binary_target = (y == label).astype(int)
        ratio = nb_log_count_ratio(x_train_w, binary_target)
        model = LinearSVC(C=0.2, max_iter=3000)
        model.fit(x_train_w.multiply(ratio).tocsr(), binary_target)
        nb_word_test[:, position] = model.decision_function(x_test_w.multiply(ratio).tocsr())
    print(f"    bloques NB-SVM listos ({round(time.time() - started, 1)}s)", flush=True)

    dense_train_values = dense_train[dense_columns].to_numpy(float)
    dense_test_values = dense_test[dense_columns].to_numpy(float)
    x_meta_test = np.concatenate(test_scores + [nb_test, dense_test_values, structured_test,
                                                supervised_test, nb_word_test], axis=1)
    assert x_meta_test.shape[1] == features.shape[1], (
        f"desalineación: {x_meta_test.shape} vs {features.shape}")

    logreg_final, mlp_final = fit_pair(np.arange(len(y)))
    probabilities_logreg, probabilities_mlp = (
        logreg_final.predict_proba(x_meta_test),
        mlp_final.model.predict_proba(mlp_final.scaler.transform(x_meta_test)))
    predictions = np.array(cfg.CLASSES)[np.argmax(
        0.5 * (probabilities_mlp + probabilities_logreg), axis=1)]

    candidate = pd.DataFrame({cfg.ID_COL: test[cfg.ID_COL].to_numpy(), cfg.TARGET: predictions})
    checks = submission.validate_submission(candidate, test[cfg.ID_COL], sample)
    path = cfg.SUBMISSIONS / "age_submission_v12.csv"
    candidate.to_csv(path, index=False)

    payload["candidato"] = {
        "version": 12,
        "modelo": "14 fuentes + NB-SVM palabra (279 features) + promedio MLP/LogReg del meta",
        "cv_promedio_combinadores": 0.6005,
        "holdout_promedio": results["promedio 0,5/0,5"]["macro_f1"],
        "checks": checks,
        "distribucion_predicha": candidate[cfg.TARGET].value_counts().reindex(cfg.CLASSES)
                                 .fillna(0).astype(int).to_dict(),
        "archivo": str(path),
        "segundos": round(time.time() - started, 1),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
    print(json.dumps(payload["candidato"], indent=2, ensure_ascii=False)[:800])
    print(f"\narchivo listo: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
