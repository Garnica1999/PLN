"""Fase avanzada — Paso 3: holdout congelado y candidato AGE_ADVANCED_V01.

Parte A — **Holdout congelado** (brief §50): se reserva el 15% de los clústeres de
near-duplicados (agrupados, así que ningún duplicado cruza la frontera). El meta-modelo se
entrena **solo con el 85% restante** y se evalúa en el holdout, comparando el baseline congelado
contra el candidato con features de segunda generación.

Nota metodológica honesta: las matrices OOF de los modelos base se generaron con 5 folds sobre
todo train, por lo que las predicciones del holdout provienen de modelos base que vieron parte de
sus datos. Eso **afecta por igual al baseline y al candidato**, así que la comparación es válida;
el nivel absoluto es ligeramente optimista y por eso se reporta junto al CV de desarrollo, nunca
en su lugar.

Parte B — **Candidato**: se calculan las features supervisadas de test usando todo train como lado
de aprendizaje (sin etiquetas de test) y se entrena el meta-modelo definitivo.

Salida: ``experiments/advanced_03_holdout.json`` y ``submissions/age_submission_v09.csv``.
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

from src import advanced_features as af  # noqa: E402
from src import config as cfg  # noqa: E402
from src import data, pipeline, submission, textometry  # noqa: E402
from src.config import set_seed  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "advanced_03_holdout.json"
HOLDOUT_SHARE = 0.15
HOLDOUT_SEED = 2026
DENSE_FEATURES = [name for name in textometry.TEXTOMETRY_FEATURES + textometry.STYLE_FEATURES
                  if name not in {"n_emoji", "emoji_diversity", "has_emoji"}]


def load_oof() -> np.ndarray:
    """Carga las 13 matrices OOF de los modelos base (mismo orden que `src.pipeline`)."""
    matrices = []
    for filename, probabilistic in pipeline.OOF_FILES:
        matrix = np.load(cfg.FEATURES / filename)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        matrices.append(np.log(np.clip(matrix, 1e-12, None)) if probabilistic else matrix)
    return np.concatenate(matrices, axis=1)


def train_meta(features: np.ndarray, y: np.ndarray, train_idx: np.ndarray) -> Pipeline:
    """Entrena el meta-modelo solo con las filas indicadas."""
    meta = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                random_state=cfg.SEED))])
    meta.fit(features[train_idx], y[train_idx])
    return meta


def main() -> int:
    """Ejecuta el holdout congelado y construye el candidato final."""
    set_seed()
    started = time.time()
    results: dict = {}

    train = data.load_train()
    test = data.load_test()
    sample = data.load_sample_submission()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]

    dense_train = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")[DENSE_FEATURES].to_numpy(float)
    dense_test = pd.read_parquet(cfg.FEATURES / "dense_test.parquet")[DENSE_FEATURES].to_numpy(float)
    structured_train = pd.read_parquet(cfg.FEATURES / "age_structured_features.parquet").to_numpy(float)
    structured_test = pd.read_parquet(cfg.FEATURES / "age_structured_features_test.parquet").to_numpy(float)
    supervised_train = np.load(cfg.FEATURES / "age_supervised_features.npy")
    base = load_oof()

    # ---------------------------------------------------------------- Parte A: holdout
    rng = np.random.default_rng(HOLDOUT_SEED)
    unique_clusters = np.unique(cluster_ids)
    holdout_clusters = set(rng.choice(unique_clusters,
                                      size=int(len(unique_clusters) * HOLDOUT_SHARE),
                                      replace=False).tolist())
    is_holdout = np.array([cluster in holdout_clusters for cluster in cluster_ids])
    development_idx = np.where(~is_holdout)[0]
    holdout_idx = np.where(is_holdout)[0]
    print(f"[holdout] {len(holdout_idx):,} filas ({100 * is_holdout.mean():.1f}%) en "
          f"{len(holdout_clusters):,} clústeres; desarrollo {len(development_idx):,}", flush=True)

    variants = {
        "baseline": np.concatenate([base, dense_train], axis=1),
        "candidato": np.concatenate([base, dense_train, structured_train, supervised_train], axis=1),
    }
    holdout_scores: dict[str, dict] = {}
    for label, features in variants.items():
        meta = train_meta(features, y, development_idx)
        predictions = meta.predict(features[holdout_idx])
        macro = float(f1_score(y[holdout_idx], predictions, average="macro", labels=cfg.CLASSES))
        accuracy = float((predictions == y[holdout_idx]).mean())
        holdout_scores[label] = {"macro_f1": round(macro, 4), "accuracy": round(accuracy, 4),
                                 "n_features": int(features.shape[1])}
        print(f"[holdout] {label:10s}: macro-F1 {macro:.4f} | accuracy {accuracy:.4f} "
              f"({features.shape[1]} features)", flush=True)

    delta_holdout = holdout_scores["candidato"]["macro_f1"] - holdout_scores["baseline"]["macro_f1"]
    results["holdout_congelado"] = {
        "share": HOLDOUT_SHARE, "seed": HOLDOUT_SEED,
        "n_holdout": int(len(holdout_idx)), "n_clusters_holdout": len(holdout_clusters),
        "resultados": holdout_scores, "delta_macro_f1": round(delta_holdout, 4),
        "confirmado": bool(delta_holdout > 0),
    }
    print(f"[holdout] delta del candidato: {delta_holdout:+.4f} "
          f"({'confirmado' if delta_holdout > 0 else 'NO confirmado'})", flush=True)

    # ---------------------------------------------------------------- Parte B: candidato
    print("\n[candidato] calculando features supervisadas de test con todo train...", flush=True)
    started_supervised = time.time()
    train_texts = train[cfg.TEXT_COL]
    test_texts = test[cfg.TEXT_COL]
    supervised_test = pd.concat([
        af.supervised_entity_features(train_texts, y, test_texts, "mention"),
        af.supervised_entity_features(train_texts, y, test_texts, "hashtag"),
        af.supervised_entity_features(train_texts, y, test_texts, "rt_source"),
        af.lexical_rarity_features(train_texts, test_texts),
        af.class_lexicon_scores(train_texts, y, test_texts),
    ], axis=1)
    expected_columns = len(open(cfg.FEATURES / "age_supervised_columns.txt").read().splitlines())
    assert supervised_test.shape[1] == expected_columns, (
        f"desalineación de features supervisadas: {supervised_test.shape[1]} vs {expected_columns}")
    supervised_test_values = supervised_test.to_numpy(float)
    print(f"[candidato] {supervised_test_values.shape} features supervisadas de test "
          f"({round(time.time() - started_supervised, 1)}s)", flush=True)

    print("[candidato] ajustando los 13 modelos base sobre todo el train...", flush=True)
    representations = pipeline.build_representations(train, test)
    test_scores, _ = pipeline.fit_base_models(representations, y, dense_train, dense_test,
                                              verbose=True)

    x_meta_test = np.concatenate(test_scores + [dense_test, structured_test,
                                                supervised_test_values], axis=1)
    x_meta_train = np.concatenate([base, dense_train, structured_train, supervised_train], axis=1)
    assert x_meta_train.shape[1] == x_meta_test.shape[1], (
        f"desalineación del meta-modelo: {x_meta_train.shape} vs {x_meta_test.shape}")

    meta = train_meta(x_meta_train, y, np.arange(len(train)))
    predictions = meta.predict(x_meta_test)

    candidate = pd.DataFrame({cfg.ID_COL: test[cfg.ID_COL].to_numpy(), cfg.TARGET: predictions})
    checks = submission.validate_submission(candidate, test[cfg.ID_COL], sample)
    path = cfg.SUBMISSIONS / "age_submission_v09.csv"
    candidate.to_csv(path, index=False)

    results["candidato"] = {
        "nombre": "AGE_ADVANCED_V01",
        "meta_features": int(x_meta_train.shape[1]),
        "checks": checks,
        "distribucion_predicha": candidate[cfg.TARGET].value_counts().reindex(cfg.CLASSES)
                                 .fillna(0).astype(int).to_dict(),
        "archivo": str(path),
        "segundos": round(time.time() - started, 1),
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== ADV-03 — HOLDOUT Y CANDIDATO ===")
    print(json.dumps({k: results[k] for k in ("holdout_congelado",)}, indent=2, ensure_ascii=False)[:700])
    print(f"\nsubmission: {path} | {x_meta_train.shape[1]} features del meta")
    print(f"checks de formato: {all(checks.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
