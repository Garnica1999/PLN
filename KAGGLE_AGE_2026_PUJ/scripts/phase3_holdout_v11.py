"""FASE 3 — Confirmación en holdout congelado y candidato v11 (meta-modelo MLP).

Hallazgo que se confirma (Fase 2 / L6)
    El meta-modelo **MLP superficial** sobre las 273 features del candidato v10 obtiene
    **0,5966 ± 0,0040** frente a **0,5906 ± 0,0047** del meta-modelo LogReg, con mejora en **5/5
    folds** (+0,0043 a +0,0080). La segmentación dura falla (tipo −0,0007; longitud −0,0015): lo que
    funciona es la **ponderación condicional suave**, que es justo lo que el diagnóstico L1 sugería
    (25,94% de filas donde un miembro acierta y el ensamble falla, rescatadas sobre todo por los
    modelos diversos: softmax ordinal 36%, centroides 31%, MLP 31%).

Protocolo
    1. **Holdout congelado** (15% de clústeres, semilla 2026, el mismo de ADV-03): el meta-modelo
       se entrena solo con el 85% de desarrollo y se evalúa en el holdout. Criterio de adopción:
       Δ ≥ +0,003 **y** mejora en al menos 4/5 folds del CV.
    2. Si se confirma, se generan las features del lado test con todo el train (sin etiquetas de
       test) y se produce la submission v11. Las matrices del meta se guardan en disco para que
       futuros experimentos no repitan los 12 minutos de cómputo.

Salida: ``experiments/phase3_holdout_v11.json`` y ``submissions/age_submission_v11.csv``.
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
from src import data, submission  # noqa: E402
from src.config import set_seed  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from phase2_ponderacion import EncodedMLP, build_features, make_model  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "phase3_holdout_v11.json"
HOLDOUT_SEED = 2026
HOLDOUT_SHARE = 0.15


def main() -> int:
    """Confirma el meta-modelo MLP en el holdout y genera el candidato v11."""
    set_seed()
    started = time.time()
    payload: dict = {}

    features, y, dense, tweet_type, length_band = build_features()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(y)]

    # ---------------------------------------------------------------- Holdout congelado
    rng = np.random.default_rng(HOLDOUT_SEED)
    unique_clusters = np.unique(cluster_ids)
    holdout_clusters = set(rng.choice(unique_clusters,
                                      size=int(len(unique_clusters) * HOLDOUT_SHARE),
                                      replace=False).tolist())
    is_holdout = np.array([cluster in holdout_clusters for cluster in cluster_ids])
    development_idx = np.where(~is_holdout)[0]
    holdout_idx = np.where(is_holdout)[0]
    print(f"[holdout] desarrollo {len(development_idx):,} | holdout {len(holdout_idx):,}", flush=True)

    holdout_results = {}
    for label, kind in (("meta LogReg (v10)", "logreg"), ("meta MLP (v11)", "mlp")):
        meta = make_model(kind)
        meta.fit(features[development_idx], y[development_idx])
        predictions = meta.predict(features[holdout_idx])
        macro = float(f1_score(y[holdout_idx], predictions, average="macro", labels=cfg.CLASSES))
        holdout_results[label] = {
            "macro_f1": round(macro, 4),
            "accuracy": round(float((predictions == y[holdout_idx]).mean()), 4),
        }
        print(f"[holdout] {label:20s}: macro-F1 {macro:.4f} | "
              f"accuracy {holdout_results[label]['accuracy']:.4f}", flush=True)

    delta_holdout = (holdout_results["meta MLP (v11)"]["macro_f1"]
                     - holdout_results["meta LogReg (v10)"]["macro_f1"])
    confirmed = delta_holdout >= 0.003
    payload["holdout"] = {"resultados": holdout_results, "delta": round(delta_holdout, 4),
                          "criterio": "Δ ≥ +0,003", "confirmado": bool(confirmed)}
    print(f"[holdout] Δ = {delta_holdout:+.4f} -> "
          f"{'CONFIRMADO' if confirmed else 'NO confirmado'}", flush=True)

    # ---------------------------------------------------------------- Candidato v11
    if not confirmed:
        print("\nNo se genera submission: el holdout no confirma el criterio.", flush=True)
        RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                                encoding="utf-8")
        return 0

    print("\n[candidato] construyendo features del lado test (todo train)...", flush=True)
    from scripts.advanced_06_submission_v10 import (nb_log_count_ratio, tokens_of, CHAR_CONFIG,
                                                    MENTION_CONFIG, DENSE_FEATURES)
    from src import advanced_features as af, pipeline, textometry
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    train = data.load_train()
    test = data.load_test()
    sample = data.load_sample_submission()
    dense_train = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")[DENSE_FEATURES].to_numpy(float)
    dense_test = pd.read_parquet(cfg.FEATURES / "dense_test.parquet")[DENSE_FEATURES].to_numpy(float)
    structured_test = pd.read_parquet(cfg.FEATURES / "age_structured_features_test.parquet").to_numpy(float)

    char = TfidfVectorizer(**CHAR_CONFIG)
    mentions = TfidfVectorizer(**MENTION_CONFIG)
    x_train = sp.hstack([char.fit_transform(train[cfg.TEXT_COL]),
                         mentions.fit_transform(tokens_of(train[cfg.TEXT_COL], r"@\w+"))]).tocsr()
    x_test = sp.hstack([char.transform(test[cfg.TEXT_COL]),
                        mentions.transform(tokens_of(test[cfg.TEXT_COL], r"@\w+"))]).tocsr()
    nb_test = np.zeros((len(test), len(cfg.CLASSES)), dtype=np.float32)
    for position, label in enumerate(cfg.CLASSES):
        binary_target = (y == label).astype(int)
        ratio = nb_log_count_ratio(x_train, binary_target)
        model = LinearSVC(C=0.2, max_iter=3000)
        model.fit(x_train.multiply(ratio).tocsr(), binary_target)
        nb_test[:, position] = model.decision_function(x_test.multiply(ratio).tocsr())
    print(f"    NB-SVM de test listo ({round(time.time() - started, 1)}s)", flush=True)

    supervised_test = pd.concat([
        af.supervised_entity_features(train[cfg.TEXT_COL], y, test[cfg.TEXT_COL], "mention"),
        af.supervised_entity_features(train[cfg.TEXT_COL], y, test[cfg.TEXT_COL], "hashtag"),
        af.supervised_entity_features(train[cfg.TEXT_COL], y, test[cfg.TEXT_COL], "rt_source"),
        af.lexical_rarity_features(train[cfg.TEXT_COL], test[cfg.TEXT_COL]),
        af.class_lexicon_scores(train[cfg.TEXT_COL], y, test[cfg.TEXT_COL]),
    ], axis=1).to_numpy(float)

    representations = pipeline.build_representations(train, test)
    test_scores, _ = pipeline.fit_base_models(representations, y, dense_train, dense_test,
                                              verbose=True)
    # Orden idéntico al lado train de `phase2_ponderacion.build_features`:
    # [13 scores del pipeline] + [NB-SVM] + densas + estructuradas + supervisadas.
    x_meta_test = np.concatenate(test_scores + [nb_test, dense_test, structured_test,
                                                supervised_test], axis=1)
    assert x_meta_test.shape[1] == features.shape[1], (
        f"desalineación: {x_meta_test.shape} vs {features.shape}")
    np.save(cfg.FEATURES / "meta_features_test_v11.npy", x_meta_test)
    np.save(cfg.FEATURES / "meta_features_train_v11.npy", features)
    print(f"    matrices del meta guardadas ({round(time.time() - started, 1)}s)", flush=True)

    meta = make_model("mlp")
    meta.fit(features, y)
    predictions = meta.predict(x_meta_test)

    candidate = pd.DataFrame({cfg.ID_COL: test[cfg.ID_COL].to_numpy(), cfg.TARGET: predictions})
    checks = submission.validate_submission(candidate, test[cfg.ID_COL], sample)
    path = cfg.SUBMISSIONS / "age_submission_v11.csv"
    candidate.to_csv(path, index=False)

    payload["candidato"] = {
        "version": 11,
        "modelo": "Ensamble de 14 fuentes + meta-modelo MLP(64) sobre 273 features",
        "cv_macro_f1": 0.5966, "cv_std": 0.0040,
        "holdout_macro_f1": holdout_results["meta MLP (v11)"]["macro_f1"],
        "checks": checks,
        "distribucion_predicha": candidate[cfg.TARGET].value_counts().reindex(cfg.CLASSES)
                                 .fillna(0).astype(int).to_dict(),
        "archivo": str(path),
        "segundos": round(time.time() - started, 1),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
    print(json.dumps(payload["candidato"], indent=2, ensure_ascii=False)[:900])
    print(f"\narchivo listo: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
