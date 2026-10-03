"""FASE 10 — Las dos vías autorizadas: retrieval train->test y POS de spaCy.

VÍA A — Retrieval train->test (inferencia, no entrenamiento)
    Para cada fila se busca su **clúster de near-duplicados** (construido sobre texto con Jaccard
    ≥ 0,8, sin etiquetas) y se resume la distribución de etiquetas de los miembros de **train** de
    ese clúster. Cumple las reglas: no entrena con test, no usa etiquetas de test, no ajusta
    umbrales con test (el umbral se fija en validación). Se evalúan dos usos: (i) como features del
    meta-modelo, (ii) como sobrescritura dura solo cuando el clúster es unánime y su miembro de
    train es un duplicado **exacto**.

VÍA B — POS tagger de spaCy (`es_core_news_md`)
    Modelo **no-transformer** (tok2vec convolucional), permitido bajo D-001. Se extraen
    distribuciones de categorías gramaticales, proporciones de clases cerradas y entropía de
    secuencias POS. Limitación conocida: el corpus es ASCII puro, lo que degrada el etiquetado.

Salida: ``experiments/phase10_dos_vias.json``.
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
from phase2_ponderacion import EncodedMLP, build_features, evaluate_global  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "phase10_dos_vias.json"
UPOS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
             "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]


# ======================================================================================
# VÍA A — Retrieval
# ======================================================================================
def retrieval_features(cluster_ids: np.ndarray, y_train: np.ndarray, train_rows: np.ndarray,
                       target_rows: np.ndarray) -> np.ndarray:
    """Resume la distribución de etiquetas de train en el clúster de cada fila objetivo.

    Returns
    -------
    numpy.ndarray
        12 columnas: 6 de la distribución mayoritaria del clúster, + pureza, + tamaño de train del
        clúster, + cobertura, + entropía, + margen, + indicador de clúster unánime.
    """
    lookup: dict[int, dict[str, int]] = {}
    for row in train_rows:
        bucket = lookup.setdefault(int(cluster_ids[row]), {})
        label = y_train[row]
        bucket[label] = bucket.get(label, 0) + 1
    output = np.zeros((len(target_rows), 12), dtype=np.float32)
    for position, row in enumerate(target_rows):
        bucket = lookup.get(int(cluster_ids[row]))
        if not bucket:
            continue
        counts = np.array([bucket.get(label, 0) for label in cfg.CLASSES], dtype=float)
        total = counts.sum()
        share = counts / total
        ordered = np.sort(share)[::-1]
        entropy = float(-(share[share > 0] * np.log2(share[share > 0])).sum())
        output[position] = np.concatenate([share, [ordered[0], total, 1.0, entropy,
                                                   ordered[0] - ordered[1],
                                                   float(len(bucket) == 1)]])
    return output


def exact_duplicate_labels(cluster_ids: np.ndarray, texts: pd.Series, y_train: np.ndarray,
                           train_rows: np.ndarray, target_rows: np.ndarray) -> np.ndarray:
    """Etiqueta recuperada por duplicado **exacto** normalizado (o cadena vacía si no hay)."""
    from src.cv import normalize_for_dedup
    normalized = normalize_for_dedup(texts).to_numpy()
    index: dict[str, list[int]] = {}
    for row in train_rows:
        index.setdefault(normalized[row], []).append(row)
    recovered = np.empty(len(target_rows), dtype=object)
    for position, row in enumerate(target_rows):
        matches = index.get(normalized[row], [])
        if not matches:
            recovered[position] = ""
            continue
        labels = y_train[np.array(matches)]
        values, counts = np.unique(labels, return_counts=True)
        recovered[position] = values[np.argmax(counts)] if counts.max() / len(matches) >= 0.75 else ""
    return recovered


# ======================================================================================
# VÍA B — POS
# ======================================================================================
def pos_features(texts: pd.Series, n_process: int = 12, batch_size: int = 512) -> pd.DataFrame:
    """Distribuciones de categorías gramaticales con spaCy (modelo no-transformer)."""
    import spacy

    nlp = spacy.load("es_core_news_md", exclude=["ner", "lemmatizer"])
    texts_list = texts.fillna("").astype(str).tolist()
    rows: list[dict] = []
    for document in nlp.pipe(texts_list, batch_size=batch_size, n_process=n_process):
        tags = [token.pos_ for token in document if not token.is_space]
        if tags:
            counts = pd.Series(tags).value_counts()
            total = len(tags)
            distribution = {f"pos_{tag}": counts.get(tag, 0) / total for tag in UPOS_TAGS}
            bigrams = list(zip(tags, tags[1:]))
            trigrams = list(zip(tags, tags[1:], tags[2:]))
            unique_tags = len(set(tags))
            rows.append({**distribution,
                         "pos_unique": unique_tags / len(UPOS_TAGS),
                         "pos_entropy": float(-sum((count / total) * np.log2(count / total)
                                                   for count in counts.values)),
                         "pos_bigram_diversity": (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0,
                         "pos_trigram_diversity": (len(set(trigrams)) / len(trigrams)) if trigrams else 0.0,
                         "pos_content_ratio": (counts.get("NOUN", 0) + counts.get("VERB", 0)
                                               + counts.get("ADJ", 0) + counts.get("ADV", 0)) / total,
                         "pos_function_ratio": (counts.get("DET", 0) + counts.get("ADP", 0)
                                                + counts.get("PRON", 0) + counts.get("CONJ", 0)
                                                + counts.get("CCONJ", 0) + counts.get("SCONJ", 0)
                                                + counts.get("AUX", 0)) / total,
                         "pos_verb_noun_ratio": counts.get("VERB", 0) / max(counts.get("NOUN", 0), 1),
                         "pos_adj_noun_ratio": counts.get("ADJ", 0) / max(counts.get("NOUN", 0), 1),
                         "pos_tokens": total})
        else:
            rows.append({**{f"pos_{tag}": 0.0 for tag in UPOS_TAGS}, "pos_unique": 0.0,
                         "pos_entropy": 0.0, "pos_bigram_diversity": 0.0,
                         "pos_trigram_diversity": 0.0, "pos_content_ratio": 0.0,
                         "pos_function_ratio": 0.0, "pos_verb_noun_ratio": 0.0,
                         "pos_adj_noun_ratio": 0.0, "pos_tokens": 0})
    return pd.DataFrame(rows)


def main() -> int:
    """Mide el aporte de retrieval y de POS al combinador."""
    set_seed()
    started = time.time()
    payload: dict = {}
    train = data.load_train()
    test = data.load_test()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()
    cluster_train, cluster_test = cluster_ids[: len(train)], cluster_ids[len(train):]
    folds = cv.make_folds(y, groups=cluster_train, n_splits=5, n_repeats=1, scheme="group")

    features, y2, _, _, _ = build_features()
    features = np.concatenate([features, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    assert np.array_equal(y, y2)
    base_mean, base_std, base_folds = evaluate_global(features, y, folds, "mlp")
    print(f"[base v12] combinador MLP: {base_mean:.4f} ± {base_std:.4f}", flush=True)

    # ---------------------------------------------------------------- VÍA A
    print("\n[A] retrieval train->test (in-fold)...", flush=True)
    retrieval = np.zeros((len(y), 12), dtype=np.float32)
    recovered_labels = np.empty(len(y), dtype=object)
    for fold in folds:
        train_rows, valid_rows = fold["train_idx"], fold["val_idx"]
        retrieval[valid_rows] = retrieval_features(cluster_train, y, train_rows, valid_rows)
        recovered_labels[valid_rows] = exact_duplicate_labels(cluster_train, train[cfg.TEXT_COL],
                                                              y, train_rows, valid_rows)
    covered = retrieval[:, 6] > 0
    exact = recovered_labels != ""
    print(f"    filas con clúster en train: {covered.sum():,} ({100 * covered.mean():.2f}%) | "
          f"duplicado exacto: {exact.sum():,} ({100 * exact.mean():.2f}%)", flush=True)
    if exact.sum() > 0:
        exact_accuracy = float((recovered_labels[exact] == y[exact]).mean())
        print(f"    acierto de la etiqueta recuperada (exactos): {exact_accuracy:.4f}", flush=True)

    with_retrieval = np.concatenate([features, retrieval], axis=1)
    retrieval_mean, retrieval_std, retrieval_folds = evaluate_global(with_retrieval, y, folds, "mlp")
    print(f"    MLP + features de retrieval: {retrieval_mean:.4f} ({retrieval_mean - base_mean:+.4f})",
          flush=True)

    # Sobrescritura dura solo para duplicados exactos
    probability_matrix = np.zeros((len(y), len(cfg.CLASSES)))
    from phase2_ponderacion import make_model
    for fold in folds:
        train_rows = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_rows = fold["val_idx"]
        encoder = EncodedMLP()
        encoder.fit(features[train_rows], y[train_rows])
        probability_matrix[valid_rows] = encoder.model.predict_proba(
            encoder.scaler.transform(features[valid_rows]))
    hard = np.array(cfg.CLASSES)[np.argmax(probability_matrix, axis=1)]
    hard[exact] = recovered_labels[exact]
    hard_macro = float(f1_score(y, hard, average="macro", labels=cfg.CLASSES))
    print(f"    sobrescritura dura en exactos: {hard_macro:.4f} "
          f"({hard_macro - base_mean:+.4f})", flush=True)

    # ---------------------------------------------------------------- VÍA B
    print("\n[B] POS de spaCy sobre train y test...", flush=True)
    pos_train = pos_features(train[cfg.TEXT_COL])
    pos_test = pos_features(test[cfg.TEXT_COL])
    pos_train.to_parquet(cfg.FEATURES / "pos_features.parquet", index=False)
    pos_test.to_parquet(cfg.FEATURES / "pos_features_test.parquet", index=False)
    zero = [c for c in pos_train.columns if pos_train[c].std() == 0]
    print(f"    train {pos_train.shape} | test {pos_test.shape} | columnas constantes: {zero}",
          flush=True)
    with_pos = np.concatenate([features, pos_train.to_numpy(float)], axis=1)
    pos_mean, pos_std, pos_folds = evaluate_global(with_pos, y, folds, "mlp")
    print(f"    MLP + POS: {pos_mean:.4f} ({pos_mean - base_mean:+.4f})", flush=True)

    # ---------------------------------------------------------------- Ambas
    with_both = np.concatenate([features, retrieval, pos_train.to_numpy(float)], axis=1)
    both_mean, both_std, both_folds = evaluate_global(with_both, y, folds, "mlp")
    print(f"    MLP + retrieval + POS: {both_mean:.4f} ({both_mean - base_mean:+.4f})", flush=True)

    payload = {
        "base_v12_mlp": {"macro_f1": base_mean, "std": base_std, "per_fold": base_folds},
        "via_a_retrieval": {
            "cobertura_pct": round(100 * float(covered.mean()), 2),
            "exactos_pct": round(100 * float(exact.mean()), 2),
            "acierto_exactos": round(exact_accuracy, 4) if exact.sum() else None,
            "como_features": {"macro_f1": round(retrieval_mean, 4),
                              "delta": round(retrieval_mean - base_mean, 4),
                              "per_fold": retrieval_folds},
            "sobrescritura_dura": {"macro_f1": round(hard_macro, 4),
                                   "delta": round(hard_macro - base_mean, 4)},
        },
        "via_b_pos": {"n_features": int(pos_train.shape[1]), "constantes": zero,
                      "macro_f1": round(pos_mean, 4), "delta": round(pos_mean - base_mean, 4),
                      "per_fold": pos_folds},
        "ambas": {"macro_f1": round(both_mean, 4), "delta": round(both_mean - base_mean, 4),
                  "per_fold": both_folds},
        "segundos": round(time.time() - started, 1),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
    print("\n=== FASE 10 — DOS VÍAS ===")
    print(f"base v12                  : {base_mean:.4f}")
    print(f"A) retrieval como features: {retrieval_mean:.4f} ({retrieval_mean - base_mean:+.4f})")
    print(f"A) sobrescritura exactos   : {hard_macro:.4f} ({hard_macro - base_mean:+.4f})")
    print(f"B) POS                    : {pos_mean:.4f} ({pos_mean - base_mean:+.4f})")
    print(f"ambas                     : {both_mean:.4f} ({both_mean - base_mean:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
