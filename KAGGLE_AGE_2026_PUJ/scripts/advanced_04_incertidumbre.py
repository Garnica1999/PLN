"""ADV-04 — Features de incertidumbre, acuerdo e interacción para el meta-modelo.

Observación (brief §38, §70)
    El meta-modelo es **lineal**: ve los scores de los 13 miembros y las features densas, pero no
    puede derivar por sí solo cantidades **no lineales** de esos scores (entropía, margen, grado de
    acuerdo). Tampoco puede expresar modulaciones del tipo "confiar más en este modelo cuando el
    tweet es corto".

Hipótesis
    H25  Añadir transformaciones no lineales de los scores (confianza, margen, entropía por miembro;
         acuerdo y varianza entre miembros) aporta información que el meta-modelo lineal no puede
         construir por sí mismo.
    H26  Las interacciones score × tipo de tweet (longitud, retuit, menciones) permiten el
         enrutamiento implícito que pide el brief (§38, §73).

Protocolo
    Leave-one-fold-out sobre las mismas particiones agrupadas. Se evalúa: baseline v09 (267) →
    +incertidumbre → +interacciones → todo. Sin reentrenar los modelos base.

Salida: ``experiments/advanced_04_incertidumbre.json``.
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
from src import cv, data, textometry  # noqa: E402
from src.config import set_seed  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "advanced_04_incertidumbre.json"
SOURCES = [("oof_E013_margenes.npy", False), ("oof_E016_margenes.npy", False),
           ("oof_E-009.npy", True), ("oof_E-011.npy", True), ("oof_E-010.npy", True),
           ("oof_E014_margenes.npy", False), ("oof_E015_ordinal.npy", False),
           ("oof_E015_softmax.npy", True), ("oof_E017_probabilidades.npy", True),
           ("oof_E019b_densas_svd.npy", True), ("oof_E021a_ordinal.npy", True),
           ("oof_E021b_rocchio.npy", True), ("oof_E028a_pares.npy", False)]
DENSE_FEATURES = [name for name in textometry.TEXTOMETRY_FEATURES + textometry.STYLE_FEATURES
                  if name not in {"n_emoji", "emoji_diversity", "has_emoji"}]


def softmax(matrix: np.ndarray) -> np.ndarray:
    """Softmax por filas (con desplazamiento numérico estable)."""
    shifted = matrix - matrix.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def uncertainty_features(scores: list[np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Transformaciones no lineales de los scores de los miembros.

    Por miembro: confianza (probabilidad máxima), margen (top-1 − top-2) y entropía.
    Entre miembros: tasa de acuerdo, reparto de votos por clase, varianza de scores por clase y
    estadísticos de la confianza y de la entropía.

    Nota: las fuentes con una sola columna (la predicción ordinal continua) se excluyen de este
    bloque porque no expresan una distribución sobre clases; su información ya entra al meta-modelo
    como valor continuo y mediante la fuente derivada de softmax.
    """
    class_scores = [matrix for matrix in scores if matrix.shape[1] == len(cfg.CLASSES)]
    probabilities = [softmax(matrix) for matrix in class_scores]
    blocks: list[np.ndarray] = []
    names: list[str] = []

    for index, probability in enumerate(probabilities):
        ordered = np.sort(probability, axis=1)
        confidence = ordered[:, -1]
        margin = ordered[:, -1] - ordered[:, -2]
        entropy = -(probability * np.log2(np.clip(probability, 1e-12, None))).sum(axis=1)
        block = np.column_stack([confidence, margin, entropy])
        blocks.append(block)
        names += [f"m{index}_conf", f"m{index}_margin", f"m{index}_entropy"]

    votes = np.stack([np.argmax(probability, axis=1) for probability in probabilities], axis=0)
    n_members = votes.shape[0]
    vote_share = np.stack([(votes == class_index).mean(axis=0) for class_index in range(len(cfg.CLASSES))],
                          axis=1)
    blocks += [vote_share,
               vote_share.max(axis=1, keepdims=True),
               (1 - vote_share.max(axis=1, keepdims=True)),
               np.stack([probability for probability in probabilities], axis=0).std(axis=0)]
    names += [f"vote_share_c{i}" for i in range(len(cfg.CLASSES))]
    names += ["vote_plurality", "vote_disagreement"]
    names += [f"score_std_c{i}" for i in range(len(cfg.CLASSES))]

    confidences = np.stack([np.sort(probability, axis=1)[:, -1] for probability in probabilities],
                           axis=1)
    entropies = np.stack([-(probability * np.log2(np.clip(probability, 1e-12, None))).sum(axis=1)
                          for probability in probabilities], axis=1)
    blocks += [confidences.mean(axis=1, keepdims=True), confidences.std(axis=1, keepdims=True),
               entropies.mean(axis=1, keepdims=True), entropies.std(axis=1, keepdims=True)]
    names += ["conf_mean", "conf_std", "entropy_mean", "entropy_std"]
    return np.hstack(blocks), names


def interaction_features(scores: list[np.ndarray], dense: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Interacciones escogidas score × tipo de tweet (brief §38, §Z)."""
    ordinal = np.load(cfg.FEATURES / "oof_E015_ordinal.npy").reshape(-1, 1)
    confidence = np.sort(softmax(scores[0]), axis=1)[:, -1:]
    n_chars = dense[:, DENSE_FEATURES.index("n_chars")].reshape(-1, 1)
    n_mention = dense[:, DENSE_FEATURES.index("n_mention")].reshape(-1, 1)
    is_rt = dense[:, DENSE_FEATURES.index("is_rt")].reshape(-1, 1)
    mention_profile = np.load(cfg.FEATURES / "age_supervised_features.npy")[:, :len(cfg.CLASSES)]
    mention_score = mention_profile.max(axis=1, keepdims=True)
    block = np.hstack([n_chars * ordinal, n_chars * confidence, is_rt * ordinal,
                       n_mention * mention_score, n_chars * mention_score])
    names = ["ix_len_x_ordinal", "ix_len_x_conf", "ix_rt_x_ordinal",
             "ix_mentions_x_mention_score", "ix_len_x_mention_score"]
    return block, names


def evaluate(features: np.ndarray, y: np.ndarray, folds: list[dict]) -> tuple[float, float, list]:
    """macro-F1 con meta-modelo LogReg y leave-one-fold-out."""
    scores = []
    for fold in folds:
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        meta = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                    random_state=cfg.SEED))])
        meta.fit(features[train_idx], y[train_idx])
        scores.append(float(f1_score(y[valid_idx], meta.predict(features[valid_idx]),
                                     average="macro", labels=cfg.CLASSES)))
    return float(np.mean(scores)), float(np.std(scores)), [round(value, 4) for value in scores]


def main() -> int:
    """Evalúa el aporte de incertidumbre e interacciones sobre el candidato v09."""
    set_seed()
    started = time.time()
    train = data.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    folds = cv.make_folds(y, groups=clusters["cluster"].to_numpy()[: len(train)],
                          n_splits=5, n_repeats=1, scheme="group")

    dense = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")[DENSE_FEATURES].to_numpy(float)
    structured = pd.read_parquet(cfg.FEATURES / "age_structured_features.parquet").to_numpy(float)
    supervised = np.load(cfg.FEATURES / "age_supervised_features.npy")

    matrices, raw = [], []
    for filename, probabilistic in SOURCES:
        matrix = np.load(cfg.FEATURES / filename)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        raw.append(matrix)
        matrices.append(np.log(np.clip(matrix, 1e-12, None)) if probabilistic else matrix)
    base = np.concatenate(matrices, axis=1)

    uncertainty, uncertainty_names = uncertainty_features(raw)
    interactions, interaction_names = interaction_features(raw, dense)
    print(f"incertidumbre: {uncertainty.shape[1]} features | interacciones: {interactions.shape[1]}",
          flush=True)

    variants = {
        "v09 (267)": np.concatenate([base, dense, structured, supervised], axis=1),
        "+ incertidumbre": np.concatenate([base, dense, structured, supervised, uncertainty], axis=1),
        "+ interacciones": np.concatenate([base, dense, structured, supervised, uncertainty,
                                           interactions], axis=1),
    }
    rows = []
    for label, features in variants.items():
        mean, std, per_fold = evaluate(features, y, folds)
        rows.append({"variante": label, "n_features": features.shape[1],
                     "macro_f1": round(mean, 4), "std": round(std, 4), "per_fold": per_fold})
        print(f"[{label:18s}] {mean:.4f} ± {std:.4f} ({features.shape[1]} features) "
              f"folds={per_fold}", flush=True)

    table = pd.DataFrame(rows)
    table.drop(columns=["per_fold"]).to_csv(cfg.TABLES / "advanced_04_variantes.csv", index=False)
    baseline = rows[0]["macro_f1"]
    best = max(rows, key=lambda row: row["macro_f1"])
    payload = {
        "variantes": [{k: v for k, v in row.items()} for row in rows],
        "mejor": best["variante"], "mejor_macro_f1": best["macro_f1"],
        "aporte_incertidumbre": round(rows[1]["macro_f1"] - baseline, 4),
        "aporte_interacciones": round(rows[2]["macro_f1"] - rows[1]["macro_f1"], 4),
        "aporte_total": round(best["macro_f1"] - baseline, 4),
        "segundos": round(time.time() - started, 1),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== ADV-04 — INCERTIDUMBRE E INTERACCIONES ===")
    print(table.drop(columns=["per_fold"]).to_string(index=False))
    print(f"\naporte de incertidumbre: {payload['aporte_incertidumbre']:+.4f}")
    print(f"aporte de interacciones: {payload['aporte_interacciones']:+.4f}")
    print(f"aporte total: {payload['aporte_total']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
