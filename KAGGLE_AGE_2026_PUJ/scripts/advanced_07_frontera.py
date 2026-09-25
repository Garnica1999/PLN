"""ADV-07 — Investigación de la frontera 18-24 | 25-34 (el punto de mayor error).

Motivación (Fase 17 y E-028a)
    Es la frontera más difícil del problema: un SVM entrenado **exclusivamente** con las dos clases
    solo acierta **0,6856**, y la confusión 18-24↔25-34 es la mayor de la matriz (15,3% y 12,5%).
    Las clases implicadas tienen los F1 más bajos (`25-34` 0,449; `18-24` 0,476).

Parte A — Diagnóstico
    Con las predicciones out-of-fold del ensamble se caracteriza la frontera: desempeño actual,
    hacia dónde se equivoca y qué perfil tienen los tweets que falla (longitud, menciones, retuit,
    rareza léxica, tipo de tweet).

Parte B — Especialista con compuerta (brief §73)
    Arquitectura "modelo global → especialista local": cuando el top-2 del ensamble cae en un par
    adyacente, un clasificador especializado —entrenado solo con las filas de ese par— decide entre
    los dos. La compuerta usa **solo la predicción** del modelo global (nunca la etiqueta).

Protocolo
    Todo con leave-one-fold-out: el especialista de cada fold se entrena con los otros cuatro.

Salida: ``experiments/advanced_07_frontera.json``.
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

RESULTS_PATH = cfg.EXPERIMENTS / "advanced_07_frontera.json"
SOURCES = [("oof_E013_margenes.npy", False), ("oof_E016_margenes.npy", False),
           ("oof_E-009.npy", True), ("oof_E-011.npy", True), ("oof_E-010.npy", True),
           ("oof_E014_margenes.npy", False), ("oof_E015_ordinal.npy", False),
           ("oof_E015_softmax.npy", True), ("oof_E017_probabilidades.npy", True),
           ("oof_E019b_densas_svd.npy", True), ("oof_E021a_ordinal.npy", True),
           ("oof_E021b_rocchio.npy", True), ("oof_E028a_pares.npy", False),
           ("oof_ADV05_nbsvm.npy", False)]
DENSE_FEATURES = [name for name in textometry.TEXTOMETRY_FEATURES + textometry.STYLE_FEATURES
                  if name not in {"n_emoji", "emoji_diversity", "has_emoji"}]
#: Fronteras adyacentes que la matriz de confusión señala como problemáticas.
FRONTIERS = [("18-24", "25-34"), ("25-34", "35-49"), ("35-49", "50-64")]


def build_meta_features() -> tuple[np.ndarray, np.ndarray]:
    """Construye la matriz de features del meta-modelo (candidato v10) y las etiquetas."""
    train = data.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    dense = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")[DENSE_FEATURES].to_numpy(float)
    structured = pd.read_parquet(cfg.FEATURES / "age_structured_features.parquet").to_numpy(float)
    supervised = np.load(cfg.FEATURES / "age_supervised_features.npy")
    matrices = []
    for filename, probabilistic in SOURCES:
        matrix = np.load(cfg.FEATURES / filename)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        matrices.append(np.log(np.clip(matrix, 1e-12, None)) if probabilistic else matrix)
    features = np.concatenate([np.concatenate(matrices, axis=1), dense, structured, supervised],
                              axis=1)
    return features, y


def make_meta() -> Pipeline:
    """Meta-modelo estándar del proyecto."""
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                random_state=cfg.SEED))])


def lofo_probabilities(features: np.ndarray, y: np.ndarray,
                       folds: list[dict]) -> np.ndarray:
    """Probabilidades out-of-fold del meta-modelo (leave-one-fold-out)."""
    probabilities = np.zeros((len(y), len(cfg.CLASSES)))
    for fold in folds:
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        meta = make_meta()
        meta.fit(features[train_idx], y[train_idx])
        probabilities[valid_idx] = meta.predict_proba(features[valid_idx])
    return probabilities


def main() -> int:
    """Ejecuta el diagnóstico de la frontera y el especialista con compuerta."""
    set_seed()
    started = time.time()
    train = data.load_train()
    dense = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")
    features, y = build_meta_features()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    folds = cv.make_folds(y, groups=clusters["cluster"].to_numpy()[: len(train)],
                          n_splits=5, n_repeats=1, scheme="group")
    payload: dict = {}

    print("[A] probabilidades out-of-fold del ensamble...", flush=True)
    probabilities = lofo_probabilities(features, y, folds)
    predictions = np.array(cfg.CLASSES)[np.argmax(probabilities, axis=1)]
    top2 = np.argsort(-probabilities, axis=1)[:, :2]
    baseline_macro = float(f1_score(y, predictions, average="macro", labels=cfg.CLASSES))
    print(f"    macro-F1 del ensamble: {baseline_macro:.4f}", flush=True)

    # ---------------------------------------------------------------- Diagnóstico de la frontera
    diagnosis: dict = {"macro_f1_ensamble": round(baseline_macro, 4), "fronteras": {}}
    for lower, upper in FRONTIERS:
        mask = np.isin(y, [lower, upper])
        accuracy = float((predictions[mask] == y[mask]).mean())
        confusion = {}
        for true_label in (lower, upper):
            sub = y == true_label
            for predicted_label in (lower, upper):
                if true_label == predicted_label:
                    continue
                confusion[f"{true_label}->{predicted_label}"] = float(
                    (predictions[sub] == predicted_label).mean())
        # Perfil de los aciertos frente a los errores en la frontera.
        frame = pd.DataFrame({
            "n_chars": dense["n_chars"].to_numpy()[mask],
            "n_words": dense["n_words"].to_numpy()[mask],
            "n_mention": dense["n_mention"].to_numpy()[mask],
            "is_rt": dense["is_rt"].to_numpy()[mask],
            "n_hashtag": dense["n_hashtag"].to_numpy()[mask],
            "n_url": dense["n_url"].to_numpy()[mask],
            "ttr": dense["ttr"].to_numpy()[mask],
            "upper_ratio": dense["upper_ratio"].to_numpy()[mask],
            "correct": (predictions[mask] == y[mask]).astype(int),
        })
        profile = frame.groupby("correct")[["n_chars", "n_words", "n_mention", "is_rt", "n_hashtag",
                                            "n_url", "ttr", "upper_ratio"]].mean()
        diagnosis["fronteras"][f"{lower}|{upper}"] = {
            "n_filas": int(mask.sum()),
            "accuracy_ensamble": round(accuracy, 4),
            "confusion": {key: round(value, 4) for key, value in confusion.items()},
            "perfil_acierto": profile.loc[1].round(3).to_dict() if 1 in profile.index else {},
            "perfil_error": profile.loc[0].round(3).to_dict() if 0 in profile.index else {},
        }
        print(f"    {lower}|{upper}: n={int(mask.sum()):,} accuracy={accuracy:.4f} "
              f"({confusion})", flush=True)

    # ---------------------------------------------------------------- Especialista con compuerta
    print("\n[B] especialista con compuerta sobre el top-2 del ensamble...", flush=True)
    specialist_results: dict = {}
    corrected = predictions.copy()
    gate_counts = {f"{a}|{b}": 0 for a, b in FRONTIERS}
    for fold in folds:
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        for lower, upper in FRONTIERS:
            pair = [lower, upper]
            # Compuerta: filas de validación cuyo top-2 del ensamble cae en el par.
            gate = np.array([set(np.array(cfg.CLASSES)[top2[index]]) == set(pair)
                             for index in valid_idx])
            if gate.sum() == 0:
                continue
            gate_indices = valid_idx[gate]
            # El especialista se entrena SOLO con las filas del par en el train del fold.
            train_mask = np.isin(y[train_idx], pair)
            if train_mask.sum() < 500:
                continue
            specialist = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                                            random_state=cfg.SEED)
            specialist.fit(features[train_idx][train_mask],
                           (y[train_idx][train_mask] == upper).astype(int))
            decision = specialist.predict(features[gate_indices])
            corrected[gate_indices] = np.where(decision == 1, upper, lower)
            gate_counts[f"{lower}|{upper}"] += int(gate.sum())
        print(f"    fold {fold['fold']} procesado ({round(time.time() - started, 1)}s)", flush=True)

    corrected_macro = float(f1_score(y, corrected, average="macro", labels=cfg.CLASSES))
    changed = int((corrected != predictions).sum())
    fixed = int(((corrected == y) & (predictions != y)).sum())
    broken = int(((corrected != y) & (predictions == y)).sum())
    specialist_results = {
        "macro_f1_con_especialista": round(corrected_macro, 4),
        "delta": round(corrected_macro - baseline_macro, 4),
        "filas_reclasificadas": changed,
        "errores_corregidos": fixed,
        "errores_introducidos": broken,
        "compuertas_activadas_por_frontera": gate_counts,
        "macro_f1_por_fold": [
            round(float(f1_score(y[fold["val_idx"]], corrected[fold["val_idx"]], average="macro",
                                 labels=cfg.CLASSES)), 4) for fold in folds],
    }
    print(f"    macro-F1: {baseline_macro:.4f} -> {corrected_macro:.4f} "
          f"({corrected_macro - baseline_macro:+.4f}) | reclasificadas {changed:,} "
          f"(+{fixed:,} corregidas, -{broken:,} introducidas)", flush=True)

    payload["diagnostico"] = diagnosis
    payload["especialista"] = specialist_results
    payload["segundos"] = round(time.time() - started, 1)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== ADV-07 — FRONTERA 18-24 | 25-34 ===")
    for name, data_frontier in diagnosis["fronteras"].items():
        print(f"{name}: n={data_frontier['n_filas']:,} accuracy={data_frontier['accuracy_ensamble']}")
        print(f"    perfil de ERROR  : {data_frontier['perfil_error']}")
        print(f"    perfil de ACIERTO: {data_frontier['perfil_acierto']}")
    print(f"\nespecialista: {specialist_results['delta']:+.4f} "
          f"(+{specialist_results['errores_corregidos']} / -{specialist_results['errores_introducidos']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
