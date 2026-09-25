"""FASE 1 / L1 + L5 — Intersección de errores y umbral de información.

Preguntas que responde
    L1 (brief §24-25): ¿cuántos tweets tienen **al menos un modelo base acertando mientras el
    ensamble falla**? Si esa fracción es alta, el fallo es de **ponderación** (y el meta-modelo
    puede arreglarse); si es baja, el fallo es de **información** (y hay que buscar representación).
    L5 (brief §19, §26, §27): ¿cómo varía el error con el **tipo estructural** de tweet y con la
    **longitud fina**? ¿Existe un umbral de información?

Salidas
    - ``reports/error_intersection_analysis.csv`` (§24): una fila por tweet con su categoría.
    - ``reports/tables/phase1_interseccion_resumen.csv`` y ``phase1_error_por_tipo.csv``.
    - ``experiments/phase1_diagnostico.json``.
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

RESULTS_PATH = cfg.EXPERIMENTS / "phase1_diagnostico.json"
#: (nombre, archivo, transformación). ``log`` solo para probabilidades; los **márgenes** van
#: en crudo porque toman valores negativos (aplicarles log destruiría su información).
MEMBER_FILES = [("SVM_300k", "oof_E013_margenes.npy", "raw"),
                ("SVM_700k", "oof_E016_margenes.npy", "raw"),
                ("LogReg", "oof_E-009.npy", "log"), ("SGD", "oof_E-011.npy", "log"),
                ("ComplementNB", "oof_E-010.npy", "log"),
                ("SVM_hashtags", "oof_E014_margenes.npy", "raw"),
                ("Ridge_ordinal", "oof_E015_ordinal.npy", "raw"),
                ("softmax_ordinal", "oof_E015_softmax.npy", "log"),
                ("LogReg_palabra", "oof_E017_probabilidades.npy", "log"),
                ("MLP_densas_svd", "oof_E019b_densas_svd.npy", "log"),
                ("ordinal_acumulativo", "oof_E021a_ordinal.npy", "log"),
                ("centroides", "oof_E021b_rocchio.npy", "log"),
                ("pares_adyacentes", "oof_E028a_pares.npy", "raw"),
                ("NB_SVM", "oof_ADV05_nbsvm.npy", "raw")]
DENSE_FEATURES = [name for name in textometry.TEXTOMETRY_FEATURES + textometry.STYLE_FEATURES
                  if name not in {"n_emoji", "emoji_diversity", "has_emoji"}]
LENGTH_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 141]


def classify_tweet_type(frame: pd.DataFrame) -> np.ndarray:
    """Taxonomía estructural de tweets con umbrales justificados por la distribución.

    Reglas (en orden de prioridad, todas verificables con las features existentes):
      - ``rt_puro``: empieza por RT y no tiene texto propio más allá de la mención fuente.
      - ``rt_con_comentario``: RT con texto añadido por el usuario (≥ 80 caracteres tras el RT).
      - ``solo_url``: el tweet es prácticamente solo un enlace.
      - ``solo_mencion``: el tweet es prácticamente solo menciones.
      - ``mencion_alta``: ≥ 3 menciones.
      - ``hashtag_alto``: ≥ 3 hashtags.
      - ``url_alta``: ≥ 1 URL con texto.
      - ``solo_texto``: sin entidades.
    """
    length = frame["n_chars"].to_numpy()
    mentions = frame["n_mention"].to_numpy()
    hashtags = frame["n_hashtag"].to_numpy()
    urls = frame["n_url"].to_numpy()
    is_rt = frame["is_rt"].to_numpy()
    words = frame["n_words"].to_numpy()

    types = np.full(len(frame), "solo_texto", dtype=object)
    types[(mentions + hashtags + urls == 0)] = "solo_texto"
    types[(urls >= 1) & (words <= 3)] = "solo_url"
    types[(mentions >= 2) & (words <= 3)] = "solo_mencion"
    types[(urls >= 1) & (words > 3)] = "url_alta"
    types[hashtags >= 3] = "hashtag_alto"
    types[mentions >= 3] = "mencion_alta"
    types[(is_rt == 1) & (length >= 80)] = "rt_con_comentario"
    types[(is_rt == 1) & (length < 80)] = "rt_puro"
    types[length <= 15] = "muy_corto"
    return types


def main() -> int:
    """Ejecuta el diagnóstico de intersección de errores y de tipos de tweet."""
    set_seed()
    started = time.time()
    train = data.load_train()
    dense = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")
    y = train[cfg.TARGET].astype(str).to_numpy()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    folds = cv.make_folds(y, groups=clusters["cluster"].to_numpy()[: len(train)],
                          n_splits=5, n_repeats=1, scheme="group")

    # --- Matrices de los miembros (predicción argmax de cada uno) ---
    member_predictions: dict[str, np.ndarray] = {}
    meta_parts = []
    for name, filename, transform in MEMBER_FILES:
        matrix = np.load(cfg.FEATURES / filename)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        if matrix.shape[1] == len(cfg.CLASSES):
            member_predictions[name] = np.array(cfg.CLASSES)[np.argmax(matrix, axis=1)]
        # el Ridge ordinal continuo no produce predicción de clase directa
        meta_parts.append(np.log(np.clip(matrix, 1e-12, None)) if transform == "log" else matrix)
    print(f"miembros con predicción de clase: {len(member_predictions)}", flush=True)

    dense_values = dense[DENSE_FEATURES].to_numpy(float)
    structured = pd.read_parquet(cfg.FEATURES / "age_structured_features.parquet").to_numpy(float)
    supervised = np.load(cfg.FEATURES / "age_supervised_features.npy")
    features = np.concatenate([np.concatenate(meta_parts, axis=1), dense_values, structured,
                               supervised], axis=1)

    # --- Predicciones OOF del ensamble ---
    print("calculando predicciones out-of-fold del ensamble...", flush=True)
    ensemble = np.empty(len(y), dtype=object)
    for fold in folds:
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        meta = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                    random_state=cfg.SEED))])
        meta.fit(features[train_idx], y[train_idx])
        ensemble[valid_idx] = meta.predict(features[valid_idx])
    ensemble = ensemble.astype(str)
    ensemble_macro = float(f1_score(y, ensemble, average="macro", labels=cfg.CLASSES))
    print(f"macro-F1 del ensamble: {ensemble_macro:.4f} ({round(time.time() - started, 1)}s)",
          flush=True)

    # --- L1: categorías de intersección ---
    correctness = np.stack([(member_predictions[name] == y).astype(int)
                            for name in member_predictions], axis=1)
    models_right = correctness.sum(axis=1)
    ensemble_right = (ensemble == y).astype(int)

    categories = np.full(len(y), "few_models_right", dtype=object)
    categories[models_right == 0] = "everyone_wrong"
    categories[models_right == 1] = "one_model_right"
    categories[(models_right >= 2) & (models_right <= 4)] = "few_models_right"
    categories[models_right >= 5] = "many_models_right"

    type_array = classify_tweet_type(dense)
    frame = pd.DataFrame({
        "true_class": y, "ensemble_prediction": ensemble,
        "models_right": models_right, "ensemble_right": ensemble_right,
        "category": categories,
        "menciones_que_aciertan": [",".join(np.array(list(member_predictions))[correctness[i] == 1])
                                   for i in range(len(y))],
        "n_chars": dense["n_chars"].to_numpy(), "tweet_type": type_array,
        "n_mention": dense["n_mention"].to_numpy(), "n_hashtag": dense["n_hashtag"].to_numpy(),
        "n_url": dense["n_url"].to_numpy(), "is_rt": dense["is_rt"].to_numpy(),
        "ttr": dense["ttr"].to_numpy(),
        "ensemble_miss": ((ensemble_right == 0) & (models_right >= 1)).astype(int),
    })
    frame.to_csv(cfg.REPORTS / "error_intersection_analysis.csv", index=False)
    print(f"[L1] reports/error_intersection_analysis.csv ({len(frame):,} filas)", flush=True)

    summary_rows = []
    for category in ("everyone_wrong", "one_model_right", "few_models_right", "many_models_right"):
        mask = frame["category"] == category
        summary_rows.append({
            "categoria": category, "n": int(mask.sum()),
            "pct": round(100 * float(mask.mean()), 2),
            "accuracy_ensamble": round(float(frame.loc[mask, "ensemble_right"].mean()), 4),
            "n_chars_medio": round(float(frame.loc[mask, "n_chars"].mean()), 1),
            "pct_rt": round(100 * float(frame.loc[mask, "is_rt"].mean()), 1),
            "menciones_medio": round(float(frame.loc[mask, "n_mention"].mean()), 2),
            "pct_solo_texto": round(100 * float((frame.loc[mask, "tweet_type"] == "solo_texto").mean()), 1),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(cfg.TABLES / "phase1_interseccion_resumen.csv", index=False)

    misses = frame[frame["ensemble_miss"] == 1]
    miss_rate = float(len(misses)) / len(frame)
    print(f"[L1] 'el ensamble falla y algún modelo acierta': {len(misses):,} filas "
          f"({100 * miss_rate:.2f}%)", flush=True)

    # Qué modelo acierta en esos casos
    model_rescue: dict[str, int] = {}
    for column in misses["menciones_que_aciertan"]:
        for name in column.split(","):
            if name:
                model_rescue[name] = model_rescue.get(name, 0) + 1
    rescue_table = pd.DataFrame(sorted(model_rescue.items(), key=lambda item: -item[1]),
                                columns=["modelo", "veces_que_rescata"])
    rescue_table["share_pct"] = (100 * rescue_table["veces_que_rescata"] / max(len(misses), 1)).round(2)
    rescue_table.to_csv(cfg.TABLES / "phase1_modelos_que_rescatan.csv", index=False)

    # Perfil de los errores que el ensamble pierde
    miss_profile = misses.groupby("true_class").size().rename("n").to_frame()
    miss_profile["share_pct"] = (100 * miss_profile["n"] / miss_profile["n"].sum()).round(2)
    miss_pairs = (misses.groupby(["true_class", "ensemble_prediction"]).size()
                  .sort_values(ascending=False).head(8).rename("n").to_frame())

    # --- L5: error por tipo de tweet y por bins finos de longitud ---
    by_type = (frame.groupby("tweet_type")
               .agg(n=("ensemble_right", "size"),
                    accuracy=("ensemble_right", "mean"),
                    error_rate=("ensemble_right", lambda s: 1 - s.mean()),
                    n_chars=("n_chars", "mean"))
               .reset_index().sort_values("n", ascending=False))
    by_type["accuracy"] = by_type["accuracy"].round(4)
    by_type["error_rate"] = by_type["error_rate"].round(4)
    by_type.to_csv(cfg.TABLES / "phase1_error_por_tipo.csv", index=False)

    frame["length_bin"] = pd.cut(frame["n_chars"], bins=LENGTH_BINS)
    by_length = (frame.groupby("length_bin", observed=True)
                 .agg(n=("ensemble_right", "size"), accuracy=("ensemble_right", "mean"),
                      error_rate=("ensemble_right", lambda s: 1 - s.mean()),
                      share_ensemble_miss=("ensemble_miss", "mean"))
                 .reset_index())
    by_length["accuracy"] = by_length["accuracy"].round(4)
    by_length["share_ensemble_miss"] = by_length["share_ensemble_miss"].round(4)
    by_length.to_csv(cfg.TABLES / "phase1_error_por_longitud.csv", index=False)

    payload = {
        "macro_f1_ensamble": round(ensemble_macro, 4),
        "interseccion": summary_rows,
        "ensemble_miss_pct": round(100 * miss_rate, 2),
        "modelos_que_rescatan": rescue_table.to_dict(orient="records"),
        "perfil_miss_por_clase": miss_profile.to_dict(),
        "pares_miss": {f"{a}->{b}": int(v) for (a, b), v in miss_pairs["n"].items()},
        "error_por_tipo": by_type.to_dict(orient="records"),
        "error_por_longitud": by_length.to_dict(orient="records"),
        "segundos": round(time.time() - started, 1),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")

    print("\n=== L1 — INTERSECCIÓN DE ERRORES ===")
    print(summary.to_string(index=False))
    print(f"\nModelos que rescatan al ensamble:")
    print(rescue_table.head(8).to_string(index=False))
    print("\n=== L5 — ERROR POR TIPO DE TWEET ===")
    print(by_type.to_string(index=False))
    print("\n=== L5 — ERROR POR LONGITUD ===")
    print(by_length.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
