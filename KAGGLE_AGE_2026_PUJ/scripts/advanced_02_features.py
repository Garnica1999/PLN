"""Fase avanzada — Paso 2: features estructuradas de segunda generación.

Ejecuta en orden:
    1. **Tests unitarios** de robustez (brief §64): texto vacío, un token, solo URL, solo mención,
       RT, múltiples hashtags, solo puntuación, solo números, repetición.
    2. **Cálculo de features locales** (familias A, B, C, D, H, I, M, N, R, S, T, U) para train y
       test -> ``features/age_structured_features.parquet``.
    3. **Catálogo** de las nuevas features -> ``reports/advanced_feature_catalog.csv``.
    4. **Evaluación del aporte** al meta-modelo del baseline congelado (LOFO), comparando:
       baseline -> +estructuradas -> +supervisadas (menciones, hashtags, fuente RT, rareza, léxico).

Salida: ``experiments/advanced_02_resultados.json``.
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
from src import cv, data, textometry  # noqa: E402
from src.config import set_seed  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "advanced_02_resultados.json"
LOCAL_BUILDERS = {
    "A_distribucion": af.distribution_features,
    "B_char": af.char_distribution_features,
    "D_riqueza": af.lexical_richness_features,
    "HT_forma": af.word_shape_features,
    "IMU_posicion": af.position_features,
    "NRS_estructura": af.structure_entropy_features,
}
BASE_SOURCES = [("oof_E013_margenes.npy", False), ("oof_E016_margenes.npy", False),
                ("oof_E-009.npy", True), ("oof_E-011.npy", True), ("oof_E-010.npy", True),
                ("oof_E014_margenes.npy", False), ("oof_E015_ordinal.npy", False),
                ("oof_E015_softmax.npy", True), ("oof_E017_probabilidades.npy", True),
                ("oof_E019b_densas_svd.npy", True), ("oof_E021a_ordinal.npy", True),
                ("oof_E021b_rocchio.npy", True), ("oof_E028a_pares.npy", False)]
DENSE_FEATURES = [name for name in textometry.TEXTOMETRY_FEATURES + textometry.STYLE_FEATURES
                  if name not in {"n_emoji", "emoji_diversity", "has_emoji"}]

EDGE_CASES = {
    "vacio": "",
    "un_token": "hola",
    "solo_url": "http://t.co/abc123",
    "solo_mencion": "@usuario",
    "rt": "RT @fuente: texto del retuit original con mas palabras",
    "multi_hashtag": "#uno #dos #tres texto",
    "repeticion": "jajajajajaja sooooo foo foo foo",
    "solo_puntuacion": "!!! ??? ... ,,,",
    "solo_numeros": "123 456 789",
    "mixto": "Hola @amigo mira http://t.co/x #TAG jajaja 123 ALLCAPS",
}


def run_unit_tests() -> dict:
    """Ejecuta la batería de casos límite y verifica que no haya infinitos ni NaN."""
    report: dict[str, dict] = {}
    for name, text in EDGE_CASES.items():
        series = pd.Series([text])
        builder_report: dict[str, str] = {}
        for builder_name, builder in LOCAL_BUILDERS.items():
            frame = builder(series)
            problems = []
            if frame.shape[1] == 0:
                problems.append("sin columnas")
            if np.isinf(frame.to_numpy(dtype=float)).any():
                problems.append("infinito")
            if frame.isna().any().any():
                problems.append("NaN")
            builder_report[builder_name] = "OK" if not problems else ",".join(problems)
        report[name] = builder_report

    # Dimensiones estables entre casos límite (misma cantidad de columnas por familia).
    for builder_name, builder in LOCAL_BUILDERS.items():
        shapes = {builder(pd.Series([text])).shape[1] for text in EDGE_CASES.values()}
        report.setdefault("dimensiones", {})[builder_name] = "OK" if len(shapes) == 1 else f"INESTABLE {shapes}"
    return report


def build_local_features(texts: pd.Series, verbose: bool = True) -> pd.DataFrame:
    """Calcula todas las familias locales para una serie de textos."""
    frames = []
    for name, builder in LOCAL_BUILDERS.items():
        started = time.time()
        frame = builder(texts)
        frames.append(frame)
        if verbose:
            print(f"    [{name}] {frame.shape[1]} features ({round(time.time() - started, 1)}s)",
                  flush=True)
    return pd.concat(frames, axis=1)


def load_base_sources() -> np.ndarray:
    """Carga las 13 matrices OOF del baseline congelado."""
    matrices = []
    for filename, probabilistic in BASE_SOURCES:
        matrix = np.load(cfg.FEATURES / filename)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        matrices.append(np.log(np.clip(matrix, 1e-12, None)) if probabilistic else matrix)
    return np.concatenate(matrices, axis=1)


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
    """Ejecuta tests, cálculo de features y evaluación del aporte."""
    set_seed()
    started = time.time()
    results: dict = {}

    # --- 1. Tests unitarios ---
    print("[1/4] tests unitarios de robustez...", flush=True)
    tests = run_unit_tests()
    failures = {case: detail for case, detail in tests.items()
                if isinstance(detail, dict) and any(value != "OK" for value in detail.values())}
    if isinstance(tests.get("dimensiones"), dict):
        failures.update({f"dimensiones_{key}": value for key, value in tests["dimensiones"].items()
                         if value != "OK"})
    results["tests_unitarios"] = {"casos": len(EDGE_CASES), "fallos": failures}
    if failures:
        print(f"    FALLOS: {failures}", flush=True)
    else:
        print(f"    {len(EDGE_CASES)} casos límite × {len(LOCAL_BUILDERS)} familias: OK, sin "
              f"infinitos, NaN ni dimensiones inestables", flush=True)

    # --- 2. Cálculo de features locales ---
    print("[2/4] calculando features locales sobre train y test...", flush=True)
    train = data.load_train()
    test = data.load_test()
    y = train[cfg.TARGET].astype(str).to_numpy()
    structured_train = build_local_features(train[cfg.TEXT_COL])
    structured_test = build_local_features(test[cfg.TEXT_COL])
    structured_train.to_parquet(cfg.FEATURES / "age_structured_features.parquet", index=False)
    structured_test.to_parquet(cfg.FEATURES / "age_structured_features_test.parquet", index=False)
    print(f"    train {structured_train.shape} | test {structured_test.shape} -> "
          f"features/age_structured_features.parquet", flush=True)
    results["features_locales"] = {"n_features": int(structured_train.shape[1]),
                                   "segundos": round(time.time() - started, 1)}

    # --- 3. Catálogo ---
    print("[3/4] generando catálogo de features...", flush=True)
    catalog_rows = []
    for family, description in af.FAMILIES.items():
        for column in structured_train.columns:
            prefix = family.split("_")[1] if "_" in family else family
            if column.startswith(prefix) or prefix in column:
                catalog_rows.append({
                    "feature_name": column, "family": family, "description": description,
                    "formula": "ver src/advanced_features.py", "dtype": "float64",
                    "train_only_dependency": "no", "leakage_risk": "nulo (local al documento)",
                    "hypothesis": "distribución/forma/posición aportan sobre las medias",
                    "source": "src/advanced_features.py", "experiment_id": "ADV-02",
                })
    catalog = pd.DataFrame(catalog_rows).drop_duplicates("feature_name")
    catalog.to_csv(cfg.REPORTS / "advanced_feature_catalog.csv", index=False)
    print(f"    reports/advanced_feature_catalog.csv ({len(catalog)} features catalogadas)",
          flush=True)

    # --- 4. Evaluación del aporte ---
    print("[4/4] evaluando el aporte al meta-modelo (LOFO)...", flush=True)
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]
    folds = cv.make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")
    dense = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")[DENSE_FEATURES].to_numpy(float)
    base = load_base_sources()

    structured = structured_train.to_numpy(float)
    baseline = np.concatenate([base, dense], axis=1)
    with_structured = np.concatenate([base, dense, structured], axis=1)

    mean_base, std_base, folds_base = evaluate(baseline, y, folds)
    print(f"    baseline (107 features): {mean_base:.4f} ± {std_base:.4f}", flush=True)
    mean_struct, std_struct, folds_struct = evaluate(with_structured, y, folds)
    print(f"    + estructuradas ({with_structured.shape[1]}): {mean_struct:.4f} ± {std_struct:.4f} "
          f"(delta {mean_struct - mean_base:+.4f})", flush=True)

    # --- Features supervisadas (in-fold) ---
    print("    calculando features supervisadas in-fold...", flush=True)
    supervised_frames = {name: np.zeros((len(train), 0)) for name in
                         ("mencion", "hashtag", "fuente_rt", "rareza", "lexico")}
    supervised_parts: list[pd.DataFrame] = []
    for fold in folds:
        train_idx, valid_idx = fold["train_idx"], fold["val_idx"]
        train_texts = train[cfg.TEXT_COL].iloc[train_idx]
        valid_texts = train[cfg.TEXT_COL].iloc[valid_idx]
        y_train = y[train_idx]
        blocks = {
            "mencion": af.supervised_entity_features(train_texts, y_train, valid_texts, "mention"),
            "hashtag": af.supervised_entity_features(train_texts, y_train, valid_texts, "hashtag"),
            "fuente_rt": af.supervised_entity_features(train_texts, y_train, valid_texts, "rt_source"),
            "rareza": af.lexical_rarity_features(train_texts, valid_texts),
            "lexico": af.class_lexicon_scores(train_texts, y_train, valid_texts),
        }
        for name, frame in blocks.items():
            if supervised_frames[name].shape[1] == 0:
                supervised_frames[name] = np.zeros((len(train), frame.shape[1]))
            supervised_frames[name][valid_idx] = frame.to_numpy(float)
        block = pd.concat(blocks.values(), axis=1)
        block.index = valid_idx
        supervised_parts.append(block)
        print(f"      fold {fold['fold']}: {block.shape[1]} features supervisadas", flush=True)

    supervised = pd.concat(supervised_parts).sort_index().to_numpy(float)
    np.save(cfg.FEATURES / "age_supervised_features.npy", supervised)
    (cfg.FEATURES / "age_supervised_columns.txt").write_text(
        "\n".join(pd.concat(supervised_parts).sort_index().columns), encoding="utf-8")

    for name, matrix in supervised_frames.items():
        with_block = np.concatenate([base, dense, structured, matrix], axis=1)
        mean_block, std_block, folds_block = evaluate(with_block, y, folds)
        delta = mean_block - mean_struct
        print(f"    + {name} ({matrix.shape[1]}): {mean_block:.4f} ± {std_block:.4f} "
              f"(delta vs estructuradas {delta:+.4f})", flush=True)
        results.setdefault("aporte_supervisado", {})[name] = {
            "macro_f1": mean_block, "std": std_block, "delta": delta, "per_fold": folds_block}

    with_all = np.concatenate([base, dense, structured, supervised], axis=1)
    mean_all, std_all, folds_all = evaluate(with_all, y, folds)
    print(f"    + TODAS las supervisadas ({with_all.shape[1]}): {mean_all:.4f} ± {std_all:.4f} "
          f"(delta vs baseline {mean_all - mean_base:+.4f})", flush=True)

    results.update({
        "baseline": {"macro_f1": mean_base, "std": std_base, "n_features": baseline.shape[1],
                     "per_fold": folds_base},
        "con_estructuradas": {"macro_f1": mean_struct, "std": std_struct,
                              "n_features": with_structured.shape[1], "per_fold": folds_struct},
        "con_todo": {"macro_f1": mean_all, "std": std_all, "n_features": with_all.shape[1],
                     "per_fold": folds_all},
        "segundos": round(time.time() - started, 1),
    })
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== ADV-02 — FEATURES DE SEGUNDA GENERACIÓN ===")
    print(f"baseline                    : {mean_base:.4f} ± {std_base:.4f}")
    print(f"+ estructuradas             : {mean_struct:.4f} ± {std_struct:.4f} "
          f"({mean_struct - mean_base:+.4f})")
    print(f"+ supervisadas + todo       : {mean_all:.4f} ± {std_all:.4f} "
          f"({mean_all - mean_base:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
