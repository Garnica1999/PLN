"""FASE 5 / L2 — Residualización por longitud (brief §12 y §42).

Observación
    La longitud domina el error (54,5% en ≤10 caracteres → 35,1% en 130-141) y es la feature más
    informativa del EDA (ε² = 0,126). Muchas features derivadas (riqueza léxica, menciones, hashtags,
    puntuación, tamaños de vocabulario) crecen mecánicamente con la longitud, de modo que su señal
    puede ser en parte **un eco de la longitud** y no información de edad independiente.

Hipótesis
    H30  El **residuo** de cada feature tras descontar su valor esperado dada la longitud contiene
         señal de edad ortogonal a la longitud, y añadirlo al meta-modelo mejora macro-F1 sobre todo
         en los tramos donde la longitud no discrimina (tweets medios).

Método (fold-safe)
    Para cada fold: se estima ``E[feature | n_chars]`` de forma NO paramétrica con 30 bins de
    cuantiles **calculados solo con el train del fold**; el residuo es
    ``feature − E[feature | n_chars]``. Se aplican los mismos cortes a validación (con recorte a los
    extremos). El bloque de residuos se añade al meta-modelo y se evalúa con LOFO.

Salida: ``experiments/phase5_residuales.json`` y ``features/length_residuals.npy``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, textometry  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from phase2_ponderacion import build_features, evaluate_global  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "phase5_residuales.json"
N_BINS = 30
MIN_ABS_CORRELATION = 0.25


def select_features(dense: pd.DataFrame, structured: pd.DataFrame,
                    min_correlation: float = MIN_ABS_CORRELATION) -> list[str]:
    """Selecciona las features densas/estructuradas más confundidas con la longitud."""
    length = dense["n_chars"].to_numpy(float)
    candidates: dict[str, np.ndarray] = {}
    for column in dense.columns:
        if column in {"n_chars", "n_words"}:
            continue
        candidates[f"dense::{column}"] = dense[column].to_numpy(float)
    for column in structured.columns:
        candidates[f"struct::{column}"] = structured[column].to_numpy(float)
    selected = []
    for name, values in candidates.items():
        if np.std(values) == 0:
            continue
        correlation = np.corrcoef(length, values)[0, 1]
        if np.isfinite(correlation) and abs(correlation) >= min_correlation:
            selected.append((name, round(float(correlation), 3)))
    selected.sort(key=lambda item: -abs(item[1]))
    return [name for name, _ in selected], selected


def conditional_mean(train_length: np.ndarray, train_values: np.ndarray,
                     target_length: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """``E[feature | n_chars]`` no paramétrico por bins de cuantiles del train.

    Los cortes se calculan **solo con train**; a validación se le aplican los mismos cortes con
    recorte a los extremos.
    """
    edges = np.unique(np.quantile(train_length, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:
        return np.full(target_length.shape, float(np.mean(train_values)))
    edges[0], edges[-1] = -np.inf, np.inf
    train_bin = np.digitize(train_length, edges[1:-1])
    target_bin = np.digitize(target_length, edges[1:-1])
    means = pd.Series(train_values).groupby(train_bin).mean()
    global_mean = float(np.mean(train_values))
    return np.array([means.get(bin_index, global_mean) for bin_index in target_bin], dtype=float)


def main() -> int:
    """Construye los residuos de longitud y evalúa su aporte al meta-modelo."""
    started = time.time()
    from src import data
    train = data.load_train()
    dense = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")
    structured = pd.read_parquet(cfg.FEATURES / "age_structured_features.parquet")

    names, correlations = select_features(dense, structured)
    print(f"features con |corr| ≥ {MIN_ABS_CORRELATION} con n_chars: {len(names)}", flush=True)
    print(f"  top: {correlations[:8]}", flush=True)

    length = dense["n_chars"].to_numpy(float)
    columns = {name: (dense[name.split("::")[1]].to_numpy(float) if name.startswith("dense::")
                      else structured[name.split("::")[1]].to_numpy(float))
               for name in names}
    matrix = np.column_stack([columns[name] for name in names])

    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    y = train[cfg.TARGET].astype(str).to_numpy()
    folds = cv.make_folds(y, groups=clusters["cluster"].to_numpy()[: len(y)],
                          n_splits=5, n_repeats=1, scheme="group")

    print("calculando residuos in-fold...", flush=True)
    residuals = np.zeros_like(matrix)
    for fold in folds:
        train_idx, valid_idx = fold["train_idx"], fold["val_idx"]
        for position in range(matrix.shape[1]):
            expected = conditional_mean(length[train_idx], matrix[train_idx, position],
                                        length[valid_idx])
            residuals[valid_idx, position] = matrix[valid_idx, position] - expected
        print(f"  fold {fold['fold']} listo ({round(time.time() - started, 1)}s)", flush=True)
    np.save(cfg.FEATURES / "length_residuals.npy", residuals)
    (cfg.FEATURES / "length_residuals_columns.txt").write_text("\n".join(names), encoding="utf-8")

    print("evaluando el aporte con el meta-modelo MLP (LOFO)...", flush=True)
    features, y2, _, _, _ = build_features()
    assert np.array_equal(y, y2)
    with_residuals = np.concatenate([features, residuals], axis=1)

    mean_base, std_base, folds_base = evaluate_global(features, y, folds, "mlp")
    print(f"  MLP sin residuos ({features.shape[1]}): {mean_base:.4f} ± {std_base:.4f}", flush=True)
    mean_res, std_res, folds_res = evaluate_global(with_residuals, y, folds, "mlp")
    print(f"  MLP con residuos ({with_residuals.shape[1]}): {mean_res:.4f} ± {std_res:.4f} "
          f"({mean_res - mean_base:+.4f})", flush=True)

    difference = np.array(folds_res) - np.array(folds_base)
    payload = {
        "features_residualizadas": names,
        "correlaciones_con_longitud": correlations,
        "n_bins": N_BINS,
        "mlp_sin_residuos": {"macro_f1": mean_base, "std": std_base, "per_fold": folds_base},
        "mlp_con_residuos": {"macro_f1": mean_res, "std": std_res, "per_fold": folds_res},
        "delta": round(mean_res - mean_base, 4),
        "folds_con_mejora": int((difference > 0).sum()),
        "mejora_por_fold": difference.round(4).tolist(),
        "segundos": round(time.time() - started, 1),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nΔ = {payload['delta']:+.4f} | mejora en {payload['folds_con_mejora']}/5 folds "
          f"| por fold: {payload['mejora_por_fold']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
