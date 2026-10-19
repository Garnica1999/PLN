"""Analisis matematico y estadistico de V57 (porte fiel de scripts/v57_analisis_matematico_estadistico.py).

Cuatro bloques sobre los OOF de 169.336 filas (miembros de V56 en orden y regla plug-in):
1. sensibilidad_simplex: perturbacion +-20% de cada coordenada de pesos (v57:66-94).
2. curva_power_mean: familia P_alpha con alpha en [-0.15, +0.15]; alpha=0 es el PoE (v57:101-119).
3. deciles_confiabilidad: deciles de confianza maxima vs exactitud (v57:125-149).
4. brecha_menciones: autocalibracion con/sin mencion (n_mention de dense_train.parquet; v57:151-152,165-178).

Los estados se escriben en ASCII (el original usa tildes: OPTIMO/SUB-OPTIMO). Sin efectos en import.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from cadena.consenso_v2 import poe_ponderado, regla_plug_in

#: Grilla de alpha del bloque Power-Mean (v57_analisis_matematico_estadistico.py:104).
ALPHAS = [-0.15, -0.10, -0.05, -0.01, 0.0, 0.01, 0.05, 0.10, 0.15]

#: Epsilon del clip de probabilidades (v57:46).
EPS = 1e-12

#: Nombres de los 9 miembros de V56 en orden de apilado (v57:48-52).
NOMBRES_V56 = ["V30 LightGBM Optuna", "V43 Cramer-SWA", "V47 Asym-Cramer", "V42 Smooth-SWA",
               "V40 SE-TabOrd", "V39 TabOrd", "V39 HistGB", "V51 Consistent-Cramer",
               "V53 Dense-SE"]


def macro_f1(y_true: np.ndarray, pred: np.ndarray) -> float:
    """Macro-F1 canonico sobre indices 0-5 (v57:62)."""
    return float(f1_score(y_true, pred, average="macro"))


def f1_por_folds(y_true: np.ndarray, pred: np.ndarray, fold_idx: np.ndarray) -> list[float]:
    """Macro-F1 por fold 0-4 (v57:63)."""
    return [float(f1_score(y_true[fold_idx == f], pred[fold_idx == f], average="macro"))
            for f in range(5)]


def sensibilidad_simplex(modelos: list[np.ndarray], y_true: np.ndarray, pesos: list[float],
                         nombres: list[str] | None = None) -> list[dict]:
    """Bloque 1: sensibilidad coordenual +-20% en el simplex (v57:66-94)."""
    w_base = np.asarray(pesos, dtype=np.float64)
    w_base = w_base / w_base.sum()
    p_base = poe_ponderado(modelos, w_base)
    f1_base = macro_f1(y_true, regla_plug_in(p_base))
    if nombres is None:
        nombres = [f"miembro_{i}" for i in range(len(modelos))]
    registros = []
    for i, nombre in enumerate(nombres):
        w_low = w_base.copy()
        w_low[i] *= 0.80
        w_low /= w_low.sum()
        f1_low = macro_f1(y_true, regla_plug_in(poe_ponderado(modelos, w_low)))

        w_high = w_base.copy()
        w_high[i] *= 1.20
        w_high /= w_high.sum()
        f1_high = macro_f1(y_true, regla_plug_in(poe_ponderado(modelos, w_high)))

        registros.append({
            "modelo": nombre,
            "peso_base": float(w_base[i]),
            "f1_minus_20pct": f1_low,
            "delta_minus_20pct": f1_low - f1_base,
            "f1_plus_20pct": f1_high,
            "delta_plus_20pct": f1_high - f1_base,
            "estado": "ESTACIONARIO" if (f1_low < f1_base and f1_high < f1_base)
                      else "NO_ESTACIONARIO",
        })
    return registros


def curva_power_mean(modelos: list[np.ndarray], y_true: np.ndarray,
                     pesos: list[float]) -> list[dict]:
    """Bloque 2: Macro-F1(alpha) de la familia Power-Mean; alpha=0 es el PoE (v57:101-119)."""
    w_base = np.asarray(pesos, dtype=np.float64)
    w_base = w_base / w_base.sum()
    p_poe = poe_ponderado(modelos, w_base)
    f1_base = macro_f1(y_true, regla_plug_in(p_poe))
    registros = []
    for alpha in ALPHAS:
        if alpha == 0.0:
            p = p_poe
        else:
            p_alpha = sum(wi * (np.clip(m, EPS, 1.0) ** alpha)
                          for wi, m in zip(w_base, modelos))
            p = p_alpha ** (1.0 / alpha)
            p = p / p.sum(axis=1, keepdims=True)
        f1 = macro_f1(y_true, regla_plug_in(p))
        registros.append({
            "alpha": alpha,
            "macro_f1": f1,
            "delta_vs_poe": f1 - f1_base,
            "estado": "OPTIMO GLOBAL ESTACIONARIO (PoE)" if alpha == 0.0 else "SUB-OPTIMO",
        })
    return registros


def deciles_confiabilidad(prob: np.ndarray, y_true: np.ndarray) -> list[dict]:
    """Bloque 3: deciles de confianza maxima vs exactitud (v57:125-149)."""
    pred = regla_plug_in(prob)
    conf = np.max(prob, axis=1)
    correct = (pred == y_true)
    deciles = pd.qcut(conf, 10, labels=False)
    registros = []
    for d in range(10):
        mask = (deciles == d)
        registros.append({
            "decil": int(d),
            "confianza_min": float(conf[mask].min()),
            "confianza_max": float(conf[mask].max()),
            "confianza_media": float(conf[mask].mean()),
            "exactitud": float(correct[mask].mean()),
            "conteo": int(mask.sum()),
        })
    return registros


def brecha_menciones(prob: np.ndarray, y_true: np.ndarray, n_mention: np.ndarray) -> dict:
    """Bloque 4: brecha confianza-exactitud con/sin mencion (v57:151-152,165-178)."""
    pred = regla_plug_in(prob)
    conf = np.max(prob, axis=1)
    correct = (pred == y_true)
    has_men = (n_mention > 0)
    return {
        "tweets_con_mencion": {
            "proporcion": float(has_men.mean()),
            "exactitud": float(correct[has_men].mean()),
            "confianza_media": float(conf[has_men].mean()),
            "gap": float(conf[has_men].mean() - correct[has_men].mean()),
        },
        "tweets_sin_mencion": {
            "proporcion": float((~has_men).mean()),
            "exactitud": float(correct[~has_men].mean()),
            "confianza_media": float(conf[~has_men].mean()),
            "gap": float(conf[~has_men].mean() - correct[~has_men].mean()),
        },
    }


def analisis_completo(modelos: list[np.ndarray], y_true: np.ndarray, pesos: list[float],
                      n_mention: np.ndarray, nombres: list[str] | None = None,
                      fold_idx: np.ndarray | None = None) -> dict:
    """Los cuatro bloques de V57 en un resumen con el esquema del JSON original."""
    prob = poe_ponderado(modelos, pesos)
    pred = regla_plug_in(prob)
    return {
        "f1_base": macro_f1(y_true, pred),
        "folds_base": None if fold_idx is None else f1_por_folds(y_true, pred, fold_idx),
        "sensibilidad_simplex": sensibilidad_simplex(modelos, y_true, pesos, nombres),
        "curva_power_mean": curva_power_mean(modelos, y_true, pesos),
        "deciles_confiabilidad": deciles_confiabilidad(prob, y_true),
        "brecha_menciones": brecha_menciones(prob, y_true, n_mention),
    }
