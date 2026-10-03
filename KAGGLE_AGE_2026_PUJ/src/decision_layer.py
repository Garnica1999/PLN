"""Infraestructura de la capa de decisión ordinal (fases ORD-1 … ORD-8).

Sustrato (todo ya existente y permitido; **ninguna representación nueva**):
    - probabilidades OOF de V12 (``features/oof_ensemble_v12.npy``);
    - bloques OOF de los miembros ordinales y de pares adyacentes ya presentes en el ensamble;
    - vista estructural ya evaluada (``features/vista_estructura_margenes.npy``);
    - features densas/estructuradas de contexto (longitud, menciones, URL, retuit, tipo de tweet).

Protocolo (corrected CV, sin holdout):
    Para cada fold se **ajusta la capa de decisión con las probabilidades OOF de los otros cuatro
    folds** y se aplica al fold retenido. Nunca se ajusta ni evalúa sobre predicciones in-sample.

Cada experimento aporta un ``construir(train_idx, sustrato) -> callable`` que devuelve la función que
transforma las probabilidades de un conjunto de filas. El evaluador se encarga de las métricas, del
análisis de complementariedad, de las fronteras y del error ordinal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from . import config as cfg
from . import cv, data

OOF_V12 = cfg.FEATURES / "oof_ensemble_v12.npy"
ESTRUCTURA = cfg.FEATURES / "vista_estructura_margenes.npy"
MEMBRES_ORDINALES = ["oof_E015_softmax", "oof_E021a_ordinal"]
PARES_ADYACENTES = "oof_E028a_pares"
GRUPOS_SIN_MENCIONES = "es_grupo_sin_menciones"
GRUPOS_SOLO_URL = "es_grupo_solo_url"
GRUPOS_MUY_CORTO = "es_grupo_muy_corto"


def sigmoid(matriz: np.ndarray) -> np.ndarray:
    """Sigmoide numéricamente estable por filas."""
    desplazada = matriz - matriz.max(axis=1, keepdims=True)
    exponencial = np.exp(desplazada)
    return exponencial / exponencial.sum(axis=1, keepdims=True)


def cargas_sustrato() -> dict:
    """Carga probabilidades OOF de V12, contexto y bloques auxiliares."""
    train = data.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    indices = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cv.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")

    oof = np.load(OOF_V12).astype(float)
    denso = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")
    contexto = pd.DataFrame({
        "n_chars": denso["n_chars"].to_numpy(float),
        "n_mention": denso["n_mention"].to_numpy(float),
        "n_hashtag": denso["n_hashtag"].to_numpy(float),
        "n_url": denso["n_url"].to_numpy(float),
        "is_rt": denso["is_rt"].to_numpy(float),
    })
    contexto["es_grupo_solo_url"] = ((denso["n_url"] >= 1) & (denso["n_words"] <= 3)).astype(float)
    contexto["es_grupo_muy_corto"] = (denso["n_chars"] <= 15).astype(float)
    contexto["es_grupo_sin_menciones"] = (denso["n_mention"] == 0).astype(float)

    auxiliares = {}
    for nombre in MEMBRES_ORDINALES + [PARES_ADYACENTES]:
        matriz = np.load(cfg.FEATURES / f"{nombre}.npy")
        auxiliares[nombre] = np.asarray(matriz, dtype=float)
    estructura = np.load(ESTRUCTURA).astype(float) if ESTRUCTURA.exists() else None

    return {"y": y, "indices": indices, "folds": folds, "oof": oof, "contexto": contexto,
            "auxiliares": auxiliares, "estructura": estructura, "clusters": clusters}


def cdf(probabilidades: np.ndarray) -> np.ndarray:
    """Función de distribución acumulada por clase (monótona por construcción)."""
    return np.cumsum(probabilidades, axis=1)


def clases_de(probabilidades: np.ndarray) -> np.ndarray:
    """Índices de clase del argmax."""
    return np.argmax(probabilidades, axis=1)


def desde_cdf(acumulada: np.ndarray) -> np.ndarray:
    """Reconstruye la distribución de clases diferenciando una CDF (recorte y renormaliza)."""
    anterior = np.concatenate([np.zeros((len(acumulada), 1)), acumulada[:, :-1]], axis=1)
    masas = np.clip(acumulada - anterior, 1e-9, None)
    return masas / masas.sum(axis=1, keepdims=True)


def desde_supervivencia(supervivencia: np.ndarray) -> np.ndarray:
    """Masa de clase a partir de la función de supervivencia S(k) = P(Y > k).

    Exige S no creciente en k (se impone con ``minimum.accumulate``): p_0 = 1 - S(0) y
    p_k = S(k-1) - S(k) para k >= 1.
    """
    s = np.minimum.accumulate(np.clip(supervivencia, 0.0, 1.0), axis=1)
    n_clases = s.shape[1] + 1
    masas = np.zeros((len(s), n_clases))
    masas[:, 0] = 1.0 - s[:, 0]
    for k in range(1, n_clases):
        masas[:, k] = (np.clip(s[:, k - 1] - s[:, k], 1e-9, None) if k < s.shape[1]
                       else np.clip(s[:, k - 1], 1e-9, None))
    total = masas.sum(axis=1, keepdims=True)
    return masas / np.where(total < 1e-12, 1.0, total)


def metricas(probabilidades: np.ndarray, reales: np.ndarray) -> dict:
    """Métricas de una capa de decisión: macro-F1, accuracy, P/R/F1, error ordinal y confusión."""
    predichos = clases_de(probabilidades)
    etiqueta = np.arange(len(cfg.CLASSES))
    precision, recall, f1, soporte = precision_recall_fscore_support(
        reales, predichos, labels=etiqueta, zero_division=0)
    distancia = np.abs(predichos - reales)
    confusion = confusion_matrix(reales, predichos, labels=etiqueta)
    return {
        "macro_f1": round(float(f1_score(reales, predichos, average="macro", labels=etiqueta)), 4),
        "accuracy": round(float((predichos == reales).mean()), 4),
        "precision": {clase: round(float(valor), 4) for clase, valor in zip(cfg.CLASSES, precision)},
        "recall": {clase: round(float(valor), 4) for clase, valor in zip(cfg.CLASSES, recall)},
        "f1_por_clase": {clase: round(float(valor), 4) for clase, valor in zip(cfg.CLASSES, f1)},
        "mae_ordinal": round(float(distancia.mean()), 4),
        "error_distancia_1_pct": round(100 * float((distancia == 1).mean()), 3),
        "error_distancia_2_pct": round(100 * float((distancia == 2).mean()), 3),
        "error_distancia_3_o_mas_pct": round(100 * float((distancia >= 3).mean()), 3),
        "confusion": confusion.tolist(),
        "n_por_clase_predicha": {clase: int(valor) for clase, valor in zip(cfg.CLASSES, confusion.sum(axis=0))},
    }


def complemento(base: np.ndarray, nueva: np.ndarray, reales: np.ndarray,
                 contexto: pd.DataFrame) -> dict:
    """Rescate, daño, fronteras y segmentos clave frente a la capa base."""
    correcto_base = clases_de(base) == reales
    correcto_nueva = clases_de(nueva) == reales
    frontera_base = (clases_de(base) == 1) & (clases_de(base) == 2)
    del frontera_base
    segmentos = {}
    for clave in (GRUPOS_SOLO_URL, GRUPOS_MUY_CORTO, GRUPOS_SIN_MENCIONES):
        mascara = contexto[clave].to_numpy() > 0
        segmentos[clave] = {
            "filas": int(mascara.sum()),
            "error_base_pct": round(100 * float((~correcto_base[mascara]).mean()), 2),
            "error_candidato_pct": round(100 * float((~correcto_nueva[mascara]).mean()), 2),
        }
    # Errores por frontera adyacente (parejas contiguas).
    fronteras = {}
    for borde in range(len(cfg.CLASSES) - 1):
        izquierda, derecha = borde, borde + 1
        cruza_base = int((((clases_de(base) == izquierda) & (reales == derecha))
                          | ((clases_de(base) == derecha) & (reales == izquierda))).sum())
        cruza_nueva = int((((clases_de(nueva) == izquierda) & (reales == derecha))
                           | ((clases_de(nueva) == derecha) & (reales == izquierda))).sum())
        fronteras[f"{cfg.CLASSES[izquierda]}|{cfg.CLASSES[derecha]}"] = {
            "errores_base": cruza_base, "errores_candidato": cruza_nueva,
            "delta": cruza_nueva - cruza_base,
        }
    return {
        "ambos_correctos": int((correcto_base & correcto_nueva).sum()),
        "solo_base": int((correcto_base & ~correcto_nueva).sum()),
        "solo_candidato": int((~correcto_base & correcto_nueva).sum()),
        "ambos_mal": int((~correcto_base & ~correcto_nueva).sum()),
        "neto": int((~correcto_base & correcto_nueva).sum() - (correcto_base & ~correcto_nueva).sum()),
        "fronteras": fronteras,
        "segmentos": segmentos,
        "filas_modificadas": int((clases_de(base) != clases_de(nueva)).sum()),
        "filas_modificadas_pct": round(100 * float((clases_de(base) != clases_de(nueva)).mean()), 3),
    }


def evaluar_capa(nombre: str, construir, sustrato: dict) -> dict:
    """Evalúa una capa de decisión con corrected CV (LOFO sobre probabilidades OOF)."""
    y = sustrato["y"]
    reales = sustrato["indices"]
    oof = sustrato["oof"]
    predicciones = np.zeros_like(oof)
    por_fold = []
    for fold in sustrato["folds"]:
        train_idx = np.concatenate([f["val_idx"] for f in sustrato["folds"]
                                    if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        aplicar = construir(train_idx, sustrato)
        predicciones[valid_idx] = aplicar(valid_idx)
        por_fold.append(float(metricas(predicciones[valid_idx], reales[valid_idx])["macro_f1"]))
    base_por_fold = [float(metricas(oof[f["val_idx"]], reales[f["val_idx"]])["macro_f1"])
                     for f in sustrato["folds"]]
    referencia = float(np.mean(base_por_fold))
    candidatos = float(np.mean(por_fold))
    positivos = int(sum(1 for nuevo, viejo in zip(por_fold, base_por_fold) if nuevo > viejo))
    resultado = metricas(predicciones, reales)
    base_global = metricas(oof, reales)
    resultado.update({
        "capa": nombre,
        "macro_f1": round(candidatos, 4),
        "macro_f1_base_por_fold": round(referencia, 4),
        "delta": round(candidatos - referencia, 4),
        "std": round(float(np.std(por_fold)), 4),
        "per_fold": [round(valor, 4) for valor in por_fold],
        "per_fold_base": [round(valor, 4) for valor in base_por_fold],
        "folds_positivos": positivos,
        "macro_f1_oof_completo": resultado["macro_f1"],
        "macro_f1_oof_completo_base": base_global["macro_f1"],
        "adoptado": bool(candidatos - referencia >= 0.003 and positivos >= 4),
    })
    resultado["complementariedad"] = complemento(oof, predicciones, reales, sustrato["contexto"])
    resultado["detalle_base"] = {"macro_f1": base_global["macro_f1"],
                                 "mae_ordinal": base_global["mae_ordinal"],
                                 "f1_por_clase": base_global["f1_por_clase"]}
    return resultado


def guardar(resultados: list[dict], ruta_json: Path, ruta_oof: Path | None = None,
            predicciones: dict | None = None) -> None:
    """Persiste los resultados de una tanda de experimentos."""
    ruta_json.write_text(json.dumps({"resultados": resultados}, indent=2, ensure_ascii=False,
                                    default=str), encoding="utf-8")
    if ruta_oof is not None and predicciones is not None:
        for nombre, matriz in predicciones.items():
            np.save(ruta_oof.parent / f"oof_capa_{nombre}.npy", matriz.astype(np.float32))
