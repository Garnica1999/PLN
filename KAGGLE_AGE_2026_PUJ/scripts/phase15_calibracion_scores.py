"""FASE 15 — Calibración de miembros y transformaciones de score antes de combinar (hipótesis D).

Observación
    El combinador recibe los bloques de los 14 miembros en **representaciones heterogéneas**: márgenes
    crudos de SVM con escalas arbitrarias, log-probabilidades de modelos lineales y densos, y una
    fuente ordinal continua. La hipótesis D es que parte de la meseta proviene de esa heterogeneidad:
    la calibración de cada miembro y la forma de promediar los combinadores son decisiones que nunca
    se probaron.

Hipótesis
    H42 (D1a) Unificar todos los bloques a log-probabilidades (softmax, temperatura 1) mejora al
         combinador al eliminar la heterogeneidad de escalas.
    H43 (D1b) Ajustar una **temperatura por miembro in-fold** (solo con filas de los folds de
         entrenamiento) reduce la sobre/sub-confianza y mejora al combinador.
    H44 (D2) Promediar los dos combinadores en espacio **logit** es mejor que en probabilidades.
    H45 (D4) La ponderación 0,5/0,5 del promedio de combinadores es un óptimo local: una curva de
         pesos evaluada en las mismas particiones no aporta una mejora superior al umbral.

Protocolo
    Leave-one-fold-out con las mismas particiones agrupadas. Las temperaturas se ajustan
    exclusivamente con las filas de los folds de entrenamiento (rejilla de 0,25 a 4,0, NLL). D2 y D4
    se evalúan sobre las probabilidades de los combinadores del fold, sin reentrenar.
    Criterio: Δ ≥ +0,003 y ≥4/5 folds positivos.

Salida: ``experiments/phase15_calibracion_scores.json``.
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

from src import config as cfg
from src import cv
from src.config import set_seed

sys.path.insert(0, str(ROOT / "scripts"))
from phase2_ponderacion import MEMBER_FILES, EncodedMLP, build_features

RESULTS_PATH = cfg.EXPERIMENTS / "phase15_calibracion_scores.json"
REJILLA_TEMPERATURA = np.round(np.arange(0.25, 4.01, 0.25), 2)


def limites_de_bloques() -> list[tuple[str, int]]:
    """Ancho de cada bloque de miembro en la matriz del meta."""
    limites = []
    for nombre, _ in MEMBER_FILES:
        matriz = np.load(cfg.FEATURES / Path(nombre).name, mmap_mode="r")
        ancho = 1 if matriz.ndim == 1 else matriz.shape[1]
        limites.append((Path(nombre).stem, ancho))
    return limites


def log_softmax(matriz: np.ndarray, temperatura: float = 1.0) -> np.ndarray:
    """log-softmax por filas (con desplazamiento numérico estable)."""
    desplazado = (matriz - matriz.max(axis=1, keepdims=True)) / temperatura
    exponencial = np.exp(desplazado)
    return desplazado - np.log(exponencial.sum(axis=1, keepdims=True))


def ajustar_temperatura(logits: np.ndarray, etiquetas: np.ndarray) -> float:
    """Temperatura que minimiza la NLL sobre las filas indicadas (rejilla cerrada)."""
    indices = np.array([cfg.CLASSES.index(valor) for valor in etiquetas])
    mejor, mejor_nll = 1.0, np.inf
    for temperatura in REJILLA_TEMPERATURA:
        log_probabilidades = log_softmax(logits, temperatura)
        nll = float(-log_probabilidades[np.arange(len(indices)), indices].mean())
        if nll < mejor_nll:
            mejor, mejor_nll = float(temperatura), nll
    return mejor


def calibrado(features: np.ndarray, indices: list[tuple[int, int]], y: np.ndarray,
              train_idx: np.ndarray, temperaturas: bool) -> tuple[np.ndarray, list[float]]:
    """Unifica (y opcionalmente calibra) los bloques de 6 columnas a log-probabilidades."""
    resultado = features.copy()
    ajustadas = []
    for inicio, ancho in indices:
        if ancho != len(cfg.CLASSES):
            continue
        bloque = features[:, inicio:inicio + ancho]
        temperatura = (ajustar_temperatura(bloque[train_idx], y[train_idx]) if temperaturas else 1.0)
        resultado[:, inicio:inicio + ancho] = log_softmax(bloque, temperatura)
        ajustadas.append(temperatura)
    return resultado, ajustadas


def evaluar(features: np.ndarray, y: np.ndarray, folds: list[dict],
            temperaturas: bool) -> dict:
    """Evalúa los bloques calibrados y recoge las probabilidades de cada combinador."""
    puntajes = {"promedio": [], "logits": []}
    temperaturas_por_fold = []
    for fold in folds:
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        limites = limites_de_bloques()
        indices, inicio = [], 0
        for _, ancho in limites:
            indices.append((inicio, ancho))
            inicio += ancho
        matriz, ajustadas = calibrado(features, indices, y, train_idx, temperaturas)
        temperaturas_por_fold.append(ajustadas)
        logreg = Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                      random_state=cfg.SEED))])
        logreg.fit(matriz[train_idx], y[train_idx])
        mlp = EncodedMLP()
        mlp.fit(matriz[train_idx], y[train_idx])
        probabilidad_logreg = logreg.predict_proba(matriz[valid_idx])
        probabilidad_mlp = mlp.model.predict_proba(mlp.scaler.transform(matriz[valid_idx]))
        for etiqueta, matriz_probabilidad in (
                ("promedio", 0.5 * (probabilidad_logreg + probabilidad_mlp)),
                ("logits", np.exp(0.5 * (np.log(np.clip(probabilidad_logreg, 1e-12, None))
                                         + np.log(np.clip(probabilidad_mlp, 1e-12, None)))))):
            predicciones = np.array(cfg.CLASSES)[np.argmax(matriz_probabilidad, axis=1)]
            puntajes[etiqueta].append(float(f1_score(y[valid_idx], predicciones, average="macro",
                                                     labels=cfg.CLASSES)))
        print(f"    fold {fold['fold']}: promedio {puntajes['promedio'][-1]:.4f} | "
              f"logits {puntajes['logits'][-1]:.4f}", flush=True)
    return {"promedio": puntajes["promedio"], "logits": puntajes["logits"],
            "temperaturas": temperaturas_por_fold}


def resumen(puntajes: list[float]) -> dict:
    """Media, desviación y detalle por fold."""
    return {"macro_f1": round(float(np.mean(puntajes)), 4),
            "std": round(float(np.std(puntajes)), 4),
            "per_fold": [round(valor, 4) for valor in puntajes]}


def main() -> int:
    """Evalúa calibración por miembro y blending en logits."""
    set_seed()
    started = time.time()
    payload: dict = {"fase": "D — calibración y transformaciones de score", "fecha": "2026-09-25"}

    features, y, _, _, _ = build_features()
    features = np.concatenate([features, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    folds = cv.make_folds(y, groups=clusters["cluster"].to_numpy()[:len(y)], n_splits=5,
                          n_repeats=1, scheme="group")

    referencia = json.loads((cfg.EXPERIMENTS / "phase12_auditoria_estado.json")
                            .read_text(encoding="utf-8"))["reproduccion_lofo_v12"]["resultado"]
    payload["baseline"] = {"macro_f1": referencia["macro_f1"], "std": referencia["std"],
                           "per_fold": referencia["per_fold"], "origen": "fase 12"}

    resultados = []
    for etiqueta, temperaturas in (("D1a log-prob (T=1)", False), ("D1b log-prob + T in-fold", True)):
        print(f"[{etiqueta}]", flush=True)
        crudo = evaluar(features, y, folds, temperaturas)
        for variante, valores in (("promedio", crudo["promedio"]), ("logits", crudo["logits"])):
            resumen_variante = resumen(valores)
            delta = round(resumen_variante["macro_f1"] - referencia["macro_f1"], 4)
            positivos = sum(1 for nuevo, viejo in zip(resumen_variante["per_fold"],
                                                      referencia["per_fold"]) if nuevo > viejo)
            resultados.append({"variante": f"{etiqueta} · promedio de combinadores",
                               "n_features": int(features.shape[1]), **resumen_variante,
                               "delta_vs_v12": delta, "folds_positivos": positivos})
            print(f"  {variante}: {resumen_variante['macro_f1']:.4f} ({delta:+.4f}) "
                  f"· {positivos}/5 folds", flush=True)
        temperaturas_vistas = np.array(crudo["temperaturas"], dtype=float)
        payload.setdefault("temperaturas_por_fold", []).append(
            {"variante": etiqueta, "media_por_miembro": np.round(temperaturas_vistas.mean(axis=0),
                                                                 3).tolist()})

    payload["variantes"] = resultados
    mejor = max(resultados, key=lambda fila: fila["macro_f1"])
    payload["mejor"] = mejor["variante"]
    payload["criterio"] = "Δ ≥ +0,003 y ≥4/5 folds positivos"
    payload["adoptado"] = bool(mejor["delta_vs_v12"] >= 0.003 and mejor["folds_positivos"] >= 4)
    payload["segundos"] = round(time.time() - started, 1)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
    print(f"\nmejor: {mejor['variante']} = {mejor['macro_f1']:.4f} "
          f"({mejor['delta_vs_v12']:+.4f}) | adoptado: {payload['adoptado']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
