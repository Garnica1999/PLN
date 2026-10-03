"""FASE 14 — Consenso y desacuerdo por familia como features del combinador (hipótesis C).

Observación
    ADV-04 midió el bloque **genérico** de incertidumbre por miembro (confianza, margen, entropía,
    acuerdo) sobre el baseline v09 con meta LogReg: **+0,0002** y se cerró. Lo que no se ha probado
    es la agregación por **familia** de modelos, las **distancias de desacuerdo** entre familias y la
    modulación **contextual** de esa información. El diagnóstico L1 (25,94% de filas rescatables)
    muestra que "quién acierta" no está en el score agregado que ve el combinador.

Hipótesis
    H39 (C1) El acuerdo entre familias y la entropía del voto familiar contienen información que los
         scores por miembro no expresan linealmente.
    H40 (C2) La divergencia Jensen-Shannon entre familias discrimina cuándo el consenso informa.
    H41 (C3) El valor del consenso depende del contexto (longitud, tipo de tweet, flags): las
         interacciones consenso × contexto permiten el enrutamiento suave que la segmentación dura no
         consiguió.

Protocolo
    Leave-one-fold-out con las **mismas particiones agrupadas**. Todas las features nuevas derivan de
    los OOF de los miembros (sin etiquetas), por lo que no requieren ajuste in-fold. Variantes:
    279 (referencia de la fase 12) -> +C1 -> +C1+C2 -> +C1+C2+C3, con el combinador de producción
    (promedio MLP + LogReg). Criterio de adopción: Δ ≥ +0,003 y ≥4/5 folds positivos.

Salida: ``experiments/phase14_consenso_familias.json``.
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
from phase2_ponderacion import EncodedMLP, build_features

RESULTS_PATH = cfg.EXPERIMENTS / "phase14_consenso_familias.json"

#: Familias de miembros (pesos de la evidencia L1: diversidad por sesgo).
FAMILIAS = {
    "svm_lineal": ["oof_E013_margenes", "oof_E016_margenes", "oof_E014_margenes"],
    "prob_lineal": ["oof_E-009", "oof_E-011"],
    "bayes": ["oof_E-010"],
    "ordinal": ["oof_E015_softmax", "oof_E021a_ordinal"],
    "prototipos": ["oof_E021b_rocchio"],
    "denso": ["oof_E019b_densas_svd"],
    "nbsvm": ["oof_ADV05_nbsvm"],
}
#: Pares de familias para las distancias de desacuerdo (las más diversas según L1).
PARES_JS = [("svm_lineal", "ordinal"), ("svm_lineal", "prob_lineal"), ("svm_lineal", "nbsvm"),
            ("ordinal", "nbsvm"), ("denso", "ordinal")]


def probabilidad(nombre: str) -> np.ndarray | None:
    """Probabilidades por clase de un miembro (softmax si el bloque es de márgenes).

    Acepta tanto el nombre con extensión como el *stem*.
    """
    ruta = cfg.FEATURES / Path(nombre).name
    if not ruta.exists():
        ruta = cfg.FEATURES / f"{Path(nombre).stem}.npy"
    matriz = np.load(ruta)
    if matriz.ndim == 1 or matriz.shape[1] != len(cfg.CLASSES):
        return None
    matriz = np.asarray(matriz, dtype=np.float64)
    if matriz.min() < 0 or matriz.max(axis=1).max() > 1.5:  # márgenes -> softmax
        matriz = matriz - matriz.max(axis=1, keepdims=True)
        matriz = np.exp(matriz)
    return matriz / matriz.sum(axis=1, keepdims=True)


def divergencia_js(izquierda: np.ndarray, derecha: np.ndarray) -> np.ndarray:
    """Divergencia Jensen-Shannon por filas entre dos matrices de probabilidad."""
    medio = 0.5 * (izquierda + derecha)

    def kl(uno: np.ndarray, otro: np.ndarray) -> np.ndarray:
        return (uno * np.log(np.clip(uno, 1e-12, None) / np.clip(otro, 1e-12, None))).sum(axis=1)

    return 0.5 * kl(izquierda, medio) + 0.5 * kl(derecha, medio)


def bloques_consenso(dense: pd.DataFrame, tipo: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Construye los bloques C1 (acuerdo), C2 (distancias) y C3 (gating contextual)."""
    probabilidades = {}
    for familia, miembros in FAMILIAS.items():
        matrices = [probabilidad(miembro) for miembro in miembros]
        if all(matriz is not None for matriz in matrices):
            probabilidades[familia] = np.mean(matrices, axis=0)
    nombres_familia = list(probabilidades)
    votos = np.column_stack([probabilidades[familia].argmax(axis=1) for familia in nombres_familia])
    # Reparto de votos familiares por clase.
    reparto = np.zeros((len(votos), len(cfg.CLASSES)))
    for fila in range(votos.shape[1]):
        reparto[np.arange(len(votos)), votos[:, fila]] += 1
    reparto = reparto / votos.shape[1]
    moda = reparto.argmax(axis=1)
    acuerdo = (votos == moda[:, None]).astype(float)
    entropia = -(np.where(reparto > 0, reparto * np.log(np.maximum(reparto, 1e-12)), 0.0)).sum(axis=1)
    ordenado = np.sort(reparto, axis=1)

    bloque_c1 = np.column_stack([acuerdo, acuerdo.sum(axis=1), entropia,
                                 ordenado[:, -1] - ordenado[:, -2],
                                 (votos == votos[:, [0]]).sum(axis=1)])
    nombres_c1 = ([f"c1_familia_{nombre}_vota_moda" for nombre in nombres_familia]
                  + ["c1_n_familias_con_moda", "c1_entropia_familias", "c1_margen_moda",
                     "c1_n_familias_con_svm"])

    distancias = [divergencia_js(probabilidades[izquierda], probabilidades[derecha])
                  for izquierda, derecha in PARES_JS]
    todas = [divergencia_js(probabilidades[a], probabilidades[b])
             for posicion, a in enumerate(nombres_familia) for b in nombres_familia[posicion + 1:]]
    bloque_c2 = np.column_stack(distancias + [np.mean(todas, axis=0), np.max(todas, axis=0)])
    nombres_c2 = ([f"c2_js_{a}_{b}" for a, b in PARES_JS] + ["c2_js_media", "c2_js_max"])

    banda = pd.qcut(dense["n_chars"], 4, labels=False).to_numpy()
    contexto = np.column_stack([
        np.eye(4)[banda],
        dense["n_mention"].gt(0).astype(float).to_numpy(),
        dense["n_hashtag"].gt(0).astype(float).to_numpy(),
        dense["n_url"].gt(0).astype(float).to_numpy(),
        dense["is_rt"].astype(float).to_numpy(),
    ])
    # Gating: reparto de votos × banda de longitud (24) + entropía × contexto (5).
    interacciones = [reparto * contexto[:, [indice]] for indice in range(4)]
    interacciones.append(entropia[:, None] * contexto[:, 4:])
    bloque_c3 = np.column_stack(interacciones)
    nombres_c3 = ([f"c3_reparto_x_banda_{indice}_{clase}" for indice in range(4)
                   for clase in cfg.CLASSES]
                  + [f"c3_entropia_x_contexto_{indice}" for indice in range(contexto.shape[1] - 4)])
    del tipo

    return (np.column_stack([bloque_c1, bloque_c2, bloque_c3]),
            nombres_c1 + nombres_c2 + nombres_c3)


def evaluar(features: np.ndarray, y: np.ndarray, folds: list[dict]) -> dict:
    """Macro-F1 LOFO del combinador de producción (promedio MLP + LogReg)."""
    puntajes = []
    for fold in folds:
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        logreg = Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                      random_state=cfg.SEED))])
        logreg.fit(features[train_idx], y[train_idx])
        mlp = EncodedMLP()
        mlp.fit(features[train_idx], y[train_idx])
        probabilidades = 0.5 * (logreg.predict_proba(features[valid_idx])
                                + mlp.model.predict_proba(mlp.scaler.transform(features[valid_idx])))
        predicciones = np.array(cfg.CLASSES)[np.argmax(probabilidades, axis=1)]
        puntajes.append(float(f1_score(y[valid_idx], predicciones, average="macro",
                                       labels=cfg.CLASSES)))
        print(f"    fold {fold['fold']}: {puntajes[-1]:.4f}", flush=True)
    return {"macro_f1": round(float(np.mean(puntajes)), 4),
            "std": round(float(np.std(puntajes)), 4),
            "per_fold": [round(valor, 4) for valor in puntajes]}


def main() -> int:
    """Evalúa los bloques de consenso y desacuerdo sobre el modelo v12."""
    set_seed()
    started = time.time()
    payload: dict = {"fase": "C — consenso, desacuerdo y gating", "fecha": "2026-09-25"}

    features, y, dense, tipo, _ = build_features()
    features = np.concatenate([features, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    folds = cv.make_folds(y, groups=clusters["cluster"].to_numpy()[:len(y)], n_splits=5,
                          n_repeats=1, scheme="group")

    consenso, nombres = bloques_consenso(dense, tipo)
    assert consenso.shape[1] == len(nombres), (
        f"desalineación de nombres: {consenso.shape[1]} columnas vs {len(nombres)} nombres")
    print(f"[features] consenso {consenso.shape} | nombres {len(nombres)}", flush=True)
    payload["features_nuevas"] = {"n": int(consenso.shape[1]), "nombres": nombres}

    referencia_path = cfg.EXPERIMENTS / "phase12_auditoria_estado.json"
    referencia = json.loads(referencia_path.read_text(encoding="utf-8"))["reproduccion_lofo_v12"]
    baseline = referencia["resultado"]
    payload["baseline"] = {"macro_f1": baseline["macro_f1"], "std": baseline["std"],
                           "per_fold": baseline["per_fold"], "origen": "reproducción de la fase 12"}

    mascaras = {"+C1": [nombre.startswith("c1") for nombre in nombres],
                "+C1+C2": [nombre.startswith(("c1", "c2")) for nombre in nombres],
                "+C1+C2+C3": [True] * len(nombres)}

    resultados = []
    for etiqueta, mascara in mascaras.items():
        matriz = np.concatenate([features, consenso[:, mascara]], axis=1)
        print(f"[{etiqueta}] {matriz.shape}", flush=True)
        resultado = evaluar(matriz, y, folds)
        delta = round(resultado["macro_f1"] - baseline["macro_f1"], 4)
        positivos = sum(1 for nuevo, viejo in zip(resultado["per_fold"], baseline["per_fold"])
                        if nuevo > viejo)
        resultados.append({"variante": etiqueta, "n_features": int(matriz.shape[1]),
                           **resultado, "delta_vs_v12": delta, "folds_positivos": positivos})
        print(f"[{etiqueta}] {resultado['macro_f1']:.4f} ({delta:+.4f}) · {positivos}/5 folds",
              flush=True)

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
