"""FASE 19 — Confirmación en el holdout congelado de una variante candidata (FASE H).

Protocolo
    Idéntico al de las fases 3 y 8: holdout = 15% de los clústeres de near-duplicados con semilla
    2026 (las mismas 25.204 filas). Los combinadores se ajustan **solo** con las filas de desarrollo y
    se evalúan en el holdout. Se re-ajusta también la referencia v12 en la misma corrida para obtener
    una comparación pareada honesta (mismo código, mismas particiones, misma semilla).

Uso
    python scripts/phase19_holdout_variante.py --variante C|C12|C123|D1a|D1b|Elibres|Esuper|G1|G2|F1|F2|F3|base

Salida: ``experiments/phase19_holdout_<variante>.json``.
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
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import cv, data  # noqa: E402
from src.config import set_seed  # noqa: E402

from phase2_ponderacion import EncodedMLP, build_features  # noqa: E402

HOLDOUT_SEED, HOLDOUT_SHARE = 2026, 0.15
RESULTADOS = cfg.EXPERIMENTS
sys.path.insert(0, str(ROOT / "scripts"))


def reparto_holdout(cluster_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Índices de desarrollo y de holdout congelado (idénticos a las fases 3 y 8)."""
    generador = np.random.default_rng(HOLDOUT_SEED)
    elegidos = set(generador.choice(np.unique(cluster_ids),
                                    size=int(len(np.unique(cluster_ids)) * HOLDOUT_SHARE),
                                    replace=False).tolist())
    en_holdout = np.array([cluster in elegidos for cluster in cluster_ids])
    return np.where(~en_holdout)[0], np.where(en_holdout)[0]


def ajustar_combinadores(features: np.ndarray, y: np.ndarray, filas: np.ndarray):
    """Ajusta el par de combinadores (LogReg + MLP) sobre las filas indicadas."""
    logreg = Pipeline([("scaler", StandardScaler()),
                       ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                  random_state=cfg.SEED))])
    logreg.fit(features[filas], y[filas])
    mlp = EncodedMLP()
    mlp.fit(features[filas], y[filas])
    return logreg, mlp


def metricas(matriz: np.ndarray, y: np.ndarray, filas: np.ndarray) -> dict:
    """Macro-F1 y accuracy de una matriz de probabilidades sobre las filas indicadas."""
    predicciones = np.array(cfg.CLASSES)[np.argmax(matriz, axis=1)]
    return {"macro_f1": round(float(f1_score(y[filas], predicciones, average="macro",
                                             labels=cfg.CLASSES)), 4),
            "accuracy": round(float((predicciones == y[filas]).mean()), 4)}


def construir(variante: str, base: np.ndarray, train: pd.DataFrame, test: pd.DataFrame,
              y: np.ndarray, dense: pd.DataFrame, clusters: np.ndarray,
              dev_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (matriz, filas_de_entrenamiento_para_el_combinador)."""
    from phase14_consenso_familias import bloques_consenso
    from phase16_ordinal_greedy import features_ordinales
    from phase17_estructura_v2 import features_libres, margenes_mencion
    from phase18_limpieza_selectiva import CLASES_CENTRALES, representantes
    from phase11_ruido_etiqueta import _audit_normalize
    from phase15_calibracion_scores import limites_de_bloques, log_softmax, ajustar_temperatura

    filas = dev_idx
    if variante == "base":
        return base, filas
    if variante.startswith("C"):
        bloque, nombres = bloques_consenso(dense, None)
        mascara = {"C": [nombre.startswith("c1") for nombre in nombres],
                   "C12": [nombre.startswith(("c1", "c2")) for nombre in nombres],
                   "C123": [True] * len(nombres)}[variante]
        return np.concatenate([base, bloque[:, mascara]], axis=1), filas
    if variante in {"D1a", "D1b"}:
        matriz = base.copy()
        for inicio, ancho in limites_de_bloques():
            if ancho != len(cfg.CLASSES):
                continue
            bloque = matriz[:, inicio:inicio + ancho]
            temperatura = (ajustar_temperatura(bloque[dev_idx], y[dev_idx])
                           if variante == "D1b" else 1.0)
            matriz[:, inicio:inicio + ancho] = log_softmax(bloque, temperatura)
        return matriz, filas
    if variante == "Elibres":
        bloque, _ = features_libres(train[cfg.TEXT_COL])
        return np.concatenate([base, bloque], axis=1), filas
    if variante == "Esuper":
        bloque = margenes_mencion(train[cfg.TEXT_COL], y, dev_idx)
        return np.concatenate([base, bloque], axis=1), filas
    if variante == "G1":
        bloque, _ = features_ordinales(dev_idx)
        return np.concatenate([base, bloque], axis=1), filas
    if variante.startswith("Vista"):
        componentes = {"VistaEstructura": ["estructura"],
                       "VistaAmpliada": ["ampliada"],
                       "VistaExtremos": ["extremos"],
                       "VistaEstructuraExtremos": ["estructura", "extremos"],
                       "VistaAmpliadaExtremos": ["ampliada", "extremos"]}[variante]
        columnas = [np.load(cfg.FEATURES / f"vista_{nombre}_margenes.npy")
                    for nombre in componentes]
        return np.concatenate([base] + columnas, axis=1), filas
    if variante.startswith("F"):
        ids_cluster = clusters.astype(str)
        textos = _audit_normalize(train[cfg.TEXT_COL]).to_numpy()
        central = np.isin(y, CLASES_CENTRALES)
        selectivo = np.where(central, np.char.add("C_", ids_cluster),
                             np.char.add("U_", np.arange(len(y)).astype(str)))
        clave = {"F1": textos, "F2": ids_cluster, "F3": selectivo}[variante]
        return base, representantes(clave, dev_idx)
    raise ValueError(f"variante desconocida: {variante}")


def main() -> int:
    """Confirma una variante en el holdout congelado frente a la referencia v12."""
    variante = "base"
    if "--variante" in sys.argv:
        variante = sys.argv[sys.argv.index("--variante") + 1]
    set_seed()
    started = time.time()
    payload: dict = {"fase": "H — holdout congelado", "variante": variante, "fecha": "2026-09-25"}

    train = data.load_train()
    test = data.load_test()
    y = train[cfg.TARGET].astype(str).to_numpy()
    features, _, dense, _, _ = build_features()
    base = np.concatenate([features, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    dev_idx, holdout_idx = reparto_holdout(clusters)
    payload["holdout"] = {"filas_desarrollo": int(len(dev_idx)), "filas_holdout": int(len(holdout_idx)),
                          "semilla": HOLDOUT_SEED, "fraccion": HOLDOUT_SHARE}

    resultados = {}
    for etiqueta in ("base", variante):
        if etiqueta in resultados:
            continue
        matriz, filas_entrenamiento = construir(etiqueta, base, train, test, y, dense, clusters,
                                                dev_idx)
        logreg, mlp = ajustar_combinadores(matriz, y, filas_entrenamiento)
        probabilidad_logreg = logreg.predict_proba(matriz[holdout_idx])
        probabilidad_mlp = mlp.model.predict_proba(mlp.scaler.transform(matriz[holdout_idx]))
        promedios = {
            "LogReg": probabilidad_logreg,
            "MLP": probabilidad_mlp,
            "promedio 0,5/0,5": 0.5 * (probabilidad_logreg + probabilidad_mlp),
        }
        resultados[etiqueta] = {clave: metricas(valor, y, holdout_idx)
                                for clave, valor in promedios.items()}
        resultados[etiqueta]["n_features"] = int(matriz.shape[1])
        resultados[etiqueta]["filas_entrenamiento"] = int(len(filas_entrenamiento))
        print(f"[holdout] {etiqueta}: {resultados[etiqueta]['promedio 0,5/0,5']}", flush=True)

    if variante != "base" and "base" in resultados:
        delta = round(resultados[variante]["promedio 0,5/0,5"]["macro_f1"]
                      - resultados["base"]["promedio 0,5/0,5"]["macro_f1"], 4)
        payload["delta_vs_base_misma_corrida"] = delta
        payload["criterio"] = "Δ ≥ +0,003 en el holdout congelado"
        payload["confirmado"] = bool(delta >= 0.003)
    payload["resultados"] = resultados
    payload["segundos"] = round(time.time() - started, 1)
    ruta = RESULTADOS / f"phase19_holdout_{variante}.json"
    ruta.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"resultado: {ruta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
