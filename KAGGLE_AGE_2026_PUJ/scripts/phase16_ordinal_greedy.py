"""FASE 16 — Señal ordinal auxiliar y selección greedy de miembros (hipótesis G).

Observación
    (G1) El meta recibe la fuente ordinal como un score continuo y probabilidades acumulativas, pero
    no una **edad implícita** agregada (esperanza del índice ordinal bajo cada distribución) ni
    medidas de **consistencia ordinal** entre miembros. La frontera central (18-24|25-34) es el
    cuello de botella y está a un paso ordinal de distancia.
    (G2) La selección/ponderación de miembros se ha explorado vía combinadores (Ridge, SGD) y
    combinaciones de subconjuntos de representaciones, pero nunca con **selección greedy con
    reemplazo** (Caruana), que es el método estándar de ensemble selection y no añade features
    nuevas.

Hipótesis
    H46 (G1) La edad implícita y la dispersión ordinal entre miembros informan las fronteras
         adyacentes más que los scores brutos.
    H47 (G2) Una media greedy de miembros seleccionados in-fold iguala o supera al meta-modelo de
         producción sin necesidad de features densas ni supervisadas.

Protocolo
    Mismas particiones agrupadas. G1: features sin etiqueta añadidas al meta (279+16) con el
    combinador de producción. G2: selección greedy con reemplazo sobre las filas de los folds de
    entrenamiento (hasta 10 pasos), evaluada en el fold de validación; se reporta la curva y el paso
    óptimo en train (sin seleccionar sobre validación).
    Criterio: Δ ≥ +0,003 y ≥4/5 folds positivos.

Salida: ``experiments/phase16_ordinal_greedy.json``.
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
from src import cv  # noqa: E402
from src.config import set_seed  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from phase2_ponderacion import EncodedMLP, MEMBER_FILES, build_features  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "phase16_ordinal_greedy.json"
PASOS_GREEDY = 10

FAMILIAS = {
    "svm_lineal": ["oof_E013_margenes", "oof_E016_margenes", "oof_E014_margenes"],
    "prob_lineal": ["oof_E-009", "oof_E-011"],
    "bayes": ["oof_E-010"],
    "ordinal": ["oof_E015_softmax", "oof_E021a_ordinal"],
    "prototipos": ["oof_E021b_rocchio"],
    "denso": ["oof_E019b_densas_svd"],
    "nbsvm": ["oof_ADV05_nbsvm"],
}


def probabilidad(nombre: str) -> np.ndarray | None:
    """Probabilidades por clase de un miembro (softmax si el bloque es de márgenes).

    Acepta tanto el nombre con extensión (``oof_E013_margenes.npy``) como el *stem*.
    """
    ruta = cfg.FEATURES / Path(nombre).name
    if not ruta.exists():
        ruta = cfg.FEATURES / f"{Path(nombre).stem}.npy"
    matriz = np.load(ruta)
    if matriz.ndim == 1 or matriz.shape[1] != len(cfg.CLASSES):
        return None
    matriz = np.asarray(matriz, dtype=np.float64)
    if matriz.min() < 0 or matriz.max(axis=1).max() > 1.5:
        matriz = matriz - matriz.max(axis=1, keepdims=True)
        matriz = np.exp(matriz)
    return matriz / matriz.sum(axis=1, keepdims=True)


def edad_implicita(probabilidades: np.ndarray) -> np.ndarray:
    """Esperanza del índice ordinal (0..5) bajo una distribución por clase."""
    return probabilidades @ np.arange(len(cfg.CLASSES))


def features_ordinales(train_idx: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Bloque G1: edad implícita por familia y medidas de consistencia ordinal.

    La orientación de la fuente ordinal continua se decide **solo con las filas de entrenamiento**
    del fold, para no usar etiquetas de validación ni siquiera en un signo.
    """
    familias = {}
    for familia, miembros in FAMILIAS.items():
        matrices = [probabilidad(miembro) for miembro in miembros]
        if all(matriz is not None for matriz in matrices):
            familias[familia] = np.mean(matrices, axis=0)
    edades = np.column_stack([edad_implicita(matriz) for matriz in familias.values()])
    continuo = np.load(cfg.FEATURES / "oof_E015_ordinal.npy").reshape(-1).astype(float)
    etiquetas = np.load(cfg.FEATURES / "oof_labels.npy", allow_pickle=True)
    indices = np.array([cfg.CLASSES.index(str(valor)) for valor in etiquetas])
    signo = 1.0 if np.corrcoef(continuo[train_idx], indices[train_idx])[0, 1] >= 0 else -1.0
    continuo = continuo * signo

    orden = np.argsort(np.argsort(edades, axis=0), axis=0)
    bloque = np.column_stack([
        edades,                                   # edad implícita por familia (7)
        edades.mean(axis=1), edades.std(axis=1),  # media y dispersión entre familias (2)
        edades.max(axis=1) - edades.min(axis=1),  # rango (1)
        continuo,                                 # fuente continua orientada (1)
        np.abs(continuo - edades.mean(axis=1)),   # desacuerdo con la media (1)
        orden.std(axis=1),                        # dispersión del ordenamiento (1)
        (np.abs(np.diff(orden, axis=1)) > 0).mean(axis=1),  # inconsistencia de orden (1)
        edades.mean(axis=1) - np.round(edades.mean(axis=1)),  # parte fraccionaria (1)
    ])
    nombres = ([f"g1_edad_{nombre}" for nombre in familias]
               + ["g1_edad_media", "g1_edad_desviacion", "g1_edad_rango", "g1_continuo",
                  "g1_desacuerdo_continuo", "g1_dispersion_orden", "g1_inconsistencia_orden",
                  "g1_parte_fraccionaria"])
    return bloque, nombres


def evaluar_ordinal(features: np.ndarray, y: np.ndarray, folds: list[dict],
                    indice_fold: int) -> dict:
    """Evalúa G1 con el bloque ordinal construido in-fold (signo decidido en entrenamiento)."""
    puntajes = []
    for fold in folds:
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        bloque, _ = features_ordinales(train_idx)
        matriz = np.concatenate([features, bloque], axis=1)
        logreg = Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                      random_state=cfg.SEED))])
        logreg.fit(matriz[train_idx], y[train_idx])
        mlp = EncodedMLP()
        mlp.fit(matriz[train_idx], y[train_idx])
        probabilidades = 0.5 * (logreg.predict_proba(matriz[valid_idx])
                                + mlp.model.predict_proba(mlp.scaler.transform(matriz[valid_idx])))
        predicciones = np.array(cfg.CLASSES)[np.argmax(probabilidades, axis=1)]
        puntajes.append(float(f1_score(y[valid_idx], predicciones, average="macro",
                                       labels=cfg.CLASSES)))
        print(f"    fold {fold['fold']}: {puntajes[-1]:.4f}", flush=True)
    del indice_fold
    return {"macro_f1": round(float(np.mean(puntajes)), 4), "std": round(float(np.std(puntajes)), 4),
            "per_fold": [round(valor, 4) for valor in puntajes]}


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
    return {"macro_f1": round(float(np.mean(puntajes)), 4), "std": round(float(np.std(puntajes)), 4),
            "per_fold": [round(valor, 4) for valor in puntajes]}


def macro_f1_rapido(reales: np.ndarray, predichos: np.ndarray) -> float:
    """Macro-F1 sobre índices de clase (implementación vectorizada)."""
    return float(f1_score(reales, predichos, average="macro",
                          labels=np.arange(len(cfg.CLASSES))))


def seleccion_greedy(pool: list[np.ndarray], y: np.ndarray, train_idx: np.ndarray,
                     steps: int = PASOS_GREEDY) -> tuple[list[int], list[float]]:
    """Selección greedy con reemplazo (Caruana) sobre las filas de entrenamiento del fold."""
    indices = np.array([cfg.CLASSES.index(str(valor)) for valor in y])
    seleccion: list[int] = []
    historial: list[float] = []
    actual = np.zeros_like(pool[0])
    for _ in range(steps):
        mejor_indice, mejor_puntaje = 0, -1.0
        for posicion, matriz in enumerate(pool):
            candidato = (actual[train_idx] * len(seleccion)
                         + matriz[train_idx]) / (len(seleccion) + 1)
            puntaje = macro_f1_rapido(indices[train_idx], candidato.argmax(axis=1))
            if puntaje > mejor_puntaje:
                mejor_indice, mejor_puntaje = posicion, puntaje
        seleccion.append(mejor_indice)
        actual = (actual * (len(seleccion) - 1) + pool[mejor_indice]) / len(seleccion)
        historial.append(round(float(mejor_puntaje), 4))
    return seleccion, historial


def main() -> int:
    """Evalúa la señal ordinal auxiliar (G1) y la selección greedy de miembros (G2)."""
    set_seed()
    started = time.time()
    payload: dict = {"fase": "G — ordinal auxiliar y greedy de miembros", "fecha": "2026-09-25"}

    features, y, _, _, _ = build_features()
    features = np.concatenate([features, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    folds = cv.make_folds(y, groups=clusters["cluster"].to_numpy()[:len(y)], n_splits=5,
                          n_repeats=1, scheme="group")
    referencia = json.loads((cfg.EXPERIMENTS / "phase12_auditoria_estado.json")
                            .read_text(encoding="utf-8"))["reproduccion_lofo_v12"]["resultado"]
    payload["baseline"] = {"macro_f1": referencia["macro_f1"], "std": referencia["std"],
                           "per_fold": referencia["per_fold"], "origen": "fase 12"}

    # ---------------------------------------------------------------- G1: ordinal auxiliar
    bloques = "g1g2"
    if "--bloques" in sys.argv:
        bloques = sys.argv[sys.argv.index("--bloques") + 1]
    _, nombres_g1 = features_ordinales(np.arange(2000))
    if "g1" in bloques:
        print(f"[G1] bloque ordinal de {len(nombres_g1)} features (construido in-fold)", flush=True)
        resultado_g1 = evaluar_ordinal(features, y, folds, 0)
        positivos = sum(1 for nuevo, viejo in zip(resultado_g1["per_fold"], referencia["per_fold"])
                        if nuevo > viejo)
        payload["g1_ordinal"] = {"n_features": int(features.shape[1] + len(nombres_g1)),
                                 **resultado_g1,
                                 "delta_vs_v12": round(resultado_g1["macro_f1"]
                                                       - referencia["macro_f1"], 4),
                                 "folds_positivos": positivos}

    # ------------------------------------------------------------ G2: greedy de miembros
    puntajes_greedy, selecciones = [], []
    if "g2" in bloques:
        nombres_pool, pool = [], []
        for nombre, _ in MEMBER_FILES:
            matriz = probabilidad(nombre)
            if matriz is not None:
                pool.append(matriz)
                nombres_pool.append(nombre)
        print(f"[G2] miembros en el pool: {len(pool)}", flush=True)
    for fold in folds:
        if "g2" not in bloques:
            break
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        seleccion, historial = seleccion_greedy(pool, y, train_idx)
        # Paso óptimo en train (sin mirar validación) y evaluación honesta en validación.
        paso_optimo = int(np.argmax(historial)) + 1
        conteo = np.bincount(seleccion[:paso_optimo], minlength=len(pool))
        promedio = np.mean([pool[posicion][valid_idx] for posicion in seleccion[:paso_optimo]],
                           axis=0)
        puntaje = macro_f1_rapido(
            np.array([cfg.CLASSES.index(str(valor)) for valor in y[valid_idx]]),
            promedio.argmax(axis=1))
        puntajes_greedy.append(float(puntaje))
        selecciones.append({"fold": fold["fold"], "paso_optimo": paso_optimo,
                            "historial_train": historial,
                            "miembros": {nombres_pool[posicion]: int(conteo[posicion])
                                         for posicion in np.argsort(-conteo) if conteo[posicion] > 0}})
        print(f"    fold {fold['fold']}: greedy {puntaje:.4f} (paso {paso_optimo})", flush=True)
    payload["g2_greedy"] = {"macro_f1": round(float(np.mean(puntajes_greedy)), 4) if puntajes_greedy else 0.0,
                            "std": round(float(np.std(puntajes_greedy)), 4) if puntajes_greedy else 0.0,
                            "per_fold": [round(valor, 4) for valor in puntajes_greedy],
                            "delta_vs_v12": round(float(np.mean(puntajes_greedy))
                                                  - referencia["macro_f1"], 4)
                            if puntajes_greedy else 0.0,
                            "selecciones": selecciones}

    for clave in [c for c in ("g1_ordinal", "g2_greedy") if c in payload]:
        seccion = payload[clave]
        seccion["adoptado"] = bool(seccion["delta_vs_v12"] >= 0.003)
    payload["segundos"] = round(time.time() - started, 1)
    ruta_resultados = cfg.EXPERIMENTS / f"phase16_ordinal_greedy_{bloques}.json"
    ruta_resultados.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                               encoding="utf-8")
    print(json.dumps({clave: {k: valor for k, valor in payload[clave].items()
                              if k in ("macro_f1", "delta_vs_v12", "adoptado", "folds_positivos")}
                      for clave in [c for c in ("g1_ordinal", "g2_greedy") if c in payload]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
