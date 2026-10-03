"""FASE 15b — Diversidad de **subconjuntos de features** en el segundo nivel (hipótesis D5).

Observación
    La fase 9 (PH9) probó añadir un tercer combinador cambiando el **algoritmo** (Ridge, SGD) sobre
    las mismas 279 features: no aportó. Pero el principio que sí funcionó en las fases 7 y 8 fue la
    **diversidad** (MLP no lineal + LogReg lineal -> +0,0027), y ese mismo principio no se ha probado
    cambiando la **vista de features**: un combinador sobre las 279 columnas frente a uno sobre el
    subconjunto "solo miembros" (los 78 scores del primer nivel + 6 del NB-SVM de palabras = 84),
    que es la vista de stacking pura sin features densas, estructuradas ni supervisadas.

Hipótesis
    H55 (D5) Promediar el combinador de producción con un combinador entrenado en un **subconjunto
         distinto de features** aporta diversidad real (distinta de la diversidad de algoritmo ya
         explorada) y supera al promedio de dos.

Protocolo
    LOFO con las mismas particiones. Por fold: MLP y LogReg sobre las 279 (producción) y MLP y LogReg
    sobre las 84 de stacking puro. Se evalúan las medias de 2, 3 y 4 combinadores contra la
    referencia de la fase 12. Criterio: Δ ≥ +0,003 y ≥4/5 folds positivos.

Salida: ``experiments/phase15b_diversidad_subsets.json``.
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

RESULTS_PATH = cfg.EXPERIMENTS / "phase15b_diversidad_subsets.json"


def columnas_miembros(ancho_total: int) -> np.ndarray:
    """Índices del subconjunto 'stacking puro': scores de miembros + NB-SVM de palabras."""
    ancho_miembros = 0
    for nombre, _ in MEMBER_FILES:
        matriz = np.load(cfg.FEATURES / nombre, mmap_mode="r")
        ancho_miembros += 1 if matriz.ndim == 1 else matriz.shape[1]
    return np.r_[0:ancho_miembros, ancho_total - 6:ancho_total]


def ajustar(matriz: np.ndarray, y: np.ndarray, filas: np.ndarray, tipo: str):
    """Ajusta el combinador indicado ('mlp' o 'logreg')."""
    if tipo == "mlp":
        modelo = EncodedMLP()
        modelo.fit(matriz[filas], y[filas])
        return lambda indices: modelo.model.predict_proba(modelo.scaler.transform(matriz[indices]))
    modelo = Pipeline([("scaler", StandardScaler()),
                       ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                  random_state=cfg.SEED))])
    modelo.fit(matriz[filas], y[filas])
    return lambda indices: modelo.predict_proba(matriz[indices])


def main() -> int:
    """Evalúa el promedio de combinadores con vistas de features diversas."""
    set_seed()
    started = time.time()
    payload: dict = {"fase": "D5 — diversidad de subconjuntos", "fecha": "2026-09-25"}

    features, y, _, _, _ = build_features()
    features = np.concatenate([features, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    seleccion = columnas_miembros(features.shape[1])
    print(f"[subset] {len(seleccion)} columnas de stacking puro de {features.shape[1]}", flush=True)
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    folds = cv.make_folds(y, groups=clusters["cluster"].to_numpy()[:len(y)], n_splits=5,
                          n_repeats=1, scheme="group")
    referencia = json.loads((cfg.EXPERIMENTS / "phase12_auditoria_estado.json")
                            .read_text(encoding="utf-8"))["reproduccion_lofo_v12"]["resultado"]
    payload["baseline"] = {"macro_f1": referencia["macro_f1"], "std": referencia["std"],
                           "per_fold": referencia["per_fold"], "origen": "fase 12"}

    etiquetas = {
        "mlp_full": "MLP(64) sobre 279",
        "logreg_full": "LogReg sobre 279",
        "mlp_subset": "MLP(64) sobre 84 (stacking puro)",
        "logreg_subset": "LogReg sobre 84 (stacking puro)",
    }
    puntajes: dict[str, list[float]] = {clave: [] for clave in etiquetas}
    combinaciones = {
        "producción (MLP_full + LogReg_full)": ("mlp_full", "logreg_full"),
        "3 vías (+ MLP_subset)": ("mlp_full", "logreg_full", "mlp_subset"),
        "3 vías (+ LogReg_subset)": ("mlp_full", "logreg_full", "logreg_subset"),
        "4 vías": ("mlp_full", "logreg_full", "mlp_subset", "logreg_subset"),
    }
    puntajes.update({nombre: [] for nombre in combinaciones})

    for fold in folds:
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        probabilidades = {}
        for clave in etiquetas:
            tipo, vista = clave.split("_")
            matriz = features if vista == "full" else features[:, seleccion]
            probabilidades[clave] = ajustar(matriz, y, train_idx, tipo)(valid_idx)
        for clave, valor in probabilidades.items():
            predicciones = np.array(cfg.CLASSES)[np.argmax(valor, axis=1)]
            puntajes[clave].append(float(f1_score(y[valid_idx], predicciones, average="macro",
                                                  labels=cfg.CLASSES)))
        for nombre, claves in combinaciones.items():
            promedio = np.mean([probabilidades[clave] for clave in claves], axis=0)
            predicciones = np.array(cfg.CLASSES)[np.argmax(promedio, axis=1)]
            puntajes[nombre].append(float(f1_score(y[valid_idx], predicciones, average="macro",
                                                   labels=cfg.CLASSES)))
        print(f"    fold {fold['fold']}: producción {puntajes['producción (MLP_full + LogReg_full)'][-1]:.4f} "
              f"| 3 vías MLP_sub {puntajes['3 vías (+ MLP_subset)'][-1]:.4f} "
              f"| 4 vías {puntajes['4 vías'][-1]:.4f}", flush=True)

    resultados = []
    for nombre, valores in puntajes.items():
        media = round(float(np.mean(valores)), 4)
        positivos = sum(1 for nuevo, viejo in zip(valores, referencia["per_fold"]) if nuevo > viejo)
        resultados.append({"variante": nombre, "macro_f1": media,
                           "std": round(float(np.std(valores)), 4),
                           "per_fold": [round(valor, 4) for valor in valores],
                           "delta_vs_v12": round(media - referencia["macro_f1"], 4),
                           "folds_positivos": positivos})
    payload["variantes"] = resultados
    mejor = max([fila for fila in resultados if "vías" in fila["variante"] or "producción" in fila["variante"]],
                key=lambda fila: fila["macro_f1"])
    payload["mejor"] = mejor["variante"]
    payload["adoptado"] = bool(mejor["delta_vs_v12"] >= 0.003 and mejor["folds_positivos"] >= 4)
    payload["segundos"] = round(time.time() - started, 1)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
    for fila in resultados:
        print(f"  {fila['variante']:38s} {fila['macro_f1']:.4f} ({fila['delta_vs_v12']:+.4f}) "
              f"· {fila['folds_positivos']}/5")
    print(f"\nmejor: {mejor['variante']} | adoptado: {payload['adoptado']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
