"""FASE 18 — Limpieza selectiva y reponderación del entrenamiento del combinador (hipótesis F).

Observación
    La fase 11 midió la limpieza **global** a nivel de combinador: excluir las 643 filas
    contradictorias aportó +0,0001 (3/5 folds) y excluir las 2.123 filas con texto duplicado
    +0,0016 (3/5 folds), ambas por debajo del umbral. Quedan sin probar dos variantes distintas:
    (a) conservar **un representante** por grupo duplicado en vez de eliminar todas las copias, y
    (b) aplicarla de forma **selectiva** a las clases centrales, donde se concentra el error.

Hipótesis
    H52 (F1) Conservar un representante por texto exacto reponderar el contenido repetido sin perder
         el caso (más suave que eliminar todas las copias).
    H53 (F2) Conservar un representante por **clúster de near-duplicados** reduce la sobre-representación
         de variantes casi idénticas, que es el mecanismo que el análisis de errores señala.
    H54 (F3) La limpieza selectiva en las clases centrales (18-24, 25-34, 35-49) preserva las clases
         extremas (bien separadas) y ataca solo la frontera.

Protocolo
    LOFO con las mismas particiones; la limpieza se aplica **solo a las filas de entrenamiento** de
    cada fold (la validación queda intacta, porque el test contendrá casos igualmente repetidos).
    Combinador de producción (promedio MLP + LogReg). Criterio: Δ ≥ +0,003 y ≥4/5 folds positivos.

Salida: ``experiments/phase18_limpieza_selectiva.json``.
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
from phase11_ruido_etiqueta import _audit_normalize  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "phase18_limpieza_selectiva.json"
CLASES_CENTRALES = ("18-24", "25-34", "35-49")


def representantes(clave: np.ndarray, train_idx: np.ndarray,
                   semilla: int = cfg.SEED) -> np.ndarray:
    """Índices de las filas de entrenamiento a **conservar** (una por grupo).

    La elección del representante es determinista: se ordena por ``(clave, desempate aleatorio)`` y
    se conserva la primera fila de cada grupo. Vectorizado para evitar bucles por grupo.

    Parameters
    ----------
    clave : numpy.ndarray
        Clave de agrupamiento por fila (texto normalizado, clúster o clave compuesta selectiva).
    train_idx : numpy.ndarray
        Filas del lado de entrenamiento del fold.
    semilla : int
        Semilla del desempate.

    Returns
    -------
    numpy.ndarray
        Subconjunto ordenado de ``train_idx`` con un representante por grupo.
    """
    generador = np.random.default_rng(semilla)
    marco = pd.DataFrame({"clave": clave[train_idx],
                          "desempate": generador.random(len(train_idx)),
                          "posicion": train_idx})
    marco = marco.sort_values(["clave", "desempate"], kind="mergesort")
    return np.sort(marco.drop_duplicates(subset="clave", keep="first")["posicion"].to_numpy())


def evaluar_variante(features: np.ndarray, y: np.ndarray, folds: list[dict],
                     clave: np.ndarray) -> dict:
    """Evalúa una variante de limpieza con el combinador de producción."""
    puntajes, tamanos = [], []
    for fold in folds:
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        conservados = representantes(clave, train_idx)
        tamanos.append({"fold": fold["fold"], "train_original": int(len(train_idx)),
                        "train_tras_limpieza": int(len(conservados)),
                        "filas_eliminadas": int(len(train_idx) - len(conservados))})
        logreg = Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                      random_state=cfg.SEED))])
        logreg.fit(features[conservados], y[conservados])
        mlp = EncodedMLP()
        mlp.fit(features[conservados], y[conservados])
        probabilidades = 0.5 * (logreg.predict_proba(features[valid_idx])
                                + mlp.model.predict_proba(mlp.scaler.transform(features[valid_idx])))
        predicciones = np.array(cfg.CLASSES)[np.argmax(probabilidades, axis=1)]
        puntajes.append(float(f1_score(y[valid_idx], predicciones, average="macro",
                                       labels=cfg.CLASSES)))
        print(f"    fold {fold['fold']}: {puntajes[-1]:.4f} "
              f"(train {len(conservados):,}/{len(train_idx):,})", flush=True)
    return {"macro_f1": round(float(np.mean(puntajes)), 4),
            "std": round(float(np.std(puntajes)), 4),
            "per_fold": [round(valor, 4) for valor in puntajes],
            "tamanos_train": tamanos}


def main() -> int:
    """Evalúa las variantes de limpieza selectiva y reponderación."""
    set_seed()
    started = time.time()
    payload: dict = {"fase": "F — limpieza selectiva y reponderación", "fecha": "2026-09-25"}

    train = data.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    features, y2, _, _, _ = build_features()
    features = np.concatenate([features, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    assert np.array_equal(y, y2)
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    ids_cluster = clusters["cluster"].to_numpy()[:len(y)].astype(str)
    textos = _audit_normalize(train[cfg.TEXT_COL]).to_numpy()
    folds = cv.make_folds(y, groups=clusters["cluster"].to_numpy()[:len(y)], n_splits=5,
                          n_repeats=1, scheme="group")
    referencia = json.loads((cfg.EXPERIMENTS / "phase12_auditoria_estado.json")
                            .read_text(encoding="utf-8"))["reproduccion_lofo_v12"]["resultado"]
    payload["baseline"] = {"macro_f1": referencia["macro_f1"], "std": referencia["std"],
                           "per_fold": referencia["per_fold"], "origen": "fase 12"}

    # Clave compuesta para la variante selectiva: solo las clases centrales se agrupan.
    central = np.isin(y, CLASES_CENTRALES)
    identidad = np.char.add("C_", ids_cluster)
    unico = np.char.add("U_", np.arange(len(y)).astype(str))
    selectivo = np.where(central, identidad, unico)

    variantes = {
        "F1 texto exacto (un representante)": textos,
        "F2 clúster de near-duplicados (un representante)": ids_cluster,
        "F3 clúster selectivo en clases centrales": selectivo,
    }
    seleccionadas = list(variantes)
    if "--variantes" in sys.argv:
        pedidas = sys.argv[sys.argv.index("--variantes") + 1].split(",")
        seleccionadas = [nombre for nombre in variantes
                         if any(nombre.startswith(pedida) for pedida in pedidas)]
    print(f"[variantes] {seleccionadas}", flush=True)
    resultados = []
    for etiqueta in seleccionadas:
        clave = variantes[etiqueta]
        print(f"[{etiqueta}]", flush=True)
        resultado = evaluar_variante(features, y, folds, clave)
        positivos = sum(1 for nuevo, viejo in zip(resultado["per_fold"], referencia["per_fold"])
                        if nuevo > viejo)
        resultados.append({"variante": etiqueta, **resultado,
                           "delta_vs_v12": round(resultado["macro_f1"]
                                                 - referencia["macro_f1"], 4),
                           "folds_positivos": positivos})
        print(f"[{etiqueta}] {resultado['macro_f1']:.4f} "
              f"({resultados[-1]['delta_vs_v12']:+.4f}) · {positivos}/5 folds", flush=True)

    payload["variantes"] = resultados
    mejor = max(resultados, key=lambda fila: fila["macro_f1"])
    payload["mejor"] = mejor["variante"]
    payload["adoptado"] = bool(mejor["delta_vs_v12"] >= 0.003 and mejor["folds_positivos"] >= 4)
    payload["segundos"] = round(time.time() - started, 1)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
    print(f"\nmejor: {mejor['variante']} = {mejor['macro_f1']:.4f} "
          f"({mejor['delta_vs_v12']:+.4f}) | adoptado: {payload['adoptado']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
