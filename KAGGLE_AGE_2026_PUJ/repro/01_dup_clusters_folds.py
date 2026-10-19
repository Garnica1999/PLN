"""Spike 01 - Reproducibilidad de dup_clusters y de los folds canonicos (Fase 0 del notebook Kaggle).

Objetivo: verificar que la reconstruccion desde los CSV crudos reproduce EXACTAMENTE:
  A) la particion de dup_clusters.parquet (clusters de near-duplicados sobre train+test);
  B) fold_idx.npy (artifacts/p33_scores/todos) a partir de los clusters GUARDADOS;
  C) fold_idx.npy a partir de los clusters RECONSTRUIDOS.

No modifica artefactos del proyecto: solo lee e imprime un informe. ASCII puro.

Uso:
    python repro/01_dup_clusters_folds.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import config as cfg  # noqa: E402
from src import cv  # noqa: E402
from src import data as datos  # noqa: E402


def diagnostico_particion(guardada: np.ndarray, reconstruida: np.ndarray) -> tuple[int, int, float]:
    """Compara dos particiones cuyas etiquetas pueden diferir.

    Devuelve (splits, merges, acuerdo):
      - splits: clusters guardados cuyos miembros caen en mas de un cluster reconstruido.
      - merges: clusters reconstruidos que mezclan miembros de mas de un cluster guardado.
      - acuerdo: fraccion de filas cuya etiqueta reconstruida mapea (por mayoria) a la guardada.
    """
    marco = pd.DataFrame({"g": guardada, "r": reconstruida})
    splits = int((marco.groupby("g")["r"].nunique() > 1).sum())
    merges = int((marco.groupby("r")["g"].nunique() > 1).sum())
    mapa = marco.groupby("r")["g"].agg(lambda s: s.value_counts().index[0])
    acuerdo = float((marco["r"].map(mapa) == marco["g"]).mean())
    return splits, merges, acuerdo


def fold_index(y: np.ndarray, grupos: np.ndarray) -> np.ndarray:
    """Reproduce el fold_idx de p33_00_scores_url.py (StratifiedGroupKFold, semilla 42)."""
    folds = cv.make_folds(y, groups=grupos, n_splits=5, n_repeats=1, scheme="group")
    indice = np.zeros(len(y), dtype=np.int64)
    for pliegue in folds:
        indice[np.asarray(pliegue["val_idx"])] = int(pliegue["fold"])
    return indice


def main() -> int:
    """Ejecuta el spike completo e imprime el informe."""
    t0 = time.time()
    train = datos.load_train()
    test = datos.load_test()
    y = train[cfg.TARGET].astype(str).to_numpy()
    print(f"[spike01] train={len(train):,} test={len(test):,}")

    guardado = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    g = guardado["cluster"].to_numpy()
    print(f"[spike01] clusters guardados: {pd.Series(g).nunique():,} unicos")

    textos = pd.concat([train[cfg.TEXT_COL], test[cfg.TEXT_COL]], ignore_index=True)
    t1 = time.time()
    nuevo = cv.build_dup_clusters(textos, stats={})
    t_rebuild = time.time() - t1
    r = nuevo["cluster"].to_numpy()
    print(f"[spike01] rebuild clusters: {t_rebuild:.1f}s | {pd.Series(r).nunique():,} unicos")

    splits, merges, acuerdo = diagnostico_particion(g, r)
    print(f"[spike01] A) particion: splits={splits} merges={merges} acuerdo={acuerdo:.8f}")

    referencia = np.load(RAIZ / "artifacts" / "p33_scores" / "todos" / "fold_idx.npy")
    fi_guardado = fold_index(y, g[: len(train)])
    acuerdo_b = float((fi_guardado == referencia).mean())
    print(f"[spike01] B) folds desde clusters guardados vs fold_idx.npy: acuerdo={acuerdo_b:.8f}")

    fi_reconstruido = fold_index(y, r[: len(train)])
    acuerdo_c = float((fi_reconstruido == referencia).mean())
    print(f"[spike01] C) folds desde clusters reconstruidos vs fold_idx.npy: acuerdo={acuerdo_c:.8f}")

    print(f"[spike01] tiempo total {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
