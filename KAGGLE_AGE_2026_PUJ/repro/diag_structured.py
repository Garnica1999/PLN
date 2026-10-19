"""Diagnostico puntual: columnas de las features estructuradas que difieren de la referencia.

Lee el cache del spike de cadena (etapa features) y el parquet guardado, y reporta por columna
el maximo y la media de la diferencia, con los nombres de ambos lados. ASCII puro.

Uso (servidor): ~/age_notebook/venv_exacto/bin/python repro/diag_structured.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CACHE = Path.home() / "age_notebook" / "runs" / "cadena_cache"

sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "repro"))


def main() -> int:
    """Compara columna a columna la reconstruccion contra el parquet guardado."""
    from cadena import dense as dense_mod
    from cadena import structured as structured_mod

    with np.load(CACHE / "features.npz") as datos:
        estr = datos["estr"]
        estr_te = datos["estr_te"]
    ref = pd.read_parquet(RAIZ / "features" / "age_structured_features.parquet")
    ref_te = pd.read_parquet(RAIZ / "features" / "age_structured_features_test.parquet")

    train = pd.read_csv(RAIZ / "data" / "raw" / "age_train.csv", sep=";", encoding="utf-8-sig")
    train["text"] = train["text"].fillna("").astype(str).str.strip()
    nombres = list(structured_mod.extraer_estructuradas(train.head(1)).columns)

    print(f"columnas reconstruidas: {len(nombres)} | referencia: {ref.shape[1]}")
    print("primeras diferencias de nombre:", [a for a, b in zip(nombres, ref.columns) if a != b][:6])
    print("orden identico:", nombres == list(ref.columns))
    print()
    diferencias = []
    for posicion, nombre in enumerate(nombres):
        dif = np.abs(estr[:, posicion].astype(np.float64) - ref.iloc[:, posicion].to_numpy(np.float64))
        diferencias.append((float(dif.max()), float(dif.mean()), nombre))
    diferencias.sort(reverse=True)
    print("top 12 columnas por max|d|:")
    for maximo, media, nombre in diferencias[:12]:
        print(f"  {nombre:32} max|d|={maximo:.3e} media|d|={media:.3e}")
    afectadas = [d for d in diferencias if d[0] > 1e-9]
    print(f"\ncolumnas con max|d| > 1e-9: {len(afectadas)} de {len(nombres)}")
    print("test: max|d| global =", float(np.abs(estr_te - ref_te.to_numpy(np.float64)).max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
