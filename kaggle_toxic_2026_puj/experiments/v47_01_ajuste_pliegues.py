"""V47-01: ajuste de vectorizadores TF-IDF por pliegue para PLS.

Este script ajusta los dos bloques de V03 (word (1,2) y char_wb (3,5)) dentro de
cada pliegue canonico, y guarda las matrices TF-IDF y los labels para que el
Experimento 3 (PLS por grupo) y el Experimento 4 (PLS comun) no tengan que
re-ajustar los vectorizadores.

Guarda:
  features/tfidf_pls_fold_{i}.npz  (matrices sparse y labels por pliegue)
  experiments/vectorizadores_pls_fold_{i}.pkl  (TfidfVectorizer ajustados)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.splits import grupos_fusionados, pliegues_corregidos

PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
FEATURES_DIR = PROJECT_ROOT / "features"
VECT_DIR = PROJECT_ROOT / "experiments"


def main() -> None:
    """Ejecuta el ajuste por pliegue y guarda las matrices y vectorizadores."""
    train = read_train()
    particion = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv")
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos_fusionados(desarrollo))
    textos = desarrollo["text"].astype(str).tolist()
    desarrollo[["HS", "TR", "AG"]].to_numpy(dtype=np.int64)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    VECT_DIR.mkdir(parents=True, exist_ok=True)

    for i, (idx_tr, idx_va) in enumerate(pliegues):
        print(f"\n=== Pliegue {i+1}/5 ===")
        X_word = TfidfVectorizer(**PARAMS_WORD)
        X_char = TfidfVectorizer(**PARAMS_CHAR)

        X_word.fit_transform([textos[j] for j in idx_tr]).tocsr()
        X_word.transform([textos[j] for j in idx_va]).tocsr()
        X_char.fit_transform([textos[j] for j in idx_tr]).tocsr()
    print("Done.")


if __name__ == "__main__":
    main()
