"""Spike 02 - Reproducibilidad del miembro E-013 (margenes LinearSVC sobre char_wb 3-6 + menciones).

Reconstruye los margenes OOF de E-013 con el MISMO procedimiento que
scripts/experimento_e013_barrido_c.py (mejor C=0.2) y los compara contra
features/oof_E013_margenes.npy. Folds: protocolo congelado del proyecto
(artifacts/p33_scores/todos/fold_idx.npy, identico al generado con sklearn >= 1.8).

NO escribe artefactos: solo imprime el informe. ASCII puro.

Uso:
    python repro/02_e013_miembro.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

RAIZ = Path(__file__).resolve().parents[1]
FEATURES = RAIZ / "features"
CARRIL = RAIZ / "artifacts" / "p33_scores" / "todos"
CLASSES = ["13-17", "18-24", "25-34", "35-49", "50-64", "65-xx"]

CHAR_CONFIG = {"analyzer": "char_wb", "ngram_range": (3, 6), "min_df": 5,
               "sublinear_tf": True, "max_features": 300_000, "lowercase": True}
MENTION_CONFIG = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2,
                  "sublinear_tf": True, "lowercase": True}
_SPACES_RE = re.compile(r"\s+")


def collapse_spaces(series: pd.Series) -> pd.Series:
    """Colapsa espacios repetidos y recorta extremos (src/preprocessing.py:98-100)."""
    return series.fillna("").astype(str).str.replace(_SPACES_RE, " ", regex=True).str.strip()


def mention_tokens(series: pd.Series) -> pd.Series:
    """Tokens de mencion (canal dedicado): @usuario en minusculas, unidos por espacio."""
    return collapse_spaces(series.str.findall(r"@\w+").apply(lambda valores: " ".join(valores).lower()))


def cargar_train() -> pd.DataFrame:
    """Lee age_train.csv como data.load_train(): separador ';', utf-8-sig, texto limpio."""
    marco = pd.read_csv(RAIZ / "data" / "raw" / "age_train.csv", sep=";", encoding="utf-8-sig")
    marco["text"] = marco["text"].fillna("").astype(str).str.strip()
    return marco


def main() -> int:
    """Reentrena E-013 (C=0.2) y compara contra la referencia guardada."""
    t0 = time.time()
    train = cargar_train()
    y = train["age_range"].astype(str).to_numpy()
    textos = train["text"]
    fold_idx = np.load(CARRIL / "fold_idx.npy")
    n = len(train)
    print(f"[spike02] train={n:,} | conteo folds: {np.bincount(fold_idx).tolist()}")

    margenes = np.zeros((n, 6), dtype=np.float32)
    puntajes = []
    for fold in range(5):
        t1 = time.time()
        tr = fold_idx != fold
        va = fold_idx == fold
        char = TfidfVectorizer(**CHAR_CONFIG)
        menciones = TfidfVectorizer(**MENTION_CONFIG)
        x_tr = hstack([char.fit_transform(textos.iloc[tr]),
                       menciones.fit_transform(mention_tokens(textos.iloc[tr]))]).tocsr()
        x_va = hstack([char.transform(textos.iloc[va]),
                       menciones.transform(mention_tokens(textos.iloc[va]))]).tocsr()
        modelo = LinearSVC(C=0.2, max_iter=3000)
        modelo.fit(x_tr, y[tr])
        decision = modelo.decision_function(x_va)
        orden = [list(modelo.classes_).index(etiqueta) for etiqueta in CLASSES]
        margenes[va] = decision[:, orden]
        macro = float(f1_score(y[va], modelo.predict(x_va), average="macro"))
        puntajes.append(macro)
        print(f"[spike02] fold {fold}: macro-F1={macro:.4f} | {time.time() - t1:.1f}s", flush=True)

    ref = np.load(FEATURES / "oof_E013_margenes.npy")
    dif = np.abs(margenes.astype(np.float64) - ref.astype(np.float64))
    print(f"[spike02] margenes: max|d|={dif.max():.3e} media|d|={dif.mean():.3e}")
    for j, etiqueta in enumerate(CLASSES):
        acuerdo = float((np.sign(margenes[:, j]) == np.sign(ref[:, j])).mean())
        print(f"[spike02] columna {etiqueta}: acuerdo de signo={acuerdo:.8f}")
    yi = np.array([CLASSES.index(v) for v in y])
    f1_recon = float(f1_score(yi, np.argmax(margenes, axis=1), average="macro"))
    f1_ref = float(f1_score(yi, np.argmax(ref, axis=1), average="macro"))
    print(f"[spike02] macro-F1 argmax reconstruido={f1_recon:.7f} | referencia={f1_ref:.7f}")

    barrido = RAIZ / "experiments" / "barrido_c_resultados.json"
    if barrido.is_file():
        datos = json.loads(barrido.read_text(encoding="utf-8"))
        folds_doc = [f["macro_f1"] for f in datos["resultados"]["0.2"]["per_fold"]]
        print(f"[spike02] folds documentados (C=0.2): {[round(v, 4) for v in folds_doc]}")
        print(f"[spike02] folds reconstruidos       : {[round(v, 4) for v in puntajes]}")

    print(f"[spike02] tiempo total {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
