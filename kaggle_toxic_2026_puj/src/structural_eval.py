"""Evaluacion OOF de V03 mas un bloque de features estructurales.

Centraliza el pipeline para que cada experimento de la familia STRUCTURAL_TEXT_FEATURES sea un
script corto y legible, sin duplicar el codigo de evaluacion.

Reglas que este modulo hace cumplir
-----------------------------------
- El control reproduce V03 exactamente. Si no, se para.
- El vectorizador TF-IDF se ajusta dentro de cada pliegue, como en V03.
- Las features estructurales son deterministas del texto: se calculan una vez, sin leakage.
- Los umbrales son los congelados de V03 (0.44 / 0.51 / 0.46).
- Los pliegues son los canonicos de V03.

Uso
---
    from src.structural_eval import evaluar_bloque, comparar, train_f1
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import metricas_multietiqueta
from src.structural_features import columnas, extraer, interacciones

ETIQUETAS = ("HS", "TR", "AG")
UMBRALES_V03 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
CORRECTED_CV_V03 = 0.7772114525165721
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
LR_PARAMS = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000,
             "solver": "liblinear", "random_state": 42}


def matriz_v03(
    textos_train: pd.Series, textos_val: pd.Series
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Representacion de V03: word (1,2) + char_wb (3,5), ajustada dentro del pliegue.

    Args:
        textos_train: textos del pliegue de entrenamiento.
        textos_val: textos del pliegue de validacion.

    Returns:
        Tupla con las matrices de entrenamiento y validacion.
    """
    word = TfidfVectorizer(**PARAMS_WORD)
    char = TfidfVectorizer(**PARAMS_CHAR)
    train = sparse.hstack(
        [word.fit_transform(textos_train), char.fit_transform(textos_train)]
    ).tocsr()
    val = sparse.hstack([word.transform(textos_val), char.transform(textos_val)]).tocsr()
    return train, val


def _matriz_bloque(
    textos: list[str], bloque: str, con_interacciones: bool
) -> pd.DataFrame:
    """Features estructurales de un bloque, con o sin interacciones.

    Args:
        textos: textos a procesar.
        bloque: uno de ``url``, ``mencion`` o ``enfasis``.
        con_interacciones: si se anaden las interacciones con la longitud.

    Returns:
        DataFrame con las columnas en orden estable.
    """
    base = pd.DataFrame(extraer(bloque, textos))
    base = base[columnas(bloque)]
    if not con_interacciones:
        return base
    tokens = np.array([len(t.split()) for t in textos], dtype=float)
    filas = [interacciones(bloque, fila, n) for fila, n in zip(base.to_dict("records"), tokens, strict=True)]
    extra = pd.DataFrame(filas)
    return pd.concat([base.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)


def evaluar_bloque(
    train_dev: pd.DataFrame,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    bloque: str | None,
    con_interacciones: bool = False,
) -> dict[str, Any]:
    """Evalua V03 solo o V03 mas un bloque estructural.

    Args:
        train_dev: DataFrame del conjunto de desarrollo.
        pliegues: pliegues canonicos de V03.
        bloque: bloque a anadir, o None para el control.
        con_interacciones: si se anaden interacciones con la longitud.

    Returns:
        Diccionario con probabilidades, predicciones, metricas por pliegue y train F1.
    """
    y_true = train_dev[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    n = len(train_dev)
    proba = np.zeros((n, 3), dtype=float)
    features = None
    if bloque is not None:
        features = _matriz_bloque(train_dev["text"].astype(str).tolist(), bloque, con_interacciones)
    metricas_pliegues = []
    train_f1s = []
    for numero, (idx_tr, idx_va) in enumerate(pliegues, start=1):
        train, val = matriz_v03(
            train_dev["text"].iloc[idx_tr], train_dev["text"].iloc[idx_va]
        )
        if features is not None:
            train = sparse.hstack([train, features.iloc[idx_tr].to_numpy()]).tocsr()
            val = sparse.hstack([val, features.iloc[idx_va].to_numpy()]).tocsr()
        pred_train = np.zeros((len(idx_tr), 3), dtype=np.int64)
        for j, etiqueta in enumerate(ETIQUETAS):
            modelo = LogisticRegression(**LR_PARAMS)
            modelo.fit(train, train_dev[etiqueta].iloc[idx_tr])
            proba[idx_va, j] = modelo.predict_proba(val)[:, 1]
            pred_train[:, j] = (modelo.predict_proba(train)[:, 1] >= UMBRALES_V03[etiqueta]).astype(int)
        pred_pliegue = np.column_stack(
            [(proba[idx_va, j] >= UMBRALES_V03[et]).astype(int) for j, et in enumerate(ETIQUETAS)]
        )
        metrica = metricas_multietiqueta(y_true[idx_va], pred_pliegue)
        metrica["pliegue"] = numero
        metricas_pliegues.append(metrica)
        train_f1s.append(metricas_multietiqueta(y_true[idx_tr], pred_train)["f1_media_columnas"])
    pred = np.column_stack(
        [(proba[:, j] >= UMBRALES_V03[et]).astype(int) for j, et in enumerate(ETIQUETAS)]
    )
    return {
        "proba": proba,
        "pred": pred,
        "metricas_pliegues": metricas_pliegues,
        "y_true": y_true,
        "train_f1": float(np.mean(train_f1s)),
    }


def verificar_control(cv_control: float) -> None:
    """Para el script si el control no reproduce V03.

    Args:
        cv_control: corrected_cv del control.

    Raises:
        SystemExit: si el control no reproduce la referencia.
    """
    if abs(cv_control - CORRECTED_CV_V03) > 1e-9:
        raise SystemExit(
            f"PARADA: el control reproduce {cv_control:.10f} y no "
            f"{CORRECTED_CV_V03:.10f}. No se puede comparar contra una base que no se sostiene."
        )


def comparar(y: np.ndarray, pred_base: np.ndarray, pred_nueva: np.ndarray) -> dict[str, Any]:
    """Rescates, danos y net entre dos predicciones.

    Args:
        y: etiquetas verdaderas.
        pred_base: predicciones de V03.
        pred_nueva: predicciones del challenger.

    Returns:
        Diccionario con el desglose.
    """
    cambios = pred_base != pred_nueva
    por_label: dict[str, Any] = {}
    rescates = dam = 0
    for j, et in enumerate(ETIQUETAS):
        bien_base = pred_base[:, j] == y[:, j]
        bien_nueva = pred_nueva[:, j] == y[:, j]
        r = int(np.count_nonzero(~bien_base & bien_nueva))
        d = int(np.count_nonzero(bien_base & ~bien_nueva))
        rescates += r
        dam += d
        por_label[et] = {
            "rescates": r, "danios": d, "neto": r - d,
            "celdas_modificadas": int(cambios[:, j].sum()),
        }
    return {
        "rescates": rescates, "danios": dam, "neto": rescates - dam,
        "celdas_modificadas": int(cambios.sum()),
        "cobertura_celdas": float(cambios.sum() / cambios.size),
        "por_label": por_label,
    }


def resumen(resultado: dict[str, Any]) -> dict[str, Any]:
    """Metricas resumidas de una evaluacion.

    Args:
        resultado: salida de ``evaluar_bloque``.

    Returns:
        Diccionario con corrected_cv, F1 por etiqueta y train F1.
    """
    m = metricas_multietiqueta(resultado["y_true"], resultado["pred"])
    return {
        "corrected_cv": float(m["f1_media_columnas"]),
        "f1_por_etiqueta": {et: float(m["f1_por_etiqueta"][et]) for et in ETIQUETAS},
        "train_f1": resultado["train_f1"],
        "gap_train_oof": resultado["train_f1"] - float(m["f1_media_columnas"]),
        "metricas_por_pliegue": resultado["metricas_pliegues"],
    }
