"""Meta-modelo y matrices del meta (porte fiel de phase2/phase3/phase7/phase25 + v19).

Ordenes EXACTOS:
- MEMBER_FILES (14 miembros, 78 columnas) + densas (35) + estructuradas (87) + supervisadas (73) = 273.
- matriz 279 (train) = 273 + ADV07 (6).  ASCII puro. Sin efectos en import.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from cadena import comun, miembros_oof_a, miembros_oof_b

#: Miembros y transformacion aplicada al construir el meta (scripts/phase2_ponderacion.py:49-55).
MEMBER_FILES = [("e013", "raw"), ("e016", "raw"), ("e009", "log"), ("e011", "log"),
                ("e010", "log"), ("e014", "raw"), ("e015_ordinal", "raw"),
                ("e015_softmax", "log"), ("e017", "log"), ("e019b", "log"),
                ("e021a", "log"), ("e021b", "log"), ("e028a", "raw"), ("adv05", "raw")]

#: Columnas densas usadas en el meta (35 de 38): se excluyen las 3 de emoji.
DENSE_EXCLUIDAS = {"n_emoji", "emoji_diversity", "has_emoji"}


class EncodedMLP:
    """MLP con etiquetas codificadas a enteros (scripts/phase2_ponderacion.py:106-125)."""

    def __init__(self) -> None:
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, batch_size=512,
                                   learning_rate_init=1e-3, max_iter=40, early_stopping=True,
                                   n_iter_no_change=5, random_state=comun.SEED)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> EncodedMLP:
        self.encoder.fit(labels)
        self.model.fit(self.scaler.fit_transform(features), self.encoder.transform(labels))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.encoder.inverse_transform(self.model.predict(self.scaler.transform(features)))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        crudo = self.model.predict_proba(self.scaler.transform(features))
        completa = np.zeros((len(features), len(comun.CLASSES)))
        for posicion, codigo in enumerate(self.model.classes_):
            completa[:, int(codigo)] = crudo[:, posicion]
        return completa


def logreg() -> Pipeline:
    """LogReg del proyecto (scripts/v13_p1b_fusion_tokens.py:79-83)."""
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                random_state=comun.SEED))])


def construir_miembros_oof(textos: pd.Series, y: np.ndarray, fold_idx: np.ndarray,
                           dense: pd.DataFrame) -> dict[str, np.ndarray]:
    """Reconstruye los 14 OOF de miembros (formas y dtypes de los artefactos guardados)."""
    return {
        "e013": miembros_oof_a.oof_e013(textos, y, fold_idx),
        "e016": miembros_oof_a.oof_e016(textos, y, fold_idx),
        "e009": miembros_oof_a.oof_e009(textos, y, fold_idx),
        "e011": miembros_oof_a.oof_e011(textos, y, fold_idx),
        "e010": miembros_oof_a.oof_e010(textos, y, fold_idx),
        "e014": miembros_oof_a.oof_e014(textos, y, fold_idx),
        "e015_ordinal": miembros_oof_a.oof_e015_ordinal(textos, y, fold_idx),
        "e015_softmax": miembros_oof_a.oof_e015_softmax(textos, y, fold_idx),
        "e017": miembros_oof_b.oof_e017(textos, y, fold_idx),
        "e019b": miembros_oof_b.oof_e019b(textos, y, fold_idx, dense),
        "e021a": miembros_oof_b.oof_e021a(textos, y, fold_idx),
        "e021b": miembros_oof_b.oof_e021b(textos, y, fold_idx),
        "e028a": miembros_oof_b.oof_e028a(textos, y, fold_idx),
        "adv05": miembros_oof_b.oof_adv05(textos, y, fold_idx),
    }


def build_features_273(oofs: dict[str, np.ndarray], densas35: np.ndarray,
                       estructuradas: np.ndarray, supervisadas: np.ndarray) -> np.ndarray:
    """Concatenacion exacta del meta (scripts/phase2_ponderacion.py:61-81)."""
    partes = []
    for nombre, transform in MEMBER_FILES:
        matriz = oofs[nombre]
        if matriz.ndim == 1:
            matriz = matriz.reshape(-1, 1)
        partes.append(np.log(np.clip(matriz, 1e-12, None)) if transform == "log" else matriz)
    features = np.concatenate([np.concatenate(partes, axis=1),
                               densas35, estructuradas, supervisadas], axis=1)
    assert features.shape[1] == 273, features.shape
    return features


def matriz_279(meta273: np.ndarray, adv07: np.ndarray) -> np.ndarray:
    """Matriz 279 del campeon (scripts/v19_b1_boosting.py:97-108)."""
    matriz = np.concatenate([meta273.astype(np.float32), adv07.astype(np.float32)], axis=1)
    assert matriz.shape[1] == 279, matriz.shape
    return matriz


def meta_v12_oof(meta273: np.ndarray, y: np.ndarray, fold_idx: np.ndarray) -> np.ndarray:
    """OOF del combinador V12: 0.5*(LogReg + EncodedMLP) sobre 273 (scripts/phase8:80-107)."""
    n = len(y)
    salida = np.zeros((n, len(comun.CLASSES)))
    for fold in range(5):
        tr = fold_idx != fold
        va = fold_idx == fold
        lr = logreg().fit(meta273[tr], y[tr])
        red = EncodedMLP().fit(meta273[tr], y[tr])
        p_lr = np.zeros((int(va.sum()), len(comun.CLASSES)))
        for posicion, etiqueta in enumerate(lr.classes_):
            p_lr[:, comun.CLASSES.index(str(etiqueta))] = lr.predict_proba(meta273[va])[:, posicion]
        p_red = red.predict_proba(meta273[va])
        salida[va] = 0.5 * (p_lr + p_red)
    return salida


def densas35(dense: pd.DataFrame) -> np.ndarray:
    """Las 35 columnas densas usadas en el meta (excluye las 3 de emoji)."""
    columnas = [c for c in dense.columns if c not in DENSE_EXCLUIDAS]
    return dense[columnas].to_numpy(float)
