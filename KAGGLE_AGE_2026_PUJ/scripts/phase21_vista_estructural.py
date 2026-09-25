"""FASE 21 — Vistas nuevas con puerta de diversidad de error (brief §13).

Observación
    Todas las representaciones actuales comparten el **contenido** del texto (char n-grams, palabras,
    menciones, NB-SVM). La fase 6 midió que el estilo aislado es casi azar (0,2793), pero eso se probó
    enmascarando el contenido, no **estructuralizando** el tweet. Dos vistas sin contenido no se han
    evaluado nunca: (a) la secuencia de **tipos** de token (palabra, mención, hashtag, URL, número,
    puntuación, mayúsculas), que describe la forma del tweet; (b) los **extremos** del tweet (primeras
    y últimas palabras), donde viven las fórmulas de apertura y cierre.

Hipótesis
    H56 Una vista estructural sin contenido aporta **diversidad de error** (acierta donde el ensamble
        falla con más frecuencia de la que justifica su exactitud individual).
    H57 Los extremos del tweet (apertura/cierre) contienen registro social no capturado por el char
        n-gram completo.

Protocolo (puerta de diversidad del brief §13)
    1. OOF de cada vista con las mismas particiones agrupadas y un LinearSVC.
    2. **Primero** la diversidad: tasa de rescate sobre los errores del v12 y acuerdo con él. Puerta:
       rescate >= 15% de los errores del v12 (las familias que rescatan en la fase B están en 14-20%).
    3. Solo si pasa la puerta se prueba su valor **en el ensemble**: se añaden sus 6 columnas al final
       del meta (mismo mecanismo que el NB-SVM de palabras) y se re-evalúa con el combinador de
       producción. Criterio de adopción: Δ ≥ +0,003 y ≥4/5 folds.

Uso::

    python scripts/phase21_vista_estructural.py

Salida: ``experiments/phase21_vista_estructural.json``.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, data  # noqa: E402
from src.config import set_seed  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from phase2_ponderacion import EncodedMLP, build_features  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "phase21_vista_estructural.json"
PUERTA_RESCATE = 0.15

TOKEN_RE = re.compile(r"@\w+|#\w+|https?://\S+|www\.\S+|\d+|[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+|[^\w\s]")
PALABRA_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")


def tipo_de_token(token: str) -> str:
    """Tipo estructural de un token (sin contenido)."""
    if token.startswith("@"):
        return "@"
    if token.startswith("#"):
        return "#"
    if token.startswith(("http", "www")):
        return "U"
    if token.isdigit():
        return "N"
    if PALABRA_RE.fullmatch(token):
        if token.isupper() and len(token) > 1:
            return "WU"
        if token[:1].isupper() and len(token) > 1:
            return "WC"
        return "W" if len(token) > 2 else "W1"
    return "P"


def vista_estructura(textos: pd.Series) -> list[str]:
    """Secuencia de tipos de token (forma del tweet, sin contenido)."""
    documentos = []
    for valor in textos.fillna("").astype(str):
        tipos = [tipo_de_token(token) for token in TOKEN_RE.findall(valor)]
        documentos.append(" ".join(tipos))
    return documentos


def vista_extremos(textos: pd.Series) -> list[str]:
    """Primeras y últimas palabras del tweet (apertura y cierre)."""
    documentos = []
    for valor in textos.fillna("").astype(str):
        palabras = PALABRA_RE.findall(valor)
        if not palabras:
            documentos.append("")
            continue
        extremos = palabras[:3] + palabras[-3:] if len(palabras) > 6 else palabras
        documentos.append(" ".join(extremos).lower())
    return documentos


def oof_de_vista(documentos: list[str], y: np.ndarray, folds: list[dict]) -> np.ndarray:
    """OOF de una vista con TF-IDF + LinearSVC (márgenes por clase)."""
    margenes = np.zeros((len(documentos), len(cfg.CLASSES)), dtype=np.float32)
    for fold in folds:
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        vectorizador = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=3,
                                       sublinear_tf=True, lowercase=True)
        x_train = vectorizador.fit_transform([documentos[posicion] for posicion in train_idx])
        x_valid = vectorizador.transform([documentos[posicion] for posicion in valid_idx])
        modelo = LinearSVC(C=0.2, max_iter=3000)
        modelo.fit(x_train, y[train_idx])
        margenes[valid_idx] = modelo.decision_function(x_valid)
        print(f"    vista OOF fold {fold['fold']} listo", flush=True)
    return margenes


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


def main() -> int:
    """Evalúa las vistas nuevas con la puerta de diversidad y su valor en el ensemble."""
    set_seed()
    started = time.time()
    payload: dict = {"fase": "§13 multivista — puerta de diversidad", "fecha": "2026-09-25"}

    train = data.load_train()
    test = data.load_test()
    y = train[cfg.TARGET].astype(str).to_numpy()
    features, _, _, _, _ = build_features()
    features = np.concatenate([features, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    folds = cv.make_folds(y, groups=clusters["cluster"].to_numpy()[:len(y)], n_splits=5,
                          n_repeats=1, scheme="group")
    referencia = json.loads((cfg.EXPERIMENTS / "phase12_auditoria_estado.json")
                            .read_text(encoding="utf-8"))["reproduccion_lofo_v12"]["resultado"]
    payload["baseline"] = {"macro_f1": referencia["macro_f1"], "std": referencia["std"],
                           "per_fold": referencia["per_fold"], "origen": "fase 12"}

    indice_real = np.array([cfg.CLASSES.index(valor) for valor in y])
    oof_v12 = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    aciertos_v12 = np.argmax(oof_v12, axis=1) == indice_real
    errores_v12 = int((~aciertos_v12).sum())

    vistas = {"estructura": vista_estructura(train[cfg.TEXT_COL]),
              "extremos": vista_extremos(train[cfg.TEXT_COL])}
    resultados = []
    for nombre, documentos in vistas.items():
        print(f"[{nombre}] OOF de la vista...", flush=True)
        margenes = oof_de_vista(documentos, y, folds)
        predicciones = np.argmax(margenes, axis=1)
        aciertos = predicciones == indice_real
        rescate = float((aciertos & ~aciertos_v12).mean())
        tasa_rescate = float((aciertos & ~aciertos_v12).sum() / max(1, errores_v12))
        acuerdo = float((predicciones == np.argmax(oof_v12, axis=1)).mean())
        resultado = {
            "vista": nombre, "features": "tfidf word(1,3)",
            "macro_f1_individual": round(float(f1_score(y, np.array(cfg.CLASSES)[predicciones],
                                                        average="macro", labels=cfg.CLASSES)), 4),
            "rescate_pct_de_errores_v12": round(100 * tasa_rescate, 2),
            "rescate_pct_de_filas": round(100 * rescate, 2),
            "acuerdo_con_v12_pct": round(100 * acuerdo, 2),
            "puerta_superada": bool(tasa_rescate >= PUERTA_RESCATE),
        }
        print(f"[{nombre}] F1 individual {resultado['macro_f1_individual']:.4f} | rescate de los "
              f"errores del v12 {resultado['rescate_pct_de_errores_v12']}% | acuerdo "
              f"{resultado['acuerdo_con_v12_pct']}% | puerta: {resultado['puerta_superada']}",
              flush=True)
        if resultado["puerta_superada"]:
            arreglo = np.concatenate([features, margenes], axis=1)
            veredicto = evaluar(arreglo, y, folds)
            positivos = sum(1 for nuevo, viejo in zip(veredicto["per_fold"],
                                                      referencia["per_fold"]) if nuevo > viejo)
            resultado.update({"n_features": int(arreglo.shape[1]), **veredicto,
                              "delta_vs_v12": round(veredicto["macro_f1"]
                                                    - referencia["macro_f1"], 4),
                              "folds_positivos": positivos,
                              "adoptado": bool(veredicto["macro_f1"] - referencia["macro_f1"] >= 0.003
                                               and positivos >= 4)})
        else:
            resultado.update({"adoptado": False,
                              "motivo_cierre": "no supera la puerta de diversidad (>= 15%)"})
        resultados.append(resultado)

    payload["vistas"] = resultados
    payload["puerta"] = f"rescate >= {PUERTA_RESCATE:.0%} de los errores del v12"
    payload["segundos"] = round(time.time() - started, 1)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
    del test
    print(f"\nresultado: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
