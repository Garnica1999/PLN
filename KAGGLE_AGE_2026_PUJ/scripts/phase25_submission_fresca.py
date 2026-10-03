"""FASE 25 — Candidato marginal con la matriz de test **fresca** (corrige la caché desalineada).

Motivo
    La matriz cacheada ``meta_features_test_v11.npy`` (creada el 2026-09-24 15:23) es **anterior a la
    corrección de `phase3b`**, que arregló el desplazamiento de 6 columnas del bloque NB-SVM entre el
    lado train y el lado test del v11. Por eso la reconstrucción del v12 desde la caché daba 93,88% y
    no 100%: la caché no es utilizable para el candidato. El pipeline **fresco** (el que produjo la
    submission v12 enviada) sí es correcto y es el que se usa aquí.

Protocolo
    1. Construir el lado test fresco con el mismo código del pipeline (13 fuentes + NB-SVM de
       char+menciones + densas + estructuradas + supervisadas + NB-SVM de palabras = 279) y guardarlo
       como ``features/meta_features_test_v12_fresca.npy``.
    2. **Control de fidelidad**: reconstruir la submission v12 con esa matriz y compararla con el
       archivo enviado (debe ser ~100%).
    3. Candidato: añadir los 6 márgenes de la vista estructural (ajustada sobre todo el train) y
       generar ``age_submission_v13_candidato_estructura_NO_ADOPTADO.csv``.
    4. Validación formal y de distribución; nada se envía a Kaggle.

Salida: ``experiments/phase25_submission_fresca.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import data, submission  # noqa: E402
from src.config import set_seed  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from phase2_ponderacion import EncodedMLP, build_features  # noqa: E402
from phase21_vista_estructural import vista_estructura  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "phase25_submission_fresca.json"
MATRIZ_FRESCA = cfg.FEATURES / "meta_features_test_v12_fresca.npy"
ARCHIVO = cfg.SUBMISSIONS / "age_submission_v13_candidato_estructura_NO_ADOPTADO.csv"
CONFIG_VISTA = {"analyzer": "word", "ngram_range": (1, 3), "min_df": 3, "sublinear_tf": True,
                "lowercase": True}


def matriz_test_fresca(train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Construye el lado test del meta con el camino del pipeline (idéntico a la fase 8)."""
    from src import advanced_features as af, pipeline
    from advanced_06_submission_v10 import (CHAR_CONFIG, MENTION_CONFIG, nb_log_count_ratio,
                                            tokens_of)

    dense_columns = pipeline.DENSE_FEATURES
    dense_train = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")[dense_columns] \
        .to_numpy(float)
    dense_test = pd.read_parquet(cfg.FEATURES / "dense_test.parquet")[dense_columns].to_numpy(float)
    structured_test = pd.read_parquet(cfg.FEATURES / "age_structured_features_test.parquet") \
        .to_numpy(float)
    supervised_test = pd.concat([
        af.supervised_entity_features(train[cfg.TEXT_COL], y, test[cfg.TEXT_COL], "mention"),
        af.supervised_entity_features(train[cfg.TEXT_COL], y, test[cfg.TEXT_COL], "hashtag"),
        af.supervised_entity_features(train[cfg.TEXT_COL], y, test[cfg.TEXT_COL], "rt_source"),
        af.lexical_rarity_features(train[cfg.TEXT_COL], test[cfg.TEXT_COL]),
        af.class_lexicon_scores(train[cfg.TEXT_COL], y, test[cfg.TEXT_COL]),
    ], axis=1).to_numpy(float)

    char = TfidfVectorizer(**CHAR_CONFIG)
    mencion = TfidfVectorizer(**MENTION_CONFIG)
    x_train = sp.hstack([char.fit_transform(train[cfg.TEXT_COL]),
                         mencion.fit_transform(tokens_of(train[cfg.TEXT_COL], r"@\w+"))]).tocsr()
    x_test = sp.hstack([char.transform(test[cfg.TEXT_COL]),
                        mencion.transform(tokens_of(test[cfg.TEXT_COL], r"@\w+"))]).tocsr()
    nb_test = np.zeros((len(test), len(cfg.CLASSES)), dtype=np.float32)
    for posicion, etiqueta in enumerate(cfg.CLASSES):
        binario = (y == etiqueta).astype(int)
        ratio = nb_log_count_ratio(x_train, binario)
        modelo = LinearSVC(C=0.2, max_iter=3000)
        modelo.fit(x_train.multiply(ratio).tocsr(), binario)
        nb_test[:, posicion] = modelo.decision_function(x_test.multiply(ratio).tocsr())
    print(f"  NB-SVM char+menciones listo", flush=True)

    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                           max_features=300_000, lowercase=True)
    x_train_w = word.fit_transform(train[cfg.TEXT_COL])
    x_test_w = word.transform(test[cfg.TEXT_COL])
    nb_word_test = np.zeros((len(test), len(cfg.CLASSES)), dtype=np.float32)
    for posicion, etiqueta in enumerate(cfg.CLASSES):
        binario = (y == etiqueta).astype(int)
        ratio = nb_log_count_ratio(x_train_w, binario)
        modelo = LinearSVC(C=0.2, max_iter=3000)
        modelo.fit(x_train_w.multiply(ratio).tocsr(), binario)
        nb_word_test[:, posicion] = modelo.decision_function(x_test_w.multiply(ratio).tocsr())
    print(f"  NB-SVM de palabras listo", flush=True)

    representaciones = pipeline.build_representations(train, test)
    test_scores, _ = pipeline.fit_base_models(representaciones, y, dense_train, dense_test,
                                              verbose=False)
    matriz = np.concatenate(test_scores + [nb_test, dense_test, structured_test, supervised_test,
                                           nb_word_test], axis=1)
    assert matriz.shape[1] == 279, f"lado test inesperado: {matriz.shape}"
    return matriz


def main() -> int:
    """Genera el candidato marginal usando la matriz fresca y valida el control del v12."""
    set_seed()
    started = time.time()
    payload: dict = {"fase": "candidato marginal (matriz fresca)", "fecha": "2026-09-25",
                     "adoptado": False,
                     "motivo": "Δ CV +0,0018 (< +0,003 pre-registrado); holdout +0,0031",
                     "nota": "corrige la matriz cacheada desalineada (previa a phase3b)"}

    train = data.load_train()
    test = data.load_test()
    muestra = data.load_sample_submission()
    y = train[cfg.TARGET].astype(str).to_numpy()
    features, _, _, _, _ = build_features()
    features = np.concatenate([features, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    assert features.shape[1] == 279

    if MATRIZ_FRESCA.exists():
        base_test = np.load(MATRIZ_FRESCA)
        print(f"[test] matriz fresca cargada de caché {base_test.shape}", flush=True)
    else:
        print("[test] construyendo el lado test fresco (pipeline completo)...", flush=True)
        base_test = matriz_test_fresca(train, test, y)
        np.save(MATRIZ_FRESCA, base_test)
        print(f"[test] matriz fresca guardada {base_test.shape}", flush=True)

    # 1) Control de fidelidad contra la submission v12 enviada (fresca y caché en el mismo proceso).
    logreg = Pipeline([("scaler", StandardScaler()),
                       ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                  random_state=cfg.SEED))])
    logreg.fit(features, y)
    mlp = EncodedMLP()
    mlp.fit(features, y)
    enviada = pd.read_csv(cfg.SUBMISSIONS / "age_submission_v12.csv")
    etiquetas_enviadas = enviada[cfg.TARGET].astype(str).to_numpy()

    def predicciones_con(matriz: np.ndarray) -> np.ndarray:
        probabilidades = 0.5 * (logreg.predict_proba(matriz)
                                + mlp.model.predict_proba(mlp.scaler.transform(matriz)))
        return np.array(cfg.CLASSES)[np.argmax(probabilidades, axis=1)]

    predicciones_v12 = predicciones_con(base_test)
    acuerdo = float((predicciones_v12 == etiquetas_enviadas).mean())
    cache_test_273 = np.load(cfg.FEATURES / "meta_features_test_v11.npy")
    cache_279 = np.concatenate([cache_test_273, base_test[:, 273:279]], axis=1)
    predicciones_cache = predicciones_con(cache_279)
    acuerdo_cache = float((predicciones_cache == etiquetas_enviadas).mean())
    acuerdo_fresca_vs_cache = float((predicciones_v12 == predicciones_cache).mean())
    payload["control_v12"] = {
        "acuerdo_fresca_con_submission_v12": round(acuerdo, 6),
        "acuerdo_cache_con_submission_v12": round(acuerdo_cache, 6),
        "acuerdo_fresca_vs_cache": round(acuerdo_fresca_vs_cache, 6),
        "diferencia_maxima_matrices": round(
            float(np.abs(base_test - cache_279).max()), 8),
        "columna_mayor_diferencia": int(np.argmax(np.abs(base_test - cache_279).max(axis=0))),
    }
    print(f"[control] fresca vs v12: {100 * acuerdo:.4f}% | caché vs v12: "
          f"{100 * acuerdo_cache:.4f}% | fresca vs caché: {100 * acuerdo_fresca_vs_cache:.4f}%",
          flush=True)
    print(f"[control] matriz: max|Δ| {payload['control_v12']['diferencia_maxima_matrices']} en la "
          f"columna {payload['control_v12']['columna_mayor_diferencia']}", flush=True)

    # 2) Candidato con la vista estructural.
    vectorizador = TfidfVectorizer(**CONFIG_VISTA)
    x_estructura_train = vectorizador.fit_transform(vista_estructura(train[cfg.TEXT_COL]))
    x_estructura_test = vectorizador.transform(vista_estructura(test[cfg.TEXT_COL]))
    modelo_vista = LinearSVC(C=0.2, max_iter=3000)
    modelo_vista.fit(x_estructura_train, y)
    margenes_train = modelo_vista.decision_function(x_estructura_train)
    margenes_test = modelo_vista.decision_function(x_estructura_test)

    x_train = np.concatenate([features, margenes_train], axis=1)
    x_test = np.concatenate([base_test, margenes_test], axis=1)
    logreg_cand = Pipeline([("scaler", StandardScaler()),
                            ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                       random_state=cfg.SEED))])
    logreg_cand.fit(x_train, y)
    mlp_cand = EncodedMLP()
    mlp_cand.fit(x_train, y)
    probabilidades = 0.5 * (logreg_cand.predict_proba(x_test)
                            + mlp_cand.model.predict_proba(mlp_cand.scaler.transform(x_test)))
    predicciones = np.array(cfg.CLASSES)[np.argmax(probabilidades, axis=1)]

    candidato = pd.DataFrame({cfg.ID_COL: test[cfg.ID_COL].to_numpy(), cfg.TARGET: predicciones})
    checks = submission.validate_submission(candidato, test[cfg.ID_COL], muestra)
    candidato.to_csv(ARCHIVO, index=False)
    conteos = candidato[cfg.TARGET].value_counts().reindex(cfg.CLASSES).fillna(0).astype(int)
    payload["candidato"] = {
        "filas": len(candidato), "checks": checks, "distribucion": conteos.to_dict(),
        "min_clase": int(conteos.min()), "max_clase": int(conteos.max()),
        "acuerdo_con_v12_pct": round(100 * float((predicciones == predicciones_v12).mean()), 4),
        "predicciones_distintas": int((predicciones != predicciones_v12).sum()),
        "archivo": str(ARCHIVO),
    }
    print(f"[candidato] {conteos.to_dict()}")
    print(f"[candidato] cambia {int((predicciones != predicciones_v12).sum()):,} filas "
          f"({100 * float((predicciones != predicciones_v12).mean()):.2f}%) respecto a la v12",
          flush=True)
    payload["segundos"] = round(time.time() - started, 1)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
    print(f"\narchivo (NO adoptado, sin enviar): {ARCHIVO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
