"""FASE 23 — Candidato marginal: vista estructural como miembro adicional (archivo, NO se envía).

Contexto
    La vista estructural (secuencia de tipos de token, sin contenido) es la única fuente de señal
    positiva de la búsqueda residual: **+0,0018 en CV** (4/5 folds, t pareado ≈ 2,7) y **+0,0031 en el
    holdout congelado** (0,6108 frente a 0,6077, con el baseline reproducido exacto). **No cumple el
    criterio pre-registrado de CV (Δ ≥ +0,003)**, así que no se adopta: este script deja el archivo
    listo para que Carlos decida si gasta un envío. No se sube nada a Kaggle.

Protocolo
    1. Control de fidelidad: reconstruir el modelo v12 con la matriz de test cacheada (273) + NB-SVM de
       palabras y compararlo con ``age_submission_v12.csv``.
    2. Candidato: se añaden los 6 márgenes de la vista estructural (ajustada sobre todo el train) como
       columnas finales del meta (mismo mecanismo que el NB-SVM de palabras) y se ajustan los dos
       combinadores sobre todo el train.
    3. Validación formal y de distribución; se guarda como
       ``submissions/age_submission_v13_candidato_estructura_NO_ADOPTADO.csv``.

Salida: ``experiments/phase23_submission_candidato.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
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

RESULTS_PATH = cfg.EXPERIMENTS / "phase23_submission_candidato.json"
ARCHIVO = cfg.SUBMISSIONS / "age_submission_v13_candidato_estructura_NO_ADOPTADO.csv"
CONFIG_VISTA = {"analyzer": "word", "ngram_range": (1, 3), "min_df": 3, "sublinear_tf": True,
                "lowercase": True}


def margenes_de_vista(entrenamiento: list[str], prueba: list[str],
                      y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ajusta la vista estructural sobre todo el train y devuelve (márgenes train, márgenes test)."""
    vectorizador = TfidfVectorizer(**CONFIG_VISTA)
    x_train = vectorizador.fit_transform(entrenamiento)
    x_test = vectorizador.transform(prueba)
    modelo = LinearSVC(C=0.2, max_iter=3000)
    modelo.fit(x_train, y)
    return modelo.decision_function(x_train), modelo.decision_function(x_test)


def bloques_nb_word(test: pd.DataFrame, train: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Reconstruye el bloque NB-SVM de palabras del lado test (6 columnas del modelo v12)."""
    from scripts.advanced_06_submission_v10 import nb_log_count_ratio
    from sklearn.svm import LinearSVC as SVM

    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                           max_features=300_000, lowercase=True)
    x_train = word.fit_transform(train[cfg.TEXT_COL])
    x_test = word.transform(test[cfg.TEXT_COL])
    margenes = np.zeros((len(test), len(cfg.CLASSES)), dtype=np.float32)
    for posicion, etiqueta in enumerate(cfg.CLASSES):
        binario = (y == etiqueta).astype(int)
        ratio = nb_log_count_ratio(x_train, binario)
        modelo = SVM(C=0.2, max_iter=3000)
        modelo.fit(x_train.multiply(ratio).tocsr(), binario)
        margenes[:, posicion] = modelo.decision_function(x_test.multiply(ratio).tocsr())
    return margenes


def main() -> int:
    """Genera el candidato marginal (archivo local, sin envío)."""
    set_seed()
    started = time.time()
    payload: dict = {"fase": "candidato marginal — vista estructural", "fecha": "2026-09-25",
                     "adoptado": False,
                     "motivo": "Δ CV +0,0018 (< +0,003 pre-registrado); holdout +0,0031"}

    train = data.load_train()
    test = data.load_test()
    muestra = data.load_sample_submission()
    y = train[cfg.TARGET].astype(str).to_numpy()
    features, _, _, _, _ = build_features()
    features = np.concatenate([features, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    assert features.shape[1] == 279

    cache_test = np.load(cfg.FEATURES / "meta_features_test_v11.npy")
    nb_word_test = bloques_nb_word(test, train, y)
    base_test = np.concatenate([cache_test, nb_word_test], axis=1)
    print(f"[test] matriz base {base_test.shape}", flush=True)

    # 1) Control de fidelidad: reconstruir el v12 y compararlo con la submission enviada.
    logreg = Pipeline([("scaler", StandardScaler()),
                       ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                  random_state=cfg.SEED))])
    logreg.fit(features, y)
    mlp = EncodedMLP()
    mlp.fit(features, y)
    probabilidades_v12 = 0.5 * (logreg.predict_proba(base_test)
                                + mlp.model.predict_proba(mlp.scaler.transform(base_test)))
    predicciones_v12 = np.array(cfg.CLASSES)[np.argmax(probabilidades_v12, axis=1)]
    enviada = pd.read_csv(cfg.SUBMISSIONS / "age_submission_v12.csv")
    acuerdo = float((predicciones_v12 == enviada[cfg.TARGET].astype(str).to_numpy()).mean())
    payload["control_v12"] = {"acuerdo_con_submission_v12": round(acuerdo, 6)}
    print(f"[control] acuerdo con age_submission_v12.csv: {100 * acuerdo:.4f}%", flush=True)

    # 2) Candidato con la vista estructural.
    margenes_train, margenes_test = margenes_de_vista(
        vista_estructura(train[cfg.TEXT_COL]), vista_estructura(test[cfg.TEXT_COL]), y)
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
    payload["control_candidato"] = {
        "filas": len(candidato), "checks": checks,
        "distribucion": conteos.to_dict(),
        "min_clase": int(conteos.min()), "max_clase": int(conteos.max()),
        "acuerdo_con_v12_pct": round(100 * float((predicciones == predicciones_v12).mean()), 4),
        "predicciones_distintas": int((predicciones != predicciones_v12).sum()),
        "archivo": str(ARCHIVO),
    }
    print(f"[candidato] {conteos.to_dict()}")
    print(f"[candidato] cambia {int((predicciones != predicciones_v12).sum()):,} filas "
          f"({100 * float((predicciones != predicciones_v12).mean()):.2f}%) respecto a la v12",
          flush=True)
    print(f"[candidato] checks: {checks}", flush=True)
    payload["segundos"] = round(time.time() - started, 1)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
    print(f"\narchivo (NO adoptado, sin enviar): {ARCHIVO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
