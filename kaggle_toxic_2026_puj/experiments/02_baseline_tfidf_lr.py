"""Baseline B1 - TF-IDF de palabras + regresion logistica (multi-etiqueta).

Pregunta que responde
---------------------
¿Que F1 local se obtiene con la representacion clasica mas simple (TF-IDF de
palabras 1-2 + regresion logistica balanceada) antes de cualquier ingenieria de
features, y sirve como primera submission valida?

Metodo
------
- Representacion: TF-IDF de palabras (1,2), min_df=2, sublinear_tf, ajustado
  SIEMPRE con el pliegue de entrenamiento (nunca con validacion ni con test).
- Modelo: tres clasificadores binarios independientes (HS, TR, AG) con
  `LogisticRegression(class_weight="balanced")`.
- Validacion: `StratifiedGroupKFold(5)` con grupos = near-duplicados
  (`data/processed/grupos_near_duplicados.csv`, D-004) y estratificacion por la
  combinacion HS x TR x AG.
- Se evalua ademas el post-proceso jerarquico `HS=0 => TR=0, AG=0` (D-001)
  sobre las predicciones out-of-fold.
- Modelo final: reajuste con todo el train y prediccion del test para generar
  `submissions/toxic_submission_v01_baseline_tfidf_lr.csv` (sin enviar).

Metricas reportadas (para comparar con la sonda E00)
----------------------------------------------------
- `f1_media_columnas`: media de la F1 binaria (clase positiva) de las 3 etiquetas.
- `f1_macro6`: media de las F1 de las 6 clases (positiva y negativa por etiqueta).
- `f1_micro`: F1 global sobre todas las celdas.
- `f1_samples`: F1 promedio por fila.
- `exact_match`: fraccion de filas con las 3 etiquetas correctas.

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/02_baseline_tfidf_lr.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module

REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
GRUPOS_PATH = PROJECT_ROOT / "data" / "processed" / "grupos_near_duplicados.csv"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"

LABELS = ("HS", "TR", "AG")
RANDOM_STATE = 42
N_SPLITS = 5
SUBMISSION_NAME = "toxic_submission_v01_baseline_tfidf_lr.csv"

VECTORIZER_PARAMS = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
    "lowercase": True,
    "strip_accents": None,
}
MODEL_PARAMS = {
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "liblinear",
    "random_state": RANDOM_STATE,
}


def log(message: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        message: texto a imprimir.
    """
    print(f"[B1] {message}", flush=True)


def cargar_grupos() -> pd.Series:
    """Carga la asignacion de grupo de near-duplicado por id.

    Returns:
        Serie indexada por ``id`` con el grupo de near-duplicado.

    Raises:
        FileNotFoundError: si la auditoria de Fase 0 no se ha ejecutado.
    """
    if not GRUPOS_PATH.exists():
        raise FileNotFoundError(
            f"Falta {GRUPOS_PATH}. Ejecuta experiments/00_auditoria_datos.py primero."
        )
    grupos = pd.read_csv(GRUPOS_PATH)
    return grupos.set_index("id")["grupo"]


def metricas_multietiqueta(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula las cuatro definiciones candidatas de F1 y el exact match.

    Args:
        y_true: matriz binaria verdadera de forma ``(n, 3)``.
        y_pred: matriz binaria predicha de forma ``(n, 3)``.

    Returns:
        Diccionario con las metricas globales y por etiqueta.
    """
    f1_positivas = [
        f1_score(y_true[:, indice], y_pred[:, indice], zero_division=0)
        for indice in range(len(LABELS))
    ]
    f1_negativas = [
        f1_score(1 - y_true[:, indice], 1 - y_pred[:, indice], zero_division=0)
        for indice in range(len(LABELS))
    ]
    return {
        "f1_media_columnas": float(np.mean(f1_positivas)),
        "f1_macro6": float(np.mean(f1_positivas + f1_negativas)),
        "f1_micro": float(
            f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0)
        ),
        "f1_samples": float(
            f1_score(y_true, y_pred, average="samples", zero_division=0)
        ),
        "exact_match": float((y_true == y_pred).all(axis=1).mean()),
        "f1_por_etiqueta": {
            etiqueta: float(valor)
            for etiqueta, valor in zip(LABELS, f1_positivas, strict=True)
        },
    }


def aplicar_postproceso_jerarquico(y_pred: np.ndarray) -> np.ndarray:
    """Fuerza TR=0 y AG=0 cuando HS=0 (estructura observada en train).

    Args:
        y_pred: matriz binaria predicha de forma ``(n, 3)``.

    Returns:
        Copia de la matriz con el post-proceso aplicado.
    """
    ajustada = y_pred.copy()
    sin_hs = ajustada[:, 0] == 0
    ajustada[sin_hs, 1] = 0
    ajustada[sin_hs, 2] = 0
    return ajustada


def validar_cruzada(
    train: pd.DataFrame, grupos: pd.Series
) -> tuple[dict, dict, dict]:
    """Ejecuta la validacion cruzada agrupada del baseline.

    Args:
        train: DataFrame de entrenamiento con etiquetas.
        grupos: serie de grupos de near-duplicado indexada por id.

    Returns:
        Tupla ``(resumen, resumen_con_postproceso, oof)`` donde ``oof`` contiene
        las predicciones out-of-fold y las probabilidades por etiqueta.
    """
    combinacion = (
        train["HS"].astype(str) + train["TR"].astype(str) + train["AG"].astype(str)
    )
    asignacion_grupos = (
        train["id"].astype(int).map(grupos).fillna(-1).astype(int).to_numpy()
    )
    if (asignacion_grupos == -1).any():
        faltantes = int((asignacion_grupos == -1).sum())
        raise ValueError(
            f"{faltantes} ids de train sin grupo de near-duplicado; "
            "regenera data/processed/grupos_near_duplicados.csv"
        )
    divisor = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    y_true = train[list(LABELS)].to_numpy()
    oof_pred = np.zeros_like(y_true)
    oof_proba = np.zeros_like(y_true, dtype=float)
    metricas_pliegues = []
    metricas_pliegues_post = []

    for numero, (indices_entrenamiento, indices_validacion) in enumerate(
        divisor.split(train, combinacion, groups=asignacion_grupos), start=1
    ):
        vectorizador = TfidfVectorizer(**VECTORIZER_PARAMS)
        matriz_entrenamiento = vectorizador.fit_transform(
            train["text"].iloc[indices_entrenamiento]
        )
        matriz_validacion = vectorizador.transform(train["text"].iloc[indices_validacion])
        for indice_etiqueta, etiqueta in enumerate(LABELS):
            modelo = LogisticRegression(**MODEL_PARAMS)
            modelo.fit(
                matriz_entrenamiento,
                train[etiqueta].iloc[indices_entrenamiento],
            )
            oof_proba[indices_validacion, indice_etiqueta] = modelo.predict_proba(
                matriz_validacion
            )[:, 1]
        oof_pred[indices_validacion] = (oof_proba[indices_validacion] >= 0.5).astype(int)
        verdad_pliegue = y_true[indices_validacion]
        metrica_pliegue = metricas_multietiqueta(
            verdad_pliegue, oof_pred[indices_validacion]
        )
        metrica_post = metricas_multietiqueta(
            verdad_pliegue, aplicar_postproceso_jerarquico(oof_pred[indices_validacion])
        )
        metrica_pliegue["pliegue"] = numero
        metrica_post["pliegue"] = numero
        metricas_pliegues.append(metrica_pliegue)
        metricas_pliegues_post.append(metrica_post)
        log(
            f"pliegue {numero}: F1 columnas {metrica_pliegue['f1_media_columnas']:.4f} "
            f"(con post-proceso {metrica_post['f1_media_columnas']:.4f})"
        )

    resumen = metricas_multietiqueta(y_true, oof_pred)
    resumen_post = metricas_multietiqueta(y_true, aplicar_postproceso_jerarquico(oof_pred))
    for nombre, lista in (("pliegues", metricas_pliegues), ("pliegues_post", metricas_pliegues_post)):
        valores = [fila["f1_media_columnas"] for fila in lista]
        destino = resumen if nombre == "pliegues" else resumen_post
        destino["f1_media_columnas_std_pliegues"] = float(np.std(valores))
    oof = {
        "prediccion": oof_pred,
        "probabilidad": oof_proba,
        "pliegues": metricas_pliegues,
        "pliegues_post": metricas_pliegues_post,
    }
    return resumen, resumen_post, oof


def ajustar_modelo_final(
    train: pd.DataFrame,
) -> tuple[TfidfVectorizer, list[LogisticRegression]]:
    """Ajusta el vectorizador y los tres modelos con todo el train.

    Args:
        train: DataFrame de entrenamiento con etiquetas.

    Returns:
        Tupla ``(vectorizador, modelos)`` ajustados.
    """
    vectorizador = TfidfVectorizer(**VECTORIZER_PARAMS)
    matriz = vectorizador.fit_transform(train["text"])
    modelos = []
    for etiqueta in LABELS:
        modelo = LogisticRegression(**MODEL_PARAMS)
        modelo.fit(matriz, train[etiqueta])
        modelos.append(modelo)
    return vectorizador, modelos


def escribir_submission(
    test: pd.DataFrame,
    vectorizador: TfidfVectorizer,
    modelos: list[LogisticRegression],
) -> Path:
    """Genera y valida la submission del baseline.

    Args:
        test: DataFrame de test con ``id`` y ``text``.
        vectorizador: vectorizador ajustado con train.
        modelos: modelos binarios ajustados con train.

    Returns:
        Ruta del archivo de submission escrito.

    Raises:
        ValueError: si la submission no cumple el formato oficial.
    """
    matriz = vectorizador.transform(test["text"])
    predicciones = np.column_stack(
        [(modelo.predict_proba(matriz)[:, 1] >= 0.5).astype(int) for modelo in modelos]
    )
    predicciones = aplicar_postproceso_jerarquico(predicciones)
    plantilla = data_module.read_submission()
    if list(plantilla["id"]) != list(test["id"]):
        raise ValueError("El orden de ids del test no coincide con la plantilla oficial")
    salida = plantilla.copy()
    for indice, etiqueta in enumerate(LABELS):
        salida[etiqueta] = predicciones[:, indice]
    if salida[list(LABELS)].to_numpy().min() < 0 or salida[list(LABELS)].to_numpy().max() > 1:
        raise ValueError("Hay valores fuera de {0, 1} en la submission")
    if salida.isna().any().any() or not salida["id"].is_unique:
        raise ValueError("Hay nulos o ids duplicados en la submission")
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    destino = SUBMISSIONS_DIR / SUBMISSION_NAME
    salida.to_csv(destino, index=False)
    return destino


def registrar_experimento(resumen: dict, resumen_post: dict, duracion: float, dimension: int) -> None:
    """Anade la fila del baseline a ``experiments.csv``.

    Args:
        resumen: metricas out-of-fold sin post-proceso.
        resumen_post: metricas out-of-fold con post-proceso jerarquico.
        duracion: segundos de entrenamiento y evaluacion.
        dimension: numero de features del vectorizador final.
    """
    fila = {
        "experiment_id": "B1",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Que F1 local da la representacion clasica mas simple?",
        "hypothesis": "TF-IDF de palabras + regresion logistica balanceada supera la plantilla todo-1",
        "preprocessing": "texto crudo, lowercase, sin stopwords",
        "feature_groups": "WORD_NGRAM",
        "representation": "TF-IDF word (1,2), min_df=2, sublinear_tf",
        "model": "LogisticRegression(class_weight=balanced, C=1.0, liblinear) x3",
        "hyperparameters": json.dumps(MODEL_PARAMS, ensure_ascii=False),
        "validation": "StratifiedGroupKFold(5) grupos=near-duplicados, estratificado por combo HSxTRxAG",
        "f1": round(resumen["f1_media_columnas"], 4),
        "macro_f1": round(resumen["f1_macro6"], 4),
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": round(duracion, 2),
        "dimension": dimension,
        "result": json.dumps(
            {
                "oof_sin_postproceso": {
                    clave: resumen[clave]
                    for clave in ("f1_media_columnas", "f1_macro6", "f1_micro", "f1_samples", "exact_match")
                },
                "oof_con_postproceso": {
                    clave: resumen_post[clave]
                    for clave in ("f1_media_columnas", "f1_macro6", "f1_micro", "f1_samples", "exact_match")
                },
                "f1_por_etiqueta_post": resumen_post["f1_por_etiqueta"],
            },
            ensure_ascii=False,
        ),
        "interpretation": "Baseline de referencia; el post-proceso jerarquico se evalua en OOF",
        "decision": "Pendiente de comparacion con B2-B5 y con la sonda E00",
        "date": "2026-09-22",
        "script": "experiments/02_baseline_tfidf_lr.py",
        "notes": "Submission v01 generada sin enviar; umbral fijo 0.5; la columna f1 guarda el valor SIN post-proceso (con post-proceso: 0.7578); weighted_f1 vacio a proposito (no es una metrica del proyecto; el F1 samples esta en result)",
    }
    existe = EXPERIMENTS_CSV.exists()
    with EXPERIMENTS_CSV.open("a", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(fila))
        if not existe:
            escritor.writeheader()
        escritor.writerow(fila)


def main() -> None:
    """Ejecuta el baseline completo y genera la submission."""
    inicio = time.perf_counter()
    train = data_module.read_train()
    test = data_module.read_test()
    grupos = cargar_grupos()
    log(f"train {train.shape} | test {test.shape} | grupos {grupos.nunique()}")

    resumen, resumen_post, oof = validar_cruzada(train, grupos)
    log(f"OOF sin post-proceso : {json.dumps(resumen, ensure_ascii=False)}")
    log(f"OOF con post-proceso : {json.dumps(resumen_post, ensure_ascii=False)}")

    vectorizador, modelos = ajustar_modelo_final(train)
    destino = escribir_submission(test, vectorizador, modelos)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"vectorizador": vectorizador, "modelos": modelos, "etiquetas": list(LABELS)},
        MODELS_DIR / "b1_tfidf_lr.joblib",
    )
    duracion = time.perf_counter() - inicio
    dimension = len(vectorizador.get_feature_names_out())

    metricas = {
        "experimento": "B1",
        "fecha": "2026-09-22",
        "oof_sin_postproceso": resumen,
        "oof_con_postproceso": resumen_post,
        "duracion_segundos": round(duracion, 2),
        "dimension_tfidf": dimension,
        "submission": str(destino.relative_to(PROJECT_ROOT)),
        "modelo": "models/b1_tfidf_lr.joblib",
        "pliegues": oof["pliegues"],
        "pliegues_post": oof["pliegues_post"],
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "baseline_b1_metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    registrar_experimento(resumen, resumen_post, duracion, dimension)

    log(f"submission escrita: {destino} ({len(pd.read_csv(destino))} filas)")
    log(f"dimension TF-IDF: {dimension} | duracion: {duracion:.2f} s")


if __name__ == "__main__":
    main()
