"""TRACK G v4: verosimilitud class-conditional y modelos estadisticos alternativos.

Contexto
--------
Dos lineas que la spec v4 (§59-§63) declara nunca probadas y que el proyecto no
ha ejecutado en ninguna fase:

1. **V4-G01**: modelos de lenguaje muy simples ajustados desde cero en el
   propio corpus (unigramas, bigramas y n-gramas de caracter) para estimar
   ``P(texto | clase)``. La revision bibliografica (`reports/estado_arte_v4.md`,
   tema 7) **no encontro** evidencia reciente transferible, asi que la hipotesis es
   propia: la verosimilitud condicional es una forma de geometries que el TF-IDF
   no expresa (dependencia de la longitud y del orden), y puede ser
   complementaria en late fusion.
2. **V4-G02 y V4-G03**: Ridge, SGD y PassiveAggressive sobre la misma union
   TF-IDF, y modelos no lineales sobre una compresion SVD. El barrido historico
   nunca los probo (solo LinearSVC, ComplementNB y un MLP).

Diseno
------
- Protocolo v2 (D-016), 5 pliegues, grupos fusionados, todo ajustado dentro del
  pliegue de entrenamiento.
- La comparacion de la fusion tardia se hace contra un meta de la **misma**
  arquitectura sin las features de verosimilitud, para aislar su aporte.
- Umbrales congelados de V03 (0.44/0.51/0.46) y bootstrap pareado.

Uso
---
    /home/carlos/venv/bin/python experiments/33_class_conditional_lm_v4.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data import read_train  # noqa: E402
from src.evaluation import metricas_multietiqueta  # noqa: E402
from src.splits import (  # noqa: E402
    ETIQUETAS,
    grupos_fusionados,
    holdout_v2,
    pliegues_corregidos,
)

OOF_PATH = PROJECT_ROOT / "features" / "oof_v03_final.npz"
CSV_PATH = PROJECT_ROOT / "reports" / "class_conditional_lm_v4.csv"
JSON_PATH = PROJECT_ROOT / "reports" / "class_conditional_lm_v4.json"
FIGURA_PATH = PROJECT_ROOT / "reports" / "figuras" / "fig_37_class_conditional_lm_v4.png"

PARAMETROS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMETROS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
ALPHA = 0.5
N_SVD = 300
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
N_BOOTSTRAP = 2000
SEMILLA = 42
REFERENCIA_V03 = 0.777211
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso con marca de seccion.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V4-G] {mensaje}", flush=True)


# ---------------------------------------------------------------------------
# Modelos de lenguaje ajustados desde cero
# ---------------------------------------------------------------------------
def unidades(texto: str) -> list[str]:
    """Tokeniza un texto enMinusculas sin acentos añadidos.

    Args:
        texto: texto del tweet.

    Returns:
        Lista de tokens.
    """
    return TOKEN_PATTERN.findall(texto.lower())


def ngramas_caracter(texto: str, rango: tuple[int, int] = (3, 4)) -> list[str]:
    """Genera n-gramas de caracter sobre el texto normalizado.

    Args:
        texto: texto del tweet.
        rango: intervalo de longitudes (inclusivo).

    Returns:
        Lista de n-gramas.
    """
    base = f" {TOKEN_PATTERN.sub(' ', texto.lower())} "
    salida = []
    for longitud in range(rango[0], rango[1] + 1):
        salida.extend(base[i : i + longitud] for i in range(max(len(base) - longitud + 1, 0)))
    return salida


def ajustar_modelos_de_lenguaje(textos: list[str], y: np.ndarray) -> list[dict[str, Counter]]:
    """Ajusta un modelo de lenguaje por etiqueta y devuelve sus contadores.

    Args:
        textos: textos del pliegue de entrenamiento.
        y: matriz binaria de etiquetas.

    Returns:
        Lista con un diccionario por etiqueta que contiene los contadores de
        unigramas, bigramas y n-gramas de caracter.
    """
    modelos = []
    for posicion in range(len(ETIQUETAS)):
        seleccion = y[:, posicion] == 1
        uni: Counter = Counter()
        bi: Counter = Counter()
        car: Counter = Counter()
        for texto in [t for t, s in zip(textos, seleccion) if s]:
            tokens = unidades(texto)
            uni.update(tokens)
            bi.update(zip(tokens, tokens[1:]))
            car.update(ngramas_caracter(texto))
        modelos.append(
            {
                "uni": uni,
                "bi": bi,
                "car": car,
                "n_uni": max(sum(uni.values()), 1),
                "n_bi": max(sum(bi.values()), 1),
                "n_car": max(sum(car.values()), 1),
                "vocab": max(len(uni), 1),
            }
        )
    return modelos


def verosimilitud(
    modelos: list[dict[str, Counter]], textos: list[str], background: Counter
) -> np.ndarray:
    """Calcula la verosimilitud logaritmica media por token para cada clase.

    Args:
        modelos: contadores por etiqueta.
        textos: textos a evaluar.
        background: frecuencias marginales (para el suavizado de unseen).

    Returns:
        Matriz de forma ``(n_textos, n_etiquetas, 3)`` con las verosimilitudes
        de unigramas, bigramas y n-gramas de caracter.
    """
    n_background = max(sum(background.values()), 1)
    vocab = max(len(background), 1)
    salida = np.zeros((len(textos), len(ETIQUETAS), 3))
    for indice, texto in enumerate(textos):
        tokens = unidades(texto)
        bigramas = list(zip(tokens, tokens[1:]))
        caracteres = ngramas_caracter(texto)
        for posicion, modelo in enumerate(modelos):
            for modelo_indice, (contador, eventos, lista) in enumerate(
                (
                    (modelo["uni"], modelo["n_uni"], tokens),
                    (modelo["bi"], modelo["n_bi"], bigramas),
                    (modelo["car"], modelo["n_car"], caracteres),
                )
            ):
                if not lista:
                    salida[indice, posicion, modelo_indice] = 0.0
                    continue
                total = 0.0
                for elemento in lista:
                    conteo = contador.get(elemento, 0)
                    # Los bigramas no tienen equivalente en el contador marginal;
                    # para ellos el prior de suavizado es uniforme.
                    prior = (
                        background.get(elemento, 0) / n_background
                        if not isinstance(elemento, tuple)
                        else 1.0 / n_background
                    )
                    total += np.log(
                        (conteo + ALPHA * max(prior, 1.0 / n_background))
                        / (eventos + ALPHA * vocab)
                    )
                salida[indice, posicion, modelo_indice] = total / len(lista)
    return salida


def features_likelihood(verosimilitudes: np.ndarray) -> np.ndarray:
    """Construye las 15 features de la linea de verosimilitud.

    Por cada uno de los tres modelos (unigramas, bigramas, caracter) se usan las
    tres verosimilitudes, el margen de la clase frente a la media de las otras dos
    y la perplejidad relativa.

    Args:
        verosimilitudes: tensor de forma ``(n, 3, 3)``.

    Returns:
        Matriz de 15 columnas.
    """
    columnas = []
    for modelo_indice in range(3):
        por_clase = verosimilitudes[:, :, modelo_indice]
        columnas.extend([por_clase[:, j] for j in range(len(ETIQUETAS))])
        for j in range(len(ETIQUETAS)):
            otras = [k for k in range(len(ETIQUETAS)) if k != j]
            columnas.append(por_clase[:, j] - por_clase[:, otras].mean(axis=1))
        columnas.extend([np.exp(-por_clase[:, j]) for j in range(len(ETIQUETAS))])
    return np.column_stack(columnas)


# ---------------------------------------------------------------------------
# Evaluacion
# ---------------------------------------------------------------------------
def _bootstrap(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_bootstrap: int = N_BOOTSTRAP
) -> tuple[float, float, float]:
    """Bootstrap pareado del delta de F1 media por ejemplo.

    Args:
        y_true: matriz verdadera.
        pred_a: predicciones de referencia.
        pred_b: predicciones del candidato.
        n_bootstrap: numero de remuestreos.

    Returns:
        Tupla `(delta, ic95_bajo, ic95_alto)`.
    """
    f1_a = (y_true == pred_a).mean(axis=1).astype(float)
    f1_b = (y_true == pred_b).mean(axis=1).astype(float)
    generador = np.random.default_rng(SEMILLA)
    n = len(f1_a)
    deltas = np.empty(n_bootstrap)
    for indice in range(n_bootstrap):
        indices = generador.integers(0, n, n)
        deltas[indice] = f1_b[indices].mean() - f1_a[indices].mean()
    percentiles = np.percentile(deltas, [2.5, 97.5])
    return float(f1_b.mean() - f1_a.mean()), float(percentiles[0]), float(percentiles[1])


def evaluar(
    nombre: str,
    proba: np.ndarray,
    y_true: np.ndarray,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    pred_ref: np.ndarray,
) -> dict[str, Any]:
    """Evalua una variante con los umbrales de V03 y con 0.5.

    Args:
        nombre: identificador de la variante.
        proba: probabilidades por etiqueta.
        y_true: matriz verdadera.
        pliegues: lista de pliegues corregidos.
        pred_ref: predicciones de referencia.

    Returns:
        Fila de resultados.
    """
    pred = (proba >= UMBRALES_V03[None, :]).astype(int)
    metricas = metricas_multietiqueta(y_true, pred)
    metricas_cinco = metricas_multietiqueta(y_true, (proba >= 0.5).astype(int))
    f1_pliegues = [
        metricas_multietiqueta(y_true[validar], pred[validar])["f1_media_columnas"]
        for _, validar in pliegues
    ]
    f1_ref = [
        metricas_multietiqueta(y_true[validar], pred_ref[validar])["f1_media_columnas"]
        for _, validar in pliegues
    ]
    delta, ic_bajo, ic_alto = _bootstrap(y_true, pred_ref, pred)
    return {
        "variante": nombre,
        "corrected_cv_umbrales_v03": round(metricas["f1_media_columnas"], 6),
        "corrected_cv_umbral_0_5": round(metricas_cinco["f1_media_columnas"], 6),
        "f1_HS": round(metricas["f1_por_etiqueta"]["HS"], 6),
        "f1_TR": round(metricas["f1_por_etiqueta"]["TR"], 6),
        "f1_AG": round(metricas["f1_por_etiqueta"]["AG"], 6),
        "pliegues_mejorados": int(np.sum(np.array(f1_pliegues) > np.array(f1_ref) + 1e-12)),
        "std_pliegues": round(float(np.std(f1_pliegues, ddof=1)), 6),
        "delta_f1_ejemplo": round(delta, 6),
        "ic95_bajo": round(ic_bajo, 6),
        "ic95_alto": round(ic_alto, 6),
        "f1_pliegues": ";".join(f"{valor:.4f}" for valor in f1_pliegues),
    }


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------
def main() -> None:
    """Ejecuta la linea de verosimilitud class-conditional y los clasificadores alternativos."""
    inicio_tiempo = time.time()
    inicio = datetime.now(timezone.utc)
    log("Verosimilitud class-conditional y modelos alternativos (TRACK G)")

    oof = np.load(OOF_PATH, allow_pickle=True)
    ids_dev = oof["ids"].astype(np.int64)
    y_dev = oof["y_true"].astype(int)
    pred_ref = oof["pred"].astype(int)
    proba_v03 = oof["proba"]

    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos[indices_dev])
    textos = desarrollo["text"].tolist()

    resultados: list[dict[str, Any]] = []
    oof_guardado: dict[str, np.ndarray] = {}

    # Features de verosimilitud (fold-local) y matrices TF-IDF.
    # 3 modelos x (3 verosimilitudes + 3 margenes + 3 perplejidades) = 27 columnas
    features_lm = np.zeros((len(desarrollo), 27))
    margen_v03 = proba_v03 - 0.5
    entropia_v03 = -(proba_v03 * np.log(np.clip(proba_v03, 1e-9, 1))
                     + (1 - proba_v03) * np.log(np.clip(1 - proba_v03, 1e-9, 1)))
    base_meta = np.column_stack([proba_v03, margen_v03, entropia_v03])
    matrices: list[tuple[sparse.csr_matrix, sparse.csr_matrix]] = []

    log("   1/4 verosimilitud class-conditional y matrices por pliegue")
    for indice_fold, (entrenar, validar) in enumerate(pliegues):
        textos_entrenar = [textos[i] for i in entrenar]
        background: Counter = Counter()
        for texto in textos_entrenar:
            background.update(unidades(texto))
        modelos = ajustar_modelos_de_lenguaje(textos_entrenar, y_dev[entrenar])
        verosimilitudes = verosimilitud(
            modelos, [textos[i] for i in validar], background
        )
        features_lm[validar] = features_likelihood(verosimilitudes)

        palabra = TfidfVectorizer(**PARAMETROS_WORD)
        caracter = TfidfVectorizer(**PARAMETROS_CHAR)
        izquierda = sparse.hstack(
            [palabra.fit_transform(textos_entrenar), caracter.fit_transform(textos_entrenar)],
            format="csr",
        )
        derecha = sparse.hstack(
            [palabra.transform([textos[i] for i in validar]),
             caracter.transform([textos[i] for i in validar])],
            format="csr",
        )
        matrices.append((izquierda, derecha))
    log(f"      verosimilitud calculada para {len(desarrollo)} documentos")

    # 2) Modelo de verosimilidad por etiqueta (standalone).
    log("   2/4 modelo standalone sobre features de verosimilitud")
    proba_lm = np.zeros((len(desarrollo), len(ETIQUETAS)))
    for indice_fold, (entrenar, validar) in enumerate(pliegues):
        for posicion in range(len(ETIQUETAS)):
            modelo = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                                        solver="liblinear", random_state=SEMILLA)
            modelo.fit(features_lm[entrenar], y_dev[entrenar, posicion])
            proba_lm[validar, posicion] = modelo.predict_proba(features_lm[validar])[:, 1]
    oof_guardado["lm:standalone"] = proba_lm
    resultados.append(evaluar("lm:standalone", proba_lm, y_dev, pliegues, pred_ref))
    log(f"      lm:standalone corrected_cv={resultados[-1]['corrected_cv_umbrales_v03']:.4f}")

    # 3) Late fusion: meta con y sin las features de verosimilitud.
    log("   3/4 late fusion (meta de 9 features de V03 frente a meta + verosimilitud)")
    for nombre, matriz in (
        ("meta_v03_9feat", base_meta),
        ("meta_v03_lm_36feat", np.column_stack([base_meta, features_lm])),
    ):
        proba_meta = np.zeros((len(desarrollo), len(ETIQUETAS)))
        for indice_fold, (entrenar, validar) in enumerate(pliegues):
            meta = Ridge(alpha=3.0)
            meta.fit(matriz[entrenar], y_dev[entrenar])
            proba_meta[validar] = np.clip(meta.predict(matriz[validar]), 0.0, 1.0)
        oof_guardado[nombre] = proba_meta
        resultados.append(evaluar(nombre, proba_meta, y_dev, pliegues, pred_ref))
        log(f"      {nombre:24s} corrected_cv={resultados[-1]['corrected_cv_umbrales_v03']:.4f}")

    # 4) Clasificadores alternativos sobre la union y no lineales sobre SVD.
    log("   4/4 clasificadores alternativos (G02) y no lineales sobre SVD (G03)")
    clasificadores = {
        "ridge": lambda: RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=SEMILLA),
        "sgd": lambda: SGDClassifier(loss="modified_huber", alpha=1e-4, class_weight="balanced",
                                     max_iter=2000, random_state=SEMILLA),
        "passive_aggressive": lambda: PassiveAggressiveClassifier(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=SEMILLA
        ),
    }
    for nombre, constructor in clasificadores.items():
        proba = np.zeros((len(desarrollo), len(ETIQUETAS)))
        for indice_fold, (entrenar, validar) in enumerate(pliegues):
            izquierda, derecha = matrices[indice_fold]
            for posicion in range(len(ETIQUETAS)):
                modelo = constructor()
                modelo.fit(izquierda, y_dev[entrenar, posicion])
                decision = modelo.decision_function(derecha)
                proba[validar, posicion] = 1.0 / (1.0 + np.exp(-decision))
        clave = f"alt:{nombre}"
        oof_guardado[clave] = proba
        resultados.append(evaluar(clave, proba, y_dev, pliegues, pred_ref))
        log(f"      {clave:24s} corrected_cv={resultados[-1]['corrected_cv_umbrales_v03']:.4f}")

    for nombre in ("rbf_svc", "hist_gb", "extra_trees"):
        proba = np.zeros((len(desarrollo), len(ETIQUETAS)))
        for indice_fold, (entrenar, validar) in enumerate(pliegues):
            izquierda, derecha = matrices[indice_fold]
            svd = TruncatedSVD(n_components=N_SVD, random_state=SEMILLA)
            reducido_entrenar = svd.fit_transform(izquierda)
            reducido_validar = svd.transform(derecha)
            for posicion in range(len(ETIQUETAS)):
                if nombre == "rbf_svc":
                    modelo = SVC(C=2.0, class_weight="balanced", gamma="scale", random_state=SEMILLA)
                elif nombre == "hist_gb":
                    modelo = HistGradientBoostingClassifier(
                        max_iter=200, learning_rate=0.1, random_state=SEMILLA
                    )
                else:
                    modelo = ExtraTreesClassifier(
                        n_estimators=300, class_weight="balanced", random_state=SEMILLA, n_jobs=8
                    )
                modelo.fit(reducido_entrenar, y_dev[entrenar, posicion])
                if hasattr(modelo, "predict_proba"):
                    proba[validar, posicion] = modelo.predict_proba(reducido_validar)[:, 1]
                else:
                    # SVC con kernel RBF solo expone decision_function; se aplica
                    # la sigmoidal en lugar de `probability=True` (5 CV internas).
                    decision = modelo.decision_function(reducido_validar)
                    proba[validar, posicion] = 1.0 / (1.0 + np.exp(-decision))
        clave = f"svd:{nombre}"
        oof_guardado[clave] = proba
        resultados.append(evaluar(clave, proba, y_dev, pliegues, pred_ref))
        log(f"      {clave:24s} corrected_cv={resultados[-1]['corrected_cv_umbrales_v03']:.4f}")

    tabla = pd.DataFrame(resultados).sort_values("corrected_cv_umbrales_v03", ascending=False)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(CSV_PATH, index=False)

    mejor = tabla.iloc[0]
    aporte_lm = float(
        tabla.loc[tabla["variante"] == "meta_v03_lm_36feat", "corrected_cv_umbrales_v03"].iloc[0]
        - tabla.loc[tabla["variante"] == "meta_v03_9feat", "corrected_cv_umbrales_v03"].iloc[0]
    )
    cumple = bool(
        float(mejor["corrected_cv_umbrales_v03"]) - REFERENCIA_V03 >= 0.005
        and int(mejor["pliegues_mejorados"]) >= 3
    )
    resumen = {
        "experiment_id": "V4-G01-G03",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duracion_s": round(time.time() - inicio_tiempo, 2),
        "referencia_v03": REFERENCIA_V03,
        "n_variantes": int(len(tabla)),
        "mejor_variante": str(mejor["variante"]),
        "mejor_corrected_cv": float(mejor["corrected_cv_umbrales_v03"]),
        "aporte_lm_a_la_fusion": round(aporte_lm, 6),
        "cumple_criterio_D016": cumple,
        "decision_familia": (
            "ADOPTAR" if cumple else "NO_SIGNAL: ni la verosimilitud class-conditional ni los "
            "clasificadores alternativos superan a V03"
        ),
        "detalle": resultados,
    }
    JSON_PATH.write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        PROJECT_ROOT / "features" / "oof_tracks_g_v4.npz",
        ids=ids_dev,
        y_true=y_dev,
        **{clave.replace(":", "__"): valor for clave, valor in oof_guardado.items()},
    )
    _escribir_figura(tabla)
    log(f"   mejor variante: {resumen['mejor_variante']} ({resumen['mejor_corrected_cv']:.4f})")
    log(f"   aporte de la verosimilitud a la fusion: {aporte_lm:+.4f}")
    log(f"   decision: {resumen['decision_familia']}")


def _escribir_figura(tabla: pd.DataFrame) -> None:
    """Dibuja el corrected_cv de las variantes del TRACK G.

    Args:
        tabla: tabla de resultados.
    """
    figura, eje = plt.subplots(figsize=(10, 5))
    colores = [
        "#C44E52" if valor <= REFERENCIA_V03 else "#4C72B0"
        for valor in tabla["corrected_cv_umbrales_v03"]
    ]
    eje.barh(tabla["variante"], tabla["corrected_cv_umbrales_v03"], color=colores)
    eje.axvline(REFERENCIA_V03, color="#55A868", linestyle="--", label="V03 = 0.7772")
    eje.set_xlabel("corrected_cv con umbrales de V03")
    eje.set_title("Verosimilitud class-conditional y clasificadores alternativos (v4, TRACK G)")
    eje.legend()
    figura.tight_layout()
    FIGURA_PATH.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(FIGURA_PATH, dpi=150)
    plt.close(figura)


if __name__ == "__main__":
    main()
