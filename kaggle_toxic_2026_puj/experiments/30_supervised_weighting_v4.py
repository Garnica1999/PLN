"""TRACK C v4: weighting supervisado de features y cost-sensitive learning.

Contexto
--------
La spec v4 (§15-§19, §42-§43) pide reponderar la geometria de la matriz antes del
clasificador: log-odds suavizado por clase, chi-cuadrado, informacion mutua,
BM25, ponderacion binaria y barrido de `class_weight` por etiqueta. La revision
bibliografica (`reports/estado_arte_v4.md`, ideas priorizadas 3 y 4) recomienda una
**unica tabla comparativa** con todas las variantes y **una sola decision** al
final, cerrando la familia entera si ninguna gana.

Diseno
------
- Protocolo v2 (D-016): 5 pliegues de `StratifiedGroupKFold` con grupos
  near-duplicado UNION pseudo-autor sobre el desarrollo (n=1.483).
- Todos los pesos (log-odds, chi-cuadrado, MI, BM25, class weights) se calculan
  **dentro del pliegue de entrenamiento** (fold-local), nunca con validacion.
- Comparacion con los **mismos umbrales congelados de V03** (0.44/0.51/0.46) para
  aislar el efecto de la geometria, mas umbral 0.5 como control.
- Bootstrap pareado contra el OOF canonico de V03 (2.000 remuestreos).
- Las matrices de cada pliegue se cachean para que el barrido de class weights no
  re-vectorice 180 veces.

Uso
---
    /home/carlos/venv/bin/python experiments/30_supervised_weighting_v4.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

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
HARDNESS_PATH = PROJECT_ROOT / "features" / "hardness_v4.npz"
CSV_PATH = PROJECT_ROOT / "reports" / "supervised_weighting_v4.csv"
JSON_PATH = PROJECT_ROOT / "reports" / "supervised_weighting_v4.json"
NPZ_PATH = PROJECT_ROOT / "features" / "oof_weighting_v4.npz"
FIGURA_PATH = PROJECT_ROOT / "reports" / "figuras" / "fig_34_supervised_weighting_v4.png"

PARAMETROS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2}
PARAMETROS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2}
N_BOOTSTRAP = 2000
SEMILLA = 42
EPS_MIN = 1e-12
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
MULTIPLICADORES = (0.8, 1.0, 1.2, 1.5)
RONDAS_COORDENADAS = 3
REJILLA_BM25 = ((0.6, 0.3), (1.2, 0.0), (1.2, 0.75), (2.0, 0.3))


def log(mensaje: str) -> None:
    """Imprime una linea de progreso con marca de seccion.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V4-C] {mensaje}", flush=True)


# ---------------------------------------------------------------------------
# Esquemas de ponderacion de terminos
# ---------------------------------------------------------------------------
def esquema_tfidf(textos_entrenar: pd.Series, textos_aplicar: pd.Series, **tfidf_kwargs):
    """Construye la union TF-IDF de palabra y caracter con opciones de TF.

    Args:
        textos_entrenar: textos del pliegue de entrenamiento.
        textos_aplicar: textos a transformar.
        **tfidf_kwargs: opciones pasadas a `TfidfVectorizer` (`binary`, `sublinear_tf`).

    Returns:
        Par de matrices dispersas (entrenamiento, aplicacion).
    """
    palabra = TfidfVectorizer(**PARAMETROS_WORD, **tfidf_kwargs)
    caracter = TfidfVectorizer(**PARAMETROS_CHAR, **tfidf_kwargs)
    izquierda = sparse.hstack(
        [palabra.fit_transform(textos_entrenar), caracter.fit_transform(textos_entrenar)],
        format="csr",
    )
    derecha = sparse.hstack(
        [palabra.transform(textos_aplicar), caracter.transform(textos_aplicar)], format="csr"
    )
    return izquierda, derecha


def esquema_bm25(
    textos_entrenar: pd.Series, textos_aplicar: pd.Series, k1: float, b: float
):
    """Aplica la ponderacion BM25 sobre conteos de palabra y caracter.

    BM25 no esta en scikit-learn, por lo que se calcula sobre `CountVectorizer`:
    ``idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / dl_medio))``, donde
    ``tf`` es el conteo del termino y ``dl`` la longitud en terminos del
    documento. El IDF y la longitud media se calculan solo con el pliegue de
    entrenamiento.

    Args:
        textos_entrenar: textos del pliegue de entrenamiento.
        textos_aplicar: textos a transformar.
        k1: parametro de saturacion de frecuencia de termino.
        b: parametro de normalizacion por longitud de documento.

    Returns:
        Par de matrices dispersas normalizadas en L2.
    """
    izquierda = None
    derecha = None
    for parametros in (PARAMETROS_WORD, PARAMETROS_CHAR):
        conteo = CountVectorizer(**parametros)
        matriz_entrenar = conteo.fit_transform(textos_entrenar)
        matriz_aplicar = conteo.transform(textos_aplicar)
        largo_medio = float(np.asarray(matriz_entrenar.sum(axis=1)).mean()) or 1.0
        frecuencia = np.asarray((matriz_entrenar > 0).sum(axis=0)).ravel()
        n_documentos = matriz_entrenar.shape[0]
        idf = np.log(1.0 + (n_documentos - frecuencia + 0.5) / (frecuencia + 0.5))
        bloque_izq = _bm25(matriz_entrenar, largo_medio, k1, b, idf)
        bloque_der = _bm25(matriz_aplicar, largo_medio, k1, b, idf)
        izquierda = (
            bloque_izq
            if izquierda is None
            else sparse.hstack([izquierda, bloque_izq], format="csr")
        )
        derecha = (
            bloque_der
            if derecha is None
            else sparse.hstack([derecha, bloque_der], format="csr")
        )
    return izquierda, derecha


def _bm25(
    matriz: sparse.csr_matrix,
    largo_medio: float,
    k1: float,
    b: float,
    idf: np.ndarray,
) -> sparse.csr_matrix:
    """Aplica la formula BM25 elemento a elemento sobre una matriz dispersa.

    Args:
        matriz: matriz de conteos.
        largo_medio: longitud media de documento del pliegue de entrenamiento.
        k1: parametro de saturacion.
        b: parametro de normalizacion por longitud.
        idf: vector de frecuencias inversas de documento.

    Returns:
        Matriz dispersa con pesos BM25 normalizada en L2.
    """
    coo = matriz.tocoo()
    largo = np.asarray(matriz.sum(axis=1)).ravel()[coo.row]
    datos = coo.data
    denominador = datos + k1 * (1.0 - b + b * largo / largo_medio)
    salida = matriz.copy().astype(np.float64)
    salida.data = datos * (k1 + 1.0) / denominador * idf[coo.col]
    return normalize(salida, norm="l2")


def pesos_por_clase(
    matriz: sparse.csr_matrix, y: np.ndarray, esquema: str, alpha: float = 1.0
) -> np.ndarray:
    """Calcula un vector de peso por feature para una etiqueta binaria.

    Los tres esquemas usan solo la matriz y las etiquetas del pliegue de
    entrenamiento:

    - ``log_odds``: ``log(df_pos + alpha) - log(df_neg + alpha)``;
    - ``chi2``: estadistico chi-cuadrado de la clase frente al resto;
    - ``mi``: informacion mutua entre la presencia del termino y la etiqueta.

    El resultado se normaliza a [0, 1] para que su escala no altere la
    regularizacion del clasificador.

    Args:
        matriz: matriz de pesos de terminos del pliegue.
        y: vector de etiquetas binarias.
        esquema: nombre del esquema.
        alpha: suavizado de las frecuencias de documento.

    Returns:
        Vector de pesos no negativos por feature.

    Raises:
        ValueError: si el esquema no existe.
    """
    presencia = (matriz > 0).astype(np.float64)
    df_pos = np.asarray(presencia[y == 1].sum(axis=0)).ravel() + alpha
    df_neg = np.asarray(presencia[y == 0].sum(axis=0)).ravel() + alpha
    if esquema == "log_odds":
        pesos = np.log(df_pos) - np.log(df_neg)
    elif esquema == "chi2":
        n_pos = float((y == 1).sum())
        n_neg = float((y == 0).sum())
        total = n_pos + n_neg
        esperados = np.vstack(
            [(df_pos / total) * n_pos, (df_neg / total) * n_neg]
        )
        observados = np.vstack([df_pos - alpha, df_neg - alpha])
        with np.errstate(divide="ignore", invalid="ignore"):
            pesos = np.nansum((observados - esperados) ** 2 / esperados, axis=0)
    elif esquema == "mi":
        total = df_pos + df_neg
        p1, p0 = df_pos / total, df_neg / total
        pi = float((y == 1).mean())
        with np.errstate(divide="ignore", invalid="ignore"):
            pesos = np.where(
                (p1 > 0) & (p0 > 0),
                p1 * np.log(p1 / max(pi, EPS_MIN))
                + p0 * np.log(p0 / max(1.0 - pi, EPS_MIN)),
                0.0,
            )
    else:
        raise ValueError(f"Esquema de peso desconocido: {esquema}")
    pesos = np.nan_to_num(pesos, nan=0.0, posinf=0.0, neginf=0.0)
    pesos = pesos - pesos.min()
    if pesos.max() > 0:
        pesos = pesos / pesos.max()
    return pesos


def _pesos_de_clase(y: np.ndarray, multiplicador: float) -> dict[int, float]:
    """Construye los pesos de clase a partir de un multiplicador del balanceo.

    Args:
        y: vector de etiquetas binarias.
        multiplicador: factor aplicado al peso de la clase positiva.

    Returns:
        Diccionario ``{clase: peso}``.
    """
    n = len(y)
    n_pos = int((y == 1).sum())
    n_neg = n - n_pos
    return {
        0: (n / (2.0 * max(n_neg, 1))),
        1: (n / (2.0 * max(n_pos, 1))) * multiplicador,
    }


# ---------------------------------------------------------------------------
# Ajuste y evaluacion
# ---------------------------------------------------------------------------
def ajustar_predecir(
    matrices: tuple[sparse.csr_matrix, sparse.csr_matrix],
    y_entrenar: np.ndarray,
    multiplicadores: np.ndarray,
    pesos_por_etiqueta: list[np.ndarray] | None = None,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Ajusta un clasificador por etiqueta y devuelve probabilidades OOF.

    Args:
        matrices: par (entrenamiento, aplicacion) de matrices dispersas.
        y_entrenar: matriz de etiquetas del entrenamiento.
        multiplicadores: multiplicadores de peso de clase por etiqueta.
        pesos_por_etiqueta: vector de peso de feature por etiqueta (opcional).
        sample_weight: peso de ejemplo (opcional).

    Returns:
        Matriz de probabilidades por etiqueta.
    """
    izquierda, derecha = matrices
    salida = np.zeros((derecha.shape[0], len(ETIQUETAS)))
    for posicion in range(len(ETIQUETAS)):
        matriz_entrenar = izquierda
        matriz_aplicar = derecha
        if pesos_por_etiqueta is not None:
            vector = pesos_por_etiqueta[posicion]
            matriz_entrenar = izquierda.multiply(vector[None, :]).tocsr()
            matriz_aplicar = derecha.multiply(vector[None, :]).tocsr()
        modelo = LogisticRegression(
            C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=SEMILLA
        )
        if not np.isclose(multiplicadores[posicion], 1.0):
            modelo.set_params(
                class_weight=_pesos_de_clase(y_entrenar[:, posicion], multiplicadores[posicion])
            )
        if sample_weight is None:
            modelo.fit(matriz_entrenar, y_entrenar[:, posicion])
        else:
            modelo.fit(
                matriz_entrenar, y_entrenar[:, posicion], sample_weight=sample_weight
            )
        salida[:, posicion] = modelo.predict_proba(matriz_aplicar)[:, 1]
    return salida


def f1_por_ejemplo(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calcula la F1 binaria por ejemplo (promedio de sus tres etiquetas).

    Con una sola instancia por celda, la F1 de la clase positiva es 1 si acierta
    y 0 si no, de modo que la F1 por ejemplo es la fraccion de celdas correctas.

    Args:
        y_true: matriz verdadera.
        y_pred: matriz predicha.

    Returns:
        Vector con la F1 por ejemplo.
    """
    return (y_true == y_pred).mean(axis=1).astype(float)


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
    f1_a = f1_por_ejemplo(y_true, pred_a)
    f1_b = f1_por_ejemplo(y_true, pred_b)
    generador = np.random.default_rng(SEMILLA)
    n = len(f1_a)
    deltas = np.empty(n_bootstrap)
    for indice in range(n_bootstrap):
        indices = generador.integers(0, n, n)
        deltas[indice] = f1_b[indices].mean() - f1_a[indices].mean()
    percentiles = np.percentile(deltas, [2.5, 97.5])
    return float(f1_b.mean() - f1_a.mean()), float(percentiles[0]), float(percentiles[1])


def evaluar_variante(
    nombre: str,
    proba: np.ndarray,
    y_true: np.ndarray,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    pred_ref: np.ndarray,
) -> dict[str, Any]:
    """Evalua una variante con los umbrales de V03 y con 0.5.

    Args:
        nombre: identificador de la variante.
        proba: probabilidades OOF.
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
    mejorados = int(np.sum(np.array(f1_pliegues) > np.array(f1_ref) + 1e-12))
    delta, ic_bajo, ic_alto = _bootstrap(y_true, pred_ref, pred)
    return {
        "variante": nombre,
        "corrected_cv_umbrales_v03": round(metricas["f1_media_columnas"], 6),
        "corrected_cv_umbral_0_5": round(metricas_cinco["f1_media_columnas"], 6),
        "f1_HS": round(metricas["f1_por_etiqueta"]["HS"], 6),
        "f1_TR": round(metricas["f1_por_etiqueta"]["TR"], 6),
        "f1_AG": round(metricas["f1_por_etiqueta"]["AG"], 6),
        "micro_f1": round(metricas["f1_micro"], 6),
        "exact_match": round(metricas["f1_media_columnas"] * 0 + metricas["exact_match"], 6),
        "std_pliegues": round(float(np.std(f1_pliegues, ddof=1)), 6),
        "pliegues_mejorados": mejorados,
        "delta_f1_ejemplo": round(delta, 6),
        "ic95_bajo": round(ic_bajo, 6),
        "ic95_alto": round(ic_alto, 6),
        "f1_pliegues": ";".join(f"{valor:.4f}" for valor in f1_pliegues),
    }


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------
def main() -> None:
    """Ejecuta la tabla unica de weighting supervisado y el barrido de class weight."""
    inicio_tiempo = time.time()
    inicio = datetime.now(timezone.utc)
    log("Weighting supervisado y cost-sensitive (TRACK C)")

    oof = np.load(OOF_PATH, allow_pickle=True)
    ids_dev = oof["ids"].astype(np.int64)
    y_dev = oof["y_true"].astype(int)
    pred_ref = oof["pred"].astype(int)

    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos[indices_dev]
    assert np.array_equal(desarrollo["id"].to_numpy(dtype=np.int64), ids_dev)
    textos = desarrollo["text"]
    pliegues = pliegues_corregidos(desarrollo, grupos_dev)

    log("   cacheando matrices por pliegue (V03 sublinear y binario)")
    cache_v03 = [
        esquema_tfidf(textos.iloc[entrenar], textos.iloc[validar], sublinear_tf=True)
        for entrenar, validar in pliegues
    ]
    cache_binario = [
        esquema_tfidf(textos.iloc[entrenar], textos.iloc[validar], binary=True)
        for entrenar, validar in pliegues
    ]

    resultados: list[dict[str, Any]] = []
    oof_guardado: dict[str, np.ndarray] = {}

    def correr(nombre: str, matrices_por_pliegue, multiplicadores, pesos_fn=None, sample_weight=None):
        """Acumula el OOF de una variante y la registra.

        Args:
            nombre: identificador de la variante.
            matrices_por_pliegue: lista de pares de matrices por pliegue.
            multiplicadores: multiplicadores de peso de clase.
            pesos_fn: funcion opcional que devuelve los pesos por etiqueta.
            sample_weight: peso de ejemplo opcional.

        Returns:
            Fila de resultados.
        """
        proba_oof = np.zeros((len(desarrollo), len(ETIQUETAS)))
        for indice, (entrenar, validar) in enumerate(pliegues):
            matrices = matrices_por_pliegue[indice]
            pesos = None
            if pesos_fn is not None:
                pesos = pesos_fn(indice, matrices[0], y_dev[entrenar])
            proba_oof[validar] = ajustar_predecir(
                matrices,
                y_dev[entrenar],
                multiplicadores,
                pesos,
                None if sample_weight is None else sample_weight[entrenar],
            )
        oof_guardado[nombre] = proba_oof
        fila = evaluar_variante(nombre, proba_oof, y_dev, pliegues, pred_ref)
        resultados.append(fila)
        log(
            f"   {nombre:38s} corrected_cv={fila['corrected_cv_umbrales_v03']:.4f} "
            f"pliegues_mejorados={fila['pliegues_mejorados']}/5"
        )
        return fila

    # 1) Geometria de la representacion con la misma clase.
    correr("geo:v03_sublinear (control)", cache_v03, np.ones(len(ETIQUETAS)))
    correr("geo:binario", cache_binario, np.ones(len(ETIQUETAS)))

    # 2) BM25 con rejilla pequena de k1 x b.
    for k1, b in REJILLA_BM25:
        cache_bm25 = [
            esquema_bm25(textos.iloc[entrenar], textos.iloc[validar], k1, b)
            for entrenar, validar in pliegues
        ]
        correr(f"geo:bm25_k{k1}_b{b}", cache_bm25, np.ones(len(ETIQUETAS)))

    # 3) Weighting por etiqueta sobre la geometria de V03.
    for esquema in ("log_odds", "chi2", "mi"):
        correr(
            f"peso:{esquema}_por_etiqueta",
            cache_v03,
            np.ones(len(ETIQUETAS)),
            pesos_fn=lambda indice, matriz, y, esquema=esquema: [
                pesos_por_clase(matriz, y[:, posicion], esquema)
                for posicion in range(len(ETIQUETAS))
            ],
        )

    # 4) Barrido de class weight por etiqueta (ascenso coordenado sobre V03).
    log("   barrido de class weight por etiqueta (ascenso coordenado)")
    multiplicadores = np.ones(len(ETIQUETAS))
    mejor_config = multiplicadores.copy()
    mejor_probabilidad = None
    mejor_metrica = -np.inf
    for ronda in range(RONDAS_COORDENADAS):
        for posicion in range(len(ETIQUETAS)):
            for valor in MULTIPLICADORES:
                prueba = multiplicadores.copy()
                prueba[posicion] = valor
                proba_oof = np.zeros((len(desarrollo), len(ETIQUETAS)))
                for indice, (entrenar, validar) in enumerate(pliegues):
                    proba_oof[validar] = ajustar_predecir(
                        cache_v03[indice], y_dev[entrenar], prueba
                    )
                metrica = metricas_multietiqueta(
                    y_dev, (proba_oof >= UMBRALES_V03[None, :]).astype(int)
                )["f1_media_columnas"]
                if metrica > mejor_metrica:
                    mejor_metrica = metrica
                    mejor_config = prueba.copy()
                    mejor_probabilidad = proba_oof
        multiplicadores = mejor_config.copy()
        log(f"      ronda {ronda + 1}: mejor F1 = {mejor_metrica:.4f} con {np.round(mejor_config, 2)}")
    clave = "clase:barrio_por_etiqueta"
    oof_guardado[clave] = mejor_probabilidad
    fila = evaluar_variante(clave, mejor_probabilidad, y_dev, pliegues, pred_ref)
    fila["multiplicadores"] = ";".join(f"{valor:.2f}" for valor in mejor_config)
    resultados.append(fila)
    log(
        f"   {clave:38s} corrected_cv={fila['corrected_cv_umbrales_v03']:.4f} "
        f"multiplicadores={fila['multiplicadores']}"
    )

    # 5) V4-C05: sample weight (downweight de la violacion) x class weight.
    # El signo es el que funciono en V4-B03: bajar el peso de los ejemplos que
    # mas violan el umbral mejoro a LR y a SVC, mientras que subirlo las empeoro.
    dureza = np.load(HARDNESS_PATH, allow_pickle=True)
    assert np.array_equal(dureza["ids"].astype(np.int64), ids_dev)
    violacion = dureza["error_score"].astype(float)
    escala = np.percentile(violacion, 75) - np.percentile(violacion, 25) or 1.0
    z = (violacion - np.median(violacion)) / escala
    pesos_dureza = np.clip(1.0 - 0.5 * z, 0.5, 1.5)
    correr(
        "combo:dureza_x_clase",
        cache_v03,
        mejor_config,
        sample_weight=pesos_dureza,
    )

    tabla = pd.DataFrame(resultados).sort_values("corrected_cv_umbrales_v03", ascending=False)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(CSV_PATH, index=False)

    referencia = 0.777211
    mejor = tabla.iloc[0]
    cumple = bool(
        float(mejor["corrected_cv_umbrales_v03"]) - referencia >= 0.005
        and int(mejor["pliegues_mejorados"]) >= 3
    )
    resumen = {
        "experiment_id": "V4-C01..C05",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duracion_s": round(time.time() - inicio_tempo, 2),
        "referencia_v03": referencia,
        "n_variantes": int(len(tabla)),
        "mejor_variante": str(mejor["variante"]),
        "mejor_corrected_cv": float(mejor["corrected_cv_umbrales_v03"]),
        "mejor_pliegues_mejorados": int(mejor["pliegues_mejorados"]),
        "mejor_ic95": [float(mejor["ic95_bajo"]), float(mejor["ic95_alto"])],
        "cumple_criterio_D016": cumple,
        "n_variantes_sobre_referencia": int(
            (tabla["corrected_cv_umbrales_v03"] > referencia).sum()
        ),
        "decision_familia": (
            "ADOPTAR" if cumple else "NO_SIGNAL: la familia de weighting se cierra sin mejora"
        ),
        "detalle": resultados,
    }
    JSON_PATH.write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        NPZ_PATH,
        ids=ids_dev,
        y_true=y_dev,
        **{clave.replace(":", "__"): valor for clave, valor in oof_guardado.items()},
    )
    _escribir_figura(tabla)
    log(f"   mejor variante: {resumen['mejor_variante']} ({resumen['mejor_corrected_cv']:.4f})")
    log(f"   cumple D-016: {cumple} -> {resumen['decision_familia']}")


def _escribir_figura(tabla: pd.DataFrame) -> None:
    """Dibuja el corrected_cv de todas las variantes del TRACK C.

    Args:
        tabla: tabla de resultados.
    """
    figura, eje = plt.subplots(figsize=(11, 6))
    colores = [
        "#C44E52" if valor <= 0.777211 else "#4C72B0"
        for valor in tabla["corrected_cv_umbrales_v03"]
    ]
    eje.barh(tabla["variante"], tabla["corrected_cv_umbrales_v03"], color=colores)
    eje.axvline(0.777211, color="#55A868", linestyle="--", label="V03 = 0.7772")
    eje.set_xlabel("corrected_cv con umbrales de V03")
    eje.set_title("Weighting supervisado y cost-sensitive (v4, TRACK C)")
    eje.legend()
    figura.tight_layout()
    FIGURA_PATH.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(FIGURA_PATH, dpi=150)
    plt.close(figura)


if __name__ == "__main__":
    main()
