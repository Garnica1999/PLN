"""TRACK H v4: el shift de calibracion de las entradas del meta.

Contexto
--------
La fase v2 dejo una hipotesis abierta y nunca medida
(`reports/advanced_research_v2.md` §4.3): el meta-modelo se entrena con scores
out-of-fold de bases ajustadas sobre 4/5 del desarrollo, pero en la prediccion
del test las bases estan ajustadas sobre el 100 % del train. Si esa diferencia de
tamano de entrenamiento **desplaza la distribucion de las probabilidades**, el meta
recibe entradas fuera del rango con el que fue entrenado, y por eso la mejora
local (+0.0102 dev, +0.0180 holdout) no transfirio al leaderboard (0.78163 frente
a 0.78306 de V03).

Este script mide ese efecto **dentro del desarrollo**, sin usar el holdout ni el
test, y comprueba si alinear las distribuciones recupera la diferencia.

Diseno (reproduce el mecanismo exacto dentro del desarrollo)
-------------------------------------------------------------
Para cada pliegue externo ``j`` con entrenamiento ``T`` y validacion ``V``:

- **regimen meta** (como se entrena el meta): las features del meta salen de un
  CV interno de 4 pliegues sobre ``T``, de modo que cada base ve 3/4 de ``T``;
- **regimen inferencia** (como se predice en el despliegue): las bases se ajustan
  sobre el 100 % de ``T`` y se aplican a ``V``.

Se comparan cuatro situaciones con la misma arquitectura de meta (Ridge alpha=3
sobre 36 features = 4 bases x 3 etiquetas x [probabilidad, margen, entropia]):

1. ``meta_control``: features de validacion tambien en regimen meta (techo del
   meta; equivalente al corrected_cv del meta v2);
2. ``meta_shift``: features de validacion en regimen inferencia, sin alinear
   (cuanto se pierde por el shift);
3. ``meta_shift_cuantil``: features de inferencia alineadas por cuantiles a la
   distribucion de entrenamiento del meta;
4. ``meta_shift_platt``: transformacion de Platt por columna, ajustada con las
   features del regimen meta.

La diferencia (2) - (1) cuantifica el dano del shift; (3) - (2) y (4) - (2)
cuanto recupera la alineacion.

Uso
---
    /home/carlos/venv/bin/python experiments/32_meta_shift_v4.py
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
from scipy.stats import ks_2samp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data import read_train  # noqa: E402
from src.evaluation import metricas_multietiqueta  # noqa: E402
from src.splits import (  # noqa: E402
    ETIQUETAS,
    combinaciones,
    grupos_fusionados,
    holdout_v2,
    pliegues_corregidos,
)

OOF_PATH = PROJECT_ROOT / "features" / "oof_v03_final.npz"
CSV_PATH = PROJECT_ROOT / "reports" / "meta_shift_v4.csv"
JSON_PATH = PROJECT_ROOT / "reports" / "meta_shift_v4.json"
NPZ_PATH = PROJECT_ROOT / "features" / "oof_meta_shift_v4.npz"
FIGURA_PATH = PROJECT_ROOT / "reports" / "figuras" / "fig_36_meta_shift_v4.png"

PARAMETROS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMETROS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
N_SUBPLIEGUES = 4
ALPHA_META = 3.0
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
N_BOOTSTRAP = 2000
SEMILLA = 42
EPS = 1e-9
REFERENCIA_V03 = 0.777211
REFERENCIA_META_V2 = 0.7874


def log(mensaje: str) -> None:
    """Imprime una linea de progreso con marca de seccion.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V4-H01] {mensaje}", flush=True)


# ---------------------------------------------------------------------------
# Bases de la referencia
# ---------------------------------------------------------------------------
def ajustar_cuatro_bases(
    textos_entrenar: pd.Series, textos_aplicar: pd.Series, y_entrenar: np.ndarray
) -> list[np.ndarray]:
    """Ajusta las cuatro bases de la referencia y devuelve sus probabilidades.

    Las bases son las del protocolo v2: M1 (word + LR), M2 (union + LR C=2),
    M3 (union + LinearSVC) y M4 (char + LR). El orden es fijo porque las features
    del meta se construyen concatenando en ese orden.

    Args:
        textos_entrenar: textos del pliegue de entrenamiento.
        textos_aplicar: textos a predecir.
        y_entrenar: matriz de etiquetas del entrenamiento.

    Returns:
        Lista con cuatro vectores de probabilidades de forma ``(n, 3)``.
    """
    palabra = TfidfVectorizer(**PARAMETROS_WORD)
    caracter = TfidfVectorizer(**PARAMETROS_CHAR)
    palabra_entrenar = palabra.fit_transform(textos_entrenar)
    palabra_aplicar = palabra.transform(textos_aplicar)
    caracter_entrenar = caracter.fit_transform(textos_entrenar)
    caracter_aplicar = caracter.transform(textos_aplicar)
    union_entrenar = sparse.hstack([palabra_entrenar, caracter_entrenar], format="csr")
    union_aplicar = sparse.hstack([palabra_aplicar, caracter_aplicar], format="csr")

    return [
        _ajustar_lineales(palabra_entrenar, palabra_aplicar, y_entrenar, "lr"),
        _ajustar_lineales(union_entrenar, union_aplicar, y_entrenar, "lr"),
        _ajustar_lineales(union_entrenar, union_aplicar, y_entrenar, "svc"),
        _ajustar_lineales(caracter_entrenar, caracter_aplicar, y_entrenar, "lr"),
    ]


def _ajustar_lineales(
    matriz_entrenar: sparse.csr_matrix,
    matriz_aplicar: sparse.csr_matrix,
    y_entrenar: np.ndarray,
    modelo: str,
) -> np.ndarray:
    """Ajusta tres modelos binarios (uno por etiqueta) y devuelve probabilidades.

    Args:
        matriz_entrenar: matriz de entrenamiento.
        matriz_aplicar: matriz a predecir.
        y_entrenar: matriz de etiquetas del entrenamiento.
        modelo: ``lr`` o ``svc``.

    Returns:
        Matriz de probabilidades por etiqueta.
    """
    salida = np.zeros((matriz_aplicar.shape[0], len(ETIQUETAS)))
    for posicion in range(len(ETIQUETAS)):
        if modelo == "lr":
            estimador = LogisticRegression(
                C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear",
                random_state=SEMILLA,
            )
        else:
            estimador = LinearSVC(
                C=1.0, class_weight="balanced", max_iter=5000, random_state=SEMILLA
            )
        estimador.fit(matriz_entrenar, y_entrenar[:, posicion])
        if hasattr(estimador, "predict_proba"):
            salida[:, posicion] = estimador.predict_proba(matriz_aplicar)[:, 1]
        else:
            decision = estimador.decision_function(matriz_aplicar)
            salida[:, posicion] = 1.0 / (1.0 + np.exp(-decision))
    return salida


# ---------------------------------------------------------------------------
# Features y transformaciones del meta
# ---------------------------------------------------------------------------
def features_meta(probas: list[np.ndarray]) -> np.ndarray:
    """Construye las 36 features del meta a partir de las cuatro bases.

    Por cada base y etiqueta se usan la probabilidad, el margen respecto a 0.5 y la
    entropia binaria, en ese orden.

    Args:
        probas: lista de cuatro vectores de probabilidades.

    Returns:
        Matriz de 36 columnas.
    """
    columnas = []
    for proba in probas:
        recorte = np.clip(proba, EPS, 1.0 - EPS)
        columnas.extend([recorte[:, j] for j in range(len(ETIQUETAS))])
        columnas.extend([recorte[:, j] - 0.5 for j in range(len(ETIQUETAS))])
        columnas.extend(
            [
                -(recorte[:, j] * np.log(recorte[:, j]) + (1 - recorte[:, j]) * np.log(1 - recorte[:, j]))
                for j in range(len(ETIQUETAS))
            ]
        )
    return np.column_stack(columnas)


def alinear_cuantiles(x_nuevo: np.ndarray, x_referencia: np.ndarray) -> np.ndarray:
    """Alinea por cuantiles una matriz nueva a la distribucion de referencia.

    Para cada columna se calcula la posicion del cuantil de cada valor nuevo
    dentro de la referencia y se devuelve el valor de la referencia en ese
    cuantil. Es la transformacion que corrige un cambio de escala sin suponer
    ninguna forma funcional.

    Args:
        x_nuevo: matriz a alinear.
        x_referencia: matriz de referencia (regimen de entrenamiento).

    Returns:
        Matriz alineada con la misma forma.
    """
    salida = np.zeros_like(x_nuevo, dtype=float)
    for columna in range(x_nuevo.shape[1]):
        referencia = np.sort(x_referencia[:, columna])
        posiciones = np.searchsorted(referencia, x_nuevo[:, columna], side="left")
        cuantiles = posiciones / max(len(referencia), 1)
        indices = np.clip(
            np.round(cuantiles * (len(referencia) - 1)).astype(int), 0, len(referencia) - 1
        )
        salida[:, columna] = referencia[indices]
    return salida


def calibrar_platt(
    x_nueva: np.ndarray, x_meta: np.ndarray, y_meta: np.ndarray
) -> np.ndarray:
    """Transforma las features nuevas con un calibrador ajustado en el regimen meta.

    Cada columna se calibra con una regresion logistica de una variable contra la
    etiqueta positiva de la base correspondiente. No usa las etiquetas de la
    validacion: el calibrador se ajusta solo con el entrenamiento del meta.

    Args:
        x_nueva: features del regimen de inferencia.
        x_meta: features del regimen de entrenamiento.
        y_meta: etiquetas del entrenamiento.

    Returns:
        Matriz transformada.
    """
    salida = np.zeros_like(x_nueva, dtype=float)
    for columna in range(x_nueva.shape[1]):
        etiqueta = (columna // 3) % len(ETIQUETAS)
        calibrador = LogisticRegression(max_iter=1000, solver="liblinear", random_state=SEMILLA)
        calibrador.fit(x_meta[:, columna].reshape(-1, 1), y_meta[:, etiqueta])
        salida[:, columna] = calibrador.predict_proba(x_nueva[:, columna].reshape(-1, 1))[:, 1]
    return salida


def medir_shift(x_meta: np.ndarray, x_inferencia: np.ndarray) -> dict[str, float]:
    """Mide el desplazamiento entre las dos distribuciones de features.

    Args:
        x_meta: features del regimen de entrenamiento del meta.
        x_inferencia: features del regimen de inferencia.

    Returns:
        Diccionario con el estadistico KS medio y maximo y la diferencia media
        absoluta de medias.
    """
    ks = [
        float(ks_2samp(x_meta[:, c], x_inferencia[:, c]).statistic)
        for c in range(x_meta.shape[1])
    ]
    diferencias = [
        float(x_inferencia[:, c].mean() - x_meta[:, c].mean())
        for c in range(x_meta.shape[1])
    ]
    return {
        "ks_medio": round(float(np.mean(ks)), 4),
        "ks_maximo": round(float(np.max(ks)), 4),
        "diferencia_media_absoluta": round(float(np.mean(np.abs(diferencias))), 5),
    }


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
    """Evalua un meta con los umbrales de V03 y con 0.5.

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
    """Ejecuta la medicion del shift de las entradas del meta."""
    inicio_tiempo = time.time()
    inicio = datetime.now(timezone.utc)
    log("Shift de calibracion de las entradas del meta (TRACK H)")

    oof = np.load(OOF_PATH, allow_pickle=True)
    ids_dev = oof["ids"].astype(np.int64)
    y_dev = oof["y_true"].astype(int)
    pred_ref = oof["pred"].astype(int)

    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos[indices_dev]
    pliegues = pliegues_corregidos(desarrollo, grupos_dev)
    textos = desarrollo["text"]

    variantes = {
        nombre: np.zeros((len(desarrollo), len(ETIQUETAS)))
        for nombre in ("meta_control", "meta_shift", "meta_shift_cuantil", "meta_shift_platt")
    }
    medidas: list[dict[str, float]] = []

    for indice_fold, (entrenar, validar) in enumerate(pliegues):
        log(f"   pliegue {indice_fold + 1}/5 (entrenar {len(entrenar)}, validar {len(validar)})")
        y_entrenar = y_dev[entrenar]
        etiquetas_internas = combinaciones(desarrollo.iloc[entrenar]).reset_index(drop=True)
        interno = StratifiedGroupKFold(
            n_splits=N_SUBPLIEGUES, shuffle=True, random_state=SEMILLA + indice_fold
        )
        pares_internos = list(
            interno.split(desarrollo.iloc[entrenar], etiquetas_internas, groups=grupos_dev[entrenar])
        )

        # Regimen meta: cada ejemplo de T recibe la prediccion de bases que
        # entrenaron sin el (3/4 de T).
        features_entrenamiento = np.zeros((len(entrenar), 36))
        features_control = np.zeros((len(validar), 36))
        for sub_entrenar, sub_validar in pares_internos:
            textos_sub = textos.iloc[entrenar[sub_entrenar]]
            y_sub = y_dev[entrenar[sub_entrenar]]
            features_entrenamiento[sub_validar] = features_meta(
                ajustar_cuatro_bases(textos_sub, textos.iloc[entrenar[sub_validar]], y_sub)
            )
            features_control += features_meta(
                ajustar_cuatro_bases(textos_sub, textos.iloc[validar], y_sub)
            )
        features_control /= len(pares_internos)

        # Regimen inferencia: bases ajustadas sobre el 100 % de T.
        features_inferencia = features_meta(
            ajustar_cuatro_bases(textos.iloc[entrenar], textos.iloc[validar], y_entrenar)
        )
        medidas.append(medir_shift(features_entrenamiento, features_inferencia))

        meta = Ridge(alpha=ALPHA_META)
        meta.fit(features_entrenamiento, y_entrenar)
        variantes["meta_control"][validar] = np.clip(
            meta.predict(features_control), 0.0, 1.0
        )
        variantes["meta_shift"][validar] = np.clip(meta.predict(features_inferencia), 0.0, 1.0)
        variantes["meta_shift_cuantil"][validar] = np.clip(
            meta.predict(alinear_cuantiles(features_inferencia, features_entrenamiento)), 0.0, 1.0
        )
        variantes["meta_shift_platt"][validar] = np.clip(
            meta.predict(
                calibrar_platt(features_inferencia, features_entrenamiento, y_entrenar)
            ),
            0.0,
            1.0,
        )
        log(
            f"      KS medio del shift = {medidas[-1]['ks_medio']:.4f} "
            f"(max {medidas[-1]['ks_maximo']:.4f})"
        )

    resultados = [
        evaluar(nombre, proba, y_dev, pliegues, pred_ref)
        for nombre, proba in variantes.items()
    ]
    tabla = pd.DataFrame(resultados).sort_values("corrected_cv_umbrales_v03", ascending=False)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(CSV_PATH, index=False)

    por_nombre = {fila["variante"]: fila for fila in resultados}
    dano = por_nombre["meta_control"]["corrected_cv_umbrales_v03"] - por_nombre["meta_shift"][
        "corrected_cv_umbrales_v03"
    ]
    recupera_cuantil = por_nombre["meta_shift_cuantil"][
        "corrected_cv_umbrales_v03"
    ] - por_nombre["meta_shift"]["corrected_cv_umbrales_v03"]
    recupera_platt = por_nombre["meta_shift_platt"]["corrected_cv_umbrales_v03"] - por_nombre[
        "meta_shift"
    ]["corrected_cv_umbrales_v03"]
    cumple = bool(
        por_nombre["meta_shift_cuantil"]["corrected_cv_umbrales_v03"] - REFERENCIA_V03 >= 0.005
        and por_nombre["meta_shift_cuantil"]["pliegues_mejorados"] >= 3
    )
    if cumple:
        decision = "ADOPTAR: la alineacion recupera la ventaja del meta"
    elif dano > 0.002:
        decision = "CERRADA: el shift es real y midible, pero la alineacion no recupera la ventaja"
    else:
        decision = "CERRADA: el shift no explica la no transferencia (dano dentro del ruido)"

    resumen = {
        "experiment_id": "V4-H01",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duracion_s": round(time.time() - inicio_tiempo, 2),
        "referencia_v03": REFERENCIA_V03,
        "referencia_meta_v2_dev": REFERENCIA_META_V2,
        "medidas_shift": medidas,
        "ks_medio_global": round(float(np.mean([m["ks_medio"] for m in medidas])), 4),
        "dano_por_shift": round(float(dano), 6),
        "recuperacion_cuantil": round(float(recupera_cuantil), 6),
        "recuperacion_platt": round(float(recupera_platt), 6),
        "cumple_criterio_D016": cumple,
        "decision": decision,
        "detalle": resultados,
    }
    JSON_PATH.write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(NPZ_PATH, ids=ids_dev, y_true=y_dev, **variantes)
    _escribir_figura(tabla)
    for fila in resultados:
        log(f"   {fila['variante']:22s} corrected_cv={fila['corrected_cv_umbrales_v03']:.4f}")
    log(f"   dano del shift = {dano:+.4f}")
    log(f"   decision: {decision}")


def _escribir_figura(tabla: pd.DataFrame) -> None:
    """Dibuja el corrected_cv de las variantes del meta.

    Args:
        tabla: tabla de resultados.
    """
    figura, eje = plt.subplots(figsize=(9, 4))
    colores = [
        "#C44E52" if valor <= REFERENCIA_V03 else "#4C72B0"
        for valor in tabla["corrected_cv_umbrales_v03"]
    ]
    eje.barh(tabla["variante"], tabla["corrected_cv_umbrales_v03"], color=colores)
    eje.axvline(REFERENCIA_V03, color="#55A868", linestyle="--", label="V03 = 0.7772")
    eje.axvline(
        REFERENCIA_META_V2, color="#DD8452", linestyle=":", label="meta v2 (dev) = 0.7874"
    )
    eje.set_xlabel("corrected_cv con umbrales de V03")
    eje.set_title("Shift de las entradas del meta (v4, TRACK H)")
    eje.legend()
    figura.tight_layout()
    FIGURA_PATH.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(FIGURA_PATH, dpi=150)
    plt.close(figura)


if __name__ == "__main__":
    main()
