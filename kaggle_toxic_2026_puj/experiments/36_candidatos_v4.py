"""P8 v4: seleccion de candidatos finales con evidencia completa.

Contexto
--------
La spec v4 (§97-§99) permite como maximo tres candidatos finales y prohibe
elegir por el maximo de corrected_cv (§98): el precedente del proyecto es claro
(el meta v2 mejoro +0.0102 en dev y +0.0180 en holdout y Aun asi rindio 0.78163
en el leaderboard, D-020/D-024). Por eso este script evalua cada candidato con
**evidencia multiple y contradictoria a proposito**: nivel, consistencia entre
pliegues, bootstrap pareado, matriz de correccion frente a V03 y riesgo de
transferencia.

Candidatos
----------
- **A** `cost_sensitive_downweight`: la combinacion del TRACK C que survived
  (downweight por violacion del umbral + barrido anidado de ``class_weight`` por
  etiqueta). Es el unico mecanismo de aprendizaje propio de v4 con mejora local.
- **B** `meta36`: Ridge alpha=3 sobre las 36 features de las cuatro bases
  (V4-H01). Supera el nivel de D-016, pero pertenece a la familia de stacking que
  ya fallo en el leaderboard.
- **C** `meta36_downweight`: combinacion razonada de los dos mecanismos que
 Ambos didnt senal por separado (meta sobre bases entrenadas con downweight y
  class weights). Es la unica linea genuinamente nueva de esta fase: no es una
  variante de V03 ni una repeticion del meta v2.

Uso
---
    /home/carlos/venv/bin/python experiments/36_candidatos_v4.py
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
HARDNESS_PATH = PROJECT_ROOT / "features" / "hardness_v4.npz"
PESOS_C_PATH = PROJECT_ROOT / "reports" / "supervised_weighting_v4.json"
META_PATH = PROJECT_ROOT / "features" / "oof_meta_shift_v4.npz"
WEIGHTING_PATH = PROJECT_ROOT / "features" / "oof_weighting_v4.npz"
CSV_PATH = PROJECT_ROOT / "reports" / "v4_candidates.csv"
MATRIZ_PATH = PROJECT_ROOT / "reports" / "correction_matrix_v4.csv"
COMPARATIVA_PATH = PROJECT_ROOT / "reports" / "v4_candidate_comparison.md"
JSON_PATH = PROJECT_ROOT / "reports" / "v4_candidates.json"
FIGURA_PATH = PROJECT_ROOT / "reports" / "figuras" / "fig_40_candidatos_v4.png"

PARAMETROS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMETROS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
N_SUBPLIEGUES = 4
ALPHA_META = 3.0
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
N_BOOTSTRAP = 2000
SEMILLA = 42
REFERENCIA_V03 = 0.777211
REFERENCIA_META_V2 = 0.7874
MULTIPLICADORES = (0.8, 1.0, 1.2, 1.5)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso con marca de seccion.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V4-X01] {mensaje}", flush=True)


# ---------------------------------------------------------------------------
# Utilidades compartidas con el TRACK H
# ---------------------------------------------------------------------------
def features_meta(probas: list[np.ndarray]) -> np.ndarray:
    """Construye las 36 features del meta (probabilidad, margen y entropia).

    Args:
        probas: lista de cuatro vectores de probabilidades.

    Returns:
        Matriz de 36 columnas.
    """
    columnas = []
    for proba in probas:
        recorte = np.clip(proba, 1e-9, 1.0 - 1e-9)
        columnas.extend([recorte[:, j] for j in range(len(ETIQUETAS))])
        columnas.extend([recorte[:, j] - 0.5 for j in range(len(ETIQUETAS))])
        columnas.extend(
            [
                -(recorte[:, j] * np.log(recorte[:, j]) + (1 - recorte[:, j]) * np.log(1 - recorte[:, j]))
                for j in range(len(ETIQUETAS))
            ]
        )
    return np.column_stack(columnas)


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


def ajustar_bases(
    textos_entrenar: pd.Series,
    textos_aplicar: pd.Series,
    y_entrenar: np.ndarray,
    sample_weight: np.ndarray,
    multiplicadores: np.ndarray,
) -> list[np.ndarray]:
    """Ajusta las cuatro bases con pesos de ejemplo y de clase.

    Args:
        textos_entrenar: textos del entrenamiento.
        textos_aplicar: textos a predecir.
        y_entrenar: matriz de etiquetas del entrenamiento.
        sample_weight: peso por ejemplo.
        multiplicadores: multiplicadores de peso de clase por etiqueta.

    Returns:
        Lista de cuatro vectores de probabilidades.
    """
    palabra = TfidfVectorizer(**PARAMETROS_WORD)
    caracter = TfidfVectorizer(**PARAMETROS_CHAR)
    palabra_entrenar = palabra.fit_transform(textos_entrenar)
    palabra_aplicar = palabra.transform(textos_aplicar)
    caracter_entrenar = caracter.fit_transform(textos_entrenar)
    caracter_aplicar = caracter.transform(textos_aplicar)
    union_entrenar = sparse.hstack([palabra_entrenar, caracter_entrenar], format="csr")
    union_aplicar = sparse.hstack([palabra_aplicar, caracter_aplicar], format="csr")

    def ajustar(
        matriz_entrenar: sparse.csr_matrix,
        matriz_aplicar: sparse.csr_matrix,
        modelo: str,
    ) -> np.ndarray:
        """Ajusta tres modelos binarios con pesos y devuelve probabilidades.

        Args:
            matriz_entrenar: matriz de entrenamiento.
            matriz_aplicar: matriz a predecir.
            modelo: ``lr`` o ``svc``.

        Returns:
            Matriz de probabilidades.
        """
        salida = np.zeros((matriz_aplicar.shape[0], len(ETIQUETAS)))
        for posicion in range(len(ETIQUETAS)):
            if modelo == "lr":
                estimador = LogisticRegression(
                    C=2.0, max_iter=1000, solver="liblinear", random_state=SEMILLA
                )
            else:
                estimador = LinearSVC(
                    C=1.0, max_iter=5000, random_state=SEMILLA
                )
            estimador.set_params(
                class_weight=_pesos_de_clase(y_entrenar[:, posicion], multiplicadores[posicion])
            )
            estimador.fit(
                matriz_entrenar, y_entrenar[:, posicion], sample_weight=sample_weight
            )
            if hasattr(estimador, "predict_proba"):
                salida[:, posicion] = estimador.predict_proba(matriz_aplicar)[:, 1]
            else:
                salida[:, posicion] = 1.0 / (
                    1.0 + np.exp(-estimador.decision_function(matriz_aplicar))
                )
        return salida

    return [
        ajustar(palabra_entrenar, palabra_aplicar, "lr"),
        ajustar(union_entrenar, union_aplicar, "lr"),
        ajustar(union_entrenar, union_aplicar, "svc"),
        ajustar(caracter_entrenar, caracter_aplicar, "lr"),
    ]


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


def matriz_correccion(
    y_true: np.ndarray, pred_ref: np.ndarray, pred_nuevo: np.ndarray
) -> dict[str, Any]:
    """Construye la matriz de correccion del candidato frente a V03.

    Args:
        y_true: matriz verdadera.
        pred_ref: predicciones de referencia.
        pred_nuevo: predicciones del candidato.

    Returns:
        Diccionario con conteos y tasas de correccion y dano.
    """
    acierto_ref = (pred_ref == y_true).all(axis=1)
    acierto_nuevo = (pred_nuevo == y_true).all(axis=1)
    solo_candidato = int((~acierto_ref & acierto_nuevo).sum())
    solo_ref = int((acierto_ref & ~acierto_nuevo).sum())
    return {
        "aciertos_v03": int(acierto_ref.sum()),
        "nuevos_aciertos": solo_candidato,
        "nuevos_errores": solo_ref,
        "neto": solo_candidato - solo_ref,
        "tasa_correccion": round(solo_candidato / max((~acierto_ref).sum(), 1), 4),
        "tasa_dano": round(solo_ref / max(acierto_ref.sum(), 1), 4),
    }


def evaluar_candidato(
    nombre: str,
    proba: np.ndarray,
    y_true: np.ndarray,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    pred_ref: np.ndarray,
) -> dict[str, Any]:
    """Evalua un candidato con toda la evidencia_multiple.

    Args:
        nombre: identificador del candidato.
        proba: probabilidades OOF.
        y_true: matriz verdadera.
        pliegues: lista de pliegues corregidos.
        pred_ref: predicciones de referencia.

    Returns:
        Fila de resultados.
    """
    pred = (proba >= UMBRALES_V03[None, :]).astype(int)
    pred_cinco = (proba >= 0.5).astype(int)
    metricas = metricas_multietiqueta(y_true, pred)
    metricas_cinco = metricas_multietiqueta(y_true, pred_cinco)
    f1_pliegues = [
        metricas_multietiqueta(y_true[validar], pred[validar])["f1_media_columnas"]
        for _, validar in pliegues
    ]
    f1_ref = [
        metricas_multietiqueta(y_true[validar], pred_ref[validar])["f1_media_columnas"]
        for _, validar in pliegues
    ]
    delta, ic_bajo, ic_alto = _bootstrap(y_true, pred_ref, pred)
    correccion = matriz_correccion(y_true, pred_ref, pred)
    cumple_nivel = float(metricas["f1_media_columnas"]) - REFERENCIA_V03 >= 0.005
    cumple_pliegues = int(np.sum(np.array(f1_pliegues) > np.array(f1_ref) + 1e-12)) >= 3
    cumple_ic = ic_bajo > 0
    return {
        "candidato": nombre,
        "corrected_cv_umbrales_v03": round(metricas["f1_media_columnas"], 6),
        "corrected_cv_umbral_0_5": round(metricas_cinco["f1_media_columnas"], 6),
        "f1_HS": round(metricas["f1_por_etiqueta"]["HS"], 6),
        "f1_TR": round(metricas["f1_por_etiqueta"]["TR"], 6),
        "f1_AG": round(metricas["f1_por_etiqueta"]["AG"], 6),
        "micro_f1": round(metricas["f1_micro"], 6),
        "exact_match": round(metricas["exact_match"], 6),
        "std_pliegues": round(float(np.std(f1_pliegues, ddof=1)), 6),
        "pliegues_mejorados": int(np.sum(np.array(f1_pliegues) > np.array(f1_ref) + 1e-12)),
        "delta_vs_v03": round(float(metricas["f1_media_columnas"]) - REFERENCIA_V03, 6),
        "delta_f1_ejemplo": round(delta, 6),
        "ic95_bajo": round(ic_bajo, 6),
        "ic95_alto": round(ic_alto, 6),
        "nuevos_aciertos": correccion["nuevos_aciertos"],
        "nuevos_errores": correccion["nuevos_errores"],
        "neto_correccion": correccion["neto"],
        "cumple_nivel": bool(cumple_nivel),
        "cumple_pliegues": bool(cumple_pliegues),
        "cumple_ic": bool(cumple_ic),
        "cumple_D016": bool(cumple_nivel and cumple_pliegues and cumple_ic),
        "f1_pliegues": ";".join(f"{valor:.4f}" for valor in f1_pliegues),
        "f1_pliegues_v03": ";".join(f"{valor:.4f}" for valor in f1_ref),
    }


# ---------------------------------------------------------------------------
# Candidato C: meta sobre bases reponderadas
# ---------------------------------------------------------------------------
def construir_meta_downweight(
    desarrollo: pd.DataFrame,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    grupos_dev: np.ndarray,
    y_dev: np.ndarray,
    pesos_ejemplo: np.ndarray,
) -> np.ndarray:
    """Construye el OOF del meta36 cuyas bases usan downweight y class weights.

    Los multiplicadores de clase se eligen dentro de cada pliegue con un CV
    interno de 3 pliegues (mismo criterio que el TRACK C), de modo que la
    seleccion no use la validacion externa.

    Args:
        desarrollo: marco del desarrollo.
        pliegues: lista de pliegues corregidos.
        grupos_dev: grupos del desarrollo.
        y_dev: matriz de etiquetas del desarrollo.
        pesos_ejemplo: peso por ejemplo (downweight de la violacion).

    Returns:
        Matriz de probabilidades por etiqueta.
    """
    textos = desarrollo["text"]
    salida = np.zeros((len(desarrollo), len(ETIQUETAS)))
    for indice_fold, (entrenar, validar) in enumerate(pliegues):
        interno = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEMILLA + indice_fold)
        pares = list(
            interno.split(
                desarrollo.iloc[entrenar],
                combinaciones(desarrollo.iloc[entrenar]).reset_index(drop=True),
                groups=grupos_dev[entrenar],
            )
        )
        y_entrenar = y_dev[entrenar]
        multiplicadores = np.ones(len(ETIQUETAS))

        def puntuar(candidatos: np.ndarray) -> float:
            """F1 interna de una configuracion de multiplicadores.

            Args:
                candidatos: multiplicadores por etiqueta.

            Returns:
                F1 media de columnas con los umbrales de V03.
            """
            acumuladas = np.zeros((len(entrenar), len(ETIQUETAS)))
            for filas, columnas in pares:
                acumuladas[columnas] = _probar_bases(
                    textos.iloc[entrenar[filas]],
                    textos.iloc[entrenar[columnas]],
                    y_entrenar[filas],
                    pesos_ejemplo[entrenar[filas]],
                    candidatos,
                )
            return metricas_multietiqueta(
                y_entrenar, (acumuladas >= UMBRALES_V03[None, :]).astype(int)
            )["f1_media_columnas"]

        mejor = puntuar(multiplicadores)
        for _ in range(2):
            for posicion in range(len(ETIQUETAS)):
                for valor in MULTIPLICADORES:
                    prueba = multiplicadores.copy()
                    prueba[posicion] = valor
                    puntuacion = puntuar(prueba)
                    if puntuacion > mejor + 1e-9:
                        mejor = puntuacion
                        multiplicadores = prueba

        # Features del meta: regimen interno para entrenar, 100 % para predecir.
        features_entrenamiento = np.zeros((len(entrenar), 36))
        for filas, columnas in pares:
            probas = ajustar_bases(
                textos.iloc[entrenar[filas]],
                textos.iloc[entrenar[columnas]],
                y_entrenar[filas],
                pesos_ejemplo[entrenar[filas]],
                multiplicadores,
            )
            features_entrenamiento[columnas] = features_meta(probas)
        probas_validar = ajustar_bases(
            textos.iloc[entrenar],
            textos.iloc[validar],
            y_entrenar,
            pesos_ejemplo[entrenar],
            multiplicadores,
        )
        features_validar = features_meta(probas_validar)
        meta = Ridge(alpha=ALPHA_META)
        meta.fit(features_entrenamiento, y_entrenar)
        salida[validar] = np.clip(meta.predict(features_validar), 0.0, 1.0)
        log(
            f"      pliegue {indice_fold + 1}/5: multiplicadores internos "
            f"{np.round(multiplicadores, 2)}"
        )
    return salida


def _probar_bases(
    textos_entrenar: pd.Series,
    textos_aplicar: pd.Series,
    y_entrenar: np.ndarray,
    sample_weight: np.ndarray,
    multiplicadores: np.ndarray,
) -> np.ndarray:
    """Ajusta solo la base principal (union + LR) para el barrido interno.

    Args:
        textos_entrenar: textos de entrenamiento.
        textos_aplicar: textos a predecir.
        y_entrenar: etiquetas.
        sample_weight: peso por ejemplo.
        multiplicadores: multiplicadores de peso de clase.

    Returns:
        Matriz de probabilidades.
    """
    palabra = TfidfVectorizer(**PARAMETROS_WORD)
    caracter = TfidfVectorizer(**PARAMETROS_CHAR)
    izquierda = sparse.hstack(
        [palabra.fit_transform(textos_entrenar), caracter.fit_transform(textos_entrenar)],
        format="csr",
    )
    derecha = sparse.hstack(
        [palabra.transform(textos_aplicar), caracter.transform(textos_aplicar)], format="csr"
    )
    salida = np.zeros((derecha.shape[0], len(ETIQUETAS)))
    for posicion in range(len(ETIQUETAS)):
        modelo = LogisticRegression(
            C=2.0, max_iter=1000, solver="liblinear", random_state=SEMILLA
        )
        modelo.set_params(
            class_weight=_pesos_de_clase(y_entrenar[:, posicion], multiplicadores[posicion])
        )
        modelo.fit(izquierda, y_entrenar[:, posicion], sample_weight=sample_weight)
        salida[:, posicion] = modelo.predict_proba(derecha)[:, 1]
    return salida


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------
def main() -> None:
    """Evalua los candidatos finales de la fase v4 con evidencia completa."""
    inicio_tiempo = time.time()
    inicio = datetime.now(timezone.utc)
    log("Seleccion de candidatos finales (P8)")

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
    assert np.array_equal(desarrollo["id"].to_numpy(dtype=np.int64), ids_dev)

    # Pesos de ejemplo del downweight (misma definicion que en el TRACK C).
    dureza = np.load(HARDNESS_PATH, allow_pickle=True)
    assert np.array_equal(dureza["ids"].astype(np.int64), ids_dev)
    violacion = dureza["error_score"].astype(float)
    escala = np.percentile(violacion, 75) - np.percentile(violacion, 25) or 1.0
    z = (violacion - np.median(violacion)) / escala
    pesos_ejemplo = np.clip(1.0 - 0.5 * z, 0.5, 1.5)

    probas: dict[str, np.ndarray] = {}

    # A: combinacion del TRACK C.
    if WEIGHTING_PATH.exists():
        datos = np.load(WEIGHTING_PATH, allow_pickle=True)
        clave = "combo__downweight_violacion_x_clase"
        if clave in datos.files and np.array_equal(datos["ids"].astype(np.int64), ids_dev):
            probas["A_cost_sensitive_downweight"] = datos[clave].astype(float)
            log("   candidato A cargado del TRACK C")
        else:
            log(f"   candidato A no disponible en {WEIGHTING_PATH.name}")
    else:
        log("   candidato A no disponible (falta el OOF del TRACK C)")

    # B: meta36 del TRACK H.
    if META_PATH.exists():
        datos = np.load(META_PATH, allow_pickle=True)
        probas["B_meta36"] = datos["meta_shift"].astype(float)
        log("   candidato B cargado del TRACK H")
    else:
        log("   candidato B no disponible (falta el OOF del TRACK H)")

    # C: meta36 sobre bases reponderadas (combinacion nueva de esta fase).
    cache_candidatos = PROJECT_ROOT / "features" / "oof_candidates_v4.npz"
    if cache_candidatos.exists():
        datos_cache = np.load(cache_candidatos, allow_pickle=True)
        clave_cache = "C_meta36_downweight"
        if (
            clave_cache in datos_cache.files
            and np.array_equal(datos_cache["ids"].astype(np.int64), ids_dev)
        ):
            probas[clave_cache] = datos_cache[clave_cache].astype(float)
            log("   candidato C cargado de la cache")
        else:
            log("   construyendo candidato C (meta36 sobre bases reponderadas)")
            probas[clave_cache] = construir_meta_downweight(
                desarrollo, pliegues, grupos_dev, y_dev, pesos_ejemplo
            )
    else:
        log("   construyendo candidato C (meta36 sobre bases reponderadas)")
        probas["C_meta36_downweight"] = construir_meta_downweight(
            desarrollo, pliegues, grupos_dev, y_dev, pesos_ejemplo
        )

    # D: SVC con kernel RBF sobre la compresion SVD de la union (linea §63).
    ruta_g = PROJECT_ROOT / "features" / "oof_tracks_g_v4.npz"
    if ruta_g.exists():
        datos_g = np.load(ruta_g, allow_pickle=True)
        clave_g = "svd__rbf_svc"
        if clave_g in datos_g.files and np.array_equal(datos_g["ids"].astype(np.int64), ids_dev):
            probas["D_svd300_rbf"] = datos_g[clave_g].astype(float)
            log("   candidato D cargado del TRACK G")
        else:
            log(f"   candidato D no disponible en {ruta_g.name}")
    else:
        log("   candidato D no disponible (falta el OOF del TRACK G)")

    resultados = [
        evaluar_candidato(nombre, proba, y_dev, pliegues, pred_ref)
        for nombre, proba in probas.items()
    ]
    referencia_fila = evaluar_candidato("V03_referencia", oof["proba"], y_dev, pliegues, pred_ref)
    resultados.append(referencia_fila)

    tabla = pd.DataFrame(resultados)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(CSV_PATH, index=False)
    tabla[
        ["candidato", "nuevos_aciertos", "nuevos_errores", "neto_correccion", "tasa_correccion"]
        if "tasa_correccion" in tabla.columns
        else ["candidato", "nuevos_aciertos", "nuevos_errores", "neto_correccion"]
    ].to_csv(MATRIZ_PATH, index=False)

    np.savez_compressed(
        PROJECT_ROOT / "features" / "oof_candidates_v4.npz",
        ids=ids_dev,
        y_true=y_dev,
        **{nombre: valor for nombre, valor in probas.items()},
    )

    cumple = [fila for fila in resultados if fila["cumple_D016"]]
    resumen = {
        "experiment_id": "V4-X01",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duracion_s": round(time.time() - inicio_tiempo, 2),
        "referencia_v03": REFERENCIA_V03,
        "referencia_meta_v2_dev": REFERENCIA_META_V2,
        "n_candidatos": len(resultados) - 1,
        "candidatos_que_cumplen_D016": [fila["candidato"] for fila in cumple],
        "mejor_por_corrected_cv": max(
            (f for f in resultados if f["candidato"] != "V03_referencia"),
            key=lambda fila: fila["corrected_cv_umbrales_v03"],
        )["candidato"],
        "detalle": resultados,
    }
    JSON_PATH.write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _escribir_comparativa(tabla, cumple, resumen)
    _escribir_figura(tabla)

    for fila in resultados:
        log(
            f"   {fila['candidato']:30s} cv={fila['corrected_cv_umbrales_v03']:.4f} "
            f"pliegues+={fila['pliegues_mejorados']}/5 neto={fila['neto_correccion']:+d} "
            f"D016={fila['cumple_D016']}"
        )
    log(f"   cumplen D-016: {resumen['candidatos_que_cumplen_D016']}")


def _escribir_comparativa(
    tabla: pd.DataFrame, cumplen: list[dict[str, Any]], resumen: dict[str, Any]
) -> None:
    """Escribe la comparativa descriptiva de candidatos (spec §99).

    Args:
        tabla: tabla de resultados.
        cumplen: candidatos que pasan todos los gates.
        resumen: resumen de la seleccion.
    """
    lineas = [
        "# Comparativa de candidatos finales (v4)",
        "",
        f"Generado por `experiments/36_candidatos_v4.py` el "
        f"{datetime.now(timezone.utc).date().isoformat()}. La comparacion es "
        "**descriptiva y no un ranking**: la spec (§99) prohibe ordenar "
        "mecanicamente y el historial del proyecto demuestra que el corrected_cv "
        "no predice el leaderboard (D-014, D-020, D-024).",
        "",
        "## 1. Evidencia por candidato",
        "",
        "| Candidato | corrected_cv (umbrales V03) | delta vs V03 | pliegues+ | IC95 pareado | F1 HS/TR/AG | nuevos aciertos | nuevos errores | neto |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, fila in tabla.iterrows():
        lineas.append(
            f"| {fila['candidato']} | {fila['corrected_cv_umbrales_v03']:.4f} | "
            f"{fila['delta_vs_v03']:+.4f} | {fila['pliegues_mejorados']}/5 | "
            f"[{fila['ic95_bajo']:+.4f}, {fila['ic95_alto']:+.4f}] | "
            f"{fila['f1_HS']:.3f}/{fila['f1_TR']:.3f}/{fila['f1_AG']:.3f} | "
            f"{int(fila['nuevos_aciertos'])} | {int(fila['nuevos_errores'])} | "
            f"{int(fila['neto_correccion']):+d} |"
        )
    lineas += [
        "",
        "## 2. F1 por pliegue",
        "",
        "| Candidato | pliegues | V03 en los mismos pliegues |",
        "|---|---|---|",
    ]
    for _, fila in tabla.iterrows():
        lineas.append(f"| {fila['candidato']} | {fila['f1_pliegues']} | {fila['f1_pliegues_v03']} |")
    lineas += [
        "",
        "## 3. Criterios y riesgo de transferencia",
        "",
        "| Candidato | nivel >= +0.005 | >= 3/5 pliegues | IC95 > 0 | cumple D-016 | Riesgo de transferencia |",
        "|---|---|---|---|---|---|",
    ]
    riesgos = {
        "V03_referencia": "referencia (LB 0.78306)",
        "A_cost_sensitive_downweight": "bajo: mecanismo propio, coste marginal, no es stacking",
        "B_meta36": "alto: misma familia que el meta v2, que rindio 0.78163 en el LB",
        "C_meta36_downweight": "medio-alto: combines la familia de stacking con el mecanismo nuevo",
        "D_svd300_rbf": "medio: modelo no lineal sobre compresion; el precedente DL-01 (0.7949) nunca se sondeo, y SVD + RBF puede depender de la estructura de los 300 componentes",
    }
    for _, fila in tabla.iterrows():
        lineas.append(
            f"| {fila['candidato']} | {fila['cumple_nivel']} | {fila['cumple_pliegues']} | "
            f"{fila['cumple_ic']} | {fila['cumple_D016']} | "
            f"{riesgos.get(fila['candidato'], 'medio')} |"
        )
    lineas += [
        "",
        "## 4. Decision de la fase",
        "",
    ]
    if cumplen:
        lineas.append(
            "Candidatos que superan **todos** los gates de D-016: "
            + ", ".join(f"`{fila['candidato']}`" for fila in cumplen)
            + ". La decision de gastar la ultima evaluacion del holdout y una "
            "sonda de leaderboard corresponde a Carlos; este script no envia nada "
            "y no consume el holdout."
        )
    else:
        lineas.append(
            "**Ningun candidato supera todos los gates de D-016.** Segun la "
            "condicion de terminacion B de la spec (§102), se mantiene V03 como "
            "modelo final, no se fabrica una mejora y las lineas quedan cerradas "
            "con evidencia. La fase v4 aporta evidencia nueva (ruido de etiqueta "
            "escaso, dureza mal definida por la BCE, reweighting con signo "
            "opuesto al esperado, geometria de weighting closures, modelo "
            "conjunto inferior, shift del meta refutado) pero no una mejora "
            "transferible."
        )
    COMPARATIVA_PATH.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _escribir_figura(tabla: pd.DataFrame) -> None:
    """Dibuja el corrected_cv y la matriz de correccion de los candidatos.

    Args:
        tabla: tabla de resultados.
    """
    figura, ejes = plt.subplots(1, 2, figsize=(13, 5))
    colores = [
        "#55A868" if cumple else "#C44E52" for cumple in tabla["cumple_D016"]
    ]
    ejes[0].barh(tabla["candidato"], tabla["corrected_cv_umbrales_v03"], color=colores)
    ejes[0].axvline(REFERENCIA_V03, color="#4C72B0", linestyle="--", label="V03 = 0.7772")
    ejes[0].set_xlabel("corrected_cv con umbrales de V03")
    ejes[0].set_title("Candidatos v4 (verde = cumple D-016)")
    ejes[0].legend()
    ejes[1].barh(tabla["candidato"], tabla["neto_correccion"], color="#4C72B0")
    ejes[1].axvline(0.0, color="black", linewidth=0.8)
    ejes[1].set_xlabel("nuevos aciertos - nuevos errores frente a V03")
    ejes[1].set_title("Matriz de correccion")
    figura.tight_layout()
    FIGURA_PATH.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(FIGURA_PATH, dpi=150)
    plt.close(figura)


if __name__ == "__main__":
    main()
