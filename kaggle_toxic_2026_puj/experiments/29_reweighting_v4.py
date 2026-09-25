"""TRACK B v4 (experimento): reponderar ejemplos por dureza OOF.

Contexto
--------
Con el mapa de dureza de ``28_hardness_v4.py`` (D-026) se establecio que en este
corpus la medida de dureza util es la **distancia a la frontera de decision**
(cercania al umbral en la direccion de la etiqueta verdadera) y la magnitud de
la violacion del umbral, y que la BCE agregada no preserva el orden de
dificultad. Este script pone a prueba la mecanica de la spec (§10-§12):
ponderar los ejemplos de entrenamiento con pesos derivados de predicciones OOF
y comprobar si la F1 mejora frente a V03.

Diseno sin fuga
---------------
Los pesos de un ejemplo nunca pueden derivarse de un modelo que lo vio en
entrenamiento. Por eso, para cada pliegue externo:

1. se hace un CV **anidado** de 4 pliegues sobre el entrenamiento externo para
   obtener la dureza de cada ejemplo de entrenamiento (modelo interno que no lo
   vio);
2. se calculan los pesos a partir de esa dureza;
3. se entrena el modelo del pliegue externo con esos pesos y se evalua en la
   parte de validacion externa.

Comparacion
-----------
Todos los candidatos se miden con los **mismos umbrales congelados de V03**
(0.44/0.51/0.46) para aislar el efecto del mecanismo de ponderacion, y con 0.5
como control. La referencia es el corrected_cv de V03 = 0.777211.

Uso
---
    /home/carlos/venv/bin/python experiments/29_reweighting_v4.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
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
    grupos_fusionados,
    holdout_v2,
    combinaciones,
    pliegues_corregidos,
)

OOF_PATH = PROJECT_ROOT / "features" / "oof_v03_final.npz"
CSV_PATH = PROJECT_ROOT / "reports" / "reweighting_v4.csv"
JSON_PATH = PROJECT_ROOT / "reports" / "reweighting_v4.json"
NPZ_PATH = PROJECT_ROOT / "features" / "oof_reweight_v4.npz"
FIGURA_PATH = PROJECT_ROOT / "reports" / "figuras" / "fig_33_reweighting_v4.png"

PARAMETROS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMETROS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
N_SUBPLIEGUES = 4
N_BOOTSTRAP = 2000
SEMILLA = 42


def log(mensaje: str) -> None:
    """Imprime una linea de progreso con marca de seccion.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V4-B03] {mensaje}", flush=True)


# ---------------------------------------------------------------------------
# Representacion
# ---------------------------------------------------------------------------
def construir_union(textos: pd.Series, entrenar: np.ndarray, aplicar: np.ndarray):
    """Ajusta los vectorizadores en `entrenar` y transforma `aplicar`.

    Args:
        textos: serie con todos los textos del desarrollo.
        entrenar: indices posicionales para ajustar.
        aplicar: indices posicionales para transformar.

    Returns:
        Matriz dispersa union para `aplicar`.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    sub_entrenar = textos.iloc[entrenar]
    vectorizador_word = TfidfVectorizer(**PARAMETROS_WORD)
    vectorizador_char = TfidfVectorizer(**PARAMETROS_CHAR)
    matriz_word = vectorizador_word.fit_transform(sub_entrenar)
    matriz_char = vectorizador_char.fit_transform(sub_entrenar)
    union_entrenar = sparse.hstack([matriz_word, matriz_char], format="csr")
    union_aplicar = sparse.hstack(
        [
            vectorizador_word.transform(textos.iloc[aplicar]),
            vectorizador_char.transform(textos.iloc[aplicar]),
        ],
        format="csr",
    )
    return union_entrenar, union_aplicar


# ---------------------------------------------------------------------------
# Pesos a partir de la dureza
# ---------------------------------------------------------------------------
def escala_robusta(valores: np.ndarray) -> np.ndarray:
    """Estandariza un vector por su rango intercuartil, de forma resistente.

    Args:
        valores: vector de durezas.

    Returns:
        Vector centrado y escalado, con valores habituales en [-2, 2].
    """
    q25, q75 = np.percentile(valores, [25, 75])
    escala = (q75 - q25) or 1.0
    return (valores - float(np.median(valores))) / escala


def calcular_pesos(
    cercania: np.ndarray, violacion: np.ndarray, esquema: str
) -> np.ndarray:
    """Calcula el peso de cada ejemplo segun el esquema elegido.

    Args:
        cercania: cercania a la frontera por etiqueta (mayor = mas dificil).
        violacion: magnitud de la violacion del umbral por etiqueta.
        esquema: nombre del esquema de ponderacion.

    Returns:
        Vector de pesos por ejemplo.

    Raises:
        ValueError: si el esquema no existe.
    """
    if esquema == "none":
        return np.ones(cercania.shape[0], dtype=float)
    z_cercania = escala_robusta(cercania.mean(axis=1))
    z_violacion = escala_robusta(violacion.mean(axis=1))
    if esquema == "mild_up":
        return np.clip(1.0 + 0.5 * z_cercania, 0.5, 1.5)
    if esquema == "medium_up":
        return np.clip(1.0 + 1.0 * z_cercania, 0.2, 2.0)
    if esquema == "down_violacion":
        return np.clip(1.0 - 0.5 * z_violacion, 0.5, 1.5)
    if esquema == "clipped_extremos":
        pesos = np.ones(cercania.shape[0], dtype=float)
        corte = np.percentile(cercania.mean(axis=1), 90)
        pesos[cercania.mean(axis=1) >= corte] = 0.5
        return pesos
    if esquema == "medium_down":
        return np.clip(1.0 - 1.0 * z_cercania, 0.2, 2.0)
    raise ValueError(f"Esquema de ponderacion desconocido: {esquema}")


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def crear_modelo(nombre: str):
    """Crea un clasificador binario con los hiperparametros de la referencia.

    Args:
        nombre: ``lr`` o ``svc``.

    Returns:
        Modelo sin ajustar.
    """
    if nombre == "lr":
        return LogisticRegression(
            C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=SEMILLA
        )
    if nombre == "svc":
        return LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=SEMILLA)
    raise ValueError(f"Modelo desconocido: {nombre}")


def dureza_de_probabilidades(
    proba: np.ndarray, y_true: np.ndarray, umbrales: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convierte probabilidades en vectores de cercania y violacion.

    Args:
        proba: probabilidades predichas.
        y_true: etiquetas verdaderas de las mismas filas.
        umbrales: umbrales por etiqueta.

    Returns:
        Tupla `(cercania, violacion)`.
    """
    margen_favor = (proba - umbrales[None, :]) * (2 * y_true - 1)
    return 1.0 - np.abs(margen_favor), np.clip(-margen_favor, 0.0, None)


# ---------------------------------------------------------------------------
# Evaluacion
# ---------------------------------------------------------------------------
def f1_por_ejemplo(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calcula la F1 promedio de las tres etiquetas para cada ejemplo.

    Args:
        y_true: matriz verdadera binaria.
        y_pred: matriz predicha binaria.

    Returns:
        Vector con la F1 por ejemplo.
    """
    valores = []
    for indice in range(y_true.shape[0]):
        f1s = []
        for posicion in range(y_true.shape[1]):
            verdadero, predicho = int(y_true[indice, posicion]), int(y_pred[indice, posicion])
            if verdadero == 1 and predicho == 1:
                f1s.append(1.0)
            elif verdadero == 0 and predicho == 0:
                f1s.append(1.0)
            else:
                f1s.append(0.0)
        valores.append(float(np.mean(f1s)))
    return np.array(valores)


def bootstrap_pareado(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    semilla: int = SEMILLA,
) -> tuple[float, float, float]:
    """Estima el delta de F1 media entre dos predicciones con bootstrap pareado.

    Args:
        y_true: matriz verdadera binaria.
        pred_a: predicciones de la referencia.
        pred_b: predicciones del candidato.
        n_bootstrap: numero de remuestreos.
        semilla: semilla del generador.

    Returns:
        Tupla `(delta, ic95_bajo, ic95_alto)`.
    """
    f1_a = f1_por_ejemplo(y_true, pred_a)
    f1_b = f1_por_ejemplo(y_true, pred_b)
    generador = np.random.default_rng(semilla)
    n = len(f1_a)
    deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        indices = generador.integers(0, n, n)
        deltas[i] = f1_b[indices].mean() - f1_a[indices].mean()
    percentiles = np.percentile(deltas, [2.5, 97.5])
    return float(f1_b.mean() - f1_a.mean()), float(percentiles[0]), float(percentiles[1])


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------
def main() -> None:
    """Ejecuta la bateria de esquemas de ponderacion sobre la referencia."""
    inicio_tiempo = time.time()
    inicio = datetime.now(timezone.utc)
    log("Reweighting por dureza OOF (TRACK B, V4-B03/B04/B05)")

    oof = np.load(OOF_PATH, allow_pickle=True)
    ids_dev = oof["ids"].astype(np.int64)
    y_dev = oof["y_true"].astype(int)
    umbrales = oof["umbrales"].astype(float)
    proba_v03 = oof["proba"]
    pred_v03_ref = oof["pred"].astype(int)

    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos[indices_dev]
    assert np.array_equal(desarrollo["id"].to_numpy(dtype=np.int64), ids_dev), (
        "el orden del desarrollo no coincide con el OOF canonico"
    )
    textos = desarrollo["text"]
    pliegues = pliegues_corregidos(desarrollo, grupos_dev)
    etiquetas_texto = combinaciones(desarrollo)

    esquemas = [
        "none",
        "mild_up",
        "medium_up",
        "medium_down",
        "down_violacion",
        "clipped_extremos",
    ]
    modelos = ["lr", "svc"]
    resultados: list[dict[str, Any]] = []
    oof_guardado: dict[str, np.ndarray] = {}

    for nombre_modelo in modelos:
        log(f"   modelo {nombre_modelo}: vectores y OOF anidado por pliegue")
        # 1) Destreza anidada por pliegue (independiente del esquema de peso).
        cercania_anidada = np.zeros((len(desarrollo), len(ETIQUETAS)))
        violacion_anidada = np.zeros((len(desarrollo), len(ETIQUETAS)))
        for indice_fold, (entrenar, validar) in enumerate(pliegues):
            interno = StratifiedGroupKFold(
                n_splits=N_SUBPLIEGUES, shuffle=True, random_state=SEMILLA + indice_fold
            )
            for indices_subtrain, indices_subval in interno.split(
                desarrollo.iloc[entrenar], etiquetas_texto.iloc[entrenar], groups=grupos_dev[entrenar]
            ):
                subtrain = entrenar[indices_subtrain]
                subval = entrenar[indices_subval]
                union_subtrain, union_subval = construir_union(textos, subtrain, subval)
                modelo = crear_modelo(nombre_modelo)
                ajuste = _ajustar_por_etiqueta(modelo, union_subtrain, y_dev[subtrain])
                proba_subval = _predecir(ajuste, union_subval)
                cercania_anidada[subval], violacion_anidada[subval] = dureza_de_probabilidades(
                    proba_subval, y_dev[subval], umbrales
                )
            log(f"      pliegue {indice_fold + 1}/5: dureza anidada calculada ({len(entrenar)} filas)")

        # 2) Cada esquema: entrenar con pesos y acumular OOF externo.
        for esquema in esquemas:
            proba_oof = np.zeros((len(desarrollo), len(ETIQUETAS)))
            for indice_fold, (entrenar, validar) in enumerate(pliegues):
                pesos = calcular_pesos(
                    cercania_anidada[entrenar],
                    violacion_anidada[entrenar],
                    esquema,
                )
                union_entrenar, union_validar = construir_union(textos, entrenar, validar)
                modelo = crear_modelo(nombre_modelo)
                ajuste = _ajustar_por_etiqueta(
                    modelo, union_entrenar, y_dev[entrenar], sample_weight=pesos
                )
                proba_oof[validar] = _predecir(ajuste, union_validar)
            clave = f"{nombre_modelo}:{esquema}"
            oof_guardado[clave] = proba_oof
            resultados.append(_evaluar(clave, proba_oof, y_dev, umbrales, pliegues, pred_v03_ref))
            log(
                f"      {clave:34s} corrected_cv={resultados[-1]['corrected_cv_umbrales_v03']:.4f} "
                f"delta={resultados[-1]['delta_f1_ejemplo']:+.4f} "
                f"IC95=[{resultados[-1]['ic95_bajo']:+.4f}, {resultados[-1]['ic95_alto']:+.4f}]"
            )

        # 3) V4-B04: ponderacion independiente por etiqueta (solo LR).
        if nombre_modelo == "lr":
            proba_oof = np.zeros((len(desarrollo), len(ETIQUETAS)))
            for indice_fold, (entrenar, validar) in enumerate(pliegues):
                union_entrenar, union_validar = construir_union(textos, entrenar, validar)
                for posicion, etiqueta in enumerate(ETIQUETAS):
                    pesos_etiqueta = calcular_pesos(
                        cercania_anidada[entrenar][:, [posicion]],
                        violacion_anidada[entrenar][:, [posicion]],
                        "medium_up",
                    )
                    modelo = LogisticRegression(
                        C=2.0, class_weight="balanced", max_iter=1000,
                        solver="liblinear", random_state=SEMILLA,
                    )
                    modelo.fit(
                        union_entrenar, y_dev[entrenar, posicion], sample_weight=pesos_etiqueta
                    )
                    proba_oof[validar, posicion] = modelo.predict_proba(union_validar)[:, 1]
            clave = "lr:medium_up_por_etiqueta"
            oof_guardado[clave] = proba_oof
            resultados.append(_evaluar(clave, proba_oof, y_dev, umbrales, pliegues, pred_v03_ref))
            log(
                f"      {clave:34s} corrected_cv={resultados[-1]['corrected_cv_umbrales_v03']:.4f} "
                f"delta={resultados[-1]['delta_f1_ejemplo']:+.4f}"
            )

    # 4) V4-B05: tres estrategias de manejo de contradicciones (control).
    log("   V4-B05: estrategias de manejo de contradicciones (control)")
    dari = np.load(PROJECT_ROOT / "features" / "label_consistency_v4.npz", allow_pickle=True)
    ids_train = dari["ids"].astype(np.int64)
    discrepancia = dari["discrepancia_etiqueta"].astype(int)
    posicion_dev = {int(identificador): indice for indice, identificador in enumerate(ids_dev)}
    contradictorio = np.zeros((len(desarrollo), len(ETIQUETAS)), dtype=int)
    for indice, identificador in enumerate(ids_dev):
        fila = np.where(ids_train == int(identificador))[0][0]
        contradictorio[indice] = discrepancia[fila]
    mascara_contradicciones = contradictorio.max(axis=1) > 0
    resultados.append(
        _estrategia_contradicciones(
            "estrategia:downweight_contradicciones",
            mascara_contradicciones,
            contradictorio,
            pliegues,
            textos,
            y_dev,
            umbrales,
            pred_v03_ref,
        )
    )
    log(f"      downweight contradicciones: corrected_cv={resultados[-1]['corrected_cv_umbrales_v03']:.4f}")
    resultados.append(
        _estrategia_contradicciones(
            "estrategia:remove_contradicciones",
            mascara_contradicciones,
            contradictorio,
            pliegues,
            textos,
            y_dev,
            umbrales,
            pred_v03_ref,
            eliminar=True,
        )
    )
    log(f"      remove contradicciones:     corrected_cv={resultados[-1]['corrected_cv_umbrales_v03']:.4f}")

    tabla = pd.DataFrame(resultados).sort_values("corrected_cv_umbrales_v03", ascending=False)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(CSV_PATH, index=False)

    mejor = tabla.iloc[0]
    referencia = tabla[tabla["variante"] == "lr:none"].iloc[0]
    resumen = {
        "experiment_id": "V4-B03-B05",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duracion_s": round(time.time() - inicio_tiempo, 2),
        "referencia_v03_corrected_cv": 0.777211,
        "referencia_recalculada_lr_none": float(referencia["corrected_cv_umbrales_v03"]),
        "n_variantes": int(len(tabla)),
        "mejor_variante": str(mejor["variante"]),
        "mejor_corrected_cv": float(mejor["corrected_cv_umbrales_v03"]),
        "mejor_delta": float(mejor["delta_f1_ejemplo"]),
        "mejor_ic95": [float(mejor["ic95_bajo"]), float(mejor["ic95_alto"])],
        "n_variantes_con_delta_positivo": int((tabla["delta_f1_ejemplo"] > 0).sum()),
        "n_variantes_con_ic95_positivo": int((tabla["ic95_bajo"] > 0).sum()),
        "criterio_adopcion": "+0.005 y >= 3/5 pliegues (D-016)",
        "cumple_criterio": bool(
            float(mejor["corrected_cv_umbrales_v03"]) - 0.777211 >= 0.005
            and int(mejor["pliegues_mejorados"]) >= 3
        ),
        "detalle": resultados,
    }
    JSON_PATH.write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    np.savez_compressed(
        NPZ_PATH,
        ids=ids_dev,
        y_true=y_dev,
        umbrales=umbrales,
        **{clave.replace(":", "__"): valor for clave, valor in oof_guardado.items()},
    )
    _escribir_figura(tabla)

    log(f"   mejor variante: {resumen['mejor_variante']} ({resumen['mejor_corrected_cv']:.4f})")
    log(f"   cumple criterio D-016: {resumen['cumple_criterio']}")
    log(f"   artefactos: {CSV_PATH.name}, {JSON_PATH.name}, {NPZ_PATH.name}, {FIGURA_PATH.name}")


# ---------------------------------------------------------------------------
# Utilidades internas del experimento
# ---------------------------------------------------------------------------
def _ajustar_por_etiqueta(
    modelo, matriz, y_true, sample_weight: np.ndarray | None = None
) -> list[Any]:
    """Ajusta un clon del modelo para cada etiqueta y devuelve la lista.

    Args:
        modelo: modelo a clonar.
        matriz: matriz de entrenamiento.
        y_true: matriz de etiquetas.
        sample_weight: pesos opcionales por ejemplo.

    Returns:
        Lista de modelos ajustados, uno por etiqueta.
    """
    from sklearn.base import clone

    ajustados = []
    for posicion in range(y_true.shape[1]):
        clon = clone(modelo)
        if sample_weight is None:
            clon.fit(matriz, y_true[:, posicion])
        else:
            clon.fit(matriz, y_true[:, posicion], sample_weight=sample_weight)
        ajustados.append(clon)
    return ajustados


def _predecir(modelos, matriz) -> np.ndarray:
    """Calcula probabilidades (o decision_function) de los modelos ajustados.

    Args:
        modelos: lista de modelos ajustados.
        matriz: matriz a predecir.

    Returns:
        Matriz de probabilidades por etiqueta.
    """
    salida = np.zeros((matriz.shape[0], len(modelos)))
    for posicion, modelo in enumerate(modelos):
        if hasattr(modelo, "predict_proba"):
            salida[:, posicion] = modelo.predict_proba(matriz)[:, 1]
        else:
            decision = modelo.decision_function(matriz)
            salida[:, posicion] = 1.0 / (1.0 + np.exp(-decision))
    return salida


def _evaluar(
    clave: str,
    proba: np.ndarray,
    y_true: np.ndarray,
    umbrales: np.ndarray,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    pred_ref: np.ndarray,
) -> dict[str, Any]:
    """Evalua una variante con los umbrales de V03 y con 0.5.

    Args:
        clave: identificador de la variante.
        proba: probabilidades OOF.
        y_true: matriz verdadera.
        umbrales: umbrales congelados de V03.
        pliegues: lista de pliegues corregidos.
        pred_ref: predicciones de referencia (V03) para el bootstrap pareado.

    Returns:
        Fila de resultados.
    """
    pred_umbrales = (proba >= umbrales[None, :]).astype(int)
    pred_cinco = (proba >= 0.5).astype(int)
    metricas = metricas_multietiqueta(y_true, pred_umbrales)
    metricas_cinco = metricas_multietiqueta(y_true, pred_cinco)
    f1_pliegues = [
        metricas_multietiqueta(y_true[validar], pred_umbrales[validar])["f1_media_columnas"]
        for _, validar in pliegues
    ]
    delta, ic_bajo, ic_alto = bootstrap_pareado(y_true, pred_ref, pred_umbrales)
    pliegues_mejorados = int(
        np.sum(np.array(f1_pliegues) > np.array(_f1_pliegues_referencia(pred_ref, pliegues, y_true)))
    )
    macro = float(
        np.mean(
            [f1_score(y_true[:, j], pred_umbrales[:, j], zero_division=0) for j in range(3)]
        )
    )
    micro = metricas["f1_micro"]
    exact = metricas["exact_match"]
    return {
        "variante": clave,
        "corrected_cv_umbrales_v03": round(metricas["f1_media_columnas"], 6),
        "corrected_cv_umbral_0_5": round(metricas_cinco["f1_media_columnas"], 6),
        "f1_HS": round(metricas["f1_por_etiqueta"]["HS"], 6),
        "f1_TR": round(metricas["f1_por_etiqueta"]["TR"], 6),
        "f1_AG": round(metricas["f1_por_etiqueta"]["AG"], 6),
        "macro_f1": round(macro, 6),
        "micro_f1": round(micro, 6),
        "exact_match": round(exact, 6),
        "std_pliegues": round(float(np.std(f1_pliegues, ddof=1)), 6),
        "f1_pliegues": ";".join(f"{valor:.4f}" for valor in f1_pliegues),
        "pliegues_mejorados": pliegues_mejorados,
        "delta_f1_ejemplo": round(delta, 6),
        "ic95_bajo": round(ic_bajo, 6),
        "ic95_alto": round(ic_alto, 6),
    }


def _f1_pliegues_referencia(
    pred_ref: np.ndarray, pliegues: list[tuple[np.ndarray, np.ndarray]], y_true: np.ndarray
) -> list[float]:
    """Calcula la F1 por pliegue de la referencia.

    Args:
        pred_ref: predicciones de referencia.
        pliegues: lista de pliegues.
        y_true: matriz verdadera.

    Returns:
        Lista de F1 por pliegue.
    """
    return [
        metricas_multietiqueta(y_true[validar], pred_ref[validar])["f1_media_columnas"]
        for _, validar in pliegues
    ]


def _estrategia_contradicciones(
    clave: str,
    mascara_contradicciones: np.ndarray,
    contradictorio: np.ndarray,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    textos: pd.Series,
    y_true: np.ndarray,
    umbrales: np.ndarray,
    pred_ref: np.ndarray,
    eliminar: bool = False,
) -> dict[str, Any]:
    """Evalua downweight o remove de ejemplos contradictorios (V4-B05).

    Args:
        clave: identificador de la variante.
        mascara_contradicciones: vector booleano de ejemplos contradictorios.
        contradictorio: matriz de discrepancia por etiqueta.
        pliegues: lista de pliegues corregidos.
        textos: serie de textos del desarrollo.
        y_true: matriz verdadera.
        umbrales: umbrales de V03.
        pred_ref: predicciones de referencia.
        eliminar: si es ``True`` se eliminan del entrenamiento.

    Returns:
        Fila de resultados.
    """
    proba_oof = np.zeros_like(y_true, dtype=float)
    for entrenar, validar in pliegues:
        indices = entrenar
        pesos = np.where(mascara_contradicciones[entrenar], 0.3, 1.0)
        if eliminar:
            indices = entrenar[~mascara_contradicciones[entrenar]]
            pesos = np.ones(len(indices))
        union_entrenar, union_validar = construir_union(textos, indices, validar)
        modelo = crear_modelo("lr")
        ajuste = _ajustar_por_etiqueta(
            modelo, union_entrenar, y_true[indices], sample_weight=pesos
        )
        proba_oof[validar] = _predecir(ajuste, union_validar)
    fila = _evaluar(clave, proba_oof, y_true, umbrales, pliegues, pred_ref)
    fila["n_eliminados"] = int(mascara_contradicciones.sum()) if eliminar else 0
    return fila


def _escribir_figura(tabla: pd.DataFrame) -> None:
    """Dibuja el corrected_cv de cada variante con su IC95.

    Args:
        tabla: tabla de resultados.
    """
    figura, eje = plt.subplots(figsize=(11, 5))
    colores = ["#C44E52" if valor <= 0.777211 else "#4C72B0" for valor in tabla["corrected_cv_umbrales_v03"]]
    eje.barh(tabla["variante"], tabla["corrected_cv_umbrales_v03"], color=colores)
    eje.axvline(0.777211, color="#55A868", linestyle="--", label="V03 = 0.7772")
    eje.set_xlabel("corrected_cv con umbrales de V03")
    eje.set_title("Reweighting por dureza OOF (v4, TRACK B)")
    eje.legend()
    figura.tight_layout()
    FIGURA_PATH.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(FIGURA_PATH, dpi=150)
    plt.close(figura)


if __name__ == "__main__":
    main()
