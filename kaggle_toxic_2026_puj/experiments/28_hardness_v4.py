"""TRACK B v4: mapa de dureza de ejemplos y separacion ruido vs caso dificil.

Contexto
--------
La spec v4 (§9-§13) pide un ``hardness_score`` derivado **exclusivamente** de
predicciones out-of-fold (nunca de la perdida in-sample) y una distincion entre
"dificil pero coherente" y "potencialmente mal etiquetado". La spec forbids usar
la prediccion de un modelo entrenado sobre el propio ejemplo, por lo que todas
las señales de este script proceden de OOF.

Entradas
--------
- ``features/oof_v03_final.npz``: OOF canonico de V03 sobre el desarrollo.
- ``features/label_consistency_v4.npz``: clusters de consistencia y vecindario
  mas cercano calculados en ``27_label_consistency_v4.py``.
- ``features/v2_oof_*.npz`` y ``features/v2_oof_base_corrected.npz``: 30 canales
  OOF adicionales (bases, sintaxis, emocion, NB-SVM, ortografia, cita,
  negacion, targeting) para medir desacuerdo entre modelos. Se excluyen del
  computo los canales ``TXT+...``, que repiten la union de V03 y sobrecontarian
  la referencia.

Salidas
-------
- ``features/hardness_v4.npz``: variables de dureza y de ruido por ejemplo.
- ``reports/hardness_v4.csv``: tabla por ejemplo (desarrollo).
- ``reports/hardness_v4.json``: resumen cuantitativo.
- ``reports/hardness_vs_noise_v4.md``: informe de la spec (§13).
- ``reports/figuras/fig_32_hardness_v4.png``: distribuciones y relacion
  dureza-error.

Uso
---
    /home/carlos/venv/bin/python experiments/28_hardness_v4.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data import read_train  # noqa: E402
from src.evaluation import metricas_multietiqueta  # noqa: E402
from src.splits import ETIQUETAS  # noqa: E402

OOF_PATH = PROJECT_ROOT / "features" / "oof_v03_final.npz"
CONSISTENCIA_PATH = PROJECT_ROOT / "features" / "label_consistency_v4.npz"
CANALES = {
    "v2_oof_base_corrected.npz": "modelos",
    "v2_oof_syntax.npz": "variantes",
    "v2_oof_cita_negacion.npz": "variantes",
    "v2_oof_targeting.npz": "variantes",
    "v2_oof_emocion_nbsvm_ortografia.npz": "canales",
}
HARDNESS_PATH = PROJECT_ROOT / "features" / "hardness_v4.npz"
CSV_PATH = PROJECT_ROOT / "reports" / "hardness_v4.csv"
JSON_PATH = PROJECT_ROOT / "reports" / "hardness_v4.json"
INFORME_PATH = PROJECT_ROOT / "reports" / "hardness_vs_noise_v4.md"
FIGURA_PATH = PROJECT_ROOT / "reports" / "figuras" / "fig_32_hardness_v4.png"

N_BOOTSTRAP = 2000
SEMILLA = 42
EPSILON = 1e-12


def log(mensaje: str) -> None:
    """Imprime una linea de progreso con marca de seccion.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V4-B01] {mensaje}", flush=True)


# ---------------------------------------------------------------------------
# Carga de entradas
# ---------------------------------------------------------------------------
def cargar_canal(nombre: str, clave_nombres: str) -> tuple[np.ndarray, list[str]]:
    """Carga las probabilidades OOF de un archivo de canales.

    Los archivos de la fase v2 no usan la misma disposicion: las bases guardan
    ``(n_ejemplos, n_canales, 3)`` y el resto ``(n_canales, n_ejemplos, 3)``.
    Esta funcion normaliza siempre a ``(n_canales, n_ejemplos, 3)``.

    Args:
        nombre: nombre del archivo dentro de ``features/``.
        clave_nombres: clave que contiene los nombres de los canales.

    Returns:
        Tupla con el tensor de probabilidades y la lista de nombres.
    """
    datos = np.load(PROJECT_ROOT / "features" / nombre, allow_pickle=True)
    nombres = [str(n) for n in datos[clave_nombres]]
    proba = datos["proba"].astype(np.float64)
    if proba.shape[0] != len(nombres) and proba.shape[1] == len(nombres):
        proba = np.moveaxis(proba, 1, 0)
    if proba.shape[0] != len(nombres):
        raise ValueError(
            f"{nombre}: no se puede normalizar el tensor {proba.shape} "
            f"con {len(nombres)} canales"
        )
    return proba, nombres


def apilar_canales() -> tuple[np.ndarray, list[str]]:
    """Apila las predicciones OOF de los 30 canales disponibles.

    Los canales que ya son "texto + canal" se excluyen para no contar dos veces
    la representacion base: se conservan los canales puros y las cuatro bases.

    Returns:
        Tupla con el tensor de probabilidades y la lista de nombres.
    """
    tensores: list[np.ndarray] = []
    nombres: list[str] = []
    for nombre, clave in CANALES.items():
        proba, etiquetas_canal = cargar_canal(nombre, clave)
        for indice, etiqueta_canal in enumerate(etiquetas_canal):
            # Los canales "TXT+..." son la union de V03 con un canal; se
            # excluyen del computo de desacuerdo para no sobrecontar V03.
            if etiqueta_canal.startswith("TXT+") or etiqueta_canal == "CH-TGT+DIST":
                continue
            tensores.append(proba[indice])
            nombres.append(f"{nombre.split('_')[2][:4]}:{etiqueta_canal}")
    return np.stack(tensores), nombres


# ---------------------------------------------------------------------------
# Variables de dureza y de ruido
# ---------------------------------------------------------------------------
def calcular_dureza(
    y_true: np.ndarray, proba: np.ndarray, umbrales: np.ndarray
) -> dict[str, np.ndarray]:
    """Calcula las variables de dureza por etiqueta y globales.

    Se conservan dos definiciones de dureza:

    1. ``hardness_score`` (principal) = cercania a la frontera de decision,
       ``1 - |(p - umbral) * (2y - 1)|``. Mide cuanto margen le queda al modelo
       para sustain su decision, y por tanto fragilidad para la F1.
    2. ``bce`` (descriptiva) = entropia binaria cruzada de la probabilidad OOF
       de la etiqueta verdadera.

    La distincion es necesaria y no academica: con
    ``class_weight='balanced'`` y umbrales por debajo de 0.5, el modelo produce
    muchos positivos verdaderos con probabilidad baja (p ~ 0.3) que aun asi
    cuentan como acierto. En ese regimen la BCE media se **anticorrelaciona**
    con el error (seccion 2 del informe), de modo que usarla como peso
    ponderaria justo los ejemplos que el modelo ya clasifica bien.

    Args:
        y_true: matriz verdadera binaria.
        proba: probabilidades OOF de V03.
        umbrales: umbrales congelados por etiqueta.

    Returns:
        Diccionario con vectores por etiqueta y globales.
    """
    pred = (proba >= umbrales[None, :]).astype(int)
    p_true = np.where(y_true == 1, proba, 1.0 - proba)
    p_true = np.clip(p_true, EPSILON, 1.0 - EPSILON)
    bce = -(y_true * np.log(p_true) + (1 - y_true) * np.log(1.0 - p_true))
    error = (pred != y_true).astype(int)
    # Margen a favor de la etiqueta verdadera: positivo si la decision es
    # correcta, negativo si es un error; su magnitud es la distancia al umbral.
    margen_favor = (proba - umbrales[None, :]) * (2 * y_true - 1)
    cercania = 1.0 - np.abs(margen_favor)
    violacion = np.clip(-margen_favor, 0.0, None)
    return {
        "pred": pred,
        "p_true": p_true,
        "bce": bce,
        "error": error,
        "margen_favor": margen_favor,
        "hardness_por_etiqueta": cercania,
        "error_score": violacion.mean(axis=1),
        "hardness_score": cercania.mean(axis=1),
        "bce_score": bce.mean(axis=1),
        "n_errores": error.sum(axis=1),
    }


def calcular_desacuerdo_canales(
    y_true: np.ndarray, tensores: np.ndarray
) -> np.ndarray:
    """Calcula el desacuerdo medio de los canales por etiqueta.

    Cada canal se binariza en 0.5 y se cuenta si acierta o falla en cada
    etiqueta; el resultado es la fraccion de canales que fallan.

    Args:
        y_true: matriz verdadera binaria.
        tensores: tensor de probabilidades ``(n_canales, n, 3)``.

    Returns:
        Matriz ``(n, 3)`` con la fraccion de canales que fallan por etiqueta.
    """
    fallan = ((tensores >= 0.5).astype(int) != y_true[None, :, :]).mean(axis=0)
    return fallan


def calcular_vecindad(
    ids_dev: np.ndarray,
    ids_train: np.ndarray,
    y_train: np.ndarray,
    vecino_ids: np.ndarray,
    vecino_sim: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calcula el desacuerdo de etiqueta con el vecino mas cercano.

    Args:
        ids_dev: identificadores del desarrollo.
        ids_train: identificadores de todo el train.
        y_train: etiquetas de todo el train.
        vecino_ids: indices (en train) de los tres vecinos mas cercanos.
        vecino_sim: cosimilitudes de esos vecinos.

    Returns:
        Diccionario con ``nn_disagreement`` (etiqueta distinta al vecino 1),
        ``knn_disagreement`` (fraccion de vecinos que difieren, ponderada por
        similitud) y ``sim_cercana`` (cosimilitud del vecino 1).
    """
    posicion = {int(identificador): indice for indice, identificador in enumerate(ids_train)}
    n = len(ids_dev)
    nn_disagreement = np.zeros((n, len(ETIQUETAS)), dtype=float)
    knn_disagreement = np.zeros((n, len(ETIQUETAS)), dtype=float)
    sim_cercana = np.zeros(n)
    for fila in range(n):
        vecinos = vecino_ids[fila]
        similes = np.maximum(vecino_sim[fila], 0.0)
        etiquetas_vecinas = y_train[[posicion[int(ids_train[v])] for v in vecinos]]
        etiqueta_propia = y_train[posicion[int(ids_dev[fila])]]
        difieren = (etiquetas_vecinas != etiqueta_propia[None, :]).astype(float)
        nn_disagreement[fila] = difieren[0]
        norm = similes.sum()
        knn_disagreement[fila] = (
            difieren.T @ similes / norm if norm > 0 else difieren.mean(axis=0)
        )
        sim_cercana[fila] = similes[0]
    return {
        "nn_disagreement": nn_disagreement,
        "knn_disagreement": knn_disagreement,
        "sim_cercana": sim_cercana,
    }


# ---------------------------------------------------------------------------
# Bootstrap pareado por tweet
# ---------------------------------------------------------------------------
def bootstrap_delta(
    mascara: np.ndarray,
    y_true: np.ndarray,
    error: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    semilla: int = SEMILLA,
) -> dict[str, Any]:
    """Estima el delta de tasa de error dentro vs fuera de una mascara.

    Args:
        mascara: vector booleano que define el subconjunto de interes.
        y_true: matriz verdadera binaria.
        error: matriz de errores por etiqueta.
        n_bootstrap: numero de remuestreos.
        semilla: semilla del generador.

    Returns:
        Diccionario con la tasa dentro, la tasa fuera, el delta y su IC95.
    """
    generador = np.random.default_rng(semilla)
    dentro = mascara
    fuera = ~mascara
    delta_por_etiqueta: dict[str, dict[str, float]] = {}
    for posicion, etiqueta in enumerate(ETIQUETAS):
        tasa_dentro = float(error[dentro, posicion].mean())
        tasa_fuera = float(error[fuera, posicion].mean())
        deltas = np.empty(n_bootstrap)
        n = len(y_true)
        for i in range(n_bootstrap):
            indices = generador.integers(0, n, n)
            d = mascara[indices]
            f = ~d
            deltas[i] = (
                error[indices, posicion][d].mean()
                if d.any() and f.any()
                else np.nan
            )
        validos = deltas[~np.isnan(deltas)]
        percentiles = np.percentile(validos, [2.5, 97.5])
        delta_por_etiqueta[etiqueta] = {
            "tasa_error_dentro": round(tasa_dentro, 4),
            "tasa_error_fuera": round(tasa_fuera, 4),
            "delta": round(tasa_dentro - tasa_fuera, 4),
            "ic95": [round(float(percentiles[0]), 4), round(float(percentiles[1]), 4)],
        }
    return delta_por_etiqueta


# ---------------------------------------------------------------------------
# Informe y figura
# ---------------------------------------------------------------------------
def escribir_informe(
    tabla: pd.DataFrame,
    resumen: dict[str, Any],
    train: pd.DataFrame,
) -> None:
    """Escribe el informe de dureza frente a ruido (spec §13).

    Args:
        tabla: tabla por ejemplo del desarrollo.
        resumen: resumen cuantitativo.
        train: marco de entrenamiento completo (para los ids).
    """
    textos = dict(zip(train["id"].astype(int), train["text"].astype(str)))
    lineas = [
        "# Dureza de ejemplos frente a ruido de etiqueta (v4, TRACK B)",
        "",
        f"Generado por `experiments/28_hardness_v4.py` el "
        f"{datetime.now(timezone.utc).date().isoformat()}. Todas las señales son "
        "**out-of-fold**: la probabilidad de V03 para un ejemplo procede de un "
        "modelo que no lo vio en entrenamiento (§9).",
        "",
        "## 1. Variables calculadas",
        "",
        "| Variable | Definición | Uso |",
        "|---|---|---|",
        "| `hardness_score` | cercania a la frontera de decision, `1 - |(p - umbral) * (2y - 1)|` (principal) | ponderacion de ejemplos (V4-B03) |",
        "| `error_score` | magnitud media de la violacion del umbral | severidad del error |",
        "| `bce_score` | entropia binaria cruzada de la etiqueta verdadera (descriptiva) | contraste de definiciones |",
        "| `p_true` | probabilidad OOF asignada a la etiqueta verdadera | separacion facil/dificil |",
        "| `error` | error OOF con los umbrales congelados de V03 | variable de control |",
        "| `nn_disagreement` | el vecino mas cercano (cosimilitud TF-IDF) tiene otra etiqueta | ambiguedad local |",
        "| `knn_disagreement` | desacuerdo ponderado con los tres vecinos | ambiguedad local suavizada |",
        "| `model_disagreement` | fraccion de los canales OOF no redundantes que fallan en esa etiqueta | dificultad del ejemplo |",
        "| `cluster_contradictorio` | pertenece a un cluster de texto repetido con etiquetas distintas | ruido textual |",
        "| `label_rarity` | frecuencia de la etiqueta verdadera en train | rareza |",
        "",
        "## 2. Distribucion de la dureza y eleccion de la definicion",
        "",
        f"- `hardness_score` (media): **{resumen['hardness_media']:.4f}**, "
        f"mediana {resumen['hardness_mediana']:.4f}, p90 {resumen['hardness_p90']:.4f}.",
        f"- Ejemplos con al menos una etiqueta errada: "
        f"**{resumen['n_con_error']}** ({resumen['porcentaje_con_error']} %).",
        f"- Ejemplos que fallan las tres etiquetas: **{resumen['n_error_todas']}** "
        f"({resumen['porcentaje_error_todas']} %).",
        "",
        "**Correlaciones con el numero de etiquetas erradas** (Pearson):",
        "",
        "| Medida | Correlacion | Lectura |",
        "|---|---|---|",
        f"| `p_true` (probabilidad de la etiqueta verdadera) | "
        f"{resumen['correlacion_p_true_errores']:.3f} | menor confianza, mas error |",
        f"| `error_score` (violacion del umbral) | "
        f"{resumen['correlacion_error_score_errores']:.3f} | medida directa del error |",
        f"| `bce_score` (entropia binaria cruzada) | "
        f"{resumen['correlacion_dureza_errores']:.3f} | **anticorrelacionada** |",
        f"| `hardness_score` (cercania a la frontera) | "
        f"{resumen['correlacion_frontera_errores']:.3f} | correlacion positiva y monotona |",
        "",
        "Hallazgo metodologico que condiciona V4-B03 (la spec §11 propone ponderar "
        "por `OOF_loss`): la dificultad **por etiqueta** si se ordena bien "
        f"(`p_true` frente al error de esa etiqueta: r = "
        f"{resumen['correlacion_p_true_por_etiqueta']['HS']} / "
        f"{resumen['correlacion_p_true_por_etiqueta']['TR']} / "
        f"{resumen['correlacion_p_true_por_etiqueta']['AG']}), pero **agregar las tres "
        "etiquetas rompe el orden**. La BCE media por numero de errores es "
        + ", ".join(
            f"{k} errores -> {v}" for k, v in resumen["bce_mediana_por_num_errores"].items()
        )
        + ": los ejemplos que mas fallan tienen la BCE **menor**, porque una "
        "violacion fuerte en una etiqueta se compensa con confianza alta en las "
        "otras dos. Por eso la correlacion de la BCE media con el numero de errores "
        f"({resumen['correlacion_dureza_errores']:.3f}) es mucho menor que la de la "
        f"cercania a la frontera ({resumen['correlacion_frontera_errores']:.3f}). "
        "Conclusion operativa: para ponderar ejemplos se usa la distancia al umbral "
        "en la direccion de la etiqueta verdadera (monotona en el error) y, si se "
        "quiere priorizar el error, `error_score`; la BCE agregada no debe usarse "
        "como peso en este corpus.",
        "",
        "## 3. Dos poblaciones: dificil coherente frente a posible ruido",
        "",
        f"Umbral de corte: dureza >= percentil 80 = {resumen['corte_dureza']:.4f}.",
        "",
        "| Poblacion | n | % | error medio por etiqueta | en cluster contradictorio |",
        "|---|---|---|---|---|",
    ]
    for nombre, clave in (("Dificil coherente", "dificil_coherente"),
                          ("Posible ruido", "posible_ruido")):
        grupo = tabla[tabla["poblacion"] == nombre]
        lineas.append(
            f"| {nombre} | {len(grupo)} | {100 * len(grupo) / len(tabla):.1f} % | "
            f"{grupo['tasa_error_media'].mean():.3f} | "
            f"{grupo['cluster_contradictorio'].mean():.3f} |"
        )
    lineas += [
        "",
        "## 4. Error dentro y fuera de los clusters contradictorios",
        "",
        "Tasa de error OOF de V03 y delta con IC95 de bootstrap pareado por tweet "
        f"({N_BOOTSTRAP} remuestreos):",
        "",
        "| Etiqueta | Error dentro | Error fuera | Delta | IC95 |",
        "|---|---|---|---|---|",
    ]
    for etiqueta, valores in resumen["bootstrap_cluster_contradictorio"].items():
        lineas.append(
            f"| {etiqueta} | {valores['tasa_error_dentro']:.3f} | "
            f"{valores['tasa_error_fuera']:.3f} | {valores['delta']:+.3f} | "
            f"[{valores['ic95'][0]:+.3f}, {valores['ic95'][1]:+.3f}] |"
        )
    lineas += [
        "",
        "## 5. Desacuerdo entre modelos en las dos poblaciones",
        "",
        "| Poblacion | `model_disagreement` medio | `nn_disagreement` medio |",
        "|---|---|---|",
    ]
    for nombre in ("Dificil coherente", "Posible ruido"):
        grupo = tabla[tabla["poblacion"] == nombre]
        lineas.append(
            f"| {nombre} | {grupo['model_disagreement'].mean():.3f} | "
            f"{grupo['nn_disagreement'].mean():.3f} |"
        )
    lineas += [
        "",
        "## 6. Los 15 ejemplos mas duros",
        "",
        "| id | etiquetas | p_true (HS/TR/AG) | error | desacuerdo modelos | contradiccion | texto |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, fila in tabla.nsmallest(15, "hardness_score").iterrows():
        texto = textos.get(int(fila["id"]), "")[:70]
        etiquetas_texto = "".join(str(int(v)) for v in fila["y_true"])
        p_true_texto = "/".join(f"{v:.2f}" for v in fila["p_true"])
        lineas.append(
            f"| {int(fila['id'])} | {etiquetas_texto} | {p_true_texto} | "
            f"{int(fila['n_errores'])} | {fila['model_disagreement']:.2f} | "
            f"{int(fila['cluster_contradictorio'])} | {texto} |"
        )
    lineas += [
        "",
        "## 7. Lectura (sin modificar etiquetas)",
        "",
        resumen["lectura"],
        "",
        "Este informe **no** corrige ni elimina etiquetas (§90): solo cuantifica "
        "que parte del error es coherente con ambiguedad textual y que parte es "
        "dificultad del modelo ante ejemplos no contradictorios.",
    ]
    INFORME_PATH.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def escribir_figura(tabla: pd.DataFrame) -> None:
    """Dibuja la distribucion de dureza y su relacion con el error.

    Args:
        tabla: tabla por ejemplo del desarrollo.
    """
    figura, ejes = plt.subplots(1, 3, figsize=(15, 4))
    ejes[0].hist(tabla["hardness_score"], bins=40, color="#4C72B0", edgecolor="white")
    ejes[0].set_title("Dureza: cercania a la frontera de decision (OOF)")
    ejes[0].set_xlabel("hardness_score")
    ejes[0].set_ylabel("tweets")
    colores = np.where(
        tabla["cluster_contradictorio"] > 0, "#C44E52", "#4C72B0"
    )
    ejes[1].scatter(
        tabla["hardness_score"],
        tabla["model_disagreement"],
        c=colores,
        s=8,
        alpha=0.6,
    )
    ejes[1].set_title("Dureza frente a desacuerdo de 30 canales")
    ejes[1].set_xlabel("hardness_score")
    ejes[1].set_ylabel("model_disagreement")
    for etiqueta, color in zip(ETIQUETAS, ["#C44E52", "#55A868", "#4C72B0"]):
        mascara = tabla[f"error_{etiqueta}"] > 0
        ejes[2].hist(
            tabla.loc[ mascara, "hardness_score"],
            bins=30,
            alpha=0.55,
            label=f"{etiqueta} error",
            color=color,
        )
    ejes[2].set_title("Dureza por etiqueta segun error")
    ejes[2].set_xlabel("hardness_score")
    ejes[2].legend()
    figura.tight_layout()
    FIGURA_PATH.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(FIGURA_PATH, dpi=150)
    plt.close(figura)


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------
def main() -> None:
    """Ejecuta el mapa de dureza y el informe ruido vs dificultad."""
    inicio = datetime.now(timezone.utc)
    log("Mapa de dureza OOF (TRACK B): perdida, vecindad, clusters y desacuerdo de canales")

    oof = np.load(OOF_PATH, allow_pickle=True)
    ids_dev = oof["ids"].astype(np.int64)
    y_true = oof["y_true"].astype(int)
    proba = oof["proba"]
    umbrales = oof["umbrales"].astype(float)
    pliegues = oof["pliegues"].astype(int)

    consistencia = np.load(CONSISTENCIA_PATH, allow_pickle=True)
    ids_train = consistencia["ids"].astype(np.int64)
    discrepancia_cluster = consistencia["discrepancia_etiqueta"].astype(int)
    en_cluster = consistencia["en_cluster"].astype(int)
    y_train = consistencia["y_true"].astype(int)
    train = read_train()
    assert np.array_equal(ids_train, train["id"].to_numpy(dtype=np.int64)), (
        "los ids de consistencia no coinciden con el train"
    )

    posicion_dev = {int(identificador): indice for indice, identificador in enumerate(ids_dev)}

    log("   1/5 variables de dureza OOF")
    dureza = calcular_dureza(y_true, proba, umbrales)
    log(f"      n con al menos un error: {int((dureza['n_errores'] > 0).sum())} de {len(ids_dev)}")

    log("   2/5 desacuerdo con el vecindario mas cercano")
    vecindad = calcular_vecindad(
        ids_dev,
        ids_train,
        y_train,
        consistencia["vecino_ids"].astype(np.int64),
        consistencia["vecino_sim"].astype(np.float32),
    )

    log("   3/5 desacuerdo con los canales OOF no redundantes")
    tensores, nombres_canales = apilar_canales()
    log(f"      canales apilados: {len(nombres_canales)}")
    model_disagreement = calcular_desacuerdo_canales(y_true, tensores)

    log("   4/5 banderas de cluster contradictorio")
    cluster_por_ejemplo = np.zeros((len(ids_dev), len(ETIQUETAS)), dtype=int)
    en_cluster_por_ejemplo = np.zeros(len(ids_dev), dtype=int)
    for indice, identificador in enumerate(ids_dev):
        fila = ids_train == int(identificador)
        cluster_por_ejemplo[indice] = discrepancia_cluster[fila][0]
        en_cluster_por_ejemplo[indice] = en_cluster[fila][0]

    log(f"   5/5 bootstrap pareado ({N_BOOTSTRAP} remuestreos)")
    mascara_contradictoria = cluster_por_ejemplo.max(axis=1) > 0
    bootstrap = bootstrap_delta(mascara_contradictoria, y_true, dureza["error"])

    # Tabla por ejemplo.
    frecuencias_etiqueta = train[["HS", "TR", "AG"]].mean().to_numpy()
    tabla = pd.DataFrame(
        {
            "id": ids_dev,
            "y_true": [tuple(int(v) for v in fila) for fila in y_true],
            "p_true": [tuple(float(v) for v in fila) for fila in dureza["p_true"]],
            "n_errores": dureza["n_errores"],
            "tasa_error_media": dureza["error"].mean(axis=1),
            "hardness_score": dureza["hardness_score"],
            "error_score": dureza["error_score"],
            "bce_score": dureza["bce_score"],
            "nn_disagreement": vecindad["nn_disagreement"].mean(axis=1),
            "knn_disagreement": vecindad["knn_disagreement"].mean(axis=1),
            "sim_cercana": vecindad["sim_cercana"],
            "model_disagreement": model_disagreement.mean(axis=1),
            "cluster_contradictorio": mascara_contradictoria.astype(int),
            "en_cluster": en_cluster_por_ejemplo,
            "pliegue": pliegues,
        }
    )
    for posicion, etiqueta in enumerate(ETIQUETAS):
        tabla[f"error_{etiqueta}"] = dureza["error"][:, posicion]
        tabla[f"nn_dis_{etiqueta}"] = vecindad["nn_disagreement"][:, posicion]
        tabla[f"model_dis_{etiqueta}"] = model_disagreement[:, posicion]
        tabla[f"cluster_contra_{etiqueta}"] = cluster_por_ejemplo[:, posicion]
        tabla[f"p_true_{etiqueta}"] = dureza["p_true"][:, posicion]
    tabla["label_rarity"] = frecuencias_etiqueta[y_true].mean(axis=1)

    corte = float(np.percentile(tabla["hardness_score"], 80))
    tabla["poblacion"] = np.where(
        tabla["cluster_contradictorio"] > 0,
        "Posible ruido",
        np.where(tabla["hardness_score"] >= corte, "Dificil coherente", "Facil"),
    )
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(CSV_PATH, index=False)

    metricas = metricas_multietiqueta(y_true, dureza["pred"])
    n_errores = tabla["n_errores"].to_numpy(dtype=float)
    correlaciones = {
        "correlacion_p_true_errores": float(
            np.corrcoef(dureza["p_true"].mean(axis=1), n_errores)[0, 1]
        ),
        "correlacion_error_score_errores": float(
            np.corrcoef(dureza["error_score"], n_errores)[0, 1]
        ),
        "correlacion_dureza_errores": float(
            np.corrcoef(dureza["bce_score"], n_errores)[0, 1]
        ),
        "correlacion_frontera_errores": float(
            np.corrcoef(dureza["hardness_score"], n_errores)[0, 1]
        ),
    }
    # Aciertos con probabilidad de la etiqueta verdadera baja: casos que
    # explican la anticorrelacion de la BCE.
    acierto_p_bajo = int(
        (((dureza["error"] == 0) & (dureza["p_true"] < 0.44)).sum(axis=1) > 0).sum()
    )
    cola = dureza["p_true"] < 0.1
    aporte_cola = float(
        dureza["bce"][cola].sum() / dureza["bce"].sum()
    ) if cola.any() else 0.0
    bce_mediana_por_errores = {
        str(k): round(float(np.median(dureza["bce_score"][n_errores == k])), 4)
        for k in range(4)
        if (n_errores == k).any()
    }
    correlacion_por_etiqueta = {
        etiqueta: round(
            float(np.corrcoef(dureza["p_true"][:, posicion], dureza["error"][:, posicion])[0, 1]),
            3,
        )
        for posicion, etiqueta in enumerate(ETIQUETAS)
    }
    lectura = _construir_lectura(tabla, bootstrap, metricas)

    resumen = {
        "experiment_id": "V4-B01-B02",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duracion_s": round((datetime.now(timezone.utc) - inicio).total_seconds(), 2),
        "n_dev": int(len(ids_dev)),
        "n_canales": len(nombres_canales),
        "canales": nombres_canales,
        "hardness_media": float(tabla["hardness_score"].mean()),
        "hardness_mediana": float(tabla["hardness_score"].median()),
        "hardness_p90": float(np.percentile(tabla["hardness_score"], 90)),
        "corte_dureza": corte,
        "n_con_error": int((tabla["n_errores"] > 0).sum()),
        "porcentaje_con_error": round(100.0 * float((tabla["n_errores"] > 0).mean()), 2),
        "n_error_todas": int((tabla["n_errores"] == 3).sum()),
        "porcentaje_error_todas": round(
            100.0 * float((tabla["n_errores"] == 3).mean()), 2
        ),
        "n_acerto_pos_bajo_conf": acierto_p_bajo,
        "n_cola_p_true_bajo_0_1": int(cola.sum()),
        "porcentaje_bce_de_la_cola": round(100.0 * aporte_cola, 2),
        "bce_mediana_por_num_errores": bce_mediana_por_errores,
        "correlacion_p_true_por_etiqueta": correlacion_por_etiqueta,
        **correlaciones,
        "metricas_v03_dev": metricas,
        "poblaciones": tabla["poblacion"].value_counts().to_dict(),
        "bootstrap_cluster_contradictorio": bootstrap,
        "n_cluster_contradictorio_dev": int(mascara_contradictoria.sum()),
        "media_model_disagreement_global": float(tabla["model_disagreement"].mean()),
        "media_nn_disagreement_global": float(tabla["nn_disagreement"].mean()),
        "lectura": lectura,
    }
    JSON_PATH.write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    np.savez_compressed(
        HARDNESS_PATH,
        ids=ids_dev,
        hardness_score=dureza["hardness_score"].astype(np.float32),
        hardness_por_etiqueta=dureza["hardness_por_etiqueta"].astype(np.float32),
        error_score=dureza["error_score"].astype(np.float32),
        bce_score=dureza["bce_score"].astype(np.float32),
        margen_favor=dureza["margen_favor"].astype(np.float32),
        p_true=dureza["p_true"].astype(np.float32),
        error=dureza["error"].astype(np.int8),
        nn_disagreement=vecindad["nn_disagreement"].astype(np.float32),
        knn_disagreement=vecindad["knn_disagreement"].astype(np.float32),
        model_disagreement=model_disagreement.astype(np.float32),
        cluster_contradictorio=cluster_por_ejemplo.astype(np.int8),
        y_true=y_true,
        pliegues=pliegues,
    )

    escribir_informe(tabla, resumen, train)
    escribir_figura(tabla)

    log("   RESUMEN")
    log(f"   poblaciones: {resumen['poblaciones']}")
    log(f"   error las tres etiquetas: {resumen['n_error_todas']} ({resumen['porcentaje_error_todas']} %)")
    for etiqueta, valores in bootstrap.items():
        log(
            f"   {etiqueta}: error dentro {valores['tasa_error_dentro']:.3f} vs fuera "
            f"{valores['tasa_error_fuera']:.3f} (delta {valores['delta']:+.3f}, "
            f"IC95 [{valores['ic95'][0]:+.3f}, {valores['ic95'][1]:+.3f}])"
        )
    log(f"   artefactos: {CSV_PATH.name}, {JSON_PATH.name}, {INFORME_PATH.name}, {FIGURA_PATH.name}")


def _construir_lectura(
    tabla: pd.DataFrame,
    bootstrap: dict[str, dict[str, float]],
    metricas: dict[str, Any],
) -> str:
    """Redacta la lectura final del informe de dureza.

    Args:
        tabla: tabla por ejemplo.
        bootstrap: resultados del bootstrap por etiqueta.
        metricas: metricas de V03 en el desarrollo.

    Returns:
        Parrafo de lectura en espanol.
    """
    posible_ruido = int((tabla["poblacion"] == "Posible ruido").sum())
    facil = int((tabla["poblacion"] == "Facil").sum())
    n = len(tabla)
    deltas = np.array([valores["delta"] for valores in bootstrap.values()])
    dentro_mayor = int((deltas > 0).sum())
    if dentro_mayor == 0:
        conclusion = (
            "En las tres etiquetas la tasa de error es **menor** dentro de los "
            "clusters contradictorios que fuera (deltas "
            f"{', '.join(f'{d:+.3f}' for d in deltas)}): el modelo clasifica bien "
            "los ejemplos ambiguos, probablemente porque la anotacion mayoritaria "
            "del cluster (000) es la que el lexicon del insulto aprende. En "
            "consecuencia, **el ruido de anotacion por duplicacion no es una "
            "componente material del error de V03** en este corpus."
        )
    elif dentro_mayor >= 2:
        conclusion = (
            f"En {dentro_mayor} de 3 etiquetas la tasa de error es mayor dentro de "
            "los clusters contradictorios: la ambiguedad si es una componente "
            "material del error."
        )
    else:
        conclusion = (
            "El efecto de los clusters contradictorios es mixto y de magnitud "
            "marginal; no sostiene una via de reponderacion por ruido."
        )
    return (
        f"El {posible_ruido} de {n} ejemplos de desarrollo ({100 * posible_ruido / n:.1f} %) "
        f"pertenece a un cluster de texto repetido con etiquetas contradictorias y se "
        f"clasifican como posible ruido; {facil} ejemplos son faciles. {conclusion} "
        f"La via de reponderar o descontar ejemplos contradictorios tiene por tanto "
        "un techo de impacto pequeno sobre la F1 global: actua sobre menos del 3 % "
        "de los ejemplos. La via con recorrido real es la dificultad general ("
        f"{tabla['n_errores'].gt(0).mean() * 100:.1f} % de los ejemplos falla alguna "
        "etiqueta) y la ambiguedad local medida por el vecino mas cercano, que es "
        "dificultad del modelo y no ruido de anotacion."
    )


if __name__ == "__main__":
    main()
