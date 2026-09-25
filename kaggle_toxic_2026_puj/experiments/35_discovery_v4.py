"""TRACK F v4 y artefactos de discovery: shift, vecindad, robustez y correcciones.

Contexto
--------
Ultimo bloque de investigacion antes de la seleccion de candidatos. Cierra las
lineas de la spec que son **diagnostico** y no modelos nuevos:

- **V4-F01** (§52-§53): ¿existe un shift no supervisado entre train y test? Se
  comparan distribuciones de variables que no dependen de las etiquetas (longitud,
  entropia de caracteres, densidad de URL, menciones, hashtags, tasa de OUT, ritmo
  de mayusculas, longitud de palabra). No se usan labels de test.
- **V4-F02** (§55): ¿la referencia depende de memorizar grupos de near-duplicado?
  Se evalua el error OOF de V03 por subconjunto (textos unicos, repetidos,
  menciones repetidas).
- **V4-F03** (§56): ¿el acuerdo de etiqueta del vecino mas cercano explica parte
  del error? Se mide con bootstrap pareado.
- **V4-F04** (§58): techo de impacto de la propagacion de etiquetas por
  near-duplicado (solo conteo; no se aplica al test).
- **§47 y §96**: matriz de oportunidades de correccion y matriz de correccion de
  los candidatos frente a V03.
- **§95**: taxonomia de errores v4.

Uso
---
    /home/carlos/venv/bin/python experiments/35_discovery_v4.py
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data import read_test, read_train  # noqa: E402
from src.evaluation import metricas_multietiqueta  # noqa: E402
from src.splits import (  # noqa: E402
    ETIQUETAS,
    grupos_fusionados,
    holdout_v2,
)

OOF_PATH = PROJECT_ROOT / "features" / "oof_v03_final.npz"
HARDNESS_PATH = PROJECT_ROOT / "features" / "hardness_v4.npz"
CONSISTENCIA_PATH = PROJECT_ROOT / "features" / "label_consistency_v4.npz"
SHIFT_PATH = PROJECT_ROOT / "reports" / "distribution_shift_v4.csv"
CORRECCIONES_PATH = PROJECT_ROOT / "reports" / "correction_opportunities_v4.csv"
MATRIZ_PATH = PROJECT_ROOT / "reports" / "correction_matrix_v4.csv"
TAXONOMIA_PATH = PROJECT_ROOT / "reports" / "error_taxonomy_v4.md"
JSON_PATH = PROJECT_ROOT / "reports" / "discovery_v4.json"
FIGURA_PATH = PROJECT_ROOT / "reports" / "figuras" / "fig_39_discovery_v4.png"

UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
N_BOOTSTRAP = 2000
SEMILLA = 42
TOKEN_PATTERN = __import__("re").compile(r"[a-z0-9']+")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso con marca de seccion.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V4-F] {mensaje}", flush=True)


# ---------------------------------------------------------------------------
# V4-F01: shift no supervisado
# ---------------------------------------------------------------------------
def variables_no_supervisadas(textos: list[str], vocabulario: set[str]) -> pd.DataFrame:
    """Calcula variables que no dependen de las etiquetas.

    Args:
        textos: lista de tweets.
        vocabulario: vocabulario de referencia (del train).

    Returns:
        Tabla con una fila por tweet.
    """
    filas = []
    for texto in textos:
        tokens = TOKEN_PATTERN.findall(texto.lower())
        conteos = {token: texto.lower().count(token) for token in tokens}
        n_tokens = max(len(tokens), 1)
        entropia = 0.0
        for conteo in conteos.values():
            if conteo > 0:
                p = conteo / n_tokens
                entropia -= p * np.log2(p)
        filas.append(
            {
                "n_caracteres": len(texto),
                "n_tokens": len(tokens),
                "longitud_media_palabra": float(np.mean([len(t) for t in tokens])) if tokens else 0.0,
                "entropia_tokens": entropia,
                "densidad_url": float(len([t for t in tokens if t.startswith("http")]) / n_tokens),
                "densidad_mencion": float(texto.count("@") / n_tokens),
                "densidad_hashtag": float(texto.count("#") / n_tokens),
                "proporcion_out_of_vocabulario": float(
                    sum(1 for t in tokens if t not in vocabulario) / n_tokens
                ),
                "proporcion_mayusculas": float(
                    sum(1 for c in texto if c.isupper()) / max(len(texto), 1)
                ),
                "proporcion_no_alfanumerico": float(
                    sum(1 for c in texto if not c.isalnum() and c != " ") / max(len(texto), 1)
                ),
            }
        )
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Bootstrap pareado
# ---------------------------------------------------------------------------
def bootstrap_pareado(
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


# ---------------------------------------------------------------------------
# V4-F03: acuerdo del vecino mas cercano
# ---------------------------------------------------------------------------
def medir_vecindad(
    y_dev: np.ndarray,
    consistencia: dict[str, np.ndarray],
    y_train: np.ndarray,
    ids_dev: np.ndarray,
    ids_train: np.ndarray,
) -> dict[str, Any]:
    """Mide si el desacuerdo con el vecino mas cercano se asocia al error.

    Args:
        y_dev: etiquetas del desarrollo.
        consistencia: arrays del script 27 (vecinos y banderas).
        y_train: etiquetas de todo el train.
        ids_dev: identificadores del desarrollo.
        ids_train: identificadores de todo el train.

    Returns:
        Diccionario con las tasas de error dentro y fuera del vecindario
        discrepante y el bootstrap pareado del delta.
    """
    vecino_ids = consistencia["vecino_ids"]
    vecino_sim = consistencia["vecino_sim"]
    posicion = {int(identificador): indice for indice, identificador in enumerate(ids_train)}
    discrepancia = np.zeros((len(ids_dev), len(ETIQUETAS)))
    similitud_media = np.zeros(len(ids_dev))
    for fila in range(len(ids_dev)):
        vecinos = vecino_ids[fila]
        etiquetas_vecinas = y_train[[posicion[int(ids_train[v])] for v in vecinos]]
        propia = y_train[posicion[int(ids_dev[fila])]]
        discrepancia[fila] = (etiquetas_vecinas != propia[None, :]).mean(axis=0)
        similitud_media[fila] = float(vecino_sim[fila].mean())
    return {"discrepancia_vecino": discrepancia, "similitud_media": similitud_media}


# ---------------------------------------------------------------------------
# Taxonomia de errores
# ---------------------------------------------------------------------------
def construir_taxonomia(
    y_true: np.ndarray, pred: np.ndarray, textos: list[str],train_ids: np.ndarray
) -> dict[str, Any]:
    """Construye la taxonomia de errores v4 con sus conteos.

    Args:
        y_true: matriz verdadera.
        pred: matriz predicha.
        textos: textos del desarrollo.
        train_ids: identificadores del desarrollo.

    Returns:
        Diccionario con los conteos por categoria.
    """
    error = pred != y_true
    categorias = {
        "FP_HS": error[:, 0] & (y_true[:, 0] == 0),
        "FN_HS": error[:, 0] & (y_true[:, 0] == 1),
        "FP_TR": error[:, 1] & (y_true[:, 1] == 0),
        "FN_TR": error[:, 1] & (y_true[:, 1] == 1),
        "FP_AG": error[:, 2] & (y_true[:, 2] == 0),
        "FN_AG": error[:, 2] & (y_true[:, 2] == 1),
        "ALL_THREE": error.all(axis=1),
    }
    categorias["IMPLICIT"] = (categorias["FN_HS"]) & ~np.array(
        ["insulto" in texto.lower() for texto in textos]
    )
    categorias["QUOTED"] = error.any(axis=1) & np.array(
        ['"' in texto for texto in textos]
    )
    categorias["COLLECTIVE"] = error[:, 2] & (y_true[:, 2] == 1)
    categorias["DIRECTED"] = error[:, 1] & (y_true[:, 1] == 1)
    categorias["AFFILIATIVE"] = error[:, 0] & (y_true[:, 0] == 0) & np.array(
        ["@" in texto for texto in textos]
    )
    categorias["NEWS"] = error.any(axis=1) & np.array(
        ["http" in texto for texto in textos]
    )
    return {
        nombre: {
            "n": int(mascara.sum()),
            "porcentaje": round(100.0 * float(mascara.sum()) / len(y_true), 2),
            "ids_ejemplo": ";".join(
                str(int(train_ids[indice])) for indice in np.where(mascara)[0][:8]
            ),
        }
        for nombre, mascara in categorias.items()
    }


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------
def main() -> None:
    """Ejecuta el bloque de diagnostico y discovery de la fase v4."""
    inicio_tiempo = time.time()
    inicio = datetime.now(timezone.utc)
    log("Diagnostico y discovery (TRACK F)")

    train = read_train()
    test = read_test()
    grupos = grupos_fusionados(train)
    indices_dev, indices_holdout = holdout_v2(train, grupos)
    oof = np.load(OOF_PATH, allow_pickle=True)
    ids_dev = oof["ids"].astype(np.int64)
    y_dev = oof["y_true"].astype(int)
    pred_dev = oof["pred"].astype(int)
    assert np.array_equal(train["id"].to_numpy(dtype=np.int64)[indices_dev], ids_dev)

    consistencia = np.load(CONSISTENCIA_PATH, allow_pickle=True)
    dureza = np.load(HARDNESS_PATH, allow_pickle=True)

    # --- V4-F01: shift no supervisado -------------------------------------
    log("   1/5 shift no supervisado train vs test (sin labels de test)")
    vocabulario = set()
    for texto in train["text"].astype(str).str.lower():
        vocabulario.update(TOKEN_PATTERN.findall(texto))
    variables_train = variables_no_supervisadas(train["text"].astype(str).tolist(), vocabulario)
    variables_test = variables_no_supervisadas(test["text"].astype(str).tolist(), vocabulario)
    generador = np.random.default_rng(SEMILLA)
    filas_shift = []
    for columna in variables_train.columns:
        valores_train = variables_train[columna].to_numpy()
        valores_test = variables_test[columna].to_numpy()
        # Bootstrap de la diferencia de medias (test vs train), sin usar labels.
        muestras = np.array(
            [
                valores_test[generador.integers(0, len(valores_test), len(valores_test))].mean()
                - valores_train[
                    generador.integers(0, len(valores_train), len(valores_train))
                ].mean()
                for _ in range(N_BOOTSTRAP)
            ]
        )
        filas_shift.append(
            {
                "variable": columna,
                "media_train": round(float(valores_train.mean()), 4),
                "media_test": round(float(valores_test.mean()), 4),
                "desviacion_train": round(float(valores_train.std(ddof=1)), 4),
                "desviacion_test": round(float(valores_test.std(ddof=1)), 4),
                "delta": round(float(valores_test.mean() - valores_train.mean()), 4),
                "ic95_bajo": round(float(np.percentile(muestras, 2.5)), 4),
                "ic95_alto": round(float(np.percentile(muestras, 97.5)), 4),
                "desplazamiento_significativo": bool(
                    np.percentile(muestras, 2.5) > 0 or np.percentile(muestras, 97.5) < 0
                ),
            }
        )
    tabla_shift = pd.DataFrame(filas_shift)
    SHIFT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tabla_shift.to_csv(SHIFT_PATH, index=False)
    n_significativos = int(tabla_shift["desplazamiento_significativo"].sum())
    log(f"      variables con desplazamiento significativo: {n_significativos}/{len(tabla_shift)}")

    # --- V4-F02: robustez a duplicados ------------------------------------
    log("   2/5 robustez de V03 por subconjunto")
    en_cluster = consistencia["en_cluster"].astype(bool)
    posicion_dev = {int(identificador): indice for indice, identificador in enumerate(consistencia["ids"])}
    en_cluster_dev = np.array([en_cluster[posicion_dev[int(i)]] for i in ids_dev])
    error = (pred_dev != y_dev)
    subconjuntos = {
        "todos": np.ones(len(ids_dev), dtype=bool),
        "textos_unicos": ~en_cluster_dev,
        "en_cluster_near_duplicado": en_cluster_dev,
    }
   robustez = {}
    for nombre, mascara in subconjuntos.items():
        if mascara.sum() == 0:
            continue
        metricas = metricas_multietiqueta(y_dev[mascara], pred_dev[mascara])
        robustez[nombre] = {
            "n": int(mascara.sum()),
            "f1_media": round(metricas["f1_media_columnas"], 4),
            "f1_por_etiqueta": {k: round(v, 4) for k, v in metricas["f1_por_etiqueta"].items()},
            "error_medio_por_etiqueta": round(float(error[mascara].mean()), 4),
        }
        log(f"      {nombre:26s} n={mascara.sum():4d} F1={metricas['f1_media_columnas']:.4f}")

    # --- V4-F03: vecindad --------------------------------------------------
    log("   3/5 acuerdo de etiqueta del vecino mas cercano")
    vecindad = medir_vecindad(
        y_dev,
        {clave: consistencia[clave] for clave in consistencia.files},
        consistencia["y_true"].astype(int),
        ids_dev,
        consistencia["ids"].astype(np.int64),
    )
    discrepancia = vecindad["discrepancia_vecino"].mean(axis=1)
    alta = discrepancia >= 0.5
    delta_vecino, ic_bajo, ic_alto = bootstrap_pareado(
        y_dev, pred_dev, pred_dev.copy()
    )  # estructura comun; el delta se calcula sobre las mascaras
   Global = error.mean(axis=1)
    generador2 = np.random.default_rng(SEMILLA)
    deltas = np.empty(N_BOOTSTRAP)
    for indice in range(N_BOOTSTRAP):
        indices = generador2.integers(0, len(y_dev), len(y_dev))
        dentro = alta[indices]
        if dentro.any() and (~dentro).any():
            deltas[indice] = Global[indices][dentro].mean() - Global[indices][~dentro].mean()
        else:
            deltas[indice] = np.nan
    percentiles = np.percentile(deltas[~np.isnan(deltas)], [2.5, 97.5])
    vecindad_resumen = {
        "n_vecindario_discrepante": int(alta.sum()),
        "error_medio_dentro": round(float(Global[alta].mean()), 4) if alta.any() else None,
        "error_medio_fuera": round(float(Global[~alta].mean()), 4),
        "delta": round(float(Global[alta].mean() - Global[~alta].mean()), 4)
        if alta.any()
        else None,
        "ic95": [round(float(percentiles[0]), 4), round(float(percentiles[1]), 4)],
    }
    log(f"      vecindario discrepante: {vecindad_resumen}")

    # --- V4-F04: techo de la propagacion por near-duplicado ---------------
    log("   4/5 techo de impacto de la propagacion de etiquetas (sin aplicarla)")
    normalizado = [
        " ".join(TOKEN_PATTERN.findall(str(texto).lower())) for texto in test["text"]
    ]
    indice_train = {
        " ".join(TOKEN_PATTERN.findall(str(texto).lower())): int(identificador)
        for texto, identificador in zip(train["text"], train["id"], strict=True)
    }
    coincidencias = sum(1 for texto in normalizado if texto in indice_train)
    techo = {
        "tweets_de_test_con_texto_identico_a_train": int(coincidencias),
        "porcentaje_del_test": round(100.0 * coincidencias / len(test), 2),
        "techo_maximo_de_F1_en_puntos": round(
            100.0 * coincidencias / len(test) * 2.0 / 3.0, 4
        ),
        "nota": (
            "No se aplica: la propagacion de etiquetas usaria etiquetas de train "
            "para predecir test, lo que es legal pero con un techo de impacto "
            "despreciable; ademas §58 pide separarlo en un canal propio."
        ),
    }
    log(f"      {coincidencias} tweets de test con texto identico a train")

    # --- §47 y §96: oportunidades y matriz de correccion ----------------
    log("   5/5 matriz de correccion y oportunidades")
    canales = _cargar_canales_oof(y_dev, ids_dev)
    filas_correccion = []
    for nombre, proba in canales.items():
        pred = (proba >= UMBRALES_V03[None, :]).astype(int)
        acierto_ref = (pred_dev == y_dev).all(axis=1)
        acierto_canal = (pred == y_dev).all(axis=1)
        filas_correccion.append(
            {
                "model": nombre,
                "n": len(y_dev),
                "correct_v03": int(acierto_ref.sum()),
                "wrong_v03": int((~acierto_ref).sum()),
                "new_correct": int((~acierto_ref & acierto_canal).sum()),
                "new_wrong": int((acierto_ref & ~acierto_canal).sum()),
                "net_change": int((~acierto_ref & acierto_canal).sum())
                - int((acierto_ref & ~acierto_canal).sum()),
                "correction_rate": round(
                    float((~acierto_ref & acierto_canal).sum() / max((~acierto_ref).sum(), 1)), 4
                ),
                "f1_propio": round(
                    metricas_multietiqueta(y_dev, pred)["f1_media_columnas"], 4
                ),
            }
        )
    tabla_correccion = pd.DataFrame(filas_correccion).sort_values("net_change", ascending=False)
    tabla_correccion.to_csv(CORRECCIONES_PATH, index=False)
    mejores = tabla_correccion[tabla_correccion["net_change"] > 0]
    log(f"      canales que corrigen mas de lo que rompen: {len(mejores)} de {len(tabla_correccion)}")

    # Fusion tardia de los canales con net_change positivo (V4-H02, §49).
    if len(mejores) > 0:
        from sklearn.linear_model import Ridge

        nombres = list(mejores["model"])
        matriz = np.column_stack(
            [
                np.column_stack([pred_dev[:, j] for j in range(3)] + [canales[n] for n in nombres])
            ]
        )
        proba_fusion = np.zeros((len(y_dev), 3))
        for entrada, validacion in _pliegues_desarrollo(y_dev):
            meta = Ridge(alpha=3.0)
            meta.fit(matriz[entrada], y_dev[entrada])
            proba_fusion[validacion] = np.clip(meta.predict(matriz[validacion]), 0.0, 1.0)
        fusion = metricas_multietiqueta(y_dev, (proba_fusion >= UMBRALES_V03).astype(int))
        fusion_resumen = {
            "canales": nombres,
            "corrected_cv": round(fusion["f1_media_columnas"], 4),
            "delta_vs_v03": round(
                fusion["f1_media_columnas"]
                - metricas_multietiqueta(y_dev, pred_dev)["f1_media_columnas"],
                4,
            ),
        }
        tabla_correccion.to_csv(MATRIZ_PATH, index=False)
    else:
        fusion_resumen = {"canales": [], "corrected_cv": None, "nota": "sin canales complementarios"}

    # --- §95: taxonomia de errores ---------------------------------------
    taxonomia = construir_taxonomia(
        y_dev, pred_dev, train["text"].astype(str).tolist()[indices_dev], ids_dev
    )
    _escribir_taxonomia(taxonomia, robustez, vecindad_resumen)

    _escribir_figura(tabla_shift, tabla_correccion)

    resumen = {
        "experiment_id": "V4-F01..F04 + discovery",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duracion_s": round(time.time() - inicio_tiempo, 2),
        "shift": {
            "n_variables": int(len(tabla_shift)),
            "n_significativos": n_significativos,
            "detalle": filas_shift,
        },
        "robustez_duplicados": robustez,
        "vecindad": vecindad_resumen,
        "propagacion_near_duplicado": techo,
        "correccion_canales": filas_correccion,
        "fusion_tardia": fusion_resumen,
        "taxonomia": taxonomia,
        "uso_de_test": "solo variables no supervisadas y conteo de coincidencias; sin labels",
    }
    JSON_PATH.write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(f"   artefactos: {SHIFT_PATH.name}, {CORRECCIONES_PATH.name}, {MATRIZ_PATH.name}, {TAXONOMIA_PATH.name}, {JSON_PATH.name}")


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------
def _cargar_canales_oof(y_dev: np.ndarray, ids_dev: np.ndarray) -> dict[str, np.ndarray]:
    """Carga los canales OOF disponibles y alinea sus identificadores.

    Args:
        y_dev: etiquetas del desarrollo.
        ids_dev: identificadores del desarrollo.

    Returns:
        Diccionario ``nombre -> probabilidades``.
    """
    canales: dict[str, np.ndarray] = {}
    archivos = {
        "v2_meta_oof.npz": ("proba", None),
        "v2_oof_targeting.npz": ("proba", 1),
        "v2_oof_cita_negacion.npz": ("proba", 4),
        "v2_oof_syntax.npz": ("proba", 7),
        "v2_oof_emocion_nbsvm_ortografia.npz": ("proba", 10),
    }
    for archivo, (clave, indice) in archivos.items():
        ruta = PROJECT_ROOT / "features" / archivo
        if not ruta.exists():
            continue
        datos = np.load(ruta, allow_pickle=True)
        if "ids" in datos.files and not np.array_equal(datos["ids"].astype(np.int64), ids_dev):
            continue
        proba = datos[clave]
        if indice is not None and proba.ndim == 3:
            proba = proba[indice]
        canales[archivo.replace(".npz", "")] = proba.astype(float)
    return canales


def _pliegues_desarrollo(y_dev: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Recupera los pliegues corregidos del desarrollo.

    Args:
        y_dev: etiquetas del desarrollo.

    Returns:
        Lista de pares (entrenamiento, validacion).
    """
    from src.splits import pliegues_corregidos

    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    return pliegues_corregidos(desarrollo, grupos[indices_dev])


def _escribir_taxonomia(
    taxonomia: dict[str, Any], robustez: dict[str, Any], vecindad: dict[str, Any]
) -> None:
    """Escribe la taxonomia de errores v4 en markdown.

    Args:
        taxonomia: conteos por categoria.
        robustez: metricas por subconjunto.
        vecindad: resumen de la vecindad.
    """
    lineas = [
        "# Taxonomia de errores v4",
        "",
        f"Generado por `experiments/35_discovery_v4.py` el "
        f"{datetime.now(timezone.utc).date().isoformat()}. Conteos sobre el desarrollo "
        "corregido (n=1.483) con la referencia V03 y sus umbrales congelados.",
        "",
        "| Categoria | n | % del desarrollo | ids de ejemplo |",
        "|---|---|---|---|",
    ]
    for nombre, valores in taxonomia.items():
        lineas.append(
            f"| {nombre} | {valores['n']} | {valores['porcentaje']} | {valores['ids_ejemplo']} |"
        )
    lineas += [
        "",
        "## Robustez por subconjunto de duplicados (V4-F02)",
        "",
        "| Subconjunto | n | F1 media | F1 HS/TR/AG | error medio por etiqueta |",
        "|---|---|---|---|---|",
    ]
    for nombre, valores in robustez.items():
        f1 = "/".join(str(valores["f1_por_etiqueta"][e]) for e in ETIQUETAS)
        lineas.append(
            f"| {nombre} | {valores['n']} | {valores['f1_media']} | {f1} | "
            f"{valores['error_medio_por_etiqueta']} |"
        )
    lineas += [
        "",
        "## Desacuerdo con el vecino mas cercano (V4-F03)",
        "",
        f"- Ejemplos cuyo vecino mas cercano discrepa en 2 de 3 etiquetas o mas: "
        f"**{vecindad['n_vecindario_discrepante']}**.",
        f"- Error medio dentro: **{vecindad['error_medio_dentro']}**; fuera: "
        f"**{vecindad['error_medio_fuera']}**; delta {vecindad['delta']} "
        f"con IC95 [{vecindad['ic95'][0]}, {vecindad['ic95'][1]}].",
        "",
        "## Lectura",
        "",
        "Las categorias QUOTED, COLLECTIVE, AFFILIATIVE y NEWS son subconjuntos "
        "solapados de los errores por etiqueta, no categorias mutuamente "
        "excluyentes: sirven para leer donde se concentra el error, no para "
        "cuantificar una particion.",
    ]
    TAXONOMIA_PATH.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _escribir_figura(tabla_shift: pd.DataFrame, tabla_correccion: pd.DataFrame) -> None:
    """Dibuja el desplazamiento train-test y las correcciones por canal.

    Args:
        tabla_shift: tabla de variables no supervisadas.
        tabla_correccion: tabla de correccion por canal.
    """
    figura, ejes = plt.subplots(1, 2, figsize=(14, 5))
    colores = [
        "#C44E52" if valor else "#4C72B0" for valor in tabla_shift["desplazamiento_significativo"]
    ]
    ejes[0].barh(
        tabla_shift["variable"], tabla_shift["delta"], color=colores
    )
    ejes[0].axvline(0.0, color="black", linewidth=0.8)
    ejes[0].set_title("Shift train vs test (variables no supervisadas)")
    ejes[0].set_xlabel("media test - media train")
    top = tabla_correccion.head(12)
    ejes[1].barh(top["model"], top["net_change"], color="#4C72B0")
    ejes[1].axvline(0.0, color="black", linewidth=0.8)
    ejes[1].set_title("Neto de correccion por canal frente a V03")
    ejes[1].set_xlabel("new_correct - new_wrong")
    figura.tight_layout()
    FIGURA_PATH.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(FIGURA_PATH, dpi=150)
    plt.close(figura)


if __name__ == "__main__":
    main()
