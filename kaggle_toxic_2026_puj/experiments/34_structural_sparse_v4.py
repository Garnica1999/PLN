"""TRACK D v4: subrepresentaciones sparse que v2 no cubrio.

Contexto
--------
La fase v2 cerro como negativo las familias estructurales, pero siempre como
**features densas concatenadas** o como canales hstack: POS y dependencias
(E2-03/E2-04), targeting y colectivo (E2-05/E2-06), cita y negacion (E2-07/E2-08),
emocion (E2-09) y ortografia (E2-10). La propia auditoria v2 dejo cuatro brechas
documentadas que la spec v4 si ataca:

- **FASE 13**: sufijos y prefijos nunca se materializaron como canal propio;
- **§44**: las interacciones (insulto x target, insulto x negacion, insulto x
  risa, colectivo x insulto, cita x insulto) nunca se representaron como
  **tokens sparse**, sino como producto de features binarias;
- **§24**: las ventanas locales alrededor de target o insulto se probaron como
  densas (E2-09-EMO-LEX), no como tokens;
- **FASE 70**: los subtipos de HS implicito (estereotipo, deshumanizacion,
  amenaza) nunca tuvieron canal.

Reglas de la spec que se respetan aqui (§85, §88, §89): no se repite el canal
denso ya descartado, y cada linea se justifica con un error medido. Por eso
V4-D04 empieza con un **diagnostico de masa**: si los subtipos de HS implicito no
alcanzan un minimo de ejemplos, la linea se cierra sin construir el canal.

Los lexicons de ``<INSULT>``, ``<COLECTIVO>``, ``<AMENAZA>`` y ``<OBJETO>`` se
inducen **dentro de cada pliegue** con log-odds suavizado sobre las etiquetas de
ese pliegue (D-007), nunca con listas externas.

Uso
---
    /home/carlos/venv/bin/python experiments/34_structural_sparse_v4.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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
CSV_PATH = PROJECT_ROOT / "reports" / "structural_sparse_v4.csv"
JSON_PATH = PROJECT_ROOT / "reports" / "structural_sparse_v4.json"
DIAGNOSTICO_PATH = PROJECT_ROOT / "reports" / "implicit_hate_diagnostic_v4.json"
FIGURA_PATH = PROJECT_ROOT / "reports" / "figuras" / "fig_38_structural_sparse_v4.png"

PARAMETROS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMETROS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
N_LEXICON = 40
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
N_BOOTSTRAP = 2000
SEMILLA = 42
REFERENCIA_V03 = 0.777211
ALFA = 0.5
MINIMO_EJEMPLOS_SUBTIPO = 30

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
MENCION_PATTERN = re.compile(r"@\w+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
CUANTIFICADORES = frozenset(
    {"todos", "todas", "todo", "toda", "todos", "siempre", "nunca", "nadie",
     "ninguno", "ninguna", "cualquier", "cualquiera", "todo", "mismo"}
)
GENERALIZADORES = frozenset({"son", "es", "hacen", "hacen", "tienen", "quiere", "quieren"})
RISAS = frozenset({"jajaja", "jaja", "jeje", "jejeje", "jjj", "jajajaja", "hahaha", "haha"})


def log(mensaje: str) -> None:
    """Imprime una linea de progreso con marca de seccion.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V4-D] {mensaje}", flush=True)


# ---------------------------------------------------------------------------
# Lexicons inducidos dentro del pliegue
# ---------------------------------------------------------------------------
def inducir_lexicon(textos: list[str], y: np.ndarray, posicion: int, n: int = N_LEXICON) -> set[str]:
    """Induce el lexicon mas asociado a una etiqueta con log-odds suavizado.

    Args:
        textos: textos del pliegue de entrenamiento.
        y: matriz de etiquetas del pliegue.
        posicion: indice de la etiqueta objetivo.
        n: numero de lemas que se conservan.

    Returns:
        Conjunto de lemas.
    """
    frecuencia_pos: dict[str, int] = {}
    frecuencia_neg: dict[str, int] = {}
    positivos = y[:, posicion] == 1
    for texto, es_positivo in zip(textos, positivos, strict=True):
        for lema in set(TOKEN_PATTERN.findall(texto.lower())):
            destino = frecuencia_pos if es_positivo else frecuencia_neg
            destino[lema] = destino.get(lema, 0) + 1
    total_pos = max(int(positivos.sum()), 1)
    total_neg = max(int((~positivos).sum()), 1)
    puntuaciones = {}
    for lema, conteo in frecuencia_pos.items():
        if conteo < 3:
            continue
        tasa_pos = (conteo + ALFA) / (total_pos + ALFA * 1.0)
        tasa_neg = (frecuencia_neg.get(lema, 0) + ALFA) / (total_neg + ALFA)
        puntuaciones[lema] = np.log(tasa_pos / tasa_neg)
    ordenados = sorted(puntuaciones.items(), key=lambda par: -par[1])[:n]
    return {lema for lema, _ in ordenados}


def inducir_categorias(
    textos: list[str], y: np.ndarray
) -> dict[str, set[str]]:
    """Induce los lexicons de las categorias simbolicas usadas en las interacciones.

    Args:
        textos: textos del pliegue de entrenamiento.
        y: matriz de etiquetas del pliegue.

    Returns:
        Diccionario con los conjuntos de lemas de insulto, colectivo, amenaza y
        objeto (deshumanizacion).
    """
    return {
        "INSULT": inducir_lexicon(textos, y, 0),
        "COLECTIVO": inducir_lexicon(textos, y, 2),
        "AMENAZA": inducir_lexicon(textos, y, 0, n=25),
        "OBJETO": inducir_lexicon(textos, y, 2, n=25),
    }


# ---------------------------------------------------------------------------
# Canales simbolicos
# ---------------------------------------------------------------------------
def secuencia_afijos(textos: list[str], longitudes: tuple[int, ...] = (2, 3, 4)) -> list[str]:
    """Construye la secuencia de simbolos de prefijo y sufijo de cada palabra.

    Args:
        textos: textos a transformar.
        longitudes: longitudes de prefijo y sufijo a extraer.

    Returns:
        Lista de cadenas con los simbolos separados por espacio.
    """
    salida = []
    for texto in textos:
        simbolos = []
        for palabra in TOKEN_PATTERN.findall(texto.lower()):
            for longitud in longitudes:
                if len(palabra) > longitud:
                    simbolos.append(f"P{longitud}:{palabra[:longitud]}")
                    simbolos.append(f"S{longitud}:{palabra[-longitud:]}")
        salida.append(" ".join(simbolos))
    return salida


def secuencia_interacciones(textos: list[str], categorias: dict[str, set[str]]) -> list[str]:
    """Construye tokens de interaccion entre categorias detectadas.

    Los pares se emiten en el orden en que aparecen en el texto para que el
    orden sea informacion (insulto antes o despues de la mencion).

    Args:
        textos: textos a transformar.
        categorias: lexicons inducidos por categoria.

    Returns:
        Lista de cadenas de tokens.
    """
    salida = []
    for texto in textos:
        palabras = [p.lower() for p in TOKEN_PATTERN.findall(texto)]
        presentes = {
            nombre: [indice for indice, palabra in enumerate(palabras) if palabra in lemas]
            for nombre, lemas in categorias.items()
        }
        marcas: list[tuple[int, str]] = []
        for nombre, indices in presentes.items():
            marcas.extend((indice, nombre) for indice in indices)
        if any(TOKEN_PATTERN.findall(texto.lower())):
            marcas.append((0, "MEN" if MENCION_PATTERN.search(texto) else "NEN"))
        if URL_PATTERN.search(texto):
            marcas.append((0, "URL"))
        if re.search(r"[a-z]{4,}", texto.lower()) and RISAS & set(palabras):
            marcas.append((0, "RISA"))
        marcas.sort()
        nombres = [nombre for _, nombre in marcas]
        tokens = nombres + [
            f"{nombres[i]}_{nombres[i + 1]}" for i in range(len(nombres) - 1)
        ]
        salida.append(" ".join(tokens))
    return salida


def secuencia_ventanas(
    textos: list[str], categorias: dict[str, set[str]], ventana: int = 3
) -> list[str]:
    """Construye tokens de la ventana local alrededor de insulto y colectivo.

    Args:
        textos: textos a transformar.
        categorias: lexicons inducidos.
        ventana: numero de palabras a cada lado del ancla.

    Returns:
        Lista de cadenas de tokens.
    """
    salida = []
    for texto in textos:
        palabras = [p.lower() for p in TOKEN_PATTERN.findall(texto)]
        tokens = []
        for ancla in ("INSULT", "COLECTIVO", "AMENAZA"):
            for indice, palabra in enumerate(palabras):
                if palabra not in categorias.get(ancla, set()):
                    continue
                izquierda = palabras[max(indice - ventana, 0) : indice]
                derecha = palabras[indice + 1 : indice + 1 + ventana]
                for distancia, lema in enumerate(izquierda[::-1], start=1):
                    if distancia <= ventana:
                        tokens.append(f"{ancla}-L{distancia}:{lema}")
                for distancia, lema in enumerate(derecha, start=1):
                    if distancia <= ventana:
                        tokens.append(f"{ancla}+R{distancia}:{lema}")
        salida.append(" ".join(tokens))
    return salida


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


def ajustar_lineales(
    matriz: sparse.csr_matrix, y: np.ndarray
) -> LogisticRegression:
    """Devuelve un clasificador con los hiperparametros de la referencia.

    Args:
        matriz: matriz de entrenamiento (no se usa, se mantiene por simetria).
        y: matriz de etiquetas (no se usa, se mantiene por simetria).

    Returns:
        Modelo sin ajustar.
    """
    return LogisticRegression(
        C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=SEMILLA
    )


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------
def main() -> None:
    """Ejecuta los canales sparse nuevos del TRACK D."""
    inicio_tiempo = time.time()
    inicio = datetime.now(timezone.utc)
    log("Subrepresentaciones sparse nuevas (TRACK D)")

    oof = np.load(OOF_PATH, allow_pickle=True)
    ids_dev = oof["ids"].astype(np.int64)
    y_dev = oof["y_true"].astype(int)
    pred_ref = oof["pred"].astype(int)
    hardness = np.load(PROJECT_ROOT / "features" / "hardness_v4.npz", allow_pickle=True)
    error_hs = hardness["error"][:, 0]

    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos[indices_dev])
    textos = desarrollo["text"].tolist()

    # Diagnostico previo de V4-D04: masa de los subtipos de HS implicito.
    log("   0/4 diagnostico de masa de HS implicito (V4-D04)")
    falso_negativo_hs = (y_dev[:, 0] == 1) & (error_hs == 1)
    categorias_global = inducir_categorias(textos, y_dev)
    palabras_por_documento = [
        [p.lower() for p in TOKEN_PATTERN.findall(texto)] for texto in textos
    ]
    contiene_insulto = [
        bool({p for p in palabras} & categorias_global["INSULT"]) for palabras in palabras_por_documento
    ]
    contiene_amenaza = [
        bool({p for p in palabras} & categorias_global["AMENAZA"]) for palabras in palabras_por_documento
    ]
    contiene_colectivo = [
        bool({p for p in palabras} & categorias_global["COLECTIVO"]) for palabras in palabras_por_documento
    ]
    subtipos = {
        "HS1_con_insulto": int(falso_negativo_hs.sum()),
        "HS1_con_insulto_y_mencion": int(
            (falso_negativo_hs & np.array([bool(MENCION_PATTERN.search(t)) for t in textos])).sum()
        ),
        "HS1_sin_insulto": int((falso_negativo_hs & ~np.array(contiene_insulto)).sum()),
        "HS1_sin_insulto_con_amenaza": int(
            (falso_negativo_hs & ~np.array(contiene_insulto) & np.array(contiene_amenaza)).sum()
        ),
        "HS1_sin_insulto_con_colectivo": int(
            (falso_negativo_hs & ~np.array(contiene_insulto) & np.array(contiene_colectivo)).sum()
        ),
    }
    DIAGNOSTICO_PATH.write_text(
        json.dumps(
            {
                "experiment_id": "V4-D04",
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_falsos_negativos_hs": int(falso_negativo_hs.sum()),
                "subtipos": subtipos,
                "umbral_minimo": MINIMO_EJEMPLOS_SUBTIPO,
                "decision": (
                    "se construye el canal de HS implicito"
                    if subtipos["HS1_sin_insulto"] >= MINIMO_EJEMPLOS_SUBTIPO
                    else "NO_SIGNAL: la masa no alcanza el minimo; linea cerrada sin canal"
                ),
                "lexicones": {nombre: sorted(lemas) for nombre, lemas in categorias_global.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"      FN de HS: {int(falso_negativo_hs.sum())}; sin insulto: {subtipos['HS1_sin_insulto']}")

    resultados: list[dict[str, Any]] = []
    canales = ("afijos", "interacciones", "ventanas")

    for canal in canales:
        log(f"   canal: {canal}")
        proba_oof = np.zeros((len(desarrollo), len(ETIQUETAS)))
        proba_solo = np.zeros((len(desarrollo), len(ETIQUETAS)))
        for indice_fold, (entrenar, validar) in enumerate(pliegues):
            textos_entrenar = [textos[i] for i in entrenar]
            textos_validar = [textos[i] for i in validar]
            categorias = inducir_categorias(textos_entrenar, y_dev[entrenar])
            if canal == "afijos":
                cadena_entrenar = secuencia_afijos(textos_entrenar)
                cadena_validar = secuencia_afijos(textos_validar)
            elif canal == "interacciones":
                cadena_entrenar = secuencia_interacciones(textos_entrenar, categorias)
                cadena_validar = secuencia_interacciones(textos_validar, categorias)
            else:
                cadena_entrenar = secuencia_ventanas(textos_entrenar, categorias)
                cadena_validar = secuencia_ventanas(textos_validar, categorias)
            vectorizador_canal = TfidfVectorizer(
                analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True
            )
            matriz_canal = vectorizador_canal.fit_transform(cadena_entrenar)
            matriz_canal_aplicar = vectorizador_canal.transform(cadena_validar)

            palabra = TfidfVectorizer(**PARAMETROS_WORD)
            caracter = TfidfVectorizer(**PARAMETROS_CHAR)
            matriz_union = sparse.hstack(
                [palabra.fit_transform(textos_entrenar), caracter.fit_transform(textos_entrenar)],
                format="csr",
            )
            matriz_union_aplicar = sparse.hstack(
                [palabra.transform(textos_validar), caracter.transform(textos_validar)], format="csr"
            )
            union = sparse.hstack([matriz_union, matriz_canal], format="csr")
            union_aplicar = sparse.hstack([matriz_union_aplicar, matriz_canal_aplicar], format="csr")

            for posicion in range(len(ETIQUETAS)):
                modelo = ajustar_lineales(union, y_dev)
                modelo.fit(union, y_dev[entrenar, posicion])
                proba_oof[validar, posicion] = modelo.predict_proba(union_aplicar)[:, 1]
                modelo_solo = ajustar_lineales(matriz_canal, y_dev)
                modelo_solo.fit(matriz_canal, y_dev[entrenar, posicion])
                proba_solo[validar, posicion] = modelo_solo.predict_proba(matriz_canal_aplicar)[:, 1]
        resultados.append(evaluar(f"d:{canal}_hstack", proba_oof, y_dev, pliegues, pred_ref))
        log(f"      d:{canal}_hstack corrected_cv={resultados[-1]['corrected_cv_umbrales_v03']:.4f}")
        resultados.append(evaluar(f"d:{canal}_standalone", proba_solo, y_dev, pliegues, pred_ref))
        log(f"      d:{canal}_standalone corrected_cv={resultados[-1]['corrected_cv_umbrales_v03']:.4f}")

    tabla = pd.DataFrame(resultados).sort_values("corrected_cv_umbrales_v03", ascending=False)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(CSV_PATH, index=False)

    mejor = tabla.iloc[0]
    cumple = bool(
        float(mejor["corrected_cv_umbrales_v03"]) - REFERENCIA_V03 >= 0.005
        and int(mejor["pliegues_mejorados"]) >= 3
    )
    resumen = {
        "experiment_id": "V4-D01-D04",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duracion_s": round(time.time() - inicio_tiempo, 2),
        "referencia_v03": REFERENCIA_V03,
        "n_variantes": int(len(tabla)),
        "mejor_variante": str(mejor["variante"]),
        "mejor_corrected_cv": float(mejor["corrected_cv_umbrales_v03"]),
        "cumple_criterio_D016": cumple,
        "decision_familia": (
            "ADOPTAR" if cumple else "NO_SIGNAL: ningun canal sparse nuevo supera a V03"
        ),
        "detalle": resultados,
    }
    JSON_PATH.write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _escribir_figura(tabla)
    log(f"   mejor variante: {resumen['mejor_variante']} ({resumen['mejor_corrected_cv']:.4f})")
    log(f"   decision: {resumen['decision_familia']}")


def _escribir_figura(tabla: pd.DataFrame) -> None:
    """Dibuja el corrected_cv de los canales sparse nuevos.

    Args:
        tabla: tabla de resultados.
    """
    figura, eje = plt.subplots(figsize=(9, 5))
    colores = [
        "#C44E52" if valor <= REFERENCIA_V03 else "#4C72B0"
        for valor in tabla["corrected_cv_umbrales_v03"]
    ]
    eje.barh(tabla["variante"], tabla["corrected_cv_umbrales_v03"], color=colores)
    eje.axvline(REFERENCIA_V03, color="#55A868", linestyle="--", label="V03 = 0.7772")
    eje.set_xlabel("corrected_cv con umbrales de V03")
    eje.set_title("Canales sparse nuevos: afijos, interacciones y ventanas (v4, TRACK D)")
    eje.legend()
    figura.tight_layout()
    FIGURA_PATH.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(FIGURA_PATH, dpi=150)
    plt.close(figura)


if __name__ == "__main__":
    main()
