"""P2 de la fase v5: baseline FastText 10M, control Word2Vec, intrinseca y downstream.

Contexto
--------
P1 demostro que el corpus de 10M tokens de Wikipedia ES cubre el 68.39 % de los
tipos del vocabulario TOXIC (88.33 % de los tokens) y que **todos** los tipos
ausentes reciben vector por subpalabras. Ahora hay que responder la pregunta que
importa: ¿esa representacion distribuida entrenada desde cero aporta algo
utilizable para HS/TR/AG, y supera o no al control sin subpalabras?

Contenido
---------
1. **Entrenamiento desde cero** de tres modelos con el binario de fastText C++,
   con la arquitectura fijada en la seccion 4 de la spec (200d, ventana 5,
   subpalabras 3-6, negative 10, ``min_count`` 3, 5 epocas, semilla fija):
   - ``FASTTEXT-BASELINE`` (subpalabras 3-6),
   - ``FASTTEXT-SEED-CONTROL`` (misma arquitectura, otra semilla: mide
     estabilidad, seccion 20),
   - ``WORD2VEC-CONTROL / FASTTEXT-SG-SUBWORDS-OFF`` (``minn=0, maxn=0``).
2. **Vectores de documento** con tres poolings de la seccion 21:
   **A** media de tokens, **B** media ponderada por TF-IDF, **C** media + maximo.
3. **Downstream** con el protocolo corregido v2 (`StratifiedGroupKFold(5)` sobre
   grupos near-duplicado UNION pseudo-autor): LogisticRegression (C=2.0,
   ``balanced``, los mismos de V03) y LinearSVC. Se reportan las dos lecturas
   comparables con V03: umbral **0.5** (V03 = 0.771778) y umbrales **ajustados**
   sobre las probabilidades OOF (V03 = 0.777211 con 0.44/0.51/0.46).
4. **Comparacion pareada con V03**: bootstrap pareado de la diferencia de la
   metrica primaria, con IC95 y conteo de pliegues ganados, sobre las mismas
   1.483 filas del desarrollo.
5. **Intrinseca** (seccion 20): vecinos, estabilidad entre semillas y
   comportamiento ante variantes ortograficas, **sin** usar labels externos para
   afirmar que un vecino es "correcto".

Lo que este script **no** hace: no usa el holdout, no lee el test para entrenar ni
para seleccionar, no genera submissions, no anade texto de TOXIC al corpus y no
ajusta nada con informacion del conjunto evaluado mas alla del ajuste de
umbrales, que se reporta por separado y en el mismo regimen que V03.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_02_fasttext_baseline_v5.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_test, read_train
from src.embedding_utils import (
    cargar_matriz_vec,
    centrar,
    entrenar_fasttext,
    normalizar_emb_normal,
    tokenizar,
    vectores_de_tipos,
    vocabulario_textos,
)
from src.evaluation import metricas_multietiqueta
from src.splits import grupos_fusionados, holdout_v2, pliegues_corregidos

EXPERIMENTO = "EBD-01..05 (P2)"
FASTTEXT_BIN = PROJECT_ROOT / "tools" / "fastText" / "build" / "fasttext"
CORPUS_10M = PROJECT_ROOT / "data" / "external" / "corpora" / "wikipedia_es" / "emb_normal" / "tokens_10M.txt"
EMB_DIR = PROJECT_ROOT / "data" / "external" / "embeddings"
REPORTE_INTRINSECA = PROJECT_ROOT / "reports" / "embedding_intrinsic_eval.md"
REPORTE_DOWNSTREAM = PROJECT_ROOT / "reports" / "embedding_downstream_v5.json"
REPORTE_DOWNSTREAM_CSV = PROJECT_ROOT / "reports" / "embedding_downstream_results.csv"
REPORTE_EQUIVALENCIA = PROJECT_ROOT / "reports" / "embedding_normalizacion_equivalencia.json"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
FIGURAS = PROJECT_ROOT / "reports" / "figuras"

# Arquitectura fijada en la seccion 4 de la spec.
ARQUITECTURA = {
    "dim": 200,
    "ventana": 5,
    "min_count": 3,
    "negativos": 10,
    "epocas": 5,
    "hilos": 16,
    "bucket": 200_000,
    "subpalabras": (3, 6),
    "muestreo": 1e-4,
    "sg": 1,
}
SEMILLA_BASELINE = 42
SEMILLA_CONTROL = 7
REPETICIONES_BOOTSTRAP = 10_000
SEMILLA_BOOTSTRAP = 20260925
REJILLA_UMBRALES = np.round(np.arange(0.05, 0.96, 0.01), 2)

MODELOS = {
    "FASTTEXT-BASELINE": {"minn": 3, "maxn": 6, "semilla": SEMILLA_BASELINE},
    "FASTTEXT-SEED-CONTROL": {"minn": 3, "maxn": 6, "semilla": SEMILLA_CONTROL},
    "WORD2VEC-CONTROL": {"minn": 0, "maxn": 0, "semilla": SEMILLA_BASELINE},
}

PALABRAS_INTRINSECAS = [
    "violencia", "inmigración", "gobierno", "mujer", "niño", "familia",
    "iglesia", "fútbol", "música", "trabajo", "ciudad", "guerra",
]
VARIANTES_ORTOGRAFICAS = [
    "callate", "cllate", "violacin", "espaa", "imbcil", "arabe", "estn",
    "mereces", "pinche", "inmigracin", "guarra", "mereca", "estupido", "perra",
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V5-P2] {mensaje}", flush=True)


def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo.

    Args:
        ruta: ruta del archivo.

    Returns:
        Hash hexadecimal.
    """
    import hashlib

    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 0. Equivalencia de la normalizacion
# ---------------------------------------------------------------------------
def verificar_equivalencia_normalizacion(textos: list[str]) -> dict[str, Any]:
    """Comprueba que la normalizacion de este script coincide con la de P1.

    ``experiments/embedding_01_corpus_wikipedia_v5.py`` conserva su propia copia
    de la normalizacion como evidencia historica. Aqui se importa el script de P1
    de forma aislada y se comparan los resultados sobre los 2.885 textos de
    TOXIC, de modo que la duplicacion queda verificada.

    Args:
        textos: textos de TOXIC sin normalizar.

    Returns:
        Diccionario con el resultado de la comparacion.
    """
    log("0/6 equivalencia de la normalizacion con P1")
    import importlib.util

    ruta_p1 = PROJECT_ROOT / "experiments" / "embedding_01_corpus_wikipedia_v5.py"
    especificacion = importlib.util.spec_from_file_location("embedding_01_p1", ruta_p1)
    modulo_p1 = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo_p1)
    diferencias = 0
    for texto in textos:
        if modulo_p1.normalizar_emb_normal(texto) != normalizar_emb_normal(texto):
            diferencias += 1
        if modulo_p1.tokenizar(texto) != tokenizar(texto):
            diferencias += 1
    resultado = {
        "textos_comparados": len(textos),
        "diferencias": diferencias,
        "cumple": diferencias == 0,
        "nota": (
            "P1 conserva su copia historica; se verifica que coincide token a token "
            "con src/embedding_utils.py sobre los textos de train y test."
        ),
    }
    log(f"   {len(textos)} textos | diferencias: {diferencias}")
    if not resultado["cumple"]:
        raise RuntimeError("La normalizacion de P2 difiere de la de P1")
    REPORTE_EQUIVALENCIA.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return resultado


# ---------------------------------------------------------------------------
# 1. Entrenamiento
# ---------------------------------------------------------------------------
def entrenar_modelos() -> dict[str, Any]:
    """Entrena los tres modelos desde cero y registra sus huellas.

    Returns:
        Diccionario con los parametros y rutas de cada modelo.
    """
    log("1/6 entrenamiento desde cero (fastText C++)")
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    registro: dict[str, Any] = {}
    for nombre, configuracion in MODELOS.items():
        prefijo = EMB_DIR / nombre.lower().replace("-", "_")
        inicio = time.time()
        if Path(f"{prefijo}.vec").exists():
            log(f"   {nombre}: ya existe, se reutiliza")
            segundos = 0.0
        else:
            resultado = entrenar_fasttext(
                FASTTEXT_BIN,
                CORPUS_10M,
                prefijo,
                dim=ARQUITECTURA["dim"],
                ventana=ARQUITECTURA["ventana"],
                min_count=ARQUITECTURA["min_count"],
                negativos=ARQUITECTURA["negativos"],
                epocas=ARQUITECTURA["epocas"],
                hilos=ARQUITECTURA["hilos"],
                semilla=configuracion["semilla"],
                minn=configuracion["minn"],
                maxn=configuracion["maxn"],
                bucket=ARQUITECTURA["bucket"],
            )
            segundos = time.time() - inicio
            log(f"   {nombre}: {segundos:.0f} s")
            _ = resultado
        ruta_vec = Path(f"{prefijo}.vec")
        with ruta_vec.open(encoding="utf-8") as archivo:
            numero, dimension = archivo.readline().split()
        registro[nombre] = {
            "id": nombre,
            "etiqueta": (
                "FASTTEXT-SG-SUBWORDS-3-6"
                if configuracion["minn"]
                else "WORD2VEC-CONTROL / FASTTEXT-SG-SUBWORDS-OFF"
            ),
            "arquitectura": {**ARQUITECTURA, "minn": configuracion["minn"], "maxn": configuracion["maxn"],
                             "semilla": configuracion["semilla"]},
            "corpus": str(CORPUS_10M.relative_to(PROJECT_ROOT)),
            "vocab_size": int(numero),
            "dim": int(dimension),
            "embedding_hash_sha256": sha256_de(ruta_vec),
            "bin_bytes": Path(f"{prefijo}.bin").stat().st_size,
            "segundos": round(segundos, 1),
        }
    return registro


# ---------------------------------------------------------------------------
# 2. Vectores de documento
# ---------------------------------------------------------------------------
def construir_vectores_documento(
    textos: list[str],
    vectores_tipos: dict[str, np.ndarray],
    idf: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Construye los vectores de documento con los tres poolings.

    Los tokens fuera de vocabulario usan el vector que fastText compone con
    subpalabras; en el control sin subpalabras esos vectores no existen y el token
    se omite del pool, lo que hace visible la diferencia entre ambos modelos.

    Args:
        textos: textos normalizados.
        vectores_tipos: vectores por tipo.
        idf: pesos TF-IDF por tipo, o ``None`` para el pooling de media simple.

    Returns:
        Diccionario ``pooling -> matriz``.
    """
    dimensiones = next(iter(vectores_tipos.values())).shape[0]
    medias = np.zeros((len(textos), dimensiones), dtype=np.float32)
    maximos = np.full((len(textos), dimensiones), -np.inf, dtype=np.float32)
    utiles_por_documento = np.zeros(len(textos), dtype=np.int32)
    for posicion, texto in enumerate(textos):
        tokens = tokenizar(texto)
        utiles = [token for token in tokens if token in vectores_tipos]
        utiles_por_documento[posicion] = len(utiles)
        if not utiles:
            continue
        valores = np.vstack([vectores_tipos[token] for token in utiles])
        medias[posicion] = valores.mean(axis=0)
        maximos[posicion] = valores.max(axis=0)
    maximos[~np.isfinite(maximos)] = 0.0
    vectores: dict[str, np.ndarray] = {"A_media": medias}
    if idf is not None:
        # Pooling B: media ponderada por TF-IDF (el peso depende solo del corpus
        # de desarrollo y no usa labels, por lo que no hay fuga de supervision).
        ponderadas = np.zeros_like(medias)
        for posicion, texto in enumerate(textos):
            tokens = [token for token in tokenizar(texto) if token in vectores_tipos]
            if not tokens:
                continue
            pesos = np.asarray([idf[token] for token in tokens], dtype=np.float32)
            valores = np.vstack([vectores_tipos[token] for token in tokens])
            ponderadas[posicion] = (valores * pesos[:, None]).sum(axis=0) / max(pesos.sum(), 1e-9)
        vectores["B_tfidf"] = ponderadas
    vectores["C_media_max"] = np.hstack([medias, maximos])
    return vectores


def construir_idf(textos: list[str]) -> dict[str, float]:
    """Calcula los pesos TF-IDF por tipo sobre el corpus de desarrollo.

    Args:
        textos: textos normalizados del desarrollo.

    Returns:
        Diccionario tipo -> peso IDF.
    """
    vectorizador = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w[\w']*\b", lowercase=False)
    vectorizador.fit(textos)
    vocabulario = vectorizador.get_feature_names_out()
    idf = vectorizador.idf_
    return {palabra: float(idf[indice]) for indice, palabra in enumerate(vocabulario)}


# ---------------------------------------------------------------------------
# 3. Evaluacion downstream
# ---------------------------------------------------------------------------
def ajustar_umbrales(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Ajusta un umbral por etiqueta maximizando la F1 de la clase positiva.

    Se hace sobre las probabilidades **fuera de pliegue** del desarrollo, que es
    el mismo regimen con el que se fijaron los umbrales de V03 (0.44/0.51/0.46).
    Se reporta siempre tambien la lectura con umbral 0.5, que es la comparacion
    mas conservative y no optimiza nada.

    Args:
        y_true: matriz binaria verdadera.
        proba: matriz de probabilidades fuera de pliegue.

    Returns:
        Vector de tres umbrales.
    """
    umbrales = np.empty(3, dtype=np.float64)
    for etiqueta in range(3):
        mejor = (0.0, 0.5)
        for umbral in REJILLA_UMBRALES:
            valor = float(
                f1_score(y_true[:, etiqueta], (proba[:, etiqueta] >= umbral).astype(int), zero_division=0)
            )
            if valor > mejor[0]:
                mejor = (valor, float(umbral))
        umbrales[etiqueta] = mejor[1]
    return umbrales


def metrica_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula la metrica primaria tal como la define el proyecto.

    Es la media column-wise de la F1 binaria de la clase positiva, exactamente
    ``f1_media_columnas`` de :mod:`src.evaluation`, pero con aritmetica vectorial
    para poder evaluarla miles de veces dentro del bootstrap.

    Args:
        y_true: matriz binaria verdadera.
        y_pred: matriz binaria predicha.

    Returns:
        Metrica primaria.
    """
    columnas = []
    for etiqueta in range(3):
        verdaderos = y_true[:, etiqueta] == 1
        predichos = y_pred[:, etiqueta] == 1
        verdaderos_positivos = float(np.sum(verdaderos & predichos))
        falsos_positivos = float(np.sum(~verdaderos & predichos))
        falsos_negativos = float(np.sum(verdaderos & ~predichos))
        denominador = 2 * verdaderos_positivos + falsos_positivos + falsos_negativos
        columnas.append(2 * verdaderos_positivos / denominador if denominador > 0 else 0.0)
    return float(np.mean(columnas))


def bootstrap_pareado(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    repeticiones: int = REPETICIONES_BOOTSTRAP,
) -> dict[str, Any]:
    """Estima el intervalo de la diferencia de la metrica primaria por bootstrap.

    Args:
        y_true: matriz binaria verdadera.
        pred_a: predicciones del candidato.
        pred_b: predicciones de la referencia.

    Returns:
        Diccionario con la diferencia observada, su IC95 y el balance de filas.
    """
    # El estimando del bootstrap es la metrica reportada, no una aproximacion
    # por fila: se reevalua `f1_media_columnas` en cada remuestreo.
    oficial_a = metricas_multietiqueta(y_true, pred_a)["f1_media_columnas"]
    oficial_b = metricas_multietiqueta(y_true, pred_b)["f1_media_columnas"]
    if abs(metrica_macro(y_true, pred_a) - oficial_a) > 1e-12:
        raise RuntimeError("metrica_macro no reproduce f1_media_columnas")
    if abs(metrica_macro(y_true, pred_b) - oficial_b) > 1e-12:
        raise RuntimeError("metrica_macro no reproduce f1_media_columnas")
    generador = np.random.default_rng(SEMILLA_BOOTSTRAP)
    n = y_true.shape[0]
    diferencias = np.empty(repeticiones, dtype=np.float64)
    for repeticion in range(repeticiones):
        indices = generador.integers(0, n, n)
        diferencias[repeticion] = metrica_macro(y_true[indices], pred_a[indices]) - metrica_macro(
            y_true[indices], pred_b[indices]
        )
    cuantiles = np.quantile(diferencias, [0.025, 0.5, 0.975])
    por_fila = metrica_por_fila(y_true, pred_a) - metrica_por_fila(y_true, pred_b)
    return {
        "diferencia_observada": round(oficial_a - oficial_b, 6),
        "ic95": [round(float(cuantiles[0]), 6), round(float(cuantiles[2]), 6)],
        "mediana_bootstrap": round(float(cuantiles[1]), 6),
        "filas_mejoradas": int(np.sum(por_fila > 1e-9)),
        "filas_igualadas": int(np.sum(np.abs(por_fila) <= 1e-9)),
        "filas_empeoradas": int(np.sum(por_fila < -1e-9)),
        "repeticiones": repeticiones,
    }


def metrica_por_fila(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calcula la F1 de cada etiqueta en cada fila.

    Solo se usa para contar cuantas filas mejora o empeora un candidato respecto
    a la referencia; **no** es la metrica primaria y no se reporta como tal.

    Args:
        y_true: matriz binaria verdadera.
        y_pred: matriz binaria predicha.

    Returns:
        Vector con la media de las tres F1 por fila.
    """
    columnas = []
    for etiqueta in range(3):
        verdaderos = y_true[:, etiqueta] == 1
        predichos = y_pred[:, etiqueta] == 1
        verdaderos_positivos = (verdaderos & predichos).astype(np.float64)
        falsos_positivos = (~verdaderos & predichos).astype(np.float64)
        falsos_negativos = (verdaderos & ~predichos).astype(np.float64)
        denominador = 2 * verdaderos_positivos + falsos_positivos + falsos_negativos
        columnas.append(np.where(denominador > 0, 2 * verdaderos_positivos / np.maximum(denominador, 1e-12), 0.0))
    return np.mean(np.column_stack(columnas), axis=1)


def evaluar_downstream(
    nombre: str,
    matriz: np.ndarray,
    y: np.ndarray,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    escalador: bool,
) -> dict[str, Any]:
    """Evalua un clasificador sobre vectores de documento con el protocolo v2.

    Args:
        nombre: identificador de la combinacion.
        matriz: vectores de documento.
        y: matriz binaria de labels.
        pliegues: pliegues corregidos.
        escalador: si aplicar ``StandardScaler`` dentro de cada pliegue.

    Returns:
        Diccionario con probabilidades OOF, metricas y comparaciones.
    """
    proba = np.zeros((matriz.shape[0], 3), dtype=np.float64)
    f1_por_pliegue: list[float] = []
    for indices_entrenamiento, indices_validacion in pliegues:
        if escalador:
            # El escalado se ajusta **solo** con los datos de entrenamiento del
            # pliegue: ajustarlo antes seria fuga de la validacion.
            escalador_local = StandardScaler().fit(matriz[indices_entrenamiento])
            bloque = escalador_local.transform(matriz[indices_entrenamiento])
            consulta = escalador_local.transform(matriz[indices_validacion])
        else:
            bloque = matriz[indices_entrenamiento]
            consulta = matriz[indices_validacion]
        # `liblinear` no admite salida multiple: se ajusta **un modelo binario por
        # etiqueta**, igual que V03 (de ahi que existan tres umbrales).
        probabilidades = np.empty((len(indices_validacion), 3), dtype=np.float64)
        for etiqueta in range(3):
            objetivo = y[indices_entrenamiento, etiqueta]
            if nombre.endswith("SVC"):
                modelo = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=42)
                modelo.fit(bloque, objetivo)
                probabilidades[:, etiqueta] = 1.0 / (
                    1.0 + np.exp(-modelo.decision_function(consulta))
                )
            else:
                modelo = LogisticRegression(
                    C=2.0, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=42
                )
                modelo.fit(bloque, objetivo)
                probabilidades[:, etiqueta] = modelo.predict_proba(consulta)[:, 1]
        proba[indices_validacion] = probabilidades
        f1_por_pliegue.append(
            float(
                metricas_multietiqueta(y[indices_validacion], (probabilidades >= 0.5).astype(int))[
                    "f1_media_columnas"
                ]
            )
        )
    umbrales_ajustados = ajustar_umbrales(y, proba)
    predicciones_05 = (proba >= 0.5).astype(int)
    predicciones_ajustadas = (proba >= umbrales_ajustados[None, :]).astype(int)
    metricas_05 = metricas_multietiqueta(y, predicciones_05)
    metricas_ajustadas = metricas_multietiqueta(y, predicciones_ajustadas)
    return {
        "combinacion": nombre,
        "corrected_cv_umbral_0.5": round(metricas_05["f1_media_columnas"], 6),
        "corrected_cv_umbrales_ajustados": round(metricas_ajustadas["f1_media_columnas"], 6),
        "umbrales_ajustados": [float(valor) for valor in umbrales_ajustados],
        "f1_por_etiqueta_0.5": metricas_05["f1_por_etiqueta"],
        "f1_por_etiqueta_ajustados": metricas_ajustadas["f1_por_etiqueta"],
        "f1_por_pliegue_0.5": [round(valor, 4) for valor in f1_por_pliegue],
        "std_pliegues_0.5": round(float(np.std(f1_por_pliegue, ddof=1)), 4),
        "proba": proba,
        "pred_0.5": predicciones_05,
        "pred_ajustados": predicciones_ajustadas,
    }


# ---------------------------------------------------------------------------
# 4. Intrinseca
# ---------------------------------------------------------------------------
def vecinos_centrados(
    matriz: np.ndarray,
    indice: dict[str, int],
    palabras: list[str],
    k: int = 5,
) -> dict[str, list[tuple[str, float]]]:
    """Devuelve los vecinos mas cercanos en coseno centrado.

    Args:
        matriz: matriz normalizada de vectores del corpus.
        indice: indice palabra -> fila.
        palabras: palabras de consulta.
        k: numero de vecinos.

    Returns:
        Diccionario palabra -> lista de (vecino, coseno).
    """
    _, centrada = centrar(matriz)
    lista = [None] * len(indice)
    for palabra, posicion in indice.items():
        lista[posicion] = palabra
    salida: dict[str, list[tuple[str, float]]] = {}
    for palabra in palabras:
        if palabra not in indice:
            salida[palabra] = []
            continue
        consulta = centrada[indice[palabra]]
        similitudes = centrada @ consulta
        orden = np.argsort(-similitudes)[: k + 1]
        salida[palabra] = [
            (lista[posicion], round(float(similitudes[posicion]), 3))
            for posicion in orden
            if lista[posicion] != palabra
        ][:k]
    return salida


def estabilidad_entre_semillas(
    indice_a: dict[str, int],
    matriz_a: np.ndarray,
    indice_b: dict[str, int],
    matriz_b: np.ndarray,
    palabras: list[str],
    k: int = 10,
) -> dict[str, Any]:
    """Mide cuanto dependen los vecinos de la semilla de inicializacion.

    Args:
        indice_a: indice del primer modelo.
        matriz_a: matriz normalizada del primer modelo.
        indice_b: indice del segundo modelo.
        matriz_b: matriz normalizada del segundo modelo.
        palabras: palabras de consulta.
        k: tamano de la lista de vecinos.

    Returns:
        Diccionario con el solapamiento de vecinos y la similitud entre modelos.
    """
    _, centrada_a = centrar(matriz_a)
    _, centrada_b = centrar(matriz_b)
    comunes = [palabra for palabra in palabras if palabra in indice_a and palabra in indice_b]
    lista_a = [None] * len(indice_a)
    for palabra, posicion in indice_a.items():
        lista_a[posicion] = palabra
    lista_b = [None] * len(indice_b)
    for palabra, posicion in indice_b.items():
        lista_b[posicion] = palabra
    solapamientos: list[float] = []
    cosenos: list[float] = []
    for palabra in comunes:
        vector_a = centrada_a[indice_a[palabra]]
        vector_b = centrada_b[indice_b[palabra]]
        cosenos.append(float(np.dot(vector_a, vector_b)))
        top_a = set(np.argsort(-(centrada_a @ vector_a))[: k + 1])
        top_b = set(np.argsort(-(centrada_b @ vector_b))[: k + 1])
        comunes_top = top_a & top_b
        comunes_top.discard(indice_a[palabra])
        comunes_top.discard(indice_b[palabra])
        solapamientos.append(len(comunes_top) / k)
    return {
        "palabras_evaluadas": len(comunes),
        "coseno_centrado_medio_entre_semillas": round(float(np.mean(cosenos)), 4) if cosenos else 0.0,
        "solapamiento_medio_top10": round(float(np.mean(solapamientos)), 4) if solapamientos else 0.0,
    }


# ---------------------------------------------------------------------------
# 5. Informes
# ---------------------------------------------------------------------------
def escribir_informes(
    modelos: dict[str, Any],
    downstream: list[dict[str, Any]],
    intrinseca: dict[str, Any],
    equivalencia: dict[str, Any],
) -> None:
    """Escribe los informes de P2.

    Args:
        modelos: registro de entrenamiento.
        downstream: resultados downstream.
        intrinseca: resultados intrinsecos.
        equivalencia: resultado de la verificacion de normalizacion.
    """
    with REPORTE_DOWNSTREAM_CSV.open("w", encoding="utf-8", newline="") as archivo:
        campos = [
            "modelo", "pooling", "clasificador", "dim",
            "corrected_cv_umbral_0.5", "corrected_cv_umbrales_ajustados",
            "delta_vs_v03_0.5", "delta_vs_v03_ajustados",
            "ic95_ajustados", "pliegues_mejorados", "filas_mejoradas", "filas_empeoradas",
            "std_pliegues_0.5", "umbrales_ajustados", "f1_HS", "f1_TR", "f1_AG", "segundos_modelo",
        ]
        escritor = csv.DictWriter(archivo, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        for fila in downstream:
            escritor.writerow({campo: fila.get(campo, "") for campo in campos})

    oof = np.load(OOF_V03, allow_pickle=True)
    lineas = [
        "# Evaluacion intrinseca de la fase v5 (corpus 10M)",
        "",
        f"Generado por `experiments/embedding_02_fasttext_baseline_v5.py` ({EXPERIMENTO}).",
        "Seccion 20 de la spec. **No se usa ningun label externo de hate speech**: los vecinos se",
        "presentan como evidencia de la geometria aprendida, sin afirmar que un vecino sea",
        "\"correcto\".",
        "",
        "## 1. Modelos entrenados desde cero",
        "",
        "| Modelo | Etiqueta | dim | subpalabras | semilla | vocabulario | sha256 del .vec | s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for nombre, registro in modelos.items():
        arquitectura = registro["arquitectura"]
        lineas.append(
            f"| {nombre} | {registro['etiqueta']} | {registro['dim']} | "
            f"{arquitectura['minn']}-{arquitectura['maxn']} | {arquitectura['semilla']} | "
            f"{registro['vocab_size']} | `{registro['embedding_hash_sha256'][:16]}` | {registro['segundos']} |"
        )
    stability = intrinseca["estabilidad"]
    lineas += [
        "",
        "## 2. Estabilidad entre semillas (seccion 20)",
        "",
        f"- Coseno centrado medio entre los vectores de `FASTTEXT-BASELINE` (semilla "
        f"{SEMILLA_BASELINE}) y `FASTTEXT-SEED-CONTROL` (semilla {SEMILLA_CONTROL}) sobre "
        f"{stability['palabras_evaluadas']} palabras: "
        f"**{stability['coseno_centrado_medio_entre_semillas']:.3f}**.",
        f"- Solapamiento medio de los 10 vecinos mas cercanos: "
        f"**{stability['solapamiento_medio_top10']:.3f}**.",
        "",
        "**Como leer estos dos numeros.** El coseno centrado entre semillas ~0 **no** significa que",
        "la representacion sea ruido: las soluciones de word2vec/fastText solo son identicas salvo",
        "rotacion y escala, de modo que las direcciones absolutas cambian con la semilla aunque la",
        "geometria sea la misma. Lo que si es comparable entre semillas es la **estructura relativa**,",
        "y por eso se reporta el solapamiento de vecinos. Si el solapamiento fuera bajo, la",
        "representacion seria principalmente ruido de inicializacion y ningun resultado downstream de",
        "la fase podria leerse como propiedad de la representacion.",
        "",
        "## 3. Vecinos mas cercanos (coseno centrado)",
        "",
    ]
    for nombre in ("FASTTEXT-BASELINE", "WORD2VEC-CONTROL"):
        if nombre not in intrinseca["vecinos"]:
            continue
        lineas += [f"### {nombre}", "", "| palabra | vecinos |", "|---|---|"]
        for palabra, vecinos in intrinseca["vecinos"][nombre].items():
            if not vecinos:
                lineas.append(f"| {palabra} | (no esta en el vocabulario del corpus) |")
                continue
            texto = ", ".join(f"`{vecino}` ({coseno:.3f})" for vecino, coseno in vecinos)
            lineas.append(f"| {palabra} | {texto} |")
        lineas.append("")
    variantes = intrinseca["variantes"]
    lineas += [
        "## 4. Variantes ortograficas (tipos word-OOV del corpus TOXIC)",
        "",
        "Estos tipos **no** estan en el vocabulario del corpus. FastText les compone un vector con",
        "subpalabras; el control Word2Vec, entrenado con `minn=0, maxn=0`, no puede. La tabla",
        "muestra el vecino mas cercano de cada variante.",
        "",
        "| variante (freq. TOXIC) | FASTTEXT-BASELINE | WORD2VEC-CONTROL |",
        "|---|---|---|",
    ]
    for entrada in variantes:
        base = entrada["fasttext"]
        control = entrada["word2vec"]
        lineas.append(
            f"| `{entrada['tipo']}` ({entrada['freq_toxic']}) | {base} | {control} |"
        )
    lineas += [
        "",
        "## 5. Limitaciones",
        "",
        "1. El corpus es enciclopedico: los vecinos de palabras como `música` o `fútbol` son",
        "   coherentes, pero eso dice poco sobre el registro informal que caracteriza al problema.",
        "2. Los vecinos de variantes ortograficas son **indicativos**: con 10M tokens la señal",
        "   subléxica es débil y no se afirma que el vecino sea el lexema correcto.",
        f"3. La normalizacion de este script se comparó token a token con la de P1 sobre los "
        f"{equivalencia['textos_comparados']} textos de TOXIC: {equivalencia['diferencias']} diferencias.",
    ]
    REPORTE_INTRINSECA.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # JSON con el detalle numerico
    serializable = {
        "experimentos": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arquitectura": {**ARQUITECTURA, "subpalabras": list(ARQUITECTURA["subpalabras"])},
        "modelos": modelos,
        "referencia_v03": {
            "corrected_cv_umbrales_congelados": float(oof["corrected_cv"][0]),
            "corrected_cv_umbral_0.5": 0.771778476256122,
            "umbrales": [float(valor) for valor in oof["umbrales"]],
            "f1_por_pliegue": [float(valor) for valor in oof["f1_pliegues"]],
        },
        "equivalencia_normalizacion": equivalencia,
        "intrinseca": intrinseca,
        "downstream": [
            {clave: valor for clave, valor in fila.items() if clave not in {"proba", "pred_0.5", "pred_ajustados"}}
            for fila in downstream
        ],
    }
    REPORTE_DOWNSTREAM.write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Ejecuta P2 completa."""
    inicio = time.time()
    log("P2: baseline FastText 10M, control Word2Vec, intrinseca y downstream")
    train = read_train()
    test = read_test()
    todos = train["text"].astype(str).tolist() + test["text"].astype(str).tolist()
    equivalencia = verificar_equivalencia_normalizacion(todos)

    grupos = grupos_fusionados(train)
    indices_dev, indices_holdout = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos[indices_dev]
    pliegues = pliegues_corregidos(desarrollo, grupos_dev)
    y = desarrollo[["HS", "TR", "AG"]].to_numpy(dtype=int)
    oof = np.load(OOF_V03, allow_pickle=True)
    ids_oof = np.sort(oof["ids"].astype(int))
    # La columna `id` del train se lee como object (cadena): se castea a entero
    # para compararla con el OOF canonico, que la guarda como int64.
    ids_dev = np.sort(desarrollo["id"].astype(int).to_numpy())
    if not np.array_equal(ids_oof, ids_dev):
        raise RuntimeError("Los ids del OOF canonico no coinciden con el desarrollo")
    textos_norm = [normalizar_emb_normal(texto) for texto in desarrollo["text"].astype(str)]
    log(f"   desarrollo: {len(desarrollo)} textos | y {y.shape} | ids alineados con el OOF canonico")

    modelos = entrenar_modelos()
    idf = construir_idf(textos_norm)
    textos_norm_todos = [normalizar_emb_normal(texto) for texto in todos]
    tipos_unicos = sorted(vocabulario_textos(textos_norm_todos).keys())

    vectores_documento: dict[str, dict[str, np.ndarray]] = {}
    for nombre, registro in modelos.items():
        if nombre == "FASTTEXT-SEED-CONTROL":
            continue  # solo se usa para la estabilidad intrinseca
        prefijo = EMB_DIR / nombre.lower().replace("-", "_")
        vectores_tipos = vectores_de_tipos(
            FASTTEXT_BIN, Path(f"{prefijo}.bin"), tipos_unicos, EMB_DIR / f"{nombre.lower()}_sifts.txt"
        )
        log(f"   {nombre}: vectores de tipo disponibles {len(vectores_tipos)} de {len(tipos_unicos)}")
        vectores_documento[nombre] = construir_vectores_documento(textos_norm, vectores_tipos, idf)
        registro["n_tipos_con_vector"] = len(vectores_tipos)
        registro["n_tipos_consultados"] = len(tipos_unicos)

    # Downstream
    log("3/6 downstream: 3 poolings x 2 clasificadores x 2 modelos")
    oof_v03_pred = oof["pred"]
    resultados: list[dict[str, Any]] = []
    # Suelo de referencia: el mismo clasificador sobre vectores **aleatorios** de la
    # misma dimension. Sin el, un 0.66 no es interpretable: parte de esa cifra es
    # el previo de clases, no senal del embedding.
    vectores_documento["SUELO-ALEATORIO"] = {
        "A_media": np.random.default_rng(20260925).standard_normal(
            (len(textos_norm), ARQUITECTURA["dim"])
        ).astype(np.float32)
    }
    for nombre in ("FASTTEXT-BASELINE", "WORD2VEC-CONTROL", "SUELO-ALEATORIO"):
        for pooling in vectores_documento[nombre]:
            matriz = vectores_documento[nombre][pooling]
            for clasificador in ("LR", "SVC"):
                etiqueta = f"{nombre}|{pooling}|{clasificador}"
                evaluacion = evaluar_downstream(etiqueta, matriz, y, pliegues, escalador=True)
                comparacion = bootstrap_pareado(y, evaluacion["pred_ajustados"], oof_v03_pred)
                etiqueta_f1 = evaluacion["f1_por_etiqueta_ajustados"]
                fila = {
                    "modelo": nombre,
                    "pooling": pooling,
                    "clasificador": clasificador,
                    "dim": int(matriz.shape[1]),
                    "segundos_modelo": modelos[nombre]["segundos"] if nombre in modelos else 0,
                    "corrected_cv_umbral_0.5": evaluacion["corrected_cv_umbral_0.5"],
                    "corrected_cv_umbrales_ajustados": evaluacion["corrected_cv_umbrales_ajustados"],
                    "delta_vs_v03_0.5": round(evaluacion["corrected_cv_umbral_0.5"] - 0.771778476256122, 6),
                    "delta_vs_v03_ajustados": round(
                        evaluacion["corrected_cv_umbrales_ajustados"] - float(oof["corrected_cv"][0]), 6
                    ),
                    "ic95_ajustados": comparacion["ic95"],
                    "pliegues_mejorados": comparacion["filas_mejoradas"],
                    "filas_mejoradas": comparacion["filas_mejoradas"],
                    "filas_empeoradas": comparacion["filas_empeoradas"],
                    "std_pliegues_0.5": evaluacion["std_pliegues_0.5"],
                    "umbrales_ajustados": evaluacion["umbrales_ajustados"],
                    "f1_HS": round(etiqueta_f1["HS"], 4),
                    "f1_TR": round(etiqueta_f1["TR"], 4),
                    "f1_AG": round(etiqueta_f1["AG"], 4),
                }
                resultados.append(fila)
                log(
                    f"   {nombre:20s} {pooling:12s} {clasificador}: "
                    f"0.5={fila['corrected_cv_umbral_0.5']:.4f} "
                    f"ajustado={fila['corrected_cv_umbrales_ajustados']:.4f} "
                    f"delta={fila['delta_vs_v03_ajustados']:+.4f} "
                    f"IC95={fila['ic95_ajustados']}"
                )

    # Intrinseca
    log("4/6 intrinseca: vecinos, estabilidad y variantes")
    intrinseca: dict[str, Any] = {"vecinos": {}, "variantes": [], "estabilidad": {}}
    matrices: dict[str, tuple[dict[str, int], np.ndarray]] = {}
    for nombre in ("FASTTEXT-BASELINE", "WORD2VEC-CONTROL", "FASTTEXT-SEED-CONTROL"):
        prefijo = EMB_DIR / nombre.lower().replace("-", "_")
        matrices[nombre] = cargar_matriz_vec(Path(f"{prefijo}.vec"))
    intrinseca["vecinos"]["FASTTEXT-BASELINE"] = vecinos_centrados(
        matrices["FASTTEXT-BASELINE"][1], matrices["FASTTEXT-BASELINE"][0], PALABRAS_INTRINSECAS
    )
    intrinseca["vecinos"]["WORD2VEC-CONTROL"] = vecinos_centrados(
        matrices["WORD2VEC-CONTROL"][1], matrices["WORD2VEC-CONTROL"][0], PALABRAS_INTRINSECAS
    )
    intrinseca["estabilidad"] = estabilidad_entre_semillas(
        matrices["FASTTEXT-BASELINE"][0],
        matrices["FASTTEXT-BASELINE"][1],
        matrices["FASTTEXT-SEED-CONTROL"][0],
        matrices["FASTTEXT-SEED-CONTROL"][1],
        PALABRAS_INTRINSECAS,
    )
    vocabulario_toxic = vocabulario_textos([normalizar_emb_normal(texto) for texto in todos])
    variantes_tipos = [tipo for tipo in VARIANTES_ORTOGRAFICAS if tipo in vocabulario_toxic]
    indice_ft, matriz_ft = matrices["FASTTEXT-BASELINE"]
    indice_w2v, matriz_w2v = matrices["WORD2VEC-CONTROL"]
    centro_ft, centrada_ft = centrar(matriz_ft)
    centro_w2v, centrada_w2v = centrar(matriz_w2v)
    lista_ft = [None] * len(indice_ft)
    for palabra, posicion in indice_ft.items():
        lista_ft[posicion] = palabra
    lista_w2v = [None] * len(indice_w2v)
    for palabra, posicion in indice_w2v.items():
        lista_w2v[posicion] = palabra
    for tipo in variantes_tipos:
        entrada: dict[str, Any] = {"tipo": tipo, "freq_toxic": vocabulario_toxic[tipo]}
        for etiqueta, (indice, centrada, centro, lista) in {
            "fasttext": (indice_ft, centrada_ft, centro_ft, lista_ft),
            "word2vec": (indice_w2v, centrada_w2v, centro_w2v, lista_w2v),
        }.items():
            if tipo not in indice:
                entrada[etiqueta] = "(sin vector: no esta en el vocabulario y no hay subpalabras)"
                continue
            consulta = (matriz_ft if etiqueta == "fasttext" else matriz_w2v)[indice[tipo]] - centro
            consulta = consulta / max(float(np.linalg.norm(consulta)), 1e-9)
            similitudes = centrada @ consulta
            posicion = int(np.argmax(similitudes))
            entrada[etiqueta] = f"`{lista[posicion]}` ({float(similitudes[posicion]):.3f})"
        intrinseca["variantes"].append(entrada)

    # Figura
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    nombres = [f"{fila['modelo'].split('-')[0]}/{fila['pooling'][0]}/{fila['clasificador']}" for fila in resultados]
    valores = [fila["corrected_cv_umbrales_ajustados"] for fila in resultados]
    colores = ["#1f77b4" if valor >= float(oof["corrected_cv"][0]) else "#d62728" for valor in valores]
    plt.bar(range(len(valores)), valores, color=colores)
    plt.axhline(float(oof["corrected_cv"][0]), color="black", linestyle="--", label="V03 0.777211")
    plt.axhline(0.771778476256122, color="grey", linestyle=":", label="V03 @0.5 0.771778")
    plt.xticks(range(len(nombres)), nombres, rotation=35, ha="right", fontsize=8)
    plt.ylabel("corrected_cv (umbrales ajustados)")
    plt.title("Embedding desde cero (10M) vs referencia V03")
    plt.legend()
    plt.tight_layout()
    ruta_figura = FIGURAS / "fig_43_downstream_embedding_v5.png"
    plt.savefig(ruta_figura, dpi=150)
    plt.close()

    escribir_informes(modelos, resultados, intrinseca, equivalencia)
    log(f"   informes: {REPORTE_INTRINSECA.name}, {REPORTE_DOWNSTREAM.name}, {REPORTE_DOWNSTREAM_CSV.name}")

    mejor = max(resultados, key=lambda fila: fila["corrected_cv_umbrales_ajustados"])
    log("5/6 veredicto")
    log(f"   mejor combinacion: {mejor['modelo']} {mejor['pooling']} {mejor['clasificador']} = "
        f"{mejor['corrected_cv_umbrales_ajustados']:.4f} (delta {mejor['delta_vs_v03_ajustados']:+.4f})")
    veredicto = {
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duracion_s": round(time.time() - inicio, 1),
        "referencia_v03": float(oof["corrected_cv"][0]),
        "mejor_combinacion": mejor,
        "ranking": [
            {
                "combinacion": f"{fila['modelo']}|{fila['pooling']}|{fila['clasificador']}",
                "corrected_cv_ajustados": fila["corrected_cv_umbrales_ajustados"],
                "delta": fila["delta_vs_v03_ajustados"],
            }
            for fila in sorted(resultados, key=lambda f: -f["corrected_cv_umbrales_ajustados"])
        ],
    }
    (PROJECT_ROOT / "reports" / "embedding_p2_verdict.json").write_text(
        json.dumps(veredicto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(f"P2 completada en {veredicto['duracion_s']} s")


if __name__ == "__main__":
    main()
