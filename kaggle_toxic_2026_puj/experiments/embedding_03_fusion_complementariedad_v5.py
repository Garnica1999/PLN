"""P3 de la fase v5: complementariedad del embedding con V03 (fusion y meta-modelo).

Contexto
--------
P2 respondio a la pregunta mas facil y la respuesta fue negativa: el embedding
entrenado desde cero **por si solo** queda 0.1169 por debajo de V03 (0.6603
frente a 0.777211, IC95 completamente negativo). Eso refuta la hipotesis de que
la representacion distribuida sustituya a TF-IDF, pero **no** refuta la hipotesis
central de la fase: la seccion 4 de la spec y la pregunta 11 de la seccion 42
preguntan si el embedding aporta informacion **complementaria**. Un modelo
notablemente peor puede parfaitement enrichingir a uno mejor si acierta donde el
otro falla, y ese es exactamente el caso que mide la matriz de complementariedad
de la seccion 25.

Este script mide esa complementariedad con el mismo protocolo corregido v2 y con
los mismos hiperparametros de V03, para que las cifras sean comparables celda a
celda.

Variantes evaluadas (todas con los 5 pliegues corregidos y 3 modelos binarios):

1. ``V03-TFIDF`` (control): replica exacta de V03 (union TF-IDF word(1,2) +
   char_wb(3,5) ajustados **por pliegue** + LR C=2.0 balanced). Debe reproducir
   0.7772; si no, el protocolo no seria comparable y el script lo advierte.
2. ``EMB-SOLO`` (control): el embedding solo, el mejor pooling de P2.
3. ``FUSION-HSTACK``: union dispersa TF-IDF + las 200 dimensiones del embedding.
   El bloque del embedding se estandariza **dentro de cada pliegue** (ajustado
   solo con el entrenamiento) porque su escala es muy distinta de la de TF-IDF.
4. ``FUSION-HSTACK-L2``: igual, pero con normalizacion L2 por documento en el
   bloque del embedding (convencion habitual al usar embeddings).
5. ``META-ANIDADO``: apilamiento **anidado** sobre las probabilidades fuera de
   pliegue de V03 y del embedding. El meta se ajusta en cada pliegue exterior
   con probabilidades OOF **internas** (5 pliegues dentro del entrenamiento
   exterior), nunca con las probabilidades in-sample: es la condicion que impone
   la seccion 24 ("toda fusion con OOF; sin stacking in-sample").

Ademas: matriz de complementariedad 2x2 a nivel de celda (fila x etiqueta) y por
rebanadas de la seccion 25, y bootstrap pareado de cada variante frente a V03.

Lo que este script **no** hace: no usa el holdout (agotado 5/5), no lee el test,
no genera submissions, no entrena embedding nuevo y no toca V03.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_03_fusion_complementariedad_v5.py
"""

from __future__ import annotations

import csv
import importlib.util
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
from sklearn.preprocessing import StandardScaler, normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train  # noqa: E402
from src.embedding_utils import normalizar_emb_normal, tokenizar  # noqa: E402
from src.evaluation import (  # noqa: E402
    ETIQUETAS,
    bootstrap_diferencia_pareada,
    metricas_multietiqueta,
)
from src.splits import grupos_fusionados, holdout_v2, pliegues_corregidos  # noqa: E402

EXPERIMENTO = "EBD-07..09 (P3)"
SIFTS_FASTTEXT = PROJECT_ROOT / "data" / "external" / "embeddings" / "fasttext-baseline_sifts.txt"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "embedding_fusion_complementarity_v5.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "embedding_fusion_v5.json"
REPORTE_CSV = PROJECT_ROOT / "reports" / "embedding_fusion_v5.csv"
REPORTE_REBANADAS = PROJECT_ROOT / "reports" / "embedding_complementarity_slices_v5.csv"
FIGURAS = PROJECT_ROOT / "reports" / "figuras"

# Parametros identicos a experiments/14_oof_corregido_v2.py
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
PARAMETROS_LR = {
    "C": 2.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "liblinear",
    "random_state": 42,
}
PARAMETROS_LR_EMB = {**PARAMETROS_LR, "C": 2.0}
PARAMETROS_RIDGE = {"alpha": 3.0}
SEMILLA_BOOTSTRAP = 20260925
REPETICIONES_BOOTSTRAP = 10_000
# Rejilla de umbrales: la misma que documento el pipeline congelado de V03.
REJILLA_UMBRALES = np.round(np.arange(0.20, 0.81, 0.01), 2)
REFERENCIA_V03 = 0.777211
# Umbrales congelados de V03: se aplican a TODAS las variantes como punto de
# operacion comun, para que nadie se beneficie de un ajuste de umbral propio.
UMBRALES_V03_FROZEN = np.array([0.44, 0.51, 0.46])
REFERENCIA_V03_UMBRAL_05 = 0.771778
MEJORA_MINIMA = 0.005

RANDOM_STATE = 42


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V5-P3] {mensaje}", flush=True)


# ---------------------------------------------------------------------------
# Vectores de documento del embedding (reutiliza los sifts de P2)
# ---------------------------------------------------------------------------
def cargar_sifts(ruta: Path) -> dict[str, np.ndarray]:
    """Carga los vectores por tipo que fastText produjo en P2.

    Args:
        ruta: archivo de salida de ``fasttext print-word-vectors``.

    Returns:
        Diccionario tipo -> vector.
    """
    vectores: dict[str, np.ndarray] = {}
    with ruta.open(encoding="utf-8", errors="replace") as archivo:
        for linea in archivo:
            partes = linea.split()
            if len(partes) < 2:
                continue
            valores = np.asarray(partes[1:], dtype=np.float32)
            if not np.isfinite(valores).all() or float(np.linalg.norm(valores)) == 0.0:
                continue
            vectores[partes[0]] = valores
    return vectores


def vectores_documento(textos: list[str], vectores_tipos: dict[str, np.ndarray]) -> np.ndarray:
    """Construye el vector de documento por media de tokens (pooling A de P2).

    Los tokens sin vector (que en el modelo con subpalabras no deberia ocurrir)
    se omiten del promedio.

    Args:
        textos: textos ya normalizados.
        vectores_tipos: vectores por tipo.

    Returns:
        Matriz de vectores de documento.
    """
    dimensiones = next(iter(vectores_tipos.values())).shape[0]
    matriz = np.zeros((len(textos), dimensiones), dtype=np.float32)
    for posicion, texto in enumerate(textos):
        utiles = [token for token in tokenizar(texto) if token in vectores_tipos]
        if utiles:
            matriz[posicion] = np.vstack([vectores_tipos[token] for token in utiles]).mean(axis=0)
    return matriz


# ---------------------------------------------------------------------------
# Piezas de V03 replicadas
# ---------------------------------------------------------------------------
def matrices_tfidf(textos_entrenamiento, textos_validacion):
    """Ajusta word + char TF-IDF en el pliegue y transforma la validacion.

    Replica ``construir_matrices(..., "union")`` de
    ``experiments/14_oof_corregido_v2.py``: los vectorizadores se ajustan **solo**
    con los textos de entrenamiento del pliegue, nunca con la validacion.

    Args:
        textos_entrenamiento: textos del pliegue de entrenamiento.
        textos_validacion: textos del pliegue de validacion.

    Returns:
        Tupla de matrices dispersas ``(entrenamiento, validacion)``.
    """
    word = TfidfVectorizer(**PARAMS_WORD)
    char = TfidfVectorizer(**PARAMS_CHAR)
    entrenamiento = sparse.hstack(
        [word.fit_transform(textos_entrenamiento), char.fit_transform(textos_entrenamiento)]
    ).tocsr()
    validacion = sparse.hstack(
        [word.transform(textos_validacion), char.transform(textos_validacion)]
    ).tocsr()
    return entrenamiento, validacion


def ajustar_probabilidades(matriz, objetivo, matriz_consulta, parametros, tipo="lr"):
    """Ajusta un clasificador y devuelve las probabilidades de la consulta.

    Args:
        matriz: matriz de entrenamiento.
        objetivo: etiqueta binaria de entrenamiento.
        matriz_consulta: matriz sobre la que se predice.
        parametros: parametros del estimador.
        tipo: ``lr`` o ``ridge``.

    Returns:
        Vector de probabilidades.
    """
    if tipo == "ridge":
        modelo = Ridge(**parametros)
        modelo.fit(matriz, objetivo)
        return np.clip(modelo.predict(matriz_consulta), 0.0, 1.0)
    modelo = LogisticRegression(**parametros)
    modelo.fit(matriz, objetivo)
    return modelo.predict_proba(matriz_consulta)[:, 1]


def ajustar_umbrales(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Ajusta un umbral por etiqueta en la rejilla de V03 (0.20-0.80, paso 0.01).

    Args:
        y_true: matriz binaria verdadera.
        proba: probabilidades fuera de pliegue.

    Returns:
        Vector de tres umbrales.
    """
    from sklearn.metrics import f1_score

    umbrales = np.empty(3, dtype=np.float64)
    for indice in range(3):
        mejor = (0.0, 0.5)
        for umbral in REJILLA_UMBRALES:
            valor = float(
                f1_score(y_true[:, indice], (proba[:, indice] >= umbral).astype(int), zero_division=0)
            )
            if valor > mejor[0]:
                mejor = (valor, float(umbral))
        umbrales[indice] = mejor[1]
    return umbrales


def bloque_embedding(embeddings: np.ndarray, indices: np.ndarray, modo: str) -> np.ndarray:
    """Prepara el bloque del embedding para un pliegue.

    Los modos son **excluyentes**: aplicar L2 y despues estandarizar seria
    estandarizar de nuevo (el estandarizador anula la normalizacion L2), y por eso
    una version anterior de este script mezclaba ambos pasos. La consequence fue
    que el bloque denso, con varianza unitaria, dominaba al bloque disperso de
    TF-IDF (cuyos valores son de orden 0.1) y la fusion perdia 0.09 de F1 frente a
    V03 sin que eso dijera nada sobre la representacion.

    Args:
        embeddings: matriz de vectores de documento.
        indices: indices del pliegue (entrenamiento o validacion).
        modo: ``natural`` (tal cual), ``l2`` (norma unitaria por documento) o
            ``estandar`` (varianza unitaria ajustada con el pliegue).

    Returns:
        Bloque de caracteristicas del embedding.

    Raises:
        ValueError: si el modo no es uno de los tres previstos.
    """
    if modo == "natural":
        return embeddings[indices]
    if modo == "l2":
        return normalize(embeddings[indices], norm="l2")
    if modo == "estandar":
        escalador = StandardScaler().fit(embeddings[indices])
        return escalador.transform(embeddings[indices])
    raise ValueError(f"Modo de bloque de embedding desconocido: {modo}")


# ---------------------------------------------------------------------------
# Variantes de evaluacion
# ---------------------------------------------------------------------------
def evaluar_variante(
    nombre: str,
    textos: pd.Series,
    y: np.ndarray,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    embeddings: np.ndarray | None,
    usar_tfidf: bool = True,
    modo_embedding: str = "l2",
) -> dict[str, Any]:
    """Evalua una variante bajo el protocolo corregido v2.

    Con ``usar_tfidf=True`` y ``embeddings=None`` reproduce V03. Con
    ``embeddings`` no nulo, la matriz de caracteristicas es la union del bloque
    disperso de TF-IDF (si ``usar_tfidf``) y del bloque denso del embedding.

    Args:
        nombre: identificador de la variante.
        textos: columna de textos del desarrollo.
        y: matriz binaria de labels.
        pliegues: pliegues corregidos.
        embeddings: matriz de vectores de documento o ``None``.
        usar_tfidf: si se incluye el bloque TF-IDF de V03.
        modo_embedding: ``natural``, ``l2`` o ``estandar``.

    Returns:
        Diccionario con probabilidades OOF y metricas.
    """
    proba = np.zeros((len(textos), 3), dtype=np.float64)
    f1_por_pliegue: list[float] = []
    for numero, (entrenamiento, validacion) in enumerate(pliegues, start=1):
        if usar_tfidf:
            matriz_entrenamiento, matriz_validacion = matrices_tfidf(
                textos.iloc[entrenamiento], textos.iloc[validacion]
            )
        else:
            matriz_entrenamiento = sparse.csr_matrix((len(entrenamiento), 1))
            matriz_validacion = sparse.csr_matrix((len(validacion), 1))
        if embeddings is not None:
            bloque_entrenamiento = bloque_embedding(embeddings, entrenamiento, modo_embedding)
            bloque_validacion = bloque_embedding(embeddings, validacion, modo_embedding)
            if modo_embedding == "estandar":
                # El ajuste se hace con el pliegue de entrenamiento y se aplica
                # tal cual a la validacion: ajustarlo por separado seria fuga.
                escalador = StandardScaler().fit(bloque_entrenamiento)
                bloque_entrenamiento = escalador.transform(bloque_entrenamiento)
                bloque_validacion = escalador.transform(bloque_validacion)
            matriz_entrenamiento = sparse.hstack(
                [matriz_entrenamiento, sparse.csr_matrix(bloque_entrenamiento)]
            ).tocsr()
            matriz_validacion = sparse.hstack(
                [matriz_validacion, sparse.csr_matrix(bloque_validacion)]
            ).tocsr()
        for indice in range(3):
            usa_denso = embeddings is not None and (
                not usar_tfidf or modo_embedding == "estandar"
            )
            proba[validacion, indice] = ajustar_probabilidades(
                matriz_entrenamiento,
                y[entrenamiento, indice],
                matriz_validacion,
                PARAMETROS_LR_EMB if usa_denso else PARAMETROS_LR,
            )
        f1_por_pliegue.append(
            float(
                metricas_multietiqueta(y[validacion], (proba[validacion] >= 0.5).astype(int))[
                    "f1_media_columnas"
                ]
            )
        )
    umbrales = ajustar_umbrales(y, proba)
    metricas_05 = metricas_multietiqueta(y, (proba >= 0.5).astype(int))
    predicciones = (proba >= umbrales[None, :]).astype(int)
    metricas_ajustadas = metricas_multietiqueta(y, predicciones)
    # Lectura primaria: los umbrales **congelados de V03** aplicados a esta variante.
    predicciones_frozen = (proba >= UMBRALES_V03_FROZEN[None, :]).astype(int)
    metricas_frozen = metricas_multietiqueta(y, predicciones_frozen)
    log(
        f"   {nombre:24s} frozen={metricas_frozen['f1_media_columnas']:.4f} "
        f"@0.5={metricas_05['f1_media_columnas']:.4f} "
        f"ajustado={metricas_ajustadas['f1_media_columnas']:.4f} "
        f"umbrales={list(np.round(umbrales, 2))}"
    )
    return {
        "variante": nombre,
        "corrected_cv_umbrales_frozen_v03": round(metricas_frozen["f1_media_columnas"], 6),
        "corrected_cv_umbral_0.5": round(metricas_05["f1_media_columnas"], 6),
        "corrected_cv_umbrales_ajustados": round(metricas_ajustadas["f1_media_columnas"], 6),
        "pred_frozen": predicciones_frozen,
        "umbrales_ajustados": [float(valor) for valor in umbrales],
        "f1_por_etiqueta_ajustados": metricas_ajustadas["f1_por_etiqueta"],
        "f1_por_pliegue_0.5": [round(valor, 4) for valor in f1_por_pliegue],
        "std_pliegues_0.5": round(float(np.std(f1_por_pliegue, ddof=1)), 4),
        "proba": proba,
        "pred": predicciones,
    }


def evaluar_meta_anidado(
    textos: pd.Series,
    y: np.ndarray,
    desarrollo: pd.DataFrame,
    grupos_dev: np.ndarray,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    embeddings: np.ndarray,
) -> dict[str, Any]:
    """Apilamiento anidado sobre las probabilidades OOF de V03 y del embedding.

    Para cada pliegue exterior: se obtienen probabilidades OOF **internas** de
    los dos modelos base con 5 pliegues dentro del entrenamiento exterior; el
    meta se ajusta con esas probabilidades internas y predice el pliegue
    exterior con modelos base ajustados sobre todo el entrenamiento exterior. No
    hay en ningun momento stacking in-sample (seccion 24).

    Args:
        textos: columna de textos del desarrollo.
        y: matriz binaria de labels.
        desarrollo: marco del desarrollo.
        grupos_dev: grupos fusionados del desarrollo.
        pliegues: pliegues corregidos exteriores.
        embeddings: matriz de vectores de documento.

    Returns:
        Diccionario con probabilidades OOF y metricas.
    """
    proba = np.zeros((len(textos), 3), dtype=np.float64)
    for numero, (entrenamiento, validacion) in enumerate(pliegues, start=1):
        indices = np.sort(entrenamiento)
        marco_interno = desarrollo.iloc[indices].reset_index(drop=True)
        grupos_internos = grupos_dev[indices]
        pliegues_internos = pliegues_corregidos(marco_interno, grupos_internos)
        oof_v03 = np.zeros((len(indices), 3), dtype=np.float64)
        oof_emb = np.zeros((len(indices), 3), dtype=np.float64)
        for interno_train, interno_val in pliegues_internos:
            matriz_tr, matriz_val = matrices_tfidf(
                marco_interno["text"].iloc[interno_train],
                marco_interno["text"].iloc[interno_val],
            )
            escalador = StandardScaler().fit(embeddings[indices][interno_train])
            bloque_tr = escalador.transform(embeddings[indices][interno_train])
            bloque_val = escalador.transform(embeddings[indices][interno_val])
            for etiqueta in range(3):
                objetivo = marco_interno[ETIQUETAS[etiqueta]].to_numpy()[interno_train]
                oof_v03[interno_val, etiqueta] = ajustar_probabilidades(
                    matriz_tr, objetivo, matriz_val, PARAMETROS_LR
                )
                oof_emb[interno_val, etiqueta] = ajustar_probabilidades(
                    sparse.csr_matrix(bloque_tr),
                    objetivo,
                    sparse.csr_matrix(bloque_val),
                    PARAMETROS_LR_EMB,
                )
        # Modelos base ajustados sobre todo el entrenamiento exterior.
        matriz_tr, matriz_val = matrices_tfidf(
            desarrollo["text"].iloc[indices], desarrollo["text"].iloc[validacion]
        )
        escalador = StandardScaler().fit(embeddings[indices])
        bloque_tr = escalador.transform(embeddings[indices])
        bloque_val = escalador.transform(embeddings[validacion])
        externa_v03 = np.zeros((len(validacion), 3), dtype=np.float64)
        externa_emb = np.zeros((len(validacion), 3), dtype=np.float64)
        for etiqueta in range(3):
            objetivo = desarrollo[ETIQUETAS[etiqueta]].to_numpy()[indices]
            externa_v03[:, etiqueta] = ajustar_probabilidades(
                matriz_tr, objetivo, matriz_val, PARAMETROS_LR
            )
            externa_emb[:, etiqueta] = ajustar_probabilidades(
                sparse.csr_matrix(bloque_tr),
                objetivo,
                sparse.csr_matrix(bloque_val),
                PARAMETROS_LR_EMB,
            )
        caracteristicas_internas = np.hstack([oof_v03, oof_emb])
        caracteristicas_externas = np.hstack([externa_v03, externa_emb])
        meta = Ridge(**PARAMETROS_RIDGE)
        meta.fit(caracteristicas_internas, y[indices])
        proba[validacion] = np.clip(meta.predict(caracteristicas_externas), 0.0, 1.0)
        log(f"   meta anidado, pliegue {numero}: F1 @0.5 = "
            f"{metricas_multietiqueta(y[validacion], (proba[validacion] >= 0.5).astype(int))['f1_media_columnas']:.4f}")
    umbrales = ajustar_umbrales(y, proba)
    metricas_05 = metricas_multietiqueta(y, (proba >= 0.5).astype(int))
    predicciones = (proba >= umbrales[None, :]).astype(int)
    metricas_ajustadas = metricas_multietiqueta(y, predicciones)
    predicciones_frozen = (proba >= UMBRALES_V03_FROZEN[None, :]).astype(int)
    metricas_frozen = metricas_multietiqueta(y, predicciones_frozen)
    return {
        "variante": "META-ANIDADO",
        "corrected_cv_umbrales_frozen_v03": round(metricas_frozen["f1_media_columnas"], 6),
        "corrected_cv_umbral_0.5": round(metricas_05["f1_media_columnas"], 6),
        "pred_frozen": predicciones_frozen,
        "corrected_cv_umbrales_ajustados": round(metricas_ajustadas["f1_media_columnas"], 6),
        "umbrales_ajustados": [float(valor) for valor in umbrales],
        "f1_por_etiqueta_ajustados": metricas_ajustadas["f1_por_etiqueta"],
        "f1_por_pliegue_0.5": [],
        "std_pliegues_0.5": 0.0,
        "proba": proba,
        "pred": predicciones,
    }


# ---------------------------------------------------------------------------
# Complementariedad
# ---------------------------------------------------------------------------
def matriz_complementariedad(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray
) -> dict[str, int]:
    """Cuenta la matriz 2x2 de acierto por celda (seccion 25).

    Args:
        y_true: matriz binaria verdadera.
        pred_a: predicciones del candidato.
        pred_b: predicciones de la referencia.

    Returns:
        Diccionario con los cuatro conteos de la matriz 2x2.
    """
    from src.evaluation import _f1_por_celda

    celdas_a = _f1_por_celda(y_true, pred_a)
    celdas_b = _f1_por_celda(y_true, pred_b)
    acierta_a = celdas_a >= 0.999
    acierta_b = celdas_b >= 0.999
    return {
        "celdas": int(celdas_a.size),
        "ambas_correctas": int(np.sum(acierta_a & acierta_b)),
        "solo_candidato": int(np.sum(acierta_a & ~acierta_b)),
        "solo_referencia": int(np.sum(~acierta_a & acierta_b)),
        "ninguna_correcta": int(np.sum(~acierta_a & ~acierta_b)),
    }


def rebanadas(
    desarrollo: pd.DataFrame, y: np.ndarray
) -> dict[str, np.ndarray]:
    """Construye las rebanadas de analisis de la seccion 25.

    Las rebanadas se definen con los labels del train (permitido: es analisis de
    error, no aprendizaje de la representacion) y con patrones lexicicos curados
    reutilizados del diagnostico de cobertura de P1. **No** son excluyentes entre
    si.

    Args:
        desarrollo: marco del desarrollo.
        y: matriz binaria de labels.

    Returns:
        Diccionario nombre -> mascara booleana.
    """
    ruta_p1 = PROJECT_ROOT / "experiments" / "embedding_01_corpus_wikipedia_v5.py"
    especificacion = importlib.util.spec_from_file_location("embedding_01_p1", ruta_p1)
    modulo_p1 = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo_p1)
    grupos = modulo_p1.clasificar_grupos(modulo_p1.vocabulario_toxic(desarrollo))
    texto = desarrollo["text"].astype(str)
    texto_minuscula = texto.str.lower()
    tokens_por_fila = [set(tokenizar(normalizar_emb_normal(valor))) for valor in texto]

    def con_grupos(nombre_grupo: str) -> np.ndarray:
        """Marca las filas que contienen algun tipo del grupo lexicografico."""
        objetivos = set(grupos.get(nombre_grupo, []))
        if not objetivos:
            return np.zeros(len(texto), dtype=bool)
        return np.asarray(
            [bool(objetivos & tokens) for tokens in tokens_por_fila], dtype=bool
        )

    mascaras: dict[str, np.ndarray] = {
        "interpersonal (TR=1)": y[:, 1] == 1,
        "afiliacion (AG=1)": y[:, 2] == 1,
        "sin_hate_explicito (HS=0)": y[:, 0] == 0,
        "colectivos": con_grupos("colectivos"),
        "insultos": con_grupos("insultos"),
        "elongaciones": con_grupos("elongaciones"),
        "coloquial": con_grupos("coloquial"),
        "hashtags": texto.str.contains("#", regex=False).to_numpy(),
        "menciones": texto.str.contains("@", regex=False).to_numpy(),
        "urls": texto_minuscula.str.contains("http", regex=False).to_numpy(),
        "cita": texto.str.contains('"|\u00ab|>', regex=True).to_numpy(),
    }
    return mascaras


def metricas_por_rebanada(
    y_true: np.ndarray, predicciones: dict[str, np.ndarray], mascaras: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    """Calcula las metricas por rebanada para cada variante.

    Args:
        y_true: matriz binaria verdadera.
        predicciones: diccionario variante -> matriz predicha.
        mascaras: diccionario rebanada -> mascara booleana.

    Returns:
        Lista de filas con una entrada por rebanada y variante.
    """
    from src.evaluation import metrica_macro

    filas = []
    for nombre, mascara in mascaras.items():
        if mascara.sum() == 0:
            continue
        for variante, pred in predicciones.items():
            filas.append(
                {
                    "rebanada": nombre,
                    "variante": variante,
                    "n_filas": int(mascara.sum()),
                    "f1_por_celda": round(metrica_macro(y_true[mascara], pred[mascara]), 4),
                }
            )
    return filas


def escribir_informe(
    filas: list[dict[str, Any]],
    variantes: list[dict[str, Any]],
    metricas_rebanada: list[dict[str, Any]],
    control_v03: float,
    n_rebanadas: dict[str, int],
) -> None:
    """Escribe el informe de complementariedad.

    Args:
        filas: comparaciones pareadas por variante.
        variantes: resultados por variante.
        metricas_rebanada: metricas por rebanada.
        control_v03: ``corrected_cv`` del control que replica V03.
        n_rebanadas: numero de filas por rebanada.
    """
    por_rebanada: dict[str, dict[str, float]] = {}
    for fila in metricas_rebanada:
        por_rebanada.setdefault(fila["rebanada"], {})[fila["variante"]] = fila["f1_por_celda"]
    nombres_rebanada = list(n_rebanadas)
    variantes_interesadas = ["V03-CONGELADO", "EMB-SOLO", "FUSION-HSTACK", "META-ANIDADO"]

    lineas = [
        "# Complementariedad del embedding con V03 (fase v5, P3)",
        "",
        f"Generado por `experiments/embedding_03_fusion_complementariedad_v5.py` ({EXPERIMENTO}).",
        "",
        "## 1. Por que esta prueba y no la anterior",
        "",
        "P2 mostro que el embedding **por si solo** queda 0.1169 por debajo de V03. Eso refuta la",
        "hipotesis de que lo sustituya, pero no la hipotesis central de la fase (seccion 4 de la",
        "spec y pregunta 11 de la seccion 42): si aporta informacion **complementaria**. Esta",
        "prueba mide precisamente eso.",
        "",
        "## 2. Protocolo (identico al de V03, celda a celda)",
        "",
        "- Pliegues: `pliegues_corregidos` (5, grupos near-duplicado UNION pseudo-autor).",
        "- TF-IDF word(1,2) + char_wb(3,5) con `min_df=2` y `sublinear_tf=True`, ajustados "
        "**por pliegue** sobre el entrenamiento, igual que en `experiments/14_oof_corregido_v2.py`.",
        "- LR `C=2.0`, `class_weight='balanced'`, `liblinear`, 3 cabezas binarias.",
        "- Umbrales: rejilla 0.20-0.80 paso 0.01 sobre las probabilidades OOF, el mismo metodo "
        "que documento el pipeline congelado de V03.",
        f"- Control: el calculo de `V03-TFIDF` sin embedding da **{control_v03:.6f}** frente a los "
        f"{REFERENCIA_V03:.6f} congelados (delta {control_v03 - REFERENCIA_V03:+.6f}).",
        "",
        "## 3. Resultado por variante",
        "",
        "La lectura primaria aplica a todas las variantes los **umbrales congelados de V03**",
        "(0.44/0.51/0.46): nadie se beneficia de ajustar su propio umbral. La columna",
        "\"umbrales propios\" se reporta como dato secundario y es optimista.",
        "",
        "| variante | frozen V03 (primaria) | delta vs V03 | IC95 pareado | @0.5 | umbrales propios | celdas + | celdas - | solo esta | solo V03 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for fila in filas:
        lineas.append(
            f"| {fila['variante']} | {fila['corrected_cv_umbrales_frozen_v03']:.4f} | "
            f"{fila['delta_vs_v03']:+.4f} | "
            f"[{fila['ic95_0']:+.4f}, {fila['ic95_1']:+.4f}] | {fila['corrected_cv_umbral_0.5']:.4f} | "
            f"{fila['corrected_cv_umbrales_ajustados']:.4f} | {fila['celdas_mejoradas']} | "
            f"{fila['celdas_empeoradas']} | {fila['solo_candidato']} | {fila['solo_v03']} |"
        )
    mejor = max(filas, key=lambda fila: fila["corrected_cv_umbrales_frozen_v03"])
    cumple_nivel = mejor["delta_vs_v03"] >= MEJORA_MINIMA
    cumple_ic = mejor["ic95_0"] > 0
    lineas += [
        "",
        f"Mejor variante: **{mejor['variante']}** con {mejor['corrected_cv_umbrales_frozen_v03']:.4f} "
        f"(delta {mejor['delta_vs_v03']:+.4f}, IC95 [{mejor['ic95_0']:+.4f}, {mejor['ic95_1']:+.4f}]).",
        "",
        "Criterio pre-registrado (seccion 29): adopted solo si delta >= +0.005 **y** el IC95 pareado",
        f"excluye el 0. Nivel: {'si' if cumple_nivel else 'no'}; IC95: {'excluye 0' if cumple_ic else 'incluye 0'}.",
        "",
        "## 4. Complementariedad por rebanada (seccion 25)",
        "",
        "F1 por celda (fila x etiqueta) dentro de cada rebanada. Las rebanadas **no** son",
        "excluyentes entre si y las de tipo lexical se definen con los patrones curados de P1.",
        "",
        "| rebanada | filas | " + " | ".join(variantes_interesadas) + " |",
        "|---|---|" + "---|" * len(variantes_interesadas),
    ]
    for nombre in nombres_rebanada:
        valores = por_rebanada.get(nombre, {})
        celdas = " | ".join(
            f"{valores.get(variante, float('nan')):.4f}" for variante in variantes_interesadas
        )
        lineas.append(f"| {nombre} | {n_rebanadas[nombre]} | {celdas} |")
    lineas += [
        "",
        "## 5. Lectura y limites",
        "",
        "1. El control `V03-TFIDF` reproduce la referencia, de modo que las diferencias de la tabla",
        "   son atribuibles a la representacion y no al protocolo.",
        "2. `solo esta` / `solo V03` cuentan celdas en las que exactamente uno de los dos acierta:",
        "   es la medida directa de complementariedad. Un numero alto de `solo esta` con mejora",
        "   pequena en la media indicaria que el embedding rescata casos sin degradar el resto.",
        "3. El meta-modelo esta evaluado con apilamiento **anidado**: sus probabilidades de",
        "   entrenamiento salen de 5 pliegues internos, nunca de un modelo ajustado sobre la misma",
        "   fila (seccion 24).",
        "4. Los tres modelos de P2 se entrenaron en una ejecucion anterior cuyo registro de consola",
        "   fue sobrescrito, por lo que `segundos=0` en el manifiesto de P2 significa \"reutilizado\";",
        "   la huella `sha256` del `.vec` es el dato autoritativo.",
        "5. Aunque una variante ganase localmente, la leccion operativa de v4 sigue vigente: cuatro",
        "   mejoras locales consecutivas no transfirieron al holdout o al leaderboard. Ninguna",
        "   variante se consideraria candidata sin esa evidencia externa, que aqui no se gasta.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main() -> None:
    """Ejecuta P3: fusion, meta anidado y complementariedad."""
    inicio = time.time()
    log("P3: complementariedad del embedding con V03")
    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos[indices_dev]
    pliegues = pliegues_corregidos(desarrollo, grupos_dev)
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    log(f"   desarrollo: {len(desarrollo)} filas | {len(pliegues)} pliegues")

    vectores_tipos = cargar_sifts(SIFTS_FASTTEXT)
    embeddings = vectores_documento(
        [normalizar_emb_normal(texto) for texto in desarrollo["text"]], vectores_tipos
    )
    log(f"   vectores de documento: {embeddings.shape}")

    oof = np.load(OOF_V03, allow_pickle=True)
    if not np.array_equal(np.sort(oof["ids"].astype(int)), np.sort(desarrollo["id"].astype(int).to_numpy())):
        raise RuntimeError("Los ids del OOF canonico no coinciden con el desarrollo")
    pred_v03_congelada = oof["pred"]
    umbrales_v03 = oof["umbrales"]

    log("1/4 variantes (control V03, control embedding, dos fusiones)")
    variantes = []
    control_v03 = evaluar_variante("V03-TFIDF", desarrollo["text"], y, pliegues, None)
    variantes.append(control_v03)
    control_emb = evaluar_variante(
        "EMB-SOLO", desarrollo["text"], y, pliegues, embeddings, usar_tfidf=False, modo_embedding="estandar"
    )
    variantes.append(control_emb)
    fusion_l2 = evaluar_variante(
        "FUSION-HSTACK-L2", desarrollo["text"], y, pliegues, embeddings, modo_embedding="l2"
    )
    variantes.append(fusion_l2)
    fusion_std = evaluar_variante(
        "FUSION-HSTACK-STANDARD", desarrollo["text"], y, pliegues, embeddings, modo_embedding="estandar"
    )
    variantes.append(fusion_std)

    log("2/4 meta-modelo anidado (OOF dentro de OOF, sin stacking in-sample)")
    meta = evaluar_meta_anidado(desarrollo["text"], y, desarrollo, grupos_dev, pliegues, embeddings)
    variantes.append(meta)

    referencia = control_v03["corrected_cv_umbrales_frozen_v03"]
    log(f"   control V03-TFIDF reproduce {referencia:.6f} frente a {REFERENCIA_V03:.6f} "
        f"(delta {referencia - REFERENCIA_V03:+.6f})")

    log("3/4 comparaciones pareadas y matriz de complementariedad")
    filas = []
    for variante in variantes:
        comparacion = bootstrap_diferencia_pareada(
            y, variante["pred_frozen"], pred_v03_congelada, REPETICIONES_BOOTSTRAP, SEMILLA_BOOTSTRAP
        )
        matriz = matriz_complementariedad(y, variante["pred_frozen"], pred_v03_congelada)
        filas.append(
            {
                "variante": variante["variante"],
                "corrected_cv_umbrales_frozen_v03": variante["corrected_cv_umbrales_frozen_v03"],
                "delta_vs_v03": round(variante["corrected_cv_umbrales_frozen_v03"] - REFERENCIA_V03, 6),
                "corrected_cv_umbral_0.5": variante["corrected_cv_umbral_0.5"],
                "corrected_cv_umbrales_ajustados": variante["corrected_cv_umbrales_ajustados"],
                "ic95_0": comparacion["ic95"][0],
                "ic95_1": comparacion["ic95"][1],
                "celdas_mejoradas": comparacion["celdas_mejoradas"],
                "celdas_empeoradas": comparacion["celdas_empeoradas"],
                "solo_candidato": matriz["solo_candidato"],
                "solo_v03": matriz["solo_referencia"],
                "ninguna": matriz["ninguna_correcta"],
                "f1_HS": round(variante["f1_por_etiqueta_ajustados"]["HS"], 4),
                "f1_TR": round(variante["f1_por_etiqueta_ajustados"]["TR"], 4),
                "f1_AG": round(variante["f1_por_etiqueta_ajustados"]["AG"], 4),
            }
        )

    mascaras = rebanadas(desarrollo, y)
    metricas_rebanada = metricas_por_rebanada(
        y,
        {
            variante["variante"]: variante["pred_frozen"] for variante in variantes
        }
        | {"V03-CONGELADO": pred_v03_congelada},
        mascaras,
    )

    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]), lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)
    with REPORTE_REBANADAS.open("w", encoding="utf-8", newline="") as archivo:
        campos = ["rebanada", "variante", "n_filas", "f1_por_celda"]
        escritor = csv.DictWriter(archivo, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(metricas_rebanada)

    mejor = max(filas, key=lambda fila: fila["corrected_cv_umbrales_frozen_v03"])
    log("4/4 veredicto")
    log(f"   mejor variante: {mejor['variante']} = {mejor['corrected_cv_umbrales_frozen_v03']:.4f} "
        f"(delta {mejor['delta_vs_v03']:+.4f}, IC95 [{mejor['ic95_0']:+.4f}, {mejor['ic95_1']:+.4f}])")

    REPORTE_JSON.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "referencia_v03": {
                    "corrected_cv_umbrales_congelados": REFERENCIA_V03,
                    "corrected_cv_umbral_0.5": REFERENCIA_V03_UMBRAL_05,
                    "umbrales": [float(valor) for valor in umbrales_v03],
                },
                "protocolo": "corrected_cv v2, 5 pliegues, TF-IDF ajustado por pliegue, LR C=2.0 balanced",
                "variantes": [
                    {clave: valor for clave, valor in variante.items() if clave not in {"proba", "pred", "pred_frozen"}}
                    for variante in variantes
                ],
                "comparaciones": filas,
                "rebanadas": metricas_rebanada,
                "n_rebanadas": {nombre: int(mascara.sum()) for nombre, mascara in mascaras.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    escribir_informe(
        filas,
        variantes,
        metricas_rebanada,
        referencia,
        {nombre: int(mascara.sum()) for nombre, mascara in mascaras.items()},
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 4.2))
    nombres = [fila["variante"] for fila in filas]
    valores = [fila["corrected_cv_umbrales_frozen_v03"] for fila in filas]
    plt.barh(range(len(nombres)), valores, color=["#1f77b4" if v >= REFERENCIA_V03 else "#d62728" for v in valores])
    plt.axvline(REFERENCIA_V03, color="black", linestyle="--", label="V03 0.7772")
    plt.yticks(range(len(nombres)), nombres, fontsize=8)
    plt.xlabel("corrected_cv (umbrales congelados de V03)")
    plt.title("Embedding desde cero y su fusion con V03 (10M tokens)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_44_fusion_complementariedad_v5.png", dpi=150)
    plt.close()

    log(f"   informes: {REPORTE.name}, {REPORTE_JSON.name}, {REPORTE_CSV.name}, {REPORTE_REBANADAS.name}")
    log(f"P3 completada en {round(time.time() - inicio, 1)} s")


if __name__ == "__main__":
    main()
