# %% [markdown]
"""# Sec 03b — Metricas de ranking manuales: Okapi BM25 y Rocchio

Esta seccion implementa **desde cero** (sin librerias de BM25 ni de
relevance feedback) las dos metricas de ranking del punto 4 del taller,
sobre la misma direccion de matching del experimento base (ADR-005):
**2.486 entrevistas = queries** y **917 segmentos de libro = coleccion**.
Reutiliza el contexto de la Sec 03 (`contexto_base`): los textos limpios
(misma parada en consumo + correcciones de lemas) y la matriz TF-IDF
base, sin releer ni re-entrenar nada.

## 1. Okapi BM25 (Robertson y Walker, 1994)

```text
score(q, d) = sum_t IDF(t) * [ tf(t,d) * (k1 + 1) ]
              / [ tf(t,d) + k1 * (1 - b + b * |d| / avgdl) ]

IDF(t) = ln( (N - df(t) + 0,5) / (df(t) + 0,5) + 1 )      con k1 = 1,5 y b = 0,75
```

- **tf(t,d)**: frecuencia cruda del termino t en el documento d (sin
  sublinearidad): la formula logaritmica de TF-IDF sustituye aqui por la
  **saturacion de tf** del denominador, que crece rapido al principio y
  tiende asintoticamente a (k1 + 1).
- **k1**: controla la saturacion (1,5): cuanto mas alto, mas peso gana la
  repeticion del termino.
- **b**: controla la normalizacion por longitud (0,75): con b=1 se
  normaliza del todo por |d|/avgdl y con b=0 no se normaliza. Un segmento
  largo (VIOLACIONES/ETNICO, media ~1.300-1.400 tokens) no acumula ventaja
  artificial sobre uno corto (CONVOCATORIA, ~420) por tener mas palabras.
- **IDF(t)**: premia terminos raros (df bajo) y castiga los ubicuos; usa
  el suavizado de Robertson-Sparck Jones, distinto del `smooth_idf` de
  scikit-learn (que usa ln((N+1)/(df+1)) + 1).
- **|d| y avgdl**: longitud del documento y longitud media de la
  coleccion, medidas en **terminos del vocabulario** (unigramas y
  bigramas de la matriz tf; cada bigrama cuenta como un termino):
  avgdl = 1.466,26 en esta coleccion. No es el numero de lemas de
  contenido (1.007,5 de media tras la parada en consumo del IR; ver la
  nota de limitaciones al final de la seccion).

Por que difiere del TF-IDF con coseno: TF-IDF multiplica tf-idf(t,d) por
idf(t) con tf logaritmico y normaliza el vector completo con norma l2; no
usa ninguna referencia a la longitud del documento, asi que dos
documentos con la misma proporcion de terminos puntuan igual aunque uno
sea 10 veces mas largo. BM25 normaliza cada termino por |d|/avgdl ANTES
de agregar (y con el parametro b), lo que lo hace robusto al tamanio
desigual de los segmentos. Aqui el vocabulario (min_df=2, max_df=0,85,
unigramas + bigramas) y la limpieza son identicos a los del TF-IDF base:
la unica diferencia de la comparacion es la funcion de puntuacion.

El ranking se persiste en `resultados/ranking_bm25.csv` (salida del
notebook, no un insumo: se escribe en cada corrida). La matriz completa de
puntuaciones (2.486 x 917) se conserva en float32 (~9 MB) y se expone en
`contexto_base["matriz_bm25"]` para la Sec 03c (heatmap top-25, valor
maximo por entrevista-tomo); el ranking persistido no cambia.

## 2. Rocchio y retroalimentacion por pseudo-relevancia

```text
q' = alpha * q + beta * (1/|D_r|) * sum_{d en D_r} d
                - gamma * (1/|D_nr|) * sum_{d en D_nr} d        con alpha=1, beta=0,75, gamma=0,15
```

- **alpha**: peso de la consulta original (1,0): ancla la expansion para
  que q' no se aleje del tema de la entrevista.
- **beta**: peso del centroide de los documentos **relevantes** (0,75):
  refuerza los terminos que comparten los mejores resultados del ranking
  base (efecto de expansion de la consulta).
- **gamma**: peso del centroide de los documentos **no relevantes**
  (0,15): resta los terminos de los peores resultados (efecto de
  contraccion), con un peso pequeno para no empujar q' fuera del
  vocabulario.

Sin juicios humanos de relevancia no hay D_r/D_nr reales: se usa
**pseudo-relevance feedback** asumiendo que los `top-k` del ranking base
(k=5) son relevantes y los `bottom-k` (los 5 peores scores) no lo son.
Es el procedimiento estandar de Rocchio cuando no hay gold (Manning et
al. 2008, cap. 9); su riesgo conocido es el *query drift*: si el top-k
esta sesgado hacia un tomo, la expansion refuerza ese sesgo. El
re-rankeo final usa similitud coseno entre q' (normalizado) y los 917
segmentos (ya normalizados l2 por el vectorizador).

El ranking se persiste en `resultados/ranking_rocchio.csv`. La
sensibilidad al tamano del feedback se documenta con la variante k=3
(D_r = top-3, D_nr = bottom-3), persistida en
`resultados/ranking_rocchio_k3.csv` (aporte integrado de Erika Rivera,
Taller_4_IR_CEV_V2): con k menor el centroide depende de los primeros
aciertos del base y el re-rankeo hereda menos su cola.

## 3. Comparacion (punto 4.3)

Sin gold estandar la comparacion es de **acuerdo entre rankings** y
cualitativa: solapamiento top-1 y top-5 entre los tres metodos,
distribucion del top-1 por tomo y ejemplos donde coinciden o difieren,
mas una discusion de las diferencias conceptuales. Se agrega una tabla
top-5 lado a lado (TF-IDF base / Okapi BM25 / Rocchio k=5) para dos
entrevistas de ejemplo (la primera y la ultima con match del ranking
base), que reproduce el formato comparativo del aporte de Erika Rivera
(Taller_4_IR_CEV_V2).
"""


# %%
"""Okapi BM25 manual (paso 1): matriz de conteos, df e IDF con auto-check.

La matriz de conteos se construye con CountVectorizer de sklearn usado
UNICAMENTE como utilidad de tokenizado/conteo (lo permite el enunciado);
la formula BM25 se implementa a mano en `puntuar_bm25` y no se llama a
ninguna funcion de ranking de librerias (rank_bm25, BM25Okapi, etc.).
"""
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

contexto_base = globals().get("contexto_base")
construir_ranking = globals().get("construir_ranking")
enriquecer_ranking = globals().get("enriquecer_ranking")
RUTA_RESULTADOS = globals().get("RUTA_RESULTADOS")
if (
    contexto_base is None
    or construir_ranking is None
    or enriquecer_ranking is None
    or RUTA_RESULTADOS is None
):
    raise RuntimeError(
        "La Sec 03b reutiliza el contexto de la Sec 03 (contexto_base con "
        "textos limpios, matrices y funciones de ranking) y las rutas de la "
        "Sec 00: ejecuta esas secciones antes de esta."
    )

BM25_K1 = 1.5  # saturacion de tf (valor estandar de la literatura)
BM25_B = 0.75  # normalizacion por longitud (valor estandar)
COLUMNAS_RANKING = ["id_doc", "rango", "id_segmento", "score", "tomo"]


def construir_matriz_tf(
    docs_lemas: list[str],
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.85,
) -> tuple[sparse.csr_matrix, CountVectorizer]:
    """Construye la matriz de conteos crudos tf(t,d) de la coleccion.

    Usa CountVectorizer de sklearn solo como utilidad de tokenizado y
    conteo (permitido por el enunciado); el vocabulario replica el del
    TF-IDF base (mismos min_df, max_df y ngram_range) para que BM25 y
    TF-IDF compitan sobre el mismo espacio de terminos.

    Args:
        docs_lemas: documentos limpios (lemas de contenido, 1 por doc).
        ngram_range: rango de n-gramas, (1,2) como el TF-IDF base.
        min_df: frecuencia documental minima para entrar al vocabulario.
        max_df: proporcion documental maxima para entrar al vocabulario.

    Returns:
        Tupla (matriz_tf, vectorizador): matriz CSR (n_docs x n_features)
        de conteos enteros y el CountVectorizer ya ajustado.
    """
    vectorizador = CountVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        lowercase=True,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        dtype=np.int32,
    )
    matriz_tf = vectorizador.fit_transform(docs_lemas)
    return matriz_tf, vectorizador


def idf_bm25(df: np.ndarray, n_docs: int) -> np.ndarray:
    """IDF de Okapi BM25 con el suavizado de Robertson-Sparck Jones.

    Formula: IDF(t) = ln( (N - df(t) + 0,5) / (df(t) + 0,5) + 1 ).

    Args:
        df: frecuencia documental por termino, forma (n_features,).
        n_docs: numero de documentos de la coleccion (N).

    Returns:
        Vector de IDF por termino (float64); un termino presente en todos
        los documentos obtiene el IDF minimo, que queda positivo.
    """
    df = np.asarray(df, dtype=np.float64)
    return np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)


def calcular_longitudes_y_df(
    matriz_tf: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula |d| por documento y df(t) por termino desde la matriz tf.

    IMPORTANTE (unidad de |d|): la longitud |d| es el numero de TERMINOS
    DEL VOCABULARIO del documento, es decir la suma de la fila de la
    matriz tf contando unigramas y bigramas (con repeticion), no el numero
    de lemas de contenido: en esta coleccion avgdl = 1.466,26 terminos del
    vocabulario frente a 1.007,5 lemas de contenido por segmento tras la
    parada del IR (cada bigrama del vocabulario cuenta como un termino
    adicional). Es la longitud que exige la formula de Okapi aplicada a
    ese espacio de terminos, y la misma que "ve" el TF-IDF de la Sec 03.
    df(t) es el numero de documentos con tf(t,d) > 0 (conteo de entradas
    no nulas de la columna).

    Args:
        matriz_tf: matriz CSR (n_docs x n_features) de conteos crudos.

    Returns:
        Tupla (longitudes, df) como arrays float64.
    """
    longitudes = np.asarray(matriz_tf.sum(axis=1)).ravel().astype(np.float64)
    df = np.asarray((matriz_tf > 0).sum(axis=0)).ravel().astype(np.float64)
    return longitudes, df


def puntuar_bm25(
    query_tf,
    matriz_tf: sparse.csr_matrix,
    idf: np.ndarray,
    longitudes: np.ndarray,
    avgdl: float,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> np.ndarray:
    """Puntua BM25 una query (o un lote) contra toda la coleccion.

    Formula implementada (Okapi BM25, Robertson y Walker 1994):
        score(q,d) = sum_t qtf(t) * IDF(t) * tf(t,d) * (k1+1)
                     / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
    donde qtf(t) es la frecuencia del termino en la query. La saturacion
    de tf y la normalizacion por longitud viven en el denominador; el IDF
    premia terminos raros. Es 100 % manual: solo scipy sparse (ninguna
    funcion de ranking de librerias).

    Args:
        query_tf: conteos de la query (CSR 1 x F) o matriz de queries
            (n x F); cualquier formato convertible a CSR.
        matriz_tf: conteos crudos de la coleccion (n_docs x F).
        idf: IDF por termino, forma (F,).
        longitudes: |d| en tokens del vocabulario, forma (n_docs,).
        avgdl: longitud media de la coleccion.
        k1: saturacion de tf (1,5).
        b: normalizacion por longitud (0,75).

    Returns:
        Array (n_docs,) con los scores si la entrada es una query y
        (n, n_docs) si es un lote; los documentos sin ningun termino de
        la query puntuan 0.
    """
    factor = k1 * (
        1.0 - b + b * np.asarray(longitudes, dtype=np.float64) / avgdl
    )
    consultas = sparse.csr_matrix(query_tf)
    coo = matriz_tf.tocoo()
    tf_vals = coo.data.astype(np.float64)
    denominador = tf_vals + factor[coo.row]
    peso_vals = tf_vals * (k1 + 1.0) / denominador * np.asarray(idf)[coo.col]
    pesos = sparse.csr_matrix(
        (peso_vals, (coo.row, coo.col)), shape=matriz_tf.shape
    )
    scores = (consultas @ pesos.T).toarray().astype(np.float64)
    return scores.ravel() if scores.shape[0] == 1 else scores


def verificar_bm25_sintetico() -> bool:
    """Auto-check de la formula BM25 con 3 documentos calculados a mano.

    Documentos sinteticos (N=3, avgdl=3):
        d1 = "gato gato perro"         (tf gato=2, |d1|=3)
        d2 = "gato raton"              (tf gato=1, |d2|=2)
        d3 = "perro raton raton raton" (|d3|=4)
    Query "gato": df(gato)=2, idf(gato)=ln((3-2+0,5)/(2+0,5)+1)=ln(1,6).
    El score esperado de cada documento se calcula con la formula
    completa en la propia celda y se compara contra `puntuar_bm25` con
    tolerancia 1e-12.

    Returns:
        True si la implementacion coincide con el calculo a mano (los
        asserts lanzan AssertionError si no).

    Raises:
        AssertionError: si algun score difiere del calculo manual.
    """
    docs = ["gato gato perro", "gato raton", "perro raton raton raton"]
    vectorizador = CountVectorizer(
        token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 1), min_df=1
    )
    matriz = vectorizador.fit_transform(docs)
    longitudes, df = calcular_longitudes_y_df(matriz)
    idf = idf_bm25(df, n_docs=3)
    scores = puntuar_bm25(
        vectorizador.transform(["gato"]),
        matriz,
        idf,
        longitudes,
        avgdl=float(longitudes.mean()),
    )
    idf_gato = float(np.log((3 - 2 + 0.5) / (2 + 0.5) + 1.0))
    esperado = np.array(
        [
            idf_gato * (2 * (BM25_K1 + 1))
            / (2 + BM25_K1 * (1 - BM25_B + BM25_B * 3 / 3)),
            idf_gato * (1 * (BM25_K1 + 1))
            / (1 + BM25_K1 * (1 - BM25_B + BM25_B * 2 / 3)),
            0.0,
        ]
    )
    assert np.allclose(scores, esperado, atol=1e-12), (scores, esperado)
    print(f"  docs sinteticos: {[len(d.split()) for d in docs]} tokens | "
          f"avgdl = {longitudes.mean():.0f} | idf(gato) = {idf_gato:.6f}")
    print(f"  esperado a mano: {np.round(esperado, 6).tolist()}")
    print(f"  puntuar_bm25   : {np.round(scores, 6).tolist()}")
    return True


print("== Okapi BM25 manual (matriz de conteos) ==")
inicio_preparacion = time.perf_counter()
matriz_tf_libros, vectorizador_tf = construir_matriz_tf(
    contexto_base["libros_limpios"]
)
matriz_tf_queries = vectorizador_tf.transform(contexto_base["entrevistas_limpios"])
longitudes_libros, df_libros = calcular_longitudes_y_df(matriz_tf_libros)
avgdl_libros = float(longitudes_libros.mean())
idf_libros = idf_bm25(df_libros, n_docs=matriz_tf_libros.shape[0])
print(
    f"Coleccion: {matriz_tf_libros.shape[0]} segmentos | "
    f"{matriz_tf_libros.shape[1]:,} terminos | nnz = {matriz_tf_libros.nnz:,}"
)
print(
    f"Longitudes: avgdl = {avgdl_libros:.2f} tokens | min = "
    f"{longitudes_libros.min():.0f} | max = {longitudes_libros.max():.0f} | "
    f"preparacion {time.perf_counter() - inicio_preparacion:.1f} s"
)
print("Auto-check BM25 (calculo a mano, 3 docs):", verificar_bm25_sintetico())


# %%
"""Puntuacion BM25 y ranking top-5 de las 2.486 entrevistas (manual).

La funcion `puntuar_bm25` recorre la formula de Okapi paso a paso sobre
matrices dispersas de scipy: construye la matriz de pesos termino-documento
(idf por termino, saturacion y normalizacion por documento) y multiplica
por el vector de conteos de cada query. `rankear_bm25` aplica el puntaje a
todo el corpus y reutiliza `construir_ranking`/`enriquecer_ranking` de la
Sec 03 (mismo desempate estable y mismos metadatos de tomo/titulo).
"""


def rankear_bm25(
    queries_lemas: list[str],
    docs_lemas: list[str],
    docs_queries: list[dict],
    docs_coleccion: list[dict],
    top_k: int = 5,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> tuple[pd.DataFrame, dict]:
    """Ejecuta BM25 completo (coleccion + queries) y arma el ranking.

    Reutiliza `construir_matriz_tf` (conteos), `calcular_longitudes_y_df`
    (|d| y df), `idf_bm25` y `puntuar_bm25` (formula), y enriquece el
    ranking con tomo y titulo del segmento via la funcion de la Sec 03.

    Args:
        queries_lemas: entrevistas limpias (1 doc por linea).
        docs_lemas: segmentos de libro limpios (1 doc por linea).
        docs_queries: metadatos de las entrevistas (id_doc).
        docs_coleccion: metadatos de los segmentos (id, tomo, titulo).
        top_k: profundidad del ranking (5).
        k1: saturacion de tf.
        b: normalizacion por longitud.

    Returns:
        Tupla (ranking, resumen, matriz_bm25): DataFrame con
        [id_doc, rango, id_segmento, score, tomo, titulo_segmento], dict
        con n_docs, n_terminos, avgdl, segundos de puntuacion y
        distribucion del top-1 por tomo, y la matriz completa de
        puntuaciones (n_queries x n_docs) en float32 (~9 MB) que la
        Sec 03c reutiliza para el heatmap por maximo sin re-puntuar. El
        ranking se construye con los scores float64 originales: la
        conversion a float32 es solo para exponer la matriz.
    """
    inicio = time.perf_counter()
    matriz_tf, vectorizador = construir_matriz_tf(docs_lemas)
    matriz_queries = vectorizador.transform(queries_lemas)
    longitudes, df = calcular_longitudes_y_df(matriz_tf)
    avgdl = float(longitudes.mean())
    idf = idf_bm25(df, n_docs=matriz_tf.shape[0])
    inicio_puntaje = time.perf_counter()
    scores = puntuar_bm25(
        matriz_queries, matriz_tf, idf, longitudes, avgdl, k1, b
    )
    segundos_puntaje = time.perf_counter() - inicio_puntaje
    ids_queries = [registro["id"] for registro in docs_queries]
    ids_coleccion = [registro["id"] for registro in docs_coleccion]
    ranking = construir_ranking(scores, ids_queries, ids_coleccion, top_k=top_k)
    ranking = enriquecer_ranking(ranking, docs_coleccion, ids_coleccion)
    top1 = ranking[ranking["rango"] == 1]
    resumen = {
        "n_docs": matriz_tf.shape[0],
        "n_terminos": matriz_tf.shape[1],
        "avgdl": avgdl,
        "segundos_puntaje": segundos_puntaje,
        "segundos_total": time.perf_counter() - inicio,
        "top1_por_tomo": top1["tomo"].value_counts().to_dict(),
    }
    return ranking, resumen, scores.astype(np.float32)


print("\n== Ranking BM25 top-5 (917 segmentos, 2.486 entrevistas) ==")
inicio_bm25 = time.perf_counter()
ranking_bm25, resumen_bm25, matriz_bm25 = rankear_bm25(
    contexto_base["entrevistas_limpios"],
    contexto_base["libros_limpios"],
    contexto_base["docs_entrevistas"],
    contexto_base["docs_libros"],
)
tiempo_bm25_total = time.perf_counter() - inicio_bm25
contexto_base["matriz_bm25"] = matriz_bm25
print(
    f"Matriz BM25 expuesta para la Sec 03c: {matriz_bm25.shape} | "
    f"{matriz_bm25.nbytes / 1024**2:.1f} MB (float32)"
)

ruta_bm25 = RUTA_RESULTADOS / "ranking_bm25.csv"
try:
    ruta_bm25.parent.mkdir(parents=True, exist_ok=True)
    ranking_bm25[COLUMNAS_RANKING].to_csv(ruta_bm25, index=False)
    print(
        f"[ok] resultados/{ruta_bm25.name}: "
        f"{len(ranking_bm25):,} filas (salida del notebook)"
    )
except OSError as exc:
    print(f"[aviso] no se pudo escribir resultados/{ruta_bm25.name}: {exc}")

print(
    f"Tiempo BM25: puntuacion {resumen_bm25['segundos_puntaje']:.1f} s | "
    f"total {tiempo_bm25_total:.1f} s"
)
print(
    f"Coleccion: {resumen_bm25['n_docs']} segmentos | "
    f"{resumen_bm25['n_terminos']:,} terminos | "
    f"avgdl = {resumen_bm25['avgdl']:.2f} tokens"
)
print(f"Top-1 BM25 por tomo: {resumen_bm25['top1_por_tomo']}")
print("\nHead del ranking BM25 (primeras 5 filas = top-5 de la 1.a query):")
print(ranking_bm25.head(5).to_string(index=False))

top1_base = contexto_base["ranking_top5"]
top1_base = top1_base[top1_base["rango"] == 1].set_index("id_doc")["id_segmento"]
top1_bm25 = ranking_bm25[
    ranking_bm25["rango"] == 1
].set_index("id_doc")["id_segmento"]
comunes_bm25 = top1_base.index.intersection(top1_bm25.index)
solapamiento_bm25 = 100.0 * (top1_base[comunes_bm25] == top1_bm25[comunes_bm25]).mean()
print(
    f"\nAuto-check de correlacion BM25-TFIDF: top-1 identico en "
    f"{solapamiento_bm25:.1f} % de las {len(comunes_bm25)} queries "
    f"comunes (umbral > 40 %): {solapamiento_bm25 > 40.0}"
)


# %% [markdown]
"""## 2. Rocchio y retroalimentacion por pseudo-relevancia

Formula del clasificador de Rocchio (1971):

```text
q' = alpha * q + beta * (1/|D_r|) * sum_{d en D_r} d
                - gamma * (1/|D_nr|) * sum_{d en D_nr} d
```

- **q**: vector de la consulta original (TF-IDF l2 de la entrevista).
- **D_r / D_nr**: documentos relevantes / no relevantes. Sin gold se usan
  los **pseudo-juicios**: top-5 del ranking TF-IDF base como D_r y los 5
  peores (bottom-5 por score) como D_nr.
- **alpha = 1,0**: ancla de la consulta original.
- **beta = 0,75**: expansion hacia los terminos compartidos por D_r.
- **gamma = 0,15**: contraccion de los terminos de D_nr.
- **q'**: vector expandido que se normaliza l2; el re-rankeo usa coseno
  contra los 917 segmentos.

El auto-check sintetico verifica cada parametro con vectores conocidos y
luego se re-rankean las 2.484 queries con match del base (las 2
entrevistas vacias no tienen D_r ni D_nr y se mantienen sin filas).
"""


# %%
"""Rocchio manual: expansion de consultas y auto-check con vectores conocidos."""


def rocchio_expandir(
    q_vec,
    docs_relevantes: sparse.csr_matrix,
    docs_no_relevantes: sparse.csr_matrix,
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15,
) -> sparse.csr_matrix:
    """Expande una query con el clasificador de Rocchio (1971).

    Formula implementada:
        q' = alpha*q + beta*(1/|D_r|)*sum_{d en D_r} d
             - gamma*(1/|D_nr|)*sum_{d en D_nr} d
    donde las sumas son centroides de las matrices de documentos
    relevantes/no relevantes (filas de la matriz TF-IDF). La salida se
    normaliza l2 para que el coseno contra la coleccion (normalizada l2)
    sea el producto punto.

    Args:
        q_vec: vector TF-IDF de la consulta (1 x F, disperso o denso).
        docs_relevantes: matriz (|D_r| x F) con las filas de D_r.
        docs_no_relevantes: matriz (|D_nr| x F) con las filas de D_nr.
        alpha: peso de la consulta original (1,0).
        beta: peso del centroide relevante (0,75).
        gamma: peso del centroide no relevante (0,15).

    Returns:
        Vector expandido q' como CSR (1 x F), con norma l2 unitaria; si
        la expansion queda en cero devuelve el vector cero.
    """
    q = sparse.csr_matrix(q_vec).reshape(1, -1)
    q_nueva = alpha * q
    if docs_relevantes.shape[0] > 0:
        suma_r = sparse.csr_matrix(docs_relevantes.sum(axis=0))
        q_nueva = q_nueva + (beta / docs_relevantes.shape[0]) * suma_r
    if docs_no_relevantes.shape[0] > 0:
        suma_nr = sparse.csr_matrix(docs_no_relevantes.sum(axis=0))
        q_nueva = q_nueva - (gamma / docs_no_relevantes.shape[0]) * suma_nr
    norma = float(np.sqrt(q_nueva.multiply(q_nueva).sum()))
    if norma == 0.0:
        return q_nueva
    return q_nueva / norma


def verificar_rocchio_sintetico() -> bool:
    """Auto-check de Rocchio con vectores conocidos y parametros verificables.

    Espacio de 3 terminos:
        q    = [1, 0, 0]
        D_r  = [[1, 0, 0], [0, 1, 0]]  (2 relevantes)
        D_nr = [[0, 0, 1]]             (1 no relevante)
    Expansion a mano:
        q' = 1*[1,0,0] + 0,75*(1/2)*[1,1,0] - 0,15*(1/1)*[0,0,1]
           = [1,375, 0,375, -0,15]  (luego normalizado l2)
    Los asserts comprueban el vector completo y que beta sube el peso de
    los terminos de D_r mientras gamma baja el de D_nr.

    Returns:
        True si la implementacion coincide (los asserts lanzan si no).

    Raises:
        AssertionError: si el vector expandido difiere del calculo manual.
    """
    q = sparse.csr_matrix(np.array([[1.0, 0.0, 0.0]]))
    docs_r = sparse.csr_matrix(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    docs_nr = sparse.csr_matrix(np.array([[0.0, 0.0, 1.0]]))
    q_exp = rocchio_expandir(q, docs_r, docs_nr, alpha=1.0, beta=0.75, gamma=0.15)
    esperado = np.array([1.0 + 0.75 * 0.5, 0.75 * 0.5, -0.15])
    esperado = esperado / np.linalg.norm(esperado)
    obtenido = q_exp.toarray().ravel()
    assert np.allclose(obtenido, esperado, atol=1e-12), (obtenido, esperado)
    assert obtenido[1] > 0.0, "beta debe atraer los terminos de D_r"
    assert obtenido[2] < 0.0, "gamma debe restar los terminos de D_nr"
    print("  q' sin normalizar a mano: [1.375, 0.375, -0.15]")
    print(f"  rocchio_expandir (norm) : {np.round(obtenido, 6).tolist()}")
    print(f"  efecto beta (termino 2 > 0): {obtenido[1] > 0.0} | "
          f"efecto gamma (termino 3 < 0): {obtenido[2] < 0.0}")
    return True


print("== Rocchio manual (expansion de consultas) ==")
print("Auto-check Rocchio (vectores conocidos):", verificar_rocchio_sintetico())


# %%
"""Ranking Rocchio: re-rankeo con pseudo-relevancia de las 2.484 queries."""


def rankear_rocchio(
    similitudes_base: np.ndarray,
    x_queries: sparse.csr_matrix,
    x_coleccion: sparse.csr_matrix,
    docs_queries: list[dict],
    docs_coleccion: list[dict],
    top_k: int = 5,
    n_relevantes: int = 5,
    n_no_relevantes: int = 5,
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15,
) -> tuple[pd.DataFrame, dict]:
    """Re-rankea las queries del base con expansion de Rocchio.

    Por cada query con match en el ranking TF-IDF base: D_r = top-5 del
    ranking base y D_nr = bottom-5 (peores scores); se expande la query
    con `rocchio_expandir` y se re-rankea por coseno contra la coleccion.
    Las queries sin match en el base (0.2 %: las 2 entrevistas vacias) no
    tienen D_r/D_nr y no generan filas.

    Args:
        similitudes_base: matriz (n_queries x n_docs) del TF-IDF base.
        x_queries: matriz TF-IDF l2 de las queries (n_queries x F).
        x_coleccion: matriz TF-IDF l2 de la coleccion (n_docs x F).
        docs_queries: metadatos de las entrevistas (id_doc).
        docs_coleccion: metadatos de los segmentos (id, tomo, titulo).
        top_k: profundidad del ranking re-rankeado (5).
        n_relevantes: tamanio de D_r (5).
        n_no_relevantes: tamanio de D_nr (5).
        alpha: peso de la query original (1,0).
        beta: peso de D_r (0,75).
        gamma: peso de D_nr (0,15).

    Returns:
        Tupla (ranking, resumen): DataFrame con
        [id_doc, rango, id_segmento, score, tomo, titulo_segmento] y dict
        con n_queries_rerankeadas, n_cambios_top1, n_docs, n_filas.
    """
    inicio = time.perf_counter()
    n_queries, n_docs = similitudes_base.shape
    scores_expandidos = np.zeros((n_queries, n_docs), dtype=np.float64)
    n_rerankeadas = 0
    n_cambios_top1 = 0
    for i in range(n_queries):
        orden_base = np.argsort(-similitudes_base[i], kind="stable")
        if similitudes_base[i, orden_base[0]] <= 0.0:
            continue
        q_expandida = rocchio_expandir(
            x_queries[i],
            x_coleccion[orden_base[:n_relevantes]],
            x_coleccion[orden_base[-n_no_relevantes:]],
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        scores_expandidos[i] = np.asarray(
            (q_expandida @ x_coleccion.T).todense()
        ).ravel()
        n_rerankeadas += 1
        orden_expandido = np.argsort(-scores_expandidos[i], kind="stable")
        if int(orden_expandido[0]) != int(orden_base[0]):
            n_cambios_top1 += 1
    ids_queries = [registro["id"] for registro in docs_queries]
    ids_coleccion = [registro["id"] for registro in docs_coleccion]
    ranking = construir_ranking(scores_expandidos, ids_queries, ids_coleccion, top_k=top_k)
    ranking = enriquecer_ranking(ranking, docs_coleccion, ids_coleccion)
    resumen = {
        "n_queries_rerankeadas": n_rerankeadas,
        "n_cambios_top1": n_cambios_top1,
        "n_cambios_top1_pct": 100.0 * n_cambios_top1 / max(n_rerankeadas, 1),
        "n_docs": n_docs,
        "n_filas": len(ranking),
        "segundos": time.perf_counter() - inicio,
    }
    return ranking, resumen


print("\n== Ranking Rocchio top-5 (re-rankeo con pseudo-relevancia) ==")
inicio_rocchio = time.perf_counter()
ranking_rocchio, resumen_rocchio = rankear_rocchio(
    contexto_base["similitudes"],
    contexto_base["x_queries"],
    contexto_base["x_coleccion"],
    contexto_base["docs_entrevistas"],
    contexto_base["docs_libros"],
)
tiempo_rocchio_total = time.perf_counter() - inicio_rocchio

ruta_rocchio = RUTA_RESULTADOS / "ranking_rocchio.csv"
try:
    ruta_rocchio.parent.mkdir(parents=True, exist_ok=True)
    ranking_rocchio[COLUMNAS_RANKING].to_csv(ruta_rocchio, index=False)
    print(
        f"[ok] resultados/{ruta_rocchio.name}: {len(ranking_rocchio):,} filas "
        "(salida del notebook)"
    )
except OSError as exc:
    print(f"[aviso] no se pudo escribir resultados/{ruta_rocchio.name}: {exc}")

print(
    f"Queries re-rankeadas: {resumen_rocchio['n_queries_rerankeadas']} | "
    f"top-1 cambia respecto al base en "
    f"{resumen_rocchio['n_cambios_top1']} "
    f"({resumen_rocchio['n_cambios_top1_pct']:.1f} %)"
)
print(
    f"Tiempo Rocchio: {tiempo_rocchio_total:.1f} s | filas: "
    f"{resumen_rocchio['n_filas']:,}"
)
top1_rocchio = ranking_rocchio[ranking_rocchio["rango"] == 1]
print(
    "Top-1 Rocchio por tomo: "
    f"{top1_rocchio['tomo'].value_counts().to_dict()}"
)

# Verificacion OPCIONAL de salida: si el CSV se escribio, re-leerlo con
# pandas debe devolver el mismo numero de filas (guard .exists()).
if ruta_rocchio.exists():
    filas_en_disco = len(pd.read_csv(ruta_rocchio))
    print(
        f"Verificacion opcional de salida: {filas_en_disco:,} filas en disco "
        f"== {len(ranking_rocchio):,} en memoria: "
        f"{filas_en_disco == len(ranking_rocchio)}"
    )


# %%
"""Variante Rocchio k=3: sensibilidad al tamano del feedback de pseudo-relevancia.

Aporte integrado del Taller_4_IR_CEV_V2 (Erika Rivera): re-rankea con
D_r = top-3 y D_nr = bottom-3 en lugar de la configuracion 5/5 de la celda
anterior. Se reutiliza `rankear_rocchio` (misma formula y mismo re-rankeo,
solo cambia el parametro); la salida nueva
`resultados/ranking_rocchio_k3.csv` se escribe con guard (salida del
notebook) y no sustituye a `ranking_rocchio.csv` (k=5).
"""

print("\n== Sensibilidad de Rocchio al parametro k del feedback (k=3 vs k=5) ==")
inicio_rocchio_k3 = time.perf_counter()
ranking_rocchio_k3, resumen_rocchio_k3 = rankear_rocchio(
    contexto_base["similitudes"],
    contexto_base["x_queries"],
    contexto_base["x_coleccion"],
    contexto_base["docs_entrevistas"],
    contexto_base["docs_libros"],
    n_relevantes=3,
    n_no_relevantes=3,
)
tiempo_rocchio_k3 = time.perf_counter() - inicio_rocchio_k3

ruta_rocchio_k3 = RUTA_RESULTADOS / "ranking_rocchio_k3.csv"
try:
    ruta_rocchio_k3.parent.mkdir(parents=True, exist_ok=True)
    ranking_rocchio_k3[COLUMNAS_RANKING].to_csv(ruta_rocchio_k3, index=False)
    print(
        f"[ok] resultados/{ruta_rocchio_k3.name}: "
        f"{len(ranking_rocchio_k3):,} filas (salida del notebook)"
    )
except OSError as exc:
    print(f"[aviso] no se pudo escribir resultados/{ruta_rocchio_k3.name}: {exc}")


def top1_por_query(ranking: pd.DataFrame) -> pd.Series:
    """Top-1 (id_segmento) por id_doc de un ranking ya enriquecido.

    Args:
        ranking: ranking con columnas id_doc, rango e id_segmento.

    Returns:
        Serie indexada por id_doc con el id_segmento del rango 1.
    """
    top1 = ranking[ranking["rango"] == 1]
    return top1.set_index("id_doc")["id_segmento"]


top1_k3 = top1_por_query(ranking_rocchio_k3)
top1_k5 = top1_por_query(ranking_rocchio)
top1_base_series = top1_por_query(contexto_base["ranking_top5"])
comunes_k3_base = top1_k3.index.intersection(top1_base_series.index)
cambios_k3_base = int(
    (top1_k3[comunes_k3_base] != top1_base_series[comunes_k3_base]).sum()
)
comunes_k3_k5 = top1_k3.index.intersection(top1_k5.index)
cambios_k3_k5 = int(
    (top1_k3[comunes_k3_k5] != top1_k5[comunes_k3_k5]).sum()
)
print(
    f"Top-1 k=3 vs TF-IDF base: cambia en {cambios_k3_base} de "
    f"{len(comunes_k3_base)} queries "
    f"({100.0 * cambios_k3_base / max(len(comunes_k3_base), 1):.1f} %); "
    f"resumen de rankear_rocchio: {resumen_rocchio_k3['n_cambios_top1']}"
)
print(
    f"Top-1 k=3 vs Rocchio k=5: cambia en {cambios_k3_k5} de "
    f"{len(comunes_k3_k5)} queries "
    f"({100.0 * cambios_k3_k5 / max(len(comunes_k3_k5), 1):.1f} %)"
)
print(f"Tiempo Rocchio k=3: {tiempo_rocchio_k3:.1f} s")


# %% [markdown]
"""## Sensibilidad al parametro k del feedback

La variante k=3 (D_r = top-3, D_nr = bottom-3; aporte de Erika Rivera) usa
un centroide mas pequeno y sensible a los primeros aciertos del ranking
base. La comparacion de top-1 contra TF-IDF y contra Rocchio k=5 (celda
anterior) mide cuanto del reordenamiento depende del tamano del feedback:
k=3 conserva mas la cabeza del ranking base, mientras que k=5 incorpora mas
vocabulario del tomo dominante. La salida
`resultados/ranking_rocchio_k3.csv` es nueva y no sustituye a
`ranking_rocchio.csv` (k=5).
"""


# %%
"""Comparacion P4.3: solapamiento de rankings, top-1 por tomo y ejemplos.

Sin gold estandar la comparacion es de acuerdo entre metodos. Se reportan
DOS lecturas distintas del acuerdo: el solapamiento de CONJUNTO del top-k
(|A_k ∩ B_k| / k medio por query, insensible al orden interno) y el
porcentaje de POSICION k (identidad exacta en la posicion k, sensible al
orden). Para k=1 ambas coinciden; para k=5 el conjunto mide cuanto
inventario comparten y la posicion mide si el orden final coincide.
Ademas se reparte el top-1 por tomo y se muestran ejemplos concretos que
coinciden o divergen (con tomo y titulo del segmento recuperado).
"""

P4_METODOS = {
    "TF-IDF base": contexto_base["ranking_top5"],
    "BM25": ranking_bm25,
    "Rocchio": ranking_rocchio,
}


def _conjuntos_top_k(ranking: pd.DataFrame, k: int) -> pd.Series:
    """Conjunto de id_segmento del top-k por query (indice = id_doc).

    Args:
        ranking: ranking con columnas id_doc/rango/id_segmento.
        k: profundidad (rangos 1..k).

    Returns:
        Serie indexada por id_doc cuyos valores son sets de id_segmento.
    """
    return (
        ranking[ranking["rango"] <= k]
        .groupby("id_doc")["id_segmento"]
        .apply(set)
    )


def solapamiento_conjunto_top_k(
    ranking_a: pd.DataFrame, ranking_b: pd.DataFrame, k: int
) -> float:
    """Solapamiento medio de CONJUNTO del top-k entre dos rankings.

    Por cada query comun se calcula |A_k ∩ B_k| / k, donde A_k y B_k son
    los conjuntos de id_segmento en los rangos 1..k de cada ranking, y se
    promedia sobre las queries. No depende del orden interno del top-k:
    dos rankings con los mismos 5 segmentos en distinto orden puntuan
    100 %. Es la metrica correcta para "comparten inventario".

    Args:
        ranking_a: ranking con columnas id_doc/rango/id_segmento.
        ranking_b: ranking con el mismo formato.
        k: profundidad del top-k (1 o 5).

    Returns:
        Porcentaje (0-100) del solapamiento medio de conjunto; NaN si no
        hay queries comunes.
    """
    izquierda = _conjuntos_top_k(ranking_a, k)
    derecha = _conjuntos_top_k(ranking_b, k)
    comunes = izquierda.index.intersection(derecha.index)
    if len(comunes) == 0:
        return float("nan")
    valores = [
        len(izquierda[id_doc] & derecha[id_doc]) / k for id_doc in comunes
    ]
    return 100.0 * float(np.mean(valores))


def pos_k_pct(ranking_a: pd.DataFrame, ranking_b: pd.DataFrame, k: int) -> float:
    """Porcentaje de queries con el MISMO segmento en la POSICION k.

    Compara la posicion exacta k de ambos rankings (identidad del
    id_segmento), por lo que es sensible al orden interno: si los dos
    rankings comparten los 5 segmentos pero en orden distinto, da 0 % en
    k=5 a pesar del conjunto identico. Para k=1 coincide con
    `solapamiento_conjunto_top_k(..., 1)`.

    Args:
        ranking_a: ranking con columnas id_doc/rango/id_segmento.
        ranking_b: ranking con el mismo formato.
        k: rango a comparar (1 o 5).

    Returns:
        Porcentaje (0-100) de queries comunes con id_segmento identico en
        la posicion k; NaN si no hay queries comunes.
    """
    izquierda = ranking_a[ranking_a["rango"] == k].set_index("id_doc")["id_segmento"]
    derecha = ranking_b[ranking_b["rango"] == k].set_index("id_doc")["id_segmento"]
    comunes = izquierda.index.intersection(derecha.index)
    if len(comunes) == 0:
        return float("nan")
    return 100.0 * (izquierda[comunes] == derecha[comunes]).mean()


print("== Comparacion P4.3: acuerdo entre metodos ==")
filas_solapamiento = []
for nombre_a, nombre_b in (
    ("TF-IDF base", "BM25"),
    ("TF-IDF base", "Rocchio"),
    ("BM25", "Rocchio"),
):
    filas_solapamiento.append(
        {
            "par": f"{nombre_a} vs {nombre_b}",
            "top1_conjunto_pct": round(
                solapamiento_conjunto_top_k(
                    P4_METODOS[nombre_a], P4_METODOS[nombre_b], 1
                ),
                1,
            ),
            "pos5_pct": round(
                pos_k_pct(P4_METODOS[nombre_a], P4_METODOS[nombre_b], 5), 1
            ),
            "top5_conjunto_pct": round(
                solapamiento_conjunto_top_k(
                    P4_METODOS[nombre_a], P4_METODOS[nombre_b], 5
                ),
                1,
            ),
            "queries_comparables": len(
                P4_METODOS[nombre_a][P4_METODOS[nombre_a]["rango"] == 1]
            ),
        }
    )
tabla_solapamiento = pd.DataFrame(filas_solapamiento)
print(
    "Acuerdo por par de metodos (top1_conjunto = |A1∩B1|/1; pos5 = "
    "identidad en la posicion 5; top5_conjunto = |A5∩B5|/5):"
)
print(tabla_solapamiento.to_string(index=False))

N_QUERIES_BASE = int((contexto_base["ranking_top5"]["rango"] == 1).sum())
print(
    f"\nTop-1 por tomo (2.486 queries; cada columna suma "
    f"{N_QUERIES_BASE}):"
)
tabla_top1 = pd.DataFrame(
    {
        nombre: ranking[ranking["rango"] == 1]["tomo"].value_counts()
        for nombre, ranking in P4_METODOS.items()
    }
).fillna(0).astype(int)
print(tabla_top1.to_string())

# ---- Ejemplos cualitativos: 3 queries donde coinciden y 3 donde difieren.
def _top1_con_titulo(ranking: pd.DataFrame, sufijo: str) -> pd.DataFrame:
    """Top-1 de un ranking con columnas renombradas para el merge.

    Args:
        ranking: ranking enriquecido (id_doc, rango, id_segmento, tomo...).
        sufijo: sufijo de las columnas de este metodo.

    Returns:
        DataFrame id_doc + id_segmento/tomo/titulo con sufijo.
    """
    top1 = ranking[ranking["rango"] == 1].copy()
    return top1.rename(
        columns={
            "id_segmento": f"id_segmento_{sufijo}",
            "tomo": f"tomo_{sufijo}",
            "titulo_segmento": f"titulo_{sufijo}",
        }
    )[
        [
            "id_doc",
            f"id_segmento_{sufijo}",
            f"tomo_{sufijo}",
            f"titulo_{sufijo}",
        ]
    ]


comparacion_top1 = (
    _top1_con_titulo(contexto_base["ranking_top5"], "base")
    .merge(_top1_con_titulo(ranking_bm25, "bm25"), on="id_doc")
    .merge(_top1_con_titulo(ranking_rocchio, "rocchio"), on="id_doc")
)
coinciden = comparacion_top1[
    (comparacion_top1["id_segmento_base"] == comparacion_top1["id_segmento_bm25"])
    & (comparacion_top1["id_segmento_bm25"] == comparacion_top1["id_segmento_rocchio"])
]
difieren = comparacion_top1[
    (comparacion_top1["id_segmento_base"] != comparacion_top1["id_segmento_bm25"])
    & (comparacion_top1["id_segmento_bm25"] != comparacion_top1["id_segmento_rocchio"])
    & (comparacion_top1["id_segmento_base"] != comparacion_top1["id_segmento_rocchio"])
]
print(
    f"\nEjemplos: {len(coinciden)} queries con top-1 identico en los 3 "
    f"metodos | {len(difieren)} con top-1 distinto en los 3"
)


def _tabla_ejemplos(
    ejemplos: pd.DataFrame, etiqueta: str, n: int = 3
) -> pd.DataFrame:
    """Tabla legible (id corto y tomo/titulo por metodo) de n ejemplos.

    Args:
        ejemplos: subconjunto de comparacion_top1.
        etiqueta: "coinciden" o "difieren" (solo para el print).
        n: numero de filas a mostrar.

    Returns:
        DataFrame con id_doc truncado y el top-1 de cada metodo.
    """
    filas = []
    for _, fila in ejemplos.head(n).iterrows():
        registro = {"id_doc": fila["id_doc"][:16] + ".."}
        for sufijo, nombre in (
            ("base", "base"),
            ("bm25", "bm25"),
            ("rocchio", "rocchio"),
        ):
            titulo = str(fila[f"titulo_{sufijo}"])[:38]
            registro[nombre] = f"{fila[f'tomo_{sufijo}']}: {titulo}"
        filas.append(registro)
    tabla = pd.DataFrame(filas)
    print(f"\n{etiqueta} (top-1 de cada metodo):")
    print(tabla.to_string(index=False))
    return tabla


tabla_coinciden = _tabla_ejemplos(coinciden, "3 queries donde los 3 metodos coinciden")
tabla_difieren = _tabla_ejemplos(difieren, "3 queries donde los 3 metodos difieren")


# %%
"""Tabla comparativa top-5 de las 3 metricas para entrevistas de ejemplo.

Adaptado del aporte de Erika Rivera (Taller_4_IR_CEV_V2): para cada query
de ejemplo se muestran lado a lado los top-5 de TF-IDF base, Okapi BM25 y
Rocchio k=5 usando los rankings YA calculados en esta seccion (no se
re-puntua nada). Las queries de ejemplo son la primera y la ultima
entrevista con match del ranking base.
"""


def top5_metodo(ranking: pd.DataFrame, id_doc: str, n: int = 5) -> list[str]:
    """Top-n de un ranking para una query como 'id_segmento (score)'.

    Args:
        ranking: ranking enriquecido (id_doc, rango, id_segmento, score).
        id_doc: entrevista (query) a extraer.
        n: profundidad del top (5).

    Returns:
        Lista de n cadenas; se rellena con "" si la query tiene menos
        filas que n (queries con pocos matches).
    """
    top = ranking[ranking["id_doc"] == id_doc].nsmallest(n, "rango")
    valores = [
        f"{fila['id_segmento']} ({fila['score']:.3f})"
        for _, fila in top.iterrows()
    ]
    return valores + [""] * (n - len(valores))


def tabla_top5_entrevista(id_doc: str) -> pd.DataFrame:
    """Tabla Top 1..Top 5 x [TF-IDF base, Okapi BM25, Rocchio k=5].

    Args:
        id_doc: entrevista de ejemplo (id presente en el ranking base).

    Returns:
        DataFrame con la columna Ranking y una columna por metodo con el
        id_segmento y el score de cada posicion.
    """
    metodos = (
        ("TF-IDF (base)", contexto_base["ranking_top5"]),
        ("Okapi BM25", ranking_bm25),
        ("Rocchio (PRF)", ranking_rocchio),
    )
    datos = {
        nombre: top5_metodo(ranking, id_doc) for nombre, ranking in metodos
    }
    datos["Ranking"] = [f"Top {i}" for i in range(1, 6)]
    return pd.DataFrame(datos)[["Ranking", *[n for n, _ in metodos]]]


print("== Tabla comparativa top-5 (3 metricas) para entrevistas de ejemplo ==")
for id_doc in (
    contexto_base["ranking_top5"]["id_doc"].iloc[0],
    contexto_base["ranking_top5"]["id_doc"].iloc[-1],
):
    print(f"\n=== Comparacion de ranking para la entrevista: {id_doc} ===")
    tabla_metodos = tabla_top5_entrevista(id_doc)
    try:
        from IPython.display import display

        display(tabla_metodos)
    except ImportError:
        print(tabla_metodos.to_string(index=False))


# %% [markdown]
"""## Discusion P4: que cambia entre TF-IDF, BM25 y Rocchio

**Cifras de la corrida (celdas de arriba).** BM25 y TF-IDF comparten el
57,0 % del top-1 (solapamiento de conjunto; para k=1 conjunto y posicion
coinciden): la senal lexica comun del conflicto ordena los mismos
segmentos, pero BM25 reordena con dos mecanismos que el coseno TF-IDF no
tiene (saturacion de tf y normalizacion por longitud con b=0,75), y con
su IDF de Robertson-Sparck Jones. El top-5 hay que leerlo en dos
niveles: el **conjunto** de los 5 segmentos coincide en el 65,7 % de los
casos (base-BM25) y solo en el 11,8 % coincide ademas la **posicion 5**
exacta; es decir, comparten la mayoria del inventario recuperado y
difieren sobre todo en el orden. Rocchio, al re-ranquear con
pseudo-relevancia, conserva el 70,1 % de los top-1 del base (es una
expansion del ranking base, no un modelo independiente) y mueve el
29,9 % restante (743 de las 2.484 queries): el centroide de los top-5
refuerza el tema dominante de cada entrevista y el bottom-5 resta el
vocabulario ajeno. Su conjunto de top-5 es identico al del base en el
100,0 % de las queries (misma posicion 5 solo en el 48,8 %): Rocchio
reordena, no sustituye el inventario.

**Interpretacion de las diferencias.**

1. **BM25 normaliza por longitud y satura tf de forma distinta.** TF-IDF
   usa tf sublineal y norma l2 del vector completo: un segmento largo y
   uno corto con la misma proporcion de terminos empatan. BM25 divide
   cada termino por tf + k1*(1-b+b*|d|/avgdl): los segmentos de
   VIOLACIONES/ETNICO (largos) y CONVOCATORIA (corto) se comparan por
   densidad de evidencia, no por volumen.
2. **El IDF de BM25 es mas agresivo** con los terminos raros
   (ln((N-df+0,5)/(df+0,5)+1) contra ln((N+1)/(df+1))+1 de sklearn): en
   un vocabulario donde el 43 % de los tipos son hapax, esto reordena
   mucho la cola.
3. **Rocchio es un re-rankeo, no un ranking desde cero.** La expansion
   asume que los top-5 del base son relevantes: sin gold, es
   pseudo-relevance feedback y hereda el sesgo del base (*query drift*).
   Con beta=0,75 y gamma=0,15 el efecto es suave, pero la importancia
   de gamma es limitada en la practica: los 5 peores scores del base
   son casi siempre 0 (ningun termino en comun), asi que su centroide
   aporta poco y el efecto dominante es la expansion por beta.
4. **Sin gold estandar no hay P@k real.** El acuerdo entre rankings mide
   consistencia, no correccion; los ejemplos cualitativos (celda de
   arriba) y la evaluacion etiquetada de 30 pares de la Sec 03 son la
   unica verificacion humana disponible. Los tres metodos comparten el
   **sesgo TESTIMONIAL** del dominio (top-1 de TESTIMONIAL: 76,8 % base,
   74,9 % BM25 sobre las 2.486 entrevistas -- 75,0 % sobre las 2.484 con
   match -- y 78,4 % Rocchio): es un efecto de registro oral del corpus,
   no de la funcion de puntuacion.
5. **Unidad de |d|/avgdl (nota metodologica).** La longitud de documento
   de BM25 se mide en terminos del vocabulario (unigramas + bigramas de
   la matriz tf; cada bigrama cuenta como un termino): avgdl = 1.466,26.
   No es el numero de lemas de contenido del segmento (1.007,5 de media
   tras la parada en consumo del IR, 923.912 lemas / 917 segmentos), que
   es la unidad del corpus preprocesado. La formula Okapi es la estandar
   aplicada a ese espacio de terminos, el mismo que usa el TF-IDF de la
   Sec 03, de modo que la comparacion entre ambos metodos es justa (si se
   usaran lemas sueltos sin bigramas, |d| y avgdl serian ~1.007,5).

**Lectura de los ejemplos.** De las 2.484 queries, 1.112 tienen el mismo
top-1 en los tres metodos y 296 lo tienen distinto en los tres. Donde los
tres coinciden suele haber un solape lexico fuerte y especifico (masacre,
secuestro, etnico: el segmento comparte el hecho y la poblacion de la
entrevista). Donde difieren, el top-1 base suele ser un segmento
testimonial generico, BM25 premia terminos raros que cambian el tomo
(NARRATIVA/HALLAZGOS) y Rocchio reordena hacia el centroide tematico de
los 5 primeros. Ninguno de los tres puede resolver la especificidad fina
sin un gold o un modelo semantico (ver Sec 04).

**Artefactos a documentar en el informe.** BM25 y Rocchio escriben
`resultados/ranking_bm25.csv` y `resultados/ranking_rocchio.csv`; ambos
son salidas del notebook (no insumos). Las casillas con menos de 5 filas
(2 entrevistas vacias en el base) no participan en el re-rankeo de
Rocchio.
"""
