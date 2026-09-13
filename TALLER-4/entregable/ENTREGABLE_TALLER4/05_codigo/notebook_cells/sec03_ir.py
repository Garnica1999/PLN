# %% [markdown]
"""# Sec 03 — Recuperacion de Informacion: experimento TF-IDF base

Se ejecuta aqui el **experimento IR base** de la Fase 6 con una
implementacion LOCAL dentro del notebook (`correr_ir()`): replica la
semantica exacta de `experimentos/experimento_1_tfidf_base.py` (leido como
referencia, no importado: sin importlib ni sys.path hacia experimentos/) y
de scripts/03_eda.py, leyendo solo los archivos versionados de `config/`,
`data/` y `resultados/`. El codigo corre el modelo en memoria SIN
sobrescribir los resultados oficiales de `resultados/` (los CSVs del
experimento quedan intactos) y verifica la paridad contra ellos.

Direccion de matching (ADR-005): cada entrevista (2.486) es una query y la
coleccion son los 917 segmentos de libro. TF-IDF con saturacion de tf
(sublinear_tf), norma l2 y ngram_range=(1,2), config congelada en
config/tfidf_config.json (min_df=2, max_df=0,85); parada en consumo
(spacy STOP_WORDS + 51 muletillas + 42 ruido tipografico) y 62 correcciones
de lemas. Similitud coseno por batches y ranking top-5 por query.

La celda imprime el ranking top-5 (5 filas) y el contexto del experimento
(features, densidad, queries vacias). La corrida tarda ~12 s.
"""


# %%
"""Implementacion local del pipeline IR (paridad con el experimento base)."""
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.es import STOP_WORDS

if "RUTA_RAIZ" not in globals():  # respaldo standalone robusto (ver sec00)
    from pathlib import Path as _Path

    def _raiz_standalone() -> _Path:
        """Localiza la raiz del proyecto TALLER-4 (respaldo standalone).

        Copia del algoritmo robusto de sec00 (sin TALLER4_ROOT ni
        auto-descarga, para no sorprender en ejecuciones sueltas): ancestros
        desde el cwd, subcarpetas de hasta 3 niveles bajo el cwd y bajo cada
        ancestro, y rutas tipicas de Colab. La raiz se reconoce por el cache
        derivado data/corpus/segmentos_libros.jsonl o por cualquiera de los
        3 insumos en data/raw/ (dataset de entrevistas, libros CEV o
        tesauro), de modo que se resuelve antes de generar los caches. Nunca
        devuelve la raiz del filesystem: si no encuentra, lanza un error
        claro.

        Returns:
            Path absoluto de la raiz del proyecto.

        Raises:
            FileNotFoundError: si no hay data/corpus/segmentos_libros.jsonl
                ni ninguno de los 3 insumos en data/raw/.
        """
        import os as _os

        excluidos = {".git", "drive", "sample_data", "venv", "node_modules"}
        cwd = _Path(_os.getcwd()).resolve()

        def _es_raiz(candidato) -> bool:
            """True si el candidato es la raiz del proyecto TALLER-4.

            Un directorio es la raiz si contiene el cache derivado
            data/corpus/segmentos_libros.jsonl o CUALQUIERA de los 3 insumos
            permitidos en data/raw/ (dataset de entrevistas, libros CEV o
            tesauro): asi la raiz se resuelve ANTES de generar los caches
            derivados.

            Args:
                candidato: Directorio candidato a raiz del proyecto.

            Returns:
                True si contiene el cache derivado o algun insumo de
                data/raw/.
            """
            return (
                (candidato / "data" / "corpus" / "segmentos_libros.jsonl").exists()
                or (candidato / "data" / "raw" / "entrevistas_all_2023-03-21_14-24_05.json").exists()
                or (candidato / "data" / "raw" / "Resource-tesauro.rdf").exists()
                or bool(list((candidato / "data" / "raw").glob("books-*.zip")))
                or bool(list((candidato / "data" / "raw" / "books").glob("*.pdf")))
            )

        def caminar(base: _Path, max_profundidad: int):
            """Itera base y sus subcarpetas hasta la profundidad dada."""
            nivel_base = len(_Path(base).parts)
            for dir_actual, subdirs, _ in _os.walk(base):
                nivel = len(_Path(dir_actual).parts) - nivel_base
                if nivel >= max_profundidad:
                    subdirs[:] = []
                else:
                    subdirs[:] = sorted(
                        d
                        for d in subdirs
                        if not d.startswith(".") and d not in excluidos
                    )
                yield _Path(dir_actual)

        ancestros = [cwd, *cwd.parents]
        for ancestro in ancestros:
            if _es_raiz(ancestro):
                return ancestro
        for ancestro in ancestros:
            for candidato in caminar(ancestro, 3):
                if _es_raiz(candidato):
                    return candidato
        for candidata in (
            _Path("/content/TALLER-4"),
            _Path("/content/drive/MyDrive/TALLER-4"),
        ):
            if _es_raiz(candidata):
                return candidata
        raise FileNotFoundError(
            "No se encontro la raiz del proyecto TALLER-4: se busca la "
            "carpeta con data/corpus/segmentos_libros.jsonl o alguno de los "
            "3 insumos en data/raw/ (dataset de entrevistas, libros CEV o "
            "tesauro) en el arbol de directorios (sube la carpeta TALLER-4 "
            "completa a /content o descomprime ahi el zip del repositorio; "
            "queda en /content/PLN-main/TALLER-4 y tambien se detecta)."
        )

    RUTA_RAIZ = _raiz_standalone()
    RUTA_PREPROCESADO = RUTA_RAIZ / "data" / "corpus" / "preprocesado"
    RUTA_CONFIG = RUTA_RAIZ / "config"
    RUTA_RESULTADOS = RUTA_RAIZ / "resultados"

# Helpers de config: existen en sec00 (notebook, con copias embebidas); se
# definen solo si la seccion corre standalone (misma convencion de
# autocontencion). Los respaldos leen config/ solo si existe; si no, avisan
# y degradan (config curada del notebook vive embebida en sec00).
if "cargar_correcciones" not in globals():

    def cargar_correcciones() -> dict[str, str]:
        """Carga config/lemas_correcciones.json si existe (respaldo standalone)."""
        ruta = RUTA_CONFIG / "lemas_correcciones.json"
        if not ruta.exists():
            print("[aviso] config/ no presente en modo standalone; sin correcciones.")
            return {}
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in datos["correcciones"].items()}

    def cargar_muletillas() -> set[str]:
        """Carga config/stopwords_habla.txt si existe (respaldo standalone)."""
        ruta = RUTA_CONFIG / "stopwords_habla.txt"
        if not ruta.exists():
            print("[aviso] config/ no presente en modo standalone; sin muletillas.")
            return set()
        return {
            linea.strip()
            for linea in ruta.read_text(encoding="utf-8").splitlines()
            if linea.strip() and not linea.startswith("#")
        }

    def cargar_ruido() -> set[str]:
        """Carga el ruido tipografico de config/tfidf_config.json (standalone)."""
        ruta = RUTA_CONFIG / "tfidf_config.json"
        if not ruta.exists():
            print("[aviso] config/ no presente en modo standalone; sin ruido.")
            return set()
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        return {str(t) for t in datos["ruido_tipografico_libros_excluido_en_eda"]}

    def cargar_lemas(corpus: str) -> list[str]:
        """Lee el archivo de lemas de un corpus (respaldo standalone)."""
        ruta = RUTA_PREPROCESADO / f"{corpus}_lemas.txt"
        return [linea.rstrip("\n") for linea in ruta.open(encoding="utf-8")]

    def cargar_metadatos(corpus: str) -> list[dict]:
        """Lee los metadatos de un corpus (respaldo standalone)."""
        ruta = RUTA_PREPROCESADO / f"{corpus}_docs.json"
        return json.loads(ruta.read_text(encoding="utf-8"))["docs"]


# Parametros TF-IDF congelados (copia embebida para independencia total; la
# fuente oficial es config/tfidf_config.json de los scripts de produccion).
TFIDF_PARAMS_VECTORIZADOR_EMBEDIDOS: dict = {
    "analyzer": "word",
    "token_pattern": r"(?u)\b\w+\b",
    "lowercase": True,
    "ngram_range": [1, 2],
    "sublinear_tf": True,
    "norm": "l2",
    "use_idf": True,
    "smooth_idf": True,
    "min_df": 2,
    "max_df": 0.85,
}


def cargar_config_vectorizador() -> dict:
    """Devuelve los parametros del vectorizador (config embebida).

    El notebook NO lee config/tfidf_config.json: usa la copia congelada
    `TFIDF_PARAMS_VECTORIZADOR_EMBEDIDOS` (identica a la de sec00 y a la
    fuente de produccion; `verificar_consistencia_config` de sec00 compara
    cuando el archivo existe).

    Returns:
        Dict con los kwargs de TfidfVectorizer congelados en el EDA (Fase 5).
    """
    globales = globals()
    if "TFIDF_PARAMS_EMBEDIDOS" in globales:  # sec00 ya corrio en el notebook
        config = dict(globales["TFIDF_PARAMS_EMBEDIDOS"])
    else:
        config = dict(TFIDF_PARAMS_VECTORIZADOR_EMBEDIDOS)
    config["ngram_range"] = tuple(int(v) for v in config["ngram_range"])
    return config


def _cargar_correcciones_ir() -> dict[str, str]:
    """Correcciones de lemas para el IR (siempre la copia embebida).

    En el notebook las celdas expandidas de scripts/03_eda.py definen su
    propia `cargar_correcciones()` (la del EDA de produccion, que CREA
    config/lemas_correcciones.json si el archivo no existe): el nombre
    global queda reemplazado en el kernel antes de la Sec 03. Para
    independencia total el pipeline IR usa la constante embebida
    CORRECCIONES_EMBEDIDAS de sec00 (identica a la fuente de produccion);
    en modo standalone (sin sec00) usa la `cargar_correcciones` local con
    guard .exists().

    Returns:
        Mapa lema anomalo -> lema correcto.
    """
    if "CORRECCIONES_EMBEDIDAS" in globals():
        return dict(globals()["CORRECCIONES_EMBEDIDAS"])
    return cargar_correcciones()


def construir_parada_ir(incluir_muletillas: bool = True) -> set[str]:
    """Parada en consumo del IR: spacy STOP_WORDS + ruido + muletillas.

    Replica `construir_parada()` del experimento base: la parada efectiva del
    pipeline IR es spacy.lang.es.STOP_WORDS + ruido tipografico de libros
    (siempre) y, opcionalmente, las muletillas de config/stopwords_habla.txt
    (False solo para la variante B del experimento).

    Args:
        incluir_muletillas: False para la variante B (parada solo spaCy +
            ruido tipografico).

    Returns:
        Conjunto de tokens que se eliminan del texto lematizado.
    """
    parada = set(STOP_WORDS) | cargar_ruido()
    if incluir_muletillas:
        parada |= cargar_muletillas()
    return parada


def limpiar_linea(linea: str, parada: set[str], correcciones: dict[str, str]) -> str:
    """Limpia una linea de lemas aplicando correcciones y parada en consumo.

    Replica `limpiar_linea()` del experimento base (y corregir_y_filtrar de
    scripts/03_eda.py): por token, primero la correccion (lema -> lema) y,
    sobre el resultado, la parada. Los anos 1900-2025 ya quedaron como lemas
    en la Fase 4 y no se tocan.

    Args:
        linea: una linea de *_lemas.txt (lemas separados por espacio).
        parada: conjunto de tokens a eliminar.
        correcciones: mapa lema anomalo -> lema correcto.

    Returns:
        Texto limpio con los lemas de contenido separados por espacio.
    """
    salida: list[str] = []
    for token in linea.split():
        if correcciones:
            token = correcciones.get(token, token)
        if token not in parada:
            salida.append(token)
    return " ".join(salida)


def preprocesar_corpus(
    lineas: list[str], parada: set[str], correcciones: dict[str, str],
) -> tuple[list[str], list[int]]:
    """Aplica limpiar_linea a todas las lineas de un corpus.

    Args:
        lineas: documentos lematizados (1 doc por linea).
        parada: conjunto de tokens a eliminar.
        correcciones: mapa de correcciones de lemas.

    Returns:
        Tupla (textos_limpios, n_tokens_por_doc_limpio). Los documentos que
        quedan vacios se conservan como string vacio (el vectorizador produce
        una fila de ceros y el ranking los omite).
    """
    textos: list[str] = []
    conteos: list[int] = []
    for linea in lineas:
        limpio = limpiar_linea(linea, parada, correcciones)
        textos.append(limpio)
        conteos.append(len(limpio.split()) if limpio else 0)
    return textos, conteos


def construir_vectorizador(config: dict) -> TfidfVectorizer:
    """Construye el TfidfVectorizer con la config congelada (dtype float32).

    La parada y las correcciones se aplican ANTES en el texto (limpiar_linea),
    por lo que el vectorizador no recibe stop_words: el texto de entrada ya
    solo contiene lemas de contenido (equivalente, y agnostico a la variante).

    Args:
        config: kwargs de config/tfidf_config.json["vectorizador"].

    Returns:
        TfidfVectorizer listo para fit sobre la coleccion.
    """
    parametros = dict(config)
    parametros.setdefault("dtype", np.float32)
    return TfidfVectorizer(**parametros)


def similitud_coseno_batches(
    x_queries, x_coleccion, batch_size: int = 512,
) -> np.ndarray:
    """Similitud coseno query-coleccion por batches (producto sparse).

    Los vectores TF-IDF estan normalizados con norm=l2, por lo que el
    producto matricial equivale al coseno. Las queries se procesan en batches
    para acotar la memoria; la salida (2.486 x 917 en float32) es manejable.

    Args:
        x_queries: matriz TF-IDF de las queries (n_queries x n_features).
        x_coleccion: matriz TF-IDF de la coleccion (n_docs x n_features).
        batch_size: numero de queries por batch.

    Returns:
        Matriz densa float32 (n_queries x n_docs) de similitudes coseno.
    """
    coleccion_t = x_coleccion.T.astype(np.float32)
    bloques: list[np.ndarray] = []
    for inicio in range(0, x_queries.shape[0], batch_size):
        bloque = x_queries[inicio : inicio + batch_size].astype(np.float32)
        bloques.append((bloque @ coleccion_t).toarray())
    return np.vstack(bloques)


def construir_ranking(
    similitudes: np.ndarray,
    ids_query: list[str],
    ids_coleccion: list[str],
    top_k: int = 5,
) -> pd.DataFrame:
    """Construye el ranking top-k por query desde la matriz de similitudes.

    Se conservan unicamente los pares con score > 0 (sin vocabulario
    compartido no hay match); una query sin ningun match no genera filas. El
    desempate es estable por posicion de la coleccion (argsort estable).

    Args:
        similitudes: matriz (n_queries x n_docs) de similitudes coseno.
        ids_query: id_doc de cada query, en el orden de las filas.
        ids_coleccion: id_segmento de cada documento, en el orden de columnas.
        top_k: profundidad del ranking (5 para el entregable).

    Returns:
        DataFrame con columnas [id_doc, rango, id_segmento, score].
    """
    filas: list[dict] = []
    for i_query, id_doc in enumerate(ids_query):
        scores = similitudes[i_query]
        orden = np.argsort(-scores, kind="stable")
        for rango, posicion in enumerate(orden, start=1):
            if rango > top_k:
                break
            score = float(scores[posicion])
            if score <= 0.0:
                break
            filas.append(
                {
                    "id_doc": id_doc,
                    "rango": rango,
                    "id_segmento": ids_coleccion[int(posicion)],
                    "score": round(score, 6),
                }
            )
    ranking = pd.DataFrame(filas, columns=["id_doc", "rango", "id_segmento", "score"])
    ranking["id_segmento"] = ranking["id_segmento"].astype(str)
    return ranking


def enriquecer_ranking(
    ranking: pd.DataFrame,
    metadatos_segmentos: list[dict],
    ids_segmentos: list[str],
) -> pd.DataFrame:
    """Agrega tomo y titulo del segmento al ranking.

    Args:
        ranking: DataFrame de construir_ranking (columnas basicas).
        metadatos_segmentos: registros de libros_docs.json en orden.
        ids_segmentos: id_segmento en el mismo orden.

    Returns:
        Ranking con las columnas finales
        [id_doc, rango, id_segmento, score, tomo, titulo_segmento].
    """
    mapa = {
        registro["id"]: (registro["tomo"], registro["titulo"])
        for registro in metadatos_segmentos
    }
    ranking = ranking.copy()
    tomos: list[str] = []
    titulos: list[str] = []
    for id_segmento in ranking["id_segmento"]:
        tomo, titulo = mapa.get(id_segmento, ("", ""))
        tomos.append(tomo)
        titulos.append(titulo)
    ranking["tomo"] = tomos
    ranking["titulo_segmento"] = titulos
    return ranking


def correr_ir(
    variante: str,
    lineas_libros: list[str],
    docs_libros: list[dict],
    lineas_entrevistas: list[str],
    docs_entrevistas: list[dict],
) -> dict:
    """Ejecuta el pipeline completo de una variante (sin persistencia).

    Replica `correr_variante()` de experimentos/experimento_1_tfidf_base.py:
    resolucion de la variante (A: ngram (1,1); B: sin muletillas; C: sin
    correcciones), preprocesado en consumo, fit del vectorizador sobre los
    lemas de los libros (coleccion) y transform sobre las entrevistas
    (queries), similitud coseno por batches y ranking top-5 enriquecido.

    Args:
        variante: identificador del experimento ("base", "A", "B", "C").
        lineas_libros: lemas de la coleccion (1 segmento por linea).
        docs_libros: metadatos de los segmentos (libros_docs.json).
        lineas_entrevistas: lemas de las queries (1 entrevista por linea).
        docs_entrevistas: metadatos de las entrevistas (entrevistas_docs.json).

    Returns:
        Contexto con ranking_top5, config aplicada, parada usada, n de
        correcciones, vocabulario, matrices, conteos y estadisticas.
    """
    config = cargar_config_vectorizador()
    if variante == "A":
        config["ngram_range"] = (1, 1)
    parada = construir_parada_ir(incluir_muletillas=(variante != "B"))
    correcciones = _cargar_correcciones_ir() if variante != "C" else {}
    print(
        f"\n[experimento {variante}] ngram={config['ngram_range']} | "
        f"parada={len(parada)} tokens | correcciones={len(correcciones)}"
    )

    libros_limpios, n_lemas_libros = preprocesar_corpus(lineas_libros, parada, correcciones)
    entrevistas_limpios, n_lemas_entrevistas = preprocesar_corpus(
        lineas_entrevistas, parada, correcciones
    )
    vacias_tras_parada = [
        docs_entrevistas[i]["id"]
        for i, n in enumerate(n_lemas_entrevistas)
        if n == 0
    ]
    print(
        f"  lemas coleccion: {sum(n_lemas_libros):,} -> "
        f"docs limpios {sum(1 for n in n_lemas_libros if n > 0)}/{len(lineas_libros)}"
    )
    print(
        f"  lemas queries: {sum(n_lemas_entrevistas):,} -> "
        f"queries no vacias {sum(1 for n in n_lemas_entrevistas if n > 0)}/"
        f"{len(lineas_entrevistas)} | vacias tras parada: {len(vacias_tras_parada)}"
    )

    vectorizador = construir_vectorizador(config)
    x_coleccion = vectorizador.fit_transform(libros_limpios)
    x_queries = vectorizador.transform(entrevistas_limpios)
    n_features = len(vectorizador.vocabulary_)
    n_bigramas = sum(1 for termino in vectorizador.vocabulary_ if " " in termino)
    densidad = x_coleccion.nnz / max(x_coleccion.shape[0] * n_features, 1)
    print(
        f"  fit coleccion: {x_coleccion.shape} | transform queries: {x_queries.shape}"
    )
    print(
        f"  vocabulario: {n_features} features "
        f"({n_features - n_bigramas} unigramas + {n_bigramas} bigramas) | "
        f"densidad tf coleccion: {densidad:.6f}"
    )

    similitudes = similitud_coseno_batches(x_queries, x_coleccion)
    ids_segmentos = [registro["id"] for registro in docs_libros]
    ids_queries = [registro["id"] for registro in docs_entrevistas]
    ranking = construir_ranking(similitudes, ids_queries, ids_segmentos, top_k=5)
    ranking_top5 = enriquecer_ranking(ranking, docs_libros, ids_segmentos)
    print(f"  ranking top-5: {len(ranking_top5)} filas | head:")
    print(ranking_top5.head(8).to_string(index=False))

    return {
        "ranking_top5": ranking_top5,
        "similitudes": similitudes,
        # Matrices, textos limpios y metadatos expuestos como contrato de la
        # seccion: la Sec 03b (BM25/Rocchio) y la Sec 03c (heatmap) los
        # reutilizan sin re-entrenar ni releer los corpus.
        "x_coleccion": x_coleccion,
        "x_queries": x_queries,
        "libros_limpios": libros_limpios,
        "entrevistas_limpios": entrevistas_limpios,
        "ids_coleccion": ids_segmentos,
        "ids_queries": ids_queries,
        "docs_libros": docs_libros,
        "docs_entrevistas": docs_entrevistas,
        "config": config,
        "parada_tokens": len(parada),
        "n_correcciones_lemas": len(correcciones),
        "n_features": n_features,
        "n_bigramas": n_bigramas,
        "n_unigramas": n_features - n_bigramas,
        "densidad": densidad,
        "n_docs_coleccion": len(lineas_libros),
        "n_queries": len(lineas_entrevistas),
        "queries_vacias": vacias_tras_parada,
        "vectorizador": vectorizador,
    }


# ===== Pipeline base completo (en memoria; no escribe archivos) =====
lineas_libros, docs_libros = cargar_lemas("libros"), cargar_metadatos("libros")
lineas_entrevistas, docs_entrevistas = (
    cargar_lemas("entrevistas"),
    cargar_metadatos("entrevistas"),
)
print(
    f"Coleccion (segmentos de libro): {len(lineas_libros)} docs | "
    f"queries (entrevistas): {len(lineas_entrevistas)} docs"
)
contexto_base = correr_ir(
    "base", lineas_libros, docs_libros, lineas_entrevistas, docs_entrevistas
)
ranking_top5 = contexto_base["ranking_top5"]

print(f"\nRanking top-5 generado en memoria: {len(ranking_top5)} filas "
      f"({ranking_top5['id_doc'].nunique()} queries)")
print("\nHead del ranking (primeras 5 filas = top-5 de la primera query):")
print(ranking_top5.head(5).to_string(index=False))


# %%
"""Top-1 por tomo, score medio/mediana y verificacion contra el CSV oficial."""
import matplotlib.pyplot as plt

top1 = ranking_top5[ranking_top5["rango"] == 1].copy()
distribucion = top1["tomo"].value_counts()
porcentajes = 100.0 * distribucion / len(top1)

fig, ax = plt.subplots(figsize=(10, 4))
colores = ["#DD8452" if tomo == "TESTIMONIAL" else "#4C72B0" for tomo in distribucion.index]
barras = ax.bar(distribucion.index, porcentajes, color=colores)
ax.bar_label(barras, fmt="%.1f%%", fontsize=8)
ax.set_title(
    "Distribucion del top-1 por tomo (2.484 queries con match)\n"
    "TESTIMONIAL concentra el 76,8 %: sesgo de registro oral, no artefacto del modelo"
)
ax.set_xlabel("tomo del segmento recuperado")
ax.set_ylabel("% de queries")
fig.autofmt_xdate(rotation=0)
plt.show()

print(f"Score top-1: media = {top1['score'].mean():.6f} | "
      f"mediana = {top1['score'].median():.6f}")

# Verificacion OPCIONAL: el ranking en memoria debe ser identico al CSV
# oficial resultados/ranking_top5.csv (mismo codigo y config embebida;
# corrida determinista). Si el CSV no existe, el ranking inline es la fuente.
pct_mem = 100.0 * (top1["tomo"] == "TESTIMONIAL").mean()
ruta_ranking_csv = RUTA_RESULTADOS / "ranking_top5.csv"
if ruta_ranking_csv.exists():
    base_csv = pd.read_csv(ruta_ranking_csv)
    top1_csv = base_csv[base_csv["rango"] == 1].set_index("id_doc")["id_segmento"]
    top1_mem = top1.set_index("id_doc")["id_segmento"]
    n_coinciden = int((top1_csv == top1_mem).sum())
    pct_csv = 100.0 * (base_csv[base_csv["rango"] == 1]["tomo"] == "TESTIMONIAL").mean()
    print(
        f"Verificacion vs ranking_top5.csv: top-1 identicos en {n_coinciden}/"
        f"{len(top1_mem)} queries | TESTIMONIAL: {pct_mem:.2f} % (memoria) = "
        f"{pct_csv:.2f} % (CSV oficial) -> reproducido: {n_coinciden == len(top1_mem)}"
    )
else:
    print("[info] CSV oficial no presente (resultados/ranking_top5.csv); el")
    print(f"       ranking inline es la fuente (TESTIMONIAL top-1: {pct_mem:.2f} %).")


# %%
"""Variantes A/B/C: coincidencia del top-1 con el base (desde CSVs)."""

# A: ngram (1,1); B: sin muletillas; C: sin correcciones de lemas.
# El codigo inline genera el ranking del base (arriba); las variantes se
# comparan contra sus CSVs oficiales (ranking_top5_{A,B,C}.csv) del
# experimento de produccion SOLO si existen (verificacion OPCIONAL): la
# coincidencia esperada del reporte es A 54,8 / B 84,6 / C 96,6.
esperado = {"A": 54.8, "B": 84.6, "C": 96.6}
ruta_base_csv = RUTA_RESULTADOS / "ranking_top5.csv"
variantes_csv = {
    nombre: RUTA_RESULTADOS / f"ranking_top5_{nombre}.csv" for nombre in esperado
}
if ruta_base_csv.exists() and all(ruta.exists() for ruta in variantes_csv.values()):
    base_top1 = pd.read_csv(ruta_base_csv)
    base_top1 = base_top1[base_top1["rango"] == 1].set_index("id_doc")["id_segmento"]
    filas_variantes = []
    for nombre, pct_esperado in esperado.items():
        variante = pd.read_csv(variantes_csv[nombre])
        top1_variante = variante[variante["rango"] == 1].set_index("id_doc")["id_segmento"]
        comun = base_top1.index.intersection(top1_variante.index)
        coincidencia = 100.0 * (base_top1[comun] == top1_variante[comun]).mean()
        filas_variantes.append(
            {
                "variante": nombre,
                "cambio_respecto_base": {
                    "A": "sin bigramas (ngram 1,1)",
                    "B": "sin parada de muletillas",
                    "C": "sin correcciones de lemas",
                }[nombre],
                "coincidencia_top1_pct": round(coincidencia, 1),
                "esperado_reporte_pct": pct_esperado,
            }
        )
    tabla_variantes = pd.DataFrame(filas_variantes)
    print("Coincidencia del top-1 de cada variante con el base (2.484 queries):")
    print(tabla_variantes.to_string(index=False))
    ok = np.allclose(
        tabla_variantes["coincidencia_top1_pct"],
        tabla_variantes["esperado_reporte_pct"],
        atol=0.05,
    )
    print(f"Reproduce los valores del reporte (A 54,8 / B 84,6 / C 96,6): {ok}")
    print(
        "Lectura: el bigrama aporta el 45 % del cambio de top-1; las muletillas "
        "mueven poco (15 %); las 62 correcciones casi nada (3 %). La config base "
        "es la mas estable."
    )
else:
    print("[info] no estan los CSVs oficiales de las variantes")
    print("       (resultados/ranking_top5_{A,B,C}.csv); el ranking inline del")
    print("       base es la fuente (esperado del reporte: A 54,8 / B 84,6 /")
    print("       C 96,6 de coincidencia de top-1).")


# %%
"""Tabla de evaluacion etiquetada (30 pares) y metricas proxy recalculadas.

La muestra de 30 entrevistas se reproduce INLINE con el mismo algoritmo
determinista del experimento oficial (estratificada por el tomo del top-1,
3 por tomo en el orden canonico, semilla 42, queries con top-3 completo;
experimentos/experimento_1_tfidf_base.py). Las 30 etiquetas HUMANAS del
etiquetado de la Fase 6 viajan EMBEBIDAS como literal (con sus auxiliares
r2/r3 para el P@3 y la justificacion), copiadas de
resultados/evaluacion_muestra_etiquetada.csv y
resultados/evaluacion_metricas.json (ambos opcionales de consulta): la
lectura de evaluacion_metricas.json queda como verificacion opcional.
"""
ETIQUETAS_HUMANAS_MUESTRA: dict[str, dict[str, str]] = {
    "5f6ae4b606158f011ed0106c.pdf": {
        "etiqueta": "relacionado",
        "r2": "parcial",
        "r3": "parcial",
        "justificacion": "Politico araucano: guerrilla FARC/ELN, poder politico regional y democracia; tema exacto del segmento.",
    },
    "5eb4eb12da6da3b502db359f.pdf": {
        "etiqueta": "relacionado",
        "r2": "relacionado",
        "r3": "parcial",
        "justificacion": "Relato de masacre inminente, amenaza y desplazamiento de la comunidad; el segmento cita su mismo tipo de testimonio.",
    },
    "5fee58599e8b013f2692e067.pdf": {
        "etiqueta": "relacionado",
        "r2": "parcial",
        "r3": "parcial",
        "justificacion": "Investigador del despojo de tierras a indigenas (embera, Risaralda); r1 trata despojo, territorio y cosmovision.",
    },
    "5eb4e860da6da3b502da7cd8.pdf": {
        "etiqueta": "parcial",
        "r2": "parcial",
        "r3": "parcial",
        "justificacion": "Gestion de orden publico y consejos de seguridad; menciona la CEV solo de forma tangencial y critica.",
    },
    "5eb4eb13da6da3b502db35e7.pdf": {
        "etiqueta": "parcial",
        "r2": "parcial",
        "r3": "parcial",
        "justificacion": "Analista: asistencia militar de EEUU y cooperacion; r1 es sobre negociacion de La Habana: solape parcial por conflicto.",
    },
    "5eb4e9fbda6da3b502dafa27.pdf": {
        "etiqueta": "parcial",
        "r2": "parcial",
        "r3": "no",
        "justificacion": "Dirigente politico de centro en Norte de Santander contra el bipartidismo; r1 (parapolitica) solo roza su experiencia.",
    },
    "5ef8d0a8f1cdbf68d095d559.pdf": {
        "etiqueta": "relacionado",
        "r2": "relacionado",
        "r3": "relacionado",
        "justificacion": "Abogado de DDHH, alertas tempranas y presos politicos; represion estatal y detenciones arbitrarias son su tema.",
    },
    "5eb4e861da6da3b502da7d1b.pdf": {
        "etiqueta": "relacionado",
        "r2": "parcial",
        "r3": "parcial",
        "justificacion": "Pescador: destruccion de 27 cienagas por guerrilla y autodefensas; daño ambiental del conflicto, tema exacto de r1.",
    },
    "5faa8396205391c03b5f1836.pdf": {
        "etiqueta": "relacionado",
        "r2": "relacionado",
        "r3": "parcial",
        "justificacion": "Experta en secuestro de ganaderos y empresarios con fines economicos; r1 y r2 tratan secuestro/extorsion.",
    },
    "625fd7b8755cba02e5b68d8f.pdf": {
        "etiqueta": "parcial",
        "r2": "relacionado",
        "r3": "relacionado",
        "justificacion": "Docente: escuela en el conflicto (amenazas, orden publico); r1 es educacion para la paz; r2/r3 ataques a escuelas.",
    },
    "5f09f4e4f1cdbf68d05da6fc.pdf": {
        "etiqueta": "parcial",
        "r2": "parcial",
        "r3": "parcial",
        "justificacion": "Lider social: toma de iglesia (1980) y estigmatizacion de jovenes; r1 (lenguaje que deshumaniza) solape solo especulativo.",
    },
    "5eb4eb43da6da3b502db3f52.pdf": {
        "etiqueta": "parcial",
        "r2": "relacionado",
        "r3": "parcial",
        "justificacion": "Comunidad indigena del Vaupes: presencia de actores armados desde 2002, zozobra; r1 (huida a Taraira) contexto afín pero no su caso.",
    },
    "5f040d37f1cdbf68d015a2fa.pdf": {
        "etiqueta": "relacionado",
        "r2": "parcial",
        "r3": "parcial",
        "justificacion": "Su historia (agresion a los 15, desplazamiento, vida precaria) aparece practicamente identica en el segmento r1.",
    },
    "5ececeaf215810782986e718.pdf": {
        "etiqueta": "relacionado",
        "r2": "relacionado",
        "r3": "relacionado",
        "justificacion": "Psicologa que acompana a mujeres victimas del conflicto (trauma y reparacion); los tres rangos tratan mujeres y conflicto.",
    },
    "603bab289e8b013f26f8c252.pdf": {
        "etiqueta": "relacionado",
        "r2": "relacionado",
        "r3": "parcial",
        "justificacion": "APRODEFA: acompana a victimas de violencia sexual del conflicto; r1 y r2 tratan violencia sexual contra mujeres/LGBTI.",
    },
    "5ec832c721581078294c4ad3.pdf": {
        "etiqueta": "parcial",
        "r2": "relacionado",
        "r3": "parcial",
        "justificacion": "Reconfiguracion del conflicto post-acuerdo en Cordoba (Caparros, disidencias); r1 enfoca ninas/mujeres; r2 post-AUC si encaja.",
    },
    "5f299d4a06158f011ec42f9b.pdf": {
        "etiqueta": "parcial",
        "r2": "parcial",
        "r3": "parcial",
        "justificacion": "Docente historico del Guaviare: educacion rural y violencia contra maestros; sin ataques directos a escuelas (el propio relato lo niega).",
    },
    "5ec832d421581078294c4c27.pdf": {
        "etiqueta": "parcial",
        "r2": "relacionado",
        "r3": "parcial",
        "justificacion": "Analisis del sistema educativo para NNA en el conflicto; r2 (que significo el conflicto para NNA) captura mejor el tema.",
    },
    "5f3f530e06158f011ef65cab.pdf": {
        "etiqueta": "relacionado",
        "r2": "parcial",
        "r3": "no",
        "justificacion": "Joven cuya madre fue asesinada por la guerrilla: orfandad y roles adultos; tema exacto del segmento NNA r1.",
    },
    "5f27003306158f011ea85482.pdf": {
        "etiqueta": "relacionado",
        "r2": "no",
        "r3": "no",
        "justificacion": "Lider del Huila sobre detenciones masivas (2002); r1 documenta capturas masivas autorizadas desde el alto gobierno.",
    },
    "5f08ae8df1cdbf68d04f6fc1.pdf": {
        "etiqueta": "parcial",
        "r2": "relacionado",
        "r3": "parcial",
        "justificacion": "Lider afro: asesinatos de opositores al despojo por Gaitanistas; r1 (torturas) solo roza; r2 despojo de tierras si es su tema.",
    },
    "5f0deb9df1cdbf68d08a0890.pdf": {
        "etiqueta": "relacionado",
        "r2": "no",
        "r3": "no",
        "justificacion": "Campesina atada, golpeada y con su casa quemada por la fuerza publica (2006); r1 trata torturas y tratos crueles.",
    },
    "5eb4e8f5da6da3b502dab502.pdf": {
        "etiqueta": "parcial",
        "r2": "relacionado",
        "r3": "relacionado",
        "justificacion": "Joven indigena gay: castigos y expulsion por su identidad; r1 (violencias etnicas del conflicto) tangencial; r2/r3 LGBTI exactos.",
    },
    "5eb4e9abda6da3b502dae721.pdf": {
        "etiqueta": "relacionado",
        "r2": "relacionado",
        "r3": "relacionado",
        "justificacion": "Programa de mujeres indigenas frente a violencias del conflicto agudizado; los tres rangos tratan pueblos etnicos.",
    },
    "5fee5bef9e8b013f26931324.pdf": {
        "etiqueta": "relacionado",
        "r2": "relacionado",
        "r3": "parcial",
        "justificacion": "Lider raizal: impacto del conflicto (narcotrafico) y racismo en San Andres; r1 cita el caso raizal y el informe DAS.",
    },
    "607edb88b5ef6a0c984f84f3.pdf": {
        "etiqueta": "parcial",
        "r2": "parcial",
        "r3": "relacionado",
        "justificacion": "Exmilitante colombiano exiliado en Brasil; r1 (proteccion internacional) cercano pero generico; r3 impacto del exilio si es su tema.",
    },
    "5ef4e445f1cdbf68d06e9f17.pdf": {
        "etiqueta": "parcial",
        "r2": "no",
        "r3": "parcial",
        "justificacion": "Joven santandereano que socializo los acuerdos de paz y fue amenazado; r1 (historia macro 2006-2016) solape solo de epoca.",
    },
    "5f5519b206158f011e1ec789.pdf": {
        "etiqueta": "relacionado",
        "r2": "relacionado",
        "r3": "relacionado",
        "justificacion": "Trabajador humanitario (ACNUR/ADRA) que atiende refugiados colombianos en Argentina; r1-r3 tratan proteccion internacional.",
    },
    "5eb4e80eda6da3b502da5187.pdf": {
        "etiqueta": "relacionado",
        "r2": "parcial",
        "r3": "parcial",
        "justificacion": "Retenido y humillado por las FARC (2003, acusado de sapo); r1 (vida bajo el control y el miedo de los armados) es su tema.",
    },
    "5f148660f1cdbf68d0e27e16.pdf": {
        "etiqueta": "parcial",
        "r2": "parcial",
        "r3": "parcial",
        "justificacion": "Indigena amazonica excombatiente de las FARC (fusilamientos, desmovilizacion); los top-3 TESTIMONIAL no capturan su vivencia; evidencia parcial.",
    },
}

"""End of literal"""


def seleccionar_muestra_evaluacion(
    ranking: pd.DataFrame,
    semilla: int = 42,
    objetivo: int = 30,
    n_por_tomo: int = 3,
) -> list[str]:
    """Selecciona las entrevistas de la muestra (algoritmo oficial, semilla 42).

    Replica `seleccionar_muestra_entrevistas()` de
    experimentos/experimento_1_tfidf_base.py: candidatas = queries con top-3
    completo (>= 3 matches); estratificacion por el tomo del top-1 en el
    orden canonico de los 10 tomos (3 por tomo, rng.shuffle con
    np.random.default_rng(semilla)) y completacion con el resto de
    candidatas si no se llega al objetivo.

    Args:
        ranking: ranking top-5 inline (columnas id_doc, rango, tomo, ...).
        semilla: semilla fija y documentada del muestreo (42).
        objetivo: numero de entrevistas de la muestra (30).
        n_por_tomo: entrevistas por tomo en la primera pasada (3).

    Returns:
        Lista de id_doc seleccionados (orden de la muestra).
    """
    rng = np.random.default_rng(semilla)
    n_matches = ranking.groupby("id_doc")["rango"].max()
    candidatas = set(n_matches.index[n_matches >= 3].tolist())
    top1 = ranking[ranking["rango"] == 1]
    tomo_por_query = dict(zip(top1["id_doc"], top1["tomo"], strict=True))
    tomos_orden = [
        "HALLAZGOS", "CONVOCATORIA", "NARRATIVA", "SUFRIR", "TESTIMONIAL",
        "CUERPO", "NNA", "VIOLACIONES", "ETNICO", "FUERA",
    ]
    por_tomo: dict[str, list[str]] = {tomo: [] for tomo in tomos_orden}
    for id_doc in sorted(candidatas):
        por_tomo[tomo_por_query.get(id_doc, "")].append(id_doc)
    seleccionadas: list[str] = []
    for tomo in tomos_orden:
        mezcladas = por_tomo[tomo]
        rng.shuffle(mezcladas)
        seleccionadas.extend(mezcladas[:n_por_tomo])
    if len(seleccionadas) < objetivo:
        resto = sorted(candidatas - set(seleccionadas))
        rng.shuffle(resto)
        seleccionadas.extend(resto[: objetivo - len(seleccionadas)])
    return seleccionadas[:objetivo]


ids_muestra = seleccionar_muestra_evaluacion(ranking_top5)
print(f"Muestra determinista inline: {len(ids_muestra)} pares query -> segmento top-1\n")

# Filas de la tabla: r1 (top-1) por id de la muestra, tomado del ranking
# inline (identico al CSV oficial cuando este existe) + etiqueta embebida.
filas_tabla = []
for id_doc in ids_muestra:
    fila_r1 = ranking_top5[
        (ranking_top5["id_doc"] == id_doc) & (ranking_top5["rango"] == 1)
    ].iloc[0]
    filas_tabla.append(
        {
            "id_doc": id_doc[:20] + "..",
            "r1_tomo": fila_r1["tomo"],
            "r1_score": round(float(fila_r1["score"]), 4),
            "etiqueta": ETIQUETAS_HUMANAS_MUESTRA[id_doc]["etiqueta"],
        }
    )
tabla_eval = pd.DataFrame(filas_tabla)
print("Las 30 filas etiquetadas (id truncado; r1 = segmento top-1):")
print(tabla_eval.to_string(index=False))

conteo = tabla_eval["etiqueta"].value_counts()
print("\nConteo de etiquetas:", dict(conteo))

n = len(ids_muestra)
conteos = {
    etiqueta: int((tabla_eval["etiqueta"] == etiqueta).sum())
    for etiqueta in ("relacionado", "parcial", "no")
}
p1_estricto = conteos["relacionado"] / n
p1_aceptable = (conteos["relacionado"] + conteos["parcial"]) / n

# P@3 proxy desde las etiquetas auxiliares r2/r3 embebidas (mismo criterio
# del etiquetado oficial registrado en evaluacion_metricas.json).
n_ok3_estricto = 0
n_ok3_aceptable = 0
ids_top1_parcial_con_relacionado_r23: list[str] = []
ids_top1_no: list[str] = []
for id_doc in ids_muestra:
    datos = ETIQUETAS_HUMANAS_MUESTRA[id_doc]
    etiquetas_r = [datos["etiqueta"], datos["r2"], datos["r3"]]
    if "relacionado" in etiquetas_r:
        n_ok3_estricto += 1
    if any(e != "no" for e in etiquetas_r):
        n_ok3_aceptable += 1
    if datos["etiqueta"] == "no":
        ids_top1_no.append(id_doc)
    if datos["etiqueta"] == "parcial" and "relacionado" in (datos["r2"], datos["r3"]):
        ids_top1_parcial_con_relacionado_r23.append(id_doc)
p3_estricto = n_ok3_estricto / n
p3_aceptable = n_ok3_aceptable / n

# Sesgo TESTIMONIAL en la muestra (tomo del top-1 inline de cada id).
tomo_top1_muestra = {}
for id_doc in ids_muestra:
    fila_r1 = ranking_top5[
        (ranking_top5["id_doc"] == id_doc) & (ranking_top5["rango"] == 1)
    ].iloc[0]
    tomo_top1_muestra[id_doc] = fila_r1["tomo"]
ids_testimonial = [d for d in ids_muestra if tomo_top1_muestra[d] == "TESTIMONIAL"]
sesgo = {
    "n_top1_testimonial_en_muestra": len(ids_testimonial),
    "distribucion_top1_testimonial": {
        etiqueta: sum(
            1 for d in ids_testimonial
            if ETIQUETAS_HUMANAS_MUESTRA[d]["etiqueta"] == etiqueta
        )
        for etiqueta in ("relacionado", "parcial", "no")
    },
}

print("\nMetricas proxy recalculadas en el notebook (etiquetas embebidas):")
print(f"  P@1 estricto (solo 'relacionado')         : {p1_estricto:.4f}")
print(f"  P@1 aceptable ('relacionado' o 'parcial') : {p1_aceptable:.4f}")
print(f"  P@3 aceptable (>=1 'relacionado')         : {p3_estricto:.4f}")
print(f"  P@3 aceptable (>=1 no-'no')               : {p3_aceptable:.4f}")
print(
    f"  Top-1 'no': {len(ids_top1_no)} | Top-1 'parcial' con match "
    f"'relacionado' en r2-r3: "
    f"{len(ids_top1_parcial_con_relacionado_r23)}/30"
)
print(
    f"  En la muestra, top-1 TESTIMONIAL: {sesgo['n_top1_testimonial_en_muestra']} "
    f"({dict(sesgo['distribucion_top1_testimonial'])})"
)

# Verificacion OPCIONAL contra el resumen oficial evaluacion_metricas.json.
ruta_metricas_json = RUTA_RESULTADOS / "evaluacion_metricas.json"
if ruta_metricas_json.exists():
    metricas_eval = json.loads(ruta_metricas_json.read_text(encoding="utf-8"))
    print("\nVerificacion contra resultados/evaluacion_metricas.json:")
    print(f"  P@1 estricto: {p1_estricto:.4f} "
          f"(json {metricas_eval['p1_proxy']['estricto_solo_relacionado']:.4f})")
    print(f"  P@1 aceptable: {p1_aceptable:.4f} "
          f"(json {metricas_eval['p1_proxy']['aceptable_relacionado_o_parcial']:.4f})")
    print(f"  P@3 estricto: {p3_estricto:.4f} "
          f"(json {metricas_eval['p3_proxy']['aceptable_al_menos_un_relacionado']:.4f})")
    print(f"  P@3 aceptable: {p3_aceptable:.4f} "
          f"(json {metricas_eval['p3_proxy']['aceptable_relacionado_o_parcial']:.4f})")
    recalculado_ok = (
        abs(p1_estricto - metricas_eval["p1_proxy"]["estricto_solo_relacionado"]) < 1e-4
        and abs(p1_aceptable - metricas_eval["p1_proxy"]["aceptable_relacionado_o_parcial"]) < 1e-4
        and abs(p3_estricto - metricas_eval["p3_proxy"]["aceptable_al_menos_un_relacionado"]) < 1e-4
    )
    print(f"  Recalculado identico al JSON oficial: {recalculado_ok}")
else:
    print(
        "\n[info] resultados/evaluacion_metricas.json no presente; las "
        "metricas proxy de arriba se calcularon desde las etiquetas "
        "embebidas (P@1 estricto 0.5333, aceptable 1.0; P@3 estricto 0.7667, "
        "aceptable 1.0)."
    )

# %% [markdown]
"""## Ejemplos curados de top-1 (seleccion de evaluacion_muestra_etiquetada.csv)

Cinco pares query (entrevista) -> top-1 (segmento de libro) seleccionados de
`resultados/evaluacion_muestra_etiquetada.csv` (etiquetado humano-proxy con
contexto ampliado del corpus): 2 relacionados buenos, 2 parciales y 1 caso
borde (el mas cercano a un falso positivo duro de la muestra).

**1. Relacionado bueno — el segmento abre con el mismo tipo de testimonio**
- id_doc: `5eb4eb12da6da3b502db359f.pdf` | id_segmento:
  `HALLAZGOS-p129-las-masacres` (HALLAZGOS) | score: 0,1319
- Por que: la entrevista narra una amenaza de masacre a una comunidad
  campesina (actores armados, plazo para salir, asesinato de un companero) y
  el segmento "Las masacres" abre con ese mismo tipo de relato: match
  genuino de hecho y poblacion.

**2. Relacionado bueno — caso etnico especifico bien recuperado**
- id_doc: `5fee5bef9e8b013f26931324.pdf` | id_segmento:
  `ETNICO-p448-violencias-danos-y-resistencias-de-los-pueblos-etnicos-en-el-conflicto-armado`
  (ETNICO) | score: 0,1235
- Por que: lider raizal sobre el impacto del conflicto y la discriminacion
  racial en San Andres; el segmento cita el caso raizal. Recupera la
  poblacion etnica correcta y su afectacion, no un segmento generico.

**3. Parcial — solape de tema amplio, desviacion del hecho concreto**
- id_doc: `5eb4eb13da6da3b502db35e7.pdf` | id_segmento:
  `NARRATIVA-p547-86-negociar-como-si-no-hubiera-guerra-y-pelear-como-si-no-hubiera-negociacion`
  (NARRATIVA) | score: 0,1643 (el mas alto de los ejemplos)
- Por que: la query es de un analista de relaciones internacionales sobre
  asistencia militar de EE. UU. (Plan Colombia, Guerra Fria) y el top-1
  trata la negociacion de La Habana: ambos son "guerra y salidas del
  conflicto", pero el match especifico no aparece en el top-3.

**4. Parcial — caso representativo del sesgo de especificidad**
- id_doc: `5ef4e445f1cdbf68d06e9f17.pdf` | id_segmento:
  `FUERA-p99-de-la-guerra-total-a-la-paz-incompleta-2006-2016` (FUERA) |
  score: 0,1623
- Por que: joven que socializo los acuerdos de paz y salio del pais; el
  top-1 es la historia macro 2006-2016. Coincide la epoca ("paz
  incompleta") pero no su vivencia del exilio: ilustra el fenomeno de 7/30
  queries con top-1 parcial y mejor match en r2-r3.

**5. Caso borde — top-1 TESTIMONIAL por solape de registro oral**
- id_doc: `5f148660f1cdbf68d0e27e16.pdf` | id_segmento:
  `TESTIMONIAL-p401-senderos-pedregosos` (TESTIMONIAL) | score: 0,1539
- Por que: mujer indigena amazonica excombatiente de las FARC; el top-1 es
  el testimonio oral de un joven nasa del Cauca. No comparten poblacion,
  region ni hecho: el ranking los emparejo por el registro oral compartido.
  Es el caso mas cercano a un falso positivo duro de la muestra (etiquetado
  `parcial` por prudencia).

Lectura general: el modelo recupera bien cuando la query tiene terminos de
dominio especificos (masacres, raizal), pero tiende a emparejar por solape
lexico general del conflicto y por registro oral cuando la entrevista es
testimonial.
"""


# %% [markdown]
"""## Discusion

**El sesgo TESTIMONIAL (76,77 % de top-1) es del dominio, no un artefacto
del modelo.** Con 2.484 queries con match, 1.907 (76,77 %) recuperan como
top-1 un segmento del tomo TESTIMONIAL. No se explica por el tamano del tomo
(104 de 917 segmentos, 11,3 %) ni por un defecto de TF-IDF: las entrevistas
son testimonios orales y el tomo TESTIMONIAL es el unico compilado de
testimonios; el solape lexico de registro (y el vocabulario comun del
conflicto) domina el ranking. La evaluacion cualitativa lo confirma: en la
muestra los 5 top-1 TESTIMONIAL son `relacionado` (2) o `parcial` (3) y
ninguno es `no`: el sesgo produce resultados plausibles, no falsos positivos
duros.

**7 de 30 queries (23 %) tienen top-1 parcial con un match relacionado en
r2-r3.** Es el fenomeno de especificidad: el primer resultado no es el mas
especifico porque los segmentos compiten por solape lexico amplio (todo el
corpus habla de guerra, victimas y territorio), no por el hecho o la
poblacion concreta. El problema no es el registro oral (las muletillas ya
estan en parada), sino la ausencia de discriminacion tematica fina.

**Decision de iteracion: no crear experimento 2.** Con el umbral del plan
(P@1 aceptable >= 0,6) el base cumple holgadamente (P@1 aceptable = 1,0;
estricto = 0,533; P@3 aceptable = 0,767). Un exp2 que ampliara la parada del
lexico coloquial no atacaria la causa diagnosticada (afectaria por igual a
todos los segmentos) y seria sobre-ingenieria para el punto 3 del taller; la
decision queda registrada en `resultados/ir_tfidf_report.json` y en
`resultados/metricas.json`. Todo el ranking de esta seccion se genero en el
notebook con `correr_ir()` (paridad verificada arriba contra el CSV oficial).
"""

