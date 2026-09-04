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

if "RUTA_RAIZ" not in globals():  # respaldo standalone (ver sec00_setup)
    from pathlib import Path as _Path

    _candidato = _Path.cwd().resolve()
    while _candidato != _candidato.parent:
        if (_candidato / "data" / "corpus" / "segmentos_libros.jsonl").exists():
            break
        _candidato = _candidato.parent
    RUTA_RAIZ = _candidato
    RUTA_PREPROCESADO = RUTA_RAIZ / "data" / "corpus" / "preprocesado"
    RUTA_CONFIG = RUTA_RAIZ / "config"
    RUTA_RESULTADOS = RUTA_RAIZ / "resultados"

# Helpers de config: existen en sec00 (notebook); se definen solo si la
# seccion corre standalone (misma convencion de autocontencion).
if "cargar_correcciones" not in globals():

    def cargar_correcciones() -> dict[str, str]:
        """Carga config/lemas_correcciones.json (respaldo standalone)."""
        datos = json.loads((RUTA_CONFIG / "lemas_correcciones.json").read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in datos["correcciones"].items()}

    def cargar_muletillas() -> set[str]:
        """Carga config/stopwords_habla.txt (respaldo standalone)."""
        ruta = RUTA_CONFIG / "stopwords_habla.txt"
        return {
            linea.strip()
            for linea in ruta.read_text(encoding="utf-8").splitlines()
            if linea.strip() and not linea.startswith("#")
        }

    def cargar_ruido() -> set[str]:
        """Carga el ruido tipografico de config/tfidf_config.json (standalone)."""
        datos = json.loads((RUTA_CONFIG / "tfidf_config.json").read_text(encoding="utf-8"))
        return {str(t) for t in datos["ruido_tipografico_libros_excluido_en_eda"]}

    def cargar_lemas(corpus: str) -> list[str]:
        """Lee el archivo de lemas de un corpus (respaldo standalone)."""
        ruta = RUTA_PREPROCESADO / f"{corpus}_lemas.txt"
        return [linea.rstrip("\n") for linea in ruta.open(encoding="utf-8")]

    def cargar_metadatos(corpus: str) -> list[dict]:
        """Lee los metadatos de un corpus (respaldo standalone)."""
        ruta = RUTA_PREPROCESADO / f"{corpus}_docs.json"
        return json.loads(ruta.read_text(encoding="utf-8"))["docs"]


def cargar_config_vectorizador() -> dict:
    """Carga los parametros del vectorizador de config/tfidf_config.json.

    Returns:
        Dict con los kwargs de TfidfVectorizer congelados en el EDA (Fase 5).
    """
    ruta = RUTA_CONFIG / "tfidf_config.json"
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    config = dict(datos.get("vectorizador", {}))
    config["ngram_range"] = tuple(int(v) for v in config["ngram_range"])
    return config


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
    correcciones = cargar_correcciones() if variante != "C" else {}
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

# Verificacion: el ranking en memoria debe ser identico al CSV oficial
# resultados/ranking_top5.csv (mismo codigo y config; corrida determinista).
base_csv = pd.read_csv(RUTA_RESULTADOS / "ranking_top5.csv")
top1_csv = base_csv[base_csv["rango"] == 1].set_index("id_doc")["id_segmento"]
top1_mem = top1.set_index("id_doc")["id_segmento"]
n_coinciden = int((top1_csv == top1_mem).sum())
pct_csv = 100.0 * (base_csv[base_csv["rango"] == 1]["tomo"] == "TESTIMONIAL").mean()
pct_mem = 100.0 * (top1["tomo"] == "TESTIMONIAL").mean()
print(
    f"Verificacion vs ranking_top5.csv: top-1 identicos en {n_coinciden}/"
    f"{len(top1_mem)} queries | TESTIMONIAL: {pct_mem:.2f} % (memoria) = "
    f"{pct_csv:.2f} % (CSV oficial) -> reproducido: {n_coinciden == len(top1_mem)}"
)


# %%
"""Variantes A/B/C: coincidencia del top-1 con el base (desde CSVs)."""

# A: ngram (1,1); B: sin muletillas; C: sin correcciones de lemas.
# El codigo inline genera el ranking del base (arriba); las variantes se
# comparan contra sus CSVs oficiales versionados (ranking_top5_{A,B,C}.csv),
# generados por el experimento de produccion con el mismo codigo: la
# coincidencia esperada del reporte es A 54,8 / B 84,6 / C 96,6.
base_top1 = pd.read_csv(RUTA_RESULTADOS / "ranking_top5.csv")
base_top1 = base_top1[base_top1["rango"] == 1].set_index("id_doc")["id_segmento"]
esperado = {"A": 54.8, "B": 84.6, "C": 96.6}
filas_variantes = []
for nombre, pct_esperado in esperado.items():
    variante = pd.read_csv(RUTA_RESULTADOS / f"ranking_top5_{nombre}.csv")
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


# %%
"""Tabla de evaluacion etiquetada (30 pares) y metricas proxy recalculadas.

Las metricas P@1 se recalculan aqui desde evaluacion_muestra_etiquetada.csv
(columna 'etiqueta' del top-1). Para P@3 se necesitan las etiquetas de r2 y
r3, que el CSV oficial no lleva (solo etiqueta el top-1): se toman del campo
'detalle_por_fila' de evaluacion_metricas.json (etiquetas auxiliares del
mismo etiquetado) y se verifican los resultados contra el resumen del JSON.
"""

evaluacion = pd.read_csv(RUTA_RESULTADOS / "evaluacion_muestra_etiquetada.csv")
print(f"Muestra etiquetada: {len(evaluacion)} pares query -> segmento top-1\n")

tabla_eval = evaluacion[
    ["id_doc", "r1_tomo", "r1_score", "etiqueta"]
].copy()
tabla_eval["id_doc"] = tabla_eval["id_doc"].str[:20] + ".."
tabla_eval["r1_score"] = tabla_eval["r1_score"].round(4)
print("Las 30 filas etiquetadas (id truncado; r1 = segmento top-1):")
print(tabla_eval.to_string(index=False))

conteo = evaluacion["etiqueta"].value_counts()
print("\nConteo de etiquetas:", dict(conteo))

metricas_eval = json.loads(
    (RUTA_RESULTADOS / "evaluacion_metricas.json").read_text(encoding="utf-8")
)
n = len(evaluacion)
conteos = {
    etiqueta: int((evaluacion["etiqueta"] == etiqueta).sum())
    for etiqueta in ("relacionado", "parcial", "no")
}
p1_estricto = conteos["relacionado"] / n
p1_aceptable = (conteos["relacionado"] + conteos["parcial"]) / n

# P@3 proxy: por query, >=1 de los rangos 1-3 con la etiqueta (r2/r3 del
# detalle_por_fila del JSON oficial, alineado por id_doc).
detalle = {d["id_doc"]: d for d in metricas_eval["detalle_por_fila"]}
n_ok3_estricto = 0
n_ok3_aceptable = 0
for _, fila in evaluacion.iterrows():
    fila_detalle = detalle[fila["id_doc"]]
    etiquetas_r = [
        fila["etiqueta"],
        fila_detalle["etiqueta_r2"],
        fila_detalle["etiqueta_r3"],
    ]
    if "relacionado" in etiquetas_r:
        n_ok3_estricto += 1
    if any(e != "no" for e in etiquetas_r):
        n_ok3_aceptable += 1
p3_estricto = n_ok3_estricto / n
p3_aceptable = n_ok3_aceptable / n

print("\nMetricas proxy recalculadas en el notebook (verificacion contra")
print("resultados/evaluacion_metricas.json):")
print(f"  P@1 estricto (solo 'relacionado')         : {p1_estricto:.4f} "
      f"(json {metricas_eval['p1_proxy']['estricto_solo_relacionado']:.4f})")
print(f"  P@1 aceptable ('relacionado' o 'parcial') : {p1_aceptable:.4f} "
      f"(json {metricas_eval['p1_proxy']['aceptable_relacionado_o_parcial']:.4f})")
print(f"  P@3 aceptable (>=1 'relacionado')         : {p3_estricto:.4f} "
      f"(json {metricas_eval['p3_proxy']['aceptable_al_menos_un_relacionado']:.4f})")
print(f"  P@3 aceptable (>=1 no-'no')               : {p3_aceptable:.4f} "
      f"(json {metricas_eval['p3_proxy']['aceptable_relacionado_o_parcial']:.4f})")
recalculado_ok = (
    abs(p1_estricto - metricas_eval["p1_proxy"]["estricto_solo_relacionado"]) < 1e-4
    and abs(p1_aceptable - metricas_eval["p1_proxy"]["aceptable_relacionado_o_parcial"]) < 1e-4
    and abs(p3_estricto - metricas_eval["p3_proxy"]["aceptable_al_menos_un_relacionado"]) < 1e-4
)
print(f"  Recalculado identico al JSON oficial: {recalculado_ok}")
print(
    f"  Top-1 'parcial' con match 'relacionado' en r2-r3: "
    f"{metricas_eval['errores_top1']['n_top1_parcial_con_match_relacionado_r2_r3']}/30"
)
sesgo = metricas_eval["sesgo_testimonial"]
print(f"  En la muestra, top-1 TESTIMONIAL: {sesgo['n_top1_testimonial_en_muestra']} "
      f"({dict(sesgo['distribucion_top1_testimonial'])})")


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

