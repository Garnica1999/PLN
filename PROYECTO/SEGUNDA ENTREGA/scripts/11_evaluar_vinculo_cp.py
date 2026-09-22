"""11_evaluar_vinculo_cp.py
Fecha: 2026-09-08
Descripcion: Benchmark M7 (Fase K) del vinculo sentencia -> articulos de la
    Constitucion Politica sobre la muestra temporal de ADR-5 (1 sentencia por
    anio, 1992-2021, semilla 42). Compara tres estrategias de retrieval frente
    a los weak labels de M2 (data/entidades_sentencias.parquet, arts_cp):
    (a) cita-exacta por regex M2 (circular por construccion: mide recall de
    extraccion y se reporta por separado), (b) BM25 clasico con el texto
    limpio completo como query (ADR-3/ADR-6) y (c) embeddings E5-multilingual
    en GPU con chunking de 512 tokens y mean-pooling (ADR-3, ADR-6: texto
    crudo para BPE). Tambien ejecuta M0 (validacion de entrada) con la
    heuristica de scripts/lib/validacion_entrada.py sobre la misma muestra de
    30 positivas y los 22 discursos de posesion de open_corpus_co_es
    (corpus de TALLER-1) como negativos.
Estado del arte aplicado: BM25 por pasajes + re-rank denso y particiones
    cronologicas (DoSSIER/COLIEE 2021; Fink 2023); E5 zero-shot para detectar
    deriva temporal; protocolo temporal de ADR-5 con P@k, R@k, F1@k y MRR.
"""

import importlib.util
import json
import random
import re
import sys
import time
import traceback
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from open_corpus_co_es import load_corpus
from rank_bm25 import BM25Okapi

CONFIG = {
    "semilla": 42,
    "k": 5,
    "ano_min": 1992,
    "ano_max": 2021,
    "tamano_chunk": 512,
    "solape_chunk": 64,
    "reserva_prefijo_tokens": 4,
    "prefijo_query": "query: ",
    "prefijo_pasaje": "passage: ",
    "modelo_e5": "intfloat/multilingual-e5-base",
    "batch_embedding": 16,
    "batch_embedding_oom": 8,
    "vram_minima_gb": 2.0,
    "ventana_cp": 60,
    "ruta_corpus": "data/corpus_sentencias.parquet",
    "ruta_entidades": "data/entidades_sentencias.parquet",
    "ruta_constitucion": "data/constitucion_articulos.parquet",
    "ruta_script_m2": "scripts/03_extraccion_entidades.py",
    "ruta_benchmark": "resultados/vinculo_cp_benchmark.json",
    "ruta_por_ano": "resultados/vinculo_cp_por_ano.csv",
    "ruta_figura": "figuras/deriva_retrieval.png",
    "ruta_m0": "resultados/m0_validacion.json",
    "periodos": [
        ("1992-2001", 1992, 2001),
        ("2002-2011", 2002, 2011),
        ("2012-2021", 2012, 2021),
    ],
    "estrategias": ["cita_exacta", "bm25", "e5"],
}

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
import validacion_entrada

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

MEDIDAS = ("p_at_k", "r_at_k", "f1_at_k", "mrr")


# ===== SECCION 1: utilidades base =====
def configurar_semillas(semilla: int) -> None:
    """Fija semillas de aleatoriedad y hilos de torch.

    Args:
        semilla: Valor entero de semilla.
    """
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.set_num_threads(32)


def elegir_dispositivo() -> str:
    """Elige GPU solo si hay VRAM libre suficiente para E5.

    La maquina comparte la GPU con procesos externos (llama-server) que
    pueden ocupar toda la VRAM. Si la memoria libre es menor que
    CONFIG['vram_minima_gb'] se degrada a CPU y se documenta en el log; no se
    cambia de modelo ni de metodologia.

    Returns:
        "cuda" si hay >= 2 GiB libres y torch la ve disponible; si no, "cpu".
    """
    if not torch.cuda.is_available():
        return "cpu"
    libre, _ = torch.cuda.mem_get_info()
    minimo = int(CONFIG["vram_minima_gb"] * 1024**3)
    if libre < minimo:
        print(
            "[aviso] VRAM libre insuficiente para E5 "
            f"({libre / 1024**3:.1f} GiB < {CONFIG['vram_minima_gb']} GiB); "
            "se ejecuta E5 en CPU (GPU ocupada por procesos externos)"
        )
        return "cpu"
    return "cuda"


def cargar_muestra_temporal(corpus: pd.DataFrame) -> pd.DataFrame:
    """Selecciona 1 sentencia por anio (1992-2021) con semilla fija.

    Ordena el corpus por (anio, id) para que la eleccion sea reproducible
    independiente del orden fisico del parquet y muestrea una fila por anio.

    Args:
        corpus: DataFrame con columnas id, tipo y ano.

    Returns:
        DataFrame con 30 filas (una por anio) ordenadas por anio.
    """
    ordenado = corpus.sort_values(["ano", "id"]).reset_index(drop=True)
    rng = random.Random(CONFIG["semilla"])
    posiciones = []
    for anio in range(CONFIG["ano_min"], CONFIG["ano_max"] + 1):
        grupo = ordenado[ordenado["ano"] == anio]
        if grupo.empty:
            continue
        posiciones.append(grupo.index[rng.randrange(len(grupo))])
    return ordenado.loc[posiciones].sort_values("ano").reset_index(drop=True)


def cargar_negativos_taller1() -> list[dict]:
    """Carga los discursos de posesion usados como negativos en M0.

    TALLER-1 no conserva archivos .txt en el repositorio; el corpus de 22
    discursos presidenciales que consume su notebook proviene del paquete
    open_corpus_co_es (load_corpus('presidentes')), que ya esta descargado
    en el venv. Se usan esos mismos textos como negativos reales.

    Returns:
        Lista de dicts con nombre, anio y texto de cada discurso.
    """
    discursos = load_corpus("presidentes")
    return [
        {
            "nombre": d["nombre"],
            "anio": str(d.get("fecha_inicio", "")),
            "texto": d["text"],
        }
        for d in discursos
    ]


def importar_extraccion_m2() -> object:
    """Importa el modulo 03_extraccion_entidades.py (M2) por importlib.

    El nombre del archivo empieza por digito y no admite import estandar; se
    carga con spec para reutilizar extraer_arts_cp sin duplicar logica.

    Returns:
        Modulo M2 con la funcion extraer_arts_cp.
    """
    ruta = ROOT / CONFIG["ruta_script_m2"]
    spec = importlib.util.spec_from_file_location("extraccion_m2", ruta)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar el modulo M2: {ruta}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# ===== SECCION 2: metricas de retrieval =====
def metricas_ranking(relevantes: list[str], ranking: list[str], k: int) -> dict:
    """Calcula P@k, R@k, F1@k y MRR de un ranking frente a los relevantes.

    Args:
        relevantes: Canones de articulos relevantes (weak labels).
        ranking: Canones ordenados por el retrieval (mejor primero).
        k: Profundidad para P@k, R@k y F1@k.

    Returns:
        Dict con p_at_k, r_at_k, f1_at_k y mrr.
    """
    rel = set(relevantes)
    top = ranking[:k]
    recuperados = sum(1 for x in top if x in rel)
    p = recuperados / k
    r = recuperados / len(rel) if rel else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    mrr = 0.0
    for pos, canon in enumerate(ranking, start=1):
        if canon in rel:
            mrr = 1.0 / pos
            break
    return {"p_at_k": p, "r_at_k": r, "f1_at_k": f1, "mrr": mrr}


def promediar_metricas(metricas: list[dict]) -> dict:
    """Promedia las cuatro medidas sobre varios casos de retrieval.

    Args:
        metricas: Lista de dicts de metricas_ranking.

    Returns:
        Dict con las medias de p_at_k, r_at_k, f1_at_k y mrr.
    """
    return {
        medida: float(np.mean([m[medida] for m in metricas])) for medida in MEDIDAS
    }


def agregar_por_periodo(por_caso: pd.DataFrame) -> dict:
    """Agrupa metricas por periodo cronologico (ADR-5).

    Args:
        por_caso: DataFrame con columnas ano, estrategia y las cuatro medidas.

    Returns:
        Dict anidado {periodo: {estrategia: {medidas medias}}}.
    """
    resultado = {}
    por_caso = por_caso.copy()
    por_caso["periodo"] = por_caso["ano"].apply(asignar_periodo)
    for nombre_periodo, _, _ in CONFIG["periodos"]:
        grupo = por_caso[por_caso["periodo"] == nombre_periodo]
        resultado[nombre_periodo] = {}
        for estrategia in CONFIG["estrategias"]:
            filas = grupo[grupo["estrategia"] == estrategia]
            resultado[nombre_periodo][estrategia] = {
                medida: float(filas[medida].mean()) for medida in MEDIDAS
            }
    return resultado


def asignar_periodo(anio: int) -> str:
    """Asigna un anio a su periodo cronologico de CONFIG.

    Args:
        anio: Anio de la sentencia.

    Returns:
        Nombre del periodo (p. ej. "1992-2001") que contiene el anio.
    """
    for nombre, inicio, fin in CONFIG["periodos"]:
        if inicio <= anio <= fin:
            return nombre
    return "desconocido"


# ===== SECCION 3: preparacion de documentos de la Constitucion =====
def preparar_documentos_constitucion() -> pd.DataFrame:
    """Carga y normaliza los 497 articulos de la Constitucion.

    Returns:
        DataFrame con columnas canon (numero del articulo), texto y tokens
        (tokenizacion sin tildes para BM25).
    """
    articulos = pd.read_parquet(CONFIG["ruta_constitucion"])
    marco = pd.DataFrame(
        {
            "canon": articulos["numero"].astype(str),
            "texto": articulos["texto"].fillna("").astype(str),
        }
    )
    marco["tokens"] = marco["texto"].apply(tokenizar)
    return marco


def normalizar_sin_tildes(texto: str) -> str:
    """Convierte un texto a minusculas y sin diacriticos.

    Args:
        texto: Texto original.

    Returns:
        Texto normalizado.
    """
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFD", texto) if not unicodedata.combining(c)
    )
    return sin_tildes.lower()


def tokenizar(texto: str) -> list[str]:
    """Tokeniza un texto en minusculas y sin tildes para BM25.

    Args:
        texto: Texto a tokenizar.

    Returns:
        Lista de tokens alfanumericos.
    """
    return re.findall(r"[a-z0-9]+", normalizar_sin_tildes(texto))


def metricas_por_caso(df_casos: pd.DataFrame) -> pd.DataFrame:
    """Evalua las estrategias sobre cada caso y devuelve filas por estrategia.

    Args:
        df_casos: DataFrame de la muestra con arts (labels) y rankings por
            estrategia anidados en la columna rankings (dict).

    Returns:
        DataFrame largo con una fila por (anio, estrategia) y sus medidas.
    """
    filas = []
    for _, caso in df_casos.iterrows():
        relevantes = caso["arts"]
        for estrategia in CONFIG["estrategias"]:
            metricas = metricas_ranking(relevantes, caso["rankings"][estrategia], CONFIG["k"])
            filas.append(
                {
                    "id": caso["id"],
                    "ano": caso["ano"],
                    "estrategia": estrategia,
                    "n_relevantes": len(relevantes),
                    **metricas,
                }
            )
    return pd.DataFrame(filas)


# ===== SECCION 4: estrategia (a) cita-exacta (regex M2) =====
def recuperar_cita_exacta(extraer_arts: object, texto: str) -> list[str]:
    """Re-extrae los articulos de la CP citados con el regex de M2.

    Args:
        extraer_arts: Modulo M2 con la funcion extraer_arts_cp.
        texto: Texto crudo de la sentencia.

    Returns:
        Lista de canones citados en orden de aparicion (ranking de la
        estrategia cita-exacta).
    """
    return extraer_arts.extraer_arts_cp(texto, CONFIG["ventana_cp"])


# ===== SECCION 5: estrategia (b) BM25 =====
def construir_indice_bm25(documentos: pd.DataFrame) -> BM25Okapi:
    """Construye el indice BM25 sobre los articulos de la Constitucion.

    Args:
        documentos: DataFrame con la columna tokens.

    Returns:
        Indice BM25Okapi listo para consultas.
    """
    return BM25Okapi(documentos["tokens"].tolist())


def recuperar_bm25(indice: BM25Okapi, canones: list[str], texto_limpio: str) -> list[str]:
    """Rankea los articulos con BM25 dado el texto limpio de la sentencia.

    Args:
        indice: Indice BM25Okapi de la Constitucion.
        canones: Canones alineados con las filas del indice.
        texto_limpio: Texto limpio (modelo clasico, ADR-6) de la sentencia.

    Returns:
        Canones ordenados por score descendente.
    """
    puntajes = indice.get_scores(tokenizar(texto_limpio))
    orden = np.argsort(puntajes)[::-1]
    return [canones[i] for i in orden]


# ===== SECCION 6: estrategia (c) E5-multilingual con chunk-mean =====
def chunkear_texto(texto: str, tokenizer: object) -> list[str]:
    """Divide un texto en chunks de <=512 tokens con solape (ADR-3).

    Reserva CONFIG['reserva_prefijo_tokens'] tokens para el prefijo
    "query: "/"passage: " que se anade al codificar. Usa los ids del
    tokenizer para que el corte respete el limite real del modelo.

    Args:
        texto: Texto crudo largo (sentencia o articulo).
        tokenizer: Tokenizer del modelo E5.

    Returns:
        Lista de chunks decodificados (sin el prefijo).
    """
    ids = tokenizer.encode(texto, add_special_tokens=False)
    maximo = CONFIG["tamano_chunk"] - CONFIG["reserva_prefijo_tokens"]
    solape = CONFIG["solape_chunk"]
    chunks: list[str] = []
    inicio = 0
    total = len(ids)
    while inicio < total:
        fin = min(inicio + maximo, total)
        chunks.append(tokenizer.decode(ids[inicio:fin], skip_special_tokens=True))
        if fin == total:
            break
        inicio = fin - solape
    return chunks


def _codificar_con_fallback(modelo: object, textos: list[str]) -> np.ndarray:
    """Codifica textos con E5 reintentando con batch menor ante OOM.

    Args:
        modelo: SentenceTransformer cargado en el dispositivo elegido.
        textos: Textos con prefijo ya incluido.

    Returns:
        Matriz de embeddings normalizados (n, dim).
    """
    try:
        return np.asarray(
            modelo.encode(
                textos,
                batch_size=CONFIG["batch_embedding"],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        )
    except RuntimeError:
        print(
            "[aviso] OOM en E5: se re-codifica con batch "
            f"{CONFIG['batch_embedding_oom']} (documentado)"
        )
        return np.asarray(
            modelo.encode(
                textos,
                batch_size=CONFIG["batch_embedding_oom"],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        )


def media_por_grupos(matriz: np.ndarray, longitudes: list[int]) -> np.ndarray:
    """Agrega por media (chunk-mean de ADR-3) y renormaliza cada grupo.

    Cada documento se codifico por chunks; los chunks de un mismo documento
    son consecutivos en la matriz y su numero viene en longitudes.

    Args:
        matriz: Embeddings planos (sum(longitudes), dim).
        longitudes: Numero de chunks de cada documento, en orden.

    Returns:
        Matriz (len(longitudes), dim) con un vector unitario por documento.
    """
    vectores = []
    inicio = 0
    for n_chunks in longitudes:
        bloque = matriz[inicio : inicio + n_chunks]
        inicio += n_chunks
        media = bloque.mean(axis=0)
        norma = float(np.linalg.norm(media))
        vectores.append(media / norma if norma > 0 else media)
    return np.vstack(vectores)


def recuperar_e5(
    vector_query: np.ndarray, matriz_pasajes: np.ndarray, canones: list[str]
) -> list[str]:
    """Rankea articulos por similitud coseno con el embedding de la sentencia.

    Args:
        vector_query: Embedding unitario (chunk-mean) de la sentencia.
        matriz_pasajes: Matriz (497, dim) de embeddings unitarios de articulos.
        canones: Canones alineados a las filas de matriz_pasajes.

    Returns:
        Canones ordenados por similitud descendente.
    """
    similitudes = matriz_pasajes @ vector_query
    orden = np.argsort(similitudes)[::-1]
    return [canones[i] for i in orden]


# ===== SECCION 7: figura de deriva temporal =====
def graficar_deriva(por_caso: pd.DataFrame) -> None:
    """Genera figuras/deriva_retrieval.png con F1@k y MRR por anio.

    Cada anio aporta una sentencia (diseno ADR-5), por lo que las series son
    ruidosas; se trazan las tres estrategias con marcas y lineas finas.

    Args:
        por_caso: DataFrame largo con ano, estrategia y medidas.
    """
    etiquetas = {
        "cita_exacta": "cita-exacta (regex M2, circular)",
        "bm25": "BM25 (texto limpio)",
        "e5": "E5-multilingual (chunk-mean)",
    }
    colores = {"cita_exacta": "#7f7f7f", "bm25": "#1f77b4", "e5": "#d62728"}
    figura, ejes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for medida, eje, titulo in (
        ("f1_at_k", ejes[0], "F1@5 sentencia->articulos CP por anio (1 caso/anio)"),
        ("mrr", ejes[1], "MRR por anio"),
    ):
        for estrategia in CONFIG["estrategias"]:
            serie = (
                por_caso[por_caso["estrategia"] == estrategia]
                .sort_values("ano")
                .reset_index(drop=True)
            )
            eje.plot(
                serie["ano"],
                serie[medida],
                marker="o",
                markersize=4,
                linewidth=1.1,
                color=colores[estrategia],
                label=etiquetas[estrategia],
            )
        eje.set_ylabel(medida.upper())
        eje.set_title(titulo, fontsize=10)
        eje.grid(True, alpha=0.3)
        eje.legend(fontsize=8, loc="upper right")
    ejes[1].set_xlabel("Anio de la sentencia")
    ejes[1].set_xticks(range(CONFIG["ano_min"], CONFIG["ano_max"] + 1, 3))
    figura.tight_layout()
    Path(CONFIG["ruta_figura"]).parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(CONFIG["ruta_figura"], dpi=150, bbox_inches="tight")
    plt.close(figura)


# ===== SECCION 8: M0 validacion de entrada =====
def evaluar_m0(df_muestra: pd.DataFrame, corpus: pd.DataFrame) -> dict:
    """Ejecuta la validacion M0 sobre 30 positivas y los discursos negativos.

    Args:
        df_muestra: Muestra temporal de 30 sentencias.
        corpus: Corpus completo (para localizar textos por id).

    Returns:
        Dict con matriz de confusion, accuracy y detalle por documento.
    """
    ids_muestra = set(df_muestra["id"].astype(str))
    positivas = corpus[corpus["id"].astype(str).isin(ids_muestra)]
    detalle_positivas = []
    for _, fila in positivas.iterrows():
        reglas = validacion_entrada.evaluar_reglas(fila["texto"])
        detalle_positivas.append(
            {
                "id": str(fila["id"]),
                "tipo": str(fila["tipo"]),
                "ano": int(fila["ano"]),
                "reglas_cumplidas": reglas["reglas_cumplidas"],
                "es_sentencia": reglas["es_sentencia"],
            }
        )
    negativas = cargar_negativos_taller1()
    detalle_negativas = []
    for discurso in negativas:
        reglas = validacion_entrada.evaluar_reglas(discurso["texto"])
        detalle_negativas.append(
            {
                "nombre": discurso["nombre"],
                "anio": discurso["anio"],
                "reglas_cumplidas": reglas["reglas_cumplidas"],
                "es_sentencia": reglas["es_sentencia"],
            }
        )
    tp = sum(1 for d in detalle_positivas if d["es_sentencia"])
    fn = len(detalle_positivas) - tp
    fp = sum(1 for d in detalle_negativas if d["es_sentencia"])
    tn = len(detalle_negativas) - fp
    total = tp + tn + fp + fn
    return {
        "descripcion": "M0: validacion de entrada (heuristica anexo 25)",
        "config": {
            "positivas": len(detalle_positivas),
            "negativas": len(detalle_negativas),
            "semilla": CONFIG["semilla"],
            "fuente_negativos": (
                "open_corpus_co_es load_corpus('presidentes'): 22 discursos de "
                "posesion (corpus consumido por TALLER-1, sin .txt en el repo)"
            ),
        },
        "matriz": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": (tp + tn) / total if total else 0.0,
        "positivas_detalle": detalle_positivas,
        "negativas_detalle": detalle_negativas,
    }


def guardar_json(ruta: str, contenido: dict) -> None:
    """Guarda un dict como JSON UTF-8 con indentacion.

    Args:
        ruta: Ruta del archivo de salida.
        contenido: Dict serializable.
    """
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with Path(ruta).open("w", encoding="utf-8") as manejador:
        json.dump(contenido, manejador, ensure_ascii=False, indent=2)


# ===== SECCION 9: orquestacion principal =====
def main() -> None:
    """Orquesta M0 y el benchmark M7 de las tres estrategias de retrieval."""
    inicio_total = time.time()
    configurar_semillas(CONFIG["semilla"])
    dispositivo = elegir_dispositivo()
    print(
        f"GPU disponible: {torch.cuda.is_available()} | "
        f"device E5: {dispositivo}"
    )

    corpus = pd.read_parquet(CONFIG["ruta_corpus"], columns=["id", "tipo", "ano", "texto"])
    corpus["ano"] = corpus["ano"].astype(int)
    muestra = cargar_muestra_temporal(corpus)
    entidades = pd.read_parquet(CONFIG["ruta_entidades"], columns=["id", "arts_cp"])
    entidades["id"] = entidades["id"].astype(str)

    # --- M0: validacion de entrada ---
    resultado_m0 = evaluar_m0(muestra, corpus)
    guardar_json(CONFIG["ruta_m0"], resultado_m0)
    print(
        f"M0: accuracy={resultado_m0['accuracy']:.3f} "
        f"FP={resultado_m0['matriz']['fp']} FN={resultado_m0['matriz']['fn']} "
        f"({resultado_m0['config']['positivas']}+ pos, "
        f"{resultado_m0['config']['negativas']} neg)"
    )

    # --- documentacion de la Constitucion ---
    articulos = preparar_documentos_constitucion()
    canones = articulos["canon"].tolist()
    canon_valido = set(canones)

    # --- weak labels (M2) por sentencia de la muestra ---
    casos = []
    ids_muestra = set(muestra["id"].astype(str))
    entidades_muestra = entidades[entidades["id"].isin(ids_muestra)]
    for _, fila in muestra.iterrows():
        sid = str(fila["id"])
        etiquetas = entidades_muestra.loc[entidades_muestra["id"] == sid, "arts_cp"]
        arts = json.loads(etiquetas.iloc[0]) if not etiquetas.empty else []
        casos.append(
            {
                "id": sid,
                "tipo": str(fila["tipo"]),
                "ano": int(fila["ano"]),
                "texto": fila["texto"],
                "arts": [a for a in arts if a in canon_valido],
            }
        )
    casos = [c for c in casos if c["arts"]]
    print(
        f"Muestra temporal: {len(muestra)} sentencias (1/anio), "
        f"{len(casos)} con weak labels no vacios"
    )

    # --- estrategia (a): cita-exacta con regex M2 ---
    modulo_m2 = importar_extraccion_m2()
    for caso in casos:
        caso["rankings"] = {}
        caso["rankings"]["cita_exacta"] = recuperar_cita_exacta(modulo_m2, caso["texto"])

    # --- estrategia (b): BM25 sobre texto limpio ---
    limpio = pd.read_parquet(CONFIG["ruta_corpus"], columns=["id", "texto_limpio"])
    limpio["id"] = limpio["id"].astype(str)
    textos_limpios = dict(zip(limpio["id"], limpio["texto_limpio"], strict=True))
    indice_bm25 = construir_indice_bm25(articulos)
    for caso in casos:
        caso["rankings"]["bm25"] = recuperar_bm25(
            indice_bm25, canones, str(textos_limpios.get(caso["id"], ""))
        )

    # --- estrategia (c): E5-multilingual chunk-mean en GPU ---
    # Import tardio (depende de GPU/red).
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer import modules as st_models

    # Pooling explicito mean-tokens: E5 usa media de todos los tokens, no CLS.
    backbone = st_models.Transformer(
        CONFIG["modelo_e5"], max_seq_length=CONFIG["tamano_chunk"]
    )
    pooling = st_models.Pooling(
        backbone.get_embedding_dimension(), pooling_mode="mean"
    )
    modelo_e5 = SentenceTransformer(modules=[backbone, pooling], device=dispositivo)
    tokenizer = modelo_e5.tokenizer

    # Pasajes: articulos de la Constitucion, una vez (chunk-mean sobre crudo).
    grupos_pasajes = [
        chunkear_texto(t, tokenizer) or [""] for t in articulos["texto"].tolist()
    ]
    longitudes_pasajes = [len(g) for g in grupos_pasajes]
    textos_pasajes = [
        CONFIG["prefijo_pasaje"] + chunk
        for grupo in grupos_pasajes
        for chunk in grupo
    ]
    inicio_e5 = time.time()
    emb_pasajes = _codificar_con_fallback(modelo_e5, textos_pasajes)
    matriz_pasajes = media_por_grupos(emb_pasajes, longitudes_pasajes)
    print(
        f"E5: {len(textos_pasajes)} chunks de articulos codificados en "
        f"{time.time() - inicio_e5:.1f} s"
    )

    # Queries: sentencias de la muestra por chunks de 512 con mean-pooling.
    grupos_query = [chunkear_texto(c["texto"], tokenizer) or [""] for c in casos]
    longitudes_query = [len(g) for g in grupos_query]
    textos_query = [
        CONFIG["prefijo_query"] + chunk for grupo in grupos_query for chunk in grupo
    ]
    inicio_query = time.time()
    emb_query = _codificar_con_fallback(modelo_e5, textos_query)
    matriz_query = media_por_grupos(emb_query, longitudes_query)
    print(
        f"E5: {len(textos_query)} chunks de sentencias codificados en "
        f"{time.time() - inicio_query:.1f} s"
    )
    for caso, vector_query in zip(casos, matriz_query, strict=True):
        caso["rankings"]["e5"] = recuperar_e5(vector_query, matriz_pasajes, canones)

    # --- evaluacion global, por anio y por periodo ---
    marco_casos = pd.DataFrame(casos)
    metricas = metricas_por_caso(marco_casos)
    globales = {
        estrategia: promediar_metricas(
            metricas[metricas["estrategia"] == estrategia].to_dict("records")
        )
        for estrategia in CONFIG["estrategias"]
    }
    por_periodo = agregar_por_periodo(metricas)

    # recall de extraccion de la cita-exacta (circular, se reporta aparte)
    recalls = []
    for caso in casos:
        predichos = set(caso["rankings"]["cita_exacta"])
        labels = set(caso["arts"])
        recalls.append(len(predichos & labels) / len(labels) if labels else 0.0)
    recall_extraccion = float(np.mean(recalls))

    # --- guardado de resultados ---
    ganadora = max(globales, key=lambda e: globales[e]["f1_at_k"])
    benchmark = {
        "fecha": datetime.now(timezone.utc).isoformat(),
        "descripcion": (
            "Benchmark M7 vinculo sentencia->articulos CP sobre weak labels "
            "M2 (ADR-5): 1 sentencia/anio 1992-2021, semilla 42"
        ),
        "config": {
            "k": CONFIG["k"],
            "modelo_e5": CONFIG["modelo_e5"],
            "chunk": CONFIG["tamano_chunk"],
            "solape": CONFIG["solape_chunk"],
            "batch_embedding": CONFIG["batch_embedding"],
            "nota_cita_exacta": (
                "La estrategia cita-exacta es circular por construccion "
                "(mismo regex M2 que genero los weak labels): su recall de "
                "extraccion se reporta por separado."
            ),
        },
        "muestra": [
            {"id": c["id"], "tipo": c["tipo"], "ano": c["ano"], "arts_cp": c["arts"]}
            for c in casos
        ],
        "recall_extraccion_cita_exacta": recall_extraccion,
        "global": globales,
        "por_periodo": por_periodo,
        "estrategia_ganadora_f1_at_k": ganadora,
    }
    guardar_json(CONFIG["ruta_benchmark"], benchmark)

    metricas[["ano", "estrategia", "n_relevantes", *MEDIDAS]].to_csv(
        CONFIG["ruta_por_ano"], index=False, encoding="utf-8"
    )
    graficar_deriva(metricas)

    # --- resumen final ---
    print("\n=== Resumen M7 (global, k=5) ===")
    for estrategia in CONFIG["estrategias"]:
        m = globales[estrategia]
        print(
            f"{estrategia:11s} P@5={m['p_at_k']:.3f} R@5={m['r_at_k']:.3f} "
            f"F1@5={m['f1_at_k']:.3f} MRR={m['mrr']:.3f}"
        )
    print(f"recall extraccion (cita-exacta, circular): {recall_extraccion:.3f}")
    for nombre, _, _ in CONFIG["periodos"]:
        f1_b = por_periodo[nombre]["bm25"]["f1_at_k"]
        f1_e = por_periodo[nombre]["e5"]["f1_at_k"]
        print(f"periodo {nombre}: F1@5 bm25={f1_b:.3f} | e5={f1_e:.3f}")
    print(f"ganadora por F1@5: {ganadora}")
    print(
        f"tiempo total: {time.time() - inicio_total:.1f} s | "
        f"salidas: {CONFIG['ruta_benchmark']}, {CONFIG['ruta_por_ano']}, "
        f"{CONFIG['ruta_figura']}, {CONFIG['ruta_m0']}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"Error fatal: {exc}")
        sys.exit(1)
