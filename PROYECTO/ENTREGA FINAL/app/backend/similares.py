"""similares.py
Fecha: 2026-10-31
Descripcion: Indice de similitud de texto del corpus (P5 de la Entrega
    Final: "sentencias similares"). Construye una sola vez una matriz TF-IDF
    dispersa sobre texto_limpio del corpus congelado y la cachea en disco
    (app/backend/cache/), de modo que el arranque no la recalcule. La carga y
    la construccion perezosa ocurren bajo lock; las consultas son un producto
    matriz-vector disperso seguido de un top-k. La similitud es coseno porque
    TfidfVectorizer normaliza cada fila a norma L2. Nunca se devuelve la
    sentencia consultada.
Estado del arte aplicado: TF-IDF sublinear con min_df y truncado de texto
    para acotar memoria y disco, matriz CSR en float32, cache versionada por
    firma del corpus, top-k con argsort estable y metadatos M5 (PageRank).
"""

import threading
import time

import joblib
import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

import core
from core import nucleo

# ===== SECCION 1: configuracion =====
CACHE_DIR = core.BACKEND_DIR / "cache"
RUTA_MATRIZ = CACHE_DIR / "tfidf_corpus.npz"
RUTA_ARTEFACTO = CACHE_DIR / "tfidf_vectorizer.joblib"
RUTA_CORPUS = nucleo.ROOT / nucleo.CONFIG["ruta_corpus"]
RUTA_C3 = core.INTEGRACION_C3_DIR / "corpus_c3.parquet"

VERSION_INDICE = 2
MAX_CARACTERES = 30000
MAX_FEATURES = 120000
MIN_DF = 3
K_DEFECTO = 5
K_MAXIMO = 20
AVISO_SIMILITUD = (
    "Similitud TF-IDF (coseno) sobre el texto limpio del corpus 1992-2026. "
    "Mide parecido lexical, no equivalencia doctrinal ni precedente."
)

_LOCK = threading.Lock()
_INDICE: dict | None = None


# ===== SECCION 2: construccion y cache en disco =====
def _firma_corpus() -> str:
    """Identifica la version del corpus que respalda el indice.

    Returns:
        Marca de modificacion del dataset de sentencias.

    Raises:
        FileNotFoundError: Si el corpus no existe.
    """
    firma_c3 = RUTA_C3.stat().st_mtime_ns if RUTA_C3.exists() else 0
    return f"{RUTA_CORPUS.stat().st_mtime_ns}-{firma_c3}"


def _construir(firma: str) -> dict:
    """Ajusta el TF-IDF y arma los metadatos del indice.

    Args:
        firma: Marca del corpus que se guarda junto al artefacto.

    Returns:
        Dict con matriz CSR, vectorizer, ids, metadatos y firma.

    Raises:
        FileNotFoundError: Si falta el corpus o la red de citas.
    """
    tabla = ds.dataset(RUTA_CORPUS, format="parquet", partitioning="hive").to_table(
        columns=["id", "numero", "texto_limpio", "tipo", "ano"]
    )
    documentos = pc.utf8_slice_codeunits(
        tabla["texto_limpio"], 0, MAX_CARACTERES
    ).to_pylist()
    ids = [str(valor) for valor in tabla["id"].to_pylist()]
    numeros = tabla["numero"].to_pylist()
    tipos = [str(valor) for valor in tabla["tipo"].to_pylist()]
    anos = [int(valor) for valor in tabla["ano"].to_pylist()]
    del tabla

    if RUTA_C3.exists():
        c3 = pq.read_table(
            RUTA_C3, columns=["id", "numero", "texto_limpio", "tipo", "ano"]
        ).to_pandas()
        documentos.extend(
            str(valor)[:MAX_CARACTERES] for valor in c3["texto_limpio"]
        )
        ids.extend(str(valor) for valor in c3["id"])
        numeros.extend(int(valor) for valor in c3["numero"])
        tipos.extend(str(valor) for valor in c3["tipo"])
        anos.extend(int(valor) for valor in c3["ano"])

    vectorizer = TfidfVectorizer(
        min_df=MIN_DF,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
        token_pattern=r"(?u)\b\w+\b",
    )
    matriz = vectorizer.fit_transform(documentos).tocsr()
    del documentos

    metadatos = core.metadatos_sentencias()
    canones = [
        f"{tipo}-{int(numero)}-{int(ano)}"
        for tipo, numero, ano in zip(tipos, numeros, anos, strict=True)
    ]
    return {
        "version": VERSION_INDICE,
        "firma": firma,
        "vectorizer": vectorizer,
        "matriz": matriz,
        "ids": ids,
        "canones": canones,
        "anos": anos,
        "tipos": tipos,
        "pageranks": [
            metadatos[id_]["pagerank"] if id_ in metadatos else None
            for id_ in ids
        ],
    }


def _guardar(artefacto: dict) -> None:
    """Escribe la matriz y el vectorizador en la cache del backend.

    Args:
        artefacto: Resultado de _construir (la matriz se serializa aparte).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    matriz = artefacto["matriz"]
    sp.save_npz(RUTA_MATRIZ, matriz, compressed=True)
    joblib.dump(
        {clave: valor for clave, valor in artefacto.items() if clave != "matriz"},
        RUTA_ARTEFACTO,
        compress=3,
    )


def _cache_valida(firma: str) -> dict | None:
    """Carga el artefacto si existe y corresponde al corpus actual.

    Args:
        firma: Marca de modificacion del corpus.

    Returns:
        El artefacto ya cargado (sin la matriz) o None si hay que reconstruir.
    """
    if not RUTA_MATRIZ.exists() or not RUTA_ARTEFACTO.exists():
        return None
    try:
        artefacto = joblib.load(RUTA_ARTEFACTO)
    except (OSError, ValueError):
        return None
    if not isinstance(artefacto, dict):
        return None
    if (
        artefacto.get("version") == VERSION_INDICE
        and artefacto.get("firma") == firma
    ):
        return artefacto
    return None


# ===== SECCION 3: carga perezosa bajo lock =====
def indice() -> dict:
    """Devuelve el indice en memoria, construyendolo la primera vez.

    Returns:
        Dict con matriz CSR, pos_por_id y metadatos por fila.

    Raises:
        LookupError: Si faltan los artefactos del corpus.
        RuntimeError: Si la construccion o la carga fallan.
    """
    global _INDICE
    if _INDICE is not None:
        return _INDICE
    with _LOCK:
        if _INDICE is not None:
            return _INDICE
        try:
            firma = _firma_corpus()
        except OSError as exc:
            raise LookupError(
                f"No existe el corpus para el indice de similitud: {exc}."
            ) from exc
        artefacto = _cache_valida(firma)
        if artefacto is not None:
            matriz = sp.load_npz(RUTA_MATRIZ).tocsr()
        else:
            try:
                artefacto = _construir(firma)
                _guardar(artefacto)
            except (OSError, ValueError, MemoryError) as exc:
                raise RuntimeError(
                    "No se pudo construir el indice TF-IDF de similitud: "
                    f"{exc}."
                ) from exc
            matriz = artefacto.pop("matriz")
        artefacto["matriz"] = matriz
        artefacto["pos_por_id"] = {
            id_: pos for pos, id_ in enumerate(artefacto["ids"])
        }
        _INDICE = artefacto
    return _INDICE


def _limitar_k(k: int) -> int:
    """Acota la profundidad de la consulta al rango [1, K_MAXIMO]."""
    return max(1, min(int(k), K_MAXIMO))


# ===== SECCION 4: consulta =====
def buscar(id_sentencia: str, k: int = K_DEFECTO) -> dict:
    """Devuelve las k sentencias mas parecidas a una del corpus.

    Args:
        id_sentencia: Id del corpus de la sentencia consultada.
        k: Numero de vecinos a devolver (1-20).

    Returns:
        Dict con id, canon, similares (id, canon, ano, tipo, pagerank,
        score), k, ms y aviso de procedencia.

    Raises:
        LookupError: Si el id no existe en el corpus indexado.
        RuntimeError: Si el indice no se puede construir o cargar.
    """
    inicio = time.perf_counter()
    profundidad = _limitar_k(k)
    datos = indice()
    posicion = datos["pos_por_id"].get(str(id_sentencia))
    if posicion is None:
        raise LookupError(
            f"El id {id_sentencia} no existe en el corpus indexado (1992-2026)."
        )
    fila = datos["matriz"].getrow(posicion)
    puntajes = np.asarray((datos["matriz"] @ fila.T).toarray()).ravel()
    # La consulta nunca se devuelve a si misma (self-loop).
    puntajes[posicion] = -1.0
    orden = np.argsort(-puntajes, kind="stable")
    similares = []
    for pos in orden:
        if len(similares) >= profundidad:
            break
        score = float(puntajes[pos])
        if score <= 0.0:
            continue
        similares.append(
            {
                "id": datos["ids"][pos],
                "canon": datos["canones"][pos],
                "ano": datos["anos"][pos],
                "tipo": datos["tipos"][pos],
                "pagerank": datos["pageranks"][pos],
                "score": round(score, 4),
            }
        )
    return {
        "id": str(id_sentencia),
        "canon": datos["canones"][posicion],
        "similares": similares,
        "k": profundidad,
        "ms": int((time.perf_counter() - inicio) * 1000),
        "aviso": AVISO_SIMILITUD,
    }
