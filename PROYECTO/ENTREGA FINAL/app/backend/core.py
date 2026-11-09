"""core.py
Fecha: 2026-10-31
Descripcion: Nucleo del backend Juris-PLN. Resuelve el sys.path hacia la
    SEGUNDA ENTREGA y expone, como funciones JSON-serializables, los
    componentes ya validados del producto: M0 (ValidadorEntrada), M2
    (ExtractorEntidades), M4 (ClasificadorArea), M7 (VinculoCP) y M5
    (ContextoJurisprudencial). Los singletons se cargan de forma perezosa y
    bajo lock: la primera peticion paga la carga y las siguientes reutilizan
    los modelos en memoria. No se duplica logica del nucleo: se importan
    lib/componentes_producto.py y el entry scripts/10_evaluar_sentencia.py.
Estado del arte aplicado: BM25 por pasajes con prior de hubs CP
    (alpha=0.75), ruta rerank opcional B2+B1 (alpha=1.5), heuristica M0
    2/3, NER regex M2 y PageRank de la red de citas M5.
"""

import importlib
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ===== SECCION 1: resolucion de rutas e import del nucleo =====
BACKEND_DIR = Path(__file__).resolve().parent
ENTREGA_FINAL_DIR = BACKEND_DIR.parents[1]
PROYECTO_DIR = ENTREGA_FINAL_DIR.parent
SEGUNDA_ENTREGA_DIR = PROYECTO_DIR / "SEGUNDA ENTREGA"
SCRIPTS_DIR = SEGUNDA_ENTREGA_DIR / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib import componentes_producto as nucleo

MODULO_ENTRADA = "10_evaluar_sentencia"
K_DEFECTO = 10
K_MAXIMO = 50
LONGITUD_MAXIMA_REFERENCIA = 80
MAX_ARTICULOS_EVIDENCIA = 5
MAX_FRAGMENTOS_EVIDENCIA = 3

# Canon de sentencia en cualquiera de sus formas: T-293/97, T-293-1997,
# "Sentencia T-293/97". El ano de dos digitos se expande en normalizar_canon.
RE_CANON = re.compile(
    r"\b(SU|C|T)\s*[-]\s*(\d{1,4})\s*(?:[/-]\s*(\d{2}|\d{4}))\b",
    re.IGNORECASE,
)
RE_ID_CORPUS = re.compile(r"^\d{1,6}$")
RE_PREFIJO_ARTICULO = re.compile(r"^(?:art[ií]culo|art\.?)\s*", re.IGNORECASE)
RE_ARTICULO_SIMPLE = re.compile(r"^\d+[A-Za-z]?$")

_LOCK = threading.Lock()
# La ruta rerank carga B1 y B2 de forma perezosa dentro de recuperar; sin
# este lock, dos primeras peticiones simultaneas pedirian los mismos pesos
# dos veces y agotarian la VRAM.
_LOCK_RERANK = threading.Lock()
_LOCK_METADATOS = threading.Lock()
_METADATOS: dict | None = None
_TEXTOS_CONSTITUCION: dict | None = None
_LOCK_CONSTITUCION = threading.Lock()
INTEGRACION_C3_DIR = SEGUNDA_ENTREGA_DIR / "data" / "integracion_c3"
_C3: dict | None = None
_LOCK_C3 = threading.Lock()


def integracion_c3() -> dict:
    """Carga la capa 2022-2026 (corpus, canones y red combinada).

    Returns:
        Dict con por_id (id -> datos), por_canon (canon -> id) y red
        (id -> pagerank y area). Si la capa no existe, devuelve vacios.
    """
    global _C3
    if _C3 is not None:
        return _C3
    with _LOCK_C3:
        if _C3 is not None:
            return _C3
        corpus_c3 = INTEGRACION_C3_DIR / "corpus_c3.parquet"
        if not corpus_c3.exists():
            _C3 = {"por_id": {}, "por_canon": {}, "red": {}}
            return _C3
        import pandas as pd

        corpus = pd.read_parquet(corpus_c3)
        red = pd.read_parquet(INTEGRACION_C3_DIR / "red_citas_1992_2026.parquet")
        por_id = {
            str(fila.id): {
                "id": str(fila.id),
                "canon": str(fila.canon),
                "tipo": str(fila.tipo),
                "ano": int(fila.ano),
                "texto": str(fila.texto),
                "texto_limpio": str(fila.texto_limpio),
            }
            for fila in corpus.itertuples(index=False)
        }
        por_canon = {datos["canon"]: id_ for id_, datos in por_id.items()}
        red_por_id = {
            str(fila.id): {
                "pagerank": float(fila.pagerank),
                "area": str(fila.area) if pd.notna(fila.area) else None,
            }
            for fila in red.itertuples(index=False)
        }
        _C3 = {"por_id": por_id, "por_canon": por_canon, "red": red_por_id}
    return _C3
_COMPONENTES: dict | None = None
_VINCULO_RERANK: object | None = None
_MODULO_ENTRADA: object | None = None


# ===== SECCION 2: singletons perezosos =====
def _modulo_entrada() -> object:
    """Importa una sola vez el entry del producto (10_evaluar_sentencia).

    Returns:
        Modulo con texto_por_id, inicializar_componentes y construir_salida.
    """
    global _MODULO_ENTRADA
    if _MODULO_ENTRADA is None:
        with _LOCK:
            if _MODULO_ENTRADA is None:
                _MODULO_ENTRADA = importlib.import_module(MODULO_ENTRADA)
    return _MODULO_ENTRADA


def componentes() -> dict:
    """Devuelve los cinco componentes del producto (singleton, ruta default).

    Returns:
        Dict con validador, extractor, clasificador, vinculo (BM25 + prior
        de hubs) y contexto.
    """
    global _COMPONENTES
    if _COMPONENTES is not None:
        return _COMPONENTES
    modulo = _modulo_entrada()
    with _LOCK:
        if _COMPONENTES is None:
            # Reutiliza exactamente el ensamblado del entry congelado
            # (usar_reranker=False: BM25 por pasajes + prior alpha=0.75).
            _COMPONENTES = modulo.inicializar_componentes(usar_reranker=False)
    return _COMPONENTES


def vinculo_rerank() -> object:
    """Devuelve el VinculoCP en modo rerank (B2+B1), cargado bajo demanda.

    VinculoCP fija la ruta de recuperacion en su constructor, por eso la ruta
    rerank necesita una instancia propia; el resto de componentes (M0, M2, M4,
    M5) se comparte con `componentes()`. Los modelos densos y el cross-encoder
    se cargan lazy dentro de VinculoCP en el primer `recuperar`.

    Returns:
        Instancia de VinculoCP con usar_reranker=True.
    """
    global _VINCULO_RERANK
    if _VINCULO_RERANK is not None:
        return _VINCULO_RERANK
    with _LOCK:
        if _VINCULO_RERANK is None:
            _VINCULO_RERANK = nucleo.VinculoCP(usar_reranker=True)
    return _VINCULO_RERANK


# ===== SECCION 3: normalizacion de referencias =====
def normalizar_canon(referencia: str) -> str | None:
    """Normaliza un canon de sentencia al formato TIPO-NUM-AAAA.

    Args:
        referencia: Texto que contiene un canon (T-293/97, T-293-1997,
            "Sentencia SU-123/20").

    Returns:
        Canon normalizado ("T-293-1997") o None si no hay canon reconocible.
    """
    if not isinstance(referencia, str):
        return None
    coincidencia = RE_CANON.search(referencia)
    if coincidencia is None:
        return None
    tipo = coincidencia.group(1).upper()
    numero = int(coincidencia.group(2))
    anio_texto = coincidencia.group(3)
    if len(anio_texto) == 4:
        anio = int(anio_texto)
    else:
        # El corpus indexado va de 1992 a 2021: 92-99 son 19xx y 00-21, 20xx.
        anio = 1900 + int(anio_texto) if int(anio_texto) >= 70 else 2000 + int(anio_texto)
    return f"{tipo}-{numero}-{anio}"


def texto_por_id(id_corpus: str) -> tuple[str, str]:
    """Resuelve un id del corpus a (texto crudo, texto limpio).

    Args:
        id_corpus: Id numerico del documento en data/corpus_sentencias.parquet.

    Returns:
        Tupla con el texto crudo y el texto limpio del corpus.

    Raises:
        LookupError: Si el id no existe en el corpus (mensaje del entry M0).
    """
    try:
        return _modulo_entrada().texto_por_id(str(id_corpus))
    except SystemExit as exc:
        c3 = integracion_c3()["por_id"].get(str(id_corpus))
        if c3 is not None:
            return c3["texto"], c3["texto_limpio"]
        detalle = str(exc.code) if exc.code else f"El id {id_corpus} no existe."
        raise LookupError(detalle) from None


def resolver_documento(referencia: str) -> dict:
    """Resuelve una referencia a un documento analizable.

    Acepta, en este orden: un id numerico del corpus, un canon de sentencia
    (solo si la referencia es corta o es una pregunta acotada, para no
    confundir una cita mencionada dentro de un texto completo con el
    documento a analizar) o texto literal.

    Args:
        referencia: Id, canon o texto de la sentencia.

    Returns:
        Dict con id, canon, origen, texto y texto_limpio.

    Raises:
        ValueError: Si la referencia es vacia o demasiado corta como texto.
        LookupError: Si el id o el canon no existen en el corpus indexado.
    """
    if not isinstance(referencia, str) or not referencia.strip():
        raise ValueError("La referencia no puede estar vacia.")
    limpia = referencia.strip()
    if RE_ID_CORPUS.fullmatch(limpia):
        crudo, limpio = texto_por_id(limpia)
        return {
            "id": limpia,
            "canon": None,
            "origen": "id",
            "texto": crudo,
            "texto_limpio": limpio,
        }
    # Una pregunta ("... fundamentan la sentencia T-293/97?") puede superar el
    # limite corto; se acepta como referencia hasta 300 caracteres si es
    # interrogativa. Un texto completo es mucho mas largo y nunca cae aqui.
    parece_consulta = len(limpia) <= LONGITUD_MAXIMA_REFERENCIA or (
        "?" in limpia and len(limpia) <= 300
    )
    if parece_consulta:
        canon = normalizar_canon(limpia)
        if canon is not None:
            id_doc = componentes()["contexto"].id_de_canon(canon)
            if id_doc is None:
                id_doc = integracion_c3()["por_canon"].get(canon)
            if id_doc is None:
                raise LookupError(
                    f"La sentencia {canon} no esta en el corpus indexado "
                    "(1992-2026)."
                )
            crudo, limpio = texto_por_id(id_doc)
            return {
                "id": id_doc,
                "canon": canon,
                "origen": "canon",
                "texto": crudo,
                "texto_limpio": limpio,
            }
    if len(limpia) < 3:
        raise ValueError("El texto es demasiado corto para analizarlo.")
    return {
        "id": None,
        "canon": None,
        "origen": "texto",
        "texto": referencia,
        "texto_limpio": referencia,
    }


def _canon_documento(documento: dict) -> str | None:
    """Devuelve el canon del documento, detectandolo si no vino resuelto.

    Args:
        documento: Salida de resolver_documento.

    Returns:
        Canon TIPO-NUM-AAAA o None si el texto no tiene encabezado.
    """
    if documento["canon"] is not None:
        return documento["canon"]
    return componentes()["extractor"].detectar_canon_encabezado(
        documento["texto"]
    )


def _limitar_k(k: int | None) -> int:
    """Acota la profundidad de recuperacion al rango [1, K_MAXIMO]."""
    if k is None:
        return K_DEFECTO
    return max(1, min(int(k), K_MAXIMO))


def texto_constitucional(numero: str, limite: int = 300) -> str | None:
    """Devuelve el texto de un articulo de la CP sin prefijos de reforma.

    El producto muestra 150 caracteres, pero el agente necesita la frase
    completa para citar sin inventar. Se quitan las marcas de acto
    legislativo y se corta en el ultimo punto antes del limite.

    Args:
        numero: Canon del articulo ("49", "art. 86", "22A").
        limite: Caracteres maximos; se respeta el final de la frase.

    Returns:
        Texto normalizado, o None si el articulo no existe.
    """
    global _TEXTOS_CONSTITUCION
    if _TEXTOS_CONSTITUCION is None:
        import pyarrow.parquet as pq

        with _LOCK_CONSTITUCION:
            if _TEXTOS_CONSTITUCION is None:
                tabla = pq.read_table(
                    nucleo.ROOT / nucleo.CONFIG["ruta_constitucion"],
                    columns=["numero", "texto"],
                ).to_pandas()
                _TEXTOS_CONSTITUCION = {
                    str(fila.numero).upper(): " ".join(str(fila.texto).split())
                    for fila in tabla.itertuples(index=False)
                }
    canon = normalizar_articulo(numero).upper()
    original = _TEXTOS_CONSTITUCION.get(canon)
    if not original:
        return None
    limpio = re.sub(r"<[^>]+>", " ", original)
    limpio = re.sub(
        r"^El nuevo texto es el siguiente:\s*", "", limpio, flags=re.IGNORECASE
    )
    limpio = " ".join(limpio.split())
    if len(limpio) > limite:
        corte = limpio.rfind(". ", 0, limite)
        if corte >= 40:
            limpio = limpio[: corte + 1]
        else:
            # La frase en curso supera el limite: se extiende hasta su punto
            # para no entregar una cita cortada a mitad de oracion.
            siguiente = limpio.find(". ", limite)
            limpio = (
                limpio[: siguiente + 1]
                if 0 <= siguiente <= limite + 160
                else limpio[:limite].rstrip()
            )
    return limpio


def metadatos_sentencias() -> dict[str, dict]:
    """Devuelve canon, ano, tipo, PageRank y area por id de sentencia.

    Une una sola vez los artefactos ligeros del corpus (entidades, red de
    citas y numeros) y cachea el resultado. Lo usan el indice de
    jurisprudencia y el de similitud, que antes repetian el mismo join.

    Returns:
        Dict id -> {canon, ano, tipo, pagerank, area_silver}.
    """
    global _METADATOS
    if _METADATOS is not None:
        return _METADATOS
    import pandas as pd
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    with _LOCK_METADATOS:
        if _METADATOS is not None:
            return _METADATOS
        entidades = pq.read_table(
            nucleo.ROOT / "data" / "entidades_sentencias.parquet",
            columns=["id", "tipo", "ano"],
        ).to_pandas()
        ruta_red = INTEGRACION_C3_DIR / "red_citas_1992_2026.parquet"
        if not ruta_red.exists():
            ruta_red = nucleo.ROOT / nucleo.CONFIG["ruta_red_citas"]
        red = pq.read_table(
            ruta_red, columns=["id", "area", "pagerank"]
        ).to_pandas()
        numeros = ds.dataset(
            nucleo.ROOT / nucleo.CONFIG["ruta_corpus"],
            format="parquet",
            partitioning="hive",
        ).to_table(columns=["id", "numero"]).to_pandas()
        meta = entidades.merge(red, on="id", how="left").merge(
            numeros, on="id", how="left"
        )
        por_id: dict[str, dict] = {}
        for fila in meta.itertuples(index=False):
            if pd.isna(fila.numero) or pd.isna(fila.ano):
                continue
            por_id[str(fila.id)] = {
                "canon": f"{fila.tipo}-{int(fila.numero)}-{int(fila.ano)}",
                "ano": int(fila.ano),
                "tipo": str(fila.tipo),
                "pagerank": (
                    float(fila.pagerank) if pd.notna(fila.pagerank) else None
                ),
                "area_silver": str(fila.area) if pd.notna(fila.area) else None,
            }
        for id_c3, datos in integracion_c3()["por_id"].items():
            red_c3 = integracion_c3()["red"].get(id_c3, {})
            por_id[id_c3] = {
                "canon": datos["canon"],
                "ano": datos["ano"],
                "tipo": datos["tipo"],
                "pagerank": red_c3.get("pagerank"),
                "area_silver": red_c3.get("area"),
            }
        _METADATOS = por_id
    return _METADATOS


def normalizar_articulo(numero: str) -> str:
    """Normaliza el canon de un articulo de la CP ("art. 13" -> "13").

    Args:
        numero: Numero o referencia del articulo ("13", "22A", "art. 13",
            "Transitorio 68").

    Returns:
        Canon del articulo tal como aparece en la Constitucion indexada.

    Raises:
        ValueError: Si el numero queda vacio tras quitar el prefijo.
    """
    limpio = RE_PREFIJO_ARTICULO.sub("", str(numero or "").strip())
    limpio = limpio.strip().rstrip(".").strip()
    if not limpio:
        raise ValueError("El numero del articulo no puede estar vacio.")
    if RE_ARTICULO_SIMPLE.fullmatch(limpio):
        limpio = limpio.upper()
    return limpio


# ===== SECCION 4: operaciones del producto =====
def validar(texto: str) -> dict:
    """Evalua la heuristica M0 y detecta el canon del encabezado.

    Args:
        texto: Texto completo del documento a validar.

    Returns:
        Dict con es_sentencia, score_validacion, reglas, canon_detectado e
        id_detectado (id del corpus cuando el canon resuelve).

    Raises:
        ValueError: Si el texto es vacio.
    """
    if not isinstance(texto, str) or not texto.strip():
        raise ValueError("El texto a validar no puede estar vacio.")
    comp = componentes()
    resultado = comp["validador"].validar(texto)
    canon = None
    id_detectado = None
    if resultado["es_sentencia"]:
        canon = comp["extractor"].detectar_canon_encabezado(texto)
        if canon is not None:
            id_detectado = comp["contexto"].id_de_canon(canon)
    return {
        "es_sentencia": bool(resultado["es_sentencia"]),
        "score_validacion": float(resultado["score_validacion"]),
        "reglas": resultado["reglas"],
        "canon_detectado": canon,
        "id_detectado": id_detectado,
    }


def vincular(texto: str, k: int = K_DEFECTO, rerank: bool = False) -> dict:
    """Vincula una sentencia con articulos de la Constitucion Politica.

    Combina los articulos citados explicitamente (M2) con los recuperados
    por el producto (M7: BM25 + prior de hubs por defecto, ruta B2+B1 con
    rerank=True), el area tematica exploratoria (M4) y las sentencias
    citadas resueltas contra la red (M5, sin self-loops). El texto de cada
    articulo se recorta a 150 caracteres.

    Args:
        texto: Id del corpus, canon de la sentencia o texto literal.
        k: Profundidad del ranking de recuperados (1-50).
        rerank: Si True usa la ruta hibrida B2+B1 (GPU, ~14 s); si False usa
            BM25 con prior de hubs (CPU, ~5-9 s).

    Returns:
        Dict con id, canon_detectado, origen, area, articulos_extraidos,
        articulos_recuperados, sentencias_citadas, resumen, rerank, k y ms.
    """
    inicio = time.perf_counter()
    documento = resolver_documento(texto)
    comp = componentes()
    recuperador = vinculo_rerank() if rerank else comp["vinculo"]
    profundidad = _limitar_k(k)
    if rerank:
        with _LOCK_RERANK:
            recuperados = recuperador.recuperar(
                documento["texto_limpio"], profundidad
            )
    else:
        recuperados = recuperador.recuperar(
            documento["texto_limpio"], profundidad
        )
    entidades = comp["extractor"].extraer(documento["texto"])
    canon = _canon_documento(documento)
    extraidos = []
    for numero in entidades["arts_cp"]:
        detalle = recuperador.detalle_articulo(numero)
        if detalle is not None:
            extraidos.append({"numero": numero, "score": None, **detalle})
    articulos = []
    for fila in recuperados:
        detalle = recuperador.detalle_articulo(fila["numero"]) or {}
        articulos.append({**fila, **detalle})
    # M5 excluye self-loops: la autocita del encabezado no es una cita.
    canones = [
        normalizar_canon(cita) or cita
        for cita in entidades["sents_citadas"]
        if cita != canon
    ]
    citadas = [
        {"canon": cita, **resuelta}
        for cita, resuelta in zip(
            canones, comp["contexto"].resolver(canones), strict=True
        )
    ]
    area = comp["clasificador"].clasificar(documento["texto"])
    area["exploratoria"] = True
    es_sentencia = bool(
        comp["validador"].validar(documento["texto"])["es_sentencia"]
    )
    if es_sentencia:
        hubs_usado = bool(recuperador.usar_hubs)
        resumen = _modulo_entrada().construir_resumen(
            es_sentencia,
            area,
            extraidos,
            recuperados,
            citadas,
            usar_reranker=bool(recuperador.usar_reranker),
            usar_hubs=hubs_usado,
            alpha_hubs=float(recuperador.alpha_hubs) if hubs_usado else 0.0,
        )
    else:
        # construir_resumen asume que no se ejecuto ningun componente; aqui el
        # vinculo si se calculo sobre el texto literal, asi que se declara.
        resumen = (
            "Texto sin encabezado de providencia; vinculo calculado sobre el "
            f"texto literal: {len(extraidos)} articulos CP extraidos y "
            f"{len(articulos)} recuperados; "
            f"{sum(1 for fila in citadas if fila['resuelta'])} de "
            f"{len(citadas)} sentencias citadas resueltas al corpus."
        )
    return {
        "id": documento["id"],
        "canon_detectado": canon,
        "origen": documento["origen"],
        "area": area,
        "articulos_extraidos": extraidos,
        "articulos_recuperados": articulos,
        "sentencias_citadas": citadas,
        "resumen": resumen,
        "rerank": bool(rerank),
        "k": profundidad,
        "ms": int((time.perf_counter() - inicio) * 1000),
    }


def _rerankear_fragmentos(
    recuperador: object,
    chunks: list[str],
    matriz: np.ndarray,
    indices_pasaje: np.ndarray,
    seleccion: list[int],
) -> list[dict]:
    """Repuntua los fragmentos candidatos con el cross-encoder del producto.

    Reutiliza la mecanica de la ruta B2+B1: el pasaje del articulo es el de
    mayor score BM25 y cada fragmento se puntua como par (chunk, pasaje).

    Args:
        recuperador: VinculoCP en modo rerank (ya cargado).
        chunks: Chunks de la query (texto de la sentencia).
        matriz: Matriz (n_chunks, n_pasajes) de scores BM25 del caso.
        indices_pasaje: Posiciones de los pasajes del articulo en la matriz.
        seleccion: Filas de chunk candidatas (top por BM25).

    Returns:
        Lista de {texto, score} ordenada por score del cross-encoder.
    """
    maximos = matriz.max(axis=0)
    mejor = int(indices_pasaje[np.argmax(maximos[indices_pasaje])])
    pasaje = recuperador._textos_pasajes[mejor]
    scores = recuperador._cargar_reranker().puntuar(
        [chunks[i] for i in seleccion], [pasaje] * len(seleccion)
    )
    pares = sorted(
        zip(seleccion, scores, strict=True), key=lambda par: -float(par[1])
    )
    return [
        {"texto": chunks[i], "score": round(float(score), 4)}
        for i, score in pares
    ]


def evidencia(
    texto: str,
    articulos: list[str],
    rerank: bool = False,
    max_fragmentos: int = MAX_FRAGMENTOS_EVIDENCIA,
) -> dict:
    """Devuelve los fragmentos de la sentencia con mayor score por articulo.

    Para cada articulo se puntua cada chunk de la query contra sus pasajes
    con la misma maquinaria de M7 (max-pooling de BM25) y se devuelven los
    fragmentos mejor puntuados. Con rerank=True los candidatos se repuntuan
    con el cross-encoder B1/bge contra el mejor pasaje del articulo.

    Args:
        texto: Id del corpus, canon de la sentencia o texto literal.
        articulos: Numeros de articulo de la CP (hasta 5).
        rerank: Si True repuntua los fragmentos con la ruta B2+B1 (GPU).
        max_fragmentos: Fragmentos por articulo (1-3).

    Returns:
        Dict con id, canon_detectado, rerank, ms y articulos, donde cada
        articulo es {numero, fragmentos: [{texto, score}]}.

    Raises:
        ValueError: Si no se indican articulos.
        LookupError: Si la referencia o un articulo no existen en el indice.
    """
    inicio = time.perf_counter()
    if not articulos:
        raise ValueError("Indique al menos un articulo de la CP.")
    documento = resolver_documento(texto)
    comp = componentes()
    recuperador = vinculo_rerank() if rerank else comp["vinculo"]
    chunks = recuperador._chunks_query(documento["texto_limpio"])
    _, matriz = recuperador._scores_bm25_caso(chunks)
    posiciones = {
        canon: pos for pos, canon in enumerate(recuperador._canones)
    }
    canon_documento = _canon_documento(documento)
    profundidad = max(1, min(int(max_fragmentos), MAX_FRAGMENTOS_EVIDENCIA))
    evidencias = []
    for numero in articulos[:MAX_ARTICULOS_EVIDENCIA]:
        canon = normalizar_articulo(numero)
        pos = posiciones.get(canon)
        if pos is None:
            raise LookupError(
                f"El articulo '{numero}' no esta en el indice de la CP."
            )
        indices_pasaje = recuperador._grupos[pos]
        scores_chunk = matriz[:, indices_pasaje].max(axis=1)
        orden = np.argsort(-scores_chunk, kind="stable")[:profundidad]
        seleccion = [int(i) for i in orden if scores_chunk[i] > 0.0]
        if rerank and seleccion:
            with _LOCK_RERANK:
                fragmentos = _rerankear_fragmentos(
                    recuperador, chunks, matriz, indices_pasaje, seleccion
                )
        else:
            fragmentos = [
                {
                    "texto": chunks[i],
                    "score": round(float(scores_chunk[i]), 4),
                }
                for i in seleccion
            ]
        evidencias.append({"numero": canon, "fragmentos": fragmentos})
    return {
        "id": documento["id"],
        "canon_detectado": canon_documento,
        "articulos": evidencias,
        "rerank": bool(rerank),
        "ms": int((time.perf_counter() - inicio) * 1000),
    }


def contexto(
    texto: str | None = None, citas: list[str] | None = None
) -> dict:
    """Resuelve las sentencias citadas con su contexto de red (M5).

    Args:
        texto: Id, canon o texto de la sentencia de la que se extraen citas.
        citas: Lista de canones ya extraidos; si se indica, ignora `texto`.

    Returns:
        Dict con id, canon_detectado, citas (canon + resolucion M5),
        n_citas y n_resueltas.

    Raises:
        ValueError: Si no se entrega ni texto ni citas.
    """
    comp = componentes()
    id_documento = None
    canon_propio = None
    if citas is None:
        if not texto:
            raise ValueError("Se requiere 'texto' o una lista de 'citas'.")
        documento = resolver_documento(texto)
        id_documento = documento["id"]
        extraccion = comp["extractor"].extraer(documento["texto"])
        canon_propio = _canon_documento(documento)
        # M5 excluye self-loops: la autocita del encabezado no es una cita.
        citas = [
            canon for canon in extraccion["sents_citadas"] if canon != canon_propio
        ]
    canones = [normalizar_canon(canon) or canon for canon in citas]
    resueltas = comp["contexto"].resolver(canones)
    detalle = []
    for canon, resuelta in zip(canones, resueltas, strict=True):
        fila = {"canon": canon, **resuelta}
        if not fila.get("resuelta"):
            id_c3 = integracion_c3()["por_canon"].get(canon)
            if id_c3 is not None:
                datos = integracion_c3()["por_id"][id_c3]
                red = integracion_c3()["red"].get(id_c3, {})
                fila.update(
                    {
                        "id": id_c3,
                        "resuelta": True,
                        "ano": datos["ano"],
                        "pagerank": red.get("pagerank"),
                        "area": red.get("area"),
                    }
                )
        detalle.append(fila)
    return {
        "id": id_documento,
        "canon_detectado": canon_propio,
        "citas": detalle,
        "n_citas": len(detalle),
        "n_resueltas": sum(1 for fila in detalle if fila["resuelta"]),
    }


def detalle_articulo(numero: str) -> dict:
    """Devuelve el texto recortado y el bloque tematico de un articulo CP.

    Args:
        numero: Canon del articulo ("13", "22A", "art. 13", "Transitorio 68").

    Returns:
        Dict con numero, texto_150chars y bloque.

    Raises:
        ValueError: Si el numero es vacio.
        LookupError: Si el articulo no existe en la Constitucion indexada.
    """
    limpio = normalizar_articulo(numero)
    detalle = componentes()["vinculo"].detalle_articulo(limpio)
    if detalle is None:
        raise LookupError(
            f"El articulo '{numero}' no existe en la Constitucion indexada."
        )
    return {
        "numero": limpio,
        "texto_150chars": detalle["texto_150chars"],
        "bloque": detalle["bloque"],
    }


# ===== SECCION 5: estado del servicio =====
def rutas_criticas() -> dict[str, Path]:
    """Devuelve las rutas de los artefactos que necesita el backend.

    Returns:
        Dict nombre -> ruta absoluta (corpus, red de citas, modelos M4,
        prior de hubs y cache del indice BM25).
    """
    return {
        "corpus": nucleo.ROOT / nucleo.CONFIG["ruta_corpus"],
        "red_citas": nucleo.ROOT / nucleo.CONFIG["ruta_red_citas"],
        "tfidf_m4": nucleo.ROOT / nucleo.CONFIG["ruta_tfidf"],
        "logreg_m4": nucleo.ROOT / nucleo.CONFIG["ruta_logreg"],
        "prior_hubs": nucleo.ROOT / nucleo.CONFIG["ruta_prior_hubs"],
        "cache_bm25": nucleo.ROOT
        / nucleo.CONFIG["dir_cache_bm25"]
        / nucleo.CONFIG["archivo_cache_bm25"],
    }


def estado() -> dict:
    """Informa la disponibilidad de artefactos y el estado de los singletons.

    No carga modelos: solo comprueba rutas, para que /health sea barato.

    Returns:
        Dict con ready, rutas (existencia), rutas_faltantes y cargados.
    """
    presentes = {clave: ruta.exists() for clave, ruta in rutas_criticas().items()}
    return {
        "ready": all(presentes.values()),
        "rutas": presentes,
        "rutas_faltantes": sorted(
            clave for clave, existe in presentes.items() if not existe
        ),
        "cargados": {
            "componentes": _COMPONENTES is not None,
            "vinculo_rerank": _VINCULO_RERANK is not None,
            "modulo_entrada": _MODULO_ENTRADA is not None,
            "metadatos": _METADATOS is not None,
        },
    }
