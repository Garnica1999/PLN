"""jurisprudencia.py
Fecha: 2026-10-31
Descripcion: Indice invertido articulo CP -> sentencias que lo citan (P2 de
    la Entrega Final: "de articulo a jurisprudencia"). Se construye una sola
    vez desde los artefactos congelados de la SEGUNDA ENTREGA: la columna
    arts_cp de data/entidades_sentencias.parquet (weak labels de M2) y
    data/red_citas_resumen.parquet (PageRank y area silver) mas los numeros
    de sentencia ligeros del corpus. La carga es perezosa (la primera
    peticion paga ~0,2 s) y queda cacheada en memoria bajo lock; no carga
    modelos ni el indice BM25.
Estado del arte aplicado: indice invertido precomputado en memoria, join
    ligero con pyarrow y pandas, y rotulado explicito de la etiqueta silver
    como referencia automatica (no verificacion humana).
"""

import json
import threading

import pandas as pd
import pyarrow.parquet as pq

import core
from core import nucleo

RUTA_ENTIDADES = nucleo.ROOT / "data" / "entidades_sentencias.parquet"
RUTA_CONSTITUCION = nucleo.ROOT / nucleo.CONFIG["ruta_constitucion"]

TOP_DEFECTO = 10
TOP_MAXIMO = 50
# Tope defensivo: M2 extrae hasta ~1.350 articulos por sentencia y el
# payload del top no debe crecer sin control.
AVISO_AREA_SILVER = (
    "El campo area_silver es la etiqueta automatica del corpus (silver M4, "
    "sin verificacion humana): es una referencia, no una clasificacion. El "
    "orden usa el PageRank de la red de citas y el total cuenta sentencias "
    "del corpus 1992-2026 cuyo bloque arts_cp incluye el articulo (weak "
    "labels de M2, no citas verificadas)."
)

_LOCK = threading.Lock()
_ARTEFACTOS: dict | None = None


def _leer_meta() -> tuple[dict[str, list[str]], set[str]]:
    """Lee el parquet de entidades y arma el indice invertido articulo -> ids.

    Los metadatos por sentencia (canon, ano, tipo, PageRank y area) salen de
    core.metadatos_sentencias(), que une los artefactos ligeros una sola vez.

    Returns:
        Tupla (indice, articulos): indice articulo -> ids que lo citan y
        conjunto de numeros de articulo existentes en la CP.

    Raises:
        FileNotFoundError: Si falta alguno de los artefactos congelados.
    """
    entidades = pq.read_table(RUTA_ENTIDADES, columns=["id", "arts_cp"]).to_pandas()
    ruta_c3 = core.INTEGRACION_C3_DIR / "entidades_c3.parquet"
    if ruta_c3.exists():
        entidades = pd.concat(
            [
                entidades,
                pq.read_table(ruta_c3, columns=["id", "arts_cp"]).to_pandas(),
            ],
            ignore_index=True,
        )
    constitucion = pd.read_parquet(RUTA_CONSTITUCION, columns=["numero"])
    indice: dict[str, list[str]] = {}
    for id_sentencia, arts_cp in zip(
        entidades["id"], entidades["arts_cp"], strict=True
    ):
        for numero in json.loads(arts_cp or "[]"):
            indice.setdefault(str(numero).upper(), []).append(str(id_sentencia))
    articulos = {
        str(numero).upper() for numero in constitucion["numero"].astype(str)
    }
    return indice, articulos


def artefactos() -> dict:
    """Devuelve el indice y los metadatos cargados (singleton bajo lock).

    Returns:
        Dict con por_id, indice y articulos (ver _leer_meta).

    Raises:
        LookupError: Si faltan los artefactos congelados de la SEGUNDA
            ENTREGA.
    """
    global _ARTEFACTOS
    if _ARTEFACTOS is not None:
        return _ARTEFACTOS
    with _LOCK:
        if _ARTEFACTOS is None:
            faltantes = [
                str(ruta)
                for ruta in (RUTA_ENTIDADES, RUTA_CONSTITUCION)
                if not ruta.exists()
            ]
            if faltantes:
                raise LookupError(
                    "Faltan artefactos del corpus para el indice de "
                    f"jurisprudencia: {', '.join(faltantes)}."
                )
            indice, articulos = _leer_meta()
            _ARTEFACTOS = {
                "por_id": core.metadatos_sentencias(),
                "indice": indice,
                "articulos": articulos,
            }
    return _ARTEFACTOS


def por_articulo(numero: str, top: int = TOP_DEFECTO) -> dict:
    """Lista las sentencias del corpus que citan un articulo de la CP.

    Args:
        numero: Numero del articulo ("86", "art. 22A"). Se normaliza con la
            misma regla que /v1/article.
        top: Tamano del ranking por PageRank (1-50).

    Returns:
        Dict con numero, total, por_ano, top (id, canon, ano, tipo,
        pagerank, area_silver) y aviso sobre la etiqueta automatica.

    Raises:
        ValueError: Si el numero de articulo es vacio.
        LookupError: Si el articulo no existe en la Constitucion indexada.
    """
    canon = core.normalizar_articulo(numero)
    datos = artefactos()
    if canon not in datos["articulos"]:
        raise LookupError(
            f"El articulo '{numero}' no existe en la Constitucion indexada."
        )
    por_id: dict[str, dict] = datos["por_id"]
    ids = datos["indice"].get(canon, [])
    profundidad = max(1, min(int(top), TOP_MAXIMO))
    filas = [
        {"id": id_sentencia, **por_id[id_sentencia]}
        for id_sentencia in ids
        if id_sentencia in por_id
    ]
    por_ano: dict[str, int] = {}
    for fila in filas:
        clave = str(fila["ano"])
        por_ano[clave] = por_ano.get(clave, 0) + 1
    top_filas = sorted(
        filas, key=lambda fila: -(fila["pagerank"] or 0.0)
    )[:profundidad]
    return {
        "numero": canon,
        "total": len(filas),
        "por_ano": dict(sorted(por_ano.items())),
        "top": top_filas,
        "aviso": AVISO_AREA_SILVER,
    }
