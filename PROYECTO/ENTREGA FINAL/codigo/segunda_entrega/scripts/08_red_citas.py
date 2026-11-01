"""08_red_citas.py
Fecha: 2026-09-09
Descripcion: Red dirigida de citas entre sentencias de la Corte
    Constitucional (modulo M5). Construye el grafo (nodo = sentencia,
    arista A -> B si A cita B) a partir de las sentencias citadas
    extraidas por el NER (canon TIPO-NUM-AAAA) resueltas al corpus; las
    citas no resueltas se ignoran con contador y los self-loops se
    excluyen. Calcula PageRank, in-degree, componentes fuertemente
    conexas, densidad y hubs vs authorities (HITS), y la evolucion del
    PageRank por periodo de emision (1992-2001, 2002-2011, 2012-2021).
Estado del arte aplicado: Fowler et al. (2007), analisis de redes de
    citas de la Corte Suprema de EE. UU.: PageRank como medida de
    precedentes landmark sin sesgo de conteo bruto de citas; HITS para
    separar hubs (sentencias que citan mucho) de authorities (sentencias
    muy citadas).
"""

import json
import random
import sys
import time
import traceback
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CONFIG = {
    "semilla": 42,
    "damping": 0.85,
    "ruta_entidades": "data/entidades_sentencias.parquet",
    "ruta_corpus": "data/corpus_sentencias.parquet",
    "ruta_areas": "data/etiquetas_areas.parquet",
    "ruta_resumen": "data/red_citas_resumen.parquet",
    "ruta_json": "resultados/red_citas.json",
    "fig_top20": "figuras/fig_red_citas_pagerank.png",
    "fig_evolucion": "figuras/fig_pagerank_evolucion.png",
    "periodos": [(1992, 2001), (2002, 2011), (2012, 2021)],
    "top_pagerank": 20,
    "top_tabla": 10,
    "pares_doc_canon_esperados": 540181,
    "aristas_resueltas_ner": 456321,
}

COLOR_TIPO = {"T": "#2f5d8a", "C": "#c0504d", "SU": "#7f9c3a"}


def cargar_datos():
    """Carga entidades, corpus (canon -> id) y areas tematicas.

    Returns:
        Tupla (df_entidades, canon_a_id, area_por_id) donde df_entidades
        tiene columnas id/tipo/ano/sents_citadas, canon_a_id mapea el
        canon TIPO-NUM-AAAA al id del documento citado y area_por_id
        mapea id -> area_primaria.

    Raises:
        AssertionError: Si los ids de las tres fuentes no coinciden.
    """
    df_corpus = pq.read_table(
        CONFIG["ruta_corpus"], columns=["id", "numero", "tipo", "ano"]
    ).to_pandas()
    canon_a_id = {
        f"{tipo}-{int(numero)}-{int(ano)}": str(id_doc)
        for tipo, numero, ano, id_doc in zip(
            df_corpus["tipo"],
            df_corpus["numero"],
            df_corpus["ano"],
            df_corpus["id"],
        )
    }
    df_ent = pq.read_table(
        CONFIG["ruta_entidades"], columns=["id", "tipo", "ano", "sents_citadas"]
    ).to_pandas()
    df_ent["id"] = df_ent["id"].astype(str)
    df_areas = pq.read_table(
        CONFIG["ruta_areas"], columns=["id", "area_primaria"]
    ).to_pandas()
    area_por_id = dict(
        zip(df_areas["id"].astype(str), df_areas["area_primaria"])
    )
    ids_ent = set(df_ent["id"])
    assert ids_ent == set(df_corpus["id"].astype(str)), (
        "ids de entidades y corpus no coinciden"
    )
    assert ids_ent == set(area_por_id), "ids de entidades y areas no coinciden"
    print(
        f"Documentos: {len(df_ent)} | canones unicos: {len(canon_a_id)} "
        f"| areas: {len(set(area_por_id.values()))}"
    )
    return df_ent, canon_a_id, area_por_id


def construir_grafo(df_ent, canon_a_id, area_por_id):
    """Construye el grafo dirigido de citas sin self-loops.

    Args:
        df_ent: DataFrame con id, tipo, ano y sents_citadas (JSON-list).
        canon_a_id: Dict canon TIPO-NUM-AAAA -> id del documento citado.
        area_por_id: Dict id -> area_primaria.

    Returns:
        Tupla (grafo, contadores) donde grafo es nx.DiGraph con atributos
        tipo/ano/area por nodo y contadores lleva las citas unicas,
        resueltas, no resueltas y self-loops excluidos.
    """
    grafo = nx.DiGraph()
    contadores = {
        "citas_unicas_doc_canon": 0,
        "citas_resueltas": 0,
        "citas_no_resueltas": 0,
        "self_loops_excluidos": 0,
        "docs_sin_citas": 0,
    }
    aristas = set()
    for fila in df_ent.itertuples(index=False):
        grafo.add_node(
            fila.id,
            tipo=fila.tipo,
            ano=int(fila.ano),
            area=area_por_id.get(fila.id),
        )
        if not fila.sents_citadas or fila.sents_citadas == "[]":
            contadores["docs_sin_citas"] += 1
            continue
        citadas = json.loads(fila.sents_citadas)
        for canon in citadas:
            contadores["citas_unicas_doc_canon"] += 1
            destino = canon_a_id.get(canon)
            if destino is None:
                contadores["citas_no_resueltas"] += 1
            elif destino == fila.id:
                contadores["self_loops_excluidos"] += 1
            else:
                contadores["citas_resueltas"] += 1
                aristas.add((fila.id, destino))
    grafo.add_edges_from(aristas)
    pct_resueltas = (
        100.0
        * contadores["citas_resueltas"]
        / contadores["citas_unicas_doc_canon"]
    )
    print(
        f"Aristas (sin self-loops): {grafo.number_of_edges()} | "
        f"citas unicas: {contadores['citas_unicas_doc_canon']} | "
        f"resueltas: {contadores['citas_resueltas']} "
        f"({pct_resueltas:.2f}%) | no resueltas: "
        f"{contadores['citas_no_resueltas']} | self-loops: "
        f"{contadores['self_loops_excluidos']}"
    )
    return grafo, contadores


def calcular_hits(grafo):
    """Calcula HITS (hubs y authorities) con respaldo ante no convergencia.

    Args:
        grafo: DiGraph de citas.

    Returns:
        Tupla (hubs, authorities, metodo) con los scores por nodo y una
        etiqueta del metodo usado ("grafo_completo", "subgrafo_nucleo" o
        "no_convergio" con dicts vacios).
    """
    try:
        hubs, authorities = nx.hits(grafo, max_iter=1000, tol=1e-8)
        return hubs, authorities, "grafo_completo"
    except nx.PowerIterationFailedError:
        nucleo = grafo.subgraph(
            [nodo for nodo, grado in grafo.degree() if grado > 0]
        )
        try:
            hubs, authorities = nx.hits(nucleo, max_iter=5000, tol=1e-8)
            return hubs, authorities, "subgrafo_nucleo"
        except nx.PowerIterationFailedError:
            print("Aviso: HITS no convergio; se reporta vacio.")
            return {}, {}, "no_convergio"


def calcular_metricas(grafo):
    """Calcula PageRank, in-degree, SCC, densidad y HITS del grafo.

    Args:
        grafo: DiGraph de citas con atributos tipo/ano/area.

    Returns:
        Dict con pagerank, indegree, n_scc, tamano_gigante, densidad,
        hubs, authorities y metodo_hits.
    """
    pagerank = nx.pagerank(
        grafo, alpha=CONFIG["damping"], max_iter=100, tol=1e-6
    )
    indegree = dict(grafo.in_degree())
    n_scc = nx.number_strongly_connected_components(grafo)
    gigante = max(nx.strongly_connected_components(grafo), key=len)
    densidad = nx.density(grafo)
    hubs, authorities, metodo_hits = calcular_hits(grafo)
    print(
        f"SCC: {n_scc} componentes | gigante: {len(gigante)} nodos "
        f"({100.0 * len(gigante) / grafo.number_of_nodes():.2f}%) | "
        f"densidad: {densidad:.6f} | HITS: {metodo_hits}"
    )
    return {
        "pagerank": pagerank,
        "indegree": indegree,
        "n_scc": n_scc,
        "tamano_gigante": len(gigante),
        "densidad": densidad,
        "hubs": hubs,
        "authorities": authorities,
        "metodo_hits": metodo_hits,
    }


def construir_resumen(grafo, metricas):
    """Construye la tabla por sentencia para reuso y figuras.

    Args:
        grafo: DiGraph con atributos tipo/ano/area.
        metricas: Dict de calcular_metricas.

    Returns:
        DataFrame con id, tipo, ano, area, pagerank, indegree, hub y
        authority, ordenado por pagerank descendente.
    """
    filas = []
    for nodo, attrs in grafo.nodes(data=True):
        filas.append(
            {
                "id": nodo,
                "tipo": attrs["tipo"],
                "ano": attrs["ano"],
                "area": attrs["area"],
                "pagerank": metricas["pagerank"][nodo],
                "indegree": metricas["indegree"][nodo],
                "hub": metricas["hubs"].get(nodo, 0.0),
                "authority": metricas["authorities"].get(nodo, 0.0),
            }
        )
    return pd.DataFrame(filas).sort_values("pagerank", ascending=False)


def tabla_top(resumen, columna, n):
    """Construye la tabla top-N de una metrica con metadatos.

    Args:
        resumen: DataFrame de construir_resumen.
        columna: Nombre de la metrica ("pagerank", "indegree", "hub",
            "authority").
        n: Tamaño del top.

    Returns:
        Lista de dicts con id, ano, tipo, area, valor de la metrica.
    """
    top = resumen.sort_values(columna, ascending=False).head(n)
    salida = []
    for _indice, fila in top.iterrows():
        salida.append(
            {
                "id": fila["id"],
                "ano": int(fila["ano"]),
                "tipo": fila["tipo"],
                "area": fila["area"],
                columna: round(float(fila[columna]), 8),
            }
        )
    return salida


def etiqueta_periodo(ano):
    """Devuelve la etiqueta del periodo de emision de un año.

    Args:
        ano: Año de emision.

    Returns:
        Cadena "AAAA-AAAA" del periodo que contiene el año.
    """
    for inicio, fin in CONFIG["periodos"]:
        if inicio <= ano <= fin:
            return f"{inicio}-{fin}"
    return "fuera_de_rango"


def top_por_periodo(resumen):
    """Calcula el top-10 de PageRank por periodo de emision.

    El PageRank es global (mide influencia de precedentes sobre todo el
    grafo, siguiendo a Fowler et al. 2007); el corte por periodo filtra
    las sentencias por su año de emision.

    Args:
        resumen: DataFrame de construir_resumen.

    Returns:
        Dict etiqueta_periodo -> lista top-10 (id, ano, tipo, area,
        pagerank).
    """
    salida = {}
    for inicio, fin in CONFIG["periodos"]:
        etiqueta = f"{inicio}-{fin}"
        mascara = resumen["ano"].between(inicio, fin)
        top = resumen[mascara].head(CONFIG["top_tabla"])
        salida[etiqueta] = [
            {
                "id": fila.id,
                "ano": int(fila.ano),
                "tipo": fila.tipo,
                "area": fila.area,
                "pagerank": round(float(fila.pagerank), 8),
            }
            for fila in top.itertuples(index=False)
        ]
    return salida


def figura_top20(resumen):
    """Genera la figura de barras del top-20 PageRank con año y tipo.

    Args:
        resumen: DataFrame de construir_resumen.
    """
    top = resumen.head(CONFIG["top_pagerank"]).iloc[::-1]
    etiquetas = [f"{fila.id} ({fila.ano})" for fila in top.itertuples()]
    colores = [COLOR_TIPO.get(fila.tipo, "#666666") for fila in top.itertuples()]
    plt.figure(figsize=(10, 8))
    plt.barh(etiquetas, top["pagerank"].to_numpy(), color=colores)
    plt.xlabel("PageRank (damping 0.85)")
    plt.title("Top-20 sentencias landmark por PageRank (red de citas)")
    marcas = [
        plt.Line2D([0], [0], color=color, lw=6, label=tipo)
        for tipo, color in COLOR_TIPO.items()
    ]
    plt.legend(handles=marcas, title="Tipo", loc="lower right")
    plt.tight_layout()
    plt.savefig(CONFIG["fig_top20"], dpi=150, bbox_inches="tight")
    plt.close()
    print("Figura guardada:", CONFIG["fig_top20"])


def figura_evolucion(tops):
    """Genera la figura del top-10 PageRank por periodo de emision.

    Args:
        tops: Dict etiqueta_periodo -> lista top-10 (salida de
            top_por_periodo).
    """
    fig, ejes = plt.subplots(1, 3, figsize=(18, 6))
    for eje, (etiqueta, filas) in zip(ejes, tops.items()):
        filas_ord = list(reversed(filas))
        etiquetas_y = [f"{f['id']} ({f['ano']})" for f in filas_ord]
        valores = [f["pagerank"] for f in filas_ord]
        colores = [COLOR_TIPO.get(f["tipo"], "#666666") for f in filas_ord]
        eje.barh(etiquetas_y, valores, color=colores)
        eje.set_title(f"Periodo {etiqueta}")
        eje.set_xlabel("PageRank")
        eje.tick_params(axis="y", labelsize=8)
    fig.suptitle("Evolucion del PageRank: top-10 por periodo de emision")
    plt.tight_layout()
    plt.savefig(CONFIG["fig_evolucion"], dpi=150, bbox_inches="tight")
    plt.close()
    print("Figura guardada:", CONFIG["fig_evolucion"])


def guardar_resultados(grafo, contadores, metricas, resumen, tops):
    """Escribe el JSON de resultados y el parquet de reuso.

    Args:
        grafo: DiGraph de citas.
        contadores: Contadores de construccion del grafo.
        metricas: Dict de calcular_metricas.
        resumen: DataFrame de construir_resumen.
        tops: Dict de top_por_periodo.
    """
    n_nodos = grafo.number_of_nodes()
    n_aristas = grafo.number_of_edges()
    resultado = {
        "fecha": datetime.now(timezone.utc).date().isoformat(),
        "grafo": {
            "n_nodos": int(n_nodos),
            "n_aristas": int(n_aristas),
            "densidad": round(float(metricas["densidad"]), 8),
            "citas_unicas_doc_canon": int(contadores["citas_unicas_doc_canon"]),
            "citas_resueltas": int(contadores["citas_resueltas"]),
            "citas_no_resueltas": int(contadores["citas_no_resueltas"]),
            "pct_resolucion_sobre_unicas": round(
                100.0
                * contadores["citas_resueltas"]
                / contadores["citas_unicas_doc_canon"],
                2,
            ),
            "self_loops_excluidos": int(contadores["self_loops_excluidos"]),
            "docs_sin_citas": int(contadores["docs_sin_citas"]),
            "coincide_con_ner": bool(
                contadores["citas_unicas_doc_canon"]
                == CONFIG["pares_doc_canon_esperados"]
                and n_aristas == CONFIG["aristas_resueltas_ner"]
            ),
        },
        "scc": {
            "n_componentes_fuertemente_conexas": int(metricas["n_scc"]),
            "tamano_componente_gigante": int(metricas["tamano_gigante"]),
            "pct_nodos_en_gigante": round(
                100.0 * metricas["tamano_gigante"] / n_nodos, 2
            ),
            "n_componentes_restantes": int(
                metricas["n_scc"] - 1 if metricas["tamano_gigante"] > 0 else 0
            ),
        },
        "pagerank_top20": tabla_top(resumen, "pagerank", CONFIG["top_pagerank"]),
        "indegree_top10": tabla_top(resumen, "indegree", CONFIG["top_tabla"]),
        "hits": {
            "metodo": metricas["metodo_hits"],
            "top_hubs": tabla_top(resumen, "hub", CONFIG["top_tabla"]),
            "top_authorities": tabla_top(
                resumen, "authority", CONFIG["top_tabla"]
            ),
        },
        "evolucion_pagerank": tops,
    }
    with open(CONFIG["ruta_json"], "w", encoding="utf-8") as fh:
        json.dump(resultado, fh, ensure_ascii=False, indent=2)
    print("JSON guardado:", CONFIG["ruta_json"])
    resumen.to_parquet(CONFIG["ruta_resumen"], index=False)
    print("Parquet guardado:", CONFIG["ruta_resumen"])


def main():
    """Orquesta la seccion M5: grafo, metricas, figuras y salidas."""
    random.seed(CONFIG["semilla"])
    np.random.seed(CONFIG["semilla"])
    t0 = time.time()
    df_ent, canon_a_id, area_por_id = cargar_datos()
    grafo, contadores = construir_grafo(df_ent, canon_a_id, area_por_id)
    t_grafo = time.time()
    metricas = calcular_metricas(grafo)
    resumen = construir_resumen(grafo, metricas)
    tops = top_por_periodo(resumen)
    figura_top20(resumen)
    figura_evolucion(tops)
    guardar_resultados(grafo, contadores, metricas, resumen, tops)
    t_fin = time.time()
    print()
    print("RESUMEN M5 - RED DE CITAS")
    print("-" * 56)
    print(f"Nodos: {grafo.number_of_nodes()} | aristas: {grafo.number_of_edges()}")
    print(f"SCC: {metricas['n_scc']} | gigante: {metricas['tamano_gigante']}")
    print("Top-5 PageRank:")
    for fila in resumen.head(5).itertuples(index=False):
        print(
            f"  {fila.id} ({fila.ano}, {fila.tipo}, {fila.area}): "
            f"{fila.pagerank:.6f}"
        )
    print(f"Tiempos: grafo {t_grafo - t0:.1f}s, metricas+figuras "
          f"{t_fin - t_grafo:.1f}s, total {t_fin - t0:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - salida con traceback completo por convencion
        traceback.print_exc()
        sys.exit(1)
