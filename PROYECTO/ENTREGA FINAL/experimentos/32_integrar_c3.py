"""32_integrar_c3.py
Fecha: 2026-11-08
Descripcion: Integra el corpus C3 2022-2026 al producto como capa adicional,
    sin modificar el corpus congelado 1992-2021 ni el modelo v2.2. Reutiliza
    el corpus C3 ya preparado (texto limpiado, entidades M2 y etiquetas M4) y
    reconstruye el texto crudo desde los HTML descargados con el parser del
    script 26. Asigna ids nuevos desde 30000 para no colisionar con el corpus
    principal (0-23749), conserva el canon en formato TIPO-NUM-AAAA y escribe:

    - data/integracion_c3/corpus_c3.parquet: id, canon, numero, tipo, ano,
      fecha, texto crudo, texto_limpio, url, estado.
    - data/integracion_c3/entidades_c3.parquet: esquema de
      entidades_sentencias.parquet, con el id nuevo.
    - data/integracion_c3/etiquetas_c3.parquet: id nuevo, area_primaria,
      top3_json, via.
    - data/integracion_c3/canon_map_c3.parquet: canon, id, tipo, ano.
    - data/integracion_c3/red_citas_1992_2026.parquet: id, tipo, ano, area,
      pagerank e indegree de la red combinada (23.742 + 2.330 nodos).
    - data/integracion_c3/resumen_integracion.json: cobertura y conteos.

Estado del arte aplicado: reutilizacion del parser de 26 y de las entidades
    de 28 por importlib, ids con espacio de nombres propio, PageRank con
    networkx sobre el grafo dirigido combinado y resolucion de citas por
    canon normalizado TIPO-NUM-AAAA.
"""

import importlib.util
import json
import logging
import time
from pathlib import Path

import networkx as nx
import pandas as pd

BASE = Path(__file__).resolve().parent
SEGUNDA = BASE.parents[1] / "SEGUNDA ENTREGA"
DATA = SEGUNDA / "data"
C3_DIR = DATA / "corpus_2022_2026"
SALIDA = DATA / "integracion_c3"
SALIDA.mkdir(parents=True, exist_ok=True)

ID_BASE = 30000
LOG = logging.getLogger("integrar_c3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def cargar_modulo(ruta: Path, nombre: str):
    """Carga un script del proyecto como modulo sin ejecutar su main."""
    especificacion = importlib.util.spec_from_file_location(nombre, ruta)
    if especificacion is None or especificacion.loader is None:
        raise ImportError(f"No se pudo cargar {ruta}")
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


def leer_metadatos() -> dict[str, dict]:
    """Lee metadatos.jsonl indexado por id canon."""
    registros = {}
    with (C3_DIR / "metadatos.jsonl").open(encoding="utf-8") as archivo:
        for linea in archivo:
            if not linea.strip():
                continue
            fila = json.loads(linea)
            registros[str(fila["id"])] = fila
    return registros


def canon_desde(tipo: str, numero: int, anio: int) -> str:
    """Arma el canon normalizado TIPO-NUM-AAAA."""
    return f"{tipo}-{int(numero)}-{int(anio)}"


def ruta_html(doc_id: str, anio: int) -> Path:
    """Replica la convencion de nombres de 26 para el HTML crudo."""
    nombre = str(doc_id).replace("/", "-")
    if nombre.startswith("SU-"):
        nombre = "SU" + nombre[3:]
    return C3_DIR / "raw" / str(int(anio)) / f"{nombre}.html"


def main() -> int:
    """Construye la capa de integracion y escribe el resumen."""
    inicio = time.time()
    mod26 = cargar_modulo(SEGUNDA / "scripts" / "26_c3_descarga_2022_2026.py", "c3_26")

    corpus = pd.read_parquet(C3_DIR / "corpus_2022_2026.parquet")
    corpus = corpus[corpus["estado"] == "ok"].reset_index(drop=True)
    metadatos = leer_metadatos()
    entidades = pd.read_parquet(C3_DIR / "entidades_2022_2026.parquet")
    etiquetas = pd.read_parquet(C3_DIR / "etiquetas_2022_2026.parquet")

    filas = []
    fallback = []
    for posicion, fila in enumerate(corpus.itertuples(index=False)):
        doc_id = str(fila.id)
        meta = metadatos.get(doc_id, {})
        anio = int(meta.get("anio") or fila.anio)
        numero = int(meta.get("numero") or 0)
        tipo = str(meta.get("tipo") or fila.tipo)
        canon = canon_desde(tipo, numero, anio)
        crudo = ""
        html = ruta_html(doc_id, anio)
        if html.exists():
            crudo = mod26.extraer_texto_html(html, mod26.CONFIG)
        if not crudo.strip():
            crudo = str(fila.texto)
            fallback.append(doc_id)
        filas.append(
            {
                "id": str(ID_BASE + posicion),
                "id_fuente": doc_id,
                "canon": canon,
                "numero": numero,
                "tipo": tipo,
                "ano": anio,
                "fecha": meta.get("fecha"),
                "texto": crudo,
                "texto_limpio": str(fila.texto),
                "url": meta.get("url_texto"),
                "estado": str(fila.estado),
            }
        )

    df_corpus = pd.DataFrame(filas)
    mapa = df_corpus[["id_fuente", "canon", "id", "tipo", "ano"]].copy()
    df_corpus.to_parquet(SALIDA / "corpus_c3.parquet", index=False)
    mapa.to_parquet(SALIDA / "canon_map_c3.parquet", index=False)
    LOG.info("Corpus C3: %d docs, %d con texto crudo", len(df_corpus), len(df_corpus) - len(fallback))

    # Entidades y etiquetas con el id nuevo.
    entidades = entidades.copy()
    entidades["id"] = entidades["id"].astype(str)
    entidades = entidades.merge(mapa, left_on="id", right_on="id_fuente", how="inner")
    entidades = entidades[["id_y", "tipo_y", "ano_y", "arts_cp", "normas", "sents_citadas", "ponente"]]
    entidades.columns = ["id", "tipo", "ano", "arts_cp", "normas", "sents_citadas", "ponente"]
    entidades.to_parquet(SALIDA / "entidades_c3.parquet", index=False)

    etiquetas = etiquetas.copy()
    etiquetas["id"] = etiquetas["id"].astype(str)
    etiquetas = etiquetas.merge(mapa, left_on="id", right_on="id_fuente", how="inner")
    etiquetas = etiquetas[["id_y", "area_primaria", "top3_json", "via"]]
    etiquetas.columns = ["id", "area_primaria", "top3_json", "via"]
    etiquetas.to_parquet(SALIDA / "etiquetas_c3.parquet", index=False)
    LOG.info("Entidades: %d | etiquetas: %d", len(entidades), len(etiquetas))

    # Red combinada: nodos del corpus principal + C3.
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    principal = ds.dataset(DATA / "corpus_sentencias.parquet", format="parquet", partitioning="hive")
    tabla_principal = principal.to_table(columns=["id", "numero", "tipo", "ano"]).to_pandas()
    tabla_principal["id"] = tabla_principal["id"].astype(str)
    tabla_principal["canon"] = [
        canon_desde(tipo, numero, anio)
        for tipo, numero, anio in zip(
            tabla_principal["tipo"], tabla_principal["numero"], tabla_principal["ano"], strict=True
        )
    ]
    canon_a_id = dict(zip(tabla_principal["canon"], tabla_principal["id"], strict=True))
    canon_a_id.update(dict(zip(mapa["canon"], mapa["id"], strict=True)))

    ent_principal = pq.read_table(DATA / "entidades_sentencias.parquet", columns=["id", "sents_citadas"]).to_pandas()
    ent_principal["id"] = ent_principal["id"].astype(str)
    ent_c3 = entidades[["id", "sents_citadas"]]

    grafo = nx.DiGraph()
    grafo.add_nodes_from(tabla_principal["id"])
    grafo.add_nodes_from(mapa["id"])
    aristas = 0
    sin_resolver = 0
    for fila in pd.concat([ent_principal, ent_c3], ignore_index=True).itertuples(index=False):
        origen = str(fila.id)
        try:
            citas = json.loads(fila.sents_citadas or "[]")
        except (TypeError, ValueError):
            citas = []
        for canon in citas:
            destino = canon_a_id.get(str(canon))
            if destino and destino != origen:
                grafo.add_edge(origen, destino)
                aristas += 1
            elif not destino:
                sin_resolver += 1
    LOG.info("Grafo: %d nodos, %d aristas, %d citas sin resolver", grafo.number_of_nodes(), aristas, sin_resolver)

    pagerank = nx.pagerank(grafo, alpha=0.85, tol=1e-8, max_iter=200)
    viejo = pq.read_table(DATA / "red_citas_resumen.parquet", columns=["id", "area"]).to_pandas()
    viejo["id"] = viejo["id"].astype(str)
    area_vieja = dict(zip(viejo["id"], viejo["area"], strict=True))
    area_nueva = dict(zip(etiquetas["id"], etiquetas["area_primaria"], strict=True))
    filas_red = []
    for doc_id in grafo.nodes:
        meta = tabla_principal[tabla_principal["id"] == doc_id]
        if not meta.empty:
            tipo = str(meta.iloc[0]["tipo"])
            anio = int(meta.iloc[0]["ano"])
            area = area_vieja.get(doc_id)
        else:
            fila_c3 = df_corpus[df_corpus["id"] == doc_id].iloc[0]
            tipo = str(fila_c3["tipo"])
            anio = int(fila_c3["ano"])
            area = area_nueva.get(doc_id)
        filas_red.append(
            {
                "id": doc_id,
                "tipo": tipo,
                "ano": anio,
                "area": area,
                "pagerank": float(pagerank[doc_id]),
                "indegree": int(grafo.in_degree(doc_id)),
            }
        )
    pd.DataFrame(filas_red).to_parquet(SALIDA / "red_citas_1992_2026.parquet", index=False)
    LOG.info("Red combinada escrita: %d filas", len(filas_red))

    top_nuevos = sorted(
        (fila for fila in filas_red if int(fila["id"]) >= ID_BASE),
        key=lambda fila: -fila["pagerank"],
    )[:5]
    resumen = {
        "fecha": pd.Timestamp.now(tz="UTC").isoformat(),
        "docs_c3": len(df_corpus),
        "docs_texto_crudo": len(df_corpus) - len(fallback),
        "docs_fallback_limpio": len(fallback),
        "fallback": fallback[:20],
        "entidades": len(entidades),
        "etiquetas": len(etiquetas),
        "nodos_red": grafo.number_of_nodes(),
        "aristas_red": aristas,
        "citas_sin_resolver": sin_resolver,
        "id_base": ID_BASE,
        "top_pagerank_c3": [
            {"id": fila["id"], "tipo": fila["tipo"], "ano": fila["ano"], "pagerank": fila["pagerank"]}
            for fila in top_nuevos
        ],
        "segundos": round(time.time() - inicio, 1),
    }
    (SALIDA / "resumen_integracion.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    LOG.info("Integracion lista en %.1f s", time.time() - inicio)
    print(json.dumps(resumen, ensure_ascii=False, indent=1)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
