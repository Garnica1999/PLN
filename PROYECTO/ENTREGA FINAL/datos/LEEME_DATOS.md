# Datos y modelos de la entrega

Los datos pesados **no se duplican** dentro de esta carpeta: se referencian desde
`SEGUNDA ENTREGA/` con enlaces simbólicos (misma máquina) y se documentan aquí
para que la entrega sea reproducible sin copiar ~47 GB.

## Enlaces (creados en esta máquina)

| Enlace | Destino | Contenido | Tamaño |
|---|---|---|---:|
| `datos/data` | `PROYECTO/SEGUNDA ENTREGA/data` | corpus, entidades, cachés de recuperación | 9,4 GB |
| `datos/modelos` | `PROYECTO/SEGUNDA ENTREGA/modelos` | B1, B2, M4, LoRA | 30 GB |
| `datos/resultados_completos` | `PROYECTO/SEGUNDA ENTREGA/resultados` | matrices y predicciones intermedias | 3,1 GB |

En el zip de la entrega los enlaces se sustituyen por esta nota: los datos se
regeneran con los scripts de `codigo/segunda_entrega/scripts/`.

## Archivos que el producto necesita en tiempo de ejecución

| Archivo | Rol | Tamaño |
|---|---|---:|
| `data/corpus_sentencias.parquet` | 23.742 sentencias históricas (id, texto, metadatos) | 1,5 GB |
| `data/integracion_c3/` | capa 2022-2026: 2.330 sentencias con texto crudo, entidades, etiquetas y red combinada | ~100 MB |
| `data/entidades_sentencias.parquet` | artículos y citas extraídos por documento | 4,3 MB |
| `data/red_citas_resumen.parquet` | PageRank y grados de la red de citas | 5 MB |
| `data/constitucion_articulos.parquet` | texto de los artículos de la CP | < 1 MB |
| `data/cache_bm25_cp/` | índice BM25 sobre la CP | 3,2 MB |
| `modelos/b1_reranker/` | reranker B1 (ruta rerank) | 8,5 GB |
| `modelos/b2_retriever/` | retriever B2 (ruta rerank) | 2,3 GB |
| `data/cache_emb_cp/` | embeddings de la CP para B2 | 13 MB |

La ruta por defecto (BM25 + prior de hubs) solo usa los cuatro primeros
archivos y corre en CPU. La ruta `rerank=True` (v2.2: B2 + B1) requiere GPU
ROCm con ~14 GB de VRAM libre; verificar antes de usarla:

```bash
rocm-smi --showmeminfo vram
rocm-smi --showpids
```

## Regeneración

| Artefacto | Script |
|---|---|
| Corpus curado | `codigo/segunda_entrega/scripts/01_curacion_corpus.py` |
| Extracción de entidades (M2) | `scripts/03_extraccion_entidades.py` |
| Red de citas y PageRank (M5) | `scripts/04_red_citas.py` |
| Indexación de la CP | `scripts/06_indice_cp.py` |
| Entrenamiento de B2 y B1 | `scripts/07_entrenar_b2.py`, `scripts/08_entrenar_b1.py` |
| M4 (área temática) | `scripts/05_clasificador_area.py` |

Los nombres exactos están en `codigo/segunda_entrega/scripts/` y en
`codigo/segunda_entrega/plan/PLAN.md`.

## Caché del producto

`app/backend/cache/tfidf_corpus.npz` (117 MB) y `tfidf_vectorizer.joblib` (1 MB)
son un índice derivado para "sentencias similares". No se versionan: se
regeneran en ~30 s al primer arranque del backend.

## Capa 2022-2026

La generó `experimentos/32_integrar_c3.py` a partir del corpus C3 ya preparado
(texto limpiado, entidades M2 y etiquetas M4) y de los HTML crudos. Asigna ids
30000 a 32329, reconstruye el texto crudo y escribe la red de citas combinada.
El backend la carga como capa adicional; si la carpeta no existe, el producto
sigue funcionando con el corpus 1992-2021.
