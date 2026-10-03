# Auditoria del corpus de la fase v5 (escala 1)

Generado por `experiments/embedding_01_corpus_wikipedia_v5.py` (E-CORP-01). Decision de gobierno: **D-030**.

## 1. Procedencia del corpus

| Campo | Valor |
|---|---|
| Fuente | Wikipedia en espanol (`eswiki`) |
| URL | https://dumps.wikimedia.org/eswiki/latest/eswiki-latest-pages-articles-multistream1.xml-p1p159400.bz2 |
| Recurso | `eswiki-latest-pages-articles-multistream1.xml-p1p159400.bz2` |
| Version | dump del 2026-09-25 |
| Licencia | CC BY-SA 4.0 (atribucion obligatoria) y texto de MediaWiki bajo CC BY-SA/GFDL |
| sha256 del archivo | `3248f107590e6a00e458fdce09616bb22726d78fe3f40106f5a1a7b05d50dabc` |
| Tamano comprimido | 306 MB |

Se descarga **una sola parte** del dump (paginas 1-159400) y no los 5.19 GB completos: la parte basta para 10M, 50M y 100M tokens, y hace la fase reproducible con un solo artefacto verificable por hash.

## 2. Pipeline aplicado

`download -> raw -> clean -> dedup -> normalize -> tokenize -> shards -> manifest`

| Paso | Detalle |
|---|---|
| Lectura | incremental con `XMLPullParser` sobre el bz2; nunca se carga el dump en RAM |
| Filtro | solo `ns=0` (articulos), sin redirecciones ni paginas de titulo con `:` |
| Limpieza | plantillas, tablas, `ref`, HTML, enlaces a archivos y categorias; se conservan las tildes |
| Deduplicacion | `sha1` del texto limpio (0 duplicados descartados) |
| Normalizacion | v5.0-EMB-NORMAL: minusculas, `url`/`usuario`/`hashtag`/`numero`, sin corrector ortografico |
| Filtro de longitud | documentos con menos de 20 tokens descartados (8) |

## 3. Conteos

| Metrica | Valor |
|---|---|
| Articulos leidos | 2.855 |
| Documentos en el corpus | 2.847 |
| Tokens | 10.002.400 |
| Tipos (vocabulario) | 271.209 |
| Tipos por token (T/TR) | 36.88 |
| Shards | 5 (~2M tokens cada uno) |
| Tiempo de construccion | 13.8 s |

Ejemplos de articulos: Andorra, Argentina, Geografía de Andorra, Demografía de Andorra, Artes visuales.

## 4. Auditoria de fuga

Se indexaron **9,737,490** 8-gramas unicos del corpus y se buscaron los de los 2.885 textos de TOXIC (train y test).

- Textos de TOXIC con coincidencia de 8 o mas tokens: **0**
- Veredicto: **CUMPLE**

Ademas, por construccion, el corpus no contiene ningun texto de TOXIC: la variante es `EXTERNAL-ONLY` y la normalizacion se aplica al dump, no al train.

## 5. Limitaciones declaradas

1. El orden del dump se respeta, de modo que la muestra se concentra en articulos antiguos; es una fuente enciclopedica, no tweets.
2. No hay registro web informal (coloquialidad, elongation, hashtags): esa carencia se mide cuantitativamente en `reports/embedding_vocabulary_coverage.md`.
3. Sin corrector ortografico ni reconstruccion de tildes ( EMB-ASCII es una variante distinta y posterior).

## 6. Artefactos

- Shards: data/external/corpora/wikipedia_es/emb_normal/tokens_shard_00.txt, data/external/corpora/wikipedia_es/emb_normal/tokens_shard_01.txt, data/external/corpora/wikipedia_es/emb_normal/tokens_shard_02.txt, data/external/corpora/wikipedia_es/emb_normal/tokens_shard_03.txt, data/external/corpora/wikipedia_es/emb_normal/tokens_shard_04.txt
- Vocabulario: `data/external/corpora/wikipedia_es/emb_normal/vocabulario.txt`
- Manifiesto: `data/external/corpora/corpus_manifest.json` y `reports/corpus_manifest_v5.json`
- Figura: `reports/figuras/fig_41_distribucion_frecuencias_v5.png`
- Figura: `reports/figuras/fig_42_cobertura_grupos_v5.png`
