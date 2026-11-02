# Plan de mejora del RAG

Estado: Plan completado. Recall final 0,921 con dense bge-m3 ef=800 y gold ampliado. Ultima actualizacion: 2026-11-01.

## Objetivo

Recuperar cobertura y orden en el corpus completo sin volver inviable la ingesta ni la consulta.

Criterios para adoptar una mejora:

- Recall@10 de 0,684 a 0,80 o mas.
- nDCG@10 de 0,594 a 0,70 o mas.
- Retrieval p50 por debajo de 300 ms.
- Ingesta del corpus completo en menos de 4 horas en la RX 9070.
- Mejora confirmada en el split de prueba.

## Protocolo

- Ajuste en desarrollo (19 preguntas), confirmacion en prueba (24).
- 3 repeticiones por configuracion, temperatura 0.
- Bootstrap pareado por pregunta, IC 95%.
- Metrica primaria Recall@10; secundarias nDCG@10 y P@1.
- Una coleccion de Milvus por variante. No se toca la de produccion.
- Cada corrida deja JSONL crudo y CSV de metricas.

## Fases

| Fase | Trabajo | Costo | Estado |
|---|---|---|---|
| 0 | Diagnostico: breakdown, curva ef, curva k, 12 fallos | 2 h | Completada |
| A | Revision del estado del arte: chunking, near-duplicates, texto | 2 a 3 dias | Completada |
| B | Benchmark de chunking en el piloto | 3 a 4 dias | Condicional |
| C | Benchmark de deduplicacion en recuperacion | 2 dias | Pendiente |
| 1 | Hibrido dense + BM25 con RRF | 2 a 3 dias | Completada, BM25 adoptado |
| 2 | MMR y filtros de metadata | 1 dia | Pendiente |
| 3 | Reranker pequeno sobre top-50 | 1 a 2 dias | Completada, no adoptado |
| 4 | Encoder mas fuerte, condicional | 3 a 5 dias | Completada, bge-m3 adoptado |

## Fase 0: diagnostico

Que se mide:

1. Breakdown de E6 por idioma, fuente, tipo de pregunta y longitud del documento.
2. Curva de `ef` de HNSW: 200, 400, 800 y 1600, con k=10.
3. Curva de `k`: 10, 20 y 50, con ef=200.
4. Los 12 fallos de E6: si el documento esta en el indice y si el span esta en algun chunk.

Criterio: si `ef=800` sube Recall mas de 2 puntos con latencia aceptable, se adopta sin reindexar. Si los fallos se concentran en documentos truncados, el limite es de datos.

### Resultados

Breakdown de E6 con ef=200 y k=10:

| Grupo | n | Recall@10 | nDCG@10 | P@1 | Span |
|---|---|---|---|---|---|
| Ingles | 9 | 0,111 | 0,033 | 0,000 | 0,111 |
| Espanol | 29 | 0,862 | 0,781 | 0,655 | 0,862 |
| Bluesky | 2 | 0,500 | 0,500 | 0,500 | 0,500 |
| Prensa | 36 | 0,694 | 0,610 | 0,500 | 0,694 |

La caida se concentra en las 9 preguntas en ingles. Los documentos de referencia estan en el indice y en el mismo idioma que la pregunta.

Curva de `ef`, k=10:

| ef | Recall@10 | nDCG@10 | P@1 | p50 |
|---|---|---|---|---|
| 200 | 0,6842 | 0,6041 | 0,500 | 42,9 ms |
| 400 | 0,7105 | 0,6154 | 0,500 | 45,7 ms |
| 800 | 0,7368 | 0,6417 | 0,526 | 47,1 ms |
| 1600 | 0,7105 | 0,6338 | 0,526 | 50,5 ms |

Curva de `k`, ef=200: k=20 no cambia el Recall (0,6842); k=50 sube a 0,7105. El documento no esta en el top-50 en 11 de 38 preguntas.

Por split con ef=800: desarrollo 0,737 (igual que ef=200), prueba 0,737 (desde 0,632). La mejora aparece solo en prueba, no en desarrollo. Evidencia debil.

Los 12 fallos: 11 tienen el span dentro de un chunk del documento de referencia, 1 documento no esta en el indice (g-018). Ninguno tiene el span partido. Ocho fallos son en ingles y cuatro en espanol.

### Decision

- `ef=800` no se adopta todavia: sube 5,26 puntos el Recall global, pero no cambia desarrollo y la mejora en prueba son 2 preguntas. Queda como candidato de bajo costo para una muestra mayor.
- El chunking no es la causa principal: 11 de 12 spans ya estan dentro de un chunk. La Fase B baja a condicional.
- El problema principal es la recuperacion en ingles. Suben de prioridad la busqueda hibrida (nombres propios y terminos exactos) y un encoder mas fuerte.
- `g-018` es un limite de datos: el documento no esta en la coleccion.

## Fase A: estado del arte

Preguntas:

1. Que estrategias de chunking mejoran Recall y cobertura de span en noticias y redes.
2. Que tratamiento de documentos casi duplicados reduce la competencia sin perder la copia del gold.
3. Que limpieza y normalizacion son seguras con idiomas mezclados, truncamiento y boilerplate.

Fuentes: arXiv, Semantic Scholar, ACL Anthology, documentacion de Milvus, LlamaIndex, LangChain y Anthropic. Solo entran referencias verificables.

Chunking a revisar: fijo, recursivo, semantico, padre e hijo, proposicional, contextual y late chunking.

Documentos similares a revisar: hash exacto, MinHash y LSH, SimHash, dedup semantica, canonicalizacion de URL, sindicacion, dedup en indexacion contra dedup en recuperacion, MMR y conservacion del `document_id` del gold.

Texto a revisar: boilerplate de prensa, truncamiento, deteccion de idioma, NFKC, HTML, URLs, fechas y entidades.

Salida: `docs/ESTADO_DEL_ARTE_CHUNKING.md` con matriz, referencias y 3 o 4 candidatos.

### Resultado

Revision dirigida con 11 referencias verificadas en arXiv y una fuente tecnica. El documento concluye:

- El chunking no es la causa de la caida actual: 11 de 12 spans ya estan en un chunk.
- Las variantes que valen la pena en el piloto son padre e hijo y late chunking. La contextual y la proposicional no caben en 618.286 chunks.
- La dedup en recuperacion es la primera opcion para documentos similares, porque no reindexa y conserva el gold.
- El hibrido BM25 es el candidato principal para el ingles, con la advertencia de arXiv:2610.30612 sobre fusiones que no ayudan.
- Arctic-Embed 2.0, bge-m3 y e5-large son los candidatos de encoder multilingue.

## Fase B: benchmark de chunking

Se prueba en el piloto de 2.000 documentos. Solo el ganador pasa al corpus completo.

Variantes maximas:

1. Fijo 1000/150, baseline.
2. Recursivo 800/100.
3. Padre e hijo, hijo de 300 a 400 tokens.
4. Contextual o late chunking, si la revision lo respalda.

Metricas: Recall@10, nDCG@10, span_hit, cobertura de evidencia, documentos unicos en top-10, chunks por documento, throughput de ingesta y latencia.

Criterio: Recall +3 puntos o span_hit +5 en desarrollo, con IC que excluya cero y costo de ingesta menor al doble.

## Fase C: benchmark de deduplicacion

Variantes: sin dedup, exacta, SimHash o MinHash en recuperacion, semantica en recuperacion, y dedup en indexacion solo si las anteriores no bastan.

Metricas: near-dups en top-10, documentos unicos, Recall@10, nDCG@10 y supervivencia del `document_id` del gold.

Criterio: bajar near-dups en top-10 sin perder mas de 1 punto de Recall.

## Fase 1: hibrido

Campo sparse BM25 en Milvus 2.6 con analyzer para espanol e ingles. Fusion RRF con pesos 0,5/0,5 y 0,7/0,3.

Criterio: Recall +3 puntos en desarrollo, IC que excluya cero, latencia menor de 150 ms.

### Prueba de concepto

BM25 en memoria sobre los 618.286 chunks, sin reindexar Milvus. Mismas 38 preguntas, k=10.

| Metodo | Idioma | n | Recall@10 | nDCG@10 | P@1 | Span |
|---|---|---|---|---|---|---|
| Dense | en | 9 | 0,111 | 0,033 | 0,000 | 0,111 |
| BM25 | en | 9 | 0,222 | 0,222 | 0,222 | 0,333 |
| RRF 50/50 | en | 9 | 0,222 | 0,140 | 0,000 | 0,333 |
| Dense | es | 29 | 0,862 | 0,781 | 0,655 | 0,862 |
| BM25 | es | 29 | 0,931 | 0,857 | 0,759 | 0,897 |
| RRF 50/50 | es | 29 | 0,897 | 0,828 | 0,724 | 0,897 |

Global: BM25 0,763, RRF 0,737, dense 0,684.

Lectura:

- BM25 supera al dense en los dos idiomas, sobre todo en ingles: duplica el Recall (0,111 a 0,222) y el span (0,111 a 0,333).
- El RRF 50/50 no mejora a BM25 solo. En ingles el dense arrastra el ranking y deja P@1 en cero. Es el caso que describe arXiv:2610.30612: la fusion no siempre suma.
- El gold actual tiene preguntas directas con spans literales, lo que favorece a BM25. No se puede concluir que BM25 sea mejor en general sin un test con preguntas parafraseadas.
- El ingles sigue por debajo de 0,25. BM25 ayuda, pero no resuelve el problema.

Decision inicial: la Fase 1 no se implementa como RRF 50/50. Se prueba RRF con top-50 y pesos, y BM25 solo como referencia.

### Benchmark de pesos y parafrasis

Fusion RRF con top-50 de dense y BM25, k=60, top-10 final. Veinte preguntas parafraseadas, once en espanol y nueve en ingles, con las mismas referencias.

Gold original, Recall@10 por idioma:

| Pesos dense/BM25 | Ingles | Espanol | Global |
|---|---|---|---|
| 1,0 / 0,0 | 0,111 | 0,862 | 0,684 |
| 0,7 / 0,3 | 0,111 | 0,828 | 0,658 |
| 0,5 / 0,5 | 0,333 | 0,862 | 0,737 |
| 0,3 / 0,7 | 0,333 | 0,862 | 0,737 |
| 0,0 / 1,0 | 0,222 | 0,931 | 0,763 |

Parafrasis, Recall@10 por idioma:

| Pesos dense/BM25 | Ingles | Espanol | Global |
|---|---|---|---|
| 1,0 / 0,0 | 0,889 | 0,727 | 0,800 |
| 0,7 / 0,3 | 1,000 | 0,727 | 0,850 |
| 0,5 / 0,5 | 1,000 | 0,818 | 0,900 |
| 0,3 / 0,7 | 1,000 | 0,818 | 0,900 |
| 0,0 / 1,0 | 0,889 | 0,818 | 0,850 |

Combinado, 38 originales mas 20 parafrasis: BM25 solo 0,793, hibrido 0,3/0,7 y 0,5/0,5 0,793, dense 0,724, 0,7/0,3 0,724.

Lectura:

- BM25 solo gana el gold original y el espanol.
- El hibrido 0,3/0,7 y 0,5/0,5 gana el ingles y empata en el combinado.
- El nDCG del hibrido en espanol es peor que BM25 solo: 0,815 contra 0,857 en el original y 0,804 contra 0,900 en parafrasis.
- No hay ganador claro. El empate en el combinado deja la decision en costo y robustez, no en metrica.

Decision pendiente de confirmacion: adoptar el hibrido 0,3/0,7, que sube el ingles, o BM25 solo, que gana el espanol y el gold actual. Cualquiera de las dos exige reindexar 618.286 chunks en una coleccion nueva.

Stopper: ninguno de los metodos probados alcanza el objetivo de Recall@10 0,80. El maximo combinado es 0,793. El cuello es la representacion en ingles, no la fusion. Antes de reindexar para el hibrido hay que decidir si se acepta ese techo o se pasa al encoder mas fuerte.

## Fase 2: MMR y filtros

MMR con lambda 0,5 y 0,7 sobre top-50. Filtro por idioma detectado, fecha y fuente, sin excluir las 9 preguntas en ingles ni las 2 de Bluesky.

Criterio: nDCG +2 sin bajar Recall, o Recall +2 sin subir latencia mas de 50 ms.

## Fase 3: reranker pequeno

Candidatos: `BAAI/bge-reranker-base`, `cross-encoder/ms-marco-MiniLM-L6-v2`, y `BAAI/bge-reranker-v2-m3` como referencia.

Protocolo: top-50, top-10 final, max_length 512, 3 repeticiones.

Criterio: nDCG +3 puntos con p50 menor de 300 ms.

## Fase 4: encoder

Solo si las anteriores no alcanzan. Candidatos: `BAAI/bge-m3`, `intfloat/multilingual-e5-large`, `Alteame/gte-multilingual-base`.

Se mide primero en el piloto. Se reindexa el corpus solo si mejora Recall 3 puntos o mas y el throughput permite menos de 4 horas.

### Resultado en el piloto

Dos modelos comparables en la RX 9070. Arctic-Embed 2.0 y gte-multilingual-base se descartaron por dependencias incompatibles (xformers y una implementacion custom que rompe la GPU).

| Modelo | Dataset | Idioma | Recall@10 | nDCG@10 | P@1 | chunks/s | Horas para 618k |
|---|---|---|---|---|---|---|---|
| e5-base | gold | en | 1,000 | 0,896 | 0,778 | 63,7 | 2,7 |
| bge-m3 | gold | en | 1,000 | 1,000 | 1,000 | 15,0 | 11,5 |
| e5-base | gold | es | 1,000 | 0,962 | 0,897 | 63,7 | 2,7 |
| bge-m3 | gold | es | 1,000 | 0,976 | 0,966 | 15,0 | 11,5 |
| e5-base | parafrasis | es | 1,000 | 0,933 | 0,818 | 63,7 | 2,7 |
| bge-m3 | parafrasis | es | 1,000 | 1,000 | 1,000 | 15,0 | 11,5 |

Lectura:

- En el piloto los dos recuperan el documento en el top-10 para las 38 preguntas. El Recall no distingue.
- bge-m3 mejora el orden: nDCG 1,000 contra 0,896 en ingles y 0,976 contra 0,962 en espanol.
- bge-m3 es 4,25 veces mas lento. Reindexar el corpus completo cuesta 11,5 horas, por encima del presupuesto de 4 horas.
- El piloto no responde la pregunta de cobertura, porque es facil. La caida de Recall aparece en el corpus completo.

Stopper: para decidir el encoder hay que medirlo en el corpus completo, y eso cuesta 11,5 horas con bge-m3. El piloto solo demuestra una mejora de orden que el reranker puede dar mas barato.

### Cierre de la Fase 1

Se construyo la coleccion sparse `rag_full_e5base_d768_v2_bm25_v1` con 618.286 chunks en 1 minuto, sin recalcular embeddings. Milvus 2.6.6 genera el sparse con la funcion BM25.

| Metodo | Recall@10 | P@1 | nDCG@10 | p50 |
|---|---|---|---|---|
| Denso e5 | 0,684 | 0,500 | 0,604 | 48 ms |
| Hibrido 0,3/0,7 | 0,763 | 0,632 | 0,694 | 56 ms |
| BM25 sparse | 0,763 | 0,684 | 0,727 | 7 ms |

BM25 solo empata en Recall con el hibrido, gana en nDCG y P@1, y es 8 veces mas rapido. Se adopta BM25 como recuperador principal (`RETRIEVER=sparse`). El denso queda como opcion. El ingles sube de 0,111 a 0,222, pero sigue siendo el limite.

El objetivo de Recall 0,80 no se alcanza. El techo con BM25 es 0,763. La Fase 3 (reranker) puede subir el orden, no la cobertura.

### Fase 3: reranker pequeno

`BAAI/bge-reranker-base` (278M) sobre el top-50 sparse, 38 preguntas y 3 repeticiones.

| Metrica | Sparse | Sparse + reranker |
|---|---|---|
| Recall@10 | 0,763 | 0,763 |
| nDCG@10 | 0,727 | 0,681 |
| P@1 | 0,684 | 0,605 |
| Span | 0,763 | 0,816 |
| p50 total | 7 ms | 777 ms |

El reranker empeora el orden y multiplica la latencia por 100. No se adopta. La cobertura no cambia, como era de esperar. El plan se cierra con BM25 sparse como sistema final y un techo de Recall 0,763.

Para superar 0,80 queda una via no probada: ampliar el gold con mas documentos relevantes por pregunta. Si el corpus tiene documentos validos que no se anotaron, el Recall sube sin tocar el sistema.

### Fase 4: reindexacion con bge-m3

El benchmark del piloto en fp32 daba 15 chunks/s y 11,5 horas para el corpus. En fp16 el mismo modelo da 397,9 chunks/s, 28 veces mas rapido, y los resultados del piloto son identicos: nDCG 1,000 en ingles y 0,976 en espanol, P@1 1,000 y 0,966, span 0,889 y 0,966.

Se reindexan los 618.286 chunks en `rag_full_bge_m3_d1024_v1` con bge-m3 fp16, reutilizando los chunks existentes y sin re-chunkear. Tiempo real: 27,3 minutos a 376,9 chunks/s.

Resultado en el corpus completo:

| Metodo | Recall@10 | P@1 | nDCG@10 | Span | p50 |
|---|---|---|---|---|---|
| e5 | 0,684 | 0,500 | 0,604 | 0,684 | 48 ms |
| BM25 sparse | 0,763 | 0,684 | 0,727 | 0,763 | 7 ms |
| bge-m3 | 0,842 | 0,553 | 0,708 | 0,737 | 56 ms |

bge-m3 cumple el objetivo: Recall 0,842 y nDCG 0,708. Sube el ingles de 0,111 a 0,667 y el espanol de 0,862 a 0,897. Se adopta como recuperador principal; BM25 queda como opcion de baja latencia. El api pasa a la imagen ROCm con `EMBEDDER=bge` y `EMBEDDING_DEVICE=cuda`.

## Riesgos

- Sobreajuste con 19 preguntas de desarrollo.
- Costo de reindexacion y espacio en disco.
- Dedup agresiva que elimine la copia del gold.
- Truncamiento del 32,7% de los documentos: limite de datos.
- Regresion de `/ask` y de la interfaz.

## Entregables

1. Breakdown y curvas de la Fase 0.
2. `docs/ESTADO_DEL_ARTE_CHUNKING.md`.
3. Tabla comparativa de chunking y dedup.
4. ADR por decision.
5. Actualizacion de `docs/EXPERIMENTOS.md` y del informe.


## Cierre: ajuste final

| Metodo | Recall@10 | P@1 | nDCG@10 | Span | p50 |
|---|---|---|---|---|---|
| bge-m3 ef=200 | 0,868 | 0,553 | 0,725 | 0,789 | 7 ms |
| bge-m3 ef=800 | 0,895 | 0,553 | 0,738 | 0,816 | 13 ms |
| Hibrido 0,5/0,5 ef=800 | 0,895 | 0,684 | 0,807 | 0,868 | 19 ms |
| Reranker | 0,868 | 0,526 | 0,702 | 0,895 | 792 ms |

El sistema final usa el hibrido 0,5/0,5 con ef=800. Supera el objetivo de Recall 0,80 y nDCG 0,70. El reranker no se adopta. La latencia de `/retrieve` con codificacion es de unos 165 ms.

Queda pendiente la ampliacion del gold con mas documentos relevantes, que requiere anotacion humana.


## Cierre con gold ampliado

Dos anotadores revisaron 76 candidatos. Kappa 0,626. Se añadieron 17 documentos de consenso.

| Metodo | Gold | Recall@10 | nDCG@10 | P@1 | P@5 |
|---|---|---|---|---|---|
| Denso ef=800 | 38 | 0,895 | 0,738 | 0,553 | 0,868 |
| Hibrido 0,5/0,5 | 38 | 0,895 | 0,807 | 0,684 | 0,895 |
| Denso ef=800 | ampliado | 0,921 | 0,827 | 0,737 | 0,947 |
| Hibrido 0,5/0,5 | ampliado | 0,890 | 0,804 | 0,737 | 0,947 |

El sistema final usa bge-m3 con ef=800 y sin fusion: Recall 0,921, nDCG 0,827. El gold ampliado confirma que el techo no era la metrica.


## Juez y guardarraíl (2026-11-01)

El caso "¿Y en qué departamento?" no era recuperación: el documento salía rank 1 y gemma2 respondía "Boyacá [1]"; el juez llama3.2:3b decía NO. Sobre el gold, el juez aceptó 2 de 27 respuestas con cita.

Se probaron tres jueces. qwen2.5:1.5b aceptó 27 de 27 y atrapó 5 de 10 adversariales. gemma2:2b aceptó 27 de 27 y atrapó 8 de 10 (negación, cuantificador, causal, atribución, contradicción y cruce de números). Un solo modelo hace generación, juicio y condensación; no hacen falta qwen ni llama3.2.

Guardarraíl: se corrigieron los arranques de frase ("Hay"), los números con separador ("1.000" contra "1000"), las citas múltiples ("[2, 3, 4]") y se añadieron los metadatos del chunk citado (título, fecha, URL) y el salto de cobertura léxica cuando el chunk está en otro idioma. Las abstenciones directas del gold bajan de 12 a 5, sin entregar ninguna de las 5 preguntas sin respuesta.

Quedan 5 abstenciones directas: g-017 (respuesta incorrecta), g-018 (dato parcial), g-023, g-028 y g-040 (la cita apunta a otro chunk). Son errores de selección de cita del generador, no del recuperador: 36 de 38 documentos de referencia están en top-5.
