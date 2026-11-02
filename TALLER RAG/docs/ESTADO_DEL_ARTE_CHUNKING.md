# Estado del arte: chunking, documentos similares y tratamiento de texto

Revision dirigida al corpus de prensa y Bluesky del Taller RAG. Fecha: 2026-11-01.

Alcance: revision de titulos y abstracts en arXiv, mas documentacion tecnica. No se leyeron los papers completos. Las metricas reportadas por cada paper no se copian aqui si el abstract no las da. Antes de implementar un metodo hay que leer la seccion de resultados.

## Preguntas

1. Que estrategias de chunking mejoran Recall y cobertura de span en noticias y redes.
2. Que tratamiento de documentos casi duplicados reduce la competencia en el top-k sin perder la copia que el gold referencia.
3. Que limpieza y normalizacion son seguras con idiomas mezclados, truncamiento y boilerplate.

## Hallazgo del diagnostico

La caida de E6 se concentra en las 9 preguntas en ingles (Recall@10 0,111) frente a las 29 en espanol (0,862). De los 12 fallos, 11 tienen el span dentro de un chunk del documento de referencia y 1 documento no esta en el indice. Ninguno tiene el span partido.

Consecuencia: el chunking no es la causa principal de la caida actual. La revision se mantiene, pero baja de prioridad frente a la representacion y la busqueda lexica.

## Chunking

| Metodo | Referencia | Que propone | Aplicabilidad | Costo | Riesgo |
|---|---|---|---|---|---|
| Fijo 1000/150 | Baseline del proyecto | Cortes por caracteres con solape | Ya implementado | Nulo | Corta evidencia |
| Recursivo | Documentacion de LangChain y LlamaIndex | Cortar por parrafo, oracion y caracter | Alta | Bajo | Similar al fijo si el texto no tiene estructura |
| Semantico por clustering | arXiv:2607.01852 | Agrupar oraciones por similitud y cortar en los cambios | Media | Medio | Sensible al umbral y al encoder |
| Estructural | arXiv:2402.05131 | Chunking por elementos estructurales del documento | Baja: el corpus es texto plano | Medio | Requiere HTML o layout |
| Padre e hijo | arXiv:2507.09935 | Hijo pequeno para buscar, padre para generar | Alta | Medio | Duplica el indice |
| Proposicional | arXiv:2312.06648 | Indexar proposiciones atomicas | Media | Alto: un LLM por chunk | 618k chunks, costo inviable |
| Late chunking | arXiv:2409.04701 | Embeddings de chunks con contexto del documento completo | Alta si se cambia el encoder | Alto: reindexar | Requiere modelo de contexto largo |
| Contextual retrieval | Anthropic, documentacion tecnica | Prefijo de contexto por chunk generado por LLM | Baja a escala | Muy alto | Una llamada por chunk |

Lectura: el chunking fijo no explica los fallos actuales. Las dos variantes que valen la pena en el piloto son padre e hijo y late chunking, y esta ultima solo si se cambia el encoder. El chunking contextual y el proposicional no caben en 618.286 chunks sin un presupuesto grande.

## Documentos similares

| Metodo | Referencia | Que propone | Aplicabilidad | Costo | Riesgo |
|---|---|---|---|---|---|
| Hash exacto | Proyecto | Eliminar duplicados exactos | Ya implementado | Nulo | Ninguno |
| SimHash o MinHash | Practica de IR | Detectar casi duplicados por shingles | Alta, ya se usa para medir | Bajo | Umbral dificil |
| Dedup semantica | arXiv:2303.09540 | Agrupar por embeddings y eliminar redundancia | Media | Medio | Puede borrar la copia del gold |
| Extraccion de contenido principal | arXiv:2111.10864 | Quitar boilerplate antes de comparar | Alta | Bajo | Requiere parser por medio |
| Dedup en recuperacion | Practica de IR | Diversificar el top-k con MMR o clusters | Alta | Bajo | Baja el Recall si es agresiva |

Lectura: la dedup en recuperacion es la primera opcion porque no reindexa y conserva el `document_id` del gold. La dedup en indexacion se prueba solo si la anterior no basta. La extraccion de contenido principal ataca el boilerplate que infla la similitud entre notas sindicadas.

## Tratamiento de texto

- Boilerplate: menus, cookies, "Lea tambien" y sindicacion. La extraccion de contenido principal (arXiv:2111.10864) es la referencia. El proyecto ya filtra paginas de error, pero no boilerplate de navegacion.
- Truncamiento: 32,7% de los documentos. Es un limite de datos. Hay que marcar los documentos truncados y no indexar la parte faltante como si existiera.
- Idioma: 13.140 documentos sin idioma declarado. La deteccion debe ser robusta y no confundir ISO con locales.
- Normalizacion: NFKC, HTML, URLs y espacios, ya aplicada en V2.
- Encoder multilingue: arXiv:2412.04506 reporta calidad competitiva en benchmarks multilingues e ingleses. Es candidato directo para el problema del ingles.

## Candidatos priorizados

1. **Hibrido BM25 + dense con RRF.** Ataca nombres propios y terminos exactos, que es donde el dense falla en ingles. Advertencia: arXiv:2610.30612 muestra que la fusion no siempre ayuda y propone evaluar "no fusionar" como resultado de primera clase. Hay que medir por consulta, no solo el promedio.
2. **Encoder multilingue mas fuerte.** Arctic-Embed 2.0 (arXiv:2412.04506), bge-m3 o e5-large. Medir primero en el piloto y en las 9 preguntas en ingles.
3. **Dedup en recuperacion.** SimHash o MMR sobre el top-50. Barato, sin reindexar, conserva el gold.
4. **Padre e hijo.** Solo si el span se parte. Hoy 11 de 12 spans ya estan en un chunk, asi que no es prioritario.

## Referencias

- Gunther, M. et al. "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models". 2024. arXiv:2409.04701.
- "Reconstructing Context: Evaluating Advanced Chunking Strategies for Retrieval-Augmented Generation". 2025. arXiv:2504.19754.
- Chen, T. et al. "Dense X Retrieval: What Retrieval Granularity Should We Use?". 2023. arXiv:2312.06648.
- "Financial Report Chunking for Effective Retrieval Augmented Generation". 2024. arXiv:2402.05131.
- "Evaluating Chunking Strategies for Retrieval-Augmented Generation on Academic Texts". 2026. arXiv:2607.01852.
- "Enhancing Retrieval Augmented Generation with Hierarchical Text Segmentation Chunking". 2025. arXiv:2507.09935.
- "DAT: Dynamic Alpha Tuning for Hybrid Retrieval in Retrieval-Augmented Generation". 2025. arXiv:2503.23013.
- "The Fusion Tax: A Diagnostic Analysis of When Hybrid Retrieval Hurts". 2026. arXiv:2610.30612.
- "Arctic-Embed 2.0: Multilingual Retrieval Without Compromise". 2024. arXiv:2412.04506.
- Abbas, A. et al. "SemDeDup: Data-efficient learning at web-scale through semantic deduplication". 2023. arXiv:2303.09540.
- "The Impact of Main Content Extraction on Near-Duplicate Detection". 2021. arXiv:2111.10864.
- Anthropic. "Contextual Retrieval". Documentacion tecnica, 2024. https://www.anthropic.com/news/contextual-retrieval
