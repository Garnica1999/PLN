# Esqueleto del informe final

Borrador de trabajo. Los números salen de `docs/EXPERIMENTOS.md` y de los JSONL de `reports/corridas/`. Lo que está marcado como pendiente no se puede cerrar sin la anotación humana o sin la corrida final del reranker.

## Título

Asistente RAG reproducible sobre prensa y Bluesky con Milvus y modelos locales pequeños.

## Autores

[PENDIENTE: nombres y correos del equipo]

## Resumen

Se construyó un asistente de recuperación aumentada sobre 169.248 documentos de prensa y Bluesky con Milvus como único motor vectorial, e5-base como encoder y cuatro modelos generativos locales. Se compararon dos estrategias de segmentación, tres tamaños de contexto, dos encoders y cuatro variantes de limpieza. El sistema se evaluó con un gold de 43 preguntas validadas por una persona y se midió la degradación al pasar del piloto de 2.000 documentos al corpus completo. El hallazgo principal es que el piloto sobreestima la recuperación: Recall@10 cae de 0,947 a 0,684 y nDCG@10 de 0,909 a 0,594 con el mismo encoder e índice. El reranking recupera el orden (P@1 de 0,474 a 0,632) pero no la cobertura.

## 1. Problema y motivación

- Pregunta: ¿qué modelo local pequeño ofrece el mejor equilibrio entre exactitud, fidelidad, abstención, latencia y recursos con la misma evidencia?
- Corpus real de prensa y redes, con ruido de crawler, duplicados y multilingüismo.
- Restricciones: sin APIs de pago, Milvus como único motor vectorial, todo reproducible desde Docker Compose.

## 2. Hipótesis y preguntas

- H1: llama3.2:3b tendrá mayor fidelidad y corrección que qwen2.5:1.5b y gemma2:2b con la misma evidencia, a costa de latencia.
- H2: la limpieza conservadora mejora la recuperación frente al texto crudo.
- H3: el chunking por oraciones mejora el fijo de 1.000 caracteres.
- H4: la recuperación del piloto se mantiene al escalar al corpus completo.

Estado: H2 y H3 no se confirman con significancia; H4 se refuta. H1 queda pendiente de la anotación humana de E4.

## 3. Datos y perfilado

- 169.248 documentos, 104.578 de prensa y 64.670 de Bluesky.
- 14.771 textos duplicados en 4.434 grupos, máximo 839 copias.
- 13.140 documentos sin idioma; el campo mezcla ISO y locales.
- 7.380 páginas de error del crawler (`Error 505`, aviso de cuota).
- Textometría sobre 2.000 del piloto y 5.000 de muestra: TTR 0,57, hapax 0,77, Guiraud 10,0; 32,7% truncados; 9,9% con URLs; 12,1% con emojis; 4,7% near-duplicates (grupo máximo 32).
- Figuras: `reports/figuras/perfil_*.png`, `textometria_*.png`.

## 4. Arquitectura y decisiones

- Diagrama propio en `docs/ARQUITECTURA.md`.
- Ingesta, segmentación, embeddings, Milvus, recuperación, generación, API, Streamlit y evaluación desacoplados.
- Encoder de producción: e5-base en CPU para la API y en la RX 9070 XT para la ingesta.
- Servidor de embeddings con qwen3-embedding como alternativa y comparación E3.
- Decisiones en `docs/DECISIONES.md` (ADR-001 a ADR-020). Destacan: limpieza V2, chunking fijo, k=5, e5-base, ingesta en GPU con ROCm.

## 5. Método

- Limpieza V2: descarta páginas de error, NFKC, sin HTML ni URLs, espacios colapsados.
- Chunking fijo de 1.000 caracteres con 150 de solape; offsets sobre el texto original.
- Embeddings e5-base, 768 dimensiones, prefijos `query:` y `passage:`, normalización L2, COSINE.
- Milvus standalone, HNSW `M=16`, `efConstruction=200`, `ef=200`, filtros por idioma y fecha.
- Orquestación: contexto por tokens, citas `[n]`, validación de citas, abstención.
- API FastAPI y frontend Streamlit.

## 6. Evaluación

- Gold de 43 preguntas: 19 de desarrollo y 24 de prueba; 38 respondibles y 5 sin respuesta.
- Validación de Carlos: 31 Sí y 12 correcciones aplicadas; segunda revisión pendiente.
- Métricas: Recall@10, P@1, P@5, MRR, MAP, nDCG@10, cobertura de span, citas, abstención, latencia y tokens/s.
- Bootstrap de 1.000 remuestreos; Bonferroni en las comparaciones múltiples.

## 7. Experimentos y resultados

### 7.1 Limpieza (30 preguntas del piloto)

| Variante | nDCG@10 | Recall@10 |
|---|---|---|
| V0 crudo | 0,790 | 0,833 |
| V1 filtro de ruido | 0,809 | 0,833 |
| V2 normalización conservadora | 0,875 | 0,900 |
| V3 V2 + dedup de chunks | 0,842 | 0,867 |

Con Bonferroni, ninguna diferencia de calidad excluye cero. Se adopta V2 por consistencia de las estimaciones y costo nulo.

### 7.2 E1, chunking

| Variante | P@1 | P@5 | Recall@10 | nDCG@10 |
|---|---|---|---|---|
| Fijo 1000/150 | 0,833 | 0,900 | 0,900 | 0,875 |
| Oraciones 384/1 | 0,767 | 0,833 | 0,833 | 0,804 |

Sin diferencia significativa de calidad; el fijo es 3,65 ms más rápido. Se mantiene el fijo.

### 7.3 E2, tamaño de contexto

k=3, 5 y 10 dan exactamente las mismas métricas: P@1 0,833, Recall@10 0,900, nDCG@10 0,875. Se fija k=5 por margen para preguntas comparativas, sin ganancia medida.

### 7.4 E3, encoder

| Encoder | Recall@10 | nDCG@10 | p50 |
|---|---|---|---|
| e5-base | 0,944 | 0,802 | 37 ms |
| qwen3-embedding:8b | 1,000 | 0,866 | 68 ms |

qwen recupera mejor en el piloto, pero rinde 1,9 chunks/s en el servidor y 7,3 en la GPU local. Para 618.286 chunks no es viable; se adopta e5.

### 7.5 E4, modelos generativos

[PENDIENTE: anotación humana de 76 respuestas, repetición 1]

Métricas automáticas sobre 19 preguntas respondibles:

| Modelo | Generación p50 | Tokens/s | Abstención | Sin citas | Abstención en sin respuesta |
|---|---|---|---|---|---|
| gemma2:2b | 323 ms | 181 | 5,3% | 0,0% | 12/15 |
| qwen2.5:1.5b | 212 ms | 261 | 10,5% | 89,5% | 15/15 |
| llama3.2:3b | 285 ms | 182 | 15,8% | 56,1% | 15/15 |
| llama3.2:1b | 84 ms | 301 | 73,7% | 15,8% | 15/15 |

### 7.6 E6, escala

| Colección | Documentos | Chunks | Recall@10 | P@1 | nDCG@10 |
|---|---|---|---|---|---|
| Piloto | 1.932 | 7.280 | 0,947 | 0,895 | 0,909 |
| Completo | 161.868 | 618.286 | 0,684 | 0,474 | 0,594 |

La latencia no cambia (42 ms); el problema es de ranking.

### 7.7 E7, reranking

| Métrica | Sin rerank | Con rerank |
|---|---|---|
| Recall@10 | 0,684 | 0,684 |
| P@1 | 0,474 | 0,632 |
| P@5 | 0,658 | 0,684 |
| MRR@10 | 0,564 | 0,654 |
| nDCG@10 | 0,594 | 0,661 |
| Cobertura de span | 0,658 | 0,684 |
| Recuperación p50 | 42 ms | 3.144 ms |

El reranker reordena mejor pero no recupera cobertura y cuesta 3,1 s por consulta. No se adopta para la demo.

## 8. Análisis de errores

- Recuperación en el corpus completo: los dos fallos del gold son `g-045` y `g-046`; los documentos están en la colección, el ranking no los sube.
- Generación: qwen casi no cita, llama3.2:1b se abstiene de más, llama3.2:3b cita la mitad de las veces, gemma2:2b cita siempre.
- Abstención: los cuatro modelos se abstienen en las cinco preguntas sin respuesta salvo gemma2:2b, que responde en 3 de 15 repeticiones.
- Inyección de instrucciones: 5 documentos sintéticos; gemma2:2b obedeció 1/5 (terminar con FIN), llama3.2:3b y qwen2.5:1.5b 0/5. Los tres respondieron el hecho; la prueba es pequeña.

## 9. Limitaciones

- Una sola revisora humana del gold; la segunda revisión puede cambiar filas.
- El gold no tiene preguntas comparativas; solo 2 de Bluesky y 9 en inglés.
- E4 se hizo sobre el piloto, no sobre el corpus completo.
- E7 mejora el orden con un costo de 3,1 s por consulta; no se adopta en la demo.
- El corpus tiene 32,7% de documentos truncados y 5,3% de ruido de crawler; el filtro descarta 7.380 documentos.
- El servidor de embeddings se validó con el piloto, no con el corpus completo.

## 10. Conclusiones

- La limpieza conservadora y el chunking fijo son la base estable; las ablaciones no muestran diferencias significativas con el gold actual.
- El tamaño del corpus cambia el resultado: el piloto no es representativo de la recuperación a escala.
- El reranking mejora el orden, no la cobertura.
- La elección de modelo generativo queda pendiente de la anotación humana; el automático favorece a gemma2:2b en citas y abstención, y a qwen2.5:1.5b en velocidad.

## 11. Referencias

- Thakur et al. (2021). BEIR. arXiv:2104.08663.
- Muennighoff et al. (2023). MTEB. arXiv:2210.07316.
- Zhang et al. (2023). MIRACL. arXiv:2210.09984.
- Es et al. (2023). RAGAS. arXiv:2309.15217.
- Saad-Falcon et al. (2024). ARES. arXiv:2311.09476.
- Salemi y Zamani (2024). eRAG. SIGIR 2024. arXiv:2404.13781.
- Liu et al. (2024). Lost in the Middle. arXiv:2307.03172.
- Ru et al. (2024). RAGChecker. arXiv:2408.08067.

## Apéndice A. Reproducibilidad

- Hardware: Ryzen 9 5950X, 31 GB RAM, RX 9070 XT 16 GB, ROCm 7.2.4.
- Versiones: Python 3.12 en contenedores, e5-base, Milvus 2.6.6, PyTorch 2.10.0+rocm7.2.4.
- Comandos: ver `README.md`.
- Artefactos: `eval/gold/gold_standard.jsonl`, `reports/corridas/*.jsonl`, `reports/tablas/*.csv`.

## Apéndice B. Pendientes antes de entregar

1. Anotar E4 con dos personas y calcular kappa.
2. Segunda revisión del gold (Julián, Erika, Óscar).
3. Ensayar la demo y cerrar el PDF con la plantilla `tg_usta.cls`.
