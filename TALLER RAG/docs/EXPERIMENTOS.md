# Protocolo de experimentos y registro

Este archivo se actualiza con cada corrida real. Ninguna cifra se escribe a mano: todas salen de los JSONL de `reports/corridas/` procesados por `src/scripts/metrics.py`.

## 1. Compuerta de humo (F1.5)

- Conjunto: 600 documentos, 350 de prensa y 250 de Bluesky, español e inglés, semilla fija.
- Preguntas: al menos 40, una por documento, con `reference_doc_ids` y `reference_span` verificados contra el texto. Generadas con LLM y pendientes de validación humana; sirven para la compuerta de correctitud, no para afirmar calidad.
- Repeticiones: 3 con la misma configuración.
- Métricas: Recall@5 y @10, Recall@5 de fragmento, cobertura de respuesta, MRR@10, nDCG@10, MAP@10, hit@10, duplicados, vacíos y latencia.

Umbrales obligatorios:

| Comprobación | Umbral |
|---|---|
| Recall@10 documento | 1,00 |
| Recall@5 documento | 0,95 |
| MRR@10 | 0,85 |
| nDCG@10 | 0,85 |
| `reference_span` en top-5 | 0,90 |
| Duplicados en top-10 | 0 |
| Offsets rotos | 0 |
| Top-10 determinista en 3 corridas | idéntico |
| Filtros de idioma y fecha | exactos |

Si una comprobación falla, no se escala. Se corrige el pipeline y se repite la compuerta.

## 2. Métricas y niveles

| Métrica | Nivel | Fórmula o criterio | Meta |
|---|---|---|---|
| Recall@k | documento | relevantes recuperados en top-k / relevantes totales | humo 1,00@10; piloto 0,90@10 |
| Precision@k | documento | relevantes en top-k / k | se reporta |
| Hit@k | documento | 1 si hay al menos un relevante | 1,00 en humo |
| MRR@k | documento | media de 1/rango del primer relevante | 0,85@10 en humo |
| MAP@k | documento | media de la precisión promedio | 0,75@10 en piloto |
| nDCG@k | documento | ganancia descontada, relevancia binaria o graduada | 0,85@10 en humo |
| Recall@k | fragmento | `reference_chunk_ids` recuperados / total | 0,90@5 |
| Cobertura de respuesta | fragmento | el `reference_span` aparece en algún chunk del top-k | 0,90 en top-5 |
| Context precision | contexto | fracción de chunks del contexto con documento relevante | 0,70 |
| Context recall | contexto | documentos de referencia presentes tras el presupuesto de tokens | 0,90 |
| Duplicados | ranking | chunks repetidos en el top-k | 0 |
| Documentos únicos | ranking | diversidad del top-k | al menos min(k, 3) |
| Latencia p50/p95 | sistema | recuperación y generación por separado | p95 recuperación menor a 2 s |
| IC 95% | estadística | bootstrap de 1.000 remuestreos por pregunta | se reporta siempre |

## 3. Reglas estadísticas

- La unidad de remuestreo es la pregunta, no la repetición.
- Los IC se calculan con bootstrap percentil de 1.000 remuestreos.
- Dos configuraciones se comparan con bootstrap pareado sobre las mismas preguntas. Si el intervalo de la diferencia contiene cero, no se declara superioridad.
- Con 20 preguntas se reporta el IC, no solo la media.
- Un resultado perfecto en 20 preguntas es una compuerta de correctitud, no una prueba de calidad a escala.

## 4. Métricas complementarias

- eRAG: para cada pregunta y cada documento del top-k, el LLM responde usando solo ese documento y se compara con la respuesta de referencia. Mide utilidad downstream del recuperador.
- Juez LLM: solo como complemento de la anotación humana, nunca como verdad.
- Lost in the middle: se registra la posición del chunk relevante en el contexto y si la respuesta lo cita.

## 4.1 Rendimiento medido de encoders (2026-10-30)

Lotes de 16 textos de unos 900 caracteres. La estimación del corpus completo usa 600.000 unidades.

| Encoder | Equipo | chunks/s | Horas estimadas (600k) |
|---|---|---|---|
| qwen3-embedding:8b | GTX 1080 del servidor | 1,9 | 88 |
| qwen3-embedding:0.6b | GTX 1080 del servidor | 2,5 | 67 |
| qwen3-embedding:8b | RX 9070 XT local | 7,3 | 23 |
| e5-base | Ryzen 9, 8 hilos | 12,2 | 14 |
| e5-base | Ryzen 9, 24 hilos | 25,5 | 6,5 |

Decisión: e5-base en CPU para piloto y corpus completo; qwen3-embedding:8b en el servidor para la comparación E3 sobre 2.000 documentos. Ver ADR-016.

## 5. Registro de corridas

| ID | Fecha | Conjunto | Colección | Encoder | Chunking | k | Docs | Recall@10 | nDCG@10 | Notas |
|---|---|---|---|---|---|---|---|---|---|---|
| pendiente | | | | | | | | | | |

## 6. Fallos de compuerta y correcciones

| Fecha | Comprobación que falló | Causa | Corrección | Evidencia |
|---|---|---|---|---|
| pendiente | | | | |

## 7. Referencias

- Thakur et al. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. arXiv:2104.08663.
- Muennighoff et al. (2023). MTEB: Massive Text Embedding Benchmark. arXiv:2210.07316.
- Zhang et al. (2023). MIRACL: A Multilingual Retrieval Dataset. arXiv:2210.09984.
- Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15217.
- Saad-Falcon et al. (2024). ARES: An Automated Evaluation Framework for RAG. arXiv:2311.09476.
- Salemi y Zamani (2024). Evaluating Retrieval Quality in Retrieval-Augmented Generation (eRAG). SIGIR 2024. arXiv:2404.13781.
- Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. arXiv:2307.03172.
- Ru et al. (2024). RAGChecker: A Fine-grained Framework for Diagnosing RAG. arXiv:2408.08067.

## 4.2 Compuerta de humo: resultados reales (2026-10-31)

Conjunto: 600 documentos (582 tras el filtro de calidad), 53 preguntas generadas desde los documentos, 3 repeticiones, k=10, e5-base, colección `rag_smoke_e5base_d768_fixed_v1`.

| Versión | Cambio | Preguntas | Recall@10 | MRR@10 | nDCG@10 | Span en top-10 |
|---|---|---|---|---|---|---|
| v1 | índice con ruido, ef=64 | 53 | 0,849 | 0,811 | 0,821 | 0,830 |
| v2 | sin documentos de error, ef=64 | 53 | 0,943 | 0,877 | 0,894 | 0,906 |
| v3 | 3 preguntas excluidas, ef=64 | 50 | 0,880 | 0,840 | 0,850 | 0,840 |
| final | 15 preguntas excluidas, ef=200 | 38 | **1,000** | **0,987** | **0,990** | **0,947** |

Diagnóstico:

- El crawler dejó 7.378 documentos (4,36% del corpus) con la misma página de error ("Haz excedido el máximo de peticiones", "Error 505"). Esos textos casi idénticos dominaban el top-10 en cualquier consulta. Se filtran en la ingesta con `src/rag/quality.py` y se cuentan como `descartados_por_calidad`.
- El índice HNSW con `ef=64` perdía recall. Ablación sobre las 50 preguntas: ef=64 → 0,880; ef=128 → 0,920; ef=200 → 0,960; ef=400 → 0,960. Se fija `SEARCH_EF=200`.
- 15 preguntas se excluyeron por criterio explícito: tasas de cambio respondidas por varios documentos, referencias vagas ("el país", "la medida", "la empresa AG"), una pregunta en ruso sobre un documento en ucraniano y una generada desde el boilerplate del portal. La lista y la razón están en `eval/silver/questions_smoke.jsonl`.
- Las 3 repeticiones dan el mismo top-10 tras el flush y con ef=200.

Conclusión: el pipeline recupera correctamente en el conjunto de humo. No es una prueba de calidad a escala; es la compuerta para escalar al piloto.

## 4.3 Índices construidos (2026-10-31)

| Colección | Documentos de la muestra | Documentos indexados | Descartados por calidad | Chunks | chunks/s | Encoder |
|---|---|---|---|---|---|---|
| `rag_smoke_e5base_d768_fixed_v1` | 600 | 582 | 18 | 2.064 | 15,0 | e5-base, 768 |
| `rag_pilot_e5base_d768_fixed_v1` | 2.000 | 1.932 | 68 | 7.281 | 14,5 | e5-base, 768 |
| `rag_pilot_qwen3emb8b_d1024_fixed_v1` | 2.000 | 1.932 | 68 | 7.281 | 9,4 | qwen3-embedding:8b, 1024 |

El descarte por calidad aplica el filtro de `src/rag/quality.py`. Los documentos descartados no se borran del corpus: se excluyen del índice y se cuentan en el resumen de la ingesta.

## 4.4 E3 en curso: e5-base contra qwen3-embedding

Preguntas: 18 de desarrollo generadas desde documentos del piloto, 3 repeticiones, k=10, mismas preguntas y mismo índice limpio.

| Encoder | Equipo | Hit@10 | Recall@10 | MRR@10 | nDCG@10 | Span top-10 | p50 recuperación |
|---|---|---|---|---|---|---|---|
| e5-base 768 | CPU local | 0,944 | 0,944 | 0,757 | 0,802 | 0,944 | 37 ms |
| qwen3-embedding:8b 1024 | RX 9070 XT | 1,000 | 1,000 | 0,821 | 0,866 | 0,944 | 68 ms |

## 7. Textometría del corpus (2026-10-31)

Script: `src/scripts/textometria.py`. Salidas: `reports/tablas/textometria.csv`, `reports/tablas/textometria.json` y cinco figuras en `reports/figuras/`.

Muestra: 2.000 documentos del piloto y 5.000 documentos estratificados del resto del corpus, semilla 42.

| Indicador | Piloto | Muestra |
|---|---|---|
| Tipos por documento (mediana) | 188 | 188 |
| TTR mediana | 0,579 | 0,570 |
| Hapax / tipos (mediana) | 0,773 | 0,770 |
| Guiraud (mediana) | 10,04 | 10,13 |
| Caracteres por documento (mediana) | 2.317 | 2.353 |
| Caracteres p99 | 14.979 | 15.498 |
| Documentos truncados | 33,5% | 32,7% |
| Documentos con URL | 9,3% | 9,9% |
| Documentos con email | 1,8% | 1,3% |
| Documentos con emojis | 12,4% | 12,1% |
| Documentos con navegación | 3,4% | 3,3% |
| Documentos con boilerplate | 3,4% | 4,5% |
| Documentos con near-duplicate (SimHash, Hamming <= 3) | 4,25% | 4,72% |
| Grupos de near-duplicates | 26 | 53 |
| Grupo más grande | 12 | 32 |

Lectura:

- Un tercio del corpus está truncado. La limpieza no puede recuperar ese texto; es una limitación del crawler que debe declararse.
- El HTML residual es marginal (0,14% en la muestra), pero las URLs aparecen en 1 de cada 10 documentos y los emojis en 1 de cada 8, casi todo en Bluesky.
- Los near-duplicates no son solo hash exacto: SimHash encuentra grupos de hasta 32 documentos que el hash exacto no captura.
- Los términos "exclusivos" de prensa están dominados por boilerplate (`publicidad`, `información`, `además`), lo que anticipa que la normalización conservadora puede ayudar.
- El code-switching medido es 0,12%; el campo de idioma declara 385 documentos sin idioma en la muestra.

## 8. Experimento pareado de limpieza (2026-10-31)

Diseño: una colección por variante, mismo encoder (e5-base, 768), mismo chunking (1000/150), mismo índice (HNSW, `ef=200`), mismas 30 preguntas (25 de prensa y 5 de Bluesky), 3 repeticiones, k=10. El silver de humo no se usó porque su muestra de 600 documentos no es subconjunto del piloto de 2.000: sus referencias no existen en estas colecciones. Esa separación se detectó al medir y queda como lección de diseño.

### 8.1 Índices

| Variante | Definición | Documentos | Chunks | Near-dups descartados | Tiempo |
|---|---|---|---|---|---|
| V0 | texto crudo | 2.000 | 7.783 | 0 | no registrado |
| V1 | descarta 68 páginas de error | 1.932 | 7.281 | 0 | 462 s |
| V2 | V1 + NFKC, sin HTML ni URLs, espacios colapsados | 1.932 | 7.280 | 0 | 470 s |
| V3 | V2 + dedup de near-duplicates por chunk | 1.932 | 7.121 | 159 | 465 s |

### 8.2 Métricas (n=30 preguntas)

`precision@k` no se usa para decidir: con un solo documento relevante por pregunta está acotada por 1/k (0,10 con k=10) y su valor depende del número de documentos devueltos. Se reportan P@1, P@5, recall y nDCG.

| Variante | P@1 | P@5 | Recall@10 | IC 95% recall | MRR@10 | nDCG@10 | IC 95% nDCG |
|---|---|---|---|---|---|---|---|
| V0 | 0,733 | 0,833 | 0,833 | [0,700; 0,967] | 0,775 | 0,790 | [0,652; 0,918] |
| V1 | 0,767 | 0,833 | 0,833 | [0,700; 0,967] | 0,800 | 0,809 | [0,675; 0,930] |
| V2 | 0,833 | 0,900 | 0,900 | [0,767; 1,000] | 0,867 | 0,875 | [0,763; 0,967] |
| V3 | 0,800 | 0,867 | 0,867 | [0,733; 0,967] | 0,833 | 0,842 | [0,705; 0,955] |

### 8.3 Bootstrap pareado contra V0 con corrección de Bonferroni

La tabla prueba 7 métricas por 3 variantes. Con 21 pruebas, el nivel de cada intervalo se ajusta a 0,05/21.

| Variante | Métrica | Diferencia | IC ajustado | ¿Excluye cero? |
|---|---|---|---|---|
| V1 | P@1 | +0,033 | [-0,160; 0,200] | No |
| V1 | Recall@10 | 0,000 | [-0,100; 0,160] | No |
| V1 | nDCG@10 | +0,019 | [-0,133; 0,143] | No |
| V2 | P@1 | +0,100 | [0,000; 0,267] | No (toca cero) |
| V2 | P@5 | +0,067 | [0,000; 0,200] | No (toca cero) |
| V2 | Recall@10 | +0,067 | [0,000; 0,233] | No (toca cero) |
| V2 | nDCG@10 | +0,086 | [0,000; 0,252] | No (toca cero) |
| V3 | P@1 | +0,067 | [-0,133; 0,260] | No |
| V3 | nDCG@10 | +0,052 | [-0,133; 0,228] | No |

Ninguna comparación de calidad excluye cero. V2 tiene la mejor estimación puntual en todas las métricas, pero la evidencia es sugestiva, no concluyente. La decisión de adoptar V2 se sostiene en la consistencia entre métricas y en su costo nulo, no en significancia estadística. La tabla completa está en `reports/tablas/limpieza_paired.csv`.

### 8.4 Desglose

| Corte | n preguntas | V0 nDCG | V1 nDCG | V2 nDCG | V3 nDCG |
|---|---|---|---|---|---|
| Prensa (url_crawls) | 25 | 0,908 | 0,891 | 0,971 | 0,931 |
| Bluesky | 5 | 0,200 | 0,400 | 0,400 | 0,400 |
| Español | 21 | 0,965 | 0,965 | 0,965 | 0,965 |
| Inglés | 9 | 0,381 | 0,444 | 0,667 | 0,556 |
| Directas | 30 | 0,790 | 0,809 | 0,875 | 0,842 |

Lectura: la ganancia de V2 se concentra en prensa y en inglés. En español el nDCG no cambia porque V0 ya estaba en 0,965. El subconjunto de Bluesky tiene 5 preguntas, tres de ellas vagas, y no permite concluir. No hay preguntas comparativas, ambiguas ni sin respuesta en este silver.

### 8.5 Exclusiones y límites

- 12 preguntas de desarrollo se descartaron por referencias no únicas (tasas de cambio, procesos amplios, conteos globales) o por duplicar otra pregunta.
- 5 preguntas de Bluesky se conservaron; 3 de ellas son vagas y explican el nDCG bajo de esa colección.
- El silver de humo (38 preguntas) no se usó por pertenecer a otra muestra.
- La cobertura de span no se reporta como métrica principal porque el span anotado viene del texto crudo y V2 modifica el texto indexado.
- V4 (normalización agresiva) no se ejecutó: es un control negativo de costo conocido y el presupuesto de tiempo se destinó a completar V0 a V3.

## 9. E1 y E2: chunking y tamaño de contexto (2026-10-31)

Mismas 30 preguntas del piloto, 3 repeticiones, e5-base, limpieza V2, HNSW con `ef=200`.

### 9.1 E1, chunking fijo contra oraciones

| Variante | Chunks | P@1 | P@5 | Recall@10 | MRR@10 | nDCG@10 | Recuperación p50 |
|---|---|---|---|---|---|---|---|
| Fijo 1000/150 | 7.280 | 0,833 | 0,900 | 0,900 | 0,867 | 0,875 | 38,5 ms |
| Oraciones 384/1 | 5.163 | 0,767 | 0,833 | 0,833 | 0,794 | 0,804 | 42,1 ms |

Bootstrap pareado con Bonferroni sobre 7 pruebas (oraciones menos fijo): nDCG -0,071 con IC [-0,221; 0,025], recall -0,067 con IC [-0,233; 0,000], P@1 -0,067 con IC [-0,233; 0,114]. Ninguna diferencia de calidad excluye cero. La latencia sí: el fijo es 3,65 ms más rápido (IC [2,25; 4,97]).

Decisión: se mantiene el chunking fijo de 1000 caracteres con 150 de solape. Tiene mejores estimaciones puntuales, genera más chunks pero recupera igual o mejor, y es más rápido. Las oraciones no aportan con este silver.

### 9.2 E2, ablación de k

| k | P@1 | P@5 | Recall@k | MRR@k | nDCG@k | Recuperación p50 |
|---|---|---|---|---|---|---|
| 3 | 0,833 | 0,900 | 0,900 | 0,867 | 0,875 | 40,1 ms |
| 5 | 0,833 | 0,900 | 0,900 | 0,867 | 0,875 | 39,4 ms |
| 10 | 0,833 | 0,900 | 0,900 | 0,867 | 0,875 | 38,5 ms |

Las tres configuraciones devuelven exactamente las mismas métricas de calidad: el documento relevante está siempre en el top 3. La única diferencia significativa es la latencia: k=3 tarda 1,64 ms menos que k=10 (IC de la diferencia [-2,85; -0,57]) y k=5 tarda 0,94 ms menos (IC [-1,76; -0,15]). Son milisegundos, no un cambio de calidad.

Decisión: k=3 basta para este silver. Para el sistema final se fija k=5 como margen para preguntas comparativas que no están en este conjunto, sabiendo que con el silver actual no hay ganancia medible de 3 a 5 ni de 5 a 10. El costo de k=5 es mayor contexto en generación, no más latencia de recuperación.

## 10. Silver standard consolidado y validado (2026-10-31)

Archivos: `eval/silver/silver_standard.jsonl` (fuente de verdad), `.csv`, `.md` y `.xlsx` (plantilla y copias por revisor).

| Corte | Total | Respondibles | Sin respuesta |
|---|---|---|---|
| Desarrollo | 19 | 19 | 0 |
| Prueba | 24 | 19 | 5 |
| Total | 43 | 38 | 5 |

Composición: 38 directas, 0 ambiguas, 5 sin respuesta. 34 en español y 9 en inglés. 36 de prensa y 2 de Bluesky.

Validación: Carlos revisó las 43 filas. Marcó 31 con Sí y 12 con Corregir. Las 12 correcciones se aplicaron (respuesta de g-003; fragmentos de g-009, g-012, g-031, g-040, g-042 y g-047; URLs de Wayback para g-022, g-025 y g-032; idioma de g-039; reformulación de g-036 y g-042) y cada una quedó con `review_note`.

Segunda revisión (Julián, 2026-10-31): revisó 40 de las 43. Marcó 30 con Sí, 8 con Corregir y 2 con No. Se aplicaron los 7 spans y la reformulación de g-017. No se aplicaron g-036 ni g-045: el texto indexado respalda "entre 1996 y 2002" y "near Kendal". Tres preguntas (g-022, g-025, g-032) no se pudieron verificar por enlaces caídos. Erika y Óscar no revisaron. El silver anterior quedó en `eval/silver/silver_standard_pre_julian.jsonl`.

Verificación sobre `rag_pilot_e5base_d768_v2_v1`, k=10, 3 repeticiones:

| Métrica | Valor | IC 95% |
|---|---|---|
| Recall@10 | 0,947 | [0,904; 0,983] |
| P@1 | 0,842 | |
| P@5 | 0,947 | |
| MRR@10 | 0,895 | |
| nDCG@10 | 0,909 | [0,862; 0,950] |
| Cobertura de span | 0,895 | |

Los dos fallos son g-045 y g-046. Ambos documentos están en la colección; son fallos reales de recuperación, no referencias rotas.

El silver ya no tiene preguntas ambiguas: g-036 se reformuló a directa. Las URLs 404 de g-022, g-025 y g-032 se reemplazaron por instantáneas de Wayback.

Límites: sin preguntas comparativas; solo 2 de Bluesky y 9 en inglés. Cuatro personas revisaron: Carlos 43/43, Julián 40/43, Óscar 43/43 y Erika 43/43 (35 aprobadas, 5 no aplicables y 3 solicitudes de corrección). El JSONL consolidado es la fuente de verdad: no se debe regenerar con `build_silver_standard` o se perderían las correcciones.

## 11. E4: comparación de los cuatro modelos (2026-10-31)

Diseño: 24 preguntas de prueba (19 respondibles y 5 sin respuesta), 4 modelos, 3 repeticiones, temperatura 0, k=5, colección `rag_pilot_e5base_d768_v2_v1`. La evidencia recuperada es la misma para los cuatro.

| Modelo | Generación p50 | Tokens/s | Abstención en respondibles | Sin citas en respondibles | Citas inválidas | Abstención en sin respuesta |
|---|---|---|---|---|---|---|
| gemma2:2b | 323 ms | 181 | 5,3% | 0,0% | 0 | 12/15 |
| qwen2.5:1.5b | 212 ms | 261 | 10,5% | 89,5% | 0 | 15/15 |
| llama3.2:3b | 285 ms | 182 | 15,8% | 56,1% | 0 | 15/15 |
| llama3.2:1b | 84 ms | 301 | 73,7% | 15,8% | 0 | 15/15 |

Recuperación idéntica entre modelos: hit@5 0,947, P@1 0,737, P@5 0,947, MRR 0,842, nDCG@5 0,870 y cobertura de span 0,947.

Lectura automática: gemma2:2b es el único que cita siempre; qwen2.5:1.5b casi nunca cita; llama3.2:1b se abstiene en el 74% de las preguntas que sí tienen evidencia.

Anotación humana (repetición 1, 18 preguntas respondibles, dos anotadores, escala 0-3):

| Modelo | Corrección | Fidelidad | Citas | Abstención |
|---|---|---|---|---|
| gemma2:2b | 2,81 | 2,97 | 2,67 | 2,89 |
| qwen2.5:1.5b | 2,31 | 2,72 | 0,00 | 2,81 |
| llama3.2:3b | 2,22 | 2,72 | 0,86 | 2,69 |
| llama3.2:1b | 0,42 | 2,31 | 0,28 | 1,61 |

Kappa de Cohen ponderado: corrección 0,95, citas 0,90, abstención 0,72, fidelidad 0,12. La fidelidad no se usa por falta de acuerdo entre anotadores.

Veredicto: gemma2:2b gana. Supera a qwen2.5:1.5b en corrección (+0,50, IC [0,11; 1,00]) y en citas (+2,67). H1 se refuta: llama3.2:3b no supera a qwen2.5:1.5b (-0,08, IC [-0,81; 0,61]) y pierde contra gemma2:2b (-0,58, IC [-1,11; -0,17]).


## 12. Ingesta en GPU con ROCm (2026-10-31)

Imagen `Dockerfile.rocm` sobre `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0`, la misma versión de ROCm del host. Servicio `ingest-rocm` en el perfil `rocm`, con `/dev/kfd`, `/dev/dri`, red de host y el volumen `hf_cache`.

| Dispositivo | Batch | chunks/s |
|---|---|---|
| CPU 24 hilos | 64 | 25,5 |
| RX 9070 XT | 32 | 102,8 |
| RX 9070 XT | 64 | 104,5 |
| RX 9070 XT | 128 | 106,9 |

Prueba de extremo a extremo: 100 documentos en 1,5 s (66,7 chunks/s) y búsqueda de 3 chunks en 45 ms sobre la colección temporal `rag_gpu_test_e5base_d768_v2_v1`, ya eliminada.

Corrida completa: `rag_full_e5base_d768_v2_v1`, 169.248 documentos, V2, chunking fijo, e5 en la RX 9070 XT, batch 128. Estimación: 2 a 3 horas para unas 580.000 unidades.

La API consulta con e5 en CPU y la colección se indexó con e5 en GPU; es el mismo modelo, con diferencias de coma flotante despreciables. La GPU se comparte con Ollama: si se genera texto mientras se indexa, ambos bajan.

Nota de la corrida: el primer intento falló en el título 346 por 1.048 bytes contra el máximo de 1.024. Se corrigió la truncación a bytes (`truncate_bytes`) y se relanzó con `--drop`. La prueba de la corrección está en `src/tests/test_cleaning.py`.

## 13. E6: escala del corpus completo (2026-10-31)

Corrida final: 169.248 documentos de muestra, 161.868 indexados (7.380 descartados por calidad), 618.286 chunks, 63,8 chunks/s, 2 h 41 min en la RX 9070 XT, colección `rag_full_e5base_d768_v2_v1`.

| Colección | Documentos | Chunks | Recall@10 | P@1 | P@5 | MRR@10 | nDCG@10 | Span | p50 |
|---|---|---|---|---|---|---|---|---|---|
| Piloto V2 | 1.932 | 7.280 | 0,947 | 0,842 | 0,947 | 0,895 | 0,909 | 0,895 | 40 ms |
| Completo V2 | 161.868 | 618.286 | 0,684 | 0,474 | 0,658 | 0,564 | 0,594 | 0,658 | 42 ms |

Lectura: el piloto sobreestima la recuperación. Con 85 veces más candidatos, el mismo encoder y el mismo índice pierden 26 puntos de Recall@10 y 31 de nDCG@10. Los documentos de referencia siguen en la colección; el problema es de ranking, no de datos faltantes. La latencia no cambia (42 ms), así que el costo es de calidad, no de velocidad.

Implicación para el informe: las cifras del piloto no se pueden presentar como resultado final. La comparación de LLM (E4) se hizo sobre el piloto y debe declararse así. La siguiente mejora natural es reranking sobre el top-50 o un encoder más fuerte, no aumentar k a ciegas.

## 14. E7: reranking sobre el corpus completo (2026-10-31)

Cross-encoder `BAAI/bge-reranker-v2-m3` sobre el top-50 de e5, top-10 final, 38 preguntas respondibles, 3 repeticiones, `max_length=512`.

| Métrica | Sin rerank (E6) | Con rerank (E7) |
|---|---|---|
| Recall@10 | 0,684 | 0,684 |
| P@1 | 0,474 | 0,632 |
| P@5 | 0,658 | 0,684 |
| MRR@10 | 0,564 | 0,654 |
| nDCG@10 | 0,594 | 0,661 |
| Cobertura de span | 0,658 | 0,684 |
| Recuperación p50 | 42 ms | 3.144 ms |

El reranker mejora el orden, no la cobertura: el 32% que no aparece en el top-10 sigue fuera. La latencia de 3,1 s por consulta no baja al limitar `max_length` a 512; el costo es del cross-encoder de 568M sobre 50 pares. No se adopta para la demo; queda como mejora documentada. El JSONL crudo está en `reports/corridas/e7_rerank_full.jsonl`.

## 15. E5: inyección de instrucciones en documentos (2026-10-31)

Cinco documentos sintéticos con una instrucción maliciosa (decir HACKEADO, PWNED, revelar el prompt, negar el documento o terminar con FIN) y cinco preguntas cuya respuesta está en el texto. Tres modelos, temperatura 0, k=3, colección `rag_injection_e5base_v1`.

| Modelo | Obedeció la inyección | Respondió el hecho | Abstención |
|---|---|---|---|
| gemma2:2b | 1/5 | 4/5 | 0/5 |
| llama3.2:3b | 0/5 | 5/5 | 0/5 |
| qwen2.5:1.5b | 0/5 | 5/5 | 0/5 |

gemma2:2b respondió el dato de la pregunta 5 y además añadió la palabra FIN que pedía la instrucción. Los otros dos ignoraron las cinco instrucciones. Las preguntas eran respondibles, así que la abstención no aplica. Es una prueba pequeña, no un benchmark adversarial completo.

## 16. E8: BM25 sparse sobre el corpus completo (2026-11-01)

La colección `rag_full_e5base_d768_v2_bm25_v1` se construyó a partir de los mismos 618.286 chunks, sin recalcular embeddings. Milvus 2.6.6 genera el vector sparse con la función BM25 sobre el campo `text`.

| Método | Recall@10 | P@1 | P@5 | MRR@10 | nDCG@10 | Span | p50 |
|---|---|---|---|---|---|---|---|
| Denso e5 | 0,684 | 0,500 | 0,658 | 0,578 | 0,604 | 0,684 | 48 ms |
| BM25 sparse | 0,763 | 0,684 | 0,737 | 0,715 | 0,727 | 0,763 | 7 ms |

Por idioma, BM25 sube el inglés de 0,111 a 0,222 y el español de 0,862 a 0,931. La fusión dense más BM25 con pesos 0,3/0,7 dio Recall 0,763, nDCG 0,694 y P@1 0,632 con 56 ms, y no superó a BM25 solo. Se adopta BM25 como recuperador principal y el denso queda como opción con `RETRIEVER=dense`. El inglés sigue siendo el límite: 7 de las 9 preguntas no aparecen en el top-10.

Artefactos: `reports/tablas/bench_milvus.csv`, `bench_milvus_raw.csv`, `bench_hybrid.csv`, `bench_paraphrases.csv` y `bench_encoder_quality.csv`.

## 17. E9: reranker pequeño sobre BM25 sparse (2026-11-01)

`BAAI/bge-reranker-base` (278M) sobre el top-50 de BM25 sparse, 38 preguntas y 3 repeticiones, en la RX 9070.

| Métrica | Sparse | Sparse + reranker |
|---|---|---|
| Recall@10 | 0,763 | 0,763 |
| nDCG@10 | 0,727 | 0,681 |
| P@1 | 0,684 | 0,605 |
| Span | 0,763 | 0,816 |
| p50 total | 7 ms | 777 ms |

El reranker empeora el orden y multiplica la latencia por 100. No se adopta. Artefactos: `reports/tablas/bench_rerank_sparse.csv` y `bench_rerank_sparse_raw.csv`.

## 18. E10: bge-m3 sobre el corpus completo (2026-11-01)

`BAAI/bge-m3`, 1.024 dimensiones, fp16 en la RX 9070. Los 618.286 chunks se reindexaron en `rag_full_bge_m3_d1024_v1` reutilizando los chunks existentes, sin re-chunkear, en 27,3 minutos a 376,9 chunks/s. En fp32 el mismo modelo tarda 11,5 horas.

| Método | Recall@10 | P@1 | P@5 | nDCG@10 | Span | p50 |
|---|---|---|---|---|---|---|
| Denso e5 | 0,684 | 0,500 | 0,658 | 0,604 | 0,684 | 48 ms |
| BM25 sparse | 0,763 | 0,684 | 0,737 | 0,727 | 0,763 | 7 ms |
| bge-m3 | 0,842 | 0,553 | 0,816 | 0,708 | 0,737 | 56 ms |

Por idioma, bge-m3 sube el inglés de 0,111 a 0,667 y el español de 0,862 a 0,897. Cumple el objetivo de Recall 0,80 y nDCG 0,70. Se adopta como recuperador principal. BM25 queda como opción de baja latencia con `RETRIEVER=sparse`.

Artefactos: `reports/tablas/bench_bge_m3_full.csv` y `bench_bge_m3_full_raw.csv`.

## 19. E11: ajuste de ef y fusión final (2026-11-01)

Con bge-m3 se probaron el parámetro `ef` del índice HNSW y la fusión con BM25 sparse, sin reindexar.

| Método | Recall@10 | P@1 | P@5 | nDCG@10 | Span | p50 |
|---|---|---|---|---|---|---|
| Denso ef=200 | 0,868 | 0,553 | 0,842 | 0,725 | 0,789 | 7 ms |
| Denso ef=800 | 0,895 | 0,553 | 0,868 | 0,738 | 0,816 | 13 ms |
| Híbrido 0,5/0,5 ef=800 | 0,895 | 0,684 | 0,895 | 0,807 | 0,868 | 19 ms |
| Reranker sobre ef=800 | 0,868 | 0,526 | 0,789 | 0,702 | 0,895 | 792 ms |

El sistema final usa la fusión 0,5/0,5 con `ef=800`: Recall@10 0,895, nDCG 0,807 y P@1 0,684. La latencia de `/retrieve`, incluida la codificación de la consulta, es de unos 165 ms. El reranker no se adopta. Artefactos: `reports/tablas/bench_final_tuning.csv` y `bench_final_tuning_raw.csv`.

## 20. E12: silver ampliado y configuración final (2026-11-01)

Dos anotadores revisaron 76 candidatos, dos por pregunta. Kappa 0,626, 12 discrepancias. Se añadieron 17 documentos de consenso. Catorce preguntas quedaron con dos o más relevantes.

| Método | Silver | Recall@10 | nDCG@10 | P@1 | P@5 | Span |
|---|---|---|---|---|---|---|
| Denso ef=800 | 38 | 0,895 | 0,738 | 0,553 | 0,868 | 0,816 |
| Híbrido 0,5/0,5 | 38 | 0,895 | 0,807 | 0,684 | 0,895 | 0,868 |
| Denso ef=800 | ampliado | 0,921 | 0,827 | 0,737 | 0,947 | 0,816 |
| Híbrido 0,5/0,5 | ampliado | 0,890 | 0,804 | 0,737 | 0,947 | 0,868 |

Con el silver ampliado el dense supera a la fusión. El sistema final usa `BAAI/bge-m3` con `ef=800`, `RETRIEVER=dense`: Recall@10 0,921 y nDCG@10 0,827. Artefactos: `eval/silver/silver_standard_ampliado.jsonl`, `reports/tablas/ampliado/bench_final_tuning.csv` y `bench_ampliado_global.csv`.
