# Métricas de la configuración final

Fecha: 2026-11-01

## Configuración

| Componente | Valor |
|---|---|
| Generador | `gemma4:e4b` |
| MTP | draft nativo de Ollama, `draft_num_predict=3` |
| Contexto | 116.384 tokens (16.384 de entrada y 100.000 de salida) |
| Recuperador | bge-m3 denso, `ef=800`, `k=5` |
| Reranker | `BAAI/bge-reranker-v2-m3` en CPU, sobre el top-5 |
| Compuerta de dominio | clasificador bge-m3 + logística, sin juez, `DOMAIN_THRESHOLD=0.5` |
| Juez de evidencia | `gemma2:2b` |
| Condensador | `gemma2:2b` |
| Razonamiento | nativo de gemma4, opcional |

Gold: 43 preguntas, 38 directas y 5 sin respuesta.

## MTP

El MTP ya estaba activo en el modelo. El manifiesto de `gemma4:e4b` trae una capa `DRAFT` y llama-server arranca con `--spec-type draft-mtp`. Los logs muestran aceptación de 0,65 a 0,74 y 2,9 a 3,2 tokens por paso. La decodificación especulativa es exacta: el draft propone y el modelo objetivo verifica.

| Draft | Tokens por segundo |
|---|---|
| 3 | 217,1 |
| 5 | 193,5 |
| 7 | 159,5 |

El draft de 3 es el mejor. No se cambió el `docker-compose.yml` por MTP; el contenedor ya lo usaba.

## Recuperación

El reranker reordena el top-5 denso. No añade ni quita documentos.

| Top-5 | Recall | nDCG | P@1 | P@5 | span hit |
|---|---|---|---|---|---|
| Denso | 0,895 | 0,809 | 0,737 | 0,947 | 0,763 |
| Reranker | 0,895 | 0,865 | 0,868 | 0,947 | 0,763 |

El nDCG sube 0,056 y el P@1 sube 0,132, sin perder recall ni P@5. La recuperación densa a `k=10` sigue en 0,912 de Recall@10 y 0,818 de nDCG.

La corrección de spans de Óscar se volvió a medir: `span_hit` se mantiene en 0,816, así que la tabla conserva la corrida limpia.

## Respuesta

| Métrica | Sin razonamiento | Con razonamiento |
|---|---|---|
| Directas entregadas | 37 de 38 | 37 de 38 |
| Abstenciones directas | 1 | 1 |
| Abstenciones en sin respuesta | 5 de 5 | 5 de 5 |
| Match estricto | 26 de 37 (70,3%) | 25 de 37 (67,6%) |
| F1 por tokens | 0,273 | 0,272 |
| Respuestas con cita | 37 de 37 (100%) | 37 de 37 (100%) |
| Contexto medio | 1.361 tokens | 1.361 tokens |
| Latencia total, media | 6.788 ms | 6.482 ms |
| Latencia total, mediana | 6.790 ms | 6.435 ms |
| Latencia total, p95 | 9.404 ms | 8.765 ms |

El razonamiento no cambia los puntajes. Cambia qué pregunta directa se abstiene: sin razonamiento se abstiene g-031 por el generador; con razonamiento se abstiene g-017 por el juez. El conteo y el match estricto son iguales.

### Revisión manual con reranker

12 respuestas no estrictas: 8 correctas, 3 parciales y 1 incorrecta.

| Categoría | n |
|---|---|
| Equivalentes | 8 |
| Parciales | 3 |
| Incorrectas | 1 |

Con la revisión, 33 de 37 son equivalentes (89,2%); 36 de 37 si se aceptan las parciales (97,3%). La incorrecta es g-046: el retrieval no trae Newcastle. La corrida sin reranker daba 32 de 37 (86,5%).

## Compuerta de dominio

Set de 160 preguntas: 60 de dominio y 100 de fuera. 103 train, 37 dev y 20 test. El test tiene 11 dominios de entidades nuevas y los 4 casos de fuera del chat.

| Métrica | Valor |
|---|---|
| Modelo | regresión logística con bge-m3 y features de retrieval |
| Dev F1 | 1,000 |
| Test objetivo (4 fuera + 11 dominio nuevo) | 14 de 15 |
| Test completo | F1 0,952 |
| Falsos rechazos de dominio | 0,091 |
| Falsos aceptados de fuera | 0,000 |

Los 40 negativos nuevos (entidades inventadas, eventos futuros y conocimiento general) bajaron los falsos aceptados de 0,333 a 0,000. Los 5 casos sin respuesta se rechazan con 0,293 a 0,436. El único falso rechazo es «¿Cuánto se aceleró la inflación en Colombia?» en 0,494; el umbral sigue en 0,5 porque el margen de dev es 0,456 a 0,559.

En la API:

- `strict=True` en los 4 fuera: `abstained=True`, `abstain_reason="fuera_dominio"`, probabilidades 0,175 a 0,308. Las 5 preguntas sin respuesta se rechazan con 0,293 a 0,436.
- `strict=True` en el Retamo: dominio, probabilidad 0,750, respuesta con citas.
- Los casos de fuera tardan 61 a 88 ms, sin router ni generación.
- El chat completo con `Modo estricto` da 14 de 14: 4 fuera se abstienen, 4 dominio responden y 6 saludos pasan.

## Chat

| Configuración | Aciertos | Charla | Dominio | Fuera |
|---|---|---|---|---|
| Sin Modo estricto | 11 de 14 | 6 de 6 | 4 de 4 | 1 de 4 |
| Con Modo estricto | 14 de 14 | 6 de 6 | 4 de 4 | 4 de 4 |

Sin la compuerta, los tres casos de "fuera" que no se abstienen son "capital de Francia", "Cien años de soledad" y "mundial 2022". El corpus contiene las respuestas y el sistema las cita. Con la compuerta se abstienen los cuatro.

## RAGAS

38 preguntas directas. Actual: juez `gemma4:e4b` y embeddings `BAAI/bge-m3`. Anterior: juez `gemma2:2b` y embeddings `qwen3-embedding:8b`.

| Métrica | Actual | Anterior |
|---|---|---|
| Faithfulness | 0,952 (n=38) | 0,925 (n=36) |
| Answer relevancy | 0,840 (n=38) | 0,798 (n=38) |
| Context precision | 0,881 (n=34) | 0,760 (n=36) |
| Context recall | 0,921 (n=38) | 0,842 (n=38) |
| Answer correctness | 0,657 (n=38) | 0,705 (n=37) |

Las 5 preguntas sin respuesta se abstienen 5 de 5 y se miden aparte. La corrida de referencia usa juez `gemma4:e4b` y embeddings `BAAI/bge-m3` con `RAGAS_TIMEOUT=900` y `RAGAS_WORKERS=6`.

Las dos corridas anteriores usaban el timeout por defecto de 180 s con 16 workers. El juez no terminaba en varias muestras y `context precision` quedaba con 4 valores de 38: su 0,750 no era comparable. Con el timeout corregido la métrica tiene 34 valores y sube a 0,881. Los artefactos de esas corridas quedan en `ragas_gemma4-e4b_bge-m3_run1.csv` y `_run2.csv`.

La arquitectura actual gana en faithfulness, answer relevancy, context precision y context recall; pierde en answer correctness (0,657 contra 0,705). La comparación con la arquitectura anterior cambia juez y embedding, así que no es completamente equivalente. La mejora pendiente es la corrección de respuesta, no la precisión del contexto.

## Latencia

| Configuración | Latencia media |
|---|---|
| k=15, sin razonamiento | 17.657 ms |
| k=5, sin reranker | 14.951 ms |
| k=5 + reranker, sin razonamiento | 6.788 ms |
| k=5 + reranker, con razonamiento | 6.482 ms |

La mayor ganancia vino de alinear el contexto del router con el del generador. Antes el router usaba 8K y el generador 116K, y Ollama recargaba el modelo en cada consulta. El MTP aporta 217 tokens por segundo. El reranker en CPU añade unos 1,1 segundos. La primera consulta de una sesión tarda unos 29 segundos por la carga del modelo.

## Limitaciones

- La revisión manual la hizo un solo anotador.
- El gold tiene 43 preguntas; los intervalos de confianza no se reportan.
- La compuerta deja pasar 3 de los 5 casos sin respuesta; el RAG se abstiene en los 5.
- RAGAS actual usa `gemma4` y `bge-m3`; la comparación con la corrida anterior cambia juez y embedding.
- El contexto de 116K usa 13,9 GB de VRAM y deja 3,2 GB libres.
- La primera consulta tarda unos 29 segundos; conviene precalentar la demo.

## Artefactos

- `plan/evaluacion_dominio_ragas.md`: plan y estado por fase.
- `eval/domain/domain_questions.jsonl` y `.csv`: 160 preguntas.
- `modelos/domain_classifier.json` y `.joblib`: compuerta.
- `reports/tablas/domain_classifier_metrics.csv` y `domain_classifier_predictions.csv`.
- `reports/tablas/metricas_final_sin.csv` y `metricas_final_con.csv`: respuesta final.
- `reports/tablas/metricas_final_revision_manual.csv`: revisión con reranker.
- `reports/tablas/metricas_rerank_top5.csv`: denso contra reranker.
- `reports/tablas/chat_eval_final_strict.csv`: chat con Modo estricto.
- `reports/tablas/ragas_gemma4-e4b_bge-m3.csv`: RAGAS actual con timeout corregido.
- `reports/tablas/ragas_gemma4-e4b_bge-m3_run1.csv` y `_run2.csv`: corridas con el timeout anterior.
- `reports/tablas/ragas_gemma2-2b.csv`: RAGAS de la arquitectura anterior.
- `docs/ARQUITECTURA.md`: arquitectura y resultados.
- `reports/corridas/metricas_final_sin.jsonl` y `metricas_final_con.jsonl`.


## Fase 6. Mejoras del RAG

Se probaron cuatro hipótesis sobre las 38 preguntas del gold. La referencia es el top-5 con reranker: recall 0,895; P@5 0,947; nDCG 0,865; P@1 0,868; 1,30 s.

| Hipótesis | Resultado | Decisión |
|---|---|---|
| H6.1 filtro por score del reranker | recall 0,838 con ratio 0,8 y 0,825 con 0,9; P@5 0,921; nDCG 0,837/0,830 | Rechazada |
| H6.2 rerank del top-20 al top-5 | recall 0,864; P@5 0,921; nDCG 0,853; span 0,868; 5,46 s | Rechazada |
| H6.3 expansión de consulta con RRF | recall 0,825; P@5 0,895; nDCG 0,739; con reranker 0,825/0,895/0,822 | Rechazada |
| H6.4 prompt v4 para fechas | 37/38; 25/37; F1 0,273; 6,98 s | Rechazada |

Ninguna variante supera al top-5 con reranker y el prompt v2. La configuración actual se mantiene. Los artefactos son `reports/tablas/mejoras_recuperacion.csv`, `mejoras_recuperacion_por_idioma.csv` y `metricas_v4_sin.csv`.

## Diversidad del contexto (MMR)

Se probó MMR sobre el top-20 para atacar el ruido del contexto. La referencia es el top-5 con reranker.

| Variante | Recall | nDCG | P@1 | P@5 | span hit | p50 |
|---|---|---|---|---|---|---|
| Top-5 con reranker | 0,895 | 0,865 | 0,868 | 0,947 | 0,763 | 1,03 s |
| MMR coseno 0,5 | 0,868 | 0,784 | 0,737 | 0,947 | 0,737 | 0,10 s |
| MMR coseno 0,7 | 0,855 | 0,781 | 0,737 | 0,947 | 0,737 | 0,10 s |
| MMR reranker 0,5 | 0,816 | 0,814 | 0,868 | 0,921 | 0,842 | 4,06 s |
| MMR reranker 0,7 | 0,816 | 0,818 | 0,868 | 0,921 | 0,842 | 4,06 s |
| Top-20 al top-5 | 0,864 | 0,853 | 0,868 | 0,921 | 0,868 | 4,06 s |

Ninguna variante supera al top-5 con reranker en recall, nDCG ni P@5. MMR con el score del reranker sube la cobertura de span (0,842 frente a 0,763) pero pierde 8 puntos de recall y duplica la latencia. Se rechaza.

### Diagnóstico de los dos fallos

- g-031: el documento de referencia está en el puesto 11 del denso top-50 y en el puesto 1 tras el reranker. El pipeline solo pasa 5 candidatos al reranker, así que nunca lo ve. Ampliar el pool lo recupera, pero cuesta 3,1 puntos de recall y 2,6 de P@5 en el conjunto.
- g-046: el documento no aparece ni en el top-50 denso ni tras el reranker. El problema está en la consulta o en el índice, no en el reordenamiento.

Los artefactos son `reports/tablas/mejoras_diversidad.csv` y `src/scripts/eval_rag_diversity.py`.
