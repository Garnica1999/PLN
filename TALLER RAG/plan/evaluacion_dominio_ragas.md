# Evaluación de dominio, revisión manual y RAGAS

Fecha: 2026-11-01

## Objetivo

Cerrar tres frentes del taller sin añadir un juez nuevo al runtime:

1. Compuerta de dominio para los casos de fuera, con un clasificador supervisado, no un LLM juez.
2. Revisión manual de las respuestas finales con el reranker.
3. Evaluación con RAGAS usando un juez local de Ollama.

## Decisiones

- El juez de evidencia actual del RAG se queda como está. No se añade otro juez al runtime.
- La compuerta de dominio usa un clasificador de embeddings y, si hace falta, features de retrieval.
- El set de dominio tiene 160 preguntas: 60 de dominio y 100 de fuera.
- El split es por entidad o tema, no aleatorio, para que el test tenga entidades no vistas.
- RAGAS se instala en un contenedor `eval` con Python 3.12. El intento en `/home/carlos/venv` se revirtió porque rompió numpy y LangChain.
- El juez de RAGAS es local, por el endpoint compatible con OpenAI de Ollama. Se decide entre `gemma4:e4b` y `gemma2:2b` según el coste.

## Fases

| Fase | Estado | Entregable |
|---|---|---|
| 0. Instalar RAGAS | Completada | `ragas` importable en el contenedor `eval` |
| 1. Generar las 160 preguntas | Completada | `eval/domain/domain_questions.jsonl` |
| 2. Entrenar y evaluar el clasificador | Completada | `modelos/domain_classifier.json` |
| 2b. Integrar la compuerta | Completada | Toggle `Modo estricto` y `strict` en la API |
| 3. Revisión manual con reranker | Completada | 33 de 37 equivalentes (89,2%) |
| 4. Exportar dataset y correr RAGAS | Completada con gemma2; pendiente gemma4 | `reports/tablas/ragas_gemma2-2b.csv` |
| 5. Actualizar informe y plan | Completada | `reports/informe/metricas_nueva_configuracion.md` |

## Fase 0. RAGAS

- Respaldo: `reports/entorno/venv_antes_ragas.txt`.
- El intento en `/home/carlos/venv` falló: subió numpy a 2.5.4, bajó openai/rich/tenacity y el import quedó roto por `langchain_community.chat_models.vertexai`.
- El venv del host se restauró a numpy 1.26.4, openai 3.3.1, rich 15.0.0 y tenacity 9.1.4. `pip check` solo muestra conflictos preexistentes.
- RAGAS quedó en un contenedor aislado con Python 3.12: `Dockerfile.eval`, `requirements-eval.txt` y el perfil `eval` del compose.
- Versiones del contenedor: `ragas 0.4.3`, `langchain 0.3.30`, `langchain-community 0.3.31`, `langchain-core 0.3.86`, `langchain-openai 0.2.14`, `langchain-ollama 0.2.3`.
- `import ragas` funciona en el contenedor. El juez local se probará con `gemma2:2b` y `gemma4:e4b`.

## Fase 1. Las 160 preguntas

Composición:

| Grupo | n | Etiqueta |
|---|---|---|
| Directas del silver | 38 | dominio |
| Nuevas de dominio desde títulos y entidades del corpus | 22 | dominio |
| Fuera del chat | 4 | fuera |
| Sin respuesta del silver | 5 | fuera |
| Conocimiento general nuevo | 51 | fuera |
| Negativos duros nuevos (entidades inventadas, futuros, general) | 40 | fuera |
| Total | 160 | |

Campos: `id`, `question`, `label`, `subtype`, `source`, `split`.

Criterios:

- Las 22 nuevas de dominio deben ser respondibles desde el corpus y no solo variantes de las mismas entidades.
- Las 51 de fuera incluyen negativos difíciles que el corpus menciona de pasada.
- El test debe tener entidades no vistas en train.

## Fase 2. Clasificador

- Base: embeddings bge-m3 de la pregunta.
- Modelos: regresión logística y LightGBM con features de retrieval.
- Métricas: precision, recall, F1, falsos rechazos de dominio y falsos aceptados de fuera.
- Objetivo: F1 mayor o igual a 0,95 y falsos rechazos menores o iguales a 2% en test.
- Si no se alcanza, fine-tune de un transformer multilingüe pequeño.
- La compuerta se activa con un toggle. Por defecto apagada hasta que el test cumpla el objetivo.

### Resultado del clasificador

- 160 preguntas: 103 train, 37 dev y 20 test.
- Modelo elegido: regresión logística con embeddings bge-m3 y features de retrieval. LightGBM no está instalado en la imagen.
- Dev F1: 1,000, con margen de 0,456 a 0,559.
- Test en el objetivo de la compuerta, 4 fuera del chat contra 11 dominios nuevos: 14 de 15.
- Test completo: F1 0,952, precision 1,000, recall 0,909. Los 5 casos sin respuesta se rechazan con 0,293 a 0,436. El único falso rechazo es «¿Cuánto se aceleró la inflación en Colombia?» en 0,494; no se movió el umbral porque el margen de dev lo deja en 0,5.
- Los 40 negativos duros nuevos bajaron los falsos aceptados de 0,333 a 0,000.
- El modelo se exportó a `modelos/domain_classifier.json` para no depender de sklearn en la API.

### Resultado de la compuerta

- `strict=True` en los 4 casos de fuera: `abstained=True`, `abstain_reason="fuera_dominio"`, probabilidades 0,186 a 0,428.
- `strict=True` en el Retamo: dominio, probabilidad 0,750, respuesta con citas `[1, 3]`.
- Los casos de fuera se resuelven en unos 41 ms, sin router ni generación.
- Toggle `Modo estricto` en la barra lateral. Por defecto apagado.
- Chat completo con `Modo estricto`: 14 de 14. Los 4 fuera se abstienen; los 4 de dominio responden; los 6 saludos pasan. Los casos de fuera tardan 61 a 88 ms.

## Fase 3. Revisión manual

- Fuente: `reports/corridas/metricas_final_sin.jsonl`.
- 12 respuestas no estrictas entregadas con el reranker.
- 8 correctas, 3 parciales y 1 incorrecta.
- 33 de 37 equivalentes (89,2%); 36 de 37 si se aceptan las parciales (97,3%).
- La incorrecta es g-046: el retrieval no trae Newcastle.
- La corrida sin reranker daba 32 de 37 (86,5%).
- CSV: `reports/tablas/metricas_final_revision_manual.csv`.

## Fase 4. RAGAS

- Dataset: `reports/ragas/dataset.jsonl` con 38 directas y `reports/ragas/negatives.jsonl` con 5 negativas.
- Las 5 sin respuesta se abstienen 5 de 5.
- Juez `gemma4:e4b` y embeddings `BAAI/bge-m3`.
- Resultados actuales: faithfulness 0,952; answer relevancy 0,840; context precision 0,881; context recall 0,921; answer correctness 0,657. La corrida usa `RAGAS_TIMEOUT=900` y `RAGAS_WORKERS=6`; con el timeout anterior, context precision quedaba con 4 valores de 38.
- Resultados anteriores (`gemma2` + `qwen3`): 0,925; 0,798; 0,760; 0,842; 0,705.
- CSV actual: `reports/tablas/ragas_gemma4-e4b_bge-m3.csv`; anterior: `reports/tablas/ragas_gemma2-2b.csv`.
- La comparación cambia juez y embedding; no es completamente equivalente.

## Fase 6. Mejoras del RAG

| Hipótesis | Cambio | Métrica que debe mover |
|---|---|---|
| H6.1 | Filtrar el top-5 por score del reranker | Subir context precision sin perder recall |
| H6.2 | Rerank del top-20 al top-5 | Recuperar g-031 y subir precisión |
| H6.3 | Expansión de consulta para g-046 y g-031 con RRF | Subir recall en los dos fallos |
| H6.4 | Ajustar el prompt para años y fechas | Mejorar g-008 y g-022 |

Criterio: cada hipótesis se prueba con el pipeline actual; se conserva solo si mejora la métrica objetivo sin perder entregas ni latencia de forma significativa.

### Resultado de la fase 6

| Hipótesis | Resultado | Decisión |
|---|---|---|
| H6.1 filtro por score | recall 0,838 con ratio 0,8 y 0,825 con 0,9; P@5 0,921; nDCG 0,837/0,830 | Rechazada |
| H6.2 top-20 al top-5 | recall 0,864; P@5 0,921; nDCG 0,853; span 0,868; 5,46 s | Rechazada |
| H6.3 expansión de consulta | recall 0,825; P@5 0,895; nDCG 0,739; con reranker 0,825/0,895/0,822 | Rechazada |
| H6.4 prompt v4 | 37/38; 25/37; F1 0,273; 6,98 s | Rechazada |

La referencia es el top-5 con reranker: recall 0,895; P@5 0,947; nDCG 0,865; P@1 0,868; 1,30 s. Ninguna variante la supera. La configuración actual se mantiene: `k=5`, reranker top-5, sin filtro, sin expansión y prompt v2. Los detalles están en `reports/tablas/mejoras_recuperacion.csv` y `metricas_v4_sin.csv`.

## Riesgos

- Python 3.14 puede no tener wheels para todas las dependencias de RAGAS.
- RAGAS puede actualizar `langchain`, `pydantic` u `openai` en el venv del host.
- Con 160 preguntas, el test tiene 20 y los intervalos de confianza siguen siendo amplios.
- El clasificador puede sobreajustar a las entidades del silver si el split no es por entidad.
- RAGAS con `gemma4` puede tardar una hora o más.

## Fase 7. Diversidad

Se probó MMR sobre el top-20 con dos señales de relevancia (coseno y score del reranker) y dos valores de lambda (0,5 y 0,7). Ninguna variante supera al top-5 con reranker: la mejor pierde 8 puntos de recall y 2,6 de P@5, y duplica la latencia. Se rechaza y la configuración actual se mantiene.

Diagnóstico: g-031 está en el puesto 11 del denso top-50 y en el 1 tras el reranker, así que se recuperaría con un pool mayor; g-046 no aparece ni en el top-50. Los detalles están en `reports/tablas/mejoras_diversidad.csv`.
