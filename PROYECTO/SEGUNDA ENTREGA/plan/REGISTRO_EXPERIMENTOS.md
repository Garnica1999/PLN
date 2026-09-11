# REGISTRO DE EXPERIMENTOS — Proyecto Juris-PLN (E1 + E2 ciclos v1/v2/v3)

Actualizado: 2026-09-11 · Mantenimiento: cada experimento se registra ANTES de avanzar de fase
(verificado por ml-reviewer). Fuente de cifras: `resultados/metricas.json` + JSONs de `resultados/`.

## Premisas metodológicas globales
1. **Evidencia primero**: cada método candidato proviene de literatura verificable (SOTA) y se mide
   contra el benchmark propio antes de adoptarse.
2. **Adopción condicionada**: Δ en la métrica objetivo debe superar el ruido (IC 95% bootstrap) y ser
   consistente en las mitades A/B; en su defecto se documenta como negativo con igual rigor.
3. **Anti-leakage estricto**: splits canónicos (`splits_m4.parquet`), test intocable hasta la evaluación
   final, exclusiones explícitas en entrenamientos de dominio (las 90 sentencias del benchmark).
4. **Reproducibilidad**: versiones congeladas con md5, seeds fijas, scripts completos por experimento.
5. **Costo asumido**: la mejora debe ser compatible con la latencia declarada del producto
   (default <= 10 s; modos opcionales <= 30 s).
6. **Trazabilidad total**: este registro es obligatorio por experimento (hipótesis, premisa, diseño,
   resultado, decisión, artefacto) y los ADRs registran las decisiones costosas de revertir.

---

## Preludio E1 (análisis del corpus, 2026)
| ID | Hipótesis | Premisa | Resultado | Decisión |
|---|---|---|---|---|
| P1 | El corpus completo (23.742 Providence) es la base de todo el proyecto | Datos completos + limpieza reproducible habilitan métricas confiables | CSV 1.67 GB; limpieza 250.012.101 tokens; regex parseo 99,97 % | Adoptado (base) |
| P2 | Colocaciones por LLR capturan frases jurídicas mejor que frecuencia/PMI | Features con significancia estadística > conteo crudo | LLR 75 % fórmulas válidas vs freq 55 %, PMI 0 % | Adoptado (keywords M2/M4) |
| P3 | La subsentencia (párrafo numerado) mejora la granularidad | Fragmentar solo si supera +15 % de señal | Párrafo = sentencia (sin doble salto); subsentencia 4.64 % | Rechazado → sentencia completa (ADR-8) |

## Ciclo v1 — E2 (2026-09-08/10)
| ID | Fecha | Hipótesis | Premisa | Diseño | Resultado | Decisión | Artefacto |
|---|---|---|---|---|---|---|---|
| C | 09-08 | PySpark+Parquet escala mejor que CSV para 23.7k docs | Rendimiento habilitante | Migración Spark 4.2/Java 17 | 237,5 s; 1.465 GB; limpieza 100 % idéntica a E1 | Adoptado (ADR-7) | `data/corpus_sentencias.parquet` |
| D | 09-08 | Lema vs stem para clásicos; POS/MTLD describen el corpus | Preprocesamiento basado en evidencia | EDA extendido (celdas 0-23 E1 intactas) | Lemas 87,3 % vs stems 54,2 %; POS NOUN>ADP>DET; MTLD ~61 | ADR-6, ADR-8 | `figuras/`, `eda_extendido/` |
| G | 09-08 | La CP canónica completa permite etiquetar y vincular | Fuente oficial (Senado) | Parseo HTML oficial | 497 arts (384 perm + 112 trans + 1 preámbulo); 19 bloques; spot-check 5/5 | Adoptado | `data/constitucion_articulos.parquet` |
| H | 09-08 | Regex+ventana extrae citas con precisión suficiente | Las citas son la fuente de weak labels y gold | NER jurídico sobre 23.742 | Precisión 1.00/1.00/1.00/0.90; cobertura CP 98,71 %; 516.529 aristas | Adoptado | `data/entidades_sentencias.parquet` |
| I | 09-08 | Etiquetar área vía citas CP con fallback léxico | Weak supervision con cobertura alta | Etiquetado M3 | 99,63 % asignadas; 16 clases; top derechos_fund 42,5 % | Adoptado (ADR-1) | `data/etiquetas_areas.parquet` |
| J | 09-08 | Lematizar el corpus completo para features clásicas | Necesario para TF-IDF (ADR-6) | spaCy 4 workers + checkpoints | 23.742/23.742; 195.959 tipos (=ref EDA) | Adoptado | `data/lemas_por_doc.parquet` |
| K1 | 09-08 | Validar entradas (providencia vs no) es separable | M0 con falsos positivos caros | 30 pos + 22 neg | Accuracy 1,0 (FP=0, FN=0) | Adoptado (muestra pequeña doc.) | `resultados/m0_validacion.json` |
| K2 | 09-08 | Benchmark de vínculo CP vs referencia de extracción | Medir antes de optimizar | 30 casos (1/año) | cita_exacta 0,70 > BM25 0,119 > E5 0,090 | Referencia | `resultados/` (v1) |
| K3 | 09-08 | La ventana corta de query mejora el vínculo (refutable) | Hipótesis del equipo a validar | 90 casos (3/año) + ablations | REFUTADA (1ª cita a mediana 41 %); BM25 0,119→0,218 con pasajes+max-pooling+stopwords+LLR | ADR-9 | `vinculo_cp_benchmark_v2.json` |
| L1 | 09-09 | Baseline clásico (LogReg/SVC) antes de transformers | Simplicidad primero | TF-IDF lemas 100k, 16 clases | LogReg 0,2852/0,2870; LinearSVC 0,2654; LightGBM descartado (RAM) | Adoptado LogReg | `m4_baseline.json` |
| L2 | 09-09 | El enmascarado elimina señal de citas sin leakage activo | Auditoría adversarial obligatoria | Revisión + prueba de oro | Rechazo v1 (0,7 citas/doc vs 10,7); v2 corregido; gap ≈0; cita-solas F1 0,0611 ≈ azar | Adoptado enmascarado v2 | ADR-2, `m4_baseline.json` |
| L3 | 09-09 | BETO contextual supera bolsa de palabras en docs largos (refutable) | Hipótesis estándar a validar | 3 épocas, 256+256, AMP | 0,2080/0,2042; gap 0,0039; LogReg gana (docs ~11k palabras vs ventana 512) | ADR-10 (LogReg; BETO comparativa) | `m4_transformer.json` |
| M | 09-09 | La red de citas y la diacronía aportan contexto jurisprudencial | Valor de producto (precedentes/evolución) | NetworkX + PageRank + log-odds + MTLD | 456.321 aristas (84,48 %); landmarks C-224/1992; MTLD 64,9→56,8 | Adoptado (contexto) | `red_citas.json`, `diacronia.json` |
| N | 09-09 | Integrar 4 componentes en CLI con JSON es viable | Producto end-to-end | `10_evaluar_sentencia.py` + lib | Demos OK (T-293/97 0,943); ~9 s/eval | Adoptado | producto v1 |
| QA | 09-10 | Gate de entrega con verificación independiente | Calidad antes de publicar | QA + auditoría rendimiento | GO: 10/10, 16/16 métricas; 8-12 s; RAM 8,1 GB; cuello init ~75 % | GO | `qa_final/` |
| Z | 09-10 | Cierre documental reproducible | Entrega completa | Informe+deck+zip | 3.773 palabras; 15 slides; zip 29 MB | Entregado | `documento_e2/`, zip v1 |

## Ciclo v2 — Mejora incremental (2026-09-10/11)
| ID | Fecha | Hipótesis | Premisa | Diseño | Resultado | Decisión | Artefacto |
|---|---|---|---|---|---|---|---|
| R | 09-10 | Medir con nDCG@10 y cuantificar contaminación antes de arreglar | Diagnóstico honesto (BEIR) | Benchmark v3 (90 casos) | 24,4 % transitorios top-5; 65/90 casos; 152 IDs ACL; nDCG@10 0,342 | Evidencia para S1 | `vinculo_cp_benchmark_v3.json` |
| S1 | 09-10 | El ruido estructural del índice (transitorios/IDs rotos) degrada el ranking | Limpiar antes de modelar | Fix 45 IDs ACL + índice permanentes | 0 contaminación; bm25 0,233/0,765/0,380; 45 IDs verificados vs Senado | Adoptado (producto) | `v3_limpio.json` |
| S2 | 09-10 | La lectura selectiva acelera el producto sin cambiar salidas | Optimización de rendimiento auditada | pyarrow filters + column pruning | 8,8→5,5 s/eval; RAM 8,0→1,6 GB; JSON idéntico | Adoptado | producto v2 |
| T1 | 09-10 | El cross-encoder con query completa mejora el ranking (refutable) | Hipótesis estándar de reranking | v4 (query ~68k truncada 512) | FALLO: F1@5 0,177 (−24 %) por truncamiento de query | Negativo documentado | `vinculo_cp_benchmark_v4.json` |
| T2 | 09-10 | El rerank POR CHUNKS con max-pooling corrige el fallo | Causa diagnosticada (contexto perdido) | v5: top-20 BM25 + 3 pares chunk-pasaje + max-pool | 0,285 (+61 % vs v4); mejor no-circular | Adoptado | `v5.json` |
| T3 | 09-10 | Ampliar candidatos (top-50 híbrido) ataca el techo de recall | F1@10 estancado por recall@20 | v6: híbrido BM25+E5 top-50 | 0,340/0,865/0,513/0,316; 3/3 metas del plan | Adoptado (ruta producto) | `v6.json` |
| V | 09-11 | El fallo de BETO v1 era del truncamiento, no del transformer | Chalkidis 2022 (chunk-mean) | Sliding window 256/128, media de chunks, AMP | 0,2433/0,2363 (+17 %/+16 % vs v1); gap 0,0070; meta 0,26 no; LogReg se mantiene | ADR-13 | `m4_transformer_chunks.json` |
| V-inc | 09-11 | Incidentes del entrenamiento | Robustez operativa | Deadlock DataLoader (Python 3.14+fork); crash scaler | Cache chunks (1,3 GB/3,6 min) + num_workers=0; fix scaler.update() 1 línea | Resueltos y documentados | `experimentos/*evidencia*` |
| U | 09-11 | Jerarquía/multietiqueta/calibración superan al plano (meta 0,32) | Estructura de clases puede ayudar | A cascada, B multietiqueta, C sesgos por clase (val) | C 0,2980 (+3,8 %); A 0,2697; B 0,2500; meta 0,32 no | ADR-14 (no cambio; C disponible) | `m4_jerarquico.json` |
| W-b | 09-11 | EuroVoc como señal auxiliar mejora la clasificación | Ontología estándar útil si el mapeo es viable | D cascada EuroVoc; E léxico (1.268 términos ES, SPARQL verificado) | D 0,2625 / E 0,2771 < C 0,2980 | No adoptado (ADR-15) | `m4_eurovoc.json`, `config/eurovoc_*.json` |
| W-a | 09-11 | Los artículos hub resisten el decay temporal y aportan prior | arXiv:2607.11109; hubs 241/86/13/2/1 | Prior de frecuencia global + rejilla α calibrada A/B | α=1,5: 0,425/0,950/0,620/0,379 (mejor estrategia); default α=0,75: 0,396/0,901/0,601 | ADOPTADO (ADR-15) | `v7.json`, `v8.json` |
| X | 09-11 | Adoptar ganadores verificados en el producto | Equivalencia producto↔benchmark | Integración VinculoCP v2 + CLI | Default 4-7 s; rerank 14-18 s; equivalencia 3/3 exacta | Adoptado (producto v2) | `componentes_producto.py` |
| QA2 | 09-11 | Gate de regresión independiente | Calidad antes de cerrar | Checks v2 + 10 casos E2E | GO: 10/10; drift clasificador 0; control negativo v1 detectado | GO | `qa_final_v2/VEREDICTO_QA.md` |
| Z2 | 09-11 | Cierre documental del ciclo | Entrega del ciclo | Informe+deck+zip | 2.499 palabras; 23 slides; zip 31,1 MiB | Entregado | `documento_e2/`, zip v2 |

## Ciclo v3 — Recuperación robusta (EN CURSO, 2026-09-11 →)
| ID | Sprint | Hipótesis | Premisa | Diseño | Resultado | Decisión | Artefacto |
|---|---|---|---|---|---|---|---|
| 0 | Paso 0 | Congelar v2 y habilitar trazabilidad v3 | Versionado congelado (ADR-16) | Commit + registro | (en curso) | — | rama git, este registro |
| A1 | A | Las métricas actuales esconden el techo real de cada etapa | Medir con IC antes de optimizar | recall@k + IC bootstrap + periodos | — | — | `c3_diagnostico_recall.json` |
| A2 | A | Embeddings legales ES superan a E5 en la 1ª etapa | Modelos de dominio (SINAI/wilfredomartel, Apache-2.0) | Comparativa 4 modelos + rerank + hubs | — | — | `benchmark_v9.json` |
| A3 | A | Un reranker legal ES supera a bge-reranker-v2-m3 | Reranker de dominio disponible | Comparativa + RRF entre rerankers | — | — | `c3_a3_rerankers.json` |
| A4 | A | RRF supera la mezcla 0,5/0,5 y ampliar candidatos ayuda | Fusión por rangos robusta | RRF k=60; top-50/100/200 | — | — | v9 |
| A5 | A | Calibración C + ensamble superan al LogReg plano | Mejoras ya entrenadas y medidas | Ensamble + class weights | — | — | `m4_clasificador_c3.json` |
| B1 | B | Un reranker entrenado con citas propias supera a los genéricos | El corpus es el activo único de dominio | Fine-tune con hard negatives (sin las 90) | — | — | `benchmark_v10.json` |
| B2 | B | Un retriever denso de dominio mejora el recall de 1ª etapa | Contrastive con citas propias | Fine-tune embedding (negativos minados) | — | — | `benchmark_v11.json` |
| B3 | B | El gold mini mide lo que el gold NER no ve | Anotación humana (40-60) | Herramienta + análisis de errores | — | — | `gold_mini.json`, `c3_analisis_errores.md` |
| B4 | B | BETO chunk-mean mejora con más épocas/datos | Val seguía subiendo | Más épocas / max_chunks 96 | — | — | (opcional) |
| C1 | C | El gold estándar (150) corrige la evaluación | Evaluación con relevancia graduada | Guía + anotación | — | — | `gold_150.json` |
| C2 | C | El LLM local mejora query/reranking (premisas en PLAN_v3) | SOTA 2026 + causa raíz medida | ALIA-7B: query understanding / rerank listwise | — | — | `c3_llm.json` |
| C3 | C | La robustez temporal se valida con corpus post-2021 | Datos que ningún modelo vio | Extensión corpus + deriva | — | — | corpus v3 |
| C4 | C | Integración final del producto | Adopción solo con gates | Producto v3 + QA v3 | — | — | producto v3 |
| C5 | C | Documentación integral del proyecto | Trazabilidad total (premisas/datos) | Informes + ADRs + auditoría de cifras | — | — | `informe_final_ciclo_v3.md` |
