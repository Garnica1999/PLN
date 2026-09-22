# Deck E2 "Juris-PLN" — Presentación de la Segunda Entrega

*12-15 diapositivas, una por encabezado ##. Sin emojis. Español.*

---

## Slide 1 — Juris-PLN: PLN sobre 23.742 sentencias de la Corte Constitucional de Colombia (1992-2021)

**Segunda Entrega — Procesamiento de Lenguaje Natural**

- Corpus: 23.742 sentencias (T 17.099 / C 6.246 / SU 397), ~261M palabras crudas, 1992-2021
- Producto: dado un texto, el sistema (1) valida si es providencia de la Corte, (2) clasifica su área temática constitucional (16 clases), (3) recupera los artículos de la CP vinculados, (4) la sitúa en su red de precedentes
- Todo reproducible: scripts 00-12, semilla 42, splits una sola vez, 10 ADRs documentados
- Veredicto QA final: GO (10/10 casos, 16/16 métricas consistentes)

## Slide 2 — El problema

- La Corte publica >23.700 providencias en 30 años: lectura manual inviable para juristas e investigadores
- Cuatro preguntas concretas sin herramienta computacional:
  - ¿Este texto es una providencia de la Corte? (validación de entrada)
  - ¿De qué área del derecho constitucional trata? (clasificación temática)
  - ¿Qué artículos de la Constitución la sustentan? (retrieval, robusto a 30 años de deriva de redacción)
  - ¿Qué precedentes la rodean y cómo ha evolucionado la Corte? (red de citas + diacronía)
- Restricciones reales: corpus sin metadatos de área, GPU compartida, documentos de ~11.000 palabras que exceden la ventana de los transformers (512 tokens)

## Slide 3 — Los datos

- Fuente: dataset Hugging Face CC-BY-4.0 (CSV 1.561 GB, 23.750 filas) -> migración a Parquet Hive (ADR-7): 23.742 sentencias (8 excluidas: 6 autos + 2 HTML), limpieza E1 verificada 100 % coincidente
- Constitución Política 1991 parseada de la fuente oficial del Senado: 497 artículos (384 permanentes + 112 transitorios + preámbulo) en 19 bloques temáticos
- Promedio 10.530 palabras limpias por sentencia; stopwords ~48 % del texto
- Etiquetas de área derivadas de la CP (ADR-1): 99,63 % del corpus etiquetado (cita CP 98,71 % + fallback "derecho a X" 0,92 %); 16 clases efectivas, muy desbalanceadas (42,5 % derechos_fundamentales ... 0,01 % elecciones)

## Slide 4 — Método: pipeline reproducible

```
CSV E1 (23.750 filas)
  |
  00_csv_a_parquet.py      -> Parquet Hive tipo/ano (23.742; ADR-7)
  01/02_constitucion.py    -> 497 artículos CP en 19 bloques
  03_extraccion_entidades  -> NER jurídico regex (M2): arts CP, normas,
  |                          sentencias citadas, ponente
  04_etiquetado_areas.py   -> 16 áreas vía CP (M3, ADR-1): 99,63 %
  05_lematizacion.py       -> lemas 23.742/23.742 (M1, 195.959 tipos)
  06_baseline_clasico.py   -> M4: TF-IDF lemas + LogReg/LinearSVC
  07_transformer.py        -> M4: BETO fine-tuned (doble escenario)
  08_red_citas.py          -> M5: grafo 456.321 aristas + PageRank
  09_diacronia.py          -> M6: áreas, log-odds, MTLD
  11/12_evaluar_vinculo    -> M7: benchmark retrieval CP (v1 -> v2)
  10_evaluar_sentencia.py  -> PRODUCTO end-to-end (CLI, JSON)
```

- 1 experimento = 1 script (ADR-4), notebook E1 intacto (diff = 0), ruff limpio, semilla 42 en todos los configs

## Slide 5 — Anti-leakage: el diseño que protege todas las métricas

- Riesgo (ADR-2): las etiquetas derivan de citas a la CP presentes en el texto -> un clasificador podría memorizar citas en vez de aprender tema
- Solución: todo modelo se evalúa en DOBLE escenario: texto completo vs citas enmascaradas -> benchmark 2x2 + gap
- Auditoría adversarial (gate L2): el reviewer RECHAZÓ el enmascarado v1 (enmascaraba 0.7 citas/doc vs 10.7 reales: números arábigos ausentes en lemas, núcleos con tilde) -> v2 corregida
- Prueba de oro: modelo entrenado SOLO con features de cita -> F1 0.0611 ~= azar -> sin leakage activo
- Resultado: gaps ~= 0 en todos los modelos (LogReg -0.0018, BETO +0.0039): el sistema aprende contenido temático

## Slide 6 — Resultado 1: clasificación de área (M4)

Benchmark 2x2 (macro-F1 test, 16 clases, splits 16.557/3.549/3.548):

| Modelo | Completo | Enmascarado | Gap |
|---|---|---|---|
| LinearSVC | 0.2654 | 0.2657 | -0.0003 |
| LogReg+TF-IDF lemas | 0.2852 | 0.2870 | -0.0018 |
| BETO fine-tuned | 0.2080 | 0.2042 | +0.0039 |

- Mejor modelo: LogReg enmascarado 0.2870 (accuracy ~59 %)
- M0 validación: accuracy 1.0 (52 casos: 30 positivos + 22 discursos presidenciales)
- Mitigación del desbalance: el producto expone top-3 de áreas con probabilidades

## Slide 7 — Hallazgo estrella 1: LogReg GANA a BETO

- Documentos ~11k palabras vs ventana 512 tokens: el truncamiento cabeza+cola (256+256) descarta ~95 % del contenido, incluidos los considerandos de fondo donde vive la señal temática
- TF-IDF sobre lemas procesa el documento completo -> gana por 8 puntos de macro-F1 (0.2870 vs 0.2042)
- Consistente con la literatura de predicción legal en textos largos (Chalkidis et al. 2022): bolsa de palabras competitiva o superior a transformers en documentos legales largos
- Curvas BETO sanas (val 0.13 -> 0.19 -> 0.21, sin saturar): podría cerrar la brecha con chunk-mean, pero el coste GPU no lo justifica (ADR-10)
- Decisión: el producto conserva LogReg; BETO guardado como comparativa

## Slide 8 — Hallazgo estrella 2: ADR-9, hipótesis refutada con evidencia

- Hipótesis inicial: usar solo los considerandos iniciales como query del retrieval (ventana corta)
- Evidencia del corpus que la REFUTA: la primera cita a la CP aparece a mediana en el 41 % del texto (solo 29,5 % en el primer cuarto); el sweep de ventanas degrada monótonamente (3.000 chars: F1@5 0.127 vs completo 0.204)
- Lo que SÍ funcionó (v1 -> v2, +84 % F1@5):
  1. Indexar la CP por pasajes (931 de 497 artículos) + max-pooling por artículo
  2. Query = texto completo en chunks (512 tokens, solape 64) + max-pooling
  3. Stopwords jurídicas + bigramas LLR del EDA (ablation 0.218 vs 0.206)
- Lección: refutar con evidencia antes de optimizar; el diagnóstico previo era incorrecto

## Slide 9 — Resultado 2: vínculo sentencia -> artículos CP (M7)

Benchmark 90 casos (3/año 1992-2021, weak labels del NER, ADR-5):

| Estrategia | F1@5 | MRR |
|---|---|---|
| Cita-exacta (referencia circular) | 0.649 | 1.0 |
| BM25 v2 (pasajes + max-pooling) | 0.218 | 0.717 |
| Híbrido BM25+E5 | 0.209 | 0.688 |
| E5 | 0.128 | 0.382 |
| BM25 v1 | 0.119 | 0.544 |

- v1 -> v2: +84 % F1@5 (0.119 -> 0.218)
- Robustez temporal: BM25 estable por periodo; E5/híbrido mejoran en 2012-2021 (híbrido 0.256)
- La cita-exacta es techo de extracción (mismo regex que generó las weak labels), no competidor: motiva usarla como feature de ranking en el producto

## Slide 10 — Resultado 3: la red de precedentes (M5) y la deriva de la Corte (M6)

- Grafo dirigido: 23.742 nodos, 456.321 aristas resueltas (84,5 %), componente gigante 67,6 %
- PageRank: dominan los landmarks fundacionales C-224/1992 y SU-2716/1997
- Evolución: la Corte cita cada vez más precedentes recientes (top-10 por periodo envejece menos que la corte misma)
- Diacronía 1992-2021 (primer vs último trienio):
  - Áreas: principios_fundamentales +20,2 pp; derechos_fundamentales -24,2 pp
  - Léxico (log-odds): incoder/rupd/bancadas -> covid/jep/coronavirus (conflicto -> posacuerdo -> pandemia)
  - MTLD: 64,9 -> 56,8 (prosa menos diversa)

## Slide 11 — Demo del producto (JSON de salida)

Entrada: sentencia T-293/97 (`10_evaluar_sentencia.py --id`), salida JSON en ~9 s:

```json
{
  "es_sentencia": true, "score_validacion": 1.0,
  "ponente": "Jose Gregorio Hernandez Galindo",
  "area": { "top3": [
      {"area": "derechos_fundamentales", "prob": 0.9429},
      {"area": "principios_fundamentales", "prob": 0.0291},
      {"area": "proteccion_aplicacion_derechos", "prob": 0.0186}],
    "via": "clasificador_m4_logreg_enmascarado" },
  "articulos_cp": { "extraidos": [art. 11, 13, 25, 49, 86, 241...],
    "recuperados_bm25": [art. 86 (156.8), 234, ...] },
  "sentencias_citadas": [...],
  "resumen": "Providencia valida... 6 articulos CP extraidos y 5 recuperados por BM25..."
}
```

- Otros casos: C-002/98 (true, 0.934, 18+5 artículos, 4 citas con PageRank) y discurso de Gaitán (false 0.0, sin análisis)
- QA final: 10/10 casos E2E pasan; 16/16 métricas canónicas sin drift; veredicto GO

## Slide 12 — Evaluación de rendimiento y calidad

- Auditoría de rendimiento (perfilado real: cProfile, RSS, VRAM):
  - 8,7-12,7 s por sentencia en frío; 1,2-5,0 s en caliente; CPU-only verificado; RAM pico 8,1 GB
  - Cuello de botella de inicialización (~75 %: lectura corpus + carga spaCy), no algorítmico; la inferencia escala solo con la longitud del documento (51 citas se resuelven en < 1 ms)
  - Veredicto: OK para entrega académica
- Incidentes de ingeniería resueltos y documentados:
  - Java 26 incompatible con Spark 4.2 -> Java 17
  - OOMs por GPU/RAM compartidas -> guards de VRAM, checkpoints por id, reanudación automática (funcionó 3 veces)
  - spaCy max_length 1M < doc máx 3M -> max_length 5M + guard por doc

## Slide 13 — Limitaciones (honestas)

| Limitación | Cifra | Mitigación |
|---|---|---|
| Macro-F1 modesto (16 clases desbalanceadas) | 0.2870 | top-3 con probabilidades en el producto |
| Resolución de citas | 84,5 % (60.208 no resueltas) | campo resuelta explícito |
| M0 con muestra pequeña | 52 casos | negativos E2E rechazados; dominio declarado |
| Cita-exacta circular | F1 0.649 = techo | reportada como referencia, no baseline |
| Weak labels en M7 | sin gold experto | verificación muestral manual (ADR-5) |

## Slide 14 — Trabajo futuro

1. Chunk-mean pooling para BETO: cubrir el 100 % del documento (curvas sin saturar lo justifican)
2. LEGAL-BERT / adaptación de dominio legal en español
3. Re-ranking cruzado top-50 -> E5 por pares en M7; extracción de citas como feature de ranking (cerrar 0.218 -> 0.649)
4. Optimización del producto: leer solo la fila pedida (-42 % tiempo frío), modo lote, BM25 vectorizado
5. Embeddings diacrónicos por lustro (Hamilton 2016) para deriva semántica ("tutela", "territorio ancestral")
6. Validación extendida de M0 con otras cortes; anotación experta muestral de áreas

## Slide 15 — Cierre

- Sistema end-to-end entregado y verificado: validación, clasificación, vínculo CP, red de precedentes, diacronía
- Tres hallazgos transferibles:
  1. La auditoría adversarial del anti-leakage detectó un enmascarado roto que habría validado métricas frágiles
  2. En documentos legales largos, TF-IDF de lemas supera a BETO (ventana 512 vs 11k palabras)
  3. Refutar hipótesis con evidencia del corpus (ADR-9) valió más que cualquier optimización: +84 % F1@5
- Reproducibilidad total: scripts 00-12, 10 ADRs, 20 referencias verificadas, QA GO
- Código, datos, modelos y figuras en `PROYECTO/SEGUNDA ENTREGA/`; informe completo en `documento_e2/informe_final_e2.md`

---

# Sección Ciclo v2 (mejora incremental)

## Slide 16 — Ciclo v2: por qué y qué se midió

- El QA v1 validaba estructura, no calidad caso a caso: T-6251/01 recuperaba 0/2 artículos con el top-5 contaminado por transitorios
- Diagnóstico: 45 IDs malformados ("Transitorio ACL01016-N"), 112 transitorios en el índice (24,4 % del top-5 en v3) y techo de recall con k=5 (gold de ~10,7 arts/sentencia)
- Faltaban nDCG@10 (estándar BEIR) y el top-10 por caso para auditar reranking
- Hipótesis pendiente: BETO v1 perdió por truncamiento (ADR-10) -> probar chunk-mean (Chalkidis 2022)
- ADR-11: v1 congelada; resultados nuevos con sufijo v3...v8, nunca sobrescribir

## Slide 17 — Retrieval: de 0.218 a 0.425 en F1@5

| Versión | Estrategia | F1@5 | MRR | nDCG@10 | F1@10 |
|---|---|---|---|---|---|
| v2 inicio | BM25 pasajes (índice sucio) | 0.218 | 0.717 | — | — |
| v3 | BM25 sucio (diagnóstico R) | 0.218 | 0.717 | 0.342 | — |
| v3-limpio | BM25 solo permanentes (S1) | 0.233 | 0.765 | 0.380 | 0.218 |
| v4 | Rerank query completa (falló) | 0.177 | 0.572 | 0.296 | 0.203 |
| v5 | Rerank por chunks top-20 | 0.285 | 0.850 | 0.438 | 0.249 |
| v6 | Híbrido top-50 + rerank | 0.340 | 0.865 | 0.513 | 0.316 |
| **v8** | **Híbrido top-50 + rerank + hubs α=1.5** | **0.425** | **0.950** | **0.620** | **0.379** |

- Default del producto (BM25 + hubs α=0.75): F1@5 0.396 / MRR 0.901 / nDCG@10 0.601
- Contaminación 24,4 % -> 0; limitación: caso 6251 sigue fuera del top-50 (recall 1ª etapa)

## Slide 18 — Hubs de la CP: prior estático adoptado (ADR-15)

- Señal: frecuencia global de citación por artículo CP (497 cánones, top-5: 241/86/13/2/1), normalizada [0,1] estilo IDF
- No-leakage por diseño: estadística de corpus idéntica para toda consulta; PageRank de co-citación descartado (Spearman 0.9828)
- Calibración honesta con mitades A/B (pares/impares) + full-90; rejilla extendida 0 -> 2.0 (óptimo interior)
- Ganadora α=1.5 sobre híbrido top-50 + rerank: F1@5 0.340 -> 0.425, MRR 0.865 -> 0.950, nDCG@10 0.513 -> 0.620, F1@10 0.316 -> 0.379
- Curva full-90 sube hasta 1.5 y cae en 2.0; default del producto con α=0.75

## Slide 19 — BETO chunk-mean: la mejora que no alcanzó (ADR-13)

- Chunk-mean BETO (256 tokens, stride 128, máx 64, mean-pooling; 52,7 chunks/doc; 5,15 h GPU)
- Test: completo 0.2433 / enmascarado 0.2363; gap leakage 0.0070; val monótona 0.1503 -> 0.1935 -> 0.2313
- +17,0 %/+15,8 % vs BETO v1 truncado (0.2080/0.2042): confirma que la derrota era del truncamiento, no del transformer
- Meta >= 0.26 NO cumplida; LogReg v1 0.2870 y C calibrada 0.2980 siguen ganando
- El producto mantiene LogReg; chunk-mean queda documentado y sus pesos guardados como comparativa

## Slide 20 — Fase U y W-b: negativos documentados (ADR-14)

- Fase U (meta macro-F1 test >= 0.32, no cumplida por ninguna variante):
  - A jerárquica 0.2697 / B multietiqueta single-label 0.2500 / **C calibración 0.2980** (+3,8 % vs v1)
  - B en métrica nativa: macro 0.4792 / micro 0.7895 (otra tarea, no comparable)
- Fase W-b EuroVoc (meta: adoptar solo si gana): D taxonomía 0.2625 y E léxico 0.2771 vs C 0.2980 -> NO adoptar
- Lección: el techo está en features/datos, no en la arquitectura; C queda disponible como candidata

## Slide 21 — QA v2: GO con drift cero

- `checks_qa_final.py` v2 (k=10, hubs, modos, integridad) vs control negativo v1 (82 hallazgos, discrimina)
- 10/10 casos E2E aprobados (8 positivos en 3 periodos + Gaitán/Petro negativos)
- Drift clasificador/validación = 0 (misma área top-1 y probabilidad ≡1e-6 que el QA v1); retrieval cambia por diseño (5->10, hubs, rerank)
- md5 canónicos intactos y rollback identificado (zip v1)
- Tiempos: 4-7 s default; 14-18 s rerank (E5 + reranker en cuda)

## Slide 22 — Producto v2

- Default: BM25 permanentes + hubs α=0.75 -> 4-7 s/eval, F1@5 0.396, top-10 con scores, 0 transitorios
- `--rerank`: híbrido top-50 + bge-reranker-v2-m3 (chunks) + hubs α=1.5 -> 14-18 s, F1@5 0.425
- Clasificación: LogReg v1 sin cambios (gates U/V no cumplidos)
- Fase S2: lectura pyarrow por id -> 8,8 a 5,5 s/eval y RAM 8,0 a 1,6 GB
- Equivalencia producto<->benchmark verificada (ids y scores, 3 casos)

## Slide 23 — Cierre del ciclo v2

- Decisiones: ADR-13 (chunk-mean: mejora documentada, no adoptada), ADR-14 (jerárquico/multietiqueta/EuroVoc: no adoptados), ADR-15 (hubs: adoptado; EuroVoc: no)
- Resultado central: F1@5 0.218 (inicio) -> 0.425 (+95 %); ruta default 0.396; +82 % del rerank sobre v3-limpio (0.233 -> 0.425)
- 26 referencias verificadas (6 nuevas del ciclo) y 15 ADRs documentados
- Artefactos: benchmarks v3-v8, m4_* v2, qa_final_v2/, informe `informe_final_ciclo_v2.md`, zip `juris_pln_e2_entrega_v2.zip`
- Reproducir: scripts 06/07/12/14 + `checks_qa_final.py` (v2)
