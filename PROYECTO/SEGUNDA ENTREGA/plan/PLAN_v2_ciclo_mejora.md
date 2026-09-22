# PLAN v2 "Juris-PLN" — Ciclo de Benchmark y Mejora Incremental

Estado: **BLOQUE 1 COMPLETADO 2026-09-10 (checkpoint superado) — BLOQUE 2 APROBADO POR CARLOS, EN EJECUCIÓN**
Orden del bloque 2 (decisión de Carlos 2026-09-10, sigue recomendación del checkpoint):
T-iter3 (recall 1ª etapa) -> V (chunk-mean BETO) -> U (jerárquico) -> W (EuroVoc+hubs) -> X (cierre).
Versionado: implementación actual congelada = **v1**; este ciclo = **v2**.

## Resultados del bloque 1 (congelados, verificados contra JSONs)
| Estrategia | F1@5 | P@5 | MRR | nDCG@10 |
|---|---|---|---|---|
| v2 sucio (inicio) | 0.218 | 0.349 | 0.717 | — |
| v3-limpio bm25 | 0.233 | 0.371 | 0.765 | 0.380 |
| v3-limpio hibrido | 0.263 | 0.416 | 0.751 | 0.420 |
| v4 rerank query completa (falló) | 0.177 | 0.276 | 0.572 | 0.296 |
| **v5 rerank por chunks** | **0.285** | **0.456** | **0.850** | **0.438** |
Metas 2/3 (F1@10 0.249 < 0.30 por techo recall@20 de la 1ª etapa BM25).
Caso 6251: arts 21/209 en posiciones 220/305 del ranking BM25 → fuera del
top-20 → el reranker no puede rescatarlos. ADR-11 y ADR-12 aceptados.

## Decisiones de Carlos (sesion 2026-09-10, vinculantes)
1. Versionado: la implementacion actual queda congelada como v1; este ciclo es v2.
2. Alcance bloque 1: R -> S1 -> S2 -> T y PAUSA antes de U/V para revision de resultados.
3. GPU 100% disponible para el proyecto: reranker y transformer en GPU; toda operacion
   lenta en CPU que sea GPU-compatible va a GPU.
4. Politica de codigo: se modifica SOBRE los archivos existentes (cambia la convencion
   v1 "1 experimento = 1 script nuevo"). Los RESULTADOS v1 son INTOCABLES: los nuevos
   benchmarks van a archivos con sufijo nuevo (_v3, _v4...), nunca se sobreescriben.

## Evidencia SOTA que sustenta el plan (verificada 2026-09-10)
- Chalkidis et al. 2022, "Processing Long Legal Documents with Pre-trained
  Transformers: Modding LegalBERT and Longformer" (arXiv:2211.00974): sliding-window/
  chunk-mean vence al truncamiento en documentos legales largos -> receta para Fase V.
- UA-StatuteRetrieval 2026 (arXiv:2607.11109): 396M citas, 20 anos; la senal de
  co-citation decae (MRR -33% a -47%); los hub articles resisten el decaimiento ->
  valida deriva temporal M7 y sugiere prior de co-citacion (arts 241/86/13 son hubs).
- DoSSIER@COLIEE 2021 (arXiv:2108.03937, ya en estado_arte v1): dos etapas BM25
  pasajes -> re-ranking BERT es la arquitectura ganadora -> Fase T.
- LegalMALR 2026 (arXiv:2601.17692) + Cross-Architecture LLM Ensembles / Feature-Based
  Reranking (arXiv:2607.11400): reranking domina statute retrieval 2026 -> Fase T.
- BEIR (arXiv:2104.08663): nDCG como metrica estandar IR que falta -> Fase R.
- bge-reranker-v2-m3 (HF, apache-2.0, 568M params, multilingue con espanol,
  18.2M descargas): cabe sobrado en 17 GB VRAM -> reranker elegido para Fase T.
- Modelos legales ES en HF: PlanTL RoBERTalex (legal-es gubernamental),
  Narrativa legal-longformer-base-4096-spanish (ventana 4096, solo 29 descargas ->
  riesgo de calidad, queda como comparacion secundaria opcional en Fase V);
  mrm8488 legalectra (7 descargas, descartado).
- Ontologias en espanol: evaluacion documental E1 (EuroVoc/LKIF/ELI/Akoma Ntoso)
  + busqueda arXiv 2026-09-10: NO existe ontologia legal colombiana/constitucional
  lista. Decision: mantener taxonomia derivada de la CP (19 bloques) como eje;
  EuroVoc-es SOLO como features auxiliares si gana en benchmark (Fase W).

## Diagnostico que motiva el ciclo (evidencia en vivo 2026-09-10)
- Caso T-6251/01: NER extrae arts 21 y 209 (ground truth), BM25 recupera 0/2;
  top-5 contaminado con 4 articulos "Transitorio" incluido ID malformado
  "Transitorio ACL01017-8".
- Indice CP: 385 permanentes + 112 transitorios + preambulo = 497; 45 IDs malformados
  (Transitorio ACL01016-N / ACL01017-N, adicionados por Acto Legislativo 1 de 2016).
- Benchmark v2: BM25 F1@5 0.218, P@5 0.349, MRR 0.717 (ranking decente, top-5 ruidoso);
  recall@5 0.183 estructuralmente capado (sentencias citan ~10.7 arts en promedio).
- QA v1 valido estructura, no calidad por caso: el fallo T-6251 no fue detectado.

## FASES DEL BLOQUE 1 (ejecutar en orden)

### Paso 0 — Documentacion de planificacion (PRIMERA ACCION al ejecutar)
- plan/PLAN.md: marcar pipeline/modelos/resultados v1 como CONGELADOS
  (implementacion v1); anadir seccion "Ciclo v2" apuntando a este archivo.
- plan/ESTADO_Y_PENDIENTES.md y plan/contexto.md: registrar ciclo v2 aprobado.
- estado_arte.md + references.bib: anadir refs nuevas de la seccion SOTA de arriba
  (Chalkidis 2022, UA-StatuteRetrieval, LegalMALR, Feature-Based Reranking, BEIR,
  bge-reranker-v2-m3).
- ADR-11 (plan/decisiones/): politica de versionado v1/v2 + cambio de convencion de
  scripts (modificar existentes; resultados v1 intocables, nuevos con sufijo).
- NOTA: este archivo PLAN_v2 ya queda guardado; el Paso 0 formaliza el resto.

### Fase R — Re-benchmark honesto (CPU, ~30 min)
- Archivo: 12_evaluar_vinculo_cp_v2.py (MODIFICAR existente).
- Anadir: nDCG@10 (BEIR); guardar top-10 por caso; diagnostico de contaminacion
  (% transitorios en top-k, IDs malformados en rankings).
- Rerun 90 casos (misma muestra seed 42 para comparabilidad).
- Salida: resultados/vinculo_cp_benchmark_v3.json (v2 intacto) + figura diagnostico.
- Objetivo: cuantificar el dano de transitorios ANTES de arreglarlo (evidencia ADR-12).

### Fase S1 — Limpieza del indice CP (CPU, ~1-2 h)
- 02_parseo_constitucion.py (MODIFICAR): corregir 45 IDs "Transitorio ACL01016-N"
  -> numeracion canonica + metadato del Acto Legislativo (verificar contra texto
  fuente del Senado); regenerar constitucion_articulos.parquet (conservar actual
  como respaldo data/constitucion_articulos_v1.parquet).
- lib/componentes_producto.py (MODIFICAR, clase VinculoCP): BM25 sobre permanentes
  por defecto; transitorios opcionales via flag; recuperar() devuelve top-10;
  invalidar cache BM25 (version de indice nueva).
- 10_evaluar_sentencia.py (MODIFICAR): campo recuperados_bm25 pasa de 5 a 10 con
  scores.
- Benchmark re-ejecutado -> comparar v3 vs v3-limpio (evidencia ADR-12).

### Fase S2 — Producto rapido (CPU, ~1 h)
- lib/componentes_producto.py (MODIFICAR): lectura por pyarrow filters (solo la
  fila del id) en vez de materializar el corpus completo.
- Impacto medido por auditoria: -3.7 s/eval, RAM 8 -> 2 GB.

### Fase T — Reranking dos etapas (GPU, ~2-4 h)
- 12_evaluar_vinculo_cp_v2.py (MODIFICAR): BM25 top-20 -> bge-reranker-v2-m3 en
  GPU (batch 16, fp16) -> top-10; anadir estrategia "bm25_rerank" al benchmark.
- lib/componentes_producto.py (MODIFICAR): VinculoCP con modo reranker opcional
  (GPU cuando este libre; fallback CPU con batch pequeno documentado).
- Descarga unica del reranker ~2.2 GB (huggingface cache).
- Salida: resultados/vinculo_cp_benchmark_v4.json.
- METAS: P@5 0.349 -> >=0.45; F1@10 >=0.30; MRR >=0.80.

### PAUSA — Checkpoint de revision (entregable del bloque 1)
- Tabla comparativa v1 -> v3 -> v3-limpio -> v4 (F1@5/@10, nDCG@10, MRR,
  contaminacion).
- ADR-12 con evidencia (transitorios + reranking dos etapas).
- Demo producto: mismos 3 casos + caso fallido T-6251 (debe pasar de 0/2 a correcto).
- Recomendacion fundada de si U/V valen la pena, con numeros en la mano.
- ESPERAR APROBACION DE CARLOS antes del bloque 2.

## FASES DEL BLOQUE 2 (APROBADAS por Carlos 2026-09-10 tras el checkpoint; ejecutar en este orden)

### T-iter3 — Ampliar recall de la 1a etapa (GPU, ~30-60 min) [PRIMERA]
- Archivo: 12_evaluar_vinculo_cp_v2.py (MODIFICAR).
- Dos variantes nuevas sobre el esquema v5 (rerank por chunks):
  (a) `bm25_rerank_chunks_top50`: candidatos = top-50 BM25 (en vez de 20);
  (b) `hibrido_rerank_chunks_top50`: candidatos = top-50 del hibrido
  (BM25+E5 0.5/0.5), que aporta recall denso que BM25 solo no tiene.
- El rerank por chunks escala lineal: 50 candidatos x 3 pares = 150 pares/caso
  (~7.500 pares en 90 casos, ~5-8 min GPU).
- Salida: `resultados/vinculo_cp_benchmark_v6.json` (version
  "v6-ciclo2-faseT-iter3"). v2/v3/v3_limpio/v4/v5 INTOCABLES.
- METAS: F1@10 >= 0.30 (la meta incumplida); MRR >= 0.85 (no regresar);
  caso 6251 ad-hoc: arts 21/209 deben ENTRAR al top-10 del ranking final
  (estan en pos 220/305 de BM25 — solo alcanzables si el hibrido los sube
  al top-50; medir y reportar).
- Decision tras v6: la mejor estrategia se adopta como recomendada del
  producto (default del flag --rerank); ADR-12 se actualiza con v6.

### Fase V — Transformer largo chunk-mean BETO (GPU, ~6-10 h) [SEGUNDA]
- Archivo: 07_transformer.py (MODIFICAR).
- Metodo Chalkidis 2022 (arXiv:2211.00974): en vez de truncar cabeza+cola
  256+256 (que descarto ~95 % del contenido y explico la derrota de BETO v1,
  ADR-10), dividir el documento en chunks de 256 tokens con sliding window,
  codificar cada chunk, y clasificar con la MEDIA de las representaciones
  (chunk-mean). Doble escenario completo/enmascarado (ADR-2) igual que v1.
- Hiperparametros: mismos del v1 (3 epocas, batch efectivo 32, lr 5e-5) para
  comparabilidad; SOLO cambia la estrategia de longitud. Guard de VRAM como
  en v1 (espera si GPU ocupada).
- Salidas: `resultados/m4_transformer_chunks.json` (NUEVO; m4_transformer.json
  v1 INTOCABLE), modelos en `modelos/m4_transformer_chunks/beto_{completo,enmascarado}/`.
- META: macro-F1 test >= 0.26 (v1: 0.2080; LogReg a vencer: 0.2870).
- Comparacion secundaria OPCIONAL (solo si el tiempo sobra tras la meta):
  Narrativa/legal-longformer-base-4096-spanish (29 descargas, riesgo calidad).
- NOTA: si chunk-mean BETO supera 0.2870 (LogReg), ADR-10 se revisa y el
  ClasificadorArea del producto cambia en la Fase X; si no, LogReg se mantiene.

### Fase U — Clasificacion jerarquica/multietiqueta (CPU+GPU, ~4-6 h) [TERCERA]
- Archivos: 04_etiquetado_areas.py + 06_baseline_clasico.py (MODIFICAR).
- 16 clases planas -> 2 niveles: (a) clasificador grueso por GRUPOS (los 19
  bloques tematicos de la CP agrupados en ~6-7 macro-grupos coherentes);
  (b) clasificador fino condicionado al grupo predicho (o multietiqueta por
  bloques citados con umbral por clase optimizado en VALIDACION, nunca test —
  skill threshold-optimization).
- Splits canonicos de data/splits_m4.parquet SIN tocar (16.557/3.549/3.548).
- Salidas: `resultados/m4_jerarquico.json` (NUEVO; m4_baseline.json INTOCABLE).
- META: macro-F1 test >= 0.32 (LogReg plano v1: 0.2870).
- Si la meta se cumple, el ClasificadorArea del producto se actualiza en X.

### Fase W — Extras con evidencia (CPU, ~2-4 h) [CUARTA — INCLUIDA por decision de Carlos]
- (a) **Prior de co-citacion (hubs)**: usar el PageRank/centralidad de los
  articulos CP en la red de citas (arts 241, 86, 13 son hubs; UA-Statute
  arXiv:2607.11109: los hubs resisten el decay temporal) como senal aditiva
  en el ranking del benchmark 12: score_final = score_base + alpha *
  pagerank_norm(articulo), con alpha calibrado en los 90 casos del benchmark
  (grid 0.0-0.5). SOLO se adopta si mejora F1@5/nDCG@10 de la mejor estrategia
  vigente; si no, queda documentado como negativo.
- (b) **EuroVoc-es como features auxiliares** en 06: mapear los 19 bloques
  de la CP a las categorias de EuroVoc (si el mapeo es viable) y anadir la
  senal como feature al clasificador. SOLO si gana en benchmark; si el mapeo
  no es viable o no gana, queda documentado como negativo (la evaluacion
  documental E1 ya anticipaba cobertura parcial).
- Ambas son experimentales: se registran en metricas.json gane o pierda.

### Fase X — Cierre v2 [QUINTA]
- Producto: adoptar las mejoras ganadoras en 10_evaluar_sentencia.py +
  lib/componentes_producto.py (default del rerank segun v6; ClasificadorArea
  segun U/V; prior de hubs segun W). Version v2 en docstrings.
- QA de regresion: actualizar checks_qa_final.py a v2 (k=10, rerank, nuevos
  artefactos) y re-ejecutar (10 casos E2E + metricas canonicas sin drift
  respecto a los valores v2 documentados).
- ADR-13 (chunk-mean transformer, revisa ADR-10 si gana), ADR-14
  (jerarquico/multietiqueta), ADR-15 (hubs/EuroVoc, con evidencia gane o
  pierda). Actualizar ADR-12 con v6.
- Informe/deck v2: actualizar documento_e2/ con seccion "Ciclo v2" (NO
  reescribir el informe v1; anadir). Zip v2 nuevo (juris_pln_e2_entrega_v2.zip).
- Ingesta Graphiti del ciclo completo + reindex codebase-memory.
- Git commit+PR: SOLO bajo pedido de Carlos.

## Riesgos y mitigaciones
- Descarga reranker ~2.2 GB: una sola vez, cache HF.
- Si la GPU vuelve a estar ocupada (UKAN externo): T degrada a CPU (documentado,
  no bloquea); guard de VRAM como en 07_transformer v1.
- 45 IDs transitorios: verificar mapeo contra fuente Senado antes de regenerar.
- legal-longformer-es sin respaldo de comunidad: solo comparacion, nunca primario.

## Checklist de ejecucion del bloque 2 (aprobado 2026-09-10)
1. T-iter3: implementar top-50 BM25 + top-50 hibrido -> revision -> run v6.
2. Fase V: chunk-mean BETO en 07 -> revision -> entrenamiento GPU (background).
3. Fase U: jerarquico/multietiqueta en 04+06 -> revision -> ejecucion.
4. Fase W: prior hubs (alpha calibrado) + EuroVoc-es auxiliar -> revision -> ejecucion.
5. Fase X: cierre (producto, QA v2, ADR-13/14/15, informe/deck/zip v2, Graphiti).
6. Git: SOLO bajo pedido de Carlos.
