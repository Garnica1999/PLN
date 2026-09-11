# PLAN v3 — "Recuperación robusta + documentación integral" (ciclo v3)

Estado: **APROBADO por Carlos 2026-09-11** · Ejecución: Paso 0 -> Sprint A -> Sprint B -> Sprint C.
Mantenimiento: este archivo se actualiza paso a paso (checkboxes y secciones de estado al cierre de cada experimento).

## Decisiones de Carlos (2026-09-11, vinculantes)
1. **Gold robusto**: mini-gold anotado de 40-60 casos (Sprint B) + gold estándar de 150 casos (Sprint C);
   relevancia graduada (2=fundamento, 1=mencionado, 0=no), guía de anotación + herramienta asistida.
2. **Congelar v2** con git (Paso 0; commit local + push/PR a cargo de Carlos por credenciales).
3. **LLM local ALIA-7B en C2**, con justificación explícita y premisas (sección propia abajo).
4. **Documentación integral**: registrar todos los datos de todos los experimentos, los métodos elegidos
   y las premisas bajo las que se eligieron (workstream transversal + cierre).

## Definición de éxito ("robusto")
- nDCG@10 >= 0.70 con gold humano (hoy 0.620 con gold NER) y F1@10 >= 0.42 (hoy 0.379).
- nDCG@10 >= 0.60 en CADA periodo (1992-2001 / 2002-2011 / 2012-2021), sin caídas.
- Cada adopción supera el IC 95% bootstrap y es consistente en las mitades A/B (anti-sobreajuste).
- Validación contra gold anotado (no solo gold NER).
- Producto: latencia <= 10 s default / <= 30 s con rerank/LLM; QA v3 GO.

## Estado de partida (ciclo v2 congelado)
- Retrieval v8: `hibrido_rerank_chunks_top50` + hubs alpha=1.5 = F1@5 0.425 / MRR 0.950 / nDCG@10 0.620 / F1@10 0.379;
  default BM25+hubs alpha=0.75 = F1@5 0.396 / MRR 0.901 / nDCG@10 0.601.
- Limitantes: recall de 1a etapa (caso 6251: art. 21 en pos. 220/305), reranker genérico (no de dominio),
  gold débil (NER), F1@5 topado ~0.65 por diseno (~10.7 artículos gold/caso).
- Clasificación: LogReg v1 0.2870; Fase U C 0.2980 (disponible, no adoptada); BETO chunk-mean 0.2433/0.2363.
- Infra: cache chunks BETO (1,3 GB), caches BM25 v3 + E5, benchmark 90 casos (3/año seed 42),
  splits canónicos (16.557/3.549/3.548), producto v2 + QA v2 GO.

## Justificación del LLM legal (ALIA-7B) — C2
**Por qué:**
- Causa raíz medida: la consulta es una sentencia completa (~68k caracteres / ~11k palabras);
  el max-pooling por chunks diluye la señal (caso 6251). Un LLM extrae el problema jurídico y los
  hechos relevantes -> consulta corta y densa que ataca el techo de recall de 1a etapa.
- SOTA 2026: los pipelines ganadores de statute retrieval combinan comprensión de consulta + reranking
  con LLM (LegalMALR arXiv:2601.17692; arXiv:2607.11400 COLIEE). Patrón dominante, no apuesta especulativa.
- Reranking listwise: razona sobre el top-20 completo con semántica jurídica (fundamento vs mención),
  alineado con el gold graduado de B/C.
- ALIA-7B-es-legal: Apache-2.0, 100% local (sin API, sin fuga de datos), GGUF Q4 (~4 GB) en llama-server.

**Premisas:**
1. Adopción solo con evidencia: debe superar nDCG@10 del pipeline vigente con IC bootstrap + A/B + gold;
   si no, se documenta negativo y no entra al producto (misma disciplina que el fallo v4 del reranker).
2. Complementa, no sustituye: 1a etapa (BM25+denso+hubs) intacta; el LLM solo (a) extrae consulta
   estructurada o (b) reordena el top-20.
3. Modo opcional del producto: el default rápido (<=10 s) jamás carga el LLM; flag explícito (<=30 s/eval),
   con caché de salidas, temperatura 0 y semilla fija.
4. Riesgo de dominio declarado: ALIA es administrativo español, no constitucional colombiano -> zero-shot
   primero; si el reranking jurídico no transfiere, se limita a extracción de hechos/problema (mas
   transferible) y/o LoRA con datos propios.
5. Costo asumido: 20-30 s/eval local aceptable para producto académico; no se usa en lotes masivos.

## Workstream de Documentación (transversal + cierre)
| Entregable | Contenido |
|---|---|
| `plan/REGISTRO_EXPERIMENTOS.md` (NUEVO) | Todos los experimentos de v1/v2/v3: ID, fase, hipótesis, premisa, diseño, métricas exactas, decisión, artefacto. Retro-relleno v1/v2 + mantenimiento continuo en v3. |
| `documento_e2/informe_final_ciclo_v3.md` (NUEVO) | Informe del ciclo (~2.500 palabras) con sección "Métodos elegidos, premisas y criterios de adopción". |
| Actualizaciones | `plan/PLAN.md` (cronología), `ESTADO_Y_PENDIENTES.md`, `contexto.md`, `estado_arte.md`, `deck_e2.md`, `README_ENTREGA.md`, ADRs 16-20. |
| Auditoría de cifras | Cada cifra de cada informe se audita contra `metricas.json`/JSONs fuente. |
| Criterios metodológicos | Documentados y citados: evidencia SOTA + benchmark propio; adopción solo si Δ > ruido (IC 95% bootstrap) y consistente A/B; nunca calibrar en test; versiones congeladas con md5; negativos documentados con igual rigor. |

## Sprints

### Paso 0 — Congelar v2 (EN EJECUCION)
- [x] Guardar este plan + ADR-16 + REGISTRO_EXPERIMENTOS (retro-relleno v1/v2).
- [x] .gitignore: excluir data/, modelos/, lematizacion/, evidencia crash (pesados).
- [x] Commit local en rama `feature/e2-congelamiento-v2`.
- [ ] Push + PR (requiere credenciales de Carlos).

### Sprint A — "Base honesta + modelos legales ES + fusión" (~5-7 días)
- [ ] A1 Métricas honestas: recall@k (10/20/50/100/200/384) por estrategia, IC 95% bootstrap
      (1000 remuestreos) para nDCG@10/MRR/F1@10, desglose por periodo y longitud de caso.
      Salida `resultados/c3_diagnostico_recall.json` + figuras.
- [ ] A2 Embeddings legales ES (parametrizar `--embedding-model`): E5 (ref) vs ALIA-MrBERT-legal (150M)
      vs BGE-M3-Legal-Spanish (568M) vs modernbert-legal (149M) vs bge-m3-legal (568M);
      + rerank chunks + hubs alpha=1.5, candidatos top-50. Salida `vinculo_cp_benchmark_v9.json`.
- [ ] A3 Reranker legal ES (parametrizar `--reranker-model`): SINAI ALIA reranker (150M) vs
      bge-reranker-v2-m3 (568M) vs RRF de ambos. Salida `c3_a3_rerankers.json`.
- [ ] A4 Fusión y profundidad: RRF (k=60) vs mezcla 0.5/0.5; candidatos top-50 vs 100 vs 200 (según A1).
- [ ] A5 Clasificador (secundario): calibración C + ensamble LogReg+BETO chunk-mean + class weights.
      Salida `m4_clasificador_c3.json`.
- Gate A: adoptar al producto (v2.1 + QA corta) todo ganador con Δ nDCG@10 > ruido bootstrap y consistente A/B.

### Sprint B — "Modelos de dominio entrenados con citas" (~1,5-2 semanas)
- [ ] B1 Fine-tune reranker: pares (chunk sentencia -> pasaje CP citado) positivos; hard negatives
      (top-BM25/E5 no citados + co-citados M5 no citados). Base: mejor de A3.
      ANTI-LEAKAGE: excluir las 90 sentencias del benchmark del train; splits canónicos.
      Salida `vinculo_cp_benchmark_v10.json`.
- [ ] B2 Retriever denso de dominio: contrastive (MultipleNegativesRankingLoss) del mejor embedding de A2.
      Salida `vinculo_cp_benchmark_v11.json`.
- [ ] B3 Mini-gold 40-60 casos (Carlos, herramienta `16_anotador_gold.py`) + análisis de errores
      categorizado (recall 1a etapa / reranker / gold ruidoso / NER). Salidas `gold_mini.json` +
      `c3_analisis_errores.md`.
- [ ] B4 BETO chunk-mean++ (opcional: más épocas / max_chunks 96).
- Gate B: nDCG@10 >= 0.66 y recall@50 +5 pp, con gold_mini como control cruzado; adopción v2.2 + QA.

### Sprint C — "Gold estándar + LLM legal + robustez temporal" (~3-4 semanas)
- [ ] C1 Gold estándar 150 casos (estratificado periodo/dificultad; guía + doble anotación parcial si hay
      apoyo; nDCG@{5,10} graduado). Re-evaluar todo contra gold.
- [ ] C2 LLM ALIA-7B (premisas arriba): (a) query understanding (hechos/problema jurídico);
      (b) reranking listwise top-20. Gate: nDCG@10 >= 0.70, latencia <= 30 s, fallback documentado.
- [ ] C3 Corpus 2022-2026: extensión + deriva temporal; reentrenar si aporta.
- [ ] C4 Producto v3 + QA v3 (checks actualizados) + ADRs 16-20.
- [ ] C5 Documentación integral (tabla del workstream) + informe/deck/zip v3 + Graphiti + reindex.
- [ ] C6 Cierre: reporte final y comparativa v1 -> v2 -> v3.

## Riesgos y mitigaciones
- Sobreajuste al benchmark de 90 -> protocolo A/B + IC bootstrap + gold; jamás calibrar en test.
- Dominio España vs Colombia -> zero-shot primero; fine-tuning propio como respaldo.
- Fuga de datos (benchmark comparte corpus) -> excluir las 90 sentencias en B1/B2; splits canónicos.
- Cuello de botella humano (anotación) -> mini-gold primero; herramienta asistida.
- GPU compartida -> guards VRAM; LLM vía llama-server existente.
- Métricas saturadas -> nDCG@10/MRR como norte (F1@5 topado ~0.65).

## Recursos verificados (2026-09-11)
| Recurso | Tamaño | Licencia | Uso |
|---|---|---|---|
| SINAI/ALIA-MrBERT legal embeddings | 150M | Apache-2.0 | 1a etapa (A2) |
| SINAI/ALIA-MrBERT legal reranker | 150M | Apache-2.0 | Rerank (A3/B1) |
| BGE-M3-Legal-Spanish / modernbert-legal / bge-m3-legal | 568M/149M/568M | Apache-2.0 | 1a etapa (A2) |
| ALIA-es-legal-7B-Instruct (+GGUF) | ~14 GB / Q4 ~4 GB | Apache-2.0 | LLM (C2) |
| Disco 409 GB libres · GPU 17,1 GB · HF cache 4,1 GB | — | — | — |

## Flujo por experimento
ml-coder implementa -> ml-reviewer aprueba -> ml-trainer ejecuta y registra (`metricas.json` + `contexto.md`
+ REGISTRO_EXPERIMENTOS) -> gate del orquestador -> ml-qa-engineer en adopciones -> scientific-writer al cierre.
Convención ADR-16: modificar archivos existentes; resultados con sufijos nuevos (`_v9+`, `c3_*`); v2 intacta.
