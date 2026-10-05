# Arquitectura — Juris-PLN

**Fecha:** 2026-10-03. Basada en la auditoría Fase 1 (solo lectura) y en el código vigente.

## 1. Vista general (flujo de datos)

CSV 1,67 GB → Parquet PySpark (script 00) → NER jurídico M2 → etiquetas de bloque M3 →
lemas M1 → modelos M4/M5/M6 → benchmark vínculo CP M7 → producto CLI (M0/M2/M4/M7/M5).

## 2. Módulos (mapa operativo real)

| Módulo | Rol | Dónde | Cifra clave |
|---|---|---|---|
| M0 validación | reglas de providencia | `lib/validacion_entrada.py` | accuracy 1,0 (52 casos) |
| M1 lematización | spaCy `es_core_news_md` | `05_lematizacion.py` | 23.742 docs; 195.959 tipos |
| M2 NER jurídico | regex + ventana | `03_extraccion_entidades.py` | CP 98,71 %; 516.529 aristas |
| M3 etiquetado | bloque mayoritario de citas + fallback | `04_etiquetado_areas.py` | 99,63 % cobertura (silver) |
| M4 clasificación | LogReg TF-IDF lemas enmascarado | `06`/`07` | 0,287 obs / 0,269 fijo; ruido OOF ~32 % |
| M5 red de citas | NetworkX + PageRank | `08_red_citas.py` | 456.321 aristas (84,48 %) |
| M6 diacronía | áreas / log-odds / MTLD | `09_diacronia.py` | MTLD 64,9 → 56,8 |
| M7 vínculo CP | BM25 + denso + rerank + hubs | `11`/`12` | nDCG@10 0,620 (silver) |

## 3. Pipeline de scripts (00-23)

00 migración CSV→Parquet · 01 descarga CP · 02 parseo CP (497 arts, 19 bloques) · 03 NER ·
04 etiquetado · 05 lematización · 06 baseline clásico (+ fases U/C3/A5) · 07 BETO
(truncado/chunk-mean) · 08 red de citas · 09 diacronía · 10 producto CLI · 11/12 benchmarks M7
(12: fases R/S1/T/W y c3 A2/A3/A4) · 13 figura comparativa · 14 precache chunks BETO ·
15 EuroVoc · 17 auditoría de ruido M4 (cleanlab) · 18 H1 · 19 H2 · 20 H2b · 21 pack de
validación humana O1 · 22 pre-anotación LLM · 23 pares del reranker B1 (en curso) ·
`checks_qa_final.py` / `checks_perf_producto.py` · `lib/` (`componentes_producto.py`,
`reranker_bge.py`, `chunks_beto.py`, `validacion_entrada.py`).

## 4. Producto

- CLI `scripts/10_evaluar_sentencia.py` (`--archivo|--id|--texto`, `--salida`, modo `--rerank`);
  5 clases en `lib/componentes_producto.py`: `ValidadorEntrada`, `ExtractorEntidades`,
  `ClasificadorArea`, `VinculoCP`, `ContextoJurisprudencial`.
- `ClasificadorArea` carga `modelos/m4_baseline/logreg_enmascarado.joblib` +
  `tfidf_enmascarado.joblib` **sin calibrar** (`predict_proba` crudo). `construir_resumen()` es
  una síntesis de estado de una línea, no un resumen extractivo/abstractivo.
- **v2.2 (2026-10-04, ADR-18):** la ruta `--rerank` usa **B2** (`modelos/b2_retriever`) como
  denso y **B1** (`modelos/b1_reranker`) como reranker + hubs α=1,5; el default (BM25+hubs
  α=0,75) no cambia. Latencias 4,6-5,5 s default / 13,9-16,3 s rerank; CPU-first en default.

## 5. Modelos

| Componente | Modelo | Estado | Métrica |
|---|---|---|---|
| M4 | LogReg TF-IDF lemas enmascarado | **adoptado** | 0,2870 obs / 0,2691 fijo |
| M4 | BETO truncado / chunk-mean | no adoptado | 0,208 / 0,243 |
| M4 | Calibración C (sesgos) | no adoptada | 0,298 (IC cruza 0) |
| M7 | bge-reranker-v2-m3 + hubs α=1,5 | adoptado (v2; superado por v2.2) | nDCG@10 0,620 |
| M7 | **B2 retriever denso + B1 reranker + hubs α=1,5** | **adoptado (v2.2, ADR-18)** | nDCG@10 0,693 (silver) / 0,783 (gold-LLM) |
| M7 | Embeddings legales ES (ALIA/BGE-M3/modernbert), SINAI reranker, RRF | no adoptados | A2/A3 negativos |
| C2 | ALIA-7B zero-shot (query understanding / rerank listwise) | no adoptado | C2 negativo (Δ −0,198 / −0,217) |
| B1 | Reranker de dominio (fine-tune con citas) | **en curso** | — |
| C2 | ALIA-es-legal-7B (query/rerank listwise) | planificado | — |

## 6. Datos y protocolo de evaluación

- Corpus 23.742 sentencias; CP 497 artículos (384 permanentes + 112 transitorios + preámbulo).
- Splits canónicos M4: 16.557/3.549/3.548 (70/15/15 estratificado, semilla 42; **no** temporal).
- Protocolo nuevo (2026-10-03): `data/splits_m4_temporal.parquet` (16.858/5.092/1.704) y
  **holdout sellado** `data/holdout_m4_bloqueado.parquet` (2.481 docs) — reservado al gate final.
- Benchmark M7: 90 casos (3/año, seed 42) con referencia **silver**; techo F1@5 0,649.
- Caches: índice BM25 CP v3, embeddings E5, chunks BETO, prior de hubs.

## 7. Infraestructura

`/home/carlos/venv` (Python 3.14.7), PySpark 4.2 + Java 17, GPU AMD RX 9070 XT ROCm 7.2 (fp16),
spaCy, transformers, rank_bm25, NetworkX, pyarrow, cleanlab; LLM local vía ollama (11434).

## 8. Convenciones y decisiones clave

ADR-2 anti-leakage doble escenario · ADR-5 weak labels 1/año · ADR-9 pasajes + max-pooling ·
ADR-10 LogReg en el producto · ADR-11/16 versionado congelado con sufijos (`_vN`, `c3_*`) ·
ADR-12 nunca truncar la query del cross-encoder · ADR-13 BETO chunk-mean (con errata
2026-10-03: el producto no usa calibración C) · ADR-15 hubs · ADR-17 gate A sin adopciones.
Un script por experimento; semilla 42; gates con IC 95 % + A/B; registro en `metricas.json` y
`REGISTRO_EXPERIMENTOS.md`.

## 9. Límites conocidos

Caso T-6251/01 (arts. 21/209 fuera del top-50 de la 1ª etapa); techo de recall de la 1ª etapa;
F1@5 topado ~0,65 por diseño; ruido del silver: ~32 % (OOF cleanlab) y acuerdo humano–silver
23,5 % (κ 0,15; validación humana O1, ADR-19: el silver no se usa como gold); probabilidades sin
calibrar (interpretar como ranking/confianza relativa, no como probabilidad calibrada);
resumen no implementado (E1-Q4); referencia de evaluación silver (mini-gold humano B3 de
50 casos confirma v2.2; gold estándar 150 pendiente); test canónico reutilizado (el holdout
nuevo queda para los gates).

Referencias cruzadas: `objetivos.md` (qué) y `AGENTS.md` (operación).
