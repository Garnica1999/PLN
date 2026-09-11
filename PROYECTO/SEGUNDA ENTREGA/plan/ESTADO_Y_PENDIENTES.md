# PLAN MAESTRO E2 "Juris-PLN" — Estado y pendientes
Actualizado: 2026-09-11 (ciclo v3 en ejecución) · Fuente de verdad: `plan/PLAN.md` (v1 congelada) + `plan/PLAN_v2_ciclo_mejora.md` (ciclo v2) + `plan/PLAN_v3_recuperacion_robusta.md` (ciclo v3).

## CICLO v3 — EN EJECUCIÓN (aprobado 2026-09-11) — "Recuperación robusta"
Meta: nDCG@10 >= 0.70 con gold humano, sin caídas por periodo (>=0.60), adopciones con IC 95% bootstrap.
- [ ] Paso 0 — Congelar v2 (git commit local hecho; push/PR requiere credenciales de Carlos) + ADR-16 + `REGISTRO_EXPERIMENTOS.md` (retro-relleno v1/v2 completo).
- [ ] Sprint A — A1 métricas honestas (recall@k + IC bootstrap) -> A2 embeddings legales ES (ALIA/BGE-M3/modernbert) -> A3 reranker legal ES (SINAI) -> A4 RRF + profundidad -> A5 clasificador (C + ensamble). Gate: Δ > ruido + consistente A/B.
- [ ] Sprint B — Fine-tune reranker con citas + hard negatives (sin las 90 del benchmark) -> retriever denso de dominio -> mini-gold 40-60 casos (Carlos) + análisis de errores.
- [ ] Sprint C — Gold estándar 150 -> LLM ALIA-7B (query understanding/rerank, premisas en PLAN_v3) -> corpus 2022-2026 -> producto v3 + QA v3 -> documentación integral + cierre.
- Documentación integral: `REGISTRO_EXPERIMENTOS.md` (datos de todos los experimentos, premisas) + informes con "Métodos elegidos, premisas y criterios de adopción".

## CICLO v2 — COMPLETADO 2026-09-11 (fases R→X, QA v2 GO, documentación cerrada)
Implementación v1 CONGELADA (resultados intocables). Resultados finales del ciclo v2:
- **Retrieval (mejor estrategia, 90 casos)**: F1@5 0.218→**0.425**, MRR 0.717→**0.950**, nDCG@10→**0.620**, F1@10→**0.379** (`hibrido_rerank_chunks_top50` + prior hubs α=1.5, v8). Default BM25+hubs α=0.75: F1@5 0.396. Contaminación transitorios 0; 45 IDs ACL corregidos; producto v2 integrado (default 4-7 s / rerank 14-18 s) con equivalencia verificada.
- **Clasificación**: LogReg v1 se mantiene (0.2870); Fase U C calibrada 0.2980 (+3.8%, meta 0.32 no); BETO chunk-mean 0.2433/0.2363 (+17%/+16% vs v1 truncado, meta 0.26 no); EuroVoc no adoptado.
- **QA v2**: GO (10/10 E2E, drift clasificador 0, integridad v2). `resultados/qa_final_v2/VEREDICTO_QA.md`.
- **Entregables**: `documento_e2/informe_final_ciclo_v2.md` (2.499 palabras), deck 23 slides, **`PROYECTO/juris_pln_e2_entrega_v2.zip` (31,1 MiB, 125 entradas)**, ADRs 11-15.
- **Incidentes resueltos**: deadlock DataLoader (Python 3.14+fork) → cache de chunks + num_workers=0; crash `scaler.update()` corregido; guards VRAM.
- **Pendiente único**: git commit+PR (SOLO bajo pedido de Carlos).

## Producto norte
Sistema que, dada una sentencia de la Corte Constitucional: (1) valida que sea
providencia de la Corte, (2) determina su área temática constitucional,
(3) identifica los artículos de la CP que la sustentan (robusto a deriva
temporal), (4) la sitúa en su contexto jurisprudencial (precedentes y
evolución 1992-2021).

## Fases COMPLETADAS (13 de 15)

| Fase | Contenido | Resultado clave |
|---|---|---|
| A | Estructura SEGUNDA ENTREGA + PLAN.md + ADRs 1-5 + hardware | Docs base listos |
| B | SOTA ampliado ES+EN (nlp-specialist) | 10 refs nuevas verificadas (total 20 en .bib): COLIEE/DoSSIER, Fink 2023, Samy 2021 (NER es), MTLD, Hamilton 2016, Fowler 2007, BETO, E5 |
| C | Migración CSV→Parquet (ADR-7) | PySpark 4.2 + Java 17 (Java 26 incompatible); 23.742 filas, limpieza 100 % idéntica E1; 1.56→1.47 GB |
| D | EDA extendido (sección 9 notebook E1, celdas 0-23 intactas) | N-gramas (top tri "accion de tutela" 344.893), colocaciones (gana LLR 75 %), lema-vs-stem (lemas 87.3 %, stems 54.2 %), POS (NOUN>ADP>DET), MTLD 61.03, granularidad (sentencia completa; subsentencia 4.64 % no supera criterio) |
| E | Review + QA del EDA | Reviewer APROBADO (5/5); QA APROBADO C/OBS (figuras 6/6, logs 12/12, consistencia OK) |
| F | ADRs 6-8 con números del corpus | ADR-6 (crudo para BPE, lemas para clásicos), ADR-7 (aceptado), ADR-8 (sentencia completa) |
| G | Constitución: descarga + parseo (M3) | 497 artículos (384 perm + 112 transitorios + preámbulo) → 19 bloques temáticos; spot-check 5/5 |
| H | NER jurídico corpus completo (M2) | Precisión muestral 1.00/1.00/1.00/0.90; cobertura CP 98.71 %; 516.529 aristas sin self-loops |
| I | Etiquetado de áreas (M3) | 99.63 % asignadas (cita CP 98.71 %, fallback 0.92 %); top: derechos_fundamentales 42.5 % |
| J | Lemas por documento (M1) | 23.742/23.742; vocabulario 195.959 tipos = 100 % coincidente con referencia EDA; data/lemas_por_doc.parquet |
| K | M0 validación + M7 benchmark retrieval (v1) | M0 accuracy 1.0 (30 pos + 22 neg discursos); v1: cita_exacta 0.70 (circular) > BM25 0.119 > E5-CPU 0.090 |
| K-2 | Recalibración M7 (petición de Carlos) | BM25 F1@5 0.119→**0.218 (+84 %)**, MRR 0.717; E5 GPU 0.128; híbrido 0.209; 90 casos (3/año); ADR-9: ventana corta REFUTADA con evidencia (1ª cita a mediana 41 % del texto); mejora real = pasajes por artículo + max-pooling chunks + stopwords jurídicas + bigramas LLR |
| L1 | M4 baseline clásico | Splits 70/15/15 UNA VEZ (16.557/3.549/3.548); TF-IDF lemas 100k; LinearSVC 0.2654/0.2657, LogReg 0.2852/**0.2870** (completo/enmascarado); accuracy ~59 % |
| L2 | Auditoría leakage (gate crítico) | Reviewer RECHAZÓ enmascarado v1 (0.7 CITA/doc vs 10.7 citas reales) → corregido (núcleos con tilde, romanos estrictos, series) → v2 gap −0.0018 (enmascarado incluso mejor); prueba de oro: features de cita solas → F1 0.0611 ≈ azar → SIN leakage activo |
| M | M5 red de citas + M6 diacronía | Grafo 23.742 nodos / 456.321 aristas (84.5 % resueltas), gigante 67.6 %; PageRank top: C-224/1992, SU-2716/1997 (landmarks fundacionales); evolución: cita precedentes cada vez más recientes. Áreas: principios +20.2 pp, derechos_fund −24.2 pp. Léxico: incoder/rupd→covid/jep/coronavirus. MTLD 64.9→56.8 (prosa menos diversa) |
| N | PRODUCTO end-to-end | `10_evaluar_sentencia.py` (CLI --archivo/--id/--texto) + `lib/componentes_producto.py`; JSON: validación, ponente, área top-3 con prob, arts CP extraídos+BM25, citas con PageRank/área, resumen. Demo: T-293/97 (true, 0.943), C-002/98 (true, 0.934), discurso (false 0.0). ~9 s/eval |

## PENDIENTE — ACTUALIZADO 2026-09-10: TODO COMPLETADO excepto git

### 1. Fase L3 — Transformer BETO — COMPLETADA 2026-09-09 19:05
- Guard de VRAM funcionó: esperó 3h05m (37 intentos) hasta que el UKAN
  externo liberó la GPU; arrancó solo a las 15:29.
- Fine-tuning BETO (bert-base-spanish-wwm-cased), 3 épocas, batch efectivo
  32, truncamiento cabeza+cola 256+256 (ADR-3), 6h40m totales GPU.
- Resultados: completo macro-F1 test **0.2080** (acc 51.7 %), enmascarado
  **0.2042** (acc 51.3 %), gap leakage 0.0039 (consistente con L2: sin
  leakage). Curvas sanas (val 0.13→0.19→0.21 en ambos escenarios).
- **BENCHMARK 2×2 FINAL (macro-F1 test)**:
  | Modelo | Completo | Enmascarado |
  |---|---|---|
  | LogReg+TF-IDF lemas (M4) | 0.2852 | **0.2870** |
  | BETO fine-tuned (L3) | 0.2080 | 0.2042 |
- **Veredicto: LogReg GANA** → el producto conserva ClasificadorArea con
  LogReg (sin cambios). BETO pierde por diseño del corpus: docs ~11k
  palabras vs ventana 512 tokens (~95 % del contenido descartado por el
  truncamiento), mientras TF-IDF ve el documento completo. Documentado en
  ADR-10. Modelos BETO guardados en modelos/m4_transformer/ por
  reproducibilidad.

### 2. QA final del producto — COMPLETADO 2026-09-10: **GO**
- ml-qa-engineer: 10 casos E2E estratificados (8 pos 2/periodo + 2 neg) —
  10/10 OK (áreas coherentes, 120 citas cruzadas contra red sin errores,
  <30 s). 16/16 métricas canónicas verificadas sin drift. Checklist GO:
  ruff, docstrings, splits intactos, modelos cargables, ADRs 1-10,
  rollback trazable. Riesgos residuales documentados (3). Salidas:
  resultados/qa_final/ (10 JSON + VEREDICTO_QA.md) + scripts/checks_qa_final.py.

### 3. Auditoría de rendimiento — COMPLETADA 2026-09-10: OK académico
- ml-performance-auditor: 8-12 s/sentencia (cuello: inicialización ~75 % —
  lectura corpus completo + spaCy lazy; NO algorítmico). Inferencia CPU-only
  confirmada (0 bytes GPU). RAM pico 8.1 GB. Escalabilidad lineal benigna.
  Top-3 optimizaciones documentadas (pyarrow filters, modo lote, BM25
  vectorizado). Salidas: resultados/qa_perf/ + AUDITORIA_RENDIMIENTO.md +
  scripts/checks_perf_producto.py.

### 4. Informe + deck + README — COMPLETADOS 2026-09-10
- scientific-writer: documento_e2/informe_final_e2.md (3.773 palabras,
  CRISP-DM completo, 8 figuras, 20 refs, 40/40 cifras auditadas contra
  JSON fuente), deck_e2.md (15 slides), README_ENTREGA.md (reproducción
  00→12, venv, Java 17).

### 5. Zip de entrega — COMPLETADO 2026-09-10
- **PROYECTO/juris_pln_e2_entrega.zip** (29 MB, 93 archivos): plan/ (con
  ADRs 1-10), scripts/ 00-12 + lib/, resultados/ (sin checkpoints
  intermedios regenerables), figuras/ (9), documento_e2/ (3 docs),
  modelos/m4_baseline/ (4 joblib). Excluidos: data/ (regenerable con
  00-02), checkpoints lematización, cache BM25, modelos BETO (840 MB,
  descargables de HF).

### 6. Único pendiente — Git (SOLO bajo pedido de Carlos)
- Commit + PR (flujo E1: rama → push → PR → merge → borrar rama remota).
  NADA commiteado aún por acuerdo (git solo bajo pedido).
- Opcional: demo interactiva (streamlit).

## Decisiones ADR registradas (plan/decisiones/)
1. Etiquetado de área vía Constitución · 2. Anti-leakage doble escenario ·
3. Docs largos: truncamiento cabeza+cola + chunk-mean · 4. Scripts SOLID +
   notebook E1 intacto · 5. Vínculo CP: weak labels + 1/año · 6. M4: crudo
   para BPE, lemas para clásicos · 7. PySpark+Parquet · 8. Granularidad:
   sentencia completa · 9. Retrieval: pasajes + max-pooling (ventana corta
   refutada).

## Incidentes y lecciones (para el informe)
- Java 26 incompatible con Spark 4.2 → Java 17 (documentado en ADR-7).
- 3 muertes silenciosas por RAM al competir con jobs externos (llama-server,
  UKAN): mitigado con n_process reducido, checkpoints por id y reanudación
  automática (funcionó 3 veces); LightGBM-sobre-TF-IDF sustituido por LogReg.
- spaCy max_length 1M < doc máx 3M → nlp.max_length=5M + guard por doc.
- Enmascarado v1 inválido (detectado por auditoría adversarial L2): los
  números arábigos ya no existen en los lemas (is_alpha) y los núcleos de
  cita llevan tilde ("artículo") → corregido en v2.
- E5 en CPU en K v1 (GPU ocupada); K-2 ya en GPU.

## Cifras canónicas del sistema (fuente: resultados/*.json)
- Corpus: 23.742 sentencias (T 17.099 / C 6.246 / SU 397), 1992-2021.
- Etiquetado: 99.63 % cobertura, 16 clases efectivas.
- M4 clásico: LogReg macro-F1 0.2870 (enmascarado), gap leakage ≈ 0.
- M7 retrieval: BM25 v2 F1@5 0.218 / MRR 0.717 (vs cita-exacta 0.649 ref).
- M0: accuracy 1.0 (52 casos: 30 positivos + 22 negativos).
- Producto: ~9 s por evaluación end-to-end.
