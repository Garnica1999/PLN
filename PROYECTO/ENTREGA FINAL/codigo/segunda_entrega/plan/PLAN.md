# PLAN MAESTRO — E2 "Juris-PLN" (aprobado por Carlos 2026-09-08)

> **CONGELADO — implementación v1 (2026-09-10).** El pipeline, modelos, scripts
> 00-12 y resultados descritos abajo corresponden a la implementación v1,
> entregada y auditada (QA GO, informe, deck, zip). NO se reescriben: los
> valores v1 en `resultados/*.json` son INTOCABLES. El ciclo de mejora actual
> es **v2**, especificado en `plan/PLAN_v2_ciclo_mejora.md` (aprobado
> 2026-09-10): benchmark v3 honesto -> limpieza índice CP -> producto rápido
> -> reranking dos etapas (GPU) -> checkpoint de pausa; bloque 2 (U/V/W/X) solo
> tras aprobación. Convención v2: se modifica SOBRE archivos existentes;
> resultados nuevos con sufijo `_v3`, `_v4`... (ADR-11).

Proyecto PLN (Javeriana) — sentencias Corte Constitucional Colombia 1992–2021.
Corpus: `../PRIMERA ENTREGA/data/Juris1992_2022.csv` (1.67 GB, 23.742 sentencias).
Entorno: `/home/carlos/venv` · Estado por fase (se actualiza al terminar cada fase).

## Producto norte
Sistema que, dada una sentencia: (1) valida que sea providencia de la Corte,
(2) determina su área temática constitucional, (3) identifica los artículos de
la CP que la sustentan (robusto a deriva temporal), (4) la sitúa en contexto
jurisprudencial (precedentes + evolución 1992–2021).

## Estado por fase
- [ ] **Fase A — Estructura + specs + ADRs 1-5**: COMPLETADA 2026-09-08 (este documento + decisiones/).
- [ ] **Fase B — SOTA ampliado ES+EN** (nlp-specialist): sección E2 en `estado_arte.md` + 6-10 refs nuevas verificadas al `.bib` + `plan/contexto_sota.md`.
- [x] **Fase C — ADR-7 + migración Parquet**: COMPLETADA 2026-09-08 por ml-coder. PySpark 4.2.0 local[*] driver 8g; Java 26 FALLA (ClassNotFoundException Cleaner) → OK con Java 17 existente (`/usr/lib/jvm/java-17-openjdk`); sin apt/sudo → documentado. 23.750→23.742 filas (8 excluidas: 6 autos + 2 HTML, posiciones coinciden). CSV 1.561 GB → Parquet Hive 1.465 GB (341 archivos tipo=/ano=). 237.5 s, RAM pico 7.78 GB. Limpieza idéntica E1 verificada 100 % (23.742/23.742 `n_palabras_clean`). Ruff OK. PRIMERA ENTREGA intacta.
- [x] **Fase D1 — Script background**: COMPLETADA 2026-09-08 por ml-coder. `scripts/eda_ext_lemas_stems_pos.py` (spaCy crudo + Snowball + MTLD 0.72, checkpoints c/500 docs, reanudable) corriendo con nohup (PID 393782, ~6.6 docs/s, ETA ~56 min). Desviación: n_process 12→6, batch 64→16 (OOM forkserver con 12 workers); mismo modelo y métricas.
- [x] **Fase D1-fix — max_length**: COMPLETADA 2026-09-08. Caída por doc 1.4M chars > spaCy max_length 1M (E088; doc máx corpus 3M id 16989). Fix: nlp.max_length=5M + guard try/except por doc. Relanzado PID 48750, reanudó en 1500 → 3500, 0 omitidos.
- [x] **Fase D2 — Sección 9 (partes rápidas)**: COMPLETADA 2026-09-08 por ml-coder. Backup md5 OK; celdas 0-23 idénticas; 18 celdas nuevas (24-41), 0 errores, ~14 min. 9.1: top bi "corte constitucional" 286.786, tri "accion de tutela" 344.893. 9.2: gana likelihood-ratio (75 % fórmulas válidas vs freq 55 %, PMI 0 %) → LLR para M2/M4. 9.6 (600 docs): párrafo = sentencia (sin doble salto en corpus); subsentencia 4.64 % con marcador → NO supera criterio +15 % (decide ADR-8). Lectoras 9.3/9.3b/9.5 skip elegante. Figuras + CSVs + verificacion_colocaciones.md + nota_ontologias.md generados.
- [x] **Fase D3 — Celdas lectoras**: COMPLETADA 2026-09-08 por ml-coder. 42/42 celdas 0 errores; celdas 0-23 idénticas; figuras viejas 5/5 bit-idénticas. Nuevas: fig_lemma_vs_stem/mtld/pos_distribucion.png + lemma_vs_stem.csv + morfologia_resumen.csv. Lema-vs-stem: lemas 87.3 %, stems 54.2 %; spacy 5227 s vs stems 1171 s; muestra 50: lema conserva (derechos→derecho), stem trunca (derech). Conclusión: M4 con BPE sin normalizar; si se normaliza, lema. POS: NOUN>ADP>DET>ADJ; C lleva VERB en top-5 (vs PROPN en T/SU). MTLD: T 61.63 > C 59.54 > SU 58.88. metricas.json idéntico salvo tiempo + nota restaurada.
- [x] **Fase E — Review + QA**: COMPLETADA 2026-09-08. ml-reviewer APROBADO (5/5 OK, sin bugs). ml-qa APROBADO CON OBSERVACIONES (6/6 figuras válidas + 5 viejas bit-idénticas; 12/12 logs legibles ~60k líneas; consistencia a-e OK; observaciones no bloqueantes: archivos sin commitear —esperado, git solo bajo pedido—).
- [x] **Fase F — ADRs 6-8**: COMPLETADA 2026-09-08. ADR-6 (M4: crudo para BPE, lemas para clásicos; stems descartados), ADR-7 (aceptado con métricas verificadas), ADR-8 (sentencia completa; párrafo=sentencia; subsentencia 4.64 % no supera criterio).
- [x] **Fase G — Constitución** (M3): COMPLETADA 2026-09-08 por ml-coder. Fuente Senado (HTTP oficial, HTTPS con timeout). 497 filas: 384 permanentes (1-380 + 22A/178A/238A/257A) + 112 transitorios + preámbulo. Spot-check 5/5 OK. `data/constitucion_articulos.parquet/.json` + `resultados/constitucion_cobertura.md`. Ruff OK.
- [x] **Fase H — NER jurídico** (M2): COMPLETADA 2026-09-08 por ml-coder. `03_extraccion_entidades.py` (regex+ventana 60, ruff OK). Precisión muestral 1.00/1.00/1.00/0.90. Cobertura: CP 98.71 %, normas 99.67 %, sents 99.94 %, ponente 85.38 %. Top CP: 241, 86, 13, 2, 1. Aristas 540.181 (516.529 sin self-loops; autocita cuerpo ≥2: 41.95 %). Incidente: relanzamiento colgado al inicio (proceso bloqueado sin progreso, detenido) → rerun desacoplado OK en 249 s. `data/entidades_sentencias.parquet` (23742×7) + `resultados/ner_cobertura.md`.
- [x] **Fase I — Etiquetado** (M3): COMPLETADA 2026-09-08 por ml-coder. `04_etiquetado_areas.py` (ruff OK). Vías: cita_cp 98.71 % (23436), fallback_derecho 0.92 % (218), sin_area 0.37 % (88). Criterio ≥95 % SÍ (99.63 %). Top: derechos_fundamentales 42.46 %, principios 27.59 %, sociales 12.05 %. Tutelas en derechos_fund 47.52 %. Áreas vacías justificadas (preámbulo/transitorios/deberes). `data/etiquetas_areas.parquet` (23742×8) + `resultados/etiquetado_cobertura.md`.
- [x] **Fase J — M1**: COMPLETADA 2026-09-08 (reanudada tras pausa). `05_lematizacion.py` background 6807 s, 23.742/23.742 docs, 195.959 tipos de lemas = 100 % coincidente con contadores EDA (inter 195.959). Salida `data/lemas_por_doc.parquet` (646 MB). Incidentes: 2 OOM por contención con llama-server/E5 → n_process 6→4, batch 16→12; reanudación por id funcionó 3 veces.
- [x] **Fase K — M7 + M0**: COMPLETADA 2026-09-08 por ml-coder. M0: accuracy 1.0, FP=0, FN=0 (30 positivos 1/año + 22 negativos discursos presidentes vía open_corpus_co_es, TALLER-1 sin .txt). Benchmark v1 (30 casos): cita_exacta (circular, recall extracción 1.0, F1@5 0.70, MRR 1.0) > BM25 (F1@5 0.119) > E5-CPU (F1@5 0.090, GPU ocupada por llama-server externo). Fix entorno: torchvision 0.29.0+rocm reinstalado.
- [x] **Fase K-2 — Recalibración M7** (petición de Carlos): COMPLETADA 2026-09-08 por ml-coder. `12_evaluar_vinculo_cp_v2.py` (GPU cuda, 209 s, 90 casos 3/año, 931 pasajes). BM25 F1@5 0.119→**0.218 (+84 %)**, MRR 0.544→0.717; E5 0.090→0.128 (+42 %); híbrido 0.5/0.5 nuevo F1@5 0.209/MRR 0.688; cita_exacta ref 0.649. Deriva temporal del v1 NO se mantiene (BM25 estable, E5/híbrido mejoran en reciente). HALLAZGO (refuta diagnóstico previo): ventana corta de query PEORA (1ª cita a mediana 41 % del texto); la mejora real = max-pooling chunks texto completo + pasajes por artículo + stopwords jurídicas + bigramas LLR (ablation 0.218 vs 0.206). Salidas v2 en resultados/ + figuras/deriva_retrieval_v2.png.
- [x] **Fase L1 — M4 baseline clásico**: COMPLETADA 2026-09-09 (orquestador tras 2 fallos de subagente; v2 tras auditoría L2). `06_baseline_clasico.py`: splits 70/15/15 estratificados UNA VEZ (16.557/3.549/3.548), TF-IDF lemas 100k features, 16 clases. Tabla 2×2 macro-F1 test: LinearSVC 0.2654/0.2657, LogReg 0.2852/0.2870 (completo/enmascarado). **Gap leakage ≈ 0 con enmascarado VÁLIDO** (v2 corregido tras RECHAZO del reviewer: nucleos con tilde + romanos estrictos + series; v1 enmascaraba 0.7 citas/doc vs 10.7 reales). Mejor: logreg_enmascarado 0.2870. Prueba de oro del reviewer: features de cita solas → F1 0.0611 ≈ azar. Accuracy ~59 %. Incidentes: LightGBM→LogReg por RAM (3 muertes silenciosas); saga→lbfgs por lentitud. Artefactos completos y trazables (shap generado por el script).
- [x] **Fase L3 — Transformer BETO**: COMPLETADA 2026-09-09 19:05. Guard VRAM esperó 3h05m (UKAN externo) y arrancó solo. 6h40m GPU, 3 épocas, 256+256 (ADR-3), doble escenario (ADR-2). Test: completo 0.2080 / enmascarado 0.2042, gap leakage 0.0039 (sin leakage, consistente con L2). **LogReg GANA (0.2870)**: docs ~11k palabras vs ventana 512 tokens descarta ~95 % del contenido; TF-IDF ve todo. ADR-10 aceptado: producto conserva LogReg; BETO guardado como comparativa.
- [x] **Fase M — M5 red de citas + M6 diacronía**: COMPLETADA 2026-09-09 por ml-coder. M5: grafo 23.742 nodos / 456.321 aristas resueltas (84.48 %), 7.688 SCC (gigante 67.6 %), PageRank top: C-224/1992, SU-2716/1997, T-985/1993 (landmarks fundacionales); evolución: la Corte cita cada vez más sus precedentes recientes. M6: áreas — crece principios_fundamentales (+20.2 pp), decae derechos_fundamentales (−24.2 pp); log-odds — sube incoder/rupd/bancadas→covid/jep/coronavirus (conflicto→posacuerdo→pandemia); MTLD desciende 64.9→56.8 (prosa menos diversa). 5 figuras + 4 CSV/JSON + red_citas_resumen.parquet. Ruff OK.
- [x] **Fase N — Producto**: COMPLETADA 2026-09-09 por ml-coder. `10_evaluar_sentencia.py` (CLI --archivo|--id|--texto, --salida) + `lib/componentes_producto.py` (5 clases). JSON: validación, ponente, área top-3 con prob, arts CP extraídos+BM25, citas resueltas con PageRank/área, resumen. Demo 3 casos: T-293/97 (true, derechos_fund 0.943, 6+5 arts), C-002/98 (true, 0.934, 18+5, 4 citas con PageRank), discurso Gaitán (false 0.0). ~9 s/eval (índice BM25 cacheado). Ruff OK. Limitaciones documentadas (macro-F1 0.287 heredado; transitorios en BM25).
- [ ] **Fase O — Futuro**: demo, informe, deck, validación extendida, zip, git (cuando Carlos lo pida).

## Cifras canónicas E1 (fuente de verdad)
23.742 sentencias (T 17.099 / C 6.246 / SU 397), 1992–2021 (2021=20 docs);
raw 260.988.316 / limpias 250.012.101 / sin stopwords 128.806.664;
promedios 10.992,68 y 10.530,37; stopwords ~48 % (306 formas norm.),
jurídicas 65 formas; top-1 sin: articulo 947.616; regex parseo 99,97 %.

## ADRs
1 etiquetado vía CP · 2 anti-leakage doble escenario · 3 docs largos 512+chunk-mean ·
4 arquitectura scripts + EDA intacto · 5 vínculo CP weak labels + temporal 1/año ·
6 preprocesamiento M4 (decide 9.3) · 7 motor PySpark+Parquet · 8 granularidad (decide 9.6).

## DoD global
Notebook E1 sección 9 ejecutada, código previo intacto (diff=0), figuras + logs texto;
ADRs 1-8 justificados; etiquetas reproducibles con cobertura; validación entrada OK;
benchmark CP 3×30 con deriva; benchmark M4 2×2 con auditoría leakage; producto
end-to-end con sentencias reales; SOTA+.bib actualizados; plan/ al día.
Git solo bajo pedido de Carlos.
