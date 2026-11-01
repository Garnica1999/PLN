# Contexto E2 (handoffs entre agentes, máx 20 líneas por entrada)

## Fase A (orquestador, 2026-09-08) — COMPLETADA
- Estructura SEGUNDA ENTREGA creada (plan/decisiones, specs, config, data, scripts/lib, modelos, resultados, figuras, experimentos); E1 queda intacta, E2 lee su CSV.
- Documentos: plan/PLAN.md (maestro con fases A-O), 00_maquina.md + config/hardware.json (venv, 32 hilos, 31 GB RAM, GPU RX 9070 XT 17.1 GB ROCm, Java 26, pyarrow 24), 00_intent.md (producto norte), ADRs 1-5.
- Pendiente instalar en venv: pyspark, transformers, datasets, networkx, sentence-transformers.
## Fase B (nlp-specialist, 2026-09-08) — COMPLETADA
- 10 refs nuevas verificadas (E2 total 20): MultiEURLEX 2021, DoSSIER/COLIEE 2021, Fink 2023, Samy 2021 (NER es), Dozier 2010, McCarthy-Jarvis 2010 (MTLD), Hamilton 2016 (diacronía), Fowler 2007 (PageRank), BETO 2020, E5 2022. Hallazgos: retrieval=BM25 pasajes+rerank denso; E5 zero-shot vs deriva; NER es spaCy híbrido F1 0.93; particiones cronológicas obligatorias.
- Fix orquestador: entrada loffler2025 duplicada en .bib → eliminada (20 entradas OK). E1 intacta.
- Siguiente: Fase C migración Parquet (ml-coder).

## Fase C (ml-coder, 2026-09-08) — COMPLETADA
- Spark 4.2.0 + Java 17 (Java 26 incompatible, documentado en ADR-7 borrador). Parquet Hive 1.465 GB, 23742 filas, limpieza E1 100 % idéntica, ruff OK, E1 intacta.
- Fase D se ejecuta en 3 pasos: D1 script background lemas/stems/POS + lanzamiento; D2 celdas notebook 9.1/9.2/9.6 + lectoras; D3 ejecución lectora tras background + corrida completa.
## Fase D1 (ml-coder, 2026-09-08) — COMPLETADA
- Script eda_ext_lemas_stems_pos.py en nohup (PID 393782, 6.6 docs/s, ETA ~56 min, checkpoints c/500 OK, MTLD media ~60 plausible, POS NOUN>ADP>DET). n_process 12→6 por OOM. Ruff OK, E1 intacta.
## Fase D1-fix (ml-coder, 2026-09-08) — COMPLETADA
- Causa caída: doc 1.4M chars > spaCy max_length 1M (E088). Fix: nlp.max_length=5M (doc máx corpus 3M, id 16989) + guard try/except por doc con log de omitidos. Relanzado PID 48750, reanudó en 1500 → 3500, 0 omitidos, ETA ~1.5 h. Ruff OK.
## Fase D2 (ml-coder, 2026-09-08) — COMPLETADA
- 18 celdas nuevas (24-41), 0 errores, backup md5 OK, celdas 0-23 idénticas. LLR gana colocaciones (75 %); granularidad no supera criterio (párrafo=sentencia; subsentencia 4.64 %).
- REGRESIÓN detectada y corregida por orquestador: re-ejecución borró `nota_raw_previo` de metricas.json → restaurada (resta solo fecha/tiempo, valores idénticos). D3 debe re-preservarla tras cada corrida.
## Incidente background (orquestador, 2026-09-08) — DIAGNOSTICADO
- Workers spaCy murieron en silencio (~doc 7000, docs gigantes multi-MB × 6 workers → OOM; journal confirma OOM-killer activo en la máquina) y el padre quedó colgado esperando el pipe eternamente (sin watchdog). 4 workers huérfanos (~13 GB) hacían trabajo perdido → detenidos por orquestador. RAM 5/31 GB. Checkpoints 500-7000 intactos (reanudación desde 7000).
## Fase D1-fix2 dos fases (ml-coder, 2026-09-08) — COMPLETADA
- Distribución longitudes: >200k=1076 docs (4.5 %), >500k=114, >1M=26, máx 3M. UMBRAL=200k (pipe cubre 95.5 %, lote en vuelo ≤3.2M chars/worker).
- Hallazgo: checkpoints tenían 6990 filas reales (28 omitidos EOFError + 78 bloque perdido) → reanudación por id los recupera. Watchdog: os._exit(3) si pipe sin checkpoint 20 min.
- Relanzado PID 481722: 4 checkpoints nuevos (~4.4 docs/s, c/2 min), RAM 10/31 estable. ETA: pipe ~1 h + largos (868) ~2-6 h + stems.
## Fase D3 (ml-coder, 2026-09-08) — COMPLETADA
- 42/42 OK, EDA intacto certificado (celdas + 5 figuras viejas bit-idénticas, metricas.json valores idénticos + nota restaurada). Clave para ADR-6: con BPE no normalizar; si se normaliza, lema. MTLD T>C>SU; C con VERB en top-5 POS.
## Fase E1 (ml-reviewer, 2026-09-08) — APROBADO (sin cambios)
- 5/5 OK: celdas 0-23 idénticas, limpieza parquet==E1 (200/200), RESUMEN aritmética+cobertura 23742/23742, n-gramas/colocaciones reproducen, granularidad reproduce. Sin bugs mayores ni fixes. Notas: metricas.json solo fecha/tiempo (ya documentado); relativizar rutas (sugerencia); riesgos residuales aceptados.
- Siguiente: Fase E2 QA (ml-qa-engineer).
## Fase E2 (ml-qa-engineer, 2026-09-08) — APROBADO CON OBSERVACIONES
- 6/6 figuras nuevas válidas + 5 viejas bit-idénticas; 12/12 logs legibles (~60k líneas); consistencia a-e OK (top bi/tri exactos, MTLD por tipo OK, metricas.json OK + nota presente, 42 celdas 0 errores). No bloqueantes: sin commit (esperado), nota_ontologias en plan/ (diseño), background en E1 (diseño).
## Fase F (orquestador, 2026-09-08) — COMPLETADA
- ADR-6 (M4: crudo para BPE, lemas para clásicos; stems descartados), ADR-7 aceptado con métricas verificadas, ADR-8 (sentencia completa; subsentencia auxiliar).
## Fase G (ml-coder, 2026-09-08) — COMPLETADA
- CP parseada: 497 filas (384 perm + 112 transit + preámbulo), spot-check 5/5, cobertura por bloque, ruff OK. Fuente Senado HTTP (HTTPS timeout).
## Fase H (ml-coder, 2026-09-08) — COMPLETADA
- NER regex validado (prec 1.00/1.00/1.00/0.90); cobertura CP 98.71 % (clave M3 OK); 516.529 aristas sin self-loops. Incidente rerun colgado → resuelto (249 s).
## Fase I (ml-coder, 2026-09-08) — COMPLETADA
- Etiquetado vía CP: 99.63 % asignadas (cita 98.71 %, fallback 0.92 %, sin área 88 docs). Top: derechos_fund 42.5 %, principios 27.6 %. Ruff OK.
- Nota: Fase J se redefine (D1 ya dio contadores+MTLD globales); J = secuencias de lemas por doc para M4-clásico (background).
## Fase J (ml-coder, 2026-09-08) — SCRIPT EN BACKGROUND
- 05_lematizacion.py (port fiel D1) corriendo PID 883147 (~5.8 docs/s, ETA ~1 h, consistencia lemas 100 % ⊆ ref). Salida final: data/lemas_por_doc.parquet.
- HALLAZGO p/M4: spaCy emite 4507 lemas multi-palabra con espacio ("hacer él", clíticos; 948k ocurr.) → M4 debe normalizar con "hacer_él" o tokenizar en consecuencia.
- Siguiente: Fase K M7 (ml-coder, independiente de J).
## Fase K (ml-coder, 2026-09-08) — COMPLETADA
- M0 accuracy 1.0 (negativos: discursos presidentes vía open_corpus_co_es). Benchmark: cita_exacta F1@5 0.70 (circular, recall 1.0) > BM25 0.119 > E5-CPU 0.090. Deriva temporal: BM25 degrada (0.175→0.088), E5 mejora (0.050→0.109). GPU ocupada por llama-server (~15.7 GB) → E5 en CPU documentado.
- Fix entorno: torchvision reinstalado (torch 2.14+rocm pareja).
- Nota J: OOM en 3000 → relanzado PID 1169131 (reanuda desde 3500), monitorear.
- Fase L: GPU OCUPADA por llama-server → L1 baseline clásico (CPU) ahora, L2 auditoría leakage (reviewer), L3 transformer diferida hasta liberar VRAM.

## Reanudación (2026-09-08 tarde, orden de Carlos: continuar + recalibrar K)
- J reanudado y COMPLETADO: 23742/23742 lemas (6807 s), 195959 tipos = 100 % ref EDA, data/lemas_por_doc.parquet 646 MB.
- K-2 recalibrada (12_evaluar_vinculo_cp_v2.py, GPU): BM25 F1@5 0.218 (+84 %), E5 0.128, híbrido 0.209, cita_exacta ref 0.649. Ventana corta REFUTADA con evidencia (ADR-9). Deriva temporal v1 no se mantiene.
## Fase L1 (orquestador, 2026-09-09) — COMPLETADA
- M4 baseline: LogReg 0.2852 / LinearSVC 0.2654 (completo); gap leakage ≈ 0 (aprende contenido, no citas). Splits guardados UNA VEZ. LightGBM descartado por RAM (3 muertes silenciosas, máquina compartida con UKAN externo).
## Fase L2 (ml-reviewer, 2026-09-09) — RECHAZO v1 → CORRECCIÓN → v2 OK
- Reviewer RECHAZÓ el enmascarado v1 (0.7 CITA/doc vs 10.7 citas reales; nucleos sin tilde; romanos falsos) pero demostró SIN leakage activo (prueba de oro: features cita solas → F1 0.0611 ≈ azar 1/16; is_alpha ya eliminó números; top-20 temático).
- Orquestador corrigió enmascarado (nucleos con tilde + plurales, romano estricto, series "11 y 13") + shap trazable para cualquier mejor modelo. Re-ejecución v2: LogReg 0.2852/0.2870, gap −0.0018 (enmascarado incluso mejor). Gate L2 SUPERADO con v2.
## Fase M (ml-coder, 2026-09-09) — COMPLETADA
- M5: PageRank landmarks fundacionales (C-224/1992 top); 456k aristas resueltas; gigante 67.6 %. M6: principios +20 pp / derechos_fund −24 pp; léxico conflicto→posacuerdo→pandemia; MTLD 64.9→56.8.
- L3 (07_transformer.py) lanzado con guard VRAM: espera a que el UKAN externo libere GPU (intento 10+, cada 5 min, máx 24 h).
## Fase N (ml-coder, 2026-09-09) — COMPLETADA
- Producto end-to-end OK: 3 demos (T-293/97 true 0.943, C-002/98 true 0.934, discurso false). ~9 s/eval. Componentes en lib/, CLI en 10_evaluar_sentencia.py.
## Fase L3 (2026-09-09 19:05) — COMPLETADA
- BETO 3 épocas 256+256: test completo 0.2080 / enmascarado 0.2042, gap 0.0039 (sin leakage). LogReg GANA (0.2870) — ventana 512 tokens descarta ~95 % de docs de ~11k palabras. ADR-10 aceptado: producto conserva LogReg sin cambios; BETO en modelos/m4_transformer/.
## Cierre E2 (2026-09-10) — COMPLETADO
- QA final: GO (10/10 casos, 16/16 métricas sin drift). Auditoría rendimiento: OK académico (8-12 s/sentencia, cuello inicialización no algorítmico, CPU-only).
- Informe CRISP-DM (3.773 palabras, 40/40 cifras auditadas) + deck 15 slides + README en documento_e2/. Zip: PROYECTO/juris_pln_e2_entrega.zip (29 MB, 93 archivos).
- ÚNICO PENDIENTE: git commit+PR bajo pedido de Carlos (nada commiteado por acuerdo). Opcional: demo streamlit.

## Ciclo v2 (2026-09-10) — BLOQUE 1 COMPLETADO, CHECKPOINT DE PAUSA
- Paso 0: docs v2 (PLAN.md congelado v1, ADR-11, 6 refs nuevas .bib 26, estado_arte sección v2).
- Fase R (v3 sucio): nDCG@10 + top10_por_caso + diagnóstico contaminación (24.4% transitorios top-5, 65/90 casos, 152 ACL en top-10). Reviewer APROBADO.
- Fase S1: 45 IDs ACL → Transitorio 68-112 (ACL01016=código AL 1/2016 del Senado; verificado 0 fallos); BM25 permanentes default (384/685) + flag transitorios; top-10 producto. v3_limpio: bm25 0.233/0.765/0.380. Reviewer APROBADO (mapeo verificado contra HTML Senado).
- Fase S2: pyarrow filters → 8.8→5.5 s, RAM 8.0→1.6 GB. Reviewer APROBADO (fix: bug cache reescritura + perf_6251 renombrado _s2).
- Fase T iter1 (v4): rerank query completa FALLÓ (0.177 F1@5, −24%: truncamiento 512 tokens del doc ~68k chars). iter2 (v5): rerank POR CHUNKS (top-20 BM25 → 3 pares chunk-pasaje → bge-reranker-v2-m3 fp16 GPU → max-pooling): **F1@5 0.285 / P@5 0.456 / MRR 0.850 / nDCG@10 0.438** — mejor no-circular; +31% F1@5 vs inicio ciclo. Metas 2/3 (F1@10 0.249 por techo recall@20). Reviewer APROBADO (bug --no-rerank sobrescribía _limpio → corregido a /tmp).
- ADR-12 aceptado (índice permanentes + rerank chunks; v4 conservado como evidencia del fallo por truncamiento). Caso 6251: arts 21/209 posiciones 220/305 en BM25 → fuera de top-20, el reranker no puede rescatarlos (limitación 1ª etapa).
- PAUSA alcanzada. Bloque 2 (U/V/W/X + candidatos: rerank top-50/híbrido, RRF, default rerank producto) SOLO con aprobación de Carlos.

## Fase T (ml-trainer, 2026-09-10) — COMPLETADA (resultado NEGATIVO)
- Run final v4: 90 casos, GPU RX 9070 XT fp16, 241.9 s (rerank 53.9 s). Salida: vinculo_cp_benchmark_v4.json (v4-ciclo2-faseT, permanentes-v3, reranker bge-reranker-v2-m3).
- bm25_rerank F1@5 0.177 / P@5 0.276 / MRR 0.572 / nDCG@10 0.296 / F1@10 0.203 — EMPEORA vs bm25 v3_limpio (F1@5 0.233, -24%; MRR 0.765, -25%) y vs híbrido (0.263). Deterioro uniforme en los 3 periodos (peor 2012-2021).
- METAS Fase T (P@5>=0.45, F1@10>=0.30, MRR>=0.80): 0/3 CUMPLIDAS.
- Hipótesis fallo: mismatch query(~68k chars truncada a 512 tok) vs pasaje(<=120 palabras) — el cross-encoder pierde el contexto que BM25 explota con 113 chunks; techo estructural = recall@20 BM25.
- Caso 6251 (ad-hoc, fuera de muestra): arts 21/209 en pos. 220/305 del ranking BM25 -> fuera del top-20 -> reranker no puede recuperarlos (limitación recall 1a etapa CONFIRMADA).
- Híbrido 0.5/0.5 sigue siendo el mejor no-circular. md5s v2/v3/v3_limpio INTACTOS. metricas.json (sección ciclo_v2) creado.
- Incidente menor: 1er lanzamiento muerto por timeout del shell (process group); relanzado con setsid -> OK.
- Siguiente paso recomendado: NO adoptar rerank como está; si se insiste, rerank sobre híbrido top-20 con query resumida/considerandos. Checkpoint PAUSA -> decisión de Carlos.

## Fase T iter2 (ml-trainer, 2026-09-10) — COMPLETADA (v5: reranking por chunks, resultado POSITIVO)
- Run final v5: 90 casos, GPU RX 9070 XT fp16, 255.9 s (rerank 70.5 s), setsid nohup sin incidentes. Salida: vinculo_cp_benchmark_v5.json (v5-ciclo2-faseT-iter2, max_tokens_par 405<=512 verificado).
- bm25_rerank_chunks: F1@5 0.285 / P@5 0.456 / MRR 0.850 / nDCG@10 0.438 / F1@10 0.249 — MEJOR estrategia no-circular; vs bm25 +22% F1@5/+11% MRR, vs híbrido +8% F1@5/+13% MRR, vs bm25_rerank v4 +61% F1@5 (hipótesis truncamiento CONFIRMADA).
- METAS Fase T: 2/3 cumplidas (P@5 0.456>=0.45 OK, MRR 0.850>=0.80 OK, F1@10 0.249<0.30 FALLA — techo: recall@20 de 1a etapa BM25 + silver multietiqueta).
- Por periodo: gana a bm25 en los 3 (+28%/+36%/+6% F1@5); vs híbrido gana 1992-2001 (+34%) y 2002-2011 (+6%), pierde 2012-2021 (-9.5%).
- md5s v2/v3/v3_limpio/v4 INTACTOS. metricas.json: añadida subsección ciclo_v2.v5_faseT_iter2. 0 transitorios en top-k (esperado).
- Recomendación: ADOPTAR bm25_rerank_chunks en el producto (mejor P@5/MRR/nDCG@10; F1@10 requiere subir recall 1a etapa: top-50 o híbrido como candidato).

## Fase T iter3 (ml-trainer, 2026-09-10) — COMPLETADA (v6: rerank top-50, 3/3 METAS)
- Run final v6: 90 casos, GPU RX 9070 XT fp16 (~15.4 GiB libres), 344.8 s (rerank 152.5 s, 2 variantes × 150 pares/caso, max_tokens_par 430<=512). Salida: vinculo_cp_benchmark_v6.json (v6-ciclo2-faseT-iter3, permanentes-v3).
- hibrido_rerank_chunks_top50 = MEJOR no-circular: F1@5 0.340 / P@5 0.520 / MRR 0.865 / nDCG@10 0.513 / F1@10 0.316 — vs v5 +19% F1@5/+27% F1@10; vs bm25_top50 +5% F1@5/+12% F1@10 (MRR 0.875 vs 0.865, única donde bm25_top50 gana).
- METAS iter3: 3/3 CUMPLIDAS (F1@10 0.316>=0.30 ✓, MRR 0.865>=0.85 ✓, P@5 0.520>=0.45 ✓) — la meta F1@10 incumplida en v5 ahora se cumple.
- Por periodo (hibrido_top50 F1@5): 0.403/0.327/0.289 — gana a v5 en los 3 (+30%/+18%/+8%); en 2012-2021 queda bajo el híbrido sin rerank (0.296) en F1@5 pero lo supera en P@5/MRR.
- Caso 6251 (predicción nota_v6): NO rescatado (arts 21/209 pos 85/355 del top-50 híbrido) — limitación recall 1a etapa DOCUMENTADA, no fallo del run. md5s v2/v3/v3_limpio/v4/v5 INTACTOS. metricas.json: subsección ciclo_v2.v6_faseT_iter3.
- Recomendación: ADOPTAR hibrido_rerank_chunks_top50 como default del --rerank del producto (mejor en 4/5 métricas globales; costo: E5 query 243 s + rerank 150 s).

## INCIDENTE Fase V (orquestador, 2026-09-11 01:00) — DEADLOCK DATALOADER (causa raíz + plan)
- 4 intentos de run completo (22:59, 00:12, 00:40, 00:59 debug) cuelgan en el MISMO punto: main en dataloader `_try_get_data` (espera lote), 8 workers idle esperando índices (0 % CPU), pin thread idle. Smokes (64-200 docs) pasan SIEMPRE; full (16 557 docs train) cuelga SIEMPRE. La tokenización en workers completa ~7 docs (warnings) y la pipeline de colas worker→pin→main se congela.
- Descartado: `set_start_method("fork")` ya estaba puesto (no lo arregla). Probado: repro mínimo default funciona; el fallo es de escala/cola. faulthandler (SIGUSR1, wrapper /tmp/opencode/dl_debug/run_debug.py) capturó pilas limpias → evidencia en experimentos/*faulthandler_evidencia.log. py-spy bloqueado (ptrace_scope=1, sin sudo).
- FIX ACORDADO: pre-tokenizar chunks a disco (script NUEVO 14, proceso sin torch/CUDA, Pool spawn, shards con checkpoint, uint16) + 07 con num_workers=0 y cache obligatoria (fail-fast). Elimina workers multiproceso en entrenamiento (sin deadlock posible) y acelera arranques (cache reutilizable). Revisión obligatoria de ml-reviewer antes de lanzar.
- v1 truncado y resultados previos INTACTOS. Comando pre-cache: 14_precache_chunks_beto.py (ETA ~50-100 min, reanudable). Después: 07_transformer.py ~5-6 h GPU.

## Fase V (ml-trainer, 2026-09-11) — RUN FINAL LANZADO (chunk-mean BETO, cache + num_workers=0)
- Comando: `setsid nohup /home/carlos/venv/bin/python scripts/07_transformer.py > experimentos/m4_transformer_chunks_stdout.log 2>&1 < /dev/null` — PID 669655, inicio 01:28. Log FileHandler: experimentos/m4_transformer_chunks.log.
- INCIDENTE DEADLOCK RESUELTO: cache chunks (1.3 GB, 48 shards) + num_workers=0; arranque sano con progreso a los ~2 min (sin cuelgue). Guard VRAM OK (17.0 GB libres), 23654 filas/16 clases, GPU 100%, ~13 GB VRAM. Sin errores/NaN.
- Progreso 15 min: `[completo] epoca 1: 4000/16557 docs, perdida 2.54→1.83, 5.8 docs/s` estable. ETA: ~48 min/epoca x 3 = ~2.4 h (completo) + ~2.4 h (enmascarado) → fin estimado ~06:45-07:15.
- Decisión overfitting: 3 épocas fijas (sin early stopping) por comparabilidad con v1; batch efectivo 32, lr 2e-5.
- AL TERMINAR verificar: resultados/m4_transformer_chunks.json + modelos/m4_transformer_chunks/beto_{completo,enmascarado}/; macro-F1 test ambos escenarios; gap train/val/test (leakage); comparación vs v1 truncado 0.2080/0.2042 y LogReg 0.2870; meta >=0.26 → completar Fase V.
- v1 INTACTO: md5 m4_transformer.json = 566a6ecaff1af97d37f46c21f5e2f133; modelos/m4_transformer/ (8 archivos, sep 9) sin tocar.

## Fase V RELANZADA (ml-trainer, 2026-09-11 08:28) — crash epoch 1->2 corregido, run final en curso
- Causa del crash previo (02:18, evidencia conservada): `RuntimeError: unscale_() is being called after step()` — la rama del paso parcial final de época no llamaba `scaler.update()`, así que el primer `unscale_()` de la época siguiente crasheaba. Fix en 07 (~línea 713) + `modelo.train()` por época; smoke de 2 épocas APROBADO (rev_fix_2epocas.json/rev_fix_2epocas.log, 08:24).
- Relanzamiento (comando aprobado): `--estrategia chunk_mean --cache-dir data/cache_chunks_beto`, PID 1700211, inicio 08:28:10, setsid nohup. Arranque sano: guard VRAM 17.0 GB OK, cache texto 48 shards, GPU 100% (~11.3 GB VRAM), 23654 filas/16 clases, primera línea 800/16557 docs a 5.8 docs/s (pérdida 2.5356 idéntica al run previo).
- ETA: ~48 min/época x 3 épocas x 2 escenarios ≈ 5-5.5 h → fin estimado ~13:45-14:15. Verificación crítica SUPERADA (09:23): época 1 cerrada 09:18:40 (val loss 1.3099, macro-F1 val 0.1503, idéntica al run que crasheó) y época 2 en progreso (800/16557 @09:20:58; 1600/16557 @09:23:17) SIN el crash de unscale_; proceso vivo. El docs/s que imprime época 2 (~0.3-0.5) es promedio acumulado desde t0 (cosmético, lógica v1), la velocidad real sigue ~5.7-5.8 docs/s. Si muere más adelante, revisar experimentos/m4_transformer_chunks_stdout.log y NO relanzar a ciegas.
- Al terminar: resultados/m4_transformer_chunks.json + modelos/m4_transformer_chunks/beto_{completo,enmascarado}/; comparar test macro-F1 vs v1 0.2080/0.2042 y LogReg 0.2870; meta >=0.26. md5 m4_transformer.json = 566a... intacto (verificado también durante el run).

## Fase U (ml-trainer, 2026-09-11) — COMPLETADA (meta 0.32 NO cumplida; ganadora C +3.8% vs v1)
- Run final `--fase-u` (CPU, en paralelo con Fase V): PID 1700365, 758.5 s (~12.6 min), setsid nohup, sin incidentes. Log: experimentos/m4_jerarquico.log (+ stdout). Salidas NUEVAS: resultados/m4_jerarquico.json, modelos/m4_jerarquico/ganadora_C_calibracion.joblib, figuras/fig_m4_jerarquico_variantes.png.
- Protocolo verificado en el run: sanity check v1 OK (0.287008 reproducido exacto) y splits canónicos reutilizados 16557/3549/3548. Selección por macro-F1 single-label de VALIDACIÓN; test solo al final.
- VAL: A 0.2785 / B 0.2883 / C 0.3792 → ganadora C_calibracion. TEST: A 0.2697 / B 0.2500 / C 0.2980 (v1 ref 0.2870). C: base reentrenada SOLO con train (evita la val contaminada del joblib v1) + sesgos por clase optimizados en val; val 0.2683→0.3792 y test 0.2742→0.2980 (+3.8% vs v1; sesgos grandes en clases minoritarias: organizacion_estado +1.55, reforma_constitucion +1.5, organismos_control +1.75).
- B multietiqueta (métrica nativa, aparte): macro-F1 val 0.5334 / test 0.4792, micro test 0.7895; su conversión single-label test 0.2500 no supera a C (el target multietiqueta es otra tarea y no es comparable con el protocolo v1). A jerárquica: grueso val 0.4553 / test 0.4033; fino test 0.2697.
- META test >=0.32 NO cumplida (0.2980; falta 0.022). Conclusión: la calibración de sesgos es la única mejora real sobre v1 (+3.8%); jerárquico/multietiqueta no aportan en single-label. Para Fase X: C es la candidata si se adopta v2; la meta 0.32 requiere otra vía (features/lemas/datos). md5 v1 INTACTOS: m4_baseline.json 60f8... y m4_transformer.json 566a... (también durante el run).

## Fase W(b) EuroVoc (ml-trainer, 2026-09-11) — COMPLETADA; D/E NO se adoptan
- Run final `--fase-u` (CPU, en paralelo con BETO, sin GPU): PID 1990812, 2022.7 s (33.7 min; E más lento de lo estimado), sin incidentes. Salidas NUEVAS: resultados/m4_eurovoc.json, modelos/m4_eurovoc/ganadora_C_calibracion.joblib, figuras/fig_m4_eurovoc_variantes.png; log experimentos/m4_eurovoc.log.
- VAL macro-F1 single-label: A 0.2785 / B 0.2883 / C 0.3792 / D 0.2617 / E 0.2770 -> ganadora C_calibracion (la misma de Fase U). TEST: A 0.2697 / B 0.2500 / C 0.2980 / D 0.2625 / E 0.2771; C reproduce exacto 0.2980095633575616 (delta 0 vs m4_jerarquico.json); meta 0.32 NO cumplida.
- D jerárquica EuroVoc (8 macro-grupos): fino test 0.2625 / grueso 0.3671 -> NO adoptar (delta test -0.0355 vs C). E features léxicas EuroVoc (1268 términos, cobertura val/test 1.0): test 0.2771 -> NO adoptar (delta -0.0209); negativo esperado (E1 anticipaba aporte escaso).
- B multietiqueta (métrica nativa): val 0.5334 macro / 0.7944 micro, test 0.4792 / 0.7895; su single-label test 0.2500 no compite. A sin cambios.
- md5 INTACTOS: m4_jerarquico.json 94425f6eab8ba03dd7b74d0f3962ca40, m4_baseline.json 60f8fd1ce4c2edef086d8ee01aaad654; config/eurovoc_bloques.json corregido d30e1410548e7b19cd719f4ed3749125 (typo, documentado).
- Registro: metricas.json -> ciclo_v2.w_faseU_eurovoc (secciones previas intactas).
- Fase W(b) cierra: C calibración candidata para Fase X; meta 0.32 requiere otra vía (features/lemas/datos). BETO (PID 1700211) sano en escenario enmascarado, no se tocó.

## Fase V completada (ml-trainer, 2026-09-11) — chunk-mean BETO: mejora vs v1, NO supera a LogReg
- BETO chunk-mean (cache + num_workers=0, 3 épocas fijas, semilla 42): test macro-F1 completo 0.2433 (acc 0.5634, mejor F1 val 0.2313) / enmascarado 0.2363 (acc 0.5403, val 0.2256); gap leakage 0.0070; 5.15 h; 52.7 chunks/doc (máx 64). Curvas sanas (F1 val 0.1503→0.1935→0.2313 y 0.1503→0.1888→0.2256).
- Verificación: resultados/m4_transformer_chunks.json (estructura completa, sin NaN/Inf) + modelos/m4_transformer_chunks/beto_{completo,enmascarado}/ con pesos completos (~439 MB c/u + config/tokenizer).
- vs BETO v1 truncado (0.2080/0.2042): chunk-mean mejora +17.0 %/+15.8 % → reduce la brecha metodológica, pero la meta de fase >= 0.26 NO se cumple (0.2433).
- LogReg v1 0.2870 y Fase U C-calibrada 0.2980 siguen ganando: el producto MANTIENE LogReg+C calibrada (ADR-10 vigente); chunk-mean documentado como mejora metodológica.
- Registro: metricas.json -> ciclo_v2.v_faseV_chunkmean (secciones previas intactas).

## Fase W-a hubs (ml-trainer, 2026-09-11) — COMPLETADA; prior de hubs MEJORA (adoptar)
- Run final --calibrar-hubs: PID 2458863, 346.1 s, setsid nohup, sin incidentes. Salidas: resultados/vinculo_cp_benchmark_v7.json (v7-ciclo2-faseW-a) + figuras/hubs_alpha_v7.png (+ por_ano/deriva/contaminación v7).
- Prior: frecuencia global de citación CP (497 cánones, top-10 241/86/13/2/1), no-leakage por diseño (estadística de corpus estilo IDF); PageRank de co-citación descartado (Spearman 0.9828).
- α=0.5 en las 4 estrategias focales (grid máx); full-90 F1@5 α=0→α=0.5: bm25 0.233→0.388, hibrido 0.263→0.374, bm25_rerank_chunks_top50 0.323→0.365, hibrido_rerank_chunks_top50 0.340→0.411 (MRR 0.865→0.938, nDCG@10 0.513→0.598, F1@10 0.316→0.361); A/B cruzada positiva en las 3 vistas → "adoptar" en las 4.
- Sanity: α=0 en v7 reproduce v6; caveat: óptimo en el borde de la rejilla (α=0.5) → valorar iter2 con rejilla extendida. md5 v2–v6 y m4_transformer.json INTACTOS.
- Registro: metricas.json -> ciclo_v2.w_faseW_a_hubs.

## Fase W-a iter2 hubs (ml-trainer, 2026-09-11) — COMPLETADA; rejilla extendida, óptimo INTERIOR (adoptar α=1.5)
- Run final iter2 --calibrar-hubs: PID 2501185, 338.2 s, sin incidentes; salidas v8 (benchmark + hubs_alpha_v8.png); md5 v2–v7 intactos.
- α full elegido: bm25 0.75, hibrido 0.75, bm25_rerank_chunks_top50 1.5, hibrido_rerank_chunks_top50 1.5; "adoptar" en las 4 focales.
- Ganadora hibrido_rerank_chunks_top50 α=1.5: F1@5 0.340→0.425, MRR 0.865→0.950, nDCG@10 0.512→0.620, F1@10 0.316→0.379; vs v7 α=0.5 +1.4/+1.2/+2.2/+1.9 pp. Curva full-90 sube hasta 1.5 (0.4246) y cae en 2.0 (0.4172) → óptimo interior, ya no borde.
- Decisión Fase X (producto): prior α=1.5 en el reranker top50 y α=0.75 en bm25/hibrido base; bm25_rerank_chunks_top50 satura ~0.5 (meseta 0.364–0.367).
- Registro: metricas.json -> ciclo_v2.w_faseW_a_iter2; ADR-10 y checks de producto sin cambios.
## Fase X — QA v2 (ml-qa-engineer + orquestador, 2026-09-11) — GO
- checks_qa_final.py v2 (v1 preservado como checks_qa_final_v1.py): k=10, hubs_usado/alpha_hubs, modos default/rerank, integridad v2. Control negativo OK (falla contra JSON v1 con 82 hallazgos).
- 10/10 casos E2E (8 pos + 2 neg; 2 en rerank) + integridad v2: EXIT 0. Drift clasificador/validación = 0 (area+prob idénticas v1↔v2); retrieval cambia por diseño (5→10, hubs 0.75/1.5, rerank híbrido top-50). Caso 6251: 21/209 siguen fuera del top-10 (techo recall 1ª etapa, documentado). Veredicto en resultados/qa_final_v2/VEREDICTO_QA.md (GO + rollback).
- md5 canónicos intactos; metricas.json intacto durante QA.

## Ciclo v3 — Paso 0 (orquestador, 2026-09-11) — CONGELAMIENTO + ARRANQUE
- Plan v3 aprobado y guardado: plan/PLAN_v3_recuperacion_robusta.md (Sprint A/B/C, meta nDCG@10>=0.70 con gold humano, IC bootstrap, premisas del LLM ALIA-7B).
- ADR-16 (versionado ciclo v3: resultados v9+/c3_*; producto v2.1/2.2/v3; criterios de adopción con IC + A/B) y plan/REGISTRO_EXPERIMENTOS.md (retro-relleno COMPLETO v1+v2 con hipótesis/premisa/resultado/decisión/artefacto).
- Congelamiento git: rama feature/e2-congelamiento-v2 con commit local (push/PR pendientes de credenciales de Carlos). .gitignore ampliado (data/, modelos/, lematizacion/, evidencia crash).
- Siguiente: Sprint A1 (métricas honestas: recall@k + IC bootstrap + periodos) vía ml-coder.
## Ciclo v3 — Sprint A1 (ml-trainer, 2026-09-11) — COMPLETADO
- Run final `--diagnostico` sobre 90 casos (347,9 s; PID 2710786; GPU RX 9070 5,1 GB usados); salidas `resultados/c3_diagnostico_recall.json` + 3 figuras `c3_*.png`; estándar en /tmp/opencode; md5 v2..v8 intactos.
- Curvas sin hubs (recall@10/50/200): bm25 0,241/0,413/0,705; híbrido 0,312/0,509/0,766; híbrido+rerank 0,369/0,509/0,509 (topado a candidatos).
- Techo reranker = recall candidatos top-50: híbrido 0,509 vs bm25 0,413 → ampliar a top-100 (recall 0,619 híbrido) es la palanca de A2/A4.
- Mejor estrategia (con_hubs, alpha adoptados): híbrido+rerank nDCG@10 0,620 IC [0,586;0,652] / MRR 0,950 / F1@5 0,424 / F1@10 0,379 (IC [0,355;0,402]) — reproduce v8.
- Sin hubs: híbrido+rerank nDCG@10 0,512 IC [0,470;0,554] vs bm25 0,380; los hubs aportan +0,108. Por periodo híbrido+rerank sin hubs: 0,559/0,474/0,505 (sin caídas); por longitud: 0,520/0,469/0,548 (mejor en largos).
- Techo silver NER: F1@5 ≤0,649 / F1@10 ≤0,742, n_gold medio 11,4 → la meta 0,70 nDCG exige gold graduado (Sprint B/C), no solo mejoras de modelo.
- Registro: `metricas.json` → `ciclo_v3.a1_diagnostico`; fila A1 en REGISTRO_EXPERIMENTOS; checkbox A1 marcado.
- Siguiente: A2 embeddings legales ES (E5 ref vs ALIA-MrBERT vs BGE-M3-Legal vs modernbert-legal), candidatos top-50 → top-100 según A1.

## Ciclo v3 — Sprint A2 (ml-trainer, 2026-09-11) — COMPLETADO (gate A2: sin adopción; ALIA mejor en contexto producto)
- Run final: PID 2808142, 1367,8 s (~23 min), GPU RX 9070 XT (pico ~5,8 GB VRAM), setsid nohup, sin incidentes. Salidas v9: `resultados/vinculo_cp_benchmark_v9.json` (v9-ciclo3-sprintA2) + `figuras/c3_a2_embeddings.png`; md5 v2..v8 intactos (8/8).
- Con hubs (alpha=1,5) nDCG@10/F1@10: e5@100 0,629/0,401 (ref) / alia@100 0,638/0,417 / bgem3@100 0,631/0,395 / modernbert@100 0,628/0,393. Sin hubs: modernbert 0,517 único que gana a e5@100 (Δ +0,021, IC [+0,0004;+0,040]).
- Pareado vs e5@100: ningún modelo bate nDCG@10 con IC en ambos contextos; victorias parciales: modernbert sin_hubs nDCG (+0,021) y alia con_hubs F1@10 (+0,017, A/B consistente) → gate A2 NO superado, sin adopción.
- Recall candidatos top-100: alia 0,723 (+10,5 pp vs e5 0,619) / bgem3 0,630 / modernbert 0,588; e5 reproduce el 0,619 de A1 (+11 pp vs top-50, confirmado). Más recall NO se traduce en nDCG con el reranker genérico sin hubs.
- Profundidad E5 @100 vs @50 legacy: sin hubs 0,496 vs 0,512 (-0,016); con hubs 0,629 vs 0,620 (+0,009) y F1@10 0,401 vs 0,379 (+0,021) → top-100 solo rentable con hubs.
- Registro: `metricas.json` → `ciclo_v3.a2_embeddings`; fila A2 en REGISTRO_EXPERIMENTOS; checkbox A2 en PLAN_v3.
- Siguiente A3: reranker legal ES (SINAI ALIA reranker / bge-reranker-v2-m3 / RRF) sobre candidatos top-100 de ALIA con hubs alpha=1,5, Δ pareado vs bge@100.
- A4: RRF BM25+denso k=60 vs mezcla y rejilla de profundidad 50/100/200 sobre el ganador de A3.

## Ciclo v3 — Sprint A3 (ml-trainer, 2026-09-11) — COMPLETADO (gate A3: sin adopción; bge se mantiene)
- Run final `--a3 --rerank`: PID 2939850, 613,7 s (~10 min), GPU RX 9070 XT fp16, setsid nohup, sin incidentes; salidas v10 (`vinculo_cp_benchmark_v10.json`, md5 dca86449…) + `figuras/c3_a3_rerankers.png` + derivado `resultados/c3_a3_rerankers.json` (md5 3d03a0a7…); md5 v2..v9 intactos.
- nDCG@10 sin_hubs: bge 0,509 / aliarerank 0,366 / rrf 0,502; con_hubs (α=1,5): bge 0,638 / aliarerank 0,583 / rrf 0,618. F1@10 con_hubs: 0,417 / 0,379 / 0,406.
- Pareado vs bge (nDCG@10): aliarerank −0,143 sin_hubs y −0,055 con_hubs (pierde con IC 95 %); rrf −0,007 sin_hubs (IC [−0,037;+0,022] cruza 0) y −0,020 con_hubs (IC [−0,038;−0,003], pierde). Ninguna variante bate a bge → GATE A3 NO SUPERADO; sin adopción, producto v2.1 sin cambios.
- Recall candidatos ALIA@100 = 0,7233 (idéntico a A2; por periodo 0,768/0,730/0,672); el cuello no se movió: bge@100 reproduce A2 exacto (0,638/0,417) y el reranker legal no alcanza el techo estructural.
- Smoke n=10 de rrf (+0,038 sin_hubs) era ruido de muestra: el run final de 90 casos lo revierte a −0,007 (IC cruza 0) y confirma rrf ≤ bge con hubs.
- Registro: `metricas.json` → `ciclo_v3.a3_rerankers`; fila A3 en REGISTRO_EXPERIMENTOS; checkbox A3 marcado en PLAN_v3.
- Siguiente A4: RRF de embeddings/fusión a nivel de scores (no de rangos), profundidad 50/100/200 sobre la base ganadora (bge+ALIA@100 con hubs).

## Ciclo v3 — Sprint A4 (ml-trainer, 2026-09-11) — COMPLETADO (gate A4: sin cambio de fusión; top-50 recomendado)
- Run final `--a4 --rerank`: PID 3039803, 1020,1 s (~17 min), GPU RX 9070 XT fp16, setsid nohup, sin incidentes; salidas v11 (`vinculo_cp_benchmark_v11.json`, md5 98ed1034…) + derivado `c3_a4_fusion_profundidad.json` (md5 64343628…) + `figuras/c3_a4_fusion_profundidad.png`; md5 v2..v10 intactos (10/10).
- Fusiones @100 (nDCG@10 / F1@10 sin_hubs | con_hubs): w05 0,509/0,319 | 0,638/0,417 (ref) · w03 0,506/0,319 | 0,637/0,416 · w07 0,522/0,327 | 0,637/0,415 · rrf1a 0,509/0,323 | 0,639/0,415.
- Pareado vs w05@100: NINGUNA fusión gana con IC 95 %; w07 sin_hubs +0,013 IC [−0,000;+0,026] roza 0 pero no gana; con_hubs todas empatadas (|Δ|≤0,003, IC cruza 0) → gate de fusión NO superado, w05 0.5/0.5 se mantiene.
- Profundidad w05: sin_hubs top50 0,536 GANA a top100 0,509 (Δ +0,027 IC [+0,014;+0,039]; F1@10 +0,022 IC>) y top200 0,496 PIERDE (Δ −0,013 IC<0); con_hubs top50 0,640 / top200 0,636 vs top100 0,638 indistinguibles (IC cruza 0). Recall w05 @50/100/200 = 0,623/0,723/0,833 (más recall ≠ más nDCG con hubs).
- Ganadora producto: fusión w05 0.5/0.5 + bge + hubs α=1,5 + k=10 con candidatos **top-50** (mejor en sin_hubs con IC, no inferior con hubs, ~mitad de costo de rerank); top-200 rechazado. Sin cambio de código en v2.1 (top-50 ya es default); sustitución E5→ALIA pendiente de Gate A.
- Smoke w07 n=10 no se replicó: a n=90 w07 empata con hubs y roza 0 sin ellos (mismo patrón que RRF en A3) → los smokes n=10 no deciden gates.
- Registro: `metricas.json` → `ciclo_v3.a4_fusion_profundidad`; fila A4 en REGISTRO_EXPERIMENTOS; checkbox A4 marcado. Siguiente A5: calibración C + ensamble + class weights (`m4_clasificador_c3.json`).

## Ciclo v3 — Sprint A5 (ml-trainer, 2026-09-11) — COMPLETADO (gate clasificador: sin adopción; producto mantiene v1)
- Run final `--fase-a5`: PID 3200478, 1294,9 s (~21,6 min), GPU RX 9070 XT fp16, setsid nohup, sin incidentes; salidas NUEVAS `resultados/m4_clasificador_c3.json` + `figuras/c3_a5_clasificador.png`; md5 intocables intactos (5/5).
- Repro C: val 0,3792 / test 0,2980095633575616 EXACTO (delta 0); BETO chunk-mean completo test 0,2433443 (val 0,2313; repro 07 20/20) y enmascarado 0,2363647 (val 0,2256), ambos en banda ±0,001.
- Ensamble C+BETO: w=0,8 en val (0,3066) → test 0,2901 (Δ −0,0079 vs C; IC95 [−0,0227;+0,0048], p=0,20); class weights balanced val 0,3960 → test 0,2896 (Δ −0,0085; IC [−0,0347;+0,0063]) y sqrt_inv val 0,3854 → test 0,2851 (Δ −0,0129; IC [−0,0403;+0,0018]) → ningún candidato A5 adopta.
- Δ pareado C vs v1 (test, bootstrap 1000 seed 42, `/tmp/opencode/bootstrap_pareado_c_vs_v1.py`): v1 0,2870082 / C 0,2980096; Δ +0,0110, media +0,0071, IC95 [−0,0228;+0,0300], prob_mejora 0,717 → IC cruza 0.
- Gate A clasificador (ADR-16: Δ > ruido + consistente): **no se adopta C**; negativo documentado y producto mantiene v1 (LogReg enmascarado, ADR-10); meta 0,32 pendiente de Sprint B/C.
- Registro: `metricas.json` → `ciclo_v3.a5_clasificador`; fila A5 en REGISTRO_EXPERIMENTOS; checkbox A5 marcado; bootstrap/log en `/tmp/opencode/`.
- Siguiente: Sprint B — B1 fine-tune reranker con pares de citas (excluyendo las 90 del benchmark), B2 retriever denso de dominio, B3 mini-gold.
## Ciclo v3 - Gate A (orquestador, 2026-09-11) — CERRADO SIN ADOPCIONES
- Sprint A completo: A1 diagnostico (recall@k + IC + techo), A2 embeddings legales ES (ALIA/BGE-M3/modernbert: senales parciales; nDCG ns -> no adopcion), A3 reranker SINAI (negativo; bge se mantiene), A4 fusion/profundidad (w05 y top-50 se mantienen; 200 pierde), A5 clasificador (ensamble/cw no superan C; C vs v1 ns -> v1 se mantiene).
- ADR-17 registra el gate con evidencia. Producto v2 intacto. Candidatos para Sprint B: ALIA embedding, reranker fine-tuned, C calibracion, profundidad con retriever mejorado.
- Leccion empirica: smokes n=10 enganaron 2/3 veces; gates solo con n=90 + IC pareado. Siguiente: Sprint B1 (fine-tune reranker con citas, excluir las 90 del benchmark).
