# README de Entrega — E2 "Juris-PLN"

**Proyecto:** Sistema de PLN sobre 23.742 sentencias de la Corte Constitucional de Colombia (1992-2021)
**Fecha de entrega:** 2026-09-09 · **Veredicto QA:** GO (`resultados/qa_final/VEREDICTO_QA.md`)

---

## 1. Qué es esta entrega

Dado un texto de entrada, el sistema: (1) valida si es providencia de la Corte Constitucional (M0), (2) clasifica su área temática constitucional en 16 clases derivadas de la CP (M4), (3) recupera los artículos de la CP vinculados, robusto a deriva temporal (M7), y (4) sitúa la sentencia en su red de precedentes con PageRank y análisis diacrónico 1992-2021 (M5/M6). El producto final es el CLI `scripts/10_evaluar_sentencia.py`.

Documentos de esta entrega:

| Documento | Ruta |
|---|---|
| Informe final (CRISP-DM) | `documento_e2/informe_final_e2.md` |
| Deck de presentación | `documento_e2/deck_e2.md` |
| Este README | `documento_e2/README_ENTREGA.md` |

## 2. Requisitos

- **Python:** 3.14.7 en `/home/carlos/venv` (venv del proyecto). Librerías clave: PySpark 4.2.0, spaCy (`es_core_news_md`), scikit-learn, transformers, torch 2.14.0+rocm7.2 (GPU AMD RX 9070 XT; el producto final es CPU-only), rank_bm25, pyarrow, optuna, SHAP.
- **Java:** 17 (`/usr/lib/jvm/java-17-openjdk`). **Obligatorio fijar `JAVA_HOME`**: Java 26 es incompatible con Spark 4.2.0 (`ClassNotFoundException: jdk.internal.ref.Cleaner`); documentado en ADR-7.
- **Hardware de referencia:** 32 hilos lógicos, 31 GB RAM, GPU AMD (solo usada para BETO y E5; el producto corre en CPU).
- **Datos de entrada:** el CSV de la Primera Entrega (`../PRIMERA ENTREGA/data/Juris1992_2022.csv`, 1.561 GB, no incluido en el zip) y acceso a la web del Senado para descargar la CP (script 01).

## 3. Cómo reproducir (orden de scripts 00 -> 12)

Todos los scripts se ejecutan desde la raíz de `SEGUNDA ENTREGA/` con el venv activado. Cada script es autónomo (1 experimento = 1 script, ADR-4), escribe sus métricas a `resultados/` y sus logs a `experimentos/`, y es reanudable o idempotente donde aplica.

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk   # obligatorio para Spark (script 00)
source /home/carlos/venv/bin/activate

# Preparación
python scripts/00_csv_a_parquet.py        # CSV -> Parquet Hive (23.742 filas; ~4 min)
python scripts/01_descarga_constitucion.py  # CP desde senado.gov.co
python scripts/02_parseo_constitucion.py   # 497 artículos -> 19 bloques
python scripts/03_extraccion_entidades.py  # NER jurídico (M2); ~250 s
python scripts/04_etiquetado_areas.py       # 16 áreas vía CP (M3); ~1 s
python scripts/05_lematizacion.py           # lemas 23.742 docs (M1); ~2 h background

# Modelado
python scripts/06_baseline_clasico.py      # M4 clásico (TF-IDF lemas + LogReg/SVC); ~2.7 h
python scripts/07_transformer.py            # M4 BETO fine-tuned; ~6.7 h GPU (guard de VRAM incluido)
python scripts/08_red_citas.py              # M5 grafo de citas + PageRank
python scripts/09_diacronia.py             # M6 áreas + log-odds + MTLD; ~36 s

# Evaluación del vínculo CP
python scripts/11_evaluar_vinculo_cp.py    # M7 benchmark v1 (30 casos)
python scripts/12_evaluar_vinculo_cp_v2.py # M7 benchmark v2 (90 casos; GPU para E5); ~209 s

# Producto final
python scripts/10_evaluar_sentencia.py --id 2195 --salida salida.json   # T-293/97
python scripts/10_evaluar_sentencia.py --archivo sentencia.txt --salida salida.json
python scripts/10_evaluar_sentencia.py --texto "..." --salida salida.json
```

Notas de reproducibilidad:

- **Semilla 42** en todos los experimentos (documentada en configs y scripts; verificada por el QA).
- **Splits UNA VEZ:** `data/splits_m4.parquet` (16.557/3.549/3.548) se generó antes de todos los experimentos M4/BETO y no debe regenerarse para comparar resultados.
- `05_lematizacion.py` y `07_transformer.py` son de larga duración: correr con `nohup` (los logs en `experimentos/` muestran el patrón usado); ambos tienen checkpoints/reanudación.
- Verificación post-ejecución: `python scripts/checks_qa_final.py` (QA E2E) y `python scripts/checks_perf_producto.py` (perfilado de rendimiento).

## 4. Estructura de carpetas

```
SEGUNDA ENTREGA/
+-- scripts/            # Pipeline 00-12 + lib/ (componentes_producto.py, validacion_entrada.py)
+-- data/               # Parquets generados (corpus, entidades, etiquetas, lemas, splits, CP)
+-- resultados/         # JSONs de métricas (fuente de verdad de todas las cifras)
|   +-- qa_final/       # VEREDICTO_QA.md (GO), AUDITORIA_RENDIMIENTO.md, 10 casos E2E
|   +-- producto_demo/  # 3 JSON de demo (T-293/97, C-002/98, discurso negativo)
+-- figuras/            # 9 PNG (confusión M4, SHAP, red citas, PageRank, MTLD, log-odds, áreas, deriva)
+-- modelos/            # m4_baseline (joblib) + m4_transformer (BETO completo/enmascarado)
+-- experimentos/       # Logs de cada ejecución
+-- plan/               # PLAN.md, ESTADO_Y_PENDIENTES.md, decisiones/ADR-1..10
+-- config/             # hardware.json
+-- estado_arte.md      # SOTA E1+E2 con 20 referencias verificadas
+-- references.bib      # Bibliografía BibTeX (20 entradas)
+-- documento_e2/       # Esta entrega: informe, deck, README
```

## 5. Qué contiene el zip de entrega

- `documento_e2/` (informe final CRISP-DM, deck, README).
- `scripts/` completo (00-12 + `lib/` + `checks_qa_final.py` + `checks_perf_producto.py`).
- `resultados/` completo: todos los JSON de métricas, `qa_final/` (veredicto GO + auditoría rendimiento + 10 casos), `producto_demo/` (3 JSON), informes de cobertura (NER, etiquetado, constitución).
- `figuras/` (9 PNG).
- `plan/` (PLAN.md, ESTADO_Y_PENDIENTES.md, 10 ADRs), `estado_arte.md`, `references.bib`, `config/`.
- `modelos/` (joblib del baseline M4; los checkpoints BETO pueden excluirse por tamaño si se prefiere, son regenerables con `07_transformer.py`).
- **No incluido:** `data/` (regenerable con los scripts 00-05 desde el CSV de E1, que tampoco se incluye por tamaño/licencia de distribución) y `experimentos/` logs (opcionales).

## 6. Cifras canónicas (verificadas por QA 16/16)

| Componente | Métrica | Valor |
|---|---|---|
| Corpus | Sentencias (T/C/SU) | 23.742 (17.099/6.246/397) |
| Etiquetado | Cobertura | 99,63 % (16 clases) |
| M0 | Accuracy | 1.0 (52 casos) |
| M4 | LogReg macro-F1 (enmascarado) | 0.2870 (gap leakage -0.0018) |
| M4 | BETO macro-F1 (enmascarado) | 0.2042 (gap +0.0039) |
| M7 | BM25 v2 F1@5 / MRR | 0.218 / 0.717 (+84 % vs v1) |
| M5 | Aristas resueltas / gigante | 456.321 (84,5 %) / 67,6 % |
| M6 | MTLD 1992 -> 2021 | 64.9 -> 56.8 |
| Producto | Tiempo por evaluación | ~9 s (8-12 s frío) |

## 7. Incidentes conocidos y soluciones (resumen)

- Java 26 incompatible con Spark 4.2.0 -> `JAVA_HOME` a Java 17 (ADR-7).
- OOMs por contención de GPU/RAM con procesos externos -> guards de VRAM (espera automática de 3h05m en BETO), checkpoints por id y reanudación (3 veces exitosa); LightGBM sustituido por LogReg.
- spaCy `max_length` 1M < documento máximo 3M chars -> `max_length=5M` + guard por documento.
- Enmascarado anti-leakage v1 inválido (0.7 citas/doc vs 10.7 reales) -> detectado por auditoría adversarial, corregido en v2; prueba de oro confirma ausencia de leakage.

## 8. Punto de partida sugerido

1. Leer `documento_e2/informe_final_e2.md` (15 min) para el panorama completo.
2. Ver el producto en acción: `python scripts/10_evaluar_sentencia.py --id 2195 --salida /tmp/demo.json` y comparar con `resultados/producto_demo/caso_a_T-293-97.json`.
3. Para revisión metodológica: `plan/decisiones/` (ADRs 1-10) y `resultados/qa_final/VEREDICTO_QA.md`.

---

## 9. Ciclo v2 (2026-09-11)

La implementación descrita arriba queda congelada como **v1** (ADR-11); el ciclo v2 añade mejoras incrementales documentadas en `plan/PLAN_v2_ciclo_mejora.md` y en el informe `documento_e2/informe_final_ciclo_v2.md`. Resumen: F1@5 del vínculo CP 0.218 -> 0.425 (híbrido top-50 + reranker + hubs α=1.5), contaminación de transitorios 24,4 % -> 0, BETO chunk-mean 0.2433 (+17 % vs truncado, sin superar a LogReg), Fase U/EuroVoc sin adopción y QA v2 GO con drift 0.

**Artefactos nuevos:**

- Informe v2: `documento_e2/informe_final_ciclo_v2.md`; deck ampliado con 8 slides ("Sección Ciclo v2").
- Benchmarks: `resultados/vinculo_cp_benchmark_v3.json`, `_v3_limpio.json`, `_v4.json`, `_v5.json`, `_v6.json`, `_v7.json`, `_v8.json`; `resultados/m4_jerarquico.json`, `m4_eurovoc.json`, `m4_transformer_chunks.json`; métricas consolidadas en `resultados/metricas.json` (sección `ciclo_v2`).
- QA v2: `resultados/qa_final_v2/` (VEREDICTO_QA.md con GO, 10 casos E2E, logs y entradas); script `scripts/checks_qa_final.py` (v2; el v1 se conserva como `checks_qa_final_v1.py`).
- Decisiones: ADRs 11-15 en `plan/decisiones/`; 26 referencias en `references.bib` (6 nuevas del ciclo).
- Figuras del ciclo en `figuras/` (contaminación v3-v8, deriva v3-v8, hubs α v7/v8, variantes U/EuroVoc, comparación sucio/limpio).
- Scripts nuevos `13`, `14`, `15`; modificados `02`, `06`, `07`, `10`, `12` y `lib/` (`componentes_producto.py`, `reranker_bge.py`, `chunks_beto.py`).

**Cómo reproducir el ciclo v2** (desde `SEGUNDA ENTREGA/` con el venv activado):

```bash
# 1) Clasificación Fase U / W-b: variantes A (jerárquica), B (multietiqueta), C (calibración), D/E (EuroVoc)
python scripts/06_baseline_clasico.py --fase-u

# 2) BETO chunk-mean (Fase V): pre-cache de chunks + entrenamiento
python scripts/14_precache_chunks_beto.py
python scripts/07_transformer.py --estrategia chunk_mean --cache-dir data/cache_chunks_beto

# 3) Retrieval (Fases R/S1/T/W): benchmark, rerank top-50 y calibración de hubs
python scripts/12_evaluar_vinculo_cp_v2.py                    # benchmark base
python scripts/12_evaluar_vinculo_cp_v2.py --calibrar-hubs   # prior de hubs (v7/v8)

# 4) QA de regresión v2 (k=10, hubs, modos, integridad md5)
python scripts/checks_qa_final.py
```

- Trazabilidad por fases en `plan/contexto.md` (entradas "Ciclo v2" a "Fase X"); decisiones en ADR-11 a ADR-15.
- Zip del ciclo: `PROYECTO/juris_pln_e2_entrega_v2.zip`; el zip v1 (`juris_pln_e2_entrega.zip`) se conserva.
- Rollback: los artefactos v1 no se modificaron; para volver a la v1 basta restaurar los componentes desde el zip v1.

---

## 10. Ciclo v3 (2026-10-05)

El ciclo v3 ("recuperación robusta") se documenta en `plan/PLAN_v3_recuperacion_robusta.md` y en el informe `documento_e2/informe_final_ciclo_v3.md`. Resumen: Sprint A sin adopciones (ADR-17); Sprint B con modelos de dominio B1+B2 -> producto v2.2 (ADR-18); B3 gold-LLM; C3 corpus 2022-2026; C2 LLM (negativo); Fase 1 (auditoría M4, H1/H2/H2b, O1); QA v2.2 GO.

**Producto v2.2 (ADR-18).** La ruta `--rerank` usa B2 (`modelos/b2_retriever`) + B1 (`modelos/b1_reranker`) + prior de hubs α=1,5; el modo default (BM25 + α=0,75) no cambia y los fallbacks a E5/bge quedan documentados con warning. Cifras v13 (90 casos, silver): nDCG@10 0,6926/0,6907 (sin/con hubs), F1@5 0,478, MRR 0,950; Δ pareado vs v12 (B1+E5): +0,0673/+0,0617 (IC>0). Control graduado gold-LLM (B3): 0,783 [0,733;0,829]. QA v2.2 **GO** (10/10 E2E, equivalencia producto↔v13 3/3 exacta, checks v3 10/10). Rollback: zip v2 (`juris_pln_e2_entrega_v2.zip`).

**C3 — corpus 2022-2026.** `data/corpus_2022_2026/`: 2.382 providencias listadas (474/584/530/520/274 por año), 2.330 documentos preparados (NER con cobertura CP 99,96 %, etiquetado 99,96 %, lemas). Evaluación temporal de la ruta v13 sobre 100 casos nuevos (20/año): nDCG@10 0,747 [0,719;0,777] vs 0,691 histórico (delta descriptivo, no pareado) -> sin deriva temporal.

**C2 — LLM ALIA zero-shot (negativo).** Query understanding Δ −0,198 y rerank listwise Δ −0,217 (nDCG@10 frente a v13); sin adopción; opción futura: LoRA con datos propios. Artefactos en `resultados/c2_llm/` y figuras `c2_query_llm.png` / `c2_rerank_llm.png`.

**B3/O1 — packs de anotación (estado).** B3: **completado (2026-10-05)** — Carlos anotó los 820/820 candidatos (50/50 casos); gold humano: B2 nDCG@10 0,7852 [0,745;0,826] vs B1 0,6903 (Δ +0,0948; prob 1,0) -> el gate humano **CONFIRMA v2.2/ADR-18** (meta ≥ 0,70 cumplida); artefactos `resultados/gold_mini/respuestas_carlos.jsonl`, `acuerdo_gold_mini.json` (κ carlos|llm 0,345; Fleiss 0,445) y `evaluacion_gold_mini.json`; las salidas solo-LLM quedaron en `backup/`. O1: `resultados/validacion_m4/` (200 documentos, 2 proxies LLM 0,415/0,299); anotación humana aún pendiente. **Terminología:** silver = referencia automática (citas NER); gold = anotación humana; las pre-anotaciones LLM son proxies, NO gold.

**Scripts nuevos (16-31).** 16 anotador gold; 17 auditoría de etiquetas; 18 H1 ruido; 19 H2 objetivo suave; 20 H2b post-proceso; 21 pack validación M4; 22 pre-anotación LLM M4; 23 pares B1; 24 entrenamiento B1; 25 retriever B2; 26 descarga C3; 27 pre-anotación gold LLM; 28 preparación C3; 29 evaluación temporal C3; 30/31 LLM C2; `checks_qa_final_v3.py` (QA v2.2).

**Cómo reproducir las evaluaciones nuevas** (desde `SEGUNDA ENTREGA/` con el venv activado):

```bash
# Sprint A (diagnóstico, modelos legales ES, fusiones, clasificador)
python scripts/12_evaluar_vinculo_cp_v2.py --diagnostico
python scripts/12_evaluar_vinculo_cp_v2.py --a3 --rerank
python scripts/12_evaluar_vinculo_cp_v2.py --a4 --rerank
python scripts/06_baseline_clasico.py --fase-a5

# Sprint B (B1/B2 + evaluaciones v12/v13)
python scripts/23_b1_pares_reranker.py
python scripts/24_b1_entrenar_reranker.py
python scripts/12_evaluar_vinculo_cp_v2.py --b1 --rerank
python scripts/25_b2_retriever_denso.py
python scripts/12_evaluar_vinculo_cp_v2.py --b2 --rerank

# B3 (pre-anotación proxy + evaluación graduada; anotación humana con --anotar)
python scripts/27_preanotacion_gold_llm.py
python scripts/16_anotador_gold.py --evaluar

# C3 (corpus 2022-2026 + deriva temporal)
python scripts/26_c3_descarga_2022_2026.py --descargar
python scripts/28_c3_preparar_corpus.py
python scripts/29_c3_eval_temporal.py

# C2 (LLM local; requiere ollama con alia-legal)
python scripts/30_c2_llm_query.py
python scripts/31_c2_llm_rerank.py

# Fase 1 (auditoría M4 y packs O1)
python scripts/17_auditoria_etiquetas_m4.py --modo todo
python scripts/21_pack_validacion_m4.py --resumen
python scripts/22_preanotacion_llm_m4.py

# QA de regresión v2.2
python scripts/checks_qa_final_v3.py
```

- Trazabilidad: `plan/REGISTRO_EXPERIMENTOS.md` y `resultados/metricas.json` (secciones `ciclo_v3`, `b1_reranker`, `b2_retriever`, `b3_gold_llm`, `b3_check_llm`, `c2_query_llm`, `c2_rerank_llm`, `c3_deriva_temporal`, `qa_v2_2`, `o1_validacion_llm_multi`, `auditoria_etiquetas_m4`, `h1_ruido_m4`, `h2_objetivo_suave`, `h2b_postproceso`); decisiones en ADR-17 y ADR-18.
- La v2 permanece congelada (ADR-11/16): los artefactos v3 usan sufijos (`_v9+`, `c3_*`) y el zip de entrega del cierre (C5) es `PROYECTO/juris_pln_e2_entrega_v3.zip` (267 entradas; espejo del v2 + artefactos v3).

## 11. Demo interactiva (Streamlit)

```bash
/home/carlos/venv/bin/python -m streamlit run scripts/demo_streamlit.py
```

Reutiliza el pipeline del CLI (`10_evaluar_sentencia.py`) sin duplicar lógica:
acepta texto pegado o un id del corpus y muestra el JSON del producto, con un
toggle para la ruta `--rerank` de v2.2 (B2 + B1). Corre en CPU; la primera
evaluación inicializa los componentes (decenas de segundos). Verificada con el
AppTest de Streamlit (UI → producto → JSON).
