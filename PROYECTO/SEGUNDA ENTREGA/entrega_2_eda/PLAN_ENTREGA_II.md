# PLAN MAESTRO — Entrega 2: "EDA Avanzado, Representación y Propuesta Metodológica"

**Proyecto:** PLN — Sentencias de la Corte Constitucional de Colombia (1992–2021)
**Carpeta de trabajo:** `PROYECTO/SEGUNDA ENTREGA/entrega_2_eda/`
**Enunciado:** `.../ENTREGA II/NLP Enunciado P2.pdf` (copia legible en `plan/enunciado_p2.md`)
**Decisión de alcance (Carlos, 2026-10-04):** SOLO el tramo 1992–2021. La extensión 2022–2026 (C3) queda EXCLUIDA de esta entrega (no se analiza, no se incorpora, no se menciona como resultado).
**Mantenimiento:** este archivo se CARGA al iniciar cada fase y se ACTUALIZA al cierre de cada fase (estado + bitácora). Es la fuente de verdad del avance.

## 1. Requisitos del enunciado → artefactos

| # | Requisito (P2) | Artefacto(s) | Fase |
|---|---|---|---|
| 1 | Correcciones y modificaciones respecto a la Entrega 1 | `informe/main.tex` sección 2 | F4 |
| 2 | EDA avanzado: ontologías/tesauros/recursos léxicos; diversidad léxica; coocurrencias; bigramas/trigramas | notebook sección 9 + sección 12; `informe` sección 4; `figuras/` | F1/F4 |
| 3 | Conjunto de datos medido (corpus preprocesado + métricas, formato legible, variables relevantes, adjunto junto al informe) | `datos/corpus_medido.parquet` + `datos/corpus_medido_muestra.csv` + `datos/diccionario_datos.md`; `informe` sección 3.3 | F1 |
| 4 | Representación de los textos (seleccionar/justificar BoW/TF-IDF/embeddings/contextuales; ejemplos de vectorización) | notebook sección 11 + `fig_representacion_tfidf.png`; `informe` sección 5 | F1/F4 |
| 5 | Propuesta metodológica de los modelos (enfoque(s) de la fase final, justificación, comparación y criterios) | `informe` sección 6 | F4 |
| 6 | Presentación PPT/Canva 10–15 diapositivas | `presentacion/Entrega_2_Presentacion.pptx` | F6 |
| 7 | Formato: informe PDF + carpeta/repositorio con notebooks y datos | `informe/main.pdf` + carpeta `entrega_2_eda/` (+ zip) | F7 |

## 2. Reglas de la entrega

1. **Intocables:** `PROYECTO/PRIMERA ENTREGA/` completa; resultados congelados de `PROYECTO/SEGUNDA ENTREGA/` (`resultados/*.json`, `data/splits_*`, `modelos/`, scripts 00–28, `plan/` interno). Esta entrega NO los modifica.
2. **Sin modelos ni resultados de modelado:** el enunciado pide una *propuesta metodológica*; los experimentos de modelado quedan fuera (no se citan métricas de modelos en informe/notebook/presentación).
3. **Alcance temporal 1992–2021** (2021 parcial: 20 documentos).
4. **Entorno:** `/home/carlos/venv/bin/python`; LaTeX: `/home/carlos/.TinyTeX/bin/x86_64-linux/` (pdflatex + bibtex).
5. **Cifras:** toda cifra proviene de `PRIMERA ENTREGA/resultados/metricas.json`, del EDA extendido (`PRIMERA ENTREGA/resultados/eda_extendido/`) o del notebook ejecutado; el informe cita la fuente. Validación con agentes revisores (ml-reviewer / ml-qa-engineer).
6. Este plan se carga/actualiza fase a fase; los handoffs de agentes lo referencian.

## 3. Estructura de la carpeta

```
entrega_2_eda/
├── PLAN_ENTREGA_II.md          ← este archivo
├── notebook/
│   └── Entrega_2_EDA_Representacion.ipynb   (copia consolidada del notebook E1 + secciones nuevas 10–13)
├── informe/
│   ├── main.tex / main.pdf / references.bib / tg_usta.cls / logo_javeriana.png
│   └── README_compilar.md
├── presentacion/
│   ├── Entrega_2_Presentacion.pptx
│   └── guion_presentacion.md
├── figuras/                    (figuras E1 regeneradas + nuevas)
├── datos/
│   ├── corpus_medido.parquet   (23.742 filas: texto, texto_limpio, métricas)
│   ├── corpus_medido_muestra.csv
│   ├── diccionario_datos.md
│   └── lemas_por_doc.parquet   (copia para reproducibilidad)
├── resultados/
│   ├── metricas_entrega_ii.json
│   └── eda_extendido/          (ngramas, colocaciones, morfología, MTLD, granularidad)
├── plan/
│   ├── enunciado_p2.md         (conversión del PDF del enunciado)
│   └── nota_ontologias.md
└── experimentos/               (logs de ejecución)
```

## 4. Fases (estado)

| Fase | Tarea | Agente | Estado |
|---|---|---|---|
| F0 | Setup: estructura, plan, assets base (cls/bib/figuras) | orquestador | ✅ 2026-10-04 |
| F1 | Notebook consolidado (copia E1 + sección 10 dataset + sección 11 representación + sección 12 log-odds + sección 13 cierre; ejecutado E2E) | ml-coder | ✅ 2026-10-04 (notebook final: 56 celdas, 0 errores, sellado; 29/29 artefactos) |
| F4a | Informe: borrador v1 (secciones 1–4, sección 6 completos; sección 3.3/sección 5 esqueleto con TODO) — EN PARALELO con F1 | scientific-writer | ✅ 2026-10-04 (main.pdf 19 pp; 6 TODO(F4b)) |
| F2 | Revisión del notebook (código, rutas, consistencia, E1 intacto) | ml-reviewer | ✅ 2026-10-04 APROBADO C/OBS (2 mayores, 2 menores) |
| F2-fix | Correcciones post-revisión + corrida de sellado (paneles de nubes, F811, nota sección 11.1, log ligado al artefacto) | ml-coder | ✅ 2026-10-04 (sha256 a5e851b0…; 0 errores; paneles OK) |
| F3 | Auditoría integral: notebook sellado + dataset + cifras del informe (F3+F5 unificadas) | ml-qa-engineer | ✅ A GO · B NO-GO → corregido y verificado (7/7 fixes) |
| F3-fix | Correcciones QA-B (masa TF-IDF, tiempos sellados, MTLD/rango, etiqueta lemas; informe + presentación) + recompilar | scientific-writer | ✅ 2026-10-04 (informe 22 pp recompilado; deck 12 slides regenerado) |
| F4b | Informe: completar sección 3.3/sección 5 + cifras nuevas + compilar PDF final | scientific-writer | ✅ 2026-10-04 (22 pp; 0 TODO; 16 refs; compilación limpia) |
| F5 | Auditoría de cifras del informe (se unifica con F3) | ml-qa-engineer | ✅ (unificada en F3: B → corregida y verificada) |
| F6 | Presentación 10–15 slides (.pptx) | scientific-writer | ✅ 2026-10-04 (12 slides; 8 figuras; verificado) |
| F7 | QA final (checklist P2 1–7) + empaquetado + plan al día | ml-qa-engineer + orquestador | ✅ 2026-10-04 (README + zips 51,5 MB / 2,0 GB + checklist) |
| F8a | Figuras adicionales en notebook (longitudes, tipos×año, Zipf, MTLD×longitud) + re-ejecución y re-sellado | ml-coder | ✅ 2026-10-04 (4 figuras; Zipf −1,4676; Spearman −0,2628; sellado) |
| F8a2 | Notebook: "§"→"sección", rutas genéricas, quitar 13.1, verificación de paneles + re-ejecución/sellado | ml-coder | ✅ 2026-10-04 (0 "§"; sha256 bb99c5d4…; 0 errores) |
| F8b | Integración de las 4 figuras en el informe con explicación + recompilar | scientific-writer | ✅ 2026-10-05 (25 pp; figs 4/5/8/10; 0 errores) |
| F8c | Verificación (sellado, figuras, cifras nuevas) + re-empaquetado de zips | ml-qa-engineer + orquestador | ✅ 2026-10-05 (QA GO; fix de tiempos aplicado; zips 53,3 MB / 2,0 GB) |
| F8d | Presentación re-estilizada con la plantilla de E1 (`PLN_PROYECTO.pptx`): resumen E1 + estado del arte + trabajo E2 | scientific-writer | ✅ 2026-10-04 (15 slides; guion; bug de tarjetas corregido; verificado) |

## 5. Spec del notebook (secciones)

- secciones 0–8: copia del notebook E1 (avances previos) con rutas portables; salidas a `figuras/` y `resultados/` de la entrega. No se altera la lógica de análisis.
- sección 9: EDA extendido (n-gramas, colocaciones LLR, lemas/stems, POS, MTLD, granularidad, ontologías) — ya existe en el notebook E1; adaptar rutas y dejar la carpeta autocontenida.
- sección 10 (nueva): Conjunto de datos medido → `datos/corpus_medido.parquet` con columnas: `id, tipo, numero, ano, texto, texto_limpio, n_chars_texto, n_palabras_raw, n_palabras_clean, n_stopwords, n_lemas, mtld` (23.742 filas) + muestra estratificada CSV + diccionario + asserts contra `metricas.json`.
- sección 11 (nueva): Representación — justificación BoW/TF-IDF/embeddings/contextuales; TF-IDF sobre lemas (`min_df=3, max_features=100_000, ngram_range=(1,2), sublinear_tf=True`) sobre los 23.742 documentos; ejemplos de vectorización (3 docs T/C/SU); figura `fig_representacion_tfidf.png`; demo opcional de embeddings (e5 local, 5 docs) → `fig_embeddings_ejemplo.png`.
- sección 12 (nueva): Evolución léxica por periodos (log-odds 1992–2001 / 2002–2011 / 2012–2021) → `fig_logodds_periodos.png` (cumple la promesa de la Entrega 1).
- sección 13 (nueva): Cierre, reproducibilidad, resumen de entregables.
- Código pesado (pasada completa spaCy/stems/MTLD) incluido en el notebook tras flag `RECALCULAR_MORFOLOGIA=False` (por defecto lee artefactos precomputados; documentado).

## 6. Spec del informe (estructura canónica)

1. Resumen + palabras clave.
2. sección 1 Introducción (contexto, motivación, objetivo del proyecto, qué trae esta entrega).
3. sección 2 Correcciones y modificaciones respecto a la Entrega 1 (corpus / visualizaciones / planteamiento + razones).
4. sección 3 Corpus y preprocesamiento (3.1 fuente y depuración; 3.2 pipeline de limpieza; 3.3 conjunto de datos medido; 3.4 medidas globales).
5. sección 4 EDA avanzado (4.1 n-gramas y colocaciones; 4.2 diversidad léxica MTLD; 4.3 morfología lemas/stems/POS; 4.4 evolución léxica por periodos; 4.5 recursos léxicos y ontologías; 4.6 granularidad).
6. sección 5 Representación de los textos (5.1 familias y criterios; 5.2 BoW/TF-IDF; 5.3 ejemplos; 5.4 embeddings/contextuales; 5.5 decisión).
7. sección 6 Propuesta metodológica (6.1 tarea y enfoque; 6.2 datos/etiquetas y validación; 6.3 modelos a comparar y criterios; 6.4 riesgos; 6.5 trabajo futuro).
8. sección 7 Conclusiones.
9. Referencias.
- Figuras objetivo: sentencias por año; top-20 (2 paneles); nubes por año; n-gramas; colocaciones; lemas-vs-stems; POS; MTLD; granularidad; representación TF-IDF; log-odds.
- Tablas: ficha del corpus; pipeline; medidas globales; n-gramas + validación; morfología; MTLD; diccionario del dataset; comparación de modelos y criterios.

## 7. Spec de la presentación (12 slides)

1 Título · 2 Problema/objetivo · 3 Corpus y correcciones vs E1 · 4 Limpieza · 5 Medidas globales + distribución temporal · 6 N-gramas/colocaciones · 7 Diversidad léxica y morfología · 8 Recursos léxicos/ontologías · 9 Dataset entregado · 10 Representación · 11 Propuesta metodológica · 12 Cierre.

## 8. DoD (criterios de aceptación)

- Notebook ejecuta de principio a fin desde kernel limpio (nbconvert) y todas las salidas quedan embebidas; verificado por ml-qa.
- `corpus_medido.parquet` = 23.742 filas, sin nulos en claves, cifras reconciliadas con `metricas.json` (auditoría QA).
- Informe compilado a PDF sin errores; cada cifra trazable a una fuente; referencias verificadas.
- Presentación 10–15 diapositivas.
- E1 y E2 intactos (verificación de no-modificación).
- Plan actualizado; bitácora completa.
- **Estado final: ✅ cumplido (2026-10-05)** — auditorías QA-A/QA-B/QA-final con veredicto GO (observaciones corregidas); figuras adicionales (F8) integradas en notebook e informe y verificadas.

## 9. Cifras canónicas de referencia (fuente de verdad: `metricas.json` + EDA extendido; verificar siempre)

- Corpus: 23.742 sentencias (T 17.099 / C 6.246 / SU 397), 1992–2021 (2021 = 20 docs).
- Palabras: raw 260.988.316 · limpias 250.012.101 · sin stopwords 128.806.664 (stopwords ≈ 48,5 %).
- Promedios por doc: raw 10.992,68 · limpias 10.530,37.
- Vocabulario: 224.202 tipos (sin stopwords; `metricas.json`) / 224.468 unigramas (EDA).
- N-gramas top: bi "corte constitucional" 286.786 · tri "accion de tutela" 344.893. Validación manual: LLR 75 % vs freq 55 % vs PMI 0 % (20 casos).
- Morfología: tokens 224.468 / lemas 195.959 (87,3 %) / stems 121.696 (54,2 %). POS: NOUN > ADP > DET > ADJ.
- MTLD global 61,03 (T 61,63 > C 59,54 > SU 58,88).
- Granularidad: sentencia 97,17 % con marcador vs subsentencia 4,64 % (criterio +15 pp no superado → sentencia completa).

## 10. Bitácora

- **2026-10-04 · F0:** alcance congelado en 1992–2021 (decisión de Carlos); enunciado convertido a Markdown; entorno verificado (nbconvert 7.17, python-pptx 1.0.2, TinyTeX TeX Live 2026, e5 en caché HF); estructura de carpetas creada; assets base copiados (cls, bib, logo, figuras E1); plan guardado.
- **2026-10-04 · F1 + F4a lanzadas en paralelo:** ml-coder (notebook consolidado; sesión `ses_ef642c1f1ffeM8Y9yckeJ36oAQ`) y scientific-writer (borrador de informe v1; sesión `ses_ef642c1f2ffelj9QdHkpcbWhTF`). Ajuste de config: scientific-writer `mode: primary→all` (backup `.bak_entrega2`) para permitir su uso como subagente.
- **2026-10-04 · F4a ✅:** borrador v1 del informe compilado con TinyTeX: `informe/main.pdf` (19 páginas, 15 referencias, secciones 1–7; sin resultados de modelos; alcance 1992–2021 verificado). Pendientes F4b: 5 bloques `TODO(F4b)` (verificación del dataset medido, agregados, log-odds, ejemplos de vectorización, demo de embeddings). El notebook (F1) sigue en ejecución.
- **2026-10-04 · F1 (avance):** notebook construido (55 celdas: 25 markdown + 30 código) y validado con smokes reales: cifras canónicas reproducidas exactas (raw 260.988.316 / clean 250.012.101 / stopwords 121.205.437 / sin stopwords 128.806.664; limpieza idéntica a E1 verificada documento a documento), `corpus_medido.parquet` (23.742 filas, ~1,5 GB) con asserts OK, TF-IDF (23.742×100.000; esparsidad 95,66 %; vocab 100.000), log-odds 42.362 filas, muestra estratificada + diccionario. Pendiente: ejecución E2E completa con nbconvert y verificaciones finales (sesión ml-coder reanudada).
- **2026-10-04 · F1 ✅:** ejecución E2E completa con nbconvert (run final 11,2 min, EXIT=0, 0 celdas de error, salidas embebidas; notebook final = versión ejecutada). Figuras nuevas: `fig_representacion_tfidf.png`, `fig_logodds_periodos.png`, `fig_embeddings_ejemplo.png` (embeddings e5 en GPU). Dataset medido 23.742 filas con sumas exactas (raw 260.988.316 · clean 250.012.101 · stopwords 121.205.437 · lemas 248.533.344 · chars 1.640.749.780). TF-IDF (23.742×100.000; nnz 103.045.759; esparsidad 95,66 %). Log-odds alineado con `09_diacronia.py` (solapamiento top-10 7/10–9/10; diferencias por lema/forma superficial y empates; nota de limitación en sección 12.3). No-modificación verificada (E1 y datos congelados intactos; los archivos recientes fuera de `entrega_2_eda/` pertenecen al job v3/C3 concurrente). Log: `experimentos/e2e_nbconvert.log`.
- **2026-10-04 · F2 ✅ (APROBADO CON OBSERVACIONES):** sin bloqueantes; fidelidad a E1 verificada (celdas 0–9 solo rutas/formato); asserts/cifras OK; sin emojis; PRIMERA ENTREGA limpia en git. Mayores: (1) el log de la corrida no quedaba ligado al artefacto final (diferencia de 661 bytes; temporal renombrado) → corrida de sellado; (2) el informe cita paneles de nubes anuales que el notebook no regenera → añadir generación en sección 6. Menores: import duplicado F811 (sección 12.1), aclarar ajuste train-only del vectorizador (sección 11.1), fila "párrafo" degenerada (nota para el informe). Nota: hay un job v3/C3 concurrente (sesión E2) que escribe en `SEGUNDA ENTREGA/scripts` y resultados ajenos; no interfiere con `entrega_2_eda/`.
- **2026-10-04 · F2-fix + F4b lanzados en paralelo:** ml-coder (correcciones + corrida de sellado con log ligado al artefacto) y scientific-writer (resolver los `TODO(F4b)` del informe con las cifras reales del notebook, integrar las 3 figuras nuevas y recompilar).
- **2026-10-04 · F4b ✅:** informe final completado con cifras reales del notebook: sección 3.3 (dataset: 23.742 filas, sin nulos, sumas reconciliadas; muestra 100 filas 72T/24C/4SU; diccionario 14 variables), sección 4.4 (log-odds: 7.166/9.779/6.797 docs por periodo; 192.277 formas únicas; top-10 sube/baja de ambas transiciones + figura + nota de limitación), sección 5.3 (3 docs T/C/SU: nnz 7.712/11.027/15.820; similitudes coseno 0,242/0,253/0,251; figura), sección 5.4 (e5 768-dim; similitud media 0,899; figura), sección 4.6 (nota párrafo degenerado). Ref nueva `hamilton2016diachronic`; 16/16 entradas citadas; `main.pdf` final: **22 páginas**, compilación limpia (0 errores, 0 refs rotas). Extensión aceptada (>objetivo 12–16 pp; cubre los 7 puntos del enunciado).
- **2026-10-04 · F6 lanzada en paralelo** con el sellado del notebook: presentación de 12 diapositivas (guion + .pptx) desde el informe final. **F3+F5 se unifican** en una auditoría integral (notebook sellado + dataset + cifras del informe) al terminar el sellado.
- **2026-10-04 · F6 ✅:** presentación completa en `presentacion/`: `guion_presentacion.md` (12 diapositivas + notas del orador), `generar_presentacion.py` (reutilizable, 16:9, python-pptx) y `Entrega_2_Presentacion.pptx` (12 diapositivas, 8 figuras embebidas, verificado con python-pptx + LibreOffice; sin desbordes). Contenido consistente con el informe final; alcance 1992–2021; sin resultados de modelos. Pendiente: sellado del notebook (F2-fix) → auditoría integral F3+F5 → empaquetado F7.
- **2026-10-04 · F2-fix ✅ (sellado):** correcciones aplicadas (3 paneles de nubes en sección 6 con criterio ≥50 docs/año y stopwords gramaticales+jurídicas; F811 corregido — ruff limpio; nota de fuga en sección 11.1). Corrida de sellado: 56 celdas / 31 de código, 0 errores, ~11,4 min; notebook final = salida exacta de la corrida (sha256 `a5e851b009220bc5eba7b3c3ec68ab47c8d14aae291eb9cc7a50f9f048dc333a`, 12.779.673 bytes) con log ligado y verificado. 29/29 artefactos; dataset 23.742 filas; `metricas_entrega_ii.json` (27 secciones) y `entorno_ejecucion.json` regenerados. Informe recompilado para tomar las figuras regeneradas.
- **2026-10-04 · F3+F5 (auditoría integral) lanzadas en dos partes paralelas** (ml-qa-engineer): A = sellado + dataset + cifras base; B = cifras del informe + presentación.
- **2026-10-04 · F3+F5-A ✅ (GO):** sha256/bytes log↔notebook exactos; 56 celdas/31 código/0 errores; paneles de nubes de la corrida; `corpus_medido.parquet` 23.742 filas, esquema 12 cols, 0 nulos, sumas exactas 5/5; muestra 100 filas (12 valores `mtld` con delta ≤1 ULP por round-trip CSV — no bloqueante); cifras base notebook == E1 == JSON; verificación doc-a-doc de limpieza E1 reproducida independientemente (23.742/23.742, 0 fallos; evidencia externa `/tmp/opencode/qa_partA_*`); `metricas_entrega_ii.json` 27 secciones coherentes con las salidas embebidas; no-modificación OK (nada bajo PRIMERA ENTREGA). Observaciones menores: la celda 55 de entregables lista 2 paneles (no bloqueante); la verificación doc-a-doc queda como evidencia externa, no como assert embebido. Parte B en curso.
- **2026-10-04 · F3+F5-B ⚠️ NO-GO (corregible):** auditoría del informe + presentación: >150 cifras trazadas; enunciado cubierto (secciones 2–6); alcance/reglas OK (16/16 refs citadas, 0 «??», PDF 22 pp; 12 slides; 8 PNG íntegros). Discrepancias: **[crítico]** masa acumulada del TF-IDF mal redactada en sección 5.3 (correcto, recomputado por QA: ≈2 % de la masa en los 100 primeros rasgos, ≈17 % en los 1.000, ≈60 % en los 10.000); **[importantes]** tiempos "39,2 s"→40,3 s (sección 4.4) y "192,2 s"→192,5 s (sección 5.3/sección 5.4; la presentación repite uno); **[sugerencias]** mediana MTLD C 59,43, media 1992–1998 ≈65,0, rango real 941–992, etiquetar "mediana de lemas" (7.116). Fixes lanzados (scientific-writer). Riesgo residual: masa 2/17/60 solo en figura (no en JSON) — verificada por recomputo independiente de QA (evidencia `/tmp/opencode/qa_tfidf_*`).
- **2026-10-04 · F3-fix ✅:** 7/7 correcciones aplicadas (masa TF-IDF ≈2/17/60; tiempos 40,3 s y 192,5 s; mediana MTLD 59,43; media 1992–1998 ≈65,0; rango 941–992; "mediana de lemas" 7.116 / palabras limpias 7.158). `main.pdf` recompilado (22 pp, 0 errores/0 warnings/0 overfull); deck regenerado (12 slides). **Verificación mecánica final (orquestador):** valores antiguos ausentes y nuevos presentes en `main.tex`, `generar_presentacion.py` y `guion_presentacion.md`; PDF y PPTX con mtime nuevo. QA-B cerrado como corregido.
- **2026-10-04 · F7 (empaquetado):** `README_ENTREGA.md` creado (mapa de la carpeta, checklist 1–7 del enunciado, reproducción, dataset y estado QA); paquetes creados y verificados: `Entrega_2_proyecto.zip` (51,5 MB, 78 archivos; sin los 2 Parquet grandes) y `Entrega_2_proyecto_completo.zip` (2,0 GB, 80 archivos; todo, con datos).
- **2026-10-04 · F8 (solicitud de Carlos):** añadir gráficas adicionales derivadas de las tablas existentes e integrarlas al informe con su explicación. Lanzada **F8a** (ml-coder): 4 figuras nuevas — distribución de longitudes por documento, composición por tipo y año, ley de Zipf (log-log) y MTLD vs longitud — con re-ejecución y re-sellado del notebook; las cifras nuevas (pendiente Zipf, Spearman) se registran en `metricas_entrega_ii.json`. Después: F8b (informe) y F8c (verificación + re-zips).
- **2026-10-04 · F8d (solicitud de Carlos):** rehacer la presentación con el estilo de la Entrega 1 (`PLN_PROYECTO.pptx`): resumen condensado de las diapositivas de E1 + estado del arte (según el enunciado de la Entrega 2) + el trabajo realizado en esta entrega; se conserva el estilo de la plantilla. Lanzada (scientific-writer) en paralelo con F8a. Además (solicitudes de Carlos aplicadas): "§"→"sección" en README y plan (informe y presentación ya estaban limpios), rutas genéricas en la sección de reproducción del README, y en cola para el notebook: "§"→"sección", rutas genéricas y eliminación de la subsección 13.1.
- **2026-10-04 · F8a ✅:** 4 figuras nuevas en el notebook (8 celdas): longitudes (mediana 7.158 / media 10.530; T 6.329 · C 9.830 · SU 21.626; máx 451.252), tipos×año (T 72,0 % · C 26,3 % · SU 1,7 %), Zipf (pendiente −1,4676; R² 0,9717) y MTLD×longitud (Spearman −0,2628). `metricas_entrega_ii.json` con 29 secciones (2 nuevas). Sellado: 64 celdas / 35 de código, 0 errores, ~13 min; sha256 `37481512e3339f35f8ac4a00f9cb4ff5a5fa53d496ef61290c15e00c7e577f7a` (12.802.249 bytes); log ligado verificado; 32/32 artefactos; corpus intacto. Lanzada **F8a2** (cambios solicitados por Carlos en el notebook: "§"→"sección", rutas genéricas, quitar 13.1, verificación de paneles + re-ejecución/sellado).
- **2026-10-04 · F8d ✅:** presentación rehecha con el estilo de E1 (`PLN_PROYECTO.pptx`): 15 diapositivas (título, contenido, resumen E1 ×2, estado del arte ×2, trabajo E2: dataset ×2, EDA avanzado ×4, representación, metodología, conclusiones), 34 imágenes embebidas; guion reescrito (15/15); backup del diseño anterior en `backup_diseno_propio/`. Verificación: sin "§", sin emojis, sin resultados de modelos; spot-check de cifras OK; bug real corregido en las tarjetas de las diapositivas 8 y 15 (texto de cuerpo perdido) + aserciones de regresión; "nueve pasos"→"ocho pasos"; MD5 de las 7 figuras E2 coincide con `figuras/`; LibreOffice convierte sin error. Nota: la sesión original del redactor quedó inutilizable por el límite de imágenes (31>30) y el cierre se hizo con una sesión nueva.
- **2026-10-04 · F8b lanzada** (sesión nueva del redactor): integrar las 4 figuras nuevas al informe con su explicación + recompilar.
- **2026-10-04 · F8a2 ✅:** notebook final con los ajustes solicitados por Carlos: 0 símbolos "§" (todo "sección X"); rutas genéricas en las instrucciones (`python`/`jupyter` + nota de requisitos); subsección 13.1 eliminada y renumerada (13.1 Re-ejecución, 13.2 Límites conocidos); verificación de entregables con los 3 paneles de nubes; warnings de torch/tqdm suprimidos en la sección 11.4. Sellado: 64 celdas / 35 de código, 0 errores, ~13 min; sha256 `bb99c5d456cd22b14da88641a147cf92f7e47cc30f514cb032cd34f587482b83` (12.801.043 bytes); log ligado verificado; 10/10 figuras clave; corpus y métricas intactos.
- **2026-10-05 · F8b ✅:** informe con las 4 figuras nuevas integradas (sección 3.4: longitudes y tipos por año; sección 4.1: Zipf; sección 4.2: MTLD vs longitud), con párrafos interpretativos, captions y referencias cruzadas; recompilado (exit 0; 0 errores/warnings/overfull; 0 referencias sin resolver); PDF final **25 páginas** (figuras 4, 5, 8 y 10 nuevas); sin "§"; sin resultados de modelos. Lanzada **F8c** (QA final + re-empaquetado).
- **2026-10-05 · F8c (QA final) ✅ GO con 1 observación:** verificados sellado (sha256 `bb99c5d4…` vs log; re-producción byte-idéntica), 64 celdas/35 código/0 errores/0 "§", 4 figuras nuevas + 3 paneles (referenciadas 21/21), informe 25 pp con figuras 4/5/8/10 y 0 "??", deck 15 slides (7 figuras con MD5 idéntico), cifras nuevas cruzadas notebook↔JSON↔informe y recomputadas independientemente desde los parquets (mediana 7158; media 10530,37; ρ −0,2628; MTLD 61,03), no-modificación OK. **Discrepancia única (no bloqueante):** el tiempo de ajuste TF-IDF citado (192,5 s) quedó desactualizado tras la re-ejecución F8a2 (sellado: 211,9 s) → en corrección (informe + deck + guion).
- **2026-10-05 · F8c ✅ (cierre):** observación corregida (TF-IDF 192,5→"≈212 s" con valor sellado 211,9; log-odds 40,3→40,5 s) en informe (recompilado, 25 pp, 0 errores), deck y guion (regenerados, 15 slides); verificación mecánica: 0 ocurrencias de los valores antiguos. Paquetes finales regenerados: `Entrega_2_proyecto.zip` (53,3 MB, 86 archivos) y `Entrega_2_proyecto_completo.zip` (2,0 GB, 88 archivos). **Entrega cerrada.**
