# Plan — Entrega 1: Corpus, Idea de Proyecto y Textometría

Proyecto PLN (Javeriana) — sentencias de la Corte Constitucional de Colombia (1992–2021).
Dataset: https://huggingface.co/datasets/Manuel/sentencias-corte-cons-colombia-1992-2021
Entorno: `/home/carlos/venv` · Plan aprobado por Carlos el 2026-09-05.

## Estado por fase (se actualiza al terminar cada fase)

- [x] **Fase 0 — Preparación**: exploración máquina (plan/00_maquina.md, config/hardware.json); descarga CSV completa a `data/Juris1992_2022.csv` (1.67 GB); verificación estructural y de parseo (23.742/23.750 = 99.97%); detección de 8 filas no-sentencia (6 autos + 2 HTML).
- [ ] **Fase 1 — Investigación SOTA** (nlp-specialist): estado_arte.md + entradas .bib verificables.
- [x] **Fase 1 — Investigación SOTA**: COMPLETADA por nlp-specialist. `estado_arte.md` (6 refs núcleo + 3 complementarias, todas verificadas por DOI/URL) y `references.bib` (10 entradas limpias, incl. @misc dataset HF). Nota clave: ningún paper indexado usa el dataset HF exacto; se cubre con los trabajos más cercanos (corpus relatoría CCC colombiana, NLP legal español). Recomendaciones: stopwords sin tildes + jurídicas; top-N global en barras + frecuencias normalizadas; lematización diferible.
- [ ] **Fase 2 — Notebook consolidado** (ml-coder): pipeline completo de textometría global, corregido y re-ejecutado.
- [x] **Fase 2 — Notebook consolidado**: COMPLETADA por ml-coder. `primera_entrega.ipynb` (23 celdas, ejecutado 5m25s, 13/13 celdas OK). Números reales: 23.742 parseados, total palabras raw REAL = 260.988.316 (el 271.907.978 hardcodeado era extrapolación de muestra; −4%), top-1 sin stopwords = 'articulo'. 5 figuras en figuras/ + resultados/metricas.json + anexo_corpus_muestra.csv (25 docs). Tras revisión: filtro de tokens 1-letra/abreviaturas aplicado (top-20 SIN limpio; total clean 250.012.101) y notas md añadidas.
- [ ] **Fase 3 — QA** (ml-reviewer + ml-qa-engineer): ruff, lógica, DoD vs enunciado.
- [x] **Fase 3 — QA**: COMPLETADA. ml-reviewer: APROBADO CON CAMBIOS MENORES → aplicados por ml-coder y re-ejecutado (5m40s; top-20 sin stopwords sin t/c/p/art; total clean 250.012.101). ml-qa-engineer: APROBADO CON OBSERVACIONES (15 asserts OK; 4 sugerencias no bloqueantes → 2 ya aplicadas al notebook (nota raw previo en md21 y nota nombre archivo en md1) + nota_raw_previo en metricas.json; las 2 cosméticas se resuelven en Fase 4: nombres PNG formales y formato numérico uniforme en el PDF).
- [ ] **Fase 4 — Informe LaTeX + PDF** (TinyTeX) + zip final con anexo corpus.
- [x] **Fase 4 — Informe LaTeX + PDF**: COMPLETADA. main.tex reescrito (secciones 1-5: Idea, Corpus con fuente HF/URL/licencia/depuración/fragmentos reales T-293/97, C-448/98, SU-476/97, Textometría con limpieza/medidas reales/top-20 con figuras/distribución temporal/nubes por año en 3 paneles, Estado del arte con 10 citas apalike resueltas, Conclusiones). PDF Entrega_1_Informe.pdf = 8 páginas (límite máx) compilado con TinyTeX (solo se instaló hyphen-spanish; fix \cite harvard→kernel para apalike). Zip regenerado (13 archivos planos, 22.5 MB): PDF + main.tex + tg_usta.cls + references.bib + logo + 5 figuras (nubes segmentadas 1992-2001/2002-2011/2012-2020 por legibilidad) + anexo_corpus_muestra.csv + ENLACE_CORPUS.txt. PDF también copiado a la raíz PRIMERA ENTREGA/.
- [ ] **Fase 5 — Cierre**: tabla de cumplimiento.
- [x] **Fase 5 — Cierre**: COMPLETADA. Tabla de cumplimiento vs enunciado en este archivo (abajo) y resumen entregado a Carlos.

## Contexto del proyecto (resumen)
- Objetivo E1 (enunciado PDF): 1) idea de proyecto (dominio, problema, justificación, 2-3 preguntas); 2) corpus descrito con evidencia (fuente, características, fragmentos, formato); 3) textometría básica (limpieza, nº docs, total palabras, promedio, top-20 con/sin stopwords, nube/barras); 4) PDF máx 8 págs + anexo corpus.
- Artefactos actuales revisados: `primera_entrega.ipynb` (42 celdas) y `Entrega_1_proyecto.zip` (main.tex + figuras + references.bib + tg_usta.cls). Brechas detectadas documentadas en contexto.md (git/PRIMERA ENTREGA/plan/contexto.md).

## Entregables esperados al cierre
1. `primera_entrega.ipynb` consolidado y re-ejecutado (textometría global real sobre el corpus).
2. `figuras/*.png` regeneradas (top-20 con/sin stopwords, nubes) + `resultados/metricas.json`.
3. `estado_arte.md` (investigación) y `references.bib` limpio.
4. `main.tex` corregido → `Entrega_1_proyecto.pdf` (≤ 8 págs) compilado con TinyTeX.
5. `Entrega_1_proyecto.zip` regenerado con PDF + fuentes + figuras + anexo corpus (muestra CSV + link HF).

## Tabla de cumplimiento vs enunciado (Fase 5, 2026-09-05)

| Requisito del enunciado | Evidencia | Estado |
|---|---|---|
| 1. Idea: dominio/área | Sección 1.1 main.tex (lenguaje jurídico, Corte Constitucional) | OK |
| 1. Problema PLN | Sección 1.2 (extracción de conocimiento de corpus extenso sin metadatos; 3 frentes) | OK |
| 1. Justificación | Sección 1.3 (volumen, escasez de recursos legales ES-LatAm, citas) | OK |
| 1. ≥2-3 preguntas investigación | Sección 1.4: 4 preguntas | OK |
| 2. Fuente del corpus | Sección 2.1: repositorio HF con URL, autor, licencia CC-BY-4.0, nota archivo vs cobertura | OK |
| 2. Características (docs/tamaño/idioma/tipo) | Sección 2.2: 23.742 sentencias T/C/SU, español, jurídico, ~11.000 palabras | OK |
| 2. Ejemplos/fragmentos | Sección 2.4: 3 fragmentos literales verificados en el anexo | OK |
| 2. Formato de almacenamiento | Sección 2.3: CSV 2 columnas, depuración 99,97 %, 8 excluidas documentadas | OK |
| 3. Limpieza básica | Sección 3.1: 8 pasos documentados (minúsculas, ñ, NFD, filtro, stopwords normalizadas 306) | OK |
| 3. Nº documentos / total palabras / promedio | Sección 3.2 tabla: 23.742 / 260.988.316 / 10.992,68 (real vs 271.9M extrapolado descartado) | OK |
| 3. Top-20 con y sin stopwords | Sección 3.3 + fig_top20_con/sin_stopwords.png (corpus completo, sin artefactos t/c/p/art) | OK |
| 3. Visualizaciones (nube o barras) | Barras top-20, barras por año, nubes anuales (3 paneles) | OK |
| 4. PDF máx 8 páginas | Entrega_1_Informe.pdf = 8 páginas (pdfinfo/pypdf verificado) | OK |
| 4. Anexo corpus | ENLACE_CORPUS.txt (URL HF) + anexo_corpus_muestra.csv (25 sentencias T/C/SU) en zip | OK |
| Extra: referencias científicas | estado_arte.md (6 núcleo + 3 complementarias verificadas) + 10 citas apalike en sección 4 | OK |
| Extra: notebook consolidado re-ejecutado | primera_entrega.ipynb (23 celdas, ejecutado 5m40s, 0 errores, figuras + metricas.json) | OK |

## Verificación de independencia del notebook (2026-09-05, petición de Carlos)
- Auditoría de dependencias: el notebook NO tiene rutas absolutas (grep /home/carlos = 0); no lee nada de fuera de su directorio salvo el CSV del corpus.
- Ajustes aplicados para autocontención (celda 3 de configuración): bootstrap NLTK stopwords (descarga automática si faltan), auto-instalación de wordcloud si no está, y descarga automática del CSV desde Hugging Face (~1,7 GB) si no existe en data/ (con mensaje de error accionable). Nueva celda markdown "0. Requisitos, datos y reproducibilidad".
- PRUEBA DE INTEGRACIÓN LIMPIA: copia del notebook + solo el CSV en /tmp/opencode/test_e1/ (sin ninguna otra carpeta del proyecto) → ejecución nbclient completa en 327 s, 0 errores, carpeta figuras/, resultados/ y data/procesado/ auto-creadas, resultados idénticos (23.742 docs, 260.988.316 raw, top-1 'articulo'), anexo 25 filas.
- Re-ejecución final en el proyecto: 24 celdas (13 code ejecutadas, 0 errores), metricas.json regenerado consistente. Figuras bit-idénticas a las del informe (mismos tamaños) → PDF/zip de la entrega siguen válidos.
