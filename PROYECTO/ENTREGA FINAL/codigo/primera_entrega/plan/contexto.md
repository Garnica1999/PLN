# Contexto (handoffs entre agentes) — Entrega 1 PLN

## Antecedente: revisión inicial (orquestador, 2026-09-05)
- Enunciado E1 = idea de proyecto + corpus + textometría básica (PDF ≤8 págs + anexo corpus).
- Entregables actuales: `primera_entrega.ipynb` (42 celdas, ejecutado en Colab) y `Entrega_1_proyecto.zip` (main.tex, tg_usta.cls, references.bib, 3 PNG).
- Brechas del código detectadas: stopwords NLTK con tildes (84/313) no se eliminan tras normalizar sin diacríticos; parseo de SENTENCIA/ANNIO con `str[:8]` frágil (validado: en la práctica el corpus usa números de 3 dígitos y el método viejo pasó el muestreo, pero se mejora a regex documentado); total de palabras hardcodeado (271.907.978); spaCy cargado y sin uso (abstract sobrevende lematización/POS); top-20 y nubes calculados sobre muestra de 10 sentencias (1 por año) en vez del corpus completo.
- Brechas del informe: texto presidencial sobrante (main.tex línea ~145); fuente HF/URL/licencia/formato ausentes; falta párrafo de limpieza; etiqueta duplicada `fig:palabras_con_stopwords`; ecuación huérfana; references.bib con refs de otro proyecto; no hay PDF en la entrega; sin anexo corpus.
- Decisión Carlos: corpus completo local con `/home/carlos/venv`; entregable = notebook consolidado re-ejecutado; PDF compilado aquí con TinyTeX (~/.TinyTeX); investigación de artículos con citas en informe + nota técnica.

## Fase 0 (orquestador, 2026-09-05) — COMPLETADA
- Perfil máquina en plan/00_maquina.md y config/hardware.json (32 hilos, 31 GB RAM, sin GPU para esta tarea).
- Dataset descargado completo: `data/Juris1992_2022.csv` (1.67 GB; columnas `['','Texto']`; licencia CC-BY-4.0; 23.750 filas).
- Escaneo completo (2m41s): parseo regex OK 23.742 (99.97%) → T 17.099 / C 6.246 / SU 397; años reales 1992–2021 (2021 = 20 docs; sin 2022); 8 filas no-sentencia: 6 autos (A-xxx) + 2 basura HTML (rows 16200/16268).
- Siguiente: Fase 1 investigación (nlp-specialist).

## Fase 1 (nlp-specialist, 2026-09-05) — COMPLETADA
- Entregables: `estado_arte.md` (fichas 6 núcleo + 3 complementarias, verificadas DOI/URL el mismo día; nota de que ningún paper indexado usa el dataset HF exacto, se cubre con corpus relatoría CCC y NLP legal español: Anzola Cortés 2021, Barbosa Becerra 2021, Soh et al. 2019, HaCohen-Kerner 2020, González-González 2023, Raj et al. 2026, Aletras 2016, Chalkidis LEGAL-BERT 2020, Löffler 2025) y `references.bib` limpio (10 entradas, claves autorAño).
- Recomendaciones metodológicas E1: (1) normalizar stopwords NLTK quitando tildes (84 formas afectadas) + añadir stopwords jurídico-procedimentales; (2) top-N global en barras sobre corpus completo + frecuencias normalizadas, separando por tipo T/C/SU; (3) lematización diferible a E2, análisis temporal por año/lustro respaldado (Raj 2026).
- Siguiente: Fase 2 notebook consolidado (ml-coder).

## Fases 2-4 (ml-coder + ml-reviewer + ml-qa-engineer, 2026-09-05) — COMPLETADAS
- F2: notebook consolidado primera_entrega.ipynb (23 celdas, 13 code ejecutadas, 5m40s, 0 errores). Correcciones: regex parseo documentado (23.742/23.750), stopwords NLTK normalizadas 306 formas (84 con tilde), filtro tokens 1-letra + 'art' en origen, spaCy eliminado, total raw REAL 260.988.316 (vs 271.907.978 extrapolado), métricas en resultados/metricas.json, 5 figuras en figuras/, anexo 25 docs (semilla 42), notas md sobre nombre archivo y discrepancia raw añadidas (md1/md21), huella venv → "venv local".
- F3: ml-reviewer APROBADO CON CAMBIOS MENORES → aplicados; ml-qa-engineer APROBADO CON OBSERVACIONES (15 asserts OK; observaciones no bloqueantes resueltas en notebook/json/informe).
- F4: informe LaTeX reescrito y compilado con TinyTeX: Entrega_1_Informe.pdf 8 páginas (10 referencias apalike resueltas; fix \cite harvard; hyphen-spanish instalado; nubes por año segmentadas en 3 paneles por legibilidad). Zip Entrega_1_proyecto.zip regenerado (13 archivos planos, 22,5 MB) con PDF + fuentes + figuras + anexo corpus + ENLACE_CORPUS.txt. main.tex ajuste final: columnas CSV ('', 'Texto').
- Estado: todas las fases 0-5 completadas. Pendiente solo decisión de Carlos sobre commit/estructura final del repo.

## Verificación de independencia (orquestador, 2026-09-05)
- Notebook autocontenido: bootstrap de recursos en celda de configuración (stopwords NLTK, wordcloud, descarga CSV HF si falta, carpetas auto-creadas), 0 rutas absolutas, 0 dependencias ocultas.
- Prueba limpia OK (327 s, /tmp/opencode/test_e1, resultados idénticos) + re-ejecución en el proyecto OK (24 celdas, 13 code, 0 errores).

## Fase 6 (2026-09-06): Informe APA 7 + diapositivas
- Entorno: tlmgr instaló apa7, apacite, babel-spanish (+deps scalerel, fancyhdr, threeparttable, substr, enumitem, endfloat, grfext); python-pptx 1.0.2 en el venv.
- Tarea A: informe_apa/Entrega_1_Informe_APA.tex (~640 líneas; apa7 stu 12pt spanish, floatsintext, natbib→apacite; precita silenciosa para et al. APA7 con 3+ autores; \BOthers→"et al."). PDF compilado: 19 páginas, 0 citas indefinidas, portada APA en español, 7 figuras con "Nota. Elaboración propia", Tabla 1, referencias apacite.
- Tarea B: scripts/generar_diapositivas_entrega1.py → diapositivas/Entrega1_Diapositivas.pptx (14 slides 16:9, 7 imágenes, notas de orador, banda azul #1F3864; nubes anuales en fila 3 paneles). Validado reabriendo pptx y render.
- Tarea C: raíz con Entrega_1_Informe_APA.pdf + Entrega1_Diapositivas.pptx; universitario renombrado a Entrega_1_Informe_universidad.pdf; Entrega_1_proyecto.zip regenerado (17 archivos planos, 48,5 MB).
- Tarea D: commit 7d23177 (rama proyecto-primera-entrega-apa) + push; **PR #10 abierto** hacia main (sin auto-merge): https://github.com/Garnica1999/PLN/pull/10. Pendiente de aprobación de Carlos. Se añadió "*.out" a .gitignore.

## Fase 7 (2026-09-06): validación cruzada + fusión deck + vuelta al informe universitario
- T0: scripts/validar_datos.py → plan/validacion_datos.md. 22 cifras validadas notebook↔decks↔informe: sin discrepancias. Fix real: informe "75 formas" dominio jurídico → 65 (como notebook); PDF recompilado. Ausencias no-discrepantes documentadas (prom_clean y top-1 exactos no están como texto en deck compañero; van en las figuras).
- T1: scripts/fusionar_deck.py → PLN_PROYECTO.pptx 22 slides en estilo del compañero (banda+iconos+fuentes Arial+tarjetas EAF2F8/2F75B5). Backup original en plan/backup_original/. Entrega1_Diapositivas.pptx intacta (copia).
- T2: Entrega_1_Informe.pdf (universitario, 8 págs, fix 65 formas) = documento principal; APA fuera de raíz/zip (informe_apa/ local).
- T3: Entrega_1_proyecto.zip regenerado (15 archivos, 37,2 MB). T4: git sin acciones; PR #10 abierto.
