# Errores encontrados y resueltos — Taller PLN "Análisis de discursos presidenciales en Colombia"

Fecha de revisión inicial: 2026-08-19 · Última actualización: 2026-08-20
Archivo revisado: `Copia_de_Taller_PLN.ipynb` (81 celdas originales)
Notebook final: `Taller_Presidentes_final.ipynb` (134 celdas, 10/10 puntos, validado)
Referencia del enunciado: `Taller_Presidentes_2610.pdf` (10 puntos)
Método: revisión técnica + verificación empírica con el corpus real (`open_corpus_co_es`, NLTK 3.9.4, spaCy 3.8.13, `es_core_news_md`)

---

## Estado general

Los **9 errores iniciales fueron corregidos en su totalidad** y, además, se implementaron **5 procesamientos de limpieza del corpus** (de-hyphenation + P1-P4) que no estaban contemplados en el enunciado pero mejoran la calidad del análisis.

---

## Errores corregidos

| # | Severidad | Descripción | Estado | Cómo se resolvió |
|---|---|---|---|---|
| E1 | Alta | Quitar tildes en `limpiar_texto` degradaba spaCy (lemas corruptos: `podremos→podrar`, `abatiéndola→abatiendolar`; NER fusionaba personas) | ✅ Corregido | Vía paralela `discurso_spacy` (minúsculas, CON tildes y puntuación) para todo el pipeline spaCy |
| E2 | Media | Error stale guardado en la celda de stopwords (`AttributeError: 'list' object has no attribute 'words'`) | ✅ Corregido | Output de error eliminado; celda convertida en markdown; "Restart and run all" |
| E3 | Media | Matcher de spaCy con 2+ patrones no implementado (punto 5) | ✅ Corregido | Sección 5.7: Matcher con 3 patrones (ADJ+NOUN, NOUN+ADP+NOUN, NOUN+ADJ), 9.818 coincidencias, comparación vs n-gramas |
| E4 | Media | Falta conteo de discursos por presidente (punto 1) | ✅ Corregido | Celda con `value_counts()` (Uribe y Santos = 2, resto 1, total 22) |
| E5 | Media | `spacy download` fallaba en local (pip fuera del PATH) | ✅ Corregido | Celda de instalación robusta (chequeo + subprocess con `sys.executable`) |
| E6 | Baja | Columna `stopwords_spacy` duplicada en dict | ✅ Corregido | Duplicado eliminado |
| E7 | Baja | Sección "5.3" duplicada | ✅ Corregido | Renumeración única 5.1–5.7 |
| E8 | Baja | Markdown decía "cuatro elementos" (son 5 claves) | ✅ Corregido | Texto corregido |
| E9 | Baja | Stopword manual `"él"` con tilde que nunca coincidía | ✅ Corregido | Stopwords manuales con ortografía correcta (`él`, `país`) |

---

## Procesamientos adicionales del corpus (implementados y validados)

Ver detalle completo en `procesamiento_corpus.md`.

| Procesamiento | Problema | Resultado |
|---|---|---|
| De-hyphenation | Palabras partidas por guion de fin de línea (552 uniones, OCR de scans de López Michelsen/Turbay) | 525 uniones aplicadas con validación léxica; 27 rechazadas (pseudopalabras evitadas) |
| P1 — Líneas plantilla | Metadata de scan insertada (números de página, "agosto 1974/78", "señal memoria / rtvc", "---", "*****") | 108 líneas eliminadas en 5 discursos |
| P2 — OCR dígito→letra | OCR confundió dígitos con letras (`pa1·a`, `c0lombia`, `unive1·sidad`) | 8 tokens corregidos con validación léxica |
| P3 — "n" suelta | OCR partió palabras: `n ación`, `n u estro` | 10 uniones corregidas (`nación`, `nuestro`, `nunca`, `norte`) |
| P4 — Funcionales duplicadas | OCR duplicó palabras: `en en`, `de de` | 11 duplicados eliminados (whitelist; `no no`/`si si` conservados por posible retórica) |

---

## Notas metodológicas vigentes

- **textstat**: el pin `textstat==0.2` del enunciado es Python 2 e incompatible con Colab actual; se usa la última versión (0.7.13, API idéntica). Fórmulas diseñadas para inglés → valores orientativos en español, comparables entre discursos.
- **MTLD** (punto 6): sobre tokens sin stopwords degenera (el TTR léxico casi nunca cruza 0,72); documentado honestamente en el notebook.
- **Puntos 6-10**: el punto 8 se implementó completo; 6, 7, 9 y 10 también quedaron implementados por el equipo. Los 10 puntos del taller están completos.
- **Artefactos**: todas las figuras se guardan en `images/` (PNG dpi=150) y los datos tabulares en CSVs (utf-8-sig) dentro de la misma carpeta; las figuras también se muestran en el notebook.
