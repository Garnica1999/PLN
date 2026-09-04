# ARQUITECTURA — Taller-4 IR sobre corpus CEV

Fecha: 2026-09-04 | Autor: ml-architect | Alcance: puntos 1-3 del taller (extracción/preparación, EDA, modelo TF-IDF) + informe parcial. ADRs asociados: `plan/decisiones/ADR-001..005`. Estado del arte: `plan/estado_del_arte.md`.

## (a) Morfología del documento a recuperar

**Decisión (resumen).** Dos tipos de documento:
1. **Colección (libros)**: cada *segmento de libro* es una sección del tomo con granularidad TOC niveles 1-2 según el tomo, con fusión de secciones cortas y techo de extensión. Documento = 300–4.000 palabras aprox. (rango objetivo 400–2.500).
2. **Queries (entrevistas)**: transcripción completa por `id_doc` (una por registro del JSON) con metadatos `{id_doc, pages}`.

**Reglas de granularidad (libros):**
- Fuente de verdad = tabla "Contenido" de cada tomo (verificada: TESTIMONIAL págs. PDF 11-16; secciones con número de página impreso al final de línea).
- Granularidad por defecto: **nivel 2 del TOC cuando exista** (secciones dentro de la parte/capítulo); si el TOC es plano (solo nivel 1) o el nivel 1 no tiene subsecciones, el documento es el nivel 1.
- **Fusión**: secciones de nivel 2 con < 300 palabras se fusionan con la sección siguiente (o la anterior si es la última del padre).
- **Techo**: una sección > 4.000 palabras sin subsecciones en TOC se divide en fragmentos consecutivos de ≤ 3.500 palabras cortando en límite de párrafo/página.
- **Descartes**: portada, catálogo editorial, comisionados, TOC, "¿Cómo citar?", bibliografía/referencias, glosario, anexos masivos, créditos/colofón, listas de siglas/tablas/figuras, contraportada. Detección por patrón del título en TOC + posición (front matter antes del primer título mapeado; back matter tras el último contenido narrativo).
- **Esquema de id de segmento**: `TOMO_<abrev>-p<inicio>-<titulo_normalizado>`; p.ej. `TESTIMONIAL-p180-el-libro-de-las-devastaciones-y-la-vida` (abreviatura por tomo: HALLAZGOS, CONVOCATORIA, NARRATIVA, SUFRIR, TESTIMONIAL, CUERPO, NNA, VIOLACIONES, ETNICO, FUERA). Metadatos: `{id_segmento, tomo, titulo, nivel_toc, pagina_inicio, pagina_fin, n_palabras, texto}`.
- **Justificación**: nivel 1 puro generaría documentos de 20-80 págs. (mezcla de temas → similitud diluida); nivel 2 da unidades temáticas coherentes para el matching con entrevistas; la fusión evita fragmentos sin vocabulario discriminativo. El rango 300-4.000 palabras equilibra especificidad y robustez del vector TF-IDF. Decisión formal: ADR-001.

## (b) Pipeline de extracción de libros

1. **Páginas**: `pdftotext -layout` página a página (paralelo 32 hilos; poppler validado) → 1 archivo txt por página en `data/corpus/paginas/<TOMO>/pNNN.txt` (JSONL con {tomo, pagina_pdf, texto}).
2. **Parser de TOC por tomo** (heurístico, config por tomo en `config/`): localizar páginas "Contenido" (primeras páginas, patrón de título en línea propia + dígito de página al final de línea, con niveles por sangría `-layout`); extraer entradas `{titulo, pagina_impresa, nivel, tomo}`. Verificado: TESTIMONIAL, Hallazgos (encabezados "Hallazgos" p29, "La Colombia herida" p31, "Acercarnos a la experiencia..." p35).
3. **Offset páginas impresas ↔ PDF**: detectar número impreso al pie/cabecera en páginas iniciales; si coincide índice, offset=0 (verificado en TESTIMONIAL); si no, calibrar con los primeros títulos mapeados.
4. **Matching TOC ↔ cuerpo**: para cada entrada TOC, buscar su título (normalizado) como línea de encabezado en el cuerpo en la página indicada (ventana p±1); las coincidencias confirmadas fijan los límites de cada segmento (inicio_i a inicio_{i+1} en orden de página, respetando jerarquía). Heurística de exclusión de falsos positivos: pie de página repetido tipo "la colombia herida 31" (línea corta que contiene número de página impreso y se repite en ≥3 páginas seguidas) y encabezados repetidos (running heads) → blacklist automática.
5. **Sanity checks obligatorios** (salida a `resultados/`): (i) % de entradas TOC mapeadas a encabezado real (umbral ≥ 85%); (ii) cobertura de páginas del tomo por algún segmento (≥ 90% de las páginas de contenido; front/back matter excluido se reporta); (iii) solapamiento/duplicación de páginas entre segmentos = 0; (iv) distribución de palabras por segmento dentro de la banda objetivo (reportar % fuera de banda < 15%); (v) segmentos sin texto útil (OCR vacío) = 0.
6. **Salida**: `data/corpus/segmentos_libros.jsonl` (+ CSV plano con texto) y `resultados/extraccion_libros_report.json`.

## (c) Normalización de entrevistas (regex sobre el texto crudo; sin tocar `data/raw/`)

Por registro `{id_doc, pages, text}` → texto normalizado (1ª pasada) en `data/corpus/entrevistas_normalizadas.jsonl`:
1. Primera línea tipo `001-VI-00019_(48201)_`: se elimina (encabezado de sistema) si está al inicio del texto; se conserva el id numérico si aparece embebido como cita interna (decisión: no es contenido testimonial).
2. Anotaciones `[INTERRUP]`, `[INC: ...]`, `[DUD: ...]`, `[PAUSA...]`, `[risas]`, `[llanto]` y similares `[...]`: se eliminan (no son habla); en `[INC:]`/`[DUD:]` se elimina la marca y se conserva el contenido entre paréntesis solo si es palabra léxica (heurística: la transcripción es la fuente; decisión de EDA si el contenido aporta). Default experimento 1: eliminar marca y contenido de `[INC...]`/`[DUD...]`; registrar conteos.
3. **Anonimizados** → categoría canónica (ver ADR-002): `ORGANIZACI[ÓO]N P[ÚU]BLICA 1/2` → `organizacion_publica`; `ORGANIZACION SOCIAL 1/2` → `organizacion_social`; `LUGAR1..3` → `lugar`; `FAMILIAR1..4` → `familiar`; `CONOCIDO1` → `conocido`; `INSTITUCION EDUCATIVA1` → `institucion_educativa`; `PARTIDO POLITICO1` → `partido_politico`; otros patrones `PALABRA+[0-9]` en mayúsculas → token anonimizado genérico `anonimo`. Justificación: conservar el papel sintáctico/semántico del referente sin re-identificar (alineado con Bender & Friedman 2018; los roles familiares/lugares son vocabulario útil para el IR con libros, que hablan de familiares/lugares de forma genérica).
4. Etiquetas de hablante en inicio de línea (`TEST:`, `ENT:`, `ENT1:`, `PRUEBA:`, `X:`, `E:`, `P:`, patrones `[A-ZÑ]{1,5}:`) → se convierten en marca única `HABLANTE:` o se eliminan según experimento; default: eliminar la etiqueta y unir el turno al flujo del texto (los turnos son del mismo testimonio). Si el análisis muestra que el cambio de hablante (entrevistador vs víctima) importa para el matching, se reintroduce como feature (Open Question).
5. Guiones `-----` (≥3 guiones): eliminados (separadores de sistema).
6. Control: normalización **sin lowercase ni lematización** en esta fase (solo estructura); el lingüístico va en (d).

## (d) Preprocesado lingüístico (spaCy `es_core_news_md`, CPU)

Aplicado por separado a (1) segmentos de libros y (2) entrevistas normalizadas → `data/corpus/preprocesado/<corpus>_lemas.txt` (1 documento por línea, lemas separados por espacio) + versión tokenizada cruda para conteos EDA:
1. **Tokenización** con `nlp.pipe(texts, batch_size=…, n_process=8-16)` (ver ADR-003).
2. **Stopwords**: `spacy.lang.es.STOP_WORDS` **+ lista extendida de muletillas del habla colombiana** verificada en corpus (eh, mmm, este, o sea, pues, digamos, bueno, entonces, listo, cierto, ajá, ah, uy, mirá, ve, jum, umm, ahh, mmm, etc. — la lista final se congela tras EDA de frecuencias con una muestra, y se guarda en `config/stopwords_habla.txt`).
3. **Lematización** (es_core_news_md): forma canónica; minúsculas.
4. **Filtros**: tokens no alfabéticos eliminados (puntuación, símbolos, números sueltos), **excepto números de 4 dígitos en rango 1900-2025** (años: relevantes para el conflicto; se conservan como token `1991` etc.) y categorías canónicas de (c) que se preservan como un solo token (p.ej. `organizacion_publica`).
5. Palabras de 1 carácter eliminadas (no quedan tras stopwords/lemas; control de sanidad).
6. **Salidas EDA aparte**: conteo de tokens por documento, tipos, TTR por muestra de tamaño fijo y Yule's K [Bestgen 2023], top-N frecuencias por corpus.
7. Presupuesto: ~35-40M de tokens en entrevistas → estimación 1-2 h con n_process=8-16 (CPU-bound, 32 hilos); libros ~10.000 págs ≈ 3-4M tokens → < 15 min. Trabajo en streaming por chunks con checkpoint por doc (reanudable).

## (e) EDA (entregables en `imagenes/` + CSVs en `resultados/`)

1. Estadísticas descriptivas por corpus: nº docs, tokens, tipos, palabras/doc, páginas/entrevista (histogramas; se espera distribución muy asimétrica por el rango 3-850K chars).
2. Diversidad léxica: TTR (con muestreo de tamaño fijo n=1.000 tokens por doc) + Yule's K; comparación libros vs entrevistas (violin/boxplot).
3. Frecuencias top-30 por corpus (uni-, bi-, trigramas) y tabla de muletillas candidatas a stopwords.
4. Nubes de palabras uni/bi/trigramas por corpus (wordcloud; 3 figuras mínimo por corpus, con máscara simple).
5. Near-duplicates entre entrevistas: similitud Jaccard de shingles (Broder 1997) o coseno sobre el 1% más largo; reportar pares > 0.85 como "casi idénticas" **sin eliminarlas** (el corpus oficial se conserva íntegro; se documenta).
6. Comparación léxica libros vs entrevistas: solapamiento de vocabulario (tipos compartidos / unión) y términos distintivos por log-odds (o chi²) top-25 en cada dirección → figuras + CSV.
7. Resultado EDA congela: lista de muletillas final, umbrales min_df/max_df definitivos para el TfidfVectorizer y cualquier ajuste a (c)/(d).

## (f) Modelo IR (TF-IDF + coseno)

- **Dirección del matching (ADR-005)**: cada **entrevista es una query** y la **colección son los segmentos de libro** (documentos a recuperar). Justificación: el entregable del taller es "determinar qué entrevistas se relacionan con testimonios/secciones de los libros": para cada entrevista se recuperan los segmentos del Informe más afines (top-k). Alternativa transpuesta (segmento como query) se descarta como modelo principal porque el Informe no "busca" entrevistas individuales; se usa solo si EDA lo sugiere para algún análisis puntual (Open Question).
- **Representación**: `TfidfVectorizer` con:
  - `analyzer='word'`, `token_pattern=r"(?u)\b\w+\b"`, `lowercase=True` sobre los **textos ya lematizados** de (d) (los lemas son el vocabulario; no se re-tokeniza con spaCy dentro del vectorizador → reproducibilidad en Colab y velocidad).
  - `ngram_range=(1,2)`: decisión de ADR-005 (bigramas capturan colocaciones del dominio: "desaparición forzada", "violencia sexual", "falsa positivos"; costo controlado por min_df).
  - `sublinear_tf=True` (Salton & Buckley 1988: saturación de tf), `norm='l2'`, `use_idf=True`, `smooth_idf=True`.
  - `min_df` y `max_df` iniciales en (g); se recalibran con los resultados de EDA (7).
- **Cálculo**: fit sobre colección (segmentos); queries transformadas con el mismo vectorizador; similitud coseno (producto matricial sparse, batches) → `resultados/ranking_top5.csv` con columnas {id_doc entrevista, rango 1-5, id_segmento, score, tomo}.
- **Evaluación cualitativa estructurada** (no hay gold): muestra aleatoria estratificada por tomo de 20-30 pares (entrevista, top-1 de cada query de la muestra); etiquetado humano 3 clases {relacionado, parcial, no} con guía breve; reporte de Precision@1/@5 y distribución de etiquetas en `resultados/evaluacion_muestra.csv` + sección del informe. Criterio de éxito del taller (didáctico): P@5 aproximado documentado con ejemplos ilustrativos positivos y negativos; NO se exige umbral numérico externo.
- Alternativas documentadas (fuera de alcance): BM25 [Robertson & Zaragoza 2009], densos/multilingües [MessIRve] → sección "trabajo futuro" del informe.

## (g) Hiperparámetros iniciales — Experimento 1 (`experimentos/experimento_1_tfidf_base.py`)

| Parámetro | Valor | Fuente |
|---|---|---|
| ngram_range | (1,2) | ADR-005 (colocaciones de dominio) |
| sublinear_tf | True | Salton & Buckley (1988) |
| norm | 'l2' | Manning et al. (2008) |
| min_df | 2 | sklearn docs; corpus segmentos (~n=10³) |
| max_df | 0.85 | sklearn docs; descarta términos de todo el tomo |
| use_idf / smooth_idf | True / True | sklearn defaults |
| top-k | 5 | enunciado del taller |
| Lemas + stopwords + muletillas | config/stopwords_habla.txt + spaCy | ADR-003 |
| Tokenización | texto pre-lematizado, 1 doc/línea | ADR-003/005 |

Experimento 2 (sensible): variar min_df/max_df o ngram_range=(1,1) y comparar por la muestra etiquetada; experimento 3 opcional: sin muletillas vs con muletillas. Ver `plan/PLAN.md` Fase 6.

## Riesgos clave
1. **Parsing de TOC heterogéneo entre tomos** (10 formatos distintos) → parser por tomo con config + sanity checks con umbrales (b.5) y revisión manual de fallos.
2. **Pies/encabezados repetidos** confundidos con títulos → blacklist automática de running heads (b.4).
3. **Queries muy largas (media 80K chars)** → TF-IDF promedia; mitigación: normalización + lemas; reportar distribución de scores; opcional chunking de entrevista (Open Question).
4. **Vocabulario oral vs académico** → puente parcial vía lemas + categorías anonimizadas; expectativa honesta de calidad (sin gold).
5. **Tiempo spaCy** sobre 200M chars → streaming + checkpoint; presupuesto 1-2 h CPU (documentado en `00_maquina.md`).
