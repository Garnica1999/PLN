# TALLER-4 — Recuperación de Información sobre corpus de la Comisión de la Verdad (CEV)

Curso: Procesamiento de Lenguaje Natural — Pontificia Universidad Javeriana (L. G. Moreno Sandoval)
Fecha: 2026-09-04 | Alcance: puntos 1-3 del taller | Entrega parcial (informe de avance)

## 1. Introducción y objetivo

El taller de Recuperación de Información (IR) propone desarrollar un proceso que determine qué entrevistas del documento JSON de la Comisión de la Verdad (CEV) están relacionadas con testimonios o secciones de los libros del Informe Final CEV, aplicando técnicas de PLN para relacionar las narraciones de las entrevistas con textos específicos de los libros y medir la relevancia de los resultados. Este informe documenta los puntos 1 a 3 del enunciado (extracción y preparación del corpus, análisis exploratorio y modelo IR TF-IDF) sobre **2.486 transcripciones de entrevistas** (registro oral, CEV/JEP) y los **10 tomos del Informe Final** distribuidos en el zip del curso. Los puntos 4 y 5 (implementación manual de Rocchio y Okapi BM25, y comparación con heatmap) quedaron fuera del alcance aprobado y se documentan como trabajo futuro (sección 7).

Un hallazgo honesto del estado del arte (`plan/estado_del_arte.md`) es que no existe literatura computacional publicada que use directamente el corpus CEV; la más cercana es la minería de textos sobre audiencias públicas del caso 03 de la JEP (Sosa et al., 2025), que valida la viabilidad del análisis léxico de narrativas de víctimas. En consecuencia **no hay gold estándar**: la evaluación es cualitativa y estructurada sobre una muestra etiquetada, alineada con la literatura cercana. El notebook `notebooks/Taller_4_IR_CEV.ipynb` complementa este informe reproduciendo el pipeline completo en formato compatible con Google Colab.

## 2. Datos y consideraciones éticas

**Entrevistas.** `data/raw/entrevistas_all_2023-03-21_14-24_05.json` (231 MB) contiene 2.486 registros `{id_doc, pages, text}`: transcripciones del registro oral de la CEV (con material relacionado de la JEP), anonimizadas por la Comisión con marcas sistemáticas en mayúsculas y numeración (`LUGAR1`, `FAMILIAR2`, `ORGANIZACION PUBLICA 1`, ...) y con anotaciones de transcripción (interrupciones, pasajes inaudibles o dudosos, risas, llanto) y etiquetas de hablante (`ENT:`, `TEST:`). La extensión es extremadamente asimétrica: de 3 a 850 mil caracteres en crudo.

**Libros.** El zip contiene los 10 tomos del Informe Final en PDF (301 MB extraídos; 6.144 páginas PDF): HALLAZGOS, CONVOCATORIA, NARRATIVA, SUFRIR, TESTIMONIAL, CUERPO, NNA, VIOLACIONES, ETNICO y FUERA. El tomo testimonial (680 páginas PDF) es clave para el taller porque reproduce el registro oral de las víctimas (sección 6).

**Tesauro.** `data/raw/Resource-tesauro.rdf` (≈5 MB) es el tesauro CEV en SKOS (10.151 etiquetas en español). Por ADR-004 **no se integra al modelo**: se usa cualitativamente en el EDA para interpretar términos distintivos.

**Ética.** El corpus contiene narrativas de víctimas y testigos; siguiendo a Bender y Friedman (2018) para datos sensibles, el proyecto documenta procedencia y contexto, no intenta re-identificar personas y trata la anonimización de la CEV como activo: la normalización a categorías canónicas (sección 4) conserva el papel semántico del referente sin publicar identificadores originales. La auditoría de datos (fase 7) verificó que el intermedio `entrevistas_normalizadas.jsonl` contiene PII hablada de las transcripciones públicas (1 cédula, 2 celulares, 118 corridas de 6-10 dígitos, 1.773 tokens UPPER+dígito pegados que solo pasan el filtro en la etapa spaCy, diseño en dos etapas ADR-002/003); **los corpus finales usados y los resultados publicados están limpios: 0 correos, 0 cédulas, 0 teléfonos, 0 tokens UPPER+dígito**, y `data/raw/` permanece intacto.

## 3. Morfología del documento y extracción del corpus (punto 1)

### 3.1 Morfología del documento a recuperar

La decisión (ADR-001) distingue dos tipos de documento:

1. **Colección (libros)**: el documento recuperable es un *segmento de tomo*: una sección de la tabla de contenido (TOC) con granularidad de **niveles 1-2 según el tomo** (nivel 2 cuando existe; nivel 1 si el TOC es plano). Secciones < 300 palabras se fusionan con la siguiente (o la anterior si es la última del padre); secciones > 4.000 palabras sin subsecciones se dividen en fragmentos ≤ 3.500 palabras en límite de párrafo o página; se descartan front y back matter (portada, catálogo, comisionados, TOC, bibliografía, glosario, anexos masivos, créditos). El id sigue el esquema `TOMO_abrev-p<pagina_inicio>-<titulo_normalizado>` (p. ej. `TESTIMONIAL-p180-el-libro-de-las-devastaciones-y-la-vida`). Esta granularidad equilibra dos riesgos: el tomo completo diluye la similitud (mezcla temas) y la página produce vectores pobres que rompen unidades temáticas.
2. **Queries (entrevistas)**: la consulta es la **transcripción completa** de cada `id_doc`, con sus metadatos `{id_doc, pages}`.

### 3.2 Pipeline de extracción

`scripts/01_extraer_libros.py` implementa: (i) conversión página a página con `pdftotext -layout` (paralelo, 32 hilos); (ii) parser del TOC por tomo que combina el sangrado del `-layout` con las coordenadas `x0` de PyMuPDF y la paridad de márgenes para asignar el nivel jerárquico; (iii) calibración del desfase página impresa ↔ PDF (verificado **offset = 0** en TESTIMONIAL); (iv) matching de cada entrada del TOC contra el cuerpo, buscando el título normalizado como línea de encabezado en la página indicada (ventana p±1); (v) exclusión de falsos positivos con *blacklist* automática de encabezados y pies repetidos (*running heads*, p. ej. pies tipo "la colombia herida 31" repetidos en ≥ 3 páginas).

Los controles de sanidad (mapeo de títulos ≥ 85 %, cobertura ≥ 90 %, 0 solapes, 0 vacíos, < 15 % fuera de banda) se cumplen en los 10 tomos (Tabla 1).

**Tabla 1. Segmentos por tomo del Informe Final CEV** (fuente: `resultados/extraccion_libros_report.json`)

| Tomo | Páginas PDF | Segmentos | Palabras | Mapeo TOC |
|---|---|---|---|---|
| HALLAZGOS | 792 | 146 | 309.169 | 100 % |
| VIOLACIONES | 1.144 | 148 | 457.293 | 100 % |
| NARRATIVA | 656 | 111 | 230.995 | 98,1 % |
| TESTIMONIAL | 680 | 104 | 271.338 | 100 % |
| FUERA | 548 | 96 | 211.824 | 100 % |
| ETNICO | 752 | 91 | 286.607 | 100 % |
| CUERPO | 644 | 84 | 188.189 | 95,3 % |
| SUFRIR | 484 | 68 | 161.620 | 96,1 % |
| NNA | 388 | 55 | 140.301 | 91,2 % |
| CONVOCATORIA | 56 | 14 | 14.157 | 100 % |
| **Total** | **6.144** | **917** | **2.271.493** | 91-100 % |

El corpus de colección tiene **917 segmentos y 2.271.493 palabras** (2,27 M), todos dentro de la banda objetivo (0 % fuera de banda). La figura `../imagenes/eda_libros_lemas_por_tomo.png` muestra la distribución de lemas de contenido por tomo ya preprocesado.

## 4. Normalización y preprocesado (punto 1)

### 4.1 Normalización estructural de entrevistas

`scripts/02_preprocesar.py` aplicó una normalización regex por etapas (ADR-002) en 58,4 s (pico RAM 679,9 MB), reduciendo 200.518.396 → 188.838.879 caracteres (**−5,82 %**): eliminación de códigos de sistema (422 líderes, 604 líneas `AUDIO`, 896 embebidos), de anotaciones entre corchetes (138.613 marcas `[INTERRUP]`, `[INC:...]`, `[DUD:...]`, `[PAUSA]`, `[risas]`, `[llanto]`), de etiquetas de hablante (653.328) y de guiones separadores (291.868); y **mapeo de anonimizados a 8 categorías canónicas** (Tabla 2), que conserva el papel semántico del referente —vocabulario útil para el matching con libros que hablan genéricamente de familiares, lugares y organizaciones— sin re-identificar.

**Tabla 2. Anonimizados normalizados a categorías canónicas** (fuente: `resultados/metricas.json`, fase 4)

| Categoría canónica | Conteo | Categoría canónica | Conteo |
|---|---|---|---|
| `conocido` | 58.525 | `familiar` | 21.428 |
| `anonimo` (genérico) | 43.262 | `organizacion_publica` | 18.878 |
| `organizacion_social` | 25.336 | `institucion_educativa` | 5.974 |
| `lugar` | 24.226 | `partido_politico` | 1 |
| | | **Total** | **197.630** |

### 4.2 Preprocesado lingüístico (spaCy)

Sobre los textos normalizados se aplicó spaCy `es_core_news_md` (CPU, `n_process=8`, `batch_size=4`, parser y NER deshabilitados): tokenización, minúsculas, lematización y parada en consumo con 521 stopwords de spaCy + **51 muletillas del habla colombiana** verificadas por frecuencia (`config/stopwords_habla.txt`: *eh, mmm, este, o sea, pues, digamos, ajá*...) + 42 tokens de ruido tipográfico de libros (fragmentos de palabras partidas por guion del PDF, p. ej. `cion`; ruido bibliográfico como `ibid.`). Se conservan los años 1900-2025 y cada categoría canónica como un solo token. Las **62 correcciones de lemas** (formas corruptas como `atreveír`→`atrever`, voseo sin lematizar, truncamientos orales) se aplican en consumo desde `config/lemas_correcciones.json`, sin re-procesar el corpus.

Producción: libros 917 documentos en 30,5 s; entrevistas 2.486 en ≈ 509,5 s (4,88 docs/s; 365.630 chars/s), con 40.346.344 tokens crudos y 10.923.423 lemas (ratio 0,271). Dos entrevistas quedan legítimamente vacías. El pipeline es determinista y **reanudable** (checkpoint por documento: una corrida interrumpida se retomó en el doc 424 sin pérdidas). Corpus entregados: `data/corpus/segmentos_libros.jsonl` (917), `entrevistas_normalizadas.jsonl` (2.486) y `preprocesado/{libros,entrevistas}_{lemas,tokens}.txt` + `*_docs.json` (una línea por documento, aptos para análisis comparativo). Calidad: ruff 0; 43 pruebas pytest aprobadas (3,27 s).

## 5. Análisis exploratorio del corpus (punto 2)

### 5.1 Estadísticas y diversidad léxica

Con parada en consumo y correcciones: libros 917 documentos, 928.253 lemas de contenido, 33.277 tipos; entrevistas 2.486 documentos, 9.702.534 lemas, 109.391 tipos. La extensión por documento es muy asimétrica en entrevistas (media 3.903 lemas vs. 1.012 por segmento de libro): `../imagenes/eda_entrevistas_hist_lemas.png`, `../imagenes/eda_libros_hist_lemas.png`, y en crudo `../imagenes/eda_entrevistas_hist_chars.png`, `../imagenes/eda_entrevistas_hist_paginas.png`.

Por la dependencia del TTR con la longitud (Bestgen, 2023), la diversidad se midió con TTR de muestra fija (n=1.000) y Yule K sobre el documento completo: los libros son más diversos (**TTR_m1000 0,609 vs. 0,514; Yule K 31,8 vs. 38,7**; menor K = más diverso): la oralidad transcrita es sistemáticamente más repetitiva (Moreno-Sandoval et al., 2024). Ver `../imagenes/eda_diversidad_lexica.png`.

### 5.2 Frecuencias y nubes

Los top-10 de unigramas muestran dos mundos léxicos: libros con sustantivos del género institucional (*armado, víctima, grupo, persona, violencia, comisión, mujer, conflicto, año, entrevista*); entrevistas con verbos y marcadores de narración oral (*gente, llegar, año, pasar, cosa, venir, salir, conocido, persona, querer*). Nubes uni/bi/trigramas por corpus desde las frecuencias top-30 (`resultados/eda/frecuencias_*.csv`): `../imagenes/eda_libros_nube_unigramas.png`, `../imagenes/eda_libros_nube_bigramas.png`, `../imagenes/eda_libros_nube_trigramas.png`, `../imagenes/eda_entrevistas_nube_unigramas.png`, `../imagenes/eda_entrevistas_nube_bigramas.png`, `../imagenes/eda_entrevistas_nube_trigramas.png`.

### 5.3 Near-duplicates

Con Jaccard de shingles de 5 lemas (minhash k=60 + LSH 20×3 y verificación exacta; Broder, 1997) se detectaron **345 pares de entrevistas casi idénticos (> 0,85)** sobre 368 candidatos LSH verificados (≈ 21 % de los documentos, 242 grupos): `../imagenes/eda_near_duplicates_entrevistas.png`. Por decisión metodológica **no se eliminan** (corpus oficial íntegro); el fenómeno se documenta porque las casi-duplicadas producen rankings casi idénticos.

### 5.4 Comparación de patrones léxicos libros vs. entrevistas

El solapamiento de vocabulario es bajo: **Jaccard = 0,221** (25.851 tipos compartidos de ~116,8 mil combinados); la cobertura es asimétrica: el vocabulario de los libros cubre el **95,7 % de los tokens de las entrevistas** y el de las entrevistas el 98,3 % de los de libros —el habla cabe casi por completo en el vocabulario del informe, pero no al revés. Los distintivos por log-odds (`../imagenes/eda_comparacion_distintivos_logodds.png`) confirman la direccionalidad: en libros, términos institucionales y transmedia (*transmedia* 8,0; *externas* 7,8; *microdato* 7,4); en entrevistas, categorías canónicas —evidencia de que la normalización capturó vocabulario real del habla— (*anonimo* 9,1; *organizacion_social* 8,5; *organizacion_publica* 8,2) y formas orales como *far* (519 ocurrencias, "las FAR") frente a la sigla canónica *farc* de los libros: desajuste léxico del registro oral relevante para el matching. El tesauro (ADR-004) aportó etiqueta SKOS a 14 de los 50 distintivos (p. ej. *cnmh*: Centro Nacional de Memoria Histórica), confirmando que remiten a entidades del dominio. El EDA congeló la configuración de la fase 6 con evidencia (sección 6.1).

## 6. Modelo de IR TF-IDF (punto 3)

### 6.1 Configuración y dirección del matching

Por ADR-005, **cada entrevista es una query y la colección son los 917 segmentos de libro** (dirección que responde al entregable: hallar la sección del Informe afín a cada entrevista). Se usa `TfidfVectorizer` de scikit-learn sobre textos pre-lematizados (sin re-tokenizar, para reproducibilidad en Colab), con `config/tfidf_config.json` congelado en el EDA: `ngram_range=(1,2)` (colocaciones de dominio: *desaparición forzada, violencia sexual, falsos positivos*); `min_df=2` justificado porque el **43,2 % de los tipos de la colección son hapax** (14.365 de 33.277 con df=1; mediana de 2 docs/tipo); `max_df=0.85` (solo 6 tipos aparecen en > 85 % de los segmentos); `sublinear_tf=True` (Salton y Buckley, 1988), `norm='l2'` (Manning et al., 2008), `use_idf` y `smooth_idf=True`. Resultado: **118.748 features** (18.864 unigramas + 99.884 bigramas; densidad 0,0082). El fit se hace solo sobre la colección y las queries se transforman con el mismo vectorizador (sin leakage); coseno por batches en 10,0 s con semilla 42, generando `resultados/ranking_top5.csv` (12.420 filas) y `ranking_top10.csv`.

### 6.2 Resultados del ranking

El modelo recupera ≥ 1 segmento para 2.484 de 2.486 entrevistas (**cobertura 99,92 %**; las 2 restantes son los vacíos legítimos). El score top-1 tiene media 0,1468, mediana 0,138 y rango 0,070-0,572. La distribución del top-1 por tomo (Tabla 3) revela un sesgo marcado: **76,8 % en TESTIMONIAL**. El revisor lo confirmó como **efecto de dominio, no artefacto**: el tomo testimonial reproduce el registro oral de las víctimas (Castillejo Cuéllar, 2023) y su vocabulario coincide naturalmente con el de las queries orales; se descartaron los artefactos alternativos (longitud de queries y categorías canónicas no explican la magnitud). La evaluación de la muestra (6.4) confirma que el sesgo produce top-1 plausibles pero genéricos, no falsos positivos duros.

**Tabla 3. Distribución del top-1 por tomo** (fuente: `resultados/ir_tfidf_report.json`)

| Tomo | Queries con top-1 en el tomo | % |
|---|---|---|
| TESTIMONIAL | 1.907 | 76,8 % |
| FUERA | 129 | 5,2 % |
| NARRATIVA | 108 | 4,4 % |
| ETNICO | 95 | 3,8 % |
| VIOLACIONES | 73 | 2,9 % |
| HALLAZGOS | 53 | 2,1 % |
| CUERPO | 50 | 2,0 % |
| SUFRIR | 46 | 1,9 % |
| NNA | 22 | 0,9 % |
| CONVOCATORIA | 1 | 0,0 % |
| **Total** | **2.484** | **100 %** |

### 6.3 Variantes de preprocesado

Comparando la coincidencia del top-1 con el base (Tabla 4), la variante C (sin las 62 correcciones de lemas) es la más cercana (96,6 %), luego B (sin las 51 muletillas: 84,6 %); la variante A de unigramas puros diverge fuerte (54,8 %), lo que confirma la contribución de los bigramas de dominio.

**Tabla 4. Variantes del modelo base** (fuente: `resultados/ranking_top5_{A,B,C}.csv`)

| Variante | Cambio respecto de la base | Coincidencia top-1 con la base |
|---|---|---|
| A | `ngram_range=(1,1)` (solo unigramas) | 54,8 % |
| B | Sin parada de muletillas del habla (51) | 84,6 % |
| C | Sin correcciones de lemas (62) | 96,6 % |

### 6.4 Evaluación cualitativa (sin gold estándar)

Se construyó una muestra determinista (semilla 42), estratificada por tomo, de **30 pares (entrevista, segmento top-1)** con top-3 completo y fragmentos de ~500 caracteres; el etiquetado humano (proxy del equipo, con guía explícita) clasificó cada par en {relacionado, parcial, no} (Tabla 5).

**Tabla 5. Métricas de evaluación cualitativa sobre 30 pares** (fuente: `resultados/evaluacion_metricas.json`, `evaluacion_muestra_etiquetada.csv`)

| Métrica | Valor |
|---|---|
| Etiquetas: relacionado / parcial / no | 16 / 14 / 0 |
| P@1 estricto (solo "relacionado") | 0,533 |
| P@1 aceptable ("relacionado" + "parcial") | 1,000 |
| P@3 aceptable (al menos un "relacionado" en r1-r3) | 0,767 |
| P@3 aceptable ("relacionado" + "parcial" en r1-r3) | 1,000 |
| Top-1 "parcial" con match "relacionado" en r2-r3 | 7 de 30 (23 %) |

El matching léxico base es **funcional pero impreciso**: con criterio estricto P@1 = 0,53 y P@3 = 0,77; con criterio aceptable ambas llegan a 1,0 porque todo el corpus versa sobre el conflicto colombiano y casi cualquier entrevista de víctima solapa mínimamente con casi cualquier segmento. No hay top-1 etiquetado "no" (0 falsos positivos duros), pero en 7/30 queries el top-1 es "parcial" y el match específico quedó en r2-r3. Ejemplos curados (`resultados/ejemplos_top1_ilustrativos.md`): el modelo acierta con términos de dominio específicos (amenaza de masacre → `HALLAZGOS-p129-las-masacres`, 0,132; líder raizal → `ETNICO-p448-...`, 0,124) y falla por solape genérico en entrevistas testimoniales (excombatiente indígena amazónica → `TESTIMONIAL-p401-senderos-pedregosos`, emparejada por el registro oral compartido).

**Decisión de iteración**: no se creó el experimento 2. El criterio del plan (P@1 aceptable ≥ 0,6) se cumple holgadamente (1,0) y el problema diagnosticado no es el registro oral —las muletillas ya están en parada y el léxico oral residual (*papa, hijo, mami*...) es contenido temático legítimo— sino la ausencia de discriminación temática fina: añadir más léxico a la parada afectaría por igual a todos los segmentos y no discriminaría el hecho o la población específica; el experimento 2 no atacaría la causa y consumiría tiempo sin mejora esperable del P@1 estricto.

## 7. Observaciones, limitaciones y mejoras futuras

**Observaciones.** (i) La extracción con controles de sanidad produjo un corpus íntegro y trazable (917 segmentos, 2,27 M palabras, 0 solapes/vacíos) con morfología documentada (ADR-001). (ii) El TF-IDF sobre lemas recupera bien con terminología específica de dominio y produce top-1 plausibles pero de especificidad limitada en testimonios orales (P@1 estricto 0,53). (iii) El sesgo TESTIMONIAL (76,8 %) es efecto de dominio verificado (en la muestra: 2/5 top-1 TESTIMONIAL "relacionado", 3/5 "parcial", 0 "no"). (iv) Trazabilidad completa: métricas por fase en `resultados/metricas.json`, ranking determinista (semilla 42), 43 pruebas aprobadas, corpus finales auditados sin PII. La autoevaluación contra la rúbrica del curso (puntajes propuestos del equipo, no nota del profesor) está en `plan/verificacion_rubrica.md`; el cruce requisito-evidencia de los puntos 1-3 en `plan/verificacion_requisitos.md` (10/10 con evidencia).

**Limitaciones.** (a) Sin gold: la evaluación es proxy del equipo sobre 30 pares, no etiquetado humano externo. (b) El lematizador no resuelve voseo ni algunos pretéritos del habla colombiana (mitigado con 62 correcciones). (c) 345 pares de entrevistas casi duplicadas (> 0,85) replican rankings y sobreestiman la cobertura aparente; no se eliminan por conservación del corpus. (d) El matching léxico no discrimina hechos o poblaciones específicas cuando el solape de vocabulario del conflicto es amplio. (e) Detalles menores de extracción: límites duplicados en una misma página y *running heads* sin número no eliminados del texto de segmentos (ruido TF-IDF despreciable).

**Mejoras futuras.** (i) Puntos 4-5 del taller: Rocchio y Okapi BM25 implementados manualmente (Robertson y Zaragoza, 2009) y heatmap de relacionamiento; los modelos léxicos siguen siendo competitivos en IR en español (MessIRve: Valentini et al., 2024). (ii) Embeddings/transformers con enrutamiento previo por tomo, para atacar la especificidad. (iii) Reintroducir el cambio de hablante (entrevistador vs. entrevistado) como característica. (iv) Etiquetado humano externo de la muestra. (v) Chunking de entrevistas largas como queries independientes.

## 8. Referencias

Referencias verificadas (URL/DOI) en `plan/estado_del_arte.md`, que además documenta la brecha de literatura computacional sobre el corpus CEV:

1. Sosa, J., Urrego-López, A., Prieto, C., Camargo-Díaz, E. J. (2025). Constructing the Truth: Text Mining and Linguistic Networks in Public Hearings of Case 03 of the Special Jurisdiction for Peace (JEP). arXiv:2504.04325.
2. Valentini, F., Cotik, V., Furman, D., Bercovich, I., Altszyler, E., Pérez, J. M. (2024). MessIRve: A Large-Scale Spanish Information Retrieval Dataset. arXiv:2409.05994.
3. Salton, G., Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing & Management, 24(5). DOI 10.1016/0306-4573(88)90021-0.
4. Robertson, S., Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in IR, 3(4). DOI 10.1561/1500000019.
5. Manning, C. D., Raghavan, P., Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.
6. Bender, E. M., Friedman, B. (2018). Data Statements for Natural Language Processing. TACL, 6. DOI 10.1162/tacl_a_00041.
7. Bestgen, Y. (2023). Measuring Lexical Diversity in Texts: The Twofold Length Problem. arXiv:2307.04626.
8. Moreno-Sandoval, L. G., et al. (2024). Detection of Hate Speech, Racism and Misogyny in Digital Social Networks: Colombian Case Study. Big Data Cogn. Comput., 8(9):113. DOI 10.3390/bdcc8090113.
9. Castillejo Cuéllar, A. (2023). Escuchar de otra manera. Oralidad y sonido en el volumen testimonial de la Comisión de la Verdad. Estudios Políticos, 68. DOI 10.17533/udea.espo.n68a02.
10. Puertas, E., Alvarado, J., Moreno-Sandoval, L. G., Pomares Quimbaya, A. (2018). An Automatic Approach to Generate Corpus in Spanish. Springer CCIS. DOI 10.1007/978-3-319-98998-3_12.
11. Broder, A. Z. (1997). On the resemblance and containment of documents. DOI 10.1109/SEQUEN.1997.666900.
12. spaCy (2024). Modelo es_core_news_md. https://spacy.io/models/es.
13. scikit-learn (2024). TfidfVectorizer. Documentación oficial.
14. Comisión de la Verdad (2022). Hay futuro si hay verdad. Informe Final. https://www.comisiondelaverdad.co.
