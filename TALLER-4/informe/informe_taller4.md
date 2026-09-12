# TALLER-4 — Recuperación de Información sobre el corpus de la Comisión de la Verdad (CEV)

Curso: Procesamiento de Lenguaje Natural — Pontificia Universidad Javeriana (L. G. Moreno Sandoval)
Integrantes: Erika Valeria Rivera Jiménez — Julián Guillermo Carrillo Meneses — Carlos Andrés Garnica Salazar — Oscar Andrés Prieto Moreno
Fecha: 2026-09-12 | **Informe final integrador (puntos 1-6 del taller)**

## 1. Introducción y objetivos

El taller propone un proceso de Recuperación de Información (IR) que determine qué entrevistas del registro oral de la Comisión de la Verdad (CEV) se relacionan con secciones de los libros del Informe Final, midiendo la relevancia de los vínculos: **2.486 entrevistas** (queries, CEV/JEP) contra **917 segmentos** de los **10 tomos** (colección).

Los seis componentes están cubiertos: (P1) extracción y preparación del corpus; (P2) análisis exploratorio; (P3) modelo TF-IDF; (P4) implementación manual de Rocchio y Okapi BM25 con funciones propias documentadas; (P5) comparación de corpus con heatmap y cuadro de vínculos; y (P6) este informe final. El notebook `notebooks/Taller_4_IR_CEV.ipynb` (120 celdas) reproduce el pipeline completo y es el entregable ejecutable.

Por ADR-005, **cada entrevista es una query y la colección son los 917 segmentos**, dirección del taller. No hay literatura computacional sobre el corpus CEV; lo más cercano es la minería de audiencias de la JEP (Sosa et al., 2025) y el análisis del tomo testimonial (Castillejo Cuéllar, 2023). **No existe gold estándar**: la evaluación es cualitativa sobre una muestra etiquetada por el equipo (proxy) y acuerdo entre rankings.

**Resultados principales.** P1: 10/10 tomos, 917 segmentos, 2.271.493 palabras, mapeo TOC 91-100 %. P2: 2.486 entrevistas normalizadas (−5,82 %) y preprocesadas (40,35 M tokens → 10,92 M lemas); TTR_m1000 0,609 vs. 0,514; Jaccard 0,221; 345 near-duplicados (>0,85). P3: 118.748 features, cobertura 99,92 %, top-1 TESTIMONIAL 76,77 %, P@1 estricto 0,533 / aceptable 1,0 sobre 30 pares. P4: BM25 y Rocchio manuales con auto-checks en verde; acuerdo top-1 base–BM25 57,0 %, base–Rocchio 70,1 %, BM25–Rocchio 50,6 %. P5: heatmap 10×10 (diagonal media 0,0725 > 0,0608 fuera) y 30 vínculos.

## 2. Datos y consideraciones éticas

**Entrevistas.** `data/raw/entrevistas_all_2023-03-21_14-24_05.json` (231 MB): 2.486 registros `{id_doc, pages, text}` del registro oral CEV/JEP, anonimizados por la Comisión con marcas en mayúsculas y numeración (`LUGAR1`, `FAMILIAR2`, `ORGANIZACION PUBLICA 1`, ...), anotaciones de transcripción (interrupciones, pasajes dudosos, risas, llanto) y etiquetas de hablante (`ENT:`, `TEST:`). Extensión muy asimétrica: de 3 a 850 mil caracteres en crudo.

**Libros.** 10 tomos del Informe Final en PDF (301 MB extraídos; 6.144 páginas): HALLAZGOS, CONVOCATORIA, NARRATIVA, SUFRIR, TESTIMONIAL, CUERPO, NNA, VIOLACIONES, ETNICO y FUERA. El tomo testimonial (680 páginas) reproduce el registro oral de las víctimas, clave en el análisis.

**Tesauro.** `data/raw/Resource-tesauro.rdf` (≈5 MB; SKOS, 10.151 etiquetas); por ADR-004 **no se integra al modelo** y se usa cualitativamente en el EDA. El notebook recibe únicamente **3 insumos descargados manualmente** del Drive (entrevistas, libros y tesauro) y genera todo lo demás (sección 9).

**Ética.** El corpus contiene narrativas de víctimas; siguiendo a Bender y Friedman (2018), se documenta procedencia y contexto, no se re-identifica a personas y se trata la anonimización de la CEV como activo: la normalización a categorías canónicas conserva el papel semántico del referente sin publicar identificadores. La auditoría verificó PII hablada en el intermedio `entrevistas_normalizadas.jsonl` (1 cédula, 2 celulares, 118 corridas de 6-10 dígitos, 1.773 tokens UPPER+dígito; etapas ADR-002/003); **los corpus finales y los resultados publicados están limpios: 0 correos, 0 cédulas, 0 teléfonos, 0 tokens UPPER+dígito**, y `data/raw/` permanece intacto.

## 3. P1 — Extracción y preparación del corpus

### 3.1 Morfología del documento a recuperar (ADR-001)

La **colección (libros)** es un *segmento de tomo*: una sección del TOC con granularidad de **niveles 1-2 según el tomo** (nivel 2 si existe; nivel 1 si el TOC es plano). Las secciones < 300 palabras se fusionan con la siguiente; las > 4.000 palabras sin subsecciones se dividen en fragmentos ≤ 3.500 palabras en límite de párrafo o página; se descartan front y back matter. El id sigue el esquema `TOMO_abrev-p<pagina_inicio>-<titulo_normalizado>`. La granularidad evita diluir temas (tomo completo) o romper unidades temáticas (página). Las **queries (entrevistas)** son la transcripción completa de cada `id_doc` con sus metadatos `{id_doc, pages}`.

### 3.2 Pipeline de extracción de libros

`scripts/01_extraer_libros.py`: (i) `pdftotext -layout` página a página (32 hilos); (ii) parser del TOC que combina sangrado, coordenadas `x0` de PyMuPDF y paridad de márgenes; (iii) calibración del desfase página impresa ↔ PDF (verificado **offset = 0** en TESTIMONIAL); (iv) matching del título normalizado (ventana p±1); y (v) *blacklist* de encabezados y pies repetidos. Los controles de sanidad (mapeo ≥ 85 %, cobertura ≥ 90 %, 0 solapes, 0 vacíos, < 15 % fuera de banda) se cumplen en los 10 tomos.

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

La colección tiene **917 segmentos y 2.271.493 palabras** (0 % fuera de banda, 0 solapes, 0 vacíos). `../imagenes/eda_libros_lemas_por_tomo.png` muestra los lemas de contenido por tomo preprocesado.

### 3.3 Normalización y preprocesado de entrevistas

`scripts/02_preprocesar.py` aplicó una normalización regex por etapas (ADR-002) en 58,4 s (pico RAM 679,9 MB): 200.518.396 → 188.838.879 caracteres (**−5,82 %**), eliminando códigos de sistema (422 líderes, 604 líneas `AUDIO`, 896 embebidos), anotaciones entre corchetes (138.613), etiquetas de hablante (653.328) y guiones (291.868), y mapeando los anonimizados a **8 categorías canónicas** (Tabla 2) que conservan el papel semántico del referente sin re-identificar.

**Tabla 2. Anonimizados normalizados a categorías canónicas** (fuente: `resultados/metricas.json`, fase 4)

| Categoría canónica | Conteo | Categoría canónica | Conteo |
|---|---|---|---|
| `conocido` | 58.525 | `familiar` | 21.428 |
| `anonimo` (genérico) | 43.262 | `organizacion_publica` | 18.878 |
| `organizacion_social` | 25.336 | `institucion_educativa` | 5.974 |
| `lugar` | 24.226 | `partido_politico` | 1 |
| | | **Total** | **197.630** |

Después se aplicó spaCy `es_core_news_md` (CPU, `n_process=8`, parser y NER deshabilitados): tokenización, minúsculas, lematización y parada en consumo con 521 stopwords de spaCy + **51 muletillas del habla colombiana** (*eh, mmm, este, o sea, pues*) + 42 tokens de ruido tipográfico de libros (palabras partidas por guion, `ibid.`). Se conservan años y categorías canónicas como tokens. Las **62 correcciones de lemas** (`atreveír`→`atrever`, voseo, truncamientos orales) se aplican en consumo desde `config/lemas_correcciones.json`.

Producción: libros 917 documentos en 30,5 s; entrevistas 2.486 en ≈ 509,5 s (4,88 docs/s), con **40.346.344 tokens crudos y 10.923.423 lemas** (ratio 0,271); 2 entrevistas quedan legítimamente vacías. El pipeline es determinista y **reanudable** (checkpoint por documento). Corpus: `data/corpus/segmentos_libros.jsonl` (917), `entrevistas_normalizadas.jsonl` (2.486) y `preprocesado/{libros,entrevistas}_{lemas,tokens}.txt` + `*_docs.json`. Calidad: ruff 0 y 51 pruebas pytest aprobadas.

## 4. P2 — Análisis exploratorio del corpus

### 4.1 Estadísticas y diversidad léxica

Con parada en consumo y correcciones: libros 917 documentos, **928.253 lemas de contenido**, 33.277 tipos; entrevistas 2.486 documentos, **9.702.534 lemas**, 109.391 tipos. La extensión es muy asimétrica en entrevistas (media 3.903 lemas vs. 1.012 por segmento): `../imagenes/eda_entrevistas_hist_lemas.png`, `../imagenes/eda_libros_hist_lemas.png`, `../imagenes/eda_entrevistas_hist_chars.png`, `../imagenes/eda_entrevistas_hist_paginas.png`. Por la dependencia del TTR con la longitud (Bestgen, 2023), la diversidad se midió con TTR de muestra fija (n=1.000) y Yule K: los libros son más diversos (**TTR_m1000 0,609 vs. 0,514; Yule K 31,8 vs. 38,7**; menor K = más diverso); la oralidad transcrita es más repetitiva (Moreno-Sandoval et al., 2024). Ver `../imagenes/eda_diversidad_lexica.png`.

### 4.2 Frecuencias, nubes y comparación léxica

Los top-10 de unigramas muestran dos mundos léxicos: libros con sustantivos institucionales (*armado, víctima, grupo, violencia, comisión, mujer, conflicto*); entrevistas con verbos de narración oral (*gente, llegar, año, pasar, cosa, venir, salir, querer*). Nubes uni/bi/trigramas por corpus desde las frecuencias top-30 (`resultados/eda/frecuencias_*.csv`): `../imagenes/eda_libros_nube_unigramas.png`, `../imagenes/eda_libros_nube_bigramas.png`, `../imagenes/eda_libros_nube_trigramas.png`, `../imagenes/eda_entrevistas_nube_unigramas.png`, `../imagenes/eda_entrevistas_nube_bigramas.png`, `../imagenes/eda_entrevistas_nube_trigramas.png`.

El solapamiento de vocabulario es bajo: **Jaccard = 0,221** (25.851 tipos compartidos); los libros cubren el **95,7 % de los tokens de las entrevistas** y las entrevistas el 98,3 % de los de libros. Los distintivos por log-odds (`../imagenes/eda_comparacion_distintivos_logodds.png`) lo confirman: en libros, términos institucionales y transmedia (*transmedia* 8,0; *externas* 7,8; *microdato* 7,4); en entrevistas, categorías canónicas —evidencia de que la normalización capturó vocabulario real del habla— (*anonimo* 9,1; *organizacion_social* 8,5; *organizacion_publica* 8,2) y formas orales como *far* (519 ocurrencias) frente a *farc* en los libros: desajuste relevante para el matching. El tesauro (ADR-004) aportó etiqueta SKOS a 14 de los 50 distintivos (p. ej. *cnmh*).

### 4.3 Near-duplicates

Con Jaccard de shingles de 5 lemas (minhash k=60 + LSH 20×3 y verificación exacta; Broder, 1997) se detectaron **345 pares casi idénticos (> 0,85)** sobre 368 candidatos verificados (**526 documentos únicos, 242 grupos, ≈ 21 %**): `../imagenes/eda_near_duplicates_entrevistas.png`. **No se eliminan** (corpus oficial íntegro); se documenta porque producen rankings casi idénticos.

## 5. P3 — Modelo de IR con TF-IDF

### 5.1 Configuración y dirección del matching

Por ADR-005, **cada entrevista es una query y la colección son los 917 segmentos de libro**. Se usa `TfidfVectorizer` de scikit-learn sobre textos pre-lematizados, con `config/tfidf_config.json` congelado en el EDA: `ngram_range=(1,2)` (colocaciones de dominio: *desaparición forzada, violencia sexual, falsos positivos*); `min_df=2` (43,2 % de hapax: 14.365 de 33.277 tipos con df=1); `max_df=0.85` (solo 6 tipos aparecen en > 85 % de los segmentos); `sublinear_tf=True` (Salton y Buckley, 1988), `norm='l2'` (Manning et al., 2008), `use_idf` y `smooth_idf=True`. Resultado: **118.748 features** (18.864 unigramas + 99.884 bigramas; densidad 0,0082). El fit se hace solo sobre la colección y las queries se transforman con el mismo vectorizador (sin leakage); coseno por batches (10,0 s, semilla 42), generando `resultados/ranking_top5.csv` (12.420 filas) y `ranking_top10.csv`.

### 5.2 Resultados del ranking

El modelo recupera ≥ 1 segmento para 2.484 de 2.486 entrevistas (**cobertura 99,92 %**; las 2 restantes son los vacíos legítimos). El score top-1 tiene media 0,1468, mediana 0,138 y rango 0,070-0,572. La distribución del top-1 revela un sesgo marcado: **76,77 % en TESTIMONIAL** (FUERA 5,19 %, NARRATIVA 4,35 %, ETNICO 3,82 %, VIOLACIONES 2,94 %; `resultados/ir_tfidf_report.json`). La revisión lo confirmó como **efecto de dominio, no artefacto**: el tomo testimonial reproduce el registro oral de las víctimas (Castillejo Cuéllar, 2023); se descartaron artefactos alternativos (longitud de queries y categorías canónicas no explican la magnitud). La evaluación confirma top-1 plausibles pero genéricos.

### 5.3 Variantes y evaluación cualitativa

Las variantes del modelo confirman la contribución de los bigramas: sin correcciones de lemas (C) el top-1 coincide en 96,6 % con el base; sin muletillas (B), 84,6 %; con solo unigramas (A), 54,8 % (`resultados/ranking_top5_{A,B,C}.csv`). Se construyó una muestra determinista (semilla 42), estratificada por tomo, de **30 pares (entrevista, segmento top-1)** con top-3 completo; el etiquetado humano-proxy clasificó cada par en {relacionado, parcial, no} (Tabla 3).

**Tabla 3. Evaluación cualitativa sobre 30 pares** (fuente: `resultados/evaluacion_metricas.json`)

| Métrica | Valor |
|---|---|
| Etiquetas: relacionado / parcial / no | 16 / 14 / 0 |
| P@1 estricto (solo "relacionado") | 0,5333 |
| P@1 aceptable ("relacionado" + "parcial") | 1,0000 |
| P@3 aceptable (al menos un "relacionado" en r1-r3) | 0,7667 |
| P@3 aceptable ("relacionado" + "parcial" en r1-r3) | 1,0000 |
| Top-1 "parcial" con match "relacionado" en r2-r3 | 7 de 30 (23 %) |

El matching léxico es **funcional pero impreciso**: con criterio estricto P@1 = 0,53 y P@3 = 0,77; con criterio aceptable llegan a 1,0 porque todo el corpus trata el conflicto y cualquier entrevista de víctima solapa mínimamente con casi cualquier segmento. No hay top-1 etiquetado "no" (0 falsos positivos duros), pero en 7/30 queries el top-1 es "parcial" y el match específico quedó en r2-r3. Ejemplos: acierta con términos específicos (amenaza de masacre → `HALLAZGOS-p129-las-masacres`, 0,132) y falla por solape genérico en testimonios (excombatiente indígena amazónica → `TESTIMONIAL-p401-senderos-pedregosos`). **Decisión de iteración**: no se creó el experimento 2; el criterio del plan (P@1 aceptable ≥ 0,6) se cumple (1,0) y el problema no es el registro oral —las muletillas ya están en parada— sino la falta de discriminación temática fina.

## 6. P4 — Métricas de relevancia: Rocchio y Okapi BM25

El punto 4 exige **implementar manualmente** ambas métricas. La sección 03b del notebook lo hace desde cero (sin `rank_bm25` ni librerías de relevance feedback) sobre la misma dirección de matching y vocabulario del base: la única diferencia frente al TF-IDF es la función de puntuación.

### 6.1 Okapi BM25

Fórmula implementada (Robertson y Walker, 1994; véase Robertson y Zaragoza, 2009):

```text
score(q, d) = sum_t qtf(t) * IDF(t) * [ tf(t,d) * (k1 + 1) ]
              / [ tf(t,d) + k1 * (1 - b + b * |d| / avgdl) ]

IDF(t) = ln( (N - df(t) + 0,5) / (df(t) + 0,5) + 1 )      con k1 = 1,5 y b = 0,75
```

`tf` es la frecuencia cruda (la saturación la aporta el denominador, que tiende a `k1+1`), `k1` controla la saturación (1,5) y `b` la normalización por longitud (0,75); el IDF de Robertson-Sparck Jones premia términos raros (con 43 % de hapax, reordena la cola). Se usa `CountVectorizer` solo como tokenizador/contador (permitido) y funciones propias documentadas: `construir_matriz_tf`, `idf_bm25`, `calcular_longitudes_y_df`, `puntuar_bm25` y `rankear_bm25`. Colección: 917 × 118.748 términos, 891.794 no nulos, **avgdl = 1.466,26**, longitudes 155-2.741.

**Unidad de |d|.** La longitud de Okapi se mide en **términos del vocabulario** (unigramas + bigramas), no en lemas de contenido (**1.007,5 de media**: 923.912/917); es la fórmula estándar sobre el mismo espacio de términos que el TF-IDF.

**Auto-check y resultados.** El auto-check sintético (3 documentos a mano) da esperado `[0.671434, 0.552945, 0.0]` = obtenido → **True**. El ranking (0,6 s de puntuación; 9,3 s total) está en `resultados/ranking_bm25.csv` (12.420 filas). Top-1: **1.863 TESTIMONIAL (75,0 % de las 2.484; 74,9 % de las 2.486)**; coincidencia con el top-1 base: **57,0 %** (auto-check > 40 %: True).

### 6.2 Rocchio con pseudo-relevancia

```text
q' = alpha * q + beta * (1/|D_r|) * sum_{d en D_r} d
                - gamma * (1/|D_nr|) * sum_{d en D_nr} d        con alpha=1, beta=0,75, gamma=0,15
```

`alpha` ancla la consulta; `beta` expande hacia los relevantes; `gamma` resta los no relevantes. Sin gold, se usa **pseudo-relevance feedback**: top-5 del base = `D_r`, bottom-5 = `D_nr` (Manning et al., 2008, cap. 9); el re-rankeo usa coseno contra los 917 segmentos (`rocchio_expandir`, `rankear_rocchio`). Auto-check con vectores conocidos: `[1.375, 0.375, -0.15]` a mano = `[0.959464, 0.261672, -0.104669]` normalizado, con beta positivo y gamma negativo → **True**. Resultado: 2.484 queries re-rankeadas, top-1 cambia en **743 (29,9 %)**, nuevo top-1 **1.947 TESTIMONIAL (78,4 %)**, 18,3 s (`resultados/ranking_rocchio.csv`, 12.420 filas).

### 6.3 Comparación y discusión

Sin gold, la comparación es de **acuerdo**: solapamiento de **conjunto** del top-k (|A_k ∩ B_k|/k) y de **posición** k exacta (Tabla 4). De 2.484 queries, **1.112 coinciden en el top-1 de los tres métodos y 296 difieren en los tres** (Tabla 5).

**Tabla 4. Acuerdo entre métodos** (2.484 queries; fuente: sección 03b del notebook)

| Par | Top-1 conjunto | Posición 5 | Conjunto top-5 |
|---|---|---|---|
| base vs. BM25 | 57,0 % | 11,8 % | 65,7 % |
| base vs. Rocchio | 70,1 % | 48,8 % | 100,0 % |
| BM25 vs. Rocchio | 50,6 % | 10,5 % | 65,7 % |

**Tabla 5. Top-1 por tomo** (fuente: `resultados/ranking_bm25.csv`, `ranking_rocchio.csv` y reporte TF-IDF)

| Tomo | base | BM25 | Rocchio |
|---|---|---|---|
| TESTIMONIAL | 1.907 (76,77 %) | 1.863 (75,0 %) | 1.947 (78,4 %) |
| FUERA | 129 | 127 | 123 |
| NARRATIVA | 108 | 103 | 84 |
| ETNICO | 95 | 115 | 99 |
| VIOLACIONES | 73 | 105 | 71 |
| HALLAZGOS | 53 | 43 | 59 |
| CUERPO | 50 | 52 | 38 |
| SUFRIR | 46 | 55 | 45 |
| NNA | 22 | 19 | 17 |
| CONVOCATORIA | 1 | 2 | 1 |

Lectura:

1. **BM25 y TF-IDF comparten señal pero ordenan distinto.** El 57,0 % de top-1 idéntico y el 65,7 % de conjunto top-5 indican señal común, pero solo el 11,8 % coincide en la posición 5 exacta. BM25 normaliza por longitud (`b=0,75`, sin ventaja por volumen) y usa un IDF más agresivo con los términos raros que el `smooth_idf` de scikit-learn.
2. **Rocchio re-rankea, no sustituye el inventario.** Conserva el 70,1 % de los top-1 y su conjunto top-5 es **idéntico al del base en el 100,0 %** (posición 5 exacta en 48,8 %). Domina `beta`; los bottom-5 casi siempre puntúan 0 y `gamma` aporta poco. Hereda el sesgo del base (*query drift*).
3. **El sesgo TESTIMONIAL es del dominio, no de la función.** Los tres métodos concentran el top-1 en TESTIMONIAL (76,77 % / 74,9 % / 78,4 %): efecto de registro oral (sección 5.2; Castillejo Cuéllar, 2023).
4. **Cualitativamente**, donde los tres coinciden hay solape específico; donde difieren, el base devuelve un testimonial genérico, BM25 premia términos raros y Rocchio reordena hacia el centroide (para una entrevista: *Algo me atravesó* vs. *¿Escuchaste eso?* vs. *Lo inhabitable*).
5. **El acuerdo mide consistencia, no acierto**: sin gold no hay P@k real para BM25/Rocchio; la verificación humana son los 30 pares de la sección 5.3.

## 7. P5 — Comparación de corpus: heatmap y cuadro de vínculos

Se entregan un **heatmap 10×10** y un **cuadro de 30 vínculos** (3 por tomo). Las **filas** agrupan las 2.486 entrevistas por el tomo del top-1 **BM25**; las **columnas** son los 10 tomos; la **celda (g, T)** es la similitud coseno TF-IDF **media** entre las entrevistas del grupo g y los segmentos del tomo T (matriz del base, sin re-entrenar). No se normaliza por número de segmentos porque la media por par ya es intensiva (afinidad media, no masa total); el CSV guarda `n_entrevistas`. La **diagonal** mide la coherencia entre la asignación BM25 y la similitud TF-IDF. Figura: `../imagenes/heatmap_entrevistas_libros.png`; matriz: `resultados/heatmap_tomos.csv`; resumen: Tabla 6.

**Tabla 6. Resumen del heatmap por grupo** (coseno TF-IDF medio; media global fuera de diagonal 0,0608; fuente: `resultados/heatmap_tomos.csv`)

| Grupo (n) | Diagonal | Media de fila | Máximo de fila | Diag. > media fila |
|---|---|---|---|---|
| HALLAZGOS (43) | 0,0666 | 0,0649 | VIOLACIONES 0,0711 | Sí |
| CONVOCATORIA (2) | 0,0767 | 0,0745 | TESTIMONIAL 0,0793 | Sí |
| NARRATIVA (103) | 0,0722 | 0,0647 | TESTIMONIAL 0,0766 | Sí |
| SUFRIR (55) | 0,0674 | 0,0630 | TESTIMONIAL 0,0725 | Sí |
| TESTIMONIAL (1.863) | 0,0904 | 0,0540 | TESTIMONIAL 0,0904 | Sí |
| CUERPO (52) | 0,0662 | 0,0603 | TESTIMONIAL 0,0749 | Sí |
| NNA (19) | 0,0702 | 0,0556 | TESTIMONIAL 0,0753 | Sí |
| VIOLACIONES (105) | 0,0655 | 0,0602 | TESTIMONIAL 0,0771 | Sí |
| ETNICO (115) | 0,0756 | 0,0619 | ETNICO 0,0756 | Sí |
| FUERA (127) | 0,0747 | 0,0608 | FUERA 0,0747 | Sí |
| **Global** | **0,0725** | — | — | **10/10** |

**Interpretación.** La diagonal media (**0,0725**) supera a la media fuera (**0,0608**) y a la media de fila en 10/10 grupos: la asignación BM25 no es ruido. TESTIMONIAL concentra 1.863 entrevistas (74,9 %) y su diagonal es la mayor (0,0904). El **máximo de fila cae en la diagonal solo en 3/10** (TESTIMONIAL, ETNICO, FUERA); en 6 de los 7 restantes cae en TESTIMONIAL y en HALLAZGOS en VIOLACIONES (0,0711 vs. 0,0666): lectura visual del sesgo de registro oral. Fuera de la diagonal dominan TESTIMONIAL (0,066-0,079) y NARRATIVA (0,045-0,078); las celdas más bajas están en CONVOCATORIA.

**Cuadro de vínculos.** `resultados/vinculos_entrevistas_libros.csv` contiene las **30 parejas** (3 por tomo) con mayor score BM25; la Tabla 7 muestra la mejor de cada tomo. Las afinidades altas corresponden a solapes temáticos reconocibles (masacres, secuestro, pueblos étnicos, violencia sexual, persecución).

**Tabla 7. Mejor vínculo entrevista → segmento por tomo (BM25)** — fuente: `resultados/vinculos_entrevistas_libros.csv`

| Tomo | Score BM25 | Título del segmento |
|---|---|---|
| HALLAZGOS | 13.122,8 | La construcción histórica del actual modelo de seguridad y sus principales rasgos |
| CONVOCATORIA | 6.296,5 | Primer paso del esclarecimiento: acoger la realidad de las víctimas |
| NARRATIVA | 21.359,9 | 83. El body count o la doctrina Vietnam |
| SUFRIR | 14.078,4 | Los obstáculos a la participación electoral y al derecho al voto |
| TESTIMONIAL | 32.700,2 | Desafiar la realidad y los futuros deseados |
| CUERPO | 8.072,2 | Afrontamientos de personas LGBTIQ+ víctimas del conflicto armado en Colombia |
| NNA | 4.667,8 | Ataques a comunidades e instituciones educativas |
| VIOLACIONES | 20.757,3 | «No me pregunten porque yo no les sé darles respuesta...»: torturas y tratos crueles |
| ETNICO | 19.036,7 | Violencias, daños y resistencias de los pueblos étnicos en el conflicto armado |
| FUERA | 15.560,5 | La persecución y la ausencia de protección en el marco de la responsabilidad del Estado |

**Limitaciones.** (1) CONVOCATORIA tiene n=2: su diagonal (0,0767) es frágil. (2) La agrupación hereda el sesgo TESTIMONIAL: mide coherencia, no corrección. (3) Sin gold, afinidad alta no implica relevancia (el léxico del conflicto infla las celdas).

## 8. P6 — Conclusiones, limitaciones y mejoras futuras

### 8.1 Conclusiones

1. **P1.** 917 segmentos y 2,27 M palabras (0 solapes/vacíos/fuera de banda) con morfología documentada (ADR-001); 2.486 entrevistas normalizadas (−5,82 %) y preprocesadas (40,35 M tokens → 10,92 M lemas), con 8 categorías canónicas y 62 correcciones.
2. **P2.** Libros más diversos (TTR_m1000 0,609 vs. 0,514; Yule K 31,8 vs. 38,7), Jaccard 0,221 con cobertura asimétrica (95,7 %/98,3 %), desajustes relevantes (*far*/*farc*) y 345 near-duplicados sin eliminar.
3. **P3.** Cobertura 99,92 % y P@1 aceptable 1,0, pero P@1 estricto 0,533 y P@3 0,767: el primer resultado no siempre es el más específico (7/30); el sesgo TESTIMONIAL (76,77 %) es efecto de dominio verificado.
4. **P4.** BM25 y Rocchio funcionan con auto-checks en verde; BM25 comparte el 57,0 % del top-1 y el 65,7 % del conjunto top-5; Rocchio reordena el 29,9 % de los top-1 conservando el 100 % del inventario. Los tres comparten el sesgo TESTIMONIAL.
5. **P5.** La diagonal (0,0725) supera a la afinidad fuera (0,0608) y a la media de fila en 10/10 grupos; el máximo de fila cae en la diagonal en 3/10, evidencia del sesgo de registro oral; los 30 vínculos muestran afinidades temáticas reconocibles.
6. **P6.** Trazabilidad completa (`resultados/metricas.json`, semilla 42, auditoría sin PII, 51 pruebas, ruff 0). Autoevaluación en `plan/verificacion_rubrica.md`; requisitos 10/10 en `plan/verificacion_requisitos.md`.

### 8.2 Limitaciones

(a) Sin gold: evaluación proxy de 30 pares. (b) Lematizador imperfecto en voseo/pretéritos (mitigado con 62 correcciones). (c) Los 345 near-duplicados replican rankings y sobreestiman la cobertura. (d) El matching léxico no discrimina hechos o poblaciones específicas. (e) El intermedio conserva PII hablada (no distribuible); los finales están limpios. (f) |d| de BM25 en términos del vocabulario (avgdl 1.466,26), no en lemas (1.007,5).

### 8.3 Mejoras futuras

(i) BM25 tuneado (k1 ∈ [0,9, 2,0], b ∈ [0,3, 1,0]). (ii) Embeddings/transformers y reranking supervisado del top-k. (iii) Gold estándar externo con acuerdo inter-etiquetador. (iv) Eliminación o agrupación de near-dups. (v) Enrutamiento por tomo para corregir el prior TESTIMONIAL. (vi) Cambio de hablante, chunking de entrevistas largas y limpieza de *running heads*.

## 9. Entregables y reproducibilidad

**Estructura** (`TALLER-4/`):

```text
config/          config curada (toc, stopwords, correcciones, tf-idf)
data/raw/        entrevistas JSON, libros (zip/PDF) y tesauro RDF (3 insumos, sin tocar)
data/corpus/     segmentos_libros.jsonl (917), entrevistas_normalizadas.jsonl (2.486), preprocesado/
notebooks/       Taller_4_IR_CEV.ipynb (ENTREGABLE, 120 celdas); notebook_cells/ (secciones fuente)
scripts/         01_extraer_libros.py, 02_preprocesar.py, 03_eda.py, 04_evaluar_muestra.py, ensamblador
experimentos/    experimento_1_tfidf_base.py (modelo IR base y variantes A/B/C)
imagenes/        14 figuras eda_*.png + heatmap_entrevistas_libros.png
resultados/      rankings TF-IDF/BM25/Rocchio, heatmap_tomos.csv, vinculos, reportes y métricas
tests/           pytest (51 pruebas); informe/ (este documento); plan/ (PLAN, ADRs, verificaciones)
```

**Ejecución local.** Con el venv del proyecto (`pandas`, `scikit-learn`, `spacy` + `es_core_news_md`, `matplotlib`, `seaborn`, `wordcloud`, `nbclient`; `poppler-utils` solo para regenerar PDFs) basta abrir el notebook y ejecutar todo. Pipeline por fases: `01_extraer_libros.py` → `02_preprocesar.py` → `03_eda.py` → `experimento_1_tfidf_base.py` (`--variante A|B|C`) → `04_evaluar_muestra.py`.

**Colab.** Subir `TALLER-4` a `/content` (o descomprimir el zip en `/content/PLN-main/TALLER-4`; también se detecta con `TALLER4_ROOT` o `AUTO_DESCARGAR_REPO=True`), descargar manualmente los 3 insumos a `data/raw/` y ejecutar todo. "Sec 0b — Regeneración" (`REGEN_AUTOMATICO=True`) regenera lo faltante con el código canónico expandido como celdas normales; la primera corrida sin caches tarda 15-60 min. `config/`, `resultados/` e `imagenes/` son opcionales: la configuración curada viaja embebida y las salidas de P4-P5 se **escriben** en cada corrida.

**Independencia (nota importante).** El notebook recibe **solo los 3 insumos** y genera todo lo demás. Evidencia: (i) ejecución aislada end-to-end con solo el `.ipynb` y los 3 insumos: **0 errores en 922 s (~15,4 min)**; (ii) sandbox de tests sin `config/`, `resultados/`, `imagenes/` ni `scripts/` (65,8 s) y 51 pruebas pytest; (iii) corrida completa en el repositorio: 61,7 s, 0 errores, modelo base idéntico al oficial en **2.484/2.484 top-1** (TESTIMONIAL 76,77 %; variantes 54,8/84,6/96,6; P@1 0,5333/1,0 y P@3 0,7667/1,0); (iv) ruff 0. Detalle en `plan/verificacion_independencia_notebook.md`.

## 10. Referencias

Referencias verificadas (URL/DOI) en `plan/estado_del_arte.md`:

1. Sosa, J., Urrego-López, A., Prieto, C., Camargo-Díaz, E. J. (2025). Constructing the Truth: Text Mining and Linguistic Networks in Public Hearings of Case 03 of the JEP. arXiv:2504.04325. https://arxiv.org/abs/2504.04325
2. Valentini, F., Cotik, V., Furman, D., Bercovich, I., Altszyler, E., Pérez, J. M. (2024). MessIRve: A Large-Scale Spanish Information Retrieval Dataset. arXiv:2409.05994. https://arxiv.org/abs/2409.05994
3. Salton, G., Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing & Management, 24(5). DOI 10.1016/0306-4573(88)90021-0
4. Robertson, S., Walker, S. (1994). Some simple effective approximations to the 2-Poisson model for probabilistic weighted retrieval. SIGIR '94. DOI 10.1007/978-1-4471-2099-5_24. Véase también Robertson, S., Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in IR, 3(4). DOI 10.1561/1500000019
5. Rocchio, J. J. (1971). Relevance feedback in information retrieval. En G. Salton (ed.), The SMART Retrieval System: Experiments in Automatic Document Processing, Prentice-Hall, pp. 313-323.
6. Manning, C. D., Raghavan, P., Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press. https://nlp.stanford.edu/IR-book/
7. Bender, E. M., Friedman, B. (2018). Data Statements for Natural Language Processing. TACL, 6. DOI 10.1162/tacl_a_00041
8. Bestgen, Y. (2023). Measuring Lexical Diversity in Texts: The Twofold Length Problem. arXiv:2307.04626. https://arxiv.org/abs/2307.04626
9. Moreno-Sandoval, L. G., et al. (2024). Detection of Hate Speech, Racism and Misogyny in Digital Social Networks: Colombian Case Study. Big Data Cogn. Comput., 8(9):113. DOI 10.3390/bdcc8090113
10. Castillejo Cuéllar, A. (2023). Escuchar de otra manera. Oralidad y sonido en el volumen testimonial de la Comisión de la Verdad. Estudios Políticos, 68. DOI 10.17533/udea.espo.n68a02
11. Broder, A. Z. (1997). On the resemblance and containment of documents. DOI 10.1109/SEQUEN.1997.666900
12. spaCy (2024). Modelo es_core_news_md. https://spacy.io/models/es
13. scikit-learn (2024). TfidfVectorizer. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
14. Comisión de la Verdad (2022). Hay futuro si hay verdad. Informe Final. https://www.comisiondelaverdad.co
