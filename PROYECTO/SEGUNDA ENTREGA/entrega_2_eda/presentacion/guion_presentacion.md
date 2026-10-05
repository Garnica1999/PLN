# Guion de la presentación — Entrega 2

**Proyecto:** Análisis de sentencias de la Corte Constitucional de Colombia (1992–2021)
**Entrega 2:** EDA avanzado, representación de textos y propuesta metodológica
**Duración sugerida:** 10–12 minutos (≈40–48 s por diapositiva)
**Formato:** 15 diapositivas (16:9), español, sin emojis. Todas las cifras provienen del informe final (`informe/main.tex`), del notebook ejecutado (`resultados/metricas_entrega_ii.json`) y de los artefactos del EDA extendido (`resultados/eda_extendido/`).

**Nota de uso:** el deck no presenta resultados de modelos: la diapositiva 14 describe la propuesta metodológica como diseño experimental. Las notas del orador de cada diapositiva están también en el PPTX (vista de notas).

---

## Diapositiva 1 — Portada

**Contenido:**
- Análisis de sentencias de la Corte Constitucional de Colombia (1992-2021)
- Entrega 2: EDA avanzado, representación de textos y propuesta metodológica
- Erika Valeria Rivera Jiménez · Julian Guillermo Carrillo Meneses · Carlos Andrés Garnica Salazar · Oscar Andrés Prieto Moreno
- Pontificia Universidad Javeriana · Octubre de 2026

**Figuras:** fondo institucional de la plantilla.

**Notas del orador:** Presentamos la Entrega 2: EDA avanzado, representación de textos y propuesta metodológica del proyecto de PLN sobre jurisprudencia constitucional colombiana.

---

## Diapositiva 2 — Contenido (ruta de la presentación)

**Contenido:**
- 01 Resumen de la Entrega 1 — Corpus, limpieza y textometría básica (03–04)
- 02 Estado del arte — Antecedentes y representación (05–06)
- 03 Corpus medido y EDA avanzado — Dataset, n-gramas, diversidad y diacronía (07–12)
- 04 Representación y metodología — TF-IDF, embeddings y propuesta de modelado (13–14)
- 05 Conclusiones — Hallazgos y trabajo futuro (15)

**Figuras:** ninguna (agenda de la plantilla).

**Notas del orador:** La ruta de la presentación: resumen de la Entrega 1, estado del arte, el trabajo de la Entrega 2 y las conclusiones con el trabajo futuro.

---

## Diapositiva 3 — Resumen de la Entrega 1: corpus, del dato crudo al analizable

**Contenido:**
- INGESTA: 23.750 registros de texto crudo no estructurado.
- EXTRACCIÓN: expresiones regulares para ID, tipo y año; 99,97 % de eficacia.
- DEPURACIÓN: 8 exclusiones (0,03 %): 6 autos + 2 fragmentos HTML.
- CONSOLIDACIÓN: 23.742 sentencias; corpus limpio y estructurado.
- VALIDACIÓN: muestra estratificada de 25 documentos para inspección.
- RESULTADO: corpus analizable, trazable y preparado para vectorización o modelado.

**Figuras:** diagrama de flujo de tarjetas (plantilla).

**Notas del orador:** Síntesis de la Entrega 1: el corpus se depuró desde 23.750 registros crudos; la extracción por expresiones regulares logró 99,97 % de eficacia; se excluyeron 8 filas (6 autos y 2 HTML) y quedaron 23.742 sentencias con una muestra estratificada de 25 documentos para inspección.

---

## Diapositiva 4 — Resumen de la Entrega 1: limpieza y textometría básica

**Contenido:**
- 23.742 sentencias · ≈ 10.993 palabras por sentencia · ≈ 48 % de tokens funcionales identificados.
- Reducción del volumen textual: 260,99 M palabras crudas → 250,01 M tokens normalizados → 128,81 M tokens sin stopwords (base informativa final).
- HALLAZGO: las sentencias son documentos extraordinariamente extensos y presentan una alta repetición de estructuras lingüísticas.

**Figuras:** tarjetas de estadísticas (plantilla).

**Notas del orador:** La limpieza en ocho pasos normalizó el texto (minúsculas, NFD, preservación de la ñ, filtro de ruido y 306 stopwords normalizadas). La textometría básica midió 260,99 millones de palabras crudas, 250,01 millones de tokens normalizados y 128,81 millones de tokens sin stopwords: las sentencias son documentos muy extensos y formulaicos.

---

## Diapositiva 5 — Estado del arte: PLN legal en español y preprocesamiento

**Contenido:**
- Anzola Cortés (2021): minería de datos sobre más de 6.000 sentencias de constitucionalidad (1992-2020): el mismo universo documental del corpus.
- Barbosa Becerra (2021): análisis léxico-discursivo de sentencias de la Corte; evidencia fórmulas estables y léxico especializado.
- HaCohen-Kerner et al. (2020): el preprocesamiento cambia los resultados según el corpus y la tarea: fundamenta documentar cada paso de limpieza.
- Soh et al. (2019): corpus de sentencias largas: cautela con las frecuencias globales; comparación de clasificadores clásicos y neuronales.
- Aletras et al. (2016) y González-González et al. (2023): el poder predictivo varía por secciones de la sentencia (TEDH) y la clasificación de sentencias en español usa diccionarios de dominio.
- Löffler et al. (2025): escasez de recursos legales anotados en español para América Latina.
- No hay trabajos indexados que usen el dataset exacto de Hugging Face (verificado 2026-09-05); los antecedentes cubren el corpus subyacente y la metodología.

**Figuras:** ninguna (panel de referencias).

**Notas del orador:** El estado del arte de esta entrega se apoya en el trabajo sobre el mismo corpus, la evidencia de que el preprocesamiento importa y los antecedentes de clasificación de sentencias largas y en español.

---

## Diapositiva 6 — Estado del arte: representación, diversidad y diacronía

**Contenido:**
- Manning et al. (2008): fundamentos de recuperación de información y ponderación TF-IDF.
- Dunning (1993): razón de verosimilitud (LLR) para la estadística de colocaciones.
- McCarthy y Jarvis (2010): MTLD como medida de diversidad léxica robusta a la longitud del texto.
- Hamilton et al. (2016): log-odds con suavizado para comparar vocabularios entre periodos.
- Chalkidis et al. (2020, 2021): LEGAL-BERT y MultiEURLEX: vocabulario jurídico distinto del general y deriva temporal que exige particiones cronológicas.
- Reimers y Gurevych (2019), Cañete et al. (2020) y Wang et al. (2022): Sentence-BERT, BETO y e5 multilingüe como representaciones densas de comparación.
- Estas referencias fundamentan la representación (TF-IDF y embeddings), la medida de colocaciones y el análisis diacrónico de esta entrega.

**Figuras:** ninguna (panel de referencias).

**Notas del orador:** Para la representación y la diacronía nos apoyamos en los fundamentos de recuperación de información, la razón de verosimilitud, la MTLD, el log-odds y los modelos contextuales en español y de dominio legal.

---

## Diapositiva 7 — Trabajo de la Entrega 2: correcciones respecto a la Entrega 1

**Contenido:**
- Dataset medido: parquet de 23.742 filas × 12 columnas con texto, texto limpio y métricas por documento.
- EDA extendido: n-gramas, colocaciones, lemas/stems/POS, MTLD, granularidad y recursos léxicos.
- Visualizaciones: siete figuras nuevas; las figuras de la Entrega 1 se conservan.
- Unidad validada: sentencia 97,17 % vs subsentencia 4,64 % (criterio pre-registrado de +15 pp).
- Periodos: contraste léxico 1992-2001 / 2002-2011 / 2012-2021 ejecutado (log-odds).
- Representación y metodología: TF-IDF sobre lemas y propuesta de modelado definidas.
- Las cifras canónicas de la Entrega 1 (23.742 documentos; 260.988.316 / 250.012.101 palabras) no requirieron corrección.

**Figuras:** ninguna (panel de síntesis).

**Notas del orador:** Respecto a la Entrega 1 no hay rectificaciones de cifras: los cambios son extensiones verificadas, del dataset medido al EDA extendido, la unidad validada, la diacronía y la representación con la propuesta metodológica.

---

## Diapositiva 8 — Corpus medido: conjunto de datos y verificación

**Contenido:**
- 23.742 filas verificadas · 12 columnas · 1.544,7 MB de tamaño del parquet.
- Esquema del conjunto medido: 260.988.316 palabras crudas → 250.012.101 palabras limpias → 248.533.344 lemas totales.
- VERIFICACIÓN: 23.742 ids únicos, sin nulos en claves; sumas reconciliadas; muestra de 100 filas + diccionario de datos.

**Figuras:** tarjetas de estadísticas (plantilla).

**Notas del orador:** El requisito 3 se cumple con un parquet de 23.742 filas por 12 columnas, verificado con aserciones duras: sin nulos, ids únicos y sumas reconciliadas con las cifras canónicas. Se adjuntan muestra estratificada, diccionario de datos y lemas por documento.

---

## Diapositiva 9 — EDA avanzado: n-gramas y colocaciones

**Contenido:**
- Figuras: top-15 de bigramas y trigramas más frecuentes; colocaciones por LLR (top-10).
- Validación manual de 20 colocaciones: LLR 75 % vs frecuencia 55 % y PMI 0 %.
- HALLAZGO: accion de tutela (344.893) es el trigrama más frecuente; el 82,7 % de los trigramas guardados responde al patrón con palabra intermedia vacía.

**Figuras:** `fig_ngramas_bi_tri.png` y `fig_colocaciones.png`.

**Notas del orador:** El corpus es fuertemente formulaico: accion de tutela domina y la validación manual de 20 colocaciones da 75 por ciento de fórmulas válidas con la razón de verosimilitud, frente a 55 por ciento de la frecuencia y 0 por ciento del PMI.

---

## Diapositiva 10 — EDA avanzado: diversidad léxica y morfología

**Contenido:**
- Figuras: MTLD por tipo de providencia y evolución anual; compresión del vocabulario (formas, lemas y stems).
- HALLAZGO: MTLD 61,03 (T 61,63 > C 59,54 > SU 58,88); descenso sostenido de 64,32 a 58,53 entre periodos; lemas 87,3 % y stems 54,2 % del vocabulario de formas.

**Figuras:** `fig_mtld_distribucion.png` y `fig_lemma_vs_stem.png`.

**Notas del orador:** La diversidad léxica desciende de forma sostenida en las tres décadas y la lematización reduce el vocabulario conservando la legibilidad: es la base de la representación TF-IDF.

---

## Diapositiva 11 — EDA avanzado: evolución léxica por periodos

**Contenido:**
- Figura: log-odds con suavizado add-0,5 (2002-2011 → 2012-2021): top-10 que sube y top-10 que baja.
- HALLAZGO: suben uariv, jep y coronavirus; bajan nilser y fomin. El método replica la referencia interna (solapamiento del top-10: 7/10 a 9/10); siglas y ruido OCR se conservan como límite documentado.

**Figuras:** `fig_logodds_periodos.png`.

**Notas del orador:** El contraste por periodos cumple la promesa de la Entrega 1: en la última década gana peso el vocabulario del conflicto armado y la pandemia. Los extremos incluyen siglas y ruido de OCR; el método es fiel a la referencia interna y se reporta el límite.

---

## Diapositiva 12 — EDA avanzado: recursos léxicos y ontologías

**Contenido:**
- EuroVoc / MultiEURLEX: tesauro multilingüe (español) de derecho de la UE; multietiqueta; aplicabilidad parcial (no mapea la Constitución ni la tutela).
- LKIF: ontología formal OWL de conceptos jurídicos; aplicabilidad baja; útil como patrón de modelado.
- ELI / ECLI: esquemas de identificación y metadatos; no son vocabularios temáticos.
- LegalRuleML / Akoma Ntoso: marcado de reglas y estructura; baja a corto plazo (anotación manual costosa).
- Decisión: taxonomía ligera desde la Constitución Política: títulos y capítulos como bloques temáticos, artículos 1-380 como etiquetas finas y respaldo "derecho a X" para tutelas.
- EuroVoc en español se conserva solo como referencia de comparación; la decisión condiciona el diseño de etiquetas de la propuesta metodológica.

**Figuras:** ninguna (panel de candidatas + decisión).

**Notas del orador:** Ninguna ontología internacional cubre el español jurídico colombiano: se evaluaron cuatro candidatas y se decidió derivar una taxonomía ligera de la propia Constitución Política.

---

## Diapositiva 13 — Representación: TF-IDF sobre lemas y embeddings

**Contenido:**
- Figuras: TF-IDF: cola larga del vocabulario y top-15 rasgos de T-001/2021; e5-base: similitud coseno entre cinco documentos (768 dimensiones).
- HALLAZGO: TF-IDF sobre lemas: 23.742 × 100.000, 95,66 % de esparsidad, ajuste en ≈ 212 s; embeddings e5: similitud media 0,899 (0,91-0,92 entre los documentos representativos).

**Figuras:** `fig_representacion_tfidf.png` y `fig_embeddings_ejemplo.png`.

**Notas del orador:** La representación principal es TF-IDF sobre lemas: interpretable, de bajo costo y con 95,66 por ciento de esparsidad. Los embeddings contextuales se evaluaron como comparación: capturan cercanía semántica con mayor costo.

---

## Diapositiva 14 — Propuesta metodológica: modelos y criterios de evaluación

**Contenido:**
- Tarea: clasificación supervisada multi-clase; tipo de providencia (T/C/SU) como categoría inicial y área temática derivada de la Constitución como objetivo principal (etiquetas a validar).
- Particiones: 70/15/15 estratificado (semilla 42, prueba intocable); vectorizador ajustado solo con entrenamiento; evaluación cronológica complementaria.
- Modelos a comparar: línea base mayoritaria; TF-IDF sobre lemas con modelos lineales; transformer en español con segmentación y agregación.
- Criterios: macro-F1 (principal, por el desbalance T 72,02 %); F1 por clase; matriz de confusión; análisis de errores; robustez por periodo; costo (tiempo, memoria, GPU).
- El protocolo fija desde ya las reglas del juego: particiones selladas, anti-leakage y métricas robustas al desbalance.

**Figuras:** ninguna (panel de la propuesta).

**Notas del orador:** La propuesta define la tarea, las particiones con prueba intocable, los modelos a comparar y los criterios: macro-F1, análisis de errores y costo. Sin resultados de modelos: es una propuesta metodológica.

---

## Diapositiva 15 — Conclusiones

**Contenido:**
- 23.742 sentencias: dataset medido verificado; cifras de la E1 sin corrección.
- 95,66 % de esparsidad TF-IDF: representación calculada sobre el corpus completo.
- 64,32 → 58,53 MTLD (1992-2021): diversidad léxica en descenso sostenido.
- CONCLUSIÓN PRINCIPAL: el género es formulaico (accion de tutela domina el vocabulario) y su léxico evoluciona; el corpus medido y la representación TF-IDF dejan lista la fase de modelado.
- SIGUIENTE ETAPA: 01 validar etiquetas temáticas con expertos · 02 comparar modelos con macro-F1 · 03 análisis de errores por periodo · 04 embeddings de dominio y recuperación · 05 resúmenes para sentencias extensas · 06 documentar y publicar el corpus medido.
- DEL CORPUS MEDIDO → A LA FASE DE MODELADO.

**Figuras:** ninguna (cierre).

**Notas del orador:** Cierre: corpus y dataset consolidados, género formulaico, diversidad decreciente con recambio temático, unidad validada y representación definida. El trabajo inmediato es validar etiquetas, comparar modelos y analizar errores por periodo.
