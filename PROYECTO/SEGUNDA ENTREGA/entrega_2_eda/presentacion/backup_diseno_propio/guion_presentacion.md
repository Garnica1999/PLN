# Guion de la presentación — Entrega 2

**Proyecto:** Análisis de sentencias de la Corte Constitucional de Colombia (1992–2021)
**Entrega 2:** EDA avanzado, representación de textos y propuesta metodológica
**Duración sugerida:** 10–12 minutos (≈45–60 s por diapositiva)
**Formato:** 12 diapositivas (16:9), español, sin emojis. Todas las cifras provienen del informe final (`informe/main.tex`) y del notebook ejecutado (`resultados/metricas_entrega_ii.json`).

---

## Diapositiva 1 — Título

**Contenido:**
- Análisis de sentencias de la Corte Constitucional de Colombia (1992–2021)
- Entrega 2: EDA avanzado, representación de textos y propuesta metodológica
- 23.742 sentencias · 1992–2021 · corpus público Hugging Face (CC-BY-4.0)
- Integrantes: Erika Valeria Rivera Jiménez · Julián Guillermo Carrillo Meneses · Carlos Andrés Garnica Salazar · Oscar Andrés Prieto Moreno
- Pontificia Universidad Javeriana · Maestría en Inteligencia Artificial · Octubre de 2026

**Figura sugerida:** ninguna (portada).

**Notas del orador:** Presentamos la Entrega 2 del proyecto de PLN sobre jurisprudencia constitucional colombiana. Esta entrega profundiza el análisis del corpus, entrega el conjunto de datos medido, define la representación de los textos y propone la metodología de modelado.

---

## Diapositiva 2 — Problema y objetivo

**Contenido:**
- Problema: corpus extenso y formulaico (≈11.000 palabras por documento), sin anotaciones temáticas; la información relevante se dispersa entre fórmulas procesales repetidas y la lectura manual de 23.742 documentos es inviable.
- Escasez documentada de recursos legales anotados en español para América Latina: un corpus depurado, medido y documentado es un aporte en sí mismo.
- Objetivo del proyecto: sistema de apoyo que (i) valida que sea providencia, (ii) determina su área temática constitucional, (iii) identifica los artículos de la CP que la sustentan y (iv) la sitúa en su contexto jurisprudencial.
- Esta entrega (requisitos 1–5 y 7): correcciones vs E1 · EDA avanzado · dataset medido · representación · propuesta metodológica.
- Alcance temporal: 1992–2021 (2021 parcial: 20 documentos).

**Figura sugerida:** ninguna.

**Notas del orador:** El problema es de PLN aplicado: extraer conocimiento de un acervo enorme y sin metadatos. Esta entrega no ejecuta los modelos finales: prepara los datos, la representación y el diseño experimental.

---

## Diapositiva 3 — Corpus y correcciones vs Entrega 1

**Contenido:**
- Corpus de análisis: 23.742 sentencias (T 17.099 · C 6.246 · SU 397), 1992–2021; fuente Hugging Face (CC-BY-4.0); 8 exclusiones documentadas (6 autos, 2 fragmentos HTML); tasa de extracción de metadatos 99,97 %.
- Las cifras canónicas de la Entrega 1 no requirieron corrección.
- Modificaciones reales: (1) dataset medido en parquet (23.742 filas × 12 columnas); (2) EDA extendido reportado (n-gramas, colocaciones, lemas/stems/POS, MTLD, granularidad); (3) siete visualizaciones nuevas; (4) unidad de análisis validada con evidencia; (5) análisis por periodos ejecutado; (6) representación y metodología definidas.

**Figura sugerida:** ninguna (panel de cifras).

**Notas del orador:** El corpus se mantiene idéntico en cifras; los cambios son extensiones verificadas contra el informe y los artefactos de la Entrega 1, con sus razones documentadas en la sección 2 del informe.

---

## Diapositiva 4 — Limpieza del texto (pipeline)

**Contenido:**
- Pipeline de 8 pasos aplicado al corpus completo: minúsculas; protección de la ñ; normalización NFD sin diacríticos; filtro de caracteres [a-zñ]; colapso de espacios; tokenización por separación en espacios; filtro de ruido (t, c, p, art); stopwords normalizadas.
- La misma transformación se aplica al corpus y a la lista de palabras vacías: quedan 306 formas efectivas (84 formas acentuadas corregidas).
- Decisión documentada: el preprocesamiento afecta los resultados y depende del corpus y la tarea (HaCohen-Kerner et al., 2020).

**Figura sugerida:** ninguna (dos columnas de pasos).

**Notas del orador:** Cada paso es medible y reproducible. El filtro de ruido evita que restos de claves de relatoría y citas de página contaminen los conteos; las stopwords se normalizan igual que el corpus para que el filtrado sea consistente.

---

## Diapositiva 5 — Medidas globales y distribución temporal

**Contenido:**
- 260.988.316 palabras crudas · 250.012.101 limpias · 121.205.437 stopwords (48,48 %) · 128.806.664 sin stopwords.
- Promedios por documento: 10.992,68 crudas / 10.530,37 limpias; mediana de lemas de 7.116 (p25: 3.959; p75: 12.654); mediana de palabras limpias: 7.158.
- Vocabulario: 224.468 formas · 195.959 lemas · 121.696 stems.
- Distribución temporal: estable 1999–2014 (rango de 941 en 2013 a 992 en 2002); descenso 2015–2020; 2021 parcial (20 documentos).

**Figura sugerida:** `fig_sentencias_por_anio.png` + `fig_top20_sin_stopwords.png`.

**Notas del orador:** Los conteos son reales sobre el corpus completo. El volumen anual es estable durante la mayor parte del periodo y el desplome de 2021 es evidencia de cobertura parcial; sin stopwords emerge el léxico jurídico dominante (articulo, derecho, ley, corte, sentencia, tutela).

---

## Diapositiva 6 — N-gramas y colocaciones

**Contenido:**
- Bigramas más frecuentes: corte constitucional (286.786) · derechos fundamentales (241.467) · seguridad social (131.119).
- Trigramas: accion de tutela (344.893) · articulo del decreto (89.352); el 82,7 % de los trigramas guardados responde al patrón con palabra intermedia vacía (fórmulas del género).
- Validación manual de 20 colocaciones: frecuencia 55 % de fórmulas válidas · PMI 0 % · razón de verosimilitud (LLR) 75 %.
- Decisión: la LLR es la medida de asociación principal (equilibra frecuencia y asociación); el PMI ordena ruido y fragmentos no españoles.

**Figura sugerida:** `fig_ngramas_bi_tri.png` + `fig_colocaciones.png`.

**Notas del orador:** El corpus es fuertemente formulaico y la validación manual lo confirma cuantitativamente. La LLR concentra fórmulas de dominio y se adopta para las fases de modelado y recuperación.

---

## Diapositiva 7 — Diversidad léxica (MTLD) y morfología

**Contenido:**
- MTLD global 61,03; por tipo: T 61,63 > C 59,54 > SU 58,88.
- Por periodos: 64,32 (1992–2001) → 60,37 (2002–2011) → 58,53 (2012–2021); pendiente −0,32 por año (r = −0,328, p < 0,001).
- Morfología: 224.468 formas → 195.959 lemas (87,3 %) → 121.696 stems (54,2 %); POS: NOUN 22,58 %, ADP 20,45 %, DET 15,57 %.
- La lematización reduce el vocabulario manteniendo la legibilidad: base de la representación TF-IDF.

**Figura sugerida:** `fig_mtld_distribucion.png` + `fig_lemma_vs_stem.png`.

**Notas del orador:** La diversidad léxica desciende de forma sostenida en las tres décadas, consistente con una estandarización del estilo de la Corte. La lematización ofrece el mejor equilibrio entre compresión de vocabulario e interpretabilidad.

---

## Diapositiva 8 — Recursos léxicos y ontologías

**Contenido:**
- Candidatas evaluadas: EuroVoc / MultiEURLEX (aplicabilidad parcial: derecho de la UE, no mapea la CP ni la tutela) · LKIF (baja; útil como patrón de modelado) · ELI / ECLI (nula como vocabulario temático) · LegalRuleML / Akoma Ntoso (baja a corto plazo; anotación manual costosa).
- Decisión: no adoptar una ontología externa tal cual; derivar una taxonomía ligera de la Constitución Política: títulos y capítulos como bloques temáticos, artículos 1–380 como etiquetas finas y respaldo "derecho a X" para tutelas.
- EuroVoc en español se conserva solo como referencia de comparación.

**Figura sugerida:** ninguna (panel de candidatas + decisión).

**Notas del orador:** Ninguna ontología internacional cubre el español jurídico colombiano. La taxonomía derivada de la propia Constitución condiciona el diseño de etiquetas de la propuesta metodológica.

---

## Diapositiva 9 — Dataset entregado

**Contenido:**
- `corpus_medido.parquet`: 23.742 filas × 12 columnas (identificación: id, tipo, numero, ano; textos: texto, texto_limpio; métricas: n_chars_texto, n_palabras_raw, n_palabras_clean, n_stopwords, n_lemas, mtld).
- Verificación: 23.742 ids únicos; sin nulos en claves; sumas reconciliadas: crudas 260.988.316 · limpias 250.012.101 · stopwords 121.205.437 · lemas 248.533.344 · caracteres 1.640.749.780; archivo de 1.544,7 MB.
- Adjuntos: `corpus_medido_muestra.csv` (100 filas por tipo y década), `diccionario_datos.md` (14 variables), `lemas_por_doc.parquet`.

**Figura sugerida:** ninguna (panel de esquema + verificación).

**Notas del orador:** El requisito 3 del enunciado queda cubierto con un archivo columnar legible, verificado contra las cifras canónicas del corpus y acompañado de muestra y diccionario de datos.

---

## Diapositiva 10 — Representación: TF-IDF + embeddings

**Contenido:**
- TF-IDF sobre lemas (representación principal): 1–2 gramos, min_df=3, 100.000 rasgos, frecuencia sublineal; matriz 23.742 × 100.000 con 103.045.759 valores no nulos (esparsidad 95,66 %); ajuste en 192,5 s; cola larga: ≈2 % de la masa de frecuencia documental en los 100 primeros rasgos, ≈17 % en los 1.000 y ≈60 % en los 10.000.
- Ejemplo T-001/2021: 7.712 rasgos activos; mayor peso: capital salud (0,065), valero (0,056), de calle (0,052).
- Similitud coseno TF-IDF entre documentos T/C/SU: 0,24–0,25 (discrimina por solapamiento léxico).
- Embeddings contextuales (e5 multilingüe, 768 dimensiones, 5 documentos, 512 tokens): similitud media 0,899; mismos documentos 0,91–0,92 → complementarios.

**Figura sugerida:** `fig_representacion_tfidf.png` + `fig_embeddings_ejemplo.png`.

**Notas del orador:** La representación principal es interpretable y de bajo costo; los embeddings se reservan como línea comparativa para similitud semántica, con mayor costo y menor interpretabilidad.

---

## Diapositiva 11 — Propuesta metodológica

**Contenido:**
- Tarea: clasificación supervisada multi-clase; tipo de providencia (T/C/SU) como categoría inicial y área temática derivada de la CP como objetivo principal (etiquetas a validar).
- Particiones: 70/15/15 estratificadas, semilla 42, prueba intocable; anti-leakage: vectorizador ajustado solo con entrenamiento, partición a nivel de documento y evaluación cronológica complementaria.
- Modelos a comparar: línea base mayoritaria → TF-IDF + modelos lineales → transformer contextual en español con segmentación y agregación.
- Criterios: macro-F1 (principal), F1 por clase, matriz de confusión, análisis de errores, robustez por periodo y costo (tiempo/memoria/GPU).

**Figura sugerida:** ninguna (tabla compacta de modelos).

**Notas del orador:** La metodología define desde ya las reglas del juego: particiones estratificadas, prueba sellada y métricas robustas al desbalance, con comparación entre la línea base clásica y un transformer en español.

---

## Diapositiva 12 — Conclusiones y trabajo futuro

**Contenido:**
- Corpus consolidado y dataset medido verificado; cifras de la Entrega 1 sin corrección.
- Género formulaico: accion de tutela como trigrama dominante; LLR 75 % como mejor medida de colocaciones.
- Diversidad léxica decreciente (64,32 → 58,53) y recambio temático del vocabulario (incoder/rupd → uariv/jep/coronavirus).
- Unidad de análisis validada (sentencia 97,17 % vs subsentencia 4,64 %); TF-IDF sobre lemas calculado (95,66 % de esparsidad) y metodología propuesta.
- Trabajo futuro: validar etiquetas temáticas con expertos; comparar modelos con macro-F1; análisis de errores por periodo y tipo; embeddings de dominio; resúmenes y extracción para sentencias extensas.

**Figura sugerida:** ninguna (cierre).

**Notas del orador:** Cerramos con los cinco hallazgos principales y el plan inmediato: validación de etiquetas, comparación de modelos y análisis de errores. El conjunto de datos medido y la propuesta metodológica dejan al proyecto listo para su fase de modelado.
