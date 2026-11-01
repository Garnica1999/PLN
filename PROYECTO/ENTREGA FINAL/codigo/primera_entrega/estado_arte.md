# Estado del arte — Corpus de sentencias de la Corte Constitucional de Colombia (Entrega 1 de PLN)

**Fecha de investigación:** 2026-09-05
**Métodos de verificación usados:** API de Hugging Face (dataset card), arXiv API, API de Crossref, OpenAlex, Semantic Scholar, DuckDuckGo y páginas oficiales (ACL Anthology, repositorio UniSabana). Todas las referencias de este documento fueron comprobadas (URL o DOI resuelto) el día de la fecha. Ninguna es inventada.

---

## 0. Nota importante sobre el dataset

**No se encontraron trabajos indexados (artículos de revista, actas de conferencia o arXiv) que citen o usen explícitamente el dataset de Hugging Face `Manuel/sentencias-corte-cons-colombia-1992-2021`.** La dataset card de HF no cita papers, y las búsquedas (arXiv, Semantic Scholar, OpenAlex, Crossref, DuckDuckGo) no arrojan publicaciones que lo referencien. En su lugar, este estado del arte cubre:

1. los trabajos más cercanos que sí procesan **el mismo corpus subyacente** (sentencias T/C/SU de la relatoría de la Corte Constitucional colombiana);
2. NLP jurídico en **español** (España y Latinoamérica);
3. metodologías de **preprocesamiento y textometría** sobre corpus de sentencias (largos, fórmulicos, multisección).

El dataset está verificado por su API: `Manuel` (usuario HF), 23.750 sentencias (1992–2021), CSV `Juris1992_2022.csv`, columnas `ID` y `Texto`, licencia **CC-BY-4.0**, creado el 2022-06-30. URL: <https://huggingface.co/datasets/Manuel/sentencias-corte-cons-colombia-1992-2021>

---

## 1. Referencias núcleo (ficha de 6)

### R1. Uso del corpus de la relatoría de la Corte Constitucional con minería de datos
**Anzola Cortés, J. S. (2021).** *Algoritmos, sentencias y derecho constitucional comparado: explorando el uso de jurisprudencia extranjera en Colombia.* Trabajo de grado (pregrado), Universidad de La Sabana. Dirección: J. F. Molano Pulido y C. F. Rojas González.
- **URL (verificada):** <https://hdl.handle.net/10818/47728>
- **Qué aporta a la E1:** es el trabajo más cercano al corpus usado: procesa **más de 6.000 sentencias de constitucionalidad de la Corte Constitucional de Colombia emitidas entre 1992 y 2020** (el mismo universo de documentos del dataset de HF) con un enfoque de minería de datos (patrones de lenguaje + asistencia humana). Muestra que el corpus de relatoría es explotable computacionalmente y documenta problemas típicos de su estructura textual (fórmulas de estilo, referencias a jurisprudencia embebidas en el texto). Útil para justificar que el proyecto se inscribe en una línea de análisis computacional de la jurisprudencia colombiana ya explorada localmente.

### R2. Análisis léxico-discursivo de sentencias de la Corte Constitucional colombiana
**Barbosa Becerra, J. J. M. (2021).** Comportamiento discursivo del enunciado "territorio ancestral" en las sentencias de la Corte Constitucional de Colombia. *Estudios de Derecho*, 78(172). Universidad de Antioquia.
- **DOI (verificado):** <https://doi.org/10.17533/udea.esde.v78n172a10>
- **Qué aporta a la E1:** análisis (cualitativo-crítico del discurso, con apoyo de rastreo léxico) sobre **sentencias de la propia Corte Constitucional colombiana** que fallaron a favor de la consulta previa. Documenta la existencia de un **léxico jurídico especializado y de fórmulas discursivas estables** en este tipo de sentencias, lo que justifica (i) esperar que el top-N de frecuencias esté dominado por fórmulas de estilo institucionales, y (ii) la necesidad de una lista de stopwords ampliada con términos jurídico-procedimentales (no solo gramaticales) para que el análisis de contenido muestre el léxico temático. Es la referencia en español que ancla el análisis de vocabulario en el corpus colombiano concreto.

### R3. Textometría y clasificación de un corpus de sentencias largas (referencia metodológica clave)
**Soh, J., Lim, H. K., & Chai, I. E. (2019).** Legal Area Classification: A Comparative Study of Text Classifiers on Singapore Supreme Court Judgments. *Proceedings of the Natural Legal Language Processing Workshop 2019* (NAACL-HLT), Minneapolis, pp. 67–77.
- **DOI (verificado):** <https://doi.org/10.18653/v1/W19-2208> — arXiv:1904.06470
- **Qué aporta a la E1:** estudia exactamente el escenario de este proyecto: **corpus de sentencias "pocos pero larguísimos"** (6.227 sentencias; el colombiano: 23.742 sentencias de ~11.000 palabras promedio). Reporta estadísticas de corpus (longitudes, vocabulario) y compara modelos clásicos (TF-IDF, tópicos) contra neuronales. Conclusión útil para la E1: en documentos legales largos las estadísticas de frecuencia deben reportarse con cuidado por el desbalance de longitudes entre documentos (los textos de plantilla dominan los conteos globales); sugiere reportar también frecuencias normalizadas por documento o por sección.

### R4. Preprocesamiento para bolsas de palabras: evidencia empírica de que la limpieza importa
**HaCohen-Kerner, Y., Miller, D., & Yigal, Y. (2020).** The influence of preprocessing on text classification using a bag-of-words representation. *PLOS ONE*, 15(5), e0232525.
- **DOI (verificado):** <https://doi.org/10.1371/journal.pone.0232525>
- **Qué aporta a la E1:** estudio empírico (14 métodos de preprocesamiento: minúsculas, números, signos, stopwords, stemming, lematización, etc.) que muestra que **la elección del preprocesamiento cambia los resultados y que su efecto depende del corpus y la tarea**. Justifica el enfoque de la E1: decisiones de limpieza explícitas, documentadas y, cuando se pueda, comparadas (análisis con/sin stopwords, con/sin normalización de tildes) en lugar de adoptar una receta por defecto. Es la referencia para responder "¿por qué limpiamos así y no de otra forma?".

### R5. Clasificación y preprocesamiento de sentencias legales en español (España)
**González-González, J., de Arriba-Pérez, F., García-Méndez, S., Busto-Castiñeira, A., & González-Castaño, F. J. (2023).** Automatic explanation of the classification of Spanish legal judgments in jurisdiction-dependent law categories with tree estimators. *Journal of King Saud University – Computer and Information Sciences*, 35(7), 101634.
- **DOI (verificado):** <https://doi.org/10.1016/j.jksuci.2023.101634> — arXiv:2404.00437
- **Qué aporta a la E1:** pipeline completo sobre **sentencias judiciales en español** (España), con preprocesamiento de textos legales y uso de **diccionarios de dominio construidos con expertos legales ("expert-in-the-loop")** para la extracción de rasgos. Documenta que el lenguaje de las sentencias en español tiene vocabulario especializado por jurisdicción y que la terminología (p. ej., las categorías de derecho) se puede capturar con frecuencias de términos. Apoya la idea de añadir al análisis una lista de vocabulario jurídico (o de comparar el top-N con y sin ese vocabulario).

### R6. Análisis temporal/diacrónico del discurso judicial con "corpus analytics"
**Raj, A. Y., V, G. R. N., Mishra, A. N., & Basha, M. S. A. (2026).** Collective Bargaining in Karnataka Public Sector Undertakings: A Computational Analysis of Indian Court Judgments Using NLP and Machine Learning. *2026 International Conference on Recent Advancement in Electrical, Computer and Communication Technologies (IECCT)*, IEEE, pp. 1–6.
- **DOI (verificado):** <https://doi.org/10.1109/IECCT68664.2026.11541803>
- **Qué aporta a la E1:** ejemplo metodológico publicado de **análisis diacrónico del discurso de sentencias**: procesa 682 sentencias de 1950–2026 con una fase explícita de *corpus analytics* (longitud, vocabulario, entidades por época) y demuestra con pruebas estadísticas que una reforma (1991) produjo **cambios medibles en la longitud y el vocabulario dominante de las sentencias**. Es el precedente más directo para justificar las nubes/barras de palabras por año en la E1: comparar el léxico de periodos permite detectar cambios de agenda y de estilo de redacción en una corte. (Análogo no colombiano; no existe equivalente publicado para la Corte Constitucional colombiana.)

---

## 2. Referencias complementarias verificadas (incluidas en references.bib)

| Ref. | Cita | URL/DOI verificado | Por qué está en el .bib |
|---|---|---|---|
| C1 | Aletras, N., Tsarapatsanis, D., Preoţiuc-Pietro, D., & Lampos, V. (2016). Predicting judicial decisions of the European Court of Human Rights: a Natural Language Processing perspective. *PeerJ Computer Science*, 2, e93. | DOI: 10.7717/peerj-cs.93 | NLP clásico (n-gramas y tópicos) sobre decisiones judiciales; muestra que distintas **secciones** de la sentencia (hechos vs. argumentación) tienen distinto poder predictivo → justifica analizar por secciones/estructura. |
| C2 | Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., & Androutsopoulos, I. (2020). LEGAL-BERT: The Muppets straight out of Law School. *Findings of EMNLP 2020*, pp. 2898–2904. | DOI: 10.18653/v1/2020.findings-emnlp.261 — arXiv:2010.02559 | Modelos de lenguaje para dominio legal: el vocabulario jurídico difiere del general; contexto para decidir (en fases 2+ del proyecto) si hace falta adaptación de dominio; en E1 justifica **diferir** cualquier etapa de modelos. |
| C3 | Löffler, C., Martínez Freile, A., & Rey Pizarro, T. (2025). Predicting potentially abusive clauses in Chilean terms of services with natural language processing. *Artificial Intelligence and Law* (Springer). | DOI: 10.1007/s10506-025-09462-w — arXiv:2502.00865 | Evidencia reciente de que los **recursos y datasets legales anotados en español (Latinoamérica) son escasos** y de que los modelos multilingües/adaptados rinden distinto en texto legal en español → contextualiza la contribución del proyecto y refuerza que un corpus limpio y caracterizado textométricamente ya es un aporte. |

---

## 3. Recomendaciones metodológicas para la entrega 1

### (a) Limpieza y normalización recomendada según la literatura
- Pipeline mínimo documentado y **aplicado por etapas medibles**: (1) extracción del año y del número de sentencia desde el encabezado (expresiones tipo "Sentencia T-001/21"); (2) eliminación de las filas que no son sentencias (autos, HTML corrupto — en este corpus: 8 de 23.750); (3) normalización de espacios, saltos de línea y unicode; (4) decisión explícita sobre **tildes**: o se conservan, o se normaliza todo a ASCII; nunca mezclar (R4, R5); (5) manejo de las **citas jurisprudenciales y normativas** embebidas (patrones T-123/21, C-456/12, SU-xxx, "artículo X de la Ley Y", expedientes T-7.8xx.xxx) y de los números de página/folio: al ser ruido de referencia, conviene normalizarlos o quitarlos antes de contar frecuencias si se quiere léxico temático (R1 documenta que estas referencias saturan el texto de las sentencias colombianas).
- R4 recomienda reportar la limpieza como decisión con efecto medible: en la E1 basta con presentar las cifras (docs, tokens) antes/después de filtrar.

### (b) Stopwords en español con/sin tildes
- **Problema concreto detectado en el proyecto (validado en disco):** las listas estándar (p. ej., NLTK español) contienen muchas formas **con tilde** (más, está, cómo, qué, cuándo…). Si el corpus se normaliza sin diacríticos y la lista de stopwords no, esas palabras dejan de filtrarse y contaminan el top-N.
- **Regla según la literatura:** aplicar la **misma transformación al corpus y a la lista de stopwords** (R4). Si se normaliza el texto a ASCII, normalizar también la lista (o usar una lista ya sin tildes); si se conservan tildes, usar la lista con tildes.
- **Recomendación adicional para corpus jurídicos (R1, R2, R5):** complementar la lista gramatical con un conjunto de **stopwords jurídico-procedimentales** propias del género ("sentencia", "corte", "magistrado", "ponente", "expediente", "demandante", "accionante", "artículo", "constitución", "derecho(s)", "ley", "tutela" — solo si no es el término de interés) y de fórmulas de estilo ("considerando", "resuelve", "primero", "segundo" cuando sean numeración), para que el top-20 sin stopwords muestre contenido temático y no plantilla.

### (c) Top-N y visualizaciones típicas en corpus legales
- El estándar en trabajos sobre sentencias es **top-N de frecuencias con barras horizontales**, acompañado de nubes solo como figura "de impacto" (R3 usa estadísticas de corpus; R6 usa barras y comparaciones por época). Se recomienda reportar top-20 (con y sin stopwords) en barras y reservar la nube como figura complementaria por año o tipo.
- **Cuidado con documentos muy largos:** en un corpus con fuerte desbalance de longitud y fórmulas repetidas (R3), los conteos globales crudos están dominados por las sentencias más largas y por la plantilla; conviene reportar también la **frecuencia relativa promedio por documento** (media del % del vocabulario de cada sentencia) o separar por tipo de providencia (T/C/SU), porque cada tipo tiene plantilla y extensión distintas (verificado en fase 0: T 17.099 / C 6.246 / SU 397).
- El enunciado pide total de palabras, promedio por doc, top-20 y nubes: calcularlos **sobre el corpus completo** (no sobre una muestra), como ya se corrigió en el plan; la literatura apoya documentar media, mediana y percentiles de longitud (R3).

### (d) Análisis temporal de vocabulario legal (nubes por año)
- La literatura de *computational legal studies* usa el tiempo como variable para detectar **cambios de agenda y de estilo** en tribunales: R6 muestra con test estadísticos cómo cambia la longitud y el léxico dominante de sentencias antes/después de una reforma; el análisis léxico por año/lustro es el equivalente descriptivo para la E1.
- Para la Corte Constitucional colombiana no hay publicación equivalente (búsqueda realizada: arXiv, Semantic Scholar, OpenAlex, Crossref, revistas nacionales); el precedente más sólido es R6 + R1 (que ya explota el mismo corpus por décadas 1992–2020).
- **Recomendación práctica:** agrupar por año es ruidoso en los extremos (2021 tiene solo 20 documentos; verificado en fase 0); agrupar por lustros o por periodos de la Corte si las nubes por año resultan ilegibles o escasas. Separar T (tutela, volumen alto) de C/SU en el análisis temporal evita que la evolución del léxico de tutelas opaque el de constitucionalidad.

### (e) Lematización: ¿necesaria en E1 o diferible?
- **Diferible.** En la E1 (top-N, nubes, promedios) la lematización aporta agrupar flexiones ("tutela/tutelas/tutelar"), pero el costo es alto (corpus de ~272 millones de tokens; spaCy sobre 23.742 documentos largos) y la ganancia para las figuras pedidas es marginal frente a una buena normalización + stopwords (R4 muestra que el efecto del stemming/lematización es dependiente del corpus y no siempre positivo).
- Documentar la decisión en el informe (transparencia metodológica) y, si se quiere estimar el efecto, comparar el top-20 con tokens crudos vs. lematizados en un subcorpus pequeño (por ejemplo 200 sentencias) y reportarlo como nota.
- Para las fases 2+ (clasificación con modelos): LEGAL-BERT y los pipelines modernos no requieren lematización (tokenizadores BPE trabajan sobre la forma superficial) (C2, R5), de modo que la decisión de E1 no compromete el resto del proyecto.

---

## 4. Nota de verificación (resumen)

- Dataset HF: página oficial + API (createdAt 2022-06-30, CC-BY-4.0, 23.750 filas) — **verificado**.
- R1: repositorio Intellectum UniSabana (handle 10818/47728) — **verificado**.
- R2: Crossref DOI 10.17533/udea.esde.v78n172a10 — **verificado**.
- R3: Crossref DOI 10.18653/v1/W19-2208 (pp. 67–77) — **verificado**.
- R4: Crossref/OpenAlex DOI 10.1371/journal.pone.0232525 — **verificado**.
- R5: arXiv + Crossref DOI 10.1016/j.jksuci.2023.101634 (vol. 35(7), art. 101634) — **verificado**.
- R6: Crossref DOI 10.1109/IECCT68664.2026.11541803 (IEEE, pp. 1–6) — **verificado**.
- C1: Crossref DOI 10.7717/peerj-cs.93 (PeerJ CS 2:e93) — **verificado**.
- C2: arXiv 2010.02559 + Crossref DOI 10.18653/v1/2020.findings-emnlp.261 (pp. 2898–2904) — **verificado**.
- C3: arXiv 2502.00865 + Crossref DOI 10.1007/s10506-025-09462-w (AI & Law, 2025) — **verificado**.
