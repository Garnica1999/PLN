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

---

## E2 — Estado del arte ampliado (enfoque Juris-PLN)

**Fecha de investigación:** 2026-09-08
**Métodos de verificación usados:** webfetch sobre arXiv API y páginas abs de arXiv, ACL Anthology, API de Crossref, portal UCM/Dialnet (intento a Semantic Scholar API devolvió 429 por límite de tasa). Las 10 referencias nuevas fueron comprobadas una por una (DOI o URL funcional) el día de la fecha. Ninguna es inventada y ninguna duplica las 10 de E1.
**Mapa de módulos usado abajo:** M0 validación de sentencia, M1 clasificación de área temática, M2 retrieval de artículos de la CP, M3 NER jurídico, M4 preprocesamiento y análisis morfológico, M5 textometría diacrónica, M6 red de precedentes y jurimetría, M7 embeddings y modelado neuronal. **Errata (2026-10-03):** este mapa es el de las referencias SOTA y difiere del mapa operativo real del proyecto (`00_intent.md`/`PLAN.md`: M0 validación, M1 lematización, M2 NER, M3 etiquetado de área, M4 clasificación, M5 red de citas, M6 diacronía, M7 vínculo CP). Se conserva por trazabilidad de las referencias; el mapa real está en `PROYECTO/arquitectura.md`.

### (a) Legal NLP en español (corpus judiciales hispanohablantes, clasificación de sentencias)

**E2-A1. Chalkidis, I., Fergadiotis, M., & Androutsopoulos, I. (2021).** MultiEURLEX — A multi-lingual and multi-label legal document classification dataset for zero-shot cross-lingual transfer. *Proceedings of EMNLP 2021*, pp. 6974–6996. DOI: <https://doi.org/10.18653/v1/2021.emnlp-main.559> — arXiv:2109.00904. **Verificado** (ACL Anthology + arXiv abs).
- **Aporte:** corpus de 65.000 leyes de la UE en 23 lenguas (incluido el español) con etiquetas multietiqueta EUROVOC; demuestra deriva conceptual temporal (exige particiones cronológicas, no aleatorias) y que el fine-tuning monolingüe destruye el conocimiento multilingüe (olvido catastrófico), mitigable con fine-tuning parcial, adaptadores o BitFit.
- **Implicación para este proyecto (M1, M7):** justifica clasificación multietiqueta por áreas con XLM-RoBERTa/BETO, partición cronológica train/val/test (1992–2021) y adaptadores si el corpus es pequeño.

### (b) Statute/constitution retrieval (COLIEE, EFRA, statute law retrieval, linking casos-legislación)

**E2-B1. Althammer, S., Askari, A., Verberne, S., & Hanbury, A. (2021).** DoSSIER@COLIEE 2021: Leveraging dense retrieval and summarization-based re-ranking for case law retrieval. *COLIEE 2021*. arXiv:2108.03937 — URL: <https://arxiv.org/abs/2108.03937>. **Verificado** (arXiv abs + API).
- **Aporte:** en COLIEE 2021 el retrieval a nivel de pasaje supera al de documento completo; la densa supera a la léxica y la mejor configuración combina BM25 con retrieval denso de pasajes (embeddings de dominio) más re-ranking BERT sobre resúmenes, porque los casos largos exceden la ventana de BERT.
- **Implicación para este proyecto (M2, M7):** justifica arquitectura en dos etapas para vincular sentencias con artículos de la CP: BM25 a nivel de pasaje como primera etapa y re-ranking neuronal denso encima.

**E2-B2. Fink, T., Recski, G., Kusa, W., & Hanbury, A. (2023).** Statute-enhanced lexical retrieval of court cases for COLIEE 2022. *JURISIN 2022*. arXiv:2304.08188 — URL: <https://arxiv.org/abs/2304.08188>. **Verificado** (arXiv abs + API).
- **Aporte:** añadir explícitamente la información de estatutos extraída a queries y documentos mejora el retrieval léxico; la fusión de rankings (rank fusion) a nivel de pasaje supera al retrieval de documento completo (alto recall, baja precisión).
- **Implicación para este proyecto (M2, M4):** justifica extraer menciones de artículos de la CP ("artículo X", "T-123/21") como rasgos de expansión de query y fusión BM25 + denso con alta exhaustividad.

### (c) NER jurídico (español y técnicas generales)

**E2-C1. Samy, D. (2021).** Reconocimiento y clasificación de entidades nombradas en textos legales en español. *Procesamiento del Lenguaje Natural*, 67, 103–114. ISSN 1135-5948. URL: <http://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/view/6381> — handle: <https://hdl.handle.net/10045/117482>. **Verificado** (portal UCM + Dialnet; revista SEPLN, en español).
- **Aporte:** primer baseline de NERC para español jurídico con cinco tipos (personas, organizaciones, lugares, fechas absolutas y referencias a leyes/artículos); metodología híbrida (regex + listas + tres modelos spaCy v3) con F1 global 0,93 y hasta 0,98 en menciones a leyes.
- **Implicación para este proyecto (M3, M4):** justifica NER con spaCy v3 e híbrido reglas + estadístico, con la etiqueta "referencia normativa" como extractor de citas a la CP y a precedentes.

**E2-C2. Dozier, C., Kondadadi, R., Light, M., Vachher, A., Veeramachaneni, S., & Wudali, R. (2010).** Named Entity Recognition and Resolution in Legal Text. En Francesconi et al. (eds.), *Semantic Processing of Legal Texts*, LNAI 6036, pp. 27–43. Springer. DOI: <https://doi.org/10.1007/978-3-642-12837-0_2>. **Verificado** (API de Crossref, 61 citas).
- **Aporte:** combina reglas de alta precisión con clasificadores estadísticos de secuencia e incluye resolución (coreferencia) de entidades, problema central en texto legal donde las mismas partes y normas se mencionan con variantes.
- **Implicación para este proyecto (M3):** justifica añadir resolución de menciones (mismo artículo/persona con distinta redacción) sobre el NER base de Samy (2021).

### (d) Análisis morfológico y lexicográfico de corpus judiciales (POS, diversidad léxica, MTLD, textometría)

**E2-D1. McCarthy, P. M., & Jarvis, S. (2010).** MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment. *Behavior Research Methods*, 42(2), 381–392. DOI: <https://doi.org/10.3758/BRM.42.2.381>. **Verificado** (API de Crossref, 648 citas).
- **Aporte:** valida MTLD como medida de diversidad léxica robusta frente a la longitud del texto (a diferencia del TTR crudo), junto a vocd-D y HD-D, con estudio de validación psicométrica.
- **Implicación para este proyecto (M4, M5):** justifica MTLD (más TTR/MATTR) para comparar riqueza léxica entre tipos T/C/SU y por año sin sesgo de longitud, dado que las sentencias varían de cientos a decenas de miles de palabras.

### (e) Textometría diacrónica y evolución léxica de corpus legales

**E2-E1. Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016).** Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change. *Proceedings of ACL 2016*. arXiv:1605.09096 — URL: <https://arxiv.org/abs/1605.09096>. **Verificado** (arXiv abs + API).
- **Aporte:** metodología robusta de embeddings diacrónicos (PPMI, SVD, word2vec alineados) validada contra cambios históricos conocidos en seis corpus de cuatro lenguas y dos siglos; propone la ley de conformidad (frecuencia) y la ley de innovación (polisemia).
- **Implicación para este proyecto (M5, M7):** justifica entrenar embeddings por lustro (1992–2021) y medir deriva semántica de términos constitucionales ("tutela", "territorio ancestral") con vecinos y distancia coseno entre periodos.

### (f) Jurimetría y redes de citas en cortes (PageRank, precedentes)

**E2-F1. Fowler, J. H., Johnson, T. R., Spriggs, J. F., Jeon, S., & Wahlbeck, P. J. (2007).** Network Analysis and the Law: Measuring the Legal Importance of Precedents at the U.S. Supreme Court. *Political Analysis*, 15(3), 324–346. DOI: <https://doi.org/10.1093/pan/mpm011>. **Verificado** (API de Crossref, 221 citas).
- **Aporte:** construye la red completa de 26.681 opiniones (1791–2005) y define puntajes de importancia (centralidad de eigenvector, inward/outward relevance) superiores al conteo crudo de citas y validados contra el comportamiento futuro de citación de cortes estatales y federales.
- **Implicación para este proyecto (M6):** justifica el grafo dirigido de citas T/C/SU con PageRank y centralidad de eigenvector, más análisis diacrónico de centralidad para detectar sentencias hito por época.

### (g) Embeddings legales en español (BETO, sentence embeddings multilingües para retrieval)

**E2-G1. Cañete, J., Chaperon, G., Fuentes, R., Ho, J.-H., Kang, H., & Pérez, J. (2020).** Spanish Pre-Trained BERT Model and Evaluation Data. *PML4DC @ ICLR 2020*. arXiv:2308.02976 — URL: <https://arxiv.org/abs/2308.02976> — modelo: <https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased>. **Verificado** (arXiv abs + model card HF).
- **Aporte:** primer BERT preentrenado solo en español (110M parámetros, versiones cased/uncased) que supera a mBERT en la mayoría de tareas del benchmark español tipo GLUE; el paper reconoce que XLM-RoBERTa (560M) lo supera en algunos benchmarks con más datos y parámetros.
- **Implicación para este proyecto (M7, M1):** justifica BETO cased como baseline neuronal en español para validación y clasificación, con XLM-RoBERTa como alternativa si hay cómputo.

**E2-G2. Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., & Wei, F. (2022).** Text Embeddings by Weakly-Supervised Contrastive Pre-training. arXiv:2212.03533 — URL: <https://arxiv.org/abs/2212.03533>. **Verificado** (arXiv abs + API).
- **Aporte:** familia E5 de embeddings (entrenamiento contrastivo con supervisión débil en CCPairs); primer modelo que supera a BM25 en zero-shot en el benchmark BEIR de retrieval y SOTA en MTEB tras fine-tuning, usable para retrieval, clustering y clasificación.
- **Implicación para este proyecto (M2, M7):** justifica `multilingual-e5` como embedding denso del re-ranking de artículos de la CP, robusto a la deriva de redacción entre sentencias (1992–2021) y texto constitucional.

---

## Nota de verificación E2 (resumen)

- E2-A1: ACL Anthology DOI 10.18653/v1/2021.emnlp-main.559 (pp. 6974–6996) + arXiv:2109.00904 — **verificado**.
- E2-B1: arXiv:2108.03937 (COLIEE 2021) — **verificado**.
- E2-B2: arXiv:2304.08188 (JURISIN 2022) — **verificado**.
- E2-C1: SEPLN vol. 67, pp. 103–114 + handle 10045/117482 (portal UCM/Dialnet) — **verificado**.
- E2-C2: Crossref DOI 10.1007/978-3-642-12837-0_2 (LNAI 6036, pp. 27–43) — **verificado**.
- E2-D1: Crossref DOI 10.3758/BRM.42.2.381 (Behav. Res. Methods 42(2), 381–392) — **verificado**.
- E2-E1: arXiv:1605.09096 (ACL 2016) — **verificado**.
- E2-F1: Crossref DOI 10.1093/pan/mpm011 (Political Analysis 15(3), 324–346) — **verificado**.
- E2-G1: arXiv:2308.02976 (PML4DC @ ICLR 2020) + model card HF dccuchile — **verificado**.
- E2-G2: arXiv:2212.03533 — **verificado**.
- Semantic Scholar API: intento documentado con error 429 (límite de tasa sin API key); la búsqueda se completó con arXiv API, ACL Anthology, Crossref y web.

---

## E2 ciclo v2 — Estado del arte adicional (verificado 2026-09-10)

Investigación realizada para el ciclo de mejora v2 (benchmark + reranking +
transformer largo). Fuente: arXiv API + Hugging Face Hub, verificada el
2026-09-10. Entradas .bib: E2-H1 a E2-H6 (total 26).

### (h) Documentos legales largos y reranking para statute retrieval

**E2-H1. Chalkidis, I., Fergadiotis, M., Malakasiotis, P., & Androutsopoulos, I. (2022).** Processing Long Legal Documents with Pre-trained Transformers: Modding LegalBERT and Longformer for Court Documents. arXiv:2211.00974 — URL: <https://arxiv.org/abs/2211.00974>. **Verificado** (arXiv API, 2026-09-10).
- **Aporte:** en documentos legales largos, el truncamiento estándar (512 tokens) descarta la mayor parte del contenido; el chunking con sliding-window y agregación (mean/max-pooling de los chunks) supera consistentemente al truncamiento, y Longformer (4096) ayuda en tareas de nivel de documento.
- **Implicación para este proyecto (Fase V del ciclo v2):** explica la derrota de BETO v1 (docs ~11k palabras vs ventana 512) y da la receta exacta: modo chunk-mean con sliding window sobre el documento completo, agregando representaciones por chunk. Es la base del ADR-3 v1 aplicado por fin al transformer.

**E2-H2. UA-StatuteRetrieval (2026).** Temporal Decay of Co-Citation Predictability: 20-Year Statute Retrieval Benchmark from 396M Ukrainian Court Citations. arXiv:2607.11109. **Verificado** (arXiv API, 2026-09-10).
- **Aporte:** benchmark longitudinal de 20 años sobre 396M citas de códigos legales: la señal de co-citación decae (MRR −33 % en artículos fijos, −47 % con split temporal), el decaimiento es no uniforme por área, y los artículos "hub" (>100k citas) resisten el decaimiento.
- **Implicación para este proyecto (M7/Fase W):** valida el análisis de deriva temporal del v1 y sugiere usar los artículos hub de la CP (241, 86, 13 — top del NER) como señal de prior en el ranking de recuperación.

**E2-H3. LegalMALR (2026).** Multi-Agent Query Understanding and LLM-Based Reranking for Chinese Statute Retrieval. arXiv:2601.17692. **Verificado** (arXiv API, 2026-09-10).
- **Aporte:** pipeline ganador en statute retrieval 2026: comprensión de la consulta multi-agente + reranking con LLM sobre candidatos de primera etapa.
- **Implicación (Fase T):** confirma que la arquitectura de dos etapas (recall amplio BM25 -> reranking) es el estándar actual; nuestra Fase T la implementa con cross-encoder en GPU.

**E2-H4. Cross-Architecture LLM Ensembles, Feature-Based Reranking and Retrieval-Augmented Prompting for Legal Information Processing (2026).** arXiv:2607.11400. **Verificado** (arXiv API, 2026-09-10).
- **Aporte:** el reranking basado en features y los ensambles de modelos dominan las tareas legales de COLIEE 2026 frente a enfoques de una sola etapa.
- **Implicación (Fase T/W):** respalda combinar señales (BM25 + densa + features de co-citación) en el reranking en lugar de un único scorer.

**E2-H5. Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021).** BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. arXiv:2104.08663. **Verificado** (arXiv API, 2026-09-10).
- **Aporte:** estandariza nDCG@10 como métrica principal de IR heterogéneo; demuestra que BM25 es un baseline fuerte que muchos modelos densos no superan zero-shot.
- **Implicación (Fase R):** justifica añadir nDCG@10 al benchmark v3 para comparabilidad con la literatura.

**E2-H6. BAAI (2024). bge-reranker-v2-m3.** Hugging Face: <https://huggingface.co/BAAI/bge-reranker-v2-m3>. **Verificado** (HF API, 2026-09-10: 18.2M descargas, apache-2.0, 568M parámetros, multilingüe con español).
- **Aporte:** cross-encoder reranker multilingüe basado en bge-m3; puntúa pares (consulta, pasaje) y se usa para reordenar los top-k de una primera etapa.
- **Implicación (Fase T):** reranker elegido para la segunda etapa: cabe sobrado en la GPU (17.1 GB), licencia apache-2.0, soporta español nativamente.

### (i) Modelos legales en español disponibles (verificación HF 2026-09-10)

- **PlanTL-GOB-ES/RoBERTalex** (366 descargas): RoBERTa legal-es gubernamental; candidato a comparativa en Fase V, pero sin benchmark público de statute retrieval.
- **Narrativa/legal-longformer-base-4096-spanish** (29 descargas): ventana 4096, atractivo por longitud pero sin respaldo de comunidad (riesgo de calidad); queda como comparación secundaria OPCIONAL, nunca primario.
- **mrm8488/legalectra-base-spanish** (7 descargas): descartado por nula adopción.
- **Conclusión:** NO existe un "LegalBERT español" consolidado; BETO (ya probado, ADR-10) sigue siendo la base razonable, con chunk-mean (E2-H1) como método de mejora.

### (j) Ontologías jurídicas en español — evaluación para el ciclo v2

- Búsqueda arXiv 2026-09-10: la literatura de ontologías legales se centra en
  representación de conocimiento y argumentación (Kelsen, LKIF), NO en mejora
  medible de statute retrieval. No existe una ontología legal colombiana o
  constitucional en español lista para usar.
- La evaluación documental de E1 (`../PRIMERA ENTREGA/plan/nota_ontologias.md`)
  ya había concluido: EuroVoc/MultiEURLEX parcial (derecho UE), LKIF baja,
  ELI/ECLI nula como contenido, LegalRuleML/Akoma Ntoso baja a corto plazo.
- **Decisión v2:** mantener la taxonomía derivada de la CP (19 bloques) como
  eje; EuroVoc-es SOLO como features auxiliares de clasificación si gana en
  benchmark (Fase W, opcional). Se registrará como ADR-14 en su momento.
